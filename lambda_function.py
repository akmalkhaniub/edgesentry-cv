"""AWS Lambda entry point for the EdgeSentry web endpoint (Lambda Function URL).

Routes
------
GET  /               → the web UI
GET  /api/health     → runtime + configuration check (OpenCV version, AWS wiring)
POST /api/run        → run a safety check: {"mode": "bedrock" | "rules"}
POST /api/decision   → a supervisor answers an approval request:
                       {"run_id": ..., "track_id": "T1", "decision": "approve" | "stop"}

Everything heavy (OpenCV 5 perception, the agent loop, S3/SNS delivery, CloudWatch
EMF metrics) lives in ``edgesentry.service``; this module only adapts HTTP and
enforces the public endpoint's usage limits (a DynamoDB counter per UTC day).
"""
from __future__ import annotations

import base64
import functools
import json
import os
import pathlib
import re
import time
import traceback

import cv2

from edgesentry.service import AwsSink, run_safety_check

_WEB = pathlib.Path(__file__).parent / "edgesentry" / "web" / "index.html"
_SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "x-frame-options": "DENY",
    "cache-control": "no-store",
}
_MAX_BODY = 4096  # this API takes a tiny JSON body; reject anything larger
_RUN_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
_TRACK_ID = re.compile(r"^T\d{1,3}$")
DECISIONS = {"approve": "Supervisor approved entry", "stop": "Supervisor ordered a protective stop"}


def _resp(status: int, body, content_type: str = "application/json") -> dict:
    if not isinstance(body, str):
        body = json.dumps(body)
    return {"statusCode": status, "headers": {"content-type": content_type, **_SECURITY_HEADERS}, "body": body}


@functools.lru_cache(maxsize=1)
def _aws() -> dict:  # pragma: no cover - only meaningful inside AWS
    import boto3
    from botocore.config import Config

    region = os.environ.get("AWS_REGION", "us-east-1")
    s3_cfg = Config(signature_version="s3v4", s3={"addressing_style": "virtual"})  # presigned evidence links
    # Bounded model calls: one slow response must not exhaust the Lambda timeout.
    bedrock_cfg = Config(connect_timeout=5, read_timeout=20, retries={"max_attempts": 2, "mode": "standard"})
    return {"s3": boto3.client("s3", region_name=region, config=s3_cfg), "sns": boto3.client("sns", region_name=region),
            "ddb": boto3.client("dynamodb", region_name=region),
            "bedrock": boto3.client("bedrock-runtime", region_name=os.environ.get("BEDROCK_REGION", region), config=bedrock_cfg)}


def _configured() -> dict:
    return {
        "evidence_bucket": bool(os.environ.get("EVIDENCE_BUCKET")),
        "alert_topic": bool(os.environ.get("ALERT_TOPIC_ARN")),
        "usage_limits": bool(os.environ.get("USAGE_TABLE")),
        "bedrock_enabled": os.environ.get("BEDROCK_ENABLED", "true").lower() == "true",
    }


def take_quota(ddb, kind: str, cap: int, now: float | None = None) -> bool:
    """Atomically count one use of ``kind`` today; False once today's cap is reached."""
    table = os.environ.get("USAGE_TABLE")
    if not table or ddb is None:
        return True
    now = time.time() if now is None else now
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    try:
        ddb.update_item(
            TableName=table, Key={"pk": {"S": f"{kind}#{day}"}},
            UpdateExpression="ADD runs :one SET expires_at = :ttl",
            ConditionExpression="attribute_not_exists(runs) OR runs < :cap",
            ExpressionAttributeValues={":one": {"N": "1"}, ":cap": {"N": str(cap)}, ":ttl": {"N": str(int(now) + 3 * 86400)}},
        )
        return True
    except Exception as exc:
        if type(exc).__name__ == "ConditionalCheckFailedException" or "ConditionalCheckFailed" in str(exc):
            return False
        raise


def handle_run(body: dict, aws_factory=None) -> dict:
    mode = body.get("mode", "bedrock")
    if mode not in ("bedrock", "rules"):
        return _resp(400, {"error": "mode must be 'bedrock' or 'rules'"})
    cfg = _configured()
    clients = (aws_factory or _aws)() if cfg["evidence_bucket"] and cfg["alert_topic"] else {}
    ddb = clients.get("ddb")
    if not take_quota(ddb, "runs", int(os.environ.get("DAILY_RUN_CAP", "1000"))):
        return _resp(429, {"error": "The public demo has reached today's run limit. Please try again tomorrow (UTC)."})
    bedrock_client = sink_factory = None
    capped = False
    if clients:
        bucket, topic = os.environ["EVIDENCE_BUCKET"], os.environ["ALERT_TOPIC_ARN"]
        sink_factory = lambda run_id: AwsSink(bucket, topic, run_id, clients["s3"], clients["sns"])  # noqa: E731
        if mode == "bedrock" and cfg["bedrock_enabled"]:
            if take_quota(ddb, "bedrock", int(os.environ.get("BEDROCK_DAILY_CAP", "150"))):
                bedrock_client = clients["bedrock"]
            else:
                capped = True
    result = run_safety_check(prefer=mode, bedrock_client=bedrock_client, sink_factory=sink_factory,
                              model_id=os.environ.get("BEDROCK_MODEL_ID") or None, emit_metrics=True)
    result["requested_mode"] = mode
    if mode == "bedrock" and result["decision_mode"] != "bedrock":
        if capped:
            result["fallback_reason"] = "daily AI-agent cap reached"
            result["fallback_notice"] = "The AI agent's daily budget for this public demo is used up."
        elif bedrock_client is None:
            result["fallback_reason"] = "AI agent not configured on this deployment"
            result["fallback_notice"] = "The AI agent is not configured on this deployment."
        else:
            result["fallback_notice"] = "The AI agent did not complete this run."
        print(json.dumps({"level": "warn", "event": "agent_fallback", "run_id": result["run_id"],
                          "reason": result.get("fallback_reason")}))
    return _resp(200, result)


def handle_decision(body: dict, aws_factory=None) -> dict:
    run_id, track_id, decision = body.get("run_id"), body.get("track_id"), body.get("decision")
    if not (isinstance(run_id, str) and _RUN_ID.match(run_id)):
        return _resp(400, {"error": "invalid run_id"})
    if not (isinstance(track_id, str) and _TRACK_ID.match(track_id)):
        return _resp(400, {"error": "invalid track_id"})
    if decision not in DECISIONS:
        return _resp(400, {"error": "decision must be 'approve' or 'stop'"})
    cfg = _configured()
    if not (cfg["evidence_bucket"] and cfg["alert_topic"]):
        return _resp(200, {"status": "recorded (offline)", "decision": decision, "label": DECISIONS[decision]})
    clients = (aws_factory or _aws)()
    s3, sns, bucket = clients["s3"], clients["sns"], os.environ["EVIDENCE_BUCKET"]
    # Only an approval request the agent actually made (its evidence is in S3) can be answered.
    try:
        s3.head_object(Bucket=bucket, Key=f"evidence/{run_id}/{track_id}-request_supervisor_approval.jpg")
    except Exception:
        return _resp(404, {"error": "no supervisor-approval request for that run and track"})
    key = f"decisions/{run_id}/{track_id}.json"
    record = {"run_id": run_id, "track_id": track_id, "decision": decision, "label": DECISIONS[decision],
              "decided_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:  # conditional write: the first answer wins, later ones are rejected
        s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(record).encode(), ContentType="application/json", IfNoneMatch="*")
    except Exception as exc:
        if "PreconditionFailed" in str(exc) or "412" in str(exc):
            return _resp(409, {"error": "this approval request was already answered"})
        raise
    msg = sns.publish(TopicArn=os.environ["ALERT_TOPIC_ARN"], Subject=f"EdgeSentry: {DECISIONS[decision]} ({track_id})"[:100],
                      Message=json.dumps(record), MessageAttributes={"action": {"DataType": "String", "StringValue": f"supervisor_{decision}"}})
    return _resp(200, {"status": "recorded", **record, "s3_key": key, "sns_message_id": msg.get("MessageId")})


def _json_body(event: dict) -> tuple[dict | None, dict | None]:
    """(body, None) on success, (None, error_response) otherwise."""
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode("utf-8", "replace")
    if len(raw) > _MAX_BODY:
        return None, _resp(413, {"error": "request body too large"})
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return None, _resp(400, {"error": "body must be JSON"})
    if not isinstance(body, dict):
        return None, _resp(400, {"error": "body must be a JSON object"})
    return body, None


def handler(event: dict, context=None, aws_factory=None) -> dict:
    http = (event.get("requestContext") or {}).get("http") or {}
    method, path = http.get("method", "GET"), event.get("rawPath", "/")
    try:
        if method == "GET" and path in ("/", "/index.html"):
            return _resp(200, _WEB.read_text(encoding="utf-8"), "text/html; charset=utf-8")
        if method == "GET" and path == "/api/health":
            return _resp(200, {"status": "ok", "opencv": cv2.__version__, "configured": _configured(),
                               "region": os.environ.get("AWS_REGION")})
        if method == "POST" and path in ("/api/run", "/api/decision"):
            body, error = _json_body(event)
            if error:
                return error
            return (handle_run if path == "/api/run" else handle_decision)(body, aws_factory)
        return _resp(404, {"error": f"no route for {method} {path}"})
    except Exception as exc:  # never leak a stack trace to the client; log it for CloudWatch
        print(json.dumps({"level": "error", "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-2000:]}))
        return _resp(500, {"error": "internal error — see CloudWatch logs"})
