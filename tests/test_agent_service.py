"""Agentic loop, guardrails, cloud service layer, overlay, and the Lambda HTTP adapter."""
import base64
import json

import numpy as np
import pytest

import lambda_function as lf
from edgesentry.agent import (
    RecordingSink, SafetyTools, clean_text, evaluate_run, perceive, run_agent, run_bedrock, to_jsonable,
)
from edgesentry.overlay import annotate_frame
from edgesentry.hazard_detector import Detection
from edgesentry.service import AwsSink, emf_metrics, run_safety_check
from edgesentry.synthetic import CELL_ZONE, build_scenario


@pytest.fixture(scope="module")
def perception():
    sc = build_scenario()
    return perceive(sc.frames, sc.timestamps, sc.background, sc.zone)


# ---------------------------------------------------------------- perception


def test_perception_tracks_every_worker_with_expected_persistence(perception):
    tracks = perception.tracks
    assert sorted(tracks) == ["T1", "T2", "T3", "T4"]
    assert tracks["T2"].verified and tracks["T2"].verified_at_s == 1.5  # in the zone from t=0, 1.5 s debounce
    assert tracks["T1"].verified and tracks["T4"].verified
    assert not tracks["T3"].verified and tracks["T3"].in_zone_s < 1.5  # passer-by: transient
    assert tracks["T2"].ppe_status == "missing" and tracks["T1"].ppe_status == "compliant"
    assert tracks["T4"].ppe_status != "compliant"  # vest in shadow fools the coarse check


def test_overlay_is_causal_and_never_mutates_input():
    sc = build_scenario()
    frame = sc.frames[5]
    before = frame.copy()
    det = Detection(id="T9", bbox=(700, 460, 770, 690))
    out = annotate_frame(frame, CELL_ZONE, "Cell", [det], {"T9"}, {"T9": 0.5}, set())
    assert np.array_equal(frame, before) and out.shape == frame.shape and not np.array_equal(out, frame)
    alarm = annotate_frame(frame, CELL_ZONE, "Cell", [det], {"T9"}, {"T9": 2.0}, {"T9"})
    assert not np.array_equal(out, alarm)  # verified intrusion draws the alarm state


# ---------------------------------------------------------------- guardrails


def test_guardrails_block_unsafe_or_unproven_actions(perception):
    tools = SafetyTools(perception)
    assert tools.trigger_estop_alert("T2", "no vest")["status"] == "blocked_by_guardrail"  # not inspected yet
    assert tools.trigger_estop_alert("T3", "x")["status"] == "blocked_by_guardrail"  # not verified
    assert tools.request_supervisor_approval("T3", "x")["status"] == "blocked_by_guardrail"
    assert tools.log_and_monitor("T2", "x")["status"] == "blocked_by_guardrail"  # verified must escalate
    assert tools.log_and_monitor("T99", "x")["status"] == "blocked_by_guardrail"  # unknown track
    tools.inspect_worker("T4")  # vest found in the close-up
    assert tools.trigger_estop_alert("T4", "x")["status"] == "blocked_by_guardrail"
    tools.inspect_worker("T2")
    assert tools.trigger_estop_alert("T2", "confirmed")["status"] == "executed"
    assert tools.request_supervisor_approval("T2", "again")["status"] == "blocked_by_guardrail"  # already resolved
    assert len(tools.blocked) == 7
    assert tools.call("nope", {})["error"].startswith("unknown tool")
    assert "bad arguments" in tools.call("inspect_worker", {"wrong": 1})["error"]
    assert "unknown track" in tools.inspect_worker("T42")["error"]


def test_rule_policy_matches_ground_truth(perception):
    run, tools = run_agent(perception, prefer="rules")
    ev = evaluate_run(perception, run, build_scenario().workers)
    assert run.mode == "rules" and run.fallback_reason is None
    assert ev["decision_accuracy"] == 1.0 and ev["false_estops"] == 0 and ev["missed_estops"] == 0
    assert {t: d["action"] for t, d in run.decisions.items()} == {
        "T1": "request_supervisor_approval", "T2": "trigger_estop_alert",
        "T3": "log_and_monitor", "T4": "request_supervisor_approval"}
    # The OpenCV re-inspection changed the outcome for T4 (coarse "missing" → vest found → no stop).
    assert tools.inspections["T4"]["vest_present"] and not tools.inspections["T2"]["vest_present"]
    assert run.summary.startswith("- **T1**")


# ---------------------------------------------------------------- Bedrock loop (scripted client)


def _use(name, **inp):
    return {"toolUse": {"toolUseId": f"id-{name}-{inp.get('track_id', '')}", "name": name, "input": inp}}


class ScriptedBedrock:
    """Stands in for bedrock-runtime: replays assistant turns and records requests."""

    def __init__(self, turns):
        self.turns, self.requests = list(turns), []

    def converse(self, **kwargs):
        self.requests.append(kwargs)
        return {"output": {"message": {"role": "assistant", "content": self.turns.pop(0)}}}


def test_bedrock_loop_with_guardrail_block_and_controller_nudge(perception):
    client = ScriptedBedrock([
        [{"text": "<thinking>Look first.</thinking>"}, _use("get_scene_report")],
        [_use("trigger_estop_alert", track_id="T2", reason="no vest")],  # blocked: not inspected
        [_use("inspect_worker", track_id="T2"), _use("inspect_worker", track_id="T4")],
        [_use("trigger_estop_alert", track_id="T2", reason="confirmed"),
         _use("request_supervisor_approval", track_id="T4", reason="vest in shadow"),
         _use("log_and_monitor", track_id="T3", reason="brief")],
        [{"text": "Done."}],  # T1 forgotten → controller nudges
        [_use("request_supervisor_approval", track_id="T1", reason="compliant")],
        [{"text": "Summary:\n- T2 stopped"}],
    ])
    run, _tools = run_agent(perception, client=client, prefer="bedrock")
    assert run.mode == "bedrock" and run.fallback_reason is None
    assert [s["tool"] for s in run.trace].count("controller") == 1
    assert client.requests[5]["toolConfig"]["toolChoice"] == {"any": {}}  # forced tool use after a nudge
    assert len(run.blocked) == 1 and run.blocked[0]["track_id"] == "T2"
    assert run.summary == "Summary:\n- T2 stopped" and "<thinking>" not in run.trace[0]["why"]
    assert evaluate_run(perception, run, build_scenario().workers)["decision_accuracy"] == 1.0


def test_incomplete_llm_run_falls_back_to_rules(perception):
    client = ScriptedBedrock([[_use("get_scene_report")], [{"text": "ok"}], [{"text": "ok"}], [{"text": "ok"}]])
    run, tools = run_agent(perception, client=client, prefer="bedrock")
    assert run.mode == "rules" and "unresolved" in run.fallback_reason
    assert run.completed_by_policy == []
    assert len(tools.decisions) == 4  # fresh tools: the fallback resolved everything itself


def test_partial_agent_keeps_its_decisions_and_policy_finishes_the_rest(perception):
    client = ScriptedBedrock([
        [_use("get_scene_report")],
        [_use("request_supervisor_approval", track_id="T1", reason="ppe"),
         _use("inspect_worker", track_id="T2"),
         _use("log_and_monitor", track_id="T3", reason="brief")],
        [_use("trigger_estop_alert", track_id="T2", reason="confirmed no vest")],
        [{"text": "done"}],
        [{"text": "still done"}],
        [{"text": "stopping"}],
    ])
    sink = RecordingSink()
    run, tools = run_agent(perception, sink=sink, client=client, prefer="bedrock")
    assert run.mode == "bedrock"
    assert run.completed_by_policy == ["T4"]
    assert "T4" in run.fallback_reason
    assert set(tools.decisions) == {"T1", "T2", "T3", "T4"}
    assert tools.decisions["T4"]["action"] == "request_supervisor_approval"
    actions = [f"{e['action']}:{e['track_id']}" for e in sink.events]
    assert actions.count("request_supervisor_approval:T1") == 1
    assert actions.count("trigger_estop_alert:T2") == 1
    assert actions.count("log_and_monitor:T3") == 1
    assert actions.count("request_supervisor_approval:T4") == 1
    assert any(s.get("by") == "policy" and s["tool"] == "inspect_worker" for s in run.trace)
    assert evaluate_run(perception, run, build_scenario().workers)["decision_accuracy"] == 1.0


def test_bedrock_step_budget_is_bounded(perception):
    client = ScriptedBedrock([[_use("get_scene_report")]] * 3)
    run = run_bedrock(SafetyTools(perception), client, max_steps=3)
    assert run.steps == 3 and run.decisions == {}


def test_clean_text_strips_model_tags_and_keeps_lines():
    assert clean_text("<thinking>x</thinking>\n\n- a  b\n-  c\n\n\n\nend") == "x\n\n- a b\n- c\n\nend"
    assert clean_text(None) == ""


# ---------------------------------------------------------------- service layer


class FakeS3:
    def __init__(self, existing=()):
        self.puts, self.existing = [], set(existing)

    def put_object(self, **kw):
        if kw.get("IfNoneMatch") == "*" and kw["Key"] in self.existing:
            raise RuntimeError("An error occurred (PreconditionFailed) when calling the PutObject operation")
        self.puts.append(kw)
        self.existing.add(kw["Key"])

    def generate_presigned_url(self, op, Params, ExpiresIn):
        return f"https://example.invalid/{Params['Key']}?exp={ExpiresIn}"

    def head_object(self, Bucket, Key):
        if Key not in self.existing:
            raise RuntimeError("404 Not Found")
        return {}


class FakeSNS:
    def __init__(self):
        self.published = []

    def publish(self, **kw):
        self.published.append(kw)
        return {"MessageId": f"m-{len(self.published)}"}


def test_aws_sink_stores_evidence_and_alerts_only_on_escalations():
    s3, sns = FakeS3(), FakeSNS()
    sink = AwsSink("bucket", "arn:topic", "run1", s3, sns)
    stop = sink.emit("trigger_estop_alert", "T2", "why", b"jpeg")
    log = sink.emit("log_and_monitor", "T3", "why", b"jpeg")
    assert stop["s3_key"] == "evidence/run1/T2-trigger_estop_alert.jpg" and stop["sns_message_id"] == "m-1"
    assert "sns_message_id" not in log and len(sns.published) == 1 and len(s3.puts) == 2
    assert json.loads(sns.published[0]["Message"])["track_id"] == "T2"


def test_run_safety_check_response_and_metrics(capsys):
    resp = run_safety_check(prefer="rules", emit_metrics=True)
    emf = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert emf["_aws"]["CloudWatchMetrics"][0]["Namespace"] == "EdgeSentry" and emf["DecisionAccuracy"] == 1.0
    assert resp["decision_mode"] == "rules" and len(resp["frames"]) == 24 and len(resp["decisions"]) == 4
    assert all(d["evidence_image"] and d["ground_truth"] for d in resp["decisions"])
    assert resp["live"]["aws_actions"] is False and resp["agent_notes"] is None
    assert resp["summary"].count("- **T") == 4  # grounded: one line per executed decision
    json.dumps(resp)  # fully serialisable


def test_emf_and_jsonable_helpers():
    line = json.loads(emf_metrics("NS", {"Service": "S"}, {"A": (1.5, "Count")}))
    assert line["A"] == 1.5 and line["_aws"]["CloudWatchMetrics"][0]["Dimensions"] == [["Service"]]
    assert to_jsonable({"a": np.float32(1.5), "b": (np.int64(2),), "c": np.bool_(True)}) == {"a": 1.5, "b": [2], "c": True}


# ---------------------------------------------------------------- Lambda HTTP adapter


def _event(method, path, body=None, b64=False):
    raw = json.dumps(body) if isinstance(body, (dict, list)) else body
    if b64 and raw is not None:
        raw = base64.b64encode(raw.encode()).decode()
    return {"rawPath": path, "requestContext": {"http": {"method": method}}, "body": raw, "isBase64Encoded": b64}


@pytest.fixture
def offline(monkeypatch):
    for k in ("EVIDENCE_BUCKET", "ALERT_TOPIC_ARN", "USAGE_TABLE", "BEDROCK_ENABLED"):
        monkeypatch.delenv(k, raising=False)


def test_handler_routes_and_validation(offline):
    page = lf.handler(_event("GET", "/"))
    assert page["statusCode"] == 200 and "text/html" in page["headers"]["content-type"]
    assert page["headers"]["x-content-type-options"] == "nosniff"
    health = json.loads(lf.handler(_event("GET", "/api/health"))["body"])
    assert health["opencv"].startswith("5.") and health["configured"]["evidence_bucket"] is False
    assert lf.handler(_event("GET", "/nope"))["statusCode"] == 404
    assert lf.handler(_event("POST", "/api/run", "{bad"))["statusCode"] == 400
    assert lf.handler(_event("POST", "/api/run", [1, 2]))["statusCode"] == 400
    assert lf.handler(_event("POST", "/api/run", {"mode": "yolo"}))["statusCode"] == 400
    assert lf.handler(_event("POST", "/api/run", "x" * 5000))["statusCode"] == 413


def test_handler_body_cannot_forge_a_response(offline):
    forged = {"statusCode": 200, "headers": {}, "body": "<script>alert(1)</script>", "mode": "rules"}
    resp = lf.handler(_event("POST", "/api/run", forged, b64=True))
    assert resp["statusCode"] == 200 and resp["headers"]["content-type"] == "application/json"
    assert json.loads(resp["body"])["decision_mode"] == "rules"


def test_offline_ai_mode_is_labelled_as_fallback(offline):
    body = json.loads(lf.handler(_event("POST", "/api/run", {"mode": "bedrock"}))["body"])
    assert body["decision_mode"] == "rules" and body["requested_mode"] == "bedrock"
    assert "not configured" in body["fallback_notice"]


class FakeDdb:
    def __init__(self, caps_hit=()):
        self.caps_hit, self.calls = set(caps_hit), []

    def update_item(self, **kw):
        kind = kw["Key"]["pk"]["S"].split("#")[0]
        self.calls.append(kind)
        if kind in self.caps_hit:
            raise type("ConditionalCheckFailedException", (Exception,), {})("cap")


@pytest.fixture
def aws_env(monkeypatch):
    monkeypatch.setenv("EVIDENCE_BUCKET", "bucket")
    monkeypatch.setenv("ALERT_TOPIC_ARN", "arn:aws:sns:us-east-1:1:topic")
    monkeypatch.setenv("USAGE_TABLE", "usage")


def _clients(ddb=None, s3=None, sns=None, bedrock=None):
    return lambda: {"s3": s3 or FakeS3(), "sns": sns or FakeSNS(), "ddb": ddb or FakeDdb(), "bedrock": bedrock}


def test_daily_run_cap_returns_429(aws_env):
    resp = lf.handler(_event("POST", "/api/run", {"mode": "rules"}), aws_factory=_clients(FakeDdb({"runs"})))
    assert resp["statusCode"] == 429


def test_bedrock_cap_falls_back_with_notice_and_delivers_to_aws(aws_env):
    s3, sns, ddb = FakeS3(), FakeSNS(), FakeDdb({"bedrock"})
    body = json.loads(lf.handler(_event("POST", "/api/run", {"mode": "bedrock"}),
                                 aws_factory=_clients(ddb, s3, sns, bedrock=object()))["body"])
    assert ddb.calls == ["runs", "bedrock"]
    assert body["decision_mode"] == "rules" and "budget" in body["fallback_notice"]
    assert body["live"]["aws_actions"] and len(s3.puts) == 4 and len(sns.published) == 3


def test_take_quota_reraises_unexpected_errors(aws_env):
    class Broken:
        def update_item(self, **kw):
            raise ValueError("boom")
    with pytest.raises(ValueError):
        lf.take_quota(Broken(), "runs", 5)
    assert lf.take_quota(None, "runs", 5) is True


def test_supervisor_decision_flow(aws_env):
    run_id, track = "20261024-120000-abc123", "T1"
    s3 = FakeS3(existing={f"evidence/{run_id}/{track}-request_supervisor_approval.jpg"})
    sns = FakeSNS()
    factory = _clients(s3=s3, sns=sns)
    ok = lf.handler(_event("POST", "/api/decision", {"run_id": run_id, "track_id": track, "decision": "approve"}), aws_factory=factory)
    assert ok["statusCode"] == 200 and json.loads(ok["body"])["s3_key"] == f"decisions/{run_id}/{track}.json"
    assert sns.published[0]["MessageAttributes"]["action"]["StringValue"] == "supervisor_approve"
    again = lf.handler(_event("POST", "/api/decision", {"run_id": run_id, "track_id": track, "decision": "stop"}), aws_factory=factory)
    assert again["statusCode"] == 409
    unknown = lf.handler(_event("POST", "/api/decision", {"run_id": run_id, "track_id": "T2", "decision": "stop"}), aws_factory=factory)
    assert unknown["statusCode"] == 404


@pytest.mark.parametrize("body", [
    {"run_id": "../../etc", "track_id": "T1", "decision": "approve"},
    {"run_id": "20261024-120000-abc123", "track_id": "T1/../x", "decision": "approve"},
    {"run_id": "20261024-120000-abc123", "track_id": "T1", "decision": "maybe"},
])
def test_supervisor_decision_validation(offline, body):
    assert lf.handler(_event("POST", "/api/decision", body))["statusCode"] == 400


def test_supervisor_decision_offline(offline):
    body = {"run_id": "20261024-120000-abc123", "track_id": "T4", "decision": "stop"}
    resp = json.loads(lf.handler(_event("POST", "/api/decision", body))["body"])
    assert resp["status"] == "recorded (offline)"


def test_handler_hides_internal_errors(offline, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("secret detail")
    monkeypatch.setattr(lf, "run_safety_check", boom)
    resp = lf.handler(_event("POST", "/api/run", {"mode": "rules"}))
    assert resp["statusCode"] == 500 and "secret" not in resp["body"]
    assert "secret detail" in capsys.readouterr().out  # but it is logged for CloudWatch


def test_recording_sink_is_offline():
    ev = RecordingSink().emit("log_and_monitor", "T3", "r", b"abc")
    assert ev["delivery"].startswith("recorded") and ev["evidence_bytes"] == 3
