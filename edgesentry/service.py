"""Cloud service layer: runs a safety check and packages the result for the web UI.

Used by the AWS Lambda handler (``lambda_function.py``) and by tests. In AWS:

* evidence snapshots for every executed action go to **S3**;
* E-stop alerts and supervisor-approval requests are published to **SNS**;
* per-run metrics are emitted as **CloudWatch Embedded Metric Format** log lines.

Offline (no AWS), the same code path records actions locally — labelled as such.
"""
from __future__ import annotations

import base64
import json
import os
import time
import uuid
from typing import Any

import cv2
import numpy as np

from .agent import (AgentRun, Perception, RecordingSink, SafetyTools, clean_text, evaluate_run, perceive, run_agent,
                    summarize_decisions, to_jsonable)
from .synthetic import Scenario, build_scenario

ACTION_LABELS = {
    "trigger_estop_alert": "Protective stop + alert",
    "request_supervisor_approval": "Supervisor approval requested",
    "log_and_monitor": "Logged — monitoring",
}
NOTIFY_ACTIONS = {"trigger_estop_alert", "request_supervisor_approval"}


class AwsSink:
    """Delivers agent actions to AWS: S3 evidence for every action, SNS for alerts."""

    def __init__(self, bucket: str, topic_arn: str, run_id: str, s3: Any, sns: Any, presign_seconds: int = 3600) -> None:
        self.bucket, self.topic_arn, self.run_id = bucket, topic_arn, run_id
        self.s3, self.sns, self.presign_seconds = s3, sns, presign_seconds
        self.events: list[dict] = []

    def emit(self, action: str, track_id: str, reason: str, evidence_jpeg: bytes | None) -> dict:
        event: dict = {"action": action, "track_id": track_id, "delivery": "aws"}
        try:
            if evidence_jpeg:
                key = f"evidence/{self.run_id}/{track_id}-{action}.jpg"
                self.s3.put_object(Bucket=self.bucket, Key=key, Body=evidence_jpeg, ContentType="image/jpeg",
                                   Metadata={"run-id": self.run_id, "track-id": track_id, "action": action})
                event["s3_key"] = key
                event["evidence_url"] = self.s3.generate_presigned_url(
                    "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=self.presign_seconds)
            if action in NOTIFY_ACTIONS and self.topic_arn:
                resp = self.sns.publish(
                    TopicArn=self.topic_arn,
                    Subject=f"EdgeSentry: {ACTION_LABELS[action]} ({track_id})"[:100],
                    Message=json.dumps({"run_id": self.run_id, "track_id": track_id, "action": action,
                                        "reason": reason, "evidence_s3_key": event.get("s3_key")}),
                    MessageAttributes={"action": {"DataType": "String", "StringValue": action}},
                )
                event["sns_message_id"] = resp.get("MessageId")
        except Exception as exc:  # pragma: no cover - surfaced in the UI, never crashes the run
            event["delivery"] = "failed"
            event["error"] = f"{type(exc).__name__}: {exc}"[:200]
        self.events.append(event)
        return event


def _jpeg_b64(img: np.ndarray, width: int | None = None, quality: int = 72) -> str:
    if width and img.shape[1] > width:
        h = int(img.shape[0] * width / img.shape[1])
        img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def emf_metrics(namespace: str, dims: dict, metrics: dict[str, tuple[float, str]]) -> str:
    """One CloudWatch Embedded Metric Format line (Lambda turns it into metrics automatically)."""
    payload: dict = {
        "_aws": {"Timestamp": int(time.time() * 1000), "CloudWatchMetrics": [{
            "Namespace": namespace, "Dimensions": [list(dims)],
            "Metrics": [{"Name": k, "Unit": u} for k, (_v, u) in metrics.items()],
        }]},
    }
    payload.update(dims)
    payload.update({k: v for k, (v, _u) in metrics.items()})
    return json.dumps(payload)


def build_response(scenario: Scenario, perception: Perception, run: AgentRun, tools: SafetyTools,
                   run_id: str, timings: dict, live: dict) -> dict:
    ev = evaluate_run(perception, run, scenario.workers)
    gt_by_track = {}
    for row in ev["rows"]:
        for t in row["tracks"]:
            gt_by_track[t] = {"worker": row["worker"], "label": row["label"], "expected": row["expected"]}
    deliveries = {e["track_id"]: e for e in getattr(tools.sink, "events", [])}
    decisions = []
    for tid, d in sorted(run.decisions.items()):
        tr = perception.tracks[tid]
        insp = tools.inspections.get(tid)
        ev_bytes = tools.evidence_jpeg(tr)
        ev_img = cv2.imdecode(np.frombuffer(ev_bytes, np.uint8), cv2.IMREAD_COLOR) if ev_bytes else None
        decisions.append({
            "track_id": tid,
            "action": d["action"],
            "action_label": ACTION_LABELS.get(d["action"], d["action"]),
            "decided_by": "policy" if tid in run.completed_by_policy else run.mode,
            "reason": clean_text(d.get("reason", "")),
            "evidence": tr.public(),
            "evidence_frame": tools._best_frame(tr),
            "inspection": insp,
            "delivery": deliveries.get(tid),
            "evidence_image": _jpeg_b64(ev_img, width=640) if ev_img is not None else "",
            "ground_truth": gt_by_track.get(tid),
        })
    return to_jsonable({
        "run_id": run_id,
        "decision_maker": "Bedrock safety agent" if run.mode == "bedrock" else "Deterministic safety policy",
        "decision_mode": run.mode,
        "fallback_reason": run.fallback_reason,
        "completed_by_policy": list(run.completed_by_policy),
        "zone": perception.zone_name,
        "camera": perception.camera,
        "clip": {"frames": len(perception.frames), "seconds": round(scenario.timestamps[-1], 2), "fps_sampled": round(1 / scenario.dt, 1)},
        # The headline summary is generated from executed actions, never from model prose;
        # the agent's own narrative is kept separately (it can misstate what it did).
        "summary": summarize_decisions(tools),
        "agent_notes": clean_text(run.summary) if run.mode == "bedrock" else None,
        "decisions": decisions,
        "tracks": [t.public() for t in sorted(perception.tracks.values(), key=lambda t: t.track_id)],
        "trace": [{**s, "why": clean_text(s.get("why", ""))} for s in run.trace],
        "guardrail_blocks": run.blocked,
        "evaluation": ev,
        "timings": timings,
        "live": live,
        "frames": [_jpeg_b64(img, width=640, quality=68) for img in perception.annotated],
        "timestamps": scenario.timestamps,
    })


def run_safety_check(prefer: str = "bedrock", seed: int = 7, bedrock_client: Any = None, sink_factory=None,
                     model_id: str | None = None, emit_metrics: bool = False) -> dict:
    """End-to-end: scenario → OpenCV perception → agent decisions → actions → response."""
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    t0 = time.perf_counter()
    scenario = build_scenario(seed=seed)
    t1 = time.perf_counter()
    perception = perceive(scenario.frames, scenario.timestamps, scenario.background, scenario.zone)
    t2 = time.perf_counter()
    sink = sink_factory(run_id) if sink_factory else RecordingSink()
    kwargs = {"model_id": model_id} if model_id else {}
    run, tools = run_agent(perception, sink=sink, client=bedrock_client, prefer=prefer, **kwargs)
    t3 = time.perf_counter()
    timings = {
        "render_ms": int((t1 - t0) * 1000),
        "perception_ms": int((t2 - t1) * 1000),
        "perception_fps": round(len(scenario.frames) / (t2 - t1), 1) if t2 > t1 else None,
        "agent_ms": int((t3 - t2) * 1000),
        "total_ms": int((t3 - t0) * 1000),
    }
    live = {
        "aws_actions": isinstance(sink, AwsSink),
        "bedrock": run.mode == "bedrock",
        "region": os.environ.get("AWS_REGION"),
    }
    resp = build_response(scenario, perception, run, tools, run_id, timings, live)
    if emit_metrics:
        ev = resp["evaluation"]
        print(emf_metrics("EdgeSentry", {"Service": "SafetyCheck", "DecisionMode": run.mode}, {
            "PerceptionMs": (timings["perception_ms"], "Milliseconds"),
            "AgentMs": (timings["agent_ms"], "Milliseconds"),
            "AgentSteps": (run.steps, "Count"),
            "GuardrailBlocks": (len(run.blocked), "Count"),
            "EStopAlerts": (sum(1 for d in run.decisions.values() if d["action"] == "trigger_estop_alert"), "Count"),
            "DecisionAccuracy": (ev["decision_accuracy"], "None"),
            "FallbackToPolicy": (1 if run.fallback_reason else 0, "Count"),
        }))
    return resp
