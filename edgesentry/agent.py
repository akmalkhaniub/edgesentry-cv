"""Agentic perception → decision → action loop for robot-cell safety.

The agent never sees pixels. It calls OpenCV 5 tools and the *results* of those
tools decide what it does next:

* ``get_scene_report``   — runs the full pipeline over the clip (plate-diff detection,
  centroid tracking, ``pointPolygonTest`` zones, temporal debounce, coarse PPE).
* ``inspect_worker``     — active perception on one track (upscaled chromaticity
  re-inspection); the outcome can flip a pending line-stop.
* action tools           — ``trigger_estop_alert``, ``request_supervisor_approval``,
  ``log_and_monitor``.

Safety policy is enforced in code, not only in the prompt (guardrails): an E-stop
alert is rejected unless the intrusion is *verified persistent* AND a re-inspection
*confirmed* the PPE violation. The decision-maker is Amazon Bedrock (tool use via the
Converse API) when available, otherwise a deterministic rule policy that drives the
same tools — labelled honestly as such.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import cv2
import numpy as np

from .detector import CentroidTracker, PlateDiffDetector, classify_ppe, hardhat_ratio, hi_vis_ratio, inspect_ppe
from .hazard_detector import Detection
from .overlay import annotate_frame
from .pipeline import EdgeSentryPipeline
from .temporal_filter import TemporalEventFilter

DEFAULT_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.amazon.nova-pro-v1:0")
ACTION_TOOLS = ("trigger_estop_alert", "request_supervisor_approval", "log_and_monitor")


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------


class TrackedDetector:
    """Plate-diff detection + centroid tracking, presented as a pipeline Detector."""

    def __init__(self, background: np.ndarray, **detector_options: Any) -> None:
        self.detector = PlateDiffDetector(background, **detector_options)
        self.tracker = CentroidTracker()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self.tracker.update(self.detector.detect(frame))


@dataclass
class TrackSummary:
    track_id: str
    first_seen_s: float
    last_seen_s: float
    in_zone_frames: list[int] = field(default_factory=list)
    in_zone_s: float = 0.0
    verified: bool = False
    verified_at_s: float | None = None
    ppe_ratio: float = 0.0
    ppe_status: str = "missing"
    hardhat: bool = False
    bbox_by_frame: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)
    unknown_frames: int = 0  # frames where the blob did not look like an upright person (fail-safe detection)

    @property
    def upright(self) -> bool:
        return self.unknown_frames * 2 <= len(self.bbox_by_frame)

    def public(self) -> dict:
        return {
            "track_id": self.track_id,
            "entered_zone": bool(self.in_zone_frames),
            "time_in_zone_s": round(self.in_zone_s, 2),
            "persistence_verified": self.verified,
            "verified_at_s": self.verified_at_s,
            "coarse_vest_ratio": round(self.ppe_ratio, 3),
            "coarse_ppe_status": self.ppe_status,
            "hardhat_detected": self.hardhat,
            "shape": "upright person" if self.upright else "unrecognized (treat as a person)",
        }


@dataclass
class Perception:
    tracks: dict[str, TrackSummary]
    annotated: list[np.ndarray]
    frames: list[np.ndarray]
    timestamps: list[float]
    fps: float
    zone_name: str
    camera: dict = field(default_factory=dict)  # what the detector had to correct (drift, lighting): camera health


def perceive(frames: list[np.ndarray], timestamps: list[float], background: np.ndarray,
             zone: list[tuple[int, int]], zone_name: str = "Robot Cell A",
             persistence_s: float = 1.5, detector_options: dict | None = None) -> Perception:
    """Run the real EdgeSentry pipeline over a clip and summarise each track."""
    tracked = TrackedDetector(background, **(detector_options or {}))
    pipe = EdgeSentryPipeline(detector=tracked, temporal=TemporalEventFilter(persistence_threshold_s=persistence_s))
    drift, light, fg = 0.0, 0.0, 0.0
    pipe.register_zone("cell", zone, zone_name)
    tracks: dict[str, TrackSummary] = {}
    annotated: list[np.ndarray] = []
    entered_at: dict[str, float] = {}  # start of each track's current stay in the zone
    verified_ids: set[str] = set()
    t0 = time.perf_counter()
    for idx, (frame, ts) in enumerate(zip(frames, timestamps)):
        res = pipe.process_frame(frame, now=ts, annotate=False)
        h = tracked.detector.health
        drift = max(drift, float(np.hypot(*h["shift_px"])))
        light, fg = max(light, abs(h["light_offset"])), max(fg, h["foreground_fraction"])
        in_zone = {v.entity_id for v in res.violations}
        for det in res.detections:
            tr = tracks.setdefault(det.id, TrackSummary(det.id, ts, ts))
            tr.last_seen_s = ts
            tr.bbox_by_frame[idx] = det.bbox
            tr.unknown_frames += det.label == "unknown"
            if det.id in in_zone:
                tr.in_zone_frames.append(idx)
        for alert in res.verified:
            tr = tracks[alert.violation.entity_id]
            tr.verified = True
            tr.verified_at_s = ts
            verified_ids.add(tr.track_id)
        entered_at = {t: entered_at.get(t, ts) for t in in_zone}
        annotated.append(annotate_frame(frame, zone, zone_name, res.detections, in_zone,
                                        {t: ts - t0_ for t, t0_ in entered_at.items()}, verified_ids))
    elapsed = time.perf_counter() - t0
    for tr in tracks.values():
        if tr.in_zone_frames:
            ts_in = [timestamps[i] for i in tr.in_zone_frames]
            tr.in_zone_s = ts_in[-1] - ts_in[0]
            ratios = [hi_vis_ratio(frames[i], tr.bbox_by_frame[i]) for i in tr.in_zone_frames]
            hats = [hardhat_ratio(frames[i], tr.bbox_by_frame[i]) for i in tr.in_zone_frames]
        else:
            idxs = sorted(tr.bbox_by_frame)
            ratios = [hi_vis_ratio(frames[i], tr.bbox_by_frame[i]) for i in idxs]
            hats = [hardhat_ratio(frames[i], tr.bbox_by_frame[i]) for i in idxs]
        tr.ppe_ratio = float(np.median(ratios)) if ratios else 0.0
        tr.ppe_status = classify_ppe(tr.ppe_ratio)
        tr.hardhat = bool(np.median(hats) >= 0.08) if hats else False
    fps = len(frames) / elapsed if elapsed > 0 else 0.0
    camera = {"max_drift_corrected_px": round(drift, 2), "max_lighting_offset": round(light, 1),
              "max_foreground_fraction": round(fg, 4)}
    return Perception(tracks=tracks, annotated=annotated, frames=frames, timestamps=timestamps, fps=fps,
                      zone_name=zone_name, camera=camera)


# ---------------------------------------------------------------------------
# Tools + guardrails
# ---------------------------------------------------------------------------


class ActionSink(Protocol):
    """Where actions go: AWS (S3 evidence + SNS) in the cloud, a recorder offline."""

    def emit(self, action: str, track_id: str, reason: str, evidence_jpeg: bytes | None) -> dict: ...


class RecordingSink:
    """Offline sink: records actions deterministically (no network)."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, action: str, track_id: str, reason: str, evidence_jpeg: bytes | None) -> dict:
        event = {"action": action, "track_id": track_id, "reason": reason, "delivery": "recorded (offline)",
                 "evidence_bytes": len(evidence_jpeg or b"")}
        self.events.append(event)
        return event


class SafetyTools:
    """The agent's tool surface. Every action passes through code-level guardrails."""

    def __init__(self, perception: Perception, sink: ActionSink | None = None) -> None:
        self.p = perception
        self.sink = sink or RecordingSink()
        self.inspections: dict[str, dict] = {}
        self.decisions: dict[str, dict] = {}
        self.blocked: list[dict] = []

    # -- perception tools --------------------------------------------------
    def get_scene_report(self) -> dict:
        return {
            "zone": self.p.zone_name,
            "clip_seconds": round(self.p.timestamps[-1] - self.p.timestamps[0], 2) if self.p.timestamps else 0,
            "frames": len(self.p.frames),
            "camera": self.p.camera,
            "tracks": [t.public() for t in sorted(self.p.tracks.values(), key=lambda t: t.track_id)],
            "zone_tracks_to_resolve": sorted(t for t, s in self.p.tracks.items() if s.in_zone_frames),
        }

    def _best_frame(self, tr: TrackSummary) -> int:
        idxs = tr.in_zone_frames or sorted(tr.bbox_by_frame)
        return idxs[len(idxs) // 2]

    def inspect_worker(self, track_id: str) -> dict:
        tr = self.p.tracks.get(track_id)
        if tr is None:
            return {"error": f"unknown track {track_id}"}
        idx = self._best_frame(tr)
        result = inspect_ppe(self.p.frames[idx], tr.bbox_by_frame[idx])
        result.update({"track_id": track_id, "frame": idx})
        self.inspections[track_id] = result
        return result

    # -- action tools (guarded) ------------------------------------------------
    def evidence_jpeg(self, tr: TrackSummary) -> bytes | None:
        idx = self._best_frame(tr)
        img = self.p.annotated[idx].copy()
        x1, y1, x2, y2 = tr.bbox_by_frame[idx]
        cv2.rectangle(img, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 255), 3)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return buf.tobytes() if ok else None

    def _act(self, action: str, track_id: str, reason: str) -> dict:
        tr = self.p.tracks.get(track_id)
        if tr is None:
            return self._block(action, track_id, f"unknown track {track_id}")
        if track_id in self.decisions:
            return self._block(action, track_id, f"track already resolved as {self.decisions[track_id]['action']}")
        if action == "trigger_estop_alert":
            if not tr.verified:
                return self._block(action, track_id, "intrusion is not verified persistent (debounce not satisfied)")
            insp = self.inspections.get(track_id)
            if insp is None:
                return self._block(action, track_id, "PPE violation not confirmed: call inspect_worker first")
            if insp.get("vest_present"):
                return self._block(action, track_id, "re-inspection found a hi-vis vest; PPE violation not confirmed")
        if action == "request_supervisor_approval" and not tr.verified:
            return self._block(action, track_id, "only verified intrusions need supervisor approval")
        if action == "log_and_monitor" and tr.verified:
            return self._block(action, track_id, "verified intrusions must be escalated (supervisor approval or E-stop), not just logged")
        delivery = self.sink.emit(action, track_id, reason, self.evidence_jpeg(tr))
        decision = {"action": action, "track_id": track_id, "reason": reason, "status": "executed", "delivery": delivery}
        self.decisions[track_id] = decision
        return decision

    def _block(self, action: str, track_id: str, why: str) -> dict:
        entry = {"action": action, "track_id": track_id, "status": "blocked_by_guardrail", "reason": why}
        self.blocked.append(entry)
        return entry

    def trigger_estop_alert(self, track_id: str, reason: str = "") -> dict:
        return self._act("trigger_estop_alert", track_id, reason)

    def request_supervisor_approval(self, track_id: str, reason: str = "") -> dict:
        return self._act("request_supervisor_approval", track_id, reason)

    def log_and_monitor(self, track_id: str, reason: str = "") -> dict:
        return self._act("log_and_monitor", track_id, reason)

    def call(self, name: str, args: dict) -> dict:
        fn: Callable[..., dict] | None = {
            "get_scene_report": lambda **_: self.get_scene_report(),
            "inspect_worker": self.inspect_worker,
            "trigger_estop_alert": self.trigger_estop_alert,
            "request_supervisor_approval": self.request_supervisor_approval,
            "log_and_monitor": self.log_and_monitor,
        }.get(name)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except TypeError as exc:
            return {"error": f"bad arguments for {name}: {exc}"}


TOOL_SPECS = [
    {"toolSpec": {"name": "get_scene_report",
                  "description": "Run the OpenCV 5 safety pipeline over the camera clip and return every tracked person with zone intrusion, persistence, and coarse PPE evidence.",
                  "inputSchema": {"json": {"type": "object", "properties": {}}}}},
    {"toolSpec": {"name": "inspect_worker",
                  "description": "Active perception: re-inspect one tracked person's PPE more carefully (upscaled, shadow-tolerant). Required before any E-stop alert.",
                  "inputSchema": {"json": {"type": "object", "properties": {"track_id": {"type": "string"}}, "required": ["track_id"]}}}},
    {"toolSpec": {"name": "trigger_estop_alert",
                  "description": "Protective stop + alert. Only for a verified persistent intrusion whose PPE violation was confirmed by inspect_worker.",
                  "inputSchema": {"json": {"type": "object", "properties": {"track_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["track_id", "reason"]}}}},
    {"toolSpec": {"name": "request_supervisor_approval",
                  "description": "Hold the robot in reduced-speed mode and ask a supervisor to confirm an authorized entry (e.g. PPE-compliant maintenance). Verified intrusions only.",
                  "inputSchema": {"json": {"type": "object", "properties": {"track_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["track_id", "reason"]}}}},
    {"toolSpec": {"name": "log_and_monitor",
                  "description": "Record a non-actionable event (e.g. a transient brush of the cell boundary) and keep monitoring.",
                  "inputSchema": {"json": {"type": "object", "properties": {"track_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["track_id", "reason"]}}}},
]

SYSTEM_PROMPT = (
    "You are the safety decision agent for an industrial robot work cell. You cannot see images; "
    "you must call tools. Procedure: (1) call get_scene_report. (2) Resolve EVERY track that entered the zone "
    "with exactly one action. (3) Before trigger_estop_alert you MUST call inspect_worker for that track and only "
    "stop the line if inspection confirms the vest is absent — a false stop halts production. "
    "(4) Verified intrusion by a PPE-compliant person -> request_supervisor_approval. "
    "(5) Intrusion that is not persistence-verified -> log_and_monitor. Give a short reason citing the evidence. "
    "A track whose shape is 'unrecognized' is a large change that does not look like an upright person (someone "
    "kneeling, an object, a group): treat it exactly like a person and never ignore it. "
    "When every zone track is resolved, reply with a one-paragraph shift summary and stop calling tools."
)


# ---------------------------------------------------------------------------
# Decision makers
# ---------------------------------------------------------------------------


@dataclass
class AgentRun:
    mode: str  # "bedrock" | "rules"
    model_id: str | None
    trace: list[dict]
    decisions: dict[str, dict]
    blocked: list[dict]
    summary: str
    steps: int
    latency_ms: int
    fallback_reason: str | None = None


_PLAIN_ACTION = {"trigger_estop_alert": "Protective stop", "request_supervisor_approval": "Supervisor approval",
                 "log_and_monitor": "Logged"}


def summarize_decisions(tools: SafetyTools) -> str:
    """Grounded summary: executed actions + measured evidence only, no model prose (Markdown, one line per track)."""
    if not tools.decisions:
        return "Nobody entered the zone during this clip."
    lines = []
    for tid, d in sorted(tools.decisions.items()):
        tr, insp = tools.p.tracks[tid], tools.inspections.get(tid)
        facts = [f"in the zone {tr.in_zone_s:.2f} s" + (" (verified)" if tr.verified else ", under the persistence threshold")]
        if insp is not None:
            facts.append(f"close-up {'found a vest' if insp.get('vest_present') else 'found no vest'} "
                         f"({insp.get('vest_coverage', 0):.0%} of torso)")
        elif tr.verified:
            facts.append(f"wide-view vest check: {tr.ppe_status}")
        lines.append(f"- **{tid}** — {_PLAIN_ACTION.get(d['action'], d['action'])}: {'; '.join(facts)}.")
    return "\n".join(lines)


def run_rules(tools: SafetyTools) -> AgentRun:
    """Deterministic policy driving the same tools (offline fallback)."""
    t0 = time.perf_counter()
    trace: list[dict] = []

    def call(name: str, args: dict, why: str) -> dict:
        out = tools.call(name, args)
        trace.append({"step": len(trace) + 1, "tool": name, "input": args, "why": why, "output": out})
        return out

    report = call("get_scene_report", {}, "Perceive: run the OpenCV 5 pipeline over the clip.")
    for t in report["tracks"]:
        tid = t["track_id"]
        if not t["entered_zone"]:
            continue
        if not t["persistence_verified"]:
            call("log_and_monitor", {"track_id": tid, "reason": f"In the cell for only {t['time_in_zone_s']:.2f} s, below the 1.5 s persistence threshold, so no stop."},
                 "Brief entry: under the persistence threshold.")
            continue
        if t["coarse_ppe_status"] != "compliant":
            insp = call("inspect_worker", {"track_id": tid},
                        f"Wide-view vest check reads '{t['coarse_ppe_status']}' ({t['coarse_vest_ratio']:.0%} hi-vis); confirm up close before stopping the line.")
            if not insp.get("vest_present"):
                call("trigger_estop_alert", {"track_id": tid, "reason": f"Verified {t['time_in_zone_s']:.2f} s in the cell, and the close-up confirms no hi-vis vest ({insp.get('vest_coverage', 0):.0%} of torso)."},
                     "The close-up confirmed the violation.")
                continue
            call("request_supervisor_approval", {"track_id": tid, "reason": f"The wide view missed the vest, but the close-up found it ({insp.get('vest_coverage', 0):.0%} of torso); likely an authorized entry."},
                 "The close-up overturned the wide-view reading, avoiding a false stop.")
            continue
        call("request_supervisor_approval", {"track_id": tid, "reason": f"Verified {t['time_in_zone_s']:.2f} s in the cell wearing PPE; confirm this is authorized maintenance."},
             "Verified entry by a worker wearing PPE.")
    return AgentRun(mode="rules", model_id=None, trace=trace, decisions=dict(tools.decisions), blocked=list(tools.blocked),
                    summary=summarize_decisions(tools), steps=len(trace), latency_ms=int((time.perf_counter() - t0) * 1000))


_TAG = re.compile(r"</?(thinking|shift_summary|summary|answer)>", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Drop model scaffolding tags (e.g. <thinking>) so the UI shows plain prose; keeps line breaks for Markdown."""
    text = re.sub(r"[ \t]+", " ", _TAG.sub(" ", text or ""))
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _unresolved(tools: SafetyTools) -> list[str]:
    return sorted(t for t, s in tools.p.tracks.items() if s.in_zone_frames and t not in tools.decisions)


def run_bedrock(tools: SafetyTools, client: Any, model_id: str = DEFAULT_MODEL_ID, max_steps: int = 16, max_nudges: int = 2) -> AgentRun:
    """LLM decision-maker via the Bedrock Converse API with tool use.

    If the model stops while zone tracks are still unresolved, the controller tells
    it which ones (with their evidence) and lets it self-correct, up to ``max_nudges``.
    """
    t0 = time.perf_counter()
    trace: list[dict] = []
    messages: list[dict] = [{"role": "user", "content": [{"text": "A new camera clip from the robot cell is ready. Assess it and act."}]}]
    summary, nudges, force_tool = "", 0, False
    for _ in range(max_steps):
        tool_config: dict = {"tools": TOOL_SPECS}
        if force_tool:
            tool_config["toolChoice"] = {"any": {}}
        resp = client.converse(
            modelId=model_id, messages=messages, system=[{"text": SYSTEM_PROMPT}],
            toolConfig=tool_config, inferenceConfig={"maxTokens": 900, "temperature": 0.0},
        )
        force_tool = False
        msg = resp["output"]["message"]
        messages.append(msg)
        texts = [b["text"] for b in msg.get("content", []) if "text" in b]
        uses = [b["toolUse"] for b in msg.get("content", []) if "toolUse" in b]
        if not uses:
            pending = _unresolved(tools)
            if pending and nudges < max_nudges:
                nudges += 1
                evidence = [tools.p.tracks[t].public() for t in pending]
                trace.append({"step": len(trace) + 1, "tool": "controller", "input": {"unresolved": pending},
                              "why": "Agent finished with zone tracks still unresolved; asking it to resolve them.", "output": {"status": "nudge"}})
                messages.append({"role": "user", "content": [{"text": (
                    f"These zone tracks are still unresolved: {', '.join(pending)}. Evidence: {json.dumps(evidence)}. "
                    "Resolve each with exactly one action tool, then give the shift summary.")}]})
                force_tool = True
                continue
            summary = clean_text(" ".join(texts))
            break
        results = []
        for use in uses:
            out = tools.call(use["name"], use.get("input") or {})
            trace.append({"step": len(trace) + 1, "tool": use["name"], "input": use.get("input") or {},
                          "why": clean_text(" ".join(texts))[:400], "output": out})
            results.append({"toolResult": {"toolUseId": use["toolUseId"], "content": [{"json": out}],
                                           "status": "error" if out.get("status") == "blocked_by_guardrail" or "error" in out else "success"}})
        messages.append({"role": "user", "content": results})
    return AgentRun(mode="bedrock", model_id=model_id, trace=trace, decisions=dict(tools.decisions), blocked=list(tools.blocked),
                    summary=summary or summarize_decisions(tools), steps=len(trace), latency_ms=int((time.perf_counter() - t0) * 1000))


def run_agent(perception: Perception, sink: ActionSink | None = None, client: Any = None,
              model_id: str = DEFAULT_MODEL_ID, prefer: str = "bedrock") -> tuple[AgentRun, SafetyTools]:
    """Run the agent; fall back to the rule policy (fresh tools) if Bedrock is unavailable or errors."""
    if prefer == "bedrock" and client is not None:
        tools = SafetyTools(perception, sink)
        try:
            run = run_bedrock(tools, client, model_id)
            # Every zone track must be resolved; otherwise the LLM run is incomplete → fall back.
            unresolved = _unresolved(tools)
            if not unresolved:
                return run, tools
            reason = f"LLM left tracks unresolved after nudges: {unresolved}"
        except Exception as exc:  # pragma: no cover - network path
            reason = f"Bedrock unavailable: {type(exc).__name__}: {exc}"[:300]
        # Fresh tools so no half-finished LLM action leaks into the fallback run.
        tools = SafetyTools(perception, RecordingSink() if sink is None else sink)
        run = run_rules(tools)
        run.fallback_reason = reason
        return run, tools
    tools = SafetyTools(perception, sink)
    return run_rules(tools), tools


# ---------------------------------------------------------------------------
# Evaluation against scenario ground truth
# ---------------------------------------------------------------------------


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_tracks(perception: Perception, workers: list) -> dict[str, str]:
    """Map track_id → ground-truth worker_id by mean IoU over shared frames."""
    mapping: dict[str, str] = {}
    for tid, tr in perception.tracks.items():
        best, best_score = None, 0.0
        for w in workers:
            shared = [f for f in tr.bbox_by_frame if f in w.poses]
            if not shared:
                continue
            score = sum(_iou(tr.bbox_by_frame[f], w.poses[f]) for f in shared) / len(shared)
            if score > best_score:
                best, best_score = w.worker_id, score
        if best is not None and best_score >= 0.5:
            mapping[tid] = best
    return mapping


def evaluate_run(perception: Perception, run: AgentRun, workers: list) -> dict:
    mapping = match_tracks(perception, workers)
    expected = {w.worker_id: w.expected_action for w in workers}
    rows, correct = [], 0
    false_stops = missed_stops = 0
    for w in workers:
        tids = [t for t, wid in mapping.items() if wid == w.worker_id]
        got = None
        for t in tids:
            if t in run.decisions:
                got = run.decisions[t]["action"]
        ok = got == expected[w.worker_id]
        correct += int(ok)
        if got == "trigger_estop_alert" and expected[w.worker_id] != "trigger_estop_alert":
            false_stops += 1
        if expected[w.worker_id] == "trigger_estop_alert" and got != "trigger_estop_alert":
            missed_stops += 1
        rows.append({"worker": w.worker_id, "label": w.label, "tracks": tids, "expected": expected[w.worker_id], "got": got, "correct": ok})
    return {
        "decision_accuracy": round(correct / len(workers), 3) if workers else 0.0,
        "false_estops": false_stops,
        "missed_estops": missed_stops,
        "tracks_matched": len(mapping),
        "rows": rows,
    }


def to_jsonable(obj: Any) -> Any:
    """Strip numpy types so results serialise cleanly."""
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
