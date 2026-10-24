"""Reproducible evaluation for EdgeSentry. Every number in the docs comes from this script.

    python scripts/evaluate.py                  # offline robustness sweep + ablation (deterministic)
    python scripts/evaluate.py --bedrock 10     # + 10 live Amazon Bedrock agent trials (needs AWS credentials)

Writes docs/EVALUATION.json and prints Markdown tables.

What is measured
----------------
* Robustness sweep — the OpenCV 5 perception + the fixed safety policy on scenario variants
  that stress known weak points (sensor noise, lighting drift vs. the clean plate, camera
  shift, deeper shadow on a vest, lower frame rate, two workers merging into one blob).
* Ablation — the same perception with a "coarse-only" policy (no close-up re-inspection),
  to show what the active-perception step buys.
* Agent trials — the Bedrock tool-using agent on the baseline clip, repeated N times
  (LLM output varies run to run): accuracy, false/missed stops, guardrail blocks, controller
  nudges, fallbacks, latency.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import statistics
import sys
import time
from dataclasses import replace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from edgesentry.agent import AgentRun, DEFAULT_MODEL_ID, evaluate_run, perceive, run_agent  # noqa: E402
from edgesentry.synthetic import GroundTruthWorker, Scenario, build_scenario, render_workers  # noqa: E402


# ---------------------------------------------------------------- scenario variants


def _map_frames(sc: Scenario, fn) -> Scenario:
    return replace(sc, frames=[fn(f, i) for i, f in enumerate(sc.frames)])


def _noise(sigma: float):
    def fn(frame, i):
        cv2.setRNGSeed(1000 + i)
        n = np.empty(frame.shape, np.int16)
        cv2.randn(n, 0, sigma)
        return np.clip(frame.astype(np.int16) + n, 0, 255).astype(np.uint8)
    return fn


def _lighting(delta: int):
    return lambda frame, i: cv2.add(frame, np.full_like(frame, delta)) if delta > 0 else cv2.subtract(frame, np.full_like(frame, -delta))


def _shift(px: int):
    m = np.float32([[1, 0, px], [0, 1, px // 2]])
    return lambda frame, i: cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def _rotate(deg: float):
    def fn(frame, i):
        h, w = frame.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        return cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return fn


def _crouching_worker() -> Scenario:
    """A worker without PPE kneels inside the cell (maintenance posture): wider than tall."""
    a = GroundTruthWorker("W-31", "Kneeling operator (no PPE)", "none", False, "trigger_estop_alert")
    for f in range(24):
        a.poses[f] = (700, 570, 800, 690)
    return render_workers([a], 24)


def _three_in_a_row() -> Scenario:
    spec = [("Operator (no PPE)", "none", "trigger_estop_alert"),
            ("Technician (full PPE)", "full", "request_supervisor_approval"),
            ("Technician (full PPE)", "full", "request_supervisor_approval")]
    workers = [GroundTruthWorker(f"W-4{k}", label, vest, vest != "none", act) for k, (label, vest, act) in enumerate(spec)]
    for f in range(24):
        for k, w in enumerate(workers):
            w.poses[f] = (640 + 72 * k, 460, 710 + 72 * k, 690)
    return render_workers(workers, 24)


def _adjacent_workers() -> Scenario:
    """Two people standing shoulder to shoulder in the cell: one without PPE, one with."""
    a = GroundTruthWorker("W-12", "Operator (no PPE)", "none", False, "trigger_estop_alert")
    b = GroundTruthWorker("W-20", "Technician beside them (full PPE)", "full", True, "request_supervisor_approval")
    for f in range(24):
        a.poses[f] = (700, 460, 770, 690)
        b.poses[f] = (772, 460, 842, 690)
    return render_workers([a, b], 24)


def variants() -> list[tuple[str, str, Scenario]]:
    base = build_scenario()
    return [
        ("baseline", "Reference clip: 4 workers, 24 frames at 4 fps", base),
        ("noise_seed_1", "Baseline with a different sensor-noise seed", build_scenario(seed=1)),
        ("noise_seed_99", "Baseline with a different sensor-noise seed", build_scenario(seed=99)),
        ("sensor_noise_8", "Extra Gaussian sensor noise, sigma 8", _map_frames(base, _noise(8))),
        ("sensor_noise_16", "Extra Gaussian sensor noise, sigma 16", _map_frames(base, _noise(16))),
        ("lighting_plus_30", "Scene 30 levels brighter than the clean plate", _map_frames(base, _lighting(30))),
        ("lighting_minus_30", "Scene 30 levels darker than the clean plate", _map_frames(base, _lighting(-30))),
        ("camera_shift_3px", "Camera bumped 3 px right / 1 px down", _map_frames(base, _shift(3))),
        ("camera_shift_8px", "Camera bumped 8 px right / 4 px down", _map_frames(base, _shift(8))),
        ("half_frame_rate", "Same script sampled at 2 fps (12 s clip)", build_scenario(dt=0.5)),
        ("deeper_shadow_0.15", "Contractor's vest keeps 15% of its brightness", render_workers(build_scenario().workers, 24, shadow=0.15)),
        ("deeper_shadow_0.08", "Contractor's vest keeps 8% of its brightness", render_workers(build_scenario().workers, 24, shadow=0.08)),
        ("adjacent_workers", "Two workers shoulder to shoulder (blobs merge)", _adjacent_workers()),
        # Stress cases beyond the design envelope: reported to show where it breaks.
        ("stress_three_in_a_row", "Three workers shoulder to shoulder", _three_in_a_row()),
        ("stress_lighting_plus_80", "Scene 80 levels brighter (highlights clip)", _map_frames(base, _lighting(80))),
        ("stress_camera_rotate_1deg", "Camera rotated 1 degree about the image centre", _map_frames(base, _rotate(1.0))),
        ("stress_crouching_worker", "Worker kneeling in the cell (wider than tall)", _crouching_worker()),
    ]


# ---------------------------------------------------------------- policies


def coarse_only(perception) -> AgentRun:
    """Ablation: stop on the wide-view PPE reading alone (no close-up re-inspection)."""
    decisions = {}
    for tid, tr in perception.tracks.items():
        if not tr.in_zone_frames:
            continue
        if not tr.verified:
            action = "log_and_monitor"
        elif tr.ppe_status != "compliant":
            action = "trigger_estop_alert"
        else:
            action = "request_supervisor_approval"
        decisions[tid] = {"action": action, "track_id": tid}
    return AgentRun(mode="coarse_only", model_id=None, trace=[], decisions=decisions, blocked=[], summary="", steps=0, latency_ms=0)


_NO_ROBUSTNESS = {"register": False, "compensate_light": False, "split_merged": False}


def evaluate_variant(name: str, desc: str, sc: Scenario) -> dict:
    perception = perceive(sc.frames, sc.timestamps, sc.background, sc.zone)
    run, _ = run_agent(perception, prefer="rules")
    ev = evaluate_run(perception, run, sc.workers)
    ab = evaluate_run(perception, coarse_only(perception), sc.workers)
    # Ablation 2: the same policy on perception without registration / lighting / blob splitting.
    plain = perceive(sc.frames, sc.timestamps, sc.background, sc.zone, detector_options=_NO_ROBUSTNESS)
    ev_plain = evaluate_run(plain, run_agent(plain, prefer="rules")[0], sc.workers)
    wrong = [f"{r['worker']}: expected {r['expected']}, got {r['got'] or 'no action'}" for r in ev["rows"] if not r["correct"]]
    return {
        "variant": name, "description": desc, "workers": len(sc.workers), "tracks": len(perception.tracks),
        "workers_matched": len(set(m for m in _matched(ev))), "perception_fps": round(perception.fps, 1),
        "decision_accuracy": ev["decision_accuracy"], "false_estops": ev["false_estops"], "missed_estops": ev["missed_estops"],
        "errors": wrong,
        "camera": perception.camera,
        "coarse_only": {"decision_accuracy": ab["decision_accuracy"], "false_estops": ab["false_estops"], "missed_estops": ab["missed_estops"]},
        "without_robustness_steps": {"decision_accuracy": ev_plain["decision_accuracy"], "false_estops": ev_plain["false_estops"],
                                     "missed_estops": ev_plain["missed_estops"]},
    }


def _matched(ev: dict):
    return [r["worker"] for r in ev["rows"] if r["tracks"]]


# ---------------------------------------------------------------- live agent trials


def bedrock_trials(n: int, region: str, model_id: str) -> dict:
    import boto3
    from botocore.config import Config

    client = boto3.client("bedrock-runtime", region_name=region,
                          config=Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 2, "mode": "standard"}))
    sc = build_scenario()
    perception = perceive(sc.frames, sc.timestamps, sc.background, sc.zone)
    trials = []
    for i in range(n):
        t0 = time.perf_counter()
        run, _ = run_agent(perception, client=client, model_id=model_id, prefer="bedrock")
        ev = evaluate_run(perception, run, sc.workers)
        trials.append({
            "trial": i + 1, "decided_by": run.mode, "fallback_reason": run.fallback_reason,
            "decision_accuracy": ev["decision_accuracy"], "false_estops": ev["false_estops"], "missed_estops": ev["missed_estops"],
            "guardrail_blocks": len(run.blocked), "controller_nudges": sum(1 for s in run.trace if s["tool"] == "controller"),
            "blocked": [{k: b[k] for k in ("action", "track_id", "reason")} for b in run.blocked],
            "nudged_for": [s["input"]["unresolved"] for s in run.trace if s["tool"] == "controller"],
            "steps": run.steps, "latency_s": round(time.perf_counter() - t0, 2),
        })
        print(f"  trial {i + 1}/{n}: {run.mode}, accuracy {ev['decision_accuracy']}, blocks {len(run.blocked)}, "
              f"nudges {trials[-1]['controller_nudges']}, {trials[-1]['latency_s']} s", flush=True)
    agent_runs = [t for t in trials if t["decided_by"] == "bedrock"]
    lat = [t["latency_s"] for t in agent_runs]
    return {
        "model_family": "Amazon Nova (Bedrock Converse API, tool use)", "region": region, "trials": trials,
        "summary": {
            "n": n, "completed_by_agent": len(agent_runs), "fell_back_to_policy": n - len(agent_runs),
            "all_correct_runs": sum(1 for t in trials if t["decision_accuracy"] == 1.0),
            "mean_accuracy": round(statistics.mean(t["decision_accuracy"] for t in trials), 3),
            "false_estops_total": sum(t["false_estops"] for t in trials), "missed_estops_total": sum(t["missed_estops"] for t in trials),
            "guardrail_blocks_total": sum(t["guardrail_blocks"] for t in trials),
            "runs_needing_a_nudge": sum(1 for t in trials if t["controller_nudges"]),
            "agent_latency_s_median": round(statistics.median(lat), 2) if lat else None,
            "agent_latency_s_max": max(lat) if lat else None,
        },
    }


# ---------------------------------------------------------------- report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bedrock", type=int, default=0, help="number of live Bedrock agent trials (0 = skip)")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--out", default=str(ROOT / "docs" / "EVALUATION.json"))
    args = ap.parse_args()

    out_path = pathlib.Path(args.out)
    previous = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

    print("Robustness sweep (OpenCV 5 perception + fixed safety policy)")
    rows = [evaluate_variant(*v) for v in variants()]
    print("\n| Variant | Tracks / workers | Accuracy | False stops | Missed stops | No close-up: accuracy / false stops | No robustness steps: accuracy / missed stops |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        co, nr = r["coarse_only"], r["without_robustness_steps"]
        print(f"| {r['variant']} | {r['tracks']} / {r['workers']} | {r['decision_accuracy']:.2f} | {r['false_estops']} | {r['missed_estops']} | "
              f"{co['decision_accuracy']:.2f} / {co['false_estops']} | {nr['decision_accuracy']:.2f} / {nr['missed_estops']} |")
    for r in rows:
        for e in r["errors"]:
            print(f"  ✗ {r['variant']}: {e}")

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
                        "platform": platform.platform(), "processor": platform.processor() or platform.machine()},
        "robustness_sweep": rows,
        "agent_trials": previous.get("agent_trials"),
    }
    if args.bedrock:
        print(f"\nLive Bedrock agent trials ({args.bedrock}) on the baseline clip")
        report["agent_trials"] = bedrock_trials(args.bedrock, args.region, args.model_id)
        report["agent_trials"]["generated_at"] = report["generated_at"]
    if report["agent_trials"]:
        s = report["agent_trials"]["summary"]
        print(f"\nAgent: {s['all_correct_runs']}/{s['n']} runs fully correct, {s['false_estops_total']} false stops, "
              f"{s['missed_estops_total']} missed, {s['guardrail_blocks_total']} guardrail blocks, "
              f"{s['runs_needing_a_nudge']} runs needed a nudge, {s['fell_back_to_policy']} fallbacks, "
              f"median {s['agent_latency_s_median']} s")
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
