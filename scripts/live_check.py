"""Measure the deployed endpoint: N safety checks per mode, with per-stage timings.

    python scripts/live_check.py https://<function-url>/ --runs 5

Writes docs/LIVE_CHECK.json. Each run really executes in AWS Lambda (and, in "bedrock" mode,
calls Amazon Bedrock) and really writes S3 evidence and publishes SNS alerts.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]


def call(url: str, mode: str) -> tuple[dict, float]:
    req = urllib.request.Request(url.rstrip("/") + "/api/run", data=json.dumps({"mode": mode}).encode(),
                                 headers={"content-type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=90) as r:
        body = json.loads(r.read())
    return body, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--modes", default="rules,bedrock")
    args = ap.parse_args()

    health = json.loads(urllib.request.urlopen(args.url.rstrip("/") + "/api/health", timeout=30).read())
    report: dict = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "health": health, "modes": {}}
    for mode in args.modes.split(","):
        runs = []
        for i in range(args.runs):
            body, wall = call(args.url, mode)
            delivered = [d.get("delivery") or {} for d in body["decisions"]]
            runs.append({
                "decided_by": body["decision_mode"], "fallback": body.get("fallback_reason"),
                "decision_accuracy": body["evaluation"]["decision_accuracy"], "false_estops": body["evaluation"]["false_estops"],
                "guardrail_blocks": len(body["guardrail_blocks"]), "steps": len(body["trace"]),
                "evidence_stored": sum(1 for d in delivered if d.get("s3_key")),
                "alerts_published": sum(1 for d in delivered if d.get("sns_message_id")),
                "lambda_ms": body["timings"], "round_trip_s": round(wall, 2),
            })
            print(f"{mode} {i + 1}/{args.runs}: {runs[-1]['decided_by']} acc={runs[-1]['decision_accuracy']} "
                  f"perception={body['timings']['perception_ms']}ms agent={body['timings']['agent_ms']}ms round-trip={wall:.2f}s", flush=True)
        warm = runs[1:] or runs  # the first call may include a cold start
        report["modes"][mode] = {
            "runs": runs,
            "summary": {
                "n": len(runs), "all_correct": sum(1 for r in runs if r["decision_accuracy"] == 1.0),
                "false_estops": sum(r["false_estops"] for r in runs),
                "perception_ms_median_warm": statistics.median(r["lambda_ms"]["perception_ms"] for r in warm),
                "perception_fps_median_warm": statistics.median(r["lambda_ms"]["perception_fps"] for r in warm),
                "decision_ms_median_warm": statistics.median(r["lambda_ms"]["agent_ms"] for r in warm),
                "round_trip_s_median_warm": statistics.median(r["round_trip_s"] for r in warm),
                "evidence_stored_per_run": statistics.median(r["evidence_stored"] for r in runs),
                "alerts_published_per_run": statistics.median(r["alerts_published"] for r in runs),
            },
        }
    out = ROOT / "docs" / "LIVE_CHECK.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({m: v["summary"] for m, v in report["modes"].items()}, indent=2))


if __name__ == "__main__":
    main()
