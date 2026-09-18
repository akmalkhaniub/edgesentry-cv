"""EdgeSentry CLI: run the pipeline on a video/RTSP stream or a synthetic demo."""
from __future__ import annotations

import argparse
import json
import sys

from .detector import HOGPeopleDetector, OnnxDetector
from .pipeline import EdgeSentryPipeline
from .synthetic import make_frame
from .hazard_detector import Detection


def _demo(pipeline: EdgeSentryPipeline) -> int:
    # A restricted zone in the lower-centre of the frame.
    pipeline.register_zone("zone_a", [(360, 320), (620, 320), (620, 520), (360, 520)], "Robot Cell A")
    people = [
        {"bbox": (450, 300, 520, 520), "hi_vis": False},  # intruder, no PPE, inside zone
        {"bbox": (100, 300, 170, 520), "hi_vis": True},   # worker outside zone
    ]
    frame = make_frame(people=people)
    dets = [
        Detection(id="p_intruder", bbox=(450, 300, 520, 520), has_hardhat=False, has_high_vis_vest=False),
        Detection(id="p_worker", bbox=(100, 300, 170, 520), has_hardhat=True, has_high_vis_vest=True),
    ]
    # Two ticks past the persistence threshold to fire a verified alert.
    pipeline.process_frame(frame, now=0.0, detections=dets)
    result = pipeline.process_frame(frame, now=2.0, detections=dets)
    print(json.dumps({
        "detections": len(result.detections),
        "violations": [v.violation_type for v in result.violations],
        "verified_alerts": len(result.verified),
        "dispatched": [{"provider": d.provider, "zone": d.zone_id, "type": d.violation_type} for d in result.dispatched],
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="edgesentry", description="Edge CV safety monitoring on OpenCV.")
    parser.add_argument("--source", help="Video file path, RTSP URL, or camera index (e.g. 0).")
    parser.add_argument("--onnx", help="Optional ONNX detector model path (else HOG people detector).")
    parser.add_argument("--zone", action="append", default=[], help="Zone polygon 'x1,y1;x2,y2;x3,y3[...]'. Repeatable.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--out", help="Optional annotated MP4 output path.")
    parser.add_argument("--demo", action="store_true", help="Run the offline synthetic demo (no camera).")
    args = parser.parse_args(argv)

    detector = OnnxDetector(args.onnx) if args.onnx else None
    pipeline = EdgeSentryPipeline(detector=detector)

    for i, spec in enumerate(args.zone):
        pts = [tuple(int(v) for v in pair.split(",")) for pair in spec.split(";")]
        pipeline.register_zone(f"zone_{i}", pts, f"Restricted Zone {i}")

    if args.demo or not args.source:
        return _demo(pipeline)

    # Live stream needs a real detector: ONNX if supplied, else HOG (full OpenCV only).
    if pipeline.detector is None:
        pipeline.detector = HOGPeopleDetector()

    source: str | int = int(args.source) if args.source.isdigit() else args.source
    total_alerts = 0
    for result in pipeline.run_stream(source, max_frames=args.max_frames, write_path=args.out):
        total_alerts += len(result.verified)
    print(f"Done. Verified alerts dispatched: {total_alerts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
