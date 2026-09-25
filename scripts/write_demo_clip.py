"""Write an annotated demo clip a judge can watch without a camera.

The person track is scripted and the detector boxes are supplied, so this shows
zone intrusion, missing hi-vis, and debounce. It is not a webcam recording.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edgesentry.hazard_detector import Detection
from edgesentry.pipeline import EdgeSentryPipeline
from edgesentry.synthetic import make_frame

OUT = ROOT / "docs" / "edgesentry-demo.mp4"
FPS = 12
WIDTH, HEIGHT = 960, 540


def main() -> None:
    pipeline = EdgeSentryPipeline()
    pipeline.register_zone("zone_a", [(360, 320), (620, 320), (620, 520), (360, 520)], "Robot Cell A")
    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise SystemExit("OpenCV could not open a video writer")
    frames = FPS * 8
    for i in range(frames):
        x = 80 + int(i * (520 / frames))
        hi_vis = x < 340
        bbox = (x, 280, x + 70, 500)
        frame = make_frame(width=WIDTH, height=HEIGHT, people=[{"bbox": bbox, "hi_vis": hi_vis}])
        det = Detection(id="worker", bbox=bbox, has_hardhat=True, has_high_vis_vest=hi_vis)
        result = pipeline.process_frame(frame, now=i / FPS, detections=[det])
        writer.write(result.annotated)
    writer.release()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {frames} frames)")


if __name__ == "__main__":
    main()
