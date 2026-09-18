# Changelog

## [Unreleased]

### Rebuilt in Python + OpenCV (2026-09-18)
The OpenCV AI Competition expects a genuine OpenCV pipeline, so the project was rebuilt
from the Node.js prototype (preserved under `legacy-js/`) into a Python package that uses
real OpenCV primitives end-to-end.

### Added
- `edgesentry/` Python package:
  - `hazard_detector.py` — restricted-zone intrusion via `cv2.pointPolygonTest`.
  - `detector.py` — `OnnxDetector` (`cv2.dnn`) + `HOGPeopleDetector`; HSV hi-vis PPE
    check via `cv2.cvtColor` + `cv2.inRange`.
  - `depth_estimator.py` — pinhole proximity estimate (OAK-D stereo stand-in).
  - `temporal_filter.py` — persistence debounce that suppresses flicker.
  - `pipeline.py` — capture → detect → zones → filter → annotate → dispatch, on
    real `cv2.VideoCapture` with `cv2.polylines`/`rectangle`/`putText` overlays.
  - `aws_dispatch.py` — boto3 SNS publish + S3 snapshot upload, with simulator fallback.
  - `cli.py` — `edgesentry --source ... --onnx ... --zone ...` and an offline `--demo`.
- `tests/test_edgesentry.py` — 8 pytest cases on synthetic frames (no camera/model needed).
- `pyproject.toml`, `requirements*.txt`, Python Dockerfile, CI on Python 3.10–3.12.

### Notes
- The detector is pluggable; the slim OpenCV 5 headless wheel omits `HOGDescriptor`, so
  the ONNX (`cv2.dnn`) backend or externally-supplied detections are the portable path
  (the HOG test skips when unavailable). Live streams/models and real AWS require
  credentials + a deployed camera; see README.
