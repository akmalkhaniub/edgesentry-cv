# Changelog

## [2.0.0] - 2026-10-24

### Added
- Agentic perception → decision → action loop (`edgesentry/agent.py`): OpenCV tools
  (`get_scene_report`, `inspect_worker`), action tools, code guardrails, Bedrock Converse
  tool-use loop with a completeness controller, and a deterministic fallback policy.
- Clean-plate detector hardening (`edgesentry/detector.py`): phase-correlation + ECC camera
  registration with foreground masking, lighting compensation, misalignment-tolerant
  envelope differencing, head-profile group splitting, fail-safe "unrecognized" shapes,
  camera-health signals.
- Cloud service and Lambda handler (`edgesentry/service.py`, `lambda_function.py`): S3
  evidence, SNS alerts, CloudWatch EMF metrics, daily usage caps, supervisor decisions.
- Web app (`edgesentry/web/index.html`) and a causal frame overlay (`edgesentry/overlay.py`).
- AWS infrastructure (`infra/`): CloudFormation template, deploy and teardown scripts.
- Evaluation (`scripts/evaluate.py`, `scripts/live_check.py`) with results in `docs/`.
- Technical report (`docs/TECHNICAL_REPORT.md`).

### Changed
- Dependencies pinned (`numpy==2.4.4`, `opencv-python-headless==5.0.0.93`); Python 3.11+.
- `EdgeSentryPipeline.draw_zones` no longer mutates the caller's frame.

### Removed
- `docs/PITCH_DECK.md`, `docs/pitch_deck.html`: they stated performance numbers that were
  never measured in this build.

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
