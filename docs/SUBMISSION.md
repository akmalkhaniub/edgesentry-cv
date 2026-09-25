# EdgeSentry submission notes

Deadline 27 Oct 2026.

## What a judge can watch and run

Demo clip: `docs/edgesentry-demo.mp4` (8 seconds, 960x540). A scripted person walks into a restricted zone, loses the hi-vis vest, and the pipeline annotates the frame. Boxes are supplied to the pipeline so the clip does not depend on a trained model.

```bash
python scripts/write_demo_clip.py
edgesentry --demo
edgesentry --bench --frames 40
```

Bench on this machine, 2026-09-25: contour detector, 40 synthetic frames, **27.24 fps** (`docs/BENCH.json`).

## Not in this submission

No bundled ONNX weights, no webcam recording, and no live AWS SNS alert.
