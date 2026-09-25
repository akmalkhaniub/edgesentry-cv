# EdgeSentry — demo script

Record the Python package, not the old Node dashboard in `legacy-js/`.

## What to show

1. `edgesentry --demo` — synthetic frame, zone intrusion, missing hi-vis, debounced alert. Say the dispatch is simulated unless AWS credentials are set.
2. `edgesentry --bench --frames 60` — read the printed FPS. It is the contour pipeline on synthetic frames, not a webcam and not a trained ONNX model.
3. If you have a clip or camera: `edgesentry --source 0 --detector hog --max-frames 100` or `--onnx path/to/model.onnx`.
4. Do not claim Luxonis stereo, a 15 ms PLC stop, or a 99.4% false-alarm rate. Those are not measured in this build.

Depth in the overlay is a bbox-height estimate, not a stereo camera.
