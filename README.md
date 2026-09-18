# 🦺 EdgeSentry — Edge CV Workplace Safety Monitor

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B%20%2F%205.x-5C3EE8.svg)](https://opencv.org)
[![AWS](https://img.shields.io/badge/AWS-SNS%20%2B%20S3-FF9900.svg)](https://aws.amazon.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> **Built for the [OpenCV AI Competition 2026 (powered by AWS)](https://opencv26.devpost.com/).**

EdgeSentry watches an industrial camera feed and raises an alert when a person enters a
restricted zone — especially without hi-vis PPE. It runs a real **OpenCV** pipeline on the
edge and dispatches verified incidents to **AWS** (SNS alert + S3 snapshot).

> **Note:** this project was rebuilt from an earlier Node.js prototype (kept under
> [`legacy-js/`](./legacy-js)) into the correct stack — **Python + OpenCV** — which is
> what the OpenCV competition requires.

## Pipeline

```
VideoCapture (RTSP / file / camera)          cv2.VideoCapture
   → Detector                                cv2.dnn ONNX  (or HOG on full OpenCV builds)
   → Spatial hazard zones                     cv2.pointPolygonTest
   → Hi-vis PPE check                          cv2.cvtColor + cv2.inRange (HSV)
   → Depth / proximity estimate                pinhole model (OAK-D stereo stand-in)
   → Temporal debounce                         persistence filter (kills flicker)
   → Annotate                                  cv2.polylines / rectangle / putText
   → AWS dispatch                              boto3 SNS publish + S3 upload (mock fallback)
```

Every geometry / vision step uses genuine OpenCV primitives. The detector is pluggable:
supply an **ONNX** person/PPE model (run via `cv2.dnn`), use the OpenCV **HOG** pedestrian
detector where available, or feed detections from any external model.

## Install & run

```bash
pip install -r requirements-dev.txt
pip install -e .

pytest -q                 # test suite (real OpenCV, synthetic frames — no camera)
edgesentry --demo         # offline end-to-end demo (prints detected violations)

# Live stream with a restricted zone polygon (x,y;x,y;...) and an ONNX detector:
edgesentry --source rtsp://cam/stream --onnx yolo_ppe.onnx \
           --zone "360,320;620,320;620,520;360,520" --out annotated.mp4
```

## AWS dispatch

Set these to publish real alerts (otherwise a deterministic simulator is used):

```bash
export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...  AWS_REGION=us-east-1
export EDGESENTRY_SNS_TOPIC_ARN=arn:aws:sns:us-east-1:123456789012:edgesentry-alerts
export EDGESENTRY_S3_BUCKET=my-edgesentry-snapshots
```

Each verified violation uploads a JPEG snapshot to S3 and publishes an SNS notification
(fits an API Gateway → Lambda → DynamoDB/SNS serverless topology).

## Testing

`pytest` covers the point-in-polygon zone logic (`cv2.pointPolygonTest`), PPE
classification, the temporal debounce + eviction, depth/proximity, HSV hi-vis detection,
and an end-to-end pipeline dispatch — all on synthetic frames, so it runs in CI without a
camera or model. CI runs on Python 3.10–3.12.
