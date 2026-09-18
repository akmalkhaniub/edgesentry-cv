# Roadmap & Milestones: EdgeSentry
**Hackathon:** OpenCV AI Competition 2026 (powered by AWS)  
**Target Submission Deadline:** October 27, 2026  

---

> **Status legend (updated 2026-09-18):** `[x]` implemented in code · `[~]` partial / needs hardware, a model, or deployed infra to verify · `[ ]` not started.
> **Reality note:** **Rebuilt in Python + OpenCV** (`edgesentry/` package; the old Node prototype is preserved under `legacy-js/`). Real OpenCV pipeline: `cv2.pointPolygonTest` zone intrusion, HSV hi-vis PPE check, `cv2.dnn` ONNX detector, `cv2.VideoCapture` streaming + annotation, and boto3 SNS/S3 dispatch with a simulator fallback (8 pytest cases pass on synthetic frames). Still needs a real camera/RTSP feed, a trained ONNX model, edge hardware, and deployed AWS infra to run in production.

## Phase 1: OpenCV Pipeline & Model Quantization (Week 1)
- [ ] Implement OpenCV Python capture loop supporting RTSP stream or test video files.
- [ ] Export and quantize ONNX vision model (person + safety gear detector) for `cv2.dnn`.
- [~] Implement polygon danger-zone editor and in-frame point projection. *(hazard-zone logic in JS)*
- [~] Add ByteTrack object tracker to eliminate false positives. *(temporal event filter stand-in)*

## Phase 2: AWS Serverless Cloud Ingestion (Week 2)
- [ ] Write AWS SAM / CDK template deploying API Gateway, Lambda, DynamoDB, and S3.
- [~] Implement secure edge authentication via AWS IoT Core / Greengrass certificates. *(mock Greengrass client)*
- [~] Implement automatic thumbnail generation and signed S3 URL dispatch in Lambda. *(mock dispatcher)*

## Phase 3: Web Console & Incident Management (Week 3)
- [~] Build responsive safety supervisor dashboard (Next.js 14 + Tailwind). *(static HTML)*
- [~] Add live zone drawing tool allowing supervisors to draw restricted polygons directly on video stream.
- [~] Implement instant push notification feed and historical violation log.

## Phase 4: Testing, Video & Devpost Submission (Week 4)
- [ ] Run automated load test simulating 10 concurrent camera streams sending violation bursts.
- [ ] Record high-quality demonstration video showcasing edge detection, AWS sync, and dashboard alerts.
- [ ] Finalize Devpost write-up and publish GitHub repository.
