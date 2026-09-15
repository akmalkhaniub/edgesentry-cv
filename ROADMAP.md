# Roadmap & Milestones: EdgeSentry
**Hackathon:** OpenCV AI Competition 2026 (powered by AWS)  
**Target Submission Deadline:** October 27, 2026  

---

## Phase 1: OpenCV Pipeline & Model Quantization (Week 1)
- [ ] Implement OpenCV Python capture loop supporting RTSP stream or test video files.
- [ ] Export and quantize ONNX vision model (person + safety gear detector) for `cv2.dnn`.
- [ ] Implement polygon danger-zone editor and in-frame point projection.
- [ ] Add ByteTrack object tracker to eliminate false positives.

## Phase 2: AWS Serverless Cloud Ingestion (Week 2)
- [ ] Write AWS SAM / CDK template deploying API Gateway, Lambda, DynamoDB, and S3.
- [ ] Implement secure edge authentication via AWS IoT Core / Greengrass certificates.
- [ ] Implement automatic thumbnail generation and signed S3 URL dispatch in Lambda.

## Phase 3: Web Console & Incident Management (Week 3)
- [ ] Build responsive safety supervisor dashboard (Next.js 14 + Tailwind).
- [ ] Add live zone drawing tool allowing supervisors to draw restricted polygons directly on video stream.
- [ ] Implement instant push notification feed and historical violation log.

## Phase 4: Testing, Video & Devpost Submission (Week 4)
- [ ] Run automated load test simulating 10 concurrent camera streams sending violation bursts.
- [ ] Record high-quality demonstration video showcasing edge detection, AWS sync, and dashboard alerts.
- [ ] Finalize Devpost write-up and publish GitHub repository.
