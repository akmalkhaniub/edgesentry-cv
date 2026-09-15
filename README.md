# OpenCV AI Competition 2026 (powered by AWS)

- **Official Challenge URL:** [https://opencv26.devpost.com/](https://opencv26.devpost.com/)
- **Organizer:** OpenCV & Amazon Web Services (AWS)
- **Host Platform:** Devpost
- **Total Prize Pool:** $20,250 USD
- **Submission Dates:** August 26, 2026 – October 27, 2026
- **Format:** Online / Global
- **Primary Themes:** Computer Vision, Embedded & IoT Systems, Machine Learning / AI, Serverless & Edge

---

## 1. Hackathon Objective & Problem Statement
The OpenCV AI Competition powered by AWS tasks developers with bridging computer vision algorithms with edge devices and scalable AWS cloud architectures.

Projects must leverage **OpenCV** (including OAK cameras, OpenCV Python/C++, or OpenCV AI Kit) together with **AWS serverless or cloud services** (AWS Lambda, AWS Greengrass, S3, DynamoDB, SageMaker) to solve critical challenges in robotics, industrial inspection, wildlife conservation, or smart cities.

### Judging Criteria
1. **Computer Vision Depth (30%):** Sophistication of the OpenCV computer vision and deep learning pipelines.
2. **Edge-to-Cloud Integration (25%):** Effective use of AWS serverless services, edge sync, and efficient data payloads.
3. **Real-World Viability (25%):** Operational efficiency, low power consumption, and accuracy in challenging environments.
4. **Usability & Documentation (20%):** Clear setup guide, open-source code quality, and working video proof-of-concept.

---

## 2. Selected Project Concept: EdgeSentry (Edge CV & Serverless Industrial Hazard Detector)
An ultra-low-power edge vision system running quantized OpenCV/YOLO models on edge hardware (e.g. Raspberry Pi / OAK / Jetson) that detects worker safety gear (PPE) violations and machinery hazard zone intrusions in real-time, syncing event metadata and cryptographic proof to AWS via IoT Greengrass and Serverless Lambda/DynamoDB.

---

## 3. Directory Structure
```
opencv-ai-aws/
├── README.md               # Challenge details, judging rules, links (this file)
├── SPECIFICATION.md        # Edge pipeline, OpenCV filters, AWS cloud topology
├── ROADMAP.md              # Milestone schedule towards October 27 deadline
├── edge-camera/            # Python / C++ OpenCV capture & quantized inference
├── cloud-serverless/       # AWS CDK / SAM template (Lambda, DynamoDB, S3)
└── web-console/            # Real-time incident & site safety dashboard
```
