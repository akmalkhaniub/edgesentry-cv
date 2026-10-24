# Submission checklist: OpenCV AI Competition 2026

Final deadline: **October 26, 2026, 11:59 PM PT**.

| Requirement | Where |
|---|---|
| Uses OpenCV 5 substantively | `edgesentry/detector.py`, `edgesentry/agent.py`, `edgesentry/overlay.py`; see report §4 |
| Meaningful component runs on AWS | Lambda (OpenCV 5 perception + agent), Bedrock, S3, SNS, DynamoDB, CloudWatch; `infra/template.yaml` |
| Technical report (problem, users, architecture, OpenCV 5, AWS, evaluation, limitations, responsible use) | `docs/TECHNICAL_REPORT.md` |
| Judge-accessible repository | https://github.com/akmalkhaniub/edgesentry-cv |
| Pinned dependencies + build / deploy / test instructions | `requirements*.txt`, `pyproject.toml`, `README.md` → Run it |
| Architecture diagram | `README.md`, report §3 (Mermaid; renders on GitHub) |
| Working web endpoint | https://jzyv7mljhphuteckul2lxqe77u0xbqev.lambda-url.us-east-1.on.aws/ |
| Video ≤ 5 minutes (team, app working, architecture, results) | `docs/demo/edgesentry_demo.mp4` → upload as public/unlisted |
| Evaluation evidence including failure cases | `docs/EVALUATION.json`, `docs/LIVE_CHECK.json`, report §7 |
| Agentic Vision Award: agent workflow diagram | `README.md`, report §5 |
| Agentic Vision Award: trace where OpenCV output changes a later action | `docs/TRACE_EXAMPLE.json`, report §5 |
| Agentic Vision Award: task success, failure handling, observability, human control | report §5, §7.3, §7.4 |
