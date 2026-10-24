# EdgeSentry: specification (v2.0.0)

What the deployed system does, as built. For design rationale and results, see
[docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md).

## HTTP API (Lambda Function URL)

| Method and path | Body | Response |
|---|---|---|
| `GET /` | — | The web app (HTML) |
| `GET /api/health` | — | `{status, opencv, configured: {evidence_bucket, alert_topic, usage_limits, bedrock_enabled}, region}` |
| `POST /api/run` | `{"mode": "bedrock" \| "rules"}` | A safety-check result (below). `429` once the daily run cap is reached |
| `POST /api/decision` | `{"run_id", "track_id", "decision": "approve" \| "stop"}` | `200` recorded · `404` no such approval request · `409` already answered · `400` invalid |

Bodies over 4 KB are rejected (`413`). Errors never include stack traces; those go to
CloudWatch Logs. Responses carry `nosniff`, `no-referrer`, `DENY` framing and `no-store` headers.

## Safety-check result (`POST /api/run`)

| Field | Meaning |
|---|---|
| `run_id` | `YYYYMMDD-HHMMSS-xxxxxx` |
| `decision_mode`, `requested_mode` | Who decided (`bedrock` or `rules`) and what was asked for |
| `fallback_reason`, `fallback_notice` | Why the fixed policy decided instead of the agent, for logs and for people |
| `completed_by_policy` | Track ids the fixed policy finished after the agent left them open. Empty when the agent finished the run, or when the whole run fell back |
| `summary` | Markdown, one line per decision, generated from executed actions and measured evidence |
| `agent_notes` | The model's own closing note (not verified; AI mode only) |
| `decisions[]` | `track_id`, `action`, `action_label`, `reason`, `evidence` (time in zone, verified, coarse PPE, hard hat, shape), `inspection` (close-up result), `delivery` (S3 key, presigned URL, SNS message id), `evidence_image` (JPEG, base64), `evidence_frame`, `ground_truth` |
| `trace[]` | Every step: `tool`, `input`, `why`, `output` (includes guardrail blocks and controller nudges) |
| `guardrail_blocks[]` | Blocked actions with reasons |
| `evaluation` | Accuracy, false and missed stops, and per-worker rows against the scenario ground truth |
| `camera` | Drift corrected (px), lighting offset, max foreground fraction |
| `timings` | `render_ms`, `perception_ms`, `perception_fps`, `agent_ms`, `total_ms` |
| `frames[]`, `timestamps[]` | The annotated clip (JPEG, base64, 640 px wide) |

## Actions and guardrails

| Action | Allowed only when | Delivery |
|---|---|---|
| `trigger_estop_alert` | intrusion persistence-verified (1.5 s) **and** `inspect_worker` ran **and** found no vest | S3 evidence + SNS |
| `request_supervisor_approval` | intrusion persistence-verified | S3 evidence + SNS |
| `log_and_monitor` | intrusion **not** verified | S3 evidence |

Every track that entered the zone must be resolved exactly once.

## Storage

* `s3://<evidence-bucket>/evidence/<run_id>/<track>-<action>.jpg`: expires after 30 days.
* `s3://<evidence-bucket>/decisions/<run_id>/<track>.json`: supervisor answers, written
  with `IfNoneMatch="*"`; expires after 90 days.
* DynamoDB `pk = "<runs|bedrock>#YYYY-MM-DD"`, `runs` (counter), `expires_at` (TTL).

## Metrics (CloudWatch EMF, namespace `EdgeSentry`)

Dimensions `Service=SafetyCheck`, `DecisionMode`. Metrics: `PerceptionMs`, `AgentMs`,
`AgentSteps`, `GuardrailBlocks`, `EStopAlerts`, `DecisionAccuracy`, `FallbackToPolicy`.
