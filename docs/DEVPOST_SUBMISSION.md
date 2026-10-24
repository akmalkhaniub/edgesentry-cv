# Devpost submission text: EdgeSentry

Copy-paste source for the Devpost form. Every number links back to a file in the repo.

**Project name:** EdgeSentry

**Tagline:** A robot-cell safety agent: OpenCV 5 sees who is really in the hazard zone,
Amazon Bedrock decides, and code guardrails make sure it never stops the line without proof.

**Category / award:** OpenCV AI Competition 2026 main track + Agentic Vision Award

**Links**
- Live app: https://jzyv7mljhphuteckul2lxqe77u0xbqev.lambda-url.us-east-1.on.aws/
- Code: https://github.com/akmalkhaniub/edgesentry-cv
- Technical report: https://github.com/akmalkhaniub/edgesentry-cv/blob/main/docs/TECHNICAL_REPORT.md

---

## Inspiration

Camera "person detected" alerts around industrial robots get switched off. They fire when
someone brushes the boundary of the cell, or when a vest falls into shadow, and every false
stop costs production. They also treat a technician doing planned maintenance the same as an
unprotected worker wandering in. We wanted a camera system that behaves like a careful
safety lead: look again before stopping anything, ask a person when an entry looks
authorised, and show the evidence for every call.

## What it does

EdgeSentry analyses the robot-cell camera and, for every person who entered the hazard zone,
takes one of three actions:

- **Protective stop + alert**: only for an intrusion that persisted 1.5 s **and** where a
  close-up re-inspection confirmed no hi-vis vest.
- **Supervisor approval**: verified entries by people wearing PPE. The supervisor answers
  "approve entry" or "stop the robot" from the web app.
- **Log and keep monitoring**: brief boundary-brushing.

Each action stores an evidence frame in S3 and, for stops and approvals, publishes an SNS
alert. The web app leads with the outcome, then shows the evidence, the annotated clip, every
step the agent took, and a check against ground truth.

## How we built it

- **OpenCV 5 perception (all classical, no trained model).** The camera is registered to the
  empty-cell reference with `phaseCorrelate` + `findTransformECC` (foreground-masked,
  accepted only if `computeECC` improves). Lighting is compensated, and the difference image
  is misalignment-tolerant (`erode`/`dilate` plate envelopes). Then Otsu, morphology,
  `connectedComponentsWithStats`, head-profile splitting of groups, centroid tracking,
  `pointPolygonTest` on foot points, and a persistence filter. PPE uses HSV `inRange` in the
  wide view. The close-up tool upscales 2×, tests chromaticity only, and measures the largest
  connected band.
- **Agent on Amazon Bedrock.** Amazon Nova Pro via the Converse API with tool use. The agent
  never sees pixels; it calls OpenCV tools (`get_scene_report`, `inspect_worker`) and action
  tools. Guardrails in code veto unproven stops, unverified approvals, and logging a verified
  intrusion. A controller sends the agent back when it leaves anyone unresolved. If Bedrock is
  unavailable, a deterministic policy runs the same tools under the same guardrails, and the UI
  says so.
- **AWS.** One Lambda (Python 3.12) behind a Function URL serves the UI and API. S3 holds
  evidence (private, encrypted, expiring), SNS sends alerts (KMS), DynamoDB caps public usage
  per day, and CloudWatch handles EMF metrics, alarms, a dashboard and X-Ray. All of it is one
  CloudFormation template and a deploy script.

## Challenges we ran into

- Our evaluation sweep broke our first detector: a 3 px camera bump made every edge
  foreground (0 of 4 workers found). Fixing it led to phase correlation + ECC registration and
  envelope differencing.
- Three bright vests pulled ECC to a false 0.9° rotation. We fixed it by masking foreground
  out of the alignment and keeping the result only if it beats identity.
- We found that OpenCV 5.0.0's `phaseCorrelate` windows its inputs **in place**, silently
  erasing our stored reference. We pass copies, and a regression test pins it.
- The language model reliably skipped the "boring" compliant worker, and sometimes tried to
  just log a verified intrusion. Prompting alone did not fix it; the controller and guardrails
  did.

## Accomplishments we're proud of

- 10/10 live Bedrock agent runs fully correct, with 0 false and 0 missed stops. The guardrails
  blocked 4 unsafe actions along the way, and 9 runs needed the controller.
- 16 of 17 robustness variants fully correct, including camera shift, 1° rotation, ±30 and
  +80 lighting, two and three workers shoulder to shoulder, and a kneeling worker. The one
  failure fails safe.
- Ablations quantify each piece: without the close-up the shadowed contractor gets a false
  stop; without the robustness steps the system misses stops in 6 variants.
- About 49 fps OpenCV perception at 1280×720 inside Lambda.

## What we learned

The interesting engineering in an "AI agent" was everything around the model: tools that
expose measured evidence, guardrails that cannot be argued with, a controller that checks
completeness, a summary generated from what actually happened, and an evaluation that keeps
finding the next failure.

## What's next

Run perception continuously on an edge device with the agent as the escalation layer, test on
real cell footage with plate re-capture, add depth for occlusion at the feet, and integrate
supervisor approvals with permit-to-work systems.

## Built with

opencv · python · numpy · amazon-bedrock · amazon-nova · aws-lambda · amazon-s3 · amazon-sns ·
amazon-dynamodb · amazon-cloudwatch · aws-x-ray · aws-cloudformation · html · javascript
