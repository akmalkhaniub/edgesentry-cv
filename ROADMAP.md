# EdgeSentry roadmap

**Competition:** OpenCV AI Competition 2026 (powered by AWS). Final deadline October 26, 2026, 11:59 PM PT.

## Done (v2.0.0)

- [x] OpenCV 5 clean-plate perception with registration (phase correlation + ECC), lighting
      compensation, envelope differencing, group splitting, fail-safe shapes, tracking,
      zone test, persistence filter.
- [x] Close-up PPE re-inspection exposed as an agent tool.
- [x] Amazon Bedrock tool-using agent with code guardrails, controller, and fixed-policy fallback.
- [x] AWS deployment: Lambda + Function URL, S3 evidence, SNS alerts, DynamoDB usage caps,
      CloudWatch metrics, alarms and dashboard, X-Ray. One CloudFormation template, plus deploy
      and teardown scripts.
- [x] Web app: outcome first, evidence cards, annotated clip, agent steps, scenario check,
      supervisor approve/stop, light/dark.
- [x] Reproducible evaluation: robustness sweep, ablations, live agent trials, live-endpoint
      measurement.

## Next

- [ ] Test on real robot-cell footage, with plate re-capture when the empty scene changes.
- [ ] Continuous edge perception (camera-side), with the cloud agent as the escalation layer.
- [ ] Depth or multi-camera input for people occluded at the feet.
- [ ] Learned PPE classifier via `cv2.dnn` alongside the colour test, evaluated on real data.
- [ ] Permit-to-work integration for supervisor approvals; authenticated supervisor access.
