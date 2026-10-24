#!/usr/bin/env bash
# Delete the EdgeSentry stack and its data. Irreversible: evidence and decisions in S3 are removed.
#
#   AWS_REGION=us-east-1 ./infra/teardown.sh            # keeps the artifact bucket
#   DELETE_ARTIFACTS=1 AWS_REGION=us-east-1 ./infra/teardown.sh
set -euo pipefail

STACK="${STACK:-edgesentry}"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
ARTIFACTS="${ARTIFACT_BUCKET:-edgesentry-artifacts-${ACCOUNT}-${REGION}}"

BUCKET="$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='EvidenceBucketName'].OutputValue" --output text 2>/dev/null || true)"
if [ -n "$BUCKET" ] && [ "$BUCKET" != "None" ]; then
  echo "==> Emptying evidence bucket ${BUCKET}"
  aws s3 rm "s3://${BUCKET}" --recursive --region "$REGION" --only-show-errors
fi

echo "==> Deleting stack ${STACK}"
aws cloudformation delete-stack --region "$REGION" --stack-name "$STACK"
aws cloudformation wait stack-delete-complete --region "$REGION" --stack-name "$STACK"

if [ "${DELETE_ARTIFACTS:-0}" = "1" ]; then
  echo "==> Deleting artifact bucket ${ARTIFACTS}"
  aws s3 rb "s3://${ARTIFACTS}" --force --region "$REGION"
fi
echo "Done."
