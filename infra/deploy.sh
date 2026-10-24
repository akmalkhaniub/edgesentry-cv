#!/usr/bin/env bash
# Build the Lambda package (Linux wheels, pinned) and deploy the EdgeSentry stack.
#
#   AWS_REGION=us-east-1 ALERT_EMAIL=you@example.com ./infra/deploy.sh
#
# Needs: AWS CLI v2 with credentials, Python 3.12+ with pip. No Docker/SAM/CDK required.
set -euo pipefail
cd "$(dirname "$0")/.."

STACK="${STACK:-edgesentry}"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
ARTIFACTS="${ARTIFACT_BUCKET:-edgesentry-artifacts-${ACCOUNT}-${REGION}}"
BUILD=".build/lambda"
ZIP=".build/edgesentry-lambda.zip"

echo "==> Building Lambda package (python3.12, manylinux x86_64 wheels)"
rm -rf .build && mkdir -p "$BUILD"
python -m pip install --quiet --disable-pip-version-check \
  --platform manylinux_2_28_x86_64 --implementation cp --python-version 3.12 \
  --only-binary=:all: --no-warn-conflicts --target "$BUILD" -r requirements-lambda.txt
python - "$BUILD" "$ZIP" <<'PY'
import pathlib, shutil, sys, zipfile
build, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
shutil.copy("lambda_function.py", build / "lambda_function.py")
shutil.copytree("edgesentry", build / "edgesentry", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)
skip = ("__pycache__", ".dist-info/RECORD", "/tests/", "numpy/f2py/", "numpy/_pyinstaller/")  # not needed at runtime
size = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted(build.rglob("*")):
        rel = p.relative_to(build).as_posix()
        if p.is_file() and not any(s in rel for s in skip) and not rel.endswith((".pyc", ".pyi")):
            z.write(p, rel)
            size += p.stat().st_size
print(f"    unzipped {size / 1e6:.1f} MB (Lambda limit 250 MB), zip {out.stat().st_size / 1e6:.1f} MB")
if size > 250e6:
    sys.exit("package too large for Lambda")
PY

HASH="$(python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:16])" "$ZIP")"
KEY="edgesentry/edgesentry-lambda-${HASH}.zip"

echo "==> Artifact bucket s3://${ARTIFACTS}"
if ! aws s3api head-bucket --bucket "$ARTIFACTS" --region "$REGION" 2>/dev/null; then
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$ARTIFACTS" --region "$REGION" >/dev/null
  else
    aws s3api create-bucket --bucket "$ARTIFACTS" --region "$REGION" \
      --create-bucket-configuration "LocationConstraint=${REGION}" >/dev/null
  fi
  aws s3api put-public-access-block --bucket "$ARTIFACTS" --region "$REGION" \
    --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  aws s3api put-bucket-encryption --bucket "$ARTIFACTS" --region "$REGION" \
    --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
fi
aws s3 cp "$ZIP" "s3://${ARTIFACTS}/${KEY}" --region "$REGION" --only-show-errors

echo "==> Deploying stack ${STACK} in ${REGION}"
aws cloudformation deploy --region "$REGION" --stack-name "$STACK" \
  --template-file infra/template.yaml --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset \
  --parameter-overrides "CodeBucket=${ARTIFACTS}" "CodeKey=${KEY}" "AlertEmail=${ALERT_EMAIL:-}" \
  --tags project=edgesentry

URL="$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='AppUrl'].OutputValue" --output text)"
echo "==> Smoke test ${URL}api/health"
curl -fsS "${URL}api/health"; echo
echo "Done. Open ${URL}"
