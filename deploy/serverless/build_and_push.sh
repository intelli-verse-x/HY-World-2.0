#!/usr/bin/env bash
# Build the baked-weights serverless image and push to private ECR.
#
# The 4090 light tier bakes ~5.8 GB of ungated vision weights (see
# weights-bake-manifest.json) into image layers so the serverless worker
# cold-starts from local disk. Run from the repo root.
#
# NOTE: building CUDA extensions (gsplat) is slow on arm64/macOS. Prefer an
# in-cluster BuildKit / x86 builder (same pattern as k8s/build-job.yaml).
set -euo pipefail

ACCOUNT="${ACCOUNT:-970547373533}"
REGION="${AWS_REGION:-us-east-1}"
REPO="${REPO:-hy-world-serverless-4090}"
TAG="${TAG:-light-$(date -u +%Y%m%d)}"
IMAGE="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}"

echo "[build] target: ${IMAGE}"

aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$REGION" \
       --image-scanning-configuration scanOnPush=true >/dev/null

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

# Weights are ungated, so no HF_TOKEN build-arg is required for the default tier.
DOCKER_BUILDKIT=1 docker build \
  -f deploy/serverless/Dockerfile \
  -t "$IMAGE" \
  "${HF_TOKEN:+--build-arg HF_TOKEN=$HF_TOKEN}" \
  .

docker push "$IMAGE"

DIGEST="$(aws ecr describe-images --repository-name "$REPO" --region "$REGION" \
  --image-ids imageTag="$TAG" --query 'imageDetails[0].imageDigest' --output text)"
echo "[build] pushed ${REPO}@${DIGEST}"
echo "[build] deploy with:"
echo "  python deploy/serverless/deploy_endpoint.py --image ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}@${DIGEST}"
