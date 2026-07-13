#!/usr/bin/env bash
set -euo pipefail

REGION="${AWS_REGION:?AWS_REGION is required}"
IMAGE_URI="${IMAGE_URI:?IMAGE_URI is required}"
JOB_S3_URI="${JOB_S3_URI:?JOB_S3_URI is required}"
MODEL_BUCKET="${MODEL_BUCKET:?MODEL_BUCKET is required}"
INSTANCE_TYPE="${INSTANCE_TYPE:?INSTANCE_TYPE is required}"
INSTANCE_HOURLY_USD="${INSTANCE_HOURLY_USD:?INSTANCE_HOURLY_USD is required}"
MAX_JOB_SECONDS="${MAX_JOB_SECONDS:-2400}"
IDLE_SECONDS="${IDLE_SECONDS:-900}"
WORK_ROOT=/var/lib/hyworld
MODEL_ROOT=/models/hf-cache
LOG_FILE=/var/log/hyworld-portable-runner.log

exec > >(tee -a "$LOG_FILE") 2>&1
mkdir -p "$WORK_ROOT/scenes" "$MODEL_ROOT"

instance_id() {
  TOKEN="$(curl -fsS -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' \
    http://169.254.169.254/latest/api/token)"
  curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id
}

terminate_self() {
  aws s3 cp "$LOG_FILE" \
    "s3://intelliverse-hyworld-private-us-east-1/worldgen-full-ops/portable/$(instance_id).log" \
    --region us-east-1 --only-show-errors || true
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$(instance_id)" >/dev/null || true
}
trap terminate_self EXIT

# The hard deadline is independent of worker health and bounds p4de compute below
# one hour. Normal completion still waits the required 15-minute idle period.
(sleep 3600; terminate_self) &

dnf install -y docker awscli jq socat
systemctl enable --now docker
nvidia-smi
df -h /

aws s3 sync "s3://${MODEL_BUCKET}/models/hy-world/hf" "$MODEL_ROOT" \
  --region "$REGION" --only-show-errors \
  --exclude 'hub/*/blobs/*' --exclude 'hub/.locks/*'
aws s3 cp "$JOB_S3_URI" "$WORK_ROOT/job.json" --region us-east-1 --only-show-errors

SECRET="$(aws secretsmanager get-secret-value --region us-west-2 \
  --secret-id hy-world/portable-runner --query SecretString --output text)"
VLM_API_KEY="$(jq -r .VLM_API_KEY <<<"$SECRET")"
DISCORD_WEBHOOK="$(jq -r .DISCORD_WEBHOOK <<<"$SECRET")"
unset SECRET

aws ecr get-login-password --region "$REGION" |
  docker login --username AWS --password-stdin \
    "970547373533.dkr.ecr.${REGION}.amazonaws.com"
docker pull "$IMAGE_URI"
docker run -d --name redis --restart unless-stopped --network host redis:7-alpine

# Translate local cleartext OpenAI-compatible calls to the public TLS gateway
# without exposing a new cluster endpoint.
socat TCP-LISTEN:4000,reuseaddr,fork \
  OPENSSL:litellm.intelli-verse-x.ai:443,verify=1,snihost=litellm.intelli-verse-x.ai &

aws s3 cp "$WORK_ROOT/job.json" \
  "s3://intelliverse-hyworld-private-us-east-1/worldgen-full-ops/portable/jobs/$(jq -r .jobId "$WORK_ROOT/job.json").json.tmp" \
  --region us-east-1 --only-show-errors
aws s3 cp "$WORK_ROOT/job.json" \
  "s3://intelliverse-hyworld-private-us-east-1/worldgen-full-ops/portable/jobs/$(jq -r .jobId "$WORK_ROOT/job.json").json" \
  --region us-east-1 --only-show-errors

docker run --rm --gpus all --shm-size 32g --network host \
  -v "$MODEL_ROOT:/models/hf-cache" \
  -e HF_HOME=/models/hf-cache -e HF_HUB_OFFLINE=1 \
  "$IMAGE_URI" bash -lc '
    nvidia-smi
    test -f /app/hyworld2/worldgen/traj_generate.py
    test -f /app/hyworld2/worldgen/video_gen.py
    test -f /app/hyworld2/worldgen/world_gs_trainer.py
    cd /app/hyworld2/panogen
    python /app/deploy/worldgen/run_pano.py --preflight-only
  '

docker run -d --name worldgen --gpus all --shm-size 32g --network host \
  -v "$MODEL_ROOT:/models/hf-cache" \
  -v "$WORK_ROOT/scenes:/workspace/scenes" \
  -e REDIS_HOST=127.0.0.1 -e REDIS_PORT=6379 \
  -e AWS_REGION=us-east-1 \
  -e AWS_S3_BUCKET_NAME=intelliverse-hyworld-private-us-east-1 \
  -e S3_CHECKPOINT_BASE=worldgen-full-checkpoints \
  -e S3_OUTPUT_BASE=worldgen-full-staging \
  -e WORLDGEN_QUEUE=pipeline:signal:worldgen-full \
  -e WORLDGEN_PROCESSING_QUEUE=pipeline:worldgen-full:processing \
  -e WORLDGEN_DONE_QUEUE=pipeline:done:worldgen-full \
  -e LLM_ADDR=127.0.0.1 -e LLM_PORT=4000 \
  -e VLM_API_KEY="$VLM_API_KEY" -e DISCORD_WEBHOOK="$DISCORD_WEBHOOK" \
  -e SAM3_REPO_ID=DiffusionWave/sam3 \
  -e SAM_BOX_REPO_ID=facebook/sam-vit-base \
  -e HF_HOME=/models/hf-cache -e HF_HUB_OFFLINE=1 \
  -e NGPU=1 -e SOURCE_COMMIT=9868083 \
  -e IMAGE_URI="$IMAGE_URI" -e INSTANCE_TYPE="$INSTANCE_TYPE" \
  -e INSTANCE_HOURLY_USD="$INSTANCE_HOURLY_USD" \
  -e MAX_JOB_SECONDS="$MAX_JOB_SECONDS" \
  -e ALLOW_PRODUCTION_PROMOTION=0 \
  "$IMAGE_URI"

docker exec redis redis-cli RPUSH pipeline:signal:worldgen-full \
  "$(jq -c . "$WORK_ROOT/job.json")"

while [[ "$(docker exec redis redis-cli LLEN pipeline:done:worldgen-full)" == "0" ]]; do
  docker ps --filter name=worldgen --format '{{.Status}}'
  sleep 30
done
docker exec redis redis-cli RPOP pipeline:done:worldgen-full |
  tee "$WORK_ROOT/result.json"
aws s3 cp "$WORK_ROOT/result.json" \
  "s3://intelliverse-hyworld-private-us-east-1/worldgen-full-ops/portable/results/$(jq -r .jobId "$WORK_ROOT/job.json").json" \
  --region us-east-1 --only-show-errors
sleep "$IDLE_SECONDS"
