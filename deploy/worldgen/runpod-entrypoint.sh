#!/usr/bin/env bash
# RunPod pod bootstrap for the HY-World 2.0 full four-stage worker.
#
# This runs INSIDE the RunPod pod as the container start command. The pod image
# is the same private hy-world-full-worker image used on EC2, so /app already
# contains the pipeline code and Python env. This script only wires up the
# runtime the worker expects (Redis, LiteLLM reachability, weights) and adds a
# self-contained lifecycle watchdog so a pod can never bill indefinitely even if
# the control plane that launched it disappears.
#
# It is intentionally credential-AGNOSTIC: it uses the standard AWS credential
# chain from the environment. The control-plane launcher is responsible for
# injecting short-lived, least-privilege AWS credentials (see
# docs/runpod-portable-runner.md). This script never prints secret values.
set -uo pipefail

# ---- FAILSAFE FIRST: arm an absolute self-terminate before anything that can
# fail, so the pod can never bill indefinitely even if setup crashes early. ----
FAILSAFE_DEADLINE_SECONDS="${HARD_DEADLINE_SECONDS:-14400}"
runpod_terminate_self() {
  local pid="${RUNPOD_POD_ID:-}"
  echo "[watchdog] terminating pod ${pid:-<self>}"
  for _ in 1 2 3; do
    if [[ -n "${RUNPOD_API_KEY:-}" && -n "$pid" ]]; then
      curl -fsS -X DELETE "https://rest.runpod.io/v1/pods/${pid}" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" >/dev/null 2>&1 && exit 0
      # python fallback (curl may be absent in the base image)
      python3 -c 'import os,urllib.request as u;u.urlopen(u.Request("https://rest.runpod.io/v1/pods/"+os.environ["RUNPOD_POD_ID"],method="DELETE",headers={"Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"]}),timeout=20)' >/dev/null 2>&1 && exit 0
    fi
    runpodctl remove pod "$pid" >/dev/null 2>&1 && exit 0
    sleep 5
  done
  poweroff -f >/dev/null 2>&1 || kill -9 1 || true
}
( sleep "$FAILSAFE_DEADLINE_SECONDS"; echo "[watchdog] failsafe deadline"; runpod_terminate_self ) &
set -e

# ---- Required runtime inputs (injected by the launcher as pod env) ----
: "${AWS_REGION:?AWS_REGION required}"
: "${MODEL_BUCKET:?MODEL_BUCKET required}"                 # e.g. intelliverse-hyworld-private-us-east-1
: "${MODEL_PREFIX:?MODEL_PREFIX required}"                 # e.g. models/hy-world/hf
: "${JOB_JSON_B64:?JOB_JSON_B64 required}"                 # base64 of the worldgen job payload
: "${INSTANCE_TYPE:?INSTANCE_TYPE required}"               # e.g. runpod-h100-80gb-sxm
: "${INSTANCE_HOURLY_USD:?INSTANCE_HOURLY_USD required}"   # verified pod rate for cost accounting
# AWS creds themselves arrive via AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
# AWS_SESSION_TOKEN in the environment (least-privilege, short-lived).
# VLM_API_KEY, DISCORD_WEBHOOK, RUNPOD_API_KEY are optional secrets in env.

MODEL_ROOT="${MODEL_ROOT:-/models/hf-cache}"
WORK_ROOT="${WORK_ROOT:-/workspace}"
IDLE_SECONDS="${IDLE_SECONDS:-900}"                        # 15-minute idle shutdown
HARD_DEADLINE_SECONDS="${HARD_DEADLINE_SECONDS:-14400}"    # absolute wall-clock cap (default 4h)
LLM_BRIDGE_PORT="${LLM_BRIDGE_PORT:-4000}"
LITELLM_HOST="${LITELLM_HOST:-litellm.intelli-verse-x.ai}"
LOG_FILE="${LOG_FILE:-/var/log/hyworld-runpod-runner.log}"
mkdir -p "$MODEL_ROOT" "$WORK_ROOT/scenes" "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

# Ship the full container log to S3 every 20s (and it flushes on exit via the
# next tick before self-terminate). This is the only way to read stage stderr,
# since RunPod pod logs are not exposed over the REST API.
JOB_LOG_NAME="$(printf '%s' "$JOB_JSON_B64" | base64 -d | python3 -c 'import sys,json;print(json.load(sys.stdin)["jobId"])' 2>/dev/null || echo runpod)"
LOG_S3="s3://${MODEL_BUCKET}/worldgen-full-ops/portable/logs/${JOB_LOG_NAME}.log"
( while true; do
    aws s3 cp "$LOG_FILE" "$LOG_S3" --region "$AWS_REGION" --only-show-errors 2>/dev/null || true
    sleep 20
  done ) &

command -v aws >/dev/null 2>&1 || pip install --no-cache-dir awscli >/dev/null 2>&1 || true
command -v redis-server >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq redis-server socat >/dev/null 2>&1) || true
command -v socat >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq socat >/dev/null 2>&1) || true

nvidia-smi || { echo "[fatal] no GPU visible"; runpod_terminate_self; exit 1; }
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
echo "[preflight] GPUs=$GPU_COUNT"

# Sync the license-controlled private weights cache. Raise concurrency so the
# ~196 GB pull saturates the pod NIC instead of the default 10 streams. Skip if
# a mounted volume already has the cache staged (idempotent reuse).
aws configure set default.s3.max_concurrent_requests 40
aws configure set default.s3.max_queue_size 20000
STAMP="$MODEL_ROOT/.sync-complete"
if [[ -f "$STAMP" ]]; then
  echo "[preflight] weights already staged on mounted volume; skipping sync"
else
  echo "[preflight] syncing weights from s3://${MODEL_BUCKET}/${MODEL_PREFIX} (t0=$(date -u +%T))"
  aws s3 sync "s3://${MODEL_BUCKET}/${MODEL_PREFIX}" "$MODEL_ROOT" \
    --region "$AWS_REGION" --only-show-errors
  echo "[preflight] weights synced (t1=$(date -u +%T))"
  touch "$STAMP"
fi

# WorldNav's traj_generate.py resolves SAM/GroundingDINO/SAM3/MoGe via
# snapshot_download(cache_dir=~/.cache/huggingface/hub), which ignores HF_HOME.
# Point that hardcoded cache dir at the synced cache so offline resolution finds
# refs/main + snapshots for every resolve_hf_checkpoint() model.
mkdir -p "$HOME/.cache/huggingface"
ln -sfn "$MODEL_ROOT/hub" "$HOME/.cache/huggingface/hub"

# Local Redis for the worker queue semantics.
redis-server --daemonize yes --save '' --appendonly no
for i in $(seq 1 30); do redis-cli ping >/dev/null 2>&1 && break; sleep 1; done

# Cleartext -> TLS bridge so the worker's plain OpenAI-compatible client can
# reach the public LiteLLM gateway without embedding a new endpoint.
socat "TCP-LISTEN:${LLM_BRIDGE_PORT},reuseaddr,fork" \
  "OPENSSL:${LITELLM_HOST}:443,verify=1,snihost=${LITELLM_HOST}" &

# Model-load preflight (fail fast before enqueuing a paid job).
cd /app/hyworld2/panogen
HF_HOME="$MODEL_ROOT" HF_HUB_OFFLINE=1 \
  python /app/deploy/worldgen/run_pano.py --preflight-only || {
    echo "[fatal] model-load preflight failed"; runpod_terminate_self; exit 1; }

# Launch the worker (consumes Redis queue on 127.0.0.1).
export REDIS_HOST=127.0.0.1 REDIS_PORT=6379
export WORLDGEN_QUEUE=pipeline:signal:worldgen-full
export WORLDGEN_PROCESSING_QUEUE=pipeline:worldgen-full:processing
export WORLDGEN_DONE_QUEUE=pipeline:done:worldgen-full
export AWS_S3_BUCKET_NAME="$MODEL_BUCKET"
export S3_CHECKPOINT_BASE=worldgen-full-checkpoints
export S3_OUTPUT_BASE=worldgen-full-staging
export LLM_ADDR=127.0.0.1 LLM_PORT="$LLM_BRIDGE_PORT"
export SAM_BOX_REPO_ID=facebook/sam-vit-base
# The cache holds DiffusionWave/sam3 (ungated mirror); the code default
# facebook/sam3 is gated and absent, so pin the mirror to match the cache.
export SAM3_REPO_ID=DiffusionWave/sam3
export HF_HOME="$MODEL_ROOT" HF_HUB_OFFLINE=1
export NGPU="$GPU_COUNT"
export ALLOW_PRODUCTION_PROMOTION=0
export SCENES_DIR="$WORK_ROOT/scenes"
python /app/deploy/worldgen/worker.py &
WORKER_PID=$!

# Enqueue the hero job once the worker is up.
JOB_JSON="$(printf '%s' "$JOB_JSON_B64" | base64 -d)"
JOB_ID="$(printf '%s' "$JOB_JSON" | python -c 'import sys,json;print(json.load(sys.stdin)["jobId"])')"
redis-cli RPUSH pipeline:signal:worldgen-full "$JOB_JSON" >/dev/null
echo "[run] enqueued job $JOB_ID"

# Wait for completion, bounded by the hard deadline watchdog above.
while [[ "$(redis-cli LLEN pipeline:done:worldgen-full)" == "0" ]]; do
  kill -0 "$WORKER_PID" 2>/dev/null || { echo "[fatal] worker exited"; runpod_terminate_self; exit 1; }
  sleep 30
done
redis-cli RPOP pipeline:done:worldgen-full | tee "$WORK_ROOT/result.json"

echo "[done] job complete; entering ${IDLE_SECONDS}s idle window before self-terminate"
sleep "$IDLE_SECONDS"
runpod_terminate_self
