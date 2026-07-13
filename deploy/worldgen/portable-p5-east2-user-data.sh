#!/usr/bin/env bash
set -euo pipefail

export AWS_REGION=us-east-2
export IMAGE_URI=970547373533.dkr.ecr.us-east-2.amazonaws.com/hy-world-full-worker@sha256:24ad1ae4b0ae26722d710efdb6c1602268c45d40230a7b5a2c96c952311829b0
export JOB_S3_URI=s3://intelliverse-hyworld-private-us-east-1/worldgen-full-ops/portable/jobs/full-nm-a-hourly-input.json
export MODEL_BUCKET=intelliverse-hyworld-private-us-east-1
export MODEL_BUCKET_REGION=us-east-1
export INSTANCE_TYPE=p5.48xlarge
export INSTANCE_HOURLY_USD=55.04
export MAX_JOB_SECONDS=2400
export IDLE_SECONDS=900
export HARD_DEADLINE_SECONDS=3450
export GPU_COUNT=8

aws s3 cp \
  s3://intelliverse-hyworld-private-us-east-1/worldgen-full-ops/portable/scripts/portable-runner-hourly.sh \
  /tmp/portable-runner.sh --region us-east-1 --only-show-errors
exec bash /tmp/portable-runner.sh
