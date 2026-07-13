#!/usr/bin/env bash
# Queue two hero candidates through the full HY-World 2.0 stack.
# Maximum planned GPU spend: 2 * 4.5h * $4.60/h = $41.40.
set -euo pipefail

NAMESPACE="${NAMESPACE:-aicart}"
QUEUE="${WORLDGEN_QUEUE:-pipeline:signal:worldgen-full}"
REDIS_HOST="${REDIS_HOST:-content-factory-redis.aicart.svc.cluster.local}"
REDIS_PW="$(kubectl -n "$NAMESPACE" get secret keda-redis-auth -o jsonpath='{.data.password}' | base64 -d)"

enqueue() {
  local job_id="$1" seed="$2" prompt="$3"
  local payload
  payload="$(JOB_ID="$job_id" SEED="$seed" PROMPT="$prompt" python3 - <<'PY'
import json, os

landmarks = [
    {"id": "nm-alley", "name": "Night market alley", "promptPhrase": "night market alley", "checkpointId": "cp-2"},
    {"id": "nm-stalls", "name": "Food stalls", "promptPhrase": "food stalls", "checkpointId": "cp-3"},
    {"id": "nm-ice-lantern", "name": "ICE-blue hanging lantern", "promptPhrase": "ICE-blue hanging lantern", "checkpointId": "cp-1"},
    {"id": "nm-neon", "name": "Neon shop signs", "promptPhrase": "neon shop signs", "checkpointId": "cp-4"},
    {"id": "nm-paving", "name": "Wet stone paving", "promptPhrase": "wet stone paving", "checkpointId": "cp-5"},
]
print(json.dumps({
    "jobId": os.environ["JOB_ID"],
    "prompt": os.environ["PROMPT"],
    "seed": int(os.environ["SEED"]),
    "style": "",
    "sceneType": "indoor",
    "outputS3Prefix": f"worldgen-full-staging/{os.environ['JOB_ID']}",
    "budgetUsd": 25.0,
    "landmarks": landmarks,
}))
PY
)"
  kubectl -n "$NAMESPACE" run "enqueue-${job_id}-$(date +%s)" --rm -i --restart=Never \
    --image=public.ecr.aws/docker/library/redis:7 -- \
    redis-cli -h "$REDIS_HOST" -a "$REDIS_PW" --no-auth-warning RPUSH "$QUEUE" "$payload"
}

PROMPT_A='Night market alley, food stalls, ICE-blue hanging lantern, neon shop signs, and wet stone paving define a dense cyberpunk Asian market at blue hour. Eye-level central arrival, continuous covered canopy and floor, coherent close-range architecture, vendor silhouettes behind counters, controlled teal magenta orange practical lighting, readable materials. No haze, bloom, clipped whites, floating debris, text artifacts, or foreground obstruction. Photorealistic.'
PROMPT_B='Night market alley, food stalls, ICE-blue hanging lantern, neon shop signs, and wet stone paving form an intimate rainy market corridor. Eye-level central arrival, continuous fabric canopy and detailed floor, layered storefront geometry, visible vendors, balanced cyan pink amber lighting, crisp signs and surfaces. No fog, glare, clipped whites, black gaps, floating objects, malformed text, or foreground obstruction. Photorealistic.'

enqueue full-nm-a 4201 "$PROMPT_A"
enqueue full-nm-b 7301 "$PROMPT_B"
echo "Queued two full-stack Night Market candidates; planned cap: \$41.40 spot GPU."
