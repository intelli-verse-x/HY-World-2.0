#!/usr/bin/env bash
# Enqueue v14 converged recipe regens for production worlds.
# Requires: kubectl access to aicart, redis secret keda-redis-auth.
# Usage: ./regen-v14-worlds.sh [world-id ...]   (default: all 5)
set -euo pipefail

REDIS_HOST="${REDIS_HOST:-content-factory-redis.aicart.svc.cluster.local}"
NAMESPACE="${NAMESPACE:-aicart}"

# Locked v14 params (matches exp-gameshow-v14 / exp-nm-final staging winners).
V14_PARAMS='{
  "pano_steps": 60,
  "pano_true_cfg": 7.5,
  "pano_upscale": 4,
  "view_size": 1280,
  "recon_target_size": 1274,
  "gs_max_points": 3000000,
  "gs_max_points_hd": 8000000,
  "splat_scale_mult": 1.8,
  "two_pass": 1,
  "fill_floor_only": 1,
  "shell_fill": 1,
  "shell_bins": 360,
  "opacity_floor": 0.1,
  "dark_cull_lum": 22,
  "dark_cull_opac": 0.5,
  "pano_tonemap": 1,
  "pano_seam_blend": 48,
  "lum_clamp_ceiling": 245
}'

declare -A PROMPTS=(
  [world-nightmarket]="Narrow cyberpunk night-market alley after rain: glowing neon signs in pink, teal and orange, steaming food stalls with hanging lanterns, holographic advertisements, wet asphalt with vivid neon reflections, cables overhead, light fog, bustling intimate atmosphere, Blade Runner mood, cinematic photorealistic detail."
  [world-lodge]="Interior of a warm alpine lakeside lodge at dusk: crackling stone fireplace, leather armchairs in a circle, exposed timber beams, bookshelves, board games on a wooden table, fairy lights, panoramic window showing a misty lake and pine mountains under an orange-purple sunset, snug inviting hygge atmosphere, warm cinematic lighting, photorealistic."
  [world-gameshow]="Interior of a futuristic television game-show studio: circular stage with glowing podiums, tiered audience seating, giant LED wall displaying vibrant gradients, neon magenta and cyan rim lighting, polished reflective black floor, confetti cannons, dramatic spotlights cutting through light haze, glamorous exciting mood, photorealistic broadcast-quality."
  [world-manor]="Interior of a grand haunted Victorian manor hall: dark wood panelling, oil portraits in gilt frames, ornate chandelier with flickering candles, checkered marble floor, tall arched windows with storm clouds outside, velvet drapes, suffocating gothic darkness with warm candle accents, photorealistic architectural detail."
  [world-museum]="Interior of a majestic natural-history museum great hall: soaring vaulted glass ceiling, marble columns, central dinosaur skeleton on a stone plinth, ornate display cases with artifacts, warm golden afternoon light streaming in, brass railings, polished stone floor with subtle reflections, awe-inspiring scholarly mood, photorealistic architectural detail."
)

if [[ $# -eq 0 ]]; then
  WORLDS=(world-nightmarket world-lodge world-gameshow world-manor world-museum)
else
  WORLDS=("$@")
fi
REDIS_PW="$(kubectl -n "$NAMESPACE" get secret keda-redis-auth -o jsonpath='{.data.password}' | base64 -d)"

for jobId in "${WORLDS[@]}"; do
  prompt="${PROMPTS[$jobId]:-}"
  if [[ -z "$prompt" ]]; then echo "unknown world: $jobId" >&2; exit 1; fi
  payload=$(python3 - <<PY
import json
print(json.dumps({
  "jobId": "$jobId",
  "prompt": """$prompt""",
  "params": json.loads('''$V14_PARAMS'''),
}))
PY
)
  echo "LPUSH $jobId"
  kubectl -n "$NAMESPACE" run "enqueue-${jobId}-$(date +%s)" --rm -i --restart=Never \
    --image=public.ecr.aws/docker/library/redis:7 -- \
    redis-cli -h "$REDIS_HOST" -a "$REDIS_PW" LPUSH pipeline:signal:worldgen "$payload"
done

echo "Queued ${#WORLDS[@]} v14 regen job(s). Watch: kubectl -n $NAMESPACE logs -f deploy/hy-world-worker"
