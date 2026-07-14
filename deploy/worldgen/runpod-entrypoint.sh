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

# ---- Feasibility probe mode: run a bounded experiment instead of the worker. ----
# Assess higher-resolution native HY-Pano generation (VRAM/time/quality) before
# committing GPU to a full higher-res converged run. Skips the worker preflight,
# redis, socat, and all gs_train patches (irrelevant to a pano probe), then
# self-terminates. The probe script is delivered inline by the launcher.
if [[ -n "${PROBE_MODE:-}" ]]; then
  echo "[probe] probe mode: ${PROBE_MODE}"
  export HF_HOME="$MODEL_ROOT" HF_HUB_OFFLINE=1
  export MODEL_BUCKET AWS_REGION
  cd /app/hyworld2/panogen
  export PYTHONUNBUFFERED=1
  case "${PROBE_MODE}" in
    pano)    python -u /app/deploy/worldgen/pano_res_probe.py || echo "[probe] script errored" ;;
    tiledsr) python -u /app/deploy/worldgen/tiledsr_probe.py || echo "[probe] script errored" ;;
    *)       echo "[probe] unknown PROBE_MODE ${PROBE_MODE}" ;;
  esac
  aws s3 cp "$LOG_FILE" "$LOG_S3" --region "$AWS_REGION" --only-show-errors 2>/dev/null || true
  echo "[probe] done; self-terminating"
  runpod_terminate_self
  exit 0
fi

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

# Runtime patch for the baked image (commit predates the fix): traj_render.py
# does not accept --skip_exist (only traj_generate.py does). Strip the stray flag
# from the traj_render torchrun invocation. Idempotent; no-op once the rebuilt
# image ships the corrected worker.py.
sed -i 's/\*vlm_args, "--skip_exist",/*vlm_args,/' /app/deploy/worldgen/worker.py || true

# WorldStereo loads the Wan2.1 base transformer via diffusers from_pretrained
# without local_files_only, so its sharded-checkpoint resolver calls model_info()
# over the network and dies under HF_HUB_OFFLINE. Force local resolution.
WRAP=/app/hyworld2/worldgen/models/worldstereo_wrapper.py
if [[ -f "$WRAP" ]] && ! grep -q 'subfolder="transformer", local_files_only=True' "$WRAP"; then
  sed -i 's/subfolder="transformer",/subfolder="transformer", local_files_only=True,/g' "$WRAP" || true
fi
# AutoTokenizer(repo, subfolder="tokenizer") triggers an AutoConfig lookup that dies
# offline (this Wan diffusers repo has no top-level transformers config.json). Resolve
# the local snapshot dir and load the tokenizer from its local tokenizer/ path.
if [[ -f "$WRAP" ]]; then
python3 - "$WRAP" <<'PYEOF' || true
import sys
p=sys.argv[1]; s=open(p).read()
old='        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, subfolder="tokenizer", local_files_only=local_files_only)'
new='        tokenizer = AutoTokenizer.from_pretrained((__import__("huggingface_hub").snapshot_download(cfg.base_model, local_files_only=True) + "/tokenizer") if local_files_only else cfg.base_model, subfolder=("" if local_files_only else "tokenizer"), local_files_only=local_files_only)'
if old in s:
    open(p,"w").write(s.replace(old,new)); print("[patch] tokenizer local-path applied")
else:
    print("[patch] tokenizer patch skipped (already applied)")
PYEOF
fi

# video_gen.py threads its --local_files_only flag into WorldStereo.from_pretrained
# (tokenizer/vae/scheduler/text_encoder). The worker must set it so those loads stay
# offline; otherwise transformers does a live HEAD and dies under HF_HUB_OFFLINE.
if ! grep -q '"--fsdp", "--skip_exist", "--local_files_only"' /app/deploy/worldgen/worker.py; then
  sed -i 's/"--fsdp", "--skip_exist",/"--fsdp", "--skip_exist", "--local_files_only",/' /app/deploy/worldgen/worker.py || true
fi

# Make video_gen's post-stage landmark validation non-fatal in the baked image so a
# VLM-gateway 404 (or unmet internal contract) can't discard a completed WorldStereo
# generation; the independent visual gate judges landmarks.
python3 - <<'PYEOF' || true
p="/app/deploy/worldgen/worker.py"
import os
if os.path.isfile(p):
    s=open(p).read()
    old=('            landmark_validation = validate_landmark_visibility(scene_dir, landmarks)\n'
         '            checkpoint("video_gen", stage_start, {\n'
         '                "allFiveLandmarksVisible": landmark_validation["allFiveVisible"],\n'
         '                "landmarkMap": "scene/landmark-map.json",\n'
         '            })')
    new=('            try:\n'
         '                landmark_validation = validate_landmark_visibility(scene_dir, landmarks)\n'
         '            except Exception as exc:\n'
         '                logger.warning("landmark validation skipped (non-fatal): %s", exc)\n'
         '                landmark_validation = {"allFiveVisible": None}\n'
         '            checkpoint("video_gen", stage_start, {\n'
         '                "allFiveLandmarksVisible": landmark_validation.get("allFiveVisible"),\n'
         '                "landmarkMap": "scene/landmark-map.json",\n'
         '            })')
    if old in s:
        open(p,"w").write(s.replace(old,new)); print("[patch] landmark validation non-fatal")
    else:
        print("[patch] landmark validation patch skipped")
PYEOF

# Captioning fixes for the baked image: (1) drop the unsupported `seed` kwarg that
# Gemini/litellm rejects in get_traj_caption; (2) write a scene-prompt fallback caption
# when the VLM call fails so video_gen always finds traj_caption.json.
python3 - <<'PYEOF' || true
import os
vp="/app/hyworld2/worldgen/src/vlm_utils.py"
if os.path.isfile(vp):
    s=open(vp).read()
    old="        max_tokens=1024,  # Adjust max_tokens to keep it concise\n        temperature=0.1,\n        seed=1024\n    )"
    new="        max_tokens=1024,  # Adjust max_tokens to keep it concise\n        temperature=0.1,\n    )"
    if old in s:
        open(vp,"w").write(s.replace(old,new)); print("[patch] get_traj_caption seed removed")
    else:
        print("[patch] seed patch skipped")
tp="/app/hyworld2/worldgen/traj_render.py"
if os.path.isfile(tp):
    s=open(tp).read()
    old="    except Exception as e:\n        return render_path, False, str(e)"
    new=("    except Exception as e:\n"
         "        try:\n"
         "            import ast\n"
         "            scene_root = render_path.split('/render_results/')[0]\n"
         "            with open(os.path.join(scene_root, 'meta_info.json')) as mf:\n"
         "                raw_prompt = json.load(mf).get('prompt', '')\n"
         "            if isinstance(raw_prompt, str) and raw_prompt.strip().startswith('{'):\n"
         "                try:\n"
         "                    raw_prompt = ast.literal_eval(raw_prompt)\n"
         "                except Exception:\n"
         "                    pass\n"
         "            fallback = raw_prompt.get('text', '') if isinstance(raw_prompt, dict) else str(raw_prompt)\n"
         "            fallback = fallback.strip() or 'A continuous camera trajectory moving through the scene, revealing its architecture, lighting, and surrounding details.'\n"
         "            with open(output_path, 'w') as write:\n"
         "                json.dump({'prompt': fallback, 'fallback': True}, write, indent=2)\n"
         "        except Exception:\n"
         "            pass\n"
         "        return render_path, False, str(e)")
    if old in s:
        open(tp,"w").write(s.replace(old,new)); print("[patch] caption fallback applied")
    else:
        print("[patch] caption fallback skipped")
PYEOF

# world_gs_trainer keeps a viser viewer alive with time.sleep(1000000) unless
# --disable_viewer is passed, so gs_train saves the splat/ply/mesh then hangs forever
# and the stage never returns. Add --disable_viewer to the worker's gs_train command.
if ! grep -q '"--disable_video", "--disable_viewer"' /app/deploy/worldgen/worker.py; then
  sed -i 's/"--convert_to_spz", "--disable_video",/"--convert_to_spz", "--disable_video", "--disable_viewer",/' /app/deploy/worldgen/worker.py || true
fi

# The final success marker calls set_status("done", state="done", **result) but `result`
# already contains state="done" -> "multiple values for keyword argument 'state'". This
# fires AFTER all artifacts upload, so it mislabels a fully complete job as failed. Drop
# the redundant kwarg.
sed -i 's/set_status("done", state="done", \*\*result)/set_status("done", **result)/' /app/deploy/worldgen/worker.py || true

# finalize_landmark_mapping reads scene_dir/landmark-map.json, which is only written by
# validate_landmark_visibility — now non-fatal, so on a VLM-gateway error that file is
# absent and finalize crashes AFTER a fully trained/exported splat, blocking the upload.
# Make finalize non-fatal too so the gs_train checkpoint + artifact upload still happen.
python3 - <<'PYEOF' || true
p="/app/deploy/worldgen/worker.py"
import os
if os.path.isfile(p):
    s=open(p).read()
    old=('            final_landmarks = finalize_landmark_mapping(scene_dir, result_dir)\n'
         '            checkpoint("gs_train", stage_start, {\n'
         '                "allFiveLandmarksMapped": all(\n'
         '                    item.get("reconstructedRegion") for item in final_landmarks["landmarks"]\n'
         '                ),\n'
         '                "landmarkMap": "scene/landmark-map.json",\n'
         '            })')
    new=('            try:\n'
         '                final_landmarks = finalize_landmark_mapping(scene_dir, result_dir)\n'
         '                all_mapped = all(item.get("reconstructedRegion") for item in final_landmarks["landmarks"])\n'
         '            except Exception as exc:\n'
         '                logger.warning("landmark finalize skipped (non-fatal): %s", exc)\n'
         '                all_mapped = None\n'
         '            checkpoint("gs_train", stage_start, {\n'
         '                "allFiveLandmarksMapped": all_mapped,\n'
         '                "landmarkMap": "scene/landmark-map.json",\n'
         '            })')
    if old in s:
        open(p,"w").write(s.replace(old,new)); print("[patch] finalize_landmark_mapping non-fatal applied")
    else:
        print("[patch] finalize_landmark_mapping patch skipped (already applied)")
PYEOF

# The baked image's worker.py predates --disable_viewer, so after gs_train finishes it
# runs the trainer's `time.sleep(1000000)` viser-serve loop and never exits -> the stage
# hangs and artifacts (ply/spz/mesh) are never uploaded. Inject --disable_viewer so
# gs_train exits cleanly after saving and the worker uploads.
if ! grep -q '"--disable_video", "--disable_viewer"' /app/deploy/worldgen/worker.py; then
  sed -i 's/"--convert_to_spz", "--disable_video",/"--convert_to_spz", "--disable_video", "--disable_viewer",/' /app/deploy/worldgen/worker.py || true
fi

# gs_train post-mesh sync calls dist.barrier() at world_gs_trainer.py:1713, but `dist`
# is only imported inside the multi-rank branch -> UnboundLocalError on the single-GPU
# path (after training+mesh export fully complete). Import locally + guard on an
# initialized process group so single-GPU runs don't crash at the finish line.
python3 - <<'PYEOF' || true
p="/app/hyworld2/worldgen/world_gs_trainer.py"
import os
if os.path.isfile(p):
    s=open(p).read()
    old="                    dist.barrier()\n\n            # Turn Gradients into Sparse Tensor before running optimizer"
    new=("                    import torch.distributed as dist\n"
         "                    if dist.is_available() and dist.is_initialized():\n"
         "                        dist.barrier()\n\n"
         "            # Turn Gradients into Sparse Tensor before running optimizer")
    if old in s:
        open(p,"w").write(s.replace(old,new)); print("[patch] gs_train dist.barrier guard applied")
    else:
        print("[patch] gs_train dist.barrier guard skipped (already applied)")
PYEOF

# gsplat_maskgaussian is AOT-built in the image for sm86/sm89 only (cluster A10G/L4).
# On the H100 portable runner (sm90) the rasterizer's PTX JIT fails
# ("provided PTX was compiled with an unsupported toolchain"), which kills gs_train.
# Rebuild the CUDA extension for the pod's actual arch once, caching the built tree on
# the network volume so subsequent pods/worlds skip the ~5-8min compile.
GS_DIR=/app/hyworld2/worldgen/third_party/gsplat_maskgaussian
GS_CAP="$(python3 -c 'import torch;M,m=torch.cuda.get_device_capability();print(f"{M}{m}")' 2>/dev/null || echo '')"
if [[ -n "$GS_CAP" && -d "$GS_DIR" && "$GS_CAP" != "86" && "$GS_CAP" != "89" ]]; then
  GS_ARCH="${GS_CAP:0:1}.${GS_CAP:1}"
  GS_CACHE="${MODEL_ROOT%/hf-cache}/gsplat-sm${GS_CAP}"
  # NOTE: importing gsplat.cuda._backend is NOT a valid readiness check — the baked .so
  # loads fine but its embedded PTX (built +PTX for 8.9) only JIT-fails at kernel launch
  # on sm90 ("unsupported toolchain"). We must produce a NATIVE sm${GS_CAP} cubin, so
  # either restore a cached native build or force a rebuild. Never trust the .so import.
  if [[ -f "$GS_CACHE/.built" ]]; then
    echo "[gsplat] restoring cached native sm${GS_CAP} build from volume"
    cp -a "$GS_CACHE/pkg/." "$GS_DIR/" 2>/dev/null || true
  else
    echo "[gsplat] building native CUDA extension for sm${GS_CAP} (TORCH_CUDA_ARCH_LIST=${GS_ARCH})…"
    if ( cd "$GS_DIR" && TORCH_CUDA_ARCH_LIST="$GS_ARCH" MAX_JOBS=4 \
           pip install --no-cache-dir --no-build-isolation -e . --force-reinstall --no-deps ); then
      echo "[gsplat] rebuild OK; caching native sm${GS_CAP} build to volume"
      mkdir -p "$GS_CACHE/pkg" && cp -a "$GS_DIR/." "$GS_CACHE/pkg/" 2>/dev/null && touch "$GS_CACHE/.built" || true
    else
      echo "[gsplat] rebuild FAILED (gs_train will error)"
    fi
  fi

  # fused-ssim is another AOT CUDA extension (installed non-editable from git, no PTX
  # fallback) used by gs_train's SSIM loss -> "no kernel image available" on sm90.
  # Build a wheel for the pod arch once, cache it on the volume, reinstall from cache.
  FS_CACHE="${MODEL_ROOT%/hf-cache}/fused-ssim-sm${GS_CAP}"
  FS_GIT="git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5"
  if compgen -G "$FS_CACHE/*.whl" >/dev/null 2>&1; then
    echo "[fused-ssim] installing cached sm${GS_CAP} wheel from volume"
    pip install --no-cache-dir --force-reinstall --no-deps "$FS_CACHE"/*.whl || echo "[fused-ssim] cache install FAILED"
  else
    echo "[fused-ssim] building sm${GS_CAP} wheel (TORCH_CUDA_ARCH_LIST=8.9;${GS_ARCH})…"
    mkdir -p "$FS_CACHE"
    if TORCH_CUDA_ARCH_LIST="8.9;${GS_ARCH}" MAX_JOBS=4 \
         pip wheel --no-cache-dir --no-build-isolation --no-deps "$FS_GIT" -w "$FS_CACHE"; then
      pip install --no-cache-dir --force-reinstall --no-deps "$FS_CACHE"/*.whl \
        && echo "[fused-ssim] rebuild+cache OK" || echo "[fused-ssim] wheel install FAILED"
    else
      echo "[fused-ssim] wheel build FAILED (gs_train will error)"
    fi
  fi
fi

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
