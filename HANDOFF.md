# HANDOFF — worldgen pipeline (builder agent -> harness engineer)

Date: 2026-07-11 ~19:45 CDT. Builder agent standing down; harness engineer is
the single owner of the worldgen pipeline from here.

## Branches in intelli-verse-x/HY-World-2.0

- `worldgen/pipeline-build-v1` (this branch, = `worldgen-pipeline` + the final
  CPU-request tweaks): full 5-stage pipeline build. Everything lives under
  `deploy/worldgen/` — `worker.py` (Redis BRPOPLPUSH worker,
  `pipeline:signal:worldgen` -> `pipeline:worldgen:processing`, heartbeat +
  stale-job resume, S3 upload + manifest), `run_pano.py` (Qwen-Image-Edit +
  HY-Pano LoRA, CPU-offloaded), `Dockerfile` (torch 2.7.1/cu128, gsplat_
  maskgaussian + navmesh compiled for sm86/sm89, pytorch3d prebuilt wheel, no
  flash-attn -> SDPA fallback), `k8s/` manifests, `README.md` runbook.
  Root: `LICENSE-NOTE.md` (Tencent license: EU/UK/KR geo-block, <1M MAU).
- `worldgen-worker` (harness agent's own commit 584bef7, includes everything
  above plus its `deploy/` variant). Not mine — do not treat this handoff as
  authoritative for it.
- Upstream-code patches on both branches: `SAM3_REPO_ID` env override
  (facebook/sam3 is HF-gated, no HF token exists anywhere in the cluster;
  deployment uses ungated mirror `DiffusionWave/sam3`), `VLM_API_KEY` env for
  the OpenAI client (VLM routed through litellm + Gemini), `cupy` ->
  `cupy-cuda12x` in requirements.txt.

## Cluster state right now (namespace aicart unless noted)

- NodePool `gpu-workloads-multi` (cluster-scoped): 4-GPU spot,
  g6.12xlarge/g5.12xlarge, label `gpu-class: multi-24gb`, reuses
  EC2NodeClass `gpu-workloads`. **0 nodes provisioned — no GPU has ever
  started. $0 GPU spend so far.**
- PVC `scratch-mirror-hy-world` (300Gi gp3, Bound, AZ of node
  ip-10-1-8-50): HF-cache layout under subPath `hf/`.
- Job `hy-world-stage-weights`: **RUNNING**, ~116 GB of ~190 GB downloaded
  (Qwen-Image-Edit-2509 and HY-World-2.0 LoRA+WorldMirror done; currently on
  Wan2.1-I2V-14B-480P-Diffusers; then WorldStereo-DMD 35G + small aux
  models). Final step mirrors the PVC to
  `s3://intelli-verse-x-media/models/hy-world/hf` (aws s3 sync). Left
  running per coordination order (>50% done). Watch:
  `kubectl -n aicart logs -f job/hy-world-stage-weights`.
- Job `hy-world-image-build`: **DELETED** (was Pending, never scheduled, 0%).
  PVC `hy-world-build-ws` (150Gi gp3) and secret `hy-world-ecr-push` (ECR
  token, valid ~12 h from 19:35 CDT) still exist for reuse.
- **NOT applied**: `deployment.yaml` (hy-world-worker) and
  `scaledobject.yaml` (worldgen-scaler). The queue name
  `pipeline:signal:worldgen` is therefore not yet watched by anything.

## ECR / S3

- ECR repo `hy-world-worker` exists, **empty** (no image pushed yet).
- S3: only `models/hy-world/build/context.tar.gz` (814 KB build context for
  the BuildKit job, built from this branch). Weight mirror prefix
  `models/hy-world/hf/` will appear when the staging job finishes.

## Next steps I was about to do (in order)

1. Wait for staging job completion (ETA ~30-60 min incl. S3 sync).
2. Re-apply `deploy/worldgen/k8s/build-job.yaml` (BuildKit, ~60-90 min on a
   4-vCPU node; context + secret already in place). Verify
   `hy-world-worker:v1` lands in ECR.
3. Apply `deployment.yaml` + `scaledobject.yaml` (replicas stay 0).
4. Smoke test: LPUSH the library prompt (exact command in
   `deploy/worldgen/README.md`), monitor logs, verify artifacts under
   `s3://intelli-verse-x-media/worldgen/<jobId>/`, confirm scale-to-zero.

## Known issues / risks

- **Biggest risk**: video_gen (WorldStereo-2, Wan 14B base) under FSDP on
  4x24 GB is tight — transformer+UMT5+SAM3+MoGe per GPU leaves little
  headroom for activations. If OOM: try 8-GPU (g5.48xlarge spot ~$4.6/h in
  us-east-1c — needs adding to the nodepool), or reduce
  `--downsampled_pts` / resolution.
- The staging pod holds the RWO scratch PVC; the GPU worker cannot mount it
  until that job finishes. Karpenter will also pin the GPU node to the PVC's
  AZ (us-east-1a/b — check `kubectl get pv` topology before assuming 1c spot
  prices).
- CPU requests >2 with limits >3500m won't schedule on the `general`
  nodepool (4-vCPU nodes); that's why the jobs were tuned down.
- litellm image endpoint verified working (gemini image gen returns b64);
  worker uses `LITELLM_MASTER_KEY` from secret `litellm-secrets`.
- ~$8-9 of NAT data-processing cost is inherent to the 190 GB HF download
  (one-time; S3 traffic itself uses the gateway endpoint, free).
