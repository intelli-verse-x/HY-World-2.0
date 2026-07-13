# RunPod portable full-stack runner

US Secure Cloud path for the HY-World 2.0 full four-stage worker
(HY-Pano 2.0 -> WorldNav -> WorldStereo 2.0 -> multi-view 3DGS), used when AWS
does not have >=80 GB GPU capacity in an enabled US region.

Evidence captured 2026-07-13, AWS account `970547373533`.

## What is verified working

- **RunPod credential**: the SecureString `/intelliverse/worldgen/runpod-api-key`
  exists in `us-east-1` (SecureString, `alias/aws/ssm`, version 1). It is read
  only at runtime and never printed.
- **RunPod auth**: the key authenticates against the current REST API
  (`GET https://rest.runpod.io/v1/pods` -> `200`). The legacy GraphQL bearer
  scheme returns `403`; this is a REST key and the launcher uses REST only.
- **No lingering resources**: pods, network volumes, and endpoints are all `0`.
- **80 GB Secure Cloud US GPUs exist and fit budget**: A100-80 SXM (~$1.49/hr),
  A100-80 PCIe, H100-80 (~$2.69/hr), H200. Even a generous 15 GPU-hour, five-world
  run is well under the remaining `$416.46` cap; budget is not the binding
  constraint.

## Status 2026-07-13: UNBLOCKED and pipeline running

The IAM trust edit below was applied: `hy-world-portable-runner` now trusts
`user/s3-user` for `sts:AssumeRole` (see `iam/portable-runner-trust.json`). The
scoped role reads the private weights cache and writes private checkpoints;
`runpod_launch.py --dry-run` confirms the full credential path.

First real run on a RunPod H100 80GB (US Secure Cloud):

- Weight sync: ~196 GB S3 -> pod in ~22 min (throughput-bound; East-DC pinning +
  40 concurrent streams help but the transfer is inherently large — use a
  persistent network volume via `--network-volume-id` to pay this once).
- **Stage 0a seed image (Gemini): OK.**
- **Stage 0b panorama (HY-Pano 2.0): OK** — 250 s, real 3.27 MB equirectangular
  Night Market panorama with all five landmarks legible.
- **Stage 1 WorldNav (`traj_generate`): FIXED root cause.** It failed because
  `traj_generate.py` resolves SAM/GroundingDINO/SAM3/MoGe via
  `snapshot_download(cache_dir=~/.cache/huggingface/hub)`, ignoring `HF_HOME`, so
  offline resolution couldn't find `facebook/sam-vit-base` etc. Fix (entrypoint,
  no image rebuild): symlink `~/.cache/huggingface/hub -> $MODEL_ROOT/hub` and pin
  `SAM3_REPO_ID=DiffusionWave/sam3` (the cached ungated mirror).

Remaining to a full four-stage world: stage a persistent volume once, then run
WorldNav -> WorldStereo (heavy; may need multi-GPU) -> multi-view 3DGS, then hand
to the independent gates. No founder blocker remains; this is GPU-time + iteration.

## The original blocker (now resolved by the trust edit)

The RunPod pod must:

1. read the **license-controlled private weights cache**
   (`s3://intelliverse-hyworld-private-us-east-1/models/hy-world/hf`, 131 objects,
   196,695,002,459 bytes) — this includes gated `tencent/HY-World-2.0`, so it
   cannot be re-pulled from a public source; and
2. write **private** stage checkpoints and artifacts to
   `worldgen-full-checkpoints/` and `worldgen-full-staging/`.

The worker uses `boto3` directly (no presigned-URL path), so the pod needs
**scoped AWS credentials**. The safe mechanism already exists as code — the
`hy-world-portable-runner` role with `portable-runner-policy.json` — but the
operator principal cannot use it:

```
$ aws sts assume-role --role-arn arn:aws:iam::970547373533:role/hy-world-portable-runner ...
AccessDenied: user/s3-user is not authorized to perform: sts:AssumeRole
```

The role's trust policy currently allows **only** `ec2.amazonaws.com`. A RunPod
pod is not an EC2 instance, so it cannot receive these credentials, and the
operator cannot mint them. Injecting the operator's own (`s3-user`) credentials
into a third-party GPU host is refused by design (over-privileged; secret
exposure). `iam:UpdateAssumeRolePolicy` and `iam:CreateAccessKey` are not in the
operator policy, so this cannot be self-served.

### Exact unblock (choose one)

- **Preferred** — add the operator principal to the runner role trust so short,
  scoped STS credentials can be minted:

  ```json
  { "Effect": "Allow",
    "Principal": { "AWS": "arn:aws:iam::970547373533:user/s3-user" },
    "Action": "sts:AssumeRole" }
  ```

  After this, `runpod_launch.py` mints 1-hour credentials scoped to
  `portable-runner-policy.json` and injects them into the pod. Nothing else is
  needed.

- **Alternative** — create a dedicated IAM access key bound to
  `portable-runner-policy.json`, store it as an SSM SecureString (e.g.
  `/intelliverse/worldgen/portable-runner-aws`), and the launcher will read and
  inject it the same way.

Both keep the pod least-privilege and keep operator credentials off the pod.

## Launch (once unblocked)

```
python deploy/worldgen/runpod_launch.py --job deploy/worldgen/jobs/full-nm-a.json
```

The launcher (`runpod_launch.py`):

1. reads the RunPod key from SSM,
2. mints scoped 1-hour pod credentials (the blocked step above),
3. registers a short-lived RunPod registry auth for the private ECR image
   `hy-world-full-worker@sha256:24ad1ae4…829b0`,
4. creates an **encrypted** network volume in a US Secure Cloud DC (weights are
   staged once and reused across all five worlds),
5. creates ONE Secure Cloud US 80 GB pod running `runpod-entrypoint.sh`,
6. enforces `--budget-usd` (default 416.46) and terminates the pod.

`runpod-entrypoint.sh` runs inside the pod and is credential-agnostic: GPU check,
weights sync, local Redis, a cleartext->TLS `socat` bridge to
`litellm.intelli-verse-x.ai:443`, `run_pano.py --preflight-only` model-load gate,
then the worker. It carries **two independent self-terminate paths** — a
hard-deadline watchdog and a post-job idle window — that call
`DELETE /v1/pods/{id}` (and `runpodctl` / `poweroff` fallbacks), so a pod can
never bill indefinitely even if the control plane disappears.

## Cost model

- A100-80 SXM: ~$1.49/hr; H100-80: ~$2.69/hr.
- One-time S3 egress for the 183 GiB weights to RunPod: ~$16.5 (first world
  only; the encrypted network volume is reused for the other four).
- Encrypted network volume (400 GB): ~$0.05-0.07/GB/mo, prorated to a few
  dollars for a short run.
- Idle shutdown 15 min; hard deadline default 4 h/pod.

## Not yet done (blocked)

Generation, gating, promotion, and Discord evidence cannot proceed until the
scoped credential exists. No pod, volume, endpoint, or registry auth has been
created; RunPod spend is `$0`.
