# Full HY-World 2.0 production pipeline

## Verified architecture

The fork at upstream commit `7f668e67c74338d50684e57be46a438459b6bbe1`
contains the complete generation path. The founder's four-part description is
directionally correct, with two important details:

1. HY-Pano 2.0's checked-in Qwen backend defaults to a native **1952×960**
   image (`pipeline_with_qwen_image.py`). No supported native 4K preset is
   documented. A larger requested tensor is not called native Ultra-HD until
   upstream validates that resolution.
2. “World composition” is multiple executable stages, not one WorldMirror
   call. `video_gen.py` runs WorldStereo 2.0 and invokes multi-GPU WorldMirror
   for generated-frame geometry/camera alignment. `gen_gs_data.py` then builds
   training data and `world_gs_trainer.py` optimizes the final multi-view 3DGS.

The production sequence is therefore:

1. HY-Pano 2.0: Gemini seed image → Qwen-Image-Edit-2509 + HY-Pano LoRA,
   native 1952×960 panorama.
2. WorldNav: VLM/SAM3/Apache-2.0 SAM/MoGe scene analysis, collision-aware trajectory
   planning, upward and reconstruction routes, then multi-GPU point-cloud
   trajectory rendering.
3. WorldStereo 2.0: FSDP DMD keyframe-video generation with a panorama memory
   bank; WorldMirror 2.0 performs multi-view depth/camera alignment inside this
   stage.
4. Multi-view composition: training-data extraction followed by regularized
   MaskGaussian/gsplat optimization and PLY/SPZ/mesh export.

The rejected shortcut (`deploy/worker.py`) uses none of WorldNav,
WorldStereo, or learned multi-view 3DGS optimization. Its output stays frozen
at production v15 for rollback only.

## Hardware and cost

Upstream recommends at least four GPUs and documents testing on eight H20s.
The initial 8×24GB A10G compatibility probe loaded every Qwen component but
OOMed at native 1952×960 step zero (44.7MiB free). Four-GPU L40S shapes were
capacity-exhausted. The corrected stage topology uses one 48GB L40S
`g6e.xlarge` for HY-Pano, pauses after its durable checkpoint, then resumes
WorldNav/WorldStereo/3DGS on four 24GB ranks.

- Expected runtime: 2.5–4.5 hours per world.
- HY-Pano L40S rate: $1.861/hour (maximum $8.37 if it consumed the whole cap).
- Resume topology and remaining-stage cost are recorded in their own manifests.
- Candidate B remains held until A's health and actual cost are known.

Karpenter normally provisions spot only; hero A uses the documented bounded
on-demand fallback. The worker has a hard 4.5-hour timeout and a
per-job budget check. KEDA scales 0→1 from an isolated full-stack queue, keeps
the pod while the processing list is non-empty, and returns to zero. Empty
nodes consolidate after five minutes. On-demand fallback is a deliberate
operator action after a bounded spot-capacity failure, never an unbounded
automatic spend. After 20 minutes without a NodeClaim launch, capture the
Karpenter events and explicitly add `on-demand` to the NodePool only after
recalculating `INSTANCE_HOURLY_USD`, the job budget, and the round cap. Revert
the capacity type immediately after the candidate.

For the first Night Market A launch, the account's G/VT Spot quota was only
64 vCPUs versus 192 required by `g5.48xlarge`. After the A10G memory ceiling
was measured, both four-L40S shapes were capacity-exhausted in tested AZs.
Stage-specific resume avoids that scarce topology: only HY-Pano requires the
48GB rank. Preflight overhead plus the single-L40S panorama and four-A10G
resume remain below the $100 round cap.

## Durable stage schema and resume

Every successful stage writes:

`s3://intelliverse-hyworld-private-us-east-1/worldgen-full-checkpoints/<job>/stages/<stage>.json`

The manifest records:

- job/stage/status and start/finish/elapsed time;
- source commit and immutable image URI;
- exact model IDs and immutable Hugging Face revisions;
- prompt, seed, conditioner types, token count, and five landmarks;
- instance type, GPU count, estimated stage cost;
- output path, byte size, and SHA-256 for every durable file;
- stage-specific validation metrics.

Files are incrementally synchronized to
`.../<job>/current/{scene,result}/`. A replacement pod restores this snapshot,
reads completed stage manifests, and resumes at the first incomplete stage.
The final provenance manifest links all stage-manifest hashes and costs.
Failure manifests and trace logs are durable; Discord receives stage,
completion, and failure alerts.

Minimum contracts:

- panorama: non-empty image and exact native dimensions;
- WorldNav: camera trajectories, objects metadata, navmesh/up routes;
- trajectory render: non-empty rendered videos;
- WorldStereo: generated videos plus all-five-landmark VLM visibility;
- GS data: non-empty aligned training set;
- GS train: non-empty PLY/SPZ and a completed landmark mapping.

## Authoring and landmark contract

Jobs are rejected unless the prompt has at most 77 whitespace tokens and
contains exactly five front-loaded landmark phrases. The full path uses
Gemini for the seed, Qwen-Image-Edit for HY-Pano, and Gemini VLM for WorldNav;
it does **not** use the shortcut's SDXL CLIP encoder. The conservative 77-token
limit remains enforced to prevent future conditioner drift.

Each landmark record contains:

`id/name → promptPhrase → trajectoryVisibility → reconstructedRegion → checkpointId`

After WorldStereo, a VLM reviews a geometry-only contact sheet and generation
fails unless all five are independently recognizable without viewer overlays.
After 3DGS training, the mapping is bound to the trained geometry artifacts.
Named story objects must reference these landmarks; Three.js overlays are
navigation/feedback only.

Night Market candidates require: market alley, food stalls, ICE-blue hanging
lantern, neon shop signs, and wet stone paving. The ICE lantern is mapped to
`cp-1` before generation.

## Promotion, capture, and rollback

Full-stack output is forced under `worldgen-full-staging/` while
`ALLOW_PRODUCTION_PROMOTION=0`. Production remains atomically rollbackable to
the existing v15 prefix.

Before promotion, each candidate requires:

- complete four-part provenance and stage validation;
- 3840×2160 High: spawn, eight yaws, up/down, mid, close, motion;
- iPhone and iPad captures and performance;
- parallax/reprojection, floor/ceiling coverage, clipping/void, shimmer,
  landmark, and named-object checks;
- full gameplay with unchanged story/audio and server-side redaction;
- independent harsh-player, HD-verifier, and wow-gate scorecards with every
  visual parameter exactly 10.0 and every gameplay/experience parameter ≥9.5.

Only after all three gates pass may an operator enable promotion, atomically
copy the selected artifacts, refresh `spatial.json`, re-author Nakama anchors,
and verify PR #123's canonical HD behavior. Audio is regenerated only if
narration text changed.

## Exact release commands

```bash
# Stage/mirror model weights
kubectl apply -f deploy/worldgen/k8s/stage-weights-job.yaml

# Build immutable image
git archive HEAD | gzip > /tmp/hy-world-full.tar.gz
aws s3 cp /tmp/hy-world-full.tar.gz \
  s3://intelliverse-hyworld-private-us-east-1/build/hy-world/context.tar.gz
kubectl apply -f deploy/worldgen/k8s/build-job.yaml

# Install isolated reserved compute, scaler, and five-minute lifecycle guard
kubectl apply -f deploy/worldgen/k8s/nodepool.yaml
kubectl apply -f deploy/worldgen/k8s/deployment.yaml
kubectl apply -f deploy/worldgen/k8s/scaledobject.yaml
kubectl apply -f deploy/worldgen/k8s/capacity-lifecycle.yaml

# Queue two hero candidates after image/weights/preflight pass
bash deploy/worldgen/enqueue-nightmarket-full.sh
```

## License release blocker

`License.txt` §5(c) prohibits using, distributing, or displaying HY-World
Outputs outside the licensed Territory (EU, UK, and South Korea are excluded).
§6(d) says Tencent claims no ownership in Outputs, but it does not override
the territorial restriction. Production promotion to a globally reachable
viewer therefore requires verified geo-enforcement and the required
machine-generated-content disclosure. Internal US staging can proceed.

## Reserved H100 lifecycle

`worldgen-fullstack-p5` is a one-node Karpenter pool that accepts only
`p5.4xlarge` reserved capacity selected by the
`ManagedBy=hy-world-fullstack,Workload=worldgen-fullstack` tags. Its dedicated
label and taint prevent unrelated pods from consuming the H100. The worker has
the only matching selectors/tolerations.

KEDA scales the worker `0→1` and keeps it while either the signal or processing
Redis list is non-empty. Its cooldown is 900 seconds. Karpenter consolidates
empty nodes after 15 minutes.

`worldgen-capacity-lifecycle` runs every five minutes. When work exists and no
reservation exists it tries one reservation per configured zone, stopping at
the first success. It never creates more than one instance. When both queues
stay empty for 15 minutes, it scales the worker to zero, waits for worker pods
to terminate, deletes only NodeClaims belonging to this pool, then cancels only
tagged reservations. State and timestamps are durable in the private S3
bucket; Discord records create, failure, idle, and cancellation transitions.
Recreating canceled capacity is best effort and may fail.

The controller image is immutable:
`970547373533.dkr.ecr.us-east-1.amazonaws.com/hy-world-capacity-lifecycle@sha256:8a13a20dad9af05622df639862bde4433b7ac47a137ed4a23d490c45241d3f13`.
The current public on-demand reference price is `$6.88/hour`; worker jobs retain
their one-hour `$6.88` hard cap. An unused ODCR is billed, which is why the
controller cancels rather than merely scaling pods.
