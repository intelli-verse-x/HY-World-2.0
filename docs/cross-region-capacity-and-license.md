# Cross-region capacity and commercial dependency record

Evidence captured 2026-07-13 for AWS account `970547373533`.

## Enabled US capacity

Only `us-east-1`, `us-east-2`, `us-west-1`, and `us-west-2` are enabled US
commercial regions in this account. EU, UK, South Korea, and restricted
locations were not queried or used.

Regional P-instance vCPU quotas are sufficient and adjustable:

- `us-east-1`: 768
- `us-east-2`: 384
- `us-west-1`: 768
- `us-west-2`: 768

No quota increase was submitted because even the largest evaluated one-node
topology fits the existing quota.

Catalog offerings:

- `p5.4xlarge`: `us-east-1`, `us-east-2`, `us-west-2`
- `p4de.24xlarge`: `us-east-1`, `us-west-2`
- `p5.48xlarge`: all four enabled US regions
- `p5en.48xlarge`: all four enabled US regions
- `p5e.48xlarge`: not a valid EC2 instance type; the valid H200 family is
  `p5en.48xlarge`
- no `p6` family offering exists in the enabled US regions

`p5.4xlarge` is the preferred topology: one 80 GiB H100 at `$6.88/hour`.
Two bounded reservation rounds had already exhausted every `us-east-1` zone.
The cross-region round then returned `InsufficientInstanceCapacity` for an
ODCR in every offered `us-east-2` and `us-west-2` zone. On-demand probes also
returned `InsufficientInstanceCapacity` with both EC2-selected and explicit
zones.

The next bounded option was one `p4de.24xlarge` in `us-west-2`: eight 80 GiB
A100 GPUs at `$27.44705/hour`. EC2-selected placement and one explicit probe in
each offered zone (`2a`, `2b`, `2c`) all returned
`InsufficientInstanceCapacity`.

Capacity Blocks existed, but each exceeded the founder's `$50` single-candidate
approval boundary:

- `p5.48xlarge`, `us-east-2a`, 24 hours starting 2026-07-14:
  `$996.67`
- `p4de.24xlarge`, `us-west-2b`, 24 hours starting 2026-07-17:
  `$425.09`

No Capacity Block was purchased. `p5.48xlarge` and `p5en.48xlarge` on-demand
rates are respectively `$55.04/hour` and `$63.296/hour`, so even a one-hour
candidate needs explicit founder reapproval.

## Portable private runner

The production worker image is replicated by digest to Ohio and Oregon:

`sha256:24ad1ae4b0ae26722d710efdb6c1602268c45d40230a7b5a2c96c952311829b0`

The 183.2 GiB model cache is copied server-side to the private, encrypted,
versioned bucket `intelliverse-hyworld-private-us-west-2`. Source and
destination initially matched at 131 objects and 196,695,002,459 bytes; the
licensed SAM replacement was then added in both regions at immutable revision
`70c1a07f894ebb5b307fd9eaaee97b9dfc16068f`.

`portable-runner.sh` uses a dedicated EC2 role, regional ECR and S3, local Redis
queue semantics, a TLS bridge to the existing LiteLLM gateway, atomic central
S3 checkpoints, real HY-Pano model-load preflight, a 15-minute idle delay, and
an independent one-hour termination deadline. The staged split-topology cap
was:

- one hour `p4de.24xlarge`: `$27.44705`
- one hour `g5.48xlarge` downstream: `$16.288`
- model transfer: about `$3.66`
- worker image transfer: about `$0.21`
- one day regional S3 cache: about `$0.14`
- two one-hour 1 TiB gp3 work volumes: about `$0.39`
- total: about `$48.14`

No instance launched, so actual GPU and EBS cost is `$0`. No tagged reservation
or portable instance remains active.

## Founder-approved hourly fallback result

The bounded hourly fallback was attempted without purchasing a Capacity Block.
`p5.48xlarge` returned `InsufficientInstanceCapacity` in every offered zone in
Virginia, Ohio, Oregon, and Northern California. `p5en.48xlarge` returned the
same result in every offered zone in Virginia, Ohio, Oregon, and Northern
California. Earlier probes had already exhausted every offered `p5.4xlarge`
zone and every Oregon `p4de.24xlarge` zone.

The exact verified Linux on-demand rates were `$55.04/hour` for `p5.48xlarge`
in the three priority regions, `$68.80/hour` in Northern California,
`$63.296/hour` for `p5en.48xlarge` in the three priority regions, and
`$79.12/hour` in Northern California. User-data deadlines were reduced by
region so a successful first launch plus ancillary work could not exceed the
founder's `$60` first-attempt cap. No launch succeeded, so GPU, EBS, and
instance runtime cost remained `$0`.

Private caches are retained in Ohio and Northern California at 155 objects and
204,011,284,867 bytes each, matching the central source exactly. They are
encrypted, versioned, block all public access, and have no GPU attachment. The
immutable worker was also replicated to Northern California at digest
`sha256:24ad1ae4b0ae26722d710efdb6c1602268c45d40230a7b5a2c96c952311829b0`.
Estimated metered ancillary cost is `$8.54`: `$8.16` cross-region cache
transfer, about `$0.22` worker-image transfer, and at most `$0.16` for
short-lived NAT/public IPv4, requests, and initial storage. Cost Explorer is
still marked estimated and account-wide, so it cannot yet isolate this run.

All temporary NAT gateways, elastic IPs, route tables, VPC endpoints, and
subnets were removed. All four enabled US regions report zero tagged
instances, volumes, and active reservations. The full-stack deployment is
zero replicas, its signal and processing queues are empty, KEDA is inactive,
and there are zero full-stack NodeClaims or nodes.

## Commercial dependency remediation

ZIM was used only in WorldNav to turn Grounding DINO sky boxes into binary
masks. Its code, package, model ID, immutable revision, and staging entry were
removed. The compatible replacement is `facebook/sam-vit-base`: Grounding
DINO still supplies XYXY boxes and a small adapter returns the same
`N × H × W` mask contract. The primary model card and upstream code license
both declare Apache License 2.0.

SDXL-Turbo was not used by the production full-stack path, but remained as the
legacy shortcut's seed default. That code path now uses the same approved
Gemini image API as the full-stack worker and contains no SDXL-Turbo model
reference or download.

This removes the two identified avoidable commercial blockers. It does not
convert Tencent HY-World into an unrestricted public-release license. Tencent
territory, MAU, attribution, provider-identity, disclosure, and explicit
founder/legal approval gates still apply. Generated output remains private;
controlled delivery remains fail-closed and direct S3 access returns 403.
