# HY-World capacity IAM audit

Applied 2026-07-13 to AWS account `970547373533`.

## Principals and attachment

- Interactive operator:
  `arn:aws:iam::970547373533:user/s3-user`
- In-cluster lifecycle controller:
  `arn:aws:iam::970547373533:role/eks-sa-ssm-role`
- Cross-region EC2 runner:
  `arn:aws:iam::970547373533:role/hy-world-portable-runner`
- Managed policy:
  `arn:aws:iam::970547373533:policy/HyWorldCapacityReservationOperator`
- Dedicated operator group:
  `arn:aws:iam::970547373533:group/hyworld-capacity-operators`

The user receives the policy only through the dedicated group because its ten
direct managed-policy attachment slots were already occupied. The controller
role has the same policy attached directly.

`operator-capacity-reservations.json` allows only:

- tagged reservation workflows for the audited P-family types in enabled US
  commercial regions,
- cancellation of reservations tagged `ManagedBy=hy-world-fullstack`,
- capacity/AZ/instance/quota/pricing inspection,
- tagged single-`p5.4xlarge` probes in Ohio/Oregon and a tagged
  `p4de.24xlarge` probe in Oregon,
- the exact portable-runner role pass and ECR replication configuration.

It does not grant fleet, networking, security-group, or general EC2
administration. `portable-runner-policy.json` limits the instance to private
model reads, central checkpoint writes, regional worker-image pulls, its
single secret, SSM management, and termination of tagged worldgen capacity.

## Procurement result

Permission propagation was verified by a successful
`CreateCapacityReservation --dry-run` (`DryRunOperation`). Six bounded live
attempts, one in each `p5.4xlarge` offered zone (`us-east-1a` through
`us-east-1f`), all returned `InsufficientInstanceCapacity`. A 24-hour
single-instance p5 Capacity Block query returned zero offerings. No reservation
was created and therefore no ODCR charge was incurred.

Do not broaden this policy to work around supply. AWS must provide a
future-dated reservation/Capacity Block or an immediate ODCR offering before
the full-stack worker can run.

## Cross-region result

`p5.4xlarge` quota was sufficient in Ohio and Oregon, but every offered zone
returned `InsufficientInstanceCapacity` for both ODCR and bounded on-demand
probes. Oregon `p4de.24xlarge` on-demand placement also failed in every offered
zone. Available 24-hour Capacity Blocks cost `$425.09` or more and were not
purchased because they exceed the founder's `$50` candidate boundary. Exact
offerings, quotas, costs, replicated asset hashes, and cleanup evidence are in
`docs/cross-region-capacity-and-license.md`.
