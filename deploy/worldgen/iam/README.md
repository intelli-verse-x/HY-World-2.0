# HY-World capacity IAM audit

Applied 2026-07-13 to AWS account `970547373533`.

## Principals and attachment

- Interactive operator:
  `arn:aws:iam::970547373533:user/s3-user`
- In-cluster lifecycle controller:
  `arn:aws:iam::970547373533:role/eks-sa-ssm-role`
- Managed policy:
  `arn:aws:iam::970547373533:policy/HyWorldCapacityReservationOperator`
- Dedicated operator group:
  `arn:aws:iam::970547373533:group/hyworld-capacity-operators`

The user receives the policy only through the dedicated group because its ten
direct managed-policy attachment slots were already occupied. The controller
role has the same policy attached directly.

`operator-capacity-reservations.json` allows only:

- one tagged `p5.4xlarge` reservation workflow in `us-east-1`,
- cancellation of reservations tagged `ManagedBy=hy-world-fullstack`,
- capacity/AZ/instance/quota inspection.

It does not grant EC2 instance, fleet, networking, security-group, or general
EC2 administration.

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
