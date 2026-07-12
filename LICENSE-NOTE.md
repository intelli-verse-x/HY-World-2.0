# License note — Tencent HY-World 2.0 community license

This fork is deployed on Intelliverse private infrastructure to generate
**internal world templates only** (the "generate-once/play-many" pipeline in
`deploy/worldgen/`). The upstream model weights and code are covered by the
Tencent Hunyuan community license (`License.txt`), which imposes constraints
we must respect if usage ever expands:

- **Territory restriction**: the license does not grant rights in the
  European Union, United Kingdom, or South Korea. Any product surface that
  serves HY-World-generated content to end users must geo-block EU/UK/KR.
- **Scale restriction**: use is limited to services with fewer than
  1,000,000 monthly active users; beyond that, a separate license from
  Tencent is required.
- **Attribution / notice**: keep `License.txt` and this notice with any
  redistribution of the code or derived weights.

Current usage (July 2026): batch generation of internal 3D world templates on
private EKS GPU nodes; artifacts are stored in a private S3 bucket and are not
distributed to end users. This is within the license terms. Revisit this note
before shipping generated worlds into any user-facing product.
