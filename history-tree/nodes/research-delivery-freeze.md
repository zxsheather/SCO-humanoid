# Research Delivery Freeze

- **Date**: 2026-05-21
- **Type**: decision
- **Outcome**: success
- **Tags**: freeze, handoff, main-branch

## Timeline and Background

Issue [`#14`](https://github.com/zxsheather/SCO-humanoid/issues/14) and ADR [`0001`](../../docs/adr/0001-freeze-research-delivery-before-new-protocol-repair.md) formally close the repo's main evidence loop. By this point the project had a repaired rough-terrain anchor, an aligned MuJoCo read, a limited PID ablation, a negative SN branch, and a negative random-stairs stress test.

## Technical Details

- Freeze commits:
  - [`1541893`](https://github.com/zxsheather/SCO-humanoid/commit/1541893) `Freeze internal research delivery package`
  - [`1cf0f27`](https://github.com/zxsheather/SCO-humanoid/commit/1cf0f27) `Update docs for completed research freeze`
- Tree-reading note:
  - the rendered tree uses one visual parent edge for layout
  - the actual freeze boundary depends on multiple completed lines, not only the adjacent
    `random-stairs` node
- Backport policy merges on 2026-05-22:
  - PR [`#31`](https://github.com/zxsheather/SCO-humanoid/pull/31) fixed `load_run` path resolution
  - PR [`#32`](https://github.com/zxsheather/SCO-humanoid/pull/32) documented bounded frozen-mainline backports
- Later post-freeze work was supposed to branch off this boundary rather than rewrite it:
  - some diagnostics were merged back as reusable archival tooling
  - additional same-question exploration continued on separate issue-tracked or local branches

## Decision Process

- The repo explicitly chose **仓库内科研交付包** over more experiment churn.
- The freeze rule was narrow:
  - allow wording and reusable diagnostic backports
  - do not reopen core experiment closure work on `main`
- See [docs/sc-ppo-next-step-direction.md](../../docs/sc-ppo-next-step-direction.md) and [docs/reproduction/final-research-delivery-checklist.md](../../docs/reproduction/final-research-delivery-checklist.md).

## Results and Impact

- `main` became a frozen archival branch rather than an active experiment trunk.
- New work after this point is historical, but it is supposed to branch off the freeze rather than rewrite it.
- Post-freeze examples:
  - [random-stairs-protocol-repair](../nodes/random-stairs-protocol-repair.md)
  - [anisotropic-constraint-diagnostic](../nodes/anisotropic-constraint-diagnostic.md)
  - [sn-task-stabilized-diagnostic](../nodes/sn-task-stabilized-diagnostic.md)
  - [objective-mismatch-diagnostic](../nodes/objective-mismatch-diagnostic.md)
  - [output-scaling-line](../nodes/output-scaling-line.md)
