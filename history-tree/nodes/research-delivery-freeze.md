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
- Backport policy merges on 2026-05-22:
  - PR [`#31`](https://github.com/zxsheather/SCO-humanoid/pull/31) fixed `load_run` path resolution
  - PR [`#32`](https://github.com/zxsheather/SCO-humanoid/pull/32) documented bounded frozen-mainline backports
- Branch read at freeze time:
  - `main` and `origin/main` are the archival line
  - local `action-scaling-line` and `output-scaling-line` point at `main`
  - local `orthogonal-actor-line` is the only branch ahead, via commit `82d6032`

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
  - [objective-mismatch-diagnostic](../nodes/objective-mismatch-diagnostic.md)
  - [output-scaling-line](../nodes/output-scaling-line.md)
