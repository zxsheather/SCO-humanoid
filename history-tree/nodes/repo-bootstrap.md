# Research Repo Bootstrap

- **Date**: 2026-05-14
- **Type**: milestone
- **Outcome**: success
- **Tags**: initial-commit, humanoid-gym, research-repo

## Timeline and Background

The project starts with the initial commit [`f22eacc`](https://github.com/zxsheather/SCO-humanoid/commit/f22eacc) on 2026-05-14. From day one, the repo is framed as a research validation workspace for smooth humanoid locomotion, not as a general RL framework.

## Technical Details

- The initial scaffold established the current repo shape: `configs/`, `scripts/`, `docs/`, `artifacts/`, and `tests/`.
- The training backbone was declared as a local checkout of `Humanoid-Gym` under [`.external/humanoid-gym`](../../.external/humanoid-gym).
- The long-lived framing appears in [README.md](../../README.md) and [CONTEXT.md](../../CONTEXT.md): reproducible evidence first, productization explicitly out of scope.

## Decision Process

- The first issue batch [`#1`](https://github.com/zxsheather/SCO-humanoid/issues/1) through [`#4`](https://github.com/zxsheather/SCO-humanoid/issues/4) already encoded the repo's early roadmap: baseline reproduction, shared metrics, heuristic baseline selection, then constrained SC-PPO.
- That issue ordering matters: later diagnostics and freeze docs keep referring back to the repo as a **科研验证型交付**, not a reusable platform.

## Results and Impact

- This bootstrap made the later evidence chain inspectable instead of ad hoc.
- Every later node in the tree assumes this same repo contract: configs are named experiment identities, artifacts are canonical evidence, and docs are part of the result.
- Related follow-up nodes: [shared-metric-schema](../nodes/shared-metric-schema.md), [sc-ppo-pid-mainline](../nodes/sc-ppo-pid-mainline.md).
