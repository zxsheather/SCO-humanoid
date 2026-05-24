# PID-Limited Ablation

- **Date**: 2026-05-21
- **Type**: experiment
- **Outcome**: mixed
- **Tags**: ablation, pid, plain-dual

## Timeline and Background

Issue [`#6`](https://github.com/zxsheather/SCO-humanoid/issues/6) was deliberately narrow. The repo did not want a full attribution matrix; it only needed to know whether plain dual ascent was viable enough to threaten the `PID-Lagrangian` mainline.

## Technical Details

- Canonical note: [docs/sc-ppo-pid-limited-ablation.md](../../docs/sc-ppo-pid-limited-ablation.md)
- Closing commit for the note: [`f9c5211`](https://github.com/zxsheather/SCO-humanoid/commit/f9c5211)
- Matched diagnostic:
  - config `sc_ppo_threshold_38_lambda_05_quantile_090_dual_001.json`
  - checkpoint `100`
  - result `selection_status = all_checkpoints_collapsed`
  - `fall_rate = 1.0`

## Decision Process

- The repo treated lower action jitter under collapse as non-evidence.
- The question was not "can plain dual ever be tuned," but "is it strong enough in the current matched diagnostic to replace the formal mainline?" The answer was no.

## Results and Impact

- `PID-Lagrangian` stayed the formal SC-PPO algorithm choice.
- The branch closed as mechanism support, not as a new headline result.
- This limited the scope of later follow-up work and kept the project from expanding into a full component-ablation study.
