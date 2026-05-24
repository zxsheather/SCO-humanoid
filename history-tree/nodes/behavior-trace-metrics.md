# Behavior Trace Metrics

- **Date**: 2026-05-22
- **Type**: experiment
- **Outcome**: mixed
- **Tags**: trace-metrics, ldlj, sparc

## Timeline and Background

Issue [`#29`](https://github.com/zxsheather/SCO-humanoid/issues/29) narrows the post-freeze question again: if the repo captures actual action or joint traces, do independent smoothness metrics move with the current local-sensitivity story or against it?

## Technical Details

- Canonical note: [docs/sc-ppo-behavior-smoothness-metric-diagnostic.md](../../docs/sc-ppo-behavior-smoothness-metric-diagnostic.md)
- Merged via PR [`#30`](https://github.com/zxsheather/SCO-humanoid/pull/30).
- Added components:
  - `scripts/baseline/_behavior_trace_metrics.py`
  - `scripts/analysis/compute_behavior_smoothness_metrics.py`
  - trace capture in `evaluate_policy.py` and forwarding in `evaluate_checkpoint_sweep.py`
- First metrics:
  - `joint_position_ldlj_mean`
  - `joint_velocity_sparc_mean`

## Decision Process

- The repo kept the evidence weak on purpose: bounded replays, small traces, no full retraining.
- Even that weak first pass was useful because the new metrics did **not** trivially agree with the old ordering.

## Results and Impact

- The branch complicated the smoothness story in a productive way.
- It showed that "independent behavior smoothness" is not yet reducible to the repo's existing local-sensitivity scalar.
- Together with [objective-mismatch-diagnostic](../nodes/objective-mismatch-diagnostic.md), it forms the main post-freeze analysis branch on `main`.
