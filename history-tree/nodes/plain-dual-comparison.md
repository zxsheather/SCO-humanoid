# Plain Dual Ascent (PPO-Lagrangian) Comparison

- **Date**: 2026-05-26
- **Type**: experiment
- **Outcome**: mixed
- **Tags**: sota-baseline, dual-ascent, pid-ablation

## Timeline and Background

The original PID-limited ablation used a narrow matched diagnostic. The later full 3-seed plain-dual
comparison tested whether standard PPO-Lagrangian dual ascent could match the current PID-Lagrangian
SC-PPO line under the canonical rough-terrain entry.

## Technical Details

- Plain dual setting: `update_mode = "dual"`, `dual_lr = 0.01`.
- Per-seed selected results:
  - seed11: succeeds with `selected = final = 400`, `fall_rate = 0.0`
  - seed17: partial, `selected = 300`
  - seed23: collapses, `selected = 0`
- Selected-checkpoint aggregate:
  - `fall_rate = 0.42`
  - `joint_acceleration_l2_mean = 108.3`
  - `action_jitter_l2_mean = 0.16`

## Decision Process

- Plain dual ascent is not universally infeasible, because seed11 succeeds.
- It is still not a stable replacement for PID-Lagrangian because seed23 catastrophically fails.

## Results and Impact

- PID-Lagrangian's main value is cross-seed stability, not simply lower smoothness metrics on an easy
  seed.
- Canonical artifact: `artifacts/analysis/rough_terrain_plain_dual_probe/comparison_summary.json`.
