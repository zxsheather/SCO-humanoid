# Action Scaling MuJoCo Replay

- **Date**: 2026-05-25
- **Type**: experiment
- **Outcome**: failure
- **Tags**: mujoco, cross-engine, action-scaling, collapse

## Timeline and Background

After action-side scaling closed as a partial/negative Isaac-side replacement, its selected
checkpoints were replayed in `MuJoCo isaac_mainline` to measure cross-engine degradation.

## Technical Details

- Replay checkpoints: selected `400 / 400 / 400`.
- MuJoCo result: all three seeds collapsed with `fall_rate = 1.0`.
- Aggregate MuJoCo dynamic smoothness:
  - `joint_acceleration_l2_mean_mean = 1835.6`
  - `action_jitter_l2_mean_mean = 8.3`
- Isaac-to-MuJoCo joint-acceleration degradation factor: about `12.7x`.

## Decision Process

- The repo closed action-side scaling as a same-question replacement direction.
- The collapse ruled out a near-term gain, schedule, or clipping retry inside the same line.

## Results and Impact

- This is the strongest negative cross-engine degradation point in the current table.
- It supports the paper claim that non-Jacobian replacement mechanisms failed to preserve
  dynamic smoothness across engines.
