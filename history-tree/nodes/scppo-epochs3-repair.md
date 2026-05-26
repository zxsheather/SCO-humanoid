# SC-PPO epochs=3 Reliability Repair

- **Date**: 2026-05-26
- **Type**: experiment
- **Outcome**: failure
- **Tags**: sc-ppo, reliability, epochs, negative

## Timeline and Background

LayerNorm benefited strongly from `num_learning_epochs = 3`, so the repo tested whether the same
schedule change would repair final-checkpoint reliability for the current `SC-PPO 3.8` mainline.

## Technical Details

| Seed | epochs=2 selected | epochs=3 selected | Change |
| ---: | ---: | ---: | --- |
| 11 | 300 | 400 | improved |
| 17 | 300 | 300 | unchanged |
| 23 | 400 | 300 | degraded |

- Selected aggregate with epochs=3:
  - `fall_rate = 0.28`
  - `joint_acceleration_l2_mean = 169.1`
  - `action_jitter_l2_mean = 0.32`
- Both smoothness metrics worsened relative to the epochs=2 `SC-PPO 3.8` mainline.

## Decision Process

- The repo rejected `num_learning_epochs = 3` as a universal SC-PPO final-checkpoint reliability fix.
- The result shows that the training schedule interacts differently with the Jacobian path than with
  LayerNorm.

## Results and Impact

- The current `SC-PPO 3.8` mainline remains selected-checkpoint based.
- Canonical artifact: `artifacts/analysis/rough_terrain_sc_ppo_epochs3_probe/comparison_summary.json`.
