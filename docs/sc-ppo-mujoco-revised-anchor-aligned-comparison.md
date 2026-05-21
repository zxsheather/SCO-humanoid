# SC-PPO MuJoCo Revised-Anchor Aligned Comparison

This note records the completed aligned `MuJoCo isaac_mainline` replay after the rough-terrain
heuristic anchor was repaired to the revised long-budget protocol.

## Protocol

- terrain mode: `isaac_mainline`
- resolved MuJoCo XML: `plane`
- reset noise: `joint_reset_noise = 0.1`
- duration: `20 episodes`, `20 seconds`
- command: `vx = 0.4`, `vy = 0.0`, `dyaw = 0.0`
- heuristic selected checkpoints: `350 / 300 / 350`
- `SC-PPO 3.8` selected checkpoints: `300 / 300 / 400`

## Revised Heuristic Anchor

Aggregate over seeds `11`, `17`, and `23`:

- `velocity_tracking_error_mean = 0.4188 +- 0.0398`
- `joint_acceleration_l2_mean = 120.7339 +- 2.6413`
- `action_jitter_l2_mean = 0.2452 +- 0.0288`
- `fall_rate = 0.0000 +- 0.0000`
- `episode_steps_mean = 2000.0 +- 0.0`
- `episode_return_mean = -465.3696 +- 196.6838`

Artifacts:

- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed11/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed17/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed23/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`

## SC-PPO 3.8

Aggregate over seeds `11`, `17`, and `23`:

- `velocity_tracking_error_mean = 0.4910 +- 0.0944`
- `joint_acceleration_l2_mean = 125.5411 +- 21.1683`
- `action_jitter_l2_mean = 0.2313 +- 0.0351`
- `fall_rate = 0.0167 +- 0.0236`
- `episode_steps_mean = 1984.7833 +- 21.5196`
- `episode_return_mean = -647.6742 +- 279.0772`

Artifacts:

- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed23/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`

## Interpretation

This aligned replay changes the earlier `MuJoCo` reading.

The old first-pass wording compared `SC-PPO 3.8` against a previous single-run heuristic
comparator. After aligning to the revised heuristic formal anchor, the result is mixed:

- revised heuristic is better on `velocity_tracking_error_mean`
- revised heuristic is better on `fall_rate`
- revised heuristic is better on `episode_steps_mean`
- revised heuristic is slightly better on `joint_acceleration_l2_mean`
- `SC-PPO 3.8` is slightly better on `action_jitter_l2_mean`

So the repo should not claim `SC-PPO 3.8` has a `MuJoCo isaac_mainline` cross-engine win against the
revised heuristic anchor.

The current safe reading is:

`SC-PPO 3.8` remains stronger than the revised heuristic anchor on the Isaac rough-terrain main
comparison, but the aligned `MuJoCo isaac_mainline` replay does not preserve that ordering. The
external-validation result should be reported as mixed evidence rather than as a partial transfer
advantage for `SC-PPO`.

## Execution Notes

The `SC-PPO 3.8` parallel run wrote metrics for all three seeds, but the concurrent process wrapper
showed exit-stage instability:

- `seed11` initially failed because `ninja` was not visible on `PATH`; rerunning with
  `/TinyNAS2024/zhuoxiang/sco-humanoid/bin` prepended to `PATH` fixed it
- `seed17` and `seed23` wrote complete metric files and then exited with `segmentation fault`

Treat the JSON metric artifacts as the canonical evidence, and avoid relying on the shell wrapper's
exit code for this MuJoCo batch unless the evaluator teardown behavior is repaired.
