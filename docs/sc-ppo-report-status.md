# SC-PPO Report-Grade Status

This note records the current report-grade reading for the repo's `受限优化与平滑性增强` direction.

It is narrower than `docs/sc-ppo-current-blockers.md` and less exploratory than the probe notes.
Use this document when the goal is to state what the repo can currently defend, what it cannot yet
defend, and which artifacts should be cited.

## Mainline result

The current formal mainline is:

- `SC-PPO threshold = 3.8`
- `PID-Lagrangian`
- `pid_integral_mode = lower_bound_clamp`
- `cost_aggregation = quantile(0.90)`

On the current `粗糙平面` task, this branch has completed a `3-seed, 400 iteration,
checkpoint-sweep` comparison and currently supports a real `方法优于启发式` claim under the repo's
shared metric schema.

Selected-checkpoint aggregate over seeds `11`, `17`, and `23`:

- `velocity_tracking_error_mean = 0.6412 ± 0.0554`
- `joint_acceleration_l2_mean = 115.9079 ± 6.9386`
- `action_jitter_l2_mean = 0.2205 ± 0.0017`
- `episode_return_mean = 100.2838 ± 2.7150`
- `fall_rate = 0.1000 ± 0.0000`

Current heuristic anchor:

- `velocity_tracking_error_mean = 1.1381`
- `joint_acceleration_l2_mean = 140.6399`
- `action_jitter_l2_mean = 0.2457`
- `fall_rate = 1.0`

Canonical supporting artifacts:

- [3.8 seed11 checkpoint sweep](../artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11/checkpoint_sweep_summary.json)
- [3.8 seed17 checkpoint sweep](../artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17/checkpoint_sweep_summary.json)
- [3.8 seed23 checkpoint sweep](../artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed23/checkpoint_sweep_summary.json)
- [heuristic anchor MuJoCo isaac_mainline](../artifacts/methods/heuristic_smoothing_sweep/heuristic_smoothing_action_rate_0050_rough_terrain/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json)

## External-validation reading

The current report-grade `MuJoCo` reading is limited to the `isaac_mainline` first-pass protocol:

- `terrain_mode = isaac_mainline`
- current resolved XML = `plane`
- `joint_reset_noise = 0.1`
- `20 episodes`
- `20 seconds`

Under this protocol, the current `3.8` mainline supports a bounded `部分迁移` claim:

- stronger task stability than the heuristic anchor
- stronger velocity tracking than the heuristic anchor
- but not stronger smoothness on the current `行为层平滑指标`

Current comparable numbers:

- heuristic anchor:
  - `velocity_tracking_error_mean = 0.6811 ± 0.1113`
  - `joint_acceleration_l2_mean = 110.2715 ± 13.0420`
  - `action_jitter_l2_mean = 0.2005 ± 0.0158`
  - `fall_rate = 0.7000`
- `SC-PPO threshold = 3.8` representative checkpoint:
  - `velocity_tracking_error_mean = 0.6206 ± 0.0458`
  - `joint_acceleration_l2_mean = 154.4672 ± 12.0365`
  - `action_jitter_l2_mean = 0.2785 ± 0.0150`
  - `fall_rate = 0.0500`

Canonical comparable artifacts:

- [heuristic MuJoCo isaac_mainline](../artifacts/methods/heuristic_smoothing_sweep/heuristic_smoothing_action_rate_0050_rough_terrain/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json)
- [3.8 MuJoCo isaac_mainline](../artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json)

## What is not report-grade yet

The repo does not yet support the following claims:

- that current smoothness gains fully transfer to `MuJoCo`
- that `MuJoCo terrain` is ready to serve as the main external validation result
- that the final checkpoint alone is sufficient for long-budget reporting
- that a broad neighborhood of tighter thresholds is interchangeable with the `3.8` mainline

This boundary is reinforced by recent local controls:

- repaired `threshold = 4.0` fails to match the `3.8` mainline and includes `seed23 -> checkpoint 0`
- `threshold = 3.6 + full_batch` looked promising on `seed11`, but its formal promotion attempt
  failed at the Isaac stage:
  - `seed11 -> checkpoint 350`
  - `seed17 -> checkpoint 350`
  - `seed23 -> checkpoint 0`
- so `3.6 + full_batch` should currently be cited only as a completed `诊断支线`, not as an active
  replacement candidate for the `3.8` mainline

Supporting promotion-failure artifact:

- [3.6 full_batch promotion outcome](./sc-ppo-fullbatch-threshold-promotion.md)

## Terrain-side status

`MuJoCo terrain` should currently be split into two non-report-grade protocol lines:

- `hfield_stress`
- `hfield_moderate`

Current reading:

- `hfield_stress` is still a transfer-pressure probe where both methods fail
- `hfield_moderate` is now a useful repair-stage intermediate protocol because it reveals a real
  survival difference, but it still does not support the desired smoothness conclusion

So neither terrain protocol should currently be folded into the main report headline.

## Recommended citation pattern

When summarizing the current project state, the safest compact wording is:

`在当前粗糙平面主实验中，repaired PID-Lagrangian SC-PPO（threshold = 3.8）已经通过 3-seed + checkpoint-sweep 结果形成对强启发式基线的主结果；在 MuJoCo isaac_mainline first pass 中，该方法显示出任务稳定性与速度跟踪的部分迁移，但当前行为层平滑指标尚未完成跨引擎转优。`
