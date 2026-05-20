# SC-PPO Report-Grade Status

This note records the current report-grade reading for the repo's `受限优化与平滑性增强`
direction.

Use it to answer two questions:

- what the repo can currently defend
- what still blocks a final report-grade claim

## Mainline reading

The current formal `SC-PPO` mainline is still:

- `threshold = 3.8`
- `PID-Lagrangian`
- `pid_integral_mode = lower_bound_clamp`
- `cost_aggregation = quantile(0.90)`

Selected-checkpoint aggregate over seeds `11`, `17`, and `23`:

- `velocity_tracking_error_mean = 0.6412 +- 0.0554`
- `joint_acceleration_l2_mean = 115.9079 +- 6.9386`
- `action_jitter_l2_mean = 0.2205 +- 0.0017`
- `episode_return_mean = 100.2838 +- 2.7150`
- `fall_rate = 0.1000 +- 0.0000`

What this still supports:

- `SC-PPO 3.8` remains the strongest completed rough-terrain method line in the repo
- the method-side evidence for this line is already at the intended `3-seed + checkpoint-sweep`
  strength

What this no longer supports by itself:

- a closed report-grade `方法优于启发式` claim against a refreshed formal heuristic anchor

## Formal baseline refresh outcome

The rough-terrain formal baseline refresh for issue `#5` is now complete across `Vanilla PPO` and
the bounded heuristic action-rate family:

- [rough-terrain formal comparison note](./baselines/rough-terrain-formal-comparison.md)
- [comparison_summary.json](../artifacts/analysis/rough_terrain_formal_comparison/comparison_summary.json)

Selected-checkpoint aggregate over seeds `11`, `17`, and `23`:

- `Vanilla PPO (formal compare)`:
  - `selected checkpoints = 0 / 0 / 0`
  - `velocity_tracking_error_mean = 1.3321 +- 0.1181`
  - `joint_acceleration_l2_mean = 83.7179 +- 13.3692`
  - `action_jitter_l2_mean = 0.0161 +- 0.0008`
  - `episode_return_mean = 4.0002 +- 0.4323`
  - `fall_rate = 1.0000 +- 0.0000`
- `PPO + heuristic smoothing (action_rate = -0.0005, formal compare)`:
  - `selected checkpoints = 0 / 0 / 0`
  - `velocity_tracking_error_mean = 1.3451 +- 0.1269`
  - `joint_acceleration_l2_mean = 83.7119 +- 14.9052`
  - `action_jitter_l2_mean = 0.0161 +- 0.0009`
  - `episode_return_mean = 4.1998 +- 0.4037`
  - `fall_rate = 1.0000 +- 0.0000`
- `PPO + heuristic smoothing (action_rate = -0.0020, formal compare)`:
  - `selected checkpoints = 0 / 0 / 0`
  - `velocity_tracking_error_mean = 1.3436 +- 0.1232`
  - `joint_acceleration_l2_mean = 85.5995 +- 13.7253`
  - `action_jitter_l2_mean = 0.0161 +- 0.0009`
  - `episode_return_mean = 4.1811 +- 0.3680`
  - `fall_rate = 1.0000 +- 0.0000`
- `PPO + heuristic smoothing (action_rate = -0.0050, formal compare)`:
  - `selected checkpoints = 0 / 0 / 0`
  - `velocity_tracking_error_mean = 1.3359 +- 0.1232`
  - `joint_acceleration_l2_mean = 80.5803 +- 14.6031`
  - `action_jitter_l2_mean = 0.0160 +- 0.0009`
  - `episode_return_mean = 4.1769 +- 0.4080`
  - `fall_rate = 1.0000 +- 0.0000`

Interpretation:

- all twelve selected checkpoints are `checkpoint 0`
- every evaluated checkpoint inside every completed baseline sweep still has `fall_rate = 1.0`
- so this is not a narrow selector mismatch or a single bad heuristic weight; the whole baseline
  side is task-invalid under the completed frozen formal-compare regime
- under the current rule in `CONTEXT.md`, the repo should now treat issue `#5` as a baseline-side
  `协议修复线` rather than as a still-open heuristic-anchor pick

## External-validation reading

The current `MuJoCo` reading remains useful, but it is now a provisional external read against the
previous single-run heuristic comparator rather than against a refreshed report-grade formal anchor.

Current comparable first-pass protocol:

- `terrain_mode = isaac_mainline`
- current resolved XML = `plane`
- `joint_reset_noise = 0.1`
- `20 episodes`
- `20 seconds`

Current comparable numbers:

- previous single-run heuristic comparator:
  - `velocity_tracking_error_mean = 0.6811 +- 0.1113`
  - `joint_acceleration_l2_mean = 110.2715 +- 13.0420`
  - `action_jitter_l2_mean = 0.2005 +- 0.0158`
  - `fall_rate = 0.7000`
- `SC-PPO threshold = 3.8` representative checkpoint:
  - `velocity_tracking_error_mean = 0.6206 +- 0.0458`
  - `joint_acceleration_l2_mean = 154.4672 +- 12.0365`
  - `action_jitter_l2_mean = 0.2785 +- 0.0150`
  - `fall_rate = 0.0500`

This still supports only a bounded `部分迁移` reading:

- stronger task stability than the previous single-run heuristic comparator
- stronger velocity tracking than the previous single-run heuristic comparator
- but not stronger behavior-level smoothness on the current `MuJoCo` metrics

Canonical comparable artifacts:

- [previous heuristic MuJoCo isaac_mainline](../artifacts/methods/heuristic_smoothing_sweep/heuristic_smoothing_action_rate_0050_rough_terrain/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json)
- [SC-PPO 3.8 MuJoCo isaac_mainline](../artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json)

## What is not report-grade yet

The repo still does not support the following claims:

- that the rough-terrain three-way formal comparison is closed
- that the current heuristic baseline is a report-grade formal anchor
- that current smoothness gains fully transfer to `MuJoCo`
- that `MuJoCo terrain` is ready to serve as the main external validation result
- that the final checkpoint alone is sufficient for long-budget reporting
- that a broad neighborhood of tighter thresholds is interchangeable with the `3.8` mainline

The completed local controls still reinforce that boundary:

- repaired `threshold = 4.0` fails to match the `3.8` mainline and includes `seed23 -> checkpoint 0`
- `threshold = 3.6 + full_batch` looked promising on `seed11`, but its formal promotion attempt
  failed at the Isaac stage:
  - `seed11 -> checkpoint 350`
  - `seed17 -> checkpoint 350`
  - `seed23 -> checkpoint 0`

## Recommended citation pattern

When summarizing the current project state, the safest compact wording is:

`在当前粗糙平面主实验中，repaired PID-Lagrangian SC-PPO（threshold = 3.8）仍然是仓库里最强的已完成方法线；但在 issue #5 的正式基线刷新中，Vanilla PPO 与 bounded heuristic action-rate 家族（-0.0005 / -0.0020 / -0.0050）都在 3-seed + checkpoint-sweep 下退化到 checkpoint 0，因此当前 blocker 已从“启发式锚点未选定”转成“baseline 侧 formal-compare 协议需要修复”。MuJoCo isaac_mainline first pass 仍然只支持相对于旧单次 heuristic comparator 的部分迁移结论，而不支持当前 smoothness 指标上的跨引擎转优。`

English-safe wording:

`On the current Isaac rough-terrain task, repaired PID-Lagrangian SC-PPO (threshold = 3.8) remains the strongest completed method line in the repo. However, the issue #5 formal baseline refresh caused Vanilla PPO and the bounded heuristic action-rate family (-0.0005 / -0.0020 / -0.0050) to collapse to checkpoint 0 under the 3-seed checkpoint-sweep protocol, so the blocker is now baseline-side formal-compare protocol repair rather than an unresolved heuristic-anchor pick. The MuJoCo isaac_mainline first pass still supports only a bounded partial-transfer reading against the previous single-run heuristic comparator, not a cross-engine smoothness win under the current metrics.`
