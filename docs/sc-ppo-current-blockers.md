# SC-PPO Current Blockers

This document records the current experimental blockers and claim-boundary limits for `SC-PPO` on
the repo's `速度跟踪行走` task under the `粗糙平面` condition.

## Primary Status

The previous primary blocker is now resolved:

`主线证据闭环已经按 revised heuristic anchor 重冻结`

Reason:

- the completed revised-protocol long-budget heuristic run now yields
  `selected checkpoints = 350 / 300 / 350`
- so the repo is no longer blocked by `whether any task-valid heuristic formal anchor exists`
- the rough-terrain Isaac `三组正式对比` can now be stated around:
  - `Vanilla PPO` raw reference
  - the revised heuristic anchor
  - `SC-PPO 3.8`

So the repo is no longer blocked by evidence-boundary closure. The remaining work is report,
README, `CONTEXT.md`, and issue-tracker reconciliation.

## External-Validation Boundary

The current external-validation boundary is:

`MuJoCo 对齐 revised heuristic anchor 后不再支持 SC-PPO 跨引擎优势`

Reason:

- the aligned `MuJoCo isaac_mainline` replay now compares revised heuristic and `SC-PPO 3.8` over
  the same three seeds
- revised heuristic is better on task stability, velocity tracking, episode length, and joint
  acceleration
- `SC-PPO 3.8` is only better on action jitter
- so the report must use `混合外部验证结论` rather than the older wording that implied a MuJoCo
  task-side transfer advantage for `SC-PPO`

## Residual Limitation

The current residual limitation is:

`当前主线仍然依赖 checkpoint sweep，而 final checkpoint 不能直接替代`

Reason:

- the `SC-PPO 3.8` mainline still depends on selected checkpoints rather than final checkpoint only
- repaired `threshold = 4.0` includes `seed23 -> checkpoint 0`
- `threshold = 3.6 + full_batch` also fails promotion at the Isaac stage because
  `seed23 -> checkpoint 0`

## Confirmed facts

Current `SC-PPO 3.8` selected-checkpoint aggregate over seeds `11`, `17`, and `23`:

- `velocity_tracking_error_mean = 0.6412 +- 0.0554`
- `joint_acceleration_l2_mean = 115.9079 +- 6.9386`
- `action_jitter_l2_mean = 0.2205 +- 0.0017`
- `episode_return_mean = 100.2838 +- 2.7150`
- `fall_rate = 0.1000 +- 0.0000`

Completed rough-terrain frozen formal baseline refresh:

- `Vanilla PPO (frozen formal compare)`:
  - `selected checkpoints = 0 / 0 / 0`
  - `velocity_tracking_error_mean = 1.3321 +- 0.1181`
  - `joint_acceleration_l2_mean = 83.7179 +- 13.3692`
  - `action_jitter_l2_mean = 0.0161 +- 0.0008`
  - `episode_return_mean = 4.0002 +- 0.4323`
  - `fall_rate = 1.0000 +- 0.0000`
- `heuristic smoothing action_rate = -0.0005 (frozen formal compare)`:
  - `selected checkpoints = 0 / 0 / 0`
  - `velocity_tracking_error_mean = 1.3451 +- 0.1269`
  - `joint_acceleration_l2_mean = 83.7119 +- 14.9052`
  - `action_jitter_l2_mean = 0.0161 +- 0.0009`
  - `episode_return_mean = 4.1998 +- 0.4037`
  - `fall_rate = 1.0000 +- 0.0000`
- `heuristic smoothing action_rate = -0.0020 (frozen formal compare)`:
  - `selected checkpoints = 0 / 0 / 0`
  - `velocity_tracking_error_mean = 1.3436 +- 0.1232`
  - `joint_acceleration_l2_mean = 85.5995 +- 13.7253`
  - `action_jitter_l2_mean = 0.0161 +- 0.0009`
  - `episode_return_mean = 4.1811 +- 0.3680`
  - `fall_rate = 1.0000 +- 0.0000`
- `heuristic smoothing action_rate = -0.0050 (frozen formal compare)`:
  - `selected checkpoints = 0 / 0 / 0`
  - `velocity_tracking_error_mean = 1.3359 +- 0.1232`
  - `joint_acceleration_l2_mean = 80.5803 +- 14.6031`
  - `action_jitter_l2_mean = 0.0160 +- 0.0009`
  - `episode_return_mean = 4.1769 +- 0.4080`
  - `fall_rate = 1.0000 +- 0.0000`

Completed repaired-budget protocol-repair probe on the previous heuristic winner:

- `selected checkpoints = 0 / 0 / 200`
- `velocity_tracking_error_mean = 1.1558 +- 0.1545`
- `joint_acceleration_l2_mean = 111.5311 +- 25.5306`
- `action_jitter_l2_mean = 0.1023 +- 0.1211`
- `episode_return_mean = 22.3952 +- 26.3617`
- `fall_rate = 0.9167 +- 0.1179`
- per-seed status:
  - `seed11 -> all_checkpoints_collapsed`
  - `seed17 -> all_checkpoints_collapsed`
  - `seed23 -> checkpoint 200`, `fall_rate = 0.75`

Completed revised-protocol long-budget heuristic anchor:

- `selected checkpoints = 350 / 300 / 350`
- `velocity_tracking_error_mean = 0.7549 +- 0.1068`
- `joint_acceleration_l2_mean = 119.8639 +- 2.1966`
- `action_jitter_l2_mean = 0.2711 +- 0.0084`
- `episode_return_mean = 100.9327 +- 11.2711`
- `fall_rate = 0.1500 +- 0.0816`

Completed aligned `MuJoCo isaac_mainline` comparison:

- revised heuristic:
  - `velocity_tracking_error_mean = 0.4188 +- 0.0398`
  - `joint_acceleration_l2_mean = 120.7339 +- 2.6413`
  - `action_jitter_l2_mean = 0.2452 +- 0.0288`
  - `fall_rate = 0.0000 +- 0.0000`
  - `episode_steps_mean = 2000.0 +- 0.0`
- `SC-PPO 3.8`:
  - `velocity_tracking_error_mean = 0.4910 +- 0.0944`
  - `joint_acceleration_l2_mean = 125.5411 +- 21.1683`
  - `action_jitter_l2_mean = 0.2313 +- 0.0351`
  - `fall_rate = 0.0167 +- 0.0236`
  - `episode_steps_mean = 1984.7833 +- 21.5196`

## Immediate Reconciliation Target

The current first-priority remediation target is:

`同步 README、CONTEXT、GitHub Issues 和报告草稿，使它们都反映已完成的主线证据闭环`

Execution-facing notes:

- do not reopen the same bounded action-rate search as if the issue were still only anchor selection
- do not spend new budget on `Vanilla PPO` repair; keep its collapse as raw-reference evidence
- do not describe aligned `MuJoCo isaac_mainline` as an `SC-PPO` win
- keep terrain-side `MuJoCo`, `PID有限消融`, and `SN` as separate follow-up lines unless they are
  explicitly promoted later

## Canonical references

- [SC-PPO report-grade status](./sc-ppo-report-status.md)
- [rough-terrain formal comparison](./baselines/rough-terrain-formal-comparison.md)
- [rough-terrain formal protocol repair probe](./baselines/rough-terrain-formal-protocol-repair-probe.md)
- [rough-terrain formal protocol revision decision](./baselines/rough-terrain-formal-protocol-revision-decision.md)
- [rough-terrain formal protocol revision long-budget test](./baselines/rough-terrain-formal-protocol-revision-long-budget.md)
- [SC-PPO MuJoCo revised-anchor aligned comparison](./sc-ppo-mujoco-revised-anchor-aligned-comparison.md)
- [SC-PPO next-step direction](./sc-ppo-next-step-direction.md)
