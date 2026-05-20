# SC-PPO Current Blockers

This document records the current experimental blockers for `SC-PPO` on the repo's
`速度跟踪行走` task under the `粗糙平面` condition.

## Primary blocker

The current primary blocker is:

`当前粗糙平面 formal-compare 协议本身需要修复，issue #5 不能按现有冻结预算闭合`

Reason:

- the rough-terrain formal baseline refresh is now complete across `Vanilla PPO` and the bounded
  heuristic action-rate family
- `Vanilla PPO` and `heuristic smoothing action_rate = -0.0005 / -0.0020 / -0.0050` all select
  `checkpoint 0` on all three seeds
- every evaluated checkpoint inside every completed baseline sweep still has `fall_rate = 1.0`
- under the current rule in `CONTEXT.md`, this is a baseline-side protocol failure rather than a
  single heuristic-weight miss

So the repo is blocked first by baseline-side report protocol, not by another local `SC-PPO`
mechanism question.

## Secondary blocker

The current secondary blocker is:

`MuJoCo 目前只支持相对于旧单次 heuristic comparator 的部分迁移读数`

Reason:

- `SC-PPO 3.8` still looks stronger than the previous single-run heuristic comparator on task
  stability and velocity tracking in `MuJoCo isaac_mainline`
- but the current `MuJoCo` smoothness metrics still do not reverse in the same direction
- and this comparator is now provisional because the rough-terrain formal baseline protocol has not
  yet produced a report-grade anchor

## Tertiary blocker

The current tertiary blocker is:

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

Completed rough-terrain formal baseline refresh:

- `Vanilla PPO (formal compare)`:
  - `selected checkpoints = 0 / 0 / 0`
  - `velocity_tracking_error_mean = 1.3321 +- 0.1181`
  - `joint_acceleration_l2_mean = 83.7179 +- 13.3692`
  - `action_jitter_l2_mean = 0.0161 +- 0.0008`
  - `episode_return_mean = 4.0002 +- 0.4323`
  - `fall_rate = 1.0000 +- 0.0000`
- `heuristic smoothing action_rate = -0.0005 (formal compare)`:
  - `selected checkpoints = 0 / 0 / 0`
  - `velocity_tracking_error_mean = 1.3451 +- 0.1269`
  - `joint_acceleration_l2_mean = 83.7119 +- 14.9052`
  - `action_jitter_l2_mean = 0.0161 +- 0.0009`
  - `episode_return_mean = 4.1998 +- 0.4037`
  - `fall_rate = 1.0000 +- 0.0000`
- `heuristic smoothing action_rate = -0.0020 (formal compare)`:
  - `selected checkpoints = 0 / 0 / 0`
  - `velocity_tracking_error_mean = 1.3436 +- 0.1232`
  - `joint_acceleration_l2_mean = 85.5995 +- 13.7253`
  - `action_jitter_l2_mean = 0.0161 +- 0.0009`
  - `episode_return_mean = 4.1811 +- 0.3680`
  - `fall_rate = 1.0000 +- 0.0000`
- `heuristic smoothing action_rate = -0.0050 (formal compare)`:
  - `selected checkpoints = 0 / 0 / 0`
  - `velocity_tracking_error_mean = 1.3359 +- 0.1232`
  - `joint_acceleration_l2_mean = 80.5803 +- 14.6031`
  - `action_jitter_l2_mean = 0.0160 +- 0.0009`
  - `episode_return_mean = 4.1769 +- 0.4080`
  - `fall_rate = 1.0000 +- 0.0000`

## Immediate remediation target

The current first-priority remediation target is:

`先修复 formal-compare 协议，再谈 report-grade heuristic anchor`

Execution-facing notes:

- do not keep expanding the same bounded action-rate search as if the issue were still only anchor
  selection
- use the current artifact set to decide whether the frozen formal-compare regime itself needs
  revision before any new report-grade baseline run
- record `Vanilla PPO` collapse as raw-reference evidence and do not spend repair budget on it
- do not advance to new report-grade `MuJoCo` claims or new algorithm branches before a task-valid
  formal anchor exists

## Canonical references

- [SC-PPO report-grade status](./sc-ppo-report-status.md)
- [rough-terrain formal comparison](./baselines/rough-terrain-formal-comparison.md)
- [SC-PPO next-step direction](./sc-ppo-next-step-direction.md)
