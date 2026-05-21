# SC-PPO Current Summary

This note summarizes the current repo state for the `受限优化与平滑性增强` direction in
[goal.md](../goal.md).

## Current conclusion

The repo still has one strong completed method line:

- repaired `SC-PPO`
- `threshold = 3.8`
- `PID-Lagrangian`
- `pid_integral_mode = lower_bound_clamp`
- `cost_aggregation = quantile(0.90)`

Its selected-checkpoint aggregate over seeds `11`, `17`, and `23` remains:

- `velocity_tracking_error_mean = 0.6412 +- 0.0554`
- `joint_acceleration_l2_mean = 115.9079 +- 6.9386`
- `action_jitter_l2_mean = 0.2205 +- 0.0017`
- `episode_return_mean = 100.2838 +- 2.7150`
- `fall_rate = 0.1000 +- 0.0000`

The baseline-side reading is now different from the earlier repair phase:

- the frozen formal compare under `64 envs x 400 iterations` still records universal collapse for
  `Vanilla PPO` and the bounded heuristic family
- the repaired-budget probe under `512 envs x 200 iterations` narrowed the old heuristic winner to
  `0 / 0 / 200` and justified explicit protocol revision
- the completed revised-protocol long-budget run under `512 envs x 400 iterations` on
  `action_rate = -0.0050` now yields `selected checkpoints = 350 / 300 / 350`

So the repo is no longer blocked by `whether any heuristic formal anchor exists at all`.
It now has a usable revised heuristic formal anchor for the rough-terrain `三组正式对比`.

Current revised heuristic anchor aggregate over seeds `11`, `17`, and `23`:

- `velocity_tracking_error_mean = 0.7549 +- 0.1068`
- `joint_acceleration_l2_mean = 119.8639 +- 2.1966`
- `action_jitter_l2_mean = 0.2711 +- 0.0084`
- `episode_return_mean = 100.9327 +- 11.2711`
- `fall_rate = 0.1500 +- 0.0816`

Compared against that revised anchor, `SC-PPO 3.8` remains stronger on the shared Isaac rough-
terrain schema:

- better `velocity_tracking_error_mean` (`0.6412` vs `0.7549`)
- better `fall_rate` (`0.1000` vs `0.1500`)
- better `joint_acceleration_l2_mean` (`115.9079` vs `119.8639`)
- better `action_jitter_l2_mean` (`0.2205` vs `0.2711`)
- `episode_return_mean` is effectively tied (`100.2838` vs `100.9327`) and remains only a
  `总回报补充指标`

This is enough to restore a defendable Isaac-side `方法优于启发式` reading.

## Current MuJoCo status

The current `MuJoCo isaac_mainline` read is now aligned with the refreshed revised heuristic formal
anchor.

Aligned `MuJoCo isaac_mainline` aggregate over seeds `11`, `17`, and `23`:

- revised heuristic anchor:
  - `velocity_tracking_error_mean = 0.4188 +- 0.0398`
  - `joint_acceleration_l2_mean = 120.7339 +- 2.6413`
  - `action_jitter_l2_mean = 0.2452 +- 0.0288`
  - `fall_rate = 0.0000 +- 0.0000`
  - `episode_steps_mean = 2000.0 +- 0.0`
- `SC-PPO threshold = 3.8`:
  - `velocity_tracking_error_mean = 0.4910 +- 0.0944`
  - `joint_acceleration_l2_mean = 125.5411 +- 21.1683`
  - `action_jitter_l2_mean = 0.2313 +- 0.0351`
  - `fall_rate = 0.0167 +- 0.0236`
  - `episode_steps_mean = 1984.7833 +- 21.5196`

Interpretation:

- the revised heuristic anchor is better on task stability, velocity tracking, episode length, and
  joint acceleration in this aligned replay
- `SC-PPO 3.8` is only better on `action_jitter_l2_mean`
- so the previous `SC-PPO has partial-transfer advantage in MuJoCo isaac_mainline` wording should
  be retired

## What this means

The repo is no longer in a state where more tiny threshold promotions are the highest-value move.
It is also no longer in a state where another bounded heuristic weight search is the right
immediate action.

The mainline evidence closure is now complete at the current claim boundary:

- Isaac rough-terrain can be reported around `Vanilla PPO` raw reference, the revised heuristic
  anchor, and `SC-PPO 3.8`
- `MuJoCo isaac_mainline` should be reported as `混合外部验证结论`, not as an `SC-PPO` transfer
  advantage
- `MuJoCo terrain` remains a separate protocol-repair line

The immediate execution order is now:

1. keep `README.md`, `CONTEXT.md`, GitHub Issues, and the report drafts aligned on the completed
   claim boundary
2. keep `PID有限消融` closed as mechanism support and keep the `SN-only` replacement branch closed
   as a negative feasibility diagnostic
3. advance `#7 随机阶梯` only as a bounded `复杂地形条件` stress test, not as a new headline method
   line or a rewrite of the rough-terrain main claim

## Detailed references

- [SC-PPO report-grade status](./sc-ppo-report-status.md)
- [rough-terrain formal comparison](./baselines/rough-terrain-formal-comparison.md)
- [rough-terrain formal protocol repair probe](./baselines/rough-terrain-formal-protocol-repair-probe.md)
- [rough-terrain formal protocol revision decision](./baselines/rough-terrain-formal-protocol-revision-decision.md)
- [rough-terrain formal protocol revision long-budget test](./baselines/rough-terrain-formal-protocol-revision-long-budget.md)
- [SC-PPO MuJoCo revised-anchor aligned comparison](./sc-ppo-mujoco-revised-anchor-aligned-comparison.md)
- [SC-PPO current blockers](./sc-ppo-current-blockers.md)
- [SC-PPO next-step direction](./sc-ppo-next-step-direction.md)
- [SC-PPO PID-limited ablation](./sc-ppo-pid-limited-ablation.md)
- [SC-PPO SN feasibility diagnostic](./sc-ppo-sn-feasibility-diagnostic.md)
