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

However, the completed issue `#5` rough-terrain formal baseline refresh changed the baseline-side
reading:

- `Vanilla PPO` formal compare selects `checkpoint 0` on all three seeds
- `heuristic smoothing action_rate = -0.0005 / -0.0020 / -0.0050` formal compare also selects
  `checkpoint 0` on all three seeds
- every evaluated checkpoint inside every completed baseline sweep still has `fall_rate = 1.0`

So the repo's current state is:

- `SC-PPO 3.8` is still the strongest completed rough-terrain line
- but the rough-terrain formal baseline side is now a protocol-repair problem, not just an
  unresolved heuristic-anchor pick
- per `CONTEXT.md`, the repo cannot treat the three-way rough-terrain comparison as report-grade
  complete until that baseline protocol is repaired

## Current MuJoCo status

The current `MuJoCo isaac_mainline` first-pass read is still useful, but it should now be read as
a provisional comparison against the previous single-run heuristic comparator.

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

Interpretation:

- `SC-PPO` still shows materially better task stability and velocity tracking in this first-pass
  cross-engine protocol
- the current smoothness metrics still do not transfer in the same direction
- and the baseline side is still provisional until the rough-terrain formal protocol is repaired

## What this means

The repo is no longer in a state where more tiny threshold promotions are the highest-value move.
It is also no longer in a state where another bounded heuristic weight search is the right
immediate action.

The immediate execution order should now be:

1. repair the rough-terrain formal-compare protocol / baseline-side regime
2. re-freeze the rough-terrain report boundary only after a task-valid formal anchor exists
3. keep `MuJoCo isaac_mainline` as a bounded partial-transfer read, not a final smoothness-transfer headline
4. only then return to post-mainline branches such as `SN`

## Detailed references

- [SC-PPO report-grade status](./sc-ppo-report-status.md)
- [rough-terrain formal comparison](./baselines/rough-terrain-formal-comparison.md)
- [SC-PPO current blockers](./sc-ppo-current-blockers.md)
- [SC-PPO next-step direction](./sc-ppo-next-step-direction.md)
