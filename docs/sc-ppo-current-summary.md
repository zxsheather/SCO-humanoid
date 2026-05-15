# SC-PPO Current Summary

This note summarizes the current repo state for the `受限优化与平滑性增强` direction in
[goal.md](/home/zhuoxiang/ECOLab/goal.md:1).

## Current conclusion

The repo now has a working `SC-PPO` branch based on repaired `PID-Lagrangian` constraint updates
with:

- `pid_integral_mode = lower_bound_clamp`
- `cost_aggregation = quantile(0.90)`
- `lambda_init = 0.5`
- current mainline `threshold = 3.8`

On the `粗糙平面` task, this mainline has completed a `3-seed, 400 iteration, checkpoint-sweep`
comparison and currently beats the repo's heuristic anchor under the shared metric schema.

Selected-checkpoint aggregate over seeds `11`, `17`, and `23`:

- `velocity_tracking_error_mean = 0.6412 ± 0.0554`
- `joint_acceleration_l2_mean = 115.9079 ± 6.9386`
- `action_jitter_l2_mean = 0.2205 ± 0.0017`
- `episode_return_mean = 100.2838 ± 2.7150`
- `fall_rate = 0.1000 ± 0.0000`

Heuristic anchor for comparison:

- `velocity_tracking_error_mean = 1.1381`
- `joint_acceleration_l2_mean = 140.6399`
- `action_jitter_l2_mean = 0.2457`
- `fall_rate = 1.0`

## What this means

- the repo is no longer blocked on the old question “can the repaired PID branch become
  behaviorally competitive”
- the answer for the current `粗糙平面` setup is now yes
- the result is not just a one-seed artifact, because it survives a `3-seed` batch
- however, the result still depends on `checkpoint sweep + selected checkpoint` reporting rather
  than the final checkpoint alone

## What has been ruled out

- the old `PID` branch failed because of negative integral debt locking the multiplier at the lower
  bound
- `lower_bound_clamp` repaired that mechanism failure
- a nearby repaired-`4.0` control does not show the same stability
- one `4.0` seed selects `checkpoint 0`
- repaired-`4.0` aggregate `fall_rate = 0.4667 ± 0.3793`
- repaired-`4.0` aggregate variance is much larger than the `3.8` mainline

So the current evidence supports:

- `3.8` is a meaningful operating point

The current evidence does not yet support:

- any claim that a broad range of nearby thresholds all work equally well
- any claim that the final checkpoint alone is enough for long-budget reporting
- any claim yet about harder terrain or MuJoCo transfer

## Recommended next step

The next highest-value step is no longer another tiny local threshold sweep.

The better next move is one of:

1. freeze the current `3.8` result and use it as the repo's current algorithm result for reporting
2. expand validation outward to a harder task condition or a stronger external check
3. only reopen local tuning if new evidence shows the current mainline is not stable enough for the
   intended report claim
