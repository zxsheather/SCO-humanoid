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

## Current MuJoCo status

The repo now also has a working `MuJoCo关键两组终验` entrypoint through:

- `scripts/baseline/evaluate_mujoco_sim2sim.py`

Current `MuJoCo` reading should be split into two layers rather than collapsed into one headline.

### Minimal comparable first pass

Under the current `terrain_mode = isaac_mainline` protocol
(`plane + joint_reset_noise = 0.1 + 20 episodes + 20 seconds` in the current repo state):

- heuristic anchor:
  - `velocity_tracking_error_mean = 0.6811 ± 0.1113`
  - `joint_acceleration_l2_mean = 110.2715 ± 13.0420`
  - `action_jitter_l2_mean = 0.2005 ± 0.0158`
  - `fall_rate = 0.7000`
  - `episode_steps_mean = 962.9`
- `SC-PPO threshold = 3.8` mainline:
  - `velocity_tracking_error_mean = 0.6206 ± 0.0458`
  - `joint_acceleration_l2_mean = 154.4672 ± 12.0365`
  - `action_jitter_l2_mean = 0.2785 ± 0.0150`
  - `fall_rate = 0.0500`
  - `episode_steps_mean = 1954.35`

Interpretation:

- `SC-PPO` currently shows materially better task stability and velocity tracking in `MuJoCo`
- however, the current smoothness metrics do not transfer in the same direction
- at this stage, `MuJoCo` supports `部分迁移`, not a full cross-engine smoothness win

Canonical formal artifacts for this protocol are now:

- heuristic:
  `artifacts/methods/heuristic_smoothing_sweep/heuristic_smoothing_action_rate_0050_rough_terrain/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `SC-PPO threshold = 3.8` representative checkpoint:
  `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- these should be cited instead of the older `plane_20ep_20s_noise01`-named duplicates

### Terrain stress status

Under the current `terrain_mode = hfield_stress`
(`terrain + joint_reset_noise = 0.1 + 5 episodes + 5 seconds`) probe:

- heuristic:
  - `velocity_tracking_error_mean = 1.1758 ± 0.3709`
  - `joint_acceleration_l2_mean = 225.1939 ± 119.3916`
  - `action_jitter_l2_mean = 0.2921 ± 0.0742`
  - `fall_rate = 1.0000`
  - `episode_steps_mean = 123.8`
- `SC-PPO checkpoint 300`:
  - `velocity_tracking_error_mean = 1.2795 ± 0.3210`
  - `joint_acceleration_l2_mean = 296.9754 ± 58.4883`
  - `action_jitter_l2_mean = 0.3663 ± 0.0538`
  - `fall_rate = 1.0000`
  - `episode_steps_mean = 129.0`

Additional `SC-PPO` MuJoCo terrain checkpoint probes at `200`, `300`, and `400` all still fail the
same way, so the current terrain issue should not be summarized as a simple selected-checkpoint
mismatch.

### Terrain repair-stage intermediate status

Under the current `terrain_mode = hfield_moderate`
(`hfield_size_override = [50.0, 50.0, 0.06, 0.02]`, `joint_reset_noise = 0.1`, `5 episodes`,
`5 seconds`) probe:

- heuristic:
  - `velocity_tracking_error_mean = 1.2872`
  - `joint_acceleration_l2_mean = 407.8357`
  - `action_jitter_l2_mean = 0.2904`
  - `fall_rate = 1.0000`
  - `episode_steps_mean = 134.6`
- `SC-PPO checkpoint 300`:
  - `velocity_tracking_error_mean = 1.3863`
  - `joint_acceleration_l2_mean = 500.5605`
  - `action_jitter_l2_mean = 0.3388`
  - `fall_rate = 0.4000`
  - `episode_steps_mean = 345.0`

Interpretation:

- this is the first repaired terrain-stage protocol in which `SC-PPO` is no longer collapsing as
  completely as in `hfield_stress`
- however, the current `行为层平滑指标` remain very poor, so `hfield_moderate` is only a
  `repair-stage intermediate protocol`, not a report-grade terrain validation line

Protocol repair note:

- the repo has now made this split explicit in the evaluator itself
- `isaac_mainline` is the comparable replay line
- `hfield_moderate` is the current repair-stage intermediate line
- `hfield_stress` is a separate transfer-pressure line
- the old boolean `terrain` switch should no longer be read as “the MuJoCo counterpart of the
  Isaac main task”

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
- current `MuJoCo isaac_mainline + noise` evidence does not yet support a full smoothness-transfer claim
- current `MuJoCo terrain` evidence is still too unstable to serve as the repo's main external
  validation result
- the new `hfield_moderate` line improves failure discrimination but is not yet physically clean
  enough to replace either `isaac_mainline` or `hfield_stress`

So the current evidence supports:

- `3.8` is a meaningful operating point
- `SC-PPO` can retain a task-stability advantage in a first-pass `MuJoCo` protocol

The current evidence does not yet support:

- any claim that a broad range of nearby thresholds all work equally well
- any claim that the final checkpoint alone is enough for long-budget reporting
- any claim that the current smoothness advantage fully transfers to `MuJoCo`
- any claim that the current `MuJoCo terrain` protocol is ready to stand in for the final
  cross-engine comparison

## Recommended next step

The next highest-value step is no longer another tiny local threshold sweep.

The better next move is one of:

1. freeze the current `Isaac mainline + MuJoCo isaac_mainline first pass` result for reporting
2. continue treating `MuJoCo terrain` as a separate protocol-repair line rather than as the
   current main external result
3. use `hfield_moderate` as the current repair-stage intermediate protocol while deciding whether
   the next move should be terrain-side repair or algorithm-side robustness work
