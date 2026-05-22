# SC-PPO Behavior Smoothness Metric Diagnostic

This note records the post-freeze branch for issue `#29`.

The branch question is narrower than a new algorithm line:

`Do independent trace-based behavior smoothness metrics move in the same direction as the repo's current local-sensitivity reading?`

## Purpose

The repo already has:

- `policy_local_sensitivity` mechanism-side logging
- shared behavior metrics such as `fall_rate`, `velocity_tracking_error_mean`,
  `joint_acceleration_l2_mean`, and `action_jitter_l2_mean`

What it previously lacked was an independent trace-based smoothness family.

This branch adds:

- optional compact episode-trace capture during Isaac evaluation
- offline trace-based smoothness summaries
- first bounded replay evidence on existing checkpoints

## Current Implementation

Added components:

- `scripts/baseline/_behavior_trace_metrics.py`
- `scripts/analysis/compute_behavior_smoothness_metrics.py`
- trace-capture support in `scripts/baseline/evaluate_policy.py`
- trace forwarding plus per-checkpoint smoothness artifacts in
  `scripts/baseline/evaluate_checkpoint_sweep.py`
- unit coverage:
  `tests/test_behavior_trace_metrics.py`
  and extended `tests/test_checkpoint_sweep_recovery.py`

Current trace-based metrics:

- `joint_position_ldlj_mean`
- `joint_velocity_sparc_mean`

## Artifact Shape

When trace capture is enabled, checkpoint replay now writes:

- `episode_traces_checkpoint_<N>.json`
- `behavior_smoothness_metrics_checkpoint_<N>.json`
- `behavior_smoothness_metrics_selected.json`

These live next to the usual per-checkpoint metrics under the method artifact directory.

## First Real Artifact Check

Single-checkpoint replay on existing `SC-PPO 3.8` mainline artifacts:

- config:
  `configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json`
- run:
  `sc_ppo_threshold_38_behavior_trace_sweep_seed11_ckpt300`
- upstream run:
  `May14_13-38-03_sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11`
- checkpoint: `300`

Observed first trace summary:

- `velocity_tracking_error_mean = 0.6791`
- `joint_acceleration_l2_mean = 151.9753`
- `action_jitter_l2_mean = 0.2549`
- `policy_local_sensitivity_cost_mean = 3.6770`
- `joint_position_ldlj_mean = -28.7675`
- `joint_velocity_sparc_mean = -24.2151`

## First Matched Pair

The first bounded matched pair compares one `seed11` selected checkpoint from each completed Isaac-side
mainline:

1. `SC-PPO 3.8`, checkpoint `300`
2. revised heuristic anchor, checkpoint `350`

Reproduction commands:

```bash
PATH=/TinyNAS2024/zhuoxiang/sco-humanoid/bin:$PATH \
/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/baseline/evaluate_checkpoint_sweep.py \
  --config configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json \
  --run-name sc_ppo_threshold_38_behavior_trace_sweep_seed11_ckpt300 \
  --load-run May14_13-38-03_sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11 \
  --checkpoints 300 \
  --episodes 1 \
  --num-envs 16 \
  --capture-traces \
  --trace-max-episodes 1 \
  --trace-max-steps 512 \
  --humanoid-gym-root /home/zhuoxiang/SCO-humanoid/.external/humanoid-gym \
  --rl-device cuda:0 \
  --sim-device cuda:0
```

```bash
PATH=/TinyNAS2024/zhuoxiang/sco-humanoid/bin:$PATH \
/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/baseline/evaluate_checkpoint_sweep.py \
  --config configs/methods/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget.json \
  --run-name heuristic_behavior_trace_sweep_seed11_ckpt350 \
  --load-run May21_03-55-49_heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed11 \
  --checkpoints 350 \
  --episodes 1 \
  --num-envs 16 \
  --capture-traces \
  --trace-max-episodes 1 \
  --trace-max-steps 512 \
  --humanoid-gym-root /home/zhuoxiang/SCO-humanoid/.external/humanoid-gym \
  --rl-device cuda:0 \
  --sim-device cuda:0
```

Observed reading:

`SC-PPO 3.8`, checkpoint `300`:

- `velocity_tracking_error_mean = 0.6791`
- `joint_acceleration_l2_mean = 151.9753`
- `action_jitter_l2_mean = 0.2549`
- `episode_return_mean = 69.9022`
- `policy_local_sensitivity_cost_mean = 3.6770`
- `joint_position_ldlj_mean = -28.7675`
- `joint_velocity_sparc_mean = -24.2151`

Revised heuristic anchor, checkpoint `350`:

- `velocity_tracking_error_mean = 0.9265`
- `joint_acceleration_l2_mean = 138.9144`
- `action_jitter_l2_mean = 0.2954`
- `episode_return_mean = 45.4078`
- `policy_local_sensitivity_cost_mean = 7.2038`
- `joint_position_ldlj_mean = -28.3102`
- `joint_velocity_sparc_mean = -23.5264`

## Current Reading

This first matched pair is intentionally weak evidence:

- only `1` captured episode per method
- trace length is truncated to `512` steps
- both quick replays still end with `fall_rate = 1.0`

So this is not a report-grade smoothness comparison.

But it is already useful as a direction check:

- the current local-sensitivity and action-jitter readings favor `SC-PPO 3.8`
- the first `LDLJ / SPARC` readings slightly favor the revised heuristic
- joint acceleration also slightly favors the revised heuristic in this probe

So the new trace-based metrics do **not** trivially collapse onto the current local-sensitivity ordering.
At minimum, they complicate the current smoothness story and justify a slightly richer checkpoint-neighborhood
replay before making any stronger alignment claim.

## Seed11 Revised-Heuristic Neighborhood

The next bounded replay was run on the revised heuristic `seed11` neighborhood:

- run:
  `heuristic_behavior_trace_neighborhood_seed11_300_350_400`
- upstream run:
  `May21_03-55-49_heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed11`
- checkpoints: `300`, `350`, `400`

Observed reading:

Checkpoint `300`:

- `fall_rate = 1.0`
- `velocity_tracking_error_mean = 0.6407`
- `joint_acceleration_l2_mean = 102.0995`
- `action_jitter_l2_mean = 0.2482`
- `episode_return_mean = 66.0412`
- `policy_local_sensitivity_cost_mean = 7.2017`
- `joint_position_ldlj_mean = -28.2216`
- `joint_velocity_sparc_mean = -23.6050`

Checkpoint `350`:

- `fall_rate = 1.0`
- `velocity_tracking_error_mean = 0.9265`
- `joint_acceleration_l2_mean = 138.9144`
- `action_jitter_l2_mean = 0.2954`
- `episode_return_mean = 45.4078`
- `policy_local_sensitivity_cost_mean = 7.2038`
- `joint_position_ldlj_mean = -28.3102`
- `joint_velocity_sparc_mean = -23.5264`

Checkpoint `400`:

- `fall_rate = 1.0`
- `velocity_tracking_error_mean = 0.6618`
- `joint_acceleration_l2_mean = 95.4117`
- `action_jitter_l2_mean = 0.2501`
- `episode_return_mean = 83.1079`
- `policy_local_sensitivity_cost_mean = 7.8296`
- `joint_position_ldlj_mean = -28.4820`
- `joint_velocity_sparc_mean = -20.5100`

Interpretation boundary:

- this replay is still collapse-only evidence because all three quick replays ended with `fall_rate = 1.0`
- the script therefore records `selection_status = all_checkpoints_collapsed` and picks checkpoint `400`
  only by the fallback composite score

Even with that limitation, the neighborhood is already informative:

- checkpoint `350` is dominated by both older shared metrics and the new trace metrics
- checkpoint `300` has the best `velocity_tracking_error_mean`, `action_jitter_l2_mean`, and
  `joint_position_ldlj_mean`
- checkpoint `400` has the best `joint_acceleration_l2_mean`, `episode_return_mean`, and
  `joint_velocity_sparc_mean`
- `policy_local_sensitivity_cost_mean` slightly prefers `300 / 350` over `400`

So the richer bounded replay strengthens the same broad conclusion:

- the new trace metrics are not reducible to a single existing smoothness proxy
- even within one method line, `LDLJ` and `SPARC` can disagree with each other
- the repo is likely observing multiple smoothness dimensions rather than one scalar notion

## Next Step

The next useful move is narrower than more training:

- keep the trace pipeline as-is
- either replay a tiny cross-seed selected-checkpoint set for both lines or stop here and open the PR
- frame the branch claim as a diagnostic result:
  trace-based smoothness metrics complicate, rather than confirm, the current local-sensitivity story

Do not promote this branch to new training budgets, formal-comparison reruns, or product claims.
