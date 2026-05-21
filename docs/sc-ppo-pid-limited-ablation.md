# SC-PPO PID-Limited Ablation

This note closes the repo's `PID有限消融` branch for issue `#6`.

The purpose is narrow: compare the current `PID-Lagrangian正式方案` against a matched
`普通对偶上升` diagnostic inside `SC-PPO`. It is mechanism support for why the formal mainline uses
PID-style multiplier feedback. It is not a broad `组件消融验证`, and it does not create a new
mainline result.

## Current Decision

The `PID有限消融` branch is closed as a bounded mechanism diagnostic:

- keep repaired `SC-PPO threshold = 3.8` with `PID-Lagrangian` as the formal mainline
- record `普通对偶上升` as a failed task-validity diagnostic at the matched threshold
- do not expand this into a multi-component ablation matrix
- do not change the main Isaac rough-terrain or aligned `MuJoCo isaac_mainline` report claims

## Formal Mainline Reference

The current formal mainline remains:

- config: `configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json`
- method: `PID-Lagrangian正式方案`
- selected checkpoints: `seed11 -> 300`, `seed17 -> 300`, `seed23 -> 400`

Selected-checkpoint aggregate over seeds `11`, `17`, and `23`:

- `velocity_tracking_error_mean = 0.6412 +- 0.0554`
- `joint_acceleration_l2_mean = 115.9079 +- 6.9386`
- `action_jitter_l2_mean = 0.2205 +- 0.0017`
- `episode_return_mean = 100.2838 +- 2.7150`
- `fall_rate = 0.1000 +- 0.0000`

This is the line that carries the repo's Isaac-side `方法优于启发式` result.

## Plain-Dual Diagnostic

The minimal matched diagnostic uses:

- config: `configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_dual_001.json`
- load run:
  `.external/humanoid-gym/logs/ecolab_sc_ppo_dual_probe/May14_10-44-08_sc_ppo_threshold_38_lambda_05_quantile_090_dual_001_rough_terrain_iter100`
- checkpoint: `100`
- evaluation: `32 envs`, `20 episodes`, `cuda:0`

Reproduction command:

```bash
PATH=/TinyNAS2024/zhuoxiang/sco-humanoid/bin:$PATH \
/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/baseline/evaluate_checkpoint_sweep.py \
  --config configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_dual_001.json \
  --run-name sc_ppo_threshold_38_lambda_05_quantile_090_dual_001_rough_terrain_iter100 \
  --load-run May14_10-44-08_sc_ppo_threshold_38_lambda_05_quantile_090_dual_001_rough_terrain_iter100 \
  --checkpoints 100 \
  --num-envs 32 \
  --episodes 20 \
  --rl-device cuda:0 \
  --sim-device cuda:0
```

The `PATH` prefix is intentional. The venv contains `ninja`, `c++`, and `g++`; launching the venv
Python by absolute path without adding the venv `bin/` directory leaves PyTorch's C++ extension
loader unable to find those tools.

Observed diagnostic result:

- `selection_status = all_checkpoints_collapsed`
- `velocity_tracking_error_mean = 1.1646`
- `joint_acceleration_l2_mean = 121.3371`
- `action_jitter_l2_mean = 0.1661`
- `episode_return_mean = 4.7101`
- `fall_rate = 1.0000`
- `policy_local_sensitivity_cost_mean = 3.8059`
- `constraint_violation_rate = 0.4091`

Local raw outputs:

- `artifacts/methods/sc_ppo_dual_probe/sc_ppo_threshold_38_lambda_05_quantile_090_dual_001_rough_terrain_iter100/metrics.json`
- `artifacts/methods/sc_ppo_dual_probe/sc_ppo_threshold_38_lambda_05_quantile_090_dual_001_rough_terrain_iter100/checkpoint_sweep_summary.json`

Tracked compact summary:

- `artifacts/analysis/sc_ppo_pid_limited_ablation/summary.json`

## Interpretation

The plain-dual diagnostic does not clear the task floor. Its lower `action_jitter_l2_mean` is not
useful as a headline smoothness win because the policy collapses with `fall_rate = 1.0` and very low
return.

This supports the current algorithm boundary:

- `PID-Lagrangian正式方案` remains the formal `SC-PPO` method
- `普通对偶上升` is not promoted to a formal candidate
- `PID有限消融` is sufficient as mechanism support, not as full component attribution

## What This Does Not Claim

This note does not claim:

- that every PID term is independently necessary
- that every dual learning rate has been exhausted
- that the plain-dual family can never be repaired
- that the final report needs a new algorithm table

The only defended conclusion is narrower: under the matched threshold diagnostic available in this
repo, `普通对偶上升` is not task-valid, so the formal mainline should stay with
`PID-Lagrangian正式方案`.
