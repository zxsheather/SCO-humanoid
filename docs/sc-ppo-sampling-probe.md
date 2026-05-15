# SC-PPO Constraint Sampling Probe

This is a `诊断支线`, not a replacement for the current `PID-Lagrangian正式方案` mainline.

## Purpose

The current blockers show a persistent `训练-versus-评估 mismatch`:

- training-side `policy_local_sensitivity_cost_*` looks more optimistic
- MuJoCo-side `行为层平滑指标` still fail to transfer cleanly

One plausible cause is that the constraint update currently sees only a very small random subset of
actor observations per PPO minibatch:

- current mainline: `algorithm.constraint.subsample_obs = 8`

This probe changes only the constraint-sampling regime while keeping the current repaired-`3.8`
PID setup fixed.

## Probe variants

- `configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_subsample32.json`
  increases the constraint sample from `8` to `32`
- `configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_full_batch.json`
  removes subsampling entirely by setting `algorithm.constraint.subsample_obs = 0`

Implementation note:

- `subsample_obs = 0` now means `full_batch`
- training artifacts record both `constraint_subsample_obs` and `constraint_sampling_mode`

## Reading rule

This probe is only meant to answer:

`如果约束更新看见更多真实 rollout 观测，乘子参与度和行为层平滑迁移会不会更像评估态？`

It does **not** change the repo's current main claim:

- the mainline remains the repaired `3.8` branch under the current `完全替换对比`
- any positive result here should first be treated as `诊断证据`
