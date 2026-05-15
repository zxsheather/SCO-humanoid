# SC-PPO Full-Batch Threshold Tightening Probe

This is a `诊断支线`, not a replacement for the repo's current formal `3-seed` mainline.

## Purpose

The completed `full_batch + threshold = 3.8 + 400 iteration` run changed the failure picture:

- `任务稳定性` and `速度跟踪` improved materially in both Isaac and `MuJoCo isaac_mainline`
- but `行为层平滑指标` still did not beat the selected heuristic anchor
- the best checkpoint moved to the middle of training instead of the final checkpoint
- the training-side multiplier still decayed back toward the lower bound late in training

The next narrow question is therefore:

`如果保持 full_batch 约束采样不变，只进一步收紧 threshold，能否在不丢掉当前任务优势的前提下，把 MuJoCo 行为层平滑指标往 heuristic 靠近？`

## Probe variants

- `configs/methods/sc_ppo_threshold_37_lambda_05_quantile_090_pid_lower_bound_clamp_full_batch.json`
- `configs/methods/sc_ppo_threshold_36_lambda_05_quantile_090_pid_lower_bound_clamp_full_batch.json`

Shared settings:

- `algorithm.constraint.subsample_obs = 0`
- `algorithm.constraint.cost_aggregation = quantile`
- `algorithm.constraint.cost_quantile = 0.9`
- `algorithm.constraint.lambda_init = 0.5`
- `algorithm.constraint.pid_integral_mode = lower_bound_clamp`
- `runner.save_interval = 50`

The smaller `save_interval` is intentional. The preceding `threshold = 3.8` run selected `checkpoint 300`
over `checkpoint 400`, so this branch should not rely on coarse `100-iteration` checkpoint spacing.

## Reading rule

Treat this branch as successful only if it preserves the current `任务守底线` while improving the
`MuJoCo isaac_mainline` `行为层平滑指标`.

Do not promote it on the basis of:

- training-side constraint traces alone
- Isaac-only gains
- final-checkpoint-only reading without `checkpoint_sweep`
