# Shared Smooth-Control Evaluation

Issue `#2` standardizes one evaluation entrypoint for all study methods:

```bash
python scripts/baseline/evaluate_policy.py --config <config-path>
```

For longer-budget runs where `最后一个 checkpoint` may not be the best stopping point, use:

```bash
python scripts/baseline/evaluate_checkpoint_sweep.py --config <config-path> --run-name <artifact-run-name> --load-run <upstream-run-dir>
```

This writes:

- `metrics_checkpoint_<N>.json` for each evaluated checkpoint
- `checkpoint_sweep_summary.json` with a sortable table and selected best checkpoint
- `metrics_selected.json` for the selected checkpoint metrics snapshot

Long-budget reporting rule:

- cite `checkpoint_sweep_summary.json` and `metrics_selected.json` as the canonical long-budget result
- do not treat the final checkpoint alone as sufficient evidence for the current `SC-PPO` branch

Current mainline note:

- the current leading `SC-PPO` branch is
  `configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json`
- its current best evidence is a completed `3-seed, 400 iteration, checkpoint-sweep` batch
- selected checkpoints are `300`, `300`, and `400` for seeds `11`, `17`, and `23`
- selected-checkpoint aggregate metrics are:
  - `velocity_tracking_error_mean = 0.6412 ± 0.0554`
  - `joint_acceleration_l2_mean = 115.9079 ± 6.9386`
  - `action_jitter_l2_mean = 0.2205 ± 0.0017`
  - `episode_return_mean = 100.2838 ± 2.7150`
  - `fall_rate = 0.1000 ± 0.0000`
- this currently beats the repo's heuristic anchor under the shared metric schema, but it still
  depends on `selected checkpoint` reporting rather than the final checkpoint alone

Current method configs:

- `configs/methods/vanilla_ppo.json`
- `configs/methods/heuristic_smoothing.json`
- `configs/methods/sc_ppo.json`

Legacy note:

- `configs/baselines/vanilla_ppo.json` remains as the original issue `#1` scaffold around the upstream default task setup.

## Common metric schema

Every method writes the same top-level fields in `metrics.json`:

- `velocity_tracking_error_mean` / `velocity_tracking_error_std`
- `fall_rate`
- `joint_acceleration_l2_mean` / `joint_acceleration_l2_std`
- `action_jitter_l2_mean` / `action_jitter_l2_std`
- `episode_return_mean` / `episode_return_std`

This maps to the project glossary as follows:

- `速度跟踪误差主指标` -> `velocity_tracking_error_*`
- `跌倒率底线指标` -> `fall_rate`
- `关节震荡主指标` -> `joint_acceleration_l2_*`
- `动作抖动次级指标` -> `action_jitter_l2_*`
- `总回报补充指标` -> `episode_return_*`

## Constraint-side support

Method configs can enable `evaluation.constraint_logging` to probe or attach:

- `policy_local_sensitivity_cost_mean`
- `policy_local_sensitivity_cost_std`
- `constraint_violation_rate`
- `lagrange_multiplier_trace_path`

If `constraint_metrics.json` exists under the method artifact directory, the evaluator keeps it under
`constraint_metrics.sidecar_metrics` and uses it to fill missing constraint fields.

If `lagrange_multiplier_trace.json` exists, the evaluator records its path in
`constraint_metrics.lagrange_multiplier_trace_path`.

For `SC-PPO`, `scripts/baseline/train_vanilla_ppo.py` now writes these sidecars from training-time
constraint statistics, so evaluation can keep both:

- rollout-time `policy_local_sensitivity_cost_*`
- training-time `lagrange_multiplier` and violation trace evidence
