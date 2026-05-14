# Shared Smooth-Control Evaluation

Issue `#2` standardizes one evaluation entrypoint for all study methods:

```bash
python scripts/baseline/evaluate_policy.py --config <config-path>
```

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
