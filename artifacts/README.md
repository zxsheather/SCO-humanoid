# Artifacts

This directory stores runtime outputs for reproducibility checks and experiment comparisons.

Shared smooth-control evaluation layout:

- `artifacts/baselines/vanilla_ppo/<run_name>/exported/policies/policy_1.pt`
- `artifacts/baselines/vanilla_ppo/<run_name>/metrics.json`
- `artifacts/baselines/vanilla_ppo/<run_name>/manifest.json`

The standardized `metrics.json` schema for issue `#2` includes:

- `velocity_tracking_error_mean` and `velocity_tracking_error_std`
- `fall_rate`
- `joint_acceleration_l2_mean` and `joint_acceleration_l2_std`
- `action_jitter_l2_mean` and `action_jitter_l2_std`
- `episode_return_mean` and `episode_return_std`
- optional nested `constraint_metrics`

Constraint-side logging can additionally populate:

- `constraint_metrics.json`
- `lagrange_multiplier_trace.json`
