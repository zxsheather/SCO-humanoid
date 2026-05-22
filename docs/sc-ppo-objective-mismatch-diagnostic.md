# SC-PPO Objective-Mismatch Diagnostic

This note records the first post-freeze instrumentation pass for issue `#19`:

`Diagnose objective mismatch between local-sensitivity constraint and task-validity floor`

This is not yet a new mechanism branch. It is a `诊断支线` that adds the minimum measurement needed
to tell whether `约束层机制证据` is moving independently from `任务守底线`.

## Instrumentation added

The branch extends `scripts/baseline/evaluate_checkpoint_sweep.py` so each evaluated checkpoint can
now expose:

- eval-side reward-independent metrics already used by the repo:
  - `fall_rate`
  - `velocity_tracking_error_mean`
  - `joint_acceleration_l2_mean`
  - `action_jitter_l2_mean`
  - `episode_return_mean`
- eval-side local-sensitivity metrics from `metrics.json`
- train-side constraint snapshot loaded directly from the checkpoint's `alg_extra_state_dict`
  `latest_stats`
- a derived `checkpoint_diagnostic_alignment.json` summary containing:
  - per-metric ranges across evaluated checkpoints
  - simple pairwise correlations when the series vary
  - explicit constant-series reasons when correlation is not meaningful
  - a `collapsed_task_floor_diagnostic` flag block

This keeps the repo inside the existing `同尺比较` frame while finally aligning train constraint
signals against eval task-validity signals checkpoint by checkpoint.

## First feedback-loop run

First replayed run:

- config:
  `configs/methods/sc_ppo_threshold_055_lambda_05_quantile_090_pid_lower_bound_clamp_anisotropic_proximal_only_positive_penalty_legacy_guard_probe.json`
- run:
  `anisotropic_constraint_t055_proxonly_pospen_legacyguard_short25_seed11`
- upstream run dir:
  `May22_03-15-40_anisotropic_constraint_t055_proxonly_pospen_legacyguard_short25_seed11`
- checkpoints:
  `0, 25`

Generated artifacts:

- `artifacts/methods/sc_ppo_anisotropic_probe/anisotropic_constraint_t055_proxonly_pospen_legacyguard_short25_seed11/checkpoint_sweep_summary.json`
- `artifacts/methods/sc_ppo_anisotropic_probe/anisotropic_constraint_t055_proxonly_pospen_legacyguard_short25_seed11/checkpoint_diagnostic_alignment.json`

## First reading

What the first pass does show:

- both evaluated checkpoints still have `fall_rate = 1.0`
- the new summary correctly marks `fall_rate` as constant instead of pretending that correlation is
  meaningful
- train-side constraint quantities move:
  - `train_policy_local_sensitivity_cost_mean: 0.2557 -> 0.3895`
  - `train_policy_local_sensitivity_cost_update: 0.3258 -> 0.4858`
  - `train_lagrange_multiplier: 0.4863 -> 0.4094`
- eval-side behavior metrics also move:
  - `velocity_tracking_error_mean: 1.5446 -> 1.4552`
  - `joint_acceleration_l2_mean: 76.7667 -> 86.9346`
  - `action_jitter_l2_mean: 0.0159 -> 0.0478`
  - `eval policy_local_sensitivity_cost_mean: 0.3457 -> 0.5086`

What it does not yet show:

- it does not yet isolate a clean `objective mismatch` case where train constraint evidence improves
  while the task floor stays collapsed
- on this two-checkpoint run, several smoothness-related signals actually worsen together while
  `fall_rate` remains pinned at `1.0`

So the first pass should be read as:

`instrumentation success, diagnosis still open`

## Updated next step

The next useful diagnostic is not more support-set tuning. It is to run the same alignment summary
on a richer checkpoint neighborhood where the branch has more than two checkpoints and where at
least one behavior-layer metric meaningfully separates.

Candidate next moves:

- rerun a reduced-budget `SC-PPO` diagnostic with denser checkpoint saves
- or replay an existing longer `SC-PPO` checkpoint neighborhood through the same alignment summary

Only after that should the branch claim `constraint target mismatch`, `aggregation mismatch`, or
`optimization-timescale mismatch`.
