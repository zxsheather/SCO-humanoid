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

## Second feedback-loop run

To increase diagnostic value, the same alignment summary was replayed on the current `SC-PPO 3.8`
mainline seed-11 checkpoint neighborhood:

- config: `configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json`
- artifact run: `sc_ppo_threshold_38_objective_mismatch_seed11`
- upstream run dir:
  `May14_13-38-03_sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11`
- checkpoints: `0, 100, 200, 300, 400`

Generated artifacts:

- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_objective_mismatch_seed11/checkpoint_sweep_summary.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_objective_mismatch_seed11/checkpoint_diagnostic_alignment.json`

## Second reading

This second pass is much more informative than the two-checkpoint collapsed run.

Observed trajectory:

- train constraint cost grows strongly across training:
  - `train_policy_local_sensitivity_cost_mean: 0.2951 -> 3.4390`
  - `train_policy_local_sensitivity_cost_update: 0.3082 -> 3.7525`
- task validity improves at the same time:
  - `fall_rate: 1.0 -> 0.2`
  - `velocity_tracking_error_mean: 1.4031 -> 0.6012`
  - `episode_return_mean: 4.2785 -> 102.8265`
- behavior-layer smoothness gets worse in the same direction as the higher local-sensitivity cost:
  - `action_jitter_l2_mean: 0.0116 -> 0.2629`
  - `joint_acceleration_l2_mean: 68.7516 -> 144.6384` with a higher peak at checkpoint `200`

The resulting alignment summary therefore shows:

- `train_policy_local_sensitivity_cost_mean` is negatively correlated with `fall_rate`
  (`pearson ~= -0.52`)
- `train_policy_local_sensitivity_cost_mean` is negatively correlated with
  `velocity_tracking_error_mean` (`pearson ~= -0.79`)
- `train_policy_local_sensitivity_cost_mean` is positively correlated with `action_jitter_l2_mean`
  (`pearson ~= 0.92`)
- `train_policy_local_sensitivity_cost_mean` is positively correlated with
  `joint_acceleration_l2_mean` (`pearson ~= 0.77`)

Interpretation:

- this is not a `no-signal` failure where the constraint metric moves independently of everything
- instead, the current local-sensitivity metric appears to move in the same direction as
  behavior-layer roughness while moving against task-validity improvement
- that is stronger evidence for `objective tension / task-floor mismatch` than for a pure
  `measurement disconnected from behavior` story

The `300/400` selector neighborhood makes the same tradeoff visible:

- both checkpoints satisfy the current task floor
- `400` has slightly better velocity tracking and higher return
- `300` has lower `action_jitter_l2_mean`, lower `joint_acceleration_l2_mean`, and lower train
  local-sensitivity cost

So the branch's current best reading is:

`the local-sensitivity objective still tracks behavior-layer roughness, but the current training path appears to spend more local sensitivity in order to buy task validity and tracking quality`

## Updated next step

The next useful diagnostic is not more support-set tuning. It is to distinguish whether that
observed tension comes primarily from:

- the constraint target itself
- the aggregation rule
- or the optimization timescale / multiplier dynamics

Candidate next moves:

- replay the same alignment summary on seed `17` and seed `23` of the `SC-PPO 3.8` mainline
- compare the same checkpoint neighborhood against one collapsed control such as plain dual ascent
- if the pattern persists, open one bounded mechanism test that changes the objective or the update
  timescale without reopening broad threshold search

Only after that should the branch claim `constraint target mismatch`, `aggregation mismatch`, or
`optimization-timescale mismatch`.
