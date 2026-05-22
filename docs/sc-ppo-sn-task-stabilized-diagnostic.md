# SC-PPO SN Task-Stabilized Diagnostic

This note defines the repo's post-freeze `任务稳定化 SN 配方` branch for issue `#25`.

It is intentionally narrow.

It does not reopen the negative `SN-only` replacement line.
It asks one different question:

`Can first-hidden-layer actor spectral normalization coexist with the current task-valid SC-PPO 3.8 recipe without immediately collapsing rough-terrain locomotion?`

## Recipe Definition

Base recipe:

- `configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json`

Hybrid diagnostic recipe:

- `configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_sn_first_hidden.json`

The hybrid keeps the full `SC-PPO 3.8` constraint path and adds only:

- `policy.actor_spectral_norm = true`
- `policy.actor_spectral_norm_output_layer = false`
- `policy.actor_spectral_norm_layer_scope = "first_hidden"`
- `policy.actor_spectral_norm_coeff = 1.0`

Everything else stays aligned with the current task-valid rough-terrain `SC-PPO 3.8` recipe.

## Diagnostic Boundary

This branch must be read as `task-stabilized feasibility`, not as a new mainline claim.

It must not be interpreted as:

- `SN` replacing the `SC-PPO` constraint mechanism
- a reopened blind `SN-only` parameter sweep
- permission to spend `3-seed` or `MuJoCo` terminal budget before single-seed feasibility is positive

## Runner

The dedicated runner is:

- `scripts/baseline/run_sn_task_stabilized_diagnostic.py`

Plan command:

```bash
PATH=/TinyNAS2024/zhuoxiang/sco-humanoid/bin:$PATH \
/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/baseline/run_sn_task_stabilized_diagnostic.py \
  --stage plan \
  --preset smoke \
  --humanoid-gym-root /home/zhuoxiang/SCO-humanoid/.external/humanoid-gym
```

First bounded run:

```bash
PATH=/TinyNAS2024/zhuoxiang/sco-humanoid/bin:$PATH \
/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/baseline/run_sn_task_stabilized_diagnostic.py \
  --stage all \
  --preset smoke \
  --skip-completed \
  --humanoid-gym-root /home/zhuoxiang/SCO-humanoid/.external/humanoid-gym
```

If smoke is operational and not obviously degenerate, the next step is:

```bash
PATH=/TinyNAS2024/zhuoxiang/sco-humanoid/bin:$PATH \
/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/baseline/run_sn_task_stabilized_diagnostic.py \
  --stage all \
  --preset short \
  --skip-completed \
  --humanoid-gym-root /home/zhuoxiang/SCO-humanoid/.external/humanoid-gym
```

## Expected Artifacts

Training and evaluation artifacts write under:

- `artifacts/methods/sc_ppo_sn_task_stabilized_probe/`

Compact summaries write under:

- `artifacts/analysis/sn_task_stabilized_diagnostic/`

## Completed Result

The bounded single-seed diagnostic has completed through `smoke`, `short`, and `medium`.

Artifact roots:

- `artifacts/methods/sc_ppo_sn_task_stabilized_probe/`
- `artifacts/analysis/sn_task_stabilized_diagnostic/`

Compact summaries:

- `artifacts/analysis/sn_task_stabilized_diagnostic/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_sn_first_hidden_rough_terrain_smoke_seed123145_summary.json`
- `artifacts/analysis/sn_task_stabilized_diagnostic/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_sn_first_hidden_rough_terrain_short_seed123145_summary.json`
- `artifacts/analysis/sn_task_stabilized_diagnostic/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_sn_first_hidden_rough_terrain_medium_seed123145_summary.json`

Observed reading:

- `smoke`: operational, but `selection_status = all_checkpoints_collapsed`
- `short`: `checkpoint = 20`, `fall_rate = 1.0`, `velocity_tracking_error_mean = 1.4921`,
  `joint_acceleration_l2_mean = 112.7087`, `action_jitter_l2_mean = 0.0595`,
  `episode_return_mean = 2.4322`
- `medium`: `checkpoint = 100`, `fall_rate = 1.0`, `velocity_tracking_error_mean = 1.3795`,
  `joint_acceleration_l2_mean = 96.1738`, `action_jitter_l2_mean = 0.1271`,
  `episode_return_mean = 3.3864`, `policy_local_sensitivity_cost_mean = 2.0286`,
  `constraint_violation_rate = 0.0`

Mechanism check:

- the trained checkpoint contains `actor.0.weight_orig`, `actor.0.weight_u`, and `actor.0.weight_v`
- later actor layers keep normal `weight` tensors
- so the intended `first_hidden` SN wiring is active and the negative result is not a missing-config bug

Interpretation:

- the hybrid recipe is operational and does not immediately destabilize optimization
- training reward and episode length rise through `100` iterations while the constraint remains below
  the `3.8` threshold
- however, the selected evaluation checkpoint still collapses with `fall_rate = 1.0`
- relative to the documented formal `SC-PPO 3.8` mainline (`fall_rate = 0.1`, `episode_return_mean = 100.2838`),
  this branch is still far below task-valid territory
- relative to the completed `SN-only first-hidden medium` diagnostic, the hybrid slightly reduces
  `joint_acceleration_l2_mean` and slightly raises return, but it does not clear task validity and does
  not produce a defensible continuation signal

Current decision:

- close this branch as a negative first-stage feasibility result
- do not promote this recipe to `3-seed` or `MuJoCo` budget
- do not reopen SN follow-up unless a genuinely different recipe changes more than this auxiliary
  first-hidden-layer addition

## Success / Failure Reading

First-stage success means:

- the run trains cleanly enough to write a checkpoint and manifest
- the hybrid emits the same shared behavior metrics and constraint-side evidence
- the selected checkpoint is not immediately `all_checkpoints_collapsed`

First-stage failure means:

- the hybrid collapses under the same single-seed bounded diagnostic
- or it becomes operationally brittle enough that the added SN is not worth carrying forward
