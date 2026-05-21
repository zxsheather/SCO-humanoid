# SC-PPO SN Prototype

This note records the repo's first implementation step for issue `#11`.

The current implementation does not promote `SN` into a new formal mainline.
It only creates the minimal path required for a `替代机制可行性诊断`.

## Current implementation scope

The repo now has a dedicated diagnostic config:

- `configs/methods/sn_ppo_rough_terrain.json`

The repo also has a dedicated reduced-budget diagnostic launcher:

- `scripts/baseline/run_sn_diagnostic.py`

This branch currently means:

- same `速度跟踪行走` task
- same `粗糙平面` training condition
- same smoothness-oriented reward disablement as the current `SC-PPO` path
- actor-side `Spectral Normalization`
- standard `PPO` training path rather than the current Jacobian-penalty
  `SCPPO` training path

## Why this shape

This is the narrowest implementation that matches the current agreed question:

`Can actor-side Spectral Normalization replace the current Jacobian-penalty path without expanding the task definition?`

The branch therefore changes only the actor-side mechanism and keeps the rest
of the evaluation flow intact.

## Preserved comparison surface

The repo still keeps:

- `scripts/baseline/train_vanilla_ppo.py`
- `scripts/baseline/evaluate_policy.py`
- `scripts/baseline/evaluate_checkpoint_sweep.py`

The evaluation side can still emit:

- `policy_local_sensitivity_cost_*`
- `constraint_violation_rate`
- the shared behavior metrics

So the branch can still participate in `同尺比较`.

## Smoke status

The branch has cleared a minimal local smoke path before this runner was added:

- `1 iteration`
- reduced `num_envs = 16`
- same `粗糙平面` task semantics

Smoke artifacts:

- `artifacts/methods/sn_ppo_probe/sn_ppo_rough_terrain_smoke_small/manifest.json`
- `artifacts/methods/sn_ppo_probe/sn_ppo_rough_terrain_smoke_small/metrics.json`

Matched `SC-PPO 3.8` smoke artifacts for same-budget comparison:

- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_smoke_small/manifest.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_smoke_small/metrics.json`

## Current smoke reading

This smoke confirms:

- the `SN` branch can train through the current wrapper path
- the branch can emit reportable `manifest.json` and `metrics.json`
- mechanism-side evidence remains readable through the existing
  `constraint_metrics` section in `metrics.json`

Current same-budget `1 episode` smoke comparison should not be read as a
performance conclusion.

It only shows that the two branches can now be inspected under the same small
evaluation surface:

- `SN` smoke:
  - `velocity_tracking_error_mean = 1.9225`
  - `joint_acceleration_l2_mean = 40.6727`
  - `action_jitter_l2_mean = 0.0225`
  - `policy_local_sensitivity_cost_mean = 0.4932`
- `SC-PPO 3.8` smoke:
  - `velocity_tracking_error_mean = 1.7877`
  - `joint_acceleration_l2_mean = 42.0266`
  - `action_jitter_l2_mean = 0.0129`
  - `policy_local_sensitivity_cost_mean = 0.2974`

The repeatable runner has also completed its first `smoke` pass:

- command:
  `python scripts/baseline/run_sn_diagnostic.py --stage all --preset smoke --skip-completed --rl-device cuda:0 --sim-device cuda:0`
- run name: `sn_ppo_rough_terrain_smoke_seed123145`
- selected checkpoint: `1`
- `selection_status = all_checkpoints_collapsed`
- `velocity_tracking_error_mean = 1.1467`
- `joint_acceleration_l2_mean = 61.0134`
- `action_jitter_l2_mean = 0.0249`
- `episode_return_mean = 4.1629`
- `fall_rate = 1.0000`
- `policy_local_sensitivity_cost_mean = 0.5163`

This confirms the new runner can train, evaluate, recover post-metrics segfaults, and write a
compact diagnostic summary. It also confirms that the current `smoke` result is not task-valid.

Tracked compact summary:

- `artifacts/analysis/sn_replacement_diagnostic/sn_ppo_rough_terrain_smoke_seed123145_summary.json`

## Current operational boundary

An attempted larger `SN` smoke at `num_envs = 64` failed with CUDA OOM in the
current local machine state.

So the current correct reading is:

- the branch is operational as a reduced-budget diagnostic path
- the branch is not yet validated as a drop-in high-parallel replacement for the
  current training budget

The repeatable command is now:

```bash
python scripts/baseline/run_sn_diagnostic.py --stage all --preset smoke --skip-completed
```

The `smoke` preset uses `16` training envs, `1` iteration, `16` evaluation envs, and `1` episode.
The `short` preset uses `32` training envs, `20` iterations, `16` evaluation envs, and `5`
episodes. Both presets intentionally stay below formal-comparison budget.

## Current non-goals

This prototype does not yet claim:

- that `SN` beats the current `SC-PPO 3.8` mainline
- that `SN` deserves `主实验三种子`
- that `SN` deserves `MuJoCo关键两组终验`
- that `SN` should replace the current report-grade branch
