# Shared Smooth-Control Evaluation

Issue `#2` standardizes one evaluation entrypoint for all study methods:

```bash
python scripts/baseline/evaluate_policy.py --config <config-path>
```

For longer-budget runs where `最后一个 checkpoint` may not be the best stopping point, use:

```bash
python scripts/baseline/evaluate_checkpoint_sweep.py --config <config-path> --run-name <artifact-run-name> --load-run <upstream-run-dir>
```

For the current `MuJoCo关键两组终验`, use:

```bash
python scripts/baseline/evaluate_mujoco_sim2sim.py --config <config-path> --run-name <artifact-run-name> --load-run <upstream-run-dir> --checkpoint <N> --terrain-mode <isaac_mainline|plane|hfield_stress>
```

This writes:

- `metrics_checkpoint_<N>.json` for each evaluated checkpoint
- `checkpoint_sweep_summary.json` with a sortable table and selected best checkpoint
- `metrics_selected.json` for the selected checkpoint metrics snapshot

The `MuJoCo` evaluator writes:

- `metrics_mujoco.json` by default
- or a caller-selected artifact such as `metrics_mujoco_plane_20ep_20s_noise01.json`
- `manifest.json` keeps the latest result under `mujoco_metrics`
- `manifest.json` also keeps named result variants under `mujoco_metrics_runs`

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

Nearest-neighbor control note:

- the completed repaired-`4.0` three-seed control is materially less stable than the `3.8`
  mainline
- its selected-checkpoint aggregate metrics are:
  - `velocity_tracking_error_mean = 0.8635 ± 0.3367`
  - `joint_acceleration_l2_mean = 120.1226 ± 26.5838`
  - `action_jitter_l2_mean = 0.1740 ± 0.1157`
  - `episode_return_mean = 65.5950 ± 43.4320`
  - `fall_rate = 0.4667 ± 0.3793`
- one `4.0` seed selects `checkpoint 0`, so this branch should be treated as a completed control
  rather than a second mainline candidate

Current method configs:

- `configs/methods/vanilla_ppo.json`
- `configs/methods/heuristic_smoothing.json`
- `configs/methods/sc_ppo.json`

Legacy note:

- `configs/baselines/vanilla_ppo.json` remains as the original issue `#1` scaffold around the upstream default task setup.

## Current MuJoCo protocol status

The current repo should distinguish between two `MuJoCo` uses:

1. `最小可比 first pass`
2. `terrain stress probe`

Current preferred first-pass protocol:

- terrain mode: `isaac_mainline`
- current resolved XML: `plane`
- reset noise: `joint_reset_noise = 0.1`
- duration: `20 episodes`, `20 seconds`
- purpose: check whether the selected policy retains a meaningful task-validity advantage across
  simulators
- note: this now resolves from the Isaac training config itself, and the evaluator will fail closed
  rather than silently swapping in `hfield` if the training-side terrain semantics change later

Current terrain probe protocol:

- terrain mode: `hfield_stress`
- XML: `terrain`
- reset noise: `joint_reset_noise = 0.1`
- typical short probe: `5 episodes`, `5 seconds`
- purpose: diagnose transfer fragility rather than serve as the current report-grade external
  result

Protocol repair note:

- the old bare `--terrain` flag is now treated as a deprecated alias for `--terrain-mode=hfield_stress`
- this prevents `MuJoCo terrain` from being confused with the current Isaac-mainline replay

Current result status:

- `SC-PPO` now shows a stronger `MuJoCo plane` task-stability result than the heuristic anchor
- however, the current `MuJoCo plane` smoothness metrics still favor the heuristic anchor
- both methods currently fail the short `MuJoCo terrain` probe, and `SC-PPO` does not recover the
  terrain result through the current `200/300/400` checkpoint neighborhood

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
