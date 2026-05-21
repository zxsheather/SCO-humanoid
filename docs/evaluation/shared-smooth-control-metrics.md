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
python scripts/baseline/evaluate_mujoco_sim2sim.py --config <config-path> --run-name <artifact-run-name> --load-run <upstream-run-dir> --checkpoint <N> --terrain-mode <isaac_mainline|plane|hfield_moderate|hfield_stress>
```

This writes:

- `metrics_checkpoint_<N>.json` for each evaluated checkpoint
- `checkpoint_sweep_summary.json` with a sortable table and selected best checkpoint
- `metrics_selected.json` for the selected checkpoint metrics snapshot

The `MuJoCo` evaluator writes:

- `metrics_mujoco.json` by default
- or a caller-selected artifact such as `metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
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

The current repo should distinguish between three `MuJoCo` uses:

1. `isaac_mainline` aligned replay
2. `terrain stress probe`
3. `terrain repair-stage intermediate probe`

Current preferred aligned replay protocol:

- terrain mode: `isaac_mainline`
- current resolved XML: `plane`
- reset noise: `joint_reset_noise = 0.1`
- duration: `20 episodes`, `20 seconds`
- purpose: check whether selected policies retain task validity across simulators and whether the
  Isaac main-comparison ordering transfers
- note: this now resolves from the Isaac training config itself, and the evaluator will fail closed
  rather than silently swapping in `hfield` if the training-side terrain semantics change later

Current formal comparable artifacts:

- revised long-budget heuristic anchor:
  `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/*/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `SC-PPO threshold = 3.8` anchor:
  `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed*/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- canonical comparison note:
  `docs/sc-ppo-mujoco-revised-anchor-aligned-comparison.md`
- these aligned three-seed artifacts replace the older single-run heuristic and single representative
  checkpoint references for report wording

Current terrain probe protocol:

- terrain mode: `hfield_stress`
- XML: `terrain`
- reset noise: `joint_reset_noise = 0.1`
- typical short probe: `5 episodes`, `5 seconds`
- purpose: diagnose transfer fragility rather than serve as the current report-grade external
  result

Current terrain repair-stage intermediate protocol:

- terrain mode: `hfield_moderate`
- base XML: `terrain`
- current repair override: `hfield_size_override = [50.0, 50.0, 0.06, 0.02]`
- reset noise: `joint_reset_noise = 0.1`
- current short probe: `5 episodes`, `5 seconds`
- current mid-budget check: `20 episodes`, `20 seconds`
- purpose: test whether a softened hfield can become a discriminative middle stage between
  `isaac_mainline` and `hfield_stress`

Protocol repair note:

- the old bare `--terrain` flag is now treated as a deprecated alias for `--terrain-mode=hfield_stress`
- this prevents `MuJoCo terrain` from being confused with the current Isaac-mainline replay
- `hfield_moderate` is explicitly not a report-grade comparable replay; it is a repair-stage
  intermediate probe

Current result status:

- report-grade `MuJoCo isaac_mainline` aligned replay now uses the revised long-budget heuristic
  anchor and `SC-PPO 3.8` over seeds `11`, `17`, and `23`
- revised heuristic aligned replay:
  - `velocity_tracking_error_mean = 0.4188 ± 0.0398`
  - `joint_acceleration_l2_mean = 120.7339 ± 2.6413`
  - `action_jitter_l2_mean = 0.2452 ± 0.0288`
  - `fall_rate = 0.0000 ± 0.0000`
  - `episode_steps_mean = 2000.0 ± 0.0`
- `SC-PPO 3.8` aligned replay:
  - `velocity_tracking_error_mean = 0.4910 ± 0.0944`
  - `joint_acceleration_l2_mean = 125.5411 ± 21.1683`
  - `action_jitter_l2_mean = 0.2313 ± 0.0351`
  - `fall_rate = 0.0167 ± 0.0236`
  - `episode_steps_mean = 1984.7833 ± 21.5196`
- the aligned replay is mixed evidence: the revised heuristic is better on task stability, velocity
  tracking, episode length, and joint acceleration, while `SC-PPO 3.8` is only slightly better on
  action jitter
- separate terrain repair-stage checks remain non-report-grade; both methods currently fail the
  short `MuJoCo terrain` probe, and `SC-PPO` does not recover the terrain result through the current
  `200/300/400` checkpoint neighborhood
- on the current `hfield_moderate` `20 episodes x 20 seconds` mid-budget check, `SC-PPO` remains
  materially more survivable than the terrain-repair heuristic comparator:
  - heuristic: `fall_rate = 1.0`, `episode_steps_mean = 236.35`
  - `SC-PPO`: `fall_rate = 0.4`, `episode_steps_mean = 1259.0`
- `velocity_tracking_error_mean` also remains slightly better for `SC-PPO`
  (`1.0210` vs `1.0975`)
- under the current `SC-PPO seed11` checkpoint sweep on this same protocol, `checkpoint 400`
  currently improves over `checkpoint 300`:
  - `ckpt200 -> fall_rate = 0.70`, `episode_steps_mean = 902.7`
  - `ckpt300 -> fall_rate = 0.40`, `episode_steps_mean = 1259.0`
  - `ckpt400 -> fall_rate = 0.35`, `episode_steps_mean = 1331.7`
- the current repaired-terrain line now also has a completed `3-seed` MuJoCo batch:
  - `seed11 -> checkpoint 400`
  - `seed17 -> checkpoint 400`
  - `seed23 -> checkpoint 300`
- over those repaired-terrain-selected checkpoints, `SC-PPO` aggregates to:
  - `velocity_tracking_error_mean = 0.9622 ± 0.0543`
  - `joint_acceleration_l2_mean = 352.6293 ± 6.5909`
  - `action_jitter_l2_mean = 0.3336 ± 0.0236`
  - `fall_rate = 0.3500 ± 0.0408`
  - `episode_steps_mean = 1346.03 ± 89.37`
- so the repaired-terrain survival advantage is no longer a `seed11` fluke, but this line still
  has checkpoint dependence and still does not beat the terrain-repair heuristic comparator on
  smoothness
- however, `joint_acceleration_l2_mean` and `action_jitter_l2_mean` still favor the
  terrain-repair heuristic comparator, so this remains a repair-stage signal rather than a solved
  terrain protocol

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
