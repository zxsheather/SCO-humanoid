# Shared Smooth-Control Evaluation

Issue `#2` standardizes one evaluation entrypoint for all study methods:

```bash
python scripts/baseline/evaluate_policy.py --config <config-path>
```

For longer-budget runs where `最后一个 checkpoint` may not be the best stopping point, use:

```bash
python scripts/baseline/evaluate_checkpoint_sweep.py --config <config-path> --run-name <artifact-run-name> --load-run <upstream-run-dir>
```

For the current full-paper `MuJoCo` selected-checkpoint replay, use:

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
- do not treat the final checkpoint alone as sufficient evidence for long-budget
  LCP-style, SC-PPO, or heuristic rows

Current full-paper primary rows:

- LCP-style soft penalty:
  `configs/methods/lcp_soft_jacobian_penalty_diagnostic.json`
- `SC-PPO 3.8 PID`:
  `configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_extended_seeds.json`
- revised heuristic:
  `configs/methods/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget.json`
- current selected checkpoints over seeds `11/17/23/29/31` are:
  - LCP-style: `300/400/400/400/400`
  - `SC-PPO 3.8 PID`: `300/300/400/400/400`
  - revised heuristic: `350/300/350/400/400`

Historical SC-PPO mainline note:

- the earlier leading `SC-PPO` branch was
  `configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json`
- its best workshop-era evidence was a completed `3-seed, 400 iteration,
  checkpoint-sweep` batch
- selected checkpoints are `300`, `300`, and `400` for seeds `11`, `17`, and `23`
- selected-checkpoint aggregate metrics are:
  - `velocity_tracking_error_mean = 0.6412 ± 0.0554`
  - `joint_acceleration_l2_mean = 115.9079 ± 6.9386`
  - `action_jitter_l2_mean = 0.2205 ± 0.0017`
  - `episode_return_mean = 100.2838 ± 2.7150`
- `fall_rate = 0.1000 ± 0.0000`
- this historical slice beat the then-current heuristic anchor under the shared
  metric schema, but the full-paper five-seed branch supersedes this as the
  current project-level claim.

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

Base method configs:

- `configs/methods/vanilla_ppo.json`
- `configs/methods/heuristic_smoothing.json`
- `configs/methods/sc_ppo.json`

Legacy note:

- `configs/baselines/vanilla_ppo.json` remains as the original issue `#1` scaffold around the upstream default task setup.

## Current MuJoCo protocol status

The current repo should distinguish between four `MuJoCo` uses:

1. `isaac_mainline` aligned replay
2. `hfield_moderate` bounded second-setting replay
3. `hfield_stress` terrain stress probe
4. actuator-bandwidth stress with simulator-side action low-pass filtering

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

- LCP-style soft penalty:
  `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/*/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
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
- `hfield_moderate` was introduced as a repair-stage intermediate probe. The
  current full-paper uses the completed five-seed run only as bounded
  no-retraining second-setting evidence, not as a broad terrain benchmark.

Current result status:

- report-grade `MuJoCo isaac_mainline` aligned replay now uses matched seeds
  `11/17/23/29/31` for LCP-style soft penalty, SC-PPO 3.8 PID, and the revised
  heuristic.
- matched five-seed `isaac_mainline` aggregate:
  - LCP-style soft penalty: `fall_rate = 0.000`,
    `velocity_tracking_error_mean = 0.406`,
    `joint_acceleration_l2_mean = 117.425`,
    `action_jitter_l2_mean = 0.195`,
    `episode_return_mean = -599.108`
  - `SC-PPO 3.8 PID`: `fall_rate = 0.010`,
    `velocity_tracking_error_mean = 0.471`,
    `joint_acceleration_l2_mean = 159.718`,
    `action_jitter_l2_mean = 0.322`,
    `episode_return_mean = -627.238`
  - revised heuristic: `fall_rate = 0.000`,
    `velocity_tracking_error_mean = 0.406`,
    `joint_acceleration_l2_mean = 111.615`,
    `action_jitter_l2_mean = 0.226`,
    `episode_return_mean = -456.370`
- the current aligned replay is mixed but interpretable: LCP is cleanest on
  action jitter and stronger than SC-PPO on dynamic smoothness, while the
  revised heuristic remains best on aggregate joint acceleration and return.
- `hfield_moderate` is no longer merely a seed11 repair note; it has a completed
  five-seed bounded second-setting replay over the same three primary methods:
  - LCP-style soft penalty: `fall_rate = 0.350`,
    `velocity_tracking_error_mean = 0.832`,
    `joint_acceleration_l2_mean = 321.113`,
    `action_jitter_l2_mean = 0.277`,
    `episode_return_mean = -566.367`
  - `SC-PPO 3.8 PID`: `fall_rate = 0.500`,
    `velocity_tracking_error_mean = 0.981`,
    `joint_acceleration_l2_mean = 386.647`,
    `action_jitter_l2_mean = 0.380`,
    `episode_return_mean = -513.230`
  - revised heuristic: `fall_rate = 0.400`,
    `velocity_tracking_error_mean = 1.003`,
    `joint_acceleration_l2_mean = 322.360`,
    `action_jitter_l2_mean = 0.358`,
    `episode_return_mean = -666.633`
- `hfield_moderate` should still be described as bounded no-retraining
  second-setting evidence, not a solved terrain protocol or broad
  multi-terrain benchmark, because all fall rates remain materially higher
  than in the primary matched replay.

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
