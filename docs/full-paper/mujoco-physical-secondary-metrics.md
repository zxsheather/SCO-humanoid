# MuJoCo Physical Secondary Metrics (#117)

Status: `complete`.

This note adds a trace-based secondary physical evaluation layer on the current five-seed selected-checkpoint MuJoCo replay. It uses the same selected checkpoints as the main mechanism comparison, rerun with compact per-timestep trace capture, and computes torque- and power-based proxies from the applied joint control and joint-velocity traces. These metrics are secondary physical evidence; they do not replace the primary task, joint-acceleration, or action-jitter metrics.

Metric definitions used throughout this note:

- `Torque RMS`: root-mean-square applied joint control over the captured trace.
- `Abs power`: mean absolute joint power proxy computed from applied joint control and joint velocity.
- `Abs energy`: time integral of absolute joint power over the captured episode.
- `Energy/m`: absolute episode energy divided by forward distance traveled.

## Main Read

- LCP is the lowest absolute-effort row on torque RMS (24.102), absolute power (85.659), and absolute episode energy (877.146). The heuristic remains best on the efficiency-style `energy/m` proxy (785.610) and on aggregate MuJoCo joint acceleration (111.819).
- This sharpens, rather than removes, the existing MuJoCo split: LCP appears physically cheaper in absolute actuation effort, while the heuristic remains more efficient per forward progress. SC-PPO is worst on absolute power (184.312) and remains worst on aggregate joint acceleration (160.900).
- Across the 15 method-seed rows, absolute power correlates with joint acceleration at 0.616 and with action jitter at 0.694; the efficiency proxy `energy/m` aligns most strongly with velocity error (0.603). The physical proxies therefore do not collapse to a single smoothness scalar.
- Because these traces use five captured episodes per selected checkpoint, this is still bounded secondary evidence rather than a new primary benchmark line or a bootstrap-backed uncertainty result.

## Five-Seed Aggregate

| Method | Checkpoints | Torque RMS | Abs power | Abs energy | Energy/m | MuJoCo Jnt acc | MuJoCo Jitter | MuJoCo Vel. err | MuJoCo Return |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LCP-style soft penalty | 300/400/400/400/400 | 24.102 | 85.659 | 877.146 | 1069.739 | 116.669 | 0.194 | 0.406 | -596.969 |
| SC-PPO 3.8 PID | 300/300/400/400/400 | 26.851 | 184.312 | 1887.355 | 1129.751 | 160.900 | 0.323 | 0.488 | -661.512 |
| Revised heuristic | 350/300/350/400/400 | 25.932 | 131.616 | 1347.749 | 785.610 | 111.819 | 0.228 | 0.404 | -437.008 |

## Per-Seed Rows

| Method | Seed | Ckpt | Trace fall | Torque RMS | Abs power | Abs energy | Energy/m | MuJoCo Jnt acc | MuJoCo Jitter | MuJoCo Return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Revised heuristic | 11 | 350 | 0.000 | 29.725 | 187.860 | 1923.682 | 762.996 | 118.454 | 0.290 | -203.668 |
| Revised heuristic | 17 | 300 | 0.000 | 22.794 | 107.228 | 1098.013 | 607.629 | 122.229 | 0.213 | -382.294 |
| Revised heuristic | 23 | 350 | 0.000 | 25.156 | 130.213 | 1333.382 | 1164.415 | 122.806 | 0.241 | -722.885 |
| Revised heuristic | 29 | 400 | 0.000 | 25.852 | 93.243 | 954.809 | 771.190 | 82.850 | 0.169 | -640.160 |
| Revised heuristic | 31 | 400 | 0.000 | 26.133 | 139.537 | 1428.859 | 621.822 | 112.756 | 0.229 | -236.033 |
| LCP-style soft penalty | 11 | 300 | 0.000 | 23.623 | 81.383 | 833.363 | 1500.732 | 110.425 | 0.164 | -751.870 |
| LCP-style soft penalty | 17 | 400 | 0.000 | 24.673 | 96.079 | 983.846 | 2039.348 | 117.470 | 0.184 | -922.671 |
| LCP-style soft penalty | 23 | 400 | 0.000 | 25.083 | 88.281 | 904.000 | 459.239 | 107.899 | 0.184 | -328.538 |
| LCP-style soft penalty | 29 | 400 | 0.000 | 24.576 | 91.805 | 940.080 | 544.202 | 109.774 | 0.190 | -336.340 |
| LCP-style soft penalty | 31 | 400 | 0.000 | 22.558 | 70.746 | 724.442 | 805.176 | 137.777 | 0.248 | -645.427 |
| SC-PPO 3.8 PID | 11 | 300 | 0.200 | 28.856 | 228.861 | 2343.532 | 1262.513 | 162.278 | 0.286 | -960.959 |
| SC-PPO 3.8 PID | 17 | 300 | 0.000 | 25.647 | 154.088 | 1577.858 | 737.731 | 103.977 | 0.221 | -322.981 |
| SC-PPO 3.8 PID | 23 | 400 | 0.000 | 24.743 | 139.633 | 1429.840 | 962.142 | 117.255 | 0.195 | -618.826 |
| SC-PPO 3.8 PID | 29 | 400 | 0.000 | 27.581 | 224.593 | 2299.828 | 1516.315 | 295.270 | 0.645 | -535.003 |
| SC-PPO 3.8 PID | 31 | 400 | 0.000 | 27.426 | 174.386 | 1785.716 | 1170.053 | 125.721 | 0.269 | -869.792 |

## Correlation with Existing MuJoCo Metrics

Correlations are descriptive Pearson correlations over the 15 method-seed rows.

| Physical metric | Corr with Jnt acc | Corr with Jitter | Corr with Vel. err |
| --- | ---: | ---: | ---: |
| Torque RMS | 0.319 | 0.462 | 0.371 |
| Abs power | 0.616 | 0.694 | 0.471 |
| Abs energy | 0.616 | 0.694 | 0.471 |
| Energy/m | 0.394 | 0.287 | 0.603 |

## Interpretation Boundary

- These are no-retraining selected-checkpoint MuJoCo traces, not hardware measurements.
- The metrics are physically grounded proxies built from simulated joint control and joint velocity; they strengthen interpretation but do not establish sim-to-real energy or actuator claims.
- The current read should remain secondary to the main mechanism result: policy-output smoothness and downstream dynamic effort are related, but they are not identical orderings.

## Source Artifacts

- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed11_mujoco_physical_trace_5ep/metrics_mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed11_mujoco_physical_trace_5ep/mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed17_mujoco_physical_trace_5ep/metrics_mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed17_mujoco_physical_trace_5ep/mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed23_mujoco_physical_trace_5ep/metrics_mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed23_mujoco_physical_trace_5ep/mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed29_mujoco_physical_trace_5ep/metrics_mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed29_mujoco_physical_trace_5ep/mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed31_mujoco_physical_trace_5ep/metrics_mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed31_mujoco_physical_trace_5ep/mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed11_mujoco_physical_trace_5ep/metrics_mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed11_mujoco_physical_trace_5ep/mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed17_mujoco_physical_trace_5ep/metrics_mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed17_mujoco_physical_trace_5ep/mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed23_mujoco_physical_trace_5ep/metrics_mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed23_mujoco_physical_trace_5ep/mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed29_mujoco_physical_trace_5ep/metrics_mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed29_mujoco_physical_trace_5ep/mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed31_mujoco_physical_trace_5ep/metrics_mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed31_mujoco_physical_trace_5ep/mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11_mujoco_physical_trace_5ep/metrics_mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11_mujoco_physical_trace_5ep/mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17_mujoco_physical_trace_5ep/metrics_mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17_mujoco_physical_trace_5ep/mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed23_mujoco_physical_trace_5ep/metrics_mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed23_mujoco_physical_trace_5ep/mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed29_mujoco_physical_trace_5ep/metrics_mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed29_mujoco_physical_trace_5ep/mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed31_mujoco_physical_trace_5ep/metrics_mujoco_physical_trace_5ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed31_mujoco_physical_trace_5ep/mujoco_physical_trace_5ep_20s_noise01.json`

## Reproduction

```bash
python scripts/baseline/run_mujoco_physical_trace_replays.py --skip-completed
python scripts/analysis/analyze_mujoco_physical_metrics.py
```

Generated runtime summary: `artifacts/analysis/mujoco_physical_secondary_metrics/summary.json`
Generated runtime note: `artifacts/analysis/mujoco_physical_secondary_metrics/summary.md`
