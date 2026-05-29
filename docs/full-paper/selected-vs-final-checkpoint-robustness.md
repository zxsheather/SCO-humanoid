# Selected-vs-Final Checkpoint Robustness (#76)

Status: `complete`.

This note audits whether the full-paper primary rows rely materially on checkpoint-sweep selection. The comparison uses the same five Isaac seeds `11/17/23/29/31` for LCP, SC-PPO, and the revised heuristic. Delta is `final checkpoint - selected checkpoint`; for fall, velocity error, joint acceleration, jitter, and sensitivity, positive deltas are worse. For return, positive deltas are better.

## Main Read

- LCP is close to final-only behavior: only seed 11 selects checkpoint 300 rather than 400, and aggregate final-vs-selected deltas are small.
- SC-PPO is dynamic-selection-sensitive: final checkpoints improve velocity and return but worsen joint acceleration and jitter.
- The revised heuristic is task-selection-sensitive: final checkpoints improve velocity/return slightly but increase fall rate and dynamic roughness.
- The paper should report checkpoint-sweep selection explicitly, but selected-checkpoint dependence is not a hidden failure mode for LCP.

## Aggregate Selected-vs-Final Audit

| Method | Classification | Changed seeds | Selected ckpts | Final ckpts | Fall delta | Vel delta | Jnt acc delta | Jitter delta | Return delta | Sens delta |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LCP-style soft penalty | near_final | 1 | 300 / 400 / 400 / 400 / 400 | 400 / 400 / 400 / 400 / 400 | 0.000 | 0.003 | 1.004 | 0.007 | 1.722 | 0.004 |
| SC-PPO 3.8 PID | dynamic_selection_sensitive | 2 | 300 / 300 / 400 / 400 / 400 | 400 / 400 / 400 / 400 / 400 | 0.000 | -0.016 | 8.932 | 0.014 | 2.961 | -0.006 |
| Revised heuristic | task_selection_sensitive | 3 | 350 / 300 / 350 / 400 / 400 | 400 / 400 / 400 / 400 / 400 | 0.050 | -0.015 | 2.463 | 0.009 | 1.392 | 0.214 |

## Selected and Final Means

| Method | Metric | Selected mean | Final mean | Delta | Preference by delta |
| --- | --- | ---: | ---: | ---: | --- |
| LCP-style soft penalty | Fall | 0.000 | 0.000 | 0.000 | unchanged |
| LCP-style soft penalty | Vel. err | 0.490 | 0.493 | 0.003 | selected_better |
| LCP-style soft penalty | Jnt acc | 117.331 | 118.336 | 1.004 | selected_better |
| LCP-style soft penalty | Jitter | 0.212 | 0.220 | 0.007 | selected_better |
| LCP-style soft penalty | Return | 118.420 | 120.142 | 1.722 | final_better |
| LCP-style soft penalty | Sensitivity | 1.890 | 1.894 | 0.004 | selected_better |
| SC-PPO 3.8 PID | Fall | 0.170 | 0.170 | 0.000 | unchanged |
| SC-PPO 3.8 PID | Vel. err | 0.606 | 0.590 | -0.016 | final_better |
| SC-PPO 3.8 PID | Jnt acc | 142.955 | 151.887 | 8.932 | selected_better |
| SC-PPO 3.8 PID | Jitter | 0.277 | 0.291 | 0.014 | selected_better |
| SC-PPO 3.8 PID | Return | 99.349 | 102.310 | 2.961 | final_better |
| SC-PPO 3.8 PID | Sensitivity | 3.630 | 3.624 | -0.006 | final_better |
| Revised heuristic | Fall | 0.150 | 0.200 | 0.050 | selected_better |
| Revised heuristic | Vel. err | 0.705 | 0.691 | -0.015 | final_better |
| Revised heuristic | Jnt acc | 115.317 | 117.780 | 2.463 | selected_better |
| Revised heuristic | Jitter | 0.260 | 0.269 | 0.009 | selected_better |
| Revised heuristic | Return | 105.326 | 106.717 | 1.392 | final_better |
| Revised heuristic | Sensitivity | 7.331 | 7.545 | 0.214 | selected_better |

## Changed-Seed Detail

| Method | Seed | Selected ckpt | Final ckpt | Fall delta | Vel delta | Jnt acc delta | Jitter delta | Return delta | Sens delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LCP-style soft penalty | 11 | 300 | 400 | 0.000 | 0.015 | 5.021 | 0.037 | 8.609 | 0.020 |
| SC-PPO 3.8 PID | 11 | 300 | 400 | 0.000 | -0.059 | 24.613 | 0.041 | 7.489 | 0.064 |
| SC-PPO 3.8 PID | 17 | 300 | 400 | 0.000 | -0.020 | 20.047 | 0.029 | 7.315 | -0.092 |
| Revised heuristic | 11 | 350 | 400 | 0.250 | 0.027 | -8.521 | -0.005 | -1.644 | 0.719 |
| Revised heuristic | 17 | 300 | 400 | 0.050 | -0.034 | 20.286 | 0.039 | 0.213 | 0.280 |
| Revised heuristic | 23 | 350 | 400 | -0.050 | -0.066 | 0.552 | 0.010 | 8.389 | 0.068 |

## Paper Wording Guidance

- Say that LCP is nearly final-checkpoint stable under the current protocol.
- Say that SC-PPO's selected checkpoint mainly protects dynamic smoothness, not task survival.
- Say that the heuristic selected checkpoint mainly protects fall-rate/task validity.
- Keep selected-checkpoint selection as an explicit protocol limitation; do not hide it in aggregate tables.

## Source Artifacts

- `artifacts/analysis/rough_terrain_extended_seeds/comparison_summary.json`
- `artifacts/analysis/rough_terrain_lcp_soft_jacobian_formal/comparison_summary.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed11/checkpoint_sweep_summary.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed11/metrics_checkpoint_350.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed11/metrics_checkpoint_400.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed17/checkpoint_sweep_summary.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed17/metrics_checkpoint_300.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed17/metrics_checkpoint_400.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed23/checkpoint_sweep_summary.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed23/metrics_checkpoint_350.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed23/metrics_checkpoint_400.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed29/checkpoint_sweep_summary.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed29/metrics_checkpoint_400.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed31/checkpoint_sweep_summary.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed31/metrics_checkpoint_400.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed11/checkpoint_sweep_summary.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed11/metrics_checkpoint_300.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed11/metrics_checkpoint_400.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed17/checkpoint_sweep_summary.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed17/metrics_checkpoint_400.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed23/checkpoint_sweep_summary.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed23/metrics_checkpoint_400.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed29/checkpoint_sweep_summary.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed29/metrics_checkpoint_400.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed31/checkpoint_sweep_summary.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed31/metrics_checkpoint_400.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11/checkpoint_sweep_summary.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11/metrics_checkpoint_300.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11/metrics_checkpoint_400.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17/checkpoint_sweep_summary.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17/metrics_checkpoint_300.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17/metrics_checkpoint_400.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed23/checkpoint_sweep_summary.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed23/metrics_checkpoint_400.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed29/checkpoint_sweep_summary.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed29/metrics_checkpoint_400.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed31/checkpoint_sweep_summary.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed31/metrics_checkpoint_400.json`

## Reproduction

```bash
/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/analysis/analyze_checkpoint_robustness.py
```

Generated runtime summary: `artifacts/analysis/checkpoint_robustness/summary.json`
Generated runtime table note: `artifacts/analysis/checkpoint_robustness/summary.md`
