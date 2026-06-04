# Full-Paper Statistical Robustness Results (#116)

Status: `complete`.

This note adds a descriptive statistical audit for the full-paper mechanism-comparison evidence. It uses matched seeds `11/17/23/29/31`, nonparametric bootstrap confidence intervals over seed means, IQM summaries, paired seed-level deltas, bootstrap rank stability, and a compact checkpoint-instability read. With five seeds, these intervals should be read as uncertainty evidence rather than strong null-hypothesis significance tests.

## Main Read

- The added IQM view preserves the mean-based read: LCP is clearly stronger than the current SC-PPO hard-constraint row on Isaac fall, velocity error, return, and sensitivity, and on MuJoCo action jitter.
- LCP's joint-acceleration advantage over SC-PPO is still directionally favorable in both Isaac and MuJoCo, but the paired bootstrap intervals overlap zero because seed-level variance remains large even under the robust aggregate.
- LCP versus the revised heuristic remains metric-dependent under both mean and IQM views: LCP is usually better on action jitter and return-sensitive Isaac task behavior, while the heuristic remains competitive or better on joint acceleration, especially in MuJoCo.
- All three primary rows are 5/5 noncollapsed under the repo's selected-checkpoint task-validity guard (`fall_rate < 1.0`); the cleaner reliability split is checkpoint dependence, which rises from LCP (1 changed seed) to SC-PPO (2) to the heuristic (3).
- Several paired confidence intervals still include zero. The paper should therefore report stable directions and uncertainty, not binary significance claims.

Representative paired reads:

- Isaac LCP vs SC-PPO, joint acceleration: LCP-style soft penalty preferred by mean, but CI includes zero ([-86.947, 18.641]).
- Isaac LCP vs heuristic, joint acceleration: Revised heuristic preferred by mean, but CI includes zero ([-13.198, 22.954]).
- MuJoCo LCP vs SC-PPO, joint acceleration: LCP-style soft penalty preferred by mean, but CI includes zero ([-118.337, 9.849]).
- MuJoCo LCP vs heuristic, joint acceleration: Revised heuristic preferred by mean, but CI includes zero ([-9.275, 20.895]).
- MuJoCo LCP vs heuristic, action jitter: LCP-style soft penalty preferred by mean, but CI includes zero ([-0.078, 0.013]).

## Mean and Bootstrap CI

| Dataset | Method | Metric | Mean | Std | 95% bootstrap CI |
| --- | --- | --- | ---: | ---: | ---: |
| isaac | LCP-style soft penalty | Fall | 0.000 | 0.000 | [0.000, 0.000] |
| isaac | LCP-style soft penalty | Vel. err | 0.490 | 0.032 | [0.462, 0.518] |
| isaac | LCP-style soft penalty | Jnt acc | 117.331 | 21.171 | [105.323, 138.663] |
| isaac | LCP-style soft penalty | Jitter | 0.212 | 0.033 | [0.188, 0.246] |
| isaac | LCP-style soft penalty | Return | 118.420 | 6.595 | [112.668, 124.098] |
| isaac | LCP-style soft penalty | Sensitivity | 1.890 | 0.064 | [1.844, 1.952] |
| isaac | SC-PPO 3.8 PID | Fall | 0.170 | 0.194 | [0.040, 0.370] |
| isaac | SC-PPO 3.8 PID | Vel. err | 0.606 | 0.087 | [0.521, 0.675] |
| isaac | SC-PPO 3.8 PID | Jnt acc | 142.955 | 52.454 | [111.983, 196.474] |
| isaac | SC-PPO 3.8 PID | Jitter | 0.277 | 0.100 | [0.220, 0.379] |
| isaac | SC-PPO 3.8 PID | Return | 99.349 | 13.959 | [86.173, 111.760] |
| isaac | SC-PPO 3.8 PID | Sensitivity | 3.630 | 0.067 | [3.564, 3.678] |
| isaac | Revised heuristic | Fall | 0.150 | 0.063 | [0.090, 0.210] |
| isaac | Revised heuristic | Vel. err | 0.705 | 0.103 | [0.631, 0.809] |
| isaac | Revised heuristic | Jnt acc | 115.317 | 8.770 | [106.422, 120.892] |
| isaac | Revised heuristic | Jitter | 0.260 | 0.018 | [0.242, 0.273] |
| isaac | Revised heuristic | Return | 105.326 | 10.335 | [95.141, 112.150] |
| isaac | Revised heuristic | Sensitivity | 7.331 | 0.237 | [7.136, 7.558] |
| mujoco | LCP-style soft penalty | Fall | 0.000 | 0.000 | [0.000, 0.000] |
| mujoco | LCP-style soft penalty | Vel. err | 0.406 | 0.060 | [0.356, 0.458] |
| mujoco | LCP-style soft penalty | Jnt acc | 117.425 | 11.234 | [109.855, 128.684] |
| mujoco | LCP-style soft penalty | Jitter | 0.195 | 0.029 | [0.174, 0.224] |
| mujoco | LCP-style soft penalty | Return | -599.108 | 236.114 | [-803.218, -394.999] |
| mujoco | SC-PPO 3.8 PID | Fall | 0.010 | 0.020 | [0.000, 0.030] |
| mujoco | SC-PPO 3.8 PID | Vel. err | 0.471 | 0.079 | [0.415, 0.546] |
| mujoco | SC-PPO 3.8 PID | Jnt acc | 159.718 | 71.345 | [113.540, 230.920] |
| mujoco | SC-PPO 3.8 PID | Jitter | 0.322 | 0.167 | [0.217, 0.487] |
| mujoco | SC-PPO 3.8 PID | Return | -627.238 | 222.350 | [-830.243, -442.453] |
| mujoco | Revised heuristic | Fall | 0.000 | 0.000 | [0.000, 0.000] |
| mujoco | Revised heuristic | Vel. err | 0.406 | 0.043 | [0.372, 0.447] |
| mujoco | Revised heuristic | Jnt acc | 111.615 | 14.597 | [97.296, 121.418] |
| mujoco | Revised heuristic | Jitter | 0.226 | 0.037 | [0.192, 0.258] |
| mujoco | Revised heuristic | Return | -456.370 | 195.030 | [-627.126, -285.614] |

## IQM and Bootstrap CI

IQM is the interquartile mean across the five matched seeds. It downweights the single highest and single lowest quarter-sample mass and is therefore a small-sample robust aggregate rather than a new test.

| Dataset | Method | Metric | IQM | 95% bootstrap CI |
| --- | --- | --- | ---: | ---: |
| isaac | LCP-style soft penalty | Fall | 0.000 | [0.000, 0.000] |
| isaac | LCP-style soft penalty | Vel. err | 0.488 | [0.454, 0.530] |
| isaac | LCP-style soft penalty | Jnt acc | 107.526 | [104.732, 144.834] |
| isaac | LCP-style soft penalty | Jitter | 0.203 | [0.184, 0.253] |
| isaac | LCP-style soft penalty | Return | 119.056 | [110.078, 125.709] |
| isaac | LCP-style soft penalty | Sensitivity | 1.868 | [1.838, 1.977] |
| isaac | SC-PPO 3.8 PID | Fall | 0.100 | [0.030, 0.415] |
| isaac | SC-PPO 3.8 PID | Vel. err | 0.633 | [0.488, 0.682] |
| isaac | SC-PPO 3.8 PID | Jnt acc | 120.502 | [110.202, 209.341] |
| isaac | SC-PPO 3.8 PID | Jitter | 0.230 | [0.219, 0.408] |
| isaac | SC-PPO 3.8 PID | Return | 100.368 | [82.328, 114.717] |
| isaac | SC-PPO 3.8 PID | Sensitivity | 3.647 | [3.541, 3.691] |
| isaac | Revised heuristic | Fall | 0.150 | [0.080, 0.220] |
| isaac | Revised heuristic | Vel. err | 0.665 | [0.629, 0.847] |
| isaac | Revised heuristic | Jnt acc | 118.604 | [103.908, 121.751] |
| isaac | Revised heuristic | Jitter | 0.263 | [0.237, 0.278] |
| isaac | Revised heuristic | Return | 109.201 | [91.518, 113.208] |
| isaac | Revised heuristic | Sensitivity | 7.305 | [7.073, 7.631] |
| mujoco | LCP-style soft penalty | Fall | 0.000 | [0.000, 0.000] |
| mujoco | LCP-style soft penalty | Vel. err | 0.400 | [0.339, 0.480] |
| mujoco | LCP-style soft penalty | Jnt acc | 112.994 | [109.157, 132.688] |
| mujoco | LCP-style soft penalty | Jitter | 0.186 | [0.170, 0.232] |
| mujoco | LCP-style soft penalty | Return | -585.989 | [-876.773, -330.653] |
| mujoco | SC-PPO 3.8 PID | Fall | 0.000 | [0.000, 0.035] |
| mujoco | SC-PPO 3.8 PID | Vel. err | 0.446 | [0.403, 0.575] |
| mujoco | SC-PPO 3.8 PID | Jnt acc | 131.023 | [108.407, 255.345] |
| mujoco | SC-PPO 3.8 PID | Jitter | 0.256 | [0.202, 0.539] |
| mujoco | SC-PPO 3.8 PID | Return | -606.710 | [-901.762, -380.972] |
| mujoco | Revised heuristic | Fall | 0.000 | [0.000, 0.000] |
| mujoco | Revised heuristic | Vel. err | 0.402 | [0.360, 0.461] |
| mujoco | Revised heuristic | Jnt acc | 117.000 | [92.134, 122.965] |
| mujoco | Revised heuristic | Jitter | 0.226 | [0.182, 0.270] |
| mujoco | Revised heuristic | Return | -430.516 | [-701.857, -253.906] |

## Paired Seed-Level Deltas

Delta is `first method - second method`. For fall, velocity error, joint acceleration, jitter, and sensitivity, lower is better. For return, higher is better.

| Dataset | Comparison | Metric | Mean delta | 95% bootstrap CI | Mean-preferred method | CI excludes zero |
| --- | --- | --- | ---: | ---: | --- | --- |
| isaac | LCP-style soft penalty - SC-PPO 3.8 PID | Fall | -0.170 | [-0.370, -0.040] | LCP-style soft penalty | true |
| isaac | LCP-style soft penalty - SC-PPO 3.8 PID | Vel. err | -0.116 | [-0.209, -0.009] | LCP-style soft penalty | true |
| isaac | LCP-style soft penalty - SC-PPO 3.8 PID | Jnt acc | -25.624 | [-86.947, 18.641] | LCP-style soft penalty | false |
| isaac | LCP-style soft penalty - SC-PPO 3.8 PID | Jitter | -0.065 | [-0.170, 0.003] | LCP-style soft penalty | false |
| isaac | LCP-style soft penalty - SC-PPO 3.8 PID | Return | 19.071 | [1.263, 36.850] | LCP-style soft penalty | true |
| isaac | LCP-style soft penalty - SC-PPO 3.8 PID | Sensitivity | -1.740 | [-1.819, -1.658] | LCP-style soft penalty | true |
| isaac | LCP-style soft penalty - Revised heuristic | Fall | -0.150 | [-0.210, -0.090] | LCP-style soft penalty | true |
| isaac | LCP-style soft penalty - Revised heuristic | Vel. err | -0.215 | [-0.302, -0.139] | LCP-style soft penalty | true |
| isaac | LCP-style soft penalty - Revised heuristic | Jnt acc | 2.015 | [-13.198, 22.954] | Revised heuristic | false |
| isaac | LCP-style soft penalty - Revised heuristic | Jitter | -0.048 | [-0.081, -0.012] | LCP-style soft penalty | true |
| isaac | LCP-style soft penalty - Revised heuristic | Return | 13.094 | [2.307, 25.388] | LCP-style soft penalty | true |
| isaac | LCP-style soft penalty - Revised heuristic | Sensitivity | -5.441 | [-5.691, -5.225] | LCP-style soft penalty | true |
| isaac | SC-PPO 3.8 PID - Revised heuristic | Fall | 0.020 | [-0.130, 0.220] | Revised heuristic | false |
| isaac | SC-PPO 3.8 PID - Revised heuristic | Vel. err | -0.099 | [-0.229, 0.018] | SC-PPO 3.8 PID | false |
| isaac | SC-PPO 3.8 PID - Revised heuristic | Jnt acc | 27.639 | [-9.028, 89.948] | Revised heuristic | false |
| isaac | SC-PPO 3.8 PID - Revised heuristic | Jitter | 0.017 | [-0.052, 0.135] | Revised heuristic | false |
| isaac | SC-PPO 3.8 PID - Revised heuristic | Return | -5.976 | [-20.941, 8.157] | Revised heuristic | false |
| isaac | SC-PPO 3.8 PID - Revised heuristic | Sensitivity | -3.702 | [-3.896, -3.509] | SC-PPO 3.8 PID | true |
| mujoco | LCP-style soft penalty - SC-PPO 3.8 PID | Fall | -0.010 | [-0.030, 0.000] | LCP-style soft penalty | false |
| mujoco | LCP-style soft penalty - SC-PPO 3.8 PID | Vel. err | -0.064 | [-0.142, 0.030] | LCP-style soft penalty | false |
| mujoco | LCP-style soft penalty - SC-PPO 3.8 PID | Jnt acc | -42.293 | [-118.337, 9.849] | LCP-style soft penalty | false |
| mujoco | LCP-style soft penalty - SC-PPO 3.8 PID | Jitter | -0.127 | [-0.301, -0.017] | LCP-style soft penalty | true |
| mujoco | LCP-style soft penalty - SC-PPO 3.8 PID | Return | 28.129 | [-303.324, 254.042] | LCP-style soft penalty | false |
| mujoco | LCP-style soft penalty - Revised heuristic | Fall | 0.000 | [0.000, 0.000] | tie | false |
| mujoco | LCP-style soft penalty - Revised heuristic | Vel. err | -0.000 | [-0.079, 0.079] | LCP-style soft penalty | false |
| mujoco | LCP-style soft penalty - Revised heuristic | Jnt acc | 5.810 | [-9.275, 20.895] | Revised heuristic | false |
| mujoco | LCP-style soft penalty - Revised heuristic | Jitter | -0.031 | [-0.078, 0.013] | LCP-style soft penalty | false |
| mujoco | LCP-style soft penalty - Revised heuristic | Return | -142.738 | [-486.818, 202.170] | Revised heuristic | false |
| mujoco | SC-PPO 3.8 PID - Revised heuristic | Fall | 0.010 | [0.000, 0.030] | Revised heuristic | false |
| mujoco | SC-PPO 3.8 PID - Revised heuristic | Vel. err | 0.064 | [-0.013, 0.158] | Revised heuristic | false |
| mujoco | SC-PPO 3.8 PID - Revised heuristic | Jnt acc | 48.103 | [-6.902, 133.170] | Revised heuristic | false |
| mujoco | SC-PPO 3.8 PID - Revised heuristic | Jitter | 0.096 | [-0.021, 0.292] | Revised heuristic | false |
| mujoco | SC-PPO 3.8 PID - Revised heuristic | Return | -170.867 | [-506.524, 104.696] | Revised heuristic | false |

## Reliability and Checkpoint Instability

Selected task-valid counts use the repo's checkpoint-sweep noncollapse guard (`fall_rate < 1.0`). `Selected=final` counts show how often the selected checkpoint already matches the final checkpoint on the same seed.

| Method | Selected task-valid seeds | Zero-fall seeds | Changed seeds | Selected=final | Checkpoint class | Isaac fall delta | Isaac Jnt acc delta | MuJoCo Jnt acc delta | MuJoCo Jitter delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| LCP-style soft penalty | 5/5 | 5/5 | 1/5 | 4/5 | near_final | 0.000 | 1.004 | -0.170 | 0.008 |
| SC-PPO 3.8 PID | 5/5 | 1/5 | 2/5 | 3/5 | dynamic_selection_sensitive | 0.000 | 8.932 | 8.689 | 0.016 |
| Revised heuristic | 5/5 | 0/5 | 3/5 | 2/5 | task_selection_sensitive | 0.050 | 2.463 | 0.249 | -0.001 |

## Bootstrap Rank Stability

Values are the fraction of bootstrap resamples in which each method is the best-ranked method for the metric.

| Dataset | Metric | Most frequent winner | LCP | SC-PPO | Heuristic |
| --- | --- | --- | ---: | ---: | ---: |
| isaac | Fall | LCP-style soft penalty | 1.000 | 0.000 | 0.000 |
| isaac | Vel. err | LCP-style soft penalty | 0.987 | 0.013 | 0.000 |
| isaac | Jnt acc | Revised heuristic | 0.389 | 0.131 | 0.480 |
| isaac | Jitter | LCP-style soft penalty | 0.957 | 0.041 | 0.002 |
| isaac | Return | LCP-style soft penalty | 0.990 | 0.005 | 0.005 |
| isaac | Sensitivity | LCP-style soft penalty | 1.000 | 0.000 | 0.000 |
| mujoco | Fall | LCP-style soft penalty / Revised heuristic | 0.445 | 0.109 | 0.445 |
| mujoco | Vel. err | LCP-style soft penalty | 0.515 | 0.010 | 0.475 |
| mujoco | Jnt acc | Revised heuristic | 0.251 | 0.064 | 0.685 |
| mujoco | Jitter | LCP-style soft penalty | 0.911 | 0.000 | 0.089 |
| mujoco | Return | Revised heuristic | 0.259 | 0.040 | 0.701 |

## Paper Wording Guidance

- Use `paired bootstrap + IQM uncertainty audit` rather than `statistical significance test`.
- It is defensible to say LCP is robustly stronger than SC-PPO in the current five-seed mechanism comparison, and that the IQM view preserves that read.
- It is not defensible to say LCP robustly dominates the revised heuristic across all metrics; the robust aggregate still leaves a joint-acceleration trade-off.
- Keep the revised heuristic as a strong reward-shaping anchor; the statistics reinforce that it is not a strawman.
- Use the reliability table to say that the main instability question is checkpoint dependence, not widespread collapse of the selected rows.

## Source Artifacts

- `artifacts/analysis/checkpoint_robustness/summary.json`
- `artifacts/analysis/rough_terrain_extended_seeds/comparison_summary.json`
- `artifacts/analysis/rough_terrain_lcp_soft_jacobian_formal/comparison_summary.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed11/checkpoint_sweep_summary.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed11/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed11/metrics_selected.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed17/checkpoint_sweep_summary.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed17/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed17/metrics_selected.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed23/checkpoint_sweep_summary.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed23/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed23/metrics_selected.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed29/checkpoint_sweep_summary.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed29/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed29/metrics_selected.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed31/checkpoint_sweep_summary.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed31/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed31/metrics_selected.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed11/checkpoint_sweep_summary.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed11/metrics_checkpoint_300.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed11/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed17/checkpoint_sweep_summary.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed17/metrics_checkpoint_400.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed17/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed23/checkpoint_sweep_summary.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed23/metrics_checkpoint_400.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed23/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed29/checkpoint_sweep_summary.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed29/metrics_checkpoint_400.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed29/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed31/checkpoint_sweep_summary.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed31/metrics_checkpoint_400.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed31/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11/checkpoint_sweep_summary.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11/metrics_selected.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17/checkpoint_sweep_summary.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17/metrics_selected.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed23/checkpoint_sweep_summary.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed23/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed23/metrics_selected.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed29/checkpoint_sweep_summary.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed29/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed29/metrics_selected.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed31/checkpoint_sweep_summary.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed31/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed31/metrics_selected.json`

## Reproduction

```bash
/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/analysis/analyze_checkpoint_robustness.py
/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/analysis/analyze_full_paper_statistics.py
```

Generated runtime summary: `artifacts/analysis/full_paper_statistics/summary.json`
Generated runtime table note: `artifacts/analysis/full_paper_statistics/summary.md`
