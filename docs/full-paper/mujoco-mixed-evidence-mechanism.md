# Matched MuJoCo Mixed-Evidence Mechanism Note (#77)

Status: `complete`.

This note explains the matched five-seed MuJoCo split without forcing a universal winner. It uses the existing selected-checkpoint MuJoCo replays for LCP, SC-PPO, and the revised heuristic; no training or replay was rerun.

## Main Read

- LCP is the cleanest policy-output row: it has the lowest aggregate MuJoCo action jitter and wins the per-seed jitter ranking on three of five seeds.
- The revised heuristic remains a strong control-path anchor: it wins aggregate MuJoCo joint acceleration and return, but not by dominating every seed.
- SC-PPO's matched MuJoCo aggregate is mostly hurt by rough dynamic outliers, especially seed 29 in joint acceleration and action jitter.
- The metric split is coherent: policy-local sensitivity regularization suppresses action-stream variability, while joint acceleration and return also depend on closed-loop tracking, PD response, contact timing, and simulator-specific dynamics.
- Existing amplification/proxy notes support policy-output/control-stream amplification as a plausible mechanism, but the current full-paper matched LCP-vs-heuristic read remains aggregate-level and correlational.

## Aggregate Metrics

| Method | Metric | Mean | Std |
| --- | --- | ---: | ---: |
| LCP-style soft penalty | Fall | 0.000 | 0.000 |
| LCP-style soft penalty | Vel. err | 0.406 | 0.060 |
| LCP-style soft penalty | Jnt acc | 117.425 | 11.234 |
| LCP-style soft penalty | Jitter | 0.195 | 0.029 |
| LCP-style soft penalty | Return | -599.108 | 236.114 |
| SC-PPO 3.8 PID | Fall | 0.010 | 0.020 |
| SC-PPO 3.8 PID | Vel. err | 0.471 | 0.079 |
| SC-PPO 3.8 PID | Jnt acc | 159.718 | 71.345 |
| SC-PPO 3.8 PID | Jitter | 0.322 | 0.167 |
| SC-PPO 3.8 PID | Return | -627.238 | 222.350 |
| Revised heuristic | Fall | 0.000 | 0.000 |
| Revised heuristic | Vel. err | 0.406 | 0.043 |
| Revised heuristic | Jnt acc | 111.615 | 14.597 |
| Revised heuristic | Jitter | 0.226 | 0.037 |
| Revised heuristic | Return | -456.370 | 195.030 |

## Per-Seed Winners

| Seed | Fall | Vel. err | Jnt acc | Jitter | Return |
| ---: | --- | --- | --- | --- | --- |
| 11 | LCP-style soft penalty / Revised heuristic | Revised heuristic | LCP-style soft penalty | LCP-style soft penalty | Revised heuristic |
| 17 | LCP-style soft penalty / SC-PPO 3.8 PID / Revised heuristic | Revised heuristic | SC-PPO 3.8 PID | LCP-style soft penalty | SC-PPO 3.8 PID |
| 23 | LCP-style soft penalty / SC-PPO 3.8 PID / Revised heuristic | LCP-style soft penalty | LCP-style soft penalty | LCP-style soft penalty | LCP-style soft penalty |
| 29 | LCP-style soft penalty / SC-PPO 3.8 PID / Revised heuristic | LCP-style soft penalty | Revised heuristic | Revised heuristic | LCP-style soft penalty |
| 31 | LCP-style soft penalty / SC-PPO 3.8 PID / Revised heuristic | Revised heuristic | Revised heuristic | Revised heuristic | Revised heuristic |

Winner counts across the five matched seeds:

| Metric | LCP | SC-PPO | Heuristic |
| --- | ---: | ---: | ---: |
| Fall | 1.8 | 1.3 | 1.8 |
| Vel. err | 2.0 | 0.0 | 3.0 |
| Jnt acc | 2.0 | 1.0 | 2.0 |
| Jitter | 3.0 | 0.0 | 2.0 |
| Return | 2.0 | 1.0 | 2.0 |

## LCP-vs-Heuristic Seed Deltas

Delta is `LCP - heuristic`. Lower is better for fall, velocity error, joint acceleration, and jitter; higher is better for return.

| Seed | Fall delta | Vel delta | Jnt acc delta | Jitter delta | Return delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 0.000 | 0.035 | -6.421 | -0.118 | -494.073 |
| 17 | 0.000 | 0.114 | -3.041 | -0.028 | -524.137 |
| 23 | 0.000 | -0.120 | -15.246 | -0.057 | 401.366 |
| 29 | 0.000 | -0.096 | 27.258 | 0.021 | 300.823 |
| 31 | 0.000 | 0.066 | 26.500 | 0.025 | -397.671 |

## Cross-Metric Coupling

Across the 15 method-seed rows:

- corr(action jitter, joint acceleration) = `0.967`
- corr(velocity error, return) = `-0.886`
- corr(joint acceleration, return) = `-0.106`
- corr(action jitter, return) = `0.031`

Correlation sensitivity checks:

| Row set | n | corr(jitter, jnt acc) | corr(vel err, return) | corr(jitter, return) |
| --- | ---: | ---: | ---: | ---: |
| All method-seed rows | 15 | 0.967 | -0.886 | 0.031 |
| Exclude all seed-29 rows | 12 | 0.604 | -0.884 | 0.005 |
| Exclude only SC-PPO seed-29 | 14 | 0.682 | -0.886 | -0.027 |
| LCP + heuristic only | 10 | 0.616 | -0.905 | 0.342 |

Interpretation: action jitter and joint acceleration are coupled but not identical. The all-row coupling is amplified by the SC-PPO seed-29 outlier; after removing that single row, the coupling remains positive but drops from very strong to moderate. Return is more strongly tied to task tracking and seed-specific rollout behavior than to a single smoothness metric. This explains why LCP can be best on action jitter while the heuristic remains better on aggregate return.

## Leave-One-Seed Stability

| Held-out seed | Best aggregate jnt acc | Best aggregate jitter | Best aggregate return | SC-PPO jnt acc |
| ---: | --- | --- | --- | ---: |
| 11 | Revised heuristic | LCP-style soft penalty | Revised heuristic | 161.031 |
| 17 | Revised heuristic | LCP-style soft penalty | Revised heuristic | 173.548 |
| 23 | Revised heuristic | LCP-style soft penalty | Revised heuristic | 170.208 |
| 29 | Revised heuristic | LCP-style soft penalty | Revised heuristic | 125.003 |
| 31 | Revised heuristic | LCP-style soft penalty | Revised heuristic | 168.800 |

Leave-one-seed aggregates preserve the main split: LCP remains the best action-jitter row in every split, while the revised heuristic remains the best joint-acceleration and return row in every split. Removing seed 29 sharply improves SC-PPO joint acceleration (`159.718 -> 125.003`), so SC-PPO's weak MuJoCo aggregate is outlier-amplified, but the LCP-vs-heuristic trade-off is not just a seed-29 artifact.

## Mechanism Interpretation

The current evidence supports a two-stage explanation:

1. Policy-local-sensitivity regularization primarily acts on the policy-output stream. This is why LCP has the strongest aggregate action-jitter profile and why it is much cleaner than SC-PPO in matched MuJoCo replay.
2. MuJoCo joint acceleration and return are downstream closed-loop outcomes. They depend not only on action jitter, but also on velocity tracking, contact timing, PD target dynamics, and simulator-specific response. The revised heuristic can therefore beat LCP on joint acceleration/return even while having higher aggregate action jitter.

This should be written as a mechanism-level trade-off, not as a contradiction and not as an LCP universal win.

## Relation to Existing Trace/Proxy Evidence

- Amplification trace note: `docs/sc-ppo-mujoco-amplification-trace-comparison.md`
- Actuator proxy note: `docs/sc-ppo-actuator-proxy-stress.md`

The older amplification trace evidence supports the broad `policy-output/control-stream amplification -> joint acceleration` pathway for high-degradation methods. It does not directly prove the LCP-vs-heuristic split because that trace slice did not include the full-paper LCP/heuristic matched five-seed rows. The actuator-proxy result similarly supports the relevance of control-path smoothness but remains a bounded diagnostic.

## Paper Wording Guidance

- Say that LCP is the strongest current local-sensitivity row and the cleanest action-jitter row.
- Say that the revised heuristic remains a competitive reward-shaping anchor and is better on matched MuJoCo joint acceleration and return.
- Say that MuJoCo evidence is mixed but interpretable as a control-path metric split.
- Do not claim that policy sensitivity alone causally determines MuJoCo return or joint acceleration.

## Source Artifacts

- `artifacts/analysis/rough_terrain_extended_seeds/comparison_summary.json`
- `artifacts/analysis/rough_terrain_lcp_soft_jacobian_formal/comparison_summary.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed11/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed17/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed23/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed29/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed31/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed11/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed17/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed23/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed29/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed31/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed23/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed29/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed31/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `docs/full-paper/statistical-robustness-results.md`
- `docs/sc-ppo-actuator-proxy-stress.md`
- `docs/sc-ppo-mujoco-amplification-trace-comparison.md`

## Reproduction

```bash
/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/analysis/analyze_mujoco_mixed_evidence.py
```

Generated runtime summary: `artifacts/analysis/mujoco_mixed_evidence/summary.json`
Generated runtime table note: `artifacts/analysis/mujoco_mixed_evidence/summary.md`
