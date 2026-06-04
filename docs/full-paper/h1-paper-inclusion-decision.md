# H1 Paper-Inclusion Decision (#106)

This note decides how the completed Unitree H1 single-seed mechanism probes
should be used in the paper. It is a claim-discipline document, not a new
experiment report.

## Inputs

Completed H1 artifacts:

- #110 repaired H1 vanilla baseline:
  `artifacts/methods/h1_vanilla_ppo_reference_audit/h1_reference_audit_seed5_iter300_env512/metrics_selected.json`
- #103 H1 LCP-style probe:
  `artifacts/methods/h1_lcp_soft_jacobian_penalty_probe/h1_lcp_soft_jacobian_penalty_seed5_iter300_env512/metrics_selected.json`
- #104 H1 revised-heuristic probe:
  `artifacts/methods/h1_heuristic_smoothing_probe/h1_heuristic_smoothing_action_rate_0050_seed5_iter300_env512/metrics_selected.json`
- #105 H1 SC-PPO probe:
  `artifacts/methods/h1_sc_ppo_pid_probe/h1_sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_seed5_iter300_env512/metrics_selected.json`

Primary-morphology reference:

- `docs/paper/full-paper-mechanism-comparison-draft.md`
- `docs/full-paper/full-paper-narrative-integration.md`
- `docs/full-paper/statistical-robustness-results.md`

## Compact H1 Table

Selected-checkpoint H1 summary:

| Method | Selected ckpt | Fall | Vel. err | Jnt acc | Jitter | Return | Sens. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H1 vanilla baseline (#110) | `250` | `0.000` | `0.380` | `18.881` | `0.338` | `116.250` | `14.517` |
| H1 revised heuristic (#104) | `200` | `0.000` | `0.382` | `15.105` | `0.275` | `115.061` | `12.997` |
| H1 LCP-style soft penalty (#103) | `300` | `0.100` | `0.488` | `371.901` | `2.434` | `95.919` | `3.958` |
| H1 SC-PPO 3.8 PID (#105) | `250` | `0.200` | `0.574` | `241.324` | `1.776` | `76.658` | `3.768` |

Immediate reading:

- The revised heuristic is the only H1 method slice that preserves the repaired
  H1 task floor while also improving downstream smoothness relative to the H1
  vanilla baseline.
- LCP-style and SC-PPO both reduce measured policy local sensitivity relative
  to the H1 baseline, but neither preserves task behavior.
- SC-PPO is better than H1 LCP on H1 joint acceleration and action jitter, but
  it is worse than H1 LCP on fall rate, tracking, and return.

## Ordering Comparison

Primary morphology, five-seed Isaac selected-checkpoint ordering:

- Fall: `LCP (0.000) < heuristic (0.150) < SC-PPO (0.170)`
- Velocity error: `LCP (0.490) < SC-PPO (0.606) < heuristic (0.705)`
- Joint acceleration: `heuristic (115.317) < LCP (117.331) < SC-PPO (142.955)`
- Action jitter: `LCP (0.212) < heuristic (0.260) < SC-PPO (0.277)`
- Return: `LCP (118.420) > heuristic (105.326) > SC-PPO (99.349)`
- Sensitivity: `LCP (1.890) < SC-PPO (3.630) < heuristic (7.331)`

H1 single-seed selected-checkpoint ordering:

- Fall: `heuristic (0.000) < LCP (0.100) < SC-PPO (0.200)`
- Velocity error: `heuristic (0.382) < LCP (0.488) < SC-PPO (0.574)`
- Joint acceleration: `heuristic (15.105) < SC-PPO (241.324) < LCP (371.901)`
- Action jitter: `heuristic (0.275) < SC-PPO (1.776) < LCP (2.434)`
- Return: `heuristic (115.061) > LCP (95.919) > SC-PPO (76.658)`
- Sensitivity: `SC-PPO (3.768) < LCP (3.958) << heuristic (12.997)`

Comparison result:

- H1 does **not** support a simple cross-morphology story that the primary
  morphology winner, LCP-style soft regularization, also transfers best to H1.
- H1 **does** support a narrower mechanism-level statement: lowering the
  policy-map sensitivity metric alone is not sufficient for good downstream
  locomotion. On H1, both Jacobian-based mechanisms reduce sensitivity relative
  to the H1 baseline, but the revised heuristic is the only method that
  preserves task validity while improving downstream smoothness.
- H1 therefore strengthens the paper only as a bounded diagnostic about
  mechanism transfer, not as claim-grade confirmation of the main five-seed
  ordering.

## Mechanism-Chain Reading

Recommended reading:

- H1 is **partially supportive** of the mechanism-chain interpretation.
- It supports the claim that policy-map sensitivity is an informative signal,
  because the H1 baseline and revised heuristic have much higher measured
  sensitivity than H1 LCP / H1 SC-PPO.
- But it also shows that moving the policy-map metric in the favorable
  direction is not enough. The downstream chain from policy-map sensitivity to
  policy-output smoothness to closed-loop dynamics depends strongly on the
  enforcement mechanism and morphology.
- The cleanest conservative summary is:
  `H1 says mechanism matters even more than the primary morphology alone would suggest.`

## Paper Action

Recommendation: **include in appendix / supplementary diagnostics, not in the
main results table.**

Reasoning:

- Evidence quality is limited: one morphology, one seed, no H1 MuJoCo replay,
  no H1 hardware evidence.
- The H1 branch is scientifically useful because it adds a meaningful negative
  transfer contrast, especially for LCP-style and SC-PPO.
- It should not compete with the main five-seed XBot-L comparison as if it
  were comparable-strength evidence.

Recommended paper usage:

- Main text:
  one short paragraph in Discussion or Limitations noting that a bounded H1
  diagnostic was run.
- Appendix / supplementary:
  one compact table with the four H1 rows and a short interpretation.

Recommended non-usage:

- Do not add H1 to the main comparison figure/table.
- Do not describe H1 as a cross-morphology validation of the main method.
- Do not open H1 MuJoCo replay for the current paper unless a later decision
  says the supplementary H1 package needs one more bounded portability check.

## Exact Claim Boundary

If included, use wording close to:

> We also ran a bounded single-seed Unitree H1 diagnostic under the same shared
> metric schema. On this second morphology, the revised heuristic preserved the
> repaired H1 task floor and improved downstream smoothness relative to the H1
> vanilla baseline, whereas the current LCP-style and SC-PPO Jacobian-based
> mechanisms did not. We treat this as supplementary mechanism-transfer
> evidence rather than as claim-grade cross-morphology validation.

Avoid wording like:

- `our method generalizes across morphologies`
- `LCP is robust across robots`
- `SC-PPO and LCP fail on humanoids in general`

## Exact Limitation Text

Recommended limitation paragraph:

> The multi-robot evidence is limited. Our second-morphology probe uses a
> single Unitree H1 task, a single seed, and Isaac-only evaluation. These H1
> results are therefore diagnostic rather than claim-grade: they show that
> mechanism transfer can differ sharply across morphologies, but they do not
> establish a stable cross-morphology ranking or any hardware-transfer claim.

## Final Decision

- Include H1 as **supplementary diagnostic evidence**.
- Do not use H1 to strengthen the main quantitative claim.
- Do not start H1 MuJoCo replay for the paper by default.
- Use H1 to sharpen the Discussion claim that `mechanism transfer is fragile`
  and that `low policy sensitivity alone does not guarantee smooth locomotion`.
