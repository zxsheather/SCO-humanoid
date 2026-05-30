# Policy perturbation audit

Issue: [#85](https://github.com/zxsheather/SCO-humanoid/issues/85)
Date: 2026-05-30
Status: **complete**

## Question

Does policy-local-sensitivity regularization actually reduce the local
policy-output response to controlled observation perturbations, or is the
mechanism story only an aggregate correlation with smoothness metrics?

## Protocol

This is an evaluation-only intervention audit. It does not retrain any policy.

- Methods: LCP-style soft penalty, SC-PPO 3.8 PID, revised heuristic.
- Seeds: `11/17/23/29/31`.
- Checkpoints: each method's existing selected checkpoint.
- Observation bank: pooled observations from LCP selected-checkpoint Isaac
  rollouts, using 16 envs, 64 steps, and up to 128 observations per seed.
- Scoring: each selected policy is evaluated offline on the same pooled
  observation bank.
- Perturbations: random L2-normalized observation perturbations with
  epsilons `0.005/0.01/0.02`, four directions per observation.
- Primary metric:
  `||delta_action|| / ||delta_observation||`, using deterministic action means.
- Primary epsilon: `0.01`.

Artifacts:

- `artifacts/analysis/policy_perturbation_audit/summary.json`
- `artifacts/analysis/policy_perturbation_audit/table_policy_perturbation_audit.md`
- `artifacts/analysis/policy_perturbation_audit/per_policy/*.json`
- `artifacts/analysis/policy_perturbation_audit/observation_banks/*.pt`

## Result

| Method | Seeds | Amplification mean | Amplification p90 | Bank sensitivity | Selected sensitivity | Selected jitter | Selected jnt acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LCP-style soft penalty | 5 | `0.082` | `0.133` | `2.308` | `1.890` | `0.212` | `117.331` |
| SC-PPO 3.8 PID | 5 | `0.153` | `0.241` | `4.307` | `3.630` | `0.277` | `142.955` |
| Revised heuristic | 5 | `0.289` | `0.468` | `8.341` | `7.331` | `0.260` | `115.317` |

Across the 15 method-seed rows:

- amplification mean vs selected policy sensitivity: Pearson `0.999`;
- amplification mean vs selected action jitter: Pearson `0.228`;
- amplification mean vs selected joint acceleration: Pearson `-0.093`.

## Interpretation

The intervention supports the local mechanism claim. On matched observations,
the LCP-style row has the smallest action response to controlled observation
perturbations, SC-PPO is intermediate, and the heuristic is largest. This is
the expected ordering if policy-local sensitivity regularization is reducing
policy-output amplification rather than only changing downstream rollout
metrics.

The weak relationship to joint acceleration is also informative. It reinforces
the paper's metric split: policy-local sensitivity directly controls the policy
output stream, while joint acceleration is a downstream closed-loop outcome
affected by tracking, contact timing, PD response, and simulator dynamics.

## Claim Boundary

This is not a closed-loop causal proof that sensitivity alone determines
locomotion smoothness or cross-engine degradation. The observation bank is drawn
from LCP selected-checkpoint rollouts, and the perturbations are local in
observation space. The result should be described as controlled local
intervention evidence for policy-output amplification.

Paper-facing wording:

> On a shared observation bank from selected-checkpoint rollouts, controlled
> observation perturbations produce the expected policy-output amplification
> ordering: LCP-style soft regularization has the smallest action response,
> SC-PPO is intermediate, and the revised heuristic is largest. This strengthens
> the local mechanism interpretation while preserving the boundary that
> downstream joint acceleration and return remain closed-loop outcomes.

## Recommendation

Use this audit as a mechanism-diagnostic result in the full-paper discussion or
appendix. It should not replace the main Isaac/MuJoCo tables, but it directly
improves the answer to reviewer concerns about whether policy-local sensitivity
is an actual policy-output mechanism.
