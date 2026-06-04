# LCP-Style Baseline Fidelity Audit (#98)

Status: `complete`.

This audit checks whether the full-paper `LCP-style soft penalty` row is
defensible as an adapted policy-sensitivity baseline rather than an official
LCP reproduction.

## Paper-Facing Parity-Gap Read

The current goal is not to claim official LCP parity. The goal is to show, in a
reviewer-facing way, which dimensions are locally aligned and which are not.

| Dimension | Official-LCP side | Local LCP-style row | Alignment read | Claim effect |
| --- | --- | --- | --- | --- |
| Penalty family | Fixed soft Lipschitz/Jacobian policy-gradient regularization | Fixed soft policy-gradient penalty on `||grad_obs log pi(a | obs)||^2` | Same family | Supports `LCP-style` wording |
| Coefficient anchor | Published fixed-coefficient gradient-penalty anchor | `lcp_weight = 0.002` in the formal row | Aligned anchor | Supports same-scale local comparison |
| Actor update path | Training-side penalty contributes to actor update | `create_graph=True` penalty contributes to PPO actor update | Aligned | Supports training-mechanism comparability |
| Penalty sampling detail | Official stack-specific rollout implementation | PPO minibatch state-action samples with subsample up to 64 actor observations | Partial | Supports adapted baseline, not code equivalence |
| Task / terrain | Official LCP task stack | Humanoid-Gym rough-terrain locomotion task | Different | No benchmark parity claim |
| Robot / morphology | Official LCP robot stack | H1-class Humanoid-Gym 12-DoF row | Different | No robot-stack parity claim |
| Checkpoint / evaluation path | Official checkpoints and evaluation pipeline | Local checkpoint sweep with Isaac/MuJoCo bridges | Different | No official checkpoint comparison |
| Hardware evidence | Official LCP real-world evidence lies outside this paper's stack | No hardware evaluation in this study | Different | No real-world comparison claim |

The paper-facing conclusion is therefore narrow: the current row is a same-task
mechanism-family comparison, not an official LCP reproduction or a claim of
parity with the original benchmark package.

## Source Mechanism Anchor

The local recipe follows the mechanism recorded in the earlier baseline recipe:

- LCP regularizes the policy through a gradient penalty on
  `||grad_s log pi(a | s)||^2` over rollout state-action samples.
- The PPO-style training objective subtracts a fixed weighted gradient penalty
  from the policy objective.
- The coefficient `lambda_gp = 0.002` is the paper-facing anchor, while nearby
  coefficients are diagnostic sensitivity checks rather than a new search for a
  best method.

## Local Implementation Check

The implementation in `LCPPPO` matches the intended local recipe:

- It uses `algorithm.lcp.lcp_weight = 0.002` for the primary formal row.
- It computes `mean(||grad_obs log pi(a_batch | obs_batch)||^2)`.
- It uses PPO minibatch observations and stored actions from rollout storage.
- It calls autograd with `create_graph=True`, so the penalty contributes to the
  actor update rather than being a logging-only side read.
- It disables the heuristic smoothness reward terms in the method config, so the
  LCP-style row is a replacement smoothness mechanism rather than a hybrid with
  action-rate reward shaping.
- It keeps evaluation-side policy-local-sensitivity logging only as a shared
  readout against the SC-PPO threshold scale.

The main local engineering difference from an unconstrained full-batch
description is computational: the implementation subsamples up to 64 minibatch
observations for the gradient penalty. This is a bounded implementation choice
for the Humanoid-Gym training stack, not a claim of official code equivalence.

## Naming Boundary

Defensible wording:

- `LCP-style soft Jacobian/Lipschitz penalty`
- `same-task LCP-style adaptation`
- `fixed-coefficient soft policy-gradient penalty`
- `SOTA-adjacent policy-sensitivity baseline`

Not defensible from the current evidence:

- `official LCP`
- `LCP reproduction`
- `official LCP checkpoint comparison`
- `state-of-the-art result over LCP`
- `proof that LCP broadly outperforms reward shaping`

## Fairness Read

The row is fair for this paper's mechanism-comparison question because it uses
the same H1 Humanoid-Gym rough-terrain task, seed set, checkpoint-sweep rule,
metric schema, and MuJoCo replay bridge as SC-PPO and the heuristic. It is not a
claim about the original LCP task, robot, checkpoints, or benchmark protocol.

The paper should keep the `LCP-style` qualifier wherever the method appears.
No primary-row implementation correction is required for the current claim
boundary. What is required is explicit documentation of the parity gap. If an
official LCP comparison is desired later, it should be opened as a separate
external-baseline issue with its own task/protocol alignment decision.

## Final Recommendation

Recommendation: keep the current row in the manuscript, but harden it with a
paper-facing parity-gap table and explicit boundary wording.

This is sufficient for the current mechanism-comparison paper because:

- the mechanism family is meaningfully aligned;
- the formal coefficient anchor is aligned;
- the actor-side training penalty is real, not a logging-only proxy; and
- the comparison is internally fair against SC-PPO and the revised heuristic
  under one shared Humanoid-Gym protocol.

It is not sufficient for any stronger claim such as official LCP parity,
benchmark-level superiority over LCP, or real-world comparison against the
official project.

## Reproduction Pointers

- Primary method config: `configs/methods/lcp_soft_jacobian_penalty_diagnostic.json`
- Training algorithm: `.external/humanoid-gym/humanoid/algo/ppo/lcp_ppo.py`
- Formal result note: `docs/full-paper/lcp-soft-penalty-formal-results.md`
- Baseline recipe note: `docs/full-paper/lcp-style-baseline-recipe.md`
