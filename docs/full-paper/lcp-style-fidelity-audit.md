# LCP-Style Baseline Fidelity Audit (#98)

Status: `complete`.

This audit checks whether the full-paper `LCP-style soft penalty` row is
defensible as an adapted policy-sensitivity baseline rather than an official
LCP reproduction.

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
boundary. If an official LCP comparison is desired later, it should be opened as
a separate external-baseline issue with its own task/protocol alignment decision.

## Reproduction Pointers

- Primary method config: `configs/methods/lcp_soft_jacobian_penalty_diagnostic.json`
- Training algorithm: `.external/humanoid-gym/humanoid/algo/ppo/lcp_ppo.py`
- Formal result note: `docs/full-paper/lcp-soft-penalty-formal-results.md`
- Baseline recipe note: `docs/full-paper/lcp-style-baseline-recipe.md`
