# Related-Work and Claim-Boundary Map (#78)

Status: `complete`.

This note defines the full-paper related-work placement and the wording boundary
for mechanism-comparison claims. It supersedes the older SC-PPO-dominance
framing for full-paper writing.

## Positioning Summary

The paper should be framed as a mechanism comparison for smooth humanoid
locomotion, not as a single-method SOTA paper.

Defensible thesis:

> Policy-local-sensitivity regularization is a useful smooth-control lens for
> humanoid locomotion. Under the same Humanoid-Gym protocol, the LCP-style soft
> Jacobian/Lipschitz penalty is the strongest current local-sensitivity row,
> SC-PPO explains the hard-constraint/PID enforcement trade-off, the revised
> heuristic remains a strong reward-shaping anchor, and OmniSafe PPO-Lag records
> a framework-interface boundary for actor-internal Jacobian costs.

## Related-Work Map

| Area | Source anchors | How to use it | Boundary |
| --- | --- | --- | --- |
| Constrained RL / safe RL | CPO, Safety Gym, PPO-Lag/PID Lagrangian, OmniSafe | Motivates separating task reward from safety/resource/smoothness costs and explains why a Lagrangian formulation is natural | Do not claim this work advances general safe-RL theory or beats external CRL libraries |
| PID-Lagrangian enforcement | PID Lagrangian methods | Positions SC-PPO's multiplier controller and the plain-dual ablation | The contribution is an enforcement-mechanism audit for this actor-internal cost, not a new PID theory result |
| Lipschitz/Jacobian policy regularization | LCP, SN-LCP, spectral normalization | Provides the closest literature-aligned policy-sensitivity mechanism family: regulate the policy map rather than only penalizing realized motion | The local LCP-style row is same-task evidence, not official LCP code/checkpoint parity |
| Smooth locomotion engineering | Action-rate/torque-rate/joint-acceleration penalties and LCP's motivation against low-pass/filter-only fixes | Positions the revised heuristic as a strong and necessary reward-shaping anchor | Do not treat heuristic shaping as a strawman; it wins some MuJoCo metrics |
| Sim-to-sim validation | Isaac Gym, MuJoCo, Humanoid-Gym | Justifies Isaac training plus aligned MuJoCo replay as an intermediate robustness stress test | Do not claim real-robot transfer or hardware safety from MuJoCo replay |

## Claim Boundaries

| Topic | Safe wording | Unsafe wording |
| --- | --- | --- |
| LCP | "LCP-style soft Jacobian/Lipschitz regularization is the closest literature-aligned same-task policy-sensitivity baseline and the strongest current local-sensitivity row." | "We reproduce official LCP" or "we beat/validate LCP" |
| SC-PPO | "SC-PPO contributes a hard-constraint/PID-Lagrangian enforcement path and exposes seed/checkpoint sensitivity." | "SC-PPO is the best method" or "SC-PPO beats SOTA" |
| Heuristic | "The revised heuristic remains a highly competitive reward-shaping anchor and wins matched MuJoCo joint acceleration/return." | "The heuristic is weak" or "LCP dominates every metric" |
| OmniSafe | "The OmniSafe PPO-Lag migration is a bounded framework-interface diagnostic for actor-internal Jacobian costs." | "OmniSafe fails" or "external constrained RL fails" |
| MuJoCo | "Matched MuJoCo evidence is mixed but interpretable as a control-path metric split." | "MuJoCo proves causal transfer" or "MuJoCo preserves the full Isaac ranking" |
| Statistics | "Five-seed bootstrap/leave-one audits support conservative descriptive claims." | "Large-sample statistical significance" |

## What SC-PPO Still Contributes

LCP being stronger does not make SC-PPO redundant. SC-PPO contributes:

- A hard-constraint formulation of policy local sensitivity in PPO.
- A PID-Lagrangian enforcement mechanism with a plain-dual ablation.
- Evidence that hard constraints can be interpretable but seed-sensitive under
  humanoid rough-terrain PPO.
- A useful contrast against fixed-coefficient soft regularization: enforcement
  details matter even when the regulated object is similar.
- A mechanism explanation for why actor-internal Jacobian costs do not drop
  cleanly into environment-side PPO-Lag interfaces.

Preferred manuscript wording:

> SC-PPO is not the strongest current row; it is the hard-constraint mechanism
> that makes the enforcement trade-off visible. The stronger empirical row is
> the LCP-style soft penalty, which suggests that the policy-local-sensitivity
> object is useful while the hard-constrained PID implementation remains brittle
> under five-seed auditing.

## LCP Boundary

The local baseline should be called `LCP-style` or `LCP-inspired` unless the
paper later establishes official code/checkpoint/task parity.

Facts that support the same-task local row:

- It uses a fixed soft penalty on `||grad_obs log pi(a | obs)||^2`.
- It uses `lcp_weight = 0.002`, matching the published LCP coefficient anchor.
- It disables heuristic smoothness rewards and uses the same Humanoid-Gym rough
  terrain protocol, checkpoint sweep, and metric schema as SC-PPO/heuristic.
- It passes the five-seed Isaac gate and has matched five-seed MuJoCo replay.

Facts that prevent official reproduction wording:

- The public LCP/MimicKit stack targets a different task/robot/checkpoint setup.
- This repo uses H1-class Humanoid-Gym rough terrain and local evaluation
  bridges.
- The paper does not compare against official LCP checkpoints under identical
  task and metric conditions.

## OmniSafe Boundary

OmniSafe should appear in related work as a mature safe-RL infrastructure and in
the project evidence as a diagnostic path, not as a failed headline baseline.

Safe wording:

> We attempted a bounded OmniSafe PPO-Lag migration to test whether a
> framework-level PPO-Lagrangian implementation could carry the same
> actor-internal Jacobian cost. The adapter/cost/evaluation hooks were possible,
> but the end-to-end diagnostic collapsed. We therefore treat this as evidence
> of an interface mismatch for this cost placement, not as evidence against
> OmniSafe, PPO-Lag, or external constrained RL.

Why:

- Standard PPO-Lag consumes environment-side rollout costs.
- The SC-PPO cost is computed during actor updates from the current policy
  derivative with respect to observations.
- A pure environment adapter would only provide a proxy cost, which would answer
  a different scientific question.

## Manuscript Placement

Recommended Related Work ordering:

1. Constrained RL for cost/reward separation, ending with the actor-internal
   cost-placement boundary.
2. Smooth humanoid locomotion and policy-map regularization, with LCP as the
   closest literature-aligned mechanism family.
3. Sim-to-sim validation and smoothness metrics, with an explicit hardware
   boundary.

Recommended Results/Discussion placement:

- Main results: LCP vs SC-PPO vs heuristic five-seed Isaac and matched MuJoCo.
- Mechanism diagnostics: selected-vs-final, statistical robustness,
  mixed-evidence MuJoCo mechanism note.
- Appendix or short discussion: OmniSafe diagnostic and why it is not promoted
  as the external baseline.

## Accepted Submission Decisions

These choices have been accepted for the current manuscript draft:

- Use `LCP-style soft Jacobian/Lipschitz penalty` on first mention and
  `LCP-style soft penalty` thereafter.
- Avoid SOTA-style wording in manuscript body text; use
  `closest literature-aligned policy-sensitivity baseline`.
- Keep OmniSafe as a short main-text framework-interface diagnostic, with the
  full collapsed diagnostic table in appendix/supplementary material.
- Mention the absence of CPO in limitations/future work rather than in the
  conclusion.

Open planning note: CPO remains worth testing as a future external-CRL
baseline, but it should be scoped as a new feasibility/implementation line
rather than treated as missing evidence for the current draft.

## Paper-Facing References

Bibliography file updated: `docs/paper/references.bib`.

New/updated keys:

- `Chen2024LCP`: updated to IROS 2025 proceedings metadata while preserving the
  existing citation key.
- `Shin2025SNLCP`: added as an adjacent spectral-normalization/Lipschitz policy
  reference.
- `Ji2024OmniSafe`: added for OmniSafe infrastructure citation.

## Source Notes

- LCP local recipe: `docs/full-paper/lcp-style-baseline-recipe.md`
- LCP formal result: `docs/full-paper/lcp-soft-penalty-formal-results.md`
- Matched MuJoCo mixed evidence: `docs/full-paper/mujoco-mixed-evidence-mechanism.md`
- OmniSafe feasibility: `docs/full-paper/omnisafe-ppolag-feasibility.md`
- OmniSafe diagnostic: `docs/full-paper/omnisafe-ppolag-diagnostic-results.md`
- Narrative integration: `docs/full-paper/full-paper-narrative-integration.md`
- Reviewer risks: `docs/paper/reviewer-risk-checklist.md`
