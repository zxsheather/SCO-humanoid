# Full-Paper Narrative Integration (#71)

## Decision

The full-paper narrative should be a mechanism-level comparison centered on
policy local sensitivity, not a paper claiming that SC-PPO beats all baselines.

Recommended positioning:

- **SC-PPO PID-Lagrangian** is the hard-constraint mainline developed in this
  repo. It is scientifically useful because it exposes how a policy-Jacobian
  constraint behaves under PPO, PID multiplier control, checkpoint selection,
  and seed variation.
- **LCP-style soft Jacobian/Lipschitz regularization** is the closest
  literature-aligned policy-sensitivity comparison for this project. It is the
  strongest current local-sensitivity baseline under the same Humanoid-Gym
  protocol, but it is not a universal winner over the revised heuristic and
  should not be described as an official LCP benchmark or as a SOTA claim.
- **OmniSafe PPO-Lag** is a framework-interface diagnostic. The
  failure says that a drop-in environment-side PPO-Lag migration is not a
  faithful way to train this actor-internal Jacobian cost in the current stack.
  It does not show that external constrained RL broadly fails.

## Evidence Summary

Generated paper-facing tables:

- Full-paper mechanism-comparison draft:
  `docs/paper/full-paper-mechanism-comparison-draft.md`
- T0 Isaac: `artifacts/analysis/paper_figures/table_full_paper_isaac_mechanism_comparison.md`
- T0b MuJoCo: `artifacts/analysis/paper_figures/table_matched_mujoco_mechanism_comparison.md`
- T0c LCP weight diagnostic: `artifacts/analysis/paper_figures/table_lcp_weight_sensitivity.md`
- T0d OmniSafe diagnostic: `artifacts/analysis/paper_figures/table_omnisafe_diagnostic.md`
- Statistical robustness audit: `docs/full-paper/statistical-robustness-results.md`
- Selected-vs-final checkpoint audit: `docs/full-paper/selected-vs-final-checkpoint-robustness.md`
- Matched MuJoCo mixed-evidence mechanism note:
  `docs/full-paper/mujoco-mixed-evidence-mechanism.md`
- Related-work / claim-boundary map:
  `docs/full-paper/related-work-claim-boundary-map.md`

Five-seed Isaac selected-checkpoint comparison:

| Method | Seeds | Selected ckpts | Fall | Vel. err | Jnt acc | Jitter | Return |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| LCP-style soft penalty | `11/17/23/29/31` | `300/400/400/400/400` | `0.000` | `0.490` | `117.331` | `0.212` | `118.420` |
| SC-PPO 3.8 PID | `11/17/23/29/31` | `300/300/400/400/400` | `0.170` | `0.606` | `142.955` | `0.277` | `99.349` |
| Revised heuristic | `11/17/23/29/31` | `350/300/350/400/400` | `0.150` | `0.705` | `115.317` | `0.260` | `105.326` |

MuJoCo selected replay:

| Method | Seeds | Fall | Vel. err | Jnt acc | Jitter | Return |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LCP-style soft penalty | `11/17/23/29/31` | `0.000` | `0.406` | `117.425` | `0.195` | `-599.108` |
| SC-PPO 3.8 PID | `11/17/23/29/31` | `0.010` | `0.471` | `159.718` | `0.322` | `-627.238` |
| Revised heuristic | `11/17/23/29/31` | `0.000` | `0.406` | `111.615` | `0.226` | `-456.370` |

OmniSafe diagnostic:

| Method / slice | Fall | Vel. err | Jnt acc | Jitter | Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| OmniSafe PPO-Lag diagnostic, seeds `23/29/31` | `1.000` | `1.468` | `79.309` | `0.008` | `4.386` |
| OmniSafe PPO-Lag final ckpt 400 | `1.000` | `1.109` | `321.328` | `0.741` | `2.714` |

The OmniSafe row is collapsed diagnostic evidence. It should be used only to
explain why OmniSafe is not the closest literature-aligned policy-sensitivity
baseline for this paper.

## Paper Thesis

Preferred thesis:

> Policy-local-sensitivity regularization is a useful lens for smooth humanoid
> control, but the enforcement mechanism matters. A soft LCP-style
> Jacobian/Lipschitz penalty is more robust than the repo's PID-Lagrangian
> SC-PPO hard-constraint implementation, while the revised heuristic remains a
> highly competitive reward-shaping anchor and a drop-in OmniSafe PPO-Lag
> migration is not a faithful baseline for actor-internal Jacobian costs.

Claims that remain defensible:

- LCP-style soft regularization is a strong same-task literature-aligned
  policy-sensitivity baseline.
- Paired bootstrap uncertainty supports the conservative claim that LCP is
  stronger than SC-PPO on Isaac fall/velocity/return/sensitivity and MuJoCo
  action jitter; joint-acceleration advantages over SC-PPO remain directional
  rather than statistically clean under five seeds.
- The formal LCP coefficient `0.002` is locally supported by the narrow #73
  diagnostic: neighboring values `0.001` and `0.004` are both worse on the
  `23/29/31` selected aggregate.
- LCP is close to final-checkpoint stable under the current five-seed protocol;
  selected-vs-final dependence is mainly a concern for SC-PPO dynamic
  smoothness and heuristic task validity.
- Matched MuJoCo evidence is mixed but mechanistically interpretable: LCP is
  the cleanest policy-output/action-jitter row, while the revised heuristic is
  better on aggregate joint acceleration and return; leave-one-seed checks
  preserve that split.
- SC-PPO shows that hard policy-Jacobian constraints can improve dynamic
  smoothness on the original three-seed slice, but five-seed evidence exposes
  seed sensitivity.
- Policy sensitivity remains a plausible mechanism behind dynamic smoothness
  and cross-engine degradation, but the current evidence is not a causal proof.
- OmniSafe migration failure is an implementation/interface boundary, not a
  negative result about constrained RL in general.

Claims to avoid:

- Do not say SC-PPO beats SOTA.
- Do not say LCP is officially reproduced unless official task/code/checkpoint
  parity is established.
- Do not claim global LCP hyperparameter optimality; #73 only tests the
  immediate `0.001 / 0.002 / 0.004` neighborhood.
- Do not say LCP dominates the revised heuristic across all metrics; matched
  five-seed MuJoCo shows the heuristic is better on joint acceleration and
  return.
- Do not turn the matched MuJoCo mechanism note into causal proof; it is an
  aggregate decomposition with correlation and leave-one-seed checks.
- Do not describe the five-seed bootstrap audit as a large-sample
  significance test.
- Do not say OmniSafe/CPO/external constrained RL fails.
- Do not use the matched five-seed MuJoCo table as a universal LCP win: it is a
  win over SC-PPO on dynamic smoothness, but the revised heuristic remains
  better on joint acceleration and return.
- Do not describe the local LCP-style baseline as official LCP code/checkpoint
  parity.

## Recommended Paper Structure

1. **Problem:** smooth control needs a metric and mechanism beyond action-rate
   reward shaping.
2. **Mechanism family:** policy local sensitivity as the shared object.
3. **Related work:** constrained RL for reward/cost separation,
   Lipschitz/Jacobian policy regularization for smooth locomotion, and
   sim-to-sim validation as an intermediate robustness check.
4. **Methods:** SC-PPO hard PID-Lagrangian constraint, LCP-style soft penalty,
   revised heuristic, and diagnostic OmniSafe bridge.
5. **Results:** five-seed Isaac audit first, then MuJoCo selected replay, then
   mechanism diagnostics.
6. **Discussion:** hard constraints are brittle across seeds; soft Jacobian
   regularization is stronger here; framework-level PPO-Lag migration is not a
   faithful drop-in for actor-internal Jacobian costs.

## Next Writing Tasks

- Convert the Markdown draft into the target submission format after human
  approval of the remaining label/placement decisions.
- Use generated T0/T0b/T0c tables as the primary full-paper comparison and
  diagnostic evidence.
- Move OmniSafe into a short negative-diagnostic paragraph in related
  work/discussion, not the main baseline table.
- Keep the workshop-era SC-PPO three-seed claim only as historical/contextual
  evidence unless the paper explicitly separates workshop and full-paper
  evidence tiers.
- Accepted manuscript wording decisions are recorded in
  `docs/full-paper/related-work-claim-boundary-map.md`: use `LCP-style soft
  Jacobian/Lipschitz penalty`, avoid SOTA-style wording in manuscript body, keep
  OmniSafe as a short main-text diagnostic with full table in appendix, and
  mention CPO absence in limitations/future work rather than conclusion.
