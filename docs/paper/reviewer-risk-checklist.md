# Reviewer Risk Checklist

This checklist maps likely reviewer objections to existing or planned evidence.
Each risk is rated by severity and the strength of the current response.

## Core Claims

### R0: "Your new LCP baseline is stronger than SC-PPO — what is the paper claiming?"
**Severity**: HIGH
**Current response**: GOOD
- The full-paper claim has been narrowed to a mechanism-level result about
  policy-local-sensitivity regularization.
- LCP-style soft Jacobian/Lipschitz regularization is positioned as the closest
  literature-aligned same-task policy-sensitivity comparison and the strongest
  current local-sensitivity row.
- SC-PPO is retained as the repo's hard-constraint PID-Lagrangian mechanism,
  useful for understanding constraint enforcement and seed sensitivity.
- The related-work and claim-boundary map now makes the local LCP boundary
  explicit: same-task LCP-style evidence, not official LCP code/checkpoint
  parity.
- We explicitly avoid claiming that SC-PPO beats SOTA or is the strongest
  current method.
- **Gap**: Official LCP checkpoint/code parity remains untested; the accepted
  manuscript label is `LCP-style soft Jacobian/Lipschitz penalty`.

### R0b: "Is `lcp_weight=0.002` cherry-picked?"
**Severity**: MEDIUM
**Current response**: ADEQUATE
- Narrow coefficient diagnostic #73 compares `0.001 / 0.002 / 0.004` on seeds
  `23/29/31` under the same checkpoint-sweep protocol.
- `0.002` is best in that local grid on fall, velocity error, joint
  acceleration, action jitter, and return.
- `0.001` appears under-regularized; `0.004` lowers sensitivity but worsens the
  aggregate policy through seed29 roughness.
- **Gap**: This is not a broad hyperparameter sweep and does not prove global
  optimality.

### R1: "The constraint threshold 3.8 is cherry-picked"
**Severity**: HIGH
**Current response**: GOOD
- Threshold sensitivity analysis shows effective window is [3.6, 3.8)
- 3.6 (tighter): seed23 collapses (cp0)
- 4.0 (looser): seed23 collapses (cp0)
- 4.2: single-seed failure, line closed
- Only 3.8 produces 3/3 task-valid results
- **Gap**: 3.7 was frozen as diagnostic, never fully tested at 3-seed

### R2: "MuJoCo is mixed evidence — SC-PPO doesn't win there"
**Severity**: HIGH
**Current response**: GOOD
- We acknowledge this openly: the full-paper MuJoCo claim is mixed and
  mechanism-level, not a claim that SC-PPO or LCP dominates every metric.
- Matched five-seed MuJoCo replay sharpens the mechanism claim: LCP reaches
  `joint_acc=117.425` and `jitter=0.195`, much better than SC-PPO
  (`joint_acc=159.718`, `jitter=0.322`).
- The LCP row does not dominate the revised heuristic: the heuristic has lower
  MuJoCo joint acceleration (`111.615`) and better return, while LCP has lower
  action jitter.
- The mixed-evidence mechanism note
  `docs/full-paper/mujoco-mixed-evidence-mechanism.md` decomposes this by
  seed and metric. LCP wins per-seed action jitter on `3/5` seeds; the revised
  heuristic wins aggregate joint acceleration and return in every leave-one-seed
  split.
- Cross-metric checks support the explanation: action jitter and joint
  acceleration are coupled but not identical, while return is more tied to
  velocity tracking and seed-specific rollout behavior.
- SC-PPO's poor aggregate is outlier-amplified by seed 29, but the
  LCP-vs-heuristic trade-off remains after excluding that seed.
- **Gap**: The mechanism note is still aggregate/correlational. It does not
  provide an intervention-level LCP-vs-heuristic causal trace or hardware
  validation.

### R3: "Results are only 3 seeds — not statistically significant"
**Severity**: MEDIUM
**Current response**: GOOD
- The full-paper Isaac audit now uses 5 seeds for SC-PPO, revised heuristic,
  and LCP.
- Matched selected-checkpoint MuJoCo replay now also covers the same five
  seeds for LCP, SC-PPO, and the revised heuristic.
- Per-seed checkpoint sweeps provide within-seed characterization.
- Selected-checkpoint aggregate reported with mean ± std
- A paired bootstrap uncertainty audit is now available in
  `docs/full-paper/statistical-robustness-results.md`.
- The audit reports seed-level paired deltas, bootstrap confidence intervals,
  and rank-stability frequencies.
- It supports conservative statements such as: LCP is clearly stronger than
  SC-PPO on Isaac fall/velocity/return/sensitivity and MuJoCo action jitter;
  joint-acceleration advantages are directional but have wide intervals.
- **Gap**: This remains a five-seed descriptive uncertainty audit, not a
  large-sample null-hypothesis significance claim.

### R4: "You only tested on one robot and one terrain"
**Severity**: MEDIUM
**Current response**: ADEQUATE
- Acknowledged as limitation
- Random stairs stress test attempted but all methods collapsed
- Cross-engine (Isaac → MuJoCo) provides a different kind of
  generalization evidence
- **Gap**: No multi-terrain or multi-robot evidence

### R5: "No real-robot validation"
**Severity**: MEDIUM
**Current response**: ADEQUATE
- Acknowledged as limitation
- MuJoCo replay provides intermediate sim-to-sim validation
- Actuator low-pass proxy stress adds bounded non-ideal control-path evidence:
  SC-PPO has the lowest proxy fall rate and smallest episode-length loss among
  SC-PPO, revised heuristic, and LayerNorm
- Jacobian constraint is presented as a sim-to-sim regularization
  hypothesis; any sim-to-real relevance remains speculative
- **Gap**: No hardware experiments

## Method Concerns

### R6: "SC-PPO requires double backward — is the compute cost justified?"
**Severity**: LOW
**Current response**: GOOD
- Jacobian computation with create_graph=True adds overhead
- But training converges within standard 400-iteration budget
- The alternative (heuristic reward tuning) has hidden human-time cost
- Plain dual ascent is cheaper but unstable (seed23 collapse)

### R7: "PID-Lagrangian vs plain dual — is PID really necessary?"
**Severity**: LOW
**Current response**: GOOD
- Full 3-seed plain dual comparison completed
- seed11 succeeds with plain dual, but seed23 collapses (sel=0)
- PID provides cross-seed stability, not per-seed performance gain
- Clear ablation result

### R8: "Why not compare against CPO / other constrained RL methods?"
**Severity**: MEDIUM
**Current response**: GOOD
- CPO feasibility formally assessed in `docs/full-paper/cpo-feasibility.md` (Issue #80, 2026-05-30).
  Conclusion: **defer**: a pure environment-side CPO adapter would not faithfully carry
  the actor-internal Jacobian cost. The #81 autograd/HVP and #82 one-update
  smokes pass, so a local/algorithm-hooked CPO-style update is technically
  plausible, but no repeated-update training run or official CPO baseline has been
  demonstrated yet.
- Plain dual ascent provides within-family Lagrangian comparison
- 8 alternative mechanism comparisons provide breadth
- LCP-style soft Jacobian/Lipschitz regularization is now the closest
  literature-aligned same-task policy-sensitivity baseline.
- OmniSafe PPO-Lag migration was tested as a bounded framework diagnostic and
  collapsed; this is reported as an interface mismatch, not as external CRL
  failing broadly.
- The related-work map records why standard environment-side PPO-Lag does not
  faithfully carry this actor-internal Jacobian cost without algorithm-level
  hooks.
- Acknowledged limitation: no official LCP checkpoint parity and no CPO row.
  CPO is a worthwhile future external-CRL implementation line, but not a
  blocker for the current mechanism-comparison draft.

## Analysis Concerns

### R9: "LDLJ/SPARC developed for human movement — valid for robots?"
**Severity**: LOW
**Current response**: GOOD
- Acknowledged as limitation in the dynamic-vs-kinematic section
- LDLJ/SPARC used as complementary evidence, not primary claims
- Primary smoothness evidence remains jnt_acc and jitter
- Trace sample size (5 episodes) acknowledged

### R10: "Selected-checkpoint dependence — why not use final checkpoint?"
**Severity**: MEDIUM
**Current response**: GOOD
- Acknowledged as limitation
- Full-paper selected-vs-final audit is available in
  `docs/full-paper/selected-vs-final-checkpoint-robustness.md`.
- LCP is close to final-only behavior: selected `300/400/400/400/400`
  versus final `400/400/400/400/400`, with small aggregate deltas.
- SC-PPO 3.8 selected: `300/300/400/400/400`; final checkpoints improve
  velocity/return but worsen joint acceleration and jitter.
- Revised heuristic selected: `350/300/350/400/400`; final checkpoints
  improve velocity/return slightly but increase fall rate and roughness.
- epochs=3 repair: mixed result
- LayerNorm epochs=3: selected=final=400, but degrades on smoothness
- Checkpoint-sweep selection is documented and reproducible
- **Gap**: No universal fix for SC-PPO final-checkpoint dynamic smoothness

### R11: "Sensitivity → degradation correlation is only 2 clean data points"
**Severity**: MEDIUM
**Current response**: ADEQUATE
- SC-PPO (3.6, ×1.08) and LayerNorm (10.7, ×3.5) are the two
  clean points with both sensitivity and non-collapsed MuJoCo data
- Action/Output Scaling sensitivity ranges are known (5.6-9.4)
  but their MuJoCo degradation is confounded by collapse (fall=1.0)
- Trend is suggestive but not statistically robust
- Matched MuJoCo traces add time-series support for policy-output/control-stream
  amplification: high-degradation methods show high jitter/joint-acceleration
  correlation with weak contact-force correlation
- Still acknowledged as correlational evidence, not an intervention-level
  causal proof or a statistically established law

## Narrative Concerns

### R12: "The paper has too many negative results — what's the positive story?"
**Severity**: LOW
**Current response**: GOOD
- Positive: policy-local-sensitivity regularization is the useful mechanism.
- Positive: LCP-style soft regularization passes the five-seed Isaac hard gate,
  preserves dynamic smoothness in MuJoCo, and clearly improves over the SC-PPO
  hard-constraint row.
- Positive: the revised heuristic remains a strong anchor, which makes the
  comparison credible rather than strawman-like.
- Positive: SC-PPO explains hard-constraint behavior and reveals why PID-style
  constraint enforcement is interpretable but seed-sensitive.
- Negative results (8 alternatives failed) serve as evidence FOR
  the robustness of the Jacobian-constraint path relative to tested
  non-Jacobian replacements, not as failures of the project
- Plain dual vs PID: positive result for PID
- Dynamic vs kinematic: positive result for both dimensions
- OmniSafe is a framework-boundary diagnostic, not a failed SOTA comparison.

### R13: "The cross-engine degradation table has different checkpoint bases"
**Severity**: LOW
**Current response**: GOOD
- The old cross-engine degradation table is now marked as historical
  workshop-era evidence in the generated manifest.
- The primary full-paper evidence uses separate generated T0/T0b tables for
  five-seed Isaac and matched five-seed MuJoCo selected-checkpoint replay.
- Table explicitly marks selected-checkpoint vs final-checkpoint rows
- SC-PPO uses selected checkpoints (300/300/400)
- LayerNorm uses final checkpoints (400/400/400)
- Checkpoint basis documented in source-artifact metadata
