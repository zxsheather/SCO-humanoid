# Reviewer Risk Checklist

This checklist maps likely reviewer objections to existing or planned evidence.
Each risk is rated by severity and the strength of the current response.

## Core Claims

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
**Current response**: ADEQUATE
- We acknowledge this openly: heuristic wins on task metrics,
  SC-PPO only on action jitter in MuJoCo
- The paper claim is about cross-engine smoothness robustness,
  NOT about SC-PPO beating heuristic in MuJoCo
- The degradation pattern (SC-PPO 1.08x vs alternatives 3.5-12.7x)
  does not depend on SC-PPO "winning" MuJoCo
- **Gap**: No physical explanation for why heuristic transfers
  task performance better than SC-PPO in MuJoCo

### R3: "Results are only 3 seeds — not statistically significant"
**Severity**: MEDIUM
**Current response**: ADEQUATE
- 3 seeds is standard for locomotion RL papers
- Per-seed checkpoint sweeps provide within-seed characterization
- Selected-checkpoint aggregate reported with mean ± std
- **Gap**: No formal statistical test (t-test, bootstrap CI)
- **Mitigation option** (deferred): extended-seed matrix (#50, #51)

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
**Current response**: ADEQUATE
- CPO assessed as 2-3 week implementation effort, deferred
- Plain dual ascent provides within-family Lagrangian comparison
- 8 alternative mechanism comparisons provide breadth
- Acknowledged as limitation
- **Mitigation option** (deferred): external baseline (#52, #53)

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
**Current response**: ADEQUATE
- Acknowledged as limitation
- SC-PPO 3.8 selected: 300/300/400 (not final)
- epochs=3 repair: mixed result
- LayerNorm epochs=3: selected=final=400, but degrades on smoothness
- Checkpoint-sweep selection is documented and reproducible
- **Gap**: No universal fix for SC-PPO final-checkpoint reliability

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
- Positive: SC-PPO beats the revised heuristic anchor on Isaac rough-
  terrain task/smoothness metrics and shows low joint-acceleration
  degradation in MuJoCo
- Negative results (8 alternatives failed) serve as evidence FOR
  the robustness of the Jacobian-constraint path relative to tested
  non-Jacobian replacements, not as failures of the project
- Plain dual vs PID: positive result for PID
- Dynamic vs kinematic: positive result for both dimensions

### R13: "The cross-engine degradation table has different checkpoint bases"
**Severity**: LOW
**Current response**: GOOD
- Table explicitly marks selected-checkpoint vs final-checkpoint rows
- SC-PPO uses selected checkpoints (300/300/400)
- LayerNorm uses final checkpoints (400/400/400)
- Checkpoint basis documented in source-artifact metadata
