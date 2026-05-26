# Paper Manuscript Skeleton

## Title (working)

> Jacobian-Based Policy Constraints Provide Cross-Engine Smoothness Robustness
> for Humanoid Locomotion

Alternative:

> Smoothness Is Not One Number: Dynamic Smoothness and Cross-Engine
> Robustness in Humanoid Locomotion

## Abstract (working)

We investigate whether hard-constraint policy optimization can produce
smoother humanoid locomotion than heuristic reward shaping, and whether
the resulting smoothness survives cross-engine replay. We compare
SC-PPO—a PPO variant that constrains the policy's Jacobian sensitivity
via PID-Lagrangian optimization—against a revised heuristic baseline and
eight alternative smoothness mechanisms (architectural, scaling-based,
constraint-object, and constraint-shape variants). On Isaac Gym rough-
terrain locomotion, SC-PPO improves velocity tracking, fall rate, joint
acceleration, and action jitter relative to the heuristic baseline. In
MuJoCo replay, SC-PPO does not dominate the heuristic on task metrics;
instead, the defensible cross-engine claim is about smoothness
degradation. SC-PPO (1.08x) and the heuristic baseline (1.01x) preserve
joint-acceleration smoothness, while the replayed non-Jacobian
replacement mechanisms degrade substantially (3.5x–12.7x). Policy
Jacobian sensitivity at the evaluated checkpoint tracks this degradation
pattern, suggesting that Jacobian constraints can act as implicit
sim-to-sim regularization. We further show that smoothness is at least
two-dimensional: architectural constraints (LayerNorm) improve
kinematic smoothness (LDLJ/SPARC), while Jacobian constraints improve
dynamic smoothness (joint acceleration, action jitter), and the current
evidence links the dynamic metrics more directly to cross-engine
smoothness robustness.

## 1. Introduction

### 1.1 Problem

- RL-trained humanoid policies often produce jittery, high-frequency
  actions that stress actuators and fail to transfer across simulators
- Heuristic reward shaping (action-rate penalties) requires tedious
  weight tuning and does not guarantee smooth behavior
- Constrained RL offers a principled alternative: treat smoothness as
  a hard constraint rather than a reward term

### 1.2 Contribution

1. SC-PPO: a PID-Lagrangian constrained PPO variant that enforces
   Jacobian sensitivity bounds during training
2. Systematic comparison against heuristic baseline and 8 alternative
   smoothness mechanisms under shared evaluation protocol
3. Cross-engine (Isaac → MuJoCo) replay showing that Jacobian
   constraints preserve smoothness similarly to a heuristic action-rate
   penalty, while replayed non-Jacobian replacements do not
4. Sensitivity → degradation evidence chain: policy Jacobian norm
   tracks cross-engine degradation factor in the completed comparisons
5. Demonstration that smoothness is two-dimensional: dynamic
   (force-level) vs kinematic (trajectory-level)

### 1.3 Scope

- Task: velocity-tracking humanoid locomotion on rough terrain
- Simulator: Isaac Gym (training), MuJoCo (cross-engine replay)
- Robot: H1-class humanoid (12 DoF)
- Evidence standard: 3-seed, checkpoint-sweep selection

## 2. Related Work

### 2.1 Constrained RL for control
- CPO (Achiam et al. 2017), PPO-Lagrangian, reward-constrained RL
- ECO framework for energy-constrained locomotion

### 2.2 Smoothness in locomotion
- Action-rate penalties, torque-rate penalties
- Spectral normalization, Lipschitz-constrained policies (LCP)
- Architectural regularization (LayerNorm, orthogonal parametrization)

### 2.3 Sim-to-sim and sim-to-real transfer
- Domain randomization
- System identification
- Cross-engine evaluation as intermediate validation

## 3. Method: SC-PPO

### 3.1 Constrained MDP formulation

- Task reward J_R(π), constraint cost J_C(π) ≤ d
- Lagrangian: L(π, λ) = J_R(π) − λ · (J_C(π) − d)

### 3.2 Policy local sensitivity constraint

- J_C = ||∂a/∂o||_F (Frobenius norm of policy Jacobian)
- Quantile aggregation: cost = Q_0.90({||J_i||_F})
- Physical meaning: limits amplification of observation perturbations
- Computation: per-action-dimension autograd with create_graph=True

### 3.3 PID-Lagrangian multiplier update

- Error: e_t = cost_t − threshold
- PID: λ_t = clamp(kp·e_t + ki·∫e + kd·Δe, 0, λ_max)
- Integral mode: lower_bound_clamp (anti-windup)
- Contrast with plain dual ascent: λ_t = clamp(λ_{t-1} + η·e_t, 0, λ_max)

### 3.4 Training procedure

- PPO backbone with adaptive KL-based learning rate
- Full replacement comparison: heuristic smoothness rewards zeroed
- Canonical protocol: 512 envs × 400 iterations, 3 seeds (11/17/23)

## 4. Experimental Protocol

### 4.1 Shared evaluation schema

- Isaac: 32 envs × 20 episodes, checkpoint sweep (0/100/200/300/400)
- MuJoCo: isaac_mainline protocol, 20 episodes × 20s
- Metrics: velocity_tracking_error, fall_rate, joint_acceleration_l2,
  action_jitter_l2, episode_return
- Constraint logging: local_sensitivity, violation_rate

### 4.2 Methods compared

**Primary comparison (3 methods)**:
- Vanilla PPO (raw reference)
- PPO + heuristic action-rate penalty (formal anchor)
- SC-PPO 3.8 (PID-Lagrangian, threshold=3.8)

**Alternative mechanism diagnostics (8 methods, all closed)**:
- Anisotropic constraint shape
- Action-rate hard constraint
- Spectral Normalization (SN) actor
- Orthogonal actor
- LayerNorm actor (reached Isaac internal challenge, failed MuJoCo
  smoothness transfer)
- Action-side scaling
- Output-side scaling
- Plain dual ascent (SC-PPO without PID)

## 5. Results

### 5.1 Isaac rough-terrain main result

[Table 1: revised heuristic vs SC-PPO selected-checkpoint aggregate]
[Figure 5: task-vs-dynamic-smoothness separation]

- SC-PPO 3.8 beats the revised heuristic anchor on velocity tracking,
  fall rate, joint acceleration, and action jitter; episode return is
  treated as supplemental and effectively tied
- Vanilla PPO collapses (selected=0/0/0)

### 5.2 Cross-engine degradation

[Table 2: 5-method degradation table]
[Figure 1: cross-engine degradation bar chart + factor plot]

- Among the replayed methods, only Jacobian constraint and heuristic
  penalty preserve joint-acceleration smoothness
- Replayed non-Jacobian replacements: 3.5x–12.7x degradation
- Action/Output Scaling also collapse (fall=1.0) in MuJoCo

### 5.3 Sensitivity → degradation evidence chain

[Data 1: sensitivity vs degradation factor]
[Figure 2: sensitivity → degradation scatter with trend line]

- SC-PPO sensitivity ~3.6 → 1.08x degradation
- LayerNorm sensitivity ~10.7 → 3.5x degradation
- Ratio of sensitivities (~3x) matches ratio of degradation factors (~3.2x)
- Treat sensitivity-vs-degradation as aggregate-level mechanism evidence
- Per-timestep MuJoCo traces support policy-output/control-stream
  amplification, but remain correlational rather than causal proof

### 5.4 PID-Lagrange multiplier dynamics

[Source table: SC-PPO 3.8 checkpoint sweep summaries]

- Multiplier stays near zero: constraint is naturally satisfied
- PID acts as safety mechanism, not active enforcement

### 5.5 Dynamic vs kinematic smoothness

[Table 6: LDLJ/SPARC comparison]
[Figure 4: LDLJ/SPARC bar chart]

- LayerNorm wins kinematic smoothness (LDLJ, SPARC)
- SC-PPO wins dynamic smoothness (jnt_acc, jitter)
- Dynamic smoothness is the metric family that currently tracks the
  cross-engine degradation pattern
- Discussion of control-pipeline intervention points

### 5.6 Ablation studies

- Threshold sensitivity: effective window [3.6, 3.8) (Table 3)
- PID vs plain dual ascent: PID provides cross-seed stability (Table 4)
- SC-PPO epochs=3 reliability repair: mixed result (Table 5)

## 6. Discussion

### 6.1 Why Jacobian constraints transfer

- Hypothesis: constraining ||∂a/∂o|| limits amplification of
  simulator-specific observation-distribution differences
- Evidence: sensitivity ratio ≈ degradation ratio
- Contrast with architectural/scaling mechanisms that do not
  directly constrain input-output sensitivity

### 6.2 Two dimensions of smoothness

- Dynamic (force-level) vs kinematic (trajectory-level)
- Both are valid but predict different properties
- In the completed comparisons, cross-engine degradation is tracked by
  dynamic smoothness metrics more than by kinematic smoothness metrics

### 6.3 Practical implications

- Jacobian constraint as a candidate implicit sim-to-sim regularizer
- Current evidence does not replace domain randomization, system
  identification, or hardware validation
- Threshold tuning is narrow but reproducible

## 7. Limitations

### 7.1 External validation
- MuJoCo isaac_mainline is mixed evidence: heuristic wins on task
  metrics, SC-PPO only on action jitter
- No real-robot validation
- MuJoCo terrain (hfield) not yet usable as report-grade evidence

### 7.2 Checkpoint dependence
- SC-PPO 3.8 mainline relies on checkpoint-sweep selection
  (selected: 300/300/400, not final-checkpoint-only)
- epochs=3 repair attempt: mixed result (1 seed improved, 1 degraded)
- LayerNorm epochs=3: selected=final=400, but degrades on smoothness

### 7.3 Statistical scope
- 3 seeds for main experiment (standard for locomotion RL, limited
  for formal statistical claims)
- Single robot morphology (H1-class)
- Single terrain type (rough terrain) for main result
- Random stairs: all methods collapsed (transfer failure, not ranking)

### 7.4 Trace evidence limitations
- 5 episodes per seed for LDLJ/SPARC (not full 20-episode sweep)
- LDLJ/SPARC developed for human movement, not validated for robots
- Only two methods compared at trace level

### 7.5 Deferred external baselines
- CPO deferred due to implementation complexity (2-3 week estimate)
- No published constrained-RL method compared at full 3-seed scale
- Plain dual ascent provides within-family comparison (SC-PPO − PID)
  but not cross-family

## 8. Conclusion

SC-PPO's Jacobian-based local-sensitivity constraint, enforced via
PID-Lagrangian optimization, produces smoother humanoid locomotion than
heuristic reward shaping on Isaac Gym rough terrain under the shared
rough-terrain metric schema. In MuJoCo replay, the stronger claim is not
that SC-PPO beats the heuristic baseline across task metrics; it is that
SC-PPO preserves its selected-checkpoint joint-acceleration smoothness
with low degradation (1.08x), while the replayed non-Jacobian
replacement mechanisms degrade substantially (3.5x–12.7x). The policy's
Jacobian sensitivity at the evaluated checkpoint tracks the degradation
factor, supporting the hypothesis that Jacobian constraints serve as
implicit sim-to-sim regularization. These findings suggest that
constraining policy input-output sensitivity is a promising strategy for
improving the cross-engine robustness of learned locomotion policies,
subject to the checkpoint, seed-count, terrain, and hardware-validation
limitations documented below.

## Appendix A: Figure/Table Index

All generated paper figures/tables are reproducible with:

```bash
/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/analysis/generate_paper_figures.py
```

| # | Type | Content | Source / generation command |
| --- | --- | --- | --- |
| T1 | Table | Isaac main result, revised heuristic vs SC-PPO | `artifacts/analysis/rough_terrain_formal_protocol_revision_long_budget/comparison_summary.json`; SC-PPO rows from `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed{11,17,23}/checkpoint_sweep_summary.json` |
| T2 | Table | Cross-engine degradation, 5 methods | `artifacts/analysis/paper_figures/table_cross_engine_degradation.md`, generated by `scripts/analysis/generate_paper_figures.py` |
| T3 | Table | Threshold sensitivity | `artifacts/analysis/paper_figures/table_threshold_sensitivity.md`, generated by `scripts/analysis/generate_paper_figures.py` |
| T4 | Table | Plain dual ascent vs PID-Lagrangian | `artifacts/analysis/paper_figures/table_plain_dual_vs_pid.md`, generated by `scripts/analysis/generate_paper_figures.py` |
| T5 | Table | SC-PPO epochs=3 reliability repair | `artifacts/analysis/paper_figures/table_scppo_epochs3_repair.md`, generated by `scripts/analysis/generate_paper_figures.py` |
| T6 | Table | LayerNorm dynamic-vs-kinematic trade-off | `artifacts/analysis/paper_figures/table_layernorm_tradeoff_ldlj_sparc.md`, generated by `scripts/analysis/generate_paper_figures.py` |
| F1 | Figure | Cross-engine degradation bars and factors | `artifacts/analysis/paper_figures/figure_cross_engine_degradation.png`, generated by `scripts/analysis/generate_paper_figures.py` |
| F2 | Figure | Sensitivity vs degradation scatter | `artifacts/analysis/paper_figures/figure_sensitivity_vs_degradation.png`, generated by `scripts/analysis/generate_paper_figures.py` |
| F3 | Figure | Sensitivity evolution | `artifacts/analysis/paper_figures/figure_sensitivity_evolution.png`, generated by `scripts/analysis/generate_paper_figures.py` |
| F4 | Figure | Dynamic-vs-kinematic smoothness | `artifacts/analysis/paper_figures/figure_ldlj_sparc.png`, generated by `scripts/analysis/generate_paper_figures.py` |
| F5 | Figure | Task-vs-dynamic-smoothness separation | `artifacts/analysis/paper_figures/figure_task_vs_smoothness.png`, generated by `scripts/analysis/generate_paper_figures.py` |
| D1 | Data | Structured figure/table data and provenance | `artifacts/analysis/paper_figures/paper_figures_data.json`; output manifest in `artifacts/analysis/paper_figures/manifest.json` |
| D2 | Data | MuJoCo amplification trace comparison | `artifacts/analysis/mujoco_amplification_trace_comparison/summary.json`, generated by `scripts/analysis/analyze_mujoco_amplification_traces.py` |

## Appendix B: Artifact Map

| Evidence layer | Canonical artifact path |
| --- | --- |
| Revised heuristic Isaac main result | `artifacts/analysis/rough_terrain_formal_protocol_revision_long_budget/comparison_summary.json` |
| SC-PPO 3.8 checkpoint sweep | `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_..._iter400_seed{11,17,23}/checkpoint_sweep_summary.json` |
| Heuristic MuJoCo | `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/..._seed{11,17,23}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json` |
| SC-PPO MuJoCo | `artifacts/methods/sc_ppo_pid_probe/..._seed{11,17,23}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json` |
| LayerNorm Isaac | `artifacts/methods/layernorm_actor_gain_reliability_probe/..._more_epochs_..._seed{11,17,23}/metrics_selected.json` |
| LayerNorm MuJoCo | `artifacts/methods/layernorm_actor_gain_reliability_probe/..._more_epochs_..._seed{11,17,23}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json` |
| Action Scaling Isaac | `artifacts/methods/action_scaling_probe/..._seed{11,17,23}/metrics_selected.json` |
| Action Scaling MuJoCo | `artifacts/methods/action_scaling_probe/..._seed{11,17,23}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json` |
| Output Scaling Isaac | `artifacts/methods/output_scaling_probe/..._seed{11,17,23}/metrics_selected.json` |
| Output Scaling MuJoCo | `artifacts/methods/output_scaling_probe/..._seed{11,17,23}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json` |
| Plain dual ascent | `artifacts/analysis/rough_terrain_plain_dual_probe/comparison_summary.json` |
| SC-PPO epochs=3 | `artifacts/analysis/rough_terrain_sc_ppo_epochs3_probe/comparison_summary.json` |
| LDLJ/SPARC traces | `artifacts/methods/sc_ppo_pid_probe/scppo38_trace20_seed{11,17,23}/behavior_smoothness_metrics_selected.json` |
| MuJoCo amplification traces | `artifacts/analysis/mujoco_amplification_trace_comparison/summary.json`; raw trace path patterns in `docs/sc-ppo-mujoco-amplification-trace-comparison.md` |
| Paper figures/tables | `/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/analysis/generate_paper_figures.py`; outputs under `artifacts/analysis/paper_figures/` |
| Cross-engine analysis | `docs/sc-ppo-cross-engine-degradation.md` |

## Appendix C: Reproduction Commands

```bash
# Environment
export PYTHON_BIN=/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python
export REPO_ROOT=/home/zhuoxiang/SCO-humanoid

# Validate environment
$PYTHON_BIN scripts/baseline/check_env.py

# Generate all paper figures
$PYTHON_BIN scripts/analysis/generate_paper_figures.py

# Run targeted tests
$PYTHON_BIN -m unittest \
  tests.test_baseline_common \
  tests.test_formal_comparison_runner \
  tests.test_layernorm_actor_diagnostic_runner

# Reproduce a single method training (example: SC-PPO 3.8)
env -u DISPLAY $PYTHON_BIN scripts/baseline/train_vanilla_ppo.py \
  --config configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json

# Reproduce checkpoint sweep (example)
$PYTHON_BIN scripts/baseline/evaluate_checkpoint_sweep.py \
  --config configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json \
  --run-name <run_name> --load-run <run_dir>

# Reproduce MuJoCo replay
bash scripts/baseline/run_mujoco_scppo38_parallel.sh

# Reproduce matched MuJoCo amplification traces
$PYTHON_BIN scripts/baseline/run_mujoco_amplification_traces.py
$PYTHON_BIN scripts/analysis/analyze_mujoco_amplification_traces.py
```
