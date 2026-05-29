# Paper Manuscript Skeleton

## Title (working)

> Policy Local Sensitivity Regularization for Smooth Humanoid Locomotion

Alternative:

> Hard Constraints, Soft Penalties, and Framework Boundaries for Smooth
> Humanoid Control

## Abstract (working)

We investigate policy local sensitivity as a mechanism for smooth
humanoid locomotion. The full-paper evidence now separates three lines:
SC-PPO, a PPO variant that constrains the policy Jacobian with a
PID-Lagrangian multiplier; an LCP-style soft Jacobian/Lipschitz penalty
as the closest SOTA-adjacent same-task baseline; and an OmniSafe
PPO-Lag migration diagnostic. On the five-seed Isaac Gym rough-terrain
audit, the LCP-style penalty is the strongest current local-sensitivity
baseline (`fall=0.000`, `joint_acc=117.331`, `jitter=0.212`), while
SC-PPO exposes useful but seed-sensitive hard-constraint behavior
(`fall=0.170`, `joint_acc=142.955`, `jitter=0.277`). In matched
five-seed MuJoCo selected replay, LCP preserves dynamic smoothness
(`joint_acc=117.425`, `jitter=0.195`) and is much stronger than SC-PPO,
but the revised heuristic remains highly competitive and is better on
MuJoCo joint acceleration and return. The OmniSafe PPO-Lag attempt is a
framework-interface negative diagnostic, not evidence that external
constrained RL broadly fails. The defensible full-paper claim is
therefore mechanism-level: policy-local-sensitivity regularization is a
useful smooth-control lens, but enforcement details matter and no single
row dominates every metric.

## 1. Introduction

### 1.1 Problem

- RL-trained humanoid policies often produce jittery, high-frequency
  actions that stress actuators and fail to transfer across simulators
- Heuristic reward shaping (action-rate penalties) requires tedious
  weight tuning and does not guarantee smooth behavior
- Constrained RL offers a principled alternative: treat smoothness as
  a hard constraint rather than a reward term

### 1.2 Contribution

1. A hard-constraint SC-PPO implementation that enforces policy-Jacobian
   sensitivity with a PID-Lagrangian multiplier.
2. A formal same-task LCP-style soft Jacobian/Lipschitz baseline, now
   positioned as the closest SOTA-adjacent comparison.
3. A conservative five-seed Isaac and selected-checkpoint MuJoCo
   comparison showing that LCP-style soft regularization is more robust
   than the current SC-PPO hard-constraint line.
4. A bounded OmniSafe PPO-Lag migration diagnostic showing that a
   drop-in environment-side PPO-Lag interface is not a faithful baseline
   for actor-internal Jacobian costs.
5. A mechanism-level interpretation connecting policy sensitivity,
   dynamic smoothness, seed sensitivity, and cross-engine replay.

### 1.3 Scope

- Task: velocity-tracking humanoid locomotion on rough terrain
- Simulator: Isaac Gym (training), MuJoCo (cross-engine replay)
- Robot: H1-class humanoid (12 DoF)
- Evidence standard: five-seed Isaac audit for SC-PPO/heuristic/LCP;
  MuJoCo selected replay where available; checkpoint-sweep selection is
  reported explicitly

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

## 3. Methods: Policy-Sensitivity Regularization

### 3.0 Evidence tiers

- SC-PPO is the repo's hard-constraint method line.
- LCP-style soft penalty is the SOTA-adjacent baseline line and the
  strongest current local-sensitivity baseline result.
- OmniSafe PPO-Lag is a negative migration diagnostic, not a promoted
  baseline.

## 3.1 SC-PPO hard constraint

#### 3.1.1 Constrained MDP formulation

- Task reward J_R(π), constraint cost J_C(π) ≤ d
- Lagrangian: L(π, λ) = J_R(π) − λ · (J_C(π) − d)

#### 3.1.2 Policy local sensitivity constraint

- J_C = ||∂a/∂o||_F (Frobenius norm of policy Jacobian)
- Quantile aggregation: cost = Q_0.90({||J_i||_F})
- Physical meaning: limits amplification of observation perturbations
- Computation: per-action-dimension autograd with create_graph=True

#### 3.1.3 PID-Lagrangian multiplier update

- Error: e_t = cost_t − threshold
- PID: λ_t = clamp(kp·e_t + ki·∫e + kd·Δe, 0, λ_max)
- Integral mode: lower_bound_clamp (anti-windup)
- Contrast with plain dual ascent: λ_t = clamp(λ_{t-1} + η·e_t, 0, λ_max)

#### 3.1.4 Training procedure

- PPO backbone with adaptive KL-based learning rate
- Full replacement comparison: heuristic smoothness rewards zeroed
- Full-paper audit protocol: 512 envs × 400 iterations, seeds
  11/17/23/29/31

### 3.2 LCP-style soft Jacobian/Lipschitz penalty

- Fixed soft penalty on `||grad_obs log pi(a|obs)||^2`
- `lcp_weight = 0.002`, no Lagrange multiplier, no PID update
- Same Humanoid-Gym task, same checkpoint sweep, same metric schema
- Method anchor: LCP-style Lipschitz/Jacobian regularization
- Boundary: local reimplementation, not official LCP checkpoint parity

### 3.3 OmniSafe PPO-Lag migration diagnostic

- Goal: test whether OmniSafe PPO-Lag can serve as an external
  constrained-RL baseline for the same actor-internal Jacobian cost
- Result: adapter/cost/update-hook smokes were possible, but the
  three-seed diagnostic collapsed
- Interpretation: framework-interface mismatch for this cost placement,
  not broad evidence that OmniSafe or external constrained RL fails

## 4. Experimental Protocol

### 4.1 Shared evaluation schema

- Isaac: 32 envs × 20 episodes, checkpoint sweep (0/100/200/300/400)
- MuJoCo: isaac_mainline protocol, 20 episodes × 20s
- Metrics: velocity_tracking_error, fall_rate, joint_acceleration_l2,
  action_jitter_l2, episode_return
- Constraint logging: local_sensitivity, violation_rate

### 4.2 Methods compared

**Primary full-paper comparison**:
- Vanilla PPO (raw reference)
- PPO + heuristic action-rate penalty (formal anchor)
- SC-PPO 3.8 (PID-Lagrangian, threshold=3.8)
- LCP-style soft Jacobian/Lipschitz penalty (`lcp_weight=0.002`)

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

**External-framework diagnostic**:
- OmniSafe PPO-Lag migration diagnostic (collapsed; diagnostic-only)

## 5. Results

### 5.1 Five-seed Isaac full-paper audit

[Table 1: LCP vs SC-PPO vs revised heuristic, five-seed selected aggregate]

- LCP-style soft penalty: selected `300/400/400/400/400`,
  `fall=0.000`, `vel_err=0.490`, `jnt_acc=117.331`,
  `jitter=0.212`, `return=118.420`
- SC-PPO 3.8: selected `300/300/400/400/400`, `fall=0.170`,
  `vel_err=0.606`, `jnt_acc=142.955`, `jitter=0.277`,
  `return=99.349`
- Revised heuristic: selected `350/300/350/400/400`,
  `fall=0.150`, `vel_err=0.705`, `jnt_acc=115.317`,
  `jitter=0.260`, `return=105.326`
- Read: LCP is the strongest current local-sensitivity row and clearly
  stronger than SC-PPO; the revised heuristic remains very competitive,
  especially on joint acceleration

### 5.2 MuJoCo selected replay

[Table 2: matched five-seed MuJoCo selected replay]

- LCP five-seed MuJoCo selected replay: `fall=0.000`,
  `vel_err=0.406`, `jnt_acc=117.425`, `jitter=0.195`
- SC-PPO five-seed MuJoCo selected replay: `fall=0.010`,
  `vel_err=0.471`, `jnt_acc=159.718`, `jitter=0.322`
- Revised heuristic five-seed MuJoCo selected replay: `fall=0.000`,
  `vel_err=0.406`, `jnt_acc=111.615`, `jitter=0.226`
- Read: LCP has the best action jitter and is much stronger than
  SC-PPO, while the revised heuristic is better on joint acceleration
  and return

### 5.3 Sensitivity → degradation evidence chain

[Data 1: sensitivity vs degradation factor]
[Figure 2: sensitivity → degradation scatter with trend line]

- LCP five-seed selected sensitivity ~1.89 → 1.001x
  joint-acceleration degradation
- SC-PPO sensitivity ~3.6 → 1.08x degradation on the historical
  `11/17/23` anchor slice
- LayerNorm sensitivity ~10.7 → 3.5x degradation
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

### 5.7 OmniSafe migration diagnostic

- Adapter, cost bridge, policy evaluator, and update-hook smokes passed
- Three-seed OmniSafe PPO-Lag diagnostic collapsed:
  `fall=1.000`, `vel_err=1.468`, `return=4.386`
- Interpretation: the drop-in PPO-Lag environment-cost interface is not
  faithful for this actor-internal Jacobian cost in the current stack
- Boundary: do not cite this as external constrained RL broadly failing

## 6. Discussion

### 6.1 Why soft sensitivity regularization is currently stronger

- LCP penalizes a sensitivity-like quantity directly throughout PPO
  updates, without relying on a delayed multiplier controller
- SC-PPO's hard constraint is interpretable but seed-sensitive in the
  five-seed audit
- Current evidence favors policy-local-sensitivity regularization as a
  mechanism, not specifically the SC-PPO PID implementation

### 6.2 Two dimensions of smoothness

- Dynamic (force-level) vs kinematic (trajectory-level)
- Both are valid but predict different properties
- In the completed comparisons, cross-engine degradation is tracked by
  dynamic smoothness metrics more than by kinematic smoothness metrics

### 6.3 Practical implications

- Policy local sensitivity as a candidate implicit sim-to-sim regularizer
- Actuator low-pass proxy stress: SC-PPO is most stable under a bounded
  non-ideal action-to-PD-target path, strengthening but not proving the
  sim-to-real-motivated smoothness argument
- Current evidence does not replace domain randomization, system
  identification, or hardware validation
- Threshold tuning is narrow but reproducible

### 6.4 Framework boundary

- OmniSafe PPO-Lag expects environment-side scalar costs and cost
  advantages
- The local-sensitivity cost is actor-internal and depends on the
  current policy derivative with respect to observations
- A faithful external baseline therefore requires an algorithm-level
  hook or a same-task soft regularizer, not a pure custom-environment
  adapter

## 7. Limitations

### 7.1 External validation
- MuJoCo isaac_mainline is mixed evidence: LCP is strong on dynamic
  smoothness, while the revised heuristic remains competitive on
  task-side velocity/return in the shared anchor slice
- No real-robot validation
- The actuator-proxy stress test is a MuJoCo diagnostic with a chosen
  50 ms action low-pass, not a calibrated actuator or hardware result
- MuJoCo terrain (hfield) not yet usable as report-grade evidence

### 7.2 Checkpoint dependence
- SC-PPO 3.8 mainline relies on checkpoint-sweep selection
  (five-seed selected: 300/300/400/400/400)
- epochs=3 repair attempt: mixed result (1 seed improved, 1 degraded)
- LayerNorm epochs=3: selected=final=400, but degrades on smoothness
- LCP selected is close to final-only behavior: only seed11 selects
  checkpoint 300; the other four seeds select final checkpoint 400

### 7.3 Statistical scope
- 5 seeds for Isaac full-paper audit; MuJoCo anchors are mixed between
  5-seed LCP and existing 3-seed SC-PPO/heuristic rows
- Single robot morphology (H1-class)
- Single terrain type (rough terrain) for main result
- Random stairs: all methods collapsed (transfer failure, not ranking)

### 7.4 Trace evidence limitations
- 5 episodes per seed for LDLJ/SPARC (not full 20-episode sweep)
- LDLJ/SPARC developed for human movement, not validated for robots
- Only two methods compared at trace level

### 7.5 External baseline boundary
- LCP-style soft regularization is the closest SOTA-adjacent
  policy-sensitivity baseline, but it is a local same-task
  reimplementation rather than official LCP checkpoint parity
- OmniSafe PPO-Lag is recorded as a negative framework-interface
  diagnostic, not a promoted baseline and not a result about external
  constrained RL broadly
- CPO remains unimplemented

## 8. Conclusion

The full-paper result is not that SC-PPO beats SOTA, and it is not that
LCP dominates every metric. The stronger and more defensible conclusion
is that policy-local-sensitivity regularization is a useful
smooth-control mechanism, and that the enforcement mechanism matters.
The LCP-style soft Jacobian/Lipschitz penalty is currently the strongest
same-task local-sensitivity baseline and preserves dynamic smoothness in
aligned MuJoCo replay, while the revised heuristic remains a highly
competitive reward-shaping anchor. SC-PPO remains valuable as an
interpretable PID-Lagrangian hard-constraint mechanism, but its
five-seed result exposes seed sensitivity. OmniSafe PPO-Lag records a
framework-interface boundary: a drop-in environment-side cost adapter is
not a faithful baseline for actor-internal Jacobian costs. These
findings support a mechanism-level paper about local sensitivity, not a
broad SOTA or hardware-transfer claim.

## Appendix A: Figure/Table Index

All generated paper figures/tables are reproducible with:

```bash
/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/analysis/generate_paper_figures.py
```

| # | Type | Content | Source / generation command |
| --- | --- | --- | --- |
| T0 | Table | Full-paper five-seed LCP/SC-PPO/heuristic audit | `docs/full-paper/lcp-soft-penalty-formal-results.md`; `artifacts/analysis/rough_terrain_lcp_soft_jacobian_formal/comparison_summary.json`; `artifacts/analysis/rough_terrain_extended_seeds/comparison_summary.json` |
| T0b | Table | Matched five-seed MuJoCo selected replay | `docs/full-paper/matched-mujoco-anchor-results.md`; `artifacts/methods/*/*seed{11,17,23,29,31}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json` |
| T1 | Table | Historical three-seed Isaac result, revised heuristic vs SC-PPO | `artifacts/analysis/rough_terrain_formal_protocol_revision_long_budget/comparison_summary.json`; SC-PPO rows from `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed{11,17,23}/checkpoint_sweep_summary.json` |
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
| D3 | Data | MuJoCo actuator-proxy stress test | `artifacts/analysis/mujoco_actuator_proxy_stress/summary.json`, generated by `scripts/analysis/analyze_mujoco_actuator_proxy_stress.py` |

## Appendix B: Artifact Map

| Evidence layer | Canonical artifact path |
| --- | --- |
| Full-paper narrative integration | `docs/full-paper/full-paper-narrative-integration.md` |
| Matched five-seed MuJoCo anchors | `docs/full-paper/matched-mujoco-anchor-results.md` |
| LCP formal result note | `docs/full-paper/lcp-soft-penalty-formal-results.md` |
| LCP Isaac five-seed summary | `artifacts/analysis/rough_terrain_lcp_soft_jacobian_formal/comparison_summary.json` |
| LCP MuJoCo selected replay | `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed{11,17,23,29,31}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json` |
| OmniSafe diagnostic result | `docs/full-paper/omnisafe-ppolag-diagnostic-results.md` |
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
| MuJoCo actuator-proxy stress | `artifacts/analysis/mujoco_actuator_proxy_stress/summary.json`; protocol note in `docs/sc-ppo-actuator-proxy-stress.md` |
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

# Reproduce MuJoCo actuator-proxy stress test
$PYTHON_BIN scripts/baseline/run_mujoco_actuator_proxy_stress.py
$PYTHON_BIN scripts/analysis/analyze_mujoco_actuator_proxy_stress.py
```
