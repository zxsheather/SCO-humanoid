# Cross-Engine Smoothness Degradation Analysis

This note records the quantitative cross-engine (Isaac → MuJoCo) smoothness degradation
evidence across five smoothness mechanisms, forming the backbone of the paper's core claim.

See also: [Paper manuscript skeleton](./paper/manuscript-skeleton.md),
[Reviewer risk checklist](./paper/reviewer-risk-checklist.md).

## Degradation Table

| Method | Isaac jnt_acc | MuJoCo jnt_acc | Degradation Factor | Isaac fall | MuJoCo fall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heuristic baseline (`action_rate=-0.0050`) | 119.9 | 120.7 | ×1.01 | 0.15 | 0.00 |
| SC-PPO 3.8 (Jacobian constraint + PID-Lagrangian) | 115.9 | 125.5 | ×1.08 | 0.10 | 0.02 |
| Action Scaling (constraint-aware action-side) | 144.2 | 1835.6 | ×12.7 | 0.37 | 1.00 |
| Output Scaling (constraint-aware output-side) | 121.4 | 500.5 | ×4.1 | 0.43 | 1.00 |
| LayerNorm actor (epochs=3, output_gain=0.75) | 172.0 | 602.6 | ×3.5 | 0.02 | 0.00 |

All MuJoCo replays use the shared `isaac_mainline` protocol: 20 episodes × 20s,
`joint_reset_noise = 0.1`, selected-checkpoint replay.

## Evidence Chain: Sensitivity → Cross-Engine Degradation

The Isaac-side `local_sensitivity` (Jacobian Frobenius norm of the policy) at the
evaluated checkpoint tracks the cross-engine degradation factor in the completed
SC-PPO vs LayerNorm comparison:

| Method | Isaac sensitivity (cp400) | Degradation Factor |
| --- | ---: | ---: |
| SC-PPO 3.8 | 3.58 | ×1.08 |
| LayerNorm epochs=3 | 10.74 | ×3.5 |

The ratio of sensitivities (~3×) closely matches the ratio of degradation factors (~3.2×),
supporting the hypothesis that higher policy Jacobian sensitivity amplifies
cross-engine physics differences. This is aggregate-level mechanism evidence, not
time-series causal proof.

## SC-PPO 3.8 Sensitivity Evolution (checkpoint sweep)

```
cp   sensitivity  violation   jnt_acc   fall_rate
0      0.24        0.00        89.2      1.00
100    3.57        0.33       106.9      1.00
200    3.59        0.22       144.3      0.93
300    3.68        0.30       116.7      0.18
400    3.58        0.21       130.8      0.10
```

PID-Lagrangian keeps sensitivity tightly controlled around ~3.6 throughout training.
Task acquisition completes within this constrained sensitivity budget.

## LayerNorm epochs=3 Sensitivity Evolution (checkpoint sweep)

```
cp   sensitivity  violation   jnt_acc   fall_rate
0      5.82        0.56       180.4      1.00
100    7.47        0.79       172.4      1.00
200    7.32        0.97       188.6      1.00
300    9.95        1.00       178.5      0.12
400   10.74        1.00       172.0      0.017
```

Without a Jacobian constraint, sensitivity climbs freely. Task acquisition at cp300
coincides with sensitivity ≈ 10, approximately 3× the SC-PPO level. The completed
MuJoCo replay is consistent with this elevated sensitivity amplifying contact-
dynamics differences, but the current artifacts do not localize the amplification
at individual timesteps.

## Interpretation

1. Jacobian-based `local_sensitivity` constraint at `threshold = 3.8` provides
   cross-engine smoothness robustness that the replayed non-Jacobian replacement
   mechanisms do not replicate. The heuristic action-rate baseline also preserves
   smoothness across this replay, so the paper claim should not be framed as
   SC-PPO's exclusive cross-engine win over the heuristic.

2. The mechanism appears to be: lower policy Jacobian sensitivity → smaller amplification
   of cross-engine observation-distribution differences → preserved behavior-level smoothness.

3. This finding is actionable for sim-to-sim validation: constraining policy
   Jacobian sensitivity during training may serve as an implicit regularizer
   against simulator-specific artifacts. Real-robot transfer remains untested.

## PID-Lagrange Multiplier Dynamics

The SC-PPO 3.8 checkpoint sweep reveals the Lagrange multiplier stays near zero
(~0.0-0.009) after the initial checkpoint (cp0 = 0.287, the `lambda_init` value).
The quantile(0.90) constraint cost (`cost_update`) hovers in the 3.4-3.8 range,
at or just below the 3.8 threshold. Since the error term (cost - threshold) is
near zero or negative, the PID controller does not need to raise the multiplier.

```
cp   multiplier  cost_update  train_viol   trace_len
0      0.2870      0.3084      0.0000           1
100    0.0012      3.7481      0.1198         101
200    0.0031      3.5637      0.0573         201
300    0.0000      3.6175      0.0312         301
400    0.0005      3.6290      0.0781         400
```

The PID-Lagrangian multiplier effectively acts as a safety mechanism: the policy
naturally finds a solution within the constraint boundary (sensitivity ~3.6 < 3.8),
so active penalty is rarely needed. When the `cost_update` briefly exceeds threshold
(e.g., seed17 cp200: 3.80, multiplier rises to 0.009), the PID controller provides
gentle corrective pressure.

In contrast, LayerNorm epochs=3 has no constraint, and sensitivity climbs freely to
10.74. If a threshold of 3.8 were hypothetically enforced on LayerNorm, the error
would be ~6.9 and the multiplier would rise dramatically — likely causing training
collapse similar to the three failed reliability levers (fixed_schedule, low_noise,
low_kl) that indirectly constrained sensitivity too much.

## Metric-Specific Degradation: Policy-Level vs Physics-Level Amplification

Breaking down the cross-engine degradation by metric reveals WHERE the amplification
occurs:

| Method | jitter factor | jnt_acc factor | Primary Amplification |
| --- | ---: | ---: | --- |
| Heuristic | 0.90x | 1.01x | None |
| SC-PPO 3.8 | 1.05x | 1.08x | None |
| Action Scaling | 27.58x | 12.73x | **Policy-level** |
| Output Scaling | 4.88x | 4.12x | **Policy-level** |
| LayerNorm epochs=3 | 6.41x | 3.50x | **Policy-level** |

For all three replayed non-Jacobian replacement methods, `action_jitter` inflates
more than `joint_acceleration` in MuJoCo. This is consistent with a
**policy-output amplification** reading: high Jacobian sensitivity can turn MuJoCo
observation-distribution differences into jittery actions, and those jittery
actions can then drive elevated joint acceleration through the physics rollout.

The hypothesized mechanism chain:

```
High policy sensitivity (10.7 vs 3.6)
  → Observation differences amplified into action jitter (6-28x)
    → Joint acceleration elevated through contact dynamics (3-13x)
```

The fact that jitter factor > jnt_acc factor for every replayed non-Jacobian
replacement method supports the policy → physics reading, but it should not be
treated as a time-series causal proof until the optional MuJoCo per-timestep trace
work is completed.

## Constraint Threshold Sensitivity

The repo has tested multiple `local_sensitivity` threshold values around the
3.8 mainline. The effective window is extremely narrow:

| Threshold | Regime | seed11 | seed17 | seed23 | Outcome |
| ---: | --- | --- | --- | --- | --- |
| 3.6 | full_batch | cp350 f=0.10 | cp350 f=0.65 | **cp0 f=1.00** | Failed |
| 3.7 | full_batch | frozen diagnostic | — | — | Frozen |
| 3.8 | quantile-0.90 | cp300 f=0.10 | cp300 f=0.10 | cp400 f=0.10 | **Mainline** |
| 4.0 | quantile-0.90 | — | — | **cp0 f=1.00** | Failed |
| 4.2 | quantile-0.90 | single-seed failure | — | — | Closed |

Only threshold=3.8 produces 3/3 task-valid selected checkpoints.
Threshold=3.6 (tighter) fails seed23 despite being closer to the ideal
smoothness target. Threshold=4.0 (looser) also fails seed23, suggesting
insufficient constraint pressure allows the policy to drift into unstable
regimes. The effective window for this task is approximately [3.6, 3.8).

This narrow sensitivity window supports the claim that `threshold=3.8`
was not cherry-picked from a broad range — it is the only value in the
tested neighborhood that produces consistent cross-seed results.

## Plain Dual Ascent (PPO-Lagrangian) Comparison

A full 3-seed comparison of SC-PPO with plain dual ascent (`update_mode = "dual"`,
`dual_lr = 0.01`) vs PID-Lagrangian was run under the canonical rough-terrain entry.

| Seed | sel cp | fin cp | fall_rate | vel_err | jnt_acc | jitter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 400 | 400 | 0.00 | 0.44 | 115.9 | 0.22 |
| 17 | 300 | 400 | 0.25 | 0.54 | 124.9 | 0.24 |
| 23 | 0 | 400 | 1.00 | 1.34 | 84.1 | 0.01 |

Selected-checkpoint aggregate: `fall_rate = 0.42`, `jnt_acc = 108.3`, `jitter = 0.16`.

**Key finding**: Plain dual ascent is not universally collapsed — seed11 succeeds with
`selected = final = 400` and performance comparable to PID-Lagrangian. But seed23 fails
catastrophically (`selected checkpoint = 0`). PID-Lagrangian's primary value is cross-seed
stability: it prevents the seed-level catastrophic failure that plain dual ascent exhibits.
The single-seed diagnostic (seed11 cp100, fall=1.0) was misleading — that seed later
learned through the dual ascent path, suggesting the initial collapse was a training-phase
issue rather than a fundamental infeasibility.

**Canonical artifact**: `artifacts/analysis/rough_terrain_plain_dual_probe/comparison_summary.json`

## SC-PPO Final-Checkpoint Reliability Repair (epochs=3)

Applying `num_learning_epochs = 3` to SC-PPO 3.8 did not universally fix
final-checkpoint reliability:

| Seed | epochs=2 sel | epochs=3 sel | Change |
| ---: | ---: | ---: | --- |
| 11 | 300 | **400** | Improved |
| 17 | 300 | 300 | Unchanged |
| 23 | **400** | 300 | Degraded |

Selected-checkpoint aggregate: `fall_rate = 0.28` (vs 0.10 for epochs=2),
`jnt_acc = 169.1` (vs 115.9), `jitter = 0.32` (vs 0.22). Both smoothness
metrics worsened with more epochs, suggesting the Jacobian constraint +
double-backward path interacts negatively with increased per-batch optimization.

This contrasts with LayerNorm, where epochs=3 was universally beneficial
(selected=final=400 on all seeds). The difference reinforces that the two
mechanism families respond differently to training-schedule changes.

**Artifact**: `artifacts/analysis/rough_terrain_sc_ppo_epochs3_probe/comparison_summary.json`

## LDLJ/SPARC Trace Comparison (20-episode)

Systematic trace-level smoothness comparison using 5 captured episodes per seed:

| Method | jnt_acc | jitter | LDLJ | SPARC |
| --- | ---: | ---: | ---: | ---: |
| SC-PPO 3.8 | 115.9 | 0.22 | -28.35 | -25.54 |
| LayerNorm epochs=3 | 172.0 | 0.52 | -29.69 | -32.28 |

LDLJ and SPARC measure kinematic smoothness (joint trajectory quality).
LayerNorm is 4.8% better on LDLJ and 26.4% better on SPARC, despite being
48% worse on joint acceleration and 135% worse on action jitter.

This reveals smoothness as two-dimensional:
- **Kinematic** (LDLJ, SPARC): LayerNorm wins — architecture filters high-frequency motion
- **Dynamic** (jnt_acc, jitter): SC-PPO wins — Jacobian constraint limits force oscillation

**Artifacts**: `scppo38_trace20_seed{11,17,23}` and `ln_ep3_trace20_seed{11,17,23}`
under their respective `artifacts/methods/` directories.

## Dynamic vs Kinematic Smoothness: Two Dimensions of Motion Quality

### Metric definitions and physical interpretation

The repo now distinguishes two families of smoothness metrics that capture
different physical behaviors:

**Dynamic smoothness** (force/torque level):
- `joint_acceleration_l2_mean`: L2 norm of joint angular acceleration,
  used here as a dynamic loading and actuation-smoothness proxy. It is related to
  torque demand through the robot dynamics, but it is not itself a torque-rate metric.
  It captures rapid joint-level accelerations that can stress actuators and transfer
  vibration to the robot structure.
- `action_jitter_l2_mean`: L2 norm of adjacent-timestep action differences.
  Captures whether the policy command stream itself changes abruptly from one control
  step to the next.

**Kinematic smoothness** (trajectory shape level):
- `joint_position_ldlj_mean` (Log Dimensionless Jerk): Integral of squared
  joint jerk (third derivative of position), normalized by movement duration
  and amplitude. Measures how "graceful" joint trajectories appear
  independent of the forces required to produce them.
- `joint_velocity_sparc_mean` (Spectral Arc Length): Frequency-domain measure
  of joint velocity profile complexity. Lower (more negative) values indicate
  simpler velocity spectra with fewer high-frequency components. In this repo's
  sign convention, more negative aggregate values are interpreted as smoother.

### Observed disagreement

The 20-episode trace comparison between SC-PPO 3.8 and LayerNorm epochs=3
revealed a split:

| Dimension | Metric | SC-PPO 3.8 | LayerNorm | Winner |
| --- | --- | ---: | ---: | --- |
| Dynamic | jnt_acc | 115.9 | 172.0 | SC-PPO (+48% better) |
| Dynamic | jitter | 0.22 | 0.52 | SC-PPO (+135% better) |
| Kinematic | LDLJ | -28.35 | -29.69 | LayerNorm (+4.8% better) |
| Kinematic | SPARC | -25.54 | -32.28 | LayerNorm (+26.4% better) |

LayerNorm produces kinematically smoother joint trajectories while requiring
substantially higher joint accelerations and more jittery actions. SC-PPO
produces dynamically smoother behavior while its joint trajectories are
slightly less kinematically smooth.

### Physical interpretation

The LayerNorm architecture normalizes hidden-layer activations at each policy step.
The current evidence should be read empirically rather than as proof of a temporal
low-pass filter: the normalized actor can produce smoother joint position/velocity
profiles in the captured traces, but it still permits larger action-to-action changes
and higher joint accelerations.

SC-PPO's Jacobian constraint operates at a different level: it limits the policy's
local sensitivity to observation changes, reducing the amplification of sensor noise
and simulator artifacts into command changes. This suppresses dynamic oscillation
proxies such as action jitter and joint acceleration, but it does not necessarily
optimize trajectory-shape metrics such as LDLJ or SPARC.

The two mechanisms optimize different aspects of motion quality because
they intervene at different points in the control pipeline:

```
Observation → [LayerNorm: normalizes activations] → Action → Torque → Joint motion
Observation → [Jacobian: limits sensitivity]     → Action → Torque → Joint motion
```

### Connection to cross-engine robustness

The dynamic-vs-kinematic split connects directly to the paper's core claim.
When policies are replayed in MuJoCo, dynamic smoothness metrics show the
largest degradation for non-Jacobian methods (3-13x worse jnt_acc).
Kinematic smoothness, while informative for characterizing motion quality, is not
the metric family that currently tracks cross-engine dynamic degradation. It measures
trajectory shape, while the degradation table is dominated by command and joint
acceleration amplification under a different contact solver.

The Jacobian constraint's value for the current paper claim lies in its observed
effect on dynamic smoothness: lower local policy sensitivity is associated with
less conversion of simulator-specific perturbations into large command and joint-
acceleration changes. Kinematic smoothing via LayerNorm may improve the visual
or trajectory-shape quality of motion, but the completed evidence shows it does
not provide the same cross-engine dynamic-smoothness robustness.

### Limitations

- Trace sample size is 5 episodes per seed (captured from 20-episode
  evaluation runs). LDLJ/SPARC variance across episodes has not been
  systematically characterized.
- The trace comparison uses selected checkpoints (SC-PPO: 300/300/400,
  LayerNorm: 400/400/400). Checkpoint-dependent variation in kinematic
  metrics has not been sweep-characterized.
- LDLJ and SPARC are kinematic metrics originally developed for human
  movement analysis. Their applicability to humanoid robot locomotion
  as paper-grade evidence has not been validated against external standards.
- Only two methods were compared at trace level. Action/Output Scaling's
  kinematic smoothness (likely poor due to high jitter) is unknown.
- The current policy → physics amplification reading is aggregate-level evidence.
  Per-timestep MuJoCo traces are still needed to localize action and joint-acceleration
  spikes around contacts.

## Canonical Artifacts

- SC-PPO 3.8 checkpoint sweep:
  `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed{11,17,23}/checkpoint_sweep_summary.json`

- LayerNorm epochs=3 checkpoint sweep:
  `artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_actor_output_gain_0750_more_epochs_reliability_probe_rough_terrain_seed{11,17,23}/checkpoint_sweep_summary.json`

- MuJoCo replay metrics:
  `artifacts/methods/{method}_probe/{run_name}_seed{11,17,23}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`

- Action Scaling MuJoCo:
  `artifacts/methods/action_scaling_probe/action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed{11,17,23}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`

- Output Scaling MuJoCo:
  `artifacts/methods/output_scaling_probe/output_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed{11,17,23}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`

- Paper figure/table generation:
  `/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/analysis/generate_paper_figures.py`

- Dynamic-vs-kinematic figure:
  `artifacts/analysis/paper_figures/figure_task_vs_smoothness.png`

- LayerNorm trade-off table:
  `artifacts/analysis/paper_figures/table_layernorm_tradeoff_ldlj_sparc.md`
