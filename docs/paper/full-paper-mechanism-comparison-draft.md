# Policy Local Sensitivity Regularization for Smooth Humanoid Locomotion

Draft status: `full-paper mechanism-comparison draft`.

This draft supersedes the older workshop-era SC-PPO-dominance framing in
`docs/paper/arxiv-workshop-manuscript.md`. It uses the current full-paper
evidence package through issues #75-#78.

Human-level TODOs before submission:

- Choose the final label: `LCP-style`, `LCP-inspired`, or `soft
  Jacobian/Lipschitz penalty`.
- Decide whether the phrase `SOTA-adjacent` should appear in the manuscript or
  remain internal planning language.
- Decide whether the OmniSafe diagnostic belongs in the main paper, appendix,
  or supplementary material.
- Decide whether the absence of an implemented CPO baseline should be stated in
  the conclusion or only in limitations.

## Abstract

Smooth humanoid locomotion is often pursued by adding action-rate or
torque-rate penalties to the reward. These penalties are practical, but they
regularize realized behavior only indirectly and require task-specific scalar
weight tuning. We study policy local sensitivity as a mechanism-level object
for smooth control: instead of only penalizing the motion that a policy has
already produced, we regulate how strongly the policy map can amplify
observation perturbations into action changes. The paper compares three
evidence lines under a shared Humanoid-Gym rough-terrain protocol: SC-PPO, a
hard policy-Jacobian constraint with a PID-Lagrangian multiplier; an LCP-style
soft Jacobian/Lipschitz penalty; and a revised action-rate heuristic. On a
five-seed Isaac Gym audit, the LCP-style penalty is the strongest current
local-sensitivity row, with zero fall rate, lower velocity error, lower action
jitter, higher return, and much lower measured policy sensitivity than SC-PPO.
SC-PPO remains useful as the hard-constraint mechanism that exposes
enforcement and seed-sensitivity trade-offs. In matched five-seed MuJoCo replay,
the LCP-style row has the lowest action jitter and is much stronger than
SC-PPO, while the revised heuristic remains better on aggregate joint
acceleration and return. This mixed MuJoCo result is not a contradiction: local
sensitivity primarily regularizes the policy-output stream, while joint
acceleration and return are downstream closed-loop outcomes involving tracking,
contact timing, PD response, and simulator dynamics. A bounded OmniSafe PPO-Lag
migration further shows that actor-internal Jacobian costs do not drop cleanly
into a standard environment-side PPO-Lagrangian interface. The resulting claim
is mechanism-level rather than SOTA-level: policy-local-sensitivity
regularization is a useful smooth-control lens, but enforcement details matter
and no single row dominates every metric.

## 1. Introduction

RL locomotion policies can satisfy task reward while producing behavior that is
dynamically rough. In humanoid locomotion this appears as high-frequency action
variation, large joint accelerations, and policies whose smoothness depends on
simulator details. A policy can look acceptable in its training engine yet
respond sharply to small observation or contact-solver differences when replayed
in a different physics engine.

The standard engineering response is reward shaping. Action-rate, torque-rate,
joint-acceleration, and similar penalties are easy to add and can work well in
practice. The cost is that the smoothness trade-off becomes a scalar tuning
problem. The policy is not directly constrained to be locally stable as a map
from observations to actions; it is only penalized after it has already emitted
actions.

This paper asks whether policy local sensitivity is a useful mechanism for
smooth humanoid control. The central object is the local derivative of the
policy output with respect to policy observations. The mechanism hypothesis is
simple: if a policy strongly amplifies small observation differences into
action differences, then simulator-specific contact and integration differences
can become action jitter and dynamic roughness. Regulating that amplification
should therefore help smoothness, but different enforcement mechanisms may have
different robustness.

The evidence does not support a simple "SC-PPO wins" story. Instead, the
strongest current result is a mechanism comparison. A soft LCP-style
Jacobian/Lipschitz penalty is more robust than the repo's current SC-PPO
hard-constraint implementation. The revised heuristic remains a strong anchor,
especially in matched MuJoCo joint acceleration and return. SC-PPO remains
scientifically useful because it exposes how a hard policy-Jacobian constraint
with PID-Lagrangian enforcement behaves under PPO, seed variation, checkpoint
selection, and cross-engine replay. OmniSafe PPO-Lag is retained as a
framework-interface diagnostic, not as a failed external constrained-RL
baseline.

The contributions are:

1. We formulate and audit policy local sensitivity as a smooth-control
   mechanism for humanoid locomotion.
2. We compare hard policy-sensitivity enforcement (SC-PPO), soft
   LCP-style Jacobian/Lipschitz regularization, and a revised action-rate
   heuristic under the same five-seed Isaac Gym protocol.
3. We add matched five-seed MuJoCo replay and show a metric split: LCP-style
   regularization is cleanest on action jitter, while the heuristic remains
   better on aggregate joint acceleration and return.
4. We report robustness audits covering paired bootstrap uncertainty,
   selected-vs-final checkpoint dependence, and local LCP coefficient
   sensitivity.
5. We document a bounded OmniSafe PPO-Lag migration diagnostic that clarifies
   the interface mismatch between environment-side PPO-Lag costs and
   actor-internal Jacobian costs.

## 2. Related Work

Constrained RL separates reward maximization from costs or limits. CPO
formulates policy optimization under explicit constraints [@Achiam2017CPO],
while Safety Gym and PPO-Lagrangian practice provide common infrastructure for
cost-aware policy learning [@Ray2019SafetyGym]. PID-Lagrangian methods address
multiplier overshoot and responsiveness in constrained deep RL
[@Stooke2020PID]. OmniSafe packages many safe-RL algorithms behind a common
infrastructure [@Ji2024OmniSafe]. Recent humanoid work also uses constrained
optimization to regulate physical costs such as energy [@Huang2026ECO]. This
paper belongs to that cost/reward separation lineage, but the cost placement is
different: the SC-PPO cost is actor-internal and depends on the derivative of
the current policy with respect to observations during PPO updates.

Smoothness in learned locomotion is commonly encouraged through reward terms on
action rate, torque rate, or acceleration. These are practical and strong
baselines; the revised heuristic in this paper is therefore treated as a
reward-shaping anchor, not as a strawman. A different family regularizes the
policy map itself. Spectral normalization constrains network Lipschitz
constants [@Miyato2018SN], and LayerNorm changes hidden activation scaling
[@Ba2016LayerNorm]. LCP directly targets smooth humanoid locomotion through a
Lipschitz/Jacobian-style policy regularizer [@Chen2024LCP], with adjacent work
exploring spectral-normalization variants for Lipschitz-constrained policies
[@Shin2025SNLCP]. Our LCP row should be read as a local same-task
LCP-style adaptation under Humanoid-Gym rough terrain, not as official LCP
checkpoint or code parity.

Cross-engine replay is intermediate validation, not hardware validation. Isaac
Gym provides high-throughput GPU simulation for robot learning
[@Makoviychuk2021IsaacGym], while MuJoCo provides a different physics engine
and contact implementation [@Todorov2012MuJoCo]. Humanoid-Gym supplies the
training scaffold and zero-shot transfer context [@Gu2024HumanoidGym]. We use
Isaac Gym for training/evaluation and aligned MuJoCo replay as a stress test
for whether dynamic smoothness survives simulator changes. We do not claim
real-robot transfer.

## 3. Methods

### 3.1 Shared PPO Task

All primary rows use the same Humanoid-Gym rough-terrain locomotion task and
PPO backbone. The robot is an H1-class humanoid with 12 DoF. The full-paper
audit uses seeds `11/17/23/29/31`, a 512-environment training budget, and a
checkpoint sweep over the trained policy. Evaluation reports fall rate,
velocity-tracking error, joint-acceleration L2, action-jitter L2, episode
return, and policy-sensitivity side reads where available.

The selected-checkpoint protocol is explicit: checkpoints are filtered by task
validity and then selected by the smoothness-first rule used throughout the
repo. Final-checkpoint behavior is reported separately in the checkpoint
robustness audit.

### 3.2 SC-PPO Hard Policy-Sensitivity Constraint

SC-PPO introduces a constraint cost based on policy local sensitivity. For a
policy `pi_theta(a | o)`, the local cost is the Frobenius norm of the
observation-to-action Jacobian:

```text
C(o) = || d pi_theta(o) / d o ||_F.
```

The batch cost for multiplier updates is a tail-sensitive statistic:

```text
J_C(theta) = Q_0.90({ C(o_i) }).
```

SC-PPO optimizes a Lagrangian objective:

```text
L(theta, lambda) = J_R(theta) - lambda * (J_C(theta) - d),
```

with threshold `d = 3.8`, `lambda_init = 0.5`, quantile aggregation, and
`lower_bound_clamp` PID integral handling. Heuristic smoothness rewards are
disabled in the SC-PPO row, so smoothness pressure comes from the Jacobian
constraint rather than reward shaping.

### 3.3 LCP-Style Soft Jacobian/Lipschitz Penalty

The LCP-style row uses a fixed soft penalty on the policy-gradient quantity
`||grad_obs log pi(a | obs)||^2`, following the mechanism family of
Lipschitz-constrained policies [@Chen2024LCP]. It uses `lcp_weight = 0.002`,
does not use a Lagrange multiplier, and disables heuristic smoothness rewards.
It shares the same task, checkpoint sweep, metric schema, and MuJoCo replay
bridge as SC-PPO and the heuristic.

This is the closest SOTA-adjacent same-task baseline in the current repo, but it
is not an official LCP reproduction. The public LCP/MimicKit stack uses a
different task, robot, checkpoint setup, and evaluation path; the paper should
therefore call this row `LCP-style` or equivalent unless official parity is
established later.

### 3.4 Revised Heuristic Reward-Shaping Anchor

The heuristic row is PPO with a tuned action-rate penalty under the formal
protocol revision. It represents the practical reward-shaping approach that the
policy-sensitivity rows must beat or explain. Because it remains highly
competitive in MuJoCo, especially on joint acceleration and return, it is a
strong anchor rather than a weak baseline.

### 3.5 OmniSafe PPO-Lag Diagnostic

OmniSafe PPO-Lag was evaluated as a bounded migration diagnostic. The adapter,
cost bridge, policy evaluation bridge, and update-hook smoke paths were
implemented, but the three-seed end-to-end diagnostic collapsed. This is not
reported as a failed external constrained-RL baseline. The result shows that a
standard environment-side PPO-Lag interface is not a faithful drop-in route for
this actor-internal Jacobian cost without algorithm-level hooks.

## 4. Experimental Protocol

The primary Isaac comparison uses five seeds, checkpoint sweeps, and the same
metric schema for LCP-style, SC-PPO, and heuristic rows. MuJoCo replay uses the
selected Isaac checkpoint for each seed under the aligned `isaac_mainline`
protocol with 20 episodes of 20 seconds and `joint_reset_noise = 0.1`.

Metrics:

- Fall rate: lower is better.
- Velocity-tracking error: lower is better.
- Joint-acceleration L2: dynamic smoothness; lower is better.
- Action-jitter L2: policy-output smoothness; lower is better.
- Episode return: higher is better.
- Policy sensitivity and violation rate: side reads for policy-local behavior.

The evidence tiers are:

- Primary full-paper rows: LCP-style soft penalty, SC-PPO 3.8 PID-Lagrangian,
  and revised heuristic.
- Robustness audits: paired bootstrap uncertainty, selected-vs-final checkpoint
  audit, LCP coefficient sensitivity.
- Diagnostics: OmniSafe PPO-Lag migration, historical workshop-era SC-PPO
  evidence, LayerNorm/dynamic-vs-kinematic smoothness, and other closed
  alternative mechanisms.

## 5. Results

### 5.1 Five-Seed Isaac Audit

Table 1 shows the primary five-seed Isaac Gym selected-checkpoint comparison.
The LCP-style soft penalty is the strongest current local-sensitivity row. It
has zero fall rate, lower velocity error, lower action jitter, higher return,
and much lower measured policy sensitivity than SC-PPO. The heuristic remains
competitive and is slightly better on Isaac joint acceleration.

| Method | Selected ckpts | Fall | Vel. err | Jnt acc | Jitter | Return | Sens. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LCP-style soft penalty | `300/400/400/400/400` | `0.000` | `0.490` | `117.331` | `0.212` | `118.420` | `1.890` |
| SC-PPO 3.8 PID | `300/300/400/400/400` | `0.170` | `0.606` | `142.955` | `0.277` | `99.349` | `3.630` |
| Revised heuristic | `350/300/350/400/400` | `0.150` | `0.705` | `115.317` | `0.260` | `105.326` | `7.331` |

The paired bootstrap audit supports the conservative statement that LCP is
stronger than SC-PPO on Isaac fall, velocity error, return, and measured
sensitivity. The joint-acceleration advantage over SC-PPO is directionally
favorable but not clean under five seeds because the confidence interval
overlaps zero. LCP also beats the heuristic on fall, velocity error, jitter,
return, and sensitivity, while heuristic joint acceleration remains
metric-competitive.

### 5.2 Matched MuJoCo Replay

Table 2 reports matched five-seed MuJoCo selected replay. The LCP-style row has
the lowest action jitter and is much stronger than SC-PPO. The revised
heuristic remains better on aggregate joint acceleration and return.

| Method | Fall | Vel. err | Jnt acc | Jitter | Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| LCP-style soft penalty | `0.000` | `0.406` | `117.425` | `0.195` | `-599.108` |
| SC-PPO 3.8 PID | `0.010` | `0.471` | `159.718` | `0.322` | `-627.238` |
| Revised heuristic | `0.000` | `0.406` | `111.615` | `0.226` | `-456.370` |

This result should be read as a control-path metric split. Policy-local
sensitivity regularization directly targets the policy-output stream, which is
consistent with LCP winning action jitter on aggregate and on three of five
seeds. Joint acceleration and return are downstream closed-loop outcomes that
also depend on velocity tracking, contact timing, PD target dynamics, and
simulator-specific response. The leave-one-seed audit preserves the split:
LCP remains the best aggregate action-jitter row in every split, while the
heuristic remains best on aggregate joint acceleration and return in every
split.

### 5.3 Cross-Engine Smoothness and Mechanism Evidence

The five-seed cross-engine factors sharpen the same interpretation. LCP has
nearly unchanged joint acceleration from Isaac to MuJoCo (`1.001x`) and lower
MuJoCo action jitter than its Isaac selected mean (`0.917x`). SC-PPO is worse
but still far from the large degradation seen in earlier non-Jacobian
replacement mechanisms. The revised heuristic also transfers smoothness well,
with joint-acceleration factor `0.968x` and jitter factor `0.869x`.

Across the matched MuJoCo method-seed rows, action jitter and joint acceleration
are positively coupled, but the all-row correlation is amplified by SC-PPO
seed 29. Excluding that row reduces the jitter/joint-acceleration correlation
from very strong to moderate. Return is more tied to velocity tracking and
seed-specific rollout behavior than to a single smoothness metric. This is why
the LCP-style row can be best on action jitter while the heuristic remains best
on aggregate return.

### 5.4 Robustness Audits

The statistical robustness audit is descriptive rather than a large-sample
significance test. With five seeds, it supports stable directions and
uncertainty intervals, not binary claims. The strongest robust statement is:
LCP is clearly stronger than the current SC-PPO hard-constraint row on Isaac
fall, velocity error, return, sensitivity, and MuJoCo action jitter.

The selected-vs-final checkpoint audit shows that LCP is close to final-only
behavior. Only seed 11 selects checkpoint 300 instead of final checkpoint 400,
and selected-to-final aggregate deltas are small. SC-PPO is more
dynamic-selection-sensitive: final checkpoints improve velocity and return but
worsen joint acceleration and jitter. The heuristic is task-selection-sensitive:
final checkpoints improve velocity and return slightly but increase fall rate
and dynamic roughness.

The LCP coefficient diagnostic tests `0.001 / 0.002 / 0.004` on seeds
`23/29/31`. The middle value, `0.002`, is best on the selected aggregate for
fall, velocity error, joint acceleration, action jitter, and return. This does
not prove global hyperparameter optimality, but it does show that the main LCP
row is not an isolated single-point accident in the immediate neighborhood.

| LCP weight | Fall | Vel. err | Jnt acc | Jitter | Return | Sens. |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.001` | `0.017` | `0.569` | `147.380` | `0.267` | `107.706` | `2.241` |
| `0.002` | `0.000` | `0.477` | `123.903` | `0.226` | `119.697` | `1.861` |
| `0.004` | `0.000` | `0.539` | `128.041` | `0.237` | `117.982` | `1.712` |

### 5.5 OmniSafe Migration Diagnostic

The OmniSafe PPO-Lag diagnostic collapsed on all three diagnostic seeds:

| Slice | Fall | Vel. err | Jnt acc | Jitter | Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| OmniSafe PPO-Lag selected diagnostic | `1.000` | `1.468` | `79.309` | `0.008` | `4.386` |
| OmniSafe PPO-Lag final ckpt 400 | `1.000` | `1.109` | `321.328` | `0.741` | `2.714` |

The low selected-checkpoint joint acceleration and jitter are not smooth
locomotion evidence because the policies fall. The useful result is the
interface diagnosis: standard PPO-Lag expects environment-side scalar costs,
whereas the SC-PPO cost is actor-internal and computed from policy derivatives
during optimization. A faithful external baseline would require algorithm-level
hooks, not only a custom environment wrapper.

## 6. Discussion

### 6.1 What the Mechanism Comparison Shows

The evidence supports policy local sensitivity as a useful smooth-control lens,
but it does not support hard-constraint SC-PPO as the strongest implementation.
The LCP-style soft penalty is currently stronger under the same task and metric
schema. This suggests that the regulated object is useful while the enforcement
mechanism matters.

Soft regularization may be easier to optimize in PPO because the penalty is
present throughout actor updates with a fixed coefficient. SC-PPO exposes the
cost of hard enforcement: the constraint is interpretable, but the PID
multiplier and checkpoint behavior are seed-sensitive. The contribution of
SC-PPO is therefore not final performance dominance; it is the mechanistic
contrast between hard constraint enforcement and soft policy-map regularization.

### 6.2 Why the Heuristic Still Matters

The revised heuristic should remain central in the paper. It is not a weak
baseline. In matched MuJoCo replay it is better than LCP on aggregate joint
acceleration and return, even though LCP is better on action jitter. This makes
the paper stronger, not weaker: the mechanism story must explain a real
trade-off rather than dismiss reward shaping.

The current reading is that action jitter is closest to the policy-output
stream, while joint acceleration and return depend on the full closed loop. A
reward-shaped policy can sometimes produce better downstream dynamics even if
its policy map is more sensitive. Conversely, a low-sensitivity policy can
produce cleaner action commands without winning every downstream metric.

### 6.3 Sim-to-Sim Scope

MuJoCo replay is valuable because it changes physics-engine details without
changing the trained policy checkpoint. It is not hardware validation. The
sim-to-sim result supports the idea that policy-output smoothness and dynamic
smoothness can be stress-tested across engines, but it does not establish
sim-to-real robustness. Domain randomization, system identification, actuator
modeling, and hardware tests remain outside the current evidence.

### 6.4 Historical Workshop-Era Evidence

The earlier workshop-era package centered on a three-seed SC-PPO-vs-heuristic
claim and on non-Jacobian replacement mechanisms that degraded more severely in
MuJoCo. That evidence remains useful context for why policy sensitivity became
the mechanism focus, but it should not carry the full-paper main claim. The
main claim path is now the five-seed Isaac and matched five-seed MuJoCo
comparison among LCP-style soft regularization, SC-PPO hard enforcement, and
the revised heuristic.

## 7. Limitations

The LCP-style row is a local same-task adaptation, not official LCP
code/checkpoint parity. The paper should not say official LCP is reproduced or
beaten.

The primary evidence uses one robot morphology, one rough-terrain task, five
training seeds, and selected-checkpoint evaluation. The selected-vs-final audit
reduces this concern for LCP but does not eliminate checkpoint dependence for
all methods.

MuJoCo replay is sim-to-sim evidence only. There is no real-robot validation,
calibrated actuator model, system identification, or multi-terrain/multi-robot
study. The actuator low-pass proxy stress test is diagnostic and should not be
written as hardware evidence.

The mechanism explanation remains aggregate-level and correlational. The
matched MuJoCo decomposition supports a control-path metric split, but it does
not prove a causal intervention from sensitivity to joint acceleration or
return. A stronger causal claim would require matched trace-level interventions
or controlled policy modifications.

The statistical audit is a five-seed uncertainty audit, not a large-sample
hypothesis test. Confidence intervals and bootstrap rank frequencies should be
reported as descriptive robustness evidence.

External constrained-RL coverage remains incomplete. OmniSafe PPO-Lag was
tested as a bounded interface diagnostic and collapsed, but this does not show
that OmniSafe, PPO-Lag, CPO, or external constrained RL broadly fail.

## 8. Conclusion

This paper should not claim that SC-PPO beats SOTA, and it should not claim
that LCP dominates every metric. The defensible result is more specific:
policy-local-sensitivity regularization is a useful mechanism for smooth
humanoid control, and the enforcement mechanism matters. A fixed soft
LCP-style Jacobian/Lipschitz penalty is the strongest current local-sensitivity
row under the same Humanoid-Gym protocol. SC-PPO remains valuable because it
shows how a hard policy-Jacobian constraint behaves under PID-Lagrangian
enforcement, seed variation, checkpoint selection, and framework migration. The
revised heuristic remains a strong reward-shaping anchor, especially in MuJoCo
joint acceleration and return. OmniSafe PPO-Lag clarifies a framework boundary:
actor-internal Jacobian costs require algorithm-level access and should not be
reduced to environment-side proxy costs. Together, these results support a
mechanism-comparison paper about local sensitivity, not a broad SOTA or
hardware-transfer claim.

## Appendix A: Evidence Map

Primary artifacts:

- Isaac full-paper table:
  `artifacts/analysis/paper_figures/table_full_paper_isaac_mechanism_comparison.md`
- Matched MuJoCo table:
  `artifacts/analysis/paper_figures/table_matched_mujoco_mechanism_comparison.md`
- LCP weight diagnostic table:
  `artifacts/analysis/paper_figures/table_lcp_weight_sensitivity.md`
- Statistical robustness:
  `docs/full-paper/statistical-robustness-results.md`
- Selected-vs-final checkpoint robustness:
  `docs/full-paper/selected-vs-final-checkpoint-robustness.md`
- MuJoCo mixed-evidence mechanism note:
  `docs/full-paper/mujoco-mixed-evidence-mechanism.md`
- Related-work and claim-boundary map:
  `docs/full-paper/related-work-claim-boundary-map.md`

## Appendix B: Suggested Figure/Table Placement

Main text:

- Table 1: Five-seed Isaac LCP/SC-PPO/heuristic comparison.
- Table 2: Matched five-seed MuJoCo selected replay.
- Table 3: LCP coefficient sensitivity or selected-vs-final checkpoint audit,
  depending on space.
- Figure 1: Sensitivity/degradation or task-vs-smoothness separation.

Appendix:

- Paired bootstrap CIs and rank stability.
- Full selected-vs-final checkpoint details.
- OmniSafe diagnostic table.
- Historical workshop-era three-seed SC-PPO evidence.
- Closed alternative mechanisms.

## Appendix C: Historical Context

The older SC-PPO workshop draft should be cited only as historical context for
the project trajectory. It used a narrower three-seed SC-PPO-vs-heuristic frame
and did not include the current LCP-style full-paper baseline, matched five-seed
MuJoCo replay, statistical audit, checkpoint robustness audit, or related-work
claim-boundary map.

## Bibliography

Use `docs/paper/references.bib`.
