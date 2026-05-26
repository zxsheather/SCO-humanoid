# Jacobian-Based Policy Constraints Provide Cross-Engine Smoothness Robustness for Humanoid Locomotion

Draft status: `arXiv / workshop-first` manuscript draft.

This draft is derived from `docs/paper/manuscript-skeleton.md` and the frozen
post-exploration evidence package. It is not a LaTeX submission yet. Citations
use Pandoc-style keys backed by `docs/paper/references.bib`.

## Abstract

Reinforcement-learning humanoid locomotion policies can achieve useful task
performance while still producing high-frequency actions and dynamic
oscillations that are sensitive to simulator details. A common engineering
response is to tune action-rate or torque-rate penalties in the reward, but this
does not directly constrain the policy's input-output sensitivity and often
requires task-specific weight search. We study an alternative formulation,
SC-PPO, a PPO variant that treats policy local sensitivity as a constrained
optimization objective. SC-PPO constrains the Frobenius norm of the policy
Jacobian with respect to observations and updates the constraint multiplier with
a PID-Lagrangian rule. On Isaac Gym rough-terrain humanoid locomotion, SC-PPO at
threshold 3.8 improves velocity tracking, fall rate, joint acceleration, and
action jitter relative to a revised heuristic action-rate baseline under a
three-seed checkpoint-sweep protocol. In aligned MuJoCo replay, however, SC-PPO
does not dominate the heuristic on task metrics. The stronger cross-engine claim
is instead about smoothness degradation: SC-PPO and the heuristic baseline
preserve joint-acceleration smoothness with degradation factors of 1.08x and
1.01x, while three replayed non-Jacobian replacement mechanisms degrade by
3.5x to 12.7x. The policy Jacobian sensitivity at evaluated checkpoints tracks
this degradation pattern, supporting the hypothesis that local-sensitivity
constraints can act as implicit sim-to-sim regularization. We also show that
smoothness is not one number: LayerNorm improves kinematic smoothness metrics
(LDLJ/SPARC), while SC-PPO improves dynamic smoothness metrics (joint
acceleration and action jitter), and the dynamic metrics are the ones that align
with cross-engine smoothness robustness in the completed evidence.

## 1. Introduction

Humanoid locomotion policies trained with on-policy reinforcement learning often
optimize task reward while exploiting simulator-specific details. A policy can
track velocity commands and remain upright in its training simulator yet still
emit rapidly varying actions, induce high joint accelerations, and degrade under
another physics engine. This matters because actuator stress, structural
vibration, and sensitivity to contact solver differences are not adequately
captured by task reward alone.

Reward shaping is the standard practical tool for addressing this problem.
Action-rate penalties, torque penalties, and related smoothness rewards are easy
to add and often effective. Their weakness is that they are indirect: they
penalize observed behavior after the policy has already produced actions, and
the trade-off against task reward depends on hand-tuned weights. A policy can
also reduce one smoothness metric while remaining sensitive to small observation
or simulator changes.

This paper studies a more direct intervention: constrain the local sensitivity
of the policy itself. We implement SC-PPO, a constrained PPO variant in which the
cost is the Frobenius norm of the policy Jacobian with respect to observations.
The method inherits the PPO training pipeline but replaces reward-only
smoothness tuning with a Lagrangian constraint on policy input-output
sensitivity. The resulting question is not whether SC-PPO is a broad
state-of-the-art constrained-RL method. The bounded question is whether a
Jacobian-based policy constraint gives more cross-engine dynamic-smoothness
robustness than several non-Jacobian replacement mechanisms under a shared
humanoid locomotion protocol.

The answer from the current evidence is nuanced. On Isaac Gym rough terrain,
SC-PPO at threshold 3.8 beats a revised heuristic action-rate baseline on task
and dynamic smoothness metrics. On aligned MuJoCo replay, the heuristic baseline
is better on task stability and velocity tracking, while SC-PPO is only slightly
better on action jitter. The main contribution is therefore not a claim that
SC-PPO wins every metric in MuJoCo. The defensible claim is that Jacobian-based
local-sensitivity constraints preserve dynamic smoothness across engines in a
way that tested non-Jacobian replacements do not.

The paper makes five contributions:

1. It formulates smooth humanoid locomotion as a constrained PPO problem where
   the constraint cost is policy local sensitivity, measured by a Jacobian
   Frobenius norm.
2. It evaluates SC-PPO against a revised heuristic action-rate baseline and
   eight alternative smoothness mechanisms under a shared rough-terrain protocol.
3. It reports a five-method Isaac-to-MuJoCo degradation comparison showing that
   SC-PPO and the heuristic baseline preserve dynamic smoothness, while
   LayerNorm, action scaling, and output scaling suffer much larger
   cross-engine degradation.
4. It connects degradation to policy sensitivity: SC-PPO has sensitivity around
   3.6 and 1.08x joint-acceleration degradation, while LayerNorm reaches
   sensitivity around 10.7 and 3.5x degradation.
5. It separates dynamic smoothness from kinematic smoothness, showing that
   LayerNorm can improve trajectory-shape metrics while still losing dynamic
   smoothness and cross-engine robustness.

The intended scope is an `arXiv / workshop-first` evidence package. The
experiments use one humanoid morphology, one primary rough-terrain task, three
training seeds, and checkpoint-sweep selection. There is no real-robot
validation, no claim of broad constrained-RL state of the art, and no external
CPO/OmniSafe baseline in the current package.

## 2. Related Work

### 2.1 Constrained reinforcement learning

Constrained reinforcement learning treats task reward and safety or resource
costs as separate objectives, typically by optimizing a constrained Markov
decision process. Classical examples include constrained policy optimization
and PPO-Lagrangian variants [@Achiam2017CPO; @Ray2019SafetyGym]. PID-style
Lagrangian updates further address oscillation and overshoot in constrained
deep RL [@Stooke2020PID]. In robotics, Lagrangian methods are attractive
because they allow reward terms and physical limits to be controlled
separately. Recent humanoid work also uses constrained optimization to regulate
energy or other hardware-relevant quantities [@Huang2026ECO].

This work follows the Lagrangian direction but changes the constrained quantity.
Instead of constraining energy directly, SC-PPO constrains the local sensitivity
of the policy. The comparison to plain dual ascent is deliberately within-family:
it tests whether the PID-Lagrangian update improves stability over a simpler
dual update under the same policy-cost definition. We do not claim that this
replaces a full external constrained-RL library comparison.

### 2.2 Smoothness in learned locomotion

Smooth locomotion is commonly encouraged with reward penalties on action rate,
torque, torque rate, or joint acceleration. These penalties are effective
engineering tools and are included here through the revised heuristic
action-rate baseline. Their limitation is that the smoothness objective is
encoded through scalar reward weights rather than a direct constraint on the
policy map.

Another line of work constrains neural-network smoothness through architectural
or normalization mechanisms, including spectral normalization, LayerNorm, and
orthogonal parametrization [@Miyato2018SN; @Ba2016LayerNorm].
Lipschitz-constrained policies are a more direct route for smooth humanoid
locomotion because they penalize or constrain the policy map itself
[@Chen2024LCP]. This repo tests several architectural and scaling alternatives
as replacement mechanisms. Most collapse before becoming task-valid; LayerNorm
is task-valid but suffers large MuJoCo dynamic-smoothness degradation.

### 2.3 Cross-engine and sim-to-real validation

Cross-engine replay is not real-world validation, but it is a useful
intermediate stress test. Isaac Gym and MuJoCo differ in contact modeling,
integration details, and numerical behavior [@Makoviychuk2021IsaacGym;
@Todorov2012MuJoCo]. A policy that is smooth only under one simulator may
amplify these differences into jittery actions or large joint accelerations in
another simulator. This paper uses the Humanoid-Gym training and sim-to-sim
scaffold [@Gu2024HumanoidGym], with Isaac Gym for training and rough-terrain
evaluation, then replays selected checkpoints in MuJoCo under an aligned
`isaac_mainline` protocol. The paper also includes a bounded MuJoCo actuator
low-pass proxy stress test, but explicitly does not claim hardware transfer.

## 3. Method: SC-PPO

### 3.1 Constrained formulation

Let pi_theta(a | o) be a policy mapping observations o to actions a. Standard
PPO optimizes a task objective J_R(theta) [@Schulman2017PPO]. SC-PPO introduces
a constraint cost J_C(theta) and optimizes the Lagrangian objective

```text
L(theta, lambda) = J_R(theta) - lambda * (J_C(theta) - d),
```

where d is a sensitivity threshold and lambda >= 0 is a learned multiplier. In
the experiments, the reward-side heuristic smoothness terms are removed from
the SC-PPO method line so that smoothness pressure is carried by the constraint
rather than by an action-rate reward term.

### 3.2 Policy local-sensitivity cost

The constraint cost is the local sensitivity of the deterministic policy action
with respect to the observation:

```text
C(o) = || d pi_theta(o) / d o ||_F.
```

For each evaluation batch, the method computes the Jacobian Frobenius norm
through autograd. The batch cost used for multiplier update is the 0.90 quantile
of per-sample local-sensitivity costs:

```text
J_C(theta) = Q_0.90({ C(o_i) }).
```

The quantile aggregation is important because humanoid locomotion contains
contact events and transient observation regimes. A mean cost can hide a tail of
high-sensitivity states, while a max cost can be dominated by isolated samples.
The 0.90 quantile gives the constraint a tail-sensitive but not single-sample
dominated update signal.

The physical interpretation is that the policy should not amplify small
observation differences into large command differences. If two simulators
produce slightly different observations around contact events, a high-Jacobian
policy can turn that difference into high action jitter. A local-sensitivity
constraint is therefore a candidate implicit regularizer for cross-engine
dynamic smoothness.

### 3.3 PID-Lagrangian multiplier update

SC-PPO uses a PID-style Lagrange multiplier controller rather than plain dual
ascent. Let

```text
e_t = J_C(theta_t) - d.
```

Plain dual ascent updates lambda by accumulating the constraint error:

```text
lambda_t = clamp(lambda_{t-1} + eta * e_t, 0, lambda_max).
```

The PID-Lagrangian update instead combines proportional, integral, and
derivative feedback with lower-bound clamping to reduce windup:

```text
lambda_t = clamp(k_p * e_t + k_i * I_t + k_d * (e_t - e_{t-1}), 0, lambda_max).
```

The current formal method uses `pid_integral_mode = lower_bound_clamp`,
`cost_aggregation = quantile(0.90)`, and threshold `d = 3.8`. The multiplier is
not interpreted as proof that every PID term is independently necessary. The
bounded claim is that the PID-Lagrangian variant is more cross-seed stable than
the matched plain-dual variant in this repo.

### 3.4 Training and checkpoint selection

Training uses the Humanoid-Gym PPO stack with the shared rough-terrain task
scaffold. The canonical formal protocol uses 512 environments, 400 training
iterations, and seeds 11, 17, and 23. Isaac evaluation uses 32 environments and
20 episodes per checkpoint. The checkpoint sweep evaluates checkpoints
0/100/200/300/400 and selects the best task-valid checkpoint according to the
repo's shared metric schema. The selected checkpoint, not necessarily the final
checkpoint, is then used for aligned MuJoCo replay.

Checkpoint-sweep selection is a limitation, but it is also part of the
reproducible evidence protocol. The main SC-PPO checkpoints are 300/300/400 for
seeds 11/17/23. The revised heuristic checkpoints are 350/300/350.

## 4. Experimental Protocol

### 4.1 Task and simulators

The primary task is velocity-tracking humanoid locomotion on Isaac Gym rough
terrain using an H1-class 12-DoF humanoid. The action is interpreted as a target
for the low-level PD control path. The MuJoCo replay protocol uses the aligned
`isaac_mainline` setting with 20 episodes, 20 seconds per episode,
`joint_reset_noise = 0.1`, and command `(vx=0.4, vy=0.0, dyaw=0.0)`.

The main metrics are:

- `velocity_tracking_error_mean`: task tracking error, lower is better.
- `fall_rate`: fraction of failed episodes, lower is better.
- `joint_acceleration_l2_mean`: dynamic smoothness proxy, lower is better.
- `action_jitter_l2_mean`: adjacent-action command variation, lower is better.
- `episode_return_mean`: supplemental reward summary.
- `policy_local_sensitivity_cost_mean`: policy Jacobian sensitivity diagnostic.

### 4.2 Compared methods

The primary rough-terrain comparison includes three rows:

- Vanilla PPO, used as a raw collapsed reference.
- PPO with revised heuristic action-rate smoothing at `action_rate = -0.0050`.
- SC-PPO threshold 3.8 with PID-Lagrangian local-sensitivity constraint.

The post-freeze same-question challenge evaluates eight additional mechanism
families:

- anisotropic local-sensitivity constraint shape,
- action-rate hard constraint,
- spectral-normalized actor,
- orthogonal actor,
- LayerNorm actor with output gain 0.75 and three learning epochs,
- action-side scaling,
- output-side scaling,
- plain dual ascent replacing PID-Lagrangian.

These rows are not all promoted to full task-valid baselines. They are included
to test whether alternative mechanisms can preserve the same dynamic-smoothness
property.

### 4.3 Evidence boundary

The main evidence standard is three seeds with checkpoint-sweep selection. The
paper reports mean and standard deviation where available, but it does not claim
formal statistical significance. External constrained-RL baselines such as CPO
or OmniSafe are deferred because integrating them into the Humanoid-Gym stack
with matched checkpoint sweeps and MuJoCo replay was judged outside the current
`arXiv / workshop-first` scope.

## 5. Results

### 5.1 Isaac rough-terrain main result

Table 1 shows the main Isaac rough-terrain comparison. Vanilla PPO collapses and
serves only as a raw reference. The revised heuristic anchor is task-valid,
which makes it the proper baseline row for the main comparison. SC-PPO 3.8
improves all shared task and dynamic-smoothness metrics relative to that
heuristic anchor: velocity error decreases from 0.755 to 0.641, fall rate from
0.150 to 0.100, joint acceleration from 119.864 to 115.908, and action jitter
from 0.271 to 0.221. Episode return is effectively tied and is treated as a
supplemental metric.

| Method | Evidence scope | Velocity error | Joint acceleration | Action jitter | Episode return | Fall rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Vanilla PPO | raw reference, selected 0/0/0 | 1.332 +- 0.118 | 83.718 +- 13.369 | 0.016 +- 0.001 | 4.000 +- 0.432 | 1.000 +- 0.000 |
| Heuristic action-rate baseline | revised anchor, selected 350/300/350 | 0.755 +- 0.107 | 119.864 +- 2.197 | 0.271 +- 0.008 | 100.933 +- 11.271 | 0.150 +- 0.082 |
| SC-PPO 3.8 | PID-Lagrangian, selected 300/300/400 | 0.641 +- 0.055 | 115.908 +- 6.939 | 0.221 +- 0.002 | 100.284 +- 2.715 | 0.100 +- 0.000 |

![Task-vs-dynamic-smoothness separation](../../artifacts/analysis/paper_figures/figure_task_vs_smoothness.png)

This result supports the Isaac-side statement that SC-PPO 3.8 is better than
the revised heuristic anchor under the shared rough-terrain metric schema. It
does not by itself establish cross-engine superiority.

### 5.2 Cross-engine degradation

Table 2 reports the paper's central cross-engine evidence. All MuJoCo rows use
the aligned `isaac_mainline` replay protocol and the same selected checkpoints
as the Isaac comparison. The crucial quantity is the degradation factor:

```text
degradation = MuJoCo joint_acceleration_l2_mean / Isaac joint_acceleration_l2_mean.
```

| Method | Isaac jnt acc | MuJoCo jnt acc | Degradation | Isaac fall | MuJoCo fall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heuristic baseline | 119.864 | 120.734 | 1.01x | 0.150 | 0.000 |
| SC-PPO 3.8 (Jacobian) | 115.908 | 125.541 | 1.08x | 0.100 | 0.017 |
| LayerNorm epochs=3 | 171.977 | 602.578 | 3.50x | 0.017 | 0.000 |
| Action Scaling | 144.174 | 1835.590 | 12.73x | 0.367 | 1.000 |
| Output Scaling | 121.362 | 500.480 | 4.12x | 0.433 | 1.000 |

![Cross-engine degradation](../../artifacts/analysis/paper_figures/figure_cross_engine_degradation.png)

The heuristic baseline and SC-PPO both preserve joint-acceleration smoothness
across engines. The three replayed non-Jacobian replacements do not:
LayerNorm degrades by 3.5x, output scaling by 4.1x, and action scaling by 12.7x.
Action scaling and output scaling also collapse in MuJoCo with fall rate 1.0.

This is the paper's safest central claim. It should not be phrased as
`SC-PPO beats the heuristic in MuJoCo`, because it does not. In aligned MuJoCo
replay, the heuristic row has lower velocity error, lower fall rate, longer
episodes, and slightly lower joint acceleration. The correct statement is that
SC-PPO preserves dynamic smoothness across engines and that several tested
non-Jacobian replacements fail to preserve it.

### 5.3 Sensitivity tracks degradation

The mechanism hypothesis is that high policy local sensitivity amplifies
cross-engine observation and contact differences into action jitter, which then
drives elevated joint accelerations. The cleanest completed comparison is
between SC-PPO and LayerNorm, because both have task-valid MuJoCo replay and
available sensitivity diagnostics.

| Method | Isaac sensitivity | Joint-acceleration degradation |
| --- | ---: | ---: |
| SC-PPO 3.8 | 3.58 | 1.08x |
| LayerNorm epochs=3 | 10.74 | 3.50x |

![Sensitivity vs degradation](../../artifacts/analysis/paper_figures/figure_sensitivity_vs_degradation.png)

The ratio of sensitivities is approximately 3x, close to the ratio of
degradation factors at approximately 3.2x. This is suggestive mechanism
evidence rather than a statistically established law. The clean sample size is
small, and the action/output scaling rows are confounded by MuJoCo collapse.
Nevertheless, the result is consistent with the intended intervention:
constraining policy sensitivity limits the conversion of simulator differences
into command-level oscillation.

The checkpoint evolution provides additional support. SC-PPO keeps sensitivity
near 3.6 throughout training once task acquisition begins, while LayerNorm
allows sensitivity to climb toward 10.7 as it becomes task-valid.

![Sensitivity evolution](../../artifacts/analysis/paper_figures/figure_sensitivity_evolution.png)

The MuJoCo amplification trace comparison localizes the effect at the
control-stream level. LayerNorm and action scaling show high
`corr(jitter, jnt_acc)` values (0.708 and 0.659) with weak
`corr(contact, jnt_acc)` values (0.063 and -0.035), while SC-PPO keeps smaller
action-jitter and joint-acceleration tails. This argues against a purely
contact-force explanation and supports a policy-output amplification reading,
still short of time-series causal proof.

### 5.4 PID-Lagrangian versus plain dual ascent

A natural concern is whether the result depends on the PID multiplier update or
whether ordinary dual ascent would be enough. The matched plain-dual comparison
uses the same threshold and local-sensitivity cost but replaces PID-Lagrangian
with `update_mode = "dual"` and `dual_lr = 0.01`.

| Method | Basis | Checkpoints 11/17/23 | Fall rate | Velocity error | Joint acceleration | Action jitter |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| PID-Lagrangian SC-PPO 3.8 | selected | 300 / 300 / 400 | 0.100 | 0.641 | 115.908 | 0.221 |
| Plain dual ascent | selected | 400 / 300 / 0 | 0.417 | 0.775 | 108.329 | 0.156 |

Plain dual ascent is not universally collapsed: seed 11 succeeds at the final
checkpoint and seed 17 is partially task-valid. The failure is cross-seed
reliability. Seed 23 collapses to checkpoint 0, giving the plain-dual selected
aggregate a fall rate of 0.417. The lower action jitter of the selected
plain-dual aggregate is not a clean smooth-control improvement because it is
partly driven by the collapsed checkpoint. The bounded interpretation is that
PID-Lagrangian mainly improves cross-seed stability, not that every PID term is
independently necessary.

### 5.5 Dynamic versus kinematic smoothness

Smoothness metrics disagree because they measure different physical aspects of
motion. The paper separates them into two families:

- Dynamic smoothness: `joint_acceleration_l2_mean` and
  `action_jitter_l2_mean`, which measure force-level and command-stream
  oscillation proxies.
- Kinematic smoothness: LDLJ and SPARC, which measure joint trajectory shape and
  velocity-spectrum complexity.

The SC-PPO versus LayerNorm comparison exposes this split.

| Method | Isaac jnt acc | Isaac jitter | MuJoCo jnt acc | MuJoCo jitter | LDLJ | SPARC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SC-PPO 3.8 | 115.908 | 0.221 | 125.541 | 0.231 | -28.345 | -25.543 |
| LayerNorm epochs=3 | 171.977 | 0.519 | 602.578 | 3.328 | -29.692 | -32.280 |

![Dynamic-vs-kinematic smoothness](../../artifacts/analysis/paper_figures/figure_ldlj_sparc.png)

LayerNorm is better on LDLJ and SPARC, meaning its joint trajectories are
kinematically smoother under those metrics. SC-PPO is better on dynamic metrics,
with lower joint acceleration and lower action jitter in Isaac and MuJoCo. The
cross-engine degradation pattern follows the dynamic metrics rather than the
kinematic metrics. This is why the paper's smoothness claim is explicitly about
dynamic smoothness robustness, not all possible notions of smoothness.

### 5.6 Actuator low-pass proxy stress

Because no real-robot validation is available, the paper includes a bounded
simulator-side actuator proxy. The test replays selected checkpoints in MuJoCo
with a first-order low-pass filter between policy action and PD target:

```text
applied_action_t = alpha * policy_action_t + (1 - alpha) * applied_action_{t-1}
```

The proxy uses time constant 0.05 seconds, control timestep 0.01 seconds, and
alpha 0.1667. It is not calibrated actuator modeling; it is a controlled
stress on the idealized action-to-PD-target path.

| Method | Nominal fall | Proxy fall | Proxy velocity error | Proxy jnt acc | Proxy raw jitter | Proxy applied jitter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SC-PPO 3.8 | 0.017 | 0.067 | 0.483 | 104.028 | 0.256 | 0.170 |
| Heuristic baseline | 0.000 | 0.250 | 0.578 | 126.616 | 0.295 | 0.194 |
| LayerNorm epochs=3 | 0.000 | 0.333 | 0.622 | 138.986 | 1.083 | 0.297 |

Under this proxy, SC-PPO has the lowest proxy fall rate, smallest episode-step
loss, lowest velocity error, and lowest raw and applied action jitter among the
three replayed rows. This strengthens the sim-to-real-motivated discussion
within a simulator-side boundary. It does not establish hardware transfer,
system identification, or a general latency-robust controller.

### 5.7 Negative and mixed results

The negative results are part of the evidence rather than incidental failures.
The same-question challenge tested whether other mechanisms could replace the
Jacobian local-sensitivity constraint while preserving dynamic smoothness:

- anisotropic local-sensitivity constraints collapsed,
- action-rate hard constraints collapsed,
- spectral-normalized actors collapsed under tested variants,
- orthogonal actors collapsed,
- LayerNorm became task-valid but degraded 3.5x in MuJoCo joint acceleration,
- action scaling and output scaling were partially task-valid in Isaac but
  collapsed in MuJoCo,
- plain dual ascent partially worked but collapsed on seed 23.

This pattern narrows the claim. The result is not that every smoothness
mechanism fails except SC-PPO. The heuristic action-rate baseline also preserves
cross-engine joint-acceleration smoothness. The result is that directly
constraining policy local sensitivity is one of the only tested mechanisms that
preserves dynamic smoothness across engines without relying on heuristic reward
shaping.

## 6. Discussion

### 6.1 Why local-sensitivity constraints help

The evidence supports the following mechanism chain:

```text
High policy sensitivity
  -> simulator-specific observation/contact differences become action jitter
  -> action jitter drives elevated joint acceleration
  -> dynamic smoothness degrades under cross-engine replay
```

SC-PPO intervenes at the first step by constraining the policy Jacobian. This is
different from LayerNorm or action/output scaling, which change the actor or
action transformation but do not directly enforce a bound on the full
observation-to-action sensitivity on the task distribution.

The multiplier dynamics also clarify the role of the constraint. In the SC-PPO
3.8 checkpoint sweep, the Lagrange multiplier stays near zero after the initial
checkpoint because the policy naturally trains close to the threshold. The
PID-Lagrangian term acts more like a safety mechanism than a constantly active
penalty. This makes the result less about large constraint penalties and more
about keeping the training trajectory inside a narrow sensitivity regime.

### 6.2 Why the heuristic baseline remains strong

The revised heuristic action-rate baseline is not a weak baseline. It preserves
joint-acceleration smoothness almost perfectly from Isaac to MuJoCo
(1.01x degradation) and is better than SC-PPO on several MuJoCo task metrics.
This matters for the narrative. The paper should not claim that Jacobian
constraints dominate heuristic reward shaping in every external validation
setting. Instead, the paper shows that Jacobian constraints can achieve
comparable cross-engine dynamic-smoothness preservation while avoiding direct
action-rate reward tuning, and that several architecture or scaling
replacements fail to do so.

### 6.3 Smoothness is multidimensional

The LayerNorm result prevents an oversimplified smoothness story. A method can
look smoother under kinematic trajectory metrics and still be dynamically worse
under command and joint-acceleration metrics. For robot deployment, dynamic
smoothness is important because it is closer to actuator loading and structural
vibration. For motion quality or animation-like trajectory analysis, kinematic
metrics are also informative. The paper therefore reports both but anchors the
cross-engine claim on dynamic smoothness.

### 6.4 Practical implications

For practitioners, the main lesson is that policy-level sensitivity should be
logged and controlled when smoothness is expected to transfer across simulator
or hardware changes. Action-rate penalties are still useful and may be simpler
to deploy. But if the goal is to understand why a policy's control stream
degrades under another simulator, local-sensitivity diagnostics provide a more
direct mechanism-side signal than aggregate reward or trajectory smoothness
alone.

## 7. Limitations

The limitations are material and should remain explicit.

First, the main experiment uses three seeds. This is common in locomotion
research but insufficient for strong statistical claims. The paper reports
per-seed behavior and mean/std summaries, but extended-seed robustness is
deferred to a possible full-paper upgrade.

Second, the main SC-PPO result depends on checkpoint-sweep selection. The
selected checkpoints are 300/300/400, not final-only. An epochs=3 repair attempt
does not solve this universally: seed 11 improves, seed 17 is unchanged, and
seed 23 degrades; the aggregate joint acceleration worsens from 115.908 to
169.132.

Third, MuJoCo replay is mixed evidence. The heuristic baseline is stronger than
SC-PPO on MuJoCo task stability, velocity tracking, episode length, and joint
acceleration. SC-PPO is only slightly better on action jitter. The paper's
cross-engine claim is therefore about degradation behavior relative to
non-Jacobian replacements, not about full MuJoCo dominance.

Fourth, the evidence uses one robot morphology and one primary rough-terrain
task. A random-stairs selected-checkpoint stress test was attempted, but all
methods collapsed under the first stairs-only protocol. That result is a
transfer failure and protocol-repair signal, not a task-valid method ranking.

Fifth, there is no real-robot validation. The actuator low-pass proxy is a
bounded MuJoCo diagnostic, not a calibrated actuator model. It does not model
motor current limits, backlash, encoder latency, communication jitter, or
hardware safety constraints.

Sixth, no full external constrained-RL library baseline is included. CPO,
OmniSafe, or an exact external LCP-style baseline would require substantial
integration to train under the Humanoid-Gym rough-terrain scaffold, emit the
shared metrics, run checkpoint sweeps, and replay task-valid checkpoints in
MuJoCo. This is deferred to future full-paper work.

Finally, the sensitivity-to-degradation link is correlational. The clean
SC-PPO/LayerNorm comparison is suggestive, and the MuJoCo traces support a
policy-output amplification interpretation, but the current evidence is not a
time-series causal proof.

## 8. Conclusion

SC-PPO shows that constraining policy local sensitivity can produce useful
dynamic-smoothness robustness in humanoid locomotion. On Isaac Gym rough
terrain, the threshold 3.8 PID-Lagrangian variant improves task and dynamic
smoothness metrics relative to a revised heuristic action-rate baseline. In
aligned MuJoCo replay, SC-PPO does not beat the heuristic across task metrics,
so the correct external-validation reading is mixed. The stronger and more
defensible result is that SC-PPO preserves joint-acceleration smoothness with
low cross-engine degradation, while tested non-Jacobian replacement mechanisms
degrade substantially. The degradation pattern tracks policy Jacobian
sensitivity, supporting the hypothesis that local-sensitivity constraints act
as implicit sim-to-sim regularization. The results motivate further work on
extended seeds, external constrained-RL baselines, broader terrain and robot
coverage, and real hardware validation.

## Appendix A. Figure and table provenance

Generated paper figures and tables are reproducible with:

```bash
/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/analysis/generate_paper_figures.py
```

| ID | Content | Source |
| --- | --- | --- |
| F1 | Cross-engine degradation | `artifacts/analysis/paper_figures/figure_cross_engine_degradation.png` |
| F2 | Sensitivity vs degradation | `artifacts/analysis/paper_figures/figure_sensitivity_vs_degradation.png` |
| F3 | Sensitivity evolution | `artifacts/analysis/paper_figures/figure_sensitivity_evolution.png` |
| F4 | Dynamic-vs-kinematic smoothness | `artifacts/analysis/paper_figures/figure_ldlj_sparc.png` |
| F5 | Task-vs-dynamic-smoothness separation | `artifacts/analysis/paper_figures/figure_task_vs_smoothness.png` |
| T2 | Cross-engine degradation table | `artifacts/analysis/paper_figures/table_cross_engine_degradation.md` |
| T3 | Threshold sensitivity table | `artifacts/analysis/paper_figures/table_threshold_sensitivity.md` |
| T4 | Plain dual vs PID table | `artifacts/analysis/paper_figures/table_plain_dual_vs_pid.md` |
| T5 | SC-PPO epochs=3 repair table | `artifacts/analysis/paper_figures/table_scppo_epochs3_repair.md` |
| T6 | LayerNorm trade-off table | `artifacts/analysis/paper_figures/table_layernorm_tradeoff_ldlj_sparc.md` |
| B1 | Draft bibliography | `docs/paper/references.bib` |

## Appendix B. Reproduction entrypoints

The canonical environment setup is:

```bash
cd /home/zhuoxiang/SCO-humanoid
export PYTHON_BIN=/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python
```

Lightweight validation:

```bash
$PYTHON_BIN scripts/baseline/check_env.py
$PYTHON_BIN -m unittest discover -s tests
git diff --check
```

Paper figures:

```bash
$PYTHON_BIN scripts/analysis/generate_paper_figures.py
```

MuJoCo amplification traces:

```bash
$PYTHON_BIN scripts/baseline/run_mujoco_amplification_traces.py
$PYTHON_BIN scripts/analysis/analyze_mujoco_amplification_traces.py
```

Actuator proxy stress:

```bash
$PYTHON_BIN scripts/baseline/run_mujoco_actuator_proxy_stress.py
$PYTHON_BIN scripts/analysis/analyze_mujoco_actuator_proxy_stress.py
```

## Appendix C. Submission checklist

- Verify bibliography metadata and venue formatting in `docs/paper/references.bib`.
- Convert Markdown figures to LaTeX figure environments.
- Decide whether the target format is workshop short paper, workshop full paper,
  or arXiv technical report.
- Keep the title and abstract aligned with the mixed MuJoCo boundary.
- Do not add claims of real-robot validation or broad constrained-RL SOTA.
- If upgrading to a stronger full-paper target, reopen the deferred extended
  seed and external baseline decisions before changing the evidence boundary.
