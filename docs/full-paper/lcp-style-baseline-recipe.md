# LCP-Style SOTA Baseline Recipe (#66)

**Branch:** `issue-66-lcp-baseline-audit`
**Issue:** #66
**Status:** recipe frozen for the next implementation slice

## Decision

The closest SOTA-adjacent comparison for the full-paper direction is not
OmniSafe PPO-Lag. It is an LCP-style soft Lipschitz/Jacobian regularization
baseline.

The official LCP release should be used as the method anchor, but its code and
checkpoints should not be reported as a direct baseline row for this repo. The
public MimicKit LCP entrypoint trains a different stack/task configuration
(`deepmimic_g1_env`, `4096` Isaac-Gym envs, and G1 walk checkpoints). Those
artifacts do not share this repo's Humanoid-Gym rough-terrain morphology,
reward/task configuration, checkpoint-selection rule, metric schema, or MuJoCo
replay bridge. A local LCP-style reimplementation is therefore required for a
same-scale comparison.

## External Anchor

Use these as the baseline definition sources:

- Paper/project: "Learning Smooth Humanoid Locomotion through
  Lipschitz-Constrained Policies" (IROS 2025), which presents LCP as a
  differentiable gradient-penalty replacement for smoothness rewards and
  low-pass filters.
- arXiv: <https://arxiv.org/abs/2410.11825>
- Project page: <https://xbpeng.github.io/projects/LCP/index.html>
- Public implementation: <https://github.com/xbpeng/MimicKit>
- MimicKit LCP README:
  <https://github.com/xbpeng/MimicKit/blob/main/docs/README_LCP.md>

Important source facts for the local recipe:

- LCP constrains the policy through a gradient penalty on
  `||grad_s log pi(a | s)||^2` over rollout state-action samples.
- The simplified optimization objective is PPO-style return maximization minus
  `lambda_gp * E[||grad_s log pi(a | s)||^2]`.
- The public README exposes the smoothness weight as `lcp_weight` and states it
  is the crucial parameter to tune for task/smoothness balance.
- The paper reports `lambda_gp = 0.002` as the effective coefficient in its main
  LCP evaluation, while noting that too-small values can leave jitter and
  too-large values can reduce task return.

## Claim Boundary

This baseline tests:

- soft Lipschitz/Jacobian regularization as a SOTA-adjacent smooth humanoid
  locomotion mechanism;
- whether a fixed gradient-penalty coefficient can replace the current hard
  `SC-PPO 3.8 + PID-Lagrangian` 策略局部敏感度 constraint under 同尺比较.

This baseline does not test:

- OmniSafe PPO-Lag or any external CMDP framework;
- action-rate, torque, fall, or joint-acceleration proxy costs;
- a hybrid `SC-PPO + LCP` method;
- real-robot transfer.

The paper may say this is an LCP-style same-task reproduction. It should not say
the repo directly evaluates official LCP checkpoints unless the official task,
robot, and metric schema are actually matched.

## Minimal Local Recipe

Use the next implementation issue (#67) to add one bounded diagnostic path:

- **Method id:** `lcp_soft_jacobian_penalty`.
- **Training base:** Humanoid-Gym PPO on `humanoid_ppo`.
- **Smoothness mechanism:** add a soft gradient penalty to the actor loss.
- **Primary penalty:** `lcp_weight * mean(||grad_obs log pi(a_batch | obs_batch)||^2)`.
- **Initial coefficient:** `lcp_weight = 0.002`, because this is the published
  LCP anchor. Do not run a coefficient sweep in the first diagnostic.
- **Observation scope:** apply the penalty to the same policy observation tensor
  used by the actor minibatch. If an implementation must choose between current
  observation and full observation history, use the full actor input first,
  matching LCP's whole-input recommendation.
- **Action samples:** use the PPO minibatch actions already stored in rollout
  storage, matching LCP's state-action-sample formulation.
- **Autograd:** compute the penalty with `create_graph=True` so it contributes
  to actor gradients, not merely to logging.
- **Heuristic rewards:** set `action_smoothness`, `dof_acc`, `base_acc`, and
  `dof_vel` smoothness rewards to `0.0`.
- **No Lagrange multiplier:** do not use PID or plain dual updates in this
  baseline. `lcp_weight` is a fixed soft-penalty coefficient.
- **Side read:** keep evaluation-side local-sensitivity logging with threshold
  `3.8` so the result remains comparable to SC-PPO, but do not treat threshold
  violation as an active training constraint.

The implementation may reuse the current SC-PPO Jacobian utility shape only for
logging or smoke validation. The active training penalty should follow the LCP
`grad log pi` form unless #67 records a concrete compatibility blocker.

## First Diagnostic Protocol

The first real diagnostic is #68 and should use:

- **Seeds:** `23 / 29 / 31`.
- **Training budget:** rough terrain, `512 envs x 400 iterations`.
- **Checkpoint grid:** `{0, 100, 200, 300, 400}`.
- **Evaluation:** `32` envs, `20` episodes per checkpoint.
- **Metrics:** velocity tracking error, fall rate, joint acceleration, action
  jitter, episode return, and local-sensitivity sidecar metrics.
- **Selection:** reuse the existing task-floor-first, then smoothest checkpoint
  rule.

The diagnostic is a 替代机制可行性诊断, not a 正式候选线 by default.

Promotion requires all three seeds to clear the existing 三种子并行起步门槛 and
逐种子硬门槛:

- each seed must have a task-valid selected checkpoint;
- no collapsed seed may be hidden by aggregate metrics;
- selected checkpoints may be used for this first gate, but a later formal
  candidate must still face the stricter Isaac-side internal challenge before
  MuJoCo budget is spent.

If the three-seed diagnostic passes, #69 should decide whether to promote the
line to a formal five-seed + MuJoCo comparison. If any seed collapses or misses
the task floor, record the result as a negative or allow only one explicitly
bounded repair; do not start coefficient sweeps by default.

## OmniSafe Boundary

Do not reopen OmniSafe as the next SOTA baseline. The OmniSafe diagnostic showed
that framework-level PPO-Lag can be bridged only invasively for this
actor-internal Jacobian cost, and the bounded end-to-end run collapsed on all
three diagnostic seeds. That result belongs in the paper as a framework-interface
negative diagnostic only if needed; it is not the SOTA smooth humanoid baseline.

The next SOTA-facing work is therefore #67, not another OmniSafe adapter.
