# CPO feasibility for actor-internal policy-sensitivity constraints

Issue: [#80](https://github.com/zxsheather/SCO-humanoid/issues/80)
Date: 2026-05-30
Status: **Recommend DEFER** (#84 accepted: diagnostic/future work only)

---

## 1. Current SC-PPO cost contract

The Jacobian/local-sensitivity cost is computed in `SCPPO._local_sensitivity_metrics()` (`sc_ppo.py:66-97`) and used inside the PPO update loop (`sc_ppo.py:212-228`). The computation chain is:

1. **Forward**: Subsample up to 8 observations, call `act_inference(obs)`, and obtain the deterministic action mean.
2. **Jacobian**: For each of the 12 action dimensions, compute the action mean derivative with respect to the observation batch via `autograd.grad(..., create_graph=True)`.
3. **Aggregate**: Compute the per-sample Frobenius norm of the policy Jacobian, then aggregate it by the configured mean/max/quantile statistic.
4. **Use in training**: Add the Lagrangian penalty `lambda * (cost - threshold)` to the actor loss and update the PID multiplier from the same actor-internal cost statistic.

Key properties:

- The cost is actor-internal: it depends on the current policy derivative with respect to observations, not on environment-side `(state, action, reward, cost)` tuples.
- `create_graph=True` is essential because the Jacobian norm must remain differentiable with respect to policy parameters.
- The current implementation uses a small stochastic observation subsample for the Jacobian cost, not the full rollout batch.

## 2. What a faithful CPO-style comparator would optimize

A faithful comparator must keep the same task reward and the same policy-local-sensitivity cost family. It should not replace the actor-internal Jacobian cost with action rate, torque, fall, joint acceleration, or any other environment-side proxy.

A CPO-style update would need, at minimum:

1. A reward-surrogate gradient `g`.
2. A constraint gradient `c = grad_theta J_C(theta)` where `J_C` is the SC-PPO-style Jacobian/local-sensitivity statistic.
3. Fisher-vector products for the policy KL trust-region system.
4. A constrained update rule, usually conjugate gradient plus a small dual solve and line search.

This would be a **CPO-style constrained optimizer for an actor-internal differentiable constraint**, not an official textbook CPO/CMDP parity result unless the cost placement, rollout estimator, trust-region objective, and line-search checks are documented end to end.

## 3. Implementation paths

### Path A: Local CPO-style update in the current PPO stack

**What it means**: Fork the current PPO/SC-PPO update, add Fisher-vector products, conjugate gradient, constrained dual solve, and line search, while reusing the existing Jacobian cost function.

**Pros**:

- Highest control over the exact actor-internal cost.
- Reuses the known Humanoid-Gym rollout, policy, evaluator, and logging stack.
- Avoids forcing the Jacobian cost into an environment-side cost API.

**Risks**:

- Computing the Jacobian cost already requires 12 `autograd.grad` calls with `create_graph=True`; differentiating that scalar cost with respect to actor parameters requires higher-order autograd through the policy derivative graph.
- CPO-style Fisher-vector products add repeated second-order KL computations on top of the Jacobian-cost graph.
- A full-rollout CPO estimate would likely be much more expensive than SC-PPO's current mini-batch/subsampled estimate.
- A mini-batch or subsampled CPO-style update may be a useful diagnostic, but it weakens any claim of official CPO parity or textbook trust-region guarantees.

**Assessment**: Technically possible as a local diagnostic, but not low-risk enough to launch as a full baseline without a smoke test.

### Path B: Adapt an external CPO implementation such as OmniSafe

**What it means**: Use an external CPO implementation for the optimizer machinery and inject the actor-internal Jacobian cost through an algorithm-level hook.

**Pros**:

- More reviewer-recognizable than a fully local implementation.
- Could reuse existing CPO machinery if the hook surface is narrow.

**Risks**:

- Standard safe-RL frameworks usually consume per-timestep environment costs from a rollout buffer. The SC-PPO cost is computed during optimization from the actor and observation batch.
- A pure environment adapter would only provide a proxy cost and would not answer this issue.
- The existing OmniSafe PPO-Lag experience showed that a faithful actor-internal cost bridge needs an algorithm/update hook, not only an environment wrapper.
- CPO's conjugate-gradient and line-search machinery makes the hook surface larger than the previous PPO-Lag diagnostic hook.

**Assessment**: Possible only as a bounded algorithm-level extension. It should not be described as "OmniSafe CPO" or official external-CPO parity unless the modified update path is documented clearly.

### Path C: First-order constrained-policy alternatives

**What it means**: Use a first-order constrained RL method, such as FOCOPS-style or PPO-Lag-style updates, instead of full CPO.

**Pros**:

- Much cheaper than a CPO-style natural-gradient/trust-region update.
- Closer to the current PPO implementation and previous OmniSafe PPO-Lag diagnostic.

**Risks**:

- Methods that rely on per-timestep cost returns, cost advantages, or a cost critic do not naturally accept the SC-PPO actor-internal Jacobian cost.
- Replacing the Jacobian cost with an environment proxy would answer a different question.
- A first-order local implementation would be another mechanism variant, not a CPO comparison.

**Assessment**: Useful as mechanism coverage, but not a substitute for a faithful CPO row.

## 4. Minimum smoke test before any CPO row

Before committing to CPO training, run a deliberately small diagnostic:

1. Load the current actor and one rollout/minibatch observation tensor.
2. Compute the SC-PPO Jacobian cost with `subsample_obs=1` and `subsample_obs=8`.
3. Compute `grad_theta J_C(theta)` and report nonzero parameter-gradient coverage, gradient norm, peak CUDA memory, and wall time.
4. Compute one KL Fisher-vector product and report shape consistency, finite values, peak CUDA memory, and wall time.
5. Run a toy conjugate-gradient solve for a few iterations and verify finite residuals.
6. Do not train a seed unless the above steps pass comfortably.

Expected failure modes:

- CUDA OOM from retaining the Jacobian derivative graph.
- `None` or zero gradients for actor parameters because the actor adapter detaches the wrong tensor.
- Non-finite gradients from the Jacobian norm or KL Hessian-vector product.
- Line search rejecting every step because the stochastic constraint estimate is too noisy.

## 5. Compute/runtime risk estimate

The exact overhead has not been measured for CPO in this repository. The risk is nevertheless high because a faithful CPO-style update would combine two expensive pieces:

| Component | Risk | Notes |
|---|---|---|
| Jacobian cost forward | High | 12 action dimensions, each requiring an `autograd.grad` call with `create_graph=True`. |
| Cost gradient | High | Backpropagates through the policy-derivative graph used by the Jacobian cost. |
| Fisher-vector products | Medium-High | Each CG iteration requires a KL Hessian-vector product. |
| Dual solve | Low | Small constrained update subproblem. |
| Line search | Medium | Repeated policy/cost/KL checks can multiply update time. |

A responsible estimate should be reported only after the smoke test records peak memory and wall time. Until then, the paper should avoid numeric claims such as "50-100x" overhead or "200-400GB" VRAM.

## 5.1 #81 smoke update

The #81 local autograd/HVP smoke passed on a constructed current-shape
Humanoid-Gym actor and synthetic observation minibatch. It verified finite
Jacobian-cost gradients for `subsample_obs=1` and `subsample_obs=8`, and a finite
shape-consistent KL Fisher-vector product. This removes the narrow objection
that the required tensors cannot be computed at all.

The result does not change the paper-facing boundary: no constrained update
solve, line search, rollout-level estimator, training run, or official CPO row
has been demonstrated yet.

## 5.2 #82 one-update update

The #82 local one-update CPO-style prototype passed on the same constructed
current-shape actor and synthetic observation batch. It computed a reward
surrogate gradient, Jacobian-cost constraint gradient, two KL Fisher-vector
conjugate-gradient solves, a small dual decision, and a backtracking line search.
The first line-search candidate was accepted under the configured KL and
constraint checks.

This narrows the remaining risk to repeated-update behavior: rollout coupling,
advantage quality, line-search reliability across updates, checkpoint quality,
and training stability.

## 5.3 #83 bounded diagnostic update

The #83 bounded one-seed diagnostic completed with seed 23, 16 training
environments, 3 iterations, checkpoints `0,1,2,3`, and 2 evaluation episodes per
checkpoint. The local CPO-style updates were finite, line search accepted all
three updates, and checkpoints were produced. However, the checkpoint sweep
reported `selection_status=all_checkpoints_collapsed`: every evaluated
checkpoint had `fall_rate=1.000`.

This means the implementation path is no longer blocked at the tensor or runner
integration level, but the current diagnostic does not provide a task-valid CPO
baseline.

## 5.4 #84 evidence decision

The #84 human decision accepts the conservative boundary: CPO remains a local
CPO-style diagnostic and future-work item, not a current paper baseline. The
evidence supports technical feasibility at the tensor, one-update, and tiny
training-loop levels, but it does not support a task-valid CPO row.

The current manuscript should not claim official CPO parity, should not add CPO
to the primary comparison table, and should not describe the collapsed bounded
diagnostic as evidence that CPO or external constrained RL broadly fails.

## 6. Recommendation: DEFER, do not reject as impossible

CPO remains reviewer-relevant, but it should not block the current mechanism-comparison manuscript. The most accurate current conclusion is:

- A pure environment-side CPO adapter is not faithful because the target cost is actor-internal.
- A local or external CPO-style implementation with algorithm-level hooks is technically possible in principle.
- The #81 tensor smoke and #82 one-update smoke pass, and #83 shows the update can run in the training loop.
- The #84 decision freezes this as diagnostic/future-work evidence for the
  current manuscript rather than opening a multi-seed CPO expansion.

For the current paper, the stronger position is to keep CPO as a limitation/future-work item while relying on:

- the within-family plain dual ascent ablation;
- the LCP-style soft Jacobian/Lipschitz regularization baseline;
- the bounded OmniSafe PPO-Lag diagnostic as a framework-interface result;
- the broader mechanism comparison suite.

## 7. Follow-up

No long or multi-seed CPO training should be launched from this issue. If CPO becomes essential for a future revision, create a separate bounded-training issue with this scope:

- justify a larger budget than #83;
- preserve exact CUDA memory and wall time;
- keep a single-seed gate before any multi-seed expansion;
- no environment-side proxy costs;
- no official CPO claim unless the modified algorithm path is documented.
