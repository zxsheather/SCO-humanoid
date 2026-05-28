# OmniSafe PPO-Lag Feasibility Note

**Issue:** #59
**Branch:** `issue-59-omnisafe-feasibility`
**Date:** 2026-05-28

## Decision

OmniSafe PPO-Lag is feasible only as a bounded custom-bridge baseline, not as a
drop-in custom-environment baseline.

Proceed to the adapter smoke path (#60) and the local-sensitivity cost bridge
(#61), but keep the #53 boundary strict: a baseline is faithful only if it uses
the same policy-local-sensitivity/Jacobian cost as SC-PPO. An action-rate,
torque, fall, or other environment-side proxy would be a different baseline and
should not be reported as the #53 external constrained-RL comparator.

## Package And Environment Status

- Selected package: `omnisafe`.
- Latest PyPI version visible from this machine: `0.5.0`
  (`python -m pip index versions omnisafe`).
- Current shell import status: not installed.
- Current shell Python: `3.13.13`.
- OmniSafe upstream metadata advertises Python `3.8` through `3.11` classifiers
  and depends on `torch >= 1.10.0` and `safety-gymnasium`.

Recommendation: do not install OmniSafe into the current Python 3.13 base
environment. Use a separate experiment environment compatible with both
Humanoid-Gym/Isaac Gym and OmniSafe, or let #60 record a precise environment
blocker before any training budget is spent.

## OmniSafe PPO-Lag Contract

OmniSafe's `PPOLag` is a standard PPO-Lagrangian algorithm:

- it inherits from OmniSafe PPO;
- it creates a `Lagrange` helper;
- it updates the Lagrange multiplier from logged mean episode cost
  (`Metrics/EpCost`);
- it combines reward and cost advantages as
  `(adv_r - lambda * adv_c) / (1 + lambda)`.

OmniSafe's custom environment contract expects `step(action)` to return:

`observation, reward, cost, terminated, truncated, info`

The on-policy adapter stores the returned `cost` into the vector on-policy buffer
and logs episode cost. This is a normal CMDP rollout-cost interface.

Sources inspected:

- OmniSafe `PPOLag`: <https://github.com/PKU-Alignment/omnisafe/blob/main/omnisafe/algorithms/on_policy/naive_lagrange/ppo_lag.py>
- OmniSafe `Lagrange`: <https://github.com/PKU-Alignment/omnisafe/blob/main/omnisafe/common/lagrange.py>
- OmniSafe custom environment template: <https://github.com/PKU-Alignment/omnisafe/blob/main/omnisafe/envs/custom_env.py>
- OmniSafe on-policy adapter: <https://github.com/PKU-Alignment/omnisafe/blob/main/omnisafe/adapter/onpolicy_adapter.py>

## SC-PPO Cost Contract

The current SC-PPO cost is not an environment-side scalar from the simulator.
It is computed inside the PPO update from the current actor:

1. take a minibatch of policy observations;
2. run `actor_critic.act_inference(obs)`;
3. compute the Jacobian of each action dimension with respect to observations
   using autograd with `create_graph=True`;
4. aggregate the Frobenius-style sensitivity norm with mean/max/quantile;
5. compare the selected statistic against the threshold;
6. add the multiplier-weighted constraint error to the PPO loss;
7. update the multiplier after the minibatch updates.

Canonical full-paper config uses:

- `threshold = 3.8`;
- `cost_aggregation = quantile`;
- `cost_quantile = 0.90`;
- `subsample_obs = 8`;
- `lambda_init = 0.5`;
- `update_mode = pid`;
- disabled heuristic smoothness rewards.

Local implementation anchors:

- `.external/humanoid-gym/humanoid/algo/ppo/sc_ppo.py`
- `configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_extended_seeds.json`
- `scripts/baseline/evaluate_policy.py`

## Interface Mismatch

The mismatch is structural:

- OmniSafe's default PPO-Lag consumes rollout costs returned by the environment
  and builds cost advantages from those stored costs.
- SC-PPO's training cost depends on the current actor derivative with respect to
  observations during optimization; the environment does not know the actor and
  cannot compute this value from `step(action)` alone.

Therefore, a pure OmniSafe custom environment can only provide a proxy cost, not
the SC-PPO Jacobian cost. That would not answer #53.

## Feasible Bridge

A faithful path is still possible if #61 keeps the custom bridge bounded:

- use OmniSafe PPO-Lag as the external PPO-Lagrangian training skeleton;
- add a small policy-local-sensitivity cost hook near the OmniSafe on-policy
  rollout/update path, where the adapter or algorithm has access to both the
  current actor and rollout observations;
- feed that computed cost into PPO-Lag's cost buffer or multiplier update without
  replacing it with an environment proxy;
- log multiplier, cost mean/update statistic, threshold, violation rate, and
  sample count so the result remains comparable to SC-PPO sidecar metrics.

This bridge is acceptable only if it remains a narrow adapter/algorithm extension.
If it requires replacing most of OmniSafe PPO-Lag with project-specific SC-PPO
logic, #61 should close the external baseline as infeasible rather than producing
a misleading comparison.

## Rejected Shortcuts

Do not use these as the #53 baseline:

- action-rate cost;
- torque cost;
- joint-acceleration cost;
- fall/collision cost;
- reward-side action smoothing;
- a Humanoid-Gym env wrapper that returns arbitrary random or diagnostic costs.

Those are different scientific questions. They may be useful future baselines,
but they do not test whether an external constrained-RL implementation can carry
the same Jacobian/local-sensitivity constraint.

## Next Steps

1. #60 should build only the Humanoid-Gym-to-OmniSafe adapter smoke path and
   verify runtime/device/shape compatibility.
2. #61 should implement or reject the bounded local-sensitivity cost bridge.
3. #62 should handle evaluation compatibility, because the current shared
   evaluator assumes rsl-rl/Humanoid-Gym checkpoints and cannot directly score an
   OmniSafe policy without an adapter.
4. #63 should run the bounded `23 / 29 / 31` diagnostic only after #60-#62 pass.

## Acceptance Criteria Status

- Package/version/import status: satisfied.
- PPO-Lag entry point and custom-environment cost contract: satisfied.
- SC-PPO cost contract compared against OmniSafe rollout cost: satisfied.
- Recommendation recorded: proceed only with bounded custom bridge; otherwise
  close as infeasible.
- Proxy costs rejected: satisfied.
