# OmniSafe Cost Bridge Result (#61)

**Branch:** `full-paper/extended-seeds`
**Issue:** #61
**Parent:** #53

## Status

The repaired #61 smoke passes as a bounded component bridge:

- it instantiates the real Humanoid-Gym rough-terrain task;
- it resets and steps one seeded environment;
- it computes the SC-PPO policy-local-sensitivity/Jacobian cost on the real
  policy observation tensor;
- it feeds that scalar cost into OmniSafe's `Lagrange` multiplier component;
- it records reward, cost, multiplier, and violation-rate diagnostics.

This is not a full OmniSafe PPO-Lag training baseline. It proves the cost and
multiplier component can be bridged, while preserving the #59 boundary that a
faithful full PPO-Lag baseline needs an algorithm/update hook rather than a pure
environment adapter.

## Repaired Smoke Result

Command:

```bash
PYTHON_BIN=/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python \
  scripts/baseline/run_omnisafe_cost_bridge_smoke.sh \
  --run-name=omnisafe_ppolag_cost_bridge_smoke_seed23_repaired \
  --num-envs=1 \
  --seed=23 \
  --rl-device=cuda:0 \
  --sim-device=cuda:0 \
  --write-failure-artifact
```

Artifact:

`artifacts/methods/omnisafe_ppolag_cost_bridge_smoke/omnisafe_ppolag_cost_bridge_smoke_seed23_repaired/omnisafe_cost_bridge_smoke.json`

Observed values:

| Field | Value |
| --- | ---: |
| observation shape | `[1, 705]` |
| privileged observation shape | `[1, 219]` |
| action shape | `[1, 12]` |
| reward mean | `0.0352` |
| Jacobian cost update | `0.3559` |
| threshold | `3.8` |
| violation rate | `0.0000` |
| multiplier | `0.5000 -> 0.4656` |
| constraint error | `-3.4441` |

The multiplier decreases because the local-sensitivity cost is below the
threshold. This matches OmniSafe's Lagrange update direction.

The direct Python process still exits with the known Isaac Gym teardown
segmentation fault after writing the complete artifact. The shell wrapper treats
that as successful only when `status=complete` is present in the artifact.

## Boundary

OmniSafe `PPOLag` consumes environment-side costs and builds cost advantages from
rollout buffers. The SC-PPO cost is actor-internal: it is computed from the
current policy Jacobian on policy observations during the PPO update. A pure
custom-environment adapter cannot provide this cost faithfully.

The repaired bridge therefore exercises only the reusable OmniSafe Lagrange
component. It does not claim that a full OmniSafe PPO-Lag baseline has been run
or that #53 is answered by a new external training result.

## Consequence

#61 should be considered complete as a feasibility/bridge smoke:

- canonical Jacobian cost is computable on real Humanoid-Gym observations;
- reward, cost, multiplier, and violation diagnostics are finite;
- no action-rate, torque, fall, or other proxy cost is used.

For #53, the next decision is explicit: either implement a narrow OmniSafe
algorithm/update hook that exposes actor observations during PPO-Lag training, or
close the external PPO-Lag baseline as too invasive for this paper. Do not cite
the #61 component smoke as an external baseline result.
