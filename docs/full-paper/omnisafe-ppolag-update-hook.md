# OmniSafe PPO-Lag Jacobian Update Hook (#65)

## Status

The bounded update-hook smoke passes. It shows that the SC-PPO
policy-local-sensitivity/Jacobian cost can be computed on a real Humanoid-Gym
policy-observation tensor, added to an OmniSafe-style actor loss, differentiated
through the actor parameters, and used to update OmniSafe's `Lagrange`
multiplier.

This does not yet produce a checkpoint. It is the missing one-update bridge
needed before #62/#63 can be resumed.

## Smoke Command

```bash
PYTHON_BIN=/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python \
  scripts/baseline/run_omnisafe_update_hook_smoke.sh \
  --run-name=omnisafe_ppolag_update_hook_smoke_seed23 \
  --num-envs=1 \
  --seed=23 \
  --rl-device=cuda:0 \
  --sim-device=cuda:0 \
  --write-failure-artifact
```

Artifact:

`artifacts/methods/omnisafe_ppolag_update_hook_smoke/omnisafe_ppolag_update_hook_smoke_seed23/omnisafe_update_hook_smoke.json`

## Observed Values

| Field | Value |
| --- | ---: |
| observation shape | `[1, 705]` |
| privileged observation shape | `[1, 219]` |
| action shape | `[1, 12]` |
| reward mean | `0.0350` |
| Jacobian cost update | `0.6887` |
| threshold | `3.8` |
| violation rate | `0.0000` |
| multiplier | `0.5000 -> 0.4689` |
| base actor loss | `-1.0000` |
| Jacobian penalty loss | `-1.5557` |
| total actor loss | `-2.5557` |
| actor grad norm | `3.4840` |

The multiplier decreases because the Jacobian cost is below threshold. The
non-zero finite actor gradient norm confirms that the Jacobian penalty is part
of the differentiable actor update, not merely logged after the fact.

The direct Python process exits with the known Isaac Gym teardown segmentation
fault after writing `status=complete`; the wrapper recovers that exit only when
the artifact is complete.

## Boundary

No site-packages files were modified. The hook reuses OmniSafe
`GaussianLearningActor` and `Lagrange`. The required PPO-Lag integration point is
`_update_actor`, where OmniSafe already has the actor, minibatch observations,
actions, log-probabilities, and advantages.

Checkpoint-producing `learn()` integration remains downstream. The smoke
supports continuing to #62/#63 only if we accept this `_update_actor` override as
a bounded external-baseline adaptation.
