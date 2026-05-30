# CPO one-update smoke result

Issue: [#82](https://github.com/zxsheather/SCO-humanoid/issues/82)
Date: 2026-05-30
Status: **Pass - proceed to #83 bounded one-seed diagnostic**

## Scope

This is a local CPO-style one-update prototype. It reuses the same
actor-internal SC-PPO Jacobian/local-sensitivity cost and runs exactly one
constrained update attempt on a constructed current-shape Humanoid-Gym
`ActorCritic` and synthetic observation minibatch.

The smoke covers:

- reward-surrogate gradient;
- Jacobian-cost constraint gradient;
- KL Fisher-vector product;
- conjugate-gradient solves for reward and cost directions;
- a small dual decision;
- backtracking line-search checks for KL, constraint value, surrogate value, and
  finite tensors.

No rollout, seed training, environment-side proxy cost, checkpoint, or official
CPO parity claim is involved.

## Command

```bash
/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python \
  scripts/baseline/run_cpo_one_update_smoke.py \
  --config=configs/methods/cpo_one_update_smoke.json
```

Artifact:

`artifacts/methods/cpo_one_update_smoke/cpo_one_update_smoke_seed23/cpo_one_update_smoke.json`

## Model And Update Contract

| Field | Value |
| --- | ---: |
| device | `cuda:0` |
| observation source | synthetic current-shape minibatch |
| observation shape | `[16, 705]` |
| action dimensions | `12` |
| actor hidden dims | `[512, 256, 128]` |
| policy parameters | `527256` |
| max KL | `0.01` |
| Jacobian subsample | `8` |
| threshold | `3.8` |

## Results

| Check | Value |
| --- | ---: |
| old surrogate | `0.0000` |
| old constraint value | `-3.5179` |
| reward-gradient norm | `5.5005` |
| cost-gradient norm | `0.4676` |
| reward CG iterations | `10` |
| reward CG final residual | `0.000916` |
| cost CG iterations | `10` |
| cost CG final residual | `0.000582` |
| dual case | `reward_only_feasible` |
| dual lambda | `26.2562` |
| dual nu | `0.0000` |
| trust-region quadratic | `0.0100` |
| step norm | `0.1034` |
| line-search accepted | `true` |
| accepted backtrack | `0` |
| accepted KL | `0.00944` |
| accepted surrogate | `0.5567` |
| accepted constraint value | `-3.5174` |
| CUDA peak allocated | `54.31 MB` |
| wall time | `0.2257 s` |

## Interpretation

The one-update path is coherent at the tensor/optimizer level. The prototype can
compute reward and cost gradients on the same local update batch, solve two
damped KL natural-gradient systems, choose a feasible CPO-style update direction,
and accept a line-search candidate under the configured KL and constraint checks.

The selected dual case is `reward_only_feasible` because the current random
policy's Jacobian cost is far below the SC-PPO threshold. That is sufficient for
this issue: it verifies the full one-update plumbing without forcing an
artificially violated constraint.

## Boundary

This is still not a CPO training baseline. It does not test rollout collection,
advantage estimation from real rewards, repeated updates, checkpoint quality,
or cross-seed stability. It should be described as a local CPO-style diagnostic,
not official CPO parity.

## Recommendation

Proceed to #83 with a bounded one-seed diagnostic only if the next step remains
strictly scoped: short budget, no proxy costs, shared metrics, and failure
artifacts preserved.

