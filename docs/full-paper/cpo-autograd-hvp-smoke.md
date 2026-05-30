# CPO autograd/HVP smoke result

Issue: [#81](https://github.com/zxsheather/SCO-humanoid/issues/81)
Date: 2026-05-30
Status: **Pass - proceed to #82 one-update prototype**

## Scope

This is a local autograd smoke for the CPO implementation path. It constructs
the current Humanoid-Gym `ActorCritic` shape and a synthetic current-shape
observation minibatch, then tests whether the SC-PPO actor-internal
Jacobian/local-sensitivity cost can coexist with:

- `grad_theta J_C(theta)`, the constraint gradient needed by CPO-style updates;
- one KL Fisher-vector product, the core operation behind CPO/TRPO-style natural
  gradient machinery.

No Isaac rollout, seed training, environment-side proxy cost, or official CPO
parity claim is involved.

## Command

```bash
/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python \
  scripts/baseline/run_cpo_autograd_hvp_smoke.py \
  --config=configs/methods/cpo_autograd_hvp_smoke.json
```

Artifact:

`artifacts/methods/cpo_autograd_hvp_smoke/cpo_autograd_hvp_smoke_seed23/cpo_autograd_hvp_smoke.json`

## Model And Tensor Contract

| Field | Value |
| --- | ---: |
| device | `cuda:0` |
| observation source | synthetic current-shape minibatch |
| observation shape | `[16, 705]` |
| action dimensions | `12` |
| actor hidden dims | `[512, 256, 128]` |
| policy parameter tensors | `9` |
| policy parameters | `527256` |

## Results

| Check | Value |
| --- | ---: |
| `subsample_obs=1` cost | `0.2822` |
| `subsample_obs=1` cost-gradient norm | `0.5343` |
| `subsample_obs=1` CUDA peak allocated | `25.71 MB` |
| `subsample_obs=1` wall time | `0.0166 s` |
| `subsample_obs=8` cost | `0.2835` |
| `subsample_obs=8` cost-gradient norm | `0.4601` |
| `subsample_obs=8` CUDA peak allocated | `26.51 MB` |
| `subsample_obs=8` wall time | `0.0190 s` |
| KL HVP shape consistent | `true` |
| KL HVP finite | `true` |
| KL HVP norm | `0.1957` |
| KL HVP CUDA peak allocated | `35.04 MB` |
| KL HVP wall time | `0.0155 s` |

CUDA first-use warm-up is excluded from the reported timings.

## Interpretation

The smoke verifies the narrow #81 question: the actor-internal Jacobian cost is
differentiable through policy parameters, and one KL Fisher-vector product can be
computed on the same Humanoid-Gym policy shape. This removes the strongest
"cannot even compute the needed tensors" objection to a local CPO-style path.

The cost-gradient coverage is expected rather than full: `std` and the final
actor bias do not affect the deterministic policy Jacobian. The KL HVP covers
all policy tensors, including `std`.

## Boundary

This result does not show that CPO training is stable, efficient, or paper-ready.
It also does not establish official CPO parity, because no rollout-level cost
estimator, constrained update solve, or line search has been implemented yet.

## Recommendation

Proceed to #82: implement a local one-update CPO-style prototype. Keep the next
step bounded to one update and preserve the same no-proxy-cost boundary.

