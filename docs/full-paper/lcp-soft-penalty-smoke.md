# LCP-Style Soft Penalty Smoke (#67)

## Status

Status: `passed`.

This is an implementation smoke for the LCP-style SOTA-adjacent baseline path.
It verifies that the local Humanoid-Gym stack can train a PPO policy with a fixed
LCP-style gradient penalty, save a loadable checkpoint, reload it through the
shared evaluator, and emit the same rough-terrain metric schema used by the
existing method families.

This is not a task-valid result and should not be cited as evidence that LCP
works or fails on the full rough-terrain task.

## Implemented Path

- Algorithm class: `LCPPPO`.
- Active penalty:
  `lcp_weight * mean(||grad_obs log pi(a_batch | obs_batch)||^2)`.
- Initial coefficient: `lcp_weight = 0.002`.
- Sampling: random subsample of up to `64` PPO minibatch observations/actions
  per minibatch update.
- Smoothness reward overrides: `action_smoothness = 0.0`, `dof_acc = 0.0`,
  `base_acc = 0.0`, and `dof_vel = 0.0`.
- No Lagrange multiplier, no PID update, and no OmniSafe dependency.
- Evaluation-side local-sensitivity readout keeps threshold `3.8` for 同尺比较
  only; it is not an active training constraint.

## Smoke Command

```bash
PYTHON_BIN=/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python \
  scripts/baseline/run_lcp_soft_penalty_smoke.sh
```

The wrapper runs:

- train: `1 env x 1 iteration`, seed `23`;
- evaluate: `1 env x 1 completed episode`, evaluation seed `123145`.

Both train and evaluation hit the known Isaac Gym teardown segmentation fault
after writing their completion artifacts. The wrapper recovered both exits
because the expected artifacts were present.

## Artifacts

- Manifest:
  `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_smoke_seed23/manifest.json`
- Training sidecar:
  `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_smoke_seed23/constraint_metrics.json`
- Penalty trace:
  `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_smoke_seed23/lcp_penalty_trace.json`
- Evaluation metrics:
  `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_smoke_seed23/metrics.json`
- Loadable checkpoint:
  `.external/humanoid-gym/logs/ecolab_lcp_soft_penalty_diagnostic/May29_05-20-48_lcp_soft_jacobian_penalty_smoke_seed23/model_1.pt`

## Observed Smoke Values

Training sidecar:

| Field | Value |
| --- | ---: |
| `lcp_weight` | `0.002` |
| `lcp_gradient_penalty_mean` | `0.1042` |
| `lcp_penalty_loss_mean` | `0.000208` |
| `lcp_grad_norm_mean` | `0.3132` |
| `lcp_subsample_obs` | `64` |
| `constraint_sample_count` | `120` |

Evaluation metrics:

| Field | Value |
| --- | ---: |
| episodes evaluated | `1` |
| fall rate | `1.000` |
| velocity tracking error | `1.708` |
| joint acceleration | `47.812` |
| action jitter | `0.015` |
| episode return | `2.931` |
| eval local sensitivity mean | `0.343` |
| eval local-sensitivity violation rate | `0.000` |

The collapsed evaluation is expected for a one-iteration smoke and has no
scientific interpretation.

## Consequence

#67 unlocks #68. The next slice can run the canonical diagnostic seeds
`23 / 29 / 31` at `512 envs x 400 iterations` using
`configs/methods/lcp_soft_jacobian_penalty_diagnostic.json`, then evaluate the
checkpoint grid `{0, 100, 200, 300, 400}` with the shared rough-terrain metrics.
