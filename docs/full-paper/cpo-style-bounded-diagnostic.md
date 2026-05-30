# CPO-style bounded diagnostic result

Issue: [#83](https://github.com/zxsheather/SCO-humanoid/issues/83)
Date: 2026-05-30
Status: **Completed - do not promote as a baseline**

## Scope

This is the bounded one-seed diagnostic following the #81 autograd/HVP smoke and
#82 one-update prototype. It tests whether the local CPO-style update can run
inside the Humanoid-Gym training loop, produce checkpoints, and be evaluated
with the shared rough-terrain checkpoint sweep.

This is not a five-seed experiment and not official CPO parity.

## Budget

| Field | Value |
| --- | ---: |
| seed | `23` |
| train envs | `16` |
| steps per env | `8` |
| max iterations | `3` |
| total train timesteps | `384` |
| checkpoints | `0, 1, 2, 3` |
| eval envs | `4` |
| eval episodes per checkpoint | `2` |
| actor-internal cost | SC-PPO Jacobian/local sensitivity |
| environment proxy costs | none |

## Command

```bash
FORCE=1 PYTHON_BIN=/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python \
  scripts/baseline/run_cpo_style_bounded_diagnostic.sh all
```

Key artifacts:

- `artifacts/methods/cpo_style_bounded_diagnostic/cpo_style_bounded_diagnostic_seed23/manifest.json`
- `artifacts/methods/cpo_style_bounded_diagnostic/cpo_style_bounded_diagnostic_seed23/cpo_update_trace.json`
- `artifacts/methods/cpo_style_bounded_diagnostic/cpo_style_bounded_diagnostic_seed23/checkpoint_sweep_summary.json`

Run directory:

`.external/humanoid-gym/logs/ecolab_cpo_style_diagnostic/May30_02-54-27_cpo_style_bounded_diagnostic_seed23`

## Training Stability

| Iter | Line Search | KL | Cost Update | Update Time | CUDA Peak |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `1.0` | `0.004470` | `0.3622` | `0.2468 s` | `58.34 MB` |
| `1` | `1.0` | `0.004755` | `0.3465` | `0.1310 s` | `58.40 MB` |
| `2` | `1.0` | `0.005932` | `0.3471` | `0.1391 s` | `58.40 MB` |

All three local CPO-style updates were finite and accepted by line search. The
Jacobian cost stayed far below the threshold `3.8`, so the constraint was not
active in this short run. The training command wrote checkpoints and manifest;
the known Isaac Gym teardown segmentation fault occurred after artifact writing
and was recovered by the wrapper.

## Checkpoint Sweep

Selection status: `all_checkpoints_collapsed`

Selected checkpoint for analysis only: `1`

| Ckpt | Fall | Vel. Err | Joint Acc | Jitter | Return | Eval Sens. |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `1.000` | `1.033` | `81.511` | `0.0136` | `4.284` | `0.354` |
| `1` | `1.000` | `1.035` | `70.838` | `0.0142` | `4.343` | `0.361` |
| `2` | `1.000` | `1.377` | `59.713` | `0.0157` | `4.703` | `0.372` |
| `3` | `1.000` | `1.377` | `59.713` | `0.0157` | `4.703` | `0.372` |

Every evaluated checkpoint fell in all evaluation episodes. The result is
therefore a failed bounded training diagnostic from a task-validity perspective,
even though the optimizer plumbing itself ran and produced checkpoints.

## Interpretation

The CPO-style mechanism is no longer blocked at the tensor, one-update, or basic
runner-integration level. The remaining problem is training usefulness: this
small diagnostic did not produce a task-valid policy. Because the run is only
three iterations, the collapse should not be interpreted as evidence that CPO,
CPO-style updates, or external constrained RL fail broadly.

The correct paper-facing interpretation is narrower:

- local CPO-style integration is technically possible;
- repeated updates can run without immediate OOM or non-finite tensors at this
  tiny budget;
- the first bounded diagnostic is not a usable baseline because all checkpoints
  collapse under rough-terrain evaluation.

## Recommendation

Do not promote CPO to the paper's baseline table from this result. Move to #84
for a human evidence decision: stop here as a documented diagnostic, redesign a
larger but still bounded CPO training run, or keep CPO in future work.
