# OmniSafe PPO-Lag Diagnostic Result (#63)

## Status

Status: `collapsed`.

This is a bounded three-seed diagnostic for the external constrained-RL baseline path. It uses the #65 Jacobian update hook and #62 OmniSafe policy evaluator; it does not replace the #51 five-seed SC-PPO record.

Artifacts:

- Summary: `artifacts/analysis/rough_terrain_omnisafe_ppolag_diagnostic/comparison_summary.json`
- Config: `configs/methods/omnisafe_ppolag_diagnostic.json`

Because every evaluated checkpoint collapsed (`fall_rate = 1.0`), the low joint-acceleration and jitter numbers are not evidence of smooth locomotion; they are artifacts of policies that fall rather than move successfully.

## Selected-Checkpoint Aggregate

| Method / anchor | Fall | Vel. err | Jnt acc | Jitter | Ep. return |
| --- | ---: | ---: | ---: | ---: | ---: |
| OmniSafe PPO-Lag diagnostic, seeds 23/29/31 | 1.000 | 1.468 | 79.309 | 0.008 | 4.386 |
| SC-PPO PID repair (threshold=3.8, lambda_init=0.5, quantile=0.90, lower-bound clamp), same 3-seed slice | 0.217 | 0.595 | 162.612 | 0.315 | 99.007 |
| PPO + heuristic smoothing (action_rate=-0.0050, formal protocol revision long budget), same 3-seed slice | 0.117 | 0.667 | 111.463 | 0.256 | 111.772 |

## Final-Checkpoint Aggregate

| Method | Fall | Vel. err | Jnt acc | Jitter | Ep. return |
| --- | ---: | ---: | ---: | ---: | ---: |
| OmniSafe PPO-Lag diagnostic, final checkpoint 400 | 1.000 | 1.109 | 321.328 | 0.741 | 2.714 |

## Per-Seed Selection

| Seed | Status | Selected ckpt | Fall | Vel. err | Jnt acc | Jitter | Ep. return |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 23 | `all_checkpoints_collapsed` | 0 | 1.000 | 1.394 | 80.220 | 0.008 | 4.423 |
| 29 | `all_checkpoints_collapsed` | 0 | 1.000 | 1.394 | 79.605 | 0.007 | 4.525 |
| 31 | `all_checkpoints_collapsed` | 0 | 1.000 | 1.615 | 78.100 | 0.008 | 4.211 |

## Boundary

- The result is diagnostic-only and should not be described as a full OmniSafe PPO-Lag replacement for SC-PPO.
- No MuJoCo replay or five-seed expansion is implied unless all selected checkpoints are task-valid and the user approves expanded budget.
- The bridge uses the same policy-local-sensitivity/Jacobian cost family; it does not substitute action-rate, torque, fall, or other proxy costs.
