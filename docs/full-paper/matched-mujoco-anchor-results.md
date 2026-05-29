# Matched Five-Seed MuJoCo Anchor Results (#72)

## Status

Status: `complete`.

Issue #72 fills the missing MuJoCo selected-checkpoint replays for the existing
five-seed SC-PPO and revised-heuristic anchors. This removes the main mismatch
left after #70: LCP had five MuJoCo seeds, while SC-PPO and heuristic MuJoCo
anchors covered only `11/17/23`.

No training was rerun and no method config was changed. The four added replays
are:

- SC-PPO 3.8 seeds `29 / 31`, selected checkpoint `400 / 400`
- Revised heuristic seeds `29 / 31`, selected checkpoint `400 / 400`

All use the existing `isaac_mainline`, `20 episodes x 20s`,
`joint_reset_noise = 0.1` protocol. Each added replay wrote the expected metrics
JSON and then hit the known MuJoCo/Isaac teardown segmentation fault; the runner
recovered each case from the completed metrics file.

## Command

```bash
scripts/baseline/run_mujoco_extended_seed_anchors.sh
```

Logs:

`artifacts/analysis/rough_terrain_extended_seeds/mujoco_added_seed_logs/{scppo38,heuristic}_seed{29,31}.log`

## Matched Five-Seed MuJoCo Aggregate

| Method | Seeds | Fall | Vel. err | Jnt acc | Jitter | Return |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LCP-style soft penalty | `11/17/23/29/31` | `0.000` | `0.406` | `117.425` | `0.195` | `-599.108` |
| SC-PPO 3.8 PID | `11/17/23/29/31` | `0.010` | `0.471` | `159.718` | `0.322` | `-627.238` |
| Revised heuristic | `11/17/23/29/31` | `0.000` | `0.406` | `111.615` | `0.226` | `-456.370` |

Per-seed added rows:

| Method | Seed | Ckpt | Fall | Vel. err | Jnt acc | Jitter | Return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SC-PPO 3.8 | 29 | `400` | `0.000` | `0.413` | `298.578` | `0.651` | `-524.427` |
| SC-PPO 3.8 | 31 | `400` | `0.000` | `0.467` | `123.389` | `0.264` | `-668.738` |
| Revised heuristic | 29 | `400` | `0.000` | `0.428` | `83.432` | `0.169` | `-634.603` |
| Revised heuristic | 31 | `400` | `0.000` | `0.347` | `112.440` | `0.225` | `-251.139` |

## Cross-Engine Factors

| Method | Isaac jnt acc | MuJoCo jnt acc | Jnt factor | Isaac jitter | MuJoCo jitter | Jitter factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LCP-style soft penalty | `117.331` | `117.425` | `1.001x` | `0.212` | `0.195` | `0.917x` |
| SC-PPO 3.8 PID | `142.955` | `159.718` | `1.117x` | `0.277` | `0.322` | `1.160x` |
| Revised heuristic | `115.317` | `111.615` | `0.968x` | `0.260` | `0.226` | `0.869x` |

## Interpretation Update

The matched five-seed MuJoCo read strengthens the conservative narrative:

- LCP remains much stronger than SC-PPO on MuJoCo dynamic smoothness and task
  stability.
- LCP has the lowest MuJoCo action jitter among the three methods.
- The revised heuristic is not a weak baseline. It matches LCP on fall rate and
  velocity, and is better on MuJoCo joint acceleration and return.
- Therefore, the full-paper claim should not say that LCP dominates all
  baselines. The safer claim is that LCP is the closest SOTA-adjacent
  policy-sensitivity baseline, while the heuristic remains a highly competitive
  reward-shaping anchor.

This result does not weaken the #71 mechanism framing. It sharpens it: soft
policy-sensitivity regularization is more robust than the current SC-PPO hard
constraint, but simple action-rate reward shaping remains competitive enough
that the paper should explain trade-offs rather than claim one universal winner.
