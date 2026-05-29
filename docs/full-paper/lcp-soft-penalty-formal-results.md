# LCP-Style Soft Penalty Formal Result (#70)

## Status

Status: `complete`.

Issue #69 explicitly approved promotion of the LCP-style soft Jacobian penalty
line to a formal candidate. Issue #70 therefore ran the approved five-seed
Isaac expansion on seeds `11 / 17 / 23 / 29 / 31`, then ran the conditional
MuJoCo selected-checkpoint replay because all five Isaac selected checkpoints
were task-valid.

This result should be treated as a strong full-paper baseline result, not as a
new SOTA claim. The implementation is a local Humanoid-Gym reimplementation of
the LCP-style policy-gradient penalty recipe, with fixed coefficient
`lcp_weight = 0.002`.

## Commands

Isaac five-seed expansion:

```bash
PYTHON_BIN=/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python \
  ISSUE_ID="#70" \
  RUN_LABEL="LCP-style five-seed formal expansion" \
  SEEDS="11 17 23 29 31" \
  SUMMARY_DIR=artifacts/analysis/rough_terrain_lcp_soft_jacobian_formal \
  scripts/baseline/run_lcp_soft_penalty_diagnostic.sh
```

MuJoCo selected-checkpoint replay:

```bash
scripts/baseline/run_mujoco_lcp_soft_jacobian_parallel.sh
```

The first MuJoCo attempt used five parallel exports and exceeded GPU memory
after seed `11` completed. The replay script now defaults to sequential
execution, skips completed metrics, and recovers non-zero teardown exits when
the expected metrics file is present.

## Artifacts

- Isaac comparison summary:
  `artifacts/analysis/rough_terrain_lcp_soft_jacobian_formal/comparison_summary.json`
- Isaac log:
  `artifacts/analysis/rough_terrain_lcp_soft_jacobian_formal/logs/lcp70-20260529T072130Z.log`
- MuJoCo logs:
  `artifacts/analysis/rough_terrain_lcp_soft_jacobian_formal/mujoco_parallel_logs/seed{11,17,23,29,31}.log`
- Per-seed Isaac/MuJoCo metrics:
  `artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed*/`

Isaac Gym checkpoint evaluation and MuJoCo replay both show the known teardown
segmentation-fault pattern after writing expected artifacts. The runs below are
counted only where the expected metrics JSON exists.

## Isaac Five-Seed Result

Selection rule: task floor first, then smoothest checkpoint.

| Method | Seeds | Selected ckpts | Fall | Vel. err | Jnt acc | Jitter | Ep. return |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| LCP-style soft penalty | `11/17/23/29/31` | `300/400/400/400/400` | `0.000` | `0.490` | `117.331` | `0.212` | `118.420` |
| SC-PPO 3.8 PID | `11/17/23/29/31` | `300/300/400/400/400` | `0.170` | `0.606` | `142.955` | `0.277` | `99.349` |
| Revised heuristic | `11/17/23/29/31` | `350/300/350/400/400` | `0.150` | `0.705` | `115.317` | `0.260` | `105.326` |

Final-checkpoint aggregate for LCP is also task-valid:

| Slice | Fall | Vel. err | Jnt acc | Jitter | Ep. return | Eval sens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LCP selected | `0.000` | `0.490` | `117.331` | `0.212` | `118.420` | `1.890` |
| LCP final ckpt 400 | `0.000` | `0.493` | `118.336` | `0.220` | `120.142` | `1.894` |

Per-seed selected rows:

| Seed | Selected ckpt | Fall | Vel. err | Jnt acc | Jitter | Ep. return | Eval sens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | `300` | `0.000` | `0.482` | `104.232` | `0.177` | `113.362` | `2.010` |
| 17 | `400` | `0.000` | `0.536` | `110.716` | `0.205` | `119.647` | `1.856` |
| 23 | `400` | `0.000` | `0.448` | `105.900` | `0.200` | `123.963` | `1.852` |
| 29 | `400` | `0.000` | `0.467` | `106.353` | `0.205` | `126.458` | `1.901` |
| 31 | `400` | `0.000` | `0.517` | `159.456` | `0.274` | `108.671` | `1.832` |

Read: LCP passes the five-seed Isaac hard gate with no collapsed selected
checkpoint. It improves task validity, velocity tracking, jitter, return, and
joint acceleration relative to SC-PPO 3.8. It is slightly worse than the revised
heuristic on Isaac joint acceleration, but better on task validity, tracking,
jitter, and return.

## MuJoCo Selected Replay

Protocol: `isaac_mainline`, `20 episodes x 20s`, `joint_reset_noise = 0.1`,
selected Isaac checkpoint per seed.

| Method / slice | Seeds | Fall | Vel. err | Jnt acc | Jitter | Ep. return |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LCP-style soft penalty | `11/17/23/29/31` | `0.000` | `0.406` | `117.425` | `0.195` | `-599.108` |
| LCP-style soft penalty | `11/17/23` | `0.000` | `0.429` | `112.498` | `0.178` | `-670.984` |
| SC-PPO 3.8 PID anchor | `11/17/23` | `0.017` | `0.491` | `125.541` | `0.231` | `-647.674` |
| Revised heuristic anchor | `11/17/23` | `0.000` | `0.419` | `120.734` | `0.245` | `-465.370` |

The existing SC-PPO and revised-heuristic MuJoCo anchors cover only
`11/17/23`, so the five-seed LCP row should not be read as a matched five-seed
MuJoCo win against those methods.

Per-seed LCP MuJoCo rows:

| Seed | Ckpt | Fall | Vel. err | Jnt acc | Jitter | Ep. return |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | `300` | `0.000` | `0.428` | `110.894` | `0.165` | `-754.436` |
| 17 | `400` | `0.000` | `0.503` | `118.100` | `0.184` | `-929.203` |
| 23 | `400` | `0.000` | `0.355` | `108.500` | `0.184` | `-329.313` |
| 29 | `400` | `0.000` | `0.332` | `110.690` | `0.190` | `-333.780` |
| 31 | `400` | `0.000` | `0.413` | `138.941` | `0.250` | `-648.810` |

Cross-engine joint-acceleration factors:

| Method / slice | Isaac jnt acc | MuJoCo jnt acc | Factor |
| --- | ---: | ---: | ---: |
| LCP, five seeds | `117.331` | `117.425` | `1.001x` |
| LCP, `11/17/23` | `106.949` | `112.498` | `1.052x` |
| SC-PPO 3.8, `11/17/23` | `115.908` | `125.541` | `1.083x` |
| Revised heuristic, `11/17/23` | `119.864` | `120.734` | `1.007x` |

Read: MuJoCo strengthens the LCP line as a smoothness baseline. On the shared
`11/17/23` anchor slice, LCP has lower joint acceleration and jitter than both
SC-PPO and the revised heuristic. It does not dominate every task-side metric:
the revised heuristic has slightly better MuJoCo velocity tracking and a less
negative MuJoCo return on this slice.

## Conservative Claim Boundary

Defensible full-paper wording:

> A fixed-coefficient LCP-style policy-gradient penalty is a strong
> SOTA-adjacent baseline under the same Humanoid-Gym protocol. It passes the
> five-seed Isaac task-validity gate and preserves low dynamic smoothness
> degradation in aligned MuJoCo replay.

Not defensible from #70 alone:

- Do not claim this proves a SOTA result.
- Do not claim SC-PPO is the strongest method on the current full-paper branch.
- Do not claim real-world robustness; MuJoCo remains sim-to-sim evidence.
- Do not claim the local implementation is equivalent to any official LCP code
  or checkpoint release.

## Decision

Issue #70 is complete.

The result changes the full-paper direction: LCP-style soft regularization is
now the strongest current smoothness/stability baseline line, while SC-PPO
remains useful as a Jacobian-constrained PID-Lagrangian mechanism but no longer
supports a simple "SC-PPO beats alternatives" narrative. The next paper task
should be #71: integrate LCP and OmniSafe evidence into the full-paper narrative
and decide whether the manuscript presents SC-PPO as the main method, LCP as a
stronger baseline, or a mechanism-level comparison centered on policy
sensitivity.
