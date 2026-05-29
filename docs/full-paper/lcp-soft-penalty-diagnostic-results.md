# LCP-Style Soft Penalty Diagnostic Result (#68)

## Status

Status: `promote-to-human-decision`.

The canonical three-seed LCP-style diagnostic completed on seeds `23 / 29 / 31`
with checkpoint sweeps over `{0, 100, 200, 300, 400}`. All three seeds selected
the final checkpoint `400`, all final checkpoints are task-valid, and no seed is
collapsed. This clears the issue #68 三种子并行起步门槛 and should move to #69
for the human decision on whether to spend five-seed and MuJoCo budget.

This is still Isaac-side diagnostic evidence only. It should not be described
as a full-paper SOTA win until #69 approves formal expansion and #70 completes.

## Run

Command:

```bash
PYTHON_BIN=/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python \
  scripts/baseline/run_lcp_soft_penalty_diagnostic.sh
```

Log:

`artifacts/analysis/rough_terrain_lcp_soft_jacobian_diagnostic/logs/lcp68-20260529T053325Z.log`

Aggregate artifact:

`artifacts/analysis/rough_terrain_lcp_soft_jacobian_diagnostic/comparison_summary.json`

The training subprocess for each seed exited with the known Isaac Gym teardown
segmentation fault after writing the expected manifest/checkpoints. The runner
recovered each case from the completed artifact. Checkpoint evaluation
subprocesses also recovered from the same teardown pattern after writing metrics.

## Selected / Final Aggregate

Selected and final aggregates are identical because all three seeds selected
checkpoint `400`.

| Method / slice | Checkpoints | Fall | Vel. err | Jnt acc | Jitter | Ep. return | Eval sens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LCP-style soft penalty, seeds 23/29/31 | `400 / 400 / 400` | `0.000` | `0.477` | `123.903` | `0.226` | `119.697` | `1.861` |

For context, the same three-seed slice recorded in the OmniSafe diagnostic note
used:

| Method / slice | Checkpoints | Fall | Vel. err | Jnt acc | Jitter | Ep. return |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SC-PPO 3.8 PID, seeds 23/29/31 | `400 / 400 / 400` | `0.217` | `0.595` | `162.612` | `0.315` | `99.007` |
| Revised heuristic, seeds 23/29/31 | `350 / 400 / 400` | `0.117` | `0.667` | `111.463` | `0.256` | `111.772` |

The contextual comparison should be read cautiously because #68 is an Isaac-side
diagnostic, not a completed formal baseline. Still, the LCP-style row is strong:
it improves task validity, velocity tracking, action jitter, and return relative
to the same-slice heuristic, while its joint acceleration remains worse than the
heuristic but better than the same-slice SC-PPO row.

## Per-Seed Selected Rows

| Seed | Status | Selected ckpt | Fall | Vel. err | Jnt acc | Jitter | Ep. return | Eval sens | Violation |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 23 | `selected` | `400` | `0.000` | `0.448` | `105.900` | `0.200` | `123.963` | `1.852` | `0.006` |
| 29 | `selected` | `400` | `0.000` | `0.467` | `106.353` | `0.205` | `126.458` | `1.901` | `0.000` |
| 31 | `selected` | `400` | `0.000` | `0.517` | `159.456` | `0.274` | `108.671` | `1.832` | `0.008` |

Seed `31` is the roughest of the three on joint acceleration and action jitter,
but it is still non-collapsed and task-valid at the final checkpoint.

## Checkpoint Progression

| Seed | Ckpt | Fall | Vel. err | Jnt acc | Jitter | Ep. return | Eval sens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 23 | 0 | `1.000` | `1.288` | `78.9` | `0.016` | `4.4` | `0.355` |
| 23 | 100 | `1.000` | `1.417` | `90.1` | `0.094` | `4.1` | `1.474` |
| 23 | 200 | `1.000` | `1.128` | `155.1` | `0.228` | `24.2` | `2.422` |
| 23 | 300 | `0.100` | `0.686` | `120.6` | `0.207` | `89.9` | `2.088` |
| 23 | 400 | `0.000` | `0.448` | `105.9` | `0.200` | `124.0` | `1.852` |
| 29 | 0 | `1.000` | `1.464` | `69.8` | `0.016` | `4.3` | `0.364` |
| 29 | 100 | `1.000` | `1.403` | `100.5` | `0.122` | `4.5` | `1.776` |
| 29 | 200 | `0.750` | `0.873` | `122.3` | `0.199` | `63.7` | `2.633` |
| 29 | 300 | `0.050` | `0.543` | `117.0` | `0.210` | `109.6` | `2.197` |
| 29 | 400 | `0.000` | `0.467` | `106.4` | `0.205` | `126.5` | `1.901` |
| 31 | 0 | `1.000` | `1.471` | `71.1` | `0.016` | `4.5` | `0.339` |
| 31 | 100 | `1.000` | `1.329` | `103.2` | `0.106` | `4.8` | `1.641` |
| 31 | 200 | `0.250` | `0.803` | `140.3` | `0.214` | `85.3` | `2.350` |
| 31 | 300 | `0.000` | `0.615` | `151.3` | `0.245` | `106.1` | `1.829` |
| 31 | 400 | `0.000` | `0.517` | `159.5` | `0.274` | `108.7` | `1.832` |

The pattern is late acquisition rather than early cherry-picking: checkpoint
`400` is selected on all three seeds.

## Training-Side Penalty Read

The active training penalty is the LCP-style gradient penalty, not the SC-PPO
policy-local-sensitivity Lagrange cost. The final trace entries are:

| Seed | Iter | LCP penalty | Penalty loss | Grad norm | Grad max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 23 | 399 | `6.445` | `0.0129` | `2.372` | `5.123` |
| 29 | 399 | `6.221` | `0.0124` | `2.302` | `5.177` |
| 31 | 399 | `5.126` | `0.0103` | `2.082` | `5.081` |

Evaluation-side local sensitivity stays well below the shared `3.8` threshold
on all selected checkpoints, with aggregate mean `1.861` and violation rate
`0.0049`.

## Decision

Classification: `promote-to-human-decision`.

Reasoning:

- Passes 逐种子硬门槛: all three selected checkpoints are non-collapsed.
- Stronger than the minimum first gate: selected checkpoint equals final
  checkpoint `400` for all three seeds.
- Improves task validity and tracking relative to the current same-slice
  baselines, while keeping action jitter competitive.
- Remaining risk is smoothness trade-off against the revised heuristic on joint
  acceleration, especially seed `31`; this requires formal five-seed and MuJoCo
  validation before any full-paper SOTA claim.

Next step: #69 should decide whether this LCP-style diagnostic earns formal
candidate promotion. Do not start #70 until that human decision is recorded.
