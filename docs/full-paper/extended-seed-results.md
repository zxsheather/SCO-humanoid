# Extended-Seed Robustness Results

**Branch:** `full-paper/extended-seeds`
**Issue:** #51
**Run:** `extended-seeds-formal-20260527T144401Z`

## Status

The extended-seed matrix completed and wrote:

- `artifacts/analysis/rough_terrain_extended_seeds/comparison_summary.json`
- `artifacts/analysis/rough_terrain_extended_seeds/logs/extended-seeds-formal-20260527T144401Z.log`

The runner exited with `exit_status=0`. Several Isaac subprocesses exited with
`-11` after writing their artifacts; `run_formal_comparison.py` recovered those
cases because the expected manifests or checkpoint-sweep summaries were present.

## Result

The full-paper promotion rule is **not met**. SC-PPO keeps a velocity-tracking
advantage over the revised heuristic anchor, but the extended five-seed read
does not preserve the original dynamic-smoothness advantage.

Selected-checkpoint aggregate over seeds `11 / 17 / 23 / 29 / 31`:

| Method | Checkpoints | Vel. err | Jnt acc | Jitter | Return | Fall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SC-PPO 3.8 | `300 / 300 / 400 / 400 / 400` | `0.606` | `142.955` | `0.277` | `99.349` | `0.170` |
| Revised heuristic | `350 / 300 / 350 / 400 / 400` | `0.705` | `115.317` | `0.260` | `105.326` | `0.150` |

Interpretation:

- SC-PPO remains better on velocity tracking: `0.606` vs `0.705`.
- SC-PPO loses joint acceleration: `142.955` vs `115.317`.
- SC-PPO loses action jitter: `0.277` vs `0.260`.
- SC-PPO is slightly worse on fall rate: `0.170` vs `0.150`.
- Therefore the five-seed audit weakens the current full-paper claim; it should
  not be presented as robustness confirmation.

## Per-Seed Read

| Method | Seed | Selected ckpt | Fall | Vel. err | Jnt acc | Jitter | Return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SC-PPO 3.8 | 11 | 300 | `0.100` | `0.684` | `120.8` | `0.222` | `96.6` |
| SC-PPO 3.8 | 17 | 300 | `0.100` | `0.563` | `106.1` | `0.218` | `103.1` |
| SC-PPO 3.8 | 23 | 400 | `0.100` | `0.677` | `120.8` | `0.221` | `101.1` |
| SC-PPO 3.8 | 29 | 400 | `0.550` | `0.652` | `247.3` | `0.475` | `76.2` |
| SC-PPO 3.8 | 31 | 400 | `0.000` | `0.456` | `119.8` | `0.250` | `119.7` |
| Revised heuristic | 11 | 350 | `0.150` | `0.632` | `119.5` | `0.271` | `106.0` |
| Revised heuristic | 17 | 300 | `0.250` | `0.893` | `122.7` | `0.261` | `85.3` |
| Revised heuristic | 23 | 350 | `0.050` | `0.740` | `117.4` | `0.281` | `111.5` |
| Revised heuristic | 29 | 400 | `0.150` | `0.627` | `98.1` | `0.227` | `109.9` |
| Revised heuristic | 31 | 400 | `0.150` | `0.635` | `118.9` | `0.259` | `113.9` |

## Seed-29 Diagnosis

The failure is concentrated in `SC-PPO seed29`. This is not a selector error:
only checkpoint `400` clears the task floor, so the selected checkpoint is the
only eligible checkpoint.

SC-PPO seed29 checkpoint sweep:

| Ckpt | Fall | Vel. err | Jnt acc | Jitter | Return |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `1.000` | `1.449` | `74.6` | `0.012` | `4.3` |
| 100 | `1.000` | `1.208` | `96.8` | `0.140` | `6.0` |
| 200 | `1.000` | `1.048` | `196.0` | `0.313` | `19.0` |
| 300 | `1.000` | `0.813` | `247.0` | `0.445` | `39.4` |
| 400 | `0.550` | `0.652` | `247.3` | `0.475` | `76.2` |

SC-PPO seed31, by contrast, reaches a usable final checkpoint without the same
dynamic-smoothness blow-up:

| Ckpt | Fall | Vel. err | Jnt acc | Jitter | Return |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | `0.350` | `0.582` | `119.5` | `0.243` | `98.2` |
| 400 | `0.000` | `0.456` | `119.8` | `0.250` | `119.7` |

The constraint-side read does not by itself explain the seed29 failure. SC-PPO
seed29 and seed31 have similar late training traces: final train quantile cost
is near the threshold (`3.826` and `3.840` against threshold `3.8`), final
violation rate is `0.140625` for both, and the final Lagrange multiplier is
near zero (`0.0021` and `0.0030`). The difference is that seed29 only acquires a
partially task-valid gait at the end, and that gait is dynamically rough.

## Full-Paper Consequence

The current full-paper path should not proceed directly to an external
constrained-RL baseline. Adding a CPO/OmniSafe-style baseline now would compare
against an SC-PPO row whose five-seed robustness is already weakened.

The next useful work is a narrow diagnostic/repair branch for seed29-style
late-acquisition instability. A bounded first probe should keep the full-paper
audit seed set fixed and test whether the SC-PPO family can avoid the
`task-valid but dynamically rough` final seed29 solution without changing the
historical workshop claim.
