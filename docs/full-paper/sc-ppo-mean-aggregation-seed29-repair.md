# SC-PPO Mean-Aggregation Seed29 Repair Diagnostic

**Branch:** `full-paper/extended-seeds`
**Issue:** #57
**Decision issue:** #56
**Run:** `scppo-mean-repair-20260528T020201Z`

## Status

The bounded diagnostic completed with `exit_status=0` and wrote:

- `artifacts/analysis/sc_ppo_mean_aggregation_seed29_repair/comparison_summary.json`
- `artifacts/analysis/sc_ppo_mean_aggregation_seed29_repair/logs/scppo-mean-repair-20260528T020201Z.log`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_mean_pid_lower_bound_clamp_seed29_repair_seed{23,29,31}/checkpoint_sweep_summary.json`

Several Isaac subprocesses exited with `-11` after writing metrics, matching the
known recovery pattern in the extended-seed run. The runner recovered because
the expected checkpoint metrics and summaries were present.

## Result

The mean-aggregation repair diagnostic **fails**. It should not be promoted to a
full `11 / 17 / 23 / 29 / 31` rerun, and it does not supersede the #51
five-seed audit.

Final-checkpoint gates from #56:

| Seed | Final ckpt | Fall | Vel. err | Jnt acc | Jitter | Return | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 23 | 400 | `0.650` | `0.945` | `158.9` | `0.285` | `64.6` | **Fail** |
| 29 | 400 | `0.950` | `0.714` | `286.8` | `0.586` | `43.4` | **Fail** |
| 31 | 400 | `1.000` | `0.856` | `223.8` | `0.404` | `34.3` | **Fail** |

Selected-checkpoint read:

| Seed | Selection status | Selected ckpt | Fall | Vel. err | Jnt acc | Jitter | Return |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 23 | `selected` | 300 | `0.150` | `0.680` | `113.0` | `0.212` | `105.3` |
| 29 | `selected` | 400 | `0.950` | `0.714` | `286.8` | `0.586` | `43.4` |
| 31 | `all_checkpoints_collapsed` | 0 | `1.000` | `1.483` | `67.7` | `0.012` | `4.4` |

Aggregate over selected checkpoints:

| Metric | Value |
| --- | ---: |
| Fall | `0.700` |
| Vel. err | `0.959` |
| Jnt acc | `155.8` |
| Jitter | `0.270` |
| Return | `51.1` |

Aggregate over final checkpoints:

| Metric | Value |
| --- | ---: |
| Fall | `0.867` |
| Vel. err | `0.838` |
| Jnt acc | `223.2` |
| Jitter | `0.425` |
| Return | `47.5` |

## Interpretation

Changing `cost_aggregation` from `quantile(0.90)` to `mean` does not repair the
seed29 failure surface. Seed29 becomes worse than the #51 SC-PPO row at the
final checkpoint: fall rate rises from `0.550` to `0.950`, joint acceleration
from `247.3` to `286.8`, and action jitter from `0.475` to `0.586`.

The canonical guard seed also fails. Seed23 has a good selected checkpoint at
`300`, but the final checkpoint drifts to `fall_rate = 0.650`, so it does not
solve the final-checkpoint reliability problem. Seed31 fully collapses across
all evaluated checkpoints.

The current reading is therefore negative:

`mean aggregation weakens the tail-sensitive constraint too much for this repair
purpose; it neither fixes seed29 nor preserves canonical final-checkpoint
reliability.`

## Consequence

The aggregation-side minimal diagnostic is closed as a negative result. Do not
expand this into an immediate `mean / max / quantile` sweep, and do not run a
full five-seed replacement matrix for this configuration.

The #51 five-seed audit remains the current full-paper SC-PPO robustness record.
