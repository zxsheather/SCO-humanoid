# LCP Weight Sensitivity Diagnostic Results (#73)

## Status

Status: `complete`.

This diagnostic tests whether the formal LCP-style baseline depends too heavily
on the single coefficient `lcp_weight = 0.002`. It runs two neighboring
coefficients on the diagnostic seed slice `23 / 29 / 31` and compares against
the existing #68/#70 coefficient.

Tested coefficients:

- `0.001`
- `0.002` (existing formal candidate)
- `0.004`

Protocol:

- 512 envs x 400 iterations
- checkpoint sweep `{0, 100, 200, 300, 400}`
- 32 envs x 20 episodes per checkpoint
- task-floor-first, then smoothest checkpoint selection

No MuJoCo replay was run for this coefficient diagnostic.

## Command

```bash
scripts/baseline/run_lcp_weight_sensitivity_diagnostic.sh
```

Log:

`artifacts/analysis/rough_terrain_lcp_weight_sensitivity_diagnostic/logs/lcp73-20260529T103356Z.log`

Comparison summaries:

- `artifacts/analysis/rough_terrain_lcp_weight_sensitivity_diagnostic/w0001/comparison_summary.json`
- `artifacts/analysis/rough_terrain_lcp_soft_jacobian_diagnostic/comparison_summary.json`
- `artifacts/analysis/rough_terrain_lcp_weight_sensitivity_diagnostic/w0004/comparison_summary.json`

## Aggregate Result

All three coefficients select final checkpoint `400 / 400 / 400` and pass the
task-validity floor. The middle coefficient is best on every selected aggregate
except raw violation rate, where the largest coefficient is lowest.

| LCP weight | Ckpts | Fall | Vel. err | Jnt acc | Jitter | Return | Eval sens | Viol. |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.001` | `400/400/400` | `0.017` | `0.569` | `147.380` | `0.267` | `107.706` | `2.241` | `0.0049` |
| `0.002` | `400/400/400` | `0.000` | `0.477` | `123.903` | `0.226` | `119.697` | `1.861` | `0.0049` |
| `0.004` | `400/400/400` | `0.000` | `0.539` | `128.041` | `0.237` | `117.982` | `1.712` | `0.0000` |

## Per-Seed Selected Rows

| Weight | Seed | Ckpt | Fall | Vel. err | Jnt acc | Jitter | Return | Eval sens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.001` | 23 | `400` | `0.000` | `0.483` | `123.617` | `0.236` | `117.851` | `2.281` |
| `0.001` | 29 | `400` | `0.000` | `0.585` | `164.916` | `0.312` | `103.640` | `2.201` |
| `0.001` | 31 | `400` | `0.050` | `0.640` | `153.608` | `0.253` | `101.628` | `2.241` |
| `0.002` | 23 | `400` | `0.000` | `0.448` | `105.900` | `0.200` | `123.963` | `1.852` |
| `0.002` | 29 | `400` | `0.000` | `0.467` | `106.353` | `0.205` | `126.458` | `1.901` |
| `0.002` | 31 | `400` | `0.000` | `0.517` | `159.456` | `0.274` | `108.671` | `1.832` |
| `0.004` | 23 | `400` | `0.000` | `0.577` | `103.834` | `0.194` | `120.026` | `1.712` |
| `0.004` | 29 | `400` | `0.000` | `0.551` | `171.128` | `0.310` | `108.999` | `1.650` |
| `0.004` | 31 | `400` | `0.000` | `0.490` | `109.161` | `0.207` | `124.920` | `1.775` |

## Interpretation

The diagnostic supports `lcp_weight = 0.002` as a robust local choice within
this narrow grid.

- `0.001` appears under-regularized: it has the highest evaluation sensitivity,
  worse joint acceleration and jitter, worse tracking, lower return, and a
  small nonzero fall rate.
- `0.004` lowers evaluation sensitivity and removes threshold violations, but
  it does not improve the aggregate policy. Its seed `29` checkpoint is
  dynamically rough (`jnt_acc=171.128`, `jitter=0.310`) and its aggregate
  tracking/return are worse than `0.002`.
- `0.002` is the best trade-off on this seed slice: zero fall, best velocity
  tracking, best joint acceleration, best action jitter, and best return.

This is not a broad hyperparameter sweep and should not be written as proof of a
globally optimal coefficient. The defensible claim is narrower: the formal LCP
result is not an isolated single-point accident within the immediate
`0.001 / 0.002 / 0.004` neighborhood.
