# LCP Weight Sensitivity Diagnostic Results (#96)

Status: `complete`.

This diagnostic tests whether the formal LCP-style baseline depends too heavily
on the single coefficient `lcp_weight = 0.002`. It compares the local
neighborhood `0.001 / 0.002 / 0.004` on the full five-seed set
`11 / 17 / 23 / 29 / 31` using the same checkpoint-selection protocol as the
primary paper tables.

## Protocol

- 512 envs x 400 training iterations.
- Checkpoint sweep `{0, 100, 200, 300, 400}`.
- 32 envs x 20 Isaac episodes per checkpoint.
- Task-floor-first, then smoothest checkpoint selection.
- The `0.002` row uses the primary formal LCP run rather than the older
  three-seed diagnostic summary.

## Reproduction

```bash
PYTHON_BIN=/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python \
ISSUE_ID='#96' \
SEEDS='11 17 23 29 31' \
RL_DEVICE=cuda:0 \
SIM_DEVICE=cuda:0 \
scripts/baseline/run_lcp_weight_sensitivity_diagnostic.sh

/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/analysis/generate_paper_figures.py
```

Comparison summaries:

- `artifacts/analysis/rough_terrain_lcp_weight_sensitivity_diagnostic/w0001/comparison_summary.json`
- `artifacts/analysis/rough_terrain_lcp_soft_jacobian_formal/comparison_summary.json`
- `artifacts/analysis/rough_terrain_lcp_weight_sensitivity_diagnostic/w0004/comparison_summary.json`

Generated paper table:

- `artifacts/analysis/paper_figures/table_lcp_weight_sensitivity.md`

## Aggregate Result

| LCP weight | Ckpts | Fall | Vel. err | Jnt acc | Jitter | Return | Eval sens | Viol. |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.001` | `400/400/400/400/400` | `0.040` | `0.579` | `138.860` | `0.262` | `108.189` | `2.284` | `0.0063` |
| `0.002` | `300/400/400/400/400` | `0.000` | `0.490` | `117.331` | `0.212` | `118.420` | `1.890` | `0.0029` |
| `0.004` | `400/400/400/400/400` | `0.000` | `0.538` | `125.291` | `0.228` | `118.708` | `1.678` | `0.0000` |

## Interpretation

The five-seed diagnostic supports `lcp_weight = 0.002` as the best local
trade-off in this narrow coefficient neighborhood.

- `0.001` appears under-regularized: it has nonzero fall rate, the highest
  measured sensitivity, worse velocity tracking, higher joint acceleration,
  higher action jitter, and lower return.
- `0.004` lowers measured sensitivity and removes threshold violations, but it
  worsens velocity error, joint acceleration, and action jitter relative to
  `0.002`; return is nearly tied.
- `0.002` remains the best aggregate choice for fall, velocity error, joint
  acceleration, action jitter, and the primary paper's selected-checkpoint
  trade-off.

This is not a broad hyperparameter sweep and should not be written as proof of a
globally optimal coefficient. The defensible claim is narrower: the primary LCP
result is not an isolated single-point accident within the immediate
`0.001 / 0.002 / 0.004` neighborhood.
