# SC-PPO Threshold Sweep Results

**Branch:** `full-paper/extended-seeds`
**Issue:** #58 (Path B)
**Run:** `threshold-sweep-20260528T050151Z`

## Status

The threshold sweep completed with `exit_status=0` and wrote:

- `artifacts/analysis/rough_terrain_threshold_sweep_diagnostic/comparison_summary.json`
- `artifacts/analysis/rough_terrain_threshold_sweep_diagnostic/logs/threshold-sweep-20260528T050151Z.log`

## Result

**No threshold passes all three seed gates.** The sweep is a negative result.

### Per-Seed Selected Checkpoint Metrics

| Thresh | Seed | Ckpt | Fall | Vel err | Jnt acc | Jitter | Sens | Gate |
|--------|------|------|------|---------|---------|--------|------|------|
| 3.6 | 23 | 400 | 0.300 | 0.664 | 153.3 | 0.252 | 3.22 | FAIL |
| 3.6 | 29 | 300 | 0.450 | 0.729 | 145.1 | 0.234 | 3.31 | FAIL |
| 3.6 | 31 | 400 | 0.150 | 0.634 | 209.7 | 0.437 | 3.39 | FAIL |
| 3.7 | 23 | 200 | 0.600 | 0.806 | 235.7 | 0.492 | 3.67 | FAIL |
| 3.7 | 29 | 300 | 0.100 | 0.574 | 115.0 | 0.197 | 3.50 | PASS |
| 3.7 | 31 | 300 | 0.050 | 0.556 | 182.9 | 0.392 | 3.85 | FAIL |
| 3.8 | 23 | 400 | 0.100 | 0.677 | 120.8 | 0.221 | 3.51 | PASS |
| 3.8 | 29 | 400 | 0.550 | 0.652 | 247.3 | 0.475 | 3.67 | FAIL |
| 3.8 | 31 | 400 | 0.000 | 0.456 | 119.8 | 0.250 | 3.70 | PASS |
| 4.0 | 23 | 400 | 0.150 | 0.725 | 165.3 | 0.296 | 3.89 | FAIL |
| 4.0 | 29 | 400 | 0.250 | 0.668 | 114.3 | 0.244 | 3.85 | PASS |
| 4.0 | 31 | 200 | 0.650 | 0.883 | 156.2 | 0.281 | 4.15 | FAIL |

### Aggregate (Selected Checkpoints, Seeds 23/29/31)

| Thresh | Fall | Vel err | Jnt acc | Jitter | Return |
|--------|------|---------|---------|--------|--------|
| 3.6 | 0.300 | 0.676 | 169.4 | 0.308 | 82.8 |
| 3.7 | 0.250 | 0.645 | 177.8 | 0.360 | 87.6 |
| 3.8 | 0.217 | 0.595 | 162.6 | 0.315 | 99.3 |
| 4.0 | 0.350 | 0.759 | 145.2 | 0.274 | 73.1 |

## Interpretation

Seed29 CAN be repaired by changing the threshold. Threshold 3.7 gives seed29
fall=0.100, jnt=115.0 (vs 3.8 baseline: fall=0.550, jnt=247.3). Threshold 4.0
also repairs seed29 (fall=0.250, jnt=114.3).

However, the repair comes at the cost of breaking seed23 or seed31:

- 3.6 breaks all three seeds (too tight).
- 3.7 breaks seed23 (fall=0.600, jnt=235.7) and seed31 (jnt=182.9).
- 4.0 breaks seed31 (fall=0.650, vel=0.883) and seed23 (jnt=165.3).
- 3.8 breaks only seed29 but preserves 23 and 31.

No single threshold works across all three diagnostic seeds. This is a
fundamental seed-sensitivity trade-off, not a parameter-tuning oversight.

## Consequence

**The threshold lever is closed as a negative result.** The diagnosis is:

> SC-PPO exhibits inherent seed sensitivity: the optimal threshold varies
> across seeds, and no single value simultaneously satisfies the promotion
> gates for seeds 23, 29, and 31. Lower thresholds over-constrain seeds
> 23/31; higher thresholds under-constrain seed 31.

Combined with the mean-aggregation repair (#57), both SC-PPO-family repair
levers attempted so far are negative:

| Lever | Result |
|-------|--------|
| cost_aggregation (quantile→mean) | All seeds worsened (#57) |
| threshold sweep (3.6/3.7/4.0) | Seed29 repairable but breaks others (#58) |

The full-paper path should not pursue further SC-PPO repair levers unless a
new hypothesis identifies a mechanism that addresses the cross-seed variance
directly (e.g., per-seed adaptive threshold, seed-conditioned constraint
scheduling). Simple parameter sweeps are insufficient.

## Next Steps

- Accept SC-PPO seed sensitivity as a documented method-level limitation.
- Consider #53 (external constrained-RL baseline) as an independent
  comparison rather than relying on SC-PPO dominance.
- Frame the full-paper narrative around the honest finding: Jacobian
  constraints work well for most seeds but show higher cross-seed variance
  than heuristic reward shaping.
