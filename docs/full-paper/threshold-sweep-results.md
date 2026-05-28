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

The gate table below uses **selected-checkpoint** metrics (task-floor-then-smoothest
selection rule). This differs from the #57 mean-aggregation diagnostic which used
final-checkpoint gates. The #58 diagnostic switches to selected-checkpoint as the
promotion basis because the selection rule is what the paper protocol uses for
method comparison; final-checkpoint values are still recorded below for completeness.
The conclusion (no threshold passes all gates) holds under both bases.

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

### Per-Seed Final-Checkpoint Metrics (completeness)

For cross-reference with #57 which used final-checkpoint gates:

| Thresh | Seed | Ckpt | Fall | Vel err | Jnt acc | Jitter | Gate |
|--------|------|------|------|---------|---------|--------|------|
| 3.6 | 23 | 400 | 0.300 | 0.664 | 153.3 | 0.252 | FAIL |
| 3.6 | 29 | 400 | 0.450 | 0.766 | 168.9 | 0.262 | FAIL |
| 3.6 | 31 | 400 | 0.150 | 0.634 | 209.7 | 0.437 | FAIL |
| 3.7 | 23 | 400 | 1.000 | 0.974 | 250.4 | 0.502 | FAIL |
| 3.7 | 29 | 400 | 0.150 | 0.569 | 126.5 | 0.215 | PASS |
| 3.7 | 31 | 400 | 0.300 | 0.640 | 197.9 | 0.420 | FAIL |
| 3.8 | 23 | 400 | 0.100 | 0.677 | 120.8 | 0.221 | PASS |
| 3.8 | 29 | 400 | 0.550 | 0.652 | 247.3 | 0.475 | FAIL |
| 3.8 | 31 | 400 | 0.000 | 0.456 | 119.8 | 0.250 | PASS |
| 4.0 | 23 | 400 | 0.150 | 0.725 | 165.3 | 0.296 | FAIL |
| 4.0 | 29 | 400 | 0.250 | 0.668 | 114.3 | 0.244 | PASS |
| 4.0 | 31 | 400 | 0.850 | 0.846 | 207.7 | 0.390 | FAIL |

### Aggregate (Selected Checkpoints, Seeds 23/29/31)

| Thresh | Fall | Vel err | Jnt acc | Jitter | Return |
|--------|------|---------|---------|--------|--------|
| 3.6 | 0.300 | 0.676 | 169.4 | 0.308 | 87.9 |
| 3.7 | 0.250 | 0.645 | 177.8 | 0.360 | 94.3 |
| 3.8 | 0.217 | 0.595 | 162.6 | 0.315 | 99.0 |
| 4.0 | 0.350 | 0.759 | 145.2 | 0.274 | 91.9 |

## Interpretation

Seed29 CAN be repaired by changing the threshold. Threshold 3.7 gives seed29
fall=0.100, jnt=115.0 (vs 3.8 baseline: fall=0.550, jnt=247.3). Threshold 4.0
also repairs seed29 (fall=0.250, jnt=114.3).

However, the repair comes at the cost of breaking seed23 or seed31:

- 3.6 breaks all three seeds (too tight).
- 3.7 breaks seed23 (fall=0.600, jnt=235.7) and seed31 (jnt=182.9).
- 4.0 breaks seed31 (fall=0.650, vel=0.883) and seed23 (jnt=165.3).
- 3.8 breaks only seed29 but preserves 23 and 31.

Within this bounded threshold grid (3.6--4.0, three diagnostic seeds), no
shared threshold satisfies all gates. The evidence is consistent with
seed-dependent sensitivity but the sample is too small to rule out that a
different threshold or a per-seed adaptive scheme could work.

## Consequence

**The threshold lever is closed as a negative result.** The diagnosis is:

> In this bounded diagnostic, SC-PPO exhibits seed-dependent threshold
> sensitivity: no tested shared value simultaneously satisfies the promotion
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
- #53 (external constrained-RL baseline) remains a human decision. Revisit
  only after a human selects one specific external baseline (CPO, OmniSafe,
  or other) and approves the integration budget. The baseline should be
  framed as an independent comparison point, not as validation of SC-PPO
  dominance.
- Frame the full-paper narrative around the honest finding: Jacobian
  constraints work well for most seeds but show higher cross-seed variance
  than heuristic reward shaping.
