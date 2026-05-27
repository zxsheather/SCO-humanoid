# Extended-Seed Robustness Matrix

**Branch:** `full-paper/extended-seeds`
**Issue:** #51
**Decision issue:** #50 (closed — deferred for workshop, approved for full-paper upgrade)

## Seed Set

| Role | Seeds |
|------|-------|
| Historical (canonical) | `11, 17, 23` |
| Added (extended) | `29, 31` |
| **Full set** | **`11, 17, 23, 29, 31`** (5 seeds) |

Historical 3-seed record is preserved unchanged. Added seeds reuse the existing
training and checkpoint-sweep tooling without altering the metric schema or
selection rule.

## Method Rows

| # | Method | Config |
|---|--------|--------|
| 1 | SC-PPO 3.8 (PID-Lagrangian, threshold=3.8) | `configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_extended_seeds.json` |
| 2 | Revised heuristic anchor (action_rate=-0.0050) | `configs/methods/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget.json` |

## Protocol

- Training: 512 envs, 400 iterations, `cuda:0`
- Evaluation: 32 envs, 20 episodes per checkpoint
- Checkpoint sweep: `{0, 100, 200, 300, 400}` (SC-PPO), `{0, 50, ..., 400}` (heuristic, saved every 50)
- Selection rule: task floor then smoothest (same as canonical protocol)
- Analysis root: `artifacts/analysis/rough_terrain_extended_seeds/`

## Promotion Rule

SC-PPO 3.8 must:
1. Remain task-valid (not collapse) across all 5 seeds
2. Continue to beat the revised heuristic anchor on Isaac dynamic-smoothness metrics (joint acceleration, action jitter)
3. Not introduce multiple checkpoint-0 collapses

## MuJoCo Replay

Only for task-valid selected checkpoints on added seeds (29, 31).
Historical 3-seed MuJoCo record (11, 17, 23) is preserved as-is.

## Artifacts

- Sweep config: `configs/sweeps/rough_terrain_extended_seeds.json`
- Launch script: `scripts/baseline/run_extended_seeds_formal.sh`
- Analysis root: `artifacts/analysis/rough_terrain_extended_seeds/`

## Launch

```bash
# Training + evaluation
./scripts/baseline/run_extended_seeds_formal.sh all

# Training only (added seeds 29, 31 — skips completed 11, 17, 23)
./scripts/baseline/run_extended_seeds_formal.sh train

# Evaluation only
./scripts/baseline/run_extended_seeds_formal.sh evaluate
```

`--skip-completed` ensures historical seeds are not re-run if already present
in the analysis root.
