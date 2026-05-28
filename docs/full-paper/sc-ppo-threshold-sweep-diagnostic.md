# SC-PPO Threshold Sweep Diagnostic

**Branch:** `full-paper/extended-seeds`
**Issue:** #58 (Path B)
**Parent:** #51 (extended-seed audit), #56/#57 (mean-aggregation repair, failed)

## Motivation

The #51 five-seed audit showed SC-PPO threshold=3.8 partially fails on seed29
(55% fall rate, jnt_acc=247.3). The mean-aggregation repair (#57) made things
worse. A natural next question: is 3.8 on the wrong side of a seed29-specific
stability boundary?

This diagnostic sweeps three thresholds (3.6, 3.7, 4.0) against the baseline 3.8
on seeds {23, 29, 31}, holding all other parameters constant.

## Sweep

| Threshold | Config |
|-----------|--------|
| 3.6 | `configs/methods/sc_ppo_threshold_36_lambda_05_quantile_090_pid_lower_bound_clamp_extended_seeds.json` |
| 3.7 | `configs/methods/sc_ppo_threshold_37_lambda_05_quantile_090_pid_lower_bound_clamp_extended_seeds.json` |
| **3.8** | `configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_extended_seeds.json` (baseline) |
| 4.0 | `configs/methods/sc_ppo_threshold_40_lambda_05_quantile_090_pid_lower_bound_clamp_extended_seeds.json` |

All configs share: `subsample_obs=8`, `cost_aggregation=quantile(0.90)`,
`pid_integral_mode=lower_bound_clamp`, `lambda_init=0.5`, heuristic
smoothness rewards disabled, 512 envs, 400 iterations.

## Protocol

- Seeds: `{23, 29, 31}` (diagnostic set, not full 5-seed)
- Checkpoint grid: `{0, 100, 200, 300, 400}`
- Evaluation: 32 envs, 20 episodes per checkpoint
- Selection rule: task floor then smoothest (unchanged)

## Promotion Gates

Seed29 must satisfy ALL:
- `joint_acceleration_l2_mean <= 172.7` (recover half the gap to heuristic seed29)
- `action_jitter_l2_mean <= 0.351` (half-gap recovery)
- `velocity_tracking_error_mean <= 0.684` (preserve task floor)
- `fall_rate <= 0.600` (preserve task floor)

Seed31 must not regress beyond guard band from #51 SC-PPO seed31:
- `joint_acceleration_l2_mean <= 125.8`
- `action_jitter_l2_mean <= 0.263`
- `velocity_tracking_error_mean <= 0.479`
- `fall_rate <= 0.050`

Seed23 must preserve canonical final-checkpoint guard band:
- `joint_acceleration_l2_mean <= 126.8`
- `action_jitter_l2_mean <= 0.232`
- `velocity_tracking_error_mean <= 0.711`
- `fall_rate <= 0.150`

## Decision Rule

- If any threshold passes ALL three seed gates → promote to full
  `{11, 17, 23, 29, 31}` rerun with that threshold.
- If no threshold passes → record as negative result, close the threshold
  lever, do not expand to a broader sweep.

This diagnostic is bounded. It does NOT replace the #51 five-seed record by
itself.

## Artifacts

- Sweep config: `configs/sweeps/rough_terrain_threshold_sweep_diagnostic.json`
- Launch script: `scripts/baseline/run_threshold_sweep_diagnostic.sh`
- Analysis root: `artifacts/analysis/rough_terrain_threshold_sweep_diagnostic/`

## Launch

```bash
cd /home/zhuoxiang/SCO-humanoid
./scripts/baseline/run_threshold_sweep_diagnostic.sh all
```
