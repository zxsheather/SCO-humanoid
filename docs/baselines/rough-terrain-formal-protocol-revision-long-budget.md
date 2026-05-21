# Rough-Terrain Formal Protocol Revision Long-Budget Test

This note records the completed first explicit revised-protocol training test after the completed
`0 / 0 / 200` repaired-budget probe.

## Why this test existed

The current evidence had already ruled out two weaker explanations:

- `64 envs` alone did not explain the frozen collapse, because the repaired-budget run still
  collapsed all three seeds at `checkpoint 50`, which already matched the frozen total sample
  budget
- `512 envs x 200 iterations` was not enough by itself to repair the anchor, because it only
  narrowed the old heuristic winner to `0 / 0 / 200`

So the next falsifiable question was:

`under the same 512-env regime, do seed11 and seed17 recover if the run is extended past the repaired-budget 200-iteration stop?`

This remained a protocol-revision test, not a new heuristic-weight search.

## Revised candidate

- method config:
  `configs/methods/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget.json`
- sweep config:
  `configs/sweeps/rough_terrain_formal_protocol_revision_long_budget.json`

Regime:

- candidate: `action_rate = -0.0050`
- seeds: `11 / 17 / 23`
- `num_envs = 512`
- `max_iterations = 400`
- `save_interval = 50`
- selector: repaired `先过底线再取最平滑`

## Success rule

This test should only be considered a meaningful protocol-revision candidate if:

- all three seeds produce non-collapsed selected checkpoints
- the resulting heuristic row is task-valid on all three seeds
- the row does not rely on `checkpoint 0` to survive

If the test still returned a mixed split such as `0 / 0 / 200` or another partially collapsed
pattern, the repo should treat that as stronger evidence that baseline protocol revision needed a
larger change than simply extending the same run.

## Completed outcome

Canonical summary artifact:

- [comparison_summary.json](../../artifacts/analysis/rough_terrain_formal_protocol_revision_long_budget/comparison_summary.json)

Selected-checkpoint aggregate over seeds `11`, `17`, and `23`:

- `selected checkpoints = 350 / 300 / 350`
- `selection statuses = selected / selected / selected`
- `velocity_tracking_error_mean = 0.7549 +- 0.1068`
- `joint_acceleration_l2_mean = 119.8639 +- 2.1966`
- `action_jitter_l2_mean = 0.2711 +- 0.0084`
- `episode_return_mean = 100.9327 +- 11.2711`
- `fall_rate = 0.1500 +- 0.0816`

Per-seed selected checkpoints:

- `seed11 -> checkpoint 350`, `fall_rate = 0.15`, `velocity_tracking_error_mean = 0.6324`,
  `joint_acceleration_l2_mean = 119.4626`
- `seed17 -> checkpoint 300`, `fall_rate = 0.25`, `velocity_tracking_error_mean = 0.8926`,
  `joint_acceleration_l2_mean = 122.7323`
- `seed23 -> checkpoint 350`, `fall_rate = 0.05`, `velocity_tracking_error_mean = 0.7398`,
  `joint_acceleration_l2_mean = 117.3970`

## Selector reading

This run clears the stricter minimum revision rule:

- all three seeds produce non-collapsed selected checkpoints
- none of the surviving rows relies on `checkpoint 0`
- the heuristic row is task-valid on all three seeds

Per-seed selector details:

- `seed11` has only one eligible checkpoint: `350`
- `seed17` uses `checkpoint 400` as the task reference, keeps `300 / 400` eligible, and selects
  `300` because the repaired rule is `先过底线再取最平滑`
- `seed23` also uses `checkpoint 400` as the task reference, keeps `350 / 400` eligible, and
  selects `350` for the same reason

Important reporting nuance:

- `manifest.json` and plain `metrics.json` still reflect the latest checkpoint rather than the
  selected checkpoint
- for example, `seed23` latest `checkpoint 400` has better raw task metrics than `350`
  (`fall_rate = 0.00`, `velocity_tracking_error_mean = 0.6734`), but the selector still keeps
  `350` because the rule is `task floor then smoothness`
- canonical long-budget citations should therefore come from `metrics_selected.json`,
  `checkpoint_sweep_summary.json`, and `comparison_summary.json`

## Current interpretation

- the repo now has a usable revised heuristic formal anchor for the rough-terrain
  `三组正式对比`
- the completed `0 / 0 / 200` repaired-budget probe should now be read as the decision trigger for
  protocol revision, not as the final anchor itself
- the earlier frozen `64 envs x 400 iterations` regime remains the historical failure record for
  the baseline-side formal refresh
- this does not by itself close the whole report: the repo still needs to re-freeze the Isaac-side
  three-way wording and decide how to align `MuJoCo关键两组终验` with this revised anchor

## Command

Run it with:

```bash
env -u DISPLAY CUDA_VISIBLE_DEVICES=1 \
  /TinyNAS2024/zhuoxiang/sco-humanoid/bin/python -u \
  scripts/baseline/run_formal_comparison.py \
  --sweep-config configs/sweeps/rough_terrain_formal_protocol_revision_long_budget.json \
  --stage all \
  --skip-completed
```

## Canonical outputs

- `artifacts/analysis/rough_terrain_formal_protocol_revision_long_budget/comparison_summary.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed11/checkpoint_sweep_summary.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed17/checkpoint_sweep_summary.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed23/checkpoint_sweep_summary.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed11/metrics_selected.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed17/metrics_selected.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed23/metrics_selected.json`
