# Rough-Terrain Formal Comparison

Issue `#5` upgrades the rough-terrain main comparison from single-run baselines to a formal
`3-seed + checkpoint-sweep` evidence set.

## Scope

- `Vanilla PPO` stays the raw reference
- the bounded heuristic baseline family is required to produce a surviving formal comparison anchor
- `SC-PPO 3.8` keeps its existing mainline evidence and is not rerun inside this note

## Frozen formal-compare setup

The completed frozen formal-compare batch used:

- `configs/methods/vanilla_ppo_full_compare.json`
- `configs/methods/heuristic_smoothing_action_rate_0005_full_compare.json`
- `configs/methods/heuristic_smoothing_action_rate_0020_full_compare.json`
- `configs/methods/heuristic_smoothing_action_rate_0050_full_compare.json`
- `configs/sweeps/rough_terrain_formal_comparison.json`
- `num_envs = 64`
- `max_iterations = 400`
- `runner.save_interval = 50`

This lower `num_envs` freeze was chosen because the shared machine state could not reliably
support the older `512 envs` baseline setting during the formal comparison window.

Run it with:

```bash
env -u DISPLAY CUDA_VISIBLE_DEVICES=1 /home/zhuoxiang/miniconda3/envs/ecolab-isaacgym/bin/python -u scripts/baseline/run_formal_comparison.py --stage all --skip-completed
```

## Completed frozen failure record

The frozen formal compare covers `Vanilla PPO` and the bounded heuristic action-rate family.

Canonical summary artifact:

- [comparison_summary.json](../../artifacts/analysis/rough_terrain_formal_comparison/comparison_summary.json)

Selected-checkpoint aggregate over seeds `11`, `17`, and `23`:

| Candidate | Selected checkpoints | `velocity_tracking_error_mean` | `joint_acceleration_l2_mean` | `action_jitter_l2_mean` | `episode_return_mean` | `fall_rate` |
| --- | --- | --- | --- | --- | --- | --- |
| `Vanilla PPO` | `0 / 0 / 0` | `1.3321 +- 0.1181` | `83.7179 +- 13.3692` | `0.0161 +- 0.0008` | `4.0002 +- 0.4323` | `1.0000 +- 0.0000` |
| `Heuristic smoothing action_rate = -0.0005` | `0 / 0 / 0` | `1.3451 +- 0.1269` | `83.7119 +- 14.9052` | `0.0161 +- 0.0009` | `4.1998 +- 0.4037` | `1.0000 +- 0.0000` |
| `Heuristic smoothing action_rate = -0.0020` | `0 / 0 / 0` | `1.3436 +- 0.1232` | `85.5995 +- 13.7253` | `0.0161 +- 0.0009` | `4.1811 +- 0.3680` | `1.0000 +- 0.0000` |
| `Heuristic smoothing action_rate = -0.0050` | `0 / 0 / 0` | `1.3359 +- 0.1232` | `80.5803 +- 14.6031` | `0.0160 +- 0.0009` | `4.1769 +- 0.4080` | `1.0000 +- 0.0000` |

Reading:

- all twelve selected checkpoints are `checkpoint 0`
- this is not just a selector quirk: every evaluated checkpoint inside every completed baseline
  sweep still has `fall_rate = 1.0`
- `Vanilla PPO` collapse should be recorded as raw-reference evidence, which is the intended role
  of the raw reference row in this matrix
- the selected heuristic family did not survive the frozen formal `3-seed + checkpoint-sweep`
  anchor standard

## Selector repair follow-up

The repo repaired one checkpoint-selector bug after the frozen run:

- `evaluate_checkpoint_sweep.py` no longer selects checkpoints by smoothness score alone
- it now applies `先过底线再取最平滑` inside one run:
  - first find the run's task reference checkpoint
  - then keep only checkpoints within the repo's `10% tracking / 5pp fall-rate` task floor
  - then select the smoothest remaining checkpoint
- if every evaluated checkpoint still has `fall_rate = 1.0`, the sweep is now marked as
  `all_checkpoints_collapsed` rather than silently pretending that `checkpoint 0` is a valid
  report-grade choice

This repair changed one historical read:

- on the previous single-run `action_rate = -0.0050` heuristic artifact trained under
  `512 envs x 200 iterations`, a retro sweep with the repaired selector picks `checkpoint 200`
  rather than `checkpoint 0`

However, after regenerating the completed `64 envs x 400 iterations` formal-compare sweeps under
the repaired selector, the full bounded heuristic family still remained `all_checkpoints_collapsed`.

So the remaining blocker was no longer selector semantics alone.
It was the frozen baseline-side formal training regime itself.

## Repaired-budget probe outcome

The prepared repaired-budget tracer bullet is complete:

- [rough-terrain formal protocol repair probe](./rough-terrain-formal-protocol-repair-probe.md)
- [probe summary artifact](../../artifacts/analysis/rough_terrain_formal_protocol_repair_probe/comparison_summary.json)

Outcome:

- `selected checkpoints = 0 / 0 / 200`
- `seed11 -> all_checkpoints_collapsed`
- `seed17 -> all_checkpoints_collapsed`
- `seed23 -> checkpoint 200`

This changed the interpretation in one important way:

- the previous heuristic winner was not universally collapsed once both the selector repair and the
  repaired `512 envs x 200 iterations` budget were applied

But it still did not close the report-grade blocker:

- the repaired-budget probe did not produce a defensible `3-seed` formal heuristic anchor
- `2 / 3` seeds remained collapsed
- and the surviving seed still recorded `fall_rate = 0.75`

So the repo had to treat the baseline blocker as:

`explicit protocol diagnosis / possible protocol revision`

## Revised protocol replacement outcome

The prepared first revised-protocol long-budget run is now complete:

- [rough-terrain formal protocol revision decision](./rough-terrain-formal-protocol-revision-decision.md)
- [rough-terrain formal protocol revision long-budget test](./rough-terrain-formal-protocol-revision-long-budget.md)
- [revision summary artifact](../../artifacts/analysis/rough_terrain_formal_protocol_revision_long_budget/comparison_summary.json)

Outcome:

- `selected checkpoints = 350 / 300 / 350`
- all three seeds are `selected`
- aggregate metrics:
  - `velocity_tracking_error_mean = 0.7549 +- 0.1068`
  - `joint_acceleration_l2_mean = 119.8639 +- 2.1966`
  - `action_jitter_l2_mean = 0.2711 +- 0.0084`
  - `episode_return_mean = 100.9327 +- 11.2711`
  - `fall_rate = 0.1500 +- 0.0816`

Interpretation:

- the frozen `64 envs x 400 iterations` heuristic rows in this note remain the historical failure
  record that triggered the baseline-side `协议修复线`
- they are no longer the current heuristic formal anchor
- per `CONTEXT.md`, `Vanilla PPO` still remains the raw-reference collapse record
- the current heuristic anchor should now come from the revised `512 envs x 400 iterations`
  long-budget regime rather than from the frozen regime shown above

## Consequence for Issue #5

Per the current rule in `CONTEXT.md`, this outcome now means:

- do not treat the frozen `64 envs x 400 iterations` heuristic row as if it had passed the
  refreshed formal-anchor standard
- use the revised long-budget heuristic row, not the frozen row, when restating the current
  rough-terrain formal anchor
- keep `Vanilla PPO` collapse as raw-reference evidence
- do not reopen bounded heuristic search as if the remaining problem were still weight choice
- move to re-freezing the rough-terrain three-way comparison wording

So this note now serves primarily as the completed failure record for the original frozen baseline
protocol, while the revised long-budget heuristic note provides the current formal anchor.

## Canonical artifacts

- [frozen formal comparison summary](../../artifacts/analysis/rough_terrain_formal_comparison/comparison_summary.json)
- [repaired-budget probe summary](../../artifacts/analysis/rough_terrain_formal_protocol_repair_probe/comparison_summary.json)
- [revised long-budget summary](../../artifacts/analysis/rough_terrain_formal_protocol_revision_long_budget/comparison_summary.json)
- [vanilla seed11 checkpoint sweep](../../artifacts/methods/vanilla_ppo_full_compare/vanilla_ppo_full_compare_rough_terrain_seed11/checkpoint_sweep_summary.json)
- [vanilla seed17 checkpoint sweep](../../artifacts/methods/vanilla_ppo_full_compare/vanilla_ppo_full_compare_rough_terrain_seed17/checkpoint_sweep_summary.json)
- [vanilla seed23 checkpoint sweep](../../artifacts/methods/vanilla_ppo_full_compare/vanilla_ppo_full_compare_rough_terrain_seed23/checkpoint_sweep_summary.json)
- [heuristic 0005 seed11 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_full_compare/heuristic_smoothing_action_rate_0005_full_compare_rough_terrain_seed11/checkpoint_sweep_summary.json)
- [heuristic 0005 seed17 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_full_compare/heuristic_smoothing_action_rate_0005_full_compare_rough_terrain_seed17/checkpoint_sweep_summary.json)
- [heuristic 0005 seed23 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_full_compare/heuristic_smoothing_action_rate_0005_full_compare_rough_terrain_seed23/checkpoint_sweep_summary.json)
- [heuristic 0020 seed11 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_full_compare/heuristic_smoothing_action_rate_0020_full_compare_rough_terrain_seed11/checkpoint_sweep_summary.json)
- [heuristic 0020 seed17 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_full_compare/heuristic_smoothing_action_rate_0020_full_compare_rough_terrain_seed17/checkpoint_sweep_summary.json)
- [heuristic 0020 seed23 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_full_compare/heuristic_smoothing_action_rate_0020_full_compare_rough_terrain_seed23/checkpoint_sweep_summary.json)
- [heuristic 0050 frozen seed11 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_full_compare/heuristic_smoothing_action_rate_0050_full_compare_rough_terrain_seed11/checkpoint_sweep_summary.json)
- [heuristic 0050 frozen seed17 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_full_compare/heuristic_smoothing_action_rate_0050_full_compare_rough_terrain_seed17/checkpoint_sweep_summary.json)
- [heuristic 0050 frozen seed23 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_full_compare/heuristic_smoothing_action_rate_0050_full_compare_rough_terrain_seed23/checkpoint_sweep_summary.json)

## Immediate next step

The immediate next step is no longer to decide whether protocol revision is necessary.
It is:

`re-freeze the rough-terrain three-way comparison around Vanilla PPO raw reference, the revised heuristic anchor, and SC-PPO 3.8`

Canonical follow-up notes:

- [rough-terrain formal protocol revision decision](./rough-terrain-formal-protocol-revision-decision.md)
- [rough-terrain formal protocol revision long-budget test](./rough-terrain-formal-protocol-revision-long-budget.md)
