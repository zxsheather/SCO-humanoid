# Rough-Terrain Formal Comparison

Issue `#5` upgrades the rough-terrain main comparison from single-run baselines to a formal
`3-seed + checkpoint-sweep` evidence set.

## Scope

- `Vanilla PPO` stays the raw reference
- the bounded heuristic baseline family is required to produce a surviving formal comparison anchor
- `SC-PPO 3.8` keeps its existing mainline evidence and is not rerun inside this note

## Frozen formal-compare setup

The completed formal-compare batch used:

- `configs/methods/vanilla_ppo_full_compare.json`
- `configs/methods/heuristic_smoothing_action_rate_0005_full_compare.json`
- `configs/methods/heuristic_smoothing_action_rate_0020_full_compare.json`
- `configs/methods/heuristic_smoothing_action_rate_0050_full_compare.json`
- `configs/sweeps/rough_terrain_formal_comparison.json`
- `num_envs = 64`
- `max_iterations = 400`
- `runner.save_interval = 50`

This lower `num_envs` freeze was chosen because the shared machine state could not reliably support
the older `512 envs` baseline setting during the formal comparison window.

Run it with:

```bash
env -u DISPLAY CUDA_VISIBLE_DEVICES=1 /home/zhuoxiang/miniconda3/envs/ecolab-isaacgym/bin/python -u scripts/baseline/run_formal_comparison.py --stage all --skip-completed
```

## Completed failure record

The formal compare now covers Vanilla PPO and the bounded heuristic action-rate family.

Canonical summary artifact:

- [comparison_summary.json](../../artifacts/analysis/rough_terrain_formal_comparison/comparison_summary.json)

Selected-checkpoint aggregate over seeds `11`, `17`, and `23`:

| Candidate | Selected checkpoints | `velocity_tracking_error_mean` | `joint_acceleration_l2_mean` | `action_jitter_l2_mean` | `episode_return_mean` | `fall_rate` |
| --- | --- | --- | --- | --- | --- | --- |
| `Vanilla PPO` | `0 / 0 / 0` | `1.3321 +- 0.1181` | `83.7179 +- 13.3692` | `0.0161 +- 0.0008` | `4.0002 +- 0.4323` | `1.0000 +- 0.0000` |
| `Heuristic smoothing action_rate = -0.0005` | `0 / 0 / 0` | `1.3451 +- 0.1269` | `83.7119 +- 14.9052` | `0.0161 +- 0.0009` | `4.1998 +- 0.4037` | `1.0000 +- 0.0000` |
| `Heuristic smoothing action_rate = -0.0020` | `0 / 0 / 0` | `1.3436 +- 0.1232` | `85.5995 +- 13.7253` | `0.0161 +- 0.0009` | `4.1811 +- 0.3680` | `1.0000 +- 0.0000` |
| `Heuristic smoothing action_rate = -0.0050` | `0 / 0 / 0` | `1.3359 +- 0.1232` | `80.5803 +- 14.6031` | `0.0160 +- 0.0009` | `4.1769 +- 0.4080` | `1.0000 +- 0.0000` |

## Reading

- All twelve selected checkpoints are `checkpoint 0`.
- This is not just a selector quirk. In every completed sweep under the full candidate set, every
  evaluated checkpoint still has `fall_rate = 1.0`, so the whole baseline side stays task-invalid
  through the full `0 -> 400` checkpoint range.
- `Vanilla PPO` collapse should be recorded as raw-reference evidence, which is the intended role
  of the raw reference row in this matrix.
- The selected heuristic family does not survive the formal `3-seed + checkpoint-sweep` anchor
  standard and therefore does not remain a report-grade heuristic anchor.

## Selector repair follow-up

The repo has now repaired one protocol bug in the checkpoint selector:

- `evaluate_checkpoint_sweep.py` no longer selects checkpoints by smoothness score alone
- it now applies `先过底线再取最平滑` inside one run:
  - first find the run's task reference checkpoint
  - then keep only checkpoints within the repo's `10% tracking / 5pp fall-rate` task floor
  - then select the smoothest remaining checkpoint
- if every evaluated checkpoint still has `fall_rate = 1.0`, the sweep is now marked as
  `all_checkpoints_collapsed` rather than silently pretending that `checkpoint 0` is a valid
  report-grade choice

This repair changes one important historical read:

- on the previous single-run `action_rate = -0.0050` heuristic artifact trained under
  `512 envs x 200 iterations`, a retro sweep with the repaired selector now picks
  `checkpoint 200` rather than `checkpoint 0`
- that retro result means the old heuristic line was not universally collapsed; part of the issue
  was the selector itself

However, after regenerating all existing `64 envs x 400 iterations` formal-compare sweeps under
the repaired selector, the full bounded heuristic family still remains `all_checkpoints_collapsed`.

So the remaining blocker is no longer selector semantics alone. It is now the baseline-side formal
training regime itself.

## Consequence for Issue #5

Per the current rule in `CONTEXT.md`, this outcome means:

- do not treat the old single-run heuristic winner as if it had passed the refreshed formal-anchor
  standard
- do not count `#5` as closed
- treat the baseline side as a protocol-repair problem before claiming a closed report-grade
  three-way comparison

So this note now serves as a completed failure record for the baseline side, and the next step is
protocol repair rather than another heuristic-anchor search.

## Canonical artifacts

- [formal comparison summary](../../artifacts/analysis/rough_terrain_formal_comparison/comparison_summary.json)
- [vanilla seed11 checkpoint sweep](../../artifacts/methods/vanilla_ppo_full_compare/vanilla_ppo_full_compare_rough_terrain_seed11/checkpoint_sweep_summary.json)
- [vanilla seed17 checkpoint sweep](../../artifacts/methods/vanilla_ppo_full_compare/vanilla_ppo_full_compare_rough_terrain_seed17/checkpoint_sweep_summary.json)
- [vanilla seed23 checkpoint sweep](../../artifacts/methods/vanilla_ppo_full_compare/vanilla_ppo_full_compare_rough_terrain_seed23/checkpoint_sweep_summary.json)
- [heuristic 0005 seed11 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_full_compare/heuristic_smoothing_action_rate_0005_full_compare_rough_terrain_seed11/checkpoint_sweep_summary.json)
- [heuristic 0005 seed17 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_full_compare/heuristic_smoothing_action_rate_0005_full_compare_rough_terrain_seed17/checkpoint_sweep_summary.json)
- [heuristic 0005 seed23 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_full_compare/heuristic_smoothing_action_rate_0005_full_compare_rough_terrain_seed23/checkpoint_sweep_summary.json)
- [heuristic 0020 seed11 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_full_compare/heuristic_smoothing_action_rate_0020_full_compare_rough_terrain_seed11/checkpoint_sweep_summary.json)
- [heuristic 0020 seed17 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_full_compare/heuristic_smoothing_action_rate_0020_full_compare_rough_terrain_seed17/checkpoint_sweep_summary.json)
- [heuristic 0020 seed23 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_full_compare/heuristic_smoothing_action_rate_0020_full_compare_rough_terrain_seed23/checkpoint_sweep_summary.json)
- [heuristic seed11 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_full_compare/heuristic_smoothing_action_rate_0050_full_compare_rough_terrain_seed11/checkpoint_sweep_summary.json)
- [heuristic seed17 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_full_compare/heuristic_smoothing_action_rate_0050_full_compare_rough_terrain_seed17/checkpoint_sweep_summary.json)
- [heuristic seed23 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_full_compare/heuristic_smoothing_action_rate_0050_full_compare_rough_terrain_seed23/checkpoint_sweep_summary.json)

## Immediate next step

The next baseline-side step is not another SC-PPO threshold promotion. It is:

`repair the formal-compare protocol before spending more report-grade budget downstream`

The current tracer-bullet repair run is:

`rerun only the old heuristic winner (action_rate = -0.0050) under a repaired formal budget: 512 envs, 200 iterations, save_interval = 50, 3 seeds, repaired checkpoint selector`

Prepared sweep config:

- `configs/sweeps/rough_terrain_formal_protocol_repair_probe.json`

Prepared command:

```bash
env -u DISPLAY CUDA_VISIBLE_DEVICES=1 \
  /home/zhuoxiang/miniconda3/envs/ecolab-isaacgym/bin/python -u \
  scripts/baseline/run_formal_comparison.py \
  --sweep-config configs/sweeps/rough_terrain_formal_protocol_repair_probe.json \
  --stage all \
  --skip-completed
```
