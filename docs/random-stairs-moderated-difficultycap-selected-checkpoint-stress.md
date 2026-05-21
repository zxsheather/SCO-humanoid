# Moderated Stair-Difficulty-Cap Random-Stairs Selected-Checkpoint Stress Test

This note defines the fourth bounded `协议修复线` for issue `#15`.

## Scope

This line keeps the first post-freeze moderation in place:

- `terrain.terrain_proportions = [0, 0, 0.5, 0, 0, 0.25, 0.25]`

It then changes exactly one further severity axis:

- `SCO_HUMANOID_STAIR_DIFFICULTY_CAP = 0.5`

So this is still an evaluation-only selected-checkpoint stress test of the already-selected
rough-terrain checkpoints. It is not retraining, not a new algorithm mainline, and not a rewrite
of the frozen rough-terrain claim.

## Why This Axis

The half-height and wide-step lines each changed stair geometry globally, but every method still
collapsed.

The next more structural knob is stair difficulty support itself. In the current upstream
`HumanoidTerrain` implementation, stair severity is still inherited directly from the shared
`difficulty ~ U(0,1)` draw. This line therefore uses a local upstream patch that teaches the
checkout to read:

- `SCO_HUMANOID_STAIR_DIFFICULTY_CAP`

The difficulty-cap protocol keeps stair width and height scaling at their default values and clips
only the stair branch to `min(difficulty, 0.5)`. `measure_heights = false` remains fixed to
preserve the existing `705`-dim actor observation contract.

## Required Local Patch

Check whether the local upstream checkout already contains the stair-difficulty patch:

```bash
python scripts/baseline/patch_humanoid_gym_stair_difficulty_env.py --check
```

Apply it once if needed:

```bash
python scripts/baseline/patch_humanoid_gym_stair_difficulty_env.py
```

The patched configs fail fast if this local patch is missing, so the sweep will not silently run
with the old hardcoded stairs.

## Selected Checkpoints

| Method | Seed 11 | Seed 17 | Seed 23 |
| --- | ---: | ---: | ---: |
| `Vanilla PPO` raw reference | `0` | `0` | `0` |
| revised heuristic anchor | `350` | `300` | `350` |
| `SC-PPO threshold = 3.8` | `300` | `300` | `400` |

## Commands

Plan the difficulty-cap run:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_moderated_difficultycap_selected_checkpoint_stress.json \
  --stage plan
```

Run the full difficulty-cap selected-checkpoint stress test:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_moderated_difficultycap_selected_checkpoint_stress.json \
  --stage all \
  --skip-completed
```

Run a bounded smoke for one selected checkpoint:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_moderated_difficultycap_selected_checkpoint_stress.json \
  --seed 11 \
  --stage all \
  --run-suffix smoke_1ep \
  --episodes 1 \
  --eval-num-envs 8 \
  --skip-completed
```

## Artifacts

Per-run artifacts will be written under:

```text
artifacts/methods/random_stairs_moderated_difficultycap_stress/
```

The aggregate summary will be written to:

```text
artifacts/analysis/random_stairs_moderated_difficultycap_selected_checkpoint_stress/comparison_summary.json
```

## Success Criterion

The first goal is still protocol discriminability:

- at least one task-valid method should stop collapsing
- only after that can the repo read metric ordering as a task-valid comparison
- if everything still has `fall_rate = 1.0`, this branch remains a protocol-failure record rather
  than a method-comparison result

## Outcome

The completed `3 seeds x 3 methods x 20 episodes` run still ended in:

- `task_validity_outcome = all_methods_collapsed`

Aggregate readout:

- `Vanilla PPO`: `fall_rate = 1.0`, `VTE = 1.3397`, `JAL2 = 123.6395`, `jitter = 0.0167`, `return = 3.1413`
- revised heuristic anchor: `fall_rate = 1.0`, `VTE = 1.4204`, `JAL2 = 246.7482`, `jitter = 0.3790`, `return = 5.1086`
- `SC-PPO 3.8`: `fall_rate = 1.0`, `VTE = 1.2987`, `JAL2 = 233.8049`, `jitter = 0.3226`, `return = 5.6586`

So capping stair difficulty at `0.5` still did not recover a task-valid evaluation regime. This
axis helped the revised heuristic more than the previous repairs on descriptive metrics, but it was
still not enough to move any method out of full collapse. At this point the repo has tried
composition moderation, half-height stairs, wide-step stairs, and stair-difficulty support capping;
the most reasonable next decision is to stop small evaluation-only protocol retunes and treat the
current random-stairs generator as misaligned with the selected rough-terrain checkpoints unless a
more invasive terrain-generation rewrite is justified.
