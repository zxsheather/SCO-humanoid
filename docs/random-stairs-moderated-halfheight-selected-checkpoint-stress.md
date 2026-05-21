# Moderated Half-Height Random-Stairs Selected-Checkpoint Stress Test

This note defines the second bounded `协议修复线` for issue `#15`.

## Scope

This line keeps the first post-freeze moderation in place:

- `terrain.terrain_proportions = [0, 0, 0.5, 0, 0, 0.25, 0.25]`

It then changes exactly one further severity axis:

- `SCO_HUMANOID_STAIR_HEIGHT_SCALE = 0.5`

So this is still an evaluation-only selected-checkpoint stress test of the already-selected
rough-terrain checkpoints. It is not retraining, not a new algorithm mainline, and not a rewrite
of the frozen rough-terrain claim.

## Why This Axis

The first moderated line reduced terrain composition severity, but every method still collapsed.

The next most direct knob is stair geometry itself. In the current upstream `HumanoidTerrain`
implementation, stair step height is hardcoded inside `.external/humanoid-gym/humanoid/utils/terrain.py`.
The outer repo therefore carries a small idempotent patch script that teaches the upstream checkout
to read:

- `SCO_HUMANOID_STAIR_HEIGHT_SCALE`

The half-height protocol keeps the mixed rough/stairs composition and halves only the stair step
height. `measure_heights = false` remains fixed to preserve the existing `705`-dim actor
observation contract.

## Required Local Patch

Check whether the local upstream checkout already contains the stair-height patch:

```bash
python scripts/baseline/patch_humanoid_gym_stair_height_scale.py --check
```

Apply it once if needed:

```bash
python scripts/baseline/patch_humanoid_gym_stair_height_scale.py
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

Plan the half-height run:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_moderated_halfheight_selected_checkpoint_stress.json \
  --stage plan
```

Run the full half-height selected-checkpoint stress test:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_moderated_halfheight_selected_checkpoint_stress.json \
  --stage all \
  --skip-completed
```

Run a bounded smoke for one selected checkpoint:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_moderated_halfheight_selected_checkpoint_stress.json \
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
artifacts/methods/random_stairs_moderated_halfheight_stress/
```

The aggregate summary will be written to:

```text
artifacts/analysis/random_stairs_moderated_halfheight_selected_checkpoint_stress/comparison_summary.json
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

- `Vanilla PPO`: `fall_rate = 1.0`, `VTE = 1.2840`, `JAL2 = 121.9797`, `jitter = 0.0166`, `return = 3.3868`
- revised heuristic anchor: `fall_rate = 1.0`, `VTE = 1.4844`, `JAL2 = 258.7346`, `jitter = 0.3912`, `return = 4.7974`
- `SC-PPO 3.8`: `fall_rate = 1.0`, `VTE = 1.3099`, `JAL2 = 230.5136`, `jitter = 0.3253`, `return = 5.8455`

So this line improved the engineering hygiene of the protocol repair, but it still did not recover a
task-valid evaluation regime. The next repair should keep this patch-backed wiring and change a more
direct traversal-discontinuity axis such as stair width or difficulty support, rather than only
shrinking stair height again.
