# Moderated Wide-Step Random-Stairs Selected-Checkpoint Stress Test

This note defines the third bounded `协议修复线` for issue `#15`.

## Scope

This line keeps the first post-freeze moderation in place:

- `terrain.terrain_proportions = [0, 0, 0.5, 0, 0, 0.25, 0.25]`

It then changes exactly one further severity axis:

- `SCO_HUMANOID_STAIR_WIDTH_SCALE = 2.0`

So this is still an evaluation-only selected-checkpoint stress test of the already-selected
rough-terrain checkpoints. It is not retraining, not a new algorithm mainline, and not a rewrite
of the frozen rough-terrain claim.

## Why This Axis

The half-height line reduced vertical discontinuity but every method still collapsed.

The next more direct traversal knob is stair transition frequency. In the current upstream
`HumanoidTerrain` implementation, stair `step_width` is hardcoded inside
`.external/humanoid-gym/humanoid/utils/terrain.py`. This line therefore uses a local upstream patch
that teaches the checkout to read:

- `SCO_HUMANOID_STAIR_WIDTH_SCALE`

The wide-step protocol keeps stair height unchanged and doubles only stair width. `measure_heights =
false` remains fixed to preserve the existing `705`-dim actor observation contract.

## Required Local Patch

Check whether the local upstream checkout already contains the stair-geometry patch:

```bash
python scripts/baseline/patch_humanoid_gym_stair_geometry_env.py --check
```

Apply it once if needed:

```bash
python scripts/baseline/patch_humanoid_gym_stair_geometry_env.py
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

Plan the wide-step run:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_moderated_widestep_selected_checkpoint_stress.json \
  --stage plan
```

Run the full wide-step selected-checkpoint stress test:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_moderated_widestep_selected_checkpoint_stress.json \
  --stage all \
  --skip-completed
```

Run a bounded smoke for one selected checkpoint:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_moderated_widestep_selected_checkpoint_stress.json \
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
artifacts/methods/random_stairs_moderated_widestep_stress/
```

The aggregate summary will be written to:

```text
artifacts/analysis/random_stairs_moderated_widestep_selected_checkpoint_stress/comparison_summary.json
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

- `Vanilla PPO`: `fall_rate = 1.0`, `VTE = 1.3023`, `JAL2 = 117.9810`, `jitter = 0.0166`, `return = 3.3371`
- revised heuristic anchor: `fall_rate = 1.0`, `VTE = 1.4970`, `JAL2 = 251.9615`, `jitter = 0.3853`, `return = 4.9512`
- `SC-PPO 3.8`: `fall_rate = 1.0`, `VTE = 1.2381`, `JAL2 = 236.5796`, `jitter = 0.3214`, `return = 5.3626`

So doubling stair width still did not recover a task-valid evaluation regime. Relative to the
composition-only line, it slightly improved `SC-PPO` velocity tracking and slightly improved the
revised heuristic smoothness metrics, but not enough to move any method out of full collapse. The
next repair should probably change difficulty support or the stair-generation rule itself, rather
than continuing with small evaluation-only geometry retunes.
