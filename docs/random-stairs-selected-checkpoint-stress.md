# Random-Stairs Selected-Checkpoint Stress Test

This note defines the first implementation path for issue `#7`.

## Scope

This is a bounded `复杂地形条件` pressure test.

It keeps the primary task fixed as `速度跟踪行走` and evaluates already-selected rough-terrain
checkpoints under random stairs. It is not a retraining protocol, not a new method line, and not a
rewrite of the completed Isaac rough-terrain claim.

## Protocol

Terrain override:

- `terrain.mesh_type = "trimesh"`
- `terrain.curriculum = false`
- `terrain.measure_heights = false`
- `terrain.num_rows = 4`
- `terrain.num_cols = 8`
- `terrain.terrain_proportions = [0, 0, 0, 0, 0, 0.5, 0.5]`

The important compatibility rule is `measure_heights = false`. Existing actors were trained with
the current `705`-dim observation contract, so the first random-stairs pass must not add height
measurements to actor observations.

Selected checkpoints:

| Method | Seed 11 | Seed 17 | Seed 23 |
| --- | ---: | ---: | ---: |
| `Vanilla PPO` raw reference | `0` | `0` | `0` |
| revised heuristic anchor | `350` | `300` | `350` |
| `SC-PPO threshold = 3.8` | `300` | `300` | `400` |

## Commands

Plan the run:

```bash
python scripts/baseline/run_random_stairs_stress_test.py --stage plan
```

Run the full selected-checkpoint stress test:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --stage all \
  --skip-completed
```

Run a bounded smoke for one selected checkpoint:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --candidate sc_ppo \
  --seed 11 \
  --stage all \
  --run-suffix smoke_1ep \
  --episodes 1 \
  --eval-num-envs 8 \
  --skip-completed
```

## Artifacts

Per-run artifacts are written under:

```text
artifacts/methods/random_stairs_stress/
```

The aggregate summary is written to:

```text
artifacts/analysis/random_stairs_selected_checkpoint_stress/comparison_summary.json
```

## Reading Rule

The result should answer whether the rough-terrain `SC-PPO 3.8` advantage survives, degrades, or
only partially transfers to random stairs. Even if `SC-PPO` wins or loses, this remains a pressure
test of selected checkpoints under a harsher `复杂地形条件`, not a new headline claim.
