# Structured-Stairs Random-Stairs Selected-Checkpoint Stress Test

This note defines the first post-`#15` terrain-generator redesign line tracked by issue `#16`.

## Scope

This line keeps the selected rough-terrain checkpoints fixed and keeps the mixed rough/stairs
composition:

- `terrain.terrain_proportions = [0, 0, 0.5, 0, 0, 0.25, 0.25]`
- `measure_heights = false`

It also keeps the moderate independent stair band from the last rewrite probe:

- `SCO_HUMANOID_STAIR_DIFFICULTY_MIN = 0.0`
- `SCO_HUMANOID_STAIR_DIFFICULTY_MAX = 0.35`

What changes is the stair topology itself. Instead of delegating stair tiles to
`terrain_utils.pyramid_stairs_terrain(...)`, this redesign patch replaces them with a structured
forward staircase segment that preserves:

- a flat spawn-side runway
- a bounded step count
- an explicit top landing

Runtime controls for the new topology:

- `SCO_HUMANOID_STRUCTURED_STAIR_STEP_COUNT = 4`
- `SCO_HUMANOID_STRUCTURED_STAIR_RUNWAY_M = 1.0`
- `SCO_HUMANOID_STRUCTURED_STAIR_LANDING_M = 1.0`

## Why This Redesign

Issue `#15` exhausted five smaller repair lines:

- composition moderation
- half-height stairs
- wide-step stairs
- stair difficulty cap
- decoupled stair-difficulty band

All five still ended in `all_methods_collapsed`.

That means the repo has strong evidence against the "one bad scalar" hypothesis. The next plausible
failure source is stair topology itself: the current `HumanoidTerrain` stair branch still uses the
same pyramid helper and therefore never tested whether the collapse comes from immediate whole-tile
stair exposure rather than raw step amplitude.

## Required Local Patch

Check whether the local upstream checkout already contains the structured-stairs patch:

```bash
python scripts/baseline/patch_humanoid_gym_structured_stairs_env.py --check
```

Apply it once if needed:

```bash
python scripts/baseline/patch_humanoid_gym_structured_stairs_env.py
```

The method configs fail fast if the local patch is missing, so the sweep will not silently run with
the old pyramid-stairs topology.

## Selected Checkpoints

| Method | Seed 11 | Seed 17 | Seed 23 |
| --- | ---: | ---: | ---: |
| `Vanilla PPO` raw reference | `0` | `0` | `0` |
| revised heuristic anchor | `350` | `300` | `350` |
| `SC-PPO threshold = 3.8` | `300` | `300` | `400` |

## Commands

Plan the structured-stairs redesign run:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_structured_stairs_selected_checkpoint_stress.json \
  --stage plan
```

Run a bounded smoke:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_structured_stairs_selected_checkpoint_stress.json \
  --seed 11 \
  --stage all \
  --run-suffix smoke_1ep \
  --episodes 1 \
  --eval-num-envs 8 \
  --skip-completed
```

Run the full structured-stairs redesign sweep:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_structured_stairs_selected_checkpoint_stress.json \
  --stage all \
  --skip-completed
```

## Success Criterion

The first goal remains protocol discriminability:

- at least one method should stop collapsing
- only then can descriptive metric ordering be interpreted as task-valid
- if everything still has `fall_rate = 1.0`, the repo should treat that as evidence against the
  current random-stairs family even under a topology-level redesign

## Outcome

The completed `3 seeds x 3 methods x 20 episodes` run still ended in:

- `task_validity_outcome = all_methods_collapsed`

Aggregate readout:

- `Vanilla PPO`: `fall_rate = 1.0`, `VTE = 1.3661`, `JAL2 = 123.1605`, `jitter = 0.0169`, `return = 3.1623`
- revised heuristic anchor: `fall_rate = 1.0`, `VTE = 1.5336`, `JAL2 = 245.2208`, `jitter = 0.3908`, `return = 4.8913`
- `SC-PPO 3.8`: `fall_rate = 1.0`, `VTE = 1.3362`, `JAL2 = 247.8558`, `jitter = 0.3350`, `return = 4.7687`

So the first topology-level redesign also failed to recover a task-valid evaluation regime. The
engineering result is positive: the repo now supports a reproducible structured-stairs patch,
independent redesign configs, isolated local upstream state, and a validated sweep path. The
scientific result is still negative: replacing `pyramid_stairs_terrain(...)` with a structured
forward staircase segment was not enough to move any method out of full collapse.

Relative to the two strongest pre-redesign baselines:

- versus the simpler stair-difficulty-cap line, the structured-stairs line did not improve any
  method on task validity and generally regressed descriptive metrics for the revised heuristic and
  `SC-PPO`
- versus the decoupled stair-band line, the structured topology slightly reduced `SC-PPO` `JAL2`
  but still worsened `VTE` and `return`, so it does not look like a meaningful recovery

The most defensible next reading is narrower now:

- the problem is not only stair scalar severity
- it is also not fixed by this first bounded topology rewrite
- any further exploration should move beyond "first structured staircase" scaffolding and either
  redesign terrain semantics more aggressively or define a retirement criterion for this external
  random-stairs family
