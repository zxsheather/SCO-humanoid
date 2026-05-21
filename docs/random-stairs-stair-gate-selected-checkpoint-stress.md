# Stair-Gate Random-Stairs Selected-Checkpoint Stress Test

This note defines the second, more aggressive terrain-generator redesign line tracked by issue
`#17`.

## Scope

This line keeps the same selected rough-terrain checkpoints, mixed rough/stairs composition, and
policy-input contract:

- `terrain.terrain_proportions = [0, 0, 0.5, 0, 0, 0.25, 0.25]`
- `measure_heights = false`
- `SCO_HUMANOID_STAIR_DIFFICULTY_MIN = 0.0`
- `SCO_HUMANOID_STAIR_DIFFICULTY_MAX = 0.35`

What changes is the obstacle semantics. Instead of a stair tile that climbs and then leaves the
robot on a raised plateau, this redesign turns each stair tile into a bounded stair gate:

- flat spawn-side runway
- bounded ascent stair segment
- short top platform
- bounded descent stair segment
- return to flat ground before tile exit

Runtime controls for the first stair-gate probe:

- `SCO_HUMANOID_STAIR_GATE_STEP_COUNT = 3`
- `SCO_HUMANOID_STAIR_GATE_RUNWAY_M = 1.0`
- `SCO_HUMANOID_STAIR_GATE_PLATFORM_M = 0.6`
- `SCO_HUMANOID_STAIR_GATE_EXIT_M = 1.0`

## Why This Redesign

The first structured-stairs redesign in issue `#16` still ended in `all_methods_collapsed`.

That line changed stair topology, but it still preserved one strong semantic property of the old
stair tiles: after climbing the staircase, the robot remained on a raised plateau for the remainder
of the tile. This second redesign isolates that factor by making the stair obstacle bounded and
forcing the terrain back to flat ground after the gate is crossed.

## Required Local Patch

Check whether the local upstream checkout already contains the stair-gate patch:

```bash
python scripts/baseline/patch_humanoid_gym_stair_gate_env.py --check
```

Apply it once if needed:

```bash
python scripts/baseline/patch_humanoid_gym_stair_gate_env.py
```

## Commands

Plan the stair-gate redesign run:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_stair_gate_selected_checkpoint_stress.json \
  --stage plan
```

Run a bounded smoke:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_stair_gate_selected_checkpoint_stress.json \
  --seed 11 \
  --stage all \
  --run-suffix smoke_1ep \
  --episodes 1 \
  --eval-num-envs 8 \
  --skip-completed
```

Run the full redesign sweep:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_stair_gate_selected_checkpoint_stress.json \
  --stage all \
  --skip-completed
```

## Success Criterion

The first goal remains protocol discriminability:

- at least one method should stop collapsing
- only then can descriptive metric ordering be interpreted as task-valid
- if everything still has `fall_rate = 1.0`, this becomes evidence that the random-stairs mismatch
  is deeper than both scalar severity and first-order tile-topology redesigns

## Outcome

The completed `3 seeds x 3 methods x 20 episodes` run still ended in:

- `task_validity_outcome = all_methods_collapsed`

Aggregate readout:

- `Vanilla PPO`: `fall_rate = 1.0`, `VTE = 1.3428`, `JAL2 = 122.8956`, `jitter = 0.0169`, `return = 3.1070`
- revised heuristic anchor: `fall_rate = 1.0`, `VTE = 1.5189`, `JAL2 = 247.3734`, `jitter = 0.3967`, `return = 4.9091`
- `SC-PPO 3.8`: `fall_rate = 1.0`, `VTE = 1.2742`, `JAL2 = 250.8991`, `jitter = 0.3329`, `return = 5.0934`

So the second redesign also failed to recover a task-valid evaluation regime. The bounded stair-gate
semantic change did slightly improve some descriptive metrics relative to the first structured-stairs
line:

- `Vanilla PPO` improved `VTE` and `JAL2`
- `SC-PPO 3.8` improved `VTE`, `jitter`, and `return`

But those gains still do not matter scientifically because every evaluated method remained at
`fall_rate = 1.0`.

The strongest current reading is now:

- the random-stairs mismatch is not fixed by scalar severity retunes
- it is not fixed by a first structured staircase topology
- it is also not fixed by a second bounded stair-gate semantic rewrite that returns to flat ground

At this point further progress should either:

- move to a substantially different external terrain family, or
- define an explicit retirement criterion for this random-stairs evaluation line instead of
  continuing local rewrites indefinitely
