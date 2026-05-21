# Decoupled Stair-Band Random-Stairs Selected-Checkpoint Stress Test

This note defines the first post-`#15` terrain-generator rewrite probe.

## Scope

This line keeps the mixed rough/stairs composition:

- `terrain.terrain_proportions = [0, 0, 0.5, 0, 0, 0.25, 0.25]`

It does not keep the old stair severity rule.

Instead, it rewrites stair tiles to sample from an independent stair difficulty band:

- `SCO_HUMANOID_STAIR_DIFFICULTY_MIN = 0.0`
- `SCO_HUMANOID_STAIR_DIFFICULTY_MAX = 0.35`

So this is no longer a small protocol-retune line. It is a bounded terrain-generator rewrite probe
for the already-selected rough-terrain checkpoints.

## Why This Rewrite

The repo has already exhausted four smaller protocol-repair lines:

- composition moderation
- half-height stairs
- wide-step stairs
- stair difficulty cap

All four still ended in `all_methods_collapsed`.

Those results suggest the main problem is not one global stair scalar, but the coupling between
stair severity and the shared `difficulty ~ U(0,1)` rule. This line therefore decouples stair tiles
from that shared support and samples stair severity from its own moderate band.

`measure_heights = false` remains fixed to preserve the existing `705`-dim actor observation
contract.

## Required Local Patch

Check whether the local upstream checkout already contains the stair-band patch:

```bash
python scripts/baseline/patch_humanoid_gym_stair_difficulty_band_env.py --check
```

Apply it once if needed:

```bash
python scripts/baseline/patch_humanoid_gym_stair_difficulty_band_env.py
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

Plan the decoupled stair-band run:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_decoupled_stairband_selected_checkpoint_stress.json \
  --stage plan
```

Run the full decoupled stair-band selected-checkpoint stress test:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_decoupled_stairband_selected_checkpoint_stress.json \
  --stage all \
  --skip-completed
```

Run a bounded smoke for one selected checkpoint:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_decoupled_stairband_selected_checkpoint_stress.json \
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
artifacts/methods/random_stairs_decoupled_stairband_stress/
```

The aggregate summary will be written to:

```text
artifacts/analysis/random_stairs_decoupled_stairband_selected_checkpoint_stress/comparison_summary.json
```

## Success Criterion

The first goal is still protocol discriminability:

- at least one task-valid method should stop collapsing
- only after that can the repo read metric ordering as a task-valid comparison
- if everything still has `fall_rate = 1.0`, this line becomes evidence that the current external
  random-stairs family is not aligned with the selected rough-terrain checkpoints even after a
  bounded terrain-generation rewrite

## Outcome

The completed `3 seeds x 3 methods x 20 episodes` run still ended in:

- `task_validity_outcome = all_methods_collapsed`

Aggregate readout:

- `Vanilla PPO`: `fall_rate = 1.0`, `VTE = 1.3306`, `JAL2 = 124.5607`, `jitter = 0.0168`, `return = 3.1220`
- revised heuristic anchor: `fall_rate = 1.0`, `VTE = 1.5060`, `JAL2 = 244.7390`, `jitter = 0.3775`, `return = 4.8337`
- `SC-PPO 3.8`: `fall_rate = 1.0`, `VTE = 1.3070`, `JAL2 = 266.6001`, `jitter = 0.3427`, `return = 5.2196`

So even after decoupling stair tiles from the shared `difficulty ~ U(0,1)` rule and resampling
them from an independent moderate band `[0.0, 0.35]`, the selected rough-terrain checkpoints still
did not recover a task-valid random-stairs evaluation regime.

Compared with the earlier bounded repair lines, this rewrite probe did not create any qualitative
change:

- every method still had `fall_rate = 1.0`
- `Vanilla PPO` stayed near its previous fully-collapsed baseline
- the revised heuristic anchor lost the small descriptive improvement it had under the simpler
  stair-difficulty-cap line
- `SC-PPO 3.8` kept better descriptive `VTE` than the heuristic anchor but regressed sharply on
  `JAL2`, so it still cannot be read as a task-valid smooth-control win

At this point the repo has tried composition moderation, half-height stairs, wide-step stairs,
stair-difficulty capping, and a heavier decoupled stair-band rewrite. All five lines still ended in
full collapse. The most defensible reading is that the current external random-stairs family is
misaligned with the selected rough-terrain checkpoints, and further progress would require either a
more invasive terrain-generator redesign or retiring this external evaluation line rather than
continuing small local retunes.
