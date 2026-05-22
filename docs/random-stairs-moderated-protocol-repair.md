# Moderated Random-Stairs Protocol Repair

This note defines the first bounded implementation slice for issue `#23`:

`Moderate random-stairs stress protocol as a post-freeze terrain-side branch`

## Scope

This branch stays inside the existing `速度跟踪行走 / 复杂地形条件 / 同尺比较` frame.

It is:

- evaluation-only
- selected-checkpoint-only
- post-freeze
- terrain-protocol repair, not a new method line

It is not:

- retraining on random stairs
- a new formal candidate line
- a rewrite of the frozen `main` package

## Moderation choice

The first completed random-stairs stress test used a stairs-only harsh mix:

- `terrain.terrain_proportions = [0, 0, 0, 0, 0, 0.5, 0.5]`

That pass established a clean but too-harsh reading:

`the selected rough-terrain checkpoints do not directly survive the stairs-only random-stairs protocol`

This moderated follow-up takes exactly one bounded next step:

- keep `stair_up` and `stair_down`
- restore most columns to rough uniform terrain
- keep `measure_heights = false` so the actor observation contract stays unchanged

The moderated terrain mix is:

- `terrain.terrain_proportions = [0, 0, 0.7, 0, 0, 0.15, 0.15]`
- `terrain.num_rows = 4`
- `terrain.num_cols = 10`

Interpretation rule:

- this protocol is successful if it becomes discriminative enough to separate
  `universal collapse` from `partial survival`
- it does not need to show an `SC-PPO` win to be useful

## Candidates

The branch reuses the same selected rough-terrain checkpoints:

| Method | Seed 11 | Seed 17 | Seed 23 |
| --- | ---: | ---: | ---: |
| `Vanilla PPO` raw reference | `0` | `0` | `0` |
| revised heuristic anchor | `350` | `300` | `350` |
| `SC-PPO threshold = 3.8` | `300` | `300` | `400` |

## Commands

Plan:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_selected_checkpoint_moderated_stress.json \
  --stage plan
```

Full run:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_selected_checkpoint_moderated_stress.json \
  --stage all \
  --skip-completed
```

Bounded smoke:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_selected_checkpoint_moderated_stress.json \
  --candidate sc_ppo \
  --seed 11 \
  --stage all \
  --run-suffix smoke_1ep \
  --episodes 1 \
  --eval-num-envs 8 \
  --skip-completed
```

If this worktree does not contain the historical upstream `logs/` directories for the selected
rough-terrain checkpoints, pass `--humanoid-gym-root` and point it at a Humanoid-Gym checkout that
does contain those logs.

## Artifact roots

Per-run artifacts:

```text
artifacts/methods/random_stairs_moderated_stress/
```

Aggregate summary:

```text
artifacts/analysis/random_stairs_selected_checkpoint_moderated_stress/comparison_summary.json
```

## Status

Current branch status:

- protocol definition added
- candidate configs added
- `load_run` resolution patched so a fresh worktree can reuse historical `.external/humanoid-gym/logs/...`
  style paths together with `--humanoid-gym-root`
- full moderated evaluation completed

## Completed Result

Aggregate over seeds `11`, `17`, and `23`:

| Method | VTE | Fall rate | JAL2 | Action jitter | Return | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `Vanilla PPO` raw reference | `1.4359 +- 0.0631` | `1.0000 +- 0.0000` | `127.5007 +- 12.5139` | `0.0167 +- 0.0008` | `3.2896 +- 0.3825` | collapsed |
| revised heuristic anchor | `1.5573 +- 0.1684` | `1.0000 +- 0.0000` | `239.3173 +- 22.3620` | `0.3870 +- 0.0315` | `6.1788 +- 0.1315` | collapsed |
| `SC-PPO threshold = 3.8` | `1.3669 +- 0.0715` | `1.0000 +- 0.0000` | `250.0731 +- 13.9761` | `0.3488 +- 0.0215` | `6.3206 +- 0.8232` | collapsed |

Artifacts:

- `artifacts/analysis/random_stairs_selected_checkpoint_moderated_stress/comparison_summary.json`
- `artifacts/methods/random_stairs_moderated_stress/`

Reading:

- the single bounded moderation step does **not** break the universal-collapse reading
- all three methods still have `fall_rate = 1.0`
- so this branch does not support a task-valid moderated random-stairs method advantage
- relative to the original stairs-only stress test, the moderation changed metric magnitudes but not
  the task-validity outcome
- the result is therefore best read as:

`mixed rough/stairs terrain proportions alone were not enough to repair the random-stairs protocol into a task-valid selected-checkpoint comparison`
