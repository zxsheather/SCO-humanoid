# Moderated Random-Stairs Selected-Checkpoint Stress Test

This note defines the first post-freeze `协议修复线` for issue `#15`.

## Scope

This is a bounded `复杂地形条件` protocol repair.

It keeps the primary task fixed as `速度跟踪行走` and evaluates the already-selected rough-terrain
checkpoints under a moderated random-stairs terrain mix. It is not retraining, not a new algorithm
mainline, and not a rewrite of the frozen rough-terrain claim.

## Moderation Strategy

The original stairs-only random-stairs protocol used:

- `terrain.terrain_proportions = [0, 0, 0, 0, 0, 0.5, 0.5]`

That produced direct selected-checkpoint transfer failure for every method.

The moderated protocol changes exactly one severity axis:

- `terrain.terrain_proportions = [0, 0, 0.5, 0, 0, 0.25, 0.25]`

Reading:

- half of the tiles move back to the repo's rough-terrain-compatible `uniform` bucket
- the remaining half stays stair-like through `stair_up` and `stair_down`
- `measure_heights = false` remains fixed to preserve the existing `705`-dim actor observation
  contract
- selected checkpoints remain unchanged from the frozen rough-terrain evidence

If this moderated protocol still collapses across methods, the next repair should change only one
further axis, such as stair amplitude, rather than mixing multiple new assumptions at once.

## Selected Checkpoints

| Method | Seed 11 | Seed 17 | Seed 23 |
| --- | ---: | ---: | ---: |
| `Vanilla PPO` raw reference | `0` | `0` | `0` |
| revised heuristic anchor | `350` | `300` | `350` |
| `SC-PPO threshold = 3.8` | `300` | `300` | `400` |

## Commands

Plan the moderated run:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_moderated_selected_checkpoint_stress.json \
  --stage plan
```

Run the full moderated selected-checkpoint stress test:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_moderated_selected_checkpoint_stress.json \
  --stage all \
  --skip-completed
```

Run a bounded smoke for one selected checkpoint:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/random_stairs_moderated_selected_checkpoint_stress.json \
  --candidate sc_ppo \
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
artifacts/methods/random_stairs_moderated_stress/
```

The aggregate summary will be written to:

```text
artifacts/analysis/random_stairs_moderated_selected_checkpoint_stress/comparison_summary.json
```

## Success Criterion

The first goal is not `SC-PPO` ranking.

The first goal is protocol discriminability:

- at least one of the task-valid methods should stop collapsing
- if both revised heuristic and `SC-PPO 3.8` remain task-valid, the repo can then read metric
  ordering under the shared schema
- if everything still has `fall_rate = 1.0`, this branch remains a protocol-failure record rather
  than a method-comparison result
