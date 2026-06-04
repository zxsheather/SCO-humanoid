# Main-Morphology Multi-Terrain Retrain Pilot (#123)

Status: `complete`.

## Goal

Run one bounded retrain-level terrain-generalization pilot on the main
morphology that is stronger than the earlier replay-only terrain checks.

## Selected Slice

Use the repaired `mixed rough/stairs` terrain mix from the earlier
main-morphology scout, but retrain directly on that setting instead of replaying
rough-terrain checkpoints.

Chosen setting:

- `terrain.mesh_type = trimesh`
- `terrain.curriculum = false`
- `terrain.measure_heights = false`
- `terrain.num_rows = 4`
- `terrain.num_cols = 8`
- `terrain.terrain_proportions = [0.0, 0.0, 0.9, 0.0, 0.0, 0.05, 0.05]`

Why this was selected:

- it stays close to the main rough-terrain distribution;
- it adds a real stair component inside Isaac training, not only replay-side
  terrain change;
- it is the strongest locally available next step after the earlier
  selected-checkpoint stress note:
  `docs/full-paper/main-morphology-robustness-scout.md`.

## Execution Read

The bounded retrain pilot was gated through the current strongest main method
first:

- method:
  `LCP-style soft penalty`
- config:
  `configs/methods/lcp_soft_jacobian_penalty_mixed_rough_stairs_retrain_pilot.json`
- seed: `23`
- budget: `256` train envs, `200` iterations, `16` eval envs, `20` episodes

Artifacts:

- checkpoint sweep:
  `artifacts/methods/main_morphology_multi_terrain_pilot/lcp_soft_jacobian_penalty_mixed_rough_stairs_retrain_pilot_seed23/checkpoint_sweep_summary.json`
- selected metrics:
  `artifacts/methods/main_morphology_multi_terrain_pilot/lcp_soft_jacobian_penalty_mixed_rough_stairs_retrain_pilot_seed23/metrics_selected.json`

## Result

The selected slice remains **task-invalid** under this bounded retrain pilot.

Observed read for the LCP gate:

- training was technically stable through `200` iterations;
- evaluation reported `all_checkpoints_collapsed`;
- evaluated checkpoints `0/100/200` all had `fall_rate = 1.0`;
- the selected checkpoint therefore fell back to checkpoint `0`, with
  `vel_err = 1.436`, `jnt_acc = 105.146`, `jitter = 0.0167`, and
  `return = 4.389`.

That is enough to stop the pilot and record a bounded no-go result. Continuing
to SC-PPO and the revised heuristic would not rescue the chosen slice into a
claim-grade three-way comparison, because the slice already failed the
task-validity gate on the current strongest main method.

## Interpretation

This does **not** mean terrain generalization is impossible for the project. It
means this specific bounded retrain slice is not yet a usable paper-facing
robustness result.

The correct read is:

- the earlier replay-only terrain stress already showed selected-checkpoint
  transfer failure on stair-bearing mixes;
- the retrain-level follow-up shows that a modest `90/5/5` mixed-terrain
  training slice with `measure_heights = false` still does not produce a
  task-valid LCP row at the current pilot budget;
- the `one terrain only` limitation is therefore still open at benchmark level,
  but now bounded by an explicit retrain no-go result rather than by omission.

## Paper-Facing Recommendation

Use this as bounded supplementary evidence only.

Defensible wording:

> A bounded retrain-level mixed rough/stairs pilot on the main morphology did
> not yield a task-valid LCP checkpoint under the current 200-iteration budget.
> The result should be read as a terrain-generalization no-go scout, not as a
> positive multi-terrain benchmark.

Do not claim:

- multi-terrain robustness has been established;
- the main methods have been fairly ranked on the mixed-terrain slice; or
- the `one terrain only` reviewer risk is fully closed.

What it does provide is a clearer boundary:

- the next terrain-generalization step would need either a longer budget, a
  different observation contract with height measurements, or explicit
  terrain-specific protocol engineering.
