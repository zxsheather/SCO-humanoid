# Rough-Terrain Formal Protocol Repair Probe

This note records the outcome of the repaired-budget tracer bullet that followed the completed
rough-terrain formal baseline refresh failure.

## Scope

The probe reruns only the previous heuristic winner under the repaired protocol components:

- candidate: `heuristic smoothing action_rate = -0.0050`
- config: `configs/methods/heuristic_smoothing_action_rate_0050_formal_protocol_repair.json`
- sweep config: `configs/sweeps/rough_terrain_formal_protocol_repair_probe.json`
- regime: `512 envs x 200 iterations x save_interval 50`
- selector: repaired `先过底线再取最平滑`
- seeds: `11 / 17 / 23`

The question is not whether another heuristic weight wins.
It is whether the previous heuristic winner survives once the collapsed `64 envs x 400 iterations`
freeze is replaced by the repo's repaired-budget probe.

## Canonical summary artifact

- [comparison_summary.json](../../artifacts/analysis/rough_terrain_formal_protocol_repair_probe/comparison_summary.json)

Selected-checkpoint aggregate over seeds `11`, `17`, and `23`:

- `selected checkpoints = 0 / 0 / 200`
- `velocity_tracking_error_mean = 1.1558 +- 0.1545`
- `joint_acceleration_l2_mean = 111.5311 +- 25.5306`
- `action_jitter_l2_mean = 0.1023 +- 0.1211`
- `episode_return_mean = 22.3952 +- 26.3617`
- `fall_rate = 0.9167 +- 0.1179`

## Per-seed read

- `seed11`:
  - `selection_status = all_checkpoints_collapsed`
  - `selected checkpoint = 0`
  - `fall_rate = 1.0`
- `seed17`:
  - `selection_status = all_checkpoints_collapsed`
  - `selected checkpoint = 0`
  - `fall_rate = 1.0`
- `seed23`:
  - `selection_status = selected`
  - `selected checkpoint = 200`
  - `fall_rate = 0.75`
  - task-floor eligible set inside the run: `checkpoint 200` only

## Reading

- The repaired-budget probe breaks the strongest universal-collapse reading from the frozen
  `64 envs x 400 iterations` regime.
- The old heuristic winner is therefore not `universally collapsed` once both the selector repair
  and the repaired `512 envs x 200 iterations` budget are applied.
- However, the probe still does **not** produce a report-grade formal heuristic anchor:
  - `2 / 3` seeds remain `all_checkpoints_collapsed`
  - the only surviving seed still has `fall_rate = 0.75`
- So issue `#5` remains open.
- The blocker has narrowed from `the whole heuristic family collapses under the frozen regime` to
  `the repaired baseline protocol still shows unresolved seed-dependent failure and does not yet
  yield a defensible 3-seed anchor`.

## Consequence

This probe rules out one weak interpretation:

- it is no longer correct to say that the previous heuristic winner fails identically under every
  repaired protocol variant

But it does **not** support the stronger closure claim:

- it is still not correct to treat the heuristic baseline as repaired, frozen, and report-grade

So the repo's next step is no longer another bounded heuristic-weight search.
It is explicit baseline-protocol diagnosis and possible revision.

## Immediate next step

The next protocol-side question is:

`why does the repaired-budget rerun rescue seed23 -> checkpoint 200 while seed11 and seed17 still collapse to checkpoint 0?`

Concretely, the repo should now:

- compare the repaired-budget seed split `0 / 0 / 200` against the previous frozen `0 / 0 / 0`
  result
- explain whether the remaining failure is dominated by:
  - seed sensitivity
  - budget sensitivity
  - or the fact that the current formal anchor rule is still too weak for report-grade use
- decide whether the rough-terrain baseline protocol must now be explicitly revised rather than
  merely repaired in place

That follow-up decision is now recorded in:

- [rough-terrain formal protocol revision decision](./rough-terrain-formal-protocol-revision-decision.md)

## Canonical artifacts

- [probe summary](../../artifacts/analysis/rough_terrain_formal_protocol_repair_probe/comparison_summary.json)
- [seed11 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_formal_protocol_repair/heuristic_smoothing_action_rate_0050_formal_protocol_repair_rough_terrain_seed11/checkpoint_sweep_summary.json)
- [seed17 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_formal_protocol_repair/heuristic_smoothing_action_rate_0050_formal_protocol_repair_rough_terrain_seed17/checkpoint_sweep_summary.json)
- [seed23 checkpoint sweep](../../artifacts/methods/heuristic_smoothing_formal_protocol_repair/heuristic_smoothing_action_rate_0050_formal_protocol_repair_rough_terrain_seed23/checkpoint_sweep_summary.json)
