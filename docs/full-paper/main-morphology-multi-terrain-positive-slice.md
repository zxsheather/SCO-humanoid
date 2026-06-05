# Main-Morphology Multi-Terrain Positive Slice (#129)

Status: `complete`.

## Goal

Produce one bounded but task-valid multi-terrain positive result on the main
morphology that is stronger than the existing replay-only terrain diagnostics
and stronger than the mixed rough/stairs retrain no-go pilot.

## Selected Slice

Use a terrain-aware retrain probe rather than another replay-only transfer
check:

- same main morphology and same locomotion stack
- train directly on `trimesh`
- enable terrain curriculum
- keep the actor observation contract unchanged (`measure_heights = false`)
- remove stair terrain from the mix
- keep a genuine multi-terrain distribution:
  `plane / obstacles / uniform roughness / slope up / slope down`
  with proportions `[0.2, 0.2, 0.4, 0.1, 0.1, 0.0, 0.0]`

Why this slice was chosen:

- the previous stair-bearing selected-checkpoint stress and stair-bearing
  retrain pilot already failed cleanly;
- this slice is still genuinely multi-terrain, but avoids reopening the
  hardest known terrain family first;
- it tests whether the current stack can produce task-valid terrain-aware
  locomotion on the main morphology at bounded cost; and
- if it succeeds, it gives the paper one bounded positive terrain result
  without pretending to be a claim-grade benchmark.

## Task-Validity Gate

The slice is treated as task-valid only if all of the following hold for the
selected checkpoint:

- `selection_status != all_checkpoints_collapsed`
- `fall_rate <= 0.20`
- `velocity_tracking_error_mean <= 1.0`

If the heuristic gate fails these criteria, the issue closes as a bounded no-go
result and records the next highest-ROI terrain follow-up.

## Execution Plan

Candidate:

- `heuristic_trimesh_curriculum`
- config:
  `configs/methods/heuristic_smoothing_action_rate_0050_trimesh_curriculum_probe.json`

Sweep:

- `configs/sweeps/main_morphology_multi_terrain_positive_slice.json`

Budget:

- seed `23`
- `256` train envs
- `200` iterations
- `16` eval envs
- `20` evaluation episodes

## Paper-Facing Boundary

Even if this slice succeeds, it is still only supplementary terrain evidence:

- one morphology
- one seed
- one terrain-aware protocol
- heuristic-only unless later expanded

The correct positive read would be:

> A bounded terrain-aware main-morphology retrain probe on a no-stairs
> multi-terrain mix produced a task-valid selected checkpoint, showing that the
> stack can support nontrivial terrain diversity under a supplementary
> protocol.

That would reduce reviewer risk `R4`, but it would not turn the paper into a
broad multi-terrain benchmark.

## Result

The selected slice remained **task-invalid** under the bounded heuristic gate.

Artifacts:

- summary:
  `artifacts/analysis/main_morphology_multi_terrain_positive_slice/comparison_summary.json`
- checkpoint sweep:
  `artifacts/methods/main_morphology_multi_terrain_positive_slice/heuristic_smoothing_action_rate_0050_trimesh_curriculum_probe_seed23/checkpoint_sweep_summary.json`
- selected metrics:
  `artifacts/methods/main_morphology_multi_terrain_positive_slice/heuristic_smoothing_action_rate_0050_trimesh_curriculum_probe_seed23/metrics_selected.json`

Observed read:

- training was technically stable through `200` iterations;
- evaluation still reported `all_checkpoints_collapsed`;
- evaluated checkpoints `0/100/200` all had `fall_rate = 1.0`; and
- the analysis-selected checkpoint `200` therefore remains task-invalid, with
  `vel_err = 1.213`, `jnt_acc = 189.084`, `jitter = 0.2031`,
  `return = 5.309`, and `policy_local_sensitivity_cost_mean = 3.584`.

This is not a selector artifact. The checkpoint sweep records
`all_checkpoints_collapsed = true`, and every evaluated checkpoint violates the
task-validity gate.

## Interpretation

This second terrain-aware retrain probe materially sharpens the failure
boundary, but it does not provide paper-usable positive terrain evidence.

The combined read across the current terrain-aware retrain probes is:

- the earlier stair-bearing mixed rough/stairs retrain pilot failed on the
  current strongest main method;
- the new no-stairs `trimesh + curriculum` heuristic probe also failed cleanly
  on the current strongest practical baseline family; and
- under the initial `measure_heights = false` observation contract, the main
  morphology did not show a task-valid non-plane multi-terrain retrain slice
  at the bounded `256 env x 200 iter` budget.

Issue `#131` then repaired the custom height-measure critic path and reran a
bounded height-aware version of the same no-stairs terrain family; that probe
also remained `all_checkpoints_collapsed`. So this note should now be read as
the first half of a sharper boundary, not as the final terrain diagnosis by
itself.

## Reviewer-Risk Read

This result does **not** materially reduce reviewer risk `R4` at claim level.
What it does provide is a cleaner boundary:

- the repo now has two explicit terrain-aware retrain no-go notes instead of a
  pure omission; and
- the remaining gap is no longer attributable only to the original
  `measure_heights = false` observation shortcut, because the repaired
  height-aware follow-up still failed cleanly.

## Follow-Up Closure

The originally proposed follow-up has now been completed:

- Issue `#131`: `Repair height-aware terrain observation support for
  main-morphology multi-terrain retraining`
- Outcome note:
  `docs/full-paper/height-aware-terrain-observation-repair.md`

That follow-up repaired the critic-side terrain observation path and confirmed
that the bounded no-stairs terrain line still collapses even after the repair.
The terrain gap is therefore sharper than this note alone originally implied.
