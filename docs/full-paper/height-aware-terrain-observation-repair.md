# Height-Aware Terrain Observation Repair (#131)

Status: `completed`.

## Goal

Repair the custom humanoid environment's `measure_heights` path so a
height-aware terrain retrain probe can run with internally consistent critic
observation dimensions.

## Root Cause Audit

The previous terrain-aware retrain lines intentionally kept
`terrain.measure_heights = false` because the custom environment's
height-measure path was not internally consistent.

The mismatch had two parts:

1. **Wrong tensor concatenation in `compute_observations()`.**
   In `humanoid/envs/custom/humanoid_env.py`, the `measure_heights` branch
   appended terrain heights to `self.obs_buf`, even though `self.obs_buf` had
   not yet been updated inside that function and was also the wrong semantic
   target. Heights belong on the critic-side privileged observation, not the
   actor history tensor.

2. **Critic-observation dimensions stayed frozen at the no-heights contract.**
   In `humanoid/envs/custom/humanoid_config.py`, the XBot-L custom config fixes
   `single_num_privileged_obs = 73`, which is correct only for the no-heights
   critic observation:

   - command input: `5`
   - position / velocity / action / reference-diff blocks: `4 * num_actions`
   - base velocity / attitude blocks: `9`
   - push-force / push-torque blocks: `5`
   - friction / body-mass blocks: `2`
   - stance / contact masks: `4`

   For the main morphology (`num_actions = 12`), that gives
   `25 + 4 * 12 = 73`.

   But when `terrain.measure_heights = true`, the base legged-robot terrain
   sampler adds a `17 x 11 = 187` point height grid. The expected
   single-frame critic observation therefore becomes:

   - XBot-L main morphology: `73 + 187 = 260`
   - H1 leg-only morphology: `65 + 187 = 252`

   The old implementation never propagated this expansion into:

   - `cfg.env.single_num_privileged_obs`
   - `cfg.env.num_privileged_obs`
   - `BaseTask.privileged_obs_buf`
   - `critic_history`
   - PPO critic input sizing

So the old `measure_heights` path was dimensionally inconsistent even before
any terrain-aware training claim could be evaluated.

## Local Repair

The local repair keeps the actor observation contract unchanged and fixes only
the critic-side terrain observation path:

- compute the no-heights critic dimension at runtime from `num_actions`
- add the height-grid dimension when `terrain.measure_heights = true`
- synchronize `single_num_privileged_obs` and `num_privileged_obs` before
  `BaseTask` allocates buffers
- append heights to the local single-frame critic observation rather than to
  `self.obs_buf`

This is the minimum repair needed to make a bounded height-aware terrain probe
meaningful.

## Planned Probe

After the repair, reopen the previous no-stairs terrain-aware slice with a
height-aware critic path:

- same main morphology
- `terrain.mesh_type = trimesh`
- `terrain.curriculum = true`
- `terrain.measure_heights = true`
- no stairs in the terrain mix
- first validate via a short smoke run, then run one bounded seed-23 probe

Configs:

- method config:
  `configs/methods/heuristic_smoothing_action_rate_0050_trimesh_curriculum_height_probe.json`
- sweep config:
  `configs/sweeps/main_morphology_height_aware_multi_terrain_probe.json`

Smoke validation:

- one-step smoke training with `32` envs and `1` iteration completed
  successfully;
- the actor still used `705` inputs, confirming that the actor observation
  contract stayed unchanged; and
- the critic expanded cleanly to `780` inputs, confirming the repaired
  `3 x 260` height-aware privileged-observation stack.

The bounded seed-23 formal probe then ran under the same no-stairs
`trimesh + curriculum` terrain mix with `256` train envs, `200` iterations,
`16` eval envs, and `20` evaluation episodes.

## Final Probe Outcome

The repair fixed the environment contract and allowed the height-aware probe to
run end-to-end, but it did **not** rescue the terrain-aware retrain line.

Artifacts:

- training/eval run:
  `artifacts/methods/main_morphology_height_aware_multi_terrain_probe/heuristic_smoothing_action_rate_0050_trimesh_curriculum_height_probe_seed23/`
- sweep summary:
  `artifacts/methods/main_morphology_height_aware_multi_terrain_probe/heuristic_smoothing_action_rate_0050_trimesh_curriculum_height_probe_seed23/checkpoint_sweep_summary.json`
- aggregate comparison summary:
  `artifacts/analysis/main_morphology_height_aware_multi_terrain_probe/comparison_summary.json`

Checkpoint sweep result:

- evaluated checkpoints: `{0, 100, 200}`
- selection status: `all_checkpoints_collapsed`
- all checkpoints had `fall_rate = 1.0`

Key per-checkpoint metrics:

- checkpoint `0`: return `4.266`, velocity error `1.422`, joint acceleration
  `167.875`, local sensitivity `0.354`
- checkpoint `100`: return `3.970`, velocity error `1.202`, joint acceleration
  `205.436`, local sensitivity `3.427`
- checkpoint `200`: return `4.521`, velocity error `1.342`, joint acceleration
  `196.300`, local sensitivity `3.164`

Using the same practical gate applied in nearby bounded probes:

- `selection_status != all_checkpoints_collapsed`
- `fall_rate <= 0.20`
- `velocity_tracking_error_mean <= 1.0`

the repaired height-aware terrain probe still fails decisively.

## Interpretation

This closes the narrow ambiguity that the earlier terrain-aware no-go result
might have been caused only by a broken `measure_heights` path. That ambiguity
is now removed:

- the code path was genuinely broken and is now internally consistent;
- a smoke run confirms the repaired actor/critic observation contracts; but
- after the repair, the bounded terrain-aware retrain probe still collapses at
  every evaluated checkpoint.

So the bottleneck is not just critic-observation wiring. In this bounded slice,
adding the terrain height grid to the critic does not make the no-stairs
`trimesh + curriculum` heuristic retrain line task-valid.

The sweep also preserves a weaker mechanism-side signal: across the three
collapsed checkpoints, higher local sensitivity tracked higher joint
acceleration (`pearson = 0.988`). That is diagnostically consistent with the
paper's broader sensitivity story, but because the entire slice stays at the
task floor, it is not usable as positive terrain-generalization evidence.

## Paper-Facing Boundary

This repair note remains infrastructure and protocol evidence, not a paper
result section by itself.

Its paper-facing value is narrower but still real:

- it justifies stating that the custom height-measurement path was repaired and
  explicitly re-tested;
- it sharpens the no-go boundary for terrain-aware retraining by showing that
  failure persists even after the critic-height observation bug is removed; and
- it prevents over-claiming that reviewer risk `R4` was resolved by terrain
  evidence. It was not.
