# LCP + Heuristic Hybrid Pilot (#121)

Status: `complete`.

## Goal

Test whether the current `LCP-style` soft Jacobian penalty and the revised
heuristic smoothing reward appear complementary, redundant, or conflicting under
one bounded rough-terrain pilot.

## Protocol

- config:
  `configs/methods/lcp_soft_jacobian_plus_heuristic_hybrid_pilot.json`
- sweep:
  `configs/sweeps/rough_terrain_lcp_hybrid_pilot.json`
- seed: `23`
- bounded budget: `256` train envs, `200` iterations, `16` eval envs, `20`
  episodes
- hybrid mechanism:
  `LCPPPO` with `lcp_weight = 0.002` plus the revised heuristic reward scales
  (`action_smoothness`, `dof_acc`, `base_acc`, `dof_vel`)

Artifacts:

- summary:
  `artifacts/analysis/rough_terrain_lcp_hybrid_pilot/comparison_summary.json`
- run manifest:
  `artifacts/methods/lcp_heuristic_hybrid_pilot/lcp_soft_jacobian_plus_heuristic_hybrid_pilot_rough_terrain_seed23/manifest.json`
- checkpoint sweep:
  `artifacts/methods/lcp_heuristic_hybrid_pilot/lcp_soft_jacobian_plus_heuristic_hybrid_pilot_rough_terrain_seed23/checkpoint_sweep_summary.json`

## Result

The bounded hybrid pilot does **not** show complementarity.

Observed read:

- training itself was stable through `200` iterations;
- the actor-side LCP penalty remained active
  (`lcp_gradient_penalty_mean = 4.266` at the train-side summary scale);
- evaluation remained collapsed across all evaluated checkpoints
  (`selection_status = all_checkpoints_collapsed`, selected checkpoint `0`,
  `fall_rate = 1.0`);
- the selected collapsed checkpoint showed very low action jitter (`0.017`), but
  that smoothness read is not claim-grade because task validity was lost.

## Comparison to Current Anchors

On the same seed-23 main-paper anchors:

- `LCP-style` selected checkpoint `400` is task-valid
  (`fall = 0.0`, `vel_err = 0.448`, `jitter = 0.200`, `return = 123.963`);
- the revised heuristic selected checkpoint `350` is also task-valid
  (`fall = 0.05`, `vel_err = 0.740`, `jitter = 0.281`, `return = 111.486`);
- the bounded hybrid pilot never reaches a task-valid checkpoint under its
  current budget.

That is enough to reject the most optimistic complementarity story for now.

## Interpretation

The current evidence is closer to **conflict or at least non-complementarity**
than to synergy.

Why:

- if the mechanisms were strongly complementary, the hybrid should at least
  clear the same basic task-validity gate on this bounded seed-23 pilot;
- instead, adding the heuristic smoothing terms to the LCP-style penalty did
  not rescue task validity and did not produce a promotable selected checkpoint;
- the collapsed selected checkpoint's low jitter is therefore best read as
  over-smoothed or under-trained behavior, not as a useful new main-method row.

## Paper-Facing Recommendation

Do not promote `LCP + heuristic` as a new mechanism line for the current paper.

Defensible wording:

> A bounded seed-23 hybrid pilot combining the LCP-style penalty with the
> revised heuristic smoothing terms did not produce a task-valid checkpoint
> under the current 200-iteration budget. This does not support an immediate
> complementarity claim between the two mechanisms.

Still possible, but unproven:

- a longer-budget hybrid could recover later;
- the hybrid might require coefficient retuning rather than direct addition.

Neither is supported by the current evidence, so the correct project read is to
defer expansion of the hybrid line.
