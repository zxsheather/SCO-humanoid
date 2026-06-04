# LCP Full-Batch Alignment Scout (#124)

Status: `complete`.

## Goal

Tighten the current `LCP-style` row along the sampling axis identified in
`#120` by replacing the local `subsample_obs = 64` estimate with a full-batch
penalty estimate inside the same Humanoid-Gym training stack.

## Protocol

- config:
  `configs/methods/lcp_soft_jacobian_penalty_full_batch_alignment_pilot.json`
- sweep:
  `configs/sweeps/rough_terrain_lcp_full_batch_alignment_scout.json`
- seed: `23`
- bounded budget: `256` train envs, `200` iterations, `16` eval envs, `20`
  episodes
- change under test: `algorithm.lcp.subsample_obs = 0`

Artifacts:

- summary:
  `artifacts/analysis/rough_terrain_lcp_full_batch_alignment_scout/comparison_summary.json`
- run manifest:
  `artifacts/methods/lcp_alignment_pilot/lcp_soft_jacobian_penalty_full_batch_alignment_pilot_rough_terrain_seed23/manifest.json`
- checkpoint sweep:
  `artifacts/methods/lcp_alignment_pilot/lcp_soft_jacobian_penalty_full_batch_alignment_pilot_rough_terrain_seed23/checkpoint_sweep_summary.json`

## Result

The full-batch variant is **technically feasible but not task-valid within this
bounded pilot budget**.

Observed read:

- training remained numerically stable through all `200` iterations;
- the actor-side penalty was materially stronger than the current
  `subsample_obs = 64` row:
  `lcp_gradient_penalty_mean = 4.636` at train-side summary scale;
- evaluation remained collapsed across all evaluated checkpoints
  (`selection_status = all_checkpoints_collapsed`, selected checkpoint `0`,
  `fall_rate = 1.0`);
- the selected collapsed checkpoint still showed very low action jitter
  (`0.017`), which confirms that stronger penalty sampling can suppress
  high-frequency output without preserving locomotion task validity at this
  pilot budget.

## Comparison to the Current Main LCP Row

The result does **not** justify replacing the current paper row.

Why:

- the current main `LCP-style` seed-23 run becomes task-valid only at later
  checkpoints (`300/400`), while the bounded full-batch scout stopped at
  `200` iterations;
- the scout therefore strengthens engineering feasibility, not benchmark parity;
- it shows that the full-batch estimator can be trained in this stack, but it
  does not yet show a task-valid same-scale replacement for the current
  `subsample_obs = 64` row.

In other words:

- positive evidence: the `#120` recommendation was correct that sampling detail
  is locally tighten-able;
- negative evidence: a bounded full-batch scout is not enough, by itself, to
  upgrade the manuscript from `LCP-style` to anything closer to official parity.

## Paper-Facing Recommendation

Use this scout to narrow the claim boundary further:

> The remaining LCP parity gap is no longer purely speculative at the sampling
> level. A closer-to-official full-batch penalty estimator is feasible in the
> local stack, but the bounded seed-23 pilot remained task-invalid at the
> current 200-iteration budget. This supports keeping the manuscript's
> `LCP-style` wording and retaining `subsample_obs = 64` as the current
> task-valid main row.

Do not claim:

- full-batch sampling is better than the current main LCP row;
- full-batch sampling is required for the paper's current mechanism result; or
- the scout achieved official LCP parity.

## Next Read

For the current paper, this closes the highest-ROI parity scout with a bounded
negative result:

- the local sampling gap was tested directly;
- the scout was feasible to run; and
- the task-valid manuscript row still remains the `subsample_obs = 64`
  adaptation.

If this line is reopened later, the next step should be a **matched 400-iter
or longer rerun** of the full-batch variant, rather than further paper wording
changes.
