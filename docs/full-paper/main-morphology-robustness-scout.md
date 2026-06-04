# Main-Morphology Robustness Scout

Issue: `#118`.

## Goal

Choose one bounded robustness slice on the primary morphology that responds more
directly to the reviewer-side `one terrain only` concern than the current
replay-only diagnostics.

## Candidate Read

- `MuJoCo hfield_moderate`: already complete and useful, but still a replay-side
  terrain change rather than a new terrain distribution inside the locomotion
  environment.
- `MuJoCo hfield_stress`: too severe for claim-grade ranking; existing notes
  already treat it as a no-go pressure line.
- actuator low-pass proxy: useful for control-path robustness, not for the
  terrain generality question.
- observation-noise sweep: already useful mechanism evidence, but not terrain
  breadth.
- first random-stairs selected-checkpoint stress: directly relevant, but the
  stairs-only protocol collapsed all methods and therefore failed as a
  task-valid comparison.

## Selected Slice

Use a repaired `mixed rough/stairs` selected-checkpoint stress test on the same
H1-class main morphology.

Protocol:

- evaluation only; no retraining
- same selected rough-terrain checkpoints as the main paper rows
- same three primary methods: LCP-style soft penalty, `SC-PPO 3.8 PID`, and the
  revised heuristic
- terrain mesh: `trimesh`
- keep `measure_heights = false` so the actor observation contract remains the
  current `705` dimensions
- use terrain proportions `[0.0, 0.0, 0.9, 0.0, 0.0, 0.05, 0.05]`

Rationale:

- this keeps most terrain mass on rough ground that is close to the training
  distribution
- it introduces a real stair component rather than only replay-side simulator
  variation
- it is materially stronger than `hfield_moderate` for the `one terrain only`
  question, but less degenerate than the earlier all-stairs protocol
- it preserves the same metric schema and selected-checkpoint comparison logic

## Commands

Plan:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/mixed_rough_stairs_selected_checkpoint_stress.json \
  --stage plan
```

Smoke:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/mixed_rough_stairs_selected_checkpoint_stress.json \
  --candidate lcp \
  --candidate sc_ppo \
  --candidate heuristic_smoothing \
  --seed 11 \
  --stage all \
  --episodes 2 \
  --run-suffix smoke2ep
```

Full run:

```bash
python scripts/baseline/run_random_stairs_stress_test.py \
  --sweep-config configs/sweeps/mixed_rough_stairs_selected_checkpoint_stress.json \
  --stage all \
  --skip-completed
```

## Claim Boundary

This slice is still bounded supplementary evidence:

- same morphology only
- selected-checkpoint transfer only
- not a retrained multi-terrain benchmark
- useful only if task-valid rows survive

If all methods still collapse, the correct read is that even a moderated
stair-bearing terrain mix remains outside direct selected-checkpoint transfer
for the current rough-terrain policies.

## Smoke Outcome

Two bounded seed-11 smoke passes were completed on the three primary methods:

1. first mix: `[0.0, 0.0, 0.7, 0.0, 0.0, 0.15, 0.15]`
2. repaired mix: `[0.0, 0.0, 0.9, 0.0, 0.0, 0.05, 0.05]`

Both used:

- selected rough-terrain checkpoints only
- `2` episodes
- `16` eval environments
- `measure_heights = false`

Artifacts:

- first smoke summary:
  `artifacts/analysis/mixed_rough_stairs_selected_checkpoint_stress_smoke2ep/comparison_summary.json`
- repaired smoke summary:
  `artifacts/analysis/mixed_rough_stairs_selected_checkpoint_stress_smoke2ep_v2/comparison_summary.json`

Observed result:

- all three primary methods (`LCP-style`, `SC-PPO 3.8`, revised heuristic)
  still recorded `fall_rate = 1.0` in both smoke passes
- the repaired `90/5/5` mix did not recover task-valid selected-checkpoint
  transfer even after sharply reducing stair exposure
- descriptive metric ordering changed across the two smoke passes, which
  reinforces that collapsed rows are not stable method evidence

## Recommendation

Do not promote the mixed rough/stairs selected-checkpoint line to a full
five-seed paper-facing robustness slice in the current branch.

Current read:

- this is a useful no-go scout, not a positive robustness result
- the strongest current paper-facing main-morphology robustness evidence remains
  the completed `hfield_moderate` replay plus the observation-noise and
  actuator-path diagnostics
- any future stair-bearing terrain line should probably reopen as a different
  issue with either retraining, terrain-specific protocol engineering, or a
  clearer task-validity target, rather than spending more selected-checkpoint
  replay budget
