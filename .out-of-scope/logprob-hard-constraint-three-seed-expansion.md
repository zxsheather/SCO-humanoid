# Logprob Hard-Constraint Three-Seed Expansion

This repo does not open canonical `11 / 17 / 23` three-seed budget for the
current `logprob-gradient hard constraint` recipe after the first `seed23`
calibration.

## Why this is out of scope

The post-freeze branch process in this repo is intentionally conservative. A
new replacement mechanism only earns three-seed budget after the initial
single-seed calibration shows two things at once:

- the train-side constrained object is numerically interpretable on its chosen
  scale
- the shared rough-terrain checkpoint sweep is not already behaviorally
  collapsed

For the current initial recipe:

- `tau = 6.0`
- `cost_aggregation = mean`
- `PID lower_bound_clamp`
- `subsample_obs = 64`
- rough terrain calibration on `seed23`

the calibration trace was finite and the train-side cost did move onto the same
rough scale as `tau`, but checkpoints `0`, `50`, and `100` all had
`fall_rate = 1.0`. That means the branch failed the repo's current entry gate
for canonical three-seed expansion.

This is a rejection of the current branch premise in the current frame, not a
universal proof that hard constraints on the logprob-gradient object can never
work. Reopening this line requires a new branch premise rather than a quiet
continuation of the same recipe.

## Prior requests

- #114 — "Run canonical three-seed logprob hard-constraint feasibility diagnostic"
