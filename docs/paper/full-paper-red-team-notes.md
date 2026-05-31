# Full-Paper Red-Team Notes (#87)

Status: `complete`.

This note records the reviewer-risk pass over the assembled LaTeX manuscript.
The goal was to tighten paper-facing claims, not to add new experimental
evidence.

## Manuscript Changes

- Reframed the strongest LCP wording as the strongest local-sensitivity row
  in the current same-task audit, not a universal method claim.
- Replaced broad "beats" phrasing with metric-specific comparisons.
- Added the SC-PPO threshold boundary: threshold 3.8 is the documented current
  hard-constraint row after bounded diagnostics, not a globally optimized
  constant.
- Added the plain-dual boundary: the matched diagnostic supports keeping
  PID-Lagrangian as the within-family formal row, but does not prove every PID
  term is independently necessary.
- Strengthened limitations around one robot, one primary terrain plus bounded
  `hfield_moderate` diagnostic, selected checkpoints, bounded local
  hyperparameter choices, five-seed descriptive statistics, sim-to-sim scope,
  and absence of hardware validation.
- Recast the conclusion from "what the paper should not claim" into a direct
  mechanism-level conclusion.

## Reviewer-Risk Coverage

- R0 / LCP stronger than SC-PPO: manuscript now states this as a same-task
  local-sensitivity result and keeps SC-PPO as hard-constraint mechanism
  evidence.
- R0b / LCP coefficient cherry-pick: manuscript reports the narrow
  0.001/0.002/0.004 diagnostic and explicitly rejects global optimality.
- R1 / SC-PPO threshold cherry-pick: manuscript now records bounded
  threshold-diagnostic evidence and its boundary.
- R2 / mixed MuJoCo evidence: manuscript keeps the control-path metric split
  and says no row dominates every metric.
- R3 / statistics: manuscript describes five seeds as descriptive uncertainty,
  not large-sample significance.
- R4 / single robot and terrain: manuscript explicitly treats
  `hfield_moderate` as bounded second-setting evidence, not a broad
  multi-terrain or multi-robot study.
- R5 / no hardware: manuscript keeps MuJoCo as sim-to-sim only.
- R8 / CPO and OmniSafe: manuscript keeps both as diagnostics/future work and
  avoids claims that external constrained RL fails.
- R10 / checkpoint dependence: manuscript reports selected-vs-final behavior
  and keeps selected checkpointing as a limitation.
- R11 / sensitivity mechanism: manuscript uses the policy perturbation audit as
  local policy-output evidence, not closed-loop causality.

## Residual Risks

- The manuscript is still a venue-neutral source; venue-specific formatting and
  page budget may require compression.
- The paper still lacks official LCP code/checkpoint parity, hardware
  validation, multi-robot evidence, a broad multi-terrain study, and broad
  hyperparameter sweeps.
- The next pass should be a human author review of section balance, title, and
  target venue fit.
