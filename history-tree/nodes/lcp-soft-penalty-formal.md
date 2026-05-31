# LCP-Style Soft Penalty Formal Baseline

- **Date**: 2026-05-29
- **Type**: experiment
- **Outcome**: success
- **Tags**: lcp-style, soft-penalty, five-seed

## Timeline and Background

After the extended-seed SC-PPO result exposed hard-constraint sensitivity, the
project added a soft Jacobian/Lipschitz regularization baseline inspired by LCP
as the closest literature-aligned policy-sensitivity comparison.

## Technical Details

- Formal result note: `docs/full-paper/lcp-soft-penalty-formal-results.md`.
- Baseline recipe and naming boundary:
  `docs/full-paper/lcp-style-baseline-recipe.md` and
  `docs/full-paper/lcp-style-fidelity-audit.md`.
- Five-seed Isaac selected aggregate:
  fall `0.000`, velocity error `0.490`, joint acceleration `117.331`, jitter
  `0.212`, return `118.420`.
- Matched five-seed MuJoCo aggregate:
  fall `0.000`, velocity error `0.406`, joint acceleration `117.425`, jitter
  `0.195`, return `-599.108`.

## Decision Process

The row was promoted because it answered the same policy-local-sensitivity
mechanism question more stably than the hard SC-PPO implementation. The project
kept the label `LCP-style` because it does not establish official LCP
code/checkpoint parity.

## Results and Impact

LCP-style soft regularization became the strongest same-task
policy-local-sensitivity row in the current paper. Matched MuJoCo replay still
keeps the revised heuristic as a strong anchor because it wins aggregate joint
acceleration and return.
