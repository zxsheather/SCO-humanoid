# Random-Stairs Protocol Repair Family

- **Date**: 2026-05-21
- **Type**: experiment
- **Outcome**: failure
- **Tags**: random-stairs, protocol-repair, terrain-side

## Timeline and Background

After the first [random-stairs-transfer-failure](../nodes/random-stairs-transfer-failure.md), the
repo did not immediately retire the terrain-side question. Issue
[`#15`](https://github.com/zxsheather/SCO-humanoid/issues/15) opened a bounded post-freeze
`协议修复线`, and later issue [`#23`](https://github.com/zxsheather/SCO-humanoid/issues/23)
recorded the clean moderated follow-up framing.

## Technical Details

- Stable public branch family:
  - `explore/moderated-random-stairs`
- Commits that define the line:
  - [`56fd6ab`](https://github.com/zxsheather/SCO-humanoid/commit/56fd6ab) `Add moderated random-stairs protocol repair scaffold`
  - [`986cf99`](https://github.com/zxsheather/SCO-humanoid/commit/986cf99) `Add moderated random-stairs protocol repair`
  - [`cb09d91`](https://github.com/zxsheather/SCO-humanoid/commit/cb09d91) `Add random-stairs repair and rewrite probes`
- The completed family covered five bounded repair axes:
  - composition moderation
  - half-height stairs
  - wide-step stairs
  - stair-difficulty cap
  - decoupled stair-difficulty band rewrite
- Every completed `3 seeds x 3 methods x 20 episodes` run still ended in
  `task_validity_outcome = all_methods_collapsed`.

## Decision Process

- The repo kept this family evaluation-only:
  - selected rough-terrain checkpoints stayed fixed
  - `measure_heights = false` stayed fixed
  - no retraining claim was allowed to sneak in through terrain repair
- The line was allowed to widen from simple moderation into stronger support-rule rewrites, but only
  while staying inside the same `复杂地形条件` comparison frame.
- Once five bounded repairs still collapsed, the repo treated the small-repair hypothesis as
  exhausted.

## Results and Impact

- This family closed as a negative protocol-repair diagnostic, not as a method comparison result.
- It directly motivated the more invasive [structured-stairs-redesign](../nodes/structured-stairs-redesign.md).
- It also strengthened the broader reading that the external random-stairs family may simply be
  misaligned with the selected rough-terrain checkpoints.
