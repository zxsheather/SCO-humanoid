# Action-Rate Constraint Diagnostic

- **Date**: 2026-05-22
- **Type**: experiment
- **Outcome**: failure
- **Tags**: action-rate, objective-side, negative

## Timeline and Background

Issue [`#21`](https://github.com/zxsheather/SCO-humanoid/issues/21) opened immediately after
[objective-mismatch-diagnostic](../nodes/objective-mismatch-diagnostic.md). The bounded question
was whether an `动作时间变化率` constraint target could stay closer to the repo's smoothness
criterion than the existing local-sensitivity objective while still clearing the rough-terrain task
floor.

## Technical Details

- Stable public branch family:
  - `explore/action-rate-constraint-diagnostic`
- Implementation commit:
  - [`a95f7ac`](https://github.com/zxsheather/SCO-humanoid/commit/a95f7ac) `Add action-rate SC-PPO diagnostic branch`
- The branch added:
  - a new `action_rate` constraint objective inside `SC-PPO`
  - generic checkpoint/eval metric extraction for non-Jacobian constraint objectives
  - two bounded short-diagnostic configs, first at threshold `0.2`, then at recalibrated threshold
    `3.0`
- Diagnostic reading from the issue-thread closure:
  - first short run exposed obvious threshold scale mismatch and multiplier saturation
  - the recalibrated `3.0` threshold removed that saturation
  - both evaluated checkpoints still remained at `fall_rate = 1.0`
  - task and smoothness signals still degraded instead of recovering

## Decision Process

- The repo explicitly distinguished three possible readings:
  - plumbing failure
  - threshold-scale mismatch
  - mechanism still unpromising after one bounded recalibration
- After the recalibrated follow-up, the only defensible reading left was the third one.
- That closed the branch as a completed negative diagnostic rather than promoting it to a larger
  sweep.

## Results and Impact

- This became the first post-`#19` objective-side replacement line to close negative with a repaired
  scale read.
- Issue [`#23`](https://github.com/zxsheather/SCO-humanoid/issues/23) later cites it as one of the
  already-closed post-freeze branches before terrain-side moderation reopened.
- It also forms part of the non-architectural replacement chain that later continued through
  [action-scaling-line](../nodes/action-scaling-line.md).
