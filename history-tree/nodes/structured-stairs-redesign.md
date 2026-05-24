# Structured-Stairs Redesign

- **Date**: 2026-05-21
- **Type**: experiment
- **Outcome**: failure
- **Tags**: random-stairs, terrain-redesign, structured-stairs

## Timeline and Background

Issue [`#16`](https://github.com/zxsheather/SCO-humanoid/issues/16) opened only after the smaller
random-stairs repair family had already failed repeatedly. The question changed from `which scalar
next` to `whether the stair generator topology itself is wrong for this evaluation line`.

## Technical Details

- Stable public branch family:
  - `explore/random-stairs-redesign`
- First redesign commit:
  - [`483debb`](https://github.com/zxsheather/SCO-humanoid/commit/483debb) `Add structured-stairs random-stairs redesign probe`
- The redesign replaced the old pyramid-stairs topology with a bounded forward staircase segment
  that preserved:
  - a flat spawn-side runway
  - a bounded step count
  - an explicit top landing
- The branch added a dedicated upstream patch path and sweep wiring for a full
  `3 seeds x 3 methods x 20 episodes` selected-checkpoint run.
- Completed outcome:
  - `Vanilla PPO`, revised heuristic, and `SC-PPO 3.8` all still had `fall_rate = 1.0`

## Decision Process

- The repo treated this as a real terrain-generator redesign, not as another tiny protocol retune.
- That justified new patch/test infrastructure, but it still kept the existing task, checkpoint
  set, and observation contract fixed.
- Because the full run still collapsed universally, the redesign was read as operationally useful
  but scientifically negative.

## Results and Impact

- The branch showed that first-order topology cleanup was not enough to recover task-valid
  discriminability.
- It directly motivated the more aggressive [stair-gate-redesign](../nodes/stair-gate-redesign.md).
- It also narrowed the remaining random-stairs question toward terrain semantics or line
  retirement, not more local scalar edits.
