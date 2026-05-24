# Stair-Gate Redesign

- **Date**: 2026-05-21
- **Type**: experiment
- **Outcome**: failure
- **Tags**: random-stairs, terrain-redesign, stair-gate

## Timeline and Background

Issue [`#17`](https://github.com/zxsheather/SCO-humanoid/issues/17) opened after the first
structured-stairs redesign still ended in universal collapse. The next hypothesis was that the
remaining failure source might be the raised plateau semantics after stair ascent.

## Technical Details

- Stable public branch family:
  - `explore/random-stairs-redesign`
- Second redesign commit:
  - [`927e618`](https://github.com/zxsheather/SCO-humanoid/commit/927e618) `Add stair-gate random-stairs redesign probe`
- The stair tile semantics changed again:
  - flat runway before the obstacle
  - bounded ascent segment
  - short top platform
  - bounded descent segment
  - return to flat ground before tile exit
- Completed full run:
  - `3 seeds x 3 methods x 20 episodes`
  - all three selected methods still remained at `fall_rate = 1.0`

## Decision Process

- This line intentionally changed semantics more aggressively than the first redesign while still
  preserving the same selected-checkpoint comparison contract.
- The repo allowed descriptive metric improvements to be noted, but not promoted, because the task
  floor never recovered.

## Results and Impact

- This second redesign also closed negative.
- Together with the earlier repair family and the first redesign, it supports a much narrower
  reading:
  - scalar severity retunes were insufficient
  - a first topology rewrite was insufficient
  - a bounded return-to-flat stair-gate rewrite was also insufficient
- The remaining defensible next move is a materially different terrain family or an explicit
  retirement criterion for the random-stairs line.
