# Random-Stairs Transfer Failure

- **Date**: 2026-05-21
- **Type**: experiment
- **Outcome**: failure
- **Tags**: random-stairs, stress-test, transfer

## Timeline and Background

Issue [`#7`](https://github.com/zxsheather/SCO-humanoid/issues/7) asked whether already-selected rough-terrain checkpoints survive under a harsher `随机阶梯` condition. The branch is evaluation-only: no retraining, no new algorithm claim, just pressure on the chosen checkpoints.

## Technical Details

- Canonical note: [docs/random-stairs-selected-checkpoint-stress.md](../../docs/random-stairs-selected-checkpoint-stress.md)
- Initial runner commit: [`1d2c96d`](https://github.com/zxsheather/SCO-humanoid/commit/1d2c96d)
- Recorded outcome commit: [`27218dd`](https://github.com/zxsheather/SCO-humanoid/commit/27218dd)
- First completed result:
  - `Vanilla PPO`, revised heuristic, and `SC-PPO 3.8` all had `fall_rate = 1.0`
  - no method produced a task-valid stairs ranking

## Decision Process

- The repo treated this as transfer failure, not as evidence that one collapsed method was "still better" than another.
- That prevented the project from quietly upgrading a stress test into a new headline claim.

## Results and Impact

- The current stairs-only protocol was closed negative on `main`.
- Any moderated or repaired stairs protocol was deferred to separate post-freeze work rather than folded back into the frozen package.
- This node is one of the explicit prerequisites later cited in [research-delivery-freeze](../nodes/research-delivery-freeze.md).
