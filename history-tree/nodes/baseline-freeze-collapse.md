# Frozen Formal Compare Collapse

- **Date**: 2026-05-20
- **Type**: incident
- **Outcome**: failure
- **Tags**: formal-compare, baseline, collapse

## Timeline and Background

Issue [`#5`](https://github.com/zxsheather/SCO-humanoid/issues/5) was supposed to promote the repo from single runs to a report-grade `3-seed + checkpoint-sweep` comparison. Instead, the first frozen formal compare became the project's biggest protocol failure record.

## Technical Details

- Commit [`f7007dc`](https://github.com/zxsheather/SCO-humanoid/commit/f7007dc) introduced [docs/baselines/rough-terrain-formal-comparison.md](../../docs/baselines/rough-terrain-formal-comparison.md).
- Frozen regime:
  - `64 envs x 400 iterations`
  - `Vanilla PPO` plus three heuristic action-rate rows
- Recorded failure:
  - all twelve selected checkpoints were `0`
  - every completed baseline sweep still had `fall_rate = 1.0`
  - the historical selector was also repaired so that "collapsed" was reported honestly

## Decision Process

- The repo did not treat the collapse as "baseline is bad, move on."
- It split the problem into:
  - selector semantics
  - training budget / protocol adequacy
- That directly led to [protocol-repair-probe](../nodes/protocol-repair-probe.md) instead of a blind new heuristic weight search.

## Results and Impact

- This node is the turning point of the mainline history.
- It converted the baseline question from candidate selection into protocol diagnosis.
- It also preserved `Vanilla PPO` collapse as raw-reference evidence rather than something to "fix."
