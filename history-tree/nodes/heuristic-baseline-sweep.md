# Bounded Heuristic Baseline Sweep

- **Date**: 2026-05-14
- **Type**: experiment
- **Outcome**: success
- **Tags**: baseline, action-rate, heuristic

## Timeline and Background

Issue [`#3`](https://github.com/zxsheather/SCO-humanoid/issues/3) defined the heuristic baseline not as a hand-picked comparison row, but as a bounded `Action Rate Penalty` sweep governed by the repo's "task floor first, smoothness second" rule.

## Technical Details

- Commit [`a3ed146`](https://github.com/zxsheather/SCO-humanoid/commit/a3ed146) added the bounded heuristic baseline sweep.
- The early baseline family later recorded in the repo was:
  - `action_rate = -0.0005`
  - `action_rate = -0.0020`
  - `action_rate = -0.0050`
- The selected candidate that survived early reasoning was the `-0.0050` line, which later became the only heuristic candidate carried into formal protocol repair and revision.

## Decision Process

- The key rule was not "pick the smoothest curve." It was "first clear task validity, then pick the smoothest remaining candidate."
- That rule later became central to the selector repair inside [baseline-freeze-collapse](../nodes/baseline-freeze-collapse.md) and [protocol-repair-probe](../nodes/protocol-repair-probe.md).

## Results and Impact

- This sweep created the baseline candidate the repo spent most of its protocol-debugging budget on.
- The later collapse of that candidate under the frozen formal compare was therefore meaningful: it showed that the problem was in the protocol, not that no heuristic had ever been selected.
