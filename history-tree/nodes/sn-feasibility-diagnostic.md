# SN Feasibility Diagnostic

- **Date**: 2026-05-21
- **Type**: experiment
- **Outcome**: failure
- **Tags**: spectral-normalization, replacement-mechanism, negative

## Timeline and Background

The Spectral Normalization line starts with issue [`#11`](https://github.com/zxsheather/SCO-humanoid/issues/11) on 2026-05-18 and is then expanded by issue [`#13`](https://github.com/zxsheather/SCO-humanoid/issues/13). Its role was not to become the new mainline immediately, but to answer whether an actor-side replacement mechanism was even operational.

## Technical Details

- Prototype note: [docs/sc-ppo-sn-prototype.md](../../docs/sc-ppo-sn-prototype.md)
- Closure note: [docs/sc-ppo-sn-feasibility-diagnostic.md](../../docs/sc-ppo-sn-feasibility-diagnostic.md)
- The branch implemented:
  - full-actor SN
  - hidden-only SN
  - hidden-only `coeff = 2.0`
  - first-hidden-only SN
- Every reduced-budget preset remained task-invalid with `fall_rate = 1.0`.

## Decision Process

- The repo explicitly checked whether the failure was just a missing-wire bug:
  - SN state existed in checkpoints
  - hidden-only variants worked technically
  - coefficient loosening did not rescue the branch
- That let the repo close the line as a **negative feasibility diagnostic**, not an implementation error.

## Results and Impact

- The current SN-only path was closed.
- Future SN work was pushed into a different "task-stabilized recipe" direction rather than more blind toggles.
- The next bounded follow-up on `main` became [random-stairs-transfer-failure](../nodes/random-stairs-transfer-failure.md), not more SN retries.
