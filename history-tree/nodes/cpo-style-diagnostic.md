# Local CPO-Style Diagnostic

- **Date**: 2026-05-30
- **Type**: experiment
- **Outcome**: mixed
- **Tags**: cpo, trust-region, future-work

## Timeline and Background

After OmniSafe PPO-Lag exposed the framework-interface boundary, the project
checked whether a local CPO-style optimizer could carry the actor-internal
Jacobian constraint more faithfully.

## Technical Details

- Feasibility note: `docs/full-paper/cpo-feasibility.md`.
- Autograd/HVP smoke: `docs/full-paper/cpo-autograd-hvp-smoke.md`.
- One-update smoke: `docs/full-paper/cpo-one-update-smoke.md`.
- Bounded training-loop diagnostic:
  `docs/full-paper/cpo-style-bounded-diagnostic.md`.
- Evidence decision: `docs/full-paper/cpo-evidence-decision.md`.

## Decision Process

The project required a low-cost feasibility answer before spending multi-seed
training budget. The accepted boundary was that local CPO-style plumbing is
interesting but should not become a paper baseline unless it produces
task-valid locomotion under a clearly documented algorithm path.

## Results and Impact

The tensor and one-update checks passed, and the bounded loop produced finite
accepted updates. The evaluated checkpoints collapsed, so CPO remains a
diagnostic/future-work line rather than a primary comparison row.
