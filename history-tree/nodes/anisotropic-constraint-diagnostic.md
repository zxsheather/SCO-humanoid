# Anisotropic Constraint Diagnostic

- **Date**: 2026-05-22
- **Type**: experiment
- **Outcome**: failure
- **Tags**: anisotropic-constraint, constraint-shape, negative

## Timeline and Background

Issue [`#18`](https://github.com/zxsheather/SCO-humanoid/issues/18) opened the first bounded
post-freeze branch that stayed inside the `策略局部敏感度` family while changing the constraint shape
instead of reopening scalar-threshold tuning.

## Technical Details

- Recommended branch family in the issue: `explore/anisotropic-constraint-diagnostic`
- Closing commit on that branch: `6343290` `Close anisotropic constraint diagnostic`
- The branch implemented coarse anisotropic support on top of the current SC-PPO constraint path,
  including a guard against under-reporting whole-policy sensitivity.
- The guarded proximal-only replay later cited by issue [`#19`](https://github.com/zxsheather/SCO-humanoid/issues/19) was:
  - run: `anisotropic_constraint_t055_proxonly_pospen_legacyguard_short25_seed11`
  - outcome: reduced-budget rough-terrain eval still ended at `fall_rate = 1.0`

## Decision Process

- The branch did find and repair an honesty problem: anisotropic masking could otherwise make the
  constrained update look cheaper than the legacy whole-policy scalar readout.
- But once that guard was in place, task-validity still did not recover.
- The repo therefore closed the broader “keep the constrained object, only change the constraint
  shape” direction instead of continuing local anisotropic weight tuning.

## Results and Impact

- This is the direct precursor to [objective-mismatch-diagnostic](../nodes/objective-mismatch-diagnostic.md).
- It changed the next question from `which anisotropic weights next` to `whether the current
  objective is itself misaligned with the task-validity floor`.
