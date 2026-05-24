# SN Task-Stabilized Diagnostic

- **Date**: 2026-05-22
- **Type**: experiment
- **Outcome**: failure
- **Tags**: spectral-normalization, task-stabilized, negative

## Timeline and Background

Issue [`#25`](https://github.com/zxsheather/SCO-humanoid/issues/25) reopened the SN question in a
much narrower form after the earlier [sn-feasibility-diagnostic](../nodes/sn-feasibility-diagnostic.md)
had already closed the `SN-only replacement` story negative. The new question was whether a small
actor-side SN stabilizer could ride alongside the task-valid `SC-PPO 3.8` recipe without
immediately destroying task validity.

## Technical Details

- Stable public branch family:
  - `explore/sn-task-stabilized-recipe`
- Implementation commit:
  - [`78a0817`](https://github.com/zxsheather/SCO-humanoid/commit/78a0817) `Add task-stabilized SN feasibility diagnostic`
- Branch note path on that branch:
  - `docs/sc-ppo-sn-task-stabilized-diagnostic.md`
- The bounded recipe kept the full `SC-PPO 3.8` constraint path and added only first-hidden-layer
  actor SN.
- Completed single-seed stages:
  - `smoke`
  - `short`
  - `medium`
- Closure reading from the branch note:
  - optimization stayed operational
  - the intended SN weights were really present in the checkpoint
  - the selected evaluation checkpoint still had `fall_rate = 1.0`

## Decision Process

- The repo treated this as `task-stabilized feasibility`, not as a reopened SN sweep.
- That meant no `3-seed` or `MuJoCo` promotion unless the bounded single-seed read first looked
  task-valid.
- Because the branch stayed fully collapsed even after verifying the wiring, it closed as a
  negative first-stage feasibility result.

## Results and Impact

- This branch preserved the distinction between:
  - `SN-only replacement` already closed earlier
  - `SN as a small stabilizer on top of SC-PPO 3.8` also closed here
- If SN is revisited again, it now needs a genuinely different recipe rather than another nearby
  first-hidden auxiliary addition.
