# Rough-Terrain Formal Protocol Revision Decision

This note records the decision path that starts from the repaired-budget protocol-repair probe and
ends at the completed first revised-protocol long-budget run.

It answers three questions:

- what the `0 / 0 / 200` result actually means
- whether the repo should keep repairing the current baseline protocol in place or explicitly
  revise it
- what the completed `350 / 300 / 350` revision run now changes

## Inputs

This decision is based on:

- the completed frozen formal compare under `64 envs x 400 iterations`
- the repaired selector semantics in `evaluate_checkpoint_sweep.py`
- the completed repaired-budget probe on the previous heuristic winner under
  `512 envs x 200 iterations`

Primary references:

- [rough-terrain formal comparison](./rough-terrain-formal-comparison.md)
- [rough-terrain formal protocol repair probe](./rough-terrain-formal-protocol-repair-probe.md)

## Derived budget facts

The upstream humanoid runner uses:

- `num_steps_per_env = 60`

So the relevant training budgets are:

- frozen formal compare:
  - `64 envs x 400 iterations`
  - total env steps = `64 x 400 x 60 = 1,536,000`
- repaired-budget probe at `checkpoint 50`:
  - `512 envs x 50 iterations`
  - total env steps = `512 x 50 x 60 = 1,536,000`
- repaired-budget probe full run:
  - `512 envs x 200 iterations`
  - total env steps = `512 x 200 x 60 = 6,144,000`

This means:

- the completed frozen formal compare and the repaired-budget probe at `checkpoint 50` use the
  same total number of environment steps
- the completed repaired-budget probe full run spends `4x` the total environment steps of the
  frozen formal compare

## What the repaired-budget probe proved

The repaired-budget probe result is:

- `selected checkpoints = 0 / 0 / 200`
- `seed11 -> all_checkpoints_collapsed`
- `seed17 -> all_checkpoints_collapsed`
- `seed23 -> checkpoint 200`

This proves three things:

1. the old heuristic winner is not universally collapsed under every repaired protocol variant
2. selector repair alone was not enough, but selector semantics also were not the only problem
3. the baseline-side failure is now a seed-sensitive convergence / protocol problem, not a simple
   weight-choice problem

## What the checkpoint trajectory said

Inside the repaired-budget probe:

- at `checkpoint 50`, which already matched the frozen formal compare's full total sample budget,
  all three seeds still had `fall_rate = 1.0`
- `seed11` and `seed17` continued to improve late-task metrics such as episode return and
  sometimes velocity error, but never broke out of `fall_rate = 1.0`
- `seed23` only became task-floor eligible at `checkpoint 200`, after the run had already consumed
  `4x` the frozen formal-compare total sample budget

So the current evidence did **not** support the explanation:

- `the freeze failed only because 64 envs was too narrow`

Why not:

- a `512 envs` run at matched total steps (`checkpoint 50`) still collapsed on all three seeds

The evidence also did **not** support the explanation:

- `the old heuristic winner is already repaired once we return to 512 envs x 200 iterations`

Why not:

- only `1 / 3` seeds survived
- and the surviving seed still recorded `fall_rate = 0.75`, which was far from report-grade task
  validity

## Decision after the repaired-budget probe

The repo therefore needed to make the following explicit decision:

- retire the frozen `64 envs x 400 iterations` baseline formal-compare regime as a report-grade
  anchor protocol
- do **not** promote the repaired-budget `512 envs x 200 iterations` probe to the new report-grade
  anchor protocol
- move from `protocol repair` language to `protocol revision` language

Reason:

- the frozen regime was too weak and produced universal collapse
- the repaired-budget probe broke universal collapse, but still failed the repo's `3-seed` anchor
  standard
- so neither regime was strong enough to serve as the repo's report-grade heuristic anchor

## Minimum revision rule

From that point onward, any revised baseline protocol had to be judged by a stricter success rule:

- it must produce a non-collapsed selected checkpoint on all three seeds
- it must not rely on `checkpoint 0` for the surviving comparison row
- it must produce a task-valid heuristic anchor before the repo re-froze the rough-terrain
  three-way comparison

Until that happened:

- the heuristic baseline had to be treated as diagnostic evidence only
- the repo should not reopen bounded action-rate search as if the remaining problem were still
  candidate selection

## Completed revision outcome

The prepared first revision test is now complete:

- [rough-terrain formal protocol revision long-budget test](./rough-terrain-formal-protocol-revision-long-budget.md)
- [comparison_summary.json](../../artifacts/analysis/rough_terrain_formal_protocol_revision_long_budget/comparison_summary.json)

Outcome:

- `selected checkpoints = 350 / 300 / 350`
- all three seeds are `selected`
- none of the surviving rows relies on `checkpoint 0`
- aggregate metrics:
  - `velocity_tracking_error_mean = 0.7549 +- 0.1068`
  - `joint_acceleration_l2_mean = 119.8639 +- 2.1966`
  - `action_jitter_l2_mean = 0.2711 +- 0.0084`
  - `episode_return_mean = 100.9327 +- 11.2711`
  - `fall_rate = 0.1500 +- 0.0816`

This means the prepared revision test satisfies the stricter minimum revision rule that this note
proposed:

- all three seeds produce non-collapsed selected checkpoints
- the row does not survive through `checkpoint 0`
- the revised heuristic row is task-valid on all three seeds

## Updated decision

The repo should now make the following updated decision explicit:

- keep the frozen `64 envs x 400 iterations` baseline formal-compare regime as the historical
  failure record that triggered the `协议修复线`
- keep the repaired-budget `512 envs x 200 iterations` probe as transition evidence, not as the
  final heuristic anchor
- replace the frozen heuristic-anchor regime with the completed revised-protocol
  `512 envs x 400 iterations` long-budget heuristic row
- keep `Vanilla PPO` collapse as raw-reference evidence per `CONTEXT.md`
- re-freeze the rough-terrain `三组正式对比` around:
  - `Vanilla PPO` raw reference
  - the revised heuristic formal anchor
  - `SC-PPO 3.8`

So the repo is no longer blocked by `whether protocol revision is necessary`.
It is now blocked by `how to restate the main Isaac-side comparison boundary and how to align the
MuJoCo key two-group read with the revised anchor`.
