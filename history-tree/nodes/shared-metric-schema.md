# Shared Metric Schema

- **Date**: 2026-05-14
- **Type**: decision
- **Outcome**: success
- **Tags**: evaluation, metrics, comparison

## Timeline and Background

Issue [`#2`](https://github.com/zxsheather/SCO-humanoid/issues/2) asked for one evaluation contract across `Vanilla PPO`, heuristic smoothing, and `SC-PPO`. This was the repo's first real anti-drift decision: methods could differ, but the readout could not.

## Technical Details

- Commit [`830fb01`](https://github.com/zxsheather/SCO-humanoid/commit/830fb01) standardized the smooth-control evaluation metrics.
- The shared schema centered the repo on reward-independent task metrics and behavior metrics:
  - `velocity_tracking_error_mean`
  - `fall_rate`
  - `joint_acceleration_l2_mean`
  - `action_jitter_l2_mean`
  - `episode_return_mean`
- Constraint-side evidence was kept comparable through the same harness; see [docs/evaluation/shared-smooth-control-metrics.md](../../docs/evaluation/shared-smooth-control-metrics.md).

## Decision Process

- The repo rejected a reward-only comparison early. That is why later protocol repair work could distinguish "task-valid anchor" from "low smoothness score but collapsed policy."
- This decision also enabled later diagnostics like [objective-mismatch-diagnostic](../nodes/objective-mismatch-diagnostic.md), because train-side constraint metrics and eval-side behavior metrics were already on one comparison surface.

## Results and Impact

- This node is the backbone of the whole project. Without it, the later baseline collapse, selector repair, MuJoCo reinterpretation, and post-freeze diagnostics would not have been comparable.
- It directly fed both [heuristic-baseline-sweep](../nodes/heuristic-baseline-sweep.md) and [sc-ppo-pid-mainline](../nodes/sc-ppo-pid-mainline.md).
