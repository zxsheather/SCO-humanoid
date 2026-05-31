# Extended-Seed Full-Paper Pivot

- **Date**: 2026-05-28
- **Type**: pivot
- **Outcome**: mixed
- **Tags**: full-paper, extended-seeds, sc-ppo

## Timeline and Background

The full-paper branch expanded the main SC-PPO and heuristic comparison from
the earlier seed slice to five seeds. This was the point where the project
stopped treating the strongest paper path as a simple SC-PPO win.

## Technical Details

- Extended seeds: `11/17/23/29/31`.
- SC-PPO 3.8 selected checkpoints: `300/300/400/400/400`.
- Revised heuristic selected checkpoints: `350/300/350/400/400`.
- The SC-PPO mean-aggregation repair and nearby threshold diagnostics did not
  remove the seed/checkpoint sensitivity.

## Decision Process

The five-seed result was scientifically more useful than the narrower
three-seed story: it exposed where hard PID-Lagrangian enforcement was brittle
and motivated a broader comparison among hard constraints, soft policy-map
regularization, and reward shaping.

## Results and Impact

This pivot directly led to the current full-paper thesis: policy-local
sensitivity is a useful smooth-control lens, but enforcement details matter and
no single method dominates every metric.
