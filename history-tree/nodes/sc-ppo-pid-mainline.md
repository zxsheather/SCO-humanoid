# SC-PPO PID Mainline

- **Date**: 2026-05-14
- **Type**: pivot
- **Outcome**: success
- **Tags**: sc-ppo, pid-lagrangian, constraint

## Timeline and Background

Issue [`#4`](https://github.com/zxsheather/SCO-humanoid/issues/4) is the repo's main method pivot: stop treating smoothness as a heuristic reward term and instead make local sensitivity a constraint handled by `PID-Lagrangian`.

## Technical Details

- Commit [`ce04e5e`](https://github.com/zxsheather/SCO-humanoid/commit/ce04e5e) integrated SC-PPO constraint training.
- The formal line later stabilized around:
  - `threshold = 3.8`
  - `PID-Lagrangian`
  - `pid_integral_mode = lower_bound_clamp`
  - `cost_aggregation = quantile(0.90)`
- Canonical references:
  - [README.md](../../README.md)
  - [docs/sc-ppo-current-summary.md](../../docs/sc-ppo-current-summary.md)
  - [docs/sc-ppo-report-status.md](../../docs/sc-ppo-report-status.md)

## Decision Process

- The repo explicitly chose a fully replacement line, not a "heuristic plus SC-PPO hybrid."
- That boundary matters later when architecture lines such as [output-scaling-line](../nodes/output-scaling-line.md) and [orthogonal-actor-line](../nodes/orthogonal-actor-line.md) are evaluated as replacement mechanisms rather than additive tweaks.

## Results and Impact

- This is the line that eventually supports the repo's Isaac-side `方法优于启发式` claim after baseline protocol revision.
- It also spawned two important side investigations:
  - [pid-limited-ablation](../nodes/pid-limited-ablation.md)
  - [sn-feasibility-diagnostic](../nodes/sn-feasibility-diagnostic.md)
