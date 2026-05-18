# SC-PPO Next-Step Direction

This note records the repo's currently agreed next-step direction for the
`受限优化与平滑性增强` line after the current `threshold = 3.8` result package.

It exists to keep `improve_report.md`-style forward-looking ideas from being
mistaken for the repo's immediate execution plan.

## Current direction

The repo's next step is split into two ordered layers:

1. `主线证据闭环`
2. one bounded post-mainline `架构级平滑优化线`

This means the repo should not immediately expand into `ALCP`, `SysID`,
`Residual RL`, visual distillation, or `VLA` work.

Those items remain valid `远期方向池` entries, but they are not the current
execution target.

## Layer 1: 主线证据闭环

The repo should first freeze the current mainline claim boundary for:

- `SC-PPO threshold = 3.8`
- `PID-Lagrangian`
- `pid_integral_mode = lower_bound_clamp`
- `cost_aggregation = quantile(0.90)`

The required report-grade reading is:

- Isaac `粗糙平面` remains the current main result
- `MuJoCo isaac_mainline` remains in the main report as a bounded
  `部分迁移结论`
- `MuJoCo terrain` is explicitly separated out as a `协议修复线`

This layer is complete only when the repo's report-facing documents, artifact
citations, and claim wording are aligned on that boundary.

## Layer 2: 闭环后的第一条新线

After `主线证据闭环`, the first new branch should be:

`SN`-based `架构级平滑优化线`

Its first stage is not a mainline challenge.
Its first stage is:

`替代机制可行性诊断`

The question is:

`Can actor-side Spectral Normalization replace the current Jacobian-penalty path without expanding the task definition and while preserving interpretable evidence?`

## Required comparison rule

This `SN` branch must use `同尺比较`.

Interpretation:

- training mechanism may change
- the repo should still evaluate the branch under the existing shared metric
  schema
- current `constraint`-side evidence should remain readable through the
  existing `policy_local_sensitivity_cost_*` and `constraint_violation_rate`
  outputs wherever feasible
- behavior-level evidence remains the same:
  `joint_acceleration_l2_mean`, `action_jitter_l2_mean`,
  `velocity_tracking_error_mean`, `fall_rate`

This prevents a mechanism replacement branch from silently changing the meaning
of success.

## Immediate non-goals

The repo should not treat the following as the immediate next step:

- widening the task into compliance-sensitive manipulation
- launching `ALCP` as the first implementation branch
- starting `SysID` or `Residual RL` work before the current locomotion result is
  frozen
- adding perception or `VLA` as if they were a direct continuation of the
  current mainline
- reopening tiny local threshold-neighborhood promotion attempts as the default
  next action

## Canonical next-step sentence

When summarizing the agreed direction, the safest compact wording is:

`当前仓库的下一步不是继续扩张研究命题，而是先完成 threshold = 3.8 主结果的主线证据闭环：将 MuJoCo isaac_mainline 固定为主报告中的部分迁移结论，并将 MuJoCo terrain 明确降格为协议修复线；在此之后，第一条新分支应是基于 Spectral Normalization 的架构级平滑优化线，并先以替代机制可行性诊断的形式，在同尺比较规则下与当前 3.8 SC-PPO 诊断版做一对一比较。`
