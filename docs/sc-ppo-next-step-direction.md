# SC-PPO Next-Step Direction

This note records the repo's currently agreed next-step direction for the
`受限优化与平滑性增强` line after the completed rough-terrain formal baseline refresh.

## Current direction

The repo's next step now has three ordered layers:

1. `修复 rough-terrain formal-compare protocol`
2. `重新冻结主线证据边界`
3. one bounded post-mainline `架构级平滑优化线`

This means the repo should not immediately expand into `ALCP`, `SysID`, `Residual RL`,
visual distillation, or `VLA` work.

## Layer 1: 修复 rough-terrain formal-compare protocol

The immediate step is no longer `MuJoCo` wording cleanup alone.
It is:

`repair the baseline-side formal-compare regime until it can produce a task-valid report-grade anchor, or explicitly revise that regime if the current freeze cannot do so`

The previously reopened heuristic family has now already been exhausted under the frozen regime:

- `action_rate = -0.0005 -> checkpoint 0 / 0 / 0`
- `action_rate = -0.0020 -> checkpoint 0 / 0 / 0`
- `action_rate = -0.0050 -> checkpoint 0 / 0 / 0`

One protocol bug has since been repaired:

- checkpoint sweep selection is no longer allowed to ignore task-validity and simply pick the
  smoothest early checkpoint

But that repair does not close `#5` by itself:

- after fast-regenerating the completed `64 envs x 400 iterations` formal-compare sweeps under the
  repaired selector, the bounded heuristic family still remains fully collapsed
- by contrast, the previous single-run `action_rate = -0.0050` artifact trained under
  `512 envs x 200 iterations` now selects `checkpoint 200` under the repaired selector

So the next tracer bullet is no longer another heuristic weight.
It is a repaired-budget rerun of the previous heuristic winner.

Required reading behind this shift:

- [rough-terrain formal comparison](./baselines/rough-terrain-formal-comparison.md)
- [SC-PPO report-grade status](./sc-ppo-report-status.md)

Execution rule:

- treat this as a `协议修复线`, not as another algorithm branch
- do not keep spending search budget inside the same bounded heuristic family until the protocol
  question is answered
- treat `Vanilla PPO` collapse as recorded raw-reference evidence, not as a repair target

Success criterion for this layer:

- the repo has an explicit repaired or revised baseline protocol that can produce a task-valid
  formal anchor
- or the repo has an explicit documented decision that the current freeze is not report-grade and
  has been replaced

Current prepared tracer bullet:

- `configs/sweeps/rough_terrain_formal_protocol_repair_probe.json`
- candidate: `action_rate = -0.0050`
- regime: `512 envs x 200 iterations x save_interval 50`
- seeds: `11 / 17 / 23`

## Layer 2: 重新冻结主线证据边界

Only after the baseline protocol is repaired should the repo return to report-facing closure work:

- freeze the rough-terrain three-way comparison wording
- keep `MuJoCo isaac_mainline` as the bounded `部分迁移` line unless new evidence changes that
- keep `MuJoCo terrain` as a protocol-repair line rather than a headline result

## Layer 3: 闭环后的第一条新线

After the baseline protocol and report boundary are repaired, the first new branch should still
be:

`SN`-based `架构级平滑优化线`

Its first stage remains:

`替代机制可行性诊断`

## Immediate non-goals

The repo should not treat the following as the immediate next step:

- reopening tiny local `SC-PPO` threshold-neighborhood promotion attempts
- rerunning the same bounded heuristic weights under unchanged assumptions as if the failure were
  only candidate choice
- promoting new `MuJoCo terrain` runs as if they solved the current blocker
- launching `ALCP` as the first implementation branch
- starting `SysID` or `Residual RL` work before the current baseline-side repair is done
- adding perception or `VLA` as if they were a direct continuation of the current locomotion line

## Canonical next-step sentence

When summarizing the agreed direction, the safest compact wording is:

`当前仓库的下一步不是继续扩张研究命题，也不是继续在现有 heuristic action-rate 家族里找一个幸存者，而是先修复粗糙平面的 formal-compare 协议：先解释为什么冻结的 baseline 侧 budget 会让整组 bounded heuristic family 全部塌到 checkpoint 0，并在必要时显式修订该协议。只有在这一步完成之后，仓库才应重新冻结主线证据边界，并在更后面再进入基于 Spectral Normalization 的架构级平滑优化线。`
