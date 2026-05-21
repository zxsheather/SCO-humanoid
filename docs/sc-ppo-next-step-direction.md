# SC-PPO Next-Step Direction

This note records the repo's currently agreed next-step direction for the
`受限优化与平滑性增强` line after the completed rough-terrain formal baseline refresh and aligned
`MuJoCo isaac_mainline` replay.

## Current direction

The repo has completed the previous two closure layers:

1. `重冻结 rough-terrain 三组正式对比`
2. `把 MuJoCo关键两组终验 写成 mixed evidence`

The current next step is:

`报告 / tracker / README 口径收口`

This means the repo should not immediately expand into `ALCP`, `SysID`, `Residual RL`,
visual distillation, or `VLA` work.

## Completed Layer 1: 重冻结 rough-terrain 三组正式对比

The protocol-revision line has now already done the repair-stage work for the heuristic row.
The sequence is:

- frozen `64 envs x 400 iterations -> 0 / 0 / 0`
- repaired-budget `512 envs x 200 iterations -> 0 / 0 / 200`
- revised long-budget `512 envs x 400 iterations -> 350 / 300 / 350`

So the repo no longer needs to ask whether protocol revision is necessary, and it no longer needs
to treat this layer as pending. The frozen claim boundary is:

`Isaac-side 三组正式对比 = Vanilla PPO raw reference + revised heuristic anchor + SC-PPO 3.8`

The historical baseline-side steps remain important context:

- the bounded heuristic family was already exhausted under the frozen regime
- one selector bug was repaired so sweeps now obey `先过底线再取最平滑`
- the repaired-budget probe justified explicit protocol revision
- the long-budget revision run produced the first surviving `3-seed` heuristic anchor again

Execution rule:

- treat `Vanilla PPO` as a raw reference rather than a promotion-gated candidate
- do not reopen bounded heuristic search
- cite selected-checkpoint artifacts rather than `manifest.json` latest-checkpoint metrics
- treat `Vanilla PPO` collapse as recorded raw-reference evidence, not as a repair target

Success criterion for this layer:

- the repo has a stable rough-terrain Isaac wording that uses the revised heuristic anchor as the
  baseline-side formal row
- the `SC-PPO vs heuristic` claim boundary and citation set are explicit

Canonical notes for this layer:

- [rough-terrain formal comparison](./baselines/rough-terrain-formal-comparison.md)
- [rough-terrain formal protocol revision decision](./baselines/rough-terrain-formal-protocol-revision-decision.md)
- [rough-terrain formal protocol revision long-budget test](./baselines/rough-terrain-formal-protocol-revision-long-budget.md)

## Completed Layer 2: 把 MuJoCo关键两组终验 写成 mixed evidence

The `MuJoCo isaac_mainline` replay is now aligned to the revised heuristic anchor:

- revised heuristic and `SC-PPO 3.8` both have `3-seed` selected-checkpoint replays
- revised heuristic is better on task stability, velocity tracking, episode length, and joint
  acceleration
- `SC-PPO 3.8` is only better on action jitter
- the report should therefore use mixed-evidence wording rather than a `部分迁移` advantage for
  `SC-PPO`
- keep `MuJoCo terrain` as a protocol-repair line rather than a headline result

## Current Layer: 报告 / tracker / README 口径收口

The immediate work is documentation and tracker consistency:

- keep `CONTEXT.md` terminology aligned with `混合外部验证结论`
- keep `README.md`, `report.md`, `report.zh.md`, and status docs on the same claim boundary
- close or update GitHub Issues whose original wording is now superseded by the completed evidence
  closure

## Next Optional Research Line

After the report/tracker closure is finished, the first new research branch should still be:

`SN`-based `架构级平滑优化线`

Its first stage remains:

`替代机制可行性诊断`

## Immediate non-goals

The repo should not treat the following as the immediate next step:

- reopening tiny local `SC-PPO` threshold-neighborhood promotion attempts
- rerunning the same bounded heuristic weights under unchanged assumptions as if the failure were
  still only candidate choice
- promoting new `MuJoCo terrain` runs as if they solved the current blocker
- launching `ALCP` as the first implementation branch
- starting `SysID` or `Residual RL` work before the current baseline-side repair is done
- adding perception or `VLA` as if they were a direct continuation of the current locomotion line

## Canonical next-step sentence

When summarizing the agreed direction, the safest compact wording is:

`当前仓库的主线证据闭环已经完成到可报告边界：Isaac 粗糙平面三组正式对比围绕 Vanilla PPO raw reference、revised heuristic anchor 和 SC-PPO 3.8 收口；对齐后的 MuJoCo isaac_mainline 不支持 SC-PPO 跨引擎胜利，应写成混合外部验证结论。下一步不是继续跑新实验，而是完成报告、README、CONTEXT 和 GitHub Issues 的口径收口。`
