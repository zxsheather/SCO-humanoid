# SC-PPO Next-Step Direction

This note records the repo's currently agreed next-step direction for the
`受限优化与平滑性增强` line after the completed rough-terrain formal baseline refresh.

## Current direction

The repo's next step now has three ordered layers:

1. `重冻结 rough-terrain 三组正式对比`
2. `把 MuJoCo关键两组终验 写成 mixed evidence`
3. one bounded post-mainline `架构级平滑优化线`

This means the repo should not immediately expand into `ALCP`, `SysID`, `Residual RL`,
visual distillation, or `VLA` work.

## Layer 1: 重冻结 rough-terrain 三组正式对比

The protocol-revision line has now already done the repair-stage work for the heuristic row.
The sequence is:

- frozen `64 envs x 400 iterations -> 0 / 0 / 0`
- repaired-budget `512 envs x 200 iterations -> 0 / 0 / 200`
- revised long-budget `512 envs x 400 iterations -> 350 / 300 / 350`

So the immediate step is no longer to ask whether protocol revision is necessary.
It is:

`freeze the Isaac-side three-way claim boundary around Vanilla PPO raw reference, the revised heuristic anchor, and SC-PPO 3.8`

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

## Layer 2: 把 MuJoCo关键两组终验 写成 mixed evidence

The `MuJoCo isaac_mainline` replay is now aligned to the revised heuristic anchor:

- revised heuristic and `SC-PPO 3.8` both have `3-seed` selected-checkpoint replays
- revised heuristic is better on task stability, velocity tracking, episode length, and joint
  acceleration
- `SC-PPO 3.8` is only better on action jitter
- the report should therefore use mixed-evidence wording rather than a `部分迁移` advantage for
  `SC-PPO`
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
  still only candidate choice
- promoting new `MuJoCo terrain` runs as if they solved the current blocker
- launching `ALCP` as the first implementation branch
- starting `SysID` or `Residual RL` work before the current baseline-side repair is done
- adding perception or `VLA` as if they were a direct continuation of the current locomotion line

## Canonical next-step sentence

When summarizing the agreed direction, the safest compact wording is:

`当前仓库的下一步不是继续扩张研究命题，也不是继续在现有 heuristic action-rate 家族里找一个幸存者。冻结 formal-compare 先给出 0 / 0 / 0，repaired-budget probe 再给出 0 / 0 / 200，而完成的 revised long-budget protocol 已经把旧 heuristic winner 修到 350 / 300 / 350。Isaac 粗糙平面三组正式对比可以围绕 Vanilla PPO raw reference、revised heuristic anchor 和 SC-PPO 3.8 重冻结；但对齐后的 MuJoCo isaac_mainline 不再支持 SC-PPO 的跨引擎优势，应写成 mixed evidence。`
