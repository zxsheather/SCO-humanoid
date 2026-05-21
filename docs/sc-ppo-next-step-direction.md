# SC-PPO Next-Step Direction

This note records the repo's currently agreed next-step direction for the
`受限优化与平滑性增强` line after the completed rough-terrain formal baseline refresh and aligned
`MuJoCo isaac_mainline` replay.

## Current direction

The repo has completed the previous two closure layers:

1. `重冻结 rough-terrain 三组正式对比`
2. `把 MuJoCo关键两组终验 写成 mixed evidence`

The current next step is:

`报告 / tracker / README 口径收口，然后进入闭环后支线选择`

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
- keep `PID有限消融` closed as mechanism support rather than reopening it as broad component
  attribution

## Completed Layer 3: PID有限消融 mechanism diagnostic

Issue `#6` is now closed as a bounded mechanism diagnostic:

- matched `普通对偶上升`, `threshold = 3.8`, checkpoint `100`
- `selection_status = all_checkpoints_collapsed`
- `velocity_tracking_error_mean = 1.1646`
- `action_jitter_l2_mean = 0.1661`
- `episode_return_mean = 4.7101`
- `fall_rate = 1.0000`

Reading:

- plain dual ascent does not clear the task floor in the available matched diagnostic
- lower action jitter is not a usable headline result when the policy collapses
- `PID-Lagrangian正式方案` remains the formal SC-PPO algorithm choice
- this is not a full component-attribution study

## Next Optional Research Line

After the report/tracker closure is finished, the next decision should be made through
`闭环后支线选择`. The two most plausible branches are:

- `随机阶梯` as a stress-test / protocol-repair branch for the current result
- `SN`-based `架构级平滑优化线` as an `替代机制可行性诊断`

Do not run both by default. The repo should choose one bounded branch and keep the current
report-grade claim unchanged.

Current selected branch:

- `SN` has been selected for the next bounded `替代机制可行性诊断`
- the repeatable launcher is `scripts/baseline/run_sn_diagnostic.py`
- the first valid command is
  `python scripts/baseline/run_sn_diagnostic.py --stage all --preset smoke --skip-completed`
- this does not promote `SN` to a formal candidate line
- first runner smoke completed, but it is not task-valid:
  `selection_status = all_checkpoints_collapsed`, `fall_rate = 1.0000`
- `short` also collapsed for both SN and matched non-SN/no-smoothness control, so the next step is
  a single-seed `medium` SN diagnostic rather than a formal promotion
- `medium` SN also collapsed, so the next step is SN parameterization or training-recipe tuning, not
  more seeds or MuJoCo
- hidden-layer-only `medium` SN also collapsed, so constraining the actor output layer is not the
  decisive blocker; the next useful implementation step is coefficient or selective-layer tuning
  under the same diagnostic budget
- hidden-layer-only `coeff = 2.0` also collapsed and worsened smoothness-side metrics, so a wider
  blind coefficient sweep is not the next best use of budget; prefer a selective-layer or
  task-stabilized recipe hypothesis if this branch continues
- first-hidden-only `medium` SN also collapsed, so the current SN-only replacement-mechanism
  diagnostic is negative; do not continue with more SN-only architecture toggles

## Immediate non-goals

The repo should not treat the following as the immediate next step:

- reopening tiny local `SC-PPO` threshold-neighborhood promotion attempts
- reopening `PID有限消融` as a broad component-attribution matrix
- rerunning the same bounded heuristic weights under unchanged assumptions as if the failure were
  still only candidate choice
- promoting new `MuJoCo terrain` runs as if they solved the current blocker
- launching `ALCP` as the first implementation branch
- starting `SysID` or `Residual RL` work before the current baseline-side repair is done
- adding perception or `VLA` as if they were a direct continuation of the current locomotion line

## Canonical next-step sentence

When summarizing the agreed direction, the safest compact wording is:

`当前仓库的主线证据闭环已经完成到可报告边界：Isaac 粗糙平面三组正式对比围绕 Vanilla PPO raw reference、revised heuristic anchor 和 SC-PPO 3.8 收口；对齐后的 MuJoCo isaac_mainline 不支持 SC-PPO 跨引擎胜利，应写成混合外部验证结论。下一步不是继续跑新实验，而是完成报告、README、CONTEXT 和 GitHub Issues 的口径收口。`

Updated after `PID有限消融` closure:

`PID有限消融 已作为有限机制诊断闭合：matched 普通对偶上升在 threshold = 3.8 的 checkpoint 100 评估中 fall_rate = 1.0，不能作为 task-valid 平滑方案；它只支持 PID-Lagrangian正式方案 的算法选择，不扩展成全组件归因研究。后续新工作应通过 闭环后支线选择 在 随机阶梯 和 SN 等候选支线中选一条推进。`
