# SC-PPO Next-Step Direction

This note records the repo's currently agreed next-step direction for the
`受限优化与平滑性增强` line after the completed rough-terrain formal baseline refresh and aligned
`MuJoCo isaac_mainline` replay.

## Current direction

The repo has completed the previous closure layers:

1. `重冻结 rough-terrain 三组正式对比`
2. `把 MuJoCo关键两组终验 写成 mixed evidence`
3. `PID有限消融`
4. `SN-only 替代机制可行性诊断`
5. `#7 随机阶梯 selected-checkpoint stress test`
6. `科研交付冻结 / 仓库内科研交付包`

The repo no longer has an unfinished closure step on `main`. The current operating rule is:

`treat main as 冻结主档案分支 and open any additional work as a post-freeze branch`

This means the repo should preserve the completed reports, tracker state, artifact pointers, and
reproduction entrypoints on `main` rather than reopening the frozen package. It may still absorb
bounded backports for `冻结边界章节` updates and reusable evaluation or diagnostic infrastructure,
but it should not immediately expand into moderated stairs, `SN` recipe redesign, `ALCP`, `SysID`,
`Residual RL`, visual distillation, or `VLA` work on the frozen branch.

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

## Documentation / tracker reconciliation rule

The documentation and tracker consistency pass has one current rule:

- keep `CONTEXT.md` terminology aligned with `混合外部验证结论`
- keep `README.md`, `report.md`, `report.zh.md`, and status docs on the same claim boundary
- keep `docs/reproduction/final-research-delivery-checklist.md` as the operational reproduction
  handoff
- keep `docs/adr/0001-freeze-research-delivery-before-new-protocol-repair.md` as the freeze
  decision record
- treat `main` as `冻结主档案分支`: backport only `冻结边界章节` updates and reusable evaluation
  or diagnostic infrastructure
- close or update GitHub Issues whose original wording is now superseded by the completed evidence
  closure
- keep `PID有限消融` closed as mechanism support rather than reopening it as broad component
  attribution
- keep the current `SN-only` replacement branch closed as a negative feasibility diagnostic

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

## Completed Layer 4: SN-only replacement-mechanism diagnostic

The repo selected `SN` as a bounded `替代机制可行性诊断` and ran the reduced-budget sequence.

Completed diagnostics:

- full-actor `smoke`
- full-actor `short`
- full-actor `medium`
- hidden-layer-only `medium`
- hidden-layer-only `coeff = 2.0`
- first-hidden-only `medium`

Current decision:

- the SN switches are operational and visible in checkpoints
- every current SN-only reduced-budget run is not task-valid
- output-layer SN, coefficient tightness at `1.0`, and first-hidden-only selectivity are not enough
  to explain or fix the collapse
- do not spend `主实验三种子` or `MuJoCo关键两组终验` budget on this SN-only branch
- future SN work should be opened as a separate task-stabilized recipe, not continued as blind
  architecture toggles

Canonical notes:

- [SC-PPO SN feasibility diagnostic](./sc-ppo-sn-feasibility-diagnostic.md)
- [SC-PPO SN prototype](./sc-ppo-sn-prototype.md)

## Completed Layer 5: #7 随机阶梯

After the SN-only branch closed negative, the selected bounded follow-up is:

`#7 随机阶梯 = selected rough-terrain checkpoints under a harsher 复杂地形条件`

Scope:

- keep the primary task as `速度跟踪行走`
- treat `随机阶梯` as a `复杂地形条件`, not a new task definition
- start with evaluation-only stress testing of selected rough-terrain checkpoints
- use the same shared metrics: velocity tracking error, fall rate, joint acceleration, action
  jitter, return, and any available constraint-side diagnostics
- do not retrain, open MuJoCo, or promote a new method line until the stress-test protocol itself is
  shown to be runnable and interpretable

Historical launcher:

```bash
python scripts/baseline/run_random_stairs_stress_test.py --stage plan
```

Protocol note:

- [Random-stairs selected-checkpoint stress test](./random-stairs-selected-checkpoint-stress.md)

Completed first-pass result:

- all selected rough-terrain checkpoints collapse under the first stairs-only random-stairs protocol
- `fall_rate = 1.0` for Vanilla PPO, the revised heuristic anchor, and `SC-PPO threshold = 3.8`
- the result is direct selected-checkpoint transfer failure, not a task-valid `SC-PPO` random-stairs
  advantage
- the next useful random-stairs step is protocol repair or moderation, but it should be opened as a
  separate post-freeze issue or branch rather than folded into `main`

## Completed Stage: 科研交付冻结

The repo has completed:

`freeze the internal research delivery package`

Completed outputs:

- align `README.md`, `CONTEXT.md`, reports, status docs, and GitHub tracker wording
- create a standalone final reproduction checklist with canonical commands and artifact paths
- run only `冻结期轻量验证`: unit tests, JSON validity, path/link sanity, and `git diff --check`

Operating rule after completion:

- do not rerun Isaac training, MuJoCo replay, random-stairs evaluation, or SN diagnostics on
  `main`
- keep the frozen package stable and use the final reproduction checklist plus freeze summary as the
  canonical handoff index
- allow only bounded `main` backports: `冻结边界章节` updates and reusable evaluation or
  diagnostic infrastructure
- if new work is opened, do it as a separate post-freeze issue or branch

## Immediate non-goals on `main`

The repo should not treat the following as the immediate next step:

- reopening #7 as active work by undoing the frozen-package boundary on `main`
- reopening tiny local `SC-PPO` threshold-neighborhood promotion attempts
- reopening `PID有限消融` as a broad component-attribution matrix
- rerunning the same bounded heuristic weights under unchanged assumptions as if the failure were
  still only candidate choice
- promoting new `MuJoCo terrain` runs as if they solved the current blocker
- continuing blind `SN-only` layer or coefficient toggles under the failed reduced-budget recipe
- launching `ALCP` as the first implementation branch
- starting `SysID` or `Residual RL` work before the current baseline-side repair is done
- adding perception or `VLA` as if they were a direct continuation of the current locomotion line

## Canonical current-state sentence

When summarizing the agreed repo state, the safest compact wording is:

`当前仓库的主线证据闭环和科研交付冻结都已经完成：Isaac 粗糙平面三组正式对比围绕 Vanilla PPO raw reference、revised heuristic anchor 和 SC-PPO 3.8 收口；对齐后的 MuJoCo isaac_mainline 不支持 SC-PPO 跨引擎胜利，应写成混合外部验证结论。PID有限消融、SN-only 负向诊断和 #7 随机阶梯 selected-checkpoint transfer failure 都已经闭合。main 分支当前作为 冻结主档案分支 承载仓库内科研交付包；它只吸收 冻结边界章节 和可复用评估/诊断基础设施的有限回灌，任何 moderated stairs、task-stabilized SN recipe 或其他新方向都应作为 post-freeze branch 单独打开。`

Updated after `PID有限消融` closure:

`PID有限消融 已作为有限机制诊断闭合：matched 普通对偶上升在 threshold = 3.8 的 checkpoint 100 评估中 fall_rate = 1.0，不能作为 task-valid 平滑方案；它只支持 PID-Lagrangian正式方案 的算法选择，不扩展成全组件归因研究。`

Updated after `SN-only` diagnostic closure:

`SN-only 替代机制可行性诊断 已作为负向结果闭合：full-actor、hidden-layer-only、coeff = 2.0 和 first-hidden-only reduced-budget runs 都未恢复 task-valid 行为，因此不继续消耗 seeds 或 MuJoCo 预算。#7 随机阶梯也已闭合作为 selected-checkpoint transfer failure；未来 moderated stairs 或 task-stabilized SN recipe 应作为 post-freeze branch 单独打开，而不是回到 main 上改写冻结包。`
