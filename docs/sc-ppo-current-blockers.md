# SC-PPO Current Blockers

This document records the current experimental blockers for `SC-PPO` on the repo's
`速度跟踪行走` task under the `粗糙平面` condition.

It is intentionally separate from `CONTEXT.md`.
These notes describe current implementation-stage and experiment-stage blockers rather than
stable domain language.

## Scope

- This is a `当前实验阻塞` document, not a glossary or ADR.
- It tracks issues that currently block the project's `方法优于启发式` claim.
- It should be updated as the training mechanism and evidence evolve.

## Primary blocker

The current primary blocker is:

`修复后的 PID 约束机制已经解锁，但行为层收益仍不足以支撑方法优势`

This is the preferred framing over both “`SC-PPO` has not yet beaten the heuristic baseline” and
the older “`SC-PPO 约束机制尚未真正发力`”.

Reason:

- “not yet beating the baseline” is still only a surface observation
- the repaired `PID` branch now shows repeated positive multiplier re-entry, so the mechanism is no
  longer fully dead
- however, the repaired `200 iteration` run still trails the best current `dual` checkpoint on the
  smoothness-sensitive metrics
- this shifts the blocker from “乘子起不来” to “乘子起来了，但起来得太晚且收益不够好”

## Secondary blocker

The current secondary blocker is:

`200 iteration 预算只能提供阶段性证据`

This should not be promoted to the primary blocker.

Reason:

- the current budget is enough to show whether the `SC-PPO` path is runnable and directionally
  promising
- the current budget is not enough to support a final `方法优于启发式` claim
- however, the stronger blocker remains the weak engagement of the constraint mechanism rather
  than the budget alone

## Explicit mechanism blocker

The current explicit mechanism blocker is:

`PID 乘子虽然已能重新抬起，但激活过晚且强度时序仍不够好`

Reason:

- in the repaired `PID` probe, `lagrange_delta` first turns positive at iteration `98`
- the multiplier then re-enters a positive regime many times and stays positive as late as
  iteration `193`
- however, that activation still starts late enough that the branch appears to pay a large
  smoothness cost after it wakes up
- this blocker is directly actionable through earlier activation pressure, starting with
  `threshold` and then `PID` gain tuning if needed

## Newly clarified blocker

The current newly clarified blocker is:

`lower-bound clamp 已修复负积分债锁死，但没有自动转化为更好的平滑指标`

Reason:

- the old `PID` blocker was a negative-integral-debt lock that swallowed late positive constraint
  error
- the repaired branch removes that failure mode and proves that repeated positive re-entry is
  possible in the current code path
- but the repaired `200 iteration` behavior is still much worse than the best `dual` checkpoint on
  `joint_acceleration_l2_mean` and `action_jitter_l2_mean`
- therefore the next question is no longer “can PID wake up at all”, but “how to make repaired PID
  wake up earlier and more usefully”

## Current leading hypothesis

The current leading hypothesis is:

`修复后的 PID 仍然进入 active regime 太晚，当前 threshold 与 PID 更新强度的组合还不对`

Evidence status:

- this is a `当前主假设`, not a confirmed root cause
- in the repaired `threshold = 4.2`, `lambda_init = 0.5`, `quantile(0.90)` branch, the first
  positive `lagrange_delta` still arrives only at iteration `98`
- once the multiplier becomes active, the branch can re-enter a positive regime repeatedly, so the
  remaining issue no longer looks like a hard logic bug
- the late-stage activation coincides with a sharp worsening in `joint_acceleration_l2_mean` and
  `action_jitter_l2_mean`
- other factors may still contribute, including the precise `PID` gain mix and the training-time
  cost summary

## First-priority remediation target

The current first-priority remediation target is:

`在 repaired PID 分支上优先让乘子更早激活，再考虑细调 PID 系数`

Reason:

- the repaired branch already proves that the multiplier can become active under the current code
  path
- the next highest-yield lever is to pull activation earlier in training without losing the repair
- the preferred tuning order is:
  1. keep `pid_integral_mode = lower_bound_clamp` fixed
  2. tighten `threshold` on the repaired branch
  3. then evaluate whether the `PID` update is too weak, too sharp, or too oscillatory

## Next short-run success criterion

The next short-run success criterion is:

`优先确认 repaired PID 能更早介入训练，并且不明显破坏平滑指标`

This should be checked before expecting `SC-PPO` to beat the heuristic baseline on short-budget runs.

Reason:

- the repaired branch has already cleared the old “multiplier completely pinned” mechanism gate
- the new short-run gate is whether earlier positive multiplier activity can coexist with an
  acceptable `joint_acceleration_l2_mean` and `action_jitter_l2_mean`
- if the branch only activates late and then overshoots behaviorally, it is still not ready for a
  stronger long-budget comparison
- the preferred short-run evidence is:
  - `lagrange_delta` turns positive earlier than the current repaired reference
  - `lagrange_multiplier` re-enters a positive regime more than once
  - `joint_acceleration_l2_mean` and `action_jitter_l2_mean` stay closer to the current `dual`
    anchor

## Validation order

The current validation order is fixed as:

1. `机制调通短实验`
2. `再进更长预算比较`

Interpretation:

- the first step is meant to validate `约束层机制证据`
- the second step is only worth running after the first step shows that the
  `PID-Lagrangian正式方案` is materially engaging the actor update
- this avoids spending long-budget runs on a constraint path that is still effectively inactive

## Documentation granularity

This document should record blockers at a concrete experimental level.

Preferred structure:

- `已确认事实`
- `当前主假设`
- `下一轮候选动作`

Rule:

- keep observed evidence separate from causal interpretation
- keep causal interpretation separate from the next tuning actions
- do not write candidate actions as if they were already confirmed conclusions

## Confirmed facts

### Current comparison status

Confirmed fact:

`在当前粗糙平面、单种子、200 iteration 预算、当前 SC-PPO 超参数设置下，SC-PPO 尚未优于已选定的强启发式基线（action_rate = -0.005）`

Evidence scope:

- task condition: `粗糙平面`
- evidence strength: single-seed
- budget: `200 iteration`
- comparison target: the currently selected heuristic winner from the bounded sweep

Minimal key numbers:

- `SC-PPO velocity_tracking_error_mean = 1.1744`
- `Heuristic winner velocity_tracking_error_mean = 1.1381`
- `SC-PPO joint_acceleration_l2_mean = 149.5817`
- `Heuristic winner joint_acceleration_l2_mean = 140.6399`
- `SC-PPO action_jitter_l2_mean = 0.2403`
- `Heuristic winner action_jitter_l2_mean = 0.2457`

Compressed interpretation:

- under the current `200 iteration` budget, `SC-PPO` is already competitive on the
  `动作抖动次级指标`, but it still trails the selected heuristic winner on both the
  `关节震荡主指标` and the `速度跟踪误差主指标`

### Current constraint engagement status

Confirmed fact:

`在当前 200 iteration run 中，lagrange_multiplier 几乎全程贴近零，仅在单次迭代短暂为正，未形成持续约束惩罚`

Evidence interpretation:

- the current constraint path is connected and logging correctly
- however, the multiplier has not yet entered a sustained active regime
- this means the present run does not yet qualify as strong evidence that the
  `PID-Lagrangian正式方案` is materially shaping the actor optimization

Minimal key numbers:

- training-side `policy_local_sensitivity_cost_mean = 5.4313`
- evaluation-side `policy_local_sensitivity_cost_mean = 5.9121`
- evaluation-side `constraint_violation_rate = 0.74`
- `lagrange_multiplier` was positive only once and otherwise remained at or effectively near `0`

### Threshold-only probe outcome

Confirmed fact:

`第一轮 threshold-only probe（5.0 与 4.5，100 iteration）未能让乘子进入持续正区间`

Evidence interpretation:

- both probes kept `lagrange_multiplier = 0.0` for the full short run
- both probes kept `constraint_penalty_loss_mean = 0.0` throughout the short run
- lowering the threshold alone was therefore insufficient to activate a sustained constraint regime

Minimal key numbers:

- `threshold = 5.0`: training-side `constraint_violation_rate = 0.0`
- `threshold = 4.5`: training-side `constraint_violation_rate = 0.015625`
- both probes: training-side `policy_local_sensitivity_cost_mean = 3.6681`
- both probes: `positive_count(lagrange_multiplier) = 0`

### Lambda-init probe outcome

Confirmed fact:

`第二轮 lambda_init probe（threshold = 4.5，lambda_init = 0.1 与 0.5，100 iteration）仍未让乘子进入持续正区间`

Evidence interpretation:

- `lambda_init = 0.1` did not produce a positive multiplier regime at any logged iteration
- `lambda_init = 0.5` only kept the multiplier positive for the first two logged iterations before
  it was driven back to zero
- this means a nonzero initial multiplier alone is still insufficient under the current
  `PID-Lagrangian正式方案`

Minimal key numbers:

- `lambda_init = 0.1`: `positive_count(lagrange_multiplier) = 0`
- `lambda_init = 0.5`: `positive_count(lagrange_multiplier) = 2`
- `lambda_init = 0.5`: `max(lagrange_multiplier) = 0.2443`
- `lambda_init = 0.5`: first logged `constraint_penalty_loss_mean = -2.0959`
- both probes still end with `lagrange_multiplier = 0.0`

### Training-versus-evaluation mismatch

Confirmed fact:

`多个 probe 中，训练侧局部敏感度统计持续低于评估侧统计，说明乘子更新所见成本可能比实际部署态更乐观`

Minimal key numbers:

- main `SC-PPO` run:
  - training-side `policy_local_sensitivity_cost_mean = 5.4313`
  - evaluation-side `policy_local_sensitivity_cost_mean = 5.9121`
- `threshold = 4.5` probe:
  - training-side `policy_local_sensitivity_cost_mean = 3.6681`
  - evaluation-side `policy_local_sensitivity_cost_mean = 4.1550`
- `threshold = 4.5, lambda_init = 0.1` probe:
  - training-side `policy_local_sensitivity_cost_mean = 3.5629`
  - evaluation-side `policy_local_sensitivity_cost_mean = 4.0115`
- `threshold = 4.5, lambda_init = 0.5` probe:
  - training-side `policy_local_sensitivity_cost_mean = 3.1580`
  - evaluation-side `policy_local_sensitivity_cost_mean = 3.7955`

Interpretation boundary:

- this does not yet prove a bug in the metric itself
- it does prove that the current multiplier update is being driven by a more optimistic statistic
  than the evaluation summary used for comparison
- that makes `cost aggregation` a higher-priority diagnostic axis than longer-budget reruns

### Cost-aggregation probe outcome

Confirmed fact:

`quantile 聚合能显著抬高训练侧更新成本，但当前 PID-Lagrangian 仍会把乘子压回零；max 聚合不足以触发约束`

Minimal key numbers:

- `threshold = 4.5, lambda_init = 0.5, quantile = 0.90`
  - final training-side `policy_local_sensitivity_cost_update = 4.5569`
  - final training-side `policy_local_sensitivity_cost_mean = 3.5052`
  - eval-side `policy_local_sensitivity_cost_mean = 3.8076`
  - `positive_count(lagrange_multiplier) = 2`
  - final `lagrange_multiplier = 0.0`
- `threshold = 4.5, lambda_init = 0.5, max aggregation`
  - final training-side `policy_local_sensitivity_cost_update = 4.3057`
  - it never exceeds `threshold = 4.5`
  - `positive_count(lagrange_multiplier) = 2`
  - final `lagrange_multiplier = 0.0`

Interpretation:

- `quantile` proves that the current training-side `mean` statistic was too weak as the sole update
  signal
- however, `quantile` alone still does not produce sustained multiplier activation
- `max` is not the preferred next branch, because it still stays below threshold at the end of the
  short run

### PID memory-lock outcome

Confirmed fact:

`在 quantile 分支中，即使 update-cost 已经晚期越过 threshold，PID 更新量仍保持为负，说明当前 PID 状态记忆会吞掉晚到的正误差信号`

Minimal key numbers:

- in the `quantile = 0.90` branch, `policy_local_sensitivity_cost_update` exceeds `threshold = 4.5`
  at iterations `96`, `97`, and `99`
- despite that, the corresponding `lagrange_delta` values remain negative
- therefore, the multiplier stays pinned at `0.0` instead of re-entering a positive regime

Interpretation boundary:

- this does not prove that `PID-Lagrangian正式方案` must be abandoned
- it does prove that the current short-run blocker is no longer just “signal too weak”
- the next branch should explicitly test whether earlier threshold crossing or a simpler dual update
  can remove this `负记忆锁死` behavior

### Dual diagnostic outcome

Confirmed fact:

`普通对偶上升` 在当前 quantile 基线上已经成功解除乘子长期贴零问题，说明当前主阻塞点已经收敛到 `PID` 更新逻辑本身`

Minimal key numbers:

- config: `threshold = 4.2`, `lambda_init = 0.5`, `cost_aggregation = quantile(0.90)`,
  `dual_lr = 0.01`
- `positive_count(lagrange_multiplier) = 21`
- `max(lagrange_multiplier) = 0.4612`
- final `lagrange_multiplier = 0.0023`
- update-cost exceeds threshold at iterations `91`, `92`, `94`, `95`, `96`, `97`
- during those late iterations, `lagrange_delta` becomes positive again instead of remaining locked
  below zero

Comparison significance:

- the matching `PID` branch under the same `threshold = 4.2` and `quantile(0.90)` base keeps
  `positive_count(lagrange_multiplier) = 2`
- the matching `PID` branch only crosses threshold once late in training and still keeps
  `lagrange_delta < 0`
- therefore the present evidence isolates the short-run blocker to the current `PID` state update,
  not to the constraint metric wiring

Current best short-run result:

- `SC-PPO dual diagnostic` currently gives the strongest `100 iteration` metric profile observed so
  far:
  - `velocity_tracking_error_mean = 0.9946`
  - `joint_acceleration_l2_mean = 93.0012`
  - `action_jitter_l2_mean = 0.1411`

Interpretation boundary:

- this is strong evidence that the constrained-training mechanism can materially help
- it is not yet permission to replace the main `PID-Lagrangian正式方案` claim in the report
- the next shortest-path step is to repair `PID` using the dual branch as a diagnostic reference

### PID no-negative-integral outcome

Confirmed fact:

`仅仅禁止负积分累计并不能修复 PID 锁死，而且会明显损伤当前短预算任务表现`

Minimal key numbers:

- config: `threshold = 4.2`, `lambda_init = 0.5`, `cost_aggregation = quantile(0.90)`,
  `integral_min = 0.0`
- `positive_count(lagrange_multiplier) = 2`
- final `lagrange_multiplier = 0.0`
- `policy_local_sensitivity_cost_update` never exceeds `threshold = 4.2`
- `velocity_tracking_error_mean = 1.4749`
- `joint_acceleration_l2_mean = 96.4734`
- `action_jitter_l2_mean = 0.1553`

Interpretation:

- removing negative integral memory alone is insufficient
- this branch does not provide evidence that the current `PID-Lagrangian正式方案` can re-enter an
  active multiplier regime under the present short-run budget
- compared with the working `dual` branch, it gives a strictly worse tradeoff on the current
  `100 iteration` evidence

### PID lower-bound-clamp repair outcome

Confirmed fact:

`lower-bound clamp 已经修复 PID 的负积分债锁死；当前分支可以反复重新进入正乘子区间，但行为收益仍然弱于最佳 dual 对照`

Minimal key numbers:

- config: `threshold = 4.2`, `lambda_init = 0.5`, `cost_aggregation = quantile(0.90)`,
  `pid_integral_mode = lower_bound_clamp`
- repaired `200 iteration` run:
  - `positive_count(lagrange_multiplier) = 51`
  - `positive_count(lagrange_delta) = 32`
  - first positive `lagrange_delta` at iteration `98`
  - positive `lagrange_multiplier` still appears at iteration `193`
- representative late-stage re-entry:
  - iteration `115`: `lagrange_delta = 0.0033`, `lagrange_multiplier = 0.0033`
  - iteration `116`: `lagrange_delta = 0.0100`, `lagrange_multiplier = 0.0133`
  - iteration `190`: `lagrange_delta = 0.0047`, `lagrange_multiplier = 0.0118`
- behavior metrics at repaired `PID 200`:
  - `velocity_tracking_error_mean = 1.0897`
  - `joint_acceleration_l2_mean = 172.0236`
  - `action_jitter_l2_mean = 0.2802`
- best current `dual` anchor at checkpoint `100` of the `200 iteration` run:
  - `velocity_tracking_error_mean = 1.0822`
  - `joint_acceleration_l2_mean = 93.1729`
  - `action_jitter_l2_mean = 0.1469`

Interpretation:

- the current `PID` branch is no longer dead on the mechanism level
- the old blocker “positive error arrives but PID still pushes downward” has been repaired
- however, the repaired branch still does not match the best current `dual` checkpoint on
  smoothness-sensitive metrics
- therefore the next shortest-path step is to keep the repair and tune for earlier, less damaging
  activation rather than reverting to the pre-repair `PID` branch

### Long-budget checkpoint-selection outcome

Confirmed fact:

`在当前 dual 长预算 run 中，最后一个 checkpoint 不是最佳停止点；如果只看最终 checkpoint，会系统性低估该分支的平滑表现`

Minimal key numbers:

- run budget: `400 iteration`
- checkpoint `100`:
  - `velocity_tracking_error_mean = 1.0822`
  - `joint_acceleration_l2_mean = 93.1729`
  - `action_jitter_l2_mean = 0.1469`
- checkpoint `200`:
  - `velocity_tracking_error_mean = 0.8880`
  - `joint_acceleration_l2_mean = 171.3763`
  - `action_jitter_l2_mean = 0.2602`
- checkpoint `300`:
  - `velocity_tracking_error_mean = 0.7713`
  - `joint_acceleration_l2_mean = 203.0927`
  - `action_jitter_l2_mean = 0.3887`
- checkpoint `400`:
  - `velocity_tracking_error_mean = 0.6626`
  - `joint_acceleration_l2_mean = 227.3379`
  - `action_jitter_l2_mean = 0.4080`

Interpretation:

- this branch improves the `速度跟踪误差主指标` late in training while steadily sacrificing the
  `关节震荡主指标` and `动作抖动次级指标`
- the correct reading of the current `400 iteration` run is therefore `late-stage tradeoff drift`,
  not “the branch is only as bad as the final checkpoint”
- the current best checkpoint on the present long-budget run is `100`, not `400`

Protocol consequence:

- every future longer-budget comparison on this branch must be reported through
  `checkpoint_sweep_summary.json`
- the selected checkpoint, not the last checkpoint alone, is the valid evidence object for the
  current long-budget comparison loop

## Next candidate actions

The next experimental actions should be recorded as:

`一个有顺序的小矩阵`

Reason:

- the current blocker is a mechanism-activation problem rather than a single obvious bug
- a one-off tuning suggestion would make it too easy to drift back into ad-hoc changes
- an ordered matrix preserves experimental discipline and makes the next validation step auditable

Preferred order:

1. adjust `threshold`
2. adjust `lambda_init`
3. if the multiplier still does not enter a sustained active regime, inspect `PID` coefficients or
   the `普通对偶上升` diagnostic branch

Diagnostic branch rule:

- `普通对偶上升` is allowed as a diagnostic branch
- it is not allowed to replace the main `PID-Lagrangian正式方案`
- its purpose is to help separate threshold or initialization problems from PID-update problems

Single-variable rule:

- the first remediation round should remain single-variable
- do not change `threshold` and `lambda_init` in the same first-round probe
- the goal of the first round is interpretability of `约束层机制证据`, not lucky multi-knob improvement

First-round variable order:

- the first remediation round should modify `threshold` first
- `lambda_init` should not be changed before the first single-variable `threshold` probe is completed
- `lambda_init` becomes relevant only after the team has observed whether a stricter or earlier-active
  constraint regime is enough to raise the multiplier into a sustained positive range

First-round threshold direction:

- the first `threshold` probe should move in the stricter direction only
- in practice, this means lowering `threshold` rather than raising it
- a looser threshold would make the current blocker harder to diagnose, because it would further
  delay the entry of the multiplier into an active regime

First-round threshold candidate set:

- the first-round candidate set is fixed as `{5.0, 4.5}`
- `5.0` is the milder stricter probe
- `4.5` is the stronger stricter probe
- the goal is to test whether the multiplier can enter a sustained active regime earlier, not to run
  a broad hyperparameter sweep

First-round short-run budget:

- the first-round `threshold` probes should use a short budget of `100 iteration`
- this budget is intended to expose early and mid-stage multiplier behavior rather than final task
  ranking
- `50 iteration` is treated as too noisy for this purpose, while `200 iteration` is treated as too
  slow for the current mechanism-tuning loop

Post-probe branching rule:

- if either `threshold = 5.0` or `threshold = 4.5` causes the `lagrange_multiplier` to enter a
  sustained positive regime early enough to matter, and if `constraint_penalty` becomes visibly
  nonzero, then the next round may consider either:
  - keeping the improved threshold and extending budget
  - or running a later single-variable `lambda_init` probe
- if both probes still leave the multiplier effectively pinned near zero, the next step should not
  be a longer-budget rerun
- in that case, the next step should move to a single-variable `lambda_init` probe and, if needed,
  the `普通对偶上升` diagnostic branch

Probe pass/fail granularity:

- the current document should keep the first-round probe outcome in qualitative form
- it should not yet hard-code numerical pass/fail thresholds for multiplier activation
- quantitative thresholds should only be frozen after the first threshold probes reveal what an
  actually active short-run constraint regime looks like in this codebase

Current branch after first-round probes:

- the threshold-only branch is currently treated as unsuccessful for mechanism activation
- the next branch is fixed as a single-variable `lambda_init` probe
- the preferred base configuration for that branch is `threshold = 4.5`, because it is closer to
  the activation boundary than `threshold = 5.0`

Current branch after lambda-init probes:

- the lambda-init-only branch is currently treated as unsuccessful for sustained mechanism
  activation
- the next branch should first validate whether `cost aggregation` is suppressing the multiplier
  signal, then move to the `普通对偶上升` diagnostic path if needed
- the preferred base configuration for that branch is `threshold = 4.5` with `lambda_init = 0.5`,
  because it produced the strongest early multiplier response so far

Current next-step order:

1. keep the repaired `PID` base fixed:
   `threshold = 4.2`, `lambda_init = 0.5`, `cost_aggregation = quantile(0.90)`,
   `pid_integral_mode = lower_bound_clamp`
2. probe a tighter `threshold` on the repaired branch before changing more than one variable
3. any longer-budget comparison on this branch must still use `checkpoint sweep + selected
   checkpoint` rather than the final checkpoint alone
4. keep the current best `dual` checkpoint as the behavior anchor, not as the immediate mainline

## No-longer-primary blockers

The following items are no longer treated as current primary blockers:

### SC-PPO code path not runnable

This is no longer a primary blocker.

Reason:

- the training smoke test has already run through the `SC-PPO` path
- the evaluation smoke test has already loaded an `SC-PPO` checkpoint successfully
- `constraint_metrics.json` and `lagrange_multiplier_trace.json` are already being exported into
  the standard artifact layout

### Heuristic baseline not yet calibrated

This is no longer a primary blocker.

Reason:

- the bounded heuristic sweep has already completed
- the current heuristic winner has already been selected as `action_rate = -0.005`
- the main comparison target for `SC-PPO` is therefore already fixed for the current phase

## Background risks

### Fall-rate floor not yet achieved

Background risk:

`当前比较中的 fall_rate 仍然偏高，最终任务守底线尚未成立`

Role:

- this is important for the eventual `任务守底线` judgment
- however, it is not the current primary blocker for the short mechanism-tuning loop
- the current shortest-path blocker remains the repaired `PID` branch's late and behaviorally costly
  engagement pattern
- this is not a `SC-PPO`-only issue in the current `200 iteration` comparison, but a problem shared by
  the current comparison group at this budget level
