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

`当前 Isaac 主结果已经成立，但 MuJoCo 只支持部分迁移，且 terrain 终验协议仍阻塞`

This is the preferred framing over both “`SC-PPO` has not yet beaten the heuristic baseline” and
the older threshold-neighborhood-first framing.

Reason:

- the repaired `PID` branch with `threshold = 3.8` now beats the current heuristic anchor on the
  shared metrics in a completed `3-seed, 400 iteration, checkpoint-sweep` comparison
- the repo now also has a working `MuJoCo` evaluator, so the project is no longer blocked on the
  absence of a cross-engine path
- however, the current `MuJoCo` picture is mixed:
  - on `isaac_mainline + joint_reset_noise = 0.1 + 20 episodes + 20 seconds`, `SC-PPO` shows much better
    `fall_rate`, longer survival, and better velocity tracking than the heuristic anchor
  - but on that same first-pass protocol, `SC-PPO` is still worse on
    `joint_acceleration_l2_mean` and `action_jitter_l2_mean`
  - on the current `terrain` probe, both methods collapse and `SC-PPO` remains worse overall
- the blocker has therefore shifted from `Isaac里能不能赢` to
  `MuJoCo里到底哪一部分结论已经成立、哪一部分还没有成立`

## Secondary blocker

The current secondary blocker is:

`当前主结果仍然依赖 checkpoint sweep，且邻近 threshold 并未同样稳定`

This should not be promoted to the primary blocker.

Reason:

- `threshold = 3.8` on the repaired branch is now the clear mainline candidate
- but the nearest-neighbor `threshold = 4.0` branch shows much larger seed variance and one
  degenerate `checkpoint 0` selection
- the current claim is therefore stronger than a one-off single-seed win and weaker than a fully
  settled broad-region result

## Explicit mechanism blocker

The current explicit mechanism blocker is:

`checkpoint selection 已成常规要求，而 threshold 邻域稳定性成为当前主要可信度问题`

Reason:

- the repaired `PID` path at `threshold = 3.8` now enters a positive-update regime much earlier
  than the repaired `4.2` reference
- the selected checkpoints for the `3-seed` run are `300`, `300`, and `400`, which means the
  current branch still cannot be summarized by the final checkpoint alone
- the completed `threshold = 4.0` three-seed control does not provide the same stability, so the
  next actionable question is no longer “does 4.0 also work”, but how broadly the `3.8` result can
  be trusted

## Newly clarified blocker

The current newly clarified blocker is:

`repaired PID + threshold = 3.8` 已经形成主结果，而 `4.0` 邻域对照反而强化了它的特殊性`

Reason:

- the old “lower-bound clamp only fixes logic, not behavior” statement is no longer current
- after tightening `threshold` to `3.8`, the repaired branch now shows both stronger activation and
  much better selected-checkpoint behavior
- in the completed `3-seed` batch, the selected metrics average to:
  - `velocity_tracking_error_mean = 0.6412 ± 0.0554`
  - `joint_acceleration_l2_mean = 115.9079 ± 6.9386`
  - `action_jitter_l2_mean = 0.2205 ± 0.0017`
  - `fall_rate = 0.1000 ± 0.0000`
- these numbers are materially better than the current heuristic anchor under the same shared
  metric schema
- by contrast, the completed repaired-`4.0` control gives:
  - `velocity_tracking_error_mean = 0.8635 ± 0.3367`
  - `joint_acceleration_l2_mean = 120.1226 ± 26.5838`
  - `action_jitter_l2_mean = 0.1740 ± 0.1157`
  - `fall_rate = 0.4667 ± 0.3793`
- one `4.0` seed selects `checkpoint 0`, so the nearby control is materially less stable than the
  `3.8` mainline

## Current leading hypothesis

The current leading hypothesis is:

`threshold = 3.8` is currently a genuinely better operating point than `4.0`, but not yet a fully generalized result`

Evidence status:

- this is a `当前主假设`, not a confirmed root cause
- compared with the repaired `4.2` reference, the repaired `3.8` branch activates earlier and
  remains competitive through long-budget selected checkpoints
- the completed `4.0` multi-seed control now shows that the nearby branch is materially less
  stable, so `3.8` is not just an arbitrary representative of a broad flat region
- however, the repo still lacks evidence about whether `3.8` remains good beyond the immediate
  local threshold neighborhood or under harder task settings

## First-priority remediation target

The current first-priority remediation target is:

`先冻结 MuJoCo isaac_mainline 的最小可比终验，再把 MuJoCo terrain 明确转成协议修复线`

Reason:

- the repo already has enough `Isaac` evidence to support a real mainline algorithm result
- the current `MuJoCo isaac_mainline + noise` protocol is now stable enough to support a
  `最小可比` external validation statement
- the current `MuJoCo terrain` protocol is not yet discriminative enough for the main report claim:
  both methods fail and `SC-PPO` checkpoint probes at `200`, `300`, and `400` do not rescue it
- this split is now explicit in code:
  - `terrain_mode = isaac_mainline` means “follow the Isaac mainline task semantics”
  - `terrain_mode = hfield_moderate` means “run the current repair-stage intermediate hfield”
  - `terrain_mode = hfield_stress` means “run the separate MuJoCo terrain pressure test”
- the highest-yield next step is therefore not another tiny threshold poke
- the immediate value now comes from:
  1. documenting the partial-transfer `MuJoCo isaac_mainline` result cleanly
  2. documenting `MuJoCo terrain` as a current blocker rather than as a silent failure
  3. deciding whether terrain repair should target protocol alignment or algorithm robustness first

## Next short-run success criterion

The next short-run success criterion is:

`优先确认当前 threshold = 3.8 主结果在更强验证条件下仍保持对 heuristic 的优势`

This should be checked before expecting `SC-PPO` to beat the heuristic baseline on short-budget runs.

Reason:

- the repaired branch has already cleared the old mechanism gate and the first long-budget
  behavior gate
- the nearby `4.0` branch does not provide the same directional stability, so the next gate should
  move outward rather than stay in the same tiny neighborhood
- the preferred short-run evidence is:
  - selected-checkpoint metrics remain clearly better than the heuristic anchor
  - seed-to-seed variance stays modest
  - best checkpoint does not collapse to a pathological early-stop artifact

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

### Current MuJoCo first-pass status

Confirmed fact:

`在当前 MuJoCo isaac_mainline + joint_reset_noise = 0.1 + 20 episodes + 20 seconds 协议下，SC-PPO 已经显示出更强的任务稳定性与跟踪，但还没有显示出更强的平滑性`

Evidence scope:

- backend: `MuJoCo sim2sim`
- evidence strength: `single selected checkpoint per method`
- protocol: `terrain_mode = isaac_mainline`, `joint_reset_noise = 0.1`, `20 episodes`, `20 seconds`
- comparison target: heuristic anchor `action_rate = -0.005`

Minimal key numbers:

- heuristic anchor:
  - `velocity_tracking_error_mean = 0.6811 ± 0.1113`
  - `joint_acceleration_l2_mean = 110.2715 ± 13.0420`
  - `action_jitter_l2_mean = 0.2005 ± 0.0158`
  - `fall_rate = 0.7000`
  - `episode_steps_mean = 962.9`
- `SC-PPO threshold = 3.8` selected checkpoint:
  - `velocity_tracking_error_mean = 0.6206 ± 0.0458`
  - `joint_acceleration_l2_mean = 154.4672 ± 12.0365`
  - `action_jitter_l2_mean = 0.2785 ± 0.0150`
  - `fall_rate = 0.0500`
  - `episode_steps_mean = 1954.35`

Interpretation:

- `SC-PPO` currently transfers a clear `任务稳定性` and `速度跟踪` advantage into `MuJoCo`
- but the current `行为层平滑指标` do not transfer in the same direction
- this means the repo now has a real `MuJoCo第一版结果`, but not yet a full cross-engine
  smoothness victory
- the formal report-grade artifact names for this protocol are now the
  `metrics_mujoco_isaac_mainline_20ep_20s_noise01.json` outputs rather than the older
  `metrics_mujoco_plane_20ep_20s_noise01.json` duplicates

### Current MuJoCo terrain blocker status

Confirmed fact:

`在当前 MuJoCo terrain + joint_reset_noise = 0.1 + 5 episodes + 5 seconds 探针下，两组方法都会失败，而 SC-PPO 没有显示出更强表现`

Evidence scope:

- backend: `MuJoCo sim2sim`
- evidence strength: `short probe`
- protocol: `terrain_mode = hfield_stress`, `joint_reset_noise = 0.1`, `5 episodes`, `5 seconds`
- comparison target: heuristic anchor `action_rate = -0.005`

Minimal key numbers:

- heuristic:
  - `velocity_tracking_error_mean = 1.1758 ± 0.3709`
  - `joint_acceleration_l2_mean = 225.1939 ± 119.3916`
  - `action_jitter_l2_mean = 0.2921 ± 0.0742`
  - `fall_rate = 1.0000`
  - `episode_steps_mean = 123.8`
- `SC-PPO checkpoint 300`:
  - `velocity_tracking_error_mean = 1.2795 ± 0.3210`
  - `joint_acceleration_l2_mean = 296.9754 ± 58.4883`
  - `action_jitter_l2_mean = 0.3663 ± 0.0538`
  - `fall_rate = 1.0000`
  - `episode_steps_mean = 129.0`
- additional `SC-PPO` checkpoint probes:
  - `checkpoint 200 -> fall_rate = 1.0000`
  - `checkpoint 400 -> fall_rate = 1.0000`

Interpretation:

- the current `MuJoCo terrain` issue should not be summarized as a simple selected-checkpoint miss
- this is currently a `协议阻塞` or stronger transfer-robustness blocker
- until this is repaired, `MuJoCo terrain` should not be treated as the repo's main external
  validation result

### Current MuJoCo terrain repair-stage status

Confirmed fact:

`当前新增的 MuJoCo hfield_moderate repair-stage protocol 已经产生了比 hfield_stress 更强的生存判别性，但还没有产生可接受的平滑性表现`

Evidence scope:

- backend: `MuJoCo sim2sim`
- evidence strength: `short probe`
- protocol:
  `terrain_mode = hfield_moderate`, `hfield_size_override = [50.0, 50.0, 0.06, 0.02]`,
  `joint_reset_noise = 0.1`, `5 episodes`, `5 seconds`
- comparison target: heuristic anchor `action_rate = -0.005`

Minimal key numbers:

- heuristic:
  - `velocity_tracking_error_mean = 1.2872`
  - `joint_acceleration_l2_mean = 407.8357`
  - `action_jitter_l2_mean = 0.2904`
  - `fall_rate = 1.0000`
  - `episode_steps_mean = 134.6`
- `SC-PPO checkpoint 300`:
  - `velocity_tracking_error_mean = 1.3863`
  - `joint_acceleration_l2_mean = 500.5605`
  - `action_jitter_l2_mean = 0.3388`
  - `fall_rate = 0.4000`
  - `episode_steps_mean = 345.0`

Interpretation:

- compared with `hfield_stress`, this repaired intermediate protocol is no longer a pure
  “both sides immediately collapse” condition
- however, its `joint_acceleration_l2_mean` remains too poor to support a clean terrain transfer
  claim
- the correct reading is therefore:
  `hfield_moderate` is worth keeping as the current repair-stage intermediate protocol, but not yet
  as a report-grade terrain endpoint

### Current repaired-PID mainline status

Confirmed fact:

`repaired PID + threshold = 3.8` has now produced a completed `3-seed, 400 iteration, checkpoint-sweep` result that beats the current heuristic anchor on all shared primary metrics`

Evidence scope:

- task condition: `粗糙平面`
- evidence strength: `3 seeds`
- budget: `400 iteration`
- checkpoint rule: `selected checkpoint from checkpoint_sweep_summary.json`
- comparison target: current heuristic anchor `action_rate = -0.005`

Minimal key numbers:

- selected-checkpoint aggregate over seeds `11`, `17`, `23`:
  - `velocity_tracking_error_mean = 0.6412 ± 0.0554`
  - `joint_acceleration_l2_mean = 115.9079 ± 6.9386`
  - `action_jitter_l2_mean = 0.2205 ± 0.0017`
  - `episode_return_mean = 100.2838 ± 2.7150`
  - `fall_rate = 0.1000 ± 0.0000`
- per-seed selected checkpoints:
  - `seed11 -> checkpoint 300`
  - `seed17 -> checkpoint 300`
  - `seed23 -> checkpoint 400`
- current heuristic anchor:
  - `velocity_tracking_error_mean = 1.1381`
  - `joint_acceleration_l2_mean = 140.6399`
  - `action_jitter_l2_mean = 0.2457`
  - `fall_rate = 1.0`

Interpretation:

- this is the first branch in the repo that now has nontrivial multi-seed evidence for
  `方法优于启发式`
- however, the branch still requires `checkpoint sweep + selected checkpoint` reporting and should
  not yet be compressed into “final checkpoint wins by default”

### Repaired-PID nearest-neighbor control status

Confirmed fact:

`repaired PID + threshold = 4.0` does not match the stability of the `3.8` mainline under the same `3-seed, 400 iteration, checkpoint-sweep` protocol`

Evidence scope:

- task condition: `粗糙平面`
- evidence strength: `3 seeds`
- budget: `400 iteration`
- checkpoint rule: `selected checkpoint from checkpoint_sweep_summary.json`

Minimal key numbers:

- selected-checkpoint aggregate over seeds `11`, `17`, `23`:
  - `velocity_tracking_error_mean = 0.8635 ± 0.3367`
  - `joint_acceleration_l2_mean = 120.1226 ± 26.5838`
  - `action_jitter_l2_mean = 0.1740 ± 0.1157`
  - `episode_return_mean = 65.5950 ± 43.4320`
  - `fall_rate = 0.4667 ± 0.3793`
- per-seed selected checkpoints:
  - `seed11 -> checkpoint 300`
  - `seed17 -> checkpoint 400`
  - `seed23 -> checkpoint 0`

Interpretation:

- this control does not replace the `3.8` mainline
- instead, it strengthens the reading that `3.8` is a meaningful operating point rather than an
  arbitrary interchangeable choice in a broad flat neighborhood

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

### Repaired PID tightened-threshold outcome

Confirmed fact:

`tightening threshold on the repaired PID branch is what converted the repair from a mechanism-only success into a behavior-level win`

Minimal key numbers:

- repaired `threshold = 4.2`, `200 iteration`:
  - `velocity_tracking_error_mean = 1.0897`
  - `joint_acceleration_l2_mean = 172.0236`
  - `action_jitter_l2_mean = 0.2802`
- repaired `threshold = 4.0`, selected checkpoint from `400 iteration` sweep:
  - `velocity_tracking_error_mean = 0.6101`
  - `joint_acceleration_l2_mean = 119.0045`
  - `action_jitter_l2_mean = 0.2459`
  - `fall_rate = 0.1`
- repaired `threshold = 3.8`, selected checkpoint from `400 iteration` sweep:
  - `velocity_tracking_error_mean = 0.4916`
  - `joint_acceleration_l2_mean = 128.4750`
  - `action_jitter_l2_mean = 0.2434`
  - `fall_rate = 0.05`

Interpretation:

- the current repo evidence no longer supports the older conclusion that repaired PID is “alive but
  still not competitive enough”
- the current stronger conclusion is that repaired PID becomes competitive after threshold
  tightening, with `3.8` as the current leading branch

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

1. treat repaired `PID` with `threshold = 3.8` as the current mainline result
2. keep `checkpoint sweep + selected checkpoint` as the required long-budget reporting rule
3. record repaired `threshold = 4.0` as the completed nearest-neighbor control, not as a co-equal
   winner
4. decide whether the next expansion should be broader validation or report freeze, not another
   immediate local threshold sweep

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
- the current shortest-path blocker is no longer catastrophic falling, but whether the present
  `3.8` multi-seed win survives the next credibility checks beyond the local threshold neighborhood
- this is not a `SC-PPO`-only issue in the current `200 iteration` comparison, but a problem shared by
  the current comparison group at this budget level
