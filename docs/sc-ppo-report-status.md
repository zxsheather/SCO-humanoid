# SC-PPO Report-Grade Status

This note records the current report-grade reading for the repo's `受限优化与平滑性增强`
direction.

Use it to answer two questions:

- what the repo can currently defend
- what limits the final report-grade claim

## Mainline reading

The current formal `SC-PPO` mainline is still:

- `threshold = 3.8`
- `PID-Lagrangian`
- `pid_integral_mode = lower_bound_clamp`
- `cost_aggregation = quantile(0.90)`

Selected-checkpoint aggregate over seeds `11`, `17`, and `23`:

- `velocity_tracking_error_mean = 0.6412 +- 0.0554`
- `joint_acceleration_l2_mean = 115.9079 +- 6.9386`
- `action_jitter_l2_mean = 0.2205 +- 0.0017`
- `episode_return_mean = 100.2838 +- 2.7150`
- `fall_rate = 0.1000 +- 0.0000`

What this currently supports:

- `SC-PPO 3.8` remains the strongest completed rough-terrain method line in the repo
- the method-side evidence for this line is already at the intended `3-seed + checkpoint-sweep`
  strength
- after the completed revised-protocol long-budget heuristic run, the repo again has a task-valid
  heuristic formal anchor at the same evidence strength
- on the shared Isaac rough-terrain metric schema, `SC-PPO 3.8` now beats that revised anchor on
  velocity tracking, fall rate, joint acceleration, and action jitter

What this does not support by itself:

- a full cross-engine smoothness-transfer claim

## Frozen baseline refresh outcome

The rough-terrain frozen baseline refresh for issue `#5` is complete across `Vanilla PPO` and the
bounded heuristic action-rate family:

- [rough-terrain formal comparison note](./baselines/rough-terrain-formal-comparison.md)
- frozen formal comparison summary:
  `artifacts/analysis/rough_terrain_formal_comparison/comparison_summary.json`

Selected-checkpoint aggregate over seeds `11`, `17`, and `23`:

- `Vanilla PPO (frozen formal compare)`:
  - `selected checkpoints = 0 / 0 / 0`
  - `velocity_tracking_error_mean = 1.3321 +- 0.1181`
  - `joint_acceleration_l2_mean = 83.7179 +- 13.3692`
  - `action_jitter_l2_mean = 0.0161 +- 0.0008`
  - `episode_return_mean = 4.0002 +- 0.4323`
  - `fall_rate = 1.0000 +- 0.0000`
- `PPO + heuristic smoothing (action_rate = -0.0005, frozen formal compare)`:
  - `selected checkpoints = 0 / 0 / 0`
  - `velocity_tracking_error_mean = 1.3451 +- 0.1269`
  - `joint_acceleration_l2_mean = 83.7119 +- 14.9052`
  - `action_jitter_l2_mean = 0.0161 +- 0.0009`
  - `episode_return_mean = 4.1998 +- 0.4037`
  - `fall_rate = 1.0000 +- 0.0000`
- `PPO + heuristic smoothing (action_rate = -0.0020, frozen formal compare)`:
  - `selected checkpoints = 0 / 0 / 0`
  - `velocity_tracking_error_mean = 1.3436 +- 0.1232`
  - `joint_acceleration_l2_mean = 85.5995 +- 13.7253`
  - `action_jitter_l2_mean = 0.0161 +- 0.0009`
  - `episode_return_mean = 4.1811 +- 0.3680`
  - `fall_rate = 1.0000 +- 0.0000`
- `PPO + heuristic smoothing (action_rate = -0.0050, frozen formal compare)`:
  - `selected checkpoints = 0 / 0 / 0`
  - `velocity_tracking_error_mean = 1.3359 +- 0.1232`
  - `joint_acceleration_l2_mean = 80.5803 +- 14.6031`
  - `action_jitter_l2_mean = 0.0160 +- 0.0009`
  - `episode_return_mean = 4.1769 +- 0.4080`
  - `fall_rate = 1.0000 +- 0.0000`

Interpretation:

- all twelve selected checkpoints are `checkpoint 0`
- every evaluated checkpoint inside every completed baseline sweep still has `fall_rate = 1.0`
- so this frozen regime should now be treated as the baseline-side failure record rather than as
  the current heuristic formal anchor

## Protocol repair probe outcome

The prepared repaired-budget follow-up is complete:

- [rough-terrain formal protocol repair probe](./baselines/rough-terrain-formal-protocol-repair-probe.md)
- [rough-terrain formal protocol revision decision](./baselines/rough-terrain-formal-protocol-revision-decision.md)
- [probe comparison_summary.json](../artifacts/analysis/rough_terrain_formal_protocol_repair_probe/comparison_summary.json)

Selected-checkpoint aggregate over seeds `11`, `17`, and `23`:

- `selected checkpoints = 0 / 0 / 200`
- `velocity_tracking_error_mean = 1.1558 +- 0.1545`
- `joint_acceleration_l2_mean = 111.5311 +- 25.5306`
- `action_jitter_l2_mean = 0.1023 +- 0.1211`
- `episode_return_mean = 22.3952 +- 26.3617`
- `fall_rate = 0.9167 +- 0.1179`

Interpretation:

- this probe broke the strongest `universal collapse` reading from the frozen `64 envs x 400`
  regime
- the old heuristic winner was therefore not identically doomed under every repaired protocol
  variant
- but the probe still did not produce a report-grade formal anchor
- so this probe should be read as the transition evidence that justified explicit protocol
  revision, not as the final anchor itself

## Revised heuristic anchor outcome

The prepared revised-protocol long-budget run is now complete:

- [rough-terrain formal protocol revision long-budget test](./baselines/rough-terrain-formal-protocol-revision-long-budget.md)
- [comparison_summary.json](../artifacts/analysis/rough_terrain_formal_protocol_revision_long_budget/comparison_summary.json)

Selected-checkpoint aggregate over seeds `11`, `17`, and `23`:

- `selected checkpoints = 350 / 300 / 350`
- `velocity_tracking_error_mean = 0.7549 +- 0.1068`
- `joint_acceleration_l2_mean = 119.8639 +- 2.1966`
- `action_jitter_l2_mean = 0.2711 +- 0.0084`
- `episode_return_mean = 100.9327 +- 11.2711`
- `fall_rate = 0.1500 +- 0.0816`

Interpretation:

- this revised row satisfies the repo's `3-seed + checkpoint-sweep` heuristic-anchor requirement
- none of the selected seeds survives through `checkpoint 0`
- `SC-PPO 3.8` remains better on the `速度跟踪误差主指标`, `跌倒率底线指标`,
  `关节震荡主指标`, and `动作抖动次级指标`
- `episode_return_mean` is effectively tied and remains only a `总回报补充指标`

So the repo can now defend an Isaac-side `方法优于启发式` reading again. The external-validation
side is closed only as `混合外部验证结论`, not as a cross-engine `SC-PPO` win.

## External-Validation Reading

The current `MuJoCo isaac_mainline` reading is now aligned against the refreshed revised heuristic
formal anchor.

Current comparable first-pass protocol:

- `terrain_mode = isaac_mainline`
- current resolved XML = `plane`
- `joint_reset_noise = 0.1`
- `20 episodes`
- `20 seconds`

Aligned comparable numbers over seeds `11`, `17`, and `23`:

- revised heuristic anchor:
  - `velocity_tracking_error_mean = 0.4188 +- 0.0398`
  - `joint_acceleration_l2_mean = 120.7339 +- 2.6413`
  - `action_jitter_l2_mean = 0.2452 +- 0.0288`
  - `fall_rate = 0.0000 +- 0.0000`
  - `episode_steps_mean = 2000.0 +- 0.0`
- `SC-PPO threshold = 3.8`:
  - `velocity_tracking_error_mean = 0.4910 +- 0.0944`
  - `joint_acceleration_l2_mean = 125.5411 +- 21.1683`
  - `action_jitter_l2_mean = 0.2313 +- 0.0351`
  - `fall_rate = 0.0167 +- 0.0236`
  - `episode_steps_mean = 1984.7833 +- 21.5196`

This supports a mixed external-validation reading:

- `SC-PPO 3.8` remains stronger in the Isaac rough-terrain main comparison
- revised heuristic is better on `MuJoCo isaac_mainline` task stability, velocity tracking, episode
  length, and joint acceleration
- `SC-PPO 3.8` is only better on `MuJoCo isaac_mainline` action jitter

Canonical comparable artifacts:

- [SC-PPO MuJoCo revised-anchor aligned comparison](./sc-ppo-mujoco-revised-anchor-aligned-comparison.md)

## PID有限消融 Reading

The limited PID ablation for issue `#6` is closed as mechanism support, not as a new headline
result.

Matched plain-dual diagnostic:

- config: `configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_dual_001.json`
- checkpoint: `100`
- `selection_status = all_checkpoints_collapsed`
- `velocity_tracking_error_mean = 1.1646`
- `joint_acceleration_l2_mean = 121.3371`
- `action_jitter_l2_mean = 0.1661`
- `episode_return_mean = 4.7101`
- `fall_rate = 1.0000`
- `constraint_violation_rate = 0.4091`

Interpretation:

- `普通对偶上升` does not clear the task floor in the available matched diagnostic
- its lower action jitter is not a usable smooth-control win because the policy collapses
- this supports keeping `PID-Lagrangian正式方案` as the formal `SC-PPO` line
- this does not support broad component attribution

Canonical artifact:

- [SC-PPO PID-limited ablation](./sc-ppo-pid-limited-ablation.md)

## SN-Only Replacement Diagnostic Reading

The `SN-only` replacement-mechanism branch is closed as a negative feasibility diagnostic.

Completed reduced-budget variants:

- full-actor `smoke`
- full-actor `short`
- full-actor `medium`
- hidden-layer-only `medium`
- hidden-layer-only `coeff = 2.0`
- first-hidden-only `medium`

Current reading:

- the implementation is operational and the checkpoints contain the expected spectral-normalization
  state
- every completed SN-only variant remains not task-valid under the diagnostic protocol
- first-hidden-only selectivity and coefficient loosening do not recover stable locomotion
- no `主实验三种子` or `MuJoCo关键两组终验` budget should be spent on this SN-only branch

Canonical artifacts:

- [SC-PPO SN feasibility diagnostic](./sc-ppo-sn-feasibility-diagnostic.md)
- `artifacts/analysis/sn_replacement_diagnostic/sn_ppo_first_hidden_rough_terrain_medium_seed123145_summary.json`

## Random-Stairs Stress Reading

Issue `#7` is closed as a selected-checkpoint stress test, not as a new method-ranking result.

Completed result:

- Vanilla PPO, the revised heuristic anchor, and `SC-PPO threshold = 3.8` were replayed from their
  selected rough-terrain checkpoints under the first stairs-only random-stairs protocol
- all three methods collapsed with `fall_rate = 1.0`
- the result is direct selected-checkpoint transfer failure, not a task-valid random-stairs
  advantage for any method

Canonical artifacts:

- [Random-stairs selected-checkpoint stress test](./random-stairs-selected-checkpoint-stress.md)
- `artifacts/analysis/random_stairs_selected_checkpoint_stress/comparison_summary.json`

## Freeze Package Status

The `科研交付冻结` pass is complete. The repo is already delivered as a `仓库内科研交付包` with
aligned reports, status docs, tracker state, artifact pointers, and a final reproduction checklist.

Freeze validation stayed lightweight:

- unit tests
- JSON/path sanity
- Markdown consistency
- `git diff --check`

The frozen package on `main` should not rerun Isaac training, MuJoCo replay, random-stairs
evaluation, or SN diagnostics. Any further work in those areas should start from a separate
post-freeze branch or issue.

Canonical freeze references:

- [Final research delivery checklist](./reproduction/final-research-delivery-checklist.md)
- [ADR 0001: freeze research delivery before new protocol repair](./adr/0001-freeze-research-delivery-before-new-protocol-repair.md)

## Post-Freeze Exploration & Cross-Engine Degradation

After the freeze, eight alternative smoothness mechanisms were tested under `同命题主线挑战`.
All are now closed. The central paper-direction finding is the cross-engine degradation pattern:

### Cross-engine degradation table

| Method | Isaac jnt_acc | MuJoCo jnt_acc | Degradation |
| --- | ---: | ---: | ---: |
| Heuristic baseline | 120 | 121 | ×1.01 |
| SC-PPO 3.8 | 116 | 126 | ×1.08 |
| LayerNorm epochs=3 | 172 | 603 | ×3.5 |
| Action Scaling | 144 | 1836 | ×12.7 |
| Output Scaling | 121 | 500 | ×4.1 |

Jacobian constraint and heuristic action-rate penalty are the only mechanisms
that preserve smoothness across engines.

### Supporting analysis

- Jacobian sensitivity causal chain: SC-PPO keeps sensitivity at ~3.6; LayerNorm
  needs ~10.7 for task acquisition, explaining the ~3.5× degradation
- PID-Lagrange multiplier dynamics: multiplier stays near zero, acting as safety
  mechanism rather than active enforcement
- Constraint threshold sensitivity: effective window is [3.6, 3.8)
- LDLJ/SPARC trace: LayerNorm wins on kinematic smoothness, SC-PPO wins on
  dynamic smoothness — revealing smoothness as two-dimensional
- SC-PPO epochs=3 repair: mixed result (seed11 improved, seed23 degraded)

Full analysis: [SC-PPO cross-engine degradation](./sc-ppo-cross-engine-degradation.md)

## What Is Not Supported

The repo still does not support the following claims:

- that current smoothness gains fully transfer to `MuJoCo`
- that `SC-PPO 3.8` beats the revised heuristic anchor on `MuJoCo isaac_mainline`
- that `MuJoCo terrain` is ready to serve as the main external validation result
- that the final checkpoint alone is sufficient for long-budget reporting
- that a broad neighborhood of tighter thresholds is interchangeable with the `3.8` mainline
- that the `PID有限消融` proves every PID term is independently necessary
- that the current `SN-only` branch is a task-valid replacement mechanism
- that the first stairs-only random-stairs protocol supports a task-valid method ranking
- that any non-Jacobian replacement mechanism tested so far (LayerNorm, action/output scaling,
  orthogonal actor) provides cross-engine smoothness robustness comparable to SC-PPO
- that `num_learning_epochs = 3` universally fixes final-checkpoint reliability for SC-PPO

The completed local controls and post-freeze exploration reinforce that boundary:

- repaired `threshold = 4.0` fails to match the `3.8` mainline and includes `seed23 -> checkpoint 0`
- `threshold = 3.6 + full_batch` looked promising on `seed11`, but its formal promotion attempt
  failed at the Isaac stage:
  - `seed11 -> checkpoint 350`
  - `seed17 -> checkpoint 350`
  - `seed23 -> checkpoint 0`
- SC-PPO epochs=3: seed11 improved (sel 300→400) but seed23 degraded (sel 400→300)
- Plain dual ascent: seed11 succeeds but seed23 collapses (sel=0)

## Recommended citation pattern

When summarizing the current project state, the safest compact wording is:

`在当前粗糙平面主实验中，repaired PID-Lagrangian SC-PPO（threshold = 3.8）仍然是仓库里最强的已完成方法线。冻结 formal-compare 曾经让 Vanilla PPO 与 bounded heuristic action-rate 家族全部塌到 checkpoint 0，而完成的 repaired-budget probe 先把旧 heuristic winner 收缩为 0 / 0 / 200，随后完成的 revised long-budget protocol 又把它修到 350 / 300 / 350。由此，仓库现在重新拥有了可防守的 3-seed heuristic formal anchor；相对于这条 revised heuristic anchor，SC-PPO 3.8 在 Isaac 粗糙平面共享指标上仍然同时更优于速度跟踪误差、跌倒率、关节震荡和动作抖动。但对齐后的 MuJoCo isaac_mainline replay 没有保持这个排序：revised heuristic 在任务稳定性、速度跟踪、episode length 和 joint acceleration 上更强，SC-PPO 3.8 只在 action jitter 上略强。因此当前外部验证应写成 mixed evidence，而不是 SC-PPO 的跨引擎优势。`

English-safe wording:

`On the current Isaac rough-terrain task, repaired PID-Lagrangian SC-PPO (threshold = 3.8) remains the strongest completed method line in the repo. The frozen baseline refresh first collapsed Vanilla PPO and the bounded heuristic action-rate family to checkpoint 0, the repaired-budget follow-up then narrowed the old heuristic winner to 0 / 0 / 200, and the completed revised long-budget protocol finally repaired that heuristic row to 350 / 300 / 350. So the repo now again has a defensible 3-seed heuristic formal anchor, and SC-PPO 3.8 remains better than that anchor on velocity tracking, fall rate, joint acceleration, and action jitter under the shared Isaac metric schema. However, the aligned MuJoCo isaac_mainline replay does not preserve that ordering: the revised heuristic anchor is better on task stability, velocity tracking, episode length, and joint acceleration, while SC-PPO 3.8 is only slightly better on action jitter. The external-validation reading should therefore be reported as mixed evidence rather than as a cross-engine SC-PPO advantage.`

**Post-freeze paper-direction summary (Chinese)**:

`冻结后，仓库在 同命题主线挑战 下系统测试了 8 种替代平滑机制（各向异性约束形状、动作时间变化率硬约束、SN、正交actor、LayerNorm actor、动作缩放、输出缩放、plain dual ascent），全部闭合为负向或混合结果。其中 LayerNorm actor（num_learning_epochs=3）是唯一通过 Isaac 内部挑战（3/3 selected=final=400）的架构侧候选，但其跨引擎 replay 显示关节加速度退化为 SC-PPO 3.8 的 3.5 倍。5 方法的跨引擎退化表（Heuristic ×1.01、SC-PPO ×1.08、LayerNorm ×3.5、Action Scaling ×12.7、Output Scaling ×4.1）构成了论文的核心证据：Jacobian 约束和启发式动作惩罚是仅有的两种能跨引擎保持平滑性的机制。Jacobian sensitivity 水平预测退化因子（SC-PPO ~3.6 → ×1.08；LayerNorm ~10.7 → ×3.5），支持 "约束 policy Jacobian sensitivity 是隐式 sim-to-sim 正则化" 的核心主张。`

**Post-freeze paper-direction summary (English)**:

`After the freeze, the repo systematically tested 8 alternative smoothness mechanisms under the same-question boundary (anisotropic constraint shape, action-rate hard constraint, SN, orthogonal actor, LayerNorm actor, action scaling, output scaling, plain dual ascent). All closed as negative or mixed. LayerNorm actor (num_learning_epochs=3) is the only architecture-side candidate to pass the full Isaac internal challenge (3/3 selected=final=400), but its cross-engine replay shows 3.5x worse joint acceleration than SC-PPO 3.8. The 5-method cross-engine degradation table (Heuristic ×1.01, SC-PPO ×1.08, LayerNorm ×3.5, Action Scaling ×12.7, Output Scaling ×4.1) forms the paper's core evidence: Jacobian constraint and heuristic action-rate penalty are the only mechanisms that preserve smoothness across engines. Jacobian sensitivity level predicts the degradation factor (SC-PPO ~3.6 → ×1.08; LayerNorm ~10.7 → ×3.5), supporting the core thesis that constraining policy Jacobian sensitivity serves as implicit sim-to-sim regularization.`
