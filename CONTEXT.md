# Humanoid Smooth Control Research

This context defines the language for a research project that validates constrained reinforcement learning methods for smoother humanoid locomotion. It exists to keep planning and reporting aligned with experimental evidence rather than framework productization.

## Language

**科研验证型交付**:
An outcome whose success is measured by reproducible experiments, comparative results, and analysis rather than reusable framework engineering.
_Avoid_: 工程产品型交付, 平台化交付

**工程产品型交付**:
An outcome whose success is measured by configurability, maintainability, and reuse across robots or constraint types.
_Avoid_: 科研验证型交付, 一次性实验代码

**诊断支线**:
A narrowly scoped experiment branch used to answer one local mechanism or threshold question before consuming formal comparison budget.
_Avoid_: 正式候选线, 主实验结果

**正式候选线**:
An experiment branch that has earned promotion from a **诊断支线** and is allowed to consume `主实验三种子` and `MuJoCo关键两组终验` budget to challenge the current mainline. Under the current three-seed-parallel rule, a bounded branch that clears the terminal gate on all three seeds earns this status directly and then must first complete its Isaac-side internal challenge before spending new cross-engine budget.
_Avoid_: 一次性探针, 已定稿主线

**同命题主线挑战**:
A post-freeze research direction that seeks a stronger **正式候选线** within the existing task definition, shared metric schema, and external-validation frame rather than expanding the research question.
_Avoid_: 扩展研究命题, 产品化转向, 另起任务定义

**终点可靠性主线挑战**:
A **同命题主线挑战** that targets final-checkpoint task validity and seeks to remove the current mainline's dependence on checkpoint selection for its core claim.
_Avoid_: 泛泛训练稳定性, 只修报告口径, 继续依赖中途峰值

**SC-PPO家族内稳态修复线**:
A **终点可靠性主线挑战** that keeps the current SC-PPO mechanism family and focuses on repairing its training or late-stage stability without changing the main comparison question. If both bounded objective-side levers close negatively under the current rules, this line is itself treated as closed within the current same-question boundary.
_Avoid_: 直接换新机制, 扩题, 只做文案层解释

**同命题负结论边界**:
A frozen reading that a bounded same-question repair line has already been tried and closed negatively within the current task and evidence frame, so any later branch must be opened explicitly as a new mechanism or diagnostic line rather than treated as a quiet continuation of the failed family. Under the current `SN` reopening rules, this boundary should also absorb the combination of the earlier `SN-only` failure and a failed first canonical task-stabilized `SN` recipe, rather than leaving `SN` as an implicit near-term retry family.
_Avoid_: 失败后换个邻域继续算同一线, 负结果不记账直接换线, 旧线续命

**PID放大器假设**:
A working diagnosis where late-stage PID multiplier dynamics are treated as an amplifier of the current objective tension rather than the primary cause of it. Under this reading, a failed bounded integral-state probe should push the next diagnostic toward an objective-side single-variable lever rather than more blind multiplier retuning, and the first such objective-side probe should hold `threshold = 3.8` fixed and change `cost_aggregation` from `quantile(0.90)` to `mean` as a single configuration instead of reopening a threshold neighborhood or starting an aggregation sweep. If that `mean` probe fails, the aggregation-side minimal diagnosis closes and the next objective-side lever becomes `constraint threshold`, starting from a looser rather than tighter direction and using the completed `threshold = 4.0` line only as a negative reference while the next active probe starts at `4.2`, with all non-threshold settings reset to the current `3.8` mainline. If `4.2` then fails any seed under the current `3/3` Isaac challenge rule, the threshold-side line also closes as a negative result rather than expanding to `4.5` or seed-specific repair.
_Avoid_: PID根因假设, 纯日志噪声假设, 直接改写目标张力结论

**三种子并行起步门槛**:
A diagnostic gate where a repair variant starts from the canonical three-seed set in parallel and must make the final checkpoint non-collapsed on each seed without obviously breaking the current task floor before broader promotion is justified. In the current `终点可靠性主线挑战`, that judgment is still supported by per-seed late-stage checkpoint sweeps rather than a final-checkpoint-only read, but those sweeps may not rescue a failed final checkpoint.
_Avoid_: 只挑单个好 seed, 中途 checkpoint 救场, 单次好看就正式主线

**基准三种子并行探针**:
A fixed three-seed parallel probe chosen for the first interpretable diagnosis pass before broader branch expansion, so different repair ideas are compared against the same seed set and failure surfaces.
_Avoid_: 每次换不同 seed 集, 只挑最好看的单 seed, 诊断和升格混在一起

**单变量积分态诊断**:
A first-pass PID diagnosis that keeps the threshold, sampling, and gain settings fixed and changes only the integral-state rule so any outcome stays attributable to multiplier-memory behavior. If the first bounded three-config batch all fails the terminal gate, this diagnosis closes as a negative result rather than expanding into a local integral-rule sweep.
_Avoid_: 阈值和积分态一起改, 多杠杆混调, 结果变好但不可解释

**终点平滑回拉门槛**:
A diagnostic gate where a repair variant must pull the final checkpoint back toward the smoother selected-checkpoint neighborhood without clearly giving up current task validity. In the first bounded batch, this means recovering at least half of the same-seed final-to-selected `joint_acceleration_l2_mean` gap while keeping `action_jitter_l2_mean` and `velocity_tracking_error_mean` within `+5%` of the current final checkpoint and `fall_rate` within `+0.05`.
_Avoid_: 只看是否 collapse, 只看回报, 终点继续漂向更粗糙解

**同种子回拉目标**:
A diagnostic reference that uses the selected checkpoint from the same seed as the smoothness pull-back target for interpreting late-stage drift. The first bounded batch reads that target through a same-seed `300 / 400 / final` neighborhood when available, but only as diagnosis around whether the final checkpoint has actually been repaired.
_Avoid_: 直接拿三种子均值做单 seed 诊断, 跨 seed 混参照, 用更好看 aggregate 掩盖局部失稳

**双参照终点门槛**:
A diagnostic gate where the final checkpoint keeps task validity relative to the current final checkpoint while its smoothness is pulled back toward the selected checkpoint from the same seed. In the first bounded batch, the task side is guarded by `velocity_tracking_error_mean <= +5%` and `fall_rate <= +0.05` versus the current final checkpoint, while the smoothness side is judged by at least `50%` recovery of the same-seed `joint_acceleration_l2_mean` gap with `action_jitter_l2_mean <= +5%`.
_Avoid_: 只看单一参照, 只追平滑不守任务, 只守任务不修终点漂移

**主线证据闭环**:
A planning stage where the repo first freezes the claim boundary, citation set, and external-validation reading of the current mainline before opening a new method branch.
_Avoid_: 结论未收口就开新坑, 把前沿方向池直接当执行计划

**科研交付冻结**:
A project stage where the completed evidence, claim boundary, reports, tracker state, and reproduction entrypoints are made consistent for delivery.
_Avoid_: 协议继续搜索, 新实验支线, 框架产品化

**仓库内科研交付包**:
A delivery package that freezes the repo's internal evidence map, reports, reproduction commands, artifact pointers, and tracker state without preparing a submission-specific paper package or reusable product.
_Avoid_: 论文投稿包, 工程产品包, 平台发布包

**冻结期轻量验证**:
Validation performed during **科研交付冻结** that checks tests, artifact pointers, JSON/Markdown consistency, and report reproducibility without generating new experimental evidence.
_Avoid_: 重跑训练, 新增正式评估, 新seed, 协议修复实验

**最终复现清单**:
An operational index inside a **仓库内科研交付包** that lists canonical commands, artifact paths, and validation checks for each completed evidence layer.
_Avoid_: README长命令堆积, 报告正文操作手册, 新实验计划

**冻结边界章节**:
A short report section that records completed post-mainline diagnostics and delivery boundaries without rewriting the main research narrative.
_Avoid_: 重写主报告, 新主结果章节, 实验计划章节

**冻结主档案分支**:
A main-branch posture where the repo stays stable as the archival research handoff line, but may still accept bounded post-freeze backports limited to **冻结边界章节** updates and reusable evaluation or diagnostic infrastructure.
_Avoid_: 借维护名义重开实验主叙事, 把机制专属实现直接回灌主线

**闭环后支线选择**:
A post-mainline decision point where the repo chooses one bounded follow-up branch after the current claim boundary is frozen.
_Avoid_: 多支线同时展开, 用新支线重写已冻结主结论

**协议修复线**:
A bounded workstream that improves evaluation-protocol alignment or discriminative power without treating the effort as a new algorithm mainline.
_Avoid_: 新方法主线, 静默失败补丁, 结果没收口就混入主报告

**部分迁移结论**:
A bounded external-validation claim where the current mainline transfers some task-relevant advantages across engines, but not the full smoothness conclusion.
_Avoid_: 混合外部验证结论, 全面跨引擎转优, 附录级边角结果

**混合外部验证结论**:
A bounded external-validation claim where cross-engine replay remains informative but does not preserve the main Isaac-side method ordering across the key metrics.
_Avoid_: 部分迁移结论, 全面跨引擎转优, 把指标分裂写成主线胜利

**架构级平滑优化线**:
A post-mainline diagnostic branch that tests whether an actor-side architectural constraint can replace the current Jacobian-penalty path without expanding the task definition. Even after both the current **各向异性约束形状线** and **动作时间变化率硬约束线** are closed negatively, the next truly different mechanism line should still remain inside the repo's current same-question low-level smoothness frame before any move into broader frontier directions such as `SysID`, `Residual RL`, perception, or `VLA`. Under the current decision tree, that next mechanism should no longer be framed as another nearby hard-constraint target plus dual-multiplier variant; it should change the mechanism form itself rather than merely reopen another hard-constraint family. Within the repo's current terminology, this **架构级平滑优化线** is the preferred next truly different mechanism line once the current hard-constraint shape and object families have both closed. Its first entry may not be a quiet reopening of the current `SN` family under a new label; it must start from an actor-side mechanism that is not already covered by the repo's current same-question negative boundary for `SN`. That first entry must also replace the current `Jacobian penalty / double backward` path as the active training-side smoothness mechanism rather than remain a Jacobian-supported actor-side variant. It should follow a **任务稳定化配方** opening: inherit the current `SC-PPO 3.8 + PID-Lagrangian` task-valid scaffold, preserve **完全替换对比**, and let the new actor-side mechanism be the sole active smoothness mechanism rather than a `Jacobian + new mechanism` hybrid or a `no-Jacobian / no-heuristic` cold-start. Under the current glossary, methods such as `Adaptive Action Scaling` do not count as this line's first-entry candidate because they are not actor-side architectural constraints. Under the current repo terminology, once `SN` is excluded by the frozen negative boundary and `Adaptive Action Scaling` is excluded by the glossary, the repo opened **正交低增益 actor 线** as a bounded first-entry candidate: orthogonal parametrization on the actor linear layers with a fixed bounded actor output gain, opened as a task-stabilized full-replacement comparison without the Jacobian/double-backward smoothness path. That candidate has now completed and closed negatively under the canonical rough-terrain `11 / 17 / 23`, `512 envs x 400 iterations` entry. The follow-up gain-isolation probe that raises `output_gain` from `0.50` to `1.00` also remains fully collapsed: all three seeds still end at `fall_rate = 1.0`, all three seeds are marked `all_checkpoints_collapsed`, and the higher-gain variant trades into materially larger `action_jitter_l2_mean` and `joint_acceleration_l2_mean` without rescuing task-valid survival. Under the current repo evidence, this rules out the “only low gain caused the failure” explanation and closes the broader orthogonal-actor first-entry candidate. The repo then opened **LayerNorm actor 线** as the next bounded first-entry candidate for **架构级平滑优化线**: actor hidden-layer activation normalization with the Jacobian/double-backward path removed from the active smoothness mechanism, first launched as a task-stabilized full-replacement comparison with `actor_output_gain = 1.00` to avoid reopening the low-gain confound. That first canonical LayerNorm recipe has now also completed under the canonical rough-terrain `11 / 17 / 23`, `512 envs x 400 iterations` entry and does not promote: selected-checkpoint aggregate `fall_rate_mean = 0.8667`, `seed11` remains `all_checkpoints_collapsed`, and although `seed17/23` partially rescue survival, the final-checkpoint aggregate still sits at `fall_rate_mean = 0.7000` with materially elevated `joint_acceleration_l2_mean`, `action_jitter_l2_mean`, and near-saturated late-stage `constraint_violation_rate` on the stronger seeds. Under the current repo evidence, this should be read as a partial task-valid rescue relative to orthogonal actor, not as a successful smooth replacement. The line therefore stays open only as a bounded repair candidate, and the next minimal follow-up is to reduce `actor_output_gain` from `1.00` toward a less aggressive value such as `0.75` without reopening a broader architecture sweep.
_Avoid_: 直接改成更大研究命题, 把效率优化和能力外扩混成一条线

**非架构型替代机制线**:
A same-question low-level smoothness mechanism family that changes the smoothness mechanism form without requiring an actor-side architectural constraint and without reopening the current hard-constraint target-plus-dual family. It stays inside the current `速度跟踪行走 / 同尺比较 / 低层平滑机制` frame and exists to hold candidates such as constraint-aware action or output modulation methods when **架构级平滑优化线** has no repo-internal named first-entry candidate.
_Avoid_: actor-side architectural constraint, current hard-constraint 家族续命, `SysID`/`Residual RL`/感知/`VLA`

**动作缩放替代机制线**:
A same-question non-architectural replacement mechanism family that changes the smoothness mechanism form by modulating policy action under constraint-aware control, rather than by reopening the current hard-constraint target-plus-dual family or by introducing an actor-side architectural constraint. Under the current glossary, this is the preferred first active line inside **非架构型替代机制线** and is the natural home for candidates in the `Adaptive Action Scaling` class while staying inside the current `速度跟踪行走 / 同尺比较 / 低层平滑机制` frame. Its first entry should also open as a **任务稳定化配方**: inherit the current `SC-PPO 3.8 + PID-Lagrangian` task-valid scaffold, preserve **完全替换对比**, use the canonical `11 / 17 / 23` at rough-terrain `512 envs x 400 iterations`, and let a constraint-aware action-side scaling path be the sole active smoothness mechanism rather than an `output-side` variant, an `action + output` mixed variant, a `Jacobian + 动作缩放` hybrid, or a `no-Jacobian / no-heuristic` cold-start. Under **同尺比较**, that first entry should still keep evaluation-side `collect_local_sensitivity = true` and preserve `local_sensitivity_threshold = 3.8` as a shared side read, so the repo can still observe whether replacing the training-side smoothness mechanism also improves or degrades the original `local_sensitivity` evidence. Its first entry gate should still use the current **三种子并行起步门槛** and the shared task-validity floor rather than inventing a new success rule based only on scaling activation, clipping rate, or standalone constraint-reduction traces. At that entry stage, per-seed checkpoint sweeps may still serve as diagnostic side reads under the current gate, but they may not rescue a failed `final checkpoint`; the stricter `selected checkpoint = final checkpoint` rule still belongs to the later Isaac-side internal challenge. If that first action-scaling batch clears the current entry gate on all three canonical seeds, it should earn **正式候选线** status directly rather than going through an action-scaling-specific extra confirmation stage. If any one seed fails that first entry gate, the first canonical action-scaling recipe should close as a negative result rather than immediately reopening `scale schedule / clipping shape / gain` neighborhood repair inside the same round. If that first canonical recipe closes negatively, the repo should also close the broader “replace the current training-side smoothness mechanism with constraint-aware action-side scaling” direction within the current same-question boundary before it explicitly opens another truly different mechanism family. Under the current repo evidence, that first canonical recipe has now completed and closed negatively at rough-terrain `512 envs x 400 iterations` on the canonical `11 / 17 / 23`: all three seeds selected the final `checkpoint = 400`, aggregate `fall_rate_mean = 0.3667`, aggregate `velocity_tracking_error_mean_mean = 0.7700`, and the mechanism saturated without meeting the shared `local_sensitivity` readout floor, with `lagrange_multiplier = 5.0`, `action_scale = 0.5`, `constraint_violation_rate ~= 1.0`, and `policy_local_sensitivity_cost_update` remaining in the `7.8 - 9.4` range against `threshold = 3.8`. Under the current decision tree, this closes the broader action-side scaling replacement direction inside the repo's current same-question boundary rather than leaving a near-term `gain / schedule / clipping` neighborhood warm.
_Avoid_: 启发式动作差分惩罚, actor-side architectural constraint, current hard-constraint 家族续命

**输出缩放替代机制线**:
A same-question non-architectural replacement mechanism family that changes the smoothness mechanism form by modulating policy output under constraint-aware control, rather than by reopening the current hard-constraint target-plus-dual family or by introducing an actor-side architectural constraint. Under the current glossary, this is a named future line inside **非架构型替代机制线**, parallel to **动作缩放替代机制线**, and is kept explicit so `output-side scaling` variants do not remain in an unnamed gray area after the action-side family has been narrowed. If this line is opened later, its first entry should also start as a **任务稳定化配方**: inherit the current `SC-PPO 3.8 + PID-Lagrangian` task-valid scaffold, preserve **完全替换对比**, use the canonical `11 / 17 / 23` at rough-terrain `512 envs x 400 iterations`, and change only the training-side smoothness mechanism to constraint-aware output-side scaling. Under the current repo implementation, the first `action_scaling_ppo` prototype scales the actor output mean inside `ActorCritic.update_distribution()` and `act_inference()` while leaving exploration `std` unchanged, so that prototype should be read as a negative result for the current implementation recipe rather than silently treated as a clean closure of this future line or as proof that a pure post-policy action-side hook has already been isolated. Under the current repo evidence, the first canonical explicit `OutputScalingPPO` recipe has now also completed and closed negatively at rough-terrain `512 envs x 400 iterations` on the canonical `11 / 17 / 23`: all three seeds selected the final `checkpoint = 400`, aggregate `fall_rate_mean = 0.4333`, aggregate `velocity_tracking_error_mean_mean = 0.7689`, aggregate `joint_acceleration_l2_mean_mean = 121.36`, and aggregate `action_jitter_l2_mean_mean = 0.2164`. Although this line was smoother than the canonical action-side recipe on aggregate joint/action smoothness metrics, it still failed the shared three-seed entry gate because one canonical seed remained clearly outside the current task-valid floor (`seed17 fall_rate = 0.85`), and the mechanism saturated without meeting the shared `local_sensitivity` readout floor, with `lagrange_multiplier = 5.0`, `output_scale = 0.5`, `constraint_violation_rate ~= 1.0`, and `policy_local_sensitivity_cost_update` remaining in the `5.6 - 6.7` range against `threshold = 3.8`. Under the current decision tree, this closes the broader output-side scaling replacement direction inside the repo's current same-question boundary rather than opening a near-term `gain / floor / schedule` neighborhood retry.
_Avoid_: action-side scaling, actor-side architectural constraint, current hard-constraint 家族续命

**各向异性约束形状线**:
A same-question non-`SN` mechanism family that keeps **策略局部敏感度** as the constrained object but replaces the current isotropic scalar constraint with an anisotropic or group-structured constraint shape. Under the current repo evidence, this is the preferred post-`SN` family before any hard-constraint line built directly on **动作时间变化率**, and its first entry should be a bounded grouped-anisotropic probe rather than a full ALCP jump. In that first entry, non-shape settings should be pulled back to the current `SC-PPO 3.8 + PID-Lagrangian` mainline so the grouped anisotropic shape is the first new variable rather than part of a mixed probe, and the repo should first backport only the minimal anisotropic geometry capability rather than also introducing `positive_part`, `update_error_mode`, or `legacy_guard_mode`. The first proximal-only grouped weights should preserve total scale as much as possible so `threshold = 3.8` remains meaningfully comparable to the current isotropic mainline, that first grouped-anisotropic probe should keep `threshold = 3.8` fixed rather than reopening threshold search, it should start directly from the canonical `11 / 17 / 23` three-seed set at the current rough-terrain `512 envs x 400 iterations` budget rather than from a single-seed short pass, and a clean `3/3` pass at that entry budget directly earns **正式候选线** status. At that entry stage, per-seed checkpoint sweeps may still serve as diagnostic side reads under the current **三种子并行起步门槛**, but they may not rescue a failed `final checkpoint`; the stricter `selected checkpoint = final checkpoint` rule belongs to the later Isaac-side internal challenge. If any one seed fails the entry gate, this first canonical anisotropic-shape recipe closes as a negative result rather than reopening distal weighting, asymmetric weighting, or threshold recalibration inside the same round. If that first canonical recipe closes negatively, the repo should also close the broader “keep **策略局部敏感度** as the constrained object and only change the constraint shape” direction within the current same-question boundary before it explicitly opens a new mechanism family.
_Avoid_: 直接改成动作时间变化率硬约束, `SN` 续命, 只换名字不换机制

**动作时间变化率硬约束线**:
A same-question non-`SN` mechanism family that changes the constrained object itself from **策略局部敏感度** to **动作时间变化率** and applies the hard constraint directly to adjacent-timestep action change. Under the current decision tree, this line should open only after the current **各向异性约束形状线** has already closed negatively, so “change constraint shape” and “change constrained object” remain two separate questions rather than a drifting mixed family. Its first entry should still preserve the current `SC-PPO 3.8 + PID-Lagrangian` mainline scaffold as much as possible: keep `PID-Lagrangian`, `cost_aggregation = quantile(0.90)`, `pid_integral_mode = lower_bound_clamp`, **完全替换对比**, and the canonical `11 / 17 / 23` at rough-terrain `512 envs x 400 iterations`, and change only the constrained object from **策略局部敏感度** to **动作时间变化率**. Because the `动作时间变化率` objective has a different metric scale from `策略局部敏感度`, the first entry should not reuse `threshold = 3.8`; it should start from a looser, already implemented action-rate scale such as `threshold = 3.0` rather than an overly hard `0.2`, so the first probe isolates the object switch instead of mixing it with an aggressive threshold shock. Under **同尺比较**, that first action-rate entry should still keep evaluation-side `collect_local_sensitivity = true` and preserve `local_sensitivity_threshold = 3.8` as a shared side read, so the repo can still ask whether changing the constrained object also improves or degrades the original local-sensitivity evidence. Its first entry gate should still use the current **三种子并行起步门槛** and the shared task-validity floor rather than inventing a new success rule based only on `action_rate` constraint reduction. At that entry stage, per-seed checkpoint sweeps may still serve as diagnostic side reads under the current gate, but they may not rescue a failed `final checkpoint`; the stricter `selected checkpoint = final checkpoint` rule still belongs to the later Isaac-side internal challenge. If that first `threshold = 3.0` action-rate batch clears the current entry gate on all three canonical seeds, it should earn **正式候选线** status directly rather than going through an action-rate-specific extra confirmation stage. If any one seed fails that first entry gate, the first canonical action-rate recipe should close as a negative result rather than immediately reopening `2.5 / 2.0 / 0.2` threshold-neighborhood repair inside the same round. If that first canonical recipe closes negatively, the repo should also close the broader “replace the constrained object with **动作时间变化率**” direction within the current same-question boundary before it explicitly opens another truly different mechanism family.
_Avoid_: 各向异性形状邻域续命, 只改名字不改约束对象, 启发式动作差分惩罚

**替代机制可行性诊断**:
A diagnostic branch stage that first asks whether a new smoothness mechanism can train, activate, and yield interpretable evidence before it is judged against the current mainline. After a **同命题负结论边界** is frozen, any continued same-question method exploration should reopen here explicitly rather than be framed as a quiet continuation of the failed family.
_Avoid_: 零实现就直接打主线, 一上来就烧正式预算

**任务稳定化配方**:
A replacement-mechanism recipe that keeps the new mechanism explicit but restores enough proven task-supporting training structure that failure can be read as a mechanism result rather than as a collapsed cold-start protocol. It should inherit a known task-valid training base rather than restarting from a `no-Jacobian / no-heuristic` cold-start, the first SN reopening should default to the current `SC-PPO 3.8 + PID-Lagrangian` mainline base rather than the revised heuristic anchor, and it should still preserve **完全替换对比** instead of reintroducing heuristic smoothness reward. In this first SN reopening, the inherited base provides shared task-stability scaffold only; `SN` is the sole active smoothness mechanism rather than part of a `Jacobian + PID + SN` hybrid, the inherited `3.8` should remain only as an evaluation-side `local_sensitivity` readout threshold rather than as an active training target, the first interpretable evidence should follow **三种子并行起步门槛** while any `smoke` run remains implementation self-check only, and the first batch should hold to one canonical full-actor SN configuration rather than reopen hidden-layer or coefficient sweeps. That canonical first config keeps actor-output-layer SN enabled and keeps `actor_spectral_norm_coeff = 1.0`, its first real three-seed training batch should align to the current rough-terrain `512 envs x 400 iterations` budget rather than reuse the old reduced-budget `short / medium` presets, and a clean `3/3` pass at that budget directly earns **正式候选线** status. At that entry stage, per-seed checkpoint sweeps may still serve as diagnostic side reads under the current **三种子并行起步门槛**, but they may not rescue a failed `final checkpoint`; the stricter `selected checkpoint = final checkpoint` rule belongs to the later Isaac-side internal challenge. If any one seed fails the entry gate, this first canonical SN task-stabilized recipe closes as a negative result rather than reopening hidden-layer or coefficient repair inside the same round or immediately opening `SN v2`.
_Avoid_: no-Jacobian/no-heuristic 冷启动, 盲目继续换层换系数, 机制问题和训练底座问题混在一起

**同尺比较**:
A comparison rule where a new mechanism is trained differently if needed, but is still evaluated under the repo's existing shared metric and constraint-evidence schema.
_Avoid_: 机制一换就连评估尺子一起换, 比较关系失真

**远期方向池**:
A set of plausible future research directions that remain documented but are not yet promoted into execution.
_Avoid_: 当前执行计划, 默认下一步都要做

**正式候选线升格门槛**:
The minimum evidence a **诊断支线** must clear before it is allowed to consume formal replication and cross-engine evaluation budget. Under the current three-seed-parallel rule, a bounded branch that clears the terminal gate on all three seeds directly earns **正式候选线** status without an extra same-kind confirmation stage. After that, it must still win its Isaac-side internal challenge before new `MuJoCo` budget is spent. In the first bounded integral-state batch, it promotes at most one configuration into broader replication, chosen by a `先过终点门槛再取最平滑` rule on the final checkpoint rather than by a new composite score.
_Avoid_: 单次现象, 最终胜负结论

**逐种子硬门槛**:
A promotion rule where each seed must first clear non-collapse and task-floor checks before aggregate evidence is allowed to count toward mainline challenge.
_Avoid_: 只看均值, 坏种子被平均掩盖

**早期checkpoint失效规则**:
A promotion rule that rejects a candidate line if its selected checkpoint peaks too early in training, even when that early checkpoint looks task-valid.
_Avoid_: 中途截图升格, 早峰值当正式主线

**方法优于启发式**:
A research claim that a constrained optimization method outperforms heuristic reward shaping on the target control objective.
_Avoid_: 全组件独立归因, 全量机制拆分验证

**组件消融验证**:
A research claim that each algorithmic component must be isolated and shown to contribute independently.
_Avoid_: 主方法对比, 有限消融

**速度跟踪行走**:
A humanoid locomotion task whose primary objective is to follow commanded velocity while maintaining stable walking.
_Avoid_: 复杂地形通用运动, 泛化运动集合

**复杂地形条件**:
Terrain variation used as a training or evaluation condition for a locomotion task rather than as the task definition itself.
_Avoid_: 主任务, 研究命题

**策略局部敏感度**:
The magnitude of policy output change with respect to observation change, typically measured by a Jacobian-related norm.
_Avoid_: 动作时间变化率, 奖励惩罚项

**动作时间变化率**:
The change in action values across adjacent control timesteps.
_Avoid_: 策略局部敏感度, LCP约束

**完全替换对比**:
An experimental setup where the constrained method removes heuristic smoothness penalties and relies on the hard constraint alone.
_Avoid_: 叠加保守版, 混合平滑机制

**PID-Lagrangian正式方案**:
The primary constrained optimization method where the Lagrange multiplier is updated with PID-style feedback rather than plain dual gradient ascent.
_Avoid_: 普通对偶上升正式方案, 临时增强项

**普通对偶上升**:
The basic Lagrange multiplier update that adjusts the multiplier directly from current constraint violation.
_Avoid_: PID-Lagrangian正式方案, 主算法定义

**平滑优先成功判据**:
A success condition where smoothness must improve materially while task performance remains above a predefined acceptable floor.
_Avoid_: 全面统治, 事后解释阈值

**任务守底线**:
A predefined tolerance that limits how much task performance is allowed to degrade relative to the comparison baseline.
_Avoid_: 无约束性能退化, 训练后再改标准

**奖励无关任务指标**:
Task metrics that remain comparable even when competing methods use different reward formulations.
_Avoid_: 总回报主判据, 混合奖励比较

**总回报补充指标**:
Episode return used only as supporting training evidence rather than as the primary cross-method comparison target.
_Avoid_: 主任务判据, 唯一结论依据

**行为层平滑指标**:
Smoothness evidence measured from time-series behavior such as action variation, joint acceleration, or torque-rate changes.
_Avoid_: 约束层主结论, Jacobian主证据

**约束层机制证据**:
Constraint-side evidence such as Jacobian norms or violation rates used to explain why a method behaves smoothly.
_Avoid_: 行为层主结论, 最终平滑判据

**关节震荡主指标**:
The primary smoothness metric based on joint-level oscillation, typically measured from joint acceleration magnitude or related physical jitter statistics.
_Avoid_: 动作抖动主指标, 仅网络输出平滑

**动作抖动次级指标**:
A secondary smoothness metric based on action variation across time, used to support but not replace joint-level evidence.
_Avoid_: 唯一主结论, 物理层主指标

**速度跟踪误差主指标**:
The primary task-performance metric defined by the difference between commanded base velocity and realized linear or angular velocity.
_Avoid_: 跌倒率主指标, 总回报主判据

**跌倒率底线指标**:
A supporting safety-style metric that checks whether a policy remains upright often enough to count as task-valid.
_Avoid_: 主任务定义, 唯一任务指标

**复杂地形分层推进**:
An execution strategy that validates the method on a simpler rough-terrain condition before promoting it to a harsher stair-like stress test.
_Avoid_: 双地形并行主实验, 一开始全矩阵展开

**三组正式对比**:
The primary experiment matrix containing Vanilla PPO, PPO with action-rate penalty, and SC-PPO under the same task definition.
_Avoid_: 两组直接对打, 缺少原始参照组

**MuJoCo必做终验**:
A mandatory cross-engine validation stage where converged policies are replayed in MuJoCo to test whether smoothness transfers beyond the training simulator.
_Avoid_: 仅Isaac Gym闭环, 可有可无的附加展示

**MuJoCo关键两组终验**:
A cross-engine validation scope that replays only the heuristic smoothing baseline and SC-PPO in MuJoCo.
_Avoid_: 三组全量终验, 把原始PPO也带入终验主线

**主实验三种子**:
An evidence standard where the main comparison on rough terrain is repeated with three random seeds and reported with central tendency and variance.
_Avoid_: 主结论单种子, 全场景都做重统计

**次级场景降级复现**:
A lighter validation standard where stress tests or cross-engine checks use fewer seeds than the main experiment.
_Avoid_: 所有场景同强度统计, 全部单次展示

**启发式小范围调参基线**:
A fairness protocol where the action-rate-penalty baseline is selected only after a predefined small hyperparameter sweep.
_Avoid_: 单权重拍脑袋基线, 未校准启发式对比

**先过底线再取最平滑**:
A baseline-selection rule that first filters candidates by task-validity thresholds and then picks the smoothest remaining option.
_Avoid_: 综合分排序, 奖励式多指标揉合

**先过终点门槛再取最平滑**:
A bounded promotion rule for the current `终点可靠性主线挑战`: first require the final checkpoint to clear the same-seed terminal gate, then choose the lowest `joint_acceleration_l2_mean` among the surviving configurations, use `action_jitter_l2_mean` as the first tie-break, and `velocity_tracking_error_mean` only as a later tie-break.
_Avoid_: final 没过也升格, 新造综合分, 用中途 checkpoint 排序

**相对退化阈值**:
A task-validity rule that bounds how much worse a method may perform relative to the selected heuristic baseline.
_Avoid_: 绝对任务阈值, 脱离对比对象的验收线

**速度误差10%退化上限**:
A task-validity threshold that allows SC-PPO to worsen the primary velocity-tracking error by at most 10% relative to the selected heuristic baseline.
_Avoid_: 5%过严阈值, 无明确数值上限

**跌倒率5个百分点退化上限**:
A task-validity threshold that allows SC-PPO to worsen the fall rate by at most five percentage points relative to the selected heuristic baseline.
_Avoid_: 绝对存活率硬门槛, 无独立生存底线

**PID有限消融**:
A narrow ablation that compares PID-Lagrangian against plain dual ascent within SC-PPO to explain multiplier stability.
_Avoid_: 双消融扩张, 全组件独立归因

## Relationships

- A **科研验证型交付** prioritizes baseline reproduction, algorithm validation, and report-ready evidence
- An **工程产品型交付** prioritizes modular abstractions and long-term extensibility
- A **诊断支线** answers a local experimental question before the repo spends budget on a broader claim
- A **正式候选线** is eligible for formal evidence collection but does not automatically become the repo's final mainline
- Under the current three-seed-parallel rule, a bounded branch that becomes a **正式候选线** should still complete an Isaac-side internal challenge before it spends new cross-engine budget
- A **同命题主线挑战** stays inside the current task and evidence boundary and aims to produce a stronger **正式候选线**
- A **终点可靠性主线挑战** is a **同命题主线挑战** focused on making the final checkpoint itself task-valid enough to carry the mainline claim
- A **SC-PPO家族内稳态修复线** is a **终点可靠性主线挑战** that first tests whether the current SC-PPO family can be stabilized before introducing a new mechanism family, and it closes as a negative result if both bounded objective-side levers fail
- A **同命题负结论边界** records that a bounded same-question line has already failed cleanly, so the repo must not continue that family implicitly under a new nearby knob
- A failed first canonical task-stabilized `SN` recipe should fold back into the repo's **同命题负结论边界** for `SN`, rather than leaving silent room for immediate `SN v2`
- Once the current `SN` reopening is folded into a **同命题负结论边界**, any further same-question method exploration must switch explicitly to a non-`SN` mechanism family
- A **PID放大器假设** treats late-stage multiplier dynamics as a leverage point inside a **SC-PPO家族内稳态修复线** without denying that objective tension may still be primary
- If the first bounded integral-state probe fails, the current **PID放大器假设** implies that the next single-variable diagnostic should move to the objective side rather than continue a blind PID retuning loop, and should first hold `threshold = 3.8` fixed while changing `cost_aggregation` from `quantile(0.90)` to `mean`
- A **三种子并行起步门槛** should be cleared before a **SC-PPO家族内稳态修复线** spends broader formal-comparison budget, and per-seed sweeps may diagnose drift but may not rescue a failed final checkpoint
- A **基准三种子并行探针** keeps first-pass diagnosis comparable by holding the initial seed set fixed while repair knobs change
- A **单变量积分态诊断** is the cleanest first probe for a **PID放大器假设** because it isolates multiplier-memory behavior from threshold changes
- If the first bounded three-config integral-state batch all fails, the repo should close that **单变量积分态诊断** as a negative result instead of turning it into a broader local hyperparameter search
- A **终点平滑回拉门槛** is stricter than mere non-collapse when the final checkpoint is already task-valid but drifts into a rougher behavior regime, and the first bounded batch quantifies that pull-back with a `50%` joint-acceleration gap recovery plus `+5% / +5% / +0.05` guardrails
- A **同种子回拉目标** keeps late-stage drift diagnosis local to one failure surface by comparing the final checkpoint against the selected checkpoint from the same seed
- The first bounded three-seed batch should include per-seed late-stage checkpoint sweeps instead of relying on final-checkpoint-only reads
- A **双参照终点门槛** combines a task floor anchored at the current final checkpoint with a smoothness pull-back target anchored at the selected checkpoint from the same seed
- A **主线证据闭环** should happen before the repo opens a materially broader post-mainline exploration branch
- A **科研交付冻结** follows **主线证据闭环** and prioritizes report, tracker, artifact, and reproduction consistency over new experiments
- A **仓库内科研交付包** is the concrete output of **科研交付冻结** when the repo is being frozen for internal research handoff rather than submission or product release
- **冻结期轻量验证** is the only validation style appropriate during **科研交付冻结** unless the maintainer explicitly reopens a new experiment branch
- A **最终复现清单** belongs to a **仓库内科研交付包** and keeps operational reproduction details out of README and report narrative
- A **冻结边界章节** updates reports with post-mainline diagnostic outcomes without reopening the main result structure
- A **冻结主档案分支** keeps `main` stable after **科研交付冻结** while still allowing bounded backports for **冻结边界章节** and reusable evaluation or diagnostic infrastructure
- A **闭环后支线选择** should choose one bounded **诊断支线** without reopening the frozen mainline claim
- A **协议修复线** exists to fix evaluation semantics or protocol quality without being mistaken for a new algorithm claim
- A **部分迁移结论** lets the repo include external validation in the main report without overstating mixed evidence
- A **混合外部验证结论** should replace a **部分迁移结论** when aligned cross-engine replay does not preserve the Isaac-side method ordering
- A **架构级平滑优化线** should stay inside the current locomotion claim and first test mechanism replacement rather than task expansion
- Even after both the current shape-only and object-replacement hard-constraint lines close, the next truly different mechanism line should still stay inside the current same-question low-level smoothness frame rather than jumping directly into `SysID`, `Residual RL`, perception, or `VLA`
- After the current shape-only and object-replacement hard-constraint lines close, the next truly different mechanism line should change mechanism form itself rather than reopen another nearby hard-constraint target plus dual-multiplier family
- Within the repo's current terminology, the preferred next truly different mechanism line after those hard-constraint families close is **架构级平滑优化线**
- The first **架构级平滑优化线** entry may not be a quiet reopening of the current `SN` family under a new label; it must start from an actor-side mechanism not already covered by the repo's current same-question negative boundary for `SN`
- The first **架构级平滑优化线** entry must replace the current `Jacobian penalty / double backward` path as the active training-side smoothness mechanism rather than remain a Jacobian-supported actor-side variant
- The first **架构级平滑优化线** entry should open as a **任务稳定化配方**: inherit the current task-valid scaffold, preserve **完全替换对比**, and let the new actor-side mechanism be the sole active smoothness mechanism rather than a `Jacobian + new mechanism` hybrid or a `no-Jacobian / no-heuristic` cold-start
- Under the current glossary, methods such as `Adaptive Action Scaling` are outside the first-entry candidate set for **架构级平滑优化线** because they are not actor-side architectural constraints
- Once `SN` and `Adaptive Action Scaling` were excluded from the first-entry candidate set, the repo opened **正交低增益 actor 线** as a bounded first-entry candidate for **架构级平滑优化线**
- The canonical `output_gain = 0.50` orthogonal actor entry closed negatively, and the follow-up gain-isolation probe with `output_gain = 1.00` also fully collapsed on the shared rough-terrain `11 / 17 / 23`, `512 envs x 400 iterations` entry
- Under the current repo evidence, that gain-isolation result rules out the “just low gain” explanation and closes the broader orthogonal-actor first-entry candidate rather than leaving a near-term `gain` neighborhood repair open
- After the orthogonal-actor first-entry candidate closed, the repo named **LayerNorm actor 线** as the next bounded first-entry candidate for **架构级平滑优化线**
- Under the current repo terminology, **LayerNorm actor 线** is distinct from `SN`, orthogonal parametrization, and scaling families because it targets hidden activation normalization rather than linear-operator geometry or action/output modulation
- A **非架构型替代机制线** keeps the repo inside the same-question low-level smoothness frame when the next candidate changes mechanism form but is not an actor-side architectural constraint
- Within the current glossary, the preferred first active line inside **非架构型替代机制线** is **动作缩放替代机制线**
- A **各向异性约束形状线** keeps the current constrained object while changing the constraint geometry, so it is the preferred first non-`SN` family after the current `SN` negative boundary, and it should first enter through a bounded grouped-anisotropic probe rather than a full ALCP rewrite
- If the first canonical **各向异性约束形状线** recipe fails, the repo should close the broader “keep **策略局部敏感度** and only change shape” direction within the current same-question boundary before opening another mechanism family
- A **动作时间变化率硬约束线** should open only after the current **各向异性约束形状线** has closed negatively, so changing the constrained object is kept distinct from changing only the constraint shape
- The first **动作时间变化率硬约束线** probe should preserve the current task-valid scaffold and change only the constrained object, rather than simultaneously retuning PID, aggregation, or seed/budget protocol
- The first **动作时间变化率硬约束线** threshold should start from a looser action-rate scale such as `3.0`, not by reusing `3.8` from `local_sensitivity` or by jumping straight to a much harder `0.2`
- Under **同尺比较**, the first **动作时间变化率硬约束线** probe should still keep `local_sensitivity = 3.8` as an evaluation-side shared readout rather than dropping the old evidence channel entirely
- The first **动作时间变化率硬约束线** entry gate should still use the repo's current **三种子并行起步门槛** and shared task floor, with `action_rate` reduction treated as branch-specific diagnosis rather than as a replacement success criterion
- If the first **动作时间变化率硬约束线** batch clears the current gate on all three canonical seeds, it should become a **正式候选线** directly rather than entering a branch-specific extra confirmation stage
- At the first **动作时间变化率硬约束线** entry gate, per-seed checkpoint sweeps may remain diagnostic side reads but may not rescue a failed `final checkpoint`; the stricter `selected checkpoint = final checkpoint` rule still belongs to the later Isaac-side internal challenge
- If any one seed fails the first **动作时间变化率硬约束线** entry gate, the first canonical action-rate recipe should close as a negative result rather than reopening an immediate threshold-neighborhood repair
- If the first canonical **动作时间变化率硬约束线** recipe fails, the repo should close the broader “replace the constrained object with **动作时间变化率**” direction within the current same-question boundary before opening another mechanism family
- A **替代机制可行性诊断** should precede any formal mainline challenge from a zero-implementation method branch, especially after a **同命题负结论边界** has closed the previous family
- A **任务稳定化配方** reopens a failed mechanism family on a task-valid training footing so the next negative result does not collapse back into the same protocol confound, and it should inherit a known task-valid base rather than a cold-start diagnostic regime
- A **同尺比较** keeps mechanism replacement interpretable by preserving the repo's existing evidence chain
- A **远期方向池** records broader ideas such as compliance, sim-to-real, or perception without forcing immediate execution
- A **正式候选线升格门槛** determines when a **诊断支线** may challenge the current mainline with formal budget
- Under the current three-seed-parallel rule, a bounded branch that clears the terminal gate on all three seeds can become a **正式候选线** directly without an extra same-kind confirmation stage
- Direct promotion to **正式候选线** does not bypass the Isaac-side internal challenge step before `MuJoCo关键两组终验`
- The Isaac-side internal challenge for a **正式候选线** need not dominate every shared metric; it should primarily show a more credible final checkpoint while preserving the main task floor
- In the current `终点可靠性主线挑战`, a credible final checkpoint means each seed's selected checkpoint must be the final checkpoint itself, not merely tied with an earlier checkpoint
- In the current `终点可靠性主线挑战`, the Isaac-side internal challenge is conjunctive: for each seed, the final checkpoint must both clear the current terminal gate and be the unique selected checkpoint
- In the current `终点可靠性主线挑战`, the Isaac-side internal challenge also keeps a strict `3/3` rule: all three seeds must pass, and `2/3` is not sufficient
- The first bounded integral-state batch should promote at most one configuration, so the repo keeps one active **正式候选线** instead of turning diagnosis into a parallel candidate race
- A **逐种子硬门槛** prevents one attractive average from hiding a collapsed or task-invalid seed
- A **早期checkpoint失效规则** prevents a fragile early peak from being mistaken for a stable long-budget operating point
- A **方法优于启发式** claim usually requires a focused comparison against a strong heuristic baseline
- A **组件消融验证** claim usually requires a larger experiment matrix than a **科研验证型交付** can safely afford
- A **速度跟踪行走** task can be evaluated under multiple **复杂地形条件**
- In this project, **复杂地形条件** stress-test the method but do not redefine the primary task beyond **速度跟踪行走**
- A **策略局部敏感度** constraint acts inside training as the hard smoothness mechanism
- **动作时间变化率** can still be used as an external metric or heuristic baseline without being the constrained quantity
- A **完全替换对比** isolates the effect of the constrained smoothness method from heuristic reward shaping
- A **PID-Lagrangian正式方案** prioritizes stable constrained training over the simplest possible algorithm definition
- **普通对偶上升** may still appear as a limited ablation, but it is not the target method in this project
- A **平滑优先成功判据** requires an explicit **任务守底线**
- A **任务守底线** must be checked with **奖励无关任务指标** when reward formulations differ
- **总回报补充指标** can still help explain optimization dynamics without serving as the main comparison target
- **行为层平滑指标** should carry the main smoothness conclusion in this project
- **约束层机制证据** explains the constrained method but does not replace behavior-level smoothness evidence
- A **关节震荡主指标** provides the main physical smoothness evidence for this project
- An **动作抖动次级指标** helps explain policy-output behavior but does not replace joint-level evaluation
- A **速度跟踪误差主指标** evaluates whether the locomotion task itself is still being solved
- A **跌倒率底线指标** guards against smooth but unusable policies
- A **复杂地形分层推进** strategy improves failure localization before spending budget on harsher terrain conditions
- A **三组正式对比** separates raw PPO instability from heuristic smoothing trade-offs and constrained smoothing behavior
- `Vanilla PPO` inside a **三组正式对比** is a raw reference rather than a **正式候选线**, so its
  collapse patterns should be recorded rather than filtered away by promotion gates
- The heuristic row inside a **三组正式对比** is a formal comparison anchor rather than a raw
  reference, so it should use the same `3-seed + checkpoint-sweep` evidence strength as the current
  `SC-PPO` mainline
- If the selected heuristic baseline fails to stay task-valid under that `3-seed + checkpoint-sweep`
  standard, the repo should reopen heuristic-anchor selection rather than treating the old single-run
  choice as still report-grade
- If the bounded heuristic family also fails under that frozen `3-seed + checkpoint-sweep`
  standard, the repo should shift into a **协议修复线** rather than keep searching the same family as
  if the missing result were only a candidate-selection problem
- A **MuJoCo必做终验** checks whether simulator-local smoothness improvements translate into more robust cross-engine behavior
- A **MuJoCo关键两组终验** keeps the cross-engine stage focused on the comparison that matters most to the final claim
- A **主实验三种子** standard supports the main claim without exploding the full experiment budget
- A **次级场景降级复现** strategy preserves breadth while keeping the main statistical effort focused
- An **启发式小范围调参基线** makes the main comparison defensible when criticizing heuristic reward shaping
- A **先过底线再取最平滑** rule keeps baseline selection aligned with the project's smoothness-first success criterion
- A **先过终点门槛再取最平滑** rule keeps the current `终点可靠性主线挑战` aligned with final-checkpoint repair rather than checkpoint-sweep rescue
- An **相对退化阈值** makes task-validity checks comparable across methods and terrain difficulty levels
- A **速度误差10%退化上限** operationalizes the main task floor for smoothness-first comparisons
- A **跌倒率5个百分点退化上限** prevents smooth but materially less stable policies from passing
- A **PID有限消融** explains the formal algorithm choice without expanding the project into a full component-attribution study

## Example dialogue

> **Dev:** "这个实施计划首先要做算法框架抽象，还是先跑出可复现实验？"
> **Domain expert:** "这里是 **科研验证型交付**，先确保基线、对比实验和分析闭环，再考虑框架化。"

> **Dev:** "主结论要证明每个组件都有效，还是先证明新方法整体强于启发式？"
> **Domain expert:** "主命题是 **方法优于启发式**，组件层面只做有限消融。"

> **Dev:** "主线证据闭环完成后，能不能同时开 PID、随机阶梯和 SN 三条线？"
> **Domain expert:** "这是 **闭环后支线选择**，一次只推进一条有边界的 **诊断支线**，不能让新支线反向改写主结论。"

> **Dev:** "随机阶梯和粗糙平面是主任务本身，还是同一个任务下的环境条件？"
> **Domain expert:** "主任务是 **速度跟踪行走**，随机阶梯和粗糙平面只是 **复杂地形条件**。"

> **Dev:** "这次硬约束到底约束相邻动作差分，还是约束策略对观测的局部敏感度？"
> **Domain expert:** "训练里的硬约束对象是 **策略局部敏感度**，不是 **动作时间变化率**。"

> **Dev:** "SC-PPO 里还保留 action rate penalty 吗？"
> **Domain expert:** "采用 **完全替换对比**，SC-PPO 移除该惩罚，只保留硬约束。"

> **Dev:** "PID 是正式算法组成部分，还是训练不稳时再补上的增强项？"
> **Domain expert:** "采用 **PID-Lagrangian正式方案**，普通版本最多只做有限消融。"

> **Dev:** "如果 SC-PPO 更平滑，但速度跟踪略差一点，算成功吗？"
> **Domain expert:** "采用 **平滑优先成功判据**，但必须提前写死 **任务守底线**。"

> **Dev:** "既然 reward 定义不同，任务守底线该看什么？"
> **Domain expert:** "看 **奖励无关任务指标**，`episode return` 只能做 **总回报补充指标**。"

> **Dev:** "最终报告里证明更平滑，应该看 Jacobian 还是看动作和关节的时间序列？"
> **Domain expert:** "主证据看 **行为层平滑指标**，Jacobian 只做 **约束层机制证据**。"

> **Dev:** "动作抖动和关节震荡，哪一个作为主平滑指标？"
> **Domain expert:** "主指标是 **关节震荡主指标**，**动作抖动次级指标** 只做补充。"

> **Dev:** "主任务表现应该主要看什么？"
> **Domain expert:** "主任务仍然看 **速度跟踪误差主指标**，**跌倒率底线指标** 只负责守底线。"

> **Dev:** "粗糙平面和随机阶梯要不要从一开始一起跑正式矩阵？"
> **Domain expert:** "采用 **复杂地形分层推进**，先在粗糙平面上调通并完成主对比，再用随机阶梯做压力测试。"

> **Dev:** "正式实验只保留启发式基线和 SC-PPO 两组，还是把原始 PPO 也放进去？"
> **Domain expert:** "采用 **三组正式对比**，否则无法清楚展示原始抖动问题和启发式取舍。"

> **Dev:** "MuJoCo 跨引擎验证是主线必做，还是资源允许时再补？"
> **Domain expert:** "采用 **MuJoCo必做终验**，否则实施计划不闭环。"

> **Dev:** "MuJoCo 终验要不要把原始 PPO 也带进去？"
> **Domain expert:** "采用 **MuJoCo关键两组终验**，只验证启发式基线和 SC-PPO。"

> **Dev:** "所有场景都要多种子吗？"
> **Domain expert:** "采用 **主实验三种子**，其他场景使用 **次级场景降级复现**。"

> **Dev:** "Action Rate Penalty 基线要不要先做一个小范围权重扫描？"
> **Domain expert:** "需要，采用 **启发式小范围调参基线**，否则主结论不公平。"

> **Dev:** "扫描出来的多个启发式权重里，哪一个进入正式对比？"
> **Domain expert:** "采用 **先过底线再取最平滑**，不要再发明一个综合分。"

> **Dev:** "任务守底线该用绝对标准还是相对启发式基线来定？"
> **Domain expert:** "采用 **相对退化阈值**，这是比较研究里更稳的口径。"

> **Dev:** "速度跟踪误差最多允许恶化多少？"
> **Domain expert:** "采用 **速度误差10%退化上限**，这是科研验证型交付里更合理的底线。"

> **Dev:** "那跌倒率怎么守底线？"
> **Domain expert:** "采用 **跌倒率5个百分点退化上限**，和相对比较口径保持一致。"

> **Dev:** "有限消融到底做多少？"
> **Domain expert:** "只做 **PID有限消融**，不要把项目扩成全组件归因研究。"

> **Dev:** "SN-only 诊断连续 collapse 后要不要继续换层、换系数、加种子？"
> **Domain expert:** "当前 **替代机制可行性诊断** 已经是负向结果，不再继续盲目 SN-only 架构开关；未来 SN 必须作为新的 task-stabilized recipe 重新立项。"

> **Dev:** "SN-only 关闭后，随机阶梯能不能接上？"
> **Domain expert:** "可以；它已经作为 **复杂地形条件** 压力测试完成 #7，不能把它写成新主线或反向改写粗糙平面主结论。"

> **Dev:** "随机阶梯压力测试也 collapse 之后，是继续修协议还是先交付？"
> **Domain expert:** "进入 **科研交付冻结**，先把已完成证据、报告、tracker 和复现入口收口；新的 terrain protocol repair 另起支线。"

> **Dev:** "这次冻结要做到论文投稿还是工程发布？"
> **Domain expert:** "都不是；冻结成 **仓库内科研交付包**，把内部证据链、报告、复现命令和 artifact 指针对齐即可。"

> **Dev:** "冻结时要不要把 Isaac/MuJoCo 都重跑一遍？"
> **Domain expert:** "不要；采用 **冻结期轻量验证**。冻结阶段检查测试、路径、JSON 和报告一致性，不生成新实验结果。"

> **Dev:** "复现命令放 README 还是 report？"
> **Domain expert:** "放独立的 **最终复现清单**。README 保持入口简洁，report 保持研究叙事，操作型索引单独维护。"

> **Dev:** "报告要不要按 SN 和随机阶梯重写一遍？"
> **Domain expert:** "不要；新增 **冻结边界章节**，记录 post-mainline diagnostics 和交付边界，不改写主结果叙事。"

> **Dev:** "冻结后的 main 是完全只读，还是还能收少量维护回灌？"
> **Domain expert:** "把 `main` 当作 **冻结主档案分支**。它保持稳定，但仍可吸收两类有限回灌：**冻结边界章节** 更新，以及可复用的评估/诊断基础设施；机制专属实现不要直接回灌。"

> **Dev:** "这轮 `seed11` 的终点平滑回拉门槛到底怎么量化？"
> **Domain expert:** "先用 **关节震荡主指标** 收回同种子 `final -> selected` 差距的至少一半；同时把 **动作抖动次级指标** 和 **速度跟踪误差主指标** 的恶化各压在 `5%` 以内，把 **跌倒率底线指标** 的恶化压在 `+0.05` 以内。"

> **Dev:** "首批积分态诊断是不是只看 final checkpoint 就够？"
> **Domain expert:** "不够；首批要补一个同种子的小 `checkpoint sweep`，至少看 `300 / 400`，如果新 run 的 `final` 超过 `400` 再把 `final` 一并纳入。"

> **Dev:** "如果 `final checkpoint` 没过门槛，但 `300` 或 `400` 过了，能不能算这批成功？"
> **Domain expert:** "不能；这条线修的是 **终点可靠性主线挑战**，所以 `final checkpoint` 必须自己过门槛，小 `checkpoint sweep` 只做诊断解释，不做成功替代。"

> **Dev:** "如果首批 3 个积分态配置里有多个都过门槛，要不要一起升格到 `3-seed`？"
> **Domain expert:** "不要；首批积分态诊断最多只升格一个配置，保持这条线还是有界的 **SC-PPO家族内稳态修复线**，不要立刻膨胀成并行候选线比较。"

> **Dev:** "如果多个积分态配置都过门槛，这一个升格名额该怎么选？"
> **Domain expert:** "采用 **先过终点门槛再取最平滑**：先要求 `final checkpoint` 通过同种子终点门槛，再按 `joint_acceleration_l2_mean` 选最优，`action_jitter_l2_mean` 做第一 tie-break，`velocity_tracking_error_mean` 只做后续 tie-break，不发明新的综合分。"

> **Dev:** "如果首批 3 个积分态配置一个都没过门槛，要不要继续在积分态规则里细扫？"
> **Domain expert:** "不要；这次 **单变量积分态诊断** 就按负结果收口，说明当前积分态规则不是足够强的一阶修复杠杆，下一步应该切到新的单变量杠杆，而不是把最小诊断膨胀成局部超参搜索。"

> **Dev:** "如果积分态最小诊断失败，下一步是继续扫 PID multiplier，还是转到 objective 侧？"
> **Domain expert:** "转到 objective 侧。现有证据已经更支持 **PID放大器假设** 而不是 `PID-specific root cause`，所以失败后的下一条单变量杠杆应优先放在 `constraint target / aggregation rule`，而不是继续盲扫 `pid_ki`。"

> **Dev:** "objective 侧第一杠杆要不要先固定 `threshold = 3.8`，只改 `cost_aggregation`？"
> **Domain expert:** "要。当前仓库已经积累了很多 `threshold` 邻域证据，所以 objective-side 的第一探针应先固定 `threshold = 3.8`，只改 `cost_aggregation`，避免重新膨胀成 threshold neighborhood 搜索。"

> **Dev:** "`cost_aggregation` 第一探针要不要先从 `quantile(0.90)` 改到 `mean`，而不是先试 `max`？"
> **Domain expert:** "要。`max` 会比当前 `quantile(0.90)` 更强调尾部，第一步更容易把这条线重新推向更硬更脆的解；`mean` 更适合作为 objective-side 的第一条单变量探针。"

> **Dev:** "这一步要不要顺手把 `max` 也一起跑成一个小 aggregation batch？"
> **Domain expert:** "不要。当前这一步只跑一个 `mean` 配置；既然已经把第一探针定义成 `quantile(0.90) -> mean`，就不要立刻把最小诊断膨胀成 aggregation sweep。"

> **Dev:** "如果这个唯一的 `mean` 探针也没过门槛，要不要再给 `max` 一个第二顺位机会？"
> **Domain expert:** "不要。`mean` 失败后就把这条 aggregation 最小诊断收口成负结果，不再补一个 `max` 尾探针。"

> **Dev:** "如果 `mean` 探针失败，aggregation 线收口后，要不要直接把下一条 objective-side 单变量杠杆定成 `constraint threshold`？"
> **Domain expert:** "要。既然 aggregation 线已经按最小诊断口径收紧到 `mean` 并在失败时闭合，下一条新问题就应转到另一个 objective-side 主分支，也就是 `constraint threshold`。"

> **Dev:** "`constraint threshold` 线的第一探针，要不要先往更松走，而不是再往更紧走？"
> **Domain expert:** "要。当前仓库里更紧方向已经留下明显的不稳证据，而这条线现在修的是 `final checkpoint` 终点漂移，所以第一反应应是先稍微放松 `constraint target`，而不是继续加压。"

> **Dev:** "这条更松的 `constraint threshold` 线，要不要把现有的 `threshold = 4.0` completed line 直接当作最近的负参考，而把新的 active probe 直接定成 `4.2`？"
> **Domain expert:** "要。`threshold = 4.0` 已经提供了最近的负参考，新预算不必重复在那里；如果这条线真正打开，新的 active probe 直接从 `4.2` 开始。"

> **Dev:** "新的 `4.2` active probe，要不要把其余设置全部拉回当前 `3.8` 主线？"
> **Domain expert:** "要。`4.2` 这条线应保持为干净的单变量 threshold probe：继续使用主线的 `cost_aggregation = quantile(0.90)`、`pid_integral_mode = lower_bound_clamp` 和其余 PID / sampling / lambda 设置，不把 aggregation 线的变动混进来。"

> **Dev:** "把单种子先行都改成三种子并行，应该怎么理解？"
> **Domain expert:** "把它作为全局规则替换：后续 bounded diagnosis 默认从 `11 / 17 / 23` 三种子并行起步，而不是先跑单 seed。各 seed 仍然各自使用同种子回拉参照，当前起始集合是 `seed11 -> checkpoint 300`、`seed17 -> checkpoint 300`、`seed23 -> checkpoint 400`。"

> **Dev:** "在这个新规则下，`4.2` 如果三种子都过门槛，要不要直接视为正式候选线？"
> **Domain expert:** "要。既然最贵的 `3-seed + per-seed checkpoint-sweep` 已经前置完成，那三种子都过门槛时就直接授予 **正式候选线** 身份，不再插一个同质确认阶段。"

> **Dev:** "`4.2` 一旦成为正式候选线，要不要先完成 Isaac 内部挑战，再决定是否进入 `MuJoCo关键两组终验`？"
> **Domain expert:** "要。三种子都过门槛只说明它拿到了候选资格；它仍然应该先在 Isaac 上按同尺口径挑战当前 `3.8` 主线，只有 Isaac 挑战成立后才进入 `MuJoCo关键两组终验`。"

> **Dev:** "这个 Isaac 内部挑战，要不要要求它至少不弱于 `3.8` 的主任务底线，并在终点可靠性目标上给出明确增益？"
> **Domain expert:** "要。这里不要求 `4.2` 在所有共享指标上全面支配 `3.8`；更合理的标准是：`4.2` 继续满足当前终点门槛，不打穿 `velocity_tracking_error_mean` 与 `fall_rate` 的主任务底线，并且在 `final checkpoint` 可信度上给出明确 Isaac-side 增益。"

> **Dev:** "这个 Isaac 内部挑战里，`final checkpoint` 更可信要不要直接定成硬规则：三种子都必须做到 `selected checkpoint = final checkpoint`？"
> **Domain expert:** "要。既然当前主线的问题就是 `final checkpoint 不能直接替代 checkpoint sweep`，那挑战它的新候选线就必须在每个 seed 上做到 `selected checkpoint = final checkpoint`，不能再让更早 checkpoint 扛主结论。"

> **Dev:** "这个硬规则要不要允许 `final checkpoint` 和更早 checkpoint 平台并列？"
> **Domain expert:** "不要。`final checkpoint` 必须是唯一 selected checkpoint；如果还允许与更早 checkpoint 并列，这条线就仍然没有真正摆脱对 checkpoint sweep 的依赖。"

> **Dev:** "这个 Isaac 内部挑战标准，要不要明确写成合取关系？"
> **Domain expert:** "要。对每个 seed 来说，`final checkpoint` 必须同时满足两条：一是继续通过当前的 **双参照终点门槛**，二是成为唯一 selected checkpoint。只满足其中一条都不够构成这条 **终点可靠性主线挑战** 的成功。"

> **Dev:** "这套合取标准，要不要要求 `11 / 17 / 23` 三个 seed `3/3` 全部通过？"
> **Domain expert:** "要。既然入口已经改成 **三种子并行起步**，这里就不能再退回 `2/3` 口径；Isaac 内部挑战必须保持 **逐种子硬门槛**，要求 `3/3` 全过。"

> **Dev:** "如果 `4.2` 在 Isaac 内部挑战里只失败一个 seed，要不要直接把整条 threshold 线判成负结果闭合？"
> **Domain expert:** "要。既然这条线已经明确采用 `3/3` 和 **逐种子硬门槛**，那任一 seed 失败都应把 `constraint threshold` 线按负结果闭合，不再继续开 `4.5` 或修单个坏 seed。"

## Flagged ambiguities

- “实施计划” was ambiguous between **科研验证型交付** and **工程产品型交付** — resolved: this project uses **科研验证型交付**
- “更好的方向” was ambiguous between **同命题主线挑战** and a broader research expansion — resolved: this round chooses **同命题主线挑战**
- “训练稳定性 / 终点可靠性” was ambiguous between generic optimization smoothness and final-checkpoint claim reliability — resolved: this round chooses **终点可靠性主线挑战**
- “先修当前家族还是直接换机制” was ambiguous between **SC-PPO家族内稳态修复线** and a new same-question mechanism line — resolved: this round chooses **SC-PPO家族内稳态修复线**
- “PID 到底是主因还是放大器” was ambiguous between a root-cause diagnosis and a narrower leverage-point diagnosis — resolved: this round chooses **PID放大器假设**
- “这轮证据门槛是不是直接三种子” was ambiguous between formal promotion and a narrower first diagnostic gate — resolved: this round chooses **三种子并行起步门槛**
- “先拿哪个 seed 做最小诊断” was ambiguous among `11 / 17 / 23` — resolved: use a **基准三种子并行探针** and start from the canonical set `11 / 17 / 23`
- “积分态诊断要不要顺手改 threshold” was ambiguous between a clean first probe and a mixed repair bundle — resolved: this round chooses **单变量积分态诊断**
- “首批三种子这轮到底看 final checkpoint collapse 还是看 late-stage 粗糙漂移” was ambiguous between a collapse gate and a stronger drift-repair gate — resolved: this round chooses **终点平滑回拉门槛**
- “终点平滑回拉 先看哪个指标” was ambiguous between equal-weight metrics and the repo's existing smoothness hierarchy — resolved: use **关节震荡主指标** as the primary gate with **动作抖动次级指标** as a guardrail
- “终点平滑回拉 应该对谁回拉” was ambiguous between same-seed and aggregate references — resolved: use a **同种子回拉目标** with the current per-seed anchors `seed11 -> checkpoint 300`, `seed17 -> checkpoint 300`, `seed23 -> checkpoint 400`
- “终点平滑回拉 时任务底线应该看谁” was ambiguous between using the selected checkpoint or the current final checkpoint as the task anchor — resolved: use a **双参照终点门槛**
- “终点平滑回拉门槛 该定多严” was ambiguous between a qualitative improvement story and a numeric diagnostic gate — resolved: the first bounded batch requires at least `50%` recovery of the same-seed final-to-selected `joint_acceleration_l2_mean` gap while keeping `action_jitter_l2_mean` and `velocity_tracking_error_mean` within `+5%` of the current final checkpoint and `fall_rate` within `+0.05`
- “首批积分态诊断 是否只看 final checkpoint” was ambiguous between a final-only read and a bounded late-stage neighborhood check — resolved: require per-seed small `checkpoint sweeps`, at least around each seed's current selected/final neighborhood, and include `final` as well when it lies beyond the current anchor checkpoint
- “小 checkpoint sweep 能不能替 final checkpoint 救场” was ambiguous between a diagnostic side read and a success substitute — resolved: no; the final checkpoint must clear the gate on its own, and the same-seed sweep is diagnostic only
- “首批多个积分态配置同时过门槛后是否并行升格” was ambiguous between parallel promotion and a single bounded candidate handoff — resolved: promote at most one configuration from the first bounded batch
- “多个过门槛积分态配置之间如何选唯一升格者” was ambiguous between composite-score ranking and a smoothness-first terminal rule — resolved: use **先过终点门槛再取最平滑**
- “首批积分态配置全失败后是否继续细扫积分态规则” was ambiguous between bounded negative closure and local hyperparameter expansion — resolved: close the first **单变量积分态诊断** as a negative result and move to a new single-variable lever
- “积分态最小诊断失败后是继续扫 PID 还是转到 objective 侧” was ambiguous between multiplier-side continuation and objective-side follow-up — resolved: move to an objective-side single-variable lever
- “objective 侧第一探针是否先重开 threshold 邻域” was ambiguous between aggregation-only diagnosis and another threshold-neighborhood search — resolved: hold `threshold = 3.8` fixed and change only `cost_aggregation`
- “`cost_aggregation` 第一探针先试 `mean` 还是 `max`” was ambiguous between a milder objective-side shift and a harder tail-focused variant — resolved: first change `quantile(0.90)` to `mean`
- “第一条 aggregation 探针是否顺手把 `max` 一起带上” was ambiguous between a single-config diagnosis and a small aggregation sweep — resolved: run only the `mean` configuration first
- “`mean` 失败后是否再补一个 `max` 探针” was ambiguous between closing the aggregation-side minimal diagnosis and giving tail-focused aggregation a second chance — resolved: do not add `max`; close the aggregation-side minimal diagnosis if `mean` fails
- “aggregation 线失败后下一条 objective-side 单变量杠杆是什么” was ambiguous between internal aggregation follow-ups and moving to the other objective-side branch — resolved: move next to `constraint threshold`
- “`constraint threshold` 线第一步是更松还是更紧” was ambiguous between loosening the target to reduce objective tension and tightening it to chase stronger mid-training smoothness — resolved: start from a looser threshold direction
- “更松的 `constraint threshold` 线第一步是否重跑 `4.0`” was ambiguous between reusing the nearest negative reference and reopening that same point as a fresh probe — resolved: treat completed `threshold = 4.0` as the negative reference and start the new active probe at `4.2`
- “`4.2` active probe 是否混入 aggregation 线改动” was ambiguous between a clean threshold-only probe and a mixed objective-side bundle — resolved: keep all non-threshold settings aligned with the current `3.8` mainline
- “后续 bounded diagnosis 是单种子先行还是三种子并行起步” was ambiguous between the earlier seed11-first path and a repo-wide parallel-first rule — resolved: replace the single-seed-first rule with **三种子并行起步门槛** and **基准三种子并行探针**
- “三种子并行跑完且全部过门槛后是否还要再加同质确认阶段” was ambiguous between direct promotion and a second same-kind confirmation loop — resolved: direct promotion to **正式候选线**
- “`4.2` 取得正式候选线身份后是否立刻进入 MuJoCo” was ambiguous between skipping Isaac-side comparison and preserving the repo's promotion order — resolved: complete the Isaac-side internal challenge first, then decide on `MuJoCo关键两组终验`
- “Isaac 内部挑战是否要求全指标全面支配 `3.8`” was ambiguous between strict domination and a reliability-first challenge rule — resolved: require preserved task floor plus a clear final-checkpoint reliability gain, not all-metric domination
- “Isaac 内部挑战里 `selected checkpoint` 是否必须等于 `final checkpoint`” was ambiguous between a strict final-reliability rule and continued checkpoint-sweep dependence — resolved: require `selected checkpoint = final checkpoint` on all three seeds
- “`selected checkpoint = final checkpoint` 是否允许平台并列” was ambiguous between a soft final-compatibility rule and a strict final-reliability rule — resolved: require the final checkpoint to be the unique selected checkpoint
- “Isaac 内部挑战里的终点门槛与唯一选点关系是二选一还是同时满足” was ambiguous between alternative gates and a conjunctive reliability rule — resolved: require both terminal-gate clearance and unique-final selection on each seed
- “Isaac 内部挑战是 `3/3` 还是允许 `2/3`” was ambiguous between a strict per-seed rule and a majority-pass rule — resolved: require `3/3` all seeds to pass
- “`4.2` 若只失败一个 seed 是否继续扩 threshold 邻域或修坏 seed” was ambiguous between bounded negative closure and neighborhood expansion — resolved: close the `constraint threshold` line as a negative result
- “`constraint threshold` 线也负向闭合后，是否继续停留在家族内局部修补” was ambiguous between extending same-family neighborhood search and closing the whole **SC-PPO家族内稳态修复线** — resolved: close the whole **SC-PPO家族内稳态修复线** as a negative result within the current same-question boundary
- “家族内稳态修复线收口后，是不是立刻把后续工作并到新机制线上” was ambiguous between immediate branch switching and first freezing the failed family as a **同命题负结论边界** — resolved: freeze the negative boundary first, then decide explicitly whether to open a new mechanism line
- “同命题负结论边界 冻结后，若继续方法探索，是否还算旧线延长” was ambiguous between silent continuation and reopening under **替代机制可行性诊断** — resolved: reopen explicitly as a new **替代机制可行性诊断**
- “新的替代机制可行性诊断 第一优先是重开 SN 还是直接跳到完全新家族” was ambiguous between reusing the repo's existing SN evidence base and abandoning it for a fresh family — resolved: reopen first as a `SN` **任务稳定化配方**
- “第一条 `SN` 任务稳定化配方 是否还能从 `no-Jacobian / no-heuristic` 冷启动” was ambiguous between reusing the failed reduced-budget regime and inheriting a proven task-valid base — resolved: inherit a known task-valid training base
- “第一条 `SN` 任务稳定化配方 默认继承哪条 task-valid 底座” was ambiguous between the revised heuristic anchor and the current `SC-PPO 3.8 + PID-Lagrangian` mainline — resolved: inherit the current `SC-PPO 3.8 + PID-Lagrangian` mainline base
- “第一条 `SN` 任务稳定化配方 是否重新叠加 heuristic smoothness reward” was ambiguous between preserving **完全替换对比** and turning the branch into a mixed smoothness recipe — resolved: preserve **完全替换对比**
- “第一条 `SN` 任务稳定化配方 继承主线底座时是否保留 `Jacobian + PID` 训练环路做混合机制” was ambiguous between a pure replacement recipe and a hybrid-enhancement recipe — resolved: inherit only the shared task-stability scaffold and make `SN` the sole active smoothness mechanism
- “第一条 `SN` 任务稳定化配方 里的 `threshold = 3.8` 还算不算训练目标” was ambiguous between carrying over a live training constraint and keeping only a comparable evaluation-side readout — resolved: keep `3.8` only as an evaluation-side `local_sensitivity` threshold
- “第一条 `SN` 任务稳定化配方 的首批可解释证据是单种子 `short` 还是直接三种子并行起步” was ambiguous between the old reduced-budget SN flow and the repo-wide parallel-first rule — resolved: use **三种子并行起步门槛** for interpretable diagnosis and keep `smoke` as implementation self-check only
- “第一条 `SN` 任务稳定化配方 是否同时重开 hidden-only / first-hidden-only / coeff 小 sweep” was ambiguous between recipe-only diagnosis and reintroducing SN parameter search — resolved: start with one canonical full-actor SN configuration only
- “canonical `full-actor SN` 第一配置 是否立刻改成 hidden-only 或 `coeff = 2.0`” was ambiguous between fixing the first recipe probe and reopening structural variants — resolved: keep output-layer SN enabled and keep `actor_spectral_norm_coeff = 1.0`
- “canonical `full-actor SN` 第一批训练预算 是沿用 reduced-budget 预设还是直接对齐主线长预算” was ambiguous between keeping the old SN feasibility presets and removing budget-shortage confounds — resolved: align the first real three-seed batch to the current rough-terrain `512 envs x 400 iterations` budget
- “canonical `full-actor SN` 在三种子长预算下若 `3/3` 过门槛，是停在可行性成立还是直接升格” was ambiguous between introducing a new intermediate tier and reusing the repo's direct-promotion rule — resolved: direct promotion to **正式候选线**
- “canonical `full-actor SN` 的首批三种子长预算过门槛时，是否立刻要求 `selected checkpoint = final checkpoint`” was ambiguous between applying the strict final-reliability rule at entry and preserving the repo's two-stage promotion structure — resolved: first use the current **三种子并行起步门槛** with sweep-as-diagnosis-only, then require `selected checkpoint = final checkpoint` in the later Isaac-side internal challenge
- “canonical `full-actor SN` 的首批三种子长预算若有任一 seed 失败，是否立刻回头开结构修补” was ambiguous between bounded negative closure and immediate SN-neighborhood repair — resolved: close the first canonical SN task-stabilized recipe as a negative result
- “首版 canonical `full-actor SN` 任务稳定化配方 若负向闭合，是否立刻继续开 `SN v2` recipe” was ambiguous between bounded closure and immediate same-family recipe iteration — resolved: close the current SN reopening rather than immediately opening `SN v2`
- “首版 canonical `full-actor SN` 任务稳定化配方 也失败后，`SN` 是否还留在当前同命题近期待办池里” was ambiguous between keeping `SN` warm for near-term retry and freezing it as part of the current **同命题负结论边界** — resolved: freeze the current `SN` reopening into the same-question negative boundary
- “`SN` 负结论边界冻结后，若还继续同命题方法探索，是否还能在 actor-side `SN` 家族附近游走” was ambiguous between silent same-family continuation and an explicit new mechanism-family switch — resolved: switch explicitly to a non-`SN` mechanism family
- “`SN` 冻结后下一条非-`SN` 机制线，是先改约束对象还是先改约束形状” was ambiguous between a direct **动作时间变化率** hard-constraint family and an anisotropic upgrade of the current constrained object — resolved: first choose a **各向异性约束形状线**
- “各向异性约束形状线 的第一步，是直接跳完整 `ALCP` 还是先走 bounded grouped-anisotropic probe” was ambiguous between a full tensorized redesign and a minimally interpretable first entry — resolved: start with a bounded grouped-anisotropic probe
- “bounded grouped-anisotropic probe 是否直接沿用远端 `threshold = 0.55 + positive_part + legacy_guard + proximal_only` 混合口径” was ambiguous between reusing a convenient but mixed probe and isolating the first shape-only change — resolved: pull non-shape settings back to the current `SC-PPO 3.8 + PID-Lagrangian` mainline and let grouped anisotropy be the first new variable
- “bounded grouped-anisotropic probe 的首探是否连 `positive_part / update_error_mode / legacy_guard_mode` 一起回灌” was ambiguous between backporting the whole remote repair stack and backporting only the minimal geometry capability — resolved: first backport only the minimal anisotropic geometry capability
- “proximal-only grouped-anisotropic 首探的权重是否直接改变量纲” was ambiguous between convenient ad hoc weights and scale-preserving shape isolation — resolved: use a total-scale-preserving proximal-only weighting so `threshold = 3.8` stays comparable to the isotropic mainline
- “首个 bounded grouped-anisotropic probe 是否同时重开 threshold 搜索” was ambiguous between shape-only isolation and mixed shape-plus-threshold retuning — resolved: keep `threshold = 3.8` fixed in the first grouped-anisotropic probe
- “首个 bounded grouped-anisotropic probe 是先跑单 seed 短预算还是直接三种子长预算” was ambiguous between a lighter first smoke-like pass and consistency with the repo's current diagnostic rule — resolved: start directly from `11 / 17 / 23` at `512 envs x 400 iterations`
- “首个 bounded grouped-anisotropic probe 若 `3/3` 过门槛，是停在形状可行还是直接升格” was ambiguous between adding a new intermediate tier and reusing the repo's direct-promotion rule — resolved: direct promotion to **正式候选线**
- “首个 bounded grouped-anisotropic probe 过 entry gate 时，是否立刻要求 `selected checkpoint = final checkpoint`” was ambiguous between applying the strict final-reliability rule at entry and preserving the repo's two-stage promotion structure — resolved: first use the current **三种子并行起步门槛** with sweep-as-diagnosis-only, then require `selected checkpoint = final checkpoint` in the later Isaac-side internal challenge
- “首个 bounded grouped-anisotropic probe 若有任一 seed 失败，是否立刻继续开形状邻域修补” was ambiguous between bounded negative closure and immediate anisotropic-neighborhood search — resolved: close the first canonical anisotropic-shape recipe as a negative result
- “首版 各向异性约束形状线 也负向闭合后，是否继续在 **策略局部敏感度** 约束对象内横向游走” was ambiguous between keeping shape-side neighborhood work alive and closing the whole shape-only direction — resolved: close the broader “keep **策略局部敏感度** and only change shape” direction before opening **动作时间变化率硬约束线**
- “首版 **动作时间变化率硬约束线** 是否顺手重调 PID / aggregation / seed protocol” was ambiguous between a clean constrained-object switch and a mixed restart recipe — resolved: keep the current `PID-Lagrangian / quantile(0.90) / lower_bound_clamp / 完全替换对比 / 11-17-23 / 512x400` scaffold and change only the constrained object
- “首版 **动作时间变化率硬约束线** 的阈值起点，是先用较松的 `3.0` 还是先上很硬的 `0.2`” was ambiguous between an interpretable first object-switch probe and an aggressive threshold stress test — resolved: start from the looser implemented `3.0` scale rather than `0.2`
- “首版 **动作时间变化率硬约束线** 是否把原来的 `local_sensitivity = 3.8` 共享读数关掉” was ambiguous between a pure new-object readout and a same-question shared-evidence comparison — resolved: keep `collect_local_sensitivity = true` and preserve `local_sensitivity_threshold = 3.8` as an evaluation-side shared readout
- “首版 **动作时间变化率硬约束线** 的成功门槛，是否改成只看 `action_rate` violation 下降” was ambiguous between a new branch-specific success rule and preservation of the repo's shared same-question gate — resolved: keep the current **三种子并行起步门槛** and shared task floor, and treat `action_rate` reduction only as diagnostic evidence
- “首版 **动作时间变化率硬约束线** 若 `3/3` 过门槛，是否还要再插一个动作时间变化率专属确认阶段” was ambiguous between direct promotion and branch-specific extra confirmation — resolved: promote directly to **正式候选线**
- “首版 **动作时间变化率硬约束线** 的 entry gate，是否允许 `checkpoint sweep` 诊断但不能拿更早 checkpoint 救 `final checkpoint`” was ambiguous between preserving the repo's current two-stage structure and inventing a branch-specific final-selection shortcut — resolved: keep sweep-as-diagnosis-only at entry, and reserve `selected checkpoint = final checkpoint` for the later Isaac-side internal challenge
- “首版 **动作时间变化率硬约束线** 若有任一 seed 失败，是否立刻继续开 `2.5 / 2.0 / 0.2` 阈值邻域修补” was ambiguous between bounded negative closure and immediate threshold search — resolved: close the first canonical action-rate recipe as a negative result
- “首版 **动作时间变化率硬约束线** 也负向闭合后，是否继续在 `action_rate` 约束对象内横向游走” was ambiguous between keeping object-replacement work alive and closing the whole action-rate direction — resolved: close the broader “replace the constrained object with **动作时间变化率**” direction before opening another truly different mechanism family
- “在当前 shape/object 两条 hard-constraint 线都收口后，下一条 truly different 机制线是否还留在同命题低层平滑框架里” was ambiguous between continuing the same question and jumping directly to broader frontier work — resolved: stay inside the current same-question low-level smoothness frame before any move to `SysID`, `Residual RL`, perception, or `VLA`
- “在当前 shape/object 两条 hard-constraint 线都收口后，下一条 truly different 机制线是否还继续 current hard-constraint 家族” was ambiguous between another nearby target-plus-dual variant and a real mechanism-form change — resolved: change the mechanism form itself rather than reopen another hard-constraint family
- “在当前 shape/object 两条 hard-constraint 线都收口后，下一条 truly different 机制线是否优先落到 **架构级平滑优化线**” was ambiguous between a same-question architectural line and reopening broader frontier work — resolved: choose **架构级平滑优化线** as the preferred next mechanism line
- “**架构级平滑优化线** 的首步是否还能是隐性的 `SN v3` 续命” was ambiguous between a genuine new actor-side mechanism and a relabeled reopening of the failed `SN` family — resolved: it must start from an actor-side mechanism not already covered by the repo's current `SN` negative boundary
- “**架构级平滑优化线** 的训练主机制是否还能继续依赖当前 `Jacobian penalty / double backward` 路径” was ambiguous between a real mechanism-form replacement and a Jacobian-supported actor-side variant — resolved: the first entry must replace the Jacobian path as the active training-side smoothness mechanism
- “**架构级平滑优化线** 的首步是否也按 **任务稳定化配方** 来开” was ambiguous between a task-valid replacement recipe and either a `Jacobian + new mechanism` hybrid or a `no-Jacobian / no-heuristic` cold-start — resolved: open it as a **任务稳定化配方** with the new actor-side mechanism as the sole active smoothness mechanism
- “`Adaptive Action Scaling` 是否算 **架构级平滑优化线** 的首步候选” was ambiguous between a general replacement mechanism and an actor-side architectural constraint — resolved: exclude it from this line's first-entry candidate set under the current glossary
- “在 `SN` 与 adaptive scaling 都被排除后，**架构级平滑优化线** 的 repo 内首版 candidate 该如何具体命名并落成可执行对象” was ambiguous between继续停留在空术语 and defining an explicit actor-side first entry — resolved: define **正交低增益 actor 线** as a bounded repo-internal first-entry candidate, using actor-layer orthogonal parametrization plus fixed bounded actor output gain, with the Jacobian/double-backward path removed from the active smoothness mechanism
- “**正交低增益 actor 线** 的 formal negative result 是否主要只是 `output_gain = 0.50` 过低导致” was ambiguous between a low-gain confound and a mechanism-level failure — resolved: no; the gain-isolation probe that raises `output_gain` to `1.00` still fully collapses on the canonical rough-terrain `11 / 17 / 23`, `512 envs x 400 iterations` entry, with all three seeds at `fall_rate = 1.0`
- “在 orthogonal actor canonical entry 与 gain-isolation follow-up 都负向后，**架构级平滑优化线** 是否仍保留 live repo 内首版 candidate” was ambiguous between keeping a nearby `gain` repair alive and closing the bounded first entry — resolved: no; close the broader orthogonal-actor first-entry candidate and return **架构级平滑优化线** to ‘no live repo-internal first-entry candidate’ until a different actor-side mechanism is explicitly named
- “在 orthogonal actor bounded candidate 关闭后，下一条 actor-side first-entry candidate 应该具体命名为什么” was ambiguous between继续停留在‘无 live candidate’状态 and explicitly naming the next architecture-line entry — resolved: name **LayerNorm actor 线** as the next bounded first-entry candidate, using actor hidden-layer activation normalization and keeping `actor_output_gain = 1.00` in the first canonical recipe
- “首版 canonical **LayerNorm actor 线** recipe 在真实 `11 / 17 / 23`、rough-terrain `512 envs x 400 iterations` 下是否已经过关并可直接升格” was ambiguous between a passed same-question replacement result and a partial-but-insufficient rescue — resolved: no; selected-checkpoint aggregate `fall_rate_mean = 0.8667`, `seed11` remains `all_checkpoints_collapsed`, and the final-checkpoint aggregate still sits at `fall_rate_mean = 0.7000`
- “既然 **LayerNorm actor 线** 不再像 orthogonal actor 一样全 collapse，是否应直接把它视为新的 live architecture solution 或扩成局部 sweep” was ambiguous between a successful mechanism replacement and a bounded repair opening — resolved: no; keep the line open only as a bounded repair candidate, with the next minimal follow-up being a lower `actor_output_gain` such as `0.75` rather than a broader architecture neighborhood sweep
- “是否要显式扩 glossary，新建一个不要求 actor-side architectural constraint 的替代机制术语” was ambiguous between forcing all same-question replacements into **架构级平滑优化线** and opening a new same-question term for non-architectural mechanism changes — resolved: add **非架构型替代机制线**
- “新 glossary 里的第一条 active line 是否直接定名为 **动作缩放替代机制线**” was ambiguous between a generic placeholder and a first concrete non-architectural replacement line — resolved: name the first active line **动作缩放替代机制线**
- “**动作缩放替代机制线** 的第一步是否也按 **任务稳定化配方** 来开” was ambiguous between a task-valid replacement recipe and either a `Jacobian + 动作缩放` hybrid or a `no-Jacobian / no-heuristic` cold-start — resolved: open it as a **任务稳定化配方** with constraint-aware action-side scaling as the sole active smoothness mechanism, while preserving the current scaffold, **完全替换对比**, canonical `11 / 17 / 23`, and `512 envs x 400 iterations`
- “**动作缩放替代机制线** 的第一步是否继续保留 evaluation-side `collect_local_sensitivity = true`，并把 `local_sensitivity_threshold = 3.8` 当作共享副读数” was ambiguous between keeping the shared evidence尺子 and redefining the readout around the new training mechanism — resolved: keep the current `local_sensitivity` evaluation readout as a shared side channel under **同尺比较**
- “**动作缩放替代机制线** 的第一步成功门槛是否继续沿用当前共享的 **三种子并行起步门槛** 和 task-validity floor” was ambiguous between using the repo-wide entry gate and inventing a scaling-specific pass rule — resolved: keep the shared three-seed entry gate rather than judging the branch by scaling activation, clipping rate, or standalone constraint reduction alone
- “**动作缩放替代机制线** 的第一步是否允许 per-seed checkpoint sweep 只做诊断副读数、但不能救失败的 `final checkpoint`” was ambiguous between diagnostic side reads and an early-checkpoint rescue rule — resolved: allow checkpoint sweeps only as side diagnosis under the current gate, not as a rescue for failed final checkpoints
- “**动作缩放替代机制线** 的第一步若在三种子入口预算下实现 `3/3` 过关，是否直接升格为 **正式候选线**” was ambiguous between using the shared promotion rule and adding an action-scaling-specific confirmation round — resolved: direct `3/3` success at entry budget earns **正式候选线** status
- “**动作缩放替代机制线** 的第一步若任一 seed 没过当前 entry gate，是否先直接负向闭合首版 canonical recipe” was ambiguous between bounded closure and immediate neighborhood repair in `scale schedule / clipping shape / gain` — resolved: close the first canonical action-scaling recipe rather than immediately wandering within local action-scaling neighborhoods
- “首版 canonical **动作缩放替代机制线** recipe 若负向闭合，是否同时关闭更宽的动作/输出缩放替代方向” was ambiguous between bounded family closure and keeping nearby action-scaling variants warm for near-term retry — resolved: close the broader constraint-aware action-side scaling replacement direction within the current same-question boundary
- “名字叫 **动作缩放替代机制线**，其首版 canonical recipe 是否仍允许 `output-side scaling` 或 `action + output` 混合也算入第一步” was ambiguous between a tightly bounded first recipe and a mixed entry definition — resolved: the first canonical recipe is action-side scaling only
- “既然名字已定为 **动作缩放替代机制线**，其更宽定义是否仍包含 `output-side scaling`” was ambiguous between a narrowly named action-side family and a broader mixed action/output family — resolved: the line itself is also action-side scaling only
- “既然 **动作缩放替代机制线** 已收紧为 action-side family，是否显式新增一条 **输出缩放替代机制线** 来承接 `output-side scaling`” was ambiguous between leaving output-side variants unnamed and giving them a parallel future-line term under **非架构型替代机制线** — resolved: add **输出缩放替代机制线** as a named parallel future line
- “**输出缩放替代机制线** 若将来开启第一步，是否也按 **任务稳定化配方** 来开” was ambiguous between a task-valid replacement recipe and an output-side-specific cold start — resolved: start it later from the current `SC-PPO 3.8 + PID-Lagrangian` scaffold with **完全替换对比** and the canonical `11 / 17 / 23` at `512 envs x 400 iterations`
- “首版 canonical **动作缩放替代机制线** recipe 在真实 `11 / 17 / 23`、rough-terrain `512 envs x 400 iterations` 下是否过关” was ambiguous between a still-pending branch and a completed same-question replacement result — resolved: no; the canonical batch closed negatively, all three seeds selected the final `checkpoint = 400`, aggregate `fall_rate_mean = 0.3667`, and the mechanism saturated at `lagrange_multiplier = 5.0` and `action_scale = 0.5` while `policy_local_sensitivity_cost_update` stayed far above `threshold = 3.8`
- “当前 repo 内 `action_scaling_ppo` prototype 的语义是否已经足够等同于纯 `action-side scaling`，从而可把这次结果同时记成纯动作侧与纯输出侧两条线的闭合” was ambiguous between a clean post-policy action hook and the current actor-output-mean scaling implementation — resolved: no; the current prototype scales actor output mean while leaving exploration `std` unchanged, so this result should be read as a negative closure of the current prototype recipe and the bounded action-side line, not as a silent closure of **输出缩放替代机制线**
- “首版 canonical **输出缩放替代机制线** recipe 在真实 `11 / 17 / 23`、rough-terrain `512 envs x 400 iterations` 下是否过关” was ambiguous between a still-pending future line and a completed same-question replacement result — resolved: no; the canonical batch closed negatively, all three seeds selected the final `checkpoint = 400`, aggregate `fall_rate_mean = 0.4333`, one canonical seed remained outside the current task-valid floor (`seed17 fall_rate = 0.85`), and the mechanism saturated at `lagrange_multiplier = 5.0` and `output_scale = 0.5` while `policy_local_sensitivity_cost_update` stayed far above `threshold = 3.8`
- “首版 canonical **输出缩放替代机制线** recipe 虽然 aggregate smoothness 好于 canonical **动作缩放替代机制线**，是否就足以保留 output-side scaling 邻域作为近期开口” was ambiguous between a partial smoothness gain and a passed shared entry gate — resolved: no; bounded same-question judgment still follows the shared three-seed entry gate and current task-valid floor, so this line closes negatively rather than reopening nearby `output-scale gain/floor/schedule` repair
- “核心命题” was ambiguous between **方法优于启发式** and **组件消融验证** — resolved: the primary claim is **方法优于启发式**
- “主线闭环后先做什么” was ambiguous between multi-branch expansion and **闭环后支线选择** — resolved: choose one bounded follow-up branch at a time
- “主任务” was ambiguous between **速度跟踪行走** and **复杂地形通用运动** — resolved: the primary task is **速度跟踪行走**
- “平滑性约束对象” was ambiguous between **策略局部敏感度** and **动作时间变化率** — resolved: the hard constraint targets **策略局部敏感度**
- “SC-PPO 是否保留动作差分惩罚” was ambiguous between mixed shaping and **完全替换对比** — resolved: use **完全替换对比**
- “拉格朗日乘子更新是否以 PID 为正式方案” was ambiguous between **PID-Lagrangian正式方案** and **普通对偶上升** — resolved: use **PID-Lagrangian正式方案**
- “成功定义” was ambiguous between all-metrics domination and **平滑优先成功判据** — resolved: use **平滑优先成功判据**
- “任务表现主判据” was ambiguous between reward-based and **奖励无关任务指标** — resolved: use **奖励无关任务指标**
- “平滑性主证据” was ambiguous between behavior-level and constraint-level evidence — resolved: use **行为层平滑指标** as the primary conclusion basis
- “主平滑指标” was ambiguous between action-level and joint-level metrics — resolved: use **关节震荡主指标** with **动作抖动次级指标**
- “主任务指标” was ambiguous between tracking-centric and survival-centric evaluation — resolved: use **速度跟踪误差主指标** with **跌倒率底线指标**
- “复杂地形如何进入主实验” was ambiguous between parallel expansion and **复杂地形分层推进** — resolved: use **复杂地形分层推进**
- “正式对比组数量” was ambiguous between direct two-way comparison and **三组正式对比** — resolved: use **三组正式对比**
- “跨引擎验证地位” was ambiguous between optional bonus work and **MuJoCo必做终验** — resolved: use **MuJoCo必做终验**
- “MuJoCo 终验范围” was ambiguous between three-group replay and **MuJoCo关键两组终验** — resolved: use **MuJoCo关键两组终验**
- “统计强度” was ambiguous between single-seed evidence and **主实验三种子** — resolved: use **主实验三种子** with **次级场景降级复现**
- “启发式基线公平性” was ambiguous between a fixed penalty weight and an **启发式小范围调参基线** — resolved: use **启发式小范围调参基线**
- “启发式基线入选规则” was ambiguous between a composite score and **先过底线再取最平滑** — resolved: use **先过底线再取最平滑**
- “任务守底线形式” was ambiguous between absolute thresholds and **相对退化阈值** — resolved: use **相对退化阈值**
- “速度跟踪退化容忍度” was ambiguous between 5% and **速度误差10%退化上限** — resolved: use **速度误差10%退化上限**
- “跌倒率底线形式” was ambiguous between an absolute survival gate and **跌倒率5个百分点退化上限** — resolved: use **跌倒率5个百分点退化上限**
- “有限消融范围” was ambiguous between PID-only and broader component ablations — resolved: use **PID有限消融**
- “Vanilla PPO 三种子是否适用正式候选线门槛” was ambiguous between promotion-gated filtering and raw-reference recording — resolved: treat `Vanilla PPO` as a raw reference and record collapse rather than rejecting it through candidate-promotion rules
- “启发式锚点三种子是否只需单次可用结果” was ambiguous between a loose baseline check and formal comparison evidence — resolved: treat the selected heuristic baseline as a formal comparison anchor and require `3-seed + checkpoint-sweep` evidence
- “启发式锚点三种子失稳后是否仍可直接进入 #5” was ambiguous between accepting the old single-run anchor and reopening baseline selection — resolved: if the selected heuristic baseline fails the formal `3-seed + checkpoint-sweep` standard, reopen heuristic-anchor selection rather than counting `#5` as complete
- “bounded heuristic family 全失败后下一步是什么” was ambiguous between continuing the same search and starting a **协议修复线** — resolved: if the bounded heuristic family all fails under the frozen `3-seed + checkpoint-sweep` regime, shift the repo to a baseline-side **协议修复线** and make any regime revision explicit
- “MuJoCo 对齐 revised heuristic anchor 后怎么写” was ambiguous between **部分迁移结论** and **混合外部验证结论** — resolved: use **混合外部验证结论** when aligned replay does not preserve the Isaac-side `SC-PPO` over heuristic ordering
- “SN-only 替代机制诊断失败后是否继续扩大结构开关” was ambiguous between more blind SN variants and closing the current diagnostic — resolved: close the current SN-only branch as a negative **替代机制可行性诊断**; future SN work needs a separate task-stabilized recipe
- “SN-only 关闭后的下一条支线” was ambiguous between reopening method work and terrain stress testing — resolved: select #7 as a bounded `随机阶梯` **复杂地形条件** stress test while preserving the rough-terrain main claim; that stress test is now closed as selected-checkpoint transfer failure
- “随机阶梯 selected-checkpoint transfer 失败后下一步是什么” was ambiguous between more protocol repair and delivery closure — resolved: enter **科研交付冻结** before opening any new terrain-side **协议修复线**
- “科研交付冻结 的交付对象是什么” was ambiguous between a paper-submission package, an engineering product release, and an internal research handoff — resolved: produce a **仓库内科研交付包**
- “科研交付冻结 是否需要重跑实验” was ambiguous between re-validation by recomputation and consistency checking — resolved: use **冻结期轻量验证** and do not generate new formal evidence
- “复现命令放在哪里” was ambiguous between README, report body, and an independent operational index — resolved: create a standalone **最终复现清单**
- “冻结时 report 是否需要重写” was ambiguous between rewriting the research narrative and appending delivery boundaries — resolved: add a **冻结边界章节** only
- “冻结后的 main 是否完全只读” was ambiguous between zero-backport archival freeze and bounded maintenance — resolved: treat `main` as a **冻结主档案分支**
