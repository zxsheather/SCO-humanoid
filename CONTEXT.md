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
An experiment branch that has earned promotion from a **诊断支线** and is allowed to consume `主实验三种子` and `MuJoCo关键两组终验` budget to challenge the current mainline.
_Avoid_: 一次性探针, 已定稿主线

**主线证据闭环**:
A planning stage where the repo first freezes the claim boundary, citation set, and external-validation reading of the current mainline before opening a new method branch.
_Avoid_: 结论未收口就开新坑, 把前沿方向池直接当执行计划

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
A post-mainline diagnostic branch that tests whether an actor-side architectural constraint can replace the current Jacobian-penalty path without expanding the task definition.
_Avoid_: 直接改成更大研究命题, 把效率优化和能力外扩混成一条线

**替代机制可行性诊断**:
A diagnostic branch stage that first asks whether a new smoothness mechanism can train, activate, and yield interpretable evidence before it is judged against the current mainline.
_Avoid_: 零实现就直接打主线, 一上来就烧正式预算

**同尺比较**:
A comparison rule where a new mechanism is trained differently if needed, but is still evaluated under the repo's existing shared metric and constraint-evidence schema.
_Avoid_: 机制一换就连评估尺子一起换, 比较关系失真

**远期方向池**:
A set of plausible future research directions that remain documented but are not yet promoted into execution.
_Avoid_: 当前执行计划, 默认下一步都要做

**正式候选线升格门槛**:
The minimum evidence a **诊断支线** must clear before it is allowed to consume formal replication and cross-engine evaluation budget.
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
- A **主线证据闭环** should happen before the repo opens a materially broader post-mainline exploration branch
- A **协议修复线** exists to fix evaluation semantics or protocol quality without being mistaken for a new algorithm claim
- A **部分迁移结论** lets the repo include external validation in the main report without overstating mixed evidence
- A **混合外部验证结论** should replace a **部分迁移结论** when aligned cross-engine replay does not preserve the Isaac-side method ordering
- A **架构级平滑优化线** should stay inside the current locomotion claim and first test mechanism replacement rather than task expansion
- A **替代机制可行性诊断** should precede any formal mainline challenge from a zero-implementation method branch
- A **同尺比较** keeps mechanism replacement interpretable by preserving the repo's existing evidence chain
- A **远期方向池** records broader ideas such as compliance, sim-to-real, or perception without forcing immediate execution
- A **正式候选线升格门槛** determines when a **诊断支线** may challenge the current mainline with formal budget
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
- An **相对退化阈值** makes task-validity checks comparable across methods and terrain difficulty levels
- A **速度误差10%退化上限** operationalizes the main task floor for smoothness-first comparisons
- A **跌倒率5个百分点退化上限** prevents smooth but materially less stable policies from passing
- A **PID有限消融** explains the formal algorithm choice without expanding the project into a full component-attribution study

## Example dialogue

> **Dev:** "这个实施计划首先要做算法框架抽象，还是先跑出可复现实验？"
> **Domain expert:** "这里是 **科研验证型交付**，先确保基线、对比实验和分析闭环，再考虑框架化。"

> **Dev:** "主结论要证明每个组件都有效，还是先证明新方法整体强于启发式？"
> **Domain expert:** "主命题是 **方法优于启发式**，组件层面只做有限消融。"

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

## Flagged ambiguities

- “实施计划” was ambiguous between **科研验证型交付** and **工程产品型交付** — resolved: this project uses **科研验证型交付**
- “核心命题” was ambiguous between **方法优于启发式** and **组件消融验证** — resolved: the primary claim is **方法优于启发式**
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
