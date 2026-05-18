# **人形机器人底层动力学受限优化与平滑性增强算法验证及前沿演进探索报告**

随着深度强化学习在足式机器人领域的广泛应用，人形机器人的底层动力学控制正在经历从传统的基于模型预测控制向端到端学习策略的范式转变。然而，在实际部署中，标准的近端策略优化算法（Proximal Policy Optimization, PPO）暴露出显著的局限性。由于仿真环境通常假设执行器具有无限带宽，未经约束的 PPO 策略倾向于利用高频扭矩抖动来最大化任务奖励，导致输出极不稳定。传统上，研究人员通过修改奖励函数（如添加动作差分奖励或扭矩惩罚）来抑制这种现象，但这种奖励塑形（Reward Shaping）方法不仅耗时费力，而且会在追踪目标（如线速度）与平滑性之间产生相互冲突的梯度，导致策略陷入次优局部解，甚至在面对未知地形时丧失鲁棒性 1。

为了从根本上解决这一痛点，将底层控制转化为受限马尔可夫决策过程（Constrained Markov Decision Process, CMDP）成为了当前最具潜力的研究方向。本报告基于在 Humanoid-Gym 基础训练管道中结合 ECO（Energy-Constrained Optimization）与 LCP（Lipschitz-Constrained Policies）思想的初始实施路径，深度剖析 Smoothed-Constrained-PPO 算法的理论框架与验证设计。此外，针对“不止于此”的进一步研发需求，本报告广泛系统地梳理了当前人形机器人学习的前沿工作，从计算效率优化、各向异性柔顺控制、感知运动集成，一直延伸至基于视觉-语言-动作模型（Vision-Language-Action, VLA）的全身协调操作（Whole-Body Loco-Manipulation），为下一阶段的算法演进提供详尽的战略蓝图。

## **理论重构：Smoothed-Constrained-PPO 算法的数学机理**

在构建 Smoothed-Constrained-PPO 算法的过程中，核心机制的转变在于摒弃将能量消耗或平滑性作为奖励惩罚项的传统做法，转而利用拉格朗日主对偶更新机制与利普希茨连续性（Lipschitz Continuity）梯度惩罚，将平滑性转化为受限优化方程中的硬性不等式约束。这一重构不仅改变了神经网络的优化轨迹，也为策略在物理硬件上的可部署性提供了理论保证。

### **引入 PPO-Lagrangian 主对偶更新机制**

在标准的无约束强化学习中，辅助惩罚（如动作速率惩罚）被线性组合到全局奖励函数中，这就要求开发者在数十个相互博弈的奖励权重之间进行极为漫长的超参数调优 1。ECO 框架的核心贡献在于其通过 CMDP 将任务目标与系统物理限制解耦，使得策略可以在最大化任务奖励的同时，严格满足特定的预算约束 1。

将 ECO 框架中的 PPO-Lagrangian 主对偶更新机制移植到 Humanoid-Gym 后，优化目标被重新定义。系统旨在最大化期望回报，同时受到成本约束的限制。其拉格朗日函数可表示为将主任务回报与动态惩罚项结合的形式，其中拉格朗日乘子作为惩罚系数在训练过程中自适应调整 3。在主更新步骤中，策略网络参数通过梯度上升最大化拉格朗日目标，并保留 PPO 固有的裁剪（Clipping）机制以确保信任域内的稳定更新；而在对偶更新步骤中，拉格朗日乘子通过梯度下降进行更新。如果策略违反了约束阈值，拉格朗日乘子会相应增加，迫使策略网络在后续迭代中重新回到可行域内；反之，若约束得到满足，乘子则会衰减，从而释放优化空间以进一步提升任务表现 3。这种机制彻底消除了手动调节平滑性奖励权重的繁琐过程。

### **利普希茨连续性作为硬性不等式约束**

在明确了优化框架后，约束项的物理意义定义至关重要。原有 ECO 框架将能量消耗作为约束，而本研究方向则创新性地将 LCP 梯度惩罚项作为硬性约束条件 1。LCP 的核心思想是约束策略网络输出对状态输入的局部敏感度。在数学上，如果一个策略函数是利普希茨连续的，那么其输出动作对状态输入的导数范数将被限制在一个常数界限内 2。

在强化学习的背景下，如果策略对微小的状态扰动（如传感器噪声或地形微小不平整）做出剧烈的非线性动作响应，就会在物理执行器上表现为高频振荡 11。通过在拉格朗日成本函数中引入 LCP 梯度惩罚，Smoothed-Constrained-PPO 将策略的利普希茨常数限制在一个预设的物理带宽阈值之内 13。这就相当于在算法底层构建了一个数学意义上的低通滤波器，从根本上阻断了高频控制信号的产生，不仅实现了极其平滑的步态，还大幅降低了物理硬件在执行过程中的磨损率 9。

## **实证评估体系：Humanoid-Gym 管道与 Sim-to-Sim 验证**

为确保 Smoothed-Constrained-PPO 算法不仅在理论上自洽，且在实际复杂环境中具备优越性，必须在 Humanoid-Gym 框架内建立一套超越单一“累积奖励”的严密评估体系，并利用多物理引擎的差异性进行深度验证。

### **复杂地形下的平滑性与鲁棒性对标设计**

在 Isaac Gym 提供的高度并行化仿真环境中，实证设计的关键在于构建足以激发高频动作的地形分布。随机阶梯（Random Stairs）与粗糙平面（Rough Planes）是评估底层平滑性算法的最佳试金石。在这些地形中，由于足端接触状态的极度非线性与高频跳变，传统添加动作差分奖励（Action Rate Penalty）的 PPO 算法往往会陷入两难：若惩罚过高，机器人会表现出僵硬的保守步态，难以跨越阶梯；若惩罚过低，机器人则会产生严重的足端滑移与关节高频抖动 3。

通过在这些地形上横向对比传统算法与 Smoothed-Constrained-PPO，可以清晰地揭示新算法在收敛曲线上的优势。由于拉格朗日机制动态调节约束压力，Smoothed-Constrained-PPO 在训练初期允许一定程度的探索，而在收敛期则能牢牢锁死动作的平滑度方差，展现出更稳定的策略进化曲线 3。

为了科学量化步态平滑度，评估不应仅局限于关节加速度方差，而应引入运动生物力学领域公认的无量纲平滑度指标。例如，对数无量纲急位（Log Dimensionless Jerk, LDLJ）和谱弧长（Spectral Arc Length, SPARC）。LDLJ 通过对加速度的导数（即急位）进行积分并标准化处理，能够精准反映控制信号中的高频次级运动；而 SPARC 则在频域内评估速度分布，对运动幅度和持续时间的变化具有极强的鲁棒性 17。此外，衡量受限强化学习算法效率的核心指标——回报成本比（Return-to-Cost Ratio, RC Ratio）也必须被纳入评估，以此证明策略在严格满足 LCP 约束的同时，并未牺牲前向移动速度与跨越障碍的能力 4。

### **跨物理引擎验证的底层逻辑**

在 Isaac Gym 完成初步训练与评估后，利用 Humanoid-Gym 自带的 Sim-to-Sim 脚本将其部署至 MuJoCo 物理引擎进行验证，是确保算法泛化能力的关键环节 21。这种验证并非单纯的软件迁移，而是对算法物理真实性的极端压力测试。

Isaac Gym 在底层接触动力学解算上，倾向于使用基于惩罚的方法（Penalty-based methods）或利用最大坐标系以实现极高的 GPU 并行计算效率。这种机制在处理多刚体接触时，容易在足底与地面之间产生微小的几何穿透（Micro-penetrations）与接触力伪影 23。未受物理规律严格约束的策略极易“过拟合”这些仿真漏洞，表现为在 Isaac Gym 中行走完美，却在实际部署中失效。相对而言，MuJoCo 基于广义坐标系，并使用严格的线性互补问题（Linear Complementarity Problem, LCP）求解器来处理接触动力学，强制保证了绝对的非穿透约束与库仑摩擦锥条件 24。

如果一种基于动作差分奖励的传统策略在 MuJoCo 中出现频繁摔倒或步态退化，即说明其平滑性是建立在利用特定物理引擎缺陷的基础上的。而 Smoothed-Constrained-PPO 由于在网络输出层面强制执行了利普希茨连续性，其生成的扭矩指令本身即符合物理执行器的低通特性。因此，这种策略在从 Isaac Gym 迁移至 MuJoCo 时，其 SPARC 与 LDLJ 指标将表现出高度的一致性与极低的性能衰减，从而在形成算法研究报告时，提供极具说服力的“仿真无关性”（Simulator-agnostic）证据 9。

| 评估维度 | 指标/验证方法 | 理论依据与测试目的 |
| :---- | :---- | :---- |
| **受限优化效率** | 回报成本比 (RC Ratio) | 量化主任务回报与 LCP 违反成本的平衡关系，验证拉格朗日更新的收敛性 4。 |
| **动作域平滑度** | 谱弧长 (SPARC) / 对数无量纲急位 (LDLJ) | 提供独立于幅度的无量纲评估，精准捕捉步态中的有害高频次级运动 17。 |
| **仿真物理无关性** | MuJoCo Sim-to-Sim 零样本测试 | 利用 MuJoCo 严格的 LCP 接触求解器，剔除策略对 Isaac Gym 穿透伪影的过拟合依赖 21。 |

## **突破瓶颈：计算效率优化与各向异性柔顺控制**

在实现并验证了 ECOLab 基座的改进算法后，“不止于此”的下一步必须直面该算法架构在可扩展性与物理柔顺性上的固有瓶颈。LCP 梯度惩罚在带来极致平滑性的同时，也引入了巨大的系统损耗，亟待通过前沿方法进行升级。

### **谱归一化（Spectral Normalization）替代梯度惩罚**

Smoothed-Constrained-PPO 的最大计算瓶颈在于梯度惩罚项的求解。为了计算动作输出相对于状态输入的导数范数，深度学习框架必须在反向传播过程中执行计算量巨大的双重求导（Double Backward Pass）。在 Isaac Gym 动辄数千个并行环境的张量运算中，这种操作会极大地吞噬 GPU 显存，迫使研究人员大幅缩减环境并行数量，从而成倍延长策略训练的时间周期，阻碍算法的快速迭代 11。

为了在保持利普希茨约束的同时恢复计算效率，引入谱归一化（Spectral Normalization, SN）技术是极为关键的下一步演进。谱归一化最初在生成对抗网络（GANs）领域被广泛用于稳定鉴别器的训练，其核心数学原理是通过对神经网络每一层的权重矩阵进行缩放，将其最大奇异值（即谱范数）强制约束为 1 10。

在强化学习的执行器网络（Actor Network）中应用谱归一化，意味着策略函数成为一系列 1-利普希茨连续层的嵌套。由于 1-利普希茨函数的复合依然是 1-利普希茨的，整个策略网络从架构层面上被内在地约束为全局利普希茨连续，而无需在损失函数中显式地计算任何梯度惩罚项 10。这种将“软惩罚优化”转化为“硬架构约束”的范式跃迁，不仅彻底免除了双重求导带来的计算灾难，使得 GPU 能够满载运行庞大的环境矩阵，同时其在真实硬件上抑制高频波动的效果与传统梯度惩罚方法高度一致，是突破现有计算瓶颈的终极方案 11。

### **各向异性利普希茨受限策略（ALCP）的探索**

现有的 LCP 以及上述的谱归一化方法，在本质上都是各向同性的（Isotropic），即它们对整个状态-动作映射空间施加了一个统一的标量利普希茨预算 28。然而，人形机器人作为一个具有极高自由度的多体动力学系统，其不同关节在不同任务中对柔顺性（Compliance）的要求截然不同。例如，在粗糙平面上行走时，踝关节为了保持横向平衡需要极高的刚度（即允许动作的快速变化），而手臂或躯干关节在遇到外界冲击时则需要较高的柔顺度以吸收能量 28。使用单一的平滑度阈值会削弱机器人的整体动态敏捷性。

未来的算法升级应探索各向异性利普希茨受限策略（Anisotropic Lipschitz-Constrained Policies, ALCP）。ALCP 的创新之处在于，它不再使用单一标量约束，而是将任务空间的刚度上限映射为一个状态相关的、作用于策略雅可比矩阵的利普希茨张量约束 28。在实现上，这一机制通过一个基于物理约束椭球体导出的铰链平方谱范数惩罚项（Hinge-squared Spectral-norm Penalty）来进行优化 28。将 ALCP 思想融合进底层控制架构，赋予了人形机器人定向相关的物理柔顺能力，这对于后续机器人脱离单纯的“行走”任务，转向涉及物理接触的复杂操作场景奠定了不可或缺的底层基础 29。

## **跨越现实鸿沟：从域随机化到残差动力学与系统辨识**

当底层平滑性算法在仿真环境中达到最优状态后，部署至真实的物理人形机器人时，不可避免地会遭遇 Sim-to-Real（仿真到现实）鸿沟。针对这一难题，前沿研究表明，单纯依赖大规模域随机化（Domain Randomization, DR）已无法满足高难度动作的需求。

### **域随机化的局限性与结构性偏差**

传统的 Sim-to-Real 过渡高度依赖域随机化技术，即在训练过程中向仿真器注入关于质量、摩擦力、电机阻尼以及质心位置的均匀分布噪声 31。虽然这种方法能够提高策略应对参数扰动的存活率，但它从根本上迫使神经网络采取一种“对冲”策略——即学习一种在所有可能物理参数下都能勉强工作的平均妥协行为。这种行为模式通常表现为极度保守、僵硬，使得机器人无法充分利用其真实的物理潜能来实现敏捷运动 31。

更为严重的是，当人形机器人被赋予实际任务（如搬运未知重量的物体）时，其总质量、惯量分布与质心位置会发生大规模的系统性偏移。这种结构性偏差是小幅度的随机扰动所无法覆盖的，会导致标称策略的急剧退化 31。

### **系统辨识（SysID）与有效载荷自适应**

为了解决结构性偏差，诸如 HALO-Humanoid 等前沿框架引入了基于梯度的两阶段系统辨识（System Identification, SysID）方法 31。不同于将被动噪声作为防御手段，该方法首先利用真实世界交互数据来校准仿真器中的标称物理模型，从而消除内在的仿真误差。随后，在第二阶段主动辨识外部未知载荷的质量分布特征 31。

通过在策略执行前显式地减少物理模型偏差，机器人能够在重载或大扰动环境下实现强化学习策略的零样本迁移（Zero-shot Transfer）。实证数据表明，相较于纯粹的域随机化方法，融合了 SysID 的策略在位置漂移和姿态追踪误差上可实现大幅下降，这为基于 ALCP 柔顺控制的复杂负载任务提供了极强的稳定性保障 31。

### **结合残差强化学习的增量动作模型**

除了系统辨识，通过残差强化学习（Residual Reinforcement Learning）构建增量动作模型（Delta Action Models）代表了另一种消除非结构化现实鸿沟（如齿轮间隙、通信延迟及复杂执行器动态）的高效途径 33。

在 ASAP（Aligning Simulation and Real Physics）框架理念中，研究人员首先在仿真环境中训练出一个高鲁棒性的基础跟踪策略（如本文构建的 Smoothed-Constrained-PPO）。在将该策略部署至物理实体后，通过采集实际状态轨迹与期望轨迹之间的偏差数据，训练一个轻量级的残差网络 33。该残差网络以上一时刻的状态和基础策略的输出为输入，仅输出一个微小的补偿扭矩（Delta Action），用于精准抵消仿真未建模的物理延迟和摩擦非线性 33。基础平滑控制策略与轻量级物理残差补偿网络的联合部署，构成了目前实现人形机器人极度敏捷且安全运动的黄金法则 33。

## **拥抱感知能力：外感知融合与教师-学生蒸馏范式**

在解决了底层动力学的平滑性与现实鸿沟之后，若要使人形机器人在非结构化的自然环境（如野外台阶、建筑工地）中获得自主移动能力，仅仅依赖本体感受（Proprioception，即关节位置、速度与 IMU 数据）构成的“盲跑”策略是远远不够的。未来的核心演进路线是将视觉深度感知（Exteroception）深度融合到运动控制闭环中 15。

### **感知内部模型（PIM）与去噪架构**

在融合激光雷达（LiDAR）或 RGB-D 深度相机的过程中，直接将高维视觉像素输入到高频底层策略网络会导致学习崩溃，因为强化学习难以同时处理低频高维特征提取与高频精细扭矩控制。前沿工作如 PILOT（Perceptive Integrated Low-level Controller）与 PIM（Perceptive Internal Model）架构展示了处理这一问题的标准范式 37。

这类框架通常将视觉传感器的数据转换为以机器人为中心的局部高程图（Elevation Maps）。由于实际物理传感器在扫描边缘（如楼梯边缘）时经常会产生遮挡与稀疏返回，导致高程图布满噪声，这些方案采用了边缘引导的不对称 U-Net（EGAU）或去噪世界模型（Denoising World Models）对地形特征进行实时重构和去噪，从而为强化学习控制策略提供清晰的物理拓扑指引 21。

### **克服信息不对称的教师-学生蒸馏策略**

在感知运动的学习过程中，由于仿真环境能够无条件提供绝对真实的物理信息，而物理部署只能依赖带有噪声的传感器，必须采用教师-学生（Teacher-Student）策略蒸馏管道 40。

1. **特权教师训练：** 首先在 Isaac Gym 等仿真环境中训练一个教师策略，该策略拥有访问“特权信息”（Privileged Information）的权限，即它能直接读取仿真引擎底层完美的无噪声地形高度矩阵、确切的地面摩擦系数以及外力扰动 40。  
2. **学生策略蒸馏：** 随后，利用 DAgger（Dataset Aggregation）等行为克隆算法训练一个纯视觉学生策略。学生网络仅接收来自深度相机的带噪图像与自身状态，被迫学习模仿教师输出的动作轨迹 40。  
3. **学生知情的教师对齐优化：** 为了进一步缩小两者之间的性能差距，前沿研究提出了“学生知情的教师训练”（Student-Informed Teacher Training）范式 42。传统的教师在拥有全局视野时，往往会做出一些“背对障碍物”的极限动作，而这对于视野受限的学生来说是无法模仿的。通过在教师的奖励函数中引入基于师生动作差异的惩罚项，可以强制教师策略改变行为模式（例如主动保持摄像机正对障碍物以减少自遮挡），从而合成出高度易于视觉模仿的平滑行为，极大提升最终部署时的避障与跨越成功率 42。

## **终极形态：基于 VLA 模型的全身协调操作（Whole-Body Loco-Manipulation）**

在底层动力学极致平滑且具备基础视觉感知能力之后，依据 awesome-humanoid-robot-learning 代码库所揭示的行业终极趋势，下一步必须向全身协调操作（Whole-Body Loco-Manipulation, WBC）迈进。过去的控制框架常常将移动（Locomotion）与操作（Manipulation）割裂开来，导致机器人在走向目标并试图抓取物体时，往往因模块边界衔接不畅而失去平衡并摔倒 45。

### **统一的视觉-语言-动作架构与潜在空间学习**

诸如 WholeBodyVLA 与 GR00T N1 等前沿框架，代表了解决这一系统性难题的方向 45。这些系统不再使用离散的硬编码状态机，而是通过层次化架构，将自然语言理解、视觉语义处理与底层全自由度控制进行无缝对接 49：

* **高层多模态基础模型（High-Level VLM Planner）：** 处理来自第一人称视角的 RGB 视频流及人类发出的自然语言指令。通过在海量的人类第一视角视频及遥操作数据上进行无动作标注的“统一潜在学习”（Unified Latent Learning），大语言视觉模型能够深入理解环境的物体可供性（Affordance）与空间关系，并输出一个连续的潜在动作流或高层次的语义动作指令（例如，“走向桌子”的潜在嵌入序列）45。  
* **底层面向移动操作的策略（LMO Controller）：** 高频率运行的强化学习底层网络（即我们经过谱归一化或 ALCP 增强的平滑策略网络）接收高层传来的潜在表征指令，自主解析为精确的全身关节扭矩。这种端到端的架构使得双臂的操作轨迹与下肢的抗干扰平衡机制深度耦合，真正实现了在大空间范围内的连贯操作（如蹲下抓取重物并搬运至推车） 45。

### **底层平滑性与高层视觉认知能力的因果关联**

深入审视这种顶级层次化架构，可以得出一个极具启示性的底层因果关系：**底层动力学的极致平滑性，是高层视觉-语言模型（VLM）能够正常运转的绝对前提。**

如果底层策略未进行严格的拉格朗日受限优化或谱归一化处理，产生的高频非线性抖动与冲击波会直接沿机械连杆传递至安装在机器人头部的 RGB-D 传感器组 28。这种剧烈的物理高频震动将彻底破坏光流算法的稳定性，导致深度估计失效，并使 VLM 视觉主干网络（如 Vision Transformer）的特征提取完全崩溃 49。

因此，从全局系统工程的视角来看，构建 Smoothed-Constrained-PPO 或探索谱归一化技术，其意义不仅在于延长执行器寿命或优化步态的视觉美感；它是为了给头部的传感器阵列提供一个极其稳定、无扰动的物理基座。只有实现了如同“斯坦尼康”般的底层物理平稳，基于 VLA 模型的视觉推理、语义规划以及复杂的全身操作，才真正具备了在现实世界中落地的可能 28。

## **结论与战略演进路线图**

从修改经验性的奖励函数，转向具有严密数学保证的受限优化方程，标志着人形机器人底层强化学习框架走向成熟。通过在 Humanoid-Gym 管道中深度融合拉格朗日主对偶机制与 LCP 梯度约束，构建的 Smoothed-Constrained-PPO 从根本上遏制了仿真中无带宽限制的扭矩激增，在不显著降低动态敏捷性的同时，赋予了步态极佳的物理平滑性与 Sim-to-Sim 鲁棒性。

针对现阶段 ECOLab 基线所遗留的计算资源瓶颈与形态局限，未来的演进路线应沿着以下四个关键阶段展开：

1. **架构级平滑优化：** 立即摒弃极其耗费 GPU 显存的 LCP 双重反向求导惩罚，采用谱归一化（Spectral Normalization）重构 PPO 的执行器网络权重，在实现全局利普希茨连续性的同时，全面恢复 Isaac Gym 的超大规模并行训练效率。  
2. **柔顺控制与现实部署：** 从标量的利普希茨约束进阶至各向异性柔顺控制（ALCP），根据人体仿生学为不同关节分配独立的刚度约束椭球。在跨越现实鸿沟时，结合轻量级的残差动力学网络（Delta Action Models）与系统辨识技术，精准补偿物理样机的传动间隙与延迟。  
3. **感知融合与蒸馏体系：** 破除纯本体感知在复杂地形下的局限，接入深度视觉信息，利用去噪世界模型过滤空间噪声，并通过融合了“学生知情”机制的 DAgger 算法，将上帝视角的特权策略无损蒸馏至物理部署网络中。  
4. **认知与行为的全域打通：** 最终将经过深度优化的平滑底层物理引擎与前沿的 Vision-Language-Action（如 WholeBodyVLA）高层决策模型对齐。利用底层策略提供绝对稳定的物理机体平台，释放 VLM 对大尺度非结构化环境的推理与协调交互能力。

通过循序渐进地突破上述每一道技术壁垒，人形机器人算法研究将不仅仅局限于“稳定行走”的范畴，而是真正踏入具备泛化智能与全场景操作能力的具身智能时代。

#### **Works cited**

1. ECO: Energy-Constrained Optimization with Reinforcement Learning for Humanoid Walking \- GitHub, accessed May 15, 2026, [https://github.com/bigai-ai/ECO-humanoid](https://github.com/bigai-ai/ECO-humanoid)  
2. Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies \- arXiv, accessed May 15, 2026, [https://arxiv.org/abs/2410.11825](https://arxiv.org/abs/2410.11825)  
3. Constraint-Aware Reinforcement Learning via Adaptive Action Scaling \- arXiv, accessed May 15, 2026, [https://arxiv.org/html/2510.11491v3](https://arxiv.org/html/2510.11491v3)  
4. Constraint-Aware Reinforcement Learning via Adaptive Action Scaling \- arXiv, accessed May 15, 2026, [https://arxiv.org/html/2510.11491v1](https://arxiv.org/html/2510.11491v1)  
5. Humanoid Locomotion as Next Token Prediction | Request PDF \- ResearchGate, accessed May 15, 2026, [https://www.researchgate.net/publication/397199371\_Humanoid\_Locomotion\_as\_Next\_Token\_Prediction](https://www.researchgate.net/publication/397199371_Humanoid_Locomotion_as_Next_Token_Prediction)  
6. Constrained Reinforcement Learning with Smoothed Log Barrier Function \- OpenReview, accessed May 15, 2026, [https://openreview.net/pdf?id=Amh95oURaE](https://openreview.net/pdf?id=Amh95oURaE)  
7. Iterative Reachability Estimation for Safe Reinforcement Learning, accessed May 15, 2026, [https://proceedings.neurips.cc/paper\_files/paper/2023/file/dca63f2650fe9e88956c1b68440b8ee9-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/dca63f2650fe9e88956c1b68440b8ee9-Paper-Conference.pdf)  
8. Policy Learning with Constraints in Model-free Reinforcement Learning: A Survey \- IJCAI, accessed May 15, 2026, [https://www.ijcai.org/proceedings/2021/0614.pdf](https://www.ijcai.org/proceedings/2021/0614.pdf)  
9. Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies \- arXiv, accessed May 15, 2026, [https://arxiv.org/html/2410.11825v3](https://arxiv.org/html/2410.11825v3)  
10. Spectral Normalization for Lipschitz-Constrained Policies on Learning Humanoid Locomotion \- ResearchGate, accessed May 15, 2026, [https://www.researchgate.net/publication/390749614\_Spectral\_Normalization\_for\_Lipschitz-Constrained\_Policies\_on\_Learning\_Humanoid\_Locomotion](https://www.researchgate.net/publication/390749614_Spectral_Normalization_for_Lipschitz-Constrained_Policies_on_Learning_Humanoid_Locomotion)  
11. \[2504.08246\] Spectral Normalization for Lipschitz-Constrained Policies on Learning Humanoid Locomotion \- arXiv, accessed May 15, 2026, [https://arxiv.org/abs/2504.08246](https://arxiv.org/abs/2504.08246)  
12. Spectral Normalization for Lipschitz-Constrained Policies on Learning Humanoid Locomotion \- arXiv, accessed May 15, 2026, [https://arxiv.org/pdf/2504.08246](https://arxiv.org/pdf/2504.08246)  
13. Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies, accessed May 15, 2026, [https://lipschitz-constrained-policy.github.io/](https://lipschitz-constrained-policy.github.io/)  
14. SafeMind: A Risk-Aware Differentiable Control Framework for Adaptive and Safe Quadruped Locomotion \- arXiv, accessed May 15, 2026, [https://arxiv.org/html/2604.09474v1](https://arxiv.org/html/2604.09474v1)  
15. BeamDojo: Learning Agile Humanoid Locomotion on Sparse Footholds \- Huayi Wang, accessed May 15, 2026, [https://why618188.github.io/beamdojo/](https://why618188.github.io/beamdojo/)  
16. ECO: Energy-Constrained Optimization with Reinforcement Learning for Humanoid Walking, accessed May 15, 2026, [https://arxiv.org/html/2602.06445v1](https://arxiv.org/html/2602.06445v1)  
17. Movement Smoothness Metrics in Human-Machine Interaction \- POLITECNICO DI TORINO Repository ISTITUZIONALE, accessed May 15, 2026, [https://iris.polito.it/retrieve/handle/11583/2980614/237e509e-ef5f-4d5b-8f29-367f65507972/065%20Movement%20smoothness%20metrics%20in%20human-machine%20interaction\_Final\_Version.pdf](https://iris.polito.it/retrieve/handle/11583/2980614/237e509e-ef5f-4d5b-8f29-367f65507972/065%20Movement%20smoothness%20metrics%20in%20human-machine%20interaction_Final_Version.pdf)  
18. Motion Smoothness Analysis of the Gait Cycle, Segmented by Stride and Associated with the Inertial Sensors' Locations \- MDPI, accessed May 15, 2026, [https://www.mdpi.com/1424-8220/25/2/368](https://www.mdpi.com/1424-8220/25/2/368)  
19. (PDF) Smoothness metrics for reaching performance after stroke. Part 1: which one to choose? \- ResearchGate, accessed May 15, 2026, [https://www.researchgate.net/publication/355659815\_Smoothness\_metrics\_for\_reaching\_performance\_after\_stroke\_Part\_1\_which\_one\_to\_choose](https://www.researchgate.net/publication/355659815_Smoothness_metrics_for_reaching_performance_after_stroke_Part_1_which_one_to_choose)  
20. (PDF) Constraint-Aware Reinforcement Learning via Adaptive Action Scaling, accessed May 15, 2026, [https://www.researchgate.net/publication/396459180\_Constraint-Aware\_Reinforcement\_Learning\_via\_Adaptive\_Action\_Scaling](https://www.researchgate.net/publication/396459180_Constraint-Aware_Reinforcement_Learning_via_Adaptive_Action_Scaling)  
21. Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer \- GitHub, accessed May 15, 2026, [https://github.com/roboterax/humanoid-gym](https://github.com/roboterax/humanoid-gym)  
22. cmjang/humanoid-DM: 达妙人形RL仿真 \- GitHub, accessed May 15, 2026, [https://github.com/cmjang/humanoid-DM](https://github.com/cmjang/humanoid-DM)  
23. Reinforcement Learning on Legged Robots, accessed May 15, 2026, [https://cs224r.stanford.edu/spring\_2023/slides/cs224r-RLLeggedRobots.pdf](https://cs224r.stanford.edu/spring_2023/slides/cs224r-RLLeggedRobots.pdf)  
24. Variations of Augmented Lagrangian for Robotic Multicontact Simulation \- IEEE Xplore, accessed May 15, 2026, [https://ieeexplore.ieee.org/iel8/8860/10778592/11027548.pdf](https://ieeexplore.ieee.org/iel8/8860/10778592/11027548.pdf)  
25. Variations of Augmented Lagrangian for Robotic Multi-Contact Simulation \- arXiv, accessed May 15, 2026, [https://arxiv.org/html/2502.16898v1](https://arxiv.org/html/2502.16898v1)  
26. (PDF) Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies, accessed May 15, 2026, [https://www.researchgate.net/publication/384938513\_Learning\_Smooth\_Humanoid\_Locomotion\_through\_Lipschitz-Constrained\_Policies](https://www.researchgate.net/publication/384938513_Learning_Smooth_Humanoid_Locomotion_through_Lipschitz-Constrained_Policies)  
27. Spectral Normalization for Lipschitz-Constrained Policies on Learning Humanoid Locomotion \- arXiv, accessed May 15, 2026, [https://arxiv.org/html/2504.08246v1](https://arxiv.org/html/2504.08246v1)  
28. Task-Specified Compliance Bounds for Humanoids via Lipschitz-Constrained Policies, accessed May 15, 2026, [https://arxiv.org/html/2603.16180v2](https://arxiv.org/html/2603.16180v2)  
29. Enforcing Task-Specified Compliance Bounds for Humanoids via Anisotropic Lipschitz-Constrained Policies \- ResearchGate, accessed May 15, 2026, [https://www.researchgate.net/publication/402612113\_Enforcing\_Task-Specified\_Compliance\_Bounds\_for\_Humanoids\_via\_Anisotropic\_Lipschitz-Constrained\_Policies](https://www.researchgate.net/publication/402612113_Enforcing_Task-Specified_Compliance_Bounds_for_Humanoids_via_Anisotropic_Lipschitz-Constrained_Policies)  
30. Real-world humanoid locomotion with reinforcement learning \- ResearchGate, accessed May 15, 2026, [https://www.researchgate.net/publication/379896180\_Real-world\_humanoid\_locomotion\_with\_reinforcement\_learning](https://www.researchgate.net/publication/379896180_Real-world_humanoid_locomotion_with_reinforcement_learning)  
31. Closing Sim-to-Real Gap for Heavy-loaded Humanoid Agile Motion Skills via Differentiable Simulation \- arXiv, accessed May 15, 2026, [https://arxiv.org/html/2603.15084v1](https://arxiv.org/html/2603.15084v1)  
32. Sim-to-Real Transfer for Locomotion Tasks on Legged Robots: A Survey \- HULKs, accessed May 15, 2026, [https://hulks.de/\_files/PA\_Luis-Scheuch.pdf](https://hulks.de/_files/PA_Luis-Scheuch.pdf)  
33. Learning Humanoid Control from Simulation to Real to Simulation \- Carnegie Mellon University's Robotics Institute, accessed May 15, 2026, [https://www.ri.cmu.edu/app/uploads/2025/05/MSR\_Thesis\_TairanHe\_2025-3.pdf](https://www.ri.cmu.edu/app/uploads/2025/05/MSR_Thesis_TairanHe_2025-3.pdf)  
34. RobotDancing: Residual-Action Reinforcement Learning Enables Robust Long-Horizon Humanoid Motion Tracking \- arXiv, accessed May 15, 2026, [https://arxiv.org/html/2509.20717v1](https://arxiv.org/html/2509.20717v1)  
35. VMTS: Vision-Assisted Teacher-Student Reinforcement Learning for Multi-Terrain Locomotion in Bipedal Robots \- arXiv, accessed May 15, 2026, [https://arxiv.org/html/2503.07049v1](https://arxiv.org/html/2503.07049v1)  
36. Learning Humanoid Locomotion with Perceptive Internal Model \- arXiv, accessed May 15, 2026, [https://arxiv.org/html/2411.14386v1](https://arxiv.org/html/2411.14386v1)  
37. Learning Humanoid Locomotion with Perceptive Internal Model | Request PDF, accessed May 15, 2026, [https://www.researchgate.net/publication/395211140\_Learning\_Humanoid\_Locomotion\_with\_Perceptive\_Internal\_Model](https://www.researchgate.net/publication/395211140_Learning_Humanoid_Locomotion_with_Perceptive_Internal_Model)  
38. PILOT: A Perceptive Integrated Low-level Controller for Loco-manipulation over Unstructured Scenes \- arXiv, accessed May 15, 2026, [https://arxiv.org/html/2601.17440v1](https://arxiv.org/html/2601.17440v1)  
39. Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning | Request PDF \- ResearchGate, accessed May 15, 2026, [https://www.researchgate.net/publication/383906786\_Advancing\_Humanoid\_Locomotion\_Mastering\_Challenging\_Terrains\_with\_Denoising\_World\_Model\_Learning](https://www.researchgate.net/publication/383906786_Advancing_Humanoid_Locomotion_Mastering_Challenging_Terrains_with_Denoising_World_Model_Learning)  
40. VIRAL: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation, accessed May 15, 2026, [https://viral-humanoid.github.io/static/viral.pdf](https://viral-humanoid.github.io/static/viral.pdf)  
41. VisualMimic Visual Humanoid Loco-Manipulation via Motion Tracking and Generation, accessed May 15, 2026, [https://arxiv.org/html/2509.20322v1](https://arxiv.org/html/2509.20322v1)  
42. Student-Informed Teacher Training \- arXiv, accessed May 15, 2026, [https://arxiv.org/html/2412.09149v2](https://arxiv.org/html/2412.09149v2)  
43. STUDENT-INFORMED TEACHER TRAINING \- Robotics and Perception Group, accessed May 15, 2026, [https://rpg.ifi.uzh.ch/docs/ICLR25\_Messikommer.pdf](https://rpg.ifi.uzh.ch/docs/ICLR25_Messikommer.pdf)  
44. Student-Informed Teacher Training \- OpenReview, accessed May 15, 2026, [https://openreview.net/forum?id=Dzh0hQPpuf](https://openreview.net/forum?id=Dzh0hQPpuf)  
45. WHOLEBODYVLA: TOWARDS UNIFIED LATENT VLA FOR WHOLE-BODY LOCO-MANIPULATION CONTROL \- OpenDriveLab, accessed May 15, 2026, [https://opendrivelab.com/WholeBodyVLA/static/pdf/WholeBodyVLA.pdf](https://opendrivelab.com/WholeBodyVLA/static/pdf/WholeBodyVLA.pdf)  
46. WholeBodyVLA: Towards Unified Latent VLA for Whole-Body Loco-Manipulation Control, accessed May 15, 2026, [https://arxiv.org/html/2512.11047v1](https://arxiv.org/html/2512.11047v1)  
47. Vision-Language-Action (VLA) Models: Concepts, Progress, Applications and Challenges, accessed May 15, 2026, [https://arxiv.org/html/2505.04769v2](https://arxiv.org/html/2505.04769v2)  
48. HEX: Humanoid-Aligned Experts for Cross-Embodiment Whole-Body Manipulation \- arXiv, accessed May 15, 2026, [https://arxiv.org/html/2604.07993v1](https://arxiv.org/html/2604.07993v1)  
49. Hierarchical Vision-Language Planning for Multi-Step Humanoid Manipulation \- arXiv, accessed May 15, 2026, [https://arxiv.org/html/2506.22827v1](https://arxiv.org/html/2506.22827v1)  
50. Vision-Language-Policy Model for Dynamic Robot Task Planning \- arXiv, accessed May 15, 2026, [https://arxiv.org/html/2512.19178v1](https://arxiv.org/html/2512.19178v1)  
51. WholeBodyVLA: Towards Unified Latent VLA for Whole-body Loco-manipulation Control, accessed May 15, 2026, [https://openreview.net/forum?id=OCJmVjyzN7](https://openreview.net/forum?id=OCJmVjyzN7)  
52. Online Motion Planning for Connected Multi-Robot Systems using Vision Language Models as High-level Planners \- GitHub Pages, accessed May 15, 2026, [https://mit-realm.github.io/VLM-gcbfplus-website/files/Online\_planning\_using\_VLMs.pdf](https://mit-realm.github.io/VLM-gcbfplus-website/files/Online_planning_using_VLMs.pdf)  
53. LLM-Grounded Dynamic Task Planning with Hierarchical Temporal Logic for Human-Aware Multi-Robot Collaboration \- arXiv, accessed May 15, 2026, [https://arxiv.org/html/2602.09472](https://arxiv.org/html/2602.09472)