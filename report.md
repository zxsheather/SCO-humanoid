# **针对底层动力学控制的受限优化与平滑性增强算法验证：可行性分析与实施计划**

## **1\. 核心研究动机与行业背景分析**

在当前具身智能与机器人控制领域，将强化学习（Reinforcement Learning, RL）策略从仿真环境零样本（Zero-Shot）迁移到真实的物理世界（Sim-to-Real）是实现通用人形机器人的核心瓶颈 1。NVIDIA 推出的 Isaac Gym 仿真器通过将物理引擎（PhysX）的数据直接保存在 GPU 显存中，并利用 PyTorch 张量（Tensors）进行端到端的交互，彻底消除了 CPU 与 GPU 之间的数据通信延迟。这种大规模并行化仿真技术使得包含数千个环境的策略训练时间从数天缩短至数小时 3。近端策略优化（Proximal Policy Optimization, PPO）算法因其实现的简单性、优异的样本效率以及对连续控制任务的鲁棒性，已经成为底层动力学控制（Kinematic Control）的事实标准 5。

### **1.1 传统 PPO 算法在底层控制中的高频抖动困境**

尽管 PPO 在最大化累积奖励方面表现出色，但其输出的动作序列在应用于人形机器人的底层关节扭矩或位置控制时，经常表现出极强的不稳定性。这种不稳定主要体现为高频的致动器抖动（Actuator Jitter）和不规则的关节加速度激增 7。由于人形机器人（如 XBot-L 拥有 60 个自由度与 54 个驱动电机 9）具有高度非线性和内在的不稳定性，任何微小的状态观测扰动（如传感器噪声或地形微小变化）在经过深度神经网络的放大后，都会导致输出动作的剧烈波动 10。在仿真环境中，由于接触模型的简化和理想化的致动器响应，这种高频抖动往往不会导致机器人摔倒，甚至可能被策略用作维持平衡的“捷径”（即策略通过高频振动来抵消重力势能）11。然而，在真实物理世界或高保真物理引擎（如 MuJoCo）中，这种策略会导致电机过热、硬件损坏或直接摔倒，使得 Sim-to-Real 迁移彻底失败 3。

### **1.2 启发式奖励塑形的局限性与耗时性**

为了缓解策略的不稳定性，业界长期依赖于繁琐的奖励塑形（Reward Shaping）。工程师通常在环境的奖励函数中添加一系列软性惩罚项（Soft Penalties），其中最经典的是“动作差分惩罚”（Action Rate Penalty），其数学形式表现为连续两个时间步输出动作差值的二范数（![][image1]）11。除了动作差分惩罚外，还包括关节加速度惩罚、基础线性加速度惩罚以及扭矩惩罚等 11。

然而，这种启发式奖励塑形面临着无法逾越的工程挑战：奖励权重的微调（Hyperparameter Tuning）极其耗时费力 16。由于这些惩罚项与主线任务（如目标速度追踪、存活奖励）被合并为单一的标量奖励信号，它们之间不可避免地会发生相互干扰 12。如果在训练初期将动作差分惩罚的权重设置过大，智能体为了避免受到惩罚，可能会选择完全不探索，表现为“冻结”在原地；如果设置过小，则在训练后期无法有效抑制高频抖动 8。这种现象被称为“模式混淆”（Mode Confusion）或“奖励干扰”，严重限制了人形机器人复杂地形适应能力的开发 12。

### **1.3 从软惩罚到硬约束：Smoothed-Constrained-PPO 的提出**

针对上述痛点，本研究方向提出了一种范式转换：摒弃传统的启发式动作差分惩罚，将策略的平滑性转化为受限马尔可夫决策过程（Constrained Markov Decision Process, CMDP）中的硬性不等式约束 19。

具体而言，本课程大作业将以 **Humanoid-Gym** (roboterax/humanoid-gym) 为基础训练框架 1，吸收北京通用人工智能研究院提出的 **ECO** (bigai-ai/ECO-humanoid) 框架中的 PPO-Lagrangian 主对偶更新机制 22，并将 **LCP**（Lipschitz-Constrained Policies，利普希茨连续性策略）的梯度惩罚项作为受限优化方程的约束目标 7。由此构建出的 **Smoothed-Constrained-PPO (SC-PPO)** 算法，能够在数学层面上保证策略的平滑性，同时避免繁琐的奖励权重调试。本分析报告将对该方向的理论支撑、资源需求、详细实施路径以及评估验证方案进行详尽的可行性论证。

## ---

**2\. 理论基础与算法架构解析**

要验证 SC-PPO 算法的有效性，必须深入解构其背后的三大理论支柱：受限马尔可夫决策过程（CMDP）、PPO-Lagrangian 优化机制以及利普希茨连续性（Lipschitz Continuity）的梯度惩罚机制。

### **2.1 受限马尔可夫决策过程（CMDP）的数学抽象**

在标准的强化学习设定中，马尔可夫决策过程（MDP）由元组 ![][image2] 组成，目标是找到一个策略 ![][image3]，最大化预期累积折扣奖励 ![][image4]。在传统的动作差分惩罚方法中，平滑性惩罚仅仅是 ![][image5] 中的一个负项 11。

CMDP 通过扩展 MDP 模型，引入了成本函数（Cost Function）集合 ![][image6] 和相应的成本阈值 ![][image7] 19。此时的优化目标转变为一个受限优化问题：

![][image8]  
![][image9]  
其中，![][image10] 代表策略执行过程中的累积预期成本 18。在 ECO 框架中，该成本被定义为机器人的预期能量消耗（电机扭矩与关节速度的乘积），要求能量消耗不得超过给定的物理预算 22。而在本课题的 SC-PPO 设定中，我们将这一约束替换为表征策略平滑度的 LCP 梯度惩罚项 7。这种公式化的优势在于，任务奖励 ![][image11]（如向目标方向前进）与平滑性约束 ![][image12] 在优化空间中被完全解耦，消除了标量化奖励塑形带来的权重博弈困境 24。

### **2.2 PPO-Lagrangian 与主对偶更新机制（Primal-Dual Optimization）**

为了求解上述 CMDP 问题，本作业将采用 PPO-Lagrangian 算法，这是一种在安全强化学习（Safe RL）中广泛应用的基于拉格朗日松弛（Lagrangian Relaxation）的主对偶（Primal-Dual）优化方法 26。

通过引入拉格朗日乘子（Lagrange Multiplier）![][image13] 作为对偶变量，我们可以将原始的受限优化问题转化为无约束的最大最小（Minimax）问题：

![][image14]  
在算法的实际执行过程中，PPO-Lagrangian 机制交替进行**主更新（Primal Update）和对偶更新（Dual Update）** 6：

1. **主更新（策略更新）**：固定当前的惩罚系数 ![][image15]，使用标准的 PPO 裁剪代理目标（Clipped Surrogate Objective）算法更新神经网络的参数 ![][image16]，以最大化修改后的目标函数 ![][image17] 24。  
2. **对偶更新（乘子更新）**：固定当前的策略 ![][image16]，通过梯度上升（Gradient Ascent）或反馈控制算法更新拉格朗日乘子 ![][image15]，以惩罚违反约束的行为 6。如果当前策略的成本超过了阈值（![][image18]），则 ![][image15] 会增加，迫使后续的策略更新更加关注平滑性；反之，如果约束已满足（![][image19]），![][image15] 会逐渐减小甚至归零，使策略能够全速优化任务奖励 24。

### **2.3 PID-Lagrangian 控制器的引入以消除震荡**

传统 PPO-Lagrangian 在对偶更新时采用简单的梯度上升法：![][image20]，其中 ![][image21] 为学习率 23。然而，大量文献及工程实践表明，由于成本估计的偏差和离策略（Off-policy）采样的延迟，直接的梯度上升容易导致拉格朗日乘子发生剧烈震荡（Oscillation） 6。乘子的过度反应会引发策略在“极度保守（完全停滞以满足约束）”和“极度冒险（无视约束以追求奖励）”之间反复横跳 6。

为了保障大作业训练的稳定性，必须在算法层面集成 **PID-Lagrangian** 机制 24。借鉴控制工程中的比例-积分-微分（Proportional-Integral-Derivative, PID）反馈控制理论，我们将拉格朗日乘子 ![][image15] 视为控制器的输出，将约束违反度视为误差信号 ![][image22] 24。更新方程将被重构为：

![][image23]  
其中：

* **比例增益 (![][image24])**：提高系统对当前违反程度的响应速度。  
* **积分增益 (![][image25])**：消除稳态误差，确保在训练后期策略始终严格满足硬性阈值。  
* **微分增益 (![][image26])**：提供阻尼效应，平抑因为环境随机性带来的瞬态成本波动，大幅增强训练的平滑性 24。

通过利用这种机制，我们能在不手动微调恒定权重的前提下，让算法自适应地寻找到在保证平滑控制前提下最大化任务表现的最优乘子 ![][image27] 25。

### **2.4 LCP（利普希茨连续性）梯度惩罚项的数学内涵**

在确立了优化框架后，核心问题在于如何科学地定义平滑性成本 ![][image28]。本方案采用 Lipschitz-Constrained Policies (LCP) 方法，将函数的利普希茨连续性（Lipschitz Continuity）作为评估平滑度的金标准 7。

在数学上，如果一个函数 ![][image29] 满足利普希茨连续性条件，意味着对于所有的输入 ![][image30]，其输出的变化率被一个常数 ![][image31]（利普希茨常数）所限制 16：

![][image32]  
等价地，这也意味着函数在其定义域内，关于输入的梯度的范数是有界的 16：

![][image33]  
在强化学习的上下文中，如果我们将策略网络 ![][image34] 约束为利普希茨连续的，这就意味着即使环境状态 ![][image35] 发生突变（由于传感器噪声或踩到复杂地形的边缘），策略输出的动作 ![][image36] 的变化也将被严格限制在一个可控的平滑范围内，从根本上消除了致动器的高频抖动 16。

为了在深度学习框架中实现这一硬性约束，LCP 采用梯度惩罚（Gradient Penalty）机制 7。在每一次 PPO 优化迭代中，计算策略的输出动作相对于输入状态（或网络权重）的雅可比矩阵范数 7。本大作业中，成本函数 ![][image28] 可定义为：

![][image37]  
结合上述 CMDP 框架，SC-PPO 的完整优化过程实质上是要求拉格朗日乘子机制自动调整惩罚力度，使得策略的期望梯度范数始终低于给定的阈值 ![][image7]（即我们预设的利普希茨常数阈值）。相比于传统的低通滤波器（Low-pass filters）会导致不可导问题并阻碍策略探索，基于梯度的惩罚项是完全可导的（Differentiable），能够完美融入现有的 PyTorch 自动微分管线中 3。

## ---

**3\. 课程大作业可行性与资源需求分析**

在 Isaac Gym 环境中实现上述高复杂度的算法重构，对软件生态的兼容性和硬件的计算显存均提出了严苛的要求。以下是对该项目资源需求与实施可行性的详尽剖析。

### **3.1 代码基座与软件栈可行性分析**

该项目的软件实现需要高度依赖于开源社区的最佳实践，整体软件栈可行性评估为**极高（Highly Feasible）**。

项目的基石为 **Humanoid-Gym**（roboterax/humanoid-gym），这是一个专为人形机器人零样本 Sim-to-Real 迁移设计的高性能强化学习框架 1。它本身已经集成了 RobotEra 公司的 XBot-S（1.2米）和 XBot-L（1.65米，具有 60 个自由度）机器人的精准物理模型，并提供了完善的奖励配置接口 1。

Humanoid-Gym 的算法后端依赖于苏黎世联邦理工学院（ETH Zurich）开发的 **rsl\_rl** 库 33。这是一个轻量级且极具扩展性的 PyTorch PPO 实现库 35。学生只需在 rsl\_rl 的 PPO 核心类中进行模块化修改（引入 lagrange\_multiplier 张量和 PID 更新步骤），即可完成 PPO-Lagrangian 的底层改造 33。

在此过程中，**ECO** (bigai-ai/ECO-humanoid) 开源仓库的源码将作为极佳的参考范本 22。由于 ECO 已经成功在类似的腿足机器人环境（如 BRUCE 机器人）中实现了基于能量约束的 PPO-Lagrangian 22，学生可以通过比对 ECO 的主对偶更新代码（如 train.py 中的调度逻辑和拉格朗日乘子的计算图）来加速自身项目的开发 22。

**核心软件依赖版本限制**： Isaac Gym Preview 4 对操作系统、Python 版本及 CUDA 编译库的兼容性要求极其严格，这也是众多学生在配置阶段失败的主因 1。为确保项目顺利推进，必须强制采用以下环境配置 1：

* **操作系统**：Ubuntu 20.04 或 22.04 LTS（WSL2 虽然可用，但可能引发张量分配异常，不建议作为主要开发环境） 39。  
* **NVIDIA 显卡驱动**：至少为版本 515，官方推荐为 525.xx 或更高（如 580.65），以支持底层 PhysX 引擎的正确调度 1。  
* **Python 版本**：严格限定为 3.8，由于 Isaac Gym 的 C++ 动态链接库 .so 文件是基于 Python 3.8 编译的，其他版本可能导致段错误（Segmentation Fault） 1。  
* **深度学习框架**：PyTorch 1.13.1，必须搭配 CUDA 11.7 进行安装，同时配套 torchvision==0.14.1 和 torchaudio==0.13.1 1。  
* **额外库**：gymnasium\[mujoco\]，用于支持后续的 Sim-to-Sim 验证模块 43。

### **3.2 硬件算力与显存资源（VRAM）评估**

尽管算法逻辑上可行，但从计算资源角度来看，该项目的实施可行性为**中等偏上（Moderate-to-High）**，其主要瓶颈在于 **GPU 显存（VRAM）的消耗**。

在传统的 Isaac Gym 训练中，因为仿真数据和神经网络全部驻留在 GPU 上（消除 CPU 瓶颈），通常可以在单张显卡上轻易开启 4096 个并行环境（Parallel Environments） 3。然而，**SC-PPO 引入的 LCP 梯度惩罚机制将对 VRAM 造成指数级的压力** 32。

具体而言，为了计算策略梯度惩罚（![][image38]），在 PyTorch 的后向传播（Backward Pass）中必须设置 create\_graph=True，以便为后续的 PPO 参数更新保留计算图，实现“对梯度的导数”的计算 32。这种二阶微分操作会缓存所有激活层的中间变量，导致 GPU 显存占用激增 31。文献指出，LCP 技术的大规模应用可能会严重限制并行训练环境的数量，从而降低数据吞吐量并延缓收敛速度 32。

**硬件配置建议矩阵**：

| 组件类别 | 基础运行配置（最低妥协方案） | 推荐开发配置（保障训练效率） | 理想验证配置（无缝全规模并行） |
| :---- | :---- | :---- | :---- |
| **GPU** | NVIDIA RTX 3070 / 4070 | NVIDIA RTX 4080 / 4090 | NVIDIA RTX Ada 6000 / A100 |
| **显存 (VRAM)** | 8 GB \- 12 GB 46 | **16 GB \- 24 GB** 48 | 48 GB 或 80 GB 46 |
| **应对策略** | 仅开启 256\~512 个并行环境，训练速度较慢 | 可开启 1024\~2048 个并行环境，兼顾效率与显存 | 可全量开启 4096 个并行环境，实现极速收敛 44 |
| **系统内存** | 32 GB 46 | 64 GB 46 | 128 GB 46 |
| **处理器** | 4 核心（如 Core i5 / Ryzen 5） 48 | 8 核心（如 Core i7 / Ryzen 7） 48 | 16 核心以上（Threadripper） 46 |
| **存储介质** | 50 GB 固态硬盘（SSD） 46 | 500 GB NVMe SSD 46 | 1TB NVMe SSD，用于高频日志落盘 46 |

**资源可行性结论**：只要学生能够访问具备至少 16GB VRAM（如 RTX 4080）的高校实验室服务器，或在个人设备上主动将并行环境数量 \--num\_envs 缩减至可用显存允许的上限，本大作业的资源需求完全可以被满足 45。

## ---

**4\. 详细且可落实的实施路径与重构计划**

为了保障课程大作业的顺利交付，整个实施流程被划分为四个循序渐进的工程阶段（建议周期为 6 至 8 周）。该实施计划明确了每一步的代码级操作与验证目标。

### **阶段一：环境管线基准测试与熟悉**

**目标**：确保底层的软硬件环境完备，成功运行原版 Humanoid-Gym，并确立基于传统“动作差分惩罚”的评估基线（Baseline）。

1. **环境配置与依赖部署**： 按照上文详述的配置创建 conda 虚拟环境。首先安装 Isaac Gym Preview 4，进入 isaacgym/python 执行 pip install \-e. 1。随后克隆 roboterax/humanoid-gym 及 rsl\_rl 代码库，并在各自目录下执行 pip install \-e. 完成本地包的链接安装 1。  
2. **基座机器人模型加载**： 在 resources/robots/XBot/ 目录下熟悉 XBot-L 机器人的 URDF/MJCF 文件，了解其质量（57kg）及 60 个自由度的映射关系 1。  
3. **Baseline 模型训练**： 执行默认训练脚本：python scripts/train.py \--task=humanoid\_ppo \--run\_name=baseline\_arp 1。在此期间，分析环境自带的启发式奖励结构。在配置文件中找到 action\_rate 惩罚项（计算方式为 ![][image1]，权重通常设为 ![][image39] 或类似负值） 11。  
4. **数据采集与 TensorBoard 日志观察**：  
   记录训练收敛时的各项指标，特别关注“任务奖励”（如线速度/角速度追踪精度）与“平滑度指标”（动作抖动幅值）。这些数据将作为后续验证 SC-PPO 有效性的根本参照。

### **阶段二：拉格朗日主对偶机制与 PID 更新代码移植**

**目标**：解耦强化学习算法层，重构 rsl\_rl 的核心逻辑，使 PPO 具备处理硬性约束的能力。

1. **参考 ECO 架构**： 研读 bigai-ai/ECO-humanoid 中 PPO-Lagrangian 的实现细节 22。ECO 框架成功地将拉格朗日乘子动态更新逻辑嵌入到了策略收集后的优化阶段 22。  
2. **重构 PPO 核心类**： 在 rsl\_rl 库的 algorithms/ppo.py 中，修改 PPO 类的初始化函数，添加一个受 PyTorch 梯度跟踪但并不作为网络权重的变量 self.lagrange\_multiplier 37。设定其初始值（如 0.0）并指定优化器。  
3. **整合 PID 控制器**： 不采用容易导致模型震荡的原始梯度上升法 27，而是在类中增加一个基于误差反馈的 PID 更新器。在每个 Epoch 结束时，计算当前批次下的约束违规期望值 ![][image22] 24。编写更新规则：将比例项 ![][image40]、积分项 ![][image41] 和微分项 ![][image42] 累加到当前的乘子上，并使用 torch.clamp 限制其非负区间（例如 ![][image43]），防止乘子在前期爆炸 24。  
4. **基础功能验证**：  
   暂时不引入计算量庞大的梯度惩罚，而是构建一个简单的“占位符”约束（例如限制某一关节的最大转角输出）。观察并绘制拉格朗日乘子 ![][image15] 随训练迭代步数的演化曲线，确保乘子能够在约束超标时迅速上升，在约束满足时缓慢下降，证明 PID-Lagrangian 更新模块运转正常。

### **阶段三：LCP 梯度惩罚机制的数学推导与张量实现**

**目标**：实现利普希茨连续性惩罚，将其作为受限优化的唯一平滑性依据，彻底构建出 SC-PPO 算法。

1. **废除传统奖励塑形**： 在 humanoid-gym 的配置脚本（如 cfg\_train 或相关 env.py 中寻找 compute\_reward 函数），将所有的软平滑性奖励（包括 action\_rate、joint\_acceleration 等）的权重（Scale）显式设定为 0 1。这确保了策略在不受传统惩罚干预的情况下运行。  
2. **构建雅可比矩阵（Jacobian）梯度图**：  
   这是大作业技术含量最高的部分。在计算 PPO 代理损失（Surrogate Loss）的阶段，需要获取策略网络 ![][image34] 输出动作均值对输入观测状态 ![][image35] 的导数。  
   Python  
   \# 代码逻辑示范（受限优化构建）  
   actions\_mean \= policy(observations)  
   \# 必须保留计算图以支持对梯度再求导  
   gradients \= torch.autograd.grad(outputs=actions\_mean.sum(),   
                                   inputs=observations,   
                                   create\_graph=True,   
                                   retain\_graph=True,   
                                   only\_inputs=True)

   (注：上述代码是简化的概念表达，实际工程中需要处理批次维度和多动作维度的张量规约问题 37)  
3. **损失函数融合**： 计算上述梯度的 L2 范数（即 ![][image28]），并与用户定义的利普希茨常数阈值 ![][image7] 进行比较 7。重构优化器的总损失函数： total\_loss \= surrogate\_loss \+ value\_loss\_coeff \* value\_loss \- entropy\_coeff \* entropy\_loss \+ self.lagrange\_multiplier.detach() \* (gradient\_norm \- d) 10。 通过使用 .detach()，确保在执行主更新时，拉格朗日乘子被视为常数，防止 PyTorch 计算图发生闭环错误 37。

### **阶段四：地形泛化训练与性能度量采集**

**目标**：在复杂地形上对比新旧算法，验证 SC-PPO 在维持鲁棒性同时解决抖动问题的优越性。

1. **地形设置**：利用 Isaac Gym 内置的程序化地形生成（Procedural Terrain Generation）接口，配置包含“随机阶梯（Random Stairs）”和“粗糙平面（Rough Planes）”的复杂场景 4。这些地形能频繁触发机器人的不规则姿态变化，最能暴露策略的高频抖动缺陷 8。  
2. **性能监控与对比实验运行**： 大幅下调环境数量（如 \--num\_envs 1024）以适配 LCP 的显存需求 45。并行或按序运行两组核心实验：  
   * **Baseline**：传统 PPO \+ 动作差分惩罚。  
   * **SC-PPO**：PPO-Lagrangian \+ 零软性惩罚 \+ LCP 梯度约束。  
3. **数据落盘**：系统性导出奖励收敛曲线、拉格朗日乘子动态变化曲线以及关节震动评估数据 52。

## ---

**5\. 实验设计、评估体系与预期成果分析**

为了确保课程大作业最终形成一份具有完整实验数据且具说服力的算法研究报告，必须建立严谨的评估指标与对比框架。

### **5.1 量化指标定义 (Quantitative Metrics)**

评估体系将摒弃单一的“平均累积奖励（Mean Reward）”，而是构建多维度的动力学度量体系，具体对比如下 2：

| 评估维度 | 指标定义及其物理意义 | 传统方法 (Action Rate Penalty) 的常见表现 | SC-PPO (LCP 约束) 的预期表现 |
| :---- | :---- | :---- | :---- |
| **动作输出抖动 (Action Jitter)** | 输出指令随时间的均方差：![][image44]。反映神经网络层面决策的平滑度。 | 高方差。在平地可能平滑，但在阶梯边缘容易出现极端剧烈抖动 52。 | 始终维持在极低阈值（即设定的 ![][image7]），具备理论上限保证 52。 |
| **关节位置震荡 (DoF Pos Jitter)** | 物理层面上关节加速度的二范数：![][image45] 11。直接关乎电机能否承受高频换向 8。 | 会为了追踪速度目标而产生妥协性的高频振荡 8。 | 显著下降，表现出流体般自然连贯（Fluid）的步态转换 16。 |
| **能量消耗 (Energy Consumption)** | 运动过程中致动器做功：![][image46] 14。抖动会产生大量无用对抗扭矩 3。 | 由于内耗较高，能量消耗大 22。 | 因为平滑的梯度控制，寄生内耗消失，能量效率被动提升 52。 |
| **任务达成度 (Task Return)** | 追踪给定的基座线速度与角速度指令的能力。反映了策略是否因追求平滑而变得过度保守 14。 | 容易陷入“模式混淆”，在平滑惩罚过高时彻底拒绝移动 12。 | 凭借 PID-Lagrangian 的乘子退火机制，任务表现可比肩甚至超越 Baseline 6。 |

### **5.2 Isaac Gym 仿真内部对比预期**

在 Isaac Gym 粗糙地形训练的前期，由于约束条件未被满足，PID-Lagrangian 机制会使得惩罚乘子 ![][image15] 迅速拉升 24。在此期间，SC-PPO 智能体可能会表现得非常“迟钝”或行进缓慢，此时其梯度范数被强制压低。随着策略逐渐掌握平滑前行的规律，梯度范数降低至阈值 ![][image7] 以下，![][image15] 乘子将会平缓衰减 28。此时，智能体将释放约束压力，加速收敛于最优任务奖励。相比之下，采用传统软性奖励的 Baseline 模型由于固定惩罚权重的存在，收敛曲线会表现出更大的波动性和方差 12。这将在 TensorBoard 的图表中形成鲜明的对比，为研究报告提供极其直观且具备理论支撑的论点材料。

## ---

**6\. MuJoCo 中的 Sim-to-Sim 跨引擎验证方案**

为了证明算法在“零样本（Zero-Shot）”迁移到真实物理世界时的鲁棒性，报告的最终高潮将落在 Sim-to-Sim（仿真到仿真）的跨引擎测试环节。这一步骤至关重要，它充当了 Sim-to-Real 的强有力代理论证 1。

### **6.1 引擎底层差异带来的严苛挑战**

用于大规模并行训练的 Isaac Gym 依赖于 NVIDIA PhysX 引擎。为了保障数千个环境在 GPU 上的同步计算效率，PhysX 通常采用相对宽容（Permissive）和轻量级的惩罚性接触模型（Penalty-based Contact Model）以及隐式致动器（Implicit Actuators）近似 3。这种底层架构有时会“宽恕”一些物理上不切实际的高频震动，导致策略在 Isaac Gym 中利用这种震动维持平衡 13。

反之，MuJoCo (Multi-Joint dynamics with Contact) 是一种基于精确求解接触力（运用高斯-赛德尔迭代等约束求解器算法）的高保真多体动力学引擎 11。MuJoCo 对摩擦力和接触面的计算极其敏锐，任何由于策略网络输出的不连贯动作而产生的“脚底微小打滑”或“高频践踏”，都会导致 MuJoCo 环境中发生灾难性的穿模、关节限位报错或直接失去平衡摔倒 11。

### **6.2 验证步骤与结果预判**

利用 roboterax/humanoid-gym 中高度集成的部署脚本即可完成这项闭环验证 1。

1. **策略导出**：将 Baseline 与 SC-PPO 训练收敛后的模型导出为 JIT (Just-In-Time) 编译的 PyTorch 格式文件（.pt） 1。  
2. **MuJoCo 加载**：在终端运行跨引擎部署命令： python scripts/sim2sim.py \--load\_model /path/to/logs/XBot\_ppo/exported/policies/policy\_sc\_ppo.pt 1。该脚本会自动加载机器人的 MJCF 描述文件，并通过模拟实时控制循环（读取状态、推断策略、下发扭矩指令）与 MuJoCo 环境进行交互 55。  
3. **结果对比预期**：  
   * 预期 **Baseline 策略** 在平坦地形尚可勉强行走，一旦在 MuJoCo 引入微小地形扰动，高频的动作变化会立刻与 MuJoCo 严苛的接触动力学发生冲突，导致模型僵直或快速崩溃倒地 11。  
   * 预期 **SC-PPO 策略** 展现出极强的迁移韧性。因为 LCP 硬约束在训练阶段就从根本上剥夺了策略利用“高频微调”作为平衡手段的可能性，迫使策略学会了利用全身动力学进行大尺度、平滑重力调整的本质运动规律 10。这使得机器人的步态在 MuJoCo 中如丝般顺滑（Smooth and Robust），验证了本课题在解决 Sim-to-Real 鸿沟上的卓越价值 10。

## ---

**7\. 潜在风险排查、技术挑战与应急预案 (Risk Management)**

涉及底层 PPO 梯度的数学重构极具挑战性。本报告识别出三项可能危及大作业进度的主要风险，并预先制定了翔实的应对预案。

### **7.1 PyTorch 计算图显存溢出 (CUDA OOM)**

* **风险描述**：计算 LCP 梯度惩罚时，torch.autograd.grad(create\_graph=True) 指令强制 PyTorch 跟踪一阶梯度的计算历史以准备二阶求导 32。在 4096 个并行环境下，前向传播、PPO Clip 缓存和双重后向传播的内存叠加将瞬间击穿 16GB 甚至 24GB 显存，引发不可恢复的崩溃 32。  
* **应急预案**：  
  1. **牺牲吞吐量换取空间**：在训练启动命令中，强行降低并行规模，添加参数 \--num\_envs 512。尽管单次迭代所需墙上时钟时间（Wall-clock Time）会增加，但完全消除了 OOM 风险 44。  
  2. **小批量梯度采样（Mini-batch Subsampling）**：在重写的 PPO 逻辑中，不必针对全部的 Batch Size 计算梯度惩罚。可以从当前批次中随机抽取 10% 的样本计算 ![][image38] 的均值作为 ![][image28]，即可在大幅缩减计算图节点的前提下，维持算法的理论有效性。

### **7.2 PID-Lagrangian 导致策略模型训练发散**

* **风险描述**：如果在主对偶更新中，拉格朗日乘子 ![][image15] 被赋予了错误的 PID 超参数（例如积分项 ![][image25] 设置过高造成深度积分饱和），乘子可能会在训练中期爆炸，致使梯度中惩罚项彻底淹没 PPO 代理损失项 6。这种“对偶变量过度响应”将直接破坏策略网络的权重参数 6。  
* **应急预案**：  
  1. **硬性削幅（Hard Clamping）**：在更新代码中加入安全机制，强制限制 ![][image47]。这样即使控制器积分跑飞，惩罚力度也是有上限的 24。  
  2. **超参扫描策略**：初始阶段完全关闭 I 和 D 项（即 ![][image48]），退化为基础步长梯度上升 27。确认能够收敛后，再逐步引入小比例的 ![][image24] 和 ![][image25]。

### **7.3 Sim-to-Sim 跨物理引擎接口映射断层**

* **风险描述**：虽然 humanoid-gym 提供了完善的转换脚本，但在自定义环境配置时，Isaac Gym 依赖的 URDF 关节排序可能与 MuJoCo 依赖的 MJCF 关节解析顺序产生微小的错位 1。如果动作指令数组下发给了错误的关节，机器人将在测试刚启动时就发生自碰撞扭曲 11。  
* **应急预案**： 不要盲目信任自动映射。部署前仔细审查 humanoid/scripts/sim2sim.py 中的 action 派发逻辑 1。通过向特定单一下标的关节下发正弦波指令，在 GUI 界面中逐一比对真实受激关节是否一致。必要时在转换脚本层面插入一个定制化的 Index Mapping Dictionary，将网络输出的动作顺序强制翻译为 MuJoCo 解析的顺序。

## ---

**8\. 总结性陈述**

本大作业实施计划围绕“底层动力学控制的受限优化与平滑性增强算法验证”这一极具学术价值的命题展开。从可行性视角来看，依托于 roboterax/humanoid-gym 成熟的训练框架 1、参考 ECO-humanoid 中 PPO-Lagrangian 的代码范式 22，并创新性地将 LCP（利普希茨连续性）梯度惩罚嵌入主对偶更新机制 7，不仅在数学理论上具有严密的推导支撑（即将 CMDP 约束求解转化为无奖励干扰的自适应优化）19，在工程实现上也通过精细的代码定位和明确的硬件资源规划展现出高度的可达性。

通过采用上述方案构建的 Smoothed-Constrained-PPO (SC-PPO) 算法，学生不仅能够深入理解并重构强化学习中的高级优化机制（Primal-Dual Optimization与 PID-Lagrangian）24，更重要的是，能够系统性解决当前具身智能界长期存在的奖励塑形“调参地狱”和高频控制抖动难题 7。最终，基于 Isaac Gym 复杂地形的训练对比数据以及 MuJoCo 高保真度跨引擎（Sim-to-Sim）的零样本验证结果 11，学生将有充足的原始素材撰写出一篇结构严谨、逻辑闭环且颇具学术突破水准的高质量算法研究报告。

#### **Works cited**

1. Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer \- GitHub, accessed May 13, 2026, [https://github.com/roboterax/humanoid-gym](https://github.com/roboterax/humanoid-gym)  
2. Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies \- People @EECS, accessed May 13, 2026, [https://people.eecs.berkeley.edu/\~sastry/pubs/Pdfs%20of%202024/ChenLearning2024.pdf](https://people.eecs.berkeley.edu/~sastry/pubs/Pdfs%20of%202024/ChenLearning2024.pdf)  
3. Reinforcement Learning Framework for Improving Real-World Performance of the Bipedal Robot SUBO-2 with Low-Backdrivability \- ResearchGate, accessed May 13, 2026, [https://www.researchgate.net/publication/403202263\_Reinforcement\_Learning\_Framework\_for\_Improving\_Real-World\_Performance\_of\_the\_Bipedal\_Robot\_SUBO-2\_with\_Low-Backdrivability](https://www.researchgate.net/publication/403202263_Reinforcement_Learning_Framework_for_Improving_Real-World_Performance_of_the_Bipedal_Robot_SUBO-2_with_Low-Backdrivability)  
4. Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning \- arXiv, accessed May 13, 2026, [https://arxiv.org/html/2511.04831v1](https://arxiv.org/html/2511.04831v1)  
5. GenPO: Generative Diffusion Models Meet On-Policy Reinforcement Learning \- arXiv, accessed May 13, 2026, [https://arxiv.org/html/2505.18763v2](https://arxiv.org/html/2505.18763v2)  
6. OFF-POLICY PRIMAL-DUAL SAFE REINFORCEMENT LEARNING \- ICLR Proceedings, accessed May 13, 2026, [https://proceedings.iclr.cc/paper\_files/paper/2024/file/2f8b56543953d60f262fb2c4b85c50b3-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2024/file/2f8b56543953d60f262fb2c4b85c50b3-Paper-Conference.pdf)  
7. Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies, accessed May 13, 2026, [https://lipschitz-constrained-policy.github.io/](https://lipschitz-constrained-policy.github.io/)  
8. Visual Imitation Enables Contextual Humanoid Control \- arXiv, accessed May 13, 2026, [https://arxiv.org/html/2505.03729v5](https://arxiv.org/html/2505.03729v5)  
9. Humanoid-Gym, accessed May 13, 2026, [https://sites.google.com/view/humanoid-gym/](https://sites.google.com/view/humanoid-gym/)  
10. Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies \- arXiv, accessed May 13, 2026, [https://arxiv.org/html/2410.11825v1](https://arxiv.org/html/2410.11825v1)  
11. Booster Gym: An End-to-End Reinforcement Learning Framework for Humanoid Robot Locomotion \- arXiv, accessed May 13, 2026, [https://arxiv.org/html/2506.15132v1](https://arxiv.org/html/2506.15132v1)  
12. The Reward Scaling Problem in Reinforcement Learning for Quadruped Robots: Unstable Bipedal Behavior, Jitter, and Command Leakage : r/reinforcementlearning \- Reddit, accessed May 13, 2026, [https://www.reddit.com/r/reinforcementlearning/comments/1s9g7a9/the\_reward\_scaling\_problem\_in\_reinforcement/](https://www.reddit.com/r/reinforcementlearning/comments/1s9g7a9/the_reward_scaling_problem_in_reinforcement/)  
13. Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer \- arXiv, accessed May 13, 2026, [https://arxiv.org/html/2404.05695v2](https://arxiv.org/html/2404.05695v2)  
14. Humanoid Parkour Learning \- arXiv, accessed May 13, 2026, [https://arxiv.org/html/2406.10759v2](https://arxiv.org/html/2406.10759v2)  
15. GitHub \- Argo-Robot/quadrupeds\_locomotion: Learn how to train a quadruped robot to walk using reinforcement learning, from defining actions and observations to designing rewards and transitioning from simulation to reality., accessed May 13, 2026, [https://github.com/Argo-Robot/quadrupeds\_locomotion](https://github.com/Argo-Robot/quadrupeds_locomotion)  
16. (PDF) Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies, accessed May 13, 2026, [https://www.researchgate.net/publication/384938513\_Learning\_Smooth\_Humanoid\_Locomotion\_through\_Lipschitz-Constrained\_Policies](https://www.researchgate.net/publication/384938513_Learning_Smooth_Humanoid_Locomotion_through_Lipschitz-Constrained_Policies)  
17. Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies, accessed May 13, 2026, [https://www.researchgate.net/publication/398055968\_Learning\_Smooth\_Humanoid\_Locomotion\_through\_Lipschitz-Constrained\_Policies](https://www.researchgate.net/publication/398055968_Learning_Smooth_Humanoid_Locomotion_through_Lipschitz-Constrained_Policies)  
18. ECO: Energy-Constrained Optimization with Reinforcement Learning for Humanoid Walking, accessed May 13, 2026, [https://www.researchgate.net/publication/400584249\_ECO\_Energy-Constrained\_Optimization\_with\_Reinforcement\_Learning\_for\_Humanoid\_Walking](https://www.researchgate.net/publication/400584249_ECO_Energy-Constrained_Optimization_with_Reinforcement_Learning_for_Humanoid_Walking)  
19. Model-based Safe Deep Reinforcement Learning via a Constrained Proximal Policy Optimization Algorithm | OpenReview, accessed May 13, 2026, [https://openreview.net/forum?id=hYa\_lseXK8](https://openreview.net/forum?id=hYa_lseXK8)  
20. Constrained Markov Decision Processes: Stochastic Modeling | Request PDF \- ResearchGate, accessed May 13, 2026, [https://www.researchgate.net/publication/357014626\_Constrained\_Markov\_Decision\_Processes\_Stochastic\_Modeling](https://www.researchgate.net/publication/357014626_Constrained_Markov_Decision_Processes_Stochastic_Modeling)  
21. ECO: Energy-Constrained Optimization with Reinforcement Learning for Humanoid Walking, accessed May 13, 2026, [https://arxiv.org/html/2602.06445v1](https://arxiv.org/html/2602.06445v1)  
22. ECO: Energy-Constrained Optimization with Reinforcement Learning for Humanoid Walking \- GitHub, accessed May 13, 2026, [https://github.com/bigai-ai/ECO-humanoid](https://github.com/bigai-ai/ECO-humanoid)  
23. Exploring Safe Reinforcement Learning for Sequential Decision Making, accessed May 13, 2026, [https://www.ri.cmu.edu/app/uploads/2023/05/MSR\_thesis\_Fan\_Yang.pdf](https://www.ri.cmu.edu/app/uploads/2023/05/MSR_thesis_Fan_Yang.pdf)  
24. PID Lagrangian \- Matthew Landers, accessed May 13, 2026, [https://mattlanders.net/pid-lagrangian.html](https://mattlanders.net/pid-lagrangian.html)  
25. Towards a Practical Understanding of Lagrangian Methods in Safe Reinforcement Learning, accessed May 13, 2026, [https://arxiv.org/html/2510.17564v2](https://arxiv.org/html/2510.17564v2)  
26. Penalized Proximal Policy Optimization for Safe Reinforcement Learning \- IJCAI, accessed May 13, 2026, [https://www.ijcai.org/proceedings/2022/0520.pdf](https://www.ijcai.org/proceedings/2022/0520.pdf)  
27. Adaptive Primal-Dual Method for Safe Reinforcement Learning \- IFAAMAS, accessed May 13, 2026, [https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2024/pdfs/p326.pdf](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2024/pdfs/p326.pdf)  
28. An Empirical Study of Lagrangian Methods in Safe Reinforcement Learning \- arXiv, accessed May 13, 2026, [https://arxiv.org/html/2510.17564v1](https://arxiv.org/html/2510.17564v1)  
29. Integrating LTL Constraints into PPO for Safe Reinforcement Learning \- arXiv, accessed May 13, 2026, [https://arxiv.org/html/2603.01292v1](https://arxiv.org/html/2603.01292v1)  
30. Augmented Proximal Policy Optimization for Safe Reinforcement Learning, accessed May 13, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/25888/25660](https://ojs.aaai.org/index.php/AAAI/article/view/25888/25660)  
31. Gradient Penalty Regularization \- Emergent Mind, accessed May 13, 2026, [https://www.emergentmind.com/topics/gradient-penalty](https://www.emergentmind.com/topics/gradient-penalty)  
32. Spectral Normalization for Lipschitz-Constrained Policies on Learning Humanoid Locomotion \- arXiv, accessed May 13, 2026, [https://arxiv.org/html/2504.08246v1](https://arxiv.org/html/2504.08246v1)  
33. GitHub \- InternRobotics/HoST: \[RSS 2025 Best Systems Paper Finalist\] Official implementation of "Learning Humanoid Standing-up Control across Diverse Postures", accessed May 13, 2026, [https://github.com/InternRobotics/HoST](https://github.com/InternRobotics/HoST)  
34. AgibotTech/agibot\_x1\_train: The reinforcement learning training code for AgiBot X1., accessed May 13, 2026, [https://github.com/AgibotTech/agibot\_x1\_train](https://github.com/AgibotTech/agibot_x1_train)  
35. shaoxiang/awesome-isaac-sim: A curated collection of essential resources, tutorials, and projects for NVIDIA Isaac Sim, the powerful platform for designing, simulating, testing, and training AI-driven robots and autonomous machines with GPU-accelerated multi-physics simulations. · GitHub, accessed May 13, 2026, [https://github.com/shaoxiang/awesome-isaac-sim](https://github.com/shaoxiang/awesome-isaac-sim)  
36. Deep Reinforcement Learning for Robotic Manipulation on UR10e \- WebThesis, accessed May 13, 2026, [https://webthesis.biblio.polito.it/38675/1/tesi.pdf](https://webthesis.biblio.polito.it/38675/1/tesi.pdf)  
37. Correct way to do Lagrange dual optimization PyTorch? \- Stack Overflow, accessed May 13, 2026, [https://stackoverflow.com/questions/77508682/correct-way-to-do-lagrange-dual-optimization-pytorch](https://stackoverflow.com/questions/77508682/correct-way-to-do-lagrange-dual-optimization-pytorch)  
38. \[2602.06445\] ECO: Energy-Constrained Optimization with Reinforcement Learning for Humanoid Walking \- arXiv, accessed May 13, 2026, [https://arxiv.org/abs/2602.06445](https://arxiv.org/abs/2602.06445)  
39. Isaac Gym \- Download Archive \- NVIDIA Developer, accessed May 13, 2026, [https://developer.nvidia.com/isaac-gym/download](https://developer.nvidia.com/isaac-gym/download)  
40. Installation — Isaac Gym documentation, accessed May 13, 2026, [https://docs.robotsfan.com/isaacgym/install.html](https://docs.robotsfan.com/isaacgym/install.html)  
41. What's the minimum GPU configuration to use Isaac gym? \- NVIDIA Developer Forums, accessed May 13, 2026, [https://forums.developer.nvidia.com/t/whats-the-minimum-gpu-configuration-to-use-isaac-gym/197681](https://forums.developer.nvidia.com/t/whats-the-minimum-gpu-configuration-to-use-isaac-gym/197681)  
42. Local Installation — Isaac Lab Documentation, accessed May 13, 2026, [https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)  
43. MuJoCo \- Gymnasium Documentation, accessed May 13, 2026, [https://gymnasium.farama.org/environments/mujoco/](https://gymnasium.farama.org/environments/mujoco/)  
44. Train a Humanoid Locomotion Policy with Isaac Lab on DGX Spark | Arm Learning Paths, accessed May 13, 2026, [https://learn.arm.com/learning-paths/laptops-and-desktops/dgx\_spark\_isaac\_robotics/4\_isaac\_rfl/](https://learn.arm.com/learning-paths/laptops-and-desktops/dgx_spark_isaac_robotics/4_isaac_rfl/)  
45. Training Environments — Isaac Lab Documentation, accessed May 13, 2026, [https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/training-environments.html](https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/training-environments.html)  
46. Isaac Sim Requirements, accessed May 13, 2026, [https://docs.isaacsim.omniverse.nvidia.com/4.2.0/installation/requirements.html](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/installation/requirements.html)  
47. isaac-sim/OmniIsaacGymEnvs: Reinforcement Learning Environments for Omniverse Isaac Gym \- GitHub, accessed May 13, 2026, [https://github.com/isaac-sim/OmniIsaacGymEnvs](https://github.com/isaac-sim/OmniIsaacGymEnvs)  
48. Isaac Sim Requirements \- NVIDIA, accessed May 13, 2026, [https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)  
49. Isaac Sim Requirements, accessed May 13, 2026, [https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html)  
50. Training & Deploying HOVER Policy — Isaac Lab Documentation, accessed May 13, 2026, [https://isaac-sim.github.io/IsaacLab/main/source/policy\_deployment/00\_hover/hover\_policy.html](https://isaac-sim.github.io/IsaacLab/main/source/policy_deployment/00_hover/hover_policy.html)  
51. Introduction to Isaac Lab Reinforcement Learning with the Isaac Humanoid \- Medium, accessed May 13, 2026, [https://medium.com/correll-lab/introduction-to-isaac-lab-reinforcement-learning-with-the-isaac-humanoid-062a4c3f6b99](https://medium.com/correll-lab/introduction-to-isaac-lab-reinforcement-learning-with-the-isaac-humanoid-062a4c3f6b99)  
52. Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies \- arXiv, accessed May 13, 2026, [https://arxiv.org/html/2410.11825v3](https://arxiv.org/html/2410.11825v3)  
53. Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies, accessed May 13, 2026, [https://lipschitz-constrained-policy.github.io/img/2025\_ICRA\_Humanoid\_Locomotion.pdf](https://lipschitz-constrained-policy.github.io/img/2025_ICRA_Humanoid_Locomotion.pdf)  
54. Inside the RL Gym: Reinforcement learning environments explained \- Toloka AI, accessed May 13, 2026, [https://toloka.ai/blog/inside-the-rl-gym-reinforcement-learning-environments-explained/](https://toloka.ai/blog/inside-the-rl-gym-reinforcement-learning-environments-explained/)  
55. Simulation \- MuJoCo Documentation \- Read the Docs, accessed May 13, 2026, [https://mujoco.readthedocs.io/en/stable/programming/simulation.html](https://mujoco.readthedocs.io/en/stable/programming/simulation.html)  
56. 如何部署到自己的机器人· Issue \#66 · roboterax/humanoid-gym \- GitHub, accessed May 13, 2026, [https://github.com/roboterax/humanoid-gym/issues/66](https://github.com/roboterax/humanoid-gym/issues/66)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGYAAAAYCAYAAAAI94jTAAAB8ElEQVR4Xu2YO0sEMRSFIyqCNoKloIWgjaCo4G+w8QfYiNj4QLCzsBFtRQWxFH+HiIWtiK2Ngp2NqKDg+3Evkxkzx7tOZmY3m4V8cCA5d7JccnazM6NUIBD4QzfphfRNmoVaQj8axAAaAWta0FB/93jXGHM488Y8gQvIMxoBa9bRUOk97oH5IcwTJJN/ZoFiZAWDPJLO0GSkRa9oBKzJE0ybqlwTC29oBKzJEwz7TWjGSIve0QhYYxvMvTG+NMYJ0qIPNAwmSPukXix4Qr37swnmgjSjtUg6SpcjcBHziYaGr13W41MVBXjzW647PvSXFcyInptaMuoJUjBfaKjouivBWwGvXvjSX1Yw1kiL0NsWvD7ttYLPSA9ZyFgOZVGL/iQ6SA9oAk6D4Tl6B4LHbJJ20BSYzKFmvaYStegP2SONK/kzTZwHg2c1e0/gMXwMdqFZY1z1x7e2uDeI82BWBW9Nj69JC9qLNadrLnDVn3fB3Kn0vTZ/E/maIdIUaVj7nUq+cag1rvrzLhiGjwr24yODz2qenydXKLWlip3f1aBof/zC9j+ZeBmMDfz8U/T8dkHZ/ho2mHjddMr1h7L9NWwwx6QT0iAWPKFMf/xil//LblX0nmsjXU7wMphAFYORnqxH0QhY046Gkvc4EAiU5gcvc8cf4/xODQAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHcAAAAYCAYAAADAm2IFAAAEJklEQVR4Xu2ZSagVRxSGj1GjJmrEAUd8IkbBCQcQB7JTiQRBhIDGna4kEEVFxQkc406yUBCDPjUgCCaLiBuDK0UQJAlqwPFFUeMYNYpDHJLzc7rsuseq7q7bdUGlP/h53f+pPl2vqruqui5RRUXF+8MobVS8dfTRRlH+04aDXqxVrK2sOSoWi84kdWmtAxW0idVNm3kMYS3WpkUTSYO7FJvnVL5zb1FtHR+w/mb9m5w/ZXV4XTqMvyjNA/1Dkt+cX0iLNoTftJHHdW1YTCapdHvlN6Jzd1Gat0zngh9I8nTUASpf90Ek13+rA1Q+dx5PtJHHTW1YoKLHtclcZl3SZklwr5PJ309ULJSsRr5HEuuvAwXZQXL9xzpA6cjTQwcigZdttjZ9LGIN02bCB5TdSDE5wZrGOkpyP7wdZUAONLQL8z+11YGCZLWJibXSgYi80oYPXyUNprLndSAiaOSXyfHPJPcbn4aD6UmSwzVs4l6I4e2tF1z/TJvMWJLYLzpQgC9Z+8g9mugvmSNU4OHBogIJs+hOaQdD22vDUbjP6p0c7yG5z/Q0HMw2khyuYROLoUfaDMA8OOuVvyTxVyq/CHb7QhjBDMjXZJ2Dj0jWFJkcJhl6i2AqbxSLr1hXrfMtJPnnWl4opo4PSVbJ6Eycv2B9bZWrB/PgYBqBziTnWeuWLDBibbDO11Bt+/qG4Nw+yCrQheS7VmMaCm9ADJDLHmJWJN4yywsl9gNo48sN75Q2cxjA+k6bzB2St3Mqa4KKGdBOn2rTxsxzGsyvN7Rp4fsHQ8E341nWMUvnSHLvtcqFgF0cXL9bByLhe7BjtQkYzlpL2fkWskZo0wYrU41ZFEzRAQvEW7QZyAySRcmvJE/8H4kukuS/lhYN4nuS65t0IAL9SHJv1AGK27kAn5nrtGmRe69OrJ3Kw/YiLvxT+YaRJHHXXL2U1UabCaOtYyx0fDmAq6Gw+FugPBeua4uAuS6PZpLceqE2MfF99y2SW+PLBT5kHdCmC1cSU9HHrM8s30z2+p8D35DEsIhxgdhp1qTk+IvacA12Q5mhx3jjknMX2NXKamQfP5Fcc0gHFL7ceHDt2BjW8uS4aG5N1or+IBXcwcObOlB5diPZ+twu5KCFZEXqws7TXBt6A+QwZWcl3kySodo3l6Ix8El1l2S/FxsYmJeK0JV1m9wdB5Ab+8fIbVbfv9eUkPUCrt9PUsaQl9tFX9ZmbVqE5KIr2iiB78YYgvHpgI/1Iri2HzGNlFlF54EObBQhuX9kDdVmAqaAedrMAk9kDNqR7Ls2Ct9bG4PB1LifMUNz+14QgFEpCGxvzddmHfg+rWKA3zHr2dYrim+zIAahubM6F5+KwWQlLIrr57VYYFRoJLl7tSUIze3bwlxN6TZtEFjFVrzd+H69q6ioeKf5H7/RJ2q0vjIWAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADsAAAAYCAYAAABEHYUrAAACkElEQVR4Xu2XTahNURTHFz35SiIkEhMfA4rJk6/XY0AMTKQUiZGkMDBTSi8DA2VgZuY9vHozKUkmEol8FIaUjwGSKJF8rf/de3vr/s86551zbvdO3F/966z/2vfsvfbH2V2RLl3qsoSNksxmowTL2Ogkf9iowC02SvBAtYLNTvBO6q1Ook6xABM8lc12slf1ns2K1C12s+orm+0EszuXzYrULRag//lstoN50tpZTbRSLM7ubTYtk1SbVFtUW1XbovAMb9xo00IuqH6wSUxXnZKw3fMoKnanaljC+Dx2ScGEX5eQLNKhf62L+aW6yKbhhoSPF8BV8U38geUV+1u1IT73Sxi7h/dOOaoaMXEayB7VRuMzs9iIoJMBNiPnJTuINJmMV+wJCcUm8Lt7JrZ475Re83xYdTA+31eNNznLW9UE1RdOSOhkP5sR5C47nrc6XrFXJbS/ImPfp2iHo5mLnQ13ZpQzEs4wwMDXmxzA7/aRB85JyC0lH94a8oBXLO7PtBMgHJk8kJ/GZmKyZIudaOKEbfNCdcDEAPmT5IHvkp1AfKDYS3jFJuZI2L747XLKJfLe2+CZ6omJ0Xi3iQHOzB0T563UIHngqWQH8Mp4b2xCssWm1WQvj6JcI9lH8WcTJ++aaijKe+ElCavIrJXm9sdj/CnGH0wOeMWuNPFp1UsTW3aIP7YGqyWbfKR6Tp5ts138+3SRZN+VOCajK9SvmmJihotdIKG/1P5sc7oJ7L67bFbFDuqjarGJLWg3g82KcLFVQP+4LVoiFTtT9dgmiCOq12xWpG6x68TfcZXBl++hZO9Kj59S8OkvQd1icR31sNkJvLNYljrF3lStYrNT4M9D3rkei4VslABf/C7/HX8BTl6jVzcKYC8AAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,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>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEYAAAAYCAYAAABHqosDAAACzElEQVR4Xu2XS8hNURiGPyL3MHCJ9B93CWEgl6KEMpABAyNRjJAhZYBMJJdcBlJKhLGJUsxcCmGCjJASSgi5X773/9bqfN5/7bP3ueWcOk+9ddbzrbP2XvvsvdY+Ih06/G+msGhRprFoJn9YtDB3NTNZZnFQ80FsgsgXzVvND+dKsTPxWjOCZYuD+QxiWYl4EZirYn4i+fWaN+TagRWazywrgcnfYKksFas9JA83mly7gHMfyzLFOrHOy7ignBKrnXFuTHDtCtaa6yxTPJLsiaYesbOa7+RSbNWc1sziQgPYptnNsiDxRsglNfnZmp+aZ+TBL815lo6BYuMNCe1LmsXlcl2sFht7gKZ3+IxM9Z0KwPNNgk7YiW5rHmi+BdfPd3Kgto+l46tmv2uj/0rXrhW8L2Gs6c4dCq5acr8Tbysssp7HwaeA38jSEX/Fw5qhVIsMZlGAOK7nRcIVAd/pz9LzRNID4xeHH8kFMb+BpSNe7Jg7/5a7L9ZvcnngfQljXSQPd4UcSM3Jg3p81JOkfgXwScz34oKY38sywVyxRZrHP6I5Si6PnWLj8FoCt4Ac4GMyefXuDjdZSvYFA/DnWAZQu+zaXcFF4rjIK+cBdq/l5CJrpef5bHJus2ZCaPtkUakmu8Q6rOKC9BzYf74gtsAyeOzwlwK7ReS5Zotrg6yT4mMyqI0Ln0uhHfv714d7YrtXFmsk4zgnNB8178R2o/diW7BnhtiXX4pN1j+PpVBLsUPKJ4x1hE9wWPApToot+uO5EOiS8tjHg4s76LzYKbQrgSfkFstGgYMPZ1kA7FTHWDoOaPqyrJK8C4N6vcfIZLvYVlktuDNHib0EzqcaKPJGXQm8duCuA099IbBI6j9GLngzrrjlJcB6hv8qe8gDrHlLWNYAJn6fZQA/TB+WzSDvtq2GSSwazDXNHJbNAu85k1m2KAtZdOhQO38BTN6+dmAwdv8AAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAAAsUlEQVR4XmNgGJZAGogLgHgmECshiVshsTHAYiD+D8S3gdgbiFWBeBoQPwdiS6gcVgCS+AfE/OgSQFDJAJG/hC4BAn8Y8JgKBSD5IHTBD1AJTnQJNIBhuC5U8Ba6BBaAofkvVBCbPwkCkEYME4kFZGtmZoBofIkugQVgtYAYmy2AOAFdEATuMkA0g1yBDYDEX6ELIgOQZlAiQTfACIhfo4lhBbsZEF74CqVTUVSMgqEIAG1gK0HBSgf2AAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAYCAYAAADDLGwtAAAAm0lEQVR4XmNgGNRAAohl0AWRwUIg/g/FRWhyGECTAaKQBV0CHaxkgCgkCECKvqILwkAPEDdB2SCFNUhyYFAJxL+gbFUGhEfY4SqAIBUqyIEkdgkqhgJAAs+xiH1HFvCACqYjC0LFGpAFlkEFkYEKVAzZKQxToILIYAmS2FKYIDeSIAgEQ/kwMRRDnKECIJwNFfsH5QvBFI0CnAAAenEoOjLVGH8AAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAnCAYAAACylRSjAAAC9klEQVR4Xu3cS6hVVRgH8NUTK+xlJdFARwVCQeig0GwWRFQEQSMJQRv0cFA0DJImUmMHRYNCCpwJNqgGPQzBUYNUDHQiqaAEET3oAdX3udfmrrs6ec+dyLmn3w/+7LW/ve++Z/ix1l67FAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJgbt/SFJdwYuaIvAgAwnVORvyM/RzZ313o3R17ri1PaHbmrLwIAMJ1s2KZpps70hWX6rS8AADCdbNiWksuah/viMn3dFwAAVqpvaw5E9kW+iTwbeT/yfXNfNlrPRI5FHqm147X+XeTpOv68XpvkqsjJvjjBo5FXmvNs4H4pw/PbrGru6b0UubovAgCsVH8242yEHqzjbNhur+OxEbumLJ4ly+bqRGRLGRqyS3ks8nJz/kBkaxneZ2ubr3wHLe8dfVaPp+vxo/FCtSGyrqvl78n34AAA5sK5ZpzN2DgzdT6yto4/ibwdub/e08rNAUe72iQ5O9fOeuVuzjV1/F7kvjp+K/JwHY+ujeys4xea+o4yNHu/NrV0b+S2rgYAsGKdbcb/1bCNy6NXlsWzcDdE9kS+LMPSZa99Xt/ovVoWPsHxe1PfXoZl2dYXZWjAchbvq6Z+oR7bv0+5RJu/DQBgxcvlx9xR+Xzk9TI0VXsjuyJ/RN4tw+zWX5HrIvvr+MPIO2VYTt0UeaIMs1xvlMXGl/8/LsPMWSvfZ8vl0HwHrnV3Gf5Pq232xvEdZXhuLq0eXLh8Ub9sCgAw93ImbJxtW86Haa+PbCyT3yfLxi9lc9Y/86fuvJ0ty8YxrS9Dw7e6LCytjn7ozgEAWKaHIkfqOD+km56sx/Ri5J7mfJKc+dsWOdTV7ywL77sBAPxvfFoWf2rjcsjPiNzaF5dwU+SpvggAMO/Gz3XkJgQAAGbQj/XY7/IEAGBG5O7PN8u/NwIAADAj8hMfucvz8f4CAACz4YPIc30RAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA5sg/KB1r7G9+rWkAAAAASUVORK5CYII=>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAiCAYAAADiWIUQAAAEC0lEQVR4Xu3cT6hVVRTH8W1aWhYVSSEOBImyIpAcpEISKChGIP0RG4rUoIggQtCRqIjo1IGFOmgSaUoIliIZhUg0ULQ0lUgNFUIhVLRCzdaPvTeuuzr3vfOeWrzr9wM/zjnr3Hvee2fyFnvvc1ICAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKDnDbN8E4stHIkFAACA2+F5y5+WJ+MJc5flaiy2pO92MyIWbrFjlt8t1y0bwrkmf8RCS/datsZiC/fFQhcrLddiEQAA3JnmpeaGTSNPP8diC/reslh0DsXCbfCC5UwsdjE5Fgbgc8uoWOxipOWE5elQ74uaTgAAgPRaam7YBksN4P/dsJ2wfBiLDd6KhQF61rIuFoMHLDst78QT/Rhu+TsWAQBA7zhbtpq2G215Jd0YrXko5UagjvS8avkh5WnM2ZaTpe6/I5ctd1tesnxWavvKsZoLffZxyxbLJ6UeqfZr2Va7yvZBy2+ufjPajkxdCcefWg5ajltOWQ6k/Df2pdu05RrLR7HYD41OqqHV/dR09KbO0wAAoJeoYXkj5WZN//xrrTqXOhu2Se6c1nSpqZP6nTGWj1Nef6XUur+mRptEjUrbETaN7Kkxqr6wvO6OB0NTlHHt3duWbZZFlq9d3X/uMcujlrmW+1NuOr1plhXp32vQujWHutZ+y4x4og+H3f6PKU/tAgCAHqUGTP/w1UzUNVZtG7aLlnfLfv3OFMt6ywQXf95Tw7Y8Fp2f3L6mY4+6Y40oxUZpoJakziZQDWK9B1pHtsCdi42d1BHG0x3VlNaW7Zcd1eZ74GlKVKOI8+OJBv5a/V0XAAAMcW+6/dVl6xsATQU+U/Zjw6anRuOonBqe3WVf9HSp+GvWkadVJfJi2XrHy3axZbzlgjv3neVld9yW1qvNKvv6nTS1Kwst75X96hG3H6dE5duy9X/bjpSbPU1ZxmnKblOikaan9ffV+96kXuvhlBvnse4cAADoMRoN25Nyg1BfsfGc5ZeUR3v06gs1K+NSbtjU7GxPee3bxPJ58U3LnJTXmO1NN0bnNHWq6+iaGi0TTSdqvZt+dpO/LN+7YzVtWjem95oNdgpwpuWDlNfH6fes4iiVRgq998Px1JTviejdbPVe1JE43SP/ypKnLBvd8c1So/2VZbrlUsrrAQEAABrVkZ3Y8Aw1aiT9e9/8dGj1RCw0qKOA9WGOSuv67gk1AACA/4RGlDQFeD6eGGI0tauROz2tGacyK41k9Wdpyg8QeHpitj4t20QPT+gVJ00BAAC4JbRY/k6gEbjNsdiCf1gCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIDqHxzwqJIrbOAsAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAYCAYAAACvKj4oAAACYElEQVR4Xu2XTUhVQRiGP6kwEQmEilJRyUWbkBKEahGIiRpB2CaINrVrmRC1aZHgotbWrlVtpJ0gEbgqdOPCv5WbAlepq0CM6O97mTn69Trn3DOdcxXiPvDCzDt/Z+bONzNXpEaNGLrZKImzbBwEF1TTbJbEvOocm5XYVP1Q/fb6pXr5V438HBHXVzXBNzaymYdkgkX4qephs2QGVFts5gGTW2QzAqxq0QXKC8ZpYTOLEXGNbnBBBCuqd2xWCcTiRzazWJLiq4/2/WwaLqquqgZVQ6phL6T7TL083JLI7y0j/tLa18tu/1mKJaoNKi+wSXSYdJ1Jg0OSPiD8Yz79nvwsTrNBVGq/Q1b84dj/oppQdYnbTrhCuPPOgAcw8cMmb+t8N2kLHglvVO2qF1RmQV9H2QyxLOGPOyVh/7lqljx8TKiu5YxqzadxWX8yZRbbT1afKGtiEyt9nzxUDHUEr5lN5ZK4w4IJ9WHBg+KOTz+UcH0cdg9MPlQnIVjGkznp85PGAx9U2+QlnGDDExzQg5i15WOUT7DeecozwTKYdykfqghvlM0KoM0VNj24H+04t30eMZ7Q5r3XXhuyNxQSbkr4u+WJ6qvqlbgKaZclykLbE9xjw/NZNcWmB0+4x+RhDBxCCdi2qyaP8uMmb8HE59iMAZ3jXwFzTdIfur2Ssqo5wbYd92mMkRYigH/9aJ6JOxQsreKuiyzwT+Rf/7Ph8Hrq0xjbXi+Wy5J+xUTxSNxKvVWtS/jkZBBH39iMACGD2GvgAgO2e9rk94Xr4i7qajAj7nQ9cLIe3UXANq7xX/IHK0+PI7Dke20AAAAASUVORK5CYII=>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAAYCAYAAAD+vg1LAAAA5klEQVR4XmNgGAUDCd4A8R8g/g/F/4B4OooKCgHMYKoDkKEX0QUpBUEMEIMD0CUoBZcYaBgMNDP4ArogpQBf+E5hgMgtAuJZDBDLP6CowAMuM+AOBmYGTLlrQPwETYxBEYiz0MTwhW8hEP9CEwNloD1oYhiGiEP5q5DEkME7IG5F4vsC8UskPhyADElC4+NyLQjALF0HxD8YIPGBFdQB8ScgnscA0XQEVRoFMDFgWorOJwsUA/FvJD47A5UM/gzE7Uh8UKTDDE5EEica8ALxNwaIwV+BeD5UnJEBYnANEJdDxUYBHQAAlrs/Ypz1zE8AAAAASUVORK5CYII=>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAAYCAYAAAD+vg1LAAAA3klEQVR4XmNgGAUDCd4A8R8g/g/F/4B4OooKCgHMYKoDkKEX0QUpBUEMEIMD0CUoBZcYaBgMNDP4ArogGlBAYjMisXECfOHLCsQvgXgqEKsAsScDJCkS5bvLDNgVSjJgF+8G4mPogopAnIUmhit8QWJC6IJAYAXEruiC6IaIQ/mrkMRA4DAQf0cTgwExdAEQABmShMbH5dpidEF8oA6IPwHxPAaI5iOo0nCAKxhAIBldgBQAMtgIXRAIvIGYG12QFNDFACn1kIEMAyTZUQwqGCAuXwPErxiwpIRRQB8AAIagNMkxiITmAAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAYCAYAAABqWKS5AAABnElEQVR4Xu2VSytFURTHl1AYKCKPPGYmJgaKJANTM2VkbCoMSMnESBkZmCilpOQDKDHx+AISpRRFMUKUUFjrrn2uc/5n3+s8bh61f/Uf7N9a9+x999ntQ+RwpKYbxX9ihPPB2cLCL7JJuqYTTgXUQtySNsflmrPHKcJCQkpJ19FixsVm3JDtsNBJ2jSEhQjIBEecS0451OKyz7kCt0ARNlYaXlHGZJvzwKnDQkRkDUvgeozPywFFaIrIKueN046FPHhHZAZ8q/GD4ANUkzbNYiEF86TP7MOChQ7S3gnwtcZPgQ8hTYXafY81zgtKC/2kc4+CrzJ+GXyAU845aWMJ1JKwy7nj1GAhB22kc4+Bl9+LnwOf5YKzQV/nbj1QjY786WPSTSiD2nfIdStzT4NvNn4YfAa548984yRHp5JzwzmkdHe+zJvrtgnd9fecZ3DjpM2N4G3Uc55I31oheCf9ZviZJMtmPtqkQTw+BJEjIh+QQtJL4TXJeNEvmoyUM25jh8IP+Sm8nZa3KR/NlWDZkWEgRv4cXTHicDgS8Amd918uyG2w7AAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,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>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAZCAYAAADnstS2AAAAhUlEQVR4XmNgGPrgPxA/AGIWNHGcAKQBhIkCExggijnQJXABkOKN6IK4AElOyWCAKFZGl8AFQIpvoQtiA+IMJDjlOhDfZYAoxhvmD4B4JRAzM0AUL0ORRQIvGVDdidMpH4D4O5pYIQNEsRSy4GeoIDYAEr8E48hABUBuxAb2MOA2aBQQBwC0ciJVn07c0AAAAABJRU5ErkJggg==>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAXCAYAAAAyet74AAAAn0lEQVR4XmNgGHQgDYgL0AWRAQcQ/wdicSC2AeKfqNIIAFJkhcZnR+KDwRYg/o0mBlLoiiwgDBUMRBaEihUhC2yDCiIDFaiYO7IgSOAvEDNB+YxAPA0qDgcsUIF7QHwACYPEUBQmQQXkkQWhYteRBZZCBZHBIixiDNlYBEF8kE0oAGQlssI+IP6GxEcBX4E4BIiroGy8wBmIRdEFRzYAABkLJxqYpSHRAAAAAElFTkSuQmCC>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADoAAAAYCAYAAACr3+4VAAACkElEQVR4Xu2XS4iNYRjHH8ZdiDFZsbCyIKQk1DQ1aYaysLKbLIgFinJb2MjGzkIuWZBmagoLS82CnY0FuYXYoJRSrsn1+fe8r/Oc/5z3O+/3nWMWOr/613l/z3v5znd5z3dEOnRIsYnFBLOdRVkWas5rNnPBsVtzhKVjieaiZh4XMlmm6WJJLNXcY5lDt+Y3BQsyizVvWDoeaM6Ez5hjkqvlskZs7BcuEKfFLko2K8QmnhXa8Ytu/dujBvwMloERzVvXfqq54dpl2Cu2Fu6OItAnG3Q+4dqvNe9dO7Je841lALcpL/pQ84tcGTAf5ijiimTewvdl/AGm+C7pZxNzoM4ud+5GfJDm4xud4IagE56rHNB3JktlqliNd0K4n+TK0C82R7MdHn3msvTsEeuEZ7QZcyR95rD5oOZ3yinBjTpXBczxkSWBPkdZenCrpQ6e6ZN0X3jklsvz4AZjp4o8k/S6EdQvs/SUubV2SHpB+Jfk3gXfCgc0L8Tm2Uc1z1fNHZaRlWIT+N22iCFJHzg8DordI3JlwHF9Dp/jHZMC/e6yjNyW4sHMOkn3h1/u2vE2r/KyAOIzH7kW2qn5ULvOMtLsjO+S+sXmU9sDv4Dax107gh17P0tiWGz8ZOfirn7WOQ9qx1iCm2JFH/ywX9CsEjsgOFxFD9w0cuCx2A4OLmle1Up1xLXWciFwVazewwWpjW0EPC5EHb1iXwpnaYPYuyLa/KXHDRSrHWQZ+CFWx8lKsU3srSu1Q2L8RpaBLWL11eRnB99WDmk+sSwJfo8Ps2yBc2LPcNvB2cPLQFVSV7Mqbb+akQGxH/Eq4K/gGMsWwIZ3kmU7OaXZyTKD6SxaYJHmCct/wRCLCaboTanDf8cf592p58NyelkAAAAASUVORK5CYII=>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFMAAAAYCAYAAACGLcGvAAADD0lEQVR4Xu2Y26tNURTGh9xCLnlwKTpHFG9ECSn7gQdJSfkDXFLkjQelPMk1TxJRUpJICSUljnJ5ktySB8qDyDW8uF/Gt8da9jzfGXPtNc92asf61VdrfWOuOedea6yx5twiFRUVItPZaMIA1Wg2K0Rmqi6wGTBCdVTVQf571XDy2o2JYvOP8kb1XfUr00/VgW4tyjNQrK8Yp1Q3s2OM09kI1cH47chjadwfJEtT8sat8EM1i82MtWI3MOeQ6nVwDrarbpHXLqyThPuDhnfZTGCYFA+G2LjgfH/mMfD6sRlhDht9yEvx59uD5WINl3EggQeqi2xmoJTwRG47HkB2b2MzwmKxPtZzoA/AOJfY9Lgn/g9LAdcvZDMDsYOO542JdkV112OuWF+7ONACWGHg7dmUnaP/+Y1wnNgPSyF2/Qqx2KjAw0Th4dVh5km8r2ZMVn1WneBAImelMbeaJN4fNLzDJtEZHHNN6y/xwd6Jxa4Gwkcmlkmoq7G+yoI161vVDQ6U4Lj0HB+lhz2XonqJpQ6eENJ9iliNwrKJO57keDneUz3jeCFFsRQGq56oHoo98DJgbM5seF3kudwXf/Ljxff3SGOtmNMhflsA/5zjFdXFWF+p4A3CXJEQQynmsVNs7Knkw6uRV88g/vJ5mQPgeVs81LRFbIrfB4C/IThH7YQXLpNCxkq8r7IMEltoPxJ7u8ryXHqOvcbx6vCNyyeOnUnINdUn8nLGsJHhDijmo5TkfFBdDs4ZrB1jfTUDDx/LsC4OlATLOx77WeDh+A8wV9E5XwzgbWSzCbhmAZvKYdXJ7Hil6msQ89gn/pyKwFf8i+oYBxLhlcTe7BwfM9Btx7ZV9VF1RKzR9TAYgJj3ioPVbGQ8VZ1nMyP/ol/hgMM31Q42I8wQ26JiG/q3wLoyT7KaWK2NJV0pcKG3oV8itm30mC0tDBiQ0gfqWduzW6z2hEwQWyIVgSyZxmYCW8R2Y/8cm8Wy5LTqlfhfcAb/92EH0ltSsvK/YKnYLiKVF6qRbFbE//CIMUQsqysqKnrNbwwnxvMOe7z7AAAAAElFTkSuQmCC>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFMAAAAYCAYAAACGLcGvAAADCElEQVR4Xu2YS6hNURjHP6E88ojyKLpXKWaKEiJnwEBSUsjQI0UKMVC3jLxlJBElE4mUUFLiKjKSvIYGBiLPMEDe3/+utZx1//db+5y1D7q0f/Wvvf7fenxnn7XXXmuLVFRUiExhowH9VCPYrBCZqrrEZsRQ1QlVG/lvVUPI622MF5d/kleqr6ofXt9Vh7vVaJ7+4vpKcUZ1y19jnPZ6qAuM3xt5JPX7g8nSkFC5Fb6pprHpWSvuBgaOql5GZbBLdZu83sI6ybg/qHiPzQwGS/FgiI2Jyoe8x8Drw2aLoL+RbGbyXOx8e7BEXMXFHMjgoeoymx4sJZzIHcMDmN072CwJXmovVJ0cKAFyvcKmxX2xf1gOaD+PTQ9iRwzPGhP1itbdZpik+qw6yYEMsMPA07PVl5Hr7Ho4TeqH5ZBqv1RcbHjkIVF4eHSYWZLuqxE1cW33k5/LeannVpPM+4OKd9kk2qNrXtP6SnqwN+Ji1yPhJQNvb6gUgXU11VeKFeLabOZACTCbeXwsPeyZFK2X2OrgH8J0n6haIG7bxB1PMLyA9a+eM7yYoljMJnF1l3OgBdDfKcPrJM/kgdjJjxXbxyMU9oqBNrHrAvgXDK9oXUz1xewUV3cOB0qyR1x/WHNj4NXI65pB68lDRSt5eNYRD2vafDbF7gPA3xCVsXbCi7dJMaMl3VeKMEOXcSCTp9Jz7DWG1wXfuJA4TiYxN1QfyQuMYsNjDijOx1ISeKe6GpWZGZLuqxHhZbeRA02C7R2P/STycP0LmKuozI0BvC1sNgBt5rKpHFOd9tcrxW1Zijgodk45zBR32trNgQbwTuKAL7/25W4ntu2q96rj4irdjIMRiFmPOFjNhuex6iKbnvBGv8YBgy+SfxNStKs+iPuw0izYV4ZJVlMNisqlQEPrQL9Q3LHRYrq0MGDE7+iDGaAax+bfYp+4I2AMksEWqQg8WpPZzKBD3Gnsv2ObuFlyVtz51nqDM/je94nNDP7ErPynWSTlzsTPVMPYrEh/8EgxUNysrqioKM1PfKLGx5qAwzkAAAAASUVORK5CYII=>

[image20]: <data:image/png;base64,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>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAZCAYAAAAIcL+IAAAAkElEQVR4XmNgGAUDAn4C8ScgPgTlPwXiE0D8H4hPwxRFAbEBEFtDJf7BJIBAGSoGBn+g9CKoICNMAgg0oWJgUAul7yELQkEaFjGwwDE0MZBbsSr0wCK2B1kgAiqIDkBiNsgCl6GCyCAJixjY55PRxB4yYFGIDYAU7UQXxAZACs3QBdEBLs9hgBIgvoEuOEAAAG2eJv3BYhASAAAAAElFTkSuQmCC>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGMAAAAYCAYAAADu3kOXAAACcklEQVR4Xu2YP2gVQRDGJyaiFipiDGjQEFBBBCEhpRDEwiIgUVsbBf9Ullr4B7SQEJEQQUQ0EU2wEEUULC3VQkQtFNJYKRaSoIVioeh8zqxvbriXdwnkZRf2Bx/v9pu992bvbmf3HlEmk8lkMg1Zxer1Zqp8YX1n/TH6WOgRJ9tZv0ny/epiyTNFMrDUQM6nvJk6YVakxDaSnJf6QOpgUG+8GTn3aJYHaID1kNXvA5Gzn2RQgz4QISdYY6zlJDn/KIaJzmhgl7aHWKO1cPS8pVmesEjYRJJjF0lZCmX1rO0UbsQS461lPTft+bCDNVFHd1i3WbdY46ybrBty2ryosl5gC7nSm01iGUl+R4wXZjNi/4HxTo9xx46qZ7mkXrvzYwG51VsvHrFekJReLJgzJNvhRvTNQY3A1tVf07vewxoB4wHrIus4yawow39ZLBwgyW2vD5D4KA8WjK/KWPB9VdWq59QDv/e5xPtpjXNqNqKDqvWzdLOG56gqXHBtzOqy3H6xTnpTeeaNBWQrSX7HnA/vvDV2qlnGQXN8lXVdj1+xPphYM5kmyRd5B9DGm6xlt/r12OCNBQSl3eeyWb0VrC2sQyEA83BoKHja9pg2+nSSbHvbtL0Y4Hfvm/ZL9dYbD+DvkG/OW0yQY3i4sYlAO1zD9/r5D+yiULtCh6c2qMDHorfGB5pMF0kukyQLMS746kIPAX2ueFPB7rHZYAagbCKvJ+p90jY2TJVZR7W7iM/CVixSMDMee5NpYZ32ZkqMsK7pMW4GytTrWjhKNlJ5KfU7muRAOQjlaR9JncZgY6eH5Ibg5RVv6JeL4Uwmk8lk0uQvIySa7jvEAPMAAAAASUVORK5CYII=>

[image23]: <data:image/png;base64,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>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAYCAYAAADpnJ2CAAABOUlEQVR4Xu2VMUtCURTH/yku4eIYtDi0NNXWdxDH8DOEQ34GJ7eWDKI12v0SLg4FOkRQizSUqKBFQ5Ge07k3zztdIXzvLeIP/uD7/89913OvB4EN68YVaUKaKb1GKoA3lbFK0Xg1/MtC1Em31ozDFmSzBxsQXVLZmnGpQjasKC9D+iZtKy8xXhA9zkPSs3pOHH1/F+5zexEnD29wB+l03z0v+wHFxt/fl/KunXeqvBAN0iNk7TmpSfogtXSRZYBwN//tkmeYN9bwuiPj/bLsxUOIX7SBgWvyAe/AeD9kIWHHBsQeJBvbQMGjY78sd8unFuQMsuDYBg7ffcEGjhok75HuSZ+kk0iF4wZy9iOnd8iQe3YhnfEs9iG1Tyr3cM2lNdOEu9uxZlrk8Pf+UmMK+cviq9Dzu2FNmQPlElXPOBlqdgAAAABJRU5ErkJggg==>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAABGklEQVR4XmNgGAWDDcwG4k9A/B8Jv0JRwcDwBUkOhL1RpYkDMM3YQBMQn0cXJAUwMkAMv4UuAQSXgdgXXZBUkM0AsSAcSYwJiP8BMReSGNngJQNq8BgC8VMkPsUAOfynQdnHENKUA5CBFxggPtGC8nFFOMkAFv5/kMSWQMXykcSwgYlA/JgBonYGmhwcvGbA7lpiffEbiPvRBZEBLoPeMkDEFdEl0ABIDR+6IAwwM0AUnEaXAAJVBojce3QJJABKwtgcBwcgr4EUhKJLQAHMd4LoElDQyoAad3CwjAFS/ryD4q8MkEwFAzIMEJeD8gIoEkFq7yHJwwAo/LvRBakJQL7jRxekFuBmIBD+lIDvDJAiHIRBwTQKBhkAAHpQTOc0aD44AAAAAElFTkSuQmCC>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAYCAYAAAAGXva8AAABPklEQVR4Xu2UPy9EURDFD6LRqVej0OiUEt9AthQfQkGn0BG1SNhGKz6KRkGikyDxL0EQQkQIzuzcm503ZhPhbrPZX3KSfefM3pu58+4DenQ7W9QT9WV0U6kAnk0mmq7GfycvGLFM7Xvzv/RBNzzyATmk6t4swRx001nj9VOf1JDxinKN6tFOUJfmuSPYeTbS791W3BlkkwNox+Ppud1LVYQ8zw/jbSdv3ngRS9Qx9L8b1CZ1QV3ZoohbxF39tlu546vOW6FOnVeh3eJ3UH/UBw6pid7waM0mA9BwzwdkDJo9+MAg1ypavIbYb7IGDWd8kMinMOyDxAL15k2yiGDTHegs7pNeoB+CzAi0Q7mr59DaE5NnpEbm55EN171Zimie8mGxDRRlED+PcIp6dV4xHtGat3T1Tp1Rk7aoR3fzDWoTWdnovM53AAAAAElFTkSuQmCC>

[image27]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAYCAYAAAAYl8YPAAAAy0lEQVR4XmNgGJHAAl2AEpAKxP+BeDu6BBqIB2JHdEFs4CUDxEBswBiIvwKxERCLAPFvIJZCUYEGTBgghoWiSyCBV0B8D10QFwAZ9gtdEAj0gPgfA8RlIBeB1AmjqMACjjDg9ioIxAGxLbogLiDEADGsDl2CXAAyDJ/riAbXgfguA8QwFjQ5ksADIF4JxMwMEMOWociSAEBp7BYSn2yvfgDi72hihQwQw/AmTHTwmQG3C0Dil9AFcQEZBogGUBhhA3sYcFs0CkYBMQAAyrko04zawNYAAAAASUVORK5CYII=>

[image28]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAYCAYAAABqWKS5AAACNklEQVR4Xu2WzUtVQRjGX0kXEVm0SIVChRbtAoMIN21sISKI0B+gggtp1yYQXEWSLSOKhHDjwhCEAnGjCEYrEb/+AHdiRWgbI/t6H2cm5z6+c+49noWC/uCBM7+ZMzN37pw5R+SM08EtFmWo1lxheRy0aKZZRtRqxjSN5Lc1F8mV5avml+avzx/Nq5IWlVMjrq8U7zSf/DXGaTqo2gfjH4kw+SL81txm6ekXN+HAG82XqAyeahbJVQQmvsIyBxck+8ejrj4qv/SOgatimUW3uJu6uCIH65oZlh5sTZ7okuEA/r0nLLNYFbujPOD+NpYe1L02nDUm2mU9N4dIdZSH1P0PxNVdjhyORrityAVaJd2XCRovsySaomvek+ckPeA3cXXzUfBQwj0LjSLwXKT6OkTWfsfRh9XBw3VD0y7uGOXOmw0XsP7VKcPFZNWVsCZ24wax/XM5OKsDjWK3BfDvDZe1r82+sEID5KyVAXDWKxt78j5LsfsA8A+jMvY+XHxsxtRJoi+eaGiIN1/MgmaXXOAqC485oDiPrRnY0cxGZeauJPqC7KWy1RDuEcsy4J57LJVRzYS/7tH8jOosXog9JxnSfNe8FdfgY2n1f1BnbRnQx8KzofnA0hNOnDmuMNjTDLPMAwbC1yHTIe4zwOKOJFYsJ4X7GBH3So+5Ju7IzAIfXjdZ5mBQ3Nu+MI/FrcKk5rPYJwxzXfODZQ4Kr3pROjXjLCtgU3OJ5XGQ+kBLcV7cv3bGieIfyMqLbep2DMkAAAAASUVORK5CYII=>

[image29]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACYAAAAYCAYAAACWTY9zAAAB7UlEQVR4Xu2WvUsdQRTFbwSjqBhJE+0MViFgpWKrgogYjEUwYAohpBRC/gEb06UVC0ELi4AoghCSWEQLizRiqWAhptBKQcEPgpqcw+ySeYfdN/Pek1T+4LAz596ZO++xd3fN7vn/PIQeqxlBvxql0AJNmyuex7UakXDPUzVj6IZuoTfQH4ml3EBVapbACLSlZgge5lNyzTrYHLSqZhlw7ydqFoMLnqnpkXXYcngLXaqZx6gVL/zKisdLJbhXGzQI/TSX/AIaKMhw/II21fSohxah3mTeAC1YfieyFuvmMgx9MJfIG5/j9wUZDsbfqZnANZPJeA/6AW1DD8yta0piPr+h72pmwQ2W1PRgvE9NUAfte/O0ecg3b6zsQgdqZsENhtT0YPypmqBL5meWfxifrxaR12nhJMZb1cyAeby3QnyxcE2bsXAS4z1qCtXm8thQIXagEzWVC4s7GJ8/yoT9W/vRG6foPOUKWlNT4eJQhxxC62qaW5u+Ozn2DzIPdXhzH+a9VFNhUlbH+Yxb9q9vN+fzUVMDTSVzShvDJ2uvAp5bRFJCbF6I19CRmikswtaehY4llscK9FnNMmDtZjVTGOTJea2VWDEq/dfY2Rtq+vArYtncfVEKzI/+MhD4HXeu5l3SCD1SM4IxNe6phL99Qm4FRCaQ7gAAAABJRU5ErkJggg==>

[image30]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAYCAYAAACFms+HAAABXklEQVR4Xu2UvytHYRTGDxkoIZRZkkIpk8lmUP4Ao5LRYjBhll9hM0rKarWxUQzKKn8AC5sUz+me9+u5R/fmvq6U3k89dc5zzr3n9L3v9xVJJBKJxH9iD1qgfB1aobxOapnVB91avAi9Qu+WX0NbFtdBrbPCg0q75WPQhMWzVP8ptc4ap3hZ8i9vpTgwA21485tUmaXHR+tPUIerfeFZ8i9juqEl6FLiF2fKZq1CLRYPSXFfA2048qbjXOpZvGzWPfRGufbql27QZeaAfJ65YarfURwoW3waGvGmETMroL09bByY2QbdWNxvNf3TnFjM6OKb3gRNkj1f9FljZinHkvXn4GFTkv0aIV+jPkYX3/amcQo9eNOImTUKXXgzFl18x5vElTci6YQOKZ+kOApdfNebRNFRqYJ+HT3zc9A8tA81c0NV9ArTe/URenE15Qzq9WYE4apk/SqD3kgk/pAPVmtgKNBDY6YAAAAASUVORK5CYII=>

[image31]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAAA0klEQVR4XmNgGAWkgtlA/AmI/yPhVygqGBi+IMmBsDeqNCqAKcIGmoD4PLogNsDIADHkFroEEFwGYl90QVwgmwFiUDiSGBMQ/wNiLiQxguAlA6q3DIH4KRKfaIAcPtOg7GMIaeIBSOMFBojLtKB8XAGPE8DC5w+S2BKoWD6SGEHwmgG77SS7CpeGtwwQcUV0CWyAmQGi+DS6BBCoMkDk3qNLYAP9DBDFoegSUABzrSC6BAwsY4Dkr3dQ/JUBkvhgQIYB4hJQWnrMAFF7D0l+FIwCALDWPUOqr0VdAAAAAElFTkSuQmCC>

[image32]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAiCAYAAADiWIUQAAAFLElEQVR4Xu3cV4gkVRTG8WNcs6IiPiisivhkRswKCipGEFFXMaAgCKJiAsH0YNYX8UVfBBFcQRBzAGVHjBhARTGiYMKcMafzceuOp89UTXdN94yz8P/Boe49NVtdU/swH7eq2gwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAsEDuyo0hfs2NEbycGwvg+dwYwbO54S7LjQXySBh/F8aL0TFhfJDXKmEOAAAm4J7caHzvdasN/jF+M4z7+iY3JuQvrztT7+k07+OUNL8qzYdZzWuN3AwusnIt/vF6remt7/W3lWu+ddNb0WxF/cXs2DA+1AhsAABM3L250XjR63wrAUSWeF0xvbe/vb0Oy80xren1rdflobet16Vh3tdHXquG+dVhPJt1vB71uiDvaKFVzd/CXNflyDCXlSmwHRfG+l0IbAAATNh9uWFlxefK1NPq2rqp14dC0Me5OabbvC5MPQXNGLj6Os1rzzC/Jozb6LP0mcvyjlkogB3ejBXeFDyzqTBe7IHt+DAmsAEAMA8eSPPtva73Oslr/9DXLbxoCyurSrql+oKVZ5e2GfiJmfIxxqFbb39aCQgHhH7+jDusnOdbXud4PTe4u5WOW10XxpE+9ymvtfOOEegcdU7abpD2VVNhHAPbK81Wt4Jvt/L7zYcHrQTJT61cX227xMB2hBHYAACYOP1hzu7ODZsZhB5rtg95He11VNi3ltfmYV4pZGRaodqsozYMP9cmn5P8kuZ7NdsPm+1NzVah4iz775mxKB73hjCOdN7Lvc7LO4bYw8rxz7ASIt8Z3D3tyTCOga2uHtZzjM/YKTB3PT+Xr22sTP93W1q5RvVz6ssXuuV8iQ0Gzbi6qFu7NbApzOrfEeAAABiTAlf2R25YeziS3FcA0urXbqkvbcdd3Wu7jtIq3mzyZ8vPudHIn11XFj+xcg5RPO6NYdxFq3ZaWRrFS173N+P1rP13kBjYfghjUSD+PfX2tfI2btdt63xtY3XRKmJ8W1XqiqNekqhiYFNwrwFNt4olX3sAANBTW2BrCxG5p5Uq3darf4yXhH07W3tgy8cYh4Lh17lpMz+jrhbW/rnNdtdmq6//2KQZVzFgjBLYREFJq5Wn5x2JVgC1elUp+NwS5lVbYFMQutbrQK/Hm55WNysFx67A1semXjdbCbX1+PVZx0OabbzOXYGtvmEcwx0AAJiDHNi28no79US37hTQ4lxvmOqhea3CnBz2KbDtHuaiP+Kfpd44Xrf225n6qowaGBQiX7VyjiusBJr4zJl+nx3CXE6w8jxeNWpgG0bX53OvH72+srKqp9uLestVwTO/kNEW2PTGrm5F68UIrQxO2eA1mFRgW+r1jNdOXu9ZeVYvvhihgLlRmHcFNtknzQEAwBzEwHam14leZ4depVWhtn4bBbb4pqXsYmVlaBJ0nj9ZexDQM2ujPFemf3txs9WKUvW+Db5lOqnA1ldbYBtmUoFtNnoZRc/JdT3DFgObrutSrx2n9wIAgDmpgW1jry9t9i+41crQMAd7vWtlZSY+G/ZFGI9Dz33pPPPbrdEHudFCq126rZdvoZ6a5l2BbT8rXxjbVnElcq76BrblVq7xG3nHhNVrNsozbPoZ/Wx+3g4AAPQUg49uF7atWkVzCV4P2/Dj9jFsFUmf9URujkBffJt1fa3HfNMt3Gpl+h42vtYDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP4//wLP+9ssEw/LswAAAABJRU5ErkJggg==>

[image33]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAApCAYAAACIn3XTAAAELklEQVR4Xu3dWajtUxwH8J8x85gyu16IlDlj8cKTkMhQuKGQQsYX3pAXPCCEpEwZ4sGD2UGmJEoeDHEjhAhFZtbvrr07a6+797774F5XPp/6ddbwP+3/Pk/f1vqv/4kAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACABdmiH1gBDugHOrv3A8XqXX+1rg8A8L/xRNPeptR2g/Zby6mdB9e1Xi71dTd2acwWtj7u+hny2t/bsmmvKC+Weq7US83YC6UeK3VRMwYAsFI91bS3jfnAdkqpP5q51t39QHFEqTtLXd+M7VXqmKY/zaalHmr6B8Y/F9hOKnVuPzjG56Veafpbl9qt6QMA/CuebtptYEu/lDqo6ae1S93WjaXXS+3ajf3W9Zfn96b9dwNbbqleVurhfmKKX6MGx3RiqUOaOQCAmWUIObvUo6VOKHVG1KBzbaldSv1U6vHBtT+UerLU81G3PtvVtKFnmnaGtTaw5ef0q2y3d/30SNR7eDvqZw1917TTmaVuKrUk6v38ODIb8X7TzqD4VwPbUaWu6wdnMPyu+Xfat50AAFiIDBUbDtqHDX7eGjWwpaNjPrCdXurBQTu9V2rdpp+ebdrbx2hgS7nqNLRWqQeafqsPdumTrp9BKn1UalGpr+anlrqvaU8LbOs37d5Wpd6MZb/nLPI75HNsd5T6rJsDAJjZaVGDxbcxH9zyubHhilKGomFgWxyjIejCUns3/TTXtHeIZQPbBaVuGbRvaCc64wLbB/3AwLhr0zVN++AYH9hyZW6nUp82c+OcHHWlbFq4a20e9XBByrCX95jbvwAAC5YrQCnDRD5jlq6OutWYLo/5k5+nxmhgy1OcmzT9NNe0xwW2lOFlg6gHESa5sR8ovuz6V5U6NOrWaXptfmqpPKE5NCmwDQ8xTAp941wc0+89vROjAe2eUh82fQCAmWVQyWfXji11/GBss6gnHHPl6a5S30RdacvAlu3cOs3rvx9c35pr2pMC25IY3Rrt5WGDPfrBGA1Va0a9n3xNRr7CY1HUz2u1108KbGnHqM/DLcQ6UUPnOFdG/X5XDPprRD2xmvdzyfAiAIBZrRf1WbJ+uy4DUc7l+MZRw87iqCtsef3w9GNvrmlPCmxpn36gkVul+Rm9e2P0WbLcdkx5bxs14ylD0rtNf1pgOzyW/X0AgP+kPEF6fz/YmWva0wLbOPtHXYXKwwzjZIh7tR+c4LwYDWGTAls+F5eHFr5o5lr7Rb1mXHmnGgCwyslt0SOjhphJ5pr2QgNbrordHNP/vVW+PDffqTZNBruzurFJgQ0AAAAAAAD438qTkGnWd48BALASDZ/3+rnUOTH/Ul0AAFYh+Q/P3+gHAQBYNexZ6rioL8k9v5sDAGAVYisUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFuZPZdyd1YzNW+0AAAAASUVORK5CYII=>

[image34]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAYCAYAAABjswTDAAACD0lEQVR4Xu2VMUiVURiGPwsrBQlDQ1sUxFykIbCWCAohcsgxwRCdIhJrKxqEcHGpob3FtIamliLaGwxSCcVJkGhIcKhFiTTft3POvd/9POd6/2v3huADD9zv/c79/3Pv+c/5RQ45uJy1QRFa4REbVos/NiiBcr6zb77DZhuWwDG4bcNKMgTXbJiBj/CJDSsFl7LFhhk4KlV6HM7Iv7kRr9Frw8AJeBVeg9dhn5efmdXkhxZlCv6yYYRR+Byesw3PMpy1IXkv7pcU825udHG24IwNFfXirtfg6zfwcr6d45FEVug+fK1q7mJyC15RuaXJBh7eYMKGik04qWqO5+pZBiQy2Qvq8xi84z9/kvQB/Q3Wwp+2Ie4GIzZUhJV6Ck+anuaiRCar0c3UQB4pfIbJK3hJ9Qi/N2wyTfjHgvxTYvRIeg5SJ7sne1zVAT1mBd5WNWH/sclinBe3EVMTuinpnizCBVVz4KCqybi4AzvAMV2qDtkLkwXYe6vqNp/FeCjp3t+G3pWsf6g6ZO/gtDd2sZfiNpHlNNyQwn2wKulT5guctyGJPcxzcMlkeswNiZ+n7bL7WoEH4nqU7//+wnYBHBP2RlnoSazDTlVrOK7RhhngCyj1g0smXOCUJJbIcw9+tWEGPsBnNsxKN/ws7sjai9+Sf0tlgc8034JVp5yl/C8TJXz2Us91jA6Jn+2HVJQdV5B29VIwxOMAAAAASUVORK5CYII=>

[image35]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAYCAYAAAAoG9cuAAAAaUlEQVR4XmNgGAXkgkQgXgbENugSMPAfiBWg7EogrkJIQcA+ID6BxAdp6ETig8EnqMR8IJZFk4MDTQaIIhh+jyqNClSA+DkDRCEKAAl8wSKGISCIxN8DxKuR+GDgAsT/GBDuKUaVHvEAANQqGExkV1LNAAAAAElFTkSuQmCC>

[image36]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAZCAYAAAAIcL+IAAAAcklEQVR4XmNgGAUDCi4B8Qsg/gbETkB8F1UaAv4DcR0S/y9UDAV8xCIIMhldDCzwHIvYF2SBEKhgOrIgVKwSWWAHVBAZyEHFWJEFp0AFkcESJLGlMEFuJEEQcIPyYWIohjhDBUA4Gyr2D8oXgikaBTgBAJv8IeeKuEwpAAAAAElFTkSuQmCC>

[image37]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAjCAYAAAApBFa1AAAFMUlEQVR4Xu3cV4gsRRTG8WMW4zWnFyNiwKyYxYTig4ErIoiIimJCxAD6oBjALKIPCgqComIAERHjyzUgqIiIImLALFcEwZxDfVSVc/pM98zc2R2ZWf4/OExV9ezMdu/DflRVtxkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAoOeKVF+l+iweWACWT3VeqldKW3SuqoPrmwAAwGxbLg4Em8SBGXRz6K/q2jXk/N903Qd992pxIFmnpX1lqpNL+8zyKiumOtL1AQDAFHk11YupXo4HWig0PBbGjk71YaqrS//UVJv/d3TudrT8+R9ZnvH6ONUXpT8pMbDt7tr7WC+0vj2kVi/vmw/fxoHgiVSrhLH7Xfs61xYF6w1dn8AGAMCU+ycOdPgj9DezXpjRklqlsLKy689V/Y5dy6sC01qlPQkxsO3h2vtaL7DdZN3X7rU4MAe/xYEOP4a+D2zXu/Y2qXZKtYYbI7ABADDl/o4DHS4OfR9W1F6vtBWstE9qvuxSXncur+/WAxMyamATXbstXV/WTnVpGJuLpXGgw13WXJJuC2z6Gz2Q6tpU2/YOE9gAAJhmp1v/zEybFVKt6fqLUv1e2gowcaZp1BA4ihrYtBx5Rqov3bFJGBTY9rNmYFNIiuc+bHn5LMs/4+unxjuaDo8DyXuWQ9htbkz71B50fR/YbnTtNgQ2AACmmJbbLghjR6S6xPI/8bqZXUHFb74/P9Wnlt93Z6o/3TGJIaZSQPR1muV9b6dY9zJqDWwnpbrb8j62ceh8NEO2l+UA2iUGtj1dOwY2UTjdoLS1b02zV132tvx5+oyjrP1czgn9uBduu1SbWt6D9kw49o1rE9gAAFgg/EyYNq1rM7/+eYuf9TnemoHqrVTnlvbDqa5yx6QrsI0jLom+Uw8sg/tSHZrqh1SHWd7Ef1zjHT2DAtv+1h/YNKv1V2m/4A8McFF5Vej1nrYcJg9xYz4oy/qWr+/P1rwTVPx194FN++0GIbABADBl6j/1rV1bfnFt0cxXpRkkP9PziOUlUm3+b7tjsyuwnT2gum4k2K281uBW3VJeNWt2gw2eNRM9FuPe0lboimGpWtbAJgpsukb1xghPd9Lqd/WBtwZlBchKQVIhUku/+plKf6dKM57flfYJ1r9X8E3XbgtsO1ieodPMppa1KwIbAABT5iXL4UV3fW7sxoftvdJSYqXluNtTfW79j5PQjFBcqhuXPvsay9+tB9oqnN1q+S5Mbe6Xry3/PgpMenyFlnjvKMek3gyx2Hp3RupGgQtLOxonsB2U6tc4WLyeaivr/b6XWe8mAB9sn7O8ZKq/z0Zu/CHX1mfo2irEfWLNZ7MphCn0VW2B7RjL5yB6f0VgAwBgymhmTM83i8Hj8dBfKfQVzry4t6pSWKohadJ0Dgdafjab1NkoBZklqZ4tfamzWpqJW+LGo3ECm2wRByy/VyFKe8vq0qb/eb+kWfezKYB6bTdw+Nmx6n1rzjK2BbZKewZ90CawAQAwI5amWtdyoLgnHJMP4kCH7+PABGnPlzzaGO2nmUTNKL6R6oBwLBo3sLV5qrw+3xhtd7nlGTT/fLQ6Psp3Phn6XYFN3xH37xHYAACYIdojFZc4vWGPANGy3rC9ZPMtzgTO1XwGNlEYGlXXtVOYHqRtP15bYNPvrvM7MdX2vcMENgAAMFsUaDQDdWw8sEDpXHUXMIENAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKbLv1/gxyX54/uJAAAAAElFTkSuQmCC>

[image38]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEIAAAAYCAYAAABOQSt5AAAC10lEQVR4Xu2XS8hNURiGX5fcL1Hkll8JExQlI0rkVjLDhH4ZGJBLEUlKCiWUgZIYuIaBTJRMRSGUEjIRE6Qk5X75Xmutf3/n+9fea+/j8Od0nnprr/f71jprr7NuG2jRopmYaI0CRoq6W7MZ+GmNEiTrMKGqZv6u2TW8Eg2zZgl6iX5YU3MX7uUuizaLNok2RkR/i8/tKlaJXluzAjdFB60Z6Ibsn05xVrTAmv8Q9nGENSvQA4n3fAKXMNsGDIWN/GVGoTG/zzbmWTPQE+lZcUy0zJol4IybI5ovWiRarLRQNDhLLeSU6Is1I6wXnRBNtQHPY9Fta2pewA3ENBvwFA1SHruRDXCeLnVkF/Mdbmnm0Q+uvYG+fAXxGb4DiXfpA5cQ21m5wayxZoIZokeqHH58DNwA5dHXGh7W32NNxSfRflVmPmegZQUSA0HewCVNMn6yYoTJ6pnT9IJ/PiCapWKaG3Cb4S0bgOvDamsqwgw7hOLlxqM/+T6D4JK+Ko//Ho/OP+G5aLx/fitqU7HAEtE+/7xWtFPFCPvVbjxN+KeD7tSGO+BMTQ4EeQ+XONaXS1VKoNvgc+wI1jknRedVmTBetKQC0+E21bx+L0d+rAbe2pjINbcVnf8ZC08F5l6EO1ksR+E2ugDbPq7KhJvaZ1V+BzcrNKx32ngBxq6qcpv3YmxHfqwTH+GSy1TglTegXzjANnaZsm33meie6IwX4/1rMoBzcANuGQ7XX/1RxaW4TpU1D0UPrJnHaLjO7LWBCC/hcp/aALL7ieaw6IPxmMMLExnqy5ZxiPtkG7IB5qm3tDZcA3NiSzMXveun4AbEb4CydwKLfsFropWqrGHeEGtWIHxONByu6w3+uV00NwtVIswQdtTOFg0//njxq5froiPWbART4I5DXpw4PetlANw+cd8GInxDdnusAveQ2B72X1PP9G66QSBcQhOsWQAvc72t2aJFnF9WabjIaNBapQAAAABJRU5ErkJggg==>

[image39]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAXCAYAAAB50g0VAAABBElEQVR4Xu2UvQ4BQRSFr0IrQdRKhVZJpfcEnsILaBQi0fIAOpUX8ABEpdX5SSRC0AgS3JvZYnKs2B23kvmSk818c2dzkv0h8nj+ixmKCHQ5N86eU4Y9FSacp5U4nDgta33htK21Kg2KV7BK7/OZEKdG3ILyWMPmxdVRahC3oMzeUZLxc5QauBQ8oyTj5V1Ux6XgASV9+NiynFLEFIIziEvBI0oy/oEyz6lFTCU4g7gUvKIk4xcoNXApGDYvro9Sg28Fm7Du0ft8InBJ8Cp0yNw8hxvMiMzeELy4orWeUviX/RPyw91y1pxVcN1xBtZMmrPhpCwnyLsvJcecJZkZj8fzV7wAymZOJ6/kyzoAAAAASUVORK5CYII=>

[image40]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAYCAYAAABurXSEAAAB2UlEQVR4Xu2WyytFcRDHv54hhY1HpCxsrMjG0ppslOyUx0YKe1ZWdpJXshVr/4CFhY0FxULEBguE8kyEGfM7zpxxbu7j1NXtfurbvb/5zjnn15yZ371AliyZzQrpnvSpdBXIAB6Vx+oM2unD21AYU6RdG0w3OZANH1mD2Cd12eB/YASy6V4VyyV9kEpU7F9xiWBrtJAu1DpKCkkTpFlSufESQvfzovu+7duRcQ5pwTK35ucU+HZi8MV7kIo3uXWsoUyWV9KzifG8JFVtr5/fVWzVxcZULIxp0gnk2nnSAumFtKGTiEHI/ardupa0Sdr6yQDyXYzn6E+uEV7VeKvNZzxvXsPXtZk1a440idjnPPtLNhhGrM3dQOIN1jBwTmlIrNmsw55h4ZwaG7TkQRJ3rEE0Qrw7ayj4WLSb4arz29NwP9s8j2L13ctpJ62ThnzLZwaS2GMNh1ehCms4xiH+AemQ9EYaDmQIXHW76XoE56gSktNPKoIM7ajysQbpxVunJwQHoA5SYT6rzyC5p8r34JxlG4xBB/wi8LAOBO3vQWavz8QjJ64ejBO+VxWpG7+PxsjgHwX7ylPBuxf/8HCbMa3uMxIeIH9Xua10X6aCHl5uy2O1zpIl4/gCCa9/TmKx/qIAAAAASUVORK5CYII=>

[image41]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEUAAAAZCAYAAABnweOlAAACuElEQVR4Xu2Yy8tNURiHXxShXDIxMKDcRUwUScYuKeX2D8j9VpKBAUVJuQ1QRiIpZkpRMpGJck8iJu73ck2E3693r/Zab+sca23b952+9lNPnfO+6+y19l7XfUQaGjqV+XARXAyXwGWFyxPtKz2Qh/A3vAInwgmF/DwJToZT4Qy4Cp4uyjvPSRdzDH6UsBGvgxIin70cnRemk3C/HW8TbeAI+SH6u27BNTrGTnjDBjMZLu3raAc7jSOpS+kl2tgHNgHuwAU2WJF9ovW8tYkEXtjA/2aNaGOXerHe8Bcc4MXqwE3F7TbRabyScFhPg8+873XjptFQm/hHBsO9cIfUsGP5c/1w8flqma6d6VJ9fYnBUf0Vnhd9GH2khmvzAjdFRwy3yTob3IqLonXctokK8Dq3IrHKuPXkpxc7WcQ2eLEYB+ET0bJHTS4Frln87UKbyOCEhA9gLHwEd3ux0fC+6KaRxBuJP9XU0cJzxH4bTIRDnXXMtYkMXDsPwW1wTpAt4QPxN5K2tLr5d6LxUTZhYJlBNpjISnjEBjNh/Z9sMELsHqO4BemaTYAxorkPNuHB7Tq5MgNPtvdssAKsPzYtBprvrp3r4XE4zssFcNizMF/UYrhR1Grr3CXhWpQKp813G6wIb9J2zCwJO5P3x4MpX2kIy08p08op0aPz+8IvooueY4ToRXlW4ULKso+9vIPrCc8FudibSGGmDXhslbID2W4+FJ+7Ra7KO1s2rIgHphzYAbmn5JHwrA1m4DrhALzkJ+qGcza3x6+LnpZzGCJaj10jcnDt5NZ/GQ6D/cp0PXwTfYehnEIp7IFbbPAv8ITqpkVVuHb4o4PLxQXve7cxW/TG+HbM/2peiq5XTwufi56meVR3D8H3jPRAVoienLlTbISb4GYjY8zxFM1y6+BauBr2l4aGhk7lD0wFvfVu34OUAAAAAElFTkSuQmCC>

[image42]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHgAAAAYCAYAAAAxkDmIAAADn0lEQVR4Xu2ZWchNURTHFyJDprwoEg9SnogXQ4kyZCrz+IJInwzlQSEPRF6QDC9KSN7w6MFQKA8oZEwZMmWeMmZc/2/v3V133b3vOefe696T9q/+fef81z5n77vHs/dHFIlEIrVguDYiqZmqjbzxhdVCm5HUTGPt16aPfayPrD9CL4tSEH0SMWhCcTgz51gjtBnJzEXWXG2GcI3nYyPrijYrpA+F84lkJ1VdYqpEwrs6wFxnTdJmFbxjTdFmpGIesnZpU7OMTAPPEl5L1m9We+HVglQ9LpKaiZSiTl9QcaKBrKfivlYspBSFIbPGH6d8r9NtWOtYO1ldVKzeJNapXH/32usLhXDNwEdBuY6znkzeI+39VjIVmDeekFnOOtt7lLl1IVx3kP8QbUqQ4CqZkdzf3if2igrAlH9emxbXuFgaHN3o33S0avhOZosnwXdKI0cx6m2BNh1u/f0pvMPWWyk8H5ii7pF5djdrD5ne/UwmEuCdB7VpQeymvcZoWGI9CUaz9pIYnEFJLCKTf3d734N1hsy2T3KIspfTx3ZK9w2EvDDbeXlF/sKkHcXYQ29W3ibWA+UBvO+ANsmsuYgdZW1hLSUzen2kKZNkcgYl4eoEX62YcULnATiEuK3NDCxmzSOTV9oG3qZNR6gh35DxsW8tR6gQvnd+Y53VJrOB/Ok1M1h3tFlHQnWlwUw0W5sVEKpbDdKt0CZoRSZ4SQeYvmRi2LeGwHrp+8GYunz+NdZjbZI5k/alB/PFNSpujr3GwQtmj3qC9TdUznbi2qXpxTpBZsmrhCwNPFSbYAeZIEaGD9dju+qAZRWZH61ZQ/6K8K2rDvjYRknQoGPFPdKgU2HtH2bv68kAKs0TjSi/XwDSDGL1Zp1kHSuKpgfv6aBND7pMdIRM739r9ZnMF66jJ5mRiy0NRhzS3hdxB9JgvdUgw9D2pqQwFjQcpnDXqU4Xh5uBj61WI7ck46lQxq9U2imn2xg6oQYdHM+E1LaQtBm8p6PyNGMoXKdVgxfrKQSHJLKzaPDMaG2mABV3iwrHqnnlBmsmqx9VX04830mbCnzMLddmLcAo0j8Aayl6YjkwzZXrACGw5rqjVJfvc/s3T8g6cderhZcFPJ+0v9ZtUBM+UGGaQmP9YD2ihNMUAfbJaOgsyB+C/25dpsZO1SFk58Xhke+jMom1rPes12R2M1i6fJxijdNmXviljUgmRpH5lso17sw5kp0mbUQikf+Bv6YQ9CcwraHoAAAAAElFTkSuQmCC>

[image43]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADYAAAAXCAYAAABAtbxOAAACZklEQVR4Xu2XP2gUQRTGnyYEMYXEQvwXsTBFOhGMiIiNSBQrG0klllZCLBODYCXYxD+VhnRiIaKCEcViooJGULERxEKwtFAxCkZEfV/eDr75ds7di3Ke6A8+7ub35mbn7e3t3Yn85+9lqMhWLrQxffJj3w35rOnXrOZCG9MjtueHXPB8YlFwXvNN80qzjmp1eaLZrunSrNWMat4lMxqzQTMjtofbVIsEFp5cY181u9wYiw+6cR06xV7ng6ujDjvE5kc20jgSWHi4seNSXmRvxtXhueas5oQ0d6njWIfI4aQ8IBdonMCNYdFn5AB8L8sKplnUYIXYsfDouVV4T6BxQq6x3DUNf4plBQtpbEzKDYBJKftA44RcY9fIAfgbLCt4rPmiua55rXmZlrNckXIDAJc0+0DjBN/YYrEXX3YuAv+CZQW5k1Z1V7wj5QbAuJhf41xwz0vkDo6zxsDfZdkkF8XW6eaC44LkGzsj5nG3jQT3vESusZvkAPw5lk1yTGyd/eQ9jT5jE1L2gcYJucYa3RV/tiEG83kjJwu3k7xnm9ic335XRFO8wEDGrdQcIOfBfPzy8OB7iNfZrNlEDnP2kfugeUMu0DiBG1sutvAS52Y1j9wYxHekg3zktGbYjeMvkUnnQO6dxUcBd9PIIrE5650DgcYJ3BjApYKFrmreau6n5XlGxN7dg1xwTImt8754PJyW58FtHGGeaj5qLom9Nnf5BhaeXGN12aPZwrKFBBaeX2kMvwX/JIGFZ6GN4e/IUZYtJrDwzGmWaZZyoYJVLFpIl9ie73HBc6TIbi60Mfh/Fvf9b/AdWx+iObpes2kAAAAASUVORK5CYII=>

[image44]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHwAAAAWCAYAAAACcfiCAAACt0lEQVR4Xu2auYsUQRjFP/HORFbBSGG98cAj8F9wYQVRPNBEwfs2MxHEwEBRWBTBMzAxFTQxEyMVRfAIxND7VlYUPL9HVTm1b2apqq6ewNn6wWPne19Nz9CP7aquHpHBWaC6zWahMzmoOiUl8CHFIUkLvFu1XLVStUq1WrVGtTZSPRJmLtVdVBcySA18vuqP1RzVLKvZVvDmqRarVqiOqT5574FC8JiTVBcyQOB32AxwVUwoD7kR4LKY9y3kBsGBn6W6kAECv8tmBL/FBIP/4hQ2qt6wSXDg56kuZIDA77EZwUhpXKKHUy/ERzYIDvwC1YUMEPh9NiNZJyacX9zIpATeJr6qPqjeqT6rRgxsR/FETEBXuJFBSuCYIk6ohnGj0D7cpR2r8zrgwC9SDaaLGTfZ1m5NseXfiELbmCCN0OuAj8OBjxEzZpPn9VovdT1RqMgRMSc8tCCLIRT4N2kec6mFlwL2DWL1X+H+E0OqwhfVKzYrwJ/PgaP/soXXTx74qRrPZguWJahgwTxaBxy4v2hbIqbPczW8A+QBPlahJuo8sXwsP/Auae5Ptd5o1TTVBjGbO7lXrI7hgeqH6qbqqZgTcl11y74e1xgaxWvVRDYz4IB4pw399fa1W8C59zy2f8FxMbdsQx5/Zw2Lnbdefdh7HcM11VI2A0xhg+DAz1E9VszcjHH4fPDc1pvdIDEbQjHzd0czUwbeuuAkbfdqPOCIZZuY5+mpcKAM989QHQsfpyDVTwoeiT5iMwI8Kr3BJsHf6TTVMWDX0B0n9alexzJDmk9uDKNU39kM4PbdoUnUY/g79VEdC27TsF4pWHD55vvZGBDIezFzP+67X6ieWWEuhYe+C5gVgsccpbpQEZzYrWwGWCRmzt+l2qPap9rfQnttf7cdu1O1Q8zPoULgJ1Q++FlVIYG/fnq/+QzETLkAAAAASUVORK5CYII=>

[image45]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB4AAAAUCAYAAACaq43EAAABQklEQVR4XuWUu0oEQRBFGwPxG0xMfPyHmZGxubGYyoIGmhsZGommfokriCDoPkIzBQ1EQbTK6a4pz1Sv4oYeKOh7b00VzDST0n9mEXoFelo433iHvoaeFs43PqDvoAvzqeldyPpE6rmNq3C+wWAIXdhNTe9W1m9Z/0S1h8EI2jND4xdwvsFg0uK/wPkGg0mL16SOpZay3ndZDc43GIyhC9q3nc+XUo9ST21chfMNBtFi7eGlU28HXgTnGwy4+DB1e5azNws/gs8aDLhYc/acBh51oeZ3Al4uze8D7yXwImp+J4gW9wJvz519kcj7ggEv0YPUwOnyx5pzXl9q3WkP5xsM/JKCvmrtu5A6y2cPtaeaMbiFJtr/Gng1qhmDG2ii/QdOr6b2mbHzC5xvMLiCLmxKnaem/0hqw2X63fVvFvFt/idw6mk8R8u8kAAAAABJRU5ErkJggg==>

[image46]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAWCAYAAABzCZQcAAAB30lEQVR4Xu2YTSsFURjHH5FYSAgrVkpeF2xkoWvjtWwQQjcLkfe3sLWTteJjKPkIPgOx9ZIokY3k5Xk65zbTv5k7Z6Zr5hS/+nXv/T/nXPNMzsycS/TP32KIHWZH2TF2XDthaBA9GNjAt3aKbWQbtPK+iW1h29gUu8Oeu+aI7eTPPLuLoQ2Uk9NAGJpJzXnCgqaGvcPQJvZINfCCBQP8TpbkRRjaxjOpA93HQgApdgGyTvYdMmvJ/JtXYSEkH2TpWvailaKtb0TmF2PooovtZfvZAf0qV/k696A4OSF10FdYMKSE/E9aLTkn1ctXZ2j8fJI6iEksGNBN/k1/sQX6vbxeumqJk0/O2c8cpCkz5N10NXw+ZLcgy8Yiqe+dxUIuGSH1RwqxEECavJtGZEwFhlmYIzVnGgu55IKtx9CADjJv2iqOSZ3ZKJRRcEPLFDwmVgbZUwxDErQs5D7+iGFSyMXmHkMDZIPiRprehMyN1GVHlzh5pG4pYTkitflws82+QWYlUdaY3+1JkDzs7S5WHthKDLNQSmqONCZXeS/62GsMbeGM1I5I9sXSiKzpW/ZGK5/lopN5SkPlhwY/DuiXHyaiIr9uLLGr7Bq7zm6AkklNxqzo8fKEhFtKL9IYJMkPrsBziUv039oAAAAASUVORK5CYII=>

[image47]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGgAAAAYCAYAAAAWPrhgAAADm0lEQVR4Xu2ZWahNURjHP1MhGUOZug8eeEFmefAiEXkwxquIvJC8ECWPSplKIQ9KlGQqGZ4MGUqZoojkRVLIPIXvf9dZ9Z3//dY+e59znZvT+dU/d/3Xt9Zea3/7rL3WJtKkSZPGY1lJ07iiQYnzhTKZykYH8UM1WjWEKxoUzBX6wxXMSglB57miznxjo8RBCeN7qRpBdVmMVN2S0PYy1RVhoYSxoZ9zqs7l1ZlsUX1QfVGtoLpIxQSB15Iz8B/iJei3apYpY4yzTTnFDCmfzzgq5+WYlN/YRxL66Wm8FIi9ZMoPVddNOZJrXBMlBC7mioLgqV2lWmOUF07Qdmk7+HmO54EYvjaW0JvkVQL92OuNLZVfGc+jt/jjhNfX8XKBQEyiGs6q3qnmqkaphqkGqfrZoApwgjAePIUM/OFsGnBdxOBfy8WSX4SfEn7FkSkS+rhjPI+74l8LHpZs9nJxTQoEG9Cmhc0q8BLkvTvg72bTsFX8eRwW3y9CTPJgriAQ413L87mcpL+EYEwwLydVQ9msEi9BZ8gD8LM2NKfEn/Q+8f28YMlE++Vc4eAlAng+lzPxOsjiIxs1YBOEnRLGgQeAgf+UTcMV8eewS4JfzQO1U8IO7pOEJbwSqfvo+VxO8lj1TEKDrlSXYr+EDUaW8uL9gvBrYOBfZdNwVPxJ75Vic/OYKaGPjVxBeIkAns9llxeq46ouEhpgkpXopDqiml9BefESdIE8AP8Am4bUO+iQ+H5RvJvMpGI8n8ttwBnoiSl7naR4y0YNeAlK7eKWsmmYLiGm1l1cHwnxOMhb4v3BA5oCS793LXhYqdhL8l71lbz1Ehrl+eTyQDWAzSrhBMVDoWWy401SjScPMQvIw/uDHygcQgeSF9kjoR98BbDEBFk2U3mJtI0B8CY4nksqywD+fTYT4JxQ5LyTghMUd5XdjYcx8xnEu2FYGn+ZMp52xLQYLx46uW2kl7StQ0Lh2Z3uupKHlcgCb7Up7yh5jOe1HiRRgXeOB84fbsMEuGnYWWF5wY3tVl6dC04QiC/l0xIOwjfKq1vB9hli7qk+q05I6AN9MbdVb9g0jJHQFgdWrDT4e21ZROC5hA+flh4S4nENjAXz85bFIve5ZhapNqm2SfhUA+XFS1A9sL+0jqCuCaqFjkpQ1pmqHjQTlAE+enrLTj35bxL0XcLWNs+n/PaiyP8vtTeYa9zK/xdsKGkOVzQocb5QkyZN2o2/Y1EPgRS4u/gAAAAASUVORK5CYII=>

[image48]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIUAAAAYCAYAAADUIj6hAAADqUlEQVR4Xu2ZW6hNQRjHP0QcIpIo1+KFJyleeCLkkjzIi5JIyQMePIlcSyQhl3K/JJHrAymeiHKJKIVSiNzvl9x9//2taX/r27P2Xpu911nnNL/6d878v5mzZs2ZPfPNbKJAIBAIBAKB2rOd9YH1R+lFrAbRJxWDxsfDuWAHSd8esXqbWDXMZL2l+Pu+itUgeqhi0LJ4OLesY31nvWYNNzEv7gV9LGfdsGaO+M0arcp4j7Gq/C+UG48xrGesVjaQY96zVqryV9ZqVS6hBckA3LMB5jZrojVzxAoq/edN8HjVkjQpjrCWWDPnjKTSd+ni8WLMJakwVXktST6BDcrLI+j3HWuS+L2smZJBJO23Gh/bbF/jNQWwZfgmALxp1nQ8p3ijwawnqpxn0O9z1iTxN1ozJYdJ2neLyl1Z34rhJgfe5Yc1Sfxb1nTopXJL9PulYjjXoK+nrEnin7FmSvR4zIp+/1wMNznQf6xyFvjILbwgeJNkxRgYld2g1JKlrP0J2sfaw9rN2klymliARmXAFod+HrMBEv++NVOCtr9Y11iTozKE3Kve2HHR2ksyRrtIxggnx/6FVuVB399Yk8r8n10+8VN5ByJvnvJ8bGA9Jqm7zcSyAs8+YU0S/4I1U+DyCah95LnV4rirlMAw1iGSulhxN7Eukkywdqpe1qA/76xJ4iNvLOEl+WdL4iwyYK9ab80MQR/PWpPExyepWo6StO1g/LTjsZAksdMgJ0nTtl7g2b6cCL53NU16WVxwwO9nAwbU6WjNBGaz1lSh+dKsLHh+0ulDn6bSkjQep0n86TZgwDKt7wMcaJtmtbBjUElY2SqR9E7w7AmrcPmCwFUbYAaQxHC7l0QD+R+WJZgQtg9DPV53qvwPBWint1KHy1/s37UgjnGxwE+z/9cDd3jQuLup1sYvLPsITLGBCDcInW0gYhX5BzBL3CVMW+V9ZF1XZeDeBR+EJCaR1NlsAxHubwyxgQg3cSxtyO9nCZ6vV5UrZE4kByMDSx2E45ZOOHqSrBC4q0AiiboPVNyBfGKtNRuBUSQvfZKk35fj4QKLSFaVGTbAzCGZSGiL8cB3PUgONbgmfkoyHkjafOd+nJZ8ezdODjbPyJo+JGN0nuT7obrdQ+EhnayZY8aRnBDqBSZLUj5Rskw3R3Bca+wlsVruWqPGYDx0PoEtDR7ymWYPbsGwxEK+ZTSPjGAttmaNwFfn2H4xAfATedYXkvuNQI7pYY1AIBAIBAL/xV/ggxya6v3r2AAAAABJRU5ErkJggg==>