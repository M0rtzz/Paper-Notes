---
title: >-
  [论文解读] MoMaGen: Generating Demonstrations under Soft and Hard Constraints for Multi-Step Bimanual Mobile Manipulation
description: >-
  [ICLR 2026][移动操作] MoMaGen 将双臂移动操作的演示数据生成建模为约束优化问题，通过硬约束（可达性、无碰撞、可见性）和软约束（导航中物体可见性、收回紧凑姿态）的协同，从单个人类遥操作演示自动生成大规模多样化数据集，训练出的视觉运动策略仅用 40 个真实演示微调即可部署到实体机器人。
tags:
  - ICLR 2026
  - 移动操作
  - 双臂协调
  - 约束优化
  - 自动数据生成
  - 模仿学习
---

# MoMaGen: Generating Demonstrations under Soft and Hard Constraints for Multi-Step Bimanual Mobile Manipulation

**会议**: ICLR 2026  
**arXiv**: [2510.18316](https://arxiv.org/abs/2510.18316)  
**代码**: [项目页面](https://momagen.github.io)  
**领域**: 机器人学习/数据生成  
**关键词**: 移动操作, 双臂协调, 约束优化, 自动数据生成, 模仿学习

## 一句话总结

MoMaGen 将双臂移动操作的演示数据生成建模为约束优化问题，通过硬约束（可达性、无碰撞、可见性）和软约束（导航中物体可见性、收回紧凑姿态）的协同，从单个人类遥操作演示自动生成大规模多样化数据集，训练出的视觉运动策略仅用 40 个真实演示微调即可部署到实体机器人。

## 研究背景与动机

- **领域现状**: 从大规模人类遥操作数据学习（模仿学习）已被证明是训练机器人操作技能的有效范式。X-Gen 系列方法（MimicGen、SkillMimicGen、DexMimicGen 等）通过以少量人类演示为种子，在仿真中自动生成 25x~350x 的数据变体，大幅降低了数据采集成本。然而，这些方法主要针对固定基座的桌面操作任务。
- **现有痛点**: 双臂移动操作面临两个核心新挑战：(1) **可达性问题**——移动底座意味着在新场景中需要重新规划底座位置，直接重放源演示的导航段在物体位置变化后常导致手臂无法到达目标；(2) **可见性问题**——移动底座自带移动相机，naive 数据增强会使任务相关物体移出相机视野，导致视觉运动策略无法做出正确决策。
- **核心矛盾**: 人类遥操作双臂移动操作极其困难（同时控制底座+两只高自由度手臂），数据收集代价高昂；但现有自动数据生成方法无法处理底座运动和相机可见性，仅适用于简单桌面任务。
- **本文目标**: 为双臂移动操作设计一个通用的自动数据生成框架，能够在场景激进随机化（物体位置、干扰物、障碍物）下仍然生成高质量、多样化的演示数据。
- **切入角度**: 将数据生成统一建模为带硬约束和软约束的优化问题。这一抽象不仅适用于移动操作新场景，还能将先前的 X-Gen 系列方法纳入同一框架——区别仅在于约束的选择不同。
- **核心 idea**: 引入可达性（硬约束）、操作时物体可见性（硬约束）、导航时物体可见性（软约束）和收回紧凑姿态（软约束）四类约束，通过采样-验证循环自动发现满足所有约束的底座位姿和全身运动轨迹。

## 方法详解

### 整体框架

MoMaGen 的核心流程：给定一个人类源演示和经过随机化的新场景配置，系统将演示分解为若干子任务（导航段 + 操作段），逐子任务生成新轨迹。对每个子任务：(1) 将源演示中末端执行器与目标物体间的相对位姿变换到新物体位置；(2) 采样满足可达性和可见性约束的底座位姿；(3) 规划底座/躯干导航轨迹（带软可见性约束）；(4) 规划手臂运动到预抓取位姿；(5) 在任务空间中回放接触丰富操作段；(6) 收回到紧凑姿态。失败时回退重采样。

### 关键设计

1. **约束优化形式化（Constrained Optimization Formulation）**:

    - 做什么: 将自动数据生成统一为约束优化问题
    - 核心思路: 优化变量为整条轨迹 $\{a_t\}_{t \in [T]}$，约束包括：系统动力学 $s_{t+1} = f(s_t, a_t)$、运动学可行性 $\mathcal{G}_{\mathrm{kin}}(s_t, a_t) \leq 0$、无碰撞 $\mathcal{G}_{\mathrm{coll}}(s_t, a_t) \geq 0$、可见性 $\mathcal{G}_{\mathrm{vis}}(s_t, a_t, o_{i(t)}) \leq 0$、接触段相对位姿保持 $\mathbf{T}_W^{E_k} = \mathbf{T}_W^{o_i} (\mathbf{T}_W^{o_{i,\text{src}}})^{-1} \mathbf{T}_W^{E_k}$、以及任务成功。目标函数 $\mathcal{L}(\cdot)$ 编码用户指定的软约束（例如轨迹长度、平滑性等）。
    - 设计动机: 先前 X-Gen 系列方法实际上也在做约束优化，只是各自选择了不同（且不充分的）约束子集。统一框架能清晰对比不同方法的差异，并为扩展新约束提供原则性基础。

2. **可达性约束与底座位姿采样（Reachability & Base Pose Sampling）**:

    - 做什么: 在新场景中采样新的底座位姿，确保手臂能到达所有目标位置
    - 核心思路: 在目标物体周围的环形区域内随机采样底座位姿 $\mathbf{T}^{\mathrm{base}}$，通过逆运动学（IK）验证所有需要的末端执行器轨迹 $\{\mathbf{T}_W^{E_k}\}$ 是否在手臂工作空间内。同时通过碰撞检测排除与家具/障碍物冲突的位姿。使用 cuRobo（GPU 加速运动生成器）进行高效的 IK 求解和无碰撞路径规划。
    - 设计动机: MimicGen/DexMimicGen 直接重放源演示的底座轨迹，在物体随机化后底座位置不再合适。特别是在 D1/D2 随机化下（物体可以出现在家具任意位置+添加障碍物），必须重新规划底座放置位置。

3. **双层可见性约束（Dual Visibility Constraints）**:

    - 做什么: 确保任务相关物体始终在相机视野中
    - 核心思路: 硬约束层——在操作阶段开始前，验证采样的底座位姿和头部相机朝向能看到目标物体（无遮挡），不满足则重新采样；软约束层——在导航阶段，给运动规划器添加一个代价项，偏好让头部相机朝向目标物体的方向，但不强制要求。数学上，采样 $\mathbf{T}^{\mathrm{cam}}$ 时需满足 $\mathcal{G}_{\mathrm{vis}}(s_t, a_t, o_{i(t)}) \leq 0$。
    - 设计动机: 视觉运动策略依赖 RGB 图像做决策，如果训练数据中目标物体频繁不可见，策略将无法学会可靠的视觉伺服行为。实验表明去除可见性约束后策略性能急剧下降（Tidy Table 从 0.40 降到 0.05）。

4. **收回策略（Retraction as Soft Constraint）**:

    - 做什么: 操作完成后将手臂和躯干收回紧凑配置
    - 核心思路: 操作子任务结束后，机器人将手臂和躯干收回到预定义的"紧凑"关节角度，减小机器人占地面积
    - 设计动机: 减小后续导航时与环境碰撞的概率，特别是在有障碍物的 D2 场景中。

### 损失函数 / 训练策略

数据生成阶段使用约束采样-验证循环（非梯度优化），不涉及传统损失函数。策略训练阶段使用标准行为克隆（BC）：$\arg\min_\theta \mathbb{E}_{(s,a) \sim \mathcal{D}} [-\log \pi_\theta(a|s)]$。

实验中使用两种策略学习方法：
- **WB-VIMA**：从头训练，输入本体感知+头部和腕部三路 RGB 图像（融合为自中心点云），输出目标关节角。
- **$\pi_0$**：在预训练基础上用 LoRA (rank=32) 微调，输入 RGB 图像+本体感知，输出目标关节角。

## 实验关键数据

### 主实验

四个家庭任务：Pick Cup（导航+抓杯子）、Tidy Table（远距离移动杯子到水槽）、Put Dishes Away（双臂独立堆盘子）、Clean Frying Pan（双臂协调刷锅）。三级随机化：D0（±15cm/±15°）、D1（家具上任意位置）、D2（D1+额外干扰物和地面障碍物）。

**数据生成成功率对比**:

| 方法 | Pick Cup | Tidy Table | Put Dishes Away | Clean Frying Pan |
|------|----------|------------|-----------------|------------------|
| MoMaGen (D0) | 0.86 | 0.80 | 0.38 | 0.51 |
| SkillMimicGen (D0) | 1.00 | 0.69 | 0.38 | 0.40 |
| DexMimicGen (D0) | 1.00 | 0.72 | 0.38 | 0.35 |
| MoMaGen (D1) | 0.60 | 0.64 | 0.34 | 0.20 |
| MoMaGen (D2) | 0.47 | 0.22 | 0.07 | 0.16 |

注意：基线方法在 D1/D2 下成功率为零（底座位姿重放后物体超出可达范围），因此省略。

**任务相关物体可见性对比**:

| 方法 | Pick Cup | Tidy Table | Put Dishes Away | Clean Frying Pan |
|------|----------|------------|-----------------|------------------|
| MoMaGen (D0) | 1.00 | 0.86 | 0.79 | 0.69 |
| SkillMimicGen (D0) | 1.00 | 0.40 | 0.71 | 0.65 |
| DexMimicGen (D0) | 1.00 | 0.39 | 0.71 | 0.67 |
| MoMaGen w/o vis. (D0) | 0.90 | 0.46 | 0.40 | 0.35 |
| MoMaGen (D1) | 0.93 | 0.89 | 0.78 | 0.80 |
| MoMaGen (D2) | 0.94 | 0.79 | 0.75 | 0.81 |

### 消融实验

**可见性约束对策略性能的影响（WB-VIMA, 1000 demos, D0）**:

| 方法 | Pick Cup 成功率 | Tidy Table 成功率 |
|------|----------------|-------------------|
| MoMaGen（完整） | 0.75 | 0.40 |
| w/o 软可见性 | ~0.55 | ~0.05 |
| w/o 硬可见性 | ~0.50 | ~0.05 |
| w/o 所有可见性 | ~0.45 | ~0.05 |

**$\pi_0$ 数据规模效应（Pick Cup D1）**:

| 演示数量 | 500 | 1000 | 2000 |
|---------|-----|------|------|
| 成功率趋势 | 较低 | 中等 | 更高 |

D1 随机化下数据量增加带来显著性能提升（覆盖更大的状态-动作空间）。

**Sim-to-Real 结果（Pick Cup D0, 40 real demos fine-tune）**:

| 方法 | 有预训练 | 无预训练 |
|------|---------|---------|
| WB-VIMA | 10% | 0% |
| $\pi_0$ | 60% | 0% |

### 关键发现

- MoMaGen 平均生成成功率 63%（D0），且是唯一能处理 D1/D2 随机化的方法——基线在 D1/D2 下成功率为零
- 可见性约束显著影响策略质量：Tidy Table 任务中，去除所有可见性约束后策略成功率从 0.40 降到 0.05，降幅 87.5%
- 数据多样性是关键：MoMaGen 的 D1 数据覆盖整个台面而非小角落，PCA 投影显示手臂/躯干关节角分布远比基线广
- $\pi_0$ 即使有强预训练权重（10k+ 小时机器人数据），仍然受益于仿真预训练——0% → 60% 成功率

## 亮点与洞察

- **统一框架视角极具洞察力**: 将所有 X-Gen 系列方法统一为"约束优化问题的不同约束选择"，这一抽象不仅清晰对比了方法差异（MimicGen 只有成功约束、SkillMimicGen 加了运动学和碰撞约束等），更为未来扩展提供了原则性基础。后续工作只需定义新的硬/软约束即可。
- **双层可见性设计体现了对视觉策略训练的深入理解**: 区分操作阶段（必须看到→硬约束）和导航阶段（最好看到→软约束），这种分层处理在工程上既保证了数据质量又不过度约束导致生成率过低。可见性对策略性能的影响之大（8 倍差距）说明这不是 nice-to-have 而是 must-have。
- **从单个演示到真实部署的完整链路**: 1 个人类演示 → 1000 个仿真变体 → 策略训练 → 40 个真实演示微调 → $\pi_0$ 60% 真实成功率，展示了 X-Gen 范式在复杂双臂移动操作场景下的完整实用价值。

## 局限与展望

- **依赖完整场景知识**: 当前假设在数据生成时有全场景信息（物体精确位姿、几何），这在仿真中自然满足但在真实世界需要额外感知系统（如 SAM2 估计物体相对位姿）
- **仅支持交替式导航-操作**: 当前框架假设导航和操作交替进行，不支持全身操作（如推开门时同时移动底座和手臂）
- **GPU 资源要求高**: 依赖 cuRobo 等 GPU 加速运动生成器，计算密集；仿真执行占总时间最大份额（底座运动规划 18 秒 vs 仿真执行 100 秒）
- **底座采样效率可改进**: 当前在环形区域内均匀随机采样，可行底座位姿稀疏时搜索变慢，可引入更智能的采样策略（偏向自由空间大的区域）
- **D2 下生成成功率显著降低**: 地面障碍物使导航空间紧张，Put Dishes Away 在 D2 下仅 7% 成功率，复杂场景仍有较大提升空间

## 相关工作与启发

- **vs MimicGen**: MimicGen 是 X-Gen 系列的开创性工作，但仅支持单臂固定基座任务，直接重放底座轨迹。MoMaGen 通过引入底座位姿采样+可达性约束突破了这一根本限制。
- **vs SkillMimicGen**: SkillMimicGen 加入了运动学和碰撞约束、支持障碍物场景，但仍然局限于固定底座单臂。MoMaGen 扩展到移动底座+双臂+主动相机的全身控制。
- **vs DexMimicGen**: DexMimicGen 支持双臂灵巧操作但无移动底座、不考虑可见性。MoMaGen 在此基础上增加了移动底座、可见性约束和障碍物处理。
- **vs DemoGen/PhysicsGen**: DemoGen 和 PhysicsGen 分别引入了碰撞自由和系统动力学约束，但均不支持移动底座和主动感知。MoMaGen 是唯一同时满足所有六类约束的方法。

## 评分

- 新颖性: ⭐⭐⭐⭐ 将数据生成统一为约束优化的框架视角新颖，可达性+双层可见性约束设计原创性强，但底层技术（IK、运动规划、cuRobo）均为已有工具
- 实验充分度: ⭐⭐⭐⭐⭐ 四个任务×三级随机化×多种基线和消融+数据多样性分析+策略训练+真实机器人部署，实验链覆盖完整
- 写作质量: ⭐⭐⭐⭐ 统一框架的形式化清晰，实验设计逻辑严密，图表信息丰富；约束优化公式可读性好
- 价值: ⭐⭐⭐⭐ 双臂移动操作的自动数据生成是社区迫切需求，框架通用性好可扩展到新任务和新机器人，但依赖完整场景信息限制了直接的真实世界应用

<!-- RELATED:START -->

## 相关论文

- [Learning Dynamics under Environmental Constraints via Measurement-Induced Bundle Structures](../../ICML2025/reinforcement_learning/learning_dynamics_under_environmental_constraints_via_measurement-induced_bundle.md)
- [Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)
- [ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation](reform_reflected_flows_for_on-support_offline_rl_via_noise_manipulation.md)
- [AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints](autotool_automatic_scaling_of_tool-use_capabilities_in_rl_via_decoupled_entropy_.md)
- [When Sensors Fail: Temporal Sequence Models for Robust PPO under Sensor Drift](when_sensors_fail_temporal_sequence_models_for_robust_ppo_under_sensor_drift.md)

<!-- RELATED:END -->
