---
title: >-
  [论文解读] DexFlyWheel: A Scalable Self-Improving Data Generation Framework for Dexterous Manipulation
description: >-
  [NeurIPS 2025][机器人][灵巧操作] 提出 DexFlyWheel，一个从单个人类示教出发、通过 IL + 残差 RL + 数据增强组成的自改进循环逐步扩展数据多样性的灵巧操作数据生成框架，在 4 个任务上生成 2000+ 示教，策略平均成功率 81.9%，真实世界迁移成功率 78.3%。
tags:
  - NeurIPS 2025
  - 机器人
  - 灵巧操作
  - 数据飞轮
  - 模仿学习
  - 残差强化学习
  - 仿真到现实
---

# DexFlyWheel: A Scalable Self-Improving Data Generation Framework for Dexterous Manipulation

**会议**: NeurIPS 2025  
**arXiv**: [2509.23829](https://arxiv.org/abs/2509.23829)  
**代码**: [项目页面](https://DexFlyWheel.github.io)  
**领域**: 强化学习  
**关键词**: 灵巧操作, 数据飞轮, 模仿学习, 残差强化学习, 仿真到现实

## 一句话总结

提出 DexFlyWheel，一个从单个人类示教出发、通过 IL + 残差 RL + 数据增强组成的自改进循环逐步扩展数据多样性的灵巧操作数据生成框架，在 4 个任务上生成 2000+ 示教，策略平均成功率 81.9%，真实世界迁移成功率 78.3%。

## 研究背景与动机

灵巧操作（Dexterous Manipulation）是机器人领域的核心挑战，多指手的高自由度和丰富接触交互要求更大、更多样、更高质量的数据集。当前数据收集方法存在以下瓶颈：

**人类遥操作**：需要大量人力，通常局限于实验室环境，可扩展性差

**运动规划方法**：对夹爪有效，但难以处理多指手的高维动作空间和复杂接触动力学

**LLM 驱动方法**：能生成高层指令，但无法提供手指级精细控制

**纯 RL 方法**：探索困难、依赖奖励工程，且产生非人类行为导致 Sim-to-Real 困难

**回放编辑方法**（如 DexMimicGen）：仅对已有示教做空间变换，无法探索新操作策略，面对物体几何变化大时失效

核心洞察：操作不同物体通常只需对操作轨迹做**微小调整**——人类示教应被视为**行为先验**而非仅仅是回放数据。

## 方法详解

### 整体框架

DexFlyWheel 分为两个阶段：

1. **Warm-up 阶段**：从单个 VR 遥操作示教出发，通过数据增强生成初始数据集 $\mathcal{D}_1$
2. **自改进数据飞轮阶段**：多次迭代 $i = \{1, 2, ..., n-1\}$，每次执行闭环流程

每次飞轮迭代包含四个步骤：
1. 训练基础策略 $\pi_{\text{base}}^i$（模仿学习）
2. 训练残差策略 $\pi_{\text{res}}^i$（残差 RL）
3. 组合策略 rollout 收集新轨迹 $\mathcal{D}_O^i$
4. 数据增强扩展多样性 → $\mathcal{D}_{i+1}$

### 关键设计

#### 1. VR 遥操作 + 数据增强（Warm-up）

使用 Apple Vision Pro 追踪人手/手腕/头部姿态，在仿真中采集**仅一条**种子示教 $d_{\text{seed}}$。

数据增强模块 $\mathcal{A}_{\text{EP}}$ 基于 MimicGen 扩展，支持多维增强：
- **环境多样性**：光照条件、桌面外观变化
- **空间多样性**：物体位姿变化
- 通过轨迹编辑 + 仿真域随机化实现

#### 2. 基础策略训练（模仿学习）

采用 Diffusion Policy 作为基础策略，输入状态 $s_t = \{s_t^{\text{vis}}, s_t^{\text{obj}}, s_t^{\text{prop}}\}$：
- $s_t^{\text{vis}}$：相机视觉输入
- $s_t^{\text{obj}}$：物体状态（6D 位姿 + 速度）
- $s_t^{\text{prop}}$：机器人本体感觉（关节角度/速度 + 末端执行器位姿）

输出动作序列 $(a_t, a_{t+1}, ..., a_{t+H})$，其中 $H$ 为预测 horizon。

#### 3. 残差强化学习

这是飞轮机制的核心创新。冻结基础策略 $\pi_{\text{base}}$，训练残差策略 $\pi_{\text{res}}$ 生成修正动作：

$$\tilde{a}_t = a_t + \alpha \cdot \triangle a_t$$

组合策略 $\pi_{\text{combined}} = \pi_{\text{base}} + \alpha \cdot \pi_{\text{res}}$。

训练中使用渐进调度：

$$\pi_{\text{combined}}(s) = \begin{cases} \pi_{\text{base}}(s) + \alpha \cdot \pi_{\text{res}}(s) & \text{概率 } \epsilon \\ \pi_{\text{base}}(s) & \text{概率 } 1 - \epsilon \end{cases}$$

$\epsilon$ 从 0 线性增加到 1，逐步将控制权从基础策略转移到残差策略。残差策略仅接收 $s_t^{\text{obj}}$ 和 $s_t^{\text{prop}}$，专注于物体适应而非从头学习操作。

#### 4. Rollout + 增强循环

组合策略在随机物体配置下执行 rollout，按任务成功过滤高质量轨迹，再通过 $\mathcal{A}_{\text{EP}}$ 增强到更多环境/空间配置，生成下一轮训练数据。

### 损失函数 / 训练策略

- **基础策略**：Diffusion Policy 的标准去噪损失
- **残差策略**：标准 RL 损失（PPO），奖励为任务成功指标
- **迭代设置**：$i = \{1, 2, 3\}$，分别生成 20、100、500 条轨迹
- **仿真平台**：OmniGibson（逼真渲染），80 种物体，12 种环境

## 实验关键数据

### 主实验：数据飞轮效果

| 任务 | i=1 成功率 | i=2 成功率 | i=3 成功率 | 场景数 |
|---|---|---|---|---|
| Grasp | 15.0% | 58.0% | **90.0%** | 3960 |
| Pour | 36.1% | 55.6% | **85.8%** | 1440 |
| Lift | 13.9% | 44.4% | **79.4%** | 1560 |
| Handover | 0.8% | 17.5% | **72.5%** | 1200 |
| **平均** | 16.5% | 43.9% | **81.9%** | 2040 |

从 i=1 到 i=3：物体种类 20×，场景数 214.7×，成功率 +396.4%。

### 与基线对比

| 方法 | Grasp | Pour | Lift | Handover | 平均 |
|---|---|---|---|---|---|
| Human Demo (Default, 20条) | 6.1% | 16.7% | 13.9% | 0.8% | 9.4% |
| Human Demo (Enhanced) | 15.0% | 36.1% | 2.5% | 0.0% | 13.4% |
| DexMimicGen (Default, 1条种子) | 30.3% | 38.9% | 28.2% | 28.3% | 31.4% |
| DexMimicGen (Enhanced, 10条种子) | 50.3% | 44.4% | 43.7% | 42.5% | 45.2% |
| **DexFlyWheel (1条种子)** | **90.0%** | **85.8%** | **79.4%** | **72.5%** | **81.9%** |

仅需 1 条种子示教，DexFlyWheel 以 81.9% 大幅超越拥有 10× 数据优势的 DexMimicGen（45.2%）。

### 数据生成效率

| 方法 | 单轨迹时间 | 500 条成功轨迹总时间 | 生成成功率 |
|---|---|---|---|
| 人类遥操作 | 60s | 12.5h | - |
| DexMimicGen | 15s | 4.4h | 63.0% |
| **DexFlyWheel** | **15s** | **2.4h** | **89.8%** |

### 消融实验

| 变体 | 影响 |
|---|---|
| w/o 残差策略 | **最大性能下降**，物体泛化能力骤降（20→8.25 种物体） |
| w/o 数据增强 $\mathcal{A}_{\text{EP}}$ | 环境和空间多样性受限 |
| w/o 残差 + w/o 增强 | 仅基础策略，性能最差 |

### 真实世界部署

通过 Digital Twin 迁移到双臂 Real-Man 机器人：
- **Dual-arm Lift**: 78.3% 成功率
- **Handover**: 63.3% 成功率

### 关键发现

1. **飞轮效应显著**：每轮迭代数据多样性和策略性能同步提升，形成良性循环
2. **残差 RL 是核心**：平均提升物体泛化 32.1%（$\pi_{\text{base}} \rightarrow \pi_{\text{combined}}$）
3. **1 条示教足矣**：无需大量人类数据，单条示教即可启动飞轮
4. **Sim-to-Real 成功**：通过 Digital Twin 实现了可靠的仿真-真实迁移

## 亮点与洞察

1. **将 LLM 自改进思想迁移到机器人**：灵感来自 LLM 的迭代自改进（RLHF 循环），但应用于机器人数据生成，跨领域思想迁移巧妙
2. **残差 RL 的精妙设计**：不从头学习操作技能，而是在模仿学习基础上做微调，符合"操作不同物体只需微小调整"的物理直觉
3. **渐进调度的稳定性**：$\epsilon$ 从 0 到 1 的线性调度避免了残差策略早期探索破坏基础策略的问题
4. **极低的人力需求**：每个任务仅 1 条示教 → 2000+ 多样化示教，数据效率极高

## 局限与展望

1. **手动设计奖励函数**：残差 RL 依赖人工设计的任务奖励，未来可探索 LLM 驱动的奖励生成
2. **缺乏触觉反馈**：当前策略和仿真不包含触觉信号，限制了接触丰富任务的性能
3. **迭代次数有限**：仅展示 3 轮迭代，更多轮次是否会收敛或退化未知
4. **OmniGibson 仿真与真实差距**：尽管使用了逼真渲染，Sim-to-Real gap 仍可能限制更复杂任务的迁移

## 相关工作与启发

- **与 DexMimicGen 的核心差异**：DexMimicGen 仅做轨迹回放编辑（空间变换），无法探索新策略；DexFlyWheel 通过残差 RL 实现策略级适应
- **与纯 RL 方法的互补**：不是用 RL 从头学习（探索困难），而是用 IL 提供先验 + RL 微调，两者互补
- **数据飞轮的通用性**：IL → 残差 RL → Rollout → 增强 的闭环范式可推广到其他机器人任务

## 评分

- 新颖性: ⭐⭐⭐⭐ (数据飞轮概念新颖，残差 RL + IL 组合设计精巧)
- 实验充分度: ⭐⭐⭐⭐⭐ (4 任务 × 多基线 × 消融 × 真实世界部署，极其全面)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，图表丰富)
- 价值: ⭐⭐⭐⭐⭐ (解决了灵巧操作数据稀缺的核心瓶颈，有实际落地价值)

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] A Snapshot of Influence: A Local Data Attribution Framework for Online Reinforcement Learning](a_snapshot_of_influence_a_local_data_attribution_framework_f.md)
- [\[CVPR 2026\] IGen: Scalable Data Generation for Robot Learning from Open-World Images](../../CVPR2026/robotics/igen_scalable_data_generation_for_robot_learning_from_open-world_images.md)
- [\[NeurIPS 2025\] HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data](himacon_discovering_hierarchical_manipulation_concepts_from_unlabeled_multi-moda.md)
- [\[NeurIPS 2025\] SutureBot: A Precision Framework & Benchmark for Autonomous End-to-End Suturing](suturebot_a_precision_framework_benchmark_for_autonomous_end-to-end_suturing.md)
- [\[NeurIPS 2025\] C-NAV: Towards Self-Evolving Continual Object Navigation in Open World](c-nav_towards_self-evolving_continual_object_navigation_in_open_world.md)

<!-- RELATED:END -->
