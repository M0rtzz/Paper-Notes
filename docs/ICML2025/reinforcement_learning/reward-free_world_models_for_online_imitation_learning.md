---
title: >-
  [论文解读] Reward-free World Models for Online Imitation Learning
description: >-
  [ICML2025][强化学习][imitation learning] 提出 IQ-MPC，一种无需显式奖励建模的世界模型在线模仿学习方法，通过逆软Q学习在潜空间中联合学习动态模型与Q函数，利用 MPPI 规划实现对高维观测和复杂动力学任务的稳定专家级模仿。 - 离线模仿学习的局限：行为克隆（BC）方法（如 Diffusi…
tags:
  - "ICML2025"
  - "强化学习"
  - "imitation learning"
  - "world model"
  - "reward-free"
  - "inverse soft-Q learning"
  - "model predictive control"
  - "latent dynamics"
---

# Reward-free World Models for Online Imitation Learning

**会议**: ICML2025  
**arXiv**: [2410.14081](https://arxiv.org/abs/2410.14081)  
**代码**: 待确认  
**领域**: 模仿学习 / 世界模型 / 模型预测控制  
**关键词**: imitation learning, world model, reward-free, inverse soft-Q learning, model predictive control, latent dynamics

## 一句话总结

提出 IQ-MPC，一种无需显式奖励建模的世界模型在线模仿学习方法，通过逆软Q学习在潜空间中联合学习动态模型与Q函数，利用 MPPI 规划实现对高维观测和复杂动力学任务的稳定专家级模仿。

## 研究背景与动机

- **离线模仿学习的局限**：行为克隆（BC）方法（如 Diffusion Policy、Implicit BC）依赖大量专家数据，但无法处理分布外（OOD）状态，容易产生误差累积和性能退化
- **在线模仿学习的挑战**：现有在线 IL 方法（GAIL、IQ-Learn、CFIL）在高维观测/动作空间和复杂动力学任务中表现不佳；基于 IRL 的 min-max 优化在 reward-policy 空间中训练不稳定
- **世界模型的潜力**：TD-MPC 系列等无解码器世界模型在 RL 任务中展现了优秀的采样效率和规划能力，但尚未有效应用到无奖励的模仿学习场景
- **核心动机**：能否利用世界模型的动态建模能力提升在线模仿学习的性能，同时完全消除对显式奖励模型的依赖？

## 方法详解

### 整体架构：IQ-MPC

IQ-MPC 由四个核心组件构成，全部在潜空间操作，无需重建原始观测：

1. **编码器** $h$：$\mathbf{z} = h(\mathbf{s})$，将状态映射到潜表示
2. **潜动态模型** $d$：$\mathbf{z}' = d(\mathbf{z}, \mathbf{a})$，预测下一潜状态
3. **Q 函数** $Q$：$\hat{q} = Q(\mathbf{z}, \mathbf{a})$，估计状态-动作值
4. **策略先验** $\pi$：$\hat{\mathbf{a}} = \pi(\mathbf{z})$，引导 MPPI 规划

系统维护两个独立回放缓冲区：专家缓冲区 $\mathcal{B}_E$ 和行为缓冲区 $\mathcal{B}_\pi$。

### 核心思想：Q-Policy 空间的无奖励优化

关键洞察：逆 Bellman 算子 $\mathcal{T}^\pi$ 建立了 Q 空间与奖励空间的双射映射：

$$r(\mathbf{s}, \mathbf{a}) = Q(\mathbf{s}, \mathbf{a}) - \gamma \mathbb{E}_{\mathbf{s}' \sim \mathcal{P}(\cdot|\mathbf{s},\mathbf{a})} V^\pi(\mathbf{s}')$$

因此无需单独训练奖励模型，奖励可直接从 Q 值和策略中解码得到。优化从 reward-policy 空间转移到 Q-policy 空间，避免了 min-max 优化的不稳定性。

### 联合训练损失

编码器、动态模型和 Q 函数的联合训练目标：

$$\mathcal{L} = \sum_{t=0}^{H} \lambda^t \left( \mathbb{E}_{(\mathbf{s}_t, \mathbf{a}_t, \mathbf{s}'_t) \sim \mathcal{B}} \| \mathbf{z}_{t+1} - \text{sg}(h(\mathbf{s}'_t)) \|_2^2 \right) + \mathcal{L}_{iq}$$

- 第一项为**一致性损失**：确保动态模型预测的潜状态与编码器编码的实际下一状态一致（sg 为 stop-gradient）
- 第二项为**逆软Q损失** $\mathcal{L}_{iq}$：采用 $\chi^2$ 正则化，包含三部分——专家数据上的 Q 估计、初始状态值函数项、以及对 Q 值幅度的正则惩罚

### 策略先验学习

策略通过最大熵 RL 目标学习：

$$\mathcal{L}_\pi = \sum_{t=0}^{H} \lambda^t \left[ \mathbb{E}_{(\mathbf{s}_t, \mathbf{a}_t) \sim \mathcal{B}} \left[ -Q(\mathbf{z}_t, \pi(\mathbf{z}_t)) + \beta \log(\pi(\cdot|\mathbf{z}_t)) \right] \right]$$

其中 $\beta$ 为固定熵系数。策略学习使用专家和行为缓冲区的混合数据。

### 梯度惩罚稳定训练

为应对 critic 判别能力过强导致策略学习失败的问题，引入 Wasserstein-1 梯度惩罚：

$$\mathcal{L}_{pen} = \sum_{t=0}^{H} \lambda^t \left[ \mathbb{E}_{(\hat{\mathbf{s}}_t, \hat{\mathbf{a}}_t) \sim \mathcal{B}} \left( \| \nabla Q(\hat{\mathbf{z}}_t, \hat{\mathbf{a}}_t) \|_2 - 1 \right)^2 \right]$$

通过专家和行为样本间的线性插值生成梯度惩罚点，强制 Q 函数满足 Lipschitz 条件。

### MPPI 规划（推理阶段）

推理时使用 MPPI（Model Predictive Path Integral）进行无梯度规划：

1. 编码当前状态 $\mathbf{z}_t = h(\mathbf{s}_t)$
2. 从高斯分布和策略先验分别采样 $N$ 和 $N_\pi$ 条动作轨迹
3. 通过动态模型 roll out，用逆 Bellman 算子解码奖励 $r(\mathbf{z},\mathbf{a}) = Q(\mathbf{z},\mathbf{a}) - \gamma V^\pi(\mathbf{z}')$
4. 累计软回报并加终端值估计 $\gamma^H V^\pi(\mathbf{z}_H)$
5. 迭代更新高斯参数 $(\mu, \sigma)$，执行第一个动作

## 实验关键数据

### 运动控制任务（DMControl, 状态输入）

| 任务 | IQL+SAC | CFIL+SAC | HyPE | **IQ-MPC** |
|------|---------|----------|------|-----------|
| Hopper Hop | 不稳定 | 不稳定 | 中等 | **稳定专家级** |
| Walker Run | 中等 | 低 | 中等 | **稳定专家级** |
| Humanoid Walk | 低 | 低 | 中等 | **稳定专家级** |
| Dog Walk | 低 | 低 | 中等 | **最优** |

- 低维任务用 100 条专家轨迹，Humanoid 用 500 条，Dog 用 1000 条

### 灵巧手操控任务（MyoSuite, 成功率）

| 任务 | IQL+SAC | CFIL+SAC | HyPE | **IQ-MPC** |
|------|---------|----------|------|-----------|
| Key Turn | 0.72±0.04 | 0.65±0.08 | 0.55±0.09 | **0.87±0.03** |
| Object Hold | 0.00±0.00 | 0.01±0.01 | 0.13±0.10 | **0.96±0.03** |
| Pen Twirl | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | **0.73±0.05** |

- 仅使用 100 条专家轨迹（每条 100 步）
- Pen Twirl 任务中基线方法成功率均为 0，IQ-MPC 达到 73%

### 视觉输入实验（DMControl, 图像观测）

- 仅替换编码器为浅层卷积网络，模型其余部分不变
- 在 Cheetah Run、Walker Run 上显著优于视觉版 IQL+SAC
- 在 Walker Walk 上与基线性能相当

### 专家轨迹数量消融

- Hopper Hop：10 条专家轨迹即可达专家级性能（100 条收敛更快）
- Object Hold：5 条专家轨迹即可达专家级性能
- 5 条轨迹在 Hopper Hop 上出现不稳定

## 亮点与洞察

1. **无奖励世界模型**：通过逆 Bellman 算子从 Q 值直接解码奖励，完全省去奖励模型训练，降低系统复杂度
2. **Q-Policy 空间优化**：将传统 reward-policy 空间的 min-max 问题转化为 Q-policy 空间的优化，理论和实验均证明训练更稳定
3. **理论保证**：证明训练目标同时最小化策略回报差异的上界（T1: 分布匹配 + T2: 动态一致性）
4. **灵巧手操控突破**：在 MyoSuite 高维肌肉骨骼控制任务中，基线方法几乎全部失败（成功率~0），IQ-MPC 仍能达到 73-96% 成功率
5. **数据效率**：仅需 5-10 条专家轨迹即可实现专家级性能，展示了出色的样本效率
6. **模态无关设计**：从状态输入到视觉输入只需更换编码器，架构高度灵活

## 局限与展望

1. **专家数据获取**：仍需从训练好的 TD-MPC2 模型采样专家轨迹，真实场景中获取高质量专家演示可能困难
2. **计算开销**：MPPI 规划需要多轮迭代采样和 roll out，推理成本高于纯策略方法
3. **确定性环境假设**：实验环境多为确定性动力学，在高随机性环境中的表现有待验证
4. **单任务设计**：每个任务独立训练一个世界模型，缺乏跨任务的泛化能力
5. **真实机器人验证缺失**：所有实验均在仿真环境中进行，sim-to-real 迁移未被讨论

## 相关工作与启发

- **IQ-Learn** (Garg et al., 2021)：本文的理论基础，提出逆软Q学习将 IRL 优化从 reward 空间转到 Q 空间
- **TD-MPC / TD-MPC2** (Hansen et al., 2022/2023)：本文的架构蓝本，提供无解码器世界模型 + MPPI 规划框架
- **GAIL** (Ho & Ermon, 2016)：经典对抗模仿学习，本文在此基础上消除了 min-max 优化的不稳定性
- **CMIL** (Kolev et al., 2024)：保守世界模型做图像操控的模仿学习，理论分析中引用了其有界次优性引理

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将无解码器世界模型与逆软Q学习结合用于在线模仿学习属于新颖组合，无奖励规划的设计简洁优雅
- 实验充分度: ⭐⭐⭐⭐ — 覆盖运动/操控/视觉三类任务，消融实验充分，但缺少真实机器人验证
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，实验说明完整，符号统一规范
- 价值: ⭐⭐⭐⭐ — 灵巧手操控上的突破性表现（基线成功率~0 vs IQ-MPC 73-96%）展示了方法的实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Continual Reinforcement Learning by Planning with Online World Models](continual_reinforcement_learning_by_planning_with_online_world_models.md)
- [\[ICML 2025\] Online Pre-Training for Offline-to-Online Reinforcement Learning](online_pre-training_for_offline-to-online_reinforcement_learning.md)
- [\[NeurIPS 2025\] Foundation Models as World Models: A Foundational Study in Text-Based GridWorlds](../../NeurIPS2025/reinforcement_learning/foundation_models_as_world_models_a_foundational_study_in_text-based_gridworlds.md)
- [\[ICML 2025\] PIGDreamer: Privileged Information Guided World Models for Safe Partially Observable RL](pigdreamer_privileged_information_guided_world_models_for_safe_partially_observa.md)
- [\[NeurIPS 2025\] Dynamics-Aligned Latent Imagination in Contextual World Models for Zero-Shot Generalization](../../NeurIPS2025/reinforcement_learning/dynamics-aligned_latent_imagination_in_contextual_world_models_for_zero-shot_gen.md)

</div>

<!-- RELATED:END -->
