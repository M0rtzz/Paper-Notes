---
title: >-
  [论文解读] Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn
description: >-
  [ICML2025][可塑性丧失] 通过 NTK 矩阵建立可塑性丧失 (plasticity loss) 与 churn（批外数据输出漂移）之间的因果联系，提出 C-CHAIN 方法在持续 RL 训练中持续抑制 churn，从而缓解可塑性丧失，在 24 个持续 RL 环境上超越已有基线。
tags:
  - ICML2025
  - 可塑性丧失
  - churn
  - Neural Tangent Kernel
  - 持续学习
  - 梯度去相关
---

# Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn

**会议**: ICML2025  
**arXiv**: [2506.00592](https://arxiv.org/abs/2506.00592)  
**代码**: [bluecontra/C-CHAIN](https://github.com/bluecontra/C-CHAIN)  
**领域**: 持续强化学习 (Continual RL)  
**关键词**: 可塑性丧失, churn, Neural Tangent Kernel, 持续学习, 梯度去相关

## 一句话总结

通过 NTK 矩阵建立可塑性丧失 (plasticity loss) 与 churn（批外数据输出漂移）之间的因果联系，提出 C-CHAIN 方法在持续 RL 训练中持续抑制 churn，从而缓解可塑性丧失，在 24 个持续 RL 环境上超越已有基线。

## 研究背景与动机

### 问题定义

深度强化学习中，智能体使用非线性函数逼近器在非平稳数据分布下训练时，会逐渐丧失**可塑性 (plasticity)**——即适应新任务/新数据分布的能力。这一现象在**持续 RL** 场景（任务序列 $\mathbb{T} = \{\mathcal{T}_1, \mathcal{T}_2, \ldots, \mathcal{T}_k\}$ 依次到达）中尤为严重。

### 已有方案的不足

已有缓解策略包括：重置休眠神经元 (ReDo)、参数正则化 (L2 Init)、反向传播变体 (Continual BP)、周期性参数重置 (Shrink & Perturb)、动态稀疏训练等。然而这些方法大多是从网络结构或参数层面进行干预，对可塑性丧失发生的**动力学机制**缺乏深入理解。

### 本文视角：Churn

**Churn** 定义为：mini-batch 训练导致的**批外数据 (out-of-batch)** 网络输出的隐式变化。设网络参数从 $\theta$ 更新到 $\theta'$，则参考数据 $\bar{x} \notin B_{\text{train}}$ 上的 churn 为：

$$C_f(\bar{x}, \theta, \Delta_\theta) = f_{\theta'}(\bar{x}) - f_\theta(\bar{x}) \approx \nabla_\theta f_\theta(\bar{x})^\top \Delta_\theta$$

本文的核心观察是：**可塑性丧失与 churn 加剧高度相关**，二者通过 NTK 矩阵建立联系。

## 方法详解

### 1. NTK 作为桥梁

经验 NTK 矩阵定义为所有数据点的梯度点积矩阵：

$$N_\theta(i,j) = \nabla_\theta f_\theta(x_i)^\top \nabla_\theta f_\theta(x_j), \quad N_\theta = G_\theta^\top G_\theta$$

其中 $G_\theta = [g(x_1), g(x_2), \ldots]$ 是所有数据点的梯度矩阵。将参数更新代入 churn 的一阶近似，得到 churn 的向量形式：

$$C_f(\theta, \Delta_\theta) \approx -\eta \, N_\theta \, S \, G_L$$

- $S$：采样矩阵（对角 0/1），标记哪些数据在训练 batch 中
- $G_L$：损失函数的梯度向量
- $\eta$：学习率

**关键结论**：NTK 矩阵 $N_\theta$ 同时决定了可塑性（其秩）和 churn 的大小，是二者的天然桥梁。

### 2. NTK 秩坍缩引发 Churn 恶化

在持续学习中，误差动态满足迭代关系：

$$\mathcal{E}_i(\theta_{t+1}) = (I - \eta \, N_{\theta_t} S_i) \, \mathcal{E}_i(\theta_t)$$

- 当 $N_\theta$ 满秩（对角正、非对角为零），学习过程稳定，类似表格型逼近
- 实际中，SGD 训练使梯度趋向相关 → NTK 非对角项增大 → **秩下降**
- 任务切换时，此前通过 churn 隐式塑造的函数景观不匹配新分布 → churn 进一步加剧
- 形成**恶性循环**：NTK 秩下降 ↔ churn 加剧 → 可塑性丧失

### 3. C-CHAIN：Continual Churn Approximated Reduction

**核心思想**：在常规 RL 训练的同时，持续最小化批外参考数据上的 churn。

**Churn 减少损失函数**：

$$L_f^{\text{cr}}(\theta) = \frac{1}{2} \mathbb{E}_{\bar{x} \in B_\text{ref}} \left[ C_f(\bar{x}, \theta, \Delta_\theta)^2 \right]$$

**总训练损失**：

$$L_{\text{total}} = L_{\text{RL}}(\theta) + \lambda \, L_f^{\text{cr}}(\theta)$$

其中 $\lambda$ 为权衡系数，$B_\text{ref}$ 为从 replay buffer 中额外采样的参考 batch。

### 双重效应的理论分析

C-CHAIN 的 churn 减少梯度对学习动态有两重作用：

| 效应 | 机制 | 作用 |
|------|------|------|
| **梯度去相关** (Gradient Decorrelation) | 抑制 NTK 矩阵 $N_\theta$ 的非对角项 | 防止秩坍缩，维持可塑性 |
| **步长自适应调整** (Step-size Adjustment) | 对批外数据的梯度进行投影 | 自适应控制更新幅度，稳定学习 |

### 算法流程

```
输入: 任务序列 T = {T_1, ..., T_k}，每任务交互步数 N
对每个任务 T_j:
    for step = 1 to N:
        1. 收集经验 (s, a, r, s')，存入 replay buffer
        2. 采样训练 batch B_train 和参考 batch B_ref
        3. 计算 RL 损失 L_RL 并记录参考 batch 的网络输出 f_θ(B_ref)
        4. 执行 RL 梯度更新 → θ'
        5. 计算 churn: C = f_θ'(B_ref) - f_θ(B_ref)
        6. 计算 churn 减少损失 L_cr = (1/2) ||C||^2
        7. 额外梯度更新以最小化 L_cr
```

## 实验关键数据

### 实验设置

- **4 大基准**：OpenAI Gym Control、ProcGen、DeepMind Control Suite、MinAtar
- **共 24 个持续 RL 环境**：每个基准构建多个任务序列
- **基线方法**：Vanilla (无干预)、L2 Init、Shrink & Perturb (S&P)、ReDo、Continual BP、PLASTIC

### 主要结果

| 基准 | 环境数 | C-CHAIN 表现 | 关键发现 |
|------|--------|-------------|---------|
| OpenAI Gym Control | 6 | 大多数环境最优 | 持续 churn 减少显著改善后期任务性能 |
| ProcGen | 6 | 多环境最优 | 在高视觉复杂度任务上优势明显 |
| DeepMind Control Suite | 6 | 大多数环境最优 | 连续控制任务中有效缓解性能退化 |
| MinAtar | 6 | 大多数环境最优 | 离散动作空间中同样有效 |

### 分析实验

- **NTK 秩追踪**：C-CHAIN 训练过程中 NTK 矩阵的有效秩显著高于 Vanilla，验证了梯度去相关效应
- **Churn 量化**：C-CHAIN 下 churn 的幅度随训练推进保持较低水平，而 Vanilla 的 churn 持续增长
- **后期任务性能**：在长序列（≥10 个任务）中，C-CHAIN 与 Vanilla 的性能差距逐渐拉大，说明可塑性保持效果随时间累积

## 亮点与洞察

1. **理论洞察新颖**：首次通过 NTK 矩阵统一了可塑性丧失和 churn 两个看似独立的现象，揭示了二者间的恶性循环机制
2. **方法简洁优雅**：C-CHAIN 只需额外采样一个参考 batch 并计算 churn 减少损失，实现简单、即插即用，可与任意 RL 算法结合
3. **双重效应的理论保证**：不仅经验有效，还从梯度去相关和步长调整两个角度提供了形式化分析
4. **实验覆盖全面**：24 个环境横跨离散/连续动作、低维/高维观测、不同物理引擎，说服力强
5. **代码开源**：GitHub 公开可复现

## 局限与展望

1. **额外计算开销**：每步需要额外前向传播参考 batch 并计算 churn 损失，增加约 30-50% 的计算量
2. **$\lambda$ 超参数敏感性**：churn 减少损失的权重需要调优，不同环境的最优值可能不同
3. **参考 batch 采样策略**：当前从 replay buffer 中均匀采样，未探索更有针对性的采样策略（如优先经验重放）
4. **仅关注前向迁移**：本文聚焦可塑性（前向迁移），未讨论灾难性遗忘（后向迁移）的影响
5. **NTK 分析依赖一阶近似**：churn 的 NTK 表达式依赖于一阶 Taylor 展开，高阶项在大步长或深网络中可能不可忽略
6. **任务边界假设**：实验中任务切换是硬切换，未测试渐变式非平稳性

## 相关工作与启发

- **Churn 原始工作** (Schaul et al., 2022; Tang & Berseth, 2024)：单 MDP 下的 churn 研究，本文是其在持续 RL 的自然扩展
- **NTK 与可塑性** (Lyle et al., 2024)：NTK 秩作为可塑性指标的经验发现，本文提供了因果层面的理论解释
- **周期性重置** (Nikishin et al., 2022; Schwarzer et al., 2023)：通过"暴力重启"恢复可塑性，C-CHAIN 提供了更温和的替代方案
- **PLASTIC** (Lee et al., 2024)：结合多种技巧的组合方法，C-CHAIN 从单一原理出发更具解释性

## 评分

- 新颖性: ⭐⭐⭐⭐ — NTK 视角统一可塑性与 churn 是全新理论贡献
- 实验充分度: ⭐⭐⭐⭐ — 24 个环境、4 大基准、包含分析实验
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，从观察到方法到验证逻辑流畅
- 价值: ⭐⭐⭐⭐ — 对持续 RL 的可塑性问题提供了理论理解和实用方案

<!-- RELATED:START -->

## 相关论文

- [Position: Lifetime Tuning is Incompatible with Continual Reinforcement Learning](position_lifetime_tuning_is_incompatible_with_continual_reinforcement_learning.md)
- [Continual Reinforcement Learning by Planning with Online World Models](continual_reinforcement_learning_by_planning_with_online_world_models.md)
- [Continual Knowledge Adaptation for Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/continual_knowledge_adaptation_for_reinforcement_learning.md)
- [Temporal-Difference Variational Continual Learning](../../NeurIPS2025/reinforcement_learning/temporal-difference_variational_continual_learning.md)
- [Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](../../ICLR2026/reinforcement_learning/principled_fast_and_meta_knowledge_learners_for_continual_reinforcement_learning.md)

<!-- RELATED:END -->
