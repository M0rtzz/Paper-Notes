---
title: >-
  [论文解读] Scaling Offline RL via Efficient and Expressive Shortcut Models
description: >-
  [NeurIPS 2025][图像生成][Offline RL] 提出 SORL，利用 shortcut models 的自一致性实现离线 RL 中高效一阶段训练与可变推理步数的策略优化，同时支持推理时的顺序和并行扩展。
tags:
  - NeurIPS 2025
  - 图像生成
  - Offline RL
  - Shortcut Models
  - Flow Matching
  - Self-Consistency
  - Test-time Scaling
---

# Scaling Offline RL via Efficient and Expressive Shortcut Models

**会议**: NeurIPS 2025  
**arXiv**: [2505.22866](https://arxiv.org/abs/2505.22866)  
**代码**: [nico-espinosadice.github.io/projects/sorl](https://nico-espinosadice.github.io/projects/sorl) (有)  
**领域**: 图像生成  
**关键词**: Offline RL, Shortcut Models, Flow Matching, Self-Consistency, Test-time Scaling

## 一句话总结

提出 SORL，利用 shortcut models 的自一致性实现离线 RL 中高效一阶段训练与可变推理步数的策略优化，同时支持推理时的顺序和并行扩展。

## 研究背景与动机

**领域现状**：离线强化学习 (Offline RL) 通过固定数据集训练智能体而无需在线探索。扩散模型和流匹配 (Flow Matching) 作为强大的生成模型可建模多模态行为分布。

**现有痛点**：将扩散/流模型应用于离线 RL 面临两大挑战：(a) 迭代噪声采样过程使策略优化困难，需反向传播穿过多步时间；(b) 推理效率低，多步生成过程缓慢。

**核心矛盾**：训练效率要求较少去噪步数（避免反向传播多步），但建模复杂分布需更多离散化步数以保证表达力。推理时既需快速生成（如自动驾驶），又需精确动作（如手术机器人）。

**本文目标**：如何在保持表达力的同时实现高效训练，并支持推理时按需扩展计算？

**切入角度**：引入 shortcut models——一种可在任意推理预算下生成高质量样本的新型生成模型。关键在于通过自一致性 (self-consistency) 将不同步数的去噪过程统一到单一模型中。

**核心 idea**：利用 shortcut models 的自一致性，在单阶段训练中解耦策略优化步数、正则化步数和推理步数，实现高效训练与灵活推理扩展。

## 方法详解

### 整体框架

SORL 基于 behavior-regularized actor-critic 架构，将策略建模为 shortcut function $s_\theta(z_t, t, h \mid x)$，该函数同时以时间步 $t$ 和步长 $h$ 为条件。通过 Euler 方法采样动作（Algorithm 2），支持任意推理步数 $M^{\text{inf}}$。

### 关键设计

**1. Shortcut Model 策略类**

- **功能**：将标准 flow matching 模型扩展为条件步长模型
- **核心思路**：模型 $s_\theta(z_t, t, h \mid x)$ 预测从 $z_t$ 到 $z_{t+h}$ 的归一化方向，使得 $z_t + s(z_t, t, h) \cdot h \approx z_{t+h}$
- **动机**：传统 flow matching 需小步长才能精确推理，shortcut model 通过学习大步跳跃实现高效推理

**2. 三组分 Actor 损失**

SORL 的训练目标包含三部分：

$$\mathcal{L}_\pi(\theta) = \mathcal{L}_{\text{QL}}(\theta) + \mathcal{L}_{\text{FM}}(\theta) + \mathcal{L}_{\text{SC}}(\theta)$$

**(a) Q 损失** — 策略优化：

$$\mathcal{L}_{\text{QL}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{a^\pi \sim \pi_\theta(\cdot|x)} [-Q_\phi(x, a^\pi)]$$

采样动作时使用最多 $M^{\text{BTT}}$ 步的推理过程（即反向传播穿越 $M^{\text{BTT}}$ 步时间）。$M^{\text{BTT}}$ 通常很小（1, 2, 4, 8），使反向传播高效。

**(b) Flow Matching 损失** — 离线数据正则化：

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}[\|s_\theta(a^t, t, 1/M^{\text{disc}} \mid x) - (a^1 - a^0)\|^2]$$

确保最小步长下模型恢复真实漂移方向 $a^1 - a^0$，其中 $a^0 \sim \mathcal{N}(0, I)$, $a^1 \sim \mathcal{D}$。

**(c) Self-Consistency 损失** — 步长一致性：

$$\mathcal{L}_{\text{SC}}(\theta) = \mathbb{E}[\|s_\theta(a^t, t, 2h \mid x) - s_{\text{target}}\|^2]$$

其中 $s_{\text{target}} = \frac{1}{2}[s_\theta(a^t, t, h \mid x) + s_\theta(a^{t+h}, t+h, h \mid x)]$，确保一大步 $2h$ 等价于两小步 $h$。

**3. 推理时扩展**

- **顺序扩展 (Sequential Scaling)**：增加推理步数 $M^{\text{inf}}$（最多到训练时的 $M^{\text{disc}}$）
- **并行扩展 (Parallel Scaling)**：Best-of-$N$ 采样——从策略中独立采样 $N$ 个动作，用 $Q$ 函数作为验证器选择最优动作：$\arg\max_{a \in \{a_1, ..., a_N\}} Q(x, a)$

### 损失函数/训练策略

- Critic 损失采用标准 Bellman 误差最小化：$\mathcal{L}_Q(\phi) = (Q_\phi(x, a^1) - r - \gamma Q_{\phi}^{\text{target}}(x', a_{x'}^\pi))^2$
- 训练为单阶段 (one-stage)，无需蒸馏或两阶段流程
- 步长 $h$ 从 2 的幂次集合中均匀采样，$t$ 在 $[0,1]$ 上均匀采样

## 实验关键数据

### 主实验

在 OGBench 任务套件上对比 10 个基线（3 个高斯策略、3 个扩散策略、4 个流策略），共 40 个任务，8 个种子。

| 环境 | BC | IQL | ReBRAC | IDQL | FQL | **SORL** |
|------|-----|-----|--------|------|-----|----------|
| antmaze-large (5 tasks) | 11 | 53 | 81 | 21 | 79 | **89±2** |
| antmaze-giant (5 tasks) | 0 | 4 | **26** | 0 | 9 | 9±6 |
| humanoidmaze-medium (5 tasks) | 2 | 33 | 22 | 1 | 58 | **64±4** |
| humanoidmaze-large (5 tasks) | 1 | 2 | 2 | 1 | 4 | 5±2 |
| antsoccer-arena (5 tasks) | 1 | 8 | 0 | 12 | 60 | **69±2** |
| cube-single (5 tasks) | 5 | 83 | 91 | 95 | 96 | **97±1** |
| cube-double (5 tasks) | 2 | 7 | 12 | 15 | **29** | 25±3 |
| scene (5 tasks) | 5 | 28 | 41 | 46 | 56 | **57±2** |

SORL 在 8 个环境中的 5 个取得最佳表现。

### 消融实验

| 推理设置 | 效果 |
|----------|------|
| $M^{\text{inf}}=1$ (1步推理) | 可行但性能较低 |
| $M^{\text{inf}}=2,4,8$ | 性能随步数递增 |
| $M^{\text{BTT}}=1$ + 并行扩展 | 可恢复 $M^{\text{BTT}}=8$ 的最优性能 |
| Best-of-8 + 更多推理步 | 超越训练时步数的泛化 |

### 关键发现

1. 固定训练预算，增加推理步数可持续提升性能（顺序扩展）
2. 减少训练计算（$M^{\text{BTT}}=1,2,4$），可通过推理时扩展恢复最优性能
3. 推理步数可泛化到超出训练时反向传播步数的范围
4. antmaze-large 上比 FQL 提升约 10 个百分点（89 vs 79），antsoccer-arena 上提升 9 个百分点（69 vs 60）

## 亮点与洞察

1. **巧妙的解耦设计**：利用 shortcut models 的自一致性，将策略优化步数 $M^{\text{BTT}}$、离散化步数 $M^{\text{disc}}$、推理步数 $M^{\text{inf}}$ 三者完全解耦
2. **理论保证**：Theorem 2 证明 shortcut model 在所有步长下生成的分布与目标分布在 2-Wasserstein 距离下接近，上界分解为离散化误差 + FM 误差 + SC 误差
3. **训练-推理计算的可交换性**：少训练多推理可恢复多训练少推理的性能，为资源受限场景提供灵活方案
4. **一阶段训练**：避免了蒸馏方法的两阶段复杂性和误差累积

## 局限与展望

1. antmaze-giant 和 humanoidmaze-large 上 SORL 表现中等，复杂长距离规划仍是挑战
2. 并行扩展（Best-of-N）的收益无理论保证——用学习的 $Q$ 函数作验证器而非真实奖励
3. 自一致性训练与标准 flow matching 相比增加了计算开销
4. 未探索自适应推理步数选择（如根据 $Q$ 梯度动态调整步数）
5. 离散化步数需为 2 的幂次是对设计空间的限制

## 相关工作与启发

- **FQL (Flow Q-Learning)**：最接近的基线，使用流模型做离线 RL 但需蒸馏得到单步推理能力；SORL 在此基础上统一了多步推理
- **Shortcut Models (Frans et al., 2024)**：SORL 首次将 shortcut models 引入离线 RL，该线工作为 LLM 推理时扩展 (test-time compute scaling) 在 RL 领域的对应
- **启发**：self-consistency 思想可能泛化到机器人策略的在线部署——推理资源充裕时多步精确执行，资源受限时单步快速响应

## 评分

⭐⭐⭐⭐ (4/5)

- 创新性 ⭐⭐⭐⭐：shortcut models + offline RL 的结合新颖，推理时扩展的思路有启发性
- 理论 ⭐⭐⭐⭐：W2 距离正则化保证原创且严谨
- 实验 ⭐⭐⭐⭐：40 个任务的全面评估，但部分环境改进有限
- 工程价值 ⭐⭐⭐⭐：一阶段训练 + 灵活推理，实用性强
