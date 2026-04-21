---
title: >-
  [论文解读] Temporal-Difference Variational Continual Learning
description: >-
  [NeurIPS 2025][持续学习] 提出TD-VCL目标函数，将变分持续学习（VCL）中的学习目标重新表示为多个过去后验估计的加权组合，揭示了与强化学习中时序差分（TD）方法的深层联系，通过"分散"正则化压力有效缓解了近似误差的逐步累积问题。
tags:
  - NeurIPS 2025
  - 持续学习
  - 变分推断
  - 时序差分
  - 灾难性遗忘
  - 贝叶斯学习
---

# Temporal-Difference Variational Continual Learning

**会议**: NeurIPS 2025  
**arXiv**: [2410.07812](https://arxiv.org/abs/2410.07812)  
**代码**: https://github.com/luckeciano/TD-VCL (有)  
**领域**: 强化学习 / 持续学习  
**关键词**: 持续学习, 变分推断, 时序差分, 灾难性遗忘, 贝叶斯学习

## 一句话总结

提出TD-VCL目标函数，将变分持续学习（VCL）中的学习目标重新表示为多个过去后验估计的加权组合，揭示了与强化学习中时序差分（TD）方法的深层联系，通过"分散"正则化压力有效缓解了近似误差的逐步累积问题。

## 研究背景与动机

持续学习（CL）要求模型在非平稳数据流上连续学习新任务，同时保持旧任务的性能。核心挑战是可塑性（学新）与记忆稳定性（记旧）之间的平衡，失衡时模型将遭受灾难性遗忘。

在贝叶斯CL框架中，变分持续学习（VCL）利用后验的递归关系：$p(\boldsymbol{\theta}|\mathcal{D}_{1:T}) \propto p(\boldsymbol{\theta}|\mathcal{D}_{1:T-1})p(\mathcal{D}_T|\boldsymbol{\theta})$，通过在线变分推断逐步更新后验。VCL的优化目标为最大化当前任务似然并约束新后验不偏离前一步后验：

$$\mathcal{L}_{VCL}^t = \mathbb{E}[\log p(\mathcal{D}_t|\boldsymbol{\theta})] - D_{KL}(q_t(\boldsymbol{\theta}) \| q_{t-1}(\boldsymbol{\theta}))$$

**核心问题**在于VCL仅依赖**最近一步的后验估计** $q_{t-1}$ 作为正则化目标。如果某一步的后验估计质量特别差，该误差将完全传递给下一步，并随着递归更新逐步累积（compounding error），最终导致当前估计严重偏离真实后验。

本文的核心insight是：**同一个优化目标可以等价地表示为多个过去后验估计和任务似然的函数**。通过将正则化"分散"到多个历史后验上，任何单个后验估计的误差影响都被稀释，正确的后验估计可以施加纠正性影响。

## 方法详解

### 整体框架

从VCL的标准KL最小化目标出发，推导出两种等价但实践中更优的目标函数：n-Step KL VCL和TD(λ)-VCL。两者本质上是对多个n-Step TD目标的组合，与强化学习中的TD学习具有深刻的结构对应关系。

### 关键设计

1. **n-Step KL正则化目标（Proposition 4.1）**: 将标准VCL目标等价改写为：

   $$q_t = \arg\max_q \mathbb{E}\left[\sum_{i=0}^{n-1}\frac{n-i}{n}\log p(\mathcal{D}_{t-i}|\boldsymbol{\theta})\right] - \sum_{i=0}^{n-1}\frac{1}{n}D_{KL}(q_t \| q_{t-i-1})$$

   这个目标的KL正则化项均匀分布在 $n$ 个过去后验上。如果某个 $q_{t-i}$ 偏差很大，它仅影响 $1/n$ 的正则化项。似然项包含多个任务的似然并按时间远近加权，**数据回放自然地从目标函数中涌现**。

2. **TD(λ)-VCL目标（Proposition 4.2）**: 在n-Step KL基础上引入几何衰减权重 $\lambda^i$，使最近的后验估计获得更大权重：

   $$\text{KL权重} \propto \frac{\lambda^i(1-\lambda)}{1-\lambda^n}, \quad \text{似然权重} \propto \frac{\lambda^i(1-\lambda^{n-i})}{1-\lambda^n}$$

   这提供了更精细的控制：$\lambda \to 0$ 退化为VCL，$\lambda = 1$ 退化为n-Step KL。该目标等价于多个TD目标的加权和（Proposition 4.4），形成了与强化学习中λ-return的精确对应。

3. **n-Step TD目标（Definition 4.3）**: 定义CL中的TD目标为 $\text{TD}_t(n) = \mathbb{E}[\sum_{i=0}^{n-1}\log p(\mathcal{D}_{t-i}|\boldsymbol{\theta})] - D_{KL}(q_t \| q_{t-n})$，对比更远的后验估计。每个TD目标在精确推断下等价于标准VCL目标，但在近似推断下提供不同的偏差-方差权衡。

### 损失函数 / 训练策略

使用高斯均场近似后验和高斯先验 $\mathcal{N}(0, \sigma^2\mathbf{I})$。KL项解析计算，期望对数似然项用重参数化技巧的蒙特卡洛近似。采用似然温度调节（likelihood-tempering）防止变分过度剪枝。测试时通过MC采样边际化近似后验来计算后验预测分布。

## 实验关键数据

### 主实验

| 基准 (t=最终) | 指标 | TD(λ)-VCL | VCL | VCL CoreSet | 提升 |
|-------------|------|-----------|-----|------------|------|
| PermutedMNIST-Hard (t=10) | Avg Acc | **0.89** | 0.78 | 0.81 | +0.11 |
| SplitMNIST-Hard (t=5) | Avg Acc | **0.66** | 0.64 | 0.62 | +0.02 |
| SplitNotMNIST-Hard (t=5) | Avg Acc | **0.58** | 0.51 | 0.51 | +0.07 |
| CIFAR100-10 (t=10) | Avg Acc | **0.71** | 0.66 | 0.65 | +0.05 |
| TinyImageNet-10 (t=10) | Avg Acc | **0.56** | 0.51 | 0.54 | +0.05 |

### 消融实验（TD目标增强其他贝叶斯CL方法）

| 方法 | PermutedMNIST (t=10) | SplitMNIST (t=5) | 说明 |
|------|---------------------|------------------|------|
| VCL | 0.78 | 0.64 | 基线 |
| TD(λ)-VCL | **0.89** | **0.67** | +0.11 / +0.03 |
| UCL | 0.73 | 0.66 | 基线 |
| TD(λ)-UCL | **0.84** | **0.70** | +0.11 / +0.04 |
| UCB | 0.77 | 0.66 | 基线 |
| TD(λ)-UCB | **0.85** | **0.69** | +0.08 / +0.03 |

### 关键发现

- TD-VCL在所有基准上一致超越标准VCL，尤其在任务数量增加时优势更加明显
- 逐任务分析（Figure 3）表明灾难性遗忘对早期任务影响最大，TD-VCL对此显著更鲁棒——Task 1在10个任务后保持约80-85%准确率vs VCL的50-60%
- TD目标对UCL和UCB同样有效，证明其通用性——与不同的变分方法正交且互补
- SplitNotMNIST-Hard中，TD-VCL是唯一在所有任务后保持非trivial准确率的方法

## 亮点与洞察

- 理论贡献优雅：证明了VCL目标的等价重写和与TD学习的联系，三个命题层层递进
- "数据回放从目标函数中自然涌现"的洞察非常有趣——旧任务似然的重估并非启发式添加而是目标函数的固有组成部分
- TD-VCL构成了从VCL（$\lambda \to 0$）到n-Step KL（$\lambda = 1$）的连续谱，提供了灵活的偏差-方差权衡机制
- 与神经科学中TD学习的联系为该方法提供了额外的动机和解释

## 局限与展望

- 需要存储多个过去后验估计，内存开销随 $n$ 增加
- 需要存储或回放旧任务数据以估计似然项（但这在目标函数中是自然的而非启发式的）
- 实验主要在相对小规模的任务和网络上进行，大规模预训练模型的行为有待验证
- 超参数 $n$ 和 $\lambda$ 的最优选择尚缺乏理论指导

## 相关工作与启发

- **vs VCL (Nguyen et al.)**: VCL仅用最近一步后验正则化，TD-VCL用多步，且回放机制从目标函数自然涌现而非启发式添加
- **vs UCL/UCB**: 这些方法改进了正则化和学习率自适应机制，TD目标与它们正交互补
- **vs EWC**: EWC用Fisher信息正则化，属于非贝叶斯方法；TD-VCL在贝叶斯框架内提供了更有原则的解决方案
- **vs TD学习（RL）**: 揭示了VCL后验更新与RL中值函数更新的结构对应，λ-return在两个领域中扮演相同的角色

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将VCL与TD学习联系起来的洞察非常独到，理论推导优雅
- 实验充分度: ⭐⭐⭐⭐ 五个基准、三种基础方法、逐任务分析，但缺少大规模实验
- 写作质量: ⭐⭐⭐⭐⭐ 动机明确，理论与直觉兼顾，图示清晰直观
- 价值: ⭐⭐⭐⭐ 对贝叶斯CL领域有重要推进，TD连接的跨域洞察启发性强

<!-- RELATED:START -->

## 相关论文

- [Continual Knowledge Adaptation for Reinforcement Learning](continual_knowledge_adaptation_for_reinforcement_learning.md)
- [Language Model Distillation: A Temporal Difference Imitation Learning Perspective](../../AAAI2026/reinforcement_learning/language_model_distillation_a_temporal_difference_imitation_learning_perspective.md)
- [Incremental Sequence Classification with Temporal Consistency](incremental_sequence_classification_with_temporal_consistency.md)
- [STAIR: Addressing Stage Misalignment through Temporal-Aligned Preference Reinforcement Learning](stair_addressing_stage_misalignment_through_temporal-aligned_preference_reinforc.md)
- [Modulation of Temporal Decision-Making in a Deep Reinforcement Learning Agent under the Dual-Task Paradigm](modulation_of_temporal_decision-making_in_a_deep_reinforcement_learning_agent_un.md)

<!-- RELATED:END -->
