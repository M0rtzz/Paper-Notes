---
title: >-
  [论文解读] Prompt Tuning Decision Transformers with Structured and Scalable Bandits
description: >-
  [NeurIPS 2025][强化学习][Transformer] 提出一种基于多臂老虎机的结构化prompt调优方法，通过将prompt分解为独立segment并利用预训练PDT作为特征提取器，将prompt搜索复杂度从组合爆炸降为线性，在多任务离线RL中显著提升冻结PDT骨干网络的推理性能。 领域现状 Prompting…
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "Transformer"
  - "提示学习"
  - "Multi-Armed Bandit"
  - "离线多任务RL"
  - "少样本泛化"
---

# Prompt Tuning Decision Transformers with Structured and Scalable Bandits

**会议**: NeurIPS 2025  
**arXiv**: [2502.04979](https://arxiv.org/abs/2502.04979)  
**代码**: 已提供（附录材料）  
**领域**: 强化学习  
**关键词**: Decision Transformer, prompt tuning, Multi-Armed Bandit, 离线多任务RL, 少样本泛化

## 一句话总结

提出一种基于多臂老虎机的结构化prompt调优方法，通过将prompt分解为独立segment并利用预训练PDT作为特征提取器，将prompt搜索复杂度从组合爆炸降为线性，在多任务离线RL中显著提升冻结PDT骨干网络的推理性能。

## 研究背景与动机

### 领域现状

Prompting Decision Transformer (PDT) 将离线RL扩展到多任务设置，通过在输入序列前拼接轨迹prompt来区分不同任务，实现了无需额外训练即可在推理时适配新任务的能力。这类似于LLM中的prompting范式——通过调整输入来改变模型行为，而无需更新模型权重。

### 现有痛点

PDT的核心问题在于**prompt采样策略过于简单**：它从专家演示集中均匀采样trajectory segment来组成prompt，完全忽略了不同prompt的信息量差异。即使在完全可观测的MDP中，不同的prompt segment对任务识别的贡献也是不均等的。采样到低信息量的prompt会削弱PDT区分任务的能力，导致性能退化。

### 已有方案的局限

已有的PDT prompt调优方法存在三个关键缺陷：

**表达力受限**：Yuan等人(2024)用目标条件替代轨迹prompt，降低了表达能力

**适用范围窄**：Hu等人(2023,2024)的生成式方法不适用于离散设置，且不遵循prompt token间的因果关系

**扩展性差**：所有方法都将prompt视为扁平化的非结构化输入，直接在MDP模态上操作，导致复杂度随prompt大小和状态/动作空间呈组合爆炸增长

### 本文切入角度

核心idea是**利用prompt的结构性**：prompt由 $J$ 个trajectory segment组成，每个segment对应prompt中的一个位置。如果为每个位置维护一个独立的reward模型，就能将搜索空间从 $O(|P|^J)$ 降至 $O(J \cdot |P|)$。同时，利用预训练PDT本身作为segment的特征提取器，解决高维空间中reward建模的可扩展性问题。

## 方法详解

### 整体框架

方法在推理时工作：给定一个预训练好的冻结PDT模型 $\pi^*(\mathbf{x};\theta)$、目标任务的小量演示集 $\mathcal{P}_i$ 和模拟器 $\mathcal{M}_i$，通过多臂老虎机(MAB)框架迭代地从 $\mathcal{P}_i$ 中选择最优trajectory prompt，最大化下游任务性能。

每轮迭代中，bandit选择一个prompt $\rho_k$，PDT使用该prompt执行一轮rollout得到累积回报 $G_i^k$，该回报作为bandit的奖励信号，用于更新reward模型。

### 关键设计

1. **结构化Bandit架构**：将prompt的 $J$ 个segment位置解耦为 $J$ 个独立的contextual MAB问题。每个位置 $j$ 维护一个独立的reward模型 $\phi_j$，预测将某个segment $\tilde{\tau}$ 放在第 $j$ 个位置时PDT的期望性能。选择prompt时，每个模型对所有候选segment生成预测，取每个位置上得分最高的segment，形成预测矩阵 $\mathbf{Y} \in \mathbb{R}^{J \times |\mathcal{P}_i|}$。探索策略支持 $\epsilon$-greedy、UCB和Thompson Sampling。

2. **PDT特征提取器**：直接在未编码的MDP模态上学习reward模型的输入维度为 $H \times (|\mathcal{S}| + |\mathcal{A}| + 1)$，随状态/动作空间线性增长。本文提出利用预训练PDT对prompt token的隐层表示作为segment的嵌入 $\Psi: \tilde{\tau} \to \mathbb{R}^d$，得到固定维度的紧凑表示，使方法可在像素空间等高维环境中高效部署。

3. **Regret分析**：假设prompt的reward可分解为segment独立贡献之和加上一个有界交互项：

$$G(\rho) = \frac{1}{J}\sum_{j=1}^{J}\phi_j(\tilde{\tau}_j) + h(\tilde{\tau}_1, \ldots, \tilde{\tau}_J), \quad |h| \leq \varepsilon$$

在此假设下，累积regret的上界为：

$$\text{Regret}(K) \leq \frac{1}{J}\sum_{j=1}^{J}\text{Regret}_j(K) + 2K\varepsilon$$

当每个slot使用标准bandit算法时，总regret为 $\mathcal{O}(\sqrt{K\log|P|} + K\varepsilon)$，保持了次线性regret。

### 损失函数 / 训练策略

每个位置的reward模型 $\phi_j$ 独立训练，使用积累的 $\langle \tilde{\tau}_j^k, G_i^k \rangle$ 数据对，优化MSE损失：
$$\mathcal{L}(\phi_j) = \text{MSE}(\hat{y}_j, y)$$
其中 $\hat{y}_j = \phi_j(X_j)$ 是预测reward，$y$ 是实际rollout回报。

## 实验关键数据

### 主实验（In-Distribution）

| 环境 | 配置 | PDT (无调优) | Hill-climbing | ZORankSGD | TS (本文) | TS$^\Psi$ (本文) |
|------|------|-------------|--------------|-----------|----------|----------------|
| Half Cheetah | J=2,H=20 | -42.68 | -29.93 | -34.77 | **-26.28** | -27.62 |
| Ant | J=1,H=5 | 694.43 | 738.56 | 735.47 | **835.38** | 800.95 |
| Pick-place | J=1,H=5 | 551.58 | 555.79 | 554.26 | **556.11** | 556.87 |

### OOD泛化实验

| 环境 | 配置 | PDT (无调优) | PDT (微调) | TS (本文) | 提升 |
|------|------|-------------|-----------|----------|------|
| Half Cheetah | J=2,H=20 | -40.95 | -39.30 | **-26.28** | 35.8% |
| Ant | J=1,H=5 | 363.90 | 306.29 | **466.11** | +28.1% |
| Pick-place | J=2,H=2 | 524.37 | 488.17 | **553.34** | +5.5% |

### Sparse 2D环境实验

| 方法 | J=1 | J=2 | J=4 |
|------|-----|-----|-----|
| 最优策略 | 10.0 | 10.0 | 10.0 |
| PDT无调优 | 0.0±2.1 | 6.3±0.8 | 8.3±0.6 |
| TS (本文) | **9.9±0.0** | **9.9±0.0** | **9.8±0.1** |
| Hill-climbing | 5.8±3.8 | 7.9±1.6 | 6.2±4.0 |

### 关键发现

- Bandit调优在所有环境中一致且显著地提升冻结PDT性能，甚至在Ant环境中超过单任务CQL oracle
- 使用PDT编码特征（$\Psi$标记）的方法性能与未编码版本相当，证明PDT提供了紧凑有效的表示
- 微调PDT骨干网络反而可能降低性能，尤其在OOD场景中
- 注意力分析表明PDT主要关注最具信息量的单个segment，支持segment独立性假设

## 亮点与洞察

- **结构化分解是关键创新**：将组合优化问题拆解为线性规模的子问题，既有理论保证又有实践效果
- **PDT自身即是最佳特征提取器**：无需额外训练编码器，直接复用预训练模型的表示
- 方法纯粹在推理时工作，不需要更新Transformer权重，部署成本极低
- 即使演示数据中只有10%是专家数据，prompt调优也能恢复接近最优性能

## 局限与展望

- 需要环境模拟器进行在线rollout来获取bandit奖励，不适用于纯离线场景
- 当演示数量增加时，搜索空间仍然组合增长——可通过学习采样器预选高潜力segment来缓解
- segment独立性假设在某些任务中可能不完全成立
- 未探索与meta-learning（如In-Context RL）结合的可能性

## 相关工作与启发

- 与LLM prompt调优的联系：InstructZero用贝叶斯优化探索软prompt，INSTINCT用神经网络替代GP——本文类似地将优化从连续空间迁移到离散prompt空间
- Prompt Diffuser通过条件生成建模合成prompt，但不适用于离散设置
- 方法可推广到其他需要推理时选择输入模板的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 结构化bandit分解prompt空间的思路优雅有效
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖多环境、ID/OOD、数据质量消融、regret分析、注意力分析
- 写作质量: ⭐⭐⭐⭐ 动机清晰、理论推导严谨，附录详实
- 价值: ⭐⭐⭐⭐ 提供了实用的推理时适配方案，计算开销低

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[NeurIPS 2025\] Greedy Algorithm for Structured Bandits: A Sharp Characterization of Asymptotic Success / Failure](greedy_algorithm_for_structured_bandits_a_sharp_characterization_of_asymptotic_s.md)
- [\[NeurIPS 2025\] A Near-optimal, Scalable and Parallelizable Framework for Stochastic Bandits Robust to Adversarial Corruptions and Beyond](a_nearoptimal_scalable_and_parallelizable_framework_for_stoc.md)
- [\[NeurIPS 2025\] Emergent World Beliefs: Exploring Transformers in Stochastic Games](emergent_world_beliefs_exploring_transformers_in_stochastic_games.md)
- [\[NeurIPS 2025\] Blending Complementary Memory Systems in Hybrid Quadratic-Linear Transformers](blending_complementary_memory_systems_in_hybrid_quadratic-linear_transformers.md)

</div>

<!-- RELATED:END -->
