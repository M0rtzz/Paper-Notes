---
title: >-
  [论文解读] Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study
description: >-
  [ACL 2026][强化学习][强化学习后训练] 首次系统研究 LLM 强化学习后训练的缩放行为，在 Qwen2.5 系列(0.5B-72B)上发现性能与训练资源之间遵循幂律关系，且学习效率随模型规模增大呈饱和趋势。 领域现状：预训练阶段的缩放定律(Scaling Laws)已被广泛研究——Kaplan et al. 和…
tags:
  - "ACL 2026"
  - "强化学习"
  - "强化学习后训练"
  - "缩放定律"
  - "数学推理"
  - "学习效率"
  - "数据复用"
---

# Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study

**会议**: ACL 2026  
**arXiv**: [2509.25300](https://arxiv.org/abs/2509.25300)  
**代码**: [GitHub](https://github.com/reasoning360/rl-scaling)  
**领域**: Reinforcement Learning / Scaling Laws  
**关键词**: 强化学习后训练, 缩放定律, 数学推理, 学习效率, 数据复用

## 一句话总结

首次系统研究 LLM 强化学习后训练的缩放行为，在 Qwen2.5 系列(0.5B-72B)上发现性能与训练资源之间遵循幂律关系，且学习效率随模型规模增大呈饱和趋势。

## 研究背景与动机

**领域现状**：预训练阶段的缩放定律(Scaling Laws)已被广泛研究——Kaplan et al. 和 Chinchilla 建立了 loss 与模型大小、数据量、计算量之间的幂律关系。然而，RL 后训练（如 GRPO、RLHF）已成为提升 LLM 推理能力的主流范式，但其缩放行为几乎完全未被系统探索。

**现有痛点**：实践者在进行 RL 后训练时缺乏指导：应该选择多大的模型？分配多少计算资源？数据量不足时是否应该重复使用数据？这些关键问题都没有定量答案，导致大量试错和资源浪费。

**核心矛盾**：预训练阶段的缩放定律是否适用于 RL 后训练？RL 后训练有其特殊性——使用可验证奖励而非交叉熵损失、采用 on-policy 采样而非 i.i.d. 数据——这些差异可能导致不同的缩放行为。

**本文目标**：通过大规模实验系统刻画 RL 后训练中模型规模、数据量、计算量与性能之间的关系，建立可预测的缩放公式。

**核心idea**：RL 后训练的 test loss 与资源消耗之间遵循 log-线性（幂律）关系 $\log L(N,X) = -k(N) \cdot \log X + E(N)$，其中学习效率 $k(N)$ 随模型规模增大而提高，但趋向饱和极限 $K_{\max}$。

## 方法详解

### 整体框架

本文是一项系统的实证研究，通过训练 63 个 LLM（覆盖 Qwen2.5 的 0.5B 到 72B 全系列，base 和 instruct 两种变体），在数学推理任务上使用 GRPO 算法进行 RL 后训练，在三种资源受限场景下测量性能并拟合缩放公式。

### 关键设计

**1. 幂律缩放公式：把 RL 后训练的性能-资源关系写成可外推的公式**

实践者最缺的是"投多少资源能换多少性能"的定量答案。本文发现 RL 后训练的 test loss 与资源消耗在对数空间上呈线性关系：$\log L(N,X) = -k(N) \cdot \log X + E(N)$，其中 $X$ 既可代入计算量 $C$ 也可代入数据量 $D$，$N$ 是模型参数量。关键在于斜率 $k(N)$——它衡量"每多投一份资源能压低多少 loss"，即学习效率。本文不把它当常数，而是用饱和函数建模 $k(N) = \frac{K_{\max}}{1+N_0/N}$：模型越大学习效率越高，但增益逐渐逼近上界 $K_{\max}$ 而非无限增长。这个饱和项正是与预训练缩放定律的关键区别——它捕捉了 RL 后训练特有的"规模红利递减"，也意味着无限堆参数并非最优策略。

**2. 模型间 / 模型内预测协议：用小模型和早期步骤外推大模型与后期轨迹**

缩放定律的价值在于"能不能预测没跑过的配置"。本文用两套协议来验证公式的外推能力：模型间预测用 0.5B–32B 拟合出的参数去预测 72B 的最终性能；模型内预测则用训练早期的若干步去预测后续的整条训练轨迹。两种协议下拟合优度都达到 $R^2 > 0.99$，说明公式既能跨模型规模、也能跨训练阶段做可靠外推——这正是缩放定律区别于"事后描述"的核心能力。

**3. 数据复用分析：在固定总训练量下回答"数据不够时能不能重复用"**

高质量推理数据稀缺是 RL 后训练的常见瓶颈，本文用受控实验直接量化复用的代价。做法是固定总数据量 $D_{\mathrm{total}}$，变化复用因子 $\tau$（满足 $D_{\mathrm{unique}} \times \tau = D_{\mathrm{total}}$），观察唯一样本数下降、重复次数上升时性能如何变化。结果是在 $\tau \leq 25$ 范围内性能对复用因子几乎不敏感——决定性能的是总训练量而非样本唯一性，直到 $\tau=100$ 才出现明显过拟合。这给数据稀缺场景一个直接结论：中等程度的复用近乎免费。

### 损失函数 / 训练策略

使用标准 GRPO 算法，二值奖励信号（正确=1，错误=0）。主要评估指标为 test loss $L = 1 - R/R_{\max}$。训练数据使用 guru-RL-92k 数学子集（约50k问题），按难度递增排序实现课程学习。每个配置重复3次确保鲁棒性。

## 实验关键数据

### 主实验

| 模型规模 | 计算效率 $k_C(N)$ | 数据效率 $k_D(N)$ | 说明 |
|---------|-------------------|-------------------|------|
| 0.5B-3B | 快速增长 | 快速增长 | 小模型阶段，规模效益显著 |
| 7B-32B | 增速放缓 | 增速放缓 | 效率增益开始饱和 |
| 72B | 接近 $K_{\max}$ | 接近 $K_{\max}$ | 饱和趋势明显 |

### 预测精度

| 预测类型 | $R^2$ | 说明 |
|---------|-------|------|
| 模型间(0.5B-32B → 72B) | >0.99 | 准确预测大模型性能 |
| 模型内(早期 → 后期) | >0.99 | 准确预测训练轨迹 |
| 跨架构(Llama 3) | >0.99 | 公式具有架构无关性 |

### 关键发现
- **更大的模型在计算和数据效率上始终更优**，但边际收益递减，32B 以上增益显著降低
- **数据复用高度有效**：$\tau \leq 25$ 时性能无显著退化，$\tau=100$ 时才出现过拟合
- **领域迁移有限**：RL 后训练在数学域内泛化良好，但对代码和逻辑推理等 OOD 任务几乎无收益，甚至可能损害某些能力
- **32B-72B 之间存在有趣的交叉**：在相同计算预算下，32B 在早期可能优于 72B（因为可以做更多步训练），揭示了模型规模与训练步数之间的权衡

## 亮点与洞察
- **RL 后训练版"Chinchilla"**：首次建立了类比预训练缩放定律的 RL 后训练缩放公式，填补了重要的理论空白
- **饱和趋势的发现**：学习效率的饱和意味着无限扩大模型规模并非 RL 后训练的最优策略，为资源分配提供了上界参考
- **数据复用的实践价值**：验证了中等程度的数据复用几乎无损性能，对数据稀缺场景具有直接的指导意义
- **跨架构验证**：在 Llama 3 系列上的验证增强了结论的普适性
- **领域迁移的警示**：RL 后训练的高度特化性（甚至可能损害其他能力）是一个重要的实践提醒

## 局限与展望
- 实验仅覆盖数学推理领域，多领域 RL 后训练的缩放行为未知
- 最大模型仅 72B，百亿级以上的饱和趋势无法实证验证
- 仅基于 GRPO 算法，其他 RL 算法（如 DAPO、PPO）是否呈现不同缩放行为有待探索
- 仅研究了密集模型，MoE 架构的 RL 后训练缩放行为未涵盖
- 缩放公式的绝对系数依赖于评估数据集和任务难度，难以普遍解释

## 相关工作与启发
- **vs Kaplan et al. (2020)**：经典预训练缩放定律聚焦于 cross-entropy loss，本文将缩放法则扩展到 RL 后训练的 reward 优化
- **vs Chinchilla (Hoffmann et al., 2022)**：Chinchilla 给出了计算最优的模型-数据比例，本文为 RL 后训练提供了类似的资源配置指南
- **vs Hilton et al. (2023)**：该工作在 CNN+RL 环境中发现幂律关系，本文在 LLM+GRPO 场景中验证了类似模式

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统研究 RL 后训练缩放行为，填补重要空白
- 实验充分度: ⭐⭐⭐⭐⭐ 63个模型、多规模、多架构、多场景的大规模实验，每个配置3次重复
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导严谨，可视化丰富
- 价值: ⭐⭐⭐⭐⭐ 为 RL 后训练的资源配置提供了定量指导，实践价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess](../../ICML2026/reinforcement_learning/how_reasoning_evolves_from_post-training_data_an_empirical_study_using_chess.md)
- [\[ACL 2026\] Why Does Reinforcement Learning Generalize? A Feature-Level Mechanistic Study of Post-Training in Large Language Models](why_does_reinforcement_learning_generalize_a_feature-level_mechanistic_study_of_.md)
- [\[ACL 2026\] Deliberative Searcher: Improving LLM Reliability via Reinforcement Learning with Constraints](deliberative_searcher_improving_llm_reliability_via_reinforcement_learning_with_.md)
- [\[ACL 2026\] LearnAlign: Data Selection for LLM Reinforcement Learning with Improved Gradient Alignment](learnalign_data_selection_for_llm_reinforcement_learning_with_improved_gradient_.md)
- [\[ACL 2026\] CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning](ce-gppo_coordinating_entropy_via_gradient-preserving_clipping_policy_optimizatio.md)

</div>

<!-- RELATED:END -->
