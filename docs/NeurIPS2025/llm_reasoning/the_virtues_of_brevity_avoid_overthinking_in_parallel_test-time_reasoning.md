---
title: >-
  [论文解读] The Virtues of Brevity: Avoid Overthinking in Parallel Test-Time Reasoning
description: >-
  [NeurIPS 2025 Workshop on Efficient Reasoning][LLM推理][过度思考] 证明在推理模型的 Best-of-N 采样中，选择最短解是一个简单、反直觉但高效的启发式方法，性能与 self-consistency 相当，token 成本显著更低，其原理在于利用了模型在"常规模式"与"过度思考模式"之间的系统性偏差。
tags:
  - NeurIPS 2025 Workshop on Efficient Reasoning
  - LLM推理
  - 过度思考
  - 最短答案启发式
  - Best-of-N
  - 自一致性
  - 推理模型
---

# The Virtues of Brevity: Avoid Overthinking in Parallel Test-Time Reasoning

**会议**: NeurIPS 2025 Workshop on Efficient Reasoning  
**arXiv**: [2510.21067](https://arxiv.org/abs/2510.21067)  
**代码**: 无  
**领域**: LLM推理 / 测试时计算  
**关键词**: 过度思考, 最短答案启发式, Best-of-N, 自一致性, 推理模型

## 一句话总结

证明在推理模型的 Best-of-N 采样中，选择最短解是一个简单、反直觉但高效的启发式方法，性能与 self-consistency 相当，token 成本显著更低，其原理在于利用了模型在"常规模式"与"过度思考模式"之间的系统性偏差。

## 研究背景与动机

**领域现状**：推理模型（如 DeepSeek-R1、Grok-3-mini）通过长链思维（CoT）在数学和代码等复杂任务上显著提升了性能。并行测试时计算（Best-of-N）通过采样 $N$ 个解并选最优进一步提升准确率。self-consistency 是最常用的启发式，选择出现最频繁的答案。

**现有痛点**：self-consistency 需要至少 $N \geq 3$ 个解才能形成多数票；对于代码生成等输出不可直接比较的任务不适用；所有 $N$ 个解都需完整生成，token 开销大。此外，现有研究发现推理模型存在"过度思考"现象——在简单问题上生成过多不必要的 token，浪费计算资源。

**核心矛盾**：如何在不依赖复杂评分机制或额外奖励模型的前提下，高效地从多个候选解中选出高质量答案？现有方法要么需要答案可比较（self-consistency），要么需要训练专用验证器（reward model），复杂度和计算成本高。

**本文目标** (1) 提供一个更简单、更通用的 Best-of-N 选择启发式；(2) 解释为什么最短解往往是正确解；(3) 降低并行推理的 token 成本。

**切入角度**：推理模型在 RL 训练中形成了一种隐性策略——当模型对解的正确性信心不足时，会通过"注水推理"稀释负奖励（因为标准 RL 算法按 token 归一化奖励），导致错误解和不确定解系统性地更长。选最短解就是在选最自信的解。

**核心 idea**：推理模型中正确解倾向于更短，选最短解就能避开过度思考的尾部分布，实现 Pareto 改进。

## 方法详解

### 整体框架

方法极其简单：对同一问题并行采样 $N$ 个解（$N=5$），选择 token 数最少的那个作为最终答案。在并行推理场景下，一旦第一个解完成，终止所有尚未完成的候选解（因为它们必然更长），实现 early stopping 节省 token。

### 关键设计

1. **双模式假说（Two-Regime Hypothesis）**:

    - 功能：解释为什么最短解启发式有效
    - 核心思路：推理模型在生成解时实际运行在两种模式中——"常规模式"（conventional regime）：模型有信心，生成紧凑、直接的推理链，解倾向于正确；"过度思考模式"（overthinking regime）：模型不确定，通过重复推理、自我纠正、犹豫表达等拉长输出，解倾向于错误。token 数的概率分布呈现双峰或右偏重尾——常规模式解集中在较短区域，过度思考模式形成长尾
    - 设计动机：这个假说统一解释了多个观察：(1) 正确解比错误解平均更短；(2) 长解的不确定性标记密度更高；(3) 超过临界点后嵌入距离不再增长

2. **临界点分析（Critical Point Analysis）**:

    - 功能：推断过度思考模式开始主导的位置
    - 核心思路：将所有解按 token 数排列，找到 token 数分布的众数（mode）作为临界点——这是常规模式解分布的峰值，也是过度思考比例开始显著上升的拐点。在临界点前后分别分析不确定性标记频率和嵌入距离趋势，观察到明显的趋势断裂：临界点前不确定性随长度正相关增长，临界点后趋势打破
    - 设计动机：临界点分析为双模式假说提供了可量化的实证支持

3. **Early Stopping 的 Pareto 改进**:

    - 功能：在并行推理中进一步节省计算
    - 核心思路：在同步 token 生成的并行场景下，最短解完成时立即终止其他候选。相比 self-consistency 必须等所有 $N$ 个解完成，最短解启发式只需等最快完成的那个。在 $N=2$ 时就能产生区分（self-consistency 需要 $N \geq 3$），适合成本敏感场景
    - 设计动机：Pareto 曲线分析清晰展示了在相同 token 预算下，最短解启发式比 self-consistency 获得更高或相当的准确率

## 实验关键数据

### 主实验（N=5, 400 道 AIME 数学题 + LiveCodeBench v5）

| 模型 | 方法 | AIME 准确率 | LiveCodeBench 准确率 |
|------|------|------------|---------------------|
| DeepSeek-R1 | 单次采样均值 | 85.0% | 76.5% |
| DeepSeek-R1 | 最短解 | **89.0%** | **79.2%** |
| DeepSeek-R1 | Self-consistency | 89.2% | *不适用* |
| DeepSeek-R1 | 最长解 | 78.2% | 76.5% |
| Qwen3-32B | 单次采样均值 | 89.5% | 78.6% |
| Qwen3-32B | 最短解 | **92.5%** | **79.5%** |
| Qwen3-32B | Self-consistency | 93.0% | *不适用* |
| Qwen3-32B | 最长解 | 85.5% | 76.8% |

### 消融分析（不确定性标记密度）

| 模型 | 长解比短解不确定性更高的比例 (AIME) | LiveCodeBench |
|------|----------------------------------|---------------|
| DeepSeek-R1 | 67.0% | 67.5% |
| Grok-3-mini | 67.4% | 63.7% |
| Qwen3-32B | 58.2% | 65.8% |

### 关键发现

- **最短解 ≈ self-consistency，但计算成本大幅降低**：在 AIME 上差距不到 1%，在 LiveCodeBench 上 self-consistency 因输出不可比较而不适用，最短解仍有效
- **选最长解比单次采样还差**：进一步验证了过度思考假说——长解系统性地更不准确
- **$N=2$ 时即有显著提升**：最短解在一对候选中就能区分，而 self-consistency 至少需 $N=3$
- **临界点后的趋势断裂**：不确定性标记密度和嵌入距离在众数附近出现明确的趋势变化，支持双模式假说
- **最短解相比最长解的 token 分布更集中**：峰值位置相同但少了长尾，说明启发式通过截断过度思考尾部起作用

## 亮点与洞察

- **极简主义方法论的胜利**：整个方法一行代码就能实现（`argmin(lengths)`），却与复杂的 self-consistency 性能相当，且适用范围更广（代码生成等不可比较输出）。这体现了"好的启发式来源于对问题结构的深刻理解"
- **过度思考的训练根因分析独到**：将过度思考归因于 GRPO/PPO 中按 token 归一化负奖励——模型学会了在不确定时"注水"以稀释惩罚。这个机制性解释比简单地说"模型生成了冗余token"更深刻
- **Early stopping 的实践价值**：在并行推理部署中，最短解启发式天然支持提前终止，对于 latency-sensitive 的应用（如实时编码助手）有直接的工程价值

## 局限与展望

- 仅在数学（AIME）和代码（LiveCodeBench）上验证，未测试自然语言推理、常识推理等任务，这些任务中长度与正确性的关系可能不同
- 假设同步并行生成（所有候选同时开始），在异步场景下 early stopping 机制需要调整
- 临界点的确定依赖于众数估计，对于分布不规则的情况可能不准确
- 未探索最短解与其他选择策略（如 reward model）的组合使用
- Workshop 论文篇幅所限，消融实验不够详尽（如不同 temperature、不同 $N$ 值的系统分析）

## 相关工作与启发

- **vs Self-consistency (Wang et al. 2023)**: Self-consistency 通过多数票选答案，需要可比较的输出且至少 3 个样本；最短解更通用且 $N=2$ 即可
- **vs Chen et al. 2025 ("Don't Think That Much")**: 聚焦于简单问题上的过度思考浪费；本文补充了困难问题上过度思考导致的系统性长度-正确性偏差
- **vs Reward Model 方法 (Zhang et al. 2025)**: Gen-RM 等需要训练专门的验证器；最短解完全不需要额外模型

## 评分

- 新颖性: ⭐⭐⭐⭐ 反直觉的简单启发式 + 过度思考双模式假说的理论解释
- 实验充分度: ⭐⭐⭐ Workshop 论文，任务类型和消融有限
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，Pareto 曲线图非常直观
- 价值: ⭐⭐⭐⭐ 对推理模型部署有直接实用价值，启发 RL 训练中的奖励设计改进

<!-- RELATED:START -->

## 相关论文

- [Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models](does_thinking_more_always_help_mirage_of_test-time_scaling_in_reasoning_models.md)
- [Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)
- [LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling](limopro_reasoning_refinement_for_efficient_and_effective_test-time_scaling.md)
- [Provable Scaling Laws for the Test-Time Compute of Large Language Models](provable_scaling_laws_for_the_testtime_compute_of_large_lang.md)
- [Let LRMs Break Free from Overthinking via Self-Braking Tuning](let_lrms_break_free_from_overthinking_via_self-braking_tuning.md)

<!-- RELATED:END -->
