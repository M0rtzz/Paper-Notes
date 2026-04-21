---
title: >-
  [论文解读] Are Your LLMs Capable of Stable Reasoning?
description: >-
  [ACL 2025][LLM/NLP][G-Pass@k] 提出 G-Pass@k 评估指标和 LiveMathBench 动态基准，从"性能上限"和"稳定性"两个维度全面评估LLM的推理能力，揭示了当前LLM在推理一致性上存在巨大提升空间。
tags:
  - ACL 2025
  - LLM/NLP
  - G-Pass@k
  - 推理稳定性
  - LiveMathBench
  - 评估指标
  - 数学推理
---

# Are Your LLMs Capable of Stable Reasoning?

**会议**: ACL 2025  
**arXiv**: [2412.13147](https://arxiv.org/abs/2412.13147)  
**代码**: https://github.com/open-compass/GPassK (有)  
**领域**: LLM评估 / 数学推理  
**关键词**: G-Pass@k, 推理稳定性, LiveMathBench, 评估指标, 数学推理

## 一句话总结

提出 G-Pass@k 评估指标和 LiveMathBench 动态基准，从"性能上限"和"稳定性"两个维度全面评估LLM的推理能力，揭示了当前LLM在推理一致性上存在巨大提升空间。

## 研究背景与动机

当前LLM在数学推理等复杂任务上取得了显著进展，但**基准测试成绩与实际应用表现之间存在显著差距**。作者认为这一差距的根源在于现有评估协议和指标的不足——它们无法全面捕捉LLM的真实能力。

现有评估方法的核心矛盾在于：

**Greedy decoding 只看"最佳单次回答"**：Pass@1 只反映贪心解码下的单次表现，无法衡量模型是否能"稳定地"给出正确答案。一个模型可能偶尔答对，但在多次采样中表现极不一致。

**Pass@k 只看"至少一次正确"**：传统 Pass@k 衡量的是 $k$ 次采样中至少有一次正确的概率，它只反映模型的性能上限（potential），完全忽略了稳定性。

**静态基准存在数据泄露风险**：像 MATH、GSM8K 这样的经典基准已被广泛使用，模型可能在训练数据中见过类似题目，导致评估结果虚高。

核心idea：**一个真正可靠的推理模型不仅要"能答对"，更要"稳定地答对"**。因此需要一个能同时量化性能上限和稳定性的指标。

## 方法详解

### 整体框架

本文的贡献分为两部分：(1) G-Pass@k 评估指标；(2) LiveMathBench 动态基准。流程为：对每个问题采样 $n$ 次生成回答 → 统计正确次数 $c$ → 用 G-Pass@k 系列指标计算在不同稳定性要求下的表现。

### 关键设计

1. **G-Pass@k 基础指标**:

    - 功能：衡量从 $n$ 次采样中随机选 $k$ 次，全部正确的概率
    - 核心思路：利用超几何分布计算。设总采样数为 $n$，正确数为 $c$，则：
    $\text{G-Pass@}k = \mathbb{E}_{\text{Questions}}\left[\frac{\binom{c}{k}}{\binom{n}{k}}\right]$
    - 设计动机：当 $k=1$ 时退化为 Pass@1，当需要全部 $k$ 次都正确时能反映稳定性。但单个 $k$ 值仍不够全面。

2. **G-Pass@$k_\tau$ 带阈值的指标**:

    - 功能：衡量从 $k$ 次采样中至少有 $\lceil\tau \cdot k\rceil$ 次正确的概率
    - 核心思路：引入阈值参数 $\tau \in [0, 1]$ 控制"成功"的标准：
    $\text{G-Pass@}k_\tau = \mathbb{E}_{\text{Questions}}\left[\sum_{j=\lceil\tau \cdot k\rceil}^{c}\frac{\binom{c}{j}\cdot\binom{n-c}{k-j}}{\binom{n}{k}}\right]$
    - 设计动机：$\tau = 0$ 时等价于传统 Pass@k（至少1次正确），$\tau = 1$ 时要求全部正确。通过调节 $\tau$ 可以连续地在"性能上限"和"完美稳定"之间权衡。

3. **mG-Pass@k 综合指标**:

    - 功能：对 $\tau \in [0.5, 1.0]$ 区间的 G-Pass@$k_\tau$ 取积分均值，一个数字衡量整体表现
    - 核心思路：
    $\text{mG-Pass@}k = 2\int_{0.5}^{1.0}\text{G-Pass@}k_\tau \, d\tau = \frac{2}{k}\sum_{i=\lceil 0.5k\rceil+1}^{k}\text{G-Pass@}\frac{i}{k}$
    - 设计动机：选择 $[0.5, 1.0]$ 区间是因为该区间更能区分模型——在低阈值下大部分模型都能通过，区分度不大。mG-Pass@k 提供了一个简洁的单值指标。

4. **LiveMathBench 动态基准**:

    - 功能：提供持续更新的、有挑战性的数学竞赛题目，减少数据泄露
    - 核心思路：从当代数学竞赛中收集新题目，定期更新（如 v202412、v202505），确保题目是模型训练数据中未见过的
    - 设计动机：解决静态基准被"刷榜"和数据泄露的问题

### 评判方法

使用 Qwen2.5-72B-Instruct 作为 judge 模型评判答案正确性，同时也发布了轻量级的 LiveMath-Judge 模型降低评估成本。

## 实验关键数据

### 主实验（LiveMathBench-202412）

| 模型 | Pass@1 | G-Pass@16₀ | G-Pass@16₀.₅ | G-Pass@16₁ | mG-Pass@16 |
|------|--------|-----------|-------------|-----------|-----------|
| Llama-3.1-8B-Inst | 24.0 | 18.2 | 11.3 | 4.5 | 10.4 |
| Qwen2.5-7B-Inst | 37.0 | 36.5 | 27.2 | 16.0 | 25.8 |
| Claude-3.5-Sonnet | 46.7 | 44.1 | 36.2 | 26.6 | 35.3 |
| Qwen2.5-Math-7B-Inst | 68.4 | 44.1 | 38.3 | 28.1 | 36.6 |
| QwQ-32B-Preview | 72.7 | 74.9 | 65.8 | 40.1 | 61.2 |
| OpenAI-o1-mini | 74.1 | 76.3 | 67.3 | 48.3 | 64.8 |
| DeepSeek-R1 | 81.1 | 83.6 | 79.1 | 69.5 | 77.6 |
| OpenAI-o3-mini | 84.7 | 85.7 | 78.8 | 65.3 | 76.8 |

### 消融/对比（AIME2025）

| 模型 | Pass@1 | mG-Pass@16 | 说明 |
|------|--------|-----------|------|
| Qwen2.5-Math-72B-Inst | 13.3 | 13.3 | 数学专用模型稳定性反而高 |
| DeepSeek-Distill-Qwen-7B | 46.7 | 36.1 | 蒸馏模型Pass@1>mG稳定性缺口大 |
| DeepSeek-R1 | 66.7 | 42.5 | 最强模型仍有24%稳定性差距 |
| OpenAI-o3-mini | 53.3 | 43.6 | o1-like模型相对稳定 |

### 关键发现

- **Pass@1 与稳定性可能脱节**：Qwen2.5-Math-7B Pass@1 高达 68.4，但 mG-Pass@16 仅 36.6，说明高 Pass@1 不代表稳定推理
- **o1-like 推理模型（🏀）整体更稳定**：DeepSeek-R1、o3-mini 的 Pass@1 与 mG-Pass 差距较小
- **数学专用模型（🏐）稳定性不一定好**：可能过拟合到特定解题模式
- **在 Hard 子集上差异更大**：大部分模型在 LiveMathBench-Hard 上 G-Pass@16₁.₀ 接近 0

## 亮点与洞察

- G-Pass@k 的核心价值在于**用一条曲线代替一个点**来评估模型，给出从"最佳表现"到"最差表现"的完整画像
- mG-Pass@k 提供了一个实用的单值指标，可直接用于排行榜
- LiveMathBench 的动态更新机制有效对抗数据泄露，值得推广到其他领域
- 基于超几何分布的公式设计简洁优雅，无需额外拟合参数
- 开源了完整评估框架，支持 OpenCompass 和 LightEval

## 局限与展望

- 目前仅在数学推理上验证，未扩展到代码生成、逻辑推理等其他需要稳定性的任务
- LiveMathBench 题量有限（~100题级别），可能不够大规模
- 仅分析了现象（不稳定），未深入探究**为什么**模型不稳定的机制性原因
- $k$ 和 $n$ 的选择对结果有影响，论文未充分讨论最优设置
- 采样温度等推理超参数也会影响稳定性，但未系统研究其交互效应

## 相关工作与启发

- **vs Pass@k (Chen et al., 2021)**: Pass@k 只看"至少一次成功"，G-Pass@k 通过阈值 $\tau$ 实现了从宽松到严格的连续评估
- **vs Greedy/Majority Voting**: Greedy 看一次，Majority Voting 看多数，G-Pass@k 提供了更细粒度的分析
- **vs LiveCodeBench**: 类似地通过动态更新防数据泄露，但 G-Pass@k 在评估维度上更丰富

## 评分

- 新颖性: ⭐⭐⭐⭐ 指标设计虽简洁但动机清晰，是在评估维度上的重要补充
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖了大量模型和多个基准，数据量充足
- 写作质量: ⭐⭐⭐⭐ 表述清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ 揭示了"稳定性"这一被忽视的评估维度，对社区有参考意义

<!-- RELATED:START -->

## 相关论文

- [CER: Confidence Enhanced Reasoning in LLMs](cer_confidence_enhanced_reasoning.md)
- [Stepwise Reasoning Disruption Attack of LLMs](stepwise_reasoning_disruption_attack_of_llms.md)
- [Beware of Your Po! Measuring and Mitigating AI Safety Risks in Role-Play Fine-Tuning of LLMs](sarft_roleplay_safety.md)
- [IPO: Your Language Model is Secretly a Preference Classifier](ipo_your_language_model_is_secretly_a_preference_classifier.md)
- [Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs](../../NeurIPS2025/llm_nlp/triplets_better_than_pairs_towards_stable_and_effective_self-play_fine-tuning_fo.md)

<!-- RELATED:END -->
