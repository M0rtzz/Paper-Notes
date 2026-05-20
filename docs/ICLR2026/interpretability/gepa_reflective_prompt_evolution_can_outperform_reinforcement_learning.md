---
title: >-
  [论文解读] GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning
description: >-
  [ICLR 2026 (Oral)][可解释性][提示优化] 提出 GEPA（Genetic-Pareto）提示优化器，通过自然语言反思从少量执行轨迹中诊断问题并迭代优化提示，在六个任务上平均超越 GRPO 6%（最高20%），同时仅使用 1/35 的采样量。
tags:
  - "ICLR 2026 (Oral)"
  - "可解释性"
  - "提示优化"
  - "进化搜索"
  - "自然语言反思"
  - "Pareto前沿"
  - "GRPO替代"
---

# GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

**会议**: ICLR 2026 (Oral)  
**arXiv**: [2507.19457](https://arxiv.org/abs/2507.19457)  
**代码**: [https://github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa)  
**领域**: 可解释性  
**关键词**: 提示优化, 进化搜索, 自然语言反思, Pareto前沿, GRPO替代

## 一句话总结
提出 GEPA（Genetic-Pareto）提示优化器，通过自然语言反思从少量执行轨迹中诊断问题并迭代优化提示，在六个任务上平均超越 GRPO 6%（最高20%），同时仅使用 1/35 的采样量。

## 研究背景与动机
大语言模型越来越多地通过强化学习方法（如 GRPO）进行下游任务的适配。然而 GRPO 等方法通常需要数千次 rollout，将丰富的执行轨迹压缩为稀疏的标量奖励信号——这丢失了大量信息。

语言本身是一种高度可解释的媒介，天然包含了比标量奖励丰富得多的学习信号。一个 LLM 的推理链条、工具调用过程和错误信息中隐含了"为什么失败"的诊断线索，但 RL 方法将这些全部丢弃，仅保留一个分数。

**核心矛盾**：RL 方法（GRPO）需要大量 rollout 但仅利用标量奖励 vs 自然语言本身携带远比标量奖励丰富的学习信号。

**切入角度**：既然 LLM 能读懂执行轨迹，为什么不让 LLM 直接反思失败原因、提出改进，从而以极少的采样实现高效优化？

**核心idea**：将提示优化建模为带反思的进化搜索过程，利用 LLM 读取完整执行轨迹进行"梯度等价"的诊断和修复，通过 Pareto 前沿选择维持多样性。

## 方法详解
GEPA 是一个通用的文本参数优化框架，核心循环为：选择→执行→反思→变异→接受。

### 整体框架
给定任意包含一个或多个 LLM 提示的 AI 系统，GEPA 的优化流程是：
1. 从 Pareto 前沿选择一个候选提示
2. 在小批量上执行，捕获完整执行轨迹（推理过程、工具调用、输出、错误信息）
3. LLM 反思器读取轨迹，用自然语言诊断失败原因
4. 基于诊断结果和从所有祖先积累的教训，生成改进的候选提示
5. 如果改进则加入候选池，更新 Pareto 前沿

### 关键设计
1. **Actionable Side Information (ASI)**: 评估器返回的诊断反馈信息，是 GEPA 的核心创新。相当于文本优化中的"梯度"——不仅告诉你分数，还告诉你为什么。例如错误信息、性能剖析、推理日志等。这让 GEPA 能从极少的样本中提取大量学习信号。

2. **Pareto 前沿选择**: 维护一组在不同任务子集上表现最优的候选提示。这避免了单一指标优化导致的过拟合，保持了搜索空间的多样性。选择候选时从 Pareto 前沿采样，确保不遗漏在特定子任务上优秀的变体。

3. **系统感知合并（System-aware Merge）**: 将两个在不同任务上各有所长的 Pareto 最优候选进行优势互补合并。LLM 分析两者各自擅长的原因，然后生成一个融合双方优势的新候选。

4. **反思式变异**: 变异不是随机的，而是基于诊断的定向修改。LLM 先读取失败轨迹，诊断"为什么这个提示在这类问题上失败"，然后针对性地修改提示。这是GEPA效率远超RL的根本原因。

### 损失函数 / 训练策略
GEPA 不使用梯度或损失函数，而是基于评估 metric 的改善来决定是否接受新候选。典型配置使用 100-500 次评估（vs GRPO 的 5000-25000+ rollouts）。
接受准则可配置：默认为任何指标改善即接受，也可设置门槛或统计显著性要求。
整个优化过程是完全无梯度的——不需要访问模型权重，只需要 API 调用能力。
这意味着 GEPA 可以优化任何 API-only 的模型（GPT-5, Claude, Gemini），这是 RL 方法无法做到的。

## 实验关键数据

### 主实验

| 任务 | 指标 | GEPA | GRPO | MIPROv2 | 提升(vs GRPO) |
|------|------|------|------|---------|---------------|
| 6任务平均 | Accuracy | - | - | - | +6% avg, up to +20% |
| AIME-2025 | Accuracy | - | - | - | +12%(vs MIPROv2) |
| GPT-4.1 Mini+AIME | Accuracy | 56.6% | - | 46.6% | +10pp |
| DSPy MATH | Accuracy | 93% | - | 67% | - |
| ARC-AGI | Accuracy | 89% | - | 32% | - |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 完整 GEPA | 最佳 | 反思+Pareto+合并全部启用 |
| 无反思 | 性能下降显著 | 退化为随机搜索 |
| 无 Pareto 选择 | 多样性丧失 | 易陷入局部最优 |
| 无系统合并 | 中等下降 | 无法互补不同子任务优势 |

### 关键发现
- GEPA 使用的 rollout 数仅为 GRPO 的 1/35，但平均性能反而高 6%
- 在 AIME-2025 上超越领先的提示优化器 MIPROv2 达 12%
- 生成的优化提示是人类可读的，包含详细的问题解决策略
- 作为推理时搜索策略在代码优化上也展现了潜力
- 已被 DSPy、MLflow、OpenAI Cookbook、Google ADK、HuggingFace 等主流框架集成

## 亮点与洞察
- 用自然语言反思替代标量奖励，是对 RL 范式的深刻反思——语言本身就是最好的梯度
- 极低的样本需求（100-500次评估）使其可以优化API模型（GPT-5, Claude），无需权重访问
- 生成的提示是可解释的"预计算推理计划"，可直接审查和理解
- Pareto前沿维护是避免过拟合的优雅方案

## 局限与展望
- 依赖高质量的反思模型（通常需要 GPT-5 级别），成本不低
- 对于需要大规模权重更新的任务（如知识注入），提示优化的天花板有限
- 搜索过程的随机性可能导致不同运行结果差异较大
- 与 RL 的公平比较存在争议——优化的对象不同（提示vs权重）
- 对于超长提示（数千token），反思和变异的质量可能下降
- 评估 metric 的设计对最终效果影响巨大，metric 不好则 GEPA 也无法优化
- 在安全对齐等需要精确控制模型内部表示的任务上，提示优化的局限性更明显

## 相关工作与启发
- **vs GRPO/PPO**: GRPO通过策略梯度优化模型权重，需要大量rollout；GEPA优化提示文本，用反思替代梯度
- **vs MIPROv2**: 此前最强的提示优化器，GEPA通过ASI和Pareto搜索在AIME等任务上超越10%+
- **vs TextGrad**: TextGrad也用文本反馈但采用梯度模拟；GEPA的进化搜索+反思更为高效

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用自然语言反思替代RL标量奖励的范式极具启发性，ICLR Oral当之无愧
- 实验充分度: ⭐⭐⭐⭐ 六个任务验证，与GRPO和MIPROv2充分对比
- 写作质量: ⭐⭐⭐⭐⭐ 动机阐述极清晰，方法直觉易懂
- 价值: ⭐⭐⭐⭐⭐ 已获得大规模工业采用（Shopify/Databricks/OpenAI等），实际影响力极大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Evolution of Concepts in Language Model Pre-Training](evolution_of_concepts_in_language_model_pre-training.md)
- [\[ICML 2026\] Interpretability Can Be Actionable](../../ICML2026/interpretability/interpretability_can_be_actionable.md)
- [\[NeurIPS 2025\] Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning](../../NeurIPS2025/interpretability/beyond_accuracy_dissecting_mathematical_reasoning_for_llms_u.md)
- [\[ICLR 2026\] Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts](exploring_interpretability_for_visual_prompt_tuning_with_cross-layer_concepts.md)
- [\[ICLR 2026\] Auditing Cascading Risks in Multi-Agent Systems via Semantic–Geometric Co-evolution](auditing_cascading_risks_in_multi-agent_systems_via_semanti-geometric_co-evolut.md)

</div>

<!-- RELATED:END -->
