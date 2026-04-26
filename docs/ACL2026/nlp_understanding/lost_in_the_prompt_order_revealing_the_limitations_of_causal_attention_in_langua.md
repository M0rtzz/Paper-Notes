---
title: >-
  [论文解读] Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models
description: >-
  [ACL 2026][NLP理解][因果注意力] 本文深入研究了大语言模型在多选题问答中对提示组件顺序的敏感性，通过系统性实验排除了训练偏差和记忆衰退假说，揭示了因果注意力掩码是导致 QOC（问题-选项-上下文）顺序性能大幅下降的根本机制。
tags:
  - ACL 2026
  - NLP理解
  - 因果注意力
  - 提示顺序敏感性
  - 多选问答
  - 信息瓶颈
  - 机制解释
---

# Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models

**会议**: ACL 2026  
**arXiv**: [2601.14152](https://arxiv.org/abs/2601.14152)  
**代码**: 无  
**领域**: NLP Understanding / Prompt Sensitivity  
**关键词**: 因果注意力, 提示顺序敏感性, 多选问答, 信息瓶颈, 机制解释

## 一句话总结

本文深入研究了大语言模型在多选题问答中对提示组件顺序的敏感性，通过系统性实验排除了训练偏差和记忆衰退假说，揭示了因果注意力掩码是导致 QOC（问题-选项-上下文）顺序性能大幅下降的根本机制。

## 研究背景与动机

**领域现状**：大语言模型对提示结构的敏感性已被广泛报道——无论是 in-context learning 中示例的排列顺序，还是多选题中选项的排列方式，都可能导致模型性能的大幅波动。然而，目前的研究大多停留在现象描述层面，我们知道"什么"会影响模型表现，却不清楚"为什么"。

**现有痛点**：在多选题问答（MCQA）任务中，一个典型的提示由三部分组成：上下文段落（C）、问题（Q）和选项（O）。直觉上，重新排列这些组件不应影响性能，因为语义内容完全不变。但实验显示，将上下文放在问题和选项之前（CQO）的表现一致性地大幅优于反向排列（QOC），在 21 个 decoder-only 模型和 4 个数据集上平均差距超过 14 个百分点。

**核心矛盾**：语义等价的不同提示排列产生了巨大的性能差异，这对 LLM 的可靠性构成了严重挑战。先前工作如 lu2022fantastically 和 pezeshkpour2024 虽然报告了类似现象，但均未深入到架构层面寻找根因。

**本文目标**：提出三个竞争性假说并通过精心设计的对照实验逐一验证或排除，最终找到导致提示顺序敏感性的核心机制，并设计针对性干预方法验证结论。

**切入角度**：从架构层面出发，通过对比 decoder-only、encoder-only 和 encoder-decoder 三种架构的行为差异，定位问题根源。

## 方法详解

### 整体框架

本文采用"提出假说→设计实验→验证/排除"的研究范式。首先在 21 个 decoder-only LLM 上量化 CQO 与 QOC 的性能差距，然后提出三个假说：(1) 训练数据偏差、(2) 选项回忆失败、(3) 因果注意力机制，通过一系列控制实验逐步缩小原因范围，最终通过针对性干预实验提供因果证据。

### 关键设计

1. **假说 1 排除：训练数据偏差检验**

    - 功能：验证 CQO 格式是否因在训练数据中更常见而导致模型对 QOC 不熟悉
    - 核心思路：对比 9 对匹配的 base/instruct 模型的性能差距 $\Delta = \text{Acc}_{\text{CQO}} - \text{Acc}_{\text{QOC}}$，以及使用最多 5-shot ICL 让模型熟悉 QOC 格式
    - 设计动机：如果训练分布是主因，instruct 模型（接触更多 CQO 格式的指令数据）应该表现出更大差距，而 few-shot 应能显著缩小差距
    - 结果：base 与 instruct 模型差距几乎相同；5-shot 仅提升 QOC 3.1%，远不能弥合差距，排除此假说

2. **假说 2 排除：选项回忆测试**

    - 功能：检验 QOC 格式下模型是否因上下文过长而"遗忘"了中间的选项（类似 lost-in-the-middle 效应）
    - 核心思路：在给出提示后要求 LLM 精确回忆每个选项，测量精确匹配率
    - 设计动机：如果选项遗忘是主因，QOC 格式下选项回忆率应显著低于 CQO
    - 结果：QOC 的选项回忆准确率与 CQO 相当甚至更高，排除此假说

3. **假说 3 验证：因果注意力机制分析**

    - 功能：证明因果注意力掩码阻止了 QOC 格式中选项 token 对上下文 token 的注意力访问
    - 核心思路：通过三个子实验验证——(a) 架构对比：对比 decoder-only（因果注意力）、encoder-decoder（双向编码器）、encoder-only（双向注意力）三种架构；(b) 上下文移除测试：比较 QOC 与 QO（完全移除上下文）的性能；(c) 注意力与归因分析：追踪逐层注意力分布和 Gradient×Input 归因
    - 设计动机：如果因果掩码是根因，使用双向注意力的架构应不受影响，且移除上下文不应改变 QOC 性能
    - 结果：decoder-only 差距 14.72%，encoder-decoder 仅 2.30%，encoder-only 仅 0.02%；QOC 性能几乎等于 QO；CQO 中上下文归因为 0.797，QOC 中仅 0.335

### 调控因素与干预实验

作者发现两个调控因素：**上下文长度**（更长的上下文导致更大差距，如 RACE-H ~305 token 差距达 20.8%）和**答案位置**（靠前选项 A 差距 22.4%，靠后选项 D 仅 9.9%）。

基于因果注意力机制的解释，设计了四种针对性干预：

- **注意力剪枝**（降低 CQO）：设置 $\text{mask}[i,j] = -\infty$（$i \in \text{Options}, j \in \text{Context}$），CQO 准确率从 69.26% 降至 42.46%
- **激活补丁**（提升 QOC）：用 CQO 的选项隐藏状态替换 QOC 的 $h_{\text{opt}}^{\text{QOC}} \leftarrow h_{\text{opt}}^{\text{CQO}}$，QOC 提升 6.0 个点
- **选项重复 QOCO**：在上下文后重复选项，使新选项 token 能访问上下文，QOC 提升 8.2 个点
- **CoT 提示**：差距从 14.72 缩小至 7.47

## 实验关键数据

### 主实验

| 方法 | LogiQA | SciQ | RACE-M | RACE-H | 平均 |
|------|--------|------|--------|--------|------|
| CQO | 39.08 | 94.16 | 74.32 | 69.48 | 69.26 |
| QOC | 32.94 | 86.89 | 49.57 | 48.76 | 54.54 |
| 差距 Δ | 6.14 | 7.27 | 24.75 | 20.72 | 14.72 |

| 架构类型 | 代表模型 | 平均差距 Δ |
|----------|----------|-----------|
| Decoder-only | LLaMA/Qwen/Gemma | 14.72% |
| Encoder-decoder | Flan-T5 | 2.30% |
| Encoder-only | BERT/RoBERTa/ALBERT | 0.02% |

### 消融实验

| 干预方法 | 目标 | 效果 |
|----------|------|------|
| 注意力剪枝（CQO） | 降低 CQO | -26.8% |
| 激活补丁（QOC） | 提升 QOC | +6.0% |
| 选项重复 QOCO | 提升 QOC | +8.2% |
| CoT 提示 | 缩小差距 | 差距 14.72→7.47 |

### 关键发现

- 因果注意力掩码是导致提示顺序敏感性的根本机制，而非训练偏差或记忆衰退
- 在 QOC 中，选项 token 的隐藏状态在计算时完全无法接触上下文信息，互信息 $I(h_O^{\text{QOC}}; C | Q, O) = 0$
- 虽然最终答案 token 可以同时访问选项和上下文，但选项表示已经是"上下文盲"的，单步解码无法弥补
- 上下文越长、正确答案位置越靠前，因果掩码的负面影响越大

## 亮点与洞察

- **机制性而非描述性**：不同于先前工作仅报告现象，本文提供了清晰的因果机制解释，通过架构对比和干预实验给出了因果证据
- **单步瓶颈理论**：提出了优雅的"single-step bottleneck"解释——即使最终 token 能看到所有信息，选项已被编码为上下文无关的表示，一步解码无法弥补
- **实用价值**：选项重复（QOCO）和 CoT 作为简单的提示工程策略即可部分缓解问题，无需修改模型
- **信息论形式化**：附录中提供了严格的信息论推导，证明 QOC 下选项表示与上下文的互信息结构性为零

## 局限与展望

- 理论分析较为基础，仅建立了结构性独立性，未深入量化信息损失的具体大小
- 本文是诊断性研究，提出了推理时的缓解方案，但未探索训练时的根本修复方案
- 实验限于 0.5B-9B 参数的模型，更大规模模型上的表现有待验证
- CoT 虽然缩小差距但残留差距仍达 7.47%，说明推理时修复的局限性

## 相关工作与启发

- **vs Lost-in-the-Middle**：虽然都涉及长上下文信息利用问题，但本文通过选项回忆实验证明 QOC 的问题不是"遗忘"，而是结构性的注意力阻断
- **vs 选项排列敏感性研究**：先前工作关注选项排列对性能的影响，本文关注更宏观的组件排列，发现了更大的性能差距
- **vs 双向注意力模型**：encoder-only 模型几乎零差距的发现为"是否在预训练中引入部分双向注意力"提供了架构设计启发

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次从架构层面系统性解释提示顺序敏感性，假说驱动的研究范式清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 21 个模型、4 个数据集、3 种架构对比、4 种干预实验，极为充分
- 写作质量: ⭐⭐⭐⭐⭐ 论证逻辑严密，假说-实验-结论链条完整，图表设计直观
- 价值: ⭐⭐⭐⭐ 对 LLM 可靠性和提示工程有重要指导意义，为未来架构改进提供了方向

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Language Models and Logic Programs for Trustworthy Tax Reasoning](../../AAAI2026/nlp_understanding/language_models_and_logic_programs_for_trustworthy_tax_reasoning.md)
- [\[ACL 2025\] Generating Diverse Training Samples for Relation Extraction with Large Language Models](../../ACL2025/nlp_understanding/generating_diverse_training_samples_for_relation_extraction_with_large_language_.md)
- [\[ACL 2025\] Rethinking Semantic Parsing for Large Language Models: Enhancing LLM Performance with Semantic Hints](../../ACL2025/nlp_understanding/rethinking_semantic_parsing_for_large_language_models_enhancing_llm_performance_.md)
- [\[ACL 2025\] Dynamic Order Template Prediction for Generative Aspect-Based Sentiment Analysis](../../ACL2025/nlp_understanding/dot_absa_template.md)
- [\[AAAI 2026\] Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives](../../AAAI2026/nlp_understanding/understanding_syllogistic_reasoning_in_llms_from_formal_and_natural_language_per.md)

<!-- RELATED:END -->
