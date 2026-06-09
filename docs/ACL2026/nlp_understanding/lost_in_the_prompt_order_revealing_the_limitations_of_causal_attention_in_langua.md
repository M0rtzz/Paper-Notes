---
title: >-
  [论文解读] Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models
description: >-
  [ACL 2026][NLP理解][因果注意力] 本文深入研究了大语言模型在多选题问答中对提示组件顺序的敏感性，通过系统性实验排除了训练偏差和记忆衰退假说，揭示了因果注意力掩码是导致 QOC（问题-选项-上下文）顺序性能大幅下降的根本机制。
tags:
  - "ACL 2026"
  - "NLP理解"
  - "因果注意力"
  - "提示顺序敏感性"
  - "多选问答"
  - "信息瓶颈"
  - "机制解释"
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

本文要回答的不是"什么会影响 LLM"，而是"为什么"——为什么语义完全等价、只是把上下文（C）、问题（Q）、选项（O）换个顺序，CQO 就一致性地大幅优于 QOC。研究范式是"提出假说 → 设计对照实验 → 验证/排除"：先在 21 个 decoder-only 模型、4 个数据集上量化出 CQO 与 QOC 平均超过 14 个百分点的差距，再提出训练偏差、选项回忆失败、因果注意力三个竞争假说，用控制实验逐一筛掉前两个、坐实第三个，最后用针对性干预实验给出因果证据。

### 关键设计

**1. 排除假说一——训练数据偏差：CQO 的优势不是"见得多"**

第一个直觉解释是 CQO 在训练数据里更常见、模型对 QOC 不熟。若如此，接触更多 CQO 式指令数据的 instruct 模型应比 base 模型差距更大，且 few-shot 让模型见识 QOC 后差距应明显缩小。作者对比 9 对匹配的 base/instruct 模型的差距 $\Delta = \text{Acc}_{\text{CQO}} - \text{Acc}_{\text{QOC}}$，并加最多 5-shot ICL。结果两类模型差距几乎一样、5-shot 只把 QOC 抬高 3.1% 远不能弥合——训练分布不是主因。

**2. 排除假说二——选项回忆失败：QOC 不是"忘了选项"**

第二个解释是 QOC 下上下文太长、把夹在中间的选项"挤忘了"，类似 lost-in-the-middle。若成立，QOC 的选项回忆率应明显低于 CQO。作者在给出提示后要求模型逐字回忆每个选项、测精确匹配率，发现 QOC 的回忆准确率与 CQO 相当甚至更高——问题不在记忆，而在别处。

**3. 验证假说三——因果注意力机制：选项被编码成了"上下文盲"的表示**

真正的根因是因果注意力掩码：在 QOC 里选项 token 排在上下文之前，掩码使它在被编码时根本无法注意到后面的上下文。作者用三个子实验锁定它：(a) 架构对比——decoder-only（因果注意力）差距 14.72%，encoder-decoder（双向编码器）只 2.30%，encoder-only（双向注意力）仅 0.02%，差距随注意力是否双向而消失；(b) 上下文移除——QOC 的表现几乎等于直接删掉上下文的 QO，说明那段上下文对选项编码"形同虚设"；(c) 注意力与 Gradient×Input 归因——CQO 中上下文归因 0.797，QOC 中仅 0.335。信息论上，QOC 下选项隐藏态与上下文的互信息结构性为零 $I(h_O^{\text{QOC}}; C \mid Q, O) = 0$：尽管最终答案 token 能同时看到选项和上下文，但选项表示早已成型且"上下文盲"，单步解码补不回来。

**4. 调控因素与干预实验：顺着机制既能放大也能修复差距**

机制解释还预测了两个调控因素并被证实：上下文越长差距越大（RACE-H ~305 token 时差距 20.8%），正确答案越靠前差距越大（选项 A 差距 22.4%、选项 D 仅 9.9%）。据此设计四种干预反向验证因果链：**注意力剪枝**把 CQO 里选项对上下文的注意力强行屏蔽 $\text{mask}[i,j] = -\infty\ (i \in \text{Options},\ j \in \text{Context})$，CQO 从 69.26% 砸到 42.46%；**激活补丁**用 CQO 的选项隐藏态替换 QOC 的 $h_{\text{opt}}^{\text{QOC}} \leftarrow h_{\text{opt}}^{\text{CQO}}$，QOC 涨 6.0 个点；**选项重复 QOCO** 在上下文后再贴一遍选项、让新选项 token 能看到上下文，QOC 涨 8.2 个点；**CoT 提示**把差距从 14.72 缩到 7.47。能精准地一手压低 CQO、一手抬高 QOC，正说明因果掩码就是那个开关。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] The Imperfective Paradox in Large Language Models](the_imperfective_paradox_in_large_language_models.md)
- [\[ACL 2026\] AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models](adaptime_enabling_adaptive_temporal_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)
- [\[NeurIPS 2025\] Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](../../NeurIPS2025/nlp_understanding/generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)
- [\[AAAI 2026\] Language Models and Logic Programs for Trustworthy Tax Reasoning](../../AAAI2026/nlp_understanding/language_models_and_logic_programs_for_trustworthy_tax_reasoning.md)

</div>

<!-- RELATED:END -->
