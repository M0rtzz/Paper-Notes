---
title: >-
  [论文解读] What Factors Affect LLMs and RLLMs in Financial Question Answering?
description: >-
  [ACL 2026][多语言/翻译][金融问答] 本文系统研究了提示方法、Agent 框架和多语言对齐方法对 LLM 和 RLLM（推理型大模型）在金融问答任务上的影响，发现现有方法本质上是通过模拟 Long CoT 来提升 LLM 性能，但对已具备 Long CoT 能力的 RLLM 效果有限。
tags:
  - "ACL 2026"
  - "多语言/翻译"
  - "金融问答"
  - "推理型大模型"
  - "Long CoT"
  - "提示方法"
  - "多语言对齐"
---

# What Factors Affect LLMs and RLLMs in Financial Question Answering?

**会议**: ACL 2026  
**arXiv**: [2507.08339](https://arxiv.org/abs/2507.08339)  
**代码**: [https://github.com/WPENGxs/LLM_RLLM_financial_analysis](https://github.com/WPENGxs/LLM_RLLM_financial_analysis)  
**领域**: 多语言/金融NLP  
**关键词**: 金融问答, 推理型大模型, Long CoT, 提示方法, 多语言对齐

## 一句话总结

本文系统研究了提示方法、Agent 框架和多语言对齐方法对 LLM 和 RLLM（推理型大模型）在金融问答任务上的影响，发现现有方法本质上是通过模拟 Long CoT 来提升 LLM 性能，但对已具备 Long CoT 能力的 RLLM 效果有限。

## 研究背景与动机

**领域现状**：大语言模型已在金融问答领域取得显著进展，研究者通过提示方法（如 CoT）、Agent 框架和多语言对齐等手段来提升 LLM 的金融推理能力。同时，推理型大模型（RLLM，如 DeepSeek-R1、O4-mini）通过 Long CoT 显著增强了复杂问题的推理能力。

**现有痛点**：尽管各种增强方法层出不穷，但缺乏系统性研究来探明哪些方法能真正释放 LLM 和 RLLM 在金融领域的潜力，尤其是在 RLLM 出现后，传统增强方法是否仍然有效尚不清楚。

**核心矛盾**：现有的提示方法和 Agent 框架主要通过延长推理链来提升性能，这与 RLLM 自带的 Long CoT 能力形成冗余，导致对 RLLM 的增益极为有限甚至产生负面效果。

**本文目标**：使用 5 个 LLM 和 4 个 RLLM，系统评估提示方法、Agent 框架和多语言对齐方法在金融问答任务上的影响。

**切入角度**：从"长推理链是性能提升的关键瓶颈"这一假设出发，通过对比 LLM 和 RLLM 在相同方法下的表现差异来验证假设。

**核心 idea**：当前提升 LLM 金融问答性能的有效方法本质上都是在模拟 Long CoT，而 RLLM 已天然具备此能力，因此传统方法对 RLLM 边际收益递减。

## 方法详解

### 整体框架

本文是一项系统性实证研究，不提出新方法，而是在 FAMMA 金融问答基准上测试 9 个模型 × 7 种方法的组合。评估涵盖三个维度：提示方法（Direct、Zero-shot CoT、Plan-and-Solve）、Agent 框架（Self-Refine、S3 Agent）和多语言对齐方法（Direct、Translate-en、Cross-lingual Prompting）。

### 关键设计

1. **提示方法对比**:

    - 功能：评估不同提示策略对 LLM/RLLM 金融推理的影响
    - 核心思路：选取三种代表性提示方法——Direct（直接输入）、Zero-shot CoT（"let's think step by step"）和 Plan-and-Solve（先理解问题再制定计划分步求解）。Plan-and-Solve 在大多数 LLM 上表现最优，但在 RLLM 上反而可能降低性能。
    - 设计动机：验证提示方法的增益来源是否为模拟 Long CoT，以及这种模拟对已有 Long CoT 能力的 RLLM 是否冗余。

2. **Agent 框架对比**:

    - 功能：评估多 Agent 协作对 LLM/RLLM 的增益
    - 核心思路：测试 Self-Refine（LLM 对自身输出反馈迭代优化，仅 1 轮）和 S3 Agent（从表层表达、语义信息、情感表达三个视角协作推理）。较小的 LLM（如 Llama-3.1-8B）从 Agent 框架获益更大，而大型 LLM 和 RLLM 的增益有限。
    - 设计动机：探索 Agent 框架是否能通过结构化协作弥补 LLM 的推理不足，以及对 RLLM 是否仍有价值。

3. **多语言对齐方法对比**:

    - 功能：评估多语言方法对中文和法语金融问答的提升效果
    - 核心思路：比较 Direct（英文提示+本地语言问题）、Translate-en（翻译为英文后回答）和 Cross-lingual Prompting（CLP，跨语言对齐提示+任务求解器两阶段）。CLP 对 LLM 效果最好（平均提升 4-5%），但对 RLLM 效果有限甚至为负。
    - 设计动机：验证多语言对齐的增益是否同样来自延长推理链，以及 RLLM 是否已通过 Long CoT 实现了自对齐。

### 损失函数 / 训练策略

本文为纯评估研究，不涉及训练。所有模型使用推理模式，开放题由 GPT-4o-mini 基于标准答案评分。

## 实验关键数据

### 主实验

| 模型 | 方法 | Overall Acc | 相比 Direct 提升 |
|--------|------|------|----------|
| DeepSeek-V3 (LLM) | Direct | 58.86 | - |
| DeepSeek-V3 (LLM) | Plan-and-Solve | 58.81 | -0.05 |
| DeepSeek-V3 (LLM) | S3 Agent | 56.81 | -2.05 |
| DeepSeek-R1-Distill-32B (RLLM) | Direct | 53.41 | - |
| DeepSeek-R1-Distill-32B (RLLM) | S3 Agent | 54.29 | +0.88 |
| O4-mini (RLLM) | Direct | 65.29 | - |
| O4-mini (RLLM) | Zero-shot CoT | 66.52 | +1.23 |
| Llama-3.1-8B (LLM) | Direct | 16.50 | - |
| Llama-3.1-8B (LLM) | S3 Agent | 24.62 | +8.12 |

### 消融实验

| 配置 | Qwen-2.5-32B | R1-Distill-32B | 说明 |
|------|---------|------|------|
| Direct | 44.88 | 53.41 | R1蒸馏后平均提升7.4% |
| Zero-shot CoT | 46.11 | 53.62 | 提示方法对 RLLM 增益微弱 |
| Plan-and-Solve | 44.06 | 53.26 | Plan-and-Solve 对 RLLM 甚至降低 |
| Self-Refine | 45.19 | 47.96 | Self-Refine 对 RLLM 大幅降低 |
| S3 Agent | 45.34 | 54.29 | Agent 协作对 RLLM 有一定增益 |

### 关键发现

- **小模型从 Agent 框架获益最大**：Llama-3.1-8B 使用 S3 Agent 后性能从 16.50% 提升到 24.62%（+49%），但大模型 DeepSeek-V3 反而下降。
- **Long CoT 是核心瓶颈**：LLM 的有效方法本质上都在模拟 Long CoT；输出 token 数与性能正相关（表 3）。RLLM 平均输出约 2000 tokens，而 LLM 仅 250-470 tokens。
- **RLLM 的自对齐能力**：RLLM 在多语言场景下通过 Long CoT 自动实现跨语言推理，无需额外的多语言对齐方法。
- **过度思考问题**：RLLM 在简单题上生成过多 token 但并未带来性能提升，存在明显的 overthinking 现象。
- **Scaling Law 仍然成立**：Qwen-3 系列从 0.6B 到 32B，参数越大性能越好，输出也越长。开启思考模式后平均提升 16.9%。

## 亮点与洞察

- **LLM vs RLLM 的系统对比**：首次在金融问答场景下系统对比了提示方法、Agent 框架和多语言方法对 LLM 和 RLLM 的差异化影响，揭示了 Long CoT 作为统一解释框架的重要性。
- **方法论启示**：对 LLM 来说应投入更多精力设计能延长推理链的方法；对 RLLM 来说，应转向更复杂的 Agent 机制来规范输出，而非简单延长思考。
- **动态 CoT 长度控制**：针对 RLLM 的 overthinking 问题，根据问题复杂度动态调整 CoT 长度将是重要研究方向。

## 局限与展望

- 所有模型仅运行一次，缺乏多次运行的统计显著性检验。
- 仅使用 FAMMA 的文本子集，未涉及多模态金融问答。
- Agent 框架（Self-Refine 仅 1 轮迭代）的探索较浅，未测试更复杂的多轮 Agent 系统。
- 未探索专门为 RLLM 设计的增强方法。

## 相关工作与启发

- **vs BloombergGPT**: BloombergGPT 训练了 500 亿参数的金融专用 LLM，本文则从推理策略角度探索通用 LLM 的金融能力释放。
- **vs FinBen**: FinBen 是综合金融基准，本文使用 FAMMA 但聚焦于方法对比而非模型排名。

## 评分

- 新颖性: ⭐⭐⭐ 研究视角有价值但不提出新方法，属于实证调查
- 实验充分度: ⭐⭐⭐⭐ 9个模型7种方法的大规模对比，数据量充足
- 写作质量: ⭐⭐⭐⭐ 分析清晰，发现总结到位
- 价值: ⭐⭐⭐⭐ 为金融 NLP 社区选择 LLM/RLLM 策略提供了实用指导

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] AskQE: Question Answering as Automatic Evaluation for Machine Translation](../../ACL2025/multilingual_mt/askqe_question_answering_as_automatic_evaluation_for_machine_translation.md)
- [\[ACL 2025\] MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering](../../ACL2025/multilingual_mt/mtvqa_benchmarking_multilingual_text-centric_visual_question_answering.md)
- [\[ACL 2026\] Cross-Cultural Transfer of Emoji Semantics and Sentiment in Financial Social Media](cross-cultural_transfer_of_emoji_semantics_and_sentiment_in_financial_social_med.md)
- [\[ACL 2026\] Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic](vocab_diet_reshaping_the_vocabulary_of_llms_via_vector_arithmetic.md)
- [\[ACL 2026\] NiuTrans.LMT: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs](niutranslmt_toward_inclusive_and_scalable_multilingual_machine_translation_with_.md)

</div>

<!-- RELATED:END -->
