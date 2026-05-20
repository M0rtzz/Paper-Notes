---
title: >-
  [论文解读] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs
description: >-
  [ACL 2025][LLM/NLP][AMR] 系统评估LLM解释和利用抽象语义表示(AMR)的能力，发现AMR增强提示在长上下文任务（如对话摘要）中可显著提升性能（Llama3.1零样本余弦相似度从66%提升至76%），但在短上下文任务中通常会降低表现。
tags:
  - "ACL 2025"
  - "LLM/NLP"
  - "AMR"
  - "结构化语义表示"
  - "LLM"
  - "长上下文理解"
  - "零样本/少样本提示"
---

# Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs

**会议**: ACL 2025  
**arXiv**: [2504.04745](https://arxiv.org/abs/2504.04745)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: AMR, 结构化语义表示, LLM, 长上下文理解, 零样本/少样本提示

## 一句话总结
系统评估LLM解释和利用抽象语义表示(AMR)的能力，发现AMR增强提示在长上下文任务（如对话摘要）中可显著提升性能（Llama3.1零样本余弦相似度从66%提升至76%），但在短上下文任务中通常会降低表现。

## 研究背景与动机

**领域现状**：AMR等结构化语义表示在编码文本高层语义方面有优势，但此前的利用方式都需要修改模型架构（如图注意力机制）。

**现有痛点**：尚不清楚通用LLM是否能直接理解和利用线性化的AMR，无需架构修改。

**核心 idea**：直接将线性化AMR作为LLM提示的一部分，评估其在不同任务和上下文长度下对LLM性能的影响。

## 方法详解

### 整体框架
系统性评估多个LLM在多个NLP任务上对AMR的理解和利用能力。AMR通过IBM的transition-based neural parser从文本中提取（AMR3-structbart-L和doc-sen-conll-amr-seed42模型），线性化后直接作为LLM提示的一部分。所有实验在零样本、3-shot和5-shot三种设置下进行。

### 关键设计

1. **三种提示策略**:

    - Context-only（仅原文）：标准基线
    - AMR-augmented（原文+AMR）：在原文基础上附加其AMR表示，评估AMR是否能增强理解
    - AMR-only（仅AMR）：去除原文仅提供AMR，评估LLM能否仅从结构化表示中提取足够信息
    - 设计动机：区分AMR是作为辅助信息还是替代信息更有价值

2. **六个评估任务**:

    - AMR到文本重建：评估LLM对AMR的基本理解能力
    - 单跳QA (SQuAD 2.0)：短上下文推理
    - 多跳QA (HotpotQA)：长上下文推理（每个问题10个文档）
    - 对话摘要 (SAMSum)：长对话理解
    - NLI (SNLI)：短文本自然语言推理
    - 文档级NLI (DocNLI)：长文本自然语言推理
    - 设计动机：覆盖从短上下文到长上下文、从理解到推理的完整光谱

3. **三个模型**:

    - Llama3.1-8B-Instruct：最新最大
    - Phi3-mini-128k-instruct：中等规模
    - Mistral-7B-Instruct-v0.1：较早较小
    - 全部使用8-bit量化版本
    - 设计动机：观察模型规模和新旧程度对AMR利用能力的影响

### 损失函数 / 训练策略
主要实验为零样本和少样本推理，无需训练。额外进行了 Llama3.1 的 rank-32 LoRA 微调实验用于 SAMSum 摘要任务比较。

## 实验关键数据

### 主实验

| 任务 | 上下文 | 模型 | Context-only | AMR-augmented | AMR-only | 提升 |
|------|--------|------|-------------|---------------|----------|------|
| SAMSum | 长对话 | Llama3.1 0-shot | 66% cos | **76%** cos | 中等 | **+10%** |
| SAMSum | 长对话 | Llama3.1 3-shot | 高 | 略高 | 中 | 正面 |
| SQuAD | 短 | Llama3.1 3-shot | 59% F1 | 52% F1 | 48% F1 | **-7%** |
| AMR→Text | - | Llama3.1 5-shot | - | - | 81% cos | 基本理解 |
| SNLI | 短 | Phi3 0-shot | 27% F1 | **39%** F1 | 25% F1 | **+12%** |

### 消融实验

| 实验 | 结果 | 说明 |
|------|------|------|
| LoRA微调(SAMSum) | 75%→76% cos | AMR微调后略有提升但不如few-shot |
| 3-shot vs 5-shot | 边际递减 | 超过3个例子收益微弱 |
| HotpotQA(长上下文) | AMR无改善 | 因为每个文档的AMR短小，堆叠多个AMR不等于长上下文AMR |

### 关键发现
- **长上下文任务显著受益**：AMR将冗长的对话压缩为结构化语义图，帮助LLM保留关键信息点。SAMSum摘要零样本提升10%
- **短上下文任务通常有害**：AMR增加了输入长度但未提供额外有用信息，反而引入了解析噪声
- **更新更大的模型获益更多**：Llama3.1(8B, 最新)从AMR中获益最多，Mistral(7B, 最早)几乎无获益
- **LLM具备基本AMR理解能力**：81%的文本重建相似度证明LLM能从线性化AMR中恢复大部分原始语义
- **AMR-only在某些任务上表现合理**：3-shot SQuAD中AMR-only达48% F1，说明AMR确实编码了足够的推理信息
- **HotpotQA与SAMSum的差异揭示关键因素**：不是"长上下文"就有效，而是"单一长文档的AMR"有效；多个短文档AMR的堆叠并不等同

## 亮点与洞察
- 首次系统评估LLM对结构化语义表示的直接理解能力，无需任何架构修改
- "长上下文有益、短上下文有害"的发现为AMR的实际应用提供了清晰指导：仅在长对话/长文档场景中使用AMR增强
- AMR-only实验的合理表现暗示了一种潜在的数据压缩策略：用AMR替代原文以节省token数量

## 局限与展望
- 仅使用LoRA微调，未尝试全量微调，可能低估了微调的潜力
- AMR解析器本身可能引入错误，影响下游性能
- 未探索其他结构化表示（如知识图谱、DRS）的效果
- HotpotQA实验未使用Chain-of-Thought提示，可能影响结论

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统评估LLM+AMR
- 实验充分度: ⭐⭐⭐⭐ 多任务多模型多策略
- 写作质量: ⭐⭐⭐⭐ 实验设计清晰
- 价值: ⭐⭐⭐ 提供了有用的实验指导

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs](is_it_just_semantics_a_case_study_of_discourse_particle_understanding_in_llms.md)
- [\[ACL 2025\] LLM-Powered Test Case Generation for Detecting Bugs in Plausible Programs](llm_test_case_gen_bugs.md)
- [\[ACL 2025\] How LLMs Comprehend Temporal Meaning in Narratives: A Case Study in Cognitive Evaluation of LLMs](how_llms_comprehend_temporal_meaning_in_narratives_a_case_study_in_cognitive_eva.md)
- [\[ACL 2025\] Can Large Language Models Understand Internet Buzzwords Through User-Generated Content](buzzword_understanding_ugc.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)

</div>

<!-- RELATED:END -->
