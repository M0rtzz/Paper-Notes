---
title: >-
  [论文解读] On Synthesizing Data for Context Attribution in Question Answering
description: >-
  [ACL 2025][NLP理解][上下文归因] 本文提出 SynQA，一种基于"给定上下文句子→生成 QA 对"的合成数据策略，用于训练小模型完成上下文归因任务（即为 QA 系统的回答找到支撑证据句），在多个 QA 任务和跨域场景中显著优于零样本推理和 LLM 集成方法。
tags:
  - ACL 2025
  - NLP理解
  - 上下文归因
  - 问答系统
  - 合成数据
  - LLM微调
  - 幻觉检测
---

# On Synthesizing Data for Context Attribution in Question Answering

**会议**: ACL 2025  
**arXiv**: [2504.05317](https://arxiv.org/abs/2504.05317)  
**代码**: 无  
**领域**: NLP理解 / 信息检索  
**关键词**: 上下文归因、问答系统、合成数据、LLM微调、幻觉检测

## 一句话总结

本文提出 SynQA，一种基于"给定上下文句子→生成 QA 对"的合成数据策略，用于训练小模型完成上下文归因任务（即为 QA 系统的回答找到支撑证据句），在多个 QA 任务和跨域场景中显著优于零样本推理和 LLM 集成方法。

## 研究背景与动机

**领域现状**：问答（QA）是 LLM 最主要的应用场景之一，但 LLM 有时会产生虚假或误导性回答（"幻觉"）。因此，将生成的答案锚定在提供的上下文信息中——即为生成文本提供证据来源——对 LLM 的可信赖性至关重要。

**现有痛点**：上下文归因任务（context attribution）需要模型标注答案中每个部分对应哪些上下文句子。这个任务的关键挑战在于：(1) 高质量的归因标注数据极度稀缺且人工标注成本高昂；(2) 直接用 LLM 做零样本推理的归因效果不稳定；(3) 现有方法要么需要很强的 LLM 能力（成本高），要么只关注简单的二元判断而非精确的句子级归因。

**核心矛盾**：需要大量训练数据来训练可靠的小模型做归因，但人工标注数据又太贵。关键问题是：如何用 LLM 的生成能力来自动合成高质量的归因训练数据？

**本文目标**：系统研究基于 LLM 的上下文归因方法（零样本、集成、微调），并提出一种有效的合成数据策略来训练廉价的小模型达到甚至超过大模型的归因效果。

**切入角度**：传统思路是让 LLM 先生成答案再标注归因，但这会引入错误。SynQA 反转了这个流程——先选定上下文句子，再让 LLM 根据这些句子生成 QA 对，这样归因关系天然内置在合成数据中。

**核心 idea**：用"先选证据句→再生成 QA 对"的逆向合成策略，利用 LLM 的文本生成长处，同时确保合成数据自带清晰的归因路径。

## 方法详解

### 整体框架

SynQA 的 pipeline 为：(1) 从文档中采样一组上下文句子作为证据；(2) 用大 LLM（如 GPT-4）根据这些句子生成一个问题和对应答案；(3) 将（问题、答案、证据句子）三元组作为训练数据；(4) 用这些数据微调小模型（如 Flan-T5、Mistral-7B）做归因分类。推理时，给定问题、上下文和生成的答案，小模型判断每个上下文句子是否是答案的支撑证据。

### 关键设计

1. **逆向合成数据策略 (SynQA)**:

    - 功能：自动生成大量高质量的上下文归因训练数据
    - 核心思路：传统方式是"给定问题→生成答案→标注哪些句子支撑答案"，SynQA 反转为"随机选定 k 个上下文句子→让 LLM 基于这些句子生成问题和答案"。这样生成的 QA 对天然地被所选句子支撑，归因标签自动获得。采样句子数量 k 从1到多变化，生成不同难度的训练样本
    - 设计动机：LLM 擅长文本生成但不擅长精确标注。逆向策略将归因标注转化为输入约束，把困难的标注任务转化为 LLM 更擅长的生成任务

2. **多方法系统性比较框架**:

    - 功能：全面评估不同 LLM 归因方法的优劣
    - 核心思路：研究三类方法：(i) 零样本推理——直接 prompt LLM 识别支撑句子，测试不同提示策略；(ii) LLM 集成——用多个 LLM 的判断进行投票，通过多数决或加权方式融合结果；(iii) 微调小 LM——在 SynQA 合成数据上微调 Flan-T5-XL（3B）和 Mistral-7B-Instruct 等小模型
    - 设计动机：需要确认在不同成本-性能权衡下的最佳选择，以及小模型微调是否真的能逼近甚至超过大模型

3. **跨域泛化评估**:

    - 功能：验证 SynQA 合成数据在不同 QA 任务和领域的迁移能力
    - 核心思路：在 Natural Questions、ExpertQA、HAGRID 等多个 QA 数据集上评测，这些数据集涵盖维基百科问答、专家领域问答等不同场景。训练数据和测试数据来自不同领域，评估 zero-shot 迁移性能
    - 设计动机：上下文归因的实际应用涉及各种领域，只在训练域内有效的方法价值有限，跨域泛化能力是实用性的关键

### 损失函数 / 训练策略

微调小模型时使用标准的序列到序列损失。输入格式为 "(question, answer, context_sentence) → {attributable, not_attributable}"。训练使用 SynQA 合成数据，不同规模的合成数据量（1k-10k 样本）进行消融实验。

## 实验关键数据

### 主实验

| 方法 | Natural Questions (F1) | ExpertQA (F1) | HAGRID (F1) | 成本 |
|------|----------------------|--------------|------------|------|
| GPT-4 零样本 | 68.2 | 62.5 | 65.4 | 高 |
| LLM 集成 (3模型) | 71.3 | 64.8 | 68.1 | 很高 |
| Flan-T5-XL + SynQA | 74.5 | 67.3 | 71.2 | 低 |
| Mistral-7B + SynQA | **76.8** | **69.1** | **73.5** | 中 |

### 消融实验

| 配置 | Natural Questions (F1) | 说明 |
|------|----------------------|------|
| Mistral-7B + SynQA (10k) | 76.8 | 完整模型 |
| Mistral-7B + SynQA (5k) | 75.2 | 数据减半影响较小 |
| Mistral-7B + SynQA (1k) | 71.4 | 数据量不足时退化明显 |
| 传统正向合成 (先问后答再标注) | 70.1 | 逆向策略优势约 +6.7% |
| 不用合成数据，直接零样本 | 68.2 | 微调带来显著提升 |

### 关键发现

- **小模型 + SynQA 微调 > 大模型零样本**：微调后的 Mistral-7B 在所有测试集上超过 GPT-4 零样本推理，成本却低得多
- **逆向合成策略有效**：SynQA 的"先选证据→后生成 QA"方式比传统"先生成→后标注"方式的数据质量明显更高，在下游微调中体现为 5-7% 的 F1 提升
- **跨域迁移良好**：在一个领域合成的数据训练的模型，可以有效迁移到其他 QA 领域，说明 SynQA 学到的归因模式具有通用性
- **用户研究验证实用性**：人工评估确认微调小模型提供的归因确实帮助用户更快地验证答案的正确性

## 亮点与洞察

- **"逆向合成"的巧妙思路**：将标注问题转化为生成问题是核心贡献。这个思想可以推广到任何"需要配对标注但标注昂贵"的 NLP 任务中——例如摘要的证据归因、对话系统的知识追溯等
- **实用性强的成本-性能权衡**：证明了在部署侧，用合成数据训练小模型比调用大 LLM API 更经济有效。对于需要大规模部署归因功能的生产系统有直接参考价值
- **LLM 集成方法的边际收益有限**：多个 LLM 投票的提升远不如用一个 LLM 生成训练数据来微调小模型，这个发现打破了"集成一定更好"的直觉

## 局限与展望

- **合成数据质量依赖 LLM 能力**：SynQA 的效果受制于生成 QA 对的 LLM 的质量，如果用较弱的 LLM 生成合成数据，效果会退化
- **只在英文上验证**：跨语言归因场景（如多语言 RAG 系统）未涉及
- **归因粒度固定在句子级别**：实际应用中可能需要更细粒度（子句级别）或更粗粒度（段落级别）的归因
- 未来可以探索将 SynQA 扩展到多模态场景——如图文混合问答中的视觉证据归因

## 相关工作与启发

- **vs ALCE (Gao et al., 2023)**: ALCE 关注的是让 LLM 在生成时自带引用，是"生成时归因"；SynQA 关注的是事后归因，两者互补
- **vs AttriBench**: AttriBench 提供了归因评测基准但不提供解决方案；SynQA 既有评测也有训练方法
- **与 RAG 系统的关联**：SynQA 训练的归因模型可以直接作为 RAG 系统的后处理模块，为检索增强生成的答案提供证据追溯

## 评分

- 新颖性: ⭐⭐⭐⭐ "逆向合成"思路新颖且有效，但合成数据微调本身已是常见 paradigm
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多方法对比、消融、用户研究都有
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述系统
- 价值: ⭐⭐⭐⭐ 对可信 AI 和 RAG 系统有直接实用价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Recursive Question Understanding for Complex Question Answering over Heterogeneous Personal Data](recursive_question_understanding_for_complex_question_answering_over_heterogeneo.md)
- [\[ACL 2025\] Attribution Methods in NLP: Navigating a Fragmented Landscape](attribution_methods_in_nlp_navigating_a_fragmented_landscape.md)
- [\[ACL 2025\] Adapting Psycholinguistic Research for LLMs: Gender-Inclusive Language in a Coreference Context](adapting_psycholinguistic_research_for_llms_gender-inclusive_language_in_a_coref.md)
- [\[ACL 2025\] iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering](iquest_an_iterative_question-guided_framework_for_knowledge_base_question_answer.md)
- [\[ACL 2025\] Active LLMs for Multi-hop Question Answering](active_llms_for_multi-hop_question_answering.md)

<!-- RELATED:END -->
