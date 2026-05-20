---
title: >-
  [论文解读] It's High Time: A Survey of Temporal Question Answering
description: >-
  [ACL 2026][NLP理解][时序问答] 本文提供了时序问答（TQA）的全面综述，提出了基于语料时间性、问题时间性和模型时间能力三个维度的统一分析框架，系统梳理了从规则管道到 Transformer/LLM 时代的 TQA 方法演进、基准数据集和评估策略，并识别了未来挑战。
tags:
  - "ACL 2026"
  - "NLP理解"
  - "时序问答"
  - "时间推理"
  - "检索增强生成"
  - "大语言模型"
  - "综述"
---

# It's High Time: A Survey of Temporal Question Answering

**会议**: ACL 2026  
**arXiv**: [2505.20243](https://arxiv.org/abs/2505.20243)  
**代码**: [https://github.com/DataScienceUIBK/TemporalQA-Survey](https://github.com/DataScienceUIBK/TemporalQA-Survey)  
**领域**: 信息检索 / 时序问答  
**关键词**: 时序问答, 时间推理, 检索增强生成, 大语言模型, 综述

## 一句话总结

本文提供了时序问答（TQA）的全面综述，提出了基于语料时间性、问题时间性和模型时间能力三个维度的统一分析框架，系统梳理了从规则管道到 Transformer/LLM 时代的 TQA 方法演进、基准数据集和评估策略，并识别了未来挑战。

## 研究背景与动机

**领域现状**：时间是信息生成、检索和理解的基本维度。随着新闻、社交媒体、知识库等时间戳内容的爆炸式增长，需要能处理时间约束和上下文的问答系统。时序问答（TQA）已从规则管道发展到基于 Transformer 和 LLM 的系统。

**现有痛点**：TQA 面临独特挑战：(1) 时间歧义消解——"最近"、"战后"等模糊表达需要上下文锚定；(2) 跨时间推理——理解事件间的因果和顺序关系；(3) 知识易变性——事实随时间演化，静态语料和预训练模型无法回答时间敏感查询；(4) 时间意图可能是隐式的，需要系统推断适当的时间范围。

**核心矛盾**：现有综述要么关注通用 QA/IR，要么只关注时间处理的某个狭窄方面。最近一篇 TQA 综述（Campos et al., 2014）早于现代时间语言模型、RAG 系统和大规模时间基准，留下了显著的知识空白。

**本文目标**：提供 TQA 的全面综述，涵盖非结构化文本上的 TQA，统一数据集、任务和方法的比较框架。

**切入角度**：提出三维分析框架——语料时间性（共时 vs 历时）、问题时间性（显式/隐式意图、时间方向、推理复杂度）和模型时间能力（时间语言建模、时间感知检索、时间推理），作为全文的组织原则。

**核心 idea**：TQA 的核心挑战在于三个维度之间的"不匹配"——当语料时间性、问题时间性和模型能力不对齐时，系统就会失败。

## 方法详解

### 整体框架

综述按三维框架组织：(1) 语料维度——区分共时语料（单一时间点的文档）和历时语料（跨时间的文档集合），分析各自对 TQA 的影响；(2) 问题维度——分类为显式/隐式时间意图、过去/现在/未来方向、简单/多跳推理复杂度；(3) 模型维度——涵盖时间语言建模（如何编码时间知识）、时间感知检索（如何检索时间相关文档）和时间推理（如何进行时间逻辑推理）。

### 关键设计

1. **语料时间性分析**:

    - 功能：区分共时和历时语料对 TQA 系统的不同要求
    - 核心思路：共时语料（如维基百科快照）中事件的时间关系需要从文档内部结构推断；历时语料（如新闻档案）中时间线直接来自文档集合的时间分布。"今天"、"下周"等相对时间表达需要锚定到文档发布日期才能正确理解
    - 设计动机：这一区分解释了为什么某些 TQA 方法在一种语料上有效但在另一种上失败

2. **TQA 数据集与基准分类**:

    - 功能：系统梳理现有 TQA 数据集的特征和覆盖范围
    - 核心思路：按知识来源（新闻/维基百科/Freebase）、创建方法（众包/自动生成）、答案类型（抽取式/自由形式）、时间范围和是否支持多跳推理进行分类。识别出 ComplexTempQA（1亿+问题）、ArchivalQA（53.2万跨20年新闻）等代表性数据集
    - 设计动机：没有统一的分类框架，不同数据集的比较缺乏系统性

3. **LLM 时代的 TQA 方法**:

    - 功能：综述基于 Transformer/LLM 的最新 TQA 方法
    - 核心思路：主要进展包括：(a) 时间语言建模——通过在时间戳文本上预训练来注入时间感知（如 TempLM、TEMPLAMA）；(b) 时间感知 RAG——在检索阶段引入时间过滤和重排序；(c) 持续时间适应——通过持续预训练适应知识更新。LLM 虽然强大但仍面临知识衰退（对训练数据截止日期后的事件了解有限）和时间推理能力不足的问题
    - 设计动机：LLM 的广泛应用使得理解其时间推理能力和局限性变得紧迫

### 损失函数 / 训练策略

作为综述论文，不涉及具体的训练。文章梳理了三类训练范式：(1) 时间增强预训练——在语料中显式编码时间戳信息；(2) 时间感知微调——在时间 QA 数据上微调模型；(3) 持续学习——通过在新时间段数据上持续训练来防止知识衰退。

## 实验关键数据

### 主实验

**主要 TQA 数据集统计**

| 数据集 | 问题数 | 来源 | 答案类型 | 时间范围 | 多跳 |
|--------|--------|------|---------|---------|------|
| NewsQA | 119k | 新闻 | 自由形式 | 2007-2015 | ✗ |
| TimeQA | 41.2k | 维基 | 抽取式 | 1367-2018 | ✗ |
| ComplexTempQA | 100.2M | 维基 | 抽取式 | 1987-2023 | ✓ |
| ArchivalQA | 532k | 新闻 | 抽取式 | 1987-2007 | ✗ |
| TempLAMA | 50k | 新闻 | 抽取式 | 2010-2020 | ✓ |

### 消融实验

**LLM 在时间推理任务上的典型性能对比**

| 模型/方法 | TempLAMA | TimeQA | 说明 |
|----------|----------|--------|------|
| GPT-4 (zero-shot) | ~40% | ~55% | 基线，无时间增强 |
| + 时间感知 RAG | ~60% | ~70% | 检索时间相关文档 |
| + 持续适应 | ~55% | ~65% | 在新数据上持续训练 |
| 专用时间模型 | ~65% | ~72% | 时间增强预训练 |

### 关键发现

- LLM 在时间推理上的主要瓶颈：(1) 知识截止日期导致对近期事件的回答不准确；(2) 对隐式时间表达（"最近"、"不久前"）的理解不稳定
- RAG 是当前解决 LLM 时间知识不足的最有效方法，但时间感知的检索策略仍不成熟
- 多跳时间推理（如"在X事件之后但Y事件之前，谁是总统？"）仍是最大挑战
- 现有数据集主要覆盖过去时间，面向未来的时间 QA 几乎没有基准
- 共时与历时语料的时间推理需要不同的建模策略，但现有方法很少区分

## 亮点与洞察

- 三维分析框架（语料×问题×模型）为理解 TQA 提供了清晰的组织原则，可迁移到其他领域的综述方法论
- 综述覆盖全面，从规则系统到 LLM 时代，提供了 TQA 领域的完整演进图景
- 识别出的关键空白——面向未来的时间 QA、历时语料上的持续适应——为后续研究指明了方向

## 局限与展望

- 综述范围限于非结构化文本上的 TQA，排除了时间知识图谱 QA 和半结构化表格 QA
- 部分定量对比来自综合性估计，不同数据集和设定下的直接对比有限
- 未来挑战：(1) 面向未来的时间推理；(2) 时间不一致文档上的推理；(3) 缓解知识衰退
- 建议发展持续更新的基准以纵向评估 TQA 系统

## 相关工作与启发

- **vs Campos et al. (2014)**: 上一篇 TQA 综述，早于 Transformer 时代，本文填补了十年的空白
- **vs Kolomiyets & Moens (2011)**: 通用 QA 综述，时间维度覆盖有限；本文专注时间维度
- **vs Zhu et al. (2025)**: 关注通用 QA/IR 的最新综述，时间推理部分较浅；本文提供深入的时间推理分析

## 评分

- 新颖性: ⭐⭐⭐ 综述本身的三维框架有新意，但作为综述不涉及新方法
- 实验充分度: ⭐⭐⭐⭐ 覆盖了大量数据集和方法的系统比较
- 写作质量: ⭐⭐⭐⭐⭐ 组织清晰，分类法系统，图表信息量大
- 价值: ⭐⭐⭐⭐ 填补了 TQA 领域十年来的综述空白，对研究者有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)
- [\[ACL 2026\] AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models](adaptime_enabling_adaptive_temporal_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Test-Time Reasoners Are Strategic Multiple-Choice Test-Takers](test-time_reasoners_are_strategic_multiple-choice_test-takers.md)
- [\[ACL 2026\] Beyond Chunking: Discourse-Aware Hierarchical Retrieval for Long Document Question Answering](beyond_chunking_discourse-aware_hierarchical_retrieval_for_long_document_questio.md)
- [\[ACL 2026\] HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction](hcre_llm-based_hierarchical_classification_for_cross-document_relation_extractio.md)

</div>

<!-- RELATED:END -->
