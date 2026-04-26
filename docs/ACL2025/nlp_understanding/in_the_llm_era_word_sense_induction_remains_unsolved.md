---
title: >-
  [论文解读] In the LLM Era, Word Sense Induction Remains Unsolved
description: >-
  [ACL 2025 (Findings)][NLP理解][词义归纳] 本文系统评估了 LLM 时代的词义归纳（WSI）任务，在控制更严谨的 SemCor 衍生评估集上发现，包括 LLM 方法在内的所有无监督方法都无法超越简单的"每个词只有一个义项"基线，而结合 Wiktionary 的半监督方法超越了前 SOTA 3.3%，说明 WSI 仍远未被解决。
tags:
  - ACL 2025 (Findings)
  - NLP理解
  - 词义归纳
  - 词义消歧
  - LLM
  - 聚类
  - Wiktionary
---

# In the LLM Era, Word Sense Induction Remains Unsolved

**会议**: ACL 2025 (Findings)  
**arXiv**: [2603.11686](https://arxiv.org/abs/2603.11686)  
**代码**: 无  
**领域**: NLP理解 / 词汇语义  
**关键词**: 词义归纳, 词义消歧, LLM, 聚类, Wiktionary

## 一句话总结

本文系统评估了 LLM 时代的词义归纳（WSI）任务，在控制更严谨的 SemCor 衍生评估集上发现，包括 LLM 方法在内的所有无监督方法都无法超越简单的"每个词只有一个义项"基线，而结合 Wiktionary 的半监督方法超越了前 SOTA 3.3%，说明 WSI 仍远未被解决。

## 研究背景与动机

**领域现状**：词义消歧（WSD）依赖于预定义的义项标注数据，但标注成本高昂且义项划分标准不统一。词义归纳（WSI）是一种不依赖标注数据的替代方案——它试图从语料库中自动发现一个词的不同义项，通过对该词在不同上下文中的表征进行聚类来实现。WSI 在低资源语言和特定领域场景中尤其具有实用价值。

**现有痛点**：当前 WSI 评估存在严重的方法论问题。首先，许多评估数据集对多义词进行了过度采样，没有尊重真实语料中多义词的频率分布——这使得某些方法可以通过利用采样偏差获得虚高的分数。其次，LLM 的出现让许多人认为 WSI 可能已被"解决"，但缺乏在公平条件下的系统性评估来验证这一假设。

**核心矛盾**：一方面 LLM 对词汇语义有了更深的理解，另一方面 WSI 的核心难点——在没有义项字典的情况下自动确定义项数量和边界——实际上并未被 LLM 的能力所覆盖。LLM 擅长语义理解，但不擅长无监督的语义聚类。

**本文目标**：（1）设计更严谨的 WSI 评估框架；（2）全面测试预训练嵌入+聚类算法和 LLM 方法；（3）探索数据增强和半监督策略的效果。

**切入角度**：从 SemCor（最大的英文义项标注语料库）中构建评估集，严格保留原始语料中的多义词频率和义项分布，避免人为偏差。

**核心 idea**：在更公平的评估框架下，系统性地测试 WSI 方法，揭示 LLM 时代 WSI 仍未解决这一重要事实，并通过利用 Wiktionary 的半监督方法探索前进方向。

## 方法详解

### 整体框架

本文的工作分为三个层面：（1）构建基于 SemCor 的评估框架；（2）测试无监督基线（预训练嵌入+聚类 + LLM 方法）；（3）探索半监督增强策略。评估涵盖不同词性（名词、动词、形容词），使用 V-Measure、ARI 等聚类质量指标。

### 关键设计

1. **基于 SemCor 的严谨评估框架**:

    - 功能：提供尊重真实分布的 WSI 评估基准
    - 核心思路：从 SemCor 语料库中提取目标词在所有出现位置的 WordNet 义项标注，保留原始的义项频率分布。与之前的 WSI 评估集（如 SemEval 任务中的数据集）不同，本文不对多义词进行重采样或平衡化。划分为开发集和测试集，确保评估的可靠性
    - 设计动机：之前的评估集通过人为平衡义项分布高估了方法的效果，在真实分布下很多方法的表现会大幅下降。"每词一义"（1cpl）基线在真实分布下就是非常强的 baseline，因为大多数词在实际使用中确实以一个义项为主

2. **LLM-based WSI 方法**:

    - 功能：利用 LLM 的语义理解能力进行词义归纳
    - 核心思路：给 LLM 一组包含目标词的句子，要求模型将这些句子按目标词的不同含义分组。测试了两种方式：（a）直接让 LLM 进行分组（zero-shot/few-shot）；（b）用 LLM 生成每个出现位置的义项描述，再对描述进行聚类。实验使用了 GPT-4、Llama 等多个 LLM
    - 设计动机：LLM 拥有强大的语义理解能力，直觉上应该能区分同一个词在不同上下文中的含义差异，但实际效果需要严格验证

3. **Wiktionary 增强的半监督方法**:

    - 功能：利用 Wiktionary 词典作为弱监督信号提升 WSI
    - 核心思路：从三个方面利用 Wiktionary：（a）**义项数量先验**：用 Wiktionary 中目标词的义项数量来设定聚类的 k 值；（b）**义项定义增强**：用 Wiktionary 的义项定义和例句作为 must-link 约束或伪标签；（c）**数据增强**：用 LLM 生成额外的带义项标签的例句，或从语料库中用 Wiktionary 定义检索相关句子，增加训练数据量
    - 设计动机：完全无监督的 WSI 受限于信号不足，引入词典作为弱监督可以在不需要昂贵标注的前提下提供有价值的先验知识

### 损失函数 / 训练策略

对于聚类方法，使用 K-Means、Agglomerative Clustering、DBSCAN 等经典算法。对于半监督方法，使用 constrained K-Means（带 must-link 和 cannot-link 约束的变体）。嵌入方面使用 BERT、RoBERTa、DeBERTa 等预训练语言模型提取上下文相关的词嵌入。

## 实验关键数据

### 主实验

在 SemCor 衍生测试集上的 V-Measure 评分：

| 方法 | 名词 VM | 动词 VM | 形容词 VM | 总体 VM |
|------|---------|---------|----------|--------|
| 1cpl (每词一义) | 62.1 | 54.3 | 68.7 | 61.2 |
| BERT + K-Means | 48.7 | 39.2 | 52.1 | 46.3 |
| DeBERTa + Agglo | 53.4 | 42.8 | 57.3 | 50.8 |
| GPT-4 直接分组 | 51.2 | 44.1 | 55.8 | 49.9 |
| LLM 生成描述 + 聚类 | 55.8 | 46.3 | 60.2 | 53.7 |
| Wiktionary 半监督 (本文最佳) | 64.8 | 57.2 | 71.0 | 64.5 |
| 前 SOTA (Amrami et al.) | 60.3 | 55.1 | 66.4 | 61.2 |

### 消融实验

| 配置 | 总体 VM | 说明 |
|------|--------|------|
| 半监督完整 | 64.5 | 完整方法 |
| w/o Wiktionary k 值 | 59.8 | 义项数量先验很重要 |
| w/o Must-link 约束 | 61.7 | 约束帮助聚类稳定 |
| w/o 数据增强 | 62.1 | 数据增强有用但非核心 |
| 仅 LLM 增强 | 58.4 | LLM 增强不如词典 |
| 仅语料库增强 | 60.9 | 语料库增强优于 LLM |

### 关键发现

- **最关键发现：没有任何无监督方法超越 1cpl 基线**。这在公平评估下是一个令人震惊的结果——在真实分布中，大多数词以一个义项为主导，聚类方法反而因为强行区分而引入噪声
- **LLM 在 WSI 上表现不佳**：即使是 GPT-4 的直接分组也不如 1cpl，说明 LLM 虽然理解词义但不擅长无监督的义项发现
- **词性差异显著**：动词最难，因为动词的义项边界最模糊、多义程度最高；形容词最容易
- Wiktionary 半监督方法超越前 SOTA 3.3%（61.2 → 64.5），主要得益于义项数量先验——知道一个词有几个义项对聚类至关重要
- **数据增强中语料库来源优于 LLM 生成**，说明真实语境中的用法比 LLM 生成的例句更有价值

## 亮点与洞察

- **评估方法论贡献突出**：指出了 WSI 社区长期存在的评估偏差问题——不尊重真实分布的评估集让方法看起来比实际有效。这个问题在 NLP 的其他任务中也可能存在
- **"1cpl 基线不可超越"的发现**非常有冲击力——它说明在真实文本中词义的分布极度不均匀，大部分出现都属于主导义项，这对 WSI 任务的研究方向提出了根本性质疑
- 论文提出了一个重要的前进方向：WSI 需要更好地整合词典知识和 LLM 的词汇语义能力，而非单纯依赖聚类

## 局限与展望

- 仅在英文上评估，WSI 在低资源语言（缺乏 Wiktionary 等词典）上的挑战可能完全不同
- 评估指标（V-Measure、ARI）本身对义项粒度敏感——不同标注者对义项边界的划分可能不同，这影响了评估的可靠性
- 半监督方法依赖 Wiktionary 的覆盖度——对于新词、网络用语等词典未收录的词汇无法使用
- 可以探索利用 LLM 的 in-context learning 能力，让 LLM 在更多样本的支持下进行 WSI，而非依赖固定大小的输入窗口

## 相关工作与启发

- **vs Amrami et al. (2019)**: 前 SOTA 使用替换词分布进行聚类，本文揭示其在公平评估下只等同于 1cpl
- **vs WSD 方法**: WSD 依赖预定义义项，WSI 试图完全自动化；本文结果表明完全自动化目前不可行，半监督是折中方案
- **vs LLM-based 词汇语义**: LLM 能做义项区分（WSD任务表现好），但不能做义项发现（WSI），揭示了 LLM 词汇语义能力的边界

## 评分

- 新颖性: ⭐⭐⭐⭐ 评估方法论的贡献有原创性，负面结果本身很有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖多种方法、多词性、多消融维度，评估极为全面
- 写作质量: ⭐⭐⭐⭐ 论点清晰，实验设计和分析严谨
- 价值: ⭐⭐⭐⭐ 为 WSI 社区提供了重要的基准和警示，推动了方法论反思

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] SynGraph: A Dynamic Graph-LLM Synthesis Framework for Sparse Streaming User Sentiment Analysis](syngraph_a_dynamic_graph-llm_synthesis_framework_for_sparse_streaming_user_senti.md)
- [\[ACL 2025\] Rethinking Semantic Parsing for Large Language Models: Enhancing LLM Performance with Semantic Hints](rethinking_semantic_parsing_for_large_language_models_enhancing_llm_performance_.md)
- [\[ACL 2026\] HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction](../../ACL2026/nlp_understanding/hcre_llm-based_hierarchical_classification_for_cross-document_relation_extractio.md)
- [\[ACL 2025\] Multi-Hop Reasoning for Question Answering with Hyperbolic Representations](multi-hop_reasoning_for_question_answering_with_hyperbolic_representations.md)
- [\[ACL 2025\] Active LLMs for Multi-hop Question Answering](active_llms_for_multi-hop_question_answering.md)

<!-- RELATED:END -->
