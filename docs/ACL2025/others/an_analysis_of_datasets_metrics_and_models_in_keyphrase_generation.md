---
title: >-
  [论文解读] An Analysis of Datasets, Metrics and Models in Keyphrase Generation
description: >-
  [ACL 2025 (GEM²)][关键短语生成] 对关键短语生成（keyphrase generation）领域50+篇论文进行系统性分析，揭示了基准数据集高度相似、评估指标计算不一致导致性能被高估等关键问题，并发布了一个强力PLM-based模型以促进未来研究。
tags:
  - ACL 2025 (GEM²)
  - 关键短语生成
  - 数据集分析
  - 评估指标
  - 预训练语言模型
  - 综述分析
---

# An Analysis of Datasets, Metrics and Models in Keyphrase Generation

**会议**: ACL 2025 (GEM²)  
**arXiv**: [2506.10346](https://arxiv.org/abs/2506.10346)  
**代码**: 无（但发布了预训练模型）  
**领域**: Others (NLP / Information Retrieval)  
**关键词**: 关键短语生成, 数据集分析, 评估指标, 预训练语言模型, 综述分析

## 一句话总结

对关键短语生成（keyphrase generation）领域50+篇论文进行系统性分析，揭示了基准数据集高度相似、评估指标计算不一致导致性能被高估等关键问题，并发布了一个强力PLM-based模型以促进未来研究。

## 研究背景与动机

关键短语生成（Keyphrase Generation, KG）是指从文档中自动生成一组能概括文档内容的词或短语。近年来该领域在模型架构、数据资源和应用场景等多个方向持续发展。然而，领域存在以下核心问题：

**缺乏系统性回顾**：过去几年KG研究分散在多条技术路线上（Seq2Seq架构、预训练模型、生成策略等），但至今没有人对这些工作进行系统梳理和对比分析，导致领域现状不清晰。

**基准数据集高度相似**：KG领域常用的几个benchmark（如KP20k、Inspec、Krapivin、NUS、SemEval等）之间存在令人担忧的高度相似性。这些数据集大多来源于学术论文，领域分布和文本风格高度重叠，导致模型可能只是在重复学习而非真正泛化。

**评估指标计算不一致**：不同论文在计算F1@K、F1@M等指标时存在显著不一致——词干化（stemming）处理方式不同、截断策略不同、present/absent keyphrase的划分标准不同，这些差异导致跨论文的性能对比不可靠，部分方法的性能被高估。

**预训练模型可获取性差**：尽管近年来基于PLM的方法表现优异，但大多数论文未开放预训练模型，阻碍了后续研究的进展和公平对比。

本文的切入角度是：作为一篇综合性分析论文（analysis paper），系统性地审视KG领域的数据集、指标和模型三个方面的问题，并通过发布强基线模型来填补可复现性的空白。

## 方法详解

### 整体框架

本文不是一篇提出新方法的论文，而是一篇系统性分析论文。其分析框架包括：
- **输入**：50+篇关键短语生成相关论文
- **分析维度**：数据集特征、评估指标一致性、模型架构演进
- **输出**：问题诊断 + 改进建议 + 强基线PLM模型

### 关键设计

1. **数据集相似性分析**:

    - 功能：量化分析常用KG基准数据集之间的相似程度
    - 核心思路：通过计算不同数据集间的文本重叠率、领域分布、文档长度分布和关键短语数量分布等统计特征，揭示数据集间的高度冗余。例如KP20k、Krapivin、SemEval等数据集都以计算机科学领域学术论文为主，文本风格和主题分布高度一致
    - 设计动机：如果benchmark之间高度相似，那么"在多个benchmark上表现好"并不能证明模型的泛化能力，这对领域评估体系的可信度构成威胁

2. **评估指标不一致性分析**:

    - 功能：梳理不同论文中F1@K、F1@M等指标的计算细节差异
    - 核心思路：对比分析以下方面的实现差异：
        - **词干化处理**：Porter Stemmer vs. 无词干化，对匹配结果影响显著
        - **Present/Absent划分**：present keyphrase（出现在文档中的）和absent keyphrase（未出现在文档中的）的判定标准不统一
        - **截断策略**：$F_1@5$ 时取前5个预测 vs. $F_1@M$ 取与标注数量相同的预测
        - **去重方式**：生成结果的去重策略各异
    - 设计动机：指标计算的微小差异可能导致性能数字差异巨大，使得跨论文对比失去意义，部分方法的优势可能来自指标计算的"优势"而非模型本身

3. **PLM-based强基线模型**:

    - 功能：训练并发布一个基于预训练语言模型的KG模型作为可复现的强基线
    - 核心思路：基于T5/BART等预训练Seq2Seq模型，在标准KG数据集上微调，采用统一的数据预处理和评估流程
    - 设计动机：解决领域内预训练模型可获取性差的问题，为后续研究提供公平的对比基准

### 分析方法论

论文对50+篇KG论文进行了系统性梳理，从以下维度分类：
- 生成范式：One2One（一次生成一个keyphrase）vs. One2Seq（一次生成所有keyphrase序列）
- 模型架构：RNN-based → Transformer-based → PLM-based的演化
- 评估设置：不同论文使用的数据集、指标和预处理方式

## 实验关键数据

### 主实验（数据集相似性分析结果）

| 数据集对 | 相似性指标 | 发现 |
|----------|-----------|------|
| KP20k ↔ Krapivin | 高 | 来源相似（学术论文），领域重叠大 |
| KP20k ↔ Inspec | 中-高 | 虽然文档长度不同，但主题分布相近 |
| KP20k ↔ SemEval | 高 | 均为CS领域论文 |
| KP20k ↔ NUS | 中 | NUS领域稍广但仍以CS为主 |

### 指标不一致性影响（消融分析）

| 评估配置 | F1@5变化 | 说明 |
|----------|---------|------|
| 有词干化 vs. 无词干化 | 显著差异 | 词干化通常提升匹配率5-10% |
| 不同present/absent划分标准 | 差异明显 | 严格 vs. 宽松判定影响absent keyphrase评估 |
| 统一评估流程 | 基准值 | 本文推荐的标准化评估 |
| 文献报告值 | 普遍偏高 | 部分论文因指标实现差异获得虚高分数 |

### 关键发现

- 常用基准数据集间存在令人担忧的高度相似性，单一领域（CS学术论文）主导了整个评估体系
- 评估指标计算的不一致性是一个被长期忽视的系统性问题，导致领域内的性能比较不可靠
- PLM-based方法已经是当前最强范式，但模型可获取性是瓶颈
- 从RNN到Transformer到PLM的架构演进带来的提升可能部分来自评估不一致的"幻觉"

## 亮点与洞察

- **数据集冗余性分析**是KG领域首次系统性的审视，对其他NLP子领域也有借鉴意义——很多领域的benchmark可能也存在类似的冗余问题
- **评估标准化**的呼吁非常及时：本文具体指出了哪些计算细节需要统一，这比泛泛的"需要标准化"更有操作性
- 发布可复现的PLM基线模型，降低该领域的入门门槛
- 作为一篇GEM²类型的分析论文，其方法论可以迁移到其他NLP任务的系统性分析中

## 局限与展望

- 分析主要聚焦英文KG研究，多语言场景覆盖不足
- 虽然揭示了问题，但未提出具体的新数据集来解决领域单一性问题
- 发布的PLM基线模型细节和性能数据需要查看完整论文才能全面评估
- 未深入分析LLM（如GPT-4、LLaMA）在KG任务上的零样本/少样本表现

## 相关工作与启发

- **vs CatSeq (Yuan et al., 2020)**: CatSeq是One2Seq范式的代表性工作，本文揭示了其在不同评估设置下的性能差异远大于不同方法间的差异
- **vs KG综述**: 不同于传统综述仅梳理方法，本文更像一篇"领域健康检查报告"，深入分析了评估体系本身的问题
- **vs NLG评估分析**: 类似于机器翻译领域对BLEU指标的反思，本文对KG评估的反思有望推动领域评估标准的改进

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性分析KG领域的数据集/指标问题，视角独到
- 实验充分度: ⭐⭐⭐⭐ 50+篇论文的系统性分析，数据集相似性分析扎实
- 写作质量: ⭐⭐⭐⭐⭐ 分析论文需要清晰的逻辑，本文组织结构很好
- 价值: ⭐⭐⭐⭐ 对KG领域评估标准的反思有重要推动作用，发布基线模型有实用价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] ERU-KG: Efficient Reference-aligned Unsupervised Keyphrase Generation](eru-kg_efficient_reference-aligned_unsupervised_keyphrase_generation.md)
- [\[ACL 2025\] Limited Generalizability in Argument Mining: State-Of-The-Art Models Learn Datasets, Not Arguments](limited_generalizability_in_argument_mining_state-of-the-art_models_learn_datase.md)
- [\[ACL 2025\] Identifying Reliable Evaluation Metrics for Scientific Text Revision](reliable_eval_metrics_scientific.md)
- [\[ACL 2025\] A Measure of the System Dependence of Automated Metrics](a_measure_of_the_system_dependence_of_automated_metrics.md)
- [\[ACL 2025\] MapQaTor: An Extensible Framework for Efficient Annotation of Map-Based QA Datasets](mapqator_an_extensible_framework_for_efficient_annotation_of_map-based_qa_datase.md)

<!-- RELATED:END -->
