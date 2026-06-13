---
title: >-
  [论文解读] Adaptive Retrieval without Self-Knowledge? Bringing Uncertainty Back Home
description: >-
  [ACL 2025][自适应检索] 对 35 种自适应检索方法（含 8 种最新方法和 27 种不确定性估计方法）进行了全面评测，发现经典的不确定性估计技术在效率和自知能力方面往往优于复杂的专用流水线，同时保持相当的 QA 性能。 检索增强生成（RAG）可以改善 LLM 的回答正确性、缓解幻觉问题，但存在两个关键挑战： 并非总…
tags:
  - "ACL 2025"
  - "自适应检索"
  - "RAG"
  - "不确定性估计"
  - "自知能力"
  - "问答系统"
---

# Adaptive Retrieval without Self-Knowledge? Bringing Uncertainty Back Home

**会议**: ACL 2025  
**arXiv**: [2501.12835](https://arxiv.org/abs/2501.12835)  
**代码**: [https://github.com/s-nlp/AdaRAGUE](https://github.com/s-nlp/AdaRAGUE)  
**领域**: 其他  
**关键词**: 自适应检索, RAG, 不确定性估计, 自知能力, 问答系统

## 一句话总结

对 35 种自适应检索方法（含 8 种最新方法和 27 种不确定性估计方法）进行了全面评测，发现经典的不确定性估计技术在效率和自知能力方面往往优于复杂的专用流水线，同时保持相当的 QA 性能。

## 研究背景与动机

检索增强生成（RAG）可以改善 LLM 的回答正确性、缓解幻觉问题，但存在两个关键挑战：

**并非总是需要检索**：当 LLM 已经掌握了相关知识时，检索可能引入无关信息，导致误差累积和外部幻觉

**计算成本高昂**：每次检索都需要调用检索器和额外的 LM 推理

为此，自适应检索方法应运而生——利用 LLM 的**自知能力**（self-knowledge，即模型识别自身知识边界的能力）来决定何时需要检索。然而，现有研究存在三个盲点：

- **忽视效率评估**：只关注检索次数，不统计 LM 调用次数（后者可能更昂贵）
- **缺乏与不确定性估计方法的比较**：如 Mean Entropy 等成熟方法从未在自适应检索场景中系统评测
- **缺乏自知能力评估**：不测量方法判断"是否需要检索"的能力本身

本文通过全面、统一的评测框架弥合了这些差距。

## 方法详解

### 整体框架

统一评测 35 种方法在 6 个数据集上的表现，使用 10 种指标覆盖三个维度：QA 性能、自知能力和效率。将不确定性估计方法接入 AdaptiveRAG 的自适应检索框架，使之具有可比性。

### 关键设计

1. **端到端自适应检索方法（8种）**：

    - **IRCoT**：在思维链推理中动态添加检索段落，直到生成答案
    - **AdaptiveRAG**：用 T5-large 分类器预测三种结果（不检索/单次检索/多次检索）
    - **FLARE**：当 token 概率低于阈值时触发检索，重新生成响应
    - **DRAGIN**：类似 FLARE 但过滤停用词，用注意力权重重构查询
    - **Rowen**：基于一致性（跨语言/跨模型）来判断是否检索
    - **SeaKR**：用不确定性模块监控内部状态，触发检索并重排序候选片段

2. **不确定性估计方法（27种，重点展示5种）**：

    - **Lexical Similarity**：基于采样响应间的平均相似度
    - **Max/Mean Entropy**：计算每个 token 的熵并用最大值/均值聚合
    - **SAR**：基于 token 相关性加权的熵聚合
    - **EigValLaplacian**：构建采样响应的加权图，计算 Laplacian 特征值之和
    - 不确定性分数用分类器转化为检索决策

3. **评测框架**：

    - **QA 指标**：In-Accuracy (InAcc)、Exact Match (EM)、F1
    - **效率指标**：检索调用次数 (RC)、LM 调用次数 (LMC)
    - **自知能力指标**：ROC-AUC、Spearman 相关系数、准确率、过度自信率、不足自信率
    - **跨数据集排名聚合**：使用逆向排名平均来公平比较

### 损失函数 / 训练策略

- 所有方法统一使用 LLaMA 3.1-8b-instruct 模型
- 统一使用 BM25 + Elasticsearch 作为检索器
- 不确定性方法的分类器在训练集上训练，选择最佳分类器报告测试集结果
- 基线方法保留各自原始设置

## 实验关键数据

### 主实验

**QA 性能与效率（部分关键方法）**：

| 方法 | NQ InAcc | NQ LMC | NQ RC | HotPot InAcc | HotPot LMC | HotPot RC |
|------|----------|--------|-------|-------------|------------|-----------|
| Never RAG | 0.446 | 1.0 | 0.00 | 0.286 | 1.0 | 0.00 |
| Always RAG | 0.496 | 1.0 | 1.00 | 0.410 | 1.0 | 1.00 |
| IRCoT | 0.478 | 2.7 | 2.70 | 0.438 | 4.4 | 4.38 |
| DRAGIN | 0.480 | 4.5 | 2.24 | 0.430 | 5.1 | 2.56 |
| RowenHybrid | 0.494 | **55.0** | 7.27 | 0.354 | **59.8** | 7.63 |
| **Mean Entropy** | 0.498 | **1.9** | **0.88** | 0.410 | **2.0** | **0.99** |
| **Best UE** | **0.512** | **1.8** | **0.81** | **0.414** | **2.0** | **0.99** |
| Ideal Oracle | 0.608 | 1.6 | 0.55 | 0.460 | 1.7 | 0.71 |

### 消融实验

**自知能力评估（ROC-AUC）**：

| 方法 | NQ | SQUAD | TQA | 2Wiki | HotPot | Musique |
|------|-----|-------|-----|-------|--------|---------|
| AdaptiveRAG | 0.54 | 0.58 | 0.49 | 0.71 | 0.62 | 0.64 |
| FLARE | 0.59 | 0.58 | 0.57 | 0.62 | 0.54 | 0.51 |
| SeaKR | 0.64 | **0.77** | **0.78** | 0.37 | 0.55 | 0.56 |
| **Max Entropy** | 0.73 | 0.72 | 0.72 | **0.73** | **0.66** | **0.68** |

### 关键发现

1. **不确定性估计方法在效率上具有压倒性优势**：每个问题仅需 ~2 次 LM 调用和 <1 次检索调用，而 RowenHybrid 需要 55-80 次 LM 调用
2. **QA 性能上不确定性方法与复杂流水线相当**：Best UE 在单跳数据集上甚至优于大多数端到端方法
3. **自知能力方面不确定性方法一致更优**：Max Entropy 的 ROC-AUC 在多数数据集上显著优于所有端到端方法
4. **没有任何单一方法在所有维度上占优**：效率≠性能≠自知能力，需要根据场景选择
5. **当前方法距离理想性能仍有明显差距**：Ideal Oracle 显示出可观的改进空间

## 亮点与洞察

- **框架性贡献突出**：首次将 27 种不确定性估计方法系统引入自适应检索场景，填补了重要的比较空白
- **挑战了"复杂等于更好"的假设**：简单的 Mean Entropy 在很多场景下优于精心设计的多步流水线
- **LM 调用次数是被忽视的关键成本**：Rowen 系列方法每个问题需要 30-80 次 LM 调用，在使用商业 API 时成本惊人
- **多维评估视角**：首次将 QA 性能、效率和自知能力三个维度统一评测，为研究者提供全景式参考
- **OOD 分析**：进一步分析了不确定性方法在分布外场景下的表现，增加了实用价值

## 局限与展望

- 仅使用了 LLaMA 3.1-8b-instruct 一个模型，不同模型可能有不同的最佳方法
- 仅使用 BM25 作为检索器，更强的检索器（如密集检索）可能改变方法间的相对排名
- 不确定性方法需要训练分类器来决定阈值，引入了额外的数据和调参需求
- 自知能力的定义依赖于 In-Accuracy 的二值标签，可能过于粗糙
- 未考虑检索到的文档质量对最终回答的影响——差的检索可能使"Always RAG"低估

## 相关工作与启发

- 统一了两个此前分离的研究领域：自适应 RAG（Su et al., 2024; Jeong et al., 2024）和不确定性估计（Fadeeva et al., 2023; Duan et al., 2023）
- 为 LLM 自知能力研究（Yin et al., 2023, 2024）提供了实用的评估框架
- 实际启发：在资源受限场景下，与其构建复杂的自适应检索流水线，不如直接使用 Mean Entropy 等简单方法

## 评分

- 新颖性: ⭐⭐⭐ 方法本身创新有限，核心贡献在于全面的比较框架和反直觉的发现
- 实验充分度: ⭐⭐⭐⭐⭐ 35 种方法、6 个数据集、10 种指标、OOD 分析、分类器复杂度分析
- 写作质量: ⭐⭐⭐⭐ 论文组织清晰，Figure 1 的多维可视化非常直观
- 价值: ⭐⭐⭐⭐⭐ 对自适应检索领域的系统化评测具有重要参考价值，结论对实践有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search](arise_risk_adaptive_search.md)
- [\[ACL 2025\] Micro-Act: Mitigate Knowledge Conflict in QA via Actionable Self-Reasoning](micro_act_knowledge_conflict_reasoning.md)
- [\[ACL 2025\] Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate](improve_rule_retrieval_and_reasoning_with_self-induction_and_relevance_reestimat.md)
- [\[ACL 2025\] ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Clause Retrieval](acord_an_expert-annotated_retrieval_dataset_for_legal_contract_drafting.md)
- [\[ACL 2025\] Uni-Retrieval: A Multi-Style Retrieval Framework for STEM's Education](uni-retrieval_a_multi-style_retrieval_framework_for_stems_education.md)

</div>

<!-- RELATED:END -->
