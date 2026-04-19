---
title: >-
  [论文解读] OLMoTrace: Tracing Language Model Outputs Back to Trillions of Training Tokens
description: >-
  [ACL 2025][LLM/NLP][训练数据溯源] 提出OLMoTrace——首个能在实时（平均4.5秒）将语言模型输出逐字追溯到其完整多万亿token训练数据的系统，基于扩展的infini-gram引擎通过后缀数组索引实现高效精确匹配，支持事实核查、创意溯源和数学能力追踪等应用场景。
tags:
  - ACL 2025
  - LLM/NLP
  - 训练数据溯源
  - 逐字匹配
  - 后缀数组
  - infini-gram
  - 开放LM
---

# OLMoTrace: Tracing Language Model Outputs Back to Trillions of Training Tokens

**会议**: ACL 2025  
**arXiv**: [2504.07096](https://arxiv.org/abs/2504.07096)  
**代码**: [allenai/infinigram-api](https://github.com/allenai/infinigram-api)  
**领域**: LLM可解释性/数据溯源  
**关键词**: 训练数据溯源, 逐字匹配, 后缀数组, infini-gram, 开放LM  

## 一句话总结

提出OLMoTrace——首个能在实时（平均4.5秒）将语言模型输出逐字追溯到其完整多万亿token训练数据的系统，基于扩展的infini-gram引擎通过后缀数组索引实现高效精确匹配，支持事实核查、创意溯源和数学能力追踪等应用场景。

## 研究背景与动机

**领域现状**: 随着LM在高风险场景中的采用增加，理解模型为何生成特定响应变得至关重要。全开放LM（如OLMo）使训练数据可访问成为可能，但现有行为追踪方法（如影响函数）因计算开销过大，无法扩展到万亿token级别的训练数据。

**现有痛点**: (1) 影响函数（Koh & Liang 2017）需要梯度信息，在万亿token规模上计算不可行；(2) 现有方法如RAG是在生成前检索以提升生成质量，而非生成后追溯以理解行为来源；(3) 搜索引擎检索的是实时网页索引，而非LM的实际训练数据，无法建立LM行为与数据的直接联系。

**核心矛盾**: 训练数据规模（4.6万亿token）与实时溯源需求之间的矛盾——需要在海量数据中快速找到与LM输出精确匹配的文本片段。

**本文目标**: 如何在实时交互条件下，从万亿token的训练数据中找到与LM输出逐字匹配的训练文档，帮助用户理解LM行为。

**切入角度**: 利用后缀数组对训练数据进行预排序索引，将精确匹配查找的时间复杂度降至 $O(L \log N)$（并行化后延迟 $O(\log N)$），使实时溯源成为可能。

**核心 idea**: 基于infini-gram后缀数组索引实现对万亿token训练数据的实时逐字匹配溯源。

## 方法详解

### 整体框架

OLMoTrace的推理流水线包含5个步骤：(1) 找到LM输出中所有在训练数据中出现的最大匹配片段；(2) 过滤保留长且独特的片段；(3) 检索包含这些片段的训练文档；(4) 合并重叠片段和同源文档；(5) 按BM25相关性对文档重排序并着色展示。系统部署在仅CPU节点上，索引存储在高IOPS SSD磁盘上。

### 关键设计

1. **快速最大匹配片段计算**
    - **功能**: 在万亿token训练数据中找到LM输出的所有最大匹配片段
    - **核心思路**: 对LM输出的每个后缀，通过infini-gram的Find查询在后缀数组上二分搜索，仅需一次查询即可找到最长匹配前缀（利用SA中相邻元素的最长公共前缀性质）；所有后缀的查询完全并行化，总时间复杂度 $O(L \log N)$
    - **设计动机**: 朴素方法需枚举 $O(L^2)$ 个子串并在 $N > 10^{12}$ 的数据中搜索；通过后缀数组的有序性，每个位置只需一次Find查询（而非二分搜索O(logL)次），实现了从 $O(L^2 N)$ 到 $O(L \log N)$ 的加速

2. **Span Unigram Probability过滤**
    - **功能**: 从大量匹配片段中筛选出最"有趣"的片段
    - **核心思路**: 计算每个片段的unigram概率（各token的unigram概率之积），保留概率最低的K个片段（$K = \lceil 0.05 \times L \rceil$）；低unigram概率意味着片段较长且含非常见词
    - **设计动机**: 相比简单取最长片段，unigram概率指标在检索文档相关性上表现更好，因为它同时考虑了长度和内容独特性

3. **BM25相关性排序与着色**
    - **功能**: 优先展示与LM输出最相关的训练文档
    - **核心思路**: 将检索到的文档集合视为"语料库"，用户prompt+LM响应视为"查询"，计算BM25分数进行排序；将分数分为高/中/低三个等级，用不同饱和度的颜色展示
    - **设计动机**: BM25与人类相关性判断有较高一致性（Spearman 0.73），且仅需CPU计算，满足实时需求

### 损失函数 / 训练策略

本文为系统工作，不涉及模型训练。核心技术挑战在于工程优化：12个数据分片，每个分片上限500B token；索引文件存储在40TB SSD（80,000 IOPS）上；关闭预取以优化吞吐量；采用批量文档检索减少延迟。

## 实验关键数据

### 主实验

系统性能与匹配统计（98个对话测试）：

| 指标 | 数值 |
|------|------|
| 训练数据规模 | 32亿文档，4.6万亿token |
| 平均响应长度 | 458 tokens |
| 平均推理延迟 | 4.46 秒 |
| 匹配片段平均长度 | 10.4 tokens |
| 高相关性文档占比 | 14% |
| 高相关性片段占比 | 19% |
| 文档来源：预训练数据 | 96.7% |
| 文档来源：中间训练 | 0.9% |
| 文档来源：后训练(SFT+DPO) | 2.4% |

### 消融实验

文档相关性评估对比：

| 评估方法 | Top-1文档 Avg Score | Top-5文档 Avg Score |
|---------|--------------------|--------------------|
| 人工评估 | 1.90 | 1.43 |
| LLM-as-Judge (初始设置) | 1.73 | 1.28 |
| LLM-as-Judge (优化后) | 1.82 | 1.50 |
| 人工与LLM评估Spearman相关 | 0.73 | - |

（评分标准: 0=无关, 1=宽泛相关, 2=同主题不同上下文, 3=直接匹配）

### 关键发现

1. 匹配片段平均10.4 tokens，说明LM输出与训练数据之间存在大量长程逐字重复
2. 96.7%的检索文档来自预训练数据，但数学重题目可能更多检索到SFT/RLVR数据
3. BM25排序在文档相关性上与人类判断合理一致（Spearman=0.73）
4. 系统能检测到"看似创意"实为训练数据中已出现的表达（如Tolkien风格故事中的"I'm going on an adventure"）
5. 数学解题步骤可精确追溯到后训练数据中的逐字匹配

## 亮点与洞察

- **首个万亿级实时溯源系统**: 将推理延迟控制在~4.5秒，达到交互级别
- **算法核心精巧**: 利用后缀数组相邻元素性质将每个位置的查询从O(logL)降到O(1)
- **全开源全开放**: 系统、模型、训练数据完全公开，树立了LM透明度标杆
- **应用启发性强**: 事实核查、创意溯源、数学推理溯源三个案例展示了系统的多面价值
- **负责任的设计**: 考虑了版权、PII、有毒内容的缓解措施

## 局限与展望

- **仅支持逐字匹配**: 不能检测语义相似但措辞不同的训练数据影响
- **非因果解释**: 匹配的训练文档不应被解释为对LM输出有因果效应
- **仅支持OLMo系列**: 需要完整训练数据访问权限，无法应用于闭源模型
- **相关性评估有限**: Top-1文档平均1.90分（0-3分制），仍有改善空间
- 可扩展方向：结合影响函数或梯度信息提供更强的因果溯源

## 相关工作与启发

- **infini-gram**: 本系统的核心引擎，将后缀数组扩展到万亿token规模
- **与RAG的本质区别**: RAG是生成前检索以增强生成，OLMoTrace是生成后溯源以理解行为
- **ORCA**: 使用影响函数追踪预训练数据，但计算代价远高于本方法
- 该工作为"LM行为与训练数据的关系"这一根本问题提供了可操作的工具

## 评分

- **新颖性**: 4/5 — 首个万亿规模实时溯源系统
- **技术深度**: 4/5 — 后缀数组算法优化精巧，系统工程完整
- **实验充分度**: 3/5 — 以系统展示为主，量化评估较初步
- **实用性**: 5/5 — 已部署可用，完全开源
- **综合评分**: 4/5

<!-- RELATED:START -->

## 相关论文

- [Training Language Model to Critique for Better Refinement](training_language_model_to_critique_for_better_refinement.md)
- [A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](a_survey_on_efficient_large_language.md)
- [Do Language Models Understand the Cognitive Tasks Given to Them? Investigations with the N-Back Paradigm](do_language_models_understand_the_cognitive_tasks_given_to_them_investigations_w.md)
- [Cool-Fusion: Fuse Large Language Models without Training](cool-fusion_fuse_large_language_models_without_training.md)
- [Self-Training Elicits Concise Reasoning in Large Language Models](self-training_elicits_concise_reasoning_in_large_language_models.md)

<!-- RELATED:END -->
