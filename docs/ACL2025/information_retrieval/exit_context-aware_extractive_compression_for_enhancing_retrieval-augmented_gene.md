---
title: >-
  [论文解读] EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation
description: >-
  [ACL 2025][RAG] 提出 EXIT——一种抽取式上下文压缩框架，通过上下文感知的句子级二分类并行选取与查询相关的句子，在 QA 准确率和推理延迟上同时优于现有的抽生式和抽取式压缩方法。
tags:
  - ACL 2025
  - RAG
  - 上下文压缩
  - 抽取式压缩
  - 句子分类
  - 推理效率
---

# EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation

**会议**: ACL 2025  
**arXiv**: [2412.12559](https://arxiv.org/abs/2412.12559)  
**代码**: [ThisIsHwang/EXIT](https://github.com/ThisIsHwang/EXIT) (有)  
**领域**: NLP-检索增强生成  
**关键词**: RAG、上下文压缩、抽取式压缩、句子分类、推理效率  

## 一句话总结

提出 EXIT——一种抽取式上下文压缩框架，通过上下文感知的句子级二分类并行选取与查询相关的句子，在 QA 准确率和推理延迟上同时优于现有的抽生式和抽取式压缩方法。

## 研究背景与动机

- **领域现状**: 检索增强生成（RAG）通过检索外部文档来增强 LLM 的回答质量，但检索模型有时无法将最相关的文档排在前面，且增加检索文档数量会带来延迟增加和准确率下降的问题
- **现有痛点**: (1) **抽生式压缩**（如 CompAct、Refiner）通过自回归生成摘要来压缩，但压缩本身的延迟极高（处理 5 篇文档需 8+ 秒）；(2) **抽取式压缩**（如 RECOMP-Extr）虽然快，但采用固定的、上下文无关的选择策略，无法适应查询复杂度和检索质量的变化
- **核心矛盾**: 抽生式方法准确但慢，抽取式方法快但效果差——需要一种既快又准的压缩方法
- **本文目标**: 设计一种既能保持抽取式压缩的速度优势，又能通过上下文感知和自适应选择来提升压缩质量的方法
- **切入角度**: 将上下文压缩重新定义为"句子级二分类问题"，利用输入文档的完整上下文进行并行化的相关性判断
- **核心 idea**: 把 RAG 的上下文压缩变成可并行的句子分类问题，每个句子在完整文档上下文中判断是否与查询相关

## 方法详解

### 整体框架

EXIT 包含三个步骤：(1) 将检索到的文档拆分为句子；(2) 对每个句子进行并行的二分类（"Yes"/"No"），评估其在完整文档上下文中与查询的相关性；(3) 按原始顺序重组被选中的句子。

### 关键设计

#### 1. 句子级分解
- **功能**: 将每篇检索文档拆分为句子
- **核心思路**: 使用 SpaCy 的规则化分句器将文档 $d_i$ 分解为句子集合 $S_i = \{s_{i1}, s_{i2}, \dots, s_{in}\}$
- **设计动机**: 句子级操作避免了 token 级压缩（如 LLMLingua）可能导致的关键短语断裂和实体关系破坏，保持了语法连贯性和语义完整性

#### 2. 上下文感知的相关性分类
- **功能**: 利用完整文档上下文评估每个句子的相关性
- **核心思路**: 给定查询 $q$、文档 $d_i$ 和句子 $s_{ij}$，用评估模型预测相关性分数：

$$r_{ij} = \frac{P(\text{"Yes"} | q, d_i, s_{ij})}{P(\text{"Yes"} | q, d_i, s_{ij}) + P(\text{"No"} | q, d_i, s_{ij})}$$

仅需预测一个 token（"Yes"/"No"的概率），所有句子可并行处理
- **设计动机**: (1) 将完整文档 $d_i$ 作为上下文传入是因为理解一个句子通常需要更广泛的文档背景；(2) 单 token 预测比多 token 自回归生成高效得多

#### 3. 自适应阈值选择
- **功能**: 根据查询和检索质量动态选择句子数量
- **核心思路**: 设置阈值 $\tau = 0.5$，保留 $r_{ij} > \tau$ 的句子。压缩后的文档 $D'$ 包含可变数量的句子
- **设计动机**: 不同查询的信息需求不同——简单问题可能只需少量句子，复杂的多跳推理则需要更多证据

### 损失函数

二元交叉熵损失：

$$\mathcal{L} = -\mathbb{1}_{l=\text{"Yes"}} \log P(\text{"Yes"}) - (1 - \mathbb{1}_{l=\text{"Yes"}}) \log P(\text{"No"})$$

训练数据包含三种样本：正样本（包含答案证据的句子）、困难负样本（同文档中的其他句子）、随机负样本（无关文档中的句子）。

### 训练策略

- 仅使用 HotpotQA 训练集（有句子级标注），但能泛化到 NQ、TQA、2WIKI
- 分类器基于 Gemma-2B-it
- 推理时 temperature=0.0

## 实验关键数据

### 主实验：4 个 QA 数据集上的表现（Llama3.1-8B-Instruct）

| 压缩方法 | 类型 | NQ-EM | TQA-EM | HQA-EM | 2WIKI-EM | Avg-EM | Avg延迟(s) |
|---------|------|-------|--------|--------|---------|--------|----------|
| 原始文档 | - | 34.6 | 58.8 | 28.1 | 16.1 | 34.4 | 1.0 |
| CompAct | Abs | 32.9 | 58.1 | 28.8 | 16.8 | 34.2 | **8.4** |
| RECOMP-Extr | Ext | 34.6 | 56.5 | 23.4 | 11.2 | 31.4 | 0.5 |
| LongLLMLingua | Ext | 30.2 | 59.4 | 28.0 | 21.5 | 34.8 | 0.9 |
| **EXIT** | Ext | **35.9** | **60.8** | **30.6** | **24.2** | **37.9** | **0.8** |

- EXIT 在 Avg-EM 上超越所有方法（包括未压缩的原始文档），同时延迟仅 0.8s

### 70B 模型上的表现

| 压缩方法 | Avg-EM | Avg-F1 | Avg延迟(s) |
|---------|--------|--------|----------|
| 原始文档 | 38.8 | 48.7 | 8.4 |
| **EXIT** | **42.5** | **52.0** | **3.5** |

- 70B 模型上 EXIT 提升更明显：EM +3.7、F1 +3.3，延迟减少 58%

### 消融实验

| 配置 | EM | F1 | Token数 |
|------|----|----|---------|
| **完整 EXIT** | **31.6** | **42.6** | 195.1 |
| 仅正样本+困难负样本 | 30.0 | 41.3 | 286.8 |
| 仅正样本+随机负样本 | 29.8 | 40.9 | 404.6 |
| 固定选 2 句 | 29.4 | 40.7 | 91.0 |
| 固定选 4 句 | 30.2 | 41.4 | 166.5 |
| 无文档上下文 | 30.4 | 42.3 | 157.4 |

### 关键发现

1. **抽取式也能超越原文档**: EXIT 证明精准的句子选择可以去除噪声信息，反而提升 QA 性能
2. **延迟 vs FLOPS 的分离**: EXIT 的 TFLOPs（35.44）高于某些方法，但因为可并行处理，实际延迟更低——说明效率不能仅看计算量，还要看并行度
3. **自适应选择至关重要**: 固定句数选择（2句/4句）均不如自适应阈值
4. **上下文感知提升精度**: 移除文档上下文后 EM 下降 1.2，证实完整上下文对判断句子相关性至关重要
5. **鲁棒性**: 随着检索文档数 $k$ 从 1 增加到 30，EXIT 的 EM 持续提升（28.2→33.1），而其他方法出现性能下降

## 亮点与洞察

- **问题重新定义的优雅**: 将"上下文压缩"重新定义为"句子分类"，自然获得并行化和上下文感知两大优势
- **端到端延迟** 的分析视角新颖——揭示了"压缩时间 + 读取时间"才是真正的效率指标，仅看 token 数是不够的
- **即插即用**设计：EXIT 与具体的检索器和读取器无关，可无修改地嵌入任何 RAG 管道
- 仅用 HotpotQA 训练就能泛化到域外数据集（NQ、TQA、2WIKI），泛化能力强

## 局限与展望

- 阈值 $\tau = 0.5$ 是固定的，可探索自适应阈值（如根据检索质量动态调整）
- 分类器（Gemma-2B）需要额外的前向传播，当查询量极大时可能成为瓶颈
- 句子分割依赖分句器质量，对于非规范文本（如表格、列表）可能效果不佳
- 未探索多跳推理中句子间的依赖关系建模
- 训练仅依赖 HotpotQA，更多样的训练数据可能进一步提升泛化

## 相关工作与启发

- **RECOMP (Xu et al., 2024)**: 同时提出了抽生式和抽取式变体，是主要基线
- **CompAct / Refiner**: 抽生式压缩的代表，延迟是其致命弱点
- **LLMLingua 系列**: token 级压缩可能破坏语义连贯性
- **Flamingo / RAG (Lewis et al., 2020)**: RAG 范式的基础工作
- **启发**: 句子级粒度可能是"语义完整性"和"压缩效率"之间的最佳平衡点

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性** ⭐⭐⭐⭐: 将压缩重构为分类问题的思路优雅，上下文感知设计有效
- **实验** ⭐⭐⭐⭐⭐: 4 个数据集、2 种 reader 规模、详细消融和延迟分析
- **实用性** ⭐⭐⭐⭐⭐: 即插即用、延迟低、效果好，工业部署价值高
- **写作** ⭐⭐⭐⭐: 图表清晰，延迟分析直观

<!-- RELATED:START -->

## 相关论文

- [A Reality Check on Context Utilisation for Retrieval-Augmented Generation](a_reality_check_on_context_utilisation_for_retrieval-augmented_generation.md)
- [Hierarchical Document Refinement for Long-context Retrieval-augmented Generation](hierarchical_document_refinement_for_long-context_retrieval-augmented_generation.md)
- [Typed-RAG: Type-Aware Decomposition of Non-Factoid Questions for Retrieval-Augmented Generation](typed-rag_type-aware_decomposition_of_non-factoid_questions_for_retrieval-augmen.md)
- [Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](gor_rag_long_context_summary.md)
- [FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation](faithfulrag_fact_level_conflict.md)

<!-- RELATED:END -->
