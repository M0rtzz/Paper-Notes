---
title: >-
  [论文解读] Logical Consistency is Vital: Neural-Symbolic Information Retrieval for Negative-Constraint Queries
description: >-
  [ACL 2025][信息检索] 提出 NS-IR，通过将自然语言查询和文档转换为一阶逻辑（FOL），利用逻辑对齐和连接词约束两项技术优化稠密检索嵌入，显著提升了负约束查询等复杂逻辑检索场景的性能。
tags:
  - ACL 2025
  - 信息检索
  - 一阶逻辑
  - 负约束查询
  - 最优传输
  - 神经符号
---

# Logical Consistency is Vital: Neural-Symbolic Information Retrieval for Negative-Constraint Queries

**会议**: ACL 2025  
**arXiv**: [2505.22299](https://arxiv.org/abs/2505.22299)  
**代码**: [有](https://github.com/xgl-git/NS-IR-main)  
**领域**: NLP / 信息检索  
**关键词**: 信息检索, 一阶逻辑, 负约束查询, 最优传输, 神经符号

## 一句话总结

提出 NS-IR，通过将自然语言查询和文档转换为一阶逻辑（FOL），利用逻辑对齐和连接词约束两项技术优化稠密检索嵌入，显著提升了负约束查询等复杂逻辑检索场景的性能。

## 研究背景与动机

当前稠密检索（Dense Retrieval）主要依赖查询和文档之间的词共现来计算嵌入相似度，忽略了查询的真实意图。这在普通查询中已有不足，在面对**负约束查询**（如"介绍姚明但不要提篮球"）时更是灾难性的——因为包含被排除关键词的文档反而会获得更高的相似度分数。

例如搜索引擎处理"不涉及 prompt engineering 的 RAG 方法"时，返回的文档往往大量包含"prompt engineering"这一被排除的关键词。核心问题在于现有检索方法缺乏对**逻辑一致性（logical consistency）**的建模能力。

一阶逻辑（FOL）可以清晰地表达否定语义（如 `¬InvolvesPromptEngineering(x)`），因此作者提出将 FOL 的逻辑语义融入自然语言嵌入中，从而同时考量语义相似度与逻辑一致性。

## 方法详解

### 整体框架

NS-IR 是一个重排序（reranking）框架，流程如下：

1. 使用 BGE 稠密检索器对查询进行初始检索，获取 Top-K 候选文档
2. 利用 LLM（GPT-4o）将查询和文档翻译为一阶逻辑（NL-FOL 翻译）
3. 再次使用 BGE 编码 NL 和 FOL 的词级嵌入
4. 应用**逻辑对齐（Logic Alignment）**和**连接词约束（Connective Constraint）**重新计算相似度分数
5. 根据新分数重排候选文档

### 关键设计

#### 1. 逻辑对齐（Logic Alignment, LA）

**功能**：将 FOL 的整体逻辑语义融入 NL 的表示中。

**核心思路**：基于最优传输（Optimal Transport, OT）的方法，衡量 NL 和 FOL 嵌入之间的分布差异。具体步骤为：

- 用 BGE 分别获取 NL 查询词嵌入 H 和 FOL 查询词嵌入 Z
- 计算成本矩阵 C（余弦距离）
- 用最优传输求解对齐矩阵 P（通过线性规划）
- 将 H、Z、P 和 CLS token 嵌入整合，更新 CLS 嵌入：$\mathscr{h}^{cls} = H^T \cdot P \cdot Z \cdot h^{cls}$

**设计动机**：通过对齐矩阵作为中间桥梁，综合 FOL 和 NL 的词分布以及上下文特征。对查询和文档分别做同样处理后，计算 score₁。

#### 2. 连接词约束（Connective Constraint, CC）

**功能**：精确反映 FOL 中不同词（尤其是逻辑连接词）对 NL 词的影响。

**核心思路**：让 FOL 中的不同词对 NL 中的词施加不同的注意力权重。关键创新在于 σ 函数的设计：

- 当 FOL 词是非否定连接词且与 NL 词无对齐关系时，σ = 1（正向增强）
- 当 FOL 词是否定连接词 ¬ 且无对齐关系时，σ = -1（负向约束，强调排除语义）
- 其他情况 σ = 0

**设计动机**：否定连接词通过减法操作施加负约束语义，使嵌入能够远离被排除的概念。

### 损失函数 / 训练策略

NS-IR 是零样本（zero-shot）方法，无需训练。最终文档推荐分数为两项技术分数之和：

$$\text{score} = \text{score}_1 + \text{score}_2$$

## 实验关键数据

### 主实验（表格）

**低资源检索与网络搜索（nDCG@10）**：

| 方法 | SciFact | ArguAna | TREC-COVID | FiQA | DBPedia | NFCorpus | DL'19 | DL'20 |
|------|---------|---------|------------|------|---------|----------|-------|-------|
| BM25 | 67.1 | 43.2 | 55.5 | 25.1 | 26.1 | 31.4 | 55.4 | 50.1 |
| BGE | 71.3 | 48.4 | 75.3 | 30.6 | 38.9 | 35.4 | 64.4 | 63.4 |
| HyDE | 71.9 | 49.6 | 78.4 | 31.3 | 38.7 | 37.3 | 67.3 | 66.8 |
| InteR | 72.1 | 50.9 | 79.2 | 33.5 | 42.1 | 39.5 | 69.7 | 67.5 |
| **NS-IR (GPT-4o)** | **75.8** | **55.1** | **81.8** | **38.4** | **46.1** | **40.7** | **68.4** | **70.5** |

**负约束查询数据集 NegConstraint（nDCG@10）**：

| 方法 | A-a | (A-a)∪B | (A-a)∪(B-b) | Total |
|------|-----|---------|-------------|-------|
| BM25 | 34.6 | 34.7 | 31.5 | 33.7 |
| BGE | 40.5 | 36.8 | 34.8 | 40.8 |
| HyDE | 55.3 | 51.5 | 50.6 | 53.1 |
| **NS-IR (GPT-4o)** | **57.9** | **54.2** | **53.7** | **56.5** |

### 消融实验（表格）

| 方法 | NegConstraint (MAP) | NegConstraint (nDCG@10) |
|------|--------------------|-----------------------|
| BGE | 36.3 | 40.8 |
| BGE + LA | 40.8 | 47.6 |
| BGE + CC | 47.8 | 46.9 |
| NS-IR (LogicLLaMA) | 50.7 | 55.2 |
| NS-IR (GPT-4o) | 53.3 | 56.5 |

### 关键发现

1. NS-IR 在所有标准检索基准上均超越 SOTA 零样本方法（相比 BGE 提升约 10%）
2. 在负约束查询场景下，CC 比 LA 更有效，表明连接词级别的细粒度逻辑操作对处理否定约束更关键
3. NS-IR 效果不依赖特定检索器，在 bge-small、bge-base 和 Contriever 上都一致优于基线
4. Top-K = 100 时性能稳定，进一步增大 K 无显著收益
5. t-SNE 可视化证实 LA 和 CC 都能使查询嵌入更接近正文档、远离负文档

## 亮点与洞察

- **新颖的问题定义**：首次系统研究负约束查询（"不要包含 X"）这一实际但被忽视的检索场景
- **逻辑-神经融合思路**：不重新训练模型，而是在推理阶段用 FOL 优化嵌入，方法优雅且即插即用
- **σ 函数设计精妙**：用 -1/0/+1 三值机制将逻辑否定直接编码为嵌入空间中的方向操作
- **NegConstraint 数据集**：填补了负约束查询评测的空白，包含三种递增复杂度的查询类型

## 局限与展望

1. 依赖 OpenAI API 进行 NL-FOL 翻译，成本较高（LogicLLaMA 效果较弱）
2. 只对 Top-K 文档做重排序，可能损失部分召回率
3. NL-FOL 翻译使用统一 prompt，未针对不同领域优化
4. 仅关注否定约束，尚未探索更复杂的集合运算（并集、交集、补集等）
5. 推理速度受 LLM 调用和 OT 求解 限制，实时性待提升

## 相关工作与启发

- 延续 HyDE、InteR 等查询改写思路，但从逻辑语义角度切入更本质
- 最优传输在 NLP 中的应用（句子对齐、跨模态表示）为嵌入空间对齐提供了理论基础
- NL-FOL 翻译近年因 LLM 能力提升而重新活跃，本文将其引入检索领域是新尝试

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将 FOL 引入信息检索、针对负约束查询的问题定义和解决方案都较新颖
- **实验充分度**: ⭐⭐⭐⭐ — 6 个 BEIR 数据集 + 2 个 Web Search + 自建 NegConstraint，消融和可视化充分
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机到方法到实验逻辑连贯，图示直观
- **价值**: ⭐⭐⭐⭐ — 负约束查询是真实场景痛点，方法零样本可用，NegConstraint 数据集有贡献价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering](neusym_rag_pdf_qa.md)
- [\[AAAI 2026\] ComLQ: Benchmarking Complex Logical Queries in Information Retrieval](../../AAAI2026/information_retrieval/comlq_benchmarking_complex_logical_queries_in_information_retrieval.md)
- [\[ACL 2025\] Any Information Is Just Worth One Single Screenshot: Unifying Search With Visualized Information Retrieval](any_information_is_just_worth_one_single_screenshot_unifying_search_with_visuali.md)
- [\[ACL 2025\] CoIR: A Comprehensive Benchmark for Code Information Retrieval Models](coir_a_comprehensive_benchmark_for_code_information_retrieval_models.md)
- [\[ACL 2025\] AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark](air-bench_automated_heterogeneous_information_retrieval_benchmark.md)

</div>

<!-- RELATED:END -->
