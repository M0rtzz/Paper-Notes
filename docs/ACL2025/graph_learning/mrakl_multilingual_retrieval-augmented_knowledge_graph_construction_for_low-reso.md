---
title: >-
  [论文解读] mRAKL: Multilingual Retrieval-Augmented Knowledge Graph Construction for Low-Resourced Languages
description: >-
  [ACL 2025][图学习][多语言知识图谱构建] 将多语言知识图谱构建（mKGC）重新定义为 QA 任务，提出基于 RAG 的 mRAKL 系统，利用非结构化单语数据作为检索源来克服低资源语言中结构化数据匮乏的困难，在 Tigrinya 和 Amharic 两种低资源语言上显著超越已有方法。
tags:
  - ACL 2025
  - 图学习
  - 多语言知识图谱构建
  - 检索增强生成
  - 低资源语言
  - 跨语言迁移
  - 知识图谱补全
---

# mRAKL: Multilingual Retrieval-Augmented Knowledge Graph Construction for Low-Resourced Languages

**会议**: ACL 2025  
**arXiv**: [2507.16011](https://arxiv.org/abs/2507.16011)  
**代码**: 即将发布  
**领域**: 图学习  
**关键词**: 多语言知识图谱构建, 检索增强生成, 低资源语言, 跨语言迁移, 知识图谱补全

## 一句话总结

将多语言知识图谱构建（mKGC）重新定义为 QA 任务，提出基于 RAG 的 mRAKL 系统，利用非结构化单语数据作为检索源来克服低资源语言中结构化数据匮乏的困难，在 Tigrinya 和 Amharic 两种低资源语言上显著超越已有方法。

## 研究背景与动机

**领域现状**: 知识图谱（KG）在问答、信息检索、语言模型增强等下游应用中至关重要，但大多数 KG 是不完整的，低资源语言中缺失信息更为严重。Wikidata 中仅 0.2% 的实体有 Amharic 标签。

**现有痛点**: 
   - 现有 mKGC 方法（如 KGT5）依赖大量结构化数据训练（如 52M 三元组），低资源语言根本不具备此条件
   - 基于 KG 嵌入的跨语言方法假设封闭世界，无法利用开放域自然语言知识
   - 预训练语言模型对低资源语言的参数化知识极度匮乏（GPT-4 在 Amharic 上 zero-shot H@1 仅 5.83%）

**核心矛盾**: 低资源语言缺乏结构化标注数据，但拥有相对更多的非结构化单语文本（如 Wikipedia 文章），如何利用这些非结构化数据来构建 KG。

**本文目标**: 为 Tigrinya（3.5k 三元组）和 Amharic（34k 三元组）这样的极低资源语言构建和补全知识图谱。

**切入角度**: 将 KG 三元组转换为 QA 对（head + relation → question, tail → answer），用 RAG 方式从 Wikipedia 检索相关段落辅助生成。

**核心 idea**: 用 RAG + 跨语言 QA 将非结构化单语数据转化为 KG 补全能力，弥补低资源语言结构化数据的不足。

## 方法详解

### 整体框架

mRAKL 由两个核心组件构成：
- **Retriever**: 从单语 Wikipedia 中检索与查询相关的句子作为上下文
- **Generator**: 基于检索到的上下文和模板化问题，生成尾实体作为答案

### 关键设计

1. **KG 到 QA 的转换**: 

    - 为 120 个关系手动准备四种语言的问题模板
    - 对每个三元组 $(h, r, t)$，将 head entity 填入关系模板得到问题，tail entity 作为答案
    - 例：三元组 (Surafel Dagnachew, place of birth, Ethiopia) → 问题"What is Surafel Dagnachew's place of birth?"

2. **跨语言实体对齐**: 

    - 在输入序列中使用语言标记 [C-LAN], [Q-LAN], [A-LAN] 指示上下文、问题和答案的语言
    - 支持跨语言链接预测：给定一种语言的 head + relation，预测另一种语言的 tail
    - 格式：`[C-LANt]C | [Q-LANt]Q? [A-LANt']`（上下文/问题语言 t，答案语言 t'可以不同）

3. **检索器设计**: 

    - **BM25**: 对四种语言分别建立单语 Wikipedia 索引
    - **LaBSE**: 多语言句嵌入模型，用对比损失微调（LaBSE 不包含 Tigrinya）
    - **(Im)perfect Retriever**: 上界实验——直接在 head entity 的 Wikipedia 文章中搜索包含 tail entity 的句子

4. **生成器训练**: 

    - 基础模型：AfriTeVa-base（包含 Tigrinya 和 Amharic 的预训练 T5 模型）
    - 用 LoRA 微调，交叉熵损失，beam search 解码，beam size=10
    - 四种训练设置：No-Context / Monolingual Self-Context / Multilingual Self-Context / Cross-Lingual Context

### 损失函数/训练策略

- 生成器使用标准交叉熵损失
- 不使用显式负采样
- 检索器 LaBSE 用对比损失微调

## 实验关键数据

### 参数化知识探测（Zero-shot H@1）

| 模型 | Tigrinya | Amharic |
|------|----------|---------|
| mT5 | - | 0.49 |
| AfriTeVa | 0.22 | 0.61 |
| Aya | 0.67 | 1.52 |
| GPT-4 | 2.23 | 5.83 |
| AfriTeVa (finetuned) | 5.13 | 29.15 |

### 主实验：单语链接预测

| 方法 | Tigrinya H@1 | Tigrinya H@10 | Amharic H@1 | Amharic H@10 |
|------|-------------|---------------|-------------|--------------|
| KGT5-No-Context | 6.91 | 28.57 | 32.58 | 52.57 |
| KGT5-Description | 5.80 | 23.44 | 32.91 | 43.32 |
| KGT5-One-Hop | 4.46 | 24.33 | 28.83 | 48.17 |
| mRAKL No-Context | 5.13 | 26.11 | 29.15 | 54.81 |
| **mRAKL Self-Context** | **11.83** | **34.59** | **41.37** | **61.87** |

### 跨语言链接预测（BM25, H@1）

| 目标语言 | Amharic 上下文 | Arabic 上下文 | English 上下文 | 平均 |
|----------|---------------|--------------|---------------|------|
| Tigrinya | 15.75 | 12.30 | 14.73 | 14.15 |
| Amharic | 38.52 | 33.58 | 38.22 | 35.27 |

### 关键发现

- **RAG 显著提升低资源语言 KGC**: mRAKL Self-Context 相比 KGT5 No-Context，Tigrinya H@1 提升 4.92 个百分点，Amharic 提升 8.79 个百分点
- **结构化上下文对低资源语言不适用**: KGT5-Description 和 KGT5-One-Hop 反而降低性能，因为低资源语言中实体描述和one-hop连接本身就缺失
- **跨语言迁移有效**: BM25 在所有设置中优于 LaBSE 检索器，且优于无上下文设置
- **同族语言更有利**: Amharic 作为 Tigrinya 的上下文语言时 H@1 最高（15.75），35.88% 的尾实体在两种语言中拼写相同
- **多语言训练提升小语种**: Multilingual Self-Context 对 Tigrinya 提升 4.69 个百分点（从 11.83 到 15.18）
- **文化/地域相关性**: Arabic 上下文在中东/亚洲相关查询上表现更好，English 在西方话题上更优

## 亮点与洞察

- **范式创新**: 将 KGC 转化为 QA + RAG，巧妙利用非结构化数据弥补低资源语言的结构化数据缺失
- **跨语言实体链接**: 通过在答案位置指定不同语言标记，隐式实现了跨语言实体对齐，无需显式对齐模型
- **模块化设计**: 检索器和生成器可独立优化，便于利用易获取的单语数据
- **真实低资源场景**: 3.5k 规模的 Tigrinya KG 是真正的低资源设置，具有很高的实际参考价值
- **迁移语言选择的洞察**: 语言家族和文化/地域相关性都影响迁移效果

## 局限与展望

- Tigrinya 数据量极小（3.5k 三元组、272 实体），结果可能不够稳定
- 检索器（BM25/LaBSE）的性能仍有很大提升空间，可尝试为低资源语言专门训练的检索器
- 问题模板需要人工为每种语言编写，扩展到更多语言时成本较高
- 实体覆盖偏差：迁移语言 KG 中缺少目标语言特有的实体（如 Eritrea 地区对 Tigrinya）
- Wikipedia 数据本身存在社会偏见，不同语言版本间的差异可能传导到 KG 中

## 相关工作与启发

- **KGT5** (Saxena et al., 2022): 将 KGC 建模为序列到序列任务的先驱工作
- **RAG** (Lewis et al., 2020): 为 LLM 注入外部知识的有效范式
- **AfriTeVa** (Ogundepo et al., 2022): 针对非洲语言的多语言 T5 模型
- 对其他低资源语言的 KG 构建有直接参考价值，如少数民族语言、方言等
- 启示：在数据匮乏情况下，充分利用非结构化数据 + 跨语言迁移是可行路线

## 评分

- **新颖性**: ⭐⭐⭐⭐ (KGC + RAG + QA 的组合是新颖的，特别是面向真正低资源语言)
- **实验充分度**: ⭐⭐⭐⭐ (多种检索器对比、跨语言设置、消融实验、定性分析全面)
- **写作质量**: ⭐⭐⭐⭐ (问题动机清晰，实验设置条理分明)
- **价值**: ⭐⭐⭐⭐ (为低资源语言 KG 构建提供了实用基线和数据集贡献)

<!-- RELATED:START -->

## 相关论文

- [SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation](simgrag_leveraging_similar_subgraphs_for_knowledge_graphs_driven_retrieval-augme.md)
- [Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation (K-RagRec)](kg_rag_recommendation.md)
- [GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](../../NeurIPS2025/graph_learning/gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)
- [Neural Graph Matching Improves Retrieval Augmented Generation in Molecular Machine Learning](../../ICML2025/graph_learning/neural_graph_matching_improves_retrieval_augmented_generation_in_molecular_machi.md)
- [ReMindRAG: Low-Cost LLM-Guided Knowledge Graph Traversal for Efficient RAG](../../NeurIPS2025/graph_learning/remindrag_low-cost_llm-guided_knowledge_graph_traversal_for_efficient_rag.md)

<!-- RELATED:END -->
