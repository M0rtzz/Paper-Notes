---
title: >-
  [论文解读] Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation (K-RagRec)
description: >-
  [ACL 2025][图学习][knowledge graph] 提出K-RagRec框架，通过从知识图谱中检索多跳子图为LLM推荐系统提供结构化、可靠的外部知识，结合基于流行度的选择性检索策略和GNN编码器，有效缓解LLM推荐中的幻觉和知识缺失问题。
tags:
  - ACL 2025
  - 图学习
  - knowledge graph
  - RAG
  - LLM recommendation
  - 图神经网络
  - sub-graph retrieval
---

# Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation (K-RagRec)

**会议**: ACL 2025  
**arXiv**: [2501.02226](https://arxiv.org/abs/2501.02226)  
**代码**: 未公开  
**领域**: 推荐系统/图学习  
**关键词**: knowledge graph, RAG, LLM recommendation, GNN, sub-graph retrieval  

## 一句话总结

提出K-RagRec框架，通过从知识图谱中检索多跳子图为LLM推荐系统提供结构化、可靠的外部知识，结合基于流行度的选择性检索策略和GNN编码器，有效缓解LLM推荐中的幻觉和知识缺失问题。

## 研究背景与动机

- **问题定义**：LLM推荐系统面临三大固有缺陷——(1) 幻觉问题（推荐不存在的物品）；(2) 知识过时（训练数据截止导致无法推荐新物品）；(3) 缺乏领域特定知识（推荐语料在预训练中有限）
- **RAG的局限**：传统文本RAG引入噪声和有害干扰，且忽略实体间的结构关系，导致LLM推理能力受限
- **知识图谱的优势**：KG提供结构化、事实性、可编辑的知识表示，是对抗幻觉的天然选择
- **技术挑战**：(1) 仅检索一阶邻居无法捕获高阶关系；(2) 无差别检索降低效率；(3) KG三元组的文本序列化无法充分利用结构信息

## 方法详解

### 整体框架

K-RagRec包含五个核心组件：(1) 多跳知识子图语义索引；(2) 基于流行度的选择性检索策略；(3) 知识子图检索；(4) 知识子图重排序；(5) 知识增强推荐生成。

### 关键设计

- **多跳子图索引**：用SentenceBERT编码KG节点和边的文本属性 → 用GNN聚合多跳邻居信息得到子图表示 $z_{g_o}$ → 存入向量数据库。$l$ 跳GNN表示等价于节点的 $l$ 跳邻域子图表示，实现从粗粒度到细粒度的灵活分块
- **流行度选择性检索**：根据物品流行度（如销量、浏览量）决定是否检索——仅对流行度低于阈值 $p$ 的冷启动物品进行KG检索。这符合幂律分布特性（少数热门物品已有充足知识，冷启动物品才需要增强），显著减少检索时间
- **子图重排序 + GNN编码**：检索Top-K子图后，以推荐prompt为query重排序取Top-N → 用第二个GNN编码器提取结构信息 → MLP投影器对齐到LLM嵌入空间作为soft prompt前缀

### 损失函数

交叉熵损失 $\mathcal{L}(Y, A)$，其中 $Y$ 为ground-truth推荐物品，$A$ 为LLM预测。仅训练两个GNN和MLP投影器的参数，LLM参数冻结。

## 实验

### 主实验（LLaMA-2，Frozen LLM + Prompt Tuning，候选集M=20）

| 方法 | ML-1M ACC | ML-1M R@3 | ML-1M R@5 | ML-20M ACC | ML-20M R@3 | Amazon ACC | Amazon R@5 |
|------|-----------|-----------|-----------|------------|------------|------------|------------|
| KG-Text | 0.076 | - | - | 0.052 | - | 0.058 | - |
| KAPING | 0.079 | - | - | 0.069 | - | 0.063 | - |
| PT w/ KG-Text | 0.078 | 0.191 | 0.308 | 0.051 | 0.152 | 0.074 | 0.245 |
| GraphToken w/ RAG | 0.268 | 0.421 | 0.466 | 0.186 | 0.433 | 0.326 | 0.624 |
| G-retriever | 0.274 | 0.532 | 0.650 | 0.342 | 0.619 | 0.275 | 0.612 |
| **K-RagRec** | **0.435** | **0.725** | **0.831** | **0.600** | **0.850** | **0.508** | **0.780** |
| 提升幅度 | +58.6% | +33.0% | +27.8% | +75.4% | +37.3% | +55.8% | +25.0% |

### 消融实验

| 变体 | 影响 |
|------|------|
| 去除多跳索引（仅一阶邻居） | 性能显著下降，无法捕获高阶关系 |
| 去除选择性检索（全体检索） | 性能略降 + 检索效率大幅下降 |
| 去除重排序 | 不相关子图干扰生成 |
| 去除GNN编码（KG文本序列化） | 结构信息利用不充分，性能下降 |
| 去除流行度策略 | 检索时间增加，热门物品检索引入噪声 |

### 关键发现

- K-RagRec在所有数据集和指标上大幅超越所有baseline，ACC提升55-75%
- 多跳子图索引相比仅检索三元组/一阶邻居，能提供更全面的物品知识视图
- 流行度选择性检索策略在保持性能的同时显著减少检索开销
- GNN编码器+投影器方式优于KG文本序列化，更好地保留了图结构信息
- Fine-tuning（LoRA w/ K-RagRec）可进一步提升3-16%性能

## 亮点

- 知识图谱RAG在推荐系统中的系统性方案，从索引、检索、重排到编码形成完整闭环
- 流行度选择性检索策略兼顾效率和效果，设计简洁且符合推荐系统的幂律分布特性
- 使用GNN作为图结构的原生编码器而非文本序列化，避免了"long context"和结构信息丢失

## 局限性

- 依赖外部KG（Freebase）的质量和覆盖度，不完整或过时的KG可能限制效果
- GNN和投影器需要训练，增加了部署复杂度，且需要推荐数据和KG的对齐
- 流行度阈值 $p$ 需要数据集特定调优，不同领域的最佳阈值可能差异很大
- 主要在电影和图书推荐上验证，电商、新闻等更复杂场景需进一步探索
- 两阶段GNN（索引+编码）增加了模型参数和训练复杂度
- 未与最新的Graph RAG方法（如GraphRAG、LightRAG）进行对比

## 相关工作

- RAG：REALM（Guu et al. 2020）、DPR（Karpukhin et al. 2020）、RETRO（Borgeaud et al. 2022）
- 图增强RAG：G-Retriever（He et al. 2024）、Retrieve-Rewrite-Answer（Wu et al. 2023b）
- LLM推荐：TALLRec（Bao et al. 2023）、推荐中的RAG初探（Di Palma 2023）
- 知识图谱推荐：KGAT、KAPING（Baek et al. 2023）
- 图Token化：GraphToken（Perozzi et al. 2024）

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](../../NeurIPS2025/graph_learning/gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)
- [SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation](simgrag_leveraging_similar_subgraphs_for_knowledge_graphs_driven_retrieval-augme.md)
- [Neural Graph Matching Improves Retrieval Augmented Generation in Molecular Machine Learning](../../ICML2025/graph_learning/neural_graph_matching_improves_retrieval_augmented_generation_in_molecular_machi.md)
- [mRAKL: Multilingual Retrieval-Augmented Knowledge Graph Construction for Low-Resourced Languages](mrakl_multilingual_retrieval-augmented_knowledge_graph_construction_for_low-reso.md)
- [RAS: Retrieval-And-Structuring for Knowledge-Intensive LLM Generation](../../ICLR2026/graph_learning/ras_retrieval-and-structuring_for_knowledge-intensive_llm_generation.md)

<!-- RELATED:END -->
