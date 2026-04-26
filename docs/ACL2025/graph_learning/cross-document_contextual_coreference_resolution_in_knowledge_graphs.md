---
title: >-
  [论文解读] Cross-Document Contextual Coreference Resolution in Knowledge Graphs
description: >-
  [ACL 2025][图学习][跨文档共指消解] 提出基于知识图谱的跨文档共指消解方法，通过动态链接机制将文本实体提及与知识图谱节点关联，结合上下文嵌入和图消息传递推理提升跨文档实体识别的精度和召回率，在多个基准数据集上超越传统方法。
tags:
  - ACL 2025
  - 图学习
  - 跨文档共指消解
  - 知识图谱
  - 实体链接
  - 图推理
  - 上下文嵌入
---

# Cross-Document Contextual Coreference Resolution in Knowledge Graphs

**会议**: ACL 2025  
**arXiv**: [2504.05767](https://arxiv.org/abs/2504.05767)  
**代码**: 无  
**领域**: graph_learning  
**关键词**: 跨文档共指消解, 知识图谱, 实体链接, 图推理, 上下文嵌入

## 一句话总结

提出基于知识图谱的跨文档共指消解方法，通过动态链接机制将文本实体提及与知识图谱节点关联，结合上下文嵌入和图消息传递推理提升跨文档实体识别的精度和召回率，在多个基准数据集上超越传统方法。

## 研究背景与动机

**领域现状**: 共指消解（coreference resolution）是 NLP 的基础任务，要求识别不同文本中指向同一实体的不同表述。跨文档共指消解（CD-CR）在知识图谱领域尤为重要，需要在多个文档间关联实体引用。

**现有痛点**: 
   - 传统共指消解方法主要关注单文档内的引用关系，跨文档场景下性能下降
   - 现有方法缺乏对知识图谱结构信息的充分利用
   - 实体提及稀疏或定义模糊时，链接准确率下降
   - 大规模图推理的计算复杂度是扩展瓶颈

**核心矛盾**: 跨文档实体引用散布在不同上下文中，仅靠文本特征难以准确判断其是否指向同一实体；而知识图谱包含丰富的结构化实体关系信息，但如何有效融合文本上下文和图结构信息仍是挑战。

**本文目标**: 设计一种方法，将知识图谱中的结构化实体关系与文本上下文嵌入有效结合，提升跨文档共指消解的精度和召回率。

**切入角度**: 将跨文档共指消解建模为图上的实体链接任务，用动态链接和消息传递来传播实体信息。

**核心 idea**: 通过动态链接机制将文本提及映射到知识图谱实体，再利用图消息传递迭代更新实体表示，实现跨文档共指消解。

## 方法详解

### 整体框架

1. 从多文档中提取实体提及 $M = \{m_1, m_2, \ldots, m_n\}$
2. 利用知识图谱 $\mathcal{G} = (V, E)$ 中的实体和关系建立链接
3. 通过上下文嵌入计算提及与实体的相似度
4. 使用图消息传递迭代更新实体表示
5. 基于精化后的表示进行共指判断

### 关键设计

#### 1. 共指消解基本框架
- **功能**: 判断跨文档的文本提及是否指向同一实体
- **核心思路**: 定义相似度函数 $S(m_i, e_j) = f(m_i, e_j, \mathcal{G})$，结合上下文嵌入和图谱信息计算提及-实体匹配分数。动态链接选择最佳候选：

$$\hat{e}_i = \arg\max_{e_j \in V} S(m_i, e_j)$$

通过阈值 $\theta$ 过滤弱链接：$\mathcal{R} = \{(e_i, e_j) | e_i, e_j \in V, S(m_i, e_i) > \theta\}$

- **设计动机**: 将实体消歧与知识图谱实体对齐统一框架，利用图结构约束消除歧义

#### 2. 知识图谱集成
- **功能**: 将共指消解建模为图上的链接任务
- **核心思路**: 为每个提及计算上下文嵌入 $e_j = f(m_j, \mathcal{C})$（$\mathcal{C}$ 来自图谱和周围文本），通过相似度矩阵 $S[m_j, v_i]$ 评估对齐强度，链接函数：$L(m_j) = \arg\max_{v_i \in V} S(e_j, v_i)$。链接后用传播算法整合：$R = \mathcal{P}(L(M), \mathcal{G})$
- **设计动机**: 知识图谱提供了实体间的先验关系，可作为跨文档实体消歧的强约束

#### 3. 增强实体链接
- **功能**: 通过图神经网络消息传递迭代精化实体表示
- **核心思路**: 用 sigmoid 函数估计链接概率：$L(t_j, v_i) = \sigma(\mathbf{f}(t_j, v_i))$。消息传递更新实体嵌入：

$$\mathbf{h}_i^{(t+1)} = \sum_{v_j \in \mathcal{N}(v_i)} \mathbf{W} \cdot \mathbf{h}_j^{(t)} + \mathbf{b}$$

迭代收敛后，用精化的嵌入进行共指判断：$C(t_j, v_i) = \mathcal{R}(\mathbf{h}_i, \mathbf{h}_j)$

- **设计动机**: 单次链接可能不够准确，消息传递可以通过邻居信息不断修正实体表示

### 损失函数/训练策略

- **模型配置**: Llama-3 和 GPT-3.5 作为上下文嵌入生成器
- **训练**: 学习率 3e-5，10 epochs，batch size 32
- **评估指标**: Precision、Recall、F1 Score

## 实验关键数据

### 主实验

不同模型和数据集上的 F1 分数：

| 模型/方法 | SP-10K | CoNLL-2012 | ConceptNet | Complex SQ |
|-----------|--------|------------|------------|------------|
| Llama-3 | 71.4 | **73.9** | - | - |
| GPT-3.5 | - | - | 68.6 | 72.5 |
| CorefUD | - | - | - | - (LexGLUE: 64.4) |
| ThaiCoref | **76.8** | **78.7** | - | - |
| Major Entity ID | - | - | 61.2 | 63.0 |
| Event Coref Bank+ | - | - | - | - (GLUE: 69.4) |
| Rationale-centric | 73.9 | 75.2 | - | - |

**ThaiCoref** 在 CoNLL-2012 上取得最高 F1 78.7%；Llama-3 的 CoNLL-2012 F1 为 73.9%。

### 消融实验

移除各组件后的性能变化（F1 Score）：

| 模型/方法 | SP-10K | CoNLL-2012 |
|-----------|--------|------------|
| Llama-3 (full) | 71.4 | 73.9 |
| Llama-3 (ablated) | 69.6 | 72.2 |
| ThaiCoref (full) | 76.8 | 78.7 |
| ThaiCoref (ablated) | 75.3 | 76.6 |
| Rationale-centric (full) | 73.9 | 75.2 |
| Rationale-centric (ablated) | 73.8 | 73.7 |

消融后各方法 F1 均有下降（1-2%），说明各组件都有贡献。

实体交互捕捉分析：Llama-3 在直接链接上精度 76.5，ThaiCoref 最高达 80.3。

### 关键发现

1. ThaiCoref 出人意料地在多个数据集上表现最优（F1 78.7%），可能受益于其细粒度标注策略
2. Llama-3 在跨文档场景中表现稳健（平均 F1 72-74%），优于 GPT-3.5
3. 知识图谱信息的融入对所有方法都带来提升，消融后性能下降 1-2%
4. Major Entity Identification 方法性能最低（F1 ~61-63%），说明仅识别主要实体不足以解决复杂共指
5. 图消息传递的迭代更新对精化实体表示有明确贡献

## 亮点与洞察

- **融合结构化与非结构化**: 将知识图谱的结构化关系和文本的非结构化上下文有效融合，是跨文档 CR 的自然方向
- **方法通用性**: 框架不依赖特定的 LLM，可灵活替换嵌入生成器
- **全面的 baseline 对比**: 涵盖了多语言（CorefUD、ThaiCoref）、事件（Event Coref Bank+）、因果（Rationale-centric）等多种共指消解范式

## 局限与展望

1. 实体提及稀疏或定义模糊时动态链接准确率可能下降
2. 训练数据多样性不足时模型泛化能力受限
3. 图推理的计算复杂度可能成为大规模数据集的瓶颈
4. 方法描述中数学符号有重叠（$e$ 既表示实体又表示嵌入），可读性可以改进
5. 消融实验不够细粒度，未单独评估动态链接、消息传递、阈值选择的独立贡献
6. 缺少与最新的 LLM-based 共指消解方法（如基于 GPT-4 的 few-shot 方法）的直接对比

## 相关工作与启发

- **Maverick** (Martinelli et al., 2024): 高效共指消解 pipeline，在参数受限环境中表现优异
- **LQCA** (Liu et al., 2024a): 长上下文共指适应方法，处理长文本中的引用关系
- **Contrastive CR** (Hsu & Horwood, 2022): 对比表示学习做跨文档事件/实体共指
- **Context Graph** (Xu et al., 2024): 利用 LLM 进行上下文图的知识表示和推理

## 评分

- **新颖性**: ⭐⭐⭐ — 知识图谱 + 共指消解的结合不算全新，但动态链接 + 消息传递框架有一定贡献
- **实验充分度**: ⭐⭐⭐ — 数据集较多但部分实验设计不够严谨，消融不够细粒度
- **写作质量**: ⭐⭐⭐ — 结构完整但部分公式符号不够清晰，related work 描述偏冗长
- **价值**: ⭐⭐⭐ — 提供了跨文档 CR 的知识图谱增强思路，但改进幅度不够突出

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Can LLMs Evaluate Complex Attribution in QA? Automatic Benchmarking using Knowledge Graphs](paper_2401_14640.md)
- [\[ACL 2025\] Extending Complex Logical Queries on Uncertain Knowledge Graphs](extending_complex_logical_queries_uncertain_knowledge_graphs.md)
- [\[ACL 2025\] SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation](simgrag_leveraging_similar_subgraphs_for_knowledge_graphs_driven_retrieval-augme.md)
- [\[ACL 2025\] Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering](kg_llm_trustworthy_qa.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](../../NeurIPS2025/graph_learning/deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)

<!-- RELATED:END -->
