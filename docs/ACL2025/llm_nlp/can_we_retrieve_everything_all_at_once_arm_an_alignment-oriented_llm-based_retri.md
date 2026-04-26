---
title: >-
  [论文解读] Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method
description: >-
  [ACL 2025][LLM/NLP][检索增强生成] 提出 ARM（Alignment-oriented Retrieval Method），通过在 LLM 解码过程中融合信息对齐（N-gram 约束解码）、结构对齐（MIP 求解器推理数据对象间关系）和自验证聚合三个模块，实现"一次检索所有"所需数据对象，在 Bird 和 OTT-QA 数据集上显著优于标准 RAG（最高 +5.2pt）和 agentic RAG/ReAct（最高 +19.3pt）。
tags:
  - ACL 2025
  - LLM/NLP
  - 检索增强生成
  - 复杂问答
  - 对齐式检索
  - 约束解码
  - 混合整数规划
---

# Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method

**会议**: ACL 2025  
**arXiv**: [2501.18539](https://arxiv.org/abs/2501.18539)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: 检索增强生成, 复杂问答, 对齐式检索, 约束解码, 混合整数规划

## 一句话总结

提出 ARM（Alignment-oriented Retrieval Method），通过在 LLM 解码过程中融合信息对齐（N-gram 约束解码）、结构对齐（MIP 求解器推理数据对象间关系）和自验证聚合三个模块，实现"一次检索所有"所需数据对象，在 Bird 和 OTT-QA 数据集上显著优于标准 RAG（最高 +5.2pt）和 agentic RAG/ReAct（最高 +19.3pt）。

## 研究背景与动机

现实中的复杂开放域问题通常需要多个异构信息源（文本、表格等）的信息来回答。例如"加州人口最多的县中，K-12 学生免费午餐资格最高的比例是多少？"可能需要 1 篇文章 + 3 个可连接的表格。

**现有方法的问题**：

**标准 RAG + 查询分解**：LLM 将问题分解为子查询，对每个子查询检索。但分解过程不了解数据集合中有什么数据以及它们如何组织，容易遗漏桥接表格（bridging tables）等未被显式提及的关键对象。

**Agentic RAG (ReAct)**：通过迭代方式，每轮根据之前检索结果决定下一步行动。问题在于：（a）探索由"还缺什么信息"而非"什么数据可用"驱动，效率低下；（b）无法联合优化前后步骤，一步出错容易推理脱轨（reasoning derailment）；（c）LLM 调用次数多，成本高。

**核心思想**：检索复杂问题的答案需要将问题与数据集合的组织结构对齐（alignment）——不仅做语义匹配，还要推理数据对象之间的连接关系。

## 方法详解

### 整体框架

ARM 将检索建模为一个生成式问题：LLM 通过一次解码过程输出一个"推理过程"来找到所有需要的数据对象。解码过程包含三个对齐步骤：
1. **信息对齐（Information Alignment）**：提取关键信息并与数据集合中的 N-gram 对齐
2. **结构对齐（Structure Alignment）**：推理已识别数据对象的连接关系，补充缺失的桥接对象
3. **自验证与聚合（Self-Verification & Aggregation）**：LLM 验证对象的相关性并通过 beam search 聚合结果

### 关键设计

**索引（Indexing）**：将表格和文章统一为文本数据对象，每个对象被分块（chunk），计算嵌入并构建 N-gram 集合（N=1~3）。表格通过名称+标题+描述+行来序列化。

**信息对齐 — 约束波束解码**：
- LLM 首先从问题中提取关键词
- 然后通过约束解码将关键词与数据集合中的 N-gram 对齐
- 使用后缀树（suffix tree）追踪有效的 token 续写
- 解码的 N-gram 用 BM25 搜索相关 chunk，结合嵌入相似度计算最终得分

**结构对齐 — 混合整数规划（MIP）**：

$$\arg\max \sum_i R_i b_i + \sum_{i,j} C_{ij} c_{ij}$$

其中 $R_i$ 是对象与问题的相关性（余弦相似度），$C_{ij}$ 是对象间的兼容性（语义 + 精确值匹配），$b_i, c_{ij} \in \{0,1\}$ 是决策变量。约束确保选择 k 个连通的对象。使用 Gurobi 求解器。

**多草案扩展**：由于信息对齐的基础集合可能不够完整，通过迭代扩展基础对象集（每步添加最兼容的 k 个对象，重复 l 步），对不同扩展集分别运行 MIP，生成多个草案。

**自验证 — LLM 作为验证器**：
- 将草案注入 LLM 解码过程（约束解码强制输出草案内容）
- LLM 综合评估所有已解码信息，选择最终数据对象
- 使用约束解码确保所选对象确实存在于草案中

**beam search 聚合**：多个 beam 产生多个推理过程，通过加权投票聚合：投票权重 = 对象名称 token 的平均 logits；投票次数 = 跨 beam 的出现次数（softmax 归一化）。最终置信度 = 两者加权和。

### 损失函数 / 训练策略

ARM 不需要额外训练 LLM——直接使用 Llama-3.1-8B-Instruct 进行推理。核心创新在于解码策略而非训练。使用 3-shot ICL 提示指导 LLM 执行检索推理。

## 实验关键数据

### 主实验

**检索性能**（表2）：

| 方法 | Bird Recall | Bird PR | OTT-QA Recall | OTT-QA PR | LLM调用数↓ |
|------|-----------|---------|-------------|-----------|-----------|
| Dense Retrieval @5 | 89.0 | 78.4 | 75.2 | 53.8 | 0 |
| ReAct (Llama3.1) | 96.7 | 93.5 | 76.0 | 55.1 | 5.26 |
| ReAct (GPT4o-mini) | 97.0 | 93.3 | 80.6 | 62.7 | 3.16 |
| **ARM** | **96.5** | **92.7** | **79.8** | **62.5** | **1** |

ARM 仅需 1 次 LLM 调用即达到与 ReAct (5+ 次调用) 相当的召回率，但精确率从 15% 提升到 42.7%（Bird），大幅减少检索噪声。

**端到端性能**（表3，两个模型的平均值）：

| 方法 | Bird Acc | OTT-QA Exact | OTT-QA F1 |
|------|---------|-------------|-----------|
| Dense Retrieval @5 | 23.6 | 37.1 | 44.2 |
| DRR + 分解 @5 | 21.2 | 42.2 | 50.9 |
| ReAct | 15.1 | 33.9 | 40.9 |
| **ARM** | **26.2** | **46.6** | **55.5** |

ARM 在 Bird 上比标准 RAG 高 2.6pt，比 ReAct 高 11.1pt；OTT-QA 上比标准 RAG 高 4.4pt F1，比 ReAct 高 14.6pt F1。

### 消融实验

**模块逐步添加**（图2，两数据集平均）：
- Dense Retrieval + 分解（基线）
- + 信息对齐：Recall +12.5pt, PR +19.8pt
- + 结构对齐：Recall 再 +1.28pt, PR +4.02pt
- + 自验证聚合：Recall 再 +5.72pt, PR +9.18pt

每个模块都有显著贡献，其中信息对齐贡献最大。

**信息对齐中关键词查找的贡献**（图3）：仅用嵌入相似度 vs 嵌入+关键词查找，后者 Recall +2.15pt, PR +3.65pt。

### 关键发现

1. 一次解码即可完成复杂检索，无需多轮迭代——这是 ARM 相比 ReAct 效率优势的核心
2. ReAct 的两大错误模式：（a）遗忘之前迭代的信息；（b）循环搜索相似关键词
3. 结构对齐模块通过 MIP 求解器推理表格间的连接关系（如可连接列、桥接实体），这是纯语义检索无法实现的
4. ARM 检索的对象更少但质量更高，减少了下游 LLM 处理的噪声

## 亮点与洞察

- **范式转变**：从"迭代检索"到"一次对齐检索"，大幅提升效率（1次 vs 5+ 次 LLM 调用）
- **检索中的推理**：将结构推理（表格连接、桥接实体）集成到检索过程中，而非留给下游 LLM
- **约束解码的创新运用**：通过 N-gram 约束解码实现信息对齐，确保生成的关键词与实际数据集合匹配
- **MIP 引入的灵活性**：可以注入领域特定的业务逻辑到检索目标中

## 局限与展望

- 依赖预先构建的 N-gram 索引和嵌入，索引构建成本未充分讨论
- MIP 求解在大规模数据集合上的可扩展性可能受限（搜索空间与对象数量相关）
- 仅在 Llama3.1-8B-Instruct 上运行 ARM，更大模型是否能进一步提升效果未知
- 结构对齐中的兼容性分数依赖于嵌入相似度和精确值匹配，可能遗漏更复杂的语义关系
- 草案扩展的步数和每步添加数量需要手动设定
- 未探索与更强的检索器（如 ColBERT）或更好的嵌入模型的组合

## 相关工作与启发

- 扩展了 Chen et al. (2024c) 的"一次检索所有表格"思路，将其扩展到表格+文本的多模态数据集合
- 与 RAG to Riches (Jain et al., 2024) 的"边生成边检索"思路相似，但 ARM 额外引入了结构推理
- 对 agentic RAG 的理性批判：迭代方式虽然能交互数据，但受限于"基于过去决策"而非"基于数据组织"
- 启发：复杂检索需要同时考虑信息相关性和数据结构，纯语义匹配不够

## 评分

- **新颖性**：9/10 — "一次对齐检索"的范式和 MIP 在检索中的应用非常新颖
- **技术深度**：9/10 — 约束解码、MIP 求解、自验证三个模块技术含量高
- **实验充分性**：8/10 — 两个代表性数据集，全面的消融和基线比较
- **实用价值**：8/10 — 大幅减少 LLM 调用成本，但依赖特定的索引和求解器
- **总体评分**：9/10

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] RetroLLM: Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation](retrollm_empowering_large_language_models_to_retrieve_fine-grained_evidence_with.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization](improving_contextual_faithfulness_of_large_language_models_via_retrieval_heads-i.md)
- [\[ACL 2025\] Pre³: Enabling Deterministic Pushdown Automata for Faster Structured LLM Generation](pre3_deterministic_pda_structured_gen.md)
- [\[ACL 2025\] Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models](salience_sparse_fine_tuning.md)

<!-- RELATED:END -->
