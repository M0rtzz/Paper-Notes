---
title: >-
  [论文解读] HELIOS: Harmonizing Early Fusion, Late Fusion, and LLM Reasoning for Multi-Granular Table-Text Retrieval
description: >-
  [ACL 2025][Table-Text Retrieval] 提出 HELIOS 三阶段图检索框架（边级早期融合 → 节点级晚期融合 → 星图级 LLM 精化），通过多粒度协调统一解决表格-文本检索中的检索单元粒度、查询依赖关系发现和高级推理三大挑战，在 OTT-QA 上实现 42.6% Answer Recall 提升。
tags:
  - ACL 2025
  - Table-Text Retrieval
  - 二部图
  - 早期融合
  - 晚期融合
  - LLM推理
  - 多跳问答
  - ColBERT
  - 图检索
---

# HELIOS: Harmonizing Early Fusion, Late Fusion, and LLM Reasoning for Multi-Granular Table-Text Retrieval

**会议**: ACL 2025  
**arXiv**: [2603.02248](https://arxiv.org/abs/2603.02248)  
**代码**: 未公开  
**领域**: 信息检索 / 表格-文本检索  
**关键词**: Table-Text Retrieval, 二部图, 早期融合, 晚期融合, LLM推理, 多跳问答, ColBERT, 图检索  

## 一句话总结

提出 HELIOS 三阶段图检索框架（边级早期融合 → 节点级晚期融合 → 星图级 LLM 精化），通过多粒度协调统一解决表格-文本检索中的检索单元粒度、查询依赖关系发现和高级推理三大挑战，在 OTT-QA 上实现 42.6% Answer Recall 提升。

## 研究背景与动机

- **问题定义**: 开放域表格-文本检索需要同时从结构化表格和非结构化文本中检索相关信息，支持跨模态多跳推理来回答复杂问题。
- **现有局限**: (1) **早期融合**（OTTeR、DoTTeR）通过实体链接预连接表格行和段落形成 fused block，但常包含查询无关段落且无法捕获动态关系；(2) **晚期融合**（CORE、COS）动态匹配但面临 beam search 的错误传播问题；(3) 两种方法都依赖语义相似度，无法处理需要列级聚合或多跳逻辑推理的查询。
- **研究动机**: 需要一个统一框架在不同检索阶段使用合适的粒度（边/节点/星图），同时结合预定义关系（早期融合）和动态发现关系（晚期融合），并引入 LLM 推理能力处理复杂查询。
- **核心创新**: 用二部图形式化表格-文本关系，设计三阶段多粒度检索框架，每个阶段在不同粒度上操作。

## 方法详解

### 整体框架

HELIOS 将语料库建模为二部图 G=(V, E)，节点为表格片段（table segment）和段落（passage），边表示它们的关系。三阶段流程：

1. **边级二部子图检索 (Edge-based Bipartite Subgraph Retrieval)** → 利用早期融合
2. **查询相关节点扩展 (Query-relevant Node Expansion)** → 利用晚期融合
3. **星图级 LLM 精化 (Star-based LLM Refinement)** → 利用 LLM 推理

### 关键设计

1. **阶段一：边级检索**: 离线通过实体链接构建二部数据图 Gd。每条边（表格片段+段落）线性化后用 ColBERTv2 编码为多向量表示，计算 late interaction 相似度。先检索 top-k₁ 边再用 all-to-all reranker 精选 top-k₂ 边，合并去重为候选子图 Gc。使用**边**而非节点作为检索单元，平衡信息完整性（避免晚期融合的部分相关问题）和噪声控制（避免早期融合的 fused block 过大问题）。

2. **阶段二：节点扩展**: 在最细粒度（节点级）进行扩展，处理早期融合引入的查询无关节点。使用 beam search 两步法：(1) **种子节点选择**——用 all-to-all reranker 识别 Gc 中与查询最相关的 top-b 节点；(2) **种子节点扩展**——基于扩展查询技术 sim([q;Γ(u)], v) 从完全二部图中检索种子节点的最相关邻居，形成扩展图 Gl。

3. **阶段三：LLM 精化**: 将扩展图分解为星图（star graph）单元送入 LLM（优于整图 prompt，+12.4%），执行两个操作：(1) **列级聚合**——恢复完整表格让 LLM 执行聚合操作（如 "最新的记录"）；(2) **段落验证**——对每条边进行二元验证，移除查询无关边。

### 粒度选择原理

| 阶段 | 粒度 | 原因 |
|------|------|------|
| 早期融合 | 边 | 比节点包含更多上下文（减少部分相关），比 fused block 更精细（减少噪声） |
| 晚期融合 | 节点 | 最细粒度，精确识别查询相关节点，避免扩展无关内容 |
| LLM 精化 | 星图 | 包含多跳关系的最小单元，比整图更有效（减少幻觉） |

## 实验

### 主实验结果 (OTT-QA Dev Set)

| 模型 | 类型 | AR@2 | AR@5 | AR@10 | AR@50 | nDCG@50 |
|------|------|------|------|-------|-------|---------|
| OTTeR | Early | 31.4 | 49.7 | 62.0 | 82.0 | 25.9 |
| DoTTeR | Early | 31.5 | 51.0 | 61.5 | 80.8 | 26.7 |
| CORE | Late | 35.3 | 50.7 | 63.1 | 83.1 | 25.4 |
| COS | Late | 44.4 | 61.6 | 70.8 | 87.8 | 33.6 |
| COS w/ ColBERT & bge | Late | 49.6 | 68.2 | 78.7 | 91.7 | 36.5 |
| DoTTeR+COS+LLM | 组合 | 50.0 | 62.4 | 70.0 | 84.7 | 34.7 |
| **HELIOS** | **统一** | **63.3** | **76.7** | **85.0** | **94.2** | **47.0** |

### 端到端 QA 结果

| 数据集 | 模型 | EM | F1 |
|-------|------|------|------|
| OTT-QA Test | COS | 54.9 | 61.5 |
| OTT-QA Test | **HELIOS** | **57.0** | **64.3** |
| MMQA Dev | COS | 54.4 | 63.7 |
| MMQA Dev | **HELIOS** | **59.6** | **69.1** |

### 消融实验

| 消除组件 | AR@2 变化 | nDCG@50 变化 | 说明 |
|---------|----------|-------------|------|
| 去除节点扩展 | -4.1 | — | 晚期融合的动态关系发现很重要 |
| 去除 LLM 精化 | -2.9 | — | LLM 推理对聚合查询尤其关键 |
| 全图 prompt vs 星图 | — | -12.4% | 分解为星图减少幻觉更有效 |
| 去除边级检索（改用节点） | 显著下降 | — | 边级比节点级检索更有效 |

### 关键发现

1. **HELIOS 在 AR@2 上比 SOTA COS 提升 42.6%，nDCG@50 提升 39.9%** — 大幅度 SOTA 突破
2. **简单堆叠强模块（DoTTeR+COS+LLM）远不如 HELIOS** — 说明关键在多粒度的协调设计而非简单组合
3. **跨数据集泛化**: 在 MMQA（非 OTT-QA 目标设计）上仍平均提升 20.9% AR
4. **与 HOLMES 对比**: EM 提升 88.4%，F1 提升 58.1% — 主要优势来自边级种子检索、查询条件推理和保留表格结构

## 亮点

- 用二部图形式统一了早期融合和晚期融合的表达，理论框架优雅
- 三阶段使用不同粒度（边→节点→星图）的设计有清晰的理论动机和实证验证
- 在 OTT-QA 上取得超大幅度 SOTA 提升，AR@2 提升 42.6%
- 对比实验证明简单堆叠模块无法达到同等效果，体现了架构设计的重要性
- 消融实验充分，每个组件的贡献都有独立验证

## 局限性

- 系统复杂度高：涉及多个编码器（ColBERTv2 + 多个 reranker + LLM），训练和推理成本较大
- LLM 精化阶段引入额外延迟，可能限制实时应用
- 实验仅在 OTT-QA 和 MMQA 两个数据集上验证，表格-文本检索的数据集有限
- 实体链接质量直接影响初始二部图构建，但文中未详细讨论链接错误的级联影响
- 代码未公开

## 相关工作

- **早期融合**: Fusion-Retriever (Chen et al., 2020a)、OTTeR (Huang et al., 2022)、DoTTeR (Kang et al., 2024) 通过实体链接预构建 fused block
- **晚期融合**: CORE (Ma et al., 2022)、COS (Ma et al., 2023) 动态形成表格-段落证据链
- **图方法**: DRAMA (Yuan et al., 2024)、HOLMES (Panda et al., 2024) 用图做多跳 QA，但限于 distractor 设置
- **表格编码**: GTR (Wang et al., 2021)、MGNETS (Chen et al., 2021) 用图方法改进表格编码

## 评分

| 维度 | 分数 (1-10) |
|------|-----------|
| 新颖性 | 8 |
| 技术深度 | 9 |
| 实验充分性 | 9 |
| 写作质量 | 8 |
| 实用价值 | 7 |
| 总分 | 8.2 |

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](../../ACL2026/information_retrieval/enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)
- [\[ACL 2025\] Mitigating Lost-in-Retrieval Problems in RAG Multi-Hop QA](mitigating_lost-in-retrieval_problems_in_retrieval_augmented_multi-hop_question_.md)
- [\[ACL 2025\] Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval](optimized_text_embedding_models_and_benchmarks_for_amharic_passage_retrieval.md)
- [\[ACL 2025\] Collapse of Dense Retrievers: Short, Early, and Literal Biases Outranking Factual Evidence](collapse_dense_retrievers.md)
- [\[ACL 2025\] Health-LLM: Personalized Retrieval-Augmented Disease Prediction System](health-llm_personalized_retrieval-augmented_disease_prediction_system.md)

<!-- RELATED:END -->
