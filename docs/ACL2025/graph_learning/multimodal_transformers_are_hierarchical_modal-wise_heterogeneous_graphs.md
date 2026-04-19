---
title: >-
  [论文解读] Multimodal Transformers are Hierarchical Modal-wise Heterogeneous Graphs
description: >-
  [ACL 2025][图学习][多模态情感分析] 从图论视角证明了多模态 Transformer（MulTs）本质上是层次化模态异质图（HMHG），并基于此提出 GsiT 模型，通过 Interlaced Mask 机制实现仅 1/3 参数的 All-Modal-In-One 融合，同时性能显著超越传统 MulTs。
tags:
  - ACL 2025
  - 图学习
  - 多模态情感分析
  - 图结构
  - Transformer
  - 权重共享
  - 注意力掩码
---

# Multimodal Transformers are Hierarchical Modal-wise Heterogeneous Graphs

**会议**: ACL 2025  
**arXiv**: [2505.01068](https://arxiv.org/abs/2505.01068)  
**代码**: https://github.com/thuiar/MMSA  
**领域**: 图学习  
**关键词**: 多模态情感分析, 图结构, Transformer, 权重共享, 注意力掩码  


## 一句话总结

从图论视角证明了多模态 Transformer（MulTs）本质上是层次化模态异质图（HMHG），并基于此提出 GsiT 模型，通过 Interlaced Mask 机制实现仅 1/3 参数的 All-Modal-In-One 融合，同时性能显著超越传统 MulTs。

## 研究背景与动机

多模态情感分析（MSA）需要融合文本、视频、音频等异质模态来识别情感。Multimodal Transformer（MulT）及其后续变体（MulTs）是当前主流范式，通过跨模态注意力（CMA）和多头自注意力（MHSA）实现融合。然而 MulTs 存在严重的效率问题：

- **参数冗余**：MulTs 将三模态拆解为成对组合分别处理，每对组合使用独立权重（6 个 CMA + 3 个 MHSA + 9 个 MLP），参数量大
- **端到端系统约束**：MSA 是端到端判别任务，模型需要在实际部署中兼顾性能和效率
- **缺乏理论分析**：MulTs 的结构冗余缺少形式化的理论分析和优化指导

作者提出一个关键洞察：对于 MSA 系统，通过设计低成本高性能模型节省的资源，在某些方面比单纯使用大模型带来的准确率提升更有意义。

## 方法详解

### 1. 理论基础：MulTs 即层次异质图（HMHG）

论文首先建立了注意力机制与图结构的等价关系：

- **引理 1**：多头跨模态注意力（CMA）等价于双模态组合的单向完全二部图的聚合；多头自注意力（MHSA）等价于单模态有向完全图的聚合
- **定理 1**：MulTs 是层次化模态异质图（HMHG）

具体地，MulTs 可以分解为：
- 底层子图 G(i,j) 和 G(i,p)：由 CMA 构建的跨模态二部图
- 中间拼接层：将跨模态融合向量拼接
- 顶层子图 G(i,i)：由 MHSA 构建的模态内有向完全图

从单一主导模态看，形成一棵树；多个主导模态的树组合形成森林结构。

### 2. Graph-Structured Interlaced-Masked Multimodal Transformer (GsiT)

基于 HMHG 定理发现的参数冗余，GsiT 将传统的森林结构压缩为单棵共享权重的树结构。核心机制是 **Interlaced Mask (IM)**，包含两部分：

**Interlaced-Multimodal-Fusion Mask (IFM)**：
- forward mask：允许 v->t, a->v, t->a 方向的注意力
- backward mask：允许 a->t, t->v, v->a 方向的注意力
- 两个反向单向环形成完整的跨模态融合，避免信息混乱

**Interlaced-Intra-Enhancement Mask (IEM)**：
- 仅允许模态内自注意力（t->t, v->v, a->a），屏蔽跨模态位置

通过在注意力得分矩阵中加上负无穷掩码实现图结构约束，使得共享权重的 MHSA 等效于独立的 CMA 操作。

### 3. Decomposition Triton 核

GsiT 拼接多模态序列后注意力图空间复杂度为 O((T_t+T_v+T_a)^2)，高于 MulTs 的 O(T_i*T_j)。为此实现了 Decomposition Triton 核：在共享 QKV 投影后，按原始序列长度分解，按 IM 指定的模态对独立执行注意力，使运行时空间复杂度与 MulTs 持平，同时静态参数量降为 1/3。

### 4. All-Modal-In-One 融合

最终函数系统从 MulTs 的 6 个 CMA + 3 个 MHSA + 9 个 MLP 压缩为 3 个 MHSA + 3 个 MLP，参数量降至传统方法的 1/3，实现所有模态在共享权重中完成融合。

## 实验关键数据

### 表 1：CMU-MOSI 和 CMU-MOSEI 主实验

| 模型 | MOSI Acc-2 (NN/NP) | MOSI Acc-7 | MOSEI Acc-2 (NN/NP) | Params (M) | FLOPS (G) |
|------|-------------------|------------|---------------------|------------|-----------|
| MulT | 79.6 / 81.4 | 36.2 | 78.1 / 83.7 | 5.251 | 26.294 |
| **GsiT** | **83.7 / 85.8** | **47.4** | **84.5 / 85.6** | **1.695** | **26.224** |
| Delta | +4.1 / +4.4 | +11.2 | +6.4 / +1.9 | **-67.7%** | -0.3% |
| TETFN | 82.4 / 84.0 | 46.1 | 81.9 / 84.3 | 5.921 | 27.558 |
| TETFN w/ HMHG | 83.2 / 85.2 | 47.1 | 84.6 / 84.8 | 2.365 | 27.488 |
| ALMT | 82.1 / 83.3 | 45.5 | 81.4 / 83.5 | 2.604 | 19.876 |
| ALMT w/ HMHG | 83.2 / 84.6 | 47.1 | 82.9 / 86.4 | 2.506 | 19.876 |

GsiT 相比 MulT：参数量减少 67.7%，Acc-2 提升 4+%，Acc-7 提升 11.2%。

### 表 2：消融实验（CMU-MOSI）

| 结构 | Acc-2 (NN/NP) | Acc-7 | MAE |
|------|---------------|-------|-----|
| **Original (对向环形)** | **83.7 / 85.8** | **47.4** | **0.713** |
| Structure-1 (非环形) | 83.5 / 85.5 | 46.5 | 0.721 |
| Structure-2 (非环形) | 83.2 / 84.9 | 43.8 | 0.729 |
| Structure-3 (非环形) | 83.4 / 85.2 | 45.5 | 0.726 |
| Self-Only (信息混乱) | 82.5 / 84.2 | 45.5 | 0.734 |

对向环形结构最优；违反 HMHG 约束的 Self-Only 性能最差，验证了信息混乱问题的存在。

## 亮点

- **理论贡献突出**：严格证明 MulTs 与 HMHG 的等价关系，将多模态融合的结构化分析提升到图论高度
- **参数效率极高**：仅用 1/3 参数即实现甚至超越原始 MulTs 性能，Acc-7 提升 11.2% 令人印象深刻
- **通用性强**：HMHG 概念可即插即用地嵌入 Self-MM、TETFN、ALMT 等多种基线模型并带来提升
- **信息混乱理论**：从 softmax 概率分布角度解释了为什么掩码设计需要遵循特定约束，提供了清晰的理论指导
- **工程落地友好**：Decomposition Triton 核确保运行时无额外计算开销

## 局限与展望

- **任务范围有限**：仅在多模态情感分析上验证，未扩展到视频问答、多模态检索等更广泛的多模态任务
- **模态缺失未考虑**：未讨论当某个模态缺失时模型的鲁棒性
- **表示学习未结合**：第一层融合编码器对中未引入对比学习等表示学习方法，是未来可探索的方向
- **仅限三模态**：当前框架针对文本/视频/音频三模态设计，扩展到更多模态时 IM 的设计需要重新考虑
- **数据集规模偏小**：CMU-MOSI/MOSEI 等经典 MSA 数据集规模有限，未在大规模数据上验证

## 与相关工作的对比

- **MulT (Tsai et al., 2019)**：GsiT 的直接理论基础，GsiT 用 1/3 参数超越其全部指标
- **Self-MM (Yu et al., 2021)**：自监督学习框架，集成 GsiT 后性能提升但效率下降（因其原本融合层极简）
- **TETFN (Wang et al., 2023)**：纯 MulTs 模型，嵌入 HMHG 后参数降 60.1%，大部分指标提升
- **ALMT (Zhang et al., 2023)**：MulTs-like 架构，HMHG 嵌入后参数降 3.8%，MOSEI 上 Acc-2 提升 2.9%
- **GAT (Velickovic et al., 2018)**：本文证明了 CMA/MHSA 与 GAT 的理论等价，是建立 HMHG 的桥梁

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ --- 图论视角重新审视 MulTs 架构，理论证明原创性很强
- 实验充分度: ⭐⭐⭐⭐ --- 四个数据集 + 多基线集成 + 消融 + 权重分布分析 + 收敛分析，较为全面
- 写作质量: ⭐⭐⭐⭐ --- 理论推导严谨，但大量 LaTeX 公式增加阅读难度
- 价值: ⭐⭐⭐⭐ --- 为多模态融合提供了新的理论框架和实用的效率优化方案

<!-- RELATED:START -->

## 相关论文

- [Unifying and Enhancing Graph Transformers via a Hierarchical Mask Framework](../../NeurIPS2025/graph_learning/unifying_and_enhancing_graph_transformers_via_a_hierarchical_mask_framework.md)
- [M3HG: Multimodal, Multi-scale, and Multi-type Node Heterogeneous Graph for Emotion Cause Triplet Extraction in Conversations](m3hg_multimodal_multi-scale_and_multi-type_node_heterogeneous_graph_for_emotion_.md)
- [Graph Tokenization for Bridging Graphs and Transformers](../../ICLR2026/graph_learning/graph_tokenization_for_bridging_graphs_and_transformers.md)
- [Fast-and-Frugal Text-Graph Transformers are Effective Link Predictors](fast-and-frugal_text-graph_transformers_are_effective_link_predictors.md)
- [Relieving the Over-Aggregating Effect in Graph Transformers](../../NeurIPS2025/graph_learning/relieving_the_over-aggregating_effect_in_graph_transformers.md)

<!-- RELATED:END -->
