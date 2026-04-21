---
title: >-
  [论文解读] Logical Characterizations of GNNs with Mean Aggregation
description: >-
  [AAAI 2026][图学习][图神经网络] 系统刻画了以均值（mean）为聚合函数的 GNN 的表达能力：非一致设定下等价于比率模态逻辑（RML）；一致设定下（相对 MSO）等价于模态逻辑（ML）；当额外要求组合函数连续、分类函数为阈值时，表达能力显著下降至交替无关模态逻辑（AFML）。
tags:
  - AAAI 2026
  - 图学习
  - 图神经网络
  - 均值聚合
  - 模态逻辑
  - 比率模态逻辑
  - 交替无关模态逻辑
  - 非一致性/一致性设定
---

# Logical Characterizations of GNNs with Mean Aggregation

**会议**: AAAI 2026  
**arXiv**: [2507.18145](https://arxiv.org/abs/2507.18145)  
**领域**: 图学习 / 图神经网络理论  
**关键词**: GNN表达能力, 均值聚合, 模态逻辑, 比率模态逻辑, 交替无关模态逻辑, 非一致性/一致性设定  

## 一句话总结

系统刻画了以均值（mean）为聚合函数的 GNN 的表达能力：非一致设定下等价于比率模态逻辑（RML）；一致设定下（相对 MSO）等价于模态逻辑（ML）；当额外要求组合函数连续、分类函数为阈值时，表达能力显著下降至交替无关模态逻辑（AFML）。

## 研究背景与动机

GNN 的聚合函数选择（sum/max/mean）直接影响模型可表达的图性质，先前工作已将 sum-GNN 与分级模态逻辑（GML）、max-GNN 与模态逻辑（ML）建立对应，但 **mean 聚合——GraphSAGE、GCN 等主流架构的核心操作——缺乏完整的逻辑刻画**。

Mean 聚合在实践中广泛使用的原因包括：
1. GraphSAGE、PinSAGE 等系统直接采用 mean 聚合
2. GCN 使用加权版本的 mean
3. Mean 聚合兼容随机采样，适合高度节点

核心问题：Mean-GNN 能精确表达哪些图性质？在不同理论设定（一致/非一致、有无连续性约束）下分别等价于何种逻辑？

## 方法详解

### 整体理论框架

论文分两大设定（非一致 vs 一致）和两类技术约束（一般 vs 连续+阈值）来刻画 Mean-GNN 的表达能力，给出双向翻译（GNN→逻辑、逻辑→GNN），并与 sum/max 聚合进行系统比较。

| 设定 | Mean (连续+阈值) | Mean (一般) | Sum | Max |
|------|------------------|-------------|-----|-----|
| 非一致 | RML | RML | GML | ML |
| 一致 (相对 MSO) | **AFML** | ML | GML | ML |
| 一致 (绝对) | >> AFML | >> ML | >> GML | ML |

### 关键设计一：比率模态逻辑（RML）与非一致设定

- 定义新逻辑 RML：模态算子 ◇^{≥r}φ 表示"至少 r 比例的后继满足 φ"
- **Mean-GNN → RML**：由于 mean 只关注邻居特征向量的比率分布，可为每层每个特征向量构造 RML 公式
- **RML → Mean-GNN**：将 RML 公式的子公式逐层编码为 GNN 的特征向量分量，利用 n² 放大因子区分不同比率
- 推论：非一致设定下 Max-GNN ⊊ Mean-GNN ⊊ Sum-GNN（严格层次）

### 关键设计二：一致设定下 Mean-GNN ∩ MSO = ML

- RML 的模态算子无法在 MSO 中表达，因此一致设定下 RML 被截断为 ML
- 关键工具：**图缩放不变性**——c 倍缩放图 c·G 不改变 mean 聚合的结果
- 利用 Ehrenfeucht-Fraïssé 博弈，证明缩放 + GML 博弈的 Duplicator 策略可从 ML 博弈策略导出
- 结果：一致设定下 Mean-GNN（相对 MSO）与 Max-GNN 表达能力相同，均等价于 ML

### 关键设计三：连续+阈值约束下 AFML 刻画

- 实际 GNN 的组合函数通常连续（支持反向传播），分类函数通常为阈值
- **Mean^{c,t}-GNN → AFML**：利用 AFML 博弈（Spoiler 只能选择一个方向）+ 图扩展引理，证明连续组合+阈值分类的 Mean-GNN 无法混合 ◇ 和 □
- **AFML → Mean^{c,t}-GNN**：将 AFML 公式中的真值编码为 (0,1] 区间的正值、假值编码为 0，利用 truncated ReLU/sigmoid 实现
- 核心发现：连续性+阈值约束使 Mean-GNN 表达能力从 ML 降至 AFML，但对 sum/max 无影响

### 损失函数/目标函数

本文为纯理论工作，不涉及训练损失函数。核心目标是建立逻辑 ↔ GNN 的双向等价翻译。

## 实验

本文为理论论文，无实验部分，但提供了完整的定理-证明体系。

### 主要结果总览

| 定理 | 内容 | 设定 |
|------|------|------|
| Thm 1 | Mean-GNN ⊆ RML ⊆ simple Mean-GNN | 非一致 |
| Thm 7 | Mean-GNN ∩ MSO ⊆ ML ⊆ simple Mean-GNN | 一致 |
| Thm 12 | Mean^{c,t}-GNN ∩ MSO ⊆ AFML ⊆ simple Mean^{c,t}-GNN | 一致 + 连续阈值 |
| Cor 1 | Max-GNN ⊊ Mean-GNN ⊊ Sum-GNN | 非一致 |
| Cor 2 | Max-GNN = Mean-GNN ∩ MSO ⊊ Sum-GNN ∩ MSO | 一致 |

### 关键消融/对比

| 特性 | Mean | Sum | Max |
|------|------|-----|-----|
| 连续+阈值约束降低表达能力？ | ✅ 从 ML 降至 AFML | ❌ | ❌ |
| 图缩放不变性 | ✅ | ❌ | ❌ |
| 激活函数通用性（非一致设定） | 任意连续非多项式 | 任意连续非多项式 | 仅 ReLU/truncated ReLU |

### 关键发现

1. **Mean 聚合的实际表达能力低于直觉预期**：在实际约束（连续组合+阈值分类）下等价于 AFML，不能混合存在性 (◇) 和全称性 (□)
2. **三种聚合函数形成严格层次**：Sum > Mean > Max（非一致），Sum > Mean = Max（一致+MSO）
3. 连续性/阈值约束仅影响 mean 聚合，不影响 sum 和 max

## 亮点

- **首次完整刻画 Mean-GNN 的逻辑对应**，填补了 GNN 表达能力理论的重要空白
- 引入 **比率模态逻辑（RML）** 和 **交替无关模态逻辑（AFML）** 两个新逻辑工具
- 揭示了 **实际约束（连续+阈值）对 mean 聚合的独特影响**——这是 sum/max 不具备的现象
- 图缩放不变性作为关键技巧，优雅地将问题从 RML 降维到 ML
- 理论结果覆盖 ReLU、truncated ReLU、sigmoid 等主流激活函数

## 局限性

- **纯理论工作**，无实验验证理论预测对实际任务的指导意义
- 一致设定下 ML → Mean-GNN 的翻译使用了不连续激活函数或不自然的分类函数（将无理数分类为 1），实用性有限
- 是否可扩展到异构聚合（不同层使用不同聚合函数）未探讨
- 未讨论加权 mean（如 GCN 的归一化方式）的逻辑刻画
- Max-GNN 的激活函数通用性问题（能否扩展到所有连续非多项式函数）仍开放

## 相关工作

- **Sum-GNN 逻辑刻画**：Barceló et al. (2020) 证明 Sum-GNN ↔ GML（相对 FO）
- **Max-GNN 逻辑刻画**：Tena Cucala et al. (2024) 将 Max-GNN 翻译为非递归 datalog
- **GNN 表达能力的非逻辑方法**：Xu et al. (ICLR 2019) 研究不同聚合函数的表达能力，但未使用逻辑刻画
- **Transformer 层的模态逻辑刻画**：与 mean 聚合紧密相关
- **单调 Mean-GNN 的逻辑解释**：Morris & Horrocks (2025)

## 评分

⭐⭐⭐⭐ (4/5)

理论贡献扎实完整，系统性地填补了 Mean-GNN 逻辑刻画的空白。RML 和 AFML 的引入为后续研究提供了精确的分析工具。但纯理论性质限制了对实际 GNN 设计的直接指导作用，且部分翻译构造的实用性不足。

<!-- RELATED:START -->

## 相关论文

- [Enhancing Logical Expressiveness in GNNs via Path-Neighbor Aggregation](enhancing_logical_expressiveness_in_graph_neural_networks_via_path-neighbor_aggr.md)
- [Adaptive Initial Residual Connections for GNNs with Theoretical Guarantees](adaptive_initial_residual_connections_for_gnns_with_theoretical_guarantees.md)
- [MoToRec: Sparse-Regularized Multimodal Tokenization for Cold-Start Recommendation](motorec_sparse-regularized_multimodal_tokenization_for_cold-start_recommendation.md)
- [RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA](rfkg-cot_relation-driven_adaptive_hop-count_selection_and_few-shot_path_guidance.md)
- [S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)

<!-- RELATED:END -->
