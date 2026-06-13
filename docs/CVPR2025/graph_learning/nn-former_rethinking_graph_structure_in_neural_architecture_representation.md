---
title: >-
  [论文解读] NN-Former: Rethinking Graph Structure in Neural Architecture Representation
description: >-
  [CVPR 2025][图学习][架构预测] NN-Former 提出混合 GNN-Transformer 架构预测器，发现现有方法忽略了"兄弟节点"（共享父/子节点）的拓扑信息，通过 Adjacency-Sibling Multihead Attention (ASMA) 和 Bidirectional Graph Isomorphism FFN (BGIFFN) 在 NAS-Bench-101/201 上 Kendall's Tau 达 0.877/0.890，延迟预测 MAPE 降低 48-64%。
tags:
  - "CVPR 2025"
  - "图学习"
  - "架构预测"
  - "NAS"
  - "Transformer"
  - "兄弟节点"
  - "延迟预测"
---

# NN-Former: Rethinking Graph Structure in Neural Architecture Representation

**会议**: CVPR 2025  
**arXiv**: [2507.00880](https://arxiv.org/abs/2507.00880)  
**代码**: [https://github.com/XuRuihan/NNFormer](https://github.com/XuRuihan/NNFormer)  
**领域**: 图学习  
**关键词**: 架构预测、NAS、图Transformer、兄弟节点、延迟预测

## 一句话总结

NN-Former 提出混合 GNN-Transformer 架构预测器，发现现有方法忽略了"兄弟节点"（共享父/子节点）的拓扑信息，通过 Adjacency-Sibling Multihead Attention (ASMA) 和 Bidirectional Graph Isomorphism FFN (BGIFFN) 在 NAS-Bench-101/201 上 Kendall's Tau 达 0.877/0.890，延迟预测 MAPE 降低 48-64%。

## 研究背景与动机

1. **领域现状**：神经架构搜索（NAS）中的性能预测器需要从架构图中预测精度或延迟。现有方法分为 GNN 类（GATES、GMAE-NAS）和 Transformer 类（NAR-Former V2、PINAT）。
2. **现有痛点**：(1) GNN 仅传播邻接信息，忽略了"兄弟"关系（如两个并行分支共享输入/输出）；(2) Transformer 的全局注意力不区分不同类型的图结构关系；(3) 现有预测器在低数据量（0.02%训练）时精度不足。
3. **核心矛盾**：图结构中"兄弟关系"（如残差连接的两个分支）包含重要设计信息，但 $A$ 矩阵中看不到（需要 $AA^T$ 或 $A^TA$），现有方法全部遗漏。
4. **本文目标**：设计同时利用邻接和兄弟拓扑的混合预测器。
5. **切入角度**：用 $AA^T$ 提取共享父节点的兄弟（如残差块的两条路径），用 $A^TA$ 提取共享子节点的兄弟（如多输入汇聚）。
6. **核心 idea**：ASMA（4 头注意力各用不同拓扑 mask）+ BGIFFN（双向图卷积前馈网络）。

## 方法详解

### 整体框架

架构图 → one-hot 操作编码 + 正弦属性编码 → $L$ 层 NN-Former（ASMA + BGIFFN）→ 全局池化 → MLP 预测精度或延迟。

### 关键设计

1. **Adjacency-Sibling Multihead Attention (ASMA)**

    - 功能：同时建模邻接和兄弟四种拓扑关系
    - 核心思路：4 个注意力头分别使用不同的拓扑 mask——Head 1: $(I+A)$ 前向邻接；Head 2: $(I+A^T)$ 后向邻接；Head 3: $(I+AA^T)$ 共父兄弟；Head 4: $(I+A^TA)$ 共子兄弟。各头：$X_i = \sigma((Q_iK_i^T \circ Mask_i)/\sqrt{h})V_i$
    - 设计动机：标准 Transformer 的全局注意力把所有节点对等价处理——但在计算图中，相邻、父子、兄弟关系的重要性完全不同

2. **Bidirectional Graph Isomorphism FFN (BGIFFN)**

    - 功能：在前馈网络中注入图结构信息
    - 核心思路：$H_g = \text{Concat}(\text{GC}(H, A), \text{GC}(H, A^T))$，正反两个方向的图卷积拼接，然后与原始特征融合：$\text{BGIFFN}(H, A) = \text{ReLU}(HW_1 + H_g)W_2$。不需要位置编码
    - 设计动机：标准 FFN 独立处理每个节点——注入图卷积后 FFN 也能感知拓扑结构

3. **无位置编码设计**

    - 功能：BGIFFN 自然提供了位置信息
    - 核心思路：由于 BGIFFN 在每层都做正反图卷积，节点的位置信息通过多层传播被隐式编码
    - 设计动机：图中的节点没有固定顺序（不像序列），传统位置编码不适用

### 损失函数 / 训练策略

L1 损失（精度预测）或 MAPE 损失（延迟预测）。AdamW 优化器。

## 实验关键数据

### 主实验

| 设定 | NAR-Former V2 | **NN-Former** | 提升 |
|------|--------------|-------------|------|
| NAS-101 (0.02%) | 0.663 | **0.709** | +6.9% |
| NAS-101 (1%) | 0.861 | **0.877** | +1.9% |
| NAS-201 (1%) | 0.752 | **0.804** | +6.9% |
| NAS-201 (10%) | 0.888 | **0.890** | +0.2% |
| 延迟-EfficientNet | 13.20 | **4.81** | -63.6% |
| 延迟-MnasNet | 7.16 | **2.54** | -64.5% |

### 消融实验

| 配置 | NAS-101 (0.1%) Tau | 说明 |
|------|-------------------|------|
| w/o 兄弟 (仅邻接) | ~0.77 | 丢失关键信息 |
| w/o BGIFFN | ~0.78 | FFN 无图感知 |
| Cross-attention替代 | 0.804 | 次优 |
| **Full NN-Former** | **0.809** | 最优 |

### 关键发现

- 在低数据量（0.02%）下提升最大（+6.9%），说明兄弟信息对数据效率至关重要
- 延迟预测提升最惊人（48-64% MAPE降低），因为延迟直接受并行结构影响——兄弟关系恰好编码了并行性
- 不需要位置编码是一个工程优势——简化了设计且自然适配不同大小的图

## 亮点与洞察

- **兄弟节点的发现**：$AA^T$ 和 $A^TA$ 提取的拓扑关系在 NAS 中被所有先前工作遗漏——简单但影响巨大
- **4 头拓扑注意力**：每个头专注一种结构关系的设计比全局注意力更高效且更有效
- **延迟预测的实用价值**：精度预测是研究导向，延迟预测是部署导向——NN-Former 在延迟上的巨大优势使其对模型部署更有价值

## 局限与展望

- 兄弟计算需要 $O(n^2)$ 矩阵乘法（$AA^T$, $A^TA$），对超大图可能成为瓶颈
- 主要在 cell-based 架构上测试，自定义架构（如 MoE、分支网络）未充分验证
- 4 个头的设计（各对应一种拓扑）是硬编码的——更多头或不同拓扑组合未探索

## 相关工作与启发

- **vs NAR-Former V2**: 纯 Transformer 方法，不区分拓扑类型。NN-Former 通过 ASMA 显式编码四种关系
- **vs GATES/GMAE-NAS**: GNN 方法仅传播邻接信息。NN-Former 通过兄弟关系弥补了 GNN 的盲区

## 评分

- 新颖性: ⭐⭐⭐⭐ 兄弟节点的发现和4头拓扑注意力有新意
- 实验充分度: ⭐⭐⭐⭐⭐ NAS-101/201精度+NNLQ延迟+全面消融
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 对NAS预测器有直接改进价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Banyan: Improved Representation Learning with Explicit Structure](../../ICML2025/graph_learning/banyan_improved_representation_learning_with_explicit_structure.md)
- [\[NeurIPS 2025\] From Sequence to Structure: Uncovering Substructure Reasoning in Transformers](../../NeurIPS2025/graph_learning/from_sequence_to_structure_uncovering_substructure_reasoning_in_transformers.md)
- [\[ICML 2026\] Rethinking Feature Alignment in Generalist Graph Anomaly Detection: A Relational Fingerprint-based Approach](../../ICML2026/graph_learning/rethinking_feature_alignment_in_generalist_graph_anomaly_detection_a_relational_.md)
- [\[CVPR 2025\] Hypergraph Vision Transformers: Images are More than Nodes, More than Edges](hypergraph_vision_transformers_images_are_more_than_nodes_more_than_edges.md)
- [\[ICML 2025\] Beyond Message Passing: Neural Graph Pattern Machine](../../ICML2025/graph_learning/beyond_message_passing_neural_graph_pattern_machine.md)

</div>

<!-- RELATED:END -->
