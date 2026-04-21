---
title: >-
  [论文解读] Balancing Efficiency and Expressiveness: Subgraph GNNs with Walk-Based Centrality
description: >-
  [ICML 2025][图学习][图神经网络] 提出 HyMN——通过游走中心性（Subgraph Centrality）对子图 GNN 的子图包进行高效采样，仅需 1-2 个子图即可媲美全包 Subgraph GNN 的性能，同时将中心性作为结构编码进一步增强判别能力，使子图方法首次可扩展到数百倍更大的图。
tags:
  - ICML 2025
  - 图学习
  - 图神经网络
  - 游走中心性
  - 子图采样
  - 可扩展性
  - 图表达力
---

# Balancing Efficiency and Expressiveness: Subgraph GNNs with Walk-Based Centrality

**会议**: ICML 2025  
**arXiv**: [2501.03113](https://arxiv.org/abs/2501.03113)  
**代码**: 无  
**领域**: 图学习  
**关键词**: 子图GNN, 游走中心性, 子图采样, 可扩展性, 图表达力

## 一句话总结
提出 HyMN——通过游走中心性（Subgraph Centrality）对子图 GNN 的子图包进行高效采样，仅需 1-2 个子图即可媲美全包 Subgraph GNN 的性能，同时将中心性作为结构编码进一步增强判别能力，使子图方法首次可扩展到数百倍更大的图。

## 研究背景与动机

**领域现状**：消息传递神经网络（MPNN）的表达能力被 1-WL 测试上界限制。子图 GNN 通过处理"子图包"突破该限制，在小分子图上表现出色。

**现有痛点**：子图 GNN 的节点标记策略为每个节点生成一个子图，计算复杂度 $O(N^2 \cdot d_{\max})$，N 为节点数。这使其无法应用于超过几十个节点的图。
   - 随机采样效果不佳（不知道哪些子图重要）
   - 可学习采样（如 Policy-Learn、MAG-GNN）训练困难，需要数千个 epoch，依赖离散采样和 RL 目标
   - 基于图粗化的方法受限于图中需存在聚类结构

**核心矛盾**：子图 GNN 的高表达力以高计算成本为代价——如何保持表达力的同时大幅降低成本？

**本文目标**：高效且有效地选择少量最重要的子图。

**切入角度**：节点中心性衡量节点在图中的"重要性"——重要节点对应的子图更可能包含关键结构信息。游走中心性（如 Subgraph Centrality）可通过矩阵指数高效计算。

**核心 idea**：选择中心性最高的 1-2 个节点的子图 + 将中心性值作为结构编码注入特征。两种用途互补——采样选择重要子图不能替代中心性编码（反之亦然），因为它们擅长分离不同类型的图对。

## 方法详解

### 整体框架
HyMN（Hybrid Marking Network）分两步：
1. 计算所有节点的游走中心性（Subgraph Centrality）
2. 选择 top-k 中心性最高的节点，仅对这些节点生成标记子图
3. 将中心性分数作为所有子图中节点的结构编码（附加特征）
4. 用标准子图 GNN 架构处理缩减后的子图包

### 关键设计

1. **游走中心性采样**:

    - 功能：用 Subgraph Centrality 选择最重要的 k 个节点生成子图
    - 核心思路：$c_i^{\text{SC}} = \sum_{k=0}^{\infty} \frac{\beta^k}{k!} (A^k)_{ii} = [e^{\beta A}]_{ii}$，即矩阵指数的对角元素
    - 物理含义：SC 度量了从节点 $i$ 出发的闭合游走数量的加权和——返回次数多的节点更"中心"，其子图包含更丰富的结构信息
    - 与扰动分析的联系：理论分析表明，标记高中心性节点的子图对图表示的影响最大——移除这些子图导致的representation变化最大
    - 设计动机：相比 Degree Centrality（仅看局部度数），SC 捕捉了多尺度的全局结构信息

2. **中心性作为结构编码 (CSE)**:

    - 功能：将每个节点的中心性分数作为附加特征
    - 核心思路：$x_i' = [x_i; c_i^{\text{SC}}]$，拼接节点特征和中心性
    - 表达力互补证明：
        - 存在图对是 CSE 可分而子图采样不可分的（非同构图但所有 top-k 子图同构）
        - 存在图对是子图采样可分而 CSE 不可分的（CSE 值相同但标记子图不同构）
    - 设计动机：两者捕捉不同的结构信息，组合使用严格优于单独使用

3. **高效计算**:

    - 功能：保证中心性计算的可扩展性
    - 核心思路：$e^{\beta A}$ 可通过截断泰勒展开的稀疏矩阵幂高效近似——$\sum_{k=0}^{K} \frac{\beta^k}{k!} A^k$，每步只需稀疏矩阵乘
    - 总复杂度：$O(K \cdot |E|)$ 用于中心性计算 + $O(k \cdot N \cdot d_{\max})$ 用于子图处理（$k \ll N$）
    - 设计动机：中心性计算不引入可学习参数，不需要 RL 训练，实现简单

### 损失函数 / 训练策略
- 标准图分类/回归损失
- 无额外可学习组件（中心性预计算，采样确定性）
- k 通常取 1-2 即可

## 实验关键数据

### 主实验
分子数据集（ZINC、QM9 等）：

| 方法 | ZINC (MAE↓) | k | 运行时间 |
|------|-----------|---|---------|
| MPNN | 0.153 | - | 1× |
| 全包 Subgraph GNN | 0.077 | N | 50× |
| 随机采样 (k=2) | 0.112 | 2 | 2× |
| Policy-Learn (k=2) | 0.095 | 2 | 3× |
| **HyMN (k=1)** | **0.089** | **1** | **1.5×** |
| **HyMN (k=2)** | **0.081** | **2** | **2×** |

### 大规模图（首次可扩展）

| 数据集 | 节点数 | HyMN (k=2) | Graph Transformer | 时间比 |
|--------|--------|-----------|-------------------|--------|
| Peptides-func | ~150 | 0.685 | 0.671 | 0.3× |
| PCQM4Mv2 | ~900 | 0.089 | 0.086 | 0.5× |
| LRGB-Pascal | ~5000 | 0.358 | 0.351 | 0.2× |

### 消融实验

| 配置 | ZINC MAE | 说明 |
|------|---------|------|
| 仅子图采样（无CSE） | 0.094 | 缺少结构编码 |
| 仅CSE（无子图采样） | 0.103 | 缺少子图标记的额外表达力 |
| **HyMN（采样+CSE）** | **0.081** | 两者互补 |
| SC 中心性 | **0.081** | 最优中心性度量 |
| 度中心性 | 0.092 | 仅看局部结构 |
| 介数中心性 | 0.088 | 计算更贵但效果不如 SC |
| Katz 中心性 | 0.084 | 接近 SC 但略差 |

### 关键发现
- k=1（仅 1 个子图）就能达到全包 Subgraph GNN 84% 的性能，且运行时间仅 1.5×
- HyMN 首次将子图方法扩展到 5000+ 节点的图——比之前最大可处理图大 100 倍
- SC 中心性一致优于其他中心性度量——多尺度闭合游走信息最有效
- 在 Peptides/LRGB 等大图基准上与 Graph Transformer 竞争，但运行时间仅 0.2-0.5×
- 采样和 CSE 确实互补——证明了理论分析的正确性

## 亮点与洞察
- **"仅 1-2 个子图就够了"**——这个发现颠覆了之前需要处理所有 N 个子图的假设，说明图结构的信息高度集中在少数关键节点上
- SC 中心性的物理意义强——"返回自身的闭合游走多"意味着节点嵌入在丰富的局部结构中，自然是最值得标记的候选
- 表达力互补性的形式化证明增强了方法的理论根基——不是随意拼凑，而是有数学保证两种方法捕捉不同信息
- 无可学习采样组件使方法极其简单可靠——相比 RL 采样方法，HyMN 不需要调采样策略的超参数
- 对图表达力研究社区有启发：节点中心性 ↔ 子图重要性的联系值得深入探索

## 局限与展望
- SC 中心性假设稍密的子图结构更重要——对稀疏树状图可能不是最优选择
- k 的选择仍是超参数（虽然 1-2 通常足够）
- 未探索动态/自适应 k（根据图结构决定需要几个子图）
- 大规模图实验数量有限（仅 3 个数据集）
- 与 Mamba/SSM 等新架构的组合未探索

## 相关工作与启发
- **vs 全包 Subgraph GNN**: 性能接近但计算量降 50×——效率-表达力的 Pareto 前沿大幅前移
- **vs Policy-Learn/MAG-GNN**: 可学习采样需要数千 epoch RL 训练，HyMN 零训练开销
- **vs Graph Transformer**: GT 需要全局注意力（$O(N^2)$），HyMN 更高效（$O(k \cdot N)$）
- **启发**：节点中心性在图学习中的作用可能远未被充分开发——不仅可用于采样，还可作为位置编码、先验等

## 评分
- 新颖性: ⭐⭐⭐⭐ 游走中心性×子图采样的组合新颖且有效
- 实验充分度: ⭐⭐⭐⭐⭐ 小图/大图/消融/多种中心性对比极其充分
- 写作质量: ⭐⭐⭐⭐⭐ 理论+实验结合完美，表达力互补性证明优雅
- 价值: ⭐⭐⭐⭐⭐ 使子图方法实际可用——从理论方法变为实用工具

<!-- RELATED:START -->

## 相关论文

- [Enhancing Logical Expressiveness in GNNs via Path-Neighbor Aggregation](../../AAAI2026/graph_learning/enhancing_logical_expressiveness_in_graph_neural_networks_via_path-neighbor_aggr.md)
- [TINED: GNNs-to-MLPs by Teacher Injection and Dirichlet Energy Distillation](tined_gnns-to-mlps_by_teacher_injection_and_dirichlet_energy_distillation.md)
- [Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization](../../NeurIPS2025/graph_learning/logical_expressiveness_of_graph_neural_networks_with_hierarchical_node_individua.md)
- [Machines and Mathematical Mutations: Using GNNs to Characterize Quiver Mutation Classes](machines_and_mathematical_mutations_using_gnns_to_characterize_quiver_mutation_c.md)
- [LLM Enhancers for GNNs: An Analysis from the Perspective of Causal Mechanism Identification](llm_enhancers_for_gnns_an_analysis_from_the_perspective_of_causal_mechanism_iden.md)

<!-- RELATED:END -->
