---
title: >-
  [论文解读] Positional Encoding meets Persistent Homology on Graphs
description: >-
  [ICML 2025][图学习][位置编码] 理论证明图上位置编码（PE）和持续同调（PH）在区分非同构图方面互不可比，提出 PiPE（Persistence-informed Positional Encoding）通过消息传递网络统一两者，可证明比单独使用任一方法更具表达力，在 ZINC/Alchemy/DrugOOD/BREC 等多个基准上一致优于纯 PE 和纯 PH 基线。
tags:
  - ICML 2025
  - 图学习
  - 位置编码
  - 持续同调
  - 图表达力
  - 图神经网络
  - PiPE
---

# Positional Encoding meets Persistent Homology on Graphs

**会议**: ICML 2025  
**arXiv**: [2506.05814](https://arxiv.org/abs/2506.05814)  
**代码**: https://github.com/Aalto-QuML/PIPE (有)  
**领域**: 图学习 / 图表达力  
**关键词**: 位置编码, 持续同调, 图表达力, GNN, PiPE

## 一句话总结

理论证明图上位置编码（PE）和持续同调（PH）在区分非同构图方面互不可比，提出 PiPE（Persistence-informed Positional Encoding）通过消息传递网络统一两者，可证明比单独使用任一方法更具表达力，在 ZINC/Alchemy/DrugOOD/BREC 等多个基准上一致优于纯 PE 和纯 PH 基线。

## 研究背景与动机

**消息传递 GNN 的表达力瓶颈**：标准消息传递 GNN 至多等价于 1-WL 测试，无法捕获连通性、环等关键结构信息。为此社区发展了两条增强路线：一是**位置编码（PE）**，利用 Laplacian 特征向量或随机游走距离为节点赋予位置感知特征；二是**持续同调（PH）**，通过拓扑数据分析捕获多尺度拓扑特征（连通分量的诞生/消亡、独立环的持续性等）。

**核心矛盾**：两种范式都能增强 GNN 表达力，但它们的相对优劣关系尚不清楚——PE 能否涵盖 PH？PH 是否比 PE 更强？能否把两者统一以获得进一步增益？这些问题缺乏严格理论刻画。

**本文切入**：通过构造性证明建立 PE 和 PH 的不可比性（各存在对方失败但自身成功的图对），并基于这一洞察设计 PiPE，在理论和实验上同时证明统一方法严格优于任一单独方法。核心 idea：**用 PE 特征驱动 PH 的过滤函数，再用 PH 的拓扑嵌入反馈增强 PE 更新，形成闭环**。

## 方法详解

### 整体框架

PiPE 的 pipeline 分为三个并行更新流：(1) **位置编码流** $\{p_v^\ell\}$：从基础 PE（如 LapPE）初始化，通过消息传递迭代更新；(2) **拓扑特征流** $\{r_v^{\ell,0}, r_v^{\ell,1}\}$：在每层利用当前位置编码计算过滤函数 $f_\ell$，生成 0 维和 1 维持续图 $\mathcal{D}_\ell^0, \mathcal{D}_\ell^1$，再通过向量化得到节点级拓扑嵌入；(3) **主干 GNN 流** $\{x_v^\ell\}$：将节点特征、位置嵌入、拓扑嵌入拼接后进行标准消息传递。最终 readout 阶段将三者融合用于图级/节点级预测。

### 关键设计

1. **PE-驱动的持续同调计算**:
    - 功能：利用可学习位置编码作为节点颜色，构建 vertex-color 过滤函数来计算持续图
    - 核心思路：在每层 $\ell$，用 MLP $f_\ell$ 将位置编码 $p_v^\ell$ 映射为过滤值，据此构建子图序列 $G_\alpha$，跟踪连通分量（0 维）和独立环（1 维）的诞生与消亡，得到持续图 $\mathcal{D}_\ell^0, \mathcal{D}_\ell^1$
    - 设计动机：Lemma 3.3 证明，基于 PE 的过滤函数产生的 0 维持续图至少与 PE 本身一样具有表达力（不会损失信息），同时 Proposition 3.4 证明 PH 能区分某些 PE 无法区分的图对，因此组合严格更强

2. **拓扑嵌入的节点级向量化**:
    - 功能：将持续图转化为每个节点的固定维度嵌入向量
    - 核心思路：0 维持续图 $|\mathcal{D}_0| = n$ 与节点数相同，可建立双射，对每个 $(birth, death)$ 元组用 MLP $\Psi_0^\ell$ 得到 $r_v^{\ell,0}$；1 维持续图对应独立环，先通过 $\Psi_1^\ell$ 向量化边级特征，再对与节点关联的边聚合得到 $r_v^{\ell,1}$
    - 设计动机：保持与消息传递框架兼容的节点级表示，使拓扑信息可以直接参与后续层的更新

3. **位置-拓扑联合消息传递更新**:
    - 功能：将拓扑嵌入反馈到位置编码更新中，形成闭环
    - 核心思路：位置编码更新公式 $p_v^{\ell+1} = \text{Upd}_\ell^p(r_v^{\ell,0}, r_v^{\ell,1}, p_v^\ell, \text{Agg}_\ell(\{(r_u^{\ell,0}, r_u^{\ell,1}, p_u^\ell) : u \in \mathcal{N}(v)\}))$，将邻居的拓扑和位置信息同时聚合
    - 设计动机：不同于 LSPE 仅用位置信息更新位置嵌入，PiPE 让拓扑信息参与位置更新，理论上使得后续层的过滤函数能编码更丰富的结构，从而产生更有区分力的持续图

### 理论分析

- **Proposition 4.1**：LPE-based PiPE 严格强于 LPE-based LSPE（在无属性图空间上）
- **Proposition 4.2**：LPE-based PiPE 严格强于 PH + LPE（仅组合但未通过消息传递统一）
- **Proposition 4.3**：RW-based PiPE 存在无法区分的图对（3-WL 可区分），指出了局限性
- **Proposition 4.4**：若 $k$-FWL 能区分两个图，则可构造过滤函数使 0 维持续图不同（强化了已有结果）

## 实验关键数据

### 主实验：BREC 表达力基准

| 数据集 | PH | PH+LPE | PiPE |
|--------|------|--------|------|
| Basic (60对) | 0.03 | 0.10 | **0.72** |
| Regular (50对) | 0.00 | 0.15 | **0.40** |
| Extension (100对) | 0.07 | 0.13 | **0.67** |
| CFI (100对) | 0.03 | 0.03 | 0.03 |

PiPE 在 Basic/Regular/Extension 上大幅领先，验证了统一方法的表达力增益。CFI 图对各方法均难以区分（需更高阶 WL）。

### 合成树任务（Perplexity ↓）

| PE 方案 | PH | C₃-B | C₃-D | Reorder-B | Reorder-D | Copy-B | Copy-D |
|---------|------|------|------|-----------|-----------|--------|--------|
| RoPE | None | 1.84 | 2.52 | 4.93 | 6.63 | 1.85 | 3.17 |
| RoPE | VC | 1.65 | 1.94 | 4.76 | 5.24 | 1.14 | 2.35 |
| RoPE | RePHINE | **1.59** | **1.77** | **4.49** | **4.70** | **1.00** | **2.05** |

在所有三个树任务上，PiPE（PE+PH）一致优于纯 PE 基线，PPL 降低幅度 13%–35%。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 纯 PE（无 PH） | 基准 | 仅位置编码 |
| 纯 PH（无 PE） | 低于 PiPE | 仅拓扑特征 |
| PH + PE（非学习组合） | 中等 | 简单拼接 |
| PiPE（学习统一） | 最优 | 闭环消息传递 |
| Identity 过滤 vs 学习过滤 | 学习过滤更优 | 可学习 $f_\ell$ 增强表达力 |

计算开销：PiPE 相比基础 PE 方法仅增加少量时间（在 Alchemy 上的每 epoch 时间与基础方法接近），主要开销来自 PH 嵌入计算。

### 关键发现

- PiPE 在 ZINC 数据集上与 PEG 组合取得最大 MAE 降低
- DrugOOD 的 OOD-Test 上，PiPE 展示最佳泛化性（ID-Test 各方法相近，OOD-Test 差距拉大）
- 在 OGBG-MOLPCBA（437.9k 图）上取得显著改进，验证大规模适用性
- 不同 PE 基础方法（LapPE/RWPE/PEG/SPE/SignNet）上均一致提升，证明框架通用性

## 亮点与洞察

- **首次严格证明 PE 与 PH 互不可比**：通过精心设计的图构造（anthracene 分子类似图、共享环图对等）给出了完整的理论刻画
- PiPE 设计理论优雅且实用，核心思路（PE 驱动 PH，PH 反馈 PE）自然统一了两种范式
- 框架即插即用：可基于任何现有 PE 方法，且兼容多种 PH 向量化方案（VC/RePHINE）
- 将拓扑数据分析与图表示学习在理论层面深度连接

## 局限与展望

- PH 计算（特别是 1 维持续同调）开销较大，对大规模图（>10K 节点）可扩展性需验证
- 目前仅考虑 0/1 维拓扑特征（连通分量和独立环），高维持续同调（空腔等）未探索
- CFI 类图对仍无法区分，说明在某些高度对称结构上仍有极限
- 计算限制在 1 维单纯复形，更高维的 Rips 复形可能带来进一步增益

## 相关工作与启发

- **Horn et al. (2022) TOGL**：首个将 PH 集成到 GNN 层的框架，PiPE 在此基础上引入 PE 驱动的过滤函数
- **Dwivedi et al. (2022) LSPE**：可学习位置编码框架，PiPE 直接扩展了其位置更新公式
- **Lim et al. (2023) SignNet/BasisNet**：处理 Laplacian 特征向量符号歧义，是 PiPE 可选的基础 PE
- **Immonen et al. (2023)**：刻画了 VC 过滤的表达力，PiPE 的 Lemma 3.3 直接推广了其结果
- **启发**：不同增强范式的互补性在更多领域（如几何学习、分子设计）值得探索

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次严格理论刻画 PE 与 PH 关系，PiPE 统一框架设计优雅
- 实验充分度: ⭐⭐⭐⭐ 覆盖表达力基准/分子预测/图分类/OOD/合成任务，消融全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论与实验衔接紧密，图示清晰，不可比性构造直观
- 价值: ⭐⭐⭐⭐ 对图表示学习表达力研究有深远影响，即插即用设计实用性强

<!-- RELATED:START -->

## 相关论文

- [Learnable Spatial-Temporal Positional Encoding for Link Prediction](learnable_spatial-temporal_positional_encoding_for_link_prediction.md)
- [Towards Graph Foundation Models: Learning Generalities Across Graphs via Task-Trees](towards_graph_foundation_models_learning_generalities_across_graphs_via_task-tre.md)
- [Balancing Efficiency and Expressiveness: Subgraph GNNs with Walk-Based Centrality](balancing_efficiency_and_expressiveness_subgraph_gnns_with_walk-based_centrality.md)
- [CoDy: Counterfactual Explainers for Dynamic Graphs](cody_counterfactual_explainers_for_dynamic_graphs.md)
- [TINED: GNNs-to-MLPs by Teacher Injection and Dirichlet Energy Distillation](tined_gnns-to-mlps_by_teacher_injection_and_dirichlet_energy_distillation.md)

<!-- RELATED:END -->
