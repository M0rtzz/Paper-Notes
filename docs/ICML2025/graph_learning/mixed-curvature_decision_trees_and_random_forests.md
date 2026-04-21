---
title: >-
  [论文解读] Mixed-Curvature Decision Trees and Random Forests
description: >-
  [ICML 2025][图学习][决策树] 将经典决策树和随机森林算法从欧几里得空间推广到混合曲率乘积流形（hyperbolic × spherical × Euclidean），通过角度重参数化（angular reformulation）构造尊重流形几何的分裂准则，在 57 个分类/回归/链路预测任务上表现优异（29 个第一，41 个前二）。
tags:
  - ICML 2025
  - 图学习
  - 决策树
  - 随机森林
  - 乘积流形
  - 混合曲率
  - 双曲空间
  - 球面空间
  - 非欧几何
---

# Mixed-Curvature Decision Trees and Random Forests

**会议**: ICML 2025  
**arXiv**: [2410.13879](https://arxiv.org/abs/2410.13879)  
**作者**: Philippe Chlenski, Quentin Chu, Raiyan R. Khan, Kaizhu Du, Antonio Khalil Moretti, Itsik Pe'er
**机构**: Columbia University
**代码**: [pchlenski/manify](https://github.com/pchlenski/manify)  
**领域**: 图学习  
**关键词**: 决策树, 随机森林, 乘积流形, 混合曲率, 双曲空间, 球面空间, 非欧几何

## 一句话总结

将经典决策树和随机森林算法从欧几里得空间推广到混合曲率乘积流形（hyperbolic × spherical × Euclidean），通过角度重参数化（angular reformulation）构造尊重流形几何的分裂准则，在 57 个分类/回归/链路预测任务上表现优异（29 个第一，41 个前二）。

## 研究背景与动机

**非欧嵌入的兴起**：大量研究表明，树状层次数据适合嵌入到双曲空间（负曲率），环状/周期数据适合嵌入到球面空间（正曲率）。现实数据往往包含多种拓扑结构，因此乘积流形 $\mathcal{P} = \mathcal{M}_1 \times \mathcal{M}_2 \times \cdots \times \mathcal{M}_k$（各分量曲率不同）成为更合适的嵌入空间。

**传统方法的局限**：经典 DT/RF 仅在欧几里得空间工作，直接应用到非欧空间会破坏几何结构。先前工作（HyperDT）仅处理纯双曲空间，无法应对混合曲率。

**乘积流形上 ML 工具匮乏**：虽然已有在乘积流形上的感知机、SVM、GCN 等，但缺少简单、可解释、无需梯度优化的非参数方法。DT/RF 因其可解释性和鲁棒性，是理想的候选工具。

**核心挑战**：如何在每个分量流形上定义"分裂超平面"，使其既尊重流形几何，又保留 DT 的递归贪心分裂框架？

## 方法详解

### 整体框架

将乘积流形 $\mathcal{P}$ 的每个分量流形上的坐标转换为角度表示，然后在角度空间中进行分裂。整体流程：

1. **预处理**：将环境坐标 $\mathbf{x} \in \mathbb{R}^{d_{\text{ambient}}}$ 映射为角度向量 $\boldsymbol{\theta} \in [-\pi, \pi)^{d_{\text{intrinsic}}}$
2. **递归分裂**：在角度空间中选择最佳分裂维度和阈值
3. **测地中点**：用流形特定的测地中点作为分裂边界
4. **随机森林集成**：对特征和样本进行随机子采样，构建多棵树

### 关键设计

#### 1. 角度重参数化

对每个分量流形 $\mathcal{M}_i$，将点投影到二维子空间并计算角度：

$$\theta(\mathbf{x}, d) = \arctan\!\left(\frac{x_0}{x_d}\right)$$

其中 $x_0$ 是参考维度坐标，$x_d$ 是目标维度坐标。

- **双曲空间 $\mathbb{H}^n$**：$x_0$ 是时间维（特殊维），投影角度自然对应于 Lorentz 模型中的方向
- **球面空间 $\mathbb{S}^n$**：同样使用第一个维度，角度对应于球面上的经度/纬度
- **欧几里得空间 $\mathbb{E}^n$**：使用哑元维 (dummy dimension = 1)，$\theta = \arctan(1/x_d)$，退化为标准欧几里得分裂

#### 2. 角度分裂准则

给定角度阈值 $\theta^*$，分裂规则为：

$$S(\mathbf{x}, d, \theta^*) = \begin{cases} 1 & \text{if } \theta(\mathbf{x},d) \in [\theta^*, \theta^* + \pi) \\ 0 & \text{otherwise} \end{cases}$$

核心比较操作 `_angular_greater` 计算 $(\theta_{\text{key}} - \theta_{\text{query}} + \pi) \bmod 2\pi \geq \pi$，确保周期性正确。

#### 3. 测地中点（Geodesic Midpoints）

为了将决策边界放在最优位置，使用流形特定的测地中点：

| 流形类型 | 中点公式 |
|---------|---------|
| 欧几里得 $\mathbb{E}^n$ | $m_{\mathbb{E}}(\theta_u, \theta_v) = \arctan\!\left(\frac{2}{u_d + v_d}\right)$ |
| 双曲 $\mathbb{H}^n$ | $m_{\mathbb{H}}(\theta_u, \theta_v) = \arctan\!\left(\frac{u_0 + v_0}{u_d + v_d}\right)$（涉及 sinh/cosh） |
| 球面 $\mathbb{S}^n$ | $m_{\mathbb{S}}(\theta_u, \theta_v) = \arctan\!\left(\frac{u_0 + v_0}{u_d + v_d}\right)$（涉及 sin/cos） |

消融实验证实，使用流形特定中点比简单角度平均更优。

#### 4. 信息增益与最佳分裂选择

- **分类任务**：使用 Gini 不纯度作为信息增益度量
- **回归任务**：使用 MSE 作为分裂准则
- 遍历所有维度和所有候选分裂点（每个训练样本的角度），选择信息增益最大的 $(d^*, \theta^*)$
- 然后找到负类中离 $\theta^*$ 最近的角度，计算测地中点作为最终分裂阈值

#### 5. 随机森林扩展（ProductSpaceRF）

- **特征子采样**：`max_features` 支持 `sqrt`、`log2`
- **样本子采样**：`max_samples` 控制 bootstrap 比例
- **集成预测**：分类用多数投票（概率平均），回归用均值

### 损失函数

本方法为非参数方法，不使用梯度优化。分裂选择通过贪心最大化信息增益实现：

$$\text{IG}(d, \theta) = H(Y) - \sum_{s \in \{L, R\}} \frac{|S_s|}{|S|} H(Y_s)$$

其中 $H$ 为 Gini 不纯度（分类）或 MSE（回归）。

## 实验关键数据

### 主实验：57 个任务总排名

| 方法 | 第1名次数 | 前2名次数 | 任务类型覆盖 |
|------|----------|----------|------------|
| **Product RF (本文)** | **29** | **41** | 分类+回归+链路预测 |
| Product DT | ~12 | ~22 | 分类+回归+链路预测 |
| Tangent RF | ~8 | ~15 | 分类+回归 |
| Sklearn RF | ~5 | ~12 | 分类+回归 |
| κ-GCN | ~4 | ~10 | 分类+链路预测 |
| Product SVM | ~3 | ~8 | 分类 |
| Product Perceptron | ~2 | ~5 | 分类 |

### 各任务类型详细结果

| 任务类型 | 数据集数量 | Product RF 排名1 | 主要竞争对手 |
|---------|-----------|-----------------|------------|
| 分类（单流形） | 15 | ~7 | Sklearn RF, Tangent RF |
| 分类（乘积流形） | 15 | ~10 | κ-GCN, Tangent RF |
| 回归 | 12 | ~6 | Tangent RF, Sklearn RF |
| 链路预测 | 15 | ~6 | κ-GCN, Fermi-Dirac |

### 消融实验关键发现

1. **中点消融**：移除测地中点（使用简单角度平均）导致分类精度下降 1-3%，回归 MSE 增大
2. **特征选择模式**：`d_choose_2`（所有维度对）在高维流形上优于 `d`（仅参考维度），但计算开销更大
3. **单流形退化**：在纯双曲数据上 ≈ HyperDT 性能，在纯欧几里得数据上 ≈ sklearn DT，验证了方法的正确性
4. **曲率估计灵敏度**：使用 greedy signature selection 估计流形签名，错误签名仍有合理性能但不如正确签名
5. **树数量**：100 棵树后收敛，与经典 RF 行为一致

## 亮点与洞察

1. **统一框架的优雅性**：通过角度重参数化，双曲/球面/欧几里得空间的分裂规则统一为同一形式，仅在中点计算上因流形类型而异。这是非欧 ML 中少见的"一个公式搞定所有曲率"的方案。

2. **无需梯度优化**：与 GCN、感知机等需要反向传播的方法不同，DT/RF 是纯贪心递归的，训练稳定、无超参数调优（学习率等），适合资源受限场景。

3. **可解释性保留**：继承了 DT 的可解释性——每个分裂节点对应特定流形分量上的角度阈值，可直接解读为"该维度方向是否超过某个测地角度"。

4. **实用性强**：代码开源（manify 库），API 设计与 scikit-learn 兼容，支持分类/回归/链路预测三类任务。

5. **实验规模充分**：57 个数据集、13 种基线、3 种任务类型，覆盖面远超单一基准测试。

## 局限性

1. **计算复杂度**：角度比较矩阵为 $O(n^2 \times d)$（batched 模式），大数据集上内存和时间受限。
2. **嵌入质量依赖**：方法假设数据已被正确嵌入到乘积流形中，嵌入质量直接影响分类/回归效果。
3. **曲率签名需要先验**：需要知道或估计正确的乘积流形签名（各分量的曲率和维度），错误签名会降低性能。
4. **仅支持 Lorentz/ambient 表示**：不支持 Poincaré 球等截面投影表示（stereographic），需要坐标转换。
5. **链路预测为间接支持**：通过 Fermi-Dirac 解码器将距离转化为分类问题，非端到端。

## 相关工作与启发

- **HyperDT (Chlenski et al., 2024)**：本文直接扩展自 HyperDT，从纯双曲推广到乘积流形
- **Product Space Forms (Tabaghi et al., 2021)**：线性分类器在乘积空间中的工作，本文借鉴了 greedy signature selection
- **κ-GCN (Bachmann et al., 2020)**：乘积流形上的 GCN，是主要基线对手
- **HGCN (Chami et al., 2019)**：双曲图卷积网络，Fermi-Dirac 解码器来源

**启发**：该工作展示了"将经典 ML 方法几何化"的范式——不是设计全新的非欧算法，而是找到恰当的坐标变换使得经典方法自然适配非欧几何。这个思路可推广到 k-means、高斯混合模型等更多经典方法。

## 评分

- **创新性**: ★★★★☆ — 角度重参数化思路优雅，但核心是 HyperDT 的自然推广
- **实用性**: ★★★★★ — 代码完善、API 友好、任务覆盖广、无需调参
- **实验充分度**: ★★★★★ — 57 个数据集 + 13 个基线 + 3 种任务类型
- **论文质量**: ★★★★☆ — 30 页完整论文，数学推导清晰，消融全面

<!-- RELATED:START -->

## 相关论文

- [WILTing Trees: Interpreting the Distance Between MPNN Embeddings](wilting_trees_interpreting_the_distance_between_mpnn_embeddings.md)
- [Towards Graph Foundation Models: Learning Generalities Across Graphs via Task-Trees](towards_graph_foundation_models_learning_generalities_across_graphs_via_task-tre.md)
- [TINED: GNNs-to-MLPs by Teacher Injection and Dirichlet Energy Distillation](tined_gnns-to-mlps_by_teacher_injection_and_dirichlet_energy_distillation.md)
- [GlycanAA: Modeling All-Atom Glycan Structures via Hierarchical Message Passing and Multi-Scale Pre-training](modeling_all-atom_glycan_structures_via_hierarchical_message_passing_and_multi-s.md)
- [On Measuring Long-Range Interactions in Graph Neural Networks](on_measuring_long-range_interactions_in_graph_neural_networks.md)

<!-- RELATED:END -->
