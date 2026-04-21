---
title: >-
  [论文解读] Cooperative Sheaf Neural Networks
description: >-
  [ICLR 2026][图学习][Sheaf Neural Networks] 提出在有向图上定义 cellular sheaf 的 in/out-degree Laplacian，构建 Cooperative Sheaf Neural Network (CSNN)，使节点能独立选择信息传播/接收策略，从而同时缓解过度挤压(oversquashing)和处理异配(heterophilic)任务。
tags:
  - ICLR 2026
  - 图学习
  - Sheaf Neural Networks
  - 协作行为
  - 有向图
  - 过度挤压
  - 异配图
---

# Cooperative Sheaf Neural Networks

**会议**: ICLR 2026  
**arXiv**: [2507.00647](https://arxiv.org/abs/2507.00647)  
**代码**: 无  
**领域**: 图学习 / 图神经网络  
**关键词**: Sheaf Neural Networks, 协作行为, 有向图, 过度挤压, 异配图

## 一句话总结

提出在有向图上定义 cellular sheaf 的 in/out-degree Laplacian，构建 Cooperative Sheaf Neural Network (CSNN)，使节点能独立选择信息传播/接收策略，从而同时缓解过度挤压(oversquashing)和处理异配(heterophilic)任务。

## 研究背景与动机

**领域现状**：Sheaf Neural Networks (SNNs) 通过在图上定义 cellular sheaf 来泛化 GNN 的扩散机制，已被证明能处理异配任务并缓解过平滑(oversmoothing)。

**现有痛点**：经典 SNNs 基于无向图，节点无法独立选择"仅传播信息"或"仅接收信息"。若某节点 $i$ 要屏蔽所有邻居的输入，必须将所有关联的 restriction map 置零 $\mathcal{F}_{i \unlhd e}=0$，这同时也阻断了 $i$ 向外传播信息的能力。

**核心矛盾**：SNNs 的 sheaf Laplacian 结构使得 PROPAGATE 蕴含 LISTEN，无法实现四种协作行为(STANDARD/LISTEN/PROPAGATE/ISOLATE)的完全解耦。

**本文目标** 让 SNN 中的节点能独立决定是否传播和/或接收信息，实现真正的协作行为，以更好地缓解 oversquashing。

**切入角度**：将无向边拆分为一对有向边，在有向图上定义 cellular sheaf 及其 in/out-degree sheaf Laplacian。

**核心 idea**：通过有向图上的 sheaf Laplacian 分离源映射 $\mathbf{S}_i$ 和目标映射 $\mathbf{T}_i$，使每个节点可独立控制信息流入和流出方向。

## 方法详解

### 整体框架

CSNN 将输入无向图转为有向图（每条无向边拆为一对有向边），为每个节点 $i$ 学习一对 conformal 映射 $\mathbf{S}_i$（源映射）和 $\mathbf{T}_i$（目标映射），然后通过组合 out-degree 和转置 in-degree sheaf Laplacian 进行归一化扩散，最后结合 NSD 风格的迭代更新完成节点表示学习。

### 关键设计

1. **有向图 Cellular Sheaf 与 In/Out-Degree Laplacian**:

    - 做什么：定义有向图上的 sheaf 结构，区分节点作为源和目标时的 restriction map
    - 核心思路：Out-degree sheaf Laplacian $L_{\mathcal{F}}^{\text{out}}(\mathbf{X})_i = \sum_{j \in N(i)} (\mathbf{S}_i^\top \mathbf{S}_i \mathbf{x}_i - \mathbf{T}_i^\top \mathbf{S}_j \mathbf{x}_j)$，In-degree 类似但用 $\mathbf{T}$ 控制接收端。通过组合 $(\Delta_\mathcal{F}^{\text{in}})^\top \Delta_\mathcal{F}^{\text{out}}$ 实现非对称扩散
    - 设计动机：无向 sheaf Laplacian 中 $\mathcal{F}_{i \unlhd e}=0$ 同时切断传入和传出（Proposition 3.1），有向拆分后 $\mathbf{S}_i=0$（不传播）和 $\mathbf{T}_i=0$（不监听）可独立设置

2. **Flat Vector Bundle 高效参数化**:

    - 做什么：用每节点仅两个 conformal 映射替代每条边的 restriction map
    - 核心思路：对所有邻居 $j$ 共享 $\mathcal{F}_{i \unlhd ij} = \mathbf{S}_i$、$\mathcal{F}_{i \unlhd ji} = \mathbf{T}_i$，通过 Householder 反射构造正交矩阵再乘以学习的正常数
    - 设计动机：一般 cellular sheaf 有 $2m$ 个 restriction map（$m$ 为边数），flat vector bundle 仅需 $2n$ 个（$n$ 为节点数），大幅降低计算量

3. **扩展感受野与选择性注意**:

    - 做什么：理论证明 CSNN 每层可访问 $2t$-hop 邻居，并可选择性忽略路径上的中间节点
    - 核心思路：通过合理配置 $\mathbf{S}$ 和 $\mathbf{T}$ 映射，使 $\partial \mathbf{x}_i^{(t)} / \partial \mathbf{x}_j^{(0)}$ 对距离为 $t$ 的目标节点 $j$ 有高灵敏度，同时对中间节点趋近零
    - 设计动机：传统 GNN $t$ 层只能访问 $t$-hop 邻居，且信息沿路径指数压缩导致 oversquashing；CSNN 的选择性注意可有效缓解

### 损失函数 / 训练策略

采用 NSD 风格的扩散迭代，restriction map 通过神经网络端到端学习。使用 Householder 反射保证正交性，乘以学习的正常数构成 conformal 映射。

## 实验关键数据

### 主实验

| 数据集 | 指标 | CSNN | 最优对比 | 提升 |
|--------|------|------|----------|------|
| roman-empire | Acc | 92.63 | BuNN 91.75 | +0.88 |
| minesweeper | AUROC | 99.07 | BuNN 98.99 | +0.08 |
| tolokers | AUROC | 85.45 | CO-GNN 84.84 | +0.61 |
| questions | AUROC | 79.31 | BuNN 78.75 | +0.56 |
| Wisconsin | Acc | 90.00 | O(d)-NSD 89.41 | +0.59 |

### 消融实验

| 配置 | NeighborsMatch 准确率 | 说明 |
|------|----------------------|------|
| CSNN (r=2~8) | 100% 全部深度 | 完美解决 oversquashing |
| BuNN (r≥7) | 71%→42% | r=7 开始退化 |
| NSD (r≥4) | 5% | 严重 oversquashing |
| GCN/GIN (r≥4) | 失败 | 无法处理长距离 |

### 关键发现

- CSNN 在 NeighborsMatch 所有树深度上保持 100% 准确率，显著优于所有 sheaf 和非 sheaf 基线
- 在 11 个节点分类数据集中 9 个取得最优，尤其在强异配数据集上表现突出
- 在 peptides-func 图分类任务上达到 73.38 AP，超过 BuNN (72.76)、GPS、SAN 等方法

## 亮点与洞察

- 从代数拓扑角度严格证明 SNNs 无法实现协作行为（Proposition 3.1），然后用有向 sheaf 优雅地解决
- Flat vector bundle 设计使参数量从 $O(m)$ 降到 $O(n)$，在理论优势之外还保证了计算效率
- 理论证明 CSNN 每层感受野为 $2t$-hop 而非传统 $t$-hop，为缓解 oversquashing 提供新思路

## 局限与展望

- 协作行为的选择通过连续参数隐式决定，未显式建模离散动作
- 在 amazon-ratings 等部分数据集上未取得最优，flat vector bundle 的简化可能牺牲了灵活性
- 仅在中等规模图上验证，大规模图（>100K 节点）的可扩展性有待评估

## 相关工作与启发

- **vs CO-GNN**: CO-GNN 使用离散 Gumbel-Softmax 动作网络选择协作模式，CSNN 通过连续参数自然实现，避免了训练不稳定和超参敏感问题
- **vs NSD**: NSD 基于无向 sheaf，CSNN 通过有向 sheaf 扩展了表达能力，在 NeighborsMatch 上表现远超 NSD
- **vs BuNN**: BuNN 也是 sheaf-based，但在 r≥7 的 oversquashing 测试中明显退化，CSNN 始终保持 100%

## 评分

- 新颖性: ⭐⭐⭐⭐ 有向 sheaf Laplacian 是全新数学构造，理论贡献扎实
- 实验充分度: ⭐⭐⭐⭐ 合成 + 11个节点分类 + 2个图分类，覆盖全面
- 写作质量: ⭐⭐⭐⭐ 定义-命题-证明结构清晰，数学严谨
- 价值: ⭐⭐⭐⭐ 为 sheaf-based GNN 提供了新的理论和实践方向

<!-- RELATED:START -->

## 相关论文

- [Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization](../../AAAI2026/graph_learning/sheaf_graph_neural_networks_via_pac-bayes_spectral_optimization.md)
- [Are We Measuring Oversmoothing in Graph Neural Networks Correctly?](are_we_measuring_oversmoothing_in_graph_neural_networks_correctly.md)
- [LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks](logicxgnn_grounded_logical_rules_for_explaining_graph_neural_networks.md)
- [Hyperbolic Busemann Neural Networks](../../CVPR2026/graph_learning/hyperbolic_busemann_neural_networks.md)
- [Pairwise is Not Enough: Hypergraph Neural Networks for Multi-Agent Pathfinding](pairwise_is_not_enough_hypergraph_neural_networks_for_multi-agent_pathfinding.md)

<!-- RELATED:END -->
