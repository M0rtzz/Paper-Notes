---
title: >-
  [论文解读] Mitigating Over-Squashing in Graph Neural Networks by Spectrum-Preserving Sparsification
description: >-
  [ICML2025][图学习][过挤压] 提出 GOKU（稠密化-稀疏化重连范式），通过将输入图视为未知稠密潜在图的谱稀疏器并求解逆稀疏化问题，在增强图连通性的同时显式保留拉普拉斯谱，有效缓解 GNN 的 over-squashing 问题。
tags:
  - ICML2025
  - 图学习
  - 过挤压
  - 图重连
  - 谱稀疏化
  - 有效电阻
  - 图分类
  - 节点分类
---

# Mitigating Over-Squashing in Graph Neural Networks by Spectrum-Preserving Sparsification

**会议**: ICML2025  
**arXiv**: [2506.16110](https://arxiv.org/abs/2506.16110)  
**代码**: [GitHub](https://github.com/Jinx-byebye/GOKU)  
**领域**: 图学习 / 图神经网络  
**关键词**: 过挤压, 图重连, 谱稀疏化, 有效电阻, 图分类, 节点分类

## 一句话总结
提出 GOKU（稠密化-稀疏化重连范式），通过将输入图视为未知稠密潜在图的谱稀疏器并求解逆稀疏化问题，在增强图连通性的同时显式保留拉普拉斯谱，有效缓解 GNN 的 over-squashing 问题。

## 研究背景与动机

**Over-squashing 问题**：GNN 采用消息传递机制，当远距离节点之间的信息需要经过图的结构瓶颈时，消息在聚合过程中被压缩和失真，导致 GNN 无法捕获长距离依赖关系。这一问题在更深网络中愈发严重，因为感受野增长但节点表示向量大小固定。

**现有图重连方法的两大局限**：

**不保留图谱性质**：Delaunay 重连仅依据节点特征构造图、完全忽略原始拓扑；Graph Transformer 依赖全连接图；Expander 重连甚至引入全新节点集。图谱（拉普拉斯特征值集合）编码了社区结构、节点中心度等关键拓扑信息，对聚类和半监督学习至关重要。

**增加边数带来计算开销和 over-smoothing 风险**：更稠密的输出图使得邻居信息过度混合，节点特征趋于一致。

**核心动机**：能否在增强连通性的同时保持稀疏性并显式保留原始图谱？作者通过实验验证了谱性质与分类任务的关联——同类图/节点的谱相似度显著高于异类（Table 1），证明保留图谱的必要性。

## 方法详解

### DSR 范式：稠密化-稀疏化重连

整体流程为 $G \xrightarrow{\text{稠密化}} G_l \xrightarrow{\text{稀疏化}} G_o$，其中 $G$ 为输入图，$G_l$ 为潜在稠密图，$G_o$ 为最终输出图。

**关键思想**：将输入图 $G$ 视为某未知稠密图 $G_l$ 经谱稀疏化后的结果。$G$ 中的结构瓶颈源于稀疏化过程中去除了对连通性至关重要的边。通过逆稀疏化恢复这些"缺失边"，再做正向稀疏化剪掉不重要的边，最终输出的 $G_o$ 同时满足：

- 增强连通性（降低有效电阻）
- 保持稀疏性（边数不超过原图）
- 显式保留图谱：$G_o \overset{(1\pm\epsilon)^2}{\approx} G$

### 稠密化阶段：非重要性谱稀疏化 (USS) 的逆过程

**USS 采样概率**：对边 $e=(u,v)$，采样概率为

$$p_e \propto \frac{\deg(u) + \deg(v) + 1}{|f_u - f_v|}$$

其中 $f_u, f_v$ 为 Fiedler 向量（拉普拉斯第二小特征值对应的特征向量）分量。直觉：$|f_u - f_v|$ 大说明 $u,v$ 连通性弱，度数小说明它们与图的其余部分连接弱——这类边对连通性至关重要，在 USS 中被优先去除。

**逆稀疏化 MLE**：将问题建模为最大似然估计，寻找最可能产生 $G$ 的潜在图 $G_l$：

$$G_l = \underset{G_d}{\arg\max}\; \mathbb{P}(\phi(G_d) = G)$$

通过独立性假设和简化，将似然分解为边级概率乘积。为降低搜索空间：假设 $G_l$ 边权均匀；从 Fiedler 值最大和度数最小的节点中构造候选边集（约 $16|E'|$ 条候选边），按目标函数值采样 $|E'|$ 条缺失边。

**理论保证**（Theorem 4.1）：当采样次数 $q \geq \frac{\kappa^2}{2\epsilon^2}\log 8$ 时，USS 以至少 3/4 的概率产生 $(1\pm\epsilon)$-谱稀疏器。

### 稀疏化阶段：重要性谱稀疏化 (ISS)

对潜在图 $G_l$ 应用基于有效电阻 (ER) 的 ISS。每条边 $(u,v)$ 的采样概率为：

$$p_e \propto (1 + S_e) \cdot R_e$$

其中 $S_e = (1+\cos(x_u, x_v))/2$ 为归一化余弦相似度，$R_e$ 为有效电阻。该设计优先保留高 ER（连通性关键）且节点特征相似（同质性增强）的边，同时过滤低相似度的噪声边。

采样至 $\beta|E|$ 条不同边为止（$0.5 < \beta \leq 1$），确保输出图不比原图更稠密。

**理论保证**（Theorem 4.2）：ISS 产生 $(1\pm\epsilon)$-谱稀疏器，采样复杂度 $O(n\log n / \epsilon^2)$。

### 复杂度

整体时间复杂度 $O(|E'|m) + \tilde{O}(m/\delta^2)$，其中 ER 近似采用 Koutis et al. (2014) 方法在亚线性时间内完成。实验验证运行时间随图规模近线性增长。

## 实验关键数据

在 10 个数据集（6 节点分类 + 4 图分类）上，与 SDRF、FoSR、BORF、GTR、DR、LASER 等 6 种 SOTA 重连方法比较。

### GCN 骨架下的分类精度（Table 2，部分）

| 方法 | Texas | Cornell | Wisconsin | Mutag | 平均排名 |
|------|-------|---------|-----------|-------|----------|
| NONE | 44.2 | 41.5 | 44.6 | 68.8 | 6.60 |
| BORF | 49.4 | 50.8 | 50.3 | 75.8 | 3.40 |
| DR | 67.8 | 57.8 | 62.8 | 80.1 | 4.55 |
| **GOKU** | **72.4** | **69.4** | **68.8** | **81.0** | **1.90** |

### GIN 骨架下的分类精度（Table 3，部分）

| 方法 | Texas | Cornell | Wisconsin | IMDB | 平均排名 |
|------|-------|---------|-----------|------|----------|
| NONE | 46.4 | 40.2 | 42.1 | 67.7 | 6.20 |
| LASER | 46.5 | 44.5 | 46.1 | 68.6 | 3.40 |
| **GOKU** | **60.2** | **49.5** | **57.6** | **71.3** | **1.80** |

GOKU 在异配图（Texas/Cornell/Wisconsin）上优势尤为突出，验证了谱保留重连对异配场景的有效性。

### 消融实验（Table 4）

| 变体 | Mutag | Chameleon | Cora |
|------|-------|-----------|------|
| GOKU-D（仅稠密化） | 78.0 | 62.7 | 86.4 |
| GOKU-S（仅稀疏化） | 76.8 | 62.9 | 86.4 |
| GOKU（完整） | **81.0** | **63.2** | **86.8** |

稠密化和稀疏化缺一不可：单独稠密化可能因过度连接导致 over-smoothing；单独稀疏化无法显著提升连通性。

### 连通性改善

在 SBM 合成图上，GOKU 重连后有效电阻分布显著左移（更低 ER = 更好连通性），同时同质度一致提升（如 L-Low: 0.385→0.433）。

## 亮点与洞察

1. **逆稀疏化视角新颖**：将输入图视为未知稠密图的谱稀疏器，通过"恢复缺失边"来增强连通性，概念优雅且有理论支撑
2. **三重保证同时实现**：增强连通性 + 保持稀疏性 + 显式保留图谱，是首个同时满足这三条的重连方法
3. **理论完备**：USS 和 ISS 均有谱稀疏器的概率保证（Theorem 4.1, 4.2），非启发式方法
4. **在异配图上大幅领先**：Texas 从 44.2→72.4，Cornell 从 41.5→69.4，提升幅度远超现有方法
5. **节点特征感知的稀疏化**：ISS 中融入余弦相似度，兼顾拓扑和特征信息

## 局限与展望

1. **加权图假设**：谱稀疏化依赖加权图来保证谱保留，对无权图在不同边数之间保留谱本身就很困难，引入的边权可能为 GNN 训练带来额外复杂度
2. **超参数 $\alpha, \beta$ 需调优**：虽然搜索空间不大（$\alpha \in \{5..30\}, \beta \in [0.5,1]$），但不同数据集的最优值可能不同
3. **稠密化阶段的候选边构建启发式**：基于 Fiedler 值和度排序选候选节点，虽有直觉支撑但缺乏理论最优保证
4. **大规模图的可扩展性**：2000 节点图需约 65-72 秒，面对更大规模图（百万节点）可能仍有瓶颈

## 相关工作与启发

- **FoSR** (Karhadkar et al., 2023)：通过添加边改善谱隙的一阶近似，但不保谱
- **BORF** (Nguyen et al., 2023)：基于曲率的重连，同时添加和删除边
- **DR** (Attali et al., 2024)：Delaunay 重连完全基于节点特征，丢弃原始拓扑
- **LASER** (Barbero et al., 2024)：局部感知重连限制在 k-hop 邻域，但不显式保谱
- **Spielman & Srivastava (2008)**：ER 基谱稀疏化的经典工作，GOKU 的 ISS 阶段建立在此基础上

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 逆稀疏化视角在图重连领域首创，DSR 范式设计精巧
- 实验充分度: ⭐⭐⭐⭐ — 10 数据集 × 3 骨架模型 × 消融 × 谱可视化，但缺大规模图实验
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、理论严谨，但符号较多读起来需耐心
- 价值: ⭐⭐⭐⭐⭐ — 在异配图上大幅提升，为 GNN 的 over-squashing 问题提供了理论与实践兼优的解决方案

<!-- RELATED:START -->

## 相关论文

- [Over-squashing in Spatiotemporal Graph Neural Networks](../../NeurIPS2025/graph_learning/over-squashing_in_spatiotemporal_graph_neural_networks.md)
- [GrokFormer: Graph Fourier Kolmogorov-Arnold Transformers](grokformer_graph_fourier_kolmogorov-arnold_transformers.md)
- [Hyperbolic-PDE GNN: Spectral Graph Neural Networks in the Perspective of A System of Hyperbolic Partial Differential Equations](hyperbolic-pde_gnn_spectral_graph_neural_networks_in_the_perspective_of_a_system.md)
- [On Measuring Long-Range Interactions in Graph Neural Networks](on_measuring_long-range_interactions_in_graph_neural_networks.md)
- [Open Your Eyes: Vision Enhances Message Passing Neural Networks in Link Prediction](open_your_eyes_vision_enhances_message_passing_neural_networks_in_link_predictio.md)

<!-- RELATED:END -->
