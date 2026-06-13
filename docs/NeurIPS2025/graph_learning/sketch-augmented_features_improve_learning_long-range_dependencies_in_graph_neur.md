---
title: >-
  [论文解读] Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks
description: >-
  [NeurIPS 2025][图学习][图神经网络] 提出Sketched Random Features (SRF)，将节点特征的核空间随机投影注入标准消息传递GNN的每一层，同时缓解过压缩、过平滑和表达力受限三大问题，理论性质完备且计算高效。 标准消息传递GNN（MPGNN）面临三个公认的基本限制： 过压缩（Oversq…
tags:
  - "NeurIPS 2025"
  - "图学习"
  - "图神经网络"
  - "过压缩"
  - "过平滑"
  - "随机特征"
  - "Johnson-Lindenstrauss变换"
---

# Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks

**会议**: NeurIPS 2025  
**arXiv**: [2511.03824](https://arxiv.org/abs/2511.03824)  
**代码**: [GitHub](https://github.com/ryienh/sketched-random-features)  
**领域**: 图学习  
**关键词**: 图神经网络, 过压缩, 过平滑, 随机特征, Johnson-Lindenstrauss变换

## 一句话总结

提出Sketched Random Features (SRF)，将节点特征的核空间随机投影注入标准消息传递GNN的每一层，同时缓解过压缩、过平滑和表达力受限三大问题，理论性质完备且计算高效。

## 研究背景与动机

标准消息传递GNN（MPGNN）面临三个公认的基本限制：

**过压缩（Oversquashing）**：远距离节点的信号经多跳传播后被压缩到固定维度向量中，信息严重丢失。感受野指数增长但隐藏维度 $|\mathbf{h}|$ 固定，导致长程依赖捕获困难

**过平滑（Oversmoothing）**：随网络深度增加，节点表示指数收敛到近乎相同的值，用Dirichlet能量衡量：$\mathcal{D}(H^{(\ell)}) = \frac{1}{N}\sum_{i}\sum_{j \in \mathcal{N}(i)}\|\mathbf{h}_i^{(\ell)} - \mathbf{h}_j^{(\ell)}\|_2^2$，该值随 $\ell$ 增长快速衰减

**表达力受限**：MPGNN的表达力不超过1-WL测试，无法区分许多非同构图

现有解法要么引入 $O(N^2)$ 的Graph Transformer，要么依赖纯随机节点特征（收敛慢）或拓扑结构编码（难以同时唯一、距离敏感且等变）。

**关键观察**：现有方法主要利用拓扑信息构建位置编码，而忽视了真实图中通常丰富的节点特征本身。SRF直接从节点特征出发，提供全局、特征距离敏感的信号。

## 方法详解

### 整体框架

SRF分两步预处理：(1) 核嵌入算子 $\mathcal{E}$ 将节点特征映射到随机核特征空间；(2) 草图算子 $\mathcal{S}$ 在节点间施加随机投影混合全局信息。产出的草图特征在GNN每层拼接到节点隐状态。

### 关键设计

1. **核嵌入算子 $\mathcal{E}$**：对每个节点特征独立应用核近似映射

   对原始特征矩阵 $X \in \mathbb{R}^{N \times F}$，考虑随机傅里叶特征等核近似：
    $\varphi_{\text{RBF}}(\mathbf{x}) = \sqrt{\frac{2}{D}} [\cos(\boldsymbol{\omega}_1^\top \mathbf{x} + b_1), \ldots, \cos(\boldsymbol{\omega}_D^\top \mathbf{x} + b_D)]^\top$
   
   满足 $\mathbb{E}[\varphi(\mathbf{x}_i)^\top \varphi(\mathbf{x}_j)] = \kappa(\mathbf{x}_i, \mathbf{x}_j)$（无偏核估计）。支持线性核、RBF核、拉普拉斯核三种选择。
   
   设计动机：核映射保留特征空间的距离关系，使后续投影有意义。

2. **加性高斯草图算子 $\mathcal{S}_{AG}$**：跨节点随机投影混合全局信息

    $\mathcal{S}_{AG}(\Phi) = \left(I + \frac{1}{\sqrt{N}} G\right) \Phi$
   
   其中 $G \in \mathbb{R}^{N \times N}$，$G_{ij} \sim \mathcal{N}(0,1)$ i.i.d.。通过 $k$ 阶投影 $\mathcal{S}_{AG}^{(k)}$ 拼接 $k$ 个独立草图实现多维估计。

   最终SRF定义为：$Z = \mathcal{S}^{(k)}(\mathcal{E}(X)) \in \mathbb{R}^{N \times (kD)}$
   
   设计动机：与传统JLT用于降维不同，这里利用随机投影实现跨节点信息混合，每个 $\mathbf{z}_i$ 包含所有节点特征的线性组合，绕过图的拓扑瓶颈。

3. **注入MPGNN**：在每一层将SRF拼接到隐状态

    $\tilde{\mathbf{h}}_i^{(\ell)} = [\mathbf{h}_i^{(\ell)} | \mathbf{z}_i]$
    $\mathbf{h}_i^{(\ell+1)} = f\left(\tilde{\mathbf{h}}_i^{(\ell)}, \{\tilde{\mathbf{h}}_j^{(\ell)} : j \in \mathcal{N}(i)\}\right)$
   
   SRF仅计算一次，之后每层复用，训练代价极低。

### 理论性质

SRF满足五个关键性质，数学上证明其缓解GNN三大限制：

| 性质 | 内容 | 对应限制 |
|------|------|---------|
| 无偏核估计 | $\mathbb{E}[\mathbf{z}_i^\top \mathbf{z}_j] = \kappa(\mathbf{x}_i, \mathbf{x}_j)$ | 过平滑 |
| 核距离敏感 | $(1-c)\|\varphi(\mathbf{x}_i) - \varphi(\mathbf{x}_j)\| \leq \|\mathbf{z}_i - \mathbf{z}_j\| \leq (1+c)\|\ldots\|$ | 过平滑 |
| 跨节点信息 | 每个 $\mathbf{z}_i$ 含所有节点嵌入的线性组合 | 过压缩 |
| 几乎必然唯一 | $P(\mathbf{z}_i \neq \mathbf{z}_j) = 1, \forall i \neq j$ | 表达力 |
| 期望置换等变 | $\mathbb{E}[f(P_\pi X)] = \mathbb{E}[P_\pi f(X)]$ | 等变性 |

### 复杂度分析

使用结构化随机矩阵(SRM)后，整体复杂度为 $O(NFD + kN^2 \log N)$，存储 $O(N)$。SRF仅预计算一次，不随训练epoch增长。

## 实验关键数据

### 合成任务：表达力与长程依赖

| 数据集 | Baseline GNN | 消融($\mathcal{S}_{id}$) | SRF ($\mathcal{S}_{AG}^{(1)}$) | SRF ($\mathcal{S}_{AG}^{(8)}$) |
|-------|-------------|----------------------|-------|-------|
| CSL (Acc↑) | 0.100 | 0.100 | 1.000 | 1.000 |
| EXP (Acc↑) | 0.518 | 0.520 | 1.000 | 1.000 |

基线无法区分的非同构图，SRF增强后完美区分。

### 真实数据集性能

| 数据集 | GINE (基线) | R-PEARL | SRF-$\mathcal{E}_{\text{RBF}}$ |
|-------|------------|---------|------|
| REDDIT-B (Acc↑) | 91.8 | 93.0 | **94.13** |
| REDDIT-M (Acc↑) | 56.9 | 59.4 | **60.53** |
| DrugOOD-Assay (AUC↑) | 71.68 | 72.24 | **72.63** |
| DrugOOD-Size (AUC↑) | 66.04 | 65.89 | **67.23** |

### 消融实验

| 配置 | REDDIT-B | REDDIT-M | 说明 |
|------|----------|----------|------|
| 无增强 | 91.8 | 56.9 | 基线GIN |
| $(\mathcal{E}_\mathcal{L}, \mathcal{S}_{id})$ | 92.56 | 58.32 | 仅核特征无草图 |
| $(\mathcal{E}_\mathcal{L}, \mathcal{S}_{AG}^{(8)})$ | **94.00** | **60.33** | 完整SRF |

草图投影带来的提升约占总增益的60%+，远超仅核特征。

### 关键发现

1. 在Tree-NeighborsMatch过压缩基准上，SRF在半径4以内保持完美准确率，半径8时仍>40%
2. Dirichlet能量实验显示SRF有效维持深层节点差异性，增加投影数 $k$ 进一步保持能量
3. SRF与PEARL位置编码互补，组合后在Peptides-struct上接近Graph Transformer性能(MAE: 0.243 vs 0.245)
4. 运行速度约为PEARL的3倍，内存约低两个数量级

## 亮点与洞察

1. **三问题一解**：单一机制同时解决过压缩、过平滑、表达力三大挑战，理论证明与实验验证兼具
2. **特征为中心**：利用节点特征而非图拓扑构建增强信号，与拓扑编码正交互补
3. **即插即用**：与GNN架构无关，可叠加到GIN、GINE、GAT等任何消息传递GNN
4. **低开销**：一次预计算，比学习型编码（PEARL）大幅节省训练时间和内存

## 局限与展望

- 在无节点特征或特征退化的图上，SRF退化为纯随机特征
- 当 $D \ll N$ 时核近似有偏差，特征维度与节点数的比值影响效果
- 目前仅考虑节点级草图，未探索边特征或子图层面的草图增强
- 期望等变而非严格等变，理论上存在单次随机化的方差

## 相关工作与启发

- **位置编码**: SignNet, PEARL, SPE
- **GNN改进**: Graph Transformer (GPS, SAN), 图重连方法
- **随机方法**: Random Node Features, Johnson-Lindenstrauss变换
- **核方法**: Random Fourier Features, 核近似

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 将JL变换创造性用于GNN增强而非降维，视角独特
- 实验充分度: ⭐⭐⭐⭐ — 合成+真实任务，消融+复杂度+架构泛化全面
- 写作质量: ⭐⭐⭐⭐⭐ — 理论推导严谨，性质与GNN限制的对应关系清晰
- 价值: ⭐⭐⭐⭐⭐ — 简单高效的即插即用方案，理论与实践价值兼具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] On Measuring Long-Range Interactions in Graph Neural Networks](../../ICML2025/graph_learning/on_measuring_long-range_interactions_in_graph_neural_networks.md)
- [\[NeurIPS 2025\] Over-squashing in Spatiotemporal Graph Neural Networks](over-squashing_in_spatiotemporal_graph_neural_networks.md)
- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [\[NeurIPS 2025\] Sound Logical Explanations for Mean Aggregation Graph Neural Networks](sound_logical_explanations_for_mean_aggregation_graph_neural_networks.md)
- [\[NeurIPS 2025\] Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization](logical_expressiveness_of_graph_neural_networks_with_hierarchical_node_individua.md)

</div>

<!-- RELATED:END -->
