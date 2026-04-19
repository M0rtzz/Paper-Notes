---
title: >-
  [论文解读] Geometric Imbalance in Semi-Supervised Node Classification
description: >-
  [NeurIPS 2025][图学习][geometric imbalance] 首次形式化定义了半监督节点分类中的"几何不平衡"概念——消息传递在类别不平衡图上导致少数类节点在黎曼流形嵌入空间中产生几何歧义，并提出 UNREAL 框架，通过双路径伪标签对齐、节点重排序和几何歧义样本丢弃三个模块系统性缓解该问题。
tags:
  - NeurIPS 2025
  - 图学习
  - geometric imbalance
  - 半监督学习
  - self-training
  - 伪标签
  - 图神经网络
  - Riemannian manifold
---

# Geometric Imbalance in Semi-Supervised Node Classification

**会议**: NeurIPS 2025  
**arXiv**: [2303.10371](https://arxiv.org/abs/2303.10371)  
**作者**: Liang Yan, Shengzhong Zhang, Bisheng Li, Menglin Yang, Chen Yang, Min Zhou, Weiyang Ding, Yutong Xie, Zengfeng Huang (Fudan University, HKUST(GZ), MBZUAI, Logs AI)
**代码**: [yanliang3612/UNREAL](https://github.com/yanliang3612/UNREAL)  
**领域**: graph_learning  
**关键词**: geometric imbalance, semi-supervised node classification, self-training, pseudo-label, GNN, Riemannian manifold

## 一句话总结

首次形式化定义了半监督节点分类中的"几何不平衡"概念——消息传递在类别不平衡图上导致少数类节点在黎曼流形嵌入空间中产生几何歧义，并提出 UNREAL 框架，通过双路径伪标签对齐、节点重排序和几何歧义样本丢弃三个模块系统性缓解该问题。

## 研究背景与动机

图数据中的类别不平衡严重影响 GNN 节点分类效果，尤其在半监督场景下，仅少量节点有标签，使信息传播更加困难。

**现有方法的局限**：
- **过采样方法（GraphSMOTE, GraphENS）**：需合成新节点/边，计算开销大且难以保证结构一致性
- **损失函数调整（ReNode, TAM）**：依赖标注节点推断拓扑偏差，缺乏通用量化框架
- **自训练方法（GraphSR, BIM）**：依赖启发式选择策略，缺乏理论保证，在严重不平衡下不稳定
- **拓扑不平衡 vs 几何不平衡**：已有工作关注的拓扑不平衡定义在原始图结构上，而本文发现的几何不平衡定义在黎曼流形嵌入空间中，是一个此前被忽视的全新问题

**核心洞察**：消息传递机制在类别不平衡的图上会放大少数类节点的嵌入歧义——这些节点在超球面上与多个类别中心近乎等距，导致伪标签分配极不可靠。MLP 不受此影响（无结构聚合），说明几何不平衡是 GNN 特有的问题。

## 方法详解

### 几何不平衡的理论基础

**形式化定义**：基于 von Mises-Fisher (vMF) 分布在单位超球面上进行分析。GNN 嵌入经 $\ell_2$ 归一化后映射到黎曼流形 $\mathbb{S}^{d-1}$，每个类别由一个 vMF 分布参数化（紧致度 $\kappa_i$ 和方向向量 $\tilde{\mu}_i$）。

**几何不平衡分数**（Definition 1）：对未标注节点 $u^j$，其几何不平衡分数定义为：
$$G(u^j) = \frac{\|\tilde{h}_{u^j}^{(l)} - \tilde{\mu}_{y^{u^j}}\|^2}{\sum_{\mathcal{C}_1 \neq \mathcal{C}_2}\|\tilde{\mu}_{\mathcal{C}_1} - \tilde{\mu}_{\mathcal{C}_2}\|^2}$$
即节点与其真实类别中心的距离占类间总分离度的比例。

**Theorem 1**：平均信息熵 $\hat{H} \propto D_{\text{intra}} / D_{\text{inter}}$，即预测不确定性随类内分散度增大、类间分离度减小而升高。

**Theorem 2**：几何不平衡分数 $\bar{G}_{\text{minor}}$ 随类别不平衡比率 $\rho$ 单调递增，理论上刻画了数据不平衡程度与嵌入空间几何歧义的直接关系。

### UNREAL 框架

框架由三个互补模块组成，可独立使用或组合。

#### 模块1：DualPath PseudoLabeler (DPAM)
- **双路径**：(1) 无监督聚类——对未标注节点嵌入做 K-Means 过聚类（$K > C$），通过聚类中心与类别中心距离分配伪标签；(2) 有监督分类——GNN 直接预测伪标签
- **对齐机制**：仅保留两条路径预测一致的节点：$\mathcal{U}_i^{\text{final}} = \tilde{\mathcal{U}}_i \cap \mathcal{U}_i$
- **作用**：聚类路径缓解分类器的多数类偏差，分类路径消除聚类的几何歧义噪声

#### 模块2：Node-Reordering (NR)
- 定义两种排序：**Confidence Ranking (CR)** 基于分类器 softmax 概率；**Geometric Ranking (GR)** 基于节点嵌入到类别中心的距离
- 通过 Rank-Biased Overlap (RBO) 度量两排序的一致性 $r_m$，自适应融合：$\mathcal{N}_m^{\text{New}} = \max\{r_m, 1-r_m\} \cdot \mathcal{S}_m + \min\{r_m, 1-r_m\} \cdot \mathcal{T}_m$
- **动态特性**：训练早期两排序分歧大，几何排序主导（更可靠）；训练后期一致性提升，分类器置信度权重增大

#### 模块3：Discarding Geometric Imbalanced Samples (DGIS)
- 定义轻量化的几何不平衡指数：$\text{GI}_u = (\beta_u - \delta_u) / \delta_u$，其中 $\delta_u$ 是到最近类别中心的距离，$\beta_u$ 是到次近中心的距离
- GI 值低表示节点在多个类别中心间几何歧义大，通过阈值过滤丢弃这些节点
- 在 DPAM 和 NR 之后执行，候选池已精简，计算高效

## 实验关键数据

### 实验设置
- **数据集**：8 个基准——Cora, CiteSeer, PubMed, Amazon-Computers（人工不平衡 $\rho$=10,20,50,100）；Computers-Random ($\rho \approx 17.7$), CS-Random ($\rho \approx 41.0$), Flickr ($\rho \approx 10.8$), Ogbn-arxiv ($\rho \approx 775.4$)（自然不平衡）
- **骨干网络**：GCN, GAT, GraphSAGE
- **基线**：GraphSMOTE, GraphENS, ReNode, TAM, GraphSR, BIM 等 10+ 种方法
- **指标**：Balanced Accuracy (bAcc.) 和 Macro-F1

### Table 2: 人工不平衡 ρ=10, GCN 骨干

| 方法 | Cora bAcc. | Cora F1 | CiteSeer bAcc. | CiteSeer F1 |
|---|---|---|---|---|
| Vanilla | 62.82±1.43 | 61.67±1.59 | 38.72±1.88 | 28.74±3.21 |
| BIM | 72.19±0.42 | 72.67±0.48 | 58.54±0.61 | 56.81±0.98 |
| GE(w TAM) | 71.69±0.36 | 72.14±0.51 | 58.01±0.68 | 56.32±1.03 |
| GSR | 70.85±0.44 | 71.37±0.63 | 59.28±0.72 | 55.96±0.95 |
| **UNREAL** | **78.33±1.04** | **76.44±1.06** | **65.63±1.38** | **64.94±1.38** |
| Δ 提升 | **+6.14** | **+3.77** | **+6.35** | **+8.13** |

### Table 3: 人工不平衡 ρ=100, SAGE 骨干

| 方法 | Cora bAcc. | Cora F1 | CiteSeer bAcc. | CiteSeer F1 | PubMed bAcc. | PubMed F1 |
|---|---|---|---|---|---|---|
| Vanilla | 52.65±0.24 | 43.79±0.47 | 36.63±0.09 | 24.12±0.09 | 62.29±0.25 | 47.02±0.38 |
| BIM | 67.75±2.13 | 64.68±1.95 | 53.83±1.62 | 53.29±1.80 | 74.38±2.08 | 73.24±1.85 |
| **UNREAL** | **73.47±2.31** | **68.30±2.11** | **59.77±2.98** | **58.92±3.07** | **77.11±0.59** | **74.03±0.81** |
| Δ 提升 | **+5.72** | **+3.62** | **+6.04** | **+5.63** | **+2.73** | **+0.79** |

在极端不平衡 (ρ=100) 下，UNREAL 仍保持显著优势，Cora bAcc. 超越最优基线 5.72 个百分点。

### 自然不平衡与大规模数据集
- **Computers-Random**（ρ≈17.7）：GCN 上 bAcc. 85.32%，超越 BIM +1.29；GAT 上 bAcc. 82.52%，超越 BIM **+5.51**
- **Ogbn-arxiv**（ρ≈775.4）：多个基线 OOM，UNREAL 仍可运行且达到 F1 51.36（SAGE），超越 BIM +1.80
- **Flickr**（ρ≈10.8）：GraphSMOTE/ReNode/GraphENS 全部 OOM，UNREAL 在 GCN 上 F1 30.60 超 BIM **+6.85**

### 消融实验 (Table 1)
以 Cora+GCN (ρ=10) 为例：
- 完整 UNREAL（CR+GR+DGIS，无 NR）：76.44 F1
- 去掉 DGIS：下降至 75.34
- 去掉 CR（仅 GR+NR+DGIS）：下降至 73.93
- 三个模块互补性强，NR+DGIS 组合在多数设置下取得最佳

## 亮点

- **理论贡献突出**：首次在黎曼流形上形式化定义几何不平衡，建立了几何不平衡与信息熵（Theorem 1）、类别不平衡比率（Theorem 2）的严格数学关系，将原本模糊的经验观察上升为可量化的理论框架
- **框架设计优雅**：三个模块（DPAM/NR/DGIS）各有明确的理论动机和实际功能，既可独立使用，又可灵活组合，且 DGIS 作为轻量近似避免了直接计算几何不平衡分数的高开销
- **实验全面且可扩展**：涵盖 8 个数据集、3 种 GNN 骨干、人工和自然不平衡设置，在极端不平衡（ρ=100, ρ≈775）下仍有效，且不像多个基线那样 OOM

## 局限性

- **仅限节点分类**：未扩展到链接预测、图分类等其他图任务（作者在结论中也指出这是未来工作）
- **K-Means 聚类假设**：在超球面嵌入上使用欧氏 K-Means 而非球面聚类（如 spherical K-Means），虽作者在附录中讨论了鲁棒性，但理论上不完全一致
- **超参数敏感性**：聚类数 $k'$ 和 DGIS 阈值 $\gamma$ 需要调优，虽然敏感性分析表明在合理范围内较稳定，但最优值因数据集而异
- **SAGE 骨干在 Computers-Random 上表现下降**：Δ 为 -3.17 bAcc.，说明框架在特定骨干-数据集组合下不总是最优
- **大规模数据集提升有限**：在 Ogbn-arxiv 上整体提升幅度较小（<1.5 bAcc.），可能因极端不平衡（ρ≈775）超出了当前方法的有效范围

## 相关工作

- **节点生成方法**：GraphSMOTE（插值少数类嵌入+预测新边）、ImGAGN（联合生成特征+拓扑）、GraphENS（ego-network 混合增强多样性）→ 计算开销大，难保结构一致性
- **拓扑感知调整**：ReNode（基于拓扑距离重加权）、TAM（利用局部拓扑和类别统计校准 logits）→ 依赖标注节点，无法量化嵌入空间中的偏差
- **自训练方法**：GraphSR（相似度过滤+强化学习选择伪标签）、BIM（影响力最大化平衡类别影响范围）、IceBerg（双平衡+解耦传播）→ 启发式选择，缺乏理论保证
- **UNREAL 的定位**：从嵌入空间几何视角出发，首次提供几何不平衡的理论量化，填补了图自训练中缺乏理论基础的空白

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 几何不平衡概念是全新的理论贡献，严格的黎曼流形分析将经验观察提升为可量化框架
- 实验充分度: ⭐⭐⭐⭐ — 8 数据集、3 骨干、多种不平衡设置，消融和敏感性分析完整；但部分设置提升有限
- 写作质量: ⭐⭐⭐⭐ — 理论推导严谨，结构清晰，符号体系虽多但定义明确
- 价值: ⭐⭐⭐⭐ — 为图不平衡学习提供了新的理论视角和实用工具，自训练框架可扩展性强

<!-- RELATED:START -->

## 相关论文

- [Uncertain Knowledge Graph Completion via Semi-Supervised Confidence Distribution Learning](uncertain_knowledge_graph_completion_via_semi-supervised_confidence_distribution.md)
- [PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](preference-driven_knowledge_distillation_for_few-shot_node_classification.md)
- [Posterior Label Smoothing for Node Classification](../../AAAI2026/graph_learning/posterior_label_smoothing_for_node_classification.md)
- [Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization](logical_expressiveness_of_graph_neural_networks_with_hierarchical_node_individua.md)

<!-- RELATED:END -->
