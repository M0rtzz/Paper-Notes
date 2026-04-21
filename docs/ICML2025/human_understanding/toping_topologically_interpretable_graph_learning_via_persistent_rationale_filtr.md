---
title: >-
  [论文解读] TopInG: Topologically Interpretable Graph Learning via Persistent Rationale Filtration
description: >-
  [ICML 2025][人体理解] TopInG 提出了一种基于持久同调的拓扑可解释图学习框架，通过学习"基本原理过滤"（rationale filtration）来识别稳定且持久的基本原理子图，引入"拓扑差异"（topological discrepancy）约束来强化基本原理子图与无关子图之间的拓扑区分，在处理多变形态的基本原理子图时显著优于现有方法。
tags:
  - ICML 2025
  - 人体理解
---

# TopInG: Topologically Interpretable Graph Learning via Persistent Rationale Filtration

**会议**: ICML 2025  
**arXiv**: [2510.05102](https://arxiv.org/abs/2510.05102)  
**领域**: 人体理解  

## 一句话总结

TopInG 提出了一种基于持久同调的拓扑可解释图学习框架，通过学习"基本原理过滤"（rationale filtration）来识别稳定且持久的基本原理子图，引入"拓扑差异"（topological discrepancy）约束来强化基本原理子图与无关子图之间的拓扑区分，在处理多变形态的基本原理子图时显著优于现有方法。

## 研究背景与动机

GNN 在科学领域取得了显著成功，但其可解释性不足限制了在关键决策中的采用。现有的内在可解释 GNN（如 GSAT、GMT、DIR）面临一个共同问题：

**隐含假设过强**：现有方法通常假设同类图的基本原理子图在形式上近乎不变。然而在现实中，**多变形态的基本原理子图**（variform rationale subgraphs）非常常见：

- 分子生物学中，具有相同生物活性的分子可能有不同的功能团（芳香环、磺酰胺基、杂环化合物）
- 社交网络中，用户影响力的结构原因各异（高度数、高介数中心性、桥接节点）

## 方法详解

### 核心思想

将图视为从核心基本原理子图 $G_X^*$ 出发"生长"出的结构：先生成关键子图，后添加辅助结构。通过持久同调追踪这一生成过程中的拓扑特征生命周期。

### 基本原理过滤学习

使用骨干 GNN 作为可学习的过滤函数 $f_\phi: G \to [0,1]^{|E|}$，为每条边分配重要性分数。这些分数诱导出一个**图过滤**——一个嵌套子图序列：

$$\mathcal{F}(G) = \{G_0, G_1, \ldots, G_{|E|}\}$$

按重要性降序逐步添加边（更重要的边先加入）。目标是学习到与 $G_X^*$ 和 $G_\epsilon^*$ 的划分一致的重要性排序。

### 拓扑差异（Topological Discrepancy）

定义拓扑差异为持久同调分布之间的 1-Wasserstein 距离：

$$d_{\text{topo}}(\mathbb{P}(\mathcal{T}_X), \mathbb{P}(\mathcal{T}_\epsilon)) \triangleq \inf_{\pi \in \Pi(\mathbb{P}(\mathcal{T}_X), \mathbb{P}(\mathcal{T}_\epsilon))} \mathbb{E}_{(P,Q) \sim \pi}[d_B(P,Q)]$$

其中 $d_B$ 为瓶颈距离。

### 总体损失函数

$$\mathcal{L}(\phi) = \mathbb{E}_G[\mathcal{L}_{\text{ce}}(\hat{y}_G, y_G)] - \alpha \mathcal{L}_{\text{topo}}(\mathbb{P}(\mathcal{T}_X), \mathbb{P}(\mathcal{T}_\epsilon))$$

通过最大化拓扑差异来增强基本原理子图与补充子图的拓扑区分。

### 可计算的下界

利用 Kantorovich 对偶和 Lipschitz 连续的可学习向量化函数（Rational Hat 结构元素）提供可计算的下界：

$$\varphi(p; \mathbf{c}, r) = \sum_{\mathbf{x} \in p} \frac{1}{1 + \|\mathbf{x} - \mathbf{c}\|_2} - \frac{1}{1 + ||r| - \|\mathbf{x} - \mathbf{c}\|_2|}$$

使用 2 头注意力机制从 $k=8$ 个 Lipschitz 表示函数中选择 top-2 最大值。

### 先验正则化

引入双混合高斯先验对边过滤值进行聚类：

$$\mathbb{P}_{prior} = w\mathcal{N}(\mu_1, r_1) + (1-w)\mathcal{N}(\mu_2, r_2)$$

固定 $w=0.5$、$\mu_1=0.25$、$\mu_2=0.75$，实现对 $G_X$ 和 $G_\epsilon$ 的无监督聚类。

### 理论保证

**定理 3.4**：假设对所有 $G$，$|E_X| < |E_\epsilon|$，且 $G_X^*$ 关于 $y_G$ 是最小的，则损失 $\mathcal{L}(\phi)$ 被 $f_\phi^*(e) = \mathbb{1}\{e \in G_X^*\}$ **唯一最优化**。

关键优势：该保证**不依赖**基本原理子图的稳定性或不变性假设，因此不受多变形态子图的影响。

## 实验

### 解释性能（AUC）

| 方法 | BA-2Motifs | BA-HouseGrid | SPMotif0.9 | BA-HouseAndGrid | BA-HouseOrGrid | Benzene |
|---|---|---|---|---|---|---|
| GIN+GSAT | 98.85 | 98.55 | 65.25 | 92.92 | 83.56 | 91.57 |
| GIN+GMT-Lin | 97.72 | 85.68 | 69.08 | 76.12 | 74.36 | 83.90 |
| **GIN+TopInG** | **99.57** | **99.24** | **80.82** | **95.35** | **88.56** | **98.22** |
| CINpp+GSAT | 91.12 | 91.04 | 80.24 | 95.17 | 69.30 | 95.40 |
| **CINpp+TopInG** | **100.00** | **99.87** | **92.82** | **100.00** | **100.00** | **98.72** |

**关键发现**：
- 在多变形态子图基准（BA-HouseAndGrid、BA-HouseOrGrid）上提升尤为显著
- CINpp+TopInG 在多个数据集上达到 100% AUC
- SPMotif0.9（高虚假相关性）上的改进表明方法有效缓解了虚假关联

### 预测性能

TopInG 在保持高解释性的同时，预测准确率也优于或匹配基线方法，克服了传统的性能-可解释性权衡。

## 亮点

- **首次将持久同调引入内在可解释 GNN**：提供了全新的拓扑视角来解决基本原理子图识别问题
- **处理多变形态子图的独特能力**：在合成数据集上展示了对现有方法的显著优势
- **理论保证不依赖不变性假设**：基本原理可以在不同实例间自由变化
- **自适应拓扑约束**：先验正则化对超参数不敏感，无需调优
- **与现有 GNN 骨干兼容**：可搭配 GIN、CINpp 等多种架构

## 局限性

- 持久同调计算增加了训练时间
- 仅考虑了 0 阶和 1 阶持久同调（连通分量和环），未利用更高阶拓扑特征
- 节点过滤到边过滤的扩展（star filtration）信息量有限
- 在 Mutag 数据集上表现略低于 MAGE
- 理论保证依赖 $|E_X| < |E_\epsilon|$ 的假设，可能不总是成立

## 评分

⭐⭐⭐⭐⭐ (5/5)

这是一篇理论与实践结合极为出色的工作。将拓扑数据分析引入可解释 GNN 是一个巧妙且原创的思路，理论保证严格，实验在多个维度上都显示出显著优势，特别是在处理多变形态子图这一关键挑战上。

<!-- RELATED:START -->

## 相关论文

- [Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups](sum-of-parts_self-attributing_neural_networks_with_end-to-end_learning_of_featur.md)
- [Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)
- [Scaling Large Motion Models with Million-Level Human Motions](scaling_large_motion_models_with_million-level_human_motions.md)
- [FedRAG: A Framework for Fine-Tuning Retrieval-Augmented Generation Systems](fedrag_a_framework_for_fine-tuning_retrieval-augmented_generation_systems.md)
- [LLaVA-ReID: Selective Multi-Image Questioner for Interactive Person Re-Identification](llava-reid_selective_multi-image_questioner_for_interactive_person_re-identifica.md)

<!-- RELATED:END -->
