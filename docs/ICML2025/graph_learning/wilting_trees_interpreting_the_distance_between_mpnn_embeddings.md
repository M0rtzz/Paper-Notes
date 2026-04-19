---
title: >-
  [论文解读] WILTing Trees: Interpreting the Distance Between MPNN Embeddings
description: >-
  [ICML 2025][图学习][MPNN] 本文发现MPNN学到的嵌入距离与任务相关的functional distance对齐（而非结构距离），并提出基于加权Weisfeiler-Leman标记树（WILT）的最优传输距离来蒸馏和解释MPNN距离，边权揭示了少量关键子图主导了嵌入空间的度量结构。
tags:
  - ICML 2025
  - 图学习
  - MPNN
  - 图距离
  - 可解释性
  - 最优传输
  - Weisfeiler-Leman
  - 图核
---

# WILTing Trees: Interpreting the Distance Between MPNN Embeddings

**会议**: ICML 2025  
**arXiv**: [2505.24642](https://arxiv.org/abs/2505.24642)  
**代码**: 无  
**领域**: 图学习/GNN可解释性  
**关键词**: MPNN, 图距离, 可解释性, 最优传输, Weisfeiler-Leman, 图核  

## 一句话总结

本文发现MPNN学到的嵌入距离与任务相关的functional distance对齐（而非结构距离），并提出基于加权Weisfeiler-Leman标记树（WILT）的最优传输距离来蒸馏和解释MPNN距离，边权揭示了少量关键子图主导了嵌入空间的度量结构。

## 研究背景与动机

MPNN在图预测任务上表现优异，但其嵌入空间的度量结构（即图之间的距离）尚未被充分理解。先前研究（Chuang & Jegelka 2022; Böker et al. 2024）试图通过结构距离（structural distance）来分析MPNN的泛化性能，但这些结构距离忽略了任务信息，并需要强假设（如Lipschitz常数约束或类间间隔假设）。

本文提出两个核心问题：
1. MPNN学到的距离 $d_{\text{MPNN}}$ 是否与任务相关的功能距离 $d_{\text{func}}$ 对齐？这种对齐是否是高性能的关键？
2. MPNN如何学到这样的度量结构？能否用可解释的方式蒸馏出来？

作者发现MPNN距离与结构距离的对齐并不一致地随训练改善，也不与性能相关；相反，与功能距离的对齐才是关键。这一发现促使作者设计了WILT来蒸馏和解释MPNN的距离函数。

## 方法详解

### 整体框架

整体流程分两阶段：
1. **分析阶段**：定义功能伪度量 $d_{\text{func}}$ 和对齐度量 $\text{ALI}_k$，验证训练后的MPNN距离确实与 $d_{\text{func}}$ 对齐，且对齐程度与预测性能高度相关。
2. **蒸馏阶段**：构建Weisfeiler-Leman标记树（WILT），在其上定义可训练的最优传输距离 $d_{\text{WILT}}$，通过最小化与 $d_{\text{MPNN}}$ 的MSE来学习边权。学到的边权直接揭示哪些WL颜色（子图模式）对MPNN嵌入距离影响最大。

### 关键设计1：功能伪度量与对齐度量

作者定义了功能伪度量来量化两个图在目标任务上的距离：

$$d_{\text{func}}(G,H) = \begin{cases} \mathbb{1}_{y_G \neq y_H} & \text{(分类)} \\ \frac{|y_G - y_H|}{\sup y - \inf y} & \text{(回归)} \end{cases}$$

为衡量MPNN距离与功能距离的对齐程度，定义了 $\text{ALI}_k$ 指标：对每个图 $G$，计算其在MPNN空间中 $k$ 近邻的平均功能距离 $A_k(G)$ 与非近邻的平均功能距离 $B_k(G)$，若 $B_k(G) > A_k(G)$（即近的图功能更相近），则说明MPNN距离与功能距离对齐：

$$\text{ALI}_k(d_{\text{MPNN}}, d_{\text{func}}) = \frac{1}{|\mathcal{D}|} \sum_{G \in \mathcal{D}} [-A_k(G) + B_k(G)]$$

实验表明该指标在训练后显著提升，且与预测性能呈强正相关（Spearman相关系数在Mutagenicity上达0.66-0.70）。

### 关键设计2：WILT距离与线性时间计算

WILT是一棵由WL算法产生的颜色层次构建的加权有根树。树的节点是WL测试中出现的所有颜色，边连接相邻迭代中同一节点的颜色。WILT距离定义为图节点的WL颜色之间的最优传输距离，其中ground metric是树上的路径长度：

$$d_{\text{WILT}}(G,H;w) = \min_{P \in \Gamma} \sum_{v_i \in V_G} \sum_{u_j \in V_H} P_{i,j} \cdot d_{\text{path}}(c_{v_i}^{(L)}, c_{u_j}^{(L)})$$

关键突破在于等价表示——由于ground metric是树上路径长度，此最优传输可以等价转化为加权曼哈顿距离：

$$d_{\text{WILT}}(G,H;w) = \sum_{c \in V(T) \setminus \{r\}} w(e_{\{c,p(c)\}}) \cdot |\nu_c^G - \nu_c^H|$$

其中 $\nu_c^G$ 是颜色 $c$ 在图 $G$ 中出现的次数。这使得计算复杂度从 $O(n^3)$ 降至 $O(|V_G|+|V_H|)$，即线性时间。

### 关键设计3：两种归一化与图核的泛化关系

为处理不同大小的图，提出两种归一化：
- **大小归一化** $\dot{d}_{\text{WILT}}$：用 $\nu_c^G / |V_G|$ 替代计数，当边权恒为 $\frac{1}{2(L+1)}$ 时退化为Wasserstein WL图核距离
- **虚节点归一化** $\bar{d}_{\text{WILT}}$：添加虚拟孤立节点使所有图节点数相同，当边权恒为 $\frac{1}{2}$ 时退化为WL最优分配核距离

理论证明了表达力关系：$\dot{d}_{\text{WILT}} < \bar{d}_{\text{WILT}} \cong d_{\text{WL}}$，且 $\dot{d}_{\text{WILT}}$ 更适合近似mean pooling的MPNN，$\bar{d}_{\text{WILT}}$ 更适合sum pooling的MPNN。

## 实验关键数据

### 表1：WILT距离近似MPNN距离的RMSE（×10⁻²）

| 数据集/池化 | $d_{\text{WWL}}$ | $d_{\text{WLOA}}$ | $\dot{d}_{\text{WILT}}$ | $\bar{d}_{\text{WILT}}$ |
|:---:|:---:|:---:|:---:|:---:|
| Mutagenicity/mean | 9.25±0.87 | 18.74±3.36 | **1.74±0.52** | 3.34±1.01 |
| Mutagenicity/sum | 12.25±0.54 | 5.98±1.60 | 1.22±0.31 | **0.82±0.17** |
| ENZYMES/mean | 12.18±0.23 | 16.79±2.33 | **2.71±0.38** | 4.64±0.67 |
| ENZYMES/sum | 11.28±0.65 | 6.83±0.41 | 9.15±0.47 | **1.43±0.10** |
| Lipophilicity/mean | 10.92±0.42 | 13.97±0.97 | **3.11±0.54** | 6.35±1.22 |
| Lipophilicity/sum | 10.83±0.73 | 10.00±1.34 | **2.50±0.67** | 2.64±0.74 |

WILT距离比固定权重的基线（WWL和WLOA）低一个数量级，size norm更适合mean pooling，dummy norm更适合sum pooling。

### 表2：ALI_k与预测性能的Spearman相关系数

| k | Mutagenicity (train/test) | ENZYMES (train/test) | Lipophilicity (train/test) |
|:---:|:---:|:---:|:---:|
| 1 | 0.66/0.70 | 0.89/0.49 | -0.65/-0.57 |
| 5 | 0.65/0.69 | 0.87/0.50 | -0.63/-0.56 |
| 10 | 0.63/0.68 | 0.87/0.47 | -0.62/-0.56 |
| 20 | 0.61/0.67 | 0.85/0.46 | -0.60/-0.55 |

分类任务（Mutagenicity, ENZYMES）上ALI越高准确率越高，回归任务（Lipophilicity）上ALI越高RMSE越低，关系一致。

## 关键发现

1. **MPNN距离与功能距离对齐，而非结构距离**：训练MPNN后，其嵌入距离自然与任务相关的功能距离对齐，且对齐程度与性能高度正相关。先前工作关注的结构距离对齐既不随训练改善，也与性能无关。
2. **仅少量关键子图决定MPNN距离**：WILT边权分布严重偏斜向零。即使用L1正则化将95%的边权置零，$d_{\text{WILT}}$ 仍优于固定权重基线。说明MPNN依赖极少数WL颜色来定义嵌入距离。
3. **识别出的关键子图与领域知识吻合**：在Mutagenicity数据集上，最大权重对应的WL颜色代表epoxide和aliphatic halide等已知致突变官能团（Kazius et al., 2005），验证了MPNN学到了有意义的化学知识。

## 亮点与洞察

- **优雅的理论框架**：将MPNN距离分析从二值表达力扩展到度量空间分析，通过WILT上的最优传输建立了可解释的桥梁。等价表示使得本需立方复杂度的Wasserstein距离降为线性时间计算。
- **统一两种经典图核**：WILT距离作为参数化框架统一了Wasserstein WL核和WL最优分配核，二者分别是特殊权重配置下的特例，揭示了不同图核之间的内在联系。
- **全局解释能力**：与大多数实例级GNN解释方法不同，WILT提供全局的度量空间解释，识别出在所有图对距离中起决定性作用的子图模式。适用于分类和回归任务。

## 局限性

1. 仅实验了GCN和GIN两种架构的固定超参数配置，未验证对其他GNN架构或不同超参数的泛化性
2. 仅分析了MPNN最终层的嵌入距离，中间层的距离演化未被探索
3. WILT受限于1-WL颜色层次，对更高阶GNN的适用性需要进一步研究
4. 实验数据集规模较小（Mutagenicity ~4337, ENZYMES 600），在大规模图上的可扩展性需验证
5. 当前方法仅适用于图级任务，节点分类任务上的适用性尚待开发

## 相关工作与启发

- **GNN度量分析**：Chuang & Jegelka (2022) 的Tree Mover's Distance和Böker et al. (2024) 的细粒度表达力工作关注结构距离，本文挑战了这一范式，转向功能距离。
- **GNN可解释性**：与GNNExplainer等实例级方法不同，本文走全局蒸馏路线（类似GraphChef），但聚焦于度量结构而非决策边界。
- **图核方法**：WILT统一了WWL核和WLOA核，提示可训练的图核设计可能兼具可解释性和高性能。
- **启发**：这种"先训练黑箱模型→再蒸馏到可解释结构"的范式值得推广到其他领域。功能距离 vs 结构距离的对比思路对理解表示学习本质有启发意义。

## 评分

| 维度 | 分数 | 说明 |
|:---:|:---:|:---|
| 新颖性 | 8/10 | 从度量空间角度理解MPNN，WILT概念新颖且统一经典图核 |
| 技术深度 | 9/10 | 理论分析扎实，表达力定理严谨，线性时间算法优美 |
| 实验充分性 | 7/10 | 数据集和架构偏少，但定量定性结果互相支撑 |
| 实用价值 | 7/10 | 为GNN解释提供了全局视角，化学领域验证有说服力 |
| 写作质量 | 8/10 | 结构清晰，问题驱动，图示直观 |
| 总评 | 8/10 | 从理论和实证两个角度出色地揭示了MPNN嵌入空间的度量结构 |

<!-- RELATED:START -->

## 相关论文

- [Mixed-Curvature Decision Trees and Random Forests](mixed-curvature_decision_trees_and_random_forests.md)
- [Towards Graph Foundation Models: Learning Generalities Across Graphs via Task-Trees](towards_graph_foundation_models_learning_generalities_across_graphs_via_task-tre.md)
- [Predicate-Conditional Conformalized Answer Sets for Knowledge Graph Embeddings](../../ACL2025/graph_learning/predicate-conditional_conformalized_answer_sets_for_knowledge_graph_embeddings.md)
- [Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs](../../ICLR2026/graph_learning/structurally_human_semantically_biased_detecting_llm-generated_references_with_e.md)
- [CoDy: Counterfactual Explainers for Dynamic Graphs](cody_counterfactual_explainers_for_dynamic_graphs.md)

<!-- RELATED:END -->
