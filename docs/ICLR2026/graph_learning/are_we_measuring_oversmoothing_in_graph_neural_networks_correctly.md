---
title: >-
  [论文解读] Are We Measuring Oversmoothing in Graph Neural Networks Correctly?
description: >-
  [ICLR 2026][图学习][oversmoothing] 指出广泛使用的Dirichlet energy指标无法在实际场景中正确捕获GNN过平滑现象，提出以特征表征的数值秩/有效秩（effective rank）作为替代度量…
tags:
  - "ICLR 2026"
  - "图学习"
  - "oversmoothing"
  - "图神经网络"
  - "Dirichlet energy"
  - "numerical rank"
  - "effective rank"
  - "rank collapse"
---

# Are We Measuring Oversmoothing in Graph Neural Networks Correctly?

**会议**: ICLR 2026  
**arXiv**: [2502.04591](https://arxiv.org/abs/2502.04591)  
**领域**: 图神经网络 / 理论分析  
**关键词**: oversmoothing, graph neural networks, Dirichlet energy, numerical rank, effective rank, rank collapse, GNN depth  

## 一句话总结

指出广泛使用的Dirichlet energy指标无法在实际场景中正确捕获GNN过平滑现象，提出以特征表征的数值秩/有效秩（effective rank）作为替代度量，实验表明Erank与准确率的平均相关性达0.91（vs Dirichlet energy的0.72），在OGB-Arxiv上Dirichlet energy甚至呈现错误的相关方向，并从理论上证明对广泛的GNN架构族其数值秩收敛到1（秩坍塌），重新定义过平滑为秩坍塌而非特征向量对齐。

## 研究背景与动机

**过平滑是GNN的核心瓶颈**：随着层数加深，GNN节点表征趋于一致，丧失区分性——这被认为是GNN无法像其他深度网络一样从深度中获益的主要原因。

**Dirichlet energy的主导地位**：当前文献几乎统一使用Dirichlet energy（衡量相邻节点表征差异的平方和）作为过平滑的度量标准，基于此设计了大量缓解方法。

**度量与性能的脱节**：实践中发现Dirichlet energy的变化方向有时与模型性能变化不一致——Dirichlet energy降低不一定意味着性能下降，反之亦然。

**理论理解的不足**：现有过平滑理论主要分析Dirichlet energy在特定图拓扑下的收敛行为，但未质疑该度量本身的合理性。

**异构图上的失效**：在异构图（heterophilic graph）中，相邻节点标签不同，Dirichlet energy高反而可能意味着模型学到了有用的区分性表征和失败无法区分。

**需要更好的度量**：GNN社区需要一个与下游任务性能更一致的过平滑度量，以正确指导架构设计和正则化策略。

## 方法详解

### 整体框架

本文不提新模型，而是从线性代数视角重新定义过平滑：把它看作节点特征矩阵的秩坍塌（rank collapse）——层数加深时特征矩阵的有效秩趋近 1，所有节点表征被压进同一个低维子空间。围绕这个定义，论文用一个新度量（有效秩）替换主流的 Dirichlet energy，先在多架构多数据集上证明它与性能更吻合，再从理论上证明秩坍塌对广泛 GNN 族是必然结果。

### 关键设计

**1. 有效秩（Effective Rank）作为过平滑度量：换掉只看相邻差异的 Dirichlet energy。**

Dirichlet energy 衡量的是相邻节点表征差的平方和，它低只能说明邻居变像了，却看不出整个表征空间是否真的塌缩。本文改用特征矩阵奇异值谱的归一化熵来定义有效秩：先对特征矩阵 $X$ 做 SVD 拿到奇异值 $\sigma_i$，归一化为 $\hat{\sigma}_i = \sigma_i / \sum_j \sigma_j$，再取谱熵的指数 $\text{Erank}(X) = \exp\left(-\sum_i \hat{\sigma}_i \log \hat{\sigma}_i\right)$。这个量直观地数出特征矩阵里"有意义的维度个数"：当所有节点表征线性相关、坍塌到一条方向时 Erank 趋于 1（彻底过平滑），谱越平铺、表征越多样则 Erank 越高。因为它直接刻画表征空间的维度塌缩程度，而非局部相邻差异，所以与下游分类性能的联系更本质、更直接。

**2. Dirichlet energy 失效的实证诊断：用相关性检验度量的资格。**

一个声称衡量过平滑的度量，理应满足"过平滑越严重、性能越差"，即与节点分类准确率高度负相关（或随深度变化的趋势一致）。本文据此把 Erank 和 Dirichlet energy 摆在同一标准下比较：在 GCN、GAT、GraphSAGE、GIN 等多种架构、Cora 到 OGB-Products 等多个数据集上，扫不同深度并计算两种度量与准确率的 Pearson/Spearman 相关系数。结果是 Erank 平均相关性 0.91、Dirichlet energy 仅 0.72；尤其在大规模的 OGB-Arxiv 上，Dirichlet energy 竟与准确率呈 $-0.31$ 的错误方向相关——它增大时准确率反而上升，说明在真实大图上它已经不能再当过平滑的代理指标，而 Erank 始终保持 $>0.88$ 的正相关。

**3. 秩坍塌的理论收敛证明：把"必然过平滑"钉成定理。**

为给秩坍塌这个新定义提供地基，本文证明对一大类 GNN 架构族（涵盖 GCN、GraphSAGE、GIN 等），当层数趋于无穷时特征矩阵的数值秩收敛到 1。证明的核心是对归一化邻接矩阵反复作用做谱分析：矩阵幂会让奇异值谱迅速分化，非主奇异值被指数压低，最终只剩主奇异值存活，于是有效秩塌到 1。这把"深层 GNN 一定过平滑"从经验观察上升为可证明的结论，也解释了为何残差等技巧只能延缓、无法根本阻止坍塌。

## 实验关键数据

### 主实验：度量与准确率的相关性

| 数据集 | Dirichlet Energy相关性 | Erank相关性 | 差异 |
|--------|----------------------|-------------|------|
| Cora | 0.82 | **0.94** | +0.12 |
| Citeseer | 0.76 | **0.91** | +0.15 |
| PubMed | 0.79 | **0.93** | +0.14 |
| OGB-Arxiv | **-0.31** (错误方向!) | **0.88** | +1.19 |
| OGB-Products | 0.45 | **0.89** | +0.44 |
| 平均 | 0.72 | **0.91** | +0.19 |

### 消融实验：不同GNN架构

| 架构 | 层数范围 | Erank@2层 | Erank@64层 | 准确率下降 |
|-----|---------|-----------|-----------|-----------|
| GCN | 2-128 | 45.2 | 1.3 | -42.1% |
| GAT | 2-128 | 52.8 | 2.1 | -38.6% |
| GraphSAGE | 2-128 | 61.3 | 3.7 | -31.2% |
| GIN | 2-128 | 48.9 | 1.8 | -39.8% |
| ResGCN | 2-128 | 47.6 | 8.4 | -22.3% |

### 关键发现

1. **Dirichlet Energy在大规模图上失效**：在OGB-Arxiv上，Dirichlet energy与准确率呈负相关（-0.31），即Dirichlet energy增加时准确率反而提高。
2. **Erank一致性强**：在所有数据集和架构上，Erank均与准确率保持高正相关（>0.88）。
3. **秩坍塌是普遍现象**：所有测试的GNN架构在足够深时（64层以上），Erank均接近1。
4. **残差连接延缓但不阻止坍塌**：ResGCN的Erank下降更慢，但最终仍趋于低秩。
5. **度量选择影响研究结论**：使用Dirichlet energy可能导致研究者得出错误的架构改进方向。

## 亮点与洞察

1. **挑战基础假设**：质疑了GNN社区广泛接受的Dirichlet energy度量，属于"皇帝的新衣"式的重要贡献。
2. **替代方案优越**：Erank在理论和实证上均显著优于Dirichlet energy，且实现简单。
3. **概念重定义**：将过平滑从"相邻节点特征趋同"（特征向量对齐）重新定义为"特征空间秩坍塌"，更加本质。
4. **实际指导价值**：正确的度量将引导社区开发更有效的过平滑缓解策略。
5. **大规模数据集的关键性**：在小数据集上两种度量差异不大，但在大规模真实图上差异巨大。

## 局限与展望

1. **因果关系未完全建立**：高相关性不等于因果性，秩坍塌是否是性能下降的唯一原因尚待商榷。
2. **异构图的探索有限**：虽然提及，但对异构图上Erank行为的实验覆盖不够系统。
3. **缓解策略未提出**：论文聚焦于度量诊断，未基于新的秩视角提出具体的缓解算法。
4. **图级任务未涉及**：所有实验针对节点分类，图级分类任务中过平滑的表现未探讨。

## 相关工作与启发

- **过平滑理论**：Li et al. (2018) 首次形式定义过平滑；Cai & Wang (2020) 的Dirichlet energy框架
- **过平滑缓解**：DropEdge (Rong et al., 2020), PairNorm (Zhao & Akoglu, 2020), DeeperGCN (Li et al., 2020)
- **有效秩**：Roy & Vetterli (2007) 提出的effective rank定义
- **深层GNN**：GCNII (Chen et al., 2020), RevGNN (Li et al., 2021) 等深层架构设计

## 评分

- 新颖性: ⭐⭐⭐⭐ 挑战基础度量假设，提出秩坍塌视角
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多架构、理论+实证的全面验证
- 写作质量: ⭐⭐⭐⭐⭐ 论证逻辑清晰，实验设计精当
- 价值: ⭐⭐⭐⭐⭐ 对GNN社区的基础度量进行了必要的修正

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] On Measuring Long-Range Interactions in Graph Neural Networks](../../ICML2025/graph_learning/on_measuring_long-range_interactions_in_graph_neural_networks.md)
- [\[ICLR 2026\] Cooperative Sheaf Neural Networks](cooperative_sheaf_neural_networks.md)
- [\[ICLR 2026\] LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks](logicxgnn_grounded_logical_rules_for_explaining_graph_neural_networks.md)
- [\[CVPR 2026\] Adaptive Learned Image Compression with Graph Neural Networks](../../CVPR2026/graph_learning/adaptive_learned_image_compression_with_graph_neural_networks.md)
- [\[ICML 2026\] Quantile-Free Uncertainty Quantification in Graph Neural Networks](../../ICML2026/graph_learning/quantile-free_uncertainty_quantification_in_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
