---
title: >-
  [论文解读] Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption
description: >-
  [AAAI 2026][图学习][无监督节点表示学习] 提出 FUEL 方法，通过以节点特征为中心的聚类方案自适应学习图卷积的使用程度，无需同配性假设即可在同配和非同配图上均获得高质量的无监督节点表示。
tags:
  - AAAI 2026
  - 图学习
  - 无监督节点表示学习
  - 图卷积
  - 非同配图
  - 聚类
  - 特征中心
---

# Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption

**会议**: AAAI 2026  
**arXiv**: [2512.15112](https://arxiv.org/abs/2512.15112)  
**作者**: Sunwoo Kim, Soo Yong Lee, Kyungho Kim, Hyunjin Hwang, Jaemin Yoo, Kijung Shin (KAIST)  
**代码**: [GitHub](https://github.com/kswoo97/unsupervised-non-homophilic)  
**领域**: graph_learning  
**关键词**: 无监督节点表示学习, 图卷积, 非同配图, 聚类, 特征中心  

## 一句话总结

提出 FUEL 方法，通过以节点特征为中心的聚类方案自适应学习图卷积的使用程度，无需同配性假设即可在同配和非同配图上均获得高质量的无监督节点表示。

## 研究背景与动机

### 问题背景
无监督节点表示学习旨在不依赖节点标签的情况下获取有意义的节点嵌入。图卷积（graph convolution）是其核心操作——通过聚合邻居信息来编码节点特征和图拓扑。在**同配图**（相似节点倾向于相连）中，图卷积效果出色；但在**非同配图**（不相似节点频繁相连）中，过度使用图卷积会使本应不同的节点获得过于相似的嵌入，反而降低表示质量。

### 已有工作的不足
- **有监督方法已有解决方案**：GPR-GNN、GADC 等通过下游任务损失（如交叉熵）学习图卷积的使用程度，能自适应调整每阶邻接矩阵的权重
- **无监督方法缺乏类似机制**：由于没有标签监督信号，无法直接优化图卷积权重。现有的非同配无监督方法（如 PolyGCL、HeterGCL）多采用多种图滤波器的对比学习，但未从根本上解决图卷积程度的自适应选择问题
- **内存效率问题**：许多现有方法（GREET、PolyGCL、HeterGCL 等）在较大规模图上出现 OOM（GPU 内存溢出），可扩展性受限

### 核心动机
能否在无标签的情况下，找到一个有效的代理指标来衡量"类别可分性"，从而指导图卷积使用程度的自适应学习？FUEL 的关键洞察是：**节点特征本身蕴含类别信息**——同类节点的特征天然相似，因此特征空间中的聚类可以作为真实类别的代理（latent class）。通过最大化这些代理类别的可分性，就能间接优化真正的类别可分性。

## 方法详解

### 核心思想：隐类别可分性（Latent-Class Separability）

FUEL 的理论基础建立在一个关键发现之上：**隐类别可分性与真实类别可分性高度正相关**。

具体地，作者通过三步建立了这个联系：
1. 在 Cora、Citeseer（同配）和 Actor、Cornell（非同配）四个数据集上，随机采样 100 组不同的图卷积权重 $(\alpha_0, \alpha_1, \alpha_2)$，生成不同程度的图卷积嵌入
2. 分别测量每组嵌入的**真实类别可分性**（线性分类器测试精度）和**隐类别可分性**（Calinski-Harabasz 指数）
3. 结果显示两者的 Spearman 秩相关系数均超过 0.82，线性回归斜率为正

理论上，在高斯分布假设下（两类等大小，特征 $x_i \sim \mathcal{N}(\mu, \sigma^2)$ 或 $\mathcal{N}(-\mu, \sigma^2)$），作者证明了 **Theorem 1**：在一定条件下，两种图卷积程度之间的类别可分性排序与隐类别可分性排序完全一致。这为使用聚类质量作为优化目标提供了严格的理论保证。

### Step 1：基于聚类的自适应图卷积学习

**自适应图卷积模型**：采用加权组合形式

$$\mathbf{X}^* = \alpha_0 \mathbf{X} + \alpha_1 \tilde{\mathbf{A}}\mathbf{X} + \alpha_2 \widetilde{\mathbf{A}^2}\mathbf{X}$$

其中 $\alpha_0 + \alpha_1 + \alpha_2 = 1$，$\alpha_k = \text{softmax}(c_k)$，$c_0, c_1, c_2$ 为可学习参数。$\mathbf{X}$ 为原始特征（0阶），$\tilde{\mathbf{A}}\mathbf{X}$ 为1-hop聚合（1阶），$\widetilde{\mathbf{A}^2}\mathbf{X}$ 为2-hop聚合（2阶）。

**聚类训练损失**由三个分量组成：

- $\mathcal{L}_1$（节点-聚类分配熵）：鼓励每个节点被高置信度分配到单个聚类，使分配分布尖锐
- $\mathcal{L}_2$（聚类均衡性）：防止所有节点坍缩到同一个聚类，保证聚类大小相对均匀
- $\mathcal{L}_3$（距离损失）：直接优化 inter-cluster 距离 / intra-cluster 距离的比值，增强聚类可分性

最终损失 $\mathcal{L}_{clus} = \mathcal{L}_1 + \mathcal{L}_2 + \lambda \mathcal{L}_3$。优化完成后，用学到的最优权重 $\alpha_0^*, \alpha_1^*, \alpha_2^*$ 生成中间嵌入 $\mathbf{H}$。

### Step 2：嵌入细化提升可分性

中间嵌入 $\mathbf{H}$ 来自简单的线性组合模型，表达能力有限。FUEL 通过前馈神经网络 $f_\theta$ 加 skip connection 进行细化：

$$\mathbf{Z} = f_\theta(\mathbf{H}) + \mathbf{H}$$

训练时固定 $\mathbf{H}$，仅优化 $\theta$。细化损失基于 $\mathbf{H}$ 空间中的最近邻关系：拉近 $N$ 个最近邻对（大概率同类）的嵌入距离，推远非最近邻对的距离。损失函数采用指数形式：

$$\mathcal{L}_{dist} = \exp\left(\frac{\sum_{v_i,v_j \in \mathcal{V}'_+} d(\mathbf{z}_i, \mathbf{z}_j)}{\tau|\mathcal{V}'_+|} - \frac{\sum_{v_k,v_\ell \in \mathcal{V}'_-} d(\mathbf{z}_k, \mathbf{z}_\ell)}{\tau|\mathcal{V}'_-|}\right)$$

指数函数的优势在于其对数化后等价于几何平均，对异常值更鲁棒。

## 实验关键数据

### 表1：节点分类准确率（×100，14个数据集）

| 方法 | 类型 | Texas (0.057) | Cornell (0.111) | Actor (0.220) | Cora (0.825) | Photo (0.849) | 平均排名 |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Naive X | 基线 | 75.1 | 71.1 | 35.4 | 58.5 | 88.5 | 14.0 |
| DGI | CL | 65.9 | 48.9 | 29.8 | 82.4 | 92.8 | 11.7 |
| BGRL | CL | 64.3 | 49.2 | 30.9 | 82.1 | 93.2 | 8.4 |
| GREET | 非同配 | 81.1 | 76.8 | 37.6 | 83.3 | 92.8 | 7.8 |
| PolyGCL | 非同配 | 80.5 | 74.8 | 35.0 | 82.6 | 91.3 | 9.1 |
| HeterGCL | 非同配 | 78.9 | 71.6 | 36.5 | 82.5 | 93.0 | 9.2 |
| **FUEL** | **本文** | **84.6** | **78.1** | **38.2** | **83.8** | **94.2** | **2.4** |

FUEL 在14个数据集中的10个上取得最佳性能，平均排名 2.4 远超第二名 GREET 的 7.8。在最低同配（Texas, h=0.057）和最高同配（Photo, h=0.849）图上均为第一。

### 表2：节点聚类 NMI（×100，14个数据集）

| 方法 | 类型 | Wisconsin | Cornell | Actor | Cora | Photo | 平均排名 |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| HGRL | 非同配 | 36.2 | 40.1 | 5.9 | 51.3 | 50.2 | 8.5 |
| GREET | 非同配 | 43.7 | 40.0 | 5.9 | 57.3 | 55.5 | 8.1 |
| GraphMAE | Gen | 16.7 | 16.5 | 1.4 | 56.5 | 53.4 | 9.2 |
| w/o Step 1 | 变体 | 17.4 | 12.4 | 0.8 | 55.9 | 71.0 | 9.5 |
| w/o Step 2 | 变体 | 43.4 | 35.1 | 5.4 | 48.4 | 66.2 | 9.3 |
| **FUEL** | **本文** | **50.1** | **43.8** | **6.7** | **59.0** | **71.3** | **2.4** |

聚类任务同样排名第一（2.4），在非同配图上优势尤为明显（Wisconsin 50.1 vs 次优 47.7，Cornell 43.8 vs 次优 43.5）。

## 亮点与洞察

- **理论与实践结合扎实**：隐类别可分性作为类别可分性代理的有效性，同时有实验相关性分析（Spearman > 0.82）和理论证明（Theorem 1），不是简单的 heuristic
- **自适应行为符合直觉**：在同配图（Cora、Citeseer）上，模型自动将大部分权重分配给图卷积项；在非同配图（Actor、Cornell）上，图卷积权重被压至近零，几乎完全依赖原始特征。这种行为无需人工调参
- **可扩展性优势**：FUEL 能成功运行在所有14个数据集上，而多数非同配基线（GREET、PolyGCL、HeterGCL 等）在 Penn94 (42k节点)、Flickr (89k) 等大图上 OOM
- **设计简洁有效**：仅3个标量参数 + 聚类中心 + 一个前馈网络，算法复杂度低，但通过精心设计的损失函数实现了强大的自适应能力

## 局限与展望

- **图卷积阶数固定为2**：仅使用 0、1、2 阶邻接矩阵的加权组合，可能无法捕获长距离依赖。更高阶或可学习阶数的扩展值得探索
- **聚类数需预设**：实验中将聚类数设为真实类别数，虽然附录显示对此不敏感，但完全无监督的聚类数估计仍是开放问题
- **理论分析受限于简化假设**：Theorem 1 假设两类等大小 + 高斯分布，与实际多类不平衡场景有差距
- **仅适用于节点级任务**：未验证在图级或边级下游任务上的效果
- **未考虑异构图和时序图**：作者在结论中指出这是未来方向

## 与相关工作的对比

- **PolyGCL** (Chen et al., 2024)：使用多种图滤波器并对比不同滤波器的嵌入，在非同配图上表现较好（平均排名9.1），但需多组滤波器增加了计算开销且无法扩展到大图
- **GREET** (Liu et al., 2023)：通过特征/拓扑相似度过滤非同配边后再做自监督学习，平均排名7.8为基线中最优，但边过滤策略可能丢失有用的跨类连接信息
- **HeterGCL** (Wang et al., 2024)：专门为非同配图设计的对比学习方法，但在同配图上无明显优势，且同样受限于内存
- **GPR-GNN / GADC**（有监督）：学习每阶邻接矩阵的系数，与 FUEL Step 1 思路类似，但依赖标签监督。FUEL 用聚类作为代理实现了无监督版本的类似功能

## 评分

- 新颖性: ⭐⭐⭐⭐ — 隐类别可分性代理 + 自适应图卷积在无监督场景中是新颖的组合，理论支撑完整
- 实验充分度: ⭐⭐⭐⭐⭐ — 14个数据集、15个基线、分类+聚类两个下游任务、完整的消融实验和设计目标验证
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机-分析-方法-验证的叙事线流畅，理论部分易于理解
- 价值: ⭐⭐⭐⭐ — 在无标签+非同配图这个实际痛点上给出了简洁有效的解决方案，平均排名大幅领先

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily](beyond_fixed_depth_adaptive_graph_neural_networks_for_node_classification_under_.md)
- [\[AAAI 2026\] UniHR: Hierarchical Representation Learning for Unified Knowledge Graph Link Prediction](unihr_hierarchical_representation_learning_for_unified_knowledge_graph_link_pred.md)
- [\[ICML 2025\] Banyan: Improved Representation Learning with Explicit Structure](../../ICML2025/graph_learning/banyan_improved_representation_learning_with_explicit_structure.md)
- [\[ACL 2026\] ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning](../../ACL2026/graph_learning/ark_answer-centric_retriever_tuning_via_kg-augmented_curriculum_learning.md)
- [\[AAAI 2026\] Posterior Label Smoothing for Node Classification](posterior_label_smoothing_for_node_classification.md)

<!-- RELATED:END -->
