---
title: >-
  [论文解读] FireGNN: Neuro-Symbolic Graph Neural Networks with Trainable Fuzzy Rules for Interpretable Medical Image Classification
description: >-
  [NeurIPS 2025][医学图像][图神经网络] 提出 FireGNN，首次将可训练模糊规则嵌入 GNN 前向传播中，利用节点度、聚类系数和标签一致性三个拓扑描述子实现内生可解释的医学图像分类，在 5 个 MedMNIST 数据集和 MorphoMNIST 上取得优于标准 GCN/GAT/GIN 及辅助任务方法的性能。
tags:
  - NeurIPS 2025
  - 医学图像
  - 图神经网络
  - 模糊规则
  - 可解释性
  - 神经符号推理
  - MedMNIST
  - 拓扑描述子
---

# FireGNN: Neuro-Symbolic Graph Neural Networks with Trainable Fuzzy Rules for Interpretable Medical Image Classification

**会议**: NeurIPS 2025  
**arXiv**: [2509.10510](https://arxiv.org/abs/2509.10510)  
**代码**: [GitHub](https://github.com/basiralab/FireGNN)  
**领域**: 医学图像分类 / 可解释性  
**关键词**: 图神经网络, 模糊规则, 可解释性, 神经符号推理, MedMNIST, 拓扑描述子

## 一句话总结

提出 FireGNN，首次将可训练模糊规则嵌入 GNN 前向传播中，利用节点度、聚类系数和标签一致性三个拓扑描述子实现内生可解释的医学图像分类，在 5 个 MedMNIST 数据集和 MorphoMNIST 上取得优于标准 GCN/GAT/GIN 及辅助任务方法的性能。

## 研究背景与动机

**医学 AI 的可解释性需求**：临床场景中模型不仅需要高准确率，还需要提供透明、可理解的推理依据以获得临床信任

**GNN 的黑箱问题**：标准 GNN（GCN、GAT）在医学图像上取得了不错的关系建模能力，但缺乏对预测原因的解释——例如肿瘤 patch 被分类为恶性时，无法知道哪些节点特征或连接最重要

**现有可解释方法的不足**：
   - 后验方法（GNNExplainer 等）在模型外部运行，可能无法忠实反映模型内部推理
   - 已有的模糊-GNN 混合方法使用固定规则模板和预定义阈值，不能适应不同数据集

**拓扑信号的浪费**：许多 GNN 使用简单启发式（如空间邻近性）建图，忽略了生物学上有意义的拓扑结构

## 方法详解

### 图构建

将每张医学图像作为一个节点 $v_i$，特征 $f_i = F(x_i)$，标签 $y_i$。通过 top-$k$ 余弦相似度构建邻接矩阵 $A$。标准 GCN 各层更新：

$$H^{(\ell+1)} = \sigma\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(\ell)}W^{(\ell)}\right)$$

其中 $\tilde{A} = A + I$，$\tilde{D}$ 为度矩阵。

### 拓扑特征提取

为每个节点 $u$ 构建事实向量 $f_u \in \mathbb{R}^3$：

$$f_u = [d(u),\; C(u),\; L(u)]$$

- $d(u)$：节点度（连接数量）
- $C(u)$：聚类系数（邻居间实际边数/最大可能边数）
- $L(u)$：2-hop 标签一致性（2跳邻域内共享标签的比例）

### 可训练模糊规则

每条规则 $i$ 有可学习阈值 $\theta_i$ 和锐度参数 $\alpha_i$，激活强度通过 sigmoid 计算：

$$r_i(u) = \sigma\left(\alpha_i(f_u[i] - \theta_i)\right) \in [0, 1]$$

三条规则对应：
- **规则 1**：IF degree $\geq \theta_1$ THEN high_connectivity
- **规则 2**：IF clustering $\geq \theta_2$ THEN high_cliquishness
- **规则 3**：IF 2-hop label agreement $\geq \theta_3$ THEN high_label_consistency

### 门控融合

将规则向量投影到 GNN 嵌入空间后与 GNN 嵌入融合：

$$e_u = W_r r(u) + b_r \in \mathbb{R}^d$$

$$g_u = \sigma(W_g[h_u \| e_u] + b_g) \in [0,1]^d$$

$$h_u' = g_u \odot h_u + (1 - g_u) \odot e_u$$

门控值 $g_u$ 自适应地在 GNN 嵌入和规则嵌入间分配权重，使模型可以解释为"该节点因高连通度和强标签一致性被分类为膀胱组织"。

### 端到端训练

GCN 权重、模糊规则参数 $\{\theta_i, \alpha_i\}$、融合权重 $(W_r, W_g)$ 联合用交叉熵损失训练。

## 实验关键数据

### 主实验：六个数据集上的 GNN 变体比较

| 方法 | OrganCMNIST ACC | OrganAMNIST ACC | OrganSMNIST ACC | TissueMNIST ACC | BloodMNIST ACC |
|------|----------------|----------------|----------------|----------------|---------------|
| GCN | 88.20±0.61 | 91.85±0.30 | 78.62±0.82 | 50.90±0.32 | — |
| GCN + Aux | 88.41±0.44 | 93.11±0.24 | 79.19±0.74 | 52.70±0.22 | — |
| **GCN + FR** | **91.41±0.61** | **94.32±0.18** | **85.05±0.43** | **65.73±0.88** | — |
| GAT | 90.31±0.28 | 93.69±0.36 | 81.80±0.68 | 51.53±0.35 | — |
| GAT + Aux | 90.88±0.49 | 93.70±0.46 | 81.69±0.68 | OOM | — |
| **GAT + FR** | **91.66±0.48** | **94.52±0.31** | **84.82±0.52** | OOM | — |
| GIN | 87.96±0.59 | 91.54±0.71 | 77.23±0.62 | 50.51±1.09 | — |
| **GIN + FR** | **89.12±1.18** | **92.48±1.82** | — | — | — |

> +FR = Fuzzy Rules; +Aux = 辅助任务（同质性预测+相似熵）。模糊规则在所有 backbone 和数据集上一致优于基线和辅助任务方法。

### 学习到的模糊规则示例（OrganCMNIST）

| 规则 | 阈值 $\theta$ | 含义 |
|------|--------------|------|
| Rule 1 (度) | 7.28 | IF degree ≥ 7.28 → high_connectivity |
| Rule 2 (聚类) | 0.18 | IF clustering ≥ 0.18 → high_cliquishness |
| Rule 3 (标签一致性) | 0.67 | IF 2-hop agreement ≥ 0.67 → high_label_consistency |

**解释示例**：某节点被预测为"膀胱"，度 = 10（≥ 7.28, 激活 0.60）、聚类系数 = 0.18（= $\theta_2$, 激活 0.50）、标签一致性 = 0.73（≥ 0.67, 激活 0.56）→ 模型因高连通度和强标签一致性做出此分类。

### 关键改进幅度

- OrganSMNIST 上 GCN+FR vs GCN：**+6.43% ACC**（78.62 → 85.05）
- TissueMNIST 上 GCN+FR vs GCN：**+14.83% ACC**（50.90 → 65.73）——提升极其显著
- OrganCMNIST 上 GAT+FR 达到最高 **91.66% ACC**

## 亮点

1. ⭐⭐⭐ **首创可训练模糊规则嵌入 GNN**：阈值和锐度从数据中学习，不依赖手工定义，实现了"数据驱动的符号推理"
2. ⭐⭐⭐ **内生可解释性**：规则激活直接提供人类可读的分类依据（如"高连通度+高标签一致性→膀胱"），优于后验解释方法
3. ⭐⭐ **通用性**：模糊规则模块可即插即用于 GCN/GAT/GIN 等多种 GNN backbone，不改变基础架构
4. ⭐⭐ **辅助任务作为基准**：系统对比了两种注入拓扑信息的方式（模糊规则 vs 辅助任务），证明了符号推理的优越性

## 局限与展望

1. **仅三条规则**：拓扑描述子限于度、聚类系数、标签一致性三个；更复杂的图特征（如介数中心性、motif 计数）可能带来更多信息
2. **数据集规模较小**：MedMNIST 数据集图像分辨率低（28×28），与真实临床场景的高分辨率医学图像差距大
3. **归纳式设置的可扩展性**：图构建依赖 top-$k$ 余弦相似度，在大规模数据集上的计算成本和图质量需要进一步验证
4. **GAT+FR 在 TissueMNIST 上 OOM**：注意力机制+模糊规则在大图上的内存问题尚未解决
5. **规则的语义粒度**：当前规则捕获的是全局拓扑属性，无法解释图像内部的局部视觉特征

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐ |
| 综合推荐 | ⭐⭐⭐ |

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

## 相关论文

- [SIC: Similarity-Based Interpretable Image Classification with Neural Networks](../../ICCV2025/medical_imaging/sic_similarity-based_interpretable_image_classification_with_neural_networks.md)
- [Random Search Neural Networks for Efficient and Expressive Graph Learning](random_search_neural_networks_for_efficient_and_expressive_graph_learning.md)
- [Neuro-Symbolic Decoding of Neural Activity](../../ICLR2026/medical_imaging/neuro-symbolic_decoding_of_neural_activity.md)
- [Interpretable Next-token Prediction via the Generalized Induction Head](interpretable_next-token_prediction_via_the_generalized_induction_head.md)
- [MuSLR: Multimodal Symbolic Logical Reasoning](muslr_multimodal_symbolic_logical_reasoning.md)

<!-- RELATED:END -->
