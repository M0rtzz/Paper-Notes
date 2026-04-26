---
title: >-
  [论文解读] Global Context-aware Representation Learning for Spatially Resolved Transcriptomics
description: >-
  [ICML2025][目标检测][空间转录组学] 提出 Spotscape 框架，通过 Similarity Telescope 模块捕获 spot 间的全局相似关系（而非仅依赖空间局部邻居），并引入原型对比学习和相似度尺度匹配策略处理多切片批次效应，在空间域识别、轨迹推断、多切片整合与对齐等任务上全面超越现有方法。
tags:
  - ICML2025
  - 目标检测
  - 空间转录组学
  - 图神经网络
  - 自监督学习
  - 全局相似度
  - 批次效应校正
  - 多切片整合
---

# Global Context-aware Representation Learning for Spatially Resolved Transcriptomics

**会议**: ICML2025  
**arXiv**: [2506.15698](https://arxiv.org/abs/2506.15698)  
**代码**: [yunhak0/Spotscape](https://github.com/yunhak0/Spotscape)  
**领域**: 空间转录组学 / 图表示学习  
**关键词**: 空间转录组学, 图神经网络, 自监督学习, 全局相似度, 批次效应校正, 多切片整合

## 一句话总结

提出 Spotscape 框架，通过 Similarity Telescope 模块捕获 spot 间的全局相似关系（而非仅依赖空间局部邻居），并引入原型对比学习和相似度尺度匹配策略处理多切片批次效应，在空间域识别、轨迹推断、多切片整合与对齐等任务上全面超越现有方法。

## 研究背景与动机

空间解析转录组学（Spatially Resolved Transcriptomics, SRT）能同时获取细胞的空间坐标与基因表达谱，是研究组织空间结构的前沿技术。当前基于图的表示学习方法（如 SEDR、STAGATE、GraphST）通过空间近邻图（SNN）聚合局部信息，但存在关键缺陷：

**局部相似度区分力不足**：生物系统的基因表达沿空间坐标连续渐变，导致局部邻居间特征差异极小，即使跨越不同空间域的边界 spot 也难以区分

**注意力机制对边界 spot 无效**：实验表明 GAT 虽提升整体聚类精度（Total CA），却降低了边界 spot 的聚类精度（Boundary CA）

**Oracle 边权也收益有限**：即使用真实标签构造完美边权（同类权重1，异类权重0），边界 CA 提升仍不显著，说明仅靠局部视角获取的信息本质上不足

**多切片批次效应**：多切片整合时，同一切片的表达谱会异常聚集，掩盖真正的生物学意义

## 方法详解

### 整体架构

Spotscape 采用 Siamese 网络结构：对原始 SNN 图 $\mathcal{G}=(X,A)$ 施加两种随机增强（节点特征掩码 + 边掩码），得到两个增强视图 $\tilde{\mathcal{G}}$ 和 $\tilde{\mathcal{G}}'$，通过共享 GNN 编码器 $f_\theta$ 分别生成表示 $\tilde{Z}$ 和 $\tilde{Z}'$。

### Similarity Telescope（核心模块）

提出关系一致性损失，通过对齐两个增强视图间所有 spot 对的余弦相似度矩阵来捕获全局关系：

$$\mathcal{L}_{\text{SC}}(\tilde{Z}, \tilde{Z}') = \text{MSE}\left(\tilde{Z}_{\text{norm}}(\tilde{Z}'_{\text{norm}})^T,\ \tilde{Z}'_{\text{norm}}(\tilde{Z}_{\text{norm}})^T\right)$$

其中 $\tilde{Z}_{\text{norm}}$ 是 L2 归一化后的表示。该损失使模型学习在不同增强下保持一致的全局相似度关系，直接优化 spot 间的相对距离。

### 重建损失（防止退化）

通过共享 MLP 解码器 $g_\theta$ 重建原始特征，防止表示坍缩：

$$\mathcal{L}_{\text{Recon}} = \text{MSE}(X, \hat{X}) + \text{MSE}(X, \hat{X}')$$

单切片总损失：$\mathcal{L}_{\text{Single}} = \lambda_{\text{SC}}\mathcal{L}_{\text{SC}} + \lambda_{\text{Recon}}\mathcal{L}_{\text{Recon}}$

### 原型对比学习（多切片）

对表示做 K-means 聚类获取原型（质心），同簇 spot 为正样本对，异簇为负样本对。重复 $T$ 次不同粒度的聚类以捕获多尺度语义：

$$l_{\text{PCL}}(\tilde{Z}_i, P_{\text{set}}) = \frac{1}{T}\sum_{t=1}^{T}\log\frac{e^{\text{sim}(\tilde{Z}_i, p_{\text{map}_t(i)}^t)/\tau}}{\sum_{j=1}^{K_t}e^{\text{sim}(\tilde{Z}_i, p_j^t)/\tau}}$$

在热身期（500 epochs）后启用，避免早期原型不准确的干扰。

### 相似度尺度匹配（消除批次效应）

核心思想：强制每个 spot 在自身切片内的 top-k 相似度均值与在其他切片的 top-k 相似度均值一致：

$$l_{\text{SS}}(H_i, \mathcal{G}^{(j)}) = \left(\text{Mean}(S_{\text{top}}^{(c)}) - \text{Mean}(S_{\text{top}}^{(j)})\right)^2$$

多切片总损失：$\mathcal{L}_{\text{Multi}} = \lambda_{\text{SC}}\mathcal{L}_{\text{SC}} + \lambda_{\text{Recon}}\mathcal{L}_{\text{Recon}} + \lambda_{\text{PCL}}\mathcal{L}_{\text{PCL}} + \lambda_{\text{SS}}\mathcal{L}_{\text{SS}}$

## 实验关键数据

### 单切片空间域识别（SDI）

| 数据集 | 方法 | ARI | NMI | CA |
|--------|------|-----|-----|-----|
| DLPFC (P1, Slice 151673) | SpaceFlow | 0.42 | 0.57 | 0.57 |
| | **Spotscape** | **0.48** | **0.64** | **0.61** |
| DLPFC (P2, Slice 151507) | SpaceFlow | 0.55 | 0.68 | 0.71 |
| | **Spotscape** | **0.60** | **0.72** | **0.76** |
| MTG-Control | SpaceFlow | 0.66 | 0.74 | 0.70 |
| | **Spotscape** | **0.73** | **0.78** | **0.75** |
| MTG-AD | CAST (次优) | 0.54 | 0.71 | 0.65 |
| | **Spotscape** | **0.68** | **0.75** | **0.77** |

Spotscape 在全部 16 个切片、4 个数据集上的 ARI/NMI/CA 均为最优。

### 多切片同质整合（DLPFC）

| 患者 | 方法 | ARI | NMI | CA |
|------|------|-----|-----|-----|
| Patient 1 | SpaceFlow | 0.48 | 0.60 | 0.60 |
| | **Spotscape** | **0.57** | **0.70** | **0.67** |
| Patient 3 | SpaceFlow | 0.51 | 0.60 | 0.69 |
| | **Spotscape** | **0.63** | **0.68** | **0.75** |

### 异质整合（MTG, CT+AD）

| 方法 | ARI | NMI | CA | Silhouette |
|------|-----|-----|-----|------------|
| CAST | 0.48 | 0.52 | 0.59 | 0.45 |
| STAligner | 0.38 | 0.54 | 0.49 | 0.62 |
| **Spotscape** | **0.72** | **0.76** | **0.81** | **0.69** |

### 多切片对齐（Mouse Embryo, LTARI）

| PASTE2 | CAST | STAligner | SLAT | **Spotscape** |
|--------|------|-----------|------|---------------|
| 0.21 | 0.10 | 0.46 | 0.41 | **0.51** |

## 亮点与洞察

1. **问题发现深刻**：通过 oracle 实验证明局部图结构即使拥有完美边权也无法有效区分边界 spot，从根本上论证了全局相似度学习的必要性
2. **Similarity Telescope 简洁有效**：仅用 MSE 对齐两个增强视图的全局相似度矩阵，无需复杂的负样本采样，直接优化 spot 间的相对距离
3. **相似度尺度匹配策略新颖**：通过匹配 top-k 相似度均值消除批次效应，思路极简却效果显著（移除后聚类性能剧烈下降）
4. **下游任务覆盖全面**：SDI、轨迹推断、基因填补、同质/异质整合、跨技术对齐，每项任务都有定量评估
5. **可扩展性好**：在 100K spots 规模数据上仍保持合理训练时间

## 局限与展望

1. **全局相似度矩阵 $O(N_s^2)$ 复杂度**：虽然实验展示了可扩展性，但全局相似度矩阵的计算和存储在超大规模数据（百万级 spot）上仍可能成为瓶颈
2. **PCL 依赖 K-means**：原型质量取决于 K-means 聚类结果，对簇数 $K$ 和初始化敏感
3. **单切片未使用 PCL**：作者因运行时间-性能权衡放弃在单切片中使用 PCL，但潜在收益未被充分探索
4. **领域分类精度依赖下游聚类**：表示学习本身不产生域标签，仍需后接 K-means 等聚类算法，聚类质量受 K 值选择影响
5. **仅在特定组织/技术上验证**：主要在脑组织（DLPFC, MTG）和少量其他组织上测试，对更多组织类型和测序平台的泛化性有待验证

## 相关工作与启发

- **STAGATE**（Dong & Zhang, 2022）：用 GAT 学习 spot 间注意力权重，本文指出其对边界 spot 效果不佳
- **SpaceFlow**（Ren et al., 2022）：用 DGI + 空间正则化，是最强单方法基线
- **GraphST**（Long et al., 2023）：用 DGI 做批次校正，本文在多切片任务中大幅超越
- **STAligner**（Zhou et al., 2023）：用互近邻 + 三元组损失做切片整合
- **CAST**（Tang et al., 2024）：用 CCA-SSG 做异质切片整合与对齐

启发：在连续特征场景中（如生物组织、遥感），局部图结构的信息瓶颈是普遍问题，全局相似度一致性约束是一个有前景的通用解决思路。

## 评分

- 新颖性: ⭐⭐⭐⭐ — Similarity Telescope 和相似度尺度匹配策略有新意，问题分析（oracle 实验）有深度
- 实验充分度: ⭐⭐⭐⭐⭐ — 5个数据集、6+项下游任务、完整消融/敏感性/可扩展性分析、10次重复+统计检验
- 写作质量: ⭐⭐⭐⭐ — 动机论证清晰，Figure 1 的直觉解释很好
- 价值: ⭐⭐⭐⭐ — 对空间转录组学表示学习有实际推动，方法通用性较强

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning](discovering_global_false_negatives_on_the_fly_for_self-supervised_contrastive_le.md)
- [\[ICML 2025\] Self-Organizing Visual Prototypes for Non-Parametric Representation Learning](self-organizing_visual_prototypes_for_non-parametric_representation_learning.md)
- [\[CVPR 2025\] Generative Modeling of Class Probability for Multi-Modal Representation Learning](../../CVPR2025/object_detection/generative_modeling_of_class_probability_for_multi_modal_representation_learning.md)
- [\[CVPR 2026\] DA-Mamba: Learning Domain-Aware State Space Model for Global-Local Alignment in Domain Adaptive Object Detection](../../CVPR2026/object_detection/da-mamba_learning_domain-aware_state_space_model_for_global-local_alignment_in_d.md)
- [\[NeurIPS 2025\] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs](../../NeurIPS2025/object_detection/sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)

<!-- RELATED:END -->
