---
title: >-
  [论文解读] Temporal Rate Reduction Clustering for Human Motion Segmentation
description: >-
  [ICCV 2025][图像分割][人体运动分割] 提出 Temporal Rate Reduction Clustering (TR²C) 方法，将最大编码率约简（MCR²）原理与时序连续性正则化相结合，联合学习符合子空间联合（UoS）分布的时序一致表示与亲和度矩阵，在五个基准上大幅刷新人体运动分割 SOTA。
tags:
  - ICCV 2025
  - 图像分割
  - 人体运动分割
  - 子空间聚类
  - 最大编码率约简
  - 时序一致性
  - 无监督时序聚类
---

# Temporal Rate Reduction Clustering for Human Motion Segmentation

**会议**: ICCV 2025  
**arXiv**: [2506.21249](https://arxiv.org/abs/2506.21249)  
**代码**: [GitHub](https://github.com/mengxianghan123/TR2C)  
**领域**: 图像分割  
**关键词**: 人体运动分割, 子空间聚类, 最大编码率约简, 时序一致性, 无监督时序聚类

## 一句话总结

提出 Temporal Rate Reduction Clustering (TR²C) 方法，将最大编码率约简（MCR²）原理与时序连续性正则化相结合，联合学习符合子空间联合（UoS）分布的时序一致表示与亲和度矩阵，在五个基准上大幅刷新人体运动分割 SOTA。

## 研究背景与动机

人体运动分割（HMS）旨在将视频帧序列划分为不同的非重叠运动段。由于标注成本高昂，HMS 通常被视为无监督时序聚类任务。现有方法主要基于子空间聚类假设，即视频帧特征近似分布在低维子空间的联合（Union of Subspaces, UoS）上。

然而，本文指出了现有方法的核心瓶颈：

**数据-假设不匹配**：包含复杂人体运动和杂乱背景的视频帧特征很难良好地符合 UoS 分布假设。已有表示学习方法（auto-encoder、图一致性等）虽然尝试学习更好的特征，但**没有证据表明学到的表示真正对齐了 UoS 结构**。

**时序先验利用不足**：视频中相邻帧大概率属于同一运动，这一先验虽被 OSC、TSC 等方法利用，但它们在特征空间不对齐的情况下效果有限。

**迁移学习方法的局限**：虽然引入了跨域对齐策略，但性能瓶颈依然存在，根本原因在于没有从表示层面解决分布对齐问题。

本文的关键洞察是：应该**联合学习**符合 UoS 结构的表示和用于分割的亲和度矩阵，同时融入时序一致性约束，让学到的特征在优化过程中自然地对齐到理想的几何结构上。

## 方法详解

### 整体框架

TR²C 的框架包含三个网络组件：编码器 $f(\cdot)$、特征头 $g(\cdot)$ 和聚类头 $h(\cdot)$。输入特征经过编码器提取共享表示，然后分别通过特征头和聚类头产生结构化表示 $\boldsymbol{Z}$ 和亲和度矩阵 $\boldsymbol{\Gamma}$，最终对 $\boldsymbol{\Gamma}$ 做谱聚类得到分割结果。

### 关键设计

1. **MCR² 原理用于联合学习表示和分割**：基于最大编码率约简原理，核心优化目标由三部分组成。$\rho(\boldsymbol{Z}, \epsilon)$ 为总编码率，衡量表示的整体体积（基于 $\log\det$ 函数）；$\rho^c(\boldsymbol{Z}, \epsilon | \boldsymbol{\Pi})$ 为类内编码率之和。最大化总体积、最小化类内体积，使表示自然趋向于彼此正交的子空间联合分布。这是首次将 MCR² 原理用于时序序列聚类任务，其几何直觉是：$\log\det(\cdot)$ 作为 $\text{rank}(\cdot)$ 的凹松弛，能有效衡量表示空间的体积。

2. **时序 Laplacian 正则化**：引入时序图 Laplacian 正则化 $r(\boldsymbol{Z}) = \text{tr}(\boldsymbol{Z}\boldsymbol{L}\boldsymbol{Z}^\top)$，其中 $\boldsymbol{L}$ 是基于滑动窗口（窗口大小 $s$）构建的图 Laplacian 矩阵。该正则化鼓励相邻帧的表示保持相似，实现时序一致性。设计动机是：单纯做 MCR² 优化会忽视视频帧的时序连续性，可能导致相邻但属于同一运动的帧被分到不同子空间。

3. **防止坍缩的总编码率最大化**：直接最小化 $\rho^c + \lambda r(\boldsymbol{Z})$ 存在平凡解（所有嵌入坍缩），类似于图神经网络中的过平滑问题。因此引入 $-\rho(\boldsymbol{Z}, \epsilon)$ 项作为正则化，通过最大化总编码率来防止表示压缩过度。最终优化目标：

$$\min_{\boldsymbol{Z}, \boldsymbol{\Pi}} -\rho(\boldsymbol{Z}, \epsilon) + \lambda_1 \rho^c(\boldsymbol{Z}, \epsilon | \boldsymbol{\Pi}) + \lambda_2 r(\boldsymbol{Z})$$

约束 $\|\boldsymbol{z}_i\|_2^2 = 1$。

4. **可微优化框架**：将离散分配矩阵 $\boldsymbol{\Pi}$ 松弛为双随机亲和矩阵 $\boldsymbol{\Gamma}$，通过 Sinkhorn 投影保证约束满足。网络参数化 $\boldsymbol{Z}$ 和 $\boldsymbol{\Gamma}$ 并使用反向传播更新，实现端到端可微训练。

### 损失函数 / 训练策略

最终损失函数为三项之和：

$$\mathcal{L} = -\mathcal{L}_\rho + \lambda_1 \mathcal{L}_{\bar{\rho}^c} + \lambda_2 \mathcal{L}_r$$

- $\mathcal{L}_\rho$：最大化总编码率，防止表示坍缩
- $\mathcal{L}_{\bar{\rho}^c}$：最小化类内编码率，促进子空间分离
- $\mathcal{L}_r$：时序 Laplacian 正则化，保持时序一致性

网络架构轻量（两层 MLP 编码器 + FC 头），$\lambda_1, \lambda_2$ 按数据集独立调参，滑动窗口固定 $s=2$，训练 500 iterations。

## 实验关键数据

### 主实验

在五个 HMS 基准上基于 HoG 特征的对比（ACC / NMI）：

| 方法 | Weiz ACC | Keck ACC | UT ACC | MAD ACC | YouTube ACC |
|------|---------|---------|--------|---------|------------|
| TSC | 61.11 | 47.81 | 53.40 | 55.56 | 90.40 |
| CDMS (迁移学习) | 65.05 | 62.07 | 66.43 | 65.36 | 67.98 |
| GCTSC (SOTA) | 85.01 | 78.64 | 87.00 | 82.97 | 95.79 |
| **TR²C (本文)** | **94.12** | **83.50** | **93.54** | **83.08** | **97.96** |

TR²C 在**未使用迁移学习**的前提下，聚类精度比迁移学习方法高约 20%，比前 SOTA GCTSC 高 5~9 个点。

### 消融实验

| 损失组合 | $\mathcal{L}_\rho$ | $\mathcal{L}_{\bar{\rho}^c}$ | $\mathcal{L}_r$ | Weiz ACC | Keck ACC | UT ACC |
|---------|:---:|:---:|:---:|---------|---------|--------|
| 仅 MCR²（无时序） | ✓ | ✓ | × | 37.30 | 47.29 | 45.79 |
| 无总编码率项 | × | ✓ | ✓ | 53.14 | 47.91 | 63.13 |
| 无类内编码率项 | ✓ | × | ✓ | 64.68 | 58.60 | 65.67 |
| **完整 TR²C** | ✓ | ✓ | ✓ | **94.07** | **86.78** | **94.05** |

三个损失项缺一不可，其中缺少 $\mathcal{L}_\rho$ 导致表示过度压缩，缺少 $\mathcal{L}_{\bar{\rho}^c}$ 导致过度分割，缺少 $\mathcal{L}_r$ 丧失时序一致性。

### 关键发现

- **表示质量**：PCA 可视化显示，原始 HoG 特征呈一维流形结构，无法清晰分割；TR²C 学到的表示呈现出明确的正交子空间联合结构
- **鲁棒性**：在高斯噪声扰动下，TR²C 表示的聚类精度下降至多 15%，而 GCTSC 下降 45%，证明 UoS 对齐带来了显著的噪声鲁棒性
- **CLIP 特征加持**：使用 CLIP 预训练特征替代 HoG 后，TR²C+CLIP 在 Weiz 上达 96.32，Keck 上达 90.86
- **计算效率**：结合 GPU 加速，TR²C 比 GCTSC 快 100 倍以上（YouTube 数据集：41s vs 8475s）

## 亮点与洞察

- **理论贡献**：首次将 MCR² 原理推广到时序聚类问题，并通过时序正则化和防坍缩机制使之适用于 HMS
- **几何解释清晰**：通过 $\log\det$ 函数衡量子空间体积，优化目标具有明确的几何意义——最大化总体积、最小化类内体积
- **简洁有效**：网络架构仅为两层 MLP + FC 层，训练速度快，且效果大幅领先

## 局限与展望

- 仅在 HoG 和 CLIP 特征上验证，未探索端到端从视频帧直接学习的方案
- 数据集规模较小（百~千帧），在长视频或大规模数据上的可扩展性待验证
- 超参数 $\lambda_1, \lambda_2$ 需要按数据集调整，自适应策略值得探索
- 仅验证了人体运动分割，可推广到更通用的时序分割任务（如动作识别、活动检测等）

## 相关工作与启发

- MCR² (Ma et al.) 提出了编码率约简原理用于有监督学习，MLC 扩展到无监督聚类，本文进一步推广到时序场景
- TSC 引入了时序图 Laplacian 正则化，TR²C 在此基础上增加了表示学习能力
- 后续可探索将 TR²C 与视频基础模型（如 VideoMAE）结合，实现更强的特征提取

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将 MCR² 原理与时序聚类首次结合，理论与方法创新并重
- **实验充分度**: ⭐⭐⭐⭐ 五个基准全面对比，含消融、可视化、鲁棒性、不同特征评估
- **写作质量**: ⭐⭐⭐⭐ 数学推导严谨，动机清晰，但公式较密集
- **价值**: ⭐⭐⭐⭐ 为时序聚类提供了新的理论框架，大幅突破 HMS 性能瓶颈

<!-- RELATED:START -->

## 相关论文

- [Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation](skeleton_motion_words_for_unsupervised_skeleton-based_temporal_action_segmentati.md)
- [MOVE: Motion-Guided Few-Shot Video Object Segmentation](move_motion-guided_few-shot_video_object_segmentation.md)
- [What If: Understanding Motion Through Sparse Interactions](what_if_understanding_motion_through_sparse_interactions.md)
- [2HandedAfforder: Learning Precise Actionable Bimanual Affordances from Human Videos](2handedafforder_learning_precise_actionable_bimanual_affordances_from_human_vide.md)
- [A Plug-and-Play Physical Motion Restoration Approach for In-the-Wild High-Difficulty Motions](a_plug-and-play_physical_motion_restoration_approach_for_in-the-wild_high-diffic.md)

<!-- RELATED:END -->
