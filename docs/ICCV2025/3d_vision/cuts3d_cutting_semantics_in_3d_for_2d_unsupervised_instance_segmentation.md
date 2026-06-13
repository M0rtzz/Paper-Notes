---
title: >-
  [论文解读] CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation
description: >-
  [3D视觉] 提出CutS3D方法，首次将3D信息（单目深度估计）引入无监督实例分割，通过在3D点云中切割语义区域来分离2D中重叠的实例，并引入空间置信度机制提升伪标签质量，在多个基准上超越CutLER等SoTA。 - 问题定义：无监督实例分割旨在不依赖人工标注的情况下，将图像中的每个物体实例单独分割出来…
tags:
  - "3D视觉"
---

# CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation

## 基本信息
- **会议**: ICCV 2025
- **arXiv**: 2411.16319
- **代码**: [leonsick.github.io/cuts3d](https://leonsick.github.io/cuts3d)
- **领域**: 3D视觉 / 无监督实例分割
- **关键词**: 无监督实例分割, 3D语义分割, Normalized Cut, 深度估计, 伪标签

## 一句话总结

提出CutS3D方法，首次将3D信息（单目深度估计）引入无监督实例分割，通过在3D点云中切割语义区域来分离2D中重叠的实例，并引入空间置信度机制提升伪标签质量，在多个基准上超越CutLER等SoTA。

## 研究背景与动机

- **问题定义**：无监督实例分割旨在不依赖人工标注的情况下，将图像中的每个物体实例单独分割出来。
- **现有方法局限**：
    - CutLER（SoTA）：利用DINO自监督特征构建语义亲和图，通过MaskCut提取伪标签。但仅考虑2D语义关系，无法分离在2D图像空间中重叠或连接的同类实例（如一前一后的网球运动员）
    - FreeSOLO、CuVLER等也受限于2D语义信息
    - 人类天然以3D方式感知世界，利用空间边界区分实例
- **关键洞察**：现代零样本单目深度估计器已无需人工标注即可获取精确3D信息，引入3D不违反无监督设定。在3D中切割语义掩码可以正确分离2D中重叠的实例。

## 方法详解

### 整体框架

CutS3D在CutLER流程基础上引入三个核心组件：（1）LocalCut：在3D中切割实例；（2）Spatial Importance Sharpening：用深度信息增强语义亲和图；（3）Spatial Confidence：评估伪标签质量并优化检测器训练。

### LocalCut：3D实例切割

1. 使用ZoeDepth零样本单目深度估计获取深度图$D$
2. 正交反投影为点云$P = \{p_1, ..., p_m\}$
3. 以NCut的初始语义二分$B$为基础，将语义区域外的点设为背景深度
4. 在点云上构建k-NN图$G^{3D}$，边权为欧氏距离
5. 用阈值$\tau_\text{knn}$截断图后，用**MinCut**（Dinic算法）在3D空间中切割实例
6. 源节点$s$和汇节点$t$的选择连接了语义和3D空间：
    - $s = p_{\lambda_\max}$（NCut最大特征值的点，语义前景）
    - $t = p_{\lambda_\min}$（NCut最小特征值的点，语义背景）
7. 最终用CRF精化掩码

### Spatial Importance Sharpening（空间重要性锐化）

**目标**：让语义亲和图感知3D边界，使初始语义掩码更完整地覆盖实例。

对深度图$D$做高斯模糊后取差，获取空间重要性图（高频深度变化区域）：

$$\Delta D = |G_\sigma * D - D|$$

归一化到$[\beta, 1.0]$（$\beta=0.45$），然后用逐元素幂运算锐化语义亲和矩阵：

$$\mathbf{W}_{i,j} = W_{i,j}^{1 - \Delta D_{n_{i,j}}}$$

深度变化剧烈处（物体边界）的语义相似度被压低，使NCut更倾向于沿3D边界切割。

### Spatial Confidence（空间置信度）

**动机**：伪标签存在歧义，需要评估质量以提供更干净的学习信号。

**计算方法**：在$\tau_\text{knn}^{min}$和$\tau_\text{knn}$之间均匀采样$T$个值，分别执行LocalCut，累加平均得到置信度图：

$$\text{SC}_{i,j} = \frac{1}{T}\sum_{t=1}^{T} \text{BC}_{i,j}(t)$$

直觉：3D边界清晰的物体在不同阈值下切割结果一致（高置信度），边界模糊的物体结果不稳定（低置信度）。

**三种使用方式**：
1. **置信度Copy-Paste选择**：仅选择高置信度掩码进行copy-paste数据增强
2. **置信度Alpha-Blending**：按置信度进行alpha混合（低置信度区域半透明）
3. **Spatial Confidence Soft Target Loss**：逐patch重加权掩码损失

$$L_\text{mask} = \sum_{(i,j)} \text{SC}_{i,j} \cdot \text{BCE}(\hat{M}_{i,j}, M_{i,j})$$

## 实验关键数据

### 零样本无监督实例分割主结果

| 方法 | COCO val2017 AP^mask | COCO val2017 AP^mask_50 | COCO20K AP^mask | COCO20K AP^mask_50 |
|------|---------------------|------------------------|-----------------|-------------------|
| FreeSOLO | 4.3 | 9.4 | 4.3 | 9.7 |
| CutLER | 9.7 | 18.9 | 10.0 | 19.6 |
| CuVLER | 9.8 | 19.3 | 10.0 | 20.0 |
| ProMerge+ | 8.9 | - | 9.0 | - |
| **CutS3D** | **10.7** | **20.8** | **10.9** | **21.3** |

CutS3D在COCO val2017上比最优竞争方法提升+0.9 AP^mask和+1.5 AP^mask_50。

### 消融实验

| 方法 | AP^box_50 | AP^box | AP^mask_50 | AP^mask |
|------|----------|--------|-----------|---------|
| CutLER (DiffNCuts) | 22.1 | 12.3 | 18.7 | 9.4 |
| + LocalCut | 22.9 | 12.5 | 18.9 | 9.5 |
| + Spatial Importance | 23.3 | 12.6 | 19.2 | 9.8 |
| + Spatial Confidence | 23.9 | 13.0 | 20.1 | 10.2 |
| + 3轮自训练 | 24.3 | 13.3 | 20.8 | 10.7 |

每个组件独立贡献提升，LocalCut + Spatial Importance相互增益最显著。

### Spatial Confidence组件分析

| 置信度Copy-Paste | Alpha Blend | SC Loss | AP^mask |
|-----------------|-------------|---------|---------|
| ✗ | ✗ | ✗ | 8.5 |
| ✓ | ✗ | ✗ | 8.8 |
| ✓ | ✓ | ✗ | 9.0 |
| ✓ | ✓ | ✓ | 9.1 |

### 零样本目标检测（6个数据集平均）

| 方法 | Average AP^box_50 | Average AP^box |
|------|------------------|----------------|
| CuVLER | 21.3 | 11.3 |
| CutLER | 21.6 | 11.6 |
| **CutS3D** | **23.9** | **12.5** |

CutS3D以单个特征提取器超越使用6个DINO模型集成的CuVLER，说明3D信息比额外特征提取器更有效。

### 深度源对比

| 深度估计器 | AP^mask_50 | AP^mask |
|-----------|-----------|---------|
| ZoeDepth | 18.0 | 9.1 |
| Kick Back & Relax | 17.8 | 9.1 |
| Marigold | 17.7 | 9.0 |
| MiDaS (Small) | 17.6 | 8.9 |

不同深度估计器效果相近，方法对深度源不敏感。

## 亮点与洞察

1. **3D信息首次用于无监督实例分割**：利用零样本深度估计不违反无监督设定，但显著提升实例分离能力
2. **语义-空间双驱动的MinCut**：NCut提供语义前景/背景定义，MinCut在3D空间执行实际切割，两者巧妙结合
3. **Spatial Importance Sharpening的互补效应**：改善初始语义掩码使LocalCut能更准确地找到3D边界
4. **Spatial Confidence的patch级质量评估**：比CuVLER的标量重加权更精细，逐patch反映伪标签可靠性
5. **单模型对抗集成**：CutS3D仅用1个特征提取器+深度估计即超越CuVLER的6模型集成

## 局限性

- 依赖单目深度估计的质量，虽然实验证明对深度源不敏感，但极端场景仍可能受限
- 仅在自然图像数据集上验证，未测试医学、遥感等特殊领域
- LocalCut的k-NN图构建在大规模图像上可能带来计算开销
- 正交投影近似可能在大深度范围场景中引入误差
- Spatial Confidence需要多次运行LocalCut，增加伪标签生成时间

## 相关工作与启发

- CutLER：基础流程的直接前身，本文在其使用的MaskCut和自训练策略上进行增强
- CuVLER：使用6模型集成和标量soft target loss，本文用3D信息和patch级置信度更高效
- DiffNCuts：对DINO微调的特征提取器，作为本文的backbone使用
- **启发**：在其他依赖2D语义的任务中引入3D信息（如零样本深度）是一个通用的增强思路

## 评分

- **新颖性**: ⭐⭐⭐⭐ （3D信息用于无监督分割的首创，空间置信度设计巧妙）
- **实验**: ⭐⭐⭐⭐ （6个数据集全面验证，消融细致，深度源对比有价值）
- **写作**: ⭐⭐⭐⭐ （图示清晰，数学推导严谨，可视化对比有说服力）
- **价值**: ⭐⭐⭐⭐ （为无监督分割引入新维度，方法简洁可推广）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)
- [\[ICCV 2025\] Disentangling Instance and Scene Contexts for 3D Semantic Scene Completion](disentangling_instance_and_scene_contexts_for_3d_semantic_scene_completion.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[ICCV 2025\] Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)
- [\[CVPR 2025\] DUNE: Distilling a Universal Encoder from Heterogeneous 2D and 3D Teachers](../../CVPR2025/3d_vision/dune_distilling_a_universal_encoder_from_heterogeneous_2d_and_3d_teachers.md)

</div>

<!-- RELATED:END -->
