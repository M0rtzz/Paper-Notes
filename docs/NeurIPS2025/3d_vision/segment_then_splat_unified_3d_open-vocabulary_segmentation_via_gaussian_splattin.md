---
title: >-
  [论文解读] Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting
description: >-
  [NeurIPS 2025][3D视觉][3D高斯溅射] 提出"先分割再重建"的新范式，在3D高斯溅射重建之前就将高斯分配到不同目标集合，从而消除几何和语义歧义，实现静态和动态场景的统一3D开放词汇分割。
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "3D高斯溅射"
  - "开放词汇分割"
  - "动态场景"
  - "CLIP嵌入"
  - "目标跟踪"
---

# Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting

**会议**: NeurIPS 2025  
**arXiv**: [2503.22204](https://arxiv.org/abs/2503.22204)  
**代码**: [GitHub](https://vulab-ai.github.io/Segment-then-Splat/)  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 开放词汇分割, 动态场景, CLIP嵌入, 目标跟踪

## 一句话总结

提出"先分割再重建"的新范式，在3D高斯溅射重建之前就将高斯分配到不同目标集合，从而消除几何和语义歧义，实现静态和动态场景的统一3D开放词汇分割。

## 研究背景与动机

3D开放词汇查询在机器人、自动驾驶和增强现实等领域至关重要。现有方法几乎都遵循"先重建再分割"（Splat-then-Segment）的范式，存在三个核心问题：

**2D像素级分割的局限性**：大多数方法（如LangSplat、LEGaussians）在3DGS旁学习一个语言场，将语言嵌入渲染成2D特征图再做查询。这本质上是2D分割，不同视角下分割结果不一致，无法获取真正的3D目标信息。

**几何-语义歧义**：在"先重建再分割"范式下，每个高斯可能编码来自多个目标的几何和语义信息，导致目标边界模糊。

**动态场景不可用**：在动态场景中，同一个高斯在不同时间步可能代表不同目标，导致高斯-目标错位（misalignment），现有方法无法直接处理。

本文的核心洞察是：如果在重建之前就把高斯分配给各个目标，那么每个高斯只属于一个目标，就不存在上述歧义问题，且天然适用于动态场景。

## 方法详解

### 整体框架

Segment then Splat 的流程分为四个阶段：（1）鲁棒目标跟踪模块提取多视角掩码；（2）根据掩码将COLMAP初始化的高斯分配到不同目标集合；（3）按目标集合约束进行重建优化；（4）为每个目标集合关联CLIP嵌入实现开放词汇查询。

### 关键设计

1. **鲁棒目标跟踪模块**：利用SAM在首帧做网格点提示分割，然后用SAM2在整个序列中跟踪目标。为应对实际挑战，设计了三个后处理策略：

    - **新目标检测**：每隔 $\Delta t$ 帧比较分割区域比例变化，检测新出现的目标
    - **多重跟踪消解**：用IoU阈值过滤重叠掩码，确保同一粒度下每个像素只属于一个目标
    - **丢失跟踪恢复**：利用高斯集合的"几何-外观距离"合并代表同一目标的不同实例：
    $d(\mathbf{G}_i, \mathbf{G}_j) = \lambda_d |\overline{\mathbf{M}_i} - \overline{\mathbf{M}_j}|_2 + (1 - \lambda_d) |\overline{\mathbf{C}_i} - \overline{\mathbf{C}_j}|_2$

2. **目标特定高斯初始化**：分析每个高斯中心在所有视角中的可见性，确定其所在目标掩码区域，分配三级（大/中/小）目标ID。对于COLMAP未覆盖的目标，随机初始化补偿高斯；另生成背景高斯填充未分割区域。

3. **多粒度约束优化**：在标准渲染损失之外引入目标级损失：
    $\mathcal{L}_{obj} = \mathcal{L}_1(M_i^p \otimes I_i, \hat{I_i^p})$
   为避免对所有目标计算损失带来的计算开销，每次迭代只随机采样 $m$ 个目标。优化顺序从小粒度到大粒度（先小→中→大），因为小目标是大目标的子集，反序会导致大目标内部结构混乱。

4. **部分掩码过滤**：3D分割能重建被遮挡区域的目标，但2D掩码监督不包含被遮挡部分，造成约束偏差。训练末期，将重建目标渲染成2D图像并计算与掩码的IoU，丢弃低IoU掩码，确保最终优化只受一致掩码指导。

### 损失函数 / 训练策略

整体损失为：$\mathcal{L} = \mathcal{L}_{render} + \mathcal{L}_{obj}$

其中渲染损失：$\mathcal{L}_{render} = (1-\lambda_r)\mathcal{L}_1(\hat{I_i}, I_i) + \lambda_r \mathcal{L}_{DSSIM}(\hat{I_i}, I_i)$

目标损失按阶段递增：stage1只优化小粒度，stage2加入中粒度，stage3三级同时优化。训练完成后对每个目标集合计算CLIP嵌入（排除部分掩码后的多视角平均），实现开放词汇查询。

## 实验关键数据

### 主实验

**静态场景分割结果（mIoU↑ / 训练时间min↓）：**

| 方法 | LERF_OVS mIoU | LERF_OVS Time | 3DOVS mIoU | 3DOVS Time |
|------|--------------|---------------|-----------|-----------|
| LangSplat (2D) | 46.37 | 62 | 82.49 | 68.9 |
| G-Grouping (2D) | 29.59 | 77 | 76.24 | 56.1 |
| OpenGaussian (3D) | 42.43 | 69.75 | 31.00 | 59.4 |
| **Ours (3D)** | **52.10** | **50.75** | **88.53** | **9.4** |

**动态场景分割结果：**

| 方法 | HyperNeRF mIoU | HyperNeRF Time | Neu3D mIoU | Neu3D Time |
|------|---------------|----------------|-----------|-----------|
| DGD (3D) | 7.83 | 1564.5 | 1.65 | 1733 |
| **Ours (3D)** | **69.48** | **218** | **44.00** | **161.3** |

### 消融实验

| 配置 | ramen mIoU | waldo_kitchen mIoU | 说明 |
|------|-----------|-------------------|------|
| 每次迭代监督1个目标 | 51.09 | 33.97 | 基线 |
| 每次迭代监督3个目标 | 54.38 | 40.71 | 较好平衡 |
| 每次迭代监督9个目标 | 56.48 | 41.59 | 边际收益递减 |
| 无部分掩码过滤 | 42.19 | 31.94 | 下降明显 |
| **有部分掩码过滤** | **54.38** | **40.71** | 复杂场景提升显著 |

### 关键发现

- 3D分割会"看到"被遮挡部分，导致GT掩码覆盖不全使mIoU略低，但这反而说明3D分割更完整
- 动态场景下优化速度比DGD快近10倍，因为无需学习动态语言场
- 鲁棒跟踪模块的每个组件（新目标检测、多重跟踪消解、丢失跟踪恢复）都有实质性贡献

## 亮点与洞察

- **范式创新**：颠覆了长期以来"先重建再分割"的思路，变为"先分割再重建"，思路简洁且效果突出
- **统一框架**：静态和动态场景用同一框架处理，动态场景无需特殊修改
- **高效性**：不需要学习额外的语言场，单次重建即可完成，训练时间大幅缩短
- **多粒度**：支持大/中/小三级粒度查询，通过优化顺序设计确保各级结构完整

## 局限与展望

- 依赖SAM2做初始分割跟踪，在高度密集、视觉相似的复杂场景中可能失败
- 无法处理涉及多目标关系的文本查询（如"桌子前面椅子上的羊"），因为语义编码是逐目标独立的
- 可引入卡尔曼滤波改善SAM2的时序稳定性

## 相关工作与启发

本文与OpenGaussian（基于对比学习+K-means聚类）和GaussianCut（基于图切割）形成对比。核心启发是：与其在重建后试图分离混合在一起的高斯语义，不如从源头就保持高斯-目标对应关系。这一思路可以推广到其他需要语义分解的3D表示学习任务中。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 范式级创新，将分割前置于重建，思路新颖且优雅
- 实验充分度: ⭐⭐⭐⭐ 覆盖静态/动态数据集，消融充分，但缺少大规模场景实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观，动机阐述到位
- 价值: ⭐⭐⭐⭐⭐ 统一静态/动态场景的3D开放词汇分割，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](../../CVPR2025/3d_vision/jopp-3d_joint_open_vocabulary_semantic_segmentation_on_point_clouds_and_panorama.md)
- [\[NeurIPS 2025\] OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations](openlex3d_a_tiered_evaluation_benchmark_for_open-vocabulary_3d_scene_representat.md)
- [\[CVPR 2026\] OnlinePG: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting](../../CVPR2026/3d_vision/onlinepg_online_open-vocabulary_panoptic_mapping_with_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Mosaic3D: Foundation Dataset and Model for Open-Vocabulary 3D Segmentation](../../CVPR2025/3d_vision/mosaic3d_foundation_dataset_and_model_for_open-vocabulary_3d_segmentation.md)
- [\[CVPR 2025\] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](../../CVPR2025/3d_vision/reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)

</div>

<!-- RELATED:END -->
