---
title: >-
  [论文解读] SUM Parts: Benchmarking Part-Level Semantic Segmentation of Urban Meshes
description: >-
  [CVPR 2025][3D视觉][城市网格语义分割] 提出首个大规模城市纹理网格部件级语义分割基准数据集 SUM Parts（覆盖 $2.5\,\text{km}^2$，21类），包含面标注和纹理像素标注两种标签类型，并开发了结合3D模板匹配和2D模板匹配的高效交互式标注工具。
tags:
  - CVPR 2025
  - 3D视觉
  - 城市网格语义分割
  - 部件级标注
  - 纹理标注
  - 交互式标注工具
  - 基准数据集
---

# SUM Parts: Benchmarking Part-Level Semantic Segmentation of Urban Meshes

**会议**: CVPR 2025  
**arXiv**: [2503.15300](https://arxiv.org/abs/2503.15300)  
**代码**: [GitHub](https://tudelft3d.github.io/SUMParts/)  
**领域**: 3D视觉  
**关键词**: 城市网格语义分割, 部件级标注, 纹理标注, 交互式标注工具, 基准数据集

## 一句话总结

提出首个大规模城市纹理网格部件级语义分割基准数据集 SUM Parts（覆盖 $2.5\,\text{km}^2$，21类），包含面标注和纹理像素标注两种标签类型，并开发了结合3D模板匹配和2D模板匹配的高效交互式标注工具。

## 研究背景与动机

- 城市场景语义分割主要集中在图像和点云上，纹理网格（textured mesh）作为更丰富的空间表示形式却被严重忽视
- 现有网格数据集主要聚焦于小规模室内场景，缺少大规模室外环境的细粒度语义标签
- 城市场景理解中缺乏部件级（part-level）语义分割，例如窗户、烟囱、道路标线等功能组件的细分
- 已有的3D标注方法多数只标注mesh顶点或面，忽略了纹理图像中更丰富的细节信息
- 纹理网格相比点云具有更好的分辨率和完整性，但缺乏对应的部件级标注数据集
- 大规模3D场景标注的人力成本极高，需要开发更高效的交互式标注工具
- 受CityGML国际标准启发，部件级分割对自动化城市建模至关重要
- 现有交互式标注方法（如SAM）在3D网格纹理场景下表现有限，尤其在旋转和尺度不变性方面

## 方法详解

### 整体框架

SUM Parts 系统包含两个核心模块：**面标注模块**（Face-based annotation）和**纹理标注模块**（Texture-based annotation），分别对三角网格面和纹理像素进行部件级语义标注。标注基于赫尔辛基城市航拍重建的纹理网格（地面采样距离约 $7.5\,\text{cm}$），覆盖40个tile共约 $2.5\,\text{km}^2$，定义了13种面标签和8种附加纹理像素标签（共21类）。系统同时提出了superpixel texture sampling策略用于将纹理标注映射到点云进行语义分割评估。

### 关键设计

**设计一：交互式3D突起提取（Interactive 3D Protrusion Selection）**
- **功能**：从过分割的平面片段中交互式提取非平面结构（如烟囱、阳台等突起物）
- **核心思路**：将突起提取形式化为二元标注问题 $l^f = \{\text{support plane}, \text{protrusion}\}$。构建候选面的对偶图 $\mathcal{G}^f$，设计包含几何特征的数据项 $D^f$ 和基于收缩球半径的平滑项 $V^f$，通过图割优化目标函数 $E^f(l^f) = \sum_i D^f(l^f_i) + \lambda^f \sum_{\{i,j\}} V^f(l^f_i, l^f_j)$
- **设计动机**：传统过分割方法在非平面区域、尖锐特征和小尺度结构上容易产生欠分割或过分割，基于图割的二元标注能更准确地分离突起物和支撑平面

**设计二：3D模板匹配（3D Template Matching）**
- **功能**：利用城市场景中重复结构（如窗户排列）实现批量标注
- **核心思路**：用户选择模板后，分别对平面片段和突起进行结构感知特征匹配。平面匹配基于几何均匀性、空间分布、方向、球形度等特征向量 $\mathbf{F}^{(\text{seg})}$；突起匹配先分解模板为平面片段种子，然后通过空间约束和尺度约束扩展候选区域，基于紧凑性、表面复杂度等特征向量 $\mathbf{F}^{(\text{str})}$ 进行匹配
- **设计动机**：城市建筑具有高度重复性，模板匹配可大幅减少人工交互次数，标注效率提升约1.73倍

**设计三：交互式2D纹理选择与模板匹配**
- **功能**：在纹理图像上实现旋转和尺度不变的区域选择与匹配
- **核心思路**：先用SLIC生成超像素，通过user click触发局部扩展（基于GMM Wasserstein距离的图割优化），再用GrabCut进行像素级细分割。2D模板匹配利用区域结构特征（形状指数、形状规则度、上下文颜色特征）替代NCC，实现旋转和尺度不变
- **设计动机**：NCC在存在旋转和尺度变化的3D城市场景中表现不佳，基于区域结构特征的匹配方法兼具鲁棒性和效率

### 损失函数/优化目标

系统使用基于能量函数的优化框架：面标注能量 $E^f$ 由数据项（基于protrusion score $p_i = d_i + \omega_i \theta_i$）和平滑项（基于收缩球半径差异 $R_{i,j}$）组成；纹理标注能量 $E^s$ 由超像素颜色相似度数据项（Wasserstein距离）和颜色差异平滑项（CIEDE2000色差）组成，均通过图割算法最小化。

## 实验关键数据

### 主实验：3D语义分割方法比较（mIoU / mAcc）

| 方法 | Face mIoU | Face mAcc | Pixel mIoU | Pixel mAcc |
|------|-----------|-----------|------------|------------|
| PointNet | 15.1 | 22.0 | 2.6 | 9.8 |
| PointNet++ | 33.1 | 46.9 | 24.7 | 35.2 |
| SparseUNet | 60.5 | 71.7 | 34.5 | 45.1 |
| KPConv | 57.5 | 64.7 | 42.6 | 58.3 |
| PointNext | 65.3 | 77.2 | 44.7 | 57.6 |
| PointTransV3 | 59.1 | 70.2 | 38.0 | 54.1 |
| **PointVector** | **70.0** | **80.7** | **47.9** | **63.8** |

### 消融实验：采样策略影响（Face Labeling mIoU平均值）

| 采样方法 | Face Avg. mIoU | Pixel Avg. mIoU |
|----------|---------------|-----------------|
| Face-centered | 48.0 | - |
| Random | 40.7 | 30.6 |
| Poisson-disk | 38.2 | 27.8 |
| Superpixel (Ours) | 44.0 | **31.5** |

### 关键发现
- PointVector在面标注和像素标注两个track上均取得最优性能（mIoU 70.0%和47.9%）
- 面中心采样在面标注上最优，因为三角形密度自然适应几何复杂度；但不适用于纹理标注
- 提出的superpixel texture sampling在像素标注上优于其他采样方法
- 交互式标注工具将面标注速度提升1.73倍，智能交互使用率超过80%

## 亮点与洞察

1. **首创Part-level城市网格基准**：填补了城市纹理网格部件级语义分割的空白，21类标签体系遵循CityGML国际标准
2. **双模态标注创新**：面标注与纹理标注双轨并行，纹理标注能捕获面标注无法表示的细节（如道路标线）
3. **无监督交互式标注**：不依赖大规模预训练数据，通过几何特征和模板匹配实现高效标注，泛化性优于SAM等深度学习方法
4. **实用的superpixel采样**：弥合了纹理标注与点云分割方法之间的gap

## 局限与展望

- 依赖几何精度和结构清晰度，对拓扑错误或低分辨率网格效果受限
- 不适用于自然场景（山脉）或复杂非规则结构（宫殿），依赖平面和突起假设
- 纹理标注对复杂纹理、杂乱背景、阴影区域和颜色区分度低的区域表现下降
- 像素标注track的mIoU最高仅47.9%，仍有很大提升空间
- 未来可结合基础模型（如SAM2）提升交互标注效率，或扩展到更多城市的跨域泛化

## 相关工作与启发

- 与ScanNet等室内网格数据集不同，本文专注于大规模室外城市场景的部件级标注
- 对CityGML LoD3自动化建模有直接推动作用
- Superpixel采样策略可推广到其他需要将纹理信息映射到3D表示的任务

## 评分

⭐⭐⭐⭐ — 高质量的基准工作，填补了城市网格部件级分割的重要空白；标注工具设计精巧但应用场景受限于规则建筑结构。

<!-- RELATED:START -->

## 相关论文

- [PartRM: Modeling Part-Level Dynamics with Large Cross-State Reconstruction Model](partrm_modeling_part-level_dynamics_with_large_cross-state_reconstruction_model.md)
- [Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation](rewis3d_reconstruction_improves_weakly-supervised_semantic_segmentation.md)
- [3×2: 3D Object Part Segmentation by 2D Semantic Correspondences](../../ECCV2024/3d_vision/3x2_3d_object_part_segmentation_by_2d_semantic_correspondenc.md)
- [BFANet: Revisiting 3D Semantic Segmentation with Boundary Feature Analysis](bfanet_revisiting_3d_semantic_segmentation_with_boundary_feature_analysis.md)
- [JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](jopp-3d_joint_open_vocabulary_semantic_segmentation_on_point_clouds_and_panorama.md)

<!-- RELATED:END -->
