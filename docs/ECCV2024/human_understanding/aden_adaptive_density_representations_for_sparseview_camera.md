---
title: >-
  [论文解读] ADen: Adaptive Density Representations for Sparse-view Camera Pose Estimation
description: >-
  [ECCV 2024][人体理解][稀疏视图] 提出ADen框架，通过生成器输出多个位姿假设+判别器评分选择最佳的方式，统一了位姿回归和概率估计范式，仅需500个自适应样本即超越需要500K均匀采样的方法，同时实现实时推理。
tags:
  - ECCV 2024
  - 人体理解
  - 稀疏视图
  - 相机位姿
  - 生成器-判别器
  - 自适应采样
  - 对比学习
---

# ADen: Adaptive Density Representations for Sparse-view Camera Pose Estimation

**会议**: ECCV 2024  
**arXiv**: [2408.09042](https://arxiv.org/abs/2408.09042)  
**领域**: 3D视觉 / 相机位姿估计  
**关键词**: 稀疏视图, 相机位姿, 生成器-判别器, 自适应采样, 对比学习  

## 一句话总结

提出ADen框架，通过生成器输出多个位姿假设+判别器评分选择最佳的方式，统一了位姿回归和概率估计范式，仅需500个自适应样本即超越需要500K均匀采样的方法，同时实现实时推理。

## 研究背景与动机

- **领域现状**：稀疏视图相机位姿估计有两种主流方法——回归法（单模预测）和概率法（均匀采样SO(3)空间）。
- **现有痛点**：回归法假设单模分布，在对称物体上表现差；概率法如RelPose需500K均匀采样才能获得足够精度，计算代价极高，且维度灾难限制了联合旋转+平移的建模。
- **核心矛盾**：精度需要密集采样 vs 采样效率决定运行时间，均匀网格在高维空间不可行。
- **本文目标**：如何用极少量样本实现高精度、多模态感知的位姿估计。
- **切入角度**：现实中位姿分布高度偏斜，仅有少数模态占主导，自适应采样远优于均匀采样。
- **核心 idea**：生成器学习从条件分布中采样少量高质量假设，判别器对比学习选出最佳假设。

## 方法详解

### 整体框架

ResNet提取逐图像特征→Transformer融合多视图特征→共享backbone分出Pose Generator（多假设生成）和Pose Discriminator（对比排序）两个head。

### 关键设计

**设计1：多假设位姿生成器**
- **功能**：生成M个位姿假设，每个为四元数+平移的7维向量。
- **核心思路**：M个可学习query嵌入通过MLP映射，结合融合特征生成M个假设。仅回归最接近GT的假设（geodesic距离最小），其余不施加损失。
- **设计动机**：避免mode collapse——不将所有假设都回归到同一GT，让模型自由探索多个模态。

**设计2：对比判别器**
- **功能**：评估每个生成假设的正确概率。
- **核心思路**：训练时将GT位姿作为正样本加入，用对比负对数似然损失训练判别器区分GT和生成假设。推理时不使用GT，选概率最高的假设。
- **设计动机**：将位姿选择转化为对比学习问题，避免了均匀采样的维度灾难。

**设计3：联合训练稳定策略**
- **功能**：在query嵌入中注入高斯噪声。
- **核心思路**：类似GAN的稳定训练技巧，防止生成器太好导致判别器梯度消失。
- **设计动机**：生成器-判别器框架的经典训练不稳定问题。

### 损失函数/训练策略

$\mathcal{L} = \mathcal{L}_g + \mathcal{L}_d$。生成器损失：最近假设的geodesic旋转距离 + L2平移距离。判别器损失：对比负对数似然。训练2000 epoch，Adam lr=1e-4。

## 实验关键数据

### 主实验

**CO3D数据集旋转精度（Acc@15°）**

| 方法 | 2-view(seen) | 5-view(seen) | 8-view(seen) |
|------|-------------|-------------|-------------|
| RelPose++ | 81.8 | 84.7 | 85.5 |
| PoseDiff | 76.0 | 77.7 | 78.5 |
| **ADen** | **84.3** | **86.5** | **87.3** |

**紧阈值精度（Acc@5°）**

| 方法 | Seen | Unseen |
|------|------|--------|
| RelPose++ | 39.5 | 27.8 |
| **ADen** | **51.2** | **36.5** |

### 消融实验

| 配置 | Acc@15° |
|------|---------|
| 仅回归（无多假设） | 82.1 |
| 仅生成器（无判别器） | 83.5 |
| 完整ADen | **84.3** |
| 去掉query噪声 | 83.0 |

### 关键发现

1. 在紧阈值（5°/10°）下ADen的优势更显著——不受网格分辨率限制。
2. ADen在Objectron和Niantic零样本迁移中也取得SOTA，泛化能力强。
3. 500个样本 > 500K均匀采样，验证了自适应采样的高效性。

## 亮点与洞察

1. 优雅地统一了回归和概率两种范式，取各家之长。
2. 自然扩展到高维空间（联合R+t），不增加样本数。
3. 实时推理速度，比RelPose++快数倍。

## 局限与展望

1. 生成器的多样性依赖可学习query的初始化，可能不够充分。
2. 判别器在高度对称物体上可能仍有困难。
3. 未探索扩散模型作为生成器的可能性。

## 相关工作与启发

- RelPose/RelPose++开创了概率位姿估计范式，ADen通过自适应采样革新了采样方式。
- 核心insight：现实位姿分布是稀疏的，不需要均匀采样整个空间。

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | ★★★★☆ |
| 实用性 | ★★★★☆ |
| 实验充分性 | ★★★★☆ |
| 写作清晰度 | ★★★★★ |

<!-- RELATED:START -->

## 相关论文

- [3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues](upose3d_uncertainty-aware_3d_human_pose_estimation_with_cross-view_and_temporal_.md)
- [Event-based Head Pose Estimation: Benchmark and Method](event-based_head_pose_estimation_benchmark_and_method.md)
- [GS-Pose: Category-Level Object Pose Estimation via Geometric and Semantic Correspondence](gs-pose_category-level_object_pose_estimation_via_geometric_and_semantic_corresp.md)
- [3D Hand Pose Estimation in Everyday Egocentric Images](3d_hand_pose_estimation_in_everyday_egocentric_images.md)

<!-- RELATED:END -->
