---
title: >-
  [论文解读] FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views
description: >-
  [CVPR 2025][3D视觉][稀疏视角重建] 提出 FLARE，一个前馈可微系统，从未标定的稀疏视角图像（2-8 张）在 0.5 秒内同时推断高质量的相机位姿、3D 几何和外观，通过级联学习范式将相机位姿作为桥梁逐步简化复杂的 3D 学习任务。 领域现状：SfM+MVS 是经典的两阶段重建方案，但在稀疏视角下特征匹配困…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "稀疏视角重建"
  - "相机估计"
  - "3D高斯"
  - "前馈"
  - "点图"
---

# FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views

**会议**: CVPR 2025  
**arXiv**: [2502.12138](https://arxiv.org/abs/2502.12138)  
**代码**: [https://zhanghe3z.github.io/FLARE/](https://zhanghe3z.github.io/FLARE/)  
**领域**: 3D视觉 / 稀疏视角重建  
**关键词**: 稀疏视角重建, 相机估计, 3D高斯, 前馈, 点图

## 一句话总结

提出 FLARE，一个前馈可微系统，从未标定的稀疏视角图像（2-8 张）在 0.5 秒内同时推断高质量的相机位姿、3D 几何和外观，通过级联学习范式将相机位姿作为桥梁逐步简化复杂的 3D 学习任务。

## 研究背景与动机

**领域现状**：SfM+MVS 是经典的两阶段重建方案，但在稀疏视角下特征匹配困难；DUSt3R/MASt3R 预测点图但依赖后优化全局配准。

**现有痛点**：DUSt3R 仅支持两两配对+全局优化，耗时且结果次优；PF-LRM 的三平面表示限制了大场景性能；现有方法无法同时高效地解决相机估计、几何重建和外观建模。

**核心矛盾**：直接从图像联合优化位姿+几何+外观极易陷入局部最优。

**本文目标**：设计级联学习范式，用相机位姿作为中间代理逐步降低学习难度。

**核心 idea**：先估计粗糙位姿 → 指导相机坐标系下的局部几何 → 投影到全局坐标系 → 生成 3D 高斯做渲染。

## 方法详解

### 整体框架

四步级联：(1) Neural Pose Predictor 估计粗糙位姿；(2) Camera-centric 几何估计在各相机坐标系下预测局部点图；(3) Global Geometry Projector 将局部点图统一到全局坐标系；(4) 3D Gaussian 回归头用于新视角合成。

### 关键设计

1. **神经位姿预测器**:

    - 功能：从稀疏视角图像直接回归相机位姿
    - 核心思路：将图像 patch 和可学习相机 latent 拼接成 1D 序列，送入小型 decoder-only transformer 预测 7D 位姿（平移+归一化四元数）
    - 设计动机：跳过特征匹配，直接回归位姿；即使不完美的位姿也提供有价值的空间初始化

2. **两阶段几何估计**:

    - 功能：从局部到全局渐进式学习几何
    - 核心思路：先在各相机坐标系下预测局部点图（与成像过程一致，简化学习），再用 learnable geometry projector 将局部点图转换到全局坐标系。训练时加噪声扰动位姿增强鲁棒性
    - 设计动机：局部预测避免了直接推理复杂的全局空间关系，分解了学习难度

3. **3D 高斯外观建模**:

    - 功能：从估计几何生成可渲染的 3D 高斯
    - 核心思路：用全局点图作为高斯中心，VGG 特征和外观特征融合后预测 opacity、rotation、scale 和球谐系数。对尺度不一致问题，预测和 GT 点图都归一化到单位空间
    - 设计动机：将几何和外观解耦，几何作为 3D 高斯的几何支撑

### 损失函数 / 训练策略

总损失 = 位姿损失（Huber）+ 几何损失（confidence-aware L2）+ 高斯渲染损失（L2 + VGG perceptual + depth）。在大规模公开数据集混合训练。

## 实验关键数据

### 主实验

在 RealEstate10K 和多个数据集上：
- 位姿估计：超越 DUSt3R、MASt3R
- 新视角合成：超越现有 pose-free 方法
- 推理速度：< 0.5 秒

### 关键发现

- 级联学习比直接联合学习显著更好
- 两阶段几何（局部→全局）比直接全局预测收敛更快
- 噪声位姿增强对推理时鲁棒性至关重要

## 亮点与洞察

- 级联学习范式的核心洞察：位姿作为 2D→3D 的桥梁降低学习复杂度
- 0.5 秒的推理速度比优化方法快几个数量级
- 可处理任意数量的输入图像

## 局限与展望

- GPU 显存限制了同时处理的图像数量
- 对相机位姿估计质量仍有一定依赖
- 在纹理缺失区域表现可能下降

## 评分

- 新颖性：8/10 — 级联学习范式设计合理
- 技术深度：8/10 — 多任务联合学习框架完整
- 实验充分度：8/10 — 多数据集验证
- 写作质量：8/10 — 结构清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views](matcha_gaussians_atlas_of_charts_for_high-quality_geometry_and_photorealism_from.md)
- [\[CVPR 2025\] Zero-Shot Monocular Scene Flow Estimation in the Wild](zero-shot_monocular_scene_flow_estimation_in_the_wild.md)
- [\[CVPR 2025\] SPARS3R: Semantic Prior Alignment and Regularization for Sparse 3D Reconstruction](spars3r_semantic_prior_alignment_and_regularization_for_sparse_3d_reconstruction.md)
- [\[CVPR 2025\] Light3R-SfM: Towards Feed-forward Structure-from-Motion](light3r-sfm_towards_feed-forward_structure-from-motion.md)
- [\[CVPR 2025\] Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](depth_any_camera_zero-shot_metric_depth_estimation_from_any_camera.md)

</div>

<!-- RELATED:END -->
