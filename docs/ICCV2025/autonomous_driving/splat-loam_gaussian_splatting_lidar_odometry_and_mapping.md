---
title: >-
  [论文解读] Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping
description: >-
  [ICCV 2025][自动驾驶][Gaussian Splatting] 首个纯基于 2D Gaussian 原语的 LiDAR 里程计与建图管线，通过球面投影驱动的可微分光栅化器同时实现高精度位姿估计和轻量化场景重建。
tags:
  - ICCV 2025
  - 自动驾驶
  - Gaussian Splatting
  - LiDAR SLAM
  - 里程计
  - 建图
  - 球面投影
---

# Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping

**会议**: ICCV 2025  
**arXiv**: [2503.17491](https://arxiv.org/abs/2503.17491)  
**代码**: [GitHub](https://github.com/rvp-group/Splat-LOAM)  
**领域**: autonomous_driving  
**关键词**: Gaussian Splatting, LiDAR SLAM, 里程计, 建图, 球面投影

## 一句话总结

首个纯基于 2D Gaussian 原语的 LiDAR 里程计与建图管线，通过球面投影驱动的可微分光栅化器同时实现高精度位姿估计和轻量化场景重建。

## 研究背景与动机

LiDAR 传感器可提供精确的几何测量，广泛用于自动驾驶中的自运动估计和环境重建。然而，传统方法面临一个核心矛盾：**精度、内存和运行时间之间的三角困境**。

- **经典方法**（点云叠加、surfel、mesh）虽然可以增量式地建立全局地图，但通常导致极大的点云或需要在精度与内存之间做出妥协。
- **NeRF 类隐式方法**（如 SHINE-Mapping、N³-Mapping、PIN-SLAM）引入了神经隐式 SDF 表示，虽然在精度和存储效率上有所提升，但需要复杂的采样策略来估计 SDF，并且在线执行时面临速度瓶颈。
- **3D Gaussian Splatting (3DGS)** 近年来在视觉 SLAM 中取得了出色成果（如 SplatAM、Gaussian-SLAM），但这些方法依赖相机图像和颜色信息，无法直接适应纯 LiDAR 数据。

本文的核心切入角度是：**LiDAR 天然提供精确的 3D 结构信息，可以天然地初始化 Gaussian 原语**，而 Gaussian Splatting 的高效性和无需空区建模的特点特别适合 LiDAR 场景。关键 idea 是将 2D Gaussian 原语与球面投影模型结合，设计专用的可微分光栅化器，实现从纯 LiDAR 数据驱动的高效 LOAM 管线。

## 方法详解

### 整体框架

系统采用关键帧驱动的局部地图策略。每当满足可见性条件时，初始化新的 Gaussian 局部模型。新帧通过帧-到-模型配准估计位姿，同时对局部模型进行迭代优化。整个流程仅使用 2D Gaussian 原语作为场景表示。

### 关键设计

1. **球面投影模型**：由于 LiDAR 提供 360° 全景输入，不适合针孔相机的投影模型。本文采用球面投影 $\phi(\mathbf{p}) = \mathbf{K}\psi(\mathbf{p})$，将 3D 点映射到方位角和俯仰角构成的图像坐标。相机内参矩阵 $\mathbf{K}$ 根据每帧点云的实际角度范围自适应计算，避免了硬编码 FoV 导致的空白区域。

2. **2D Gaussian 原语定义与光栅化**：每个 2D Gaussian 由透明度 $o$、中心 $\boldsymbol{\mu}$、两个切向量 $\mathbf{t}_\alpha, \mathbf{t}_\beta$ 和缩放 $(s_\alpha, s_\beta)$ 定义。渲染时采用 tile-based α-blending，将图像分为 16×16 tiles，对每个 tile 内按距离排序的 Gaussian 从前往后积分得到距离 $d$、法线 $\mathbf{n}$ 和透明度 $o$。

3. **显式光线-Splat 交点计算**：放弃了原始 3DGS 的局部仿射近似（在球面投影下会产生数值不稳定和畸变），采用显式的光线-Splat 三平面交点求解，通过求解齐次线性方程组获得交点坐标 $(\alpha, \beta)$。

4. **球面 Bounding Box 处理**：球面图像在水平边界 $\{0, W\}$ 处存在坐标奇异性。本文设计了一种先将顶点移到图像中心、计算范围、再反向映射的方法，确保即使 splat 位于传感器后方也能正确计算 bounding box。

5. **帧-到-模型配准**：结合几何和光度两种配准方式。几何配准使用基于 PCA 的 kd-tree 进行点到面 ICP，光度配准在球面投影的距离图上最小化投影范围误差。位姿更新通过 Lie 代数 $\mathfrak{se}(3)$ 局部参数化，使用二阶 Gauss-Newton 法求解。

### 损失函数 / 训练策略

建图的总损失函数为：

$$\mathcal{L}_{\text{map}} = \mathcal{L}_d + \lambda_o \mathcal{L}_o + \lambda_n \mathcal{L}_n + \lambda_s \sum_{i=1}^{N} \mathcal{L}_{s_i}$$

- $\mathcal{L}_d$：深度图的 L1 误差，仅在有效像素上计算，带距离加权
- $\mathcal{L}_o$：透明度覆盖损失，驱动 splat 覆盖有效测量区域  
- $\mathcal{L}_n$：法线自正则化，对齐 splat 法线与渲染深度图梯度估计的表面法线
- $\mathcal{L}_s$：尺度正则化，允许 splat 扩展到阈值 $\tau_s$ 后才进行惩罚，支持各向异性 splat

里程计损失为几何和光度之和：$\mathcal{L}_{\text{odom}} = \mathcal{L}_{\text{geo}} + \mathcal{L}_{\text{photo}}$。

关键帧选择按几何分布采样，保证最新关键帧有≥40% 被选中概率。不执行 opacity reset 以避免灾难性遗忘。

## 实验关键数据

### 主实验

在 Newer College 和 Oxford Spires 数据集上使用 GT 位姿评估建图质量：

| 方法 | Acc↓ | Com↓ | C-l1↓ | F-score↑ |
|------|------|------|-------|----------|
| OpenVDB | 11.45 | 4.38 | 7.92 | 88.85 |
| VoxBlox | 20.36 | 12.64 | 16.50 | 64.63 |
| N³-Mapping | 6.32 | 9.75 | 8.04 | 94.54 |
| PIN-SLAM | 15.28 | 10.50 | 12.89 | 88.05 |
| **Splat-LOAM** | **6.64** | **4.09** | **5.37** | **96.74** |

里程计评估：在 NC、VBR、Oxford Spires、Mai City 四个数据集上，Splat-LOAM 结合几何+光度配准的方案与 MAD-ICP 等 SOTA 方法竞争力相当。

### 消融实验

| Mesh 提取方法 | Acc↓ | Com↓ | C-l1↓ | F-score↑ |
|-----------|------|------|-------|----------|
| Marching Cubes | 16.76 | 5.53 | 11.14 | 76.76 |
| Poisson (centers) | 10.15 | 6.70 | 8.43 | 92.33 |
| **Ours (rendered)** | **6.64** | **4.09** | **5.37** | **96.74** |

配准方法消融：联合使用几何+光度因子的效果优于任何单一方法（Point-to-Point/Point-to-Plane/仅光度），验证了混合配准的设计合理性。

### 关键发现

- 活跃原语数量在 200K-300K 时建图 FPS 趋于稳定（受光栅化器饱和影响）
- 系统对 LiDAR 运动畸变敏感，速度+位姿联合估计是重要的改进方向
- 与 N³-Mapping 相比，Splat-LOAM 在细节保真度上更优，后者趋于过度平滑

## 亮点与洞察

- 首个纯 LiDAR Gaussian Splatting LOAM，填补了该方向的空白
- 球面投影下处理坐标奇异性的 bounding box 计算方案非常巧妙
- 尺度正则化损失 $\mathcal{L}_s$ 的阈值设计允许各向异性 splat，比直接最小化平均偏差更灵活
- GPU 内存需求极低，适合机器人实时感知场景

## 局限与展望

- 对 LiDAR 运动畸变敏感，需要同时估计速度
- 未包含 Loop Closure 模块，长距离轨迹会有累积漂移
- 未利用 LiDAR 的强度/反射率信息用于外观建模
- 球面投影光栅化器的紧凑 bounding box 计算为近似方案，存在改进空间

## 相关工作与启发

- PIN-SLAM 的神经点云方案虽然全局一致性好，但内存和速度不理想
- 视觉 Gaussian SLAM（SplatAM、Splat-SLAM）的思路可以反向适配到 LiDAR
- GS-LiDAR 的周期振动 Gaussian 设计针对动态物体，与本文静态场景建图互补

## 评分

- 新颖性：⭐⭐⭐⭐ 首个纯 LiDAR GS LOAM
- 技术深度：⭐⭐⭐⭐ 球面光栅化器设计完整
- 实验充分度：⭐⭐⭐⭐ 多数据集多指标评估
- 实用价值：⭐⭐⭐⭐ 低 GPU 需求，适合机器人
- 总体推荐：⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting](gs-livm_real-time_photo-realistic_lidar-inertial-visual_mapping_with_gaussian_sp.md)
- [GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting](6dopegs_online_6d_object_pose_estimation_using_gaussian_spla.md)
- [AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)
- [A Constrained Optimization Approach for Gaussian Splatting from Coarsely-posed Images and Noisy Lidar Point Clouds](a_constrained_optimization_approach_for_gaussian_splatting_from_coarsely-posed_i.md)

<!-- RELATED:END -->
