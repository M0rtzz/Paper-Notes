---
title: >-
  [论文解读] LiDAR-RT: Gaussian-based Ray Tracing for Dynamic LiDAR Re-Simulation
description: >-
  [CVPR 2025][自动驾驶][LiDAR仿真] 本文提出LiDAR-RT，将3D高斯原语与NVIDIA OptiX硬件加速光线追踪相结合，首次实现动态驾驶场景下实时且物理精确的LiDAR重新仿真，渲染速度达30 FPS，训练仅需2小时，远超NeRF方案的0.2 FPS和15小时。
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "LiDAR仿真"
  - "3D高斯"
  - "光线追踪"
  - "动态场景"
  - "新视角合成"
  - "OptiX"
---

# LiDAR-RT: Gaussian-based Ray Tracing for Dynamic LiDAR Re-Simulation

**会议**: CVPR 2025  
**arXiv**: [2412.15199](https://arxiv.org/abs/2412.15199)  
**代码**: [zju3dv/LiDAR-RT](https://github.com/zju3dv/LiDAR-RT)  
**领域**: 自动驾驶  
**关键词**: LiDAR仿真, 3D高斯, 光线追踪, 动态场景, 新视角合成, OptiX

## 一句话总结

本文提出LiDAR-RT，将3D高斯原语与NVIDIA OptiX硬件加速光线追踪相结合，首次实现动态驾驶场景下实时且物理精确的LiDAR重新仿真，渲染速度达30 FPS，训练仅需2小时，远超NeRF方案的0.2 FPS和15小时。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：LiDAR传感器是自动驾驶中3D感知的核心组件，LiDAR仿真对于扩展训练数据和验证感知算法至关重要。现有方法存在以下局限：

1. **传统仿真器** (CARLA, AirSim)：存在严重的sim-to-real差距，需要大量人工创建虚拟资产
2. **显式重建方法** (LiDARsim, PCGen)：依赖surfel/mesh等显式表示，对几何质量敏感，且仅支持静态场景
3. **NeRF方法** (NFL, LiDAR4D, DyNFL)：虽然渲染质量好，但训练成本极高（15小时+），渲染速度极慢（0.2 FPS），难以处理复杂动态场景

核心动机：能否将3D Gaussian Splatting的高效性与光线追踪的物理精确性相结合，实现实时LiDAR仿真？

## 方法详解

### 整体框架

LiDAR-RT由四个部分组成：
1. **动态场景表示**：将场景分解为静态背景和多个动态物体，各自用高斯原语表示
2. **高斯光线追踪**：基于BVH加速结构和代理几何体的前向渲染
3. **可微渲染**：前向序反向传播策略支持端到端优化
4. **光线丢弃优化**：UNet网络精细化传感器级光线丢弃效果

### 关键设计

**1. 增强的高斯原语**

在标准3DGS参数（位置μ、协方差Σ、不透明度σ）基础上，引入LiDAR物理特性参数：
- **反射强度 ζ**：用SH系数建模视角相关的反射强度
- **光线丢弃概率 β**：通过两个logit值 $(β_{drop}, β_{hit})$ 和softmax函数建模，同样用SH系数表示视角依赖性

动态物体通过场景图处理：在局部坐标系定义高斯参数，通过跟踪的旋转矩阵和平移向量变换到世界坐标系。

**2. 基于代理几何的光线追踪**

- 采用2D高斯盘作为原语形式，用一对共面三角形作为代理几何体
- 相比AABB包围盒，共面三角形更紧凑地包裹高斯原语，减少网格数量
- 采样位置直接等于光线交点，无需近似处理
- 使用NVIDIA OptiX框架进行BVH构建和hardware-accelerated光线发射

**3. 分块渲染策略**

将每条光线分成多个chunk：
- 每个chunk包含固定数量的交点，仅在chunk内部排序
- 对每个交点计算高斯响应和LiDAR属性（ζ, β），通过体渲染公式累积
- 当所有高斯遍历完毕或累积透射率低于阈值时停止

### 损失函数

$$\mathcal{L} = \lambda_d \mathcal{L}_d + \lambda_i \mathcal{L}_i + \lambda_r \mathcal{L}_r + \lambda_{CD} \mathcal{L}_{CD}$$

- $\mathcal{L}_d$：深度L1损失
- $\mathcal{L}_i$：反射强度L1损失
- $\mathcal{L}_r$：光线丢弃BCE损失
- $\mathcal{L}_{CD}$：Chamfer Distance损失，联合监督场景几何

光线丢弃分为场景级（环境因素如反射材料）和传感器级（硬件噪声），后者使用UNet后处理精细化。

## 实验关键数据

### Waymo Open Dataset (64×2650分辨率)


### 主实验

| 方法 | FPS | 存储 | Depth RMSE↓ | Depth MedAE↓ | SSIM↑ | CD↓ | F-score↑ |
|------|-----|------|-------------|--------------|-------|-----|----------|
| LiDAR-NeRF | 0.98 | 1.6GB | 7.726 | 0.052 | 0.682 | 0.182 | 0.918 |
| DyNFL | 0.21 | 14.9GB | 6.979 | 0.039 | 0.708 | 0.118 | 0.779 |
| LiDAR4D | 0.17 | 7.7GB | 6.623 | 0.038 | 0.701 | 0.106 | 0.944 |
| **LiDAR-RT** | **20.1** | **1.37GB** | **6.458** | **0.034** | **0.733** | **0.100** | **0.946** |

### 关键对比

- **速度**：LiDAR-RT (20.1 FPS) vs LiDAR4D (0.17 FPS) — **118倍加速**
- **存储**：1.37 GB vs 14.9 GB (DyNFL) — **10倍压缩**
- **训练**：约2小时 vs 15小时 (LiDAR4D) — **7.5倍加速**
- **渲染质量**：在深度和点云指标上全面领先或持平

### KITTI-360 Dataset

在KITTI-360上同样取得最优深度和点云渲染质量，并且支持灵活的场景编辑（物体移除、添加、传感器配置变更）。

## 亮点与洞察

1. **技术路线创新**：首次将3DGS的高效表示与物理级光线追踪结合用于LiDAR仿真，解决了rasterization无法处理圆柱形range image投影的固有限制
2. **硬件加速的工程实现**：基于OptiX的BVH构建和any-hit程序设计，将GPU RT core的硬件能力充分释放到LiDAR渲染任务
3. **前向序反向传播**：巧妙解决了光线追踪中无法像tile-based rasterizer那样维护全局排序缓冲的问题
4. **实用性强**：支持场景编辑（物体增删、传感器参数变更），可直接服务于仿真数据增强

## 局限与展望

1. 对动态物体的建模依赖准确的跟踪标注框，tracking质量直接影响重建效果
2. 高斯原语的密度控制策略（分裂/剪枝）直接沿用3DGS，未针对LiDAR的稀疏特性优化
3. UNet后处理增加了推理时间，破坏了端到端的优雅性
4. 仅在Waymo和KITTI-360上验证，未评估跨数据集泛化能力

## 相关工作

- **LiDAR仿真**：LiDARsim → PCGen → NFL → LiDAR4D → DyNFL
- **动态场景重建**：3DGS → S3Gaussian → OmniRe → PVG
- **高斯光线追踪**：3DGRT, Gaussian Ray Tracing (GRT) — 但这些仅用于相机传感器
- **LiDAR物理建模**：NFL首先详细建模了LiDAR传感器的物理特性（强度、光线丢弃等）

## 评分

- **新颖性**：4/5 — 首次将高斯+光线追踪应用于LiDAR仿真，技术路线新颖
- **有效性**：5/5 — 速度提升百倍且质量不降，实际价值极大
- **清晰度**：4/5 — 渲染pipeline描述详尽，代理几何体设计图示清晰
- **意义**：5/5 — 实时LiDAR仿真是自动驾驶simulation的关键需求

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)
- [\[AAAI 2026\] LiDAR-GS++: Improving LiDAR Gaussian Reconstruction via Diffusion Priors](../../AAAI2026/autonomous_driving/lidar-gsimproving_lidar_gaussian_reconstruction_via_diffusion_priors.md)
- [\[ICCV 2025\] Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](../../ICCV2025/autonomous_driving/splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)
- [\[CVPR 2025\] Zero-Shot 4D Lidar Panoptic Segmentation](zero-shot_4d_lidar_panoptic_segmentation.md)
- [\[AAAI 2026\] LiDARCrafter: Dynamic 4D World Modeling from LiDAR Sequences](../../AAAI2026/autonomous_driving/lidarcrafter_dynamic_4d_world_modeling_from_lidar_sequences.md)

</div>

<!-- RELATED:END -->
