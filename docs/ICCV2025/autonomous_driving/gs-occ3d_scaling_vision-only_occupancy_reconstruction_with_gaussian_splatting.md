---
title: >-
  [论文解读] GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting
description: >-
  [ICCV 2025][自动驾驶][occupancy reconstruction] 提出 GS-Occ3D，一个可扩展的纯视觉 occupancy 重建框架，通过 Octree-based Gaussian Surfel 表示和地面/静态/动态三层解耦建模，实现了全 Waymo 数据集的纯视觉 occupancy 标注生成，在下游任务上达到与 LiDAR 标注可比甚至更好的零样本泛化性能。
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "occupancy reconstruction"
  - "Gaussian splatting"
  - "vision-only"
  - "auto-labeling"
  - "3D重建"
---

# GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting

**会议**: ICCV 2025  
**arXiv**: [2507.19451](https://arxiv.org/abs/2507.19451)  
**代码**: [项目主页](https://github.com/baijunye/GS-Occ3D)  
**领域**: 自动驾驶  
**关键词**: occupancy reconstruction, Gaussian splatting, vision-only, auto-labeling, 3D reconstruction

## 一句话总结

提出 GS-Occ3D，一个可扩展的纯视觉 occupancy 重建框架，通过 Octree-based Gaussian Surfel 表示和地面/静态/动态三层解耦建模，实现了全 Waymo 数据集的纯视觉 occupancy 标注生成，在下游任务上达到与 LiDAR 标注可比甚至更好的零样本泛化性能。

## 研究背景与动机

3D Occupancy 预测是自动驾驶感知和规划的关键基础，但现有方案面临严重的**可扩展性瓶颈**：

**LiDAR 标注成本极高**：主流 occupancy 标签（如 Occ3D）依赖 LiDAR 点云，需要昂贵的专业测量车辆，难以利用大量众包数据

**纯视觉重建挑战巨大**：稀疏视角、动态遮挡、弱纹理区域、长距离轨迹导致几何退化

**现有方法不适合大规模标注**：
   - NeRF 方法过平滑或需全体积处理，扩展性差
   - GS 方法优化渲染质量而非几何精度，直接用于 occupancy 重建会产生碎片化几何
   - Mesh 表示需要大量后处理，不适合自动流水线

**动态物体被忽视**：多数先前方法仅处理静态场景，无法建模运动物体的 occupancy

核心目标：**构建一个无需 LiDAR、无需几何先验、可重建完整 Waymo 数据集的纯视觉 occupancy 标注方案。**

## 方法详解

### 整体框架

三阶段流水线：

1. **几何重建**：将场景分解为地面/静态背景/动态物体分别建模
2. **标签生成（Label Curation）**：帧级划分 + 多帧聚合 + 光线投射
3. **下游训练**：用生成的纯视觉标签监督 occupancy 预测模型

### 关键设计

1. **Octree-based Gaussian Surfel**: 利用 SfM 生成的稀疏点云作为初始骨架，构建动态八叉树结构。每个体素生成 $m$ 个 Gaussian Surfel 原语，作为局部表面近似。层级数 $K$ 由相机中心到点云的距离分布自适应决定：

    $K = \lfloor \log_2(d_{max}/d_{min}) \rceil + 1$

   各层级体素中心通过分层量化计算：$\mathbf{V}_L = \{\lfloor \mathbf{P}/(\epsilon/2^L) \rceil \cdot (\epsilon/2^L)\}$

   设计动机：八叉树在保持内存效率的同时，粗层建模全局结构（道路、墙壁），细层捕获高频细节（植被、边界），且训练时可自适应扩展/收缩。使用累积 LOD 而非单一 LOD 来增强跨尺度几何完整性。

2. **地面专项重建（Ground Gaussians）**: 弱纹理地面是纯视觉重建的难点。将相机位姿投影到 xy 平面初始化地面 surfel，z 坐标根据最近相机加固定高度偏移调整，朝向继承相机旋转。设计动机：显式建模地面这一场景主导结构元素，确保大面积一致性，特别是上下坡场景。配合平面正则化损失保持平坦性。

3. **动态物体重建**: 使用基于 RGB 的 3D 目标跟踪初始化每个动态车辆的边界框和位姿 $(\mathbf{R}_t, \mathbf{t}_t)$，在框内采样固定数量点。为缓解初始位姿噪声，引入可学习校正：

    $\mathbf{R}_t' = \mathbf{R}_t \Delta\mathbf{R}_t, \quad \mathbf{t}_t' = \mathbf{t}_t + \Delta\mathbf{t}_t$

4. **标签生成流水线**:

    - **帧级划分**：以相机位姿为中心定义感知范围，均匀采样形成单帧点云
    - **多帧聚合**：对动态物体跨帧聚合点云（变换到 box 坐标系后拼接），解决稀疏性问题
    - **光线投射体素化**：从相机到每个占据体素投射射线，仅第一个被击中的体素标为"已观测"，其余为"未观测"，显式处理遮挡

### 损失函数 / 训练策略

$$L = L_{rgb} + \lambda_{geo}L_{geo} + \lambda_{obj}L_{obj} + \lambda_{road}L_{road} + \lambda_{sky}L_{sky}$$

- $L_{rgb}$：L1 + D-SSIM 重建损失
- $L_{geo} = \lambda_s L_s + \lambda_d L_d + \lambda_n L_n$：surfel 正则化 + 深度畸变 + 深度-法线一致性
- $L_{obj}$：物体不透明度图的熵损失，促进前景/背景清晰解耦
- $L_{road}$：邻近 surfel 高度变化正则
- $L_{sky}$：天空区域渲染不透明度的 BCE 损失

## 实验关键数据

### 几何重建（Waymo Static-32）

| 方法 | CD ↓ | PSNR ↑ | 显存 (GB) | 训练时间 |
|------|------|--------|----------|---------|
| NeuS (w/ LiDAR) | 0.76 | 13.24 | 31 | 5.0h |
| StreetSurf (w/ LiDAR) | 0.90 | 26.85 | 21 | 1.5h |
| 2DGS | 1.23 | 25.60 | 15 | 1.0h |
| GVKF | 0.82 | 25.87 | 24 | 2.0h |
| **GS-Occ3D (纯视觉)** | **0.56** | **26.89** | **10** | **0.8h** |

- 纯视觉方法**超越所有使用 LiDAR 辅助**的方法（NeuS、StreetSurf），CD 仅 **0.56**

### 下游 Occupancy 预测

| 训练标签 | 评估集 | IoU ↑ | F1 ↑ | Prec. ↑ | Rec. ↑ |
|---------|-------|-------|------|---------|--------|
| Ours (Waymo) | Occ3D-Val (Waymo) | 44.7 | 61.8 | 58.2 | 65.9 |
| Occ3D (Waymo) | Occ3D-Val (Waymo) | 57.4 | 73.0 | 62.9 | 87.0 |
| **Ours (Waymo)** | **Occ3D-Val (nuScenes)** | **33.4** | **50.1** | **62.5** | **41.8** |
| Occ3D (Waymo) | Occ3D-Val (nuScenes) | 31.4 | 47.8 | 38.8 | 62.1 |

- Waymo 域内评估略低于 LiDAR 标签（44.7 vs 57.4），属合理范围
- **零样本 nuScenes 泛化超越 LiDAR 标签** (33.4 vs 31.4 IoU)，精度更高（62.5 vs 38.8）

### 消融实验

| 配置 | CD ↓ | PSNR ↑ | 说明 |
|------|------|--------|------|
| 5cam 输入 (本文) | 0.56 | 26.89 | 最优 |
| 3cam 输入 (本文) | 0.66 | 26.96 | 仍然领先其他方法 |
| 5cam GVKF | 0.82 | 25.87 | 对照 |
| 无 Ground Gaussians | - | - | 出现孔洞和异常凸起 |

### 关键发现

- 点云直接表示优于 mesh 转换：mesh 引入后处理损失（孔洞、天空包围伪影）
- Ground Gaussians 对弱纹理区域至关重要，消除了地面孔洞和畸变
- 纯视觉标签可生成 66 类语义（vs Occ3D 的 16 类），包含摩托车、车道线等 LiDAR 易遗漏的类别
- 5 camera 对本方法帮助更大（0.56 vs 0.66），而其他方法反而因前向多视图歧义退化

## 亮点与洞察

- **范式转换**：从"LiDAR 标注 → 训练模型"到"纯视觉重建 → 自动生成标签 → 训练模型"，降低成本几个数量级
- **重建完整 Waymo 数据集**：首次用纯视觉方法完成，展示真正的工程可扩展性
- **优于 LiDAR 的四个方面**：更广覆盖范围、更强零样本泛化、更廉价丰富语义、恶劣天气潜力

## 局限与展望

- 相机仅提供前方和侧方视角，后方信息缺失不可避免
- 夜间和曝光问题导致有效视觉范围减小
- ego-static 场景（车辆静止）下纯视觉方法失败
- 目前仅提供几何标签，语义+几何联合重建是未来重点

## 相关工作与启发

- Octree-GS 框架在其他大规模场景任务（城市重建、室内建模）中也有应用潜力
- 纯视觉标签在泛化性上的优势提示：训练数据多样性可能比精度更重要
- 地面专项建模思路可扩展到其他主导结构（如天花板、墙面）

## 评分

- 新颖性：⭐⭐⭐⭐ — 纯视觉 occupancy 重建 + 全 Waymo 数据集标注
- 技术深度：⭐⭐⭐⭐ — Octree Surfel + 三层解耦 + 完整标注流水线
- 实验充分度：⭐⭐⭐⭐⭐ — 重建/下游/泛化/消融/与 LiDAR 对比，非常全面
- 实用价值：⭐⭐⭐⭐⭐ — 直接支撑大规模自动驾驶数据标注，工程价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] 6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting](6dopegs_online_6d_object_pose_estimation_using_gaussian_spla.md)
- [\[ICCV 2025\] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)
- [\[AAAI 2026\] Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction](../../AAAI2026/autonomous_driving/visiononly_gaussian_splatting_for_collaborative_semantic_occupancy_p.md)
- [\[ICCV 2025\] GaussRender: Learning 3D Occupancy with Gaussian Rendering](gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)
- [\[ICCV 2025\] Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)

</div>

<!-- RELATED:END -->
