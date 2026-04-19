---
title: >-
  [论文解读] Leveraging 2D Priors and SDF Guidance for Dynamic Urban Scene Rendering
description: >-
  [ICCV 2025][自动驾驶][动态场景渲染] 提出 UGSDF 方法，将 SDF 网络与 3D Gaussian Splatting 联合学习来建模动态城市场景中的物体，仅使用 2D 先验（深度网络+点跟踪器）即可实现 SOTA 渲染效果，无需 LiDAR 数据、3D 运动标注或人体模板。
tags:
  - ICCV 2025
  - 自动驾驶
  - 动态场景渲染
  - 3D Gaussian Splatting
  - SDF
  - 2D先验
  - 新视角合成
---

# Leveraging 2D Priors and SDF Guidance for Dynamic Urban Scene Rendering

**会议**: ICCV 2025  
**arXiv**: [2510.13381](https://arxiv.org/abs/2510.13381)  
**代码**: [GitHub](https://dynamic-ugsdf.github.io/)  
**领域**: 自动驾驶  
**关键词**: 动态场景渲染, 3D Gaussian Splatting, SDF, 2D先验, 新视角合成

## 一句话总结

提出 UGSDF 方法，将 SDF 网络与 3D Gaussian Splatting 联合学习来建模动态城市场景中的物体，仅使用 2D 先验（深度网络+点跟踪器）即可实现 SOTA 渲染效果，无需 LiDAR 数据、3D 运动标注或人体模板。

## 研究背景与动机

动态城市场景的重建和渲染是自动驾驶仿真的核心需求，对 3D 检测、运动规划和安全场景仿真至关重要。当前方法存在以下局限：

**过度依赖 3D 标注**：场景图方法（如 OmniRe）需要 3D bounding box tracklets、LiDAR 数据甚至 SMPL 人体模板，获取成本高昂

**NeRF vs 3DGS vs SDF 各有短板**：NeRF/3DGS 视觉质量好但几何精度有限，SDF 几何精确但需要密集表示才能达到相当的视觉保真度

**动态分解方法受限**：场景分解方法（如 S3Gaussians）只用单一动态场建模所有物体，缺乏精细控制

作者的核心洞察是：2D 目标无关先验（深度估计+点跟踪）结合 SDF+3DGS 的双重表示，可以在不需要任何 3D 标注的情况下，实现甚至超越使用 3D 标注方法的渲染质量。

## 方法详解

### 整体框架

UGSDF 以 RGB 图像序列为输入，配合相机参数和动态物体 mask，可选 LiDAR 输入。核心流程分四步：
1. **规范化构建**：利用 UniDepth 深度网络和 CoTracker 点跟踪器将动态物体的多帧点云对齐到规范坐标系
2. **运动建模**：通过可学习基轨迹线性组合表示 Gaussian 运动
3. **SDF+3DGS 联合表示**：SDF 控制几何精度，3DGS 负责高保真渲染
4. **双向引导优化**：SDF 引导 Gaussian 的增密/剪枝，Gaussian 引导 SDF 的射线采样

### 关键设计

1. **基于 2D 先验的动态物体建模**：使用 SAM2 生成逐帧分割 mask，CoTracker 跟踪像素轨迹，UniDepth 估计度量深度。通过反投影和跨帧 warp 构建规范帧点云，完全避免对 3D tracklet 或 SMPL 模板的依赖。核心公式为点云反投影：

$$\mathbf{x}_i = \mathbf{D}_t^c(\boldsymbol{p}) \times (\mathbf{K}^c)^{-1} \tilde{\boldsymbol{p}}_i$$

2. **SDF 变形网络**：建模动态物体的几何。网络包含三个子模块：变形网络 $\varphi_{def}$ 将观测点映射到规范空间，拓扑感知网络 $\varphi_{hyp}$ 处理拓扑变化（如行人运动），多分辨率特征网格 $\mathcal{V}$ 编码几何细节。SDF 值通过 $S_i = \varphi_{sdf}(\mathbf{v}_i, \mathbf{w}_{i,t})$ 获得。引入可学习变形码 $\mathbf{z}_t$ 使网络适应不同时间步的形变。

3. **SDF-Gaussian 双向引导**：这是方法的核心创新。

    - **SDF → Gaussian 增密**：将物体周围空间划分为 $N^3$ 立方网格，查询每个网格中心的 SDF 值。SDF 值低于阈值 $\tau_s$ 表明靠近表面，若该网格内 Gaussian 数量不足 $\tau_n$，则从深度图反投影采样新 Gaussian 进行增密。
    - **SDF → Gaussian 剪枝**：通过多时间步 SDF 值累积判断 Gaussian 是否远离表面：

$$\sum_{t} \exp\left(\frac{-S_i(t) + \sum_{j \in \text{NN}(i)} S_j(t)}{\gamma}\right) < \tau_{pr}$$

    - **Gaussian → SDF 射线采样**：利用 Gaussian 渲染的深度图 $\hat{\mathbf{D}}_t$ 缩小 SDF 射线采样范围，提升表面重建精度。

4. **Gaussian 运动表示**：采用可学习基轨迹的线性组合建模运动，位置和旋转分别由共享运动系数加权：

$$\boldsymbol{\mu}(t) = \boldsymbol{\mu}_o + \sum_{j=1}^B \mathbf{c}_j(t) \mathbf{b}_j^\mu(t)$$

并施加稀疏性惩罚，促使仅用少数基轨迹泛化。

### 损失函数 / 训练策略

SDF 损失包含 RGB 渲染损失 $\mathcal{L}_{rgb}$、深度损失 $\mathcal{L}_d$、自由空间损失 $\mathcal{L}_{fs}$、Eikonal 正则化 $\mathcal{L}_{eik}$ 和平滑损失 $\mathcal{L}_{sm}$。3DGS 损失包含逐帧的颜色、深度、mask L2 损失，以及基于 2D 跟踪和深度的运动约束。两个表示交替训练，联合迭代优化。

## 实验关键数据

### 主实验

**Waymo Open Dataset (NOTR 分割)**

| 方法 | 输入 | 重建 PSNR↑ | 重建 SSIM↑ | NVS PSNR↑ | NVS SSIM↑ | NVS LPIPS↓ |
|------|------|-----------|-----------|----------|----------|-----------|
| StreetGS | M,T | 29.11 | 0.921 | 25.71 | 0.764 | 0.218 |
| OmniRe | T,M,S | 33.79 | 0.942 | 29.35 | 0.780 | 0.186 |
| **UGSDF** | M,PT | **33.98** | **0.944** | **30.63** | **0.871** | **0.129** |
| UGSDF w/o LiDAR | M,PT | 33.88 | 0.942 | 30.32 | 0.871 | 0.145 |

**Waymo 车辆 vs 行人细分**

| 方法 | 行人 PSNR(重建) | 车辆 PSNR(重建) | 行人 PSNR(NVS) | 车辆 PSNR(NVS) |
|------|---------------|---------------|--------------|--------------|
| OmniRe | 28.15 | 28.91 | 24.36 | 27.57 |
| **UGSDF** | 27.89 | **30.34** | **25.48** | **28.68** |

### 消融实验

| 配置 | 行人 PSNR↑ | 车辆 PSNR↑ | 说明 |
|------|-----------|-----------|------|
| Full | 27.89 | 30.34 | 完整模型 |
| w/o SG4GP | 22.47 | 22.27 | 去除 SDF 引导 Gaussian 分布，性能严重下降 |
| w/ Sparse | 24.82 | 25.14 | 使用更稀疏表示，行人退化更明显 |
| w/o GPS4S | 25.82 | 27.83 | 去除 Gaussian 引导 SDF 采样 |

### 关键发现

- 即使不使用 LiDAR，UGSDF 也能超越使用 3D tracklet 和 SMPL 模板的 OmniRe
- SDF 引导 Gaussian 分布（SG4GP）是最关键组件，去除后 PSNR 下降超过 5dB
- 对薄物体（行人、骑行者）保持密集表示尤为重要
- 车辆类别上 UGSDF 大幅领先 OmniRe（+1.4 PSNR），行人略低但差距很小
- 方法还支持场景编辑任务：物体移除、场景分解、场景组合

## 亮点与洞察

- **用 2D 先验替代 3D 标注**：这一思路极具实用价值，大幅降低了数据获取成本，对自动驾驶仿真的可扩展性意义重大
- **SDF+3DGS 互补的双重表示**：创造性地利用 SDF 的几何精度增强 3DGS 的 Gaussian 分布质量，同时用 3DGS 加速 SDF 的训练，形成正向循环
- **对非刚性物体的鲁棒建模**：通过拓扑感知网络和运动基轨迹，能处理骑行者、行人等复杂运动物体
- 首次在动态城市场景中将 SDF 和 3DGS 结合用于单个动态物体建模

## 局限与展望

- 对 CoTracker 的 2D 跟踪质量敏感，跟踪失败会影响运动估计
- SDF 网络的表达能力不及 SMPL 模板，行人类别上仍略逊于 OmniRe
- 偏离训练轨迹较远的新视角合成质量下降（所有方法共有问题）
- 未来可探索引入视频生成模型的先验来增强重建质量

## 相关工作与启发

- 构建于场景图方法（OmniRe）和变形方法（DeformGS）之上，但简化了输入需求
- SDF+3DGS 联合学习的思路可推广到室内场景重建
- 2D 先验替代 3D 标注的范式可能改变自动驾驶数据标注的工作流

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将 SDF 和 3DGS 联合用于动态城市场景，双向引导机制设计巧妙
- **实验充分度**: ⭐⭐⭐⭐ — Waymo 和 KITTI 双数据集验证，消融充分，定性结果丰富
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，图示直观，但部分公式符号较多
- **价值**: ⭐⭐⭐⭐ — 大幅降低动态场景重建的标注需求，对自动驾驶仿真有高实用价值

<!-- RELATED:START -->

## 相关论文

- [SparseLaneSTP: Leveraging Spatio-Temporal Priors with Sparse Transformers for 3D Lane Detection](sparselanestp_leveraging_spatio-temporal_priors_with_sparse_transformers_for_3d_.md)
- [GaussRender: Learning 3D Occupancy with Gaussian Rendering](gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)
- [Extrapolated Urban View Synthesis Benchmark](extrapolated_urban_view_synthesis_benchmark.md)
- [SimWorld-Robotics: Synthesizing Photorealistic and Dynamic Urban Environments for Multimodal Robot Navigation and Collaboration](../../NeurIPS2025/autonomous_driving/simworld-robotics_synthesizing_photorealistic_and_dynamic_urban_environments_for.md)
- [FlowScene: Learning Temporal 3D Semantic Scene Completion via Optical Flow Guidance](../../NeurIPS2025/autonomous_driving/learning_temporal_3d_semantic_scene_completion_via_optical_flow_guidance.md)

<!-- RELATED:END -->
