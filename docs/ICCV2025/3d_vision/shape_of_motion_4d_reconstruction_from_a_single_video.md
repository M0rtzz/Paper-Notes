---
title: >-
  [论文解读] Shape of Motion: 4D Reconstruction from a Single Video
description: >-
  [ICCV 2025][3D视觉][4D重建] 提出基于 $\mathbb{SE}(3)$ 运动基的动态 3D 高斯表示，从单目视频中恢复全局一致的 3D 运动轨迹，同时实现实时新视角合成和长程 3D 跟踪，在 iPhone 和 Kubric 数据集上全面超越先前方法。
tags:
  - ICCV 2025
  - 3D视觉
  - 4D重建
  - 动态场景
  - 3D高斯
  - SE(3)运动基
  - 单目视频
---

# Shape of Motion: 4D Reconstruction from a Single Video

**会议**: ICCV 2025  
**arXiv**: [2407.13764](https://arxiv.org/abs/2407.13764)  
**代码**: [项目页面](https://shape-of-motion.github.io/)  
**领域**: 3D视觉  
**关键词**: 4D重建, 动态场景, 3D高斯, SE(3)运动基, 单目视频

## 一句话总结

提出基于 $\mathbb{SE}(3)$ 运动基的动态 3D 高斯表示，从单目视频中恢复全局一致的 3D 运动轨迹，同时实现实时新视角合成和长程 3D 跟踪，在 iPhone 和 Kubric 数据集上全面超越先前方法。

## 研究背景与动机

从单目视频中重建动态 3D 场景是一个长期存在的视觉难题，极度欠约束——同一时刻运动主体只能从单一视角观察。

现有方法的局限：

**多视图方法**: 需要同步多相机或 LiDAR 传感器，限制了实际应用

**短程场景流**: 仅计算连续帧间的 3D 运动（如 DynIBaR），无法捕捉跨视频的长程 3D 轨迹

**变形场方法**: 如 HyperNeRF、Deformable-3DGS，通过规范空间到视图空间的映射建模运动，但在大运动场景中表现不佳

**帧空间3D跟踪**: 如 DELTA、SpatialTracker，预测的运动在帧坐标系中，混杂了物体和相机运动

**两个核心洞察**:
- **运动的低维结构**: 图像空间的动态可能复杂且不连续，但底层 3D 运动是连续简单刚性运动的组合
- **数据驱动先验的互补性**: 单目深度估计和长程 2D 跟踪提供互补但含噪的线索，可以融合为全局一致的表示

## 方法详解

### 整体框架

输入 RGB 视频 + 相机参数 + 单目深度图 + 2D 长程跟踪点，优化动态 3D 高斯场景表示。场景分为 **静态高斯**（标准 3DGS）和 **动态高斯**（带运动轨迹），联合优化和渲染。

### 关键设计

1. **$\mathbb{SE}(3)$ 运动基表示**:

    - 定义 $B \ll N$ 个全局共享的基轨迹 $\{\mathbf{T}_{0 \to t}^{(b)}\}_{b=1}^B$（实验中 $B=10$）
    - 每个高斯的位姿变换通过加权组合表达:
    $\mathbf{T}_{0 \to t} = \sum_{b=0}^{B} \mathbf{w}^{(b)} \mathbf{T}_{0 \to t}^{(b)}, \quad \|\mathbf{w}^{(b)}\|=1$
    - 运动用 6D 旋转 + 平移参数化，旋转和平移分别用相同权重加权组合
    - **设计动机**: 紧凑的运动基显式正则化轨迹为低维结构，鼓励运动相似的高斯共享相似系数，等价于场景的 **软刚性运动分解**

2. **轨迹光栅化**:

    - 通过 alpha 合成渲染每个像素在时间 $t'$ 的世界坐标 3D 位置:
    ${}^{w}\hat{\mathbf{X}}_{t \to t'}(\mathbf{p}) = \sum_{i \in H(\mathbf{p})} T_i \alpha_i \boldsymbol{\mu}_{i,t'}$
    - 投影到 2D 即得 2D 对应关系，取第三分量得重投影深度
    - 实现了从任意查询帧的任意像素到任意目标帧的完整 3D 轨迹

3. **初始化策略**:

    - 选择最多 3D 跟踪点可见的帧作为规范帧 $t_0$
    - 对含噪 3D 跟踪的速度向量做 k-means 聚类，从 $B$ 个聚类初始化运动基
    - 每个聚类内用加权 Procrustes 对齐初始化基变换
    - 先优化 1000 步拟合 3D 跟踪观测，再进入主训练

4. **数据驱动先验融合**:

    - **深度先验**: Depth Anything 估计相对深度，对齐到度量尺度后作为监督
    - **2D 跟踪先验**: TAPIR 提供前景长程 2D 跟踪
    - **运动分割**: Track-Anything 用少量点击生成前景 mask
    - **相机位姿**: MegaSaM 估计初始相机参数，训练中联合优化

### 损失函数 / 训练策略

**重建损失（逐帧）**:
$$L_{recon} = \|\hat{\mathbf{I}} - \mathbf{I}\|_1 + \lambda_{depth}\|\hat{\mathbf{D}} - \mathbf{D}\|_1 + \lambda_{mask}\|\hat{\mathbf{M}} - \mathbf{M}\|_1$$

**对应损失（跨帧）**:
- 2D 跟踪损失: $L_{track-2d} = \|\mathbf{U}_{t \to t'} - \hat{\mathbf{U}}_{t \to t'}\|_1$
- 跟踪深度损失: $L_{track-depth} = \|\hat{\mathbf{d}}_{t \to t'} - \hat{\mathbf{D}}(\mathbf{U}_{t \to t'})\|_1$

**物理先验**:
$$L_{rigidity} = \|dist(\hat{\mathbf{X}}_t, \mathcal{C}_k(\hat{\mathbf{X}}_t)) - dist(\hat{\mathbf{X}}_{t'}, \mathcal{C}_k(\hat{\mathbf{X}}_{t'}))\|_2^2$$
距离保持损失，约束 k 近邻间距跨时间稳定，增强刚性运动先验。

训练 500 epochs，Adam 优化器，初始化 40k 动态 + 100k 静态高斯，300 帧视频约 2 小时（A100），渲染 ~140 fps。

## 实验关键数据

### 主实验

iPhone 数据集全任务评估（14 序列）:

| 方法 | 3D EPE↓ | $\delta_{3D}^{.05}$↑ | AJ↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|---------|----------------------|-----|-------|-------|--------|
| HyperNeRF | 0.182 | 28.4 | 10.1 | 15.99 | 0.59 | 0.51 |
| D-3DGS | 0.151 | 33.4 | 14.0 | 11.92 | 0.49 | 0.66 |
| TAPIR+DA | 0.114 | 38.1 | 27.8 | - | - | - |
| SpatialTracker | 0.125 | 37.7 | 24.9 | - | - | - |
| **Ours** | **0.082** | **43.0** | **34.4** | **16.72** | **0.63** | **0.45** |

- **唯一同时在 3D 跟踪、2D 跟踪、新视角合成三个任务上达到 SOTA 的方法**
- 3D EPE 比最强跟踪基线 TAPIR+DA 降低 28%，AJ 提升 +6.6
- NVS 在 PSNR 上比所有动态重建基线领先

Kubric 数据集 3D 跟踪:

| 方法 | EPE↓ | $\delta_{3D}^{.05}$↑ | $\delta_{3D}^{.10}$↑ |
|------|------|----------------------|----------------------|
| CoTracker+DA | 0.19 | 34.4 | 56.5 |
| TAPIR+DA | 0.20 | 34.0 | 56.2 |
| **Ours** | **0.16** | **39.8** | **62.2** |

### 消融实验

关键组件消融（iPhone 数据集 3D 跟踪）:

| 配置 | EPE↓ | $\delta_{3D}^{.05}$↑ | 说明 |
|------|------|----------------------|------|
| 无运动基（独立轨迹） | ~0.12 | ~35 | 低维正则化缺失 |
| 无 rigidity 损失 | 退化 | 退化 | 物理先验重要 |
| 无深度监督 | 退化 | 退化 | 单目约束不足 |
| **完整模型** | **0.082** | **43.0** | - |

### 关键发现

- 运动基的 PCA 可视化与刚性运动组高度相关（如旋转的叶片显示为一致颜色）
- 从含噪 2D 跟踪+深度 "提升" 为 3D 跟踪的性能远不如本方法，说明全局融合的重要性
- 即使只有 10 个运动基也足以表达复杂场景运动

## 亮点与洞察

- **统一框架**: 首次在单一表示中同时实现长程 3D 跟踪和高质量 NVS
- **运动的 "形状"**: 3D 轨迹的几何模式本身就传达了丰富的运动语义信息
- **低维运动假设的力量**: $B=10$ 个 SE(3) 基就能表达复杂场景，且自然提供了运动分割能力
- **实时渲染**: 140 fps 的渲染速度使其适用于交互式应用

## 局限与展望

- 需要前景 mask 的人工标注（虽然只需几次点击）
- 初始化的含噪 3D 跟踪依赖 TAPIR 和 Depth Anything 的质量
- 训练时间约 2 小时/序列，难以做到实时
- 运动基数量固定为 10，更复杂的场景可能需要自适应调整

## 相关工作与启发

- **DynMF**: 同样使用运动基，但用神经网络参数化，本文用显式 SE(3) 更可解释
- **TAPIR/CoTracker**: 强大的 2D 跟踪先验，但缺乏 3D 理解
- **Depth Anything**: 单目深度先验的关键来源
- **MegaSaM**: 从单目视频估计相机参数

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (SE(3) 运动基 + 多先验融合)
- 技术深度: ⭐⭐⭐⭐⭐ (完整的初始化-优化管线，多任务统一)
- 实验充分度: ⭐⭐⭐⭐⭐ (三数据集三任务全面评估)
- 实用价值: ⭐⭐⭐⭐ (实时渲染，但训练较慢)

<!-- RELATED:START -->

## 相关论文

- [Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)
- [Geo4D: Leveraging Video Generators for Geometric 4D Scene Reconstruction](geo4d_leveraging_video_generators_for_geometric_4d_scene_reconstruction.md)
- [4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](../../CVPR2025/3d_vision/4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)
- [Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image](image_as_an_imu_estimating_camera_motion_from_a_single_motion-blurred_image.md)
- [MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer](../../CVPR2026/3d_vision/more_motion-aware_feed-forward_4d_reconstruction_transformer.md)

<!-- RELATED:END -->
