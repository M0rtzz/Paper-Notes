---
title: >-
  [论文解读] Back on Track: Bundle Adjustment for Dynamic Scene Reconstruction
description: >-
  [ICCV 2025][3D视觉][Bundle Adjustment] 提出 BA-Track 框架，利用 3D 点追踪器将观测运动分解为相机运动和物体运动，使传统 Bundle Adjustment 能同时处理静态与动态场景元素，实现精确的相机位姿估计和时间一致的稠密重建。
tags:
  - ICCV 2025
  - 3D视觉
  - Bundle Adjustment
  - Dynamic Scene
  - Motion Decoupling
  - 3D Tracking
  - Depth Refinement
---

# Back on Track: Bundle Adjustment for Dynamic Scene Reconstruction

**会议**: ICCV 2025  
**arXiv**: [2504.14516](https://arxiv.org/abs/2504.14516)  
**代码**: https://wrchen530.github.io/projects/batrack  
**领域**: 3D Vision / Dynamic SLAM  
**关键词**: Bundle Adjustment, Dynamic Scene, Motion Decoupling, 3D Tracking, Depth Refinement

## 一句话总结

提出 BA-Track 框架，利用 3D 点追踪器将观测运动分解为相机运动和物体运动，使传统 Bundle Adjustment 能同时处理静态与动态场景元素，实现精确的相机位姿估计和时间一致的稠密重建。

## 研究背景与动机

传统 SLAM/SfM 系统依赖极线约束，本质假设场景是静态的。当场景包含运动物体时，极线约束被违反，导致系统失效。现有应对策略各有缺陷：

**过滤动态元素**：检测并移除运动物体后再做 BA → 重建不完整，丢失动态物体信息

**独立建模运动**：分别估计相机和物体运动 → 运动估计容易不一致

**单目深度回归**：利用深度先验逐帧重建 → 帧间深度尺度不一致，难以全局对齐

BA-Track 的核心洞察：不是丢弃动态点，而是**推断动态点的相机引起运动分量**。动态点在其局部参考系中变为"伪静态"，极线约束重新适用于所有点。

## 方法详解

### 整体框架

BA-Track 包含三个阶段：
1. **运动解耦 3D 追踪器**（前端）：将观测运动分解为静态分量（相机运动）和动态分量（物体运动）
2. **Bundle Adjustment**（后端）：基于静态分量对所有点（包括动态点）做 BA，恢复相机位姿和稀疏 3D 结构
3. **全局精炼**：利用 BA 的稀疏深度估计对齐稠密单目深度图

### 关键设计

1. **运动解耦 3D 追踪器**

    - 使用双网络架构而非单网络：
        - 追踪器 $\mathcal{T}$（6 层 Transformer）：预测**总运动** $X_{total}$、可见性 $v$ 和静态/动态标签 $m$
        - 动态追踪器 $\mathcal{T}_{dyn}$（3 层 Transformer，更轻量）：预测**动态分量** $X_{dyn}$
    - 运动解耦公式：$X_{static} = X_{total} - m \cdot X_{dyn}$
    - $m$ 作为门控因子：$m=0$（静态点）时静态分量 = 总运动；$m=1$（动态点）时减去物体运动
    - 输入包含 RGB 特征和深度特征（来自单目深度模型 ZoeDepth），增强 3D 推理能力
    - 设计动机：实验发现单网络同时学习视觉追踪和运动模式是次优的，双网络分工更有效

2. **RGB-D Bundle Adjustment**

    - 基于 DPVO 框架扩展为 RGB-D BA
    - 滑动窗口提取查询点的 3D 轨迹，得到局部轨迹张量 $\mathbf{X} \in \mathbb{R}^{L \times N \times S \times 3}$
    - 重投影误差：
    $\arg\min_{\{\mathbf{T}_t\},\{\mathbf{Y}\}} \sum_{|i-j|\leq S} \sum_n W_n^i(j) \|\mathcal{P}_j(\mathbf{x}_n^i, y_n^i) - X_n^t(j)\|_\rho + \alpha \|y_n^i - d(\mathbf{X}_n^i)\|^2$
    - 置信权重 $W_n^i(j) = v_n^i(j) \cdot (1 - m_n^i)$：动态点权重低，主要依靠静态点恢复位姿
    - 用 Gauss-Newton + Schur 分解高效求解

3. **全局深度精炼**

    - BA 仅调整稀疏查询点的深度，其余点未被优化
    - 引入 2D 尺度网格 $\theta_t \in \mathbb{R}^{H_g \times W_g}$（分辨率低于原图），对深度图做逐像素缩放：
    $\hat{D}_t[\mathbf{x}] = \theta_t[\mathbf{x}] \cdot D_t[\mathbf{x}]$
    - 深度一致性损失：对齐稠密深度与稀疏 BA 轨迹
    - 场景刚性损失：静态点之间的 3D 距离在帧间保持不变
    $\mathcal{L}_{rigid} = \sum_{|i-j|<S} \sum_{(a,b) \in N} W_{static}(\|P_a^i(j) - P_b^i(j)\| - \|P_a^i - P_b^i\|)$

### 损失函数 / 训练策略

追踪器训练损失：
$$\mathcal{L}_{total} = \mathcal{L}_{3D} + w_1 \mathcal{L}_{vis} + w_2 \mathcal{L}_{dyn}$$

- $\mathcal{L}_{3D}$：总运动和静态分量的 L1 损失，每次迭代都有指数衰减权重 $\gamma^{K-k}$
- $\mathcal{L}_{vis}$/$\mathcal{L}_{dyn}$：可见性和动态标签的二元交叉熵
- 训练数据：TAP-Vid-Kubric（11000 序列），静态轨迹 GT 由相机位姿反投影生成
- $w_1 = w_2 = 5$，$K = 4$ 次迭代更新，窗口 $S = 12$

## 实验关键数据

### 主实验（相机位姿评估 - ATE）

| 方法 | MPI Sintel | AirDOS Shibuya | Epic Fields |
|------|-----------|---------------|-------------|
| DROID-SLAM | 0.175 | 0.256 | 1.424 |
| DPVO | 0.115 | 0.146 | 0.394 |
| LEAP-VO | 0.089 | 0.031 | 0.486 |
| MonST3R | 0.108 | (0.512) | — |
| **BA-Track** | **0.034** | **0.028** | **0.385** |

在 Sintel 上 ATE 从 0.089 降至 0.034，相对提升 **62%**。

深度评估（Abs Rel ↓）：

| 方法 | MPI Sintel | AirDOS Shibuya | Bonn |
|------|-----------|---------------|------|
| ZoeDepth | 0.467 | 0.571 | 0.087 |
| MonST3R | 0.335 | (0.208) | 0.063 |
| **BA-Track** | **0.408** | **0.299** | **0.084** |

### 消融实验

运动解耦方式对比（Sintel ATE）：

| 设置 | 轨迹类型 | 动态掩码 | ATE |
|------|---------|---------|-----|
| (a) 仅总运动 | Total | 无 | 0.137 |
| (b) 总运动+掩码 | Total | ✓ | 0.047 |
| (c) 直接预测静态 | Static* | 无 | 0.091 |
| (e) 运动解耦 | Total-Dynamic | 无 | 0.065 |
| **(f) 解耦+掩码** | Total-Dynamic | ✓ | **0.034** |

深度精炼消融（Bonn crowd2）：

| $\mathcal{L}_{depth}$ | $\mathcal{L}_{rigid}$ | Abs Rel | $\delta<1.25$ |
|---|---|---------|---------|
| ✗ | ✗ | 0.121 | 89.6% |
| ✓ | ✗ | 0.103 | 94.8% |
| ✗ | ✓ | 0.117 | 88.4% |
| ✓ | ✓ | **0.089** | **95.0%** |

### 关键发现

- 运动解耦使 ATE 从 0.137 降至 0.065（相对降低 53%），加上动态掩码进一步降至 0.034
- 单网络直接预测静态分量（Static*）不如双网络解耦方案
- 两种深度精炼损失互补：深度一致性重要性更大，刚性约束提供额外增益
- MonST3R 在 48GB GPU 上最多处理 90 帧，BA-Track 内存效率更高

## 亮点与洞察

- **理念精妙**：不是"去掉"动态元素，而是"转化"动态元素为伪静态，让经典 BA 重新适用
- **双网络解耦设计**有说服力：通过消融实验充分验证了单网络学习视觉追踪+运动模式的次优性
- **混合方案**的典范：将传统优化（BA）与学习先验（3D 追踪器）有机结合
- 全局精炼用轻量尺度网格而非神经网络，参数少且高效

## 局限与展望

- 深度精炼使用简单的尺度网格，更复杂的变形模型（如神经网络）可能进一步提升
- 依赖单目深度先验的质量（ZoeDepth/UniDepth）
- 追踪器在合成数据（Kubric）上训练，域迁移到真实场景可能有性能损失
- 未探索与 3D Gaussian Splatting 等新型表示的结合

## 相关工作与启发

- 与 DROID-SLAM、DPVO 的关系：BA-Track 扩展了传统 BA 框架以支持动态场景
- 与 MonST3R 的对比：MonST3R 用光流掩码过滤动态区域，BA-Track 利用动态区域信息
- 运动解耦的思想可推广到其他需要处理混合运动的任务（如自动驾驶中的动静分离）
- 验证了"传统优化 + 学习先验"的混合范式在动态场景中的巨大潜力

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 运动解耦使 BA 适用于动态场景，思路新颖且直觉清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 三大数据集、相机位姿+深度+重建全面评估、充分消融
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，技术描述详细
- 价值: ⭐⭐⭐⭐⭐ 对动态场景 SLAM/重建有重要推动作用

<!-- RELATED:START -->

## 相关论文

- [Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)
- [BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curv.md)
- [Dynamic Point Maps: A Versatile Representation for Dynamic 3D Reconstruction](dynamic_point_maps_a_versatile_representation_for_dynamic_3d_reconstruction.md)
- [Scene Coordinate Reconstruction Priors](scene_coordinate_reconstruction_priors.md)

<!-- RELATED:END -->
