---
title: >-
  [论文解读] Outdoor Monocular SLAM with Global Scale-Consistent 3D Gaussian Pointmaps
description: >-
  [ICCV 2025][3D视觉][3DGS SLAM] 提出 S3PO-GS，通过将 3DGS 渲染的 pointmap 作为锚点建立尺度自一致的跟踪模块，结合基于 patch 的 pointmap 动态建图机制，在 RGB-only 室外场景中实现了无累积尺度漂移的高精度定位与高保真新视角合成。
tags:
  - ICCV 2025
  - 3D视觉
  - 3DGS SLAM
  - 单目视觉
  - 尺度一致性
  - Pointmap
  - 室外场景
---

# Outdoor Monocular SLAM with Global Scale-Consistent 3D Gaussian Pointmaps

**会议**: ICCV 2025  
**arXiv**: [2507.03737](https://arxiv.org/abs/2507.03737)  
**代码**: [项目页面](https://3dagentworld.github.io/S3PO-GS/)  
**领域**: 3D视觉  
**关键词**: 3DGS SLAM, 单目视觉, 尺度一致性, Pointmap, 室外场景

## 一句话总结

提出 S3PO-GS，通过将 3DGS 渲染的 pointmap 作为锚点建立尺度自一致的跟踪模块，结合基于 patch 的 pointmap 动态建图机制，在 RGB-only 室外场景中实现了无累积尺度漂移的高精度定位与高保真新视角合成。

## 研究背景与动机

3D Gaussian Splatting (3DGS) 由于高保真实时渲染能力已成为 SLAM 领域的热门选择，但在 **纯 RGB 室外场景** 中面临两个核心挑战：

**缺乏几何先验**: 基于可微渲染管线的跟踪方法（如 MonoGS）在复杂室外环境中容易陷入局部最优，收敛困难

**尺度漂移问题**: 引入独立跟踪模块（如 Photo-SLAM、OpenGS-SLAM）虽然补充了几何约束，但需要维护外部模块与 3DGS 地图之间的尺度对齐。在大角度旋转和位移场景中，累积误差会导致严重的尺度漂移

**核心洞察**: 预训练的 pointmap 模型（如 MASt3R）可以提供几何先验，但其尺度与 3DGS 场景不一致。作者提出 **不让预训练模型参与位姿估计本身**，而是仅用它建立像素级 2D-3D 对应关系——3D 坐标完全来自 3DGS 渲染的 pointmap，从根本上避免了尺度对齐问题。

## 方法详解

### 整体框架

S3PO-GS 包含三个核心模块：3DGS 场景表示、自一致 3DGS Pointmap 跟踪、基于 Patch 的 Pointmap 动态建图。系统先用 MASt3R 初始化 3DGS 地图（优化1000步），随后对每个新帧执行跟踪与建图。

### 关键设计

1. **Pointmap 锚定位姿估计 (PAPE)**:

    - 从相邻关键帧的视角渲染 3DGS 深度图 $D_{ak}$，通过相机内参反投影得到 **渲染 pointmap** $X_{ak}^r$
    - 利用预训练模型（MASt3R/DUSt3R）对当前帧与关键帧生成 **预训练 pointmap** $X_{ak}^p, X_n^p$，通过最近邻匹配建立帧间像素对应
    - 传播对应关系链 $X_{ak}^r \leftrightarrow I_{ak} \leftrightarrow I_n$，得到 **2D-3D 对应**
    - 用 RANSAC + PnP 求解相对位姿 $\mathbf{T}_n^{rel}$
    - **关键优势**: 3D 坐标来自 3DGS 渲染，尺度天然与场景一致，预训练模型仅作为 "桥梁" 不参与位姿计算

2. **位姿优化**:

    - 在 PnP 初始位姿基础上，通过 3DGS 可微渲染管线最小化光度损失 $L_{pho} = \|I(\mathcal{G}, T) - \bar{I}\|_1$
    - 将位姿 $T \in SE(3)$ 线性化至李代数 $\mathfrak{se}(3)$，显式计算 CUDA 管线内的梯度
    - 仅需 **5 次迭代** 即可达到传统方法 100 次迭代的精度

3. **Patch 级尺度对齐**:

    - 将渲染 pointmap $X^r$ 和预训练 pointmap $X^p$ 切分为 $P \times P$ 的 patch
    - 选择统计分布相似的 patch（均值/标准差差异小于阈值）进行归一化
    - 在归一化空间中找 "正确点" 集合 $CP$，计算缩放因子: $\sigma' = \frac{\mu(X^r[CP])}{\mu(X^p[CP])}$
    - 迭代对齐直至缩放因子稳定，得到 $\hat{X}^p = \sigma \times X^p$
    - 若正确点不足，利用已对齐的相邻关键帧 pointmap 计算补救缩放因子

4. **Pointmap 替换机制**:

    - 在关键帧插入新高斯时，用对齐后的预训练 pointmap $\hat{X}^p$ 检测渲染 pointmap $X^r$ 中的 "错误点"
    - 偏差超过阈值时替换: $\hat{X}^r(x) = \hat{X}^p(x)$ if $|X^r(x) - \hat{X}^p(x)| > \epsilon_m \times \hat{X}^p(x)$
    - 随机稀疏下采样控制高斯数量

### 损失函数 / 训练策略

总损失包含三项联合优化关键帧窗口内的位姿和高斯地图：

$$\min_{T_k, \mathcal{G}} \sum_{k \in \mathcal{W}} \alpha L_{pho}^k + (1-\alpha) L_{geo}^k + \lambda_{iso} L_{iso}$$

- **光度损失** $L_{pho}$: L1 渲染重建损失
- **几何损失** $L_{geo} = \|X^r - \hat{X}^p\|_1$: pointmap 对齐监督
- **各向同性正则** $L_{iso}$: 防止高斯椭球过度拉伸

## 实验关键数据

### 主实验

在 Waymo、KITTI、DL3DV 三个室外数据集上全面评估：

| 数据集 | 指标 | S3PO-GS (本文) | OpenGS-SLAM | MonoGS | GlORIE-SLAM |
|--------|------|----------------|-------------|--------|-------------|
| Waymo | ATE↓ | **0.622** | 0.839 | 8.529 | 0.589 |
| Waymo | PSNR↑ | **26.73** | 23.99 | 21.80 | 18.83 |
| KITTI | ATE↓ | **1.048** | 3.224 | 9.493 | 1.134 |
| KITTI | PSNR↑ | **20.03** | 15.61 | 14.78 | 15.49 |
| DL3DV | ATE↓ | **0.032** | 0.141 | 0.274 | 0.492 |
| DL3DV | PSNR↑ | **29.97** | 24.75 | 24.99 | 16.20 |

NVS 在所有数据集上 PSNR 提升 +2.73/+4.42/+4.98；跟踪在 KITTI、DL3DV 上取得最佳，Waymo 与 GlORIE-SLAM 可比。

### 消融实验

| 配置 | ATE↓ | PSNR↑ | 说明 |
|------|------|-------|------|
| w/o 位姿优化 | 1.79 | 24.45 | 无精细化导致跟踪退化 |
| w/o 尺度对齐 | 3.50 | 23.49 | 未对齐的 pointmap 引入错误监督 |
| w/o 点替换 | 1.35 | 25.59 | 错误高斯插入影响建图质量 |
| w/o $L_{geo}$ | 3.73 | 25.70 | 缺乏几何监督导致位移感知下降 |
| **完整模型** | **0.62** | **26.73** | - |

收敛迭代消融（Waymo 405841 场景 ATE）：

| 迭代数 | MonoGS | OpenGS-SLAM | S3PO-GS |
|--------|--------|-------------|---------|
| 5 | 12.6 | 4.17 | **0.55** |
| 50 | 2.98 | 0.85 | **0.49** |
| 100 | 1.70 | 0.80 | **0.46** |

### 关键发现

- PAPE 模块使位姿估计仅需 5 次迭代即可收敛（MonoGS 需 50+ 次）
- 直接将预训练 pointmap 加入 MonoGS（无本文处理）会因尺度漂移导致严重几何模糊
- Patch 级本地对齐比全局对齐更稳健，避免了异常值污染

## 亮点与洞察

- **设计精巧**: 让预训练模型仅做 "牵线搭桥"（建立对应关系），不直接参与位姿计算，从架构层面避免了尺度对齐问题
- **即插即用**: 该思路可推广到其他需要融合外部几何先验的 3DGS SLAM 系统
- **极低迭代收敛**: 5 次迭代即达稳定精度，对实时系统非常友好

## 局限与展望

- 未引入回环检测（Loop Closure），长序列漂移仍可能累积
- 依赖预训练 pointmap 模型的推理速度，可能限制实时性
- 仅在静态场景评估，未处理动态物体

## 相关工作与启发

- **MASt3R/DUSt3R**: 提供强大的预训练 pointmap 先验，但直接用于 SLAM 存在尺度问题
- **MonoGS**: 3DGS SLAM 先驱，但室外场景收敛困难
- **GlORIE-SLAM**: 基于帧间关系的跟踪精度高，但不支持 NVS

## 评分

- 新颖性: ⭐⭐⭐⭐ (尺度自一致的设计思路新颖)
- 技术深度: ⭐⭐⭐⭐ (完整的跟踪-建图管线)
- 实验充分度: ⭐⭐⭐⭐ (三数据集+充分消融)
- 实用价值: ⭐⭐⭐⭐ (直接面向自动驾驶等室外场景)

<!-- RELATED:START -->

## 相关论文

- [Benchmarking Egocentric Visual-Inertial SLAM at City Scale](benchmarking_egocentric_visualinertial_slam_at_city_scale.md)
- [4D Gaussian Splatting SLAM](4d_gaussian_splatting_slam.md)
- [Global-Aware Monocular Semantic Scene Completion with State Space Models](global-aware_monocular_semantic_scene_completion_with_state_space_models.md)
- [MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM](../../CVPR2025/3d_vision/magic-slam_multi-agent_gaussian_globally_consistent_slam.md)
- [VTGaussian-SLAM: RGBD SLAM for Large Scale Scenes with Splatting View-Tied 3D Gaussians](../../ICML2025/3d_vision/vtgaussian-slam_rgbd_slam_for_large_scale_scenes_with_splatting_view-tied_3d_gau.md)

<!-- RELATED:END -->
