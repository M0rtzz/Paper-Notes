---
title: >-
  [论文解读] SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video
description: >-
  [CVPR 2025][3D视觉][动态场景重建] SplineGS 提出了一种基于三次 Hermite 样条的动态 3DGS 框架，通过运动自适应样条（MAS）和运动自适应控制点剪枝（MACP）建模动态高斯的连续轨迹，同时联合优化相机参数，在无需 COLMAP 的情况下实现了 SOTA 动态新视角合成和实时渲染。
tags:
  - CVPR 2025
  - 3D视觉
  - 动态场景重建
  - 3D高斯泼溅
  - 样条曲线
  - 单目视频
  - 无COLMAP
---

# SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video

**会议**: CVPR 2025  
**arXiv**: [2412.09982](https://arxiv.org/abs/2412.09982)  
**代码**: https://kaist-viclab.github.io/splinegs-site/  
**领域**: 3D视觉  
**关键词**: 动态场景重建, 3D高斯泼溅, 样条曲线, 单目视频, 无COLMAP

## 一句话总结

SplineGS 提出了一种基于三次 Hermite 样条的动态 3DGS 框架，通过运动自适应样条（MAS）和运动自适应控制点剪枝（MACP）建模动态高斯的连续轨迹，同时联合优化相机参数，在无需 COLMAP 的情况下实现了 SOTA 动态新视角合成和实时渲染。

## 研究背景与动机

**领域现状**：动态场景的新视角合成是 3D 视觉的核心挑战。现有动态 3DGS 方法通过 MLP（D3DGS，慢）、时空网格（4DGS，分辨率受限）或多项式轨迹（STGS，固定阶数）建模形变，各有不足。

**现有痛点**：MLP 严重拖慢渲染速度；网格方法在动态细节上有天花板；固定阶数的多项式无法适应场景中运动复杂度的差异（静止背景 vs 复杂运动的气球）。此外，大多数方法依赖 COLMAP，而 COLMAP 在动态单目视频上经常失败。

**核心矛盾**：需要一种既能精确建模连续运动轨迹、又能自适应调整运动复杂度、还能保持实时渲染速度的形变表示。

**本文目标**：设计 COLMAP-free 的动态 3DGS 框架，用极少参数精确表示动态高斯的连续轨迹，支持运动复杂度自适应。

**切入角度**：计算机图形学中样条曲线是建模连续曲线的经典工具——用少量控制点就能表示光滑的分段三次曲线，既灵活又高效。

**核心 idea**：用三次 Hermite 样条表示动态 3D 高斯的位置轨迹，控制点数量通过运动自适应剪枝自动确定。

## 方法详解

### 整体框架

SplineGS 将 3D 高斯分为静态集和动态集。动态高斯的位置 $\mu(t) = S(t, \mathbf{P})$ 由三次 Hermite 样条定义。训练分两阶段：(1) Warm-up 只优化相机参数；(2) 主训练联合优化高斯属性、样条控制点和相机参数。

### 关键设计

1. **运动自适应样条（Motion-Adaptive Spline, MAS）**:

    - 功能：为每个动态 3D 高斯建模时间连续的位置轨迹
    - 核心思路：三次 Hermite 样条在相邻控制点间生成 C1 连续的分段三次曲线，切向量通过差分近似。控制点初始化利用 CoTracker 的 2D tracking 和 UniDepth 的 metric depth 将 2D 轨迹反投影到 3D，再用最小二乘拟合
    - 设计动机：与多项式不同，样条是分段定义的，避免了高次多项式的数值不稳定（Runge 现象）。计算只需查找相邻控制点并插值，速度极快（5.63 ns vs MLP 的 149 ns）

2. **运动自适应控制点剪枝（MACP）**:

    - 功能：自动为每个高斯确定最优控制点数量
    - 核心思路：每 100 迭代尝试用少一个控制点的新样条做最小二乘近似，若图像空间投影误差小于阈值则接受剪枝。简单运动用 2-3 个控制点，复杂运动用 6-8 个
    - 设计动机：固定控制点数量对简单运动过多（过拟合降速）或对复杂运动不够（欠拟合降质）

3. **COLMAP-free 相机参数估计**:

    - 功能：从纯单目视频估计相机内外参
    - 核心思路：浅层 MLP 预测每帧旋转/平移，焦距全帧共享。通过光度一致性（静态区域跨帧颜色对齐）和几何一致性（3D 点对齐）联合优化，运动掩码排除动态区域
    - 设计动机：COLMAP 在动态单目视频上经常失败（DAVIS 数据集完全无法工作）

### 损失函数 / 训练策略

Warm-up 损失：光度一致性 + 几何一致性。主训练损失 6 项：RGB L1、深度、motion mask Dice、光度一致性、渲染深度光度一致性、几何一致性。1K warm-up + 20K 主训练迭代。

## 实验关键数据

### 主实验

NVIDIA 数据集新视角合成（COLMAP-free）：

| 方法 | PSNR | LPIPS | FPS |
|------|------|-------|-----|
| RoDynRF | 25.38 | 0.079 | 0.45 |
| MoSca | 26.61 | 0.069 | N/A |
| **SplineGS** | **27.21** | **0.053** | **400** |

新视角+新时间合成（更挑战设定）：

| 方法 | PSNR | LPIPS | tOF |
|------|------|-------|-----|
| DynNeRF | 23.36 | 0.219 | 0.921 |
| RoDynRF | 21.58 | 0.221 | 2.138 |
| **SplineGS** | **25.92** | **0.098** | **0.703** |

### 消融实验

| 形变模型 | PSNR | LPIPS | 延迟 (ns) |
|---------|------|-------|----------|
| MLP | 23.51 | 0.125 | 149.41 |
| Grid | 25.48 | 0.090 | 98.89 |
| Poly (3rd) | 25.14 | 0.111 | 1.80 |
| Poly (10th) | 24.38 | 0.120 | 7.71 |
| Bezier | 27.19 | 0.060 | 8.78 |
| **MAS (ours)** | **27.21** | **0.053** | 5.63 |

| MACP 配置 | PSNR | LPIPS | 延迟 |
|-----------|------|-------|------|
| w/o MACP (Nc=4) | 26.62 | 0.065 | 5.34 |
| w/o MACP (Nc=Nf) | 27.08 | 0.054 | 6.11 |
| **Full (MACP)** | **27.21** | **0.053** | 5.63 |

### 关键发现

- MAS 在所有形变模型中质量最优，速度接近最快的多项式（5.63 vs 1.80 ns），远快于 MLP（149 ns）
- 高次多项式（10th）反而比低次（3rd）差，验证了数值不稳定性
- 新时间合成任务上优势更明显（+2.5 dB），样条的连续性天然支持时间插值
- 去掉光度一致性后 PSNR 降至 17.49，是相机估计的基石

## 亮点与洞察

- **经典图形学工具的回归**：在 MLP 和 grid 主导的动态建模中，回归样条是简洁高效的选择。样条天然具有连续性、局部可控性和高效计算性
- **MACP 的自适应策略**：像"智能压缩"一样自动为每个高斯分配运动复杂度预算。Nc heatmap 直观展示了不同区域的运动复杂度
- **COLMAP-free 的实用价值**：在 DAVIS 等 in-the-wild 视频上 COLMAP 完全失败，SplineGS 能从零恢复相机参数

## 局限与展望

- 仍依赖 CoTracker 和 UniDepth 作为先验
- 只建模了位置轨迹，旋转和尺度是逐时间步的参数
- 极端运动或拓扑变化场景未验证

## 相关工作与启发

- **vs D3DGS**: MLP 形变速度慢质量差（23.02 vs 27.21），SplineGS 证明显式参数化远超隐式方法
- **vs STGS**: 固定阶多项式速度最快但质量有限，样条提供更好的质量-速度权衡
- **vs RoDynRF**: 同为 COLMAP-free 但基于 NeRF 极慢，SplineGS 速度快 890 倍且质量更好

## 评分

- 新颖性: ⭐⭐⭐⭐ 样条轨迹 + MACP 组合巧妙，核心思想相对直观
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集，NVS + NVT 评价，消融全面
- 写作质量: ⭐⭐⭐⭐ 方法清晰，公式严谨，可视化丰富
- 价值: ⭐⭐⭐⭐⭐ COLMAP-free + 实时渲染 + SOTA 质量，实用价值极高

<!-- RELATED:START -->

## 相关论文

- [MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion](mp-sfm_monocular_surface_priors_for_robust_structure-from-motion.md)
- [SAT-HMR: Real-Time Multi-Person 3D Mesh Estimation via Scale-Adaptive Tokens](sat-hmr_real-time_multi-person_3d_mesh_estimation_via_scale-adaptive_tokens.md)
- [MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos](megasam_accurate_fast_and_robust_structure_and_motion_from_casual_dynamic_videos.md)
- [SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos](slam3r_real-time_dense_scene_reconstruction_from_monocular_rgb_videos.md)
- [4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)

<!-- RELATED:END -->
