---
title: >-
  [论文解读] MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting
description: >-
  [CVPR 2026][3D视觉][4D重建] 提出 MotionScale，一个可扩展的 4D 高斯泼溅框架，通过基于聚类的自适应运动场和渐进式优化策略，从单目视频中高保真重建大规模动态场景的外观、几何和运动，在 DyCheck 上 PSNR 达到 17.98，3D 跟踪 EPE 降至 0.070，显著超越现有方法。
tags:
  - CVPR 2026
  - 3D视觉
  - 4D重建
  - 高斯泼溅
  - 动态场景
  - 运动场
  - 单目视频
---

# MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting

**会议**: CVPR 2026  
**arXiv**: [2603.29296](https://arxiv.org/abs/2603.29296)  
**代码**: [项目主页](https://hrzhou2.github.io/motion-scale-web/)  
**领域**: 3D视觉  
**关键词**: 4D重建, 高斯泼溅, 动态场景, 运动场, 单目视频

## 一句话总结

提出 MotionScale，一个可扩展的 4D 高斯泼溅框架，通过基于聚类的自适应运动场和渐进式优化策略，从单目视频中高保真重建大规模动态场景的外观、几何和运动，在 DyCheck 上 PSNR 达到 17.98，3D 跟踪 EPE 降至 0.070，显著超越现有方法。

## 研究背景与动机

1. **领域现状**：动态4D场景重建是计算机视觉的核心挑战。近年来，NeRF 和 3DGS 在静态或轻度动态场景中取得了很好的效果，特别是多视角设置下。最近的工作开始结合 2D 几何/运动先验（如深度估计、点跟踪）与 4DGS 来从单目视频重建场景。

2. **现有痛点**：现有方法虽然能在已观测视点产生合理的视图合成，但在几何精度和长序列时间一致性上存在明显缺陷。具体表现为几何畸变、运动轨迹不连贯、在大尺度场景和长视频中难以扩展。

3. **核心矛盾**：作者识别出两个关键瓶颈——(1) **欠约束的几何**：监督信号主要依赖视角依赖的外观信号，缺乏强制3D结构一致性的能力；(2) **累积的时间漂移**：运动模型依赖缺乏3D感知的2D跟踪先验，长序列中误差不可避免地累积，导致几何崩塌和运动轨迹不一致。

4. **本文目标** 如何设计一个既够表达力又可扩展的运动表示，并配合稳定的优化策略，实现从单目视频对大规模动态场景的高保真4D重建。

5. **切入角度**：作者观察到全局变形场或固定容量架构难以处理多样化的局部运动，提出用聚类驱动的运动场自适应扩展模型容量。

6. **核心 idea**：通过聚类中心的基变换来参数化运动场，并配合自适应分裂/裁剪机制和前景-背景解耦的渐进式优化，实现可扩展的4D高斯泼溅。

## 方法详解

### 整体框架

输入是一段单目视频序列 $\{I_t\}_{t=1}^T$（无已知相机参数）。首先利用预训练的 2D 模型提取单目深度图、前景 mask 和 2D 稠密点轨迹，并用 $\pi^3$ 估计初始相机位姿。场景表示为静态背景 + 动态前景的组合，动态前景由一组标准空间的 3D 高斯表示和一个可扩展运动场驱动。优化过程采用渐进式策略，先在初始时间窗口优化基础表示，然后逐步扩展到新帧。

### 关键设计

1. **聚类中心运动场 (Cluster-Centric Motion Field)**:

    - 功能：用层次化的运动模型表示动态高斯的时变运动
    - 核心思路：将动态高斯分为 $K$ 个不相交的聚类 $\{\mathcal{C}_k\}$。每个聚类有一个全局刚性变换 $\mathbf{G}_k^t \in SE(3)$ 和 $B$ 个细粒度基变换 $\mathcal{B}_k^t$。每个高斯通过可学习的系数向量 $\mathbf{w}_i$ 混合本聚类的基变换得到局部变换，再与全局变换复合得到最终状态：$\boldsymbol{\mu}_i^t = \mathbf{R}_{k,g}^t(\mathbf{R}_{i,\ell}^t \boldsymbol{\mu}_i^0 + \mathbf{t}_{i,\ell}^t) + \mathbf{t}_{k,g}^t$。这种设计让每个高斯只受单个聚类影响，计算高效，同时基变换的混合提供了表达非刚性变形的能力。
    - 设计动机：全局 MLP 或固定数量时间基函数的变形场表达力不足以处理局部多样的运动模式。聚类化设计允许运动场的容量随场景复杂度自适应扩展。

2. **自适应控制机制 (Adaptive Control)**:

    - 功能：动态调整聚类拓扑——分裂运动不一致的聚类，裁剪太小的聚类
    - 核心思路：在长期优化阶段，对每个聚类提取其高斯在传播窗口内的 3D 轨迹作为特征描述子，先用 HDBSCAN 发现密度子聚类，再用凝聚聚类分离两个候选组，若质心距离超过阈值则执行分裂。分裂时复制原始运动参数给两个新聚类，保证优化稳定性。同时裁剪过小的聚类来保持紧凑表示。
    - 设计动机：类似 3DGS 的致密化策略，当单个聚类内的高斯运动出现明显非刚性差异时，说明当前表示不够细粒度，需要分裂以捕获更精细的运动模式。

3. **渐进式优化策略 (Progressive Optimization)**:

    - 功能：将长视频的优化分解为可管理的步骤，确保时间一致性
    - 核心思路：分为两个解耦的传播阶段：(a) **背景扩展**：检测新帧中未被覆盖的区域，从深度图采样新高斯，同时联合优化相机位姿进行亚像素级精修；(b) **前景传播**：通过三阶段精修——初始对齐（单向跟踪损失避免污染已优化帧）→ 短期一致性（双向跟踪损失强化局部时间一致性）→ 长期精修（全序列采样帧对解决累积漂移），逐步建立全局运动一致性。
    - 设计动机：直接在全序列上优化会导致不稳定和漂移。渐进式方法让优化从局部到全局，并通过解耦前景和背景避免两者的优化冲突。

### 损失函数 / 训练策略

- **跟踪损失** $L_{\text{track}}$：最小化渲染 2D 轨迹与 CoTracker3 先验之间的差异
- **深度一致性损失** $L_{\text{depth}}$：确保渲染深度与单目深度先验在跟踪位置的一致性
- **光度损失 (RGB)**：标准的图像重建损失
- **ARAP 正则化**：as-rigid-as-possible 约束保持运动局部刚性
- **Shadow Gaussians**：引入专门的"阴影高斯"来建模运动物体投射的瞬变阴影，仅用光度损失和分割约束优化，不施加几何和运动监督

## 实验关键数据

### 主实验

| 数据集 | 指标 | MotionScale | Shape of Motion | GFlow | 4D-Fly |
|--------|------|-------------|-----------------|-------|--------|
| DyCheck | PSNR↑ | **17.98** | 16.72 | - | 17.03 |
| DyCheck | SSIM↑ | **0.70** | 0.63 | - | 0.60 |
| DyCheck | LPIPS↓ | **0.40** | 0.45 | - | 0.37 |
| NVIDIA | PSNR↑ | **26.75** | 23.37 | - | 22.52 |
| NVIDIA | SSIM↑ | **0.78** | 0.75 | - | 0.69 |

| 方法 | EPE↓ | δ³ᴅ.05↑ | δ³ᴅ.10↑ | AJ↑ | δ_avg↑ | OA↑ |
|------|------|---------|---------|-----|--------|-----|
| MotionScale | **0.070** | **47.0** | **76.4** | **37.7** | **50.6** | **87.4** |
| Shape of Motion | 0.082 | 43.0 | 73.3 | 34.4 | 47.0 | 86.6 |
| SpatialTracker | 0.125 | 37.7 | 63.9 | 24.9 | 36.9 | 73.5 |

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | AJ↑ | δ_avg↑ | OA↑ |
|------|-------|-------|--------|-----|--------|-----|
| Full Model | 17.98 | 0.70 | 0.40 | 37.7 | 50.6 | 87.4 |
| Global Bases | 16.70 | 0.63 | 0.45 | 34.2 | 46.6 | 86.1 |
| w/o Adaptive Control | 17.21 | 0.67 | 0.42 | 34.9 | 47.0 | 86.6 |
| w/o Pose Ref. | 17.45 | 0.67 | 0.41 | - | - | - |
| w/o Shadow | 16.26 | 0.60 | 0.50 | - | - | - |
| w/o FG Propagation | 16.97 | 0.64 | 0.42 | 34.4 | 46.9 | 86.4 |

### 关键发现

- **聚类运动场 vs 全局基**：聚类设计比使用全局共享基的 baseline（类似 Shape of Motion）PSNR 提升 1.28，AJ 提升 3.5，证明局部化运动基对细粒度非刚性变形至关重要
- **自适应控制**的移除导致 PSNR 下降 0.77，AJ 下降 2.8，表明动态调整聚类拓扑对维持运动精度非常关键
- **Shadow Gaussians** 的移除影响最大（PSNR 从 17.98 降到 16.26），且缺少阴影表示会导致前景高斯向阴影区域过度扩展，产生几何膨胀和鬼影伪影
- **位姿精修**虽然定量提升不算大，但定性可视化表明它对保持尖锐纹理非常重要

## 亮点与洞察

- **聚类化运动场的可扩展性设计**非常巧妙：每个高斯只受一个聚类影响，计算成本几乎恒定，但通过分裂机制可以无限扩展容量。这种"固定计算+动态容量"的设计思路可以迁移到其他需要可扩展表示的任务
- **前景传播的三阶段精修**是一个工程上非常重要的策略：先单向对齐（防止新帧噪声污染已有好结果），再双向一致性，最后全局精修。这种从保守到激进的优化顺序值得在其他渐进式优化场景中借鉴
- **Shadow Gaussians** 的引入解决了一个常被忽视但很重要的问题：动态物体投射的阴影。将它独立建模而非让前景高斯来解释阴影，既简化了问题又避免了几何伪影

## 局限与展望

- 依赖于预训练的 2D 先验模型（深度、分割、跟踪），这些模型的失败模式会传播到最终重建
- 自适应聚类分裂依赖 HDBSCAN + 距离阈值，可能在极端运动模式下不够鲁棒
- 仅在有限的数据集（DAVIS、DyCheck、NVIDIA）上验证，缺乏更大规模的室外场景评估
- K-means 初始化的聚类划分对最终结果的影响未充分探讨

## 相关工作与启发

- **vs Shape of Motion**：SoM 使用全局共享的运动基函数，本文使用聚类局部化的运动基并支持自适应扩展。本文在所有指标上显著优于 SoM，特别是在长序列和大运动场景
- **vs GFlow**：GFlow 在大位移下容易产生"云状"伪影和运动不连续，本文通过渐进式优化和聚类约束保持了几何清晰度和运动连续性
- **vs 4D-Fly**：两者 PSNR 接近但本文在 SSIM 和 3D 跟踪上有明显优势，说明聚类运动场在几何一致性方面的优越性

## 评分

- 新颖性: ⭐⭐⭐⭐ 聚类化运动场+自适应分裂的思路有明确的创新点，但整体仍在 4DGS 框架内演进
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集全面评估，NVS + 3D/2D跟踪多维度指标，消融实验详尽
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，pipeline图直观，但公式密度较高
- 价值: ⭐⭐⭐⭐ 在单目4D重建方向推进了SOTA，可扩展运动场的设计有实际应用价值

<!-- RELATED:START -->

## 相关论文

- [4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4.md)
- [MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer](more_motion-aware_feed-forward_4d_reconstruction_transformer.md)
- [MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second](movies_motion-aware_4d_dynamic_view_synthesis_in_one_second.md)
- [Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos](learning_explicit_continuous_motion_representation_for_dynamic_gaussian_splattin.md)
- [AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction](aerodgs_physically_consistent_dynamic_gaussian_splatting_for_single-sequence_aer.md)

<!-- RELATED:END -->
