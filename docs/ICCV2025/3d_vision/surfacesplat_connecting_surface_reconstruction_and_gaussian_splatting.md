---
title: >-
  [论文解读] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting
description: >-
  [ICCV 2025][3D视觉][表面重建] SurfaceSplat 提出了一种混合方法，将 SDF（有符号距离函数）和 3D 高斯溅射（3DGS）双向连接：SDF 提供粗糙几何来增强 3DGS 的渲染质量，而 3DGS 渲染的新视角图像反过来用于细化 SDF 的表面重建精度，在 DTU 和 MobileBrick 数据集上同时超越了表面重建和新视角合成的 SOTA。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "表面重建"
  - "高斯溅射"
  - "SDF"
  - "稀疏视角"
  - "新视角合成"
---

# SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting

**会议**: ICCV 2025  
**arXiv**: [2507.15602](https://arxiv.org/abs/2507.15602)  
**代码**: [https://github.com/aim-uofa/SurfaceSplat](https://github.com/aim-uofa/SurfaceSplat)  
**领域**: 3D视觉  
**关键词**: 表面重建, 高斯溅射, SDF, 稀疏视角, 新视角合成

## 一句话总结

SurfaceSplat 提出了一种混合方法，将 SDF（有符号距离函数）和 3D 高斯溅射（3DGS）双向连接：SDF 提供粗糙几何来增强 3DGS 的渲染质量，而 3DGS 渲染的新视角图像反过来用于细化 SDF 的表面重建精度，在 DTU 和 MobileBrick 数据集上同时超越了表面重建和新视角合成的 SOTA。

## 研究背景与动机

**领域现状**：从稀疏视角图像进行表面重建和新视角渲染是 3D 视觉的核心挑战。目前主流有两条技术路线：一是基于 SDF 的神经隐式表面方法（如 NeuS、VolRecon），通过学习有符号距离场来获取几何；二是基于 3D 高斯溅射（3DGS）的方法，通过优化离散的 3D 高斯基元实现快速高质量渲染。

**现有痛点**：SDF 方法擅长捕捉全局几何一致性，但在精细细节上表现不佳，因为隐式表示的分辨率受限于网络容量，并且在稀疏视角下容易过度平滑。3DGS 方法渲染质量优秀、速度快，但其离散点云表示缺乏全局几何约束，容易产生浮动的高斯点（floaters）和不一致的表面。

**核心矛盾**：SDF 的全局几何约束能力与 3DGS 的局部细节渲染能力难以兼得。两种表示各有优势，但此前的方法只用其中一种，或者只是简单地用一种初始化另一种，没有建立双向的协同优化关系。

**本文目标**：设计一个统一框架，让 SDF 和 3DGS 互相增强——SDF 为 3DGS 提供几何先验，3DGS 为 SDF 提供更多视角的监督信号。

**切入角度**：作者观察到 SDF 和 3DGS 的优缺点恰好互补：SDF 擅长全局几何但缺细节，3DGS 擅长细节但缺全局一致性。如果能让两者互相"教学"，就可以同时提升两方面的性能。

**核心 idea**：建立 SDF→3DGS 和 3DGS→SDF 的双向信息流——用 SDF 的粗糙几何来约束 3DGS 的高斯分布，同时用 3DGS 渲染的高质量新视角图像作为 SDF 优化的额外监督。

## 方法详解

### 整体框架

SurfaceSplat 的输入是稀疏视角图像（及相机参数），输出是高质量的表面网格和新视角渲染图像。系统包含两个并行优化的分支：SDF 分支（基于神经隐式表面）和 3DGS 分支（基于 3D 高斯溅射）。两个分支通过精心设计的信息传递机制双向耦合，在联合优化过程中互相增强。

### 关键设计

1. **SDF→3DGS 几何增强**:

    - 功能：利用 SDF 的全局几何知识来改善 3DGS 的高斯分布
    - 核心思路：从 SDF 网络中提取等值面（0-level set），将网格顶点和法向量信息用于初始化和约束 3DGS 的高斯基元。具体来说，用 SDF 提取的表面法向量来正则化高斯的朝向，用表面位置来约束高斯中心的分布范围。这样可以有效减少 3DGS 中常见的浮动高斯问题，让高斯基元更紧密地贴合真实表面。
    - 设计动机：3DGS 在稀疏视角下容易出现 floaters，根本原因是缺乏全局几何约束。SDF 天然提供了表面先验，将其用于约束高斯分布是直觉上合理且高效的做法。

2. **3DGS→SDF 细节细化**:

    - 功能：利用 3DGS 渲染的新视角图像为 SDF 提供额外监督信号
    - 核心思路：3DGS 可以高效渲染任意视角的图像。在训练过程中，从 3DGS 渲染额外的"虚拟视角"图像，将这些图像作为 SDF 优化的伪标签监督。这相当于用 3DGS 的渲染能力来"增强"SDF 的训练数据——原本只有稀疏几个视角的真实图像，现在多了更多视角的渲染图像作观。为避免引入渲染伪影，采用置信度加权策略，只让高置信度的渲染像素参与 SDF 监督。
    - 设计动机：SDF 在稀疏视角下表现不佳的一个重要原因是训练数据太少。3DGS 作为一种优秀的插值器，可以在已有视角之间合成高质量的新视角图像，弥补 SDF 训练数据不足的问题。

3. **联合优化策略**:

    - 功能：协调两个分支的训练过程
    - 核心思路：采用交替优化的方式——先预训练 SDF 分支获取初步几何，然后用初步几何初始化 3DGS，之后两个分支交替优化并互相传递信息。每隔固定步数更新一次 SDF→3DGS 的几何约束，同时持续用 3DGS 的渲染图像监督 SDF。为防止两个分支互相传播错误（error accumulation），引入了渐进式耦合策略，初期两个分支较独立，后期逐渐加强耦合强度。
    - 设计动机：直接端到端联合训练容易导致训练不稳定，因为两个分支都处于未收敛状态时互相传递的信号噪声很大。渐进式耦合让两个分支先各自学到基本能力，再逐步建立协作。

### 损失函数 / 训练策略

总损失包含 SDF 分支损失和 3DGS 分支损失：SDF 分支使用 RGB 渲染损失、深度正则化和 eikonal 正则化；3DGS 分支使用标准的光度损失和 SSIM 损失。双向信息传递通过额外的法向量一致性损失 $L_{normal}$ 和虚拟视角监督损失 $L_{pseudo}$ 实现，后者带有置信度加权：$L_{pseudo} = \sum_i w_i \|I_{SDF}^i - I_{3DGS}^i\|_1$。

## 实验关键数据

### 主实验

在 DTU 数据集（3 个输入视角）上的表面重建质量（Chamfer Distance, mm）和新视角合成质量（PSNR）：

| 方法 | CD↓ (mm) | PSNR↑ | SSIM↑ | 类型 |
|------|---------|-------|-------|------|
| SparseNeuS | 1.40 | 23.12 | 0.872 | SDF |
| VolRecon | 1.20 | 23.58 | 0.881 | SDF |
| C2F2NeuS | 1.13 | 24.01 | 0.889 | SDF |
| DNGaussian | - | 24.68 | 0.899 | 3DGS |
| CoR-GS | - | 24.32 | 0.895 | 3DGS |
| SurfaceSplat | **0.98** | **25.21** | **0.912** | 混合 |

在 MobileBrick 数据集上的结果：

| 方法 | CD↓ (mm) | F-score↑ | PSNR↑ |
|------|---------|----------|-------|
| NeuS | 2.85 | 0.62 | 22.34 |
| 3DGS | - | - | 24.89 |
| SurfaceSplat | **2.12** | **0.74** | **25.67** |

### 消融实验

| 配置 | CD↓ (mm) | PSNR↑ | 说明 |
|------|---------|-------|------|
| Full model | 0.98 | 25.21 | 完整模型 |
| w/o SDF→3DGS | 1.08 | 24.53 | 去掉几何增强后 3DGS 有更多 floaters |
| w/o 3DGS→SDF | 1.21 | 25.18 | 去掉虚拟视角监督后 SDF 精度下降明显 |
| w/o 渐进耦合 | 1.15 | 24.87 | 直接强耦合导致训练不稳定 |
| 只用 SDF 分支 | 1.35 | 23.45 | 纯 SDF 基线 |
| 只用 3DGS 分支 | - | 24.68 | 纯 3DGS 基线，无表面重建 |

### 关键发现

- 双向信息流的两个方向贡献不同：3DGS→SDF 的虚拟视角监督对表面重建质量的提升更大（CD 从 0.98 升到 1.21），而 SDF→3DGS 的几何增强对渲染质量贡献更大（PSNR 从 25.21 降到 24.53）
- 渐进式耦合策略是稳定训练的关键，直接强耦合会导致两个分支互相传播早期错误
- 在更极端的稀疏视角（如 2 个视角）下，SurfaceSplat 相比纯 SDF 或纯 3DGS 方法的优势更明显，因为互补效应在数据更稀缺时更重要

## 亮点与洞察

- **双向互补的优雅设计**：让两种表示互相"教学"的思路简洁而有效。SDF 教 3DGS 几何，3DGS 教 SDF 细节，形成良性循环。这种互补设计思路可以迁移到其他存在互补表示的任务中。
- **虚拟视角增广策略**：用 3DGS 渲染新视角来增强 SDF 训练数据，是一种低成本的数据增强方式。关键技巧是用置信度加权过滤低质量渲染，避免引入噪声。
- **渐进耦合**：解决了双分支系统早期互相传播错误的工程难题，这种"先独立后协作"的训练策略在多任务学习中具有普遍参考价值。

## 局限与展望

- 训练时间比单一方法更长，因为需要同时优化两个分支并进行信息传递
- 对相机参数的准确性仍有依赖，虽然稀疏视角鲁棒但如果相机内外参不准确，两个分支可能传播错误几何
- 目前仅在物体级数据集（DTU、MobileBrick）上验证，对大场景的适用性尚不确定
- 3DGS→SDF 的虚拟视角选择策略较简单（随机采样），更智能的视角选择可能进一步提升效果

## 相关工作与启发

- **vs NeuS/VolRecon**: 纯 SDF 方法在稀疏视角下全局一致但细节不足。SurfaceSplat 通过引入 3DGS 的渲染能力来补充细节监督。
- **vs DNGaussian/CoR-GS**: 纯 3DGS 方法渲染好但缺乏表面几何。SurfaceSplat 通过 SDF 提供几何约束解决 floaters 问题。
- **vs 2DGS/SuGaR**: 这些方法尝试在 3DGS 中加入表面约束，但仍是单一表示。SurfaceSplat 的双表示设计更灵活，能同时输出高质量网格和渲染。

## 评分

- 新颖性: ⭐⭐⭐⭐ SDF+3DGS 双向互补的设计思路新颖，但两种表示结合的想法并非全新
- 实验充分度: ⭐⭐⭐⭐ DTU 和 MobileBrick 是标准 benchmark，消融实验充分
- 写作质量: ⭐⭐⭐⭐ 动机清晰、方法描述详细
- 价值: ⭐⭐⭐⭐ 同时提升重建和渲染质量，在稀疏视角场景下有实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](../../AAAI2026/3d_vision/sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[CVPR 2025\] DepthSplat: Connecting Gaussian Splatting and Depth](../../CVPR2025/3d_vision/depthsplat_connecting_gaussian_splatting_and_depth.md)
- [\[ICCV 2025\] MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction](mugs_multi-baseline_generalizable_gaussian_splatting_reconstruction.md)
- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)
- [\[ECCV 2024\] Surface Reconstruction from 3D Gaussian Splatting via Local Structural Hints](../../ECCV2024/3d_vision/surface_reconstruction_from_3d_gaussian_splatting_via_local_structural_hints.md)

</div>

<!-- RELATED:END -->
