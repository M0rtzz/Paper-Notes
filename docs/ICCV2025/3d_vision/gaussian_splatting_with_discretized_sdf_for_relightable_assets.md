---
title: >-
  [论文解读] Gaussian Splatting with Discretized SDF for Relightable Assets
description: >-
  [ICCV 2025][3D视觉][逆渲染] 本文提出将连续SDF离散化为高斯基元的额外属性，通过SDF-to-opacity变换统一高斯和SDF表示，配合投影一致性损失和球面初始化，在仅用4G显存的前提下实现了超越现有高斯逆渲染方法的重光照质量。
tags:
  - ICCV 2025
  - 3D视觉
  - 逆渲染
  - 3D高斯溅射
  - 离散化SDF
  - 几何正则化
  - 重光照
---

# Gaussian Splatting with Discretized SDF for Relightable Assets

**会议**: ICCV 2025  
**arXiv**: [2507.15629](https://arxiv.org/abs/2507.15629)  
**代码**: https://github.com/NK-CS-ZZL/DiscretizedSDF  
**领域**: 3D视觉  
**关键词**: 逆渲染, 3D高斯溅射, 离散化SDF, 几何正则化, 重光照

## 一句话总结

本文提出将连续SDF离散化为高斯基元的额外属性，通过SDF-to-opacity变换统一高斯和SDF表示，配合投影一致性损失和球面初始化，在仅用4G显存的前提下实现了超越现有高斯逆渲染方法的重光照质量。

## 研究背景与动机

从多视角RGB图像中分解几何、材质和光照（逆渲染）是一个经典且重要的问题，对增强现实、虚拟现实等下游应用至关重要。其中，几何质量是材质和光照估计的前提——只有足够好的法线和表面，才能得到合理的BRDF分解。

**SDF的优势与NeRF时代的成功**：SDF作为几何先验已在NeuS等NeRF框架中证明了其有效性。TensoSDF、NeRO等基于SDF的方法能对各种材质（包括高反射物体）进行鲁棒的几何-材质分解。但代价是昂贵的ray marching，训练和渲染速度慢。

**3DGS进入逆渲染的困境**：3D高斯溅射（3DGS）凭借实时渲染和快速训练成为新宠。但直接在GS框架中做逆渲染面临严峻的几何约束不足问题。GShader、GS-IR等早期方法仅利用法线对齐等弱约束，分解质量有限。

**现有解决方案的代价**：为了弥补几何约束的不足，GS-ROR、GSDF等方法引入额外的连续SDF网络与高斯联合优化。这确实提升了几何质量，但带来三个问题：(1) 显存大幅增加（GS-ROR需22G vs 本文4G）；(2) 需要复杂优化策略（warm-up、多阶段训练）来平衡两种表示；(3) 模型复杂度增加。

**本文的核心insight**：能不能把SDF直接编码到高斯基元本身上？这样不需要额外的网络，用一个统一表示就同时拥有GS的灵活性和SDF的鲁棒性。具体做法是给每个高斯添加一个SDF值属性，通过SDF-to-opacity变换将SDF和高斯的opacity联系起来，然后通过splat渲染SDF。

## 方法详解

### 整体框架

基于2DGS（2D高斯溅射）搭建deferred shading逆渲染pipeline。每个2D高斯椭盘拥有标准的PBR属性（albedo、roughness等）、缩放和旋转外，额外编码一个SDF值 $s_i$。opacity不再直接学习，而是通过SDF-to-opacity变换导出。使用Principled BRDF + split-sum近似作为着色模型。

### 关键设计

1. **离散化SDF与SDF-to-Opacity变换**:

    - 功能：在每个高斯基元上编码一个SDF采样值，通过变换函数将其映射为opacity
    - 核心思路：SDF-to-opacity变换是一个钟形函数：
    $o_i = \mathcal{T}_\gamma(s_i) = \frac{4 \cdot e^{-\gamma s_i}}{(1 + e^{-\gamma s_i})^2}$
   其中 $s_i$ 是第 $i$ 个高斯的SDF值，$\gamma$ 是全局可学习参数控制变换的方差。SDF值为0时opacity最大（=1），离表面越远opacity越小
    - 设计动机：这个变换将SDF和高斯渲染无缝连接——SDF定义了表面位置（零水平集），opacity的分布自然地使高斯聚集在表面附近。无需额外网络，仅用高斯基元就编码了SDF信息

2. **中位数损失（Median Loss）**:

    - 功能：解决SDF值 $s_i$ 和变换参数 $\gamma$ 联合优化的收敛困难
    - 核心思路：发现所有高斯无符号距离的中位数 $|s|_m$ 是收敛的良好指标。当中位数距离处的opacity过高时，变换需要变窄。推导 $\gamma_m$ 使得 $\mathcal{T}_{\gamma_m}(|s|_m) = 0.5$：
    $\gamma_m = -\frac{\log(3 - 2\sqrt{2})}{|s|_m}$
   中位数损失引导 $\gamma$ 快速逼近 $\gamma_m$：$\mathcal{L}_\gamma = \max(\gamma_m - \gamma, 0)$
    - 设计动机：没有这个损失，$\gamma$ 和 $s_i$ 的搜索空间太大，收敛极慢。中位数提供了全局统计信息来指导变换参数

3. **投影一致性损失（Projection-based Consistency Loss）**:

    - 功能：在离散化表示下正则化SDF，替代无法使用的Eikonal损失
    - 核心问题：连续SDF可以用Eikonal损失（$|\nabla f|=1$）正则化，但离散化的SDF只能获取梯度方向（即法线），无法计算梯度幅值，因此经典Eikonal损失不可用
    - 核心思路：将每个高斯投影到SDF的零水平集上得到投影点 $\mu_{proj} = \mu_i - s_i \frac{\nabla f_i}{|\nabla f_i|}$，要求投影点的深度与alpha-blended表面的深度一致：
    $\mathcal{L}_p = \frac{1}{N}\sum_{i \in N} \begin{cases} \epsilon_{proj}^i, & \epsilon_{proj}^i \leq \varepsilon \\ 0, & \epsilon_{proj}^i > \varepsilon \end{cases}$
   其中使用阈值 $\varepsilon$ 过滤被遮挡面或自交叉导致的异常高斯
    - 设计动机：作者证明了该损失是Eikonal条件的近似，能在离散表示下有效正则化SDF。1K迭代后启用，此时粗糙几何已稳定

4. **球面初始化（Spherical Initialization）**:

    - 功能：用单位球面上均匀采样的点初始化前景物体的高斯
    - 核心思路：借鉴NeuS等体积渲染SDF方法中球面初始化的成功经验
    - 设计动机：避免错误的早期几何陷入局部最优。普通初始化容易导致早期出现错误几何（如破碎表面），且难以恢复

### 损失函数 / 训练策略

总损失：$\mathcal{L} = \mathcal{L}_c + \lambda_n\mathcal{L}_n + \lambda_d\mathcal{L}_d + \lambda_\gamma\mathcal{L}_\gamma + \lambda_p\mathcal{L}_p + \lambda_{sm}\mathcal{L}_{sm} + \lambda_m\mathcal{L}_m$

各项含义：$\mathcal{L}_c$（渲染损失）、$\mathcal{L}_n$（法线一致性）、$\mathcal{L}_d$（distortion）继承自2DGS；$\mathcal{L}_\gamma$（中位数损失）和$\mathcal{L}_p$（投影一致性损失）是本文提出；$\mathcal{L}_{sm}$（PBR属性平滑正则化）、$\mathcal{L}_m$（可选mask损失）。使用Adam优化器训练30K迭代。

## 实验关键数据

### 主实验（Glossy Blender重光照）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 训练时间 | 显存 |
|------|-------|-------|--------|---------|------|
| GShader | 16.95 | 0.8430 | 0.2365 | 0.5h | 4G |
| GS-IR | 16.26 | 0.6494 | 0.1361 | 0.5h | 8G |
| R3DG | 17.09 | 0.8258 | 0.1260 | 1h | 20G |
| GS-ROR | 23.39 | 0.9140 | 0.0769 | 1.5h | 22G |
| **Ours** | **24.52** | **0.9229** | **0.0762** | **1h** | **4G** |

### 消融实验（Glossy Blender, 3个场景）

| 配置 | Angel PSNR/SSIM | Horse PSNR/SSIM | Teapot PSNR/SSIM |
|------|----------------|-----------------|------------------|
| 基线(无SDF) | 20.23/0.8533 | 20.72/0.8998 | 19.22/0.8674 |
| + 离散SDF+中位数损失 | 21.09/0.8815 | 22.13/0.9120 | 23.26/0.9223 |
| + 球面初始化(单独) | 21.43/0.8881 | 21.96/0.9102 | 22.97/0.9176 |
| + SDF+球面初始化 | 21.59/0.8910 | 22.87/0.9284 | 23.45/0.9222 |
| + SDF+投影损失 | 21.65/0.8917 | 23.21/0.9395 | 24.05/0.9289 |
| + SDF+投影+球面(完整) | **22.03/0.8919** | **24.01/0.9481** | **24.19/0.9293** |

### 关键发现

- 在Glossy Blender上PSNR超出GS-ROR 1.13dB（24.52 vs 23.39），同时显存仅需4G（GS-ROR需22G，5倍差距）
- 在TensoIR合成数据集上同样最优：平均PSNR 27.78 vs GS-ROR的27.07
- Chamfer Distance几何质量在Glossy Blender上平均0.0107，大幅领先GS-ROR的0.0140和R3DG的0.0315
- 法线MAE在Shiny Blender上平均6.48°，优于GS-ROR的7.23°
- 三个组件（离散SDF、投影损失、球面初始化）逐步加入均带来持续提升，证明每个设计的必要性

## 亮点与洞察

- **统一表示的优雅**：不引入任何额外网络或表示，将SDF直接编码在高斯属性上，一个表示同时获得GS的渲染效率和SDF的几何鲁棒性
- **理论贡献**：证明了投影一致性损失是Eikonal条件的近似，为离散化SDF的正则化提供了理论基础
- **极致的效率**：4G显存+1小时训练就超过需要22G+1.5小时的GS-ROR，实际中非常有意义
- **中位数统计量的巧妙使用**：用全局统计信息（中位数）来指导局部参数（变换宽度），解决联合优化的锚定问题

## 局限与展望

- 仅考虑直接光照，对复杂遮挡和间接照明的场景可能失效，需要加入间接照明项
- 在复杂相互反射的场景中（如Toaster、Luyu）仍不如引入完整SDF网络的GS-ROR
- 目前聚焦于物体级重光照，尚未扩展到无界场景
- 未涉及mesh提取和BRDF参数导出

## 相关工作与启发

- **NeuS/3DGSR**：将SDF引入体积渲染/高斯框架的先驱，本文的SDF-to-opacity变换灵感来源
- **Neural-Pull**：投影到零水平集的思想与本文的投影一致性损失共享理论基础
- **GS-ROR**：当前Gaussian逆渲染SOTA，引入额外SDF网络，本文的直接对标方法
- 启发：将"额外网络"变为"属性编码"的思路可以推广到其他需要额外几何先验的GS方法中

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (离散化SDF编码到高斯属性上的思路非常新颖)
- 实验充分度: ⭐⭐⭐⭐⭐ (3个合成数据集+1个真实数据集+充分消融+多种定量指标)
- 写作质量: ⭐⭐⭐⭐⭐ (理论推导严谨，图表精美)
- 价值: ⭐⭐⭐⭐⭐ (显存5倍节省+性能提升，强实用价值)

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Discretized Gaussian Representation for Tomographic Reconstruction](discretized_gaussian_representation_for_tomographic_reconstruction.md)
- [\[CVPR 2025\] HRAvatar: High-Quality and Relightable Gaussian Head Avatar](../../CVPR2025/3d_vision/hravatar_high-quality_and_relightable_gaussian_head_avatar.md)
- [\[CVPR 2025\] RNG: Relightable Neural Gaussians](../../CVPR2025/3d_vision/rng_relightable_neural_gaussians.md)
- [\[CVPR 2025\] SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Rendering](../../CVPR2025/3d_vision/svg-ir_spatially-varying_gaussian_splatting_for_inverse_rendering.md)
- [\[ICCV 2025\] REPARO: Compositional 3D Assets Generation with Differentiable 3D Layout Alignment](reparo_compositional_3d_assets_generation_with_differentiable_3d_layout_alignmen.md)

<!-- RELATED:END -->
