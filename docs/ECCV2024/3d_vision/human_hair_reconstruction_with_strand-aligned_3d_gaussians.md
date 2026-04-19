---
title: >-
  [论文解读] Human Hair Reconstruction with Strand-Aligned 3D Gaussians
description: >-
  [ECCV 2024][3D视觉][头发重建] 本文提出 Gaussian Haircut，通过经典发丝多段线和 3D 高斯基元的双表示（strand-aligned 3D Gaussians），结合 3D 方向场提升和粗到细的发丝拟合优化策略，从多视角图像重建出可直接用于图形引擎编辑、渲染和物理仿真的高保真发丝级发型，速度比之前方法快 10 倍以上。
tags:
  - ECCV 2024
  - 3D视觉
  - 头发重建
  - 3D高斯溅射
  - 发丝级建模
  - 双表示
  - 粗到细优化
---

# Human Hair Reconstruction with Strand-Aligned 3D Gaussians

**会议**: ECCV 2024  
**arXiv**: [2409.14778](https://arxiv.org/abs/2409.14778)  
**代码**: https://eth-ait.github.io/GaussianHaircut  
**领域**: 3D视觉 / 数字人  
**关键词**: 头发重建、3D高斯溅射、发丝级建模、双表示、粗到细优化

## 一句话总结

本文提出 Gaussian Haircut，通过经典发丝多段线和 3D 高斯基元的双表示（strand-aligned 3D Gaussians），结合 3D 方向场提升和粗到细的发丝拟合优化策略，从多视角图像重建出可直接用于图形引擎编辑、渲染和物理仿真的高保真发丝级发型，速度比之前方法快 10 倍以上。

## 研究背景与动机

**领域现状**：高保真人类头发的 3D 重建是数字人建模中最具挑战性的子问题之一。行业标准是基于发丝（strand）的表示——用 3D 多段线描述每根头发的几何形状，这种表示可以直接导入 Unreal Engine 等图形引擎进行渲染和物理仿真。近年来基于 3D 高斯溅射（3DGS）的方法在人体头像重建方面取得了突破性进展，但这些方法将头发作为可渲染的视觉表面来建模，使用非结构化的高斯基元，无法提取出可用于仿真的发丝结构。

**现有痛点**：（1）**非结构化 vs 结构化的鸿沟**——3DGS 的高斯基元虽然渲染效果好，但本质上是一组无组织的点云，无法表达头发的内部拓扑结构（哪些基元属于同一根发丝、发丝的生长方向等）。（2）**方向图的噪声问题**——基于图像的发丝建模依赖 2D 方向图（orientation maps）来推断 3D 发丝方向，传统方法使用 Gabor 滤波器从 RGB 图像计算方向图，但这些方向图inherently noisy，严重影响后续发丝拟合的精度。（3）**可见部分 vs 内部结构**——图像只能观察到头发外表面，但真实发型的内部结构（发丝如何从头皮生长、如何在内部交织）对物理仿真至关重要。先前方法（如 Neural Haircut）虽然引入了扩散先验来推断内部结构，但重建速度极慢。

**核心矛盾**：发丝级重建需要结构化的表示（多段线）来支持下游应用，但结构化表示缺乏可微渲染能力，无法利用光度学约束；而 3DGS 具有优秀的可微渲染能力但缺乏结构性。如何将两者结合是核心难题。

**本文目标** 设计一种将发丝多段线和 3D 高斯基元统一的双表示方案，使得发丝重建既能利用可微渲染的光度学约束来提升几何精度，又能输出图形引擎可用的结构化发丝。

**切入角度**：作者提出一个关键观察：3DGS 中的高斯基元在学习细长结构（如头发）时，会自然地沿发丝方向拉长——协方差矩阵的最大方差方向与发丝方向对齐。这意味着 3DGS 本身就在隐含地学习 3D 方向场。将这个方向场显式提取出来，就可以将非结构化的高斯"提升"为结构化的发丝表示。

**核心 idea**：将高斯基元绑定到发丝线段上形成 strand-aligned Gaussians，既保留发丝的结构化表示又获得可微渲染能力，实现两全其美。

## 方法详解

### 整体框架

方法分为两个阶段。**第一阶段（3D Line Lifting）**：使用修改版的 3DGS 从多视角图像重建场景，同时进行相机参数优化。3DGS 的协方差矩阵被用来提取 3D 方向场和以高质量方向图。**第二阶段（Hair Strands Fitting）**：基于第一阶段得到的方向场和渲染结果，通过粗到细的优化策略拟合发丝级发型。粗阶段在潜在空间中优化发型的全局结构，细阶段在显式坐标空间中优化每根发丝的精细几何。整个过程中使用 strand-aligned 3D Gaussians 作为可微渲染的桥梁。

### 关键设计

1. **基于 3DGS 的 3D 方向场提升（3D Line Lifting with Unstructured Gaussians）**:

    - 功能：从多视角图像中重建场景的 3D 几何，并提取去噪后的头发方向场
    - 核心思路：在标准 3DGS 上做两个关键扩展。首先，引入 BARF 风格的 6-DoF 相机参数优化作为 SfM 初始估计的残差，解决头发场景中 SfM 定位不准确的问题。其次，为每个高斯基元添加额外的可学习属性：头发分割标签 $l$、方向置信度 $\tau$，并利用协方差矩阵 $\Sigma = RSS^TR^T$ 的最大方差方向作为 3D 发丝方向 $\beta_i$。方向监督损失设计为 $\mathcal{L}_{dir} = \sum_p \tau_p \min\{d(\beta_p, \hat{\beta}_p), d(\beta_p, \hat{\beta}_p) \pm \pi\} - \log \tau_p$，其中 $d$ 是角度差，$\hat{\beta}_p$ 是 Gabor 滤波器计算的真值方向。置信度 $\tau_p$ 的引入允许模型在方向不确定的区域（如头发边缘、缠绕区域）自动降低权重，比直接强制对齐 Gabor 方向更加鲁棒
    - 设计动机：传统方法直接使用 Gabor 滤波器的方向图做 3D 提升，噪声严重。而 3DGS 的协方差矩阵天然编码了局部几何方向，利用这一特性提升的方向图在实验中比 Gabor 滤波器降低了约 1° 的平均角误差，且在光照不佳和头发缠绕区域改善显著

2. **Strand-Aligned 3D Gaussians 双表示**:

    - 功能：将发丝多段线和 3D 高斯基元统一为可微渲染的结构化表示
    - 核心思路：对于每根发丝 $S_k = \{p_l^k\}$，在相邻控制点 $p_l^k$ 和 $p_{l+1}^k$ 形成的每个线段上放置一个 3D 高斯。该高斯的参数完全由所在线段决定：缩放向量 $s_l^k = \{\frac{1}{2}\|p_{l+1}^k - p_l^k\|_2, \epsilon, \epsilon\}$（沿线段方向拉长，正交方向设为极小值 $\epsilon$），旋转quaternion对齐 x 轴与线段方向，不透明度设为 1。每个高斯还有可训练的球谐系数 $f_l^k$ 用于颜色建模。这样，对发丝坐标的梯度可以通过高斯的渲染过程反向传播——修改发丝控制点位置会改变高斯的位置、缩放和方向，进而改变渲染结果，形成从像素到发丝几何的完整可微路径
    - 设计动机：先前方法（如 Neural Haircut）将发丝渲染为网格进行可微渲染，但网格化过程丢失了高频细节且渲染速度慢。Strand-aligned Gaussians 利用 3DGS 成熟的光栅化管线，渲染效率极高，且单个高斯的参数自由度足以捕获发丝的粗细变化

3. **粗到细的发丝优化（Coarse-to-Fine Strands Fitting）**:

    - 功能：从初始化到精细几何的渐进式发型重建
    - 核心思路：发型表示为头皮纹理图 $H$，每个纹素存储一根发丝的 3D 多段线。由于自由度极高，直接优化容易崩溃。**粗阶段**：使用预训练的编码器 $\mathcal{E}$ 和解码器 $\mathcal{G}$ 将发型映射到低维潜在空间 $Z = \mathcal{E}(H)$，在潜在空间中优化。由于内存限制，每次只解码 1000 根引导发丝 $H'$，再通过 K 近邻插值上采样到 10000 根来渲染——这个上采样技巧是保证粗阶段光度学损失有效的关键。**细阶段**：从潜在图解码出 30000 根完整发丝，然后冻结解码器，直接在 3D 坐标空间中优化每根发丝的控制点。两个阶段都使用基于扩散模型的 SDS 正则化来确保内部发丝结构的真实性
    - 设计动机：潜在空间约束提供了强大的发型先验，防止早期优化阶段出现不合理的发丝形状；显式坐标优化则在先验引导的基础上追求精细的几何细节。两者互补

### 损失函数 / 训练策略

第一阶段：$\mathcal{L}_{gaussian} = \mathcal{L}_{rgb} + \lambda_{seg}\mathcal{L}_{seg} + \lambda_{dir}\mathcal{L}_{dir}$，包含颜色重建、分割掩码和方向监督。

第二阶段：$\mathcal{L}_{strand} = \mathcal{L}_{rgb} + \lambda_{seg}\mathcal{L}_{seg} + \lambda_{dir}\mathcal{L}_{dir} + \lambda_{sds}\mathcal{L}_{sds}$，额外添加扩散先验的 SDS 损失。损失权重设为 $\lambda_{seg} = \lambda_{dir} = 10^{-1}$，$\lambda_{sds} = 10^{-2}$。总训练时间约 6 小时（RTX 4090）。

## 实验关键数据

### 主实验

**真实场景定性对比**（与 Neural Haircut 对比）:

| 指标 | Neural Haircut | Gaussian Haircut (Ours) |
|------|---------------|------------------------|
| 重建质量 | 可见表面基本准确，内部结构粗糙 | 内外结构均更精确，缠绕区域显著改善 |
| 优化时间 | ~60 小时 | **~6 小时** |
| 加速比 | 1× | **>10×** |
| 可仿真性 | 可以但动态不够真实 | 可以且动态更合理 |

**合成场景定量对比**（方向图误差）:

| 方法 | 平均角误差↓ |
|------|-----------|
| Gabor 滤波器 | 8° |
| Ours (3D Lifting) | **7°** |

### 消融实验

| 配置 | 重建质量 | 说明 |
|------|---------|------|
| Full model | 最佳 | 完整方法 |
| w/o fine fitting | 明显退化 | 缺少精细几何优化，发丝轮廓粗糙 |
| w/o synthetic renders | 退化 | 在光照差和缠绕区域精度下降 |
| w/o strands upsampling | 严重退化 | 粗阶段光度学损失失效，细阶段无法收敛 |
| w/o $\mathcal{L}_{dir}$ | 退化 | 方向约束缺失导致发丝走向混乱 |
| w/o $\mathcal{L}_{sds}$ | 内部结构退化 | 扩散先验对不可见区域的结构至关重要 |
| w/o $\mathcal{L}_{rgb}$ | 外观不匹配 | 颜色损失是外表面精度的主要驱动 |

### 关键发现

- **Strand upsampling 技巧至关重要**——去除后细阶段完全无法收敛（尤其在密集发丝区域），因为粗阶段仅有 1000 根引导发丝无法覆盖足够的图像区域来计算有效的光度学损失
- **粗阶段 → 细阶段的过渡**是方法成功的关键——仅用粗阶段（潜在空间优化）可以获得大致合理的发型，但边缘和细节不够；仅用细阶段（直接坐标优化）无法从随机初始化收敛
- **方向损失和 SDS 损失互补**——$\mathcal{L}_{dir}$ 主要约束可见表面的发丝走向，$\mathcal{L}_{sds}$ 主要约束不可见内部的发丝分布
- 相机优化对头发场景尤为重要——标准 SfM （COLMAP）在头发区域的定位误差较大，加入 BARF 风格的相机优化后渲染质量明显提升
- 重建结果可直接导入 Unreal Engine 进行物理仿真，动态效果合理，这得益于发丝连接到 FLAME 头部模型且内部结构真实

## 亮点与洞察

- **双表示的巧妙统一**是本文最核心的贡献——strand-aligned Gaussians 既保持发丝的结构完整性（每个高斯严格绑定到一根发丝的一个线段），又完全兼容 3DGS 的高效可微渲染管线。这种"结构化的非结构化表示"范式可以推广到其他细长结构的重建中（如管线、电缆、植物枝条）
- **利用高斯协方差矩阵隐含编码的方向信息**是一个非常聪明的观察——别人用 3DGS 渲染颜色，这篇论文额外提取了方向场这个"免费"的几何副产品
- **10 倍速度提升**在不牺牲质量的前提下实现，使得发丝级重建从研究工具走向实际生产成为可能

## 局限与展望

- **卷发建模困难**——发丝先验基于从根到尖的生长模型，对于高度卷曲的发型（如爆炸头、非洲辫）表达能力不足。需要更灵活的发丝先验模型
- **不支持复杂编织结构**——如辫子、发髻等具有交织拓扑的发型超出了当前方法的能力范围
- **非结构化光照条件**——虽然方法不要求实验室光照，但极端逆光或强反射条件下性能未充分评估
- 6 小时的重建时间对生产来说仍然较长，可以考虑引入 feed-forward 的初始化方法加速收敛
- 目前仅支持静态发型重建，动态发型（如风中飘动的头发）需要额外的时序建模

## 相关工作与启发

- **vs Neural Haircut**: 本文的前序工作，使用 NeuS + Chamfer 损失重建发丝，本文用 Strand-aligned Gaussians 替代了两个关键环节（方向图提升和可微渲染），实现了 10× 加速和质量提升
- **vs GaussianHair (Luo et al.)**: 并行工作，同样使用 3DGS 重建头发，但需要控制光照的工作室拍摄环境。本文支持非约束拍摄且引入了相机优化方案
- **vs Neural Strands (Rosu et al.)**: 提供了本文使用的发丝编码器-解码器先验。本文的贡献在于结合 3DGS 提升了重建精度和速度
- **启发**：strand-aligned Gaussians 的设计范式——将高斯基元绑定到结构化的几何基元上——可以推广到树木建模、管线重建等需要结构化表示的场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Strand-aligned Gaussians 是一种原创性很强的表示方法，巧妙统一了两种看似不兼容的范式
- 实验充分度: ⭐⭐⭐⭐ 真实和合成场景的评估完整，消融覆盖所有组件，但定量指标有限（主要是方向角误差）
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义精准，方法直觉清晰，图表质量极高
- 价值: ⭐⭐⭐⭐⭐ 首次实现可直接用于图形引擎的高质量发丝重建，对数字人和影视行业有直接价值

<!-- RELATED:START -->

## 相关论文

- [CGHair: Compact Gaussian Hair Reconstruction with Card Clustering](../../CVPR2026/3d_vision/cghair_compact_gaussian_hair_reconstruction_with_card_clustering.md)
- [Reconstruction and Simulation of Elastic Objects with Spring-Mass 3D Gaussians](reconstruction_and_simulation_of_elastic_objects_with_spring-mass_3d_gaussians.md)
- [WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors](../../ICCV2025/3d_vision/strandhead_text_to_hair-disentangled_3d_head_avatars_using_human-centric_priors.md)
- [Thermal3D-GS: Physics-induced 3D Gaussians for Thermal Infrared Novel-view Synthesis](thermal3d-gs_physics-induced_3d_gaussians_for_thermal_infrared_novel-view_synthe.md)

<!-- RELATED:END -->
