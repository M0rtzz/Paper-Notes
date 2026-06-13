---
title: >-
  [论文解读] Volumetric Surfaces: Representing Fuzzy Geometries with Layered Meshes
description: >-
  [CVPR 2025][3D视觉][多层网格] 提出 Volumetric Surfaces 表示方法，通过学习自适应间距的多层半透明 SDF 网格壳（k-SDF），以固定顺序光栅化渲染，实现在低功耗笔记本和智能手机上对毛发等模糊几何的实时高质量视图合成。 实时视图合成在移动设备上面临处理能力、内存和散热等严格约束…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "多层网格"
  - "体积表面"
  - "模糊几何"
  - "实时渲染"
  - "移动端渲染"
---

# Volumetric Surfaces: Representing Fuzzy Geometries with Layered Meshes

**会议**: CVPR 2025  
**arXiv**: [2409.02482](https://arxiv.org/abs/2409.02482)  
**代码**: [autonomousvision.github.io/volsurfs](https://autonomousvision.github.io/volsurfs)  
**领域**: 3D视觉 / 实时渲染  
**关键词**: 多层网格, 体积表面, 模糊几何, 实时渲染, 移动端渲染

## 一句话总结

提出 Volumetric Surfaces 表示方法，通过学习自适应间距的多层半透明 SDF 网格壳（k-SDF），以固定顺序光栅化渲染，实现在低功耗笔记本和智能手机上对毛发等模糊几何的实时高质量视图合成。

## 研究背景与动机

实时视图合成在移动设备上面临处理能力、内存和散热等严格约束。现有方法可分为两大范式，各有局限：

1. **基于表面的方法（如 BakedSDF, MobileNeRF, BOG）**：每条光线仅需一个采样点，速度快，但无法准确建模毛发、毛绒等模糊几何（fuzzy geometry），因为所有外观信息被压缩在单个表面点上
2. **基于体积的方法（如 3DGS, SMERF）**：通过多点采样渲染能很好表示模糊材质，但存在三个性能问题——(P1) 每条光线需要大量（几十到上百个）采样点，(P2) 体积渲染需要跳过空白空间需额外数据结构（增加内存带宽），(P3) splatting 需要按距离排序原语（在 GPGPU 能力有限的平台上难以高效实现）

核心洞察：**纹理壳（Textured Shells）**是计算机图形学中模拟模糊表面的经典技术——使用多层同心半透明网格。本文将这一思路与可微渲染结合，学习自适应间距的多层 SDF 壳，实现 (P1) 有限且有界的采样点数（3-9个），(P2) 通过光栅化高效找到采样位置，(P3) 固定顺序渲染，无需排序。

## 方法详解

### 整体框架

两阶段训练+烘焙 pipeline：(1) 隐式表面阶段——先训练标准 NeuS 模型获得主表面 SDF，然后初始化并训练 k-SDF 模型（k 个自适应间距的 SDF 壳 + 视角依赖透明度场 + 颜色场）；(2) 烘焙阶段——将 k 个 SDF 提取为轻量网格（Marching Cubes + 网格简化至原始面数的 0.02%），生成 UV atlas，训练神经 SH 纹理并烘焙为 PNG 图像。最终在 WebGL 渲染器中以固定顺序光栅化所有网格。

### 关键设计

1. **k-SDF 表示**:
    - 功能：将 k 个表面建模为围绕主 SDF 的壳层，保证层间不相交且可按固定顺序遍历
    - 核心思路：一个主 SDF $d$ 加上 $k-1$ 个偏移场 $\{o_2, ..., o_k\}$，每个表面的 SDF 为 $d_i = d + o_i$。偏移通过对预测的相对偏移 $\hat{o}_i$ 做累积和（正方向和负方向分别累积）得到绝对偏移，确保层间顺序。每个表面配备视角依赖透明度 $\alpha(\mathbf{x}, \mathbf{v})$。渲染公式为 $k$ 个表面的固定顺序 alpha blending：$\mathcal{R}(\mathbf{r}) = \sum_{i=1}^k \mathcal{C}_i \mathcal{A}_i w_i$
    - 设计动机：壳层结构保证从外到内的固定遍历顺序，完全避免了 3DGS 的排序问题。自适应间距（通过学习偏移）比均匀间距更有效地利用有限层数

2. **训练策略（两阶段 + β调度）**:
    - 功能：确保稳定训练并最终得到尖锐的、可烘焙为网格的 SDF
    - 核心思路：第一阶段训练标准 NeuS 模型 100k iterations，β 从大（模糊密度）指数调度到小（尖锐密度），获得稳定的主表面。第二阶段用该主表面初始化 k-SDF（其余 $k-1$ 个壳以均匀间距 $\Delta o = (1/\beta_2)\pi/\sqrt{3}$ 分布），继续训练 50k iterations 至密度完全尖锐化。训练时使用 Eikonal 损失 + 曲率平滑损失确保 SDF 质量。占据网格（256³）用于空间跳跃，多表面层次化重要性采样用于高效的体积渲染
    - 设计动机：直接从头训练 k-SDF 会产生全透明的退化解。预训练的不透明主表面作为锚点初始化防止此问题。所有壳初始化在主表面内侧效果最佳（增加模型容量）

3. **混合分辨率神经纹理 + 烘焙**:
    - 功能：将隐式表示转换为可在移动设备上实时渲染的轻量显式资产
    - 核心思路：每个网格简化到约 2MB，生成 UV atlas 后训练每表面的神经 SH 纹理。关键创新是混合分辨率设计——基础颜色使用最高分辨率（2048²），高阶 SH 系数使用低分辨率（256²），大幅减少内存（每网格约 14MB vs 全分辨率 0.5GB）。训练时模拟 OpenGL 双线性插值（在纹素中心预测并插值），使训练结果与实时渲染器精确匹配。最终烘焙为 PNG 图像（Sigmoid 压缩 + 量化到 [0,255]）
    - 设计动机：全分辨率存储不现实，而高阶 SH 系数的空间变化通常较低频，可用低分辨率无损表示

### 损失函数 / 训练策略

两阶段训练损失：$\mathcal{L} = \mathcal{L}_c + \lambda_e\mathcal{L}_e + \lambda_s\mathcal{L}_s$（$\lambda_e=0.04$, $\lambda_s=0.65$）。$\mathcal{L}_c$ 为逐像素 L1 颜色损失，$\mathcal{L}_e$ 为 Eikonal 损失（约束 SDF 梯度模为 1），$\mathcal{L}_s$ 为曲率平滑损失（推向平滑解以便烘焙为轻量网格）。纹理阶段训练 15k iterations，仅用 L1 颜色损失。

透明度衰减：对支持表面，透明度乘以角度依赖权重 $\alpha_w = 2 \cdot \text{Sigmoid}(10 \cdot |\mathbf{v} \cdot \mathbf{n}|) - 1$，在掠射角处衰减为 0，避免边界处的硬切割。

## 实验关键数据

### 主实验（Shelly 数据集，模糊几何）

| 方法 | PSNR↑ | 智能手机FPS↑ | 笔电FPS↑ | 存储(MB)↓ |
|------|-------|-------------|---------|----------|
| MobileNeRF | 29.30 | 24 | 35 | 194 |
| 3DGS-50K | 32.73 | 20 | 160 | 12 |
| 3DGS | 35.44 | 8 | 18 | 57 |
| PermutoSDF | 29.85 | - | - | - |
| **Ours (3-Mesh)** | 33.39 | **65** | 145 | 46 |
| **Ours (5-Mesh)** | 34.25 | 55 | 90 | 77 |
| **Ours (7-Mesh)** | **34.50** | 42 | 70 | 110 |
| **Ours (9-Mesh)** | 34.38 | 35 | 55 | 140 |

### 跨数据集评估

| 数据集 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|-------|-------|--------|
| Custom（毛绒） | 3DGS | 37.34 | 0.982 | 0.147 |
| Custom | Ours (7-Mesh) | 35.63 | 0.977 | 0.169 |
| DTU (83,105) | 3DGS | 38.06 | 0.989 | 0.086 |
| DTU | Ours (9-Mesh) | 37.17 | 0.987 | 0.083 |

### 消融实验

| 配置 | PSNR↑ | 说明 |
|------|-------|------|
| 1-SDF (NeuS baseline) | ~29.85 | 单表面，模糊几何失败 |
| k-SDF 外+内初始化 | 较低 | 外壳利用率低 |
| k-SDF 全内初始化 | 更高 | 增加模型容量 |
| 5-SDF vs 5-Mesh | Mesh更高 | 固定几何+纹理优化更稳定 |
| 无透明度衰减 | 边界硬切割 | 角度依赖衰减更平滑 |
| 混合分辨率 vs 全2048² | 混合略优 | 低阶更精细高阶无需高分辨 |

### 关键发现

- 7 层网格是最佳平衡点：质量（34.50 dB）、速度（智能手机 42 FPS）和存储（110 MB）都合理
- 在智能手机上，3DGS（8 FPS）无法实时，而本文方法（42 FPS）轻松达到 30 FPS 实时标准
- 本文方法的 PSNR 与 3DGS（35.44 dB）差距约 1 dB，但渲染速度快 5× 以上
- 自适应壳间距会在实体结构处聚集、在模糊区域保持较大间距，自动适应场景
- 9 层网格反而质量下降，因为深层表面对像素颜色贡献小，梯度弱，优化慢

## 亮点与洞察

- **问题定义精准**：明确指出体积渲染的三个性能瓶颈（采样数、空间跳跃、排序）并逐一解决
- **图形学经典概念的现代化**：将纹理壳概念与可微渲染结合，学习自适应间距而非均匀间距
- **k-SDF 的壳约束设计**巧妙——通过累积和保证层间顺序，无需排序即可固定顺序 alpha blending
- **混合分辨率纹理**策略很实用——SH 系数按阶分辨率，高效利用存储
- **WebGL 部署**：最终资产（网格+PNG纹理）可直接在浏览器中渲染，跨平台兼容性好

## 局限与展望

- 图像质量仍低于 3DGS 约 1 dB，在需要最高质量的场景下有取舍
- 9 层以上的扩展受限于梯度消失问题
- 目前主要在物体级场景验证，大规模室外场景的适用性未充分探索
- 网格简化比例极高（0.02%），对某些精细几何可能过度简化
- 未来可探索与更高效的 SH 表示或其他视角依赖模型的结合

## 相关工作与启发

- **vs 3DGS**: 3DGS 质量更高但排序开销使其在移动设备上无法实时；本文无排序设计在低端设备上快 5×+
- **vs MobileNeRF**: MobileNeRF 也面向移动端但仅单表面，无法处理模糊几何；本文多层设计提升 5 dB
- **vs AdaptiveShells**: AdaptiveShells 用单 SDF + 空间变化核 + 体积渲染；本文用多 SDF + 光栅化采样，更适合移动端
- **vs GaussianShellMaps**: 也用多层网格但壳间距固定且用 splatting，仍无法在低端手机上实时

## 评分

- 新颖性: ⭐⭐⭐⭐ k-SDF 壳表示 + 自适应间距学习 + 混合分辨率纹理是系统性创新
- 实验充分度: ⭐⭐⭐⭐ 在多个数据集上与多种基线对比，速度/质量/存储的三维分析全面
- 写作质量: ⭐⭐⭐⭐⭐ 论文结构优秀，问题动机清晰，公式推导严谨，图示直观
- 价值: ⭐⭐⭐⭐⭐ 解决了模糊几何在移动端实时渲染的关键问题，有直接的工业应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] LUCAS: Layered Universal Codec Avatars](lucas_layered_universal_codec_avatars.md)
- [\[CVPR 2025\] Dynamic Neural Surfaces for Elastic 4D Shape Representation and Analysis](dynamic_neural_surfaces_for_elastic_4d_shape_representation_and_analysis.md)
- [\[CVPR 2025\] SimAvatar: Simulation-Ready Avatars with Layered Hair and Clothing](simavatar_simulation-ready_avatars_with_layered_hair_and_clothing.md)
- [\[CVPR 2025\] MeshArt: Generating Articulated Meshes with Structure-Guided Transformers](meshart_generating_articulated_meshes_with_structure-guided_transformers.md)
- [\[CVPR 2025\] Layered Motion Fusion: Lifting Motion Segmentation to 3D in Egocentric Videos](layered_motion_fusion_lifting_motion_segmentation_to_3d_in_egocentric_videos.md)

</div>

<!-- RELATED:END -->
