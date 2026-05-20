---
title: >-
  [论文解读] MVPaint: Synchronized Multi-View Diffusion for Painting Anything 3D
description: >-
  [CVPR 2025][3D视觉][3D纹理生成] MVPaint 提出了一个三阶段的 3D 纹理生成框架——同步多视图生成 (SMG) + 空间感知 3D 补绘 (S3I) + UV 精炼 (UVR)…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D纹理生成"
  - "多视图一致性"
  - "UV精炼"
  - "3D补绘"
  - "扩散模型"
---

# MVPaint: Synchronized Multi-View Diffusion for Painting Anything 3D

**会议**: CVPR 2025  
**arXiv**: [2411.02336](https://arxiv.org/abs/2411.02336)  
**代码**: [http://mvpaint.github.io](http://mvpaint.github.io)  
**领域**: 3D视觉  
**关键词**: 3D纹理生成, 多视图一致性, UV精炼, 3D补绘, 扩散模型

## 一句话总结
MVPaint 提出了一个三阶段的 3D 纹理生成框架——同步多视图生成 (SMG) + 空间感知 3D 补绘 (S3I) + UV 精炼 (UVR)，通过在图像域而非 latent 域进行多视图同步、在 3D 点云空间而非 UV 空间进行补绘、以及空间感知的接缝平滑算法，在 Objaverse 和 GSO 两个 T2T 基准上全面超越 SOTA。

## 研究背景与动机

1. **领域现状**：3D 纹理生成（Text-to-Texture, T2T）是 3D 资产制作的关键步骤。现有方法主要分两类：(a) 迭代方法（TEXTure、Text2Tex）逐视角渲染深度图并用扩散模型生成纹理；(b) 同步方法（SyncMVD）在多视图 DDIM 过程中同步 UV 空间的 latent。

2. **现有痛点**：(a) **多视图不一致**：独立的多视角生成导致局部风格/纹理不连续；(b) **Janus 问题**：SyncMVD 的注意力复用限于邻近视图，经常产生多面怪现象；(c) **UV 依赖性强**：Paint3D、Meta 3D TextureGen 依赖连续的 UV 展开，当 UV atlas 被随机打包到 UV 图像中时产生纹理不连续；(d) **纹理过于平滑**缺乏细节。

3. **核心矛盾**：latent 空间（如 32×32）分辨率太低，难以建立与 UV 空间的精确映射，导致 latent 域同步效果差。同时，UV 展开会将 3D 相邻区域映射到 2D 不相邻区域，直接在 UV 空间补绘无法保证 3D 空间一致性。

4. **本文目标** (a) 如何在不依赖 UV 展开质量的情况下生成无缝、一致的 3D 纹理？(b) 如何高效地为未观察区域补绘纹理？(c) 如何在提升分辨率的同时修复 UV 接缝？

5. **切入角度**：作者提出在解码后的图像域（256×256）而非 latent 域（32×32）进行多视图同步，在 3D 点云空间而非 UV 空间进行补绘，在 3D 近邻结构中进行接缝修复。

6. **核心 idea**：将纹理生成的三大难题（一致性、完整性、平滑性）各自在最适合的空间中解决——图像域同步、3D 空间补绘、3D 近邻平滑。

## 方法详解

### 整体框架
三个阶段：Stage 1 (SMG) 使用 ControlNet 控制的 MVDream 模型生成低分辨率多视图图像，在去噪过程中通过 UV 空间在图像域做多视图同步，再用 SDXL I2I 模型精炼到 1K；Stage 2 (S3I) 将多视图图像投影到 UV 空间得到不完整 UV 贴图，将未涂色区域转化为 3D 点云补绘问题，通过空间感知颜色传播迭代补全；Stage 3 (UVR) 对 UV 贴图做超分（1K→2K），再用空间感知接缝平滑算法修复 UV 展开造成的纹理断裂。

### 关键设计

1. **同步多视图生成 (Synchronized Multi-view Generation, SMG)**:

    - 功能：生成多视图一致的初始纹理，避免 Janus 问题
    - 核心思路：用 MVDream 作为 T2MV 基础模型，训练 ControlNet 以深度/法线图引导。关键创新在于同步方式——不在 32×32 的 latent 空间同步（分辨率太低无法建立精确 UV 映射），而是在去噪过程中将中间 latent 解码为 256×256 图像，通过逆 UV 映射将各视图投影到 UV 空间（以视图方向与表面法线的余弦角为权重加权融合 $\mathbf{T}'_{\text{sync}} = \sum_i^N \cos(\mathbf{v}_i, \mathbf{n}_{uv}) \mathbf{T}'_i$），再从 UV 空间栅格化回各视图，编码为同步后的 latent 继续去噪。单次同步即可，多次反而不稳定
    - 设计动机：SVD等方法在 latent 域同步效果差是因为 latent 分辨率低。在 image 域做同步虽然多了一次 decode+encode，但映射更精确。MVDream 自带的多视图先验能有效避免 Janus 问题

2. **空间感知 3D 补绘 (Spatial-aware 3D Inpainting, S3I)**:

    - 功能：为 SMG 未观察到的区域完成纹理补全
    - 核心思路：将问题从 UV 空间转移到 3D 点云空间。从不完整 UV 贴图中，将有色像素生成带 RGB 的 3D 点 $\mathbf{P}_v$，无色像素生成零色的 3D 点 $\mathbf{P}_u$。通过空间感知颜色传播 (SCP) 算法迭代填色：每次迭代为每个无色点找 k 近邻有色点，根据欧氏距离 $d_j$ 和法线相似度 $\mathbf{n}_j \cdot \mathbf{n}_i$ 加权计算颜色值。法线相似度用分段函数 $f(x)$ 映射——法线差异大（$<0.5$）时权重极低（$10^{-8}$），近乎平行（$>0.9$）时权重提升到 10，确保颜色只在法线一致的曲面上传播
    - 设计动机：直接在 UV 空间补绘的问题是 3D 相邻区域在 UV 中可能被映射到不相邻位置，尤其当 UV atlas 高度碎片化时。在 3D 空间做补绘天然避免了这个问题，且不受 UV 展开结果的影响

3. **UV 精炼模块 (UV Refinement, UVR)**:

    - 功能：提升 UV 贴图分辨率到 2K 并修复 UV 展开造成的接缝
    - 核心思路：两步处理。(a) UV 超分：使用 Image-to-Image Upscale 扩散模型将 1K UV 贴图超分到 2K，增加纹理细节。(b) 空间感知接缝平滑 (SSA)：提取 UV 贴图中的有效区域二值图，通过连通性分析和边缘检测定位接缝掩码 $\mathbf{m}_{\text{seam}}$。将 UV 贴图重采样为 3D 带色点云，构建非接缝点的 kd-tree，用法线余弦相似度和距离加权重新着色接缝点颜色
    - 设计动机：UV 超分后接缝更加明显（因为超分模型不理解 UV 展开的不连续性）。在 3D 空间找近邻做颜色平滑可以修复这种 2D 空间中不连续但 3D 空间实际相邻的纹理断裂

### 损失函数 / 训练策略
- SMG 的 ControlNet 采用 ControlNet 标准训练方案在 102K Objaverse 样本上训练
- 8 视图均匀分布方位角，交替 ±30° 仰角
- S3I 和 SSA 是无学习的算法，不需要训练
- 文本标注由 CogVLM-2 描述3D物体的类别、纹理和外观，再由 LLM 提取关键词

## 实验关键数据

### 主实验

Objaverse T2T 基准:

| 方法 | FID↓ | KID↓ | CLIP↑ | 用户评分(整体)↑ | 一致性↑ |
|------|------|------|-------|----------------|---------|
| TEXTure | 28.03 | 7.60 | **20.30** | 3.81 | 3.31 |
| Paint3D | 25.28 | 5.19 | 19.27 | 3.85 | 3.51 |
| SyncMVD | 26.99 | 5.72 | 20.19 | 3.96 | 3.59 |
| **MVPaint** | **20.89** | **3.45** | 19.87 | **4.19** | **3.98** |

GSO T2T 基准（泛化性测试）:

| 方法 | FID↓ | KID↓ | 无缝性↑ | 一致性↑ |
|------|------|------|---------|---------|
| TEXTure | 24.76 | 5.50 | 3.97 | 3.65 |
| Paint3D | 37.29 | 10.24 | 3.14 | 3.45 |
| SyncMVD | 26.96 | 5.37 | 4.12 | 3.71 |
| **MVPaint** | **20.02** | **3.12** | **4.51** | **4.21** |

### 消融实验

| 配置 | FID↓ | KID↓ | CLIP↑ |
|------|------|------|-------|
| 完整 MVPaint | **20.89** | **3.45** | 19.87 |
| w/o MV Sync | 21.42 | 3.72 | 19.90 |
| w/o MV Diff（用单视图扩散替代MVDream） | 27.63 | 5.82 | 20.44 |
| w/o Geo. Refinement | 21.17 | 3.67 | 20.00 |
| w/o 3D Inpainting | 20.91 | 3.56 | 19.87 |
| w/o Seam Smoothing | 20.82 | 3.54 | 19.92 |

### 关键发现
- **多视图扩散模型 (MVDream) 的贡献最大**：去掉 MVDream 改用单视图扩散模型后 FID 恶化 6.74 点，是所有消融中影响最大的——多视图先验是避免 Janus 问题的关键
- **图像域同步有效但不是最关键**：去掉同步后 FID 仅恶化 0.53，说明 MVDream 本身的多视图一致性已经不错，同步是锦上添花
- **Paint3D 在 GSO 上严重退化**（FID 从 25.28 涨到 37.29），因为其子模块在 Objaverse 上训练过拟合。MVPaint 维持了良好的跨域泛化
- **无缝性评分差距巨大**：MVPaint 在 GSO 上无缝性评分 4.51 vs SyncMVD 的 4.12，印证了 SSA 算法的有效性

## 亮点与洞察
- **"在正确的空间解决正确的问题"**是本文最核心的设计哲学——同步在 image 域做（分辨率高）、补绘在 3D 空间做（保拓扑连续）、接缝修复在 3D 近邻做（跨越 UV 边界）。这种思路对任何涉及多空间表示的任务都有启发
- **S3I 的无学习补绘算法**令人印象深刻——仅靠 k 近邻 + 法线加权的简单颜色传播就能完成复杂几何体的纹理补全，无需训练任何模型。这种基于几何的 inpainting 方法在数据有限的场景中尤其有价值
- **法线分段加权函数 f(x)** 的设计很精巧——法线夹角小于 0.5（大约 60°）时几乎不传播颜色，大于 0.9（约 26°）时传播权重提升 10 倍，有效防止颜色跨曲面泄露。这个 trick 可迁移到点云着色、3D 语义传播等任务

## 局限与展望
- 需要训练 ControlNet 控制 MVDream，额外增加了训练负担
- 8 视图的初始生成分辨率较低（256×256），限制了初始纹理的精度
- S3I 的迭代颜色传播是渐进扩散式的，对于大面积未观察区域可能产生过于平滑的结果
- CLIP score 不是最优的（TEXTure 更高），因为 TEXTure 的 Janus 问题反而可能提高某些角度的 CLIP 匹配
- 未探索视频扩散模型作为多视图先验的可能性

## 相关工作与启发
- **vs TEXTure/Text2Tex**: 迭代式方法逐视角生成纹理，多视图不一致且速度慢。MVPaint 的 SMG 一次性生成多视图一致纹理
- **vs SyncMVD**: SyncMVD 在 latent 域同步但分辨率低，仍有 Janus 问题。MVPaint 在 image 域同步更精确，且使用 MVDream 从根本上避免 Janus
- **vs Paint3D**: Paint3D 依赖连续 UV 展开且泛化性差。MVPaint 的 S3I 和 SSA 在 3D 空间操作，不受 UV 展开质量影响

## 评分
- 新颖性: ⭐⭐⭐⭐ 三个模块各有创意，整体框架设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 两个基准测试+用户研究+完整消融
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰、图示直观、问题阐述到位
- 价值: ⭐⭐⭐⭐ 为 3D 纹理生成提供了工业可用的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Material Anything: Generating Materials for Any 3D Object via Diffusion](material_anything_generating_materials_for_any_3d_object_via_diffusion.md)
- [\[CVPR 2025\] Sharp-It: A Multi-view to Multi-view Diffusion Model for 3D Synthesis and Manipulation](sharp-it_a_multi-view_to_multi-view_diffusion_model_for_3d_synthesis_and_manipul.md)
- [\[CVPR 2025\] MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](zero-shot_novel_view_and_depth_synthesis_with_multi-view_geometric_diffusion.md)
- [\[CVPR 2025\] 3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement](3denhancer_consistent_multi-view_diffusion_for_3d_enhancement.md)
- [\[CVPR 2025\] PCDreamer: Point Cloud Completion Through Multi-view Diffusion Priors](pcdreamer_point_cloud_completion_through_multi-view_diffusion_priors.md)

</div>

<!-- RELATED:END -->
