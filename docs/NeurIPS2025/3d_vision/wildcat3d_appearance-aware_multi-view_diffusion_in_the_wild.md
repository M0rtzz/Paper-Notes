---
title: >-
  [论文解读] WildCAT3D: Appearance-Aware Multi-View Diffusion in the Wild
description: >-
  [NeurIPS 2025][3D视觉][新视角合成] 提出WildCAT3D，通过显式建模图像的全局外观条件，扩展多视角扩散模型（CAT3D）从野外互联网数据（如旅游照片）中学习场景级新视角合成，同时支持外观控制生成。 新视角合成（NVS）近年来在物体级场景上取得了显著进展，特别是以CAT3D为代表的多视角扩散范式。然而…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "新视角合成"
  - "多视角扩散"
  - "外观建模"
  - "场景级NVS"
  - "野外数据"
---

# WildCAT3D: Appearance-Aware Multi-View Diffusion in the Wild

**会议**: NeurIPS 2025  
**arXiv**: [2506.13030](https://arxiv.org/abs/2506.13030)  
**代码**: [项目页面](https://wildcat3d.github.io)  
**领域**: 3D视觉  
**关键词**: 新视角合成, 多视角扩散, 外观建模, 场景级NVS, 野外数据

## 一句话总结

提出WildCAT3D，通过显式建模图像的全局外观条件，扩展多视角扩散模型（CAT3D）从野外互联网数据（如旅游照片）中学习场景级新视角合成，同时支持外观控制生成。

## 研究背景与动机

新视角合成（NVS）近年来在物体级场景上取得了显著进展，特别是以CAT3D为代表的多视角扩散范式。然而，**场景级NVS**仍面临重大挑战：

**核心矛盾**：

**多视角训练数据稀缺**：干净的多视角数据主要来自合成渲染或众包视频中的孤立物体，数据多样性和许可证都受限。

**野外数据丰富但不一致**：互联网上有大量场景图像（如旅游照片），但同一场景的不同照片在光照、天气、遮挡等方面差异很大，与现有多视角扩散架构不兼容。

**本文的核心insight**：不一致的数据可以通过显式解耦"内容"和"外观"来利于训练——在去噪目标视角时，模型能"偷看"每个视角的粗粒度外观信息（如天气、宽高比），但不泄漏精细细节。推理时则将源视角的外观嵌入复制到所有目标视角，确保生成的一致性。

## 方法详解

### 整体框架

WildCAT3D基于CAT3D框架扩展，核心改动有两处：（1）添加外观编码分支，（2）引入warp条件机制。总体输入形状为 $v \times (2k + d + 7) \times n \times n$，其中 $k+7$ 是CAT3D原始通道（latent+相机嵌入+二值掩码），$d$ 是外观嵌入通道，$k$ 是warp latent通道。模型建模分布 $p(\mathbf{I}^u | \mathbf{I}^o, \mathbf{c}^a, A_\phi(\mathbf{I}^a), \mathbf{w}^o)$。

### 关键设计

1. **可泛化外观编码器（Generalizable Appearance Encoder）**：一个轻量卷积网络，将图像latent压缩为低维向量（$d=8$维），作为信息瓶颈——编码天气、光照、宽高比等粗粒度全局外观，而无法泄漏精细图像内容。训练时该嵌入被复制到 $n \times n$ 空间位置并拼接到CAT3D输入通道，允许模型在训练中观察到所有视角（包括被加噪的未观测视角）的外观条件。编码器与去噪目标联合训练，在推理时可泛化到新场景。

2. **外观感知条件引导（Appearance-Aware CFG）**：直接对外观条件应用标准CFG会导致过饱和伪影（因为外观嵌入与图像亮度/色彩平衡相关）。本文设计了定制的CFG方案：在"无条件"设置中，保留外观条件但丢弃其他观测视角条件。即 $p^{(\text{uncond})}(\mathbf{I}^u | \mathbf{c}^u, A_\phi(\mathbf{I}^a))$，模型在条件和无条件设置中都能"偷看"所有视角的外观嵌入。

3. **外观条件推理**：推理时选取第一个观测视角的外观嵌入 $\mathbf{a}_0 = A_\phi(\mathbf{I}_0)$，复制到所有未观测视角的外观通道中。利用这一机制还可实现外观迁移——注入外部图像的外观嵌入，或通过CLIP文本检索找到匹配外观的图像进行文本驱动的外观控制。

4. **Warp条件机制（Warp Conditioning）**：解决单视角NVS固有的尺度模糊问题。用DepthAnything估计源视角深度，通过RANSAC对齐到COLMAP点云获得度量尺度，将源视角像素反投影到3D点云再渲染到每个目标相机视角。Warp图像的VAE latent作为额外条件通道注入，提示场景的正确放置位置。

### 损失函数 / 训练策略

- 先在CO3D和Re10K数据集上训练基础CAT3D模型
- 再在MegaScenes和CO3D上微调为完整WildCAT3D模型
- 使用与原始LDM相同的去噪损失
- 默认 $v=8$ 视角槽位，1个观测+7个未观测随机场景视角
- 推理时可增加到 $v=16$ 用于视频生成

## 实验关键数据

### 主实验（单视角NVS基准）

| 方法 | DTU PSNR↑ | DTU FID↓ | Mip-NeRF360 PSNR↑ | Mip-NeRF360 FID↓ |
|------|-----------|---------|-------------------|-----------------|
| ZeroNVS (released) | 5.799 | 160.0 | 6.999 | 137.0 |
| MS NVS | 8.795 | 85.96 | 14.06 | 64.41 |
| **WildCAT3D** | **10.77** | **57.32** | **14.77** | **42.17** |

| 方法 | Re10K PSNR↑ | Re10K FID↓ | MegaScenes PSNR↑ | MegaScenes FID↓ |
|------|-------------|-----------|-------------------|-----------------|
| MS NVS | 17.22 | 60.01 | 13.40 | 11.58 |
| **WildCAT3D** | **21.58** | **24.70** | **13.92** | **9.871** |

### 消融实验

| 配置 | DTU PSNR↑ | DTU FID↓ | MipNeRF PSNR↑ | MipNeRF FID↓ | 说明 |
|------|-----------|---------|---------------|-------------|------|
| WildCAT3D | **10.77** | **57.32** | **14.77** | **42.17** | 完整模型 |
| -warp | 9.795 | - | - | - | 去掉warp条件 |
| -warp-app | - | - | - | - | 去掉warp和外观 → 输出不一致 |

### 关键发现

- WildCAT3D在所有数据集上均超越先前SOTA，且训练使用的数据源**更少**
- 在分布外数据集（DTU物体级、Mip-NeRF 360场景级）上优势尤其显著
- 去掉warp条件后视角对齐明显变差，去掉外观建模后输出图像不一致
- 外观嵌入可自然聚类为有意义的组（如夜景、蓝天、室内等），证明编码器学到了有用表示
- 支持外观插值和文本控制的外观编辑等新应用

## 亮点与洞察

- **将"数据不一致"转化为优势**的思路非常优雅：通过显式建模外观变化，不一致的野外数据反而成为丰富的训练资源
- 外观编码器设计为信息瓶颈（仅8维），既够编码全局外观又不泄漏内容，精妙的工程选择
- 定制的CFG策略避免过饱和问题，体现了对扩散模型行为的深入理解
- Warp条件机制通过注入粗粒度几何约束解决尺度模糊，且不强制约束允许模型纠正深度错误

## 局限与展望

- 训练时依赖COLMAP的SfM点云进行深度对齐，对SfM失败的场景不适用
- 外观建模粒度较粗（全局向量），局部外观差异（如部分阴影）可能无法精确控制
- 目前仅展示了单视角输入的场景级NVS，多视角输入的场景效果未充分评估
- 生成质量仍受限于底层LDM的能力

## 相关工作与启发

- **CAT3D**：本文直接扩展的基线方法，多视角扩散模型的SOTA
- **MegaScenes**：大规模野外场景数据集，提供了训练数据和NVS基线
- **NeRF-W / Ha-NeRF**：此前在野外数据上建模外观变化的方法，但需要逐场景优化
- 启发：信息瓶颈式的外观编码可推广到其他多视角生成任务中处理数据不一致性

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次将野外旅游照片引入多视角扩散模型训练，外观解耦设计巧妙
- **实验充分度**: ⭐⭐⭐⭐ 多数据集评估+消融+应用展示全面
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，方法描述系统
- **价值**: ⭐⭐⭐⭐⭐ 解锁了丰富的野外数据用于场景级3D生成，开拓性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Sharp-It: A Multi-view to Multi-view Diffusion Model for 3D Synthesis and Manipulation](../../CVPR2025/3d_vision/sharp-it_a_multi-view_to_multi-view_diffusion_model_for_3d_synthesis_and_manipul.md)
- [\[CVPR 2025\] MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](../../CVPR2025/3d_vision/zero-shot_novel_view_and_depth_synthesis_with_multi-view_geometric_diffusion.md)
- [\[CVPR 2025\] EnvGS: Modeling View-Dependent Appearance with Environment Gaussian](../../CVPR2025/3d_vision/envgs_modeling_view-dependent_appearance_with_environment_gaussian.md)
- [\[NeurIPS 2025\] Scaffold Diffusion: Sparse Multi-Category Voxel Structure Generation with Discrete Diffusion](scaffold_diffusion_sparse_multi-category_voxel_structure_generation_with_discret.md)
- [\[CVPR 2025\] 3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement](../../CVPR2025/3d_vision/3denhancer_consistent_multi-view_diffusion_for_3d_enhancement.md)

</div>

<!-- RELATED:END -->
