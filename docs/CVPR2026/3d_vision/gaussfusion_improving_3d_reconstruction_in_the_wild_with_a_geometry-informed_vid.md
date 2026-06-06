---
title: >-
  [论文解读] GaussFusion: Improving 3D Reconstruction in the Wild with A Geometry-Informed Video Generator
description: >-
  [CVPR 2026][3D视觉][3D高斯溅射] 提出 GaussFusion，一个几何信息引导的视频到视频生成模型，通过渲染包含深度、法线、不透明度和协方差的 Gaussian Primitives Buffer（GP-Buffer）来条件化视频生成器，有效去除 3DGS 重建中的浮动伪影、闪烁和模糊…
tags:
  - "CVPR 2026"
  - "3D视觉"
  - "3D高斯溅射"
  - "视频生成模型"
  - "几何先验"
  - "伪影修复"
  - "实时推理"
---

# GaussFusion: Improving 3D Reconstruction in the Wild with A Geometry-Informed Video Generator

**会议**: CVPR 2026  
**arXiv**: [2603.25053](https://arxiv.org/abs/2603.25053)  
**代码**: 无  
**领域**: 3D视觉 / 新视角合成  
**关键词**: 3D高斯溅射, 视频生成模型, 几何先验, 伪影修复, 实时推理

## 一句话总结
提出 GaussFusion，一个几何信息引导的视频到视频生成模型，通过渲染包含深度、法线、不透明度和协方差的 Gaussian Primitives Buffer（GP-Buffer）来条件化视频生成器，有效去除 3DGS 重建中的浮动伪影、闪烁和模糊，且能同时适用于优化式和前馈式两种重建范式，蒸馏版本达到 16 FPS 实时推理。

## 研究背景与动机
1. **领域现状**：3D 高斯溅射（3DGS）已成为主流的 3D 重建表示方法，分为优化式（per-scene optimization）和前馈式（feed-forward prediction）两条技术路线。
2. **现有痛点**：两种范式在稀疏视角和覆盖不足场景下仍会产生严重伪影——浮动物体（floaters）、闪烁（flickering）、模糊（blur）和几何错误。现有修复方法（如 Difix3D、GenFusion、ExploreGS）仅基于 RGB 渲染进行条件化，无法处理大面积浮动物体和缺失区域；且通常只针对某一种重建范式训练，无法跨范式泛化。
3. **核心矛盾**：现有方法仅利用了高斯原语的颜色信息，忽略了深度、不透明度、法线、协方差等丰富的几何线索。同时，训练数据中缺乏多样化的伪影模拟，导致模型过拟合于特定重建流程。
4. **本文目标** 如何训练单一模型既能处理优化式 3DGS 的伪影又能处理前馈式 3DGS 的伪影？
5. **切入角度**：（1）将 3DGS 的全部原语属性编码为像素对齐的视频表示（GP-Buffer），提供比纯 RGB 更丰富的几何线索；（2）设计综合的伪影模拟 pipeline 覆盖多种退化模式。
6. **核心 idea**：用包含完整高斯原语几何信息的 GP-Buffer 来条件化视频生成模型，结合跨范式的伪影模拟策略实现通用的 3DGS 修复。

## 方法详解

### 整体框架
给定一个已有的 3DGS 重建 $\mathcal{G}$，首先沿新视角轨迹渲染 GP-Buffer（颜色、深度、法线、不透明度、协方差不确定性），然后将 GP-Buffer 编码并注入到基于 Wan-2.1 的流匹配视频生成器中，生成去伪影的高质量视频帧。生成的帧用于进一步优化 3D 重建。输入是多视图图像+相机参数，输出是修复后的 3DGS 表示。

### 关键设计

1. **Gaussian Primitives Buffer (GP-Buffer)**

    - 功能：将 3DGS 原语的完整多模态信息编码为像素对齐的视频表示
    - 核心思路：渲染 5 个通道——颜色 $\mathbf{C}$、不透明度 $A$、深度 $D$、法线 $\mathbf{N}$、几何不确定性 $\mathbf{U}$。法线通过有限差分从相机空间位置图获得 $\mathbf{N}(\mathbf{u}) = \text{normalize}(\partial_u \mathbf{P}_{\text{cam}} \times \partial_v \mathbf{P}_{\text{cam}})$。几何不确定性通过 alpha-blending 渲染逆协方差矩阵的唯一元素得到，低纹理区域用少量大高斯表示（低数值），高频区域则数值较高——这个通道提供了局部结构规整性的度量。
    - 设计动机：仅用 RGB 条件化时，模型难以区分正确的渲染和伪影（尤其是大面积缺失和几何错误）。GP-Buffer 的几何通道为模型提供了"看穿"伪影的能力。消融实验证明每增加一个几何模态都能持续提升性能。

2. **Geometry Adapter (GA)**

    - 功能：将编码后的 GP-Buffer 信息注入到视频生成器的 DiT 主干中
    - 核心思路：GP-Buffer 的 5 个模态分别通过 VAE 编码为视频潜变量，拼接后经 3D 卷积对齐空间和通道维度。GA 块作为 DiT 的并行侧网络，包含自注意力（处理几何特征）和交叉注意力（融合文本描述），产生几何感知特征 $\mathbf{x}_g$ 加到主流的视频潜变量上：$\mathbf{x} \leftarrow \mathbf{x} + \mathbf{x}_g$。训练时冻结基础模型，仅训练 GA 层。
    - 设计动机：直接将条件潜变量加到噪声潜变量上（如 GenFusion、ExploreGS 的做法）是次优的；GA 通过层级化的几何特征注入实现了更好的几何对齐，消融显示 PSNR 从 20.90 提升到 22.55。

3. **综合伪影模拟 Pipeline**

    - 功能：生成覆盖多种重建范式的训练数据
    - 核心思路：四种伪影来源策略——（1）稀疏视角模拟：随机保留 5% 帧（优于均匀下采样的策略），（2）多样化初始化：SfM、随机点云、MapAnything 密集点图，（3）配对重建：干净模型用全部视图+完整优化，退化模型用稀疏子集+减少优化步数，（4）前馈退化：直接渲染前馈模型（DepthSplat）的预测高斯——引入前馈特有的几何不一致和半透明伪影。总共生成 75K+ 配对视频样本。
    - 设计动机：之前方法只用均匀降采样+欠拟合来模拟伪影，这导致模型只能修复优化式 3DGS 的伪影。混合多种退化模式使模型获得跨范式泛化能力。

### 损失函数 / 训练策略
使用流匹配目标 $\mathcal{L} = \mathbb{E}[\|u_\theta(x_t, c, t) - v_t\|^2]$ 训练。采用两阶段微调策略实现高效推理：第一阶段用 Distribution Matching Distillation (DMD) 将多步生成器蒸馏为 4 步模型；第二阶段冻结蒸馏模型，仅微调 GA 层。基础模型 Wan-2.1-1.3B，GA 额外引入 0.6B 参数。8×H200 GPU 训练 100K 步。

## 实验关键数据

### 主实验（DL3DV 数据集，优化式 3DGS 修复）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | 推理速度 |
|------|--------|--------|---------|-------|---------|
| Splatfacto (baseline) | 17.42 | 0.605 | 0.412 | 6.49 | 118.3 FPS |
| GenFusion | 18.36 | 0.690 | 0.391 | 9.98 | 1.1 FPS |
| Difix3D+ | 20.10 | 0.765 | 0.302 | 4.22 | 12.8 FPS |
| ExploreGS | 20.69 | 0.760 | 0.345 | 6.27 | 1.2 FPS |
| **Ours (Full)** | **22.55** | **0.832** | **0.278** | **3.93** | 4.3 FPS |
| **Ours (Few-step)** | **22.49** | **0.842** | **0.288** | 7.38 | **15.1 FPS** |

### 消融实验（GP-Buffer 模态消融，DL3DV）

| RGB | Depth | Normal | Alpha | Cov. | PSNR ↑ | LPIPS ↓ | FID ↓ |
|-----|-------|--------|-------|------|--------|---------|-------|
| ✓ | | | | | 19.15 | 0.385 | 15.45 |
| ✓ | ✓ | | | | 19.29 | 0.361 | 10.54 |
| ✓ | ✓ | ✓ | | | 19.74 | 0.355 | 10.29 |
| ✓ | ✓ | ✓ | ✓ | | 19.96 | 0.344 | 8.61 |
| ✓ | ✓ | ✓ | ✓ | ✓ | **20.75** | **0.329** | **6.72** |

### 关键发现
- GP-Buffer 的每个几何模态都有独立贡献。协方差不确定性通道（Cov.）虽然被忽视，但带来了最大的 FID 改善（8.61→6.72）。
- 联合训练（混合多个数据集和退化类型）比单数据集训练效果更好，证明了跨范式伪影模拟的重要性。
- GaussFusion 在前馈模型 DepthSplat 上也能提升性能（PSNR 21.77→22.80），而 Difix3D+ 和 ExploreGS 反而降低了前馈模型的 PSNR。
- 蒸馏后的 4 步模型在 PSNR/SSIM/LPIPS 上几乎不降，但 FID 略高（3.93→7.38），实现了 16 FPS 的实时推理。
- Geometry Adapter 比直接加条件潜变量的方式在 PSNR 上高出 1.6 dB。

## 亮点与洞察
- **GP-Buffer 的设计非常有洞察力**：通过渲染高斯原语的完整属性（而不仅仅是颜色），为修复模型提供了"X光"般的透视能力。特别是协方差不确定性通道，让模型能识别出哪些区域是由少量大高斯覆盖的（即质量差的区域）。
- **范式无关的修复**：通过综合伪影模拟策略，单一模型可以同时处理优化式和前馈式 3DGS 的伪影——之前没有方法能做到这一点。这对实际部署非常重要。
- **蒸馏策略的实用性**：16 FPS 的实时推理速度使得 GaussFusion 可以在渲染时"即时"修复帧，而不需要离线处理。

## 局限与展望
- 作为视频生成模型，即使蒸馏后仍引入 0.6B 额外参数，对内存和计算的要求较高。
- 生成的帧在极端视角变化时可能丢失高频细节（蒸馏后 FID 升高）。
- 当前方法生成修复帧后仍需要重新优化 3DGS，整体流程不是端到端的。
- GP-Buffer 中的 VAE 编码器原本为 RGB 设计，虽然对其他模态的重建误差 <1%，但未来使用专门的多模态编码器可能更好。

## 相关工作与启发
- **vs Difix3D+**：基于图像扩散模型独立处理每帧，缺乏多视图一致性，无法去除大面积浮动伪影。GaussFusion 通过视频生成器保证了时间一致性。
- **vs MVSplat360**：专门为 MVSplat 前馈模型定制，无法泛化到优化式 3DGS。GaussFusion 通过混合训练实现了跨范式通用性。
- **vs ExploreGS / GenFusion**：条件化策略（仅 RGB）和训练数据不够多样，限制了修复能力和泛化性。GaussFusion 的 GP-Buffer 和综合伪影模拟解决了这两个问题。
- 这篇论文的核心启示是：在 3D 重建修复任务中，**充分利用重建本身的几何信息**比仅靠外部生成先验更有效。

## 评分
- 新颖性: ⭐⭐⭐⭐ GP-Buffer 的设计思路和范式无关训练策略都是有意义的创新
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多范式、多消融、速度对比，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构合理，动机阐述清晰
- 价值: ⭐⭐⭐⭐⭐ 实时推理+跨范式泛化使其在实际部署中有很高价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Text-to-3D by Stitching a Multi-view Reconstruction Network to a Video Generator](../../ICLR2026/3d_vision/text-to-3d_by_stitching_a_multi-view_reconstruction_network_to_a_video_generator.md)
- [\[CVPR 2026\] BulletGen: Improving 4D Reconstruction with Bullet-Time Generation](bulletgen_improving_4d_reconstruction_with_bullet-time_generation.md)
- [\[ICCV 2025\] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](../../ICCV2025/3d_vision/vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)
- [\[CVPR 2026\] Scene Grounding In the Wild](scene_grounding_in_the_wild.md)
- [\[CVPR 2026\] GaussianGrow: Geometry-aware Gaussian Growing from 3D Point Clouds with Text Guidance](gaussiangrow_geometry-aware_gaussian_growing_from_3d_point_clouds_with_text_guid.md)

</div>

<!-- RELATED:END -->
