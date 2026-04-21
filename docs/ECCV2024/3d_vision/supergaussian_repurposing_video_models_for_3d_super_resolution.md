---
title: >-
  [论文解读] SuperGaussian: Repurposing Video Models for 3D Super Resolution
description: >-
  [ECCV 2024][3D视觉][3D超分辨率] 提出SuperGaussian，通过复用预训练视频上采样模型实现3D超分辨率，无需类别特定训练，可处理多种3D输入格式（高斯、NeRF、网格等），输出高质量Gaussian Splat模型。
tags:
  - ECCV 2024
  - 3D视觉
  - 3D超分辨率
  - 视频上采样
  - Gaussian Splatting
  - 类别无关
  - 3D生成
---

# SuperGaussian: Repurposing Video Models for 3D Super Resolution

**会议**: ECCV 2024  
**arXiv**: [2406.00609](https://arxiv.org/abs/2406.00609)  
**代码**: [项目页面](https://supergaussian.github.io)  
**领域**: 3D视觉  
**关键词**: 3D超分辨率, 视频上采样, Gaussian Splatting, 类别无关, 3D生成

## 一句话总结

提出SuperGaussian，通过复用预训练视频上采样模型实现3D超分辨率，无需类别特定训练，可处理多种3D输入格式（高斯、NeRF、网格等），输出高质量Gaussian Splat模型。

## 研究背景与动机

当前3D生成模型的细节质量远不及图像和视频生成模型。主要原因：(1) 3D表示（体素网格、三平面）的分辨率受限；(2) 高质量3D训练数据稀缺（最大仅百万级 vs 图像十亿级）。关键观察是任何3D表示都可以渲染为视频，因此可以复用成熟的视频上采样模型来提升3D质量，同时利用视频的时间一致性来保证3D一致性。

## 方法详解

### 整体框架

两步流程：(1) 从低分辨率3D输入沿平滑轨迹渲染低分辨率视频，使用预训练视频上采样器进行4×上采样；(2) 对上采样后的视频进行3D Gaussian Splatting优化，得到高保真3D输出。

### 关键设计

**视频上采样先验**: 使用VideoGigaGAN作为视频上采样器，相比逐帧图像上采样，视频模型的时间一致性显著减少3D重建后的模糊问题。在MVImgNet数据集上微调视频上采样器以处理低分辨率高斯渲染的特有退化。

**领域适配微调**: 从MVImgNet生成低/高分辨率视频对——先将图像降采样至64×64，拟合低分辨率高斯，渲染作为输入；原始视频resize至256×256作为目标。使用Charbonnier回归损失+LPIPS感知损失+GAN损失联合微调。

**3D优化**: 使用标准Gaussian Splatting优化流程，2K步即可完成。已知相机参数直接提供，无需SfM估计。使用L1+SSIM损失。

### 损失函数

微调阶段：Charbonnier损失(权重10) + LPIPS损失(权重15) + GAN损失(权重0.05) + R1正则化

## 实验关键数据

### MVImgNet数据集对比

低分辨率Gaussian Splatting上采样（64→256px）：

| 方法 | LPIPS↓ | NIQE↓ | FID↓ | IS↑ |
|------|--------|-------|------|-----|
| Instruct-G2G | 0.1867 | 8.33 | 32.56 | 10.52 |
| Super-NeRF | 0.2204 | 8.84 | 37.54 | 10.40 |
| Pre-hoc image | 0.1524 | 7.65 | 27.04 | 11.27 |
| **SuperGaussian** | **0.1290** | **6.80** | **24.32** | **11.69** |

### Blender合成数据集对比

4×上采样（200→800px），使用TensoRF作为3D表示：

| 方法 | LPIPS↓ | PSNR↑ | SSIM↑ |
|------|--------|-------|-------|
| FastSR-NeRF | 0.075 | 30.47 | 0.944 |
| NeRF-SR | 0.076 | 28.46 | 0.921 |
| **SuperGaussian** | **0.067** | 28.44 | 0.923 |

### 关键发现

- 视频先验 vs 图像先验：视频上采样后3D重建更清晰，图像上采样因帧间不一致导致模糊
- 微调对严重退化输入（4K高斯或1K步NeRF）效果显著，甚至可恢复可读的中文字符
- 上采样轨迹越靠近目标物体效果越好
- 整个流程约141秒完成，是所有基线中效率最高的

## 亮点与洞察

1. **简洁而通用**的模块化设计：3D→视频→上采样→3D，每个环节可独立替换升级
2. 利用视频模型的时间一致性来弥补3D一致性缺失，比图像模型+各种一致性增强策略更简单有效
3. 类别无关、输入格式无关，可直接集成到现有3D工作流中

## 局限性

- 依赖预训练视频模型的泛化能力
- 无法恢复输入中缺失/遮挡的区域
- 视频上采样器的推理速度受限

## 相关工作与启发

将2D生成先验应用于3D是热门方向（如DreamFusion用图像扩散），本文首次系统性地证明视频先验优于图像先验用于3D超分。框架可随视频模型进步（如Sora）持续升级。

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实用性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](../../ICCV2025/3d_vision/bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)
- [LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation](lgm_large_multi-view_gaussian_model_for_high-resolution_3d_content_creation.md)
- [Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)
- [Arbitrary-Scale 3D Gaussian Super-Resolution](../../AAAI2026/3d_vision/arbitrary-scale_3d_gaussian_super-resolution.md)
- [Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](../../ICCV2025/3d_vision/repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)

<!-- RELATED:END -->
