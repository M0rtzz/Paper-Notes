---
title: >-
  [论文解读] Arbitrary-Scale 3D Gaussian Super-Resolution
description: >-
  [AAAI 2026][3D视觉][3DGS] 提出Arbi-3DGSR集成框架，通过尺度感知渲染、生成先验引导优化和渐进超分三个核心组件，首次实现单个3DGS模型支持任意（包括非整数）倍率的高分辨率渲染，在×5.7倍率下PSNR比3DGS提升6.59dB，且保持85 FPS实时速度。
tags:
  - AAAI 2026
  - 3D视觉
  - 3DGS
  - 超分辨率
  - scale-aware rendering
  - generative prior
  - progressive training
---

# Arbitrary-Scale 3D Gaussian Super-Resolution

**会议**: AAAI 2026  
**arXiv**: [2508.16467](https://arxiv.org/abs/2508.16467)  
**代码**: https://github.com/huimin-zeng/Arbi-3DGSR  
**领域**: 3D视觉 / 3D高斯溅射 / 超分辨率  
**关键词**: 3DGS, arbitrary-scale super-resolution, scale-aware rendering, generative prior, progressive training

## 一句话总结
提出Arbi-3DGSR集成框架，通过尺度感知渲染、生成先验引导优化和渐进超分三个核心组件，首次实现单个3DGS模型支持任意（包括非整数）倍率的高分辨率渲染，在×5.7倍率下PSNR比3DGS提升6.59dB，且保持85 FPS实时速度。

## 研究背景与动机

**领域现状**：高分辨率新视角合成（HRNVS）要求从低分辨率稀疏视图重建3D模型并渲染HR视图。近年3DGS方法凭借显式点云表示实现了实时渲染加速，但现有3DGS超分方法（SuperGS、SRGS、GaussianSR等）只能处理固定整数倍率（如×2、×4），需要为不同倍率训练独立模型。

**现有痛点**：（1）固定倍率限制了灵活性，忽略了3D世界的内在连续性；（2）直接用vanilla 3DGS渲染任意倍率会产生混叠伪影，因为缺乏尺度感知能力；（3）在3DGS后级联2D超分器虽然可以支持任意倍率，但增加了框架复杂度且严重降低渲染效率（StableSR仅0.13 FPS）。

**核心矛盾**：任意倍率渲染需要同时解决三个相互关联的挑战——不同倍率下的抗锯齿渲染、无HR ground truth时的细节约束、以及跨尺度结构一致性——而现有方法最多只能处理其中一个。

**本文目标** 用单个3DGS模型实现任意倍率（包括1x到8x之间的非整数倍率如3.5x、5.7x）的高质量HR渲染，同时保持结构一致性和实时速度。

**切入角度**：作者观察到3DGS的高斯带宽和像素积分窗口都应随目标分辨率自适应调整——通过将scale factor注入渲染管线的两个关键环节（3D滤波和2D Mip滤波），可以实现抗锯齿的多尺度渲染。同时利用扩散模型的生成先验在latent space提供细节监督，避免显式HR supervision。

**核心 idea**：将scale factor作为一等公民注入3DGS渲染管线的3D滤波和2D Mip滤波中，配合生成先验的latent蒸馏和渐进式训练，单模型实现任意倍率超分。

## 方法详解

### 整体框架
输入是一组低分辨率视图，输出是任意目标倍率 $s$ 的高分辨率渲染结果。框架包含三个核心组件：尺度感知渲染（训练和推理阶段都使用）使3DGS能根据目标分辨率自适应调整渲染行为；生成先验引导优化（训练阶段）利用StableSR的去噪过程为HR渲染提供细节监督；渐进超分（训练阶段）将训练分为多个阶段逐步提升目标倍率以维持跨尺度一致性。

### 关键设计

1. **尺度感知渲染（Scale-Aware Rendering）**:

    - 功能：使同一组高斯基元能够根据目标分辨率自适应调整渲染行为，避免不同倍率下的混叠和模糊
    - 核心思路：分为3D和2D两级滤波。3D尺度感知平滑滤波将scale factor $s$ 引入最大采样率计算 $\hat{r}_i(s) = \max(\mathbb{I}_k(G_i^{3D}) \cdot f_k \cdot s_k / d_k)$，从而自适应约束高斯带宽。2D尺度感知Mip滤波将积分窗口大小设为 $\varepsilon_k = \varepsilon / s_k$，使像素着色的积分窗口与实际像素面积匹配。作者通过1D逼近误差分析证明，固定窗口在不同尺度下积累误差，而自适应窗口始终保持低误差
    - 设计动机：Mip-Splatting的原始滤波器使用固定参数，无法适应不同目标分辨率。高倍率渲染需要更窄的信号带宽和更小的积分窗口，低倍率则相反

2. **生成先验引导优化（Generative Prior-Guided Optimization）**:

    - 功能：在没有HR ground truth的情况下，利用预训练扩散模型（StableSR）提供纹理细节监督
    - 核心思路：包含两个子模块。（a）Latent蒸馏采样（LDS Loss）：对LR视图和当前SR渲染分别进行条件扩散过程，在latent space中计算异步时间步的噪声预测差异 $\nabla_\theta \mathcal{L}_{LDS} = \mathbb{E}_{\hat{n}}[w(\hat{n}) \cdot (\epsilon_\phi(z_{SR}^{\hat{n}}) - \epsilon_\phi(z_{LR}^n)) \cdot \partial I_{SR}^t / \partial \theta]$，让SR latent逼近具有丰富结构信息的LR latent。与SDS Loss不同，LDS比较的是异步latent的噪声差异而非同一时间步，这提供了结构监督的同时容忍了生成先验带来的像素级不对齐。（b）正交参考细化：选择场景中相互接近正交的视图子集，对这些视图执行完整去噪得到HR参考图，施加像素级纹理损失 $\mathcal{L}_{tex} = \mathbb{I}_{ortho} \cdot \|I_{SR}^t - I_{Ref}^t\|^2$
    - 设计动机：直接用生成HR参考做像素级监督会因为邻近视图的生成不一致性导致模糊和伪影。LDS在latent空间操作避免了像素级不对齐问题；正交视图策略确保参考图之间无重叠区域，避免冲突信息

3. **渐进超分（Progressive Super-Resolving）**:

    - 功能：将训练过程分为多个阶段逐步增大目标倍率，保证跨尺度结构一致性
    - 核心思路：训练分为×2→×4→×8三个阶段。每个阶段从上一阶段的高斯基元初始化，随机从已有的倍率集合中采样进行训练。阶段间施加结构损失 $\mathcal{L}_{str}$，将当前阶段HR渲染下采样后与上一阶段渲染结果对齐，使用MSE和D-SSIM的加权组合
    - 设计动机：直接用随机倍率混合训练（w/o PSR）会导致优化不稳定，小倍率和大倍率的需求相互冲突。渐进策略确保模型先学好低倍率细节再逐步扩展

### 损失函数 / 训练策略
总损失为三项加权和：$\mathcal{L} = \lambda_1 \mathcal{L}_{LDS} + \lambda_2 \mathcal{L}_{tex} + \lambda_3 \mathcal{L}_{str}$。训练在单张A6000 GPU上每场景约57分钟，内存占用约7GB。渲染阶段无额外计算开销。LR输入通过对原始图像做8倍Bicubic下采样得到，训练过程不使用原始HR图像。

## 实验关键数据

### 主实验

在4个基准数据集上与7种方法对比（Blender, Mip-NeRF360, Tanks&Temples, Deep Blending），评估整数和非整数倍率：

| 方法 | Blender ×4 PSNR↑ | Blender ×4 FID↓ | MipNeRF360 ×8 PSNR↑ | MipNeRF360 ×5.7 PSNR↑ | T&T ×4 PSNR↑ |
|------|-------------------|-----------------|---------------------|----------------------|--------------|
| 3DGS | 17.84 | 208.17 | 19.92 | 20.33 | 16.24 |
| Mip-Splatting | 22.25 | 109.44 | 24.51 | 25.02 | 20.97 |
| Analytic-Splatting | 23.57 | 141.30 | 23.04 | 23.41 | 19.42 |
| GaussianSR | 23.03 | 118.02 | 24.10 | 24.20 | 20.63 |
| **Ours** | **24.32** | **86.27** | **24.85** | **24.99** | **21.14** |

### 消融实验（Mip-NeRF360）
| 配置 | ×2 PSNR | ×4 PSNR | ×8 PSNR | ×2 FID |
|------|---------|---------|---------|--------|
| Full model | 26.23 | 25.18 | 24.85 | 36.52 |
| w/o 3D-SASF | 26.13 | 24.85 | 24.39 | 41.58 |
| w/o 2D-SAMF | 25.53 | 24.83 | 24.61 | 36.86 |
| w/o PSR | 26.03 | 24.51 | 23.91 | 37.92 |
| w/o GPO | 25.23 | 24.51 | 24.27 | 99.69 |
| Pseudo HR | 23.96 | 23.36 | 23.19 | 111.15 |
| SDS loss | 23.52 | 22.91 | 22.71 | 72.64 |

### 关键发现
- GPO贡献最大：去掉后PSNR在×2上下降1dB，FID从36.52暴增到99.69，说明生成先验对感知质量至关重要
- 渐进超分对高倍率影响显著：w/o PSR在×8上PSNR下降0.94dB
- LDS Loss远优于传统替代方案：Pseudo HR和SDS Loss分别导致PSNR下降2.27dB和2.71dB（×2）
- 效率优势明显：85 FPS vs StableSR的0.13 FPS（快908倍），存储仅0.79GB

## 亮点与洞察
- **任意倍率的统一模型**：首次将arbitrary-scale超分引入3DGS领域，单模型覆盖整数和非整数倍率；这个思路可迁移到NeRF或其他3D表示
- **LDS Loss设计精巧**：通过比较异步时间步的latent噪声而非像素差异，既利用了扩散模型的生成先验又避免了视图不一致性，比SDS Loss高2.71dB PSNR
- **正交视图策略**：用几何约束（正交视图无重叠区域）来解决生成一致性问题，是一个通用的多视图一致性保障思路

## 局限与展望
- 仅处理静态场景，未扩展到动态3DGS（如4D-GS）
- 生成先验依赖StableSR的预训练质量，在极高倍率（>×8）时可能引入非真实纹理
- 训练时间57min/场景，主要开销来自扩散模型推理，可探索更轻量的先验来源（如ESRGAN系列）
- 未探索跨场景泛化能力——每个场景仍需独立训练

## 相关工作与启发
- **vs GaussianSR**: GaussianSR虽然渲染速度更快（126 FPS），但使用随机倍率混合训练且无结构一致性约束，在所有指标上落后于本文方法。GaussianSR存储更小（0.56GB vs 0.79GB）但训练时间长4.5倍（256min vs 57min）
- **vs Mip-Splatting**: Mip-Splatting提供了抗锯齿基础但不涉及超分辨率，本文在其滤波器基础上引入scale factor实现了尺度感知能力。在Blender ×3.5上PSNR高2.13dB
- **vs Analytic-Splatting**: 理论上更精确的像素积分，但在实际HR渲染中产生高频伪影，本文在Mip-NeRF360上×8 PSNR高1.81dB

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次定义Arbi-3DGSR问题，尺度感知渲染和LDS Loss设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 4个基准、7种baseline、5种倍率、完整消融和效率分析
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，技术描述完整，公式推导严谨
- 价值: ⭐⭐⭐⭐ 实时+灵活倍率对3DGS实际部署有重要意义

<!-- RELATED:START -->

## 相关论文

- [IE-SRGS: An Internal-External Knowledge Fusion Framework for High-Fidelity 3D Gaussian Splatting Super-Resolution](ie-srgs_an_internal-external_knowledge_fusion_framework_for_high-fidelity_3d_gau.md)
- [SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](../../CVPR2026/3d_vision/sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [A3GS: Arbitrary Artistic Style into Arbitrary 3D Gaussian Splatting](../../ICCV2025/3d_vision/a3gs_arbitrary_artistic_style_into_arbitrary_3d_gaussian_spl.md)
- [3DSR: Bridging Diffusion Models and 3D Representations for 3D Consistent Super-Resolution](../../ICCV2025/3d_vision/bridging_diffusion_models_and_3d_representations_a_3d_consis.md)
- [SuperGaussian: Repurposing Video Models for 3D Super Resolution](../../ECCV2024/3d_vision/supergaussian_repurposing_video_models_for_3d_super_resolution.md)

<!-- RELATED:END -->
