---
title: >-
  [论文解读] GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting
description: >-
  [ICCV 2025][3D视觉][视线重定向] 提出GazeGaussian，首个基于3D高斯溅射（3DGS）的高保真视线重定向方法，通过双流3DGS模型分别建模面部和眼部区域，设计显式的高斯眼球旋转表示和表情引导神经渲染器，在视线精度、合成质量和渲染速度上全面超越现有方法。
tags:
  - ICCV 2025
  - 3D视觉
  - 视线重定向
  - 3D高斯溅射
  - 头部头像合成
  - 双流模型
  - 眼球旋转表示
---

# GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting

**会议**: ICCV 2025  
**arXiv**: [2411.12981](https://arxiv.org/abs/2411.12981)  
**代码**: [GitHub](https://ucwxb.github.io/GazeGaussian)  
**领域**: 3D视觉  
**关键词**: 视线重定向, 3D高斯溅射, 头部头像合成, 双流模型, 眼球旋转表示

## 一句话总结

提出GazeGaussian，首个基于3D高斯溅射（3DGS）的高保真视线重定向方法，通过双流3DGS模型分别建模面部和眼部区域，设计显式的高斯眼球旋转表示和表情引导神经渲染器，在视线精度、合成质量和渲染速度上全面超越现有方法。

## 研究背景与动机

视线估计（Gaze Estimation）是计算机视觉的基础任务，但现有估计器在面对分布外数据时泛化能力不足。视线重定向（Gaze Redirection）通过将输入图像的视线方向操控到目标方向来生成增强数据，从而提升视线估计器的泛化性。

**现有方法的问题**：

**2D方法**（STED等）：将视线重定向表述为2D图像操作，忽略了头部和视线操控的固有3D特性，导致空间一致性差、合成保真度有限

**NeRF-based方法**（GazeNeRF等）：计算开销大，渲染效率低，且通过隐式地旋转特征图来改变视线方向，控制不够精确

**3DGS-based头部方法**（Gaussian Head Avatar等）：忽略了精确的视线方向控制，无法在不同被试之间泛化

**核心动机**：3DGS的无结构特性天然适合对眼球进行显式的刚性旋转控制，但需要解决两个关键问题：(a) 如何将面部动画和视线运动解耦，(b) 如何在不同被试之间实现泛化。

## 方法详解

### 整体框架

GazeGaussian包含三个核心组件：
1. 使用预训练neutral mesh初始化双流3DGS（面部 + 眼部）
2. 通过面部变形场和眼球旋转场将canonical高斯变换到目标空间
3. 通过表情引导神经渲染器（EGNR）生成最终的视线重定向图像

前处理包括：背景去除、视线方向归一化、面部跟踪获取每帧的身份/表情编码和相机位姿。

### 关键设计

1. **双流高斯表示与面部变形（Face Deformation Branch）**：

   在canonical空间构建面部高斯 $\{\mu_0^f, z_0^f, R_0^f, S_0^f, \alpha_0^f\}$，其中 $z_0^f \in \mathbb{R}^{128}$ 为逐点特征向量。

   关键创新是基于距离的影响权重机制：
   - 计算每个高斯中心 $\mu$ 到3D面部landmarks的最小距离 $d$
   - 近landmark区域（$d < d_1 = 0.15$）主要受表情编码 $\tau$ 影响，$\lambda_\tau = 1$
   - 远区域（$d > d_2 = 0.25$）主要受头部位姿 $\gamma$ 影响，$\lambda_\tau = 0$
   - 过渡区域平滑插值：$\lambda_\tau = (d_2 - d)/(d_2 - d_1)$
   
   变形通过MLP实现：$\mu^f = \mu_0^f + \lambda_\tau E_\mu^f(\mu_0^f, \tau) + \lambda_\gamma P_\mu^f(\mu_0^f, \gamma)$

2. **高斯眼球旋转表示（Gaussian Eye Rotation）**：

   与面部分支不同，眼部分支的scaling被约束为球形 $S_0^e \in \mathbb{R}^{N \times 1}$，符合眼球的旋转特性。

   核心设计：先在canonical空间旋转眼部高斯，再加入表情编码生成变形偏移。由于视线标签含噪声，使用两个独立MLP预测旋转偏差：
   
   $$\mu^e = E_\mu^e(\mu_0^e, \tau) + G_\mu^e(\mu_0^e, \varphi) \mu_0^e$$
   
   其中 $\varphi$ 是归一化后的视线方向。这种显式旋转方式比GazeNeRF的隐式特征图旋转更精确，充分利用了3DGS的可控性。

3. **表情引导神经渲染器（Expression-Guided Neural Renderer, EGNR）**：

   为解决跨被试泛化问题，将表情潜在编码 $\tau$ 通过slice cross-attention注入到UNet渲染器的瓶颈特征中：
   
   $$z_b' = z_b + z_b \cdot \text{Attn}(q = \tau, k = z_b, v = z_b)$$
   
   这使渲染器能够感知被试特异信息，生成更真实的个性化面部细节。

### 损失函数 / 训练策略

**图像合成损失**：对面部、眼部和头部三个区域的渲染图像和特征图分别监督：

$$\mathcal{L}_\mathcal{I}^e = \|\mathcal{I}_{gt} - \mathcal{I}_e\|_1 + \lambda_{SSIM}(1 - SSIM(\mathcal{I}_{gt}, \mathcal{I}_e)) + \lambda_{VGG} VGG(\mathcal{I}_{gt}, \mathcal{I}_e)$$

其中 $\lambda_{SSIM} = \lambda_{VGG} = 0.1$。总图像损失含6项（3个渲染图 + 3个特征图的前3通道RGB）。

**视线重定向损失**：使用预训练视线估计器计算角度误差：

$$\mathcal{L}_\mathcal{G} = \mathcal{E}_{ang}(\psi^g(\mathcal{I}_h), \psi^g(\mathcal{I}_{gt}))$$

最终损失 $\mathcal{L} = 1.0 \cdot \mathcal{L}_\mathcal{I} + 0.1 \cdot \mathcal{L}_\mathcal{G}$。

## 实验关键数据

### 主实验（ETH-XGaze数据集内评估）

| 方法 | Gaze↓ | Head Pose↓ | SSIM↑ | PSNR↑ | LPIPS↓ | FID↓ | ID↑ | FPS↑ |
|------|-------|-----------|-------|-------|--------|------|-----|------|
| STED | 16.217 | 13.153 | 0.726 | 17.530 | 0.300 | 115.020 | 24.347 | 18 |
| HeadNeRF | 12.117 | 4.275 | 0.720 | 15.298 | 0.294 | 69.487 | 46.126 | 35 |
| GazeNeRF | 6.944 | 3.470 | 0.733 | 15.453 | 0.291 | 81.816 | 45.207 | 46 |
| Gaussian Head Avatar | 30.963 | 13.563 | 0.638 | 12.108 | 0.359 | 74.560 | 27.272 | 91 |
| **GazeGaussian** | **6.622** | **2.128** | **0.823** | **18.734** | **0.216** | **41.972** | **67.749** | **74** |

GazeGaussian在视线精度上略优于GazeNeRF（6.622° vs 6.944°），头部位姿精度提升38%（2.128° vs 3.470°），图像质量全面领先（PSNR提升3.28dB），FPS达74（GazeNeRF的1.6倍）。

### 消融实验

| Two-stream | Gaussian Eye Rep. | Expression-Guided | Gaze↓ | Head↓ | SSIM↑ | PSNR↑ | LPIPS↓ | FID↓ | ID↑ |
|:-:|:-:|:-:|-------|-------|-------|-------|--------|------|------|
| ✓ | | | 13.651 | 2.981 | 0.753 | 16.376 | 0.272 | 55.481 | 38.941 |
| ✓ | ✓ | | 13.489 | 3.149 | 0.751 | 16.365 | 0.274 | 54.327 | 41.521 |
| ✓ | | ✓ | 8.883 | 2.635 | - | - | - | - | - |
| ✓ | ✓ | ✓ | **6.622** | **2.128** | **0.823** | **18.734** | **0.216** | **41.972** | **67.749** |

三个组件缺一不可：双流模型提供面部-眼部解耦基础，高斯眼球旋转表示使视线误差从13.65°降到6.62°，表情引导渲染器大幅提升身份保持（ID从38.9→67.7）。

### 跨数据集泛化

| 方法 | Columbia Gaze↓ | Columbia ID↑ | MPII Gaze↓ | MPII ID↑ | GazeCapture Gaze↓ | GazeCapture ID↑ |
|------|---------------|-------------|-----------|---------|-------------------|----------------|
| GazeNeRF | 9.464 | 23.157 | 14.933 | 30.981 | 10.463 | 19.025 |
| GazeGaussian | **7.415** | **59.788** | **10.943** | **41.505** | **9.752** | **44.007** |

在三个跨数据集评估中，GazeGaussian均达到最佳视线精度和身份保持。

### 关键发现

1. Gaussian Head Avatar虽然渲染速度快（91FPS），但缺乏视线解耦机制导致视线重定向完全失败（30.96°误差）
2. GazeNeRF的隐式特征图旋转在极端视线方向下效果不佳
3. 显式的高斯眼球旋转在精度和可控性上显著优于隐式方法

## 亮点与洞察

- **首创3DGS视线重定向**：利用3DGS的无结构特性实现显式眼球旋转控制，概念简洁有效
- **双流解耦设计巧妙**：面部和眼部使用不同的变形/旋转策略，符合各自的物理运动特性
- **距离权重机制**：基于到landmark距离的影响权重平滑过渡表情和位姿控制，避免硬分界
- **表情引导渲染器提升泛化**：通过cross-attention注入被试信息，解决了3DGS-based方法通常面临的单人限制

## 局限与展望

- 训练仍需要14.4K图像，数据效率有提升空间
- 眼球旋转表示将scaling约束为球形，可能对非球形眼部结构建模不够精确
- 未考虑眼睛闭合、眨眼等更复杂的眼部动作
- 视线标签噪声通过MLP偏差补偿，但根本性的标签质量问题未解决

## 相关工作与启发

- Gaussian Head Avatar提供了3DGS头部建模的基础框架，本文在其基础上增加了视线控制能力
- GazeNeRF建立了双流面部-眼部解耦范式，本文将其从NeRF迁移到3DGS并增加显式控制
- 表情引导渲染器的cross-attention设计可推广到其他条件化渲染任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个3DGS-based视线重定向方法，显式眼球旋转设计独特
- **实验充分度**: ⭐⭐⭐⭐⭐ 四个数据集评估、完整消融、跨数据集泛化、下游任务验证
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，公式推导完整，可视化对比直观
- **价值**: ⭐⭐⭐⭐ 实用性强，视线重定向和数据增强有明确应用场景

<!-- RELATED:START -->

## 相关论文

- [RTGaze: Real-Time 3D-Aware Gaze Redirection from a Single Image](../../AAAI2026/3d_vision/rtgaze_real-time_3d-aware_gaze_redirection_from_a_single_image.md)
- [Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis](gaussian_variation_field_diffusion_for_high-fidelity_video-to-4d_synthesis.md)
- [PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors](../../NeurIPS2025/3d_vision/planargs_high-fidelity_indoor_3d_gaussian_splatting_guided_by_vision-language_pl.md)
- [SegmentDreamer: Towards High-Fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation](segmentdreamer_towards_high-fidelity_text-to-3d_synthesis_with_segmented_consist.md)
- [Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction](momentum-gs_momentum_gaussian_self-distillation_for_high-quality_large_scene_rec.md)

<!-- RELATED:END -->
