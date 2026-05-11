---
title: >-
  [论文解读] Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction
description: >-
  [ICCV 2025][3D视觉][3D Gaussian Splatting] 提出DiffusionGS，将3D高斯点云"烘焙"进扩散模型的去噪器中，实现单阶段、视图一致的单视图3D物体生成和场景重建，配合场景-物体混合训练策略和RPPC相机条件编码，在PSNR/FID上大幅超越现有方法，推理速度仅需约6秒。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D Gaussian Splatting
  - 扩散模型
  - 单视图3D生成
  - 场景重建
  - 混合训练
---

# Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction

**会议**: ICCV 2025  
**arXiv**: [2411.14384](https://arxiv.org/abs/2411.14384)  
**代码**: [https://caiyuanhao1998.github.io/project/DiffusionGS/](https://caiyuanhao1998.github.io/project/DiffusionGS/)  
**领域**: 三维视觉  
**关键词**: 3D Gaussian Splatting, 扩散模型, 单视图3D生成, 场景重建, 混合训练

## 一句话总结

提出DiffusionGS，将3D高斯点云"烘焙"进扩散模型的去噪器中，实现单阶段、视图一致的单视图3D物体生成和场景重建，配合场景-物体混合训练策略和RPPC相机条件编码，在PSNR/FID上大幅超越现有方法，推理速度仅需约6秒。

## 研究背景与动机

单视图Image-to-3D是一个重要且富有挑战的任务，现有方法存在三条技术路线及各自问题：

**两阶段方法（主流）**：先用2D扩散生成多视图图像，再用3D重建模型拟合。核心缺陷是2D扩散无法保证3D一致性，当提示视角方向变化时容易崩溃

**one-stage 3D扩散方法**：基于triplane-NeRF的3D扩散模型。但triplane分辨率受限，体渲染速度慢，难以扩展到大场景

**单视图场景重建**：依赖单目深度估计器，在严重遮挡或大视角变化时容易失效

此外，现有3D数据稀缺（场景级仅约90K样本），且物体和场景数据分布差异大（物体无背景+环绕相机，场景有密集背景+轨迹相机），直接混合训练可能不收敛。

## 方法详解

### 整体框架

DiffusionGS在每个扩散时间步 $t$ 直接预测像素对齐的3D高斯点云 $\mathcal{G}_\theta$，而非预测噪声 $\epsilon$。这些高斯点云通过可微光栅化渲染为多视图图像进行2D监督。推理使用30步DDIM采样。

输入：1张干净条件视图 $\mathbf{x}_{con}$ + $N$ 张噪声视图 $\mathcal{X}_t$ + 对应的视角条件
输出：$(N+1) \times H \times W$ 个像素对齐高斯基元

### 关键设计

1. **像素对齐3D高斯扩散去噪器**：

    - 每个像素预测一个高斯基元 $G_t^{(k)}(\boldsymbol{\mu}, \boldsymbol{\Sigma}, \alpha, \boldsymbol{c})$，共 $N_g = (N+1)HW$ 个
    - 高斯中心位于像素对齐射线上：$\boldsymbol{\mu}_t^{(k)} = \boldsymbol{o}^{(k)} + u_t^{(k)} \boldsymbol{d}^{(k)}$
    - 深度通过近远端线性插值：$u_t^{(k)} = w_t^{(k)} u_{near} + (1 - w_t^{(k)}) u_{far}$
    - 去噪器为Transformer架构，每个block含MSA + MLP + LN，通过adaLN注入时间步条件
    - 输出通过高斯解码器映射为14通道的per-pixel高斯参数图
    - 采用 $x_0$-prediction（而非 $\epsilon$-prediction），确保每步生成干净的纹理和完整的3D结构

2. **场景-物体混合训练策略**：

    - **视角选择约束**：对噪声视图和条件视图的位置角 $\theta_{cd}^{(i)} \leq \theta_1$ 及方向角 $\cos(\varphi_1)$ 设限，保证视图有足够重叠
    - **双高斯解码器**：物体级和场景级使用独立的MLP解码器，具有不同的深度范围（物体[0.1, 4.2]，场景[0, 500]），微调阶段移除一个
    - **分布对齐**：控制相机条件、高斯点云分布和成像深度的一致性

3. **Reference-Point Plücker Coordinate (RPPC)**：

    - 传统Plücker坐标 $\boldsymbol{r} = (\boldsymbol{o} \times \boldsymbol{d}, \boldsymbol{d})$ 的力矩向量在感知深度和3D几何方面存在局限
    - RPPC用射线上距世界坐标原点最近的点替代力矩向量：
    $\boldsymbol{r} = (\boldsymbol{o} - (\boldsymbol{o} \cdot \boldsymbol{d})\boldsymbol{d}, \boldsymbol{d})$
    - 满足4D光场的平移不变性假设
    - 参考点直接编码射线位置和相对深度，通过skip connection流经每个Transformer block

### 损失函数 / 训练策略

去噪损失由L2损失和VGG-19感知损失加权组成：
$$\mathcal{L}_{de} = \mathcal{L}_2(\hat{\mathcal{X}}_{(0,t)}, \mathcal{X}_0) + \lambda \cdot \mathcal{L}_{VGG}(\hat{\mathcal{X}}_{(0,t)}, \mathcal{X}_0)$$

新视图损失 $\mathcal{L}_{nv}$ 结构同上。点分布损失 $\mathcal{L}_{pd}$ 用于训练warm-up，将高斯点云分布正则化到目标标准差 $\sigma_0=0.5$ 附近。

总训练目标：
$$\mathcal{L} = (\mathcal{L}_{de} + \mathcal{L}_{nv}) \cdot \mathbf{1}_{iter>iter_0} + \mathcal{L}_{pd} \cdot \mathbf{1}_{iter \leq iter_0} \cdot \mathbf{1}_{object}$$

训练流程：
- 混合训练：32×A100，Objaverse+MVImgNet+RealEstate10K+DL3DV10K，40K iterations
- 分别微调：64×A100，物体80K/场景54K iterations
- 高分辨率微调：256→512分辨率，20K iterations

## 实验关键数据

### 主实验（单视图物体生成）

| 方法 | ABO PSNR↑ | ABO FID↓ | GSO PSNR↑ | GSO FID↓ | 推理时间 |
|------|-----------|----------|-----------|----------|----------|
| LGM | 16.01 | 86.32 | 14.27 | 75.55 | 4.1s |
| GS-LRM | 18.78 | 123.55 | 17.70 | 112.96 | - |
| DMV3D | 23.69 | 32.28 | 20.82 | 33.48 | 31.4s |
| **DiffusionGS** | **25.89** | **9.03** | **22.07** | **11.52** | **5.8s** |

单视图场景重建（RealEstate10K）：

| 方法 | PSNR↑ | FID↓ | LPIPS↓ |
|------|-------|------|--------|
| PixelNeRF | 17.46 | 159.52 | 0.5525 |
| Splatter-Image | 18.21 | 120.35 | 0.4839 |
| Flash3D | 20.29 | 35.03 | 0.3610 |
| **DiffusionGS** | **21.63** | **15.87** | **0.2743** |

### 消融实验（GSO数据集）

| 配置 | PSNR↑ | SSIM↑ | FID↓ | 说明 |
|------|-------|-------|------|------|
| Baseline（无时间步控制） | 17.63 | 0.7928 | 118.31 | 起点 |
| + DiffusionGS框架 | 20.57 | 0.8120 | 47.86 | +2.94dB |
| + 点分布损失 $\mathcal{L}_{pd}$ | 20.94 | 0.8423 | 28.41 | +0.37dB |
| + 混合训练 | 21.73 | 0.8515 | 17.79 | +0.79dB |
| + RPPC | **22.07** | **0.8545** | **11.52** | +0.34dB |

用户研究（25人，6分制）：DiffusionGS得分4.88，大幅领先LGM(3.04)、DMV3D(3.16)、12345++(3.81)

### 关键发现

- **3D一致性是核心优势**：每步预测3D高斯消除了2D多视图扩散的视图不对齐问题，即使提示视图不是正面也能保持几何正确
- **混合训练提升纹理真实感**：场景数据引入真实世界纹理先验，减少合成数据训练导致的不真实纹理
- **RPPC改善深度感知**：相比传统Plücker坐标，RPPC在场景重建上提升0.28dB PSNR和7.09 FID
- **无需深度估计器**：通过沿相机轨迹生成多视图预测更精细的高斯点云，在遮挡区域表现优于依赖深度估计的方法
- 与SOTA 2D方法PhotoNVS+post-hoc GS相比，DiffusionGS在6s内完成PhotoNVS需要2478s的工作，且PSNR高出6.32dB

## 亮点与洞察

- "将3DGS烘焙进扩散去噪器"是一个优雅的设计，用像素对齐高斯巧妙解决了3D表示与扩散框架的兼容问题
- $x_0$-prediction而非$\epsilon$-prediction的选择有道理：噪声高斯没有纹理信息，会破坏视图一致性
- 混合训练的视角约束设计实用，两个角度限制分别控制位置和朝向，保证训练稳定
- RPPC的设计直觉清晰：参考点比力矩向量更直接地编码了射线在3D空间中的位置

## 局限与展望

- 训练资源需求巨大（32-64×A100），限制了学术团队的复现
- 场景级数据仅约90K样本且视角变化小，更大规模和多样化的数据可能进一步提升
- 像素对齐高斯的数量与分辨率线性相关，高分辨率下内存和计算压力大
- 当前仅支持固定近远端深度范围，面对深度变化极大的场景可能受限
- 文本到3D依赖外部2D生成器（Stable Diffusion/FLUX/Sora），端到端方案可能更优

## 相关工作与启发

- 相比DMV3D（triplane-NeRF扩散），DiffusionGS用3DGS替代NeRF，速度快5倍+且支持场景级任务
- 与Flash3D/VistaDream的核心区别在于不依赖单目深度估计器，通过扩散过程本身学习几何
- 混合训练策略可扩展至更多3D数据源（室内扫描、自动驾驶数据等）

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 将3DGS融入扩散去噪器、RPPC设计和混合训练策略都是创新贡献
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖物体/场景生成+重建、多数据集、用户研究、详尽消融
- **写作质量**: ⭐⭐⭐⭐ Pipeline图清晰，公式推导完整，但符号较多
- **价值**: ⭐⭐⭐⭐⭐ 统一物体生成和场景重建的框架，6秒推理速度极具实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion](sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)
- [\[ICCV 2025\] A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision](a_lesson_in_splats_teacher-guided_diffusion_for_3d_gaussian_splats_generation_wi.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[ICCV 2025\] CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image](catsplat_contextaware_transformer_with_spatial_guidance_for.md)
- [\[ICCV 2025\] WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions](wonderplay_dynamic_3d_scene_generation_from_a_single_image_and_actions.md)

</div>

<!-- RELATED:END -->
