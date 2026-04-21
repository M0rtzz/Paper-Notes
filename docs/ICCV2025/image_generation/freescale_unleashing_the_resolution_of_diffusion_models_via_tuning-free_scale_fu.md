---
title: >-
  [论文解读] FreeScale: Unleashing the Resolution of Diffusion Models via Tuning-Free Scale Fusion
description: >-
  [ICCV 2025][图像生成][高分辨率生成] 提出 FreeScale，一种无需训练的推理范式，通过尺度融合（Scale Fusion）机制从不同感受野尺度提取并融合信息（全局高频 + 局部低频），配合定制化级联上采样和约束膨胀卷积，首次在单张 A800 GPU 上实现了 8K 分辨率的文本到图像生成，同时支持高分辨率视频生成。
tags:
  - ICCV 2025
  - 图像生成
  - 高分辨率生成
  - 扩散模型
  - 无需训练
  - 尺度融合
  - 频率分解
---

# FreeScale: Unleashing the Resolution of Diffusion Models via Tuning-Free Scale Fusion

**会议**: ICCV 2025  
**arXiv**: [2412.09626](https://arxiv.org/abs/2412.09626)  
**代码**: http://haonanqiu.com/projects/FreeScale.html  
**领域**: 图像生成  
**关键词**: 高分辨率生成, 扩散模型, 无需训练, 尺度融合, 频率分解

## 一句话总结

提出 FreeScale，一种无需训练的推理范式，通过尺度融合（Scale Fusion）机制从不同感受野尺度提取并融合信息（全局高频 + 局部低频），配合定制化级联上采样和约束膨胀卷积，首次在单张 A800 GPU 上实现了 8K 分辨率的文本到图像生成，同时支持高分辨率视频生成。

## 研究背景与动机

视觉扩散模型通常在有限分辨率上训练（如 SDXL 训练分辨率为 1024²），直接在更高分辨率下推理会产生严重的重复模式（object repetition）问题。由于高分辨率数据稀缺且训练成本高昂，无需训练的高分辨率生成方法成为研究热点。

**现有方法的分层分析**：
- **SDXL 直接推理（DI）**：产生大量重复的完整物体，视觉结构完全丧失
- **ScaleCrafter**：通过膨胀卷积扩大感受野，解决了全局重复，但引入了局部重复（如多个眼睛/鼻子）
- **DemoFusion**：通过融合局部 patch 和全局 patch，几乎消除局部重复，但将多余信号转移到了背景，导致小物体重复
- **FouriScale**：在频率域中移除高频信号，完全消除所有重复，但暴力的频率编辑导致颜色和纹理异常

**核心矛盾**：当模型生成超出训练分辨率的内容时，高频信息不可避免地增加，导致误差累积产生各种类型的重复模式。现有方法各解决了部分问题，但都引入了新的副作用。

**核心 idea**：不同感受野尺度下的自注意力特征具有互补性——全局注意力能正确聚合高频信号的位置，局部注意力能增强局部细节质量。通过frequency-aware 的融合，可以取两者之长。

## 方法详解

### 整体框架

FreeScale 由三个组件构成：
1. **定制化级联上采样**：从训练分辨率开始，逐步通过加噪-去噪生成更高分辨率
2. **约束膨胀卷积**：仅在下采样块和中间块应用膨胀卷积，扩大感受野
3. **尺度融合**：在自注意力层中，分别计算全局和局部注意力，按频率分量进行融合

### 关键设计

1. **定制化级联上采样（Tailored Self-Cascade Upscaling）**：

    - 功能：提供合理的视觉结构作为高分辨率生成的基础
    - 核心思路：先用训练分辨率生成基础图像 $z_0^r$，经 VAE 解码、上采样到高分辨率后重新编码，再添加噪声到时间步 $K$ 后去噪：$\tilde{z}_K^{2r} \sim \mathcal{N}(\sqrt{\bar{\alpha}_K} \phi(z_0^r), \sqrt{1-\bar{\alpha}_K} \mathbf{I})$
    - 细节控制：引入缩放余弦衰减因子 $c = ((1 + \cos(\frac{T-t}{T}\pi))/2)^\alpha$ 混合中间 latent 和去噪 latent：$\hat{z}_t^r = c \times \tilde{z}_t^r + (1-c) \times z_t^r$。$\alpha$ 可以是空间变化的 2D 张量，允许对不同语义区域设置不同的细节级别
    - 设计动机：低分辨率中间结果决定了整体布局，高分辨率阶段只负责添加细节。RGB 空间上采样比 latent 空间上采样引入的模糊能抑制多余的高频信息（对图像有利，但对视频有害）

2. **约束膨胀卷积（Restrained Dilated Convolution）**：

    - 功能：扩大卷积感受野以避免物体重复
    - 核心思路：与 ScaleCrafter 不同，**仅在下采样块和中间块**应用膨胀卷积，**不在上采样块**中使用（因为上采样块的膨胀卷积会引入杂乱纹理）。而且在最后几个时间步恢复原始卷积（因为此时只渲染细节，结构已固定）
    - 设计动机：ScaleCrafter 在所有层应用膨胀卷积导致纹理问题。通过约束应用范围，保留了膨胀卷积消除重复的好处，同时避免了副作用

3. **尺度融合（Scale Fusion）**：

    - 功能：融合全局和局部自注意力的互补信息，消除所有类型的重复
    - 核心思路：在自注意力层中，同时计算全局注意力 $h_{out}^{global} = \text{SelfAttention}(h_{in})$ 和局部注意力（将 $h_{in}$ 切分为重叠的 patch，独立计算自注意力后重组）。然后通过高斯模糊 $G$ 进行频率分解融合：
   
    $h_{out}^{fusion} = \underbrace{h_{out}^{global} - G(h_{out}^{global})}_{\text{全局高频}} + \underbrace{G(h_{out}^{local})}_{\text{局部低频}}$
   
    - 设计动机：局部注意力的高质量细节对应低频语义（局部区域内的对象布局），但其高频信号会分散到错误位置导致小物体重复。全局注意力能正确定位高频信号的位置。因此，取全局的高频 + 局部的低频 = 最优结果

### 损失函数 / 训练策略

- 完全无需训练，仅涉及推理时的策略
- 高斯模糊 $G$ 的核大小控制高/低频的分界，是可调超参数
- 图像生成使用 RGB 空间上采样，视频生成使用 latent 空间上采样
- 所有实验在单张 A800 GPU 上完成
- 可进一步扩展到局部语义编辑（利用 1× 中间结果的语义掩码在交叉注意力中注入不同文本语义）

## 实验关键数据

### 主实验（SDXL 图像生成质量指标）

| 方法 | 分辨率 | FID↓ | KID↓ | FID_c↓ | KID_c↓ | IS↑ | 时间(min) |
|------|--------|------|------|--------|--------|-----|----------|
| SDXL-DI | 2048² | 64.31 | 0.008 | **31.04** | **0.004** | 10.42 | 0.648 |
| ScaleCrafter | 2048² | 67.55 | 0.013 | 60.15 | 0.020 | 11.40 | 0.653 |
| DemoFusion | 2048² | 65.86 | 0.016 | 63.00 | 0.024 | 13.28 | 1.441 |
| FouriScale | 2048² | 68.97 | 0.016 | 69.66 | 0.026 | 11.06 | 1.224 |
| **FreeScale** | **2048²** | **44.72** | **0.001** | 36.28 | 0.006 | 12.75 | 0.853 |
| SDXL-DI | 4096² | 134.08 | 0.044 | 42.38 | **0.009** | 7.04 | 5.456 |
| **FreeScale** | **4096²** | **49.80** | **0.004** | **71.37** | 0.029 | **12.57** | 6.240 |

### 消融实验（图像 4096²，各组件贡献）

| 配置 | FID↓ | KID↓ | FID_c↓ | KID_c↓ | IS↑ | 说明 |
|------|------|------|--------|--------|-----|------|
| 无 Scale Fusion | 68.12 | 0.012 | 100.07 | 0.037 | 12.42 | 局部重复明显 |
| 膨胀加在上采样块 | 67.45 | 0.011 | 98.56 | 0.035 | 12.54 | 杂乱纹理 |
| Latent 空间上采样 | 65.08 | 0.009 | 88.63 | 0.029 | 11.31 | 眼部等伪影 |
| **完整 FreeScale** | **49.80** | **0.004** | **71.37** | **0.029** | **12.57** | 全面最优 |

### 关键发现
- FreeScale 在 2048² 和 4096² 分辨率下的 FID 分别为 44.72 和 49.80，远优于所有基线（次优 >64）
- 尺度融合是最关键组件：移除后 FID 从 49.80 上升到 68.12（+37%）
- RGB 空间上采（vs latent 上采）对图像生成更有利（FID: 49.80 vs 65.08），但对视频则相反
- 约束膨胀卷积的位置至关重要：上采样块中的膨胀卷积引入杂乱纹理
- 视频生成同样有效：FreeScale 的 FVD (484.71) 大幅优于 DemoFusion (537.61) 和 ScaleCrafter (723.76)
- 推理时间与基线相当甚至更少：4096² 图像仅需 6.24 分钟

## 亮点与洞察
- "全局高频 + 局部低频"的频率分解融合思路极其优雅，用高斯模糊这样简单的操作就实现了两种尺度的最优互补
- 对高分辨率生成中各类重复模式的分层分析（全局重复 → 局部重复 → 小物体重复）逻辑清晰，解释了为什么单一方案无法解决所有问题
- 灵活的区域细节控制（空间变化的 $\alpha$）和局部语义编辑能力（Fig. 4-5）展示了实用价值
- 首次实现 8K 文本到图像生成是一个引人注目的里程碑

## 局限与展望
- 仅在 UNet 结构模型（SDXL、VideoCrafter2）上验证，DiT 架构模型（FLUX）的自注意力机制不同，可能需要调整
- 高斯模糊的核大小是固定的超参数，自适应选择可能带来进一步提升
- 8K 推理虽然在单 GPU 上可行，但时间开销仍然较大
- 尺度融合引入了额外的局部注意力计算（patch 切分和独立处理），限制了可扩展性
- 视频生成部分的基线方法有限（FouriScale 因不兼容未参与对比）

## 相关工作与启发
- ScaleCrafter → DemoFusion → FouriScale → FreeScale 构成了一条清晰的高分辨率生成方法演进线
- MultiDiffusion 的 patch 融合思想是 FreeScale 局部注意力的基础，FreeScale 的创新在于加入了频率感知的融合
- "不同尺度下的特征具有不同的频率特性"这一观察可以推广到其他多尺度处理任务

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-Resolution](../../CVPR2025/image_generation/faithdiff_unleashing_diffusion_priors_for_faithful_image_super-resolution.md)
- [FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model](freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)
- [Free4D: Tuning-free 4D Scene Generation with Spatial-Temporal Consistency](free4d_tuning-free_4d_scene_generation_with_spatial-temporal_consistency.md)
- [StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)
- [REPA-E: Unlocking VAE for End-to-End Tuning of Latent Diffusion Transformers](repae_unlocking_vae_for_endtoend_tuning_of_latent_diffusion.md)

<!-- RELATED:END -->
