---
title: >-
  [论文解读] XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution
description: >-
  [ECCV 2024][图像生成] XPSR 利用多模态大语言模型（MLLM）提取高层和低层语义先验，通过 Semantic-Fusion Attention 和 Degradation-Free Constraint 引导扩散模型实现高保真、高真实感的图像超分辨率。
tags:
  - ECCV 2024
  - 图像生成
---

# XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution

**会议**: ECCV 2024  
**arXiv**: [2403.05049](https://arxiv.org/abs/2403.05049)  
**领域**: 图像生成

## 一句话总结

XPSR 利用多模态大语言模型（MLLM）提取高层和低层语义先验，通过 Semantic-Fusion Attention 和 Degradation-Free Constraint 引导扩散模型实现高保真、高真实感的图像超分辨率。

## 研究背景与动机

扩散模型凭借强大的生成先验在图像超分辨率（ISR）领域获得了越来越多的关注，但面临两个根本挑战：

**语义信息不足**：低分辨率（LR）图像经历了严重退化，ISR 模型难以从中提取准确的语义信息。现有方法如 StableSR 和 DiffBIR 将提示词设为空，完全依赖 LR 图像本身提取语义；PASD 和 SeeSR 使用标签模型提取物体标签，但缺乏空间位置关系和场景理解等复杂信息

**退化建模不充分**：现有方法未能感知 LR 图像中的固有失真（如模糊、噪声、压缩伪影），而低层语义先验对建模退化过程、实现更清晰的恢复至关重要

XPSR 的核心洞察是：**高层语义先验提供丰富的物体和场景信息，低层语义先验帮助建模退化机制**，两者结合可以同时实现语义准确和细节丰富的超分辨率结果。

## 方法详解

### 整体框架

XPSR 分为两个阶段：

1. **语义先验生成**：利用 LLaVA（MLLM）分别生成高层描述（物体、场景、空间关系）和低层描述（清晰度、噪声、色彩、光照质量），再通过 CLIP 文本编码器获得两类 embedding
2. **条件图像恢复**：以 Stable Diffusion v1.5 为骨干，ControlNet 作为控制器，通过 Semantic-Fusion Attention 融合多层级语义先验，同时在训练时附加 Degradation-Free Constraint

### 关键设计

**MLLM 语义提取**：
- 高层语义：使用指令 "Please provide a descriptive summary of the content of this image"，获取物体描述、空间位置、场景信息
- 低层语义：使用指令 "Please describe the quality of this image and evaluate it based on factors such as clarity, color, noise, and lighting"，获取质量、清晰度、噪声等退化描述

**Semantic-Fusion Attention (SFA)**：
- 简单的串行 cross-attention 会导致信息覆盖，因此设计了并行结构
- 高层和低层 embedding 分别通过各自的 cross-attention 分支处理
- 再通过融合 attention（Q 来自高层结果，K/V 来自低层结果）实现自适应特征选择：
$$x_{k+1} = \mathcal{CA}_f(\mathcal{CA}_h(x_k, c_h), \mathcal{CA}_l(x_k, c_l))$$
- UNet 侧仅使用高层 attention（因为 UNet 输入是噪声，不需要低层退化理解）

**Degradation-Free Constraint (DFC)**：
- 在 ControlNet 的图像编码器上添加像素空间约束：将中间特征映射为 RGB 图像，与下采样 HR 图像对齐
- 在 ControlNet 的 UNet 编码器上添加潜在空间约束：将特征映射到潜在空间与 HR 潜在表示对齐
$$\mathcal{L}_{DFC} = \sum_{i=1}^{3} \|x_{hr,i} - \hat{x}_i\|_1 + \sum_{j=1}^{3} \|z_{hr,j} - \hat{z}_j\|_1$$

### 损失函数

总体训练损失为扩散损失和 DFC 的加权和：

$$\mathcal{L} = \mathcal{L}_D + \lambda \mathcal{L}_{DFC}$$

其中 $\mathcal{L}_D$ 为标准去噪损失，$\lambda = 0.05$。推理时使用 classifier-free guidance（$\lambda_s = 5.5$），负提示为 "blurry, dotted, noise, unclear, low-res, over-smoothed"。

## 实验关键数据

### 主实验

**表1：合成数据集（DIV2K-Val）上与 SOTA 的比较**

| 方法 | PSNR↑ | SSIM↑ | FID↓ | MANIQA↑ | CLIPIQA↑ | MUSIQ↑ |
|------|-------|-------|------|---------|----------|--------|
| Real-ESRGAN | 24.30 | 0.6324 | 44.34 | 0.3756 | 0.5205 | 59.76 |
| StableSR | 23.26 | 0.5670 | 28.32 | 0.4173 | 0.6752 | 65.19 |
| DiffBIR | 23.49 | 0.5568 | 34.55 | 0.4598 | 0.6731 | 65.57 |
| PASD | 23.59 | 0.5899 | 39.74 | 0.4440 | 0.6573 | 66.58 |
| SeeSR | 23.56 | 0.5981 | 28.89 | 0.5046 | 0.6959 | 68.35 |
| **XPSR** | 22.80 | 0.5627 | 33.38 | **0.6080** | **0.7816** | **69.99** |

**表2：真实世界数据集（RealSR）上的比较**

| 方法 | PSNR↑ | MANIQA↑ | CLIPIQA↑ | MUSIQ↑ |
|------|-------|---------|----------|--------|
| Real-ESRGAN | 25.68 | 0.3736 | 0.4487 | 60.37 |
| StableSR | 24.69 | 0.4167 | 0.6200 | 65.25 |
| SeeSR | 25.31 | 0.5370 | 0.6638 | 69.56 |
| **XPSR** | 24.19 | **0.6059** | **0.7354** | **70.23** |

### 消融实验

**表3：语义先验类型消融（DRealSR / RealSR）**

| 高层 | 低层 | MANIQA↑ (DRealSR) | MUSIQ↑ (DRealSR) | MANIQA↑ (RealSR) | MUSIQ↑ (RealSR) |
|------|------|-------------------|------------------|-------------------|-----------------|
| ✗ | ✗ | 0.6078 | 67.31 | 0.6426 | 70.95 |
| ✗ | ✓ | 0.6213 | 68.73 | 0.6438 | 71.05 |
| ✓ | ✗ | 0.5505 | 65.79 | 0.5752 | 68.74 |
| ✓ | ✓ | 0.5713 | 67.84 | 0.6059 | 70.23 |

**表4：SFA 融合方式消融（DRealSR）**

| 融合方式 | SSIM↑ | MANIQA↑ | MUSIQ↑ |
|----------|-------|---------|--------|
| 仅高层 | 0.7298 | 0.5472 | 65.04 |
| 仅低层 | 0.7578 | 0.5034 | 63.97 |
| 串行连接 | 0.6834 | 0.5614 | 65.99 |
| **SFA（并行）** | **0.7220** | **0.5713** | **67.84** |

### 关键发现

1. XPSR 在 MANIQA、CLIPIQA、MUSIQ 等无参考质量指标上全面领先，在 DIV2K-Val 上 MANIQA  超越 SeeSR 10.34%
2. 扩散模型在参考指标（PSNR/SSIM）上落后 GAN 方法，但生成的图像在人眼感知上更真实——指标体系需更新
3. 高层先验的缺失导致保真度指标（SSIM, FID）大幅下降，低层先验的缺失导致质量指标（MANIQA, MUSIQ）恶化
4. 并行 SFA 优于串行连接，避免了信息覆盖问题

## 亮点与洞察

- **MLLM 赋能 ISR**：首次深入探索多模态大语言模型为扩散超分提供跨模态语义先验的价值，将 ISR 任务与 MLLM 的感知能力对齐
- **双层语义解耦**：清晰地将语义先验分为高层（内容理解）和低层（退化感知），并通过可视化实验直观展示了各层先验对生成结果的影响
- **DFC 的像素-潜在双空间约束**：在两个空间同时施加退化消除约束，强制 ControlNet 学习语义保持而非退化相关的特征

## 局限性

1. MLLM 推理本身带来额外计算开销，且 LLaVA 对严重退化图像的描述准确性有限
2. 在参考度量（PSNR/SSIM）上不如 GAN 方法，保真度和真实感之间仍存在权衡
3. 对真实世界退化的适应性仍受训练数据退化管道的覆盖范围限制
4. CLIP 文本编码器的 77 token 上限可能截断过长的 MLLM 生成描述

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)
- [OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis](rejection_sampling_imle_designing_priors_for_better_few-shot_image_synthesis.md)
- [Lazy Diffusion Transformer for Interactive Image Editing](lazy_diffusion_transformer_for_interactive_image_editing.md)

<!-- RELATED:END -->
