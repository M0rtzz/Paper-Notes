---
title: >-
  [论文解读] Dual Prompting Image Restoration with Diffusion Transformers (DPIR)
description: >-
  [图像生成] 提出 DPIR，基于 SD3 (Diffusion Transformer) 的图像修复模型，通过轻量级低质量图像条件分支和全局-局部视觉双提示(dual prompting)分支，从多角度引入退化图像信息，首次系统性地将 DiT 应用于图像修复并取得 SOTA 性能。
tags:
  - 图像生成
---

# Dual Prompting Image Restoration with Diffusion Transformers (DPIR)

## 一句话总结

提出 DPIR，基于 SD3 (Diffusion Transformer) 的图像修复模型，通过轻量级低质量图像条件分支和全局-局部视觉双提示(dual prompting)分支，从多角度引入退化图像信息，首次系统性地将 DiT 应用于图像修复并取得 SOTA 性能。

## 研究背景与动机

图像修复(IR)领域中，基于扩散模型的方法(StableSR、SUPIR)已展现强大的生成能力，但它们均基于 U-Net 架构的 LDM。新一代扩散 Transformer (DiT) 如 SD3 凭借更好的可扩展性和长距离依赖建模能力，展现出更优的生成质量。

**关键挑战**：如何有效地将低质量(LQ)图像信息注入 DiT？
- **ControlNet 方案**：复制编码层，但针对 U-Net 设计，不适合由 ViT block 组成的 DiT
- **轻量适配器(T2I-Adapter/StableSR)**：性能中等
- **DiT 缺少 skip connection**：不像 U-Net 能通过跳跃连接在各层保持低质图像信息——LQ 信息在深层逐渐丢失
- **文本描述不足**：纯文本 prompt 无法完整描述图像的视觉特征(纹理、结构)

本文的核心观点：在图像修复中，应该用**视觉提示替代/补充文本提示**来引导 DiT，因为图像的细节信息远比文字描述丰富。

## 方法详解

### 整体框架

DPIR 基于 SD3，包含两个控制分支：(1) 轻量级 LQ 图像条件分支，通过少量卷积层+自适应特征对齐将 LQ 先验注入 DiT 第一层；(2) 双提示控制分支，提取全局+局部视觉特征替代 CLIP 文本嵌入，与 T5 文本提示拼接后作为 cross-attention 的条件输入每个 DiT block。

### 关键设计

#### 1. 退化鲁棒 VAE 编码器

- **功能**：将退化图像编码到 SD3 的 16 通道 latent 空间，使 latent 表示对退化具有鲁棒性
- **核心思路**：微调 SD3 的 VAE 编码器 $\mathcal{E}_{dr}$，添加 L1 + LPIPS + GAN 损失进行监督。GAN 损失防止 VAE 产生过于平滑的结果
- **设计动机**：SD3 原始 VAE 仅针对高质量图像训练，直接编码退化图像会产生不准确的 latent。16 通道 VAE 优于 SDXL 的 4 通道，提供更丰富的初始条件

#### 2. 轻量级 LQ 图像条件分支

- **功能**：高效地将 LQ 图像先验注入 DiT backbone
- **核心思路**：受 ControlNeXt 启发，使用少量卷积层提取 LQ 特征 $\mathcal{F}_c(z_{\text{LQ}})$，经自适应特征对齐 $\eta(\cdot; \mu, \sigma)$ 归一化后加到 DiT 第一层输出上。对齐函数使用 DiT 第一层输出的均值/方差进行归一化
- **设计动机**：避免 ControlNet 式重量级复制，仅在第一层注入条件即可。自适应对齐解决条件特征和主干特征的分布不匹配问题。可训练参数 $\phi_c \ll \theta_d$，保持高效

#### 3. 全局-局部双提示控制分支

- **功能**：提供丰富的视觉条件控制，替代 DiT 中的 CLIP 文本嵌入
- **核心思路**：将 LQ 图像送入两个 CLIP 图像编码器，提取局部视觉 token $c_{\text{local}}^{\text{vis}}$ 和 pooled 嵌入 $c_{\text{pool}}$ 替代原始 CLIP 文本嵌入。同时从周边区域裁剪全局 patch，提取全局视觉 token $c_{\text{global}}^{\text{vis}}$ 捕获上下文语义。全局+局部 token 与 T5 文本提示拼接形成双提示 $c_{\text{dual}}$，通过 cross-attention 注入每个 DiT block
- **设计动机**：DiT 缺少 U-Net 的 skip connection，仅靠轻量条件分支不足以保持 LQ 信息。视觉提示替换文本提示可以传达文本无法描述的结构和纹理信息。全局 patch 弥补了局部 patch 缺乏全局语义的问题，使视觉 token 更接近原始文本 token 的语义层次

### 损失函数

$$\mathcal{L}_{CFM} = \mathbb{E}_{t, p_t(z|\epsilon), p(\epsilon)} \left[\|v_\theta(z_t, t) - (\epsilon - x_0)\|^2\right]$$

VAE 微调损失：$\|\mathcal{D}(\mathcal{E}_{dr}(x_{\text{LQ}})) - x_{\text{HQ}}\|_1 + \alpha \mathcal{L}_{lpips} + \beta \mathcal{L}_{GAN}$

## 实验关键数据

### 主实验表 (4×超分辨率)

**DIV2K-Val 数据集**：

| 方法 | PSNR↑ | LPIPS↓ | DISTS↓ | CLIPIQA↑ | MUSIQ↑ |
|------|-------|--------|--------|----------|--------|
| Real-ESRGAN | 22.62 | 0.3982 | 0.2240 | 0.5661 | 63.90 |
| StableSR | 22.87 | 0.3925 | 0.2085 | 0.4974 | 57.28 |
| SinSR | 22.10 | 0.4416 | 0.2160 | 0.6919 | 65.13 |
| SUPIR | 21.23 | 0.4152 | 0.1873 | 0.5239 | 66.49 |
| **DPIR (ours)** | 21.61 | **0.3622** | **0.1677** | **0.7416** | **71.94** |

**RealSR 数据集**：

| 方法 | LPIPS↓ | DISTS↓ | CLIPIQA↑ | MUSIQ↑ |
|------|--------|--------|----------|--------|
| Real-ESRGAN | 0.2827 | 0.1936 | 0.5157 | 64.46 |
| SUPIR | 0.3996 | 0.2268 | 0.5223 | 58.68 |
| **DPIR (ours)** | **0.2641** | **0.1642** | **0.6625** | **69.28** |

### 消融表

- 文本提示 vs 双提示：双提示策略在所有感知指标上一致优于纯文本提示（见 Figure 1 定性对比）
- 全局-局部联合训练优于仅局部：全局 patch 提供上下文语义，改善修复的全局一致性

### 关键发现

- **感知质量指标大幅领先**：DPIR 在 CLIPIQA、MUSIQ、DISTS 等感知指标上全面超越所有 baseline
- **PSNR 非最优**：DPIR 的 PSNR 略低于 StableSR/Real-ESRGAN，体现了生成质量与像素保真度的权衡
- **DiT > U-Net**：SD3 的 DiT backbone 配合适当的条件注入，修复质量超越 SDXL-based 的 SUPIR
- **训练数据规模**：使用超过 2000 万高质量图像训练，充分利用 DiT 的可扩展性

## 亮点与洞察

1. **视觉提示替代文本提示**：核心创新——在修复任务中，用 CLIP 图像特征替换 CLIP 文本特征作为 DiT 的条件控制，比文本描述更精确地传达视觉信息
2. **全局-局部分层设计**：解决了高分辨率修复中局部 patch 缺乏全局语义的问题，巧妙对接预训练 DiT 中文本嵌入的功能角色
3. **DiT 适配方法论**：为如何将 DiT 用于生成以外的任务提供了系统性方案——轻量条件分支 + 提示替换

## 局限与展望

1. **PSNR 指标较低**：生成式修复的固有问题，过于追求感知质量可能牺牲像素级别的保真度
2. **推理速度**：DiT + 迭代去噪的推理成本高于 Real-ESRGAN 等单步方法
3. **CLIP 视觉编码器的局限**：CLIP 图像编码器对严重退化图像的特征提取质量可能下降
4. **仅验证 4× 超分**：未系统验证去噪、去模糊等其他修复任务
5. **训练成本高**：2000 万+ 图像 + DiT backbone 的联合训练，计算成本显著

## 相关工作与启发

- **SUPIR**：同样基于扩散模型做修复，使用 SDXL + LLaVA caption，但局限于 U-Net 和纯文本提示
- **ControlNeXt**：轻量条件注入方法，本文条件分支的设计基础
- **SD3/DiT**：展示了 Transformer 架构在生成任务上的优势，本文首次将其系统应用于修复
- **启发**：预训练 T2I 模型中的文本-图像对齐机制可以灵活替换——用视觉特征直接替代文本特征是更自然的修复条件注入方式

## 评分

⭐⭐⭐⭐

首次系统性将 DiT 引入图像修复，视觉双提示替代文本提示的设计直觉精准。感知质量指标 SOTA，但 PSNR 表现一般。方法实用性和工程完成度高。

<!-- RELATED:START -->

## 相关论文

- [Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_for_unified_image_generation_and_understanding.md)
- [DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](dual-interrelated_diffusion_model_for_few-shot_anomaly_image_generation.md)
- [EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](../../ICCV2025/image_generation/edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)
- [Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers](../../ICCV2025/image_generation/lay-your-scene_natural_scene_layout_generation_with_diffusion_transformers.md)
- [Multi-party Collaborative Attention Control for Image Customization](multi-party_collaborative_attention_control_for_image_customization.md)

<!-- RELATED:END -->
