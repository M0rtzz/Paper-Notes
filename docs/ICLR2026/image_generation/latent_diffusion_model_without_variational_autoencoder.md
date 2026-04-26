---
title: >-
  [论文解读] Latent Diffusion Model without Variational Autoencoder
description: >-
  [ICLR 2026][图像生成][自监督表征] 提出 SVG，用冻结的 DINOv3 自监督特征替代 VAE 潜在空间构建扩散模型，通过轻量残差编码器补充细粒度细节，实现更快训练、更高效推理和跨任务通用的视觉表征。
tags:
  - ICLR 2026
  - 图像生成
  - 自监督表征
  - DINOv3
  - 无VAE潜在扩散
  - 统一特征空间
  - 少步生成
---

# Latent Diffusion Model without Variational Autoencoder

**会议**: ICLR 2026  
**arXiv**: [2510.15301](https://arxiv.org/abs/2510.15301)  
**代码**: [GitHub](https://github.com/shiml20/SVG)  
**领域**: 扩散模型 / 视觉表征  
**关键词**: 自监督表征, DINOv3, 无VAE潜在扩散, 统一特征空间, 少步生成

## 一句话总结
提出 SVG，用冻结的 DINOv3 自监督特征替代 VAE 潜在空间构建扩散模型，通过轻量残差编码器补充细粒度细节，实现更快训练、更高效推理和跨任务通用的视觉表征。

## 研究背景与动机
- VAE+Diffusion 范式存在三大局限：训练/推理低效、少步采样质量差、VAE 特征缺乏语义判别性
- VAE 潜在空间语义纠缠严重（t-SNE 可视化显示不同类别高度混合），导致速度场方向矛盾，需要更多采样步骤
- 现有加速方法（REPA、VA-VAE）通过对齐 VFM 特征改善，但仅是治标，未根本改变潜在空间结构
- 假设：具有清晰语义判别性的潜在空间可大幅加速扩散训练

## 方法详解

### 整体框架
SVG 由三部分组成：冻结的 DINOv3 编码器 + 轻量残差编码器 → SVG 特征空间 → SVG Decoder 重建图像。扩散模型直接在 SVG 特征空间训练。

### 关键设计

1. **SVG 自编码器**: 冻结 DINOv3-ViT-S/16+ 编码器产生 $16 \times 16 \times 384$ 特征图（对 256×256 图像）。残差编码器（ViT）捕获 DINO 特征缺失的细粒度细节，输出拼接后形成完整 SVG 特征。残差分布通过批统计对齐到 DINO 特征分布：
$$\hat{F}_R = \frac{F_R - \mu(F_R)}{\sigma(F_R)} \cdot \sigma(F_D) + \mu(F_D)$$

2. **SVG Diffusion**: 不同于 VAE 的 $16 \times 16 \times 4$ 潜在空间，SVG 在 $16 \times 16 \times 384$ 高维特征空间上训练扩散模型。通常高维训练不稳定，但 SVG 特征的良好语义分散性使训练稳定高效。使用 SiT 设置的 flow matching 目标训练。

3. **语义分散性分析**: 通过 t-SNE 可视化和 toy example 展示：语义清晰分离的特征空间中，同类样本速度方向一致、异类方向分明，简化优化并减少采样步数。

### 损失函数 / 训练策略
两阶段训练：
- 阶段一：仅训练残差编码器和 SVG 解码器（重建损失 + 分布对齐），冻结 DINOv3
- 阶段二：训练 SVG Diffusion（SiT 设置，QK-Norm，per-channel 归一化）

## 实验关键数据

### 主实验（ImageNet 256×256）

| 方法 | Tokenizer | 训练Epoch | Steps | gFID w/o CFG | gFID w/ CFG |
|------|-----------|-----------|-------|-------------|-------------|
| DiT-XL | SD-VAE | 1400 | 250 | 9.62 | 2.27 |
| SiT-XL | SD-VAE | 1400 | 250 | 9.35 | 2.15 |
| REPA-XL | SD-VAE | 800 | 250 | 5.90 | 1.42 |
| SiT-XL (SD-VAE) | SD-VAE | 80 | **25** | 22.58 | 6.06 |
| SiT-XL (VA-VAE) | VA-VAE | 80 | **25** | 7.29 | 4.13 |
| **SVG-XL** | **SVGTok** | **80** | **25** | **6.57** | **3.54** |
| **SVG-XL** | **SVGTok** | **500** | **25** | **3.94** | **2.10** |

### 少步生成比较

| 方法 | Steps | FID w/o CFG | FID w/ CFG |
|------|-------|-------------|------------|
| SiT-XL (SD-VAE) | 5 | 69.38 | 29.48 |
| SiT-XL (VA-VAE) | 5 | 74.46 | 35.94 |
| **SVG-XL** | **5** | **12.26** | **9.03** |
| SiT-XL (SD-VAE) | 10 | 32.81 | 10.26 |
| **SVG-XL** | **10** | **9.39** | **6.49** |

### 关键发现
- 25 步 SVG-XL（80 epoch）FID=6.57，远优于同步数 SiT-XL 的 22.58
- 仅需 5 步即可达到 FID=12.26（SiT 需 250 步才能达到类似水平）
- SVG 特征空间保留了 DINOv3 的语义判别能力（线性探测准确率接近原始 DINO）
- 残差编码器对色彩和高频细节的重建至关重要
- DINOv3 在所有 VFM 中最适合作为统一特征空间

## 亮点与洞察
- 首次证明自监督特征可直接用于生成建模，打破 VAE 是潜在扩散唯一选择的定式
- 语义分散性→训练效率的因果关系分析很有洞察力（toy example 直观展示）
- 实现了生成、感知、理解任务通用的统一特征空间
- 5 步生成的超强性能展示了语义结构化潜在空间的降维效应

## 局限与展望
- 目前仅在 ImageNet 256×256 上验证，未扩展到文本引导生成或高分辨率
- SVG 特征维度高（384 vs VAE 的 4），内存开销更大
- 依赖特定的 DINOv3 模型，其他自监督方法（如 MAE、SigLIP）效果较差
- 重建质量（rFID=0.65）略逊于最优 VAE

## 相关工作与启发
- REPA、VA-VAE 等对齐方法启发了本工作，但 SVG 更根本地替换了特征空间
- 与 MAR 等自回归方法形成互补：SVG 为连续扩散提供了更优潜在空间
- 启发：未来视觉生成可能不再需要专门训练 VAE

## 技术细节补充
- DINOv3-ViT-S/16+ 编码器产生 $16 \times 16 \times 384$ 特征（vs SD-VAE 的 $16 \times 16 \times 4$）
- 残差编码器使用 ViT 架构（timm 库实现），与 DINOv3 特征通道拼接
- SVG Decoder 沿用 VA-VAE 的解码器架构设计
- SVG 特征空间做 per-channel 归一化以稳定高维扩散训练
- DiT 中 patch embedding 层替换为简单线性投影（384→模型维度）
- 隐状态通道数通常>384（如 DiT 中为 1152），因此 SVG 不导致推理低效
- 线性探测准确率：DINOv3 原始 86.4%，SVG（冻结 DINO 部分）85.2%，语义能力基本保留
- MAE 和 SigLIP 编码器的重建能力不足以支撑高质量生成
- SVG-XL 1400 epoch 25 步 FID=3.36 (w/o CFG) / 1.92 (w/ CFG)，接近 SOTA
- 支持模型尺度缩放：SVG-B(130M) 到 SVG-XL(675M) 均有效
- 通过分分析分辨检查任务证明 SVG 特征可用于感知和理解

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次去除 VAE 直接用自监督特征做扩散，思路新颖有说服力
- 实验充分度: ⭐⭐⭐⭐ 消融充分但缺少大尺度/文本引导实验
- 写作质量: ⭐⭐⭐⭐⭐ 动机分析透彻，可视化有力
- 价值: ⭐⭐⭐⭐⭐ 可能改变潜在扩散模型的设计范式

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Diffusion Alignment as Variational Expectation-Maximization](diffusion_alignment_as_variational_expectation-maximization.md)
- [\[ICLR 2026\] Purrception: Variational Flow Matching for Vector-Quantized Image Generation](purrception_variational_flow_matching_for_vector-quantized_image_generation.md)
- [\[AAAI 2026\] T-LoRA: Single Image Diffusion Model Customization Without Overfitting](../../AAAI2026/image_generation/t-lora_single_image_diffusion_model_customization_without_overfitting.md)
- [\[CVPR 2025\] Efficient Personalization of Quantized Diffusion Model without Backpropagation (ZOODiP)](../../CVPR2025/image_generation/efficient_personalization_of_quantized_diffusion_model_without_backpropagation.md)
- [\[ICLR 2026\] GLYPH-SR: Can We Achieve Both High-Quality Image Super-Resolution and High-Fidelity Text Recovery via VLM-Guided Latent Diffusion Model?](glyph-sr_can_we_achieve_both_high-quality_image_super-resolution_and_high-fideli.md)

<!-- RELATED:END -->
