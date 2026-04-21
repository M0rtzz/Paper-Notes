---
title: >-
  [论文解读] Spectral Image Tokenizer
description: >-
  [ICCV 2025][图像生成][image tokenizer] 提出 Spectral Image Tokenizer (SIT)，用离散小波变换 (DWT) 将图像从空域转换到频域后再进行 token 化，使 token 序列天然地按"粗到细"排列，从而支持多分辨率重建、渐进式生成、文本引导上采样与编辑等传统 raster-scan tokenizer 无法实现的能力。
tags:
  - ICCV 2025
  - 图像生成
  - image tokenizer
  - discrete wavelet transform
  - autoregressive generation
  - coarse-to-fine
  - multiscale
  - VQ-VAE
---

# Spectral Image Tokenizer

**会议**: ICCV 2025  
**arXiv**: [2412.09607](https://arxiv.org/abs/2412.09607)  
**作者**: Carlos Esteves, Mohammed Suhail, Ameesh Makadia (Google Research)
**领域**: image_generation  
**关键词**: image tokenizer, discrete wavelet transform, autoregressive generation, coarse-to-fine, multiscale, VQ-VAE

## 一句话总结

提出 Spectral Image Tokenizer (SIT)，用离散小波变换 (DWT) 将图像从空域转换到频域后再进行 token 化，使 token 序列天然地按"粗到细"排列，从而支持多分辨率重建、渐进式生成、文本引导上采样与编辑等传统 raster-scan tokenizer 无法实现的能力。

## 研究背景与动机

### 现有方法的核心问题

当前主流的图像 tokenizer（如 VQGAN、ViT-VQGAN）将图像切分为空间 patch，按光栅扫描 (raster scan) 顺序排列成 token 序列。这种方式存在几个根本性问题：

**序列顺序不自然**：raster scan 从左上到右下逐行扫描，自回归模型在预测下一个 token 时，"已知信息"是图像上半部分的部分重建——这既不符合人类视觉感知（先看整体后看细节），也给条件建模带来困难。

**分辨率固定**：传统 ViT-based tokenizer 的 patch 大小固定，分辨率变化会导致序列长度剧烈变化（分辨率翻倍，序列长度变为 4 倍），训练和推理代价急剧上升。

**无法渐进解码**：生成过程中，前 50% 的 token 只能重建图像的上半部分，无法得到完整图像的粗略预览。

**缺乏频域先验利用**：自然图像的能量主要集中在低频，高频细节天然具有更高的可压缩性，但空域 tokenizer 对所有 patch 一视同仁。

### 本文出发点

作者观察到小波变换的**多尺度分解**特性与自回归生成天然契合：低频近似系数对应图像的粗略版本，逐级高频细节系数对应越来越精细的纹理。如果将 DWT 系数按从低频到高频的顺序编码为 token，则自回归生成本质上就是"先生成整体轮廓，再逐步补充细节"。

## 方法详解

### 整体框架

SIT 的流程为：**输入图像 → Haar DWT 多级分解 → 分尺度 patch 化 → 分尺度线性嵌入 → Transformer 编码器 → 向量量化(双码本) → Transformer 解码器 → 分尺度反投影 → IDWT 重建**。

生成模型 AR-SIT 在 SIT 之上添加一个自回归 Transformer，逐 token 预测量化离散码，最终通过 SIT 解码器还原图像。

### 关键设计

#### 1. 频域 Patch 化 (Spectral Patchification)

对输入图像施加 L 级 Haar DWT，得到 1 组近似系数 (approximation) 和 L 组细节系数 (horizontal/vertical/diagonal details)。定义 S = L + 1 个尺度，每个尺度固定使用 N 个 token（实验中 N = 256）：

- **第 1 尺度**（近似）：对最粗的低频近似系数切分为 N 个 patch，每个 patch 大小约 32×32×3
- **第 s 尺度**（细节）：将对应的 H/V/D 三个方向的细节系数在通道维度拼接，再切分为 N 个 patch

由于高频尺度的系数空间分辨率更大，相同数量的 token 意味着更大的 patch → **高频被更强地压缩**，符合自然图像的频谱特性。

对比 ViT-VQGAN：256×256 图像用 8×8 patch 得到 1024 token；SIT 用 4 尺度 × 256 token/尺度 = 同样 1024 token，但分辨率增加到 512×512 时，只需增加 1-2 个尺度（+256/+512 token），而非 4 倍增长。

#### 2. Approximation-Detail Transformer (ADTransformer)

由于近似系数和细节系数的分布截然不同（近似类似自然图像，细节接近零均值高斯），作者在 Transformer 的内部层中使用**分尺度参数**：

- Layer Norm 和 MLP 使用不同的参数处理近似 token 和细节 token
- Self-attention 的 QKV 投影在所有尺度间共享（以保持跨尺度交互）
- 参数量几乎不变（MLP/LN 参数远小于 attention）

#### 3. Scale-Causal Attention

引入**分尺度因果注意力掩码**：第 s 尺度的 token 只能 attend 到第 1 到第 s 尺度的所有 token（块三角矩阵形式）。这一设计的关键作用：

- **编码器 SC**：允许对不同分辨率的输入编码（低分辨率只激活前几个尺度）
- **解码器 SC**：允许对部分 token 序列解码为粗略图像（渐进解码）
- 可分别应用于编码器/解码器以支持不同应用：
    - 多尺度重建：编码器 + 解码器均 SC
    - 粗到细生成：仅解码器 SC (SIT-SCD)
    - 图像上采样：仅编码器 SC (SIT-SCE)
    - 图像编辑：仅编码器 SC

#### 4. 双码本量化

编码器输出的 token 特征经过向量量化映射到离散码本：

- **近似码本** $Q_{\text{approx}}$：用于第 1 尺度的近似 token
- **细节码本** $Q_{\text{details}}$：用于第 2 至第 S 尺度的细节 token

码本大小和特征维度与 ViT-VQGAN 基线一致（8192），但因为分开处理，同一个码元在不同尺度位置对应不同的语义。

### 损失函数

沿用 ViT-VQGAN 的损失组合，在空域 (IDWT 重建后) 计算：

| 损失项 | 权重 | 说明 |
|--------|------|------|
| L2 重建损失 | 1.0 | 像素级均方误差 |
| 感知损失 (Perceptual) | 0.1 | 基于预训练网络特征 |
| 对抗损失 (Adversarial) | 0.1 | 判别器指导生成真实纹理 |
| 码本承诺损失 (Commitment) | 0.25 | 稳定码本学习 |

关键改动：**去掉** logit-laplace 损失（后续工作 [Parti] 证实其有害），并引入**谱归一化 (Spectral Normalization)** 解决对抗损失导致的训练不稳定。

## 实验关键数据

### 主实验 1：多尺度图像重建 (ImageNet)

| 模型 | 分辨率 | LPIPS ↓ | PSNR ↑ | FID ↓ | IS ↑ |
|------|--------|---------|--------|-------|------|
| ViT-VQGAN | 256² | 0.163 | 23.8 | 1.20 | 194.6 |
| **SIT-4** | 256² | **0.144** | 24.0 | **1.20** | **199.5** |
| **SIT-5** | 256² | **0.135** | **24.5** | **0.97** | **202.3** |
| ViT-VQGAN | 512² | 0.320 | 22.4 | 6.92 | 151.5 |
| **SIT-6** | 512² | **0.239** | **23.1** | **1.74** | **203.7** |

SIT-SC 模型无需重训练即可处理多种分辨率：

| 模型 | 分辨率 | LPIPS ↓ | PSNR ↑ | FID ↓ |
|------|--------|---------|--------|-------|
| SIT-SC-5 | 128² | 0.159 | 27.1 | 2.13 |
| SIT-SC-5 | 64² | 0.111 | 31.3 | 1.39 |
| SIT-SC-5 | 32² | 0.029 | 36.8 | 0.31 |

### 主实验 2：类条件图像生成 (ImageNet 256²)

| 模型 | 参数量 | FID ↓ | IS ↑ |
|------|--------|-------|------|
| AR-ViT-VQGAN | 650M | 8.37 | 111.8 |
| **AR-SIT-4** | 650M | **6.95** | **138.3** |
| LlamaGen-L | 343M | 4.08 | 198.5 |
| VAR | 310M | 3.30 | 274.4 |
| **AR-SIT-4*** | 350M | **4.06** | 190.9 |

在公平对比下（相同架构/训练协议），AR-SIT-4 的 FID 从 8.37 降至 6.95。引入改进超参后的 AR-SIT-4* 达到 4.06，与 LlamaGen 持平。

### 主实验 3：文本引导生成 (MS-COCO)

| 模型 | 分辨率 | FID ↓ | 吞吐量 (imgs/s) ↑ | 显存效率 (imgs/GB) ↑ |
|------|--------|-------|------------------|-------------------|
| Parti350M | 256² | 12.4 | 7.8 | 12.0 |
| AR-SIT-SCD-4 | 256² | 12.6 | 6.5 | 8.0 |
| Parti350M | 64² | 10.5 | 7.6 | 12.0 |
| AR-SIT-SCD-4 | 64² | 11.4 | **24.5** | **16.0** |
| Parti350M | 32² | 5.8 | 7.7 | 7.7 |
| AR-SIT-SCD-4 | 32² | 7.6 | **74.7** | **28.0** |

低分辨率生成时，AR-SIT 的速度和显存效率优势巨大（32² 时吞吐量提升 ~10 倍）。

### 消融实验

| 配置 | 说明 | FID ↓ |
|------|------|-------|
| SIT-4 (Haar) | 默认配置 | 1.20 |
| SIT-4 (LeGall 5/3) | 更复杂小波 | 更高 |
| SIT-4 (CDF 9/7) | JPEG2000 小波 | 更高 |
| 无 ADTransformer | 共享所有层参数 | 更高 |
| 无 scale-causal | Dense attention | 略低（但失去多尺度能力） |

**关键发现**：
- **Haar 小波最优**：尽管是最简单的小波，但比压缩领域常用的 LeGall 5/3 和 CDF 9/7 效果更好，推测与较大滤波器支撑导致的边界泄漏有关
- 增大码本或序列长度虽改善重建，但反而损害生成质量（tokenizer-generator trade-off）
- 文本引导上采样 FID 从 12.6（纯文本）降至 6.2（给定低分辨率图像）

## 亮点与洞察

1. **频域与自回归的天然对齐**：小波变换的多尺度性质与自回归"从已知条件预测未知"的范式完美结合——低频 → 高频对应粗 → 细，远比 raster scan 的行扫描顺序自然
2. **一个 tokenizer 多种应用**：通过灵活组合编码器/解码器的 scale-causal 掩码，同一框架支持多尺度重建、渐进生成、上采样和编辑，无需重新训练
3. **序列长度优雅扩展**：分辨率倍增只需加一两个尺度（+N token），而非 4 倍增长，对高分辨率生成至关重要
4. **低分辨率预览的实用价值**：在交互式生成场景中，可以先快速生成多个粗略候选（仅前 25% token），用户从中选择后再补全剩余细节
5. **512² 高分辨率稳定性**：ViT-VQGAN 在 512² 训练时出现严重不稳定，而 SIT 无需调参即可成功训练

## 局限性

1. **文本到图像指标未明显超越基线**：在公平对比下，AR-SIT 的 text-to-image FID 与 Parti350M 基本持平，更好的 tokenizer 不一定带来更好的生成模型（Chang et al. 也有类似观察）
2. **模型规模较小**：实验仅测试了 350M-650M 参数的 AR 模型，而 Parti 最大可达 22B；大规模设定下的表现未知
3. **Haar 小波的局限**：虽然在这些实验中 Haar 最优，但其缺乏平滑性和消失矩等理论优良性质，在更大分辨率或更复杂场景下可能不是最优选择
4. **与 SOTA 的差距**：AR-SIT-4* 的 FID(4.06) 接近 LlamaGen 但不及 VAR(3.30)，后者使用卷积 tokenizer 和额外技巧（AdaLN、attention normalization）
5. **生成速度在全分辨率下略慢**：由于每个 token 必须经过 IDWT，全分辨率下的吞吐量低于空域方法

## 相关工作与启发

- **VQ-VAE / VQGAN / ViT-VQGAN**：SIT 的直接基线，通过频域替代空域 patch 化实现升级
- **VAR / RQ-VAE**：同为多尺度思想，但操作在潜在空间的残差上而非输入频谱上；SIT 的优势在于真正的多分辨率输入/输出能力
- **Dieleman (2024) "Diffusion is spectral autoregression"**：从理论角度论证扩散模型本质上在频域从低到高生成，SIT 则做到了"字面意义上的频谱自回归"
- **LlamaGen / FAR**：更新的 AR 生成方法，使用不同的 tokenizer 设计；SIT 的多尺度能力是正交贡献，可与这些改进结合

**启发**：频域表示为视觉生成提供了一种被低估的归纳偏置。未来可探索将 SIT 与更大规模 AR 模型、更先进的码本设计（如 FSQ）、视频生成等方向结合。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | 频域 tokenizer 思路新颖，多应用统一框架设计巧妙 |
| 技术深度 | 4 | DWT 与 Transformer 的结合设计完整，消融充分 |
| 实验充分度 | 4 | 多任务（重建/生成/上采样/编辑）验证全面，有公平基线对比 |
| 写作质量 | 4 | 动机清晰，图示精美，问题阐述到位 |
| 实用价值 | 3.5 | 渐进生成和多分辨率能力实用，但绝对性能未超越 SOTA |
| **综合** | **4** | 一篇扎实的方法论文，为 AR 图像生成引入频域先验的新范式 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [Holistic Tokenizer for Autoregressive Image Generation](holistic_tokenizer_for_autoregressive_image_generation.md)
- [DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)
- [M2SFormer: Multi-Spectral and Multi-Scale Attention with Edge-Aware Difficulty Guidance for Image Forgery Localization](m2sformer_multi-spectral_and_multi-scale_attention_with_edge-aware_difficulty_gu.md)
- [TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](../../CVPR2025/image_generation/tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)
- [EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation](../../CVPR2025/image_generation/evotok_a_unified_image_tokenizer_via_residual_latent_evolution_for_visual_unders.md)

<!-- RELATED:END -->
