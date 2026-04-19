---
title: >-
  [论文解读] CoD: A Diffusion Foundation Model for Image Compression
description: >-
  [CVPR 2026][图像生成][压缩导向扩散] 提出首个面向压缩的扩散基础模型 CoD，从零训练学习端到端的压缩-生成联合优化，替换 Stable Diffusion 后在下游扩散编解码器中实现超低码率（0.0039 bpp）下的 SOTA 性能，训练成本仅为 SD 的 0.3%。
tags:
  - CVPR 2026
  - 图像生成
  - 压缩导向扩散
  - 基础模型
  - rectified flow
  - 像素空间扩散
  - 率失真感知
---

# CoD: A Diffusion Foundation Model for Image Compression

**会议**: CVPR 2026  
**arXiv**: [2511.18706](https://arxiv.org/abs/2511.18706)  
**代码**: [GitHub](https://github.com/microsoft/GenCodec/tree/main/CoD)  
**领域**: 图像压缩 / 扩散模型  
**关键词**: 压缩导向扩散, 基础模型, rectified flow, 像素空间扩散, 率失真感知

## 一句话总结

提出首个面向压缩的扩散基础模型 CoD，从零训练学习端到端的压缩-生成联合优化，替换 Stable Diffusion 后在下游扩散编解码器中实现超低码率（0.0039 bpp）下的 SOTA 性能，训练成本仅为 SD 的 0.3%。

## 研究背景与动机

现有扩散编解码器（PerCo、DiffEIC、OSCAR 等）通常构建在 Stable Diffusion 之上以继承其生成先验。但文本条件从压缩角度看是次优的：

**文本描述能力有限**：人类文本难以精细描述自然图像的空间和纹理细节。

**离散词汇不可微**：文本编码器（如 BLIP-2）和扩散模型（如 SD）无法做联合端到端优化，无法进行率失真优化。

**实证证据**：DiffC 的零样本实验表明文本条件在低码率时实际上**损害**压缩性能。

核心洞察：如果把图像字幕生成器看作编码器、扩散模型看作解码器，这本质上就是一个压缩系统——但文本作为中间表示是低效的。用神经网络学习的原生图像 token 替代文本，并端到端联合训练压缩和生成，才是正确方向。

## 方法详解

### 整体框架

CoD 架构简洁：原生图像编码器 → 信息瓶颈（向量量化）→ 条件解码器 → 扩散模型（DiT backbone + DDT head）。在像素空间和隐空间都有实现。

### 关键设计

1. **条件编码与信息瓶颈**: 编码器用残差块 + 注意力层将图像压缩到 1/32 分辨率。信息瓶颈采用向量量化（VQ），码本大小 $N = 2^4 = 16$，对应 $4 \text{ bits} / (32 \times 32) = 0.0039 \text{ bpp}$ 的超低码率。这迫使扩散模型学习强大的生成能力来补偿信息丢失。条件解码器将量化 token 重建为 1/16 分辨率的中间条件。

2. **统一的 Rectified Flow 训练**: CoD 预测速度场 $v_t = x - \epsilon$（线性插值调度 $x_t = t \cdot x + (1-t) \cdot \epsilon$），使用 rectified flow 损失。关键发现是标准 RF 损失**只保证结构一致性而非颜色**。为此提出统一训练：随机选择 $\alpha\%$ 样本用 $t \in [0,1]$ 训练（优化感知），其余用 $t=0$ 训练。在 $t=0$ 时 RF 损失等价于单步重建 MSE：

   $$\mathcal{L}_{\text{RF}}|_{t=0} = \text{MSE}(v_0, v_0^{\text{pred}}) = \text{MSE}(x, \hat{x}_0)$$

   这在 RF 框架中自然融入了失真项，实现了率-失真-感知的联合优化。

3. **像素空间 vs. 隐空间**: 隐空间 CoD 在 SD-VAE 隐空间操作（2×2 patch embedding → 1/16），受限于 VAE 重建质量上限（~26 dB PSNR，0.6 bpp 码率天花板）。像素空间 CoD 用 16×16 patch embedding 直接建模原始像素，DDT head 每个特征预测一个神经场重建 16×16 patch。像素空间不受 VAE 限制，可覆盖 0.0039-4 bpp 的宽码率范围，PSNR 可达 ~47 dB 近无损级别。

4. **零样本失真-感知控制**: 统一训练赋予 CoD 通过采样步数直接控制失真-感知权衡的能力。25 步获得最佳感知质量；减为 1 步时 PSNR 提升 3.4 dB（16.2→19.6 dB），中间步数平滑插值。

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{\text{RF}} + \lambda \cdot \mathcal{L}_{\text{REPA}} + \beta \cdot \mathcal{L}_C + \gamma \cdot \mathcal{L}_{\text{aux}}$$

其中 $\mathcal{L}_{\text{REPA}}$ 是 DINOv2 特征对齐损失，$\mathcal{L}_C$ 是码本承诺损失，$\mathcal{L}_{\text{aux}}$ 是辅助头（重建原始像素 + DINOv2 特征）。训练分两阶段：256×256（400k 步）→ 512×512（150k 步），4 张 A100 约 5 天。

## 实验关键数据

### 主实验

**像素空间比较**（Kodak 512×512）：

| 方法 | 码率 (bpp) | PSNR↑ | FID↓ | 说明 |
|------|-----------|-------|------|------|
| VTM | ~0.2 | 基准 | - | 传统编解码器 |
| Pixel-CoD+DiffC | ~0.2 | **≈VTM** | **远优** | BD-Rate -2.1% vs VTM |
| MS-ILLM (GAN) | ~0.2 | 较低 | 较高 | 感知质量以 PSNR 为代价 |
| HiFiC (GAN) | ~0.2 | 低 | 中等 | 同上 |

**隐空间比较**（超低码率）：

| 方法 | 码率 (bpp) | 重建质量 | 说明 |
|------|-----------|---------|------|
| CoD (latent) + DiffC | <0.02 | **SOTA** | 超低码率优势显著 |
| SD-based DiffC | <0.02 | 差 | 文本条件在低码率有害 |
| PerCo (SD) | 0.0036 | 中等 | 依赖文本+图像条件 |
| OSCAR | ~0.01 | 较好 | 去文本但仍基于 SD |

### 消融实验 / Scaling Law

| 模型规模 (参数量) | 压缩性能 | 说明 |
|-----------------|---------|------|
| 49M CoD | 已优于 MS-ILLM (181M) | GAN 方法参数更多但效果差 |
| 114M CoD | 明显更优 | |
| 330M CoD | 进一步提升 | 清晰的 scaling law 趋势 |

### 关键发现

- **像素扩散的潜力被严重低估**：像素空间 CoD 能同时达到 VTM 级 PSNR 和超越 GAN 的感知质量，是首次证明扩散编解码器能在失真和感知两方面同时取胜
- **文本条件确实有害**：DiffC on SD 加入文本条件后 LPIPS 在低码率变差，CoD 条件直接提升
- **训练成本极低**：~20 A100 GPU days vs SD 的 ~6250 天（0.3%），完全开源数据可复现
- **49M 参数就能打败 181M 的 GAN 编解码器**，证明压缩性能提升来自算法而非模型规模

## 亮点与洞察

- 从压缩理论角度重新审视"文本条件在扩散编解码器中的角色"，得出文本有害的反直觉结论，并提供了清晰的理论解释
- 统一 RF 训练将 $t=0$ 的单步重建等价于 MSE 失真优化，在连续流框架中自然融入了率-失真-感知三方优化
- 通过采样步数控制失真-感知权衡是一个零成本的附加能力，无需额外训练
- 像素空间扩散的全面复兴：以往认为隐空间扩散全面优于像素空间，本文证明像素空间在高码率和宽范围方面有不可替代的优势

## 局限与展望

- 目前仅支持 512×512 分辨率，扩展到 2K+ 需要大幅增加计算成本
- 与所有扩散编解码器一样，推理速度不满足实时编码需求（虽然单步蒸馏版本已接近实时）
- 码率的最低值固定为 0.0039 bpp（受 VQ 码本大小限制），灵活码率控制需要额外设计
- 未在视频压缩上验证，时序扩展是自然方向

## 相关工作与启发

- **DiffC** [Theis et al.] 提出了零样本扩散压缩的理论框架，CoD 为其提供了更适合的基础模型
- **PerCo** [Careil et al.] 证明了扩散模型在极低码率压缩的潜力，但依赖文本条件
- **CDC** [Yang et al.] 是早期像素空间扩散编解码器探索，但需要感知损失且未考虑 scaling law
- 率-失真-感知三方权衡理论 [Blau & Michaeli] 是本文优化目标的理论基础

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首个面向压缩的扩散基础模型，统一训练策略和像素空间复兴都是重要贡献
- **实验充分度**: ⭐⭐⭐⭐⭐ 像素/隐空间双线比较 + 多基准 + scaling law + 零样本控制 + 视觉对比
- **写作质量**: ⭐⭐⭐⭐⭐ 从问题分析到方法设计再到实验验证环环相扣，insight 深刻
- **价值**: ⭐⭐⭐⭐⭐ 0.3%训练成本+全开源数据+SOTA性能，对扩散压缩领域有基础性推动

<!-- RELATED:START -->

## 相关论文

- [DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)
- [Zatom-1: A Multimodal Flow Foundation Model for 3D Molecules and Materials](../../ICLR2026/image_generation/zatom-1_a_multimodal_flow_foundation_model_for_3d_molecules_and_materials.md)
- [HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images](adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)
- [Steering One-Step Diffusion Model with Fidelity-Rich Decoder for Fast Image Compression](../../AAAI2026/image_generation/steering_one-step_diffusion_model_with_fidelity-rich_decoder_for_fast_image_comp.md)
- [GRAM: A Generative Foundation Reward Model for Reward Generalization](../../ICML2025/image_generation/gram_a_generative_foundation_reward_model_for_reward_generalization.md)

<!-- RELATED:END -->
