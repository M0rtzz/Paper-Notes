---
title: >-
  [论文解读] Latent Diffusion Models with Masked AutoEncoders
description: >-
  [ICCV 2025][图像生成][潜在扩散模型] 系统性地分析了 LDM 中自编码器应具备的三个关键属性（潜空间平滑性、感知压缩质量、重建质量），发现现有自编码器无法同时满足，提出 Variational Masked AutoEncoders (VMAEs)…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "潜在扩散模型"
  - "掩码自编码器"
  - "变分自编码器"
  - "感知压缩"
  - "潜空间平滑性"
---

# Latent Diffusion Models with Masked AutoEncoders

**会议**: ICCV 2025  
**arXiv**: [2507.09984](https://arxiv.org/abs/2507.09984)  
**代码**: [https://github.com/isno0907/ldmae](https://github.com/isno0907/ldmae)  
**领域**: 图像生成  
**关键词**: 潜在扩散模型, 掩码自编码器, 变分自编码器, 图像生成, 感知压缩, 潜空间平滑性

## 一句话总结

系统性地分析了 LDM 中自编码器应具备的三个关键属性（潜空间平滑性、感知压缩质量、重建质量），发现现有自编码器无法同时满足，提出 Variational Masked AutoEncoders (VMAEs)，结合 MAE 的层次化特征和 VAE 的概率编码，在仅 13.4% 参数和 4.1% GFLOPs 的条件下显著提升生成质量（ImageNet-1K gFID: 5.98 vs SD-VAE 的 6.49）。

## 研究背景与动机

**为什么自编码器设计对 LDM 至关重要？** 潜在扩散模型（LDM）的核心思想是将去噪过程从像素空间转移到压缩的潜空间，而自编码器决定了这个潜空间的性质。然而，关于自编码器应该具备什么属性，以及不同设计选择如何影响整个 LDM 框架，现有研究探索不够充分。

**三个关键属性的提出**：

**潜空间平滑性**：潜表示的微小扰动不应导致生成结果剧变。扩散模型的去噪过程会引入预测误差，光滑的潜空间能容忍这些误差

**感知压缩质量**：有效压缩掉感知细节的同时保留语义信息。但"语义"和"感知细节"之间没有清晰边界——它们是一个从像素级到物体级的连续谱

**重建质量**：解码器应能准确重建原图，且需在感知级（rFID、LPIPS）和像素级（PSNR、SSIM）两个层面评估

**现有自编码器各自的短板**：
- **AE/DAE**（确定性编码）：潜空间稀疏，不满足平滑性
- **VAE**：潜空间最平滑但重建质量差
- **SD-VAE**：感知压缩过于激进（压缩到物体级），丢失细粒度特征

## 方法详解

### 整体框架

VMAE 采用对称 ViT 架构（编码器-解码器），结合概率编码、掩码预测损失和感知损失，同时满足三个属性。

### 平滑潜空间（Sec 4.1）

采用概率编码 $q_\phi(\mathbf{z}|\mathbf{x})$ 而非固定向量，通过 KL 散度约束潜分布趋向高斯先验：

$$\mathcal{L}_{\text{reg}} = \mathbb{E}_{p_{\text{data}}(\mathbf{x}), p(\mathbf{x}_v|\mathbf{x})} [D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}_v) | p(\mathbf{z}))]$$

关键设计：允许预测均值可学习（不强制为零），仅约束方差为单位方差，满足 VP（Variance Preserving）条件的同时保留区分性特征。

确定性编码器（AE/DAE）将输入映射为离散稀疏点，潜空间中大部分位置不可解码；概率编码器将输入映射为分布，形成连续潜空间，噪声邻域内的点都可解码。

### 层次化感知压缩（Sec 4.2）

利用 MAE 的掩码预测目标实现层次化压缩：

$$\mathcal{L}_{\text{M}} = \mathbb{E}[-\log p_\theta(\mathbf{x}_m | \mathbf{z})]$$

其中编码器仅接收可见区域 $\mathbf{x}_v$，解码器基于潜变量 $\mathbf{z}$ 预测被掩码区域 $\mathbf{x}_m$。

**为什么 MAE 能实现层次化压缩？** 最新研究表明，MAE 的掩码预测训练使编码特征在嵌入空间中形成层次化聚类——从抽象物体级到更简单的视觉模式级逐步分化。这种层次结构既有利于扩散模型训练（高层集群简化学习），又保留了细粒度信息（多层次区分性特征支撑高重建质量）。

相比之下，SD-VAE 的压缩过于激进：特征在物体级聚类后无法进一步区分不同部分（如长颈鹿的毛皮花纹），导致解码时细节丢失。

### 感知重建（Sec 4.3）

在可见区域应用重建损失和 LPIPS 感知损失：

$$\mathcal{L}_{\text{R}} = \mathbb{E}[-\log p_\theta(\mathbf{x}_v | \mathbf{z})]$$

$$\mathcal{L}_{\text{P}} = \mathbb{E}\left[\sum_l w_l \|\psi_l(\mathbf{x}) - \psi_l(\hat{\mathbf{x}})\|_2^2\right]$$

其中 $\psi_l$ 为预训练 VGG 的第 $l$ 层特征提取。

### 完整训练目标

$$\mathcal{L}_{\text{VMAE}} = \mathcal{L}_{\text{R}} + \lambda_{\text{M}} \cdot \mathcal{L}_{\text{M}} + \lambda_{\text{P}} \cdot \mathcal{L}_{\text{P}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}}$$

## 实验关键数据

### 主实验：生成性能对比

| 自编码器 | ImageNet gFID↓ | sFID↓ | IS↑ | Prec↑ | Rec↑ | CelebA gFID↓ |
|---------|---------------|-------|-----|-------|------|-------------|
| AE | 12.92 | 12.65 | 124.0 | 0.724 | 0.339 | 24.80 |
| DAE | 8.60 | 12.12 | 160.3 | 0.797 | 0.402 | 21.42 |
| VAE | 34.60 | 22.32 | 54.6 | 0.517 | 0.415 | 32.33 |
| SD-VAE | 6.49 | 5.60 | 173.3 | 0.819 | 0.429 | 9.00 |
| **VMAE** | **5.98** | **5.16** | **185.5** | **0.844** | **0.435** | **7.61** |

VMAE 在所有生成指标上均超越 SD-VAE，ImageNet gFID 提升 0.51，IS 提升 12.2，CelebA gFID 提升 1.39。确定性自编码器（AE/DAE）因缺乏平滑潜空间而性能显著下降。

### 重建性能对比

| 模型 | L1↓ | PSNR↑ | SSIM↑ | LPIPS↓ | rFID↓ |
|------|-----|-------|-------|--------|-------|
| AE | 0.0218 | 32.18 | 0.895 | 0.172 | 6.21 |
| DAE | 0.0237 | 31.31 | 0.887 | 0.175 | 3.97 |
| VAE | 0.0281 | 29.40 | 0.825 | 0.281 | 17.41 |
| SD-VAE | 0.0223 | 29.85 | 0.853 | 0.099 | 1.89 |
| **VMAE** | **0.0221** | **31.52** | **0.890** | **0.062** | **0.89** |

VMAE 同时在像素级（PSNR 31.52 vs SD-VAE 29.85）和感知级（LPIPS 0.062 vs 0.099，rFID 0.89 vs 1.89）均取得最优。

### 消融实验：各损失项贡献

| 损失组合 | PSNR↑ | SSIM↑ | rFID↓ | LPIPS↓ | gFID↓ |
|---------|-------|-------|-------|--------|-------|
| Baseline (MSE only) | 32.18 | 0.906 | 6.21 | 0.172 | 12.92 |
| + Masking loss $\mathcal{L}_M$ | 32.01 | 0.913 | 4.66 | 0.130 | 8.92 |
| + Latent regularizer $\mathcal{L}_{reg}$ | 31.14 | 0.881 | 1.59 | 0.112 | 6.32 |
| + Perceptual loss $\mathcal{L}_P$ | **31.52** | **0.889** | **0.89** | **0.062** | **5.98** |

每个组件都有明确贡献：掩码损失引入层次化压缩（gFID -4.0）→ 潜空间正则化提供平滑性（gFID -2.6）→ 感知损失恢复视觉细节（LPIPS -0.050）。

### 模型效率对比

| 指标 | AE/DAE/VAE/SD-VAE | VMAE |
|------|-------------------|------|
| 模型大小 | 319.7 MB | 42.7 MB (13.4%) |
| GFLOPs | 17,331.3 | 703.9 (4.1%) |
| 训练时间 | 24 hr | 9 hr (37.5%) |

VMAE 在参数量、计算量和训练时间上均大幅领先，且因为推理更快，LDM 每次训练迭代的编码开销也更低。

## 亮点与洞察

1. **系统性分析框架**：三个属性（平滑性/压缩/重建）的提出及其量化评估方法为自编码器设计提供了清晰的评价体系
2. **层次化压缩的理论洞察**：SD-VAE 的过度压缩（物体级聚类但内部特征纠缠）是其重建细节损失的根源，这一发现对理解 LDM 瓶颈非常有价值
3. **效率提升惊人**：仅用 4.1% 的 GFLOPs 就超越了 SD-VAE，得益于 ViT 架构的高效 patchification 和轻量设计
4. **VAE 的教训**：最平滑的潜空间（VAE）反而生成最差，证明平滑性是必要但非充分条件——重建质量同等重要

## 局限性

- 实验主要在 256×256 分辨率进行，未验证高分辨率场景
- 仅替换了 LDM 中的自编码器，扩散模型骨干（DiT/UNet）未做适配优化
- MAE 的随机掩码策略可能在某些结构化数据上不是最优选择
- 与 FLUX/SD3 等最新 VAE（16通道）的对比不够充分

## 相关工作与启发

- **SD-VAE (StableDiffusion3)**：当前 LDM 的标准自编码器，VMAE 在所有维度上超越
- **MAE**：提供了层次化特征的基础，VMAE 将其扩展为概率编码
- **DC-AE**：追求高空间压缩率，与 VMAE 的目标互补
- **VA-VAE**：通过与视觉基础模型对齐改进 VAE，另一条增强路线
- **启发**：自编码器不只是 LDM 的"预处理工具"——它的潜空间属性直接决定扩散训练效率和生成质量，是值得深入投资的组件

## 评分 ⭐⭐⭐⭐

- 创新性：⭐⭐⭐⭐ — 三属性分析框架 + MAE-VAE 结合的设计方向新颖
- 实验充分度：⭐⭐⭐⭐⭐ — 自编码器属性分析 + 生成对比 + 重建对比 + 消融 + 效率全面覆盖
- 实用价值：⭐⭐⭐⭐ — 可直接替代 SD-VAE 用于 LDM 训练，且效率大幅提升
- 写作质量：⭐⭐⭐⭐⭐ — 分析深入系统，雷达图等可视化效果出色

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching](lazymar_accelerating_masked_autoregressive_models_via_feature_caching.md)
- [\[NeurIPS 2025\] Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders](../../NeurIPS2025/image_generation/learning_interpretable_features_in_audio_latent_spaces_via_sparse_autoencoders.md)
- [\[ICCV 2025\] HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation](hypdae_hyperbolic_diffusion_autoencoders_for_hierarchical_few-shot_image_generat.md)
- [\[ICCV 2025\] What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization](whats_in_a_latent_leveraging_diffusion_latent_space_for_domain_generalization.md)
- [\[ICCV 2025\] REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers](repa-e_unlocking_vae_for_end-to-end_tuning_of_latent_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
