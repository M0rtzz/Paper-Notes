---
title: >-
  [论文解读] REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder
description: >-
  [图像生成] 提出 REGEN，用扩散 Transformer（DiT）替代传统 VAE 解码器作为视频的再生式解码器，通过"生成而非精确重建"的学习范式突破视频时序压缩瓶颈，实现最高 32× 时序压缩。 当前视频生成的潜扩散模型（LDM）严重依赖视频嵌入器将视频压缩到潜空间进行建模。以 MAGVIT-v2 为代表的 SOT…
tags:
  - "图像生成"
---

# REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder

> **会议**: ICCV 2025
> **arXiv**: [2503.08665](https://arxiv.org/abs/2503.08665)
> **代码**: [项目页面](https://bespontaneous.github.io/REGEN/)
> **领域**: 图像生成
> **关键词**: video embedder, diffusion transformer, temporal compression, latent diffusion, video generation

## 一句话总结

提出 REGEN，用扩散 Transformer（DiT）替代传统 VAE 解码器作为视频的再生式解码器，通过"生成而非精确重建"的学习范式突破视频时序压缩瓶颈，实现最高 32× 时序压缩。

## 研究背景与动机

当前视频生成的潜扩散模型（LDM）严重依赖视频嵌入器将视频压缩到潜空间进行建模。以 MAGVIT-v2 为代表的 SOTA 视频嵌入器通常实现 8× 空间压缩但仅 4× 时序压缩，更高的时序压缩对训练和推理效率至关重要但面临根本性的瓶颈。

**核心矛盾**：传统编码器-解码器架构中，提高压缩率必然导致信息丢失，解码器无法从稀疏潜变量精确重建高频细节，形成"压缩-重建"的根本权衡。

**关键洞察**：在潜扩散建模的语境中，潜空间的核心属性应该是**生成视觉上合理的内容**，而非忠实还原输入视频。这一松弛的准则使得大幅提高压缩率成为可能。

基于此，REGEN 将传统的 encoder-decoder 转变为 **encoder-generator** 范式：编码器只需保留核心语义和结构信息，DiT 解码器负责合成逼真的细节。

## 方法详解

### 整体框架

REGEN 由两个核心模块构成（Fig. 2）：

1. **时空视频编码器**：将输入视频编码为两帧紧凑潜表示（content latent $z_c$ + motion latent $z_m$）
2. **DiT 生成式解码器**：以潜变量为条件，通过扩散过程从噪声重新合成视频

### 关键设计 1：时空视频编码器

采用 MAGVIT-v2 风格的因果 3D 卷积编码器，对 $k+1$ 帧视频编码为两帧潜特征：

$$z_c, z_m = E(x_{input})$$

- $z_c$：**内容潜帧**（content latent），因果性仅含首帧信息
- $z_m$：**运动潜帧**（motion latent），编码其余帧的压缩运动信息
- 两者均使用 8 个潜通道，空间压缩 8×，时序压缩可达 8×/16×/32×

### 关键设计 2：潜变量条件模块（Content-Aware PE）

这是 REGEN 的核心创新。传统 DiT 使用固定位置编码（PE），难以泛化到训练时未见的分辨率和宽高比。REGEN 将位置编码替换为**内容感知位置编码**，由编码的潜变量生成：

$$C_e(x, y, t_f | [z_c, z_m]) = M_s\left(z_c(x,y) \oplus M_t(t_f | z_m(x,y))\right)$$

其中 $M_t$ 是 SIREN 网络，将时间坐标 $t_f$ 映射为特征向量并受 $z_m$ 调制；$M_s$ 是线性投影器。生成的扩展潜变量 $z_e$ 与 DiT 的 token embedding 和 timestep embedding 相加作为输入。

这种设计：(1) 完全移除 DiT 原始的空间/时序 PE；(2) 自然支持任意分辨率和宽高比；(3) 支持时序插值和外推。

### 训练目标

端到端联合训练编码器和 DiT 解码器，使用标准扩散去噪损失：

$$\mathcal{L}(\theta) = \|\epsilon - \epsilon_\theta(x^t_{target}, [z_c, z_m])\|^2$$

### DiT 解码器配置

- 24 层 Transformer blocks，16 头，隐藏维度 2048
- Patch size = 8（与空间下采样比匹配）
- 支持完整时空自注意力

## 实验

### 主实验：高压缩率重建比较（Tab. 1）

| 方法 | 压缩率 | MCL-JCV PSNR | MCL-JCV rFVD ↓ | DAVIS PSNR | DAVIS rFVD ↓ |
|------|--------|-------------|----------------|-----------|-------------|
| MAGVIT-v2 | 8×8×8 | 29.14 | 72.07 | 24.75 | 125.03 |
| **REGEN** | 8×8×8 | **32.74** | **29.88** | **29.34** | **89.98** |
| MAGVIT-v2 | 8×8×16 | 26.62 | 185.69 | 21.21 | 417.43 |
| **REGEN** | 8×8×16 | **30.41** | **92.48** | **26.27** | **235.13** |
| MAGVIT-v2 | 8×8×32 | 22.97 | 536.01 | 18.23 | 1080.15 |
| **REGEN** | 8×8×32 | **28.71** | **224.56** | **23.49** | **522.20** |

REGEN 在所有压缩率上全面超越 MAGVIT-v2，且优势随压缩率增加而扩大。在 32× 时序压缩下，REGEN rFVD 仅为 MAGVIT-v2 的约 50%。

### 基础 4× 压缩比较（Tab. 2, 512×512）

| 方法 | PSNR | SSIM | rFVD ↓ |
|------|------|------|--------|
| OmniTokenizer | 24.63 | 0.710 | 93.35 |
| WF-VAE | 31.00 | 0.804 | 55.01 |
| VidTok | 32.06 | 0.836 | 38.85 |
| MAGVIT-v2 | 31.49 | 0.829 | 28.63 |
| **REGEN** | **32.94** | **0.857** | **22.40** |

即使在基础 4× 压缩下，REGEN 也超越了所有专为该设置定制的 SOTA 方法。

### 条件机制消融（Tab. 3）

| 方法 | 192×320 PSNR | 384×640 PSNR | 384×640 rFVD ↓ |
|------|-------------|-------------|----------------|
| In-context 条件 | 25.71 | 23.39 | 441.98 |
| **Ours (内容感知 PE)** | **26.04** | **29.41** | **57.01** |

关键发现：in-context conditioning 在更高分辨率下出现严重网格状伪影（rFVD 暴涨至 442），而 REGEN 的内容感知 PE 能优雅泛化到未见分辨率。

### 少步与单步采样

DiT 解码器在单步采样下即可获得高质量重建，无需外部蒸馏。PSNR 随步数减少略有提升（减少锐化），rFVD 略有上升。这归因于编码潜变量提供了极强的条件信号。

## 亮点与洞察

1. **范式转移**："encoder-generator" 替代 "encoder-decoder"，从"精确重建"转向"合理生成"，打破压缩-重建权衡
2. **Content-Aware PE** 一举解决位置编码泛化、条件注入和任意分辨率支持三个问题
3. 解码器支持**单步推理**而无需蒸馏，极大降低实际部署成本
4. 32× 时序压缩使文本到视频生成的潜帧数减少 ~5×，大幅降低训练和推理成本

## 局限性

- 生成式解码引入随机性，每次解码可能产生微小差异
- 极高压缩下仍存在高运动区域的时序伪影
- DiT 解码器的计算开销仍值得进一步优化

## 相关工作

- 视频嵌入器：MAGVIT-v2、OmniTokenizer、VidTok、WF-VAE
- 扩散自编码器：DiffAE、PDAE
- 视频 LDM：CogVideoX、HunyuanVideo

## 评分

- **新颖性**: ★★★★★ — encoder-generator 范式具有开创性
- **技术深度**: ★★★★☆ — 内容感知 PE 设计精巧，理论与实践结合好
- **实验质量**: ★★★★★ — 多压缩率、多数据集全面对比，消融充分
- **写作质量**: ★★★★★ — 核心洞察明确，论述逻辑清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Learning to See in the Extremely Dark](learning_to_see_in_the_extremely_dark.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [\[ICCV 2025\] SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)
- [\[NeurIPS 2025\] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network](../../NeurIPS2025/image_generation/denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)
- [\[NeurIPS 2025\] Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior](../../NeurIPS2025/image_generation/diff-icmh_harmonizing_machine_and_human_vision_in_image_compression_with_generat.md)

</div>

<!-- RELATED:END -->
