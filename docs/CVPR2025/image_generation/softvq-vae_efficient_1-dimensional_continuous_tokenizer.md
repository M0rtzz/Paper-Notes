---
title: >-
  [论文解读] SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer
description: >-
  [CVPR 2025][图像生成][图像分词器] SoftVQ-VAE 通过将 VQ-VAE 的硬分类后验改为软分类后验（每个潜 token 自适应聚合多个 codeword），实现了完全可微的连续图像分词器，仅用 32-64 个 1D token 就能将 256×256 和 512×512 图像压缩到极高比率，使 SiT-XL 在 ImageNet 上达到 1.78 FID 的同时推理吞吐量提升 18-55 倍。
tags:
  - CVPR 2025
  - 图像生成
  - 图像分词器
  - 软向量量化
  - 高压缩率
  - 连续潜空间
  - 生成效率
---

# SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer

**会议**: CVPR 2025  
**arXiv**: [2412.10958](https://arxiv.org/abs/2412.10958)  
**代码**: [https://github.com/Hhhhhhao/continuous_tokenizer](https://github.com/Hhhhhhao/continuous_tokenizer)  
**领域**: 图像生成 / 图像Tokenizer  
**关键词**: 图像分词器, 软向量量化, 高压缩率, 连续潜空间, 生成效率

## 一句话总结

SoftVQ-VAE 通过将 VQ-VAE 的硬分类后验改为软分类后验（每个潜 token 自适应聚合多个 codeword），实现了完全可微的连续图像分词器，仅用 32-64 个 1D token 就能将 256×256 和 512×512 图像压缩到极高比率，使 SiT-XL 在 ImageNet 上达到 1.78 FID 的同时推理吞吐量提升 18-55 倍。

## 研究背景与动机

**领域现状**：去噪生成模型（DiT、SiT、MAR）依赖图像分词器将原始图像编码为潜 token。主流分词器为 KL-VAE（连续高斯后验）和 VQ-VAE（离散分类后验），通常将 256×256 图像编码为至少 256 个 2D token。

**现有痛点**：(1) Transformer 生成模型的计算复杂度与 token 数量成二次方关系，256+ 个 token 严重制约训练和推理效率；(2) 进一步提高压缩率时，KL-VAE 面临后验坍塌问题，VQ-VAE 则因离散量化的梯度断裂（straight-through trick）导致重建和潜空间质量急剧下降；(3) 现有分词器的潜空间缺乏语义判别性，不利于下游生成模型学习。

**核心矛盾**：高压缩率需要每个 token 携带更多信息，但 KL-VAE 的高斯约束和 VQ-VAE 的一对一量化都限制了单个 token 的表达能力。

**本文目标**：设计一种能用极少 1D token（32-64 个）实现高质量重建和生成的连续图像分词器。

**切入角度**：如果允许每个潜 token 不是匹配到单一 codeword（VQ-VAE），而是以软权重聚合多个 codeword，就能在保持 codebook 结构化优势的同时大幅提升表达能力。

**核心 idea**：将 VQ-VAE 的 arg min 硬分配改为 softmax 软分配：$q_\phi(\mathbf{z}|\mathbf{x}) = \text{Softmax}(-\|\hat{\mathbf{z}} - \mathcal{C}\|_2 / \tau)$，每个 token 成为 codebook 中多个 codeword 的加权和，完全可微且无需 straight-through trick。

## 方法详解

### 整体框架

使用 ViT 架构的编码器-解码器。编码器接收图像 patch token 和 $L$ 个 1D 可学习 query token，通过自注意力让 query token 聚合图像信息。编码器输出经过 SoftVQ 模块（与 codebook 的软匹配）得到最终潜 token。解码器接收潜 token 和 $N$ 个 mask token，重建像素值。

### 关键设计

1. **软向量量化（SoftVQ）**:

    - 功能：将编码器输出映射到高表达力的连续潜空间
    - 核心思路：给定编码器输出 $\hat{\mathbf{z}}$ 和可学习 codebook $\mathcal{C} \in \mathbb{R}^{K \times D}$，计算软后验 $q_\phi(\mathbf{z}|\mathbf{x}) = \text{Softmax}(-\|\hat{\mathbf{z}} - \mathcal{C}\|_2 / \tau)$，其中温度 $\tau = 0.07$。最终潜 token 为 $\mathbf{z} = q_\phi(\mathbf{z}|\mathbf{x}) \mathcal{C}$，即 codebook 中所有 codeword 的加权和。KL 正则化为 $\mathcal{L}_{\text{kl}} = H(q_\phi) - H(\mathbb{E}_{\mathbf{x}} q_\phi)$（鼓励个体后验尖锐但整体均匀使用 codebook）。这整个过程完全可微，无需 codebook loss 或 commit loss
    - 设计动机：VQ-VAE 的 K-Means 分配限制了每个 token 只能对应一个 codeword，SoftVQ 的 Soft K-Means 允许每个 token 利用整个 codebook 的表达能力，在极少 token 数量下仍能保持高信息密度

2. **1D 可学习潜 Token 与 ViT 架构**:

    - 功能：支持任意长度的 1D 潜 token 序列，实现灵活的压缩率
    - 核心思路：编码器输入为图像 patch token（$N = HW/P^2$ 个）拼接 $L$ 个可学习 query token，通过自注意力学习信息聚合，输出仅保留 query token 对应的部分。解码器用可学习 mask token 作为查询，与潜 token 拼接后通过自注意力重建。1D 位置编码使 token 数量与图像分辨率解耦
    - 设计动机：传统 2D 网格 token 的数量受空间分辨率固定约束（如 $32 \times 32 = 1024$），1D query token 可以自由设定长度（32、64、128 等）

3. **潜空间语义对齐**:

    - 功能：让潜 token 携带语义判别特征，提升下游生成质量
    - 核心思路：将每个潜 token 复制 $N/L$ 次展开到与图像 patch 相同长度，通过投影 MLP 与预训练视觉编码器（如 DINOv2）的特征计算余弦相似度损失 $\mathcal{L}_{\text{align}}$。得益于 SoftVQ 的完全可微性，对齐梯度可直接流到编码器和 codebook
    - 设计动机：KL-VAE 的高斯约束和 VQ-VAE 的梯度断裂都使语义对齐难以有效传播；SoftVQ 的可微性从根本上解决了这个问题

### 损失函数 / 训练策略

总损失 = 重建损失 + 感知损失 + 对抗损失 + $\mathcal{L}_{\text{kl}}$ + $\mathcal{L}_{\text{align}}$。温度 $\tau = 0.07$，codebook 大小 $K = 8192$，潜维度 $D = 32$。ViT-Base/Large 编码器和解码器。在 ImageNet 上训练 300 epochs。下游生成用 DiT/SiT/MAR 训练。

## 实验关键数据

### 主实验 — ImageNet 256×256 生成

| 分词器 | Token 数 | SiT-XL FID ↓ | SiT-XL 推理吞吐量 ↑ |
|--------|---------|--------|------------|
| SD-VAE (KL) | 1024 | 2.06 | 1.0× |
| SDXL-VAE (KL) | 1024 | 2.12 | 1.0× |
| TiTok | 128 | 2.77 | 5.3× |
| DC-AE | 256 | 2.32 | 3.2× |
| SoftVQ-VAE | 64 | **1.78** | **18×** |
| SoftVQ-VAE | 32 | 2.33 | **18×** |

### 512×512 生成

| 分词器 | Token 数 | SiT-XL FID ↓ | 推理吞吐量 ↑ |
|--------|---------|--------|---------|
| SD-VAE | 4096 | 3.14 | 1.0× |
| SoftVQ-VAE | 64 | **2.21** | **55×** |

### 消融实验

| 变体 | 64 token rFID ↓ | 32 token rFID ↓ |
|------|---------|---------|
| KL-VAE (ViT) | 5.42 | 12.8 |
| VQ-VAE (ViT) | 3.85 | 8.7 |
| SoftVQ-VAE | **1.48** | **2.12** |
| + 语义对齐 | 1.48 | 2.12 |

### 关键发现

- 仅 **64 个 token** 的 SoftVQ-VAE 在 256×256 上实现 **1.78 FID**，超越使用 1024 token 的 SD-VAE (2.06)，同时推理吞吐量提升 18×
- 在 512×512 上效果更显著：64 token 达到 **2.21 FID** 和 **55× 吞吐量**，因为原始 4096 token 的二次复杂度被极大压缩
- SoftVQ 在 32 token 极端压缩下（256×256 图像压缩为 32 个标量 token）仍保持 2.33 FID，而 KL-VAE 在此压缩率下 FID 飙升至 12.8
- 语义对齐不改善重建指标但显著提升生成 FID（2.3× 训练收敛加速），证明生成质量更依赖潜空间的语义结构而非重建能力

## 亮点与洞察

- **从 VQ-VAE 到 SoftVQ-VAE 的改动极其简洁**（仅将 argmin 换成 softmax），但带来的效果提升巨大，堪称"最小改动最大收益"的典范
- **"更少 token = 更好生成"**的反直觉发现令人印象深刻：64 token 比 1024 token 的 FID 更低，说明高压缩迫使 token 学习更紧凑和语义的表示
- **完全可微带来的连锁优势**——无需 codebook loss、commit loss、straight-through trick，同时可以直接做语义对齐

## 局限与展望

- 1D token 与 2D 空间结构完全解耦，可能丢失局部空间关系信息
- 当前仅在 ImageNet 上验证，文本到图像的复杂场景（如 COCO）有待验证
- 32 token 的极端压缩在高分辨率图像上的表现未知
- 与 autoregressive 生成范式（如 LLaMA-based 图像生成）的兼容性有待探索

## 相关工作与启发

- **vs TiTok**：TiTok 也使用 1D token 但需要额外解码器且 128 token 时 FID 2.77；SoftVQ-VAE 用 64 token 达到 1.78，架构更简洁效果更好
- **vs DC-AE**：DC-AE 在 256 token 时 FID 2.32 且进一步压缩质量急剧下降；SoftVQ-VAE 的软量化使极端压缩成为可能
- **vs REPA**：REPA 在生成模型中间层做特征对齐；SoftVQ-VAE 在分词器潜空间做对齐，等价于在生成模型输入空间对齐，更根本

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将 VQ 的硬分配改为软分配的思路简洁优雅，效果显著
- 实验充分度: ⭐⭐⭐⭐⭐ 三种生成模型(DiT/SiT/MAR)、多种分辨率、与多种分词器的全面对比
- 写作质量: ⭐⭐⭐⭐ 从 KL-VAE/VQ-VAE 的统一视角出发推导自然
- 价值: ⭐⭐⭐⭐⭐ 为图像生成效率提升了一个数量级，对整个视觉生成领域有深远影响

<!-- RELATED:START -->

## 相关论文

- [TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)
- [Efficient Adversarial Attacks on High-dimensional Offline Bandits](../../ICLR2026/image_generation/efficient_adversarial_attacks_on_high-dimensional_offline_bandits.md)
- [Spectral Image Tokenizer](../../ICCV2025/image_generation/spectral_image_tokenizer.md)
- [EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation](evotok_a_unified_image_tokenizer_via_residual_latent_evolution_for_visual_unders.md)
- [DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](../../ICCV2025/image_generation/dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)

<!-- RELATED:END -->
