---
title: >-
  [论文解读] Holistic Tokenizer for Autoregressive Image Generation
description: >-
  [ICCV 2025][图像生成][Image Tokenizer] 提出 Hita，一种全局-局部（holistic-to-local）图像 tokenizer，通过可学习全局查询捕获纹理/材质/形状等全局属性，结合双码本量化和因果注意力融合模块，在不修改 AR 模型架构的前提下，将 ImageNet 256×256 生成 FID 降至 2.59、训练收敛加速 2.1 倍，并支持零样本风格迁移和图像补全。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "Image Tokenizer"
  - "Autoregressive Generation"
  - "Holistic Tokens"
  - "GAN"
  - "LlamaGen"
---

# Holistic Tokenizer for Autoregressive Image Generation

**会议**: ICCV 2025  
**arXiv**: [2507.02358](https://arxiv.org/abs/2507.02358)  
**代码**: [https://github.com/CVMI-Lab/Hita](https://github.com/CVMI-Lab/Hita)  
**领域**: 图像生成 / 自回归模型 / 图像Tokenization  
**关键词**: Image Tokenizer, Autoregressive Generation, Holistic Tokens, VQGAN, LlamaGen

## 一句话总结
提出 Hita，一种全局-局部（holistic-to-local）图像 tokenizer，通过可学习全局查询捕获纹理/材质/形状等全局属性，结合双码本量化和因果注意力融合模块，在不修改 AR 模型架构的前提下，将 ImageNet 256×256 生成 FID 降至 2.59、训练收敛加速 2.1 倍，并支持零样本风格迁移和图像补全。

## 研究背景与动机

### 问题背景
自回归（AR）图像生成模型遵循 GPT 范式，先用 VQVAE/VQGAN 将图像编码为离散 token 序列，再用 Llama 等因果 Transformer 逐 token 预测。现有 tokenizer 主要基于 patch 级别表示，缺乏全局信息。

### 核心挑战

**全局信息缺失**：Patch-level token 只携带局部信息，AR 模型难以在逐步生成时维持全局一致性

**因果注意力的局限**：AR 模型使用因果注意力逐 token 预测，无法看到后续 token，导致长程依赖困难

**零样本补全质量差**：当 LlamaGen 等模型被要求补全图像下半部分时，生成内容与上半部分语义不一致（如鱼变成"鱼-鸟"混合体）

### 关键动机
如果 tokenizer 能在 token 序列开头提供全局信息（如整体纹理、颜色、材质等），这些全局 token 就能作为前缀引导后续 patch token 的生成，无需修改 AR 模型本身。

## 方法详解

### 整体框架
Hita 包含三个核心模块：(1) 全局特征提取——可学习查询从 patch 嵌入和基础模型特征中捕获全局属性；(2) 双码本独立量化——全局 token 和 patch token 使用独立码本；(3) token 融合与解码——轻量级因果融合模块优先处理全局 token 后送入解码器重建图像。

### 全局特征提取
基于 VQGAN encoder $\mathcal{E}(\cdot)$ 提取 patch 嵌入，同时引入 $M$ 个可学习查询 $Q \in \mathbb{R}^{M \times C}$ 通过注意力机制从全局 patch 嵌入中聚合信息。使用预训练 DINOv2 注入语义特征：

$$Q, Z = \mathcal{E}_{\text{trans}}(Q \oplus \mathcal{E}(I) \oplus \mathcal{H}(I))$$
$$\overline{Q}, \overline{Z} = \mathcal{E}_{\text{causal}}(Q \oplus Z)$$

其中 $\mathcal{H}(\cdot)$ 是 DINOv2，$\oplus$ 是拼接操作。关键设计：**因果 Transformer** $\mathcal{E}_{\text{causal}}$ 将全局查询放在序列前面、patch 嵌入（raster-scan 顺序）放在后面，使隐空间自然对齐 AR 模型的因果生成模式。

### 双码本独立量化
全局查询 $\overline{Q}$ 和 patch 嵌入 $\overline{Z}$ 使用独立码本分别量化：
- 全局量化器 $\mathcal{Q}_H(\cdot)$：专用码本捕获全局属性
- Patch 量化器 $\mathcal{Q}_P(\cdot)$：标准 patch 级码本
- 两个码本均采用 $\ell_2$ 归一化 + 低维向量 + 大码本尺寸（16,384）策略

### Token 融合与解码（关键创新）
直接将全局和 patch token 拼接送入 Transformer + decoder 会导致**全局码本坍塌**——因为 patch token 通过跳跃连接直接影响对应 patch 的重建，绕过了全局 token。

解决方案：在因果 Transformer 融合后，**取最后 $k$ 个全局 token 替换前 $k$ 个 patch token** 送入解码器：

$$\tilde{Q}, \tilde{Z} = \hat{\mathcal{E}}_{\text{causal}}(\hat{Q} \oplus \hat{Z}_p)$$
$$\hat{I} = \mathcal{D}(\mathcal{R}(\tilde{Q}_{[-k:]} \oplus \tilde{Z}_{[:-k]}))$$

这使得 patch token 供给解码器的信息不完整，必须与全局 token 交互才能补偿，从而避免了退化解。

### 训练目标
总损失：$\mathcal{L} = \alpha \cdot \mathcal{L}_{vq} + \lambda \cdot \mathcal{L}_{AE}$，其中：
- $\mathcal{L}_{vq} = \mathcal{L}_{vq}(\overline{Q}) + \mathcal{L}_{vq}(\overline{Z}_p)$（两个码本的量化损失之和）
- $\mathcal{L}_{AE} = \mathcal{L}_2 + \mathcal{L}_P(\text{LPIPS}) + \lambda_G \cdot \mathcal{L}_G(\text{PatchGAN})$

### AR 生成
训练好 Hita 后，标准 Llama AR 模型无需任何修改即可接入：先生成全局 token 作为前缀提示，再逐步生成 patch token 序列，最终经融合模块+解码器转换为图像。

## 实验

### 主实验：ImageNet 256×256 类条件生成

| 模型 | 参数量 | FID↓ | IS↑ | Precision↑ | Recall↑ |
|------|--------|------|------|----------|--------|
| LDM-4 | 400M | 3.60 | 247.7 | 0.87 | 0.48 |
| DiT-XL/2 | 675M | 2.27 | 278.2 | 0.83 | 0.57 |
| LlamaGen-B | 111M | 8.31 | 154.7 | 0.84 | 0.38 |
| **Hita-B** | 111M | **5.85** | **212.3** | 0.84 | 0.41 |
| LlamaGen-L | 343M | 4.24 | 206.7 | 0.83 | 0.49 |
| **Hita-L** | 343M | **3.75** | **262.1** | 0.85 | 0.48 |
| LlamaGen-XXL | 1.4B | 2.89 | 236.2 | 0.81 | 0.56 |
| **Hita-XXL** | 1.4B | **2.70** | **274.8** | 0.84 | 0.55 |
| LlamaGen-3B | 3B | 2.61 | 251.9 | 0.80 | 0.56 |
| **Hita-2B** | 2B | **2.59** | **281.9** | 0.84 | 0.56 |

Hita 在所有尺寸下均显著优于 LlamaGen，2B 模型以更少参数超越 LlamaGen-3B。超越 LDM-4 扩散模型 0.7 FID 和 19.6 IS。

### 消融实验

| 配置 | rFID↓ | gFID↓ | gIS↑ | 线性探测↑ |
|------|-------|-------|------|----------|
| Baseline（无全局） | 1.31 | 9.37 | 162.6 | 14.2 |
| + 可学习查询 | 1.15 | 6.32 | 187.9 | 28.2 |
| + DINOv2 注入 | **1.03** | **5.85** | **212.3** | **36.6** |

可学习查询本身即可显著提升重建和生成质量（gFID 降 3.05），DINOv2 注入进一步增强语义表示。线性探测从 14.2 提升至 36.6 证明全局 token 确实捕获了丰富的语义信息。

### $k$ 值消融（Token 融合设计验证）
- $k=0$ 时全局码本使用率坍塌至极低值，patch token 完全绕过全局 token
- $k>0$ 时全局码本正常使用，$k=4$ 取得最优重建和生成质量
- 这验证了强制 patch token 依赖全局 token 的设计必要性

### 训练加速
达到 FID=4.22 的训练时间减少 **2.1 倍**——全局 token 作为前缀引导使 AR 模型收敛更快。

## 亮点与洞察
- **全局-局部分离的 tokenization 范式**：区别于 TiTok 的压缩式 1D token 和 VAR 的多尺度 token，Hita 明确区分语义全局信息和空间局部信息
- **零样本能力涌现**：训练好的 tokenizer 直接支持风格迁移（替换全局 token）和图像补全，无需额外训练
- **token 融合防坍塌机制的巧妙设计**：通过截断 patch token 的直接重建通路，强制模型利用全局信息，既解决了码本坍塌又提升了生成质量
- **与 AR 模型无缝兼容**：不需要修改 Llama 架构或引入双向注意力

## 局限性
- 仅在 ImageNet 256×256 类条件生成上验证，缺少文本引导高分辨率生成的实验
- 全局 token 的语义可解释性主要通过风格迁移间接展示，缺乏系统化的表示分析
- 增加了约 128 个额外 token（从 441→569），推理序列长度约增加 29%
- 未与最新的 VAR、MAR 等引入双向注意力的方法做公平对比

## 相关工作
- **Image Tokenizer**：VQVAE/VQGAN、ViT-VQGAN、RQ-VAE、MAGVIT-v2、TiTok、VQGAN-LC
- **AR 图像生成**：DALL-E、Parti、LlamaGen、VAR、MAR、Show-o
- **语义注入**：DINOv2 特征提取与融合

## 评分
- 新颖性：⭐⭐⭐⭐ — 全局-局部分离思路新颖，token 融合防坍塌设计精巧
- 技术深度：⭐⭐⭐⭐ — 对码本坍塌问题的分析和解决方案设计严谨
- 实验充分度：⭐⭐⭐⭐ — 全面的消融和与主流方法的对比
- 实用价值：⭐⭐⭐⭐ — 代码开源，与 Llama 无缝兼容，零样本能力实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Spectral Image Tokenizer](spectral_image_tokenizer.md)
- [\[ICCV 2025\] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)
- [\[ICCV 2025\] Grouped Speculative Decoding for Autoregressive Image Generation](grouped_speculative_decoding_for_autoregressive_image_generation.md)
- [\[ICCV 2025\] Music-Aligned Holistic 3D Dance Generation via Hierarchical Motion Modeling](music-aligned_holistic_3d_dance_generation_via_hierarchical_motion_modeling.md)
- [\[ICML 2026\] End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer](../../ICML2026/image_generation/end-to-end_autoregressive_image_generation_with_1d_semantic_tokenizer.md)

</div>

<!-- RELATED:END -->
