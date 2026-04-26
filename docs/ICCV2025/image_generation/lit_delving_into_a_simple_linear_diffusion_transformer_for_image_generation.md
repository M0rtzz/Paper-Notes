---
title: >-
  [论文解读] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation
description: >-
  [图像生成] > 系统研究如何将预训练 DiT 安全高效地转换为线性注意力版本 LiT，提出 5 条实践指南（深度卷积增强、少头策略、权重继承、选择性加载、混合蒸馏），仅需 DiT 训练步数的 20% 即可达到可比性能。
tags:
  - 图像生成
---

# LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation

| 信息 | 内容 |
|------|------|
| 会议 | ICCV 2025 |
| arXiv | [2501.12976](https://arxiv.org/abs/2501.12976) |
| 代码 | - |
| 领域 | 图像生成 · 扩散模型 · 高效架构 |
| 关键词 | linear attention, diffusion transformer, knowledge distillation, efficient generation, DiT |

## 一句话总结

> 系统研究如何将预训练 DiT 安全高效地转换为线性注意力版本 LiT，提出 5 条实践指南（深度卷积增强、少头策略、权重继承、选择性加载、混合蒸馏），仅需 DiT 训练步数的 20% 即可达到可比性能。

## 研究背景与动机

### DiT 的效率瓶颈

扩散 Transformer (DiT) 在图像生成中展现了强大的商业潜力，但其自注意力模块的二次复杂度 $\mathcal{O}(N^2D)$ 在高分辨率任务中带来严重的延迟和显存问题：
- 2048px 分辨率下，线性注意力比 softmax 注意力快近 **9 倍**
- DiT-S/2 在 2048px 下显存从 ~14GB 降至 ~4GB

### 现有方案的不足

已有高效注意力方案（如 SANA、Mediators、CLEAR）为 DiT 引入了精心设计的修改，但**牺牲了架构的简洁性**。问题是：

> 能否使用最简单的纯线性注意力，让 LiT 成为 DiT 的安全、高效替代品？

### 研究分解

将问题分解为两部分：
1. 什么类型的线性注意力架构适合图像生成？
2. 什么训练策略能有效优化线性 DiT？

## 方法详解

### 整体框架

LiT 保持 DiT 的宏观/微观架构设计不变，仅将所有 softmax 自注意力替换为简单的线性注意力。通过 5 条实践指南确保替换后的性能。

### 线性注意力基础

标准自注意力：$\text{Sim}(\mathbf{Q}, \mathbf{K}) = \exp(\mathbf{Q}\mathbf{K}^\top / \sqrt{d})$，复杂度 $\mathcal{O}(N^2D)$

线性注意力用核函数 $\phi(\cdot)$ 替代：

$$\mathbf{O}_i = \frac{\phi(\mathbf{Q}_i) \left(\sum_{j=1}^{N} \phi(\mathbf{K}_j)^\top \mathbf{V}_j\right)}{\phi(\mathbf{Q}_i) \left(\sum_{j=1}^{N} \phi(\mathbf{K}_j)^\top\right)}$$

复杂度降至 $\mathcal{O}(ND^2/h)$。

### Guideline 1：深度卷积即可

直接使用 ReLU 线性注意力替换 softmax 注意力会导致 FID 大幅上升（S/2: 68.40→88.46）。实验对比三种修复方案：

| 方案 | DiT-S/2 FID↓ | DiT-B/2 FID↓ |
|------|:---:|:---:|
| Softmax 基线 | 68.40 | 43.47 |
| ReLU 线性基线 | 88.46 | 56.92 |
| + 深度卷积 (DWC) | **63.66** | **42.11** |
| + Focused Linear | 63.05 | 40.58 |
| + GELU 核 | 70.83 | 58.86 |

**仅加一个深度卷积（kernel=5）即超越 softmax 注意力！** 原因：预测噪声时，模型倾向于依赖邻近像素信息，DWC 弥补了线性注意力缺乏的局部性。Focused function（设计用于分类的锐化分布）在噪声预测中反而不必要。

### Guideline 2：少头=免费午餐

理论上，线性注意力的计算量与头数 $h$ 负相关：$\mathcal{O}(ND^2/h)$。但实验发现——

**减少头数增加 GMACs 但不增加实际延迟！**

| DiT | 头数 | FID↓ | IS↑ |
|-----|:---:|:---:|:---:|
| S/2 | 2 | **63.24** | 22.07 |
| S/2 | 6 (默认) | 63.66 | 22.16 |
| S/2 | 96 | 78.76 | 17.46 |
| XL/2 | 4 | **20.82** | 65.52 |
| XL/2 | 16 (默认) | 21.69 | 63.06 |

少头策略提升了理论计算预算，按 Scaling Law 提高了模型的性能上限。此外，不同头之间的注意力图高度同质（余弦相似度 >0.5），说明少数头即可承载主要信息。

### Guideline 3：从收敛的 DiT 初始化

| 预训练步数 | FID↓ |
|:---:|:---:|
| 无预训练 | 63.24 |
| 200K | 57.84 |
| 400K | 56.07 |
| 600K | 54.80 |
| 800K | **53.83** |

预训练越充分的 DiT 权重越适合作为 LiT 的初始化，即使两者架构不完全一致。原因可能是 DiT 中不同模块的功能解耦——FFN、adaLN 等共享组件的知识可以直接迁移。

### Guideline 4：不加载注意力权重

| 加载策略 | FID↓ |
|----------|:---:|
| 不加载注意力 | **54.80** |
| 加载 Q, K, V | 55.29 |
| 加载 K, V | 55.07 |
| 加载 V | 54.93 |
| 加载 Q | 54.82 |

线性注意力与 softmax 注意力的计算范式不同（线性注意力直接计算 $\mathbf{K}^\top \mathbf{V}$），强行加载反而干扰优化。建议加载除注意力外的所有预训练参数。

### Guideline 5：混合知识蒸馏

不仅蒸馏预测噪声，还蒸馏反向扩散过程的方差：

$$\mathcal{L} = \mathcal{L}_{\text{simple}} + \lambda_1 \underbrace{\|\epsilon^{(\mathcal{T})} - \epsilon^{(\mathcal{S})}\|^2}_{\text{噪声蒸馏}} + \lambda_2 \underbrace{\|\Sigma^{(\mathcal{T})} - \Sigma^{(\mathcal{S})}\|^2}_{\text{方差蒸馏}}$$

| $\lambda_1$ | $\lambda_2$ | FID↓ |
|:---:|:---:|:---:|
| 0.0 | 0.0 (无蒸馏) | 53.83 |
| 0.5 | 0.0 (仅噪声) | 51.13 |
| 0.0 | 0.05 (仅方差) | 53.49 |
| **0.5** | **0.05** | **50.79** |

方差蒸馏需适度（$\lambda_2=0.05$），因为去噪能力才是扩散模型的核心。

## 实验

### 类条件 ImageNet 256×256

| 模型 | 训练步数 | FID↓ | IS↑ |
|------|:---:|:---:|:---:|
| DiT-XL/2 | 400K | 19.47 | - |
| DiG-XL/2 (GLA) | 400K | 18.53 | 68.53 |
| **LiT-XL/2** | **100K** | **12.90** | **95.80** |
| DiT-XL/2-G (cfg=1.50) | 7M | 2.27 | 278.24 |
| **LiT-XL/2-G (cfg=1.50)** | **1.4M (20%)** | **2.32** | **265.20** |

LiT 仅需 DiT **20% 的训练步数** (1.4M vs 7M) 即达到可比 FID (2.32 vs 2.27)。

### 类条件 ImageNet 512×512

| 模型 | FID↓ |
|------|:---:|
| DiT-XL/2-G (cfg=1.50) | 3.04 |
| **LiT-XL/2-G (cfg=1.50)** | **3.69** |

512px 下仅用 ~23% 的训练步数（700K vs 3M），FID 差距仅 0.65。

### 文本到图像（GenEval 评估）

| 模型 | 参数量 | Overall↑ |
|------|:---:|:---:|
| PixArt-Σ | 0.6B | 0.52 |
| SDv2.1 | 0.9B | 0.50 |
| **LiT (1024px)** | **0.6B** | **0.48** |

从 PixArt-Σ 转换的 LiT 保持了可比的 GenEval 分数，证明指南可推广到文本到图像生成。

## 亮点与洞察

1. **可操作的实践指南**：提供了 5 条即插即用的规则，降低了线性 DiT 的实践门槛
2. **"免费午餐"发现**：少头线性注意力增加 GMACs 但不增加延迟，这是一个有趣的硬件特性洞察
3. **效率惊人**：LiT-XL/2 仅 100K 步即超越 DiT-XL/2 的 400K 步结果
4. **可在笔记本上部署**：作者在 Windows 11 笔记本上离线部署 LiT-0.6B，实现 1K 分辨率图像生成

## 局限性

- 在 512px 下 FID 仍有 0.65 的差距（3.69 vs. 3.04），线性注意力在高分辨率下仍有（小幅）性能代价
- 文本到图像的 GenEval 总分略低于 PixArt-Σ（0.48 vs. 0.52），在多对象/位置/属性上有差距
- 方差蒸馏的理论解释不够充分
- 仅在 DiT 和 PixArt-Σ 上验证，未扩展到更多架构（如 SD3、FLUX）

## 相关工作

- **高效注意力**：SANA (Mix-FFN)、Attention Mediators、CLEAR (循环窗口)
- **线性注意力**：Flatten Transformer、EfficientViT、FLatten 等
- **扩散模型 SSM/GLA**：DiG (GLA)、DiM (SSM)、DiffuSSM
- **知识蒸馏**：噪声蒸馏、采样步数蒸馏等

## 评分

| 维度 | 分数 |
|------|:----:|
| 创新性 | ⭐⭐⭐⭐ |
| 有效性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐⭐⭐ |
| 综合推荐 | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)
- [\[ICCV 2025\] OminiControl: Minimal and Universal Control for Diffusion Transformer](ominicontrol_minimal_and_universal_control_for_diffusion_transformer.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](../../CVPR2025/image_generation/dual_diffusion_for_unified_image_generation_and_understanding.md)
- [\[ICCV 2025\] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)
- [\[ICCV 2025\] Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers](lay-your-scene_natural_scene_layout_generation_with_diffusion_transformers.md)

<!-- RELATED:END -->
