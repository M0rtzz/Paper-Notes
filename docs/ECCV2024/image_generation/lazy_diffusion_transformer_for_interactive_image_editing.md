---
title: >-
  [论文解读] Lazy Diffusion Transformer for Interactive Image Editing
description: >-
  [ECCV 2024][图像生成] 提出 LazyDiffusion，一种非对称编码器-解码器 Transformer 架构，通过上下文编码器压缩全局信息并仅在 mask 区域执行扩散去噪，实现了与全图生成方法质量相当但速度提升 10 倍的交互式图像编辑。
tags:
  - ECCV 2024
  - 图像生成
---

# Lazy Diffusion Transformer for Interactive Image Editing

**会议**: ECCV 2024  
**arXiv**: [2404.12382](https://arxiv.org/abs/2404.12382)  
**领域**: 图像生成

## 一句话总结

提出 LazyDiffusion，一种非对称编码器-解码器 Transformer 架构，通过上下文编码器压缩全局信息并仅在 mask 区域执行扩散去噪，实现了与全图生成方法质量相当但速度提升 10 倍的交互式图像编辑。

## 研究背景与动机

当前基于扩散模型的 inpainting 方法存在严重的计算浪费：

**RegenerateImage 方法**（如 SDXL、SD Inpaint）：生成整张图片后仅保留 mask 区域像素，其余全部丢弃

**RegenerateCrop 方法**（如实际软件中常用的裁剪方案）：仅处理 mask 周围的小区域裁切，虽然更快但丢失了全局上下文，导致语义不一致

交互式编辑场景中，用户的修改通常仅覆盖图像的 10-20%，全图重新生成极度浪费。本文提出"懒惰"生成策略——只生成需要的像素。

## 方法详解

### 整体框架

LazyDiffusion 解耦生成过程为两个步骤：

1. **全局上下文编码器 $E$**（ViT）：处理整张画布和 mask，提取 $N=4096$ 个 token，然后**仅保留** mask 区域对应的 $N_{hole}$ 个 token 作为压缩上下文。编码器在扩散循环外仅运行一次
2. **增量扩散解码器 $D$**（PixArt-α 变体）：在每个去噪步骤中，仅处理 mask 区域对应的 token，条件化在压缩上下文和文本提示词上

**关键思想**：编码器的自注意力使每个 token 能编码全局信息，因此丢弃非 mask token 后仍保留了完整的语义上下文。

### 关键设计

**Token Dropping**：编码器输出的 $N$ 个 token 通过 max-pooling mask 筛选，仅保留 mask 区域的 $N_{hole}$ 个 token。这创建了信息瓶颈，鼓励编码器将全局上下文压缩到 mask 位置的 token 中。

**上下文条件化**：解码器通过在隐藏维度上拼接噪声 token $\mathcal{X}_{hole}^t$ 和上下文 token $\mathcal{T}_{hole}$ 实现条件化：$\mathcal{X}_{hole}^{t-1} = D(\mathcal{X}_{hole}^t \oplus \mathcal{T}_{hole}; t, \mathbf{c})$

**潜在空间操作**：使用 Stable Diffusion 的预训练 VAE（8× 下采样），在 4 通道潜在空间中操作。最终输出通过 Poisson blending 消除接缝。

**训练**：编码器从头训练，解码器从 PixArt-α 预训练权重初始化。联合训练 100K iterations，56 张 A100 GPU，batch size 224。

### 损失函数

使用 Improved DDPM 目标函数进行训练，对 mask 区域的潜在像素进行去噪重建。

## 实验关键数据

### 主实验

在 OpenImages 10K 图像上的定量比较（1024×1024 分辨率）：

| 方法 | CLIP Score ↑ | FID ↓ | 备注 |
|---|---|---|---|
| SD2-crop | 0.21 | 6.95 | 参考，不同架构 |
| SDXL | 0.21 | 6.88 | 参考，不同架构 |
| RegenerateCrop (PixArt) | 0.19 | 9.35 | 裁剪方案 |
| RegenerateImage (PixArt) | 0.19 | 7.38 | 全图生成 |
| **LazyDiffusion** | **0.19** | **7.70** | 仅 mask 区域 |

运行时间对比（10% mask，单张 A100）：

| 方法 | 运行机制 | 延迟 |
|---|---|---|
| RegenerateImage | 全图 4096 token | ~374ms/步 |
| RegenerateCrop | 固定裁剪 | 固定 |
| **LazyDiffusion** | 仅 mask token | **~28ms/步 (10% mask)** |

用户研究（1778 个回复，48 名用户）：

| 对比方法 | 用户偏好 LazyDiffusion |
|---|---|
| vs RegenerateCrop | **81.0%** |
| vs SD2-crop | **82.5%** |
| vs RegenerateImage | 46.1% |
| vs SDXL | 48.5% |

### 消融实验

编码器开销分析：

| 组件 | 耗时 |
|---|---|
| 上下文编码器 | 73ms（仅运行一次） |
| 潜在空间编码器 | 97ms |
| 潜在空间解码器 | 176ms |
| T5 文本编码器 | 21ms |
| 扩散解码器（10% mask，单步）| 28ms |
| 扩散解码器（全图，单步）| 374ms |

### 关键发现

- 10% mask 时实现 **10× 加速**，25% mask 时与 RegenerateCrop 持平
- FID 仅比全图生成方法增加 4%（7.70 vs 7.38），但比裁剪方法低 26%（7.70 vs 9.35）
- 压缩上下文保留了核心语义信息——在需要高度语义一致性的场景（如托盘上生成相同样式的面包）中表现与全图方法相当
- 加速优势在高扩散步数和高分辨率场景下更为显著

## 亮点与洞察

1. **问题定义精准**：从"生成全图再裁切"到"只生成需要的像素"，计算量与编辑区域成正比
2. **MAE 逆向设计**：与 Masked Autoencoder 相反——编码器处理所有 token，解码器仅处理 mask token
3. **渐进式交互**：将图像生成成本分摊到多次用户交互中，真正实现了扩散模型的交互式使用
4. **正交贡献**：与快速采样、蒸馏等扩散加速方法正交，可叠加使用
5. **支持多模态条件**：除文本外，还支持草图引导等局部条件化（类 SDEdit）

## 局限性

- 编码器仍需处理全图（二次方复杂度），对超高分辨率图像可能成为瓶颈
- 生成区域偶尔出现微妙的颜色偏移，需要 Poisson blending 后处理
- 训练数据为内部数据集（2.2 亿图像），难以完全复现

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 非对称编码解码 + token dropping 的设计优雅且有效
- **实用性**: ⭐⭐⭐⭐⭐ — 10× 加速使扩散模型首次可用于交互式编辑管线
- **实验充分度**: ⭐⭐⭐⭐ — 定量+用户研究+渐进式生成 demo，但缺少开源基线对比
- **论文质量**: ⭐⭐⭐⭐⭐ — 写作清晰，图示丰富，动机→方法→实验逻辑严密

<!-- RELATED:START -->

## 相关论文

- [ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)
- [XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](xpsr_cross-modal_priors_for_diffusion-based_image_super-resolution.md)
- [WildVidFit: Video Virtual Try-On in the Wild via Image-Based Controlled Diffusion Models](wildvidfit_video_virtual_try-on_in_the_wild_via_image-based_controlled_diffusion.md)
- [Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)
- [OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)

<!-- RELATED:END -->
