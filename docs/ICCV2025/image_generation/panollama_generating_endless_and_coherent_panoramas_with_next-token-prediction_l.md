---
title: >-
  [论文解读] PanoLlama: Generating Endless and Coherent Panoramas with Next-Token-Prediction LLMs
description: >-
  [图像生成] 提出 PanoLlama，通过 token 重定向策略将固定尺寸的视觉自回归（VAR）模型扩展为无限全景生成，实现免训练的 next-crop prediction，在连贯性、保真度和美学上超越联合扩散等方法。
tags:
  - 图像生成
---

# PanoLlama: Generating Endless and Coherent Panoramas with Next-Token-Prediction LLMs

> **会议**: ICCV 2025
> **arXiv**: [2411.15867](https://arxiv.org/abs/2411.15867)
> **代码**: [GitHub](https://github.com/0606zt/PanoLlama)
> **领域**: 全景图生成·自回归模型·图像生成
> **关键词**: panoramic image generation, next-token prediction, LlamaGen, token redirection, training-free

## 一句话总结

提出 PanoLlama，通过 token 重定向策略将固定尺寸的视觉自回归（VAR）模型扩展为无限全景生成，实现免训练的 next-crop prediction，在连贯性、保真度和美学上超越联合扩散等方法。

## 研究背景与动机

全景图生成（Panoramic Image Generation, PIG）旨在生成任意长度的连贯图像，在艺术创作、历史修复等场景有广泛需求。现有方法的局限：

**联合扩散（Joint Diffusion）方法的多层一致性难题**：MultiDiffusion、SyncDiffusion 等方法将全景潜空间分割为裁剪块，独立去噪后融合。它们依赖启发式连接策略（加权平均、梯度引导、深度注意力），难以同时保证低级（颜色、边缘）和高级（语义、布局）的多层连贯性。

**Inpainting 方法的视野受限**：仅根据前一个裁剪块推断下一个内容，缺乏全局布局和结构规划。

**VAR 模型的固定尺寸限制**：LlamaGen 等自回归模型虽然天然适合序列生成，但受训练范式限制只能生成固定大小图像（如 512×512）。

**核心洞察**：全景生成的本质——在保持多级连贯性的同时逐步扩展图像——天然对齐 next-token prediction 范式。低级连贯性依赖相邻裁剪块的连接，高级连贯性需要考虑整个序列的全局过渡，这正是自回归模型擅长的。

## 方法详解

### 理论建模

将全景图 $x'$ 分解为有序裁剪块序列 $\{x_i\}$，建模为联合概率分布：

$$P(x') = \prod_{i=1}^{n} P(x_i | x_1, x_2, \ldots, x_{i-1})$$

与 inpainting（仅条件于 $x_{i-1}$）和联合扩散（条件于 $x_{i-1}, x_{i+1}$）相比，自回归范式利用了**所有前序裁剪块**的信息。

### 整体框架（Fig. 2）

PanoLlama 由三部分构成：
1. **文本编码**：文本 prompt $y$ 经编码器 $f_\mathcal{E}$ 编码为条件嵌入 $s$
2. **Next-Crop Prediction**：token 生成器 $f_\mathcal{G}$ 自回归生成图像 token，并通过重定向策略扩展
3. **Token 解码**：连接的 token 序列 $V$ 经图像 tokenizer 解码器 $f_{\mathcal{T}d}$ 转为全景图

### 关键设计：Token 重定向（Training-Free）

**垂直扩展**：当位置索引 $k$ 达到 token 上限 $p$ 时，重定向到 $p - r\sqrt{p}$ 重新开始，以 $v_1$ 的最后 $p - r\sqrt{p}$ 个 token 作为 $v_2$ 的起始条件：

$$v_2 = f_\mathcal{G}(v_{1, r\sqrt{p}}, \ldots, v_{1, p})$$

**水平扩展（交错法）**：逐行扩展，每行 $v_{i-1}^j$ 的最后 $\sqrt{p} - c$ 个 token 作为 $v_i^j$ 的起始条件，每次扩展 $c$ 列：

$$v_i^j = f_\mathcal{G}(v_{i-1, \epsilon(v_{i-1}^j) - \sqrt{p} + c}, \ldots, v_{i-1, \epsilon(v_{i-1}^j)})$$

扩展步幅 $u = c / \sqrt{p}$ 控制质量与效率的平衡。

### 与现有方法的统一视角

| 方法 | 建模 | 条件范围 |
|------|------|---------|
| Inpainting | $P(x_i | x_{i-1})$ | 仅左邻 |
| Joint Diffusion | $P(x_i | x_{i-1}, x_{i+1})$ | 左右邻 |
| **PanoLlama** | $P(x_i | x_1, \ldots, x_{i-1})$ | **所有前序** |

## 实验

### 主实验：定量比较（Tab. 1, 512×5120 全景图）

| 方法 | LPIPS ↓ | TV ↓ | SSIM ↑ | FID ↓ (相对) | CLIP-aesthetic ↑ | 时间 ↓ |
|------|---------|------|--------|------------|-----------------|--------|
| MultiDiffusion | 0.694 | 0.061 | 0.184 | +3.16 | 6.84 | 1809s |
| SyncDiffusion | 0.582 | 0.058 | 0.263 | +8.75 | 6.94 | 7233s |
| MAD | 0.520 | 0.040 | 0.268 | +23.09 | 6.90 | 1924s |
| StreamMD | 0.637 | 0.055 | 0.257 | +53.50 | 6.75 | 241s |
| **PanoLlama** | **0.410** | **0.021** | **0.305** | **+2.27** | **6.97** | **726s** |

在**连贯性**（核心指标）上，PanoLlama 全面碾压所有基线：
- TV 较最佳基线 MAD 提升 **47.50%**
- LPIPS 提升 21.15%
- SSIM 提升 13.81%
- FID 相对损失仅 +2.27（最小），推理速度 3-10× 快于大多数方法

### 扩展步幅消融（Fig. 4）

| 步幅 $u$ | PanoLlama COH | MAD COH | MultiDiffusion COH |
|----------|--------------|---------|-------------------|
| 1/8 | 0.18 | 0.35 | 0.52 |
| 3/4 | 0.19 | 0.42 | 0.58 |
| 1 (无重叠) | 0.24 | 0.72 | 0.80 |

关键发现：随步幅增大，其他方法质量急剧下降，而 PanoLlama 保持稳定低 COH 分数——对扩展步幅具有极强鲁棒性，实现了更优的质量-效率平衡。

### 全景尺寸消融（Fig. 5）

从 2× 到 10× 分辨率，其他 PIG 方法随尺寸增大连贯性明显下降，PanoLlama 保持稳定——有效处理更大全景的挑战。

### 用户研究

1000 个 prompt × 2000 张全景图的大规模评估，涵盖 25 个主题 100+ 子主题。PanoLlama 在宏大场景（海景、草原）上表现尤佳，在复杂密集场景（人群、图案）上有一定挑战。

## 亮点与洞察

1. **范式创新**：将 PIG 从联合扩散重新定义为 next-crop prediction，理论上更优（利用全部前序信息）
2. **免训练（Training-Free）**：仅通过 token 重定向即可将固定尺寸 VAR 模型推广到无限全景生成
3. **丰富应用**：支持多尺度扩展、无 mask 布局控制、多引导合成——这些是其他 PIG 方法无法实现的
4. **新基准**：构建了 1000 prompt × 100+ 主题的标准化评估数据集

## 局限性

- 受预训练 VAR 模型的固定 token 容量限制，仅通过部分前序 token 近似全局依赖
- 基于 LlamaGen 的图像质量受限于该模型本身的生成能力
- 水平扩展中的交错生成策略增加了实现复杂性

## 相关工作

- 联合扩散：MultiDiffusion、SyncDiffusion、TwinDiffusion、MAD
- Inpainting 全景：BLD
- 视觉自回归：LlamaGen、VQGAN、MaskGIT

## 评分

- **新颖性**: ★★★★★ — 全景生成范式的根本重构
- **技术深度**: ★★★★☆ — token 重定向策略简洁有效
- **实验质量**: ★★★★★ — 大规模评估、多维度消融、新基准
- **写作质量**: ★★★★★ — 统一理论视角清晰，对比分析精到

<!-- RELATED:START -->

## 相关论文

- [TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](../../CVPR2026/image_generation/tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)
- [REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder](regen_learning_compact_video_embedding_with_re-generative_decoder.md)
- [Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)

<!-- RELATED:END -->
