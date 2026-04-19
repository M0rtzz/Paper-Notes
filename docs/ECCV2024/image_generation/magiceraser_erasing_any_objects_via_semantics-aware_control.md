---
title: >-
  [论文解读] MagicEraser: Erasing Any Objects via Semantics-Aware Control
description: >-
  [ECCV 2024][图像生成] 提出基于扩散模型的对象擦除框架 MagicEraser，通过内容初始化、提示调优和语义感知注意力重聚焦三阶段设计，无需用户输入文本即可高质量擦除对象并生成和谐背景。
tags:
  - ECCV 2024
  - 图像生成
---

# MagicEraser: Erasing Any Objects via Semantics-Aware Control

**会议**: ECCV 2024  
**arXiv**: [2410.10207](https://arxiv.org/abs/2410.10207)  
**领域**: 图像生成

## 一句话总结

提出基于扩散模型的对象擦除框架 MagicEraser，通过内容初始化、提示调优和语义感知注意力重聚焦三阶段设计，无需用户输入文本即可高质量擦除对象并生成和谐背景。

## 研究背景与动机

对象擦除任务要求移除图像中指定对象并生成与周围环境和谐的背景，是图像修复的一个特殊子任务。现有方法面临两大挑战：

**GAN 方法的局限**: LaMa、MAT 等方法在简单重复纹理（天空、草地）上表现良好，但面对复杂纹理或光照不一致的背景时，生成内容模糊且不协调

**扩散模型方法的困境**: Stable Diffusion Inpainting 需要高质量文本提示才能产生合理结果。短提示（如"湖上的船"）容易生成新的不期望对象，而获取精确长描述对普通用户极不友好

深层原因在于：(1) 传统修复训练使用随机掩码，模型学到的是"恢复缺失区域"而非"生成和谐背景"；(2) 扩散模型的全局文本条件与局部擦除区域之间存在语义不对齐。

## 方法详解

### 整体框架

MagicEraser 基于 Stable Diffusion Inpainting 构建，分为两阶段：

**阶段一：内容初始化** — 使用预训练 GAN 模型（Big-LaMa）粗略填充擦除区域
**阶段二：可控生成** — 两个即插即用模块控制扩散生成过程

### 关键设计

**1. 内容初始化**

直接从随机噪声生成（denoising strength s=1）容易偏离原始图像。使用预训练修复模型（LaMa）初始化擦除区域内容，再编码为 VAE 潜变量作为扩散起点（s=0.9），兼顾纹理和谐与避免生成不必要对象。

**2. 提示调优 (Prompt Tuning)**

结合 Textual Inversion 和 LoRA 实现无需用户输入的自动提示：
- 定义占位符 "R∗" 表示"背景补全"概念，训练其 token embedding v∗
- 利用全景分割自动获取背景标签（如"天空""沙滩"），构建提示 "A photo of R∗ sky"
- 以 50% 概率使用短提示、50% 概率使用 LLAVA 生成的长描述进行训练
- 仅训练 LoRA 参数和 v∗，避免破坏预训练模型的生成能力

**3. 语义感知注意力重聚焦**

利用 Mask2Former 全景分割结果将像素分为三类：
- **掩码区域 (m)**: 需要填充的区域
- **正区域 (p)**: 语义属于背景的区域
- **负区域 (n)**: 语义类似于被擦除对象的区域

调制 self-attention：增强掩码区域与正区域的注意力交互，抑制与负区域和自身的交互：

$$M = W_{pos} \cdot Mask_{pos} - W_{neg} \cdot Mask_{neg}$$

$$A' = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d}}\right)$$

**4. 训练数据构建 (OLRD)**

创新的数据构建策略：从原始图像中选取对象，将其平移到分割标注为背景的区域并混合，构建"对象-掩码-干净背景"三元组，直接教会模型"擦除对象恢复背景"的概念。

### 损失函数

LoRA 微调 + Textual Inversion 的联合优化：

$$L = \mathbb{E}_{z_0, z_{masked}, m, t, y, \epsilon_t}\left[\|\epsilon_t - \epsilon_{\theta, \phi}(z'_t, t, \tau(y), m)\|^2\right]$$

## 实验关键数据

### 主实验

| 方法 | 数据集 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|------|--------|--------|--------|---------|-------|
| MAT | OpenImages | 26.994 | 0.949 | 0.030 | 31.30 |
| Co-Mod | OpenImages | 26.446 | 0.941 | 0.033 | 30.40 |
| LaMa | OpenImages | 21.618 | 0.936 | 0.055 | 37.10 |
| SD Inpainting | OpenImages | 26.096 | 0.942 | 0.036 | 31.10 |
| **MagicEraser** | **OpenImages** | **28.123** | **0.947** | **0.032** | **30.02** |
| MAT | RealHM | 21.484 | 0.843 | 0.107 | 51.73 |
| SD Inpainting | RealHM | 21.758 | 0.846 | 0.116 | 45.05 |
| **MagicEraser** | **RealHM** | **23.620** | **0.861** | **0.101** | **46.56** |

MagicEraser 在三个数据集上 PSNR 指标大幅领先，OpenImages 上超过 MAT 1.1dB，RealHM 上超过 SD Inpainting 1.9dB。

### 消融实验

| 模型配置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|---------|--------|--------|---------|-------|
| Baseline + 随机掩码训练 | 21.331 | 0.815 | 0.134 | 52.10 |
| Baseline + OLRD | 22.130 | 0.834 | 0.119 | 50.73 |
| + 内容初始化 | 22.891 | 0.840 | 0.109 | 48.91 |
| + 注意力重聚焦 | 23.277 | 0.844 | 0.110 | 48.93 |
| + 提示调优 | 23.311 | 0.858 | 0.104 | 47.94 |
| **MagicEraser (全部)** | **23.620** | **0.861** | **0.101** | **46.56** |

每个组件均有正贡献：OLRD 数据构建是基础（+0.8dB），内容初始化提供良好起点（+0.7dB），提示调优贡献大于注意力重聚焦。

### 关键发现

- MagicEraser 超过商业产品 Adobe Photoshop Generative Fill（23.620 vs 22.913 PSNR）和 Google Photos Eraser（vs 20.310）
- 提示调优（全局语义控制）比注意力重聚焦（局部空间控制）更重要，两者互补性好
- OLRD 数据构建相比传统随机掩码训练带来 0.8dB PSNR 提升
- 训练免注意力重聚焦模块是免训练的，仅在初始去噪步骤（t=1~0.7）调制

## 亮点与洞察

1. **用户友好设计**: 无需手动输入文本提示，通过自动全景分割和学习的 R∗ 概念完成擦除，大幅降低使用门槛
2. **数据构建策略巧妙**: 将对象平移到背景区域的做法避免了传统修复训练的"恢复前景"偏差，直接训练模型进行"背景补全"
3. **语义感知注意力调制**: 不同于搜索式 loss 优化注意力图，直接修改 attention score 更高效且无需训练
4. **实际应用价值高**: 超越商业产品的性能表明该方法具有落地潜力

## 局限性

- 对全景分割模型的依赖较强，分割质量直接影响正/负区域划分的准确性
- 仅在 512×512 分辨率下实验，高分辨率场景性能待验证
- λ_pos 和 λ_neg 等超参数需手动设定
- 对于擦除区域占图像比例极大的情况，性能可能退化

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 语义感知注意力重聚焦和 OLRD 数据构建策略新颖
- **技术深度**: ⭐⭐⭐⭐ — 多阶段协同设计完整，每个组件有明确动机
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三个数据集 + 商业产品对比 + 完整消融
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，与传统修复的区别阐述到位

<!-- RELATED:START -->

## 相关论文

- [ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs](ziplora_any_subject_in_any_style_by_effectively_merging_loras.md)
- [LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [Text2Place: Affordance-aware Text Guided Human Placement](text2place_affordance-aware_text_guided_human_placement.md)
- [NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation](neusdfusion_a_spatial-aware_generative_model_for_3d_shape_completion_reconstruct.md)
- [Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)

<!-- RELATED:END -->
