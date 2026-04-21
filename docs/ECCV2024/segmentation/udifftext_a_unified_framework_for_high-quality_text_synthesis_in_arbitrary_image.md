---
title: >-
  [论文解读] UDiffText: A Unified Framework for High-quality Text Synthesis in Arbitrary Images via Character-aware Diffusion Models
description: >-
  [ECCV 2024][图像分割][Text Synthesis] 提出 UDiffText，通过设计轻量级字符级文本编码器替换 CLIP encoder、引入基于字符分割图的 local attention loss 和 STR loss 微调 cross-attention 层，并在推理阶段对 noised latent 进行 refinement，实现在任意图像中合成高精度、视觉协调的文本，SeqAcc 全面超越 SOTA。
tags:
  - ECCV 2024
  - 图像分割
  - Text Synthesis
  - 扩散模型
  - Character-level Encoder
  - 注意力机制
  - Scene Text Editing
---

# UDiffText: A Unified Framework for High-quality Text Synthesis in Arbitrary Images via Character-aware Diffusion Models

**会议**: ECCV 2024  
**arXiv**: [2312.04884](https://arxiv.org/abs/2312.04884)  
**代码**: https://github.com/ZYM-PKU/UDiffText (有)  
**领域**: 文本图像生成 / 场景文字编辑  
**关键词**: Text Synthesis, diffusion model, Character-level Encoder, Cross-Attention, Scene Text Editing

## 一句话总结

提出 UDiffText，通过设计轻量级字符级文本编码器替换 CLIP encoder、引入基于字符分割图的 local attention loss 和 STR loss 微调 cross-attention 层，并在推理阶段对 noised latent 进行 refinement，实现在任意图像中合成高精度、视觉协调的文本，SeqAcc 全面超越 SOTA。

## 研究背景与动机

**领域现状**：基于扩散模型的 Text-to-Image (T2I) 生成方法（如 Stable Diffusion、DALL-E 3、Midjourney）在通用图像生成方面已达到令人惊叹的水平，但在生成包含文字的图像时经常出现严重拼写错误——字符缺失、错误或多余。

**现有痛点**：
1. CLIP/T5 文本编码器以 word/subword 为单位进行 tokenize，无法感知单词内部的字符结构
2. 即使 DALL-E 3 使用了大参数量的 T5 encoder，面对引号中的文字仍然不稳定
3. 现有方法要么依赖额外的字形图作为视觉引导（GlyphControl, GlyphDraw），要么需要拼接分割 mask 作为输入（TextDiffuser），pipeline 复杂且灵活度有限

**核心矛盾**：T2I 模型的 catastrophic neglect（灾难性忽略）和 incorrect attribute binding（属性绑定错误）导致文字渲染不准确，而现有编码范式无法提供字符级别的精确引导。

**本文切入角度**：设计轻量字符级编码器 + 局部注意力约束 + 推理时 latent refinement，仅微调 cross-attention 参数即可显著提升文字渲染精度。

**核心 idea**：用字符级编码器提供区分度高的 character embedding，让 cross-attention 层学到每个字符对应的精确空间区域。

## 方法详解

### 整体框架

基于 Stable Diffusion v2.0 Inpainting 版本构建。输入为噪声文本图像 $\mathbf{x}_0 + \mathbf{n}$、文本区域二值 mask $\mathcal{M}$ 和被 mask 遮挡后的图像 $\mathbf{x}_{\mathcal{M}}$，以及目标文本 $\mathcal{T}$。输出为在 mask 区域渲染了目标文字的图像。整体分为三步：(1) 训练字符级编码器，(2) 微调 U-Net cross-attention 层，(3) 推理阶段 noised latent refinement。

### 关键设计

1. **Character-level (CL) Text Encoder**:

    - 功能：将目标单词按字符级别编码为 embedding 序列，替代原始 CLIP encoder
    - 核心思路：构建 codebook 将字符索引映射为可学习 embedding，经 position embedding 后送入 Transformer 生成输出 $(B, L, d_{emb})$
    - 设计动机：CLIP/T5 按 word/subword tokenize，无法感知字符内部结构；ByT5 等字符级模型参数量太大（20B 级别）。本文编码器仅 302M 参数
    - 训练方式：使用对比学习损失 $\mathcal{L}_{clip} = -\text{CS}(W_t \mathbf{e}_{text}, W_i \mathbf{e}_{image})$ 对齐文本与 ViTSTR 图像特征，同时用多标签分类损失 $\mathcal{L}_{ce} = \text{CE}(\mathcal{H}_{MLC}(\mathbf{e}_{text}), Ids)$ 确保 embedding 具有高区分度
    - 总损失：$\mathcal{L} = \mathcal{L}_{clip} + \lambda_{ce} \mathcal{L}_{ce}$

2. **Local Attention Loss + STR Loss 微调训练**:

    - 功能：利用字符级分割图监督 cross-attention map，使每个字符的注意力精确聚焦于对应区域
    - 核心思路：对字符序列 $\mathcal{T} = \{c^1, c^2, \dots, c^L\}$，其对应分割图 $\mathcal{S}_T = \{S^1, S^2, \dots, S^L\}$。从 U-Net 提取 cross-attention map $\mathcal{A}_i$，计算 local attention loss：
    $\mathcal{L}_{loc} = \frac{1}{C}\sum_{i=1}^{C}\left\{\frac{1}{L}\sum_{j=1}^{L}\max(\mathbb{G}(A_i^j) \odot (J - S^j)) - \frac{1}{L}\sum_{j=1}^{L}\max(\mathbb{G}(A_i^j) \odot S^j)\right\}$
    - 补充 STR loss：用预训练 OCR 模型对去噪结果的文本区域做识别，计算交叉熵 $\mathcal{L}_{str} = \text{CE}(S(D_\theta(\cdot) \odot \mathcal{M}), \mathcal{T})$
    - 设计动机：仅用 DSM loss 的 L2 距离衡量像素均值差异，无法保证字符渲染正确性；local attention loss 迫使注意力聚焦在正确的字符区域
    - 关键点：训练时只更新 cross-attention 层参数（75.9M / 891M），冻结其余参数保持原有图像生成能力

3. **Noised Latent Refinement（推理阶段）**:

    - 功能：在推理时优化初始噪声和每步 latent，解决 catastrophic neglect 问题
    - 核心思路：(a) 采样 $N$ 个初始噪声，快速运行 2 步去噪，选择 $\mathcal{L}_{aae}$ 最小的噪声作为最优初始噪声；(b) 在每个timestep 通过梯度更新 $\mathbf{z}_t' = \mathbf{z}_t - \alpha_t \cdot \nabla_{\mathbf{z}_t} \mathcal{L}_{aae}$ 精炼 latent
    - $\mathcal{L}_{aae}$ 设计：最大化每个字符注意力在 mask 区域内的最大值的最小值，即 $\mathcal{L}_{aae} = -\frac{1}{C}\sum_{i=1}^{C}\min_{1 \le j \le N}(\max(\mathbb{G}(A_i^j) \odot \mathcal{M}))$
    - 设计动机：即使经过 local attention loss 训练，模型仍可能忽略某些字符。Refinement 确保每个字符在注意力中都被"激活"

### 损失函数 / 训练策略

完整训练目标：$\mathcal{L} = \mathcal{L}_{DSM} + \lambda_{loc}\mathcal{L}_{loc} + \lambda_{str}\mathcal{L}_{str}$

超参数设置：$\lambda_{ce} = 0.1$，$\lambda_{loc} = 0.01$，$\lambda_{str} = 0.001$

训练策略：
- CL 编码器先用对比学习训练 8k steps（batchsize=256, lr=1e-5），然后冻结
- U-Net 在 SynthText 上训练 100k steps + LAION-OCR 上训练 100k steps（batchsize=64, lr=5e-5, 图像 512×512）
- 推理时使用 50 sampling steps，CFG scale = 5.0

## 实验关键数据

### 主实验

在 ICDAR13、TextSeg、LAION-OCR 三个数据集上与 MOSTEL、SD-Inpainting、DiffSTE、TextDiffuser 对比：

| 方法 | SeqAcc-Recon (ICDAR13) | SeqAcc-Edit (ICDAR13) | FID↓ | LPIPS↓ |
|------|----------------------|---------------------|------|--------|
| MOSTEL | 68.0% | 28.0% | 25.09 | 0.0605 |
| SD-Inpainting | 29.0% | 7.0% | 26.78 | 0.0696 |
| DiffSTE | 37.0% | 29.0% | 51.67 | 0.1050 |
| TextDiffuser | 81.0% | 75.0% | 32.25 | 0.0834 |
| **UDiffText** | **91.0%** | **83.0%** | **15.79** | **0.0564** |

TextSeg 上 SeqAcc-Recon：93%（vs TextDiffuser 68%），LAION-OCR 上 SeqAcc-Editing：78%（vs TextDiffuser 64%）

### 消融实验

| 配置 | SeqAcc-Recon (%) | 说明 |
|------|----------------|------|
| Base (SD v2.0 + CLIP) | 8.0 | 基线 |
| + CL encoder | 40.0 | 字符编码器 +32% |
| + $\mathcal{L}_{loc}$ | 54.0 | 局部注意力损失 +14% |
| + $\mathcal{L}_{str}$ | 65.0 | STR 损失 +11% |
| + Refinement | 76.0 | 推理优化 +11% |

### 关键发现
- CL encoder 贡献最大（+32%），是解决文字拼写问题的核心组件
- Local attention loss 让注意力聚焦于正确字符区域，可视化清晰展示每个字符的精确 attention map
- Refinement 是收尾一击，解决 catastrophic neglect 的残余问题
- 在 SimpleBench 上将 SDXL 的文字渲染准确率从 8.0% 提升至 60.0%

## 亮点与洞察

- **知识互补微调**（Knowledge Complement）：仅更新 cross-attention 层参数，让模型在保持图像生成能力的同时学习字符形状和外观，是一种高效微调范式
- **字符分割图作为注意力监督信号**：这种将空间分割信息注入注意力机制的做法具有通用性，可迁移到任何需要精确空间控制的生成任务
- **推理时 latent refinement**：通过梯度优化 noised latent 解决 catastrophic neglect，是一种不需要重训练的推理增强策略
- **轻量字符编码器**：相比 ByT5（20B）用 302M 参数就实现了有效的字符级编码，设计简洁实用

## 局限与展望

- 当图像背景较简单时，模型依赖视觉上下文渲染文字的能力受限
- 当前只能处理最多 12 个字符的短文本，无法生成长段落或长文档
- 推理时的 refinement 需要额外的前向和反向传播，增加了推理时间
- 未来可考虑逐词合成来处理长文本序列

## 相关工作与启发

- **vs TextDiffuser**: TextDiffuser 拼接分割 mask 作为输入、使用 character-aware loss，但需要额外的 segmentor 模块。UDiffText 更简洁，仅用文本条件即可
- **vs GlyphControl**: GlyphControl 用 ControlNet 注入字形参考图像，需要预渲染参考图。UDiffText 直接从字符编码器获取引导
- **vs Attend-and-Excite**: borrowing 了 generative semantic nursing 的思路，但针对文字渲染任务设计了新的 $\mathcal{L}_{aae}$ 目标

## 评分

- 新颖性: ⭐⭐⭐⭐ 字符级编码器 + local attention loss + 推理 refinement 三件套设计完整
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多任务评估，消融实验清晰，可视化有说服力
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，方法描述详细，公式规范
- 价值: ⭐⭐⭐⭐ 文字渲染是 T2I 生成的核心痛点，方法实用且有开源代码

<!-- RELATED:START -->

## 相关论文

- [SegGen: Supercharging Segmentation Models with Text2Mask and Mask2Img Synthesis](seggen_supercharging_segmentation_models_with_text2mask_and_mask2img_synthesis.md)
- [UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis](../../ICCV2025/segmentation/uniglyph_unified_segmentation-conditioned_diffusion_for_precise_visual_text_synt.md)
- [Diffusion Models for Open-Vocabulary Segmentation](diffusion_models_for_open-vocabulary_segmentation.md)
- [EAFormer: Scene Text Segmentation with Edge-Aware Transformers](eaformer_scene_text_segmentation_with_edge-aware_transformers.md)
- [Text-guided Controllable Diffusion for Realistic Camouflage Images Generation](../../AAAI2026/segmentation/text-guided_controllable_diffusion_for_realistic_camouflage_images_generation.md)

<!-- RELATED:END -->
