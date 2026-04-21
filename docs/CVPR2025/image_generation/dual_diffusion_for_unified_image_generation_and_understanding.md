---
title: >-
  [论文解读] Dual Diffusion for Unified Image Generation and Understanding
description: >-
  [图像生成] 提出 Dual Diffusion Transformer (D-DiT)，在单一 MM-DiT 架构中同时使用连续扩散建模图像分布和离散掩码扩散建模文本分布，是首个端到端的全扩散多模态模型，支持图像生成、图像描述和视觉问答等全套任务。
tags:
  - 图像生成
---

# Dual Diffusion for Unified Image Generation and Understanding

## 一句话总结

提出 Dual Diffusion Transformer (D-DiT)，在单一 MM-DiT 架构中同时使用连续扩散建模图像分布和离散掩码扩散建模文本分布，是首个端到端的全扩散多模态模型，支持图像生成、图像描述和视觉问答等全套任务。

## 研究背景与动机

当前多模态生成领域存在明显的方法割裂：扩散模型在文本到图像生成(T2I)上表现优异(如 Stable Diffusion、FLUX)，而自回归模型在视觉语言理解(I2T)上占据主导(如 LLaVA、BLIP-2)。自然的问题是：能否将这两种能力统一在一个模型中？

对于自回归模型，已有大量工作证明其可以反向生成图像(LLM + visual tokenizer)。但对于扩散模型，实现反向文本生成一直很困难，原因在于缺乏经验上有效的离散扩散过程。现有的多模态扩散模型(UniDiffuser、Versatile Diffusion)要么依赖自回归模型解码文本 latent，要么在预训练 LLM 上附加扩散损失(Show-O、Transfusion)，本质上仍依赖 next-token prediction 进行文本生成。

本文的核心动机是：利用离散掩码扩散(masked diffusion)的最新进展，构建一个**纯扩散**的多模态模型，无需任何自回归组件即可同时完成图像生成和文本生成。

## 方法详解

### 整体框架

D-DiT 基于 SD3 的 MM-DiT 双分支 Transformer 架构。图像分支输出 flow matching 的速度场预测，文本分支输出离散掩码扩散的去噪 token 预测。两个分支在每一层注意力中相互交互（cross-attention）。训练时，使用联合损失函数 $L_{\text{dual}} = L_{\text{image}} + \lambda_{\text{text}} L_{\text{text}}$ 同时训练两个模态的条件生成。

### 关键设计

#### 1. 跨模态联合扩散训练

- **功能**：在同一模型中同时学习 $p(\mathbf{x}^{(\text{img})}|\mathbf{x}^{(\text{txt})})$ 和 $p(\mathbf{x}^{(\text{txt})}|\mathbf{x}^{(\text{img})})$
- **核心思路**：图像分支使用连续 flow matching（速度场回归），文本分支使用离散掩码扩散（masked token prediction）。训练时不同时对两个模态加噪——当训练文本生成时图像保持无噪声，反之亦然
- **设计动机**：不同模态天然适合不同扩散方式：连续向量适合 flow matching，离散 token 适合 absorbing state diffusion。联合训练允许两个模态共享 Transformer 参数，形成统一表示

#### 2. 基于掩码扩散的文本 In-filling 机制

- **功能**：支持视觉问答(VQA)等需要条件文本生成的任务
- **核心思路**：在采样时将问题 token 保持不变（不加掩码），只对答案部分进行掩码并迭代去噪。利用离散扩散的天然 in-filling 能力完成条件文本补全
- **设计动机**：此前的扩散多模态模型(UniDiffuser、Versatile Diffusion)在 CLIP latent 空间做文本扩散，无法进行 token 级别的条件补全。掩码扩散直接在 token 空间工作，天然支持 in-filling

#### 3. 从 SD3 预训练权重初始化与三阶段训练

- **功能**：快速适应文本生成能力，同时保持图像生成质量
- **核心思路**：利用 SD3 预训练的 DiT 权重初始化，在文本分支顶部添加线性头。三阶段训练：(1) 30M 图文对上联合预训练 60K 步；(2) 高质量理解数据集上微调 200K 步并解冻 mask token embedding；(3) 视觉指令微调
- **设计动机**：SD3 的 MM-DiT 已有强大的图文对齐能力，只需少量文本数据即可适配文本生成。使用 T5 编码器中已有的 `<extra_id0>` 作为 mask token，减少域差距

### 损失函数

$$L_{\text{image}} = \mathbb{E}_{t, q^{(\text{img})}} \|\mathbf{v}_\theta(\mathbf{x}_t^{(\text{img})}, t, \mathbf{x}^{(\text{txt})}) - (\epsilon - \mathbf{x}^{(\text{img})})\|_2^2$$

$$L_{\text{text}} = \mathbb{E}_{q^{(\text{txt})}} \left[-\frac{1}{K}\sum_{i=1}^{K} \log[\mathbf{x}_\theta(\mathbf{x}_{t_i}^{(\text{txt})}, \mathbf{x}^{(\text{img})}) \cdot \mathbf{x}] / t_i \right]$$

$$L_{\text{dual}} = L_{\text{image}} + \lambda_{\text{text}} L_{\text{text}}$$

## 实验关键数据

### 主实验表

**T2I 生成 (GenEval)**：

| 模型 | Overall | Single Obj | Two Obj | Counting | Colors | Position | Color Attr |
|------|---------|-----------|---------|----------|--------|----------|------------|
| SD3 | 0.62 | 0.98 | 0.74 | 0.63 | 0.67 | 0.34 | 0.36 |
| D-DiT (ours) | **0.65** | 0.97 | **0.80** | 0.54 | **0.76** | 0.32 | **0.50** |
| Show-O | 0.68 | 0.98 | 0.80 | 0.66 | 0.84 | 0.31 | 0.50 |

**I2T 理解 (VQA/Captioning)**：

| 模型 | VQAv2 | VizWiz | GQA | POPE | MME |
|------|-------|--------|-----|------|-----|
| D-DiT 256 | 60.7 | 33.9 | 52.2 | 79.7 | 1089 |
| D-DiT 512 | 64.1 | 35.1 | 55.1 | 83.2 | 1213 |
| Show-O | 61.0 | 28.6 | 48.7 | 82.0 | 1097 |

### 消融表

- 联合训练**不会导致图像生成灾难性遗忘**：D-DiT 保持了 SD3 的图像生成质量，部分指标(颜色准确性)甚至有所提升
- 文本 in-filling 采样步数对 VQA (短文本) 用 16 步即可，对话/长文本用 256 步

### 关键发现

- 从 SD3 预训练检查点初始化后，仅需 25B text token 的训练量即可产生有意义的文本输出
- 纯扩散模型在 VQA 任务上首次实现了与 Show-O 等混合模型的竞争力
- 掩码扩散的并行采样特性使得文本生成可以非自回归进行

## 亮点与洞察

1. **范式突破**：首个端到端纯扩散多模态模型，证明扩散模型可以同时建模连续图像和离散文本，不依赖自回归解码
2. **优雅的训练目标**：联合损失函数极其简洁，仅是两个模态扩散损失的加权和，无需复杂的训练策略
3. **强迁移能力**：利用 SD3 预训练权重和 T5 的 mask token，以极少的文本数据快速获得文本生成能力
4. **架构对称性**：图像和文本分支在同一 Transformer 中对称处理，通过 cross-attention 自然交互

## 局限与展望

1. **文本理解性能仍有差距**：与专门的 VLM (LLaVA-1.5, InternVL) 相比，VQA 性能仍有明显差距
2. **未训练纯文本生成**：模型从未在纯文本数据上训练，限制了其语言建模能力
3. **训练数据规模较小**：仅用 ~40M 图文对训练，远少于主流 VLM 的训练数据量
4. **推理效率**：离散扩散的迭代采样比自回归生成慢，尤其是长文本需要 256 步
5. **模型规模**：2B 参数，远小于主流 VLM，scaling 行为值得探索

## 相关工作与启发

- **Show-O / Transfusion**：混合扩散+自回归方案，仍依赖 AR 做文本生成
- **MDLM / SEDD**：离散扩散语言建模的理论基础，使得纯扩散文本生成成为可能
- **SD3 / FLUX**：MM-DiT 架构的基础，提供了天然的双分支结构
- **启发**：跨模态统一的关键不在于统一扩散过程，而在于统一架构+共享参数，让不同模态使用各自最适合的扩散方式

## 评分

⭐⭐⭐⭐

创新性强（首个纯扩散多模态模型），方法简洁优雅，但当前性能与自回归 VLM 仍有差距，实用性有待提升。

<!-- RELATED:START -->

## 相关论文

- [DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](dual-interrelated_diffusion_model_for_few-shot_anomaly_image_generation.md)
- [Dual Prompting Image Restoration with Diffusion Transformers (DPIR)](dual_prompting_image_restoration_with_diffusion_transformers.md)
- [InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](../../NeurIPS2025/image_generation/infinitystar_unified_spacetime_autoregressive_modeling_for_v.md)
- [LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](../../ICCV2025/image_generation/lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)
- [Multi-party Collaborative Attention Control for Image Customization](multi-party_collaborative_attention_control_for_image_customization.md)

<!-- RELATED:END -->
