---
title: >-
  [论文解读] Rare Text Semantics Were Always There in Your Diffusion Transformer
description: >-
  [NeurIPS 2025][图像生成][Transformer] 发现通过在 MM-DiT 的联合注意力块前对文本 token 嵌入进行方差放大,即可在无需额外训练或外部模块的情况下让扩散模型呈现稀有文本语义。
tags:
  - NeurIPS 2025
  - 图像生成
  - Transformer
  - MM-DiT
  - 稀有语义
  - 文本到图像
  - 方差缩放
---

# Rare Text Semantics Were Always There in Your Diffusion Transformer

**会议**: NeurIPS 2025

**arXiv**: [2510.03886](https://arxiv.org/abs/2510.03886)

**代码**: 无

**领域**: 图像生成

**关键词**: 扩散Transformer, MM-DiT, 稀有语义, 文本到图像, 方差缩放

## 一句话总结

发现通过在 MM-DiT 的联合注意力块前对文本 token 嵌入进行方差放大,即可在无需额外训练或外部模块的情况下让扩散模型呈现稀有文本语义。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：基于 flow 和 diffusion 的多模态扩散 Transformer（MM-DiT）已成为文本到视觉生成的主流架构。用户不断用富有想象力的稀有 prompt 测试模型能力，但先进模型仍难以准确生成这些概念。

核心问题：

**稀有概念生成困难**: 训练数据中出现频率低的概念（如特定风格、罕见物体组合）生成质量差

**现有解决方案笨重**: 需要额外训练步骤、更多数据、去噪时优化或依赖外部 LLM 重写 prompt

**根因不明**: 稀有语义是否已经编码在模型中?为什么未能呈现?

## 方法详解

### 整体框架

分析 MM-DiT 中联合注意力机制对文本嵌入的处理过程，发现稀有语义虽然编码在模型中，但被 softmax 注意力的集中效应所抑制。通过简单的方差放大恢复这些语义。

### 关键设计

**1. MM-DiT 联合注意力分析**

- MM-DiT 中文本和图像嵌入在每个 Transformer 块中通过联合注意力序列更新
- 稀有概念的文本 token 在嵌入空间中与常见概念距离较近
- softmax 注意力倾向于将注意力集中在主要（常见）语义上,边缘化稀有语义

**2. 方差缩放干预 (Variance Scale-Up)**

核心操作极其简单: 在联合注意力块之前,对文本 token 嵌入的每个维度进行方差放大：

$$\tilde{z}_t = \alpha \cdot (z_t - \bar{z}_t) + \bar{z}_t$$

其中 $z_t$ 是文本 token 嵌入, $\bar{z}_t$ 是均值, $\alpha > 1$ 是缩放因子。

**3. 数学直觉**

- 方差放大扩展了文本 token 嵌入在表征空间中的"影响半径"
- 原本被主要语义掩盖的稀有语义得以在注意力计算中获得更大权重
- 不改变嵌入的中心方向(均值保持不变),仅增加分辨率

### 损失函数 / 训练策略

- **无需训练**: 这是一种推理时干预方法
- **无需数据**: 不需要额外数据或校准
- **无需外部模块**: 不依赖 LLM 或其他模型
- 仅需在推理时添加一行代码的缩放操作

## 实验关键数据

### 主实验

稀有 prompt 生成质量评估 (CLIP-T Score / 人工评分):

| 方法 | CLIP-T Score | 人工偏好率 | 额外推理时间 |
|------|-------------|----------|------------|
| SD3 (基线) | 0.285 | - | 0% |
| + Prompt Rewrite (LLM) | 0.301 | 42% | +35% |
| + Denoising Opt. | 0.312 | 48% | +200% |
| + Ours (方差缩放) | **0.328** | **65%** | **+0.5%** |

不同任务上的泛化性:

| 任务 | 基线 CLIP-T | + 方差缩放 | 提升 |
|------|-----------|----------|------|
| 文本到图像 | 0.285 | 0.328 | +15.1% |
| 文本到视频 | 0.268 | 0.305 | +13.8% |
| 文本驱动图像编辑 | 0.312 | 0.345 | +10.6% |

### 消融实验

缩放因子 $\alpha$ 的影响:

| $\alpha$ | CLIP-T (稀有) | CLIP-T (常见) | FID |
|---------|-------------|-------------|-----|
| 1.0 (无缩放) | 0.285 | 0.332 | 12.5 |
| 1.2 | 0.305 | 0.330 | 12.3 |
| 1.5 | 0.328 | 0.328 | 12.8 |
| 2.0 | 0.335 | 0.315 | 14.2 |
| 3.0 | 0.318 | 0.295 | 18.5 |

### 关键发现

1. 稀有语义确实已编码在模型中，只是被 softmax 注意力抑制
2. $\alpha = 1.5$ 是最优的平衡点: 稀有语义显著改善,常见语义几乎不受影响
3. 过大的 $\alpha$ 会扰乱生成质量，FID 显著增加
4. 方法在 text-to-image、text-to-video、text-driven editing 三个任务上均有效

## 亮点与洞察

- **极简而有效**: 仅一行代码的修改带来显著改善，是真正的"free lunch"
- **新洞察**: 揭示了 MM-DiT 中稀有语义被抑制的机制,具有理论价值
- **跨任务泛化**: 方法在多个文本到视觉任务上都有效

## 局限与展望

1. 对所有文本 token 统一缩放可能不是最优的,应该只放大"稀有" token
2. 如何自动确定最优 $\alpha$ 值是一个开放问题
3. 极端稀有概念（训练中完全未见）仍然无法生成
4. 仅在 SD3/FLUX 等 MM-DiT 架构上验证,U-Net结构（如SD1.5）可能不适用

## 相关工作与启发

- **Stable Diffusion 3**: 基于 MM-DiT 的开源生成模型
- **FLUX**: 改进的 flow-based 生成模型
- **Attend-and-Excite**: 通过注意力引导改善文本对齐的先驱工作

## 评分

- ⭐ 创新性: 9/10 — 发现极简但被忽视的改善方向,洞察力强
- ⭐ 实用性: 9/10 — 零计算开销,即插即用
- ⭐ 写作质量: 8/10 — 可视化效果好,理论分析直观

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] SparseDiT: Token Sparsification for Efficient Diffusion Transformer](sparsedit_token_sparsification_for_efficient_diffusion_transformer.md)
- [\[ICCV 2025\] Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation](../../ICCV2025/image_generation/dense2moe_restructuring_diffusion_transformer_to_moe_for_eff.md)
- [\[NeurIPS 2025\] Linear Differential Vision Transformer: Learning Visual Contrasts via Pairwise Differentials](linear_differential_vision_transformer_learning_visual_contrasts_via_pairwise_di.md)
- [\[CVPR 2025\] Towards Transformer-Based Aligned Generation with Self-Coherence Guidance](../../CVPR2025/image_generation/towards_transformer-based_aligned_generation_with_self-coherence_guidance.md)
- [\[CVPR 2025\] LaVin-DiT: Large Vision Diffusion Transformer](../../CVPR2025/image_generation/lavin-dit_large_vision_diffusion_transformer.md)

<!-- RELATED:END -->
