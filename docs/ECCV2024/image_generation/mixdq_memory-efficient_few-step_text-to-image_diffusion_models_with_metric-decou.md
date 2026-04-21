---
title: >-
  [论文解读] MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed Precision Quantization
description: >-
  [ECCV 2024][图像生成] 针对少步扩散模型量化的特殊挑战，提出 MixDQ 混合精度量化方法，通过 BOS 感知量化处理文本嵌入中的异常值、度量解耦的敏感性分析分离质量与内容影响，在 1-step SDXL-turbo 上实现 W4A8 无损量化。
tags:
  - ECCV 2024
  - 图像生成
---

# MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed Precision Quantization

**会议**: ECCV 2024  
**arXiv**: [2405.17873](https://arxiv.org/abs/2405.17873)  
**领域**: 图像生成

## 一句话总结

针对少步扩散模型量化的特殊挑战，提出 MixDQ 混合精度量化方法，通过 BOS 感知量化处理文本嵌入中的异常值、度量解耦的敏感性分析分离质量与内容影响，在 1-step SDXL-turbo 上实现 W4A8 无损量化。

## 研究背景与动机

少步扩散模型（如 SDXL-turbo）将去噪步骤从 10-100 步压缩到 1-4 步，大幅降低计算量。但模型内存消耗仍达 5-10GB，限制了移动端部署。

然而，对少步模型量化面临独特挑战：

**量化敏感性更高**: 多步模型中迭代去噪可补偿量化误差，但少步模型缺少这个纠错机制。实验发现 30-step SDXL 的 W8A8 量化影响很小，但 1-step SDXL-turbo 的 W8A8 即完全崩溃

**文本对齐被破坏**: 量化不仅降低图像质量，还改变生成内容（如提示"山羊"却生成"北极熊"），现有方法完全忽略了这一问题

**敏感性分布长尾**: 层间敏感性差异极大，少数高敏感层成为瓶颈，统一位宽策略被拖累

## 方法详解

### 整体框架

MixDQ 由三个步骤组成：
1. BOS 感知量化 — 解决高敏感层的文本嵌入异常值
2. 度量解耦敏感性分析 — 分离图像质量和内容的敏感性
3. 整数规划位宽分配 — 在给定预算下获得最优混合精度配置

### 关键设计

**1. BOS 感知量化**

分析发现大量高敏感层来自交叉注意力中的 to_k 和 to_v 线性层，它们的输入是 CLIP 文本嵌入。文本嵌入中第 1 个 token（BOS，句首标记）的特征幅度极大（823.5 vs 其余 token 的 10-15），直接量化会导致大部分值趋近于 0。

关键发现：BOS token 的特征对所有提示都相同。因此：
- 预计算 BOS token 经过 to_k/to_v 层的浮点输出并缓存
- 量化时跳过 BOS token，仅对其余 token 进行量化
- 推理时将缓存的浮点输出与量化反量化的其余 token 拼接
- 额外存储开销仅 640-1280 元素/层

**2. 度量解耦敏感性分析**

现有 SQNR 敏感性度量存在致命问题：内容改变（高质量但与原图不同的图像）导致的 SQNR 下降（-3.51dB）远大于质量退化（可辨识的伪影、-0.26dB）。这导致位宽分配过度保护内容相关层而忽视质量相关层。

解决方案——将层和度量双重解耦：
- **内容相关层**（交叉注意力 + FFN）→ 用 SSIM 度量内容变化
- **质量相关层**（自注意力 + 卷积）→ 用 SQNR 度量质量退化

两组独立评估敏感性，避免"不公平竞争"。

**3. 整数规划位宽分配**

将位宽配置建模为整数规划问题，候选位宽 $b \in \{2, 4, 8\}$：

$$\max_{c_{i,b}} \sum_{i=1}^{N}\sum_{b} c_{i,b} \cdot \mathcal{S}_{i,b} \quad \text{s.t.} \sum_{b} c_{i,b} = 1, \quad \sum_{i,b} c_{i,b} \cdot \mathcal{M}_{i,b} \leq \mathcal{B}$$

对内容相关层和质量相关层分别求解，使用 OR-tools 库可在秒级完成求解。

### 损失函数

MixDQ 为后训练量化方法（PTQ），不涉及训练或损失函数。仅使用 1024 个 COCO 提示进行校准。

## 实验关键数据

### 主实验

**SDXL-turbo (1-step) 在完整 COCO 上的量化比较**

| 方法 | 位宽 (W/A) | 压缩倍率 | FID ↓ | CLIP Score ↑ | ImageReward ↑ |
|------|-----------|---------|-------|-------------|---------------|
| FP16 | 16/16 | 1× | 17.15 | 0.2722 | 0.8631 |
| Naive PTQ | 8/8 | 4× | 103.96 | 0.1478 | -1.7446 |
| Q-Diffusion | 8/8 | 4× | 76.18 | 0.1772 | -1.3112 |
| Q-Diffusion | 4/16 | 4× | 22.58 | 0.2685 | 0.6847 |
| **MixDQ** | **4/16** | **4×** | **17.23** | **0.2693** | **0.8254** |
| **MixDQ** | **4/8** | **8×** | **17.68** | **0.2698** | **0.7822** |
| **MixDQ** | **3.66/16** | **4.4×** | **17.40** | **0.2682** | **0.7528** |

现有方法在 W8A8 即完全失败，MixDQ 在 W4A8（8 倍压缩）下 FID 仅增加 0.5。

**W8A8 与更多量化方法的对比**

| 方法 | FID ↓ | CLIP ↑ | ImageReward ↑ |
|------|-------|--------|---------------|
| FP16 | 84.51 | 0.26 | 0.84 |
| Q-Diffusion | 149.15 (+64.6) | 0.16 (-0.10) | -1.69 |
| PTQD | 340.74 (+256.2) | 0.12 (-0.14) | -2.28 |
| EDA-DM | 137.98 (+53.5) | 0.16 (-0.10) | -1.71 |
| Non-Uniform FP8 | 101.73 (+17.2) | 0.24 (-0.01) | 0.16 |
| **MixDQ** | **83.39 (-1.12)** | **0.27 (+0.01)** | **0.84 (+0.00)** |

MixDQ 是唯一在 W8A8 下达到零性能损失的方法。

### 消融实验

**硬件资源节省 (RTX 4080)**

| 配置 (W/A) | 内存节省 | 推理加速 |
|-----------|---------|---------|
| W8A8 | 1.87× | 1.52× |
| W4A16 | 3.03× | 1× |
| W4A8 | 3.03× | 1.52× |

W4A8 配置同时实现 3 倍内存减少和 1.5 倍延迟加速；量化层仅引入约 0.03ms overhead。

### 关键发现

- BOS token 特征幅度是其余 token 的 50-80 倍，是量化崩溃的主要元凶
- BOS token 在不同提示下特征完全相同，可以安全地预计算并跳过量化
- SQNR 对内容变化的惩罚远大于对质量退化的惩罚，导致基于 SQNR 的位宽分配严重偏向保护内容层
- 保留仅 1% 最敏感层为 FP16 就能大幅提升整体性能，开销可忽略
- 少步模型比多步模型量化困难得多，迭代去噪本质上是一种误差修正机制

## 亮点与洞察

1. **BOS 异常值发现极具洞察力**: 将 CLIP 的 "attention sink" 现象与量化失败联系起来，解决方案简洁（跳过 BOS）且几乎无开销
2. **度量解耦思想有广泛价值**: 不仅适用于量化，任何多指标优化的模型压缩任务（如剪枝、蒸馏）都应该考虑指标间的交互和偏差
3. **实验结果令人印象深刻**: 在所有现有方法 W8A8 即崩溃的场景下，MixDQ 实现 W4A8 几乎无损，差距巨大
4. **整数规划秒级求解**: 避免了搜索式混合精度的高昂计算成本

## 局限性

- 仅在 SDXL-turbo 和 LCM-lora 两个模型上验证，对更多架构的泛化性待确认
- 2-bit 量化配置下质量仍有明显退化
- BOS 感知量化依赖 CLIP 的特定行为，其他文本编码器（如 T5）是否有类似现象未讨论
- 校准数据（1024 个 COCO 提示）的选择对结果的影响未深入分析

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — BOS 感知量化和度量解耦分析均为全新贡献，洞察深刻
- **技术深度**: ⭐⭐⭐⭐⭐ — 问题分析透彻，从现象到方法的推导逻辑清晰
- **实验充分度**: ⭐⭐⭐⭐⭐ — FP16/多种量化方法对比 + 硬件实测 + 完整消融
- **写作质量**: ⭐⭐⭐⭐⭐ — 图示精美，现象-分析-方法-验证的逻辑链完整

<!-- RELATED:START -->

## 相关论文

- [Memory-Efficient Fine-Tuning for Quantized Diffusion Model](memory-efficient_fine-tuning_for_quantized_diffusion_model.md)
- [Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis](rejection_sampling_imle_designing_priors_for_better_few-shot_image_synthesis.md)
- [M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)
- [LCM-Lookahead for Encoder-based Text-to-Image Personalization](lcm-lookahead_for_encoder-based_text-to-image_personalization.md)

<!-- RELATED:END -->
