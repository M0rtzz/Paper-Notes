---
title: >-
  [论文解读] MacDiff: Unified Skeleton Modeling with Masked Conditional Diffusion
description: >-
  [ECCV 2024][图像生成] 首次将扩散模型用于骨架表征学习，提出 Masked Conditional Diffusion（MacDiff）框架，通过语义编码器提取掩码骨架的表征来引导条件扩散解码器进行去噪，统一了骨架的判别式和生成式建模。
tags:
  - ECCV 2024
  - 图像生成
---

# MacDiff: Unified Skeleton Modeling with Masked Conditional Diffusion

**会议**: ECCV 2024  
**arXiv**: [2409.10473](https://arxiv.org/abs/2409.10473)  
**领域**: 图像生成

## 一句话总结

首次将扩散模型用于骨架表征学习，提出 Masked Conditional Diffusion（MacDiff）框架，通过语义编码器提取掩码骨架的表征来引导条件扩散解码器进行去噪，统一了骨架的判别式和生成式建模。

## 研究背景与动机

骨架数据的自监督学习主要分为对比学习和重建两大范式。对比学习方法依赖正负样本对的构造，存在假负样本问题，且只学习判别性信息，泛化能力有限。重建方法（如 MAE）过度关注低层次信号重建，表征中包含大量与高层语义无关的信息。

虽然扩散模型在图像生成领域取得了巨大成功，但其表征学习能力尚未被充分探索，尤其面对骨架数据的空间稀疏性和时间冗余性。直接使用扩散模型预测噪声并不能显式地学习有意义的判别式表征。因此，如何挖掘扩散模型在骨架表征学习中的潜力，同时保持生成能力，成为了值得探索的方向。

## 方法详解

### 整体框架

MacDiff 由两个核心组件组成：
1. **语义编码器 (Semantic Encoder)**：接收掩码后的骨架序列，提取高层次的紧凑表征
2. **去噪解码器 (Denoising Decoder)**：以编码器输出的表征为条件，执行条件扩散去噪

输入骨架序列首先被分割为时间维度上的 patch 并嵌入为 token。对编码器输入施加 90% 的高掩码率随机掩码，既引入信息瓶颈去除冗余，又加速训练。编码器输出局部表征后通过池化得到全局表征。

### 关键设计

1. **Patchify + Random Masking**: 将 T₀×V×3 的骨架数据沿时间维度分割为 patch，90% 的高掩码率构建紧致信息瓶颈
2. **AdaLN 条件注入**: 解码器中用 Adaptive Layer Norm 替代标准 LN，通过 scale 和 shift 操作将编码器表征注入去噪过程
3. **局部-全局表征融合**: 将未掩码位置填充局部表征、掩码位置填充全局表征，解决过平滑问题，保留 token 多样性
4. **Inverse-cosine 噪声调度**: 将所有时间步的噪声水平拉向中等程度，优于常用的 cosine 或 linear 调度
5. **扩散数据增强**: 利用预训练扩散解码器生成标签保持的训练数据，在标注数据稀缺时显著提升微调性能

### 损失函数

标准条件扩散损失，采用 ε-prediction：

$$\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon}\left[\|\epsilon - \mathcal{D}(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t, \mathcal{E}(\mathcal{M}(x_0)))\|^2\right]$$

理论分析表明，MacDiff 的生成目标等价于对比学习目标（对齐掩码视图与加噪视图的互信息 $I(Z; X_t)$）加上互补重建目标（$I(X; Z|X_t)$），后者要求表征包含更多对比学习遗漏的任务相关信息。

## 实验关键数据

### 主实验

| 方法 | 类型 | NTU60 xsub | NTU60 xview | NTU120 xsub | NTU120 xset | PKU I |
|------|------|------------|-------------|-------------|-------------|-------|
| 3s-CrosSCLR | 对比学习 | 77.8 | 83.4 | 67.9 | 66.7 | 84.9 |
| 3s-AimCLR | 对比学习 | 78.9 | 83.8 | 68.2 | 68.8 | 87.4 |
| 3s-ActCLR | 对比学习 | 84.3 | 88.8 | 74.3 | 75.7 | - |
| MAMP | 重建(J) | 84.9 | 89.1 | 78.6 | 79.1 | 92.2 |
| PCM3 | 多任务(J) | 83.9 | 90.4 | 76.5 | 77.5 | - |
| **MacDiff** | **生成(J)** | **86.4** | **91.0** | **79.4** | **80.2** | **92.8** |

线性评估中，MacDiff 仅用单流（Joint）即超越了所有三流集成方法和先前的 MAE 方法。

### 消融实验

| 配置 | NTU60 xsub | NTU60 xview |
|------|------------|-------------|
| SkeletonMAE (Transformer) | 88.5 | 94.7 |
| MAMP (Transformer) | 93.1 | 97.5 |
| MotionBERT (DSTformer) | 93.0 | 97.2 |
| UPS (Transformer, 监督) | 92.6 | 97.0 |
| **MacDiff (Transformer)** | **92.7** | **97.3** |

在监督微调模式下，MacDiff 达到与全监督统一模型 UPS 和 MotionBERT 可比的性能，验证了扩散模型学到的表征具有高度通用性。

### 半监督微调与数据增强

| 方法 | NTU60 xsub 1% | NTU60 xsub 10% | NTU60 xview 1% | NTU60 xview 10% |
|------|---------------|----------------|----------------|-----------------|
| SkeletonMAE | 54.4 | 80.6 | 54.6 | 83.5 |
| MAMP | 66.0 | 88.0 | 68.7 | 91.5 |
| MacDiff (无增强) | 65.6 | 88.2 | 77.3 | 92.5 |
| **MacDiff (有增强)** | **72.0** | **89.2** | **79.2** | **93.1** |

扩散数据增强在 1% 标注数据下带来 6.4% 的巨大提升（65.6→72.0），超越 MAMP 6.0%。

### 关键发现

- 90% 掩码率是最优选择，50%/80% 分别为 82.7/83.8，而 0%（无掩码）仅 79.3
- Inverse-cosine 噪声调度显著优于 cosine 和 linear 调度
- 增强数据与真实数据的最优比例随标注量递减：1%/2%/10% 时分别为 2.0/0.5/0.25
- 运动重建中 MacDiff 的 MPJPE（0.033）仅为 SkeletonMAE（0.191）的 1/6
- 局部-全局表征融合有效缓解了 Transformer 的过平滑问题

## 亮点与洞察

1. **首次证明扩散模型可以作为有效的骨架表征学习器**，打破了生成模型表征能力不足的刻板印象
2. **理论分析非常扎实**：从互信息角度证明 MacDiff 目标包含对比学习并额外保留更多任务相关语义，提供了下游性能的更强理论保证
3. **统一框架**：同一模型同时支持动作识别（判别）和骨架生成/数据增强（生成），避免了预训练组件的浪费
4. **掩码率的巧妙选择**：90% 高掩码率既约束了表征维度，又大幅减少计算量

## 局限性

- 目前仅在骨架数据上验证，是否能推广到更复杂的序列数据（如视频、点云）有待探索
- 扩散采样仍需 DDIM 多步迭代，生成速度较 MAE 类方法慢
- 多模态输入（如 RGB+骨架）的融合未被讨论
- 对比学习和生成学习的贡献解耦分析不够细致

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将扩散模型用于骨架自监督表征学习，理论分析角度新颖
- **技术深度**: ⭐⭐⭐⭐⭐ — 互信息理论分析完整严谨，信息瓶颈设计精巧
- **实验充分度**: ⭐⭐⭐⭐ — 三大数据集 + 多种评估协议 + 消融实验丰富
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，理论与实验结合紧密

<!-- RELATED:START -->

## 相关论文

- [NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model](nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)
- [SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)
- [Lazy Diffusion Transformer for Interactive Image Editing](lazy_diffusion_transformer_for_interactive_image_editing.md)
- [Memory-Efficient Fine-Tuning for Quantized Diffusion Model](memory-efficient_fine-tuning_for_quantized_diffusion_model.md)
- [Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)

<!-- RELATED:END -->
