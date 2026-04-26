---
title: >-
  [论文解读] Enhancing Creative Generation on Stable Diffusion-based Models
description: >-
  [CVPR 2025][图像生成][创意生成] 本文提出 C3（Creative Concept Catalyst），一种免训练方法，通过在 Stable Diffusion 的去噪过程中选择性放大特征来增强创意生成能力，并提供基于创意两个主要维度的放大因子选择指南。
tags:
  - CVPR 2025
  - 图像生成
  - 创意生成
  - 扩散模型
  - 特征放大
  - 去噪过程
  - 免训练
---

# Enhancing Creative Generation on Stable Diffusion-based Models

**会议**: CVPR 2025  
**arXiv**: [2503.23538](https://arxiv.org/abs/2503.23538)  
**代码**: 无  
**领域**: 扩散模型 / 图像生成  
**关键词**: 创意生成, Stable Diffusion, 特征放大, 去噪过程, 免训练

## 一句话总结
本文提出 C3（Creative Concept Catalyst），一种免训练方法，通过在 Stable Diffusion 的去噪过程中选择性放大特征来增强创意生成能力，并提供基于创意两个主要维度的放大因子选择指南。

## 研究背景与动机

**领域现状**：Stable Diffusion 及其蒸馏变体（如 SDXL-Turbo、LCM 等）在文本到图像生成方面已达到高保真度和强文本-图像对齐，成为创意内容生成的主流工具。用户通常通过精心设计的 prompt 来控制生成内容的风格和创意程度。

**现有痛点**：尽管这些模型生成质量很高，但**创意能力受限**。在 prompt 中加入"creative"等词汇很少能真正产生预期的创意效果。模型倾向于生成符合训练数据分布的"典型"图像，难以突破常规产生新颖、意外的视觉组合。现有增强创意的方法大多需要额外训练（如微调模型权重）或复杂的 prompt 工程，计算成本高且不灵活。

**核心矛盾**：扩散模型的训练目标是学习数据分布并从中采样，这天然倾向于生成分布中的高概率样本。而创意生成恰恰需要偏离高概率区域，探索低概率但有意义的组合。直接增大噪声或随机性会导致质量下降而非创意提升。

**本文目标**：在不额外训练的前提下，增强 Stable Diffusion 基模型的创意生成能力，同时保持生成质量。

**切入角度**：作者分析了扩散模型去噪过程中不同特征的作用，发现某些中间特征与创意程度相关。选择性放大这些特征可以推动生成结果偏离"典型"区域，产生更具创意的输出。

**核心 idea**：在去噪过程中对 U-Net 的特定层特征施加放大因子，选择性增强与创意相关的特征通道，无需修改模型权重或额外训练。

## 方法详解

### 整体框架
C3 是一个即插即用的推理时方法。在标准 Stable Diffusion 的去噪循环中，C3 在特定时间步对 U-Net 内部的特征图施加放大操作。放大策略基于创意的两个维度（新颖性和多样性）设定不同的放大因子。最终输出保持了较高的图像质量，同时展现出更丰富的创意表达。

### 关键设计

1. **选择性特征放大**:

    - 功能：在去噪过程中增强与创意相关的特征表示
    - 核心思路：在 U-Net 的特定层（主要是中间层和解码器早期层），对特征图按通道维度施加放大因子 $\alpha$。放大操作为 $\hat{f} = \alpha \cdot f$，其中 $f$ 为原始特征，$\alpha > 1$ 为放大系数。放大并非对所有特征均匀施加，而是选择性地作用于与创意相关的特征通道。具体地，在去噪的早期时间步（决定全局结构和语义的阶段）施加较强的放大，在后期时间步（细化细节的阶段）减弱或不放大。
    - 设计动机：去噪早期阶段决定图像的全局布局和概念组合，放大此阶段的特征推动模型探索非典型的概念组合。后期阶段放大可能导致伪影，因此需要时间步自适应的策略。

2. **创意双维度放大指南**:

    - 功能：提供系统性的放大因子选择策略
    - 核心思路：将创意分解为两个主要维度——**新颖性**（Novelty，生成结果与常见图像的偏离程度）和**多样性**（Diversity，多次生成结果之间的差异程度）。对于增强新颖性，需要在特征空间中推动生成轨迹远离数据分布的模式；对于增强多样性，需要在采样过程中增大随机性的表达空间。不同的放大因子组合对应不同的创意偏好：高新颖性 + 低多样性产生独特但风格一致的创意输出；低新颖性 + 高多样性产生风格多变但不太激进的创意输出。
    - 设计动机：创意是一个多维概念，不能用单一参数控制。双维度框架让用户可以根据具体需求（如概念设计 vs 风格探索）灵活调整创意方向。

3. **免训练即插即用设计**:

    - 功能：在不修改模型权重的情况下增强创意
    - 核心思路：C3 仅在推理时修改 U-Net 的前向传播过程，对特定层的输出做放大操作。不需要梯度回传、不修改 attention 权重、不需要额外的编码器或适配器。整个方法可以用几行代码实现在任何基于 Stable Diffusion 的模型上。
    - 设计动机：免训练方法的计算成本几乎为零，且天然兼容所有 Stable Diffusion 变体（SD1.5、SDXL、SD Turbo、LCM 等），部署极为方便。

### 损失函数 / 训练策略
本方法无需训练，仅在推理阶段对特征进行放大操作。

## 实验关键数据

### 主实验：创意生成效果

| 模型 | 方法 | 新颖性评分 | 多样性评分 | 图像质量（FID） |
|------|------|-----------|-----------|-------------|
| SD 1.5 | baseline | 基线 | 基线 | 基线 |
| SD 1.5 | + C3 | 显著提升 | 显著提升 | 轻微下降 |
| SDXL | baseline | 基线 | 基线 | 基线 |
| SDXL | + C3 | 显著提升 | 显著提升 | 轻微下降 |
| SD Turbo | + C3 | 显著提升 | 显著提升 | 轻微下降 |

### 消融实验

| 配置 | 新颖性 | 质量 | 说明 |
|------|-------|------|------|
| Full C3 | 高 | 良好 | 完整方法 |
| 均匀放大（不按时间步） | 中 | 差 | 后期时间步放大导致伪影 |
| 仅放大解码器 | 中 | 良好 | 缺少对全局语义的影响 |
| 仅放大编码器 | 低 | 良好 | 对创意提升有限 |

### 关键发现
- 放大因子 $\alpha$ 的选择对新颖性和质量有直接 trade-off：$\alpha$ 过大导致失真，$\alpha$ 过小创意不足
- 中间层特征对创意的贡献最大，浅层和深层特征放大效果有限
- 在去噪早期（约前 30-50% 时间步）施加放大效果最佳
- 方法在所有测试的 SD 变体上均有效，体现了通用性

## 亮点与洞察
- **极致简洁**：几行代码就能在任意 Stable Diffusion 模型上增强创意，不需要训练、不需要数据、不需要额外模型。这种"推理时干预"的思路在实际部署中非常有价值。
- **创意量化框架**：将创意分解为新颖性和多样性两个可独立调控的维度，为创意生成的评估和控制提供了结构化框架。
- **可迁移到其他生成任务**：特征放大的思路不限于 Stable Diffusion，理论上可以应用于任何基于去噪的生成模型（如 DiT、视频扩散等），用于探索生成空间的低概率区域。

## 局限与展望
- **创意评估的主观性**：创意本身就是一个主观概念，不同人对"创意"的理解不同。自动化评估指标可能无法完全反映人类对创意的感知
- **放大因子的敏感性**：不同的 prompt 和模型可能需要不同的放大因子，缺少自动化的参数选择机制
- **可能产生不连贯内容**：过度放大可能导致语义不连贯的图像，如不合理的物体组合或扭曲的结构
- **质量下降**：虽然标注为"轻微"，但 FID 的上升说明创意增强是以一定质量为代价的
- 未来可探索：自适应放大因子选择、结合 CLIP 评分的创意质量平衡、扩展到视频和 3D 生成

## 相关工作与启发
- **vs Prompt 工程**：通过修改文本 prompt 来引导创意（如使用"surreal""abstract"等词汇），但效果有限且不可控；C3 在特征层面直接干预，控制更精准
- **vs LoRA/DreamBooth 微调**：微调方法可以改变模型风格和创意倾向，但需要数据和训练时间；C3 完全免训练
- **vs Classifier-Free Guidance (CFG)**：CFG 通过推理时调整条件/无条件预测的权重来控制生成，但主要影响文本对齐度而非创意；C3 在特征空间操作，影响维度不同且互补

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统性地研究扩散模型的创意增强问题，特征放大 idea 简洁有效
- 实验充分度: ⭐⭐⭐ 跨多个 SD 变体验证，但缺少大规模人类评估和与更多基线的对比
- 写作质量: ⭐⭐⭐⭐ 创意双维度框架清晰，但方法部分技术细节可进一步充实
- 价值: ⭐⭐⭐⭐ 免训练即插即用的方案实用性极高，对创意应用场景有直接帮助

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Redefining <Creative> in Dictionary: Towards an Enhanced Semantic Understanding of Creative Generation](redefining_creative_in_dictionary_towards_an_enhanced_semantic_understanding_of_.md)
- [\[CVPR 2025\] Not All Parameters Matter: Masking Diffusion Models for Enhancing Generation Ability](not_all_parameters_matter_masking_diffusion_models_for_enhancing_generation_abil.md)
- [\[NeurIPS 2025\] Training-Free Constrained Generation with Stable Diffusion Models](../../NeurIPS2025/image_generation/training-free_constrained_generation_with_stable_diffusion_models.md)
- [\[CVPR 2025\] Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](enhancing_image_aesthetics_with_dual-conditioned_diffusion_models_guided_by_mult.md)
- [\[CVPR 2025\] Enhancing Facial Privacy Protection via Weakening Diffusion Purification](enhancing_facial_privacy_protection_via_weakening_diffusion_purification.md)

<!-- RELATED:END -->
