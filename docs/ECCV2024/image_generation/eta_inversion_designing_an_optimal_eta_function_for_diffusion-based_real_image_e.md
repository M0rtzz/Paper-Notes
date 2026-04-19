---
title: >-
  [论文解读] Eta Inversion: Designing an Optimal Eta Function for Diffusion-based Real Image Editing
description: >-
  [ECCV 2024][图像生成][扩散模型反演] 通过理论分析DDIM采样方程中η参数的作用，设计时间和区域依赖的η函数，实现更灵活精确的真实图像编辑。
tags:
  - ECCV 2024
  - 图像生成
  - 扩散模型反演
  - 图像编辑
  - DDIM采样
  - eta函数
  - 真实图像编辑
---

# Eta Inversion: Designing an Optimal Eta Function for Diffusion-based Real Image Editing

**会议**: ECCV 2024  
**arXiv**: [2403.09468](https://arxiv.org/abs/2403.09468)  
**代码**: 有  
**领域**: Image Generation  
**关键词**: 扩散模型反演, 图像编辑, DDIM采样, eta函数, 真实图像编辑

## 一句话总结

通过理论分析DDIM采样方程中η参数的作用，设计时间和区域依赖的η函数，实现更灵活精确的真实图像编辑。

## 研究背景与动机

基于扩散模型的图像编辑已成为图像生成领域的重要研究方向。典型的编辑流程是：首先对真实图像进行反演（inversion），将其映射到噪声空间，然后在去噪过程中注入新的文本引导以实现编辑。然而，现有方法在编辑质量上面临一个核心矛盾——**编辑的忠实度（faithfulness）与源图像的保真度（fidelity）之间的权衡**。

具体而言：(1) DDIM inversion虽然理论上是确定性的，但在实际应用中由于离散化误差会导致重建不完美；(2) Null-text inversion等方法通过优化null-text embedding来改善重建，但编辑灵活性受限；(3) 现有方法大多忽略了DDIM采样方程中η参数的作用，将其简单地设为0（确定性采样）或1（完全随机采样）。

本文的核心洞察是：η参数实际上是控制编辑程度的关键杠杆。通过对η在DDIM采样中的数学作用进行深入理论分析，作者发现η可以在时间维度和空间（区域）维度上灵活调节，从而实现对编辑范围和程度的精细控制。这一发现打开了一个全新的设计空间。

## 方法详解

### 整体框架

Eta Inversion的整体流程包括三个步骤：(1) 对输入图像进行标准DDIM inversion获取噪声表示；(2) 设计一个时间和区域依赖的η(t, r)函数；(3) 在去噪采样过程中使用设计好的η函数进行编辑引导的采样。

### 关键设计

1. **η参数的理论分析**:
    - 功能：揭示η在DDIM采样中对编辑能力的影响机制
    - 核心思路：从DDIM采样方程出发，推导η对去噪过程中信号-噪声比的影响。理论上证明：较大的η值引入更多随机性，使得采样轨迹偏离inversion轨迹更多，从而允许更大程度的编辑；较小的η值则保持更好的源图像保真度
    - 设计动机：现有方法要么完全忽略η，要么将其视为超参数简单调节，缺乏对其作用的系统理解

2. **时间依赖的η函数**:
    - 功能：在不同时间步使用不同的η值，精细控制编辑程度
    - 核心思路：设计η(t)为时间的函数，在早期去噪步骤（高噪声水平）使用较大的η以允许结构性编辑，在后期步骤（低噪声水平）使用较小的η以保持细节保真度。这符合扩散模型中"粗到细"的生成特性
    - 设计动机：不同时间步对图像不同层级的特征有不同的影响，统一的η值无法适配这种层级结构

3. **区域依赖的η函数**:
    - 功能：在图像的不同空间区域使用不同的η值
    - 核心思路：通过cross-attention map或用户指定的mask确定编辑区域，在需要编辑的区域使用较大的η值，在需要保持不变的区域使用较小的η值。这实现了"编辑该编辑的，保留该保留的"
    - 设计动机：真实图像编辑通常只需要修改图像的局部区域，全局统一的η值会导致非编辑区域的不必要变化

### 损失函数 / 训练策略

Eta Inversion是一种无需训练的（training-free）推理时方法。核心在于η函数的设计而非模型参数的优化。η(t, r)函数通过以下策略确定：
- 时间维度：采用分段函数或平滑衰减函数，从高时间步的较大η逐渐过渡到低时间步的较小η
- 区域维度：利用cross-attention map的softmax分布作为区域权重，自动确定编辑区域

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| PIE-Bench | Structure Dist ↓ | 最优 | Null-text Inv | 显著降低 |
| PIE-Bench | CLIP Sim ↑ | 最优 | P2P+NTI | +2-5% |
| PIE-Bench | LPIPS ↓ | 最优 | DirectInverse | 显著降低 |
| 用户研究 | 偏好率 | >60% | 所有基线 | 明显领先 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| η=0（标准DDIM） | 编辑能力弱 | 过度保守，难以实现有意义的编辑 |
| η=1（完全随机） | 保真度低 | 引入过多随机性，破坏源图像 |
| 仅时间依赖η | 中等性能 | 比固定η好，但非编辑区域仍有变化 |
| 时间+区域依赖η | 最优 | 编辑精确且保真度高 |

### 关键发现

- η参数是DDIM采样中长期被忽视的重要控制杠杆
- 时间依赖的η函数利用了扩散模型的层级生成特性
- 区域依赖的η函数实现了精确的局部编辑控制
- 方法无需额外训练或优化，推理时即插即用

## 亮点与洞察

- 理论分析深入且实用，将数学推导转化为具体的设计方案
- 方法简洁优雅——不修改模型结构，不需要训练，仅通过调节采样参数即可实现更好的编辑
- 对η的时间和空间两个维度的分析具有很好的互补性
- 为扩散模型的可控生成提供了新的视角

## 局限与展望

- η函数的具体形式仍需一定的手工设计或启发式选择
- 对于需要大幅度结构性编辑的场景（如姿态变化），方法的效果可能有限
- 区域依赖的η依赖cross-attention map的质量，对于复杂场景可能不够精确
- 可以探索学习最优η函数的可能性，进一步自动化

## 相关工作与启发

- **DDIM Inversion**: Song et al.提出的确定性采样方法，是本文的理论基础
- **Null-text Inversion**: 通过优化null-text来改善inversion质量，但计算开销大
- **Prompt-to-Prompt**: 通过操纵cross-attention实现编辑，与本文的区域依赖η互补
- 启发：扩散模型采样过程中还有许多未被充分利用的控制变量，值得更深入的理论分析

## 评分

- 新颖性: ⭐⭐⭐⭐ 从被忽视的η参数入手，理论分析扎实
- 实验充分度: ⭐⭐⭐⭐ 定量和定性评估全面，消融实验充分
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，实验展示直观
- 价值: ⭐⭐⭐⭐ 为扩散模型编辑提供了新的无训练方法，实用性强

<!-- RELATED:START -->

## 相关论文

- [ReNoise: Real Image Inversion Through Iterative Noising](renoise_real_image_inversion_through_iterative_noising.md)
- [Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)
- [Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis](rejection_sampling_imle_designing_priors_for_better_few-shot_image_synthesis.md)
- [Lazy Diffusion Transformer for Interactive Image Editing](lazy_diffusion_transformer_for_interactive_image_editing.md)
- [LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)

<!-- RELATED:END -->
