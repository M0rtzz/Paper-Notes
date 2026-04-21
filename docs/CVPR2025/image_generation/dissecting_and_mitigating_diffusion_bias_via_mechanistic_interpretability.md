---
title: >-
  [论文解读] Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability
description: >-
  [CVPR 2025][图像生成][扩散模型偏见] 提出 DiffLens，用 k-稀疏自编码器将 U-Net 瓶颈层的多义性神经元分解为单义性特征，通过归因分析识别偏见特征并进行干预，实现扩散模型性别偏见的精确缓解（FD 从 0.564 降至 0.046）同时保持甚至提升生成质量。
tags:
  - CVPR 2025
  - 图像生成
  - 扩散模型偏见
  - 机制可解释性
  - 稀疏自编码器
  - 特征干预
  - 公平性
---

# Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability

**会议**: CVPR 2025  
**arXiv**: [2503.20483](https://arxiv.org/abs/2503.20483)  
**代码**: https://foundation-model-research.github.io/difflens (项目页)  
**领域**: 扩散模型 / AI安全  
**关键词**: 扩散模型偏见, 机制可解释性, 稀疏自编码器, 特征干预, 公平性

## 一句话总结
提出 DiffLens，用 k-稀疏自编码器将 U-Net 瓶颈层的多义性神经元分解为单义性特征，通过归因分析识别偏见特征并进行干预，实现扩散模型性别偏见的精确缓解（FD 从 0.564 降至 0.046）同时保持甚至提升生成质量。

## 研究背景与动机

**领域现状**：扩散模型存在系统性偏见——对"医生"、"CEO"等提示倾向生成特定性别/种族的图像。现有去偏见方法主要在数据层面（平衡训练集）或提示层面（修改文本），缺乏对模型内部偏见来源的理解和精确干预。

**现有痛点**：数据层面方法需要重新训练模型，成本高；提示修改方法是启发式的，无法处理隐式偏见；无一方法能精确定位"哪些神经元/特征导致了偏见"。

**核心矛盾**：U-Net 中的神经元是多义性的（一个神经元同时参与多种语义），难以直接分离偏见成分而不影响正常生成。

**本文目标** 精确定位和干预扩散模型内部的偏见特征，在缓解偏见的同时不损害（甚至提升）生成质量。

**切入角度**：用稀疏自编码器（SAE）将多义性激活分解为单义性特征，然后用积分梯度归因精确识别哪些特征编码了偏见属性（如性别），最后通过缩放/添加这些特征的激活来控制偏见。

**核心 idea**：k-稀疏自编码器分解 U-Net 瓶颈层多义性激活为单义性特征 + 积分梯度识别偏见特征 + 特征干预缓解偏见。

## 方法详解

### 关键设计

1. **k-稀疏自编码器 (k-SAE)**：在 U-Net 瓶颈层训练 SAE 将 d 维激活分解为 N>>d 的稀疏单义性特征（每步仅 k 个非零）。每个特征对应一个可解释的语义概念

2. **偏见特征识别**：用积分梯度计算每个 SAE 特征对属性分类器输出的贡献，高贡献特征被标记为偏见特征。设阈值 τ 筛选 top 贡献特征

3. **特征干预**：缩放（减弱）偏见特征的激活值来缓解偏见，或添加（增强）来控制特定属性。干预精确作用于识别出的偏见特征，不影响其他特征

## 实验关键数据

| 设置 | 方法 | 性别 FD↓ | FID↓ | CLIP-I↑ |
|------|------|---------|------|---------|
| 无条件(P2) | 原始 | 0.226 | 33.38 | — |
| | **DiffLens** | **0.002** | **31.93** | **0.9479** |
| 条件(SD) | 原始 | 0.564 | 120.06 | — |
| | **DiffLens** | **0.046** | **112.83** | — |

性别偏见几乎完全消除，FID 同时降低（质量提升），CLIP-I 0.9479 说明干预后图像内容高度保持。

### 关键发现
- 去偏见后 FID 反而下降（31.93 vs 33.38），说明偏见是生成质量的一种"负担"
- 性别去偏非常有效（FD 0.002），但种族去偏较难（FD 0.401），可能因为种族在模型中的编码更分散

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将机制可解释性引入扩散模型去偏见是全新视角
- 实验充分度: ⭐⭐⭐⭐ 无条件+条件模型，性别/年龄/种族多属性
- 写作质量: ⭐⭐⭐⭐⭐ 方法论清晰，三阶段逐步推进
- 价值: ⭐⭐⭐⭐⭐ 对 AI 公平性有重要意义，精确干预比数据/提示方法更优雅

<!-- RELATED:START -->

## 相关论文

- [Bias for Action: Video Implicit Neural Representations with Bias Modulation](bias_for_action_video_implicit_neural_representations_with_bias_modulation.md)
- [Towards a Mechanistic Explanation of Diffusion Model Generalization](../../ICML2025/image_generation/towards_a_mechanistic_explanation_of_diffusion_model_generalization.md)
- [Implicit Bias Injection Attacks against Text-to-Image Diffusion Models](implicit_bias_injection_attacks_against_text-to-image_diffusion_models.md)
- [DMQ: Dissecting Outliers of Diffusion Models for Post-Training Quantization](../../ICCV2025/image_generation/dmq_dissecting_outliers_of_diffusion_models_for_post-training_quantization.md)
- [How Bias Binds: Measuring Hidden Associations for Bias Control in Text-to-Image Compositions](../../AAAI2026/image_generation/how_bias_binds_measuring_hidden_associations_for_bias_control_in_text-to-image_c.md)

<!-- RELATED:END -->
