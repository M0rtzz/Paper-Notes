---
title: >-
  [论文解读] Image Quality Assessment: Investigating Causal Perceptual Effects with Abductive Counterfactual
description: >-
  [CVPR 2025][FR-IQA] 提出基于溯因反事实推理的全参考图像质量评估方法（FR-IQA），通过建模"如果没有某种失真，感知质量会如何变化"的因果关系来更准确地评估图像质量。
tags:
  - CVPR 2025
  - FR-IQA
  - 因果感知
  - 溯因推理
  - 反事实
  - 失真建模
---

# Image Quality Assessment: Investigating Causal Perceptual Effects with Abductive Counterfactual

**会议**: CVPR 2025  
**arXiv**: [2412.16939](https://arxiv.org/abs/2412.16939)  
**代码**: 待确认  
**领域**: 图像质量评估  
**关键词**: FR-IQA, 因果感知, 溯因推理, 反事实, 失真建模

## 一句话总结
提出基于溯因反事实推理的全参考图像质量评估方法（FR-IQA），通过建模"如果没有某种失真，感知质量会如何变化"的因果关系来更准确地评估图像质量。

## 研究背景与动机
**领域现状**：FR-IQA通过比较失真图像与参考图像来评估质量，SSIM、LPIPS等方法广泛使用。

**现有痛点**：传统方法基于统计相关性而非因果建模，无法捕获人类感知对不同失真的因果响应机制。

**本文目标** 建模失真类型与感知质量之间的因果关系，而非仅统计相关。

**核心 idea**：通过溯因反事实推理分离每种失真对感知质量的独立因果贡献。

## 方法详解

### 关键设计
1. **失真因果图构建**：建模不同失真类型之间以及失真-感知之间的因果关系结构。
2. **溯因反事实推理**：推断"移除某种失真后"的感知质量变化，分离各失真的独立影响。
3. **因果效应聚合**：将各失真的因果效应加权融合为最终的质量评分。

## 实验关键数据

### 关键发现
- 在 LIVE、TID2013、CSIQ 等基准上 SRCC/PLCC 优于 SSIM 和 LPIPS
- 因果分析提供了"哪种失真对感知影响最大"的可解释输出
- 对多重失真叠加场景的评估更准确

## 亮点与洞察
- 将因果推理引入 IQA 是一个有启发性的跨领域融合
- 可解释的失真归因对图像修复优化有实际指导价值

## 局限与展望
- 因果图的结构假设可能不完全准确
- 计算开销比简单统计指标更大

<!-- RELATED:START -->

## 相关论文

- [Counterfactual Explanations on Robust Perceptual Geodesics](../../ICLR2026/causal_inference/counterfactual_explanations_on_robust_perceptual_geodesics.md)
- [Transferring Causal Effects using Proxies](../../NeurIPS2025/causal_inference/transferring_causal_effects_using_proxies.md)
- [Isolated Causal Effects of Natural Language](../../ICML2025/causal_inference/isolated_causal_effects_of_natural_language.md)
- [Conformal Prediction for Causal Effects of Continuous Treatments](../../NeurIPS2025/causal_inference/conformal_prediction_for_causal_effects_of_continuous_treatments.md)
- [Estimating Causal Effects in Gaussian Linear SCMs with Finite Data](../../ICML2025/causal_inference/estimating_causal_effects_in_gaussian_linear_scms_with_finite_data.md)

<!-- RELATED:END -->
