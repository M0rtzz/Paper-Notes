---
title: >-
  [论文解读] Towards Fine-Grained Interpretability: Counterfactual Explanations for Misclassification
description: >-
  [CVPR 2025][可解释性][反事实解释] 针对模型错误分类场景，提出细粒度的反事实解释方法
tags:
  - CVPR 2025
  - 可解释性
  - 反事实解释
  - 错误分类
  - 归因分析
---

# Towards Fine-Grained Interpretability: Counterfactual Explanations for Misclassification

**会议**: CVPR 2025  
**arXiv**: 待公开  
**代码**: 待确认  
**领域**: 可解释性  
**关键词**: 反事实解释, 错误分类, 细粒度归因, 模型调试

## 一句话总结
针对模型错误分类的场景，提出细粒度的反事实解释方法 — 不仅展示"模型看了什么"，还能回答"如果哪些区域不同，模型就能正确分类"。

## 研究背景与动机
**领域现状**：归因方法（如 GradCAM、SHAP）可以可视化模型的决策依据，但在错误分类场景下缺乏足够的解释力。

**现有痛点**：传统归因方法仅展示模型关注的区域，但在错误分类场景下，用户更需要知道"哪些特征导致了错误"和"怎样改变输入才能得到正确结果"。

**本文目标** 为错误分类提供可操作的细粒度反事实解释。

**核心 idea**：找到图像中最小的改变区域，使模型从错误分类翻转为正确分类，并可视化这些关键区域。

## 方法详解

### 关键设计
1. **反事实搜索**：在特征空间中搜索使分类结果翻转的最小扰动。
2. **空间化归因**：将反事实扰动映射回输入空间，定位导致错误的关键区域。
3. **语义对比**：对比正确和错误分类的特征差异，提供语义层面的解释。

## 实验关键数据

### 关键发现
- 反事实解释比 GradCAM 在错误诊断上提供了更有用的信息
- 找到的关键区域与人类直觉高度一致
- 对模型调试和改进有实际指导价值

## 亮点与洞察
- 从"模型看了什么"到"改变什么能修正错误"的解释粒度提升
- 对模型调试和数据增强策略设计有直接应用价值

## 局限与展望
- 反事实搜索的计算开销可能较大
- 生成的反事实可能在某些情况下不自然

<!-- RELATED:START -->

## 相关论文

- [Counterfactual Explanations on Robust Perceptual Geodesics](../../ICLR2026/causal_inference/counterfactual_explanations_on_robust_perceptual_geodesics.md)
- [Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations](../../NeurIPS2025/causal_inference/few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)
- [Performative Validity of Recourse Explanations](../../NeurIPS2025/causal_inference/performative_validity_of_recourse_explanations.md)
- [Synthesising Counterfactual Explanations via Label-Conditional Gaussian Mixture Variational Autoencoders](../../ICLR2026/causal_inference/synthesising_counterfactual_explanations_via_label-conditional_gaussian_mixture_.md)
- [LLM Interpretability with Identifiable Temporal-Instantaneous Representation](../../NeurIPS2025/causal_inference/llm_interpretability_with_identifiable_temporal-instantaneous_representation.md)

<!-- RELATED:END -->
