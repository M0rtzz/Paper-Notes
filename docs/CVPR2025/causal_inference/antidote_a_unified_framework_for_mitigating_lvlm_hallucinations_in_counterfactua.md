---
title: >-
  [论文解读] Antidote: A Unified Framework for Mitigating LVLM Hallucinations in Counterfactual Presupposition
description: >-
  [CVPR 2025][LVLM幻觉][反事实推理] 提出Antidote统一框架，解决大型视觉语言模型在反事实预设场景下的幻觉问题
tags:
  - CVPR 2025
  - LVLM幻觉
  - 反事实推理
  - 视觉语言模型
  - 幻觉缓解
---

# Antidote: A Unified Framework for Mitigating LVLM Hallucinations in Counterfactual Presupposition

**会议**: CVPR 2025  
**arXiv**: 待公开  
**代码**: 待确认  
**领域**: LVLM 幻觉缓解  
**关键词**: LVLM幻觉, 反事实预设, 统一框架, 视觉忠实性

## 一句话总结
提出 Antidote 统一框架，专门解决大型视觉语言模型在面对含反事实预设的问题（如"图中的大象是什么颜色的？"而图中没有大象）时的幻觉生成问题。

## 研究背景与动机
**领域现状**：LVLM 在跨模态任务上取得了令人印象深刻的结果，但幻觉问题仍是核心挑战。

**现有痛点**：已有幻觉缓解工作主要关注物体感知幻觉（将不存在的物体描述为存在），但忽视了反事实预设场景 — 当用户的问题包含错误前提时，模型应拒绝而非配合编造。

**本文目标** 让 LVLM 在面对反事实预设的提问时能识别错误前提并拒绝幻觉回答。

**核心 idea**：通过统一框架检测问题中的反事实预设，引导模型拒绝基于错误前提的回答。

## 方法详解

### 关键设计
1. **反事实预设检测**：判断用户问题是否包含与图像内容矛盾的前提假设。
2. **忠实性引导解码**：在生成时强化视觉证据的权重，抑制语言先验导致的配合倾向。
3. **拒绝回答机制**：当检测到反事实预设时，生成解释性的拒绝回答而非编造信息。

## 实验关键数据

### 关键发现
- 在反事实预设场景下显著降低了幻觉率
- 对正常问题的回答质量不受影响
- 跨多个 LVLM（LLaVA、InternVL 等）均有效

## 亮点与洞察
- 关注了被忽视的反事实预设幻觉场景，是一种更深层的幻觉类型
- 拒绝回答的能力对 AI 安全性至关重要

## 局限与展望
- 反事实预设的检测精度直接影响框架的可用性
- 过度保守可能导致正常问题也被误判为反事实

<!-- RELATED:START -->

## 相关论文

- [Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding](seeing_far_and_clearly_mitigating_hallucinations_in_mllms_with_attention_causal_.md)
- [Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression](../../CVPR2026/causal_inference/cipher_counterfactual_diffusion_hallucination_sup.md)
- [FitCF: A Framework for Automatic Feature Importance-guided Counterfactual Example Generation](../../ACL2025/causal_inference/fitcf_a_framework_for_automatic_feature_importance-guided_counterfactual_example.md)
- [Towards Fine-Grained Interpretability: Counterfactual Explanations for Misclassification](towards_fine-grained_interpretability_counterfactual_explanations_for_misclassif.md)
- [Image Quality Assessment: Investigating Causal Perceptual Effects with Abductive Counterfactual](image_quality_assessment_investigating_causal_perceptual_effects_with_abductive_.md)

<!-- RELATED:END -->
