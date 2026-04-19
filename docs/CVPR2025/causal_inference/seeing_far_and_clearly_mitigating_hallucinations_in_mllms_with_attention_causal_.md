---
title: >-
  [论文解读] Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding
description: >-
  [CVPR 2025][MLLM幻觉][因果解码] 提出注意力因果解码方法，区分并缓解初始幻觉和雪球幻觉两种类型
tags:
  - CVPR 2025
  - MLLM幻觉
  - 因果解码
  - 注意力机制
  - 初始幻觉
  - 雪球幻觉
---

# Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding

**会议**: CVPR 2025  
**arXiv**: [2505.16652](https://arxiv.org/abs/2505.16652)  
**代码**: 待确认  
**领域**: MLLM 幻觉缓解  
**关键词**: 注意力因果解码, 初始幻觉, 雪球幻觉, MLLM, 视觉忠实性

## 一句话总结
将 MLLM 幻觉分为初始幻觉（首次出现的错误描述）和雪球幻觉（基于已有错误的累积放大），提出注意力因果解码方法分别针对两种类型进行缓解。

## 研究背景与动机
**领域现状**：MLLM（如 LLaVA、GPT-4V）在视觉问答中常产生幻觉 — 描述图像中不存在的内容。

**现有痛点**：现有方法笼统地处理所有幻觉，未区分幻觉产生的机制差异。实际上初始幻觉（视觉信息不足）和雪球幻觉（错误在解码中传播放大）有不同的成因和解决方案。

**本文目标** 针对不同类型幻觉的差异化缓解策略。

**核心 idea**：通过分析注意力权重中的视觉-语言因果关系，对初始幻觉增强视觉注意力，对雪球幻觉截断错误传播链。

## 方法详解

### 关键设计
1. **幻觉类型分类**：通过注意力模式识别当前生成 token 是产生初始幻觉还是雪球幻觉。
2. **视觉注意力增强**：对初始幻觉倾向的位置增强视觉 token 的注意力权重。
3. **因果链截断**：对雪球幻觉倾向的位置降低对前文可能错误 token 的注意力权重。

## 实验关键数据

### 关键发现
- 在 POPE、CHAIR 等幻觉评估基准上显著降低幻觉率
- 对两种幻觉类型的分别处理优于统一处理
- 不需要额外训练，可即插即用地应用于现有 MLLM

## 亮点与洞察
- 幻觉分类学（初始 vs 雪球）的提出本身就有分析价值
- 即插即用的解码策略部署成本极低

## 局限与展望
- 注意力模式分析增加了解码延迟
- 幻觉类型的判断可能不完全准确
- 可以与训练阶段的幻觉缓解方法结合

<!-- RELATED:START -->

## 相关论文

- [Antidote: A Unified Framework for Mitigating LVLM Hallucinations in Counterfactual Presupposition](antidote_a_unified_framework_for_mitigating_lvlm_hallucinations_in_counterfactua.md)
- [Action-Guided Attention for Video Action Anticipation](../../ICLR2026/causal_inference/action-guided_attention_for_video_action_anticipation.md)
- [Joint Scheduling of Causal Prompts and Tasks for Multi-Task Learning](joint_scheduling_of_causal_prompts_and_tasks_for_multi-task_learning.md)
- [Image Quality Assessment: Investigating Causal Perceptual Effects with Abductive Counterfactual](image_quality_assessment_investigating_causal_perceptual_effects_with_abductive_.md)
- [Flattery, Fluff, and Fog: Diagnosing and Mitigating Idiosyncratic Biases in Preference Models](../../ICLR2026/causal_inference/flattery_fluff_and_fog_diagnosing_and_mitigating_idiosyncratic_biases_in_prefere.md)

<!-- RELATED:END -->
