---
title: >-
  [论文解读] ProAPO: Progressively Automatic Prompt Optimization for Visual Classification
description: >-
  [CVPR 2025][提示优化][视觉分类] 提出渐进式自动提示优化方法，利用LLM生成的视觉描述逐步优化VLM的分类提示
tags:
  - CVPR 2025
  - 提示优化
  - CLIP
  - LLM
  - 视觉分类
  - 自动化提示
---

# ProAPO: Progressively Automatic Prompt Optimization for Visual Classification

**会议**: CVPR 2025  
**arXiv**: [2502.19844](https://arxiv.org/abs/2502.19844)  
**代码**: 待确认  
**领域**: 提示优化  
**关键词**: 自动提示优化, VLM, LLM, 视觉分类, 渐进式优化

## 一句话总结
提出 ProAPO 渐进式自动提示优化方法，通过 LLM 生成视觉描述并迭代优化，自动为 VLM 发现最优分类提示，无需人工设计或梯度训练。

## 研究背景与动机
**领域现状**：VLM（如 CLIP）的分类性能高度依赖文本提示质量，人工设计提示耗时且不可扩展。

**现有痛点**：手工提示无法适应所有数据集；软提示需要标注数据训练；LLM 生成的描述可能含噪声。

**本文目标** 自动化地为不同分类任务找到最优提示。

**核心 idea**：用 LLM 生成候选描述，然后渐进式评估和筛选，自动优化出最佳分类提示组合。

## 方法详解

### 关键设计
1. **LLM 候选生成**：利用 LLM 为每个类别生成多组视觉属性描述作为候选提示。
2. **渐进式评估**：从粗到细地评估候选提示的分类效果，逐步淘汰低效提示。
3. **组合优化**：搜索最佳提示组合而非单个提示，利用提示间的互补性。

## 实验关键数据

### 关键发现
- 在 11 个分类基准上平均超越手工提示和 CoOp 等训练方法
- 完全无需训练数据或梯度计算
- 优化后的提示具有良好的可解释性

## 亮点与洞察
- 将提示工程从人工劳动转变为自动化流水线
- 渐进式策略有效控制了搜索空间

## 局限与展望
- LLM 生成的候选质量决定了优化上界
- 评估阶段需要验证集，零标注场景下需要替代评估指标

<!-- RELATED:START -->

## 相关论文

- [Enhancing Few-Shot Class-Incremental Learning via Training-Free Bi-Level Modality Calibration](enhancing_few-shot_class-incremental_learning_via_training-free_bi-level_modalit.md)
- [ASCIIBench: Evaluating Language-Model-Based Understanding of Visually-Oriented Text](../../NeurIPS2025/aigc_detection/asciibench_evaluating_language-model-based_understanding_of_visually-oriented_te.md)
- ["Jutters"](../../NeurIPS2025/aigc_detection/jutters.md)
- [People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text](../../ACL2025/aigc_detection/chatgpt_user_ai_text_detection.md)
- [Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code](../../NeurIPS2025/aigc_detection/classical_planning_with_llm-generated_heuristics_challenging_the_state_of_the_ar.md)

<!-- RELATED:END -->
