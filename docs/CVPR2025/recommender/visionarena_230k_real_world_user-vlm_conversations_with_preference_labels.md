---
title: >-
  [论文解读] VisionArena: 230k Real World User-VLM Conversations with Preference Labels
description: >-
  [CVPR 2025][VLM评估][数据集] 发布VisionArena大规模真实用户-VLM对话数据集（230K条），包含偏好标签用于VLM基准评估
tags:
  - CVPR 2025
  - VLM评估
  - 偏好数据集
  - 众包标注
  - 真实对话
---

# VisionArena: 230k Real World User-VLM Conversations with Preference Labels

**会议**: CVPR 2025  
**arXiv**: [2412.08687](https://arxiv.org/abs/2412.08687)  
**代码**: 待确认  
**领域**: VLM 评估  
**关键词**: VLM基准, 偏好数据, 众包评估, 真实对话, 大规模数据集

## 一句话总结
发布 VisionArena，当前最大规模的众包真实用户-VLM 对话数据集（230K 条对话+偏好标签），为 VLM 评估提供反映真实使用场景的基准。

## 研究背景与动机
**领域现状**：VLM 快速发展，但现有基准多为人工设计的评测集，与真实用户的使用模式存在差距。

**现有痛点**：实验室设计的 VQA 数据集无法覆盖真实用户的多样化需求（创意生成、细节分析、多轮对话等），导致基准排名与实际体验不一致。

**本文目标** 构建反映真实世界 VLM 使用场景的大规模评估基准。

**核心 idea**：通过众包平台收集真实用户与 VLM 的自然对话和偏好判断，构建大规模真实基准。

## 方法详解

### 关键设计
1. **众包对话收集**：用户在平台上同时与两个匿名 VLM 对话并选择偏好。
2. **多维度偏好标注**：偏好标签包含整体质量、事实性、有用性等多维度。
3. **ELO 评分系统**：基于偏好标签计算各 VLM 的 ELO 排名。

## 实验关键数据

### 关键发现
- 230K 对话覆盖了大量真实世界视觉问题类型，远超现有评测集的多样性
- VLM 在真实场景中的排名与实验室基准有显著差异
- 用户偏好中"有用性"比"准确性"有更高权重

## 亮点与洞察
- 填补了 VLM 评估中真实用户视角的空白
- ELO 排名系统为 VLM 发展提供了持续更新的基准

## 局限与展望
- 众包用户群体可能存在人口统计偏差
- 英文对话为主，多语言覆盖不足

<!-- RELATED:START -->

## 相关论文

- [Wide-Horizon Thinking and Simulation-Based Evaluation for Real-World LLM Planning with Multifaceted Constraints](../../NeurIPS2025/recommender/wide-horizon_thinking_and_simulation-based_evaluation_for_real-world_llm_plannin.md)
- [FineVQ: Fine-Grained User Generated Content Video Quality Assessment](finevq_fine-grained_user_generated_content_video_quality_assessment.md)
- [Beyond Single Labels: Improving Conversational Recommendation through LLM-Powered Data Augmentation](../../ACL2025/recommender/beyond_single_labels_improving_conversational_recommendation_through_llm-powered.md)
- [Aligning LLMs by Predicting Preferences from User Writing Samples](../../ICML2025/recommender/aligning_llms_by_predicting_preferences_from_user_writing_samples.md)
- [PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model](../../ICML2025/recommender/parm_multi-objective_test-time_alignment_via_preference-aware_autoregressive_rew.md)

<!-- RELATED:END -->
