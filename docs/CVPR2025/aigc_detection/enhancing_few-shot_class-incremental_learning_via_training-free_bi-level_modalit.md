---
title: >-
  [论文解读] Enhancing Few-Shot Class-Incremental Learning via Training-Free Bi-Level Modality Calibration
description: >-
  [CVPR 2025][小样本增量学习] 提出无需训练的双层模态校准框架，利用CLIP等预训练VLM实现小样本类增量学习
tags:
  - CVPR 2025
  - 小样本增量学习
  - 模态校准
  - CLIP
  - 无训练框架
---

# Enhancing Few-Shot Class-Incremental Learning via Training-Free Bi-Level Modality Calibration

**会议**: CVPR 2025  
**arXiv**: 待公开  
**代码**: 待确认  
**领域**: 小样本增量学习  
**关键词**: FSCIL, 模态校准, CLIP, 无训练, 视觉语言模型

## 一句话总结
提出无需额外训练的双层模态校准框架，利用 CLIP 等预训练视觉语言模型的跨模态对齐能力实现小样本类增量学习，在避免灾难性遗忘的同时学习新类。

## 研究背景与动机
**领域现状**：FSCIL 要求模型用极少样本学习新类且不遗忘旧类，现有方法多依赖视觉模型并需额外训练。

**现有痛点**：训练阶段的更新不可避免地干扰已学知识；现有方法未充分利用 VLM 的跨模态泛化能力。

**本文目标** 利用 VLM 的预训练知识，在不训练的情况下实现 FSCIL。

**核心 idea**：通过冻结 CLIP 模型 + 双层校准（特征级和预测级），无需任何参数更新即可实现增量学习。

## 方法详解

### 关键设计
1. **特征级校准**：将视觉特征和文本特征在共享空间中对齐校准，利用文本原型作为类别表示。
2. **预测级校准**：在分类决策时平衡新旧类别的预测置信度差异。
3. **无训练原型更新**：新类到来时仅需计算少量样本的平均特征作为新原型。

## 实验关键数据

### 关键发现
- 无训练方法性能接近甚至超越需要训练的 SOTA FSCIL 方法
- 完全避免了灾难性遗忘，因为没有参数更新
- 在 miniImageNet、CUB-200、CIFAR-100 上均有竞争力的表现

## 亮点与洞察
- "无训练"范式极大降低了增量学习的部署复杂度
- VLM 的文本原型为类别表示提供了天然的稳定性

## 局限与展望
- 依赖 CLIP 预训练质量，对预训练数据未覆盖的领域效果有限
- 文本原型的描述质量影响分类精度

<!-- RELATED:START -->

## 相关论文

- [Learning to Rewrite: Generalized LLM-Generated Text Detection](../../ACL2025/aigc_detection/learning_to_rewrite_generalized_llm-generated_text_detection.md)
- [Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training](../../ACL2025/aigc_detection/greater_adversarial_mgt_detection.md)
- [Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction](../../ACL2025/aigc_detection/reliably_bounding_false_positives_a_zero-shot_machine-generated_text_detection_f.md)
- [Fine-grained Image Aesthetic Assessment: Learning Discriminative Scores from Relative Ranks](../../CVPR2026/aigc_detection/fine-grained_image_aesthetic_assessment_learning_discriminative_scores_from_rela.md)
- [ProAPO: Progressively Automatic Prompt Optimization for Visual Classification](proapo_progressively_automatic_prompt_optimization_for_visual_classification.md)

<!-- RELATED:END -->
