---
title: >-
  CVPR2025 因果推理方向 6篇论文解读
description: >-
  6篇CVPR2025 因果推理方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔗 因果推理

**📷 CVPR2025** · **6** 篇论文解读

**[Adventurer: Optimizing Vision Mamba Architecture Designs for Efficiency](adventurer_optimizing_vision_mamba_architecture_designs_for_efficiency.md)**

:   提出 Adventurer 系列视觉模型，通过"头部平均池化 token"和"层间翻转"两个简单设计将图像输入适配到单向因果扫描框架中，使 Mamba 架构在视觉任务上实现 4-6 倍于现有 Vision Mamba 的训练速度，同时保持与 ViT 相当甚至更优的精度。

**[Antidote: A Unified Framework for Mitigating LVLM Hallucinations in Counterfactual Presupposition](antidote_a_unified_framework_for_mitigating_lvlm_hallucinations_in_counterfactua.md)**

:   提出 Antidote 统一框架，专门解决大型视觉语言模型在面对含反事实预设的问题（如"图中的大象是什么颜色的？"而图中没有大象）时的幻觉生成问题。

**[Image Quality Assessment: Investigating Causal Perceptual Effects with Abductive Counterfactual](image_quality_assessment_investigating_causal_perceptual_effects_with_abductive_.md)**

:   提出基于溯因反事实推理的全参考图像质量评估方法（FR-IQA），通过建模"如果没有某种失真，感知质量会如何变化"的因果关系来更准确地评估图像质量。

**[Joint Scheduling of Causal Prompts and Tasks for Multi-Task Learning](joint_scheduling_of_causal_prompts_and_tasks_for_multi-task_learning.md)**

:   提出 JSCPT，通过联合调度因果提示（消除虚假相关）和任务学习顺序（利用动态任务关系），优化多任务提示学习的性能。

**[Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding](seeing_far_and_clearly_mitigating_hallucinations_in_mllms_with_attention_causal_.md)**

:   将 MLLM 幻觉分为初始幻觉（首次出现的错误描述）和雪球幻觉（基于已有错误的累积放大），提出注意力因果解码方法分别针对两种类型进行缓解。

**[Towards Fine-Grained Interpretability: Counterfactual Explanations for Misclassification](towards_fine-grained_interpretability_counterfactual_explanations_for_misclassif.md)**

:   针对模型错误分类的场景，提出细粒度的反事实解释方法 — 不仅展示"模型看了什么"，还能回答"如果哪些区域不同，模型就能正确分类"。
