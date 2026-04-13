---
title: >-
  CVPR2025 LLM效率方向 4篇论文解读
description: >-
  4篇CVPR2025 LLM效率方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM效率

**📷 CVPR2025** · 共 **4** 篇

**[Associative Transformer](associative_transformer.md)**

:   提出 Associative Transformer (AiT)，通过在 Transformer 中引入可学习的显式记忆模块和 Hopfield 网络进行 token 重建，以更少的参数实现优于 ViT 的分类和关系推理性能。

**[Efficient Data Driven Mixture-Of-Expert Extraction From Trained Networks](efficient_data_driven_mixture-of-expert_extraction_from_trained_networks.md)**

:   提出一种从预训练 ViT 中自动提取 MoE（Mixture-of-Experts）变体的方法：先聚类 MLP 层的输出激活模式，再据此抽取对应的子网络作为专家，无需从头训练 MoE，在 ImageNet-1k 上仅需少量微调即可恢复 98% 原始性能，同时将 FLOPs 和模型大小分别减少 36% 和 32%。

**[Kac Kolmogorov-Arnold Classifier For Continual Learning](kac_kolmogorov-arnold_classifier_for_continual_learning.md)**

:   首次将 Kolmogorov-Arnold Network (KAN) 应用于持续学习，通过将 B-spline 替换为径向基函数 (RBF) 构建分类器 KAC，仅增加 0.23M 参数即可在多种持续学习方法上获得一致且显著的性能提升（CUB200 40-step 最高 +20.70%）。

**[Language Guided Concept Bottleneck Models For Interpretable Continual Learning](language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)**

:   将语言引导的概念瓶颈模型 (CBM) 整合到持续学习中，通过 ChatGPT 生成类别概念、CLIP 编码的概念对齐模块和语义引导的原型增强策略，在 ImageNet-subset 上实现最终准确率 +3.06% 的提升，同时提供透明可解释的决策过程。
