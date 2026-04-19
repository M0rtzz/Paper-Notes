---
title: >-
  CVPR2025 LLM效率方向 11篇论文解读
description: >-
  11篇CVPR2025 LLM效率方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM效率

**📷 CVPR2025** · **11** 篇论文解读

**[Associative Transformer](associative_transformer.md)**

:   提出 Associative Transformer (AiT)，通过在 Transformer 中引入可学习的显式记忆模块和 Hopfield 网络进行 token 重建，以更少的参数实现优于 ViT 的分类和关系推理性能。

**[CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction](care_transformer_mobile-friendly_linear_visual_transformer_via_decoupled_dual_in.md)**

:   本文提出CARE（deCoupled duAl-interactive lineaR attEntion）机制，通过非对称特征解耦策略将局部归纳偏置和长程依赖的学习过程分而治之，配合动态记忆单元和双交互模块充分利用跨特征互补性，在ImageNet-1K上以0.7/1.9 GMACs达到78.4/82.1% top-1精度，在移动端实现极低延迟。

**[Efficient Data Driven Mixture-of-Expert Extraction from Trained Networks](efficient_data_driven_mixture-of-expert_extraction_from_trained_networks.md)**

:   提出一种从预训练 ViT 中自动提取 MoE（Mixture-of-Experts）变体的方法：先聚类 MLP 层的输出激活模式，再据此抽取对应的子网络作为专家，无需从头训练 MoE，在 ImageNet-1k 上仅需少量微调即可恢复 98% 原始性能，同时将 FLOPs 和模型大小分别减少 36% 和 32%。

**[Improving Accuracy and Calibration via Differentiated Deep Mutual Learning](improving_accuracy_and_calibration_via_differentiated_deep_mutual_learning.md)**

:   提出 Diff-DML（Differentiated Deep Mutual Learning），通过差异化训练策略（DTS）和多样性保持学习目标（DPLO）两个核心设计，在保持集成模型预测多样性的同时，同时提升准确率和不确定性校准质量。

**[KAC: Kolmogorov-Arnold Classifier for Continual Learning](kac_kolmogorov-arnold_classifier_for_continual_learning.md)**

:   首次将 Kolmogorov-Arnold Network (KAN) 应用于持续学习，通过将 B-spline 替换为径向基函数 (RBF) 构建分类器 KAC，仅增加 0.23M 参数即可在多种持续学习方法上获得一致且显著的性能提升（CUB200 40-step 最高 +20.70%）。

**[Language Guided Concept Bottleneck Models for Interpretable Continual Learning](language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)**

:   提出语言引导的概念瓶颈模型（Language Guided Concept Bottleneck Model），将概念瓶颈网络的可解释性与持续学习结合，通过语言模型引导的概念定义实现任务间的知识迁移和可解释的增量学习。

**[LOCORE: Image Re-ranking with Long-Context Sequence Modeling](locore_image_re-ranking_with_long-context_sequence_modeling.md)**

:   提出 LoCoRe（Long-Context Re-ranker），首次实现基于局部描述子的列表级（list-wise）图像重排序，利用 Longformer 长上下文序列模型同时处理查询图像和整个候选列表的局部描述子，通过捕获候选图像间的传递关系显著提升重排序性能。

**[Low-Rank Adaptation in Multilinear Operator Networks for Security-Preserving Incremental Learning](low-rank_adaptation_in_multilinear_operator_networks_for_security-preserving_inc.md)**

:   针对全同态加密（Leveled FHE）场景下多线性算子网络的灾难性遗忘问题，提出了一种结合低秩适应（LoRA）和梯度投影记忆（GPM）机制的增量学习方法，在保障数据安全的前提下实现持续学习。

**[Seeing What Matters: Empowering CLIP with Patch Generation-to-Selection](seeing_what_matters_empowering_clip_with_patch_generation-to-selection.md)**

:   提出 CLIP-PGS（Patch Generation-to-Selection），一种简洁有效的掩码策略，通过渐进式的"生成-选择"过程——先预选候选掩码patch、再用 Sobel 边缘检测保护关键语义区域、最后用最优传输归一化精细化选择——在提升 CLIP 训练效率（降至 0.5-0.6× 训练时间）的同时在零样本分类、检索等任务上取得 SOTA。

**[Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training](spatial-ttt_streaming_visual-based_spatial_intelligence_with_test-time_training.md)**

:   本文提出 Spatial-TTT，通过测试时训练（TTT）机制将模型的部分参数（快速权重）作为紧凑非线性记忆，配合混合架构和空间预测机制，从无界视频流中持续积累和组织3D空间证据，在视频空间理解基准上达到 SOTA。

**[Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention for Transformer](spiking_transformer_introducing_accurate_addition-only_spiking_self-attention_fo.md)**

:   本文提出 Accurate Addition-Only Spiking Self-Attention（A²OS²A），通过融合二值、ReLU 和三值脉冲神经元的混合策略，在保持纯加法计算（无乘法）的前提下显著提升脉冲Transformer精度，ImageNet-1K 上达到 78.66%。
