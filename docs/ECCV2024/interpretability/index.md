---
title: >-
  ECCV2024 可解释性方向 4篇论文解读
description: >-
  4篇ECCV2024 可解释性方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔬 可解释性

**🎞️ ECCV2024** · 共 **4** 篇

**[Detailsemnet Elevating Signature Verification Through Detail-Semantic Integratio](detailsemnet_elevating_signature_verification_through_detail-semantic_integratio.md)**

:   提出DetailSemNet用于离线签名验证，通过Detail-Semantics Integrator将特征解耦为细节和语义两个分支分别处理，并引入基于EMD的局部结构匹配，在多个多语言签名数据集上取得SOTA。

**[Improving Intervention Efficacy Via Concept Realignment In Concept Bottleneck Mo](improving_intervention_efficacy_via_concept_realignment_in_concept_bottleneck_mo.md)**

:   本文发现 Concept Bottleneck Models (CBMs) 中人工干预效率低下的原因在于干预时各概念独立处理、忽视了概念间关联，提出了一个轻量级的 Concept Intervention Realignment Module (CIRM)，在干预后自动重新对齐相关概念的预测值，将达到目标性能所需的干预次数最多减少 70%。

**[Plot Text-Based Person Search With Part Slot Attention For Corresponding Part Di](plot_text-based_person_search_with_part_slot_attention_for_corresponding_part_di.md)**

:   提出 PLOT 框架，利用基于 Slot Attention 的 Part Discovery Module 自动发现跨模态（图像-文本）对应的人体部件，结合 Text-based Dynamic Part Attention（TDPA）动态调整各部件重要性，无需部件级标注即可在三个 benchmark 上全面超越 SOTA。

**[Poa Pre-Training Once For Models Of All Sizes](poa_pre-training_once_for_models_of_all_sizes.md)**

:   POA 提出在自监督自蒸馏框架中引入**弹性学生分支**，通过参数共享和随机子网络采样，**一次预训练即可同时产出上百个不同大小的预训练模型**（如从 ViT-L 直接提取 ViT-S/B），各子网络在 k-NN、线性探测和下游任务上均达到 SOTA 水平。
