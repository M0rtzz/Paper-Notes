---
title: >-
  ICCV2025 信息检索/RAG方向 6篇论文解读
description: >-
  6篇ICCV2025 信息检索/RAG方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 信息检索/RAG

**📹 ICCV2025** · **6** 篇论文解读

**[Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation](aligning_information_capacity_between_vision_and_language_via_dense-to-sparse_fe.md)**

:   提出D2S-VSE框架，通过两阶段训练（稠密文本预训练+稠密到稀疏特征蒸馏微调）增强视觉语义嵌入的信息容量，解决图文匹配中图像与文本信息密度不对称的核心问题。

**[LangBridge: Interpreting Image as a Combination of Language Embeddings](langbridge_interpreting_image_as_a_combination_of_language_embeddings.md)**

:   LangBridge 通过将视觉特征显式分解为 LLM 词汇嵌入的线性组合，实现了可解释的视觉-语言对齐，并支持跨 LLM 的预训练无关适配器迁移。

**[MonSTeR: a Unified Model for Motion, Scene, Text Retrieval](monster_a_unified_model_for_motion_scene_text_retrieval.md)**

:   提出 **MonSTeR**——首个**运动-场景-文本三模态检索模型**，通过受拓扑深度学习启发的高阶关系建模，构建统一隐空间以捕获三模态之间的内在依赖关系，在多项检索任务上大幅超越仅依赖单模态表征的基线，并可用于人-场景交互模型的评估。

**[OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation](ocr_hinders_rag_evaluating_the_cascading_impact_of_ocr_on_retrieval-augmented_ge.md)**

:   提出 OHRBench——首个评估 OCR 对 RAG 系统级联影响的基准，包含 7 个领域的 8561 张文档图像和 8498 个 QA 对，系统性地揭示了 OCR 产生的语义噪声（Semantic Noise）和格式噪声（Formatting Noise）对检索和生成两阶段的不同影响模式。

**[Representation Shift: Unifying Token Compression with FlashAttention](representation_shift_unifying_token_compression_with_flashattention.md)**

:   提出 Representation Shift，一种无需训练、模型无关的 token 重要性度量方法，通过计算 token 在网络层前后的表征变化量来衡量重要性，从而首次实现 token 压缩与 FlashAttention 的兼容，在视频理解和图像分类上取得高达 5.5× 的加速。

**[ViLU: Learning Vision-Language Uncertainties for Failure Prediction](vilu_learning_vision-language_uncertainties_for_failure_prediction.md)**

:   提出 ViLU，一个针对 VLM 零样本预测的后验不确定性量化框架，通过交叉注意力融合视觉嵌入、预测文本嵌入和图像条件文本表示，构建不确定性感知的多模态表征，在 13 个分类数据集和大规模图文数据集上显著超越现有失败预测方法。
