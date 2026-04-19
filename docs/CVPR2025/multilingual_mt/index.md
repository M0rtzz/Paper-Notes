---
title: >-
  CVPR2025 多语言/翻译方向 3篇论文解读
description: >-
  3篇CVPR2025 多语言/翻译方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🌐 多语言/翻译

**📷 CVPR2025** · **3** 篇论文解读

**[Harnessing Frozen Unimodal Encoders for Flexible Multimodal Alignment](harnessing_frozen_unimodal_encoders_for_flexible_multimodal_alignment.md)**

:   提出一种新的视觉-语言对齐框架：冻结预训练好的单模态视觉编码器（DINOv2）和语言编码器（All-Roberta-Large），仅训练轻量MLP投影层实现多模态对齐，以20倍数据缩减和65倍计算缩减达到了CLIP级别甚至超越的性能。

**[Semantic and Expressive Variation in Image Captions Across Languages](semantic_and_expressive_variations_in_image_captions_across_languages.md)**

:   系统性证明了不同语言的图像描述在语义内容（对象、关系、属性）和表达方式（具象度、语调、真实性）上存在显著的分布差异，多语言描述集相比单语言提供更丰富的视觉信息（+46% 对象、+66.1% 关系、+66.8% 属性），为多语言数据训练视觉模型提供了实证支撑。

**[SMTPD: A New Benchmark for Temporal Prediction of Social Media Popularity](smtpd_a_new_benchmark_for_temporal_prediction_of_social_media_popularity.md)**

:   构建首个时间对齐的社交媒体流行度时序预测基准SMTPD（282K YouTube样本，30天连续观测），并提出基于多模态特征提取+LSTM时序回归的baseline框架，发现早期流行度（EP）是准确预测后续流行度的关键。
