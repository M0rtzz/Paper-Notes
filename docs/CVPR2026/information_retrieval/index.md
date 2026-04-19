---
title: >-
  CVPR2026 信息检索/RAG方向 8篇论文解读
description: >-
  8篇CVPR2026 信息检索/RAG方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 信息检索/RAG

**📷 CVPR2026** · **8** 篇论文解读

**[Beyond Global Similarity: Towards Fine-Grained, Multi-Condition Multimodal Retrieval](beyond_global_similarity_towards_fine-grained_multi-condition_multimodal_retriev.md)**

:   提出MCMR（Multi-Conditional Multimodal Retrieval）大规模基准，通过双证据设计（部分属性仅可从图像推断、部分仅可从文本获取）确保检索任务不可被单模态解决，系统评估5个检索器和7个MLLM重排器，揭示模态不对称性和细粒度推理差距。

**[CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering](cc-vqa_conflict-_and_correlation-aware_method_for_mitigating_knowledge_conflict_.md)**

:   提出 CC-VQA，一种 training-free 的知识冲突缓解方法，通过视觉中心的上下文冲突推理和相关度引导的编码/解码两阶段策略，在 E-VQA、InfoSeek、OK-VQA 三个基准上取得 3.3%-6.4% 的绝对精度提升。

**[Explaining CLIP Zero-shot Predictions Through Concepts](explaining_clip_zero-shot_predictions_through_concepts.md)**

:   本文提出 EZPC，通过学习一个线性投影矩阵将 CLIP 的图像-文本嵌入映射到可解释的概念空间，在几乎不损失零样本分类精度的前提下（CIFAR-100/CUB/ImageNet-100 上 H-mean 仅差约 1%），为 CLIP 的预测提供基于人类可理解概念的忠实解释，且推理开销仅增加约 0.1ms。

**[M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG](m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag.md)**

:   提出首个大规模多语言多文化多模态 RAG 评估框架 M4-RAG，覆盖 42 种语言和 189 个国家的 80K+ 文化 VQA 实例，系统性揭示了 RAG 对小模型有效但无法随模型规模正向扩展、跨语言检索存在严重性能退化的关键发现。

**[Mind the Way You Select Negative Texts: Pursuing the Distance Consistency in OOD Detection with VLMs](mind_the_way_you_select_negative_texts_pursuing_the_distance_consistency_in_ood_.md)**

:   指出现有基于 VLM 的 OOD 检测方法使用模态内距离（文本-文本或图像-图像）选择负文本，与 CLIP 优化的跨模态距离不一致，提出 InterNeg 从文本和视觉两个视角系统地利用跨模态距离，在 ImageNet 上实现 FPR95 降低 3.47%。

**[NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval](nanovdr_distilling_a_2b_vision-language_retriever_into_a_70m_text-only_encoder_f.md)**

:   NanoVDR 利用查询-文档的模态不对称性，将 2B VLM 教师的查询编码能力通过 pointwise cosine alignment 蒸馏到 69M 纯文本编码器，在 ViDoRe 基准上保留 95.1% 教师性能、查询延迟降低 50 倍，训练仅需 13 GPU 小时。

**[NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval](nanovdr_distilling_a_2b_visionlanguage_retriever_i.md)**

:   NanoVDR 利用查询-文档的不对称性，将 2B 参数的 VLM 文档检索器通过 pointwise cosine alignment 蒸馏成 69M 的纯文本查询编码器，在 ViDoRe 基准上保留 95.1% 的教师模型性能，查询延迟降低 50 倍，训练仅需 13 GPU 小时。

**[RobustVisRAG: Causality-Aware Vision-Based Retrieval-Augmented Generation under Visual Degradations](robustvisrag_causality-aware_vision-based_retrieval-augmented_generation_under_v.md)**

:   提出 RobustVisRAG，一个因果引导的双路径框架，通过非因果路径捕获退化信号、因果路径学习纯净语义来解耦 VisRAG 中的语义-退化纠缠，在真实世界退化条件下检索、生成和端到端性能分别提升 7.35%、6.35% 和 12.40%，同时保持干净数据上的性能。
