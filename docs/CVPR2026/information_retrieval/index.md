---
title: >-
  CVPR2026 信息检索/RAG论文汇总 · 9篇论文解读
description: >-
  9篇CVPR2026的信息检索/RAG 方向论文解读，涵盖多模态、RAG、对齐/RLHF、问答、少样本学习、自监督学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2026"
  - "信息检索/RAG"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "RAG"
  - "对齐/RLHF"
  - "问答"
  - "少样本学习"
  - "自监督学习"
item_list:
  - u: "beyond_global_similarity_towards_fine-grained_multi-condition_multimodal_retriev/"
    t: "Beyond Global Similarity: Towards Fine-Grained, Multi-Condition Multimodal Retrieval"
  - u: "bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment/"
    t: "BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment"
  - u: "cc-vqa_conflict-_and_correlation-aware_method_for_mitigating_knowledge_conflict_/"
    t: "CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering"
  - u: "explaining_clip_zero-shot_predictions_through_concepts/"
    t: "Explaining CLIP Zero-shot Predictions Through Concepts"
  - u: "m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag/"
    t: "M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG"
  - u: "muco_multi-turn_contrastive_learning_for_multimodal_embedding_model/"
    t: "MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model"
  - u: "nanovdr_distilling_a_2b_vision-language_retriever_into_a_70m_text-only_encoder_f/"
    t: "NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval"
  - u: "pinpoint_evaluation_of_composed_image_retrieval_with_explicit_negatives_multi-im/"
    t: "PinPoint: Evaluation of Composed Image Retrieval with Explicit Negatives, Multi-Image Queries, and Paraphrase Testing"
  - u: "robustvisrag_causality-aware_vision-based_retrieval-augmented_generation_under_v/"
    t: "RobustVisRAG: Causality-Aware Vision-Based Retrieval-Augmented Generation under Visual Degradations"
item_total: 9
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 信息检索/RAG

**📷 CVPR2026** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (22)](../../ICML2026/information_retrieval/index.md) · [💬 ACL2026 (73)](../../ACL2026/information_retrieval/index.md) · [🔬 ICLR2026 (31)](../../ICLR2026/information_retrieval/index.md) · [🤖 AAAI2026 (26)](../../AAAI2026/information_retrieval/index.md) · [🧠 NeurIPS2025 (30)](../../NeurIPS2025/information_retrieval/index.md) · [📹 ICCV2025 (7)](../../ICCV2025/information_retrieval/index.md)

🔥 **高频主题：** 多模态 ×5 · RAG ×2

**[Beyond Global Similarity: Towards Fine-Grained, Multi-Condition Multimodal Retrieval](beyond_global_similarity_towards_fine-grained_multi-condition_multimodal_retriev.md)**

:   提出MCMR（Multi-Conditional Multimodal Retrieval）大规模基准，通过双证据设计（部分属性仅可从图像推断、部分仅可从文本获取）确保检索任务不可被单模态解决，系统评估5个检索器和7个MLLM重排器，揭示模态不对称性和细粒度推理差距。

**[BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment](bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)**

:   提出 BRIDGE 系统，通过 FORGE（RL 训练的查询对齐模型）将噪声多模态查询蒸馏为检索优化的纯文本查询，配合 LENS 推理增强检索器，在 MM-BRIGHT 上达到 29.7 nDCG@10，作为插件进一步将 Nomic-Vision 提升到 33.3，超越最佳纯文本检索器。

**[CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering](cc-vqa_conflict-_and_correlation-aware_method_for_mitigating_knowledge_conflict_.md)**

:   提出 CC-VQA，一种 training-free 的知识冲突缓解方法，通过视觉中心的上下文冲突推理和相关度引导的编码/解码两阶段策略，在 E-VQA、InfoSeek、OK-VQA 三个基准上取得 3.3%-6.4% 的绝对精度提升。

**[Explaining CLIP Zero-shot Predictions Through Concepts](explaining_clip_zero-shot_predictions_through_concepts.md)**

:   本文提出 EZPC，通过学习一个线性投影矩阵将 CLIP 的图像-文本嵌入映射到可解释的概念空间，在几乎不损失零样本分类精度的前提下（CIFAR-100/CUB/ImageNet-100 上 H-mean 仅差约 1%），为 CLIP 的预测提供基于人类可理解概念的忠实解释，且推理开销仅增加约 0.1ms。

**[M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG](m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag.md)**

:   提出首个大规模多语言多文化多模态 RAG 评估框架 M4-RAG，覆盖 42 种语言和 189 个国家的 80K+ 文化 VQA 实例，系统性揭示了 RAG 对小模型有效但无法随模型规模正向扩展、跨语言检索存在严重性能退化的关键发现。

**[MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model](muco_multi-turn_contrastive_learning_for_multimodal_embedding_model.md)**

:   MuCo 提出了一种基于多轮对话的对比学习框架，利用 MLLM 的对话能力在单次前向传播中同时处理多个关联的 query-target 对，大幅提升训练效率，并在 MMEB 和 M-BEIR 检索基准上取得 SOTA 性能。

**[NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval](nanovdr_distilling_a_2b_vision-language_retriever_into_a_70m_text-only_encoder_f.md)**

:   NanoVDR 利用查询-文档的模态不对称性，将 2B VLM 教师的查询编码能力通过 pointwise cosine alignment 蒸馏到 69M 纯文本编码器，在 ViDoRe 基准上保留 95.1% 教师性能、查询延迟降低 50 倍，训练仅需 13 GPU 小时。

**[PinPoint: Evaluation of Composed Image Retrieval with Explicit Negatives, Multi-Image Queries, and Paraphrase Testing](pinpoint_evaluation_of_composed_image_retrieval_with_explicit_negatives_multi-im.md)**

:   提出 PinPoint 基准，包含 7,635 个查询和 329K 人工验证的相关性判断，通过显式负样本、多图像查询、释义变体和人口统计元数据四个维度，揭示了现有 CIR 方法在假阳性抑制、语言鲁棒性和多图像推理上的严重缺陷，并提出基于 MLLM 的无训练重排方法作为改进基线。

**[RobustVisRAG: Causality-Aware Vision-Based Retrieval-Augmented Generation under Visual Degradations](robustvisrag_causality-aware_vision-based_retrieval-augmented_generation_under_v.md)**

:   提出 RobustVisRAG，一个因果引导的双路径框架，通过非因果路径捕获退化信号、因果路径学习纯净语义来解耦 VisRAG 中的语义-退化纠缠，在真实世界退化条件下检索、生成和端到端性能分别提升 7.35%、6.35% 和 12.40%，同时保持干净数据上的性能。
