---
title: >-
  ACL2025 时间序列方向 5篇论文解读
description: >-
  5篇ACL2025 时间序列方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**💬 ACL2025** · **5** 篇论文解读

**[Context-Aware Sentiment Forecasting via LLM-based Multi-Perspective Role-Playing Agents](context_aware_sentiment_forecasting_agents.md)**

:   提出一个基于 LLM 的多视角角色扮演框架（MPR），通过主观 Agent 模拟用户发帖、客观 Agent（微调的"心理学家"LLM）审查行为一致性，以迭代纠正的方式预测社交媒体用户对实时事件的未来情感反应，在宏观和微观层面均大幅超越传统方法。

**[CTPD: Cross-Modal Temporal Pattern Discovery for Enhanced Multimodal Electronic Health Records Analysis](ctpd_cross-modal_temporal_pattern_discovery_for_enhanced_multimodal_electronic_h.md)**

:   提出 CTPD 框架，利用 Slot Attention 从多模态 EHR 数据（不规则时间序列+临床笔记）中发现跨模态共享的时序原型模式，通过 TP-NCE 对比损失对齐两模态的时序语义，在 MIMIC-III 的死亡率预测和表型分类任务上取得 SOTA。

**[G2S: A General-to-Specific Learning Framework for Temporal Knowledge Graph Forecasting with Large Language Models](g2s_a_general-to-specific_learning_framework_for_temporal_knowledge_graph_foreca.md)**

:   提出 G2S 框架，将时序知识图谱（TKG）预测中的通用模式（时序结构规律）与场景信息（具体实体/关系）解耦，先在匿名化时序结构上学习通用模式，再注入场景信息，有效提升 LLM 在 TKG 预测中的泛化能力。

**[LETS-C: Leveraging Text Embedding for Time Series Classification](lets-c_leveraging_text_embedding_for_time_series_classification.md)**

:   提出 LETS-C——将时间序列数字化为文本字符串后用 text embedding 模型编码，与原始时间序列元素级相加融合后送入轻量 CNN+MLP 分类头，在 UEA 10 个多变量时间序列数据集上以仅 14.5% 的可训练参数量超越 OneFitsAll（GPT-2 微调）等 27 个 baseline 达到 SOTA。

**[Time-MQA: Time Series Multi-Task Question Answering with Context Enhancement](time-mqa_time_series_multi-task_question_answering_with_context_enhancement.md)**

:   提出Time-MQA框架和TSQA数据集（~200k QA对），将时间序列的预测、填补、异常检测、分类和开放式推理问答统一到自然语言问答范式下，通过持续预训练LLM使其具备时间序列理解和推理能力。
