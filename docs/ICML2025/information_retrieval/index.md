---
title: >-
  ICML2025 信息检索/RAG方向 6篇论文解读
description: >-
  6篇ICML2025 信息检索/RAG方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 信息检索/RAG

**🧪 ICML2025** · **6** 篇论文解读

**[Don't Lag, RAG: Training-Free Adversarial Detection Using RAG](dont_lag_rag_training-free_adversarial_detection_using_rag.md)**

:   本文提出 VRAG 框架，通过构建对抗补丁数据库 + 视觉检索增强生成（VRAG）+ VLM 推理的免训练 pipeline，实现对多种对抗补丁攻击的高效检测，Gemini-2.0 达到 98% 准确率，开源模型 UI-TARS-72B-DPO 达 95%。

**[Expert Evaluation of LLM World Models: A High-Tc Superconductivity Case Study](expert_evaluation_of_llm_world_models_a_high-t_c_superconductivity_case_study.md)**

:   以高温超导（HTS）领域为案例，构建了专家级数据集（1,726篇论文 + 67道专家问题），系统评估6种LLM系统的科学文献理解能力，发现基于精选文献的RAG系统在事实完整性和证据支持方面显著优于通用闭源模型。

**[POQD: Performance-Oriented Query Decomposer for Multi-Vector Retrieval](poqd_performance-oriented_query_decomposer_for_multi-vector_retrieval.md)**

:   提出 POQD，一个面向性能的查询分解框架，利用 LLM-based Prompt Optimizer 迭代优化查询分解 prompt，并通过交替训练算法联合优化 prompt 和下游 RAG 模型参数，在检索和端到端 QA 任务上大幅超越现有方法。

**[RAPID: Long-Context Inference with Retrieval-Augmented Speculative Decoding](rapid_long-context_inference_with_retrieval-augmented_speculative_decoding.md)**

:   提出 RAPID，将 RAG 与 Speculative Decoding 结合：用 RAG drafter（在短检索上下文上运行的 LLM）为长上下文目标 LLM 生成候选 token，并通过推理时知识迁移增强目标分布，在长上下文推理中同时实现 >2× 加速和生成质量提升。

**[Rethinking Addressing in Language Models via Contextualized Equivariant Positional Encoding](rethinking_addressing_in_language_models_via_contexualized_equivariant_positiona.md)**

:   本文提出 TAPE（contexTualized equivariAnt Position Encoding），通过在各层动态地根据序列内容更新位置编码来取代传统的固定位置模式，同时强制排列和正交等变性以保证稳定性，在语言建模、算术推理和长上下文检索任务上显著超越现有位置编码方法。

**[Understanding Synthetic Context Extension via Retrieval Heads](understanding_synthetic_context_extension_via_retrieval_heads.md)**

:   本文通过系统实验揭示了合成上下文扩展（synthetic context extension）为何有效的机制：合成数据训练出的"检索头"（retrieval heads）与真实数据训练出的检索头高度重叠，检索头的召回率可以预测下游长上下文任务的性能，并通过注意力剔除（attention knockout）和激活修补（activation patching）从机制层面证明了检索头的必要性。
