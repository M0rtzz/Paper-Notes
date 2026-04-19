---
title: >-
  ICML2025 信号/通信方向 5篇论文解读
description: >-
  5篇ICML2025 信号/通信方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📡 信号/通信

**🧪 ICML2025** · **5** 篇论文解读

**[Eigenspectrum Analysis of Neural Networks without Aspect Ratio Bias](eigenspectrum_analysis_of_neural_networks_without_aspect_ratio_bias.md)**

:   论文提出 FARMS（Fixed-Aspect-Ratio Matrix Subsampling），通过固定长宽比子矩阵采样来消除权重特征谱分析中的长宽比偏差，从而显著提升基于 HT-SR 的分层学习率分配与模型剪枝效果。

**[Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization](fourier_position_embedding_enhancing_attentions_periodic_extension_for_length_ge.md)**

:   通过将 RoPE 中每个维度从单一频率扩展为傅里叶级数表示，并裁剪欠训练的低频分量，实现注意力机制的可靠周期性扩展，从而大幅提升 LLM 的长度泛化能力。

**[Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization](large_language_model_llm-enabled_in-context_learning_for_wireless_network_optimi.md)**

:   提出基于 LLM 上下文学习（In-context Learning）的基站功率控制算法，通过自然语言任务描述和经验池驱动的示例选择，在不更新模型参数的条件下达到接近传统深度强化学习的性能。

**[Reward-Augmented Data Enhances Direct Preference Alignment of LLMs](reward-augmented_data_enhances_direct_preference_alignment_of_llms.md)**

:   提出一种**奖励增强的数据重标注方法**，通过将偏好对条件化于奖励分数构建扩增数据集，使DPO能感知回复质量全谱，缓解高质量rejected回复被遗忘和低质量chosen回复被盲目学习的问题，在多个基准上一致性大幅提升DPO性能。

**[SepLLM: Accelerate Large Language Models by Compressing One Segment into One Separator](sepllm_accelerate_large_language_models_by_compressing_one_segment_into_one_sepa.md)**

:   提出 SepLLM，利用分隔符 token（标点符号等）天然压缩文本段落信息的特性，仅保留 Initial + Separator + Neighboring 三类 token 的 KV 缓存，在保持性能的同时大幅减少注意力计算和内存占用。
