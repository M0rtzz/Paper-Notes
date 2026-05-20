---
title: >-
  ICML2025 信号/通信方向5篇论文解读
description: >-
  5篇ICML2025的信号/通信方向论文解读，涵盖 LLM、压缩/编码等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "信号/通信"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "压缩/编码"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📡 信号/通信

**🧪 ICML2025** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (5)](../../CVPR2026/signal_comm/index.md) · [🔬 ICLR2026 (8)](../../ICLR2026/signal_comm/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/signal_comm/index.md) · [🧠 NeurIPS2025 (12)](../../NeurIPS2025/signal_comm/index.md) · [📹 ICCV2025 (3)](../../ICCV2025/signal_comm/index.md) · [💬 ACL2025 (1)](../../ACL2025/signal_comm/index.md)

🔥 **高频主题：** LLM ×2

**[Deep Electromagnetic Structure Design Under Limited Evaluation Budgets](deep_electromagnetic_structure_design_under_limited_evaluation_budgets.md)**

:   提出 Progressive Quadtree-based Search (PQS) 方法，通过四叉树层次化表示压缩电磁结构的高维设计空间，并利用基于一致性的样本选择机制在有限仿真预算下高效搜索优质设计，相比生成式方法节省 75~85% 的评估成本。

**[Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization](fourier_position_embedding_enhancing_attentions_periodic_extension_for_length_ge.md)**

:   通过将 RoPE 中每个维度从单一频率扩展为傅里叶级数表示，并裁剪欠训练的低频分量，实现注意力机制的可靠周期性扩展，从而大幅提升 LLM 的长度泛化能力。

**[Generative Social Choice: The Next Generation](generative_social_choice_the_next_generation.md)**

:   将生成式社会选择框架扩展至带成本/预算约束和近似查询的场景，提出 DemocraticProcess 算法并给出近乎最优的近似比例代表性理论保证，实现了实用系统 PROSE（基于 GPT-4o）在药物评论和城市治理数据集上验证有效性。

**[Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization](large_language_model_llm-enabled_in-context_learning_for_wireless_network_optimi.md)**

:   提出基于 LLM 上下文学习（In-context Learning）的基站功率控制算法，通过自然语言任务描述和经验池驱动的示例选择，在不更新模型参数的条件下达到接近传统深度强化学习的性能。

**[SepLLM: Accelerate Large Language Models by Compressing One Segment into One Separator](sepllm_accelerate_large_language_models_by_compressing_one_segment_into_one_sepa.md)**

:   提出 SepLLM，利用分隔符 token（标点符号等）天然压缩文本段落信息的特性，仅保留 Initial + Separator + Neighboring 三类 token 的 KV 缓存，在保持性能的同时大幅减少注意力计算和内存占用。
