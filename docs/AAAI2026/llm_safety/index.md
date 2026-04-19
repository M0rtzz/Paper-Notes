---
title: >-
  AAAI2026 LLM安全方向 5篇论文解读
description: >-
  5篇AAAI2026 LLM安全方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔒 LLM安全

**🤖 AAAI2026** · **5** 篇论文解读

**[CATFormer: When Continual Learning Meets Spiking Transformers With Dynamic Thresholds](catformer_when_continual_learning_meets_spiking_transformers_with_dynamic_thresh.md)**

:   提出 CATFormer，一种基于脉冲视觉 Transformer 的无数据重放持续学习框架，通过上下文自适应的动态放电阈值实现任务特定的神经元兴奋性调节，在长达 100 个任务序列中不仅不遗忘反而准确率提升（"逆向遗忘"现象）。

**[Designing Truthful Mechanisms for Asymptotic Fair Division](designing_truthful_mechanisms_for_asymptotic_fair_division.md)**

:   提出 PRD（Proportional Response with Dummy）机制，首次在渐近公平分配设定下实现了"期望真实性 + 多项式时间可计算 + 高概率无嫉妒"三重保证，且仅需 $m = \Omega(n \log n)$ 个物品，回答了 Manurangsi & Suksompong 提出的开放问题。

**[Hallucination Stations: On Some Basic Limitations of Transformer-Based Language Models](hallucination_stations_on_some_basic_limitations_of_transformer-based_language_m.md)**

:   从计算复杂度理论出发证明 Transformer LLM 每步推理复杂度为 $O(N^2 \cdot d)$，基于时间层次定理（Hartmanis-Stearns），任何需要超过此复杂度的计算任务——如 $O(n^3)$ 矩阵乘法、$O(n^k)$ token 组合、TSP 验证等——LLM 必然无法正确完成（即产生幻觉），且 LLM Agent 也无法验证此类任务的正确性。

**[LLM Targeted Underperformance Disproportionately Impacts Vulnerable Users](llm_targeted_underperformance_disproportionately_impacts_vulnerable_users.md)**

:   系统实验表明，主流LLM（GPT-4、Claude 3 Opus、Llama 3-8B）对英语水平较低、教育程度较低、非美国出身的用户，在信息准确性、真实性和拒绝回答方面存在显著的歧视性表现下降，使最脆弱的用户成为最不可靠的信息服务对象。

**[PANDA: Patch and Distribution-Aware Augmentation for Long-Tailed Exemplar-Free Continual Learning](panda_--_patch_and_distribution-aware_augmentation_for_long-tailed_exemplar-free.md)**

:   提出 PANDA 框架，通过 CLIP 引导的语义 patch 移植实现任务内类别平衡，并借助可学习的分布平滑机制缓解任务间分布偏移，以即插即用方式提升基于预训练模型的无样本存储持续学习在长尾场景下的性能。
