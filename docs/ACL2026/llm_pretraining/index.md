---
title: >-
  ACL2026 预训练方向5篇论文解读
description: >-
  5篇ACL2026的预训练方向论文解读，涵盖 LLM、布局/合成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📚 预训练

**💬 ACL2026** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (8)](../../CVPR2026/llm_pretraining/index.md) · [🔬 ICLR2026 (26)](../../ICLR2026/llm_pretraining/index.md) · [🤖 AAAI2026 (6)](../../AAAI2026/llm_pretraining/index.md) · [🧠 NeurIPS2025 (48)](../../NeurIPS2025/llm_pretraining/index.md) · [📹 ICCV2025 (9)](../../ICCV2025/llm_pretraining/index.md) · [🧪 ICML2025 (30)](../../ICML2025/llm_pretraining/index.md)

**[Commonsense Knowledge with Negation: A Resource to Enhance Negation Understanding](commonsense_knowledge_with_negation_a_resource_to_enhance_negation_understanding.md)**

:   提出自动为现有常识知识库增添否定的方法，构建超过 200 万三元组的否定常识语料库（¬Atomic 和 ¬Anion），并证明在其上预训练可以提升 LLM 的否定理解能力。

**[Compact Example-Based Explanations for Language Models](compact_example-based_explanations_for_language_models.md)**

:   本文提出选择相关性分数（Selection Relevance Score），一种无需重训练的指标来评估训练样本子集作为示例解释的质量，并证明常见的"选最高影响力"策略常不如随机选择，进而提出平衡影响力与代表性的新策略。

**[SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization](sage_sign-adaptive_gradient_for_memory-efficient_llm_optimization.md)**

:   本文提出 SAGE 优化器，通过 Lion 风格的符号更新方向和一个 $O(d)$ 内存开销的自适应阻尼缩放因子，解决了轻量级优化器在嵌入层上失败的"嵌入层困境"，在 Llama 模型（最大 1.3B）上以显著更低的优化器内存达到新的 SOTA 困惑度。

**[SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models](script_a_subcharacter_compositional_representation_injection_module_for_korean_p.md)**

:   本文提出 SCRIPT，一个模型无关的即插即用模块，通过双通道策略将韩文 Hangul 的子字符（Jamo）组合知识注入现有子词级 PLM 的嵌入层，无需重新预训练即可在韩语 NLU/NLG 任务上获得一致提升，并使嵌入空间更好地捕捉语法规律和语义变化。

**[Working Memory Constraints Scaffold Learning in Transformers under Data Scarcity](working_memory_constraints_scaffold_learning_in_transformers_under_data_scarcity.md)**

:   本文将人类工作记忆约束（固定窗口、指数衰减、逻辑衰减、首因-近因效应）集成到 GPT-2 注意力机制中，在发展可信的小规模语料（10M/100M 词）上从头训练，发现这些约束在数据稀缺时显著提升语法准确率和人类阅读时间的预测力，且促进注意力头的功能专门化。
