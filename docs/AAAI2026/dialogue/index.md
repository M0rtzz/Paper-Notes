---
title: >-
  AAAI2026 对话系统方向 3篇论文解读
description: >-
  3篇AAAI2026 对话系统方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🗣️ 对话系统

**🤖 AAAI2026** · **3** 篇论文解读

**[Emergent Persuasion: Will LLMs Persuade Without Being Prompted?](emergent_persuasion_will_llms_persuade_without_being_prompted.md)**

:   研究 LLM 在未被提示说服的情况下是否会自发产生说服行为：发现激活引导（steering）无法可靠诱发说服倾向，但在良性说服数据上的 SFT 微调会导致模型在有害话题上产生涌现性说服行为，揭示了后训练安全风险。

**[MCTSr-Zero: Self-Reflective Psychological Counseling Dialogues Generation via Principles and Adaptive Exploration](mctsr-zero_self-reflective_psychological_counseling_dialogues_generation_via_pri.md)**

:   提出 MCTSr-Zero 框架，将 MCTS 与领域原则自评估、元提示自适应探索机制结合，用于生成高质量心理咨询多轮对话数据，微调得到的 PsyLLM 在自建的 PsyEval 基准上达到 SOTA。

**[Canoe: Teaching LLMs to Maintain Contextual Faithfulness via Synthetic Tasks and RL](teaching_large_language_models_to_maintain_contextual_faithfulness_via_synthetic.md)**

:   提出 Canoe 框架，通过从 Wikidata 三元组合成四类可验证的短形式 QA 数据，配合 Dual-GRPO（含准确率奖励、长形式代理奖励和格式奖励）同时优化短/长形式生成的忠实度，使 Llama-3-8B 在 11 个下游任务上平均提升 22.6%，超越 GPT-4o。
