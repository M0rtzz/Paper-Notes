---
title: >-
  ACL2026 机器人/具身智能方向7篇论文解读
description: >-
  7篇ACL2026的机器人/具身智能方向论文解读，涵盖推理、LLM、Agent、对齐/RLHF、导航、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 机器人/具身智能

**💬 ACL2026** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (38)](../../CVPR2026/robotics/index.md) · [🔬 ICLR2026 (47)](../../ICLR2026/robotics/index.md) · [🤖 AAAI2026 (38)](../../AAAI2026/robotics/index.md) · [🧠 NeurIPS2025 (57)](../../NeurIPS2025/robotics/index.md) · [📹 ICCV2025 (26)](../../ICCV2025/robotics/index.md) · [🧪 ICML2025 (16)](../../ICML2025/robotics/index.md)

🔥 **高频主题：** 推理 ×2 · LLM ×2

**[Can AI-Generated Persuasion Be Detected? Persuaficial Benchmark and AI vs. Human Linguistic Differences](can_ai-generated_persuasion_be_detected_persuaficial_benchmark_and_ai_vs_human_l.md)**

:   本文引入 Persuaficial——一个覆盖六种语言的高质量 AI 生成说服性文本多语言基准，系统评估了 LLM 生成的说服性文本与人类撰写的说服性文本在自动检测难度上的差异，发现微妙的 AI 说服比人类说服更难检测（F1 下降约 20%），而过度强化的说服反而更容易被发现。

**[Debating the Unspoken: Role-Anchored Multi-Agent Reasoning for Half-Truth Detection](debating_the_unspoken_role-anchored_multi-agent_reasoning_for_half-truth_detecti.md)**

:   提出RADAR框架，通过角色锚定（政客 vs 科学家）的多智能体辩论来检测基于遗漏上下文的半真半假信息，配合双阈值自适应早停机制，在噪声检索条件下一致超越单智能体和传统多智能体基线。

**[DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)**

:   提出 DeCoVec（Decoding Space based Task Vector），一个无训练、非侵入式的框架，通过对比 few-shot 和 zero-shot prompt 的输出 logit 分布差异构建解码空间中的任务向量，注入解码过程引导生成，在 TruthfulQA、Math-500 和 AQUA-RAT 上比标准 few-shot 基线平均提升高达 5.50 准确率。

**[On Safety Risks in Experience-Driven Self-Evolving Agents](on_safety_risks_in_experience-driven_self-evolving_agents.md)**

:   本文系统研究经验驱动自进化Agent的安全风险，发现仅从无害任务积累的经验也导致安全性显著退化（ASR上升13-49%），根因是经验的执行导向本质强化了行动而非拒绝。

**[Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)**

:   本文提出"推理劫持"(Reasoning Hijacking) 这一新型攻击范式，通过在数据通道注入虚假决策标准来操纵 LLM 的推理逻辑而非改变任务目标，实现高攻击成功率且能绕过基于意图检测的防御方法。

**[VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions](vln-nf_feasibility-aware_vision-and-language_navigation_with_false-premise_instr.md)**

:   本文提出 VLN-NF 基准——首个要求 VLN agent 在 3D 部分可观测环境中识别虚假前提指令并输出 NOT-FOUND 的任务，配套提出 REV-SPL 评估指标和 ROAM 两阶段混合框架，ROAM 达到 6.1 REV-SPL，比监督基线提升 45%。

**[XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants](xoxo_stealthy_cross-origin_context_poisoning_attacks_against_ai_coding_assistant.md)**

:   揭示了 AI 编码助手自动收集上下文的设计漏洞，提出 Cross-Origin Context Poisoning（XOXO）攻击：通过语义保持的代码变换（如变量重命名）毒化共享代码库，使 GitHub Copilot 等助手在不知情的情况下生成有漏洞的代码，对 8 个 SOTA 模型平均攻击成功率达 73.20%。
