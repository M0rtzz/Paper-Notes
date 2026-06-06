---
title: >-
  ICML2025 对话系统方向3篇论文解读
description: >-
  3篇ICML2025的对话系统方向论文解读，涵盖个性化生成、Agent、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "对话系统"
  - "论文解读"
  - "论文笔记"
  - "个性化生成"
  - "Agent"
  - "LLM"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🗣️ 对话系统

**🧪 ICML2025** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (4)](../../ICML2026/dialogue/index.md) · [💬 ACL2026 (26)](../../ACL2026/dialogue/index.md) · [📷 CVPR2026 (1)](../../CVPR2026/dialogue/index.md) · [🔬 ICLR2026 (5)](../../ICLR2026/dialogue/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/dialogue/index.md) · [🧠 NeurIPS2025 (5)](../../NeurIPS2025/dialogue/index.md)

**[Agent WARPP: Workflow Adherence via Runtime Parallel Personalization](agent_warpp_workflow_adherence_via_runtime_parallel_personalization.md)**

:   提出 WARPP，一个无需训练的多智能体框架，在运行时根据用户属性动态剪枝条件分支工作流，并通过并行化的 Personalizer 智能体与模块化域特定智能体协同执行，在提升工具调用精度和参数保真度的同时减少 token 消耗。

**[Investigating Non-Transitivity in LLM-as-a-Judge](investigating_non-transitivity_in_llm-as-a-judge.md)**

:   揭示了 LLM-as-a-Judge 框架中评判偏好的**非传递性**问题（A>B, B>C 不能推出 A>C），证明固定基线模型的排名方式不可靠，提出基于循环赛 + Bradley-Terry 模型的排名方法及高效的 Swim 锦标赛策略。

**[Position: Uncertainty Quantification Needs Reassessment for Large-language Model Agents](position_uncertainty_quantification_needs_reassessment_for_large-language_model_.md)**

:   本文是一篇 Position Paper，通过梳理文献中 aleatoric 和 epistemic 不确定性的多种相互矛盾的定义，论证传统二分法在 LLM 交互场景中根本性失效，并提出 underspecification uncertainty（任务/上下文欠规范）、interactive learning（通过追问减少不确定性）和 output uncertainty（用自然语言而非标量表达不确定性）三个新研究方向。
