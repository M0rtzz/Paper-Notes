---
title: >-
  ICML2026 multi_agent方向6篇论文解读
description: >-
  6篇ICML2026的 multi_agent 方向论文解读，涵盖 Agent、LLM、扩散模型、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "multi_agent"
  - "论文解读"
  - "论文笔记"
  - "Agent"
  - "LLM"
  - "扩散模型"
  - "推理"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📄 multi_agent

**🧪 ICML2026** · **6** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (21)](../../ACL2026/multi_agent/index.md) · [💬 ACL2025 (8)](../../ACL2025/multi_agent/index.md)

🔥 **高频主题：** Agent ×5 · LLM ×4

**[E-mem: Multi-Agent Based Episodic Context Reconstruction for LLM Agent Memory](e-mem_multi-agent_based_episodic_context_reconstruction_for_llm_agent_memory.md)**

:   E-mem 把"预处理压缩成嵌入/图"的传统记忆范式改成"保留原始上下文 + 小模型助手就地推理"的情景重构范式：master agent 只做全局规划，多个 SLM assistant 各自守着一段未压缩的原文，按多路由检索激活后再做局部推理回传证据，在 LoCoMo 上 F1 反超 SOTA 7.75 个点的同时把 token 消耗砍掉 70%。

**[EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions](engiagent_fully_connected_coordination_of_llm_agents_for_solving_open-ended_engi.md)**

:   EngiAgent 把工程问题求解拆成 Analyzer/Modeler/Verifier/Solver/Evaluator 五个专家 Agent，再用一个**全连接协调器**动态路由反馈（而不是走固定流水线），让 GPT-4o 上工程任务的可行解率从 5.66%（zero-shot）/7.55%（MM-Agent）一跃到 64.15%，平均比此前 SOTA 提升约 7 倍。

**[MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](maspo_joint_prompt_optimization_for_llm-based_multi-agent_systems.md)**

:   MASPO 通过多粒度联合评价（局部有效性 + 前瞻潜力 + 全局对齐）+ 错位案例驱动的进化束搜索，在不依赖标注的前提下端到端地为整条多智能体链路联合优化角色提示词，6 个任务上平均提升约 2.9 分。

**[OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration](omac_a_holistic_optimization_framework_for_llm-based_multi-agent_collaboration.md)**

:   本文把多智能体系统的优化空间形式化为五个维度（两个功能维度 + 三个结构维度），用"Semantic Initializer 生成 + Contrastive Comparator 对比改进"的双 actor 算法在每个维度上做监督式优化，再迭代联合优化多个维度，在 HumanEval / MMLU / MATH 上稳定打败 DyLAN、ADAS、AFlow 等基线。

**[RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation](radar_redundancy-aware_diffusion_for_multi-agent_communication_structure_generat.md)**

:   RADAR 把多 LLM-Agent 系统的通信拓扑设计建模为一个"冗余感知"的离散图扩散过程，用 effective size 作为指导信号一步步增量生成 query-自适应的协作图，在 6 个基准上同时拿到更高准确率、更低 token 消耗和更强鲁棒性。

**[Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs](systematic_failures_in_collective_reasoning_under_distributed_information_in_mul.md)**

:   本文将社会心理学的 Hidden Profile 范式搬到多智能体 LLM 评测里，构建 65 任务的 HiddenBench，在 15 个前沿 LLM 上系统揭示：单 agent 在 Full Profile 下能 80.7% 答对的同类任务，多 agent 在分布式信息下仅 30.1%，根本失败模式是**不会主动 elicit 别人没说出来的信息**，而轻量结构化沟通协议能跨家族大幅缓解。
