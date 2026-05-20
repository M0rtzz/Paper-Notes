---
title: >-
  ACL2026 multi_agent方向21篇论文解读
description: >-
  21篇ACL2026的 multi_agent 方向论文解读，涵盖 Agent、LLM、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2026"
  - "multi_agent"
  - "论文解读"
  - "论文笔记"
  - "Agent"
  - "LLM"
  - "推理"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📄 multi_agent

**💬 ACL2026** · **21** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (6)](../../ICML2026/multi_agent/index.md) · [💬 ACL2025 (8)](../../ACL2025/multi_agent/index.md)

🔥 **高频主题：** Agent ×19 · LLM ×8 · 推理 ×3

**[BookAgent: Orchestrating Safety-Aware Visual Narratives via Multi-Agent Cognitive Calibration](bookagent_orchestrating_safety-aware_visual_narratives_via_multi-agent_cognitive.md)**

:   BookAgent 是一个安全感知的多智能体框架，通过**价值对齐故事板（VAS）+ 迭代跨模态精炼（ICR）+ 时序认知校准（TCC）**三阶段闭环架构，从用户草稿端到端生成高质量、角色一致、内容安全的绘本故事。

**[CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](cia_inferring_the_communication_topology_from_llm-based_multi-agent_systems.md)**

:   本文提出 CIA（Communication Inference Attack），在严格黑盒只能观测最终输出的设定下，通过对抗性查询诱导多智能体系统暴露中间 agent 的推理输出，再用全局偏置解纠缠 + LLM 弱监督建模语义相关性，成功反演出 MAS 的通信拓扑，平均 AUC 0.87、峰值 0.99。

**[Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games](collaborative_multi-agent_scripts_generation_for_enhancing_imperfect-information.md)**

:   提出一个协作式多智能体框架用于自动生成高质量剧本杀游戏脚本和训练数据，通过两阶段训练策略（CoT 微调 + GRPO 强化学习配合 ScoreAgent 奖励塑形）增强 VLM 在不完全信息下的多跳推理能力，在 WhodunitBench 上显著提升 VLM 的叙事推理、事实提取和欺骗抵御能力。

**[Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](diversity_collapse_in_multi-agent_llm_systems_structural_coupling_and_collective.md)**

:   本文通过评估超过 10,000 个研究提案，从模型智能、智能体认知和系统动力学三个层次系统揭示了多智能体 LLM 系统中的"多样性崩溃"现象：更强的模型、权威驱动的角色分配和密集的通信拓扑都会抑制语义多样性，根本原因是交互结构而非模型能力不足。

**[Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search](efficient_multi-agent_system_training_with_data_influence-oriented_tree_search.md)**

:   提出 DITS，把 "训练数据影响分数 (influence score)" 而非传统 Q-value 作为 MCTS 树搜索和偏好数据选择的指挥棒，并为不可微指标推导出一个 "前向推理就能算" 的影响分数估计公式，使 MAS 在 7 个数据集 / 3 个多智能体任务上平均比 Optima-iSFT-DPO 再提升 2.5–2.7%。

**[MAGEO: From Experience to Skill — Multi-Agent Generative Engine Optimization via Reusable Strategy Learning](from_experience_to_skill_multi-agent_generative_engine_optimization_via_reusable.md)**

:   本文将生成引擎优化（GEO）从逐实例启发式优化重构为策略学习问题，提出 MAGEO 多智能体框架——执行层由偏好/规划/编辑/评估四个智能体协作，学习层将验证有效的编辑模式蒸馏为可复用的引擎特定策略技能，并引入 Twin Branch 因果评估协议和 DSV-CF 双轴指标，在三个主流引擎上显著优于启发式基线。

**[From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)**

:   本文构建了JurisCQAD——一个包含43000+真实中文法律咨询的大规模数据集，并提出JurisMA多智能体框架，通过法律元素图进行结构化任务分解和动态多Agent协作（管理Agent+格式检查+法条检索），在LawBench上显著优于通用和法律专用LLM。

**[HACHIMI: Scalable and Controllable Student Persona Generation via Orchestrated Agents](hachimi_scalable_and_controllable_student_persona_generation_via_orchestrated_ag.md)**

:   HACHIMI 把"学生画像生成"形式化为 TAD-PG（理论对齐 + 分布可控）任务，用"提议–验证–修订"多智能体框架配合神经符号验证器和分层采样，产出 100 万条 1–12 年级合成学生画像；在 CEPS / PISA 2022 群体级评测中显示出明显的「保真梯度」——数学与好奇心相关构念高度对齐，而幸福感和家庭动态构念则只能弱对齐。

**[Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate](latent_agents_a_post-training_procedure_for_internalized_multi-agent_debate.md)**

:   提出 IMAD（Internalized Multi-Agent Debate）框架，用 SFT + GRPO 两阶段后训练把多智能体辩论"内化"进单个 LLM，token 消耗最多减少 93%，并通过激活引导证明内化后的模型在隐空间中保留了可分离、可控的"智能体子空间"。

**[MASFactory: A Graph-centric Framework for Orchestrating LLM-Based Multi-Agent Systems with Vibe Graphing](masfactory_a_graph-centric_framework_for_orchestrating_llm-based_multi-agent_sys.md)**

:   MASFactory 把 LLM 多智能体系统建模成 Node / Edge 计算图，提出 "Vibe Graphing" 三阶段流水线（Role Assignment → Structure Design → Semantic Completion）把自然语言意图编译成可执行 MAS 工作流，并提供 Context/Message Adapter、ComposedGraph 模板复用与 VS Code 可视化；7 个 benchmark 上复现 5 个代表性 MAS 且效果持平甚至更好，端到端 Vibe Graphing 把 ChatDev 1511 行代码压到 45 行、API 成本比 Vibe Coding 低 1 个数量级。

**[MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering](mata_multi-agent_framework_for_reliable_and_flexible_table_question_answering.md)**

:   提出 MATA 多Agent表格问答框架，通过调度器优先选择推理路径（CoT/PoT/text2SQL）、置信度检查器筛选答案、法官Agent仲裁，实现模型无关的高效准确表格QA，在10个LLM上平均EM提升40.1%。

**[Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)**

:   提出 MALMAS，一个记忆增强的 LLM 多智能体系统用于表格数据自动特征生成，通过六个专职 Agent 分工探索不同特征空间维度 + 三级记忆机制（过程/反馈/概念）实现跨轮迭代优化，在 16 个分类和 7 个回归数据集上超越现有基线。

**[OxyGent: Making Multi-Agent Systems Modular, Observable, and Evolvable via Oxy Abstraction](oxygent_making_multi-agent_systems_modular_observable_and_evolvable_via_oxy_abst.md)**

:   OxyGent 把 agent、工具、LLM 与推理流程统一封装成可插拔的 Oxy 原子组件，并用权限驱动的动态规划与 OxyBank 数据回流机制，让工业级多智能体系统更容易搭建、监控和持续演化。

**[Preference Estimation via Opponent Modeling in Multi-Agent Negotiation](preference_estimation_via_opponent_modeling_in_multi-agent_negotiation.md)**

:   提出将 LLM 提取的自然语言偏好信号与贝叶斯对手建模框架结合的偏好估计方法，在多方多议题谈判中通过语言似然函数融合定性线索和定量出价信息，将完全达成协议率从 37% 提升至 62%。

**[RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems](roadmapper_a_multi-agent_system_for_roadmap_generation_of_solving_complex_resear.md)**

:   本文提出 RoadMap 研究路线图生成基准和 RoadMapper 多智能体系统，用知识检索、逻辑/粒度批评、修订与 DPO 评估器组成闭环，在中英文复杂科研问题上比直接提示平均提升约 7-9 分，并显著降低专家设计路线图的时间成本。

**[Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems](seeing_the_whole_elephant_a_benchmark_for_failure_attribution_in_llm-based_multi.md)**

:   TraceElephant 主张多智能体失败归因应在开发者真实可见的完整执行轨迹下评测，并提供 220 条失败 trace、责任 agent 与关键失败 step 标注，证明 full observability 能将 step-level 归因从输出-only 的 16% 提升到 28%-30% 以上。

**[SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)**

:   本文提出 SILO-BENCH，一个角色无关的多智能体 LLM 分布式协调基准，包含 30 个算法任务、三个通信复杂度级别、54 种配置共 1620 个实验，揭示了关键的"通信-推理鸿沟"：智能体能自发形成合理通信拓扑并积极交换信息，但系统性地无法将分布式状态整合为正确答案。

**[Social Dynamics as Critical Vulnerabilities that Undermine Objective Decision-Making in LLM Collectives](social_dynamics_as_critical_vulnerabilities_that_undermine_objective_decision-ma.md)**

:   这篇论文证明，LLM 多智能体系统中的代表智能体不仅会受自身推理能力限制，还会被同伴数量、同伴能力、论证长度和修辞风格等“社会动力学”显著影响，从而在有客观答案的任务上做出错误决策。

**[To Trust or Not to Trust: Attention-Based Trust Management for LLM Multi-Agent Systems](to_trust_or_not_to_trust_attention-based_trust_management_for_llm_multi-agent_sy.md)**

:   本文为 LLM 多智能体系统（LLM-MAS）提出首个全面的"可信度"定义（基于 Grice 合作原则的六个正交维度），发现 LLM 的注意力模式可区分不同类型的可信度违规，据此设计了轻量级的 A-Trust 评估方法和端到端的信任管理系统（TMS），在多种攻击下将恶意消息检测率提升至 77-90%。

**[Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration](towards_robust_real-world_spreadsheet_understanding_with_multi-agent_multi-forma.md)**

:   提出 SpreadsheetAgent，一种两阶段多智能体框架，通过代码执行、视觉和 LaTeX 三种格式的渐进式区域读取与交叉验证，在不超出 LLM 上下文限制的前提下实现鲁棒的真实世界电子表格理解。

**[When Identity Skews Debate: Anonymization for Bias-Reduced Multi-Agent Reasoning](when_identity_skews_debate_anonymization_for_bias-reduced_multi-agent_reasoning.md)**

:   这篇论文指出多智能体辩论中的 LLM 会因为“谁说的”而不是“说了什么”改变立场，并通过响应匿名化与 Identity Bias Coefficient 量化和削弱这种身份驱动偏差。
