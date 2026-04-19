---
title: >-
  ICML2025 LLM Agent方向 13篇论文解读
description: >-
  13篇ICML2025 LLM Agent方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**🧪 ICML2025** · **13** 篇论文解读

**[AdvAgent: Controllable Blackbox Red-teaming on Web Agents](advagent_controllable_blackbox_red-teaming_on_web_agents.md)**

:   提出 AdvAgent，一个基于强化学习（DPO）的黑盒红队测试框架，训练一个对抗 prompter 模型自动生成不可见的 HTML 对抗 prompt，注入网页后可误导 GPT-4V 驱动的 Web Agent 执行攻击者指定的目标动作（如将买微软股票改为买英伟达），在 440 个任务上达到 97.5% 攻击成功率，且对现有防御手段仍保持 88.8% 以上的有效性。

**[AGACCI: Affiliated Grading Agents for Criteria-Centric Interface in Educational Coding Contexts](agacci_affiliated_grading_agents_for_criteria-centric_interface_in_educational_c.md)**

:   AGACCI 提出一个由 9 个专门化 Agent 组成的多 Agent 评估框架，将教育编程作业的评估任务分解为 rubric 解析、代码执行验证、可视化评估、解释性推理评估等角色，通过协作实现比单模型 baseline 更准确、一致且可解释的 rubric 对齐反馈。

**[Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction](aguvis_unified_pure_vision_agents_for_autonomous_gui_interaction.md)**

:   提出 Aguvis，首个完全基于纯视觉的跨平台自主 GUI Agent 框架，通过统一视觉观察空间、标准化动作空间和内心独白（inner monologue）机制，在离线和在线基准上取得 SOTA，无需依赖闭源模型。

**[AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML](automl-agent_a_multi-agent_llm_framework_for_full-pipeline_automl.md)**

:   本文提出 AutoML-Agent，一个基于多智能体 LLM 协作的全流水线 AutoML 框架，通过检索增强规划策略（Retrieval-Augmented Planning）扩大搜索空间、将任务分解为并行执行的子任务由专业化 Agent 分别完成、并引入多阶段验证机制保障代码生成质量，在 7 类任务 14 个数据集上实现了更高的自动化成功率和模型性能。

**[Evaluating Retrieval-Augmented Generation Agents for Autonomous Scientific Discovery in Astrophysics](evaluating_retrieval-augmented_generation_agents_for_autonomous_scientific_disco.md)**

:   本文构建了宇宙学领域的 RAG 评测基准 CosmoPaperQA（105 个专家 QA 对），系统评估了 9 种 RAG agent 配置（涵盖商业 API、混合架构、学术工具），发现 OpenAI RAG 方案以 91.4% 准确率领先，并校准了可替代人工评审的 LLM-as-a-Judge 系统。

**[From Debate to Equilibrium: Belief-Driven Multi-Agent LLM Reasoning via Bayesian Nash Equilibrium](from_debate_to_equilibrium_belief-driven_multi-agent_llm_reasoning_via_bayesian_.md)**

:   将多 LLM 协调建模为不完全信息博弈，提出 ECON 框架，通过贝叶斯纳什均衡（BNE）实现隐式信念驱动的多 Agent 协调推理，无需显式消息传递即可获得理论收敛保证，在六个推理基准上平均提升 11.2%。

**[From Passive to Active Reasoning: Can Large Language Models Ask the Right Questions under Incomplete Information?](from_passive_to_active_reasoning_can_large_language_models_ask_the_right_questio.md)**

:   本文提出 AR-Bench，一个专门评估 LLM 主动推理能力的基准，包含侦探案件、情境谜题和猜数字三类任务，实验发现 GPT-4o 等最先进模型在需要主动提问获取缺失信息的场景中表现远逊于人类，揭示了被动推理与主动推理之间的巨大鸿沟。

**[GuardAgent: Safeguard LLM Agents via Knowledge-Enabled Reasoning](guardagent_safeguard_llm_agents_by_a_guard_agent_via_knowledge-enabled_reasoning.md)**

:   GuardAgent 是首个"用 Agent 守护 Agent"的框架，通过将安全规则动态转化为可执行的护栏代码来检查目标 Agent 的动作是否违规，在医疗访问控制和 Web 安全控制两个新基准上分别达到 98%+ 和 83%+ 的护栏准确率。

**[Improving LLM Agent Planning with In-Context Learning via Atomic Fact Augmentation and Lookahead Search](improving_llm_agent_planning_with_in-context_learning_via_atomic_fact_augmentati.md)**

:   提出 LWM-Planner，从交互轨迹中提取"原子事实"增强 LLM 世界模型模拟，结合递归前瞻搜索实现纯 in-context 的 Agent 规划改进，在 ALFWorld 等任务上显著优于 ReAct 和 Reflexion。

**[Open Source Planning & Control System with Language Agents for Autonomous Scientific Discovery](open_source_planning_control_system_with_language_agents_for_autonomous_scientif.md)**

:   本文提出 cmbagent，一个由约 30 个 LLM Agent 组成的多智能体系统，采用 Planning & Control 策略编排无人干预的科研工作流，各 Agent 分别负责论文检索、代码编写、结果解读、输出评审等专业任务，并可在本地执行代码；该系统成功完成了博士级别的宇宙学任务（用超新星数据测量宇宙学参数），在两个基准测试集上优于当前最先进的 LLM。

**[TAMAS: Benchmarking Adversarial Risks in Multi-Agent LLM Systems](tamas_benchmarking_adversarial_risks_in_multi-agent_llm_systems.md)**

:   本文提出 TAMAS，首个系统评估多智能体 LLM 系统安全性的基准，覆盖 5 个高风险领域、6 种攻击类型、300 个对抗样本和 10 个骨干模型，揭示多智能体系统在协作场景中存在严重的对抗脆弱性，并引入 ERS 指标衡量安全-效用权衡。

**[Towards LLM Agents for Earth Observation](towards_llm_agents_for_earth_observation.md)**

:   本文提出 UnivEARTH——一个包含 140 个 yes/no 问题的地球观测基准，涵盖 13 个主题和 17 种卫星传感器，评估发现最佳 LLM Agent（使用 Google Earth Engine 生成代码）的准确率仅 33%，主要受限于 58% 的代码无法运行。

**[xChemAgents: Agentic AI for Explainable Quantum Chemistry](xchemagents_agentic_ai_for_explainable_quantum_chemistry.md)**

:   xChemAgents 提出了一个 Selector-Validator 双 Agent 协作框架，将物理感知的推理注入多模态分子性质预测中：Selector Agent 自适应选择稀疏加权描述符子集并给出自然语言解释，Validator Agent 通过量纲一致性和标度律检验迭代验证，在 QM9 基准上实现最高 22% 的 MAE 降低。
