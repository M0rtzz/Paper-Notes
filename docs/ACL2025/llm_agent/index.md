---
title: >-
  ACL2025 LLM Agent方向 45篇论文解读
description: >-
  45篇ACL2025 LLM Agent方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**💬 ACL2025** · **45** 篇论文解读

**[A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems](a_multi-agent_framework_for_mitigating_dialect_biases_in_privacy_policy_question.md)**

:   提出一个双agent协作框架(方言Agent + 隐私政策Agent)，通过将非标准英语方言翻译为标准美式英语(SAE)并进行迭代验证，在不需要重训练或方言特定微调的前提下，显著降低隐私政策问答中的方言偏差并提升整体性能。

**[Agentic Knowledgeable Self-Awareness](agentic_knowledgeable_self-awareness.md)**

:   本文提出 KnowSelf，一种数据驱动方法，通过在 agent 自探索轨迹上标注特殊 token 来标识不同思维情境（快速思考/慢速思考/知识思考），经两阶段训练（SFT + RPO）使 agent 模型学会自主判断何时需要调用外部知识，以最小知识消耗代价达到最优规划效果。

**[Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools](agentic_reasoning_tools.md)**

:   Agentic Reasoning 提出了一个将 Web 搜索、代码执行和知识图谱记忆（Mind-Map）三种 Agent 工具集成到 LLM 推理过程中的框架，在 DeepSeek-R1 上将 Humanity's Last Exam 准确率从 9.4% 提升到 23.8%（+14.4%），GPQA 从 71.5% 到 81.2%，接近 OpenAI Deep Research 水平。

**[AndroidGen: Building an Android Language Agent under Data Scarcity](androidgen_agent_data_scarcity.md)**

:   提出 AndroidGen 框架，通过经验检索（ExpSearch）、反思规划（ReflectPlan）、自动校验（AutoCheck）和步骤级评判（StepCritic）四个模块，在高质量训练数据稀缺的条件下增强LLM的Android操作能力，并通过自动生成轨迹数据训练出无需人工标注的开源移动端agent。

**[Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)**

:   提出 Bel Esprit 多 Agent 对话框架，通过 Mentalist（需求澄清）→ Builder（管线构建）→ Inspector（验证）→ Matchmaker（模型分配）四步协作，将用户模糊的自然语言需求自动转化为多模型 AI 管线图，在 441 条管线数据上达到 25.2% EM 和 37.0 GED（GPT-4o Builder）。

**[Beyond Numeric Rewards: In-Context Dueling Bandits with LLM Agents](beyond_numeric_rewards_in-context_dueling_bandits_with_llm_agents.md)**

:   系统评估了 LLM 在 Dueling Bandits（偏好反馈强化学习）中的零样本上下文决策能力，发现 GPT-4 Turbo 在弱遗憾（weak regret）上表现出色但强遗憾（strong regret）存在差距，进而提出 LEAD 框架（LLM with Enhanced Algorithmic Dueling），通过将经典 DB 算法与 LLM 智能体细粒度自适应融合来同时获得理论保证和鲁棒性。

**[Caution for the Environment: Multimodal LLM Agents are Susceptible to Environmental Distractions](caution_environment_gui_agent_distractions.md)**

:   本文首次系统研究了多模态 GUI Agent 对环境干扰（弹窗广告、推荐内容等）的脆弱性，在无恶意攻击的自然场景下，即使最强的 MLLM（包括GPT-4o）也有 20-40% 的概率被环境中的无关内容分散注意力而执行偏离用户目标的操作。

**[Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration](dpt_agent_dual_process.md)**

:   提出 DPT-Agent，首个将双过程理论（Dual Process Theory）系统化地融入语言智能体框架的方法——用有限状态机(FSM)+code-as-policy 作为快速直觉的 System 1，用心智理论(ToM)+异步反思的 LLM 作为慢速深思的 System 2，首次实现了自主的实时同步人机协作（在 Overcooked 困难版中）。

**[EMULATE: A Multi-Agent Framework for Determining the Veracity of Atomic Claims by Emulating Human Actions](emulate_a_multi-agent_framework_for_determining_the_veracity_of_atomic_claims_by.md)**

:   提出 EMULATE 多智能体事实核查框架，通过 7 个专职 LLM agent 模拟人类验证声明的完整行为链（搜索→排序→内容评估→证据充分性判断→分类），在三个事实核查 benchmark 上的 Macro-F1 和 Weighted-F1 均超越现有方法。

**[Explorer: Scaling Exploration-Driven Web Trajectory Synthesis for Multimodal Web Agents](explorer_scaling_exploration-driven_web_trajectory_synthesis_for_multimodal_web_.md)**

:   提出 Explorer——一个可扩展的多智能体 pipeline，通过自主网页探索和逐步精炼来合成大规模多模态 web 轨迹数据集（94K 成功轨迹，49K+ URL，720K 截图），训练的 Explorer-7B 在 Mind2Web-Live、MiniWob++ 等 benchmark 上达到甚至超过 GPT-4 水平。

**[FACT-AUDIT: An Adaptive Multi-Agent Framework for Dynamic Fact-Checking Evaluation of Large Language Models](fact_audit_factcheck.md)**

:   提出FACT-AUDIT——一个基于重要性采样和多智能体协作的自适应动态事实核查评估框架，通过动态生成测试数据、迭代探测模型弱点、并同时评估verdict预测和justification质量，全面审计LLM的事实核查能力边界。

**[GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](gui_explorer_autonomous.md)**

:   提出 GUI-explorer，一个无需训练的 GUI agent，通过自主探索收集功能感知的交互轨迹，并以无监督方式从状态转换三元组中挖掘 transition-aware 知识，在 SPA-Bench 和 AndroidWorld 上分别达到 53.7% 和 47.4% 的任务成功率。

**[GUICourse: From General Vision Language Model to Versatile GUI Agent](guicourse_from_general_vision_language_model_to_versatile_gui_agent.md)**

:   本文提出GUICourse——一套用于从通用视觉语言模型（VLM）训练多功能GUI代理的数据集系列（GUIEnv/GUIAct/GUIChat），通过两阶段训练流程先增强OCR和定位能力、再注入GUI知识，使得3.1B参数的小模型也能在网页和手机GUI导航任务上取得有效表现。

**[GuideBench: Benchmarking Domain-Oriented Guideline Following for LLM Agents](guidebench_guideline_following.md)**

:   提出 GuideBench 基准测试，系统评估 LLM 在领域导向指南遵循方面的能力，覆盖 7 个任务类别共 1272 个实例，从规则遵循、规则更新鲁棒性和人类偏好对齐三个维度评估 18 个 LLM，发现当前模型在复杂领域规则遵循上仍有较大提升空间。

**[Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement](gödel_agent_a_self-referential_agent_framework_for_recursive_self-improvement.md)**

:   提出 Gödel Agent，一种受 Gödel Machine 启发的自引用 Agent 框架，能通过 Python monkey patching 在运行时读取和修改自身代码（包括修改自身的修改逻辑），实现递归自改进，在 DROP/MGSM/MMLU/GPQA 上超越手工设计和元学习优化的 Agent。

**[iAgent: LLM Agent as a Shield between User and Recommender Systems](iagent_llm_agent_as_a_shield_between_user_and_recommender_systems.md)**

:   提出用户-Agent-平台三层范式，在用户和推荐系统之间插入 LLM Agent 作为保护层，通过指令解析、知识获取、重排序和动态用户画像实现个性化推荐，在四个数据集上平均提升 16.6%，同时有效缓解回音室效应和低活跃用户的不公平问题。

**[LegalAgentBench: Evaluating LLM Agents in Legal Domain](legalagentbench_evaluating_llm_agents_in_legal_domain.md)**

:   提出 LegalAgentBench，一个面向中国法律领域的 LLM Agent 综合评测基准，包含 17 个真实语料库、37 个工具和 300 个覆盖多跳推理与写作的任务，通过关键词匹配和过程进度率实现细粒度评估。

**[Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models](llm_agent_image_classification.md)**

:   提出 Conditional Concept Bottleneck Models (CoCoBMs) 和 LLM-driven Concept Agent 框架，通过类别条件化的概念评分机制和基于环境反馈的动态概念库优化，在 6 个数据集上提升分类准确率 6% 的同时将可解释性提升约 30%。

**[LocAgent: Graph-Guided LLM Agents for Code Localization](locagent_graph-guided_llm_agents_for_code_localization.md)**

:   LocAgent 将代码库解析为有向异构图（含 contain/import/invoke/inherit 四种关系），并设计统一工具（SearchEntity/TraverseGraph/RetrieveEntity）引导 LLM Agent 进行多跳推理，实现高精度代码定位，在文件级达到 92.7% 准确率，同时通过微调开源模型将成本降低 86%。

**[Adaptive Tool Use in Large Language Models with Meta-Cognition Trigger](meco_metacognition_tool_use.md)**

:   提出 MeCo（Meta-Cognition Trigger），通过表示工程从 LLM 内部提取"元认知信号"——模型对自身能力的自我评估——来自适应决定是否需要调用外部工具，无需微调且计算开销极小，在多个骨干模型和基准上显著改善工具使用决策的准确性。

**[METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)**

:   提出 METAL，一个基于 VLM 的多智能体框架，将图表生成任务（chart-to-code）分解为生成、视觉评审、代码评审和修订四个专门化智能体的迭代协作，在 ChartMIMIC 基准上比现有最佳方法提升 5.2% F1，并展现了测试时缩放（test-time scaling）现象。

**[Unveiling Privacy Risks in LLM Agent Memory](mextra_agent_memory_privacy.md)**

:   本文系统研究了 LLM Agent 记忆模块的隐私风险，提出 MEXTRA 黑盒记忆提取攻击，通过精心设计的定位-对齐攻击 prompt 和自动化多样 prompt 生成方法，在医疗和网购两种 Agent 上成功提取大量私人查询记录。

**[A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems](multi_agent_dialect_bias_privacy_qa.md)**

:   构建 Dialect Agent（方言翻译+审查）与 Privacy Policy Agent（领域回答）的双 Agent 迭代协作框架，通过注入方言语言学知识的提示工程，在无需重训练的前提下同时提升隐私政策 QA 的整体准确率和跨方言公平性。

**[MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents](multiagentbench_evaluating_the_collaboration_and_competition_of_llm_agents.md)**

:   提出 MultiAgentBench 基准测试和 MARBLE 框架，系统评估 LLM 多智能体系统在协作与竞争场景中的表现，包含 6 种交互场景（研究、Minecraft、数据库、编程、讨价还价、狼人杀），引入基于里程碑的 KPI 指标和协调评分，发现 gpt-4o-mini 整体任务分最高、图结构协调协议在研究场景中表现最佳、认知规划可提升里程碑达成率 3%。

**[Multiple LLM Agents Debate for Equitable Cultural Alignment](multiple_llm_agents_debate_for_equitable.md)**

:   提出 Multi-Agent Debate 框架，让两个 LLM agent 围绕文化场景进行辩论并由 judge LLM 仲裁，在 NormAd-eti 基准上显著提升文化适应准确率和跨文化群体公平性，使 7-9B 小模型达到 27B 模型的性能水平。

**[NexusSum: Hierarchical LLM Agents for Long-Form Narrative Summarization](nexussum_narrative_summarization.md)**

:   提出 NexusSum，一个三阶段多Agent LLM框架（对话转描述→层次摘要→迭代压缩），无需微调即可处理书籍/电影/电视剧等长叙事文本的摘要生成，在 BookSum 上 BERTScore 提升达 30%。

**[OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents](os-kairos_adaptive_interaction_for_mllm-powered_gui_agents.md)**

:   提出 OS-Kairos，通过协作探测框架标注每步置信度分数并微调进基座模型，使 GUI Agent 能在每步预测置信度、自主决定执行或请求人类干预，在复杂场景下任务成功率 (TSR) 从 OS-Atlas-Pro-7B 的 14.29% 提升到 88.20%，在 AITZ 和 Meta-GUI 基准上也有 24~87% 的绝对提升。

**[OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use](os_agents_a_survey_on_mllm-based_agents_for_general_computing_devices_use.md)**

:   全面综述了基于多模态大语言模型的操作系统 Agent（OS Agents），系统梳理了其基础概念（环境/观察/动作空间）、核心能力（理解/规划/动作落地）、构建方法（基础模型+Agent框架）和评估体系，涵盖 30+ 基础模型和 20+ Agent 框架的分类对比。

**[OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use](os_agents_survey_mllm.md)**

:   系统综述了基于多模态大语言模型（MLLM）的操作系统智能体（OS Agents），从基本概念（环境/观测/动作空间）、核心能力（理解/规划/定位）、构建方法（基础模型+智能体框架）到评估基准全面梳理，揭示了该领域从虚拟助手到通用计算设备自动化的演进路径。

**[OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis](os_genesis_gui_agent_trajectory.md)**

:   提出 OS-Genesis，一种交互驱动的 GUI Agent 轨迹合成 pipeline，通过先让 agent 在环境中探索交互再反向推导任务（Reverse Task Synthesis），结合轨迹奖励模型 (TRM) 过滤质量，生成高质量多样化的训练轨迹，在 AndroidWorld 上性能接近翻倍。

**[PaSa: An LLM Agent for Comprehensive Academic Paper Search](pasa_an_llm_agent_for_comprehensive_academic_paper_search.md)**

:   PaSa 是一个基于 LLM 的学术论文搜索智能体，通过自主调用搜索工具、阅读论文和导航引用网络来实现全面准确的学术文献检索，经 RL 训练后在真实场景中大幅超越 Google Scholar 和 GPT-4o。

**[Play2Prompt: Zero-shot Tool Instruction Optimization for LLM Agents via Tool Play](play2prompt_zero-shot_tool_instruction_optimization_for_llm_agents_via_tool_play.md)**

:   提出 Play2Prompt，通过让 LLM 自主"玩"工具（试探输入输出行为）来零样本地生成工具使用示例和优化工具文档，无需任何标注数据即可显著提升 LLM agent 的工具调用能力。

**[R2D2: Remembering, Replaying and Dynamic Decision Making with a Reflective Agentic Memory](r2d2_reflective_agentic_memory.md)**

:   R2D2 提出了一个结合 Remember（经验回放缓冲区 + A* 搜索导航）和 Reflect（错误反思 + 反思记忆存储）两范式的 Web Agent 框架，将 Web 导航从 Unknown MDP 转化为 Known MDP，在 WebArena 上导航错误减少 50%，任务完成率提升 3 倍，超越 SOTA 17%。

**[REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)**

:   本文提出 REPRO-Bench，一个包含 112 个社会科学论文实例的基准，用于评估 AI Agent 自动化评估论文可重复性的能力；现有最佳 Agent 准确率仅 21.4%（低于随机猜测的 25%），作者进一步开发的 REPRO-Agent 将准确率提升至 36.6%（71% 相对提升）。

**[REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_social_science_.md)**

:   提出 REPRO-Bench，包含 112 个社会科学论文的可复现性评估任务，发现现有 AI Agent（最高准确率仅 21.4%）远不足以自动化该流程，并据此开发 REPRO-Agent 将准确率提升至 36.6%。

**[Select, Read, and Write: A Multi-Agent Framework of Full-Text-based Related Work Generation](select_read_and_write_a_multi-agent_framework_of_full-text-based_related_work_ge.md)**

:   提出 Select-Read-Write 三 Agent 协同框架，通过图感知的阅读顺序决策和共享工作记忆机制，实现基于论文全文（而非摘要）的 Related Work 自动生成，在 Llama3-8B / Claude-3-Haiku / GPT-4o 三个基座模型上均取得一致提升，Citation Graph 策略效果最优。

**[Self-Taught Agentic Long-Context Understanding](self_taught_agentic_long_ctx.md)**

:   提出 AgenticLU 框架，通过 Chain-of-Clarifications (CoC) 工作流让 LLM 自主生成澄清问题并检索相关上下文，再通过 SFT+DPO 两阶段微调将树搜索路径蒸馏到模型中，使 8B 模型在 128K 长上下文 QA 任务上大幅超越基线。

**[SMART: Self-Aware Agent for Tool Overuse Mitigation](smart_self-aware_agent_for_tool_overuse_mitigation.md)**

:   揭示 LLM Agent 中的"工具过度使用"现象（≥30% 的工具调用是不必要的），提出 SMART 元认知范式，通过在推理步骤中显式标注"知识驱动 vs 工具依赖"来训练 Agent 的自感知能力，7B 模型匹配 GPT-4o 水平。

**["sudo rm -rf agentic_security" | SUDO: Screen-based Universal Detox2tox Offense](sudo_rm_-rf_agentic_security.md)**

:   提出SUDO攻击框架，通过Detox2tox三阶段流水线将恶意请求伪装为无害指令再恢复攻击载荷，配合基于检查清单反馈的动态迭代优化，系统性攻破Claude CUA、MANUS等计算机使用Agent的安全防护，最高达41.33%攻击成功率。

**[SynWorld: Virtual Scenario Synthesis for Agentic Action Knowledge Refinement](synworld_agentic_action_knowledge.md)**

:   SynWorld 提出让 Agent 在合成的虚拟场景中通过蒙特卡洛树搜索（MCTS）来探索和优化动作知识（工具描述和工作流），使 Agent 能够自主适应新环境的工具使用，在 ToolBench 上比 ReAct 基线提升约 9 个百分点。

**[Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning](table_critic_multi_agent.md)**

:   提出 Table-Critic 多智能体框架，通过 Judge-Critic-Refiner-Curator 四个专门化 Agent 的协作批评与迭代精化，配合自进化模板树累积批评知识，在 WikiTQ 和 TabFact 上分别实现 73.7% 和 91.7% 的准确率，大幅超越现有方法。

**[The Behavior Gap: Evaluating Zero-shot LLM Agents in Complex Task-Oriented Dialogs](the_behavior_gap_evaluating_zero-shot_llm_agents_in_complex_task-oriented_dialog.md)**

:   提出综合评估框架量化 LLM agent 与人类专家在任务导向对话中的"行为差距"，从 dialog acts、工具使用、知识利用三个维度系统诊断行为偏差，发现行为差距与任务复杂度高度相关（$r=0.963$），通过行为注入缩小差距可平均提升 24.3% 性能。

**[Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models](theorem-of-thought_a_multi-agent_framework_for_abductive_deductive_and_inductive.md)**

:   提出 Theorem-of-Thought (ToTh) 框架，通过三个并行智能体分别模拟溯因、演绎和归纳推理，将推理轨迹构建为形式推理图并利用 NLI 校准的贝叶斯置信传播选出最连贯推理链，在符号和数值推理上一致优于 CoT、Self-Consistency 和 CoT-Decoding。

**[ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](toolhop_multi_hop_tool_use.md)**

:   提出 ToolHop——一个包含 995 个多跳查询和 3912 个本地可执行工具的基准数据集，通过"查询驱动"的数据构建方式（先有查询再造工具）确保工具间有真实依赖关系和可验证答案，评测 14 个 LLM 发现最强的 GPT-4o 准确率仅 49%，揭示了不同模型家族在工具使用上的显著策略差异。

**[ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](toolhop_multi_hop_tool_use_benchmark.md)**

:   提出 ToolHop——一个包含 995 个多跳查询和 3912 个本地可执行工具的基准数据集，通过"查询驱动"的数据构建方式（先有查询再造工具）确保工具间有真实依赖关系和可验证答案，评测 14 个 LLM 发现最强的 GPT-4o 准确率仅 49%，揭示了不同模型家族在工具使用上的显著策略差异。
