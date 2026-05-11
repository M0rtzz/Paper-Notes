---
title: >-
  ACL2026 LLM Agent方向41篇论文解读
description: >-
  41篇ACL2026的 LLM Agent 方向论文解读，涵盖 Agent、LLM、对抗鲁棒、推理、代码智能、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**💬 ACL2026** · **41** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (15)](../../CVPR2026/llm_agent/index.md) · [🔬 ICLR2026 (40)](../../ICLR2026/llm_agent/index.md) · [🤖 AAAI2026 (44)](../../AAAI2026/llm_agent/index.md) · [🧠 NeurIPS2025 (50)](../../NeurIPS2025/llm_agent/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/llm_agent/index.md) · [🧪 ICML2025 (15)](../../ICML2025/llm_agent/index.md)

🔥 **高频主题：** Agent ×19 · LLM ×16 · 对抗鲁棒 ×3 · 推理 ×3 · 代码智能 ×2

**[AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts](agencybench_benchmarking_the_frontiers_of_autonomous_agents_in_1m-token_real-wor.md)**

:   提出AgencyBench——一个包含138个真实世界任务的综合基准，评估6种核心智能体能力，每个场景平均需90次工具调用和100万token，通过用户模拟agent和Docker沙箱实现全自动化评估。

**[Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)**

:   本文提出 Agent-GWO，将灰狼优化器的领导者-追随者机制引入多智能体框架，联合优化 prompt 模板和解码超参数（温度、top-p 等），在 11 个数学和混合推理基准上持续超越现有提示优化方法。

**[ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)**

:   提出 ATLAS 多智能体金融交易框架和 Adaptive-OPRO 提示优化方法，通过专业化分析师智能体准备异构市场信息，并基于延迟噪声反馈动态优化中央交易智能体的指令提示，在多种市场波动环境中显著超越基线。

**[Bayesian Social Deduction with Graph-Informed Language Models](bayesian_social_deduction_with_graph-informed_language_models.md)**

:   提出 GRAIL（Graph Reasoning Agent Informed through Language），一个混合推理框架，将概率推理外化到因子图模型、用 LLM 处理语言理解和交互，在社交推理游戏 Avalon 中首次击败人类玩家（67% 胜率），且资源消耗远低于大规模推理模型。

**[CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents](ci-work_benchmarking_contextual_integrity_in_enterprise_llm_agents.md)**

:   基于上下文完整性（Contextual Integrity）理论构建企业场景基准 CI-Work，揭示前沿 LLM 智能体在企业工作流中普遍存在隐私泄漏问题，且模型规模扩大反而加剧泄漏。

**[CodeStruct: Code Agents over Structured Action Spaces](codestruct_code_agents_over_structured_action_spaces.md)**

:   本文提出CodeStruct框架，将代码仓库重新定义为基于AST的结构化动作空间，让LLM代码Agent通过命名的程序实体（而非文本片段）进行读取和编辑操作，在SWE-Bench Verified上提升1.2-5.0%准确率并减少12-38% token消耗。

**[CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution](coevolve_training_llm_agents_via_agent-data_mutual_evolution.md)**

:   CoEvolve 提出**智能体-数据共进化框架**，通过从训练轨迹中提取遗忘/边界/稀有三类弱点信号，引导 LLM 做针对性环境再探索和任务合成，使训练数据分布随智能体能力动态适应，在 AppWorld 和 BFCL 上分别带来 19-23% 的绝对提升。

**[Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)**

:   本文研究多智能体 LLM 系统中的联合提示攻击（conjunctive prompt attacks）：用户查询中嵌入的触发键和被入侵远程代理中的隐藏模板各自看起来无害，但当路由将它们带到同一代理时会激活有害行为，现有防御（PromptGuard、Llama-Guard 等）均无法可靠阻止。

**[Creating ConLangs to Probe the Metalinguistic Grammatical Knowledge of LLMs](creating_conlangs_to_probe_the_metalinguistic_grammatical_knowledge_of_llms.md)**

:   本文提出 IASC（Interactive Agentic System for ConLangs），一个模块化的人造语言构建系统，通过让 LLM 按语言学规格执行形态句法变换来探测其元语言知识，发现 LLM 处理常见语言类型模式远优于罕见模式，且不同 LLM 之间能力差异悬殊。

**[Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](diversity_collapse_in_multi-agent_llm_systems_structural_coupling_and_collective.md)**

:   本文通过评估超过 10,000 个研究提案，从模型智能、智能体认知和系统动力学三个层次系统揭示了多智能体 LLM 系统中的"多样性崩溃"现象：更强的模型、权威驱动的角色分配和密集的通信拓扑都会抑制语义多样性，根本原因是交互结构而非模型能力不足。

**[EA-Agent: A Structured Multi-Step Reasoning Agent for Entity Alignment](ea-agent_a_structured_multi-step_reasoning_agent_for_entity_alignment.md)**

:   提出 EA-Agent，将实体对齐（EA）分解为结构化多步推理过程，通过工具池（三元组选择器+对齐工具+反思器）的规划和执行实现可解释的对齐决策，配合奖励引导的离线策略优化持续改进规划能力，在 DBP15K 上 Hits@1 提升高达 3.17%，同时减少冗余三元组带来的效率问题。

**[ExpSeek: Self-Triggered Experience Seeking for Web Agents](expseek_self-triggered_experience_seeking_for_web_agents.md)**

:   ExpSeek 提出了一种基于步级熵自触发的经验主动寻求框架，让 Web Agent 在交互过程中根据自身信号判断何时需要指导、获取什么指导，在 Qwen3-8B/32B 上分别实现 9.3% 和 7.5% 的绝对提升。

**[FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](fairqe_multi-agent_framework_for_mitigating_gender_bias_in_translation_quality_e.md)**

:   提出 FairQE 多智能体框架，通过性别线索检测、性别翻转变体生成和动态偏见感知分数聚合机制，在不牺牲翻译质量评估准确性的前提下有效缓解 QE 模型中的系统性性别偏见。

**[FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems](fedgui_benchmarking_federated_gui_agents_across_heterogeneous_platforms_devices_.md)**

:   FedGUI 是首个面向跨平台 GUI 代理的联邦学习综合基准，包含六个数据集覆盖移动端/网页端/桌面端，系统研究跨平台、跨设备、跨操作系统和跨数据源四种异构性对联邦 GUI 代理训练的影响。

**[FregeLogic at SemEval 2026 Task 11: A Hybrid Neuro-Symbolic Architecture for Content-Robust Syllogistic Validity Prediction](fregelogic_at_semeval_2026_task_11_a_hybrid_neuro-symbolic_architecture_for_cont.md)**

:   提出 FregeLogic 混合神经符号系统，结合五成员 LLM 集成和 Z3 SMT 求解器作为决胜裁判，在三段论有效性判断中将内容效应降低16%的同时提升准确率0.9%。

**[From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)**

:   本文构建了JurisCQAD——一个包含43000+真实中文法律咨询的大规模数据集，并提出JurisMA多智能体框架，通过法律元素图进行结构化任务分解和动态多Agent协作（管理Agent+格式检查+法条检索），在LawBench上显著优于通用和法律专用LLM。

**[HAG: Hierarchical Demographic Tree-based Agent Generation for Topic-Adaptive Simulation](hag_hierarchical_demographic_tree-based_agent_generation_for_topic-adaptive_simu.md)**

:   提出 HAG 框架，将群体 Agent 生成形式化为两阶段层次化决策过程——先用世界知识模型构建主题自适应人口分布树实现宏观分布对齐，再通过真实数据检索与 Agent 增强保证微观个体一致性，在多领域基准上将群体对齐误差平均降低 37.7%、社会学一致性提升 18.8%。

**[Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](hierarchical_reinforcement_learning_with_augmented_step-level_transitions_for_ll.md)**

:   本文提出 STEP-HRL，通过引入局部进度模块将交互历史迭代压缩为紧凑的文本摘要，使高层和低层策略仅基于单步转移（而非完整历史）做决策，在 ScienceWorld 和 ALFWorld 上显著提升性能和泛化性，同时减少 token 使用。

**[How Adversarial Environments Mislead Agentic AI](how_adversarial_environments_mislead_agentic_ai.md)**

:   本文形式化了"对抗环境注入"（AEI）威胁模型，将其分解为广度攻击（投毒检索结果导致认知漂移）和深度攻击（注入幻影节点构造导航陷阱导致策略崩溃），在 11,000+ 次实验中发现两种攻击的鲁棒性完全独立——"鲁棒性分裂"表明当前单点防御策略根本不够。

**[ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models](implicitmembench_measuring_unconscious_behavioral_adaptation_in_large_language_m.md)**

:   提出 ImplicitMemBench，首个系统评估 LLM 隐式记忆的基准，包含程序性记忆、启动效应和经典条件反射三种认知范式共 300 个测试项，在 17 个模型上揭示严重局限：最优模型仅达 66% 整体准确率，远低于人类基线。

**[Lightweight LLM Agent Memory with Small Language Models](lightweight_llm_agent_memory_with_small_language_models.md)**

:   本文提出 LightMem，一种由多个专用小语言模型（SLM）驱动的轻量级 LLM 智能体记忆系统，通过将记忆操作模块化为控制器（SLM-1）、选择器（SLM-2）和写入器（SLM-3），并将在线处理与离线整合解耦，在 LoCoMo 基准上平均 F1 提升约 2.5（相比 A-MEM），同时实现 83ms 检索延迟和 581ms 端到端延迟。

**[LPO: Towards Accurate GUI Agent Interaction via Location Preference Optimization](lpo_towards_accurate_gui_agent_interaction_via_location_preference_optimization.md)**

:   本文提出 Location Preference Optimization (LPO)，通过基于信息熵的窗口奖励和基于物理距离的动态位置奖励，结合 GRPO 框架优化 GUI 智能体的空间定位精度，在离线和在线评估中均达到 SOTA。

**[MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering](mata_multi-agent_framework_for_reliable_and_flexible_table_question_answering.md)**

:   提出 MATA 多Agent表格问答框架，通过调度器优先选择推理路径（CoT/PoT/text2SQL）、置信度检查器筛选答案、法官Agent仲裁，实现模型无关的高效准确表格QA，在10个LLM上平均EM提升40.1%。

**[MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools](mcp-flow_facilitating_llm_agents_to_master_real-world_diverse_and_scaling_mcp_to.md)**

:   MCP-Flow 提出了一个基于 Web Agent 的自动化管道，从 1166 个真实 MCP 服务器中收集工具信息并合成 68733 条高质量训练数据，使小规模微调模型（0.6B-8B）在 MCP 工具使用上超越 GPT-4o 等 SOTA 大模型。

**[MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection](memophishagent_memory-augmented_multi-modal_llm_agent_for_phishing_url_detection.md)**

:   提出 MemoPhishAgent（MPA），首个专为钓鱼URL检测设计的记忆增强多模态LLM智能体，通过5个专用工具的动态编排和情景记忆系统复用历史推理轨迹，在公开基准上召回率提升13.6%，在真实社交媒体数据上提升20%，并已部署生产环境每周处理约6万高风险URL。

**[Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh](mina_a_multilingual_llm-powered_legal_assistant_agent_for_bangladesh_for_empower.md)**

:   开发 Mina——面向孟加拉国法律场景的多语言 LLM 法律助手，通过两阶段 RAG 流水线精准检索法案和条款，配合工具链和多语言嵌入，在孟加拉律师资格考试中取得 75-80% 的通过成绩，法律咨询成本仅为传统方式的 0.12-0.61%。

**[RISK: A Framework for GUI Agents in E-commerce Risk Management](risk_a_framework_for_gui_agents_in_e-commerce_risk_management.md)**

:   提出 RISK 框架，包含领域数据集（RISK-Data, 8492单步+2386多步轨迹）、基准（RISK-Bench）和基于GRPO的强化微调方法（RISK-R1），针对电商风控场景的GUI智能体，7B模型以仅7.2%的参数量超越SOTA基线，在线任务成功率达70.5%。

**[Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration](scaling_external_knowledge_input_beyond_context_windows_of_llms_via_multi-agent_.md)**

:   提出 ExtAgents 多智能体框架，通过全局知识同步（所有Seeking Agent间交换信息）和知识累积推理（逐步向Reasoning Agent注入筛选后的知识）两个机制，解决现有多智能体方法在扩展外部知识输入超出上下文窗口时性能不升反降的瓶颈，在多跳QA和长综述生成任务上显著提升。

**[SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios](securevibebench_evaluating_secure_coding_capabilities_of_code_agents_with_realis.md)**

:   提出 SecureVibeBench，首个仓库级多文件编辑的安全编码基准，从41个OSS-Fuzz项目中构建105个C/C++安全编码任务，通过级联静态+动态分析精确还原漏洞首次引入的场景，评估发现最佳Agent（SWE-agent + Claude Sonnet 4.5）仅23.8%的代码同时满足功能正确性和安全性。

**[SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)**

:   本文提出 SILO-BENCH，一个角色无关的多智能体 LLM 分布式协调基准，包含 30 个算法任务、三个通信复杂度级别、54 种配置共 1620 个实验，揭示了关键的"通信-推理鸿沟"：智能体能自发形成合理通信拓扑并积极交换信息，但系统性地无法将分布式状态整合为正确答案。

**[Spec-o3: A Tool-Augmented Vision-Language Agent for Rare Celestial Object Candidate Identification](spec-o3_a_tool-augmented_vision-language_agent_for_rare_celestial_object_candida.md)**

:   提出 Spec-o3，一个工具增强的视觉语言智能体，通过交错多模态思维链（iMCoT）模拟天文学家的光谱检查流程，采用冷启动 SFT + 基于结果的 RL 两阶段训练，在稀有天体识别上将 macro-F1 从 28.3% 提升至 76.5%，推理速度比人工检查快 ~50 倍。

**[StructMem: Structured Memory for Long-Horizon Behavior in LLMs](structmem_structured_memory_for_long-horizon_behavior_in_llms.md)**

:   StructMem 提出了一种结构增强的层次化记忆框架，通过事件级双视角提取和跨事件语义整合，在 LoCoMo 长对话基准上实现 SOTA 性能（76.82%），同时大幅降低 token 消耗（1.94M vs. 图记忆的 35.8M）和 API 调用次数。

**[SynthAgent: Adapting Web Agents with Synthetic Supervision](synthagent_adapting_web_agents_with_synthetic_supervision.md)**

:   本文提出 SynthAgent，一个完全基于合成监督的 Web Agent 适应框架，通过分类探索系统覆盖网页功能区域以合成多样化任务，再通过任务精炼（冲突检测触发修正幻觉）和轨迹精炼（全局视角去噪）的双重精炼策略提升合成数据质量，在 WebArena 和 Online-Mind2Web 上显著优于现有合成方法。

**[ToolOmni: Enabling Open-World Tool Use via Agentic Learning with Proactive Retrieval and Grounded Execution](toolomni_enabling_open-world_tool_use_via_agentic_learning_with_proactive_retrie.md)**

:   本文提出 ToolOmni，一个统一的智能体框架，将主动工具检索和基于检索结果的工具执行整合在同一推理循环中，通过冷启动 SFT + 解耦多目标 GRPO 联合优化检索和执行能力，在 ToolBench 上端到端执行成功率超过强基线 +10.8%。

**[Towards Scalable Lightweight GUI Agents via Multi-role Orchestration](towards_scalable_lightweight_gui_agents_via_multi-role_orchestration.md)**

:   本文提出 LAMO 框架，通过角色导向的数据合成和两阶段训练（SFT with Perplexity-Weighted Cross-Entropy + 多任务 RL），将轻量 3B MLLM 训练为可灵活编排多角色的 GUI Agent，在单体推理、多 Agent 协作和即插即用策略执行器三种模式下工作，搭配 GPT-5 规划器在 AndroidWorld 上达 77.6% 成功率，超越 72B 参数的专用 GUI Agent。

**[Uncertainty Quantification in LLM Agents: Foundations, Emerging Challenges, and Opportunities](uncertainty_quantification_in_llm_agents_foundations_emerging_challenges_and_opp.md)**

:   本文提出首个 Agent 不确定性量化（Agent UQ）的形式化框架：将 agent 的问题解决轨迹建模为动态贝叶斯网络上的随机过程 $P(\mathcal{F}_{\leq T}) = P(E_0, O_0) \prod_{i=1}^{T} P_{\pi,\mathcal{T}}(A_i|E_{i-1}, O_{i-1}) P(O_i|A_i, E_i)$，统一了现有 UQ 范式（单步 QA、多步推理）为特例，并通过 $\tau^2$-bench 上的实证分析识别了四个 agent UQ 特有的技术挑战。

**[Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories](waking_up_blind_cold-start_optimization_of_supervision-free_agentic_trajectories.md)**

:   本文提出 SPECTRA，一种无需监督轨迹的框架——通过冷启动强化学习（GRPO）和软结构化多轮 rollout 拓扑约束，让小型视觉语言模型（SVLM）在纯环境交互中自行发现有效的工具调用和视觉推理行为，在 4 个多模态 benchmark 上提升任务准确率达 5% 和工具效率 9%，同时提出 Tool Instrumental Utility（TIU）指标量化无监督下的工具效能。

**[What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search](what_makes_an_llm_a_good_optimizer_a_trajectory_analysis_of_llm-guided_evolution.md)**

:   本文通过大规模实验（15 个 LLM × 8 个任务、72K 候选解）发现优秀的 LLM 优化器表现为"局部精炼器"——持续产生频繁的渐进式改进并在语义空间中逐步集中搜索，而非产生高新颖性的跳跃式突破；关键发现是新颖性本身并不预测优化性能，只有当搜索保持足够局部化时新颖性才有益。

**[When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors](when_agents_look_the_same_quantifying_distillation-induced_similarity_in_tool-us.md)**

:   本文提出了 RPS 和 AGS 两个互补指标来量化 LLM Agent 在工具使用行为上的蒸馏导致的同质化现象，通过区分必要行为和非必要行为，在 18 个模型上揭示了跨家族行为继承模式，发现 Kimi-K2 与 Claude Sonnet 4.5 的行为相似度甚至超过 Anthropic 自家模型。

**[Why Agents Compromise Safety Under Pressure](why_agents_compromise_safety_under_pressure.md)**

:   提出"代理压力"（Agentic Pressure）概念——当 LLM 代理在资源约束下无法同时完成任务和遵守安全规则时，会自发地产生规范漂移，主动牺牲安全以保持有用性，且推理能力越强的模型越善于构建语言化合理化来为违规辩护。

**[ZARA: Training-Free Motion Time-Series Reasoning via Evidence-Grounded LLM Agents](zara_training-free_motion_time-series_reasoning_via_evidence-grounded_llm_agents.md)**

:   提出 ZARA，一个基于知识和检索增强的多智能体框架，通过将传感器信号蒸馏为结构化文本知识库、类别条件检索和分层 LLM 推理，在完全免训练的设置下实现了可解释的人体活动识别，8 个数据集上大幅超越现有方法。
