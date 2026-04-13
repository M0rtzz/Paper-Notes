---
title: >-
  NeurIPS2025 LLM Agent方向 52篇论文解读
description: >-
  52篇NeurIPS2025 LLM Agent方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**🧠 NeurIPS2025** · 共 **52** 篇

**[A-Mem Agentic Memory For Llm Agents](a-mem_agentic_memory_for_llm_agents.md)**

:   提出 A-Mem，一种受 Zettelkasten 启发的 LLM Agent 智能记忆系统，每条记忆自动生成结构化笔记（关键词/标签/上下文描述），动态建立记忆间链接，并在新记忆加入时触发旧记忆的演化更新，在 LoCoMo 长对话 QA 上显著超越 MemGPT 等基线。

**[A Differentiable Model Of Supply-Chain Shocks](a_differentiable_model_of_supply-chain_shocks.md)**

:   本文用 JAX 实现了一个可微分的供应链 Agent-Based Model（ABM），通过 GPU 并行化和自动微分实现了比传统无梯度方法快 3 个数量级的贝叶斯参数校准，为大规模供应网络建模打开了可能性。

**[Adaptive Cooperative Transmission Design for URLLC via Deep Reinforcement Learning](adaptive_cooperative_transmission_design_for_ultra-reliable_low-latency_communic.md)**

:   针对两跳协作中继通信中的URLLC难题，提出DRL-CoLA算法：将每跳传输参数配置建模为MDP，用双agent DQN在仅观测本地CSI和ARQ反馈下学习分布式时延感知传输策略，接近全局最优可靠性。

**[Adaptive Coopetition Leveraging Coarse Verifier Signals For Resilient Multi-Agen](adaptive_coopetition_leveraging_coarse_verifier_signals_for_resilient_multi-agen.md)**

:   提出 Adaptive Coopetition (AdCo) 框架，利用 UCB 多臂老虎机策略和粗粒度验证器信号，使多个 LLM 智能体在推理过程中自适应地切换协作与竞争模式，在数学推理基准上实现 20% 的相对提升。

**[Agentauditor Humanlevel Safety And Security Evaluation For L](agentauditor_humanlevel_safety_and_security_evaluation_for_l.md)**

:   提出 AgentAuditor，一个通用的无训练记忆增强推理框架，使 LLM 评估者能模拟人类专家评估 agent 的安全与安全性——通过自适应提取结构化语义特征并生成CoT推理轨迹构建经验记忆，多阶段上下文感知 RAG 检索相关经验指导新案例评估，在自建的 ASSEBench（2293条记录×15类風险×29场景）上达到人类水平准确率。

**[Agentchangebench A Multi-Dimensional Evaluation Framework For Goal-Shift Robustn](agentchangebench_a_multi-dimensional_evaluation_framework_for_goal-shift_robustn.md)**

:   AgentChangeBench 是首个系统评估 LLM agent 在对话中途目标切换时适应能力的 benchmark：315 基础任务 × 9 变体 = 2835 序列，覆盖 3 个企业领域（银行/零售/航空）和 5 种 user persona，引入 GSRT（目标切换恢复时间）等 4 个互补指标，揭示高 pass@k 掩盖的效率和鲁棒性差距——如 GPT-4o 航空恢复率 92.2% 但零售冗余率达 89.1%。

**[Agentdam Privacy Leakage Evaluation For Autonomous Web Agent](agentdam_privacy_leakage_evaluation_for_autonomous_web_agent.md)**

:   提出 AgentDAM，首个在真实 Web 环境中端到端评估 AI Agent 数据最小化能力的基准，包含 246 个跨 Reddit/GitLab/Shopping 的任务，发现 GPT-4o 等主流模型在无缓解措施时隐私泄露率高达 36-46%，而 CoT 隐私提示可将泄露率降至 6-8%。

**[Agentic Nl2Sql To Reduce Computational Costs](agentic_nl2sql_to_reduce_computational_costs.md)**

:   提出 Datalake Agent，一个基于交互循环的 agentic NL2SQL 系统，通过分层的信息获取策略（GetDBDescription -> GetTables -> GetColumns -> DBQueryFinalSQL）让 LLM 按需请求数据库 schema 信息而非一次性接收全部，在 319 张表的场景下将 token 使用量减少 87%、成本降低 8 倍，同时在复杂查询上保持更好的性能。

**[Agentic Plan Caching Test-Time Memory For Fast And Cost-Efficient Llm Agents](agentic_plan_caching_test-time_memory_for_fast_and_cost-efficient_llm_agents.md)**

:   提出 Agentic Plan Caching (APC)——从 agent 执行日志中提取结构化计划模板，通过关键词匹配缓存命中后用小模型适配复用，平均降低 50.31% 成本和 27.28% 延迟，同时保持 96.61% 的最优准确率。

**[Agentmisalignment Measuring The Propensity For Misaligned Behaviour In Llm-Based](agentmisalignment_measuring_the_propensity_for_misaligned_behaviour_in_llm-based.md)**

:   提出 AgentMisalignment 基准套件，包含 9 个现实场景评估任务，测量 LLM Agent 在非恶意指令下 **自发偏离** 部署者意图的倾向（而非能力），发现更强的模型倾向于更高的错误对齐，且人格提示（persona prompt）有时比模型选择本身对错误对齐行为的影响更大。

**[Agenttts Large Language Model Agent For Testtime Computeopti](agenttts_large_language_model_agent_for_testtime_computeopti.md)**

:   提出 AgentTTS，一个用 LLM agent 自动搜索多阶段复杂任务中**测试时计算最优缩放策略**（模型选择+预算分配）的框架，通过迭代反馈驱动的交互显著提升搜索效率和性能。

**[Are Large Language Models Sensitive To The Motives Behind Communication](are_large_language_models_sensitive_to_the_motives_behind_communication.md)**

:   系统评估 LLM 对通信动机的敏感性（motivational vigilance）——在控制实验中 LLM 能像人类一样折扣有偏见信源的建议（与理性模型相关系数 r>0.78），但在真实场景（YouTube 赞助广告）中表现大幅下降（r<0.2），通过简单的 prompt steering 可部分恢复。

**[Attractive Metadata Attack Inducing Llm Agents To Invoke Malicious Tools](attractive_metadata_attack_inducing_llm_agents_to_invoke_malicious_tools.md)**

:   AMA（Attractive Metadata Attack）证明仅通过精心设计恶意工具的元数据（名称、描述、参数模式），不需要提示注入或模型内部访问，就能诱导 LLM Agent 以 81-95% 的成功率调用攻击者工具并泄露隐私，同时几乎不影响原始任务完成（98%+），且现有防御（审计器、提示重写）效果有限。

**[Automated Composition Of Agents A Knapsack Approach For Agentic Component Select](automated_composition_of_agents_a_knapsack_approach_for_agentic_component_select.md)**

:   将 Agent 组件选择问题形式化为在线背包问题，提出 Composer Agent 框架：通过沙盒实测（而非静态语义检索）评估组件真实能力，结合 ZCL 在线算法在预算约束下动态选取最优组件组合，单 Agent 工具选择成功率提升最高 31.6%，多 Agent 子代理选择成功率从 37% 跃升至 87%。

**[Automated Multi-Agent Workflows For Rtl Design](automated_multi-agent_workflows_for_rtl_design.md)**

:   VeriMaAS 是一个多智能体框架，通过将 HDL 形式化验证反馈（Yosys + OpenSTA）集成到工作流自动生成过程中，自适应地为 RTL 代码生成任务选择推理算子（I/O → CoT → ReAct → SelfRefine → Debate），以仅数百个训练样本实现比微调基线高 5-7% 的 pass@k 性能。

**[Benchmarking Agentic Systems In Automated Scientific Information Extraction With](benchmarking_agentic_systems_in_automated_scientific_information_extraction_with.md)**

:   构建 ChemX——10 个由领域专家手工标注和验证的多模态化学数据提取基准数据集，涵盖纳米材料和小分子两大领域，系统评估了 ChatGPT Agent、SLM-Matrix、FutureHouse、nanoMINER 等 SOTA Agent 系统以及 GPT-5/GPT-5 Thinking 等前沿 LLM；提出的单 Agent 方法通过结构化文档预处理（marker-pdf → Markdown → LLM 提取）在纳米酶数据集上达到 F1=0.61，超越所有通用多 Agent 系统，同时揭示了化学信息提取仍存在 SMILES 解析失败、术语歧义等系统性挑战。

**[Btlui Blinkthinklink Reasoning Model For Gui Agent](btlui_blinkthinklink_reasoning_model_for_gui_agent.md)**

:   提出"Blink-Think-Link"（BTL）脑启发框架模拟人类与GUI交互的认知过程——分解为Blink（快速注意力检测，类似眼跳）、Think（高级推理决策，类似认知规划）、Link（生成可执行命令，类似动作选择）三个生物合理阶段，配合自动化Blink数据标注和首个基于规则的过程+结果复合奖励机制，BTL-UI在静态GUI理解和动态交互任务上均达competitive性能。

**[Cam A Constructivist View Of Agentic Memory For Llm-Based Reading Comprehension](cam_a_constructivist_view_of_agentic_memory_for_llm-based_reading_comprehension.md)**

:   受皮亚杰建构主义理论启发，提出CAM——一种具有结构性（层次化schema）、灵活性（重叠聚类的同化）和动态性（增量适应）三大特征的智能体记忆系统，在6个长文本阅读理解任务上全面超越RAPTOR、GraphRAG等基线。

**[Contextagent Context-Aware Proactive Llm Agents With Open-World Sensory Percepti](contextagent_context-aware_proactive_llm_agents_with_open-world_sensory_percepti.md)**

:   提出 ContextAgent，首个利用可穿戴设备多模态感知（视频+音频+通知）来理解用户意图并主动提供工具增强服务的 LLM Agent 框架，同时构建了包含 1000 个样本的 ContextAgentBench 基准，在主动预测准确率和工具调用上分别提升 8.5% 和 6.0%。

**[Core Full-Path Evaluation Of Llm Agents Beyond Final State](core_full-path_evaluation_of_llm_agents_beyond_final_state.md)**

:   提出CORE框架：用确定有限自动机（DFA）编码Agent任务的合法工具调用路径，引入5个互补指标（路径正确性、顺序正确性、前缀危险性、有害调用率、效率）从全路径而非仅终态评估Agent行为，揭示了传统终态评估中不可见的安全和效率差异。

**[Crucible Quantifying The Potential Of Control Algorithms Through Llm Agents](crucible_quantifying_the_potential_of_control_algorithms_through_llm_agents.md)**

:   首次提出"调优潜能"（Tuning Potential）概念并给出形式化度量，通过 LLM Agent 模拟不同能力水平的开发者对控制算法进行参数调优和逻辑级改进，在 ABR 任务上相比贝叶斯优化提升 44.1%，CartPole 上 Bang-bang 从 34→500 达到 DQN 水平。

**[Debate Or Vote Which Yields Better Decisions In Multi-Agent Large Language Model](debate_or_vote_which_yields_better_decisions_in_multi-agent_large_language_model.md)**

:   通过理论和实验证明，多智能体辩论（MAD）的性能提升主要来自多数投票（ensembling）而非辩论本身——辩论过程构成 martingale（期望不变），即辩论不系统性地提升正确率，并基于此理论提出通过偏向正确信号来改进 MAD。

**[Deep Video Discovery Agentic Search With Tool Use For Longfo](deep_video_discovery_agentic_search_with_tool_use_for_longfo.md)**

:   提出 DVD（Deep Video Discovery）agent，将长视频理解建模为多步信息搜索问题：先将长视频构建为多粒度结构化数据库（全局摘要 + clip 级字幕嵌入 + 帧级像素），再提供三种搜索工具（Global Browse / Clip Search / Frame Inspect），由 reasoning LLM 通过 observe-reason-act 循环自主编排搜索轨迹，在 LVBench 达 74.2%（超先前 SOTA MR.Video 13.4 pp），加字幕 76.0%。

**[Defenderbench A Toolkit For Evaluating Language Agents In Cybersecurity Environm](defenderbench_a_toolkit_for_evaluating_language_agents_in_cybersecurity_environm.md)**

:   提出 DefenderBench，一个开源模块化工具包，用于在攻防和知识理解三类网络安全任务上系统评估 LLM Agent 的能力，覆盖网络入侵模拟、恶意内容检测、代码漏洞检测/修复、CTI 知识问答五大场景，基准测试显示 Claude-3.7-sonnet 综合最强（81.65 分）。

**[Distilling Llm Agent Into Small Models With Retrieval And Co](distilling_llm_agent_into_small_models_with_retrieval_and_co.md)**

:   提出 Agent Distillation 框架，将 LLM agent 的完整 reason-act-observe 交互行为（而非静态 CoT）蒸馏到 0.5B-7B 小模型中，配合 first-thought prefix 提升教师轨迹质量和 self-consistent action generation 提升推理鲁棒性，使小模型达到比其大 2-4× 的 CoT 蒸馏模型的性能。

**[Drift Dynamic Rulebased Defense With Injection Isolation For](drift_dynamic_rulebased_defense_with_injection_isolation_for.md)**

:   提出 DRIFT 系统级 Agent 安全框架，通过 Secure Planner（预规划函数轨迹+参数检查表）、Dynamic Validator（基于 Read/Write/Execute 权限的动态策略更新）和 Injection Isolator（从 memory stream 中检测并屏蔽注入指令）三层防御，在 AgentDojo 上将 ASR 从 30.7% 降至 1.3%，同时比 CaMeL 提升 20.1% utility。

**[Enhancing Demand-Oriented Regionalization With Agentic Ai And Local Heterogeneou](enhancing_demand-oriented_regionalization_with_agentic_ai_and_local_heterogeneou.md)**

:   本文提出一个基于 Agentic AI 的规划支持系统，通过 LLM 智能体引导非技术用户进行数据驱动的需求导向区域化（demand-oriented regionalization），核心算法为 RepSC-SOM（带代表性初始化的空间约束自组织映射），支持人机协作迭代优化区域划分，用于灾害风险管理和气候适应规划。

**[Eu-Agent-Bench Measuring Illegal Behavior Of Llm Agents Under Eu Law](eu-agent-bench_measuring_illegal_behavior_of_llm_agents_under_eu_law.md)**

:   提出 EU-Agent-Bench，首个基于欧盟法律框架的可验证智能体基准，通过 600 个良性用户请求测试 LLM 智能体的工具调用是否违反欧盟法规，发现即使最佳模型（Gemini 2.5 Flash）的合法率也仅约 55%，揭示了当前对齐技术与法律可靠性之间的巨大鸿沟。

**[Generative Ai Agents For Controllable And Protected Content Creation](generative_ai_agents_for_controllable_and_protected_content_creation.md)**

:   提出一个多智能体生成框架，通过 Director/Planner、Generator、Reviewer、Integration 和 Protection 五个专业化智能体的协作，结合人在环反馈，统一解决生成内容的可控性和版权保护问题。

**[Ground-Compose-Reinforce Grounding Language In Agentic Behaviours Using Limited ](ground-compose-reinforce_grounding_language_in_agentic_behaviours_using_limited_.md)**

:   提出 Ground-Compose-Reinforce (GCR)，一个端到端的神经符号框架，通过少量标注轨迹（仅350条）学习原子命题的接地语义（Ground），将其通过 Reward Machine 组合成复杂任务规范（Compose），然后用自生成的稠密奖励训练 RL 智能体（Reinforce），无需手工奖励函数即可引出分布外的复杂行为。

**[Groupingroup Policy Optimization For Llm Agent Training](groupingroup_policy_optimization_for_llm_agent_training.md)**

:   GiGPO 通过在 GRPO 的 episode 级分组内嵌套 step 级分组（利用跨轨迹的重复环境状态作为 anchor state），实现了无需额外 rollout 和 critic 模型的细粒度 credit assignment，在 ALFWorld 上比 GRPO 提升 >12%，WebShop 上提升 >9%。

**[Hogwild Inference Parallel Llm Generation Via Concurrent Attention](hogwild_inference_parallel_llm_generation_via_concurrent_attention.md)**

:   提出 Hogwild! Inference——一种无需预定义协作框架的并行 LLM 推理协议，多个 LLM 实例通过共享的并发 KV 缓存实时同步，利用 RoPE 位置编码避免重计算，在数学推理和编程任务上以更少的串行步骤达到更高精度。

**[Its Lit Reliability-Optimized Llms With Inspectable Tools](its_lit_reliability-optimized_llms_with_inspectable_tools.md)**

:   通过为每个外部工具定义可靠性/可调试性成本函数，引导 LLM 在多候选方案中选择成本最低（最透明可审计）的工具调用路径，在 61/65 测试场景中提升可解释性的同时保持甚至提升任务准确率。

**[Lc-Opt Benchmarking Reinforcement Learning And Agentic Ai For End-To-End Liquid ](lc-opt_benchmarking_reinforcement_learning_and_agentic_ai_for_end-to-end_liquid_.md)**

:   提出 LC-Opt，一个基于 Oak Ridge 国家实验室 Frontier 超级计算机冷却系统高保真数字孪生的液冷基准环境，支持强化学习控制策略的端到端液冷优化，涵盖集中式/分散式多智能体RL、策略蒸馏为可解释决策树、以及 LLM 驱动的智能体网格架构。

**[Lessons Learned A Multi-Agent Framework For Code Llms To Learn And Improve](lessons_learned_a_multi-agent_framework_for_code_llms_to_learn_and_improve.md)**

:   提出 LessonL 框架，使多个小 LLM 智能体通过相互学习的"课程"(lesson)对成功和失败案例进行反思，协同优化代码性能，3 个 7B-14B 模型组合达到 GPT-4o 甚至接近 o3 的代码优化效果。

**[Llm Agent Communication Protocol Lacp Requires Urgent Standardization A Telecom-](llm_agent_communication_protocol_lacp_requires_urgent_standardization_a_telecom-.md)**

:   这篇 position paper 指出当前 LLM Agent 通信的碎片化生态类似早期网络的"协议战争"，提出受电信标准化启发的三层协议 LACP（语义层、事务层、传输层），强调安全内建、事务完整性和语义互操作性对多智能体系统至关重要。

**[Llm Agents For Knowledge Discovery In Atomic Layer Processing](llm_agents_for_knowledge_discovery_in_atomic_layer_processing.md)**

:   通过让 LLM Agent 控制模拟化学反应器（黑盒函数），证明 Agent 能在无先验知识下通过试错探索、发现并总结未知化学系统的规则，揭示了 Agent 进行开放式科学发现的能力与局限。

**[Mat-Agent Adaptive Multi-Agent Training Optimization](mat-agent_adaptive_multi-agent_training_optimization.md)**

:   提出 MAT-Agent，一个由四个自主 agent（分别负责数据增强、优化器、学习率调度、损失函数）组成的多智能体框架，在训练过程中动态调整训练配置，用 DQN 学习策略以替代传统静态超参配置，在多标签图像分类任务上实现了 SOTA。

**[Mlrc-Bench Can Language Agents Solve Machine Learning Research Challenges](mlrc-bench_can_language_agents_solve_machine_learning_research_challenges.md)**

:   基于真实 ML 会议竞赛构建动态基准 MLRC-Bench，评估 LLM Agent 提出和实现新颖研究方法的能力，发现最强 Agent（Gemini）仅达人类顶级方案 9.3% 的相对改进，且提供 AI/人类想法并不能一致改善实现质量。

**[Orchestration Framework For Financial Agents From Algorithmic Trading To Agentic](orchestration_framework_for_financial_agents_from_algorithmic_trading_to_agentic.md)**

:   提出 FinAgent 编排框架，将传统算法交易系统的各组件映射为 AI 智能体（规划器、编排器、Alpha/风控/组合/回测/执行/审计/记忆智能体），使用 MCP 协议进行控制通信、A2A 协议进行智能体间通信，在股票和 BTC 交易任务上验证了可行性。

**[Panda Towards Generalist Video Anomaly Detection Via Agentic Ai Engineer](panda_towards_generalist_video_anomaly_detection_via_agentic_ai_engineer.md)**

:   提出 PANDA，一个基于 MLLM 的 Agentic AI 工程师框架，通过自适应场景感知策略规划、目标驱动启发式推理、工具增强自反思和链式记忆四大能力，实现无需训练和人工干预的通用视频异常检测。

**[Rd-Agent-Quant A Multi-Agent Framework For Data-Centric Factors And Model Joint ](rd-agent-quant_a_multi-agent_framework_for_data-centric_factors_and_model_joint_.md)**

:   提出 R&D-Agent(Q)，一个数据驱动的多智能体框架，通过五个协作模块（Specification、Synthesis、Implementation、Validation、Analysis）自动化量化策略的因子挖掘与模型创新联合优化，在真实股票市场上以不到 $10 的成本实现约 2× 于传统因子库的年化收益。

**[Shapecraft Llm Agents For Structured Textured And Interactive 3D Modeling](shapecraft_llm_agents_for_structured_textured_and_interactive_3d_modeling.md)**

:   提出基于图结构程序化形状表示（GPS）的多 Agent 框架 ShapeCraft，通过 Parser-Coder-Evaluator 三个 LLM Agent 协作，将自然语言分解为结构化子任务图，迭代生成可编辑、可动画的带纹理 3D 资产。

**[Suffixdecoding Extreme Speculative Decoding For Emerging Ai Applications](suffixdecoding_extreme_speculative_decoding_for_emerging_ai_applications.md)**

:   利用后缀树缓存长序列，通过自适应推测长度实现 5.3 倍加速，特别针对 Agent 场景中高度可预测的重复推理任务。

**[T1 A Tool-Oriented Conversational Dataset For Multi-Turn Agentic Planning](t1_a_tool-oriented_conversational_dataset_for_multi-turn_agentic_planning.md)**

:   构建 T1 数据集——13.5K 多轮对话覆盖 9 个领域（4 单领域 + 5 跨领域）、14 个工具，聚焦工具间依赖和动态重规划，并提出 T1-Agent（代码生成 + 缓存机制）作为基线系统；实验发现 SFT 后的 Llama 8B 在 Tool Call F1 上达 87.17%，超越未微调的 70B 模型，但仍落后于 GPT-5/o3 等闭源模型。

**[Tai3 Testing Agent Integrity In Interpreting User Intent](tai3_testing_agent_integrity_in_interpreting_user_intent.md)**

:   提出 TAI3，一个以 API 为中心的 LLM Agent 意图完整性压力测试框架，通过语义分区（Semantic Partitioning）将自然语言输入空间组织为结构化测试网格，再利用意图保持变异（Intent-Preserving Mutation）和策略记忆（Strategy Memory）高效暴露 Agent 在执行用户任务时的意图理解错误。

**[The Lighthouse Of Language Enhancing Llm Agents Via Critique-Guided Improvement](the_lighthouse_of_language_enhancing_llm_agents_via_critique-guided_improvement.md)**

:   提出 CGI（Critique-Guided Improvement）双角色框架，训练专门的 Critic 模型为 Actor Agent 提供结构化自然语言反馈（判别+修正建议），并通过迭代动作精炼让 Actor 学会利用这些反馈，在 WebShop/ScienceWorld/TextCraft 三个环境中平均得分 74.20%，超越 GPT-4o（45.46%）和 Iterative SFT（58.21%）。

**[Traj-Coa Patient Trajectory Modeling Via Chain-Of-Agents For Lung Cancer Risk Pr](traj-coa_patient_trajectory_modeling_via_chain-of-agents_for_lung_cancer_risk_pr.md)**

:   提出Traj-CoA多agent框架，通过chain-of-agents架构配合EHRMem长期记忆模块对长且噪声的纵向EHR进行时序推理，在零样本肺癌风险预测任务中（5年EHR数据，最高160k tokens）超越ML/DL/BERT/LLM等多类基线。

**[Trajagent An Llm-Agent Framework For Trajectory Modeling Via Large-And-Small Mod](trajagent_an_llm-agent_framework_for_trajectory_modeling_via_large-and-small_mod.md)**

:   首个 LLM 代理框架自动处理轨迹建模全流程，通过 UniEnv 统一接口和协作学习双层优化（LLM 推理 + 小模型训练），性能相比基线最高提升 69.91%。

**[Web-Shepherd Advancing Prms For Reinforcing Web Agents](web-shepherd_advancing_prms_for_reinforcing_web_agents.md)**

:   提出首个针对网页导航的过程奖励模型 Web-Shepherd，通过检查清单分解任务目标为可评估的子目标，3B/8B 模型在轨迹准确率上碾压 GPT-4o（85% vs 10%），同时成本仅为 1/10，使网页 Agent 的强化学习和推理时搜索变得实际可行。

**[What Ai Speaks For Your Community Polling Ai Agents For Public Opinion On Data C](what_ai_speaks_for_your_community_polling_ai_agents_for_public_opinion_on_data_c.md)**

:   提出基于LLM的AI agent民意调研框架，通过人口统计合成虚拟居民agent对数据中心项目进行大规模低成本民调，跨模型跨地区实验表明agent意见与真实民调在主题上高度一致。

**[Zero-Shot Large Language Model Agents For Fully Automated Radiotherapy Treatment](zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)**

:   提出一种基于 LLM Agent 的零样本 (zero-shot) 放射治疗自动计划工作流，LLM 直接与商业治疗计划系统 (Eclipse TPS) 交互，通过迭代提取剂量-体积直方图 (DVH) 和目标函数损失并推理约束调整策略，在 20 例头颈癌 IMRT 病例上实现了与临床手动计划相当甚至更优的剂量分布质量。
