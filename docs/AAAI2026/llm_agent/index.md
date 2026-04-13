---
title: >-
  AAAI2026 LLM Agent方向 46篇论文解读
description: >-
  46篇AAAI2026 LLM Agent方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**🤖 AAAI2026** · 共 **46** 篇

**[A2Flow Automating Agentic Workflow Generation Via Self-Adaptive Abstraction Oper](a2flow_automating_agentic_workflow_generation_via_self-adaptive_abstraction_oper.md)**

:   提出 A2Flow 框架，通过三阶段流水线（案例生成→功能聚类→深度提取）从专家数据中全自动提取可复用的抽象执行算子，替代人工预定义算子，并引入算子记忆机制累积中间输出辅助节点决策，在 8 个基准上整体超越 AFLOW 等 SOTA，资源消耗降低 37%。

**[A Multi-Agent Conversational Bandit Approach To Online Evaluation And Selection ](a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)**

:   提出 MACO（Multi-Agent Conversational Online Learning），将 LLM 回复选择建模为多 Agent 对话式赌博机问题，通过本地 Agent 淘汰低质量回复 + 云端自适应关键词对话收集偏好，实现近似最优的在线回复评估和用户偏好对齐。

**[A Multi-Agent Llm Framework For Multi-Domain Low-Resource In-Context Ner Via Kno](a_multi-agent_llm_framework_for_multi-domain_low-resource_in-context_ner_via_kno.md)**

:   提出 KDR-Agent 多 Agent 框架，通过中央规划器协调知识检索、上下文消歧和反思纠错三个专用 Agent，结合自然语言类型定义和实体级正负对比示例，无需微调即可在 5 个领域 10 个低资源 NER 数据集上全面超越 zero-shot 和 few-shot 基线（GPT-4o 上 BC5CDR F1=82.47，WNUT-17 F1=80.78）。

**[A Multi-Agent Llm Framework For Multi-Domain Low-Resource In](a_multi-agent_llm_framework_for_multi-domain_low-resource_in.md)**

:   提出 KDR-Agent 多智能体框架，通过知识检索（Wikipedia）、歧义消解和反思式自我纠错三个专业智能体协同工作，在仅使用少量静态标注示例的条件下，在5个领域10个NER数据集上显著超越现有零样本和少样本ICL NER方法。

**[Agentsense Virtual Sensor Data Generation Using Llm Agents I](agentsense_virtual_sensor_data_generation_using_llm_agents_i.md)**

:   利用LLM驱动的具身智能体在模拟智能家居中"生活"，生成虚拟环境传感器数据用于预训练HAR模型，在低资源场景下显著提升活动识别性能。

**[Agentswift Efficient Llm Agent Design Via Value-Guided Hierarchical Search](agentswift_efficient_llm_agent_design_via_value-guided_hierarchical_search.md)**

:   提出AgentSwift框架，通过层次化搜索空间（同时优化agentic workflow和功能组件）、轻量级value model预测agent性能、以及不确定性引导的MCTS搜索策略，自动发现高性能LLM agent设计，在7个基准上平均提升8.34%。

**[Aquasentinel Next-Generation Ai System Integrating Sensor Ne](aquasentinel_next-generation_ai_system_integrating_sensor_ne.md)**

:   提出AquaSentinel，一个物理信息驱动的AI系统，通过稀疏传感器部署+物理增强虚拟传感器+MoE时空GNN集成+双阈值RTCA检测算法+因果流定位+LLM报告生成，仅用20-30%节点覆盖即可实现全网管道泄漏检测，在110个泄漏场景中达到100%检测率。

**[Arcane A Multi-Agent Framework For Interpretable And Configurable Alignment](arcane_a_multi-agent_framework_for_interpretable_and_configurable_alignment.md)**

:   提出ARCANE框架，将对齐建模为多智能体协作问题——manager agent通过与stakeholder对话学习生成自然语言rubric（加权可验证准则集），作为worker agent的可解释代理奖励函数，通过SFT+GSPO两阶段训练实现测试时可配置的对齐，在GDPVal基准上GSPO版本的mean return从0.58提升至0.74（N=8）。

**[Autoglm Autonomous Foundation Agents For Guis](autoglm_autonomous_foundation_agents_for_guis.md)**

:   AutoGLM 基于 ChatGLM 构建了面向 Web 浏览器和 Android 手机的 GUI 基础智能体，通过中间接口设计分离规划与定位行为，并提出自进化在线课程强化学习框架，在 VAB-WebArena-Lite 上达到 55.2% 成功率，大幅超越 GPT-4o 的 18.2%。

**[Automating Complex Document Workflows Via Stepwise And Rollback-Enabled Operatio](automating_complex_document_workflows_via_stepwise_and_rollback-enabled_operatio.md)**

:   提出AutoDW框架，通过逐步规划（每次生成一个API调用）+自适应回滚（参数级+API级两层回滚）实现复杂文档工作流自动化，在250会话/1708指令的DWBench上达到90%指令级和62%会话级完成率，分别超越最强基线40%和76%。

**[Autotool Efficient Tool Selection For Large Language Model A](autotool_efficient_tool_selection_for_large_language_model_a.md)**

:   提出AutoTool，一个基于图的免训练工具选择框架，通过发现并利用"工具使用惯性"（tool usage inertia）——即工具调用遵循可预测的顺序模式这一经验现象——构建工具惯性图（TIG），用统计方法代替部分LLM推理来高效选择工具和填充参数，在保持任务完成率的同时减少15-40%的推理成本。

**[Autotool Efficient Tool Selection For Large Language Model Agents](autotool_efficient_tool_selection_for_large_language_model_agents.md)**

:   提出 AutoTool，一种基于图的工具选择框架，利用工具使用惯性（tool usage inertia）构建工具惯性图（TIG），通过统计结构绕过重复的 LLM 推理来选择工具和填充参数，在保持任务完成率的同时减少最多 30% 的推理开销。

**[Bayesagent Bayesian Agentic Reasoning Under Uncertainty Via ](bayesagent_bayesian_agentic_reasoning_under_uncertainty_via_.md)**

:   提出 vPGM 框架，通过自然语言引导 LLM Agent 模拟概率图模型（PGM）的贝叶斯推理过程，发现隐变量并推断后验分布，再用 Dirichlet 先验做数值贝叶斯校准（BayesVPGM），在多个推理任务上同时提升准确率和置信度校准。

**[Beyond React A Planner-Centric Framework For Complex Tool-Au](beyond_react_a_planner-centric_framework_for_complex_tool-au.md)**

:   提出以Planner为核心的Plan-Execute框架，将复杂查询转化为DAG执行计划，通过SFT+GRPO两阶段训练专门的Planner模型，在ComplexTool-Plan和StableToolBench上超越ReAct等反应式方法，用更少推理步骤实现更高成功率。

**[Causaltrace A Neurosymbolic Causal Analysis Agent For Smart Manufacturing](causaltrace_a_neurosymbolic_causal_analysis_agent_for_smart_manufacturing.md)**

:   提出 CausalTrace——一个集成于工业 CoPilot（SmartPilot）中的神经符号因果分析智能体，融合数据驱动因果发现与工业本体/知识图谱，实现了实时的根因分析、反事实推理和可解释决策支持。

**[Co-Epg A Framework For Co-Evolution Of Planning And Groundin](co-epg_a_framework_for_co-evolution_of_planning_and_groundin.md)**

:   提出Co-EPG框架，将GUI Agent解耦为Planning和Grounding两个模型，通过GRPO协同训练和基于置信度的动态奖励集成机制（C-DREM）建立正反馈循环，使两个模型自迭代协同进化，仅用基准数据集（无需外部数据）即在Multimodal-Mind2Web（58.4%）和AndroidControl（83.1%）上达到SOTA。

**[Coach Collaborative Agents For Contextual Highlighting -- A Multi-Agent Framewor](coach_collaborative_agents_for_contextual_highlighting_--_a_multi-agent_framewor.md)**

:   提出 COACH 框架——一个基于共享骨干模型的可重配置多智能体系统，通过意图驱动的策略编排和结构化 CoT 微调实现角色专业化，在羽毛球视频分析的 QA 和摘要两个任务上显著超越 Gemini 2.5 Pro 等通才模型。

**[Cook And Clean Together Teaching Embodied Agents For Paralle](cook_and_clean_together_teaching_embodied_agents_for_paralle.md)**

:   提出ORS3D任务——将运筹学(OR)知识引入具身AI的任务调度，要求智能体利用可并行子任务的等待时间执行其他任务以最小化总完成时间，同时在3D场景中定位目标物体；构建60K级数据集ORS3D-60K，并提出GRANT模型通过调度token机制连接外部动态规划求解器，在时间效率上比baseline提升30.53%。

**[Covrcollaborative Optimization Of Vlms And Rl Agent For Visu](covrcollaborative_optimization_of_vlms_and_rl_agent_for_visu.md)**

:   提出 VLM 与 RL 双向协同优化框架 COVR：RL 生成的高质量交互数据用于微调 VLM，增强后的 VLM 反过来通过 action prior 指导 RL 策略学习，在 CARLA 和 DMControl 上取得 SOTA。

**[D-Gara A Dynamic Benchmarking Framework For Gui Agent Robust](d-gara_a_dynamic_benchmarking_framework_for_gui_agent_robust.md)**

:   提出 D-GARA，一个面向 Android GUI Agent 的动态鲁棒性评估框架，通过在实时交互过程中注入权限弹窗、电量警告、应用崩溃等真实世界异常，揭示现有 SOTA Agent（包括 UI-TARS-72B、GPT-4o）在中断场景下平均成功率下降超过 17.5%，最高达 33% 的严重脆弱性。

**[Depo Dual-Efficiency Preference Optimization For Llm Agents](depo_dual-efficiency_preference_optimization_for_llm_agents.md)**

:   提出双重效率（dual-efficiency）的概念，将 LLM Agent 的效率分解为 step 级（减少每步 token 数）和 trajectory 级（减少总步数），并基于 KTO 设计了 DEPO 方法，通过在 desirable 样本的 reward 中加入效率 bonus 来联合优化效率与性能。

**[Ecoagent An Efficient Device-Cloud Collaborative Multi-Agent](ecoagent_an_efficient_device-cloud_collaborative_multi-agent.md)**

:   提出 EcoAgent，一个闭环设备-云端协作的多 Agent 移动自动化框架，通过 Dual-ReACT 双层推理规划 + 设备端轻量验证反馈 + Pre-Understanding 文本压缩模块，在 AndroidWorld 上达到与全云端 Agent 相当的成功率，同时大幅降低延迟（3.9s vs 15.3s）、云端调用（降89%）和上行数据量（降48.6倍）。

**[Finrpt Dataset Evaluation System And Llm-Based Multi-Agent Framework For Equity ](finrpt_dataset_evaluation_system_and_llm-based_multi-agent_framework_for_equity_.md)**

:   首次系统化地定义股票研究报告（ERR）自动生成任务——构建 FinRpt 数据集（6,825篇中英文高质量研报，整合7类金融数据），提出11指标评估体系和9 Agent协作的FinRpt-Gen生成框架（含评级修正/专家审查/润色三阶段增强），人类评估显示生成报告质量接近专家撰写。

**[From Biased Chatbots To Biased Agents Examining Role Assignment Effects On Llm A](from_biased_chatbots_to_biased_agents_examining_role_assignment_effects_on_llm_a.md)**

:   首个系统性案例研究，揭示基于人口统计学的 persona 分配会导致 LLM Agent 在 5 个操作领域的任务执行中出现最高 26.2% 的性能下降，证明 persona 诱导的偏见从文本生成延伸到了行动决策层面。

**[History-Aware Reasoning For Gui Agents](history-aware_reasoning_for_gui_agents.md)**

:   提出 HAR 框架，通过构建反思学习场景、合成纠错指南、设计混合 RL 奖励函数（含 Memory-Augmented Reward），将 GUI Agent 的推理模式从"历史无感知"转变为"历史感知"，3B 模型在 AITW/Mind2Web/GUI-Odyssey 等多个 benchmark 上超越更大模型。

**[Liecraft A Multi-Agent Framework For Evaluating Deceptive Capabilities In Langua](liecraft_a_multi-agent_framework_for_evaluating_deceptive_capabilities_in_langua.md)**

:   设计LieCraft多人隐藏角色博弈框架（约束满足问题确保平衡），评估12个LLM的战略欺骗能力，发现所有测试的前沿LLM（含GPT-4）在激励下都展现90%+的欺骗率——安全训练未消除策略性撒谎能力。

**[Llandmark A Multi-Agent Framework For Landmark-Aware Multimodal Interactive Vide](llandmark_a_multi-agent_framework_for_landmark-aware_multimodal_interactive_vide.md)**

:   提出 LLandMark 模块化多 Agent 框架，通过地标知识增强、LLM 辅助图像检索和 OCR 精炼模块，在越南大规模视频检索挑战赛（HCMAIC 2025）中实现地标感知的多模态交互式视频检索，总分 77.40/88。

**[Llmtm Benchmarking And Optimizing Llms For Temporal Motif Analysis In Dynamic Gr](llmtm_benchmarking_and_optimizing_llms_for_temporal_motif_analysis_in_dynamic_gr.md)**

:   提出 LLMTM——首个评估 LLM 处理动态图中时序 motif 分析能力的综合基准，包含 6 类任务覆盖 9 种时序 motif 类型，评估 9 个模型后发现 LLM 对时序 motif 的识别能力随 motif 复杂度快速下降。提出结构感知分派器（Structure-Aware Dispatcher），根据图的结构属性和认知负荷智能路由查询到标准 LLM 提示或工具增强 Agent，在维持高准确率的同时降低计算成本。

**[Loss-Guided Auxiliary Agents For Overcoming Mode Collapse In Gflownets](loss-guided_auxiliary_agents_for_overcoming_mode_collapse_in_gflownets.md)**

:   提出 LGGFN（Loss-Guided GFlowNets），用辅助 GFlowNet 的探索直接由主 GFlowNet 的训练损失驱动——辅助 Agent 的奖励 = 原始奖励 + λ·主模型损失，优先采样主模型理解不足的区域，在网格/序列/贝叶斯结构学习任务上分别发现 40× 更多唯一模式、99% 探索误差降低。

**[Medla A Logic-Driven Multi-Agent Framework For Complex Medic](medla_a_logic-driven_multi-agent_framework_for_complex_medic.md)**

:   提出 MedLA，首个基于三段论逻辑树的医学多 Agent 推理框架：每个 Agent 将推理组织为显式的逻辑树（大前提-小前提-结论三段论节点），多个 Agent 通过图引导的多轮讨论在前提级别对齐和修正逻辑树，在 MedDDx 上超越所有基线 7.4%（8B 模型），在医学 QA 上以 8B 模型达到 69.9% 平均准确率（超 70B RAG 模型）。

**[Moralreason Generalizable Moral Decision Alignment For Llm Agents Using Reasonin](moralreason_generalizable_moral_decision_alignment_for_llm_agents_using_reasonin.md)**

:   使用Group Relative Policy Optimization (GRPO)在推理层面训练LLM进行道德框架对齐，在Moral-Reason-QA数据集（680个高歧义场景）上实现功利主义对齐分数从0.207提升到0.964的分布外泛化。

**[Parallelism Meets Adaptiveness Scalable Documents Understanding In Multi-Agent L](parallelism_meets_adaptiveness_scalable_documents_understanding_in_multi-agent_l.md)**

:   提出自适应协调的多 Agent LLM 框架，通过并行竞争评估、动态任务路由和双向反馈机制，在高复杂度金融文档分析任务中实现 27% 的合规准确率提升和 74% 的修订率降低。

**[Pertouch Vlm-Driven Agent For Personalized And Semantic Image Retouching](pertouch_vlm-driven_agent_for_personalized_and_semantic_image_retouching.md)**

:   提出 PerTouch 框架，结合基于 Stable Diffusion + ControlNet 的语义区域级修图模型和 VLM 驱动的 Agent（含反馈重思考机制和场景感知记忆），实现精细化、个性化的图像修图。

**[Physics-Informed Autonomous Llm Agents For Explainable Power Electronics Modulat](physics-informed_autonomous_llm_agents_for_explainable_power_electronics_modulat.md)**

:   提出PHIA系统：LLM规划器通过聊天接口收集设计需求，协调物理信息神经网络代理模型（层次化PINN）和优化算法自主迭代生成电力转换器调制设计方案，MAE降低63.2%、设计速度提升33倍、20位专家验证可用性。

**[Probench Benchmarking Gui Agents With Accurate Process Infor](probench_benchmarking_gui_agents_with_accurate_process_infor.md)**

:   提出 ProBench，首个同时评估"最终状态"和"操作过程"的移动端 GUI Agent benchmark：200+ 挑战性任务覆盖 34 个中英文主流 App，通过 Process Provider（Structure Description Converter + MLLM Summarizer）自动捕获精确的中间过程信息，评估发现最强模型 Gemini 2.5 Pro 也仅完成 40.1% 任务，暴露了 grounding 不足、历史操作感知差、任务规划过于简化三大普遍问题。

**[Promoting Sustainable Web Agents Benchmarking And Estimating Energy Consumption ](promoting_sustainable_web_agents_benchmarking_and_estimating_energy_consumption_.md)**

:   首次系统性地从实证基准测试和理论估算两个角度量化了 Web Agent 的能耗与碳排放，发现更高能耗并不等于更好性能，并倡导在评测中引入能效指标。

**[Prune4Web Dom Tree Pruning Programming For Web Agent](prune4web_dom_tree_pruning_programming_for_web_agent.md)**

:   提出 Prune4Web，通过"LLM 生成评分函数参数 + 固定启发式模板执行"的编程式 DOM 剪枝方法实现 25-50 倍候选元素缩减：三阶段 pipeline（Planner 分解子任务 → Programmatic Filter 生成评分函数剪枝 DOM → Grounder 执行操作），3B 模型在 Multimodal-Mind2Web 上达到 52.4% Step SR（超越所有同参数量基线甚至部分 9.6B/32B 模型），低级 grounding 准确率从 46.8% 提升至 88.28%。

**[Real-Time Trust Verification For Safe Agentic Actions Using Trustbench](real-time_trust_verification_for_safe_agentic_actions_using_trustbench.md)**

:   提出TrustBench双模式框架：(1) 基准模式——结合传统指标和LLM-as-a-Judge评估8个信任维度，学习Agent置信度与实际正确率的校准映射；(2) 验证模式——在Agent制定行动后、执行前实时计算信任分数，阻止87%的有害行动，延迟低于200ms，通过领域插件（医疗/金融/QA）实现专业化验证。

**[Reflection-Driven Control For Trustworthy Code Agents](reflection-driven_control_for_trustworthy_code_agents.md)**

:   提出 Reflection-Driven Control 模块，将"自我反思"从事后补丁提升为 Agent 推理过程中的一等控制回路，通过轻量自检、证据驱动修复和反思记忆库三个组件，在安全代码生成任务上显著提升代码安全率。

**[Some A Realistic Benchmark For Llm-Based Social Media Agents](some_a_realistic_benchmark_for_llm-based_social_media_agents.md)**

:   提出 SoMe，首个全面评估 LLM 社交媒体 Agent 的 benchmark：8 个任务覆盖帖子分析、用户理解、综合推理，917 万帖子 + 6591 用户 + 1.7 万标注查询，配套 8 个 MCP 兼容工具，评估 13 个主流 LLM 发现最强模型仅 54.33 分（满分 100），揭示了推理能力≠Agent能力、工具调用幻觉普遍存在等关键发现。

**[Structured Personalization Modeling Constraints As Matroids For Data-Minimal Llm](structured_personalization_modeling_constraints_as_matroids_for_data-minimal_llm.md)**

:   将 LLM Agent 个性化中的结构化约束（逻辑依赖 + 层级配额）形式化为层叠拟阵（laminar matroid），证明贪心算法在此约束下仍具有常数因子近似保证，解决了有依赖关系和层级限制的数据最小化选择问题。

**[Thucy An Llm-Based Multi-Agent System For Claim Verification Across Relational D](thucy_an_llm-based_multi-agent_system_for_claim_verification_across_relational_d.md)**

:   提出首个跨数据库、跨表的多 Agent 声明验证系统 Thucy，由 Verifier 领导三个专家 Agent（Data/Schema/SQL Expert），对数据源完全无先验知识，能自主发现、推理并生成 SQL 证据，在 TabFact 上超越 SOTA 5.6 个百分点（94.3%）。

**[Tongui Internet-Scale Trajectories From Multimodal Web Tutor](tongui_internet-scale_trajectories_from_multimodal_web_tutor.md)**

:   TongUI 提出从互联网上的多模态教程（视频+图文）自动转化为 GUI 操作轨迹数据的框架，构建了百万级的 GUI-Net-1M 数据集，用于微调 Qwen2.5-VL 模型，在多个 grounding 和 navigation 基准上超越或接近 UI-TARS 等 SOTA。

**[Towards Trustworthy Multi-Turn Llm Agents Via Behavioral Guidance](towards_trustworthy_multi-turn_llm_agents_via_behavioral_guidance.md)**

:   提出任务完成框架，通过任务分析器（Task Profiler）、推理模块（Reasoning Module）和生成模块（Generation Module）三组件协同进化，使 LLM Agent 在多轮交互环境中实现可验证和可靠的行为引导。

**[When Refusals Fail Unstable Safety Mechanisms In Long-Context Llm Agents](when_refusals_fail_unstable_safety_mechanisms_in_long-context_llm_agents.md)**

:   系统研究 LLM Agent 在长上下文填充下的安全行为变化：发现声称支持 1M-2M token 的模型在 100K token 时已出现 >50% 的性能崩溃，拒绝率以不可预测的方式波动（GPT-4.1-nano 从 5% 升至 40%，Grok 4 Fast 从 80% 降至 10%），揭示了长上下文 Agent 系统的严重安全隐患。

**[With Great Capabilities Come Great Responsibilities Introducing The Agentic Risk](with_great_capabilities_come_great_responsibilities_introducing_the_agentic_risk.md)**

:   提出 Agentic Risk & Capability (ARC) 框架，从能力（Capability）视角系统化地识别、评估和缓解智能体 AI 系统的安全与安全风险，为组织级治理提供可操作的结构化方法论。
