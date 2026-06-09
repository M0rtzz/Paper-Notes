---
title: >-
  ICML2026 LLM Agent方向43篇论文解读
description: >-
  43篇ICML2026的 LLM Agent 方向论文解读，涵盖 LLM、Agent、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "LLM Agent"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "Agent"
  - "推理"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**🧪 ICML2026** · **43** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (78)](../../ACL2026/llm_agent/index.md) · [📷 CVPR2026 (15)](../../CVPR2026/llm_agent/index.md) · [🔬 ICLR2026 (40)](../../ICLR2026/llm_agent/index.md) · [🤖 AAAI2026 (30)](../../AAAI2026/llm_agent/index.md) · [🧠 NeurIPS2025 (39)](../../NeurIPS2025/llm_agent/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/llm_agent/index.md)

🔥 **高频主题：** LLM ×18 · Agent ×12 · 推理 ×5

**[A Minimal Agent for Automated Theorem Proving](a_minimal_agent_for_automated_theorem_proving.md)**

:   本文提出 AxProverBase——一个极简的 Lean 4 定理证明智能体，仅靠"编译器反馈 + 自管理笔记本 + 轻量工具搜索"三个组件，在不微调的前沿 LLM（Claude Opus）上达到甚至超越 Hilbert/Seed-Prover 等专用系统，成本却低出 100 倍。

**[ACON: Optimizing Context Compression for Long-horizon LLM Agents](acon_optimizing_context_compression_for_long-horizon_llm_agents.md)**

:   Acon 用失败轨迹对比来优化自然语言压缩指南，同时压缩 agent 的历史和观察上下文，在 AppWorld、OfficeBench 和多目标 QA 上把峰值 token 降低 26% 到 54%，并保持或提升长程任务成功率。

**[Agent-Omit: Adaptive Context Omission for Efficient LLM Agents](agent-omit_adaptive_context_omission_for_efficient_llm_agents.md)**

:   通过 Monte-Carlo rollout 量化"哪些回合的 thought / observation 可以省"，再用冷启动 SFT + 双采样 omit-aware GRPO 训出能自适应跳过冗余思考和观测的 8B agent，五个基准上 token 用量大降而准确率与七大前沿模型持平。

**[Agent JIT Compilation for Latency-Optimizing Web Agent Planning and Scheduling](agent_jit_compilation_for_latency-optimizing_web_agent_planning_and_scheduling.md)**

:   这篇论文把网页 Computer-Use Agent 从逐步截图-调用 LLM-执行的循环，改造成类似 JIT 编译器的系统：把自然语言任务编译成可校验、可缓存、可并行调度的代码计划，从而让 JIT-Planner 比 Browser-Use 快 10.4×且准确率高 28pp，让 JIT-Scheduler 比 OpenAI CUA 快 2.4×且准确率高 9pp。

**[AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction](agentxray_white-boxing_agentic_systems_via_workflow_reconstruction.md)**

:   作者把"对黑盒 agent 系统反推一个等价白盒 workflow"作为新任务 AWR，用 MCTS 在 agent 原语序列空间中搜索，再配上一种基于评分动态着色的 Red-Black 剪枝来平衡深度与宽度，在五个真实领域上实现可解释的白盒重建。

**[Answer Only as Precisely as Justified: Calibrated Claim-Level Specificity Control for Agentic Systems](answer_only_as_precisely_as_justified_calibrated_claim-level_specificity_control.md)**

:   这篇论文把 agentic 系统里的“说得过细但证据不够”建模为 claim 级过度承诺问题，并提出 calibrated CSS：对每个原子 claim 在精确表述、粗粒度回退和省略之间做校准选择，在 LongFact 全量实验中将 OAU 从无后处理的 0.8460 提升到 0.9130，同时保留 0.9381 的特异性。

**[AutoRPA: Efficient GUI Automation through LLM-Driven Code Synthesis from Interactions](autorpa_efficient_gui_automation_through_llm-driven_code_synthesis_from_interact.md)**

:   提出 AutoRPA 框架，通过翻译器-构建器流水线将 ReAct 风格 GUI Agent 的交互轨迹自动蒸馏为可复用的 RPA 函数，结合混合修复策略迭代优化代码，在保持甚至超越原始 Agent 成功率的前提下减少 82%~96% 的 token 消耗。

**[Constitutional Black-Box Monitoring for Scheming in LLM Agents](constitutional_black-box_monitoring_for_scheming_in_llm_agents.md)**

:   本文提出一套端到端的"宪法式黑盒监控"框架，利用两条合成数据流水线（STRIDE 和 Gloom）生成 2,000 条合成轨迹来优化提示分类器，在仅观察外部可见的工具调用与输出（不看 CoT）的条件下检测 LLM 代理的阴谋行为，发现简单的 prompt grid search 即可饱和性能，更激进的优化反而导致过拟合。

**[EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)**

:   EvolveR 给 LLM agent 套一个「在线交互 → 离线自蒸馏成原则库 → GRPO 策略进化」的闭环生命周期：agent 不再丢弃过去轨迹，而是把自己的成功失败抽象成可检索的「策略原则」，再用 RL 学会**如何用自己的原则**去解新问题，在 7 个多跳 QA benchmark 上明显跑赢 Search-R1 等 RL agent baseline。

**[ExCyTIn-Bench: Evaluating LLM Agents on Cyber Threat Investigation](excytin-bench_evaluating_llm_agents_on_cyber_threat_investigation.md)**

:   本文构建了首个评测 LLM Agent 端到端做"网络威胁调查"的 benchmark ExCyTIn-Bench：从真实 Azure 租户的 57 张安全日志表里，用 alert-entity 二部图自动生成 7542 道带证据链的 SQL 问答题，并提供 MySQL 环境让 Agent 通过查询日志、多跳追踪证据来回答，目前最强模型 Claude-Opus-4.5 也只能拿 0.606 的 reward。

**[Hallucinations Undermine Trust; Metacognition is a Way Forward](hallucinations_undermine_trust_metacognition_is_a_way_forward.md)**

:   本文是一篇 position paper，论证"彻底消除 LLM 幻觉"在原理上无法逃避一个"区分度税"（discrimination gap → utility tax）；作者主张把目标从"消灭幻觉"改为**忠实表达不确定性**（faithful uncertainty），并把这种 metacognition 视为 agentic LLM 调用工具时不可或缺的控制层。

**[HawkesLLM: Semantic Uncertainty Propagation in Agentic Text Simulation](hawkesllm_semantic_uncertainty_propagation_in_agentic_text_simulation.md)**

:   HawkesLLM 把多变量 Hawkes 点过程嫁接到 LLM 智能体文本模拟循环中：Hawkes 负责安排"何时由哪个节点生成"以及"用哪些历史节点的输出作为压缩记忆"，LLM 只负责把被选中的记忆写成下一条事件，在 GDELT Artemis II 新闻级联上获得了在紧凑提示预算下仍随时间上升的后段语义对齐度。

**[Hunt Instead of Wait: Evaluating Deep Data Research on Large Language Models](hunt_instead_of_wait_evaluating_deep_data_research_on_large_language_models.md)**

:   本文提出 Deep Data Research（DDR）这一开放式 agentic 任务范式——只给 LLM 一个结构化数据库和最小工具集（SQL+Python），不给任何具体问题或回合上限，要求模型自主探索、提出假设并决定何时停止；并构建 DDR-Bench（MIMIC-IV / GLOBEM / 10-K，291 个实体、2058 条 checklist），用从非结构化文本抽取的可验证 fact checklist 客观评测主流 LLM 的"investigatory intelligence"，结果显示即使 Claude 4.5 Sonnet 也只能拿到 47.7% 平均准确率。

**[Internalizing Agency from Reflective Experience](internalizing_agency_from_reflective_experience.md)**

:   本文提出 LEAFE 框架，让 LLM agent 通过反思失败轨迹生成「失败→回滚→修正→成功」的经验数据，再用 SFT 蒸馏出 feedback-grounded 的恢复能力，在 CodeContests、WebShop、ALFWorld 等长程任务上把 Pass@128 拉高最多 14%，远胜 GRPO 等 outcome-driven RL。

**[Learning Efficient Guardrails for Compliance](learning_efficient_guardrails_for_compliance.md)**

:   本文构造了一个 60k 规模的 PolicyGuardBench（5 个域、733 条标准化轨迹 × 2195 条原子策略 → 6 万 trajectory-policy 对，含跨子域和前缀截断设置），并基于 Qwen3-4B-Instruct 全参数 SFT 出一个轻量 guardrail 模型 PolicyGuard-4B，在 22.5 ms/样本的延迟下取得 90.14% 准确率 / 87.59% F1，匹配甚至超过 70B 级开源模型和 Claude-Sonnet-4，并展现了强跨域泛化（LODO OOD F1≈0.91）。

**[Lifting Traces to Logic: Programmatic Skill Induction with Neuro-Symbolic Learning for Long-Horizon Agentic Tasks](lifting_traces_to_logic_programmatic_skill_induction_with_neuro-symbolic_learnin.md)**

:   NSI 把 LLM agent 的交互轨迹 "提升" 为带显式条件分支和动态变量绑定的神经符号工作流图，使技能从无状态脚本进化成可状态感知的逻辑程序，在 ALFWorld / WebShop / TextCraft 上分别拿到 98.0 / 76.5 / 95.2 的成功率，全面碾压 ASI 和 AWM 等编程式技能基线。

**[LLM Agents Are the Antidote to Walled Gardens](llm_agents_are_the_antidote_to_walled_gardens.md)**

:   这是一篇 ICML 2026 立场论文，主张 LLM 智能体能够通过自动格式转换 + 拟人化 UI 交互"绕过"主流平台的 API 封闭策略，实现"通用互操作性"（universal interoperability），从而瓦解传统网络效应造成的"围墙花园"，但同时需要 ML 社区主动建立 agent-friendly 接口、安全机制和生态基础设施来管控随之而来的安全、法律与新一层 lock-in 风险。

**[MCP-Persona: 用环境模拟评估 LLM agent 在真实个人化应用上的能力](mcp-persona_benchmarking_llm_agents_on_real-world_personal_applications_via_envi.md)**

:   MCP-Persona 是首个针对真实个人化 MCP 工具（Slack/Rednote/Instagram/Lark 等 12 服务器）的 LLM agent benchmark；提出 Tool-Traverse + Context-Tree + Persona-Gen 三套方法，用 LLM 自动 synthesize Python simulator 代码避免真实账号问题；测 10+ SOTA agent 发现连 Claude-Sonnet-4.5 也只达 38.66% Acc，证明个人化工具使用是被严重低估的能力短板。

**[NaviAgent: Graph-Driven Bilevel Planning for Scalable Tool Orchestration](naviagent_graph-driven_bilevel_planning_for_scalable_tool_orchestration.md)**

:   NaviAgent 把 LLM 工具调用拆成"高层四选一决策 + 低层图上路径搜索"两层，由一个用 HGT 训练的 Tool World Navigation Model（TWNM）显式建模工具之间的结构与行为依赖，在 ToolBench/API-Bank 与 50 个真实 RapidAPI 上把任务成功率（TSR）相对最强基线整体提升 4.3–18.2 个点，同时显著减少调用步数。

**[On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM Agents](on_information_self-locking_in_reinforcement_learning_for_active_reasoning_of_ll.md)**

:   针对 LLM agent 在多轮主动推理中"动作选择(AS)"与"信念跟踪(BT)"互相拖累、outcome-only RL 训练陷入低信息自锁(SeL)的失效模式，本文给出 POMDP 视角下的耦合梯度分析与"自锁区"形式化定义，并提出 AReW：用环境/读出层即可获得的方向性 critique 对 stepwise advantage 做加性 reweighting，在 9 个主动推理任务上最高带来 60 分性能提升。

**[OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents](otora_a_unified_red_teaming_framework_for_reasoning-level_denial-of-service_in_l.md)**

:   OTora 提出一种全新的攻击范式 Reasoning-Level Denial-of-Service（R-DoS）：不破坏任务正确性，而是通过两阶段红队管线（先用插入感知优化诱导 agent 主动访问攻击者控制的外部资源，再在该资源里投放经 ICL 遗传搜索优化的「思考型 payload」）让 LLM agent 进入持续多轮的过度推理状态，在 WebShop / Email / OS 三类 agent 上实现 10× 推理 token 膨胀和数量级延迟攻击，且最终任务准确率几乎不变。

**[Persona2Web: Benchmarking Personalized Web Agents for Contextual Reasoning with User History](persona2web_benchmarking_personalized_web_agents_for_contextual_reasoning_with_u.md)**

:   本文提出首个针对个性化 web agent 的开放网页 benchmark Persona2Web，用「隐式用户历史 + 三档模糊查询 + 推理感知评分」逼迫 agent 从浏览记录中推断用户偏好来消歧 ambiguous query；在 GPT-4.1 / o3 等 5 个主流模型上，即使提供历史，level-2 query 的成功率也只有 13%，揭示当前 web agent 缺乏真正的个性化能力。

**[Position: Agentic AI Orchestration Should Be Bayes-Consistent](position_agentic_ai_orchestration_should_be_bayes-consistent.md)**

:   这篇 position paper 主张：不要再尝试让 LLM 本身 "Bayesian"（那条路在工程上和理论上都跳不过去），而是把贝叶斯结构搬到 agentic AI 的**编排控制层**——让控制器维护一个低维任务级隐变量的信念，按 Bayes 规则在 agent/工具返回的"消息观测"上更新，并用期望效用或 value-of-information 做路由、停止、升级和预算分配。

**[Position: Assistive Agents Need Accessibility Alignment](position_assistive_agents_need_accessibility_alignment.md)**

:   这是一篇 position paper，作者通过对 417 篇文献中 778 个盲人辅助任务实例做系统综述，论证 "accessibility alignment" 应当被视为与 helpful/harmless/honest 并列的 Agent 一级对齐目标，并提出覆盖目标-交互-风险-生命周期四维度的设计 pipeline。

**[Post-Training LLMs as Better Decision-Making Agents: A Regret-Minimization Approach](post-training_llms_as_better_decision-making_agents_a_regret-minimization_approa.md)**

:   作者提出 Iterative RMFT，把 LLM 自己 rollout 出来的决策轨迹按 regret 从低到高排序，挑出最优的 $k$ 条用 SFT 反复微调模型，从而在不依赖任何已知最优算法（如 UCB/FTRL）也不需要人工设计 CoT 模板的前提下，让 LLM 在多臂赌博机、在线学习和非平稳赌博机这三类语言化决策任务上自动涌现出 no-regret 行为和合理的探索-利用平衡。

**[PragLocker: Protecting Agent Intellectual Property in Untrusted Deployments via Non-Portable Prompts](praglocker_protecting_agent_intellectual_property_in_untrusted_deployments_via_n.md)**

:   PragLocker 用 "代码符号初始化 + 黑盒目标模型反馈下的噪声注入" 两阶段策略，把 agent system prompt 编码成一段只能在 target LLM 上 work、迁移到其它任意 LLM 都会失效的 obfuscated text，从而在 prompt 被部署侧窃取时让攻击者无法在自己的 LLM 上复用。

**[Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks](probabilistic_modeling_of_latent_agentic_substructures_in_deep_neural_networks.md)**

:   作者把神经网络（特别是 LLM）形式化为多个隐式子代理（每个是 outcome 上的概率分布）通过对数加权池化合成的复合代理，并在认知效用 $W_i(o)=\log P_i(o)$ 框架下证明了 "严格一致受益（strict unanimity）" 在线性池化或二元 outcome 下不可能、但 $|\mathcal O|\ge 3$ 下可行，进而推出"显式让 Waluigi 先显形再压制"严格优于"只强化 Luigi"的对齐原则。

**[Process Reward Agents for Steering Knowledge-Intensive Reasoning](process_reward_agents_for_steering_knowledge-intensive_reasoning.md)**

:   把过程奖励模型从"事后打分"重构成一个**在线 agent**：在每个推理步实时决定是否检索证据并给出奖励，借助 beam search 对冻结策略的候选轨迹进行剪枝，使 Qwen3-4B 在 MedQA 上达到 81.9% 的 4B-scale SOTA，且能直接迁移到 0.5B–8B 各种未见骨干（最高带来 25.7% 提升）。

**[Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents](recovering_policy-induced_errors_benchmarking_and_trajectory_synthesis_for_robus.md)**

:   针对 GUI 智能体在真实部署中容易陷入"自己造的错误"无法恢复这一痛点，作者一边搭出 GUI-RobustEval（1216 个可执行测试，覆盖 11 种策略诱发错误 + 4 档错误深度）做细粒度评测，一边提出 RoTS——一种基于轨迹树的在线数据合成框架：在成功子树上用脆弱度 UCB 主动暴露新错误，在失败子树上用邻居经验做长程恢复回滚，最终合成 800k 反思数据，使 RoTS-32B 在 OSWorld 上拿到 47.4% SR / 33.8% All-Pass@4 的开源 SOTA。

**[ReflexGrad: Within-Episode Failure Recovery in LLM Agents via Progress-Gated Dual-Process Routing](reflexgrad_within-episode_failure_recovery_in_llm_agents_via_progress-gated_dual.md)**

:   ReflexGrad 把 TextGrad 的"每 3 步局部梯度精调"作为快过程、把 Reflexion 风格的"连续低分触发的因果重规划"作为慢过程，用一条进度门控路由规则在**同一个 episode 内**无示范地切换两者，在 ALFWorld 134 任务上把 Qwen-3-8B 从 35.1% 拉到 75.4%（+40.3pp），并在算力对等条件下击败 1-shot 的 LATS / ToT / Self-Refine。

**[Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use](reward_hacking_benchmark_measuring_exploits_in_llm_agents_with_tool_use.md)**

:   RHB 构造了一套现实工具型多步任务（独立 + 链式两种模式，含数据流水线、日志取证、性能优化、多文件重建四大家族）来量化 LLM agent 的奖励黑客行为，跨 13 个前沿模型发现 RL 后训练显著提高 exploit 率（DeepSeek-V3 0.6% vs R1-Zero 13.9%），且 exploit 率随链长上升、在更难变体上即使近零率模型也会"复发"，而轻量级环境硬化能在不损害任务成功率前提下把 exploit 率减少 87.7%。

**[Rule2DRC: Benchmarking LLM Agents for DRC Script Synthesis with Execution-Guided Test Generation](rule2drc_benchmarking_llm_agents_for_drc_script_synthesis_with_execution-guided_.md)**

:   作者构建了 Rule2DRC —— 一个含 1000 条自然语言设计规则、13921 个评测版图的大规模 EDA 基准，通过 KLayout 引擎做执行级别打分而非代码相似度对比，并提出 SplitTester：把 N 个候选 DRC 脚本按"在当前测试下是否输出一致"做聚类，每轮挑「分数 × 簇大小」最大的簇生成新版图把它拆开，最后让 judge LLM 在 Top-3 候选与其差异化测试上选最优。

**[SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety](safeharbor_hierarchical_memory-augmented_guardrail_for_llm_agent_safety.md)**

:   SafeHarbor 把 LLM Agent 的安全防御从「静态粗粒度分类器」升级为「动态分层记忆树 + 双分数门控」，通过对抗规则生成 + 信息熵自演化让 GPT-4o 在保持 93%+ 拒绝率的同时把 benign 工具调用成功率拉到 63.6%，显著缓解 over-refusal 问题。

**[Scaling, Benchmarking, and Reasoning of Vision-Language Agents for Mobile GUI Navigation](scaling_benchmarking_and_reasoning_of_vision-language_agents_for_mobile_gui_navi.md)**

:   小米团队针对 VLM 移动 GUI 智能体提出"数据-评测-推理"三位一体的系统性研究：发布 16k 任务 / 674 个中文 App 的 HyperTrack 数据集和支持 30+ 模型的 GUIEvalKit 评测工具，证明 DAPO 风格 RL 在 OOD 场景明显胜过 SFT、并用半在线评测 SOEval 揭示了"显式 reasoning 会牺牲 PASS@1 稳定性但提升 PASS@n 多样性"的核心权衡。

**[Scaling Small Agents Through Strategy Auctions](scaling_small_agents_through_strategy_auctions.md)**

:   论文提出 sale（Strategy Auctions for Workload Efficiency）：让大小不一的 Qwen3 智能体在每个任务上提交"策略短计划"作为竞拍标书，按 cost-minus-value 选出执行者，并用历史竞拍记忆让便宜 agent 持续精炼自己的标书；在 deep search 与 coding 上既超过最大模型的 pass@1，又把对最大 agent 的依赖降低 52%、总开销降低 35%。

**[SE-GA: Memory-Augmented Self-Evolution for GUI Agents](se-ga_memory-augmented_self-evolution_for_gui_agents.md)**

:   SE-GA 给基于 VLM 的 GUI 智能体配了一套"情景+语义+经验"三层记忆库（TTME）+ 一个两阶段记忆增强自演化训练流程（MASE，SFT→改进版 GRPO），把 Qwen2.5-VL-7B 在 ScreenSpot 推到 89.0、AndroidControl-High 推到 75.8、AndroidWorld 推到 39.0，全面超越同规模基线甚至打平 72B 模型。

**[Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents](skill-pro_learning_reusable_skills_from_experience_via_non-parametric_ppo_for_ll.md)**

:   Skill-Pro 把 LLM agent 的交互经验显式抽成"激活+执行+终止"三件套的 Skill，用语义梯度生成候选 Skill、再用 PPO 风格的信任域验证 (PPO Gate) 决定是否纳入，最终在 ALFWorld / Mastermind 上以 ~800 token 的极小记忆库实现 0.85+ 的复用率和显著性能提升。

**[Talk, Judge, Cooperate: Gossip-Driven Indirect Reciprocity in Self-Interested LLM Agents](talk_judge_cooperate_gossip-driven_indirect_reciprocity_in_self-interested_llm_a.md)**

:   本文提出 ALIGN，让一群完全自利、去中心化的 LLM 智能体通过五档语气的公开"八卦"消息互相评价、形成声誉、惩罚背叛，从而在无中心监管的捐赠博弈、投资博弈和电商市场中稳定地建立间接互惠合作，并发现推理型 LLM 比 chat 型 LLM 更能按博弈论激励"该合作时才合作"。

**[Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction](think_twice_before_you_act_enhancing_agent_behavioral_safety_with_thought_correc.md)**

:   本文提出 Thought-Aligner——一个 1.5B/7B 的轻量级即插即用安全模型，在 LLM agent 的 think-act-observe 循环里、在每个动作执行前对中间思维做因果纠偏，把 6 个主流 LLM 在 ToolEmu/Agent-SafetyBench 上的行为安全率从约 50% 拉到约 90%，同时帮助度还提升约 5%。

**[Towards a Science of AI Agent Reliability](towards_a_science_of_ai_agent_reliability.md)**

:   论文借鉴航空 / 核能 / 汽车等安全关键工程的成熟做法，把 AI agent 的"可靠性"分解为一致性、鲁棒性、可预测性、安全四个维度共 12 个与准确率无关的指标，在 GAIA 和 $\tau$-bench 两个基准上系统评测 15 个前沿模型，得出"过去 24 个月准确率猛涨、可靠性几乎没动"这一行业级结论。

**[Towards Feedback-to-Plan Decisions for Self-Evolving LLM Agents in CUDA Kernel Generation](towards_feedback-to-plan_decisions_for_self-evolving_llm_agents_in_cuda_kernel_g.md)**

:   针对自演化 LLM agent 写 CUDA kernel 的场景，提出 CUDAnalyst：通过"冻结某一代中间程序状态 + 选择性注入/屏蔽反馈"做生成级干预，并用 Banzhaf 联盟博弈解构 debugger / analyzer / profiler 三类反馈的边际贡献和高阶交互，得出"显式 plan 只有在反馈对齐时才有用、强模型的 plan 可向同家族弱模型迁移"等四条结论，并据此设计出 2.08×–10.32× 超过 torch.compile 的 CuGEdit 插件。

**[Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining](video2gui_synthesizing_large-scale_interaction_trajectories_for_generalized_gui_.md)**

:   Video2GUI 用「元数据粗筛 → 视频质量精筛 → Gemini-3-Pro 提任务/动作 → 高分辨率三帧精确空间 grounding」四段流水线把 5 亿条 YouTube 视频元数据炼成 WildGUI（12.7M 轨迹、124.5M 截图、1500+ 应用），并把 Qwen2.5-VL/Mimo-VL 在多个 GUI grounding 与 agent benchmark 上提升 5–20%。

**[Weasel: 通过重要性-多样性数据选择实现 Web Agent 的域外泛化](weasel_out-of-domain_generalization_for_web_agents_via_importance-diversity_data.md)**

:   通过结合目标相关性和多样性的轨迹步骤选择方法，Weasel 在减少训练数据到原始 20% 的同时实现 9.7-12.5 倍训练加速，并显著提升 Web Agent 在未见域上的泛化能力。
