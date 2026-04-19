---
title: >-
  AAAI2026 强化学习方向 66篇论文解读
description: >-
  66篇AAAI2026 强化学习方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎮 强化学习

**🤖 AAAI2026** · **66** 篇论文解读

**[A Course Correction in Steerability Evaluation: Revealing Miscalibration and Side Effects in LLMs](a_course_correction_in_steerability_evaluation_revealing_mis.md)**

:   本文提出了一个基于多维目标空间的 LLM 可操控性（steerability）评估框架，将 steering error 分解为校准偏差（miscalibration）和副作用（side effects/orthogonality），在文本改写任务上发现即使是最强的 LLM 也会产生严重副作用，prompt engineering 无效、best-of-N 采样代价高、RL 微调有改善但仍未彻底解决。

**[A Learning Framework For Cooperative Collision Avoidance of UAV Swarms Leveraging Domain Knowledge](a_learning_framework_for_cooperative_collision_avoidance_of_.md)**

:   提出 reMARL 框架，将图像处理中的主动轮廓模型（active contour）作为领域知识引入多智能体强化学习的奖励设计，使无人机集群仅通过最大化个体奖励即可学会协作避撞，在大规模集群（≤10架）中性能显著优于 COMA/VDN/QMIX/MAPPO 等 SOTA MARL 方法，反应时间比元启发式方法快 98.75%，能耗降低 85.37%。

**[A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses](a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)**

:   提出 MACO 多智能体会话式 Bandit 框架，通过本地 agent 的在线淘汰和云服务器的自适应偏好查询机制，实现 LLM 响应的在线评估与用户偏好对齐，达到 $\tilde{O}(\sqrt{dMT})$ 的近优 regret 界。

**[Aligning Machiavellian Agents: Behavior Steering via Test-Time Policy Shaping](aligning_machiavellian_agents_behavior_steering_via_test-tim.md)**

:   提出一种测试时策略塑形方法，通过轻量级伦理属性分类器在推理阶段插值修改预训练 RL 智能体的动作概率分布，无需重训练即可实现对多种伦理属性的细粒度行为引导。

**[BAMAS: Structuring Budget-Aware Multi-Agent Systems](bamas_structuring_budget-aware_multi-agent_systems.md)**

:   提出 BAMAS 框架，通过整数线性规划（ILP）在预算约束下选择最优 LLM 组合，再用强化学习策略选择最佳协作拓扑（线性/星型/反馈/规划驱动），在 GSM8K/MBPP/MATH 上达到与 SOTA 多 Agent 系统相当的准确率，同时成本降低最高 86%。

**[Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning](behaviour_policy_optimization_provably_lower_variance_return_estimates_for_off-p.md)**

:   提出 Behaviour Policy Optimization (BPO)，通过优化一个专用行为策略来采集离策略数据，使得回报估计的方差可证明低于在策略采集，从而提升 REINFORCE 和 PPO 的样本效率与稳定性。

**[Beyond Monotonicity: Revisiting Factorization Principles in Multi-Agent Q-Learning](beyond_monotonicity_revisiting_factorization_principles_in_multi-agent_q-learnin.md)**

:   通过动力系统分析证明：在近似贪心探索策略下，非单调值分解Q学习中所有违反IGM一致性的零损失解都是不稳定鞍点，只有IGM一致解才是稳定吸引子，因此无需单调性约束即可可靠收敛到最优解。

**[Beyond the Lower Bound: Bridging Regret Minimization and Best Arm Identification in Lexicographic Bandits](beyond_the_lower_bound_bridging_regret_minimization_and_best_arm_identification_.md)**

:   提出两种消除式算法 LexElim-Out 和 LexElim-In，首次在词典序多目标赌博机中同时解决遗憾最小化（RM）和最优臂识别（BAI）问题，其中 LexElim-In 通过跨目标信息共享突破了单目标问题的已知下界。

**[Bi-Level Contextual Bandits for Individualized Resource Allocation under Delayed Feedback](bi-level_contextual_bandits_for_individualized_resource_allocation_under_delayed.md)**

:   提出 MetaCUB——一种双层上下文赌博机框架，在延迟反馈、动态人群、冷却约束和公平性要求下实现个体化资源分配，元层优化子群预算分配保证公平，基层利用 UCB 策略选择最有潜力的个体。

**[ChartEditor: A Reinforcement Learning Framework for Robust Chart Editing](charteditor_a_reinforcement_learning_framework_for_robust_chart_editing.md)**

:   提出 ChartEditVista 基准（7,964 样本、31 种图表类型）和 ChartEditor 模型，通过 GRPO 强化学习框架结合新颖的 rendering reward，仅用 3B 参数即在图表编辑任务上超越 GPT-4o 和多个 72B 级模型。

**[CHDP: Cooperative Hybrid Diffusion Policies for RL in Parametric Environments](chdp_cooperative_hybrid_diffusion_policies_for_reinforcement_learning_in_paramet.md)**

:   将混合动作空间问题建模为两个agent的全合作博弈，分别用离散和连续扩散策略生成动作，通过顺序更新和Q引导码本解决策略冲突与高维可扩展性问题，成功率最高提升19.3%。

**[Deep (Predictive) Discounted Counterfactual Regret Minimization](deep_predictive_discounted_counterfactual_regret_minimization.md)**

:   提出VR-DeepDCFR+和VR-DeepPDCFR+两种无模型神经CFR算法，通过自举累积优势估计、折扣裁剪机制和基线方差缩减，首次将高级表格CFR变体（DCFR+/PDCFR+）有效整合到神经网络近似框架中，在典型不完全信息博弈中实现更快收敛。

**[DeepProofLog: Efficient Proving in Deep Stochastic Logic Programs](deepprooflog_efficient_proving_in_deep_stochastic_logic_programs.md)**

:   提出DeepProofLog（DPrL），一种基于随机逻辑程序的神经符号系统，通过在每个证明步骤引入神经网络参数化，并建立SLD解析过程与MDP的形式化映射，使得动态规划和强化学习技术可用于高效推理与学习，显著提升了神经符号系统的可扩展性。

**[DiffOP: Reinforcement Learning of Optimization-Based Control Policies via Implicit Policy Gradients](diffop_reinforcement_learning_of_optimization-based_control_policies_via_implici.md)**

:   提出 DiffOP 框架，将优化型控制策略（如 MPC）视为可微分模块，通过隐式微分推导解析策略梯度，实现端到端强化学习训练，并给出首个非渐近收敛保证。

**[Discounted Cuts: A Stackelberg Approach to Network Disruption](discounted_cuts_a_stackelberg_approach_to_network_disruption.md)**

:   提出折扣切割（Discounted Cuts）数学模型，将经典 Most Vital Links 问题建模为 Stackelberg 博弈，系统研究8种折扣切割变体的计算复杂性分类，证明所有变体在有界亏格图上均可多项式时间求解。

**[Distilling Deep Reinforcement Learning into Interpretable Fuzzy Rules: An Explainable AI Framework](distilling_deep_reinforcement_learning_into_interpretable_fuzzy_rules_an_explain.md)**

:   提出层次化 Takagi-Sugeno-Kang (TSK) 模糊分类器系统，将深度 RL 的神经网络策略蒸馏为人类可读的 IF-THEN 模糊规则，引入三个量化可解释性度量（FRAD、FSC、ASG），在 Lunar Lander 连续控制任务上以 81.48% 的保真度超越决策树 21 个百分点。

**[Do It for HER: First-Order Temporal Logic Reward Specification in Reinforcement Learning](do_it_for_her_first-order_temporal_logic_reward_specification_in_reinforcement_l.md)**

:   提出基于有限迹一阶时序逻辑模理论（LTLfMT）的新型奖励规范框架，用一阶逻辑公式替代手工编码的标注函数，结合 CRM 和 HER 解决逻辑规范固有的稀疏奖励问题，在连续控制任务中取得显著改进。

**[Does Self-Evaluation Enable Wireheading in Language Models?](does_self-evaluation_enable_wireheading_in_language_models.md)**

:   本文理论证明并实验验证了当语言模型的自我评估与奖励信号耦合时，模型会系统性地膨胀自评分（wireheading），而解耦自评分与奖励可以缓解这一问题；在Llama-3.1-8B和Mistral-7B上三个任务的实验表明，摘要等模糊任务中自评分膨胀高达0.92。

**[DRMD: Deep Reinforcement Learning for Malware Detection under Concept Drift](drmd_deep_reinforcement_learning_for_malware_detection_under_concept_drift.md)**

:   本文首次将Android恶意软件检测重新表述为一步马尔可夫决策过程（MD-MDP），并训练基于PPO的深度强化学习智能体DRMD，在单一策略中统一了样本分类、拒绝和主动学习，在多年跨期评估中实现了平均8.66（仅分类）和10.90（含拒绝）的AUT提升，显著优于传统监督学习分类器应对概念漂移的能力。

**[Efficient Multiagent Planning via Shared Action Suggestions](efficient_multiagent_planning_via_shared_action_suggestions.md)**

:   提出 MCAS 算法，通过在去中心化 POMDP 中仅共享"建议动作"来推断其他智能体的信念状态，在大幅降低通信开销和计算复杂度的同时实现接近集中式方法的协调性能。

**[Explaining Decentralized Multi-Agent Reinforcement Learning Policies](explaining_decentralized_multi-agent_reinforcement_learning_policies.md)**

:   提出首个针对去中心化多智能体强化学习（MARL）策略的可解释方法，包括基于 Hasse 图的策略摘要生成和基于查询的自然语言解释（When/Why Not/What），在四个 MARL 领域展示了通用性和计算效率，用户研究表明显著提升了人类对策略的理解和问答表现。

**[Formal Verification of Diffusion Auctions](formal_verification_of_diffusion_auctions.md)**

:   首次提出面向扩散拍卖（diffusion auctions）的形式化逻辑框架，引入 $n$ 卖家扩散激励逻辑 $\mathcal{L}^n$ 及其策略扩展 $\mathcal{SL}^n$，支持对拍卖属性（如 Nash 均衡、卖家策略存在性）的模型检测验证，分别建立了 P 和 PSPACE-complete 的复杂度结果。

**[G-UBS: Towards Robust Understanding of Implicit Feedback via Group-Aware User Behavior Simulation](g-ubs_towards_robust_understanding_of_implicit_feedback_via_group-aware_user_beh.md)**

:   提出 G-UBS（Group-aware User Behavior Simulation）范式，通过用户群组管理器（UGM）基于 LLM 的"总结-聚类-反思"流程生成群组画像，结合用户反馈建模器（UFM）的群组感知强化学习训练，在隐式反馈噪声下实现鲁棒的用户行为理解，同时构建了首个多模态隐式反馈视频推荐基准 IF-VR。

**[Good-for-MDP State Reduction for Stochastic LTL Planning](good-for-mdp_state_reduction_for_stochastic_ltl_planning.md)**

:   提出一种新的 Good-for-MDP（GFM）自动机状态约简技术，通过 GFM→DBA→DCA→GFG 最小化→0/1-PA 的变换链显著减少自动机状态数量；同时为 $\textsf{GF}\varphi$（$\varphi$ 为 co-safety 公式）类公式提供直接的单指数构造方法，相比一般的双指数构造实现了指数级的状态数减少。

**[HCPO: Hierarchical Conductor-Based Policy Optimization in Multi-Agent Reinforcement Learning](hcpo_hierarchical_conductor-based_policy_optimization_in_multi-agent_reinforceme.md)**

:   提出 HCPO 算法，通过引入 conductor（指挥者）机制增强多智能体联合策略的表达能力和探索效率，构建类似 Gaussian mixture model 的联合策略框架，并证明两级策略更新的单调改进保证。

**[In-Token Rationality Optimization: Towards Accurate and Concise LLM Reasoning via Self-Feedback](in-token_rationality_optimization_towards_accurate_and_concise_llm_reasoning_via.md)**

:   提出 InTRO 框架，通过将模型的生成策略与其answer-conditioned后验对齐（KL散度最小化），在单次前向传播中实现token级探索和自生成反馈，从而在不依赖外部监督的情况下提升LLM推理的准确性与简洁性。

**[InfiGUI-G1: Advancing GUI Grounding with Adaptive Exploration Policy Optimization](infigui-g1_advancing_gui_grounding_with_adaptive_exploration_policy_optimization.md)**

:   针对GUI定位中语义对齐的探索瓶颈，提出Adaptive Exploration Policy Optimization (AEPO)框架，通过多答案生成策略强制广泛探索、自适应探索奖励函数动态引导以及共线惩罚机制确保探索质量，显著提升多模态大模型在复杂GUI定位任务上的表现。

**[Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation](intention-guided_cognitive_reasoning_for_egocentric_long-term_action_anticipatio.md)**

:   提出INSIGHT框架，一个面向第一人称长期动作预测的两阶段统一框架：第一阶段通过手-物交互区域特征提取和动词-名词共现矩阵增强动作表示；第二阶段引入基于GRPO的强化学习认知推理模块，模拟"感知→推理→回答"的结构化认知过程进行意图推断和动作预测。

**[Know your Trajectory -- Trustworthy Reinforcement Learning Deployment through Importance-Based Trajectory Analysis](know_your_trajectory_--_trustworthy_reinforcement_learning_deployment_through_im.md)**

:   提出一种基于状态重要性指标的轨迹级解释框架，通过结合Q值差异和目标亲和度（radical term）对轨迹进行排序，并通过反事实推演验证所选最优轨迹的鲁棒优越性，为RL策略提供"为什么选这条路而非那条路？"的可信解释。

**[Language Model Distillation: A Temporal Difference Imitation Learning Perspective](language_model_distillation_a_temporal_difference_imitation_learning_perspective.md)**

:   从模仿学习/逆强化学习的视角重新审视语言模型蒸馏，提出利用教师模型输出分布的稀疏性（top-p token集中了96%以上概率质量），构建top-p MDP进行时序差分（TD）学习，证明了在缩减动作空间中的最优策略具有可界的次优性保证，并以IQL算法为基础实现的Bellman Distill方法在多个模型家族上超越了现有蒸馏方法。

**[Learning to Generate and Extract: A Multi-Agent Collaboration Framework for Zero-shot Document-level Event Arguments Extraction](learning_to_generate_and_extract_a_multi-agent_collaboration_framework_for_zero-.md)**

:   提出"提议-评估-修改"多智能体协作框架（生成智能体+评估智能体）解决零样本文档级事件论元提取（ZS-DEAE），通过生成智能体合成未见事件的训练数据，评估智能体评分引导强化学习迭代优化，同时提升合成数据质量和抽取性能。

**[ManiLong-Shot: Interaction-Aware One-Shot Imitation Learning for Long-Horizon Manipulation](manilong-shot_interaction-aware_one-shot_imitation_learning_for_long-horizon_man.md)**

:   提出 ManiLong-Shot 框架，通过交互感知的任务分解、不变区域预测和区域匹配三个模块，仅在10个短序列任务上训练即可泛化到20个未见长序列操作任务，单次模仿成功率 30.2%，相对SOTA提升22.8%。

**[MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management](mars_a_meta-adaptive_reinforcement_learning_framework_for_risk-aware_multi-agent.md)**

:   提出 MARS 框架，通过异构多智能体集成（每个智能体有不同风险偏好和 Safety-Critic）与元自适应控制器（MAC）的两层架构，在动态市场条件下实现风险感知的投资组合管理，显著降低最大回撤和波动率。

**[MARS: Multi-Agent Adaptive Reasoning with Socratic Guidance for Automated Prompt Optimization](mars_multi-agent_adaptive_reasoning_with_socratic_guidance_f.md)**

:   提出 MARS 五智能体框架做自动提示优化（APO）：Planner 生成任务特定的优化轨迹，Teacher-Critic-Student 三体进行苏格拉底对话式迭代精炼 prompt（模拟文本空间中的伪梯度下降），Target 执行并反馈，整体建模为 POMDP，在 17 个数据集上平均超越前 SOTA（PE2）6.04%（通用任务）和 6.42%（领域任务），且仅需 1-shot 训练数据。

**[MathSmith: Towards Extremely Hard Mathematical Reasoning by Forging Synthetic Problems with a Reinforced Policy](mathsmith_towards_extremely_hard_mathematical_reasoning_by_forging_synthetic_pro.md)**

:   提出 MathSmith 框架，通过从 PlanetMath 随机抽取数学概念对、采用9种预定义难度策略生成数学题目、并利用 GRPO 强化学习联合优化结构有效性/推理复杂度/答案一致性，生成的高难度合成问题在 AIME 和 OlympiadBench 上显著提升 LLM 数学推理能力。

**[MMhops-R1: Multimodal Multi-hop Reasoning](mmhops-r1_multimodal_multi-hop_reasoning.md)**

:   提出了 MMhops 基准（31K 样本、3-4 跳推理深度）和 MMhops-R1 框架，通过强化学习训练 MLLM 自主规划推理路径、动态调用图像/文本检索器，实现多模态多跳推理，7B 模型超越 72B 基线和现有 mRAG 方法。

**[Object-Centric Latent Action Learning](object-centric_latent_action_learning.md)**

:   提出以物体为中心的潜在动作学习框架，利用自监督的物体分解（VideoSAUR）将场景中任务相关实体与视觉干扰（动态背景等）分离，使潜在动作模型（LAPO）在有干扰的视频中性能退化减少约50%，并通过线性动作探针自动选择控制相关的 slot。

**[Object-Centric World Models for Causality-Aware Reinforcement Learning](object-centric_world_models_for_causality-aware_reinforcement_learning.md)**

:   提出 STICA 框架，通过统一的以物体为中心的 Transformer 架构实现世界模型、策略网络和价值网络，其中世界模型将观测分解为独立物体的隐状态进行 token 级动力学预测，策略和价值网络通过因果注意力机制估计 token 级因果关系实现因果感知决策，在 Safety Gym 和 OCVRL 基准上显著超越 DreamerV3 等 SOTA 方法。

**[One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow](one-step_generative_policies_with_q-learning_a_reformulation_of_meanflow.md)**

:   将MeanFlow重新形式化为残差映射 $g(a_t,b,t) = a_t - u(a_t,b,t)$，实现一步噪声→动作的生成式策略，无需蒸馏或多步ODE积分，可直接与Q-learning联合训练，在OGBench和D4RL的73个任务上取得强性能。

**[Partial Action Replacement: Tackling Distribution Shift in Offline MARL](partial_action_replacement_tackling_distribution_shift_in_offline_marl.md)**

:   提出部分动作替换（PAR）原理，从理论上证明在分解行为策略下分布偏移随偏离智能体数量线性增长（而非联合动作空间的指数增长），并基于此开发 SPaCQL 算法，通过 Q 函数集成的不确定性动态加权不同 PAR 策略，在 Random 和 Medium-Replay 数据集上显著超越所有基线。

**[Perturbing Best Responses in Zero-Sum Games](perturbing_best_responses_in_zero-sum_games.md)**

:   本文研究在零和博弈的最优响应预言机（BRO）中引入随机扰动，证明了随机虚拟对弈（SFP）在纯策略数量 $n$ 上可实现 $O(\frac{\log n}{\varepsilon^2})$ 的期望迭代次数，并提出了随机双重预言机（SDO）算法，在特定博弈结构下同样实现对数级收敛。

**[Provably Efficient Multi-Objective Bandit Algorithms under Preference-Centric Customization](provably_efficient_multi-objective_bandit_algorithms_under_preference-centric_cu.md)**

:   首次从理论角度研究显式用户偏好下的多目标多臂赌博机（MO-MAB）定制化优化问题，提出 PAMO-MAB 框架并针对"未知偏好"和"隐藏偏好"两种场景分别设计 PRUCB-UP 和 PRUCB-HP 算法，通过偏好估计 + 偏好感知优化的双组件框架实现近最优遗憾界，证明了 preference-free 算法在 Pareto 前沿冲突时必然产生 $\Omega(T)$ 线性遗憾。

**[QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation](qimeng-kernel_macro-thinking_micro-coding_paradigm_for_llm-based_high-performanc.md)**

:   提出 MTMC（Macro Thinking Micro Coding）分层框架，通过强化学习驱动轻量LLM产生高层优化策略（Macro Thinking），再由通用LLM逐步实现代码（Micro Coding），将GPU内核生成的正确性和性能问题解耦，在KernelBench上实现近100%准确率和2.2×超越专家优化PyTorch Eager内核的加速。

**[Realistic Curriculum Reinforcement Learning for Autonomous and Sustainable Marine Vessel Navigation](realistic_curriculum_reinforcement_learning_for_autonomous_and_sustainable_marin.md)**

:   提出一个课程强化学习（CRL）框架用于自主且可持续的海洋船舶航行，集成了基于真实AIS数据的仿真环境、扩散模型增强的动态海上交通模拟、以及机器学习燃油消耗预测模块，通过多目标奖励函数同时优化航行安全性、排放减少、时效性和目标完成。

**[Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination](reasoning_or_memorization_unreliable_results_of_reinforcement_learning_due_to_da.md)**

:   本文通过系统性的数据泄露审计揭示了Qwen2.5系列在MATH-500等标准数学基准上存在严重的数据污染问题，指出近期"虚假奖励也能提升数学推理"的发现是污染所致的虚假结论，并构建了完全无泄露的RandomCalculation基准验证只有正确奖励信号才能带来真实的推理提升。

**[Reasoning with Exploration: An Entropy Perspective](reasoning_with_exploration_an_entropy_perspective.md)**

:   本文从熵（entropy）的视角分析LLM中探索性推理行为（关键token、自我反思、稀有行为）与高熵区域的正相关性，提出一种极简的熵基优势函数塑形方法——仅需一行代码修改——即可显著增强LLM的Pass@K推理能力边界。

**[ReGal: A First Look at PPO-based Legal AI for Judgment Prediction and Summarization in India](regal_a_first_look_at_ppo-based_legal_ai_for_judgment_prediction_and_summarizati.md)**

:   本文首次将基于PPO的强化学习（RLAIF）应用于印度法律领域的判决预测与摘要生成任务，虽然性能未超越SFT和商业模型，但作为定位论文（position paper）揭示了RL在法律NLP中的关键挑战与未来方向。

**[Revealing POMDPs: Qualitative and Quantitative Analysis for Parity Objectives](revealing_pomdps_qualitative_and_quantitative_analysis_for_parity_objectives.md)**

:   本文证明了揭示型POMDPs（revealing POMDPs）在奇偶目标（parity objectives）下的极限确定性分析（limit-sure analysis）等价于几乎确定性分析（EXPTIME-complete），且定量分析（quantitative analysis）也可在EXPTIME内完成，解决了该子类的两个重要开放问题。

**[Risk-Sensitive Exponential Actor Critic](risk-sensitive_exponential_actor_critic.md)**

:   针对 entropic risk measure 下 policy gradient 的高方差和数值不稳定问题，推导了完整的 on/off-policy 风险敏感策略梯度定理，并提出 rsEAC 算法，通过 log-domain critic 参数化和梯度归一化裁剪机制实现稳定的风险敏感连续控制。

**[RLSLM: A Hybrid Reinforcement Learning Framework Aligning Rule-Based Social Locomotion Model with Human Social Norms](rlslm_a_hybrid_reinforcement_learning_framework_aligning_rule-based_social_locom.md)**

:   本文提出RLSLM，一种将心理学实验驱动的规则式社交运动模型（SLM）嵌入强化学习奖励函数的混合框架，使智能体在人群环境中高效学习符合人类社交规范的导航策略，VR实验验证其舒适度评分显著优于现有规则式基线。

**[SafeMIL: Learning Offline Safe Imitation Policy from Non-Preferred Trajectories](safemil_learning_offline_safe_imitation_policy_from_non-preferred_trajectories.md)**

:   本文提出SafeMIL，通过将代价函数学习建模为多实例学习（MIL）问题，从有限的非偏好轨迹和大量无标签轨迹中学习安全的模仿策略，在不需要逐步reward/cost标注的情况下，实现约束满足性能比最佳基线提升3.7倍。

**[Scalable Multi-Objective and Meta Reinforcement Learning via Gradient Estimation](scalable_multi-objective_and_meta_reinforcement_learning_via_gradient_estimation.md)**

:   本文提出PolicyGradEx，通过一阶梯度近似和代理模型高效估计任意任务子集上的策略适应性能，构建任务亲和度矩阵并通过凸优化进行任务分组，在多目标RL和元RL基准上平均超越SOTA基线16%，速度提升高达26倍。

**[Speculative Sampling with Reinforcement Learning](speculative_sampling_with_reinforcement_learning.md)**

:   提出 Re-SpS，首个将推测采样（Speculative Sampling）的草稿树超参数优化建模为 MDP 并用强化学习求解的框架，通过特征复用和动作缓存两大设计，在不损失输出保真度的前提下，相比 EAGLE-3 实现最高 1.12× 的额外加速。

**[Start Small, Think Big: Curriculum-based Relative Policy Optimization for Visual Grounding](start_small_think_big_curriculum-based_relative_policy_optimization_for_visual_g.md)**

:   发现 CoT 推理在视觉定位任务中可能适得其反，提出 CuRPO（Curriculum-based Relative Policy Optimization），利用 CoT 长度和 gIoU 奖励作为数据复杂度指标进行课程式 RL 训练，在 RefCOCO 上比 Visual-RFT 提升最高 +12.52 mAP。

**[STELAR-Vision: Self-Topology-Aware Efficient Learning for Aligned Reasoning in Vision](stelar-vision_self-topology-aware_efficient_learning_for_aligned_reasoning_in_vi.md)**

:   提出 STELAR-Vision，一个拓扑感知的视觉语言推理训练框架，通过 TopoAug 数据生成管线引入 Chain/Tree/Graph 多种推理拓扑结构，配合 SFT+RL 后训练，在分布内外数据集上分别提升 9.7% 和最高 28.4% 的准确率，并通过 Frugal Learning 减少 18.1% 的输出长度。

**[TAdaRAG: Task Adaptive Retrieval-Augmented Generation via On-the-Fly Knowledge Graph Construction](tadarag_task_adaptive_retrieval-augmented_generation_via_on-the-fly_knowledge_gr.md)**

:   提出 TAdaRAG，一个任务自适应的 RAG 框架，通过意图驱动的模板路由、监督微调和 REINFORCE 强化学习实现实时知识图谱构建，有效解决传统 RAG 的分块截断幻觉、推理链断裂和无关信息干扰三大问题，在 6 个公开数据集和 1 个商业场景基准上取得 SOTA。

**[Test-driven Reinforcement Learning in Continuous Control](test-driven_reinforcement_learning_in_continuous_control.md)**

:   提出 Test-driven Reinforcement Learning (TdRL) 框架，用多个测试函数（pass-fail 测试定义最优目标 + indicative 测试引导学习）替代单一奖励函数表示任务目标，通过字典序启发式轨迹比较学习回报函数，在 DeepMind Control Suite 上匹配或超越手工奖励方法，天然支持多目标优化。

**[TextShield-R1: Reinforced Reasoning for Tampered Text Detection](textshield-r1_reinforced_reasoning_for_tampered_text_detection.md)**

:   提出 TextShield-R1，首个基于强化学习的多模态大模型篡改文本检测方法，通过取证持续预训练（从自然图像到文本图像的课程）、GRPO 强化学习（五种精心设计的奖励函数减少标注依赖）和 OCR 矫正（利用 MLLM 的文本识别能力提升定位精度），配合新提出的 TFR 基准（45K+ 图像、16 种语言、10 种篡改技术），显著推进了可解释性篡改文本检测的 SOTA。

**[Think, Speak, Decide: Language-Augmented Multi-Agent Reinforcement Learning for Economic Decision-Making](think_speak_decide_language-augmented_multi-agent_reinforcement_learning_for_eco.md)**

:   提出 LAMP 框架，通过 Think–Speak–Decide 三阶段流水线将 LLM 驱动的语言推理与 MARL 策略优化相融合，使经济决策智能体能够理解和利用自然语言信息（如新闻、对话），在经济仿真环境中累计回报超越纯 MARL 基线 63.5%、LLM-only 基线 34.0%。

**[Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction](thinker_training_llms_in_hierarchical_thinking_for_deep_search_via_multi-turn_in.md)**

:   提出 Thinker 框架，通过分层思维（breadth decomposition + depth solving）和双重表征（自然语言 + 逻辑函数）实现结构化的深度搜索推理，配合知识边界判定减少不必要检索，以 SFT 方式训练，在多个 QA 基准上显著超越 RL-based deep search 方法。

**[TowerMind: A Tower Defence Game Learning Environment and Benchmark for LLM as Agents](towermind_a_tower_defence_game_learning_environment_and_benchmark_for_llm_as_age.md)**

:   提出 TowerMind，一个基于塔防游戏的轻量级多模态环境，用于评估 LLM 的长期规划和决策能力，揭示了当前 LLM 与人类专家之间仍存在显著性能差距（最佳模型仅达人类专家 42% 的得分），并识别出规划验证不足、缺乏多终态思维、动作空间利用不充分等行为缺陷。

**[Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach](vision-language_reasoning_for_geolocalization_a_reinforcement_learning_approach.md)**

:   提出 Geo-R，一个无需检索的推理驱动图像地理定位框架，通过 Chain-of-Region 层次化推理范式和基于 Haversine 距离的坐标对齐奖励的强化学习策略，在 IM2GPS3K 上 1km 准确率达 18.10%，超越所有无检索方法并逼近检索方法。

**[Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning](well_begun_half_done_reinforcement_learning_with_prefix_optimization_for_llm_rea.md)**

:   发现 LLM 推理中的"起始锁定效应"（Beginning Lock-in Effect）——初始推理过程显著决定后续轨迹和最终结果，据此提出 PPPO 方法，仅优化前缀 token（约 26% 的 token）即可实现高达 18.02% 的准确率提升，同时减少输出 token 数量达 18.35%。

**[When Eyes and Ears Disagree: Can MLLMs Discern Audio-Visual Confusion?](when_eyes_and_ears_disagree_can_mllms_discern_audio-visual_confusion.md)**

:   发现多模态大语言模型（MLLMs）在音视觉信息不对称时严重受视觉主导而无法识别缺失音频的"音视觉混淆"现象，提出 AV-ConfuseBench 基准和 RL-CoMM 方法（引入外部音频模型做参考的阶梯式推理奖励 + 答案置信度优化），在仅用约 20% 训练数据的情况下提升基线模型准确率 10~30%。

**[Where and What Matters: Sensitivity-Aware Task Vectors for Many-Shot Multimodal In-Context Learning](where_and_what_matters_sensitivity-aware_task_vectors_for_many-shot_multimodal_i.md)**

:   提出 STV 框架，通过激活差值（activation delta）识别对上下文信息敏感的注意力头位置，并利用强化学习从预聚类的激活库中选择最优任务向量进行插入，在不增加输入长度的前提下实现高效的多模态 many-shot 上下文学习。

**[Where to Start Alignment? Diffusion Large Language Model May Demand a Distinct Position](where_to_start_alignment_diffusion_large_language_model_may_demand_a_distinct_po.md)**

:   首次系统分析扩散大语言模型（dLLM）的安全特性，发现与自回归 LLM 不同，dLLM 中**中间 token** 对安全性更关键，且攻击者受限于模型固有的顺序生成倾向难以操控中间 token，基于此不对称性提出 MOSA（Middle-tOken Safety Alignment）防御方法。
