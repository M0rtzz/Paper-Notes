---
title: >-
  ICML2025 强化学习方向 78篇论文解读
description: >-
  78篇ICML2025 强化学习方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎮 强化学习

**🧪 ICML2025** · **78** 篇论文解读

**[A Theoretical Study of (Hyper) Self-Attention through the Lens of Interactions: Representation, Training, Generalization](a_theoretical_study_of_hyper_self-attention_through_the_lens_of_interactions_rep.md)**

:   从"交互实体"统一视角出发，证明单层线性 self-attention 以 $\Theta(|\mathcal{S}|^2)$ 参数高效表示、学习并泛化成对交互函数（全连接网络需 $\Omega(L^2|\mathcal{S}|^2)$），并在此理论基础上提出 HyperFeatureAttention（特征级交互耦合）和 HyperAttention（高阶多实体交互）两个新模块，在语言建模中降低了 perplexity。

**[Action-Constrained Imitation Learning](action-constrained_imitation_learning.md)**

:   形式化了"动作约束模仿学习(ACIL)"新问题——受限Agent从无约束专家学习，提出DTWIL通过MPC+DTW距离生成替代性约束轨迹来消除占用度量失配，在多种机器人任务上显著优于基线。

**[Action-Dependent Optimality-Preserving Reward Shaping (ADOPS)](action-dependent_optimality-preserving_reward_shaping.md)**

:   提出ADOPS方法，通过查询critic网络的外在/内在值函数估计，仅在内在奖励会改变最优动作偏好时调整奖励，从而实现action-dependent的optimality-preserving reward shaping，突破了PBRS只能处理action-independent形式的限制，在Montezuma's Revenge上超越所有先前的optimality-preserving方法和baseline RND。

**[Actor-Critics Can Achieve Optimal Sample Efficiency](actor-critics_can_achieve_optimal_sample_efficiency.md)**

:   本文首次证明 Actor-Critic 算法在一般函数逼近（general function approximation）和需要策略性探索的设定下可以达到 $O(1/\epsilon^2)$ 的最优样本复杂度，通过整合乐观探索、离策略 Critic 估计和稀疏策略切换，并将结果扩展到混合 RL 设定。

**[Adversarial Cooperative Rationalization: The Risk of Spurious Correlations in Even Clean Datasets](adversarial_cooperative_rationalization_the_risk_of_spurious_correlations_in_eve.md)**

:   揭示协作理据化框架（RNP）中的隐蔽缺陷——即使在干净数据集上，生成器的采样偏差也会引入理据与标签间的虚假相关，提出对抗检测+指令干预方法，在文本和图分类上显著超越现有方法。

**[Automatic Reward Shaping from Confounded Offline Data](automatic_reward_shaping_from_confounded_offline_data.md)**

:   提出首个理论上有保障的数据驱动方法，从含未观测混淆因子的离线数据中自动学习基于势的奖励整形函数 (PBRS)，通过因果贝尔曼最优方程上界最优状态值作为势函数，并证明所得 Q-UCB Shaping 算法在伪次优状态-动作对上享有比 vanilla Q-UCB 更优的 gap-dependent regret bound。

**[BEAVER: Building Environments with Assessable Variation for Evaluating Multi-Objective Reinforcement Learning](beaver_building_environments_with_assessable_variation_for_evaluating_multi-obje.md)**

:   提出 BEAVER 基准——首个面向建筑能源管理的多目标上下文强化学习评估框架，通过参数化热动力学和气候区域构建可控环境变化，系统评估现有 MORL 算法的跨环境泛化能力。

**[Benchmarking Quantum Reinforcement Learning](benchmarking_quantum_reinforcement_learning.md)**

:   提出量子强化学习（QRL）的严格基准测试方法论——基于样本复杂度的统计估计器和统计显著性定义的"超越"概念，在新设计的 6G 波束管理环境上进行迄今最大规模（100 seeds）的 QRL vs 经典 RL 比较，发现先前关于 QRL 优越性的声称需要更审慎看待。

**[Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC](beyond_the_rainbow_high_performance_deep_reinforcement_learning_on_a_desktop_pc.md)**

:   提出 BTR（Beyond The Rainbow）——整合 6 项 RL 改进到 Rainbow DQN 中，在单台桌面 PC 上 12 小时内训练 Atari-60 达到 IQM 7.4（Rainbow 为 1.9），并首次成功训练智能体玩马里奥银河、马里奥赛车和真人快打等 3D 游戏。

**[BRITE: Bootstrapping Reinforced Thinking Process to Enhance Language Model Reasoning](brite_bootstrapping_reinforced_thinking_process_to_enhance_language_model_reason.md)**

:   提出 BRITE——通过自举（bootstrapping）方式迭代收集和强化 LLM 的中间思维过程，结合过程级奖励模型和 PPO 训练，持续提升 LLM 在数学推理等任务上的表现。

**[Conceptual Belief-Informed Reinforcement Learning](conceptual_belief-informed_reinforcement_learning.md)**

:   提出 HI-RL（Human Intelligence-RL）——将认知科学中的概念抽象和概率先验信念机制引入 RL，从经验中提取高层概念并构建概念关联的自适应先验来指导值函数/策略更新，作为算法无关插件一致提升 DQN/PPO/SAC/TD3 的样本效率。

**[Continual Reinforcement Learning by Planning with Online World Models](continual_reinforcement_learning_by_planning_with_online_world_models.md)**

:   提出 FTL Online Agent (OA)，通过在线学习的 Follow-The-Leader 浅层世界模型 + 模型预测控制（MPC）规划来实现持续强化学习，该世界模型从构造上免疫灾难性遗忘，且具有理论遗憾界保证 $\mathcal{O}(\sqrt{K^2 D \log(T)})$，在专门设计的 Continual Bench 上全面超越基于深度世界模型的方法。

**[Controlling Underestimation Bias in Constrained Reinforcement Learning for Safe Exploration](controlling_underestimation_bias_in_constrained_reinforcement_learning_for_safe_.md)**

:   提出 MICE（Memory-driven Intrinsic Cost Estimation）——通过闪光灯记忆机制存储历史高代价状态，构建内在代价信号来纠正代价值函数的低估偏差，在约束 RL 的训练过程中显著减少约束违反次数。

**[Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making](counterfactual_effect_decomposition_in_multi-agent_sequential_decision_making.md)**

:   提出一种双层因果分解框架，将多智能体序列决策中某动作的总反事实效应（TCFE）系统地分解为"通过智能体行为传播的效应"（tot-ASE）和"通过状态转移传播的效应"（r-SSE），并分别用 Shapley 值和内在因果贡献（ICC）进一步归因到每个智能体和每个状态变量。

**[Craftium: An Extensible Framework for Creating Reinforcement Learning Environments](craftium_bridging_flexibility_and_efficiency_for_rich_3d_single-_and_multi-agent.md)**

:   Craftium 基于开源 Minetest 游戏引擎构建了一个灵活高效的 3D RL 环境创建框架，通过 Lua API 实现完全自定义，同时提供标准 Gymnasium 接口和五个基准环境。

**[Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination](cross-environment_cooperation_enables_zero-shot_multi-agent_coordination.md)**

:   > 提出跨环境合作（CEC）范式，通过在程序化生成的大量多样化环境中进行自对弈训练（而非增加伙伴多样性），使智能体学习到通用的合作规范，从而在从未见过的新环境中与从未见过的新伙伴实现零样本协调。

**[Decoding Rewards in Competitive Games: Inverse Game Theory with Entropy Regularization](decoding_rewards_in_competitive_games_inverse_game_theory_with_entropy_regulariz.md)**

:   提出基于熵正则化的零和博弈逆问题统一框架，利用 Quantal Response Equilibrium (QRE) 在线性假设下建立奖励函数的可辨识性条件，并给出从观测动作恢复奖励函数的置信集构造算法，附带 $\mathcal{O}(T^{-1/2})$ 收敛速率保证。

**[Demystifying the Paradox of Importance Sampling with an Estimated History-Dependent Behavior Policy in Off-Policy Evaluation](demystifying_the_paradox_of_importance_sampling_with_an_estimated_history-depend.md)**

:   本文从理论上揭示了"在 OPE 中使用估计的历史依赖行为策略比使用真实行为策略反而更好"这一悖论的根本原因——估计行为策略隐式地将 IS 估计器投影到更约束的空间，降低渐近方差但增加有限样本偏差。

**[Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning](divide_and_conquer_grounding_llms_as_efficient_decision-making_agents_via_offlin.md)**

:   GLIDER 引入参数高效的层次化结构——高层策略学习抽象的分步计划并指导低层控制器执行，通过离线层次化 RL 将复杂长时域决策分解为连贯的 CoT 推理子任务，在 ScienceWorld 和 ALFWorld 上取得一致的性能提升和更强的泛化能力。

**[Diving into Self-Evolving Training for Multimodal Reasoning](diving_into_self-evolving_training_for_multimodal_reasoning.md)**

:   通过强化学习视角重新审视多模态推理中的自演化训练（Self-Evolving Training），系统性地分析训练方法、奖励模型和提示变体三大关键因素，并提出基于 Reward-Pass@K 的自适应温度调节机制来缓解训练饱和问题，最终形成 M-STaR 框架，在多个基准上取得一致提升。

**[Embedding Safety into RL: A New Take on Trust Region Methods](embedding_safety_into_rl_a_new_take_on_trust_region_methods.md)**

:   提出 C-TRPO 算法，通过修改策略空间的几何结构（在 KL 散度中嵌入约束感知的障碍项），使信赖域天然只包含安全策略，从而在训练全程保障约束满足，同时保持与 SOTA 相当的回报性能。

**[Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration](enhancing_cooperative_multi-agent_reinforcement_learning_with_state_modelling_an.md)**

:   提出 SMPE² 算法，通过变分推断学习有意义的状态信念表示并结合对抗式内在探索，在部分可观测的合作多智能体环境中显著提升协调能力，在 MPE、LBF、RWARE 三个基准上超越 SOTA。

**[Ergodic Generative Flows](ergodic_generative_flows.md)**

:   提出 Ergodic Generative Flows (EGFs)，通过有限个全局微分同胚构建生成流，利用遍历性 (ergodicity) 保证通用性，并设计 KL-weakFM 损失实现无需独立奖励模型的模仿学习训练，在 NASA 地球科学数据集上以 30 倍更小的模型超越基线。

**[EVOLvE: Evaluating and Optimizing LLMs For In-Context Exploration](evolve_evaluating_and_optimizing_llms_for_in-context_exploration.md)**

:   提出 BanditBench 基准和三种增强策略（推理时算法引导、Few-shot 示范、Oracle 行为微调），系统评估并改善 LLM 在 bandit 环境中的上下文探索能力，使小模型通过算法蒸馏超越大模型。

**[Exploring Large Action Sets with Hyperspherical Embeddings using von Mises-Fisher Sampling](exploring_large_action_sets_with_hyperspherical_embeddings_using_von_mises-fishe.md)**

:   提出 vMF-exp，通过在超球面上采样 von Mises-Fisher 分布向量再做最近邻检索，实现对大规模动作集（百万级）的可扩展探索，理论证明在均匀分布假设下渐近等价于 Boltzmann 探索，并成功部署于 Deezer 音乐推荐系统。

**[Extreme Value Policy Optimization for Safe Reinforcement Learning](extreme_value_policy_optimization_for_safe_reinforcement_learning.md)**

:   提出 EVO 算法，将极值理论 (EVT) 引入约束强化学习，用广义 Pareto 分布 (GPD) 建模代价尾部的极端样本，并设计极端分位数约束与极端优先回放机制，在训练中实现零约束违反的同时保持竞争性策略性能。

**[Fast and Robust: Task Sampling with Posterior and Diversity Synergies for Adaptive Decision-Makers in Randomized Environments](fast_and_robust_task_sampling_with_posterior_and_diversity_synergies_for_adaptiv.md)**

:   提出 PDTS（Posterior and Diversity Synergized Task Sampling），将鲁棒主动任务采样建模为无穷臂老虎机问题，通过后验采样替代 UCB 并引入多样性正则化，以极简实现在 Domain Randomization 和 Meta-RL 中达到接近最坏情况的鲁棒适应性能。

**[Flow of Reasoning: Training LLMs for Divergent Reasoning with Minimal Examples](flow_of_reasoning_training_llms_for_divergent_reasoning_with_minimal_examples.md)**

:   提出 Flow of Reasoning (FoR)，将多步 LLM 推理建模为 DAG 上的马尔可夫流，借助 GFlowNet 的轨迹平衡目标微调 LLM，使其仅用极少训练样本（如15个）即可采样出概率正比于奖励的多条高质量且多样化的推理路径。

**[Gradual Transition from Bellman Optimality Operator to Bellman Operator in Online Reinforcement Learning](gradual_transition_from_bellman_optimality_operator_to_bellman_operator_in_onlin.md)**

:   提出 Annealed Q-learning (AQ-L)，通过期望分位损失（expectile loss）的参数 τ 从接近1退火至0.5，实现从 Bellman 最优算子到 Bellman 算子的平滑过渡，在连续动作空间中既加速了早期学习又抑制了后期过估计偏差，与 TD3/SAC 结合后在多种运动控制和操控任务上显著优于基线。

**[Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning](graph-assisted_stitching_for_offline_hierarchical_reinforcement_learning.md)**

:   提出 Graph-Assisted Stitching (GAS) 框架，用基于图搜索的子目标选择替代显式高层策略学习，通过时间距离表示 (TDR) 空间中的聚类构图与最短路径规划，在离线 HRL 中实现高效的跨轨迹拼接，在最具挑战的 antmaze-giant-stitch 任务上从前 SOTA 的 1.0 飙升至 88.3。

**[Graph-Supported Dynamic Algorithm Configuration for Multi-Objective Combinatorial Optimization](graph-supported_dynamic_algorithm_configuration_for_multi-objective_combinatoria.md)**

:   提出 GS-MODAC，利用 GNN 将目标空间中的解映射为图结构来学习状态表征，结合 PPO 实现对多目标进化算法（MOEA）参数的动态配置，在调度和路由两类 NP-hard 组合优化问题上超越静态和已有 DRL 方法，并展现出跨问题规模和目标数量的泛化能力。

**[Heterogeneous Data Game: Characterizing the Model Competition Across Multiple Data Sources](heterogeneous_data_game_characterizing_the_model_competition_across_multiple_dat.md)**

:   提出"异构数据博弈"框架分析多ML模型提供者在异构数据源上的竞争，揭示三种PNE模式（不存在/同质化/异质化），给出不同选择模型和市场结构下PNE存在性的充分必要条件。

**[Hierarchical Reinforcement Learning with Targeted Causal Interventions](hierarchical_reinforcement_learning_with_targeted_causal_interventions.md)**

:   提出 HRC 框架，将层次强化学习中的子目标关系建模为因果图，通过因果发现算法学习子目标结构，并基于因果效应优先级进行**定向干预**，显著降低长时域稀疏奖励任务的训练代价。

**[Hierarchical Reinforcement Learning with Uncertainty-Guided Diffusional Subgoals](hierarchical_reinforcement_learning_with_uncertainty-guided_diffusional_subgoals.md)**

:   提出 HIDI 框架，以条件扩散模型建模子目标分布，并引入高斯过程 (GP) 先验进行不确定性正则化与子目标选择，在长时域连续控制任务上显著超越现有层次强化学习方法。

**[KEA: Keeping Exploration Alive by Proactively Coordinating Exploration Strategies](kea_keeping_exploration_alive_by_proactively_coordinating_exploration_strategies.md)**

:   提出 KEA 方法，通过引入标准智能体与新颖性增强智能体的动态切换机制，主动协调不同探索策略，解决 SAC 与新颖性探索结合时因策略交互导致的冗余采样和低效探索问题。

**[Learning Dynamics under Environmental Constraints via Measurement-Induced Bundle Structures](learning_dynamics_under_environmental_constraints_via_measurement-induced_bundle.md)**

:   提出一种几何框架，利用测量过程自然诱导的纤维丛结构统一处理测量不确定性、系统约束和动力学学习：在纤维丛上定义测量感知控制屏障函数(mCBF)，结合Neural ODE学习连续时间动力学，在三个机器人控制任务上实现96.3%成功率和99.3%约束满足率。

**[Learning Mean Field Control on Sparse Graphs](learning_mean_field_control_on_sparse_graphs.md)**

:   提出 Local Weak Mean Field Control (LWMFC) 框架，利用局部弱收敛理论将平均场控制扩展到幂律系数 γ>2 的极稀疏图上，配合两系统近似与可扩展 RL 算法，在合成和真实网络上大幅超越基于 Lp graphon 和 graphex 的方法。

**[Learning Progress Driven Multi-Agent Curriculum](learning_progress_driven_multi-agent_curriculum.md)**

:   提出 SPMARL，以基于 TD 误差的学习进度（而非回报）驱动智能体数量的自适应课程分布，解决多智能体稀疏奖励任务中回报估计高方差与信用分配困难两大问题。

**[Learning to Incentivize in Repeated Principal-Agent Problems with Adversarial Agent Arrivals](learning_to_incentivize_in_repeated_principal-agent_problems_with_adversarial_ag.md)**

:   首次研究 agent 以对抗顺序到达的重复 principal-agent 问题，在 greedy 和 smooth 两种响应模型下分别给出了紧的 regret 上下界，核心思路是将激励设计问题规约为对抗线性 bandit。

**[Learning to Trust Bellman Updates: Selective State-Adaptive Regularization for Offline RL](learning_to_trust_bellman_updates_selective_state-adaptive_regularization_for_of.md)**

:   提出选择性状态自适应正则化（SSAR），用神经网络为每个状态动态生成正则化系数，并仅在高质量动作上施加约束，统一了CQL（值正则化）和TD3+BC（策略约束）两大离线RL范式，在D4RL离线和O2O场景均大幅超越基线。

**[Learning Utilities from Demonstrations in Markov Decision Processes](learning_utilities_from_demonstrations_in_markov_decision_processes.md)**

:   本文提出 Utility Learning (UL) 问题，通过从演示中推断智能体的效用函数来捕捉其风险态度，设计了两个可证明高效的算法并分析了样本复杂度和可辨识性。

**[Leveraging Skills from Unlabeled Prior Data for Efficient Online Exploration](leveraging_skills_from_unlabeled_prior_data_for_efficient_online_exploration.md)**

:   提出 SUPE 方法，将无标签离线轨迹数据"用两次"——既用于 VAE 技能预训练，又通过 UCB 伪标签转化为高层 off-policy 数据加速在线探索，在 42 个稀疏奖励任务上全面超越已有方法。

**[LineFlow: A Framework to Learn Active Control of Production Lines](lineflow_a_framework_to_learn_active_control_of_production_lines.md)**

:   提出 LineFlow，一个可扩展的开源 Python 框架，用于模拟任意复杂度的生产线并训练 RL 智能体进行主动产线控制（自适应路由、工人重分配、调度等），同时给出了若干子问题的数学最优解作为基准。

**[Mastering Massive Multi-Task Reinforcement Learning via Mixture-of-Expert Decision Transformer](mastering_massive_multi-task_reinforcement_learning_via_mixture-of-expert_decisi.md)**

:   提出 M3DT 框架，将 MoE 引入 Decision Transformer 实现参数分离——通过任务分组让每个专家只学习一个小任务子集的特定知识，配合三阶段训练机制（骨干→专家→路由器）避免梯度冲突，增加专家数既扩展参数又降低任务负载，成功将离线多任务 RL 扩展到 160 个仿真控制任务。

**[Maximum Total Correlation Reinforcement Learning](maximum_total_correlation_reinforcement_learning.md)**

:   提出最大化轨迹总相关（Total Correlation）作为 RL 的归纳偏置，鼓励策略产生简单、可压缩的轨迹，从而在不牺牲任务性能的前提下显著提升对观测噪声、动作噪声和动力学变化的零样本鲁棒性。

**[Meta-Black-Box-Optimization through Offline Q-function Learning (Q-Mamba)](meta-black-box-optimization_through_offline_q-function_learning.md)**

:   提出 Q-Mamba，首个离线 MetaBBO 框架，通过 Q 函数分解 + 保守 Q 学习 + Mamba 架构，在不到在线方法一半训练预算下达到可比甚至更优的 BBO 算法配置性能。

**[Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn](mitigating_plasticity_loss_in_continual_reinforcement_learning_by_reducing_churn.md)**

:   通过 NTK 矩阵建立可塑性丧失 (plasticity loss) 与 churn（批外数据输出漂移）之间的因果联系，提出 C-CHAIN 方法在持续 RL 训练中持续抑制 churn，从而缓解可塑性丧失，在 24 个持续 RL 环境上超越已有基线。

**[Non-stationary Online Learning for Curved Losses: Improved Dynamic Regret via Mixability](non-stationary_online_learning_for_curved_losses_improved_dynamic_regret_via_mix.md)**

:   利用 mixability（可混合性）概念替代传统 KKT 分析，提出基于指数权重+fixed-share更新的连续空间在线学习框架，将弯曲损失函数（squared/logistic loss）的动态遗憾中对维度 $d$ 的依赖从 $O(d^{10/3})$ 大幅改进至 $O(d)$。

**[Of Mice and Machines: A Comparison of Learning Between Real World Mice and RL Agents](of_mice_and_machines_a_comparison_of_learning_between_real_world_mice_and_rl_age.md)**

:   系统比较真实小鼠与RL智能体在捕食者-猎物迷宫中的行为差异，发现RL缺乏自我保护本能，提出创伤启发安全缓冲（TISB）和方差惩罚TD学习（VP-TDMPC-2）两种机制，将智能体与小鼠的状态访问重叠率从20.9%提升至86.1%。

**[On the Dynamic Regret of Following the Regularized Leader: Optimism with History Pruning](on_the_dynamic_regret_of_following_the_regularized_leader_optimism_with_history_.md)**

:   本文提出 OptFPRL 算法，通过在 Follow the Regularized Leader (FTRL) 框架中引入**历史梯度裁剪 (History Pruning)** 机制，首次为 FTRL 在紧凑集上建立了数据依赖的动态遗憾保证，动态遗憾完全由预测误差调控，在预测完美时可达零遗憾。

**[Online Pre-Training for Offline-to-Online Reinforcement Learning](online_pre-training_for_offline-to-online_reinforcement_learning.md)**

:   提出 OPT 方法，在离线预训练和在线微调之间引入"在线预训练"阶段，通过新增一个独立值函数并用元适应目标训练，解决离线预训练智能体因值估计不准而导致在线微调性能下降的问题，在 D4RL 基准上平均提升约 30%。

**[Optimal and Practical Batched Linear Bandit Algorithm](optimal_and_practical_batched_linear_bandit_algorithm.md)**

:   BLAE 通过将**臂消除策略**与**正则化 G-最优设计**深度融合，首次在批量线性 Bandit 问题中**同时**实现了 large-$K$ 和 small-$K$ 两种体制下的极小极大最优遗憾（仅差 $\log T$ 因子），同时保持 $\mathcal{O}(\log\log T)$ 的最低批次复杂度和卓越的实际性能。

**[Optimizing Language Models for Inference Time Objectives using Reinforcement Learning](optimizing_language_models_for_inference_time_objectives_using_reinforcement_lea.md)**

:   提出在 RL 训练阶段显式优化推理时 k-sample 目标（pass@k / majority voting），通过 leave-one-out 控制变量构造无偏低方差梯度估计，在 MATH 和 CodeContests 上显著提升推理时性能。

**[Pessimism Principle Can Be Effective: Towards a Framework for Zero-Shot Transfer RL](pessimism_principle_can_be_effective_towards_a_framework_for_zero-shot_transfer_.md)**

:   提出基于悲观主义原则的迁移RL框架：用鲁棒MDP构建目标域性能保守下界作为代理目标优化，设计Averaged Operator和Minimal Pessimism两种代理及分布式算法，确保安全迁移并避免负迁移。

**[PIGDreamer: Privileged Information Guided World Models for Safe Partially Observable RL](pigdreamer_privileged_information_guided_world_models_for_safe_partially_observa.md)**

:   提出 ACPOMDPs 理论框架并构建 PIGDreamer，在训练阶段利用特权信息（如底层状态、传感器数据）通过表征对齐、特权预测器和非对称 Critic 三种方式增强基于世界模型的安全 RL，在部分可观测环境中以仅 28% 的额外训练时间获得 136% 的性能提升。

**[Position: Lifetime Tuning is Incompatible with Continual Reinforcement Learning](position_lifetime_tuning_is_incompatible_with_continual_reinforcement_learning.md)**

:   这篇 position paper 指出持续强化学习研究中的关键方法论缺陷——lifetime tuning（在整个生命周期上调参）会掩盖算法的真实持续学习能力，并提出 k%-percent tuning 作为更合理的评估替代方案。

**[Principal-Agent Bandit Games with Self-Interested and Exploratory Learning Agents](principal-agent_bandit_games_with_self-interested_and_exploratory_learning_agent.md)**

:   本文研究重复委托-代理赌臂博弈中，代理基于经验均值做决策（而非已知真实均值）且可能随机探索时，如何设计委托人的激励算法使后悔界达到 $\tilde{O}(\sqrt{T})$ 或 $\tilde{O}(T^{2/3})$，显著优于先前 $\tilde{O}(T^{11/12})$ 的结果。

**[ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification](revise_learning_to_refine_at_test-time_via_intrinsic_self-verification.md)**

:   提出 ReVISE 框架，通过引入 `[refine]` 特殊 token 和两阶段课程学习（先学自验证、再学自纠错），使 LLM 在推理时能内省式地验证并修正自身推理轨迹，无需外部验证器或复杂 RL 训练。

**[Reward-free World Models for Online Imitation Learning](reward-free_world_models_for_online_imitation_learning.md)**

:   提出 IQ-MPC，一种无需显式奖励建模的世界模型在线模仿学习方法，通过逆软Q学习在潜空间中联合学习动态模型与Q函数，利用 MPPI 规划实现对高维观测和复杂动力学任务的稳定专家级模仿。

**[Robot-Gated Interactive Imitation Learning with Adaptive Intervention Mechanism](robot-gated_interactive_imitation_learning_with_adaptive_intervention_mechanism.md)**

:   提出自适应干预机制 AIM，通过学习代理 Q 函数模拟人类干预决策，让机器人主动请求专家帮助，相比不确定性基线 Thrifty-DAgger 在人类接管成本和学习效率上提升 40%。

**[Robust Noise Attenuation via Adaptive Pooling of Transformer Outputs](robust_noise_attenuation_via_adaptive_pooling_of_transformer_outputs.md)**

:   本文将 Transformer 输出的池化操作形式化为向量量化问题，证明 AvgPool 和 MaxPool 在信噪比 (SNR) 变化时存在性能崩溃风险，并提出基于交叉注意力的自适应池化方法 (AdaPool)，在理论上可在任意 SNR 下逼近信号最优量化器，在 RL、关系推理和视觉任务中均表现出优越的鲁棒性。

**[Robust Offline Reinforcement Learning with Linearly Structured f-Divergence Regularization](robust_offline_reinforcement_learning_with_linearly_structured_f-divergence_regu.md)**

:   提出 d-rectangular linear RRMDP (d-RRMDP) 框架，将潜在线性结构同时引入转移核和 f-散度正则化，设计 R2PVI 算法在离线数据下学习鲁棒策略，证明了 instance-dependent 的次优性上界，并通过信息论下界验证算法接近最优。

**[Safety Certificate against Latent Variables with Partially Unidentifiable Dynamics](safety_certificate_against_latent_variables_with_partially_unidentifiable_dynami.md)**

:   提出基于概率空间不变性条件的安全证书设计方法，利用因果强化学习从含潜变量的离线数据中学习边际化 Q 函数，在离线与在线统计分布不一致的情况下仍能保证长期安全性，并证明了安全动作的持续可行性。

**[Scaling Value Iteration Networks to 5000 Layers for Extreme Long-Term Planning](scaling_value_iteration_networks_to_5000_layers_for_extreme_long-term_planning.md)**

:   提出 Dynamic Transition VIN (DT-VIN)，通过引入动态转移核增强隐式 MDP 的表征能力，并设计自适应 highway loss 缓解梯度消失，将 VIN 成功扩展至 5000 层，在 $100 \times 100$ 迷宫中实现 1800 步长期规划（原版 VIN 仅支持 $25 \times 25$ 迷宫中 120 步规划）。

**[SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models](sensei_semantic_exploration_guided_by_foundation_models_to_learn_versatile_world.md)**

:   提出 SENSEI 框架：利用 VLM 成对比较观测图像的"有趣程度"，蒸馏出语义内在奖励，再与集成不确定性驱动的新颖性奖励结合，通过世界模型实现语义有意义的无任务探索，并显著加速下游任务学习。

**[Sliding Puzzles Gym: A Scalable Benchmark for State Representation in Visual Reinforcement Learning](sliding_puzzles_gym_a_scalable_benchmark_for_state_representation_in_visual_rein.md)**

:   本文提出 Sliding Puzzles Gym (SPGym)，一个将经典 8-拼图改造为视觉 RL 任务的基准，通过独立调节图片池大小来精确控制视觉表征学习的复杂度，实验揭示当前方法在视觉多样性增大时的根本性记忆化局限。

**[Solving Zero-Sum Convex Markov Games](solving_zero-sum_convex_markov_games.md)**

:   本文首次为两人零和凸马尔可夫博弈（cMG）中的独立策略梯度方法提供了全局收敛到Nash均衡的理论保证，通过非凸正则化将问题化归为非凸-pPL min-max优化，并设计了嵌套/交替策略梯度算法。

**[Stealing That Free Lunch: Exposing the Limits of Dyna-Style Reinforcement Learning](stealing_that_free_lunch_exposing_the_limits_of_dyna-style_reinforcement_learnin.md)**

:   本文揭示 Dyna 风格模型强化学习算法（MBPO、ALM）在 OpenAI Gym 表现优异但在 DeepMind Control Suite (DMC) 中**严重失效**的现象，系统分析模型误差、过估计偏差和可塑性损失等原因，发现即使使用完美模型 MBPO 也无法一致超越 SAC，表明"没有免费午餐"。

**[Stochastic Encodings for Active Feature Acquisition](stochastic_encodings_for_active_feature_acquisition.md)**

:   本文提出 SEFA (Stochastic Encodings for Feature Acquisition)，一种基于随机潜变量模型的主动特征获取方法，通过在正则化潜空间中跨多种未观测特征实现进行推理来替代 RL 和贪心 CMI 最大化，在合成和真实数据集（含癌症分类）上一致超越所有基线。

**[T1: Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling](t1_advancing_language_model_reasoning_through_reinforcement_learning_and_inferen.md)**

:   T1 通过合成包含 trial-and-error 和 self-verification 的 CoT 数据进行 SFT 初始化，再结合过采样、熵奖励和动态锚点正则化来扩展 RL 训练，使开源 LLM 在复杂数学推理上超越 QwQ-32B-Preview 等模型，并展现出推理时间缩放（inference scaling）行为。

**[Test-Time Adaptation with Binary Feedback](test-time_adaptation_with_binary_feedback.md)**

:   本文提出 BiTTA，一个利用二元反馈（正确/错误）的测试时自适应框架，通过强化学习驱动的双路径优化策略，在严重域偏移下以最小标注成本实现 13.3% 的准确率提升。

**[The Challenge of Teaching Reasoning to LLMs Without RL or Distillation](the_challenge_of_teaching_reasoning_to_llms_without_rl_or_distillation.md)**

:   仅用 20 个来自推理模型 QwQ-32B-Preview 的长 CoT 样例轻量微调 Qwen2.5-32B 就能超越 72B 的数学指令模型，但用非推理模型或人工生成的 CoT 无法达到同等效果，表明推理 CoT 中存在难以复制的"潜在质量"。

**[LEAST: The Courage to Stop — Overcoming Sunk Cost Fallacy in Deep RL](the_courage_to_stop_overcoming_sunk_cost_fallacy_in_deep_reinforcement_learning.md)**

:   提出 Learn to Stop（LEAST），一种轻量级自适应 episode 提前终止机制：维护最近 K 个 episode 的 Q 值和梯度幅值缓冲区，用步级中位数构造质量阈值 $\epsilon_i$ 和学习潜力权重 $\omega_i$，当当前 Q 值低于 $\omega_i \times \epsilon_i$ 时终止并重置；在 MuJoCo 四任务上为 TD3/SAC/REDQ 均带来显著提升（归一化分数从 0.65 提升到 0.70+），DMC 视觉 RL 的 Finger Turn Hard 任务收敛速度加快约 30%。

**[The Impact of On-Policy Parallelized Data Collection on Deep Reinforcement Learning Networks](the_impact_of_on-policy_parallelized_data_collection_on_deep_reinforcement_learn.md)**

:   系统研究 on-policy RL 中并行数据采集的两个维度（并行环境数 $N_{\text{envs}}$ vs 轨迹长度 $N_{\text{RO}}$）对 PPO 性能的影响，发现在固定数据预算下增加并行环境数比增加轨迹长度更有效，且更大的数据集可改善网络可塑性和优化稳定性。

**[The Sample Complexity of Online Strategic Decision Making with Information Asymmetry and Knowledge Transportability](the_sample_complexity_of_online_strategic_decision_making_with_information_asymm.md)**

:   在信息不对称（代理拥有隐私类型和动作作为混淆变量）且需要跨分布知识迁移的在线强化学习场景中，提出基于非参数工具变量（NPIV）方法的模型算法 OPME，证明以 $\tilde{O}(1/\epsilon^2)$ 样本复杂度学得 $\epsilon$-最优策略，并匹配对应下界。

**[VinePPO: Refining Credit Assignment in RL Training of LLMs](vineppo_refining_credit_assignment_in_rl_training_of_llms.md)**

:   VinePPO 利用语言环境可从任意中间状态重置的特性，用蒙特卡洛 (MC) rollout 替换 PPO 中的 value network 进行无偏值估计，在数学推理任务上以更少的墙钟时间（最高 3 倍加速）超越 PPO/GRPO/RLOO 的峰值性能，并展现出更强的泛化斜率。

**[Wasserstein Policy Optimization](wasserstein_policy_optimization.md)**

:   提出 Wasserstein Policy Optimization (WPO)，将最优传输理论中的 Wasserstein 梯度流投影到参数空间，得到一种兼具确定性策略梯度（DPG）利用动作值梯度和经典随机策略梯度（SPG）支持任意分布的闭式更新规则，无需重参数化技巧。

**[Zero-Shot Generalization of Vision-Based RL Without Data Augmentation](zero-shot_generalization_of_vision-based_rl_without_data_augmentation.md)**

:   提出 ALDA（Associative Latent DisentAnglement），通过解耦表示学习+联想记忆机制实现视觉RL在未见环境中的零样本泛化，无需数据增强即可媲美使用千万级外部数据的方法。
