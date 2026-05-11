---
title: >-
  ICLR2026 强化学习方向139篇论文解读
description: >-
  139篇ICLR2026的强化学习方向论文解读，涵盖强化学习、推理、LLM、对抗鲁棒、Agent、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎮 强化学习

**🔬 ICLR2026** · **139** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (34)](../../ACL2026/reinforcement_learning/index.md) · [📷 CVPR2026 (19)](../../CVPR2026/reinforcement_learning/index.md) · [🤖 AAAI2026 (71)](../../AAAI2026/reinforcement_learning/index.md) · [🧠 NeurIPS2025 (172)](../../NeurIPS2025/reinforcement_learning/index.md) · [📹 ICCV2025 (7)](../../ICCV2025/reinforcement_learning/index.md) · [🧪 ICML2025 (81)](../../ICML2025/reinforcement_learning/index.md)

🔥 **高频主题：** 强化学习 ×44 · 推理 ×23 · LLM ×14 · 对抗鲁棒 ×14 · Agent ×12

**[A Unifying View of Coverage in Linear Off-Policy Evaluation](a_unifying_view_of_coverage_in_linear_off-policy_evaluation.md)**

:   提出了一种新的覆盖性参数——**特征-动态覆盖**（feature-dynamics coverage），通过工具变量视角对经典算法 LSTDQ 进行新颖的有限样本分析，统一了线性离策略评估中各种不同覆盖性定义，解决了该领域长期存在的碎片化问题。

**[AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking](abstral_augmenting_llms_reasoning_by_reinforcing_abstract_thinking.md)**

:   提出 AbstRaL，通过强化学习教 LLM 学习推理问题的数学抽象（将具体数字/名称替换为符号变量、提取通用公式），然后用符号求解器推导答案，在 GSM 扰动 benchmark 上几乎完全消除了分布偏移导致的性能下降，并在 OOD 数学/通用推理任务上也有隐式提升。

**[AMPED: Adaptive Multi-objective Projection for balancing Exploration and skill Diversification](amped_adaptive_multi-objective_projection_for_balancing_exploration_and_skill_di.md)**

:   提出AMPED框架，在技能预训练阶段用梯度手术（PCGrad）平衡探索（熵+RND）和技能多样性（AnInfoNCE）之间的梯度冲突，在微调阶段用SAC-based技能选择器自适应选择最优技能，在Maze和URLB基准上超越DIAYN/CeSD/CIC等SBRL基线。

**[APPLE: Toward General Active Perception via Reinforcement Learning](apple_toward_general_active_perception_via_reinforcement_learning.md)**

:   提出APPLE——一种结合强化学习与监督学习的通用主动感知框架，将主动感知建模为POMDP，奖励函数设计为RL奖励减去预测损失，梯度自然分解为策略梯度和预测损失梯度两部分，基于off-policy算法（SAC/CrossQ）和共享ViViT骨干网络，在5个不同任务基准上验证通用性，其中CrossQ变体无需逐任务调参且训练效率提高53%。

**[ARM-FM: Automated Reward Machines via Foundation Models for Compositional Reinforcement Learning](arm-fm_automated_reward_machines_via_foundation_models_for_compositional_reinfor.md)**

:   提出ARM-FM框架，利用基础模型（GPT-4o等）从自然语言任务描述自动生成语言对齐奖励机器（LARM）——包括自动机结构、可执行标签函数和每个状态的自然语言描述——为RL agent提供组合式密集奖励信号，在MiniGrid/Craftium(3D Minecraft)/Meta-World等环境中解决标准RL完全无法学习的稀疏奖励长程任务，并实现零样本任务泛化。

**[AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization](autoqd_automatic_discovery_of_diverse_behaviors_with_quality-diversity_optimizat.md)**

:   提出 AutoQD，通过随机傅里叶特征（RFF）将策略的占据度量嵌入有限维空间，再用加权 PCA 降维得到行为描述符，实现无需人工设计 BD 的 QD 优化，在 6 个连续控制任务上全面超越手工 BD 和现有无监督 QD 方法。

**[AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints](autotool_automatic_scaling_of_tool-use_capabilities_in_rl_via_decoupled_entropy_.md)**

:   提出解耦自适应熵约束 (Decoupled Adaptive Entropy Constraints) 的强化学习策略，使 LLM 在工具调用任务中根据问题难度自动切换长/短推理模式，在提升 9.8% 准确率的同时减少约 81% 的推理 token 开销。

**[AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](awm_accurate_weight-matrix_fingerprint_for_large_language_models.md)**

:   提出 AWM，一种无需训练的 LLM 权重矩阵指纹方法，利用线性分配问题（LAP）恢复嵌入层的置换和符号翻转，再用无偏 CKA 消除 Q/K 矩阵的正交变换影响，在 150 对 LLM 上实现完美 AUC（1.0），对 SFT、持续预训练（5.5T token）、RL、多模态扩展、剪枝、upcycling 六类后训练均鲁棒，30 秒内完成。

**[BA-MCTS: Bayes Adaptive Monte Carlo Tree Search for Offline Model-based RL](bayes_adaptive_monte_carlo_tree_search_for_offline_model-based_reinforcement_lea.md)**

:   首次将贝叶斯自适应 MDP（BAMDP）引入离线模型基 RL，提出 Continuous BAMCP 解决连续状态/动作空间的贝叶斯规划，结合悲观奖励惩罚和搜索基策略迭代（"RL + Search"范式），在 D4RL 12 个任务上显著超越 19 个基线（Cohen's $d > 1.8$），并成功应用于核聚变 tokamak 控制。

**[Boolean Satisfiability via Imitation Learning](boolean_satisfiability_via_imitation_learning.md)**

:   提出 ImitSAT，首个基于模仿学习的 CDCL 求解器分支策略：通过将求解器运行压缩为无冲突的 KeyTrace 专家序列，将分支决策建模为前缀条件的自回归预测任务，以少量查询预算显著减少传播次数和求解时间，并在结构化 SAT 问题上展现良好泛化能力。

**[Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains?](breaking_barriers_do_reinforcement_post_training_gains_transfer_to_unseen_domain.md)**

:   通过观察性研究（18 个开源 RPT 模型）和干预性研究（单域 GRPO 训练），系统揭示了强化后训练（RPT/RLVR）的泛化局限：RPT 在训练域内提升显著，但跨域泛化不一致——结构化域（数学↔代码）可互相迁移，但无法泛化到非结构化域（法律/金融/医疗），且这一结论跨算法、模型规模和训练步数保持一致。

**[Chain-of-Context Learning: Dynamic Constraint Understanding for Multi-Task VRPs](chain-of-context_learning_dynamic_constraint_understanding_for_multi-task_vrps.md)**

:   提出 Chain-of-Context Learning (CCL)，通过 Relevance-Guided Context Reformulation（RGCR，自适应聚合约束信息构建上下文）和 Trajectory-Shared Node Re-embedding（TSNR，跨轨迹共享节点更新避免冗余计算）实现逐步动态的约束感知解码，在 48 种 VRP 变体（16 分布内 + 32 分布外）上全面超越现有方法。

**[Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models](co-rewarding_stable_self-supervised_rl_for_eliciting_reasoning_in_large_language.md)**

:   Co-rewarding 提出自监督 RL 框架，通过数据侧（对比改写问题的跨视角一致性）和模型侧（EMA 教师模型提供伪标签）两种互补监督方式，解决自奖励 RL 中的训练崩溃问题，在无人工标签条件下多项数学推理基准上达到甚至超过 RLVR（有标签）的性能。

**[Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning](continuous-time_value_iteration_for_multi-agent_reinforcement_learning.md)**

:   提出 VIP（Value Iteration via PINN）框架，首次将物理信息神经网络（PINN）用于求解连续时间多智能体强化学习中的 HJB 偏微分方程，并引入 Value Gradient Iteration（VGI）模块迭代精炼价值梯度，在连续时间 MPE 和 MuJoCo 多智能体任务上始终优于离散时间和连续时间基线。

**[Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)**

:   CalibRL 将专家数据重新定义为分布校准基线（而非严格模仿目标），通过 LeakyReLU 不对称激活 + 优势加权实现对 MLLM 推理训练中探索-利用平衡的精细控制，解决 RLVR 中的熵崩溃问题，在几何推理等任务上大幅超越 GRPO/DAPO。

**[Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets](cross-embodiment_offline_reinforcement_learning_for_heterogeneous_robot_datasets.md)**

:   系统研究跨形态离线 RL 预训练范式，发现次优数据比例和机器人多样性增加时梯度冲突导致负迁移，提出基于形态图距离的 Embodiment Grouping（EG）策略将机器人按形态聚类后分组更新 actor，在 16 种机器人平台的 locomotion benchmark 上显著缓解负迁移（70% 次优数据集上 IQL+EG 比 IQL 提升 34%）。

**[CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning](cuda-l1_improving_cuda_optimization_via_contrastive_reinforcement_learning.md)**

:   提出 CUDA-L1，一个基于对比强化学习（Contrastive RL）的三阶段流水线框架，将初始 CUDA 能力较弱的 LLM 训练为高效的 CUDA 优化器，在 KernelBench 的 250 个 CUDA 内核上实现平均 3.12× 加速，峰值达 120×，并可跨 GPU 架构迁移。

**[Deep SPI: Safe Policy Improvement via World Models](deep_spi_safe_policy_improvement_via_world_models.md)**

:   构建了安全策略改进（SPI）的理论框架，将世界模型和表示学习与策略更新保证统一起来：通过基于重要性比率的邻域算子约束策略更新，确保单调改进和收敛；结合局部转移/奖励损失控制世界模型质量和表示稳定性，提出 DeepSPI 算法在 ALE-57 基准上匹配或超越 PPO 和 DeepMDP。

**[Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization](distributionally_robust_cooperative_multi-agent_reinforcement_learning_via_robus.md)**

:   提出 Distributionally Robust IGM (DrIGM) 原则，将分布鲁棒优化引入协作多智能体 RL 的值分解框架，使得 VDN/QMIX/QTRAN 等经典方法能够在训练环境与部署环境存在分布偏移时仍保持稳健的去中心化执行性能。

**[DiVE-k: Differential Visual Reasoning for Fine-grained Image Recognition](dive-k_differential_visual_reasoning_for_fine-grained_image_recognition.md)**

:   提出 DiVE-k 框架，利用大视觉语言模型自身的 top-k 生成结果构造选择题，通过 GRPO 强化学习训练模型进行差异化视觉推理，在细粒度图像识别的 base-to-novel 泛化上大幅超越现有方法。

**[Divide, Harmonize, Then Conquer It: Shooting Multi-Commodity Flow Problems with Multimodal Language Models](divide_harmonize_then_conquer_it_shooting_multi-commodity_flow_problems_with_mul.md)**

:   提出 Pram 框架，首次利用多模态语言模型（MLM）求解多商品流（MCF）问题，通过分区将原问题分解为子问题，以多智能体强化学习（MARL）协调各子问题的全局一致性，理论证明收敛到最优解，实测速度比 LP 快 1-2 个数量级且性能接近最优。

**[Don't Just Fine-tune the Agent, Tune the Environment](dont_just_fine-tune_the_agent_tune_the_environment.md)**

:   提出 Environment Tuning 训练范式，通过结构化课程、可操作的环境增强反馈和细粒度进度奖励，使 LLM agent 仅用 400 个训练样本即可从零学会复杂的多轮工具使用，同时实现优异的分布外泛化能力。

**[Dual-Robust Cross-Domain Offline Reinforcement Learning Against Dynamics Shifts](dual-robust_cross-domain_offline_reinforcement_learning_against_dynamics_shifts.md)**

:   首次在跨域离线 RL 中同时处理训练时鲁棒性（源域-目标域动力学不匹配）和测试时鲁棒性（部署环境动力学偏移）：提出 DROCO 算法，核心是 Robust Cross-Domain Bellman (RCB) 算子——对源域数据施加鲁棒 Bellman 更新、对目标域数据施加标准 in-sample 更新，并通过对偶重构将不可处理的动力学不确定性映射为状态空间扰动，在 D4RL 基准上总分 1105.2 超越次优方法 14%，且在 hard 级别动力学扰动下性能退化仅为基线的一半。

**[Dual Goal Representations](dual_goal_representations.md)**

:   提出"对偶目标表征"（dual goal representation），用"所有状态到目标状态的时间距离集合"来编码目标，理论证明该表征对最优策略恢复是充分的且天然过滤外生噪声，并设计基于非对称内积参数化的实用学习算法，在 OGBench 20 个任务上作为可插拔模块一致提升三种主流离线 GCRL 方法的性能。

**[DVLA-RL: Dual-Level Vision-Language Alignment with Reinforcement Learning Gating for Few-Shot Learning](dvla-rl_dual-level_vision-language_alignment_with_reinforcement_learning_gating_.md)**

:   提出 DVLA-RL 框架，通过双层语义构建（DSC）生成互补的低层属性和高层描述，并以 RL 门控注意力（RLA）动态平衡自注意力和交叉注意力在不同网络层的贡献，实现从低层到高层的层次化视觉-语言对齐，在 9 个少样本学习基准上达到 SOTA。

**[Echo: Towards Advanced Audio Comprehension via Audio-Interleaved Reasoning](echo_towards_advanced_audio_comprehension_via_audio-interleaved_reasoning.md)**

:   提出音频交错推理（audio-interleaved reasoning）新范式，将音频视为推理过程中的主动组件而非静态上下文，使 LALM 在推理时动态定位并重新聆听音频片段。通过 SFT+RL 两阶段训练框架和结构化数据生成流水线，构建 Echo 模型，在专家级和通用音频理解基准上超越 GPT-4o 和 Gemini-2.0-Flash。

**[Efficient Estimation of Kernel Surrogate Models for Task Attribution](efficient_estimation_of_kernel_surrogate_models_for_task_attribution.md)**

:   提出核代理模型（KernelSM）用于任务归因，通过 RBF 核岭回归捕获任务间的非线性交互效应，结合梯度投影的高效估计算法避免重复训练，在数学推理、上下文学习和多目标 RL 等场景下相比线性代理和影响函数基线提升 25% 相关性。

**[EGG-SR: Embedding Symbolic Equivalence into Symbolic Regression via Equality Graph](egg-sr_embedding_symbolic_equivalence_into_symbolic_regression_via_equality_grap.md)**

:   提出 Egg-SR 统一框架，通过等价图（e-graph）将符号等价性嵌入 MCTS、DRL 和 LLM 三类符号回归方法中，分别实现子树剪枝、梯度方差降低和反馈提示增强。理论证明 Egg-MCTS 收紧遗憾界、Egg-DRL 降低梯度估计方差，实验验证一致提升表达式发现精度。

**[Emergence of Spatial Representation in an Actor-Critic Agent with Hippocampus-Inspired Sequence Generator](emergence_of_spatial_representation_in_an_actor-critic_agent_with_hippocampus-in.md)**

:   受海马体 CA3 区内在递归回路启发，提出最小序列生成器（shift register）与 actor-critic 结合，在稀疏视觉输入下实现迷宫导航，同时涌现出位置场、DG 正交化、距离相关空间核和任务依赖重映射等神经生物学现象。

**[Entropy-Preserving Reinforcement Learning (REPO / ADAPO)](entropy-preserving_reinforcement_learning.md)**

:   本文揭示了策略梯度 RL 算法在 LLM 后训练中系统性导致策略熵坍缩的理论根因（优势函数与对数概率的正相关性），并提出两种互补的解法：REPO（通过修改优势函数去相关）和 ADAPO（自适应非对称裁剪），在交互式工具使用任务上实现 SOTA 性能。

**[ExGRPO: Learning to Reason from Experience](exgrpo_learning_to_reason_from_experience.md)**

:   首次系统研究什么样的推理经验对RLVR最有价值，发现中等难度问题+低熵轨迹最有效，据此提出ExGRPO框架进行经验管理和混合策略优化，在数学推理上平均+3.5分，通用推理+7.6分。

**[Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward](exploration_vs_exploitation_rethinking_rlvr_through_clipping_entropy_and_spuriou.md)**

:   通过理论推导和跨模型实验，证明 RLVR 中裁剪偏差提供的学习信号可忽略不计（≤1/17），真正起作用的是裁剪对策略熵的隐式压缩效应，并提出奖励误标模型解释为何随机奖励能让强模型获益。

**[FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning](fapo_flawed-aware_policy_optimization_for_efficient_and_reliable_reasoning.md)**

:   针对 RLVR 训练中"答案正确但推理有缺陷"的 flawed-positive rollout 问题，提出 FAPO 算法：用 GenRM 检测缺陷推理，通过无参数奖励惩罚机制实现"先利用后抑制"的自然学习轨迹，同时提升结果正确性、过程可靠性和训练稳定性。

**[Flow Actor-Critic for Offline Reinforcement Learning (FAC)](flow_actor-critic_for_offline_reinforcement_learning.md)**

:   FAC 首次联合利用流模型（continuous normalizing flow）同时构建表达力强的 actor 策略和基于精确密度估计的 critic 惩罚机制，通过识别 OOD 区域对 Q 值进行选择性保守估计，在 OGBench 55 个任务上以 60.3 平均分大幅超越此前最佳的 43.6。

**[From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning](from_narrow_to_panoramic_vision_attention-guided_cold-start_reshapes_multimodal_.md)**

:   发现多模态 LLM 的推理性能与视觉注意力分数（VAS）高度相关（r=0.96），提出 AVAR 框架通过视觉锚定数据合成、注意力引导训练目标和视觉锚定奖励塑造三个阶段提升 VAS，在 77 个基准上平均提升 7%。

**[From Observations to Events: Event-Aware World Model for Reinforcement Learning](from_observations_to_events_event-aware_world_model_for_reinforcement_learning.md)**

:   提出 Event-Aware World Model (EAWM)，一个通用框架，通过从原始观测中自动生成事件并学习事件感知表征，在不需要手工标签的情况下，将现有 MBRL 基线性能提升 10%–45%，在 Atari 100K、Craftax 1M、DeepMind Control 500K、DMC-GB2 500K 上均创新 SOTA。

**[From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for RL of Open-ended Generation](from_verifiable_dot_to_reward_chain_harnessing_verifiable_reference-based_reward.md)**

:   提出 RLVRR 框架，将 RLVR（强化学习+可验证奖励）从数学/代码推理扩展到开放式文本生成：从高质量参考答案中提取关键词序列（内容奖励）和可执行 Python 检查函数（风格奖励），构成"奖励链"替代单点验证信号，在 10+ 个 benchmark 上以 10K 数据超越 100K SFT 和高级奖励模型。

**[GraphOmni: A Comprehensive and Extensible Benchmark Framework for Large Language Models on Graph-theoretic Tasks](graphomni_a_comprehensive_and_extensible_benchmark_framework_for_large_language_.md)**

:   提出 GraphOmni 基准框架，在 7 种图类型 × 7 种序列化格式 × 9 种提示策略的 241K 查询上系统评估 11 个 LLM 的图论推理能力，揭示三维度间的复杂交互效应，并设计 RL 引导的组合搜索方法以 25% 成本保持约 90% 最优准确率。

**[Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving](helix_evolutionary_reinforcement_learning_for_open-ended_scientific_problem_solv.md)**

:   提出 HELIX 框架，将强化学习（GRPO）与进化算法（NSGA-II）结合用于开放式科学问题求解：RL 迭代优化策略，进化机制平衡解的质量与多样性，in-context learning 利用历史解指导探索，仅用 14B 模型在圆填充、机器学习任务等 20 个任务中超越 GPT-4o 流水线。

**[How Far Can Unsupervised RLVR Scale LLM Training?](how_far_can_unsupervised_rlvr_scale_llm_training.md)**

:   对无监督可验证奖励强化学习（URLVR）进行全面分析，揭示所有内在奖励方法本质上都是在"锐化"模型初始分布，导致先升后降的不可避免崩溃模式；提出Model Collapse Step作为模型先验指标，并指出外部奖励方法是突破可扩展性瓶颈的方向。

**[How LLMs Learn to Reason: A Complex Network Perspective](how_llms_learn_to_reason_a_complex_network_perspective.md)**

:   本文从复杂网络视角提出"稀疏概念网"理论来统一解释RLVR训练中四个令人困惑的现象（V形响应长度、两阶段学习曲线、灾难性遗忘、策略坍塌），揭示它们都源于平均度约为2的稀疏推理图的拓扑自组织，并据此设计Annealed-RLVR算法在数学推理基准上超越标准RLVR。

**[InFOM: Intention-Conditioned Flow Occupancy Models](infom_intention_flow_occupancy.md)**

:   InFOM 通过变分推断学习潜在意图编码器、用 flow matching 建模意图条件化的折扣状态占据度量，实现了 RL 中的高效预训练与微调，在 36 个状态任务和 4 个图像任务上比基线提升 1.8 倍中位回报和 36% 成功率。

**[Is Pure Exploitation Sufficient in Exogenous MDPs with Linear Function Approximation?](is_pure_exploitation_sufficient_in_exogenous_mdps_with_linear_function_approxima.md)**

:   证明在外生MDP（Exo-MDP，不确定性仅来自独立于智能体动作的外生输入）中，纯利用（无探索）策略即可达到次线性遗憾界——表格情形下PTO算法达到 $\tilde{O}(H^2|\Xi|\sqrt{K})$，线性函数逼近下LSVI-PE算法遗憾与特征维度和外生状态空间多项式相关、与内生状态/动作空间无关。

**[LadderSym: A Multimodal Interleaved Transformer for Music Practice Error Detection](laddersym_a_multimodal_interleaved_transformer_for_music_practice_error_detectio.md)**

:   提出LadderSym架构解决音乐练习错误检测任务，通过交替式跨流对齐模块（Ladder）克服晚期融合的对齐不足，并用符号乐谱提示（Sym）减少纯音频乐谱的频率歧义，在MAESTRO-E上将漏音F1从26.8%提升到56.3%。

**[Latent Wasserstein Adversarial Imitation Learning](latent_wasserstein_adversarial_imitation_learning.md)**

:   提出LWAIL方法，用ICVF从少量随机数据学习动态感知的潜空间表示，将Wasserstein距离的"地面度量"从欧氏距离升级为潜空间距离，仅用单条状态轨迹即可达到专家级模仿性能。

**[Learning from Synthetic Data Improves Multi-hop Reasoning](learning_from_synthetic_data_improves_multi-hop_reasoning.md)**

:   发现在完全虚构的规则生成合成数据上做RLVR训练，能显著提升LLM在真实多跳推理任务上的表现（Qwen3-0.6B提升56%-131%），因为模型学到了知识组合这一通用推理技能而非记忆事实知识。

**[Learning to Generate Unit Test via Adversarial Reinforcement Learning](learning_to_generate_unit_test_via_adversarial_reinforcement_learning.md)**

:   提出UTRL框架，通过对抗RL迭代训练单元测试生成器和代码生成器——测试生成器学习生成能区分LLM代码与正确代码的判别性测试用例，代码生成器学习通过这些测试——Qwen3-4B训练后超越GPT-4.1的测试生成质量。

**[Learning to Orchestrate Agents in Natural Language with the Conductor](learning_to_orchestrate_agents_in_natural_language_with_the_conductor.md)**

:   用GRPO训练一个7B Qwen2.5模型作为"Conductor"，通过自然语言输出完整的Agent工作流（子任务指令+worker分配+通信拓扑访问列表），协调GPT-5/Claude Sonnet 4/Gemini 2.5 Pro等frontier模型，仅用960题×200迭代训练，在7个推理benchmark上平均77.27%超越所有单模型（GPT-5为74.78%）和多Agent基线。

**[Learning to Play Multi-Follower Bayesian Stackelberg Games](learning_to_play_multi-follower_bayesian_stackelberg_games.md)**

:   首次系统研究多追随者贝叶斯Stackelberg博弈（BSG）的在线学习问题，通过对领导者策略空间进行"最佳响应区域"几何分割，在类型反馈下实现 $\tilde{O}(\sqrt{\min\{L, nK\} \cdot T})$ 的遗憾界——该界不随追随者数 $n$ 呈多项式增长，并证明了几乎匹配的 $\Omega(\sqrt{\min\{L, nK\}T})$ 下界。

**[Less is More: Clustered Cross-Covariance Control for Offline RL](less_is_more_clustered_cross-covariance_control_for_offline_rl.md)**

:   本文揭示了离线RL中标准平方误差目标会引入有害的TD交叉协方差，并提出C⁴（Clustered Cross-Covariance Control for TD）方法，通过分区缓冲区采样和显式梯度校正惩罚来抑制这一效应，在小数据集和OOD区域主导的场景下实现高达30%的回报提升。

**[LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)**

:   提出 LongRLVR，通过在 RLVR 训练中引入可验证的上下文奖励（context reward），解决长上下文场景下仅靠最终答案奖励导致的上下文定位（grounding）梯度消失问题，显著提升 LLM 长上下文推理能力。

**[LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning](longwriter-zero_mastering_ultra-long_text_generation_via_reinforcement_learning.md)**

:   提出 LongWriter-Zero：从基础模型出发，不依赖任何标注或合成数据，仅通过 GRPO 强化学习 + 三维度复合奖励模型（长度 / 质量 / 格式），涌现出超长高质量文本生成能力，在 WritingBench 上以 32B 参数量超越 DeepSeek-R1 和 Qwen3-235B 等 100B+ 模型。

**[LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts](loongrl_rl_for_reasoning_long_contexts.md)**

:   提出 LoongRL，通过构建 KeyChain 合成数据进行强化学习训练，使 LLM 涌现出 plan–retrieve–reason–recheck 的长上下文推理模式，仅在 16K 上下文上训练即可泛化到 128K，14B 模型达到 74.2 分接近 o3-mini (74.5) 和 DeepSeek-R1 (74.9)。

**[MARS-Sep: Multimodal-Aligned Reinforced Sound Separation](mars-sep_multimodal-aligned_reinforced_sound_separation.md)**

:   MARS-Sep 将查询条件声音分离重新建模为强化学习问题，通过分解 Beta 掩码策略在时频域上进行随机决策，并利用渐进式对齐的多模态编码器提供语义奖励信号，在信号保真度和语义一致性上同时取得提升。

**[Menlo: From Preferences to Proficiency – Evaluating and Modeling Native-like Quality Across 47 Languages](menlo_from_preferences_to_proficiency_--_evaluating_and_modeling_native-like_qua.md)**

:   提出 Menlo 框架，基于受众设计理论将母语级响应质量分解为四个维度，构建了覆盖 47 种语言变体的 6423 条标注偏好对数据集，并发现配对评估+RL 训练的 LLM 裁判可达到接近人类标注员的水平。

**[MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding](mergemix_a_unified_augmentation_paradigm_for_visual_and_multi-modal_understandin.md)**

:   MergeMix 提出了一种基于 token merging 的 mixup 数据增强方法，通过双部软匹配在注意力空间生成混合图像，并将混合比例作为偏好优化中的软边际，在图像分类和多模态大模型两个场景下统一了 SFT 和 RL 训练范式。

**[Metis-SPECS: Decoupling Multimodal Learning via Self-distilled Preference-based Cold Start](metis-specs_decoupling_multimodal_learning_via_self-distilled_preference-based_c.md)**

:   提出 SPECS 三阶段冷启动框架——先通过自蒸馏生成偏好数据（仅区分格式差异），再用 DPO 做格式预对齐作为冷启动，最后接 GRPO 微调——解耦了格式学习和推理学习，实现 MEGA-Bench +4.1%、MathVista +12.2% 的一致性能提升。

**[ROMI: Model-based Offline RL via Robust Value-Aware Model Learning with Implicitly Differentiable Adaptive Weighting](model-based_offline_rl_via_robust_value-aware_model_learning_with_implicitly_dif.md)**

:   ROMI 通过 Wasserstein 对偶将动力学不确定集转化为状态不确定集来实现鲁棒的价值感知模型学习，并用隐式可微的自适应加权机制平衡动力学精度与价值感知，解决了 RAMBO 方法中的 Q 值低估和梯度爆炸问题，在 D4RL 和 NeoRL 上达到模型基离线 RL 的 SOTA。

**[Model Predictive Adversarial Imitation Learning for Planning from Observation](model_predictive_adversarial_imitation_learning_for_planning_from_observation.md)**

:   提出 MPAIL（Model Predictive Adversarial Imitation Learning），将 MPPI 规划器嵌入对抗模仿学习循环，首次实现端到端的仅观测规划框架（Planning-from-Observation），在泛化性、鲁棒性、可解释性和样本效率上全面优于基于策略的 AIL 方法，并在真实世界机器人导航中从单条观测演示成功部署。

**[MoMaGen: Generating Demonstrations under Soft and Hard Constraints for Multi-Step Bimanual Mobile Manipulation](momagen_generating_demonstrations_under_soft_and_hard_constraints_for_multi-step.md)**

:   MoMaGen 将双臂移动操作的演示数据生成建模为约束优化问题，通过硬约束（可达性、无碰撞、可见性）和软约束（导航中物体可见性、收回紧凑姿态）的协同，从单个人类遥操作演示自动生成大规模多样化数据集，训练出的视觉运动策略仅用 40 个真实演示微调即可部署到实体机器人。

**[MVR: Multi-view Video Reward Shaping for Reinforcement Learning](mvr_multi-view_video_reward_shaping_for_reinforcement_learning.md)**

:   提出 MVR 框架，利用多视角视频的视频-文本相似度学习状态相关性函数，结合状态依赖的奖励塑形（自动衰减 VLM 引导），在 HumanoidBench 和 MetaWorld 共 19 个任务上超越现有 VLM 奖励方法。

**[Near-Optimal Second-Order Guarantees for Model-Based Adversarial Imitation Learning](near-optimal_second-order_guarantees_for_model-based_adversarial_imitation_learn.md)**

:   提出 MB-AIL（基于模型的对抗模仿学习）算法，在一般函数逼近下建立了无视域（horizon-free）的二阶样本复杂度上界，结合新构建的困难实例上的信息论下界，证明 MB-AIL 在在线交互的样本复杂度上达到极小极大最优（相差对数因子）。

**[Nearly-Optimal Bandit Learning in Stackelberg Games with Side Information](nearly-optimal_bandit_learning_in_stackelberg_games_with_side_information.md)**

:   本文通过将Stackelberg博弈中的领导者效用空间线性化，提出了一种约简到线性上下文赌臂问题的算法，在带侧信息的赌臂反馈设置下将遗憾界从 $\tilde{O}(T^{2/3})$ 改进到近似最优的 $\tilde{O}(T^{1/2})$。

**[Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)**

:   提出 Generative Trajectory Policy (GTP)，通过学习 ODE 完整解映射统一扩散、流匹配和一致性模型，配合分数近似和值驱动引导两项关键适配技术，在 D4RL 上达到 SOTA。

**[On Discovering Algorithms for Adversarial Imitation Learning](on_discovering_algorithms_for_adversarial_imitation_learning.md)**

:   提出 DAIL——首个元学习对抗性模仿学习算法：将 AIL 分解为密度比估计和奖励赋值(RA)两阶段，用 LLM 引导的进化搜索自动发现最优 RA 函数 $r_{\text{disc}}$，在未见环境和策略优化器上泛化并超越所有人工设计基线。

**[On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification](on_the_generalization_of_sft_a_reinforcement_learning_perspective_with_reward_re.md)**

:   从RL策略梯度视角数学证明SFT梯度隐式编码了逆概率加权(1/π_θ)的病态奖励结构→低概率token梯度过大导致泛化受限，提出DFT(Dynamic Fine-Tuning)仅需一行代码修改(CE loss乘token概率：$-p\log p$)消除逆概率加权→在数学推理/代码生成/多模态任务上大幅超越SFT，离线RL设定下甚至超越GRPO/PPO。

**[On the $O(1/T)$ Convergence of Alternating Gradient Descent-Ascent in Bilinear Games](on_the_o1t_convergence_of_alternating_gradient_descent-ascent_in_bilinear_games.md)**

:   首次证明交替梯度下降上升（AltGDA）在有约束双线性零和博弈中以 $O(1/T)$ 速率收敛到Nash均衡（存在内部NE时），比同步GDA的 $O(1/\sqrt{T})$ 快，用能量函数衰减刻画轨迹碰撞边界时的"摩擦"效应，并通过性能估计编程（PEP）进一步优化步长。

**[One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning](one_model_for_all_tasks_leveraging_efficient_world_models_in_multi-task_planning.md)**

:   提出 ScaleZero，通过在统一世界模型中引入 MoE 架构解决多任务学习中的梯度冲突和可塑性崩塌问题，结合动态参数扩展（DPS）策略自适应分配模型容量，单个多任务模型在 Atari/DMC/Jericho 三个基准上达到与单任务专家模型相当的性能，同时减少约 28.5% 的环境交互。

**[Online Minimization of Polarization and Disagreement via Low-Rank Matrix Bandits](online_minimization_of_polarization_and_disagreement_via_low-rank_matrix_bandits.md)**

:   将Friedkin-Johnsen观点动力学模型下极化+分歧最小化问题首次形式化为在线低秩矩阵bandit问题（OPD-Min），提出两阶段算法OPD-Min-ESTR通过子空间估计将维度从 $|V|^2$ 降至 $O(|V|)$，在合成和真实网络上显著优于全维度线性bandit基线。

**[Online Prediction of Stochastic Sequences with High Probability Regret Bounds](online_prediction_of_stochastic_sequences_with_high_probability_regret_bounds.md)**

:   重新审视有限时间范围 $T$ 下随机序列的通用预测经典问题，首次给出以高概率成立的消退遗憾界（形式为 $O(T^{-1/2}\delta^{-1/2})$），与已有的期望遗憾界 $O(T^{-1/2})$ 形式高度一致，并证明在不附加额外假设时 $\delta$ 的指数无法改进。

**[Optimistic Task Inference for Behavior Foundation Models](optimistic_task_inference_behavior_models.md)**

:   提出 OpTI-BFM——在 Behavior Foundation Model 测试时，不需要完整奖励函数或标注数据集，而是通过与环境交互仅 5 个 episode 即可推断任务并恢复 Oracle 性能，核心是利用 successor features 的线性结构将任务推断归约为线性 bandit 问题并用 UCB 策略乐观探索，提供正式的 regret bound。

**[P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling](p-genrm_personalized_generative_reward_model_with_test-time_user-based_scaling.md)**

:   提出 P-GenRM，首个个性化生成式奖励模型：通过三阶段训练（PSI 监督微调构建结构化评价链→CRE 强化学习增强缺失偏好下的推理→难负例课程学习提升鲁棒性）将混合偏好信号转化为场景自适应的用户画像与评分标准，再引入双粒度测试时 scaling（个体级多次采样聚合 + 原型级协同过滤借用相似用户偏好），在 PersonalRewardBench 上超越前 SOTA 2.31%、测试时 scaling 额外提升 3%，且能泛化到未见用户。

**[ParaS2S: Benchmarking and Aligning Spoken Language Models for Paralinguistic-Aware Speech-to-Speech Interaction](paras2s_benchmarking_and_aligning_spoken_language_models_for_paralinguistic-awar.md)**

:   提出 ParaS2S 框架——包含一个评估副语言感知（emotion/sarcasm/age/gender）的语音到语音基准 ParaS2SBench，以及一个基于 GRPO 的 RL 对齐框架 ParaS2SAlign，使 S2S 模型能够在极少标注数据下习得根据说话风格调整回复的能力。

**[Partially Equivariant Reinforcement Learning in Symmetry-Breaking Environments](partially_equivariant_reinforcement_learning_in_symmetry-breaking_environments.md)**

:   提出部分群不变MDP (PI-MDP) 框架，通过可学习的门控函数 $\lambda(s,a)$ 在状态-动作空间中逐点切换等变与标准Bellman更新，从理论上证明局部对称性破缺会经过折扣回溯放大 $1/(1-\gamma)$ 倍产生全局值函数误差，而PI-MDP可将误差严格限制在破缺区域内；实例化为PE-DQN和PE-SAC两种算法，在Grid-World、MuJoCo运动、机械臂操作等任务上全面超越严格等变和近似等变基线。

**[PolicyFlow: Policy Optimization with Continuous Normalizing Flow in Reinforcement Learning](policyflow_policy_optimization_with_continuous_normalizing_flow_in_reinforcement.md)**

:   提出PolicyFlow，将连续归一化流(CNF)策略无缝嵌入PPO框架：通过沿插值路径的速度场变化近似重要性比率（避免全流路径反向传播），并引入受布朗运动启发的隐式熵正则器防止模式坍缩，在MultiGoal/PointMaze/IsaacLab/MuJoCo等环境中达到或超越高斯PPO和流式基线(FPO/DPPO)的性能。

**[Post-training Large Language Models for Diverse High-Quality Responses](post-training_large_language_models_for_diverse_high-quality_responses.md)**

:   提出 DQO（Diversity Quality Optimization），基于行列式点过程（DPP）在语义嵌入空间中定义多样性度量，将其与奖励信号联合优化，使 LLM 后训练同时提升语义多样性和响应质量，可叠加在 GRPO/PPO 之上。

**[PreferThinker: Reasoning-based Personalized Image Preference Assessment](preferthinker_reasoning-based_personalized_image_preference_assessment.md)**

:   提出 PreferThinker，通过引入通用视觉偏好画像（preference profile）连接不同用户，采用 predict-then-assess 的 CoT 推理范式进行可解释的个性化图像偏好评估，结合冷启动 SFT + GRPO 强化学习及 similarity-aware 预测奖励，7B 模型超越 GPT-4o（+5.2%）和 Claude 3.7（+5.1%）。

**[Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](principled_fast_and_meta_knowledge_learners_for_continual_reinforcement_learning.md)**

:   受人脑海马体-大脑皮层交互机制启发，提出 FAME 双学习器框架，通过快速学习器进行知识迁移、元学习器进行知识整合，在原则性地最小化灾难性遗忘的前提下实现高效的持续强化学习。

**[QuRL: Efficient Reinforcement Learning with Quantized Rollout](qurl_efficient_reinforcement_learning_with_quantized_rollout.md)**

:   提出 QuRL 方法，通过量化 actor 模型加速 RL 训练中的 rollout 阶段，设计自适应裁剪范围（ACR）解决量化导致的训练崩溃问题，并提出更新感知量化（UAQ）解决权重更新远小于量化误差的尺度失配问题，实现 20%~80% 的推理吞吐量提升且不损失性能。

**[REA-RL: Reflection-Aware Online Reinforcement Learning for Efficient Reasoning](rea-rl_reflection-aware_online_reinforcement_learning_for_efficient_reasoning.md)**

:   提出REA-RL框架，通过蒸馏训练的小型反思模型在线识别并截断过度思考token生成修订路径，配合反思奖励防止RL训练中模型退化为无反思的朴素CoT，在DeepSeek-R1-Distill-Qwen-7B上实现推理token开销降低36%且准确率零损失。

**[Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment](reasoning_as_representation_rethinking_visual_reinforcement_learning_in_image_qu.md)**

:   通过系统实验揭示了 RL 训练的推理型 IQA 模型泛化能力的本质机制——推理过程本质上是将冗余的视觉表示转换为紧凑的跨域对齐文本表示——并基于此提出 RALI 算法，通过对比学习直接对齐图像与这些文本表示，以不到 5% 的参数和推理时间达到了可比的泛化性能。

**[Reasoning Boosts Opinion Alignment in LLMs](reasoning_boosts_opinion_alignment_in_llms.md)**

:   用 GRPO 强化学习训练 LLM 通过结构化推理对齐个体政治观点，SFT+GRPO 在美国/德国/瑞士三国数据集上一致优于 ICL 和 ORPO 基线，但系统性揭示了左右翼偏差和 Neutral 立场预测的根本困难。

**[RebuttalAgent: Strategic Persuasion in Academic Rebuttal via Theory of Mind](rebuttalagent_strategic_persuasion_in_academic_rebuttal_via_theory_of_mind.md)**

:   首次将心智理论（ToM）引入学术 rebuttal，提出 ToM-Strategy-Response 三阶段框架：先建模审稿人心理状态，再制定说服策略，最后生成证据支撑的回复，结合自奖励 RL 训练和专用 Rebuttal-RM 评估器，平均指标超越基座模型 18.3%。

**[References Improve LLM Alignment in Non-Verifiable Domains](references_improve_llm_alignment_in_non-verifiable_domains.md)**

:   提出参考引导的LLM-as-Judge方法(RefEval)，用高质量参考输出作为"软验证器"，使LLM-judge准确率提升6.8%；进而构建两阶段自改进流程(SFT蒸馏+参考引导DPO)，在AlpacaEval/Arena-Hard上分别超过SFT蒸馏+19.2/+16.5，匹配微调奖励模型ArmoRM的性能，证明无需人类偏好标注即可实现非可验证域的高效LLM对齐。

**[ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation](reform_reflected_flows_for_on-support_offline_rl_via_noise_manipulation.md)**

:   提出ReFORM方法，通过学习一个反射流噪声生成器来操纵行为克隆流策略的源分布，以**构造性方式**实现支撑约束，避免OOD问题的同时保持策略表达力，无需超参数调节。

**[Regret-Guided Search Control for Efficient Learning in AlphaZero](regret-guided_search_control_for_efficient_learning_in_alphazero.md)**

:   提出 RGSC（Regret-Guided Search Control）框架，通过训练一个 regret 网络识别高遗憾值状态并优先从这些状态重新开始自我对弈，模拟人类"反复复盘错误"的学习方式，在 9×9 围棋、10×10 黑白棋和 11×11 Hex 上平均超越 AlphaZero 77 Elo。

**[Pruning as a Cooperative Game: Surrogate-Assisted Layer Contribution Estimation for Large Language Models](remix_reinforcement_routing_for_mixtures_of_loras_in_llm_finetuning.md)**

:   将LLM层剪枝建模为合作博弈（每层=玩家，模型性能=效用）→精确Shapley值计算不可行（$2^L$种组合）→提出两阶段近似：(1)分层蒙特卡洛采样生成mask+评估PPL作为监督信号→(2)训练轻量代理网络预测任意mask的性能→高效估算每层Shapley值→捕获层间依赖→显著优于静态启发式剪枝基线。

**[ReMix: Reinforcement Routing for Mixtures of LoRAs in LLM Finetuning](remix_reinforcement_routing_lora.md)**

:   ReMix 发现现有 Mixture-of-LoRAs 模型存在严重的路由权重坍缩问题（即使激活 k>1 个 LoRA，有效 LoRA 数也迅速降到 1），提出用非可学习的常数路由权重确保所有激活 LoRA 平等贡献，并用 RLOO 强化学习梯度估计器训练路由器，显著优于 SOTA PEFT 方法。

**[ReMoT: Reinforcement Learning with Motion Contrast Triplets](remot_reinforcement_learning_with_motion_contrast_triplets.md)**

:   ReMoT 提出一个统一的训练范式，通过规则驱动的运动对比三元组数据集（ReMoT-16K）和 Group Relative Policy Optimization（GRPO）组合奖励优化，系统性地提升 VLM 在时空一致性推理上的能力，在时空推理任务上实现 25.1% 的性能跃升。

**[Retaining Suboptimal Actions to Follow Shifting Optima in Multi-Agent RL](retaining_suboptimal_actions_to_follow_shifting_optima_in_multi-agent_reinforcem.md)**

:   提出 S2Q（Successive Sub-value Q-learning），通过逐步学习 $K$ 个 sub-value 函数显式保留次优联合动作，结合 Softmax 行为策略在候选间优先级采样，解决合作 MARL 中值分解方法因最优点动态漂移而收敛到次优策略的根本问题。

**[Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning](rethinking_policy_diversity_in_ensemble_policy_gradient_in_large-scale_reinforce.md)**

:   从理论上分析了集成策略梯度方法中策略间多样性对学习效率的影响，提出通过KL散度约束调控多样性的Coupled Policy Optimization（CPO），在大规模并行环境中实现高效稳定的探索。

**[Revisiting Matrix Sketching in Linear Bandits: Achieving Sublinear Regret via Dyadic Block Sketching](revisiting_matrix_sketching_in_linear_bandits_achieving_sublinear_regret_via_dya.md)**

:   本文揭示了现有基于矩阵草图的线性Bandit方法在流数据频谱尾部较重时会退化为线性遗憾的根本缺陷，提出 Dyadic Block Sketching 多尺度草图框架，通过动态加倍草图大小控制全局逼近误差至预设参数 $\epsilon$，使算法在无需预知流矩阵频谱性质的情况下始终保证次线性遗憾，并在频谱友好场景下自适应恢复单尺度方法的计算效率。

**[RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning](rewardmap_tackling_sparse_rewards_in_fine-grained_visual_reasoning_via_multi-sta.md)**

:   提出RewardMap框架，通过难度感知的细节奖励设计和从简单感知到复杂推理的多阶段RL课程学习策略，克服细粒度视觉推理中的稀疏奖励问题。

**[RLP: Reinforcement as a Pretraining Objective](rlp_reinforcement_as_a_pretraining_objective.md)**

:   提出RLP（Reinforcement Learning Pretraining），一种信息增益驱动的RL预训练目标，通过奖励能提升下一token预测概率的思维链（CoT），将RL从后训练阶段前移到预训练阶段，实现无验证器的密集奖励信号。

**[RM-R1: Reward Modeling as Reasoning](rm-r1_reward_modeling_as_reasoning.md)**

:   将奖励建模重新定义为推理任务，提出RM-R1系列推理奖励模型（ReasRM），通过推理蒸馏+RL训练以及Chain-of-Rubrics（CoR）机制，在三大奖励模型基准上平均超越70B和GPT-4o模型达4.9%。

**[Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)**

:   本文研究 RL 中一种新型威胁——行为目标攻击（adversary 通过篡改观测来引导 victim 执行特定目标策略），提出不需要白盒访问的 BIA 攻击方法和基于时间折扣的 TDRT 防御方法，TDRT 在保持对攻击鲁棒性的同时比现有防御（SA-PPO）的原始任务性能高 28.2%。

**[Robust Multi-Objective Controlled Decoding of Large Language Models](robust_multi-objective_controlled_decoding_of_large_language_models.md)**

:   提出RMOD（Robust Multi-Objective Decoding），一种推理时算法，通过求解最小最大博弈的Nash均衡来动态计算最坏情况目标权重，在无需先验权重信息的情况下实现LLM的鲁棒多目标对齐。

**[Routing, Cascades, and User Choice for LLMs](routing_cascades_and_user_choice_for_llms.md)**

:   将LLM路由建模为provider-user Stackelberg博弈，证明最优路由几乎总是静态无级联的阈值规则，揭示质量/成本排序不一致时的用户-提供商不对齐，以及低流失惩罚下provider被激励通过throttling延迟来降低成本但损害用户效用。

**[RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](rulereasoner_reinforced_rule-based_reasoning_via_domain-aware_dynamic_sampling.md)**

:   RuleReasoner 通过构建多样化的规则推理数据集 RuleCollection-32K 和提出域感知动态采样（Dads）策略，在 RLVR 框架下训练 8B 模型，在域内推理任务上比 OpenAI-o1 高 4.1%，在域外任务上高 10.4%，同时训练效率提升 ~1.4×。

**[Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form](safe_continuous-time_multi-agent_reinforcement_learning_via_epigraph_form.md)**

:   提出首个显式处理状态约束的连续时间多智能体RL框架，通过Epigraph形式将不连续的约束值函数转化为连续表示，结合改进的PINN actor-critic方法实现安全、稳定的连续时间多智能体控制。

**[Sample-efficient and Scalable Exploration in Continuous-Time RL](sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)**

:   提出 COMBRL 算法，通过最大化外在奖励与模型认知不确定性的加权和，在连续时间模型基 RL 中实现可扩展且样本高效的探索，并具有次线性后悔理论保证。

**[Scalable Exploration for High-Dimensional Continuous Control via Value-Guided Flow](scalable_exploration_for_high-dimensional_continuous_control_via_value-guided_fl.md)**

:   提出Qflex(Q-guided Flow Exploration)——在高维连续动作空间中实现可扩展探索的RL方法：从可学习源分布沿Q函数诱导的概率流传输动作→探索与任务相关梯度对齐(而非各向同性噪声)→在多种高维基准上超越高斯/扩散RL基线,成功控制700执行器的全身人体肌骨模型执行敏捷复杂动作。

**[Scalable In-Context Q-Learning](scalable_in-context_q-learning.md)**

:   提出 S-ICQL——将动态规划（Q-learning）和世界模型引入监督式 ICRL 框架，通过多头 Transformer 同时预测策略和情境值函数，预训练世界模型构建轻量级精确提示，advantage-weighted regression 提取策略，在离散和连续环境中从次优数据学习时一致超越所有基线。

**[Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning](self-harmony_learning_to_harmonize_self-supervision_and_self-play_in_test-time_r.md)**

:   提出 Self-Harmony 框架，通过让单一模型扮演两个角色（Solver 求解原始问题 + Reframer 改述问题），将答案在原始和改述视角下的调和平均得分作为伪标签选择标准，替代传统多数投票，在 30 个实验设置中 28 个达到 SOTA，且训练零失败。

**[Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)**

:   提出 SISL（Self-Improving Skill Learning），通过解耦高层策略和技能改进策略，结合最大回报重标注的技能优先级机制，在噪声离线演示数据下实现鲁棒的技能学习，显著提升基于技能的元强化学习在长时域任务中的性能。

**[Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning](shop-r1_rewarding_llms_to_simulate_human_behavior_in_online_shopping_via_reinfor.md)**

:   提出 Shop-R1 框架，通过分层奖励机制和难度感知缩放的强化学习方法，显著提升 LLM 模拟真实人类在线购物行为的能力，相比 SFT 基线精确匹配提升超过 65%。

**[Single Index Bandits: Generalized Linear Contextual Bandits with Unknown Reward Functions](single_index_bandits_generalized_linear_contextual_bandits_with_unknown_reward_f.md)**

:   提出单指标赌博机（SIB）问题——将广义线性赌博机扩展到奖励函数未知的设定，基于 Stein 方法设计了一族高效算法（STOR/ESTOR/GSTOR），在单调递增奖励函数下实现了近最优遗憾界 $\tilde{O}(\sqrt{T})$。

**[Solving Football by Exploiting Equilibrium Structure of 2p0s Differential Games with One-Sided Information](solving_football_by_exploiting_equilibrium_structure_of_2p0s_differential_games_.md)**

:   证明单边信息二人零和微分博弈中 Nash 均衡策略的原子结构——知情玩家 P1 的均衡策略集中在至多 $I$ 个动作原型上（$I$ = 博弈类型数），使博弈树复杂度从 $U^{2K}$ 降到 $I^K$，在美式足球 11v11 连续动作空间中（传统复杂度 $10^{440}$）实现 M1 MacBook 30 分钟求解。

**[Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning](solving_parameter-robust_avoid_problems_with_unknown_feasibility_using_reinforce.md)**

:   提出 Feasibility-Guided Exploration (FGE)，同时识别可行参数子集并学习在该子集上安全的策略，解决可行性未知的参数鲁棒避障问题，在 MuJoCo 任务中比最佳现有方法多覆盖 50% 以上。

**[Spectral Bellman Method: Unifying Representation and Exploration in RL](spectral_bellman_method_unifying_representation_and_exploration_in_rl.md)**

:   提出 Spectral Bellman Method (SBM)，从零内在 Bellman 误差 (IBE) 条件出发发现 Bellman 算子与特征协方差的谱结构联系，推导出新的表示学习目标，并自然地统一了表示学习和 Thompson Sampling 探索。

**[SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)**

:   提出 SPELL 框架，让一个 LLM 同时扮演出题者、答题者和验证者三个角色进行自我博弈强化学习，无需人类标注即可持续提升长文本推理能力，在 6 个长上下文基准上一致提升性能。

**[SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)**

:   提出 SPIRAL 框架，让 LLM 在多轮零和游戏中进行自我博弈训练，通过角色条件优势估计（RAE）稳定训练，在无领域特定数据的情况下将推理能力提升最高 10%，并发现不同游戏发展出互补的认知能力。

**[Spotlight on Token Perception for Multimodal Reinforcement Learning](spotlight_on_token_perception_for_multimodal_reinforcement_learning.md)**

:   提出 VPPO（Visually-Perceptive Policy Optimization），通过量化每个 token 的视觉依赖度，在轨迹级和 token 级两个层次对学习信号进行精细化调控，显著提升大视觉语言模型的多模态推理能力。

**[Stackelberg Coupling of Online Representation Learning and Reinforcement Learning](stackelberg_coupling_of_online_representation_learning_and_reinforcement_learnin.md)**

:   提出 SCORER 框架，将 Deep Q-Learning 中的表征学习和值函数学习建模为 Stackelberg 博弈，通过双时间尺度更新（Q 网络为 leader 慢更新、编码器为 follower 快更新）实现稳定协同适应，无需改变网络结构即可提升性能。

**[Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty](stop_unnecessary_reflection_training_lrms_for_efficient_reasoning_with_adaptive_.md)**

:   提出 ARLCP（Adaptive Reflection and Length Coordinated Penalty），一种自适应强化学习方法，根据问题复杂度动态调节反思惩罚和长度惩罚的权重，在保持或提升准确性的同时大幅减少推理 token 消耗。

**[Strict Subgoal Execution: Reliable Long-Horizon Planning in Hierarchical Reinforcement Learning](strict_subgoal_execution_reliable_long-horizon_planning_in_hierarchical_reinforc.md)**

:   提出 SSE（Strict Subgoal Execution）框架，通过**前沿经验回放（FER）** 严格区分子目标到达成功与失败，配合解耦探索策略和失败感知路径优化，在每个高层步骤内强制完成子目标到达，显著减少高层决策步数并提升长时程任务成功率。

**[SUSD: Structured Unsupervised Skill Discovery through State Factorization](susd_structured_unsupervised_skill_discovery_through_state_factorization.md)**

:   提出 SUSD（Structured Unsupervised Skill Discovery），通过将状态空间分解为独立因子并为每个因子分配专属技能变量，结合好奇心驱动的因子加权机制，实现在多物体/多智能体复杂环境中发现覆盖全部可控因子的多样化技能。

**[$\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving](textbfre2_unlocking_llm_reasoning_via_reinforcement_learning_with_re-solving.md)**

:   本文提出 Re² 方法，通过纯强化学习训练 LLM 学会在推理过程中主动放弃无效思维链并重新开始求解，将罕见的 redo 行为从 0.5% 提升至 30% 以上，在相同训练计算预算下显著超越标准 RLVR 方法。

**[The Sample Complexity of Online Reinforcement Learning: A Multi-Model Perspective](the_sample_complexity_of_online_reinforcement_learning_a_multi-model_perspective.md)**

:   本文为连续状态-动作空间下的非线性动力系统提出了一套在线强化学习算法，通过多模型后验采样和确定性等价策略实现对未知系统的在线学习，并给出了从有限模型集到参数化模型族的非渐近策略遗憾保证。

**[Thermodynamics of Reinforcement Learning Curricula](thermodynamics_of_reinforcement_learning_curricula.md)**

:   本文利用非平衡热力学中的过剩功（excess work）最小化框架，将RL中的课程学习形式化为任务空间上的测地线优化问题，并推导出基于摩擦张量的温度退火算法MEW，在MuJoCo Humanoid任务上超越标准SAC温度调节方法。

**[Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization](thinking_on_the_fly_test-time_reasoning_enhancement_via_latent_thought_policy_op.md)**

:   本文提出潜在思维策略优化（LTPO），一种无需更新模型参数的测试时推理增强框架，通过将中间潜在"思维"向量视为可优化的动态参数，利用在线策略梯度方法和内在置信度奖励信号来增强冻结LLM的推理能力。

**[Toward a Dynamic Stackelberg Game-Theoretic Framework for Agent-Based Conversational AI Defense Against LLM Jailbreaking](toward_a_dynamic_stackelberg_game-theoretic_framework_for_agent-based_conversat.md)**

:   将 LLM 越狱攻防形式化为动态 Stackelberg 扩展形式博弈，结合快速扩展随机树 (RRT) 搜索提示空间，提出 Purple Agent 防御架构实现"红队思维，蓝队行动"的预见性防御。

**[Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control](towards_bridging_the_gap_between_large-scale_pretraining_and_efficient_finetunin.md)**

:   LIFT提出预训练-微调三阶段框架：(i) 大规模并行SAC预训练实现零样本部署；(ii) 基于拉格朗日动力学的物理先验世界模型离线预训练；(iii) 确定性动作执行+世界模型内随机探索的高效微调，在Booster T1和Unitree G1人形机器人上验证了从仿真到真实世界的全流程。

**[Towards Strategic Persuasion with Language Models](towards_strategic_persuasion_with_language_models.md)**

:   本文以贝叶斯说服（Bayesian Persuasion）框架为理论基础，提出了一套系统评估和训练LLM策略性说服能力的方法，发现前沿模型已具备显著的策略性说服能力，且即使是小型LLM也可通过强化学习大幅提升说服效果。

**[TPRU: Advancing Temporal and Procedural Understanding in Large Multimodal Models](tpru_advancing_temporal_and_procedural_understanding_in_large_multimodal_models.md)**

:   TPRU构建了大规模多图像时序理解数据集（24,750个QA对、126,000张图像），覆盖机器人操作、GUI导航等4个具身场景的3种互补任务（时序排序、下一帧预测、前帧回溯），并通过强化学习微调使7B模型在时序理解上超越GPT-4o。

**[TRACED: Transition-aware Regret Approximation with Co-learnability for Environment Design](traced_transition-aware_regret_approximation_with_co-learnability_for_environmen.md)**

:   TRACED改进无监督环境设计（UED）中的regret近似——在传统PVL基础上加入转移预测误差（ATPL）捕获动力学模型失配，并引入Co-Learnability度量任务间迁移效益，在MiniGrid和BipedalWalker上以10k更新超越所有baseline的20k更新性能。

**[Transitive RL: Value Learning via Divide and Conquer](transitive_rl_value_learning_via_divide_and_conquer.md)**

:   本文提出 Transitive Reinforcement Learning（TRL），一种基于分治范式的新型值函数学习算法，利用目标条件RL中固有的三角不等式结构，将值函数更新递归分解为子问题，在长时间跨度任务上实现了优于TD学习和蒙特卡洛方法的性能。

**[Trinity: An Evolved LLM Coordinator](trinity_an_evolved_llm_coordinator.md)**

:   Trinity设计了一个轻量级coordinator（0.6B SLM + ~10K可训练参数的head），通过sep-CMA-ES优化，在多轮对话中将查询分配给不同LLM并指定Thinker/Worker/Verifier三种角色，在LiveCodeBench上达到86.2% pass@1的SOTA，在4个分布内和4个分布外任务上一致超越所有单模型和多agent基线。

**[TROLL: Trust Regions improve Reinforcement Learning for Large Language Models](troll_trust_regions_improve_reinforcement_learning_for_large_language_models.md)**

:   本文提出 TROLL（Trust Region Optimization for Large Language models），用可微分的离散信任域投影替代PPO中的裁剪（clipping）机制，实现了基于原则性KL约束的token级策略更新，在数学推理和代码生成任务上一致性地优于PPO裁剪方法。

**[UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings](ume-r1_exploring_reasoning-driven_generative_multimodal_embeddings.md)**

:   提出 UME-R1，首次探索推理驱动的生成式多模态嵌入范式，通过两阶段训练（冷启动SFT + 强化学习）让嵌入模型先推理再生成表示，在 MMEB-V2 基准的 78 个任务上显著超越传统判别式嵌入模型。

**[Understanding and Improving Hyperbolic Deep Reinforcement Learning](understanding_and_improving_hyperbolic_deep_reinforcement_learning.md)**

:   通过闭式梯度分析揭示双曲深度 RL 中 Poincaré Ball 保角因子爆炸和大范数嵌入导致 PPO 信赖域失效的根源，提出 Hyper++（RMSNorm + 可学习缩放 + HL-Gauss + Hyperboloid）四组件方案，在 ProcGen 16 环境和 Atari-5 上全面超越先前基线。

**[Unsupervised Learning of Efficient Exploration: Pre-training Adaptive Policies via Self-Imposed Goals](unsupervised_learning_of_efficient_exploration_pre-training_adaptive_policies_vi.md)**

:   提出 ULEE，一种无监督元学习方法，通过对抗式自生成目标课程训练自适应策略，在 XLand-MiniGrid 基准上实现高效探索与少样本适应。

**[Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning](unveiling_the_cognitive_compass_theory-of-mind-guided_multimodal_emotion_reasoni.md)**

:   构建基于心智理论（ToM）的层次化多模态情感理解基准 HitEmotion，并提出 TMPO 框架通过中间心理状态作为过程级监督来增强 MLLM 的情感推理能力。

**[Value Flows](value_flows.md)**

:   Value Flows 首次将流匹配（flow matching）引入分布式 RL——学习一个向量场使生成的概率密度路径自动满足分布式 Bellman 方程，通过 flow derivative ODE 高效估计回报方差实现置信度加权优先学习，在 OGBench 62 个任务上平均 1.3× 成功率提升，回报分布估计精度比 C51/CODAC 好 3×+。

**[VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)**

:   针对大型推理模型（LRM）训练中广泛使用的基于参考答案的奖励系统，构建了 VerifyBench 和 VerifyBench-Hard 两个评测基准，通过严格的人工标注评估各类验证系统的准确性，发现即使最强模型在困难样本上也仅达约 88% 准确率，揭示了当前验证系统的显著改进空间。

**[Virne: A Comprehensive Benchmark for RL-based Network Resource Allocation in NFV](virne_a_comprehensive_benchmark_for_rl-based_network_resource_allocation_in_nfv.md)**

:   提出 Virne——一个面向网络功能虚拟化资源分配（NFV-RA）的综合基准框架，集成 30+ 种算法和 gym 风格环境，支持云、边缘、5G 等多场景的系统评估。

**[Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity](whatever_remains_must_be_true_filtering_drives_reasoning_in_llms_shaping_diversi.md)**

:   提出 DMVR 框架和 α-DPG 算法，通过显式定义"过滤掉错误答案"的目标分布并用 α-散度族来逼近，统一了 RLVR（Reverse KL）和拒绝采样微调（Forward KL），在 Lean 定理证明上实现了精度-覆盖率 Pareto 前沿的最优表现。

**[When Sensors Fail: Temporal Sequence Models for Robust PPO under Sensor Drift](when_sensors_fail_temporal_sequence_models_for_robust_ppo_under_sensor_drift.md)**

:   本文研究PPO在时间持续性传感器故障下的鲁棒性，提出将Transformer和SSM等序列模型集成到PPO中，推导了随机传感器故障下无限时间horizon奖励退化的高概率上界，并在MuJoCo实验中验证Transformer-PPO在严重传感器dropout下显著优于MLP、RNN和SSM基线。

**[WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)**

:   WIMLE将隐式最大似然估计（IMLE）扩展到model-based RL，学习能捕获多模态转移动力学的随机世界模型，通过ensemble+latent采样估计预测不确定性，用不确定性加权合成数据的RL目标，在40个连续控制任务上实现超越模型-free和model-based强基线的样本效率和渐近性能。
