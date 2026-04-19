---
title: >-
  AAAI2026 其他方向 113篇论文解读
description: >-
  113篇AAAI2026 其他方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📂 其他

**🤖 AAAI2026** · **113** 篇论文解读

**[A Fast Heuristic Search Approach for Energy-Optimal Profile Routing for Electric Vehicles](a_fast_heuristic_search_approach_for_energy-optimal_profile_.md)**

:   提出基于多目标A*搜索的label-setting方法（Pr-A*），在初始电量未知时高效求解电动车能耗最优路径（profile搜索），通过profile支配关系剪枝避免传统方法中复杂的profile合并操作，在大规模路网上性能接近已知初始电量的标准A*搜索。

**[A Graph-Theoretical Perspective on Law Design for Multiagent Systems](a_graph-theoretical_perspective_on_law_design_for_multiagent.md)**

:   将多智能体系统中的法律设计问题（包括"有用法律"和"无责任缺口法律"）形式化为超图上的顶点覆盖问题，证明了两类法律最小化问题都是NP-hard的，并给出了基于超图顶点覆盖近似算法的多项式时间近似方案。

**[A Mind Cannot Be Smeared Across Time](a_mind_cannot_be_smeared_across_time.md)**

:   从Stack Theory出发引入时间语义模块，形式化证明存在性时间实现不保持合取（Temporal Gap），提出Chord（要求时间窗口内共同实例化）vs Arpeggio（只需成分在窗口内出现）两种意识假设，论证严格串行硬件在Chord假设下不可能承载需要多个同时贡献者的意识。

**[A New Strategy for Verifying Reach-Avoid Specifications in Neural Feedback Systems](a_new_strategy_for_verifying_reach-avoid_specifications_in_neural_feedback_syste.md)**

:   提出FaBRe（Forward and Backward Reachability）策略，首次开发了针对ReLU神经网络控制器的后向可达集过近似和欠近似算法（GSS/ICH/LEB），并将其与前向可达性分析结合，构成统一的reach-avoid验证框架，旨在突破纯前向分析的可扩展性瓶颈。

**[A Phase Transition for Opinion Dynamics with Competing Biases](a_phase_transition_for_opinion_dynamics_with_competing_biase.md)**

:   在有向随机图上建模两种对立力量（外部颠覆性偏差 vs 个体顽固性）对二元观点传播的影响，证明系统存在尖锐相变：偏差超过临界阈值 $p_c$ 时群体快速达成新共识，低于阈值则长期处于亚稳极化状态，且临界点仅由度序列的两个简单统计量决定。

**[A Switching Framework for Online Interval Scheduling with Predictions](a_switching_framework_for_online_interval_scheduling_with_pr.md)**

:   针对不可撤销的在线区间调度问题，提出 SemiTrust-and-Switch 框架和 SmoothMerge 随机算法，通过在信任预测和经典贪心算法之间切换/融合，在预测准确时趋近最优（一致性），预测错误时性能优雅退化（鲁棒性和平滑性），并证明了该框架在特定实例上的紧性。

**[A Topological Rewriting of Tarski's Mereogeometry](a_topological_rewriting_of_tarskis_mereogeometry.md)**

:   在 Coq 定理证明器中扩展 λ-MM 库，将基于 Leśniewski 部分学（mereology）的 Tarski 固体几何重写为具备完整拓扑结构的形式化系统，证明部分学类对应正则开集、满足 Kuratowski 内部公理且具有 Hausdorff（T2）性质，从而为定性空间推理提供了统一的部分学-几何-拓扑理论框架。

**[Agent-SAMA: State-Aware Mobile Assistant](agent-sama_state-aware_mobile_assistant.md)**

:   提出Agent-SAMA，首次将有限状态机（FSM）引入移动端GUI Agent，将UI屏幕建模为状态、用户操作建模为转移，通过四个专门化Agent协作实现状态感知的任务规划、执行验证和错误恢复，在跨App基准上成功率提升最高12%、恢复率提升13.8%。

**[Align When They Want, Complement When They Need! Human-Centered Ensembles for Adaptive Human-AI Collaboration](align_when_they_want_complement_when_they_need_human-centere.md)**

:   揭示了人机协作中"互补性"（complementarity）与"对齐性"（alignment）之间存在根本性权衡——单一模型无法同时优化二者，提出自适应AI集成框架，通过Rational Routing Shortcut（RRS）机制在对齐模型和互补模型之间动态切换，团队准确率较标准AI提升最高9%。

**[AMS-IO-Bench and AMS-IO-Agent: Benchmarking and Structured Reasoning for Analog and Mixed-Signal Integrated Circuit Input/Output Design](ams-io-bench_and_ams-io-agent_benchmarking_and_structured_re.md)**

:   提出AMS-IO-Agent，一个基于LLM的领域专用智能体，通过结构化意图图(Intent Graph)和领域知识库将自然语言设计意图转化为可生产的模拟混合信号IC I/O环设计，配套提出首个AMS I/O环自动化基准AMS-IO-Bench，在28nm CMOS流片中验证了智能体生成的I/O环可直接用于实际芯片制造。

**[An Epistemic Perspective on Agent Awareness](an_epistemic_perspective_on_agent_awareness.md)**

:   本文首次将 agent awareness（智能体感知/意识）视为一种知识形式，区分了 de re（关于物理对象的）和 de dicto（关于概念/描述的）两种感知模态，并基于 2D 语义学提出了一个可靠且完备的逻辑系统来刻画这两种模态与标准"事实知识"模态之间的相互作用。

**[Approximation Algorithm for Constrained k-Center Clustering: A Local Search Approach](approximation_algorithm_for_constrained_k-center_clustering_.md)**

:   研究带 cannot-link (CL) 和 must-link (ML) 实例级约束的 k-center 聚类问题，提出基于支配匹配集（dominating matching set, DMS）转化的局部搜索框架，在不相交 CL 集条件下首次通过局部搜索达到最优近似比 2，解决了该领域一个开放问题。

**[Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit](area-optimal_control_strategies_for_heterogeneous_multi-agen.md)**

:   研究异构速度下多追逐者-单逃避者的追逃博弈——定义逃避者安全可达集为所有追逐者-逃避者对的 Apollonius 圆的交集，将捕获策略建模为追逐者最小化/逃避者最大化该交集面积的零和博弈，推导出闭式瞬时最优航向控制律，仿真验证追逐者可系统性缩小安全区域实现保证捕获。

**[Automated Reproducibility Has a Problem Statement Problem](automated_reproducibility_has_a_problem_statement_problem.md)**

:   提出基于科学方法的可复现性形式化问题定义，将经验性AI研究表示为假设-实验-解释的图结构，并用LLM自动从20篇论文中提取该结构，经原作者评审验证其有效性。

**[Autonomous Concept Drift Threshold Determination](autonomous_concept_drift_threshold_determination.md)**

:   证明了固定阈值不可能在所有场景下最优、动态阈值严格优于静态阈值，并提出DTD算法：在漂移检测信号触发后启动三模型比较阶段，根据候选模型表现自适应调整检测阈值。

**[Bandit Learning in Housing Markets](bandit_learning_in_housing_markets.md)**

:   本文首次将多臂老虎机（MAB）框架引入住房市场（单边匹配市场），定义了基于核（core）概念的遗憾值，并分别提出去中心化 ETC 和中心化 UCB 两种算法，证明了 $\mathcal{O}(N\log T / \Delta_{\min}^2)$ 的去中心化遗憾上界与匹配的下界，建立了阶最优性。

**[Bayesian Network Structural Consensus via Greedy Min-Cut Analysis](bayesian_network_structural_consensus_via_greedy_min-cut_analysis.md)**

:   提出 MCBNC 算法，基于最小割（min-cut）分析量化边的结构支持度，并将其嵌入贪心等价搜索（GES）的后向阶段来迭代剪枝融合贝叶斯网络中的冗余边，在不访问数据的情况下生成更稀疏、更精确的共识结构，适用于联邦学习场景。

**[Beyond World Models: Rethinking Understanding in AI Models](beyond_world_models_rethinking_understanding_in_ai_models.md)**

:   本文通过三个来自科学哲学的案例研究（多米诺计算机、数学证明、玻尔原子理论），论证世界模型（world models）框架不足以刻画人类级别的"理解"，指出仅靠追踪状态和状态转换无法捕获理解所需的抽象推理、动机洞察和问题情境把握能力。

**[Bilevel MCTS for Amortized O(1) Node Selection in Classical Planning](bilevel_mcts_for_amortized_o1_node_selection_in_classical_planning.md)**

:   提出双层MCTS（Bilevel MCTS），在MCTS选中的叶节点处运行深度比例预算的最优优先搜索，将节点选择均摊复杂度从 $O(\log N)$ 降至 $O(1)$，辅以树崩塌（Tree Collapsing）减少动作选择步数，最终整合为 Nεbula 规划器，在IPC2018/2023基准上以192.2/230.6解题数（5min/30min）超越LAMA、DecStar、NOLAN、SM-Type-LAMA等全部SOTA。

**[Bipartite Mode Matching for Vision Training Set Search from a Hierarchical Data Server](bipartite_mode_matching_for_vision_training_set_search_from_a_hierarchical_data_.md)**

:   提出层级数据服务器 + 二部图模式匹配（BMM）框架，通过多粒度层级聚类组织大规模源数据、用匈牙利算法一对一匹配源域和目标域的语义模式（modes），从而搜索出与目标域分布差距最小的训练集，在行人重识别和目标检测任务上显著优于已有训练集搜索方法。

**[Boosting Adversarial Transferability via Ensemble Non-Attention](boosting_adversarial_transferability_via_ensemble_non-attention.md)**

:   提出 NAMEA（Non-Attention Meta Ensemble Attack），首次利用集成模型的非注意力区域（non-attention areas）融合 CNN 和 ViT 的可迁移信息，结合元学习梯度优化，在跨架构对抗迁移性上平均超越 SOTA 方法 AdaEA 和 SMER 分别 15.0% 和 9.6%。

**[Cash Flow Underwriting with Bank Transaction Data: Advancing MSME Financial Inclusion in Malaysia](cash_flow_underwriting_with_bank_transaction_data_advancing_msme_financial_inclu.md)**

:   提出基于银行流水数据的端到端现金流承保工作流，构建首个马来西亚 MSME（中小微企业）银行账单数据集（611 条贷款记录），验证银行交易衍生特征相比传统申请信息可将逻辑回归模型的 AUROC 从 0.672 提升至 0.850，显著增强对缺乏信用记录的中小微企业的信用评估能力。

**[CAT-Net: A Cross-Attention Tone Network for Cross-Subject EEG-EMG Fusion Tone Decoding](cat-net_a_cross-attention_tone_network_for_cross-subject_eeg-emg_fusion_tone_dec.md)**

:   提出 CAT-Net（Cross-Attention Tone Network），通过空间-时间特征提取分支 + 交叉注意力融合机制 + 域对抗训练，仅用 20 个 EEG 通道和 5 个 EMG 通道实现中文四声调分类，在有声/无声语音条件下分别达到 87.83%/88.08% 准确率，跨被试评估下达到 83.27%/85.10%，全面超越 8 种基线方法。

**[CellStream: Dynamical Optimal Transport Informed Embeddings for Reconstructing Cellular Trajectories from Snapshots Data](cellstream_dynamical_optimal_transport_informed_embeddings_for_reconstructing_ce.md)**

:   提出 CellStream，一种将自编码器与非平衡动态最优传输（unbalanced dynamical OT）联合学习的深度学习框架，从离散时间点的单细胞快照数据中同时学习低维嵌入和连续细胞动态轨迹，在时间一致性和速度一致性上显著优于现有方法。

**[Center-Outward q-Dominance: A Sample-Computable Proxy for Strong Stochastic Dominance in Multi-Objective Optimisation](center-outward_q-dominance_a_sample-computable_proxy_for_strong_stochastic_domin.md)**

:   基于最优传输理论中的中心向外分布函数，提出 q-dominance 关系作为强一阶随机支配（strong FSD）的可计算近似，证明全分位数范围的 q-dominance 可推导出强 FSD，并给出显式样本量阈值控制 Type I 错误，在超参数调优排名和噪声多目标优化中验证了其实用性。

**[Certified Branch-and-Bound MaxSAT Solving (Extended Version)](certified_branch-and-bound_maxsat_solving_extended_version.md)**

:   为 Branch-and-Bound MaxSAT 求解器实现了基于 VeriPB 证明系统的认证，覆盖了 look-ahead 边界方法和多值决策图（MDD）编码两大核心技术，在 MaxCDCL 求解器上的实验表明证明日志的中位开销仅 19%，填补了 MaxSAT 求解范式认证的最后空白。

**[Certified but Fooled! Breaking Certified Defences with Ghost Certificates](certified_but_fooled_breaking_certified_defences_with_ghost_certificates.md)**

:   提出 GhostCert，一种基于显著性区域的对抗攻击方法，能在保持扰动不可感知的同时误导分类器并伪造大半径的认证证书（ghost certificates），在 ImageNet 上对包括 DensePure 在内的 SOTA 认证防御取得显著优于 Shadow Attack 的攻击成功率和更大的伪造认证半径。

**[Clinician-in-the-Loop Smart Home System to Detect Urinary Tract Infection Flare-Ups via Uncertainty-Aware Decision Support](clinician-in-the-loop_smart_home_system_to_detect_urinary_tract_infection_flare-.md)**

:   提出一种临床医师参与闭环的智能家居系统，利用环境传感器数据提取行为标记，结合新颖的共形校准区间（CCI）方法量化预测不确定性，实现对老年人尿路感染（UTI）发作的可靠检测与"不确定时弃权"的决策支持。

**[Controllable Financial Market Generation with Diffusion Guided Meta Agent](controllable_financial_market_generation_with_diffusion_guided_meta_agent.md)**

:   提出Diffusion Guided Meta Agent（DigMA）模型，将可控金融市场生成形式化为条件生成任务，用条件扩散模型捕捉市场状态动态（中间价收益率与订单到达率的时变分布参数），结合具有金融经济学先验的Meta Agent生成订单流，在可控性和生成保真度上均超越现有方法。

**[Cost-Free Neutrality for the River Method](cost-free_neutrality_for_the_river_method.md)**

:   针对River投票方法的并行宇宙打破平局（PUT）问题，证明其获胜者集合可在多项式时间内计算（相比Ranked Pairs的NP-完全性），提出Fused-Universe（FUN）算法，一次遍历同时模拟所有可能的打破平局方式，并为每个获胜者提供构造性证书。

**[Data Complexity of Querying Description Logic Knowledge Bases under Cost-Based Semantics](data_complexity_of_querying_description_logic_knowledge_bases_under_cost-based_s.md)**

:   系统研究加权描述逻辑知识库在代价语义下的查询应答的数据复杂度，证明最优代价语义在$\Delta_2^p$内可解，并给出一个令人惊喜的正面结果：在DL-Lite$_{\text{bool}}^{\mathcal{H}}$本体和固定代价界限下，实例查询的确定回答和合取查询的可能回答可通过一阶重写实现最低数据复杂度（AC$^0$）。

**[Deadline-Aware, Energy-Efficient Control of Domestic Immersion Hot Water Heaters](deadline-aware_energy-efficient_control_of_domestic_immersion_hot_water_heater.md)**

:   提出一种基于截止时间感知的家用热水器节能控制方法，通过 Gymnasium 仿真环境比较 bang-bang 基线、MCTS 规划器和 PPO 策略，证明 PPO 在相同物理条件下能节省高达 69% 的能量。

**[DECOR: Deep Embedding Clustering with Orientation Robustness](decor_deep_embedding_clustering_with_orientation_robustness.md)**

:   提出 DECOR 框架，通过旋转不变的等变卷积自编码器（RCAE）+ 非参数聚类（DeepDPM）+ 集成异常检测，实现晶圆图缺陷模式的方向鲁棒聚类。

**[DeepRWCap: Neural-Guided Random-Walk Capacitance Solver for IC Design](deeprwcap_neural-guided_random-walk_capacitance_solver_for_ic_design.md)**

:   提出 DeepRWCap，一种机器学习引导的随机游走电容求解器，通过两阶段神经网络架构预测转移核来加速IC设计中的多介质域电容提取，在10个工业测试案例上实现平均1.24%误差和23%加速。

**[Depth-Synergized Mamba Meets Memory Experts for All-Day Image Reflection Separation](depth-synergized_mamba_meets_memory_experts_for_all-day_image_reflection_separat.md)**

:   提出 DMDNet，通过深度感知扫描策略（DAScan）引导 Mamba 关注显著结构，结合深度协同状态空间模型（DS-SSM）抑制模糊特征传播，并引入记忆专家补偿模块（MECM）利用跨图像历史知识，实现全天候（白天+夜间）的图像反射分离。

**[Description Logics with Two Types of Definite Descriptions: Complexity, Expressiveness, and Automated Deduction](description_logics_with_two_types_of_definite_descriptions_complexity_expressive.md)**

:   引入描述逻辑 ALC 的两种定冠描述扩展——局部定冠描述 $\{ι C\}$ 和全局定冠描述 $ι C.D$，证明三个扩展逻辑的可满足性问题均为 ExpTime-complete，但全局定冠描述严格比局部更具表达力（$\mathcal{ALC}\iota_L < \mathcal{ALC}\iota_G = \mathcal{ALC}\iota$），并给出表列演算决策过程及实验评估。

**[Designing Incident Reporting Systems for Harms from General-Purpose AI](designing_incident_reporting_systems_for_harms_from_general-purpose_ai.md)**

:   通过文献综述和九个安全关键行业（核能、航空、医疗等）的案例研究，提出了 AI 事件报告系统制度设计的七维框架，为美国通用 AI 事件报告的政策设计提供系统性指导。

**[DeToNATION: Decoupled Torch Network-Aware Training on Interlinked Online Nodes](detonation_decoupled_torch_network-aware_training_on_interlinked_online_nodes.md)**

:   提出 FlexDeMo——一种将全分片数据并行（FSDP）与解耦动量优化相结合的混合分片训练策略，在节点内使用 FSDP 分片、节点间仅同步快速移动的动量分量，实现了接近全同步 AdamW 的损失收敛同时显著加速训练。

**[DFDT: Dynamic Fast Decision Tree for IoT Data Stream Mining on Edge Devices](dfdt_dynamic_fast_decision_tree_for_iot_data_stream_mining_on_edge_devices.md)**

:   提出 DFDT（Dynamic Fast Decision Tree），一种面向 IoT 边缘设备的内存受限数据流挖掘算法，通过活动感知预剪枝、动态 grace period、自适应 tie threshold 三重机制有机调控树的增长，实现精度-内存-运行时间的最优权衡。

**[DiffMM: Efficient Method for Accurate Noisy and Sparse Trajectory Map Matching via One Step Diffusion](diffmm_efficient_method_for_accurate_noisy_and_sparse_trajectory_map_matching_vi.md)**

:   提出 DiffMM，首次将扩散模型引入地图匹配任务，通过路段感知轨迹编码器和一步 Shortcut 扩散过程，在稀疏轨迹和复杂路网上实现了精度和效率的双重提升，推理速度比次优方法快约 17 倍。

**[DS-ATGO: Dual-Stage Synergistic Learning via Forward Adaptive Threshold and Backward Gradient Optimization for Spiking Neural Networks](ds-atgo_dual-stage_synergistic_learning_via_forward_adaptive_threshold_and_backw.md)**

:   针对SNN训练中因膜电位分布偏移导致的脉冲发放不均衡和梯度消失问题，提出前向自适应阈值+后向阈值驱动梯度优化的双阶段协同学习算法DS-ATGO，在CIFAR10/100和ImageNet上以低时延实现SOTA性能。

**[DW-DGAT: Dynamically Weighted Dual Graph Attention Network for Neurodegenerative Disease Diagnosis](dw-dgat_dynamically_weighted_dual_graph_attention_network_for_neurodegenerative_.md)**

:   针对神经退行性疾病（PD/AD）早期诊断中的多指标数据融合、异质信息提取和类别不平衡三大挑战，提出动态加权双图注意力网络DW-DGAT，通过通用数据融合策略、微观-宏观双层图特征学习和动态类别权重生成机制，在PPMI和ADNI3数据集上大幅超越14种基线方法。

**[Enhancing Control Policy Smoothness by Aligning Actions with Predictions from Preceding States](enhancing_control_policy_smoothness_by_aligning_actions_with_predictions_from_pr.md)**

:   提出 **ASAP（Action Smoothing by Aligning Actions with Predictions from Preceding States）**，一种基于**转移诱导相似状态定义**的强化学习动作平滑方法，通过空间约束（对齐前一状态的预测动作）和时间约束（惩罚二阶动作差异）有效抑制高频动作振荡，在 Gymnasium 和 Isaac-Lab 环境中优于现有方法。

**[Expandable and Differentiable Dual Memories with Orthogonal Regularization for Exemplar-free Continual Learning](expandable_and_differentiable_dual_memories_with_orthogonal_regularization_for_e.md)**

:   提出 **EDD（Expandable and Differentiable Dual Memory）**，一种**无需存储旧样本**的持续学习方法，通过**可微分的共享记忆和任务特定记忆**将数据分解为可复用的子特征，结合**记忆扩展-剪枝**和**正交正则化**机制，在 CIFAR-10/100 和 Tiny-ImageNet 上超越 14 种 SOTA 方法，最终准确率分别达到 55.13%、37.24% 和 30.11%。

**[Expressive Temporal Specifications for Reward Monitoring](expressive_temporal_specifications_for_reward_monitoring.md)**

:   利用量化线性时序逻辑（LTLf[F]）自动合成**量化奖励监控器（QRM）**，为强化学习智能体在运行时生成密集的连续值奖励流，从根本上缓解布尔语义下长时任务的稀疏奖励问题。

**[Extreme Value Monte Carlo Tree Search for Classical Planning](extreme_value_monte_carlo_tree_search_for_classical_planning.md)**

:   利用 Peaks-Over-Threshold 极值理论（POT EVT）为经典规划中 MCTS 的 Full Bellman Backup 提供统计理论基础，提出 UCB1-Uniform bandit 算法，用均匀分布（Generalized Pareto 的特例）的 MLE 估计指导动作选择，在 Pyperplan 上以 $10^4$ 节点预算超越 GBFS 67.8 个实例、超越 Softmin-Type(h) 33.2 个实例。

**[Faster Certified Symmetry Breaking Using Orders With Auxiliary Variables](faster_certified_symmetry_breaking_using_orders_with_auxiliary_variables.md)**

:   通过引入辅助变量编码字典序来替代大整数编码，对 VeriPB 证明系统进行本质重设计，使 SAT 对称性破坏的证明生成和验证在理论和实践上均获得数量级加速。

**[Finding Diverse Solutions Parameterized by Cliquewidth](finding_diverse_solutions_parameterized_by_cliquewidth.md)**

:   将"寻找多样化解"的参数化框架从treewidth扩展到更强的cliquewidth图参数，证明任何基于cliquewidth分解的单调动态规划都可以以极小额外开销转换为求解多样化版本的算法，并提出了一族新的Venn多样性度量函数。

**[Formal Abductive Latent Explanations for Prototype-Based Networks](formal_abductive_latent_explanations_for_prototype-based_networks.md)**

:   本文针对原型网络（如ProtoPNet）的解释可能具有误导性的问题，提出了溯因潜在解释（ALE），在潜在空间中构造满足形式化保证的充分条件解释，无需调用外部求解器，算法可扩展到多种数据集上的标准分类和细粒度分类任务。

**[From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm](from_decision_trees_to_boolean_logic_a_fast_and_unified_shap_algorithm.md)**

:   本文提出Woodelf算法，通过将决策树集成模型转化为加权析取范式（WDNF）的伪布尔公式，在统一框架下实现了Background SHAP和Path-Dependent SHAP的线性时间计算，在大规模数据集上实现CPU 16-31倍、GPU 24-333倍的加速。

**[From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback](from_sequential_to_recursive_enhancing_decision-focused_learning_with_bidirectio.md)**

:   本文首次提出递归决策聚焦学习（R-DFL）框架，通过在预测模块与优化模块之间引入双向反馈回路，突破了传统顺序式 DFL 的单向信息流限制，并设计了显式展开和隐式微分两种梯度传播方法，在报童问题和二部匹配问题上显著提升了最终决策质量。

**[Guided Perturbation Sensitivity (GPS): Detecting Adversarial Text via Embedding Stability and Word Importance](guided_perturbation_sensitivity_gps_detecting_adversarial_text_via_embedding_sta.md)**

:   提出 Guided Perturbation Sensitivity (GPS) 框架，通过对重要词进行遮蔽并测量嵌入表示的稳定性变化来检测对抗文本样本，在3个数据集、3种攻击、2个模型上实现85%+检测准确率，且无需重训练即可跨数据集/攻击/模型泛化。

**[CAE: Hierarchical Semantic Alignment for Image Clustering](hierarchical_semantic_alignment_for_image_clustering.md)**

:   结合名词级（WordNet）和描述级（Flickr 图片描述）两种互补语义，通过最优传输对齐构建语义空间并自适应融合，实现 training-free 的图像聚类，在 ImageNet-1K 上准确率提升 4.2%。

**[Higher-Order Responsibility](higher-order_responsibility.md)**

:   本文研究顺序决策机制中的高阶责任问题，证明了两个核心定理：(1) $n$ 个智能体的机制必然是 $n$ 阶无间隙的（即总能找到某阶责任人）；(2) 判定机制是否为 $d$ 阶无间隙的问题是 $\Pi_{2d+1}$-完全的。

**[How Hard is it to Explain Preferences Using Few Boolean Attributes?](how_hard_is_it_to_explain_preferences_using_few_boolean_attributes.md)**

:   本文系统研究了用布尔属性模型（BAM）解释偏好数据的计算复杂性：证明了当属性数 $k \geq 3$ 时问题是NP完全的，$k \leq 2$ 时线性可解；进一步对投票人数 $n$、候选项数 $m$、属性数 $k$ 等参数给出了完整的参数化复杂性全景图，并分析了已知部分信息（cares/has）时问题难度的变化。

**[How Hard Is It to Rig a Tournament When Few Players Can Beat or Be Beaten by the Favorite?](how_hard_is_it_to_rig_a_tournament_when_few_players_can_beat_or_be_beaten_by_the.md)**

:   本文提出两个新的结构化参数——目标选手在锦标赛有向图中的入度 $k$ 和出度 $\ell$——用于分析锦标赛赛程操纵问题 (TFP)，证明 TFP 在以这两个参数为参数时均是 FPT 的，其中入度参数化的算法设计涉及复杂的结构分析和颜色编码技术。

**[How to Marginalize in Causal Structure Learning?](how_to_marginalize_in_causal_structure_learning.md)**

:   本文利用可处理概率电路（Probabilistic Circuits）替代传统动态规划方法来执行贝叶斯结构学习中的边际化任务，通过一种新颖的两阶段训练策略（先学习完整父集分数再渐进式微调边际查询），消除了候选父节点集数量的人为限制，从而在 TRUST 框架上取得了更好的后验分布估计效果。

**[How Wide and How Deep? Mitigating Over-Squashing of GNNs via Channel Capacity Constrained Estimation](how_wide_and_how_deep_mitigating_over-squashing_of_gnns_via_channel_capacity_con.md)**

:   本文从信息论视角出发，将谱图神经网络建模为通信信道，提出信道容量约束估计框架 C3E，将 GNN 隐藏维度与深度的选择形式化为一个非线性规划问题，在训练前即可估计最优架构参数，有效缓解信息过度压缩（over-squashing），在 9 个数据集上一致提升了表示学习效果。

**[HyperSHAP: Shapley Values and Interactions for Explaining Hyperparameter Optimization](hypershap_shapley_values_and_interactions_for_explaining_hyperparameter_optimiza.md)**

:   HyperSHAP 提出一套基于 Shapley 值和 Shapley 交互的博弈论框架来解释超参数优化（HPO），通过定义消融、灵敏度、可调性和优化器偏差四类解释博弈，提供比 fANOVA 更具可操作性的超参数重要性分析。

**[I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks](i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)**

:   I2E 提出一个超高效的图像到事件流转换框架，通过模拟微扫视眼动并用高度并行化的卷积实现比先前方法快 300 倍以上的转换速度，首次支持 SNN 训练的在线数据增强，在 I2E-ImageNet 上达到 60.50% 的事件分类 SOTA，并通过合成数据预训练 + 真实数据微调的 sim-to-real 范式在 CIFAR10-DVS 上创下 92.5% 的历史最佳。

**[Improved Differentially Private Algorithms for Rank Aggregation](improved_differentially_private_algorithms_for_rank_aggregation.md)**

:   针对差分隐私下的排名聚合问题，提出了改进的近似算法：首次研究footrule排名聚合问题并给出近最优算法（可推导出Kemeny问题的2-近似），同时通过结合二路边际查询和无偏估计技术改进了Kemeny排名聚合的PTAS加性误差（指数从3降至65/22）。

**[Incremental Maintenance of DatalogMTL Materialisations](incremental_maintenance_of_datalogmtl_materialisations.md)**

:   提出 DRed$_{\text{MTL}}$ 算法，将经典 Delete/Rederive 增量维护技术扩展到 DatalogMTL（带度量时序逻辑的 Datalog），通过在周期化物化表示上设计新的 seminaïve 评估算子和周期识别算法，实现高效增量更新，性能可达重新物化的数量级提升。

**[Intermediate N-Gramming: Deterministic and Fast N-Grams For Large N and Large Datasets](intermediate_n-gramming_deterministic_and_fast_n-grams_for_large_n_and_large_dat.md)**

:   提出 Intergrams 多遍扫描算法，利用较短 n-gram 作为前缀递推过滤候选更长 n-gram，充分利用处理器缓存层次结构实现缓存友好的内存访问模式，在 TB 级数据集上比此前最快的 hash-gramming 方法加速 6-33 倍，同时几乎精确恢复所有 top-k n-gram。

**[Intrinsic Barriers and Practical Pathways for Human-AI Alignment: An Agreement-Based Complexity Analysis](intrinsic_barriers_and_practical_pathways_for_human-ai_alignment_an_agreement-ba.md)**

:   本文将 AI 对齐形式化为 $\langle M,N,\varepsilon,\delta\rangle$-agreement 多目标优化问题，从通信复杂度角度证明了对齐的信息论下界（编码"所有人类价值观"本质上不可行），同时给出了无界/有界理性智能体的显式可达算法和紧致上界，揭示了在大状态空间下 reward hacking 全局不可避免的理论根基。

**[Judging by the Rules: Compliance-Aligned Framework for Modern Slavery Statement Monitoring](judging_by_the_rules_compliance-aligned_framework_for_modern_slavery_statement_m.md)**

:   提出以"合规对齐法官"（CA-Judge）为核心的训练框架，利用规则级对齐反馈训练 3B 参数的 CALLM 模型，使其生成基于法定条款的可追溯合规判断理由，在现代奴役声明的句子级合规分类任务上超越 GPT-4o 和 DeepSeek-R1。

**[LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval](leanrag_knowledge-graph-based_generation_with_semantic_aggregation_and_hierarchi.md)**

:   提出 LeanRAG 框架，通过语义聚合算法在层次化知识图谱的摘要节点间自动构建显式关系打破"语义孤岛"，并基于最近公共祖先（LCA）的自底向上检索策略高效导航层次结构，在四个 QA 基准上取得 SOTA 同时减少 46% 的检索冗余。

**[Learning Compact Latent Space for Representing Neural Signed Distance Functions with High-fidelity Geometry Details](learning_compact_latent_space_for_representing_neural_signed_distance_functions_.md)**

:   提出一种双分支架构（泛化分支+过拟合分支）来学习多个神经SDF的紧凑潜空间，结合共享spatial feature grid和新颖的带宽采样策略，在保持紧凑latent code的同时恢复高保真几何细节，在Stanford Models、ShapeNet和D-FAUST上均达到SOTA。

**[Learning Fair Representations with Kolmogorov-Arnold Networks](learning_fair_representations_with_kolmogorov-arnold_networks.md)**

:   提出将Kolmogorov-Arnold网络（KAN）引入对抗去偏框架，利用KAN的样条函数架构提供理论上的Lipschitz连续性和平滑性保证，并设计自适应 $\lambda$ 更新机制动态平衡公平性与准确率，在UCI大学录取数据集上实现了公平性指标的显著提升。

**[Learning Network Dismantling Without Handcrafted Inputs](learning_network_dismantling_without_handcrafted_inputs.md)**

:   提出MIND（Message Iteration Network Dismantler），通过全新的All-to-One注意力机制和消息迭代轮廓（Message Iteration Profiles）消除GNN对手工特征的依赖，仅利用原始邻接信息就能在百万节点级真实网络上实现SOTA的网络拆解性能，同时具有最低的计算复杂度 $O(|V|+|E|)$。

**[LILAD: Learning In-context Lyapunov-stable Adaptive Dynamics Models](lilad_learning_in-context_lyapunov-stable_adaptive_dynamics_models.md)**

:   提出 LILAD 框架，利用 GPT-2 的 in-context learning 能力同时学习动力学模型和 Lyapunov 函数，在保证全局指数稳定性的同时实现对非平稳参数化动力系统的自适应辨识，在多个基准系统上超越 ICL、MAML 等基线。

**[Local Guidance for Configuration-Based Multi-Agent Pathfinding](local_guidance_for_configuration-based_multi-agent_pathfinding.md)**

:   提出局部引导（Local Guidance）概念改进 LaCAM 的多智能体路径规划，通过在每个配置生成步为每个智能体构造局部时空路径来缓解拥塞，最高可将解的代价降低 50%，同时保持 1000 智能体下几秒内完成。

**[Lost in Time? A Meta-Learning Framework for Time-Shift-Tolerant Physiological Signal Transformation](lost_in_time_a_meta-learning_framework_for_time-shift-tolerant_physiological_sig.md)**

:   提出 ShiftSyncNet，一个基于元学习双层优化的框架，通过 SyncNet 学习训练样本对之间的时间偏移量并利用傅里叶变换的相移性质自动校正标签对齐，在三个数据集上分别提升了 9.4%、6.0% 和 12.8% 的波形转换精度。

**[Measuring Model Performance in the Presence of an Intervention](measuring_model_performance_in_the_presence_of_an_intervention.md)**

:   针对存在干预（intervention）时 AI 模型评估偏差的问题，提出 Nuisance Parameter Weighting (NPW) 方法，通过对 RCT 治疗组数据进行因果加权，实现无偏的 AUROC 估计，使样本效率提升 5 倍，显著改善了模型选择和假设检验的统计功效。

**[MeshA*: Efficient Path Planning With Motion Primitives](mesha_efficient_path_planning_with_motion_primitives.md)**

:   提出 MeshA* 算法，将 lattice-based 路径规划从"在运动基元层面搜索"转变为"在网格单元层面搜索并同时拟合基元序列"，通过定义"扩展网格单元"（extended cell）新搜索空间，在保证完备性和最优性的同时，实现相比标准 LBA* 1.5x-2x 的运行时加速。

**[MF-Speech: Achieving Fine-Grained and Compositional Control in Speech Generation via Factor Disentanglement](mf-speech_achieving_fine-grained_and_compositional_control_in_speech_generation_.md)**

:   提出MF-Speech框架，通过多目标优化将语音信号解耦为高纯度的内容、音色和情绪三个独立因子表示，再利用动态融合和层级风格自适应归一化（HSAN）实现细粒度的组合式语音生成控制，在多因子组合语音生成任务上显著超越现有方法（WER=4.67%, SECS=0.5685）。

**[Model Counting for Dependency Quantified Boolean Formulas](model_counting_for_dependency_quantified_boolean_formulas.md)**

:   本文首次研究了依赖量化布尔公式（DQBF）的模型计数问题，证明了即使仅含两个存在量词变量的 #2-DQBF 就已是 #EXP-完全的，并基于 BDD 符号可达性技术实现了一个实用的 2-DQBF 模型计数器 sharp2DQR，在大依赖集上显著优于基于展开的基线方法。

**[On the Edge of Core (Non-)Emptiness: An Automated Reasoning Approach to Approval-Based Multi-Winner Voting](on_the_edge_of_core_non-emptiness_an_automated_reasoning_approach_to_approval-ba.md)**

:   针对基于认可的多赢者投票中核稳定性（core stability）是否总存在这一重大开放问题，提出基于混合整数线性规划（MILP）的自动推理框架，证明了新的存在性结果，发现了核稳定性与其他公理（如 Lindahl 可定价性）之间此前未知的关系，并推翻了一个已有猜想。

**[On the Information Processing of One-Dimensional Wasserstein Distances with Finite Samples](on_the_information_processing_of_one-dimensional_wasserstein_distances_with_fini.md)**

:   本文通过Poisson过程框架，解析刻画了一维Wasserstein距离在有限样本下同时编码概率密度函数的逐点密度差异（rate difference）和支撑差异（support difference）的能力，并在神经脉冲数据和氨基酸接触频率数据上验证了其实际价值。

**[On the Variability of Concept Activation Vectors](on_the_variability_of_concept_activation_vectors.md)**

:   对 TCAV 方法中概念激活向量（CAV）的变异性进行首次理论分析，证明 CAV 的方差以 $O(1/N)$ 速率衰减（$N$ 为随机样本数），而 TCAV 分数的方差因"边界点"保持 $O(1)$，需通过多次运行平均以 $O(1/s)$ 降低。

**[Online Linear Regression with Paid Stochastic Features](online_linear_regression_with_paid_stochastic_features.md)**

:   研究了在线线性回归中特征被噪声污染、学习者可以**付费降低噪声强度**的新问题设定，证明了已知噪声协方差时最优遗憾率为 $\widetilde{\mathcal{O}}(\sqrt{T})$、未知时为 $\widetilde{\mathcal{O}}(T^{2/3})$，并给出匹配的下界，所有界关于时间 $T$ 的依赖都是阶最优的。

**[Optimal Welfare in Noncooperative Network Formation under Attack](optimal_welfare_in_noncooperative_network_formation_under_attack.md)**

:   在Goyal等人(WINE 2016)提出的非合作网络形成博弈模型中，证明了自私智能体创建的均衡网络在面对包括maximum disruption在内的广泛攻击者类别（超二次扰动攻击者SQD）时，仍能维持渐近最优的社会福利$n^2 - O(n)$，解决了一个长期开放问题。

**[OR-R1: Automating Modeling and Solving of Operations Research Optimization Problems](or-r1_automating_modeling_and_solving_of_operations_research_optimization_proble.md)**

:   OR-R1提出了一个数据高效的两阶段训练框架（SFT + TGRPO），仅使用ORLM所需1/10的合成数据即达到67.7%的平均求解准确率，超越现有SOTA方法，并通过测试时强化学习将单次生成（Pass@1）与多次生成（Pass@8）的性能差距从13%缩小到7%。

**[ParaMETA: Towards Learning Disentangled Paralinguistic Speaking Styles Representations](parameta_towards_learning_disentangled_paralinguistic_speaking_styles_representa.md)**

:   提出 ParaMETA，一种统一的副语言说话风格表示学习框架，通过 META 空间正则化和任务特定子空间投影实现情感、年龄、性别、语言等说话风格的解耦表示，同时支持下游的多任务分类和风格可控语音合成。

**[Parameterized Approximation Algorithms for TSP on Non-Metric Graphs](parameterized_approximation_algorithms_for_tsp_on_non-metric_graphs.md)**

:   本文针对非度量图上的旅行商问题（TSP），提出了关于参数 $p$（违反三角不等式的顶点数）和 $q$（最小违反集大小）的改进FPT近似算法，将 $p$ 参数下的近似比从2.5改进到1.5，$q$ 参数下从11改进到3。

**[PIPHEN: Physical Interaction Prediction with Hamiltonian Energy Networks](piphen_physical_interaction_prediction_with_hamiltonian_energy_networks.md)**

:   提出PIPHEN分布式物理认知-控制框架，通过物理交互预测网络（PIPN）进行"语义蒸馏"将高维感知数据压缩至原始数据量的5%以下，再由基于哈密顿能量守恒的HEN控制器生成协调动作，从而解决多机器人系统的"共享大脑困境"。

**[Predict and Resist: Long-Term Accident Anticipation under Sensor Noise](predict_and_resist_long-term_accident_anticipation_under_sensor_noise.md)**

:   提出统一框架，将基于扩散模型的双层去噪模块与时间感知的Actor-Critic强化学习模型结合，在传感器噪声条件下实现鲁棒的长期交通事故预测，在三个基准数据集上取得了准确率（AP）和平均事故前预警时间（mTTA）的最优性能。

**[Private Frequency Estimation via Residue Number Systems](private_frequency_estimation_via_residue_number_systems.md)**

:   提出 ModularSubsetSelection (MSS)，一种基于剩余数系统（RNS）的本地差分隐私频率估计协议，在保持与 SubsetSelection 和 PGR 相当的估计精度的同时，显著降低通信开销（比 SS 减少达一半）、大幅加速服务器解码（比 PGR 快 11-448 倍）、并实现最低的数据重建攻击成功率。

**[Provably Data-Driven Projection Method for Quadratic Programming](provably_data-driven_projection_method_for_quadratic_programming.md)**

:   将数据驱动的投影矩阵学习从线性规划（LP）扩展到凸二次规划（QP），通过提出"展开主动集方法"在 Goldberg-Jerrum 框架下建模 QP 最优值的计算过程，从而建立了投影矩阵学习的伪维度上界和泛化保证。

**[RcAE: Recursive Reconstruction Framework for Unsupervised Industrial Anomaly Detection](rcae_recursive_reconstruction_framework_for_unsupervised_industrial_anomaly_dete.md)**

:   提出递归卷积自编码器（RcAE），通过参数共享的多步迭代重建逐步抑制异常并保留正常细节，配合跨递归检测模块（CRD）利用多步重建动态实现鲁棒的异常定位，在仅需10%扩散模型参数的条件下达到可比的SOTA性能。

**[Reimagining Anomalies: What if Anomalies Were Normal?](reimagining_anomalies_what_if_anomalies_were_normal.md)**

:   提出首个面向无监督图像异常检测的反事实解释框架，通过训练生成器将异常样本修改为被检测器视为正常的多个解纠缠反事实，从语义层面回答“如果异常是正常的，它应该是什么样子？”，提供远超传统热力图的深层解释能力。

**[Rethinking Flow and Diffusion Bridge Models for Speech Enhancement](rethinking_flow_and_diffusion_bridge_models_for_speech_enhancement.md)**

:   本文提出了一个统一的理论框架，将语音增强中的 flow matching、score-based diffusion 和 Schrödinger bridge 模型统一为在配对数据之间构造不同高斯概率路径的过程，并揭示了这类生成模型每一步采样本质上等价于预测式语音增强，进而利用预测范式中的高性能骨干网络、改进损失函数和微调策略来增强桥模型性能。

**[Reward Redistribution via Gaussian Process Likelihood Estimation](reward_redistribution_via_gaussian_process_likelihood_estimation.md)**

:   本文提出了基于高斯过程似然的奖励重分配框架 GP-LRR，通过核函数显式建模 state-action 对之间的相关性，利用 leave-one-out 策略最大化轨迹回报的边际似然来学习逐步奖励函数，理论证明传统 MSE 方法是其退化特例，并在 MuJoCo 基准上配合 SAC 实现了优越的样本效率和策略性能。

**[Semi-Supervised High Dynamic Range Image Reconstructing via Bi-Level Uncertain Area Masking](semi-supervised_high_dynamic_range_image_reconstructing_via_bi-level_uncertain_a.md)**

:   提出半监督 HDR 重建框架，通过**不确定性估计分支**评估伪 HDR 标签质量，在 patch 和像素两个层面掩码不可靠区域，仅用 6.7% HDR 真值即可达到与全监督 SOTA 可比的性能。

**[ShortageSim: Simulating Drug Shortages under Information Asymmetry](shortagesim_simulating_drug_shortages_under_information_asymmetry.md)**

:   提出 ShortageSim，首个基于 **LLM 多智能体**的药品短缺模拟框架，建模 FDA 监管者、制造商和购买者在信息不对称下的战略决策，在历史短缺数据上实现对解决滞后时间 84% 的预测改善，为监管策略评估提供受控测试平台。

**[Spike Imaging Velocimetry: Dense Motion Estimation of Fluids Using Spike Cameras](spike_imaging_velocimetry_dense_motion_estimation_of_fluids_using_spike_cameras.md)**

:   提出 Spike Imaging Velocimetry（SIV），首次将**脉冲相机**（20000Hz 超高时间分辨率）应用于流体测速，设计细节保持层次变换（DPHT）、图编码器（GE）和多尺度速度精炼（MSVR）三个针对流体特性的模块，并构建了 PSSD 数据集，在稳态湍流、高速流和 HDR 场景上全面超越现有基线。

**[STEM Faculty Perspectives on Generative AI in Higher Education](stem_faculty_perspectives_on_generative_ai_in_higher_education.md)**

:   通过对美国一所大型公立大学 29 名 STEM 教师的焦点小组研究，揭示教师如何在教学中整合 GenAI、观察到的学生学习益处与挑战，以及所需的制度支持——发现 GenAI 将教师劳动从内容创建转向专家审核，且可能掩盖学生能力不足。

**[Structural Approach to Guiding a Present-Biased Agent](structural_approach_to_guiding_a_present-biased_agent.md)**

:   在Kleinberg-Oren模型的委托-代理扩展中，系统性地研究了T-path-Editing问题的参数化复杂性，给出了以树宽和路径代价多样性为参数的FPT算法，并建立了紧的困难性结果，全面刻画了引导当前偏差agent完成关键任务的可处理-不可处理边界。

**[Structure-Aware Encodings of Argumentation Properties for Clique-width](structure-aware_encodings_of_argumentation_properties_for_clique-width.md)**

:   本文设计了从抽象论辩问题到(Q)SAT的有向分解引导(DDG)归约，线性保持团宽(clique-width)，为所有常见论辩语义（stable、admissible、complete、preferred、semi-stable、stage）在扩展存在性、论元接受性和计数问题上建立了以团宽为参数的可处理性上界，并证明了在ETH假设下这些归约的开销不可显著改进。

**[SVD-NO: Learning PDE Solution Operators with SVD Integral Kernels](svd-no_learning_pde_solution_operators_with_svd_integral_kernels.md)**

:   提出 SVD-NO，通过显式参数化积分核的奇异值分解（SVD）来构建神经算子，在保持高表达力的同时实现 $O(ndL)$ 的线性计算复杂度，在 5 个 PDE 基准上达到新 SOTA。

**[Symbolic Planning and Multi-Agent Path Finding in Extremely Dense Environments with Unassigned Agents](symbolic_planning_and_multi-agent_path_finding_in_extremely_dense_environments_w.md)**

:   提出 Block Rearrangement Problem (BRaP) 形式化定义，并设计五种基于配置空间搜索、PDDL 符号规划和 MAPF 的求解算法，其中 BR-LaCAM 在最大 80×80 的极端密集网格上达到 92% 成功率和毫秒级求解速度。

**[SynWeather: Weather Observation Data Synthesis across Multiple Regions and Variables via a General Diffusion Transformer](synweather_weather_observation_data_synthesis_across_multiple_regions_and_variab.md)**

:   构建了首个支持统一多区域多变量的气象观测数据合成数据集SynWeather（覆盖4个区域×4种变量×6颗卫星），并提出基于Diffusion Transformer的通用概率生成模型SynWeatherDiff，通过文本提示引导在多个合成任务上超越专用模型和现有通用模型。

**[Tab-PET: Graph-Based Positional Encodings for Tabular Transformers](tab-pet_graph-based_positional_encodings_for_tabular_transformers.md)**

:   Tab-PET 提出从表格特征间关联关系中估计图结构，利用图拉普拉斯特征向量构造位置编码（PE）注入 Tabular Transformer，理论和实验均证明 PE 可降低嵌入的有效秩从而提升泛化，在 50 个数据集上为 TabTransformer / SAINT / FT-Transformer 带来一致改进，且 Spearman 关联图效果最佳。

**[TaylorPODA: A Taylor Expansion-Based Method to Improve Post-Hoc Attributions for Opaque Models](taylorpoda_a_taylor_expansion-based_method_to_improve_post-hoc_attributions_for_.md)**

:   在Taylor展开框架下提出精确性(precision)、联合性(federation)、零偏差(zero-discrepancy)三个公设规范特征归因，并引入自适应属性(adaptation)通过AUP目标优化交互效应的分配权重，成为唯一同时满足所有公设和属性的事后模型无关归因方法。

**[TDSNNs: Competitive Topographic Deep Spiking Neural Networks for Visual Cortex Modeling](tdsnns_competitive_topographic_deep_spiking_neural_networks_for_visual_cortex_mo.md)**

:   提出拓扑深度脉冲神经网络（TDSNNs），通过设计时空约束（STC）损失函数，在深层SNN中成功复现灵长类视觉皮层从V1到IT的层级拓扑组织，既保持了高任务性能（ImageNet top-1无精度下降），又显著超越现有拓扑ANN的脑相似性。

**[The Limitations and Power of NP-Oracle-Based Functional Synthesis Techniques](the_limitations_and_power_of_np-oracle-based_functional_synthesis_techniques.md)**

:   系统性地从理论层面研究了依赖NP预言机的函数综合方法的能力与局限，证明了朴素的逐位学习方法在多输出场景下必然失败、基于Resolution插值的方法会产生指数级电路，同时证明了NP预言机是高效综合的必要条件，并给出了利用NP预言机在多项式时间内综合小规模Skolem函数的正面结果。

**[The Publication Choice Problem](the_publication_choice_problem.md)**

:   提出"出版选择问题"这一博弈论框架，建模研究者的出版策略与期刊影响力之间的双向互动关系，证明了纯策略均衡的存在性和唯一性，并分析了 Spotlight 论文标签对学术生态的影响。

**[Theoretical and Empirical Analysis of Lehmer Codes to Search Permutation Spaces with Evolutionary Algorithms](theoretical_and_empirical_analysis_of_lehmer_codes_to_search_permutation_spaces_.md)**

:   首次对 Lehmer 编码（逆序向量）在进化算法中搜索排列空间的效率进行严格的数学运行时分析，证明 Lehmer 编码的 EA 在多数基准函数上达到 $O(n^2 \log n)$ 或 $O(n^2)$ 的期望运行时间，与经典表示持平或更优，并在 LOP 和 QAP 实际问题上验证其实用性。

**[Towards Temporal Fusion Beyond the Field of View for Camera-based Semantic Scene Completion](towards_temporal_fusion_beyond_the_field_of_view_for_camera-based_semantic_scene.md)**

:   提出 C3DFusion 模块，通过在 3D 空间中显式对齐历史帧和当前帧的点特征，首次系统解决相机基 SSC 中视野外（out-of-frame）区域的时序补全问题，在 SemanticKITTI 和 SSCBench-KITTI-360 上取得 SOTA。

**[Tractable Weighted First-Order Model Counting with Bounded Treewidth Binary Evidence](tractable_weighted_first-order_model_counting_with_bounded_treewidth_binary_evid.md)**

:   提出一种在域大小上多项式时间的算法，计算带有有界树宽二元证据的 $\text{FO}^2$ 和 $\text{C}^2$ 片段的加权一阶模型计数（WFOMC），并解决了有界树宽有界度图上的稳定座位安排开放问题。

**[Variance Computation for Weighted Model Counting with Knowledge Compilation Approach](variance_computation_for_weighted_model_counting_with_knowledge_compilation_appr.md)**

:   本文将加权模型计数 (WMC) 的权重视为具有方差的随机变量，提出在 structured d-DNNF 表示上多项式时间计算 WMC 方差的算法，同时证明了在 structured DNNF、d-DNNF 和 FBDD 上该问题不可解（除非 P=NP），并将其应用于贝叶斯网络推理中参数不确定性的量化。

**[Verification-Guided Context Optimization for Tool Calling via Hierarchical LLMs-as-editors](verification-guided_context_optimization_for_tool_calling_via_hierarchical_llms-.md)**

:   提出VGCO框架，利用LLM作为分层编辑器，通过验证引导的方式迭代优化工具文档和知识库上下文，显著提升大规模工具调用场景下的检索召回、工具选择和参数填充准确率。

**[Whispering Agents: An Event-Driven Covert Communication Protocol for the Internet of Agents](whispering_agents_an_event-driven_covert_communication_protocol_for_the_internet.md)**

:   首次形式化定义了智能体互联网（IoA）中的"隐蔽事件通道"（Covert Event Channel），并设计了 ΠCCAP 协议，通过在智能体对话的存储、时序和行为维度中嵌入秘密数据，实现了高容量、高鲁棒性且对 LLM 审查员不可感知的隐蔽通信。

**[Why Isn't Relational Learning Taking Over the World?](why_isnt_relational_learning_taking_over_the_world.md)**

:   本文是一篇 position paper，深入分析了关系学习（relational learning）未能在 AI 领域占据主导地位的原因——包括数据集不够真实、评估方法存在根本缺陷、缺少负例、聚合操作的理论困境等——并指出了让关系学习发挥应有潜力所需的关键改进方向。
