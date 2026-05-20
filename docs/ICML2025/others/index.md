---
title: >-
  ICML2025 其他方向92篇论文解读
description: >-
  92篇ICML2025的其他方向论文解读，涵盖对抗鲁棒、推理、Agent、少样本学习、域适应、生物分子等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "其他"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
  - "推理"
  - "Agent"
  - "少样本学习"
  - "域适应"
  - "生物分子"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📂 其他

**🧪 ICML2025** · **92** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (27)](../../ICML2026/others/index.md) · [📷 CVPR2026 (44)](../../CVPR2026/others/index.md) · [🔬 ICLR2026 (74)](../../ICLR2026/others/index.md) · [🤖 AAAI2026 (125)](../../AAAI2026/others/index.md) · [🧠 NeurIPS2025 (147)](../../NeurIPS2025/others/index.md) · [📹 ICCV2025 (48)](../../ICCV2025/others/index.md)

🔥 **高频主题：** 对抗鲁棒 ×6

**[Access Controls Will Solve the Dual-Use Dilemma](access_controls_will_solve_the_dual-use_dilemma.md)**

:   提出基于访问控制的概念框架来解决AI安全中的双用途困境（dual-use dilemma），通过用户身份验证获取真实世界上下文，结合内容分类实现细粒度的权限管理，同时缓解过度拒绝（over-refusal）和不足拒绝（under-refusal）问题。

**[Adversarial Combinatorial Semi-bandits with Graph Feedback](adversarial_combinatorial_semi-bandits_with_graph_feedback.md)**

:   本文将图反馈（graph feedback）引入对抗组合半臂赌博机（combinatorial semi-bandits）框架，提出 OSMD-G 算法，建立了最优遗憾（regret）界 $\widetilde{\Theta}(S\sqrt{T} + \sqrt{\alpha S T})$，其中 $S$ 是组合决策大小，$\alpha$ 是反馈图的独立数，关键技术在于利用随机化轮换舍入（randomized swap rounding）实现负相关采样。

**[AutoAL: Automated Active Learning with Differentiable Query Strategy Search](autoal_automated_active_learning_with_differentiable_query_strategy_search.md)**

:   提出首个可微的主动学习策略搜索框架 AutoAL，通过 SearchNet 和 FitNet 两个网络在双层优化框架下协同训练，自动从多个候选 AL 策略中为给定任务选出最优策略，在自然图像和医学图像数据集上一致超越所有候选策略及其他 SOTA 方法。

**[Avoiding Catastrophe in Online Learning by Asking for Help](avoiding_catastrophe_in_online_learning_by_asking_for_help.md)**

:   提出一个全新的在线学习理论框架来处理灾难性（不可逆）错误：将回报定义为避灾概率、目标函数为回报之积（总体避灾概率），引入导师求助机制和Local Generalization假设，证明不可能结果（不求助则必灾难）和可能结果（策略类可学则后悔和求助率同时趋零），将标准在线学习的子线性后悔提升为子常数后悔。

**[Beyond Entropy: Region Confidence Proxy for Wild Test-Time Adaptation](beyond_entropy_region_confidence_proxy_for_wild_test-time_adaptation.md)**

:   揭示熵最小化在野外测试时适应（WTTA）中的根本局限——局部区域内语义相似样本的预测不一致导致冲突优化动态，提出 ReCAP 框架用概率区域建模和有限到无穷渐近近似将不可处理的区域置信度转化为高效可优化的代理目标，在 ImageNet-C 上一致超越 SOTA。

**[Bipartite Ranking From Multiple Labels: On Loss Versus Label Aggregation](bipartite_ranking_from_multiple_labels_on_loss_versus_label_aggregation.md)**

:   本文从理论上分析了多标签二部排序（bipartite ranking）中两种聚合策略——损失聚合（loss aggregation）与标签聚合（label aggregation）——的Bayes最优解，揭示了损失聚合会产生"标签独裁"（label dictatorship）现象（某一标签因边际偏斜度而主导排序），而标签聚合能更均衡地对待所有标签。

**[Constrained Hamiltonian Systems on Observation-Induced Fiber Bundles: Theory of Symmetry and Integrability](constrained_hamiltonian_systems_on_observation-induced_fiber_bundles_theory_of_s.md)**

:   提出"观测诱导纤维丛"几何框架，将部分可观测系统中的观测不确定性从外部扰动内化为纤维坐标的内禀变化，在此结构上统一处理状态约束与观测约束，建立了完整的辛几何、可积性、对称性与守恒律理论。

**[Continuous-Time Analysis of Heavy Ball Momentum in Min-Max Games](continuous-time_analysis_of_heavy_ball_momentum_in_min-max_games.md)**

:   通过连续时间ODE建模，系统揭示了Heavy Ball动量在min-max博弈中与极小化问题截然不同的行为：**更小的动量**（包括负动量）能扩大收敛步长范围并引导轨迹走向更浅梯度区域，而**交替更新**比同步更新收敛更快且放大了这一正则化效应。

**[Curvature Enhanced Data Augmentation for Regression](curvature_enhanced_data_augmentation_for_regression.md)**

:   提出 CEMS（Curvature-Enhanced Manifold Sampling），利用数据流形的二阶近似（曲率信息）生成合成样本，用于回归任务的数据增强，在分布内和分布外场景均取得 SOTA 或接近 SOTA 的性能。

**[DRO-BAS: Decision Making under the Exponential Family DRO with Bayesian Ambiguity Sets](decision_making_under_the_exponential_family_distributionally_robust_optimisatio.md)**

:   提出 DRO-BAS 框架，利用贝叶斯后验信念构建两种后验知情的不确定集（BASPP 和 BASPE），在指数族共轭模型下可化为高效单阶段随机规划，在 Newsvendor 和 Portfolio 问题上 Pareto 支配现有 Bayesian DRO 方法。

**[Democratic AI is Possible. The Democracy Levels Framework Shows How It Might Work](democratic_ai_is_possible_the_democracy_levels_framework_shows_how_it_might_work.md)**

:   提出"Democracy Levels"（民主等级）框架，将 AI 决策权从单方面权威向民主系统的转移划分为 L0–L5 六个等级，并配套维度评估体系和实操工具，为 AI 治理的民主化提供了系统性路线图。

**[Discrepancy Minimization in Input-Sparsity Time](discrepancy_minimization_in_input-sparsity_time.md)**

:   提出首个实值矩阵差异最小化的输入稀疏时间算法，组合版 $\widetilde{O}(\mathrm{nnz}(A)+n^3)$、快速矩阵乘法版 $\widetilde{O}(\mathrm{nnz}(A)+n^{2.53})$，逼近 herdisc 的对数因子保证不变，几乎弥合了实值矩阵与二值矩阵之间的计算鸿沟。

**[Discrete Neural Algorithmic Reasoning](discrete_neural_algorithmic_reasoning.md)**

:   提出离散神经算法推理器(DNAR)，通过特征离散化、硬注意力和连续/离散数据流分离三大组件，迫使神经网络沿有限预定义状态执行算法轨迹，在 BFS/DFS/Dijkstra/Prim/MIS 等任务上实现**100%完美测试得分**，并可形式化证明所学算法的正确性。

**[Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift](diverse_prototypical_ensembles_improve_robustness_to_subpopulation_shift.md)**

:   提出 Diversified Prototypical Ensemble (DPE)，用多个多样化的原型分类器替换标准线性分类头，通过显式（inter-prototype similarity loss）和隐式（bootstrap 采样）两种多样化策略，在不需要子群标注的情况下自适应发现子群决策边界，显著提升 worst-group accuracy。

**[Diversity By Design: Leveraging Distribution Matching for Offline Model-Based Optimization](diversity_by_design_leveraging_distribution_matching_for_offline_model-based_opt.md)**

:   提出 DynAMO，通过将设计多样性显式建模为分布匹配问题，在离线模型基础优化（MBO）中同时发现高质量和高多样性的候选设计方案。

**[DSP: Dynamic Sequence Parallelism for Multi-Dimensional Transformers](dsp_dynamic_sequence_parallelism_for_multi-dimensional_transformers.md)**

:   针对多维 Transformer（如视频生成中的时空注意力模型）中现有序列并行方法只能沿单一维度分片导致大量冗余通信的问题，提出 Dynamic Sequence Parallelism (DSP)，通过在计算阶段之间动态切换并行维度（而非在模块内部通信），利用高效 all-to-all 操作实现 resharding，端到端吞吐提升 32.2%~10×，通信量减少至少 50%。

**[Efficient Network Automatic Relevance Determination](efficient_network_automatic_relevance_determination.md)**

:   将自动相关性确定（ARD）从单输出扩展到多输出回归场景，提出 NARD 框架联合估计稀疏回归系数和输出精度矩阵，并设计 Sequential/Surrogate/Hybrid 三种加速算法将复杂度从 $\mathcal{O}(d^3)$ 降至 $\mathcal{O}(p^2)$。

**[Efficient Optimization with Orthogonality Constraint: a Randomized Riemannian Submanifold Method](efficient_optimization_with_orthogonality_constraint_a_randomized_riemannian_sub.md)**

:   提出随机黎曼子流形下降方法 (RSDM)，通过将每步更新限制在随机低维子流形上，将正交约束优化中 retraction 操作的复杂度从 $O(np^2)$ 降至 $O(r^3)$，同时保持与全空间黎曼梯度下降相匹配的总计算复杂度。

**[Enhancing Certified Robustness via Block Reflector Orthogonal Layers and Logit Annealing Loss](enhancing_certified_robustness_via_block_reflector_orthogonal_layers_and_logit_a.md)**

:   本文提出了一种高效的低秩正交层参数化方法（BRO Layer）和一种退火机制的损失函数（Logit Annealing Loss），用于构建具有更强认证鲁棒性的 Lipschitz 神经网络 BRONet，在 CIFAR-10/100、Tiny-ImageNet 和 ImageNet 上达到 SOTA。

**[Exploiting Similarity for Computation and Communication-Efficient Decentralized Optimization](exploiting_similarity_for_computation_and_communication-efficient_decentralized_.md)**

:   提出 Stabilized Proximal Decentralized Optimization (SPDO) 方法及其加速版本，在近端去中心化优化框架下同时实现最优的通信和计算复杂度——通过稳定化投影技术放松子问题精度要求（从随迭代递增变为恒定），并用平均函数相似性 $\delta$ 替代最大相似性 $\delta_{\max}$ 来降低通信开销。

**[Feature Learning beyond the Lazy-Rich Dichotomy: Insights from Representational Geometry](feature_learning_beyond_the_lazy-rich_dichotomy_insights_from_representational_g.md)**

:   提出用**流形容量 (manifold capacity)** 及其关联的几何度量 (GLUE) 来刻画特征学习的丰富程度，超越传统的 lazy vs rich 二分法，揭示了不同学习阶段、学习策略以及在神经科学和 OOD 泛化问题中的新洞察。

**[Fishers for Free? Approximating the Fisher Information Matrix by Recycling the Squared Gradient Accumulator](fishers_for_free_approximating_the_fisher_information_matrix_by_recycling_the_sq.md)**

:   本文系统分析了 Adam 优化器的平方梯度累积器（Squisher）与 Fisher 信息矩阵对角线之间的理论联系，证明 Squisher 可以作为 Fisher 对角线的免费近似，在模型合并、持续学习、稀疏化等五大应用中表现与 Fisher 相当。

**[Fixed-Confidence Multiple Change Point Identification under Bandit Feedback](fixed-confidence_multiple_change_point_identification_under_bandit_feedback.md)**

:   提出了固定置信度下分段常数 bandit 中多变点识别问题，给出实例相关的采样复杂度下界，并设计了简单高效且渐近最优的 MCPI（Multiple Change Point Identification）算法。

**[Fixing the Loose Brake: Exponential-Tailed Stopping Time in Best Arm Identification](fixing_the_loose_brake_exponential-tailed_stopping_time_in_best_arm_identificati.md)**

:   揭示了经典固定置信度最佳臂识别算法（Successive Elimination、KL-LUCB）存在永不停止的正概率事件，并提出 FC-DSH 和元算法 BrakeBooster 两种方案，首次实现了停止时间的指数尾衰减保证，且不损失实例依赖复杂度（仅差对数因子）。

**[Fully Dynamic Euclidean Bi-Chromatic Matching in Sublinear Update Time](fully_dynamic_euclidean_bi-chromatic_matching_in_sublinear_update_time.md)**

:   本文首次提出了欧氏双色匹配问题的全动态亚线性更新算法，对于任意固定 $\varepsilon > 0$，实现 $O(1/\varepsilon)$ 近似比和 $O(n^{\varepsilon})$ 更新时间，可用于高效监控分布漂移（Wasserstein距离）。

**[General Agents Contain World Models](general_agents_contain_world_models.md)**

:   本文从理论上证明：任何能在多步目标导向任务上泛化的智能体，必然隐式学到了一个其环境的预测模型（世界模型），且该模型可以从智能体的策略中提取出来——智能体越强、目标越复杂，其隐含的世界模型越准确。

**[Generation from Noisy Examples](generation_from_noisy_examples.md)**

:   将 Kleinberg & Mullainathan (2024) 的"极限语言生成"理论框架扩展至噪声样本流场景，提出 Noisy Closure 维度，完整刻画了均匀噪声依赖可生成性的充要条件，并证明所有可数假设类在有限噪声下仍可非均匀生成。

**[GPU-friendly and Linearly Convergent First-order Methods for Certifying Optimal $k$-sparse GLMs](gpu-friendly_and_linearly_convergent_first-order_methods_for_certifying_optimal_.md)**

:   提出GPU友好的线性收敛一阶方法，通过复合重构+对偶间隙重启策略，将透视松弛求解加速1-2个数量级，实现大规模稀疏GLM的最优性认证。

**[Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update](heavy-tailed_linear_bandits_huber_regression_with_one-pass_update.md)**

:   提出基于 Online Mirror Descent 的单遍 Huber 回归算法 Hvt-UCB，用于重尾噪声线性 bandit，将每轮计算复杂度从 $\mathcal{O}(t\log T)$ 降至 $\mathcal{O}(1)$，同时保持最优且依赖实例的 regret 界。

**[Hierarchical Refinement: Optimal Transport to Infinity and Beyond](hierarchical_refinement_optimal_transport_to_infinity_and_beyond.md)**

:   提出 Hierarchical Refinement (HiRef) 算法，通过递归求解低秩最优传输子问题来动态构建多尺度数据分区，以对数线性时间和线性空间复杂度获得完整的双射 Monge 映射，将最优传输扩展到百万级数据集。

**[How Do Transformers Learn Variable Binding in Symbolic Programs?](how_do_transformers_learn_variable_binding_in_symbolic_programs.md)**

:   通过训练Transformer在合成程序上做变量解引用(dereference)，揭示了三阶段发展轨迹：(1)随机预测→(2)浅层启发式→(3)系统性解引用机制，因果干预证明模型学会将残差流用作可寻址内存空间。

**[If Open Source Is to Win, It Must Go Public](if_open_source_is_to_win_it_must_go_public.md)**

:   本文论证了开源 AI 在当前实践下无法独立实现 AI 民主化——模型权重只是"惰性代码"，需要大量资本才能激活——必须嵌入公共 AI 基础设施（公共资金 + 公共访问 + 公共治理 + 私人承诺）才能成为真正的公共产品。

**[Improved Exploration in GFlowNets via Enhanced Epistemic Neural Networks](improved_exploration_in_gflownets_via_enhanced_epistemic_neural_networks.md)**

:   将 Epistemic Neural Networks (ENN/epinet) 集成到 GFlowNets 中实现不确定性驱动的探索，提出 ENN-GFN-Enhanced 算法，在 HyperGrid 和序列生成任务上显著改善模式发现效率和分布学习质量。

**[Improved Generalization Bounds for Transductive Learning by Transductive Local Complexity and Its Applications](improved_generalization_bounds_for_transductive_learning_by_transductive_local_c.md)**

:   提出转导局部复杂度（TLC）框架，将经典的局部 Rademacher 复杂度扩展到转导学习设定，获得了与归纳学习几乎一致的超额风险界（仅差对数因子），并解决了十年未决的开放问题。

**[Improved Learning via k-DTW: A Novel Dissimilarity Measure for Curves](improved_learning_via_k-dtw_a_novel_dissimilarity_measure_for_curves.md)**

:   提出 $k$-DTW——一种对多边形曲线的新型不相似度量，仅关注遍历中**最大的 $k$ 个距离之和**，兼具 DTW 的鲁棒性与 Fréchet 距离的度量性质，并首次证明了曲线聚类的**无维度依赖**学习界。

**[K²IE: Kernel Method-based Kernel Intensity Estimators for Inhomogeneous Poisson Processes](k2ie_kernel_method-based_kernel_intensity_estimators_for_inhomogeneous_poisson_p.md)**

:   提出 K²IE——基于 RKHS 最小二乘正则化的核强度估计器，证明其 representer theorem 的对偶系数恒为 1，从而将经典核强度估计 (KIE) 与现代核方法在理论上统一，同时兼顾 KIE 的高效性与核方法的边缘校正优势。

**[LapSum -- One Method to Differentiate Them All: Ranking, Sorting and Top-k Selection](lapsum_--_one_method_to_differentiate_them_all_ranking_sorting_and_top-k_selecti.md)**

:   提出 LapSum，基于 Laplace 分布累积密度函数之和的闭式可逆公式，统一解决可微 ranking、sorting、top-k 选择和置换矩阵四大排序问题，时间复杂度仅 $O(n\log n)$、空间 $O(n)$，在大规模场景下显著优于现有方法。

**[Latent Variable Estimation in Bayesian Black-Litterman Models](latent_variable_estimation_in_bayesian_black-litterman_models.md)**

:   将经典 Black-Litterman 组合优化模型中的主观投资者观点 $(q, \Omega)$ 视为隐变量，通过贝叶斯网络从市场特征数据中自动推断，消除对人工主观输入的依赖，在 30 年道琼斯和 20 年 ETF 数据上 Sharpe 比率提升约 50%、换手率降低约 55%。

**[Learning-Augmented Algorithms for MTS with Bandit Access to Multiple Predictors](learning-augmented_algorithms_for_mts_with_bandit_access_to_multiple_predictors.md)**

:   在度量任务系统(MTS)中，当算法仅能以 bandit 方式（每步只查询一个启发式且需连续查询 $m$ 步才能观测状态）访问 $\ell$ 个启发式时，本文给出了 regret 为 $O(\text{OPT}^{2/3})$ 的算法，并证明该界是紧的。

**[Learning-Augmented Hierarchical Clustering](learning-augmented_hierarchical_clustering.md)**

:   本文研究借助分裂预言机（splitting oracle）的辅助信息来突破层次聚类的近似硬度障碍，获得 Dasgupta 目标的 $O(1)$ 常数近似和 Moseley-Wang 目标的 $(1-o(1))$ 近似，并推广到流式和并行计算场景。

**[Learning Distances from Data with Normalizing Flows and Score Matching](learning_distances_from_data_with_normalizing_flows_and_score_matching.md)**

:   本文提出利用 normalizing flows 和 score matching 学习密度函数与得分函数，从而高效计算基于密度的 Fermat 距离，解决了传统图方法在高维空间中收敛慢、路径粗糙的问题。

**[Lightspeed Geometric Dataset Distance via Sliced Optimal Transport](lightspeed_geometric_dataset_distance_via_sliced_optimal_transport.md)**

:   提出 s-OTDD（sliced optimal transport dataset distance），通过 Moment Transform Projection（MTP）将标签分布映射为标量，实现近线性复杂度的数据集距离计算，速度远超 OTDD 且性能相当。

**[Maximum Coverage in Turnstile Streams with Applications to Fingerprinting Measures](maximum_coverage_in_turnstile_streams_with_applications_to_fingerprinting_measur.md)**

:   首次在 turnstile 流模型（支持任意插入/删除）下给出最大覆盖问题的单遍流算法，空间 $\tilde{O}(d/\varepsilon^3)$、更新时间 $\tilde{O}(1)$，并将其推广到隐私指纹识别（fingerprinting）场景，实验比先前方法快 210×。

**[Modified K-means Algorithm with Local Optimality Guarantees](modified_k-means_algorithm_with_local_optimality_guarantees.md)**

:   首次指出经典K-means算法并不总是收敛到局部最优解这一长期误解，并提出LO-K-means修改方案，在不增加单步计算复杂度的前提下保证收敛到连续或离散意义下的局部最优解。

**[Multiple-Policy Evaluation via Density Estimation](multiple-policy_evaluation_via_density_estimation.md)**

:   提出 CAESAR 算法，通过两阶段方法（粗估计访问分布 + 最优采样分布下的密度比估计）同时评估 K 个策略，实现非渐近、实例依赖的样本复杂度，核心技术是"粗估计"——仅需 $O(1/\epsilon)$ 样本即可获得常数倍精度的分布近似。

**[Near-Optimal Consistency-Robustness Trade-Offs for Learning-Augmented Online Knapsack Problems](near-optimal_consistency-robustness_trade-offs_for_learning-augmented_online_kna.md)**

:   提出一族基于简洁预测（临界值的点预测或区间预测）的在线背包算法，在consistency与robustness之间实现近Pareto最优的权衡，并给出分数解到整数解的通用转换方法。

**[Near Optimal Best Arm Identification for Clustered Bandits](near_optimal_best_arm_identification_for_clustered_bandits.md)**

:   在多智能体聚类多臂赌博机设置下，提出 Cl-BAI 和 BAI-Cl 两种算法，利用聚类结构大幅降低最优臂识别的样本复杂度，并证明 BAI-Cl++ 在 $M$ 为常数时达到 minimax 最优。

**[NeuronTune: Towards Self-Guided Spurious Bias Mitigation](neurontune_towards_self-guided_spurious_bias_mitigation.md)**

:   NeuronTune 提出一种**无需组标签**的自引导去偏方法：通过对比模型隐空间中正确/错误预测样本的神经元激活差异，识别受虚假偏差影响的维度并将其置零，再重训最后一层分类器，从而显著提升 worst-group accuracy。

**[Nonparametric Modern Hopfield Models](nonparametric_modern_hopfield_models.md)**

:   本文提出现代 Hopfield 模型的非参数框架，将记忆存储与检索过程建模为非参数回归问题，由此推导出首个具有亚二次复杂度的高效稀疏结构现代 Hopfield 模型，并提供了完备的理论分析（检索误差界、噪声鲁棒性、指数记忆容量）。

**[On Fine-Grained Distinct Element Estimation](on_fine-grained_distinct_element_estimation.md)**

:   提出以**成对碰撞数** $C$（pairwise collisions）作为分布式去重计数问题的细粒度复杂度参数，设计了通信量随 $C$ 减小而显著降低的协议，打破了此前 $\Omega(\alpha/\varepsilon^2)$ 的最坏情况下界，并给出了所有参数区间的匹配下界。

**[On the Importance of Gaussianizing Representations](on_the_importance_of_gaussianizing_representations.md)**

:   基于信息论动机（正态分布同时是最优信号与最差噪声分布），提出 Normality Normalization 层：在常规归一化之后用 Power Transform 高斯化激活值，并注入缩放高斯噪声进行正则化，在 ViT/ResNet 上普遍提升泛化与鲁棒性，且不引入额外可学习参数。

**[Online Sparsification of Bipartite-Like Clusters in Graphs](online_sparsification_of_bipartite-like_clusters_in_graphs.md)**

:   提出了一种**近线性时间的在线图稀疏化算法**，能在保留图的二部图式聚类（bipartite-like clusters）结构的前提下，将边数压缩到 $\widetilde{O}(n)$，同时适用于无向图和有向图，显著加速现有聚类算法。

**[OOD-Chameleon: Is Algorithm Selection for OOD Generalization Learnable?](ood-chameleon_is_algorithm_selection_for_ood_generalization_learnable.md)**

:   将 OOD 泛化的训练算法选择形式化为可学习的多标签分类问题，在"数据集的数据集"上训练选择器，仅凭数据集统计特征（偏移程度、数据规模等）即可先验地预测最佳训练算法（ERM / GroupDRO / 重采样 / Logits 调整），在合成、视觉、语言 7 个应用上验证了选择器学到了可迁移的非平凡决策规则。

**[Optimal Auction Design in the Joint Advertising](optimal_auction_design_in_the_joint_advertising.md)**

:   本文针对联合广告场景（零售商与供应商共同竞标广告位）提出最优拍卖机制：单槽位下给出Myerson式闭式最优解，多槽位下设计BundleNet神经网络以bundle为单位构建IC约束，在保证近似激励兼容的同时最大化平台收入。

**[Optimal Sensor Scheduling and Selection for Continuous-Discrete Kalman Filtering with Auxiliary Dynamics](optimal_sensor_scheduling_and_selection_for_continuous-discrete_kalman_filtering.md)**

:   提出一种面向连续-离散卡尔曼滤波 (CD-KF) 的最优传感器调度框架：将多传感器观测建模为独立 Poisson 过程，推导后验协方差矩阵的可微上界，利用梯度优化方法联合优化观测频率与辅助动力学输入，并通过 Wasserstein-2 最优量化确定性地选取观测时刻。

**[PAC Learning with Improvements](pac_learning_with_improvements.md)**

:   提出"带改进的 PAC 学习"框架：当 agent 能真正提升自身特征至多 $r$ 时，保守分类器可实现零误差（将标准 PAC 中不可能的目标变为可能），有限 VC 维既非充分也非必要条件，改进学习与标准 PAC 和策略性分类存在本质分离。

**[Permutation Equivariant Neural Networks for Symmetric Tensors](permutation_equivariant_neural_networks_for_symmetric_tensors.md)**

:   本文首次研究了以对称张量为输入的置换等变神经网络，给出了对称幂空间之间所有线性置换等变函数的两种完整刻画，实验证明该方法在数据效率和泛化能力上显著优于标准 MLP。

**[Position: Solve Layerwise Linear Models First to Understand Neural Dynamical Phenomena](position_solve_layerwise_linear_models_first_to_understand_neural_dynamical_phen.md)**

:   提出**动态反馈原则 (Dynamical Feedback Principle)**，论证逐层线性模型（layerwise linear models）足以统一解释 neural collapse、emergence、lazy/rich regime 和 grokking 四大深度学习动力学现象，呼吁优先研究逐层结构而非非线性激活。

**[Positional Attention: Expressivity and Learnability of Algorithmic Computation](positional_attention_expressivity_and_learnability_of_algorithmic_computation.md)**

:   提出 **Positional Transformer**——注意力权重仅由位置编码决定、与输入数据无关的 Transformer 变体，证明其保持了与 MPC 并行计算模型等价的表达力（仅增加 $O(\log n)$ 深度代价），并在算法任务上展现出显著更优的分布外泛化能力。

**[Practical Principles for AI Cost and Compute Accounting](practical_principles_for_ai_cost_and_compute_accounting.md)**

:   针对 AI 监管中计算量/成本阈值的核算标准模糊问题，提出七项原则来封堵蒸馏漏洞等规避手段、避免抑制安全措施、并实现跨企业一致实施，为 EU AI Act 等法规的落地提供理论框架。

**[Prediction-Powered Adaptive Shrinkage Estimation](prediction-powered_adaptive_shrinkage_estimation.md)**

:   将Prediction-Powered Inference (PPI)与经验贝叶斯收缩有机结合，提出PAS两阶段估计方法——先在每个问题内利用ML预测做方差缩减，再跨问题利用ML预测作为收缩目标做自适应收缩，通过CURE无偏风险估计自动调优收缩参数，理论证明渐近最优。

**[Prediction via Shapley Value Regression (ViaSHAP)](prediction_via_shapley_value_regression.md)**

:   提出 ViaSHAP，将 Shapley 值的计算融入模型训练过程，使得推理时通过对 Shapley 值求和直接得到预测，无需后验解释器，在表格数据上达到 XGBoost 级别的预测精度，同时 Shapley 值近似质量显著优于 FastSHAP。

**[Principled Algorithms for Optimizing Generalized Metrics in Binary Classification](principled_algorithms_for_optimizing_generalized_metrics_in_binary_classificatio.md)**

:   本文提出了优化广义分类指标（如 $F_\beta$、Jaccard、加权准确率等）的有原则算法 METRO，基于 $H$-一致性界和代理损失理论，将指标优化重新表述为广义代价敏感学习问题，具有有限样本泛化保证。

**[Probably Approximately Global Robustness Certification](probably_approximately_global_robustness_certification.md)**

:   提出基于 ε-net 采样的概率近似全局鲁棒性（PAG）认证框架，所需样本量与输入维度、类别数和模型架构无关，可高效认证大规模神经网络的全局鲁棒性。

**[Provably Efficient Algorithm for Best Scoring Rule Identification in Online Principal-Agent Information Acquisition](provably_efficient_algorithm_for_best_scoring_rule_identification_in_online_prin.md)**

:   本文在委托-代理（principal-agent）在线信息获取框架下研究最佳评分规则识别（Best Scoring Rule Identification, BSRI）问题，提出 OIAFC（固定置信度）和 OIAFB（固定预算）两种算法，首次建立了实例依赖的样本复杂度上界 $\widetilde{O}(MH_\Delta)$，并将实例无关的样本复杂度从已有工作的 $\widetilde{O}(C_O^3 K^6 \epsilon^{-3})$ 大幅改进至 $\widetilde{O}(MK\epsilon^{-2})$。

**[Provably Improving Generalization of Few-Shot Models with Synthetic Data](provably_improving_generalization_of_few-shot_models_with_synthetic_data.md)**

:   提出一个理论框架量化合成数据与真实数据的分布差异对少样本分类泛化能力的影响，并基于该理论设计了联合优化数据划分与模型训练的算法，在10个基准数据集上超越SOTA。

**[Randomized Dimensionality Reduction for Euclidean Maximization and Diversity Measures](randomized_dimensionality_reduction_for_euclidean_maximization_and_diversity_mea.md)**

:   证明了对一大类欧氏最大化问题（最大匹配、最大TSP、最大生成树、子图多样性等），使用数据无关的高斯 JL 变换将维度降至 $O(\lambda)$（$\lambda$ 为数据集倍增维度）即可近似保持所有候选解的值，并证明该依赖是紧的。

**[Regression for the Mean: Auto-Evaluation and Inference with Few Labels through Post-hoc Regression](regression_for_the_mean_auto-evaluation_and_inference_with_few_labels_through_po.md)**

:   将 PPI++ 中调参 $\lambda$ 的过程重新解释为事后回归（post-hoc regression），提出 Ridge-PPI 和 Sigmoid-PPI 两种改进方法，在少标签（$n < 50$）场景下显著降低均值估计方差，优于经典估计和 PPI++。

**[Residual Matrix Transformers: Scaling the Size of the Residual Stream](residual_matrix_transformers_scaling_the_size_of_the_residual_stream.md)**

:   用外积记忆矩阵替换 Transformer 的残差流向量，使残差流大小可独立于模型参数量和 FLOPS 扩展，在相同 loss 下节省 58% FLOPS、25% 参数和 41% 训练 token。

**[Rethinking Aleatoric and Epistemic Uncertainty](rethinking_aleatoric_and_epistemic_uncertainty.md)**

:   本文指出机器学习中 aleatoric/epistemic 不确定性二分法存在根本性概念混淆，提出基于决策理论的替代框架，将预测不确定性、可约/不可约分解、预测性能和数据分散度统一在一个连贯的理论体系中，并揭示了 BALD 作为 epistemic uncertainty 估计器的局限性。

**[Revisiting Instance-Optimal Cluster Recovery in the Labeled Stochastic Block Model](revisiting_instance-optimal_cluster_recovery_in_the_labeled_stochastic_block_mod.md)**

:   针对标签随机块模型 (LSBM)，提出 IAC (Instance-Adaptive Clustering) 算法，通过一次谱聚类 + 迭代似然改进两阶段策略，首次以 $\mathcal{O}(n(\log n)^3)$ 复杂度实现匹配实例特定信息论下界的社区恢复，同时提供期望和高概率双重保证。

**[Revisiting the Predictability of Performative, Social Events](revisiting_the_predictability_of_performative_social_events.md)**

:   本文用现代学习理论工具（performative prediction + outcome indistinguishability）重新回答了20世纪社会科学中的经典问题：在预测会主动影响结果的情况下，社会事件是否仍可被准确预测？答案是肯定的——但这种"准确"的预测可能毫无用处。

**[Sampling from Binary Quadratic Distributions via Stochastic Localization](sampling_from_binary_quadratic_distributions_via_stochastic_localization.md)**

:   首次将随机局部化 (Stochastic Localization, SL) 框架应用于一般二元二次分布 (BQD) 采样，证明经过足够SL迭代后后验分布几乎处处满足 Poincaré 不等式，从而保证离散 MCMC 采样器多项式时间混合，并在 QUBO 组合优化问题上验证了一致的采样效率提升。

**[Scalable Equilibrium Sampling with Sequential Boltzmann Generators](scalable_equilibrium_sampling_with_sequential_boltzmann_generators.md)**

:   SBG通过Transformer架构规范化流(TarFlow)和退火Langevin动力学的序列蒙特卡洛，首次在笛卡尔坐标系中实现六肽(66原子)系统的高效平衡采样。

**[Score Matching with Missing Data](score_matching_with_missing_data.md)**

:   本文将 score matching 及其主要扩展适配到缺失数据场景，提出两种变体——重要性加权（IW）方法和变分方法，在图模型估计等任务上展示了不同场景下各自的优势。

**[Softmax is not Enough (for Sharp Size Generalisation)](softmax_is_not_enough_for_sharp_size_generalisation.md)**

:   本文从理论上证明了 softmax 注意力在输入规模增大时**必然发生系数分散（dispersion）**，无法保持对少量关键元素的尖锐聚焦，并提出自适应温度（adaptive temperature）作为缓解手段。

**[Sparse-Pivot: Dynamic Correlation Clustering for Node Insertions](sparse-pivot_dynamic_correlation_clustering_for_node_insertions.md)**

:   提出 Sparse-Pivot 算法，在节点动态插入的 Correlation Clustering 问题中以摊销 $O_\varepsilon(\log^{O(1)} n)$ 的数据库操作实现 $(20+\varepsilon)$-近似，大幅改善了 Cohen-Addad et al. (ICML 2024) 的近似因子，并在实验中全面优于基线。

**[Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry](sparse_training_from_random_initialization_aligning_lottery_ticket_masks_using_w.md)**

:   从权重对称性角度解释彩票假说(LTH)掩码不能迁移到新初始化的原因，并提出通过置换匹配对齐LTH掩码与新初始化的优化盆地来实现稀疏训练。

**[SUICA: Learning Super-high Dimensional Sparse Implicit Neural Representations for Spatial Transcriptomics](suica_learning_super-high_dimensional_sparse_implicit_neural_representations_for.md)**

:   提出 SUICA，通过图增强自编码器将超高维稀疏空间转录组数据压缩到紧凑嵌入空间，再用隐式神经表示（INR）建模坐标到嵌入的连续映射，实现跨多种 ST 平台的空间填补、基因填补和去噪。

**[Suitability Filter: A Statistical Framework for Classifier Evaluation in Real-World Settings](suitability_filter_a_statistical_framework_for_classifier_evaluation_in_real-wor.md)**

:   本文提出 Suitability Filter 框架，利用模型输出的"适用性信号"（suitability signals）在无标签的用户数据上检测分类器性能退化，通过统计假设检验判断准确率是否相比测试集显著下降。

**[Symmetry-Aware GFlowNets](symmetry-aware_gflownets.md)**

:   揭示 GFlowNets 在图生成中因等价动作（不同动作产出同构图）导致的系统性采样偏差——节点生成偏向低对称图、片段生成偏向高对称组件，提出通过终态自同构群大小缩放奖励的简单修正方法 SA-GFN，仅需一次自同构群计算即可实现无偏采样。

**[Symmetry-Robust 3D Orientation Estimation](symmetry-robust_3d_orientation_estimation.md)**

:   提出一种对旋转对称性鲁棒的两阶段3D朝向估计流水线：第一阶段通过商回归（quotient regression）将朝向恢复到八面体对称群的等价类内，第二阶段通过分类器预测24个八面体翻转之一以完成精确复原，在ShapeNet上取得SOTA。

**[SynDaCaTE: A Synthetic Dataset for Evaluating Part-Whole Hierarchical Inference](syndacate_a_synthetic_dataset_for_evaluating_part-whole_hierarchical_inference.md)**

:   提出SynDaCaTE合成数据集和Mereological Inference框架，将部分-整体层次推断分解为Image-to-Parts和Parts-to-Wholes两个可独立评估的子任务，通过精心设计的控制实验证明CapsNet的瓶颈在于从图像提取部件而非从部件推断整体，同时发现置换等变的SetTransformer在部件到整体推断中显著优于所有基线（超过10倍精度优势）。

**[TANGO: Clustering with Typicality-Aware Nonlocal Mode-Seeking and Graph-Cut Optimization](tango_clustering_with_typicality-aware_nonlocal_mode-seeking_and_graph-cut_optim.md)**

:   提出"典型性(typicality)"概念，从全局视角量化数据点作为模式(聚类中心)的置信度，结合改进的路径相似度与图割优化，实现无需人工阈值设定的自动模式检测与聚类。

**[The Price of Freedom: Exploring Expressivity and Runtime Tradeoffs in Equivariant Networks](the_price_of_freedom_exploring_expressivity_and_runtime_tradeoffs_in_equivariant.md)**

:   本文系统分析了 $E(3)$-等变神经网络中多种张量积操作的表达力与运行时间权衡，发现理论复杂度与实际性能差距悬殊，并提出基于球面网格的简化 Gaunt 张量积实现，在 MACE 原子间势能训练中加速 30%。

**[Theoretical Performance Guarantees for Partial Domain Adaptation via Partial Optimal Transport](theoretical_performance_guarantees_for_partial_domain_adaptation_via_partial_opt.md)**

:   本文基于部分最优传输理论推导了部分领域自适应（PDA）的泛化界，证明了部分 Wasserstein 距离作为领域对齐项和提出的理论驱动权重方案的合理性，并据此开发了实用算法 WARMPOT。

**[Time-Aware World Model for Adaptive Prediction and Control](time-aware_world_model_for_adaptive_prediction_and_control.md)**

:   提出时间感知世界模型 TAWM，通过将时间步长 $\Delta t$ 作为显式输入条件并在训练中混合多种 $\Delta t$ 采样，使模型能以单步预测适应任意时间分辨率的推理，且不增加训练样本量。

**[To Each Metric Its Decoding: Post-Hoc Optimal Decision Rules of Probabilistic Hierarchical Classifiers](to_each_metric_its_decoding_post-hoc_optimal_decision_rules_of_probabilistic_hie.md)**

:   本文提出了针对概率层次分类器的后处理最优解码框架，为不同评价指标（如层次 $F_\beta$）推导了最优决策规则，在候选集限于节点集时给出通用算法，对子集预测推导了专门的层次 $hF_\beta$ 最优策略。

**[Truly Self-Improving Agents Require Intrinsic Metacognitive Learning](truly_self-improving_agents_require_intrinsic_metacognitive_learning.md)**

:   本文提出一个形式化框架论证了真正的自我改进 Agent 需要具备内在元认知学习能力（而非外在的、人为设计的固定循环），该框架包含三个组件：元认知知识、元认知规划和元认知评估，并分析了现有自改进 Agent 的不足和实现内在元认知的路径。

**[Understanding Mode Connectivity via Parameter Space Symmetry](understanding_mode_connectivity_via_parameter_space_symmetry.md)**

:   通过参数空间的连续对称性（如 $GL_h(\mathbb{R})$）分析神经网络损失函数最小值集合的拓扑连通性，推导出线性网络最小值的连通分量数为 $2^{l-1}$，并证明 skip connection 可减少该数目，同时给出对称性诱导的显式低损失连接曲线及线性模式连通性近似成立的充分条件。

**[UnHiPPO: Uncertainty-Aware Initialization for State Space Models](unhippo_uncertainty-aware_initialization_for_state_space_models.md)**

:   本文扩展了 HiPPO 理论以处理带噪声的测量数据，将 SSM 的初始化问题重新表述为线性随机控制问题，推导出不确定性感知的动力学初始化方案，在不增加运行时间的前提下显著提升 SSM 的噪声鲁棒性。

**[WGFormer: An SE(3)-Transformer Driven by Wasserstein Gradient Flows for Molecular Generation](wgformer_an_se3-transformer_driven_by_wasserstein_gradient_flows_for_molecular_g.md)**

:   本文提出 WGFormer，一种由 Wasserstein 梯度流驱动的 SE(3)-Transformer，在自编码器框架内通过最小化原子潜在混合模型上的能量函数来优化分子构象，在基态构象预测任务上一致超越 SOTA。
