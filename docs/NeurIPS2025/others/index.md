---
title: >-
  NeurIPS2025 其他方向 135篇论文解读
description: >-
  135篇NeurIPS2025 其他方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📂 其他

**🧠 NeurIPS2025** · 共 **135** 篇

**[4Dgt Learning A 4D Gaussian Transformer Using Realworld Mono](4dgt_learning_a_4d_gaussian_transformer_using_realworld_mono.md)**

:   提出4DGT——一种基于4D高斯的Transformer模型，完全在真实世界单目带位姿视频上训练，以前馈方式在几秒内完成动态场景重建，显著优于同类前馈网络，并达到与优化类方法可比的精度。

**[A Differentiable Model Of Supply-Chain Shocks](a_differentiable_model_of_supply-chain_shocks.md)**

:   用 JAX 实现可微分的供应链 Agent-Based Model（~1000 家企业），通过 GPU 并行化 + 自动微分实现比传统 ABC 快 3 个数量级的贝叶斯参数校准，为全球供应链网络的冲击传播建模铺平道路。

**[A Generalized Label Shift Perspective For Crossdomain Gaze E](a_generalized_label_shift_perspective_for_crossdomain_gaze_e.md)**

:   本文将跨域视线估计(CDGE)问题建模为广义标签偏移(GLS)问题，指出现有域不变表示学习方法在标签偏移存在时理论上不充分，提出基于截断高斯分布的连续重要性重加权和概率感知条件算子差异(PCOD)来联合纠正标签偏移和条件偏移，在多个backbone上平均降低误差12%~27%。

**[A Sustainable Ai Economy Needs Data Deals That Work For Gene](a_sustainable_ai_economy_needs_data_deals_that_work_for_gene.md)**

:   本文通过分析73个公开数据交易案例，揭示了ML价值链中的"经济数据处理不等式"——从原始数据到模型权重再到合成输出，每一步都提炼了技术信号但剥夺了数据生成者的经济权益，并提出EDVEX框架来构建更公平的数据交换市场。

**[A Theoretical Framework For Grokking Interpolation Followed By Riemannian Norm M](a_theoretical_framework_for_grokking_interpolation_followed_by_riemannian_norm_m.md)**

:   本文从纯优化角度严格证明了 grokking 现象的成因：带小 weight decay 的梯度流在 $\lambda\to 0$ 极限下呈现两阶段动力学——先快速收敛到训练损失的临界流形 $\mathcal{M}$，再在 $t\approx 1/\lambda$ 时沿流形做黎曼梯度流以最小化 $\ell_2$ 范数，从而延迟实现泛化。

**[A Unified Framework For Variable Selection In Modelbased Clu](a_unified_framework_for_variable_selection_in_modelbased_clu.md)**

:   提出了一个统一框架（SelvarMNARz），在高斯混合模型聚类中同时完成变量选择和MNAR（Missing Not At Random）缺失数据建模，通过两阶段策略（LASSO排序 + BIC角色分配）实现高维场景下的高效推理，并给出了可辨识性和选择一致性的理论保证。

**[Active Measurement Efficient Estimation At Scale](active_measurement_efficient_estimation_at_scale.md)**

:   提出Active Measurement框架，结合AI检测器的自适应重要性采样和迭代人工标注，实现大规模科学测量（如鸟类计数、疟疾检测）的无偏估计，将原始检测器3.78的误差率降至0.06，同时提供理论保证的置信区间。

**[Acurank Uncertainty-Aware Adaptive Computation For Listwise Reranking](acurank_uncertainty-aware_adaptive_computation_for_listwise_reranking.md)**

:   通过基于TrueSkill模型的不确定性估计，动态调整重排序子集大小和验证范围，在实现更优精度效率权衡的同时避免过度计算。

**[Adaptive Data Analysis For Growing Data](adaptive_data_analysis_for_growing_data.md)**

:   首次为动态/增长数据场景下的自适应数据分析提供泛化界，允许分析者根据当前数据规模和历史查询结果自适应地调度统计查询，在数据不断积累时获得更紧的准确性保证。

**[Addressing Mark Imbalance In Integrationfree Neural Marked T](addressing_mark_imbalance_in_integrationfree_neural_marked_t.md)**

:   论文针对现实事件流中常见的 mark 类别长尾失衡问题，提出基于先验归一化概率的阈值学习策略，并设计 integration-free 的神经 MTPP 架构，先预测 mark 再预测 time，在避免昂贵数值积分的同时显著提升稀有事件的 mark 与到达时间预测性能。

**[Adjoint Schrödinger Bridge Sampler](adjoint_schrödinger_bridge_sampler.md)**

:   提出 Adjoint Schrödinger Bridge Sampler (ASBS)，通过将 Schrödinger Bridge 问题重新解释为随机最优控制问题，消除了先前扩散采样器的 memoryless 条件限制，支持任意源分布（如高斯、谐波先验），使用可扩展的 matching 目标无需重要性权重估计，在多粒子能量函数和分子构象生成上全面超越先前方法。

**[Adjusted Count Quantification Learning On Graphs](adjusted_count_quantification_learning_on_graphs.md)**

:   将经典的 Adjusted Classify & Count (ACC) 量化方法扩展到图结构数据，提出结构重要性采样（SIS）和邻域感知ACC两种技术，分别解决图量化中的结构协变量偏移和非同质性边问题。

**[Adpretrain Advancing Industrial Anomaly Detection Via Anomaly Representation Pre](adpretrain_advancing_industrial_anomaly_detection_via_anomaly_representation_pre.md)**

:   首次提出面向工业异常检测的专用表示预训练框架 ADPretrain，通过角度和范数导向的对比损失在大规模异常检测数据集 RealIAD 上学习残差特征表示，替换五种主流嵌入式 AD 方法的原始特征后在五个数据集、五个骨干网络上取得一致性提升。

**[Alias-Free Vit Fractional Shift Invariance Via Linear Attention](alias-free_vit_fractional_shift_invariance_via_linear_attention.md)**

:   提出Alias-Free ViT，通过两个关键组件实现Vision Transformer对整数和亚像素平移的鲁棒性：(1) 抗混叠下采样和非线性层设计，(2) 基于交叉协方差的线性注意力（shift-equivariant），在图像分类中保持竞争力的同时显著提升对抗性平移鲁棒性。

**[An Empirical Investigation Of Neural Odes And Symbolic Regression For Dynamical ](an_empirical_investigation_of_neural_odes_and_symbolic_regression_for_dynamical_.md)**

:   本文系统研究了 Neural ODE (NODE) 在含噪合成数据上的外推能力，并探索了将 NODE 作为数据增强工具、与符号回归 (SR) 结合以从有限数据中恢复动力学方程的流水线，结果表明该组合方案能从仅 10% 的仿真数据中恢复三个控制方程中的两个及第三个的良好近似。

**[EPHAD: An Evidence-Based Post-Hoc Adjustment Framework for Anomaly Detection Under Data Contamination](an_evidence-based_post-hoc_adjustment_framework_for_anomaly_detection_under_data.md)**

:   EPHAD 提出一种测试时后处理框架来修正在被污染数据上训练的异常检测模型——在不接触训练流程/数据的前提下，用多模态基础模型（CLIP）或经典方法（LOF）等"证据"在测试时调整模型输出，在 8 个视觉+26 个表格 AD 数据集上有效提升性能。

**[Are Pixel-Wise Metrics Reliable For Sparse-View Computed Tomography Reconstructi](are_pixel-wise_metrics_reliable_for_sparse-view_computed_tomography_reconstructi.md)**

:   揭示 PSNR/SSIM 等像素级指标无法反映稀疏视图 CT 重建中解剖结构完整性（相关性仅 0.16-0.30），提出基于自动分割的解剖感知指标（NSD/clDice）和 CARE 框架——在扩散模型训练中加入分割引导损失，大器官结构完整性提升 32%、血管提升 36%。

**[Autoscidact Automated Scientific Discovery Through Contrastive Embedding And Hyp](autoscidact_automated_scientific_discovery_through_contrastive_embedding_and_hyp.md)**

:   提出 AutoSciDACT 管线：先用有监督对比学习将高维科学数据压缩到 4 维嵌入空间，再用 NPLM（New Physics Learning Machine）似然比检验对嵌入空间中的分布偏差进行统计量化，在天文、粒子物理、病理、图像和合成数据集上以 ≤1% 的信号注入比例实现 ≥3σ 发现。

**[Brain-Like Processing Pathways Form In Models With Heterogeneous Experts](brain-like_processing_pathways_form_in_models_with_heterogeneous_experts.md)**

:   在异构 Mixture-of-Experts 模型中，异构专家并不会自动形成处理通路；本文提出三个受大脑启发的归纳偏置（路由代价、任务表现缩放、专家 Dropout），使模型形成类似大脑"皮层-皮层下"动态通路的 Mixture-of-Pathways 架构。

**[Computable Universal Online Learning](computable_universal_online_learning.md)**

:   在 universal online learning 框架中引入可计算性约束，证明了"数学上可学习"不等于"可用计算机程序实现的可学习"，并给出了 agnostic 和 proper 变体下可计算学习的精确刻画。

**[Contextual Dynamic Pricing With Heterogeneous Buyers](contextual_dynamic_pricing_with_heterogeneous_buyers.md)**

:   首次系统研究买家类型异质（$K_\star$ 种未知类型）的上下文动态定价问题，提出基于乐观后验采样 (OPS) 的算法实现 $\tilde{O}(K_\star\sqrt{dT})$ 遗憾界（对 $d$ 和 $T$ 最优），并在非上下文情形通过方差感知自适应离散化算法 ZoomV 实现 $\tilde{O}(\sqrt{K_\star T})$ 最优遗憾。

**[Continuous Thought Machines](continuous_thought_machines.md)**

:   提出 Continuous Thought Machine (CTM)，通过私有参数化的 Neuron-Level Models (NLMs) 产生神经元级时间动力学，并以神经同步矩阵作为核心潜在表征，在迷宫求解、ImageNet 分类、奇偶校验等任务上展现复杂推理、自适应计算和可解释注意力行为。

**[Coreset For Robust Geometric Median Eliminating Size Dependency On Outliers](coreset_for_robust_geometric_median_eliminating_size_dependency_on_outliers.md)**

:   首次消除鲁棒几何中位数 coreset 大小对异常值数 $m$ 的依赖：在 $n \geq 4m$ 条件下，$d=1$ 时实现最优 coreset 大小 $\tilde{\Theta}(\varepsilon^{-1/2} + \frac{m}{n}\varepsilon^{-1})$，高维时实现 $\tilde{O}(\varepsilon^{-2}\min\{\varepsilon^{-2}, d\})$，核心技术是新颖的**非逐分量误差分析**。

**[Coresets For Clustering Under Stochastic Noise](coresets_for_clustering_under_stochastic_noise.md)**

:   首次系统研究噪声数据下 $(k,z)$-聚类 coreset 构造问题，提出新的代理误差度量 $\mathsf{Err}_\alpha$ 替代传统 $\mathsf{Err}$，在温和数据假设下实现 coreset 大小缩减 $\text{poly}(k)$ 倍、质量保证收紧 $\text{poly}(k)$ 倍，并设计噪声感知的 cluster-wise 采样算法。

**[Deep Continuous-Time State-Space Models For Marked Event Sequences](deep_continuous-time_state-space_models_for_marked_event_sequences.md)**

:   S2P2 将线性 Hawkes 过程与深度状态空间模型结合，通过堆叠多层隐式线性 Hawkes (LLH) 层 + 非线性激活构建高表达力的连续时间 MTPP 模型，利用并行扫描实现线性复杂度和亚线性时间，在 8 个真实数据集上平均提升 33% 预测似然。

**[Deep Legendre Transform](deep_legendre_transform.md)**

:   DLT 利用凸共轭的隐式 Fenchel 表示 $f^*(\nabla f(x)) = \langle x, \nabla f(x) \rangle - f(x)$ 将凸共轭计算转化为标准回归问题，避免求解 max/min-max 优化，且能提供后验误差估计，结合 KAN 还可获得精确解析解。

**[Depth-Bounds For Neural Networks Via The Braid Arrangement](depth-bounds_for_neural_networks_via_the_braid_arrangement.md)**

:   本文证明了在 $\mathcal{B}_d^0$-conforming 约束下，ReLU 网络精确表示 $\max\{0, x_1, \ldots, x_d\}$ 需要 $\Omega(\log \log d)$ 层——这是首个不限制权重的非常数深度下界；同时证明 rank-(3,2) maxout 网络可以计算 7 个数的最大值，说明标准上界不紧。

**[Depth-Supervised Fusion Network For Seamless-Free Image Stitching](depth-supervised_fusion_network_for_seamless-free_image_stitching.md)**

:   DSFN 提出深度一致性约束的无缝图像拼接方法：通过深度感知的两阶段变换估计解决大视差对齐，软缝合区域扩散实现自然融合，结合重参数化策略提升效率，在 UDIS-D 和 IVSD 数据集上全面超越 SOTA。

**[Directional Non-Commutative Monoidal Structures For Compositional Embeddings In ](directional_non-commutative_monoidal_structures_for_compositional_embeddings_in_.md)**

:   提出一种基于方向性非交换幺半群算子的代数框架，为多维组合嵌入提供统一数学基础，将 SSM 递归、Transformer 自注意力和 RoPE 位置编码统一为特例。

**[Distributionally Robust Feature Selection](distributionally_robust_feature_selection.md)**

:   本文提出一种模型无关的分布鲁棒特征选择方法，通过向协变量注入可控高斯噪声实现离散选择的连续松弛，并优化 Bayes 最优预测器的条件方差，使选出的特征子集能在多个子群体上同时训练出高质量下游模型。

**[Double Descent Meets Out-Of-Distribution Detection Theoretical Insights And Empi](double_descent_meets_out-of-distribution_detection_theoretical_insights_and_empi.md)**

:   本文首次揭示 post-hoc OOD 检测中存在 double descent 现象——OOD 检测性能随模型宽度在插值阈值附近出现谷值后再次上升，通过随机矩阵理论提供理论解释，并提出基于 Neural Collapse 的 NC1 判据来识别最佳模型复杂度区间。

**[Dpa A One-Stop Metric To Measure Bias Amplification In Classification Datasets](dpa_a_one-stop_metric_to_measure_bias_amplification_in_classification_datasets.md)**

:   本文提出 Directional Predictability Amplification (DPA)，一种基于可预测性的偏差放大度量指标，是唯一同时满足方向性、适用于平衡/非平衡数据集、能正确识别正负偏差放大的一站式指标，通过测量模型偏差与数据集偏差的相对变化来量化偏差放大程度。

**[Efficient Kernelized Learning In Polyhedral Games Beyond Full-Information From C](efficient_kernelized_learning_in_polyhedral_games_beyond_full-information_from_c.md)**

:   提出基于核化（kernelization）的框架，在部分信息反馈设定下为多面体博弈（Colonel Blotto、图拟阵拥堵博弈、网络拥堵博弈）设计了计算高效的无遗憾学习算法，显著改进了学习粗关联均衡（CCE）的运行时复杂度。

**[Efficient Parametric Svd Of Koopman Operator For Stochastic Dynamical Systems](efficient_parametric_svd_of_koopman_operator_for_stochastic_dynamical_systems.md)**

:   提出基于 low-rank approximation (LoRA) 的目标函数来学习随机动力系统 Koopman 算子的 top-k 奇异函数，完全避免了 VAMPnet/DPNet 中数值不稳定的矩阵分解操作，且梯度天然无偏。

**[Emergency Response Measures For Catastrophic Ai Risk](emergency_response_measures_for_catastrophic_ai_risk.md)**

:   本文分析了如何将前沿安全政策（Frontier Safety Policies, FSPs）模型整合到中国四阶段应急响应框架中，以应对来自先进AI系统的灾难性风险（如大规模杀伤性武器扩散、失控事件等）。

**[Equivariance By Contrast Identifiable Equivariant Embeddings From Unlabeled Fini](equivariance_by_contrast_identifiable_equivariant_embeddings_from_unlabeled_fini.md)**

:   提出 Equivariance by Contrast (EbC)，一种仅用编码器的方法，从观测对 $(\mathbf{y}, g \cdot \mathbf{y})$ 中联合学习等变嵌入空间和隐式群表示，使有限群作用在潜空间中对应可逆线性映射，并提供可辨识性理论保证。

**[Evaluating In Silico Creativity An Expert Review Of Ai Chess Compositions](evaluating_in_silico_creativity_an_expert_review_of_ai_chess_compositions.md)**

:   使用生成式神经网络（自回归Transformer、离散扩散、MaskGit）+强化学习生成国际象棋谜题，通过奖励函数筛选具有唯一解和反直觉性的谜题，并邀请三位世界级国际象棋专家评审AI生成谜题的创造力和美学品质。

**[Evobrain Dynamic Multi-Channel Eeg Graph Modeling For Time-Evolving Brain Networ](evobrain_dynamic_multi-channel_eeg_graph_modeling_for_time-evolving_brain_networ.md)**

:   提出 EvoBrain——首次从理论上证明 **显式动态图建模** 优于隐式静态图、**time-then-graph** 架构表达力严格优于其他两种动态 GNN 范式(graph-then-time / time-and-graph)，并据此设计双流 Mamba + Laplacian PE 增强的 GCN 模型，在 TUSZ 和 CHB-MIT 数据集的癫痫检测与早期预测任务上取得 AUROC 提升 23%、F1 提升 30% 的显著效果，同时训练速度比 SOTA 快 17 倍。

**[Evolutionary Prediction Games](evolutionary_prediction_games.md)**

:   提出"演化预测博弈"框架，用演化博弈论分析预测算法与用户群体之间的反馈循环，揭示理想学习器导致竞争排斥（强者生存），而实际学习器（有限数据/代理损失/过参数化）反而能促成群体间的稳定共存与互利共生。

**[Exact Learning Of Arithmetic With Differentiable Agents](exact_learning_of_arithmetic_with_differentiable_agents.md)**

:   提出可微有限状态转换器（DFST），一种图灵完备且端到端可微的模型族，在 2D 符号网格上通过观察专家算术计算的中间步骤（Policy-Trajectory Observations）训练，仅用 20 个样本（最长 3 位数加法）即可完美泛化到 3850 位二进制加法、2450 位十进制加法，未发现任何错误。

**[Faithful Group Shapley Value](faithful_group_shapley_value.md)**

:   提出 Faithful Group Shapley Value (FGSV)，唯一满足含"忠实性"在内五条公理的组级数据估值方法，有效防御"空壳公司攻击"（通过拆分子组不当膨胀估值），并设计了 $O(n \cdot \text{Poly}(\log n))$ 复杂度的高效近似算法。

**[Flashmd Long-Stride Universal Prediction Of Molecular Dynamics](flashmd_long-stride_universal_prediction_of_molecular_dynamics.md)**

:   提出 FlashMD，基于 GNN 直接预测分子动力学轨迹的位置与动量跨步演化，实现比传统 MD 积分器大 1–2 个数量级的时间步长跨越，并在架构中融入哈密顿动力学约束，推广到任意热力学系综和通用化学体系。

**[Flowmoe A Scalable Pipeline Scheduling Framework For Distributed Mixture-Of-Expe](flowmoe_a_scalable_pipeline_scheduling_framework_for_distributed_mixture-of-expe.md)**

:   通过统一的流水线调度和优先级驱动的all-reduce张量分块，实现MHA、门控、专家计算和A2A/all-reduce通信的完全重叠，训练时间减少13-57%。

**[Fostering The Ecosystem Of Ai For Social Impact Requires Expanding And Strengthe](fostering_the_ecosystem_of_ai_for_social_impact_requires_expanding_and_strengthe.md)**

:   本文主张 AI for Social Impact (AISI) 领域的学术生态需要双轨改革：拓宽"影响力"的定义以认可非部署/非方法创新的贡献，同时对已部署系统采用因果推断级别的严格评估标准。

**[Fsnet Feasibility-Seeking Neural Network For Constrained Optimization With Guara](fsnet_feasibility-seeking_neural_network_for_constrained_optimization_with_guara.md)**

:   提出 FSNet 框架，将**可微的可行性求解步骤**集成到神经网络中，通过最小化约束违反的无约束优化来保证约束满足，同时支持端到端训练，在凸/非凸、光滑/非光滑问题上均显著快于传统求解器且保持可行性。

**[Gaussian Process Upper Confidence Bound Achieves Nearly-Optimal Regret In Noise-](gaussian_process_upper_confidence_bound_achieves_nearly-optimal_regret_in_noise-.md)**

:   本文证明 GP-UCB 在 noise-free GP bandit 问题中可达到 nearly-optimal regret，首次在 SE 核下实现 $O(1)$ 常数累积遗憾、在 Matérn 核（$d < \nu$）下实现 $O(1)$ 累积遗憾，弥合了 GP-UCB 理论与实践之间的长期差距。

**[Generalized Linear Mode Connectivity For Transformers](generalized_linear_mode_connectivity_for_transformers.md)**

:   提出统一对称性框架（置换、半置换、正交、可逆变换四级层次），首次在 Vision Transformer 和 GPT-2 上实现零/近零 barrier 的线性模式连通性（LMC），并扩展至多模型融合与异构宽度对齐。

**[Graph Alignment Via Birkhoff Relaxation](graph_alignment_via_birkhoff_relaxation.md)**

:   本文首次为图对齐问题的 Birkhoff 松弛（将排列矩阵约束松弛为双随机矩阵约束）提供了理论保证，在高斯 Wigner 模型下证明了最优解的相变行为：当噪声 $\sigma = o(n^{-1})$ 时松弛解接近真实排列，当 $\sigma = \Omega(n^{-0.5})$ 时松弛解远离真实排列。

**[Harnessing Feature Resonance Under Arbitrary Target Alignment For Out-Of-Distrib](harnessing_feature_resonance_under_arbitrary_target_alignment_for_out-of-distrib.md)**

:   发现 Feature Resonance 现象——优化已知 ID 节点表征时未知 ID 节点的表征变化显著大于 OOD 节点，且该现象与标签无关，据此提出无需多类标签的图 OOD 节点检测框架 RSL，在 13 个数据集上达到 SOTA。

**[Hessian-Guided Perturbed Wasserstein Gradient Flows For Escaping Saddle Points](hessian-guided_perturbed_wasserstein_gradient_flows_for_escaping_saddle_points.md)**

:   提出扰动Wasserstein梯度流(PWGF)算法，通过基于Hessian构造的高斯过程注入噪声扰动，使概率测度优化能够高效逃离鞍点并达到二阶最优性。

**[Houselayout3D A Benchmark And Training-Free Baseline For 3D Layout Estimation In](houselayout3d_a_benchmark_and_training-free_baseline_for_3d_layout_estimation_in.md)**

:   提出 HouseLayout3D——首个面向大规模多层建筑的真实世界 3D layout 估计基准，以及 MultiFloor3D——一个无需训练的基线方法，通过组合现代 3D 重建和分割模型在多层建筑 layout 估计上超越现有深度学习方法。

**[How Should We Evaluate Data Deletion In Graph-Based Ann Indexes](how_should_we_evaluate_data_deletion_in_graph-based_ann_indexes.md)**

:   针对图索引（graph-based ANNS）中数据删除缺乏系统评估方法的问题，形式化定义了三种删除策略（逻辑删除、物理删除、重建），提出一套面向实际部署的评估框架和指标体系，并在 HNSW 上实验分析后提出 Deletion Control 算法，可根据精度需求动态选择删除策略。

**[Hypergraphrag Retrieval-Augmented Generation Via Hypergraph-Structured Knowledge](hypergraphrag_retrieval-augmented_generation_via_hypergraph-structured_knowledge.md)**

:   提出 HyperGraphRAG，首个基于超图 (hypergraph) 结构的 RAG 方法，通过超边 (hyperedge) 建模 n 元关系（n≥2），克服了现有图谱 RAG 方法受限于二元关系的瓶颈，在医学、农业、计算机科学和法律等领域的问答任务中全面超越 StandardRAG 和 GraphRAG 系列方法。

**[Impact Of Layer Norm On Memorization And Generalization In Transformers](impact_of_layer_norm_on_memorization_and_generalization_in_transformers.md)**

:   系统揭示了LayerNorm在Pre-LN和Post-LN Transformer中的**截然不同**角色：Pre-LN中LN对学习至关重要，移除会破坏泛化；Post-LN中LN驱动记忆化，移除可抑制记忆化并恢复真实标签。

**[Improving Decision Trees Through The Lens Of Parameterized Local Search](improving_decision_trees_through_the_lens_of_parameterized_local_search.md)**

:   从参数化复杂度的视角分析决策树的局部搜索优化操作，揭示问题的难度来源，并证明特征数与值域大小的组合可实现固定参数可解(FPT)，同时提供了概念验证实现。

**[Improving Forecasts Of Suicide Attempts For Patients With Little Data](improving_forecasts_of_suicide_attempts_for_patients_with_little_data.md)**

:   提出 Latent Similarity Gaussian Process (LSGP)，将患者嵌入连续隐空间以捕获异质性，使数据稀少的患者能从相似患者"借用"预测趋势，从而改进基于 EMA 数据的自杀未遂预测。

**[Incomplete Multi-View Clustering Via Hierarchical Semantic Alignment And Coopera](incomplete_multi-view_clustering_via_hierarchical_semantic_alignment_and_coopera.md)**

:   提出 HSACC 框架，通过双层语义空间设计（低层互信息对齐 + 高层自适应加权融合）和联合优化的缺失视图隐式恢复策略，解决不完整多视图聚类中静态融合和两阶段流水线（先补全后聚类）的误差传播问题，在 5 个基准数据集上全面超越 SOTA。

**[Information-Computation Tradeoffs For Noiseless Linear Regression With Oblivious](information-computation_tradeoffs_for_noiseless_linear_regression_with_oblivious.md)**

:   对无噪声线性回归在Oblivious污染模型下，形式化证明任何高效Statistical Query算法都需要 $\tilde{\Omega}(d^{1/2}/\alpha^2)$ 的VSTAT复杂度，给出了 $1/\alpha$ 的二次依赖对高效算法具有本质性的计算下界证据。

**[Is Prm Necessary Problem-Solving Rl Implicitly Induces Prm Capability In Llms](is_prm_necessary_problem-solving_rl_implicitly_induces_prm_capability_in_llms.md)**

:   令人惊讶地，纯RL训练无需显式PRM监督即可诱发出强大的过程理解能力，且现有PRMs在SOTA模型上甚至不如简单多数投票有效。

**[Keep It On A Leash Controllable Pseudo-Label Generation Towards Realistic Long-T](keep_it_on_a_leash_controllable_pseudo-label_generation_towards_realistic_long-t.md)**

:   提出 Controllable Pseudo-label Generation (CPG) 框架，通过可控的自强化优化循环将可靠伪标签逐步纳入标注集，在已知分布上构建 Bayes-optimal 分类器，从而在未标注数据分布完全未知的 Realistic LTSSL 场景下实现最高 15.97% 的准确率提升。

**[Kernel Conditional Tests From Learning-Theoretic Bounds](kernel_conditional_tests_from_learning-theoretic_bounds.md)**

:   提出将学习算法的置信界转化为条件假设检验的统一框架，基于核岭回归构建了有限样本保证的条件两样本检验，首次支持非i.i.d.数据与在线采样场景。

**[Learning-Augmented Online Bipartite Fractional Matching](learning-augmented_online_bipartite_fractional_matching.md)**

:   本文提出了两个学习增强算法（LAB 和 PAW），用于在线二部分数匹配问题，在给定可能不准确的建议匹配的情况下，首次在整个鲁棒性范围内 Pareto 优于朴素的 CoinFlip 策略。

**[Learning-Augmented Streaming Algorithms For Correlation Clustering](learning-augmented_streaming_algorithms_for_correlation_clustering.md)**

:   提出了首个面向相关聚类（Correlation Clustering）的学习增强流算法，利用成对距离预测，在完全图上实现优于3的近似比（$\tilde{O}(n)$ 空间），在一般图上实现 $O(\log|E^-|)$ 近似比（$\tilde{O}(n)$ 空间），在空间-近似比权衡上显著改进了已有的非学习算法。

**[Learning Dynamics Of Rnns In Closed-Loop Environments](learning_dynamics_of_rnns_in_closed-loop_environments.md)**

:   从数学理论上揭示了 RNN 在闭环（agent-环境交互）与开环（监督学习）训练下呈现根本不同的学习动力学，闭环学习遵循三阶段过程，由短期策略改进与长期稳定性之间的竞争驱动。

**[Learning Non-Equilibrium Diffusions With Schrödinger Bridges From Exactly Solvab](learning_non-equilibrium_diffusions_with_schrödinger_bridges_from_exactly_solvab.md)**

:   将Schrödinger桥问题从布朗运动参考过程推广到多变量Ornstein-Uhlenbeck（mvOU）参考过程，推导高斯情形精确解，并提出无模拟的mvOU-OTFM算法处理一般分布。

**[Learning To Condition A Neural Heuristic For Scalable Mpe Inference](learning_to_condition_a_neural_heuristic_for_scalable_mpe_inference.md)**

:   提出 Learning to Condition (L2C)，用注意力神经网络学习对变量-值对进行评分，指导概率图模型中 MPE 推理的条件化决策，在保持解质量的同时大幅缩减搜索空间。

**[Look-Ahead Reasoning On Learning Platforms](look-ahead_reasoning_on_learning_platforms.md)**

:   在学习平台的用户-算法交互中形式化 level-$k$ 前瞻推理，证明个体自私的高阶推理只加速收敛但不改变均衡（无长期收益），而集体协调的收益由学习者-用户效用函数的对齐程度决定，提供了刻画协调收益上界的理论框架。

**[Maszero Designing Multiagent Systems With Zero Supervision](maszero_designing_multiagent_systems_with_zero_supervision.md)**

:   MAS-ZERO 是首个推理时自动 MAS 设计框架，通过 meta-agent 迭代设计、批评和改进 MAS 配置（包括任务分解和 sub-MAS 分配），无需验证集和训练，在推理（+16.69%）、编程（+16.66%）和搜索代理（+5.45%）任务上均超越手动和自动 MAS baseline，同时保持 Pareto 最优的准确率-成本权衡。

**[Maxsup Overcoming Representation Collapse In Label Smoothing](maxsup_overcoming_representation_collapse_in_label_smoothing.md)**

:   通过解析 Label Smoothing (LS) 的损失函数，发现其包含一个在错误分类时放大错误的"误差放大项"，导致类内特征坍缩；提出 Max Suppression (MaxSup) 方法，将惩罚目标从 ground-truth logit 转移至 top-1 logit，消除误差放大效应同时保留有益正则化。

**[Megstate Phoneme Decoding From Magnetoencephalography Signals](megstate_phoneme_decoding_from_magnetoencephalography_signals.md)**

:   提出 MEGState，一种融合多分辨率卷积和传感器级 SSM 的架构，用于从脑磁图(MEG)信号中解码音素，在 LibriBrain 数据集上显著超越基线方法。

**[Meta-Learning Three-Factor Plasticity Rules For Structured Credit Assignment Wit](meta-learning_three-factor_plasticity_rules_for_structured_credit_assignment_wit.md)**

:   本文提出一种元学习框架，通过外层梯度优化自动发现局部的新赫布式突触可塑性规则，使循环神经网络仅利用稀疏延迟奖励信号就能完成结构化的信用分配，为理解生物神经网络的学习机制提供了新视角。

**[Metafind Scene-Aware 3D Asset Retrieval For Coherent Metaverse Scene Generation](metafind_scene-aware_3d_asset_retrieval_for_coherent_metaverse_scene_generation.md)**

:   MetaFind 是一个场景感知的三模态（文本+图像+点云）3D 资产检索框架，通过引入 SE(3) 等变的空间-语义图神经网络 (ESSGNN) 编码场景布局信息，实现了在元宇宙场景生成中风格一致、空间合理的迭代式资产检索。

**[Micadangelo Fine-Grained Reconstruction Of Constrained Cad Models From 3D Scans](micadangelo_fine-grained_reconstruction_of_constrained_cad_models_from_3d_scans.md)**

:   MiCADangelo 模拟人类 CAD 设计师的逆向工程流程，通过多平面截面分析提取 2D 模式，预测带约束的参数化草图并优化拉伸参数，首次在 3D CAD 逆向工程中实现了包含草图约束的完整参数化模型重建。

**[Military Ai Needs Technically-Informed Regulation To Safeguard Ai Research And I](military_ai_needs_technically-informed_regulation_to_safeguard_ai_research_and_i.md)**

:   本文针对 AI 驱动的致命性自主武器系统 (AI-LAWS) 提出了基于系统行为（而非标签或意图）的监管标准，论证了 AI 研究者必须参与军事 AI 监管的全生命周期，并提出了五项具体的政策建议。

**[Modeling Cell Dynamics And Interactions With Unbalanced Mean Field Schrödinger B](modeling_cell_dynamics_and_interactions_with_unbalanced_mean_field_schrödinger_b.md)**

:   提出 Unbalanced Mean Field Schrödinger Bridge (UMFSB) 框架和 CytoBridge 深度学习算法，从稀疏时间快照数据中同时建模细胞的非平衡随机动力学和细胞间交互。

**[Modeling Neural Activity With Conditionally Linear Dynamical Systems](modeling_neural_activity_with_conditionally_linear_dynamical_systems.md)**

:   提出条件线性动力系统（CLDS），通过高斯过程先验让线性动力系统参数随观测到的实验协变量非线性变化，在保留线性模型可解释性和高效推断的同时建模神经回路的非线性动态。

**[Moesd Unveil Speculative Decodings Potential For Accelerating Sparse Moe](moesd_unveil_speculative_decodings_potential_for_accelerating_sparse_moe.md)**

:   揭示投机解码在中等批大小下对MoE比对稠密模型更有效，通过目标效率指标捕捉系统级瓶颈，建立可靠的性能建模，达到2.29×加速。

**[Mutualvpr A Mutual Learning Framework For Resolving Supervision Inconsistencies ](mutualvpr_a_mutual_learning_framework_for_resolving_supervision_inconsistencies_.md)**

:   提出 MutualVPR 互学习框架，通过特征驱动的自适应 K-means 聚类动态分配场景类别标签，解决分类式 VPR 方法中由视角变化和遮挡导致的监督不一致问题。

**[Neural Network For Simulating Radio Emission From Extensive Air Showers](neural_network_for_simulating_radio_emission_from_extensive_air_showers.md)**

:   用简单全连接神经网络替代计算昂贵的 CoREAS 蒙特卡洛模拟，快速预测广延大气簇射（EAS）的射电脉冲，并在 $X_{\text{max}}$ 重建任务中达到与传统模拟可比的分辨率。

**[Non-Clairvoyant Scheduling With Progress Bars](non-clairvoyant_scheduling_with_progress_bars.md)**

:   引入"进度条"信息模型作为透视与非透视调度之间的插值框架，针对对抗性和随机性进度条分别设计了具有最优一致性-鲁棒性权衡的调度算法，同时推进了学习增强调度的理论前沿。

**[Normalization In Attention Dynamics](normalization_in_attention_dynamics.md)**

:   将不同归一化方案（Post-LN、Pre-LN、Mix-LN、Peri-LN、nGPT、sqrt-scaling）统一建模为球面上交互粒子系统的速度调节机制，从理论上揭示了各方案对 token 聚类动力学和表示坍缩的不同影响，识别 Peri-LN 为理想选择。

**[On A Geometry Of Interbrain Networks](on_a_geometry_of_interbrain_networks.md)**

:   本文提出利用离散图曲率（Forman-Ricci 和 Ollivier-Ricci 曲率）分析超扫描（hyperscanning）中的脑间网络动态重构，克服传统基于相关性的脑间同步性指标在机制性解释方面的局限。

**[On Agnostic Pac Learning In The Small Error Regime](on_agnostic_pac_learning_in_the_small_error_regime.md)**

:   本文在不可知 PAC 学习的小误差域（$\tau \approx d/m$）中，构造了一个基于 ERM 聚合的计算高效学习器，实现了 $c \cdot \tau + O(\sqrt{\tau d/m} + d/m)$ 的误差上界（$c \leq 2.1$），匹配了已知下界，推进了不可知学习的精确复杂度刻画。

**[On The Surprising Effectiveness Of Large Learning Rates Under Standard Width Sca](on_the_surprising_effectiveness_of_large_learning_rates_under_standard_width_sca.md)**

:   揭示在标准参数化(SP)下，cross-entropy 损失函数使得"不稳定"区间实际分为灾难性不稳定和受控发散两个子区间：在受控发散区间（学习率 $\eta_n = \Theta(n^{-1/2})$）logits 发散但梯度和激活保持稳定，从而首次为 SP 提供了一个实用的、具有特征学习能力的无穷宽极限。

**[On Topological Descriptors For Graph Products](on_topological_descriptors_for_graph_products.md)**

:   系统研究在图的（box）积上施加各种滤过时拓扑描述子（欧拉特征 EC 和持续同调 PH）的表达能力，证明 PH 图积描述子严格强于对单图的计算，而 EC 不具备此性质，并给出高效 PH 计算算法。

**[On Universality Classes Of Equivariant Networks](on_universality_classes_of_equivariant_networks.md)**

:   本文证明等变神经网络的分离能力（区分对称等价输入的能力）不足以完全刻画其表达能力——具有相同分离能力的模型可能拥有不同的逼近能力，并给出了浅层不变网络通用性类的完整刻画及失败的充分条件。

**[One Sample Is Enough To Make Conformal Prediction Robust](one_sample_is_enough_to_make_conformal_prediction_robust.md)**

:   提出 RCP1（单样本鲁棒共形预测），通过认证共形过程本身而非单个 conformity score，仅需一次随机扰动前向传播即可获得比需要 100 次前向传播的 SOTA 方法更小的鲁棒预测集。

**[Optimism Without Regularization Constant Regret In Zero-Sum Games](optimism_without_regularization_constant_regret_in_zero-sum_games.md)**

:   首次证明无正则化的Optimistic Fictitious Play在2×2零和博弈中获得O(1)常数遗憾，匹配了正则化Optimistic FTRL的最优率，同时证明Alternating Fictitious Play的遗憾下界为Ω(√T)，分离了乐观和交替在无正则化情况下的能力。

**[Optimized Learned Count-Min Sketch](optimized_learned_count-min_sketch.md)**

:   提出 OptLCMS，通过将分数空间分区并用 KKT 条件解析求解 CMS 参数、动态规划优化阈值，大幅加速构建过程，同时提供不可容忍误差概率的理论保证。

**[OrbitZoo: Real Orbital Systems Challenges for RL](orbitzoo_real_orbital_systems_challenges_for_reinforcement_learning.md)**

:   构建OrbitZoo，基于工业标准库Orekit的多智能体RL环境，支持碰撞规避和协同机动，经Starlink真实数据验证MAPE仅0.16%。

**[Ortholoc Uav 6-Dof Localization And Calibration Using Orthographic Geodata](ortholoc_uav_6-dof_localization_and_calibration_using_orthographic_geodata.md)**

:   提出OrthoLoC——首个大规模UAV-正射影像配对数据集（16,425张，47地点，19城市），用于6-DoF定位和标定评估，AdHoP技术匹配精度提升95%、平移误差降低63%。

**[Overfitting In Adaptive Robust Optimization](overfitting_in_adaptive_robust_optimization.md)**

:   揭示自适应鲁棒优化（ARO）中策略脆弱性与机器学习过拟合的类比关系：自适应策略在不确定性集内表现优异但集外易失效，提出约束特定的不确定性集大小作为"正则化"手段来平衡鲁棒性和自适应性。

**[Polymath Evaluating Mathematical Reasoning In Multilingual Contexts](polymath_evaluating_mathematical_reasoning_in_multilingual_contexts.md)**

:   提出Value-Guided Search(VGS)——通过token级价值模型指导块级束搜索，无需预定义"步骤"，相对多数投票在竞赛数学上准确度提升+14.5%，同时推理计算效率提升30%，超越现有PRM方案。

**[Position There Is No Free Bayesian Uncertainty Quantification](position_there_is_no_free_bayesian_uncertainty_quantification.md)**

:   本文从频率学派视角质疑贝叶斯不确定性量化（UQ）的有效性，将贝叶斯更新重新解释为模型集成的优化问题，并提出基于PAC框架的校准算法以构建具有频率学派保证的预测区间。

**[Prediction-Powered Semi-Supervised Learning With Online Power Tuning](prediction-powered_semi-supervised_learning_with_online_power_tuning.md)**

:   将预测驱动推断（PPI）框架扩展到半监督学习训练过程中，提出无偏梯度估计器，并设计在线AdaGrad算法动态调节伪标签与真实标签的相对权重 $\lambda$，在保证无偏性的同时实现与最优固定 $\lambda$ 匹配的收敛速率。

**[Private Evolution Converges](private_evolution_converges.md)**

:   为Private Evolution（PE）合成数据生成算法提供了首个不依赖不现实假设的收敛性理论保证，证明在正确的超参数设置下PE输出的 $(ε,δ)$-DP 合成数据集的1-Wasserstein距离为 $\tilde{O}(d(nε)^{-1/d})$。

**[Product Distribution Learning With Imperfect Advice](product_distribution_learning_with_imperfect_advice.md)**

:   本文研究在给定不完美建议分布的情况下学习布尔超立方体上乘积分布的问题，提出了一种高效算法，当建议质量足够好时样本复杂度可实现关于维度 $d$ 的次线性依赖。

**[Radar Benchmarking Language Models On Imperfect Tabular Data](radar_benchmarking_language_models_on_imperfect_tabular_data.md)**

:   提出 Radar 基准，通过对真实表格数据注入五类数据工件（缺失值、错误值、异常值、格式不一致、逻辑不一致），系统评估语言模型在不完美表格数据上的数据感知推理能力，揭示即使是前沿模型在引入数据工件后性能也大幅下降。

**[Recurrent Self-Attention Dynamics An Energy-Agnostic Perspective From Jacobians](recurrent_self-attention_dynamics_an_energy-agnostic_perspective_from_jacobians.md)**

:   本文从动力系统的 Jacobian 分析视角，突破传统能量函数框架的对称性约束，揭示了归一化层在抑制自注意力谱范数和振荡分量方面的关键作用，发现高性能循环自注意力模型的 Lyapunov 指数趋近于零（临界态），并基于此提出谱正则化方法显著提升推理性能。

**[Redundancy-Aware Test-Time Graph Out-Of-Distribution Detection](redundancy-aware_test-time_graph_out-of-distribution_detection.md)**

:   提出 RedOUT 框架，通过最小化结构熵构建编码树来消除图结构中的冗余信息，结合冗余感知图信息瓶颈(ReGIB)原理，在测试时无需修改预训练模型参数即可有效区分ID和OOD图样本，在10个数据集对上平均AUC达87.46%。

**[Regression Trees Know Calculus](regression_trees_know_calculus.md)**

:   揭示常叶回归树中隐含的梯度信息——通过相邻节点均值差的有限差分类比，高效提取梯度估计，进而将活跃子空间（Active Subspace）和集成梯度（Integrated Gradient）等微分工具引入树模型，拓展了树模型的可解释性和预测改进能力。

**[Reliable Active Learning From Unreliable Labels Via Neural Collapse Geometry](reliable_active_learning_from_unreliable_labels_via_neural_collapse_geometry.md)**

:   提出NCAL-R框架，利用神经坍缩（Neural Collapse）的几何正则性指导主动学习的样本选择，通过类均值对齐扰动（CMAP）和特征波动（FF）两个互补信号，在标签噪声和分布偏移条件下实现可靠的主动学习。

**[Research Learning To Reason With Search For Llms Via Reinforcement Learning](research_learning_to_reason_with_search_for_llms_via_reinforcement_learning.md)**

:   ReSearch框架将搜索操作嵌入推理链中作为第一类原语，通过GRPO强化学习自动学习何时何如搜索，无需任何推理步骤的监督标注，在多跳QA任务上相对基线平均提升15.81%。

**[Resnets Are Deeper Than You Think](resnets_are_deeper_than_you_think.md)**

:   证明残差网络与前馈网络居于不同的函数空间（非简单重参数化），并通过后训练部分线性化实验表明变深度架构（类ResNet）即使在排除可训练性差异后仍优于固定深度架构，暗示残差连接提供了超越优化的归纳偏好。

**[Rethinking Pca Through Duality](rethinking_pca_through_duality.md)**

:   通过 Difference-of-Convex (DC) 框架重新审视 PCA,建立了核化和样本外推广能力,揭示了同步迭代是 DCA 的特例,并提出了鲁棒 $\ell_1$-PCA 的核化对偶公式。

**[Revisiting Agnostic Boosting](revisiting_agnostic_boosting.md)**

:   提出新的不可知 Boosting 算法,在非常一般的假设下大幅改善了此前工作的样本复杂度,并建立近匹配下界,从而在对数因子意义下解决了不可知 Boosting 的样本复杂度问题。

**[Rnns Perform Task Computations By Dynamically Warping Neural Representations](rnns_perform_task_computations_by_dynamically_warping_neural_representations.md)**

:   提出一种黎曼几何框架来分析RNN如何通过动态变形（warping）神经表示的几何结构来执行计算任务，揭示了动态变形是RNN计算的基本特征。

**[Robust Sampling For Active Statistical Inference](robust_sampling_for_active_statistical_inference.md)**

:   提出基于预算保持路径的鲁棒采样策略，通过在均匀采样和主动采样之间最优插值，确保估计器的方差永远不比两者中任何一个更差，解决了主动统计推断中不确定性估计不准确导致性能恶化的问题。

**[Sad Neural Networks Divergent Gradient Flows And Asymptotic Optimality Via O-Min](sad_neural_networks_divergent_gradient_flows_and_asymptotic_optimality_via_o-min.md)**

:   利用 o-minimal 结构的数学工具，证明了使用常见光滑激活函数（sigmoid、tanh、softplus、GELU 等）的全连接网络的梯度流存在二元性：要么收敛到临界点，要么发散到无穷大且损失收敛到渐近临界值。特别地，对多项式目标函数，证明了损失无法精确取零但可任意接近零，从而导致参数必然发散。

**[Sample-Adaptivity Tradeoff In On-Demand Sampling](sample-adaptivity_tradeoff_in_on-demand_sampling.md)**

:   系统研究了按需采样（on-demand sampling）中样本复杂度与自适应轮次之间的权衡关系，在可实现设定下证明 $r$ 轮算法的最优样本复杂度为 $dk^{\Theta(1/r)}/\varepsilon$，在不可知设定下提出仅需 $\widetilde{O}(\sqrt{k})$ 轮即可达近最优样本复杂度的LazyHedge算法，并引入OODS抽象框架建立了近紧的轮次复杂度下界。

**[Scalable Gpu-Accelerated Euler Characteristic Curves Optimization And Differenti](scalable_gpu-accelerated_euler_characteristic_curves_optimization_and_differenti.md)**

:   提出针对现代GPU（Ampere架构）优化的欧拉特征曲线（ECC）CUDA内核，实现16-2000倍加速，并引入可微PyTorch层支持端到端拓扑特征学习。

**[Scalable Inference Of Functional Neural Connectivity At Submillisecond Timescale](scalable_inference_of_functional_neural_connectivity_at_submillisecond_timescale.md)**

:   开发连续时间Poisson过程GLM的蒙特卡洛（MC）和多项式近似（PA）方法，支持亚毫秒精度的神经功能连接推断，在大规模神经记录上实现分钟级训练并揭示与已知海马解剖结构一致的连接模式。

**[Semi-Infinite Nonconvex Constrained Min-Max Optimization](semi-infinite_nonconvex_constrained_min-max_optimization.md)**

:   针对带有无穷多非凸约束的非凸 min-max 优化问题，提出 iDB-PD（不精确动态障碍原始-对偶）算法，在 Łojasiewicz 正则条件下建立了首个全局非渐近收敛保证，稳定性 $\mathcal{O}(\epsilon^{-3})$、可行性 $\mathcal{O}(\epsilon^{-6\theta})$、互补松弛 $\mathcal{O}(\epsilon^{-3\theta/(1-\theta)})$。

**[Semi-Supervised Graph Anomaly Detection Via Robust Homophily Learning](semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)**

:   提出RHO (Robust Homophily Learning)方法，通过自适应频率响应滤波器(AdaFreq)和图正常性对齐(GNA)模块，解决半监督图异常检测中正常节点同质性多样性的问题，在8个真实数据集上超越现有方法。

**[Sheaf Cohomology Of Linear Predictive Coding Networks](sheaf_cohomology_of_linear_predictive_coding_networks.md)**

:   将线性预测编码(PC)网络形式化为细胞层(cellular sheaf)结构，利用层上同调和Hodge分解分析循环拓扑中的"内部矛盾"如何阻碍学习，揭示权重初始化的全局接线模式决定网络可学习性。

**[Sign-In To The Lottery Reparameterizing Sparse Training From Scratch](sign-in_to_the_lottery_reparameterizing_sparse_training_from_scratch.md)**

:   本文发现稀疏网络从头训练(PaI)性能差的根本原因是无法像dense-to-sparse方法那样学习正确的参数符号，为此提出Sign-In重参数化方法（θ=m⊙w），通过引入内部自由度来促进符号翻转，理论证明其能解决一种互补于过参数化的符号翻转情况，实验中显著提升了稀疏从头训练的性能。

**[Smrs Advocating A Unified Reporting Standard For Surrogate Models In The Artific](smrs_advocating_a_unified_reporting_standard_for_surrogate_models_in_the_artific.md)**

:   提出代理模型报告标准(SMRS)——一个轻量级、模块化的报告框架，系统性地记录代理模型管线中的关键设计和评估选择，旨在改善AI时代代理模型的可靠性、可复现性和跨领域知识转移。

**[Space Spike-Aware Consistency Enhancement For Test-Time Adaptation In Spiking Ne](space_spike-aware_consistency_enhancement_for_test-time_adaptation_in_spiking_ne.md)**

:   提出SPACE，首个专为脉冲神经网络(SNN)设计的无源单样本测试时自适应(TTA)方法，通过最大化增强样本间脉冲行为特征图的一致性，在多个数据集和架构上实现鲁棒适应。

**[Stable Matching With Ties Approximation Ratios And Learning](stable_matching_with_ties_approximation_ratios_and_learning.md)**

:   研究有并列偏好的双边匹配市场，提出最优稳定份额(OSS)比率概念衡量公平性，证明稳定匹配分布下OSS-ratio为$\Omega(N)$但一般匹配分布下可达$O(\log N)$（渐近紧），并将离线近似结果扩展到bandit学习场景。

**[Statistical Inference For Gradient Boosting Regression](statistical_inference_for_gradient_boosting_regression.md)**

:   提出统一的梯度提升回归统计推断框架，通过将dropout和并行训练整合到Boulevard正则化中，证明了相应的中心极限定理，从而构建了内置的置信区间、预测区间和变量重要性假设检验，并发现增大dropout率和并行树数量能显著提升信号恢复（最高达2倍和4倍）。

**[Statistical Inference Under Performativity](statistical_inference_under_performativity.md)**

:   建立了表演性预测（performative prediction）下的端到端统计推断框架：为重复风险最小化（RRM）算法推导中心极限定理，提出数据驱动的协方差估计方法，并将预测驱动推断（PPI）推广到表演性设置以获得更精确的估计和更紧的置信区间。

**[The Computational Complexity Of Counting Linear Regions In Relu Neural Networks](the_computational_complexity_of_counting_linear_regions_in_relu_neural_networks.md)**

:   系统梳理了ReLU网络"线性区域"的六种非等价定义，证明对所有定义计数线性区域都是#P-hard的（一层隐藏层即如此），并在多层网络中证明了强不可近似结果和多项式空间上界。

**[The Cost Of Robustness Tighter Bounds On Parameter Complexity For Robust Memoriz](the_cost_of_robustness_tighter_bounds_on_parameter_complexity_for_robust_memoriz.md)**

:   研究 ReLU 网络鲁棒记忆（robust memorization）的参数复杂度，即在保证每个训练样本 $\mu$-邻域内预测一致的条件下插值任意数据集所需的参数数量，在鲁棒性比率 $\rho = \mu/\epsilon$ 的全范围 $(0,1)$ 内建立了更紧的上下界。

**[The Parameterized Complexity Of Computing The Vc-Dimension](the_parameterized_complexity_of_computing_the_vc-dimension.md)**

:   本文系统研究了计算VC维的参数化复杂性，证明朴素穷举算法在ETH假设下是渐近最优的，提出按最大度参数化的FPT 1-可加近似算法和按树宽参数化的2^{O(tw·log tw)}·|V|时间精确算法。

**[The Persistence Of Neural Collapse Despite Low-Rank Bias](the_persistence_of_neural_collapse_despite_low-rank_bias.md)**

:   本文从理论上证明了深度神经坍缩（DNC）在深层无约束特征模型中由于 L2 正则化引起的低秩偏差而全局次优，同时首次解释了 DNC 在实践中持续出现的原因——其解空间维度随网络宽度增长快于低秩解。

**[The Structural Complexity Of Matrix-Vector Multiplication](the_structural_complexity_of_matrix-vector_multiplication.md)**

:   证明对于 corrupted VC-dimension 为 $d$ 的布尔矩阵 $\mathbf{M} \in \{0,1\}^{m \times n}$，矩阵-向量乘法可在 $\widetilde{O}(nm^{1-1/d}+m)$ 时间内完成，首次为结构化矩阵提供了真亚二次时间上界，推翻了 OMv 猜想在结构化输入上的适用性，并导出了动态 Laplacian 求解器、有效电阻、三角检测等问题的首个高精度亚二次算法。

**[Tight Bounds On The Distortion Of Randomized And Deterministic Distributed Votin](tight_bounds_on_the_distortion_of_randomized_and_deterministic_distributed_votin.md)**

:   本文研究分布式投票模型中的度量扭曲 (metric distortion) 问题，针对四种代价目标 ($\text{avg-avg}$, $\text{avg-max}$, $\text{max-avg}$, $\text{max-max}$)，在确定性和随机机制下给出了改进的紧界或近紧界，几乎完整地刻画了这一模型的扭曲特性。

**[Training The Untrainable Introducing Inductive Bias Via Representational Alignme](training_the_untrainable_introducing_inductive_bias_via_representational_alignme.md)**

:   提出Guidance方法，通过逐层表征对齐（CKA）将一个网络（guide）的架构归纳偏置迁移到另一个原本"不可训练"的网络（target），从而使FCN能做图像分类、RNN逼近Transformer的语言建模性能。

**[Transfer Learning For Benign Overfitting In High-Dimensional Linear Regression](transfer_learning_for_benign_overfitting_in_high-dimensional_linear_regression.md)**

:   提出两步式Transfer MNI方法，在高维过参数化线性回归中通过"保留目标信号+零空间迁移源知识"机制增强良性过拟合的泛化能力，刻画了模型偏移和协变量偏移下的非渐近excess risk，并发现了"免费午餐"协变量偏移区间。

**[Uncertainty Estimation By Flexible Evidential Deep Learning](uncertainty_estimation_by_flexible_evidential_deep_learning.md)**

:   提出 $\mathcal{F}$-EDL，通过将 EDL 中的 Dirichlet 分布推广为 Flexible Dirichlet (FD) 分布来建模类别概率分布，从而在保持单次前向传播效率的同时，显著增强不确定性估计在噪声、长尾、分布偏移等复杂场景下的泛化能力。

**[Uncertainty Quantification For Reduced-Order Surrogate Models Applied To Cloud M](uncertainty_quantification_for_reduced-order_surrogate_models_applied_to_cloud_m.md)**

:   本文提出一种后验的、模型无关的不确定性量化框架，利用共形预测为潜空间降阶模型的重建、潜在动力学和端到端预测提供统计有效的预测区间，并在云微物理ROM上验证。

**[Uniformer Unified And Efficient Transformer For Reasoning Across General And Cus](uniformer_unified_and_efficient_transformer_for_reasoning_across_general_and_cus.md)**

:   提出 UniFormer，一种面向 GPU 和 FPGA 跨平台部署的统一高效 Transformer 架构，通过双分支注意力机制（全局线性注意力 + 局部块注意力）实现了高并行性和计算存储融合。

**[Variational Regularized Unbalanced Optimal Transport Single Network Least Action](variational_regularized_unbalanced_optimal_transport_single_network_least_action.md)**

:   提出 Var-RUOT，通过将正则化非平衡最优传输（RUOT）问题的最优性必要条件融入参数化和损失设计，仅需学习单个标量场即可求解 RUOT，获得更低作用量的解并提升训练稳定性；同时分析了增长惩罚函数对生物先验的影响。

**[Webthinker Empowering Large Reasoning Models With Deep Research Capability](webthinker_empowering_large_reasoning_models_with_deep_research_capability.md)**

:   WebThinker赋予大型推理模型(LRM)自主的网络搜索与导航能力，通过Think-Search-Draft策略实现推理、信息采集与报告生成的无缝交织，经RL优化后在复杂推理与科学报告生成任务上超越o1与Gemini。

**[Weight Weaving Parameter Pooling For Data-Free Model Merging](weight_weaving_parameter_pooling_for_data-free_model_merging.md)**

:   本文提出Weight Weaving，一种即插即用的无数据模型合并增强方法，通过在缩放因子搜索空间上对模型参数进行池化操作（如平均、随机选择），消除了对评估数据的依赖，在多任务学习、持续学习和域泛化三个场景中平均准确率最高提升15.9个百分点。
