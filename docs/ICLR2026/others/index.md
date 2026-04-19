---
title: >-
  ICLR2026 其他方向 69篇论文解读
description: >-
  69篇ICLR2026 其他方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📂 其他

**🔬 ICLR2026** · **69** 篇论文解读

**[A Federated Generalized Expectation-Maximization Algorithm for Mixture Models with an Unknown Number of Components](a_federated_generalized_expectation-maximization_algorithm_for_mixture_models_wi.md)**

:   提出 FedGEM 算法，通过客户端本地 EM 步后构建不确定性集、服务器利用不确定性集交集检测聚类重叠并推断全局聚类数，首次实现在全局聚类数未知情况下的联邦聚类，并提供了概率收敛保证。

**[A Representer Theorem for Hawkes Processes via Penalized Least Squares Minimization](a_representer_theorem_for_hawkes_processes_via_penalized_least_squares_minimizat.md)**

:   为线性多元 Hawkes 过程在 RKHS 框架下的触发核估计建立了新型表示定理，证明最优估计器可用等价核在数据点上的线性组合表示且对偶系数全部解析地等于 1，无需求解对偶优化问题，从而实现高效可扩展的非参数估计。

**[A Scalable Inter-edge Correlation Modeling in CopulaGNN for Link Sign Prediction](a_scalable_inter-edge_correlation_modeling_in_copulagnn_for_link_sign_prediction.md)**

:   将 CopulaGNN 从节点级扩展到边级，通过将相关矩阵构造为边嵌入的 Gramian 矩阵并利用 Woodbury 恒等式重构条件概率分布，实现了在签名图上对边间统计依赖的可扩展建模，用于链接符号预测任务。

**[A Single Architecture for Representing Invariance Under Any Space Group](a_single_architecture_for_representing_invariance_under_any_space_group.md)**

:   设计了一种可自适应任意空间群不变性的单一架构 (Crystal Fourier Transformer)，通过解析推导群操作对傅里叶系数的约束来构造对称适配的傅里叶基，用约束的对偶图表示实现了跨 230 个空间群的参数共享和零样本泛化。

**[Active Learning for Decision Trees with Provable Guarantees](active_learning_for_decision_trees_with_provable_guarantees.md)**

:   为决策树主动学习提供首个理论保证：(1) 首次分析决策树的不一致系数（disagreement coefficient）并给出 $O(\ln^{OPT}(n))$ 上界；(2) 提出首个达到乘法误差 $(1+\epsilon)$ 保证的二分类主动学习算法；结合两者实现数据集大小的多对数标签复杂度。

**[Addressing Divergent Representations from Causal Interventions on Neural Networks](addressing_divergent_representations_causal.md)**

:   系统性地揭示因果干预（activation patching、DAS、SAE 等）会将模型内部表征推离自然分布，理论区分"无害偏移"与"有害偏移"两类情况，并提出 Counterfactual Latent (CL) loss 来约束干预表征不偏离流形，在 7B LLM 上验证可减少偏移同时保持干预准确率。

**[Agnostics: Learning to Synthesize Code in Any Programming Language with a Universal RL Environment](agnostics_learning_to_code_in_any_programming_language_via_reinforcement_with_a_.md)**

:   提出Agnostics，一种语言无关的后训练pipeline：将编程任务统一为I/O行为规范格式，用通用验证器+GRPO强化学习训练LLM在任何编程语言上编码，使Qwen 4B在Lua/Julia/R/OCaml/Fortran五种低资源语言上达到匹敌16B-70B模型的SOTA水平。

**[An Information-Theoretic Framework For Optimizing Experimental Design To Distinguish Probabilistic Neural Codes](an_information-theoretic_framework_for_optimizing_experimental_design_to_disting.md)**

:   提出 **information gap** 这一信息论度量，通过推导在似然编码和后验编码假说下解码器交叉熵性能差异的解析表达式（本质是真实后验与任务边际化代理后验之间的 KL 散度），定量评估给定实验设计区分两种概率神经编码假说的能力，并通过最大化该度量来优化刺激先验分布，实现理论驱动的最优实验设计。

**[ANO: Faster is Better in Noisy Landscapes](ano_faster_is_better_in_noisy_landscape.md)**

:   提出 Ano 优化器，将更新方向和幅度解耦——方向用动量的符号（sign）确保噪声鲁棒，幅度用瞬时梯度绝对值（而非动量幅度）确保响应速度，配合改进的 Yogi 式方差估计，在噪声和非平稳环境（如 RL）中显著优于 Adam/Lion/Adan，同时在标准任务上保持竞争力。

**[AnyUp: Universal Feature Upsampling](anyup_universal_feature_upsampling.md)**

:   AnyUp 提出首个**编码器无关**的可学习特征上采样方法，通过 feature-agnostic 卷积层和窗口注意力机制，仅训练一次即可对任意视觉特征在任意分辨率间进行高质量上采样，在语义分割、深度估计等任务上达到 SOTA。

**[Articulation in Motion: Prior-Free Part Mobility Analysis for Articulated Objects](articulation_in_motion_prior-free_part_mobility_analysis_for_articulated_objects.md)**

:   提出AiM（Articulation in Motion）框架，从交互视频和初始状态扫描中无需部件数量先验地重建铰接物体——通过双高斯表征（静态GS + 可变形GS）实现动静解耦，结合顺序RANSAC进行无先验部件分割和关节估计，辅以SDMD模块处理新暴露的静态区域，在复杂6部件物体（Storage）上以79.34% mean IoU大幅超越需先验的ArtGS（52.23%）。

**[CaDrift: A Time-dependent Causal Generator of Drifting Data Streams](cadrift_a_time-dependent_causal_generator_of_drifting_data_streams.md)**

:   提出 CaDrift，一个基于结构因果模型（SCM）的时间依赖合成数据流生成框架，通过 EWMA 平滑和自回归噪声引入时序相关性，并通过修改因果映射函数实现可控的分布漂移、协变量漂移、严重漂移和局部漂移，填补了现有数据流生成器既不因果又不时序依赖的空白。

**[Characterizing and Optimizing the Spatial Kernel of Multi Resolution Hash Encodings](characterizing_and_optimizing_the_spatial_kernel_of_multi_resolution_hash_encodi.md)**

:   从物理系统角度分析 Instant-NGP 的多分辨率哈希编码（MHE），推导出其点扩展函数（PSF）的闭式近似，发现有效分辨率由平均分辨率 $N_{\text{avg}}$ 而非最细分辨率 $N_{\max}$ 决定，且存在网格引起的各向异性，并提出零开销的 Rotated MHE（R-MHE）通过逐层旋转输入坐标消除各向异性。

**[CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning](chlu_the_causal_hamiltonian_learning_unit_as_a_symplectic_primitive_for_deep_lea.md)**

:   CHLU 是一种基于相对论哈密顿力学和辛积分的计算学习原语，通过强制相空间体积守恒和引入因果速度上限，解决了 LSTM 的梯度爆炸/消失和 Neural ODE 的信息耗散问题，实现无限时域稳定性和热力学生成能力。

**[Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevance Assessment](completing_missing_annotation_multi-agent_debate_for_accurate_and_scalable_relev.md)**

:   提出DREAM——基于对立立场初始化的多Agent多轮辩论框架用于IR相关性标注：一致时自动标注、分歧时交给人工(含辩论历史辅助)。达到95.2% balanced accuracy且仅3.5%需人工介入，据此构建BRIDGE基准数据集，发现29,824个原有基准缺失的相关标注(原标注的428%)，修正了检索系统排名偏差和RAG中检索-生成性能不匹配问题。

**[Compositional Diffusion with Guided Search for Long-Horizon Planning](compositional_diffusion_long_horizon_planning.md)**

:   提出 CDGS（Compositional Diffusion with Guided Search），通过在扩散去噪过程中嵌入基于种群的搜索机制（迭代重采样 + 似然剪枝），解决组合式扩散模型在多模态局部分布合成时的模式平均问题，从短时域模型采样出全局一致的长时域规划。

**[Consistent Low-Rank Approximation](consistent_low-rank_approximation.md)**

:   提出并系统研究"一致低秩近似"问题——在流数据中逐行到达的矩阵上维护近最优 rank-$k$ 近似的同时最小化解的总变化量（recourse），证明加性误差下 $O(k/\varepsilon \cdot \log(nd))$ recourse 可行，乘性 $(1+\varepsilon)$ 误差下 $k^{3/2}/\varepsilon^2 \cdot \text{polylog}$ recourse 可行，并给出 $\Omega(k/\varepsilon \cdot \log(n/k))$ 的下界。

**[Directional Sheaf Hypergraph Networks: Unifying Learning on Directed and Undirected Hypergraphs](directional_sheaf_hypergraph_networks_unifying_learning_on_directed_and_undirect.md)**

:   本文提出 Directional Sheaf Hypergraph Networks (DSHN)，通过将 Cellular Sheaf 理论与有向超图的方向信息结合，构造了一种复值 Hermitian Laplacian 算子，统一并推广了现有的图和超图 Laplacian，在 7 个真实数据集上相对准确率提升 2%–20%。

**[Distributed Algorithms for Euclidean Clustering](distributed_algorithms_for_euclidean_clustering.md)**

:   在分布式环境下为 Euclidean $(k,z)$-clustering 构造 $(1+\varepsilon)$-coreset，在 coordinator 模型和 blackboard 模型中均达到通信复杂度的最优下界（至多差 polylog 因子）。

**[Distributionally Robust Classification for Multi-Source Unsupervised Domain Adaptation](distributionally_robust_classification_for_multi-source_unsupervised_domain_adap.md)**

:   提出一种分布鲁棒学习框架，通过联合建模目标域协变量分布和条件标签分布的不确定性，在目标数据极度稀缺或源域存在虚假相关性的UDA场景中显著提升泛化性能。

**[DA-AC: Distributions as Actions — A Unified RL Framework for Diverse Action Spaces](distributions_as_actions_a_unified_framework_for_diverse_action_spaces.md)**

:   DA-AC 提出将动作分布的参数（如 softmax 概率或 Gaussian 均值/方差）作为 Agent 的"动作"输出，将动作采样过程移入环境，从而用统一的确定性策略梯度框架处理离散/连续/混合动作空间，理论证明方差严格低于 LR 和 RP 估计器，并在 40+ 环境上取得 competitive 或 SOTA 性能。

**[Enhancing Generative Auto-bidding with Offline Reward Evaluation and Policy Search](enhancing_generative_auto_bidding.md)**

:   提出 AIGB-Pearl，为生成式自动竞价方法引入离线轨迹评估器和 KL-Lipschitz 约束的分数最大化方案，使生成模型能在理论保证下安全地突破静态离线数据的性能天花板，在淘宝真实广告系统上实现 GMV +3% 的显著提升。

**[Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks](entropic_confinement_and_mode_connectivity_in_overparameterized_neural_networks.md)**

:   揭示了低损失路径上曲率的系统性增长会产生熵力屏障，即使路径能量平坦，SGD噪声也会将优化动力学约束在最小值附近的平坦区域，从而解释了"模式连通但动力学受限"的悖论。

**[Evaluating GFlowNet from Partial Episodes for Stable and Flexible Policy-Based Training](evaluating_gflownet_from_partial_episodes_for_stable_and_flexible_policy-based_t.md)**

:   建立GFlowNet中状态流函数与策略评价函数之间的理论联系，提出子轨迹评价平衡（Sub-EB）目标用于可靠学习评价函数，增强策略基GFlowNet训练的稳定性和灵活性。

**[Exchangeability of GNN Representations with Applications to Graph Retrieval](exchangeability_gnn_representations.md)**

:   发现训练好的 GNN 节点嵌入沿特征维度是可交换随机变量（即 $p(\mathbf{X}) = p(\mathbf{X}\pi)$ 对任意维度排列 $\pi$ 成立），利用此性质通过维度排序将基于传输距离（EMD/Wasserstein）的图相似度近似为欧氏距离，构建统一的局部敏感哈希（LSH）框架 GraphHash，在子图匹配和图编辑距离（GED）检索任务上以 AUC 指标一致超越 FourierHashNet、DiskANN、IVF、CORGII、SWWL 等基线，可扩展到 100 万图语料库。

**[Fast and Stable Riemannian Metrics on SPD Manifolds via Cholesky Product Geometry](fast_and_stable_riemannian_metrics_on_spd_manifolds_via_cholesky_product_geometr.md)**

:   揭示Cholesky流形上的简单乘积结构，基于此提出两种快速且数值稳定的SPD度量（PCM和BWCM），所有黎曼算子均有闭式表达式，在SPD深度学习中实现效果、效率和稳定性的三重提升。

**[FastLSQ: Solving PDEs in One Shot via Fourier Features with Exact Analytical Derivatives](fastlsq_solving_pdes_in_one_shot_via_fourier_features_with_exact_analytical_deri.md)**

:   利用正弦基函数的循环导数闭式结构，实现了无需自动微分、无需迭代训练的 PDE 一次性求解框架，在线性 PDE 上 0.07s 达到 $10^{-7}$ 精度，非线性 PDE 上 <9s 达到 $10^{-8}$–$10^{-9}$ 精度，比 PINNs 快数千倍且精确数个数量级。

**[Federated ADMM from Bayesian Duality](federated_admm_from_bayesian_duality.md)**

:   从变分贝叶斯(VB)视角推导出ADMM的贝叶斯对偶结构，证明经典ADMM是VB在各向同性高斯族上的特例，并导出Newton-like（二次目标一轮收敛）和Adam-like（深度异构场景+7%准确率）两个新扩展。

**[FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity Tradeoff](fire_frobenius_isometry_reinitialization.md)**

:   将持续学习中的稳定性-可塑性平衡形式化为约束优化问题——最小化权重偏差（稳定性）同时约束权重正交性（可塑性），得到正交 Procrustes 问题的闭式解 $\tilde{W}^* = W(W^\top W)^{-1/2}$（极分解），通过 Newton-Schulz 迭代高效实现（<1% 额外时间），在视觉持续学习、LLM 持续预训练和 RL 上全面超越 S&P 等基线。

**[From Movement to Cognitive Maps: RNNs Reveal How Locomotor Development Shapes Hippocampal Spatial Coding](from_movement_to_cognitive_maps.md)**

:   结合幼鼠运动发育的聚类分析和浅层 RNN 预测学习模型，首次计算性地证明运动统计特征的发育变化（爬行→行走→奔跑→成年）驱动了海马空间调谐神经元（位置细胞、方向细胞、联合编码细胞）的序贯涌现，定量复现大鼠海马记录数据的发育时间线，并预测了联合位置-方向编码细胞在发育中逐渐增多这一现象且在实验数据中得到验证。

**[Harpoon: Generalised Manifold Guidance for Conditional Tabular Diffusion](harpoon_generalised_manifold_guidance_for_conditional_tabular_diffusion.md)**

:   将流形理论从图像扩展到表格数据扩散模型，证明任意可微推理时损失的梯度都位于数据流形切线空间中（不限于平方误差损失），据此提出Harpoon方法在推理时沿流形引导无条件样本满足多样化表格约束。

**[HEEGNet: Hyperbolic Embeddings for EEG](heegnet_hyperbolic_embeddings_for_eeg.md)**

:   首次系统验证EEG数据具有双曲性（层次结构），提出HEEGNet混合双曲网络架构，结合欧几里得编码器提取时空频谱特征和双曲编码器捕捉层次关系，配合创新的粗到细域适应策略(DSMDBN)，在视觉诱发电位、情感识别和颅内EEG多个跨域任务上达到SOTA。

**[Hilbert-Guided Sparse Local Attention](hilbert-guided_sparse_local_attention.md)**

:   利用Hilbert空间填充曲线将2D图像token重排为保持空间邻近性的1D序列，大幅提升局部注意力的块稀疏率（空块比例从87.5%到96.9%），结合FlexAttention实现窗口注意力4倍和滑动注意力18倍加速，精度损失极小。

**[Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime](implicit_bias_of_per-sample_adam_on_separable_data_departure_from_the_full-batch.md)**

:   首次证明mini-batch Adam的隐式偏差与full-batch不同：构造数据集使单样本Adam收敛到 $\ell_2$ 最大间隔分类器（而full-batch Adam收敛到 $\ell_\infty$），并通过AdamProxy刻画一般数据集上的数据自适应Mahalanobis范数间隔最大化行为。

**[Improving Black-Box Generative Attacks via Generator Semantic Consistency](improving_black-box_generative_attacks_via_generator_semantic_consistency.md)**

:   通过分析生成器中间层特征的语义退化现象，提出基于 Mean Teacher 的语义结构感知框架，在生成器早期层进行自特征蒸馏以保持语义一致性，从而增强对抗样本在跨模型、跨域、跨任务场景中的可迁移性。

**[In-Context Algebra](in-context_algebra.md)**

:   本文设计了一个 **in-context 代数任务**——令 token 成为纯变量、每条序列重新随机分配含义——发现 Transformer 在此设定下不再学习经典的傅里叶/几何表示，而是涌现出三种 **符号推理机制**（交换复制、单位元识别、闭包消去），并揭示了训练过程中这些能力按阶段性相变依次出现的规律。

**[Jackpot: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch RL](jackpot_optimal_budgeted_rejection_sampling_for_extreme_actor-policy_mismatch_re.md)**

:   提出 Jackpot 框架，通过 Optimal Budget Rejection Sampling（OBRS）以可控接受预算在 token 级别拒绝/重加权 rollout 样本，理论证明任意预算下都能严格缩小 actor-policy 间 KL 散度，配合 rollout 模型联合训练与蒸馏，使小模型（如 Qwen3-1.7B）rollout 训练大模型（如 Qwen3-8B）达到接近 on-policy 的性能。

**[Key and Value Weights Are Probably All You Need: On the Necessity of the Query, Key, and Value Weight Triplet in Self-Attention](key_and_value_weights_are_probably_all_you_need_on_the_necessity_of_the_query_ke.md)**

:   理论证明Transformer自注意力中Query/Key/Value权重三元组存在冗余——Query权重可被替换为单位矩阵（减少25%注意力参数），GPT风格模型从头训练验证在适当超参数调整下性能不降，且训练在3倍更低权重衰减下仍然稳定。

**[Latent Equivariant Operators for Robust Object Recognition: Promises and Challenges](latent_equivariant_operators_for_robust_object_recognition_promises_and_challeng.md)**

:   在潜空间中学习/预定义等变移位算子来处理旋转和平移等群变换，推理时通过KNN搜索推断变换参数并恢复到标准pose后分类，在MNIST上展示了训练范围外变换的成功外推能力，相比传统网络和等变网络更灵活，但向复杂数据集扩展仍面临挑战。

**[Latent Fourier Transform](latent_fourier_transform.md)**

:   提出 LatentFT 框架，在扩散自编码器的潜在时间序列表征上应用离散傅里叶变换按时间尺度分离音乐模式，训练时使用随机相关对数频率掩码让解码器学习从部分频谱重建，推理时用户通过指定频率掩码选择性保留/混合不同时间尺度的音乐元素，在条件生成和音乐融合任务上全面超越 ILVR/Guidance/Codec Filtering/RAVE 等基线，29 名音乐家听力测试统计显著确认其音质和融合能力优越。

**[LPWM: Latent Particle World Models for Object-Centric Stochastic Dynamics](latent_particle_world_models_self-supervised_object-centric_stochastic_dynamics_.md)**

:   LPWM 是首个能扩展到真实世界多物体数据集的自监督物体中心世界模型，核心创新是为每个粒子学习独立的潜在动作分布（per-particle latent actions），通过因果时空 Transformer 并行编码所有帧，支持动作/语言/图像目标/多视角等多种条件生成，在视频预测上达到 SOTA 并展示了模仿学习能力（OGBench task3 成功率 89%）。

**[Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation](learning_adaptive_distribution_alignment_with_neural_characteristic_function_for.md)**

:   提出ADAlign框架，利用神经特征函数在谱域自适应对齐源/目标图分布——无需手动选择对齐标准，自动识别每个迁移场景中最显著的分布差异。在10个数据集16个迁移任务上达SOTA，同时降低内存和训练时间。

**[Learning on a Razor's Edge: Identifiability and Singularity of Polynomial Neural Networks](learning_on_a_razors_edge_identifiability_and_singularity_of_polynomial_neural_n.md)**

:   本文利用代数几何工具，对多项式激活的 MLP 和 CNN 进行了系统性分析：证明了 MLP 的有限可辨识性和 CNN 的唯一可辨识性，揭示了稀疏子网络对应神经流形的奇异点，并从"临界暴露性"角度给出了 MLP 稀疏偏差的几何解释——而 CNN 不具备这种偏差。

**[Learning Structure-Semantic Evolution Trajectories for Graph Domain Adaptation](learning_structure-semantic_evolution_trajectories_for_graph_domain_adaptation.md)**

:   提出DiffGDA——首个将扩散模型引入图域适应(GDA)的方法，用随机微分方程(SDE)建模源图到目标图的连续时间结构-语义联合演化过程，配合基于密度比的域感知引导网络驾驶扩散轨迹朝向目标域，理论证明收敛到最优适应路径，在8个真实数据集14个迁移任务上全面超越SOTA。

**[LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models](lipnext_scaling_up_lipschitz-based_certified_robustness_to_billion-parameter_mod.md)**

:   提出LipNeXt——首个无约束、无卷积的1-Lipschitz架构，通过正交流形优化学习正交矩阵 + 由Theorem 1理论驱动的Spatial Shift Module实现空间混合，成功扩展到十亿参数规模，在CIFAR-10/100、Tiny-ImageNet和ImageNet上全面刷新认证鲁棒精度(CRA) SOTA，ImageNet上 $\varepsilon=1$ 时CRA提升达+8%。

**[Lipschitz Bandits with Stochastic Delayed Feedback](lipschitz_bandits_with_stochastic_delayed_feedback.md)**

:   首次系统研究连续臂空间 Lipschitz bandit 在随机延迟反馈下的学习问题，针对有界延迟提出 Delayed Zooming 算法（通过 lazy update 机制保持 $\Delta(x) \leq 6r_t(x)$ 的子最优 gap 界），针对无界延迟提出 DLPP 分阶段剪枝策略（遗憾与延迟分位数 $Q(p)$ 挂钩），并建立实例相关下界证明 DLPP 近最优。

**[Missing Mass for Differentially Private Domain Discovery](missing_mass_for_differentially_private_domain_discovery.md)**

:   从 missing mass（缺失质量）角度重新审视差分隐私域发现问题，首次为简单且可扩展的 Weighted Gaussian Mechanism (WGM) 在 Zipfian 数据上证明了近最优的 $\ell_1$ 缺失质量上界和无分布假设的 $\ell_\infty$ 缺失质量保证，并将 WGM 作为域发现前置步骤应用于未知域的 private top-$k$ 和 $k$-hitting set 问题，在六个真实数据集上验证了理论结果。

**[Neural Force Field: Few-shot Learning of Generalized Physical Reasoning](neural_force_field_few-shot_learning_of_generalized_physical_reasoning.md)**

:   提出Neural Force Field（NFF），将物体交互建模为连续力场，通过神经算子学习力场函数并用ODE积分器解码轨迹，在I-PHYRE（100条轨迹）、N-body（200条轨迹）、PHYRE（0.012M数据,比SOTA少267倍）三个基准上实现少样本SOTA，跨场景RMSE降低32-64%,规划任务接近人类水平。

**[Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning](noisy-pair_robust_representation_alignment_for_positive-unlabeled_learning.md)**

:   提出 NcPU 非对比 PU 学习框架，通过对标准非对比损失做 sqrt 变换（NoiSNCL）让 clean pair 梯度主导训练、用 PhantomGate 提供保守负监督并支持 regret 回退，两个模块在 EM 框架下迭代互利；在不依赖辅助负样本或预估类先验的前提下，CIFAR-100 上与监督学习差距从 14.26% 缩至 <1.4%，xBD 灾损评估上同样达到 SOTA。

**[On the Impact of the Utility in Semivalue-based Data Valuation](on_the_impact_of_the_utility_in_semivalue-based_data_valuation.md)**

:   本文通过引入"空间签名"（spatial signature）的几何表示，将数据估值中的 utility 选择问题统一建模为单位圆上的方向旋转问题，并提出了一个量化鲁棒性的指标 $R_p$，揭示了 Banzhaf 值在不同 utility 下表现出最高的排序稳定性。

**[On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets](on_the_lipschitz_continuity_of_set_aggregation_functions_and_neural_networks_for.md)**

:   系统研究了三种常用集合聚合函数（sum/mean/max）和注意力机制在三种多集距离函数下的Lipschitz连续性，推导出集合神经网络的Lipschitz常数上界，并将其与扰动稳定性和分布偏移泛化联系起来。

**[Optimizer Choice Matters for the Emergence of Neural Collapse](optimizer_choice_matters_for_the_emergence_of_neural_collapse.md)**

:   通过 3,900+ 次训练实验和理论分析，揭示了优化器选择（特别是权重衰减的耦合方式）对 Neural Collapse 现象涌现起关键决定性作用——AdamW（解耦权重衰减）无法产生 Neural Collapse，而 SGD 和 Adam（耦合权重衰减）可以。

**[Out of the Shadows: Exploring a Latent Space for Neural Network Verification](out_of_the_shadows_exploring_a_latent_space_for_neural_network_verification.md)**

:   将 zonotope 视为高维超立方体的"投影（影子）"，发现输入集和输出包围体共享同一潜空间，据此提出规范驱动的输入细化方法，将输出端的不安全约束反向传递到输入空间来剪枝，使分支定界子问题数减少 60-65%，且所有运算均为矩阵操作从而实现高效 GPU 加速，在 VNN-COMP'24 八个基准上与 α-β-CROWN 等顶级工具取得可比性能。

**[Oversmoothing, Oversquashing, Heterophily, Long-Range, and More: Demystifying Common Beliefs in Graph Machine Learning](oversmoothing_oversquashing_heterophily_long-range_and_more_demystifying_common_.md)**

:   本文系统梳理了图机器学习领域围绕 oversmoothing、oversquashing、同质/异质性和长程依赖的九个常见误区，通过简洁反例逐一反驳，将"oversquashing"拆分为**计算瓶颈**和**拓扑瓶颈**两个独立概念，厘清了领域中广泛存在的概念混淆。

**[OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection](owleye_zero-shot_learner_for_cross-domain_graph_data_anomaly_detection.md)**

:   提出 OwlEye 框架，利用基于成对距离统计的跨域特征对齐将异构图嵌入共享空间，从多图中提取 attribute-level 和 structure-level 正常模式存入可扩展字典，并通过截断注意力重建机制在完全零样本条件下检测未见图的异常节点，8 数据集平均 AUPRC 36.17% 超越最强 baseline ARC 约 5.4 个百分点。

**[Predicting Kernel Regression Learning Curves from Only Raw Data Statistics](predicting_kernel_regression_learning_curves_from_only_raw_data_statistics.md)**

:   提出 Hermite 特征结构假设（HEA），仅用数据协方差矩阵和目标函数的 Hermite 分解两个统计量，就能解析预测旋转不变核在真实图像数据集（CIFAR-5m、SVHN、ImageNet）上的学习曲线（测试误差 vs 样本量），并证明该假设在高斯数据下成立，且 MLP 在特征学习 regime 下也按 HEA 预测的顺序学习 Hermite 多项式。

**[Probabilistic Kernel Function for Fast Angle Testing](probabilistic_kernel_function_for_fast_angle_testing.md)**

:   本文研究高维欧氏空间中的角度测试问题，提出两个基于参考角度的确定性概率核函数 $K_S^1$ 和 $K_S^2$，分别用于角度比较和角度阈值判断，无需高斯分布的渐近假设即可获得理论保证，并将其应用于近似最近邻搜索（ANNS），在 HNSW 图上实现 2.5×–3× 的 QPS 加速。

**[Revisiting Sharpness-Aware Minimization: A More Faithful and Effective Implementation](revisiting_sharpness-aware_minimization_a_more_faithful_and_effective_implementa.md)**

:   对 SAM 的底层机制提出新的直觉解释——扰动点梯度近似局部最大值方向，并揭示其不精确性及多步退化问题，进而提出 XSAM 通过显式估计最大值方向实现更忠实更有效的锐度感知最小化。

**[Scalable Random Wavelet Features: Efficient Non-Stationary Kernel Approximation with Convergence Guarantees](scalable_random_wavelet_features_efficient_non-stationary_kernel_approximation_w.md)**

:   提出 Random Wavelet Features (RWF)，通过从小波族中随机采样构建可扩展的非平稳核近似，保留随机特征的线性时间复杂度，同时具有正定性、无偏性和一致收敛保证。

**[SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding](seed_towards_more_accurate_semantic_evaluation_for_visual_brain_decoding.md)**

:   提出 SEED（Semantic Evaluation for Visual Brain Decoding），一个结合 Object F1、Cap-Sim 和 EffNet 三个互补指标的组合评估度量，在与人类评估的对齐度上显著超越现有所有指标。

**[Speculative Actions: A Lossless Framework for Faster AI Agents](speculative_actions_faster_ai_agents.md)**

:   借鉴 CPU 推测执行和 LLM 推测解码的思想，提出 Speculative Actions 框架：在慢速 Actor（大模型）计算时用快速 Speculator（小模型）预测未来动作并预执行，匹配时跳过等待实现无损加速，在 Chess/电商/问答等场景实现 15-30% 延迟降低，置信度动态分支策略用 40% 更少 token 达到近似 3 条推测的加速效果。

**[t-SNE Exaggerates Clusters, Provably](t-sne_exaggerates_clusters_provably.md)**

:   从理论上严格证明 t-SNE 存在两个根本性失败模式：（1）无法从输出推断输入聚类的强度，（2）无法忠实地展示极端离群点——即使输入毫无聚类结构或存在极端离群点，t-SNE 也可能产生完美聚类的可视化。

**[The Counting Power of Transformers](the_counting_power_of_transformers.md)**

:   证明 Transformer 不仅能捕获（半）线性计数性质，还能表达所有**半代数计数性质**（即多元多项式不等式的布尔组合），从而推广了先前关于 Transformer 计数能力的所有结果，并由此推导出新的不可判定性结论。

**[The Hot Mess of AI: How Does Misalignment Scale With Model Intelligence and Task Complexity?](the_hot_mess_of_ai_how_does_misalignment_scale_with_model_intelligence_and_task_.md)**

:   将AI模型错误分解为偏差（systematic misalignment）和方差（incoherent behavior），发现：推理越长→越不连贯；更大模型在困难任务上更不连贯。这暗示未来超级AI更可能表现为"工业事故"式的不可预测失败，而非一致追求错误目标。

**[The Invisibility Hypothesis: Promises of AGI and the Future of the Global South](the_invisibility_hypothesis_promises_of_agi_and_the_future_of_the_global_south.md)**

:   提出"不可见性假说"（Invisibility Hypothesis），论证AI系统日益成为经济和政治分配的协调层时将系统性偏向"机器可读"个体，全球南方的非正式工人因缺乏数字可验证性而被管理性排斥（managed exclusion），核心风险从job displacement转向relevance loss，且排斥具有自我强化特性。

**[The Price of Robustness: Stable Classifiers Need Overparameterization](the_price_of_robustness_stable_classifiers_need_overparameterization.md)**

:   建立了不连续分类器的稳定性-泛化界，证明了分类任务中的"鲁棒性代价定律"：任何参数量 $p \approx n$ 的插值分类器必然不稳定，实现高稳定性需要 $p \approx nd$ 量级的过参数化。

**[Towards Sustainable Investment Policies Informed by Opponent Shaping](towards_sustainable_investment_policies_informed_by_opponent_shaping.md)**

:   形式化证明 InvestESG 模拟环境在何种条件下构成社会困境，并应用 Advantage Alignment 对抗塑形算法引导经济智能体走向可持续投资均衡。

**[Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition](training_deep_normalization-free_spiking_neural_networks_with_lateral_inhibition.md)**

:   提出基于皮层兴奋-抑制（E-I）回路的无归一化学习框架 DeepEISNN，通过 E-I Init 和 E-I Prop 两项技术实现深度 SNN 的稳定端到端训练，兼顾性能与生物合理性。

**[When to Retrain after Drift: A Data-Only Test of Post-Drift Data Size Sufficiency](when_to_retrain_after_drift_a_data-only_test_of_post-drift_data_size_sufficiency.md)**

:   CALIPER提出了一种检测器和模型无关的、仅依赖数据的检验方法，通过跟踪加权局部回归的代理误差随局部性参数$\theta$的单调性变化，来估计突发概念漂移后重训练所需的最小数据量，无需实际重训练下游模型。
