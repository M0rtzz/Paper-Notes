---
title: >-
  NeurIPS2025 预训练/数据方向 39篇论文解读
description: >-
  39篇NeurIPS2025 预训练/数据方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📚 预训练/数据

**🧠 NeurIPS2025** · 共 **39** 篇

**[Ai Progress Should Be Measured By Capability-Per-Resource Not Scale Alone A Fram](ai_progress_should_be_measured_by_capability-per-resource_not_scale_alone_a_fram.md)**

:   本文以 position paper 的形式挑战"规模至上主义"，提出以**能力-每-资源（Capability-Per-Resource, CPR）**取代单纯的规模扩张来衡量 AI 进步，并给出一套基于梯度引导的资源分配理论框架——通过发布"梯度蓝图"元数据，使下游适配者仅微调高影响力参数子集即可在资源占用大幅降低的同时保持接近全参数微调的性能。

**[Alternating Gradient Flows A Theory Of Feature Learning In Two-Layer Neural Netw](alternating_gradient_flows_a_theory_of_feature_learning_in_two-layer_neural_netw.md)**

:   提出交替梯度流（AGF）理论框架解释神经网络的逐步"鞍到鞍"特征学习动力学——将训练建模为休眠神经元的效用最大化和活跃神经元的代价最小化的交替过程，统一了对角线性网络、注意力模型和模块加法的特征选择分析，预测与实际梯度流高度一致。

**[An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems](an_empirical_investigation_of_neural_odes_and_symbolic_regression_for_dynamical_.md)**

:   系统实证研究 Neural ODE 和符号回归（SR）在动力系统建模中的组合使用：NODE 可以在动态相似条件下外推到新边界条件，SR 可以从有噪声数据中恢复控制方程，且用 NODE 训练数据（仅 10% 原始数据）生成的数据也能让 SR 恢复大部分方程。

**[Beyond Benign Overfitting In Nadaraya-Watson Interpolators](beyond_benign_overfitting_in_nadaraya-watson_interpolators.md)**

:   通过调节 Nadaraya-Watson 插值器中的单一带宽参数 $\beta$，精确刻画了从灾难性过拟合（$\beta < d$）→ 良性过拟合（$\beta = d$）→ 温和过拟合（$\beta > d$）的完整相变谱，证明高估数据内禀维度比低估更安全。

**[Breaking The Frozen Subspace Importance Sampling For Low-Rank Optimization In Ll](breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)**

:   发现GaLore等低秩优化方法的主导子空间在预训练中会"冻结"（相邻子空间重叠度趋近1），导致权重更新卡在固定低秩子空间中；提出SARA（重要性采样子空间选择），按奇异值权重随机采样奇异向量构建子空间，证明收敛性的同时将低秩优化器与全秩Adam的性能差距缩小最高46%。

**[Broken Tokens Your Language Model Can Secretly Handle Non-Canonical Tokenization](broken_tokens_your_language_model_can_secretly_handle_non-canonical_tokenization.md)**

:   揭示 LLM 能秘密处理非标准分词（如将"Hello"拆为"He"+"llo"而非标准的"Hello"整词token）——即使输入的 token 序列与训练时不同，模型表现出惊人的鲁棒性，且这种能力来自嵌入空间中子词嵌入的线性组合近似整词嵌入的特性。

**[Conformal Risk Training End-To-End Optimization Of Conformal Risk Control](conformal_risk_training_end-to-end_optimization_of_conformal_risk_control.md)**

:   本文将 Conformal Risk Control (CRC) 从期望损失扩展到一般化的 Optimized Certainty-Equivalent (OCE) 风险度量（包含 CVaR 等尾部风险），并提出"共形风险训练"方法，通过在训练中端到端地微分共形风险控制过程，在保持可证明风险保证的同时显著改善平均情况性能。

**[Differentiable Hierarchical Visual Tokenization](differentiable_hierarchical_visual_tokenization.md)**

:   提出一种端到端可微分的层次化视觉分词器，以像素级粒度自适应图像内容进行 token 划分，利用信息准则进行层次模型选择，可直接替换 ViT 的固定 patch 分词，并支持光栅-矢量转换。

**[Disaggregation Reveals Hidden Training Dynamics The Case Of Agreement Attraction](disaggregation_reveals_hidden_training_dynamics_the_case_of_agreement_attraction.md)**

:   通过将聚合的语法评测指标**分解**到实验条件层面并追踪训练过程中的变化，发现语言模型的语法学习并非渐进单调的，而是经历了一系列**隐藏的突破阶段**——先学习词频偏好、再学习局部上下文（n-gram），最后逐步掌握更远距离的语法依赖关系。

**[Does Object Binding Naturally Emerge In Large Pretrained Vision Transformers](does_object_binding_naturally_emerge_in_large_pretrained_vision_transformers.md)**

:   通过定义 IsSameObject 谓词并设计二次探针，证明大规模预训练 ViT（尤其是 DINO、CLIP）自然涌现了目标绑定能力，该信号编码在低维子空间中并主动引导注意力机制，挑战了认知科学界认为 ViT 缺乏绑定能力的观点。

**[Efficient Pre-Training Of Llms Via Topology-Aware Communication Alignment On Mor](efficient_pre-training_of_llms_via_topology-aware_communication_alignment_on_mor.md)**

:   提出 Arnold 调度系统，通过将 LLM 训练的通信模式（DP/PP group）与数据中心物理网络拓扑对齐，在模拟中将通信组最大跨度减少 1.67x，在 9600+ GPU 生产级训练中端到端性能提升 10.6%。

**[Enhancing Training Data Attribution With Representational Optimization](enhancing_training_data_attribution_with_representational_optimization.md)**

:   提出 AirRep（Attentive Influence Ranking Representation），一种基于表示学习的训练数据归因方法，通过可训练编码器和注意力池化机制，在推理效率比梯度方法快约 80 倍的同时，达到甚至超越 SOTA 梯度方法的归因精度。

**[Final-Model-Only Data Attribution With A Unifying View Of Gradient-Based Methods](final-model-only_data_attribution_with_a_unifying_view_of_gradient-based_methods.md)**

:   明确提出"仅有最终模型"(FiMO)的训练数据归因设定，将问题从"贡献度"重构为"敏感性"度量，提出 further training 作为金标准，并统一推导出多种梯度方法（Grad-Dot、影响函数、TRAK、DataInf 等）均为 further training 的不同阶近似。

**[Flatness Is Necessary Neural Collapse Is Not Rethinking Generalization Via Grokk](flatness_is_necessary_neural_collapse_is_not_rethinking_generalization_via_grokk.md)**

:   利用 grokking（延迟泛化）作为因果探针，证明 **relative flatness 是泛化的（潜在）必要条件**，而 neural collapse 虽常伴随泛化出现，但并非必要——它只是通往 flatness 的一条路径。

**[Gemstones A Model Suite For Multi-Faceted Scaling Laws](gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)**

:   Gemstones开源4000+检查点数据集（至2B参数），系统研究宽度-深度-训练代币在缩放律中的影响，揭示缩放律对设计选择的高度敏感性。

**[Generalization Bounds For Rank-Sparse Neural Networks](generalization_bounds_for_rank-sparse_neural_networks.md)**

:   本文证明了利用神经网络权重矩阵近似低秩结构的泛化界，当 Schatten $p$ 拟范数较小时，样本复杂度仅为 $\widetilde{O}(WrL^2)$，其中 $W$, $L$, $r$ 分别为宽度、深度和权重矩阵的秩。

**[Global Minimizers Of Sigmoid Contrastive Loss](global_minimizers_of_sigmoid_contrastive_loss.md)**

:   首次在实践相关的 N≫d 区间严格刻画了 Sigmoid 对比损失（SigLIP）在可训练温度和偏置下的全局最小值几何结构，提出了 (m, b_rel)-Constellation 这一新型组合对象，并用其解释了 SigLIP 的检索成功、模态间隙现象，以及提出了显式 relative bias 参数化改进训练动态。

**[Gradient-Weight Alignment As A Train-Time Proxy For Generalization In Classifica](gradient-weight_alignment_as_a_train-time_proxy_for_generalization_in_classifica.md)**

:   提出 Gradient-Weight Alignment (GWA)，通过量化每个训练样本梯度与模型权重的方向一致性（cosine similarity），在训练过程中无需验证集即可准确预测泛化性能、确定最佳早停时机，并定位有影响力的训练样本。

**[How Does Sequence Modeling Architecture Influence Base Capabilities Of Pre-Train](how_does_sequence_modeling_architecture_influence_base_capabilities_of_pre-train.md)**

:   通过"限定领域预训练 + OOD 测试"的评估框架揭示 Mamba/RWKV 等 stateful 架构存在基础能力退化，并归纳出关键设计原则——"全序列任意选择能力"（full-sequence visibility + real relation calculation + non-uniform distribution），用极简的 Top-1 Element/Chunk Selection 架构验证该原则可恢复至接近 Transformer 的基础能力。

**[Language Model Behavioral Phases Are Consistent Across Archi](language_model_behavioral_phases_are_consistent_across_archi.md)**

:   论文在 Transformer、Mamba、RWKV，不同数据集与参数规模（14M 到 12B）上系统分析 1400+ checkpoints，发现语言模型预训练中存在高度一致的行为阶段；词级行为变化最多可由 unigram 频率、n-gram 概率、语义相似度三类简单启发式解释（最高约 98% 方差）。

**[Learning The Wrong Lessons Syntactic-Domain Spurious Correlations In Language Mo](learning_the_wrong_lessons_syntactic-domain_spurious_correlations_in_language_mo.md)**

:   揭示 LLM 学会了句法模板（PoS n-gram）与领域之间的虚假关联，导致跨域性能骤降，甚至可利用此关联绕过安全拒绝机制（refusal bypass），在 OLMo-2 上将拒绝率从 40% 降至 2.5%。

**[Learning To Flow From Generative Pretext Tasks For Neural Architecture Encoding](learning_to_flow_from_generative_pretext_tasks_for_neural_architecture_encoding.md)**

:   提出 FGP（Flow-based Generative Pre-training），通过让编码器重建"流代理"（flow surrogate）这一架构信息流的简化表征，使任意结构的编码器无需专用的异步消息传递设计即可捕获信息流，在性能预测中 Precision@1% 最高提升 106%。

**[Leveraging Importance Sampling To Detach Alignment Modules From Large Language M](leveraging_importance_sampling_to_detach_alignment_modules_from_large_language_m.md)**

:   提出 Residual Alignment Model (RAM)，将 LLM 对齐过程形式化为重要性采样，将大模型分解为冻结的 Proposal Module 和可训练的小型 Residual Aligner，以不到 1/8 参数实现可比甚至超越全参数 SFT/DPO 的对齐效果，同时解决了首 token 延迟问题。

**[Memory Mosaics At Scale](memory_mosaics_at_scale.md)**

:   Memory Mosaics v2 将关联存储网络扩展至 10B 参数、1T token 训练规模，在新任务学习和上下文学习上显著超越同规模甚至 8T token 训练的 Transformer。

**[Nemotron-Climb Clustering-Based Iterative Data Mixture Bootstrapping For Languag](nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)**

:   NVIDIA 提出 CLIMB 框架，通过嵌入聚类 + 迭代自举搜索自动发现最优预训练数据混合比例，在 1B 模型上超过 Llama-3.2-1B 达 2.0%，并发布了 1.2T token 的 ClimbLab 语料库和 400B token 的 ClimbMix 高质量数据集。

**[Neural Collapse Under Gradient Flow On Shallow Relu Networks For Orthogonally Se](neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)**

:   首次证明在正交可分数据上，两层ReLU网络的梯度流（GF）在小初始化下可证收敛到Neural Collapse（NC）解，揭示了GF隐式偏置（早期神经元对齐+渐近最大间隔偏置）在促进NC出现中的关键作用。

**[Optimal Online Change Detection Via Random Fourier Features](optimal_online_change_detection_via_random_fourier_features.md)**

:   提出 Online RFF-MMD 算法，通过随机 Fourier 特征近似 MMD 统计量并嵌入到二进制网格的序贯检验框架中，实现了无需训练数据、无需窗口参数的在线非参数变点检测，运行时间和空间复杂度均为对数级，并证明了检测延迟的 minimax 最优性。

**[Power Lines Scaling Laws For Weight Decay And Batch Size In Llm Pre-Training](power_lines_scaling_laws_for_weight_decay_and_batch_size_in_llm_pre-training.md)**

:   提出了一套针对 LLM 预训练中权重衰减 $\lambda$ 和批大小 $B$ 的幂律缩放定律（power laws），通过 AdamW 时间尺度 $\tau$ 的概念统一了超参数缩放关系，使得在大规模训练前即可准确预测最优超参数。

**[Predict Training Data Quality Via Its Geometry In Metric Space](predict_training_data_quality_via_its_geometry_in_metric_space.md)**

:   提出基于持久同调（Persistent Homology）的训练数据多样性度量方法，证明数据的几何/拓扑结构特征能够有效预测模型性能，优于传统基于熵的Vendi Score等指标。

**[Prescribe Predicting Single-Cell Responses With Bayesian Estimation](prescribe_predicting_single-cell_responses_with_bayesian_estimation.md)**

:   提出 PRESCRIBE 框架，通过多变量深度证据回归联合建模单细胞扰动预测中的认知不确定性（模型对输入的不熟悉程度）和随机不确定性（生物系统固有的随机性），生成伪 E-distance 作为统一的不确定性代理指标，过滤不可靠预测后准确率提升 3% 以上。

**[Quantifying Task-Relevant Representational Similarity Using Decision Variable Co](quantifying_task-relevant_representational_similarity_using_decision_variable_co.md)**

:   本文提出基于决策变量相关（DVC）的新方法来衡量两个神经表征在分类任务上的逐试次一致性，发现深度网络在 ImageNet 上准确率越高反而与猴脑 V4/IT 的 DVC 越低，对抗训练和大规模数据集预训练也无法缩小这一差距。

**[Retrospective Incontext Learning For Temporal Credit Assignm](retrospective_incontext_learning_for_temporal_credit_assignm.md)**

:   论文提出 RICL（Retrospective In-Context Learning），利用 LLM 的预训练知识把环境中的稀疏奖励回溯性转化为稠密 advantage supervision，再结合在线策略迭代框架 RICOL，在 BabyAI 四个场景中以更高样本效率达到与传统在线 RL 相当的收敛表现，展示了 LLM 在 temporal credit assignment 上的潜力。

**[Scalable Fingerprinting Of Large Language Models](scalable_fingerprinting_of_large_language_models.md)**

:   提出 Perinucleus 采样方法生成可扩展的 LLM 指纹，能在 Llama-3.1-8B 上嵌入 24,576 个指纹（比现有方法多两个数量级）且不损害模型能力，并通过理论和实验证明大规模指纹是抵御共谋攻击的关键。

**[Scaling Embedding Layers In Language Models](scaling_embedding_layers_in_language_models.md)**

:   提出Scone方法，通过为高频n-gram学习上下文化的嵌入（用独立Transformer模型训练），在推理时将这些嵌入卸载到主存/SSD，实现"训练时用更多计算但推理时不增加加速器资源"的新缩放范式，1B参数模型超越1.9B基线。

**[Superposition Yields Robust Neural Scaling](superposition_yields_robust_neural_scaling.md)**

:   揭示表示叠加（superposition）是神经缩放定律的核心驱动力：在强叠加区间，损失**通用地**与模型维度成反比（$L \propto 1/m$），且该行为与数据频率分布的具体形式无关，这与实际 LLM 的缩放行为一致。

**[The Curse Of Depth In Large Language Models](the_curse_of_depth_in_large_language_models.md)**

:   揭示 Pre-LN Transformer 中输出方差指数增长导致深层退化为恒等映射的根本原因，提出无参数的 LayerNorm Scaling（LNS）策略——仅在 LayerNorm 后乘以 $1/\sqrt{\ell}$，将方差从指数增长压缩为多项式增长，在 130M-7B 全规模上稳定改进困惑度 5-8%。

**[Through The River Understanding The Benefit Of Schedule-Free Methods For Languag](through_the_river_understanding_the_benefit_of_schedule-free_methods_for_languag.md)**

:   从 River-Valley 损失景观的几何视角深入分析 Schedule-Free (SF) 优化器，揭示 SF-AdamW 在不需要学习率衰减或权重平均的情况下自动沿"河流"方向优化，并提出改进变体解决动量敏感性和大批量训练的局限性。

**[Understanding And Enhancing Mask-Based Pretraining Towards Universal Representat](understanding_and_enhancing_mask-based_pretraining_towards_universal_representat.md)**

:   用高维线性回归理论精确刻画了 mask-based pretraining 中掩码率对测试风险的影响（偏差-方差分解），揭示了最优掩码率依赖于任务和模型大小，并据此提出 R2MAE（随机随机掩码），在视觉、语言、DNA、单细胞模型上一致超越固定掩码率。

**[Zeus Zero-Shot Embeddings For Unsupervised Separation Of Tabular Data](zeus_zero-shot_embeddings_for_unsupervised_separation_of_tabular_data.md)**

:   ZEUS 是首个面向表格数据的零样本聚类方法，通过在合成数据集上预训练一个 Transformer 编码器来学习可泛化的表示，使得新数据集无需任何额外训练或调参即可在单次前向传播中完成高质量聚类。
