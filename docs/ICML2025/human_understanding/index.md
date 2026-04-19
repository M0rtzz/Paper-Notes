---
title: >-
  ICML2025 人体理解方向 39篇论文解读
description: >-
  39篇ICML2025 人体理解方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧑 人体理解

**🧪 ICML2025** · **39** 篇论文解读

**[A Generalizable Physics-Enhanced State Space Model for Long-Term Dynamics Forecasting in Complex Environments](a_generalizable_physics-enhanced_state_space_model_for_long-term_dynamics_foreca.md)**

:   提出 Phy-SSM，将部分已知的物理知识融入深度状态空间模型（SSM），通过动力学分解（已知/未知矩阵）和物理状态正则化，实现对噪声大、不规则采样数据的长期动力学精准预测与外推。

**[AAAR-1.0: Assessing AI's Potential to Assist Research](aaar-10_assessing_ais_potential_to_assist_research.md)**

:   提出 AAAR-1.0 基准，通过公式推断、实验设计、论文弱点发现、审稿质量鉴别四个专家级任务，系统评估 LLM 辅助科研的真实能力，揭示当前模型在深度研究任务上仍有显著不足。

**[Access Controls Will Solve the Dual-Use Dilemma](access_controls_will_solve_the_dual-use_dilemma.md)**

:   提出基于访问控制的概念框架来解决AI安全中的双用途困境（dual-use dilemma），通过用户身份验证获取真实世界上下文，结合内容分类实现细粒度的权限管理，同时缓解过度拒绝（over-refusal）和不足拒绝（under-refusal）问题。

**[Beyond CVaR: Leveraging Static Spectral Risk Measures for Enhanced Decision-Making in Distributional Reinforcement Learning](beyond_cvar_leveraging_static_spectral_risk_measures_for_enhanced_decision-makin.md)**

:   提出首个在分布式 RL 框架内优化一般静态谱风险度量（SRM）的算法，超越了仅限于简单 CVaR 的现有方法，通过利用回报分布实现闭式外层优化和中间风险度量的时间分解，在多种风险设置中超越现有风险敏感 DRL 模型。

**[Deep Electromagnetic Structure Design Under Limited Evaluation Budgets](deep_electromagnetic_structure_design_under_limited_evaluation_budgets.md)**

:   提出 Progressive Quadtree-based Search (PQS) 方法，通过四叉树层次化表示压缩电磁结构的高维设计空间，并利用基于一致性的样本选择机制在有限仿真预算下高效搜索优质设计，相比生成式方法节省 75~85% 的评估成本。

**[Diffusion Sampling Correction via Approximately 10 Parameters](diffusion_sampling_correction_via_approximately_10_parameters.md)**

:   提出PCA-based Adaptive Search (PAS)方法，利用采样轨迹处于高维空间低维子空间的几何特性，通过PCA提取少量正交基向量并仅学习约10个坐标参数来修正现有快速采样器的截断误差，在单张A100上亚分钟训练即可将DDIM在CIFAR10上的FID从15.69降至4.37（NFE=10）。

**[Doubly Robust Fusion of Many Treatments for Policy Learning](doubly_robust_fusion_of_many_treatments_for_policy_learning.md)**

:   提出校准加权治疗融合（Calibration-Weighted Treatment Fusion）方法，通过双重稳健地合并具有相似效果的治疗组来降低动作空间维度，使得现有多臂策略学习方法（如策略树）可高效应用于大量治疗选项的个体化推荐场景。

**[DSSD: Efficient Edge-Device LLM Deployment and Collaborative Inference via Distributed Split Speculative Decoding](dssd_efficient_edge-device_llm_deployment_and_collaborative_inference_via_distri.md)**

:   提出分布式拆分推测解码（DSSD）框架，将推测解码的验证阶段拆分到设备端和边缘端，用一次下行传输（LLM的单个词表分布）替代多次上行传输（SLM的$\gamma$个词表分布），在保持推理质量不变的前提下大幅降低通信延迟。

**[Efficient Logit-based Knowledge Distillation of Deep Spiking Neural Networks for Full-Range Timestep Deployment](efficient_logit-based_knowledge_distillation_of_deep_spiking_neural_networks_for.md)**

:   提出一种时间维度解耦的 logit 蒸馏框架，利用 SNN 固有的时空特性，将训练目标分解到每个时间步，实现单模型在全范围推理时间步上的高性能部署，无需为不同时间步重新训练。

**[Enhancing Decision-Making of Large Language Models via Actor-Critic](enhancing_decision-making_of_large_language_models_via_actor-critic.md)**

:   提出 LAC（LLM-based Actor-Critic）框架，通过 token logits 的正/负结果概率比构建 Q 函数（Critic），并用 KL 约束闭式解实现无梯度策略优化（Actor），在 ALFWorld、BabyAI-Text、WebShop 三个基准上用 7B/8B 模型超越 GPT-4 + ReAct。

**[Enhancing Parallelism in Decentralized Stochastic Convex Optimization](enhancing_parallelism_in_decentralized_stochastic_convex_optimization.md)**

:   提出 Decentralized Anytime SGD (DAT-SGD)，通过在渐变平均查询点上计算梯度来缓解共识距离偏差，将去中心化随机凸优化的并行度上界从 $\mathcal{O}(\rho^{1/2} N^{1/4})$ 提升至 $\mathcal{O}(\rho \sqrt{N})$，在高连通拓扑下首次匹配中心化学习的速率。

**[Erwin: A Tree-based Hierarchical Transformer for Large-scale Physical Systems](erwin_a_tree-based_hierarchical_transformer_for_large-scale_physical_systems.md)**

:   提出 Erwin，一种基于 ball tree 分层结构的 Transformer 架构，通过将注意力计算限制在固定大小的局部球区域内，实现线性时间复杂度，同时通过渐进式粗化/细化和跨球交互机制捕获多尺度特征，在宇宙学、分子动力学、PDE 求解和粒子流体动力学多个领域达到 SOTA。

**[FedRAG: A Framework for Fine-Tuning Retrieval-Augmented Generation Systems](fedrag_a_framework_for_fine-tuning_retrieval-augmented_generation_systems.md)**

:   FedRAG 提出了一个同时支持集中式和联邦式架构的 RAG 系统微调框架，填补了 RAG 生态系统中缺乏统一微调工具的空白，并通过轻量级抽象实现了从集中式到联邦式训练的无缝转换。

**[From Logits to Hierarchies: Hierarchical Clustering made Simple](from_logits_to_hierarchies_hierarchical_clustering_made_simple.md)**

:   提出 L2H（Logits to Hierarchies）算法，仅利用预训练平面聚类模型的 logits 输出，通过掩码 softmax 和迭代合并策略，在无需微调的情况下构建高质量层次聚类，大幅超越专用深度层次聚类模型，且在 ImageNet 规模数据集上 CPU 运行不到一分钟。

**[Generative Social Choice: The Next Generation](generative_social_choice_the_next_generation.md)**

:   将生成式社会选择框架扩展至带成本/预算约束和近似查询的场景，提出 DemocraticProcess 算法并给出近乎最优的近似比例代表性理论保证，实现了实用系统 PROSE（基于 GPT-4o）在药物评论和城市治理数据集上验证有效性。

**[If Open Source Is to Win, It Must Go Public](if_open_source_is_to_win_it_must_go_public.md)**

:   本文论证了开源 AI 在当前实践下无法独立实现 AI 民主化——模型权重只是"惰性代码"，需要大量资本才能激活——必须嵌入公共 AI 基础设施（公共资金 + 公共访问 + 公共治理 + 私人承诺）才能成为真正的公共产品。

**[Improving Model Alignment through Collective Intelligence of Open-Source LLMs](improving_model_alignment_through_collective_intelligence_of_open-source_llms.md)**

:   本文提出 Mixture of Agents Alignment（MoAA），利用多个开源 LLM 的集体智慧生成高质量的对齐数据（SFT 数据和偏好数据），显著提升目标模型在 Arena-Hard 和 AlpacaEval2 上的表现，并展示了无需外部强监督的自我提升能力。

**[KELPS: A Framework for Verified Multi-Language Autoformalization via Semantic-Syntactic Alignment](kelps_a_framework_for_verified_multi-language_autoformalization_via_semantic-syn.md)**

:   提出基于断言逻辑的中间表示——知识方程(Knowledge Equation)，实现自然语言数学命题到多种形式语言(Lean4/Coq/Isabelle)的规则化翻译，在 MiniF2F 上 pass@1 句法准确率达 88.9%，超越 DeepSeek-V3 和 Herald。

**[LLaVA-ReID: Selective Multi-Image Questioner for Interactive Person Re-Identification](llava-reid_selective_multi-image_questioner_for_interactive_person_re-identifica.md)**

:   本文定义了交互式行人重识别（Inter-ReID）新任务，构建了 Interactive-PEDES 多轮对话数据集，并提出 LLaVA-ReID——一个基于选择性多图像上下文和前瞻性监督的大多模态问题生成模型，通过迭代对话逐步细化目标人物描述。

**[Log-Sum-Exponential Estimator for Off-Policy Evaluation and Learning](log-sum-exponential_estimator_for_off-policy_evaluation_and_learning.md)**

:   提出基于 log-sum-exponential (LSE) 算子的新型非线性估计器，用于离线策略评估与学习，在重尾奖励和噪声倾向分数场景下显著降低方差并提供理论保证。

**[Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation](merge-friendly_post-training_quantization_for_multi-target_domain_adaptation.md)**

:   首次系统分析量化引入的离散化噪声如何破坏模型融合效果，提出 HDRQ（Hessian and Distance Regularizing Quantization）——通过 Hessian 正则化平坦化损失曲面 + 距离正则化保持量化模型间权重对齐 + 噪声采样舍入消除舍入歧义，使量化模型在多目标域适应融合中性能接近全精度水平。

**[MERIT: Maximum-normalized Element-wise Ratio for Language Model Large-batch Training](merit_maximum-normalized_element-wise_ratio_for_language_model_large-batch_train.md)**

:   提出 MERIT 优化器，通过最大范数归一化与逐元素信任比率扩展 LAMB，有效解决大批量训练中注意力 logit 爆炸导致的性能退化问题。

**[Provably Improving Generalization of Few-Shot Models with Synthetic Data](provably_improving_generalization_of_few-shot_models_with_synthetic_data.md)**

:   提出一个理论框架量化合成数据与真实数据的分布差异对少样本分类泛化能力的影响，并基于该理论设计了联合优化数据划分与模型训练的算法，在10个基准数据集上超越SOTA。

**[Reactivation: Empirical NTK Dynamics Under Task Shifts](reactivation_empirical_ntk_dynamics_under_task_shifts.md)**

:   首次系统实证研究了持续学习中NTK的动态行为，发现任务切换会一致性地触发NTK的突变——即使在lazy学习体制下，NTK的范数、速度和对齐指标都在任务边界出现急剧偏差，揭示了一种被称为"重激活"（reactivation）的特征学习现象，并通过区分概念性和频率性分布偏移精确定位了驱动因素。

**[RULEBREAKERS: Challenging LLMs at the Crossroads between Formal Logic and Human-like Reasoning](rulebreakers_challenging_llms_at_the_crossroads_between_formal_logic_and_human-l.md)**

:   构建首个大规模"规则破坏者"数据集 RULEBREAKERS（25,600 实例），系统评估 7 个 LLM 在形式逻辑推理与事实知识冲突时的表现，发现模型普遍倾向过度刚性地应用逻辑规则而忽略常识，与人类推理行为存在显著偏离。

**[SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability](saebench_a_comprehensive_benchmark_for_sparse_autoencoders_in_language_model_int.md)**

:   提出 SAEBench——一个包含 8 项评估指标的综合基准，系统评测稀疏自编码器（SAE）在语言模型可解释性中的表现，揭示了代理指标（稀疏-保真度）与下游任务性能之间的严重脱节。

**[Scaling Large Motion Models with Million-Level Human Motions](scaling_large_motion_models_with_million-level_human_motions.md)**

:   本文提出 MotionLib（首个百万级运动数据集，120 万条序列）、MotionBook（无损特征 + 2D 无查找运动分词器）和 Being-M0（大型运动模型），首次在运动生成领域展示了数据和模型规模的 scaling law。

**[Semantic Shift Estimation via Dual-Projection and Classifier Reconstruction for Exemplar-Free Class-Incremental Learning](semantic_shift_estimation_via_dual-projection_and_classifier_reconstruction_for_.md)**

:   提出 DPCR 方法，通过双投影（任务级 TSSP + 类别级 CIP）估计语义漂移，并用岭回归无BP地重建分类器，同时解决无样例类增量学习中的语义漂移和决策偏差问题，在多个基准上超越 SOTA。

**[Sketch-Plan-Generalize: Learning and Planning with Neuro-Symbolic Programmatic Representations for Inductive Spatial Concepts](sketch-plan-generalize_learning_and_planning_with_neuro-symbolic_programmatic_re.md)**

:   提出 SPG（Sketch-Plan-Generalize）——一种神经符号智能体框架，将归纳式概念学习分解为三阶段流水线：概念签名推断（Sketch）、基于 MCTS 的 grounded 动作序列搜索（Plan）、以及 LLM 驱动的程序归纳泛化（Generalize），在从少量演示中学习可组合、可泛化的空间抽象概念方面显著优于纯 LLM 和纯神经方法。

**[Sparse Spectral Training and Inference on Euclidean and Hyperbolic Neural Networks](sparse_spectral_training_and_inference_on_euclidean_and_hyperbolic_neural_networ.md)**

:   提出 Sparse Spectral Training (SST)，通过在谱域上每步更新全部奇异值、按奇异值大小多项式采样选择性更新奇异向量，并周期性 re-SVD 保持正交性，实现接近全秩训练的预训练效果，同时显存开销与 LoRA 相当。

**[Streamline Without Sacrifice — Squeeze out Computation Redundancy in LMM](streamline_without_sacrifice_--_squeeze_out_computation_redundancy_in_lmm.md)**

:   提出 ProxyV，通过引入少量代理视觉 token（proxy vision tokens）替代原始视觉 token 参与 LLM 解码层中的重计算操作（自注意力、FFN），在保留全部视觉信息的前提下大幅压缩计算冗余，甚至在部分设定下提升性能。

**[Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups](sum-of-parts_self-attributing_neural_networks_with_end-to-end_learning_of_featur.md)**

:   SOP 提出了一种将任意可微分模型转换为基于分组的自归因神经网络（SANN）的框架，通过端到端学习特征分组实现了在 SANN 中的 SOTA 性能，并从理论上证明了逐特征 SANN 的误差下界和分组 SANN 的零误差可达性。

**[TabFlex: Scaling Tabular Learning to Millions with Linear Attention](tabflex_scaling_tabular_learning_to_millions_with_linear_attention.md)**

:   用线性注意力替换 TabPFN 中的 softmax 注意力，将表格分类的 ICL 方法从小数据集扩展到百万级样本，实现 2× 以上加速且性能不降。

**[TopInG: Topologically Interpretable Graph Learning via Persistent Rationale Filtration](toping_topologically_interpretable_graph_learning_via_persistent_rationale_filtr.md)**

:   TopInG 提出了一种基于持久同调的拓扑可解释图学习框架，通过学习"基本原理过滤"（rationale filtration）来识别稳定且持久的基本原理子图，引入"拓扑差异"（topological discrepancy）约束来强化基本原理子图与无关子图之间的拓扑区分，在处理多变形态的基本原理子图时显著优于现有方法。

**[Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)**

:   FlashTrace 提出了一种高效的多 token 归因方法，通过跨度聚合（span-wise aggregation）将多 token 目标的归因复杂度从 $\mathcal{O}(M \cdot N)$ 降至 $\mathcal{O}(N)$，并通过递归归因（recursive attribution）机制追溯推理链中的重要性传播，实现了 130 倍以上的速度提升。

**[Truly Self-Improving Agents Require Intrinsic Metacognitive Learning](truly_self-improving_agents_require_intrinsic_metacognitive_learning.md)**

:   本文提出一个形式化框架论证了真正的自我改进 Agent 需要具备内在元认知学习能力（而非外在的、人为设计的固定循环），该框架包含三个组件：元认知知识、元认知规划和元认知评估，并分析了现有自改进 Agent 的不足和实现内在元认知的路径。

**[Validating Mechanistic Interpretations: An Axiomatic Approach](validating_mechanistic_interpretations_an_axiomatic_approach.md)**

:   借鉴程序分析中抽象解释的思想，提出一组公理化框架来形式化定义和验证神经网络的机制解释（mechanistic interpretation），并在 2-SAT 求解器和模加法两个案例中验证了该框架的有效性。

**[VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters](visionts_visual_masked_autoencoders_are_free-lunch_zero-shot_time_series_forecas.md)**

:   将时间序列重构为图像，利用 ImageNet 预训练的 MAE（Masked Autoencoder）在**零样本**设置下进行时序预测，无需任何时序数据训练即可匹敌甚至超越专门的时序基础模型。

**[What Limits Virtual Agent Application? OmniBench: A Scalable Multi-Dimensional Benchmark for Essential Virtual Agent Capabilities](what_limits_virtual_agent_application_omnibench_a_scalable_multi-dimensional_ben.md)**

:   本文提出 OmniBench——一个基于图结构的可扩展虚拟 Agent 基准，通过自动化流水线合成可控复杂度的任务，配合 OmniEval 多维评估框架，在 20 个应用场景中生成 36K 个任务，系统揭示了虚拟 Agent 在不同能力维度上的短板。
