---
title: >-
  NeurIPS2025 LLM 评测方向79篇论文解读
description: >-
  79篇NeurIPS2025的 LLM 评测方向论文解读，涵盖 LLM、对齐/RLHF、对抗鲁棒、域适应、少样本学习、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "NeurIPS2025"
  - "LLM 评测"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "对齐/RLHF"
  - "对抗鲁棒"
  - "域适应"
  - "少样本学习"
  - "推理"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📊 LLM 评测

**🧠 NeurIPS2025** · **79** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (8)](../../ICML2026/llm_evaluation/index.md) · [💬 ACL2026 (79)](../../ACL2026/llm_evaluation/index.md) · [📷 CVPR2026 (25)](../../CVPR2026/llm_evaluation/index.md) · [🔬 ICLR2026 (53)](../../ICLR2026/llm_evaluation/index.md) · [🤖 AAAI2026 (39)](../../AAAI2026/llm_evaluation/index.md) · [📹 ICCV2025 (27)](../../ICCV2025/llm_evaluation/index.md)

🔥 **高频主题：** LLM ×12 · 对齐/RLHF ×7 · 对抗鲁棒 ×4 · 域适应 ×2 · 少样本学习 ×2

**[A High-Dimensional Statistical Method for Optimizing Transfer Quantities in Multi-Source Transfer Learning](a_highdimensional_statistical_method_for_optimizing_transfer.md)**

:   提出基于K-L散度和高维统计分析的理论框架，用于确定多源迁移学习中每个源任务的最优样本迁移数量，避免"用所有源数据"带来的负迁移问题，在DomainNet和Office-Home上超过SOTA 1.0-1.5%的同时减少47.85%的样本使用量和35.19%的训练时间。

**[A Standardized Benchmark for Multilabel Antimicrobial Peptide Classification](a_standardized_benchmark_for_multilabel_antimicrobial_peptide_classification.md)**

:   提出 **ESCAPE**——首个标准化的多标签抗菌肽分类基准，整合 27 个公开数据库共 80,000+ 肽段，并设计基于双分支 Transformer + 双向交叉注意力的 Baseline 模型，在 mAP 上相对第二名提升 2.56%。

**[A Unified Framework for Provably Efficient Algorithms to Estimate Shapley Values](a_unified_framework_for_provably_efficient_algorithms_to_estimate_shapley_values.md)**

:   提出统一框架将 KernelSHAP、LeverageSHAP 等 Shapley 值估计器纳入随机草图（sketching）视角，首次为 KernelSHAP 提供非渐近理论保证，并通过算法改进（Poisson 近似等）将方法扩展到 CIFAR-10 等高维数据集。

**[AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners](adastar_adaptive_data_sampling_for_training_self-taught_reasoners.md)**

:   发现 STaR（自我教学推理器）的随机数据采样导致观测训练频率严重不平衡（简单题过度训练、难题训练不足），提出 AdaSTaR——通过自适应多样性采样（优先欠训练样本）和自适应课程采样（根据模型强度调节难度），在 6 个基准上全部取得最高准确率同时减少 58.6% 训练 FLOPs。

**[Aggregation Hides OOD Generalization Failures from Spurious Correlations](aggregation_hides_out-of-distribution_generalization_failures_from_spurious_corr.md)**

:   揭示 OOD 泛化 benchmark 中"聚合掩蔽"现象——aggregate 评估显示 accuracy-on-the-line（ID 与 OOD 准确率正相关），但 OODSelect 方法可从同一 OOD 数据中找到大规模语义连贯子集（最高达 75%），这些子集上 ID 越高 OOD 反而越低（Pearson R 低至 -0.92），证明虚假相关的危害被聚合评估系统性隐藏。

**[Asymmetric Duos: Sidekicks Improve Uncertainty](asymmetric_duos_sidekicks_improve_uncertainty.md)**

:   Asymmetric Duos（AD）将一个大模型与一个小"sidekick"配对——通过温度加权的 logit 平均融合两者预测，在仅增加 10-20% FLOPs 的条件下达到接近 5× 深度集成的不确定性估计质量，RN50 AD（5% FLOPs 额外开销）在 AUROC/AURC/SAC@98 上接近 m=5 深度集成（400% 额外 FLOPs）。

**[Bayesian Evaluation of Large Language Model Behavior](bayesian_evaluation_of_large_language_model_behavior.md)**

:   提出基于 Beta-Binomial 贝叶斯模型的 LLM 行为评估框架，通过对每个 prompt 的随机生成结果建模 $\theta_m$ 后验分布，量化评估指标的统计不确定性，并引入 Thompson sampling 等序贯采样策略以更少的 API 调用获得更窄的置信区间。

**[Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks](belief-calibrated_multi-agent_consensus_seeking_for_complex_nlp_tasks.md)**

:   提出 Belief-Calibrated Consensus Seeking (BCCS) 框架，通过引入信念（belief）校准的共识判断、冲突感知的协作者分配和领导者选择三个模块，让多智能体系统在复杂NLP任务上达成更稳定的共识，在 MATH 和 MMLU 上的困难任务分别提升 2.23% 和 3.95%。

**[Benchmarking is Broken — Don't Let AI be its Own Judge](benchmarking_is_broken_--_dont_let_ai_be_its_own_judge.md)**

:   系统性批评当前 AI 基准评估的根本缺陷——数据污染（MMLU 45%+ 重叠）、选择性报告、缺乏监考——并提出 PeerBench 方案：借鉴高考/GRE 的监考范式，用滚动更新的保密题库 + 同行评审质量控制 + 声誉加权评分 + 加密承诺机制构建下一代 AI 评估基础设施。

**[Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection](benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)**

:   在统一的零样本和少样本 prompt 框架下系统评估 GPT-4o、Claude-3.7 和 Grok-3-Beta 三个商用 LLM 在钓鱼 URL 检测任务上的表现，发现少样本 prompt 可显著提升所有模型性能，Grok-3-Beta 在平衡数据集上取得最佳 F1（0.9399），但不同模型在精度-召回率权衡上呈现差异化行为模式。

**[Beyond the Singular: Revealing the Value of Multiple Generations in Benchmark Evaluation](beyond_the_singular_revealing_the_value_of_multiple_generations_in_benchmark_eva.md)**

:   将LLM基准评测形式化为层级统计模型，理论证明多次随机生成（k>1）能降低benchmark分数估计方差，并引入prompt级难度指标$\mathbb{P}(\text{correct})$和数据地图用于基准质量控制。

**[Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations](beyond_the_surface_enhancing_llm-as-a-judge_alignment_with_human_via_internal_re.md)**

:   提出LAGER框架，通过聚合LLM中间层到最终层的score token logits并计算期望分数，无需微调模型即可将LLM评判与人类评分的对齐度提升最高7.5%，且不需要思维链推理步骤就能匹配或超过推理类方法。

**[Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport](bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)**

:   提出 Bispectral Optimal Transport (BOT)，将离散最优传输中的代价矩阵从原始像素距离替换为 bispectrum（群 Fourier 不变量）距离，使得传输计划在保持信号结构的同时精确消除群作用（如旋转）带来的变异，在旋转变换的 MNIST 等数据集上将类别保持准确率从 33% 提升至 84%。

**[BLINK-Twice: You See But Do You Observe? A Reasoning Benchmark on Visual Perception](blink-twice_you_see_but_do_you_observe_a_reasoning_benchmark_on_visual_perceptio.md)**

:   提出视觉中心推理 benchmark BLINK-Twice（345 张视觉挑战图 + 103 个对抗样本 + 896 个 VQA + 1725 个推理步骤标注），通过 7 类视觉错觉场景评估 MLLM "看到但未观察到"的推理能力，发现最强模型 Gemini-2.5 Pro 的 G-Acc 仅 26.9%，多轮图像观察和主动视觉交互是提升方向。

**[Can Large Language Models Master Complex Card Games?](can_large_language_models_master_complex_card_games.md)**

:   系统评估LLM在8种复杂卡牌游戏上的学习能力，发现通过高质量游戏数据的SFT，LLM可以接近强游戏AI的水平，并能同时掌握多个游戏，但通用能力会下降（可通过混入通用指令数据缓解）。

**[CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data](climb_class-imbalanced_learning_benchmark_on_tabular_data.md)**

:   提出 Climb——迄今最全面的表格数据类别不平衡学习基准，涵盖 73 个真实数据集和 29 种 CIL 算法，通过大规模实验揭示了朴素重平衡往往无效、集成方法至关重要、数据质量比不平衡本身更影响性能等实用洞察。

**[CodeAssistBench (CAB): Dataset & Benchmarking for Multi-turn Chat-Based Code Assistance](codeassistbench_cab_dataset_benchmarking_for_multi-turn_chat-based_code_assistan.md)**

:   提出 CodeAssistBench (CAB)，第一个评估多轮、项目级编程辅助的全自动 Benchmark，从 GitHub Issues 自动构建 3,286 个真实编程求助场景，涵盖 7 种语言 214 个仓库，揭示 SOTA 模型在 StackOverflow 问题上 70-83% 但在 post-cutoff 仓库上仅 7-16% 的巨大鸿沟。

**[ComPO: Preference Alignment via Comparison Oracles](compo_preference_alignment_via_comparison_oracles.md)**

:   针对DPO中噪声偏好对（preferred和dispreferred响应相似）导致的似然位移和冗长问题，提出基于比较oracle的零阶偏好对齐方法ComPO，将数据分为干净/噪声子集，用DPO处理干净数据、用ComPO提取噪声数据中的信号，在AlpacaEval 2等benchmark上持续提升LC win rate。

**[Conformal Online Learning of Deep Koopman Linear Embeddings](conformal_online_learning_of_deep_koopman_linear_embeddings.md)**

:   提出 COLoKe 框架，将 conformal prediction 重新解读为模型一致性诊断工具，仅在 Koopman 模型的预测误差超过动态校准阈值时才触发参数更新，从而实现对非线性动力系统的高效在线 Koopman 线性嵌入学习。

**[Conformal Prediction in The Loop: A Feedback-Based Uncertainty Model for Trajectory Optimization](conformal_prediction_in_the_loop_a_feedback-based_uncertainty_model_for_trajecto.md)**

:   提出 Feedback-Based Conformal Prediction (Fb-CP) 框架，将已执行轨迹的信息反馈给 CP 以动态调整预测区域大小，在缩减时域轨迹优化中同时保证覆盖率和显著提升轨迹性能。

**[ConfTuner: Training Large Language Models to Express Their Confidence Verbally](conftuner_training_large_language_models_to_express_their_confidence_verbally.md)**

:   ConfTuner 提出 tokenized Brier score 损失函数（理论证明为 proper scoring rule），仅需 2000 个样本 + 4 分钟 LoRA 微调即可让 LLM 输出校准的语言化置信度（如"我80%确定"），ECE 最大降低 60.9%，支持自我纠错和模型级联等下游应用。

**[Consistent Supervised-Unsupervised Alignment for Generalized Category Discovery](consistent_supervised-unsupervised_alignment_for_generalized_category_discovery.md)**

:   提出 NC-GCD 框架，通过预分配固定的 Equiangular Tight Frame (ETF) 原型为已知类和新类建立统一优化目标，结合语义一致性匹配器 (SCM) 稳定跨迭代伪标签分配，在 6 个 GCD 基准上显著提升新类发现精度。

**[Cost-Sensitive Freeze-thaw Bayesian Optimization for Efficient Hyperparameter Tuning](cost-sensitive_freeze-thaw_bayesian_optimization_for_efficient_hyperparameter_tu.md)**

:   CFBO 将用户定义的效用函数（成本 vs 性能的权衡）引入冻结-解冻贝叶斯优化框架，结合自适应停止准则和基于 LC mixup 的迁移学习，在多保真度 HPO 基准上实现了成本-性能最优权衡。

**[Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models](creativity_or_brute_force_using_brainteasers_as_a_window_into_the_problem-solvin.md)**

:   构建Braingle Brainteaser基准（242数学+236逻辑谜题），系统评估LLM在脑筋急转弯上的推理策略——发现模型有时能产生创造性洞察式解法，但也常在有巧妙解法可用时退回暴力穷举，且自纠错能力有限、将叙事→数学格式翻译可小幅提升性能。

**[Decoupled Entropy Minimization](decoupled_entropy_minimization.md)**

:   将经典熵最小化（EM）解耦为两个对立部分——Cluster Aggregation Driving Factor (CADF，奖励主导类别)和 Gradient Mitigation Calibrator (GMC，惩罚高置信类别)，揭示了经典 EM 的两个固有缺陷（reward collapse 和 easy-class bias），提出 AdaDEM 通过归一化奖励和边际熵校准来修复这些问题，在半监督学习、域适应、强化学习等多任务上显著提升。

**[Document Summarization with Conformal Importance Guarantees](document_summarization_with_conformal_importance_guarantees.md)**

:   首次将Conformal Prediction应用于文档摘要，通过校准句子重要性分数的阈值，为抽取式摘要提供用户可控的覆盖率($1-\alpha$)和召回率($\beta$)的严格统计保证，方法模型无关且仅需小规模校准集。

**[Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling](efficient_semantic_uncertainty_quantification_in_language_models_via_diversity-s.md)**

:   提出 diversity-steered sampling 框架：在解码时注入基于 NLI 的语义相似度惩罚来驱动生成语义多样化的样本，并用重要性加权+控制变量纠正偏差降低方差，在仅 16 个样本下即可准确估计 LLM 的语义熵（偶然不确定性）和互信息（认知不确定性）。

**[EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving](evalearn_quantifying_the_learning_capability_and_efficiency_of_llms_via_sequenti.md)**

:   提出 EvaLearn 基准，通过**序列化问题求解**范式评估 LLM 的学习能力和学习效率，揭示静态能力强的模型不一定具备更强的学习潜力。

**[Exploiting Task Relationships in Continual Learning via Transferability-Aware Task Embeddings](exploiting_task_relationships_in_continual_learning_via_transferability-aware_ta.md)**

:   提出基于 H-score 可迁移性度量的任务嵌入（H-embedding），并将其嵌入超网络框架，通过在嵌入空间中显式建模任务间关系来指导持续学习中的参数生成，在 rehearsal-free 设定下取得 SOTA 最终准确率。

**[Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training](exploiting_vocabulary_frequency_imbalance_in_language_model_pre-training.md)**

:   通过控制实验揭示大词表提升语言模型性能的根本机制：**扩大词表降低分词文本的 Kolmogorov 复杂度，利用词频不平衡让高频词损失大幅下降，驱动全局交叉熵下降和下游任务提升**。

**[Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)**

:   将选择性SSM（Mamba）展开为注意力形式，利用覆盖数技术推导出受连续时间状态矩阵谱横断面$s_{\mathbf{A}}$控制的泛化界——$s_{\mathbf{A}}<0$时泛化界与序列长度无关，$s_{\mathbf{A}}\geq0$时指数增长，并证明这种依赖不可消除。

**[Heterogeneous Adversarial Play in Interactive Environments](heterogeneous_adversarial_play_in_interactive_environments.md)**

:   提出 **HAP（Heterogeneous Adversarial Play）**，将教师-学生交互形式化为极小极大博弈：教师网络自动生成针对学生弱点的挑战任务，学生策略不断适应进化，形成无需手工设计的自适应课程——在多任务 RL 环境中超越 SOTA 基线，且生成的课程对人类学习者同样有效。

**[HouseLayout3D: A Benchmark and Training-Free Baseline for 3D Layout Estimation in the Wild](houselayout3d_a_benchmark_and_training-free_baseline_for_3d_layout_estimation_in.md)**

:   提出 HouseLayout3D——首个面向大规模多层建筑的真实世界 3D layout 估计基准，以及 MultiFloor3D——一个无需训练的基线方法，通过组合现代 3D 重建和分割模型在多层建筑 layout 估计上超越现有深度学习方法。

**[HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization](hybridnorm_towards_stable_and_efficient_transformer_training_via_hybrid_normaliz.md)**

:   提出 HybridNorm 混合归一化策略——注意力模块用 QKV 归一化解耦梯度、FFN 用 Post-Norm 增强正则化，在 550M-7B 规模上同时获得 Pre-Norm 的训练稳定性和 Post-Norm 的泛化性能，7B 模型下游任务平均提升 2.45%。

**[Hyperbolic Fine-Tuning for Large Language Models](hyperbolic_fine-tuning_for_large_language_models.md)**

:   发现 LLM token 嵌入具有幂律分布和树状双曲结构，据此提出 HypLoRA——在 Lorentz 双曲流形上直接执行低秩适配（避免切空间映射的相消效应），在算术推理和常识推理任务上相比标准 LoRA 取得显著提升（如 Qwen2.5-7B 上 M.AVG +7.5%）。

**[Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion](incomplete_multi-view_clustering_via_hierarchical_semantic_alignment_and_coopera.md)**

:   提出HSACC框架通过双层语义空间设计（低层互信息一致性+高层自适应加权融合）和协同优化的隐式缺失视图恢复，在五个基准数据集上显著超越现有不完整多视图聚类方法。

**[Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities](ineq-comp_benchmarking_human-intuitive_compositional_reasoning_in_automated_theo.md)**

:   提出 Ineq-Comp 基准，通过对简单不等式种子问题施加人类直觉可轻松处理的组合变换（变量复制、代数重写），揭示当前 LLM 形式化定理证明器在组合推理上的根本性缺陷——即使 DeepSeek-Prover-V2-7B 也有 20%+ 的性能下降。

**[Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning](keep_it_on_a_leash_controllable_pseudo-label_generation_towards_realistic_long-t.md)**

:   提出 Controllable Pseudo-label Generation (CPG) 框架，通过可控的自强化优化循环将可靠伪标签逐步纳入标注集，在已知分布上构建 Bayes-optimal 分类器，从而在未标注数据分布完全未知的 Realistic LTSSL 场景下实现最高 15.97% 的准确率提升。

**[LCDB 1.1: A Database Illustrating Learning Curves Are More Ill-Behaved Than Previously Thought](lcdb_11_a_database_illustrating_learning_curves_are_more_ill-behaved_than_previo.md)**

:   构建了大规模高分辨率学习曲线数据库 LCDB 1.1，证明样本学习曲线的"病态行为"（非单调、非凸）比此前认为的普遍两倍，约 15% 的曲线显著不良，且特征缩放难以修复。

**[Learning Generalizable Shape Completion with SIM(3) Equivariance](learning_generalizable_shape_completion_with_sim3_equivariance.md)**

:   提出首个 SIM(3) 等变形状补全网络 SIMECO，通过特征规范化→相似不变几何推理→变换恢复的三阶段模块设计，在去偏评估协议下超越所有增广和等变基线，KITTI 上 MMD 降低 17%、OmniObject3D 上 CD-$\ell_1$ 降低 14%，且在更严格协议下仍优于竞争者在其偏向性设置下的表现。

**[Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)**

:   提出 DoRA（Distribution-aware optimization for Robust Alignment），通过训练分布分类器为每个样本分配校准权重，结合 KL-DRO 框架最小化最坏情况损失，以模型无关的即插即用方式提升多种对齐算法在分布偏移下的鲁棒性，在 DPO/RRHF/LIRE 等基线上一致提升性能。

**[LTD-Bench: Evaluating Large Language Models by Letting Them Draw](ltd-bench_evaluating_large_language_models_by_letting_them_draw.md)**

:   LTD-Bench 通过让 LLM 画画（生成点阵或代码绘图）来评估其空间推理能力，将抽象的评分指标转化为直观可视的输出，揭示了当前先进 LLM 在建立语言与空间概念双向映射方面的严重不足。

**[MEIcoder: Decoding Visual Stimuli from Neural Activity by Leveraging Most Exciting Inputs](meicoder_decoding_visual_stimuli_from_neural_activity_by_leveraging_most_excitin.md)**

:   提出 MEIcoder，利用神经元特异性的最激励输入(MEI)作为生物学先验、SSIM 损失和对抗训练，从初级视觉皮层(V1)的神经群体活动中实现 SOTA 级别的视觉刺激重建，尤其在小数据集和少量神经元场景下表现突出。

**[Mind the Gap: Removing the Discretization Gap in Differentiable Logic Gate Networks](mind_the_gap_removing_the_discretization_gap_in_differentiable_logic_gate_networ.md)**

:   提出 Gumbel Logic Gate Networks (Gumbel LGNs)，通过在逻辑门选择中注入 Gumbel 噪声并使用直通估计器 (ST estimator)，将可微逻辑门网络的离散化差距减少 98%，训练速度提升 4.5 倍，未使用神经元比例降为 0%。

**[Model-Behavior Alignment under Flexible Evaluation: When the Best-Fitting Model Isn't the Right One](model-behavior_alignment_under_flexible_evaluation_when_the_best-fitting_model_i.md)**

:   通过大规模模型恢复实验证明，即使使用 450 万行为数据，基于线性探测（linear probing）的灵活评估方法在 20 个视觉模型中的模型恢复准确率仍低于 80%，揭示了预测准确性与模型可辨识性之间的根本性权衡，质疑了当前"最佳拟合即最优模型"的研究范式。

**[Model Context Protocol for Vision Systems: Audit, Security, and Protocol Extensions](model_context_protocol_for_vision_systems_audit_security_and_protocol_extensions.md)**

:   首个对MCP在视觉系统中部署的协议级审计研究，分析91个公开MCP服务器发现78%存在schema不一致、89%缺乏运行时验证，并提出语义schema、可视化记忆、运行时验证器等协议扩展方案。

**[MVSMamba: Multi-View Stereo with State Space Model](mvsmamba_multi-view_stereo_with_state_space_model.md)**

:   提出MVSMamba，首个基于Mamba架构的多视图立体(MVS)网络，通过参考视角中心的动态扫描策略实现高效的视内和视间全局全方向特征聚合，在DTU和Tanks-and-Temples上以最优效率达到SOTA性能。

**[Normal-Abnormal Guided Generalist Anomaly Detection](normal-abnormal_guided_generalist_anomaly_detection.md)**

:   NAGL 框架首次在通用异常检测（GAD）中引入正常+异常混合参考样本，通过残差挖掘（RM）和异常特征学习（AFL）两个注意力模块，在残差空间中学习可迁移的异常模式，仅用 1 个异常样本即可在跨域场景中大幅超越仅使用正常参考的方法。

**[Not All Splits Are Equal: Rethinking Attribute Generalization Across Unrelated Categories](not_all_splits_are_equal_rethinking_attribute_generalization_across_unrelated_ca.md)**

:   本文首次系统评估了属性预测任务中训练/测试划分策略对泛化性能的影响,提出了基于 LLM 语义分组、嵌入相似度、嵌入聚类和超类标签的四种渐进式难度划分方案,发现无监督聚类划分在不依赖标注的情况下实现了与真值超类划分相当的去泄漏效果,同时保留了更好的预测性能。

**[On Evaluating LLM Alignment by Evaluating LLMs as Judges](on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)**

:   本文系统研究了 LLM 的生成能力与评估能力之间的一致性（GE-consistency），发现两者在强偏好预言机下高度相关（Spearman ρ=0.96），据此提出 AlignEval 基准——通过评估 LLM 作为评判者的能力来衡量其对齐水平，无需 LLM-as-Judge 直接评估模型输出，与 AlpacaEval/Arena-Hard 相当甚至更优。

**[On the Entropy Calibration of Language Models](on_the_entropy_calibration_of_language_models.md)**

:   系统研究语言模型的熵校准问题（生成文本的熵是否匹配在人类文本上的 log loss），发现由于数据分布的幂律特性（$\alpha \approx 1$），误差积累随模型规模的改善极为缓慢（scaling exponent $\approx -0.05$），并从理论上证明了在多项式时间内可以在不牺牲多样性的前提下校准熵。

**[Open-Insect: Benchmarking Open-Set Recognition of Novel Species in Biodiversity Monitoring](open-insect_benchmarking_open-set_recognition_of_novel_species_in_biodiversity_m.md)**

:   提出Open-Insect——首个面向昆虫物种发现的大规模细粒度开放集识别基准数据集，涵盖三个地理区域和三类开放集划分，系统评测38种OSR算法，发现简单的后验方法（如MSP）在细粒度场景中仍是强基线，同时验证了领域相关辅助数据对提升OSR性能的关键作用。

**[OptiTree: Hierarchical Thoughts Generation with Tree Search for LLM Optimization Modeling](optitree_hierarchical_thoughts_generation_with_tree_search_for_llm_optimization_.md)**

:   提出 OptiTree，通过构建建模树（modeling tree）组织运筹优化问题的层次化分类与建模思维，利用树搜索将复杂问题自适应分解为更简单的子问题序列，显著提升 LLM 在优化建模任务上的准确率（在多个困难基准上提升超过 10%）。

**[PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation](parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)**

:   本文提出 PARROT，一个面向跨系统 SQL 翻译（SQL-to-SQL）的实际且真实的基准测试，包含来自 38 个开源基准和真实业务场景的 598 个核心翻译对（扩展到 28,003 对），覆盖 22 种生产级数据库系统，揭示当前最强 LLM 的平均准确率低于 38.53%。

**[PaTH Attention: Position Encoding via Accumulating Householder Transformations](path_attention_position_encoding_via_accumulating_householder_transformations.md)**

:   提出 PaTH（Position encoding via accumulating Householder Transformations），一种数据依赖的乘法位置编码方案，通过累积 Householder 变换替代 RoPE 的静态旋转矩阵，在理论表达力和实际语言建模性能上均优于 RoPE。

**[PFΔ: A Benchmark Dataset for Power Flow under Load, Generation, and Topology Variations](pfδ_a_benchmark_dataset_for_power_flow_under_load_generation_and_topology_variat.md)**

:   PFΔ 是首个同时涵盖负荷、发电机出力和拓扑变化的电力潮流基准数据集，包含 859,800 个求解实例、六种电网规模和接近不可行的极端工况，并提出标准化评估任务来系统评测 ML 方法在多种运行条件下的表现。

**[Put CASH on Bandits: A Max K-Armed Problem for Automated Machine Learning](put_cash_on_bandits_a_max_k-armed_problem_for_automated_machine_learning.md)**

:   针对 AutoML 中的联合算法选择和超参数优化（CASH）问题，通过数据驱动分析揭示了 HPO 奖励分布的有界左偏特性，提出了专门适配该特性的极端 Bandit 算法 MaxUCB，在理论和实验上均优于现有方法。

**[RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases](rdb2g-bench_a_comprehensive_benchmark_for_automatic_graph_modeling_of_relational.md)**

:   本文提出 RDB2G-Bench——首个评估关系数据库到图建模方法的基准框架，包含 5 个真实 RDB、12 个预测任务和约 5 万个预计算的图模型-性能对，并对 10 种自动图建模方法进行了系统比较。

**[Reliably Detecting Model Failures in Deployment Without Labels](reliably_detecting_model_failures_in_deployment_without_labels.md)**

:   提出D3M(Disagreement-Driven Deterioration Monitoring)，一种基于变分贝叶斯后验采样的三阶段模型监控算法，在无标签、无训练数据的部署场景下可靠检测模型性能退化，同时对非退化性偏移保持低误报率。

**[Rethinking Evaluation of Infrared Small Target Detection](rethinking_evaluation_of_infrared_small_target_detection.md)**

:   系统性地揭示了红外小目标检测（IRSTD）现有评估协议的三大局限，提出包含混合层级指标hIoU、系统化错误分析方法和跨数据集评估设置的层次化分析框架。

**[Rethinking Losses for Diffusion Bridge Samplers](rethinking_losses_for_diffusion_bridge_samplers.md)**

:   本文揭示了扩散桥采样器中流行的 Log Variance (LV) 损失存在的理论缺陷——不满足数据处理不等式且梯度与 rKL 不等价——并提出用 log-derivative trick 计算 rKL 梯度（rKL-LD），在多个基准上一致性超越 LV 损失，且训练更加稳定、对超参数不敏感。

**[RGB-to-Polarization Estimation: A New Task and Benchmark Study](rgb-to-polarization_estimation_a_new_task_and_benchmark_study.md)**

:   本文首次定义从标准RGB图像估计偏振分量（S₁/S₂/S₃）的新任务，构建涵盖修复型与生成型方法的首个系统基准，发现预训练MAE在像素精度上综合最优（PSNR 24.74），修复型方法整体显著优于扩散生成型方法，且预训练权重迁移是关键优势。

**[Risk Management for Mitigating Benchmark Failure Modes: BenchRisk](risk_management_for_mitigating_benchmark_failure_modes_benchrisk.md)**

:   基于NIST风险管理流程，系统分析了26个LLM基准测试中的57种失败模式，提出196种缓解策略，并构建了BenchRisk元评估框架对基准测试本身的可靠性进行评分。

**[Robust Hallucination Detection in LLMs via Adaptive Token Selection](robust_hallucination_detection_in_llms_via_adaptive_token_selection.md)**

:   HaMI 将幻觉检测建模为多示例学习（MIL）问题，将生成序列视为 token 实例的"bag"，通过联合优化 token 选择和幻觉检测来自适应地定位最具指示性的 token，在四个 QA 基准上以 AUROC 大幅超越所有现有方法（最高提升 11.9%）。

**[scMRDR: A Scalable and Flexible Framework for Unpaired Single-Cell Multi-Omics Data Integration](scmrdr_a_scalable_and_flexible_framework_for_unpaired_single-cell_multi-omics_da.md)**

:   提出scMRDR框架，基于β-VAE将单细胞多组学数据的潜在表征解耦为模态共享和模态特异成分，通过等距正则化、对抗训练和掩码重建损失实现非配对多组学数据的可扩展整合。

**[Semi-Supervised Regression with Heteroscedastic Pseudo-Labels](semi-supervised_regression_with_heteroscedastic_pseudo-labels.md)**

:   提出基于异方差建模的不确定性感知伪标签框架，通过双层优化动态校准每个伪标签的不确定性，避免错误伪标签对回归模型的负面影响，在多个 SSR 基准上取得 SOTA。

**[Small Language Models as Compiler Experts: Auto-Parallelization for Heterogeneous Systems](small_language_models_as_compiler_experts_auto-parallelization_for_heterogeneous.md)**

:   系统评估了三个小于 1.5B 参数的语言模型（gemma3、llama3.2、qwen2.5）在编译器自动并行化任务上的能力，使用六种推理策略在 11 个真实世界内核上实现平均 6.81x 加速、峰值 43.25x，证明小模型可作为强大的编译器优化推理引擎。

**[SPROD: Spurious-Aware Prototype Refinement for Reliable Out-of-Distribution Detection](spurious-aware_prototype_refinement_for_reliable_out-of-distribution_detection.md)**

:   SPROD 是一种后置（post-hoc）OOD 检测方法，专门应对训练数据中的虚假相关——通过将每个类别的原型细分为"正确分类"和"误分类"子组（后者共享虚假特征），配合 K-means 式精炼和距离式（生成式）评分，在 5 个虚假相关 OOD 基准上平均 AUROC 85.1%（+4.8% vs 次优 KNN），FPR@95 49.0%（-9.3% vs 次优）。

**[Test-Time Adaptation by Causal Trimming](test-time_adaptation_by_causal_trimming.md)**

:   提出 TACT 方法，通过数据增强 + PCA 识别表征空间中的非因果方向，并在测试时将表征和类原型沿该方向的投影移除，从而减少模型对非因果特征的依赖，显著提升分布偏移下的预测性能。

**[The Geometry of Cortical Computation: Manifold Disentanglement and Predictive Dynamics in VCNet](the_geometry_of_cortical_computation_manifold_disentanglement_and_predictive_dyn.md)**

:   本文提出VCNet——一种模拟灵长类视觉皮层宏观组织的神经网络架构，用几何和动力系统语言重新诠释双流分离（流形解缠）和预测编码（测地线精炼），以0.04MB的极小体积在Spots-10上达到92.1%（比DenseNet蒸馏版高10%），在光场分类上以3.52MB达到74.4%（超MobileNetV2 2.3%）。

**[Thought Communication in Multiagent Collaboration](thought_communication_in_multiagent_collaboration.md)**

:   提出 ThoughtComm 框架，将多智能体通信形式化为隐变量生成模型，证明了在非参数条件下共享思想和私有思想均可辨识，并通过稀疏正则化自编码器提取潜在思想、经前缀注入回馈给每个智能体，在数学推理任务上相比当前 SOTA 的 Multiagent Finetuning 平均提升 19.06%。

**[Tight Lower Bounds and Improved Convergence in Performative Prediction](tight_lower_bounds_and_improved_convergence_in_performative_prediction.md)**

:   在 performative prediction 框架下，首次证明了 Repeated Risk Minimization (RRM) 收敛率的紧致性，并提出 Affine Risk Minimizers (ARM) 算法类，通过利用历史训练快照的数据实现更广泛问题类上的收敛。

**[Time Travel is Cheating: Going Live with DeepFund for Real-Time Fund Investment Benchmarking](time_travel_is_cheating_going_live_with_deepfund_for_real-time_fund_investment_b.md)**

:   提出 DeepFund——首个实时基金投资 benchmark 工具，通过多智能体架构（Financial Planner + Analyst Team + Portfolio Manager）连接实时股市数据，避免传统回测中 LLM "时间旅行"导致的信息泄露问题。在 24 个交易日的实盘测试中，9 个旗舰 LLM 只有 Grok 3 实现盈利，揭示了当前 LLM 在主动基金管理中的重大局限。

**[Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era](towards_implicit_aggregation_robust_image_representation_for_place_recognition_i.md)**

:   提出 ImAge（Implicit Aggregation），在 Transformer 骨干网络的特定层插入可学习聚合 Token，利用内在自注意力机制将 patch 特征隐式聚合为全局描述符，完全消除了额外聚合器的需要。以最小的描述符维度（6144）和最快推理速度，在多个 VPR 数据集上超越 SALAD、BoQ 等 SOTA，并在 MSLS Challenge 排行榜排名第 1。

**[Turbocharging Gaussian Process Inference with Approximate Sketch-and-Project](turbocharging_gaussian_process_inference_with_approximate_sketch-and-project.md)**

:   提出 ADASAP 算法，通过近似子空间预条件、分布式计算和 Nesterov 加速，将 sketch-and-project 方法扩展到大规模 GP 推断，首次将精确 GP 推断扩展到 $>3\times10^8$ 样本规模，同时在理论上证明了 SAP 方法的 condition number-free 收敛性。

**[Unlocking Transfer Learning for Open-World Few-Shot Recognition](unlocking_transfer_learning_for_open-world_few-shot_recognition.md)**

:   提出两阶段方法，通过开集感知元学习 + 开集无关迁移学习，首次将迁移学习范式成功应用于少样本开集识别 (FSOSR)，在 miniImageNet 和 tieredImageNet 上达到 SOTA。

**[What Does It Take to Build a Performant Selective Classifier?](what_does_it_take_to_build_a_performant_selective_classifier.md)**

:   首次对选择性分类的性能差距（selective classification gap）进行有限样本分解，将差距归因于五个源头——贝叶斯噪声、逼近误差、排序误差、统计噪声和实现偏差，并证明单调校准方法对缩小差距效果有限。

**[Words That Unite The World: A Unified Framework for Deciphering Central Bank Communications Globally](words_that_unite_the_world_a_unified_framework_for_deciphering_central_bank_comm.md)**

:   本文构建了迄今最全面的央行货币政策语料库 WCB（38万+句子、25家央行、跨28年），定义三个NLP任务（立场检测、时间分类、不确定性估计），通过15,075次基准实验发现聚合多银行数据训练的模型显著优于单银行训练，证实了"整体大于部分之和"的原则。

**[Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator](your_pre-trained_llm_is_secretly_an_unsupervised_confidence_calibrator.md)**

:   发现 LLM 后训练（SFT/RLHF/DPO）破坏了预训练模型的置信度校准，提出 DACA 方法利用预训练模型的良好校准性，仅在预测一致样本上对齐置信度，实现无标签的后训练模型校准，ECE 最高改善 15.08%。
