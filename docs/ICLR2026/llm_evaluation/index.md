---
title: >-
  ICLR2026 LLM评测方向 46篇论文解读
description: >-
  46篇ICLR2026 LLM评测方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📊 LLM评测

**🔬 ICLR2026** · **46** 篇论文解读

**[Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms](accessible_realistic_and_fair_evaluation_of_positive-unlabeled_learning_algorith.md)**

:   提出首个 PU 学习统一基准，系统解决两个关键问题：(1) 用代理准确率和代理 AUC 实现无负样本的模型选择；(2) 发现并通过将正样本并入无标签集的简单校准方法解决单样本设置下的内部标签偏移问题，使双样本算法在单样本评估中得到公平比较。

**[AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)**

:   构建首个面向麻醉学推理的综合数据集套件AnesSuite，包含评测基准AnesBench（7972道三级认知难度双语选择题）和三组训练数据集（AnesCorpus/AnesQA/AnesR1），基于此训练的Morpheus模型通过SFT+GRPO让7B模型追平14B基线，同时揭示了当前最强LLM在复杂临床推理（System 2）上的显著瓶颈。

**[ASIDE: Architectural Separation of Instructions and Data in Language Models](aside_architectural_separation_of_instructions_and_data_in_language_models.md)**

:   提出 ASIDE，一种在 token embedding 层面通过正交旋转区分指令和数据的架构级改造，仅需修改前向传播并在标准指令微调数据上训练，即可显著提升指令-数据分离度和 prompt injection 鲁棒性，无需任何安全专项训练。

**[AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite](astabench_benchmarking_ai_agents.md)**

:   AI2 团队针对现有科研 Agent 基准的 5 大方法学缺陷，构建了首个覆盖科学研究全流程的 Agent 评估套件 AstaBench，包含 4 大类 11 个子基准共 2400+ 问题，配备基于 Semantic Scholar 的生产级可控搜索工具和 9 类科研优化 Asta Agent 基线，对 57 个 Agent（22 类）进行了迄今最大规模的系统评估，发现尽管在文献检索等单项任务上取得了进展，AI 在端到端科学研究辅助方面仍远未达标。

**[Benchmarking Overton Pluralism in LLMs](benchmarking_overton_pluralism_in_llms.md)**

:   提出 OvertonBench 框架，通过大规模人类研究（1208名美国代表性参与者、60个主观问题、8个LLM）将 Overton 多元主义形式化为集合覆盖度指标 OvertonScore，发现当前所有模型得分仅 0.35–0.41（理论上限为 1.0），并构建了与人类判断高度相关（ρ=0.88）的自动化评测工具。

**[Breaking the Correlation Plateau: On the Optimization and Capacity Limits of Attention-Based Regressors](breaking_the_correlation_plateau_on_the_optimization_and_capacity_limits_of_atte.md)**

:   本文首次从理论上分析了注意力回归模型在联合 MSE+PCC 训练时出现的"PCC平台期"现象——发现其根源在于 MSE 优化与 PCC 梯度之间的冲突以及 softmax 凸聚合的表达力上界——并提出 ECA（Extrapolative Correlation Attention）框架，通过缩放残差聚合、色散感知温度 softmax 和色散归一化 PCC 损失三个组件突破该限制。

**[Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation and Beyond](can_you_hear_me_now_a_benchmark_for_long-range_graph_propagation_and_beyond.md)**

:   本文提出 ECHO 基准，包含 3 个合成任务和 2 个基于密度泛函理论（DFT）的真实化学任务，要求图神经网络在 17–40 跳范围内有效传播信息，系统评估了 11 种 GNN 架构的长程传播能力。

**[Conformal Prediction Adaptive to Unknown Subpopulation Shifts](conformal_prediction_adaptive_to_unknown_subpopulation_shifts.md)**

:   针对子群体偏移（subpopulation shift）下标准 conformal prediction 失效的问题，提出三种自适应算法：利用学习的 domain classifier 加权校准数据（Algorithm 1/2）或利用嵌入相似度加权（Algorithm 3），在不完美甚至无 domain 标签的情况下仍能保证覆盖率，并应用于视觉分类和 LLM 幻觉检测。

**[CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](counselbench_llm_mental_health_qa.md)**

:   联合100名持证心理健康专家构建CounselBench双组件基准——CounselBench-EVAL（2,000条六维度专家评估）和CounselBench-Adv（120个对抗性问题+1,080条响应标注），系统性揭示LLM在心理健康开放式问答中表面得分高但存在过度泛化、擅自医疗建议等安全隐患，同时证明LLM-as-Judge在安全关键领域严重不可靠。

**[DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science](dare-bench_evaluating_modeling_and_instruction_fidelity_of_llms_in_data_science.md)**

:   DARE-bench 是一个面向数据科学任务的大规模可验证基准，包含 6300 个 Kaggle 衍生任务，支持 ML 建模和指令遵循两类评估，提供训练集支持 SFT 和 RL——SFT 将 Qwen3-32B 提升 1.83×，RL 将 Qwen3-4B 提升 8× 以上。

**[Disentangling Shared and Private Neural Dynamics with SPIRE: A Latent Modeling Framework for Deep Brain Stimulation](disentangling_shared_and_private_neural_dynamics_with_spire_a_latent_modeling_fr.md)**

:   提出 SPIRE（Shared–Private Inter-Regional Encoder），一种非线性双潜空间自编码器框架，通过跨区域对齐与正交解缠损失将多脑区颅内记录分解为共享与专属子空间，仅在基线数据训练即可检测 DBS 刺激引发的频率依赖性网络重组。

**[Do We Really Need Permutations? Impact of Model Width on Linear Mode Connectivity](do_we_really_need_permutations_impact_of_model_width_on_linear_mode_connectivity.md)**

:   实证表明无需参数置换，仅靠增加模型宽度即可实现独立训练模型间的线性模式连通性（LMC），并提出"逐层指数加权连通性"（LEWC）解释这一现象的机理。

**[Enabling Fine-Grained Operating Points for Black-Box LLMs](enabling_fine-grained_operating_points_for_black-box_llms.md)**

:   发现黑盒 LLM 的语言化概率仅输出 16-23 个唯一值（低基数问题），导致 PR/ROC 曲线粗糙无法精细调优；通过注入参数化噪声和可选的 MLP 校正，将唯一值从 16 个提升到 20,000+，在仅需 1-2 次 API 调用的条件下达到 20 次采样的性能。

**[How Reliable is Language Model Micro-Benchmarking?](how_reliable_is_language_model_micro-benchmarking.md)**

:   提出 Minimum Detectable Ability Difference (MDAD) 元评估指标，系统揭示了 micro-benchmark 在极小规模下无法可靠区分性能差距小的模型对，且当样本量达到 ~250 时随机采样与精心设计的 micro-benchmark 方法表现相当。

**[Improving Set Function Approximation with Quasi-Arithmetic Neural Networks](improving_set_function_approximation_with_quasi-arithmetic_neural_networks.md)**

:   提出QUANN（准算术神经网络），用可逆神经网络实现可学习的Kolmogorov均值作为池化操作，首次实现机器学习版本的广义中心趋势度量，QUANN是均值可分解集合函数的通用近似器，且学到的嵌入跨任务迁移性更强。

**[In-Context Learning of Temporal Point Processes with Foundation Inference Models](in-context_learning_of_temporal_point_processes_with_foundation_inference_models.md)**

:   提出 FIM-PP——首个面向标记时间点过程（MTPP）的基础推断模型，在 72K 合成点过程（1440 万事件）上预训练 Transformer 来上下文推断条件强度函数，零样本即可匹配专用模型数小时训练的性能，微调几分钟后在四个真实数据集的多事件预测上全面刷新 SOTA。

**[LCA: Local Classifier Alignment for Continual Learning](lca_local_classifier_alignment_for_continual_learning.md)**

:   提出 Local Classifier Alignment (LCA) 损失函数，通过在类原型高斯分布的局部区域内同时最小化分类损失和损失灵敏度，解决持续学习中 backbone 增量合并后分类器不匹配的问题，配合增量 PEFT 合并策略 (IM)，在 7 个基准数据集上达到整体 85.6% 的平均精度，大幅超越 SOTA。

**[Measuring Uncertainty Calibration](measuring_uncertainty_calibration.md)**

:   针对二分类器 $L_1$ 校准误差的有限样本估计问题，分别在有界变差和有界导数两种结构假设下，提出了首个非渐近、分布无关的可认证上界方法，其中有界导数假设通过对分类器输出施加微小扰动即可保证，实验表明在 $10^7$ 样本量下可将校准误差上界控制在约 0.02。

**[Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets](mitigating_spurious_correlation_via_distributionally_robust_learning_with_hierar.md)**

:   提出层次化DRO框架，同时捕获组间（group proportion shifts）和组内（intra-group distributional shifts）不确定性。使用W_∞距离在语义空间定义组内模糊集，在标准基准上达SOTA，且在新设计的少数群体分布偏移设置下——其他方法均失败时——仍保持强鲁棒性。

**[MOSIV: Multi-Object System Identification from Videos](mosiv_multi-object_system_identification_from_videos.md)**

:   提出MOSIV——首个从多视角视频进行多物体系统辨识的完整框架：(1) 物体感知的4D动态高斯重建每个物体的几何与运动 → (2) 高斯到连续体提升构建MPM仿真粒子 → (3) 可微MPM模拟器前向滚动+几何对齐目标(3D Chamfer + 2D轮廓)反传优化每个物体的连续材料参数($E, \nu, \mu$) → 在包含弹性/塑性/流体/沙粒四种材料的接触丰富合成基准上，PSNR 达30.51 vs OmniPhysGS 25.93，Chamfer距离降低9.4倍，建立多物体长期物理仿真新基准。

**[Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses](multi-llm_adaptive_conformal_inference_for_reliable_llm_responses.md)**

:   提出 MACI（Multi-LLM Adaptive Conformal Inference），通过**累积乘积型 conformity score** + **多 LLM 集成**的 factuality 评分 + **组条件校准**，在严格保证用户指定错误率的同时，显著提升 LLM 回复中事实性声明的保留率。

**[Noise-Aware Generalization: Robustness to In-Domain Noise and Out-of-Domain Generalization](noise-aware_generalization_robustness_to_in-domain_noise_and_out-of-domain_gener.md)**

:   首次形式化了 Noise-Aware Generalization (NAG) 问题——在标签噪声下同时追求域内鲁棒性和域外泛化能力，并提出 DL4ND 方法通过跨域比较检测噪声标签，在 7 个数据集上最高提升 12.5%。

**[Non-Clashing Teaching in Graphs: Algorithms, Complexity, and Bounds](non-clashing_teaching_in_graphs_algorithms_complexity_and_bounds.md)**

:   研究图中闭邻域概念类的非冲突教学问题，提供精确匹配的算法上下界（N-NCTD⁺ 的 $2^{\mathcal{O}(|E|)}$ 紧界）、对 treedepth/vertex cover 参数化的 FPT 算法（含首个负面标签 FPT 结果），以及平面图和单位正方形图的组合上界，全面推进了非冲突教学的计算与组合理解。

**[Optimal Transport-Induced Samples against Out-of-Distribution Overconfidence](optimal_transport-induced_samples_against_out-of-distribution_overconfidence.md)**

:   利用半离散最优传输（OT）的几何奇异边界定位语义模糊区域，在其附近生成代理OOD样本（OTIS），训练时通过置信度抑制损失迫使模型在结构性不确定区域给出均匀预测，从而系统性地缓解DNN的OOD过度自信问题。

**[PlanetAlign: A Comprehensive Python Library for Benchmarking Network Alignment](planetalign_a_comprehensive_python_library_for_benchmarking_network_alignment.md)**

:   提出 PlanetAlign，一个集成 18 个跨 6 个领域的数据集、14 种覆盖三大类别（一致性、嵌入、最优传输）方法和标准化评估流程的 PyTorch 网络对齐基准库，通过大规模系统实验揭示了 OT 类方法（PARROT/JOENA）在有效性上的全面领先以及各类方法在可扩展性和鲁棒性上的差异化表现。

**[Predicting LLM Reasoning Performance with Small Proxy Model](predicting_llm_reasoning_performance_with_small_proxy_model.md)**

:   提出 rBridge，通过使用 frontier 模型的推理 trace 作为 gold label 并按 token 级任务对齐加权 NLL，使 ≤1B 的小模型能有效预测 13B-32B 大模型的推理性能，在数据集排名任务中实现 100× 以上的计算节省。

**[Preference Leakage: A Contamination Problem in LLM-as-a-judge](preference_leakage_a_contamination_problem_in_llm-as-a-judge.md)**

:   首次定义并系统研究 LLM-as-a-Judge 中的 **偏好泄漏 (Preference Leakage)** 问题——当合成数据生成器 $M_G$ 与评估器 $M_J$ 存在关联（同模型/继承/同家族）时，评委会对"相关学生模型"产生系统性偏好，同模型场景下 PLS 高达 28.7%（Arena-Hard），且该偏差比自中心偏差更隐蔽、更难检测。

**[Prompt and Parameter Co-Optimization for Large Language Models](prompt_and_parameter_co-optimization_for_large_language_models.md)**

:   提出 MetaTuner 框架，通过共享 meta encoder 同时生成 prompt 和 LoRA 参数，将离散 prompt 优化与连续参数微调统一为端到端可优化的联合框架，在数学推理和问答任务上大幅超越单独优化的方法。

**[RankLLM: Weighted Ranking of LLMs by Quantifying Question Difficulty](rankllm_weighted_ranking_of_llms_by_quantifying_question_difficulty.md)**

:   提出 RankLLM，一个基于有向二部图双向分数传播的非参数化框架，联合估计题目难度和模型能力，实现难度感知的 LLM 排名，与人类判断达到 90% 一致性。

**[Revisiting the Past: Data Unlearning with Model State History](revisiting_the_past_data_unlearning_with_model_state_history.md)**

:   提出 MSA（Model State Arithmetic）算法，利用训练中间检查点构造"遗忘向量"，通过参数空间算术运算移除特定数据对模型的影响，在 TOFU 和 RESTOR 基准上一致优于 NPO、RMU、GradDiff 等现有遗忘方法，且即使不用保留集也能保持模型效用。

**[Same Content, Different Representations: A Controlled Study for Table QA](same_content_different_representations_a_controlled_study_for_t.md)**

:   首个控制变量研究：在保持表格内容完全相同的条件下变换表示形式（结构化 vs 半结构化），系统评估 NL2SQL、LLM、混合三类方法在不同表格大小/模式质量/查询复杂度下的鲁棒性，发现表示形式是影响 Table QA 性能的一阶因素。

**[SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs](simpletom_exposing_the_gap_between_explicit_tom_inference_and_implicit_tom_appli.md)**

:   SimpleToM 揭示了 LLM 在 Theory of Mind 上的关键缺陷：前沿模型能准确推断他人心理状态（显式 ToM），但在将此知识应用于行为预测和行为判断时性能急剧下降（应用 ToM），暴露了"知道什么"与"如何使用所知"之间的重大鸿沟。

**[Soft Quality-Diversity Optimization](soft_quality-diversity_optimization.md)**

:   提出 Soft QD Score 作为无需行为空间离散化的质量多样性优化新目标，并据此推导出可微分算法 SQUAD，在高维行为空间中具有更好的可扩展性，且在标准基准上与 SOTA 竞争力相当。

**[Spectral Attention Steering for Prompt Highlighting](spectral_attention_steering_for_prompt_highlighting.md)**

:   提出 SEKA/AdaSEKA，通过对 key embedding 进行谱分解学习"相关性子空间"，在注意力计算前直接编辑 key 向量来实现 prompt highlighting，无需存储完整注意力矩阵，与 FlashAttention 完全兼容，且开销极低（+0.03s/sample）。

**[Subliminal Signals in Preference Labels](subliminal_signals_in_preference_labels.md)**

:   证明偏好标签可以作为隐蔽通信通道：即使学生模型生成的是语义无关的数字序列，有偏见的裁判模型仅通过二值偏好标签就能向学生模型传递潜意识行为特征，且这种传递在迭代对齐中会增强。

**[TabStruct: Measuring Structural Fidelity of Tabular Data](tabstruct_measuring_structural_fidelity_of_tabular_data.md)**

:   提出 TabStruct 评估框架和 global utility 指标，在不需要真实因果图的情况下衡量表格数据生成器对因果结构的保真度，在 29 个数据集上系统比较 13 种生成器，发现扩散模型在全局结构保持上显著优于其他方法。

**[Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis.md)**

:   提出TED(Talk, Evaluate, Diagnose)框架，通过通用可复用的expert/non-expert persona模板实现用户感知的动态Agent评估、grading notes+LLM-as-judge+MaxProgressRate@k等新指标进行细粒度效率评估、自动化错误发现和聚类提供可操作的改进反馈，在τ²-bench和ToolSandbox上揭示新的Agent性能洞察。

**[Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)**

:   提出 APF 框架，通过 Rayleigh 商引导的异常感知预训练和粒度自适应微调，解决图异常检测中标签稀缺和同质性差异的双重挑战。

**[Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction](truthfulness_despite_weak_supervision_evaluating_and_training_llms_using_peer_pr.md)**

:   提出将博弈论中的 Peer Prediction 机制应用于 LLM 评估和训练，通过衡量参与者答案的互预测性来区分诚实与欺骗回答，无需真值标签即可实现诚实性激励，展现出惊人的"逆向缩放"特性——专家越弱反而越能抵抗强模型的欺骗。

**[UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking](uis-digger_towards_comprehensive_research_agent_systems_for_real-world_unindexed.md)**

:   识别并形式化"未索引信息检索"(UIS) 问题——搜索引擎无法直接检索的动态网页/嵌入文件/交互式内容，提出首个 UIS 基准 UIS-QA（110 题）和多 Agent 框架 UIS-Digger，以 ~30B 参数模型经 SFT+RFT 训练后达到 27.27% 准确率，超越集成 O3/GPT-4.1 的系统。

**[Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework](unpacking_human_preference_for_llms_demographically_aware_evaluation_with_the_hu.md)**

:   提出 HUMAINE 框架，通过 23,404 名人口统计学分层参与者对 28 个模型的多维度评估，揭示了人类偏好中年龄是最大分歧轴、单一排行榜掩盖关键差异的发现。

**[vCache: Verified Semantic Prompt Caching](vcache_verified_semantic_prompt_caching.md)**

:   提出 vCache——首个具有**用户定义错误率保证**的语义缓存系统，通过在线学习为每个缓存嵌入独立估计最优相似度阈值，无需预训练即可在满足正确性约束下实现最高 12.5× 缓存命中率提升和 26× 错误率降低。

**[When and Where to Reset Matters for Long-Term Test-Time Adaptation](when_and_where_to_reset_matters_for_long-term_test-time_adaptation.md)**

:   ASR提出自适应选择性重置方案，通过预测集中度 $\mathcal{C}_t$ 动态判断何时重置（避免固定周期的次优性），通过从output层向input层渐进的层选择策略判断重置哪些层（保留有价值的适应知识），配合importance-aware正则化恢复被重置的关键知识和on-the-fly适应调整，在CCC-Hard上比SOTA提升44.12%。

**[When Priors Backfire: On the Vulnerability of Unlearnable Examples to Pretraining](when_priors_backfire_on_the_vulnerability_of_unlearnable_examples_to_pretraining.md)**

:   揭示了不可学习样本 (UEs) 在预训练模型上的根本性脆弱性——预训练先验使模型能绕过扰动捷径学到真实语义，并提出 BAIT 框架通过将扰动绑定到错误标签来对抗预训练先验。

**[When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling](when_to_ensemble_identifying_token-level_points_for_stable_and_fast_llm_ensembli.md)**

:   提出 SAFE（Stable And Fast LLM Ensembling），通过 Generate-Verify-Ensemble 循环在 token 级别选择性地集成多个异构分词器 LLM，解决长序列生成中分词不匹配导致的 OOV-like 污染问题，仅在不到 1% 的 token 上集成即可提升效果，MATH500 上将 UniTE 从 59.6% 提升到 77.4%。

**[Which LLM Multi-Agent Protocol to Choose?](which_llm_multi-agent_protocol_to_choose.md)**

:   本文提出ProtocolBench基准和ProtocolRouter路由器，首次系统性比较了多Agent系统中的通信协议（A2A、ACP、ANP、Agora等）在任务成功率、延迟、消息开销和鲁棒性四个维度上的差异，并通过可学习的协议路由器实现场景自适应的协议选择，最高降低18.1%的故障恢复时间。
