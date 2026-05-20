---
title: >-
  ICML2025 LLM 评测方向50篇论文解读
description: >-
  50篇ICML2025的 LLM 评测方向论文解读，涵盖 LLM、域适应、对抗鲁棒、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "LLM 评测"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "域适应"
  - "对抗鲁棒"
  - "少样本学习"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📊 LLM 评测

**🧪 ICML2025** · **50** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (8)](../../ICML2026/llm_evaluation/index.md) · [💬 ACL2026 (79)](../../ACL2026/llm_evaluation/index.md) · [📷 CVPR2026 (25)](../../CVPR2026/llm_evaluation/index.md) · [🔬 ICLR2026 (53)](../../ICLR2026/llm_evaluation/index.md) · [🤖 AAAI2026 (39)](../../AAAI2026/llm_evaluation/index.md) · [🧠 NeurIPS2025 (79)](../../NeurIPS2025/llm_evaluation/index.md)

🔥 **高频主题：** LLM ×6 · 域适应 ×3 · 对抗鲁棒 ×3 · 少样本学习 ×2

**[AAAR-1.0: Assessing AI's Potential to Assist Research](aaar-10_assessing_ais_potential_to_assist_research.md)**

:   提出 AAAR-1.0 基准，通过公式推断、实验设计、论文弱点发现、审稿质量鉴别四个专家级任务，系统评估 LLM 辅助科研的真实能力，揭示当前模型在深度研究任务上仍有显著不足。

**[Addressing Imbalanced Domain-Incremental Learning through Dual-Balance Collaborative Experts (DCE)](addressing_imbalanced_domain-incremental_learning_through_dual-balance_collabora.md)**

:   DCE 提出频率感知专家组 + 动态专家选择器的双阶段训练框架，同时解决域增量学习中域内类别不平衡和跨域类别分布偏移两个难题，在四个 benchmark 上达到 SOTA。

**[Are LLM Belief Updates Consistent with Bayes' Theorem?](are_llm_belief_updates_consistent_with_bayes_theorem.md)**

:   本文提出贝叶斯一致性系数（BCC）来量化 LLM 的信念更新是否符合贝叶斯定理，发现更大、更强的预训练模型在给定新证据时，其信念更新与贝叶斯定理更一致。

**[Bounded Rationality for LLMs: Satisficing Alignment at Inference-Time](bounded_rationality_for_llms_satisficing_alignment_at_inference-time.md)**

:   提出 SITAlign——基于有界理性的满意决策框架，在推理时最大化主要目标（如有用性）同时确保次要目标（如无害性）满足阈值约束，通过对偶理论求解，在 GPT-4 评估上相比多目标解码 SOTA 提升 22.3% 胜率。

**[Communicating Activations Between Language Model Agents](communicating_activations_between_language_model_agents.md)**

:   提出让 LLM 智能体通过中间层激活（而非自然语言）进行通信的方法——在模型 B 的前向传播中间层注入模型 A 的激活向量，无需额外参数和数据，在多项推理基准上比自然语言通信提升 27%，计算量仅为 1/4。

**[Consistency in Language Models: Current Landscape, Challenges, and Future Directions](consistency_in_language_models_current_landscape_challenges_and_future_direction.md)**

:   系统综述了 LLM 一致性研究的全景，提出包含逻辑一致性（否定/对称/传递）、语义一致性、事实/信息一致性和非逻辑一致性（道德/规范）的分类体系，分析了 2019-2025 年间评测方法的不足，并呼吁建立标准化多语言基准和跨学科方法。

**[Cooperation of Experts: Fusing Heterogeneous Information with Large Margin](cooperation_of_experts_fusing_heterogeneous_information_with_large_margin.md)**

:   提出 Cooperation of Experts (CoE) 框架，将异构信息编码为多重网络，通过两级专家设计与大间隔置信张量优化实现专家**协作**（而非竞争），在节点分类任务上全面超越现有 MoE 和多重网络方法。

**[Correlated Errors in Large Language Models](correlated_errors_in_large_language_models.md)**

:   本文通过对超过350个LLM的大规模实证分析，发现不同LLM之间存在高度相关的错误模式——在两个模型都出错时约60%的情况下会选择同一个错误答案，且越准确的模型相关性越高；进而研究了这种相关性对LLM-as-Judge评估和招聘市场的下游影响。

**[CostFilter-AD: Enhancing Anomaly Detection through Matching Cost Filtering](costfilter-ad_enhancing_anomaly_detection_through_matching_cost_filtering.md)**

:   将立体匹配/光流估计中的**代价体滤波（cost volume filtering）**思想引入无监督异常检测（UAD），构造输入与模板之间的匹配代价体，并通过3D U-Net 加双流注意力引导进行去噪滤波，作为通用后处理插件可同时提升重建型和嵌入型 UAD 方法的性能，在 MVTec-AD 和 VisA 上取得 SOTA。

**[Cross-regularization: Adaptive Model Complexity through Validation Gradients](cross-regularization_adaptive_model_complexity_through_validation_gradients.md)**

:   提出 Cross-regularization（交叉正则化），通过验证集梯度直接优化正则化参数（权重范数、噪声尺度、增强强度），在单次训练中收敛到交叉验证最优解，消除手动调参需求。

**[DataDecide: How to Predict Best Pretraining Data with Small Experiments](datadecide_how_to_predict_best_pretraining_data_with_small_experiments.md)**

:   > 本文构建了 DataDecide——迄今最大规模的开放模型套件（25 种数据配方 × 14 种模型规模 × 3 个随机种子），系统研究如何用小规模实验预测最佳预训练数据，发现单一小规模排名（如 150M 参数）即可达到约 80% 的成对决策准确率，且连续似然代理指标仅需目标计算量 0.01% 即可让多个基准任务的预测准确率超过 80%。

**[DiLQR: Differentiable Iterative Linear Quadratic Regulator via Implicit Differentiation](dilqr_differentiable_iterative_linear_quadratic_regulator_via_implicit_different.md)**

:   本文提出 DiLQR 框架，通过在 iLQR 控制器的不动点上施加隐式微分，得到解析梯度解，将反向传播的计算复杂度从随迭代数线性增长降为 $O(1)$ 常数，实现最高 128× 加速，同时学习性能比传统神经网络策略提升 $10^6$ 倍。

**[Disentangling and Integrating Relational and Sensory Information in Transformer Architectures](disentangling_and_integrating_relational_and_sensory_information_in_transformer_.md)**

:   本文提出了 Dual Attention Transformer（DAT），通过在标准注意力机制中引入"关系注意力"头，将感知信息和关系信息解耦后并行处理再整合，在关系推理基准、数学问题求解、图像识别和语言建模等任务上均展现出显著的数据效率和参数效率提升。

**[EnIGMA: Interactive Tools Substantially Assist LM Agents in Finding Security Vulnerabilities](enigma_interactive_tools_substantially_assist_lm_agents_in_finding_security_vuln.md)**

:   EnIGMA 是一个用于自主解决 Capture The Flag (CTF) 挑战的 LM agent，通过引入新型交互式 Agent 工具（调试器和服务器连接工具），首次使 LM agent 能够运行交互式终端程序，在 4 个基准的 390 个 CTF 挑战上取得 SOTA，并发现了 "soliloquizing" 这一新的幻觉现象。

**[MultiCogEval: Evaluating LLMs Across Multi-Cognitive Levels](evaluating_llms_across_multi-cognitive_levels_from_medical_knowledge_mastery_to_.md)**

:   受 Bloom 分类法启发，提出多认知层次评估框架 MultiCogEval，从知识掌握、综合应用、情景问题解决三个层次评估 LLM 医学能力，发现所有模型性能随认知复杂度增加显著下降，且模型规模在高层次更关键。

**[Faster and Stronger: When ANN-SNN Conversion Meets Parallel Spiking Calculation](faster_and_stronger_when_ann-snn_conversion_meets_parallel_spiking_calculation.md)**

:   首次将并行脉冲计算与 ANN-SNN 转换结合，建立数学等价映射关系，在超低时间步（4步）下实现 ImageNet Top-1 72.90%，推理速度加速 19~38 倍。

**[FEDTAIL: Federated Long-Tailed Domain Generalization with Sharpness-Guided Gradient Matching](fedtail_federated_long-tailed_domain_generalization_with_sharpness-guided_gradie.md)**

:   FedTAIL 提出了一个联邦域泛化框架，通过梯度一致性正则化、逐类锐度感知最小化和曲率感知动态加权三个模块，同时解决域偏移和长尾类别不平衡的双重挑战，在多个基准上达到 SOTA。

**[Feedforward Few-shot Species Range Estimation](feedforward_few-shot_species_range_estimation.md)**

:   提出 FS-SINR（Few-shot Spatial Implicit Neural Representations），一种基于 Transformer 的前馈式少样本物种分布估计模型，无需针对新物种重新训练即可从少量观测位置（甚至零个）一次前传预测空间分布，在 IUCN 和 S&T 基准上以 2-6% 的计算时间超越 LE-SINR 等需要重新训练的方法。

**[Fleet of Agents: Coordinated Problem Solving with Large Language Models](fleet_of_agents_coordinated_problem_solving_with_large_language_models.md)**

:   提出Fleet of Agents(FoA)——用遗传粒子滤波思想协调多Agent的LLM推理：生成多个Agent各自探索→基于启发式价值函数重采样→动态分支适应发现的方案，平均比SOTA方法提升5%质量同时仅需40%的成本。

**[Fully Heteroscedastic Count Regression with Deep Double Poisson Networks](fully_heteroscedastic_count_regression_with_deep_double_poisson_networks.md)**

:   提出 Deep Double Poisson Network (DDPN)，通过输出 Double Poisson 分布的参数实现离散计数回归中的完全异方差性，支持任意高或低的预测方差，在精度、校准和 OOD 检测上全面超越现有基线。

**[Function Encoders: A Principled Approach to Transfer Learning in Hilbert Spaces](function_encoders_a_principled_approach_to_transfer_learning_in_hilbert_spaces.md)**

:   提出基于 Hilbert 空间几何视角的迁移学习分类体系（凸包插值 / 线性张成外推 / 全空间外推），并设计 Function Encoder 方法利用可学习神经网络基函数实现三种迁移，在多项基准上超越 MAML、Transformer 等方法。

**[G-Sim: Generative Simulations with Large Language Models and Gradient-Free Calibration](g-sim_generative_simulations_with_large_language_models_and_gradient-free_calibr.md)**

:   提出 G-Sim 混合框架，利用 LLM 自动设计仿真器的因果结构（子模块与连接关系），再通过无梯度优化（GFO）或仿真推断（SBI）对数值参数进行经验校准，在迭代循环中不断改进，生成可靠、可干预的通用仿真器。

**[GLGENN: A Novel Parameter-Light Equivariant Neural Networks Architecture Based on Clifford Geometric Algebras](glgenn_a_novel_parameter-light_equivariant_neural_networks_architecture_based_on.md)**

:   提出广义Lipschitz群等变神经网络(GLGENN)，利用几何代数中grade involution和reversion定义的四个基本子空间实现权重共享，在保持伪正交群等变性的同时大幅减少可训练参数（约为CGENN的1/2至1/3），在多个基准任务上匹配或超越CGENN。

**[Gradient Aligned Regression via Pairwise Losses](gradient_aligned_regression_via_pairwise_losses.md)**

:   提出 GAR（Gradient Aligned Regression），通过在标签空间引入两个成对差异损失（误差方差 + 负Pearson相关系数）来对齐预测函数与真实函数的梯度，并利用 DRO 鲁棒聚合三个子损失，实现与传统回归损失相同的线性复杂度，同时在多个基准上超越 MAE/MSE 及对比学习方法。

**[How Much Can We Forget about Data Contamination?](how_much_can_we_forget_about_data_contamination.md)**

:   通过受控实验系统量化数据污染对 LLM benchmark 评估的影响，发现在超过 Chinchilla 最优五倍以上的训练数据量下，即使 144 次重复的污染数据也能被完全遗忘；进一步证明权重衰减是遗忘的关键机制，并据此推断 Llama 3 405B 等大型模型已遗忘训练早期的数据。

**[Hyperband-based Bayesian Optimization for Black-box Prompt Selection](hyperband-based_bayesian_optimization_for_black-box_prompt_selection.md)**

:   提出 HbBoPs 方法，结合结构感知深度核高斯过程（对 instruction 和 few-shot exemplar 分别编码）与 Hyperband 多保真度调度器，在黑盒 LLM 的 prompt 选择问题上同时实现样本高效和查询高效，在十个基准和三个 LLM 上超越所有 SOTA 方法。

**[Improved and Oracle-Efficient Online $\ell_1$-Multicalibration](improved_and_oracle-efficient_online_ell_1-multicalibration.md)**

:   提出将在线 $\ell_1$-multicalibration 归约为新定义的在线线性乘积优化 (OLPO) 问题，分别达到 $\widetilde{O}(T^{-1/3})$（改进速率）和 $\widetilde{O}(T^{-1/4})$（oracle 高效速率）的多校准误差上界。

**[Improving Generalization with Flat Hilbert Bayesian Inference](improving_generalization_with_flat_hilbert_bayesian_inference.md)**

:   提出 Flat Hilbert Bayesian Inference (FHBI)，将 SAM 的平坦性概念从有限维欧氏空间推广到无限维再生核希尔伯特空间 (RKHS)，并与粒子采样贝叶斯推断结合，在 VTAB-1K 基准上以 73.7% 平均 Top-1 准确率全面超越九个基线方法。

**[Improving the Effective Receptive Field of Message-Passing Neural Networks](improving_the_effective_receptive_field_of_message-passing_neural_networks.md)**

:   本文形式化了 MPNN 中有效感受野（ERF）的概念，证明节点贡献随距离指数衰减（二项式分布），并提出 IM-MPNN 架构通过多尺度图粗化和跨尺度信息交织来扩展 ERF，在 LRGB 等长程依赖基准上显著提升。

**[Latent Imputation before Prediction: A New Computational Paradigm for De Novo Peptide Sequencing](latent_imputation_before_prediction_a_new_computational_paradigm_for_de_novo_pep.md)**

:   LIPNovo 提出在肽段预测前，通过隐空间补全（latent imputation）来弥补质谱中碎片缺失信息的新范式，利用可学习peak queries和二部匹配补全理论peak隐表示，在三个基准上大幅超越 CasaNovo 等 SOTA（氨基酸精度提升 5.6%-20%）。

**[Learning Distribution-Wise Control in Representation Space for Language Models](learning_distribution-wise_control_in_representation_space_for_language_models.md)**

:   将表示微调（Representation Fine-tuning）中的确定性节点替换为随机节点，通过重参数化技巧学习潜在分布而非单点变换，在常识推理和数学推理任务上取得了一致性能提升，尤其在早期层的干预效果最为显著。

**[Learning Safe Strategies for Value Maximizing Buyers in Uniform Price Auctions](learning_safe_strategies_for_value_maximizing_buyers_in_uniform_price_auctions.md)**

:   针对重复统一价格多物品拍卖中带有RoI约束的价值最大化买家，提出"安全竞标策略"概念，证明其仅需满足温和的不超出竞价条件，并设计多项式时间在线学习算法实现 $\widetilde{O}(M\sqrt{mT})$ 的遗憾界。

**[Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation](leveraging_online_olympiad-level_math_problems_for_llms_training_and_contaminati.md)**

:   利用 Art of Problem Solving (AoPS) 论坛的社区内容，构建了 652K 奥赛级数学 QA 对的训练集 AoPS-Instruct 和带时间戳的抗污染评估集 LiveAoPSBench，揭示了 LLM 在旧数据上的高表现可能源于预训练数据泄露而非真正推理能力。

**[LLM-SRBench: A New Benchmark for Scientific Equation Discovery with LLMs](llm-srbench_a_new_benchmark_for_scientific_equation_discovery_with_large_languag.md)**

:   提出LLM-SRBench基准（239题/4个科学领域），通过方程变换(LSR-Transform)和合成问题(LSR-Synth)防止LLM的记忆化，当前最好方法仅达31.5%符号准确率。

**[Meek Models Shall Inherit the Earth](meek_models_shall_inherit_the_earth.md)**

:   基于 Chinchilla 缩放定律的数学建模，证明在固定分布的 next-token 目标下，计算缩放的递减收益将导致 SOTA 大模型相对于低计算预算"弱模型"（meek models）的能力优势最终收敛趋零，论证 AI 能力的民主化是当前缩放范式下的必然趋势，现有基于算力的 AI 治理策略需要根本性重新设计。

**[On Temperature Scaling and Conformal Prediction of Deep Classifiers](on_temperature_scaling_and_conformal_prediction_of_deep_classifiers.md)**

:   首次系统研究 Temperature Scaling (TS) 校准对 Conformal Prediction (CP) 方法的影响，揭示 TS 在改善 APS/RAPS 类条件覆盖率的同时会增大预测集尺寸的反直觉现象，建立了完整的非单调理论解释并提出实用指南。

**[PhantomWiki: On-Demand Datasets for Reasoning and Retrieval Evaluation](phantomwiki_on-demand_datasets_for_reasoning_and_retrieval_evaluation.md)**

:   提出 PhantomWiki——一个按需生成虚构世界语料库和 QA 对的评测框架，通过上下文无关文法（CFG）控制推理难度、调节宇宙规模控制检索难度，实现对 LLM 推理与检索能力的解耦评估，同时天然抵抗数据泄漏。

**[Position: AI Evaluation Should Learn from How We Test Humans](position_ai_evaluation_should_learn_from_how_we_test_humans.md)**

:   提出将人类心理测量学中的自适应测试范式系统性引入AI评估，通过估计题目特征（难度/区分度/猜测因子）实现高效、可靠的模型能力评估，仅需3%的题目即可准确重建完整benchmark分数。

**[Position: Theory of Mind Benchmarks are Broken for Large Language Models](position_theory_of_mind_benchmarks_are_broken_for_large_language_models.md)**

:   这篇 Position Paper 指出当前大多数 LLM Theory of Mind（ToM）基准只测“能否预测他人行为”（Literal ToM），却没有测“能否基于该预测采取最优响应”（Functional ToM），因此会系统性高估模型在真实交互中的适应能力。

**[IBDR: Promoting Ensemble Diversity with Interactive Bayesian Distributional Robustness](promoting_ensemble_diversity_with_interactive_bayesian_distributional_robustness.md)**

:   提出IBDR贝叶斯推断框架，通过在乘积分布空间上引入交互式损失和Wasserstein分布鲁棒性优化，构建兼顾多样性与低锐度的粒子集成，在VTAB-1K上以ViT-B/16实现73.6%平均准确率超越所有基线。

**[Provably Cost-Sensitive Adversarial Defense via Randomized Smoothing](provably_cost-sensitive_adversarial_defense_via_randomized_smoothing.md)**

:   基于 randomized smoothing 框架提出"代价敏感认证半径"（cost-sensitive certified radius），首次实现可扩展到大模型与高维数据的代价敏感对抗鲁棒性认证与训练，在保持整体准确率的同时显著提升对高代价误分类的鲁棒性。

**[Random Registers for Cross-Domain Few-Shot Learning](random_registers_for_cross-domain_few-shot_learning.md)**

:   在跨域小样本学习（CDFSL）中发现可学习 prompt 会损害目标域泛化性能，而用随机噪声替代（即随机寄存器）反而能持续提升性能，并基于此提出 REAP 方法，通过在图像语义区域添加随机寄存器来增强注意力扰动，实现高效的域无关特征学习。

**[Ranked Entropy Minimization for Continual Test-Time Adaptation](ranked_entropy_minimization_for_continual_test-time_adaptation.md)**

:   提出 Ranked Entropy Minimization (REM)，通过渐进式遮挡策略构建预测难度的显式排序结构，结合遮挡一致性损失和熵排序损失，解决了熵最小化方法在持续测试时自适应(CTTA)中的模型崩塌问题，同时保持了计算效率。

**[ResearchTown: Simulator of Human Research Community](researchtown_simulator_of_human_research_community.md)**

:   提出 ResearchTown，一个基于 agent-data 图和 TextGNN（文本空间消息传递）的多智能体框架，将人类科研社区建模为异构图，统一模拟论文阅读、论文写作和审稿三大核心研究活动，并通过节点掩码预测任务 (ResearchBench) 进行可扩展、客观的仿真质量评估。

**[Runtime Analysis of Evolutionary NAS for Multiclass Classification](runtime_analysis_of_evolutionary_nas_for_multiclass_classification.md)**

:   首次对进化神经架构搜索(ENAS)在多类分类问题上进行运行时理论分析，证明 one-bit 和 bit-wise 变异的 (1+1)-ENAS 算法均以 $O(rM\ln rM)$ 期望运行时找到最优架构，说明简单的 one-bit 变异即可与复杂的 bit-wise 变异媲美。

**[Sample Efficient Demonstration Selection for In-Context Learning](sample_efficient_demonstration_selection_for_in-context_learning.md)**

:   本文提出了一种样本高效的上下文学习(ICL)示例选择方法，能够在有限的标注预算下高效地选择最佳示例组合，显著提升 LLM 的 ICL 性能，同时大幅减少所需的标注数据量。

**[Set-Valued Predictions for Robust Domain Generalization](set_valued_predictions_for_robust_domain_generalization.md)**

:   提出集值预测器（set-valued predictor）解决域泛化（DG）中的鲁棒性问题：输出标签子集而非单一标签，使预测在尽可能多的未见域上满足预定义的覆盖率要求，同时最小化预测集大小。

**[The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph](the_best_of_both_worlds_bridging_quality_and_diversity_in_data_selection_with_bi.md)**

:   提出 GraphFilter 方法，将 SFT 数据集建模为句子-n-gram 的二部图，通过乘法优先级函数同时优化数据质量和多样性，在 3 个模型 6 个基准上全面超越 9 种基线方法。

**[UI-Evol: Automatic Knowledge Evolving for Computer Use Agents](ui-evol_automatic_knowledge_evolving_for_computer_use_agents.md)**

:   提出UI-Evol即插即用模块，通过Retrace（从截图还原实际动作序列）和Critique（对比外部知识诊断偏差并修正）两阶段自主进化GUI任务知识，在OSWorld基准上将Agent S2的成功率从19.5%提升到22%+，同时将行为标准差降低约4倍，显著增强了计算机操作代理的可靠性。

**[Unlocking Post-hoc Dataset Inference with Synthetic Data](unlocking_post-hoc_dataset_inference_with_synthetic_data.md)**

:   提出通过合成生成held-out数据集并结合后校准（post-hoc calibration）来实现无需真实held-out集的数据集推断（Dataset Inference），通过suffix completion生成高质量合成数据、双分类器校准解耦生成偏移与成员信号，在15个多样化文本数据集上实现高置信度版权检测且低误报率。
