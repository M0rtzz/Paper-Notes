---
title: >-
  NeurIPS2025 可解释性方向 80篇论文解读
description: >-
  80篇NeurIPS2025 可解释性方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔬 可解释性

**🧠 NeurIPS2025** · **80** 篇论文解读

**[A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)**

:   发现并系统研究了 SAE 中的"特征吸收"现象：看似单义的 SAE latent 会在特定 token 上不激活，其特征方向被更具体的子 latent "吸收"，这是层级特征+稀疏性损失的必然结果，对 SAE 用于可靠解释 LLM 构成根本挑战。

**[A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis](a_unified_reasoning_framework_for_holistic_zeroshot_video_an.md)**

:   提出一个完全零样本、无需训练的视频异常分析框架，通过Intra-Task Reasoning（置信度门控的自我精化）和Inter-Task Chaining（从时序检测到空间定位到语义理解的级联prompt传递），在4个benchmark上全面超越先前零样本方法4-6% AUC。

**[AdaptGrad: Adaptive Sampling to Reduce Noise](adaptgrad_adaptive_sampling_to_reduce_noise.md)**

:   AdaptGrad通过分析SmoothGrad噪声的理论来源——超范围采样行为，提出自适应调整每个输入维度的高斯采样方差以控制额外噪声上限的方法，在几乎消除梯度噪声的同时揭示更丰富的细节特征，方法极简且可与任意梯度解释方法组合。

**[Additive Models Explained: A Computational Complexity Approach](additive_models_explained_a_computational_complexity_approach.md)**

:   对广义可加模型（GAM）的多种解释类型进行系统的计算复杂度分析，覆盖 54 种"组件模型 × 输入域 × 解释方法"组合，揭示 GAM 的解释复杂度高度依赖输入域类型——这是决策树、神经网络等其他 ML 模型从未展现的独特现象，挑战了"可加即可解释"的直觉假设。

**[AgentiQL: An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation](agentiql_an_agent-inspired_multi-expert_framework_for_text-to-sql_generation.md)**

:   提出 AgentiQL，一个多专家 agent 框架用于 Text-to-SQL：reasoning agent 分解问题为子问题，coding agent 生成子查询，refinement 步骤校正列选择，adaptive router 在基线解析器和模块化 pipeline 之间智能路由，使用 14B 开源模型达到 86.07% EX（Spider），接近 GPT-4 SOTA(89.65%)。

**[An Analysis of Concept Bottleneck Models: Measuring, Understanding, and Mitigating the Impact of Noisy Annotations](an_analysis_of_concept_bottleneck_models_measuring_understanding_and_mitigating_.md)**

:   本文首次系统研究了标注噪声对概念瓶颈模型(CBM)的影响，发现约23%的"易感概念"驱动了大部分性能退化，并提出训练阶段使用SAM + 推理阶段基于不确定性干预的两阶段缓解策略来恢复模型鲁棒性。

**[Are Greedy Task Orderings Better Than Random in Continual Linear Regression?](are_greedy_task_orderings_better_than_random_in_continual_linear_regression.md)**

:   本文系统分析了持续线性回归中贪心任务排序（最大化连续任务间不相似度）与随机排序的收敛性差异，揭示了贪心排序在高秩设定下可媲美随机排序，但在一般秩设定下单遍贪心可能灾难性失败，而允许重复的贪心排序收敛速率为 $\mathcal{O}(1/\sqrt[3]{k})$。

**[ARECHO: Autoregressive Evaluation via Chain-Based Hypothesis Optimization for Speech Multi-Metric Estimation](arecho_autoregressive_evaluation_via_chain-based_hypothesis_optimization_for_spe.md)**

:   ARECHO 将语音多指标评估建模为链式自回归 token 预测任务——设计统一的语音信息 token 化管线处理 87 个异质指标（数值/类别/有界/无界），通过动态分类链显式捕捉指标间依赖关系（如可懂度-自然度相关性），配合两步置信度导向解码减少误差传播，在增强/生成/噪声三类语音评估中全面超越 UniVERSA 基线（Avg Test MSE 23.26 vs 96.99，-76%）。

**[ARC-JSD: Attributing Response to Context via Jensen-Shannon Divergence Driven Mechanistic Study](attributing_response_to_context_a_jensen-shannon_divergence_driven_mechanistic_s.md)**

:   ARC-JSD 提出基于 Jensen-Shannon 散度的 RAG 上下文归因方法——通过比较有/无特定上下文句子时模型输出分布的 JSD 差异，无需微调/梯度计算即可定位回答所依赖的上下文，计算效率比 baseline 快 3 倍，Top-1 归因准确率平均提升 10.7%，并通过 Logit Lens 揭示归因相关的注意力头集中在高层。

**[Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models](auditing_meta-cognitive_hallucinations_in_reasoning_large_language_models.md)**

:   系统性审计推理大模型（RLLM）中幻觉的产生与传播机制，发现长 CoT 中的反思（reflection）会通过元认知偏差放大幻觉而非纠正它，即使在幻觉源头进行干预也难以改变最终结果（chain disloyalty），揭示现有幻觉检测方法在多步推理场景下严重不足。

**[Base Models Know How to Reason, Thinking Models Learn When](base_models_know_how_to_reason_thinking_models_learn_when.md)**

:   通过无监督 SAE 聚类发现 thinking model 的推理机制分类，然后用 steering vector 在基座模型上激活这些潜在推理能力，混合模型恢复高达 91% 的 thinking-base 性能差距（无需权重更新），证明基座模型已具备推理能力，thinking model 只是学会了"何时"部署它们。

**[Better Estimation of the Kullback-Leibler Divergence Between Language Models](better_estimation_of_the_kullback--leibler_divergence_between_language_models.md)**

:   提出 KL 散度的 Rao-Blackwell 化 Monte Carlo 估计器——在每个位置对下一个 token 的分布求精确 KL（而非只用采样的 token），理论证明无偏且方差严格不超过标准 MC 估计器，零额外计算开销，在 RLHF 情感控制任务中使训练更稳定、模型更频繁出现在 Pareto 前沿（78%）。

**[Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning](beyond_accuracy_dissecting_mathematical_reasoning_for_llms_u.md)**

:   提出 SPARKLE 三轴分析框架（计划执行、知识整合、子问题分解）细粒度剖析 RL 如何改变 LLM 推理行为，发现 RL 主要增强了知识整合能力和计划灵活性而非计划执行能力，并提出 SparkleRL-PSS 多阶段 RL 训练 pipeline 通过 partial step scaffolding 有效利用难题数据。

**[Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)**

:   提出基于SVD奇异向量的方向级可解释性框架，通过对注意力头和MLP的增广矩阵统一SVD分解+可学习对角掩码（KL+L₁），发现单组件内存在正交低秩子函数叠加——IOI任务仅需~9%方向即可KLD=0.21复现模型行为。

**[Beyond Token Probes: Hallucination Detection via Activation Tensors with ACT-ViT](beyond_token_probes_hallucination_detection_via_activation_tensors_with_act-vit.md)**

:   将LLM的全部隐层激活组织为"激活张量"（层×token×隐维度），类比图像用ViT处理，设计ACT-ViT架构支持跨LLM联合训练，在15个LLM-数据集组合上一致超越传统probing方法，并展现出对未见数据集和未见LLM的强零样本/少样本迁移能力。

**[Bigram Subnetworks: Mapping to Next Tokens in Transformer Language Models](bigram_subnetworks_mapping_to_next_tokens_in_transformer_language_models.md)**

:   通过连续稀疏化在Transformer语言模型中找到仅包含~10M参数的bigram子网络，它们集中在第一个MLP层，足以复现bigram预测（$r>0.95$），且被消融后模型性能大幅下降，证明这些子网络是语言模型中既必要又充分的最小next-token预测电路。

**[Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)**

:   提出 Causal Head Gating (CHG)，通过对 Transformer 的每个 attention head 学习一个可微门控系数并结合正/负正则化，将 head 分为促进（facilitating）、干扰（interfering）、无关（irrelevant）三类，无需人工标签或 prompt 模板即可发现因果子电路，并扩展为对比 CHG 以分离 ICL 和指令遵循的独立电路。

**[CBMAS: Cognitive Behavioral Modeling via Activation Steering](cbmas_cognitive_behavioral_modeling_via_activation_steering.md)**

:   CBMAS提出一个将激活引导作为连续诊断工具的框架，通过密集α扫描和注入-读取层解耦，将认知偏差分析从"有偏差/无偏差"的二元判断升级为可追踪翻转点、传播路径和衰减模式的连续轨迹分析，在GPT-2 Small上揭示了安抚行为在浅层强烈编码但向深层快速衰减的规律。

**[CHiQPM: Calibrated Hierarchical Interpretable Image Classification](chiqpm_calibrated_hierarchical_interpretable_image_classification.md)**

:   CHiQPM 提出一种校准的层次化可解释图像分类方法，通过二次规划选择和分配特征给类别，构建层次化解释路径，并内置可解释的 Conformal Prediction 集合预测，在保持黑盒模型 99% 准确率的同时提供全局和局部可解释性。

**[Cognitive Mirrors: Exploring the Diverse Functional Roles of Attention Heads in LLM Reasoning](cognitive_mirrors_exploring_the_diverse_functional_roles_of_attention_heads_in_l.md)**

:   提出CogQA基准数据集和多类probing框架，系统分析LLM中注意力头的认知功能特化现象，发现认知头具有稀疏性、普遍性和层级化功能组织特征，去除认知头显著降低推理性能，增强则提升准确率。

**[ConceptScope: Characterizing Dataset Bias via Disentangled Visual Concepts](conceptscope_characterizing_dataset_bias_via_disentangled_visual_concepts.md)**

:   提出 ConceptScope 框架，利用在视觉基础模型表征上训练的稀疏自编码器（SAE）自动发现和量化数据集中的视觉概念偏差，无需人工标注即可将概念分类为 target / context / bias 三类。

**[Conditional Distribution Compression via the Kernel Conditional Mean Embedding](conditional_distribution_compression_via_the_kernel_conditional_mean_embedding.md)**

:   首次提出针对**条件分布**（而非联合分布）的压缩算法，利用核条件均值嵌入（KCME）定义新度量 AMCMD，并设计线性时间算法 ACKIP 构建保留条件分布统计特性的压缩数据集。

**[Curvature Tuning: Provable Training-free Model Steering From a Single Parameter](curvature_tuning_provable_training-free_model_steering_from_a_single_parameter.md)**

:   提出 Curvature Tuning（CT），通过在激活函数中注入单个超参数 $\beta$ 来可证明地调节模型决策边界的曲率，无需修改权重即可提升泛化和鲁棒性，同时作为微调方法参数量远少于 LoRA rank 1。

**[Dataset Distillation for Pre-Trained Self-Supervised Vision Models](dataset_distillation_for_pre-trained_self-supervised_vision_models.md)**

:   提出 Linear Gradient Matching 方法，为预训练自监督视觉模型蒸馏合成数据集：每类仅需一张合成图就能训练出接近全数据集表现的线性分类器，且蒸馏图像可跨模型架构迁移。

**[Deep Modularity Networks with Diversity-Preserving Regularization](deep_modularity_networks_with_diversity-preserving_regularization.md)**

:   在 Deep Modularity Networks (DMoN) 基础上引入三项多样性保持正则化（距离、方差、熵），显式促进特征空间中的簇间分离和分配多样性，在特征丰富的图数据集上显著提升聚类质量。

**[Deep Value Benchmark: Measuring Whether Models Generalize Deep Values or Shallow Preferences](deep_value_benchmark_measuring_whether_models_generalize_deep_values_or_shallow_.md)**

:   提出 Deep Value Benchmark (DVB)，通过"先混淆后解混淆"的实验设计，测量 LLM 是学习了深层人类价值观还是仅记住了表层偏好模式，发现所有模型的深层价值泛化率 (DVGR) 仅为 0.30，远低于随机水平。

**[Distributional Autoencoders Know the Score](distributional_autoencoders_know_the_score.md)**

:   本文为 Distributional Principal Autoencoder (DPA) 提供了精确的理论保证：证明了最优编码器的等值面几何与数据分布的 score 函数之间的闭合形式关系，并证明了超出流形维度的潜在分量与数据条件独立，从而统一了分布学习与内在维度发现两个长期目标。

**[Do Different Prompting Methods Yield a Common Task Representation?](do_different_prompting_methods_yield_a_common_task_representation_in_language_mo.md)**

:   通过将函数向量（Function Vectors）方法从 few-shot 示例推广到文本指令，发现不同提示方式（demonstrations vs. instructions）并不会在 LLM 中诱导出统一的任务表征，而是激活部分重叠但主要不同的注意力头机制。

**[Dynamic Algorithm for Explainable k-medians Clustering under lp Norm](dynamic_algorithm_for_explainable_k-medians_clustering_under_lp_norm.md)**

:   本文提出首个适用于一般 $\ell_p$ 范数的可解释 k-medians 聚类算法，实现 $\tilde{O}(p(\log k)^{1+1/p-1/p^2})$ 近似比（改进了 p=2 的已知最优界），并给出首个动态版本：在中心集合的插入/删除下，以 $O(d \log^3 k)$ 摊还更新时间和 $O(\log k)$ 重分配次数维护可解释聚类。

**[Efficient Vision-Language Reasoning via Adaptive Token Pruning](efficient_vision-language_reasoning_via_adaptive_token_pruning.md)**

:   提出 Adaptive Token Pruning (ATP)，一种免训练的即插即用模块，通过融合 ViT CLS 注意力（模态内显著性）和 CLIP 文本-图像相似度（模态间相关性）来筛选最有信息量的视觉 token，在 VQA/GQA/COCO Captioning 上以约 40% FLOPs 降低和 1.5 倍加速换取不到 1% 的精度损失。

**[Emergence of Linear Truth Encodings in Language Models](emergence_of_linear_truth_encodings_in_language_models.md)**

:   提出 **Truth Co-occurrence Hypothesis (TCH)**——真实陈述倾向于与其他真实陈述共现——并通过一个最简单的单层 Transformer 玩具模型，端到端地展示了线性真值子空间如何通过两阶段训练动态（先记忆 → 后编码真值）自然涌现，为理解 LLM 中广泛报告的线性真值表示提供了首个机制性解释。

**[Empowering Decision Trees via Shape Function Branching](empowering_decision_trees_via_shape_function_branching.md)**

:   提出 Shape Generalized Tree (SGT)，在决策树每个内部节点使用可学习的轴对齐形状函数替代传统线性阈值分裂，以更紧凑的树结构捕捉非线性特征效应，同时保持可解释性。

**[Encoding and Understanding Astrophysical Information in Large Language Model-Generated Summaries](encoding_and_understanding_astrophysical_information_in_large_language_model-gen.md)**

:   探究LLM嵌入是否能编码从X射线天文观测导出的物理量（硬度比、幂律指数、变异性），发现结构化prompt设计可将物理属性聚类纯度提升5.9%-57.5%，稀疏自编码器揭示LLM通过识别天体类型来推断未显式给出的物理参数。

**[Evaluating LLMs in Open-Source Games](evaluating_llms_in_open-source_games.md)**

:   通过开源游戏（智能体提交程序而非原始行动）这一新范式，系统评估 LLM 在战略推理、互相学习和合作博弈中的能力，发现 LLM 可自动发现近似程序平衡。

**[Explaining Similarity in Vision-Language Encoders with Weighted Banzhaf Interactions](explaining_similarity_in_vision-language_encoders_with_weighted_banzhaf_interact.md)**

:   FIxLIP 提出基于加权 Banzhaf 交互指数的博弈论框架，统一分解视觉-语言编码器（如 CLIP、SigLIP-2）的相似度预测为一阶token归因和二阶跨模态/模态内交互，在效率和忠实度上均超越现有一阶归因方法。

**[FaCT: Faithful Concept Traces for Explaining Neural Network Decisions](fact_faithful_concept_traces_for_explaining_neural_network_decisions.md)**

:   提出 FaCT，一种结合 B-cos 变换和稀疏自编码器 (SAE) 的内在可解释模型，能够**忠实地**将模型预测分解为概念贡献（Logit = $\sum$ 概念贡献），并将每个概念忠实地可视化到输入像素级别（概念激活 = $\sum$ 像素贡献），同时提出基于 DINOv2 的 C²-score 用于评估概念一致性。

**[Fantastic Features and Where to Find Them: A Probing Method to Combine Features from Multiple Foundation Models](fantastic_features_and_where_to_find_them_a_probing_method_to_combine_features_f.md)**

:   提出 ComBo，一种基于 probing 的轻量级 adapter，通过仿射投影压缩多个冻结基础模型多层激活，再用小型 transformer 融合，无需反向传播即可高效整合多模型互补表征，在 VTAB-1k 上超越先前 probing 方法并匹配蒸馏方法。

**[FastDINOv2: Frequency Based Curriculum Learning Improves Robustness and Training Speed](fastdinov2_frequency_based_curriculum_learning_improves_robustness_and_training_.md)**

:   提出 FastDINOv2，一种两阶段频率课程学习策略：先用低分辨率图像训练 75% epochs 学习低频特征以加速收敛，再用全分辨率+高斯噪声 patching 训练 25% epochs 平衡频率偏置，实现 1.6× 加速、2.25× FLOPs 节省，同时增强鲁棒性。

**[From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit](from_flat_to_hierarchical_extracting_sparse_representations_with_matching_pursui.md)**

:   提出 MP-SAE，将经典 Matching Pursuit 算法展开为 SAE 的序列化编码器，通过残差引导的贪心特征选择实现条件正交性，能捕捉标准 SAE 无法发现的层次结构、非线性可及和跨模态特征，并天然支持推理时自适应稀疏度调节。

**[Geometric Priors for Generalizable World Models via Vector Symbolic Architecture](geometric_priors_for_generalizable_world_models_via_vector_symbolic_architecture.md)**

:   提出将 Vector Symbolic Architecture (VSA) 中的 Fourier Holographic Reduced Representation (FHRR) 作为几何先验引入世界模型，通过 element-wise 复数乘法建模状态转移，在离散 GridWorld 上实现 87.5% 的 zero-shot 泛化准确率和 4 倍于 MLP 的噪声鲁棒性。

**[H-SPLID: HSIC-based Saliency Preserving Latent Information Decomposition](h-splid_hsic-based_saliency_preserving_latent_information_decomposition.md)**

:   提出 H-SPLID，通过将隐空间显式分解为**显著（任务相关）**和**非显著（任务无关）**两个子空间，结合 HSIC 正则化实现信息压缩，证明预测偏差上界受显著子空间维度和 HSIC 控制，在无对抗训练条件下显著提升对非显著区域扰动的鲁棒性。

**[How Do Transformers Learn Implicit Reasoning?](how_do_transformers_learn_implicit_reasoning.md)**

:   在精细控制的符号环境中从零训练 Transformer，发现多跳隐式推理经历"记忆→分布内泛化→跨分布泛化"三个阶段，核心机制不是中间实体的可解码性，而是其在余弦空间中的聚类一致性——同一中间实体的表示在不同查询中形成紧密聚类时，推理能力才涌现。

**[Improving Perturbation-based Explanations by Understanding the Role of Uncertainty Calibration](improving_perturbation-based_explanations_by_understanding_the_role_of_uncertain.md)**

:   揭示了不确定性校准（模型置信度与实际准确率的对齐）与扰动式可解释性方法质量之间的根本联系，证明模型在扰动输入下的误校准直接损害全局和局部解释质量，并提出 ReCalX 通过扰动级别自适应温度缩放显著改善解释的鲁棒性和保真度。

**[Knowing When to Stop: Efficient Context Processing via Latent Sufficiency Signals](knowing_when_to_stop_efficient_context_processing_via_latent_sufficiency_signals.md)**

:   本文提出 dynamic context cutoff，通过探测 Transformer 特定注意力头中编码的"信息充分性信号"，训练轻量分类器判断模型何时已获取足够上下文，实现提前终止处理，在6个QA数据集上平均提高3.4%准确率同时减少1.33×token消耗。

**[Latent Principle Discovery for Language Model Self-Improvement](latent_principle_discovery_for_language_model_self-improvement.md)**

:   STaPLe 提出后验正则化的蒙特卡洛 EM 算法，让 7-8B 小模型自行发现指导自我修正的"原则"（latent principle），通过迭代发现-学习循环实现自我改进，在 AlpacaEval 上提升 8-10% 胜率、MT-Bench 平均提升 +0.3，并可通过聚类压缩至可解释的 constitution。

**[Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning](learning_to_focus_causal_attention_distillation_via_gradient-guided_token_prunin.md)**

:   提出Learning to Focus (LeaF)框架，通过梯度引导识别训练数据中的"混淆token"（confounding tokens），在知识蒸馏过程中剪枝这些token以构建反事实样本，使学生模型的注意力对齐到教师模型关注的关键上下文token上，从而提升数学推理和代码生成的准确性。

**[LLM Probing with Contrastive Eigenproblems: Improving Understanding and Applicability of CCS](llm_probing_with_contrastive_eigenproblems_improving_understanding_and_applicabi.md)**

:   本文对无监督探测方法 CCS（Contrast-Consistent Search）进行了深入分析，提出将 CCS 重新表述为特征值问题（Contrastive Eigenproblems），获得闭式解和可解释的特征值，避免了 CCS 对随机初始化的敏感性，并自然扩展到多变量设置。

**[Minimizing False-Positive Attributions in Explanations of Non-Linear Models](minimizing_false-positive_attributions_in_explanations_of_non-linear_models.md)**

:   针对非线性模型的XAI解释中抑制变量(suppressor variable)导致的假阳性归因问题，提出PatternLocal方法，将局部判别式代理模型权重转换为生成式表示，在XAI-TRIS基准、MRI人工病灶和EEG运动想象三个数据集上显著减少了假阳性特征归因。

**[Monte Carlo Expected Threat (MOCET) Scoring](monte_carlo_expected_threat_mocet_scoring.md)**

:   提出 MOCET（Monte Carlo Expected Threat）评分框架，通过将 LLM 生成的生物武器制造协议分解为逐步 Bernoulli 试验，结合 k-NN 语义嵌入的成功概率估计和蒙特卡洛模拟，生成可解释的、可自动化的威胁量化指标，用于衡量 LLM 在生物安全领域的真实世界风险。

**[MoPFormer: Motion-Primitive Transformer for Wearable-Sensor Activity Recognition](mopformer_motion-primitive_transformer_for_wearable-sensor_activity_recognition.md)**

:   提出 MoPFormer，将可穿戴传感器信号分解为运动原语（motion primitives）序列，通过 Transformer 建模原语间的时序依赖关系，在多个 HAR 基准上超越 SOTA 并保持轻量化。

**[OrdShap: Feature Position Importance for Sequential Black-Box Models](ordshap_feature_position_importance_for_sequential_black-box_models.md)**

:   提出 OrdShap，一种针对序列模型的特征归因方法，首次将特征的**值重要性（Value Importance, VI）**与**位置重要性（Position Importance, PI）**解耦，基于 Sanchez-Bergantiños 博弈论值提供理论保证。

**[Out of Control -- Why Alignment Needs Formal Control Theory (and an Alignment Control Stack)](out_of_control_--_why_alignment_needs_formal_control_theory_and_an_alignment_con.md)**

:   本文是一篇 position paper，主张将形式化最优控制理论作为 AI 对齐研究的核心工具，并提出"对齐控制栈"(Alignment Control Stack, ACS)——一个从物理硬件层到社会治理层的十层分层框架，用于系统地组织和分析不同对齐方法的测量、控制与互操作性。

**[Partial Information Decomposition via Normalizing Flows in Latent Gaussian Distributions](partial_information_decomposition_via_normalizing_flows_in_latent_gaussian_distr.md)**

:   提出两个互补工具：Thin-PID 是一种高效高斯 PID 算法（比已有方法快 10×），Flow-PID 用 normalizing flow 将任意输入分布转换为高斯再计算 PID，解决了 PID 在连续高维数据上不可行的问题，并证明了"联合高斯解是否最优"这一开放问题。

**[Probabilistic Token Alignment for Large Language Model Fusion](probabilistic_token_alignment_for_large_language_model_fusion.md)**

:   将 LLM 融合中的 token 对齐问题重新建模为最优传输（Optimal Transport）问题，用动态 token 配对 + Sinkhorn 算法实现"软"概率对齐取代传统硬映射，在 6 大基准 78 个任务上相比 FuseLLM 平均提升 +1.72%，同时在困难任务上大幅缓解性能退化（从 -13.04% 降至 -4.07%）。

**[Rectifying Shortcut Behaviors in Preference-based Reward Learning](rectifying_shortcut_behaviors_in_preference-based_reward_learning.md)**

:   提出 PRISM（Preference-based Reward Invariance for Shortcut Mitigation），将 reward hacking 统一建模为 shortcut learning 问题，通过群不变核（group-invariant kernels）和随机特征映射近似来同时缓解多种 spurious correlation（冗长性、谄媚、语气等），在 out-of-distribution 偏好数据和下游策略模型上一致提升表现。

**[Saying the Unsaid: Revealing the Hidden Language of Multimodal Systems Through Telephone Games](saying_the_unsaid_revealing_the_hidden_language_of_multimodal_systems_through_te.md)**

:   通过多轮"电话游戏"（图像→文本→图像循环）利用多模态系统的偏好偏差，量化系统隐含空间中概念间的连接强度（即"隐含语言"），贡献Telescope数据集（10,000+概念对），建立可在测试时扩展的多模态系统"世界地图"。

**[scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)**

:   提出 scPilot 框架和 scBench 基准，让LLM直接在单细胞RNA-seq数据上进行"组学原生推理"（读取标记基因→提出假设→调用工具验证→迭代修正），实现细胞类型标注准确率提升11%、轨迹推断graph-edit distance降低30%。

**[Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)**

:   从理论上证明自监督对比学习（DCL）近似等价于一种有监督对比损失（NSCL），两者差距以 $O(1/C)$ 速度随类别数增加而消失；进一步证明 NSCL 全局最优解满足 Neural Collapse（增强坍缩 + 类内坍缩 + Simplex ETF），并提出基于方向性 CDNV 的更紧的 few-shot 误差界。

**[SHAP Values via Sparse Fourier Representation](shap_values_via_sparse_fourier_representation.md)**

:   提出 FourierShap 算法，先将黑盒预测器近似为稀疏 Fourier 表示，再利用 Fourier 基函数的 SHAP 值闭式公式高效计算特征归因，实现相比 KernelShap 10-10000 倍的加速，同时支持精度-效率的可调权衡。

**[Simulating Society Requires Simulating Thought](simulating_society_requires_simulating_thought.md)**

:   本文提出从"行为主义"模式转向"认知建模"范式，通过 GenMinds 框架用因果信念图建模 LLM Agent 的内部推理过程，并设计 RECAP 基准从可追溯性、人口统计敏感性和干预一致性三维度评估推理保真度。

**[Sloth: Scaling Laws for LLM Skills to Predict Multi-Benchmark Performance Across Families](sloth_scaling_laws_for_llm_skills_to_predict_multi-benchmark_performance_across_.md)**

:   提出Skills Scaling Laws (Sloth)，通过假设LLM性能由低维潜在技能（如推理、指令遵循）驱动，利用benchmark间的相关性构建跨模型家族的缩放定律，用少量家族数据即可预测大模型在多个benchmark上的表现。

**[SpEx: A Spectral Approach to Explainable Clustering](spex_a_spectral_approach_to_explainable_clustering.md)**

:   提出SpEx，基于谱图划分的通用可解释聚类方法，可将任意参考聚类（无需质心）通过坐标切割决策树"圆化"为可解释聚类，或直接在kNN图上进行无参考聚类。

**[Steering Information Utility in Key-Value Memory for Language Model Post-Training](steering_information_utility_in_key-value_memory_for_language_model_post-trainin.md)**

:   提出 InfoSteer，一种轻量级方法，将 Transformer 的 FFN 层视为关联键值记忆，通过前向传播干预（提升低活跃记忆向量的 key coefficient）和反向传播正则化（最大化 key 分布熵）来促进预训练知识在后训练阶段的充分利用。在 Qwen/LLaMA/Gemma 三个系列 6 个模型上，15 个 ID+OOD 任务一致提升，且被引导的 LM 展现出自适应信息分配行为。

**[TangledFeatures: Robust Feature Selection in Highly Correlated Spaces](tangledfeatures_robust_feature_selection_in_highly_correlated_spaces.md)**

:   提出 TangledFeatures，一个以**特征稳定性**为核心目标的选择框架，通过相关性图聚类→集成代表选择→随机森林精炼的三阶段管线，在高度相关的特征空间中实现跨重采样高度可复现且与领域知识一致的特征子集，在丙氨酸二肽骨架扭转角预测中验证有效。

**[The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?](the_non-linear_representation_dilemma_is_causal_abstraction_enough_for_mechanist.md)**

:   证明了当因果抽象（causal abstraction）中的对齐映射不受线性约束时，任意神经网络都可以被映射到任意算法，使得因果抽象变得平凡而无信息量，由此提出"非线性表示困境"——在对齐映射的复杂度与准确度之间缺乏原则性的权衡方式。

**[The Trilemma of Truth in Large Language Models](the_trilemma_of_truth_in_large_language_models.md)**

:   提出 sAwMIL（稀疏感知多实例学习）三类探测框架，结合 MIL 和保形预测，将 LLM 内部激活分类为 true/false/neither，揭示真假信号并非简单的双向对称编码，而是跨越多维子空间的分布式表征。

**[Time-Evolving Dynamical System for Learning Latent Representations of Mouse Visual Cortex](time-evolving_dynamical_system_for_learning_latent_representations_of_mouse_visu.md)**

:   提出TE-ViDS，一种时序潜变量模型，将视觉神经活动分解为与视觉刺激相关的外部表征和反映内部状态的内部表征，通过时间演化结构和对比学习实现最优的自然场景/视频解码性能。

**[How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis](toward_explainable_offline_rl_analyzing_representations_in_intrinsically_motivat.md)**

:   提出一个系统性的事后可解释性框架，分析内在动机（基于Random Network Distillation）如何塑造Elastic Decision Transformer的嵌入空间几何结构，揭示不同内在动机变体创造了根本不同的表示结构——EDT-SIL促进紧凑表示，EDT-TIL增强正交性——且嵌入属性与任务性能存在强烈的环境特异性相关。

**[Toward Real-world Text Image Forgery Localization: Structured and Interpretable Data Synthesis](toward_real-world_text_image_forgery_localization_structured_and_interpretable_d.md)**

:   提出基于傅里叶级数的篡改合成框架 FSTS，通过从67名人类参与者收集的16750个真实篡改实例中建模"不可见分布"（篡改操作参数的高维分布），生成更贴近真实世界的合成训练数据，显著提升文本图像篡改定位模型的泛化能力。

**[Towards Interpretability Without Sacrifice: Faithful Dense Layer Decomposition with Mixture of Decoders](towards_interpretability_without_sacrifice_faithful_dense_layer_decomposition_wi.md)**

:   提出 Mixture of Decoders (MxD)，将 LLM 的 MLP 层分解为数万个稀疏激活的专家子层（layer-level sparsity），每个专家通过 Hadamard 乘积张量分解实现满秩线性变换，在稀疏性-准确性权衡上显著优于 Transcoders，同时保持可解释性。

**[Towards Scaling Laws for Symbolic Regression](towards_scaling_laws_for_symbolic_regression.md)**

:   首次系统研究符号回归（SR）中的缩放定律，证明基于 Transformer 的端到端 SR 在三个数量级的计算范围内遵循幂律缩放趋势，并给出最优 token-to-parameter ratio $\approx 15$、batch size 和学习率随模型规模增长的经验规律。

**[Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders](transformer_key-value_memories_are_nearly_as_interpretable_as_sparse_autoencoder.md)**

:   系统比较了Transformer前馈层（FF）的键值记忆特征与稀疏自编码器（SAE）学到的特征的可解释性，发现两者在现有评测指标上表现相当，FF-KV在某些方面甚至更优，质疑了SAE作为特征发现工具的必要性。

**[Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms](tropical_attention_neural_algorithmic_reasoning_for_combinatorial_algorithms.md)**

:   Tropical Attention用热带代数几何替代softmax点积注意力，在热带射影空间中进行分段线性推理，实现与组合算法多面体决策结构的对齐，首次将神经算法推理扩展到NP-hard问题，在长度/数值/噪声三种OOD泛化上全面超越softmax基线。

**[Uncovering Graph Reasoning in Decoder-only Transformers with Circuit Tracing](uncovering_graph_reasoning_in_decoder-only_transformers_with_circuit_tracing.md)**

:   通过电路追踪 (circuit tracing) 框架分析 decoder-only Transformer 在图推理任务上的内部机制，发现了 token merging 和 structural memorization 两个核心推理机制。

**[URLs Help, Topics Guide: Understanding Metadata Utility in LLM Training](urls_help_topics_guide_understanding_metadata_utility_in_llm_training.md)**

:   系统评估了三类元数据（URL、质量分数、主题/格式域信息）作为预训练上下文的效果：发现只有 URL 能加速训练（100B token 用 60B 即达到相同下游性能），且仅在长 prompt（5-shot）下有效；质量分数和主题域信息不加速训练但可用于 classifier-free guidance 实现可控生成。

**[VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity](vadtree_explainable_training-free_video_anomaly_detection_via_hierarchical_granu.md)**

:   提出 VADTree，一种训练无关的视频异常检测框架，利用预训练的通用事件边界检测（GEBD）模型构建层次粒度感知树（HGTree），实现对不同时间跨度异常事件的自适应采样和多粒度推理，在 UCF-Crime、XD-Violence 和 MSAD 三个基准上取得训练无关方法SOTA，甚至超越部分弱监督方法。

**[ValuePilot: A Two-Phase Framework for Value-Driven Decision-Making](valuepilot_a_two-phase_framework_for_value-driven_decision-making.md)**

:   提出 ValuePilot 两阶段框架，通过数据集生成工具包（DGT）构建价值标注场景，再用决策模块（DMM）结合用户个性化价值偏好进行多准则决策，在与人类决策对齐方面超过 GPT-5 等强基线。

**[VL-SAE: Interpreting and Enhancing Vision-Language Alignment with a Unified Concept Set](vlsae_interpreting_and_enhancing_visionlanguage_alignment_wi.md)**

:   提出VL-SAE，一种带有距离编码器和模态特定解码器的稀疏自编码器，将视觉和语言表示的语义映射到统一概念集，从而解释和增强VLM的视觉-语言对齐机制，在零样本分类平均提升0.6-0.9%，在POPE幻觉消除上超越专用方法VCD。

**[What Happens During the Loss Plateau? Understanding Abrupt Learning in Transformers](what_happens_during_the_loss_plateau_understanding_abrupt_learning_in_transforme.md)**

:   系统研究 Transformer 训练中的"突变学习"现象，揭示 loss 平台期内模型已学会部分解、同时表现出输出重复偏差和表示坍缩，并证明注意力图的缓慢学习是关键瓶颈，相关发现在 Pythia/OLMo 等 LLM 预训练早期也得到验证。

**[Why Is Attention Sparse in Particle Transformer?](why_is_attention_sparse_in_particle_transformer.md)**

:   本文系统性地分析了 Particle Transformer（ParT）在 jet tagging 任务中训练后出现的近乎二值化稀疏 attention 现象，通过跨数据集对比和消融实验揭示了稀疏性主要源自 attention 机制自身而非物理启发的 interaction 矩阵，但 interaction 矩阵通过影响绝大多数 token 的 argmax 选择对最终性能不可或缺。
