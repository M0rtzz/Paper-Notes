---
title: >-
  AAAI2026 可解释性方向 32篇论文解读
description: >-
  32篇AAAI2026 可解释性方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔬 可解释性

**🤖 AAAI2026** · 共 **32** 篇

**[A Closer Look At Knowledge Distillation In Spiking Neural Ne](a_closer_look_at_knowledge_distillation_in_spiking_neural_ne.md)**

:   针对ANN→SNN知识蒸馏中教师ANN连续特征/logits与学生SNN离散稀疏spike特征/logits之间分布差异被忽视的问题，提出基于显著性缩放激活图蒸馏（SAMD）和噪声平滑logits蒸馏（NLD）的CKDSNN框架，在CIFAR-10/100、ImageNet-1K和CIFAR10-DVS上均取得SNN训练的新SOTA。

**[A Coherence-Based Measure Of Agi](a_coherence-based_measure_of_agi.md)**

:   指出现有 AGI 评分用算术平均隐含"可补偿"假设（强项弥补弱项），提出基于广义均值连续谱的一致性度量 $\text{AGI}_{\text{AUC}}$：在补偿性参数 $p \in [-1, 1]$ 上积分，惩罚能力不均衡，暴露被算术平均掩盖的瓶颈。

**[Adaptive Evidential Learning For Temporal-Semantic Robustnes](adaptive_evidential_learning_for_temporal-semantic_robustnes.md)**

:   提出 DEMR 框架，将深度证据回归（DER）引入视频时刻检索任务，通过 Reflective Flipped Fusion 模块缓解模态不平衡、通过 Geom-regularizer 修复原始 DER 中不确定性估计的反直觉偏差，在标准和去偏数据集上均取得了显著提升。

**[Attention Gathers Mlps Compose A Causal Analysis Of An Action-Outcome Circuit In](attention_gathers_mlps_compose_a_causal_analysis_of_an_action-outcome_circuit_in.md)**

:   通过机械可解释性方法逆向工程 Video Vision Transformer（ViViT）的内部电路，揭示注意力头负责"收集证据"、MLP 模块负责"组合概念"的分工机制，证明模型在简单分类任务中隐藏了超越训练目标的语义知识。

**[Beyond Hallucinations A Composite Score For Measuring Reliability In Open-Source](beyond_hallucinations_a_composite_score_for_measuring_reliability_in_open-source.md)**

:   提出 Composite Reliability Score (CRS)，将校准度、鲁棒性和不确定性量化三个维度统一为单一可解释指标，对 10 个开源 LLM 在 5 个 QA 数据集上进行系统评估，发现 Mistral-8x22B 综合可靠性最高（CRS=0.81），而模型大小并不直接决定可靠性。

**[Concepts From Representations Post-Hoc Concept Bottleneck Models Via Sparse Deco](concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)**

:   提出 PCBM-ReD，通过从预训练视觉编码器中自动提取概念、MLLM 标注/过滤、重建引导选择，再利用 CLIP 视觉-文本对齐将图像表示稀疏分解为概念嵌入的线性组合，构建事后概念瓶颈模型，在 11 个分类任务上达到 SOTA 精度且保持可解释性。

**[Crosscheck-Bench Diagnosing Compositional Failures In Multim](crosscheck-bench_diagnosing_compositional_failures_in_multim.md)**

:   构建CrossCheck-Bench——首个专注于多模态矛盾检测与解决的诊断基准，包含15K QA对、3层推理复杂度和7种原子能力，发现13个SOTA VLM从感知匹配到逻辑矛盾检测性能一致下降，CoT/SoM等提示策略收效甚微，仅交错符号推理+视觉grounding的方法才有稳定提升。

**[Data Whitening Improves Sparse Autoencoder Learning](data_whitening_improves_sparse_autoencoder_learning.md)**

:   本文将经典稀疏编码中的 PCA 白化（whitening）引入现代稀疏自编码器（SAE）训练，通过理论分析和仿真证明白化能使优化景观更凸更各向同性，在 SAEBench 上的实验表明白化显著提升可解释性指标（Sparse Probing +7.3%、SCR +54%、TPP +372%），尽管重构质量略有下降。

**[Distribution-Based Feature Attribution For Explaining The Predictions Of Any Cla](distribution-based_feature_attribution_for_explaining_the_predictions_of_any_cla.md)**

:   提出首个基于数据分布的特征归因方法 DFAX，通过比较目标实例在目标类与非目标类条件概率之差来量化特征重要性，首次给出特征归因的形式化定义，在10个数据集上显著优于SHAP/LIME等基线且速度快数个数量级。

**[Drexperts Differential Refinement Of Distortion-Aware Experts For Blind Image Qu](drexperts_differential_refinement_of_distortion-aware_experts_for_blind_image_qu.md)**

:   提出DR.Experts框架，利用DA-CLIP获取失真类型先验，通过差分精炼注意力机制（DSDM）将失真注意力与语义注意力分离以纯化失真特征，再通过动态失真加权模块（DDWM）按感知影响自适应加权各类失真特征，在5个BIQA基准上达到SOTA。

**[Elementarynet A Non-Strategic Neural Network For Predicting Human Behavior In No](elementarynet_a_non-strategic_neural_network_for_predicting_human_behavior_in_no.md)**

:   提出 ElementaryNet，一种**可证明不具备策略性推理能力**的神经网络架构，用于建模博弈中人类的"level-0"（非策略性）行为，在预测准确率上与 GameNet（当前 SOTA）无统计差异，同时具备更好的可解释性。

**[Enhancing Binary Encoded Crime Linkage Analysis Using Siamese Network](enhancing_binary_encoded_crime_linkage_analysis_using_siamese_network.md)**

:   提出基于 **Siamese Autoencoder** 的犯罪关联分析框架，通过 **decoder 阶段融合地理时间特征** 和 **领域专家驱动的数据降维策略**，在英国 NCA 的真实 ViCLAS 数据库上实现了 AUC 提升最高 9%，为高维稀疏二进制编码犯罪数据提供了有效的机器学习解决方案。

**[Explainable Melanoma Diagnosis With Contrastive Learning And Llm-Based Report Ge](explainable_melanoma_diagnosis_with_contrastive_learning_and_llm-based_report_ge.md)**

:   提出 CEFM 框架，通过跨模态对比学习将 ViT 视觉特征与基于 ABCD 规则的临床特征（不对称性、边界、颜色）对齐，再由 CLIP + DeepSeek 生成结构化诊断报告，在 ISIC 数据集上达到 92.79% 准确率和 0.961 AUC，专家评分可解释性达 4.6/5。

**[Finding The Translation Switch Discovering And Exploiting The Task-Initiation Fe](finding_the_translation_switch_discovering_and_exploiting_the_task-initiation_fe.md)**

:   利用稀疏自编码器（SAE）发现 LLM 中控制翻译任务启动的"翻译启动特征"，通过因果干预验证其功能（增强特征→提升翻译质量/减少幻觉，消除特征→产生幻觉），并将该机制洞察转化为实用的数据选择策略——优先在"机制困难"样本上微调，显著提升数据效率和抑制幻觉。

**[Finevau A Novel Human-Aligned Benchmark For Fine-Grained Video Anomaly Understan](finevau_a_novel_human-aligned_benchmark_for_fine-grained_video_anomaly_understan.md)**

:   本文提出FineVAU基准，将视频异常理解 (VAU) 分解为事件(What)、实体(Who)、地点(Where)三个维度，设计了与人类感知高度对齐的FV-Score评估指标，并通过全自动LVLM辅助管线构建了FineW³数据集，实验揭示当前LVLM在细粒度异常事件感知上的关键短板。

**[Flashkat Understanding And Addressing Performance Bottlenecks In The Kolmogorov-](flashkat_understanding_and_addressing_performance_bottlenecks_in_the_kolmogorov-.md)**

:   深入分析 KAT（Kolmogorov-Arnold Transformer）训练慢 123 倍的根因，发现瓶颈并非 FLOPs 而是反向传播中**梯度累积的内存停顿**（atomic add 导致全局内存竞争），提出 FlashKAT 通过重构 GPU 核函数将训练加速 **86.5 倍**并降低近一个数量级的梯度舍入误差。

**[Flexible Concept Bottleneck Model](flexible_concept_bottleneck_model.md)**

:   本文提出Flexible Concept Bottleneck Model (FCBM)，通过引入超网络动态生成概念权重和可学习温度的sparsemax模块，实现了概念池的动态适配（包括完全替换），并在5个公开数据集上以相似的有效概念数达到了与SOTA基线可比的精度，仅需单个epoch微调即可适应全新概念集。

**[Fourierpet Deep Fourier-Based Unrolled Network For Low-Count Pet Reconstruction](fourierpet_deep_fourier-based_unrolled_network_for_low-count_pet_reconstruction.md)**

:   发现低剂量 PET 的三类退化在频域可分离——泊松噪声/光子不足导致高频相位扰动，衰减校正误差抑制低频幅度——据此提出 FourierPET：基于 ADMM 展开的频率感知重建框架，仅 0.44M 参数在三个数据集上全面 SOTA。

**[Gatera Token-Aware Modulation For Parameter-Efficient Fine-Tuning](gatera_token-aware_modulation_for_parameter-efficient_fine-tuning.md)**

:   提出 GateRA，在 PEFT 方法（LoRA/DoRA/HiRA）中引入轻量级 token 感知门控模块，通过 sigmoid 门控动态调整每个 token 的适配强度——对分布内/简单 token 抑制更新以保留预训练知识，对挑战性 token 放大适配。结合熵正则化促进近二值门控决策，在常识推理（+1.1%）、对话和数学推理上一致优于 HiRA。

**[Genepheno Interpretable Gene Knockout-Induced Phenotype Abnormality Prediction F](genepheno_interpretable_gene_knockout-induced_phenotype_abnormality_prediction_f.md)**

:   本文提出 GenePheno，首个从基因序列端到端预测基因敲除诱导表型异常的可解释多标签预测框架，通过对比式多标签学习捕获表型间相关性、互斥正则化强制生物学一致性、以及基因本体（GO）瓶颈层提供可解释性，在 4 个数据集上取得 SOTA 的基因中心 $F_{\max}$ 和表型中心 AUC。

**[Hskbenchmark Modeling And Benchmarking Chinese Second Language Acquisition In La](hskbenchmark_modeling_and_benchmarking_chinese_second_language_acquisition_in_la.md)**

:   提出 HSKBenchmark，首个面向 LLM 中文二语习得（SLA）分阶段建模与写作评估的基准，包含 HSK 3-6 级教材（6.76M tokens）、16K 合成指令数据、30 个测试题目及语言学评估系统，配合课程式微调框架模拟人类习得轨迹。

**[Hypothesis Generation Via Llm-Automated Language Bias For Ilp](hypothesis_generation_via_llm-automated_language_bias_for_ilp.md)**

:   提出首个端到端框架：多Agent LLM系统（Actor/Critic）自动从原始文本构建ILP语言偏差（谓词系统+类型声明+模式约束），Translator将文本翻译为Prolog事实，再由MAXSYNTH求解器基于MDL原则归纳全局最优规则集。在SHOES和ZENDO任务上分别达88.3%和81.3%准确率，跨4种LLM方差<5%。

**[Imad Intelligent Multi-Agent Debate For Efficient And Accura](imad_intelligent_multi-agent_debate_for_efficient_and_accura.md)**

:   iMAD 提出选择性触发多Agent辩论的框架：先让单Agent生成带自我批判的结构化响应，从中提取 41 个可解释的语言/语义特征，用轻量 MLP 分类器（FocusCal 损失训练）判断是否需要触发 MAD，在 6 个 QA/VQA 数据集上减少高达 92% 的 Token 开销，同时提升准确率高达 13.5%。

**[Induce Align Predict Zero-Shot Stance Detection Via Cognitive Inductive Reasonin](induce_align_predict_zero-shot_stance_detection_via_cognitive_inductive_reasonin.md)**

:   提出CIRF（Cognitive Inductive Reasoning Framework），受认知科学启发，从原始文本中无监督归纳一阶逻辑推理模式（schema），构建多关系schema图，用图核模型对齐输入与schema模板实现可解释的零样本立场推理，在SemEval-2016、VAST和COVID-19-Stance上达到SOTA，仅30%数据即可匹配全量。

**[Llm Circuit Analyses Consistent Across Training And Scale](llm_circuit_analyses_consistent_across_training_and_scale.md)**

:   本文首次系统追踪 decoder-only LLM 的内部电路（circuits）在 3000 亿 token 训练过程中和 70M–2.8B 参数规模间的演化，发现虽然具体注意力头会发生更替，但执行的算法保持稳定，且跨规模具有一致性，表明在小模型上做的电路分析可推广到更大模型和更长训练。

**[Probing Preference Representations A Multi-Dimensional Evaluation And Analysis M](probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)**

:   提出 MRMBench 基准，通过 6 个维度（无害性、有帮助性、正确性、连贯性、复杂性、冗长性）的探针任务评估奖励模型是否有效捕获多维偏好，发现探针性能与 PPO 对齐质量强相关（Pearson $r > 0.8$），并提出推理时探针方法将 AlpacaEval win rate 从 57.3% 提升至 62.5%。

**[Quiet Feature Learning In Algorithmic Tasks](quiet_feature_learning_in_algorithmic_tasks.md)**

:   在 10 个算法任务（18,544 次训练运行，$10^9$-$10^{16}$ FLOPs）上发现，Transformer 的损失平台期并非学习停滞——模型在此期间悄悄学习了"安静特征"（中间算法子程序），这些特征不直接降低输出损失但对最终性能因果必要（消融后准确率下降 41-75%）。这挑战了用损失曲线判断训练进展的常规做法。

**[Scope Intrinsic Semantic Space Control For Mitigating Copyright Infringement In ](scope_intrinsic_semantic_space_control_for_mitigating_copyright_infringement_in_.md)**

:   将LLM版权侵权缓解问题重新定义为内在语义空间控制，利用稀疏自编码器(SAE)将隐状态映射到高维稀疏空间，识别版权敏感子空间并在解码时钳制其激活，无需外部过滤器或参数更新即可有效减少版权内容复制，同时保持模型通用能力。

**[Som Directions Are Better Than One Multi-Directional Refusal Suppression In Lang](som_directions_are_better_than_one_multi-directional_refusal_suppression_in_lang.md)**

:   证明LLM的拒绝行为并非由单一方向编码，而是形成低维流形，利用自组织映射（SOM）提取多个拒绝方向并通过贝叶斯优化搜索最优消融组合，在多个模型上超越单方向基线和专用越狱算法。

**[Spark Query-Aware Unstructured Sparsity With Recoverable Kv Cache Channel Prunin](spark_query-aware_unstructured_sparsity_with_recoverable_kv_cache_channel_prunin.md)**

:   提出SparK——一种training-free的KV cache通道级非结构化剪枝方法，通过query-aware的saliency评估选择关键通道+recovery机制恢复被剪枝通道的贡献，在80%剪枝率下性能损失<5%，与token eviction方法正交互补，可额外减少30%+ KV cache存储。

**[Toc Tree-Of-Claims Search With Multi-Agent Language Models](toc_tree-of-claims_search_with_multi-agent_language_models.md)**

:   提出 Tree-of-Claims (ToC) 框架，将专利权利要求编辑建模为结构化搜索问题，通过 MCTS 与 EditorAgent/ExaminerAgent 多智能体协作，在新颖性、范围保持和语义一致性之间联合优化，比零/少样本 LLM 基线平均提升约 8% 综合分。

**[Voices Faces And Feelings Multi-Modal Emotion-Cognition Captioning For Mental He](voices_faces_and_feelings_multi-modal_emotion-cognition_captioning_for_mental_he.md)**

:   提出情感-认知协同多模态描述（ECMC）任务和框架，通过双流BridgeNet从视频、音频、文本中提取情感和认知特征，利用LLaMA生成自然语言描述，为心理健康评估提供可解释的情感-认知画像，显著提升辅助诊断的准确性和可解释性。
