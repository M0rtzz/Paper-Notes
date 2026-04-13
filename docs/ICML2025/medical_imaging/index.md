---
title: >-
  ICML2025 医学图像方向 64篇论文解读
description: >-
  64篇ICML2025 医学图像方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**🧪 ICML2025** · 共 **64** 篇

**[Adios Antibody Development Via Opponent Shaping](adios_antibody_development_via_opponent_shaping.md)**

:   将多智能体强化学习中的对手塑形（Opponent Shaping）引入抗体设计，提出 ADIOS 元学习框架：外层循环优化抗体，内层循环模拟病毒适应性逃逸，使设计出的"塑形抗体"（shapers）不仅能对抗当前病毒变种，还能主动引导病毒向更弱、更易被靶向的方向进化。

**[Aligning Protein Conformation Ensemble Generation With Physical Feedback](aligning_protein_conformation_ensemble_generation_with_physical_feedback.md)**

:   提出 Energy-based Alignment (EBA)，将物理力场的能量反馈融入扩散生成模型的微调过程，通过 Boltzmann 因子加权的分类目标函数对齐生成分布与物理能量景观，在 ATLAS MD 基准上实现蛋白质构象集合生成的 SOTA 性能。

**[Bayesian Inference For Correlated Human Experts And Classifiers](bayesian_inference_for_correlated_human_experts_and_classifiers.md)**

:   提出通用贝叶斯框架来建模相关人类专家和分类器之间的联合标注行为，通过潜在表示捕捉专家间相关性，用模拟推断评估额外查询的效用，在医学分类和图像标注中大幅减少专家查询次数同时保持预测准确率。

**[Boosting Masked Ecg-Text Auto-Encoders As Discriminative Learners](boosting_masked_ecg-text_auto-encoders_as_discriminative_learners.md)**

:   D-BETA 提出了一种融合生成式掩码自编码器与增强判别能力的对比学习框架，通过 ECG-Text Sigmoid (ETS) 损失和最近邻负采样策略 (N3S)，在 ECG-文本跨模态表征学习中显著超越现有方法，在仅用 1% 训练数据的线性探测中平均 AUC 提升 15%，零样本性能提升 2%。

**[Certification For Differentially Private Prediction In Gradient-Based Training](certification_for_differentially_private_prediction_in_gradient-based_training.md)**

:   提出 Abstract Gradient Training (AGT) 框架，通过凸松弛与界传播技术计算训练过程中模型参数的可达集上界，从而利用平滑敏感度机制大幅收紧隐私预测的隐私分析，在医学影像和 NLP 任务上实现比全局敏感度紧数个数量级的隐私界。

**[Cfp-Gen Combinatorial Functional Protein Generation Via Diffusion Language Model](cfp-gen_combinatorial_functional_protein_generation_via_diffusion_language_model.md)**

:   提出 CFP-Gen——一种大规模扩散语言模型，通过注释引导特征调制（AGFM）和残基级控制编码（RCFE）实现多模态功能约束（功能注释 + 序列基序 + 3D 结构）的组合蛋白质生成，F1 分数比 ESM3 提升 30%。

**[Comrecgc Global Graph Counterfactual Explainer Through Common Recourse](comrecgc_global_graph_counterfactual_explainer_through_common_recourse.md)**

:   本文首次形式化了图神经网络的**公共补救 (Common Recourse)** 全局反事实解释问题，证明该问题是 NP-hard 的，并提出了 ComRecGC 算法——通过多头顶点增强随机游走 (Multi-head VRRW) 寻找反事实图，再用 DBScan 聚类提取公共补救，在 NCI1、Mutagenicity、AIDS、Proteins 四个真实数据集上，覆盖率全面超越现有基线 10%–30%。

**[Context Matters Query-Aware Dynamic Long Sequence Modeling Of Gigapixel Images](context_matters_query-aware_dynamic_long_sequence_modeling_of_gigapixel_images.md)**

:   提出 Querent 框架——通过 query-aware 的动态区域重要性评估实现千亿像素全切片图像（WSI）中的高效长程上下文建模，在理论上有界逼近完整自注意力，在 10+ 个 WSI 数据集的生物标志物预测/基因突变预测/癌症分型/生存分析中超越 SOTA。

**[Deepseq High-Throughput Single-Cell Rna Sequencing Data Labeling Via Web Search-](deepseq_high-throughput_single-cell_rna_sequencing_data_labeling_via_web_search-.md)**

:   提出 DeepSeq 流水线，利用大语言模型（尤其是具备实时网络搜索能力的 Agentic GPT-4o）对单细胞RNA测序数据进行自动化细胞类型标注，最高准确率达 82.5%，解决了大规模组学数据标注的吞吐量瓶颈。

**[Deltashap Explaining Prediction Evolutions In Online Patient Monitoring With Sha](deltashap_explaining_prediction_evolutions_in_online_patient_monitoring_with_sha.md)**

:   DeltaSHAP 是一种专为在线患者监护系统设计的可解释AI算法，通过将 Shapley 值适配到时序场景，解释连续预测之间的**变化量**而非绝对预测值，同时提供特征归因的**方向和幅度**，在 MIMIC-III 基准上实现 62% 的解释质量提升和 33% 的计算时间缩减。

**[Designing Cyclic Peptides Via Harmonic Sde With Atom-Bond Modeling](designing_cyclic_peptides_via_harmonic_sde_with_atom-bond_modeling.md)**

:   提出 CpSDE 框架，通过谐波 SDE 生成模型 (AtomSDE) 和残基类型预测器 (ResRouter) 的交替采样，首次实现基于 3D 受体结构的全类型环肽设计，在稳定性和亲和力上超越现有线性肽设计方法。

**[Do Multiple Instance Learning Models Transfer](do_multiple_instance_learning_models_transfer.md)**

:   首次系统评估计算病理学中 MIL 模型的迁移学习能力，发现在 pancancer 数据集上预训练的 MIL 模型能够跨器官、跨任务泛化，以不到 10% 的预训练数据超越自监督 slide foundation model（CHIEF、GigaPath）。

**[Doubly Protected Estimation For Survival Outcomes Utilizing External Controls Fo](doubly_protected_estimation_for_survival_outcomes_utilizing_external_controls_fo.md)**

:   提出一种双重保护（doubly protected）的生存结局估计框架，通过密度比加权校正协变量偏移、DR-Learner检测结局漂移并选择性借用可比外部对照，在保证一致性和效率提升的同时对外部数据异质性具有鲁棒性。

**[Eccdnamamba A Pre-Trained Model For Ultra-Long Eccdna Sequence Analysis](eccdnamamba_a_pre-trained_model_for_ultra-long_eccdna_sequence_analysis.md)**

:   eccDNAMamba 是首个面向环状DNA的双向状态空间编码器，结合BPE分词、环状数据增强和SpanBERT式预训练，在保持线性时间复杂度的同时支持高达200Kbp的超长eccDNA序列建模，在癌症分类和真实eccDNA识别任务上显著超越DNABERT-2、HyenaDNA和Caduceus。

**[Eeg-Language Pretraining For Highly Label-Efficient Clinical Phenotyping](eeg-language_pretraining_for_highly_label-efficient_clinical_phenotyping.md)**

:   本文首创 EEG-语言模型（ELM），在15000份EEG记录和临床报告上训练，结合时间序列裁剪、文本分割和多实例学习策略，首次实现了EEG的零样本分类和跨模态检索，在低标注场景下病理检测性能显著优于纯EEG自监督方法。

**[Efficient Molecular Conformer Generation With So3-Averaged Flow Matching And Ref](efficient_molecular_conformer_generation_with_so3-averaged_flow_matching_and_ref.md)**

:   提出 SO(3)-Averaged Flow 训练目标，通过解析地对旋转群 SO(3) 上所有旋转取平均来消除先验-数据分布间的旋转对齐需求，结合 Reflow+蒸馏实现高质量的少步乃至单步分子构象生成。

**[Efficient Noise Calculation In Deep Learning-Based Mri Reconstructions](efficient_noise_calculation_in_deep_learning-based_mri_reconstructions.md)**

:   提出基于 Jacobian Sketching 的高效方法，通过随机相向量探测 DL 重建网络的 Jacobian 对角元，以无偏估计加速 MRI 重建中的体素级噪声方差，计算和内存需求降低一个数量级以上，与 Monte Carlo 参考相关系数达 99.8%。

**[Elucidating The Design Space Of Multimodal Protein Language Models](elucidating_the_design_space_of_multimodal_protein_language_models.md)**

:   系统性地探索了基于token的多模态蛋白质语言模型（PLM）的设计空间，通过比特级离散建模、几何感知架构、表征对齐和多聚体数据扩展四个维度的创新，将650M参数模型的折叠RMSD从5.52降至2.36，超越3B基线模型，接近专用折叠模型水平。

**[Empower Structure-Based Molecule Optimization With Gradient Guided Bayesian Flow](empower_structure-based_molecule_optimization_with_gradient_guided_bayesian_flow.md)**

:   提出 MolJO 框架，利用贝叶斯流网络（BFN）的连续可微参数空间 $\boldsymbol{\theta}$，实现对分子坐标（连续）和原子类型（离散）的联合梯度引导优化，并设计滑动窗口后向校正策略平衡探索与利用，在 CrossDocked2020 上以 51.3% Success Rate 大幅领先现有方法。

**[Enhancing Statistical Validity And Power In Hybrid Controlled Trials A Randomiza](enhancing_statistical_validity_and_power_in_hybrid_controlled_trials_a_randomiza.md)**

:   提出基于 Fisher 随机化检验（FRT）+ 保形选择性借用（CSB）的混合对照试验推断框架，实现有限样本精确的 I 类错误率控制和模型无关的统计推断，通过自适应阈值最小化 MSE，在保持严格 I 类错误控制的同时提升检验功效。

**[Evaluating Llms Across Multi-Cognitive Levels From Medical Knowledge Mastery To ](evaluating_llms_across_multi-cognitive_levels_from_medical_knowledge_mastery_to_.md)**

:   受 Bloom 分类法启发，提出多认知层次评估框架 MultiCogEval，从知识掌握、综合应用、情景问题解决三个层次评估 LLM 医学能力，发现所有模型性能随认知复杂度增加显著下降，且模型规模在高层次更关键。

**[Flexibility-Conditioned Protein Structure Design With Flow Matching](flexibility-conditioned_protein_structure_design_with_flow_matching.md)**

:   提出 BackFlip（从骨架预测残基级柔性）和 FliPS（以柔性 profile 为条件的 SE(3)-等变 flow matching 模型），首次实现根据目标柔性分布生成具有期望动态特性的蛋白质骨架结构，并通过 300 ns 分子动力学模拟验证。

**[Foundation Models For Clinical Records At Health System Scale](foundation_models_for_clinical_records_at_health_system_scale.md)**

:   提出 GPT-EHR，一种基于下一次就诊事件预测的生成式预训练框架，在 NYU Langone 129 万患者的纵向 EHR 数据上训练 decoder-only Transformer，零样本即可预测痴呆和膝骨关节炎发病，性能媲美全量微调的 BERT 基线，同时揭示并解决了重复事件 token 造成评估指标虚高的关键陷阱。

**[From Token To Rhythm A Multi-Scale Approach For Ecg-Language Pretraining](from_token_to_rhythm_a_multi-scale_approach_for_ecg-language_pretraining.md)**

:   MELP 提出了一种多尺度 ECG-语言预训练模型，通过 Token/Beat/Rhythm 三个层次的跨模态监督信号，结合心脏学专业语言模型预训练，在零样本分类、线性探测和迁移学习中全面超越现有 ECG 自监督和多模态方法。

**[Genmol A Drug Discovery Generalist With Discrete Diffusion](genmol_a_drug_discovery_generalist_with_discrete_diffusion.md)**

:   提出 GenMol，一个基于掩码离散扩散（Masked Discrete Diffusion）的通用分子生成框架，通过非自回归双向并行解码生成 SAFE 序列，并引入片段重掩码（fragment remasking）和分子上下文引导（MCG），用**单一模型**覆盖从头生成、片段约束生成、目标导向 hit 生成和先导化合物优化四大药物发现场景，全面超越此前最优方法。

**[Geometric Generative Modeling With Noise-Conditioned Graph Networks](geometric_generative_modeling_with_noise-conditioned_graph_networks.md)**

:   提出 Noise-Conditioned Graph Networks (NCGNs)，使 GNN 架构根据噪声级别动态调整消息传递的范围和图分辨率：高噪声时用远程连接+低分辨率，低噪声时用局部连接+高分辨率，在 3D 点云、空间转录组和图像生成中均超越固定架构基线。

**[Geometric Representation Condition Improves Equivariant Molecule Generation](geometric_representation_condition_improves_equivariant_molecule_generation.md)**

:   GeoRCG 提出两阶段分子生成框架——先生成低维的几何表示(informative representation)，再以此为条件生成完整分子，在条件分子生成任务上平均提升 50%，同时可将扩散步数从 1000 减少到 100。

**[I2Moe Interpretable Multimodal Interaction-Aware Mixture-Of-Experts](i2moe_interpretable_multimodal_interaction-aware_mixture-of-experts.md)**

:   提出I2MoE——端到端MoE框架用不同交互专家+弱监督交互损失显式建模多模态交互（唯一性/冗余/协同），加上重加权模型提供样本级和数据集级解释，在医疗和通用数据集上一致提升性能和可解释性。

**[Idpa Instance Decoupled Prompt Attention For Incremental Medical Object Detectio](idpa_instance_decoupled_prompt_attention_for_incremental_medical_object_detectio.md)**

:   提出 iDPA 框架，通过实例级 Prompt 生成（IPG）和解耦 Prompt 注意力（DPA）两大模块，在冻结的视觉-语言目标检测模型上实现增量医学目标检测（IMOD），仅训练 1.4% 的参数即在 13 个跨模态医学数据集上全面超越 SOTA。

**[Implementing Adaptations For Vision Autoregressive Model](implementing_adaptations_for_vision_autoregressive_model.md)**

:   本文首次系统实现并评测了Vision AutoRegressive（VAR）模型的各种适配方法（FFT/LoRA/LNTuning）及差分隐私适配，发现VAR在非DP场景下显著超越扩散模型适配（DiffFit），收敛速度更快、计算效率更高，但DP适配性能仍然不佳，揭示了隐私保护图像生成领域的重要研究空白。

**[Improved Off-Policy Reinforcement Learning In Biological Sequence Design](improved_off-policy_reinforcement_learning_in_biological_sequence_design.md)**

:   提出 δ-Conservative Search (δ-CS)，一种面向生物序列设计的新型 off-policy 搜索方法，通过对高分离线序列进行 token 级噪声注入（以概率 δ 随机遮蔽）再用 GFlowNet 策略去噪，并根据代理模型不确定性自适应调节保守程度，在 DNA、RNA、蛋白质和肽设计任务上显著优于现有方法。

**[Langdaug Langevin Data Augmentation For Multi-Source Domain Generalization In Me](langdaug_langevin_data_augmentation_for_multi-source_domain_generalization_in_me.md)**

:   LangDAug 提出用基于能量模型(EBM)的 Langevin 动力学在多源域之间插值生成中间域增强样本，理论证明其诱导正则化效果并约束 Rademacher 复杂度，在眼底和前列腺 MRI 分割上超越 SOTA 域泛化方法。

**[Ldmol A Text-To-Molecule Diffusion Model With Structurally Informative Latent Sp](ldmol_a_text-to-molecule_diffusion_model_with_structurally_informative_latent_sp.md)**

:   提出 LDMol，通过 SMILES 枚举对比学习构建结构感知的潜在空间，在该空间上训练条件扩散模型实现文本到分子生成，首次让扩散模型在文本数据生成任务上超越自回归模型。

**[Leveraging Partial Smiles Validation Scheme For Enhanced Drug Design In Reinforc](leveraging_partial_smiles_validation_scheme_for_enhanced_drug_design_in_reinforc.md)**

:   提出 PSV-PPO 算法，在自回归 SMILES 分子生成的每一步引入部分 SMILES 验证（PSV）真值表，实时惩罚无效 token，在保持分子有效性的同时增强化学空间探索能力。

**[Mastering Multiple-Expert Routing Realizable H-Consistency And Strong Guarantees](mastering_multiple-expert_routing_realizable_h-consistency_and_strong_guarantees.md)**

:   本文为多专家路由(learning to defer)问题提出了新的代理损失函数和高效算法，建立了可实现 H-一致性、H-一致性界和 Bayes 一致性的理论保证，覆盖单阶段和两阶段两种学习场景。

**[Medxpertqa Benchmarking Expert-Level Medical Reasoning And Understanding](medxpertqa_benchmarking_expert-level_medical_reasoning_and_understanding.md)**

:   MedXpertQA 构建了包含 4460 题、覆盖 17 个专科和 11 个身体系统的专家级医学 QA 基准，通过严格的筛选增强和数据合成防泄漏，评估了 18 个主流模型，并专门设计了推理子集用于评估 o1 类推理模型。

**[Mf-Lal Drug Compound Generation Using Multi-Fidelity Latent Space Active Learnin](mf-lal_drug_compound_generation_using_multi-fidelity_latent_space_active_learnin.md)**

:   提出 MF-LAL 框架，将多保真度代理模型与分子生成模型统一到层次化潜空间中，通过主动学习高效整合分子对接（低保真）和结合自由能计算（高保真）两类预言机，生成具有显著更优结合自由能的候选药物分子（平均 ABFE 得分提升约 50%）。

**[Multivariate Conformal Selection](multivariate_conformal_selection.md)**

:   将 Conformal Selection 从单变量响应推广到多变量设定，提出区域单调性 (Regional Monotonicity) 概念，设计距离型 (mCS-dist) 和学习型 (mCS-learn) 两种非一致性分数，在有限样本下保证 FDR 控制并提升选择功效。

**[Network Sparsity Unlocks The Scaling Potential Of Deep Reinforcement Learning](network_sparsity_unlocks_the_scaling_potential_of_deep_reinforcement_learning.md)**

:   本文发现简单的一次性随机剪枝就能解锁深度 RL 的扩展潜力——稀疏网络比配备 SOTA 架构的稠密网络实现更高的参数效率、更强的可塑性保持和更少的梯度干扰。

**[Neural Stochastic Differential Equations On Compact State Spaces Theory Methods ](neural_stochastic_differential_equations_on_compact_state_spaces_theory_methods_.md)**

:   本文提出基于随机生存理论的神经 SDE 参数化方法 (WSP)，确保 SDE 轨迹可证明地约束在紧多面体空间内，具有连续动力学和良好归纳偏置，克服了 chain-rule 方法和反射 SDE 的缺陷。

**[On The Vulnerability Of Applying Retrieval-Augmented Generation Within Knowledge](on_the_vulnerability_of_applying_retrieval-augmented_generation_within_knowledge.md)**

:   本文系统揭示了 RAG 检索系统在知识密集型领域（医疗、法律）中面临的**通用投毒攻击**漏洞，提出"正交增强"性质解释攻击成因，并设计基于分布感知距离的检测防御方法，在几乎所有场景中达到近乎完美的检测率。

**[Out-Of-Distribution Detection Methods Answer The Wrong Questions](out-of-distribution_detection_methods_answer_the_wrong_questions.md)**

:   本文系统论证了当前主流OOD检测方法（基于特征和基于logit）在根本上回答了错误的问题——它们检测的是"特征是否异常"或"模型是否不确定"，而非"输入是否来自不同分布"，并证明了各种常见改进策略也无法解决这一根本性错位。

**[Polyconf Unlocking Polymer Conformation Generation Through Hierarchical Generati](polyconf_unlocking_polymer_conformation_generation_through_hierarchical_generati.md)**

:   提出 PolyConf——首个专为聚合物构象生成设计的层次化生成框架：Phase 1 用掩码自回归模型（MAR）+ 扩散过程在随机顺序下生成各重复单元的局部构象，Phase 2 用 SO(3) 扩散模型生成朝向变换以将局部构象组装为完整聚合物构象；同时构建了首个聚合物构象基准 PolyBench（5万+聚合物，~2000原子/构象），在所有结构和能量指标上均大幅超越现有方法 25%+。

**[Protein Structure Tokenization Benchmarking And New Recipe](protein_structure_tokenization_benchmarking_and_new_recipe.md)**

:   提出 **StructTokenBench**——首个全面评估蛋白质结构分词器 (PST) 的基准框架，从下游有效性、敏感性、独特性和 codebook 利用效率四个维度评估现有方法，并提出 **AminoAseed** 策略通过 codebook 重参数化和 Pareto 最优配置显著改善 VQ-VAE 型分词器的质量（相比 ESM3 提升 6.31%、利用率提升 124%）。

**[Protriever End-To-End Differentiable Protein Homology Search For Fitness Predict](protriever_end-to-end_differentiable_protein_homology_search_for_fitness_predict.md)**

:   提出 Protriever，首个端到端可微的蛋白质同源序列检索框架，将检索器与阅读器联合训练，在蛋白质适应性预测任务上达到序列模型 SOTA，同时比传统 MSA 检索快两个数量级。

**[Raptor Scalable Train-Free Embeddings For 3D Medical Volumes Leveraging Pretrain](raptor_scalable_train-free_embeddings_for_3d_medical_volumes_leveraging_pretrain.md)**

:   提出 Raptor（Random Planar Tensor Reduction），一种完全免训练的方法，利用冻结的 2D 基础模型（DINOv2-L）对 3D 医学体积沿三轴提取视觉 token，再通过随机投影大幅压缩维度，在 10 个医学任务上超越所有需要大规模预训练的 SOTA 方法。

**[Reliable Algorithm Selection For Machine Learning-Guided Design](reliable_algorithm_selection_for_machine_learning-guided_design.md)**

:   提出一种设计算法选择方法，通过将候选设计算法配置的成功判定形式化为多重假设检验问题，结合预测驱动推断（Prediction-Powered Inference）技术校正预测误差，以高概率保证选出在未标注设计分布上满足用户定义成功准则的算法配置。

**[Roll The Dice Look Before You Leap Going Beyond The Creative Limits Of Next-Toke](roll_the_dice_look_before_you_leap_going_beyond_the_creative_limits_of_next-toke.md)**

:   本文设计了一套最小化算法任务来量化语言模型的"创造力极限"，证明 next-token 学习在需要"思维跳跃"的开放式任务中是近视的，而多 token 方法（teacherless 训练、离散扩散模型）以及输入层噪声注入（seed-conditioning）能显著提升生成的多样性与原创性。

**[Safer A Calibrated Risk-Aware Multimodal Recommendation Model For Dynamic Treatm](safer_a_calibrated_risk-aware_multimodal_recommendation_model_for_dynamic_treatm.md)**

:   提出 SAFER 框架，融合结构化 EHR 与临床笔记的多模态信息，通过 KL 散度度量标签不确定性并结合保形推断控制 FDR，为高风险动态治疗推荐提供统计安全保障。

**[Scalable Generation Of Spatial Transcriptomics From Histology Images Via Whole-S](scalable_generation_of_spatial_transcriptomics_from_histology_images_via_whole-s.md)**

:   提出 STFlow，一种基于 flow matching 的生成模型，通过建模整张切片的基因表达联合分布来显式捕获细胞间交互，并采用局部空间注意力实现高效全切片编码，在 HEST-1k 和 STImage-1K4M 上相对最优基线提升 18%。

**[Scalable Non-Equivariant 3D Molecule Generation Via Rotational Alignment](scalable_non-equivariant_3d_molecule_generation_via_rotational_alignment.md)**

:   提出 **RADM (Rotationally Aligned Diffusion Model)**，通过学习样本相关的 SO(3) 旋转变换构建对齐的潜空间，使非等变扩散模型能够有效生成 3D 分子，在生成质量上媲美 SOTA 等变模型，同时提供更好的可扩展性和采样效率。

**[Scssl-Bench Benchmarking Self-Supervised Learning For Single-Cell Data](scssl-bench_benchmarking_self-supervised_learning_for_single-cell_data.md)**

:   提出 scSSL-Bench，一个系统性 benchmark，在 9 个单细胞数据集上评估 19 种自监督学习方法在批次校正、细胞类型注释和缺失模态预测三个下游任务上的表现，揭示了通用 SSL 方法与领域专用方法之间的任务特异性权衡。

**[Sgd Jittering A Training Strategy For Robust And Accurate Model-Based Architectu](sgd_jittering_a_training_strategy_for_robust_and_accurate_model-based_architectu.md)**

:   提出 SGD jittering 训练策略，在模型迭代重建过程中逐步注入零均值高斯噪声，理论证明其同时提升模型鲁棒性和泛化精度，且无需对抗训练的高计算开销。

**[Space Your Genomic Profile Predictor Is A Powerful Dna Foundation Model](space_your_genomic_profile_predictor_is_a_powerful_dna_foundation_model.md)**

:   提出 SPACE（Species-Profile Adaptive Collaborative Experts），论证**监督式基因组图谱预测**比无监督序列预训练能学到更有效的 DNA 表征，并通过物种感知 MoE 编码器和双门控解码器在 18 项 NT 下游任务中 11 项 SOTA。

**[Steering Protein Language Models](steering_protein_language_models.md)**

:   首次将LLM中的Activation Steering技术应用于蛋白质语言模型(PLM)：通过编辑内部激活来引导蛋白质生成/优化朝向目标属性，无需额外训练，并提出基于steering vector相异度的突变位点识别算法。

**[Supercharging Graph Transformers With Advective Diffusion](supercharging_graph_transformers_with_advective_diffusion.md)**

:   提出 Advective Diffusion Transformer（AdvDIFFormer），一种物理启发的图Transformer模型，通过结合非局部扩散（全局注意力）和对流（局部消息传递）两种机制，在拓扑分布偏移下具有可证明的泛化误差控制能力，优于仅依赖局部扩散的GNN。

**[The Brains Bitter Lesson Scaling Speech Decoding With Self-Supervised Learning](the_brains_bitter_lesson_scaling_speech_decoding_with_self-supervised_learning.md)**

:   开发神经科学启发的自监督 pretext 任务和异构脑信号处理架构，将 MEG 语音解码扩展至约 400 小时/900 名被试，超越 SOTA 15-27%，首次以非侵入式数据匹配手术级解码性能，并展现跨数据集、跨被试、跨任务的泛化能力。

**[The Disparate Benefits Of Deep Ensembles](the_disparate_benefits_of_deep_ensembles.md)**

:   系统揭示 Deep Ensembles 的"差异化收益效应"：集成带来的性能提升在不同受保护群体间分配不均，往往有利于已优势群体，导致公平性指标恶化；并发现集成成员间的 per-group 预测多样性差异是根本原因，Hardt 后处理可有效缓解。

**[The Four Color Theorem For Cell Instance Segmentation](the_four_color_theorem_for_cell_instance_segmentation.md)**

:   将四色定理引入细胞实例分割，将每个细胞视为"国家"、背景为"海洋"，用仅 4 类语义分割替代实例分割，并设计渐进训练策略和编码变换方法解决四色编码的非唯一性问题，在多种成像模式上达到 SOTA 性能同时大幅降低模型复杂度。

**[Training Flexible Models Of Genetic Variant Effects From Functional Annotations](training_flexible_models_of_genetic_variant_effects_from_functional_annotations.md)**

:   利用现代快速线性代数技术训练大规模神经网络预测基因变异效应——将连锁不平衡(LD)矩阵做带状近似+小批量训练，首次在百万变异规模上优化全似然而非摘要统计量，发现更大模型只在全似然下才有效。

**[Training Flexible Models Of Genetic Variant Effects From Functional Annotations ](training_flexible_models_of_genetic_variant_effects_from_functional_annotations_.md)**

:   本文提出 DeepWAS（Deep genome Wide Association Studies），利用现代快速线性代数技术（带状矩阵近似 + 迭代求解）解决 GWAS 中大规模 LD 矩阵求逆的计算瓶颈，首次实现用大规模神经网络最大化全似然来训练功能注释驱动的遗传变异效应预测模型，且发现只有在全似然训练下（而非传统 summary statistics 拟合）更大的模型才能带来更好的性能。

**[Unimomo Unified Generative Modeling Of 3D Molecules For De Novo Binder Design](unimomo_unified_generative_modeling_of_3d_molecules_for_de_novo_binder_design.md)**

:   提出 UniMoMo，首个统一小分子、肽和抗体三类分子的 3D binder 设计框架，使用“块图”作为统一表示、迭代全原子自编码器压缩潜空间、E(3)-等变扩散模型生成，在三个基准上超越领域特定模型。

**[Unisim A Unified Simulator For Time-Coarsened Dynamics Of Biomolecules](unisim_a_unified_simulator_for_time-coarsened_dynamics_of_biomolecules.md)**

:   UniSim提出首个跨域（小分子/肽链/蛋白质）全原子时间粗化动力学模拟器，通过多头预训练学习统一原子表示+随机插值学习状态转移+力引导模块适配不同化学环境。

**[Weisfeiler And Leman Go Gambling Why Expressive Lottery Tickets Win](weisfeiler_and_leman_go_gambling_why_expressive_lottery_tickets_win.md)**

:   首次从理论上将 GNN 的表达力（Weisfeiler-Leman 测试）与彩票假说（LTH）联系起来，提出并证明了强表达力彩票假说（SELTH），证明稀疏初始化的 GNN 中存在保持 1-WL 表达力的可训练子网络，且表达力更强的稀疏初始化更可能成为"中奖彩票"，同时展示了不当剪枝导致的不可恢复表达力损失在药物发现等场景中的严重后果。
