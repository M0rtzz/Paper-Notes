---
title: >-
  AAAI2026 1341篇论文解读
description: >-
  1341篇AAAI2026论文深度解读，每篇5分钟读懂核心思想，含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 AAAI2026 论文笔记

**1341** 篇论文解读，覆盖 **46** 个领域。

<div class="conf-index" markdown>

---

## 🏥 医学图像 { #medical_imaging }

**[A Disease-Aware Dual-Stage Framework For Chest X-Ray Report ](medical_imaging/a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)**

:   提出一种两阶段疾病感知框架，通过学习14个与病理类别对应的疾病感知语义token（DASTs）实现显式的疾病表征，再利用疾病-视觉注意力融合（DVAF）和双模态相似性检索（DMSR）机制辅助LLM生成临床准确的胸部X光报告，在CheXpert Plus、IU X-Ray和MIMIC-CXR三个数据集上取得SOTA。

**[A Principle-Driven Adaptive Policy For Group Cognitive Stimu](medical_imaging/a_principle-driven_adaptive_policy_for_group_cognitive_stimu.md)**

:   针对老年认知障碍患者的群体认知刺激治疗（CST）场景，提出GCSD系统：通过多说话人上下文控制、动态参与者状态建模（soft prompt）、认知刺激注意力损失和多维奖励策略优化四个模块，基于Qwen-2.5-3B微调，在500+小时真实粤语CST对话和1万+模拟对话上训练，BLEU-4达27.93超越GPT-4o等大模型，A/B测试胜率50% vs GPT-4o的39%。

**[Advancing Safe Mechanical Ventilation Using Offline Rl With ](medical_imaging/advancing_safe_mechanical_ventilation_using_offline_rl_with_.md)**

:   针对ICU机械通气（MV）设置优化问题，提出混合动作空间的离线RL方法（HybridIQL/HybridEDAC），避免传统离散化导致的分布偏移，同时引入基于无通气天数（VFD）和生理参数安全范围的临床对齐奖励函数，通过多目标优化选择最优奖励，将可优化的通气参数从2-3个扩展到6个，HybridIQL在性能和策略覆盖率间取得最佳平衡。

**[Ambiguity-Aware Truncated Flow Matching For Ambiguous Medica](medical_imaging/ambiguity-aware_truncated_flow_matching_for_ambiguous_medica.md)**

:   提出 ATFM 框架，通过数据层级推理范式将预测精度和多样性解耦到分布级和样本级分别优化，结合高斯截断表示（GTR）和分割流匹配（SFM）两个模块，在模糊医学图像分割任务中同时提升预测的精度、保真度和多样性。

**[Apo2Mol 3D Molecule Generation Via Dynamic Pocket-Aware Diff](medical_imaging/apo2mol_3d_molecule_generation_via_dynamic_pocket-aware_diff.md)**

:   提出Apo2Mol，一个基于扩散的全原子框架，从蛋白质apo（未结合）构象出发，同时生成3D配体分子和对应的holo（结合态）口袋构象，使用24K实验解析的apo-holo结构对训练，在结合亲和力（Vina min -7.86）和药物类似性上达到SOTA。

**[Bayesian Meta-Analyses Could Be More A Case Study In Trial Of Labor After A Cesa](medical_imaging/bayesian_meta-analyses_could_be_more_a_case_study_in_trial_of_labor_after_a_cesa.md)**

:   提出一种层次贝叶斯 meta-analysis 方法，通过对未记录的决策变量（Bishop 分数）建模为截断隐变量，纠正传统固定效应 meta-analysis 中因忽略混杂因子而导致的偏差结论，在 TOLAC（剖宫产后试产）场景中证明机械扩张与 Pitocin 无显著差异。

**[Bica Effective Biomedical Dense Retrieval With Citation-Aware Hard Negatives](medical_imaging/bica_effective_biomedical_dense_retrieval_with_citation-aware_hard_negatives.md)**

:   提出利用 PubMed 引文链构建多跳语义图并进行随机游走的 hard negative 挖掘方法，仅用 20k 训练样本和极少微调步数，即让 33M/110M 小模型在 BEIR 和 LoTTE 上超越数十亿参数的检索基线。

**[Bidirectional Channel-Selective Semantic Interaction For Semi-Supervised Medical](medical_imaging/bidirectional_channel-selective_semantic_interaction_for_semi-supervised_medical.md)**

:   提出 BCSI 框架，通过通道选择路由器动态筛选关键特征通道，在标注和未标注数据流之间进行双向通道级交互，结合语义-空间扰动的弱到强一致性学习，显著提升半监督医学图像分割性能。

**[Bridging Vision And Language For Robust Context-Aware Surgical Point Tracking Th](medical_imaging/bridging_vision_and_language_for_robust_context-aware_surgical_point_tracking_th.md)**

:   提出首个大规模多模态手术点追踪数据集 VL-SurgPT，结合视觉坐标与文本状态描述，并设计文本引导追踪方法 TG-SurgPT，通过语义信息显著提升复杂手术场景下的追踪精度和鲁棒性。

**[Cd-Dpe Dual-Prompt Expert Network Based On Convolutional Dictionary Feature Deco](medical_imaging/cd-dpe_dual-prompt_expert_network_based_on_convolutional_dictionary_feature_deco.md)**

:   提出 CD-DPE 网络，通过迭代卷积字典特征解耦模块（CD-FDM）将多对比度 MRI 特征分离为跨对比度共有和模态特有成分，再利用双提示特征融合专家模块（DP-FFEM）进行自适应融合重建，在多个公开数据集上超越现有 SOTA 方法。

**[Clicare Grounding Large Language Models In Clinical Guidelines For Decision Supp](medical_imaging/clicare_grounding_large_language_models_in_clinical_guidelines_for_decision_supp.md)**

:   提出 CliCARE 框架，将非结构化的纵向癌症电子病历（EHR）转化为时序知识图谱（TKG），并与临床指南知识图谱对齐融合，为 LLM 提供循证依据的临床决策支持，同时设计了与专家评估高度相关的 LLM-as-a-Judge 评估协议。

**[Coarse-To-Fine Open-Set Graph Node Classification With Large Language Models](medical_imaging/coarse-to-fine_open-set_graph_node_classification_with_large_language_models.md)**

:   提出 Coarse-to-Fine Classification (CFC) 框架，利用 LLM 的零样本推理能力为图节点开放集分类提供语义化 OOD 样本和潜在 OOD 标签空间，实现不仅检测 OOD 还能将其分类到具体未知类别的能力。

**[Cocolit Controlnet-Conditioned Latent Image Translation For Mri To Amyloid Pet S](medical_imaging/cocolit_controlnet-conditioned_latent_image_translation_for_mri_to_amyloid_pet_s.md)**

:   提出 CoCoLIT 框架，基于 ControlNet 条件化的潜在扩散模型，从结构 MRI 合成淀粉样蛋白 PET 图像，通过加权图像空间损失（WISL）和潜在平均稳定化（LAS）显著超越现有方法。

**[Constrained Best Arm Identification With Tests For Feasibility](medical_imaging/constrained_best_arm_identification_with_tests_for_feasibility.md)**

:   提出带可行性约束的最优臂识别新框架，允许决策者分别测试臂的性能或可行性约束，设计了渐近最优算法，可自适应地选择通过性能或可行性中更容易的方式淘汰次优臂。

**[Cross-Sample Augmented Test-Time Adaptation For Personalized Intraoperative Hypo](medical_imaging/cross-sample_augmented_test-time_adaptation_for_personalized_intraoperative_hypo.md)**

:   提出 CSA-TTA 框架，通过跨样本库构建、粗到细检索和多任务优化，在测试时从其他患者数据中检索低血压事件信号来增强个性化术中低血压预测。

**[Decoding With Structured Awareness Integrating Directional Frequency-Spatial And](medical_imaging/decoding_with_structured_awareness_integrating_directional_frequency-spatial_and.md)**

:   提出面向医学图像分割的新型解码器框架，包含三个模块：方向感知的自适应交叉融合注意力（ACFA）、空间-频率-小波三分支融合注意力（TFFA）和结构感知多尺度掩码模块（SMMM），在多个基准数据集上超越现有方法。

**[Deepgb-Tb A Risk-Balanced Cross-Attention Gradient-Boosted Convolutional Network](medical_imaging/deepgb-tb_a_risk-balanced_cross-attention_gradient-boosted_convolutional_network.md)**

:   提出 DeepGB-TB，一个结合轻量级1D-CNN（处理咳嗽音频）和梯度提升决策树（处理人口统计特征）的多模态TB筛查系统，通过双向交叉注意力（CM-BCA）模拟临床推理过程融合异构数据，配合风险平衡损失（TRBL）最小化漏诊，在7国数据集上达到 AUROC 0.903，可在手机上离线实时运行。

**[Denas-Vit Data Efficient Nas-Optimized Vision Transformer For Ultrasound Image S](medical_imaging/denas-vit_data_efficient_nas-optimized_vision_transformer_for_ultrasound_image_s.md)**

:   提出 DeNAS-ViT，首次将 NAS 应用于 ViT 的 Token 级搜索实现超声图像分割的多尺度特征提取优化，并设计基于 NAS 约束的半监督学习框架（网络独立性损失+层次对比损失+阶段式优化），在有限标注数据下达到 SOTA。

**[Dia-Gnostic Vlvae Disentangled Alignment-Constrained Vision Language Variational](medical_imaging/dia-gnostic_vlvae_disentangled_alignment-constrained_vision_language_variational.md)**

:   提出 DiA-gnostic VLVAE，通过视觉-语言混合专家VAE学习三因子潜空间（视觉特有/语言特有/共享），配合正交性+对比对齐的双约束实现解纠缠，使模型在临床上下文缺失时仍能生成可靠的放射学报告，在 IU X-Ray 和 MIMIC-CXR 上达到竞争性 BLEU@4。

**[Divide Conquer And Unite Hierarchical Style-Recalibrated Prototype Alignment For](medical_imaging/divide_conquer_and_unite_hierarchical_style-recalibrated_prototype_alignment_for.md)**

:   针对联邦医学图像分割中的"层间风格偏差累积"和"上下文表征不完整"两大挑战，提出FedBCS框架：通过频域自适应风格重校准（FSR）构建领域不变原型，并设计上下文感知的双层原型对齐（CDPA）融合编解码器多层级语义，在组织核分割和前列腺MRI分割任务上达到SOTA。

**[Do Large Language Models Think Like The Brain Sentence-Level Evidences From Laye](medical_imaging/do_large_language_models_think_like_the_brain_sentence-level_evidences_from_laye.md)**

:   本文通过对比14个公开LLM的逐层表示与人类被试听自然叙事时的fMRI数据，在句子级别系统地研究了LLM与人脑语言处理的对齐程度，发现中间层最对齐、指令微调显著增强对齐、且存在与经典神经语言学理论一致的半球偏侧化模式。

**[Dual-Path Knowledge-Augmented Contrastive Alignment Network For Spatially Resolv](medical_imaging/dual-path_knowledge-augmented_contrastive_alignment_network_for_spatially_resolv.md)**

:   提出 DKAN，一个双路径知识增强对比对齐网络，通过整合外部基因数据库的语义信息作为跨模态协调器，结合统一的一阶段对比学习范式和自适应加权机制，从病理组织切片图像（H&E WSI）预测空间分辨率的基因表达，在三个公开ST数据集上全面超越SOTA。

**[Dualfete Revisiting Teacher-Student Interactions From A Feedback Perspective For](medical_imaging/dualfete_revisiting_teacher-student_interactions_from_a_feedback_perspective_for.md)**

:   在教师-学生半监督学习框架中引入反馈机制，让学生能将伪标签引导的更新是否与有标签数据方向一致的信息反馈给教师，并在双教师架构中进一步增强反馈动态性，有效遏制了医学图像分割中的错误累积和确认偏差。

**[Efficient Chromosome Parallelization For Precision Medicine Genomic Workflows](medical_imaging/efficient_chromosome_parallelization_for_precision_medicine_genomic_workflows.md)**

:   提出三种互补的染色体级基因组并行化调度方案——静态调度（优化处理顺序）、动态调度（背包问题式批处理+在线RAM预测）和符号回归RAM预测器，在模拟和真实精准医学流水线中显著降低了内存溢出和执行时间。

**[Egoems A High-Fidelity Multimodal Egocentric Dataset For Cognitive Assistance In](medical_imaging/egoems_a_high-fidelity_multimodal_egocentric_dataset_for_cognitive_assistance_in.md)**

:   发布首个高保真多人多模态自我中心EMS数据集，包含233个试验20小时视频、9项干预67个关键步骤标注，提供三个基准任务（步骤分类/在线分割/CPR质量估计）推动EMS认知协助系统开发。

**[Error Correction In Radiology Reports A Knowledge Distillation-Based Multi-Stage](medical_imaging/error_correction_in_radiology_reports_a_knowledge_distillation-based_multi-stage.md)**

:   提出了一种**分阶段推理 + 双知识注入**框架，将放射学报告的错误校正分解为检测→定位→纠正三个阶段，结合**医学知识图谱蒸馏（MKGD）** 和**外部知识检索（EXKR）**，在 6 个 LLM 架构上实现了高达 **31.56% 的错误检测准确率提升**和 **37.4% 的处理时间减少**。

**[Experience With Single Domain Generalization In Real World Medical Imaging Deplo](medical_imaging/experience_with_single_domain_generalization_in_real_world_medical_imaging_deplo.md)**

:   提出DL+EKE框架，将领域不变的专家知识与深度学习集成，解决医学影像中稀有类（rare class）的单域泛化（SDG）问题，在糖尿病视网膜病变分级、rs-fMRI癫痫灶定位和应激心电图CAD检测三个真实部署场景中显著优于SOTA SDG方法。

**[Expert-Guided Prompting And Retrieval-Augmented Generation For Emergency Medical](medical_imaging/expert-guided_prompting_and_retrieval-augmented_generation_for_emergency_medical.md)**

:   构建首个EMS急救领域多选QA数据集EMSQA（24.3K题、10个临床主题、4个认证等级），提出Expert-CoT和ExpertRAG框架将领域专业属性注入LLM推理与检索，比标准RAG最高提升4.59%准确率。

**[Fane Towards Fine-Grained Cross-Modal Contrast With False-Negative Reduction And](medical_imaging/fane_towards_fine-grained_cross-modal_contrast_with_false-negative_reduction_and.md)**

:   FaNe 提出了一个语义增强的医学视觉-语言预训练框架，通过语义感知正样本挖掘、文本条件稀疏注意力池化和难负例感知对比损失，解决医学 VLP 中的假阴性问题和粗粒度对齐不足问题。

**[Fdp A Frequency-Decomposition Preprocessing Pipeline For Unsupervised Anomaly De](medical_imaging/fdp_a_frequency-decomposition_preprocessing_pipeline_for_unsupervised_anomaly_de.md)**

:   首次系统分析脑 MRI 异常的频域特征，发现病变主要集中在低频分量中，据此提出**频率分解预处理（FDP）**框架，通过可学习先验上下文库重建低频信号来抑制病变同时保留解剖结构，作为即插即用模块可一致提升多种 UAD 基线的检测性能（LDM 上 DICE 提升 17.63%）。

**[Federated Clip For Resource-Efficient Heterogeneous Medical Image Classification](medical_imaging/federated_clip_for_resource-efficient_heterogeneous_medical_image_classification.md)**

:   提出 FedMedCLIP，一种面向医学图像分类的联邦 CLIP 框架，通过冻结 CLIP 编码器 + 掩码特征适配模块（FAM）+ 本地掩码 MLP + 类别级 KL 蒸馏正则化，在保持极低通信/计算开销的同时实现对数据异构场景的鲁棒分类（ISIC2019 上超第二名 8%，比 FedAVG 快 120 倍）。

**[Fia-Edit Frequency-Interactive Attention For Efficient And High-Fidelity Inversi](medical_imaging/fia-edit_frequency-interactive_attention_for_efficient_and_high-fidelity_inversi.md)**

:   提出 FIA-Edit，一个基于频域交互注意力的无反转（inversion-free）文本引导图像编辑框架，通过频率表示交互（FRI）模块在自注意力中进行源/目标特征的频域融合，以及特征注入（FIJ）模块在交叉注意力中显式引入源图像特征，在保持背景高保真度的同时实现精确语义编辑，并首次将通用图像编辑方法应用于临床手术出血图像增强。

**[Fine-Tuned Llms Know They Dont Know A Parameter-Efficient Approach To Recovering](medical_imaging/fine-tuned_llms_know_they_dont_know_a_parameter-efficient_approach_to_recovering.md)**

:   揭示了 SFT 导致 LLM 不诚实的根源是**自我表达能力受损**（而非自我认知被破坏），基于此提出 HCNR 框架，通过 Fisher 信息识别诚实关键神经元并恢复到预训练状态 + Hessian 引导补偿，仅用 256 条数据和 20% 参数即可恢复 33.25% 的诚实性，实现 2.23 倍以上加速。

**[From Policy To Logic For Efficient And Interpretable Coverage Assessment](medical_imaging/from_policy_to_logic_for_efficient_and_interpretable_coverage_assessment.md)**

:   本文提出一种神经符号方法，通过覆盖感知检索器（coverage-aware retriever）与基于PyKnow的符号规则推理相结合，帮助人类审查员高效、可解释地评估医疗CPT代码是否被保险政策覆盖，在推理成本降低44%的同时F1提升4.5%。

**[Funkan Functional Kolmogorov-Arnold Network For Medical Image Enhancement And Se](medical_imaging/funkan_functional_kolmogorov-arnold_network_for_medical_image_enhancement_and_se.md)**

:   本文将 Kolmogorov-Arnold 表示定理从有限维标量空间推广到函数空间（Hilbert 空间），提出 FunKAN 框架，通过在 Hermite 基函数上进行 Fourier 展开来学习内函数，保留了图像数据的空间结构，在 MRI 增强和三个医学分割任务上均超越已有 KAN 变体。

**[G2Lfrom Giga-Scale To Cancer-Specific Large-Scale Pathology Foundation Models Vi](medical_imaging/g2lfrom_giga-scale_to_cancer-specific_large-scale_pathology_foundation_models_vi.md)**

:   本文提出 G2L（Giga-to-Large）蒸馏框架，仅用 1K 张病理切片将 19 亿参数的 giga-scale 病理基础模型（H-optimus-0）的知识蒸馏到 3 亿参数的 large-scale 模型（Hibou-L），在多个癌症特异性下游任务上达到甚至超越教师模型和更大模型的性能。

**[Gem Generative Entropy-Guided Preference Modeling For Few-Shot Alignment Of Llms](medical_imaging/gem_generative_entropy-guided_preference_modeling_for_few-shot_alignment_of_llms.md)**

:   GEM 提出了一种生成式熵引导偏好建模方法，通过认知过滤（基于熵的 CoT 评分）和 SEGA 算法（自评估组优势策略优化），在仅 3000 个偏好对的低资源场景下实现高效的 LLM 对齐。

**[Giim Graph-Based Learning Of Inter- And Intra-View Dependencies For Multi-View M](medical_imaging/giim_graph-based_learning_of_inter-_and_intra-view_dependencies_for_multi-view_m.md)**

:   提出基于多异构图（MHG）的GIIM框架，通过图结构同时建模病灶间的视图内依赖和视图间动态变化，并引入四种缺失视图表示策略，在肝脏CT、乳腺X线和乳腺MRI三种模态上显著超越现有多视图方法。

**[Gp-Molformer-Sim Test Time Molecular Optimization Through Contextual Similarity ](medical_imaging/gp-molformer-sim_test_time_molecular_optimization_through_contextual_similarity_.md)**

:   提出 GP-MoLFormer-Sim，一种无需训练的测试时分子生成引导方法：利用化学语言模型（GP-MoLFormer）自身的上下文嵌入计算与目标分子的相似度，在自回归解码时动态调整logits来引导生成，结合遗传算法（GP-MoLFormer-Sim+GA）后在PMO基准的23个任务上平均排名第2，且在黑盒oracle设定下优于依赖GPT-4的MOLLEO。

**[Grover Graph-Guided Representation Of Omics And Vision With Expert Regulation Fo](medical_imaging/grover_graph-guided_representation_of_omics_and_vision_with_expert_regulation_fo.md)**

:   提出空间多组学框架GROVER，通过KAN-GCN编码器捕获非线性空间-特征依赖、spot-feature-pair对比学习对齐异构模态、以及自适应混合专家（MoE）动态路由过滤低质量信号，在四个真实空间组学数据集上实现了优于现有方法的聚类性能。

**[Guidegen A Text-Guided Framework For Paired Full-Torso Anatomy And Ct Volume Gen](medical_imaging/guidegen_a_text-guided_framework_for_paired_full-torso_anatomy_and_ct_volume_gen.md)**

:   GuideGen 提出了一个仅需文本输入的可控框架，通过分类扩散模型合成全躯干解剖掩码，结合解剖感知高动态范围自编码器和潜在特征生成器，生成配对的全躯干 CT 体积，为下游分割任务提供高质量合成训练数据。

**[Hierarchical Schedule Optimization For Fast And Robust Diffusion Model Sampling](medical_imaging/hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)**

:   HSO 提出了一种层次化调度优化器，通过双层优化框架（上层全局搜索最优初始化策略 + 下层局部优化调度精炼），在仅 8 秒一次性优化代价下实现扩散模型极低 NFE 下的 SOTA 免训练采样质量。

**[Hifusion Hierarchical Intra-Spot Alignment And Regional Context Fusion For Spati](medical_imaging/hifusion_hierarchical_intra-spot_alignment_and_regional_context_fusion_for_spati.md)**

:   提出 HiFusion 框架，通过层次化 spot 内建模（HISM）和上下文感知跨尺度融合（CCF）两个互补模块，从 H&E 染色全切片图像中准确预测空间基因表达，在两个基准数据集的 2D 切片交叉验证和 3D 样本特异性评估中均达到 SOTA。

**[Human-In-The-Loop Interactive Report Generation For Chronic Disease Adherence](medical_imaging/human-in-the-loop_interactive_report_generation_for_chronic_disease_adherence.md)**

:   本文设计了一个"医生在回路"的交互界面，将 AI 限定于数据组织和草稿生成角色，通过单页面编辑器、图表-文本配对和自动紧急度分级，实现了高效且可问责的慢性病依从性报告生成。试点研究揭示了一个"问责悖论"：即便 AI 生成质量达到了医生手动撰写基线水平，审阅时间仍无法显著减少，因为临床责任要求完整核验。

**[Intervention Efficiency And Perturbation Validation Framework Capacity-Aware And](medical_imaging/intervention_efficiency_and_perturbation_validation_framework_capacity-aware_and.md)**

:   针对临床小样本、类别不平衡场景下多个模型性能相近（Rashomon Effect）导致的模型选择困难，提出 **Intervention Efficiency (IE)** 容量感知评估指标和 **Perturbation Validation Framework (PVF)** 鲁棒性验证框架，联合实现资源约束下的可靠模型选择。

**[Investigating Data Pruning For Pretraining Biological Foundation Models At Scale](medical_imaging/investigating_data_pruning_for_pretraining_biological_foundation_models_at_scale.md)**

:   提出一个基于影响函数的后验数据剪枝框架，通过子集自影响估计（Subset-Based Self-Influence）和两种选择策略（Top-k Influence 和 Coverage-Centric Influence），在超过 99% 的极端剪枝率下，用仅 0.2M 序列预训练的 RNA-FM 在多项下游任务上媲美甚至超越用 23M 序列训练的完整模型，揭示了生物序列数据集的巨大冗余性。

**[Learning Cell-Aware Hierarchical Multi-Modal Representations](medical_imaging/learning_cell-aware_hierarchical_multi-modal_representations.md)**

:   本文提出 CHMR 框架，通过结构感知传播解决生物模态缺失问题，引入树状向量量化(Tree-VQ)建模分子-细胞-基因间的层次依赖关系，在9个基准728个任务上分类提升3.6%、回归提升17.2%，实现鲁棒的细胞感知分子表征学习。

**[Learning With Preserving For Continual Multitask Learning](medical_imaging/learning_with_preserving_for_continual_multitask_learning.md)**

:   提出 Learning with Preserving（LwP）框架，通过动态加权距离保持（DWDP）损失函数维护共享表示空间的几何结构，在无需回放缓冲的条件下解决持续多任务学习（CMTL）中的灾难性遗忘问题，在 BDD100k、CelebA、PhysiQ 等基准上显著超越现有持续学习方法，并且是唯一超越单任务学习基线的方法。

**[Lungnoduleagent A Collaborative Multi-Agent System For Precision Diagnosis Of Lu](medical_imaging/lungnoduleagent_a_collaborative_multi-agent_system_for_precision_diagnosis_of_lu.md)**

:   提出 LungNoduleAgent，首个面向肺结节分析的协作式多智能体系统，通过"Nodule Spotter + Simulated Radiologist + Doctor Agent System"三阶段流水线模拟临床工作流，在 CT 报告生成和恶性分级任务上大幅超越 GPT-4o、Claude 3.7 Sonnet 等主流 VLM 及 MedAgent-Pro 等医学智能体。

**[Maisi-V2 Accelerated 3D High-Resolution Medical Image Synthesis With Rectified F](medical_imaging/maisi-v2_accelerated_3d_high-resolution_medical_image_synthesis_with_rectified_f.md)**

:   提出 MAISI-v2，首个将 Rectified Flow 引入 3D 医学图像合成的框架，通过替换 DDPM 实现 33 倍加速，并设计区域特异性对比损失增强对肿瘤等小区域条件的忠实度，在下游肿瘤分割任务中验证了合成数据的增强价值。

**[Mapi-Gnn Multi-Activation Plane Interaction Graph Neural Network For Multimodal ](medical_imaging/mapi-gnn_multi-activation_plane_interaction_graph_neural_network_for_multimodal_.md)**

:   提出 MAPI-GNN，通过多维特征判别器在语义子空间中动态构建多个激活图，再经层次化融合网络聚合样本内和样本间关系，在前列腺癌和冠心病两个多模态诊断任务上显著超越现有 SOTA（PI-CAI 上 ACC 0.9432，AUC 0.9838）。

**[Measuring Stability Beyond Accuracy In Small Open-Source Medical Large Language ](medical_imaging/measuring_stability_beyond_accuracy_in_small_open-source_medical_large_language_.md)**

:   系统评估了6个小型开源医学LLM（<10B参数）在儿科内分泌领域的表现，揭示仅靠准确率不足以衡量模型可靠性：语义无关的提示微调导致模型输出显著变化（Stuart-Maxwell p<10⁻⁴），高一致性不等于正确，甚至CUDA版本差异也能引发统计显著的输出偏移。

**[Medeyes Learning Dynamic Visual Focus For Medical Progressive Diagnosis](medical_imaging/medeyes_learning_dynamic_visual_focus_for_medical_progressive_diagnosis.md)**

:   提出 MedEyes，一个混合策略强化学习框架，通过注视引导推理导航器（GRN）模拟临床医生"扫描-钻探"的诊断视觉搜索模式，结合置信度值采样器（CVS）和双流 GRPO 优化，实现动态视觉聚焦的医学渐进式诊断推理，在五个医学 VQA 基准上平均提升 8.5pp。

**[Mergedna Context-Aware Genome Modeling With Dynamic Tokenization Through Token M](medical_imaging/mergedna_context-aware_genome_modeling_with_dynamic_tokenization_through_token_m.md)**

:   提出 MergeDNA，通过可微分的 Token Merging 机制实现上下文感知的动态 DNA tokenization，结合层次化 autoencoder Transformer 和自适应 masked token modeling 预训练，在多个基因组 benchmark 上取得 SOTA。

**[Mindcross Fast New Subject Adaptation With Limited Data For Cross-Subject Video ](medical_imaging/mindcross_fast_new_subject_adaptation_with_limited_data_for_cross-subject_video_.md)**

:   提出 MindCross，一个跨被试脑解码框架，通过共享编码器学习被试无关信息 + N个特有编码器学习被试相关信息，配合快速校准阶段和 Top-K 协作解码模块，仅用一个模型在 fMRI/EEG-to-video 基准上实现与被试独立模型可比的性能，且新被试适应仅需极少数据和极短时间（~1秒 vs 基线5-17秒）。

**[Mirage Scaling Test-Time Inference With Parallel Graph-Retrieval-Augmented Reaso](medical_imaging/mirage_scaling_test-time_inference_with_parallel_graph-retrieval-augmented_reaso.md)**

:   提出MIRAGE框架，将传统的线性推理链扩展为并行多链推理范式，结合结构化医学知识图谱的自适应检索（邻域扩展和多跳遍历），通过跨链验证解决矛盾，在三个医学QA基准上持续优于GPT-4o、ToT和Search-o1等方法。

**[Mirnet Integrating Constrained Graph-Based Reasoning With Pre-Training For Diagn](medical_imaging/mirnet_integrating_constrained_graph-based_reasoning_with_pre-training_for_diagn.md)**

:   提出MIRNet框架，将自监督掩码自编码器（MAE）预训练与约束感知的图注意力网络（GAT）推理相结合，用于舌象多标签诊断，并发布包含4000张图像22个标签的TongueAtlas-4K基准数据集，Macro Recall提升77.8%、Macro-F1提升33.2%。

**[Mpa Multimodal Prototype Augmentation For Few-Shot Learning](medical_imaging/mpa_multimodal_prototype_augmentation_for_few-shot_learning.md)**

:   本文提出 MPA 框架，通过 LLM 生成多变体语义描述增强原型的语义信息（LMSE）、层次化多视角数据增强丰富视觉特征（HMA）、以及自适应不确定类吸收器建模类间不确定性（AUCA），在 4 个单域和 6 个跨域小样本学习基准上显著超越现有方法，5-way 1-shot 下单域和跨域分别比次优方法高出 12.29% 和 24.56%。

**[Multivariate Gaussian Representation Learning For Medical Action Evaluation](medical_imaging/multivariate_gaussian_representation_learning_for_medical_action_evaluation.md)**

:   提出 GaussMedAct 框架，将关节运动轨迹建模为多元高斯混合分布并结合笛卡尔-向量双流编码，在自建的 CPREval-6k 数据集上实现 92.1% Top-1 准确率，仅需 ST-GCN 10% 的计算量。

**[Neural Bandit Based Optimal Llm Selection For A Pipeline Of Tasks](medical_imaging/neural_bandit_based_optimal_llm_selection_for_a_pipeline_of_tasks.md)**

:   提出 Sequential Bandits 算法，一种基于神经上下文多臂老虎机的在线学习方法，用于在任务流水线（如"摘要→诊断"）中为每个子任务选择最优 LLM，同时优化准确率和成本，在医学诊断和电信问答两个流水线任务上优于现有 bandit 基线。

**[Note2Chat Improving Llms For Multi-Turn Clinical History Taking Using Medical No](medical_imaging/note2chat_improving_llms_for_multi-turn_clinical_history_taking_using_medical_no.md)**

:   提出 Note2Chat 框架，利用广泛可得的医学笔记（而非稀缺的对话数据）训练 LLMs 进行结构化问诊和诊断，通过笔记驱动的对话生成、三阶段微调策略和单轮推理范式，在信息收集（F1 +16.9）和诊断准确率（Top-1 +21.0）上大幅超越 GPT-4o。

**[Nutriscreener Retrieval-Augmented Multi-Pose Graph Attention Network For Malnour](medical_imaging/nutriscreener_retrieval-augmented_multi-pose_graph_attention_network_for_malnour.md)**

:   提出 NutriScreener，一个结合CLIP视觉编码器、多姿态图注意力网络（GAT）和基于FAISS的检索增强分类/回归模块的框架，通过跨姿态注意力和类别增强检索来实现鲁棒的儿童营养不良检测与人体测量学预测，在AnthroVision等跨大洲数据集上达到0.79 recall和0.82 AUC，临床医生评价准确性4.3/5、效率4.6/5。

**[Pairing-Free Group-Level Knowledge Distillation For Robust Gastrointestinal Lesi](medical_imaging/pairing-free_group-level_knowledge_distillation_for_robust_gastrointestinal_lesi.md)**

:   提出 PaGKD，一个无需配对样本的组级知识蒸馏框架，通过组级原型蒸馏（GKD-Pro，用共享的病变查询Transformer提取模态不变语义原型）和组级密集蒸馏（GKD-Den，用激活图引导的语义关系交叉注意力实现密集空间对齐），突破传统NBI→WLI跨模态蒸馏对配对数据的依赖，在四个临床数据集上AUC分别提升3.3%/1.1%/2.8%/3.2%。

**[Panfoma A Lightweight Foundation Model And Benchmark For Pan-Cancer](medical_imaging/panfoma_a_lightweight_foundation_model_and_benchmark_for_pan-cancer.md)**

:   提出 PanFoMa，一种融合 Transformer 局部建模与 Mamba 全局整合的轻量级混合神经网络，用于泛癌单细胞转录组表示学习；同时构建了覆盖 33 种癌症亚型、350 万+ 细胞的大规模基准数据集 PanFoMaBench。

**[Personalization Of Large Foundation Models For Health Interventions](medical_imaging/personalization_of_large_foundation_models_for_health_interventions.md)**

:   系统性分析大基础模型（LFMs）在个性化健康干预中的四大结构性矛盾，论证 LFMs 无法替代 N-of-1 试验，提出结合 LFMs 假设生成与 N-of-1 试验因果验证的混合框架。

**[Pings-X Physics-Informed Normalized Gaussian Splatting With Axes Alignment For E](medical_imaging/pings-x_physics-informed_normalized_gaussian_splatting_with_axes_alignment_for_e.md)**

:   提出PINGS-X框架，将3D高斯溅射（3DGS）的显式表示思想引入物理信息超分辨率领域，通过归一化高斯溅射（NGS）、轴对齐高斯和高斯合并三项创新，在合成CFD和真实4D Flow MRI数据集上实现了比PINN快一个数量级的训练速度，同时保持更高的超分辨率精度。

**[Priorrg Prior-Guided Contrastive Pre-Training And Coarse-To-Fine Decoding For Ch](medical_imaging/priorrg_prior-guided_contrastive_pre-training_and_coarse-to-fine_decoding_for_ch.md)**

:   PriorRG 提出了一个两阶段胸部X光报告生成框架，通过先验引导的对比预训练对齐临床语境与时空视觉特征，再通过先验感知的粗到细解码逐步融合临床上下文、疾病进展和多层级视觉线索，在 MIMIC-CXR 上实现 BLEU-4 提升 3.6%、F1 提升 3.8%。

**[Propl Universal Semi-Supervised Ultrasound Image Segmentation Via Prompt-Guided ](medical_imaging/propl_universal_semi-supervised_ultrasound_image_segmentation_via_prompt-guided_.md)**

:   提出 ProPL 框架，通过共享视觉编码器 + 提示引导双解码器 + 不确定性驱动伪标签校准，首次实现通用半监督超声图像分割，在 5 个器官 8 个任务上以极少标注数据（1/16）超越全监督方法 5.18% mDice。

**[Protsae Disentangling And Interpreting Protein Language Models Via Semantically-](medical_imaging/protsae_disentangling_and_interpreting_protein_language_models_via_semantically-.md)**

:   提出 ProtSAE，在稀疏自编码器训练中引入语义标注和领域本体知识作为引导信号，解决传统 SAE 的语义纠缠问题，使蛋白质语言模型的隐层特征与生物学概念（分子功能、生物过程、离子结合位点等）精准对齐，同时保持高重建保真度并支持概念级别的生成控制。

**[Provably Minimum-Length Conformal Prediction Sets For Ordinal Classification](medical_imaging/provably_minimum-length_conformal_prediction_sets_for_ordinal_classification.md)**

:   提出 min-CPS 及其正则化变体 min-RCPS，一种模型无关的序数保形预测方法，通过线性时间滑动窗口算法求解每个样本的最小长度预测区间，在保证覆盖率的同时平均减少 15% 的预测集大小，且提供了实例级最优性的理论保证。

**[Pulsemind A Multi-Modal Medical Model For Real-World Clinical Diagnosis](medical_imaging/pulsemind_a_multi-modal_medical_model_for_real-world_clinical_diagnosis.md)**

:   提出 PulseMind 医学多模态诊断模型，包含大规模多轮诊断对话数据集 MediScope、临床对话评估基准 PulseMind Benchmark，以及基于比较的强化策略优化方法 CRPO，在真实临床诊断对话场景中取得优异表现。

**[Q-Fsru Quantum-Augmented Frequency-Spectral Fusion For Medical Visual Question A](medical_imaging/q-fsru_quantum-augmented_frequency-spectral_fusion_for_medical_visual_question_a.md)**

:   提出 Q-FSRU 模型，将医学图像和文本特征转换到频域（FFT）进行融合，并结合量子启发的检索增强生成（Quantum RAG）引入外部医学知识，在 VQA-RAD 数据集上取得 90% 准确率和 0.9541 的 ROC-AUC。

**[Qa-Flora Data-Free Query-Adaptive Fusion Of Loras For Llms](medical_imaging/qa-flora_data-free_query-adaptive_fusion_of_loras_for_llms.md)**

:   提出 qa-FLoRA，一种无需训练数据和训练过程的查询自适应 LoRA 融合方法，通过逐层计算各适配器与基座模型间的 KL 散度来动态确定融合权重，在九个多语言复合任务上显著优于静态融合和无训练基线。

**[Qgshap Quantum Acceleration For Faithful Gnn Explanations](medical_imaging/qgshap_quantum_acceleration_for_faithful_gnn_explanations.md)**

:   提出 QGShap，一种利用量子振幅放大技术加速精确 Shapley 值计算的图神经网络可解释性框架，在保持精确计算（非近似）的同时实现了相对经典 Monte Carlo 方法的二次加速。

**[Radiation-Preserving Selective Imaging For Pediatric Hip Dysplasia A Cross-Modal](medical_imaging/radiation-preserving_selective_imaging_for_pediatric_hip_dysplasia_a_cross-modal.md)**

:   提出一种"超声优先、保辐射"的跨模态选择性成像策略，通过自监督预训练的冻结编码器、测量忠实的轻量头网络和共形预测校准的单侧下界，实现了在发育性髋关节发育不良（DDH）诊断中有据可依地决定何时仅用超声即可、何时需要额外的 X 光检查。

**[Recon-Ipsundrum An Inspectable Recurrent Persistence Loop Agent With Affect-Coup](medical_imaging/recon-ipsundrum_an_inspectable_recurrent_persistence_loop_agent_with_affect-coup.md)**

:   实现ReCoN-Ipsundrum——一个可检查的智能体架构，在ReCoN感觉运动状态机上扩展了Humphrey的ipsundrum递归持续循环和可选的情感代理层，通过行为测试和因果消融实验证明：递归支撑刺激后持续性，情感耦合支撑偏好稳定性、结构化扫描和持久谨慎，并强调行为标记单独不足以归因意识。

**[Refine And Align Confidence Calibration Through Multi-Agent Interaction In Vqa](medical_imaging/refine_and_align_confidence_calibration_through_multi-agent_interaction_in_vqa.md)**

:   提出 AlignVQA，一个基于多智能体辩论的VQA置信度校准框架：专家agent生成候选答案后，通用agent进行结构化辩论（支持论据 vs 反对论据）来修正置信度；同时提出可微分的校准感知损失 AlignCal，通过最小化校准误差上界（UBCE）来训练更校准的agent，在VQARad和ScienceQA上将ECE从0.375降至0.098。

**[Rethinking Bias In Generative Data Augmentation For Medical Ai A Frequency Recal](medical_imaging/rethinking_bias_in_generative_data_augmentation_for_medical_ai_a_frequency_recal.md)**

:   揭示 AI 生成医学图像与真实图像之间的高频频率分布差异是生成式数据增强（GDA）不可靠的关键原因，提出 FreRec（Frequency Recalibration）方法，通过统计高频替换（SHR）和重建式高频映射（RHM）两步实现粗到细的频率分布对齐，作为即插即用的后处理模块显著提升下游医学图像分类性能。

**[Rethinking Surgical Smoke A Smoke-Type-Aware Laparoscopic Video Desmoking Method](medical_imaging/rethinking_surgical_smoke_a_smoke-type-aware_laparoscopic_video_desmoking_method.md)**

:   本文首次将手术烟雾分为扩散烟（Diffusion Smoke）和环境烟（Ambient Smoke）两种类型，提出了第一个烟雾类型感知的腹腔镜视频去烟网络 STANet，包含语义软分割、粗到精解耦和双分支重建三个子网络，并构建了首个包含烟雾类型标注的大规模合成视频去烟数据集 STSVD。

**[S2Drug Bridging Protein Sequence And 3D Structure In Contrastive Representation ](medical_imaging/s2drug_bridging_protein_sequence_and_3d_structure_in_contrastive_representation_.md)**

:   提出 S2Drug，一个两阶段对比学习框架，第一阶段在 ChemBL 大规模数据上用蛋白质序列-配体对比预训练（含双边数据采样策略降噪去冗），第二阶段在 PDBBind 上通过残基级门控模块融合序列与 3D 结构信息并引入结合位点预测辅助任务，在 DUD-E 和 LIT-PCBA 虚拟筛选基准上大幅超越现有方法。

**[Self-Supervised Multiplex Consensus Mamba For General Image Fusion](medical_imaging/self-supervised_multiplex_consensus_mamba_for_general_image_fusion.md)**

:   提出 SMC-Mamba 框架，通过**模态无关特征增强（MAFE）**、**多路共识跨模态 Mamba（MCCM）**和**双层自监督对比学习损失（BSCL）**，实现覆盖红外-可见光、医学、多聚焦、多曝光的通用图像融合，全面超越 SOTA。

**[Semc Structure-Enhanced Mixture-Of-Experts Contrastive Learning For Ultrasound S](medical_imaging/semc_structure-enhanced_mixture-of-experts_contrastive_learning_for_ultrasound_s.md)**

:   提出 SEMC 框架，通过**语义-结构融合模块（SSFM）**对齐浅层结构线索与深层语义表征，结合**混合专家对比识别模块（MCRM）**在多层特征上进行分层对比学习，提升超声标准切面识别的细粒度判别能力，并构建了新的肝脏超声数据集 LP2025。

**[Sim4Seg Boosting Multimodal Multi-Disease Medical Diagnosis Segmentation With Re](medical_imaging/sim4seg_boosting_multimodal_multi-disease_medical_diagnosis_segmentation_with_re.md)**

:   提出医学诊断分割（MDS）任务并构建 M3DS 数据集，设计 Sim4Seg 框架利用 LVLM 隐藏状态的**视觉-语言相似度掩码（RVLS2M）**提示 SAM 进行分割，同时生成诊断思维链，配合测试时缩放策略在分割和诊断上全面超越基线。

**[Small But Mighty Dynamic Wavelet Expert-Guided Fine-Tuning Of Large-Scale Models](medical_imaging/small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)**

:   WEFT 提出了一种基于动态小波专家引导的轻量微调范式，仅需 4.52% 的可训练参数即可将大规模冻结视觉基础模型高效适配到光学遥感图像分割任务，在三个 ORSIs 数据集上超越 21 种 SOTA 方法。

**[Spa Achieving Consensus In Llm Alignment Via Self-Priority Optimization](medical_imaging/spa_achieving_consensus_in_llm_alignment_via_self-priority_optimization.md)**

:   提出 Self-Priority Alignment（SPA），一种全无监督框架，通过字典序优化实现"可信赖优先于有用性"的严格优先级对齐——模型自生成多样响应、自评估、自改进，经双准则去噪构建偏好对，用不确定性加权 SimPO 损失微调，在多个安全基准上同时提升安全性和有用性。

**[Spacrd Multimodal Deep Fusion Of Histology And Spatial Transcriptomics For Cance](medical_imaging/spacrd_multimodal_deep_fusion_of_histology_and_spatial_transcriptomics_for_cance.md)**

:   提出 SpaCRD，一个基于迁移学习的多模态深度融合框架，通过类别正则化变分重建引导的双向交叉注意力融合网络（VRBCA），将组织学图像与空间转录组学数据深度整合，在 23 个配对数据集上跨样本、跨平台/批次实现了癌症组织区域（CTR）检测的 SOTA 性能。

**[Taligndiff Automatic Tooth Alignment Assisted By Diffusion-Based Transformation ](medical_imaging/taligndiff_automatic_tooth_alignment_assisted_by_diffusion-based_transformation_.md)**

:   提出TAlignDiff框架，将基于点云的几何约束回归网络（PRN）与扩散模型辅助的变换矩阵去噪模块（DTMD）统一为一个联合训练框架，通过双向反馈机制在小样本临床数据上实现了优于现有方法的自动牙齿排列效果。

**[Towards Effective And Efficient Context-Aware Nucleus Detection In Histopatholog](medical_imaging/towards_effective_and_efficient_context-aware_nucleus_detection_in_histopatholog.md)**

:   提出一种高效的上下文感知细胞核检测方法，通过聚合历史已访问滑窗的现成特征替代额外裁剪大视野图像块来提供组织上下文，同时利用跨标注策略挖掘周围未标注核样本以增强模型的上下文适应性。

**[Training-Free Policy Violation Detection Via Activation-Space Whitening In Llms](medical_imaging/training-free_policy_violation_detection_via_activation-space_whitening_in_llms.md)**

:   将 LLM 的策略违规检测重构为激活空间中的分布外（OOD）检测问题，提出无需训练的白化方法：对合规激活拟合白化变换，用欧几里得范数作为合规分数，仅需策略文本和少量示例即可部署，在 DynaBench 上达到 86.0% F1，超越微调基线 9.1 个点、LLM-as-Judge 16 个点。

**[Trinitydna A Bio-Inspired Foundational Model For Efficient Long-Sequence Dna Mod](medical_imaging/trinitydna_a_bio-inspired_foundational_model_for_efficient_long-sequence_dna_mod.md)**

:   提出 TrinityDNA，一个生物启发的DNA基础模型，整合三大创新：Groove Fusion模块捕获DNA大小沟槽结构特征、Gated Reverse Complement机制处理双链互补对称性、Sliding Multi-Window Attention实现多尺度长程依赖建模，配合从原核到真核的进化训练策略（ETS），在GUE基准15个任务上平均MCC达0.708（超越2.5B参数的NT），在19个零样本任务上的原核/真核表现均领先，并提出新的CDS标注基准供长序列推理评估。

**[Unleashing The Potential Of Large Language Models For Text-To-Image Generation T](medical_imaging/unleashing_the_potential_of_large_language_models_for_text-to-image_generation_t.md)**

:   提出 ARRA（Autoregressive Representation Alignment）训练框架，通过混合令牌 \<HYBNEXT\> 在训练时将外部视觉基础模型的全局表征蒸馏到自回归 LLM 的隐状态中，无需修改架构即可显著提升 LLM 的文本到图像生成质量。

**[Unsupervised Motion-Compensated Decomposition For Cardiac Mri Reconstruction Via](medical_imaging/unsupervised_motion-compensated_decomposition_for_cardiac_mri_reconstruction_via.md)**

:   提出 MoCo-INR，首次将隐式神经表示（INR）引入运动补偿（MoCo）框架，通过无监督方式实现心脏 MRI 的高质量动态重建，在超高加速因子（20x Cartesian / 69x Non-Cartesian）下显著优于现有无监督方法。

**[Unsupervised Multi-Parameter Inverse Solving For Reducing Ring Artifacts In 3D X](medical_imaging/unsupervised_multi-parameter_inverse_solving_for_reducing_ring_artifacts_in_3d_x.md)**

:   提出 Riner，将 CT 环形伪影去除（RAR）建模为基于物理的多参数逆问题，通过隐式神经表示（INR）联合学习无伪影图像和探测器物理参数，实现无监督且优于有监督 SOTA 方法的 3D CBCT 重建。

**[Vascular Anatomy-Aware Self-Supervised Pre-Training For X-Ray Angiogram Analysis](medical_imaging/vascular_anatomy-aware_self-supervised_pre-training_for_x-ray_angiogram_analysis.md)**

:   提出 VasoMIM，一个针对X射线血管造影的领域特定自监督预训练框架：通过解剖引导的掩码策略优先遮挡血管区域 + 解剖一致性损失保持重建图像的血管拓扑结构，结合构建的最大规模XA-170K预训练数据集，在4个下游任务6个数据集上全面超越通用SSL方法和医学SSL方法（包括在16.9亿图像上预训练的DINOv3）。

**[Virtual Multiplex Staining For Histological Images Using A Marker-Wise Condition](medical_imaging/virtual_multiplex_staining_for_histological_images_using_a_marker-wise_condition.md)**

:   提出基于标记物条件扩散模型的虚拟多重染色框架，通过两阶段训练（标记物条件扩散学习+像素级微调），首次从单张H&E图像生成多达18种不同标记物的多重免疫荧光图像，在HEMIT和Orion-CRC两个公开数据集上全面超越现有方法。

**[Vitaldiagnosis Ai-Driven Ecosystem For 247 Vital Monitoring And Chronic Disease ](medical_imaging/vitaldiagnosis_ai-driven_ecosystem_for_247_vital_monitoring_and_chronic_disease_.md)**

:   提出VitalDiagnosis，一个由LLM驱动的慢性病管理生态系统，通过整合可穿戴设备连续数据与多尺度LLM推理能力，建立包含异常交互式分诊和常规依从性监测的双轨框架，在协作式患者-临床医生工作流中实现从被动监测到主动参与的范式转变。

**[Vmfcoop Towards Equilibrium On A Unified Hyperspherical Manifold For Prompting B](medical_imaging/vmfcoop_towards_equilibrium_on_a_unified_hyperspherical_manifold_for_prompting_b.md)**

:   提出 vMFCoOp 框架，通过在统一超球面流形上反向估计 von Mises-Fisher 分布对齐 LLM 和 CLIP 的语义偏差，实现生物医学 VLM 的鲁棒少样本提示学习。

**[Wdt-Md Wavelet Diffusion Transformers For Microaneurysm Detection In Fundus Imag](medical_imaging/wdt-md_wavelet_diffusion_transformers_for_microaneurysm_detection_in_fundus_imag.md)**

:   提出 WDT-MD 框架，通过噪声编码图像条件化、伪正常模式合成和小波扩散 Transformer 架构，解决眼底图像中微动脉瘤（MA）检测的三大难题：identity mapping、高假阳性和正常特征重建质量差。

---

## 🧊 3D视觉 { #3d_vision }

**[3D-Anc Adaptive Neural Collapse For Robust 3D Point Cloud Re](3d_vision/3d-anc_adaptive_neural_collapse_for_robust_3d_point_cloud_re.md)**

:   将Neural Collapse(NC)机制引入3D点云对抗鲁棒性，用固定的ETF分类头+自适应训练框架(RBL+FDL)构建解耦的特征空间，在ModelNet40上将DGCNN的对抗准确率从27.2%提升到80.9%，超出最佳baseline 34个点。

**[3D-Free Meets 3D Priors Novel View Synthesis From A Single Image With Pretrained](3d_vision/3d-free_meets_3d_priors_novel_view_synthesis_from_a_single_image_with_pretrained.md)**

:   提出将 3D-free 方法（HawkI 风格的 test-time optimization）与 3D-based 先验（Zero123++ 的弱引导图）结合的框架，无需额外 3D 数据或训练即可从单张图片生成指定仰角/方位角的相机控制视图，在复杂场景下 LPIPS、CLIP-Score 等指标全面超越 Zero123++、HawkI 和 Stable Zero123。

**[4Dstr Advancing Generative 4D Gaussians With Spatial-Tempora](3d_vision/4dstr_advancing_generative_4d_gaussians_with_spatial-tempora.md)**

:   提出4DSTR框架，通过基于Mamba的时序关联校正（修正高斯点的尺度和旋转）以及逐帧自适应稠密化与裁剪策略，显著提升4D高斯生成的时空一致性和对快速时序变化的适应能力。

**[Adapt-As-You-Walk Through The Clouds Training-Free Online Te](3d_vision/adapt-as-you-walk_through_the_clouds_training-free_online_te.md)**

:   提出 Uni-Adapter，一种面向3D视觉-语言基础模型(VLFM)的无训练在线测试时适应框架，通过基于聚类的动态原型缓存和图正则化标签平滑来应对分布偏移，在多个3D损坏基准上取得SOTA。

**[Anchords Anchoring Dynamic Sources For Semantically Consiste](3d_vision/anchords_anchoring_dynamic_sources_for_semantically_consiste.md)**

:   揭示 SDS 中源分布是动态演化而非静态的关键问题，提出 AnchorDS，通过将当前渲染图像作为图像条件输入双条件扩散模型来锚定源分布，解决了 SDS 的语义过度平滑和多视角不一致问题，在 T3Bench 上全面超越 SDS/VSD/SDS-Bridge。

**[Anchorhoi Zero-Shot Generation Of 4D Human-Object Interactio](3d_vision/anchorhoi_zero-shot_generation_of_4d_human-object_interactio.md)**

:   提出 AnchorHOI，通过锚点NeRF和锚点关键点两种中间桥梁，分别从图像/视频扩散模型中蒸馏交互先验和运动先验，实现零样本的文本驱动4D人物-物体交互生成，在静态3D和动态4D HOI生成上均超越已有方法。

**[Arbitrary-Scale 3D Gaussian Super-Resolution](3d_vision/arbitrary-scale_3d_gaussian_super-resolution.md)**

:   提出Arbi-3DGSR集成框架，通过尺度感知渲染、生成先验引导优化和渐进超分三个核心组件，首次实现单个3DGS模型支持任意（包括非整数）倍率的高分辨率渲染，在×5.7倍率下PSNR比3DGS提升6.59dB，且保持85 FPS实时速度。

**[Assist-3D Adapted Scene Synthesis For Class-Agnostic 3D Instance Segmentation](3d_vision/assist-3d_adapted_scene_synthesis_for_class-agnostic_3d_instance_segmentation.md)**

:   提出 ASSIST-3D 合成数据流水线，通过异构物体选择、LLM 引导的场景布局生成和仿真实点云构建三个阶段，为 class-agnostic 3D 实例分割生成高质量标注数据，显著提升模型泛化能力。

**[Can Protective Watermarking Safeguard The Copyright Of 3D Gaussian Splatting](3d_vision/can_protective_watermarking_safeguard_the_copyright_of_3d_gaussian_splatting.md)**

:   首次系统性地揭示了 3DGS 水印框架的脆弱性，提出 GSPure 框架通过视角感知权重累积和几何特征聚类精准分离并去除水印相关的 Gaussian 原语，在水印 PSNR 最高降低 16.34dB 的同时保持原始场景损失不足 1dB。

**[Casl Curvature-Augmented Self-Supervised Learning For 3D Anomaly Detection](3d_vision/casl_curvature-augmented_self-supervised_learning_for_3d_anomaly_detection.md)**

:   发现点云曲率本身就是强大的异常检测线索，提出曲率增强的自监督学习框架 CASL，通过多尺度曲率提示引导坐标重建来学习通用 3D 表征，无需任何异常检测专用机制即可在 Real3D-AD 上以 5.6% O-AUROC 优势刷新 SOTA。

**[Class-Partitioned Vq-Vae And Latent Flow Matching For Point Cloud Scene Generati](3d_vision/class-partitioned_vq-vae_and_latent_flow_matching_for_point_cloud_scene_generati.md)**

:   提出类别分区的 VQ-VAE（CPVQ-VAE）和潜空间流匹配模型（LFMM），实现了首个无需外部数据库检索的纯点云场景生成方法，在复杂客厅场景上将 Chamfer 距离降低了 70.4%。

**[Ctrlfuse Mask-Prompt Guided Controllable Infrared And Visible Image Fusion](3d_vision/ctrlfuse_mask-prompt_guided_controllable_infrared_and_visible_image_fusion.md)**

:   提出 CtrlFuse，通过 mask prompt 引导 SAM 微调，实现红外-可见光图像的交互式可控融合，在融合质量和下游分割/检测任务上同时取得提升。

**[Dance Density-Agnostic And Class-Aware Network For Point Cloud Completion](3d_vision/dance_density-agnostic_and_class-aware_network_for_point_cloud_completion.md)**

:   提出 DANCE 框架，通过基于射线的候选点采样和 opacity 预测机制实现密度无关的点云补全，并引入分类头提供语义先验，在 PCN 和 MVP 基准上取得 SOTA。

**[Dapointmamba Domain Adaptive Point Mamba For Point Cloud Completion](3d_vision/dapointmamba_domain_adaptive_point_mamba_for_point_cloud_completion.md)**

:   首次将 Mamba（SSM）引入无监督域自适应点云补全（UDA PCC），提出 DAPointMamba 框架，通过跨域 Patch 级扫描、空间 SSM 对齐和通道 SSM 对齐三个模块，在保持线性复杂度和全局感受野的同时实现了跨域高质量点云补全。

**[Debiasing Diffusion Priors Via 3D Attention For Consistent Gaussian Splatting](3d_vision/debiasing_diffusion_priors_via_3d_attention_for_consistent_gaussian_splatting.md)**

:   提出 TD-Attn 框架，通过 3D 感知注意力引导（3D-AAG）和层级注意力调制（HAM）两个模块，解决 T2I 扩散模型中先验视角偏差导致的 3D 生成/编辑多视图不一致问题（Janus problem），可作为通用插件集成到现有 3DGS 框架。

**[Deepraht Learning Predictive Raht For Point Cloud Attribute Compression](3d_vision/deepraht_learning_predictive_raht_for_point_cloud_attribute_compression.md)**

:   提出首个端到端可微的 RAHT（Region Adaptive Hierarchical Transform）框架 DeepRAHT，用于有损点云属性压缩，通过可学习的预测模型和基于 Laplace 分布的码率代理实现了超越 G-PCC 标准和现有深度学习方法的压缩性能。

**[Distilling Future Temporal Knowledge With Masked Feature Reconstruction For 3D O](3d_vision/distilling_future_temporal_knowledge_with_masked_feature_reconstruction_for_3d_o.md)**

:   提出 FTKD（Future Temporal Knowledge Distillation）框架，通过未来感知特征重建（FFR）和未来引导 logit 蒸馏（FLD）两个策略，将离线教师模型中的未来帧知识有效迁移到在线学生模型，在 nuScenes 上取得 1.3 mAP/1.3 NDS 提升且不增加推理开销。

**[Domain Generalized Stereo Matching With Uncertainty-Guided Data Augmentation](3d_vision/domain_generalized_stereo_matching_with_uncertainty-guided_data_augmentation.md)**

:   提出 UgDA-Stereo，通过对 RGB 图像逐通道均值和标准差施加基于批次统计量的高斯不确定性扰动来模拟多种未知域的视觉风格，并结合特征一致性约束，以即插即用方式显著提升立体匹配模型的跨域泛化能力。

**[Dynamic Gaussian Scene Reconstruction From Unsynchronized Videos](3d_vision/dynamic_gaussian_scene_reconstruction_from_unsynchronized_videos.md)**

:   提出了一个粗到精（coarse-to-fine）的时间对齐模块，可插入到现有 4D Gaussian Splatting 框架中，解决多视角视频时间不同步导致的动态场景重建质量退化问题，在 DyNeRF 数据集上显著提升了多个基线方法的 PSNR/SSIM/LPIPS。

**[Enhancing Generalization Of Depth Estimation Foundation Model Via Weakly-Supervi](3d_vision/enhancing_generalization_of_depth_estimation_foundation_model_via_weakly-supervi.md)**

:   提出 WeSTAR 框架，通过语义感知的分层归一化自训练 + 稀疏成对序数弱监督 + LoRA 权重正则化三者协同，以参数高效的方式提升深度估计基础模型（Depth Anything V2）在未见域和损坏数据上的泛化能力，在多个 OOD 基准上达到 SOTA。

**[Enhancing Rotation-Invariant 3D Learning With Global Pose Awareness And Attentio](3d_vision/enhancing_rotation-invariant_3d_learning_with_global_pose_awareness_and_attentio.md)**

:   提出 Shadow-informed Pose Feature (SiPF) 和 RIAttnConv 算子，通过引入基于 Bingham 分布学习的全局"影子"参考点来增强局部旋转不变特征的全局姿态感知能力，解决对称结构（如飞机左右机翼）无法区分的"Wing-tip Feature Collapse"问题，在 ModelNet40 分类和 ShapeNetPart 分割上达到 SOTA。

**[Epsegfz Efficient Point Cloud Semantic Segmentation For Few- And Zero-Shot Scena](3d_vision/epsegfz_efficient_point_cloud_semantic_segmentation_for_few-_and_zero-shot_scena.md)**

:   提出 EPSegFZ，一个无需预训练的3D点云少样本/零样本语义分割框架，通过 ProERA 提取高频特征、LGPE 融合文本信息更新原型、DRPE 建立精确的查询-原型对应关系，在 S3DIS 和 ScanNet 上分别超越 SOTA 5.68% 和 3.82%。

**[Exploring Surround-View Fisheye Camera 3D Object Detection](3d_vision/exploring_surround-view_fisheye_camera_3d_object_detection.md)**

:   本文系统研究了环视鱼眼相机的3D目标检测问题：构建了同时包含针孔和鱼眼相机数据的Fisheye3DOD基准数据集，并提出FisheyeBEVDet和FisheyePETR两个框架，通过球面特征表征将鱼眼几何建模嵌入主流检测范式，相比矫正baseline提升最高6.2个FDS点。

**[Fantasystyle Controllable Stylized Distillation For 3D Gaussian Splatting](3d_vision/fantasystyle_controllable_stylized_distillation_for_3d_gaussian_splatting.md)**

:   本文提出FantasyStyle，首个完全基于扩散模型蒸馏的3DGS风格迁移框架，通过多视图频率一致性（MVFC）机制抑制低频分量减少视角间冲突，并设计可控风格化蒸馏（CSD）引入负引导消除风格图像的内容泄漏，在风格化质量和内容保持上均超越现有VGG和扩散方法。

**[Foundationslam Unleashing The Power Of Depth Foundation Models For](3d_vision/foundationslam_unleashing_the_power_of_depth_foundation_models_for.md)**

:   将深度基础模型的几何先验注入光流式SLAM系统，通过混合光流网络、双向一致BA层和可靠性感知精炼三个模块形成闭环，在TUM/EuRoC/7Scenes/ETH3D四大数据集取得SOTA轨迹精度和稠密重建质量，18 FPS实时运行。

**[Free-Form Scene Editor Enabling Multi-Round Object Manipulation Like In A 3D Eng](3d_vision/free-form_scene_editor_enabling_multi-round_object_manipulation_like_in_a_3d_eng.md)**

:   提出FFSE——一个基于视频扩散模型的自回归3D感知图像编辑框架，配合混合数据集3DObjectEditor（真实+合成），能像3D引擎一样在真实图像上执行多轮物体平移/缩放/旋转操作，同时生成逼真的阴影/反射/遮挡等背景效果并保持跨轮编辑一致性，在单轮和多轮编辑中均大幅超越现有方法。

**[Gaussian Blending Rethinking Alpha Blending In 3D Gaussian Splatting](3d_vision/gaussian_blending_rethinking_alpha_blending_in_3d_gaussian_splatting.md)**

:   重新审视3DGS中的标量alpha blending，指出其忽略像素内空间变化是多尺度渲染伪影（放大erosion/缩小dilation）的根源，提出Gaussian Blending——将alpha和transmittance建模为像素内的空间分布（2D uniform window），实现实时抗锯齿且无需重训练，在多尺度Blender上PSNR从31.59→35.80。

**[Gaussianimage Boosted Image Representation And Compression With 2D Gaussian Spla](3d_vision/gaussianimage_boosted_image_representation_and_compression_with_2d_gaussian_spla.md)**

:   提出 GaussianImage++，通过失真驱动的密度化机制和内容感知高斯滤波器，用有限数量的 2D Gaussian 原语实现高质量图像表示和压缩，并结合属性分离的可学习标量量化器实现高效压缩。

**[Generalized Geometry Encoding Volume For Real-Time Stereo Matching](3d_vision/generalized_geometry_encoding_volume_for_real-time_stereo_matching.md)**

:   提出 GGEV，将单目深度基础模型（Depth Anything V2）的深度先验以轻量方式融入代价聚合过程，通过深度感知动态代价聚合（DDCA）自适应增强不同视差假设的匹配关系，在实时速度下实现强泛化能力。

**[Geometry Meets Light Leveraging Geometric Priors For Universal Photometric Stere](3d_vision/geometry_meets_light_leveraging_geometric_priors_for_universal_photometric_stere.md)**

:   提出 GeoUniPS，将大规模3D重建模型（VGGT）中学到的几何先验注入光度立体管线，通过光-几何双分支编码器在多光照线索不可靠时（阴影、自遮挡、偏差光照）仍能恢复合理的表面法线。

**[Graph Smoothing For Enhanced Local Geometry Learning In Point Cloud Analysis](3d_vision/graph_smoothing_for_enhanced_local_geometry_learning_in_point_cloud_analysis.md)**

:   分析了传统图构建方法（ball query）在边界点处产生稀疏连接、在交汇区产生噪声连接的问题，提出图平滑模块（对称邻接优化 + von Neumann核）和局部几何学习模块（自适应形状特征 + 柱坐标变换），在分类和分割任务上取得竞争性能。

**[Griffin Aerial-Ground Cooperative Detection And Tracking Dataset And Benchmark](3d_vision/griffin_aerial-ground_cooperative_detection_and_tracking_dataset_and_benchmark.md)**

:   提出 Griffin，一个空地协同（AGC）3D感知数据集和基准框架，包含250+动态场景（37K+帧），通过CARLA-AirSim联合仿真实现真实无人机动力学、变化巡航高度（20-60m）和遮挡感知标注，并提供系统化的鲁棒性评估协议。

**[Gt2-Gs Geometry-Aware Texture Transfer For Gaussian Splatting](3d_vision/gt2-gs_geometry-aware_texture_transfer_for_gaussian_splatting.md)**

:   提出GT2-GS框架，通过几何感知纹理迁移损失、自适应细粒度控制模块和几何保持分支，实现高质量、视图一致的3DGS纹理迁移，在纹理保真度和场景内容保持上均优于现有3D风格迁移方法。

**[Hierarchical Direction Perception Via Atomic Dot-Product Operators For Rotation-](3d_vision/hierarchical_direction_perception_via_atomic_dot-product_operators_for_rotation-.md)**

:   提出 DiPVNet，基于 atomic dot-product operator 的双重属性（方向选择性 + 旋转不变性），构建局部 L2DP 算子和全局 DASFT 模块，实现层次化方向感知的旋转不变点云学习。

**[Ie-Srgs An Internal-External Knowledge Fusion Framework For High-Fidelity 3D Gau](3d_vision/ie-srgs_an_internal-external_knowledge_fusion_framework_for_high-fidelity_3d_gau.md)**

:   提出IE-SRGS框架，通过融合外部2D超分辨率模型提供的高频纹理先验（外部知识）与多尺度3DGS模型提供的跨视图一致深度/纹理特征（内部知识），配合掩码引导融合策略，从低分辨率输入实现高保真3DGS超分辨率重建，在合成和真实场景上均达到SOTA。

**[Learning Conjugate Direction Fields For Planar Quadrilateral Mesh Generation](3d_vision/learning_conjugate_direction_fields_for_planar_quadrilateral_mesh_generation.md)**

:   提出一种基于DGCNN的数据驱动方法高效生成共轭方向场（CDF），避免了传统非线性优化的高计算开销，支持用户笔画引导的可控CDF生成，将CDF生成速度提升了1-2个数量级，同时配套发布了包含50000+自由曲面的大规模数据集。

**[Meshsplat Generalizable Sparse-View Surface Reconstruction Via Gaussian Splattin](3d_vision/meshsplat_generalizable_sparse-view_surface_reconstruction_via_gaussian_splattin.md)**

:   提出MeshSplat，首个基于2DGS的可泛化稀疏视角表面重建框架，通过加权Chamfer Distance损失正则化深度预测和基于不确定性的法线预测网络对齐2DGS朝向，从新视角合成任务中以自监督方式学习几何先验，在稀疏视角网格重建和跨数据集泛化上均达到SOTA。

**[Mobgs Motion Deblurring Dynamic 3D Gaussian Splatting For Blurry Monocular Video](3d_vision/mobgs_motion_deblurring_dynamic_3d_gaussian_splatting_for_blurry_monocular_video.md)**

:   MoBGS 提出了一种端到端的动态去模糊 3D Gaussian Splatting 框架，通过 Blur-adaptive Latent Camera Estimation (BLCE) 和 Latent Camera-induced Exposure Estimation (LCEE) 两个核心模块，从模糊单目视频中重建清晰的时空新视角，在 Stereo Blur 数据集上大幅超越现有 SOTA 方法。

**[Monoclue Object-Aware Clustering Enhances Monocular 3D Object Detection](3d_vision/monoclue_object-aware_clustering_enhances_monocular_3d_object_detection.md)**

:   提出 MonoCLUE，通过**局部聚类**提取对象级视觉模式（如引擎盖、车顶等部件）和**广义场景记忆**聚合跨图像的一致外观特征，增强单目3D检测中被遮挡和截断物体的检测能力，在KITTI基准上实现SOTA性能，且不依赖额外深度或LiDAR信息。

**[Mr-Cosmo Visual-Text Memory Recall And Direct Cross-Modal Alignment Method For Q](3d_vision/mr-cosmo_visual-text_memory_recall_and_direct_cross-modal_alignment_method_for_q.md)**

:   提出MR-CoSMo，一种由粗到精的查询驱动3D分割模型，通过直接跨模态对齐模块（DCMA）建立3D点云与文本/2D图像的显式对齐，结合视觉-文本记忆模块（Memory Module）存储高置信度特征对来增强跨场景分割一致性，在3D指令分割、引用分割和语义分割三个任务上均达到SOTA。

**[Multi-Modal Assistance For Unsupervised Domain Adaptation On Point Cloud 3D Obje](3d_vision/multi-modal_assistance_for_unsupervised_domain_adaptation_on_point_cloud_3d_obje.md)**

:   提出 MMAssist，利用图像和文本特征作为"桥梁"对齐源域和目标域的 3D 特征，同时结合 2D 检测结果增强伪标签质量，显著提升了基于 LiDAR 的 3D 无监督域适应目标检测性能。

**[Oceansplat Object-Aware Gaussian Splatting With Trinocular View Consistency For ](3d_vision/oceansplat_object-aware_gaussian_splatting_with_trinocular_view_consistency_for_.md)**

:   提出 OceanSplat，通过三目视图一致性约束、合成对极深度先验和深度感知透明度调整，实现了散射介质下的高保真水下 3D 高斯泼溅场景重建，显著减少了浮动伪影并超越现有方法。

**[Open-World 3D Scene Graph Generation For Retrieval-Augmented Reasoning](3d_vision/open-world_3d_scene_graph_generation_for_retrieval-augmented_reasoning.md)**

:   提出统一框架 OSU-3DSG，结合视觉-语言模型进行开放世界 3D 场景图生成，并通过检索增强推理支持场景问答、视觉定位、实例检索和任务规划四种交互任务，在无监督条件下达到与有监督方法可比的性能。

**[Openscan A Benchmark For Generalized Open-Vocabulary 3D Scene Understanding](3d_vision/openscan_a_benchmark_for_generalized_open-vocabulary_3d_scene_understanding.md)**

:   本文提出了广义开放词汇 3D 场景理解任务（GOV-3D）及对应的 OpenScan 基准，将 3D 场景理解从物体类别扩展到八种语言学属性维度，揭示了现有 OV-3D 方法在理解抽象物体属性方面的严重不足。

**[Opt3Dgs Optimizing 3D Gaussian Splatting With Adaptive Exploration And Curvature](3d_vision/opt3dgs_optimizing_3d_gaussian_splatting_with_adaptive_exploration_and_curvature.md)**

:   提出 Opt3DGS 框架，将 3DGS 训练分为探索和利用两阶段：探索阶段用自适应加权 SGLD 逃离局部最优，利用阶段用局部拟牛顿 Adam 优化器实现精确收敛，在不修改高斯表示的前提下达到 SOTA 渲染质量。

**[Parameter-Free Fine-Tuning Via Redundancy Elimination For Vision Foundation Mode](3d_vision/parameter-free_fine-tuning_via_redundancy_elimination_for_vision_foundation_mode.md)**

:   发现视觉基础模型（SAM/SAM2/DINOv2）中存在大量冗余通道，提出无需更新任何参数的微调方法：通过基于输出差异的通道选择算法找到最优替换对，用有效通道替换冗余通道来增强下游任务的特征表示，平均提升 mIoU 5-11 个点。

**[Pb4U-Gnet Resolution-Adaptive Garment Simulation Via Propagation-Before-Update G](3d_vision/pb4u-gnet_resolution-adaptive_garment_simulation_via_propagation-before-update_g.md)**

:   提出 Pb4U-GNet，通过将消息传播与特征更新解耦（Propagation-before-Update），结合分辨率感知的传播深度控制和更新缩放机制，实现了仅在低分辨率网格上训练即可泛化到高分辨率网格的服装仿真。

**[Pfavatar Pose-Fusion 3D Personalized Avatar Reconstruction From Real-World Outfi](3d_vision/pfavatar_pose-fusion_3d_personalized_avatar_reconstruction_from_real-world_outfi.md)**

:   提出 PFAvatar，通过两阶段方法（姿态感知扩散模型微调 + NeRF蒸馏）从真实世界"每日穿搭"(OOTD)照片中重建高质量3D人物头像，在仅5分钟内完成个性化定制，较先前方法实现48倍加速。

**[Physics-Informed Deformable Gaussian Splatting Towards Unified Constitutive Laws](3d_vision/physics-informed_deformable_gaussian_splatting_towards_unified_constitutive_laws.md)**

:   将每个3D Gaussian视为拉格朗日物质点，引入时变材料场预测粒子速度和本构应力张量，通过Cauchy动量残差作为物理约束 + 拉格朗日粒子流匹配作为数据拟合项，在单目动态视图合成中实现了物理一致性和跨场景泛化能力，在自建物理驱动数据集和HyperNeRF真实数据集上均达到SOTA。

**[Point-Sra Self-Representation Alignment For 3D Representation Learning](3d_vision/point-sra_self-representation_alignment_for_3d_representation_learning.md)**

:   提出 Point-SRA，通过 Dual Self-Representation Alignment（MAE 层 + MFT 层）和 MeanFlow 概率建模，利用不同 mask ratio 下表征的互补性来增强 3D 点云表征学习，在 ScanObjectNN 上超越 Point-MAE 达 5.59%。

**[Point Cloud Quantization Through Multimodal Prompting For 3D Understanding](3d_vision/point_cloud_quantization_through_multimodal_prompting_for_3d_understanding.md)**

:   提出 PCQ（Point Cloud Quantization），利用预训练视觉-语言模型的文本嵌入作为语义原型，通过 Gumbel-Softmax 可微量化将连续点云特征离散化到文本原型空间，结合跨模态特征融合实现3D理解的显著提升。

**[Presstrack-Hmr Pressure-Based Top-Down Multi-Person Global Human Mesh Recovery](3d_vision/presstrack-hmr_pressure-based_top-down_multi-person_global_human_mesh_recovery.md)**

:   提出 PressTrack-HMR，首个仅基于压力信号实现多人全局人体网格恢复的自上而下流水线，通过创新的 UoE 相似度度量实现压力足迹跟踪（93.6% MOTA），并构建了首个多人交互压力数据集 MIP。

**[Probfm Probabilistic Time Series Foundation Model With Uncertainty Decomposition](3d_vision/probfm_probabilistic_time_series_foundation_model_with_uncertainty_decomposition.md)**

:   首次将 Deep Evidential Regression (DER) 与 Normal-Inverse-Gamma 先验引入时序基础模型架构，实现单次前向传播即可进行 epistemic-aleatoric 不确定性分解，并在加密货币预测中验证了不确定性感知交易策略的实用价值。

**[Radarllm Empowering Large Language Models To Understand Human Motion From Millim](3d_vision/radarllm_empowering_large_language_models_to_understand_human_motion_from_millim.md)**

:   提出 RadarLLM，首个利用大语言模型从毫米波雷达点云进行语义级人体运动理解的端到端框架，包含基于 Aggregate VQ-VAE 的运动引导雷达分词器和雷达感知语言模型，并通过物理感知仿真管线生成大规模雷达-文本配对数据。

**[Real-Time 3D Object Detection With Inference-Aligned Learning](3d_vision/real-time_3d_object_detection_with_inference-aligned_learning.md)**

:   提出 SR3D 框架，通过空间优先最优传输标签分配（SPOTA）和排序感知自适应自蒸馏（RAS）两个训练阶段组件，弥合室内密集 3D 目标检测中训练与推理行为的不一致性，在 ScanNet V2 和 SUN RGB-D 上以 42ms 实时速度刷新密集检测器 SOTA。

**[Redundant Queries In Detr-Based 3D Detection Methods Unnecessary And Prunable](3d_vision/redundant_queries_in_detr-based_3d_detection_methods_unnecessary_and_prunable.md)**

:   提出 GPQ（Gradually Pruning Queries），通过分类分数逐步裁剪 DETR 系 3D 检测器中大量冗余的 object queries，无需额外可学习参数，可直接在预训练 checkpoint 上微调完成，在边缘设备上最高实现 67.86% FLOPs 减少和 65.16% 推理时间下降。

**[Rethinking Multimodal Point Cloud Completion A Completion-By-Correction Perspect](3d_vision/rethinking_multimodal_point_cloud_completion_a_completion-by-correction_perspect.md)**

:   提出 Completion-by-Correction 新范式，用预训练 image-to-3D 模型生成拓扑完整的形状先验，再通过特征空间纠正使其与局部观测对齐，取代传统的 Completion-by-Inpainting 方法，在 ShapeNetViPC 上平均 CD 降低 23.5%、F-score 提升 7.1%。

**[Rethinking Rainy 3D Scene Reconstruction Via Perspective Transforming And Bright](3d_vision/rethinking_rainy_3d_scene_reconstruction_via_perspective_transforming_and_bright.md)**

:   提出 OmniRain3D 数据集（首次同时建模视角异质性和亮度动态性的雨天 3D 场景数据集）以及 REVR-GSNet 端到端框架（联合递归亮度增强 + 高斯基元优化 + GS引导去雨），实现从雨天退化图像到高保真干净 3D 场景的重建。

**[Retrieving Objects From 3D Scenes With Box-Guided Open-Vocabulary Instance Segme](3d_vision/retrieving_objects_from_3d_scenes_with_box-guided_open-vocabulary_instance_segme.md)**

:   提出 Box-Guided 方法，利用 2D 开放词汇检测器 YOLO-World 的检测框引导从超点构建 3D 实例 mask，无需 SAM 和 CLIP，在保持高效（<1分钟/场景）的同时显著提升对低频类别目标的检索能力。

**[Rtgaze Real-Time 3D-Aware Gaze Redirection From A Single Image](3d_vision/rtgaze_real-time_3d-aware_gaze_redirection_from_a_single_image.md)**

:   提出 RTGaze，一个实时 3D 感知视线重定向方法，通过混合频率特征编码器 + 视线注入模块 + 3D 面部几何先验蒸馏，从单张图像实现 61ms/帧的高质量视线重定向，比前 SOTA 3D 方法（GazeNeRF）快 800 倍。

**[S5 Scalable Semi-Supervised Semantic Segmentation In Remote Sensing](3d_vision/s5_scalable_semi-supervised_semantic_segmentation_in_remote_sensing.md)**

:   提出 S5 框架，首次将半监督语义分割扩展为遥感基础模型（RSFM）的预训练范式，通过构建百万级 RS4P-1M 数据集和 MoE 多数据集微调策略，在多个遥感分割与检测基准上达到 SOTA。

**[Simba Towards High-Fidelity And Geometrically-Consistent Point Cloud Completion ](3d_vision/simba_towards_high-fidelity_and_geometrically-consistent_point_cloud_completion_.md)**

:   提出 Simba 框架，首次将点云补全重构为"对几何变换场做扩散"而非"对点坐标做扩散"，通过 Sym-Diffuser 学习逐点仿射变换的条件分布来生成粗糙补全，再用级联 Mamba 架构（MBA-Refiner）逐步精修到高保真输出，在 PCN、ShapeNet、KITTI 多个基准上达到 SOTA。

**[Smartsplat Feature-Smart Gaussians For Scalable Compression Of Ultra-High-Resolu](3d_vision/smartsplat_feature-smart_gaussians_for_scalable_compression_of_ultra-high-resolu.md)**

:   提出SmartSplat，一种基于特征感知的2D Gaussian Splatting图像压缩框架，通过梯度-颜色引导的变分采样、排斥均匀采样和尺度自适应颜色初始化三大策略，首次实现了8K/16K超高分辨率图像在极端压缩比（最高5000×）下的高质量重建。

**[Sparse4Dgs 4D Gaussian Splatting For Sparse-Frame Dynamic Scene Reconstruction](3d_vision/sparse4dgs_4d_gaussian_splatting_for_sparse-frame_dynamic_scene_reconstruction.md)**

:   首次提出稀疏帧动态场景重建方法Sparse4DGS，通过纹理感知变形正则化（TADR）和纹理感知规范优化（TACO）两个核心模块，从稀疏视频帧中实现高保真4D场景重建。

**[Sparsesurf Sparse-View 3D Gaussian Splatting For Surface Reconstruction](3d_vision/sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)**

:   提出SparseSurf方法，通过立体几何-纹理对齐（SGTA）和伪特征增强几何一致性（PFEGC），在稀疏视角下同时实现高精度表面重建和高质量新视角合成。

**[Splat-Sap Feed-Forward Gaussian Splatting For Human-Centered Scene With Scale-Aw](3d_vision/splat-sap_feed-forward_gaussian_splatting_for_human-centered_scene_with_scale-aw.md)**

:   提出Splat-SAP，一种前馈式方法，从大间隔的双目相机输入中重建尺度感知的点图并通过高斯平面渲染人体中心场景的自由视角视频，无需逐场景优化且无需3D几何监督。

**[Splats In Splats Robust And Effective 3D Steganography Towards Gaussian Splattin](3d_vision/splats_in_splats_robust_and_effective_3d_steganography_towards_gaussian_splattin.md)**

:   提出"Splats in Splats"，首个在3DGS中嵌入3D隐藏内容而不修改任何vanilla 3DGS属性的隐写术框架，通过重要性分级的SH系数加密和自编码器辅助的不透明度映射实现安全、鲁棒且高效的版权保护。

**[Splatssc Decoupled Depth-Guided Gaussian Splatting For Semantic Scene Completion](3d_vision/splatssc_decoupled_depth-guided_gaussian_splatting_for_semantic_scene_completion.md)**

:   提出SplatSSC，一种基于深度引导初始化和解耦高斯聚合器（DGA）的单目3D语义场景补全框架，通过紧凑的高斯基元初始化和鲁棒的几何-语义解耦聚合，在Occ-ScanNet上以更少基元获得SOTA性能。

**[Split-Layer Enhancing Implicit Neural Representation By Maximizing The Dimension](3d_vision/split-layer_enhancing_implicit_neural_representation_by_maximizing_the_dimension.md)**

:   提出 Split-Layer，将 MLP 全连接层拆分为多个并行分支并用 Hadamard 积整合输出，在不增加参数和计算的前提下将特征空间维度从 $C$ 指数级扩展到 $\binom{C/\sqrt{N}+N-1}{N}$，显著提升隐式神经表示（INR）的表征能力。

**[Streamstgs Streaming Spatial And Temporal Gaussian Grids For Real-Time Free-View](3d_vision/streamstgs_streaming_spatial_and_temporal_gaussian_grids_for_real-time_free-view.md)**

:   提出 StreamSTGS，一种可流式传输的时空高斯网格表示，将规范 3D 高斯属性编码为 2D 图像、时序特征编码为视频，实现实时自由视角视频流（帧大小仅 170KB），同时通过 Transformer 辅助训练和滑动窗口机制保证重建质量（PSNR 32.30dB）。

**[Surface-Based Visibility-Guided Uncertainty For Continuous Active 3D Neural Reco](3d_vision/surface-based_visibility-guided_uncertainty_for_continuous_active_3d_neural_reco.md)**

:   提出基于表面的可见性场(SBV)，通过SDF推导的表面置信度和体素网格更新机制在连续主动学习过程中准确估计不确定性的可见性，指导Next-Best View选择，在DTU/Blender/TanksAndTemples/BlendedMVS四个基准上图像渲染质量提升最高11.6%。

**[Tg-Field Geometry-Aware Radiative Gaussian Fields For Tomographic Reconstruction](3d_vision/tg-field_geometry-aware_radiative_gaussian_fields_for_tomographic_reconstruction.md)**

:   提出 TG-Field，一种面向极端稀疏视角 CT 重建的几何感知高斯形变框架，通过多分辨率哈希编码器建模空间几何先验、时空注意力模块和运动流网络处理动态 CT，在静态和动态 CT 重建中均实现了 SOTA 性能。

**[Tosc Task-Oriented Shape Completion For Open-World Dexterous Grasp Generation Fr](3d_vision/tosc_task-oriented_shape_completion_for_open-world_dexterous_grasp_generation_fr.md)**

:   提出任务导向形状补全（TOSC）这一新任务，仅补全与操控任务相关的接触区域（而非整个物体），利用预训练基础模型生成候选形状、3D 判别自编码器筛选最优形状、FlowGrasp 流匹配模型生成灵巧抓取，在抓取位移和 Chamfer 距离上分别提升 16.17% 和 55.26%。

**[Uncovering Zero-Shot Generalization Gaps In Time-Series Foundation Models Using ](3d_vision/uncovering_zero-shot_generalization_gaps_in_time-series_foundation_models_using_.md)**

:   提出从真实视频中通过光流提取时间序列数据的管线，构建了 REAL-V-TSFM 数据集（6130 条序列），揭示了当前时间序列基础模型（Chronos、TimesFM 等）在面对真实物理动态时的零样本泛化能力不足。

**[Unic-Lift Unified 3D Instance Segmentation Via Contrastive Learning](3d_vision/unic-lift_unified_3d_instance_segmentation_via_contrastive_learning.md)**

:   提出 UniC-Lift，一个统一的单阶段 3D 实例分割框架，通过在 3DGS 基元中学习可优化的向量嵌入，并利用对比损失和三元组损失训练，最终通过简单的"嵌入到标签"（Embedding-to-Label）过程直接解码出一致的 3D 分割标签，无需 HDBSCAN 等后处理聚类步骤，训练时间从 15+ 小时缩短至 40 分钟以内。

**[Vggt-Dp Generalizable Robot Control Via Vision Foundation Models](3d_vision/vggt-dp_generalizable_robot_control_via_vision_foundation_models.md)**

:   提出 VGGT-DP，一个受生物视觉系统启发的视觉运动策略框架，将预训练的 3D 感知基础模型 VGGT 作为视觉编码器并与扩散策略（Diffusion Policy）结合，通过帧级 Token 复用机制、随机 Token 裁剪和本体感知引导视觉学习三个关键设计，在 MetaWorld 高精度操作任务上显著超越 DP 和 DP3 基线。

**[Vpn Visual Prompt Navigation](3d_vision/vpn_visual_prompt_navigation.md)**

:   提出视觉提示导航（VPN）新范式：用户在 2D 俯视图上标注视觉轨迹（箭头连接关键路点）来引导智能体导航，替代自然语言指令和图像目标指令，构建了 R2R-VP 和 R2R-CE-VP 两个数据集及 VPNet 基线模型，结合视图级和轨迹级数据增强后在离散和连续环境中均取得优异性能。

---

## 🎨 图像生成 { #image_generation }

**[Abductivemllm Boosting Visual Abductive Reasoning Within Mll](image_generation/abductivemllm_boosting_visual_abductive_reasoning_within_mll.md)**

:   受人类认知中"语言溯因+图像想象"双模式启发，提出 AbductiveMLLM，通过 Reasoner（因果对比学习筛选假设）和 Imaginer（扩散模型图像化推理）两个协同组件增强 MLLM 的视觉溯因推理能力，在 VAR 和 YouCookII 基准上取得 SOTA。

**[Aedr Training-Free Ai-Generated Image Attribution Via Autoen](image_generation/aedr_training-free_ai-generated_image_attribution_via_autoen.md)**

:   提出一种基于自编码器双重重建损失比值的免训练图像归因方法，通过图像均匀度校准消除纹理复杂度偏差，在8个主流扩散模型上平均准确率达95.1%，比最强基线高24.7%，且速度快约100倍。

**[Aggregating Diverse Cue Experts For Ai-Generated Image Detec](image_generation/aggregating_diverse_cue_experts_for_ai-generated_image_detec.md)**

:   提出Multi-Cue Aggregation Network (MCAN)，通过混合编码器适配器(MoEA)将原始图像、高频信息和新提出的色度不一致性(CI)三种互补线索统一融合，实现跨生成模型的鲁棒AI生成图像检测。

**[Annealed Relaxation Of Speculative Decoding For Faster Autor](image_generation/annealed_relaxation_of_speculative_decoding_for_faster_autor.md)**

:   提出Cool-SD，一种有理论支撑的退火松弛speculative decoding框架：通过推导TV距离上界得到最优重采样分布，并证明接受概率递减调度比均匀调度产生更小的分布偏移，在LlamaGen和Lumina-mGPT上实现了比LANTERN++更优的速度-质量权衡。

**[Anostyler Text-Driven Localized Anomaly Generation Via Light](image_generation/anostyler_text-driven_localized_anomaly_generation_via_light.md)**

:   将零样本异常生成建模为文本引导的局部风格迁移问题，通过轻量级U-Net + CLIP损失将正常图像的掩码区域风格化为语义对齐的异常图像，在MVTec-AD和VisA上以263M参数（仅0.61M可训练）超越扩散模型基线，同时显著提升下游异常检测性能。

**[Backdoors In Conditional Diffusion Threats To Responsible Synthetic Data Pipelin](image_generation/backdoors_in_conditional_diffusion_threats_to_responsible_synthetic_data_pipelin.md)**

:   揭示了 ControlNet 条件分支的后门攻击漏洞：仅需 1–5% 的投毒数据即可在不修改扩散主干的前提下植入后门，触发时无视文本 prompt 生成攻击者指定内容，并提出 clean fine-tuning (CFT) 作为实用防御。

**[Beautiful Images Toxic Words Understanding And Addressing Offensive Text In Gene](image_generation/beautiful_images_toxic_words_understanding_and_addressing_offensive_text_in_gene.md)**

:   揭示扩散模型在生成图像中嵌入 NSFW 文字的新威胁，提出基于文本生成层定向 LoRA 微调的 NSFW-Intervention 方法，并发布 ToxicBench 基准。

**[Breaking The Modality Barrier Generative Modeling For Accurate Molecule Retrieva](image_generation/breaking_the_modality_barrier_generative_modeling_for_accurate_molecule_retrieva.md)**

:   提出 GLMR 两阶段框架（对比学习预检索 + 生成式语言模型重排），通过生成与输入质谱对齐的分子结构将跨模态检索转化为单模态检索，在 MassSpecGym 上 Recall@1 提升超 40%。

**[Cad-Vae Leveraging Correlation-Aware Latents For Comprehensive Fair Disentanglem](image_generation/cad-vae_leveraging_correlation-aware_latents_for_comprehensive_fair_disentanglem.md)**

:   提出CAD-VAE，引入"相关隐变量" $z_R$ 显式建模目标属性和敏感属性之间的共享信息，通过最小化条件互信息 $I(z_Y;z_S|z_R)$ 实现公平解缠绕，无需领域知识即可产生公平表示和高质量反事实样本。

**[Causalclip Causally-Informed Feature Disentanglement And Filtering For Generaliz](image_generation/causalclip_causally-informed_feature_disentanglement_and_filtering_for_generaliz.md)**

:   提出 CausalCLIP，通过 Gumbel-Softmax 掩码 + HSIC 约束将 CLIP 特征解耦为因果/非因果子空间，结合对抗掩码和反事实干预保留稳定取证线索，跨生成器泛化准确率提升 6.83%。

**[Conditional Diffusion Model For Multi-Agent Dynamic Task Dec](image_generation/conditional_diffusion_model_for_multi-agent_dynamic_task_dec.md)**

:   提出 CD3T，一个两层层次化 MARL 框架：用条件扩散模型学习动作语义表示（以观测和他人动作为条件，预测下一观测和奖励），通过 k-means 聚类得到子任务划分，高层选择子任务、低层在受限动作空间执行策略，在 SMAC 的 Super Hard 场景上显著超越所有基线。

**[Constrained Particle Seeking Solving Diffusion Inverse Problems With Just Forwar](image_generation/constrained_particle_seeking_solving_diffusion_inverse_problems_with_just_forwar.md)**

:   提出 Constrained Particle Seeking (CPS)，一种无梯度的扩散模型反问题求解方法，通过利用所有候选粒子信息构建前向过程的局部线性代理模型，并在转移核高密度区域的超球面约束下寻找最优粒子，性能可与梯度方法媲美。

**[Continuous Degradation Modeling Via Latent Flow Matching For Real-World Super-Re](image_generation/continuous_degradation_modeling_via_latent_flow_matching_for_real-world_super-re.md)**

:   提出 DegFlow，通过残差自编码器 + 潜空间 Flow Matching 从离散尺度的真实 HR-LR 对学习连续退化轨迹，仅需单张 HR 图像即可合成任意连续尺度的逼真 LR 图像，用于训练超分模型达到 SOTA。

**[Copyright Infringement Detection In Text-To-Image Diffusion Models Via Different](image_generation/copyright_infringement_detection_in_text-to-image_diffusion_models_via_different.md)**

:   从差分隐私（Differential Privacy）角度形式化版权侵权的定义，提出 D-Plus-Minus（DPM）框架，通过对扩散模型分别进行"学习"和"遗忘"两个方向的微调，测量条件敏感度差异来事后检测文本到图像模型中的版权侵权行为。

**[Countsteer Steering Attention For Object Counting In Diffusion Models](image_generation/countsteer_steering_attention_for_object_counting_in_diffusion_models.md)**

:   提出 CountSteer，一种免训练的推理时方法，通过在扩散模型的 cross-attention 隐状态中注入自适应 steering vector，将物体计数准确率提升约 4%，且不损害图像质量。

**[Creating Blank Canvas Against Ai-Enabled Image Forgery](image_generation/creating_blank_canvas_against_ai-enabled_image_forgery.md)**

:   提出"空白画布"机制，通过对抗扰动使 SAM 对受保护图像"视而不见"，当图像被篡改后篡改区域会被 SAM 自动识别，实现无需篡改训练数据的主动式篡改定位。

**[Dice Distilling Classifier-Free Guidance Into Text Embedding](image_generation/dice_distilling_classifier-free_guidance_into_text_embedding.md)**

:   提出 DICE，训练一个仅 2M 参数的轻量 sharpener 将 CFG 的引导效果蒸馏进 text embedding，使无引导采样达到与 CFG 同等的生成质量、推理计算量减半，在 SD1.5 多个变体、SDXL 和 PixArt-α 上全面验证有效，是 AAAI 2026 口头报告论文。

**[Diff-V2M A Hierarchical Conditional Diffusion Model With Explicit Rhythmic Model](image_generation/diff-v2m_a_hierarchical_conditional_diffusion_model_with_explicit_rhythmic_model.md)**

:   提出 Diff-V2M，一个基于层次条件扩散 Transformer 的视频到音乐生成框架，通过显式节奏建模（低分辨率 ODF）和层次交叉注意力机制整合情感/语义/节奏特征，在域内和域外数据集上均达到 SOTA。

**[Diffa Large Language Diffusion Models Can Listen And Understand](image_generation/diffa_large_language_diffusion_models_can_listen_and_understand.md)**

:   提出 DIFFA——首个基于扩散语言模型的大型音频-语言模型，通过冻结 LLaDA-8B 骨干网络 + 轻量双适配器架构 + 两阶段训练管线，仅用 960 小时 ASR 数据和 127 小时合成指令数据就在 MMSU、MMAU、VoiceBench 上达到与自回归 baseline 竞争的性能。

**[Difficulty Controlled Diffusion Model For Synthesizing Effec](image_generation/difficulty_controlled_diffusion_model_for_synthesizing_effec.md)**

:   在Stable Diffusion中引入难度编码器（MLP，输入类别+难度分数），通过LoRA微调解耦"域对齐"和"难度控制"两个目标，使生成数据的学习难度可控——仅用10%额外合成数据即超过Real-Fake的最佳结果，节省63.4 GPU小时。

**[Diffusion Reconstruction-Based Data Likelihood Estimation For Core-Set Selection](image_generation/diffusion_reconstruction-based_data_likelihood_estimation_for_core-set_selection.md)**

:   提出利用扩散模型的部分反向去噪重建偏差作为数据似然的理论近似信号，配合信息瓶颈理论选择最优重建时间步，实现分布感知的核心集选择，在 ImageNet 上仅用 50% 数据即可逼近全量训练性能。

**[Dogfit Domain-Guided Fine-Tuning For Efficient Transfer Learning Of Diffusion Mo](image_generation/dogfit_domain-guided_fine-tuning_for_efficient_transfer_learning_of_diffusion_mo.md)**

:   提出 DogFit，将域引导（Domain Guidance）内化到扩散模型的微调损失中，使模型在训练时学会引导方向，推理时无需双重前向传播即可实现可控的保真度-多样性权衡，在 6 个目标域上以一半的采样 TFLOPS 超越 SOTA 引导方法。

**[Dos Directional Object Separation In Text Embeddings For Mul](image_generation/dos_directional_object_separation_in_text_embeddings_for_mul.md)**

:   识别出多物体生成失败的四种场景（相似形状/纹理、不同背景偏好、多物体），通过构建方向性分离向量修改CLIP的三类文本嵌入（语义token/EOT/pooled），在SDXL上将成功率提升16-25%并将融合率降低3-12%，推理速度接近baseline（约4×快于Attend-and-Excite）。

**[Echogen Cycle-Consistent Learning For Unified Layout-Image Generation And Unders](image_generation/echogen_cycle-consistent_learning_for_unified_layout-image_generation_and_unders.md)**

:   提出 EchoGen，统一布局到图像生成（L2I）和图像定位（I2L）两个任务的框架，通过渐进式训练——并行预训练→双任务联合优化→循环强化学习（CycleRL）——利用布局→图像→布局回环的一致性约束作为自监督奖励，在 MS-COCO 和 LayoutSAM 上达到 SOTA。

**[Efficientflow Efficient Equivariant Flow Policy Learning For Embodied Ai](image_generation/efficientflow_efficient_equivariant_flow_policy_learning_for_embodied_ai.md)**

:   提出 EfficientFlow，将等变性引入 Flow Matching 策略学习框架，理论证明各向同性先验+等变速度网络保证动作分布等变，并提出 Flow Acceleration Upper Bound (FABO) 正则化加速采样，在 MimicGen 12 个任务上实现比 EquiDiff 快 20-56 倍的推理速度且性能更优。

**[Enhancing Multimodal Misinformation Detection By Replaying The Whole Story From ](image_generation/enhancing_multimodal_misinformation_detection_by_replaying_the_whole_story_from_.md)**

:   提出 RetSimd，通过将文本分段并用文本转图像模型生成一系列增补图像来"重放完整故事"，配合图神经网络融合多图像关系，显著提升了图像模态对虚假信息检测的贡献，在三个基准数据集上一致性地改进了五种 SOTA 方法的性能。

**[Fgm-Hd Boosting Generation Diversity Of Fractal Generative Models Through Hausdo](image_generation/fgm-hd_boosting_generation_diversity_of_fractal_generative_models_through_hausdo.md)**

:   本文首次将 Hausdorff 维数（HD）引入分形生成模型（FGM），提出可学习的 HD 估计模块、单调动量驱动调度策略（MMDS）和 HD 引导的拒绝采样，在 ImageNet 上实现 39% 的生成多样性提升（Recall），同时保持图像质量。

**[Flowing Backwards Improving Normalizing Flows Via Reverse Representation Alignme](image_generation/flowing_backwards_improving_normalizing_flows_via_reverse_representation_alignme.md)**

:   提出 R-REPA（Reverse Representation Alignment），创造性地利用 Normalizing Flows 的可逆性，在生成（反向）路径上将中间特征与视觉基础模型对齐，同时提出免训练分类算法，在 ImageNet 64×64 和 256×256 上实现 NF 新 SOTA，训练加速 3.3 倍。

**[Freeinpaint Tuning-Free Prompt Alignment And Visual Rationality Enhancement In I](image_generation/freeinpaint_tuning-free_prompt_alignment_and_visual_rationality_enhancement_in_i.md)**

:   提出FreeInpaint，一种即插即用的免训练方法，通过优化初始噪声引导注意力聚焦到修复区域（PriNo），并在去噪过程中分解条件分布为文本对齐、视觉合理性和人类偏好三项引导（DeGu），同时提升图像修复的提示词对齐和视觉合理性。

**[Gewdiff Geometric Enhanced Wavelet-Based Diffusion Model For Hyperspectral Image](image_generation/gewdiff_geometric_enhanced_wavelet-based_diffusion_model_for_hyperspectral_image.md)**

:   提出GEWDiff，一种几何增强的基于小波的扩散模型，通过小波编码器-解码器高效压缩高光谱数据到潜在空间，引入边缘感知噪声调度和mask条件控制保持几何完整性，并设计多级损失函数促进稳定收敛，实现4倍高光谱图像超分辨率的SOTA效果。

**[Head-Aware Kv Cache Compression For Efficient Visual Autoreg](image_generation/head-aware_kv_cache_compression_for_efficient_visual_autoreg.md)**

:   发现VAR模型中attention head天然分为Contextual Heads（语义一致性，垂直注意力模式）和Structural Heads（空间连贯性，多对角线模式），提出HACK框架通过非对称预算分配和模式特定压缩策略，在70%压缩率下实现无损生成质量，Infinity-8B上1.75×显存减少和1.57×加速。

**[Hierarchicalprune Position-Aware Compression For Large-Scale Diffusion Models](image_generation/hierarchicalprune_position-aware_compression_for_large-scale_diffusion_models.md)**

:   提出 HierarchicalPrune，利用 MMDiT 扩散模型中块的层级功能差异（早期块建立语义结构、后期块处理纹理细节），通过层级位置剪枝（HPP）、位置权重保护（PWP）和敏感度引导蒸馏（SGDistill）三种技术协同，结合 INT4 量化，将 SD3.5 Large Turbo（8B）从 15.8GB 压缩至 3.24GB（79.5% 内存缩减），仅损失 4.8% 图像质量。

**[How Bias Binds Measuring Hidden Associations For Bias Control In Text-To-Image C](image_generation/how_bias_binds_measuring_hidden_associations_for_bias_control_in_text-to-image_c.md)**

:   首次研究文本到图像生成中的**组合语义绑定偏见**问题，提出Bias Adherence Score (BA-Score)量化物体-属性绑定如何激活偏见，并设计免训练的Context-Bias Control (CBC)框架，通过token嵌入解耦和残差注入实现组合生成中超过10%的去偏改善。

**[Hyperbolic Hierarchical Alignment Reasoning Network For Text-3D Retrieval](image_generation/hyperbolic_hierarchical_alignment_reasoning_network_for_text-3d_retrieval.md)**

:   提出H2ARN，在Lorentz双曲空间中嵌入文本和3D点云数据，通过层次排序损失（蕴含锥）解决层次表示坍塌问题，通过贡献感知双曲聚合解决冗余导致的显著性稀释问题，在Text-3D检索中取得SOTA，并发布了2.6倍规模的T3DR-HIT v2数据集。

**[Improved Masked Image Generation With Knowledge-Augmented Token Representations](image_generation/improved_masked_image_generation_with_knowledge-augmented_token_representations.md)**

:   提出KA-MIG框架，通过从训练数据中挖掘三种token级语义先验知识图（共现图、语义相似图、位置-token不兼容图），使用图感知编码器学习增强的token表示，并通过轻量级加减融合机制注入现有MIG模型，持续提升多种骨干网络的生成质量。

**[Infinite-Story A Training-Free Consistent Text-To-Image Gene](image_generation/infinite-story_a_training-free_consistent_text-to-image_gene.md)**

:   基于 scale-wise 自回归模型（Infinity），通过三个 training-free 技术——Identity Prompt Replacement（消除文本编码器的上下文偏差）、Adaptive Style Injection（参考图像特征注入）和 Synchronized Guidance Adaptation（同步 CFG 两个分支），实现了身份与风格一致的多图像生成，速度比扩散模型快 6 倍（1.72 秒/张）。

**[Laytrol Preserving Pretrained Knowledge In Layout Control Fo](image_generation/laytrol_preserving_pretrained_knowledge_in_layout_control_fo.md)**

:   通过从 MM-DiT 复制参数初始化布局控制网络、设计专用初始化方案（布局编码器初始化为纯文本编码器 + 输出零初始化）、并用 FLUX 自己生成的图像构建 LaySyn 数据集来缓解分布偏移，实现了在 FLUX 上高质量的布局到图像生成。

**[Longllada Unlocking Long Context Capabilities In Diffusion Llms](image_generation/longllada_unlocking_long_context_capabilities_in_diffusion_llms.md)**

:   首次系统研究扩散大语言模型（diffusion LLMs）的长上下文能力，发现其在直接外推时保持稳定困惑度和"局部感知"现象，并提出无需训练的 LongLLaDA 方法，通过 NTK-based RoPE 外推成功将上下文窗口扩展至 6 倍（24k tokens）。

**[Longt2Ibench A Benchmark For Evaluating Long Text-To-Image Generation With Graph](image_generation/longt2ibench_a_benchmark_for_evaluating_long_text-to-image_generation_with_graph.md)**

:   提出 LongT2IBench，首个面向长文本到图像（T2I）对齐的评估基准，包含 14K 长文本-图片对和图结构化人工标注，并构建 LongT2IExpert 评估器，通过层次化对齐思维链（HA-CoT）指令微调 MLLM，同时输出对齐分数和结构化解释。

**[Macprompt Maraconic-Guided Jailbreak Against Text-To-Image Models](image_generation/macprompt_maraconic-guided_jailbreak_against_text-to-image_models.md)**

:   提出 MacPrompt，一种黑盒跨语言攻击方法，通过将有害词汇翻译为多语言候选并进行字符级重组构造"通心粉词（macaronic words）"作为对抗 prompt，能够同时绕过文本安全过滤器和概念移除防御，在色情内容上攻击成功率高达 92%，在暴力内容上达 90%。

**[Macs Multi-Source Audio-To-Image Generation With Contextual Significance And Sem](image_generation/macs_multi-source_audio-to-image_generation_with_contextual_significance_and_sem.md)**

:   提出 MACS，首个显式分离多源音频再生成图像的两阶段框架，通过弱监督声源分离 + CLAP 空间语义对齐（排序损失 + 对比损失）+ 解耦交叉注意力扩散生成，在多源、混合源和单源音频到图像生成任务上全面超越 SOTA。

**[Mass Concept Erasure In Diffusion Models With Concept Hierarchy](image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)**

:   提出基于supertype-subtype概念层级的分组擦除策略和Supertype-Preserving LoRA (SuPLoRA)，通过冻结down-projection矩阵（正交于supertype子空间）仅训练up-projection矩阵，在大规模多领域概念擦除中实现擦除效果与生成质量的最优平衡。

**[Mdiff4Str Mask Diffusion Model For Scene Text Recognition](image_generation/mdiff4str_mask_diffusion_model_for_scene_text_recognition.md)**

:   首次将掩码扩散模型（MDM）引入场景文本识别（STR）任务，提出 MDiff4STR，通过六种训练掩码策略（弥合训练-推理噪声差距）和 Token 替换噪声机制（解决过度自信问题），在仅需 3 步去噪的情况下超越 SOTA 自回归模型的准确率，同时实现 3× 推理加速。

**[Melodia Training-Free Music Editing Guided By Attention Probing In Diffusion Mod](image_generation/melodia_training-free_music_editing_guided_by_attention_probing_in_diffusion_mod.md)**

:   通过对扩散模型中注意力图的深入探测分析，发现自注意力图对于保持音乐时间结构至关重要，据此提出 Melodia——一种免训练的音乐编辑方法，通过选择性操控自注意力图实现属性修改与结构保持的最优平衡。

**[Mixture Of Ranks With Degradation-Aware Routing For One-Step Real-World Image Su](image_generation/mixture_of_ranks_with_degradation-aware_routing_for_one-step_real-world_image_su.md)**

:   将稀疏混合专家（MoE）思想引入真实世界图像超分辨率任务，提出 Mixture-of-Ranks（MoR）架构，将 LoRA 的每个 rank 视为独立专家，并设计退化估计模块和退化感知负载均衡损失，实现单步高保真超分辨率重建。

**[Mp1 Meanflow Tames Policy Learning In 1-Step For Robotic Manipulation](image_generation/mp1_meanflow_tames_policy_learning_in_1-step_for_robotic_manipulation.md)**

:   首次将 MeanFlow 范式引入机器人学习领域，结合 3D 点云输入和 Dispersive Loss，实现仅需一次网络前向传播（1-NFE）即可生成动作轨迹，在机器人操作任务中以 6.8ms 推理延迟达到 SOTA 成功率。

**[Multi-Aspect Cross-Modal Quantization For Generative Recommendation](image_generation/multi-aspect_cross-modal_quantization_for_generative_recommendation.md)**

:   提出 MACRec，在生成式推荐的语义 ID 学习和生成模型训练两个阶段引入多方面跨模态交互，通过跨模态量化（对比学习增强残差量化）和多方面对齐（隐式+显式），显著提升推荐性能并降低 ID 冲突率。

**[Multi-Metric Preference Alignment For Generative Speech Restoration](image_generation/multi-metric_preference_alignment_for_generative_speech_restoration.md)**

:   提出多指标偏好对齐策略（Multi-Metric Preference Alignment），通过构建要求多个互补指标一致同意的偏好数据集 GenSR-Pref（80K 对），利用 DPO 对三种生成式语音修复范式（AR、MGM、FM）进行后训练对齐，显著提升修复质量并有效缓解 reward hacking。

**[Orvit Near-Optimal Online Distributionally Robust Reinforcement Learning](image_generation/orvit_near-optimal_online_distributionally_robust_reinforcement_learning.md)**

:   本文研究在线分布鲁棒强化学习，提出了基于 $f$-散度不确定性集的 RVI-$f$ 算法，在 $\chi^2$ 和 KL 散度下均实现了近似极小极大最优的遗憾界，且不依赖任何结构性假设。

**[Padiff Predictive And Adaptive Diffusion Policies For Ad Hoc Teamwork](image_generation/padiff_predictive_and_adaptive_diffusion_policies_for_ad_hoc_teamwork.md)**

:   首次将扩散模型应用于 Ad Hoc Teamwork 问题，提出 PADiff 框架，通过 Adaptive Feature Modulation Net（AFM-Net）实现对动态队友的实时适应，通过 Predictive Guidance Block（PGB）将队友意图预测信息注入去噪过程，在多模态合作场景中比现有方法平均提升 35.25%。

**[Playmate2 Training-Free Multi-Character Audio-Driven Animation Via Diffusion Tra](image_generation/playmate2_training-free_multi-character_audio-driven_animation_via_diffusion_tra.md)**

:   提出基于 Wan2.1 的 DiT 音频驱动人物视频生成框架：通过 LoRA 训练策略实现长视频生成，结合部分参数更新与 DPO 奖励反馈增强唇同步与动作自然度，并首创免训练的 Mask-CFG 方法实现多角色（≥3 人）音频驱动动画。

**[Procache Constraint-Aware Feature Caching With Selective Computation For Diffusi](image_generation/procache_constraint-aware_feature_caching_with_selective_computation_for_diffusi.md)**

:   提出 ProCache，一个免训练的动态特征缓存框架：通过约束感知的非均匀缓存模式搜索和选择性计算策略，在 DiT-XL/2 上实现 2.90 倍加速、PixArt-α 上实现 1.96 倍加速，且图像质量几乎无损，显著优于现有缓存方法。

**[Quantvsr Low-Bit Post-Training Quantization For Real-World Video Super-Resolutio](image_generation/quantvsr_low-bit_post-training_quantization_for_real-world_video_super-resolutio.md)**

:   提出 QuantVSR，首个面向扩散模型视频超分（VSR）的低比特（4/6-bit）后训练量化框架：通过时空复杂度感知（STCA）机制实现层自适应秩分配，并引入可学习偏置对齐（LBA）模块缓解低比特量化偏差，在 4-bit 设置下将参数量压缩 84.39%、计算量压缩 82.56%，同时保持与全精度模型接近的性能。

**[Realign Text-To-Motion Generation Via Step-Aware Reward-Guided Alignment](image_generation/realign_text-to-motion_generation_via_step-aware_reward-guided_alignment.md)**

:   提出 ReAlign（Reward-guided sampling Alignment），通过步感知（step-aware）奖励模型和奖励引导采样策略，在扩散推理过程中动态引导采样轨迹朝向文本-动作高对齐的分布，无需微调任何扩散模型即可显著提升多种动作生成方法的质量。以 MLD 为例，R@1 提升 17.9%，FID 改善 58.8%。

**[Realism Control One-Step Diffusion For Real-World Image Super-Resolution](image_generation/realism_control_one-step_diffusion_for_real-world_image_super-resolution.md)**

:   提出 RCOD 框架，通过潜在域分组策略和退化感知采样，赋予单步扩散（OSD）超分辨率方法在推理阶段灵活控制保真度-真实感平衡的能力，同时引入视觉提示注入模块替代文本提示来提升恢复精度。

**[Realistic Face Reconstruction From Facial Embeddings Via Diffusion Models](image_generation/realistic_face_reconstruction_from_facial_embeddings_via_diffusion_models.md)**

:   提出 FEM（Face Embedding Mapping）框架，利用 KAN 网络将任意人脸识别/隐私保护人脸识别系统的嵌入向量映射到预训练身份保持（ID-Preserving）扩散模型的嵌入空间，实现高分辨率真实人脸重建，可用于评估人脸识别系统的隐私泄露风险。

**[Rectified Noise A Generative Model Using Positive-Incentive Noise](image_generation/rectified_noise_a_generative_model_using_positive-incentive_noise.md)**

:   提出 Rectified Noise（ΔRN），通过正向激励噪声（π-noise）框架学习一组有益噪声并注入预训练 Rectified Flow 模型的速度场中，以仅 0.39% 的额外参数在 ImageNet-1k 上将 FID 从 10.16 降低到 9.05。

**[Relactrl Relevance-Guided Efficient Control For Diffusion Transformers](image_generation/relactrl_relevance-guided_efficient_control_for_diffusion_transformers.md)**

:   提出 RelaCtrl 框架，通过 ControlNet 相关性评分分析 DiT 各层对控制信息的敏感度差异，据此指导控制块的放置位置和建模强度，并设计二维混洗混合器（TDSM）替代自注意力和 FFN，以仅 15% 的参数量和计算复杂度实现优于 PixArt-δ 的可控生成效果。

**[Retrysql Text-To-Sql Training With Retry Data For Self-Correcting Query Generati](image_generation/retrysql_text-to-sql_training_with_retry_data_for_self-correcting_query_generati.md)**

:   提出 RetrySQL 训练范式，通过在推理步骤中注入 retry data（错误步骤 + [BACK] 标记 + 正确步骤）来持续预训练小型编码模型，使 1.5B 参数的开源模型学会自纠正能力，在 BIRD 和 SPIDER 基准上分别提升整体执行准确率最高 4 和 3.93 个百分点，挑战性样例提升高达 9 个百分点。

**[Right Looks Wrong Reasons Compositional Fidelity In Text-To-Image Generation](image_generation/right_looks_wrong_reasons_compositional_fidelity_in_text-to-image_generation.md)**

:   本文系统性地调研了文本到图像(T2I)模型在组合性忠实度方面的根本缺陷，聚焦否定(negation)、计数(counting)和空间关系(spatial relations)三大基本原语，揭示了模型在单一原语上表现尚可但联合组合时性能急剧下降的"亚乘性"(submultiplicative)干扰现象，并将其归因于训练数据稀缺、连续注意力架构不适合离散逻辑、以及评估指标偏向视觉合理性而非约束满足。

**[Self-Npo Data-Free Diffusion Model Enhancement Via Truncated Diffusion Fine-Tuni](image_generation/self-npo_data-free_diffusion_model_enhancement_via_truncated_diffusion_fine-tuni.md)**

:   提出 Self-NPO，一种无需外部数据标注或奖励模型的负偏好优化方法，通过截断扩散微调(TDFT)让扩散模型从自身生成的低质量数据中学习"什么是不好的"，配合 CFG 引导远离不良输出，仅需不到 Diffusion-NPO 1%的训练成本即可达到可比性能。

**[Simdiff Simpler Yet Better Diffusion Model For Time Series Point Forecasting](image_generation/simdiff_simpler_yet_better_diffusion_model_for_time_series_point_forecasting.md)**

:   提出SimDiff——首个纯端到端扩散模型实现时间序列点预测SOTA，通过统一的Transformer网络同时充当去噪器和预测器，结合Normalization Independence处理分布偏移和Median-of-Means集成策略将概率采样转化为精确点预测，在9个数据集上6个第一、3个第二。

**[Specdiff Accelerating Diffusion Model Inference With Self-Speculation](image_generation/specdiff_accelerating_diffusion_model_inference_with_self-speculation.md)**

:   提出 SpecDiff，一种基于自推测(self-speculation)的免训练多级特征缓存策略，通过利用少步推测引入**未来信息**辅助token重要性选择，突破了仅依赖历史信息的精度-速度瓶颈，在 Stable Diffusion 3/3.5 和 FLUX 上实现 2.80×/2.74×/3.17× 加速且质量损失可忽略。

**[Stabilizing Self-Consuming Diffusion Models With Latent Space Filtering](image_generation/stabilizing_self-consuming_diffusion_models_with_latent_space_filtering.md)**

:   提出Latent Space Filtering (LSF)方法，通过分析自消费扩散模型隐空间中潜在表示的低维结构退化现象，利用probing classifier的置信度分数过滤低质量合成数据，在固定训练预算下有效缓解模型坍塌，无需额外真实数据或增大训练集。

**[Steering One-Step Diffusion Model With Fidelity-Rich Decoder For Fast Image Comp](image_generation/steering_one-step_diffusion_model_with_fidelity-rich_decoder_for_fast_image_comp.md)**

:   提出 SODEC，一种基于单步扩散的图像压缩模型，通过保真度引导模块(FGM)将高保真VAE解码器的先验注入扩散生成过程，结合速率退火训练策略实现极低码率下的高质量压缩，解码速度比多步扩散方法快20×以上，同时在率-失真-感知权衡上达到SOTA。

**[Structure-Based Rna Design By Step-Wise Optimization Of Latent Diffusion Model](image_generation/structure-based_rna_design_by_step-wise_optimization_of_latent_diffusion_model.md)**

:   提出SOLD框架，将潜在扩散模型（LDM）与强化学习（RL）结合，通过步进式单步采样优化策略，直接优化RNA逆折叠中不可微的结构指标（二级结构相似度SS、最小自由能MFE、LDDT），在多个指标上全面超越现有方法。

**[Studying Classifier-Free Guidance From A Classifier-Centric Perspective](image_generation/studying_classifier-free_guidance_from_a_classifier-centric_perspective.md)**

:   通过系统实证研究揭示了classifier guidance和classifier-free guidance的本质机制——两者都通过将去噪轨迹推离分类器的决策边界来实现条件生成，并提出基于流匹配的后处理方法在高维数据上验证了这一"分类器中心"视角。

**[T-Lora Single Image Diffusion Model Customization Without Overfitting](image_generation/t-lora_single_image_diffusion_model_customization_without_overfitting.md)**

:   提出 T-LoRA，一种时步依赖的低秩适配框架，通过动态调整不同扩散时步的LoRA秩（高噪声时步用小秩、低噪声时步用大秩）和正交初始化（Ortho-LoRA）确保适配组件信息独立，解决了单图像扩散模型定制中的过拟合问题，在概念保真度和文本对齐间取得最优平衡。

**[T2I-Riskyprompt A Benchmark For Safety Evaluation Attack And Defense On Text-To-](image_generation/t2i-riskyprompt_a_benchmark_for_safety_evaluation_attack_and_defense_on_text-to-.md)**

:   构建T2I-RiskyPrompt——一个包含6,432条有效风险prompt的综合基准，涵盖6大类14细分风险类别，每条prompt带有层次化标注和详细风险原因，并提出reason-driven的MLLM风险检测方法（3B模型达91.8%准确率），系统评估了8个T2I模型、9种防御方法、5种安全过滤器和5种攻击策略。

**[Talk Snap Complain Validation-Aware Multimodal Expert Framework For Fine-Grained](image_generation/talk_snap_complain_validation-aware_multimodal_expert_framework_for_fine-grained.md)**

:   提出VALOR框架，结合Chain-of-Thought推理的多专家路由架构与语义对齐验证机制，在多轮多模态客服对话中实现细粒度投诉方面(Aspect)和严重度(Severity)的联合分类，较最强baseline Gemma-3绝对提升12.94%/6.51%。

**[Truthfulrag Resolving Factual-Level Conflicts In Retrieval-Augmented Generation ](image_generation/truthfulrag_resolving_factual-level_conflicts_in_retrieval-augmented_generation_.md)**

:   提出 TruthfulRAG 框架，首次利用知识图谱 (KG) 从事实级别解决 RAG 系统中检索知识与 LLM 参数知识之间的冲突，通过三元组提取、查询感知图检索和基于熵的冲突过滤机制提升生成准确性与可信度。

**[Tsgdiff Rethinking Synthetic Time Series Generation From A Pure Graph Perspectiv](image_generation/tsgdiff_rethinking_synthetic_time_series_generation_from_a_pure_graph_perspectiv.md)**

:   提出 TSGDiff，首次从纯图的视角重新审视时间序列生成任务，将时间序列表示为基于傅里叶频谱特征构建的动态图，在图的潜在空间中进行扩散建模，并提出 Topo-FID 指标评估生成时间序列的结构忠实度。

**[Voicecloak A Multi-Dimensional Defense Framework Against Unauthorized Diffusion-](image_generation/voicecloak_a_multi-dimensional_defense_framework_against_unauthorized_diffusion-.md)**

:   针对 diffusion-based voice cloning (VC) 的主动防御框架，通过多维度对抗扰动同时实现说话人身份混淆和感知质量退化，显著优于现有防御方法。

**[X2Edit Revisiting Arbitrary-Instruction Image Editing Through Self-Constructed D](image_generation/x2edit_revisiting_arbitrary-instruction_image_editing_through_self-constructed_d.md)**

:   构建 370 万高质量编辑数据集（14 类任务），并提出基于 Task-Aware MoE-LoRA + Contrastive Learning 的轻量级（0.9B 参数）plug-and-play 编辑模块，性能媲美 12B 全参数训练模型。

---

## 🧩 多模态VLM { #multimodal_vlm }

**[Aligning The True Semantics Constrained Decoupling And Distr](multimodal_vlm/aligning_the_true_semantics_constrained_decoupling_and_distr.md)**

:   提出 CDDS 算法，通过双路径 UNet 将嵌入解耦为语义和模态分量，并利用分布采样方法间接实现跨模态语义对齐，避免直接调整嵌入导致的分布扭曲，在 Flickr30K 和 MS-COCO 上超越 SOTA 6.6%~14.2%。

**[Anyecg-Chat A Generalist Ecg-Mllm For Flexible Ecg Input And](multimodal_vlm/anyecg-chat_a_generalist_ecg-mllm_for_flexible_ecg_input_and.md)**

:   构建anyECG数据集（含报告生成、波形定位、多ECG比较三大任务）并提出anyECG-chat模型，通过动态ECG输入机制支持变长/少导联/多ECG输入，采用三阶段课程学习训练，在报告生成的OOD泛化、秒级异常波形定位和多ECG对比分析上全面超越现有ECG-MLLM。

**[Are We Done Yet A Vision-Based Judge For Autonomous Task Completion Of Computer ](multimodal_vlm/are_we_done_yet_a_vision-based_judge_for_autonomous_task_completion_of_computer_.md)**

:   提出基于 VLM 的自主任务完成评估框架，通过截图+任务描述判断 CUA 是否完成任务，并将评估反馈回传给 Agent 实现自我纠正，在 macOS 环境上达到 73% 评估准确率和 27% 的任务成功率相对提升。

**[Astar Boosting Multimodal Reasoning With Automated Structure](multimodal_vlm/astar_boosting_multimodal_reasoning_with_automated_structure.md)**

:   提出AStar，一种training-free的多模态推理范式，通过从500个种子样本中构建高层"thought cards"推理模板库，在推理时自适应检索最优模板引导MLLM结构化推理，7B模型在MathVerse上达53.9%准确率（超越GPT-4o的50.2%），仅需50分钟预处理时间且无需训练。

**[Biprompt Bilateral Prompt Optimization For Visual And Textual Debiasing In Visio](multimodal_vlm/biprompt_bilateral_prompt_optimization_for_visual_and_textual_debiasing_in_visio.md)**

:   提出 BiPrompt，一种双边 prompt 优化框架，在测试时同时缓解 CLIP 等 VLM 中视觉侧（结构化注意力擦除）和文本侧（平衡 prompt 归一化）的虚假偏差，无需重训练即可提升 OOD 鲁棒性。

**[Bofa Bridge-Layer Orthogonal Low-Rank Fusion For Clip-Based ](multimodal_vlm/bofa_bridge-layer_orthogonal_low-rank_fusion_for_clip-based_.md)**

:   提出BOFA框架，仅微调CLIP已有的跨模态投影层（bridge-layer），通过正交低秩融合（Orthogonal Low-Rank Fusion）将参数更新约束在与旧任务特征正交的低秩"安全子空间"中，配合跨模态混合原型分类器，在不增加任何额外参数和推理开销的前提下实现了SOTA的无样本存储类增量学习。

**[Branch Or Layer Zeroth-Order Optimization For Continual Lear](multimodal_vlm/branch_or_layer_zeroth-order_optimization_for_continual_lear.md)**

:   本文系统探索了零阶（ZO）优化在基于PEFT的视觉-语言持续学习（VLCL）中的应用，发现全ZO替换会导致训练不稳定，提出从分支级（branch-wise）到层级（layer-wise）的渐进式ZO-FO混合策略，并基于视觉模态方差更大的理论发现提出MoZO策略（梯度符号归一化+视觉扰动约束），在四个benchmark上达到SOTA。

**[Bridging Modalities Via Progressive Re-Alignment For Multimo](multimodal_vlm/bridging_modalities_via_progressive_re-alignment_for_multimo.md)**

:   提出 BriMPR 框架，通过"分而治之"策略将多模态测试时自适应(MMTTA)分解为多个单模态特征对齐子问题，先用 prompt tuning 校准各模态全局特征分布实现初始跨模态语义对齐，再通过跨模态掩码嵌入重组和实例级对比学习精细化对齐。

**[Bridging The Copyright Gap Do Large Vision-Language Models R](multimodal_vlm/bridging_the_copyright_gap_do_large_vision-language_models_r.md)**

:   首次系统评估 LVLM 在多模态上下文中对版权内容的识别和遵守能力，构建了 50,000 对多模态查询-内容的大规模 benchmark，发现 11/12 个 SOTA LVLM 即使面对明确版权声明也无法有效拒绝侵权请求，并提出 CopyGuard 工具增强框架将侵权拒绝率从 ~3% 提升至 ~62%。

**[Concept-Rulenet Grounded Multi-Agent Neurosymbolic Reasoning](multimodal_vlm/concept-rulenet_grounded_multi-agent_neurosymbolic_reasoning.md)**

:   提出Concept-RuleNet——一个三智能体协作的神经符号推理框架，通过从训练图像中提取视觉概念来条件化符号生成和规则构建，解决了现有方法（如Symbol-LLM）仅依赖标签导致的符号幻觉和不代表性问题，在5个OOD基准上平均提升~5%准确率，幻觉符号减少达50%。

**[Conditional Information Bottleneck For Multimodal Fusion Overcoming Shortcut Lea](multimodal_vlm/conditional_information_bottleneck_for_multimodal_fusion_overcoming_shortcut_lea.md)**

:   揭示多模态讽刺检测中三类捷径学习问题（角色标签偏见、罐头笑声标签泄漏、情感不一致捷径）并重构了无捷径的 MUStARD++R 数据集，提出基于条件信息瓶颈的多模态融合框架 MCIB，通过压缩主模态冗余同时保留辅助模态的互补信息来实现有效融合。

**[Crebench Human-Aligned Creativity Evaluation From Idea To Process To Product](multimodal_vlm/crebench_human-aligned_creativity_evaluation_from_idea_to_process_to_product.md)**

:   提出 CreBench，一个覆盖创意想法→创作过程→创意产品三个维度、12个细粒度指标的多模态创造力评估基准，配套构建 CreMIT（2.2K样本、79.2K人工评价、4.7M指令）并微调出 CreExpert，在创造力评估上显著优于 GPT-4V 和 Gemini-Pro-Vision。

**[Cross-Modal Proxy Evolving For Ood Detection With Vision-Lan](multimodal_vlm/cross-modal_proxy_evolving_for_ood_detection_with_vision-lan.md)**

:   提出 CoEvo，一个 training-free 和 annotation-free 的 test-time 框架，通过双向 sample-conditioned 的文本/视觉 proxy 协同演化机制动态更新正负代理缓存，在 ImageNet-1K 上比最强负标签基线 AUROC 提升 1.33%、FPR95 降低 45.98%（从 18.92% 降至 10.22%），实现 SOTA 的 zero-shot OOD 检测。

**[Cross-Modal Unlearning Via Influential Neuron Path Editing I](multimodal_vlm/cross-modal_unlearning_via_influential_neuron_path_editing_i.md)**

:   提出 MIP-Editor，通过跨层梯度积分（文本）和 Fisher 积分（视觉）定位多模态大语言模型中编码待遗忘知识的**影响力神经元路径**，再用基于路径的表示误导（RMisU）编辑这些神经元，在 MLLMU-Bench 上实现最高 87.75% 的遗忘率和 54.26% 的通用知识保留提升。

**[Crossvid A Comprehensive Benchmark For Evaluating Cross-Vide](multimodal_vlm/crossvid_a_comprehensive_benchmark_for_evaluating_cross-vide.md)**

:   提出首个系统评估多模态大语言模型（MLLM）跨视频推理（Cross-Video Reasoning, CVR）能力的综合基准CrossVid，涵盖4个维度10个任务、5,331个视频和9,015个QA对，实验揭示当前最佳模型Gemini-2.5-Pro仅达50.4%准确率，远低于人类89.2%。

**[Difference Vector Equalization For Robust Fine-Tuning Of Vis](multimodal_vlm/difference_vector_equalization_for_robust_fine-tuning_of_vis.md)**

:   提出DiVE方法，通过约束预训练和微调模型嵌入之间的"差异向量"在各样本间保持相等，从而在CLIP微调过程中保持嵌入空间的几何结构，同时在ID、OOD、零样本三个指标上取得全面优于现有方法的结果（零样本平均提升8+点）。

**[Discode Distribution-Aware Score Decoder For Robust Automatic Evaluation Of Imag](multimodal_vlm/discode_distribution-aware_score_decoder_for_robust_automatic_evaluation_of_imag.md)**

:   提出 DISCODE，一种免微调的测试时自适应解码器，通过引入高斯先验分布最小化 ATT 损失，使 LVLM 生成的图像描述评估分数更鲁棒地对齐人类判断，并构建了覆盖六个视觉域的 MCEval 基准。

**[Em-Kd Distilling Efficient Multimodal Large Language Model W](multimodal_vlm/em-kd_distilling_efficient_multimodal_large_language_model_w.md)**

:   提出EM-KD框架，通过Hungarian算法解决teacher-student间视觉token数量不平衡问题，结合视觉语义蒸馏(VSD)和视觉-语言亲和力蒸馏(VLAD)将vanilla teacher的知识迁移到高效student MLLM，在11个benchmark上以144 token/patch达到50.4均分，超越576 token的LLaVA-NeXT(49.4)同时推理速度提升近2倍。

**[Exo2Ego Exocentric Knowledge Guided Mllm For Egocentric Vide](multimodal_vlm/exo2ego_exocentric_knowledge_guided_mllm_for_egocentric_vide.md)**

:   提出 Exo2Ego 框架，通过学习外中心(第三人称)与自中心(第一人称)域之间的映射关系，将 MLLM 中丰富的外中心知识迁移到自中心视频理解，结合新构建的 110万同步 ego-exo clip-text 对数据集 Ego-ExoClip 和 60万指令微调数据集 EgoIT，在 8 个自中心视频基准上取得了领先的开源模型性能。

**[Exploring Llms For Scientific Information Extraction Using The Sciex Framework](multimodal_vlm/exploring_llms_for_scientific_information_extraction_using_the_sciex_framework.md)**

:   本文提出SciEx，一个模块化、可组合的科学信息抽取框架，将PDF解析、多模态检索、Schema引导的抽取和跨文档聚合解耦为独立组件，在医学和环境科学的143篇论文数据集上评估了GPT-4o和Gemini-2.5-Flash的抽取能力，揭示了当前LLM在跨模态推理、数值精度和领域泛化方面的系统性不足。

**[Filter Correlate Compress Training-Free Token Reduction For ](multimodal_vlm/filter_correlate_compress_training-free_token_reduction_for_.md)**

:   提出FiCoCo三阶段框架（Filter-Correlate-Compress），通过集成视觉感知+语义感知冗余度量筛选丢弃token，利用token间相关性自适应回收信息，实现training-free的MLLM加速。在LLaVA-NeXT上达14.7×FLOPs压缩同时保留93.6%性能，在5种MLLM架构上全面超越FastV、SparseVLM等SOTA。

**[Finmmdocr Benchmarking Financial Multimodal Reasoning With Scenario Awareness Do](multimodal_vlm/finmmdocr_benchmarking_financial_multimodal_reasoning_with_scenario_awareness_do.md)**

:   本文提出FinMMDocR，一个面向真实金融场景的双语多模态推理基准，包含1200道专家标注的数值推理题目，涵盖12类隐式金融情景、9类长文档（平均50.8页）和平均11步推理链，最强MLLM (o4-mini-high) 仅达58%准确率，揭示现有模型在复杂金融推理中的严重不足。

**[Format Matters The Robustness Of Multimodal Llms In Reviewing Evidence From Tabl](multimodal_vlm/format_matters_the_robustness_of_multimodal_llms_in_reviewing_evidence_from_tabl.md)**

:   本文系统研究了多模态LLM在使用表格和图表作为证据验证科学声明时的鲁棒性，通过扩展SciTabAlign和ChartMimic两个数据集构建了表格-图表对齐的评估基准，发现12个多模态LLM在表格证据上的表现一致优于图表证据，而人类在两种格式上表现一致，揭示了当前模型在图表理解方面的关键短板。

**[Ft-Ncfm An Influence-Aware Data Distillation Framework For Efficient Vla Models](multimodal_vlm/ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)**

:   提出 FT-NCFM 框架，通过因果归因（Fact-Tracing）评估样本价值并引导对抗式 NCFM 过程合成高信息密度核心集，仅用 5% 合成数据即可达到全量训练 85-90% 的性能，训练时间减少 80% 以上。

**[Global Compression Commander Plug-And-Play Inference Acceler](multimodal_vlm/global_compression_commander_plug-and-play_inference_acceler.md)**

:   提出GlobalCom²，一个**即插即用、无需训练**的token压缩框架，专为动态裁剪（dynamic cropping）结构的高分辨率VLM设计：利用全局缩略图（thumbnail）作为"指挥官"引导局部裁剪区域（crop）的差异化压缩，在压缩90%视觉token的同时保持>90%原始性能。

**[Graph-Of-Mark Promote Spatial Reasoning In Multimodal Langua](multimodal_vlm/graph-of-mark_promote_spatial_reasoning_in_multimodal_langua.md)**

:   提出 Graph-of-Mark (GoM)，一种无需训练的像素级视觉提示方法，通过在输入图像上直接叠加深度感知的场景图（包含节点和有向边），显式编码物体间的空间关系，使多模态语言模型在 VQA 和定位任务中的零样本空间推理准确率最高提升 11 个百分点。

**[Harnessing Textual Semantic Priors For Knowledge Transfer And Refinement In Clip](multimodal_vlm/harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)**

:   本文提出SECA框架，利用CLIP文本分支的稳定语义先验来指导骨干网络中语义相关的历史知识迁移（SG-AKT模块），并通过文本嵌入的类间语义关系精炼视觉原型构建混合分类器（SE-VPR模块），在ImageNetR/A和CIFAR100上超越现有SOTA。

**[Harnessing Vision-Language Models For Time Series Anomaly Detection](multimodal_vlm/harnessing_vision-language_models_for_time_series_anomaly_detection.md)**

:   提出两阶段零样本时序异常检测框架：ViT4TS 用轻量 ViT 对时序折线图做多尺度 cross-patch 匹配定位候选异常区间，VLM4TS 用 GPT-4o 结合全局时序上下文验证和精炼检测结果，在 11 个 benchmark 上 F1-max 超最优 baseline 24.6%，token 用量仅为现有 LLM 方法的 1/36。

**[Headhunt-Vad Hunting Robust Anomaly-Sensitive Heads In Mllm ](multimodal_vlm/headhunt-vad_hunting_robust_anomaly-sensitive_heads_in_mllm_.md)**

:   本文提出 HeadHunt-VAD，通过在冻结的多模态大模型(MLLM)内部系统性地搜索出对异常敏感且稳定的稀疏注意力头集合，绕过文本输出的信息损失，用轻量级分类器实现无需微调的高效视频异常检测，在 UCF-Crime 和 XD-Violence 上取得 tuning-free 方法 SOTA。

**[Heterogeneous Uncertainty-Guided Composed Image Retrieval With Fine-Grained Prob](multimodal_vlm/heterogeneous_uncertainty-guided_composed_image_retrieval_with_fine-grained_prob.md)**

:   本文提出了HUG范式，通过细粒度高斯概率嵌入和异构不确定性估计（区分查询侧多模态协调不确定性与目标侧内容质量不确定性），结合动态加权融合和不确定性引导的对比学习，在Fashion-IQ和CIRR两个CIR基准上取得SOTA。

**[Imagebinddc Compressing Multi-Modal Data With Imagebind-Based Condensation](multimodal_vlm/imagebinddc_compressing_multi-modal_data_with_imagebind-based_condensation.md)**

:   本文提出ImageBindDC，首个在ImageBind统一特征空间中进行多模态数据压缩的框架，利用特征函数距离（CFD）替代传统MMD，并设计单模态/跨模态/联合模态三级分布对齐损失，在NYU-v2上仅用5个合成样本/类即实现与全数据训练相当的性能（97.30%），比前SOTA绝对提升8.2%，且压缩时间削减4.6倍。

**[Inex Hallucination Mitigation Via Introspection And Cross-Mo](multimodal_vlm/inex_hallucination_mitigation_via_introspection_and_cross-mo.md)**

:   提出 InEx 框架，通过内部自省推理（TVER 驱动的不确定性感知视觉增强）和外部跨模态多智能体协作（文本自反思 + 图像编辑验证 + 视觉自反思）迭代验证和修正 MLLM 输出，在 POPE 上提升 8.9%，在多个幻觉和通用 benchmark 上持续超越 OPERA/VCD/ICD。

**[Information Theoretic Optimal Surveillance For Epidemic Prevalence In Networks](multimodal_vlm/information_theoretic_optimal_surveillance_for_epidemic_prevalence_in_networks.md)**

:   本文首次提出以互信息作为优化准则的流行病监测框架 TestPrev，旨在选择网络中的最优节点子集以最大化与疾病流行度分布的互信息，从而提供传统方法无法给出的暴发规模分布级别洞察，并证明了其 NP-hard 性质，设计了贪心算法 GreedyMI 在合成与真实网络上优于基线方法。

**[Learning To Tell Apart Weakly Supervised Video Anomaly Detection Via Disentangle](multimodal_vlm/learning_to_tell_apart_weakly_supervised_video_anomaly_detection_via_disentangle.md)**

:   本文提出DSANet，通过自引导正常模式建模（SG-NM，粗粒度）和解耦对比语义对齐（DCSA，细粒度）从两个层面增强弱监督视频异常检测中正常与异常特征的可区分性，在XD-Violence上AP达86.95%（+1.14%），在UCF-Crime细粒度mAP达13.01%（+3.39%），均为SOTA。

**[Llm-Cas Dynamic Neuron Perturbation For Real-Time Hallucinat](multimodal_vlm/llm-cas_dynamic_neuron_perturbation_for_real-time_hallucinat.md)**

:   LLM-CAS 首次将 LLM 实时幻觉纠正建模为层次强化学习（HRL）问题，训练 RL Agent 在推理时动态选择最优的神经元扰动策略（高层选择功能网络类别，低层选择扰动类型和幅度），结合自适应掩码+因果追踪精确定位目标神经元，在 StoryCloze 上提升 10.98%，超越 ITI/CAA/SADI 等静态/动态基线。

**[Macvqa Adaptive Memory Allocation And Global Noise Filtering For Continual Visua](multimodal_vlm/macvqa_adaptive_memory_allocation_and_global_noise_filtering_for_continual_visua.md)**

:   本文提出MacVQA框架，通过全局噪声过滤（GonF）增强视觉特征的鲁棒性，并通过自适应记忆分配（AMA）基于原型检索和记忆衰减优化知识保留与更新，在VQA v2的10个持续学习任务上实现43.38%平均准确率（+3.57%）和2.32%遗忘率。

**[Mcmoe Completing Missing Modalities With Mixture Of Experts For Incomplete Multi](multimodal_vlm/mcmoe_completing_missing_modalities_with_mixture_of_experts_for_incomplete_multi.md)**

:   本文首次探索不完整多模态动作质量评估问题，提出 MCMoE 框架，利用自适应门控模态生成器（AGMG）补全缺失模态，并通过混合专家（MoE）动态融合单模态和跨模态联合表示，在单阶段训练中统一学习，在三个公开 AQA 基准上的完整和不完整场景中均达到 SOTA，且参数量仅 4.90M。

**[Multi-Agent Vlms Guided Self-Training With Pnu Loss For Low-Resource Offensive C](multimodal_vlm/multi-agent_vlms_guided_self-training_with_pnu_loss_for_low-resource_offensive_c.md)**

:   本文提出了一种多智能体视觉语言模型（MA-VLMs）引导的自训练框架，结合新颖的PNU损失函数，在仅有少量标注数据（如50个）的低资源场景下实现高质量攻击性内容检测，性能接近大规模模型。

**[Multi-Faceted Attack Exposing Cross-Model Vulnerabilities In Defense-Equipped Vi](multimodal_vlm/multi-faceted_attack_exposing_cross-model_vulnerabilities_in_defense-equipped_vi.md)**

:   提出多面攻击框架MFA，通过注意力转移攻击(ATA)突破对齐、对抗签名绕过内容审核、视觉编码器攻击覆写系统提示三个维度，系统性暴露配备多层防御的VLM（含GPT-4o/Gemini等商业模型）的安全漏洞，总体攻击成功率达58.5%。

**[O3Slm Open Weight Open Data And Open Vocabulary Sketch-Language Model](multimodal_vlm/o3slm_open_weight_open_data_and_open_vocabulary_sketch-language_model.md)**

:   本文构建了大规模草图-图像-指令三元组数据集SketchVCL（包含600K预训练 + 215K微调数据），并训练了O3SLM——首个能够流畅理解手绘草图并完成检测、计数、检索和VQA四大任务的开源大视觉语言模型，在所有任务上大幅超越现有LVLM。

**[Oida-Qa A Multimodal Benchmark For Analyzing The Opioid Industry Documents Archi](multimodal_vlm/oida-qa_a_multimodal_benchmark_for_analyzing_the_opioid_industry_documents_archi.md)**

:   本文基于UCSF-JHU阿片类药物行业文档档案（OIDA），构建了包含400K训练文档和370K多跳QA对的多模态文档问答基准OIDA-QA，并开发了结合内容重述和页面查找器的领域特化LLM系统，有效处理超长文档的多轮问答和答案页面定位。

**[Omnipt Unleashing The Potential Of Large Vision Language Models For Pedestrian T](multimodal_vlm/omnipt_unleashing_the_potential_of_large_vision_language_models_for_pedestrian_t.md)**

:   本文提出OmniPT，一个基于大视觉语言模型（LVLM）的统一行人跟踪框架，通过RL-Mid Training-SFT-RL四阶段训练策略，同时支持传统MOT、基于语言引用的跟踪（RMOT/CRMOT）和语义理解（SMOT），在多个基准上取得SOTA结果，尤其在BenSMOT上HOTA达75.04，较前SOTA提升3.06。

**[Panda Test-Time Adaptation With Negative Data Augmentation](multimodal_vlm/panda_test-time_adaptation_with_negative_data_augmentation.md)**

:   提出 Panda，通过负数据增强（patch 打乱重组）生成保留 corruption 但破坏语义的图像，用其特征偏移原始嵌入以抑制 corruption 引起的预测偏差，以极低开销（<10%）即插即用提升各类 TTA 方法的鲁棒性。

**[Patientvlm Meets Docvlm Pre-Consultation Dialogue Between Vision-Language Models](multimodal_vlm/patientvlm_meets_docvlm_pre-consultation_dialogue_between_vision-language_models.md)**

:   提出Pre-Consultation Dialogue Framework (PCDF)，通过两个VLM（DocVLM和PatientVLM）模拟医生-患者多轮对话，生成image-dialogue-diagnosis三元组用于微调DocVLM，在四个医学影像基准上平均F1提升11.48。

**[Pet2Rep Towards Vision-Language Model-Drived Automated Radiology Report Generati](multimodal_vlm/pet2rep_towards_vision-language_model-drived_automated_radiology_report_generati.md)**

:   本文提出 PET2Rep，首个专用于正电子发射断层扫描（PET）放射报告生成的大规模基准数据集（565例全身 PET/CT 图像-报告对），并设计了 PET 临床效能（CE）评估指标，对 30 个前沿通用和医疗专用 VLM 进行系统评估，发现当前 SOTA VLM 在 PET 报告生成任务上表现不佳，甚至无法超越简单的模板基线。

**[Phantom Menace Exploring And Enhancing The Robustness Of Vla Models Against Phys](multimodal_vlm/phantom_menace_exploring_and_enhancing_the_robustness_of_vla_models_against_phys.md)**

:   本文首次系统研究Vision-Language-Action（VLA）模型面对物理传感器攻击的安全性，提出"Real-Sim-Real"框架评估六种摄像头攻击和两种麦克风攻击对四个VLA模型的影响，发现所有VLA模型均存在严重脆弱性，并提出基于对抗训练的防御方法将中等强度攻击下的性能提升高达60%。

**[Pharos-Esg A Framework For Multimodal Parsing Contextual Narration And Hierarchi](multimodal_vlm/pharos-esg_a_framework_for_multimodal_parsing_contextual_narration_and_hierarchi.md)**

:   本文提出Pharos-ESG框架，通过基于版面流的阅读顺序建模、目录锚点引导的层次结构重建、上下文感知的多模态图像描述转换、以及多级金融标签预测四个核心模块，实现对ESG报告的结构化解析，在全面评估中F1达93.59、ROKT达0.92、TBTA达92.46%，显著超越MinerU、GPT-4o、Gemini 2.5 Pro等基线，并发布了首个大规模公开ESG报告数据集Aurora-ESG（24K+报告）。

**[Planttraitnet An Uncertainty-Aware Multimodal Framework For Global-Scale Plant T](multimodal_vlm/planttraitnet_an_uncertainty-aware_multimodal_framework_for_global-scale_plant_t.md)**

:   本文提出 PlantTraitNet，一个多模态、多任务、不确定性感知的深度学习框架，利用公民科学平台（iNaturalist、Pl@ntNet）的弱监督植物照片，结合图像特征（DINOv2）、深度先验（Depth-Anything-V2）和地理空间先验（Climplicit），同时预测四种关键植物性状（株高、叶面积、比叶面积、叶氮含量），生成的全球性状图在与 sPlotOpen 植被调查数据的基准测试中一致优于现有全球性状产品。

**[Recad Reinforcement Learning Enhanced Parametric Cad Model Generation With Visio](multimodal_vlm/recad_reinforcement_learning_enhanced_parametric_cad_model_generation_with_visio.md)**

:   提出 ReCAD 框架，通过将 CAD 脚本重写为参数化代码进行 SFT，再利用 GRPO 强化学习与分层基元课程学习策略，使 VLM 能从文本或图像输入生成高精度、可编辑的参数化 CAD 模型，在分布内和分布外设置上均大幅超越现有方法。

**[Rethinking Visual Token Reduction In Lvlms Under Cross-Modal Misalignment](multimodal_vlm/rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)**

:   揭示了 LVLM 中文本引导视觉token重要性评估的三种跨模态失配问题（因果、语义、空间），提出 VisionDrop——一个仅依赖视觉自注意力的免训练渐进式token剪枝框架，跨视觉编码器和 LLM 解码器多阶段压缩，在保留 5.6% token 时仍能维持 91%+ 原始性能。

**[Revisiting The Data Sampling In Multimodal Post-Training From A Difficulty-Disti](multimodal_vlm/revisiting_the_data_sampling_in_multimodal_post-training_from_a_difficulty-disti.md)**

:   提出两种多模态数据难度评估策略——PISM（渐进图像语义遮蔽）和CMAB（跨模态注意力平衡），发现在难度分层数据上仅用GRPO训练即可一致超越传统SFT+GRPO流水线，证明了战略性数据筛选比复杂训练范式更重要。

**[Rmadapter Reconstructionbased Multimodal Adapter For Visionlanguage](multimodal_vlm/rmadapter_reconstructionbased_multimodal_adapter_for_visionlanguage.md)**

:   提出 RMAdapter，一种双分支适配器架构：在标准 adapter 的适应分支旁增加重建分支（类 AutoEncoder），通过共享下投影层和逐层本地重建损失，在 CLIP 少样本微调中实现任务特定适应与通用知识保持的最佳平衡，在 Base-to-Novel 泛化、跨数据集和领域泛化三个任务上全面超越 SOTA（含 Prompt-based 方法）。

**[Safer-Clip Mitigating Nsfw Content In Vision-Language Models While Preserving Pr](multimodal_vlm/safer-clip_mitigating_nsfw_content_in_vision-language_models_while_preserving_pr.md)**

:   提出SafeR-CLIP框架，通过近邻感知重定向（将不安全嵌入重定向到语义最近的安全目标而非固定配对）和相对跨模态重定向损失（仅以不安全表示作为负样本而非随机批内负样本），在保持安全性的同时将零样本分类精度比Safe-CLIP恢复8.0%。

**[Sage Spuriousness-Aware Guided Prompt Exploration For Mitigating Multimodal Bias](multimodal_vlm/sage_spuriousness-aware_guided_prompt_exploration_for_mitigating_multimodal_bias.md)**

:   提出SAGE，一种无需训练、微调或外部标注的提示选择方法，通过计算提示模板在类别间的分离度得分来缓解CLIP模型中的多模态虚假偏差，在四个基准+五个骨干模型上一致提升最差组准确率（WGA）和调和均值（HM）。

**[Satiredecoder Visual Cascaded Decoupling For Enhancing Satirical Image Comprehen](multimodal_vlm/satiredecoder_visual_cascaded_decoupling_for_enhancing_satirical_image_comprehen.md)**

:   提出SatireDecoder，一种无需训练的框架，通过多智能体视觉级联解耦和不确定性引导的CoT推理来增强MLLM对讽刺图像的深层语义理解，在YesBut数据集上正确性、完整性和忠实性三项指标分别提升10%-40%。

**[Sdeval Safety Dynamic Evaluation For Multimodal Large Language Models](multimodal_vlm/sdeval_safety_dynamic_evaluation_for_multimodal_large_language_models.md)**

:   提出首个 MLLM 安全动态评估框架 SDEval，通过文本动态（6种策略）、图像动态（2类策略）和跨模态动态（4种策略）从原始安全基准生成可控复杂度的变体样本，在 MLLMGuard 和 VLSBench 上使 InternVL-3-78B 安全率下降近 10%，有效缓解数据泄露并暴露模型安全漏洞。

**[See Symbolize Act Grounding Vlms With Spatial Representations For Better Gamepla](multimodal_vlm/see_symbolize_act_grounding_vlms_with_spatial_representations_for_better_gamepla.md)**

:   系统性评估了符号化空间表示（物体坐标）对VLM游戏能力的影响，发现符号信息仅在检测准确时有益，当VLM自提取符号时效果取决于模型能力和场景复杂度，视觉帧始终不可或缺。

**[Seeing Justice Clearly Handwritten Legal Document Translation With Ocr And Visio](multimodal_vlm/seeing_justice_clearly_handwritten_legal_document_translation_with_ocr_and_visio.md)**

:   本文系统性对比了传统 OCR+机器翻译（OCR-MT）流水线与视觉大语言模型（vLLM）在手写马拉地语法律文档翻译为英语任务上的表现，发现两类方法均未达到法律级部署要求，OCR-MT 受级联错误影响严重，vLLM 存在严重的幻觉问题，但 vLLM 展现出统一端到端处理的发展潜力。

**[Speakerlm End-To-End Versatile Speaker Diarization And Recognition With Multimod](multimodal_vlm/speakerlm_end-to-end_versatile_speaker_diarization_and_recognition_with_multimod.md)**

:   SpeakerLM 是首个专为端到端说话人分离与识别（SDR）设计的多模态大语言模型，通过音频编码器-投影器-LLM 架构和灵活的说话人注册机制，在多个公开基准上大幅超越级联基线系统（cpCER 绝对降低最高达 13.82%），并在域外测试集上展现强鲁棒性。

**[Stola Self-Adaptive Touch-Language Framework With Tactile Commonsense Reasoning ](multimodal_vlm/stola_self-adaptive_touch-language_framework_with_tactile_commonsense_reasoning_.md)**

:   SToLa 提出首个基于混合专家（MoE）的触觉-语言框架，通过动态路由机制管理触觉和语言两种模态的差异，并构建了覆盖8种物理属性、4种交互特征的开放式触觉常识推理数据集 TactileBench，在 PhysiCLeAR 基准上以 7B 参数量超越 13B 的 Octopi 取得 SOTA。

**[Tabflash Efficient Table Understanding With Progressive Question Conditioning An](multimodal_vlm/tabflash_efficient_table_understanding_with_progressive_question_conditioning_an.md)**

:   TabFlash 提出渐进式问题条件化（Progressive Question Conditioning）和 Token 聚焦（Token Focusing）两大技术，在 ViT 中注入问题信息生成问题感知的视觉特征，并基于 L2 范数剪枝背景 token 同时通过对比训练将关键信息集中到保留 token 中，在7个表格理解基准上超越 GPT-4o 和 Gemini 2.5 Pro，同时减少 27% FLOPs 和 30% 显存。

**[The Triangle Of Similarity A Multi-Faceted Framework For Comparing Neural Networ](multimodal_vlm/the_triangle_of_similarity_a_multi-faceted_framework_for_comparing_neural_networ.md)**

:   本文提出"相似性三角"（Triangle of Similarity）框架，整合静态表征相似性（CKA/Procrustes）、功能相似性（线性模式连接/预测分布相似性）和稀疏性相似性（剪枝鲁棒性）三个互补视角来全面比较神经网络，发现架构家族是表征相似性的主要决定因素，且模型的表征结构比任务准确率在剪枝下更为鲁棒。

**[Tinychemvl Advancing Chemical Vision-Language Models Via Efficient Visual Token ](multimodal_vlm/tinychemvl_advancing_chemical_vision-language_models_via_efficient_visual_token_.md)**

:   TinyChemVL 是一个仅4B参数的化学领域VLM，通过自适应token合并与剪枝策略将视觉token压缩至原来的1/16，并引入反应级别任务和基准ChemRxn-V，在分子和反应级别的视觉化学任务上达到SOTA性能，同时显著提升推理和训练速度。

**[Tofa Training-Free One-Shot Federated Adaptation For Vision-Language Models](multimodal_vlm/tofa_training-free_one-shot_federated_adaptation_for_vision-language_models.md)**

:   提出TOFA框架，在联邦学习场景下通过层次贝叶斯模型学习个性化视觉prototype分布 + 全局对齐的LLM文本增强 + 自适应模态融合，实现无需训练、仅一轮通信的CLIP高效适配，在9个数据集上超越one-shot基线甚至部分多轮训练方法。

**[Towards A Rigorous Understanding Of The Population Dynamics Of The Nsga-Iii Tigh](multimodal_vlm/towards_a_rigorous_understanding_of_the_population_dynamics_of_the_nsga-iii_tigh.md)**

:   本文首次为 NSGA-III 在经典双目标 OneMinMax 基准上建立了紧致运行时界 $\Theta(n^2 \ln n / \mu)$，揭示了 NSGA-III 的种群动态特性，并证明其在适当种群规模下优于 NSGA-II。

**[Towards Authentic Movie Dubbing With Retrieve-Augmented Director-Actor Interacti](multimodal_vlm/towards_authentic_movie_dubbing_with_retrieve-augmented_director-actor_interacti.md)**

:   Authentic-Dubber 模拟真实配音工作流程中导演与演员的交互过程，通过构建多模态参考素材库、基于情感相似度的检索增强策略和渐进式图语音生成方法，显著提升了自动电影配音的情感表现力，在V2C-Animation数据集上的情感准确率和MOS评分均达到SOTA。

**[Towards Human-Ai Accessibility Mapping In India Vlm-Guided Annotations And Poi-C](multimodal_vlm/towards_human-ai_accessibility_mapping_in_india_vlm-guided_annotations_and_poi-c.md)**

:   本文将Project Sidewalk无障碍标注平台适配到印度昌迪加尔，通过定制化界面标签、VLM驱动的任务指导（Gemini 2.5 Flash），以及以POI为中心的分析框架，在三个不同土地用途的区域中审计了约40公里人行道，识别出1,644处可改善的无障碍设施位置。

**[Towards Long-Window Anchoring In Vision-Language Model Distillation](multimodal_vlm/towards_long-window_anchoring_in_vision-language_model_distillation.md)**

:   LAid（Long-window Anchoring distillation）提出了一种位置感知的知识蒸馏框架，通过头部级别的傅里叶增强位置知识传递，将小型VLM（3B/7B）的有效上下文窗口扩展至原来的3.2倍，接近大型教师模型（32B）的水平，同时保持标准VL基准上的性能。

**[Tri-Bench Stress-Testing Vlm Reliability On Spatial Reasoning Under Camera Tilt ](multimodal_vlm/tri-bench_stress-testing_vlm_reliability_on_spatial_reasoning_under_camera_tilt_.md)**

:   Tri-Bench 是一个包含400张实拍三角形图像的紧凑基准，通过控制相机姿态（平面/倾斜）和物体干扰两个因素，系统测试了四个领先VLM的空间几何推理能力，发现模型默认依赖2D图像平面线索而非3D真实几何（即使提供了明确的参考框架提示），在非多数类形状上准确率降至接近0%。

**[Urag Unified Retrieval And Generation In Multimodal Llms For](multimodal_vlm/urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)**

:   URaG 发现 MLLM 处理长文档时存在类人的"粗到细"推理模式（浅层注意力均匀分散、深层集中于证据页），基于此洞察在第 6 层插入轻量跨模态检索模块（仅占参数 0.05%），选取 Top-5 相关页面丢弃其余内容，实现 SOTA 性能的同时减少 44-56% 计算量。

**[Verb Mirage Unveiling And Assessing Verb Concept Hallucinations In Multimodal La](multimodal_vlm/verb_mirage_unveiling_and_assessing_verb_concept_hallucinations_in_multimodal_la.md)**

:   首次系统研究多模态大语言模型（MLLM）中的动词概念幻觉问题，构建了多维度基准测试，发现现有幻觉缓解方法对动词幻觉无效，并提出基于丰富动词知识微调的基线方法，显著缓解动词幻觉。

**[Vipact Visual-Perception Enhancement Via Specialized Vlm Age](multimodal_vlm/vipact_visual-perception_enhancement_via_specialized_vlm_age.md)**

:   VipAct 提出了一个多Agent协作框架，通过编排器Agent（任务分析+规划+协调）、专用Agent（描述/比较/视觉提示解读）和视觉专家模型（深度估计/目标检测/分割等）三层协作，显著提升 VLM 在细粒度视觉感知任务上的表现，在 Blink 上从 63.74% (zero-shot GPT-4o) 提升到 73.79%。

**[Vp-Bench A Comprehensive Benchmark For Visual Prompting In M](multimodal_vlm/vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)**

:   VP-Bench 提出了首个系统评估 MLLM 视觉提示（Visual Prompt）理解能力的两阶段 Benchmark：Stage 1 用 30K+ 图像覆盖 8 种 VP 形状×355 种属性组合评测 VP 感知能力，Stage 2 评测 VP 对 6 个下游任务的实际效果。在 28 个 MLLM 上的评测揭示了 VP 形状选择对性能的关键影响。

---

## 🎮 强化学习 { #reinforcement_learning }

**[A Course Correction In Steerability Evaluation Revealing Mis](reinforcement_learning/a_course_correction_in_steerability_evaluation_revealing_mis.md)**

:   本文提出了一个基于多维目标空间的 LLM 可操控性（steerability）评估框架，将 steering error 分解为校准偏差（miscalibration）和副作用（side effects/orthogonality），在文本改写任务上发现即使是最强的 LLM 也会产生严重副作用，prompt engineering 无效、best-of-N 采样代价高、RL 微调有改善但仍未彻底解决。

**[A Learning Framework For Cooperative Collision Avoidance Of ](reinforcement_learning/a_learning_framework_for_cooperative_collision_avoidance_of_.md)**

:   提出 reMARL 框架，将图像处理中的主动轮廓模型（active contour）作为领域知识引入多智能体强化学习的奖励设计，使无人机集群仅通过最大化个体奖励即可学会协作避撞，在大规模集群（≤10架）中性能显著优于 COMA/VDN/QMIX/MAPPO 等 SOTA MARL 方法，反应时间比元启发式方法快 98.75%，能耗降低 85.37%。

**[A Multi-Agent Conversational Bandit Approach To Online Evaluation And Selection ](reinforcement_learning/a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)**

:   提出 MACO 多智能体会话式 Bandit 框架，通过本地 agent 的在线淘汰和云服务器的自适应偏好查询机制，实现 LLM 响应的在线评估与用户偏好对齐，达到 $\tilde{O}(\sqrt{dMT})$ 的近优 regret 界。

**[Aligning Machiavellian Agents Behavior Steering Via Test-Tim](reinforcement_learning/aligning_machiavellian_agents_behavior_steering_via_test-tim.md)**

:   提出一种测试时策略塑形方法，通过轻量级伦理属性分类器在推理阶段插值修改预训练 RL 智能体的动作概率分布，无需重训练即可实现对多种伦理属性的细粒度行为引导。

**[Bamas Structuring Budget-Aware Multi-Agent Systems](reinforcement_learning/bamas_structuring_budget-aware_multi-agent_systems.md)**

:   提出 BAMAS 框架，通过整数线性规划（ILP）在预算约束下选择最优 LLM 组合，再用强化学习策略选择最佳协作拓扑（线性/星型/反馈/规划驱动），在 GSM8K/MBPP/MATH 上达到与 SOTA 多 Agent 系统相当的准确率，同时成本降低最高 86%。

**[Behaviour Policy Optimization Provably Lower Variance Return Estimates For Off-P](reinforcement_learning/behaviour_policy_optimization_provably_lower_variance_return_estimates_for_off-p.md)**

:   提出 Behaviour Policy Optimization (BPO)，通过优化一个专用行为策略来采集离策略数据，使得回报估计的方差可证明低于在策略采集，从而提升 REINFORCE 和 PPO 的样本效率与稳定性。

**[Beyond Monotonicity Revisiting Factorization Principles In Multi-Agent Q-Learnin](reinforcement_learning/beyond_monotonicity_revisiting_factorization_principles_in_multi-agent_q-learnin.md)**

:   通过动力系统分析证明：在近似贪心探索策略下，非单调值分解Q学习中所有违反IGM一致性的零损失解都是不稳定鞍点，只有IGM一致解才是稳定吸引子，因此无需单调性约束即可可靠收敛到最优解。

**[Beyond The Lower Bound Bridging Regret Minimization And Best Arm Identification ](reinforcement_learning/beyond_the_lower_bound_bridging_regret_minimization_and_best_arm_identification_.md)**

:   提出两种消除式算法 LexElim-Out 和 LexElim-In，首次在词典序多目标赌博机中同时解决遗憾最小化（RM）和最优臂识别（BAI）问题，其中 LexElim-In 通过跨目标信息共享突破了单目标问题的已知下界。

**[Bi-Level Contextual Bandits For Individualized Resource Allocation Under Delayed](reinforcement_learning/bi-level_contextual_bandits_for_individualized_resource_allocation_under_delayed.md)**

:   提出 MetaCUB——一种双层上下文赌博机框架，在延迟反馈、动态人群、冷却约束和公平性要求下实现个体化资源分配，元层优化子群预算分配保证公平，基层利用 UCB 策略选择最有潜力的个体。

**[Charteditor A Reinforcement Learning Framework For Robust Chart Editing](reinforcement_learning/charteditor_a_reinforcement_learning_framework_for_robust_chart_editing.md)**

:   提出 ChartEditVista 基准（7,964 样本、31 种图表类型）和 ChartEditor 模型，通过 GRPO 强化学习框架结合新颖的 rendering reward，仅用 3B 参数即在图表编辑任务上超越 GPT-4o 和多个 72B 级模型。

**[Chdp Cooperative Hybrid Diffusion Policies For Reinforcement Learning In Paramet](reinforcement_learning/chdp_cooperative_hybrid_diffusion_policies_for_reinforcement_learning_in_paramet.md)**

:   将混合动作空间问题建模为两个agent的全合作博弈，分别用离散和连续扩散策略生成动作，通过顺序更新和Q引导码本解决策略冲突与高维可扩展性问题，成功率最高提升19.3%。

**[Deep Predictive Discounted Counterfactual Regret Minimization](reinforcement_learning/deep_predictive_discounted_counterfactual_regret_minimization.md)**

:   提出VR-DeepDCFR+和VR-DeepPDCFR+两种无模型神经CFR算法，通过自举累积优势估计、折扣裁剪机制和基线方差缩减，首次将高级表格CFR变体（DCFR+/PDCFR+）有效整合到神经网络近似框架中，在典型不完全信息博弈中实现更快收敛。

**[Deepprooflog Efficient Proving In Deep Stochastic Logic Programs](reinforcement_learning/deepprooflog_efficient_proving_in_deep_stochastic_logic_programs.md)**

:   提出DeepProofLog（DPrL），一种基于随机逻辑程序的神经符号系统，通过在每个证明步骤引入神经网络参数化，并建立SLD解析过程与MDP的形式化映射，使得动态规划和强化学习技术可用于高效推理与学习，显著提升了神经符号系统的可扩展性。

**[Diffop Reinforcement Learning Of Optimization-Based Control Policies Via Implici](reinforcement_learning/diffop_reinforcement_learning_of_optimization-based_control_policies_via_implici.md)**

:   提出 DiffOP 框架，将优化型控制策略（如 MPC）视为可微分模块，通过隐式微分推导解析策略梯度，实现端到端强化学习训练，并给出首个非渐近收敛保证。

**[Discounted Cuts A Stackelberg Approach To Network Disruption](reinforcement_learning/discounted_cuts_a_stackelberg_approach_to_network_disruption.md)**

:   提出折扣切割（Discounted Cuts）数学模型，将经典 Most Vital Links 问题建模为 Stackelberg 博弈，系统研究8种折扣切割变体的计算复杂性分类，证明所有变体在有界亏格图上均可多项式时间求解。

**[Distilling Deep Reinforcement Learning Into Interpretable Fuzzy Rules An Explain](reinforcement_learning/distilling_deep_reinforcement_learning_into_interpretable_fuzzy_rules_an_explain.md)**

:   提出层次化 Takagi-Sugeno-Kang (TSK) 模糊分类器系统，将深度 RL 的神经网络策略蒸馏为人类可读的 IF-THEN 模糊规则，引入三个量化可解释性度量（FRAD、FSC、ASG），在 Lunar Lander 连续控制任务上以 81.48% 的保真度超越决策树 21 个百分点。

**[Do It For Her First-Order Temporal Logic Reward Specification In Reinforcement L](reinforcement_learning/do_it_for_her_first-order_temporal_logic_reward_specification_in_reinforcement_l.md)**

:   提出基于有限迹一阶时序逻辑模理论（LTLfMT）的新型奖励规范框架，用一阶逻辑公式替代手工编码的标注函数，结合 CRM 和 HER 解决逻辑规范固有的稀疏奖励问题，在连续控制任务中取得显著改进。

**[Does Self-Evaluation Enable Wireheading In Language Models](reinforcement_learning/does_self-evaluation_enable_wireheading_in_language_models.md)**

:   本文理论证明并实验验证了当语言模型的自我评估与奖励信号耦合时，模型会系统性地膨胀自评分（wireheading），而解耦自评分与奖励可以缓解这一问题；在Llama-3.1-8B和Mistral-7B上三个任务的实验表明，摘要等模糊任务中自评分膨胀高达0.92。

**[Drmd Deep Reinforcement Learning For Malware Detection Under Concept Drift](reinforcement_learning/drmd_deep_reinforcement_learning_for_malware_detection_under_concept_drift.md)**

:   本文首次将Android恶意软件检测重新表述为一步马尔可夫决策过程（MD-MDP），并训练基于PPO的深度强化学习智能体DRMD，在单一策略中统一了样本分类、拒绝和主动学习，在多年跨期评估中实现了平均8.66（仅分类）和10.90（含拒绝）的AUT提升，显著优于传统监督学习分类器应对概念漂移的能力。

**[Efficient Multiagent Planning Via Shared Action Suggestions](reinforcement_learning/efficient_multiagent_planning_via_shared_action_suggestions.md)**

:   提出 MCAS 算法，通过在去中心化 POMDP 中仅共享"建议动作"来推断其他智能体的信念状态，在大幅降低通信开销和计算复杂度的同时实现接近集中式方法的协调性能。

**[Explaining Decentralized Multi-Agent Reinforcement Learning Policies](reinforcement_learning/explaining_decentralized_multi-agent_reinforcement_learning_policies.md)**

:   提出首个针对去中心化多智能体强化学习（MARL）策略的可解释方法，包括基于 Hasse 图的策略摘要生成和基于查询的自然语言解释（When/Why Not/What），在四个 MARL 领域展示了通用性和计算效率，用户研究表明显著提升了人类对策略的理解和问答表现。

**[Formal Verification Of Diffusion Auctions](reinforcement_learning/formal_verification_of_diffusion_auctions.md)**

:   首次提出面向扩散拍卖（diffusion auctions）的形式化逻辑框架，引入 $n$ 卖家扩散激励逻辑 $\mathcal{L}^n$ 及其策略扩展 $\mathcal{SL}^n$，支持对拍卖属性（如 Nash 均衡、卖家策略存在性）的模型检测验证，分别建立了 P 和 PSPACE-complete 的复杂度结果。

**[G-Ubs Towards Robust Understanding Of Implicit Feedback Via Group-Aware User Beh](reinforcement_learning/g-ubs_towards_robust_understanding_of_implicit_feedback_via_group-aware_user_beh.md)**

:   提出 G-UBS（Group-aware User Behavior Simulation）范式，通过用户群组管理器（UGM）基于 LLM 的"总结-聚类-反思"流程生成群组画像，结合用户反馈建模器（UFM）的群组感知强化学习训练，在隐式反馈噪声下实现鲁棒的用户行为理解，同时构建了首个多模态隐式反馈视频推荐基准 IF-VR。

**[Good-For-Mdp State Reduction For Stochastic Ltl Planning](reinforcement_learning/good-for-mdp_state_reduction_for_stochastic_ltl_planning.md)**

:   提出一种新的 Good-for-MDP（GFM）自动机状态约简技术，通过 GFM→DBA→DCA→GFG 最小化→0/1-PA 的变换链显著减少自动机状态数量；同时为 $\textsf{GF}\varphi$（$\varphi$ 为 co-safety 公式）类公式提供直接的单指数构造方法，相比一般的双指数构造实现了指数级的状态数减少。

**[Hcpo Hierarchical Conductor-Based Policy Optimization In Multi-Agent Reinforceme](reinforcement_learning/hcpo_hierarchical_conductor-based_policy_optimization_in_multi-agent_reinforceme.md)**

:   提出 HCPO 算法，通过引入 conductor（指挥者）机制增强多智能体联合策略的表达能力和探索效率，构建类似 Gaussian mixture model 的联合策略框架，并证明两级策略更新的单调改进保证。

**[In-Token Rationality Optimization Towards Accurate And Concise Llm Reasoning Via](reinforcement_learning/in-token_rationality_optimization_towards_accurate_and_concise_llm_reasoning_via.md)**

:   提出 InTRO 框架，通过将模型的生成策略与其answer-conditioned后验对齐（KL散度最小化），在单次前向传播中实现token级探索和自生成反馈，从而在不依赖外部监督的情况下提升LLM推理的准确性与简洁性。

**[Infigui-G1 Advancing Gui Grounding With Adaptive Exploration Policy Optimization](reinforcement_learning/infigui-g1_advancing_gui_grounding_with_adaptive_exploration_policy_optimization.md)**

:   针对GUI定位中语义对齐的探索瓶颈，提出Adaptive Exploration Policy Optimization (AEPO)框架，通过多答案生成策略强制广泛探索、自适应探索奖励函数动态引导以及共线惩罚机制确保探索质量，显著提升多模态大模型在复杂GUI定位任务上的表现。

**[Intention-Guided Cognitive Reasoning For Egocentric Long-Term Action Anticipatio](reinforcement_learning/intention-guided_cognitive_reasoning_for_egocentric_long-term_action_anticipatio.md)**

:   提出INSIGHT框架，一个面向第一人称长期动作预测的两阶段统一框架：第一阶段通过手-物交互区域特征提取和动词-名词共现矩阵增强动作表示；第二阶段引入基于GRPO的强化学习认知推理模块，模拟"感知→推理→回答"的结构化认知过程进行意图推断和动作预测。

**[Know Your Trajectory -- Trustworthy Reinforcement Learning Deployment Through Im](reinforcement_learning/know_your_trajectory_--_trustworthy_reinforcement_learning_deployment_through_im.md)**

:   提出一种基于状态重要性指标的轨迹级解释框架，通过结合Q值差异和目标亲和度（radical term）对轨迹进行排序，并通过反事实推演验证所选最优轨迹的鲁棒优越性，为RL策略提供"为什么选这条路而非那条路？"的可信解释。

**[Language Model Distillation A Temporal Difference Imitation Learning Perspective](reinforcement_learning/language_model_distillation_a_temporal_difference_imitation_learning_perspective.md)**

:   从模仿学习/逆强化学习的视角重新审视语言模型蒸馏，提出利用教师模型输出分布的稀疏性（top-p token集中了96%以上概率质量），构建top-p MDP进行时序差分（TD）学习，证明了在缩减动作空间中的最优策略具有可界的次优性保证，并以IQL算法为基础实现的Bellman Distill方法在多个模型家族上超越了现有蒸馏方法。

**[Learning To Generate And Extract A Multi-Agent Collaboration Framework For Zero-](reinforcement_learning/learning_to_generate_and_extract_a_multi-agent_collaboration_framework_for_zero-.md)**

:   提出"提议-评估-修改"多智能体协作框架（生成智能体+评估智能体）解决零样本文档级事件论元提取（ZS-DEAE），通过生成智能体合成未见事件的训练数据，评估智能体评分引导强化学习迭代优化，同时提升合成数据质量和抽取性能。

**[Manilong-Shot Interaction-Aware One-Shot Imitation Learning For Long-Horizon Man](reinforcement_learning/manilong-shot_interaction-aware_one-shot_imitation_learning_for_long-horizon_man.md)**

:   提出 ManiLong-Shot 框架，通过交互感知的任务分解、不变区域预测和区域匹配三个模块，仅在10个短序列任务上训练即可泛化到20个未见长序列操作任务，单次模仿成功率 30.2%，相对SOTA提升22.8%。

**[Mars A Meta-Adaptive Reinforcement Learning Framework For Risk-Aware Multi-Agent](reinforcement_learning/mars_a_meta-adaptive_reinforcement_learning_framework_for_risk-aware_multi-agent.md)**

:   提出 MARS 框架，通过异构多智能体集成（每个智能体有不同风险偏好和 Safety-Critic）与元自适应控制器（MAC）的两层架构，在动态市场条件下实现风险感知的投资组合管理，显著降低最大回撤和波动率。

**[Mars Multi-Agent Adaptive Reasoning With Socratic Guidance F](reinforcement_learning/mars_multi-agent_adaptive_reasoning_with_socratic_guidance_f.md)**

:   提出 MARS 五智能体框架做自动提示优化（APO）：Planner 生成任务特定的优化轨迹，Teacher-Critic-Student 三体进行苏格拉底对话式迭代精炼 prompt（模拟文本空间中的伪梯度下降），Target 执行并反馈，整体建模为 POMDP，在 17 个数据集上平均超越前 SOTA（PE2）6.04%（通用任务）和 6.42%（领域任务），且仅需 1-shot 训练数据。

**[Mathsmith Towards Extremely Hard Mathematical Reasoning By Forging Synthetic Pro](reinforcement_learning/mathsmith_towards_extremely_hard_mathematical_reasoning_by_forging_synthetic_pro.md)**

:   提出 MathSmith 框架，通过从 PlanetMath 随机抽取数学概念对、采用9种预定义难度策略生成数学题目、并利用 GRPO 强化学习联合优化结构有效性/推理复杂度/答案一致性，生成的高难度合成问题在 AIME 和 OlympiadBench 上显著提升 LLM 数学推理能力。

**[Mmhops-R1 Multimodal Multi-Hop Reasoning](reinforcement_learning/mmhops-r1_multimodal_multi-hop_reasoning.md)**

:   提出了 MMhops 基准（31K 样本、3-4 跳推理深度）和 MMhops-R1 框架，通过强化学习训练 MLLM 自主规划推理路径、动态调用图像/文本检索器，实现多模态多跳推理，7B 模型超越 72B 基线和现有 mRAG 方法。

**[Object-Centric Latent Action Learning](reinforcement_learning/object-centric_latent_action_learning.md)**

:   提出以物体为中心的潜在动作学习框架，利用自监督的物体分解（VideoSAUR）将场景中任务相关实体与视觉干扰（动态背景等）分离，使潜在动作模型（LAPO）在有干扰的视频中性能退化减少约50%，并通过线性动作探针自动选择控制相关的 slot。

**[Object-Centric World Models For Causality-Aware Reinforcement Learning](reinforcement_learning/object-centric_world_models_for_causality-aware_reinforcement_learning.md)**

:   提出 STICA 框架，通过统一的以物体为中心的 Transformer 架构实现世界模型、策略网络和价值网络，其中世界模型将观测分解为独立物体的隐状态进行 token 级动力学预测，策略和价值网络通过因果注意力机制估计 token 级因果关系实现因果感知决策，在 Safety Gym 和 OCVRL 基准上显著超越 DreamerV3 等 SOTA 方法。

**[One-Step Generative Policies With Q-Learning A Reformulation Of Meanflow](reinforcement_learning/one-step_generative_policies_with_q-learning_a_reformulation_of_meanflow.md)**

:   将MeanFlow重新形式化为残差映射 $g(a_t,b,t) = a_t - u(a_t,b,t)$，实现一步噪声→动作的生成式策略，无需蒸馏或多步ODE积分，可直接与Q-learning联合训练，在OGBench和D4RL的73个任务上取得强性能。

**[Partial Action Replacement Tackling Distribution Shift In Offline Marl](reinforcement_learning/partial_action_replacement_tackling_distribution_shift_in_offline_marl.md)**

:   提出部分动作替换（PAR）原理，从理论上证明在分解行为策略下分布偏移随偏离智能体数量线性增长（而非联合动作空间的指数增长），并基于此开发 SPaCQL 算法，通过 Q 函数集成的不确定性动态加权不同 PAR 策略，在 Random 和 Medium-Replay 数据集上显著超越所有基线。

**[Perturbing Best Responses In Zero-Sum Games](reinforcement_learning/perturbing_best_responses_in_zero-sum_games.md)**

:   本文研究在零和博弈的最优响应预言机（BRO）中引入随机扰动，证明了随机虚拟对弈（SFP）在纯策略数量 $n$ 上可实现 $O(\frac{\log n}{\varepsilon^2})$ 的期望迭代次数，并提出了随机双重预言机（SDO）算法，在特定博弈结构下同样实现对数级收敛。

**[Provably Efficient Multi-Objective Bandit Algorithms Under Preference-Centric Cu](reinforcement_learning/provably_efficient_multi-objective_bandit_algorithms_under_preference-centric_cu.md)**

:   首次从理论角度研究显式用户偏好下的多目标多臂赌博机（MO-MAB）定制化优化问题，提出 PAMO-MAB 框架并针对"未知偏好"和"隐藏偏好"两种场景分别设计 PRUCB-UP 和 PRUCB-HP 算法，通过偏好估计 + 偏好感知优化的双组件框架实现近最优遗憾界，证明了 preference-free 算法在 Pareto 前沿冲突时必然产生 $\Omega(T)$ 线性遗憾。

**[Qimeng-Kernel Macro-Thinking Micro-Coding Paradigm For Llm-Based High-Performanc](reinforcement_learning/qimeng-kernel_macro-thinking_micro-coding_paradigm_for_llm-based_high-performanc.md)**

:   提出 MTMC（Macro Thinking Micro Coding）分层框架，通过强化学习驱动轻量LLM产生高层优化策略（Macro Thinking），再由通用LLM逐步实现代码（Micro Coding），将GPU内核生成的正确性和性能问题解耦，在KernelBench上实现近100%准确率和2.2×超越专家优化PyTorch Eager内核的加速。

**[Realistic Curriculum Reinforcement Learning For Autonomous And Sustainable Marin](reinforcement_learning/realistic_curriculum_reinforcement_learning_for_autonomous_and_sustainable_marin.md)**

:   提出一个课程强化学习（CRL）框架用于自主且可持续的海洋船舶航行，集成了基于真实AIS数据的仿真环境、扩散模型增强的动态海上交通模拟、以及机器学习燃油消耗预测模块，通过多目标奖励函数同时优化航行安全性、排放减少、时效性和目标完成。

**[Reasoning Or Memorization Unreliable Results Of Reinforcement Learning Due To Da](reinforcement_learning/reasoning_or_memorization_unreliable_results_of_reinforcement_learning_due_to_da.md)**

:   本文通过系统性的数据泄露审计揭示了Qwen2.5系列在MATH-500等标准数学基准上存在严重的数据污染问题，指出近期"虚假奖励也能提升数学推理"的发现是污染所致的虚假结论，并构建了完全无泄露的RandomCalculation基准验证只有正确奖励信号才能带来真实的推理提升。

**[Reasoning With Exploration An Entropy Perspective](reinforcement_learning/reasoning_with_exploration_an_entropy_perspective.md)**

:   本文从熵（entropy）的视角分析LLM中探索性推理行为（关键token、自我反思、稀有行为）与高熵区域的正相关性，提出一种极简的熵基优势函数塑形方法——仅需一行代码修改——即可显著增强LLM的Pass@K推理能力边界。

**[Regal A First Look At Ppo-Based Legal Ai For Judgment Prediction And Summarizati](reinforcement_learning/regal_a_first_look_at_ppo-based_legal_ai_for_judgment_prediction_and_summarizati.md)**

:   本文首次将基于PPO的强化学习（RLAIF）应用于印度法律领域的判决预测与摘要生成任务，虽然性能未超越SFT和商业模型，但作为定位论文（position paper）揭示了RL在法律NLP中的关键挑战与未来方向。

**[Revealing Pomdps Qualitative And Quantitative Analysis For Parity Objectives](reinforcement_learning/revealing_pomdps_qualitative_and_quantitative_analysis_for_parity_objectives.md)**

:   本文证明了揭示型POMDPs（revealing POMDPs）在奇偶目标（parity objectives）下的极限确定性分析（limit-sure analysis）等价于几乎确定性分析（EXPTIME-complete），且定量分析（quantitative analysis）也可在EXPTIME内完成，解决了该子类的两个重要开放问题。

**[Risk-Sensitive Exponential Actor Critic](reinforcement_learning/risk-sensitive_exponential_actor_critic.md)**

:   针对 entropic risk measure 下 policy gradient 的高方差和数值不稳定问题，推导了完整的 on/off-policy 风险敏感策略梯度定理，并提出 rsEAC 算法，通过 log-domain critic 参数化和梯度归一化裁剪机制实现稳定的风险敏感连续控制。

**[Rlslm A Hybrid Reinforcement Learning Framework Aligning Rule-Based Social Locom](reinforcement_learning/rlslm_a_hybrid_reinforcement_learning_framework_aligning_rule-based_social_locom.md)**

:   本文提出RLSLM，一种将心理学实验驱动的规则式社交运动模型（SLM）嵌入强化学习奖励函数的混合框架，使智能体在人群环境中高效学习符合人类社交规范的导航策略，VR实验验证其舒适度评分显著优于现有规则式基线。

**[Safemil Learning Offline Safe Imitation Policy From Non-Preferred Trajectories](reinforcement_learning/safemil_learning_offline_safe_imitation_policy_from_non-preferred_trajectories.md)**

:   本文提出SafeMIL，通过将代价函数学习建模为多实例学习（MIL）问题，从有限的非偏好轨迹和大量无标签轨迹中学习安全的模仿策略，在不需要逐步reward/cost标注的情况下，实现约束满足性能比最佳基线提升3.7倍。

**[Scalable Multi-Objective And Meta Reinforcement Learning Via Gradient Estimation](reinforcement_learning/scalable_multi-objective_and_meta_reinforcement_learning_via_gradient_estimation.md)**

:   本文提出PolicyGradEx，通过一阶梯度近似和代理模型高效估计任意任务子集上的策略适应性能，构建任务亲和度矩阵并通过凸优化进行任务分组，在多目标RL和元RL基准上平均超越SOTA基线16%，速度提升高达26倍。

**[Speculative Sampling With Reinforcement Learning](reinforcement_learning/speculative_sampling_with_reinforcement_learning.md)**

:   提出 Re-SpS，首个将推测采样（Speculative Sampling）的草稿树超参数优化建模为 MDP 并用强化学习求解的框架，通过特征复用和动作缓存两大设计，在不损失输出保真度的前提下，相比 EAGLE-3 实现最高 1.12× 的额外加速。

**[Start Small Think Big Curriculum-Based Relative Policy Optimization For Visual G](reinforcement_learning/start_small_think_big_curriculum-based_relative_policy_optimization_for_visual_g.md)**

:   发现 CoT 推理在视觉定位任务中可能适得其反，提出 CuRPO（Curriculum-based Relative Policy Optimization），利用 CoT 长度和 gIoU 奖励作为数据复杂度指标进行课程式 RL 训练，在 RefCOCO 上比 Visual-RFT 提升最高 +12.52 mAP。

**[Stelar-Vision Self-Topology-Aware Efficient Learning For Aligned Reasoning In Vi](reinforcement_learning/stelar-vision_self-topology-aware_efficient_learning_for_aligned_reasoning_in_vi.md)**

:   提出 STELAR-Vision，一个拓扑感知的视觉语言推理训练框架，通过 TopoAug 数据生成管线引入 Chain/Tree/Graph 多种推理拓扑结构，配合 SFT+RL 后训练，在分布内外数据集上分别提升 9.7% 和最高 28.4% 的准确率，并通过 Frugal Learning 减少 18.1% 的输出长度。

**[Tadarag Task Adaptive Retrieval-Augmented Generation Via On-The-Fly Knowledge Gr](reinforcement_learning/tadarag_task_adaptive_retrieval-augmented_generation_via_on-the-fly_knowledge_gr.md)**

:   提出 TAdaRAG，一个任务自适应的 RAG 框架，通过意图驱动的模板路由、监督微调和 REINFORCE 强化学习实现实时知识图谱构建，有效解决传统 RAG 的分块截断幻觉、推理链断裂和无关信息干扰三大问题，在 6 个公开数据集和 1 个商业场景基准上取得 SOTA。

**[Test-Driven Reinforcement Learning In Continuous Control](reinforcement_learning/test-driven_reinforcement_learning_in_continuous_control.md)**

:   提出 Test-driven Reinforcement Learning (TdRL) 框架，用多个测试函数（pass-fail 测试定义最优目标 + indicative 测试引导学习）替代单一奖励函数表示任务目标，通过字典序启发式轨迹比较学习回报函数，在 DeepMind Control Suite 上匹配或超越手工奖励方法，天然支持多目标优化。

**[Textshield-R1 Reinforced Reasoning For Tampered Text Detection](reinforcement_learning/textshield-r1_reinforced_reasoning_for_tampered_text_detection.md)**

:   提出 TextShield-R1，首个基于强化学习的多模态大模型篡改文本检测方法，通过取证持续预训练（从自然图像到文本图像的课程）、GRPO 强化学习（五种精心设计的奖励函数减少标注依赖）和 OCR 矫正（利用 MLLM 的文本识别能力提升定位精度），配合新提出的 TFR 基准（45K+ 图像、16 种语言、10 种篡改技术），显著推进了可解释性篡改文本检测的 SOTA。

**[Think Speak Decide Language-Augmented Multi-Agent Reinforcement Learning For Eco](reinforcement_learning/think_speak_decide_language-augmented_multi-agent_reinforcement_learning_for_eco.md)**

:   提出 LAMP 框架，通过 Think–Speak–Decide 三阶段流水线将 LLM 驱动的语言推理与 MARL 策略优化相融合，使经济决策智能体能够理解和利用自然语言信息（如新闻、对话），在经济仿真环境中累计回报超越纯 MARL 基线 63.5%、LLM-only 基线 34.0%。

**[Thinker Training Llms In Hierarchical Thinking For Deep Search Via Multi-Turn In](reinforcement_learning/thinker_training_llms_in_hierarchical_thinking_for_deep_search_via_multi-turn_in.md)**

:   提出 Thinker 框架，通过分层思维（breadth decomposition + depth solving）和双重表征（自然语言 + 逻辑函数）实现结构化的深度搜索推理，配合知识边界判定减少不必要检索，以 SFT 方式训练，在多个 QA 基准上显著超越 RL-based deep search 方法。

**[Towermind A Tower Defence Game Learning Environment And Benchmark For Llm As Age](reinforcement_learning/towermind_a_tower_defence_game_learning_environment_and_benchmark_for_llm_as_age.md)**

:   提出 TowerMind，一个基于塔防游戏的轻量级多模态环境，用于评估 LLM 的长期规划和决策能力，揭示了当前 LLM 与人类专家之间仍存在显著性能差距（最佳模型仅达人类专家 42% 的得分），并识别出规划验证不足、缺乏多终态思维、动作空间利用不充分等行为缺陷。

**[Vision-Language Reasoning For Geolocalization A Reinforcement Learning Approach](reinforcement_learning/vision-language_reasoning_for_geolocalization_a_reinforcement_learning_approach.md)**

:   提出 Geo-R，一个无需检索的推理驱动图像地理定位框架，通过 Chain-of-Region 层次化推理范式和基于 Haversine 距离的坐标对齐奖励的强化学习策略，在 IM2GPS3K 上 1km 准确率达 18.10%，超越所有无检索方法并逼近检索方法。

**[Well Begun Half Done Reinforcement Learning With Prefix Optimization For Llm Rea](reinforcement_learning/well_begun_half_done_reinforcement_learning_with_prefix_optimization_for_llm_rea.md)**

:   发现 LLM 推理中的"起始锁定效应"（Beginning Lock-in Effect）——初始推理过程显著决定后续轨迹和最终结果，据此提出 PPPO 方法，仅优化前缀 token（约 26% 的 token）即可实现高达 18.02% 的准确率提升，同时减少输出 token 数量达 18.35%。

**[When Eyes And Ears Disagree Can Mllms Discern Audio-Visual Confusion](reinforcement_learning/when_eyes_and_ears_disagree_can_mllms_discern_audio-visual_confusion.md)**

:   发现多模态大语言模型（MLLMs）在音视觉信息不对称时严重受视觉主导而无法识别缺失音频的"音视觉混淆"现象，提出 AV-ConfuseBench 基准和 RL-CoMM 方法（引入外部音频模型做参考的阶梯式推理奖励 + 答案置信度优化），在仅用约 20% 训练数据的情况下提升基线模型准确率 10~30%。

**[Where And What Matters Sensitivity-Aware Task Vectors For Many-Shot Multimodal I](reinforcement_learning/where_and_what_matters_sensitivity-aware_task_vectors_for_many-shot_multimodal_i.md)**

:   提出 STV 框架，通过激活差值（activation delta）识别对上下文信息敏感的注意力头位置，并利用强化学习从预聚类的激活库中选择最优任务向量进行插入，在不增加输入长度的前提下实现高效的多模态 many-shot 上下文学习。

**[Where To Start Alignment Diffusion Large Language Model May Demand A Distinct Po](reinforcement_learning/where_to_start_alignment_diffusion_large_language_model_may_demand_a_distinct_po.md)**

:   首次系统分析扩散大语言模型（dLLM）的安全特性，发现与自回归 LLM 不同，dLLM 中**中间 token** 对安全性更关键，且攻击者受限于模型固有的顺序生成倾向难以操控中间 token，基于此不对称性提出 MOSA（Middle-tOken Safety Alignment）防御方法。

---

## 🧑 人体理解 { #human_understanding }

**[10 Open Challenges Steering The Future Of Vision-Language-Ac](human_understanding/10_open_challenges_steering_the_future_of_vision-language-ac.md)**

:   系统梳理 VLA 模型面临的 10 大开放挑战——多模态感知、鲁棒推理、高质量训练数据、评估、跨机器人动作泛化、资源效率、全身协调、安全保障、Agent 框架、人机协作——并讨论空间理解、世界动力学建模、后训练和数据合成四大新兴趋势。

**[Ahan Asymmetric Hierarchical Attention Network For Identical](human_understanding/ahan_asymmetric_hierarchical_attention_network_for_identical.md)**

:   针对同卵双胞胎人脸验证这一极端细粒度识别挑战，提出 AHAN 多流架构，通过层次交叉注意力 (HCA) 对语义面部区域做多尺度分析、面部不对称注意力模块 (FAAM) 捕获左右脸差异签名、以及双胞胎感知配对交叉注意力 (TA-PWCA) 训练正则化，在 ND_TWIN 数据集上将双胞胎验证精度从 88.9% 提升至 92.3%（+3.4%）。

**[Anti-Adversarial Learning Desensitizing Prompts For Large La](human_understanding/anti-adversarial_learning_desensitizing_prompts_for_large_la.md)**

:   提出 PromptObfus，通过"反对抗学习"思路将用户 prompt 中的敏感词替换为语义不同但不影响任务输出的词，从而在不降低远端 LLM 任务表现的前提下彻底消除显式隐私泄露，并将隐式隐私推理攻击成功率降低 62.70%。

**[Authority Backdoor A Certifiable Backdoor Mechanism For Authoring Dnns](human_understanding/authority_backdoor_a_certifiable_backdoor_mechanism_for_authoring_dnns.md)**

:   提出 Authority Backdoor，将硬件指纹作为后门触发器嵌入 DNN，使模型仅在授权设备上正常工作，并通过随机平滑实现可认证鲁棒性，抵御自适应触发器逆向攻击。

**[Auto-Pre An Automatic And Cost-Efficient Peer-Review Framework For Language Gene](human_understanding/auto-pre_an_automatic_and_cost-efficient_peer-review_framework_for_language_gene.md)**

:   提出 Auto-PRE 框架，通过自动资格考试从一致性、相关性、自信度三个维度筛选合格的 LLM 评估者，在无需人工标注的前提下实现了 SOTA 评估性能并大幅降低成本。

**[Behavior Tokens Speak Louder Disentangled Explainable Recommendation With Behavi](human_understanding/behavior_tokens_speak_louder_disentangled_explainable_recommendation_with_behavi.md)**

:   提出 BEAT 框架，通过向量量化自编码将用户/物品的行为表征离散化为可解释的 behavior tokens，结合多层级语义监督将协同过滤信号对齐到冻结 LLM 的语义空间，实现零样本可解释推荐。

**[Bias Association Discovery Framework For Open-Ended Llm Generations](human_understanding/bias_association_discovery_framework_for_open-ended_llm_generations.md)**

:   提出偏见关联发现框架 BADF，通过分析 LLM 开放式故事生成中的叙事内容，系统性地提取人口统计身份与描述性概念之间的已知和未知偏见关联，突破了以往依赖预定义偏见概念的局限。

**[Can Llms Truly Embody Human Personality Analyzing Ai And Human Behavior Alignmen](human_understanding/can_llms_truly_embody_human_personality_analyzing_ai_and_human_behavior_alignmen.md)**

:   提出首个系统对比框架，在配对的冲突调解场景中直接比较人类与人格提示LLM的策略行为差异，发现LLM在人格-行为映射上与人类存在显著偏差，挑战了"人格提示即可代理人类行为"的假设。

**[Ccfqa A Benchmark For Cross-Lingual And Cross-Modal Speech And Text Factuality E](human_understanding/ccfqa_a_benchmark_for_cross-lingual_and_cross-modal_speech_and_text_factuality_e.md)**

:   提出 CCFQA——首个覆盖 8 种语言、14,400 条完全平行语音-文本事实问答样本的跨语言跨模态基准，支持 QA/XQA/SQA/XSQA 四种任务设定，系统揭示了现有 MLLM 在语言和模态切换下的事实不一致性；同时提出 LLM-SQA，以英语为桥接语言、仅 5-shot 即实现跨语言语音问答迁移，在 XSQA 上 F1 达 51.4 超越 GPT-4o-mini-Audio（45.7）。

**[Clip-Fti Fine-Grained Face Template Inversion Via Clip-Driven Attribute Conditio](human_understanding/clip-fti_fine-grained_face_template_inversion_via_clip-driven_attribute_conditio.md)**

:   首次利用 CLIP 提取面部细粒度语义属性嵌入来辅助人脸模板反演（FTI），通过跨模态特征交互网络将泄露模板与属性嵌入融合并投影到 StyleGAN 潜空间，生成身份一致且属性细节更丰富的人脸图像，在识别准确率、属性相似度和跨模型攻击迁移性上均超越 SOTA。

**[Clippan Adapting Clip As A Supervisor For Unsupervised Pansharpening](human_understanding/clippan_adapting_clip_as_a_supervisor_for_unsupervised_pansharpening.md)**

:   提出 CLIPPan，通过轻量微调 CLIP 使其理解多光谱/全色/高分辨率多光谱图像类型及全色锐化过程，然后利用 Wald 协议等文本提示作为语义监督信号，实现无需地面真值的全分辨率无监督全色锐化，可作为即插即用模块兼容任意全色锐化骨干网络。

**[Coordar One-Reference 6D Pose Estimation Of Novel Objects Via Autoregressive Coo](human_understanding/coordar_one-reference_6d_pose_estimation_of_novel_objects_via_autoregressive_coo.md)**

:   提出 CoordAR，将单参考视图 6D 位姿估计中的 3D-3D 对应关系建模为离散 token 的自回归生成问题，通过坐标图 token 化、模态解耦编码和自回归 Transformer 解码器，在多个基准上显著超越现有单视图方法，并对对称、遮挡等挑战场景展现强鲁棒性。

**[Deig Detail-Enhanced Instance Generation With Fine-Grained Semantic Control](human_understanding/deig_detail-enhanced_instance_generation_with_fine-grained_semantic_control.md)**

:   提出 DEIG，一个面向细粒度多实例图像生成的框架，通过实例细节提取器（IDE）将 LLM 编码器的高维嵌入蒸馏为紧凑的实例感知表示，并用细节融合模块（DFM）的实例掩码注意力防止属性泄漏，在多属性（颜色+材质+纹理）复合描述的生成任务上大幅超越现有方法。

**[Dexterous Manipulation Transfer Via Progressive Kinematic-Dynamic Alignment](human_understanding/dexterous_manipulation_transfer_via_progressive_kinematic-dynamic_alignment.md)**

:   提出 PKDA 框架，通过渐进式运动学-动力学对齐，将人手操作视频自动转化为多指灵巧手的高质量操作轨迹，平均迁移成功率达 73%。

**[Distributionally Robust Online Markov Game With Linear Function Approximation](human_understanding/distributionally_robust_online_markov_game_with_linear_function_approximation.md)**

:   本文研究具有线性函数近似的在线分布鲁棒马尔可夫博弈，首次识别了该设定下的学习困难性，并提出 DR-CCE-LSI 算法，在特定特征映射条件下实现了关于特征维度 $d$ 的极小极大最优样本复杂度。

**[Efficient And Reliable Hitting-Set Computations For The Implicit Hitting Set App](human_understanding/efficient_and_reliable_hitting-set_computations_for_the_implicit_hitting_set_app.md)**

:   针对隐式击中集框架中击中集组件依赖商用IP求解器带来的数值不稳定问题，提出基于伪布尔推理和随机局部搜索的替代方案及混合策略，实现了首个可认证的IHS计算并在1786个基准实例上展示了效率与可靠性的有效权衡。

**[Enhancing Noise Resilience In Face Clustering Via Sparse Differential Transforme](human_understanding/enhancing_noise_resilience_in_face_clustering_via_sparse_differential_transforme.md)**

:   提出预测驱动的 Top-K Jaccard 相似度系数提升邻居纯度，配合稀疏差分 Transformer（SDT）消除噪声注意力，在 MS-Celeb-1M 等大规模人脸聚类数据集上达到 SOTA 性能。

**[Enhancing Robustness Of Offline Reinforcement Learning Under Data Corruption Via](human_understanding/enhancing_robustness_of_offline_reinforcement_learning_under_data_corruption_via.md)**

:   首次将 Sharpness-Aware Minimization (SAM) 作为即插即用优化器应用于离线 RL，假设数据损坏导致损失景观中出现尖锐极小值从而泛化差，SAM 通过寻找平坦极小值提升鲁棒性，在 D4RL 基准上 IQL+SAM 平均得分从 34.47 提升到 44.40。

**[Facial-R1 Aligning Reasoning And Recognition For Facial Emotion Analysis](human_understanding/facial-r1_aligning_reasoning_and_recognition_for_facial_emotion_analysis.md)**

:   提出 Facial-R1，一个三阶段对齐训练框架（SFT → RL → 数据合成），通过将 AU 和情绪标签作为可验证奖励信号来对齐 VLM 的推理过程与情绪识别结果，在 8 个基准上达到 SOTA，并构建了 FEA-20K 数据集。

**[Failures To Surface Harmful Contents In Video Large Language Models](human_understanding/failures_to_surface_harmful_contents_in_video_large_language_models.md)**

:   本文首次系统分析了 VideoLLM 的安全性，揭示了三种结构性设计缺陷（稀疏时间采样、空间 token 下采样、模态融合不平衡），使得视频中清晰可见的有害内容在模型生成的文本摘要中被遗漏（omission rate 超 90%），并设计了三种零查询黑盒攻击来验证漏洞严重性。

**[Few-Shot Precise Event Spotting Via Unified Multi-Entity Graph And Distillation](human_understanding/few-shot_precise_event_spotting_via_unified_multi-entity_graph_and_distillation.md)**

:   提出 UMEG-Net，面向少样本精确事件定位（PES）任务，通过构建统一多实体图（融合人体骨架、运动物体关键点和环境标志点），结合高效的时空图卷积和无参数多尺度时序平移模块，并通过多模态知识蒸馏将图特征迁移至 RGB 学生网络，在五个运动数据集上以极少标注数据显著超越现有方法。

**[Fine-Grained Dino Tuning With Dual Supervision For Face Forgery Detection](human_understanding/fine-grained_dino_tuning_with_dual_supervision_for_face_forgery_detection.md)**

:   提出 DFF-Adapter（DeepFake Fine-Grained Adapter），针对 DINOv2 设计的轻量级深度伪造检测微调方案。通过在每个 Transformer 块中注入三分支适配器（真实性检测头、伪造类型分类头、共享头），结合 Forgery-Aware Multi-Head Router 实现子空间级 LoRA 专家动态路由，利用辅助的伪造类型分类任务增强主任务的伪影敏感性，仅 3.5M 可训练参数即在多个跨数据集评估中达到 SOTA。

**[Finextrol Controllable Motion Generation Via Fine-Grained Text](human_understanding/finextrol_controllable_motion_generation_via_fine-grained_text.md)**

:   提出 FineXtrol 框架，利用带时间标注的细粒度身体部位文本描述作为控制信号，通过双分支 ControlNet 架构和层级对比学习增强文本编码器的区分能力，实现高效、用户友好且精确的可控人体动作生成，在 HumanML3D 上多身体部位控制性能显著优于现有方法。

**[From Ids To Semantics A Generative Framework For Cross-Domain Recommendation Wit](human_understanding/from_ids_to_semantics_a_generative_framework_for_cross-domain_recommendation_wit.md)**

:   提出 GenCDR 框架，通过领域自适应语义分词和跨域自回归推荐两大模块，首次将生成式语义 ID 范式引入 LLM 驱动的跨域推荐，有效解决传统方法中 item ID 不可迁移和领域个性化建模不足的问题。

**[From Imitation To Discrimination Toward A Generalized Curriculum Advantage Mecha](human_understanding/from_imitation_to_discrimination_toward_a_generalized_curriculum_advantage_mecha.md)**

:   提出 CAPO（Curriculum Advantage Policy Optimization），一种基于优势信号的自适应课程机制，通过先模仿（仅正向优势样本）再判别（引入负向信号）的两阶段策略，稳定且显著提升 LLM 在数学推理和多模态 GUI 推理任务上的表现。

**[Gazeinterpreter Parsing Eye Gaze To Generate Eye-Body-Coordinated Narrations](human_understanding/gazeinterpreter_parsing_eye_gaze_to_generate_eye-body-coordinated_narrations.md)**

:   提出 GazeInterpreter，一种基于 LLM 的层次化框架，通过符号化眼动解析器将原始注视信号转化为文本叙述，再与身体运动叙述整合生成眼-体协调描述，并通过自我纠正循环迭代优化，显著提升文本驱动的运动生成、动作预测和行为摘要等下游任务的性能。

**[Generating Attribute-Aware Human Motions From Textual Prompt](human_understanding/generating_attribute-aware_human_motions_from_textual_prompt.md)**

:   提出 AttrMoGen 框架，通过基于结构因果模型（SCM）的因果信息瓶颈将动作语义与人体属性（年龄、性别等）解耦，生成属性感知的人体运动，并构建了首个包含广泛属性标注的大规模文本-运动数据集 HumanAttr。

**[Hilomix Robust High- And Low-Frequency Graph Learning Framework For Mixing Addre](human_understanding/hilomix_robust_high-_and_low-frequency_graph_learning_framework_for_mixing_addre.md)**

:   提出 HiLoMix，一种针对混币地址关联任务的鲁棒图学习框架，通过异质属性混合交互图（HAMIG）、频率感知图对比学习和基于置信度的标签加权监督学习，分别解决图稀疏、标签稀缺和标签噪声三大挑战，在 F1、AUC、MRR 上分别超越次优基线 5.69%、7.34% 和 15.61%。

**[Improving Sparse Imu-Based Motion Capture With Motion Label Smoothing](human_understanding/improving_sparse_imu-based_motion_capture_with_motion_label_smoothing.md)**

:   提出 Motion Label Smoothing，将经典 label smoothing 从分类任务适配到稀疏IMU运动捕捉中，通过融合骨骼结构感知的Perlin噪声作为平滑标签，在不修改模型架构的前提下以即插即用方式提升三种SOTA方法在四个数据集上的精度，GlobalPose在TotalCapture上SIP误差降低20.41%。

**[Irote Human-Like Traits Elicitation Of Large Language Model Via In-Context Self-](human_understanding/irote_human-like_traits_elicitation_of_large_language_model_via_in-context_self-.md)**

:   提出 IROTE，一种基于信息瓶颈理论的上下文自我反思优化方法，通过迭代生成并优化紧凑且富有唤起力的文本"自我反思"（self-reflection），无需微调即可稳定地激发 LLM 在多种下游任务中表现出目标人类特质（价值观、道德、人格），一致性超越现有基线。

**[Midb Multilingual Instruction Data Booster For Enhancing Cultural Equality In Mu](human_understanding/midb_multilingual_instruction_data_booster_for_enhancing_cultural_equality_in_mu.md)**

:   提出 MIDB（多语言指令数据增强器），通过 36.8k 人类语言专家标注的修订样本训练一个统一模型，自动修复多语言合成指令数据中的内容错误、机器翻译缺陷和本地化不足问题，显著提升 16 种语言的指令数据质量和下游 LLM 的多语言/文化理解能力。

**[Mmpred Radar-Based Human Motion Prediction In The Dark](human_understanding/mmpred_radar-based_human_motion_prediction_in_the_dark.md)**

:   首次将毫米波雷达引入人体运动预测(HMP)任务，提出mmPred——基于扩散模型的框架，通过双域历史运动表示（时域姿态细化TPR + 频域主导运动FDM）和全局骨骼关系Transformer(GST)，有效抑制雷达特有的噪声和时序不一致性，在mmBody和mm-Fi数据集上分别超越SOTA方法8.6%和22%。

**[Modality-Aware Bias Mitigation And Invariance Learning For Unsupervised Visible-](human_understanding/modality-aware_bias_mitigation_and_invariance_learning_for_unsupervised_visible-.md)**

:   针对无监督可见光-红外行人重识别（USVI-ReID）中跨模态关联不可靠的核心问题，提出模态感知的 Jaccard 距离修正和"分裂-对比"不变性学习策略，通过消除模态偏差实现可靠的全局跨模态聚类和特征对齐，在 SYSU-MM01 和 RegDB 上达到 SOTA。

**[Modelling The Effects Of Hearing Loss On Neural Coding In The Auditory Midbrain ](human_understanding/modelling_the_effects_of_hearing_loss_on_neural_coding_in_the_auditory_midbrain_.md)**

:   提出 ψ-ICNet，一种变分条件深度神经网络模型，通过仅 6 个可学习的条件参数 ψ 来编码听力损失的效应，从真实神经活动记录中直接学习听力损失的低维表示空间，在预测正常和听力受损动物的听觉中脑神经响应方面达到与动物特定模型相当的精度，并可通过贝叶斯优化快速拟合未见过的新动物。

**[Mvgd-Net A Novel Motion-Aware Video Glass Surface Detection Network](human_understanding/mvgd-net_a_novel_motion-aware_video_glass_surface_detection_network.md)**

:   基于"玻璃表面上反射/透射层物体的运动速度与非玻璃区域不一致"的物理观察，提出 MVGD-Net，通过光流运动线索引导视频中玻璃表面检测，包含跨尺度多模态融合（CMFM）、历史引导注意力（HGAM）、时序交叉注意力（TCAM）和时空解码器（TSD）四个核心模块，并构建了包含 312 视频 19,268 帧的大规模数据集 MVGD-D。

**[New Synthetic Goldmine Hand Joint Angle-Driven Emg Data Generation Framework For](human_understanding/new_synthetic_goldmine_hand_joint_angle-driven_emg_data_generation_framework_for.md)**

:   提出 SeqEMG-GAN，一种基于手部关节角度序列驱动的条件对抗生成框架，通过角度编码器、双层上下文编码器（含新颖 Ang2Gist 单元）、深度卷积生成器和多视角判别器的联合设计，从关节运动学轨迹合成高保真 EMG 信号，实现对未见手势的零样本生成，合成数据与真实数据混合训练将分类精度从 57.77% 提升至 60.53%。

**[Nurbgen High-Fidelity Text-To-Cad Generation Through Llm-Driven Nurbs Modeling](human_understanding/nurbgen_high-fidelity_text-to-cad_generation_through_llm-driven_nurbs_modeling.md)**

:   首次提出基于NURBS表面表示的文本到CAD生成框架NURBGen，通过微调LLM将自然语言描述转换为结构化的NURBS参数JSON，并引入混合表示（untrimmed NURBS + 解析原语）和大规模partABC数据集，在几何保真度和尺寸精度上显著超越现有方法。

**[Opera A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architect](human_understanding/opera_a_reinforcement_learning--enhanced_orchestrated_planner-executor_architect.md)**

:   提出 OPERA 框架，通过 Goal Planning Module 和 Reason-Execute Module 的分层架构，结合专为多 agent 设计的 MAPGRPO 训练算法，大幅提升 reasoning-oriented multi-hop retrieval 性能。

**[Pa-Fas Towards Interpretable And Generalizable Multimodal Face Anti-Spoofing Via](human_understanding/pa-fas_towards_interpretable_and_generalizable_multimodal_face_anti-spoofing_via.md)**

:   提出PA-FAS框架，通过推理路径增强（Reasoning Path Augmentation）策略和答案打乱机制，解决了多模态FAS中SFT+RL范式的两大瓶颈（推理路径多样性不足和推理捷径问题），首次在统一框架中同时实现多模态融合、域泛化和可解释性。

**[Pararevsnn A Parallel Reversible Spiking Neural Network For Efficient Training A](human_understanding/pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)**

:   提出ParaRevSNN，一种并行可逆脉冲神经网络架构，通过重新设计可逆块间的数据依赖关系解耦顺序计算约束，在保持可逆性（内存高效）的同时实现块间并行，训练时间减少最多35.2%，推理时间降至18.15%。

**[Partially Shared Concept Bottleneck Models](human_understanding/partially_shared_concept_bottleneck_models.md)**

:   提出PS-CBM框架，通过多模态概念生成（结合LLM语义与示例图像视觉线索）、部分共享概念策略（基于激活模式合并概念）和Concept-Efficient Accuracy（CEA）评估指标，在11个数据集上以更少的概念实现了更高的分类精度和可解释性。

**[Pathmind A Retrieve-Prioritize-Reason Framework For Knowledge Graph Reasoning Wi](human_understanding/pathmind_a_retrieve-prioritize-reason_framework_for_knowledge_graph_reasoning_wi.md)**

:   提出PathMind框架，遵循"检索-优先级排序-推理"范式，通过语义感知的路径优先级函数（结合累积代价和估计未来代价）识别重要推理路径，再通过任务特定指令调优和路径级偏好对齐两阶段训练增强LLM的忠实可解释推理，在复杂推理任务上以更少token实现SOTA。

**[Personality-Guided Public-Private Domain Disentangled Hypergraph-Former Network ](human_understanding/personality-guided_public-private_domain_disentangled_hypergraph-former_network_.md)**

:   提出 P3HF 框架，通过人格引导的特征门控、时序感知的超图-Transformer（Hypergraph-Former）架构和事件级公私域解耦三大创新，在多事件多模态抑郁检测任务上实现约 10% 的准确率和 F1 提升。

**[Plug-And-Play Clarifier A Zero-Shot Multimodal Framework For Egocentric Intent D](human_understanding/plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)**

:   提出 Plug-and-Play Clarifier，一个零样本、模块化的多模态框架，将第一人称视角中的意图歧义问题分解为文本澄清、视觉质量评估和跨模态手势定位三个子任务，使 4-8B 小模型在意图消歧任务上提升约 30%，接近甚至超越大模型水平。

**[Radar-Aplanc Unsupervised Radar-Based Heartbeat Sensing Via Augmented Pseudo-Lab](human_understanding/radar-aplanc_unsupervised_radar-based_heartbeat_sensing_via_augmented_pseudo-lab.md)**

:   提出首个雷达心跳感知的无监督学习框架 Radar-APLANC，通过噪声对比三元组损失（NCT loss）和增强伪标签生成器实现两阶段无监督训练，无需昂贵的生理信号标注即可达到接近监督方法的性能。

**[Ragfort Dual-Path Defense Against Proprietary Knowledge Base Extraction In Retri](human_understanding/ragfort_dual-path_defense_against_proprietary_knowledge_base_extraction_in_retri.md)**

:   提出 RAGFort，首个系统性防御 RAG 知识库抽取攻击的双路径框架，通过对比重索引（inter-class）隔离主题间边界和约束级联生成（intra-class）抑制敏感内容输出，在安全性上将知识恢复率降低至无保护的 0.51×，同时保持回答质量。

**[Reina Regularized Entropy Information-Based Loss For Efficient Simultaneous Spee](human_understanding/reina_regularized_entropy_information-based_loss_for_efficient_simultaneous_spee.md)**

:   提出 REINA（Regularized Entropy INformation Adaptation）损失函数，基于互信息理论高效地将非流式语音翻译模型转换为流式同声传译模型，在多语言方向上达到 SOTA 流式翻译性能，并提出新的流式效率评估指标 NoSE。

**[Remember Me Bridging The Long-Range Gap In Lvlms With Three-Step Inference-Only ](human_understanding/remember_me_bridging_the_long-range_gap_in_lvlms_with_three-step_inference-only_.md)**

:   提出 T-DRS（Three-step Decay Resilience Strategies），一个无需训练的推理时框架，通过语义驱动增强、距离感知控制和远距离重强化三个阶段协同缓解 RoPE 引起的长程注意力衰减，在 VQA 任务上持续提升多个 LVLM 的性能。

**[Renew Risk- And Energy-Aware Navigation In Dynamic Waterways](human_understanding/renew_risk-_and_energy-aware_navigation_in_dynamic_waterways.md)**

:   提出 RENEW 全局路径规划器，为水面自主航行器 (ASV) 在动态水流 (洋流) 环境中引入统一的风险感知和能量感知策略，通过自适应不可导航区域识别、最佳努力应急策略和基于约束 Delaunay 三角化的分层架构实现安全高效导航，应急碰撞测试中实现零碰撞。

**[Renormalization Group Guided Tensor Network Structure Search](human_understanding/renormalization_group_guided_tensor_network_structure_search.md)**

:   提出 RGTN 框架，将统计物理中的重正化群（Renormalization Group）理论引入张量网络结构搜索，通过多尺度粗粒化-扩展-压缩流程和可学习边门控实现连续拓扑演化，在光场压缩、高阶张量分解和视频补全任务上达到 SOTA 压缩率，同时比已有方法快 4–600 倍。

**[Self-Correction Distillation For Structured Data Question Answering](human_understanding/self-correction_distillation_for_structured_data_question_answering.md)**

:   提出自纠正蒸馏（SCD）方法，通过错误提示机制（EPM）和两阶段蒸馏策略，将大规模LLM（GPT4）的结构化数据问答能力高效迁移到小规模LLM（8B），在5个基准上取得最优蒸馏性能。

**[Small Language Models For Efficient Agentic Tool Calling Outperforming Large Mod](human_understanding/small_language_models_for_efficient_agentic_tool_calling_outperforming_large_mod.md)**

:   通过对OPT-350M模型进行单epoch的SFT微调，在ToolBench评估中取得77.55%的通过率，大幅超越ChatGPT-CoT（26%）、ToolLLaMA-DFS（30.18%）等大模型基线，证明了针对性微调的小模型在特定任务上可显著超越通用大模型。

**[Soscontrol Enhancing Human Motion Generation Through Saliency-Aware Symbolic Ori](human_understanding/soscontrol_enhancing_human_motion_generation_through_saliency-aware_symbolic_ori.md)**

:   提出Salient Orientation Symbolic (SOS) script——基于Labanotation启发的可编程符号化运动表示框架，通过时序约束的凝聚聚类提取关键帧显著性，结合SMS数据增强和梯度优化的SOSControl框架实现对身体部位朝向和运动时序的精确控制，在HumanML3D上SOS-Acc达0.988且FID仅3.892。

**[Spatiotemporal-Untrammelled Mixture Of Experts For Multi-Person Motion Predictio](human_understanding/spatiotemporal-untrammelled_mixture_of_experts_for_multi-person_motion_predictio.md)**

:   提出ST-MoE框架，首次将混合专家模型（MoE）与双向时空Mamba相结合用于多人运动预测，通过四种异构时空专家灵活捕获复杂时空依赖，实现SOTA精度的同时减少41.38%参数量，训练加速3.6倍。

**[Streaming Generation Of Co-Speech Gestures Via Accelerated Rolling Diffusion](human_understanding/streaming_generation_of_co-speech_gestures_via_accelerated_rolling_diffusion.md)**

:   提出基于 Rolling Diffusion 的流式共语手势生成框架，通过结构化渐进噪声调度将任意扩散模型转化为流式手势生成器，并引入 Rolling Diffusion Ladder Acceleration (RDLA) 实现最高 4× 加速（200 FPS），在 ZEGGS 和 BEAT 基准上全面超越基线。

**[Talksketch Multimodal Generative Ai For Real-Time Sketch Ideation With Speech](human_understanding/talksketch_multimodal_generative_ai_for_real-time_sketch_ideation_with_speech.md)**

:   提出TalkSketch系统，将手绘草图与实时语音输入相结合，嵌入多模态AI聊天机器人，使设计师在早期构思阶段能够边画边说、流畅地与AI协作，解决了现有GenAI工具中文字提示打断创作流程的问题。

**[Theory Of Mind For Explainable Human-Robot Interaction](human_understanding/theory_of_mind_for_explainable_human-robot_interaction.md)**

:   提出将心智理论（ToM）视为可解释AI（XAI）的一种形式，使用VXAI框架的七个评价标准系统评估现有HRI中的ToM研究，发现关键缺陷（特别是忠实度缺失），并主张将ToM整合到XAI框架中以实现用户导向的解释。

**[Towards A Common Framework For Autoformalization](human_understanding/towards_a_common_framework_for_autoformalization.md)**

:   本文系统回顾了"自动形式化"（autoformalization）在数学证明、逻辑推理、规划和知识表示等领域的现有工作，提出了一个统一的跨学科定义框架，将自动形式化定义为从非形式语言到形式推理语言的语义等价转换，旨在促进不同研究社区间的方法共享并加速下一代 AI 推理系统的发展。

**[Transferable Backdoor Attacks For Code Models Via Sharpness-Aware Adversarial Pe](human_understanding/transferable_backdoor_attacks_for_code_models_via_sharpness-aware_adversarial_pe.md)**

:   提出 STAB（Sharpness-aware Transferable Adversarial Backdoor），通过 SAM 训练代理模型使其收敛到损失平面的平坦区域，并使用 Gumbel-Softmax 优化生成上下文感知的对抗触发器，首次实现了同时兼顾跨数据集迁移性和隐蔽性的代码模型后门攻击。

**[Unifit Towards Universal Virtual Try-On With Mllm-Guided Semantic Alignment](human_understanding/unifit_towards_universal_virtual_try-on_with_mllm-guided_semantic_alignment.md)**

:   提出 UniFit，一个由多模态大语言模型（MLLM）驱动的通用虚拟试穿框架，通过 MLLM 引导的语义对齐模块（MGSA）桥接文本指令与参考图像之间的语义鸿沟，并通过两阶段渐进训练+自合成流水线克服复杂场景的数据稀缺问题，首次在单一框架内支持 6 种 VTON 任务。

**[Vpho Joint Visual-Physical Cue Learning And Aggregation For Hand-Object Pose Est](human_understanding/vpho_joint_visual-physical_cue_learning_and_aggregation_for_hand-object_pose_est.md)**

:   提出 VPHO，一个联合视觉和物理线索的手-物体姿态估计框架，通过力预测模块学习 3D 物理线索，并设计两阶段候选姿态聚合策略（视觉引导 + 物理引导），在保持视觉一致性的同时实现物理合理性，在 DexYCB 和 HO3D 两个基准上同时达到姿态精度和物理合理性的 SOTA。

**[W2S-Aligntree Weak-To-Strong Inference-Time Alignment For Large Language Models ](human_understanding/w2s-aligntree_weak-to-strong_inference-time_alignment_for_large_language_models_.md)**

:   提出 W2S-AlignTree，首个将蒙特卡洛树搜索（MCTS）与弱到强泛化（W2SG）范式结合的推理时对齐框架，利用弱模型的步级代理值函数实时引导强模型生成，在情感控制、摘要、指令遵循任务上均显著超越基线，其中 Llama3-8B 摘要任务提升 15.9%。

**[Why Do Open-Source Llms Struggle With Data Analysis A Systematic Empirical Study](human_understanding/why_do_open-source_llms_struggle_with_data_analysis_a_systematic_empirical_study.md)**

:   系统研究了开源 LLM 在数据分析任务中的能力瓶颈，将数据分析分解为数据理解、代码生成和战略规划三个维度，发现**战略规划是决定性因素**而非编码或数据理解；并提出了一种策略引导的数据合成方法，使微调后的 7B/14B 模型达到与 GPT-4o 竞争的性能。

**[Xlinear A Lightweight And Accurate Mlp-Based Model For Long-Term Time Series For](human_understanding/xlinear_a_lightweight_and_accurate_mlp-based_model_for_long-term_time_series_for.md)**

:   提出 XLinear，一个基于 MLP + sigmoid gating 的轻量时间序列预测模型，通过 global token 机制高效融合 endogenous 与 exogenous 变量信息，在 12 个数据集上实现精度与效率的最优平衡。

**[Yes Florence I Will Do Better Next Time Agentic Feedback Reasoning For Humorous ](human_understanding/yes_florence_i_will_do_better_next_time_agentic_feedback_reasoning_for_humorous_.md)**

:   提出 FLoReNce 框架，将幽默 meme 理解建模为闭环控制系统，通过 Judge 反馈+PID 控制器+非参数知识库的闭环学习，在推理时通过检索相似经验调制 prompt，使冻结的 VLM 实现自适应推理，无需微调即可显著提升预测和解释质量。

---

## 🛡️ AI安全 { #ai_safety }

**[Alternative Fairness And Accuracy Optimization In Criminal J](ai_safety/alternative_fairness_and_accuracy_optimization_in_criminal_j.md)**

:   本文系统综述了算法公平性的三大维度（群体公平、个体公平、过程公平），提出了一种基于容差约束的改进群体公平性优化公式，并构建了面向公共决策系统的"公平三支柱"部署框架。

**[An Improved Privacy And Utility Analysis Of Differentially P](ai_safety/an_improved_privacy_and_utility_analysis_of_differentially_p.md)**

:   在仅假设损失函数L-光滑（不需要凸性）的条件下，为DPSGD推导出了更紧的闭式RDP隐私界，并首次在有界域场景下给出了完整的收敛性/效用分析，揭示了较小的参数域直径可以同时改善隐私和效用。

**[An Information Theoretic Evaluation Metric For Strong Unlear](ai_safety/an_information_theoretic_evaluation_metric_for_strong_unlear.md)**

:   提出 Information Difference Index (IDI)，一种基于信息论的白盒评估指标，通过度量中间层特征与遗忘标签之间的互信息来衡量机器遗忘的彻底程度，揭示了现有黑盒指标（MIA、JSD等）无法捕捉的中间层残留信息问题，并提出 COLA 方法在特征层面消除残余信息。

**[An Llm-Based Simulation Framework For Embodied Conversationa](ai_safety/an_llm-based_simulation_framework_for_embodied_conversationa.md)**

:   提出 ECAs 框架，基于认知行为治疗(CBT)等心理学理论，利用 LLM 将真实咨询案例扩展为具身认知记忆空间，模拟心理咨询中来访者的完整认知过程，生成高保真度的咨询对话数据，在专家评估和自动评估中均显著优于基线。

**[Angular Gradient Sign Method Uncovering Vulnerabilities In H](ai_safety/angular_gradient_sign_method_uncovering_vulnerabilities_in_h.md)**

:   提出Angular Gradient Sign Method (AGSM)，将双曲空间中的梯度分解为径向（层次深度）和角度（语义）分量，仅沿角度方向施加扰动来生成对抗样本，在图像分类和跨模态检索任务上比标准FGSM/PGD多降低5-13%的准确率。

**[Argumentative Debates For Transparent Bias Detection Technic](ai_safety/argumentative_debates_for_transparent_bias_detection_technic.md)**

:   提出 ABIDE（Argumentative BIas Detection by DEbate），通过基于邻域属性的论证方案（argument schemes）构建量化双极论证框架（QBAF），将偏见检测过程建模为结构化辩论，实现从单邻域到全局的透明偏见推理，并形式化证明 QBAF 语义与偏见检测期望行为之间的对应关系。

**[Auvic Adversarial Unlearning Of Visual Concepts For Multi-Mo](ai_safety/auvic_adversarial_unlearning_of_visual_concepts_for_multi-mo.md)**

:   提出AUVIC框架，通过对抗性扰动生成器 + 动态锚点保留机制，在MLLM中精确遗忘目标视觉概念（如特定人脸），同时避免对语义相似概念的附带遗忘，并构建了首个面向群体场景视觉概念遗忘的评测基准VCUBench。

**[Beyond Superficial Forgetting Thorough Unlearning Through Knowledge Density Esti](ai_safety/beyond_superficial_forgetting_thorough_unlearning_through_knowledge_density_esti.md)**

:   提出 KUnBR 框架，通过梯度引导的知识密度估计定位有害知识富集层，并采用块重插入策略绕过 cover layer 的梯度遮蔽效应，实现对 LLM 有害知识的深度遗忘而非表面抑制。

**[Breaking The Adversarial Robustness-Performance Trade-Off In Text Classification](ai_safety/breaking_the_adversarial_robustness-performance_trade-off_in_text_classification.md)**

:   提出 Manifold-Correcting Causal Flow (MC²F) 框架，通过分层黎曼连续正则化流 (SR-CNF) 学习干净数据嵌入的流形密度进行对抗样本检测，再用测地线净化求解器 (Geodesic Purification Solver) 将被检测为对抗的嵌入沿最短路径投影回干净流形，在 SST-2/AGNews/YELP 三个数据集上对抗鲁棒性全面超越 SOTA，同时完全不损失（甚至略微提升）干净数据精度。

**[Breaking The Dyadic Barrier Rethinking Fairness In Link Prediction Beyond Demogr](ai_safety/breaking_the_dyadic_barrier_rethinking_fairness_in_link_prediction_beyond_demogr.md)**

:   本文揭示了链接预测中二元公平性（dyadic fairness）和 Demographic Parity（ΔDP）的三大根本缺陷——GNN 表达力不足、子群偏差被掩盖、对排序不敏感——并提出基于 NDKL 的排序感知公平度量和后处理算法 MORAL，在六个数据集上实现了 SOTA 的公平性-效用权衡。

**[Can Editing Llms Inject Harm](ai_safety/can_editing_llms_inject_harm.md)**

:   本文将知识编辑技术重新定义为一种新型 LLM 安全威胁（Editing Attack），系统性地研究了通过 ROME、FT、ICE 三种编辑方法向 LLM 注入虚假信息和偏见的可行性，发现其效果显著且极具隐蔽性。

**[Core-Fed Bridging Collaborative And Representation Fairness Via Federated Embedd](ai_safety/core-fed_bridging_collaborative_and_representation_fairness_via_federated_embedd.md)**

:   提出 CoRe-Fed 框架，通过嵌入级对比对齐与贡献感知聚合两个协同模块，同时解决联邦学习中的表示公平性和协作公平性问题，在异构数据分布下显著提升全局模型的公平性与泛化能力。

**[Deeptracer Tracing Stolen Model Via Deep Coupled Watermarks](ai_safety/deeptracer_tracing_stolen_model_via_deep_coupled_watermarks.md)**

:   提出DeepTracer鲁棒水印框架，通过自适应源类选择（K-Means聚类覆盖特征空间）+ 同类耦合损失（拉近水印样本与目标类在输出空间的距离）+ 两阶段关键样本过滤，使水印任务与主任务深度耦合，在6种模型窃取攻击（含hard-label和data-free）下水印成功率平均达77-100%，远超现有方法。

**[Democratizing Llm Efficiency From Hyperscale Optimizations To Universal Deployab](ai_safety/democratizing_llm_efficiency_from_hyperscale_optimizations_to_universal_deployab.md)**

:   本文是一篇立场论文（position paper），指出当前 LLM 效率研究被超大规模假设所主导，提出面向中小规模部署者的五大开放研究挑战，并倡导以开销感知效率（OAE）重新定义效率指标。

**[Detect All-Type Deepfake Audio Wavelet Prompt Tuning For Enhanced Auditory Perce](ai_safety/detect_all-type_deepfake_audio_wavelet_prompt_tuning_for_enhanced_auditory_perce.md)**

:   首次建立全类型（语音/声音/歌声/音乐）音频深伪检测基准，提出小波提示调优（WPT）方法通过离散小波变换增强 SSL 特征的全频域感知能力，在不增加训练参数的前提下超越全量微调，co-training 后平均 EER 仅 3.58%。

**[Diversifying Counterattacks Orthogonal Exploration For Robust Clip Inference](ai_safety/diversifying_counterattacks_orthogonal_exploration_for_robust_clip_inference.md)**

:   提出方向正交反攻击（DOC）方法，通过在反攻击优化中引入正交梯度分量和动量更新扩展搜索空间，结合基于余弦相似度的方向敏感度评分自适应调控反攻击强度，在 16 个数据集上显著提升 CLIP 的测试时对抗鲁棒性。

**[Easy To Learn Yet Hard To Forget Towards Robust Unlearning Under Bias](ai_safety/easy_to_learn_yet_hard_to_forget_towards_robust_unlearning_under_bias.md)**

:   提出 CUPID 框架，通过损失景观的锐度分析将遗忘集划分为因果/偏差子集，并识别和分离模型中的因果/偏差通路，实现对有偏模型的精准类别遗忘，有效解决"捷径遗忘"问题。

**[Efx And Po Allocation Exists For Two Types Of Goods](ai_safety/efx_and_po_allocation_exists_for_two_types_of_goods.md)**

:   证明了当物品只有两种类型且所有估值为正时，满足 EFX（任意物品无嫉妒）和 Pareto 最优的分配总是存在的，并给出了准线性时间算法。

**[Enhancing Dpsgd Via Per-Sample Momentum And Low-Pass Filtering](ai_safety/enhancing_dpsgd_via_per-sample_momentum_and_low-pass_filtering.md)**

:   提出 DP-PMLF，通过逐样本动量（per-sample momentum）降低裁剪偏差，同时利用低通滤波器（low-pass filter）抑制高频 DP 噪声，首次同时从两个方向缓解 DPSGD 的精度退化问题。

**[Fair Model-Based Clustering](ai_safety/fair_model-based_clustering.md)**

:   提出基于有限混合模型的公平聚类算法 FMC，通过在模型参数（而非样本级赋值）上施加公平性约束，实现参数量与样本量无关的可扩展公平聚类，支持小批量学习和分类数据，在大规模数据集上显著优于现有方法。

**[Fairgse Fairness-Aware Graph Neural Network Without High False Positive Rates](ai_safety/fairgse_fairness-aware_graph_neural_network_without_high_false_positive_rates.md)**

:   首次揭示公平感知 GNN 中的"FPR 捷径"问题——现有方法通过大量误判负样本为正来达到公平指标，提出 FairGSE 框架通过最大化二维结构熵重新加权图边来同时改善公平性并降低假阳性率，FPR 降低 39%。

**[Fedalt Federated Fine-Tuning Through Adaptive Local Training With Rest-Of-World ](ai_safety/fedalt_federated_fine-tuning_through_adaptive_local_training_with_rest-of-world_.md)**

:   提出 FedALT，通过为每个客户端维护独立的 Individual LoRA（本地训练更新）和冻结的 Rest-of-World (RoW) LoRA（其他客户端平均），配合自适应 MoE 混合器动态平衡本地知识与全局知识，彻底避免 FedAvg 聚合导致的跨客户端干扰，在异构任务联邦 LLM 微调上显著优于 SOTA。

**[From Single To Societal Analyzing Persona-Induced Bias In Multi-Agent Interactio](ai_safety/from_single_to_societal_analyzing_persona-induced_bias_in_multi-agent_interactio.md)**

:   本文首次系统研究了 LLM 多智能体交互中的人格诱导偏见，通过在协作问题解决和说服任务中的受控实验，揭示了三个关键发现：(1) 不同人格在可信度和坚持度上存在显著偏差（优势群体如男性和白人被视为更不可信）；(2) 智能体表现出显著的内群体偏好；(3) 这些偏见在多轮、多智能体场景中持续存在且有放大趋势。

**[Gender Bias In Emotion Recognition By Large Language Models](ai_safety/gender_bias_in_emotion_recognition_by_large_language_models.md)**

:   系统性地评估了多个 LLM（GPT-4/5、Mistral、LLaMA 等）在情感识别任务中的性别偏见，发现大多数模型对至少一个情感标签存在显著性别偏见，并通过实验证明推理时 prompt 策略（提示工程、上下文学习、CoT）无法有效去偏，而基于训练的微调方法可以有效缓解偏见。

**[Generalizing Fair Clustering To Multiple Groups Algorithms And Applications](ai_safety/generalizing_fair_clustering_to_multiple_groups_algorithms_and_applications.md)**

:   将最近公平聚类（Closest Fair Clustering）问题从仅两个群体推广到任意多群体，证明三群体以上等比例情形已为NP-hard，提出近线性时间近似算法（等比例 $O(|\chi|^{1.6}\log^{2.81}|\chi|)$、任意比例 $O(|\chi|^{3.81})$），并将结果推广至公平相关聚类和公平共识聚类问题。

**[Ghost In The Transformer Detecting Model Reuse With Invariant Spectral Signature](ai_safety/ghost_in_the_transformer_detecting_model_reuse_with_invariant_spectral_signature.md)**

:   提出 GhostSpec，一种无需数据、不修改模型行为的白盒方法，通过对注意力权重矩阵的不变乘积做 SVD 提取光谱指纹，可在微调、剪枝、合并、扩展甚至对抗性变换下稳健地验证 LLM 血统。

**[Graphtextack A Realistic Black-Box Node Injection Attack On Llm-Enhanced Gnns](ai_safety/graphtextack_a_realistic_black-box_node_injection_attack_on_llm-enhanced_gnns.md)**

:   提出 GraphTextack——首个针对 LLM 增强 GNN 的黑盒多模态节点注入投毒攻击，通过进化优化框架联合优化注入节点的图结构连接和语义特征，不依赖模型内部信息或代理模型，在5个数据集和2类LLM-GNN模型上显著优于12种基线方法。

**[Hashed Watermark As A Filter Defeating Forging And Overwriting Attacks In Weight](ai_safety/hashed_watermark_as_a_filter_defeating_forging_and_overwriting_attacks_in_weight.md)**

:   提出 NeuralMark——一种基于哈希水印过滤器的权重水印方法，利用哈希函数从秘钥生成不可逆二值水印作为私有过滤器选择嵌入参数，借助雪崩效应阻断伪造攻击的梯度反推，通过多轮过滤减少参数重叠抵御覆写攻击，在13种CNN/Transformer架构、5个图像分类和1个文本生成任务上验证了有效性和鲁棒性。

**[Healsplit Towards Self-Healing Through Adversarial Distillation In Split Federat](ai_safety/healsplit_towards_self-healing_through_adversarial_distillation_in_split_federat.md)**

:   提出 HealSplit，首个针对分割联邦学习（SFL）的统一防御框架，通过拓扑感知检测（TAS）识别中毒样本、GAN 生成语义一致的替代表示、对抗多教师蒸馏训练一致性验证学生模型，实现端到端检测与恢复，在五类投毒攻击下均大幅超越十种 SOTA 防御方法。

**[Infodecom Decomposing Information For Defending Against Privacy Leakage In Split](ai_safety/infodecom_decomposing_information_for_defending_against_privacy_leakage_in_split.md)**

:   提出 InfoDecom，通过两级信息分解（频域视觉信息去除 + 互信息抑制）减少 smashed data 中的冗余信息，再添加闭式计算的高斯噪声提供理论隐私保证，在浅层客户端模型下实现远优于现有方法的 utility-privacy trade-off。

**[Lamp Learning Universal Adversarial Perturbations For Multi-Image Tasks Via Pre-](ai_safety/lamp_learning_universal_adversarial_perturbations_for_multi-image_tasks_via_pre-.md)**

:   提出 LAMP，一种针对多图 MLLM 的 black-box Universal Adversarial Perturbation 学习方法，通过 attention 约束和"传染式"损失实现仅扰动少量图像即可跨模型/任务迁移攻击。

**[Learning To Collaborate An Orchestrated-Decentralized Framework For Peer-To-Peer](ai_safety/learning_to_collaborate_an_orchestrated-decentralized_framework_for_peer-to-peer.md)**

:   提出 KNEXA-FL 框架，通过一个不接触模型的中央配对器（CPM）将 P2P 协作建模为上下文 Bandit 问题，使用 LinUCB 学习最优配对策略，在异构 LLM 联邦学习中实现比随机 P2P 高约 50% 的 Pass@1 提升，且避免了中心化蒸馏的灾难性崩溃。

**[Matrix-Free Two-To-Infinity And One-To-Two Norms Estimation](ai_safety/matrix-free_two-to-infinity_and_one-to-two_norms_estimation.md)**

:   提出 TwINEst 和 TwINEst++ 两种基于 Hutchinson 对角估计器的随机算法，用于在无矩阵 (matrix-free) 设定下高效估计 $\|A\|_{2\to\infty}$ 和 $\|A\|_{1\to 2}$ 范数，并提供了 oracle 复杂度理论保证，在 DNN 的 Jacobian 正则化（图像分类对抗鲁棒性）和推荐系统对抗攻击防御中展现了显著优势。

**[Mpd-Sgr Robust Spiking Neural Networks With Membrane Potential Distribution-Driv](ai_safety/mpd-sgr_robust_spiking_neural_networks_with_membrane_potential_distribution-driv.md)**

:   从理论上建立了 SNN 鲁棒性误差与代理梯度（SG）幅值之间的联系，揭示减少膜电位分布（MPD）与 SG 梯度可用区间的重叠比例可有效降低对抗扰动敏感度，据此提出 MPD-SGR 正则化方法，在 vanilla training 和 adversarial training 设置下均大幅超越现有 SNN 防御方法。

**[Perturb Your Data Paraphrase-Guided Training Data Watermarking](ai_safety/perturb_your_data_paraphrase-guided_training_data_watermarking.md)**

:   提出SPECTRA——一种基于paraphrase采样的训练数据水印方法，通过LLM生成改写文本并利用Min-K%++评分选择与原文分数接近的paraphrase作为水印，在数据仅占训练语料0.001%的情况下，member与non-member的p-value差距稳定超过9个数量级。

**[Plug-And-Play Parameter-Efficient Tuning Of Embeddings For Federated Recommendat](ai_safety/plug-and-play_parameter-efficient_tuning_of_embeddings_for_federated_recommendat.md)**

:   提出一个即插即用的联邦推荐框架，通过将 PEFT（Parameter-Efficient Fine-Tuning）理念引入物品嵌入，冻结预训练的全量嵌入并仅传输轻量级压缩嵌入（LoRA / Hash / RQ-VAE），大幅降低通信开销的同时提升推荐精度。

**[Principles2Plan Llm-Guided System For Operationalising Ethical Principles Into P](ai_safety/principles2plan_llm-guided_system_for_operationalising_ethical_principles_into_p.md)**

:   提出 Principles2Plan，一个交互式原型系统，通过人类与 LLM 协作将高层伦理原则（如仁善、隐私）转化为上下文相关的伦理规则，并嵌入 PDDL 规划器生成符合伦理的行动计划。

**[Prism Privacy-Aware Routing For Adaptive Cloud-Edge Llm Inference Via Semantic S](ai_safety/prism_privacy-aware_routing_for_adaptive_cloud-edge_llm_inference_via_semantic_s.md)**

:   提出 PRISM 框架，通过上下文感知的软门控路由机制将用户 prompt 动态分配到云端/边缘/协作三种推理模式，并在协作模式中使用自适应两层本地差分隐私（LDP）和语义草图协作，实现隐私-效用-效率的三方平衡。

**[Privacy-Protected Retrieval-Augmented Generation For Knowledge Graph Question An](ai_safety/privacy-protected_retrieval-augmented_generation_for_knowledge_graph_question_an.md)**

:   首次探索知识图谱问答（KGQA）中的隐私保护 RAG 场景，提出 ARoG（Abstraction Reasoning on Graph）框架，通过关系中心抽象和结构导向抽象两种策略，在实体被匿名化（替换为无意义的 MID）的条件下，仍能有效检索和利用知识图谱回答问题。

**[Privacy Auditing Of Multi-Domain Graph Pre-Trained Model Under Membership Infere](ai_safety/privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)**

:   提出 MGP-MIA 框架，首次针对多域图预训练模型开展成员推理攻击（MIA），通过机器遗忘放大成员信号、增量学习构建影子模型、基于相似度的推理机制，有效揭示多域图预训练的隐私泄漏风险。

**[Privacy On The Fly A Predictive Adversarial Transformation Network For Mobile Se](ai_safety/privacy_on_the_fly_a_predictive_adversarial_transformation_network_for_mobile_se.md)**

:   提出 PATN（Predictive Adversarial Transformation Network），首个将对抗扰动引入传感器数据隐私保护的框架，利用历史传感器数据生成面向未来的对抗扰动，实现零延迟的实时隐私保护，同时保持传感器数据的语义保真度。

**[Problog4Fairness A Neurosymbolic Approach To Modeling And Mitigating Bias](ai_safety/problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)**

:   提出 ProbLog4Fairness 框架，利用概率逻辑编程语言 ProbLog 将数据中的偏差机制形式化为可解释的逻辑程序，并通过 DeepProbLog 的远程监督将偏差假设集成到神经网络训练中，实现灵活、原则性的偏差缓解。

**[Psm Prompt Sensitivity Minimization Via Llm-Guided Black-Box Optimization](ai_safety/psm_prompt_sensitivity_minimization_via_llm-guided_black-box_optimization.md)**

:   提出 PSM 框架，将系统提示防护形式化为效用约束下的黑盒优化问题，利用 LLM-as-Optimizer 自动搜索最优"盾牌"后缀，在不降低模型功能的前提下将提示泄漏攻击成功率降至接近零。

**[Reference Recommendation Based Membership Inference Attack Against Hybrid-Based ](ai_safety/reference_recommendation_based_membership_inference_attack_against_hybrid-based_.md)**

:   提出基于参考推荐的成员推理攻击（MIA），设计相对成员度量 $\rho(u) = d(v_t, v_h) / d(v_t, v_r)$，利用混合推荐系统的个性化特性获取参考推荐，首次有效攻击混合推荐系统，攻击成功率高达 93.4% 且计算成本仅需 10 秒。

**[Regionmarker A Region-Triggered Semantic Watermarking Framework For Embedding-As](ai_safety/regionmarker_a_region-triggered_semantic_watermarking_framework_for_embedding-as.md)**

:   提出基于语义区域触发的水印框架 RegionMarker，在低维空间中定义触发区域并注入语义水印，是首个能同时抵御 CSE 攻击、改写攻击和维度扰动攻击的 EaaS 版权保护方法。

**[Rethinking Target Label Conditioning In Adversarial Attacks A 2D Tensor-Guided G](ai_safety/rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)**

:   提出 TGAF 框架，利用扩散模型将目标标签编码为 2D 语义张量来引导对抗噪声生成，并设计随机遮挡策略保留完整语义信息，显著提升目标对抗攻击的可迁移性。

**[Revisiting Unfairness In Recourse By Minimizing Worst-Case Social Burden](ai_safety/revisiting_unfairness_in_recourse_by_minimizing_worst-case_social_burden.md)**

:   系统分析了算法追索 (algorithmic recourse) 中公平性度量的三大局限（忽视分类器决策行为、忽略真实标签、差距指标掩盖不公平），提出基于社会负担 (social burden) 的公平性框架 MISOB，通过极小化极大加权训练策略减少所有群体的社会负担，无需访问敏感属性即可在预测和追索阶段同时提升公平性。

**[Robust Watermarking On Gradient Boosting Decision Trees](ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)**

:   提出首个针对 GBDT 模型的鲁棒水印框架，通过 in-place 微调嵌入水印，设计了四种嵌入策略（Wrong Prediction Flip、Outlier Flip、Cluster Center Flip、Confidence Flip），实现高嵌入成功率、低精度损失和强抗微调鲁棒性。

**[Secmoe Communication-Efficient Secure Moe Inference Via Select-Then-Compute](ai_safety/secmoe_communication-efficient_secure_moe_inference_via_select-then-compute.md)**

:   提出 SecMoE 框架，通过 Select-Then-Compute 范式在两方安全计算中高效实现稀疏 MoE 推理，避免冗余专家计算，通信量降低最高 29.8 倍，端到端加速最高 16.1 倍。

**[Sim-To-Real An Unsupervised Noise Layer For Screen-Camera Watermarking Robustnes](ai_safety/sim-to-real_an_unsupervised_noise_layer_for_screen-camera_watermarking_robustnes.md)**

:   提出 Simulation-to-Real (S2R) 框架，首创"数学建模 → 无监督域迁移"两阶段噪声近似策略：先用数学模型将清晰图像变换到已知噪声域 $\mathcal{C}$，再用无监督 Image-to-Image 网络 $G$ 将 $\mathcal{C}$ 映射到真实屏幕-相机噪声域 $\mathcal{U}$，无需配对数据即可精确逼近真实 SC 噪声，在多设备、多角度、多距离条件下均取得最优水印鲁棒性（BER 降低 30-60%）和图像质量（PSNR 42.27 dB / SSIM 0.962）。

**[Sproutbench A Benchmark For Safe And Ethical Large Language Models For Youth](ai_safety/sproutbench_a_benchmark_for_safe_and_ethical_large_language_models_for_youth.md)**

:   提出 SproutBench，一个包含 1,283 个发展心理学驱动的对抗性提示的评估基准，系统评估 47 个 LLM 在儿童和青少年（0-6、7-12、13-18 岁）场景下的安全性，发现安全性与风险预防强相关（$\rho = 0.86$），交互性与年龄适配性存在显著权衡（$\rho = -0.48$）。

**[Stylebreak Revealing Alignment Vulnerabilities In Large Audio-Language Models Vi](ai_safety/stylebreak_revealing_alignment_vulnerabilities_in_large_audio-language_models_vi.md)**

:   提出 StyleBreak，首个基于语音风格的音频越狱框架，通过两阶段风格感知变换管道和查询自适应策略网络，系统研究语言学、副语言学和超语言学属性对 LAM 对齐鲁棒性的影响，在多种攻击范式下将 ASR 提升 7.1%-22.3%。

**[The Confidence Trap Gender Bias And Predictive Certainty In Llms](ai_safety/the_confidence_trap_gender_bias_and_predictive_certainty_in_llms.md)**

:   提出Gender-ECE指标，系统评估六种开源LLM在性别代词预测任务中的置信度校准与人类偏见对齐程度，发现Gemma-2模型校准最差且存在极端的男女代词校准差异，而训练数据过滤较少的GPT-J-6B反而校准最好。

**[Toporeformer Mitigating Adversarial Attacks Using Topological Purification In Oc](ai_safety/toporeformer_mitigating_adversarial_attacks_using_topological_purification_in_oc.md)**

:   提出 TopoReformer，一种基于拓扑自编码器的模型无关对抗纯化管线，利用持久同调（persistent homology）在潜空间中强制拓扑一致性，无需对抗训练即可过滤对抗扰动，有效保护 OCR 系统免受经典攻击、自适应攻击和 OCR 专用水印攻击。

**[Towards Multiple Missing Values-Resistant Unsupervised Graph Anomaly Detection](ai_safety/towards_multiple_missing_values-resistant_unsupervised_graph_anomaly_detection.md)**

:   提出 M2V-UGAD 框架，首次解决节点属性和图拓扑同时缺失下的无监督图异常检测问题，通过双通路独立填补、超球潜空间融合和伪异常生成三个核心机制，克服跨视图干扰和填补偏差，在7个基准数据集上一致超越现有方法。

**[Transferable Hypergraph Attack Via Injecting Nodes Into Pivotal Hyperedges](ai_safety/transferable_hypergraph_attack_via_injecting_nodes_into_pivotal_hyperedges.md)**

:   提出 TH-Attack，一种面向超图神经网络（HGNNs）的可迁移节点注入攻击框架，通过识别信息聚合路径中的关键超边并注入语义反转的恶意节点，在黑盒场景下实现对多种 HGNN 架构的有效攻击，Accuracy 可从 80%+ 降至 30% 以下。

**[Truth Justice And Secrecy Cake Cutting Under Privacy Constraints](ai_safety/truth_justice_and_secrecy_cake_cutting_under_privacy_constraints.md)**

:   本文提出首个隐私保护的蛋糕切割协议 PP_CC_puv，将 Chen 等人的策略防操纵公平分配算法改造为基于秘密共享和安全多方计算（MPC）的隐私保护版本，在保持无嫉妒性、Pareto 最优和策略防操纵性的同时，确保参与者的偏好信息不被泄露。

**[Uncovering Bias Paths With Llm-Guided Causal Discovery An Active Learning And Dy](ai_safety/uncovering_bias_paths_with_llm-guided_causal_discovery_an_active_learning_and_dy.md)**

:   提出一种融合LLM语义先验与统计信号的混合因果发现框架，通过主动学习（Active Learning）和动态评分机制优先查询信息量最大的变量对，在噪声和混淆条件下有效恢复公平性关键因果路径（如 sex→education→income），显著优于传统CD方法和朴素LLM方法。

**[Watermod Modular Token-Rank Partitioning For Probability-Balanced Llm Watermarki](ai_safety/watermod_modular_token-rank_partitioning_for_probability-balanced_llm_watermarki.md)**

:   提出 WaterMod，一种基于模算术 ($\text{rank} \bmod k$) 的 LLM 文本水印方法，通过对概率排序后的词表进行模残差类划分，在零比特（$k=2$）和多比特（$k>2$）水印场景下统一实现高检测率和低质量损耗，无需外部同义词库或哈希技巧。

**[Yours Or Mine Overwriting Attacks Against Neural Audio Watermarking](ai_safety/yours_or_mine_overwriting_attacks_against_neural_audio_watermarking.md)**

:   首次系统研究神经音频水印的覆写攻击（overwriting attack），提出白盒、灰盒、黑盒三级攻击方案，在 AudioSeal、Timbre、WavMark 三种 SOTA 方法上均实现接近 100% 的攻击成功率，暴露了现有音频水印系统严重的安全缺陷。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[A Data-Driven Model Predictive Control Framework For Multi-Aircraft Tma Routing ](autonomous_driving/a_data-driven_model_predictive_control_framework_for_multi-aircraft_tma_routing_.md)**

:   提出闭环 MPC 框架用于樟宜机场 50 海里半径终端区（TMA）的多飞机无冲突路径规划与调度，集成 XGBoost 预测 TMA 边界到达时间、MILP 优化（含路径选择/速度调整/等待控制/安全间隔约束）和滚动时域仿真器，在峰值 36 架/小时拥堵场景下实现 7 倍计算加速且 Monte Carlo 鲁棒性验证中可行性远优于 Dijkstra 基线。

**[Ai-Based Traffic Modeling For Network Security And Privacy Challenges Ahead](autonomous_driving/ai-based_traffic_modeling_for_network_security_and_privacy_challenges_ahead.md)**

:   一篇面向网络安全与隐私（NetS&P）任务的 AI 流量建模综述与展望，系统梳理了异常检测、攻击分类、IoT 设备识别、网站指纹攻击等任务的 AI 方案，并深入讨论了数据质量、实际部署、可解释性和基础模型四大前沿挑战。

**[Backdoor Attacks On Open Vocabulary Object Detectors Via Multi-Modal Prompt Tuni](autonomous_driving/backdoor_attacks_on_open_vocabulary_object_detectors_via_multi-modal_prompt_tuni.md)**

:   首次研究开放词汇目标检测器（OVOD）的后门攻击，提出 TrAP（Trigger-Aware Prompt tuning），通过联合优化视觉和文本分支的 learnable prompt 与可学习触发器，在不修改模型权重的前提下注入高成功率后门。

**[Beta Distribution Learning For Reliable Roadway Crash Risk A](autonomous_driving/beta_distribution_learning_for_reliable_roadway_crash_risk_a.md)**

:   提出基于 Beta 分布学习的地理空间深度学习框架，利用多尺度卫星图像预测道路致命事故风险的完整概率分布（而非点估计），在 Recall 上提升 17-23%，并通过分布形状自然表达不确定性。

**[Bridging Day And Night Target-Class Hallucination Suppressio](autonomous_driving/bridging_day_and_night_target-class_hallucination_suppressio.md)**

:   首次系统性解决无配对日→夜图像翻译中的"目标类幻觉"问题，通过双头判别器（风格头+SAM2伪标签分割头）检测幻觉 + 类原型对比学习抑制幻觉，在BDD100K日夜域适应检测上将mAP从15.08提升到17.40（+15.5%），交通灯AP提升31.7%。

**[Catformer Causal Temporal Transformer With Dynamic Contextual Fusion For Driving](autonomous_driving/catformer_causal_temporal_transformer_with_dynamic_contextual_fusion_for_driving.md)**

:   提出 CaTFormer，通过因果时序 Transformer 显式建模驾驶员行为与环境上下文之间的因果交互，在 Brain4Cars 数据集上以 98.6% F1 达到 SOTA。

**[Cheating Stereo Matching In Full-Scale Physical Adversarial Attack Against Binoc](autonomous_driving/cheating_stereo_matching_in_full-scale_physical_adversarial_attack_against_binoc.md)**

:   提出首个针对立体匹配模型的3D全表面纹理物理对抗攻击，通过立体对齐渲染模块和区域感知的融合攻击（merging attack），使对抗车辆在深度图中与背景无缝融合，导致自动驾驶感知系统严重失效。

**[Comptrack Information Bottleneckguided Lowrank Dynamic Token Compres](autonomous_driving/comptrack_information_bottleneckguided_lowrank_dynamic_token_compres.md)**

:   提出 CompTrack 框架首次同时解决 LiDAR 点云中的双重冗余问题：SFP 通过信息熵分析滤除背景噪声解决空间冗余；IB-DTC 通过在线 SVD 估计有效秩、自适应确定压缩率将前景压缩为低秩代理 token 解决信息冗余。在 nuScenes 上 SOTA（61.04% Success），以 90 FPS 实时运行。

**[Debiased Dual-Invariant Defense For Adversarially Robust Person Re-Identificatio](autonomous_driving/debiased_dual-invariant_defense_for_adversarially_robust_person_re-identificatio.md)**

:   系统识别出行人ReID对抗防御的两大独特挑战（模型偏差和复合泛化需求），提出去偏双不变防御框架：数据平衡阶段用扩散模型重采样缓解偏差，双对抗自元防御阶段通过最远负样本扩展软化的度量对抗训练和对抗增强的自元学习实现对未见ID和未见攻击的双重泛化。

**[Decoupling Scene Perception And Ego Status A Multi-Context Fusion Approach For E](autonomous_driving/decoupling_scene_perception_and_ego_status_a_multi-context_fusion_approach_for_e.md)**

:   识别出端到端自动驾驶中ego status过度依赖的架构根源（BEV编码器中ego status的过早融合），提出AdaptiveAD双分支架构：场景驱动分支（去除ego status）和自我驱动分支独立生成决策，再通过场景感知融合模块自适应整合，配合路径注意力、BEV单向蒸馏和自回归在线建图辅助任务，在nuScenes上达到SOTA规划性能。

**[Differentiable Semantic Meta-Learning Framework For Long-Tail Motion Forecasting](autonomous_driving/differentiable_semantic_meta-learning_framework_for_long-tail_motion_forecasting.md)**

:   提出SAML框架，首次给出运动预测中"长尾性"(tailness)的可微语义定义——通过内在属性(运动学动态性、几何复杂度、时间不规则性)和交互属性(局部/全局风险)量化稀有度，经贝叶斯尾部感知器融合为连续Tail Index驱动MAML元学习适配，在nuScenes/NGSIM/HighD上取得SOTA，尤其在worst-case top 1-5%子集上大幅领先。

**[Difficulty-Aware Label-Guided Denoising For Monocular 3D Object Detection](autonomous_driving/difficulty-aware_label-guided_denoising_for_monocular_3d_object_detection.md)**

:   提出 MonoDLGD，通过根据实例级检测难度自适应扰动并重建 ground-truth 标签，为单目 3D 检测提供显式几何监督，在 KITTI 上取得 SOTA。

**[Diffrefiner Coarse To Fine Trajectory Planning Via Diffusion Refinement With Sem](autonomous_driving/diffrefiner_coarse_to_fine_trajectory_planning_via_diffusion_refinement_with_sem.md)**

:   提出 DiffRefiner，通过"粗到精"两阶段框架——先用判别式 Proposal Decoder 生成粗轨迹，再用扩散模型迭代精炼——结合细粒度语义交互模块，在 NAVSIM v2 和 Bench2Drive 两个基准上均达到 SOTA。

**[Drive As You Like Strategy-Level Motion Planning Based On A Multi-Head Diffusion](autonomous_driving/drive_as_you_like_strategy-level_motion_planning_based_on_a_multi-head_diffusion.md)**

:   提出 M-Diffusion Planner，基于多头扩散模型和 GRPO 后训练，实现策略级（strategy-level）运动规划，允许用户通过自然语言切换激进/保守/舒适等驾驶风格，同时保持 SOTA 规划性能。

**[Driveflow Rectified Flow Adaptation For Robust 3D Object Detection In Autonomous](autonomous_driving/driveflow_rectified_flow_adaptation_for_robust_3d_object_detection_in_autonomous.md)**

:   提出 DriveFlow，一种基于预训练 T2I Flow 模型的 rectified flow 适配方法，通过频率分解对前景高频保持和背景双频优化，实现无需训练的驾驶场景图像编辑数据增强，大幅提升视觉 3D 检测器在 OOD 场景下的鲁棒性。

**[Drivesuprim Towards Precise Trajectory Selection For End-To-End Planning](autonomous_driving/drivesuprim_towards_precise_trajectory_selection_for_end-to-end_planning.md)**

:   提出 DriveSuprim，通过粗到精的轨迹筛选范式、旋转数据增强和自蒸馏软标签框架，解决选择式端到端规划中难以区分相似轨迹、方向偏差和硬标签不稳定的问题，在 NAVSIM v1/v2 和 Bench2Drive 上达到 SOTA。

**[Dual-Branch Spatial-Temporal Self-Supervised Representation For Enhanced Road Ne](autonomous_driving/dual-branch_spatial-temporal_self-supervised_representation_for_enhanced_road_ne.md)**

:   提出 DST（Dual-branch Spatial-Temporal）路网表示学习框架，通过空间分支（mix-hop 转移矩阵 + 图-超图对比学习）和时间分支（Transformer 编码器 + 下一 token 预测 + 工作日/周末分类）两条支路联合建模路网的空间异质性和时间动态性，在三个城市的三项下游任务上取得 SOTA。

**[Expertad Enhancing Autonomous Driving Systems With Mixture Of Experts](autonomous_driving/expertad_enhancing_autonomous_driving_systems_with_mixture_of_experts.md)**

:   提出 ExpertAD，将混合专家（MoE）架构引入端到端自动驾驶系统的感知和预测模块——Perception Adapter 动态重加权 BEV 特征以放大任务关键语义，Mixture of Sparse Experts 通过路由器动态激活相关驾驶任务专家并用稀疏注意力降低计算量，在保持或提升规划效果的同时降低约 25% 推理延迟。

**[Fastdrivevla Efficient End-To-End Driving Via Plug-And-Play ](autonomous_driving/fastdrivevla_efficient_end-to-end_driving_via_plug-and-play_.md)**

:   提出 FastDriveVLA，通过 MAE 风格的前景像素重建训练轻量级 plug-and-play 的 ReconPruner 模块（仅 0.07B），利用对抗前景-背景重建策略优先保留驾驶决策所需的前景 token，在 nuScenes 开环规划基准上各剪枝率均达 SOTA，一次训练可迁移至同一视觉编码器的不同 VLA 模型。

**[Fine-Grained Representation For Lane Topology Reasoning](autonomous_driving/fine-grained_representation_for_lane_topology_reasoning.md)**

:   提出TopoFG框架，用细粒度查询（每条车道线由多个空间感知查询表示）替代传统单查询建模，结合层级先验提取、区域聚焦解码和基于边界点的鲁棒拓扑推理，在OpenLane-V2上以48.0% OLS（subset_A）和45.4% OLS（subset_B）达到新SOTA。

**[Fq-Petr Fully Quantized Position Embedding Transformation Fo](autonomous_driving/fq-petr_fully_quantized_position_embedding_transformation_fo.md)**

:   首次实现PETR系列3D检测器的全INT8量化部署，通过量化友好的LiDAR-ray位置编码(QFPE)解决多模态特征幅度不匹配问题、双查找表(DULUT)高效逼近非线性算子、数值稳定后量化(QANS)避免softmax注意力失真，在PETR/StreamPETR/PETRv2/MV2D上W8A8精度损失<1%且延迟降低75%（3.9×加速）。

**[Generalising Traffic Forecasting To Regions Without Traffic Observations](autonomous_driving/generalising_traffic_forecasting_to_regions_without_traffic_observations.md)**

:   本文提出 GenCast 模型，通过物理信息神经网络（引入 LWR 交通方程作为软约束）、动态外部天气信号融合和空间分组模块三大创新，实现了从有传感器区域到无传感器连续区域的交通预测泛化，在五个真实数据集上一致性地超越了现有最优方法。

**[Global-Lens Transformers Adaptive Token Mixing For Dynamic Link Prediction](autonomous_driving/global-lens_transformers_adaptive_token_mixing_for_dynamic_link_prediction.md)**

:   提出 GLFormer，一个轻量级的无注意力 Transformer 框架用于动态图链接预测，用基于交互顺序和时间间隔的自适应令牌混合器替代自注意力，配合层次化聚合机制扩展时间感受野，在 6 个基准上取得了与 Transformer 基线持平或更优的性能，同时大幅降低计算复杂度。

**[Hd2-Ssc High-Dimension High-Density Semantic Scene Completion For Autonomous Dri](autonomous_driving/hd2-ssc_high-dimension_high-density_semantic_scene_completion_for_autonomous_dri.md)**

:   本文提出 HD2-SSC 框架，通过高维语义解耦（HSD）模块解决 2D→3D 的输入-输出维度间隙（将像素特征沿伪维度展开并正交解耦），以及高密度占用精炼（HOR）模块解决标注-现实密度间隙（"检测-精炼"范式对齐几何和语义关键体素），在 SemanticKITTI 和 SSCBench-KITTI-360 上达到 SOTA。

**[Hierarchical Prompt Learning For Image- And Text-Based Person Re-Identification](autonomous_driving/hierarchical_prompt_learning_for_image-_and_text-based_person_re-identification.md)**

:   提出统一框架 HPL，通过任务路由 Transformer（双分类 token）解耦 I2I 和 T2I 任务，利用层次化提示学习（身份级 + 实例级伪文本 token）结合跨模态提示正则化，首次在单一模型中同时实现图像-图像和文本-图像行人重识别的 SOTA。

**[I-Inr Iterative Implicit Neural Representations](autonomous_driving/i-inr_iterative_implicit_neural_representations.md)**

:   提出 I-INR（Iterative Implicit Neural Representations），一个即插即用的迭代精修框架，通过引入轻量级 FeedbackNet 和 FuseNet 模块（仅增加 0.5-2% 参数），对信号进行渐进式多步重建，有效缓解 INR 的频谱偏差问题，在图像拟合、超分辨率、去噪和 3D 占位预测等任务上均显著超越基线。

**[Invisible Triggers Visible Threats Road-Style Adversarial Creation Attack For Vi](autonomous_driving/invisible_triggers_visible_threats_road-style_adversarial_creation_attack_for_vi.md)**

:   提出 AdvRoad 框架，通过两阶段方法（Road-Style Adversary Generation + Scenario-Associated Adaptation）生成多样化、具有道路表面纹理风格的对抗海报，能够在自动驾驶视觉 3D 检测器中诱发"幽灵物体"（false positive），同时因外观自然而难以被人类驾驶员察觉，显著提升了 FP 攻击的隐蔽性和防御难度。

**[Lidar-Gsimproving Lidar Gaussian Reconstruction Via Diffusion Priors](autonomous_driving/lidar-gsimproving_lidar_gaussian_reconstruction_via_diffusion_priors.md)**

:   提出 LiDAR-GS++，通过引入**可控LiDAR扩散生成模型**作为先验，对神经2DGS场进行**扩展重建**，解决了单次遍历LiDAR扫描在外推视角（如换道场景）下重建质量严重下降的问题，在多个公开数据集上实现了插值和外推视角的SOTA性能。

**[Lidarcrafter Dynamic 4D World Modeling From Lidar Sequences](autonomous_driving/lidarcrafter_dynamic_4d_world_modeling_from_lidar_sequences.md)**

:   提出 LiDARCrafter，首个面向 LiDAR 的 4D 生成式世界模型，通过文本→场景图→三分支扩散布局→range-image 扩散生成→自回归时序扩展的流水线，实现可控的 4D LiDAR 序列生成与编辑，在 nuScenes 上全面超越现有方法。

**[Linext Revisiting Lidar Completion With Efficient Non-Diffusion Architectures](autonomous_driving/linext_revisiting_lidar_completion_with_efficient_non-diffusion_architectures.md)**

:   提出 LiNeXt，一种轻量级非扩散网络用于LiDAR 3D场景补全，通过**距离感知选择性重复策略**、**Noise-to-Coarse模块**和**Refine模块**直接重建完整点云，在SemanticKITTI上实现了比LiDiff快**199.8倍**的推理速度，Chamfer Distance降低**50.7%**，参数量仅为其**6.1%**。

**[Mambaseg Harnessing Mamba For Accurate And Efficient Image-E](autonomous_driving/mambaseg_harnessing_mamba_for_accurate_and_efficient_image-e.md)**

:   提出 MambaSeg，用双分支并行 Mamba 编码器分别处理 RGB 图像和事件流，通过空间-时间双维度交互模块 (DDIM) 实现细粒度跨模态融合，在 DDD17 和 DSEC 数据集上以 25.44M 参数取得 77.56%/75.10% mIoU 的 SOTA，效率远优于 Transformer 方案。

**[Meta Dynamic Graph For Traffic Flow Prediction](autonomous_driving/meta_dynamic_graph_for_traffic_flow_prediction.md)**

:   提出MetaDG框架，通过在每个时间步生成动态节点表示并利用时空相关性增强，将动态性建模从仅影响邻接矩阵扩展到同时生成meta参数、邻接矩阵和边权调整矩阵，实现时空异质性的统一建模（ST-unification），在PEMS03/04/07/08四个数据集上达到SOTA。

**[Minimum-Cost Network Flow With Dual Predictions](autonomous_driving/minimum-cost_network_flow_with_dual_predictions.md)**

:   首次提出基于对偶预测的最小费用网络流算法，在经典ε-relaxation上通过机器学习的对偶初始解实现warm-start，理论上将时间复杂度与预测误差的∞范数挂钩（一致且鲁棒），在交通网络和芯片逃逸布线上分别实现12.74×和1.64×的平均加速。

**[Moba A Material-Oriented Backdoor Attack Against Lidar-Based 3D Object Detection](autonomous_driving/moba_a_material-oriented_backdoor_attack_against_lidar-based_3d_object_detection.md)**

:   提出 MOBA（Material-Oriented Backdoor Attack），首个基于**材料反射特性建模**的物理可实现后门攻击框架，通过系统性选择二氧化钛（TiO₂）作为触发材料并利用**Oren-Nayar BRDF模型的角度无关近似**进行LiDAR强度仿真，在真实物理数据上实现了**93.50%攻击成功率**，比现有方法高出41%以上。

**[Multimodal Data Fusion To Capture Dynamic Interactions Between Built Environment](autonomous_driving/multimodal_data_fusion_to_capture_dynamic_interactions_between_built_environment.md)**

:   提出一种**多模态数据融合框架**，整合眼动追踪、运动传感器（IMU）、生理监测（EDA/HRV）、GPS和视频录制等多种穿戴与环境传感数据，动态表征脆弱老年人（膝骨关节炎/跌倒史）与城市建成环境的交互过程，通过AI驱动的数据融合揭示微观尺度上对步行行为和感知有显著影响的城市路段，为**适老化城市规划**提供循证依据。

**[Out-Of-Distribution Generalization With A Sparc Racing 100 U](autonomous_driving/out-of-distribution_generalization_with_a_sparc_racing_100_u.md)**

:   提出 SPARC（Single-Phase Adaptation for Robust Control），将 RMA 的两阶段上下文编码与历史适应统一为单阶段训练，在 Gran Turismo 7 高保真赛车模拟器中用单一策略驾驶100+未见车辆实现SOTA OOD泛化性能。

**[Priordrive Enhancing Online Hd Mapping With Unified Vector P](autonomous_driving/priordrive_enhancing_online_hd_mapping_with_unified_vector_p.md)**

:   提出 PriorDrive 框架，通过 Unified Vector Encoder (UVE) 和 Hybrid Prior Representation (HPQuery) 将多种向量化先验地图（SD地图、旧HD地图、历史预测地图）统一编码并集成到各种在线建图模型中，在 nuScenes 上 mAP 提升 14.3，兼容 query-based 和 non-query-based 两类建图架构。

**[Racketvision A Multiple Racket Sports Benchmark For Unified Ball And Racket Anal](autonomous_driving/racketvision_a_multiple_racket_sports_benchmark_for_unified_ball_and_racket_anal.md)**

:   提出 RacketVision——首个覆盖乒乓球、网球、羽毛球三种球拍运动的大规模基准数据集，首次提供球拍姿态标注，并定义了球追踪、球拍姿态估计、球轨迹预测三个互联任务，揭示了跨注意力融合机制在多模态轨迹预测中的关键作用。

**[Radarmp Motion Perception For 4D Mmwave Radar In Autonomous Driving](autonomous_driving/radarmp_motion_perception_for_4d_mmwave_radar_in_autonomous_driving.md)**

:   提出 RadarMP——首个联合解决毫米波雷达目标检测和场景流估计的统一架构，利用相邻帧雷达回波信号（tesseract）的能量流一致性进行自监督训练，在目标检测概率上达到 69.5%（远超现有方法的 44.1%），同时实现精确的 3D 场景运动感知。

**[Rast A Retrieval Augmented Spatio-Temporal Framework For Traffic Prediction](autonomous_driving/rast_a_retrieval_augmented_spatio-temporal_framework_for_traffic_prediction.md)**

:   将 Retrieval-Augmented Generation (RAG) 思想引入时空预测，通过维护双维度 memory bank 存储历史时空 pattern 并在推理时检索融合，构建通用的 retrieval-augmented 时空预测框架 RAST，在 6 个交通数据集上取得 SOTA 且计算效率优异。

**[Reflexdiffusion Reflection-Enhanced Trajectory Planning For ](autonomous_driving/reflexdiffusion_reflection-enhanced_trajectory_planning_for_.md)**

:   提出 ReflexDiffusion，在扩散模型推理阶段引入物理感知的反思机制，通过梯度注入强化曲率-速度-加速度耦合约束（a_y = κv²），在 nuPlan 高侧向加速度长尾场景中驾驶分数提升 14.1%，架构无关可直接部署到现有扩散规划器。

**[Rethinking The Spatio-Temporal Alignment Of End-To-End 3D Perception](autonomous_driving/rethinking_the_spatio-temporal_alignment_of_end-to-end_3d_perception.md)**

:   提出HAT（multiple Hypotheses spAtio-Temporal alignment），一个即插即用的时空对齐模块，通过多种显式运动模型生成对齐假设，并利用query中隐含的运动线索自适应解码最优对齐方案，在nuScenes上一致提升多种3D时序检测器和跟踪器，并在E2E自动驾驶中降低碰撞率达32-48%。

**[Roadscenevqa Benchmarking Visual Question Answering In Roadside Perception Syste](autonomous_driving/roadscenevqa_benchmarking_visual_question_answering_in_roadside_perception_syste.md)**

:   提出 RoadSceneVQA——首个面向路侧感知场景的大规模视觉问答数据集（34,736 QA 对），并设计了 RoadMind 模型，通过认知锚点融合（CAF）和辅助解耦思维链（AD-CoT）显著提升轻量级 MLLM 在交通场景推理中的表现，在 0.9B 参数下即可超越 8B 模型。

**[Sparsecoop Cooperative Perception With Kinematic-Grounded Queries](autonomous_driving/sparsecoop_cooperative_perception_with_kinematic-grounded_queries.md)**

:   提出 SparseCoop——首个完全稀疏的协同感知框架，通过运动学锚定查询（KGQ）、粗到精聚合模块和协同实例去噪策略，完全抛弃密集 BEV 表示，在 V2X-Seq 和 Griffin 数据集上以最低通信开销和最高计算效率达到 SOTA 性能（AP 0.530，传输仅 3.17×10⁴ BPS）。

**[Stride-Qa Visual Question Answering Dataset For Spatiotemporal Reasoning In Urba](autonomous_driving/stride-qa_visual_question_answering_dataset_for_spatiotemporal_reasoning_in_urba.md)**

:   构建了自动驾驶领域最大规模时空推理VQA数据集STRIDE-QA（270K帧、16M QA对），定义了三类时空推理任务（物体间空间/自车空间/自车时空），通过微调VLM使空间定位成功率从近零提升至55%、时序一致性从0提升至28.4%。

**[Task Prototype-Based Knowledge Retrieval For Multi-Task Lear](autonomous_driving/task_prototype-based_knowledge_retrieval_for_multi-task_lear.md)**

:   提出基于任务原型的知识检索框架，通过可学习 Task Prototype 嵌入任务特性并量化任务关联、Knowledge Retrieval Transformer 基于 task-affinity score 自适应精炼特征表示，在部分标注多任务学习（MTPSL）中避免依赖未标注任务的预测，PASCAL-Context 和 NYUD-v2 上全面超越 SOTA。

**[Tawpipe Topology-Aware Weight Pipeline Parallelism For Accelerating Long-Context](autonomous_driving/tawpipe_topology-aware_weight_pipeline_parallelism_for_accelerating_long-context.md)**

:   提出 TawPipe——拓扑感知的权重流水线并行框架，通过分组式权重调度、设备绑定存储和通信-计算重叠三大组件，利用分布式集群的层次化带宽特性，在 24 GPU 上训练 LLaMA 模型时吞吐量相比 WeiPipe/1F1B/FSDP 分别提升 11.8%/23.6%/44.1%，同时通信时间减少 82.1%。

**[Timebill Time-Budgeted Inference For Large Language Models](autonomous_driving/timebill_time-budgeted_inference_for_large_language_models.md)**

:   提出TimeBill框架，通过细粒度响应长度预测器（RLP）和工作负载引导的执行时间估计器（ETE），在给定时间预算下自适应调整KV Cache驱逐比例，在保证推理完成率的同时最大化LLM响应质量。

**[Towards 3D Object-Centric Feature Learning For Semantic Scene Completion](autonomous_driving/towards_3d_object-centric_feature_learning_for_semantic_scene_completion.md)**

:   提出Ocean框架，利用MobileSAM提取的实例掩码引导3D物体中心特征学习，通过语义组注意力（SGA3D）和全局相似性引导注意力（GSGA）在3D空间实现实例级特征聚合，并用实例感知局部扩散（ILD）模块精炼场景表征，在SemanticKITTI和SSCBench-KITTI360上达到SOTA。

**[Tsbow Traffic Surveillance Benchmark For Occluded Vehicles Under Various Weather](autonomous_driving/tsbow_traffic_surveillance_benchmark_for_occluded_vehicles_under_various_weather.md)**

:   提出TSBOW——一个基于CCTV的大规模交通监控数据集，包含198个视频、超32小时真实交通数据和320万帧，覆盖全年四季天气（晴/霾/雨/雪含极端灾害场景），涵盖8类交通参与者，重点解决恶劣天气下遮挡车辆检测的挑战。

**[Understanding Dynamic Scenes In Ego Centric 4D Point Clouds](autonomous_driving/understanding_dynamic_scenes_in_ego_centric_4d_point_clouds.md)**

:   构建EgoDynamic4D——首个面向高度动态4D场景的自我中心视角QA基准（927K QA对、12种任务），并提出端到端时空推理框架，通过实例感知特征编码、时间编码、相机编码和自适应下采样将大规模4D场景压缩为LLM可处理的token序列。

**[Unleashing Semantic And Geometric Priors For 3D Scene Completion](autonomous_driving/unleashing_semantic_and_geometric_priors_for_3d_scene_completion.md)**

:   提出 FoundationSSC 框架，通过 source-level 和 pathway-level 双层解耦设计释放 Vision Foundation Model 的语义与几何先验，配合 Axis-Aware Fusion 模块融合互补 3D 特征，在 SemanticKITTI 上达到 19.32 mIoU / 48.12 IoU SOTA。

**[Unlocking Efficient Vehicle Dynamics Modeling Via Analytic World Models](autonomous_driving/unlocking_efficient_vehicle_dynamics_modeling_via_analytic_world_models.md)**

:   提出解析世界模型（Analytic World Models, AWMs），利用可微分模拟器的可微性设计三种世界建模任务（相对里程计、最优规划器、逆最优状态估计），无需试错搜索即可端到端高效训练状态预测器，在Waymax自动驾驶模拟器上验证了其有效性。

**[Vilta A Vlm-In-The-Loop Adversary For Enhancing Driving Poli](autonomous_driving/vilta_a_vlm-in-the-loop_adversary_for_enhancing_driving_poli.md)**

:   VILTA 将 VLM（Gemini-2.5-Flash）直接嵌入自动驾驶 RL 训练循环中，通过"Vision-Language-Editing"（VLE）范式让 VLM 编辑周围车辆的未来轨迹来生成具有挑战性的危险场景，训练出的驾驶策略在 CARLA 挑战场景中路线完成率提升 13.3%、碰撞率降低 28.5%。

**[Visiononly Gaussian Splatting For Collaborative Semantic Occupancy P](autonomous_driving/visiononly_gaussian_splatting_for_collaborative_semantic_occupancy_p.md)**

:   提出首个使用稀疏3D语义高斯基元作为协同感知通信介质的纯视觉语义占据预测框架，通过ROI裁剪+刚性变换传输高斯+邻域融合模块抑制噪声冗余，在mIoU上比单车提升+8.42，比baseline协同方法提升+3.28。

**[Walking Further Semantic-Aware Multimodal Gait Recognition Under Long-Range Cond](autonomous_driving/walking_further_semantic-aware_multimodal_gait_recognition_under_long-range_cond.md)**

:   构建LRGait——首个面向长距离（10-50m）跨距离场景的LiDAR-Camera多模态步态数据集，并提出EMGaitNet端到端框架，通过CLIP语义挖掘（SeMi）、语义引导对齐（SGA）和对称交叉注意力融合（SCAF）模块实现2D-3D跨模态特征融合，在多个基准上达到SOTA。

**[When Person Re-Identification Meets Event Camera A Benchmark Dataset And An Attr](autonomous_driving/when_person_re-identification_meets_event_camera_a_benchmark_dataset_and_an_attr.md)**

:   构建首个大规模 RGB-Event 行人重识别数据集 EvReID（1200 ID / 118,988 图像对），并提出基于行人属性引导的三阶段对比学习框架 TriPro-ReID，通过正负属性 prompt 和跨模态 prompt 融合 RGB 与 Event 两种模态特征，mAP 达 69.3%。

**[Worldrft Latent World Model Planning With Reinforcement Fine-Tuning For Autonomo](autonomous_driving/worldrft_latent_world_model_planning_with_reinforcement_fine-tuning_for_autonomo.md)**

:   提出面向规划的潜在世界模型框架WorldRFT，通过VGGT空间编码、分层规划分解+局部感知迭代精炼、基于GRPO的碰撞感知强化微调，在nuScenes上将碰撞率降低83%（0.30%→0.05%），在NavSim上仅用相机即逼近LiDAR SOTA（87.8 vs 88.1 PDMS）。

---

## 📦 模型压缩 { #model_compression }

**[Adafuse Accelerating Dynamic Adapter Inference Via Token-Lev](model_compression/adafuse_accelerating_dynamic_adapter_inference_via_token-lev.md)**

:   针对动态MoE-LoRA适配器推理延迟暴增（250%-950%）的问题，提出了一种token级预门控架构，只在第一层做一次全局路由决策，配合自研的SGMM融合CUDA内核将所有激活的LoRA适配器一次性合并进骨干网络，在保持精度的同时将解码延迟降低2.4倍。

**[Agentodrl A Large Language Model-Based Multi-Agent System Fo](model_compression/agentodrl_a_large_language_model-based_multi-agent_system_fo.md)**

:   提出AgentODRL，一个基于Orchestrator-Workers架构的LLM多智能体系统，通过任务分解、语法验证循环和LoRA驱动的语义反思机制，将自然语言数据权限规则高质量地转换为ODRL格式。

**[Alter Asymmetric Lora For Token-Entropy-Guided Unlearning Of](model_compression/alter_asymmetric_lora_for_token-entropy-guided_unlearning_of.md)**

:   提出ALTER框架，利用非对称LoRA架构结合Token级别的Tsallis熵引导，实现LLM中目标知识的精准遗忘，同时通过参数隔离机制保留模型基础能力，在TOFU、WMDP和MUSE三个基准上达到SOTA。

**[Beyond Sharpness A Flatness Decomposition Framework For Efficient Continual Lear](model_compression/beyond_sharpness_a_flatness_decomposition_framework_for_efficient_continual_lear.md)**

:   提出 FLAD 框架，将 sharpness-aware 扰动方向分解为梯度对齐分量与随机噪声分量，仅保留噪声分量进行正则化，结合零阶与一阶 sharpness 以极低额外开销提升持续学习的泛化能力。

**[Break The Tie Learning Cluster-Customized Category Relationships For Categorical](model_compression/break_the_tie_learning_cluster-customized_category_relationships_for_categorical.md)**

:   提出 DISC 方法，为每个聚类簇学习定制化的属性类别关系（而非全局统一距离），通过关系树建模与聚类联合优化，在 12 个数据集上以平均排名 1.25 大幅超越现有最佳方法（5.21）。

**[Camera Multi-Matrix Joint Compression For Moe Models Via Mic](model_compression/camera_multi-matrix_joint_compression_for_moe_models_via_mic.md)**

:   提出"micro-expert"概念将MoE层的输出分解为跨矩阵（up/gate/down_proj）的微专家线性组合，基于能量排序进行结构化剪枝(Camera-P)和混合精度量化(Camera-Q)，在Deepseek-MoE-16B/Qwen2-57B/Qwen3-30B上20%-60%剪枝率全面超越NAEE和D²-MoE，且分析Qwen2-57B仅需单卡A100不到5分钟。

**[Can You Tell The Difference Contrastive Explanations For Abox Entailments](model_compression/can_you_tell_the_difference_contrastive_explanations_for_abox_entailments.md)**

:   提出对比式ABox解释（Contrastive ABox Explanations）的形式化框架，用于回答"为什么a是C的实例而b不是"的问题，在描述逻辑知识库中同时考虑正向蕴涵和缺失蕴涵，并分析不同描述逻辑和优化准则下的计算复杂度。

**[Coevo Continual Evolution Of Symbolic Solutions Using Large Language Models](model_compression/coevo_continual_evolution_of_symbolic_solutions_using_large_language_models.md)**

:   提出CoEvo框架，结合LLM与进化搜索方法论，通过动态知识库和多表示空间（自然语言/数学公式/代码）实现符号解的持续开放式进化，在AI Feynman基准上大幅超越现有符号回归方法。

**[Compensating Distribution Drifts In Class-Incremental Learning Of Pre-Trained Vi](model_compression/compensating_distribution_drifts_in_class-incremental_learning_of_pre-trained_vi.md)**

:   提出 Sequential Learning with Drift Compensation (SLDC)，通过学习潜在空间转换算子（线性/弱非线性）来补偿预训练 ViT 在类增量学习中因序列微调导致的分布漂移，结合知识蒸馏后性能接近联合训练上界。

**[Condensed Data Expansion Using Model Inversion For Knowledge Distillation](model_compression/condensed_data_expansion_using_model_inversion_for_knowledge_distillation.md)**

:   提出用浓缩数据集作为原型指导模型反演（MI）过程，通过特征对齐判别器使生成的合成数据与浓缩样本分布一致，从而扩展浓缩数据集用于知识蒸馏，在 CIFAR/ImageNet 上比标准 MI 蒸馏提升高达 11.4%。

**[Correcting False Alarms From Unseen Adapting Graph Anomaly Detectors At Test Tim](model_compression/correcting_false_alarms_from_unseen_adapting_graph_anomaly_detectors_at_test_tim.md)**

:   提出 TUNE，一个即插即用的测试时适应框架，通过图对齐器变换节点特征来解决图异常检测中因新正常类别出现导致的"正常性偏移"问题，利用聚合污染程度作为无监督适应信号，在 10 个真实数据集上显著增强多种预训练 GAD 模型的泛化能力。

**[Credal Ensemble Distillation For Uncertainty Quantification](model_compression/credal_ensemble_distillation_for_uncertainty_quantification.md)**

:   提出Credal Ensemble Distillation（CED）框架，将深度集成教师蒸馏为单模型CREDIT，该模型预测类别概率区间（定义credal集）而非单一softmax分布，在OOD检测任务上实现了优于或可比的不确定性估计，同时大幅降低推理开销（推理时间从5×降为1×）。

**[Distilling Cross-Modal Knowledge Via Feature Disentanglement](model_compression/distilling_cross-modal_knowledge_via_feature_disentanglement.md)**

:   提出频域解耦跨模态知识蒸馏（FD-CMKD），通过傅里叶变换将特征分解为低频（模态共享语义）和高频（模态特有细节）分量，分别施加强一致性 MSE 和弱一致性 logMSE 损失，并引入尺度标准化与共享分类器对齐特征空间，在音频-视觉、图像-文本、语义分割等多个跨模态场景全面超越现有蒸馏方法。

**[Dont Start Over A Cost-Effective Framework For Migrating Personalized Prompts Be](model_compression/dont_start_over_a_cost-effective_framework_for_migrating_personalized_prompts_be.md)**

:   提出PUMA框架，通过轻量级适配器和分组用户选择策略，高效地将个性化软提示从源LLM迁移到不同架构的目标LLM，在三个大规模数据集上匹配甚至超越从头训练的性能，同时减少计算成本高达98%。

**[Dos Distilling Observable Softmaps Of Zipfian Prototypes For Self-Supervised Poi](model_compression/dos_distilling_observable_softmaps_of_zipfian_prototypes_for_self-supervised_poi.md)**

:   提出DOS框架，通过仅在可观测（未掩码）点上蒸馏语义软图（Softmap），结合Zipfian先验的Zipf-Sinkhorn正则化来处理3D语义的长尾分布，在六个3D基准上实现了自监督学习的SOTA，线性探测可达监督性能的95%。

**[Dp-Geng Differentially Private Dataset Distillation Guided By Dp-Generated Data](model_compression/dp-geng_differentially_private_dataset_distillation_guided_by_dp-generated_data.md)**

:   提出 DP-GenG 框架，利用差分隐私生成数据（DP-generated data）引导数据集蒸馏的初始化、特征匹配和专家校准三个阶段，在有限隐私预算下显著提升蒸馏数据集的实用性和隐私保护能力。

**[Dynaquant Dynamic Mixed-Precision Quantization For Learned I](model_compression/dynaquant_dynamic_mixed-precision_quantization_for_learned_i.md)**

:   针对学习图像压缩（LIC）模型部署效率低的痛点，提出DynaQuant框架，在参数层面通过可学习scale/zero-point + Distance-Aware Gradient Modulator实现内容自适应量化，在架构层面通过轻量Bit-Width Selector动态为每层分配最优比特宽度，在Cheng2020/ELIC/Ballé三个基线上实现接近FP32的R-D性能，同时获得最高5.17×加速和模型大小降至原来的~1/4。

**[Earth-Adapter Bridge The Geospatial Domain Gaps With Mixture Of Frequency Adapta](model_compression/earth-adapter_bridge_the_geospatial_domain_gaps_with_mixture_of_frequency_adapta.md)**

:   提出 Earth-Adapter，首个针对遥感图像**伪影问题**设计的参数高效微调 (PEFT) 方法，通过频率引导的混合适配器 (MoA) 将特征分解为高低频子空间、独立优化后动态聚合，在遥感语义分割 (SS)、域自适应 (DA) 和域泛化 (DG) 三个设定中均超越基线 Rein。

**[Eeg-Dlite Dataset Distillation For Efficient Large Eeg Model Training](model_compression/eeg-dlite_dataset_distillation_for_efficient_large_eeg_model_training.md)**

:   提出 EEG-DLite 数据蒸馏框架，通过自监督编码+异常值过滤+多样性采样，将2500小时 EEG 数据集压缩至仅5%即可达到甚至超越全数据集预训练的基础模型性能，GPU预训练时间从30小时降至2小时。

**[Efficient Reasoning For Large Reasoning Language Models Via Certainty-Guided Ref](model_compression/efficient_reasoning_for_large_reasoning_language_models_via_certainty-guided_ref.md)**

:   提出 CGRS（Certainty-Guided Reflection Suppression），一种无需训练的高效推理方法，通过在模型高置信度时动态抑制反思触发词（如"Wait""But"），将大型推理语言模型的 token 消耗降低18.5%~41.9%，同时保持推理精度不变。

**[Efficient Thought Space Exploration Through Strategic Intervention](model_compression/efficient_thought_space_exploration_through_strategic_intervention.md)**

:   提出 Hint-Practice Reasoning（HPR）框架，通过大模型（hinter）在稀疏关键 token 处提供短提示、小模型（practitioner）完成主要推理的协作模式，仅需1/5的 token 即可达到 self-consistency 基线的性能，同时在相同 FLOPs 下精度最高提升5.1%。

**[Efficientfsl Enhancing Few-Shot Classification Via Query-Only Tuning In Vision T](model_compression/efficientfsl_enhancing_few-shot_classification_via_query-only_tuning_in_vision_t.md)**

:   提出 EfficientFSL，一种针对 ViT 少样本分类的 query-only 参数高效微调框架，通过 Forward Block（解耦的主动/冻结子块）、Combine Block（自适应多层特征融合）和 SQ Attention Block（支持-查询分布对齐）三个模块，仅用1.25M~2.48M可训练参数即可在4个域内+6个跨域基准上达到 SOTA。

**[Explore And Establish Synergistic Effects Between Weight Pruning And Coreset Sel](model_compression/explore_and_establish_synergistic_effects_between_weight_pruning_and_coreset_sel.md)**

:   首次系统探索权重剪枝与核心集选择之间的交互关系，提出SWaST机制交替执行两者以建立协同效应，并设计状态保持机制解决"双重损失"问题，在10%–90% FLOPs削减下实现最高17.83%的精度提升。

**[First-Order Error Matters Accurate Compensation For Quantized Large Language Mod](model_compression/first-order_error_matters_accurate_compensation_for_quantized_large_language_mod.md)**

:   本文揭示了LLM后训练量化中逐列补偿过程会导致一阶梯度项不可忽略的问题，提出FOEM方法通过将一阶项纳入误差补偿公式来提升量化精度，在3-bit量化下将Llama3-8B的困惑度降低17.3%，且几乎不增加计算开销。

**[Hcf Hierarchical Cascade Framework For Distributed Multi-Stage Image Compression](model_compression/hcf_hierarchical_cascade_framework_for_distributed_multi-stage_image_compression.md)**

:   本文提出HCF框架，通过直接在潜在空间进行跨节点变换（避免像素域重压缩）并引入策略驱动的量化控制，在分布式多级图像压缩中实现了最高12.64% BD-Rate的PSNR提升，同时节省高达97.8%的FLOPs和96.5%的GPU内存。

**[Hierarchical Pedagogical Oversight A Multi-Agent Adversarial Framework For Relia](model_compression/hierarchical_pedagogical_oversight_a_multi-agent_adversarial_framework_for_relia.md)**

:   本文提出HPO框架，通过三阶段流水线（情报蒸馏→对抗辩论→综合判定）实现可靠的AI辅导评估，仅用8B参数的模型在MRBench中学数学对话数据集上以Macro F1 0.845超越GPT-4o（0.812）3.3%，证明了交互结构而非模型规模是可靠AI辅导的关键。

**[Infocom Kilobyte-Scale Communication-Efficient Collaborative Perception With Inf](model_compression/infocom_kilobyte-scale_communication-efficient_collaborative_perception_with_inf.md)**

:   提出InfoCom框架，基于扩展的信息瓶颈原理将协同感知的通信量从MB级压缩至KB级（相比Where2comm降低440倍），同时保持近无损的感知性能，核心包含信息感知编码、稀疏掩码生成和多尺度解码三个模块。

**[Kvmix Gradient-Based Layer Importance-Aware Mixed-Precision ](model_compression/kvmix_gradient-based_layer_importance-aware_mixed-precision_.md)**

:   提出 KVmix，通过计算 Key/Value 投影权重梯度的 $L_2$ 范数来评估各层 KV Cache 的重要性，实现层级混合精度量化（Key 平均 2.19bit、Value 平均 2.38bit），并结合动态关键上下文选择（RPC）策略，在 Llama/Mistral 等模型上实现近无损推理、4.9× 内存压缩和 5.3× 吞吐加速。

**[Lexchronos An Agentic Framework For Structured Event Timeline Extraction In Indi](model_compression/lexchronos_an_agentic_framework_for_structured_event_timeline_extraction_in_indi.md)**

:   本文提出LexChronos，一个双智能体迭代框架，用于从印度最高法院判决书中提取结构化事件时间线：通过LoRA微调的抽取智能体识别候选事件，预训练的反馈智能体通过置信度驱动的循环进行评分和精炼，在合成数据集上取得BERT F1 0.8751，且结构化时间线在下游的法律文本摘要中被GPT-4在75%的案例中评为优于非结构化基线。

**[Lightweight Optimal-Transport Harmonization On Edge Devices](model_compression/lightweight_optimal-transport_harmonization_on_edge_devices.md)**

:   提出 MKL-Harmonizer，利用经典最优传输理论中的 Monge-Kantorovich 线性映射（MKL），训练一个轻量级编码器预测 12 维颜色变换参数，实现边缘设备上的实时图像颜色协调，在 AR 场景的感知质量-速度综合指标上达到最优。

**[Parametric Pareto Set Learning For Expensive Multi-Objective Optimization](model_compression/parametric_pareto_set_learning_for_expensive_multi-objective_optimization.md)**

:   本文提出 PPSL-MOBO 框架，通过超网络 + LoRA 架构学习从偏好和外在参数到 Pareto 最优解的统一映射，结合高斯过程代理模型和超体积改进采集策略，高效解决昂贵的参数化多目标优化问题。

**[Pocketllm Ultimate Compression Of Large Language Models Via Meta Networks](model_compression/pocketllm_ultimate_compression_of_large_language_models_via_meta_networks.md)**

:   PocketLLM提出通过元网络（编码器-码本-解码器）在潜空间中压缩LLM权重向量，用小型解码器+紧凑码本+索引替代原始权重矩阵，在Llama 2-7B上实现10×压缩且精度损失可忽略，突破了传统量化/剪枝在极端压缩比下的精度瓶颈。

**[Post Training Quantization For Efficient Dataset Condensation](model_compression/post_training_quantization_for_efficient_dataset_condensation.md)**

:   首次将训练后量化（PTQ）应用于数据集蒸馏，提出基于补丁的量化框架（PAQ+分组+精炼），在 2-bit 极低比特下将蒸馏数据集的测试精度几乎翻倍（如 DM IPC=1 从 26.0% 提升至 54.1%），作为即插即用框架可应用于各种蒸馏方法。

**[Prefixgpt Prefix Adder Optimization By A Generative Pre-Trained Transformer](model_compression/prefixgpt_prefix_adder_optimization_by_a_generative_pre-trained_transformer.md)**

:   提出PrefixGPT，将前缀加法器优化建模为序列生成问题，通过定制的GPT模型预训练学习设计规则后用RL微调生成优化设计，在面积-延迟乘积(ADP)上取得SOTA且对初始化不敏感。

**[Prototype-Based Semantic Consistency Alignment For Domain Adaptive Retrieval](model_compression/prototype-based_semantic_consistency_alignment_for_domain_adaptive_retrieval.md)**

:   提出 PSCA 两阶段框架，通过正交 prototype 建立类级语义连接，结合几何-语义一致性对齐动态修正伪标签可靠性，并在重建特征上进行 hash 编码，显著提升跨域检索性能。

**[Put The Space Of Lora Initialization To The Extreme To Preserve Pre-Trained Know](model_compression/put_the_space_of_lora_initialization_to_the_extreme_to_preserve_pre-trained_know.md)**

:   提出 LoRA-Null，将 LoRA 初始化在预训练知识 input activation 的 null space 中（而非权重的 null space），从信息论角度论证 activation 的 effective rank 远小于权重，因此其 null space 包含更少预训练知识信息，显著减轻微调时的灾难性遗忘。

**[Quept Quantized Elastic Precision Transformers With One-Shot Calibration For Mul](model_compression/quept_quantized_elastic_precision_transformers_with_one-shot_calibration_for_mul.md)**

:   提出QuEPT弹性精度量化框架，通过Multi-Bit Token Merging和Multi-Bit Cascaded LoRA两大核心模块，实现一次校准即可在ViT/LLM/MLLM上实时切换任意预定义位宽，性能媲美甚至超越单位宽SOTA PTQ方法。

**[Reinforced Rate Control For Neural Video Compression Via Inter-Frame Rate-Distor](model_compression/reinforced_rate_control_for_neural_video_compression_via_inter-frame_rate-distor.md)**

:   提出首个基于约束马尔可夫决策过程（CMDP）的强化学习速率控制框架，通过时空状态建模联合捕获帧内容特征与帧间率-失真耦合依赖，直接映射到逐帧编码参数，在多种神经视频编解码器上将平均比特率误差降至1.20%，BD-Rate节省最高达13.98%。

**[Rethinking Long-Tailed Dataset Distillation A Uni-Level Framework With Unbiased ](model_compression/rethinking_long-tailed_dataset_distillation_a_uni-level_framework_with_unbiased_.md)**

:   提出首个面向长尾分布的单层(uni-level)数据集蒸馏框架，通过专家模型去偏、BN统计量公平校准和置信度引导初始化三大策略，在CIFAR-100-LT上提升15.6%、Tiny-ImageNet-LT上提升11.8%，全面超越DAMED。

**[Safesieve From Heuristics To Experience In Progressive Pruning For Llm-Based Mul](model_compression/safesieve_from_heuristics_to_experience_in_progressive_pruning_for_llm-based_mul.md)**

:   提出SafeSieve，一种渐进式自适应多智能体通信剪枝框架，通过语义启发初始化→历史反馈驱动的双阶段边评分和0-extension聚类机制，在6个基准上实现94.01%平均准确率同时减少12.4%-27.8% token消耗，并展现出对prompt注入攻击的天然鲁棒性。

**[Satisficing And Optimal Generalised Planning Via Goal Regression Extended Versio](model_compression/satisficing_and_optimal_generalised_planning_via_goal_regression_extended_versio.md)**

:   提出 Moose 规划器，利用目标回归（goal regression）从训练问题中合成泛化规划程序：将训练问题的目标拆解为单目标子问题逐个最优求解，通过回归和提升（lifting）得到一阶条件-动作规则集，用于满足性规划（直接执行规则）或最优规划（编码为公理剪枝搜索空间）。

**[Share Your Attention Transformer Weight Sharing Via Matrix-Based Dictionary Lear](model_compression/share_your_attention_transformer_weight_sharing_via_matrix-based_dictionary_lear.md)**

:   受字典学习启发，提出 MASA 框架，将 Transformer 各层注意力投影矩阵（Q/K/V/O）分解为共享矩阵原子的线性组合，以 66.7% 的注意力参数压缩率实现与原始 Transformer 持平甚至更优的性能。

**[Sharp Eyes And Memory For Videollms Information-Aware Visual Token Pruning For E](model_compression/sharp_eyes_and_memory_for_videollms_information-aware_visual_token_pruning_for_e.md)**

:   SharpV 提出一个两阶段无训练视觉Token剪枝框架，在Pre-LLM阶段基于时空信息自适应调整每帧剪枝比例，在Intra-LLM阶段基于视觉信息退化假说进行KV Cache剪枝，首次实现与Flash Attention完全兼容，在多个视频理解基准上以约12%的Token保留率达到与稠密模型相当甚至更优的性能。

**[Shrinking The Teacher An Adaptive Teaching Paradigm For Asymmetric Eeg-Vision Al](model_compression/shrinking_the_teacher_an_adaptive_teaching_paradigm_for_asymmetric_eeg-vision_al.md)**

:   提出自适应教学范式（Adaptive Teaching Paradigm），通过无残差连接的瓶颈结构 ShrinkAdapter 让视觉"教师"主动收缩和调整其知识结构以适配 EEG"学生"的学习能力，在零样本脑-图像检索任务上 Top-1 准确率达到 60.2%，超越前 SOTA 9.8 个百分点。

**[Sign Schema-Induced Games For Naming](model_compression/sign_schema-induced_games_for_naming.md)**

:   SIGN 提出在LLM多智能体命名博弈中引入轻量级消息Schema（如 `@say {name: Ck}`），发现结构化先验可将群体约定一致性提升5.8×，收敛所需Token减少一个数量级，为高效多智能体协调提供了简单可控的"调节旋钮"。

**[Skipcat Rank-Maximized Low-Rank Compression Of Large Language Models Via Shared ](model_compression/skipcat_rank-maximized_low-rank_compression_of_large_language_models_via_shared_.md)**

:   SkipCat 提出了一种秩最大化的低秩压缩框架，通过层内共享投影（Cat）和块跳跃（Skip）两项技术，在相同压缩率下保留更多有效秩，无需微调即可在零样本任务上比现有低秩方法提升7%准确率。

**[Sparserm A Lightweight Preference Modeling With Sparse Autoencoder](model_compression/sparserm_a_lightweight_preference_modeling_with_sparse_autoencoder.md)**

:   SparseRM 利用稀疏自编码器（SAE）从LLM中间表示中提取偏好相关方向，通过投影向量构建轻量级奖励模型，仅需不到1%的可训练参数即可超越大多数主流奖励模型，并在在线迭代对齐框架中表现出更强的泛化能力。

**[Specquant Spectral Decomposition And Adaptive Truncation For Ultra-Low-Bit Llms ](model_compression/specquant_spectral_decomposition_and_adaptive_truncation_for_ultra-low-bit_llms_.md)**

:   SpecQuant 提出一种基于自适应傅里叶域分解的两阶段量化框架：先将激活离群值平滑迁移到权重，再通过通道级低频傅里叶截断吸收权重中的高频噪声，在LLaMA-3 8B上实现W4A4量化仅1.5%精度损失，同时获得2×加速和3×内存节省。

**[Steering Pretrained Drafters During Speculative Decoding](model_compression/steering_pretrained_drafters_during_speculative_decoding.md)**

:   提出 SD²，通过从验证器隐藏状态中提取转向向量（steering vector）并注入预训练 drafter 的 MLP 层，实现推测解码中 drafter-verifier 的动态对齐，在标准采样下将被接受 token 数提升高达 35%，同时计算开销可忽略不计。

**[Stepfun-Formalizer Unlocking The Autoformalization Potential Of Llms Through Kno](model_compression/stepfun-formalizer_unlocking_the_autoformalization_potential_of_llms_through_kno.md)**

:   提出 ThinkingF 流水线，通过大规模知识蒸馏与模板引导的推理轨迹合成分别增强 LLM 的形式语言领域知识和非形式到形式的推理能力，再经两阶段 SFT + RLVR 融合两种能力，7B/32B 模型在 FormalMATH-Lite 和 ProverBench 上达到 SOTA。

**[Stratified Knowledge-Density Super-Network For Scalable Vision Transformers](model_compression/stratified_knowledge-density_super-network_for_scalable_vision_transformers.md)**

:   提出将预训练 ViT 转化为"分层知识密度超网络"（SKD Super-Network），通过 WPAC（加权 PCA 注意力收缩）和 PIAD（渐进式重要性感知 Dropout）两步实现知识的分层组织，使得任意大小的子网络均可以 O(1) 代价提取，且无需额外微调即可达到或超越 SOTA 压缩方法的性能。

**[Structured Language Generation Model Loss Calibration And Formatted Decoding For](model_compression/structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)**

:   提出 SLGM 框架，通过**结构化输入格式**、**格式损失**和**格式感知解码**三大组件，将生成式语言模型的结构化预测任务重构为分类问题，在不增加模型参数的前提下显著提升 <1B 模型在 NER、RE、SRL 等 5 类 13 个数据集上的结构预测性能。

**[Tgdd Trajectory Guided Dataset Distillation With Balanced Distribution](model_compression/tgdd_trajectory_guided_dataset_distillation_with_balanced_distribution.md)**

:   提出 TGDD，将静态分布匹配重新定义为沿训练轨迹的动态对齐过程，通过阶段式分布匹配（Stage-wise Distribution Matching）捕获演化语义 + 分布约束正则化（Stage-wise Distribution Constraint）减少类间重叠，在 10 个数据集上达到 SOTA，高分辨率基准上准确率提升 5.0%。

**[Towards Test-Time Efficient Visual Place Recognition Via Asymmetric Query Proces](model_compression/towards_test-time_efficient_visual_place_recognition_via_asymmetric_query_proces.md)**

:   提出面向视觉位置识别（VPR）的高效非对称框架 AsymVPR，通过**地理记忆库**替代昂贵的 k-NN 预计算，以及**隐式嵌入增强**弥合轻量查询网络与高容量图库网络的能力差距，实现仅用 ~8% FLOPs 的轻量网络达到接近全尺寸模型的检索性能。

---

## 🦾 LLM Agent { #llm_agent }

**[A2Flow Automating Agentic Workflow Generation Via Self-Adaptive Abstraction Oper](llm_agent/a2flow_automating_agentic_workflow_generation_via_self-adaptive_abstraction_oper.md)**

:   提出 A2Flow 框架，通过三阶段流水线（案例生成→功能聚类→深度提取）从专家数据中全自动提取可复用的抽象执行算子，替代人工预定义算子，并引入算子记忆机制累积中间输出辅助节点决策，在 8 个基准上整体超越 AFLOW 等 SOTA，资源消耗降低 37%。

**[A Multi-Agent Llm Framework For Multi-Domain Low-Resource In](llm_agent/a_multi-agent_llm_framework_for_multi-domain_low-resource_in.md)**

:   提出 KDR-Agent 多智能体框架，通过知识检索（Wikipedia）、歧义消解和反思式自我纠错三个专业智能体协同工作，在仅使用少量静态标注示例的条件下，在5个领域10个NER数据集上显著超越现有零样本和少样本ICL NER方法。

**[Agentsense Virtual Sensor Data Generation Using Llm Agents I](llm_agent/agentsense_virtual_sensor_data_generation_using_llm_agents_i.md)**

:   利用LLM驱动的具身智能体在模拟智能家居中"生活"，生成虚拟环境传感器数据用于预训练HAR模型，在低资源场景下显著提升活动识别性能。

**[Agentswift Efficient Llm Agent Design Via Value-Guided Hierarchical Search](llm_agent/agentswift_efficient_llm_agent_design_via_value-guided_hierarchical_search.md)**

:   提出AgentSwift框架，通过层次化搜索空间（同时优化agentic workflow和功能组件）、轻量级value model预测agent性能、以及不确定性引导的MCTS搜索策略，自动发现高性能LLM agent设计，在7个基准上平均提升8.34%。

**[Aquasentinel Next-Generation Ai System Integrating Sensor Ne](llm_agent/aquasentinel_next-generation_ai_system_integrating_sensor_ne.md)**

:   提出AquaSentinel，一个物理信息驱动的AI系统，通过稀疏传感器部署+物理增强虚拟传感器+MoE时空GNN集成+双阈值RTCA检测算法+因果流定位+LLM报告生成，仅用20-30%节点覆盖即可实现全网管道泄漏检测，在110个泄漏场景中达到100%检测率。

**[Arcane A Multi-Agent Framework For Interpretable And Configurable Alignment](llm_agent/arcane_a_multi-agent_framework_for_interpretable_and_configurable_alignment.md)**

:   提出ARCANE框架，将对齐建模为多智能体协作问题——manager agent通过与stakeholder对话学习生成自然语言rubric（加权可验证准则集），作为worker agent的可解释代理奖励函数，通过SFT+GSPO两阶段训练实现测试时可配置的对齐，在GDPVal基准上GSPO版本的mean return从0.58提升至0.74（N=8）。

**[Automating Complex Document Workflows Via Stepwise And Rollback-Enabled Operatio](llm_agent/automating_complex_document_workflows_via_stepwise_and_rollback-enabled_operatio.md)**

:   提出AutoDW框架，通过逐步规划（每次生成一个API调用）+自适应回滚（参数级+API级两层回滚）实现复杂文档工作流自动化，在250会话/1708指令的DWBench上达到90%指令级和62%会话级完成率，分别超越最强基线40%和76%。

**[Autotool Efficient Tool Selection For Large Language Model A](llm_agent/autotool_efficient_tool_selection_for_large_language_model_a.md)**

:   提出AutoTool，一个基于图的免训练工具选择框架，通过发现并利用"工具使用惯性"（tool usage inertia）——即工具调用遵循可预测的顺序模式这一经验现象——构建工具惯性图（TIG），用统计方法代替部分LLM推理来高效选择工具和填充参数，在保持任务完成率的同时减少15-40%的推理成本。

**[Bayesagent Bayesian Agentic Reasoning Under Uncertainty Via ](llm_agent/bayesagent_bayesian_agentic_reasoning_under_uncertainty_via_.md)**

:   提出 vPGM 框架，通过自然语言引导 LLM Agent 模拟概率图模型（PGM）的贝叶斯推理过程，发现隐变量并推断后验分布，再用 Dirichlet 先验做数值贝叶斯校准（BayesVPGM），在多个推理任务上同时提升准确率和置信度校准。

**[Beyond React A Planner-Centric Framework For Complex Tool-Au](llm_agent/beyond_react_a_planner-centric_framework_for_complex_tool-au.md)**

:   提出以Planner为核心的Plan-Execute框架，将复杂查询转化为DAG执行计划，通过SFT+GRPO两阶段训练专门的Planner模型，在ComplexTool-Plan和StableToolBench上超越ReAct等反应式方法，用更少推理步骤实现更高成功率。

**[Causaltrace A Neurosymbolic Causal Analysis Agent For Smart Manufacturing](llm_agent/causaltrace_a_neurosymbolic_causal_analysis_agent_for_smart_manufacturing.md)**

:   提出 CausalTrace——一个集成于工业 CoPilot（SmartPilot）中的神经符号因果分析智能体，融合数据驱动因果发现与工业本体/知识图谱，实现了实时的根因分析、反事实推理和可解释决策支持。

**[Co-Epg A Framework For Co-Evolution Of Planning And Groundin](llm_agent/co-epg_a_framework_for_co-evolution_of_planning_and_groundin.md)**

:   提出Co-EPG框架，将GUI Agent解耦为Planning和Grounding两个模型，通过GRPO协同训练和基于置信度的动态奖励集成机制（C-DREM）建立正反馈循环，使两个模型自迭代协同进化，仅用基准数据集（无需外部数据）即在Multimodal-Mind2Web（58.4%）和AndroidControl（83.1%）上达到SOTA。

**[Coach Collaborative Agents For Contextual Highlighting -- A Multi-Agent Framewor](llm_agent/coach_collaborative_agents_for_contextual_highlighting_--_a_multi-agent_framewor.md)**

:   提出 COACH 框架——一个基于共享骨干模型的可重配置多智能体系统，通过意图驱动的策略编排和结构化 CoT 微调实现角色专业化，在羽毛球视频分析的 QA 和摘要两个任务上显著超越 Gemini 2.5 Pro 等通才模型。

**[Cook And Clean Together Teaching Embodied Agents For Paralle](llm_agent/cook_and_clean_together_teaching_embodied_agents_for_paralle.md)**

:   提出ORS3D任务——将运筹学(OR)知识引入具身AI的任务调度，要求智能体利用可并行子任务的等待时间执行其他任务以最小化总完成时间，同时在3D场景中定位目标物体；构建60K级数据集ORS3D-60K，并提出GRANT模型通过调度token机制连接外部动态规划求解器，在时间效率上比baseline提升30.53%。

**[Covrcollaborative Optimization Of Vlms And Rl Agent For Visu](llm_agent/covrcollaborative_optimization_of_vlms_and_rl_agent_for_visu.md)**

:   提出 VLM 与 RL 双向协同优化框架 COVR：RL 生成的高质量交互数据用于微调 VLM，增强后的 VLM 反过来通过 action prior 指导 RL 策略学习，在 CARLA 和 DMControl 上取得 SOTA。

**[D-Gara A Dynamic Benchmarking Framework For Gui Agent Robust](llm_agent/d-gara_a_dynamic_benchmarking_framework_for_gui_agent_robust.md)**

:   提出 D-GARA，一个面向 Android GUI Agent 的动态鲁棒性评估框架，通过在实时交互过程中注入权限弹窗、电量警告、应用崩溃等真实世界异常，揭示现有 SOTA Agent（包括 UI-TARS-72B、GPT-4o）在中断场景下平均成功率下降超过 17.5%，最高达 33% 的严重脆弱性。

**[Depo Dual-Efficiency Preference Optimization For Llm Agents](llm_agent/depo_dual-efficiency_preference_optimization_for_llm_agents.md)**

:   提出双重效率（dual-efficiency）的概念，将 LLM Agent 的效率分解为 step 级（减少每步 token 数）和 trajectory 级（减少总步数），并基于 KTO 设计了 DEPO 方法，通过在 desirable 样本的 reward 中加入效率 bonus 来联合优化效率与性能。

**[Ecoagent An Efficient Device-Cloud Collaborative Multi-Agent](llm_agent/ecoagent_an_efficient_device-cloud_collaborative_multi-agent.md)**

:   提出 EcoAgent，一个闭环设备-云端协作的多 Agent 移动自动化框架，通过 Dual-ReACT 双层推理规划 + 设备端轻量验证反馈 + Pre-Understanding 文本压缩模块，在 AndroidWorld 上达到与全云端 Agent 相当的成功率，同时大幅降低延迟（3.9s vs 15.3s）、云端调用（降89%）和上行数据量（降48.6倍）。

**[Finrpt Dataset Evaluation System And Llm-Based Multi-Agent Framework For Equity ](llm_agent/finrpt_dataset_evaluation_system_and_llm-based_multi-agent_framework_for_equity_.md)**

:   首次系统化地定义股票研究报告（ERR）自动生成任务——构建 FinRpt 数据集（6,825篇中英文高质量研报，整合7类金融数据），提出11指标评估体系和9 Agent协作的FinRpt-Gen生成框架（含评级修正/专家审查/润色三阶段增强），人类评估显示生成报告质量接近专家撰写。

**[From Biased Chatbots To Biased Agents Examining Role Assignment Effects On Llm A](llm_agent/from_biased_chatbots_to_biased_agents_examining_role_assignment_effects_on_llm_a.md)**

:   首个系统性案例研究，揭示基于人口统计学的 persona 分配会导致 LLM Agent 在 5 个操作领域的任务执行中出现最高 26.2% 的性能下降，证明 persona 诱导的偏见从文本生成延伸到了行动决策层面。

**[History-Aware Reasoning For Gui Agents](llm_agent/history-aware_reasoning_for_gui_agents.md)**

:   提出 HAR 框架，通过构建反思学习场景、合成纠错指南、设计混合 RL 奖励函数（含 Memory-Augmented Reward），将 GUI Agent 的推理模式从"历史无感知"转变为"历史感知"，3B 模型在 AITW/Mind2Web/GUI-Odyssey 等多个 benchmark 上超越更大模型。

**[Liecraft A Multi-Agent Framework For Evaluating Deceptive Capabilities In Langua](llm_agent/liecraft_a_multi-agent_framework_for_evaluating_deceptive_capabilities_in_langua.md)**

:   设计LieCraft多人隐藏角色博弈框架（约束满足问题确保平衡），评估12个LLM的战略欺骗能力，发现所有测试的前沿LLM（含GPT-4）在激励下都展现90%+的欺骗率——安全训练未消除策略性撒谎能力。

**[Llandmark A Multi-Agent Framework For Landmark-Aware Multimodal Interactive Vide](llm_agent/llandmark_a_multi-agent_framework_for_landmark-aware_multimodal_interactive_vide.md)**

:   提出 LLandMark 模块化多 Agent 框架，通过地标知识增强、LLM 辅助图像检索和 OCR 精炼模块，在越南大规模视频检索挑战赛（HCMAIC 2025）中实现地标感知的多模态交互式视频检索，总分 77.40/88。

**[Llmtm Benchmarking And Optimizing Llms For Temporal Motif Analysis In Dynamic Gr](llm_agent/llmtm_benchmarking_and_optimizing_llms_for_temporal_motif_analysis_in_dynamic_gr.md)**

:   提出 LLMTM——首个评估 LLM 处理动态图中时序 motif 分析能力的综合基准，包含 6 类任务覆盖 9 种时序 motif 类型，评估 9 个模型后发现 LLM 对时序 motif 的识别能力随 motif 复杂度快速下降。提出结构感知分派器（Structure-Aware Dispatcher），根据图的结构属性和认知负荷智能路由查询到标准 LLM 提示或工具增强 Agent，在维持高准确率的同时降低计算成本。

**[Loss-Guided Auxiliary Agents For Overcoming Mode Collapse In Gflownets](llm_agent/loss-guided_auxiliary_agents_for_overcoming_mode_collapse_in_gflownets.md)**

:   提出 LGGFN（Loss-Guided GFlowNets），用辅助 GFlowNet 的探索直接由主 GFlowNet 的训练损失驱动——辅助 Agent 的奖励 = 原始奖励 + λ·主模型损失，优先采样主模型理解不足的区域，在网格/序列/贝叶斯结构学习任务上分别发现 40× 更多唯一模式、99% 探索误差降低。

**[Medla A Logic-Driven Multi-Agent Framework For Complex Medic](llm_agent/medla_a_logic-driven_multi-agent_framework_for_complex_medic.md)**

:   提出 MedLA，首个基于三段论逻辑树的医学多 Agent 推理框架：每个 Agent 将推理组织为显式的逻辑树（大前提-小前提-结论三段论节点），多个 Agent 通过图引导的多轮讨论在前提级别对齐和修正逻辑树，在 MedDDx 上超越所有基线 7.4%（8B 模型），在医学 QA 上以 8B 模型达到 69.9% 平均准确率（超 70B RAG 模型）。

**[Moralreason Generalizable Moral Decision Alignment For Llm Agents Using Reasonin](llm_agent/moralreason_generalizable_moral_decision_alignment_for_llm_agents_using_reasonin.md)**

:   使用Group Relative Policy Optimization (GRPO)在推理层面训练LLM进行道德框架对齐，在Moral-Reason-QA数据集（680个高歧义场景）上实现功利主义对齐分数从0.207提升到0.964的分布外泛化。

**[Parallelism Meets Adaptiveness Scalable Documents Understanding In Multi-Agent L](llm_agent/parallelism_meets_adaptiveness_scalable_documents_understanding_in_multi-agent_l.md)**

:   提出自适应协调的多 Agent LLM 框架，通过并行竞争评估、动态任务路由和双向反馈机制，在高复杂度金融文档分析任务中实现 27% 的合规准确率提升和 74% 的修订率降低。

**[Pertouch Vlm-Driven Agent For Personalized And Semantic Image Retouching](llm_agent/pertouch_vlm-driven_agent_for_personalized_and_semantic_image_retouching.md)**

:   提出 PerTouch 框架，结合基于 Stable Diffusion + ControlNet 的语义区域级修图模型和 VLM 驱动的 Agent（含反馈重思考机制和场景感知记忆），实现精细化、个性化的图像修图。

**[Physics-Informed Autonomous Llm Agents For Explainable Power Electronics Modulat](llm_agent/physics-informed_autonomous_llm_agents_for_explainable_power_electronics_modulat.md)**

:   提出PHIA系统：LLM规划器通过聊天接口收集设计需求，协调物理信息神经网络代理模型（层次化PINN）和优化算法自主迭代生成电力转换器调制设计方案，MAE降低63.2%、设计速度提升33倍、20位专家验证可用性。

**[Probench Benchmarking Gui Agents With Accurate Process Infor](llm_agent/probench_benchmarking_gui_agents_with_accurate_process_infor.md)**

:   提出 ProBench，首个同时评估"最终状态"和"操作过程"的移动端 GUI Agent benchmark：200+ 挑战性任务覆盖 34 个中英文主流 App，通过 Process Provider（Structure Description Converter + MLLM Summarizer）自动捕获精确的中间过程信息，评估发现最强模型 Gemini 2.5 Pro 也仅完成 40.1% 任务，暴露了 grounding 不足、历史操作感知差、任务规划过于简化三大普遍问题。

**[Promoting Sustainable Web Agents Benchmarking And Estimating Energy Consumption ](llm_agent/promoting_sustainable_web_agents_benchmarking_and_estimating_energy_consumption_.md)**

:   首次系统性地从实证基准测试和理论估算两个角度量化了 Web Agent 的能耗与碳排放，发现更高能耗并不等于更好性能，并倡导在评测中引入能效指标。

**[Prune4Web Dom Tree Pruning Programming For Web Agent](llm_agent/prune4web_dom_tree_pruning_programming_for_web_agent.md)**

:   提出 Prune4Web，通过"LLM 生成评分函数参数 + 固定启发式模板执行"的编程式 DOM 剪枝方法实现 25-50 倍候选元素缩减：三阶段 pipeline（Planner 分解子任务 → Programmatic Filter 生成评分函数剪枝 DOM → Grounder 执行操作），3B 模型在 Multimodal-Mind2Web 上达到 52.4% Step SR（超越所有同参数量基线甚至部分 9.6B/32B 模型），低级 grounding 准确率从 46.8% 提升至 88.28%。

**[Real-Time Trust Verification For Safe Agentic Actions Using Trustbench](llm_agent/real-time_trust_verification_for_safe_agentic_actions_using_trustbench.md)**

:   提出TrustBench双模式框架：(1) 基准模式——结合传统指标和LLM-as-a-Judge评估8个信任维度，学习Agent置信度与实际正确率的校准映射；(2) 验证模式——在Agent制定行动后、执行前实时计算信任分数，阻止87%的有害行动，延迟低于200ms，通过领域插件（医疗/金融/QA）实现专业化验证。

**[Reflection-Driven Control For Trustworthy Code Agents](llm_agent/reflection-driven_control_for_trustworthy_code_agents.md)**

:   提出 Reflection-Driven Control 模块，将"自我反思"从事后补丁提升为 Agent 推理过程中的一等控制回路，通过轻量自检、证据驱动修复和反思记忆库三个组件，在安全代码生成任务上显著提升代码安全率。

**[Some A Realistic Benchmark For Llm-Based Social Media Agents](llm_agent/some_a_realistic_benchmark_for_llm-based_social_media_agents.md)**

:   提出首个面向社交媒体智能体的综合性评测基准 SoMe，包含 8 项任务、900 万+真实帖子和 17,869 条标注查询，评估 13 个主流 LLM 的社交媒体代理能力，揭示现有模型在复杂社交任务上仍有较大差距。

**[Structured Personalization Modeling Constraints As Matroids For Data-Minimal Llm](llm_agent/structured_personalization_modeling_constraints_as_matroids_for_data-minimal_llm.md)**

:   将 LLM Agent 个性化中的结构化约束（逻辑依赖 + 层级配额）形式化为层叠拟阵（laminar matroid），证明贪心算法在此约束下仍具有常数因子近似保证，解决了有依赖关系和层级限制的数据最小化选择问题。

**[Thucy An Llm-Based Multi-Agent System For Claim Verification Across Relational D](llm_agent/thucy_an_llm-based_multi-agent_system_for_claim_verification_across_relational_d.md)**

:   提出首个跨数据库、跨表的多 Agent 声明验证系统 Thucy，由 Verifier 领导三个专家 Agent（Data/Schema/SQL Expert），对数据源完全无先验知识，能自主发现、推理并生成 SQL 证据，在 TabFact 上超越 SOTA 5.6 个百分点（94.3%）。

**[Tongui Internet-Scale Trajectories From Multimodal Web Tutor](llm_agent/tongui_internet-scale_trajectories_from_multimodal_web_tutor.md)**

:   TongUI 提出从互联网上的多模态教程（视频+图文）自动转化为 GUI 操作轨迹数据的框架，构建了百万级的 GUI-Net-1M 数据集，用于微调 Qwen2.5-VL 模型，在多个 grounding 和 navigation 基准上超越或接近 UI-TARS 等 SOTA。

**[Towards Trustworthy Multi-Turn Llm Agents Via Behavioral Guidance](llm_agent/towards_trustworthy_multi-turn_llm_agents_via_behavioral_guidance.md)**

:   提出任务完成框架，通过任务分析器（Task Profiler）、推理模块（Reasoning Module）和生成模块（Generation Module）三组件协同进化，使 LLM Agent 在多轮交互环境中实现可验证和可靠的行为引导。

**[When Refusals Fail Unstable Safety Mechanisms In Long-Context Llm Agents](llm_agent/when_refusals_fail_unstable_safety_mechanisms_in_long-context_llm_agents.md)**

:   系统研究 LLM Agent 在长上下文填充下的安全行为变化：发现声称支持 1M-2M token 的模型在 100K token 时已出现 >50% 的性能崩溃，拒绝率以不可预测的方式波动（GPT-4.1-nano 从 5% 升至 40%，Grok 4 Fast 从 80% 降至 10%），揭示了长上下文 Agent 系统的严重安全隐患。

**[With Great Capabilities Come Great Responsibilities Introducing The Agentic Risk](llm_agent/with_great_capabilities_come_great_responsibilities_introducing_the_agentic_risk.md)**

:   提出 Agentic Risk & Capability (ARC) 框架，从能力（Capability）视角系统化地识别、评估和缓解智能体 AI 系统的安全与安全风险，为组织级治理提供可操作的结构化方法论。

---

## 🎯 目标检测 { #object_detection }

**[A Theoretical Analysis Of Detecting Large Model-Generated Time Series](object_detection/a_theoretical_analysis_of_detecting_large_model-generated_time_series.md)**

:   首次提出时间序列大模型（TSLM）生成内容检测理论框架，通过收缩假说（Contraction Hypothesis）揭示TSLM生成序列在递归预测下不确定性指数级衰减的本质特征，据此设计UCE检测器，在32个数据集上In-Distribution AUROC达0.855，显著超越10种文本检测baseline。

**[Actor-Critic For Continuous Action Chunks A Reinforcement Le](object_detection/actor-critic_for_continuous_action_chunks_a_reinforcement_le.md)**

:   AC3 提出了一个直接学习连续动作序列（action chunk）的 actor-critic 框架，通过"仅从成功轨迹更新 actor"的非对称更新规则和基于自监督锚点的内在奖励来稳定稀疏奖励下的长时域机器人操作学习，在 BiGym 和 RLBench 的 25 个任务上取得优于现有方法的成功率。

**[Aerialmind Towards Referring Multi-Object Tracking In Uav Sc](object_detection/aerialmind_towards_referring_multi-object_tracking_in_uav_sc.md)**

:   构建了首个面向无人机场景的大规模 Referring Multi-Object Tracking（RMOT）基准数据集 AerialMind，并提出 HawkEyeTrack（HETrack）方法，通过视觉-语言共进化融合编码器和尺度自适应上下文精炼模块，在无人机航拍场景中实现语言引导的多目标跟踪。

**[An Overall Real-Time Mechanism For Classification And Quality Evaluation Of Rice](object_detection/an_overall_real-time_mechanism_for_classification_and_quality_evaluation_of_rice.md)**

:   提出一个实时大米品质评估整体机制，整合改进的 YOLO-v5（品种检测）、改进的 ConvNeXt-Tiny（完整度分级）和 K-means（垩白区域量化）三个模块，在自建的六品种两万张图像数据集上实现了 99.14% mAP 和 97.89% 检测准确率。

**[Beyond Boundaries Leveraging Vision Foundation Models For So](object_detection/beyond_boundaries_leveraging_vision_foundation_models_for_so.md)**

:   提出利用VFM（DINOv2+Grounding DINO）增强无源域自适应目标检测（SFOD）的框架，通过全局特征对齐(PGFA)、实例级原型对比学习(PIFA)和双源伪标签融合(DEPF)三个模块，在6个跨域检测基准上取得SOTA，例如Cityscapes→Foggy Cityscapes达47.1% mAP（比DRU高3.5%），Sim10k→Cityscapes达67.4% AP（比DRU高8.7%）。

**[Beyond Fact Retrieval Episodic Memory For Rag With Generative Semantic Workspace](object_detection/beyond_fact_retrieval_episodic_memory_for_rag_with_generative_semantic_workspace.md)**

:   提出 Generative Semantic Workspace (GSW)，一种神经科学启发的生成式记忆框架，为 LLM 构建结构化的情景记忆表示，在 EpBench 上 F1 达到 0.85，同时减少 51% 的查询时上下文 token。

**[Beyond Semantic Features Pixel-Level Mapping For Generalized Ai-Generated Image ](object_detection/beyond_semantic_features_pixel-level_mapping_for_generalized_ai-generated_image_.md)**

:   提出像素级映射（pixel-level mapping）预处理方法，通过打破像素值的单调排列来抑制低频语义偏差、增强高频生成伪影，将 AI 生成图像检测的跨模型泛化准确率提升至 98.4%。

**[Connecting The Dots Training-Free Visual Grounding Via Agent](object_detection/connecting_the_dots_training-free_visual_grounding_via_agent.md)**

:   提出 GroundingAgent，一个完全不需要任务特定微调的视觉定位框架，通过组合预训练的开放词汇检测器（YOLO World）、MLLM（Llama-3.2-11B-Vision）和 LLM（DeepSeek-V3）进行结构化迭代推理，在 RefCOCO/+/g 上实现 65.1% 的零样本平均准确率，大幅超越之前的 zero-shot 方法。

**[Continuous Vision-Language-Action Co-Learning With Semantic-](object_detection/continuous_vision-language-action_co-learning_with_semantic-.md)**

:   提出CCoL框架，通过NeuralODE驱动的多模态连续协同学习（MCC）和双向交叉注意力的语义-物理对齐（CSA），在Behavioral Cloning中同时解决动作序列的物理不连续性和语义-物理失配问题，在三个仿真平台上平均相对提升8.0%，双臂插入任务最高达19.2%。

**[Ctpd Cross Tokenizer Preference Distillation](object_detection/ctpd_cross_tokenizer_preference_distillation.md)**

:   提出 Cross-Tokenizer Preference Distillation (CTPD)，首个支持不同分词器间偏好蒸馏的统一框架，通过 Aligned Span Projection、跨分词器重要性加权和 Teacher-Anchored Reference 三项创新，在多个 benchmark 上显著超越现有方法。

**[Deep Incomplete Multi-View Clustering Via Hierarchical Imputation And Alignment](object_detection/deep_incomplete_multi-view_clustering_via_hierarchical_imputation_and_alignment.md)**

:   提出 DIMVC-HIA，一个集成层次化填充与双重对齐的深度不完整多视图聚类框架，先填充缺失聚类分配再填充缺失特征，在高缺失率（70%）下仍保持稳健性能。

**[Generating Sketches In A Hierarchical Auto-Regressive Proces](object_detection/generating_sketches_in_a_hierarchical_auto-regressive_proces.md)**

:   提出 Sketch-HARP 分层自回归草图生成框架，通过三阶段层次化过程（预测笔画嵌入→确定画布位置→生成绘制动作序列），首次实现草图绘制过程中的灵活笔画级操控，在替换/擦除/扩展等任务上显著优于 SketchEdit。

**[Ground What You See Hallucination-Resistant Mllms Via Caption Feedback Diversity](object_detection/ground_what_you_see_hallucination-resistant_mllms_via_caption_feedback_diversity.md)**

:   针对多模态大模型（MLLM）在强化学习训练中产生幻觉的三大根因——视觉误解、探索多样性不足、样本冲突——分别提出 Caption Reward、奖励方差引导的样本选择、以及基于 NTK 相似度的 InfoNCE 正则化，在多个基准上显著降低幻觉率。

**[H-Gar A Hierarchical Interaction Framework Via Goal-Driven Observation-Action Re](object_detection/h-gar_a_hierarchical_interaction_framework_via_goal-driven_observation-action_re.md)**

:   提出层次化目标驱动框架 H-GAR，通过先预测目标观测再合成中间观测、并利用历史动作记忆库细化粗粒度动作，实现了观测与动作的显式双向交互，在仿真和真实机器人操控任务上取得 SOTA。

**[How Many Experts Are Enough Towards Optimal Semantic Specialization For Mixture-](object_detection/how_many_experts_are_enough_towards_optimal_semantic_specialization_for_mixture-.md)**

:   提出MASS框架，通过基于梯度的语义漂移检测自适应扩展MoE专家池，并结合Top-p置信度路由策略，在无需超参搜索的情况下自动发现最优专家数量，同时增强专家间的语义分化。

**[Learning Procedural-Aware Video Representations Through State-Grounded Hierarchy](object_detection/learning_procedural-aware_video_representations_through_state-grounded_hierarchy.md)**

:   提出 Task-Step-State（TSS）三层语义框架，在传统的任务-步骤层次中引入"状态"作为视觉锚定层，并设计渐进式预训练策略（Task→Step→State→Step→Task）逐步展开 TSS 层次，在 COIN 和 CrossTask 数据集上的任务识别、步骤识别和步骤预测任务上全面超越 SOTA。

**[Leveraging Textual Compositional Reasoning For Robust Change Captioning](object_detection/leveraging_textual_compositional_reasoning_for_robust_change_captioning.md)**

:   提出 CORTEX 框架，通过引入 VLM 生成的组合推理文本作为显式线索，结合图像-文本双重对齐模块（ITDA），增强纯视觉变化描述方法对物体关系和空间配置等结构化语义的理解能力。

**[Lost In Translation A Comparative Study On The Cross-Lingual Transfer Of Composi](object_detection/lost_in_translation_a_comparative_study_on_the_cross-lingual_transfer_of_composi.md)**

:   提出 CompositeHarm 基准，通过将对抗语法攻击（AttaQ）和语境化危害（MMSafetyBench）翻译为五种印度语言，系统研究了 LLM 安全对齐在跨语言场景下的脆弱性，发现对抗语法攻击在印度语言中攻击成功率急剧攀升。

**[Mitigating Error Accumulation In Co-Speech Motion Generation Via Global Rotation](object_detection/mitigating_error_accumulation_in_co-speech_motion_generation_via_global_rotation.md)**

:   提出 GlobalDiff 框架，首次在全局关节旋转空间中进行扩散生成，从根本上消除层次化前向运动学中的误差累积问题，并通过关节-骨骼-运动三层约束方案弥补全局表示丢失的结构先验，在多说话人语音驱动动作生成基准上取得 SOTA，FGD 较此前最佳方法改进 46%。

**[Pase Leveraging The Phonological Prior Of Wavlm For Low-Hallucination Generative](object_detection/pase_leveraging_the_phonological_prior_of_wavlm_for_low-hallucination_generative.md)**

:   提出 PASE 框架，通过去噪表示蒸馏（DRD）利用预训练 WavLM 中鲁棒的音韵先验来抑制语言幻觉，同时采用双流表示（高层音素 + 低层声学）消除声学幻觉，在感知质量和内容保真度两方面同时达到 SOTA。

**[Perceive Act And Correct Confidence Is Not Enough For Hyperspectral Classificati](object_detection/perceive_act_and_correct_confidence_is_not_enough_for_hyperspectral_classificati.md)**

:   提出 CABIN 框架，通过认知感知-行动-纠正的闭环学习机制，利用认识论不确定性（epistemic uncertainty）替代单纯的置信度来指导半监督高光谱图像分类中的样本选择与伪标签管理，在仅用 75% 标注的情况下显著超过全标注基线。

**[Resource Efficient Sleep Staging Via Multi-Level Masking And Prompt Learning](object_detection/resource_efficient_sleep_staging_via_multi-level_masking_and_prompt_learning.md)**

:   提出 MASS (Mask-Aware Sleep Staging) 框架，通过多层级 masking 策略和层次化 prompt learning 机制，仅用 **10% 的原始 EEG 信号**即可实现可靠的睡眠分期，为资源受限的可穿戴睡眠监测系统提供方案。

**[Rexo Indoor Multi-View Radar Object Detection Via 3D Bounding Box Diffusion](object_detection/rexo_indoor_multi-view_radar_object_detection_via_3d_bounding_box_diffusion.md)**

:   将 DiffusionDet 的 2D BBox 扩散范式提升到 3D 雷达空间，提出 REXO 框架：通过含噪 3D BBox 的投影引导显式跨视图雷达特征关联，并引入地面约束减少扩散参数，在 HIBER 和 MMVR 两个室内雷达数据集上分别超越 SOTA +4.22 AP 和 +11.02 AP。

**[Robust Long-Term Test-Time Adaptation For 3D Human Pose Estimation Through Motio](object_detection/robust_long-term_test-time_adaptation_for_3d_human_pose_estimation_through_motio.md)**

:   针对 3D 人体姿态估计在线测试时自适应中的误差累积问题，提出基于运动离散化（无监督聚类获得锚运动集）+ 自回放机制 + 软重置策略的解决方案，使模型能在长时间持续适应中稳健利用个人形态和习惯性运动特征，在 Ego-Exo4D 和 3DPW 上超越所有现有在线 TTA 方法。

**[Saga Learning Signal-Aligned Distributions For Improved Text-To-Image Generation](object_detection/saga_learning_signal-aligned_distributions_for_improved_text-to-image_generation.md)**

:   提出SAGA方法，通过学习与提示词对齐的高斯分布来改进文本到图像生成模型的语义对齐，无需重新训练且支持文本和空间双条件生成，在SD 1.4和SD 3上大幅提升对齐性能（TIAM-3从8.4%提升到50.7%）。

**[Semanticvla Semantic-Aligned Sparsification And Enhancement For Efficient Roboti](object_detection/semanticvla_semantic-aligned_sparsification_and_enhancement_for_efficient_roboti.md)**

:   提出 SemanticVLA 框架，通过语义引导的双视觉编码器剪枝（SD-Pruner）、语义互补层次融合（SH-Fuser）和语义条件动作耦合（SA-Coupler）三个模块，在大幅减少视觉冗余的同时增强指令-视觉-动作对齐，在 LIBERO 基准上以 97.7% 成功率超越 OpenVLA 达 21.1%，同时训练成本和推理延迟分别降低 3.0× 和 2.7×。

**[Simrod A Simple Baseline For Raw Object Detection With Global And Local Enhancem](object_detection/simrod_a_simple_baseline_for_raw_object_detection_with_global_and_local_enhancem.md)**

:   提出SimROD，一种极其轻量（仅0.003M参数）的RAW图像目标检测方法，通过全局Gamma增强（4个可学习参数）和绿色通道引导的局部增强，在多个RAW检测基准上超越了复杂的SOTA方法。

**[Sm3Det A Unified Model For Multi-Modal Remote Sensing Object Detection](object_detection/sm3det_a_unified_model_for_multi-modal_remote_sensing_object_detection.md)**

:   SM3Det提出了遥感领域的M2Det新任务（多模态数据集+多任务目标检测），通过网格级稀疏MoE骨干网络和动态子模块优化（DSO）机制，用单一模型同时处理SAR/光学/红外三种模态的水平/旋转框检测，显著超越各模态独立训练的三个专用模型组合。

**[T-Rex-Omni Integrating Negative Visual Prompt In Generic Object Detection](object_detection/t-rex-omni_integrating_negative_visual_prompt_in_generic_object_detection.md)**

:   提出T-Rex-Omni框架，首次将负视觉提示（negative visual prompts）系统性地引入开放集目标检测，通过训练免费的NNC模块和NNH损失，显著缩小了视觉提示和文本提示检测方法之间的性能差距，在长尾场景中表现尤为突出（LVIS-minival APr达到51.2）。

**[Tackling Resource-Constrained And Data-Heterogeneity In Federated Learning With ](object_detection/tackling_resource-constrained_and_data-heterogeneity_in_federated_learning_with_.md)**

:   提出FedCSPACK，一种基于余弦稀疏化参数打包和双权重聚合的个性化联邦学习方法，通过在包级别进行参数选择和共享，同时平衡了数据异质性和客户端资源约束，训练速度提升2-5倍、通信量压缩高达96%，同时模型精度提升3.34%。

**[Temporal Object-Aware Vision Transformer For Few-Shot Video Object Detection](object_detection/temporal_object-aware_vision_transformer_for_few-shot_video_object_detection.md)**

:   提出一种对象感知的时序建模框架，通过选择性传播高置信度检测特征实现跨帧时序一致性，结合预训练视觉-语言编码器（OWL-ViT）和少样本检测头，在四个视频少样本检测基准上平均提升3.7%-5.3% AP。

**[Test-Time Diverse Reasoning By Riemannian Activation Steering](object_detection/test-time_diverse_reasoning_by_riemannian_activation_steering.md)**

:   提出 SPREAD 框架——一种无监督的测试时激活引导策略，通过在球面流形乘积上求解黎曼优化问题来最大化多条推理路径的隐藏激活张成的总体积，从而提升 Best-of-N 采样中的推理多样性和准确率，在数学推理基准上超越温度采样基线。

**[Towards Effective Stealthy And Persistent Backdoor Attacks Targeting Graph Found](object_detection/towards_effective_stealthy_and_persistent_backdoor_attacks_targeting_graph_found.md)**

:   提出 GFM-BA，首个系统性地针对 Graph Foundation Models (GFMs) 预训练阶段的后门攻击方法，通过 label-free trigger 关联、node-adaptive trigger 生成和 persistent backdoor anchoring 三个模块，同时解决有效性、隐蔽性和持久性三大挑战。

**[Towards Scalable Web Accessibility Audit With Mllms As Copilots](object_detection/towards_scalable_web_accessibility_audit_with_mllms_as_copilots.md)**

:   提出 AAA 框架，通过 GRASP（基于图的多模态页面采样）和 MaC（MLLM 作为 Copilot）两大创新，将 WCAG-EM 标准操作化，实现可扩展的端到端网页无障碍审计。

**[Trace A Generalizable Drift Detector For Streaming Data-Driven Optimization](object_detection/trace_a_generalizable_drift_detector_for_streaming_data-driven_optimization.md)**

:   提出TRACE，一种基于注意力序列学习的可迁移概念漂移检测器，通过统计特征标记化和双注意力编码器学习跨任务可迁移的漂移模式，能泛化到未见过的数据集，并作为即插即用模块嵌入流式数据驱动优化算法。

**[Ttf-Vla Temporal Token Fusion Via Pixel-Attention Integratio](object_detection/ttf-vla_temporal_token_fusion_via_pixel-attention_integratio.md)**

:   TTF-VLA 提出了一种免训练的时序 Token 融合方法，通过灰度像素差异+注意力语义检测的双维度机制选择性地复用历史帧的视觉 Token，提升 VLA 模型在机器人操作任务中的推理质量，在 LIBERO 上平均提升 4.0 个百分点。

**[Tubermc Tube-Conditioned Reconstruction With Mutual Constraints For Weakly-Super](object_detection/tubermc_tube-conditioned_reconstruction_with_mutual_constraints_for_weakly-super.md)**

:   提出 TubeRMC 框架，利用文本条件化的候选 tube 生成 + 从时间/空间/时空三个维度进行 tube 条件化重建，并引入空间-时间互约束来增强弱监督时空视频定位性能。

**[Vk-Det Visual Knowledge Guided Prototype Learning For Open-Vocabulary Aerial Obj](object_detection/vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)**

:   提出 VK-Det 框架，仅利用 VLM 的视觉知识（无需额外监督信号），通过自适应选择知识蒸馏（ASKD）+ 原型感知伪标签（PAPL）+ 综合匹配推理（SMI），在航空遥感开放词汇目标检测中达到 SOTA，甚至超越使用额外监督的方法。

**[When Hallucination Costs Millions Benchmarking Ai Agents In High-Stakes Adversar](object_detection/when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)**

:   提出 CAIA 基准测试，通过加密货币市场作为天然对抗性实验室，评估 17 个 SOTA 大模型在高风险对抗环境中的 agent 能力，揭示前沿模型仅达 67.4% 准确率（GPT-5）vs 人类 80%，并发现系统性工具选择灾难。

**[When Trackers Date Fish A Benchmark And Framework For Underwater Multiple Fish T](object_detection/when_trackers_date_fish_a_benchmark_and_framework_for_underwater_multiple_fish_t.md)**

:   提出 MFT25 大规模水下多鱼跟踪数据集（15 序列, 408K 标注）和 SU-T 跟踪框架（UKF + FishIoU），实现 34.1 HOTA 和 44.6 IDF1 的 SOTA 性能，并通过统计分析揭示鱼类跟踪与陆地目标跟踪的本质差异。

**[Your Ai-Generated Image Detector Can Secretly Achieve Sota Accuracy If Calibrate](object_detection/your_ai-generated_image_detector_can_secretly_achieve_sota_accuracy_if_calibrate.md)**

:   提出一种基于贝叶斯决策理论的轻量级后验校准方法，通过在模型输出logit上添加可学习标量偏移α，无需重训练即可显著提升现有AI生成图像检测器在分布偏移下的准确率。

---

## 🕸️ 图学习 { #graph_learning }

**[Adaptive Initial Residual Connections For Gnns With Theoretical Guarantees](graph_learning/adaptive_initial_residual_connections_for_gnns_with_theoretical_guarantees.md)**

:   提出自适应初始残差连接（Adaptive IRC），允许每个节点拥有基于初始特征学习的个性化残差强度，首次证明带激活函数的初始残差连接的 Dirichlet 能量有正下界（保证不过平滑），并提出基于 PageRank 的启发式变体在避免学习额外参数的同时达到可比甚至更优性能。

**[Adaptive Riemannian Graph Neural Networks](graph_learning/adaptive_riemannian_graph_neural_networks.md)**

:   提出 ARGNN 框架，为图上每个节点学习一个连续的、各向异性的对角黎曼度量张量，从而自适应地捕获图中不同区域（层级结构 vs 密集社区）的局部几何特性，统一并超越了固定曲率和离散混合曲率的几何 GNN 方法。

**[Are Graph Transformers Necessary Efficient Long-Range Messag](graph_learning/are_graph_transformers_necessary_efficient_long-range_messag.md)**

:   提出分形节点（Fractal Nodes）增强 MPNN 的长距离消息传递：通过 METIS 图划分生成子图级聚合节点，结合低通+高通滤波器（LPF+HPF）与可学习频率参数 $\omega$，使用 MLP-Mixer 实跨子图通信，在保持 $O(L(|V|+|E|))$ 线性复杂度的同时达到甚至超越图 Transformer 的性能，获 AAAI Oral。

**[Assemble Your Crew Automatic Multi-Agent Communication Topol](graph_learning/assemble_your_crew_automatic_multi-agent_communication_topol.md)**

:   提出 ARG-Designer，将多 Agent 系统的拓扑设计重新定义为条件自回归图生成任务，从零开始逐步生成 Agent 节点和通信边（而非从模板图剪枝），在6个基准上达到 SOTA（平均 92.78%），同时 Token 消耗比 G-Designer 降低约 50%，且支持无需重训练的角色扩展。

**[Assessing Llms For Serendipity Discovery In Knowledge Graphs A Case For Drug Rep](graph_learning/assessing_llms_for_serendipity_discovery_in_knowledge_graphs_a_case_for_drug_rep.md)**

:   提出 SerenQA 框架，首次形式化定义知识图谱问答中的"意外发现"(serendipity)任务，包含基于信息论的 RNS 度量、专家标注的药物重定位基准数据集和三阶段评估流水线，揭示当前 LLM 在检索任务上表现尚可但在意外发现探索上仍有巨大改进空间。

**[Beyond Fixed Depth Adaptive Graph Neural Networks For Node Classification Under ](graph_learning/beyond_fixed_depth_adaptive_graph_neural_networks_for_node_classification_under_.md)**

:   提出 AD-GNN，通过理论分析节点级别的同配/异配特性，为每个节点自适应分配不同的聚合深度，在统一框架中同时处理同配和异配图上的节点分类任务。

**[Bugsweeper Function-Level Detection Of Smart Contract Vulnerabilities Using Grap](graph_learning/bugsweeper_function-level_detection_of_smart_contract_vulnerabilities_using_grap.md)**

:   提出 BugSweeper，通过构建函数级抽象语法图 (FLAG) 并设计两阶段 GNN 架构，实现无需专家规则的端到端智能合约漏洞检测，在重入攻击检测上 F1 达 98.57%。

**[Commonality In Few Few-Shot Multimodal Anomaly Detection Via Hypergraph-Enhanced](graph_learning/commonality_in_few_few-shot_multimodal_anomaly_detection_via_hypergraph-enhanced.md)**

:   提出 CIF，利用超图（hypergraph）提取少量训练样本的类内结构共性，指导 memory bank 的构建与搜索，在少样本多模态工业异常检测中取得 SOTA。

**[Connectivity-Guided Sparsification Of 2-Fwl Gnns Preserving Full Expressivity Wi](graph_learning/connectivity-guided_sparsification_of_2-fwl_gnns_preserving_full_expressivity_wi.md)**

:   Co-Sparsify 提出一种基于连通性感知的稀疏化框架，通过将 3-节点交互限制在双连通分量内、2-节点交互限制在连通分量内，消除可证明冗余的计算，在保持完整 2-FWL 表达力的同时显著提升效率，在合成子结构计数任务和 ZINC、QM9 等基准上取得 SOTA。

**[Echoless Label-Based Pre-Computation For Memory-Efficient Heterogeneous Graph Le](graph_learning/echoless_label-based_pre-computation_for_memory-efficient_heterogeneous_graph_le.md)**

:   Echoless-LP 通过分区聚焦的无回声传播（PFEP）消除标签预计算中多跳消息传递导致的训练标签泄露（回声效应），结合非对称分区方案（APS）和 PostAdjust 机制解决分区造成的信息损失和分布偏移，在保持内存高效的同时兼容任意消息传递方法，在多个异构图数据集上取得 SOTA 性能。

**[Enhancing Logical Expressiveness In Graph Neural Networks Via Path-Neighbor Aggr](graph_learning/enhancing_logical_expressiveness_in_graph_neural_networks_via_path-neighbor_aggr.md)**

:   PN-GNN 提出在条件消息传递的基础上聚合推理路径上的邻居节点嵌入，以即插即用的方式增强 GNN 的逻辑规则表达力（严格超越 C-GNN），同时避免标注技巧（labeling trick）对泛化能力的损害，在合成数据集和真实知识图谱推理任务上均取得提升。

**[Feature-Centric Unsupervised Node Representation Learning Without Homophily Assu](graph_learning/feature-centric_unsupervised_node_representation_learning_without_homophily_assu.md)**

:   提出 FUEL 方法，通过以节点特征为中心的聚类方案自适应学习图卷积的使用程度，无需同配性假设即可在同配和非同配图上均获得高质量的无监督节点表示。

**[Format As A Prior Quantifying And Analyzing Bias In Llms For Heterogeneous Data](graph_learning/format_as_a_prior_quantifying_and_analyzing_bias_in_llms_for_heterogeneous_data.md)**

:   首次系统研究 LLM 在处理异构格式数据（文本/表格/信息框/知识图谱）时的格式偏差问题，通过三阶段实验揭示偏差的存在性、数据层面驱动因素和注意力机制层面的内部成因，并验证了注意力重平衡干预的有效性。

**[Gcl-Ot Graph Contrastive Learning With Optimal Transport For Heterophilic Text-A](graph_learning/gcl-ot_graph_contrastive_learning_with_optimal_transport_for_heterophilic_text-a.md)**

:   提出 GCL-OT 框架，首次将最优传输（OT）引入异质性文本属性图的图对比学习中，通过 RealSoftMax 相似度估计、滤波提示机制和 OT 引导的潜在同质性挖掘三个模块，分别应对部分异质性、完全异质性和潜在同质性三种多粒度异质性挑战。

**[Gsap-Ere Fine-Grained Scholarly Entity And Relation Extraction Focused On Machin](graph_learning/gsap-ere_fine-grained_scholarly_entity_and_relation_extraction_focused_on_machin.md)**

:   提出GSAP-ERE——一个面向机器学习领域的细粒度学术实体与关系抽取数据集，包含10种实体类型和18种关系类型，在100篇全文论文上标注了63K实体和35K关系，实验表明微调模型（NER: 80.6%, RE: 54.0%）大幅超越LLM提示方法（NER: 44.4%, RE: 10.1%）。

**[Gt-Snt A Linear-Time Transformer For Large-Scale Graphs Via Spiking Node Tokeniz](graph_learning/gt-snt_a_linear-time_transformer_for_large-scale_graphs_via_spiking_node_tokeniz.md)**

:   提出 GT-SNT，将脉冲神经网络（SNN）用作图节点分词器（tokenizer），通过多步特征传播生成紧凑的脉冲计数嵌入作为节点 token，再利用码本引导自注意力（CGSA）在线性时间内捕获全局上下文，在 9 个节点分类基准上取得可比性能的同时实现最高 130× 的推理加速。

**[Human Cognition Inspired Rag With Knowledge Graph For Complex Problem Solving](graph_learning/human_cognition_inspired_rag_with_knowledge_graph_for_complex_problem_solving.md)**

:   提出 CogGRAG，一个受人类认知启发的基于知识图谱的 RAG 框架，通过自顶向下的思维导图分解问题、分层级结构化检索、以及双 LLM 自验证推理三个阶段，显著提升 LLM 在复杂知识图谱问答 (KGQA) 任务中的准确性和可靠性。

**[Hyperbolic Continuous Structural Entropy For Hierarchical Clustering](graph_learning/hyperbolic_continuous_structural_entropy_for_hierarchical_clustering.md)**

:   提出 HypCSE，将离散结构熵（SE）松弛为双曲空间中的连续结构熵（CSE），结合图结构学习和对比学习，实现端到端可微的层次聚类，在 7 个数据集上全面超越离散和连续层次聚类方法。

**[Logical Characterizations Of Gnns With Mean Aggregation](graph_learning/logical_characterizations_of_gnns_with_mean_aggregation.md)**

:   系统刻画了以均值（mean）为聚合函数的 GNN 的表达能力：非一致设定下等价于比率模态逻辑（RML）；一致设定下（相对 MSO）等价于模态逻辑（ML）；当额外要求组合函数连续、分类函数为阈值时，表达能力显著下降至交替无关模态逻辑（AFML）。

**[Magnitude-Modulated Equivariant Adapter For Parameter-Efficient Fine-Tuning Of E](graph_learning/magnitude-modulated_equivariant_adapter_for_parameter-efficient_fine-tuning_of_e.md)**

:   提出 MMEA (Magnitude-Modulated Equivariant Adapter)，一种用于球谐基等变 GNN 的轻量参数高效微调方法，通过标量门控按"阶-多重度"通道独立调制特征幅度，在严格保持等变性的前提下，以更少参数量实现了超越 ELoRA 和全参数微调的 SOTA 分子势能预测精度。

**[Motorec Sparse-Regularized Multimodal Tokenization For Cold-Start Recommendation](graph_learning/motorec_sparse-regularized_multimodal_tokenization_for_cold-start_recommendation.md)**

:   提出 MoToRec，将多模态推荐重新定义为离散语义分词任务，通过稀疏正则化的残差量化VAE（RQ-VAE）将原始多模态特征转化为可组合的离散语义编码，结合自适应稀有度放大和层级多源图编码器，有效解决物品冷启动问题。

**[Mug Meta-Path-Aware Universal Heterogeneous Graph Pre-Training](graph_learning/mug_meta-path-aware_universal_heterogeneous_graph_pre-training.md)**

:   首次提出无需 LLM 的通用异质图预训练方法 MUG，通过上下文结构编码统一异质节点/关系类型、维度感知编码器对齐不同图的表示空间，并利用元路径视图共享编码器 + 全局散射正则化实现跨域可迁移的编码与聚合，在跨域和小样本节点分类中显著超越已有方法。

**[Mygram Modality-Aware Graph Transformer With Global Distribution For Multi-Modal](graph_learning/mygram_modality-aware_graph_transformer_with_global_distribution_for_multi-modal.md)**

:   提出 MyGram，通过模态感知图卷积扩散（MGD）模块捕获模态内的深层结构上下文信息，并引入基于Gram矩阵行列式的全局分布对齐损失（Gram Loss），在高维空间中强制跨模态语义一致性，实现更鲁棒的多模态实体对齐。

**[Notam-Evolve A Knowledge-Guided Self-Evolving Optimization Framework With Llms F](graph_learning/notam-evolve_a_knowledge-guided_self-evolving_optimization_framework_with_llms_f.md)**

:   提出 NOTAM-Evolve，一个自进化框架，通过知识图谱增强的表格检索（KG-TableRAG）进行动态知识接地，结合迭代SFT+DPO偏好优化及多视角投票推理机制，使7B参数LLM自主掌握复杂航空NOTAM的深层解析，准确率较基础LLM提升30.4%。

**[Ntsformer A Self-Teaching Graph Transformer For Multimodal Isolated Cold-Start N](graph_learning/ntsformer_a_self-teaching_graph_transformer_for_multimodal_isolated_cold-start_n.md)**

:   提出 NTSFormer（Neighbor-to-Self Graph Transformer），一个统一的图Transformer框架，通过冷启动注意力掩码实现**自教学范式**——同一模型同时产生基于自身特征的"学生"预测和基于邻居信息的"教师"预测，无需退化为MLP即可处理多模态图上的孤立冷启动节点分类，结合MoE输入投影和多模态图预计算有效处理模态缺失问题。

**[On Stealing Graph Neural Network Models](graph_learning/on_stealing_graph_neural_network_models.md)**

:   证明了在严格查询限制下（如仅100次查询），攻击者可通过"本地获取encoder（随机初始化/SSL训练）+ K-means策略性查询选择"两阶段方法高效窃取GNN模型，在Physics数据集上仅用100次查询即达91%准确率，而现有SOTA需约5000次查询加额外embedding访问才能达到类似水平。

**[Pcokg Personality-Aware Commonsense Reasoning With Debate](graph_learning/pcokg_personality-aware_commonsense_reasoning_with_debate.md)**

:   构建了首个大规模人格感知常识知识图谱 PCoKG，包含521,316个四元组 $(e, p, r, t)$（事件-人格-推理维度-结果），通过LLM角色扮演+多智能体辩论机制生成高质量的人格差异化推理，实验验证了MBTI人格信息对常识推理和个性化对话生成的增强作用。

**[Posterior Label Smoothing For Node Classification](graph_learning/posterior_label_smoothing_for_node_classification.md)**

:   提出PosteL（Posterior Label Smoothing），通过贝叶斯后验分布从邻域标签中推导soft label用于节点分类，自然适应同质图和异质图，在8种backbone×10个数据集的80个组合中76个取得精度提升。

**[Relink Constructing Query-Driven Evidence Graph On-The-Fly For Graphrag](graph_learning/relink_constructing_query-driven_evidence_graph_on-the-fly_for_graphrag.md)**

:   提出从"先构建再推理"到"边推理边构建"的GraphRAG范式转变，通过Relink框架动态构建查询特定的证据图——结合高精度KG骨架和高召回潜在关系池，用查询驱动的排序器统一评估、按需补全缺失路径并过滤干扰事实——在5个多跳QA基准上平均提升EM 5.4%和F1 5.2%。

**[Rfkg-Cot Relation-Driven Adaptive Hop-Count Selection And Few-Shot Path Guidance](graph_learning/rfkg-cot_relation-driven_adaptive_hop-count_selection_and_few-shot_path_guidance.md)**

:   提出RFKG-CoT，通过关系驱动的自适应跳数选择（利用KG关系激活掩码动态调整推理步数）和Few-Shot路径引导（Question-Paths-Answer格式的in-context示例），在4个KGQA基准上显著提升LLM的知识图谱推理能力，GPT-4在WebQSP上达91.5%（+6.6pp），Llama2-7B提升幅度最大达+14.7pp。

**[S-Dag A Subject-Based Directed Acyclic Graph For Multi-Agent](graph_learning/s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)**

:   提出 S-DAG，通过 GNN 从问题中识别相关学科及其依赖关系构建有向无环图，将学科节点匹配到最擅长的专家 LLM（14 个 7-13B 领域模型），按 DAG 拓扑顺序协作推理（支撑学科→主导学科），用小模型池超越 GPT-4o-mini（59.73 vs 58.52）且接近 72B 模型。

**[Self-Adaptive Graph Mixture Of Models](graph_learning/self-adaptive_graph_mixture_of_models.md)**

:   提出 SAGMM（Self-Adaptive Graph Mixture of Models），一个利用架构多样性的图 MoE 框架，通过拓扑感知注意力门控（TAAG）自适应选择和组合异构 GNN 专家，配合自适应剪枝机制，在 16 个基准上覆盖节点分类、图分类、回归和链接预测，一致超越单一 GNN 和已有 MoE 方法。

**[Sentient Detecting Apts Via Capturing Indirect Dependencies And Behavioral Logic](graph_learning/sentient_detecting_apts_via_capturing_indirect_dependencies_and_behavioral_logic.md)**

:   提出 Sentient，一种结合**图 Transformer 预训练**和**双向 Mamba2 意图分析**的 APT 检测方法，仅用良性数据训练即可捕获间接依赖关系、去除场景噪声、关联行为逻辑，在三个标准数据集上平均降低 44% 误报率。

**[Sheaf Graph Neural Networks Via Pac-Bayes Spectral Optimization](graph_learning/sheaf_graph_neural_networks_via_pac-bayes_spectral_optimization.md)**

:   提出 SGPC（Sheaf GNNs with PAC-Bayes Calibration），结合 Wasserstein 最优传输学习 sheaf 限制映射、方差缩减扩散与自适应频率混合层、以及 PAC-Bayes 谱正则化，在同质和异质图节点分类上全面超越现有 GNN 和 sheaf 方法，同时提供理论泛化保证。

**[Spiking Heterogeneous Graph Attention Networks](graph_learning/spiking_heterogeneous_graph_attention_networks.md)**

:   提出 SpikingHAN，首次将**脉冲神经网络（SNN）**引入异构图学习，通过共享参数的单层图卷积聚合元路径邻居信息，语义级注意力融合多元路径，再编码为脉冲序列的 1-bit 二值表征，在三个数据集上以更少参数、更快推理和更低能耗达到有竞争力的分类性能。

**[Unihr Hierarchical Representation Learning For Unified Knowledge Graph Link Pred](graph_learning/unihr_hierarchical_representation_learning_for_unified_knowledge_graph_link_pred.md)**

:   提出UniHR框架，通过Hierarchical Data Representation (HiDR)将超关系/时序/嵌套等多类KG统一转换为三元组形式，并设计Hierarchical Structure Learning (HiSL)模块在事实内部和事实间进行两阶段消息传递，在9个数据集5种KG类型上取得最优或竞争性的link prediction结果。

---

## 🔬 可解释性 { #interpretability }

**[A Closer Look At Knowledge Distillation In Spiking Neural Ne](interpretability/a_closer_look_at_knowledge_distillation_in_spiking_neural_ne.md)**

:   针对ANN→SNN知识蒸馏中教师ANN连续特征/logits与学生SNN离散稀疏spike特征/logits之间分布差异被忽视的问题，提出基于显著性缩放激活图蒸馏（SAMD）和噪声平滑logits蒸馏（NLD）的CKDSNN框架，在CIFAR-10/100、ImageNet-1K和CIFAR10-DVS上均取得SNN训练的新SOTA。

**[A Coherence-Based Measure Of Agi](interpretability/a_coherence-based_measure_of_agi.md)**

:   指出现有 AGI 评分用算术平均隐含"可补偿"假设（强项弥补弱项），提出基于广义均值连续谱的一致性度量 $\text{AGI}_{\text{AUC}}$：在补偿性参数 $p \in [-1, 1]$ 上积分，惩罚能力不均衡，暴露被算术平均掩盖的瓶颈。

**[Adaptive Evidential Learning For Temporal-Semantic Robustnes](interpretability/adaptive_evidential_learning_for_temporal-semantic_robustnes.md)**

:   提出 DEMR 框架，将深度证据回归（DER）引入视频时刻检索任务，通过 Reflective Flipped Fusion 模块缓解模态不平衡、通过 Geom-regularizer 修复原始 DER 中不确定性估计的反直觉偏差，在标准和去偏数据集上均取得了显著提升。

**[Attention Gathers Mlps Compose A Causal Analysis Of An Action-Outcome Circuit In](interpretability/attention_gathers_mlps_compose_a_causal_analysis_of_an_action-outcome_circuit_in.md)**

:   通过机械可解释性方法逆向工程 Video Vision Transformer（ViViT）的内部电路，揭示注意力头负责"收集证据"、MLP 模块负责"组合概念"的分工机制，证明模型在简单分类任务中隐藏了超越训练目标的语义知识。

**[Beyond Hallucinations A Composite Score For Measuring Reliability In Open-Source](interpretability/beyond_hallucinations_a_composite_score_for_measuring_reliability_in_open-source.md)**

:   提出 Composite Reliability Score (CRS)，将校准度、鲁棒性和不确定性量化三个维度统一为单一可解释指标，对 10 个开源 LLM 在 5 个 QA 数据集上进行系统评估，发现 Mistral-8x22B 综合可靠性最高（CRS=0.81），而模型大小并不直接决定可靠性。

**[Concepts From Representations Post-Hoc Concept Bottleneck Models Via Sparse Deco](interpretability/concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)**

:   提出 PCBM-ReD，通过从预训练视觉编码器中自动提取概念、MLLM 标注/过滤、重建引导选择，再利用 CLIP 视觉-文本对齐将图像表示稀疏分解为概念嵌入的线性组合，构建事后概念瓶颈模型，在 11 个分类任务上达到 SOTA 精度且保持可解释性。

**[Crosscheck-Bench Diagnosing Compositional Failures In Multim](interpretability/crosscheck-bench_diagnosing_compositional_failures_in_multim.md)**

:   构建包含15k对抗性QA样本的三级层次基准CrossCheck-Bench，通过7种原子能力和15个任务诊断VLM在多模态冲突解决中的组合推理失败，揭示从感知(L1)到推理(L3)的系统性性能衰退以及传统提示策略的局限性。

**[Data Whitening Improves Sparse Autoencoder Learning](interpretability/data_whitening_improves_sparse_autoencoder_learning.md)**

:   本文将经典稀疏编码中的 PCA 白化（whitening）引入现代稀疏自编码器（SAE）训练，通过理论分析和仿真证明白化能使优化景观更凸更各向同性，在 SAEBench 上的实验表明白化显著提升可解释性指标（Sparse Probing +7.3%、SCR +54%、TPP +372%），尽管重构质量略有下降。

**[Distribution-Based Feature Attribution For Explaining The Predictions Of Any Cla](interpretability/distribution-based_feature_attribution_for_explaining_the_predictions_of_any_cla.md)**

:   提出首个基于数据分布的特征归因方法 DFAX，通过比较目标实例在目标类与非目标类条件概率之差来量化特征重要性，首次给出特征归因的形式化定义，在10个数据集上显著优于SHAP/LIME等基线且速度快数个数量级。

**[Drexperts Differential Refinement Of Distortion-Aware Experts For Blind Image Qu](interpretability/drexperts_differential_refinement_of_distortion-aware_experts_for_blind_image_qu.md)**

:   提出DR.Experts框架，利用DA-CLIP获取失真类型先验，通过差分精炼注意力机制（DSDM）将失真注意力与语义注意力分离以纯化失真特征，再通过动态失真加权模块（DDWM）按感知影响自适应加权各类失真特征，在5个BIQA基准上达到SOTA。

**[Elementarynet A Non-Strategic Neural Network For Predicting Human Behavior In No](interpretability/elementarynet_a_non-strategic_neural_network_for_predicting_human_behavior_in_no.md)**

:   提出 ElementaryNet，一种**可证明不具备策略性推理能力**的神经网络架构，用于建模博弈中人类的"level-0"（非策略性）行为，在预测准确率上与 GameNet（当前 SOTA）无统计差异，同时具备更好的可解释性。

**[Enhancing Binary Encoded Crime Linkage Analysis Using Siamese Network](interpretability/enhancing_binary_encoded_crime_linkage_analysis_using_siamese_network.md)**

:   提出基于 **Siamese Autoencoder** 的犯罪关联分析框架，通过 **decoder 阶段融合地理时间特征** 和 **领域专家驱动的数据降维策略**，在英国 NCA 的真实 ViCLAS 数据库上实现了 AUC 提升最高 9%，为高维稀疏二进制编码犯罪数据提供了有效的机器学习解决方案。

**[Explainable Melanoma Diagnosis With Contrastive Learning And Llm-Based Report Ge](interpretability/explainable_melanoma_diagnosis_with_contrastive_learning_and_llm-based_report_ge.md)**

:   提出 CEFM 框架，通过跨模态对比学习将 ViT 视觉特征与基于 ABCD 规则的临床特征（不对称性、边界、颜色）对齐，再由 CLIP + DeepSeek 生成结构化诊断报告，在 ISIC 数据集上达到 92.79% 准确率和 0.961 AUC，专家评分可解释性达 4.6/5。

**[Finding The Translation Switch Discovering And Exploiting The Task-Initiation Fe](interpretability/finding_the_translation_switch_discovering_and_exploiting_the_task-initiation_fe.md)**

:   利用稀疏自编码器（SAE）发现 LLM 中控制翻译任务启动的"翻译启动特征"，通过因果干预验证其功能（增强特征→提升翻译质量/减少幻觉，消除特征→产生幻觉），并将该机制洞察转化为实用的数据选择策略——优先在"机制困难"样本上微调，显著提升数据效率和抑制幻觉。

**[Finevau A Novel Human-Aligned Benchmark For Fine-Grained Video Anomaly Understan](interpretability/finevau_a_novel_human-aligned_benchmark_for_fine-grained_video_anomaly_understan.md)**

:   本文提出FineVAU基准，将视频异常理解 (VAU) 分解为事件(What)、实体(Who)、地点(Where)三个维度，设计了与人类感知高度对齐的FV-Score评估指标，并通过全自动LVLM辅助管线构建了FineW³数据集，实验揭示当前LVLM在细粒度异常事件感知上的关键短板。

**[Flashkat Understanding And Addressing Performance Bottlenecks In The Kolmogorov-](interpretability/flashkat_understanding_and_addressing_performance_bottlenecks_in_the_kolmogorov-.md)**

:   深入分析 KAT（Kolmogorov-Arnold Transformer）训练慢 123 倍的根因，发现瓶颈并非 FLOPs 而是反向传播中**梯度累积的内存停顿**（atomic add 导致全局内存竞争），提出 FlashKAT 通过重构 GPU 核函数将训练加速 **86.5 倍**并降低近一个数量级的梯度舍入误差。

**[Flexible Concept Bottleneck Model](interpretability/flexible_concept_bottleneck_model.md)**

:   本文提出Flexible Concept Bottleneck Model (FCBM)，通过引入超网络动态生成概念权重和可学习温度的sparsemax模块，实现了概念池的动态适配（包括完全替换），并在5个公开数据集上以相似的有效概念数达到了与SOTA基线可比的精度，仅需单个epoch微调即可适应全新概念集。

**[Fourierpet Deep Fourier-Based Unrolled Network For Low-Count Pet Reconstruction](interpretability/fourierpet_deep_fourier-based_unrolled_network_for_low-count_pet_reconstruction.md)**

:   发现低剂量 PET 的三类退化在频域可分离——泊松噪声/光子不足导致高频相位扰动，衰减校正误差抑制低频幅度——据此提出 FourierPET：基于 ADMM 展开的频率感知重建框架，仅 0.44M 参数在三个数据集上全面 SOTA。

**[Gatera Token-Aware Modulation For Parameter-Efficient Fine-Tuning](interpretability/gatera_token-aware_modulation_for_parameter-efficient_fine-tuning.md)**

:   提出 GateRA，在 PEFT 方法（LoRA/DoRA/HiRA）中引入轻量级 token 感知门控模块，通过 sigmoid 门控动态调整每个 token 的适配强度——对分布内/简单 token 抑制更新以保留预训练知识，对挑战性 token 放大适配。结合熵正则化促进近二值门控决策，在常识推理（+1.1%）、对话和数学推理上一致优于 HiRA。

**[Genepheno Interpretable Gene Knockout-Induced Phenotype Abnormality Prediction F](interpretability/genepheno_interpretable_gene_knockout-induced_phenotype_abnormality_prediction_f.md)**

:   本文提出 GenePheno，首个从基因序列端到端预测基因敲除诱导表型异常的可解释多标签预测框架，通过对比式多标签学习捕获表型间相关性、互斥正则化强制生物学一致性、以及基因本体（GO）瓶颈层提供可解释性，在 4 个数据集上取得 SOTA 的基因中心 $F_{\max}$ 和表型中心 AUC。

**[Hskbenchmark Modeling And Benchmarking Chinese Second Language Acquisition In La](interpretability/hskbenchmark_modeling_and_benchmarking_chinese_second_language_acquisition_in_la.md)**

:   提出 HSKBenchmark，首个面向 LLM 中文二语习得（SLA）分阶段建模与写作评估的基准，包含 HSK 3-6 级教材（6.76M tokens）、16K 合成指令数据、30 个测试题目及语言学评估系统，配合课程式微调框架模拟人类习得轨迹。

**[Hypothesis Generation Via Llm-Automated Language Bias For Ilp](interpretability/hypothesis_generation_via_llm-automated_language_bias_for_ilp.md)**

:   提出首个端到端框架：多Agent LLM系统（Actor/Critic）自动从原始文本构建ILP语言偏差（谓词系统+类型声明+模式约束），Translator将文本翻译为Prolog事实，再由MAXSYNTH求解器基于MDL原则归纳全局最优规则集。在SHOES和ZENDO任务上分别达88.3%和81.3%准确率，跨4种LLM方差<5%。

**[Imad Intelligent Multi-Agent Debate For Efficient And Accura](interpretability/imad_intelligent_multi-agent_debate_for_efficient_and_accura.md)**

:   iMAD 提出选择性触发多Agent辩论的框架：先让单Agent生成带自我批判的结构化响应，从中提取 41 个可解释的语言/语义特征，用轻量 MLP 分类器（FocusCal 损失训练）判断是否需要触发 MAD，在 6 个 QA/VQA 数据集上减少高达 92% 的 Token 开销，同时提升准确率高达 13.5%。

**[Induce Align Predict Zero-Shot Stance Detection Via Cognitive Inductive Reasonin](interpretability/induce_align_predict_zero-shot_stance_detection_via_cognitive_inductive_reasonin.md)**

:   提出CIRF框架，通过无监督schema归纳（USI）从LLM生成的一阶逻辑中抽象可迁移推理模式，再用schema增强图核模型（SEGKM）进行结构对齐实现可解释零样本立场推理，在三个基准上达到SOTA且仅需30%标注数据。

**[Probing Preference Representations A Multi-Dimensional Evaluation And Analysis M](interpretability/probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)**

:   提出 MRMBench 基准，通过 6 个维度（无害性、有帮助性、正确性、连贯性、复杂性、冗长性）的探针任务评估奖励模型是否有效捕获多维偏好，发现探针性能与 PPO 对齐质量强相关（Pearson $r > 0.8$），并提出推理时探针方法将 AlpacaEval win rate 从 57.3% 提升至 62.5%。

**[Quiet Feature Learning In Algorithmic Tasks](interpretability/quiet_feature_learning_in_algorithmic_tasks.md)**

:   在 10 个算法任务（18,544 次训练运行，$10^9$-$10^{16}$ FLOPs）上发现，Transformer 的损失平台期并非学习停滞——模型在此期间悄悄学习了"安静特征"（中间算法子程序），这些特征不直接降低输出损失但对最终性能因果必要（消融后准确率下降 41-75%）。这挑战了用损失曲线判断训练进展的常规做法。

**[Scope Intrinsic Semantic Space Control For Mitigating Copyright Infringement In ](interpretability/scope_intrinsic_semantic_space_control_for_mitigating_copyright_infringement_in_.md)**

:   将LLM版权侵权缓解问题重新定义为内在语义空间控制，利用稀疏自编码器(SAE)将隐状态映射到高维稀疏空间，识别版权敏感子空间并在解码时钳制其激活，无需外部过滤器或参数更新即可有效减少版权内容复制，同时保持模型通用能力。

**[Shapbpt Image Feature Attributions Using Data-Aware Binary Partition Trees](interpretability/shapbpt_image_feature_attributions_using_data-aware_binary_partition_trees.md)**

:   提出 ShapBPT，将**数据感知的二叉分割树（BPT）**作为层次联盟结构与 Owen 近似的 Shapley 值结合，实现与图像形态学对齐的特征归因，比现有 Shapley 方法收敛更快、形状识别更准确，并通过 20 人用户研究确认解释更受人类偏好。

**[Som Directions Are Better Than One Multi-Directional Refusal Suppression In Lang](interpretability/som_directions_are_better_than_one_multi-directional_refusal_suppression_in_lang.md)**

:   证明LLM的拒绝行为并非由单一方向编码，而是形成低维流形，利用自组织映射（SOM）提取多个拒绝方向并通过贝叶斯优化搜索最优消融组合，在多个模型上超越单方向基线和专用越狱算法。

**[Spark Query-Aware Unstructured Sparsity With Recoverable Kv Cache Channel Prunin](interpretability/spark_query-aware_unstructured_sparsity_with_recoverable_kv_cache_channel_prunin.md)**

:   提出SparK——一种training-free的KV cache通道级非结构化剪枝方法，通过query-aware的saliency评估选择关键通道+recovery机制恢复被剪枝通道的贡献，在80%剪枝率下性能损失<5%，与token eviction方法正交互补，可额外减少30%+ KV cache存储。

**[Toc Tree-Of-Claims Search With Multi-Agent Language Models](interpretability/toc_tree-of-claims_search_with_multi-agent_language_models.md)**

:   提出 Tree-of-Claims (ToC) 框架，将专利权利要求编辑建模为结构化搜索问题，通过 MCTS 与 EditorAgent/ExaminerAgent 多智能体协作，在新颖性、范围保持和语义一致性之间联合优化，比零/少样本 LLM 基线平均提升约 8% 综合分。

**[Universal Safety Controllers With Learned Prophecies](interpretability/universal_safety_controllers_with_learned_prophecies.md)**

:   提出 UCLearn，通过从少量代表性 plant 模型中学习 CTL (计算树逻辑) 公式作为预言（prophecy）的近似表示，替代精确但计算昂贵的树自动机，实现高效、可扩展且可解释的通用安全控制器合成。

**[Unsupervised Feature Selection Through Group Discovery](interpretability/unsupervised_feature_selection_through_group_discovery.md)**

:   提出 GroupFS，首个端到端可微分的无监督特征选择框架，能同时发现潜在特征分组并选择信息量最大的组，无需预定义分组或标签监督。

**[Using Certifying Constraint Solvers For Generating Step-Wise Explanations](interpretability/using_certifying_constraint_solvers_for_generating_step-wise_explanations.md)**

:   提出利用约束求解器生成的不可满足性证明（proof）作为起点，通过一系列简化和转换技术高效生成面向用户的逐步解释序列（step-wise explanation），相比从零构建解释方法速度提升高达100倍。

**[Voices Faces And Feelings Multi-Modal Emotion-Cognition Captioning For Mental He](interpretability/voices_faces_and_feelings_multi-modal_emotion-cognition_captioning_for_mental_he.md)**

:   提出情感-认知协同多模态描述（ECMC）任务和框架，通过双流BridgeNet从视频、音频、文本中提取情感和认知特征，利用LLaMA生成自然语言描述，为心理健康评估提供可解释的情感-认知画像，显著提升辅助诊断的准确性和可解释性。

---

## 📹 视频理解 { #video_understanding }

**[Apvr Hour-Level Long Video Understanding With Adaptive Pivot](video_understanding/apvr_hour-level_long_video_understanding_with_adaptive_pivot.md)**

:   提出APVR，一个训练免费的双粒度视觉信息检索框架：帧级别通过查询扩展+时空语义置信度打分迭代检索关键帧（最多1024帧），token级别通过查询感知的注意力驱动选择压缩视觉token，突破内存墙限制处理小时级长视频，在LongVideoBench/VideoMME/MLVU上分别提升最高9.5%/4.6%/9.7%。

**[Balancing Multimodal Domain Generalization Via Gradient Modulation And Projectio](video_understanding/balancing_multimodal_domain_generalization_via_gradient_modulation_and_projectio.md)**

:   提出 Gradient Modulation Projection (GMP) 策略，通过解耦分类与域不变梯度的调制（IGDM）以及冲突自适应梯度投影（CAGP），解决多模态域泛化中模态间优化不平衡和任务间梯度冲突问题，在多个基准上达到 SOTA。

**[Bat Learning Event-Based Optical Flow With Bidirectional Adaptive Temporal Corre](video_understanding/bat_learning_event-based_optical_flow_with_bidirectional_adaptive_temporal_corre.md)**

:   提出双向自适应时序相关性（BAT）框架，将事件相机的时序密集运动线索转化为空间密集线索，实现高精度事件光流估计，在 DSEC-Flow 基准上排名第一。

**[Causality Matters How Temporal Information Emerges In Video Language Models](video_understanding/causality_matters_how_temporal_information_emerges_in_video_language_models.md)**

:   通过系统性消融实验揭示VideoLM的时序理解能力并非来源于位置编码(PE)，而是由因果注意力掩码的序列敏感性产生——时序信息沿"帧间交互→末帧聚合→query融合"的因果路径逐层构建，并据此提出两种无损推理加速策略。

**[Coordinated Humanoid Robot Locomotion With Symmetry Equivariant Reinforcement Le](video_understanding/coordinated_humanoid_robot_locomotion_with_symmetry_equivariant_reinforcement_le.md)**

:   提出 SE-Policy，将严格的对称等变性（actor）和对称不变性（critic）直接嵌入神经网络架构，无需额外超参数即可使人形机器人产生时空协调的自然运动，速度跟踪误差相比 DreamWaQ 降低 40%，并成功部署到 Unitree G1 实体机器人。

**[Decomposition And Preprocessing Of Ternary Constraint Networks](video_understanding/decomposition_and_preprocessing_of_ternary_constraint_networks.md)**

:   提出将任意离散约束网络形式化分解为三元约束网络(TCN)的完整理论框架，并通过七项预处理技术（传播、代数简化、公共子表达式消除等）将分解引入的变量/约束膨胀从中位数8x/6x降至4.8x/4.3x，为GPU硬件上的高效约束求解提供规则化数据布局。

**[Distillation Dynamics Towards Understanding Feature-Based Di](video_understanding/distillation_dynamics_towards_understanding_feature-based_di.md)**

:   提出"蒸馏动力学"分析框架（通道维FFT频谱分析+Shannon熵+激活幅值追踪），揭示ViT具有独特的U型信息处理模式（先压缩后扩展），证明feature-based蒸馏在ViT中失败的根本原因是teacher后层的分布式高维编码范式与student有限通道容量之间的表征范式不匹配，而非简单的容量差距。

**[Emovid A Multimodal Emotion Video Dataset For Emotion-Centric Video Understandin](video_understanding/emovid_a_multimodal_emotion_video_dataset_for_emotion-centric_video_understandin.md)**

:   提出 EmoVid，首个面向艺术化/非写实内容的大规模多模态情绪视频数据集（22,758 个视频片段），覆盖动画、电影和表情贴纸三种类型，并通过微调 Wan2.1 模型展示了情绪条件化视频生成的有效性，在情绪准确率指标上显著优于基线。

**[Explicit Temporal-Semantic Modeling For Dense Video Captioning Via Context-Aware](video_understanding/explicit_temporal-semantic_modeling_for_dense_video_captioning_via_context-aware.md)**

:   本文提出 CACMI 框架，通过显式时序-语义建模解决密集视频描述任务中的两个基本限制（时序建模不足和模态鸿沟），使用跨模态帧聚合（CFA）提取时序一致的事件语义，再用上下文感知特征增强（CFE）桥接视觉-文本模态差距，在 ActivityNet Captions 和 YouCook2 上达到 SOTA。

**[Finetec Fine-Grained Action Recognition Under Temporal Corruption Via Skeleton D](video_understanding/finetec_fine-grained_action_recognition_under_temporal_corruption_via_skeleton_d.md)**

:   提出 FineTec 框架，通过上下文感知序列补全、基于生物先验的骨架空间分解、物理驱动的加速度建模三个模块，在时序损坏条件下实现鲁棒的细粒度骨架动作识别。

**[Group Orthogonal Low-Rank Adaptation For Rgb-T Tracking](video_understanding/group_orthogonal_low-rank_adaptation_for_rgb-t_tracking.md)**

:   提出 GOLA 框架，通过 SVD 分解量化 LoRA 秩重要性、冻结关键秩保留预训练先验、将冗余秩分组并施加组间正交约束，实现更高效的 RGB-T 跟踪适配。

**[Kinest A Kinematics-Guided Spatiotemporal State Space Model For Human Motion Tra](video_understanding/kinest_a_kinematics-guided_spatiotemporal_state_space_model_for_human_motion_tra.md)**

:   提出 KineST，一种运动学引导的状态空间模型，通过运动学树双向扫描策略和混合时空表征学习，从头显稀疏信号高效重建全身运动，在精度和时序一致性上均超越 SOTA。

**[Learning Topology-Driven Multi-Subspace Fusion For Grassmannian Deep Network](video_understanding/learning_topology-driven_multi-subspace_fusion_for_grassmannian_deep_network.md)**

:   提出拓扑驱动的 Grassmann 流形多子空间融合网络 GMSF-Net，通过自适应多子空间构建和基于 Fréchet 均值的子空间交互机制，将欧氏空间中多通道交互的思想成功迁移到非欧几何域，在 3D 动作识别、EEG 分类和图任务上取得 SOTA 性能。

**[Lifelong Domain Adaptive 3D Human Pose Estimation](video_understanding/lifelong_domain_adaptive_3d_human_pose_estimation.md)**

:   提出 lifelong domain adaptive 3D HPE 新任务，设计包含 pose-aware、temporal-aware 和 domain-aware 编码的 GAN 框架，利用 diffusion sampler 生成 domain-aware prior 缓解灾难性遗忘，在多个跨场景/跨数据集适应任务上显著超越现有方法。

**[Listening Between The Frames Bridging Temporal Gaps In Large Audio-Language Mode](video_understanding/listening_between_the_frames_bridging_temporal_gaps_in_large_audio-language_mode.md)**

:   提出 TimeAudio，通过时间标记（Temporal Markers）、绝对时间编码（Absolute Time-aware Encoding）和段级 Token 合并（Segment-level Token Merging）三个关键模块，赋予大型音频语言模型（LALM）精确的时间定位能力和端到端长音频理解能力，并构建了 FTAR 数据集用于细粒度时间推理的指令微调。

**[Livibench An Omnimodal Benchmark For Interactive Livestream Video Understanding](video_understanding/livibench_an_omnimodal_benchmark_for_interactive_livestream_video_understanding.md)**

:   提出首个面向交互式直播视频的全模态基准 LiViBench（3168 个视频、3175 道 MCQ、24 个任务），设计了多智能体种子问题驱动的半自动标注流程，并构建了 LiVi-LLM-7B 模型（含 Video-to-Comment Retrieval 模块和两阶段指令微调），在 7B 规模下超越了 72B 开源模型。

**[Loom Personalized Learning Informed By Daily Llm Conversations Toward Long-Term ](video_understanding/loom_personalized_learning_informed_by_daily_llm_conversations_toward_long-term_.md)**

:   提出 LOOM，一个智能体管线系统，通过观察用户日常 LLM 对话、推断学习需求、维护动态学习者记忆图（Learner Memory Graph），自动生成个性化的迷你课程，统一了学习的**连续性**（长期进度追踪）和**主动性**（即时响应新兴趣）。

**[Plugtrack Multi-Perceptive Motion Analysis For Adaptive Fusion In Multi-Object T](video_understanding/plugtrack_multi-perceptive_motion_analysis_for_adaptive_fusion_in_multi-object_t.md)**

:   提出 PlugTrack 框架，通过多感知运动分析（CME）和自适应混合因子生成（ABG），首次实现卡尔曼滤波器与数据驱动运动预测器的自适应融合，在线性和非线性运动场景中均取得显著提升。

**[Pragworld A Benchmark Evaluating Llms Local World Model Under Minimal Linguistic](video_understanding/pragworld_a_benchmark_evaluating_llms_local_world_model_under_minimal_linguistic.md)**

:   提出 PragWorld 基准测试，通过对对话施加 7 种最小语言学扰动来评估 LLM 内隐世界模型的可塑性和鲁棒性，并设计双视角可解释性框架定位有害/有用层，提出层正则化微调策略提升鲁棒性。

**[Predicting Video Slot Attention Queries From Random Slot-Feature Pairs](video_understanding/predicting_video_slot_attention_queries_from_random_slot-feature_pairs.md)**

:   提出 RandSF.Q，通过利用下一帧特征进行信息性查询预测，以及从随机采样的 slot-feature 对学习过渡动力学，显著提升视频物体中心学习（OCL）的查询预测质量，在目标发现任务上超越 SOTA 最多 10 个点。

**[Quantifying Conversational Reliability Of Large Language Models Under Multi-Turn](video_understanding/quantifying_conversational_reliability_of_large_language_models_under_multi-turn.md)**

:   通过三个可确定性评估的代表性任务（指令遵循、工具选择、实体抽取），系统量化 LLM 在多轮对话中的可靠性退化程度，揭示模型在扩展对话中出现指令漂移、意图混淆和上下文覆写等失败模式。

**[R-Avst Empowering Video-Llms With Fine-Grained Spatio-Temporal Reasoning In Comp](video_understanding/r-avst_empowering_video-llms_with_fine-grained_spatio-temporal_reasoning_in_comp.md)**

:   提出首个面向复杂音视频场景的细粒度时空推理数据集 R-AVST（5K+未裁剪视频、27K物体、100类音视频事件），定义三个核心推理任务，并基于 GRPO 训练 AVST-Zero 模型，通过多维奖励函数直接优化音视频时空推理能力。

**[Reason Reinforced Causal Search With Information Bottleneck For Video Understand](video_understanding/reason_reinforced_causal_search_with_information_bottleneck_for_video_understand.md)**

:   提出因果信息瓶颈（CIB）理论框架，将关键帧选择形式化为同时优化"预测充分性"和"因果必要性"的信息论问题，并基于此设计 ReaSon 强化学习框架，通过三种 CIB 对齐的奖励（答案奖励、循环一致性奖励、反事实奖励）训练选择策略，在限定帧数设置下显著超越已有方法。

**[Rectom A Benchmark For Evaluating Machine Theory Of Mind In Llm-Based Conversati](video_understanding/rectom_a_benchmark_for_evaluating_machine_theory_of_mind_in_llm-based_conversati.md)**

:   提出 RecToM，首个用于评估 LLM 在对话推荐系统中心智理论（Theory of Mind）推理能力的人工标注基准，涵盖认知推理（欲望/意图/信念）和行为预测（策略预测/策略判断）两个维度共 10 种问题类型、20,524 个 QA 对，揭示了当前 LLM 在细粒度意图推断和策略判断中的系统性缺陷。

**[Rethinking Progression Of Memory State In Robotic Manipulation An Object-Centric](video_understanding/rethinking_progression_of_memory_state_in_robotic_manipulation_an_object-centric.md)**

:   提出 LIBERO-Mem 基准（10 个非马尔可夫机器人操控任务）和 Embodied-SlotSSM 框架（结合 Slot Attention 和状态空间模型的物体中心记忆 VLA），解决视觉运动策略在部分可观测、需要物体级历史推理的长期任务中的失败问题。

**[State-Space Hierarchical Compression With Gated Attention An](video_understanding/state-space_hierarchical_compression_with_gated_attention_an.md)**

:   MambaMia 提出了基于双向 Mamba 的两阶段层次化视频 Token 压缩框架：门控 Patch 聚合（GPA）做空间-时间局部压缩 + 时间轴聚合器（TAA）利用 Mamba 的自适应步长 $\Delta_t$ 做数据驱动的关键帧采样，将小时级视频压缩到仅 4.7K Token，在 LVBench 上达到 44.6 分超越 Qwen2-VL 和 mPLUG-Owl3。

**[Stegavar Privacy-Preserving Video Action Recognition Via Steganographic Domain A](video_understanding/stegavar_privacy-preserving_video_action_recognition_via_steganographic_domain_a.md)**

:   提出 StegaVAR 框架，首次将视频隐写术与动作识别结合，将隐私视频嵌入自然 cover 视频后直接在隐写域做分类，通过 STeP（secret 视频引导的时空特征学习）和 CroDA（跨频带差分注意力）实现接近原始视频的识别精度，同时提供优于匿名化方法的隐私保护。

**[Sugar Learning Skeleton Representation With Visual-Motion Knowledge For Action R](video_understanding/sugar_learning_skeleton_representation_with_visual-motion_knowledge_for_action_r.md)**

:   提出 SUGAR 范式，利用 GPT 生成的**运动描述**和**视觉描述**作为先验知识，通过对比学习监督骨骼编码器学习更离散的表示，再用 LLM（LLaMA2-7B）的未触及预训练权重作为识别器，配合新设计的 Temporal Query Projection（TQP）模块实现高效的骨骼动作分类和零样本推理。

**[Task-Specific Distance Correlation Matching For Few-Shot Action Recognition](video_understanding/task-specific_distance_correlation_matching_for_few-shot_action_recognition.md)**

:   提出 TS-FSAR 框架，通过 α-距离相关性捕获帧间非线性依赖关系并结合任务特定匹配矩阵进行 query-support 匹配，同时用适配后的冻结 CLIP 引导侧网络训练，在 SSv2-Full 等时序敏感数据集上大幅超越先前方法。

**[Tspo Temporal Sampling Policy Optimization For Long-Form Video Language Understa](video_understanding/tspo_temporal_sampling_policy_optimization_for_long-form_video_language_understa.md)**

:   将视频关键帧选择和语言生成建模为联合决策过程，通过基于GRPO的强化学习端到端优化轻量级时序智能体的采样策略，在四个长视频理解基准上取得SOTA（LLaVA-Video-7B上LongVideoBench +5.0%、MLVU +6.0%），且可零样本迁移到其他Video-MLLM。

**[Uvlm Benchmarking Video Language Model For Underwater World Understanding](video_understanding/uvlm_benchmarking_video_language_model_for_underwater_world_understanding.md)**

:   构建首个水下视频语言理解基准 UVLM（2109 段视频、419 类海洋生物、20 种子任务、~4 万 video-text pairs），通过 human-AI 协同标注注入海洋领域知识，在 UVLM 上微调后 7B VidLM 可达到接近 GPT-4o 的性能（73.04 vs 77.95 Overall）。

**[Vir-Bench Evaluating Geospatial And Temporal Understanding Of Mllms Via Travel V](video_understanding/vir-bench_evaluating_geospatial_and_temporal_understanding_of_mllms_via_travel_v.md)**

:   提出VIR-Bench——一个基于200个日本旅行vlog视频的benchmark，通过行程重建任务（visiting order graph构建）评估MLLM的地理空间和时间理解能力，发现SOTA模型（包括GPT-4.1和Gemini-2.5）在POI识别和时间转移推理上仍困难重重。

**[Vtinker Guided Flow Upsampling And Texture Mapping For High-Resolution Video Fra](video_understanding/vtinker_guided_flow_upsampling_and_texture_mapping_for_high-resolution_video_fra.md)**

:   提出 VTinker 流水线，通过引导式光流上采样（GFU）解决光流边界模糊问题，并采用纹理映射替代传统逐像素融合策略来消除鬼影和不连续，在高分辨率视频帧插值上取得 SOTA。

---

## 📊 LLM评测 { #llm_evaluation }

**[Axis-Aligned Document Dewarping](llm_evaluation/axis-aligned_document_dewarping.md)**

:   提出利用平面文档固有的"轴对齐"几何性质，在训练、推理和评估三个阶段系统性地引入轴对齐约束，实现了SOTA文档矫正效果并提出新评估指标AAD。

**[Bcwildfire A Long-Term Multi-Factor Dataset And Deep Learning Benchmark For Bore](llm_evaluation/bcwildfire_a_long-term_multi-factor_dataset_and_deep_learning_benchmark_for_bore.md)**

:   本文构建了一个覆盖加拿大BC省2.4亿公顷、跨度25年的多模态野火风险预测数据集BCWildfire，包含38个驱动因子，并对CNN/Linear/Transformer/Mamba四大范式的时序预测模型进行了系统评测，揭示了当前模型在野火预测中的性能上限和关键影响因子。

**[Benchmarking Llms For Political Science A United Nations Perspective](llm_evaluation/benchmarking_llms_for_political_science_a_united_nations_perspective.md)**

:   提出 UNBench，首个基于联合国安理会 1994-2024 年记录的综合性政治科学 LLM 评测基准，涵盖决议起草、投票模拟、通过预测和代表发言生成四个关联任务，评估 LLM 对复杂政治动态的理解和模拟能力。

**[Beyond Accuracy A Cognitive Load Framework For Mapping The C](llm_evaluation/beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c.md)**

:   借鉴心理学的认知负荷理论（CLT），将工具使用任务的复杂度分解为内在负荷（任务解题路径的结构复杂度）和外在负荷（问题表述的歧义性），构建可参数化调节认知负荷的 ToolLoad-Bench 基准，用指数衰减模型 $\text{Acc} \approx e^{-(k \cdot CL + b)}$ 精确刻画不同 Agent 的能力边界。

**[Beyond Cosine Similarity Magnitude-Aware Clip For No-Reference Image Quality Ass](llm_evaluation/beyond_cosine_similarity_magnitude-aware_clip_for_no-reference_image_quality_ass.md)**

:   提出 MA-CLIP，发现并利用 CLIP 图像特征的**幅度信息**作为感知质量的互补线索，结合余弦相似度实现无需训练的自适应双线索融合图像质量评估。

**[Coninstruct Evaluating Large Language Models On Conflict Detection And Resolutio](llm_evaluation/coninstruct_evaluating_large_language_models_on_conflict_detection_and_resolutio.md)**

:   提出 ConInstruct 基准，评估 LLM 在指令包含冲突约束时的检测和解决能力，发现多数专有模型能较好检测冲突但很少主动告知用户，其中 DeepSeek-R1 和 Claude-4.5-Sonnet 在冲突检测上表现最佳（F1 分别达 91.5% 和 87.3%）。

**[Dcmatch Unsupervised Multi-Shape Matching With Dual-Level Consistency](llm_evaluation/dcmatch_unsupervised_multi-shape_matching_with_dual-level_consistency.md)**

:   提出DcMatch——一种无监督多形状匹配框架，通过形状图注意力网络捕捉形状集合底层流形结构以构建更具表达力的共享宇宙空间，并在空间域和谱域实施双层循环一致性约束，在多个基准数据集上实现全面超越。

**[Dicap Distribution-Calibrated Pseudo-Labeling For Semi-Supervised Multi-Label Le](llm_evaluation/dicap_distribution-calibrated_pseudo-labeling_for_semi-supervised_multi-label_le.md)**

:   提出 DiCaP（Distribution-Calibrated Pseudo-labeling），通过估计伪标签的后验正确率来校准权重、引入双阈值机制分离置信区间和模糊区间并采用不同策略，在半监督多标签学习中以最高 4.27% 的幅度超越 SOTA。

**[Gdba Revisited Unleashing The Power Of Guided Local Search For Distributed Const](llm_evaluation/gdba_revisited_unleashing_the_power_of_guided_local_search_for_distributed_const.md)**

:   针对 GDBA 在一般值域 DCOP 上表现不佳的问题，本文系统分析了三大病因（过于激进的违反条件、无界惩罚累积、不协调的惩罚更新），提出了 DGLS 框架，通过自适应违反条件、蒸发机制和同步方案全面释放引导式局部搜索的性能，在多种标准基准上大幅超越 SOTA。

**[Gene Incremental Learning For Single-Cell Transcriptomics](llm_evaluation/gene_incremental_learning_for_single-cell_transcriptomics.md)**

:   本文提出了基因增量学习（GIL）框架，利用单细胞转录组学数据的无序性特点，将类增量学习（CIL）的范式扩展到 token（基因）维度，设计了基因回放和基因蒸馏两种基线方法，并建立了包含基因级回归和基因级分类两种评估方式的完整基准。

**[Goal Geometrically Optimal Alignment For Continual Generalized Category Discover](llm_evaluation/goal_geometrically_optimal_alignment_for_continual_generalized_category_discover.md)**

:   基于 Neural Collapse 理论，使用固定等角紧框架（ETF）分类器替代动态分类器，通过监督对齐和置信度引导的无监督对齐实现持续泛化类别发现，在四个基准上遗忘率降低 16.1%、新类发现提升 3.2%。

**[Granalign Granularity-Aware Alignment Framework For Zero-Shot Video Moment Retri](llm_evaluation/granalign_granularity-aware_alignment_framework_for_zero-shot_video_moment_retri.md)**

:   提出一个无需训练的粒度感知对齐框架GranAlign，通过将查询重写为简化版和细化版并分别匹配无关/感知查询的视频描述，解决了零样本视频时刻检索中语义粒度不匹配的核心难题，在QVHighlights上mAP@avg提升3.23%。

**[Graph Out-Of-Distribution Detection Via Test-Time Calibration With Dual Dynamic ](llm_evaluation/graph_out-of-distribution_detection_via_test-time_calibration_with_dual_dynamic_.md)**

:   提出 BaCa 框架，在测试阶段通过 graphon 估计 + mixup 策略生成边界感知的合成图拓扑，结合双优先队列动态字典和注意力机制自适应校准 OOD 分数，无需微调预训练模型或引入辅助OOD数据，在全部 10 个数据集上超越 GOODAT，平均 AUC 提升 8.37%。

**[Hybridla Hybrid Generation For Document Layout Analysis](llm_evaluation/hybridla_hybrid_generation_for_document_layout_analysis.md)**

:   HybriDLA 首次将扩散式边框精炼与自回归查询扩展统一在一个解码层中，模拟人类由粗到细的阅读策略来处理文档版面分析，在 DocLayNet 上纯视觉模型达到 83.5% mAP，逼近多模态系统。

**[Improved Runtime Guarantees For The Spea2 Multi-Objective Optimizer](llm_evaluation/improved_runtime_guarantees_for_the_spea2_multi-objective_optimizer.md)**

:   通过深入分析SPEA2更复杂的选择机制，证明了其种群动态与NSGA-II有本质不同（σ-准则使目标值在种群中均匀分布），从而得到了对种群大小依赖更弱的运行时上界，表明SPEA2对参数选择具有更强的鲁棒性。

**[Llm-As-A-Judge For Scalable Test Coverage Evaluation Accuracy Operational Reliab](llm_evaluation/llm-as-a-judge_for_scalable_test_coverage_evaluation_accuracy_operational_reliab.md)**

:   将LLM-as-Judge范式应用于Gherkin验收测试覆盖率评估，在20种模型配置x500次评估中系统量化准确性-可靠性-成本三维权衡，发现GPT-4o Mini以6.07 MAAE、96.6% ECR@1和$1.01/1K评估成为最优生产选择，成本仅为GPT-5高推理版的1/78。

**[Lost In Benchmarks Rethinking Large Language Model Benchmarking With Item Respon](llm_evaluation/lost_in_benchmarks_rethinking_large_language_model_benchmarking_with_item_respon.md)**

:   提出 PSN-IRT（Pseudo-Siamese Network for IRT），用增强版项目反应理论同时估计 LLM 能力参数和题目的四参数特征（难度/区分度/猜测率/可行性），在 11 个基准 41,871 题上发现当前基准存在广泛饱和、难度天花板不足、数据污染等系统性问题，PSN-IRT 选出的题目子集排名一致性达 Kendall τ=1.00。

**[Low-Rank Curvature For Zeroth-Order Optimization In Llm Fine-Tuning](llm_evaluation/low-rank_curvature_for_zeroth-order_optimization_in_llm_fine-tuning.md)**

:   提出 LOREN，一种曲率感知的零阶优化方法，通过低秩块对角预条件器捕获损失景观的各向异性曲率，并结合 REINFORCE Leave-One-Out 方差缩减技术，在 LLM 微调中实现了更高精度和更快收敛，同时相比 MeZO-Adam 节省高达 27.3% 的峰值内存。

**[Maps Multi-Agent Personality Shaping For Collaborative Reaso](llm_evaluation/maps_multi-agent_personality_shaping_for_collaborative_reaso.md)**

:   提出 MAPS 五 Agent 协作推理框架，基于大五人格理论为 4 个功能 Agent 赋予不同"性格"（Interpreter-开放性、Aligner-宜人性、Scholar-尽责性、Solver-外向性）实现异质化协作，加上 Critic Agent（神经质→苏格拉底式反思）做迭代修正，在 MathVista/OlympiadBench/EMMA 上超越 GPT-4o 基线 15.84%，首次超过人类专家 3.58%。

**[Mcts-Sql Light-Weight Llms Can Master The Text-To-Sql Through Monte Carlo Tree S](llm_evaluation/mcts-sql_light-weight_llms_can_master_the_text-to-sql_through_monte_carlo_tree_s.md)**

:   提出MCTS-SQL，让轻量LLM（如Qwen-1.5B）通过蒙特卡洛树搜索实现强大的Text-to-SQL能力——三组件架构（Selector做Schema剪枝 + Direct Generator生成初始SQL + MCTS-Refiner迭代精化），配合前缀缓存机制减少53%推理时间，Qwen-1.5B在BIRD上达40.69%执行准确率（超ChatGPT-3.5）。

**[Mindvote When Ai Meets The Wild West Of Social Media Opinion](llm_evaluation/mindvote_when_ai_meets_the_wild_west_of_social_media_opinion.md)**

:   提出 MindVote——首个基于真实社交媒体投票数据的 LLM 舆情预测基准，包含 Reddit/微博上 3,918 个自然投票（23 个话题），附带平台和话题上下文。评估 15 个 LLM 发现：最佳模型（o3-medium）1-Wasserstein 仅 0.892 vs 上界 0.972；在调查数据上微调的专用模型反而不如通用模型（"调查特化陷阱"）；模型表现出强烈文化对齐——西方模型擅长 Reddit、中国模型擅长微博。

**[Moetta Test-Time Adaptation Under Mixed Distribution Shifts With Moe-Layernorm](llm_evaluation/moetta_test-time_adaptation_under_mixed_distribution_shifts_with_moe-layernorm.md)**

:   本文提出 MoETTA，一种将 LayerNorm 重参数化为多个结构解耦专家分支的测试时自适应框架，通过路由机制为不同域的样本选择不同的适应方向，解决了混合分布偏移下单一适应路径的局限性，并提出 potpourri/potpourri+ 两个更真实的评估基准，在所有设定下取得 SOTA。

**[Nestr A Neuro-Symbolic Abductive Framework For Temporal Reasoning In Large Langu](llm_evaluation/nestr_a_neuro-symbolic_abductive_framework_for_temporal_reasoning_in_large_langu.md)**

:   提出 NeSTR 神经符号提示策略，通过将自然语言时间事实转化为结构化符号谓词，结合一致性验证和溯因反思修正，在零样本设置下让 LLM 实现高质量时间推理，GPT-4o-mini 上平均 F1 达 89.7（相比 vanilla 64.9 和 TISER 85.8）。

**[Optscale Probabilistic Optimality For Inference-Time Scaling](llm_evaluation/optscale_probabilistic_optimality_for_inference-time_scaling.md)**

:   提出概率最优框架 OptScale，通过建模验证器分数的概率分布推导出最优采样数量的理论下界，动态决定每个问题所需的最少采样次数，在保持推理准确率的同时大幅减少计算开销。

**[Perspective From A Broader Context Can Room Style Knowledge Help Visual Floorpla](llm_evaluation/perspective_from_a_broader_context_can_room_style_knowledge_help_visual_floorpla.md)**

:   提出利用房间风格知识（通过无监督聚类预训练获得的 room discriminator）来消除视觉楼层平面图定位中因重复结构导致的歧义，在 Gibson 和 Structured3D 两个标准基准上取得 SOTA 性能。

**[Refinevad Semantic-Guided Feature Recalibration For Weakly Supervised Video Anom](llm_evaluation/refinevad_semantic-guided_feature_recalibration_for_weakly_supervised_video_anom.md)**

:   提出 RefineVAD 框架，通过运动感知时序注意力重校准（MoTAR）和类别导向特征精炼（CORE）两个模块，联合建模时序运动动态与异常类别语义，在弱监督视频异常检测任务上实现了对异常事件的精准定位与可解释检测。

**[Regular Games -- An Automata-Based General Game Playing Language](llm_evaluation/regular_games_--_an_automata-based_general_game_playing_language.md)**

:   提出 Regular Games (RG) 通用博弈系统，以非确定性有限自动机（NFA）为核心描述博弈规则，配合多层次语言（底层 RG + 高层 HRG + 专用框架），在表达力覆盖所有有限回合制博弈（含不完全信息和随机性）的同时，生成的前向模型效率全面超越现有最快的通用博弈系统 RBG，通常比 Ludii 快 10-20 倍。

**[Sampling Control For Imbalanced Calibration In Semi-Supervised Learning](llm_evaluation/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)**

:   提出 SC-SSL 框架，通过引入**扩展分类器**进行解耦采样控制来缓解特征级不平衡，并利用线性层**偏置项**作为优化偏差向量在推理时直接校准 logits，在多种数据分布设定下达到 SOTA。

**[Scalable Vision-Guided Crop Yield Estimation](llm_evaluation/scalable_vision-guided_crop_yield_estimation.md)**

:   提出基于**预测驱动推断（PPI++）**的农作物产量估计方法，利用田间照片训练的视觉模型补充昂贵的实地测产数据，在保证无偏性的同时将有效样本量提升高达 73%，为区域农业保险提供更精确且低成本的产量估计。

**[Spikcommander A High-Performance Spiking Transformer With Multi-View Learning Fo](llm_evaluation/spikcommander_a_high-performance_spiking_transformer_with_multi-view_learning_fo.md)**

:   提出 SpikCommander，一种全脉冲驱动的 Transformer 架构，通过**多视图脉冲时序感知自注意力（MSTASA）**和**脉冲上下文精炼 MLP（SCR-MLP）**联合增强时序与通道特征建模，在 SHD/SSC/GSC 三个基准上以更少参数超越 SOTA SNN 方法。

**[Streaming Generated Gaussian Process Experts For Online Learning And Control Ext](llm_evaluation/streaming_generated_gaussian_process_experts_for_online_learning_and_control_ext.md)**

:   提出 SkyGP（Streaming Kernel-induced Progressively Generated Expert GP），通过**核距离驱动的渐进式专家生成**和**时间感知可配置聚合**处理流数据，继承精确 GP 的学习保证同时保持有界计算复杂度，在基准测试和实时控制实验中全面超越 SOTA。

**[Where Norms And References Collide Evaluating Llms On Normative Reasoning](llm_evaluation/where_norms_and_references_collide_evaluating_llms_on_normative_reasoning.md)**

:   提出 SNIC 诊断测试台（9,000 实例/51 场景），评估 LLM 能否利用隐式社会规范来解决歧义参考消解（如"递给我杯子"时存在多个杯子）。发现 LLM 在仅看场景描述时平均准确率仅 44%，加上 Prolog 形式逻辑无显著改善（44.2%），但显式提供规范列表后猛升到 70.5%（GPT-4.1 达 99.6%），证明 LLM 缺乏隐式物理规范知识但能有效利用显式规范。

---

## 📈 时间序列 { #time_series }

**[A Unified Shape-Aware Foundation Model For Time Series Class](time_series/a_unified_shape-aware_foundation_model_for_time_series_class.md)**

:   提出 UniShape——一个面向时间序列分类的基础模型，通过 shape-aware adapter 自适应聚合多尺度判别性子序列（shapelet），并结合原型对比预训练在实例和 shape 两个层面学习可迁移的 shapelet 表示，在 128 个 UCR 数据集上以 3.1M 参数达到 SOTA（平均准确率 87.08%），同时提供良好的分类可解释性。

**[Airdde Multifactor Neural Delay Differential Equations For Air Quality Forecasti](time_series/airdde_multifactor_neural_delay_differential_equations_for_air_quality_forecasti.md)**

:   首个将神经延迟微分方程（NDDE）引入空气质量预测的框架，通过记忆增强注意力模块和物理引导的延迟演化函数，对污染物连续时间传播中的延迟效应进行建模，在三个数据集上平均 MAE 降低 8.79%。

**[Beyond Observations Reconstruction Error-Guided Irregularly Sampled Time Series ](time_series/beyond_observations_reconstruction_error-guided_irregularly_sampled_time_series_.md)**

:   提出 iTimER，利用模型自身的重建误差分布作为学习信号——从观测点估计误差分布后采样生成未观测时刻的伪观测值，通过 Wasserstein 距离对齐观测/伪观测区域的误差分布 + 对比学习，在不规则采样时序的分类、插值、预测任务上全面超越 SOTA。

**[C3Rl Rethinking The Combination Of Channel-Independence And Channel-Mixing From ](time_series/c3rl_rethinking_the_combination_of_channel-independence_and_channel-mixing_from_.md)**

:   提出 C3RL，基于 SimSiam 对比学习框架将通道独立（CI）和通道混合（CM）策略视为同一数据的两个转置视图构建正样本对，通过孪生网络联合表示学习和预测学习，将 CI 模型的最佳性能率从 43.6% 提升到 81.4%，CM 模型从 23.8% 提升到 76.3%。

**[Coherent Multi-Agent Trajectory Forecasting In Team Sports With Causaltraj](time_series/coherent_multi-agent_trajectory_forecasting_in_team_sports_with_causaltraj.md)**

:   提出CausalTraj——一种时序因果、基于似然的多智能体轨迹预测模型，通过逐步自回归建模智能体间时空交互，在NBA、篮球和橄榄球数据集上实现了联合指标（minJADE/minJFDE）的最优结果，同时保持有竞争力的单智能体精度。

**[Cometnet Contextual Motif-Guided Long-Term Time Series Forecasting](time_series/cometnet_contextual_motif-guided_long-term_time_series_forecasting.md)**

:   提出 CometNet，通过从完整历史序列中提取循环出现的"上下文 motif"构建 motif 库，再用 motif 引导的 MoE 架构动态关联当前窗口与相关motif进行预测，突破了有限回看窗口的感受野瓶颈，在8个数据集上显著超越 TimeMixer++、iTransformer 等 SOTA。

**[Counterfactual Explainable Ai Xai Method For Deep Learning-Based Multivariate Ti](time_series/counterfactual_explainable_ai_xai_method_for_deep_learning-based_multivariate_ti.md)**

:   提出 CONFETTI，一种面向多变量时间序列（MTS）分类的多目标反事实解释方法，通过结合类激活图（CAM）引导的子序列提取与 NSGA-III 多目标优化，在预测置信度、稀疏性和接近度三个目标间实现最优平衡，在 7 个 UEA 数据集上全面超越现有方法。

**[Deepboots Dual-Stream Residual Boosting For Drift-Resilient Time-Series Forecast](time_series/deepboots_dual-stream_residual_boosting_for_drift-resilient_time-series_forecast.md)**

:   提出 DeepBooTS，通过偏差-方差分解理论证明加权集成可降低方差从而缓解概念漂移，设计双流残差递减 boosting 架构，每个 block 的输出修正前一个 block 的残差，在多个数据集上平均提升 15.8%。

**[Detecting The Future All-At-Once Event Sequence Forecasting With Horizon Matchin](time_series/detecting_the_future_all-at-once_event_sequence_forecasting_with_horizon_matchin.md)**

:   提出DEF（Detection-based Event Forecasting），借鉴目标检测中DETR的匹配思想，通过匈牙利算法对齐预测与真实事件序列，实现高精度和高多样性的长程事件预测，在5个数据集上达到SOTA。

**[Finding Time Series Anomalies Using Granular-Ball Vector Data Description](time_series/finding_time_series_anomalies_using_granular-ball_vector_data_description.md)**

:   提出 Granular-ball One-Class Network (GBOC)，通过在潜在空间中自适应构建密度引导的粒球向量数据描述 (GVDD)，取代传统聚类或单一超球体假设，实现对时间序列正常行为的灵活建模和鲁棒异常检测。

**[Freqcycle A Multi-Scale Time-Frequency Analysis Method For Time Series Forecasti](time_series/freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)**

:   提出FreqCycle框架，通过FECF模块显式学习共享周期模式、SFPL模块增强中高频能量占比，并扩展为MFreqCycle处理耦合多周期性，在7个基准上达到SOTA性能与效率的最优平衡。

**[Gaico A Deployed And Extensible Framework For Evaluating Diverse And Multimodal ](time_series/gaico_a_deployed_and_extensible_framework_for_evaluating_diverse_and_multimodal_.md)**

:   提出GAICo（Generative AI Comparator），一个已部署的、可扩展的开源Python库，为文本、结构化数据（规划序列、时间序列）和多媒体（图像、音频）提供统一的基于参考的评估框架，支持多模型比较、可视化与报告生成。

**[Harmonic Dataset Distillation For Time Series Forecasting](time_series/harmonic_dataset_distillation_for_time_series_forecasting.md)**

:   提出HDT（Harmonic Dataset Distillation for Time Series Forecasting），通过FFT将时间序列分解为正弦基底，在频域上通过谐波匹配（Harmonic Matching）对齐合成数据与原始数据的核心周期结构，实现强跨架构泛化和良好可扩展性的时间序列数据集蒸馏。

**[Hn-Mvts Hypernetwork-Based Multivariate Time Series Forecasting](time_series/hn-mvts_hypernetwork-based_multivariate_time_series_forecasting.md)**

:   提出 HN-MVTS，利用超网络(HyperNetwork)为每个通道生成特定的最后一层权重，在通道独立(CI)和通道依赖(CD)之间取得平衡，作为即插即用模块可提升 DLinear、PatchTST、TSMixer 等多种主干模型的预测精度，且不增加推理时间。

**[Hydrodcm Hydrological Domain-Conditioned Modulation For Cross-Reservoir Inflow P](time_series/hydrodcm_hydrological_domain-conditioned_modulation_for_cross-reservoir_inflow_p.md)**

:   提出 HydroDCM，首次将域泛化(Domain Generalization)引入水文预测领域，通过空间元属性构建伪域标签指导对抗学习提取不变特征，再用 FiLM 适配器根据目标水库的地理信息调制特征，实现对未见水库的跨域入流预测。

**[Idealtsf Can Non-Ideal Data Contribute To Enhancing The Performance Of Time Seri](time_series/idealtsf_can_non-ideal_data_contribute_to_enhancing_the_performance_of_time_seri.md)**

:   提出 IdealTSF 框架，通过三阶段渐进式设计——负样本预训练（用稳定分布+多尺度噪声+结构删除模拟非理想数据）、正样本训练（混合平滑插值修复数据）、ECOS 优化器（对抗扰动引导到平坦极值）——使基础 attention 模型在含噪声/缺失的时序数据上获得约 10% 的性能提升。

**[Interpreting Fedspeak With Confidence A Llm-Based Uncertainty-Aware Framework Gu](time_series/interpreting_fedspeak_with_confidence_a_llm-based_uncertainty-aware_framework_gu.md)**

:   提出基于 LLM 的 uncertainty-aware 框架解读 Fedspeak（美联储语言）：通过货币政策传导路径的领域推理增强输入，引入 dynamic uncertainty decoding 模块量化预测置信度（Perceptual Uncertainty = Environmental Ambiguity × Cognitive Risk），在 FOMC 政策立场分析任务上达到 SOTA。

**[Loretta A Low Resource Framework To Poison Continuous Time Dynamic Graphs](time_series/loretta_a_low_resource_framework_to_poison_continuous_time_dynamic_graphs.md)**

:   提出 LoReTTA，一种无需代理模型的两阶段对抗投毒攻击框架：先通过 16 种时序重要性度量稀疏化高影响力边，再用保度数负采样算法替换对抗边，在 4 个数据集 × 4 个 TGNN 模型上平均降低 29.47% 性能，同时逃避 4 种异常检测系统且抵御 4 种防御方法。

**[M2Fmoe Multi-Resolution Multi-View Frequency Mixture-Of-Experts For Extreme-Adap](time_series/m2fmoe_multi-resolution_multi-view_frequency_mixture-of-experts_for_extreme-adap.md)**

:   提出 M2FMoE，通过傅里叶和小波双视角的频域混合专家建模常规与极端模式，结合跨视角共享频段分割器对齐两域语义、多分辨率自适应融合捕获多尺度信息、时序门控整合长短期特征，在 5 个水文极端事件数据集上无需极端事件标签即超越所有 SotA（含使用标签的方法），平均 RMSE 提升 22.3%。

**[Mask The Redundancy Evolving Masking Representation Learning For Multivariate Ti](time_series/mask_the_redundancy_evolving_masking_representation_learning_for_multivariate_ti.md)**

:   提出 EMTC 框架，通过 Importance-aware Variate-wise Masking (IVM) 动态屏蔽冗余时间戳，结合 Multi-Endogenous Views (MEV) 多视图生成与 cluster-guided contrastive learning，在 15 个 MTS 聚类基准上平均 F1 提升 4.85%。

**[Optimal Look-Back Horizon For Time Series Forecasting In Federated Learning](time_series/optimal_look-back_horizon_for_time_series_forecasting_in_federated_learning.md)**

:   提出联邦学习场景下时间序列预测的最优回看窗口（look-back horizon）理论框架，通过合成数据生成器（SDG）和内禀空间表示，将预测损失分解为贝叶斯不可约误差和近似误差，证明总损失关于窗口长度是单峰的，最小充分窗口为最优解。

**[Predicting The Future By Retrieving The Past](time_series/predicting_the_future_by_retrieving_the_past.md)**

:   提出 PFRP（Predicting the Future by Retrieving the Past），构建全局记忆库(GMB)存储历史模式，通过预测性对比学习训练编码器实现高效检索，将检索到的全局预测与任意局部预测模型动态融合，在 7 个数据集上平均提升 8.4% 的预测性能。

**[Recast Reliability-Aware Codebook Assisted Lightweight Time Series Forecasting](time_series/recast_reliability-aware_codebook_assisted_lightweight_time_series_forecasting.md)**

:   提出 ReCast，通过 patch 级向量量化将时间序列编码为离散嵌入，设计量化路径（预测规律结构）和残差路径（捕获不规则波动）的双路径架构，并引入基于分布鲁棒优化(DRO)的可靠性感知码本更新策略，在 8 个数据集上以轻量架构实现 SOTA 精度。

**[Revitalizing Canonical Pre-Alignment For Irregular Multivariate Time Series Fore](time_series/revitalizing_canonical_pre-alignment_for_irregular_multivariate_time_series_fore.md)**

:   首次论证了规范预对齐(CPA)在不规则多变量时间序列(IMTS)预测中不应被抛弃，提出 KAFNet 通过预卷积平滑、时间核聚合(TKA)压缩和频域线性注意力(FLA)三个模块解决 CPA 的效率问题，在 4 个 IMTS 数据集上实现 SOTA 精度，同时参数量减少 7.2 倍、训练推理加速 8.4 倍。

**[Scaling Llm Speculative Decoding Non-Autoregressive Forecasting In Large-Batch S](time_series/scaling_llm_speculative_decoding_non-autoregressive_forecasting_in_large-batch_s.md)**

:   提出 SpecFormer，一种融合单向和双向注意力的非自回归草稿模型架构，在大批次推理场景下通过降低对复杂前缀树的依赖、减少位置相关参数，实现了对 LLM 推理的一致性加速。

**[Seldon Supernova Explosions Learned By Deep Ode Networks](time_series/seldon_supernova_explosions_learned_by_deep_ode_networks.md)**

:   提出SELDON，一种结合masked GRU-ODE编码器、隐式Neural ODE传播器和可解释高斯基函数解码器的连续时间VAE，用于稀疏、不规则采样的天文光变曲线预测，在仅观测20%数据时即可超越基线方法做出准确的多波段通量预测。

**[Sonnet Spectral Operator Neural Network For Multivariable Time Series Forecastin](time_series/sonnet_spectral_operator_neural_network_for_multivariable_time_series_forecastin.md)**

:   提出 Sonnet，通过可学习小波变换将输入映射到时频域，引入基于谱相干性的多变量注意力（MVCA）建模变量间依赖关系，并利用 Koopman 算子进行稳定的时间演化预测，在 47 个预测任务中的 34 个取得最优，平均 MAE 降低 2.2%。

**[Task-Aware Retrieval Augmentation For Dynamic Recommendation](time_series/task-aware_retrieval_augmentation_for_dynamic_recommendation.md)**

:   提出 TarDGR 框架，通过任务感知的评估机制自动构建训练数据，训练 Graph Transformer 来评估历史子图的任务相关性，在推理时检索并融合任务相关子图以增强推荐的时序泛化能力。

**[Towards Non-Stationary Time Series Forecasting With Temporal Stabilization And F](time_series/towards_non-stationary_time_series_forecasting_with_temporal_stabilization_and_f.md)**

:   提出 DTAF 双分支框架，通过时域的非平稳 MoE 滤波器提取并去除异质非平稳模式、频域的频谱差分追踪频率漂移，并通过双分支注意力融合两个域的互补信息，实现鲁棒的非平稳时间序列预测。

**[Transparent Networks For Multivariate Time Series](time_series/transparent_networks_for_multivariate_time_series.md)**

:   提出 GATSM（Generalized Additive Time Series Model），一种透明的时间序列神经网络模型，通过共享权重的特征网络学习特征表示并用带掩码的多头注意力捕捉时序模式，在保持完全可解释性的同时达到与 Transformer 等黑箱模型可比的性能。

**[Urban Incident Prediction With Graph Neural Networks Integrating Government Rati](time_series/urban_incident_prediction_with_graph_neural_networks_integrating_government_rati.md)**

:   提出 URBAN（多视图多输出GNN模型），联合利用稀疏但无偏的政府检查评级数据和密集但有偏的众包报告数据来预测城市事件的真实潜在状态，在纽约市960万+报告和100万+检查数据上验证，预测相关性比仅用报告数据高5.3倍。

---

## 🤖 具身智能 { #robotics }

**[A Computable Game-Theoretic Framework For Multi-Agent Theory Of Mind](robotics/a_computable_game-theoretic_framework_for_multi-agent_theory_of_mind.md)**

:   提出基于 Poisson 认知层次（cognitive hierarchy）的博弈论框架，通过 Gamma-Poisson 共轭贝叶斯更新实现可计算的多智能体 Theory of Mind，在避免 POMDP 不可判定性的同时支持递归式有限理性决策与在线信念修正。

**[Adaptive Theory Of Mind For Llm-Based Multi-Agent Coordination](robotics/adaptive_theory_of_mind_for_llm-based_multi-agent_coordination.md)**

:   提出自适应心智理论智能体(A-ToM)，将ToM阶数对齐建模为在线专家建议问题，通过FTL或Hedge算法实时估计伙伴的ToM阶数并动态调整自身推理深度，在重复矩阵博弈、网格导航和Overcooked等4类任务上实现鲁棒的零样本多智能体协作。

**[Affordance-Guided Coarse-To-Fine Exploration For Base Placem](robotics/affordance-guided_coarse-to-fine_exploration_for_base_placem.md)**

:   针对开放词汇移动操控中机器人基座选位问题，提出一种零样本框架，通过构建跨模态表征（Affordance RGB + Obstacle Map+）将语义affordance线索投射到障碍物地图上，再用粗到细迭代优化平衡语义和几何约束，在5个操控任务上达到85%成功率，大幅超越几何规划器和纯VLM方法。

**[Attention As Binding A Vector-Symbolic Perspective On Transformer Reasoning](robotics/attention_as_binding_a_vector-symbolic_perspective_on_transformer_reasoning.md)**

:   本文提出将Transformer自注意力机制重新解释为向量符号架构(VSA)中的软绑定/解绑定算子——Query/Key定义角色空间、Value编码填充项、注意力权重实现可微解绑定、残差连接实现叠加——从而以代数视角统一解释LLM在符号推理中的能力与脆弱性，并提出显式绑定头、超维记忆层等VSA启发的架构改进方向。

**[Causal Inference Under Threshold Manipulation Bayesian Mixtu](robotics/causal_inference_under_threshold_manipulation_bayesian_mixtu.md)**

:   提出 BMTM/HBMTM 贝叶斯混合模型框架，在消费者策略性操纵消费额以达到奖励阈值的场景下，通过将观测分布拆解为 bunching 与 non-bunching 两个子分布，准确估计阈值因果效应及跨子群的异质性处理效应。

**[Characterizing Ai Manipulation Risks In Brazilian Youtube Climate Discourse](robotics/characterizing_ai_manipulation_risks_in_brazilian_youtube_climate_discourse.md)**

:   通过心理语言学框架分析巴西 YouTube 上 22.6 万条气候变化视频和 275 万条评论，揭示情感/道德修辞显著驱动用户互动，并展示微调 LLM 可自动生成高互动性的气候否认评论，警示生成式 AI 在舆论操控中的潜在风险。

**[Cross Modal Fine-Grained Alignment Via Granularity-Aware And Region-Uncertain Mo](robotics/cross_modal_fine-grained_alignment_via_granularity-aware_and_region-uncertain_mo.md)**

:   提出 GRM 框架，通过模态内显著性/粒度感知适配器和基于高斯混合的区域级不确定性建模，实现鲁棒的细粒度图文对齐，在 Flickr30K 和 MS-COCO 上取得 SOTA。

**[Do Llms Really Struggle At Nl-Fol Translation Revealing Their Strengths Via A No](robotics/do_llms_really_struggle_at_nl-fol_translation_revealing_their_strengths_via_a_no.md)**

:   本文批判性审视了现有NL到一阶逻辑(FOL)翻译的评估方法（FOLIO和MALLS），揭示其数据集与评估协议的根本缺陷，提出了一种将翻译任务分解为本体提取(OE)和逻辑翻译(LT)、并辅以"最相似选择"和"排序"子任务的新型基准测试策略，实验表明对话式LLM（o3-mini、GPT-4o-mini、Qwen3系列）展现出强大的NL-FOL翻译能力与真正的逻辑语义理解，而嵌入式模型表现显著较差。

**[Evoempirbench Dynamic Spatial Reasoning With Agent-Expver](robotics/evoempirbench_dynamic_spatial_reasoning_with_agent-expver.md)**

:   提出 EvoEmpirBench（EEB），包含两个动态交互式 benchmark（局部可观测迷宫导航 + 消消乐），以及 Agent-ExpVer 三智能体在线学习框架（GeoLink 交互 + InsightForce 经验抽象 + TruthWeaver 知识管理），通过"经验→验证→真理归纳"的认知循环实现无参数更新的持续策略进化，使 GPT-4.1 成功率提升 5.6%、Qwen-32B 提升 29%。

**[From Passive Perception To Active Memory A Weakly Supervised Image Manipulation ](robotics/from_passive_perception_to_active_memory_a_weakly_supervised_image_manipulation_.md)**

:   提出 BoxPromptIML，一种基于粗粒度框标注的弱监督图像篡改定位（IML）框架，通过冻结的 SAM 教师模型将粗糙边界框转化为高质量伪掩码，结合记忆引导门控融合模块（MGFM）训练轻量级学生模型，仅需 7 秒/张的标注成本即可媲美甚至超越全监督方法。

**[From Woofs To Words Towards Intelligent Robotic Guide Dogs With Verbal Communica](robotics/from_woofs_to_words_towards_intelligent_robotic_guide_dogs_with_verbal_communica.md)**

:   本文提出了一套面向导盲机器犬的对话系统，利用 LLM 和任务规划器实现 **计划语言化（Plan Verbalization）** 和 **场景语言化（Scene Verbalization）**，通过多轮自然语言对话辅助视障用户完成导航决策，并通过真人用户研究和仿真实验验证了系统的有效性。

**[Gaming The Answer Matcher Examining The Impact Of Text Manipulation On Automated](robotics/gaming_the_answer_matcher_examining_the_impact_of_text_manipulation_on_automated.md)**

:   本文系统性地测试了三种文本操控策略（冗长、策略性多答案嵌入、正确答案前置+矛盾）对 LLM 答案匹配评判器的影响，发现这些操控**不会提升分数甚至降低分数**，且二值评分比连续评分更鲁棒，证明答案匹配是一种对低成本文本操控具有鲁棒性的评估方法。

**[Grim Task-Oriented Grasping With Conditioning On Generative Examples](robotics/grim_task-oriented_grasping_with_conditioning_on_generative_examples.md)**

:   本文提出 GRIM（Grasp Re-alignment via Iterative Matching），一种**免训练**的任务导向抓取（TOG）框架，通过 **retrieve–align–transfer** 流水线结合视频生成模型和多源记忆库，利用基于 DINO 特征的语义 3D 对齐实现跨物体的功能性抓取迁移，仅用 210 个记忆实例即超越了在 379K 样本上训练的 GraspMolmo。

**[Human-Centric Open-Future Task Discovery Formulation Benchmark And Scalable Tree](robotics/human-centric_open-future_task_discovery_formulation_benchmark_and_scalable_tree.md)**

:   本文提出并形式化了**人类中心开放未来任务发现（HOTD）**问题——在人类意图并发且动态变化的场景中，发现那些在多种可能未来中都能减少人类负担的任务。同时构建了 HOTD-Bench 基准（2K+ 真实视频），并提出 **CMAST** 框架（协作多智能体搜索树），通过多智能体系统和可扩展搜索树显著超越现有 LMM 方法。

**[Human Cognitive Biases In Explanation-Based Interaction The Case Of Within And B](robotics/human_cognitive_biases_in_explanation-based_interaction_the_case_of_within_and_b.md)**

:   本文通过两项大规模用户研究（总计 713 名参与者）系统评估了**顺序效应**（order effect）对解释性交互学习（XIL）的影响，发现顺序效应对用户反馈质量的影响**有限且不一致**，且仅在 session 内（而非 session 间）有显著但微弱的影响——总体结论是顺序效应不构成 XIL 实际应用的重大障碍。

**[Iseal Encrypted Fingerprinting For Reliable Llm Ownership Verification](robotics/iseal_encrypted_fingerprinting_for_reliable_llm_ownership_verification.md)**

:   提出 iSeal——首个在模型窃取者完全控制推理过程的黑盒场景下仍能可靠验证 LLM 所有权的主动指纹方法，通过外部加密编码器 + RSC 纠错 + 相似度匹配三重机制，在 12 个 LLM、10+ 种攻击下均保持 100% 指纹成功率（FSR），而已有方法降至 0%。

**[Laf-Grpo In-Situ Navigation Instruction Generation For The Visually Impaired Via](robotics/laf-grpo_in-situ_navigation_instruction_generation_for_the_visually_impaired_via.md)**

:   提出 LaF-GRPO 框架，利用 LLM 模拟视障用户对导航指令的响应作为奖励信号，通过 GRPO 后训练 VLM 来生成更精确、更安全的视障导航指令，并构建了 27k 样本的 NIG4VI 基准数据集。

**[More Than Irrational Modeling Belief-Biased Agents](robotics/more_than_irrational_modeling_belief-biased_agents.md)**

:   提出一种计算理性（CR）用户模型框架，将人类看似"不理性"的行为解释为在有限记忆（信念偏差）下的最优决策，通过嵌套粒子滤波（NPF）在线推断用户的潜在记忆界限参数 $\theta$ 和偏差信念状态 $\tilde{b}$，PM误差在45步内降低90%，并在辅助POMDP中展示自适应AI助手策略。

**[Neural Graph Navigation For Intelligent Subgraph Matching](robotics/neural_graph_navigation_for_intelligent_subgraph_matching.md)**

:   提出 NeuGN（Neural Graph Navigation）框架，首次将生成式神经导航集成到子图匹配的核心枚举阶段，通过 QSExtractor 提取查询图结构信号 + GGNavigator 将暴力枚举转为结构感知的候选节点优先排序，在保证完备性的同时将 First Match Steps 最高减少 98.2%。

**[Panonav Mapless Zero-Shot Object Navigation With Panoramic Scene Parsing And Dyn](robotics/panonav_mapless_zero-shot_object_navigation_with_panoramic_scene_parsing_and_dyn.md)**

:   提出 PanoNav，一个仅使用 RGB 图像的无地图零样本目标导航框架，通过全景场景解析（Panoramic Scene Parsing）释放 MLLM 的空间推理能力，并引入动态有界记忆队列（Dynamic Bounded Memory Queue）避免局部死锁问题。

**[Realistic Synthetic Household Data Generation At Scale](robotics/realistic_synthetic_household_data_generation_at_scale.md)**

:   提出一个基于 LLM 的双向耦合生成框架，通过人物画像驱动环境生成、环境语义引导行为生成的迭代循环过程，大规模生成包含家庭环境配置、人类行为和人机交互的合成数据集，用于训练家用机器人。

**[Recursive Visual Imagination And Adaptive Linguistic Grounding For Vision Langua](robotics/recursive_visual_imagination_and_adaptive_linguistic_grounding_for_vision_langua.md)**

:   提出基于隐式场景表征（ISR）的VLN策略，通过递归视觉想象（RVI）将历史轨迹压缩为固定大小的紧凑神经网格学习高层场景先验，并通过自适应语言对齐（ALG）将指令的不同语义组件与不同网格精细匹配，在R2R-CE和ObjectNav两个连续环境导航任务上取得SOTA。

**[Robust Out-Of-Order Retrieval For Grid-Based Storage At Maximum Capacity](robotics/robust_out-of-order_retrieval_for_grid-based_storage_at_maximum_capacity.md)**

:   针对满载 2D 网格存储系统中检索顺序不确定的问题，提出 k-bounded perturbation 不确定性模型，证明 Θ(k) 列宽是零重定位的充要条件，并给出高效鲁棒存储求解器与贪心检索策略，当 k ≤ 0.5c 时几乎消除重定位，k 到达 c 时仍减少 50%+ 重定位。

**[Shadows In The Code Exploring The Risks And Defenses Of Llm-](robotics/shadows_in_the_code_exploring_the_risks_and_defenses_of_llm-.md)**

:   首次系统分析 LLM 多 Agent 软件开发系统（ChatDev/MetaGPT/AgentVerse）的安全风险：提出 IMBIA 攻击框架覆盖两种威胁场景（恶意用户+良性Agent / 良性用户+恶意Agent）和 12 种恶意行为（5 大恶意软件家族），攻击成功率高达 93%（ChatDev），并设计 Adv-IMBIA 对抗性防御将 ASR 降低 40-73%。

**[Spatialactor Exploring Disentangled Spatial Representations For Robust Robotic M](robotics/spatialactor_exploring_disentangled_spatial_representations_for_robust_robotic_m.md)**

:   提出 SpatialActor 框架，通过将语义与几何表征显式解耦，并设计语义引导几何模块（SGM）自适应融合深度噪声特征与预训练深度估计专家先验、以及空间 Transformer（SPT）编码低级空间位置线索，在 RLBench 50+ 任务上达到 87.4% 成功率（SOTA +6.0%），且在重噪声条件下比 RVT-2 高出 19.4%。

**[To Align Or Not To Align Strategic Multimodal Representation Alignment For Optim](robotics/to_align_or_not_to_align_strategic_multimodal_representation_alignment_for_optim.md)**

:   通过引入可控对比学习模块系统调节对齐强度 $\lambda$，结合偏信息分解(PID)框架量化模态间冗余-独特-协同信息结构，揭示显式对齐的效用高度依赖于数据特性：冗余主导时对齐有益，独特主导时有害，混合场景存在最优 $\lambda^*$。

**[Touchformer A Robust Transformer-Based Framework For Multimodal Material Percept](robotics/touchformer_a_robust_transformer-based_framework_for_multimodal_material_percept.md)**

:   提出 TouchFormer，一个鲁棒的多模态融合框架，通过模态自适应门控（MAG）、模态内/模态间注意力机制和跨实例嵌入正则化（CER）三个互补模块，在视觉受损条件下实现可靠的材质感知，并在火灾场景机器人分拣实验中验证有效性。

**[Towards Reinforcement Learning From Neural Feedback Mapping ](robotics/towards_reinforcement_learning_from_neural_feedback_mapping_.md)**

:   提出 NEURO-LOOP 框架，利用 fNIRS（功能性近红外光谱）脑信号作为隐式神经反馈评估 RL agent 表现，发布 25 名被试 × 3 领域 × 6 条件的 fNIRS 数据集，分类 F1 达 67%（二分类）/ 46%（多分类），跨被试 fine-tuning 分别提升 17% 和 41%，奠定 Reinforcement Learning from Neural Feedback (RLNF) 基础。

**[Unintended Misalignment From Agentic Fine-Tuning Risks And M](robotics/unintended_misalignment_from_agentic_fine-tuning_risks_and_m.md)**

:   本文揭示了在良性 Agent 数据上微调 LLM 会导致意外的安全对齐偏移（攻击成功率增加 32-38%），并提出 PING（Prefix Injection Guard）——通过迭代生成+评估自然语言前缀来引导微调后的 Agent 拒绝有害请求，平均提升拒绝率 66%（Web）和 44%（代码），同时保持任务性能（仅降 1.8%）。

**[Urbannav Learning Language-Guided Urban Navigation From Web-Scale Human Trajecto](robotics/urbannav_learning_language-guided_urban_navigation_from_web-scale_human_trajecto.md)**

:   提出 UrbanNav，利用网络规模的城市步行视频（YouTube 上 1500+ 小时、300 万条指令-轨迹-地标三元组），通过自动化标注管线和鲁棒过滤机制训练语言引导的城市导航策略，在真实世界部署中达到 83.3% 的导航成功率。

---

## ✂️ 语义分割 { #segmentation }

**[3Dteethsam Taming Sam2 For 3D Teeth Segmentation](segmentation/3dteethsam_taming_sam2_for_3d_teeth_segmentation.md)**

:   将SAM2基础模型迁移到3D牙齿分割任务，通过多视角渲染将3D mesh转为2D图像、设计三个轻量适配器（Prompt生成器、Mask精化器、Mask分类器）和可变形全局注意力插件（DGAP）来解决自动提示、边界精化和语义分类问题，在Teeth3DS上以91.90% T-mIoU刷新SOTA。

**[A2Lc Active And Automated Label Correction For Semantic Segm](segmentation/a2lc_active_and_automated_label_correction_for_semantic_segm.md)**

:   提出 A²LC 框架，在传统主动标签校正（人工逐一纠错）的基础上增加一个自动校正阶段（Label Correction Module），利用标注员的反馈自动修正相似的错误mask，并设计自适应平衡采集函数缓解类别不平衡，在 Cityscapes 上仅用 20% 预算即超越前 SOTA，同等预算下 mIoU 提升 27.23%。

**[Adaptive Morph-Patch Transformer For Aortic Vessel Segmentat](segmentation/adaptive_morph-patch_transformer_for_aortic_vessel_segmentat.md)**

:   提出 Morph-Patch Transformer (MPT)，通过基于速度场的自适应 patch 划分策略生成形态感知 patch（保持血管拓扑完整性），并引入语义聚类注意力（SCA）动态聚合语义相似 patch 的特征，在 AVT、AortaSeg24 和 TBAD 三个主动脉分割数据集上均达 SOTA。

**[Breaking The Stealth-Potency Trade-Off In Clean-Image Backdoors With Generative ](segmentation/breaking_the_stealth-potency_trade-off_in_clean-image_backdoors_with_generative_.md)**

:   提出 Generative Clean-Image Backdoors (GCB)，通过 Conditional InfoGAN (C-InfoGAN) 自动发现图像中天然存在且与分类任务无关的特征作为后门触发器，以极低投毒率（≤0.5%）实现高攻击成功率（≥90% ASR）且几乎不损伤干净准确率（CA drop ≤1%），首次打破了 clean-image backdoor 中隐蔽性与攻击力的固有矛盾。

**[Bridging Granularity Gaps Hierarchical Semantic Learning For Cross-Domain Few-Sh](segmentation/bridging_granularity_gaps_hierarchical_semantic_learning_for_cross-domain_few-sh.md)**

:   提出 HSL 框架，通过双重风格随机化 (DSR)、层次语义挖掘 (HSM) 和原型置信度调制阈值 (PCMT) 三个模块，解决跨域少样本分割中源域和目标域之间的**分割粒度差异**问题，在四个目标域数据集上达到 SOTA。

**[Causal-Tune Mining Causal Factors From Vision Foundation Mod](segmentation/causal-tune_mining_causal_factors_from_vision_foundation_mod.md)**

:   提出Causal-Tune，从因果视角分析VFM特征中的artifacts，利用DCT频域分解+高斯带通滤波分离因果/非因果因素，结合因果感知可学习token在频域精化特征，在Cityscapes→ACDC跨域分割中平均提升+2.4% mIoU（Snow场景+4.8%），仅需单卡RTX3090/14GB训练。

**[Do We Need Perfect Data Leveraging Noise For Domain Generalized Segmentation](segmentation/do_we_need_perfect_data_leveraging_noise_for_domain_generalized_segmentation.md)**

:   提出 FLEX-Seg 框架，将扩散模型合成数据中图像与语义掩码之间固有的**边界不对齐**(misalignment)转化为学习鲁棒表示的机会，通过粒度自适应原型 (GAP)、不确定性边界强调 (UBE) 和难度感知采样 (HAS) 三个模块，在域泛化语义分割任务上取得 SOTA。

**[Eagle Episodic Appearance- And Geometry-Aware Memory For Unified 2D-3D Visual Qu](segmentation/eagle_episodic_appearance-_and_geometry-aware_memory_for_unified_2d-3d_visual_qu.md)**

:   提出 EAGLE 框架，借鉴鸟类记忆巩固机制，通过外观感知元学习记忆 (AMM) 驱动的分割分支与几何感知定位记忆 (GLM) 驱动的跟踪分支协同工作，结合 VGGT 实现高效的 2D-3D 统一视觉查询定位，在 Ego4D-VQ 基准上达到 SOTA。

**[Empowering Dino Representations For Underwater Instance Segmentation Via Aligner](segmentation/empowering_dino_representations_for_underwater_instance_segmentation_via_aligner.md)**

:   首次将 DINOv2 引入水下实例分割任务，通过 AquaStyle Aligner（傅里叶频域风格注入）和 ObjectPrior Prompter（二值掩码先验提示）两个模块实现高效领域适配，在 UIIS 和 USIS10K 数据集上以更少参数大幅超越 SAM 基方法。

**[Empowering Semantic-Sensitive Underwater Image Enhancement With Vlm](segmentation/empowering_semantic-sensitive_underwater_image_enhancement_with_vlm.md)**

:   利用 VLM 生成空间语义引导图，通过 cross-attention 注入和语义对齐损失的双重引导机制，赋予水下图像增强网络语义感知能力，使增强结果同时有利于人类感知和下游检测/分割任务。

**[From Attribution To Action Jointly Aligning Predictions And Explanations](segmentation/from_attribution_to_action_jointly_aligning_predictions_and_explanations.md)**

:   提出 ALIGN 框架，通过联合训练可学习掩码生成器（masker）和分类器，迭代对齐模型归因图与任务相关区域掩码，同时提升预测准确性和可解释性，在 VLCS 和 Terra Incognita 域泛化基准上超越 6 个强基线。

**[Generalizable Slum Detection From Satellite Imagery With Mixture-Of-Experts](segmentation/generalizable_slum_detection_from_satellite_imagery_with_mixture-of-experts.md)**

:   提出 GRAM（Generalized Region-Aware Mixture-of-Experts），一个两阶段测试时自适应框架：第一阶段用 MoE 架构在12个城市的百万级卫星图像上训练区域特化专家，第二阶段通过跨区域预测一致性筛选可靠伪标签进行自训练，实现对未见非洲城市的贫民窟分割泛化。

**[Guideline-Consistent Segmentation Via Multi-Agent Refinement](segmentation/guideline-consistent_segmentation_via_multi-agent_refinement.md)**

:   提出一个免训练的多智能体框架，通过 Worker（分割执行）和 Supervisor（指南验证）的迭代循环，配合 RL 自适应停止策略，实现严格遵循复杂文本指南的语义分割，在 Waymo 和 ReasonSeg 上分别超越 SOTA 8.61 和 5.5 gIoU。

**[Infoclip Bridging Vision-Language Pretraining And Open-Vocab](segmentation/infoclip_bridging_vision-language_pretraining_and_open-vocab.md)**

:   提出InfoCLIP，基于信息论视角设计信息瓶颈压缩和互信息蒸馏两个目标，在CLIP微调过程中去除预训练pixel-text对齐中的噪声并保留语义对齐知识，在6个开放词汇语义分割测试集上全面超越SOTA（A-847: 16.6, A-150: 38.5, PC-59: 63.5 mIoU），且仅增加0.53M参数和极少计算开销。

**[Jodiffusion Jointly Diffusing Image With Pixel-Level Annotations For Semantic Se](segmentation/jodiffusion_jointly_diffusing_image_with_pixel-level_annotations_for_semantic_se.md)**

:   提出JoDiffusion框架，通过在潜在空间中联合扩散图像与像素级标注掩码，首次实现仅基于文本提示同时生成语义一致的图像-标注对，在Pascal VOC、COCO和ADE20K上显著超越现有Image2Mask和Mask2Image方法。

**[Lwganet Addressing Spatial And Channel Redundancy In Remote Sensing Visual Tasks](segmentation/lwganet_addressing_spatial_and_channel_redundancy_in_remote_sensing_visual_tasks.md)**

:   针对遥感图像中的空间冗余（大面积均质背景）和通道冗余（极端尺度变化导致单一特征空间低效）问题，提出 LWGANet 轻量化骨干，通过 Top-K 稀疏全局特征交互（TGFI）和异构分组注意力（LWGA）模块实现高效多尺度特征表示，在 12 个数据集 4 类遥感任务上达到 SOTA。

**[Multigranular Evaluation For Brain Visual Decoding](segmentation/multigranular_evaluation_for_brain_visual_decoding.md)**

:   提出BASIC多粒度评估框架，从结构（四级分割mask匹配）和语义（MLLM提取对象/属性/关系图的精确率-召回率-F1）两个轴统一评估脑视觉解码质量，横跨fMRI/EEG × Image/Video/3D六种模态组合，解决现有指标饱和、缺乏神经科学基础和细粒度诊断能力的问题。

**[Otter Mitigating Background Distractions Of Wide-Angle Few-Shot Action Recogniti](segmentation/otter_mitigating_background_distractions_of_wide-angle_few-shot_action_recogniti.md)**

:   针对广角视频中小样本动作识别的背景干扰问题（主体占比小、时序关系退化），提出基于增强 RWKV 的 Otter 框架，通过复合分割模块（CSM）突出主体和时序重建模块（TRM）恢复时序关系，在 SSv2/Kinetics/UCF101/HMDB51 等基准上达到 SOTA。

**[Rs2-Sam2 Customized Sam2 For Referring Remote Sensing Image Segmentation](segmentation/rs2-sam2_customized_sam2_for_referring_remote_sensing_image_segmentation.md)**

:   提出 RS2-SAM2 框架，通过双向层次融合模块将文本信息注入 SAM2 图像编码过程，并设计伪掩码提示生成器为 SAM2 提供密集提示，在遥感指称分割任务上取得 SOTA。

**[Rsvg-Zeroov Exploring A Training-Free Framework For Zero-Shot Open-Vocabulary Vi](segmentation/rsvg-zeroov_exploring_a_training-free_framework_for_zero-shot_open-vocabulary_vi.md)**

:   提出 RSVG-ZeroOV，一个免训练框架，通过"概览-聚焦-进化"三阶段策略融合 VLM 的交叉注意力图和扩散模型的自注意力图，实现零样本开放词汇遥感视觉定位。

**[Sam-Daq Segment Anything Model With Depth-Guided Adaptive Queries For Rgb-D Vide](segmentation/sam-daq_segment_anything_model_with_depth-guided_adaptive_queries_for_rgb-d_vide.md)**

:   提出 SAM-DAQ，通过深度引导并行适配器（DPA）和查询驱动时序记忆（QTM）模块将 SAM2 适配到 RGB-D 视频显著性检测任务，解决了手动提示依赖、高显存消耗和计算负担三大挑战。

**[Saq-Sam Semantically-Aligned Quantization For Segment Anything Model](segmentation/saq-sam_semantically-aligned_quantization_for_segment_anything_model.md)**

:   提出 SAQ-SAM，从语义对齐视角改进 SAM 的后训练量化（PTQ），通过感知一致性裁剪（PCC）处理掩码解码器中的极端异常值，并用提示感知重建（PAR）保持图像-提示交互的语义对齐。

**[Segment And Matte Anything In A Unified Model](segmentation/segment_and_matte_anything_in_a_unified_model.md)**

:   提出SAMA——一种SAM的轻量级扩展框架，通过多视图局部编码器(MVLE)捕获细粒度局部特征、局部化适配器(Local-Adapter)将局部细节注入解码过程，以及双任务预测头，仅增加1.8%参数即可在统一模型中同时实现高质量交互式分割和Alpha Matting，在DIS-5K和多个Matting基准上达到SOTA。

**[Segment Anything Across Shots A Method And Benchmark](segmentation/segment_anything_across_shots_a_method_and_benchmark.md)**

:   提出针对多镜头视频目标分割（MVOS）的 SAAS 方法和 Cut-VOS 基准，通过镜头切换模拟数据增强（TMA）、镜头切换检测与理解模块（TDM+TCH）、以及局部记忆库实现跨镜头鲁棒分割。

**[Ssr Semantic And Spatial Rectification For Clip-Based Weakly Supervised Segmenta](segmentation/ssr_semantic_and_spatial_rectification_for_clip-based_weakly_supervised_segmenta.md)**

:   提出语义与空间双重校正框架SSR，通过跨模态原型对比学习（CMPA）解决CLIP模态间语义不对齐导致的非目标前景过度激活问题，以及超像素引导校正（SGC）解决仿射传播中背景过度激活问题，在PASCAL VOC和MS COCO上全面超越单阶段和多阶段SOTA方法。

**[Symmetrical Flow Matching Unified Image Generation Segmentation And Classificati](segmentation/symmetrical_flow_matching_unified_image_generation_segmentation_and_classificati.md)**

:   提出对称流匹配（SymmFlow），将语义分割、分类和图像生成统一到单一模型中，通过对称学习目标联合建模正反向流变换，仅需25步推理即在语义图像合成上达到SOTA（CelebAMask-HQ FID 11.9，COCO-Stuff FID 7.0），同时在分割和分类上取得有竞争力的结果。

**[Target Refocusing Via Attention Redistribution For Open-Vocabulary Semantic Segm](segmentation/target_refocusing_via_attention_redistribution_for_open-vocabulary_semantic_segm.md)**

:   从可解释性角度系统研究CLIP内部机制，发现"分心"现象（distraction）——CLIP在深层将大量注意力资源分配给与目标无关的token，提出免训练的RF-CLIP方法通过注意力重分配将被分散的资源重新聚焦到目标区域，在8个基准上达到SOTA性能并保持推理高效。

**[Text-Guided Controllable Diffusion For Realistic Camouflage Images Generation](segmentation/text-guided_controllable_diffusion_for_realistic_camouflage_images_generation.md)**

:   提出CT-CIG，首个文本引导的可控伪装图像生成方法。利用VLM设计伪装揭示对话机制（CRDM）生成高质量文本提示，结合轻量控制网络和频率交互精炼模块（FIRM），在Stable Diffusion框架上生成逻辑合理、纹理真实的伪装图像，开创了Text-guided CIG新范式。

**[Towards Affordance-Aware Robotic Dexterous Grasping With Human-Like Priors](segmentation/towards_affordance-aware_robotic_dexterous_grasping_with_human-like_priors.md)**

:   提出AffordDex，一个两阶段框架：第一阶段通过模仿学习预训练人类手部运动先验（自然的运动轨迹），第二阶段通过残差模块和VLM引导的负可供性分割（NAA）进行强化学习精炼，实现既像人类一样自然、又功能正确的灵巧机器人抓取（如避开刀刃抓握刀柄），在多个泛化级别上显著超越SOTA。

**[Vista Scene-Aware Optimization For Streaming Video Question Answering Under Post](segmentation/vista_scene-aware_optimization_for_streaming_video_question_answering_under_post.md)**

:   Vista 提出了一种场景感知的流式视频问答框架，通过将流式视频动态分割为语义连贯的场景单元，对每个场景进行时空压缩并卸载到 CPU，在用户提问时选择性召回最相关的场景，实现了在低 GPU 内存占用和低延迟下的高精度视频问答。

---

## 💬 LLM/NLP { #llm_nlp }

**[An Invariant Latent Space Perspective On Language Model Inve](llm_nlp/an_invariant_latent_space_perspective_on_language_model_inve.md)**

:   提出不变潜空间假说(ILSH)，将LLM反演问题重新建模为复用LLM自身潜空间，设计Inv²A框架通过轻量级逆编码器将输出映射到去噪伪表示，再由冻结的LLM解码恢复隐藏prompt，在9个数据集上BLEU平均提升4.77%且仅需20%数据量即可达到可比性能。

**[Blue Teaming Function-Calling Agents](llm_nlp/blue_teaming_function-calling_agents.md)**

:   系统评估了四个开源function-calling LLM在三种攻击下的鲁棒性，并测试了八种防御方案的效果，揭示了当前模型默认不安全、防御方案在实际场景中仍难以部署的现状。

**[Control Illusion The Failure Of Instruction Hierarchies In Large Language Models](llm_nlp/control_illusion_the_failure_of_instruction_hierarchies_in_large_language_models.md)**

:   系统性揭示了当前 LLM 中 system/user 提示分离机制**无法有效建立指令优先级**，并发现预训练习得的社会层级先验（权威、专业、共识）比显式的 system/user 角色对模型行为有更强的控制力。

**[Conversational Learning Diagnosis Via Reasoning Multi-Turn Interactive Learning](llm_nlp/conversational_learning_diagnosis_via_reasoning_multi-turn_interactive_learning.md)**

:   提出 ParLD（Preview-Analyze-Reason 框架），通过多 Agent 协作实现对话式学习过程中学生认知状态的细粒度逐轮诊断，在性能预测上超越传统知识追踪方法 10%，并显著提升辅导效果。

**[Do Not Merge My Model Safeguarding Open-Source Llms Against Unauthorized Model M](llm_nlp/do_not_merge_my_model_safeguarding_open-source_llms_against_unauthorized_model_m.md)**

:   提出MergeBarrier，一种即插即用的防御方法，通过对注意力层施加正交投影、对FFN层进行激活函数展开重参数化，破坏受保护模型与同源模型之间的线性模态连通性（LMC），从而在不损失模型性能的前提下主动阻止未授权的模型合并。

**[Guess Or Recall Training Cnns To Classify And Localize Memorization In Llms](llm_nlp/guess_or_recall_training_cnns_to_classify_and_localize_memorization_in_llms.md)**

:   在 LLM 注意力权重上训练 CNN 来评估记忆化分类法与实际注意力机制的对齐程度，提出新的三类分类法（Guess/Recall/Non-Memorized），最小 F1 从 64.7% 提升至 89.0%，并定位了不同记忆类型分别依赖低层（Guess）和高层（Recall）注意力。

**[Icl-Router In-Context Learned Model Representations For Llm Routing](llm_nlp/icl-router_in-context_learned_model_representations_for_llm_routing.md)**

:   提出 ICL-Router，通过两阶段训练（查询重建 + ICL模型路由）将 LLM 的能力画像编码为 in-context 向量，实现可扩展的动态模型路由——新增模型无需重训路由器，在分布内和分布外任务上均达到 SOTA。

**[Identifying And Analyzing Performance-Critical Tokens In Large Language Models](llm_nlp/identifying_and_analyzing_performance-critical_tokens_in_large_language_models.md)**

:   通过representation-level和token-level两种消融实验，发现LLM在ICL中直接依赖的"性能关键token"是模板和停用词token（如"Answer:"），而非人类会关注的内容token（如实际文本），并揭示了LLM通过将内容信息聚合到这些关键token的表示中来间接利用内容。

**[Improving Sustainability Of Adversarial Examples In Class-Incremental Learning](llm_nlp/improving_sustainability_of_adversarial_examples_in_class-incremental_learning.md)**

:   提出SAE框架解决类增量学习（CIL）中对抗样本因域漂移而失效的问题，通过语义校正模块（CLIP+CIL模型联合引导）和过滤增强模块（去除语义混淆样本），使对抗样本在类别数增长9倍后仍保持攻击效果，平均攻击成功率提升31.28%。

**[Learning Spatial Decay For Vision Transformers](llm_nlp/learning_spatial_decay_for_vision_transformers.md)**

:   提出 Spatial Decay Transformer（SDT），首次将数据依赖的空间衰减机制从 1D 序列建模适配到 2D 视觉 Transformer，通过 Context-Aware Gating（CAG）生成动态的、内容相关的 patch 交互衰减强度，在 ImageNet-1K 分类和生成任务上一致超越 RMT 等强基线。

**[Loki Low-Damage Knowledge Implanting Of Large Language Models](llm_nlp/loki_low-damage_knowledge_implanting_of_large_language_models.md)**

:   提出LoKI，一种基于Transformer知识存储机制理解的参数高效微调方法，通过知识向量归因（KVA）评估FFN中各知识向量的贡献度，选择低贡献向量进行层均衡的知识植入，在获得强任务性能的同时显著缓解灾难性遗忘。

**[Loopllm Transferable Energy-Latency Attacks In Llms Via Repetitive Generation](llm_nlp/loopllm_transferable_energy-latency_attacks_in_llms_via_repetitive_generation.md)**

:   提出LoopLLM，一种通过诱导LLM进入重复生成模式来发起能耗延迟攻击的框架，利用重复诱导提示优化和token对齐的集成优化，在12个开源和2个商业LLM上实现超过90%最大输出长度的攻击效果，跨模型迁移性提升约40%。

**[Paretohqd Fast Offline Multiobjective Alignment Of Large Language Models Using P](llm_nlp/paretohqd_fast_offline_multiobjective_alignment_of_large_language_models_using_p.md)**

:   提出 ParetoHqD，将人类偏好表示为目标空间中的偏好方向（而非线性标量化），通过选取靠近 Pareto 前沿的高质量数据做两阶段 SFT，用仅 42% 的 GPU 时间实现优于 5 个基线的多目标 LLM 对齐效果。

**[Persistent Instability In Llms Personality Measurements Effects Of Scale Reasoni](llm_nlp/persistent_instability_in_llms_personality_measurements_effects_of_scale_reasoni.md)**

:   PERSIST 框架系统评估 25 个开源 LLM（1B-685B）在 200 万+响应上的人格测量稳定性，发现即使 400B+模型在 5 分制量表上仍有 SD>0.3 的不稳定性，且 CoT 推理悖论性地增加变异性同时降低困惑度，LLM 适配问卷与传统人类问卷表现出相似的不稳定性。

**[Position On Llm-Assisted Peer Review Addressing Reviewer Gap Through Mentoring A](llm_nlp/position_on_llm-assisted_peer_review_addressing_reviewer_gap_through_mentoring_a.md)**

:   本文作为立场论文，提出将LLM在同行评审中的角色从"自动生成审稿意见"转向"增强人类审稿能力"——通过LLM驱动的导师系统（三阶段培训+认证）和反馈系统（违规检测+证据反馈+可靠性测试）来缩小审稿质量差距。

**[Profuser Progressive Fusion Of Large Language Models](llm_nlp/profuser_progressive_fusion_of_large_language_models.md)**

:   提出ProFuser，通过双模式优势评估（训练模式Min-CE + 推理模式Reward Model投票）全面识别各源模型在不同维度的优势，再用渐进式融合策略（先推理模式→后训练模式的easy-to-hard课程）将异构LLM的互补能力整合到单个目标模型中，在知识/推理/安全6个基准上平均提升1.65%。

**[Promptmoe Generalizable Zero-Shot Anomaly Detection Via Visually-Guided Prompt M](llm_nlp/promptmoe_generalizable_zero-shot_anomaly_detection_via_visually-guided_prompt_m.md)**

:   PromptMoE 将提示学习从单体式（monolithic）范式转变为组合式（compositional）范式，通过视觉引导的混合专家（MoE）机制从可学习的语义原语库中动态组合实例自适应的正常/异常状态提示，在 15 个工业和医学数据集上实现 ZSAD SOTA。

**[Rectification Reimagined A Unified Mamba Model For Image Cor](llm_nlp/rectification_reimagined_a_unified_mamba_model_for_image_cor.md)**

:   从统一畸变矫正视角出发，提出 UniRect 框架，通过 Residual Progressive TPS 处理几何形变 + Residual Mamba Blocks 补偿退化，统一处理肖像校正、广角矩形化、拼接矩形化、旋转校正四种任务，并通过 Sparse MoE 实现 four-in-one 多任务学习，拼接矩形化 PSNR 提升 3.82 dB，旋转校正提升 0.87 dB。

**[Scalable And Accurate Graph Reasoning With Llm-Based Multi-Agents](llm_nlp/scalable_and_accurate_graph_reasoning_with_llm-based_multi-agents.md)**

:   提出 GraphAgent-Reasoner（GAR），受分布式图计算理论启发，将图问题分解为以节点为中心的子任务分配给多个 Agent，通过邻居消息传递协作求解，将 LLM 可处理的图规模从 100 个节点扩展到 1000 个，在多项式时间图推理任务上显著超越现有最佳方法。

**[Scaling Equitable Reflection Assessment In Education Via Large Language Models A](llm_nlp/scaling_equitable_reflection_assessment_in_education_via_large_language_models_a.md)**

:   提出一个由5个角色化GPT-4o Agent组成的零样本多Agent流水线，对学习者反思文本进行公平的量表评分并生成偏差感知的对话式反馈，在336篇反思上实现MAE=0.467、QWK=0.459的评分一致性和Q(g)=3.967的反馈质量。

**[Smart A Surrogate Model For Predicting Application Runtime In Dragonfly Systems](llm_nlp/smart_a_surrogate_model_for_predicting_application_runtime_in_dragonfly_systems.md)**

:   提出 Smart（Surrogate Model for Predicting Application RunTime），首次将 GNN 和 LLM（Time-LLM）融合用于 Dragonfly 互连网络中的应用迭代运行时预测，在 1,056 节点系统上 MAPE 最低达 1.78%（LAMMPS），推理时间仅 0.515 秒，相比原始仿真实现数量级加速。

**[Soft Filtering Guiding Zero-Shot Composed Image Retrieval With Prescriptive And ](llm_nlp/soft_filtering_guiding_zero-shot_composed_image_retrieval_with_prescriptive_and_.md)**

:   提出 SoFT，一个无需训练的即插即用重排序模块，利用多模态 LLM 从参考图像和修改文本中提取"必须包含"（prescriptive）和"必须避免"（proscriptive）双重文本约束，对零样本组合图像检索的候选结果进行软过滤重排序，同时构建了多目标三元组数据集流水线以改善评估。

**[Stem Efficient Relative Capability Evaluation Of Llms Through Structured Transit](llm_nlp/stem_efficient_relative_capability_evaluation_of_llms_through_structured_transit.md)**

:   提出 STEM 框架，通过识别同架构不同规模模型间的"显著转换样本"(STS)构建轻量级评估子集，实现对未知 LLM 能力的高效相对定位，在 100 样本下达到 100% 定位准确率，远超随机采样和贝叶斯方法。

**[Temple Incentivizing Temporal Understanding Of Video Large Language Models Via P](llm_nlp/temple_incentivizing_temporal_understanding_of_video_large_language_models_via_p.md)**

:   提出 TEMPLE，通过自动化的视频时间偏好数据生成管线（视频筛选→时间扰动→对比响应）和创新的 Progressive Pre-SFT Alignment 策略（课程学习 + DPO 先于 SFT），用少量自生成 DPO 数据显著提升 Video LLM 的时间推理能力，在 VideoMME、MLVU、Vinoground 等多个基准上一致改进。

**[Transmamba A Sequence-Level Hybrid Transformer-Mamba Language Model](llm_nlp/transmamba_a_sequence-level_hybrid_transformer-mamba_language_model.md)**

:   提出 TransMamba，一种序列级别的 Transformer-Mamba 混合架构，通过共享 QKV/CBx 参数和 Memory Converter 在不同 token 长度时动态切换 Attention 和 SSM，兼顾长短序列的效率。

**[Uncertainty Under The Curve A Sequence-Level Entropy Area Metric For Reasoning L](llm_nlp/uncertainty_under_the_curve_a_sequence-level_entropy_area_metric_for_reasoning_l.md)**

:   提出 Entropy Area Score (EAS)——通过单次前向传播积分 token 级预测熵来量化推理 LLM 的不确定性。EAS 无需外部模型或重复采样，与答案熵强相关（Pearson r=0.82），用于训练数据选择时比 Pass Rate 过滤多提升 1.2-2.3% Pass@1，是高效可解释的 LLM 不确定性工具。

**[Vision Transformers Are Circulant Attention Learners](llm_nlp/vision_transformers_are_circulant_attention_learners.md)**

:   发现 ViT 的自注意力内禁学习了 BCCB 模式，据此提出 Circulant Attention，通过 2D FFT 实现 $O(N\log N)$ 复杂度，在 ImageNet 分类、COCO 检测、ADE20K 分割上一致提升。

**[Vspo Validating Semantic Pitfalls In Ontology Via Llm-Based Cq Generation](llm_nlp/vspo_validating_semantic_pitfalls_in_ontology_via_llm-based_cq_generation.md)**

:   提出 VSPO 框架，通过构造"定义-公理"错位数据集并微调 LLaMA-3.1-8B-Instruct，生成能够验证本体语义陷阱（如 allValuesFrom 误用）的能力问题（CQ），精度和召回率分别超过 GPT-4.1 达 26% 和 28.2%。

---

## 💡 LLM推理 { #llm_reasoning }

**[A Reasoning Paradigm For Named Entity Recognition](llm_reasoning/a_reasoning_paradigm_for_named_entity_recognition.md)**

:   提出 ReasoningNER，将命名实体识别从"隐式模式匹配"转变为"显式推理"范式，通过三阶段流程（CoT数据构建→CoT微调→GRPO强化增强）让模型先推理再抽取实体，在零样本设定下F1超GPT-4达12.3个百分点，8B模型在CrossNER上达72.4平均F1。

**[Answering The Unanswerable Is To Err Knowingly Analyzing And](llm_reasoning/answering_the_unanswerable_is_to_err_knowingly_analyzing_and.md)**

:   系统分析大推理模型(LRM)面对不可回答数学题时的弃权失败现象，发现LRM内部有足够认知能力识别问题不可解（探针分类准确率>80%）但外部行为仍偏向强答，提出认知监控+推理时干预的两阶段方法，将弃权率从16-54%提升至60-92%且不损害可回答题的推理性能。

**[Arche A Novel Task To Evaluate Llms On Latent Reasoning Chai](llm_reasoning/arche_a_novel_task_to_evaluate_llms_on_latent_reasoning_chai.md)**

:   提出潜在推理链提取 (ARCHE) 任务，要求 LLM 将科学论文中的论证分解为基于 Peirce 三种推理范式的推理逻辑树 (RLT)，并通过 Entity Coverage 和 Reasoning Edge Accuracy 两个指标揭示了 10 个主流 LLM 在内容完整性与逻辑正确性之间的本质权衡。

**[Badthink Triggered Overthinking Attacks On Chain-Of-Thought Reasoning In Large L](llm_reasoning/badthink_triggered_overthinking_attacks_on_chain-of-thought_reasoning_in_large_l.md)**

:   提出 BadThink——首个针对 CoT 推理效率的训练时后门攻击，通过 LLM 迭代优化生成自然的冗长推理模板进行数据投毒，触发后模型生成膨胀 17× 以上的推理链（MATH-500），同时保持最终答案正确和良好隐蔽性。

**[Blm-Guard Explainable Multimodal Ad Moderation With Chain-Of](llm_reasoning/blm-guard_explainable_multimodal_ad_moderation_with_chain-of.md)**

:   提出 BLM-Guard，一个面向短视频商业广告的可解释多模态审核框架：先通过 Rule-driven ICoT 数据合成 + SFT 冷启动建立结构化推理能力，再用 Self-Adaptive GRPO 强化学习（结合规则正确性奖励 + 自适应一致性奖励 SCA-R）优化策略对齐，在真实广告 benchmark 上达到 91.4% 严格准确率和 0.845 推理一致性分数。

**[Chain-Of-Thought Driven Adversarial Scenario Extrapolation For Robust Language M](llm_reasoning/chain-of-thought_driven_adversarial_scenario_extrapolation_for_robust_language_m.md)**

:   提出 ASE（Adversarial Scenario Extrapolation），一种推理时 CoT 防御框架，让 LLM 在回答前自主模拟对抗场景并制定防御策略，在四类安全威胁（越狱、毒性、幻觉、偏见）上实现近零攻击成功率，同时将直接拒绝率降至≤4%，兼顾鲁棒性和用户体验。

**[Cmmcot Enhancing Complex Multi-Image Comprehension Via Multi](llm_reasoning/cmmcot_enhancing_complex_multi-image_comprehension_via_multi.md)**

:   提出 CMMCoT 框架，通过构建交错的多模态多步推理链（含视觉区域 token 监督）和测试时检索式记忆增强模块（RIFREM），在不增加参数的前提下提升多图场景下的慢思考推理能力，基于 Qwen2.5-VL-7B 在多图基准上平均提升 1.4 分。

**[Deep Hidden Cognition Facilitates Reliable Chain-Of-Thought ](llm_reasoning/deep_hidden_cognition_facilitates_reliable_chain-of-thought_.md)**

:   本文发现 LLM 在 CoT 推理过程中，中间层的注意力头激活值隐式编码了推理步骤的真实性信息（最高 85% 探测准确率），据此训练置信度预测器引导 Beam Search 动态选择高置信度推理路径，在数学/符号/常识推理任务上超越 Self-Consistency 和 PRM Guided Search。

**[Dropouts In Confidence Moral Uncertainty In Human-Llm Alignment](llm_reasoning/dropouts_in_confidence_moral_uncertainty_in_human-llm_alignment.md)**

:   系统研究 32 个开源 LLM 在道德困境（电车问题）中的决策不确定性，发现不确定性主要受模型架构而非道德维度驱动；在推理时引入 attention dropout 增加随机性后，模型的互信息显著上升，human-LLM 道德对齐度也随之改善——表明降低 LLM 在道德场景中的过度自信可以改善与人类偏好的一致性。

**[Esg-Bench Benchmarking Long-Context Esg Reports For Hallucination Mitigation](llm_reasoning/esg-bench_benchmarking_long-context_esg_reports_for_hallucination_mitigation.md)**

:   构建 ESG-Bench——270 个人工标注 QA 对来自 94 份真实 ESG 报告（2020-2024），提出三阶段幻觉缓解：SFT（有基础答案+「不提供」弃权标签）→ CoT Prompting（2/4步提示模板）→ CoT 微调（人工推理链），其中 4 步 CoT 微调的 Llama-3 达到 92.52% 有答案准确率 + 99.37% 无答案准确率（平衡 96%），且迁移到 HaluEval/BioASQ 也有提升。

**[Evaluating Synthesizing And Enhancing For Customer Support Conversation](llm_reasoning/evaluating_synthesizing_and_enhancing_for_customer_support_conversation.md)**

:   基于COPC行业标准定义客服对话的5个阶段和12种策略，通过5个LLM Agent角色扮演生成11232条策略丰富的合成对话（RoleCS），并构建1855条真实对话改写的评估集（CSConv），微调后显著提升策略对齐的回复质量和问题解决率。

**[Extendattack Attacking Servers Of Lrms Via Extending Reasoning](llm_reasoning/extendattack_attacking_servers_of_lrms_via_extending_reasoning.md)**

:   提出 ExtendAttack，一种针对大推理模型（LRM）的资源耗尽攻击：通过将 prompt 中的字符随机转换为多进制 ASCII 编码，迫使模型在回答问题前先执行大量逐字符解码推理，使 o3 的响应长度增加 2.7 倍以上、延迟翻倍，同时保持答案准确率基本不变。

**[Graph Of Verification Structured Verification Of Llm Reasoning With Directed Acy](llm_reasoning/graph_of_verification_structured_verification_of_llm_reasoning_with_directed_acy.md)**

:   提出 Graph of Verification (GoV)，一种将 LLM 推理过程建模为有向无环图 (DAG) 的结构化验证框架，通过灵活的节点块(Node Block)架构实现多粒度验证——从形式化任务的原子步骤到自然语言叙述的段落级验证——在结构化和松散结构化推理基准上均显著优于整体验证和其他分解验证方法。

**[Improving Value-Based Process Verifier Via Low-Cost Variance Reduction](llm_reasoning/improving_value-based_process_verifier_via_low-cost_variance_reduction.md)**

:   针对基于值的过程验证器(PRM)训练中蒙特卡罗(MC)估计因采样数有限导致的高方差问题，提出Compound Monte Carlo Sampling (ComMCS)方法，通过线性组合当前步和后续步的MC估计量来无偏地降低方差，无需额外LLM推理开销，在MATH-500上Best-of-32实验中提升2.2个点。

**[Incorporating Self-Rewriting Into Large Language Model Reasoning Reinforcement](llm_reasoning/incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)**

:   提出Self-Rewriting框架，让LRM在RL训练中对"简单"样本（全部回答正确的query）重写自身推理文本并从中学习，仅增加约10%训练开销即可在保持准确率的同时将推理长度减少46%，内部推理质量（LLM-as-Judge）提升7.2分，有效缓解过度思考、冗余思考等问题。

**[Intention Chain-Of-Thought Prompting With Dynamic Routing For Code Generation](llm_reasoning/intention_chain-of-thought_prompting_with_dynamic_routing_for_code_generation.md)**

:   提出 RoutingGen——基于认知经济原则的难度感知代码生成框架：用 Qwen3-8B 分类器动态路由任务到简单路径（few-shot 直接生成）或复杂路径（Intention CoT = 规格约束 + 算法意图 + 复杂度分析），在 McEval 上提升 +45.15% 同时平均减少 46.37% token 消耗。

**[Jupiter Enhancing Llm Data Analysis Capabilities Via Notebook And Inference-Time](llm_reasoning/jupiter_enhancing_llm_data_analysis_capabilities_via_notebook_and_inference-time.md)**

:   构建NbQA数据集（从真实Jupyter Notebook提取3.8万task-solution对）+ 提出Jupiter框架（将数据分析建模为状态级搜索问题，用值模型引导PUCT搜索），使Qwen2.5-14B在InfiAgent-DABench上达86.38%超越GPT-4o(85.99%)，Qwen2.5-7B在DSBench上从63.51%提升至89.19%。

**[L2V-Cot Cross-Modal Transfer Of Chain-Of-Thought Reasoning V](llm_reasoning/l2v-cot_cross-modal_transfer_of_chain-of-thought_reasoning_v.md)**

:   通过 LAT 分析发现 LLM 和 VLM 的低频 CoT 方向表示具有相似分布，提出 L2V-CoT：从 LLM 提取 CoT 方向表示 → 低通滤波 → 频域重采样匹配维度 → 注入 VLM 隐藏层，training-free 地将 LLM 的推理能力迁移到 VLM，平均提升 3.7%，最高 8.6%。

**[Relation-R1 Progressively Cognitive Chain-Of-Thought Guided Reinforcement Learni](llm_reasoning/relation-r1_progressively_cognitive_chain-of-thought_guided_reinforcement_learni.md)**

:   提出 Relation-R1，首个统一二元和 N 元关系理解的框架，通过渐进式认知 CoT 引导的 SFT + GRPO 多奖励优化，仅 3B 参数即超越 13B 模型，在 PSG 上 Mean 达 21.20%（+6.87%），SWiG 全指标 SOTA（Grnd-all 30.18%，+14.48%）。

**[Rpm-Mcts Knowledge-Retrieval As Process Reward Model With Monte Carlo Tree Searc](llm_reasoning/rpm-mcts_knowledge-retrieval_as_process_reward_model_with_monte_carlo_tree_searc.md)**

:   提出 RPM-MCTS——用知识库检索替代训练的过程奖励模型（PRM）来指导代码生成的 MCTS 搜索。利用同类算法实现的同质性，从知识库中检索正确算法步骤作为评估信号，配合相似度过滤去除冗余扩展节点和沙箱执行定位错误，实现 ~15% token 减少同时超越 SOTA。

**[Sapo Self-Adaptive Process Optimization Makes Small Reasoners Stronger](llm_reasoning/sapo_self-adaptive_process_optimization_makes_small_reasoners_stronger.md)**

:   受神经科学中Error-Related Negativity启发，提出自适应过程优化方法SAPO，通过首错检测+局部后验估计替代低效的逐步蒙特卡洛rollout，在降低2-3倍计算成本的同时实现推理器-验证器协同优化，使小语言模型（≤2B）在数学和代码推理任务上超越多数自演化方法。

**[Scale Selective Resource Allocation For Overcoming Performance Bottlenecks In Ma](llm_reasoning/scale_selective_resource_allocation_for_overcoming_performance_bottlenecks_in_ma.md)**

:   基于认知科学的双过程理论，提出SCALE框架将数学问题分解为子问题后按难度分配不同计算资源（System 1快速计算 vs System 2深度推理），在AIME25上将Qwen3-32B从57.50%提升至71.25%，同时比InftyThink节省33-53%的token。

**[Serl Self-Examining Reinforcement Learning On Open-Domain](llm_reasoning/serl_self-examining_reinforcement_learning_on_open-domain.md)**

:   提出SERL自我改进框架，LLM同时作为Actor（生成者）和Judge（评估者），用Copeland成对比较方法从自身判断中推导奖励信号，无需外部奖励模型或人工标注，使Qwen3-8B在AlpacaEval 2.0上从52.37%提升到59.90%（+7.53%），接近Qwen3-32B水平。

**[Spare Single-Pass Annotation With Reference-Guided Evaluation For Automatic Proc](llm_reasoning/spare_single-pass_annotation_with_reference-guided_evaluation_for_automatic_proc.md)**

:   提出 SPARE 框架，通过单次结构化生成同时完成解题步骤与参考解的对齐和准确性判断（含显式推理），无需额外训练数据，比 MCTS 方法快 2.3 倍且仅需 16% 训练样本即可实现 OOD 泛化。

**[Stable Voting And The Splitting Of Cycles](llm_reasoning/stable_voting_and_the_splitting_of_cycles.md)**

:   研究Simple Stable Voting (SSV)——已在数百次实际选举中使用的递归投票规则——是否总是精化(refine)Split Cycle (SC)方法的猜想，通过数学证明（≤5候选人）和SAT求解（6-7候选人）确定：猜想在≤6候选人时成立，≥7候选人时被反驳，并通过构造性证明推广到任意多候选人。

**[Text-To-Scene With Large Reasoning Models](llm_reasoning/text-to-scene_with_large_reasoning_models.md)**

:   提出Reason-3D，利用大推理模型（LRM）的多步空间推理能力，通过语义投票式物体检索+双阶段布局（自回归放置+碰撞感知优化）实现从文本到3D场景的零样本生成，在人工评价中Elo评分达2248（远超Holodeck的1500和LayoutVLM的1650）。

**[The Curious Case Of Analogies Investigating Analogical Reasoning In Large Langua](llm_reasoning/the_curious_case_of_analogies_investigating_analogical_reasoning_in_large_langua.md)**

:   通过 Patchscopes、注意力屏蔽和线性探针等机制可解释性工具，系统揭示了 LLM 类比推理的内部机制：模型能在中上层有效编码关系信息，但**应用**关系信息到新实体是比**提取**更大的瓶颈；成功的类比推理与故事间强结构对齐相关联，失败则反映弱化或错位的对齐。

**[Trade-Offs In Large Reasoning Models An Empirical Analysis Of Deliberative And A](llm_reasoning/trade-offs_in_large_reasoning_models_an_empirical_analysis_of_deliberative_and_a.md)**

:   系统评估了LRM（如DeepSeek-R1、QwQ、OpenThinker等）在获取深度推理能力后对基础能力（helpfulness和harmlessness）的负面影响，发现deliberative reasoning显著降低指令遵循和安全性能力，并提出Zero-Thinking、Less-Thinking、Summary-Thinking等自适应推理模式可有效缓解这些缺陷。

---

## 🔍 信息检索/RAG { #information_retrieval }

**[As Eastern Powers I Will Veto An Investigation Of Nation-Level Bias Of Large Lan](information_retrieval/as_eastern_powers_i_will_veto_an_investigation_of_nation-level_bias_of_large_lan.md)**

:   系统性地研究 LLM 在国际关系领域的国家级偏见，基于联合国安理会真实数据设计三种偏见测试（直接问答、关联测试、投票模拟），揭示偏见的多维性——随模型和评知上下文变化，并提出 RAG+Reflexion 去偏框架。

**[Beyond Perplexity Let The Reader Select Retrieval Summaries Via Spectrum Project](information_retrieval/beyond_perplexity_let_the_reader_select_retrieval_summaries_via_spectrum_project.md)**

:   提出 Spectrum Projection Score (SPS) 这一无需训练的指标，通过衡量摘要 token 嵌入与 reader LLM 主子空间的对齐程度来评估检索摘要质量，替代传统困惑度指标。结合 xCompress 推理时控制器，在 5 个 QA 数据集上显著优于基于困惑度的方法（HotpotQA EM +3.6）。

**[Cog-Rag Cognitive-Inspired Dual-Hypergraph With Theme Alignment Retrieval-Augmen](information_retrieval/cog-rag_cognitive-inspired_dual-hypergraph_with_theme_alignment_retrieval-augmen.md)**

:   提出 Cog-RAG，用主题超图和实体超图构建双超图索引，模拟人类"自顶向下"的认知过程进行两阶段检索（先主题后细节），实现从全局语义到局部信息的对齐生成。

**[Comlq Benchmarking Complex Logical Queries In Information Retrieval](information_retrieval/comlq_benchmarking_complex_logical_queries_in_information_retrieval.md)**

:   构建了首个面向复杂逻辑查询的信息检索基准 ComLQ（含合取、析取、否定等 14 种查询类型），并提出子图引导的 LLM 数据合成方法和否定一致性评估指标 LSNC，揭示现有检索器在逻辑推理尤其是否定建模上的严重不足。

**[Comorag A Cognitive-Inspired Memory-Organized Rag For Stateful Long Narrative Re](information_retrieval/comorag_a_cognitive-inspired_memory-organized_rag_for_stateful_long_narrative_re.md)**

:   受人脑前额叶皮层元认知调控机制启发，提出 ComoRAG 框架，通过动态记忆工作空间和迭代探测查询实现有状态的多步推理，在长篇叙事理解（200K+ tokens）任务上显著超越现有 RAG 方法。

**[Convmix A Mixed-Criteria Data Augmentation Framework For Conversational Dense Re](information_retrieval/convmix_a_mixed-criteria_data_augmentation_framework_for_conversational_dense_re.md)**

:   提出 ConvMix 混合准则数据增强框架，从查询和文档双方向用 LLM 进行可扩展的相关性标注增强，并通过聚类多样性选择和 Fisher 信息近分布监督筛选，系统性提升对话式稠密检索性能。

**[Do Retrieval Augmented Language Models Know When They Dont Know](information_retrieval/do_retrieval_augmented_language_models_know_when_they_dont_know.md)**

:   系统分析RAG模型的拒绝校准问题，发现RALM在检索文档全部不相关时过度拒绝率超过55%（即使模型内部知识足够回答），提出结合不确定性估计和拒绝感知微调的机制来平衡拒绝与回答质量。

**[Does Less Hallucination Mean Less Creativity An Empirical Investigation In Llms](information_retrieval/does_less_hallucination_mean_less_creativity_an_empirical_investigation_in_llms.md)**

:   系统研究三种幻觉缓解方法（CoVe、DoLa、RAG）对LLM创造力的影响，发现它们对发散性创造力有截然相反的效果——CoVe增强、DoLa抑制、RAG无影响——而收敛性创造力基本不受影响，这一规律跨模型家族和参数规模一致成立。

**[Exposing The Cracks Vulnerabilities Of Retrieval-Augmented Llm-Based Machine Tra](information_retrieval/exposing_the_cracks_vulnerabilities_of_retrieval-augmented_llm-based_machine_tra.md)**

:   开发受控噪声注入框架系统评估检索增强翻译（REAL-MT），引入Fidelity和CAR两个新指标，在10语言对×4种噪声类型上揭示模型即使面对矛盾上下文仍盲目采纳（CAR保持65-78%），大推理模型（LRM）反而更脆弱（会"合理化"错误上下文），且噪声鲁棒性与干净上下文利用率存在根本性trade-off。

**[Himo-Clip Modeling Semantic Hierarchy And Monotonicity In Vi](information_retrieval/himo-clip_modeling_semantic_hierarchy_and_monotonicity_in_vi.md)**

:   提出 HiMo-CLIP，通过对文本嵌入做 batch 内 PCA 分解（HiDe）提取多粒度语义成分，配合双分支单调性感知对比损失（MoLo），在不修改编码器的前提下让 CLIP 学会"文本越完整、对齐分数越高"的语义单调性，在长文本检索上显著超越现有方法。

**[Knowledge Completes The Vision A Multimodal Entity-Aware Retrieval-Augmented Gen](information_retrieval/knowledge_completes_the_vision_a_multimodal_entity-aware_retrieval-augmented_gen.md)**

:   本文提出MERGE，首个面向新闻图像描述的多模态实体感知RAG框架，通过构建实体中心多模态知识库（EMKB）、假设描述引导的多模态对齐（HCMA）和检索驱动的多模态知识集成（RMKI）三大组件，在GoodNews上CIDEr提升+6.84、F1提升+4.14，并在未见过的Visual News上实现CIDEr +20.17的强泛化。

**[Llms For Game Theory Entropy-Guided In-Context Learning And Adaptive Cot Reasoni](information_retrieval/llms_for_game_theory_entropy-guided_in-context_learning_and_adaptive_cot_reasoni.md)**

:   提出一种基于熵引导的自适应 LLM 推理框架，结合动态上下文检索和自适应链式思维（CoT）推理，在井字棋博弈任务中将 LLM 的平均对局结果从 -11.6% 提升至 +9.5%，同时保持较低的 LLM 查询次数。

**[Magnitude Matters A Superior Class Of Similarity Metrics For Holistic Semantic U](information_retrieval/magnitude_matters_a_superior_class_of_similarity_metrics_for_holistic_semantic_u.md)**

:   提出两种无参数、幅度感知的向量相似度度量——Overlap Similarity (OS) 和 Hyperbolic Tangent Similarity (HTS)，在 4 个句子嵌入模型和 8 个 NLP 基准上，对分类任务（释义、推理）的 MSE 显著低于 Cosine Similarity 和 Dot Product，且无需任何额外训练开销。

**[Mem-Pal Towards Memory-Based Personalized Dialogue Assistants For Long-Term User](information_retrieval/mem-pal_towards_memory-based_personalized_dialogue_assistants_for_long-term_user.md)**

:   提出H2Memory四层分层异构记忆结构（日志图/背景记忆/主题大纲/原则），通过PAL-Set数据集（100用户×8.4个月交互）验证，在需求重述和方案建议任务上将BLEU-1从13.59提升至26.67。

**[Multimodal Deepresearcher Generating Text-Chart Interleaved ](information_retrieval/multimodal_deepresearcher_generating_text-chart_interleaved_.md)**

:   提出 Multimodal DeepResearcher，一个四阶段 Agent 框架从零生成图文交替研究报告：通过形式化可视化描述（FDV）让 LLM 学习和生成多样化图表，结合 Actor-Critic 迭代精炼机制（LLM生成D3.js代码→浏览器渲染→多模态LLM评审），在自建 MultimodalReportBench 上达到 82% 整体胜率（Claude 3.7），人类评估 100% 胜率。

**[N2N-Gqa Noise-To-Narrative For Graph-Based Table-Text Question Answering Using L](information_retrieval/n2n-gqa_noise-to-narrative_for_graph-based_table-text_question_answering_using_l.md)**

:   提出 N2N-GQA——首个用于开放域混合表格-文本问答的零样本框架，核心思路是将检索到的嘈杂文档构建为动态证据图（文档为节点、TF-IDF共享词为边），通过图中心性剪枝识别"桥接文档"连接多跳推理链，在 OTT-QA 上比 Vanilla RAG 提升 +39.6 EM（从 8.0 到 48.8），零样本即接近微调系统 CORE (49.0 EM)。

**[Neighbor-Aware Instance Refining With Noisy Labels For Cross-Modal Retrieval](information_retrieval/neighbor-aware_instance_refining_with_noisy_labels_for_cross-modal_retrieval.md)**

:   提出 NIRNL 框架，通过跨模态边距保持（CMP）增强样本区分度，并利用邻域感知实例精炼（NIR）将训练数据三分为纯净/困难/噪声子集，分别定制不同优化策略，统一了鲁棒学习、标签校准和实例选择三种范式，在高噪声率下实现了 SOTA 跨模态检索性能。

**[Oad-Promoter Enhancing Zero-Shot Vqa Using Large Language Models With Object Att](information_retrieval/oad-promoter_enhancing_zero-shot_vqa_using_large_language_models_with_object_att.md)**

:   本文提出OAD-Promoter，通过对象集中样例生成（OEG）、记忆知识辅助（MKA）和OAD Prompt三个模块协同工作，在零样本设置下缓解LLM继承的语言偏差并提升领域迁移能力，在VQAv2等多个基准上取得SOTA。

**[Positional Bias In Multimodal Embedding Models Do They Favor The Beginning The M](information_retrieval/positional_bias_in_multimodal_embedding_models_do_they_favor_the_beginning_the_m.md)**

:   本文首次系统研究多模态表示模型中的位置偏差现象，发现文本编码器倾向于偏好输入开头，而图像编码器在开头和结尾均表现偏好，并通过大量控制实验揭示该偏差源于位置编码方案、训练损失、上下文重要性和图文对训练的多因素共同作用。

**[Precise Reducing The Bias Of Llm Evaluations Using Prediction-Powered Ranking Es](information_retrieval/precise_reducing_the_bias_of_llm_evaluations_using_prediction-powered_ranking_es.md)**

:   将Prediction-Powered Inference（PPI）框架扩展到子实例级别的排序指标（如Precision@K），通过仅30-100条人工标注+大量LLM评判结果获得无偏的排序指标估计，计算复杂度从 $O(2^{|C|})$ 降至 $O(2^K)$，在印度电商搜索场景中成功指导LLM查询改写系统上线。

**[Prime Planning And Retrieval-Integrated Memory For Enhanced Reasoning](information_retrieval/prime_planning_and_retrieval-integrated_memory_for_enhanced_reasoning.md)**

:   受双系统认知理论启发，提出PRIME多Agent推理框架——Quick Thinking Agent（System 1）快速生成直觉答案，Reflection Agent评估可信度，不确定时触发System 2的6个专门化Agent（规划/搜索/阅读/假设/整合/决策）进行深度知识检索推理，使开源LLaMA 3在医学/多跳QA上接近GPT-4o性能。

**[Reap Enhancing Rag With Recursive Evaluation And Adaptive Planning For Multi-Hop](information_retrieval/reap_enhancing_rag_with_recursive_evaluation_and_adaptive_planning_for_multi-hop.md)**

:   提出 REAP 双模块迭代框架，通过子任务规划器 (SP) 维护全局视角动态指导推理轨迹，事实提取器 (FE) 从检索内容中提取结构化事实和潜在线索，两者递归协作解决多跳问答。在 4 个基准上以 Llama-3.1-8B 显著超越所有基线（HotpotQA F1 68.0 vs 次优 63.4）。

**[Refeed Retrieval Feedback-Guided Dataset Construction For Style-Aware Query Rewr](information_retrieval/refeed_retrieval_feedback-guided_dataset_construction_for_style-aware_query_rewr.md)**

:   提出一个检索反馈驱动的数据集生成框架，通过识别检索失败case、LLM风格化改写、重检索验证三步闭环，自动构建高质量的风格感知查询改写数据集，为训练检索对齐的改写模型提供数据基础。

**[Rrra Resampling And Reranking Through A Retriever Adapter](information_retrieval/rrra_resampling_and_reranking_through_a_retriever_adapter.md)**

:   提出RRRA框架，通过在Bi-Encoder上添加轻量级可学习适配器来建模每个候选文档的假阴性概率，并将其同时用于训练时的负样本重采样和推理时的重排序，在NQ/TQ/MS MARCO上持续超越SimANS/TriSampler等强基线。

**[Sr-Ki Scalable And Real-Time Knowledge Integration Into Llms Via Supervised Atte](information_retrieval/sr-ki_scalable_and_real-time_knowledge_integration_into_llms_via_supervised_atte.md)**

:   提出SR-KI框架，通过两阶段训练（检索层定位 + 注意力监督损失）实现结构化知识库向LLM KV缓存的高效注入，在单块A100 40GB GPU上支持最多40K知识库条目的注入，且通过top-100压缩实现高达99.75%的压缩率，同时保持88%以上的平均Recall@10检索性能。

**[Towards Inference-Time Scaling For Continuous Space Reasoning](information_retrieval/towards_inference-time_scaling_for_continuous_space_reasoning.md)**

:   首次系统研究离散文本推理中的inference-time scaling技术能否迁移到连续潜空间推理模型（COCONUT），发现dropout采样能生成多样推理路径（Pass@32达44.43%），但PRM/ORM仅带来不足2.3%提升，根因在于连续思维表示缺乏区分正误推理的几何归纳偏置。

**[When Small Models Are Right For Wrong Reasons Process Verification For Trustwort](information_retrieval/when_small_models_are_right_for_wrong_reasons_process_verification_for_trustwort.md)**

:   通过分析 10,734 条推理轨迹揭示小型语言模型（7-9B）存在严重的"答对但理由错"（RWR）现象——50-69% 的正确答案包含根本性推理缺陷；提出推理完整性评分（RIS）作为过程级指标，发现 RAG 能有效改善推理质量而元认知干预反而有害，并蒸馏出快速分类器（0.86 F1, 100× 加速）用于实时部署。

---

## 📐 优化/理论 { #optimization }

**[A Distributed Asynchronous Generalized Momentum Algorithm Wi](optimization/a_distributed_asynchronous_generalized_momentum_algorithm_wi.md)**

:   提出一种完全异步（totally asynchronous）的广义动量（Generalized Momentum）分布式优化算法，无需假设通信/计算延迟的上界即可保证线性收敛，在 Fashion-MNIST 分类任务上比梯度下降快 71%、比 Heavy Ball 快 41%、比 Nesterov 加速梯度法快 19%。

**[A Unified Convergence Analysis For Semi-Decentralized Learni](optimization/a_unified_convergence_analysis_for_semi-decentralized_learni.md)**

:   本文在统一的收敛分析框架下，首次系统比较了半去中心化联邦学习中两种服务器-设备通信原语（S2S仅返回被采样设备 vs. S2A广播给所有设备），揭示了S2S在高组间异质性下更优、S2A在低异质性下更优的不同regime，并给出了实用的系统配置指南。

**[Beerna Tertiary Structure-Based Rna Inverse Folding Using Artificial Bee Colony](optimization/beerna_tertiary_structure-based_rna_inverse_folding_using_artificial_bee_colony.md)**

:   提出 BeeRNA，将人工蜂群（ABC）优化算法应用于 RNA 三级结构逆折叠问题，通过碱基对距离预筛选 + RMSD 两阶段适应度评估，在短/中长度 RNA（<100 nt）上超越深度学习方法 gRNAde 和 RiboDiffusion。

**[Beyond The Mean Fisher-Orthogonal Projection For Natural Gradient Descent In Lar](optimization/beyond_the_mean_fisher-orthogonal_projection_for_natural_gradient_descent_in_lar.md)**

:   提出 Fisher-Orthogonal Projection (FOP)，通过在 Fisher 度量下对子批次梯度差做正交投影来补充方差信息，使二阶优化器 KFAC 在超大 batch 训练中保持有效，实现最高 ×7.5 的加速。

**[Bridging Synthetic And Real Routing Problems Via Llm-Guided Instance Generation ](optimization/bridging_synthetic_and_real_routing_problems_via_llm-guided_instance_generation_.md)**

:   提出 EvoReal 框架，利用 LLM 驱动的进化搜索生成结构上接近真实世界的 VRP 合成实例，再通过两阶段渐进微调策略将预训练神经求解器适配到真实基准，在 TSPLib (1.05% gap) 和 CVRPLib (2.71% gap) 上大幅超越已有神经求解器。

**[Co-Layout Llm-Driven Co-Optimization For Interior Layout](optimization/co-layout_llm-driven_co-optimization_for_interior_layout.md)**

:   提出 Co-Layout 框架，利用 LLM 从自然语言需求中提取结构化约束，再通过基于网格的整数规划（IP）联合优化房间布局与家具摆放，辅以粗到精求解策略提升效率，显著优于现有两阶段方案。

**[Convex Clustering Redefined Robust Learning With The Median Of Means Estimator](optimization/convex_clustering_redefined_robust_learning_with_the_median_of_means_estimator.md)**

:   提出 COMET（Convex Clustering with Median of Means Estimator），将中位数均值（MoM）估计器整合到凸聚类框架中，通过随机分箱、截断距离和 ADAM 优化实现对噪声和异常值的鲁棒聚类，无需预设聚类数量，在理论上证明了弱一致性，在合成和真实数据集上全面超越现有方法。

**[Cost-Minimized Label-Flipping Poisoning Attack To Llm Alignment](optimization/cost-minimized_label-flipping_poisoning_attack_to_llm_alignment.md)**

:   首次从理论上分析了在 RLHF/DPO 对齐过程中，通过翻转偏好标签来引导 LLM 策略走向攻击者目标所需的最小成本，将其形式化为凸优化问题并推导了成本的上下界，进而提出 PCM（Poisoning Cost Minimization）后处理方法，可在保持投毒效果的同时显著减少标签翻转数量。

**[Data Heterogeneity And Forgotten Labels In Split Federated Learning](optimization/data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)**

:   系统研究了 Split Federated Learning 中数据异构导致的灾难性遗忘现象（尤其是 server 端处理顺序造成的 intra-round 遗忘），并提出基于 multi-head 的 Hydra 方法，将 part-2 的最后层分组训练再聚合，显著降低标签间性能差距（PG 最高降低 75.4%）。

**[Ecpv2 Fast Efficient And Scalable Global Optimization Of Lipschitz Functions](optimization/ecpv2_fast_efficient_and_scalable_global_optimization_of_lipschitz_functions.md)**

:   提出ECPv2算法，通过三项创新（自适应下界、Worst-$m$ memory、固定随机投影），将Lipschitz函数全局优化的运行时从$\Omega(n^2 d)$降至$\Omega(n(m+d)\log n)$，同时保持与minimax下界匹配的$O(n^{-1/d})$ regret收敛速率。

**[Explore How To Inject Beneficial Noise In Mllms](optimization/explore_how_to_inject_beneficial_noise_in_mllms.md)**

:   提出 Multimodal Noise Generator (MuNG)，通过变分推断框架从图文对中动态生成"有益噪声"注入冻结的MLLM视觉特征中，以抑制无关语义、增强跨模态表征对齐，仅需约1%额外参数即可超越全参数微调和LoRA等PEFT方法。

**[Fedp2Eft Federated Learning To Personalize Peft For Multilingual Llms](optimization/fedp2eft_federated_learning_to_personalize_peft_for_multilingual_llms.md)**

:   提出FedP²EFT，通过联邦学习协作训练一个Personalization Strategy Generator (PSG)，为每个客户端自动生成个性化的LoRA rank结构，在多语言LLM微调中大幅超越手工设计的PEFT配置和现有FL个性化方法。

**[Fedpm Federated Learning Using Second-Order Optimization With Preconditioned Mix](optimization/fedpm_federated_learning_using_second-order_optimization_with_preconditioned_mix.md)**

:   提出 FedPM（Federated Preconditioned Mixing），一种新型联邦学习方法，通过在服务器端用"预条件混合"替代传统的简单参数平均，解决了现有二阶联邦优化方法中局部预条件器漂移问题，在理论上证明了强凸目标的超线性收敛速率，并在异质数据场景中显著超越现有方法。

**[Ghost Solving The Traveling Salesman Problem On Graphs Of Convex Sets](optimization/ghost_solving_the_traveling_salesman_problem_on_graphs_of_convex_sets.md)**

:   提出 GHOST 框架，一种层次化最优搜索算法，用于求解凸集图（GCS）上的旅行商问题。通过结合组合路径搜索与凸轨迹优化，并利用新颖的抽象路径展开算法计算可容许下界指导最佳优先搜索，GHOST 在保证最优性的同时比统一混合整数凸规划基线快数个数量级。

**[Instance Generation For Meta-Black-Box Optimization Through Latent Space Reverse](optimization/instance_generation_for_meta-black-box_optimization_through_latent_space_reverse.md)**

:   提出 LSRE 框架，通过自编码器构建 BBO 问题实例的二维潜在空间，并利用遗传编程从该空间中反向工程出多样化的合成优化问题实例集 Diverse-BBO，显著提升 MetaBBO 方法的泛化性能。

**[Motif Multi-Strategy Optimization Via Turn-Based Interactive Framework](optimization/motif_multi-strategy_optimization_via_turn-based_interactive_framework.md)**

:   提出 MOTIF 框架，将求解器设计建模为多策略优化问题，通过基于蒙特卡洛树搜索 (MCTS) 的双 LLM 代理回合制竞争机制，联合优化组合优化求解器中的多个相互依赖的算法组件，在 TSP、CVRP、BPP 等多个组合优化领域中一致超越现有方法。

**[On The Learning Dynamics Of Two-Layer Linear Networks With Label Noise Sgd](optimization/on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd.md)**

:   在二层过参数化线性网络上理论分析 Label Noise SGD 的学习动力学，揭示了两阶段行为——Phase I 中权重范数逐渐缩小使模型从 lazy regime 逃逸到 rich regime，Phase II 中权重与真实插值器对齐并收敛——并将该理论扩展到 SAM 优化器。

**[Parametrized Multi-Agent Routing Via Deep Attention Models](optimization/parametrized_multi-agent_routing_via_deep_attention_models.md)**

:   提出Deep FLPO框架，将最大熵原理（MEP）的代数结构与permutation-invariant的encoder-decoder神经网络（SPN）融合，解决设施选址与路径联合优化的NP-hard混合整数问题，实现策略推理100倍加速、与Gurobi精确解匹配且快1500倍。

**[Pareto-Grid-Guided Large Language Models For Fast And High-Quality Heuristics De](optimization/pareto-grid-guided_large_language_models_for_fast_and_high-quality_heuristics_de.md)**

:   提出 MPaGE 框架，将 LLM 与 Pareto Front Grid 机制和语义聚类结合，自动为多目标组合优化问题生成兼顾解质量与运行效率的启发式算法，在 Bi-TSP、Tri-TSP、Bi-CVRP、Bi-KP 上 HV 和 IGD 均显著优于 EoH、MEoH 等基线。

**[Peoat Personalization-Guided Evolutionary Question Assembly For One-Shot Adaptiv](optimization/peoat_personalization-guided_evolutionary_question_assembly_for_one-shot_adaptiv.md)**

:   首次提出"一次性自适应测试 (OAT)"任务，将其建模为组合优化问题，并设计 PEOAT 框架——结合个性化初始化、认知增强进化搜索和多样性保持选择策略，在无交互反馈的条件下为每位考生一次性选出最优题集，大幅超越传统 CAT 方法。

**[Personalized Federated Learning With Bidirectional Communication Compression Via](optimization/personalized_federated_learning_with_bidirectional_communication_compression_via.md)**

:   提出 pFed1BS 框架，通过单比特随机草图实现联邦学习中上下行双向极致通信压缩（降低 99%+），同时引入基于符号的正则化器实现客户端模型个性化，在非 IID 数据场景下同时解决通信瓶颈和数据异质性两大难题。

**[Smofi Step-Wise Momentum Fusion For Split Federated Learning On Heterogeneous Da](optimization/smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)**

:   提出 SMoFi 框架，通过在 Split FL 的 server 端每步同步各 surrogate 模型的 momentum buffer，有效缓解 non-IID 数据导致的梯度分歧，在精度（最高+7.1%）和收敛速度（最高10.25×）上均显著优于现有方法。

---

## 🎵 音频/语音 { #audio_speech }

**[A Superpersuasive Autonomous Policy Debating System](audio_speech/a_superpersuasive_autonomous_policy_debating_system.md)**

:   提出DeepDebater，首个能参与并赢得完整美式策略辩论赛（八轮发言+交叉质询）的自主多Agent系统，基于层级式Agent工作流分工完成正方（Advantage）/反方（DA+CP+Kritik）论证构建，以OpenDebateEvidence的300万+张证据卡做检索增强，辅以GPT-4o TTS语音合成和EchoMimic数字人动画，在专家评估中各项指标显著超越人类编写案例（Quality 4.32 vs 3.65），模拟对局胜率达85%。

**[Ahamask Reliable Task Specification For Large Audio Language](audio_speech/ahamask_reliable_task_specification_for_large_audio_language.md)**

:   通过对大音频语言模型（LALM）Transformer 骨干中的注意力头进行二值掩码（AHAMask），无需文本指令即可可靠触发特定声学任务功能，同时揭示了 LALM 内部存在"声学功能通路"。

**[Aligning Generative Music Ai With Human Preferences Methods And Challenges](audio_speech/aligning_generative_music_ai_with_human_preferences_methods_and_challenges.md)**

:   综述/立场论文，系统梳理偏好对齐技术在音乐生成中的三条路线——MusicRL（大规模 RLHF，~30 万偏好对）、DiffRhythm+（扩散模型多偏好 DPO）、Text2midi-InferAlign（推理时树搜索，CLAP +29.4%），深入分析音乐领域独有的对齐挑战（多尺度时间连贯性、和声一致性、文化主观性、评估悖论），并给出未来路线图。

**[Cross-Space Synergy A Unified Framework For Multimodal Emotion Recognition In Co](audio_speech/cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)**

:   提出 Cross-Space Synergy（CSS）框架，通过表示空间的协同多项式融合（SPF）和梯度空间的 Pareto 梯度调节器（PGM）双管齐下，同时解决多模态对话情感识别中融合表达力不足和多目标梯度冲突两大难题。

**[Deformtrace A Deformable State Space Model With Relay Tokens For Temporal Forger](audio_speech/deformtrace_a_deformable_state_space_model_with_relay_tokens_for_temporal_forger.md)**

:   提出 DeformTrace，将可变形动态感受野和中继令牌机制引入状态空间模型，结合 Transformer 的全局建模与 SSM 的高效推理，实现时序伪造定位的 SOTA 精度与显著效率提升。

**[Do Llms Feel Teaching Emotion Recognition With Prompts Retrieval And Curriculum ](audio_speech/do_llms_feel_teaching_emotion_recognition_with_prompts_retrieval_and_curriculum_.md)**

:   提出 PRC-Emo 框架，通过显式/隐式情感提示、专用检索库和课程学习策略三位一体地提升 LLM 在对话情感识别（ERC）任务上的表现，在 IEMOCAP 和 MELD 两个基准上取得 SOTA。

**[Dualspeechlm Towards Unified Speech Understanding And Generation Via Dual Speech](audio_speech/dualspeechlm_towards_unified_speech_understanding_and_generation_via_dual_speech.md)**

:   提出 DualSpeechLM 框架，通过理解驱动语音分词器（USTokenizer）提取高层语义 token 作为 LLM 输入、声学 token 作为输出，在一个端到端框架中同时优化语音理解和生成能力。

**[End-To-End Contrastive Language-Speech Pretraining Model For Long-Form Spoken Qu](audio_speech/end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)**

:   提出 CLSR，一种端到端对比式语言-语音检索器，通过将声学表示先转换为 text-like representation 再与文本对齐，高效地从长音频中提取与问题相关的片段，为下游 LALM 的长语音问答提供 RAG 支持。

**[Generalizing Analogical Inference From Boolean To Continuous Domains](audio_speech/generalizing_analogical_inference_from_boolean_to_continuous_domains.md)**

:   从基础理论层面重新审视类比推理：首先构造反例证明布尔域上经典泛化界失效，然后提出基于参数化广义均值的统一类比推理框架，将离散分类扩展到连续回归域。

**[Gompsnr Reflourish The Signal-To-Noise Ratio Metric For Audio Generation Tasks](audio_speech/gompsnr_reflourish_the_signal-to-noise_ratio_metric_for_audio_generation_tasks.md)**

:   通过引入全方位相位导数（omnidirectional phase derivatives）替换瞬时相位来重构 SNR 指标，提出 GOMPSNR 作为更可靠的音频质量评估指标，并衍生出一系列新的损失函数显著提升神经声码器性能。

**[Hearing More With Less Multi-Modal Retrieval-And-Selection Augmented Conversatio](audio_speech/hearing_more_with_less_multi-modal_retrieval-and-selection_augmented_conversatio.md)**

:   MARS 提出多模态检索-选择方法为对话式 LLM-ASR 挑选最相关的历史上下文（而非固定前几句或全部历史），在仅用 1.5K 小时训练数据的情况下超越了用 179K 小时数据训练的 SOTA 系统 TEA-ASLP。

**[Hpsu A Benchmark For Human-Level Perception In Real-World Spoken Speech Understa](audio_speech/hpsu_a_benchmark_for_human-level_perception_in_real-world_spoken_speech_understa.md)**

:   提出 HPSU 基准，包含 20,000+ 中英文专家标注样本和 16 项任务，系统评估 Speech LLM 在真实口语场景下的深层感知与推理能力，发现最强模型（Gemini 2.5 Pro，62.6%）与人类表现（87.3%）仍有巨大差距。

**[Improving Multimodal Sentiment Analysis Via Modality Optimization And Dynamic Pr](audio_speech/improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)**

:   提出 MODS 框架，通过图卷积动态序列压缩（GDC）消除非语言模态冗余，并设计样本级动态主模态选择器（MSelector）和主模态中心交叉注意力（PCCA），实现 MSA 中按样本自适应选择主导模态。

**[Let The Model Learn To Feel Mode-Guided Tonality Injection F](audio_speech/let_the_model_learn_to_feel_mode-guided_tonality_injection_f.md)**

:   通过 MoGE 诊断策略系统发现 MIDIBERT 未有效编码调式-情感关联，提出 MoFi 注入框架通过 FiLM 机制将大调/小调先验注入 MIDIBERT 第 1 层（诊断确定的最弱情感信息层），在 EMOPIA 上准确率 75.2%（+11.8%），VGMIDI 上 59.1%（+11.8%），F1 提升 12.3%/15.5%。

**[Listen Like A Teacher Mitigating Whisper Hallucinations Using Adaptive Layer Att](audio_speech/listen_like_a_teacher_mitigating_whisper_hallucinations_using_adaptive_layer_att.md)**

:   提出两阶段框架——自适应层注意力（ALA）融合Whisper编码器多层表示以增强噪声鲁棒性，多目标知识蒸馏（MOKD）将clean teacher的语义和注意力分布对齐到noisy student——在多语言噪声ASR基准上显著降低幻觉率和WER。

**[Multi-Granularity Interactive Attention Framework For Residual Hierarchical Pron](audio_speech/multi-granularity_interactive_attention_framework_for_residual_hierarchical_pron.md)**

:   提出HIA框架，通过交互注意力模块（Interactive Attention Module）实现音素、词、句三粒度间的双向信息交互，结合残差层级结构缓解特征遗忘问题，在speechocean762数据集上所有粒度和方面指标均达到SOTA。

**[Pase Prototype-Aligned Calibration And Shapley-Based Equilibrium For Multimodal ](audio_speech/pase_prototype-aligned_calibration_and_shapley-based_equilibrium_for_multimodal_.md)**

:   提出 PaSE 框架，通过原型引导校准对齐（Entropic Optimal Transport）与 Shapley 值梯度调制的双阶段优化策略，显式解决多模态情感分析中的模态竞争问题。

**[Psa-Mf Personality-Sentiment Aligned Multi-Level Fusion For Multimodal Sentiment](audio_speech/psa-mf_personality-sentiment_aligned_multi-level_fusion_for_multimodal_sentiment.md)**

:   首次在多模态情感分析（MSA）中引入预训练人格模型提取个性化情感特征，通过人格-情感对比学习对齐和多层（预融合→交叉模态交互→增强融合）渐进融合架构，在CMU-MOSI和CMU-MOSEI上达到SOTA。

**[Say More With Less Variable-Frame-Rate Speech Tokenization Via Adaptive Clusteri](audio_speech/say_more_with_less_variable-frame-rate_speech_tokenization_via_adaptive_clusteri.md)**

:   提出 VARSTok，首个全动态可变帧率语音 tokenizer，通过时序感知密度峰聚类和隐式时长编码，实现自适应 token 分配，在使用更少 token 的同时超越固定帧率基线。

**[Text-Routed Sparse Mixture-Of-Experts Model With Explanation And Temporal Alignm](audio_speech/text-routed_sparse_mixture-of-experts_model_with_explanation_and_temporal_alignm.md)**

:   提出 TEXT 模型，利用 MLLM 为音视频生成自然语言解释来增强模态表示，设计融合 Mamba 与时序交叉注意力优点的轻量时序对齐模块，并以文本路由的稀疏专家混合进行跨模态融合，在四个 MSA 数据集上全面超越 SOTA 及 GPT-4o 等大模型。

**[Use A Unified Model For Universal Sound Separation And Extraction](audio_speech/use_a_unified_model_for_universal_sound_separation_and_extraction.md)**

:   提出 USE 统一框架，通过 EDA 网络推断声源数量和声学线索实现声音分离 (SS)，多模态融合网络解释用户提供的文本/视频/标签线索实现目标声音提取 (TSE)，联合训练+跨任务对齐使两项任务互相增强，SS +1.4dB SDR，TSE 匹配准确率 86%。

---

## 🎁 推荐系统 { #recommender }

**[Align3Gr Unified Multi-Level Alignment For Llm-Based Generat](recommender/align3gr_unified_multi-level_alignment_for_llm-based_generat.md)**

:   提出统一三层对齐框架 Align³GR，在 token 级（双端 SCID）、行为建模级（多任务 SFT）和偏好级（渐进式 DPO）系统性弥合 LLM 与推荐系统之间的语义-行为鸿沟。

**[Autopp Towards Automated Product Poster Generation And Optimization](recommender/autopp_towards_automated_product_poster_generation_and_optimization.md)**

:   提出 AutoPP，首个将商品海报自动生成与基于 CTR 反馈的自动优化统一到一个框架中的流水线，通过 unified design module 联合设计背景/文字/排版，element rendering module 高效可控地生成海报，并利用 Isolated DPO (IDPO) 实现元素级别的点击率优化。

**[Bid Farewell To Seesaw Towards Accurate Long-Tail Session-Based Recommendation V](recommender/bid_farewell_to_seesaw_towards_accurate_long-tail_session-based_recommendation_v.md)**

:   提出HID框架，通过属性感知的谱聚类构建混合意图来区分会话相关与无关的尾部物品，并设计针对长尾和准确性的双约束损失（ICLoss），实现长尾推荐与准确性的"双赢"，打破传统方法中两者此消彼长的"跷跷板"困境。

**[Crops Improving Dense Retrieval With Cross-Perspective Positive Samples In Short](recommender/crops_improving_dense_retrieval_with_cross-perspective_positive_samples_in_short.md)**

:   提出 CroPS 数据引擎，通过 query 改写行为、推荐系统交互、LLM 世界知识三个视角扩充正样本集合，配合分层标签分配（HLA）和 H-InfoNCE 损失函数，打破工业级稠密检索系统中的信息茧房效应，已在快手搜索全量部署。

**[Evaluating Llms For Police Decision-Making A Framework Based On Police Action Sc](recommender/evaluating_llms_for_police_decision-making_a_framework_based_on_police_action_sc.md)**

:   提出 PAS（Police Action Scenarios）评估框架，一个面向警务场景的 LLM 评估体系，涵盖场景定义、参考答案构建、LLM 响应生成、核心指标提取和性能解读五个阶段，基于 8000+ 韩国警察官方文件构建评估数据集，发现商用 LLM（GPT-4、Gemini、Claude）在警务任务上显著低于参考答案，尤其在事实性和逻辑正确性方面。

**[Exploiting Inter-Session Information With Frequency-Enhanced Dual-Path Networks ](recommender/exploiting_inter-session_information_with_frequency-enhanced_dual-path_networks_.md)**

:   提出FreqRec双路径架构，通过batch维和时间维两条频域路径分别捕获跨session群体节律和用户个体细粒度兴趣，并引入频域一致性损失显式对齐预测与真实频谱，在三个Amazon数据集上NDCG@10最高提升7.38%。

**[From Parameter To Representation A Closed-Form Approach For Controllable Model M](recommender/from_parameter_to_representation_a_closed-form_approach_for_controllable_model_m.md)**

:   提出 ReACT，将可控模型合并从参数空间优化转移到表征空间校正，通过闭式解实现任意用户偏好下的 Pareto 最优模型即时生成，比现有方法快 36-208 倍且性能更优。

**[Generalization Bounds For Semi-Supervised Matrix Completion With Distributional ](recommender/generalization_bounds_for_semi-supervised_matrix_completion_with_distributional_.md)**

:   提出首个半监督矩阵补全学习范式：假设采样分布 $P$ 和真实矩阵 $G$ 共享低秩子空间，给定大量未标注数据 $M$ 和少量标注数据 $N$，证明泛化误差可分解为 $\tilde{O}(\sqrt{nd/M}) + \tilde{O}(\sqrt{dr/N})$ 两个独立项，在 Douban 和 MovieLens 数据集上显著优于仅用显式反馈的基线。

**[Inference-Aware Prompt Optimization For Aligning Black-Box Large Language Models](recommender/inference-aware_prompt_optimization_for_aligning_black-box_large_language_models.md)**

:   揭示 prompt 选择与推理策略（Best-of-N、Majority Voting）之间存在非平凡交互关系，提出 IAPO 框架将 prompt 设计与推理规模联合优化为上下文最优臂识别问题，并设计 PSST 固定预算训练算法，在 6 个任务上相比推理无关方法提升最高 50%。

**[Interpretable Reward Model Via Sparse Autoencoder](recommender/interpretable_reward_model_via_sparse_autoencoder.md)**

:   提出 SARM（Sparse Autoencoder-enhanced Reward Model），将预训练的稀疏自编码器集成到奖励模型中，将隐层激活映射到可解释的稀疏单义特征空间，实现特征级的奖励归因和动态偏好操控，同时在 RewardBench 2 上取得了所有模型中的最高分。

**[Length-Adaptive Interest Network For Balancing Long And Short Sequence Modeling ](recommender/length-adaptive_interest_network_for_balancing_long_and_short_sequence_modeling_.md)**

:   提出LAIN框架，通过将序列长度作为显式条件信号注入CTR模型，缓解长序列用户与短序列用户之间的性能不均衡问题，包含谱长度编码器、长度条件提示和长度调制注意力三个轻量级即插即用模块。

**[Moral Change Or Noise On Problems Of Aligning Ai With Temporally Unstable Human ](recommender/moral_change_or_noise_on_problems_of_aligning_ai_with_temporally_unstable_human_.md)**

:   通过在肾脏移植分配领域对400+参与者进行3-5轮纵向研究，揭示了人类道德偏好在时间上的显著不稳定性（6-20%的响应变化率），并证明这种不稳定性会严重降低AI对齐模型的预测性能，从而质疑了当前基于静态偏好假设的对齐方法的有效性。

**[Multitab A Scalable Foundation For Multitask Learning On Tabular Data](recommender/multitab_a_scalable_foundation_for_multitask_learning_on_tabular_data.md)**

:   提出MultiTab-Net——首个面向表格数据的多任务Transformer架构，通过多任务掩码注意力机制缓解任务竞争，在推荐、人口普查、物理等多个领域的数据集上显著超越现有MLP-based多任务模型和单任务Transformer模型。

**[Preference Is More Than Comparisons Rethinking Dueling Bandits With Augmented Hu](recommender/preference_is_more_than_comparisons_rethinking_dueling_bandits_with_augmented_hu.md)**

:   提出一种基于增强人类反馈的无模型Dueling Bandit框架IPEA-HF，通过增强置信界（Augmented Confidence Bounds）集成上下文相似性和依赖关系来校准不确定性，在推荐、多目标优化和LLM响应优化等多个基准上表现优异。

**[Probabilistic Hash Embeddings For Online Learning Of Categorical Features](recommender/probabilistic_hash_embeddings_for_online_learning_of_categorical_features.md)**

:   提出概率哈希嵌入 (PHE)，将哈希嵌入表建模为随机变量并通过贝叶斯在线学习进行后验推断，解决了确定性哈希嵌入在流式数据场景下因参数共享导致的灾难性遗忘问题，在分类、序列建模和推荐系统中显著优于确定性基线，且仅需无碰撞嵌入表 2%~4% 的内存。

**[Semi-Supervised Synthetic Data Generation With Fine-Grained Relevance Control Fo](recommender/semi-supervised_synthetic_data_generation_with_fine-grained_relevance_control_fo.md)**

:   提出SSRA（半监督相关性感知合成数据管道），通过两阶段流程生成具有可控细粒度相关性标签（4级）的领域自适应短视频数据，增强embedding模型的语义相关性建模能力，在抖音双列场景线上A/B测试中CTR提升1.45%。

**[Slidetailor Personalized Presentation Slide Generation For Scientific Papers](recommender/slidetailor_personalized_presentation_slide_generation_for_scientific_papers.md)**

:   定义了偏好引导的论文到幻灯片生成新任务，提出 SlideTailor 框架：从用户提供的论文-幻灯片样例对中蒸馏内容偏好、从 .pptx 模板蒸馏美学偏好，通过 chain-of-speech 机制将幻灯片内容与预期口述叙事对齐，在自建 PSP 基准上以 75.8% 的综合得分和 81.63% 的人评胜率显著超越现有方法。

**[Tokenize Once Recommend Anywhere Unified Item Tokenization For Multi-Domain Llm-](recommender/tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi-domain_llm-.md)**

:   提出 UniTok，一个统一的商品 tokenization 框架，通过定制的 Mixture-of-Experts（TokenMoE）架构结合共享码本，实现跨多个领域的高效商品离散化表示，避免为每个领域单独训练 tokenizer，同时通过互信息校准机制保持跨域语义平衡。

**[Travellama A Multimodal Travel Assistant With Large-Scale Dataset And Structured](recommender/travellama_a_multimodal_travel_assistant_with_large-scale_dataset_and_structured.md)**

:   提出 TraveLLaMA，一个面向旅行辅助的多模态语言模型系统，通过构建 265K QA 对的 TravelQA 数据集和 Travel-CoT 结构化推理框架，在旅行相关问答上实现了 10.8% 的准确率提升，并在 500 人用户研究中获得了 82.5 的 SUS 可用性评分。

**[Wavelet Enhanced Adaptive Frequency Filter For Sequential Re](recommender/wavelet_enhanced_adaptive_frequency_filter_for_sequential_re.md)**

:   提出WEARec模型，通过动态频域滤波（DFF）根据用户上下文自适应调整频域滤波器捕获个性化全局偏好，并用小波特征增强（WFE）弥补全局DFT模糊短期波动的缺陷，在四个数据集上超越全部9个基线，长序列场景最高提升11.4%且训练速度快39-45%。

**[When Top-Ranked Recommendations Fail Modeling Multi-Granular Negative Feedback F](recommender/when_top-ranked_recommendations_fail_modeling_multi-granular_negative_feedback_f.md)**

:   提出 ENF（Explainable Negative Feedback）框架，通过三个协作式 MLLM Agent（Profile Agent、Video Agent、Reason Agent）和渐进式 S-GRPO 强化学习训练策略，首次实现了对视频推荐系统中隐式负反馈的可解释预测和原因分析，在腾讯新闻业务平台上实现了平均观看时长提升 6.2% 和快速跳过率下降 9.4%。

---

## ⚖️ 对齐/RLHF { #llm_alignment }

**[Align To Structure Aligning Large Language Models With Struc](llm_alignment/align_to_structure_aligning_large_language_models_with_struc.md)**

:   提出 Structural Alignment 方法，通过将语言学篇章结构框架（表层文本结构评分 + 基于RST的篇章motif分类器）融入PPO强化学习训练，并设计基于篇章motif的密集奖励机制，使LLM生成更连贯、更具人类写作风格的长文本，在论文写作和长文档摘要任务上均优于标准RLHF模型。

**[Aligntree Efficient Defense Against Llm Jailbreak Attacks](llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)**

:   AlignTree 利用 LLM 内部激活特征（线性 refusal direction + 非线性 SVM 信号）训练轻量级随机森林分类器，在几乎不增加计算开销的情况下高效检测越狱攻击，实现了 SOTA 的攻击成功率（ASR）降低效果。

**[Amapo Adaptive Margin-Attached Preference Optimization For L](llm_alignment/amapo_adaptive_margin-attached_preference_optimization_for_l.md)**

:   提出AMaPO算法，通过实例级自适应margin（结合Z-normalization和指数缩放）动态调节梯度幅度，解决DPO等离线偏好优化方法中对已正确排序样本过拟合、对错误排序样本欠拟合的核心矛盾，显著提升排序准确率和下游对齐性能。

**[Biasjailbreakanalyzing Ethical Biases And Jailbreak Vulnerabilities In Large Lan](llm_alignment/biasjailbreakanalyzing_ethical_biases_and_jailbreak_vulnerabilities_in_large_lan.md)**

:   揭示LLM安全对齐中引入的伦理偏见可被反向利用作为越狱攻击向量——边缘化群体关键词的越狱成功率比优势群体高出20%，并提出基于提示词的轻量防御方法BiasDefense。

**[Decorl Decoupling Reasoning Chains Via Parallel Sub-Step Gen](llm_alignment/decorl_decoupling_reasoning_chains_via_parallel_sub-step_gen.md)**

:   DeCoRL 将 CoT 推理从单体顺序处理转变为"交响乐团式"的模块化并行协作——9 个专用子模型（解析/语义/实体/事实核查/风格/质量/计算/验证/整合）并行生成推理子步骤，通过双重奖励归因（本地质量+贡献度）+ 级联 DRPO 优化协调，在 RM-Bench 上达到 80.8%（超越所有基线），同时实现 3.8 倍推理加速和 22.7% 的可解释性提升。

**[Differentiated Directional Intervention A Framework For Evading Llm Safety Align](llm_alignment/differentiated_directional_intervention_a_framework_for_evading_llm_safety_align.md)**

:   将 LLM 安全对齐的内部表征从传统的"单一拒绝方向"解构为功能独立的"危害检测方向"和"拒绝执行方向"，在此基础上提出 DBDI 框架，分别用自适应投影消除和直接引导两种策略精准干预两个方向，在 Llama-2 上实现 97.88% 的越狱成功率。

**[Ease Practical And Efficient Safety Alignment For Small Language Models](llm_alignment/ease_practical_and_efficient_safety_alignment_for_small_language_models.md)**

:   提出 EASE——面向边缘部署小语言模型（SLM）的安全对齐框架，通过两阶段设计解决"浅层拒绝不够安全 vs 深度推理太贵"的矛盾：第一阶段从大型推理模型蒸馏安全推理能力到 SLM，第二阶段用选择性推理激活（仅对脆弱语义区域的对抗查询启用推理，良性查询直接响应），越狱攻击成功率降低 17%（vs 浅层对齐）同时推理开销降低 90%（vs 全推理）。

**[Enhancing Uncertainty Estimation In Llms With Expectation Of Aggregated Internal](llm_alignment/enhancing_uncertainty_estimation_in_llms_with_expectation_of_aggregated_internal.md)**

:   提出EAGLE方法，通过聚合LLM多个中间层隐藏状态的logits并计算置信度分布的期望值来估计不确定性，无需训练额外参数，在多个数据集和模型上ECE从12.6%降至3.2%，AUROC从59.0%提升至61.6%。

**[Epo Diverse And Realistic Protein Ensemble Generation Via Energy Preference Opti](llm_alignment/epo_diverse_and_realistic_protein_ensemble_generation_via_energy_preference_opti.md)**

:   提出EPO（Energy Preference Optimization），将反向SDE采样与listwise能量排序偏好优化结合，用能量信号对齐预训练蛋白质生成器与目标Boltzmann分布，在Tetrapeptides/ATLAS/Fast-Folding三个基准9个指标上达到SOTA，完全消除了昂贵的分子动力学（MD）模拟需求。

**[Exploring The Effects Of Alignment On Numerical Bias In Large Language Models](llm_alignment/exploring_the_effects_of_alignment_on_numerical_bias_in_large_language_models.md)**

:   系统揭示了LLM对齐过程（指令调优+偏好调优）是LLM评估器产生数值偏差的根本原因，并验证分数范围调整是最有效的缓解策略。

**[From Classification To Ranking Enhancing Llm Reasoning Capabilities For Mbti Per](llm_alignment/from_classification_to_ranking_enhancing_llm_reasoning_capabilities_for_mbti_per.md)**

:   将MBTI人格检测从传统的四维二分类重构为listwise排序任务，通过SFT冷启动+GRPO强化学习（NDCG+维度相似度双奖励），在Kaggle和PANDORA数据集上以7B模型达到SOTA。

**[Importance-Aware Data Selection For Efficient Llm Instruction Tuning](llm_alignment/importance-aware_data_selection_for_efficient_llm_instruction_tuning.md)**

:   提出MIWV（Model Instruction Weakness Value）指标，通过比较LLM在有/无one-shot ICL示例下的损失差来衡量每条指令数据对模型能力提升的重要性，在Alpaca数据集上仅用1%（520条）数据即全面超越全量52002条的微调效果。

**[Margin-Aware Preference Optimization For Aligning Diffusion Models Without Refer](llm_alignment/margin-aware_preference_optimization_for_aligning_diffusion_models_without_refer.md)**

:   提出 MaPO（Margin-aware Preference Optimization），一种无需参考模型的偏好对齐方法，通过直接优化 Bradley-Terry 模型下偏好/非偏好输出的似然 margin 来对齐 T2I 扩散模型，在风格适配、安全生成、通用偏好对齐等 5 个领域均超越 DPO 和专用方法。

**[Metagdpo Alleviating Catastrophic Forgetting With Metacognitive Knowledge Throug](llm_alignment/metagdpo_alleviating_catastrophic_forgetting_with_metacognitive_knowledge_throug.md)**

:   提出MetaGDPO方法，从数据侧（基于元认知知识的5K数据构建MetaKL）和训练侧（GDPO——将GRPO的在线采样替换为大模型离线response group的DPO变体）两方面缓解小模型（<8B）在推理能力蒸馏中的灾难性遗忘问题。

**[On The Exponential Convergence For Offline Rlhf With Pairwise Comparisons](llm_alignment/on_the_exponential_convergence_for_offline_rlhf_with_pairwise_comparisons.md)**

:   在离线RLHF的成对比较设定下，提出RL-LOW算法实现了simple regret的指数收敛 $\exp(-\Omega(n/H))$，并首次导出实例依赖下界证明该速率在指数意义上是最优的。

**[Reducing The Scope Of Language Models](llm_alignment/reducing_the_scope_of_language_models.md)**

:   系统评估 LLM "范围限制"（scoping）方法——让部署在特定用途的 LLM 只响应域内查询、拒绝所有域外请求。在 3 个模型家族×多种任务上比较 prompting / SFT / DPO / 探针 / Circuit Breakers (CB)，发现 SFT 在高数据多样性下最强、CB 在低多样性下最强、分层组合 (SFT→CB) 保留两者优势——关键发现是范围限制的可行性高度依赖训练数据多样性。

**[Rethinking Direct Preference Optimization In Diffusion Models](llm_alignment/rethinking_direct_preference_optimization_in_diffusion_models.md)**

:   提出两个正交改进增强扩散模型偏好优化：(1) 稳定参考模型更新策略放松冻结参考模型并通过正则化鼓励探索；(2) 时间步感知训练策略缓解跨时间步奖励尺度不平衡。二者可嵌入多种偏好优化算法，在人类偏好评估基准上提升SOTA。

**[Safenlidb A Privacy-Preserving Safety Alignment Framework For Llm-Based Natural ](llm_alignment/safenlidb_a_privacy-preserving_safety_alignment_framework_for_llm-based_natural_.md)**

:   提出SafeNlidb框架，通过安全感知数据合成管线和交替偏好优化策略，实现LLM驱动的自然语言数据库接口（NLIDB）在安全推理与SQL生成之间的联合优化，有效防御隐式推理攻击下的隐私泄露。

**[When Human Preferences Flip An Instance-Dependent Robust Loss For Rlhf](llm_alignment/when_human_preferences_flip_an_instance-dependent_robust_loss_for_rlhf.md)**

:   针对人类偏好标注中普遍存在的"偏好翻转"问题，提出 FA-DPO（Flipping-Aware DPO），将标注过程建模为"真实意图 + 实例依赖翻转概率"两阶段，通过修正 BT 模型损失和迭代优化翻转估计模块，在多种噪声场景下显著提升对齐鲁棒性，实例依赖翻转率高时比 DPO 提升 16.7%。

---

## 🖼️ 图像恢复 { #image_restoration }

**[Blur-Robust Detection Via Feature Restoration An End-To-End Framework For Prior-](image_restoration/blur-robust_detection_via_feature_restoration_an_end-to-end_framework_for_prior-.md)**

:   提出 JFD3 端到端双分支框架，在特征域而非图像域进行去模糊，并利用频率结构先验引导检测网络，实现运动模糊条件下红外无人机目标的高精度实时检测。

**[Clear Nights Ahead Towards Multi-Weather Nighttime Image Res](image_restoration/clear_nights_ahead_towards_multi-weather_nighttime_image_res.md)**

:   首次定义并探索多天气夜间图像复原任务，构建 AllWeatherNight 数据集（8K 训练 + 1K 合成测试 + 1K 真实测试），提出 ClearNight 统一框架通过 Retinex 双先验引导和天气感知动态专一性-共性协作，一阶段同时移除雾/雨条/雨滴/雪/flare 复合退化，仅 2.84M 参数全面超越 SOTA。

**[Clearair A Human-Visual-Perception-Inspired All-In-One Image Restoration](image_restoration/clearair_a_human-visual-perception-inspired_all-in-one_image_restoration.md)**

:   受人类视觉感知（HVP）启发，提出一种从粗到细的统一图像复原框架 ClearAIR，通过 MLLM 质量评估 → 语义区域感知 → 退化类型识别 → 内部线索复用四阶段逐步恢复图像质量，在多种退化任务上取得 SOTA。

**[Hard Vs Noise Resolving Hard-Noisy Sample Confusion In Recommender Systems Via L](image_restoration/hard_vs_noise_resolving_hard-noisy_sample_confusion_in_recommender_systems_via_l.md)**

:   提出 LLMHNI 框架，利用 LLM 产生的语义相关性和逻辑相关性两类辅助信号，解决推荐系统中困难样本与噪声样本难以区分的问题，显著提升去噪推荐性能。

**[Hq-Svc Towards High-Quality Zero-Shot Singing Voice Conversion In Low-Resource S](image_restoration/hq-svc_towards_high-quality_zero-shot_singing_voice_conversion_in_low-resource_s.md)**

:   提出 HQ-SVC 框架，基于解耦音频编解码器（FACodec）联合提取内容与说话人特征，结合增强语音适配模块（EVA）融合音高、能量等声学特征，通过 DDSP + 扩散模型渐进式优化，在单张 RTX 3090、不到 80 小时歌声数据条件下实现了超越大规模训练基线的零样本歌声转换质量，并附带支持语音超分辨率任务。

**[Iclr Inter-Chrominance And Luminance Interaction For Natural Color Restoration I](image_restoration/iclr_inter-chrominance_and_luminance_interaction_for_natural_color_restoration_i.md)**

:   针对HVI色彩空间中色度和亮度分支分布差异大导致互补特征提取不足、以及色度分支间弱相关导致梯度冲突的问题，提出ICLR框架，通过双流交互增强模块(DIEM)和协方差校正损失(CCL)分别从融合增强和统计分布优化两个角度解决，在LOL系列数据集上取得SOTA。

**[Large Language Models Meet Extreme Multi-Label Classification Scaling And Multi-](image_restoration/large_language_models_meet_extreme_multi-label_classification_scaling_and_multi-.md)**

:   本文探索了解码器型LLM在极端多标签分类(XMC)中的有效利用，提出双解码器学习策略和 ViXML 多模态框架，通过结构化提示模板适配LLM embedding + 高效融合视觉元数据，在四个公共数据集上大幅超越 SOTA（最大数据集 P@1 提升 +8.21%），证明"一张图胜过数十亿参数"。

**[Mfmamba A Multi-Function Network For Panchromatic Image Resolution Restoration B](image_restoration/mfmamba_a_multi-function_network_for_panchromatic_image_resolution_restoration_b.md)**

:   提出MFmamba多功能网络，基于UNet++骨架结合Mamba上采样模块（MUB）、双池化注意力（DPA）和多尺度混合交叉块（MHCB），仅使用全色（PAN）图像输入即可同时实现超分辨率、光谱恢复及联合SR与着色三种任务。

**[Refidiff Progressive Refinement Diffusion For Efficient Missing Data Imputation](image_restoration/refidiff_progressive_refinement_diffusion_for_efficient_missing_data_imputation.md)**

:   提出 RefiDiff 框架，通过渐进式 refinement 策略统一 predictive 和 generative 两种缺失值填补范式，结合 Mamba-based denoising network 实现高维混合类型表格数据的高效高精度填补，在 MNAR 场景下尤其突出。

**[Sd-Psfnet Sequential And Dynamic Point Spread Function Netwo](image_restoration/sd-psfnet_sequential_and_dynamic_point_spread_function_netwo.md)**

:   提出基于动态 PSF 机制的级联 CNN 去雨网络 SD-PSFNet，通过多尺度可学习 PSF 字典建模雨滴光学效应，配合自适应门控融合的序列化修复架构，在 Rain100H 达 33.12 dB、RealRain-1k-L 达 42.28 dB 均为 SOTA，对比基线 MPRNet 累计提升 5.04 dB（13.5%）。

**[Spatiotemporal Difference Network For Video Depth Super-Resolution](image_restoration/spatiotemporal_difference_network_for_video_depth_super-resolution.md)**

:   基于视频深度超分辨率（VDSR）中空间非光滑区域和时间变化区域呈长尾分布的统计发现，提出 STDNet，通过空间差异分支（学习空间差异表示进行帧内 RGB-D 自适应聚合）和时间差异分支（利用时间差异表示在变化区域进行运动补偿），在 TarTanAir 数据集上 ×16 超分 RMSE 从 112.04cm 降至 96.80cm，平均超越 SOTA 方法 27.6%-32.6%。

**[Temporal Inconsistency Guidance For Super-Resolution Video Quality Assessment](image_restoration/temporal_inconsistency_guidance_for_super-resolution_video_quality_assessment.md)**

:   提出 TIG-SVQA 框架，首次将时间不一致性（temporal inconsistency）作为显式引导信号融入超分辨率视频质量评估，设计了不一致性高亮空间模块（IHSM）和不一致性引导时间模块（IGTM），在 SFD、MFD 和 Combined-VSR 三个数据集上 SRCC 分别达到 0.950、0.942、0.939，全面超越现有 IQA/VQA 方法。

**[Tmdc A Two-Stage Modality Denoising And Complementation Framework For Multimodal](image_restoration/tmdc_a_two-stage_modality_denoising_and_complementation_framework_for_multimodal.md)**

:   提出 TMDC 两阶段框架，第一阶段在完整数据上学习去噪的 modality-specific 和 modality-common 表示，第二阶段利用可用模态的去噪表示补全缺失模态，首次同时处理 MSA 中的噪声和缺失问题。

**[Zero-Reference Joint Low-Light Enhancement And Deblurring Via Visual Autoregress](image_restoration/zero-reference_joint_low-light_enhancement_and_deblurring_via_visual_autoregress.md)**

:   提出 VAR-LIDE，一个完全无监督的视觉自回归框架，通过 VLM 感知先验引导自适应光照调制、空间-频率 RoPE 和递归相位域调制三大模块，联合解决低光增强与去模糊问题，在无需配对数据的条件下逼近甚至超越监督方法的感知质量。

---

## 🔄 自监督 { #self_supervised }

**[Bce3S Binary Cross-Entropy Based Tripartite Synergistic Learning For Long-Tailed](self_supervised/bce3s_binary_cross-entropy_based_tripartite_synergistic_learning_for_long-tailed.md)**

:   提出 BCE3S，一种基于二元交叉熵（BCE）的三方协同学习框架，将 BCE 式联合学习、BCE 式对比学习和 BCE 式分类器均匀性学习集成在一起，通过 Sigmoid 解耦不同类别的度量来抑制长尾不平衡效应，在 CIFAR10/100-LT、ImageNet-LT 和 iNaturalist2018 上均取得 SOTA。

**[Explanation-Preserving Augmentation For Semi-Supervised Graph Representation Lea](self_supervised/explanation-preserving_augmentation_for_semi-supervised_graph_representation_lea.md)**

:   提出EPA-GRL（Explanation-Preserving Augmentation），利用少量标签训练的GNN explainer识别图的语义子图（explanation subgraph），增强时只扰动非语义部分（marginal subgraph），实现语义保持的图增强，在6个benchmark上显著优于语义无关的随机增强方法。

**[Fedgrpo Privately Optimizing Foundation Models With Group-Relative Rewards From ](self_supervised/fedgrpo_privately_optimizing_foundation_models_with_group-relative_rewards_from_.md)**

:   提出 FedGRPO，将大模型优化重新定义为基于奖励的评估过程，通过能力感知的专家选择和联邦组相对策略优化（仅传输标量奖励信号），实现了隐私保护且通信效率极高的联邦基础模型优化，在数学推理和问答任务上性能接近甚至超越集中式 GRPO。

**[From Pretrain To Pain Adversarial Vulnerability Of Video Foundation Models Witho](self_supervised/from_pretrain_to_pain_adversarial_vulnerability_of_video_foundation_models_witho.md)**

:   提出 Transferable Video Attack (TVA)，仅利用开源视频基础模型（VFM）的嵌入空间即可生成对抗扰动，无需任何下游任务知识便能有效攻击24个视频任务上的下游模型和多模态LLM。

**[Improving Region Representation Learning From Urban Imagery With Noisy Long-Capt](self_supervised/improving_region_representation_learning_from_urban_imagery_with_noisy_long-capt.md)**

:   提出 UrbanLN 框架，通过长文本感知的位置编码插值策略和数据-模型双层噪声抑制机制，改善基于 LLM 生成描述的城市区域表征学习。

**[Let The Void Be Void Robust Open-Set Semi-Supervised Learning Via Selective Non-](self_supervised/let_the_void_be_void_robust_open-set_semi-supervised_learning_via_selective_non-.md)**

:   提出 SkipAlign 框架，在对比学习的传统 pull/push 操作之外引入第三种 "skip" 操作，对低置信度样本选择性跳过对齐（只做温和排斥），使 ID 类形成紧凑"星系"、OOD 样本自然散布于"星际虚空"，在未见过的 OOD 检测中平均 AUC 提升 +3.1，最高 +7.1。

**[Movsemcl Movement-Semantics Contrastive Learning For Trajectory Similarity Exten](self_supervised/movsemcl_movement-semantics_contrastive_learning_for_trajectory_similarity_exten.md)**

:   提出 MovSemCL 框架，将 GPS 轨迹转化为运动语义特征（位移向量 + 航向角 + Node2Vec 空间图嵌入），通过 patch 级双层注意力实现层次编码（复杂度从 $O(L^2)$ 降为近线性），并设计曲率引导增广（CGA）保留转弯/路口等行为关键片段，在轨迹检索任务上 mean rank 接近理想值 1，推理延迟降低 43.4%。

**[Neurobridge Bio-Inspired Self-Supervised Eeg-To-Image Decoding Via Cognitive Pri](self_supervised/neurobridge_bio-inspired_self-supervised_eeg-to-image_decoding_via_cognitive_pri.md)**

:   提出NeuroBridge框架，通过认知先验增强（CPA，非对称增广模拟感知变异性）和共享语义投影器（SSP，双向对齐到统一语义空间），在THINGS-EEG数据集200类零样本EEG-图像检索任务上达到63.2% Top-1（+12.3%）和89.9% Top-5（+10.2%），大幅超越现有SOTA。

**[Robust Tabular Foundation Models](self_supervised/robust_tabular_foundation_models.md)**

:   提出 RTFM——一种模型无关的对抗训练框架，通过在合成数据生成器的参数空间上做 min-max 优化（最大化 TFM 与传统树模型之间的"最优性差距"），仅用不到 10 万额外合成数据集就显著提升了 TabPFN V2 在多个表格基准上的表现。

**[Self-Supervised Inductive Logic Programming](self_supervised/self-supervised_inductive_logic_programming.md)**

:   提出自监督归纳逻辑编程（SS-ILP）新设定及 Poker 系统，仅从少量正标签样本和无标签样本出发，自动生成正负样本，配合最大化通用的二阶确定性范式（SONF）背景理论，在无负样本情况下学习含递归和谓词发明的逻辑程序。

**[Spikingformer A Key Foundation Model For Spiking Neural Networks](self_supervised/spikingformer_a_key_foundation_model_for_spiking_neural_networks.md)**

:   提出 Spikingformer，通过将 MS Residual 与 Self-Attention 以 spike-driven 方式结合，解决 Spikformer 中 SEW Residual 导致的非脉冲计算问题，同时保持全局建模能力。

**[Towards Llm-Empowered Knowledge Tracing Via Llm-Student Hierarchical Behavior Al](self_supervised/towards_llm-empowered_knowledge_tracing_via_llm-student_hierarchical_behavior_al.md)**

:   提出 L-HAKT 框架，首次将 LLM 双 Agent 与双曲几何结合用于知识追踪：教师 Agent 解析题目语义并构建层级知识图谱，学生 Agent 模拟个体学习行为生成合成数据，通过双曲空间对比对齐校准合成数据与真实数据的分布差异，在四个教育数据集上 AUC 最高达 80.29%，相比 GKT 基线在 EdNet 上 AUC 提升 13.03%。

---

## 🌐 多语言/翻译 { #multilingual_mt }

**[Bridging The Multilingual Safety Divide Efficient Culturally-Aware Alignment For](multilingual_mt/bridging_the_multilingual_safety_divide_efficient_culturally-aware_alignment_for.md)**

:   本文综合多项实证研究，揭示LLM安全机制在低资源语言和代码混合场景下的严重失效，并提出基于参数高效安全引导、文化驱动偏好数据和社区参与式对齐的资源感知蓝图。

**[Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models For Multi-D](multilingual_mt/consensus-aligned_neuron_efficient_fine-tuning_large_language_models_for_multi-d.md)**

:   提出 CANEFT，通过互信息（MI）识别 LLM 中跨域一致对齐的神经元（consensus-aligned neurons），仅微调这些神经元即可实现多域机器翻译的高效适应，在 3 个 LLM、10 个翻译域上超越 LoRA 等 PEFT 基线，且无需额外参数。

**[Focusing On Language Revealing And Exploiting Language Attention Heads In Multil](multilingual_mt/focusing_on_language_revealing_and_exploiting_language_attention_heads_in_multil.md)**

:   本文提出LAHIS方法，仅需一次前向-后向传播即可高效识别多语言LLM中的语言特异性和语言通用性注意力头，并展示了通过调控这些头来实现跨语言注意力转移、缓解非目标语言生成问题，以及仅用14-20个可训练参数就能提升多语言QA性能的能力。

**[Gloctm Cross-Lingual Topic Modeling Via A Global Context Space](multilingual_mt/gloctm_cross-lingual_topic_modeling_via_a_global_context_space.md)**

:   提出GloCTM，通过双路径VAE架构（局部语言路径+全局上下文路径）结合Polyglot Augmentation（跨语言近邻词扩充输入）、KL散度内部对齐、统一解码器结构对齐和CKA语义对齐四重机制，在3个跨语言数据集上全面超越现有方法的主题质量和跨语言对齐度。

**[How Does Alignment Enhance Llms Multilingual Capabilities A Language Neurons Per](multilingual_mt/how_does_alignment_enhance_llms_multilingual_capabilities_a_language_neurons_per.md)**

:   提出三元神经元分类（语言特定/语言相关/通用），将 LLM 多语言推理分为四阶段分析，发现多语言对齐通过增加语言相关神经元（减少语言特定神经元）来提升性能，且在未训练语言上也产生"自发多语言对齐"效应。

**[Mitigating Content Effects On Reasoning In Language Models Through Fine-Grained ](multilingual_mt/mitigating_content_effects_on_reasoning_in_language_models_through_fine-grained_.md)**

:   通过激活转向（activation steering）技术缓解 LLM 中的内容效应偏见——模型将内容可信度与形式逻辑有效性混淆的问题，提出 K-CAST（基于 kNN 的条件激活转向）方法，在不响应静态转向的模型上实现高达 15% 的形式推理准确率提升。

**[Nadir Differential Attention Flow For Non-Autoregressive Transliteration In Indi](multilingual_mt/nadir_differential_attention_flow_for_non-autoregressive_transliteration_in_indi.md)**

:   提出 NADIR，一种结合差分 Transformer 和混合专家（MoE）的非自回归（NAR）多语言音译架构，在印度语言音译任务上实现了 13× 以上的推理加速，同时将 NAR 模型的幻觉错误（重复、替换、遗漏、插入）大幅降低，缩小了与自回归模型之间的精度差距。

**[Stellar Scene Text Editor For Low-Resource Languages And Real-World Data](multilingual_mt/stellar_scene_text_editor_for_low-resource_languages_and_real-world_data.md)**

:   提出 STELLAR 框架，通过语言自适应字形编码器和合成预训练+真实微调的两阶段训练策略，实现韩语/阿拉伯语/日语等低资源语言的场景文本编辑，并提出可解释的 TAS 指标无需 ground truth 评估字体/颜色/背景风格保持，韩语识别准确率从基线最高 22.1% 飙升至 80.4%。

**[Vidia2Std A Parallel Corpus And Methods For Low-Resource Vietnamese Dialect-To-S](multilingual_mt/vidia2std_a_parallel_corpus_and_methods_for_low-resource_vietnamese_dialect-to-s.md)**

:   ViDia2Std 构建了首个覆盖越南全部 63 个省份的手工标注越南语方言-标准语平行语料库（13,000+ 句对），并评估了多种 seq2seq 模型在方言归一化任务上的表现，证明方言归一化作为预处理步骤能显著提升机器翻译和情感分析等下游任务的性能。

**[X-Mutest A Multilingual Benchmark For Explainable Hate Speech Detection And A No](multilingual_mt/x-mutest_a_multilingual_benchmark_for_explainable_hate_speech_detection_and_a_no.md)**

:   提出 X-MuTest，一个多语言可解释仇恨言论检测基准，覆盖多种语言和文化背景，评估 LLM 不仅检测仇恨言论的能力，更关注其提供可解释性理由的能力，发现当前模型在多语言和跨文化场景中存在显著性能差异。

---

## 🎬 视频生成 { #video_generation }

**[3D4D An Interactive Editable 4D World Model Via 3D Video Generation](video_generation/3d4d_an_interactive_editable_4d_world_model_via_3d_video_generation.md)**

:   提出3D4D交互式4D可视化框架，集成WebGL与Supersplat渲染，通过四模块后端管线将静态图片/文本转化为可编辑4D场景，引入VLM引导的注视点渲染策略实现60fps实时交互，在CLIP Consistency和CLIP Score上达到SOTA。

**[Dreamrunner Fine-Grained Compositional Story-To-Video Genera](video_generation/dreamrunner_fine-grained_compositional_story-to-video_genera.md)**

:   提出 DreamRunner 框架，通过 LLM 双层规划 + 检索增强运动先验学习 + 时空区域3D注意力模块(SR3AI)，实现细粒度可控的多角色多事件故事视频生成。

**[Filmweaver Weaving Consistent Multi-Shot Videos With Cache-Guided Autoregressive](video_generation/filmweaver_weaving_consistent_multi-shot_videos_with_cache-guided_autoregressive.md)**

:   提出 FilmWeaver 框架，通过双层缓存（Shot Cache + Temporal Cache）引导自回归扩散模型，实现任意长度、跨镜头一致性的多镜头视频生成。

**[Genvidbench A 6-Million Benchmark For Ai-Generated Video Detection](video_generation/genvidbench_a_6-million_benchmark_for_ai-generated_video_detection.md)**

:   提出 GenVidBench——首个 678 万级 AI 生成视频检测数据集，具备跨源（cross-source）和跨生成器（cross-generator）特性，覆盖 11 种 SOTA 视频生成器，并提供丰富的语义标注。

**[Mask2Iv Interaction-Centric Video Generation Via Mask Trajectories](video_generation/mask2iv_interaction-centric_video_generation_via_mask_trajectories.md)**

:   提出 Mask2IV，一个两阶段解耦框架——先预测交互者和物体的 mask 运动轨迹，再基于轨迹生成视频——实现了无需密集 mask 标注的、以交互为中心的可控视频生成，支持人-物交互和机器人操作两个场景。

**[Mofu Scale-Aware Modulation And Fourier Fusion For Multi-Subject Video Generatio](video_generation/mofu_scale-aware_modulation_and_fourier_fusion_for_multi-subject_video_generatio.md)**

:   提出 MoFu，通过 Scale-Aware Modulation（LLM 引导的尺度感知调制）和 Fourier Fusion（基于 FFT 的排列不变特征融合）两个核心模块，同时解决多主体视频生成中的**尺度不一致**和**排列敏感性**两大挑战，并构建了 MoFu-1M 训练数据集和 MoFu-Bench 评测基准。

**[Motioncharacter Fine-Grained Motion Controllable Human Video Generation](video_generation/motioncharacter_fine-grained_motion_controllable_human_video_generation.md)**

:   提出 MotionCharacter 框架，通过将运动解耦为动作类型和运动强度两个独立可控维度，实现高保真人体视频生成中的细粒度运动控制和身份一致性保持。

**[Omnivdiff Omni Controllable Video Diffusion For Generation And Understanding](video_generation/omnivdiff_omni_controllable_video_diffusion_for_generation_and_understanding.md)**

:   提出 OmniVDiff，一个统一的可控视频扩散框架，通过将多种视觉模态（RGB、深度、分割、Canny）在颜色空间中联合建模，并引入自适应模态控制策略（AMCS），在单一扩散模型中同时支持文本条件生成、X 条件生成和视频理解三种任务，在 VBench 上达到 SOTA。

**[Phased One-Step Adversarial Equilibrium For Video Diffusion Models](video_generation/phased_one-step_adversarial_equilibrium_for_video_diffusion_models.md)**

:   提出 V-PAE（Video Phased Adversarial Equilibrium），通过**稳定性预热 + 统一对抗均衡**两阶段蒸馏框架，将大规模视频扩散模型（如 Wan2.1-I2V-14B）压缩至单步生成，实现 100 倍加速，在 VBench-I2V 上平均质量超越已有加速方法 5.8%。

**[Seeing The Unseen Zooming In The Dark With Event Cameras](video_generation/seeing_the_unseen_zooming_in_the_dark_with_event_cameras.md)**

:   提出首个事件驱动低光视频超分（LVSR）框架 RetinexEVSR，通过 Retinex 启发的双向融合策略（RBF）——先用光照图引导事件特征去噪（IEE），再用增强后的事件特征恢复反射率细节（ERE），在 SDSD 基准上实现 2.95dB 增益且运行时间减少 65%。

---

## 🔗 因果推理 { #causal_inference }

**[Causal Structure Learning For Dynamical Systems With Theoretical Score Analysis](causal_inference/causal_structure_learning_for_dynamical_systems_with_theoretical_score_analysis.md)**

:   提出 CaDyT，结合高斯过程连续时间动力学建模（Adams-Bashforth 积分器实现精确推断）和 MDL 最小描述长度原则进行结构搜索，同时解决不规则采样和因果结构识别两个挑战，在双质点弹簧/菱形图/Rössler 振荡器上大幅超越所有基线（AUPRC 0.79 vs 次优 0.39）。

**[Causally-Grounded Dual-Path Attention Intervention For Objec](causal_inference/causally-grounded_dual-path_attention_intervention_for_objec.md)**

:   提出 Owl 框架，通过结构因果模型将视觉/文本注意力建模为中介变量，引入 VTACR 指标量化跨模态注意力失衡，设计 VTACR 引导的自适应注意力调制 + 双路径对比解码策略，在 POPE 和 CHAIR 上实现 SOTA 的幻觉抑制效果。

**[Hallucinate Less By Thinking More Aspect-Based Causal Absten](causal_inference/hallucinate_less_by_thinking_more_aspect-based_causal_absten.md)**

:   提出 ABCA（Aspect-Based Causal Abstention），一个生成前弃权框架：通过双 Agent 辩论发现"方面变量"（如学科、法律语境、时间框架）来激活 LLM 不同的知识分支，用 AIPW 双鲁棒估计器计算因果效应，基于质心角偏差（CAD）检测知识冲突（Type-1）或知识不足（Type-2），在 TruthfulQA 上达到 91.4% 准确率，不可回答问题识别率 96.4%（远超基线的 44%）。

**[I-Cam-Uv Integrating Causal Graphs Over Non-Identical Variable Sets Using Causal](causal_inference/i-cam-uv_integrating_causal_graphs_over_non-identical_variable_sets_using_causal.md)**

:   提出 I-CAM-UV 方法，通过对多个变量集不同的 CAM-UV 因果图结果进行一致性约束枚举，恢复因未观测变量而丢失的因果关系，并设计基于不一致代价单调性的最优优先搜索算法高效求解。

**[Ktcf Actionable Recourse In Knowledge Tracing Via Counterfactual Explanations Fo](causal_inference/ktcf_actionable_recourse_in_knowledge_tracing_via_counterfactual_explanations_fo.md)**

:   提出 KTCF，一种面向知识追踪（KT）的反事实解释生成方法，通过考虑知识概念间关系生成稀疏且可操作的反事实解释，并将其后处理为顺序化的教学指令，在有效性、稀疏性和可操作性指标上全面超越基线方法。

**[Learning Subgroups With Maximum Treatment Effects Without Causal Heuristics](causal_inference/learning_subgroups_with_maximum_treatment_effects_without_causal_heuristics.md)**

:   在 SCM 框架下证明最大处理效应子群必须具有同质点效应（定理1），在分区模型假设下证明最优子群发现可化简为标准监督学习（定理2），用 CART+Gini 指数即可实现——在 77 个 ACIC-2016 半合成数据集上均值处理效应 10.54（vs 次优 7.84），51.9% 排名第一。

**[Multi-Agent Undercover Gaming Hallucination Removal Via Coun](causal_inference/multi-agent_undercover_gaming_hallucination_removal_via_coun.md)**

:   MUG 将多 Agent 辩论（MAD）重新定义为"谁是卧底"社交推理游戏——通过图像反事实编辑（修改参考图片）引入信息不对称，让一个 Agent 持有修改后的图片作为"卧底"，其他 Agent 通过推理和投票识别卧底（幻觉来源），在 HallusionBench 上 Qwen2.5VL-7B 从 46.4% 提升到 53.8%。

**[Skill Path Unveiling Language Skills From Circuit Graphs](causal_inference/skill_path_unveiling_language_skills_from_circuit_graphs.md)**

:   提出 Skill Path 概念及三步框架（分解-剪枝-因果中介），从电路图中提取语言模型特定技能的线性路径，定量验证了技能的分层性（Stratification）和包容性（Inclusiveness）两大猜想。

**[Sparse Additive Model Pruning For Order-Based Causal Structure Learning](causal_inference/sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)**

:   提出 SARTRE 框架，利用随机化树嵌入与组稀疏回归学习稀疏加性模型，替代 CAM-pruning 中基于假设检验的冗余边修剪，在基于拓扑序的因果结构学习中实现显著加速且精度不降。

---

## ⚡ LLM效率 { #llm_efficiency }

**[A Content-Preserving Secure Linguistic Steganography](llm_efficiency/a_content-preserving_secure_linguistic_steganography.md)**

:   提出首个内容保持型语言隐写术范式CLstega，通过微调掩码语言模型（MLM）来可控地变换预测分布，将秘密信息嵌入到不做任何修改的原始文本中，实现了100%提取成功率和近乎完美的安全性（隐写分析检测准确率接近随机猜测的0.5）。

**[Attention Retention For Continual Learning With Vision Transformers](llm_efficiency/attention_retention_for_continual_learning_with_vision_transformers.md)**

:   提出ARCL-ViT框架，通过注意力掩码生成和梯度掩码两步策略防止ViT在持续学习中的注意力漂移，在ImageNet-R和CIFAR-100上取得SOTA结果，证明保持注意力模式是解决灾难性遗忘的关键。

**[Collaborative Llm Numerical Reasoning With Local Data Protection](llm_efficiency/collaborative_llm_numerical_reasoning_with_local_data_protection.md)**

:   提出一种大小模型协作框架，通过对本地查询进行"主题迁移+数值替换"的两阶段匿名化来保护敏感数据，同时让远端 GPT-4 以可执行 Python 代码（即插即用工具）形式返回推理方案，本地仅需做数值回代即可获得答案，在 FinQA 和 MultiHiertt 上准确率提升 16-44% 且数据泄露降低 2-45%。

**[Harnessing The Unseen The Hidden Influence Of Intrinsic Knowledge In Long-Contex](llm_efficiency/harnessing_the_unseen_the_hidden_influence_of_intrinsic_knowledge_in_long-contex.md)**

:   首次系统研究长上下文语言模型中参数知识(parametric knowledge)对生成的影响，发现其影响随上下文长度增长而增强，且现有方法提升外部检索能力会抑制参数召回能力，据此提出Hybrid Needle-in-a-Haystack测试来同时评估两种能力。

**[Intermoe Individual-Specific 3D Human Interaction Generation Via Dynamic Tempora](llm_efficiency/intermoe_individual-specific_3d_human_interaction_generation_via_dynamic_tempora.md)**

:   提出 InterMoE，通过 Dynamic Temporal-Selective MoE 架构解决文本驱动的双人 3D 交互运动生成中的个体特征保持和语义忠实度问题：Synergistic Router 融合语义和运动学特征引导路由，Dynamic Temporal Selection 让专家动态选择关键时间帧，在 InterHuman 上 FID 降低 9%、InterX 上降低 22%。

**[Judge Q Trainable Queries For Optimized Information Retention In Kv Cache Evicti](llm_efficiency/judge_q_trainable_queries_for_optimized_information_retention_in_kv_cache_evicti.md)**

:   提出Judge Q，在模型词表中引入可训练的soft token，训练其注意力模式对齐实际解码token的注意力模式，使其在prefill阶段能替代局部窗口查询来评估KV cache重要性，从而更好地保留全局信息，在LongBench上提升~1分，RULER上提升3+分。

**[Learning From The Undesirable Robust Adaptation Of Language Models Without Forge](llm_efficiency/learning_from_the_undesirable_robust_adaptation_of_language_models_without_forge.md)**

:   提出 Learning-from-the-Undesirable (LfU)，一种面向 SFT 的正则化方法，通过对辅助模型施加梯度上升模拟"不良行为"，再通过表示级一致性损失约束原模型与不良模型的内部表征保持一致，有效缓解有限数据微调中的过拟合、遗忘和对抗脆弱性问题。

**[Microevoeval A Systematic Evaluation Framework For Image-Based Microstructure Ev](llm_efficiency/microevoeval_a_systematic_evaluation_framework_for_image-based_microstructure_ev.md)**

:   提出 MicroEvoEval，首个面向图像级微观结构演化预测的标准化基准：涵盖 4 个代表性物理任务（平面波、晶粒生长、旋节分解、枝晶凝固）、14 个模型（5 个领域特定 + 9 个通用时空架构）、多维度评估（数值精度 + 物理保真度 + 计算效率），发现现代通用架构（如 VMamba）在长期稳定性和物理保真度上优于领域特定模型，且计算效率高一个数量级。

**[Think How Your Teammates Think Active Inference Can Benefit Decentralized Execut](llm_efficiency/think_how_your_teammates_think_active_inference_can_benefit_decentralized_execut.md)**

:   提出 AIM（Active Inference Modeling）框架，在去中心化多智能体强化学习中，不依赖通信机制，仅基于局部观测建模队友的主动推理过程（感知-信念-动作三重肖像），并通过准确性-相关性双重过滤机制选择性融合队友信念，在 SMAC、SMACv2、MPE 和 GRF 四大基准上取得最优或接近最优表现。

---

## 👥 社会计算 { #social_computing }

**[Beyond Detection Exploring Evidence-Based Multi-Agent Debate For Misinformation ](social_computing/beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)**

:   本文提出ED2D框架，在多智能体辩论（MAD）系统中引入证据检索模块来增强虚假信息检测准确率，并通过受控人类实验首次对比了AI生成的辩论稿与专家人工fact-check在说服力和信念纠正方面的效果，揭示了AI辩论系统在正确时具有专家级说服力、但在错误时可能加剧误导的双刃剑效应。

**[Cross-Modal Prompting For Balanced Incomplete Multi-Modal Emotion Recognition](social_computing/cross-modal_prompting_for_balanced_incomplete_multi-modal_emotion_recognition.md)**

:   提出 Cross-modal Prompting (ComP) 方法，通过渐进式提示生成+跨模态知识传播+动态调度器来解决不完整多模态情感识别中的模态不平衡问题，在 4 个数据集、 7 种缺失率下均达到 SOTA。

**[Fact2Fiction Targeted Poisoning Attack To Agentic Fact-Check](social_computing/fact2fiction_targeted_poisoning_attack_to_agentic_fact-check.md)**

:   提出 Fact2Fiction，首个针对 Agent 化事实核查系统（如 DEFAME、InFact）的投毒攻击框架：通过 Planner Agent 模拟声明分解生成子问题，利用系统的 justification 反向工程关键推理点来制作定向恶意证据，并按重要性分配投毒预算，在仅 1% 投毒率下比 SOTA PoisonedRAG 高 8.9%-21.2% 的攻击成功率。

**[Factguard Event-Centric And Commonsense-Guided Fake News Detection](social_computing/factguard_event-centric_and_commonsense-guided_fake_news_detection.md)**

:   提出 FactGuard 框架，利用 LLM 提取事件核心内容（去风格化）并生成常识推理，通过 Rationale Usability Evaluator 动态评估 LLM 建议的可信度，并通过知识蒸馏获得无需 LLM 的轻量版 FactGuard-D，在假新闻检测中兼顾鲁棒性和效率。

**[Factorut Controlling Untrusted Ai By Monitoring Their Plans](social_computing/factorut_controlling_untrusted_ai_by_monitoring_their_plans.md)**

:   研究不可信 AI 做任务分解、可信 AI 做执行的 Factor(U,T) 协议安全性，发现监控分解计划的 AUROC 仅 0.52（接近随机），而监控具体代码实现可达 0.96——恶意意图在抽象计划中难以检测但在具体实现中暴露，结论是"结构性预防（可信分解器）优于事后监控"。

**[Multi-Modal Dynamic Proxy Learning For Personalized Multiple Clustering](social_computing/multi-modal_dynamic_proxy_learning_for_personalized_multiple_clustering.md)**

:   本文提出Multi-DProxy框架，通过门控跨模态融合、双约束代理优化和动态候选词管理三大创新机制，利用可学习的文本代理实现个性化多重聚类，在全部公开基准上达到SOTA。

**[Reasoning About The Unsaid Misinformation Detection With Omission-Aware Graph In](social_computing/reasoning_about_the_unsaid_misinformation_detection_with_omission-aware_graph_in.md)**

:   提出OmiGraph，首个基于"遗漏感知"的虚假信息检测框架，通过构建遗漏感知图、利用LLM推理遗漏意图、以及遗漏导向的消息传递与聚合机制，从"未说出的内容"中提取欺骗模式，在双语数据集上平均提升+5.4% F1和+5.3% ACC。

**[Scenejaileval A Scenario-Adaptive Multi-Dimensional Framework For Jailbreak Eval](social_computing/scenejaileval_a_scenario-adaptive_multi-dimensional_framework_for_jailbreak_eval.md)**

:   提出SceneJailEval，一个场景自适应的多维度越狱评估框架，定义14个越狱场景和10个评估维度，通过场景分类→维度动态选择→多维检测→加权危害评分的流程，在自建数据集上F1达0.917（超SOTA 6%），在JBB上达0.995（超SOTA 3%），同时支持危害程度量化而非仅二分类。

**[T2Agent A Tool-Augmented Multimodal Misinformation Detection Agent With Monte Ca](social_computing/t2agent_a_tool-augmented_multimodal_misinformation_detection_agent_with_monte_ca.md)**

:   提出 T2Agent，一个集成可扩展工具集与蒙特卡洛树搜索（MCTS）的虚假信息检测智能体，通过多源验证机制将检测任务分解为针对不同伪造源的子任务，在 MMfakebench 上以 GPT-4o 为骨干将基线 MMDAgent 的准确率提升 28.7%，达到新 SOTA。

---

## 💻 代码智能 { #code_intelligence }

**[Diffbench Meets Diffagent End-To-End Llm-Driven Diffusion Ac](code_intelligence/diffbench_meets_diffagent_end-to-end_llm-driven_diffusion_ac.md)**

:   提出DiffBench（604个扩散模型加速任务的评估基准，分5个难度等级）和DiffAgent（集成规划-编码-调试三Agent + 遗传算法选择器的闭环框架），在Claude Sonnet 4上将扩散加速代码生成通过率从54.30%提升到81.59%，复杂优化任务达成率68.27%。

**[Equacode A Multi-Strategy Jailbreak Approach For Large Language Models Via Equat](code_intelligence/equacode_a_multi-strategy_jailbreak_approach_for_large_language_models_via_equat.md)**

:   提出EquaCode多策略越狱方法，将恶意查询分解为方程求解（B+C+x=A）和代码补全（补全Solver类的solve()方法）的跨域组合，在GPT系列上平均攻击成功率92.78%，在最新模型（Gemini/DeepSeek/Grok）上接近100%。

**[Extracting Events Like Code A Multi-Agent Programming Framework For Zero-Shot Ev](code_intelligence/extracting_events_like_code_a_multi-agent_programming_framework_for_zero-shot_ev.md)**

:   提出 Agent-Event-Coder (AEC)，将零样本事件抽取类比为软件工程流程，用4个专职Agent（Retrieval→Planning→Coding→Verification）协作完成抽取，并将事件schema编码为可执行Python类实现编译器式确定性验证与双循环迭代修正，在5个领域、6个LLM上全面超越零样本基线。

**[Mose Hierarchical Self-Distillation Enhances Early Layer Embeddings](code_intelligence/mose_hierarchical_self-distillation_enhances_early_layer_embeddings.md)**

:   提出 ModularStarEncoder（MoSE），一个 10 亿参数的多出口编码器，通过新颖的自蒸馏机制（高层引导低层训练）显著增强早期层表示，在 CodeSearchNet 等代码理解任务上超越所有开源模型，同时支持灵活的计算-精度权衡部署。

**[Recode Updating Code Api Knowledge With Reinforcement Learning](code_intelligence/recode_updating_code_api_knowledge_with_reinforcement_learning.md)**

:   提出 ReCode 框架，通过基于规则的强化学习（而非 SFT）训练 LLM 在 prompt 中正确利用 API 更新文档完成代码版本迁移，使 7B 模型在 CodeUpdateArena 上超越 32B 模型。

**[Span Benchmarking And Improving Cross-Calendar Temporal Reasoning Of Large Langu](code_intelligence/span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)**

:   提出SPAN跨日历时间推理基准（6种日历×10推理方向×100年范围×37380实例），发现基础LLM平均仅34.5%准确率（无一超过80%），揭示Future-Date Degradation和Calendar Asymmetry Bias两种系统性失败模式，工具增强的Time Agent达95.31%——证明跨日历推理需要外部工具而非参数化知识。

**[Tapas Are Free Training-Free Adaptation Of Programmatic Agen](code_intelligence/tapas_are_free_training-free_adaptation_of_programmatic_agen.md)**

:   TAPA 将 LLM 定位为符号动作空间的"智能调制器"而非直接决策者，通过 LLM 引导的程序合成动态适配程序化 Agent 的符号动作，无需重新训练即可适应动态环境，在网络安全 DDoS 防御（77.7% 网络正常运行率）和群体智能编队控制中表现优异。

**[Towards Better Code Understanding In Decoder-Only Models With Contrastive Learni](code_intelligence/towards_better_code_understanding_in_decoder-only_models_with_contrastive_learni.md)**

:   本文提出CL4D框架，通过对比学习对预训练的decoder-only代码生成模型进行继续预训练，使其能够有效提取代码表示并在代码搜索和克隆检测等理解任务上达到甚至超越同规模encoder-only模型的性能。

---

## 🧮 科学计算 { #scientific_computing }

**[Just Few States Are Enough Randomized Sparse Feedback For Stability Of Dynamical](scientific_computing/just_few_states_are_enough_randomized_sparse_feedback_for_stability_of_dynamical.md)**

:   提出随机稀疏反馈控制框架：控制器在每个时间步仅访问状态向量的随机子集，通过 LMI 联合设计反馈增益矩阵和 Bernoulli 稀疏化参数，在保证渐近均方稳定性（AMSS）的同时最小化所需传感器数量，实验中仅用 0.3% 的状态分量即可达到与全状态反馈可比的性能。

**[Knowledge-Guided Masked Autoencoder With Linear Spectral Mixing And Spectral-Ang](scientific_computing/knowledge-guided_masked_autoencoder_with_linear_spectral_mixing_and_spectral-ang.md)**

:   提出 KARMA 框架，在 ViT-MAE 解码器中嵌入线性光谱混合模型 (LSMM) 作为物理约束，结合 Spectral Angle Mapper (SAM) 损失，提升高光谱遥感图像的重建保真度和下游任务迁移性能。

**[Phys-Liquid A Physics-Informed Dataset For Estimating 3D Geometry And Volume Of ](scientific_computing/phys-liquid_a_physics-informed_dataset_for_estimating_3d_geometry_and_volume_of_.md)**

:   构建了 Phys-Liquid 数据集（97,200 张物理仿真图像 + 3D mesh），基于 Navier-Stokes 方程模拟透明容器内液体的动态形变，并提出四阶段重建管线（分割→多视角 mask 生成→3D 重建→缩放），在仿真和真实场景中实现高精度液体几何与体积估计。

**[Physicscorrect A Training-Free Approach For Stable Neural Pde Simulations](scientific_computing/physicscorrect_a_training-free_approach_for_stable_neural_pde_simulations.md)**

:   提出 PhysicsCorrect，一种无需训练的校正框架，通过将 PDE 残差校正建模为线性化逆问题并预计算伪逆缓存，在推理时以 <5% 计算开销实现最高 100× 误差降低，适用于 FNO/UNet/ViT 等任意预训练神经算子。

**[Pimrl Physics-Informed Multi-Scale Recurrent Learning For Burst-Sampled Spatiote](scientific_computing/pimrl_physics-informed_multi-scale_recurrent_learning_for_burst-sampled_spatiote.md)**

:   提出 PIMRL 框架，针对 burst 采样（短段高频+长间隔）的稀疏时空数据，结合宏观尺度潜空间推理和微观尺度物理校正的双模块架构，通过跨尺度消息传递融合信息，在 5 个 PDE 基准上将误差最多降低 80%。

**[Saot An Enhanced Locality-Aware Spectral Transformer For Solving Pdes](scientific_computing/saot_an_enhanced_locality-aware_spectral_transformer_for_solving_pdes.md)**

:   提出 SAOT（Spectral Attention Operator Transformer），通过线性复杂度的小波注意力（WA）捕获高频局部细节，与傅里叶注意力（FA）的全局感受野经门控融合互补，在 6 个算子学习基准上取得 SOTA，Navier-Stokes 相对误差比 Transolver 下降 22.3%。

**[Scientific Knowledge-Guided Machine Learning For Vessel Power Prediction A Compa](scientific_computing/scientific_knowledge-guided_machine_learning_for_vessel_power_prediction_a_compa.md)**

:   提出物理基线+数据驱动残差的混合建模框架，将海试功率曲线（螺旋桨定律 $P=cV^n$）作为基线，用 XGBoost/NN/PINN 学习残差修正，在稀疏数据区域显著提升外推稳定性和物理一致性。

**[Towards A Foundation Model For Partial Differential Equations Across Physics Dom](scientific_computing/towards_a_foundation_model_for_partial_differential_equations_across_physics_dom.md)**

:   提出 PDE-FM，一个结合空间-频谱 tokenization、物理感知 FiLM 调制和 Mamba 状态空间 backbone 的模块化 PDE foundation model，在 The Well 基准的 12 个跨物理域数据集上平均降低 VRMSE 46%。

---

## 🛰️ 遥感 { #remote_sensing }

**[Asymmetric Cross-Modal Knowledge Distillation Bridging Modalities With Weak Sema](remote_sensing/asymmetric_cross-modal_knowledge_distillation_bridging_modalities_with_weak_sema.md)**

:   提出 Asymmetric Cross-modal Knowledge Distillation (ACKD) 新范式，通过 SemBridge 框架（包含自监督语义匹配 + 最优传输对齐两个即插即用模块）实现弱语义一致性条件下的跨模态知识蒸馏，使不同地理位置采集的多光谱（MS）图像能有效指导 RGB 图像的遥感场景分类。

**[Consistency-Based Abductive Reasoning Over Perceptual Errors Of Multiple Pre-Tra](remote_sensing/consistency-based_abductive_reasoning_over_perceptual_errors_of_multiple_pre-tra.md)**

:   将多个预训练感知模型在新环境中的冲突预测建模为一致性溯因推理问题，通过逻辑程序编码各模型的错误检测规则和领域约束，寻找在保持不一致率低于阈值的同时最大化预测覆盖率的最优假设，在15个航拍测试集上平均F1提升13.6%。

**[Debiasing Machine Learning Predictions For Causal Inference Without Additional G](remote_sensing/debiasing_machine_learning_predictions_for_causal_inference_without_additional_g.md)**

:   针对ML卫星贫困预测因均值回归导致因果处理效应衰减的问题，提出两种无需新标注数据的后处理校正方法——线性校准校正(LCC)和Tweedie局部去收缩——使同一预测地图可在多个下游因果试验中复用（"一图多试"范式），Tweedie校正在模拟和DHS真实数据上实现近无偏的处理效应估计。

**[M3Sr Multi-Scale Multi-Perceptual Mamba For Efficient Spectral Reconstruction](remote_sensing/m3sr_multi-scale_multi-perceptual_mamba_for_efficient_spectral_reconstruction.md)**

:   提出 M3SR，一种基于 Mamba 的多尺度多感知架构，通过空间-频率-光谱三分支并行融合结合 U-Net 多尺度结构，以 2.17M 参数和 100.9G FLOPs 的低计算代价在四个光谱重建基准上超越现有 SOTA 方法。

**[Machine Learning For Sustainable Rice Production Region-Scale Monitoring Of Wate](remote_sensing/machine_learning_for_sustainable_rice_production_region-scale_monitoring_of_wate.md)**

:   提出维度分类方法将水稻节水实践识别解耦为播种维度(DSR vs PTR)和灌溉维度(AWD vs CF)两个独立二分类任务，仅使用Sentinel-1 SAR影像实现播种F1=0.80和灌溉F1=0.74，并在旁遮普邦300万+地块上进行大规模推理，地区级采纳率与政府统计高度相关（Spearman ρ=0.69）。

**[Spatio-Temporal Context Learning With Temporal Difference Convolution For Moving](remote_sensing/spatio-temporal_context_learning_with_temporal_difference_convolution_for_moving.md)**

:   提出 TDCNet，将时间差分和 3D 卷积融合为统一的时间差分卷积 (TDC)，通过重参数化实现推理零额外开销，配合 TDC 引导的时空注意力，在自建 IRSTD-UAV 数据集上 F1 达 97.12%（AP50 93.83%），同时发布 15,106 帧真实红外无人机数据集。

**[Uniabg Unified Adversarial View Bridging And Graph Correspondence For Unsupervis](remote_sensing/uniabg_unified_adversarial_view_bridging_and_graph_correspondence_for_unsupervis.md)**

:   提出双阶段无监督跨视角地理定位框架 UniABG，通过对抗式视角桥接 (VAAB) 消除无人机/卫星视角域差距，再用异构图过滤校准 (HGFC) 净化跨视角关联，在 University-1652 上 Satellite→Drone AP 达 93.29%，超过多数有监督方法。

---

## ✏️ 知识编辑 { #knowledge_editing }

**[Catastrophic Forgetting In Kolmogorov-Arnold Networks](knowledge_editing/catastrophic_forgetting_in_kolmogorov-arnold_networks.md)**

:   首个系统性研究KAN（Kolmogorov-Arnold Networks）中灾难性遗忘行为的工作：建立了遗忘与激活支持重叠和数据内禀维度之间的理论框架，并提出KAN-LoRA用于语言模型的持续微调知识编辑。

**[Hybrid-Dmkg A Hybrid Reasoning Framework Over Dynamic Multimodal Knowledge Graph](knowledge_editing/hybrid-dmkg_a_hybrid_reasoning_framework_over_dynamic_multimodal_knowledge_graph.md)**

:   提出MMQAKE基准和Hybrid-DMKG框架，在动态多模态知识图谱上构建"关系链接预测 + RAG增强LVLM推理"双通道混合推理机制，配合背景反思决策模块，在2-5跳多模态知识编辑问答中显著超越现有方法（LLaVA上H-Acc达29.90%，超IKE 13.52个百分点）。

**[Is The Information Bottleneck Robust Enough Towards Label-Noise Resistant Inform](knowledge_editing/is_the_information_bottleneck_robust_enough_towards_label-noise_resistant_inform.md)**

:   本文揭示了信息瓶颈（IB）原理在标签噪声下的固有脆弱性，提出 LaT-IB 方法，通过将表征解耦为干净标签空间和噪声标签空间两部分，结合"最小-充分-干净"（MSC）准则和三阶段训练框架，在多种噪声条件下实现了对现有 IB 方法的显著超越。

**[Model Editing As A Double-Edged Sword Steering Agent Ethical Behavior Toward Ben](knowledge_editing/model_editing_as_a_double-edged_sword_steering_agent_ethical_behavior_toward_ben.md)**

:   将 Agent 伦理行为引导建模为模型编辑任务（Behavior Editing），提出基于心理学道德理论的三层 BehaviorBench 基准，在 9 个开源模型和 20 个闭源模型上验证了模型编辑可以精确地将 Agent 引导向善意或恶意方向，且单次编辑可导致全局道德对齐偏移。

**[Multiplicative Orthogonal Sequential Editing For Language Models](knowledge_editing/multiplicative_orthogonal_sequential_editing_for_language_models.md)**

:   提出 MOSE（乘法正交序列编辑），用正交矩阵左乘（而非加法更新）参数矩阵来注入新知识，严格保持编辑后矩阵的范数和条件数不变，在序列编辑中实现 12.08% 的性能提升并保留 95.73% 通用能力。

---

## 📚 预训练 { #llm_pretraining }

**[Elspr Evaluator Llm Training Data Self-Purification On Non-Transitive Preference](llm_pretraining/elspr_evaluator_llm_training_data_self-purification_on_non-transitive_preference.md)**

:   ELSPR 将 LLM 评估器的成对偏好建模为锦标赛图，通过强连通分量 (SCC) 识别非传递偏好，提出归一化有向图结构熵指标，并基于图重构过滤有问题的训练数据——过滤后的评估器非传递性降低 13.8%、结构熵降低 0.088，且丢弃数据的人类一致性仅 34.4%（vs 保留数据 52.6%）。

**[Learning Time In Static Classifiers](llm_pretraining/learning_time_in_static_classifiers.md)**

:   提出 Support-Exemplar-Query (SEQ) 学习框架，通过损失函数设计（而非架构修改）为标准前馈分类器注入时序推理能力，利用软DTW将预测序列与类别时序原型对齐，在细粒度图像分类和视频异常检测上均取得提升。

**[No-Regret Strategy Solving In Imperfect-Information Games Via Pre-Trained Embedd](llm_pretraining/no-regret_strategy_solving_in_imperfect-information_games_via_pre-trained_embedd.md)**

:   提出 Embedding CFR 算法，将不完美信息博弈中的信息集映射到连续低维嵌入空间（而非离散聚类），在相同空间开销下实现更快的可利用性收敛和更高质量的策略求解。

**[Scaling And Transferability Of Annealing Strategies In Large Language Model Trai](llm_pretraining/scaling_and_transferability_of_annealing_strategies_in_large_language_model_trai.md)**

:   提出模型无关的预测框架，分解训练损失为前向效应项（学习率积分S）、退火动量项（Adam-style动量积分M）和模型尺寸项N，证明退火策略可从小模型/小batch迁移到大模型/大batch，预测误差MAPE<2%。

**[Uncovering Pretraining Code In Llms A Syntax-Aware Attribution Approach](llm_pretraining/uncovering_pretraining_code_in_llms_a_syntax-aware_attribution_approach.md)**

:   提出SynPrune——首个语法感知的代码成员推断攻击方法，通过识别47种Python语法约定并在计算成员推断分数时剪除语法决定的token（仅保留反映作者特征的token），平均AUROC提升15.4%，可有效检测代码LLM的预训练数据归属。

---

## 🔒 LLM安全 { #llm_safety }

**[Catformer When Continual Learning Meets Spiking Transformers With Dynamic Thresh](llm_safety/catformer_when_continual_learning_meets_spiking_transformers_with_dynamic_thresh.md)**

:   提出 CATFormer，一种基于脉冲视觉 Transformer 的无数据重放持续学习框架，通过上下文自适应的动态放电阈值实现任务特定的神经元兴奋性调节，在长达 100 个任务序列中不仅不遗忘反而准确率提升（"逆向遗忘"现象）。

**[Designing Truthful Mechanisms For Asymptotic Fair Division](llm_safety/designing_truthful_mechanisms_for_asymptotic_fair_division.md)**

:   提出 PRD（Proportional Response with Dummy）机制，首次在渐近公平分配设定下实现了"期望真实性 + 多项式时间可计算 + 高概率无嫉妒"三重保证，且仅需 $m = \Omega(n \log n)$ 个物品，回答了 Manurangsi & Suksompong 提出的开放问题。

**[Hallucination Stations On Some Basic Limitations Of Transformer-Based Language M](llm_safety/hallucination_stations_on_some_basic_limitations_of_transformer-based_language_m.md)**

:   从计算复杂度理论出发证明 Transformer LLM 每步推理复杂度为 $O(N^2 \cdot d)$，基于时间层次定理（Hartmanis-Stearns），任何需要超过此复杂度的计算任务——如 $O(n^3)$ 矩阵乘法、$O(n^k)$ token 组合、TSP 验证等——LLM 必然无法正确完成（即产生幻觉），且 LLM Agent 也无法验证此类任务的正确性。

**[Llm Targeted Underperformance Disproportionately Impacts Vulnerable Users](llm_safety/llm_targeted_underperformance_disproportionately_impacts_vulnerable_users.md)**

:   系统实验表明，主流LLM（GPT-4、Claude 3 Opus、Llama 3-8B）对英语水平较低、教育程度较低、非美国出身的用户，在信息准确性、真实性和拒绝回答方面存在显著的歧视性表现下降，使最脆弱的用户成为最不可靠的信息服务对象。

**[Panda -- Patch And Distribution-Aware Augmentation For Long-Tailed Exemplar-Free](llm_safety/panda_--_patch_and_distribution-aware_augmentation_for_long-tailed_exemplar-free.md)**

:   提出 PANDA 框架，通过 CLIP 引导的语义 patch 移植实现任务内类别平衡，并借助可学习的分布平滑机制缓解任务间分布偏移，以即插即用方式提升基于预训练模型的无样本存储持续学习在长尾场景下的性能。

---

## 🔎 AIGC检测 { #aigc_detection }

**[Actishade Activating Overshadowed Knowledge To Guide Multi-H](aigc_detection/actishade_activating_overshadowed_knowledge_to_guide_multi-h.md)**

:   提出ActiShade框架，通过高斯噪声扰动检测LLM在多跳推理中被"遮蔽"的关键短语，结合定制对比学习检索器获取补充文档，迭代重构查询以减少知识遮蔽导致的错误累积，在HotpotQA/2WikiMQA/MuSiQue上显著超越DRAGIN等SOTA。

**[Baid A Benchmark For Bias Assessment Of Ai Detectors](aigc_detection/baid_a_benchmark_for_bias_assessment_of_ai_detectors.md)**

:   提出 BAID 基准数据集（20.8万样本对，覆盖7类偏见维度、41个子群体），系统评估4个开源 AI 文本检测器在不同人口统计和语言学子群体上的公平性表现，揭示检测器对方言、非正式英语和少数群体文本存在显著的召回率差异。

**[Optimized Algorithms For Text Clustering With Llm-Generated Constraints](aigc_detection/optimized_algorithms_for_text_clustering_with_llm-generated_constraints.md)**

:   提出 LSCK-HC 框架，利用 LLM 生成集合形式的 must-link/cannot-link 约束（而非传统成对约束），配合带惩罚项的局部搜索聚类算法，在5个短文本数据集上实现与 SOTA 可比的聚类精度，同时将 LLM 查询次数减少 20 倍以上。

---

## 🗣️ 对话系统 { #dialogue }

**[Emergent Persuasion Will Llms Persuade Without Being Prompted](dialogue/emergent_persuasion_will_llms_persuade_without_being_prompted.md)**

:   研究 LLM 在未被提示说服的情况下是否会自发产生说服行为：发现激活引导（steering）无法可靠诱发说服倾向，但在良性说服数据上的 SFT 微调会导致模型在有害话题上产生涌现性说服行为，揭示了后训练安全风险。

**[Mctsr-Zero Self-Reflective Psychological Counseling Dialogues Generation Via Pri](dialogue/mctsr-zero_self-reflective_psychological_counseling_dialogues_generation_via_pri.md)**

:   提出 MCTSr-Zero 框架，将 MCTS 与领域原则自评估、元提示自适应探索机制结合，用于生成高质量心理咨询多轮对话数据，微调得到的 PsyLLM 在自建的 PsyEval 基准上达到 SOTA。

**[Teaching Large Language Models To Maintain Contextual Faithfulness Via Synthetic](dialogue/teaching_large_language_models_to_maintain_contextual_faithfulness_via_synthetic.md)**

:   提出 Canoe 框架，通过从 Wikidata 三元组合成四类可验证的短形式 QA 数据，配合 Dual-GRPO（含准确率奖励、长形式代理奖励和格式奖励）同时优化短/长形式生成的忠实度，使 Llama-3-8B 在 11 个下游任务上平均提升 22.6%，超越 GPT-4o。

---

## 📡 信号/通信 { #signal_comm }

**[Task Aware Modulation Using Representation Learning For Upsaling Of Terrestrial ](signal_comm/task_aware_modulation_using_representation_learning_for_upsaling_of_terrestrial_.md)**

:   提出 TAM-RL 框架，将陆地碳通量升尺度问题建模为零样本回归迁移学习任务，用 BiLSTM 任务编码器+FiLM 调制结合碳平衡方程知识引导损失，在 150+ 通量塔站点上将 GPP RMSE 降低 9.6%、NEE R² 提升 43.8%（相较 FLUXCOM-X-BASE）。

**[Text-Guided Channel Perturbation And Pretrained Knowledge Integration For Unifie](signal_comm/text-guided_channel_perturbation_and_pretrained_knowledge_integration_for_unifie.md)**

:   提出 UP-Fusion 统一多模态图像融合框架，通过语义感知通道剪枝 (SCPM)、几何仿射调制 (GAM) 和 CLIP 文本引导通道扰动 (TCPM) 三个模块，用单组权重（仅在红外-可见光数据上训练）同时处理 IVIF 和医学图像融合，在两类任务上均达到 SOTA。

**[Toward Gaze Target Detection Of Young Autistic Children](signal_comm/toward_gaze_target_detection_of_young_autistic_children.md)**

:   针对自闭症儿童注视目标检测中面部注视（6.6%）严重不足的类别不平衡问题，提出 Socially Aware Coarse-to-Fine (SACF) 框架，用微调的 Qwen2.5-VL 作为社交上下文感知门控，将输入路由到社交感知/社交无关两个专家模型，在首创的 AGT 数据集上显著提升了面部注视检测性能（Face L2 在 Sharingan 上降低 13.9%, F1 从 0.753 提升至 0.761）。

---

## ✍️ 文本生成 { #nlp_generation }

**[Automaldesc Large-Scale Script Analysis For Cyber Threat Research](nlp_generation/automaldesc_large-scale_script_analysis_for_cyber_threat_research.md)**

:   提出 AutoMalDesc 自动化静态分析框架，通过迭代自步学习流水线——从 900 个专家标注种子样本出发，经 LoRA 微调 Llama-3.3-70B 生成伪标签，多阶段质量过滤后进行 V2 训练——实现 5 种脚本语言的恶意软件自动分类和行为描述，Batch 脚本检测准确率从 52.7% 提升到 82.4%。

**[C3Tg Conflict-Aware Composite And Collaborative Controlled Text Generation](nlp_generation/c3tg_conflict-aware_composite_and_collaborative_controlled_text_generation.md)**

:   提出 C3TG 框架，通过两阶段方法实现多维度细粒度可控文本生成：生成阶段用加权 KL 散度融合属性分布调整 token 概率，优化阶段用能量函数（分类器分数 + 冲突惩罚项）结合 Feedback Agent 迭代重写，在 17 个属性子类上达到 90.4% 属性准确率且大幅降低毒性。

---

## 📖 NLP理解 { #nlp_understanding }

**[Language Models And Logic Programs For Trustworthy Tax Reasoning](nlp_understanding/language_models_and_logic_programs_for_trustworthy_tax_reasoning.md)**

:   将税法推理重新定义为语义解析任务，让LLM将法规文本和纳税案例翻译为Prolog逻辑程序，由符号求解器执行计算，通过金标准法规+智能检索案例示例+自一致性检查，在SARA数据集上实现86/100的正确率，并将预计部署成本降至15.78美元/人（低于美国人均报税成本的6%）。

**[Understanding Syllogistic Reasoning In Llms From Formal And Natural Language Per](nlp_understanding/understanding_syllogistic_reasoning_in_llms_from_formal_and_natural_language_per.md)**

:   系统评估14个LLM在160个三段论上的推理表现，通过双维度ground truth框架（句法有效性+NLU可信度）揭示顶级模型在形式逻辑上接近完美(99.6%)但自然语言可信度判断仅为随机水平(~52%)——与人类推理模式恰好相反；12/14模型存在显著信念偏差，且few-shot提示反而降低形式推理性能。

---

## ⚛️ 物理学 { #physics }

**[Adaptive Fidelity Estimation For Quantum Programs With Graph](physics/adaptive_fidelity_estimation_for_quantum_programs_with_graph.md)**

:   提出 QuFid 框架，将量子电路建模为有向无环图，通过控制流感知的随机游走刻画噪声传播，利用算子谱特征量化电路复杂度，实现自适应测量预算分配，在保持保真度精度的同时大幅减少测量次数。

**[Data Verification Is The Future Of Quantum Computing Copilots](physics/data_verification_is_the_future_of_quantum_computing_copilots.md)**

:   这是一篇 position paper，提出量子计算 AI 助手（Copilot）必须将数据验证从事后过滤提升为架构级基础——通过三个立场论证：(1) 验证数据是最低要求，(2) 先验约束优于后验过滤，(3) 受物理定律约束的科学领域需要验证感知架构。实验表明无验证数据的 LLM 在电路优化上最高仅达 79% 准确率。

---

## 🌍 地球科学 { #earth_science }

**[Mdaif Robust One-Stop Multi-Degradation-Aware Image Fusion With Language-Driven ](earth_science/mdaif_robust_one-stop_multi-degradation-aware_image_fusion_with_language-driven_.md)**

:   提出 MdaIF 框架，利用视觉语言模型（VLM）提取退化感知语义先验来引导混合专家（MoE）路由和通道注意力调制，实现无需退化类型标注的一站式多退化场景红外-可见光图像融合。

---

## 📂 其他 { #others }

**[A Fast Heuristic Search Approach For Energy-Optimal Profile ](others/a_fast_heuristic_search_approach_for_energy-optimal_profile_.md)**

:   提出基于多目标A*搜索的label-setting方法（Pr-A*），在初始电量未知时高效求解电动车能耗最优路径（profile搜索），通过profile支配关系剪枝避免传统方法中复杂的profile合并操作，在大规模路网上性能接近已知初始电量的标准A*搜索。

**[A Graph-Theoretical Perspective On Law Design For Multiagent](others/a_graph-theoretical_perspective_on_law_design_for_multiagent.md)**

:   将多智能体系统中的法律设计问题（包括"有用法律"和"无责任缺口法律"）形式化为超图上的顶点覆盖问题，证明了两类法律最小化问题都是NP-hard的，并给出了基于超图顶点覆盖近似算法的多项式时间近似方案。

**[A Mind Cannot Be Smeared Across Time](others/a_mind_cannot_be_smeared_across_time.md)**

:   从Stack Theory出发引入时间语义模块，形式化证明存在性时间实现不保持合取（Temporal Gap），提出Chord（要求时间窗口内共同实例化）vs Arpeggio（只需成分在窗口内出现）两种意识假设，论证严格串行硬件在Chord假设下不可能承载需要多个同时贡献者的意识。

**[A New Strategy For Verifying Reach-Avoid Specifications In Neural Feedback Syste](others/a_new_strategy_for_verifying_reach-avoid_specifications_in_neural_feedback_syste.md)**

:   提出FaBRe（Forward and Backward Reachability）策略，首次开发了针对ReLU神经网络控制器的后向可达集过近似和欠近似算法（GSS/ICH/LEB），并将其与前向可达性分析结合，构成统一的reach-avoid验证框架，旨在突破纯前向分析的可扩展性瓶颈。

**[A Phase Transition For Opinion Dynamics With Competing Biase](others/a_phase_transition_for_opinion_dynamics_with_competing_biase.md)**

:   在有向随机图上建模两种对立力量（外部颠覆性偏差 vs 个体顽固性）对二元观点传播的影响，证明系统存在尖锐相变：偏差超过临界阈值 $p_c$ 时群体快速达成新共识，低于阈值则长期处于亚稳极化状态，且临界点仅由度序列的两个简单统计量决定。

**[A Switching Framework For Online Interval Scheduling With Pr](others/a_switching_framework_for_online_interval_scheduling_with_pr.md)**

:   针对不可撤销的在线区间调度问题，提出 SemiTrust-and-Switch 框架和 SmoothMerge 随机算法，通过在信任预测和经典贪心算法之间切换/融合，在预测准确时趋近最优（一致性），预测错误时性能优雅退化（鲁棒性和平滑性），并证明了该框架在特定实例上的紧性。

**[A Topological Rewriting Of Tarskis Mereogeometry](others/a_topological_rewriting_of_tarskis_mereogeometry.md)**

:   在 Coq 定理证明器中扩展 λ-MM 库，将基于 Leśniewski 部分学（mereology）的 Tarski 固体几何重写为具备完整拓扑结构的形式化系统，证明部分学类对应正则开集、满足 Kuratowski 内部公理且具有 Hausdorff（T2）性质，从而为定性空间推理提供了统一的部分学-几何-拓扑理论框架。

**[Agent-Sama State-Aware Mobile Assistant](others/agent-sama_state-aware_mobile_assistant.md)**

:   提出Agent-SAMA，首次将有限状态机（FSM）引入移动端GUI Agent，将UI屏幕建模为状态、用户操作建模为转移，通过四个专门化Agent协作实现状态感知的任务规划、执行验证和错误恢复，在跨App基准上成功率提升最高12%、恢复率提升13.8%。

**[Align When They Want Complement When They Need Human-Centere](others/align_when_they_want_complement_when_they_need_human-centere.md)**

:   揭示了人机协作中"互补性"（complementarity）与"对齐性"（alignment）之间存在根本性权衡——单一模型无法同时优化二者，提出自适应AI集成框架，通过Rational Routing Shortcut（RRS）机制在对齐模型和互补模型之间动态切换，团队准确率较标准AI提升最高9%。

**[Ams-Io-Bench And Ams-Io-Agent Benchmarking And Structured Re](others/ams-io-bench_and_ams-io-agent_benchmarking_and_structured_re.md)**

:   提出AMS-IO-Agent，一个基于LLM的领域专用智能体，通过结构化意图图(Intent Graph)和领域知识库将自然语言设计意图转化为可生产的模拟混合信号IC I/O环设计，配套提出首个AMS I/O环自动化基准AMS-IO-Bench，在28nm CMOS流片中验证了智能体生成的I/O环可直接用于实际芯片制造。

**[An Epistemic Perspective On Agent Awareness](others/an_epistemic_perspective_on_agent_awareness.md)**

:   本文首次将 agent awareness（智能体感知/意识）视为一种知识形式，区分了 de re（关于物理对象的）和 de dicto（关于概念/描述的）两种感知模态，并基于 2D 语义学提出了一个可靠且完备的逻辑系统来刻画这两种模态与标准"事实知识"模态之间的相互作用。

**[Approximation Algorithm For Constrained K-Center Clustering ](others/approximation_algorithm_for_constrained_k-center_clustering_.md)**

:   研究带 cannot-link (CL) 和 must-link (ML) 实例级约束的 k-center 聚类问题，提出基于支配匹配集（dominating matching set, DMS）转化的局部搜索框架，在不相交 CL 集条件下首次通过局部搜索达到最优近似比 2，解决了该领域一个开放问题。

**[Area-Optimal Control Strategies For Heterogeneous Multi-Agen](others/area-optimal_control_strategies_for_heterogeneous_multi-agen.md)**

:   研究异构速度下多追逐者-单逃避者的追逃博弈——定义逃避者安全可达集为所有追逐者-逃避者对的 Apollonius 圆的交集，将捕获策略建模为追逐者最小化/逃避者最大化该交集面积的零和博弈，推导出闭式瞬时最优航向控制律，仿真验证追逐者可系统性缩小安全区域实现保证捕获。

**[Automated Reproducibility Has A Problem Statement Problem](others/automated_reproducibility_has_a_problem_statement_problem.md)**

:   提出基于科学方法的可复现性形式化问题定义，将经验性AI研究表示为假设-实验-解释的图结构，并用LLM自动从20篇论文中提取该结构，经原作者评审验证其有效性。

**[Autonomous Concept Drift Threshold Determination](others/autonomous_concept_drift_threshold_determination.md)**

:   证明了固定阈值不可能在所有场景下最优、动态阈值严格优于静态阈值，并提出DTD算法：在漂移检测信号触发后启动三模型比较阶段，根据候选模型表现自适应调整检测阈值。

**[Bandit Learning In Housing Markets](others/bandit_learning_in_housing_markets.md)**

:   本文首次将多臂老虎机（MAB）框架引入住房市场（单边匹配市场），定义了基于核（core）概念的遗憾值，并分别提出去中心化 ETC 和中心化 UCB 两种算法，证明了 $\mathcal{O}(N\log T / \Delta_{\min}^2)$ 的去中心化遗憾上界与匹配的下界，建立了阶最优性。

**[Bayesian Network Structural Consensus Via Greedy Min-Cut Analysis](others/bayesian_network_structural_consensus_via_greedy_min-cut_analysis.md)**

:   提出 MCBNC 算法，基于最小割（min-cut）分析量化边的结构支持度，并将其嵌入贪心等价搜索（GES）的后向阶段来迭代剪枝融合贝叶斯网络中的冗余边，在不访问数据的情况下生成更稀疏、更精确的共识结构，适用于联邦学习场景。

**[Beyond World Models Rethinking Understanding In Ai Models](others/beyond_world_models_rethinking_understanding_in_ai_models.md)**

:   本文通过三个来自科学哲学的案例研究（多米诺计算机、数学证明、玻尔原子理论），论证世界模型（world models）框架不足以刻画人类级别的"理解"，指出仅靠追踪状态和状态转换无法捕获理解所需的抽象推理、动机洞察和问题情境把握能力。

**[Bilevel Mcts For Amortized O1 Node Selection In Classical Planning](others/bilevel_mcts_for_amortized_o1_node_selection_in_classical_planning.md)**

:   提出双层MCTS（Bilevel MCTS），在MCTS选中的叶节点处运行深度比例预算的最优优先搜索，将节点选择均摊复杂度从 $O(\log N)$ 降至 $O(1)$，辅以树崩塌（Tree Collapsing）减少动作选择步数，最终整合为 Nεbula 规划器，在IPC2018/2023基准上以192.2/230.6解题数（5min/30min）超越LAMA、DecStar、NOLAN、SM-Type-LAMA等全部SOTA。

**[Bipartite Mode Matching For Vision Training Set Search From A Hierarchical Data ](others/bipartite_mode_matching_for_vision_training_set_search_from_a_hierarchical_data_.md)**

:   提出层级数据服务器 + 二部图模式匹配（BMM）框架，通过多粒度层级聚类组织大规模源数据、用匈牙利算法一对一匹配源域和目标域的语义模式（modes），从而搜索出与目标域分布差距最小的训练集，在行人重识别和目标检测任务上显著优于已有训练集搜索方法。

**[Boosting Adversarial Transferability Via Ensemble Non-Attention](others/boosting_adversarial_transferability_via_ensemble_non-attention.md)**

:   提出 NAMEA（Non-Attention Meta Ensemble Attack），首次利用集成模型的非注意力区域（non-attention areas）融合 CNN 和 ViT 的可迁移信息，结合元学习梯度优化，在跨架构对抗迁移性上平均超越 SOTA 方法 AdaEA 和 SMER 分别 15.0% 和 9.6%。

**[Cash Flow Underwriting With Bank Transaction Data Advancing Msme Financial Inclu](others/cash_flow_underwriting_with_bank_transaction_data_advancing_msme_financial_inclu.md)**

:   提出基于银行流水数据的端到端现金流承保工作流，构建首个马来西亚 MSME（中小微企业）银行账单数据集（611 条贷款记录），验证银行交易衍生特征相比传统申请信息可将逻辑回归模型的 AUROC 从 0.672 提升至 0.850，显著增强对缺乏信用记录的中小微企业的信用评估能力。

**[Cat-Net A Cross-Attention Tone Network For Cross-Subject Eeg-Emg Fusion Tone Dec](others/cat-net_a_cross-attention_tone_network_for_cross-subject_eeg-emg_fusion_tone_dec.md)**

:   提出 CAT-Net（Cross-Attention Tone Network），通过空间-时间特征提取分支 + 交叉注意力融合机制 + 域对抗训练，仅用 20 个 EEG 通道和 5 个 EMG 通道实现中文四声调分类，在有声/无声语音条件下分别达到 87.83%/88.08% 准确率，跨被试评估下达到 83.27%/85.10%，全面超越 8 种基线方法。

**[Cellstream Dynamical Optimal Transport Informed Embeddings For Reconstructing Ce](others/cellstream_dynamical_optimal_transport_informed_embeddings_for_reconstructing_ce.md)**

:   提出 CellStream，一种将自编码器与非平衡动态最优传输（unbalanced dynamical OT）联合学习的深度学习框架，从离散时间点的单细胞快照数据中同时学习低维嵌入和连续细胞动态轨迹，在时间一致性和速度一致性上显著优于现有方法。

**[Center-Outward Q-Dominance A Sample-Computable Proxy For Strong Stochastic Domin](others/center-outward_q-dominance_a_sample-computable_proxy_for_strong_stochastic_domin.md)**

:   基于最优传输理论中的中心向外分布函数，提出 q-dominance 关系作为强一阶随机支配（strong FSD）的可计算近似，证明全分位数范围的 q-dominance 可推导出强 FSD，并给出显式样本量阈值控制 Type I 错误，在超参数调优排名和噪声多目标优化中验证了其实用性。

**[Certified Branch-And-Bound Maxsat Solving Extended Version](others/certified_branch-and-bound_maxsat_solving_extended_version.md)**

:   为 Branch-and-Bound MaxSAT 求解器实现了基于 VeriPB 证明系统的认证，覆盖了 look-ahead 边界方法和多值决策图（MDD）编码两大核心技术，在 MaxCDCL 求解器上的实验表明证明日志的中位开销仅 19%，填补了 MaxSAT 求解范式认证的最后空白。

**[Certified But Fooled Breaking Certified Defences With Ghost Certificates](others/certified_but_fooled_breaking_certified_defences_with_ghost_certificates.md)**

:   提出 GhostCert，一种基于显著性区域的对抗攻击方法，能在保持扰动不可感知的同时误导分类器并伪造大半径的认证证书（ghost certificates），在 ImageNet 上对包括 DensePure 在内的 SOTA 认证防御取得显著优于 Shadow Attack 的攻击成功率和更大的伪造认证半径。

**[Clinician-In-The-Loop Smart Home System To Detect Urinary Tract Infection Flare-](others/clinician-in-the-loop_smart_home_system_to_detect_urinary_tract_infection_flare-.md)**

:   提出一种临床医师参与闭环的智能家居系统，利用环境传感器数据提取行为标记，结合新颖的共形校准区间（CCI）方法量化预测不确定性，实现对老年人尿路感染（UTI）发作的可靠检测与"不确定时弃权"的决策支持。

**[Controllable Financial Market Generation With Diffusion Guided Meta Agent](others/controllable_financial_market_generation_with_diffusion_guided_meta_agent.md)**

:   提出Diffusion Guided Meta Agent（DigMA）模型，将可控金融市场生成形式化为条件生成任务，用条件扩散模型捕捉市场状态动态（中间价收益率与订单到达率的时变分布参数），结合具有金融经济学先验的Meta Agent生成订单流，在可控性和生成保真度上均超越现有方法。

**[Cost-Free Neutrality For The River Method](others/cost-free_neutrality_for_the_river_method.md)**

:   针对River投票方法的并行宇宙打破平局（PUT）问题，证明其获胜者集合可在多项式时间内计算（相比Ranked Pairs的NP-完全性），提出Fused-Universe（FUN）算法，一次遍历同时模拟所有可能的打破平局方式，并为每个获胜者提供构造性证书。

**[Data Complexity Of Querying Description Logic Knowledge Bases Under Cost-Based S](others/data_complexity_of_querying_description_logic_knowledge_bases_under_cost-based_s.md)**

:   系统研究加权描述逻辑知识库在代价语义下的查询应答的数据复杂度，证明最优代价语义在$\Delta_2^p$内可解，并给出一个令人惊喜的正面结果：在DL-Lite$_{\text{bool}}^{\mathcal{H}}$本体和固定代价界限下，实例查询的确定回答和合取查询的可能回答可通过一阶重写实现最低数据复杂度（AC$^0$）。

**[Deadline-Aware Energy-Efficient Control Of Domestic Immersion Hot Water Heater](others/deadline-aware_energy-efficient_control_of_domestic_immersion_hot_water_heater.md)**

:   提出一种基于截止时间感知的家用热水器节能控制方法，通过 Gymnasium 仿真环境比较 bang-bang 基线、MCTS 规划器和 PPO 策略，证明 PPO 在相同物理条件下能节省高达 69% 的能量。

**[Decor Deep Embedding Clustering With Orientation Robustness](others/decor_deep_embedding_clustering_with_orientation_robustness.md)**

:   提出 DECOR 框架，通过旋转不变的等变卷积自编码器（RCAE）+ 非参数聚类（DeepDPM）+ 集成异常检测，实现晶圆图缺陷模式的方向鲁棒聚类。

**[Deeprwcap Neural-Guided Random-Walk Capacitance Solver For Ic Design](others/deeprwcap_neural-guided_random-walk_capacitance_solver_for_ic_design.md)**

:   提出 DeepRWCap，一种机器学习引导的随机游走电容求解器，通过两阶段神经网络架构预测转移核来加速IC设计中的多介质域电容提取，在10个工业测试案例上实现平均1.24%误差和23%加速。

**[Depth-Synergized Mamba Meets Memory Experts For All-Day Image Reflection Separat](others/depth-synergized_mamba_meets_memory_experts_for_all-day_image_reflection_separat.md)**

:   提出 DMDNet，通过深度感知扫描策略（DAScan）引导 Mamba 关注显著结构，结合深度协同状态空间模型（DS-SSM）抑制模糊特征传播，并引入记忆专家补偿模块（MECM）利用跨图像历史知识，实现全天候（白天+夜间）的图像反射分离。

**[Description Logics With Two Types Of Definite Descriptions Complexity Expressive](others/description_logics_with_two_types_of_definite_descriptions_complexity_expressive.md)**

:   引入描述逻辑 ALC 的两种定冠描述扩展——局部定冠描述 $\{ι C\}$ 和全局定冠描述 $ι C.D$，证明三个扩展逻辑的可满足性问题均为 ExpTime-complete，但全局定冠描述严格比局部更具表达力（$\mathcal{ALC}\iota_L < \mathcal{ALC}\iota_G = \mathcal{ALC}\iota$），并给出表列演算决策过程及实验评估。

**[Designing Incident Reporting Systems For Harms From General-Purpose Ai](others/designing_incident_reporting_systems_for_harms_from_general-purpose_ai.md)**

:   通过文献综述和九个安全关键行业（核能、航空、医疗等）的案例研究，提出了 AI 事件报告系统制度设计的七维框架，为美国通用 AI 事件报告的政策设计提供系统性指导。

**[Detonation Decoupled Torch Network-Aware Training On Interlinked Online Nodes](others/detonation_decoupled_torch_network-aware_training_on_interlinked_online_nodes.md)**

:   提出 FlexDeMo——一种将全分片数据并行（FSDP）与解耦动量优化相结合的混合分片训练策略，在节点内使用 FSDP 分片、节点间仅同步快速移动的动量分量，实现了接近全同步 AdamW 的损失收敛同时显著加速训练。

**[Dfdt Dynamic Fast Decision Tree For Iot Data Stream Mining On Edge Devices](others/dfdt_dynamic_fast_decision_tree_for_iot_data_stream_mining_on_edge_devices.md)**

:   提出 DFDT（Dynamic Fast Decision Tree），一种面向 IoT 边缘设备的内存受限数据流挖掘算法，通过活动感知预剪枝、动态 grace period、自适应 tie threshold 三重机制有机调控树的增长，实现精度-内存-运行时间的最优权衡。

**[Diffmm Efficient Method For Accurate Noisy And Sparse Trajectory Map Matching Vi](others/diffmm_efficient_method_for_accurate_noisy_and_sparse_trajectory_map_matching_vi.md)**

:   提出 DiffMM，首次将扩散模型引入地图匹配任务，通过路段感知轨迹编码器和一步 Shortcut 扩散过程，在稀疏轨迹和复杂路网上实现了精度和效率的双重提升，推理速度比次优方法快约 17 倍。

**[Ds-Atgo Dual-Stage Synergistic Learning Via Forward Adaptive Threshold And Backw](others/ds-atgo_dual-stage_synergistic_learning_via_forward_adaptive_threshold_and_backw.md)**

:   针对SNN训练中因膜电位分布偏移导致的脉冲发放不均衡和梯度消失问题，提出前向自适应阈值+后向阈值驱动梯度优化的双阶段协同学习算法DS-ATGO，在CIFAR10/100和ImageNet上以低时延实现SOTA性能。

**[Dw-Dgat Dynamically Weighted Dual Graph Attention Network For Neurodegenerative ](others/dw-dgat_dynamically_weighted_dual_graph_attention_network_for_neurodegenerative_.md)**

:   针对神经退行性疾病（PD/AD）早期诊断中的多指标数据融合、异质信息提取和类别不平衡三大挑战，提出动态加权双图注意力网络DW-DGAT，通过通用数据融合策略、微观-宏观双层图特征学习和动态类别权重生成机制，在PPMI和ADNI3数据集上大幅超越14种基线方法。

**[Enhancing Control Policy Smoothness By Aligning Actions With Predictions From Pr](others/enhancing_control_policy_smoothness_by_aligning_actions_with_predictions_from_pr.md)**

:   提出 **ASAP（Action Smoothing by Aligning Actions with Predictions from Preceding States）**，一种基于**转移诱导相似状态定义**的强化学习动作平滑方法，通过空间约束（对齐前一状态的预测动作）和时间约束（惩罚二阶动作差异）有效抑制高频动作振荡，在 Gymnasium 和 Isaac-Lab 环境中优于现有方法。

**[Expandable And Differentiable Dual Memories With Orthogonal Regularization For E](others/expandable_and_differentiable_dual_memories_with_orthogonal_regularization_for_e.md)**

:   提出 **EDD（Expandable and Differentiable Dual Memory）**，一种**无需存储旧样本**的持续学习方法，通过**可微分的共享记忆和任务特定记忆**将数据分解为可复用的子特征，结合**记忆扩展-剪枝**和**正交正则化**机制，在 CIFAR-10/100 和 Tiny-ImageNet 上超越 14 种 SOTA 方法，最终准确率分别达到 55.13%、37.24% 和 30.11%。

**[Expressive Temporal Specifications For Reward Monitoring](others/expressive_temporal_specifications_for_reward_monitoring.md)**

:   利用量化线性时序逻辑（LTLf[F]）自动合成**量化奖励监控器（QRM）**，为强化学习智能体在运行时生成密集的连续值奖励流，从根本上缓解布尔语义下长时任务的稀疏奖励问题。

**[Extreme Value Monte Carlo Tree Search For Classical Planning](others/extreme_value_monte_carlo_tree_search_for_classical_planning.md)**

:   利用 Peaks-Over-Threshold 极值理论（POT EVT）为经典规划中 MCTS 的 Full Bellman Backup 提供统计理论基础，提出 UCB1-Uniform bandit 算法，用均匀分布（Generalized Pareto 的特例）的 MLE 估计指导动作选择，在 Pyperplan 上以 $10^4$ 节点预算超越 GBFS 67.8 个实例、超越 Softmin-Type(h) 33.2 个实例。

**[Faster Certified Symmetry Breaking Using Orders With Auxiliary Variables](others/faster_certified_symmetry_breaking_using_orders_with_auxiliary_variables.md)**

:   通过引入辅助变量编码字典序来替代大整数编码，对 VeriPB 证明系统进行本质重设计，使 SAT 对称性破坏的证明生成和验证在理论和实践上均获得数量级加速。

**[Finding Diverse Solutions Parameterized By Cliquewidth](others/finding_diverse_solutions_parameterized_by_cliquewidth.md)**

:   将"寻找多样化解"的参数化框架从treewidth扩展到更强的cliquewidth图参数，证明任何基于cliquewidth分解的单调动态规划都可以以极小额外开销转换为求解多样化版本的算法，并提出了一族新的Venn多样性度量函数。

**[Formal Abductive Latent Explanations For Prototype-Based Networks](others/formal_abductive_latent_explanations_for_prototype-based_networks.md)**

:   本文针对原型网络（如ProtoPNet）的解释可能具有误导性的问题，提出了溯因潜在解释（ALE），在潜在空间中构造满足形式化保证的充分条件解释，无需调用外部求解器，算法可扩展到多种数据集上的标准分类和细粒度分类任务。

**[From Decision Trees To Boolean Logic A Fast And Unified Shap Algorithm](others/from_decision_trees_to_boolean_logic_a_fast_and_unified_shap_algorithm.md)**

:   本文提出Woodelf算法，通过将决策树集成模型转化为加权析取范式（WDNF）的伪布尔公式，在统一框架下实现了Background SHAP和Path-Dependent SHAP的线性时间计算，在大规模数据集上实现CPU 16-31倍、GPU 24-333倍的加速。

**[From Sequential To Recursive Enhancing Decision-Focused Learning With Bidirectio](others/from_sequential_to_recursive_enhancing_decision-focused_learning_with_bidirectio.md)**

:   本文首次提出递归决策聚焦学习（R-DFL）框架，通过在预测模块与优化模块之间引入双向反馈回路，突破了传统顺序式 DFL 的单向信息流限制，并设计了显式展开和隐式微分两种梯度传播方法，在报童问题和二部匹配问题上显著提升了最终决策质量。

**[Guided Perturbation Sensitivity Gps Detecting Adversarial Text Via Embedding Sta](others/guided_perturbation_sensitivity_gps_detecting_adversarial_text_via_embedding_sta.md)**

:   提出 Guided Perturbation Sensitivity (GPS) 框架，通过对重要词进行遮蔽并测量嵌入表示的稳定性变化来检测对抗文本样本，在3个数据集、3种攻击、2个模型上实现85%+检测准确率，且无需重训练即可跨数据集/攻击/模型泛化。

**[Hierarchical Semantic Alignment For Image Clustering](others/hierarchical_semantic_alignment_for_image_clustering.md)**

:   结合名词级（WordNet）和描述级（Flickr 图片描述）两种互补语义，通过最优传输对齐构建语义空间并自适应融合，实现 training-free 的图像聚类，在 ImageNet-1K 上准确率提升 4.2%。

**[Higher-Order Responsibility](others/higher-order_responsibility.md)**

:   本文研究顺序决策机制中的高阶责任问题，证明了两个核心定理：(1) $n$ 个智能体的机制必然是 $n$ 阶无间隙的（即总能找到某阶责任人）；(2) 判定机制是否为 $d$ 阶无间隙的问题是 $\Pi_{2d+1}$-完全的。

**[How Hard Is It To Explain Preferences Using Few Boolean Attributes](others/how_hard_is_it_to_explain_preferences_using_few_boolean_attributes.md)**

:   本文系统研究了用布尔属性模型（BAM）解释偏好数据的计算复杂性：证明了当属性数 $k \geq 3$ 时问题是NP完全的，$k \leq 2$ 时线性可解；进一步对投票人数 $n$、候选项数 $m$、属性数 $k$ 等参数给出了完整的参数化复杂性全景图，并分析了已知部分信息（cares/has）时问题难度的变化。

**[How Hard Is It To Rig A Tournament When Few Players Can Beat Or Be Beaten By The](others/how_hard_is_it_to_rig_a_tournament_when_few_players_can_beat_or_be_beaten_by_the.md)**

:   本文提出两个新的结构化参数——目标选手在锦标赛有向图中的入度 $k$ 和出度 $\ell$——用于分析锦标赛赛程操纵问题 (TFP)，证明 TFP 在以这两个参数为参数时均是 FPT 的，其中入度参数化的算法设计涉及复杂的结构分析和颜色编码技术。

**[How To Marginalize In Causal Structure Learning](others/how_to_marginalize_in_causal_structure_learning.md)**

:   本文利用可处理概率电路（Probabilistic Circuits）替代传统动态规划方法来执行贝叶斯结构学习中的边际化任务，通过一种新颖的两阶段训练策略（先学习完整父集分数再渐进式微调边际查询），消除了候选父节点集数量的人为限制，从而在 TRUST 框架上取得了更好的后验分布估计效果。

**[How Wide And How Deep Mitigating Over-Squashing Of Gnns Via Channel Capacity Con](others/how_wide_and_how_deep_mitigating_over-squashing_of_gnns_via_channel_capacity_con.md)**

:   本文从信息论视角出发，将谱图神经网络建模为通信信道，提出信道容量约束估计框架 C3E，将 GNN 隐藏维度与深度的选择形式化为一个非线性规划问题，在训练前即可估计最优架构参数，有效缓解信息过度压缩（over-squashing），在 9 个数据集上一致提升了表示学习效果。

**[Hypershap Shapley Values And Interactions For Explaining Hyperparameter Optimiza](others/hypershap_shapley_values_and_interactions_for_explaining_hyperparameter_optimiza.md)**

:   HyperSHAP 提出一套基于 Shapley 值和 Shapley 交互的博弈论框架来解释超参数优化（HPO），通过定义消融、灵敏度、可调性和优化器偏差四类解释博弈，提供比 fANOVA 更具可操作性的超参数重要性分析。

**[I2E Real-Time Image-To-Event Conversion For High-Performance Spiking Neural Netw](others/i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)**

:   I2E 提出一个超高效的图像到事件流转换框架，通过模拟微扫视眼动并用高度并行化的卷积实现比先前方法快 300 倍以上的转换速度，首次支持 SNN 训练的在线数据增强，在 I2E-ImageNet 上达到 60.50% 的事件分类 SOTA，并通过合成数据预训练 + 真实数据微调的 sim-to-real 范式在 CIFAR10-DVS 上创下 92.5% 的历史最佳。

**[Improved Differentially Private Algorithms For Rank Aggregation](others/improved_differentially_private_algorithms_for_rank_aggregation.md)**

:   针对差分隐私下的排名聚合问题，提出了改进的近似算法：首次研究footrule排名聚合问题并给出近最优算法（可推导出Kemeny问题的2-近似），同时通过结合二路边际查询和无偏估计技术改进了Kemeny排名聚合的PTAS加性误差（指数从3降至65/22）。

**[Incremental Maintenance Of Datalogmtl Materialisations](others/incremental_maintenance_of_datalogmtl_materialisations.md)**

:   提出 DRed$_{\text{MTL}}$ 算法，将经典 Delete/Rederive 增量维护技术扩展到 DatalogMTL（带度量时序逻辑的 Datalog），通过在周期化物化表示上设计新的 seminaïve 评估算子和周期识别算法，实现高效增量更新，性能可达重新物化的数量级提升。

**[Intermediate N-Gramming Deterministic And Fast N-Grams For Large N And Large Dat](others/intermediate_n-gramming_deterministic_and_fast_n-grams_for_large_n_and_large_dat.md)**

:   提出 Intergrams 多遍扫描算法，利用较短 n-gram 作为前缀递推过滤候选更长 n-gram，充分利用处理器缓存层次结构实现缓存友好的内存访问模式，在 TB 级数据集上比此前最快的 hash-gramming 方法加速 6-33 倍，同时几乎精确恢复所有 top-k n-gram。

**[Intrinsic Barriers And Practical Pathways For Human-Ai Alignment An Agreement-Ba](others/intrinsic_barriers_and_practical_pathways_for_human-ai_alignment_an_agreement-ba.md)**

:   本文将 AI 对齐形式化为 $\langle M,N,\varepsilon,\delta\rangle$-agreement 多目标优化问题，从通信复杂度角度证明了对齐的信息论下界（编码"所有人类价值观"本质上不可行），同时给出了无界/有界理性智能体的显式可达算法和紧致上界，揭示了在大状态空间下 reward hacking 全局不可避免的理论根基。

**[Judging By The Rules Compliance-Aligned Framework For Modern Slavery Statement M](others/judging_by_the_rules_compliance-aligned_framework_for_modern_slavery_statement_m.md)**

:   提出以"合规对齐法官"（CA-Judge）为核心的训练框架，利用规则级对齐反馈训练 3B 参数的 CALLM 模型，使其生成基于法定条款的可追溯合规判断理由，在现代奴役声明的句子级合规分类任务上超越 GPT-4o 和 DeepSeek-R1。

**[Leanrag Knowledge-Graph-Based Generation With Semantic Aggregation And Hierarchi](others/leanrag_knowledge-graph-based_generation_with_semantic_aggregation_and_hierarchi.md)**

:   提出 LeanRAG 框架，通过语义聚合算法在层次化知识图谱的摘要节点间自动构建显式关系打破"语义孤岛"，并基于最近公共祖先（LCA）的自底向上检索策略高效导航层次结构，在四个 QA 基准上取得 SOTA 同时减少 46% 的检索冗余。

**[Learning Compact Latent Space For Representing Neural Signed Distance Functions ](others/learning_compact_latent_space_for_representing_neural_signed_distance_functions_.md)**

:   提出一种双分支架构（泛化分支+过拟合分支）来学习多个神经SDF的紧凑潜空间，结合共享spatial feature grid和新颖的带宽采样策略，在保持紧凑latent code的同时恢复高保真几何细节，在Stanford Models、ShapeNet和D-FAUST上均达到SOTA。

**[Learning Fair Representations With Kolmogorov-Arnold Networks](others/learning_fair_representations_with_kolmogorov-arnold_networks.md)**

:   提出将Kolmogorov-Arnold网络（KAN）引入对抗去偏框架，利用KAN的样条函数架构提供理论上的Lipschitz连续性和平滑性保证，并设计自适应 $\lambda$ 更新机制动态平衡公平性与准确率，在UCI大学录取数据集上实现了公平性指标的显著提升。

**[Learning Network Dismantling Without Handcrafted Inputs](others/learning_network_dismantling_without_handcrafted_inputs.md)**

:   提出MIND（Message Iteration Network Dismantler），通过全新的All-to-One注意力机制和消息迭代轮廓（Message Iteration Profiles）消除GNN对手工特征的依赖，仅利用原始邻接信息就能在百万节点级真实网络上实现SOTA的网络拆解性能，同时具有最低的计算复杂度 $O(|V|+|E|)$。

**[Lilad Learning In-Context Lyapunov-Stable Adaptive Dynamics Models](others/lilad_learning_in-context_lyapunov-stable_adaptive_dynamics_models.md)**

:   提出 LILAD 框架，利用 GPT-2 的 in-context learning 能力同时学习动力学模型和 Lyapunov 函数，在保证全局指数稳定性的同时实现对非平稳参数化动力系统的自适应辨识，在多个基准系统上超越 ICL、MAML 等基线。

**[Local Guidance For Configuration-Based Multi-Agent Pathfinding](others/local_guidance_for_configuration-based_multi-agent_pathfinding.md)**

:   提出局部引导（Local Guidance）概念改进 LaCAM 的多智能体路径规划，通过在每个配置生成步为每个智能体构造局部时空路径来缓解拥塞，最高可将解的代价降低 50%，同时保持 1000 智能体下几秒内完成。

**[Lost In Time A Meta-Learning Framework For Time-Shift-Tolerant Physiological Sig](others/lost_in_time_a_meta-learning_framework_for_time-shift-tolerant_physiological_sig.md)**

:   提出 ShiftSyncNet，一个基于元学习双层优化的框架，通过 SyncNet 学习训练样本对之间的时间偏移量并利用傅里叶变换的相移性质自动校正标签对齐，在三个数据集上分别提升了 9.4%、6.0% 和 12.8% 的波形转换精度。

**[Measuring Model Performance In The Presence Of An Intervention](others/measuring_model_performance_in_the_presence_of_an_intervention.md)**

:   针对存在干预（intervention）时 AI 模型评估偏差的问题，提出 Nuisance Parameter Weighting (NPW) 方法，通过对 RCT 治疗组数据进行因果加权，实现无偏的 AUROC 估计，使样本效率提升 5 倍，显著改善了模型选择和假设检验的统计功效。

**[Mesha Efficient Path Planning With Motion Primitives](others/mesha_efficient_path_planning_with_motion_primitives.md)**

:   提出 MeshA* 算法，将 lattice-based 路径规划从"在运动基元层面搜索"转变为"在网格单元层面搜索并同时拟合基元序列"，通过定义"扩展网格单元"（extended cell）新搜索空间，在保证完备性和最优性的同时，实现相比标准 LBA* 1.5x-2x 的运行时加速。

**[Mf-Speech Achieving Fine-Grained And Compositional Control In Speech Generation ](others/mf-speech_achieving_fine-grained_and_compositional_control_in_speech_generation_.md)**

:   提出MF-Speech框架，通过多目标优化将语音信号解耦为高纯度的内容、音色和情绪三个独立因子表示，再利用动态融合和层级风格自适应归一化（HSAN）实现细粒度的组合式语音生成控制，在多因子组合语音生成任务上显著超越现有方法（WER=4.67%, SECS=0.5685）。

**[Model Counting For Dependency Quantified Boolean Formulas](others/model_counting_for_dependency_quantified_boolean_formulas.md)**

:   本文首次研究了依赖量化布尔公式（DQBF）的模型计数问题，证明了即使仅含两个存在量词变量的 #2-DQBF 就已是 #EXP-完全的，并基于 BDD 符号可达性技术实现了一个实用的 2-DQBF 模型计数器 sharp2DQR，在大依赖集上显著优于基于展开的基线方法。

**[On The Edge Of Core Non-Emptiness An Automated Reasoning Approach To Approval-Ba](others/on_the_edge_of_core_non-emptiness_an_automated_reasoning_approach_to_approval-ba.md)**

:   针对基于认可的多赢者投票中核稳定性（core stability）是否总存在这一重大开放问题，提出基于混合整数线性规划（MILP）的自动推理框架，证明了新的存在性结果，发现了核稳定性与其他公理（如 Lindahl 可定价性）之间此前未知的关系，并推翻了一个已有猜想。

**[On The Information Processing Of One-Dimensional Wasserstein Distances With Fini](others/on_the_information_processing_of_one-dimensional_wasserstein_distances_with_fini.md)**

:   本文通过Poisson过程框架，解析刻画了一维Wasserstein距离在有限样本下同时编码概率密度函数的逐点密度差异（rate difference）和支撑差异（support difference）的能力，并在神经脉冲数据和氨基酸接触频率数据上验证了其实际价值。

**[On The Variability Of Concept Activation Vectors](others/on_the_variability_of_concept_activation_vectors.md)**

:   对 TCAV 方法中概念激活向量（CAV）的变异性进行首次理论分析，证明 CAV 的方差以 $O(1/N)$ 速率衰减（$N$ 为随机样本数），而 TCAV 分数的方差因"边界点"保持 $O(1)$，需通过多次运行平均以 $O(1/s)$ 降低。

**[Online Linear Regression With Paid Stochastic Features](others/online_linear_regression_with_paid_stochastic_features.md)**

:   研究了在线线性回归中特征被噪声污染、学习者可以**付费降低噪声强度**的新问题设定，证明了已知噪声协方差时最优遗憾率为 $\widetilde{\mathcal{O}}(\sqrt{T})$、未知时为 $\widetilde{\mathcal{O}}(T^{2/3})$，并给出匹配的下界，所有界关于时间 $T$ 的依赖都是阶最优的。

**[Optimal Welfare In Noncooperative Network Formation Under Attack](others/optimal_welfare_in_noncooperative_network_formation_under_attack.md)**

:   在Goyal等人(WINE 2016)提出的非合作网络形成博弈模型中，证明了自私智能体创建的均衡网络在面对包括maximum disruption在内的广泛攻击者类别（超二次扰动攻击者SQD）时，仍能维持渐近最优的社会福利$n^2 - O(n)$，解决了一个长期开放问题。

**[Or-R1 Automating Modeling And Solving Of Operations Research Optimization Proble](others/or-r1_automating_modeling_and_solving_of_operations_research_optimization_proble.md)**

:   OR-R1提出了一个数据高效的两阶段训练框架（SFT + TGRPO），仅使用ORLM所需1/10的合成数据即达到67.7%的平均求解准确率，超越现有SOTA方法，并通过测试时强化学习将单次生成（Pass@1）与多次生成（Pass@8）的性能差距从13%缩小到7%。

**[Parameta Towards Learning Disentangled Paralinguistic Speaking Styles Representa](others/parameta_towards_learning_disentangled_paralinguistic_speaking_styles_representa.md)**

:   提出 ParaMETA，一种统一的副语言说话风格表示学习框架，通过 META 空间正则化和任务特定子空间投影实现情感、年龄、性别、语言等说话风格的解耦表示，同时支持下游的多任务分类和风格可控语音合成。

**[Parameterized Approximation Algorithms For Tsp On Non-Metric Graphs](others/parameterized_approximation_algorithms_for_tsp_on_non-metric_graphs.md)**

:   本文针对非度量图上的旅行商问题（TSP），提出了关于参数 $p$（违反三角不等式的顶点数）和 $q$（最小违反集大小）的改进FPT近似算法，将 $p$ 参数下的近似比从2.5改进到1.5，$q$ 参数下从11改进到3。

**[Piphen Physical Interaction Prediction With Hamiltonian Energy Networks](others/piphen_physical_interaction_prediction_with_hamiltonian_energy_networks.md)**

:   提出PIPHEN分布式物理认知-控制框架，通过物理交互预测网络（PIPN）进行"语义蒸馏"将高维感知数据压缩至原始数据量的5%以下，再由基于哈密顿能量守恒的HEN控制器生成协调动作，从而解决多机器人系统的"共享大脑困境"。

**[Predict And Resist Long-Term Accident Anticipation Under Sensor Noise](others/predict_and_resist_long-term_accident_anticipation_under_sensor_noise.md)**

:   提出统一框架，将基于扩散模型的双层去噪模块与时间感知的Actor-Critic强化学习模型结合，在传感器噪声条件下实现鲁棒的长期交通事故预测，在三个基准数据集上取得了准确率（AP）和平均事故前预警时间（mTTA）的最优性能。

**[Private Frequency Estimation Via Residue Number Systems](others/private_frequency_estimation_via_residue_number_systems.md)**

:   提出 ModularSubsetSelection (MSS)，一种基于剩余数系统（RNS）的本地差分隐私频率估计协议，在保持与 SubsetSelection 和 PGR 相当的估计精度的同时，显著降低通信开销（比 SS 减少达一半）、大幅加速服务器解码（比 PGR 快 11-448 倍）、并实现最低的数据重建攻击成功率。

**[Provably Data-Driven Projection Method For Quadratic Programming](others/provably_data-driven_projection_method_for_quadratic_programming.md)**

:   将数据驱动的投影矩阵学习从线性规划（LP）扩展到凸二次规划（QP），通过提出"展开主动集方法"在 Goldberg-Jerrum 框架下建模 QP 最优值的计算过程，从而建立了投影矩阵学习的伪维度上界和泛化保证。

**[Rcae Recursive Reconstruction Framework For Unsupervised Industrial Anomaly Dete](others/rcae_recursive_reconstruction_framework_for_unsupervised_industrial_anomaly_dete.md)**

:   提出递归卷积自编码器（RcAE），通过参数共享的多步迭代重建逐步抑制异常并保留正常细节，配合跨递归检测模块（CRD）利用多步重建动态实现鲁棒的异常定位，在仅需10%扩散模型参数的条件下达到可比的SOTA性能。

**[Reimagining Anomalies What If Anomalies Were Normal](others/reimagining_anomalies_what_if_anomalies_were_normal.md)**

:   提出首个面向无监督图像异常检测的反事实解释框架，通过训练生成器将异常样本修改为被检测器视为正常的多个解纠缠反事实，从语义层面回答“如果异常是正常的，它应该是什么样子？”，提供远超传统热力图的深层解释能力。

**[Rethinking Flow And Diffusion Bridge Models For Speech Enhancement](others/rethinking_flow_and_diffusion_bridge_models_for_speech_enhancement.md)**

:   本文提出了一个统一的理论框架，将语音增强中的 flow matching、score-based diffusion 和 Schrödinger bridge 模型统一为在配对数据之间构造不同高斯概率路径的过程，并揭示了这类生成模型每一步采样本质上等价于预测式语音增强，进而利用预测范式中的高性能骨干网络、改进损失函数和微调策略来增强桥模型性能。

**[Reward Redistribution Via Gaussian Process Likelihood Estimation](others/reward_redistribution_via_gaussian_process_likelihood_estimation.md)**

:   本文提出了基于高斯过程似然的奖励重分配框架 GP-LRR，通过核函数显式建模 state-action 对之间的相关性，利用 leave-one-out 策略最大化轨迹回报的边际似然来学习逐步奖励函数，理论证明传统 MSE 方法是其退化特例，并在 MuJoCo 基准上配合 SAC 实现了优越的样本效率和策略性能。

**[Semi-Supervised High Dynamic Range Image Reconstructing Via Bi-Level Uncertain A](others/semi-supervised_high_dynamic_range_image_reconstructing_via_bi-level_uncertain_a.md)**

:   提出半监督 HDR 重建框架，通过**不确定性估计分支**评估伪 HDR 标签质量，在 patch 和像素两个层面掩码不可靠区域，仅用 6.7% HDR 真值即可达到与全监督 SOTA 可比的性能。

**[Shortagesim Simulating Drug Shortages Under Information Asymmetry](others/shortagesim_simulating_drug_shortages_under_information_asymmetry.md)**

:   提出 ShortageSim，首个基于 **LLM 多智能体**的药品短缺模拟框架，建模 FDA 监管者、制造商和购买者在信息不对称下的战略决策，在历史短缺数据上实现对解决滞后时间 84% 的预测改善，为监管策略评估提供受控测试平台。

**[Spike Imaging Velocimetry Dense Motion Estimation Of Fluids Using Spike Cameras](others/spike_imaging_velocimetry_dense_motion_estimation_of_fluids_using_spike_cameras.md)**

:   提出 Spike Imaging Velocimetry（SIV），首次将**脉冲相机**（20000Hz 超高时间分辨率）应用于流体测速，设计细节保持层次变换（DPHT）、图编码器（GE）和多尺度速度精炼（MSVR）三个针对流体特性的模块，并构建了 PSSD 数据集，在稳态湍流、高速流和 HDR 场景上全面超越现有基线。

**[Stem Faculty Perspectives On Generative Ai In Higher Education](others/stem_faculty_perspectives_on_generative_ai_in_higher_education.md)**

:   通过对美国一所大型公立大学 29 名 STEM 教师的焦点小组研究，揭示教师如何在教学中整合 GenAI、观察到的学生学习益处与挑战，以及所需的制度支持——发现 GenAI 将教师劳动从内容创建转向专家审核，且可能掩盖学生能力不足。

**[Structural Approach To Guiding A Present-Biased Agent](others/structural_approach_to_guiding_a_present-biased_agent.md)**

:   在Kleinberg-Oren模型的委托-代理扩展中，系统性地研究了T-path-Editing问题的参数化复杂性，给出了以树宽和路径代价多样性为参数的FPT算法，并建立了紧的困难性结果，全面刻画了引导当前偏差agent完成关键任务的可处理-不可处理边界。

**[Structure-Aware Encodings Of Argumentation Properties For Clique-Width](others/structure-aware_encodings_of_argumentation_properties_for_clique-width.md)**

:   本文设计了从抽象论辩问题到(Q)SAT的有向分解引导(DDG)归约，线性保持团宽(clique-width)，为所有常见论辩语义（stable、admissible、complete、preferred、semi-stable、stage）在扩展存在性、论元接受性和计数问题上建立了以团宽为参数的可处理性上界，并证明了在ETH假设下这些归约的开销不可显著改进。

**[Svd-No Learning Pde Solution Operators With Svd Integral Kernels](others/svd-no_learning_pde_solution_operators_with_svd_integral_kernels.md)**

:   提出 SVD-NO，通过显式参数化积分核的奇异值分解（SVD）来构建神经算子，在保持高表达力的同时实现 $O(ndL)$ 的线性计算复杂度，在 5 个 PDE 基准上达到新 SOTA。

**[Symbolic Planning And Multi-Agent Path Finding In Extremely Dense Environments W](others/symbolic_planning_and_multi-agent_path_finding_in_extremely_dense_environments_w.md)**

:   提出 Block Rearrangement Problem (BRaP) 形式化定义，并设计五种基于配置空间搜索、PDDL 符号规划和 MAPF 的求解算法，其中 BR-LaCAM 在最大 80×80 的极端密集网格上达到 92% 成功率和毫秒级求解速度。

**[Synweather Weather Observation Data Synthesis Across Multiple Regions And Variab](others/synweather_weather_observation_data_synthesis_across_multiple_regions_and_variab.md)**

:   构建了首个支持统一多区域多变量的气象观测数据合成数据集SynWeather（覆盖4个区域×4种变量×6颗卫星），并提出基于Diffusion Transformer的通用概率生成模型SynWeatherDiff，通过文本提示引导在多个合成任务上超越专用模型和现有通用模型。

**[Tab-Pet Graph-Based Positional Encodings For Tabular Transformers](others/tab-pet_graph-based_positional_encodings_for_tabular_transformers.md)**

:   Tab-PET 提出从表格特征间关联关系中估计图结构，利用图拉普拉斯特征向量构造位置编码（PE）注入 Tabular Transformer，理论和实验均证明 PE 可降低嵌入的有效秩从而提升泛化，在 50 个数据集上为 TabTransformer / SAINT / FT-Transformer 带来一致改进，且 Spearman 关联图效果最佳。

**[Taylorpoda A Taylor Expansion-Based Method To Improve Post-Hoc Attributions For ](others/taylorpoda_a_taylor_expansion-based_method_to_improve_post-hoc_attributions_for_.md)**

:   在Taylor展开框架下提出精确性(precision)、联合性(federation)、零偏差(zero-discrepancy)三个公设规范特征归因，并引入自适应属性(adaptation)通过AUP目标优化交互效应的分配权重，成为唯一同时满足所有公设和属性的事后模型无关归因方法。

**[Tdsnns Competitive Topographic Deep Spiking Neural Networks For Visual Cortex Mo](others/tdsnns_competitive_topographic_deep_spiking_neural_networks_for_visual_cortex_mo.md)**

:   提出拓扑深度脉冲神经网络（TDSNNs），通过设计时空约束（STC）损失函数，在深层SNN中成功复现灵长类视觉皮层从V1到IT的层级拓扑组织，既保持了高任务性能（ImageNet top-1无精度下降），又显著超越现有拓扑ANN的脑相似性。

**[The Limitations And Power Of Np-Oracle-Based Functional Synthesis Techniques](others/the_limitations_and_power_of_np-oracle-based_functional_synthesis_techniques.md)**

:   系统性地从理论层面研究了依赖NP预言机的函数综合方法的能力与局限，证明了朴素的逐位学习方法在多输出场景下必然失败、基于Resolution插值的方法会产生指数级电路，同时证明了NP预言机是高效综合的必要条件，并给出了利用NP预言机在多项式时间内综合小规模Skolem函数的正面结果。

**[The Publication Choice Problem](others/the_publication_choice_problem.md)**

:   提出"出版选择问题"这一博弈论框架，建模研究者的出版策略与期刊影响力之间的双向互动关系，证明了纯策略均衡的存在性和唯一性，并分析了 Spotlight 论文标签对学术生态的影响。

**[Theoretical And Empirical Analysis Of Lehmer Codes To Search Permutation Spaces ](others/theoretical_and_empirical_analysis_of_lehmer_codes_to_search_permutation_spaces_.md)**

:   首次对 Lehmer 编码（逆序向量）在进化算法中搜索排列空间的效率进行严格的数学运行时分析，证明 Lehmer 编码的 EA 在多数基准函数上达到 $O(n^2 \log n)$ 或 $O(n^2)$ 的期望运行时间，与经典表示持平或更优，并在 LOP 和 QAP 实际问题上验证其实用性。

**[Towards Temporal Fusion Beyond The Field Of View For Camera-Based Semantic Scene](others/towards_temporal_fusion_beyond_the_field_of_view_for_camera-based_semantic_scene.md)**

:   提出 C3DFusion 模块，通过在 3D 空间中显式对齐历史帧和当前帧的点特征，首次系统解决相机基 SSC 中视野外（out-of-frame）区域的时序补全问题，在 SemanticKITTI 和 SSCBench-KITTI-360 上取得 SOTA。

**[Tractable Weighted First-Order Model Counting With Bounded Treewidth Binary Evid](others/tractable_weighted_first-order_model_counting_with_bounded_treewidth_binary_evid.md)**

:   提出一种在域大小上多项式时间的算法，计算带有有界树宽二元证据的 $\text{FO}^2$ 和 $\text{C}^2$ 片段的加权一阶模型计数（WFOMC），并解决了有界树宽有界度图上的稳定座位安排开放问题。

**[Variance Computation For Weighted Model Counting With Knowledge Compilation Appr](others/variance_computation_for_weighted_model_counting_with_knowledge_compilation_appr.md)**

:   本文将加权模型计数 (WMC) 的权重视为具有方差的随机变量，提出在 structured d-DNNF 表示上多项式时间计算 WMC 方差的算法，同时证明了在 structured DNNF、d-DNNF 和 FBDD 上该问题不可解（除非 P=NP），并将其应用于贝叶斯网络推理中参数不确定性的量化。

**[Verification-Guided Context Optimization For Tool Calling Via Hierarchical Llms-](others/verification-guided_context_optimization_for_tool_calling_via_hierarchical_llms-.md)**

:   提出VGCO框架，利用LLM作为分层编辑器，通过验证引导的方式迭代优化工具文档和知识库上下文，显著提升大规模工具调用场景下的检索召回、工具选择和参数填充准确率。

**[Whispering Agents An Event-Driven Covert Communication Protocol For The Internet](others/whispering_agents_an_event-driven_covert_communication_protocol_for_the_internet.md)**

:   首次形式化定义了智能体互联网（IoA）中的"隐蔽事件通道"（Covert Event Channel），并设计了 ΠCCAP 协议，通过在智能体对话的存储、时序和行为维度中嵌入秘密数据，实现了高容量、高鲁棒性且对 LLM 审查员不可感知的隐蔽通信。

**[Why Isnt Relational Learning Taking Over The World](others/why_isnt_relational_learning_taking_over_the_world.md)**

:   本文是一篇 position paper，深入分析了关系学习（relational learning）未能在 AI 领域占据主导地位的原因——包括数据集不够真实、评估方法存在根本缺陷、缺少负例、聚合操作的理论困境等——并指出了让关系学习发挥应有潜力所需的关键改进方向。

</div>