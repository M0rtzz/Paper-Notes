---
title: >-
  NeurIPS2025 医学图像方向92篇论文解读
description: >-
  92篇NeurIPS2025的医学图像方向论文解读，涵盖医学影像、多模态、语义分割、时序预测、对齐/RLHF、自监督学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "NeurIPS2025"
  - "医学图像"
  - "论文解读"
  - "论文笔记"
  - "医学影像"
  - "多模态"
  - "语义分割"
  - "时序预测"
  - "对齐/RLHF"
  - "自监督学习"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**🧠 NeurIPS2025** · **92** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (34)](../../ICML2026/medical_imaging/index.md) · [💬 ACL2026 (35)](../../ACL2026/medical_imaging/index.md) · [📷 CVPR2026 (103)](../../CVPR2026/medical_imaging/index.md) · [🔬 ICLR2026 (48)](../../ICLR2026/medical_imaging/index.md) · [🤖 AAAI2026 (88)](../../AAAI2026/medical_imaging/index.md) · [📹 ICCV2025 (31)](../../ICCV2025/medical_imaging/index.md)

🔥 **高频主题：** 医学影像 ×40 · 多模态 ×8 · 语义分割 ×8 · 时序预测 ×5 · 对齐/RLHF ×4

**[3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks](3drad_a_comprehensive_3d_radiology_medvqa_dataset_with_multi.md)**

:   提出 3D-RAD——首个大规模3D医学VQA基准，包含170K条CT影像问答数据，覆盖六类临床任务（含创新性的多时相诊断任务），并配套136K训练集，揭示了现有VLM在3D时序推理上的严重不足。

**[A Novel Approach to Classification of ECG Arrhythmia Types with Latent ODEs](a_novel_approach_to_classification_of_ecg_arrhythmia_types_with_latent_odes.md)**

:   将路径最小化 Latent ODE 的编码器与梯度提升决策树（GBDT）组合为两阶段 ECG 心律失常分类流水线，在 MIT-BIH 数据集上的 macro AUC-ROC 从 360Hz 的 0.984 仅降至 45Hz 的 0.976，展示了对采样频率变化的强鲁棒性。

**[A Unified Solution to Video Fusion: From Multi-Frame Learning to Benchmarking](a_unified_solution_to_video_fusion_from_multi-frame_learning_to_benchmarking.md)**

:   提出首个统一视频融合框架 UniVF（基于多帧学习 + 光流特征 warping + 时序一致性损失），并构建首个覆盖四大融合任务（多曝光、多焦点、红外-可见光、医学）的视频融合基准 VF-Bench，在全部子任务上取得 SOTA。

**[A Variational Manifold Embedding Framework for Nonlinear Dimensionality Reduction](a_variational_manifold_embedding_framework_for_nonlinear_dimensionality_reductio.md)**

:   提出一种变分流形嵌入框架，将降维问题形式化为最优嵌入映射的优化问题（最小化先验分布与数据分布pullback之间的KL散度），在理论上统一了PCA与非线性降维方法，并利用变分法（Euler-Lagrange方程）和Noether定理为最优嵌入提供了可解释性约束。

**[Active Target Discovery under Uninformative Prior: The Power of Permanent and Transient Memory](active_target_discovery_under_uninformative_prior_the_power_of_permanent_and_tra.md)**

:   提出 EM-PTDM 框架，受神经科学双记忆系统启发，利用预训练扩散模型作为"永久记忆"并结合基于 Doob's h-transform 的轻量"瞬时记忆"模块，在**无领域先验数据**的条件下实现高效的主动目标发现，理论保证先验单调改进。

**[Brain-tuning Improves Generalizability and Efficiency of Brain Alignment in Speech Models](brain-tuning_improves_generalizability_and_efficiency_of_brain_alignment_in_spee.md)**

:   提出 Multi-brain-tuning 方法，通过联合多个被试的 fMRI 数据微调预训练语音模型，将脑对齐所需数据量降低 5 倍，同时脑对齐度提升最高 50%，并可泛化到全新被试和数据集。

**[Brain Harmony: A Multimodal Foundation Model Unifying Morphology and Function into 1D Tokens](brain_harmony_a_multimodal_foundation_model_unifying_morphology_and_function_int.md)**

:   首个统一脑结构形态（T1 sMRI）与功能动态（fMRI）的多模态脑基础模型，通过几何谐波预对齐和时序自适应 Patch Embedding（TAPE）将高维神经影像压缩为紧凑的 1D token 表示，在神经发育/退行性疾病诊断和认知预测任务上全面超越先前方法。

**[BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals](brainomni_a_brain_foundation_model_for_unified_eeg_and_meg_signals.md)**

:   提出 BrainOmni——首个统一 EEG 和 MEG 的脑信号基础模型，通过 BrainTokenizer（含物理传感器编码器）将异构脑电/脑磁信号离散化为统一 token，再用 Criss-Cross Transformer 进行自监督掩码预测预训练，在阿尔茨海默病检测上提升 11.7 个百分点，并实现对完全未见设备的零样本重建泛化。

**[Bridging Graph and State-Space Modeling for Intensive Care Unit Length of Stay Prediction](bridging_graph_and_state-space_modeling_for_intensive_care_unit_length_of_stay_p.md)**

:   提出 S2G-Net，将 Mamba 状态空间模型的时序编码与多视图图神经网络（GraphGPS）进行双路融合，用于 ICU 住院时长（LOS）预测，在 MIMIC-IV 数据集上全面超越序列模型、图模型和混合基线。

**[Care-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment](care-pd_a_multi-site_anonymized_clinical_dataset_for_parkinsons_disease_gait_ass.md)**

:   发布 Care-PD——目前最大的面向帕金森病步态分析的多站点匿名 3D 网格数据集（9 个队列、8 个临床中心、362 名受试者、8477 段步行），并在 UPDRS 步态评分和运动预训练任务上提供系统性 benchmark，证明在 Care-PD 上微调可将 MPJPE 从 60.8mm 降至 7.5mm，F1 提升 17 个百分点。

**[Convolutional Monge Mapping between EEG Datasets to Support Independent Component Labeling](convolutional_monge_mapping_between_eeg_datasets_to_support_independent_componen.md)**

:   本文扩展 CMMN（Convolutional Monge Mapping Normalization）方法，提出通道平均 PSD + $\ell_1$ 归一化质心和 subject-to-subject 匹配两种策略，生成单一时域滤波器实现不同通道数的 EEG 数据集间域适应，在独立成分（IC）脑/非脑分类中 F1 从 0.77 提升至 0.84，超越 ICLabel（0.88→0.91）。

**[CureAgent: A Training-Free Executor-Analyst Framework for Clinical Reasoning](cureagent_a_training-free_executor-analyst_framework_for_clinical_reasoning.md)**

:   CureAgent 提出 Executor-Analyst 协作框架，将精确工具调用（TxAgent/Llama-8B 做 Executor）与高层临床推理（Gemini 2.5 做 Analyst）解耦，配合分层集成（Stratified Ensemble）的 Late Fusion 拓扑保留证据多样性，在 CURE-Bench 上达到 83.8% 准确率（无需端到端微调），揭示了上下文-性能悖论和动作空间维度灾难两个关键 scaling 发现。

**[CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays](cxreasonbench_a_benchmark_for_evaluating_structured_diagnostic_reasoning_in_ches.md)**

:   提出 CheXStruct + CXReasonBench，一个基于胸部X光的结构化诊断推理评估框架，通过多路径、多阶段评估揭示现有 LVLM 在中间推理步骤上的严重不足。

**[DCA: Graph-Guided Deep Embedding Clustering for Brain Atlases](dca_graph-guided_deep_embedding_clustering_for_brain_atlases.md)**

:   DCA（Deep Cluster Atlas）提出图引导深度嵌入聚类框架，结合预训练 Swin-UNETR 的体素级时空嵌入和 KNN 图空间正则化，通过 KL 散度对齐软分配与图谱聚类辅助标签，生成功能一致且空间连续的个性化脑图谱，在 HCP 数据集上同态性提升 98.8%、轮廓系数提升 29%，并在自闭症诊断、认知解码等下游任务中超越现有图谱。

**[Demo: Generative AI helps Radiotherapy Planning with User Preference](demo_generative_ai_helps_radiotherapy_planning_with_user_preference.md)**

:   提出 Flexible Dose Proposer (FDP)，通过两阶段训练框架（VQ-VAE 预训练 + 多条件编码）实现基于滑块的用户偏好交互式 3D 剂量分布预测，并集成到 Eclipse 临床治疗计划系统中，在头颈部癌症放疗场景中超越 Varian RapidPlan。

**[Demo: Guide-RAG: Evidence-Driven Corpus Curation for Retrieval-Augmented Generation in Long COVID](demo_guide-rag_evidence-driven_corpus_curation_for_retrieval-augmented_generatio.md)**

:   系统评估了六种 RAG 语料库配置用于长新冠（Long COVID）临床问答，发现将临床指南与高质量系统综述结合的 GS-4 配置在 faithfulness、relevance 和 comprehensiveness 三维度上一致优于单指南和大规模文献库方案，并提出 Guide-RAG 框架和 LongCOVID-CQ 评估数据集。

**[DermaCon-IN: A Multi-concept Annotated Dermatological Image Dataset of Indian Skin Disorders](dermacon-in_a_multi-concept_annotated_dermatological_image_dataset_of_indian_ski.md)**

:   构建了 DermaCon-IN——首个以印度肤色为主的密集标注皮肤病图像数据集（5,450 张 / 3,002 患者 / 245 种诊断），提供三级层次诊断标签、47 个病灶描述符和 49 个解剖位置标注，并用 CNN/ViT/概念瓶颈模型进行基准评测。

**[DIsoN: Decentralized Isolation Networks for Out-of-Distribution Detection in Medical Imaging](dison_decentralized_isolation_networks_for_out-of-distribution_detection_in_medi.md)**

:   提出 Decentralized Isolation Networks (DIsoN)，通过训练二分类器将测试样本从训练数据中"隔离"来检测 OOD，并通过去中心化参数交换实现在不共享数据的情况下利用训练数据信息，在 4 个医学影像数据集 12 个 OOD 检测任务上取得 SOTA。

**[Doctor Approved: Generating Medically Accurate Skin Disease Images through AI-Expert Feedback](doctor_approved_generating_medically_accurate_skin_disease_images_through_ai-exp.md)**

:   提出 MAGIC 框架，通过将皮肤科专家定义的临床检查清单转化为 MLLM（如 GPT-4o）可执行的评估反馈，利用 DPO 或奖励模型微调扩散模型，生成临床准确的皮肤病图像用于数据增强，在 20 类皮肤病分类任务上提升 +9.02%，少样本场景提升 +13.89%。

**[Domain-Adaptive Transformer for Data-Efficient Glioma Segmentation in Sub-Saharan MRI](domain-adaptive_transformer_for_data-efficient_glioma_segmentation_in_sub-sahara.md)**

:   提出 SegFormer3D+，一种面向撒哈拉以南非洲异质 MRI 数据的域自适应 Transformer 架构，通过直方图匹配、影像组学分层采样、频率感知双路径编码器和双注意力机制，在仅 60 例标注数据微调下实现胶质瘤分割 mean Dice 0.81，超越 nnU-Net +2.5%。

**[Dual Mixture-of-Experts Framework for Discrete-Time Survival Analysis](dual_mixture-of-experts_framework_for_discrete-time_survival_analysis.md)**

:   提出双混合专家（Dual MoE）框架用于离散时间生存分析，结合特征编码器 MoE（建模患者亚组异质性）与风险网络 MoE（捕获时间动态），在 METABRIC 和 GBSG 乳腺癌数据集上提升 time-dependent C-index 最高 0.04。

**[DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs](dyg-mamba_continuous_state_space_modeling_on_dynamic_graphs.md)**

:   DyG-Mamba 将连续状态空间模型（SSM）引入动态图学习，设计时间跨度感知的连续 SSM——用 Ebbinghaus 遗忘曲线启发的指数衰减函数建模不规则时间间隔，配合谱范数约束的输入依赖参数实现 Lipschitz 鲁棒性，在 12 个动态图基准上平均排名 2.42（vs DyGFormer 2.92），且保持 $O(bdL)$ 线性复杂度。

**[Dynamic Causal Discovery in Alzheimer's Disease through Latent Pseudotime Modelling](dynamic_causal_discovery_in_alzheimers_disease_through_latent_pseudotime_modelli.md)**

:   将 BN-LTE（贝叶斯网络+潜在时间嵌入）应用于 ADNI 真实 AD 数据，推断随疾病伪时间演变的动态因果图，伪时间预测诊断 AUC 0.82 远超年龄 0.59，并揭示了新型生物标志物 NfL/GFAP 与传统 AD 标志物之间的动态因果关系。

**[EEGReXferNet: A Lightweight Gen-AI Framework for EEG Subspace Reconstruction via Cross-Subject Transfer Learning and Channel-Aware Embedding](eegrexfernet_a_lightweight_gen-ai_framework_for_eeg_subspace_reconstruction_via_.md)**

:   提出 EEGReXferNet，一种轻量级生成式 AI 框架，通过邻域通道感知输入选择、频带特定子窗口卷积编解码、动态滑窗隐空间和参考统计量缩放，在跨被试迁移学习设置下实现 EEG 子空间重建，参数减少约 45%、推理延迟 <1ms，同时保持 PSD 相关性 $\geq 0.95$ 和谱图 RV 系数 $\geq 0.85$。

**[EndoBench: A Comprehensive Evaluation of Multi-Modal Large Language Models for Endoscopy Analysis](endobench_a_comprehensive_evaluation_of_multi-modal_large_language_models_for_en.md)**

:   提出 EndoBench，首个覆盖 4 种内窥镜场景、12 项临床任务、5 级视觉提示粒度的综合 MLLM 评估基准，包含 6,832 个经临床验证的 VQA 对，对 23 个 MLLM 的评估显示商用模型整体领先但仍落后人类专家。

**[EWC-Guided Diffusion Replay for Exemplar-Free Continual Learning in Medical Imaging](ewc-guided_diffusion_replay_for_exemplar-free_continual_learning_in_medical_imag.md)**

:   提出将类条件 DDPM 扩散重放与弹性权重巩固（EWC）相结合的无样本持续学习框架，在 MedMNIST v2（8 个 2D/3D 任务）和 CheXpert 上实现了 AUROC 0.851，相比 DER++ 遗忘率降低超 30%，接近联合训练上界（0.869），同时完全无需存储患者原始数据。

**[Exploring and Leveraging Class Vectors for Classifier Editing](exploring_and_leveraging_class_vectors_for_classifier_editing.md)**

:   提出 Class Vector（类向量），通过计算预训练与微调模型在潜空间中类别质心的差异来捕获类别级适应，利用线性和独立性两个性质，通过简单向量算术实现分类器编辑（遗忘、环境适应、对抗防御），无需重训练即可完成潜空间注入，或用 <1.5K 参数在 1.5 秒内完成权重空间映射。

**[FairGRPO: Fair Reinforcement Learning for Equitable Clinical Reasoning](fairgrpo_fair_reinforcement_learning_for_equitable_clinical_reasoning.md)**

:   提出 FairGRPO，一种层级式公平强化学习算法，通过自适应重要性加权（基于群体表示量和任务难度）解决临床 AI 中的人群表现差异问题，在 7 个临床数据集（280K样本，5种模态）上将预测平价降低 27.2%、F1 提升 12.49%，并发布首个公平性优化的临床 VLLM——FairMedGemma-4B。

**[Faithful Summarization of Consumer Health Queries: A Cross-Lingual Framework with LLMs](faithful_summarization_of_consumer_health_queries_a_cross-lingual_framework_with.md)**

:   提出结合 TextRank 抽取式句子选择和医学命名实体识别 (NER) 来引导 LLM 生成忠实医学摘要的框架，在英文 MeQSum 和孟加拉语 BanglaCHQ-Summ 数据集上通过微调 LLaMA-2-7B 实现质量和忠实性的一致提升，SummaC 达 0.57，人工评估 82% 摘要保留关键医学信息。

**[FAPEX: Fractional Amplitude-Phase Expressor for Robust Cross-Subject Seizure Prediction](fapex_fractional_amplitude-phase_expressor_for_robust_cross-subject_seizure_pred.md)**

:   提出 FAPEX 框架，通过可学习的分数阶神经帧算子 (FrNFO) 实现自适应时频分解，结合幅度-相位交叉编码和空间相关性聚合，在 12 个跨物种、跨模态的癫痫预测基准上全面超越 33 个基线方法。

**[Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling](few-shot_learning_from_gigapixel_images_via_hierarchical_vision-language_alignme.md)**

:   提出 HiVE-MIL，一个层级视觉-语言 MIL 框架，通过构建统一异构图建模跨尺度层级关系（5× 和 20×）和同尺度多模态对齐，配合文本引导的动态过滤机制和层级对比损失，在 TCGA 肺/乳腺/肾癌三个数据集的 16-shot 设置下全面超越已有方法，Macro F1 最高提升 4.1%。

**[FireGNN: Neuro-Symbolic Graph Neural Networks with Trainable Fuzzy Rules for Interpretable Medical Image Classification](firegnn_neuro-symbolic_graph_neural_networks_with_trainable_fuzzy_rules_for_inte.md)**

:   提出 FireGNN，首次将可训练模糊规则嵌入 GNN 前向传播中，利用节点度、聚类系数和标签一致性三个拓扑描述子实现内生可解释的医学图像分类，在 5 个 MedMNIST 数据集和 MorphoMNIST 上取得优于标准 GCN/GAT/GIN 及辅助任务方法的性能。

**[FOXES: A Framework For Operational X-ray Emission Synthesis](foxes_a_framework_for_operational_x-ray_emission_synthesis.md)**

:   提出 FOXES，一个基于 Vision Transformer 的框架，将太阳多通道 EUV 观测图像翻译为软 X 射线（SXR）通量，整体 Pearson 相关达到 0.982，为远端太阳耀斑检测和更完整的耀斑目录构建奠定基础。

**[From Black Box to Biomarker: Sparse Autoencoders for Interpreting Speech Models of Parkinson's Disease](from_black_box_to_biomarker_sparse_autoencoders_for_interpreting_speech_models_o.md)**

:   将大语言模型可解释性研究中的稀疏自编码器（SAE）技术适配到语音帕金森病检测系统中，提出 Mask-based SAE 解决小数据集限制，发现模型预测主要基于低能量区域的频谱通量和频谱平坦度，并进一步揭示这些特征与 MRI 壳核体积显著相关，建立了从模型内部表征到临床生物标志物的桥梁。

**[Generalizable, Real-Time Neural Decoding with Hybrid State-Space Models](generalizable_real-time_neural_decoding_with_hybrid_state-space_models.md)**

:   POSSM 提出了一种混合 SSM-注意力架构，结合 spike 级别 tokenization 和循环状态空间模型骨干，实现了可泛化的实时神经解码，在保持与 Transformer 可比的精度的同时，推理速度提升最高 9 倍。

**[Generating Multi-Table Time Series EHR from Latent Space with Minimal Preprocessing](generating_multi-table_time_series_ehr_from_latent_space_with_minimal_preprocess.md)**

:   提出 RawMed——首个以最小有损预处理合成多表时序 EHR 原始数据的框架：将事件文本化 → Residual Quantization 压缩至离散潜空间 → 自回归 Transformer 建模时序动态，在保真度、临床效用和隐私保护上全面超越现有基线。

**[GeoDynamics: A Geometric State-Space Neural Network for Understanding Brain Dynamics on Riemannian Manifolds](geodynamics_a_geometric_state-space_neural_network_for_understanding_brain_dynam.md)**

:   提出GeoDynamics，将经典状态空间模型(SSM)从欧几里得空间推广到对称正定(SPD)流形，通过加权Frechet均值聚合和正交群平移实现流形上的状态演化，在脑连接组（AD/PD/ASD早期诊断）和人体动作识别上均取得SOTA。

**[GFlowNets for Learning Better Drug-Drug Interaction Representations](gflownets_for_learning_better_drug-drug_interaction_representations.md)**

:   针对药物-药物相互作用（DDI）预测中严重的类别不平衡问题，本文提出将 GFlowNet 与变分图自编码器（VGAE）结合，通过奖励引导的生成采样为稀有交互类型生成合成样本，从而增强模型在罕见但临床关键的交互类型上的预测能力。

**[H-DDx: A Hierarchical Evaluation Framework for Differential Diagnosis](h-ddx_a_hierarchical_evaluation_framework_for_differential_diagnosis.md)**

:   H-DDx 提出基于 ICD-10 分类层级的鉴别诊断评估框架——将预测和真实诊断扩展到祖先节点后计算层级 F1（HDF1），奖励"临床相关的近似正确"而非仅精确匹配，评估 22 个 LLM 后发现领域特化模型（MediPhi）在 HDF1 上从第 20 名升至第 2 名（Top-5 指标完全遮蔽其优势）。

**[ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression](imagenet-trained_cnns_are_not_biased_towards_texture_revisiting_feature_reliance.md)**

:   通过系统化的特征抑制框架（而非冲突选择实验）重新评估 CNN 的特征依赖性，发现 CNN 并非天然偏向纹理，而是主要依赖局部形状特征；且不同领域（CV/MI/RS）的特征依赖模式显著不同。

**[Interpretable Next-token Prediction via the Generalized Induction Head](interpretable_next-token_prediction_via_the_generalized_induction_head.md)**

:   提出 Induction-Gram (GIM)，一种结合精确n-gram匹配与模糊匹配的可解释语言模型，通过构建"广义归纳头"在输入上下文中检索相似序列进行下一token预测，比可解释基线提升最高25%p准确率，并在fMRI脑响应预测中提升20%。

**[Large Language Models as Medical Codes Selectors: A Benchmark Using the International Classification of Primary Care](large_language_models_as_medical_codes_selectors_a_benchmark_using_the_internati.md)**

:   构建了一个 extract-retrieve-select 框架的医学编码基准，在 33 个 LLM 上评估 ICPC-2 编码选择能力，发现 28 个模型 F1>0.8，证明 LLM 无需微调即可有效自动化初级保健编码。

**[LoMix: Learnable Weighted Multi-Scale Logits Mixing for Medical Image Segmentation](lomix_learnable_weighted_multi-scale_logits_mixing_for_medical_image_segmentatio.md)**

:   LoMix 提出通过组合突变模块（CMM）生成多尺度 logits 的"突变体"——4 种融合算子（加法/乘法/拼接/注意力加权）× 所有子集组合——配合 NAS 风格的 Softplus 可学习权重自动平衡各 logits 的贡献，在 Synapse 8 器官分割上 DICE 从 80.9% 提升到 85.1%（+4.2%），5% 训练数据下提升 +9.23%。

**[Magical: Medical Lay Language Generation via Semantic Invariance and Layperson-tailored Adaptation](magical_medical_lay_language_generation_via_semantic_invariance_and_layperson-ta.md)**

:   提出 Magical，一种面向医学通俗语言生成（MLLG）的非对称 LoRA 架构，通过共享矩阵 A 上的语义不变性约束和多个独立矩阵 B 实现语义保真与多样化通俗风格生成，在减少 31.66% 可训练参数的同时超越所有 LoRA 变体。

**[Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation](mamba_goes_home_hierarchical_soft_mixture-of-experts_for_3d_medical_image_segmen.md)**

:   提出Mamba-HoME架构，将层次化Soft MoE（HoME）与Mamba SSM结合，通过两级token路由机制实现局部-全局特征建模，在CT/MRI/US三种模态的3D医学图像分割任务上超越现有SOTA方法，同时保持线性计算复杂度。

**[MATCH: Multi-faceted Adaptive Topo-Consistency for Semi-Supervised Histopathology Segmentation](match_multi-faceted_adaptive_topo-consistency_for_semi-supervised_histopathology.md)**

:   提出MATCH框架，通过将拓扑推理与半监督学习的"扰动鲁棒性"原则紧密耦合，利用跨随机扰动和时间训练快照的双层拓扑一致性，自适应识别可靠拓扑结构而无需人工阈值，显著降低了组织病理学图像分割中的拓扑错误。

**[MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks](medagentboard_benchmarking_multi-agent_collaboration_with_conventional_methods_f.md)**

:   提出 MedAgentBoard，一个系统评估多智能体协作、单 LLM 和传统方法在多样化医学任务上表现的综合基准，揭示多智能体协作并不总是优于强单模型或专用传统方法。

**[Meta-Learning an In-Context Transformer Model of Human Higher Visual Cortex](meta-learning_an_in-context_transformer_model_of_human_higher_visual_cortex.md)**

:   提出BraInCoRL（Brain In-Context Representation Learning），一种基于Transformer的元学习框架，通过上下文学习（in-context learning）从少量刺激-响应样本直接预测新被试的体素级神经响应，无需微调即可适应新被试和新刺激，仅用100张图片就接近在9000张图片上完整训练的参考模型的性能。

**[Mind the (Data) Gap: Evaluating Vision Systems in Small Data Applications](mind_the_data_gap_evaluating_vision_systems_in_small_data_applications.md)**

:   在 NeWT 生态分类基准上系统比较了 MLLMs（如 Gemini、Qwen2.5-VL）和视觉编码器+SVM 在"小数据区间"（10~1000 标注样本）的表现，发现 MLLMs 在 10-30 个样本后即触顶，而视觉方法持续近对数增长，呼吁社区重视小数据评估。

**[MIRA: Medical Time Series Foundation Model for Real-World Health Data](mira_medical_time_series_foundation_model_for_real-world_health_data.md)**

:   提出 MIRA，一个专为医学不规则时间序列设计的基础模型，通过连续时间旋转位置编码、频率特定 MoE 和 Neural ODE 外推模块，在 4540 亿个观测点上预训练，零样本预测性能在 OOD 和 ID 场景中分别平均降低 8% 和 6% 的误差。

**[Modeling X-ray Photon Pile-up with a Normalizing Flow](modeling_x-ray_photon_pile-up_with_a_normalizing_flow.md)**

:   提出基于Normalizing Flow的仿真推断(SBI)框架，通过CNN提取空间分辨的X射线光谱特征并输入神经样条流，实现在存在光子堆叠效应(pile-up)情况下对天体物理源参数的精确后验估计，显著优于传统PSF核心剪除方法。

**[MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)**

:   提出 MoRE-Brain，一种神经科学启发的 fMRI 视觉解码框架，采用层级混合专家（MoE）架构模拟大脑视觉通路的专门化处理，配合动态时间-空间双路由机制引导扩散模型生成图像，在保持高保真重建的同时实现了高效跨被试泛化和前所未有的机制可解释性。

**[Multimodal Bayesian Network for Robust Assessment of Casualties in Autonomous Triage](multimodal_bayesian_network_for_robust_assessment_of_casualties_in_autonomous_tr.md)**

:   提出基于专家知识驱动的贝叶斯网络决策支持框架，融合多个计算机视觉模型的输出来评估伤亡人员状况，无需训练数据且支持不完整信息推断，在DARPA Triage Challenge中将分诊准确率从14%提升至53%，诊断覆盖率从31%提升至95%。

**[Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment](multimodal_disease_progression_modeling_via_spatiotemporal_disentanglement_and_m.md)**

:   提出 DiPro 框架，通过区域感知的时空解耦（分离静态解剖与动态病理特征）和多时间尺度对齐（局部-全局融合 CXR 与 EHR），解决了纵向胸部X光序列的冗余问题和跨模态时间错位挑战，在疾病进展识别和 ICU 预测任务上达到 SOTA。

**[NeurIPT: Foundation Model for Neural Interfaces](neuript_foundation_model_for_neural_interfaces.md)**

:   NeurIPT是一个面向多样化脑机接口(BCI)应用的EEG基础模型，通过振幅感知掩码预训练(AAMP)、渐进式专家混合(PMoE)架构、3D电极空间编码和脑叶内/跨脑叶池化(IILP)四大创新设计，在八个下游BCI任务上实现了SOTA性能。

**[Online Feedback Efficient Active Target Discovery in Partially Observable Environments](online_feedback_efficient_active_target_discovery_in_partially_observable_enviro.md)**

:   提出 DiffATD，利用扩散模型的逆向过程构建 belief 分布来平衡探索与利用，在部分可观测环境中无需任何监督训练即可高效发现目标区域，适用于医学影像、物种发现和遥感等多领域。

**[Ordinal Label-Distribution Learning with Constrained Asymmetric Priors for Imbalanced Retinal Grading](ordinal_label-distribution_learning_with_constrained_asymmetric_priors_for_imbal.md)**

:   提出 CAP-WAE（Constrained Asymmetric Prior Wasserstein Autoencoder），通过非对称先验、序数边距正交紧凑损失和方向感知序数损失三重创新，解决糖尿病视网膜病变分级中长尾分布和序数结构的挑战，在多个 DR 基准上达到 SOTA。

**[Orochi: Versatile Biomedical Image Processor](orochi_versatile_biomedical_image_processor.md)**

:   提出 Orochi——首个面向底层生物医学图像处理的通用基础模型，通过任务相关联合嵌入预训练（TJP）和多头层级 Mamba 架构，在配准、融合、复原和超分辨率四大任务上以轻量微调（<5% 参数）即可达到或超越专用 SOTA 模型。

**[Pancakes: Consistent Multi-Protocol Image Segmentation Across Biomedical Domains](pancakes_consistent_multi-protocol_image_segmentation_across_biomedical_domains.md)**

:   提出 Pancakes 框架，给定来自未见过领域的生物医学图像集合，自动生成多个合理分割协议（protocol）的标签图，且同一协议下不同图像的标签具有**语义一致性**——同一标签在所有图像中指代相同的解剖结构。

**[PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation](physiowave_a_multi-scale_wavelet-transformer_for_physiological_signal_representa.md)**

:   提出 PhysioWave，一种基于可学习小波分解和频率引导掩码的多尺度 Transformer 架构，首次为 EMG 和 ECG 构建大规模预训练基础模型，并通过多模态融合框架在单模态和多模态生理信号任务上取得 SOTA 性能。

**[PolyPose: Deformable 2D/3D Registration via Polyrigid Transformations](polypose_deformable_2d3d_registration_via_polyrigid_transformations.md)**

:   提出PolyPose，一种基于多刚体变换（polyrigid）的可变形2D/3D配准方法，利用"骨骼是刚体"这一解剖学先验，将复杂3D形变场参数化为多个刚体变换在切空间 $\mathfrak{se}(3)$ 中的加权组合，无需正则化和超参数调优即可从少至两张X光片实现精确的3D体积配准。

**[Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)**

:   这篇立场论文系统综述了LLM在非结构化临床转录文本主题分析中的应用现状，发现评估方法高度碎片化，并提出以有效性(Validity)、可靠性(Reliability)、可解释性(Interpretability)三维度为核心的标准化评估框架。

**[QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training](qoq-med_building_multimodal_clinical_foundation_models_with_domain-aware_grpo_tr.md)**

:   QoQ-Med 构建了覆盖 9 个临床模态（1D ECG + 6 类 2D 影像 + 2 类 3D 扫描）的多模态临床基础模型，提出域感知相对策略优化（DRPO）——通过层级温度缩放（域间 × 域内 K-means 聚类）解决模态/难度不平衡问题，在 261 万指令调优对上训练后平均 F1 达 0.295（vs GRPO 0.193，+52.8%），8 个模态中 6 个最优。

**[RAD: Towards Trustworthy Retrieval-Augmented Multi-modal Clinical Diagnosis](rad_towards_trustworthy_retrieval-augmented_multi-modal_clinical_diagnosis.md)**

:   提出检索增强诊断框架RAD，通过从多源医学语料中检索疾病指南并注入多模态模型的特征提取和跨模态融合全流程，同时引入双轴可解释性评估体系，在四个不同解剖部位的数据集上达到SOTA。

**[RadZero: Similarity-Based Cross-Attention for Explainable Vision-Language Alignment in Chest X-ray](radzero_similarity-based_cross-attention_for_explainable_vision-language_alignme.md)**

:   提出 RadZero 框架及核心组件 VL-CABS（基于相似度的视觉语言交叉注意力），在胸部X光上实现可解释的、细粒度的视觉语言对齐，支持零样本分类、定位和分割多任务。

**[RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis](ram-w600_a_multi-task_wrist_dataset_and_benchmark_for_rheumatoid_arthritis.md)**

:   首个公开的多任务腕骨常规X光数据集RAM-W600，包含1048张影像，支持腕骨实例分割和SvdH骨侵蚀评分两大任务，并提供全面的基准测试。

**[RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification](raxss_retrieval-augmented_sparse_sampling_for_explainable_variable-length_medica.md)**

:   提出RAxSS框架，将检索增强机制引入随机稀疏采样(SSS)流水线，通过窗口内相似度加权聚合替代均匀平均，在保持变长医学时间序列分类性能的同时提供从"哪里"到"为什么"的可解释性证据链。

**[Revisiting End-to-End Learning with Slide-level Supervision in Computational Pathology](revisiting_end-to-end_learning_with_slide-level_supervision_in_computational_pat.md)**

:   重新审视计算病理中切片级监督的端到端(E2E)学习，首次揭示稀疏注意力MIL在E2E训练中导致的优化困难，提出ABMILX通过多头注意力和全局注意力校正模块解决该问题，使E2E训练的ResNet在多个基准上超越SOTA基础模型。

**[Riemannian Flow Matching for Brain Connectivity Matrices via Pullback Geometry](riemannian_flow_matching_for_brain_connectivity_matrices_via_pullback_geometry.md)**

:   提出DiffeoCFM，利用全局微分同胚诱导的拉回度量，将黎曼流形上的条件流匹配等价转化为欧几里得空间中的标准CFM，实现对脑连接矩阵（SPD/相关矩阵）的高效生成，同时严格保持流形约束，在3个fMRI和2个EEG数据集上达到SOTA。

**[Scalable Diffusion Transformer for Conditional 4D fMRI Synthesis](scalable_diffusion_transformer_for_conditional_4d_fmri_synthesis.md)**

:   提出首个用于体素级全脑4D fMRI条件生成的扩散Transformer，结合3D VQ-GAN潜空间压缩、CNN-Transformer混合骨干网络和AdaLN-Zero+交叉注意力的强条件注入，在HCP七种认知任务上实现任务激活图相关0.83、RSA达0.98和完美条件特异性。

**[Scaling Laws and Pathologies of Single-Layer PINNs: Network Width and PDE Nonlinearity](scaling_laws_and_pathologies_of_single-layer_pinns_network_width_and_pde_nonline.md)**

:   对单层PINN在典型非线性PDE上建立了经验缩放定律，发现了双重优化失败：宽度缩放病理（误差不随宽度下降）和复合病理（非线性加剧此失败），证明优化而非近似容量是主要瓶颈。

**[Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation](self-supervised_learning_of_echocardiographic_video_representations_via_online_c.md)**

:   提出 DISCOVR，一种自监督双分支框架，通过在线语义聚类蒸馏将图像编码器的细粒度空间语义传递到视频编码器的时序表示中，在六个跨胎儿/儿科/成人心脏超声数据集上实现了异常检测、分类和分割的全面领先。

**[Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)**

:   提出 FGNO（Flow-Guided Neural Operator），将 Flow Matching 与算子学习结合用于时间序列自监督预训练，通过 STFT 实现分辨率不变的函数空间学习，并将流时间（flow time）和网络层作为控制特征粒度的"旋钮"，在生物医学任务上显著优于 MAE 等基线。

**[Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)**

:   提出 HeteroTissue-Diffuse（HTD），一种双条件 Latent Diffusion 模型，通过同时以语义分割图和真实组织裁剪块（visual crop）作为条件来生成异质性病理图像，在 Camelyon16 上将 Fréchet Distance 从 430 降至 72（6 倍改善），合成数据训练的 DeepLabv3+ 分割 IoU 与真实数据仅差 1-2%，并通过自监督聚类扩展到 11765 张无标注 TCGA 全幻灯片图像。

**[Sequential Attention-based Sampling for Histopathological Analysis](sequential_attention-based_sampling_for_histopathological_analysis.md)**

:   提出 SASHA 框架，结合层次注意力多实例学习 (HAFED) 与深度强化学习 (RL)，仅采样 10-20% 的高分辨率 patch 即可达到全分辨率 SOTA 方法的分类性能，推理速度提升 4-8 倍，WSI 压缩率超 16 倍。

**[Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs](shallow_robustness_deep_vulnerabilities_multi-turn_evaluation_of_medical_llms.md)**

:   提出MedQA-Followup框架系统评估医学LLM的多轮鲁棒性，发现模型在单轮扰动下表现尚可（浅层鲁棒性），但在多轮追问中准确率可从91.2%暴跌至13.5%（深层脆弱性），且间接上下文操纵比直接错误建议更具破坏力。

**[SMMILE: An Expert-Driven Benchmark for Multimodal Medical In-Context Learning](smmile_an_expert-driven_benchmark_for_multimodal_medical_in-context_learning.md)**

:   提出 SMMILE——首个由 11 位医学专家驱动的多模态医学上下文学习（ICL）基准，包含 111 道问题（517 个图文问答三元组）覆盖 6 个医学专科和 13 种成像模态，系统性揭示了当前 MLLM 在医学多模态 ICL 上的严重不足以及上下文示例质量和顺序对性能的关键影响。

**[STAMP: Spatial-Temporal Adapter with Multi-Head Pooling](stamp_spatial-temporal_adapter_with_multi-head_pooling.md)**

:   STAMP 为时间序列基础模型（TSFM）设计了仅 750K 参数的轻量空间-时间适配器，通过三组位置编码（token/空间/时间）+ 交叉 GMLP 混合 + 多头注意力池化，使冻结的 TSFM（如 MOMENT 385M）在 8 个 EEG 数据集上与 29M 参数的 EEG 专用模型（CBraMod）竞争或超越，在 BCIC-IV-2a 上 Kappa 比 CBraMod 高 193%。

**[STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology](starc-9_a_large-scale_dataset_for_multi-class_tissue_classification_for_crc_hist.md)**

:   提出 STARC-9 大规模结直肠癌组织分类数据集（63 万张图片、9 类组织）及其构建框架 DeepCluster++，通过自编码器特征提取 + K-means 聚类 + 等频分箱采样确保形态多样性，在该数据集上训练的模型显著超越 NCT 和 HMU 训练的模型。

**[Surf2CT: Cascaded 3D Flow Matching Models for Torso 3D CT Synthesis from Skin Surface](surf2ct_cascaded_3d_flow_matching_models_for_torso_3d_ct_synthesis_from_skin_sur.md)**

:   提出 Surf2CT，一种级联式 3D Flow Matching 框架，首次实现仅从外部体表扫描和人口学数据（年龄、性别、身高、体重）合成完整的高分辨率 3D CT 体积，无需任何内部成像输入。

**[SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning](synbrain_enhancing_visual-to-fmri_synthesis_via_probabilistic_representation_lea.md)**

:   提出 SynBrain 框架，通过 BrainVAE 将 fMRI 响应建模为视觉语义条件的概率分布，并用 S2N Mapper 实现一步式语义到神经空间的映射，在视觉-fMRI 合成任务上显著超越 MindSimulator（MSE 降低 65%，Pearson 提升 96%），且合成的 fMRI 可有效增强少样本跨被试解码性能。

**[The Biased Oracle: Assessing LLMs' Understandability and Empathy in Medical Diagnoses](the_biased_oracle_assessing_llms_understandability_and_empathy_in_medical_diagno.md)**

:   系统评估 GPT-4o 和 Claude-3.7 在医疗诊断沟通中的可读性和共情能力，发现两者均产生超标的阅读难度（9-13 年级 vs 推荐的 6-8 年级），情感共情随诊断类型和患者教育水平显著变化，且 LLM-as-Judge 存在严重自我偏见（GPT 对自身共情评分膨胀 ~0.3 分）。

**[The Boundaries of Fair AI in Medical Image Prognosis: A Causal Perspective](the_boundaries_of_fair_ai_in_medical_image_prognosis_a_causal_perspective.md)**

:   FairTTE是首个系统研究医学影像中时间-事件(TTE)预测公平性的综合框架，利用因果分析量化五种偏差来源，通过训练超过20000个模型揭示了现有公平性方法的局限性，特别是在分布偏移下公平性难以维持的根本挑战。

**[The Human Brain as a Combinatorial Complex](the_human_brain_as_a_combinatorial_complex.md)**

:   提出一种数据驱动的框架，利用 S-信息和 O-信息等信息论度量从 fMRI 时间序列中直接构建组合复形（Combinatorial Complexes），将脑区间的高阶协同交互编码到拓扑结构中，为拓扑深度学习应用于脑网络分析奠定基础。

**[THUNDER: Tile-level Histopathology image UNDERstanding benchmark](thunder_tile-level_histopathology_image_understanding_benchmark.md)**

:   提出 THUNDER，一个面向数字病理学基础模型的 tile 级别综合基准，支持 23 个基础模型在 16 个数据集上的高效比较，覆盖下游任务性能、特征空间分析、鲁棒性和不确定性评估。

**[Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation](toward_a_vision-language_foundation_model_for_medical_data_multimodal_dataset_an.md)**

:   构建首个越南语 PET/CT 图像-报告数据集 ViMed-PET（2,757 例全身 PET/CT 体积 + 完整临床报告），通过数据增强策略和三阶段微调流程显著提升 VLM 在医学报告生成和 VQA 任务上的表现，并提出基于临床关键信息的评估指标。

**[Towards Self-Supervised Foundation Models for Critical Care Time Series](towards_self-supervised_foundation_models_for_critical_care_time_series.md)**

:   基于双轴Transformer（BAT）架构，在多个ICU数据集上进行自监督预训练，构建重症监护时间序列基础模型，在小数据集场景下显著优于监督学习基线。

**[UniMRSeg: Unified Modality-Relax Segmentation via Hierarchical Self-Supervised Compensation](unimrseg_unified_modality-relax_segmentation_via_hierarchical_self-supervised_co.md)**

:   提出UniMRSeg，一种统一的模态缺失分割框架，通过层次化自监督补偿机制（HSSC）——从输入级模态重建、特征级对比学习到输出级一致性约束——用100%共享参数在所有可能的模态组合下实现最优平均性能和最小性能波动。

**[Unpaired Image-to-Image Translation for Segmentation and Signal Unmixing](unpaired_image-to-image_translation_for_segmentation_and_signal_unmixing.md)**

:   提出 Ui2i 模型，在 CycleGAN 基础上通过 UNet 生成器、近似双向谱归一化替代特征归一化、通道-空间注意力和尺度增强，实现高内容保真度的无配对图像翻译，成功用于 IHC→H&E 域适应核分割及单通道免疫荧光信号解混两大生物医学任务。

**[Variational Autoencoder with Normalizing Flow for X-ray Spectral Fitting](variational_autoencoder_with_normalizing_flow_for_x-ray_spectral_fitting.md)**

:   将归一化流 (NF) 嵌入自编码器架构中，对黑洞 X 射线双星的 NICER 光谱数据进行快速物理参数推断和完整后验分布估计，比传统 MCMC 方法快约 2000 倍，且精度可比拟。

**[VQ-Seg: Vector-Quantized Token Perturbation for Semi-Supervised Medical Image Segmentation](vq-seg_vector-quantized_token_perturbation_for_semi-supervised_medical_image_seg.md)**

:   提出 VQ-Seg，首次将向量量化引入半监督医学图像分割，用量化扰动模块（QPM）替代传统 dropout 实现更可控的特征扰动，并结合双分支架构和基础模型引导对齐来弥补量化信息损失。

**[Zebra: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding](zebra_towards_zero-shot_cross-subject_generalization_for_universal_brain_visual_.md)**

:   提出 Zebra，首个零样本脑视觉解码框架，通过对抗训练与残差分解将 fMRI 表征解耦为主体不变和语义特定成分，无需对新被试做微调即可实现跨被试的视觉重建泛化。
