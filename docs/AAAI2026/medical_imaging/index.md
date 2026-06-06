---
title: >-
  AAAI2026 医学图像方向88篇论文解读
description: >-
  88篇AAAI2026的医学图像方向论文解读，涵盖医学影像、语义分割、对齐/RLHF、LLM、对抗鲁棒、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "AAAI2026"
  - "医学图像"
  - "论文解读"
  - "论文笔记"
  - "医学影像"
  - "语义分割"
  - "对齐/RLHF"
  - "LLM"
  - "对抗鲁棒"
  - "多模态"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**🤖 AAAI2026** · **88** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (34)](../../ICML2026/medical_imaging/index.md) · [💬 ACL2026 (35)](../../ACL2026/medical_imaging/index.md) · [📷 CVPR2026 (103)](../../CVPR2026/medical_imaging/index.md) · [🔬 ICLR2026 (48)](../../ICLR2026/medical_imaging/index.md) · [🧠 NeurIPS2025 (92)](../../NeurIPS2025/medical_imaging/index.md) · [📹 ICCV2025 (31)](../../ICCV2025/medical_imaging/index.md)

🔥 **高频主题：** 医学影像 ×39 · 语义分割 ×11 · 对齐/RLHF ×8 · LLM ×7 · 对抗鲁棒 ×6

**[A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)**

:   提出一种两阶段疾病感知框架，通过学习14个与病理类别对应的疾病感知语义token（DASTs）实现显式的疾病表征，再利用疾病-视觉注意力融合（DVAF）和双模态相似性检索（DMSR）机制辅助LLM生成临床准确的胸部X光报告，在CheXpert Plus、IU X-Ray和MIMIC-CXR三个数据集上取得SOTA。

**[A Principle-Driven Adaptive Policy for Group Cognitive Stimulation Dialogue for Elderly with Cognitive Impairment](a_principle-driven_adaptive_policy_for_group_cognitive_stimu.md)**

:   针对老年认知障碍患者的群体认知刺激治疗（CST）场景，提出GCSD系统：通过多说话人上下文控制、动态参与者状态建模（soft prompt）、认知刺激注意力损失和多维奖励策略优化四个模块，基于Qwen-2.5-3B微调，在500+小时真实粤语CST对话和1万+模拟对话上训练，BLEU-4达27.93超越GPT-4o等大模型，A/B测试胜率50% vs GPT-4o的39%。

**[Advancing Safe Mechanical Ventilation Using Offline RL With Hybrid Actions and Clinically Aligned Rewards](advancing_safe_mechanical_ventilation_using_offline_rl_with_.md)**

:   针对ICU机械通气（MV）设置优化问题，提出混合动作空间的离线RL方法（HybridIQL/HybridEDAC），避免传统离散化导致的分布偏移，同时引入基于无通气天数（VFD）和生理参数安全范围的临床对齐奖励函数，通过多目标优化选择最优奖励，将可优化的通气参数从2-3个扩展到6个，HybridIQL在性能和策略覆盖率间取得最佳平衡。

**[Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation](ambiguity-aware_truncated_flow_matching_for_ambiguous_medica.md)**

:   提出 ATFM 框架，通过数据层级推理范式将预测精度和多样性解耦到分布级和样本级分别优化，结合高斯截断表示（GTR）和分割流匹配（SFM）两个模块，在模糊医学图像分割任务中同时提升预测的精度、保真度和多样性。

**[Bayesian Meta-Analyses Could Be More: A Case Study in Trial of Labor After a Cesarean-section Outcomes and Complications](bayesian_meta-analyses_could_be_more_a_case_study_in_trial_of_labor_after_a_cesa.md)**

:   提出一种层次贝叶斯 meta-analysis 方法，通过对未记录的决策变量（Bishop 分数）建模为截断隐变量，纠正传统固定效应 meta-analysis 中因忽略混杂因子而导致的偏差结论，在 TOLAC（剖宫产后试产）场景中证明机械扩张与 Pitocin 无显著差异。

**[Bidirectional Channel-selective Semantic Interaction for Semi-Supervised Medical Segmentation](bidirectional_channel-selective_semantic_interaction_for_semi-supervised_medical.md)**

:   提出 BCSI 框架，通过通道选择路由器动态筛选关键特征通道，在标注和未标注数据流之间进行双向通道级交互，结合语义-空间扰动的弱到强一致性学习，显著提升半监督医学图像分割性能。

**[Bridging Vision and Language for Robust Context-Aware Surgical Point Tracking: The VL-SurgPT Dataset and Benchmark](bridging_vision_and_language_for_robust_context-aware_surgical_point_tracking_th.md)**

:   提出首个大规模多模态手术点追踪数据集 VL-SurgPT，结合视觉坐标与文本状态描述，并设计文本引导追踪方法 TG-SurgPT，通过语义信息显著提升复杂手术场景下的追踪精度和鲁棒性。

**[CD-DPE: Dual-Prompt Expert Network Based on Convolutional Dictionary Feature Decoupling for Multi-Contrast MRI Super-Resolution](cd-dpe_dual-prompt_expert_network_based_on_convolutional_dictionary_feature_deco.md)**

:   提出 CD-DPE 网络，通过迭代卷积字典特征解耦模块（CD-FDM）将多对比度 MRI 特征分离为跨对比度共有和模态特有成分，再利用双提示特征融合专家模块（DP-FFEM）进行自适应融合重建，在多个公开数据集上超越现有 SOTA 方法。

**[CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records](clicare_grounding_large_language_models_in_clinical_guidelines_for_decision_supp.md)**

:   提出 CliCARE 框架，将非结构化的纵向癌症电子病历（EHR）转化为时序知识图谱（TKG），并与临床指南知识图谱对齐融合，为 LLM 提供循证依据的临床决策支持，同时设计了与专家评估高度相关的 LLM-as-a-Judge 评估协议。

**[Coarse-to-Fine Open-Set Graph Node Classification with Large Language Models](coarse-to-fine_open-set_graph_node_classification_with_large_language_models.md)**

:   提出 Coarse-to-Fine Classification (CFC) 框架，利用 LLM 的零样本推理能力为图节点开放集分类提供语义化 OOD 样本和潜在 OOD 标签空间，实现不仅检测 OOD 还能将其分类到具体未知类别的能力。

**[CoCoLIT: ControlNet-Conditioned Latent Image Translation for MRI to Amyloid PET Synthesis](cocolit_controlnet-conditioned_latent_image_translation_for_mri_to_amyloid_pet_s.md)**

:   提出 CoCoLIT 框架，基于 ControlNet 条件化的潜在扩散模型，从结构 MRI 合成淀粉样蛋白 PET 图像，通过加权图像空间损失（WISL）和潜在平均稳定化（LAS）显著超越现有方法。

**[Cross-Sample Augmented Test-Time Adaptation for Personalized Intraoperative Hypotension Prediction](cross-sample_augmented_test-time_adaptation_for_personalized_intraoperative_hypo.md)**

:   提出 CSA-TTA 框架，通过跨样本库构建、粗到细检索和多任务优化，在测试时从其他患者数据中检索低血压事件信号来增强个性化术中低血压预测。

**[Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation](decoding_with_structured_awareness_integrating_directional_frequency-spatial_and.md)**

:   提出面向医学图像分割的新型解码器框架，包含三个模块：方向感知的自适应交叉融合注意力（ACFA）、空间-频率-小波三分支融合注意力（TFFA）和结构感知多尺度掩码模块（SMMM），在多个基准数据集上超越现有方法。

**[DeepGB-TB: A Risk-Balanced Cross-Attention Gradient-Boosted Convolutional Network for Rapid, Interpretable Tuberculosis Screening](deepgb-tb_a_risk-balanced_cross-attention_gradient-boosted_convolutional_network.md)**

:   提出 DeepGB-TB，一个结合轻量级1D-CNN（处理咳嗽音频）和梯度提升决策树（处理人口统计特征）的多模态TB筛查系统，通过双向交叉注意力（CM-BCA）模拟临床推理过程融合异构数据，配合风险平衡损失（TRBL）最小化漏诊，在7国数据集上达到 AUROC 0.903，可在手机上离线实时运行。

**[DeNAS-ViT: Data Efficient NAS-Optimized Vision Transformer for Ultrasound Image Segmentation](denas-vit_data_efficient_nas-optimized_vision_transformer_for_ultrasound_image_s.md)**

:   提出 DeNAS-ViT，首次将 NAS 应用于 ViT 的 Token 级搜索实现超声图像分割的多尺度特征提取优化，并设计基于 NAS 约束的半监督学习框架（网络独立性损失+层次对比损失+阶段式优化），在有限标注数据下达到 SOTA。

**[DiA-gnostic VLVAE: Disentangled Alignment-Constrained Vision Language Variational AutoEncoder for Robust Radiology Reporting with Missing Modalities](dia-gnostic_vlvae_disentangled_alignment-constrained_vision_language_variational.md)**

:   提出 DiA-gnostic VLVAE，通过视觉-语言混合专家VAE学习三因子潜空间（视觉特有/语言特有/共享），配合正交性+对比对齐的双约束实现解纠缠，使模型在临床上下文缺失时仍能生成可靠的放射学报告，在 IU X-Ray 和 MIMIC-CXR 上达到竞争性 BLEU@4。

**[Divide, Conquer and Unite: Hierarchical Style-Recalibrated Prototype Alignment for Federated Medical Segmentation](divide_conquer_and_unite_hierarchical_style-recalibrated_prototype_alignment_for.md)**

:   针对联邦医学图像分割中的"层间风格偏差累积"和"上下文表征不完整"两大挑战，提出FedBCS框架：通过频域自适应风格重校准（FSR）构建领域不变原型，并设计上下文感知的双层原型对齐（CDPA）融合编解码器多层级语义，在组织核分割和前列腺MRI分割任务上达到SOTA。

**[Do Large Language Models Think Like the Brain? Sentence-Level Evidences from Layer-Wise Embeddings and fMRI](do_large_language_models_think_like_the_brain_sentence-level_evidences_from_laye.md)**

:   本文通过对比14个公开LLM的逐层表示与人类被试听自然叙事时的fMRI数据，在句子级别系统地研究了LLM与人脑语言处理的对齐程度，发现中间层最对齐、指令微调显著增强对齐、且存在与经典神经语言学理论一致的半球偏侧化模式。

**[DualFete: Revisiting Teacher-Student Interactions from a Feedback Perspective for Semi-supervised Medical Image Segmentation](dualfete_revisiting_teacher-student_interactions_from_a_feedback_perspective_for.md)**

:   在教师-学生半监督学习框架中引入反馈机制，让学生能将伪标签引导的更新是否与有标签数据方向一致的信息反馈给教师，并在双教师架构中进一步增强反馈动态性，有效遏制了医学图像分割中的错误累积和确认偏差。

**[DW-DGAT: Dynamically Weighted Dual Graph Attention Network for Neurodegenerative Disease Diagnosis](dw-dgat_dynamically_weighted_dual_graph_attention_network_for_neurodegenerative_.md)**

:   针对神经退行性疾病（PD/AD）早期诊断中的多指标数据融合、异质信息提取和类别不平衡三大挑战，提出动态加权双图注意力网络DW-DGAT，通过通用数据融合策略、微观-宏观双层图特征学习和动态类别权重生成机制，在PPMI和ADNI3数据集上大幅超越14种基线方法。

**[EgoEMS: A High-Fidelity Multimodal Egocentric Dataset for Cognitive Assistance in Emergency Medical Services](egoems_a_high-fidelity_multimodal_egocentric_dataset_for_cognitive_assistance_in.md)**

:   发布首个高保真多人多模态自我中心EMS数据集，包含233个试验20小时视频、9项干预67个关键步骤标注，提供三个基准任务（步骤分类/在线分割/CPR质量估计）推动EMS认知协助系统开发。

**[Error Correction in Radiology Reports: A Knowledge Distillation-Based Multi-Stage Framework](error_correction_in_radiology_reports_a_knowledge_distillation-based_multi-stage.md)**

:   提出了一种**分阶段推理 + 双知识注入**框架，将放射学报告的错误校正分解为检测→定位→纠正三个阶段，结合**医学知识图谱蒸馏（MKGD）** 和**外部知识检索（EXKR）**，在 6 个 LLM 架构上实现了高达 **31.56% 的错误检测准确率提升**和 **37.4% 的处理时间减少**。

**[Experience with Single Domain Generalization in Real World Medical Imaging Deployments](experience_with_single_domain_generalization_in_real_world_medical_imaging_deplo.md)**

:   提出DL+EKE框架，将领域不变的专家知识与深度学习集成，解决医学影像中稀有类（rare class）的单域泛化（SDG）问题，在糖尿病视网膜病变分级、rs-fMRI癫痫灶定位和应激心电图CAD检测三个真实部署场景中显著优于SOTA SDG方法。

**[Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering](expert-guided_prompting_and_retrieval-augmented_generation_for_emergency_medical.md)**

:   构建首个EMS急救领域多选QA数据集EMSQA（24.3K题、10个临床主题、4个认证等级），提出Expert-CoT和ExpertRAG框架将领域专业属性注入LLM推理与检索，比标准RAG最高提升4.59%准确率。

**[FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention](fane_towards_fine-grained_cross-modal_contrast_with_false-negative_reduction_and.md)**

:   FaNe 提出了一个语义增强的医学视觉-语言预训练框架，通过语义感知正样本挖掘、文本条件稀疏注意力池化和难负例感知对比损失，解决医学 VLP 中的假阴性问题和粗粒度对齐不足问题。

**[FDP: A Frequency-Decomposition Preprocessing Pipeline for Unsupervised Anomaly Detection in Brain MRI](fdp_a_frequency-decomposition_preprocessing_pipeline_for_unsupervised_anomaly_de.md)**

:   首次系统分析脑 MRI 异常的频域特征，发现病变主要集中在低频分量中，据此提出**频率分解预处理（FDP）**框架，通过可学习先验上下文库重建低频信号来抑制病变同时保留解剖结构，作为即插即用模块可一致提升多种 UAD 基线的检测性能（LDM 上 DICE 提升 17.63%）。

**[FIA-Edit: Frequency-Interactive Attention for Efficient and High-Fidelity Inversion-Free Text-Guided Image Editing](fia-edit_frequency-interactive_attention_for_efficient_and_high-fidelity_inversi.md)**

:   提出 FIA-Edit，一个基于频域交互注意力的无反转（inversion-free）文本引导图像编辑框架，通过频率表示交互（FRI）模块在自注意力中进行源/目标特征的频域融合，以及特征注入（FIJ）模块在交叉注意力中显式引入源图像特征，在保持背景高保真度的同时实现精确语义编辑，并首次将通用图像编辑方法应用于临床手术出血图像增强。

**[Fine-Tuned LLMs Know They Don't Know: A Parameter-Efficient Approach to Recovering Honesty](fine-tuned_llms_know_they_dont_know_a_parameter-efficient_approach_to_recovering.md)**

:   揭示了 SFT 导致 LLM 不诚实的根源是**自我表达能力受损**（而非自我认知被破坏），基于此提出 HCNR 框架，通过 Fisher 信息识别诚实关键神经元并恢复到预训练状态 + Hessian 引导补偿，仅用 256 条数据和 20% 参数即可恢复 33.25% 的诚实性，实现 2.23 倍以上加速。

**[From Policy to Logic for Efficient and Interpretable Coverage Assessment](from_policy_to_logic_for_efficient_and_interpretable_coverage_assessment.md)**

:   本文提出一种神经符号方法，通过覆盖感知检索器（coverage-aware retriever）与基于PyKnow的符号规则推理相结合，帮助人类审查员高效、可解释地评估医疗CPT代码是否被保险政策覆盖，在推理成本降低44%的同时F1提升4.5%。

**[FunKAN: Functional Kolmogorov-Arnold Network for Medical Image Enhancement and Segmentation](funkan_functional_kolmogorov-arnold_network_for_medical_image_enhancement_and_se.md)**

:   本文将 Kolmogorov-Arnold 表示定理从有限维标量空间推广到函数空间（Hilbert 空间），提出 FunKAN 框架，通过在 Hermite 基函数上进行 Fourier 展开来学习内函数，保留了图像数据的空间结构，在 MRI 增强和三个医学分割任务上均超越已有 KAN 变体。

**[G2L: From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Efficient Fine-Tuning](g2lfrom_giga-scale_to_cancer-specific_large-scale_pathology_foundation_models_vi.md)**

:   本文提出 G2L（Giga-to-Large）蒸馏框架，仅用 1K 张病理切片将 19 亿参数的 giga-scale 病理基础模型（H-optimus-0）的知识蒸馏到 3 亿参数的 large-scale 模型（Hibou-L），在多个癌症特异性下游任务上达到甚至超越教师模型和更大模型的性能。

**[GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs](gem_generative_entropy-guided_preference_modeling_for_few-shot_alignment_of_llms.md)**

:   GEM 提出了一种生成式熵引导偏好建模方法，通过认知过滤（基于熵的 CoT 评分）和 SEGA 算法（自评估组优势策略优化），在仅 3000 个偏好对的低资源场景下实现高效的 LLM 对齐。

**[GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](giim_graph-based_learning_of_inter-_and_intra-view_dependencies_for_multi-view_m.md)**

:   提出基于多异构图（MHG）的GIIM框架，通过图结构同时建模病灶间的视图内依赖和视图间动态变化，并引入四种缺失视图表示策略，在肝脏CT、乳腺X线和乳腺MRI三种模态上显著超越现有多视图方法。

**[Graph-Theoretic Consistency for Robust and Topology-Aware Semi-Supervised Histopathology Segmentation](graph-theoretic_consistency_for_robust_and_topology-aware_semi-supervised_histop.md)**

:   本文提出 TGC（Topology Graph Consistency）框架，通过对齐预测图与参考图之间的拉普拉斯谱、连通分量数和邻接统计量来引入图论拓扑约束，在仅 5-10% 标注下实现接近全监督的组织病理学分割性能。

**[GROVER: Graph-guided Representation of Omics and Vision with Expert Regulation for Cancer Survival Prediction](grover_graph-guided_representation_of_omics_and_vision_with_expert_regulation_fo.md)**

:   提出空间多组学框架GROVER，通过KAN-GCN编码器捕获非线性空间-特征依赖、spot-feature-pair对比学习对齐异构模态、以及自适应混合专家（MoE）动态路由过滤低质量信号，在四个真实空间组学数据集上实现了优于现有方法的聚类性能。

**[GuideGen: A Text-Guided Framework for Paired Full-Torso Anatomy and CT Volume Generation](guidegen_a_text-guided_framework_for_paired_full-torso_anatomy_and_ct_volume_gen.md)**

:   GuideGen 提出了一个仅需文本输入的可控框架，通过分类扩散模型合成全躯干解剖掩码，结合解剖感知高动态范围自编码器和潜在特征生成器，生成配对的全躯干 CT 体积，为下游分割任务提供高质量合成训练数据。

**[Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)**

:   HSO 提出了一种层次化调度优化器，通过双层优化框架（上层全局搜索最优初始化策略 + 下层局部优化调度精炼），在仅 8 秒一次性优化代价下实现扩散模型极低 NFE 下的 SOTA 免训练采样质量。

**[Human-in-the-Loop Interactive Report Generation for Chronic Disease Adherence](human-in-the-loop_interactive_report_generation_for_chronic_disease_adherence.md)**

:   本文设计了一个"医生在回路"的交互界面，将 AI 限定于数据组织和草稿生成角色，通过单页面编辑器、图表-文本配对和自动紧急度分级，实现了高效且可问责的慢性病依从性报告生成。试点研究揭示了一个"问责悖论"：即便 AI 生成质量达到了医生手动撰写基线水平，审阅时间仍无法显著减少，因为临床责任要求完整核验。

**[Intervention Efficiency and Perturbation Validation Framework: Capacity-Aware and Robust Clinical Model Selection under the Rashomon Effect](intervention_efficiency_and_perturbation_validation_framework_capacity-aware_and.md)**

:   针对临床小样本、类别不平衡场景下多个模型性能相近（Rashomon Effect）导致的模型选择困难，提出 **Intervention Efficiency (IE)** 容量感知评估指标和 **Perturbation Validation Framework (PVF)** 鲁棒性验证框架，联合实现资源约束下的可靠模型选择。

**[Learning with Preserving for Continual Multitask Learning](learning_with_preserving_for_continual_multitask_learning.md)**

:   提出 Learning with Preserving（LwP）框架，通过动态加权距离保持（DWDP）损失函数维护共享表示空间的几何结构，在无需回放缓冲的条件下解决持续多任务学习（CMTL）中的灾难性遗忘问题，在 BDD100k、CelebA、PhysiQ 等基准上显著超越现有持续学习方法，并且是唯一超越单任务学习基线的方法。

**[LungNoduleAgent: A Collaborative Multi-Agent System for Precision Diagnosis of Lung Nodules](lungnoduleagent_a_collaborative_multi-agent_system_for_precision_diagnosis_of_lu.md)**

:   提出 LungNoduleAgent，首个面向肺结节分析的协作式多智能体系统，通过"Nodule Spotter + Simulated Radiologist + Doctor Agent System"三阶段流水线模拟临床工作流，在 CT 报告生成和恶性分级任务上大幅超越 GPT-4o、Claude 3.7 Sonnet 等主流 VLM 及 MedAgent-Pro 等医学智能体。

**[MAISI-v2: Accelerated 3D High-Resolution Medical Image Synthesis with Rectified Flow and Region-specific Contrastive Loss](maisi-v2_accelerated_3d_high-resolution_medical_image_synthesis_with_rectified_f.md)**

:   提出 MAISI-v2，首个将 Rectified Flow 引入 3D 医学图像合成的框架，通过替换 DDPM 实现 33 倍加速，并设计区域特异性对比损失增强对肿瘤等小区域条件的忠实度，在下游肿瘤分割任务中验证了合成数据的增强价值。

**[MAMA-Memeia! Multi-Aspect Multi-Agent Collaboration for Depressive Symptoms Identification in Memes](mama-memeia_multi-aspect_multi-agent_collaboration_for_depressive_symptoms_ident.md)**

:   本文提出 MAMAMemeia，一个基于认知分析疗法（CAT）能力框架的多智能体多方面协作讨论框架，用于从社交媒体表情包中识别抑郁症状，同时引入 RESTOREx 资源（含 LLM 生成和人工标注的解释），在 macro-F1 上超越 30+ 种方法 7.55%。

**[MAPI-GNN: Multi-Activation Plane Interaction Graph Neural Network for Multimodal Medical Diagnosis](mapi-gnn_multi-activation_plane_interaction_graph_neural_network_for_multimodal_.md)**

:   提出 MAPI-GNN，通过多维特征判别器在语义子空间中动态构建多个激活图，再经层次化融合网络聚合样本内和样本间关系，在前列腺癌和冠心病两个多模态诊断任务上显著超越现有 SOTA（PI-CAI 上 ACC 0.9432，AUC 0.9838）。

**[MCTSr-Zero: Self-Reflective Psychological Counseling Dialogues Generation via Principles and Adaptive Exploration](mctsr-zero_self-reflective_psychological_counseling_dialogues_generation_via_pri.md)**

:   提出 MCTSr-Zero 框架，将 MCTS 与领域原则自评估、元提示自适应探索机制结合，用于生成高质量心理咨询多轮对话数据，微调得到的 PsyLLM 在自建的 PsyEval 基准上达到 SOTA。

**[Measuring Stability Beyond Accuracy in Small Open-Source Medical Large Language Models for Pediatric Endocrinology](measuring_stability_beyond_accuracy_in_small_open-source_medical_large_language_.md)**

:   系统评估了6个小型开源医学LLM（<10B参数）在儿科内分泌领域的表现，揭示仅靠准确率不足以衡量模型可靠性：语义无关的提示微调导致模型输出显著变化（Stuart-Maxwell p<10⁻⁴），高一致性不等于正确，甚至CUDA版本差异也能引发统计显著的输出偏移。

**[MedEyes: Learning Dynamic Visual Focus for Medical Progressive Diagnosis](medeyes_learning_dynamic_visual_focus_for_medical_progressive_diagnosis.md)**

:   提出 MedEyes，一个混合策略强化学习框架，通过注视引导推理导航器（GRN）模拟临床医生"扫描-钻探"的诊断视觉搜索模式，结合置信度值采样器（CVS）和双流 GRPO 优化，实现动态视觉聚焦的医学渐进式诊断推理，在五个医学 VQA 基准上平均提升 8.5pp。

**[MindCross: Fast New Subject Adaptation with Limited Data for Cross-subject Video Reconstruction from Brain Signals](mindcross_fast_new_subject_adaptation_with_limited_data_for_cross-subject_video_.md)**

:   提出 MindCross，一个跨被试脑解码框架，通过共享编码器学习被试无关信息 + N个特有编码器学习被试相关信息，配合快速校准阶段和 Top-K 协作解码模块，仅用一个模型在 fMRI/EEG-to-video 基准上实现与被试独立模型可比的性能，且新被试适应仅需极少数据和极短时间（~1秒 vs 基线5-17秒）。

**[MIRAGE: Scaling Test-Time Inference with Parallel Graph-Retrieval-Augmented Reasoning Chains](mirage_scaling_test-time_inference_with_parallel_graph-retrieval-augmented_reaso.md)**

:   提出MIRAGE框架，将传统的线性推理链扩展为并行多链推理范式，结合结构化医学知识图谱的自适应检索（邻域扩展和多跳遍历），通过跨链验证解决矛盾，在三个医学QA基准上持续优于GPT-4o、ToT和Search-o1等方法。

**[MIRNet: Integrating Constrained Graph-Based Reasoning with Pre-training for Diagnostic Medical Imaging](mirnet_integrating_constrained_graph-based_reasoning_with_pre-training_for_diagn.md)**

:   提出MIRNet框架，将自监督掩码自编码器（MAE）预训练与约束感知的图注意力网络（GAT）推理相结合，用于舌象多标签诊断，并发布包含4000张图像22个标签的TongueAtlas-4K基准数据集，Macro Recall提升77.8%、Macro-F1提升33.2%。

**[MPA: Multimodal Prototype Augmentation for Few-Shot Learning](mpa_multimodal_prototype_augmentation_for_few-shot_learning.md)**

:   本文提出 MPA 框架，通过 LLM 生成多变体语义描述增强原型的语义信息（LMSE）、层次化多视角数据增强丰富视觉特征（HMA）、以及自适应不确定类吸收器建模类间不确定性（AUCA），在 4 个单域和 6 个跨域小样本学习基准上显著超越现有方法，5-way 1-shot 下单域和跨域分别比次优方法高出 12.29% 和 24.56%。

**[Multivariate Gaussian Representation Learning for Medical Action Evaluation](multivariate_gaussian_representation_learning_for_medical_action_evaluation.md)**

:   提出 GaussMedAct 框架，将关节运动轨迹建模为多元高斯混合分布并结合笛卡尔-向量双流编码，在自建的 CPREval-6k 数据集上实现 92.1% Top-1 准确率，仅需 ST-GCN 10% 的计算量。

**[Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks](neural_bandit_based_optimal_llm_selection_for_a_pipeline_of_tasks.md)**

:   提出 Sequential Bandits 算法，一种基于神经上下文多臂老虎机的在线学习方法，用于在任务流水线（如"摘要→诊断"）中为每个子任务选择最优 LLM，同时优化准确率和成本，在医学诊断和电信问答两个流水线任务上优于现有 bandit 基线。

**[NeuroBridge: Bio-Inspired Self-Supervised EEG-to-Image Decoding via Cognitive Priors and Bidirectional Semantic Alignment](neurobridge_bio-inspired_self-supervised_eeg-to-image_decoding_via_cognitive_pri.md)**

:   提出NeuroBridge框架，通过认知先验增强（CPA，非对称增广模拟感知变异性）和共享语义投影器（SSP，双向对齐到统一语义空间），在THINGS-EEG数据集200类零样本EEG-图像检索任务上达到63.2% Top-1（+12.3%）和89.9% Top-5（+10.2%），大幅超越现有SOTA。

**[Note2Chat: Improving LLMs for Multi-Turn Clinical History Taking Using Medical Notes](note2chat_improving_llms_for_multi-turn_clinical_history_taking_using_medical_no.md)**

:   提出 Note2Chat 框架，利用广泛可得的医学笔记（而非稀缺的对话数据）训练 LLMs 进行结构化问诊和诊断，通过笔记驱动的对话生成、三阶段微调策略和单轮推理范式，在信息收集（F1 +16.9）和诊断准确率（Top-1 +21.0）上大幅超越 GPT-4o。

**[NutriScreener: Retrieval-Augmented Multi-Pose Graph Attention Network for Malnourishment Screening](nutriscreener_retrieval-augmented_multi-pose_graph_attention_network_for_malnour.md)**

:   提出 NutriScreener，一个结合CLIP视觉编码器、多姿态图注意力网络（GAT）和基于FAISS的检索增强分类/回归模块的框架，通过跨姿态注意力和类别增强检索来实现鲁棒的儿童营养不良检测与人体测量学预测，在AnthroVision等跨大洲数据集上达到0.79 recall和0.82 AUC，临床医生评价准确性4.3/5、效率4.6/5。

**[CountVid: Open-World Object Counting in Videos](open-world_object_counting_in_videos.md)**

:   提出 CountVid 模型和 VideoCount 数据集，首次系统研究开放世界视频物体计数任务——给定文本或图像描述指定目标物体，枚举视频中所有独特实例，通过组合图像计数模型和可提示视频分割追踪模型解决遮挡、重复出现等挑战，在包含 TAO、MOT20、企鹅群和 X 射线金属结晶等多样化场景上显著优于多种强基线。

**[Pairing-free Group-level Knowledge Distillation for Robust Gastrointestinal Lesion Classification in White-Light Endoscopy](pairing-free_group-level_knowledge_distillation_for_robust_gastrointestinal_lesi.md)**

:   提出 PaGKD，一个无需配对样本的组级知识蒸馏框架，通过组级原型蒸馏（GKD-Pro，用共享的病变查询Transformer提取模态不变语义原型）和组级密集蒸馏（GKD-Den，用激活图引导的语义关系交叉注意力实现密集空间对齐），突破传统NBI→WLI跨模态蒸馏对配对数据的依赖，在四个临床数据集上AUC分别提升3.3%/1.1%/2.8%/3.2%。

**[PanFoMa: A Lightweight Foundation Model and Benchmark for Pan-Cancer Pathology Image Analysis](panfoma_a_lightweight_foundation_model_and_benchmark_for_pan-cancer.md)**

:   提出 PanFoMa，一种融合 Transformer 局部建模与 Mamba 全局整合的轻量级混合神经网络，用于泛癌单细胞转录组表示学习；同时构建了覆盖 33 种癌症亚型、350 万+ 细胞的大规模基准数据集 PanFoMaBench。

**[Personality-guided Public-Private Domain Disentangled Hypergraph-Former Network for Multimodal Depression Detection](personality-guided_public-private_domain_disentangled_hypergraph-former_network_.md)**

:   提出 P3HF 框架，通过人格引导的特征门控、时序感知的超图-Transformer（Hypergraph-Former）架构和事件级公私域解耦三大创新，在多事件多模态抑郁检测任务上实现约 10% 的准确率和 F1 提升。

**[Personalization of Large Foundation Models for Health Interventions](personalization_of_large_foundation_models_for_health_interventions.md)**

:   系统性分析大基础模型（LFMs）在个性化健康干预中的四大结构性矛盾，论证 LFMs 无法替代 N-of-1 试验，提出结合 LFMs 假设生成与 N-of-1 试验因果验证的混合框架。

**[PINGS-X: Physics-Informed Normalized Gaussian Splatting with Axes Alignment for Efficient Super-Resolution of 4D Flow MRI](pings-x_physics-informed_normalized_gaussian_splatting_with_axes_alignment_for_e.md)**

:   提出PINGS-X框架，将3D高斯溅射（3DGS）的显式表示思想引入物理信息超分辨率领域，通过归一化高斯溅射（NGS）、轴对齐高斯和高斯合并三项创新，在合成CFD和真实4D Flow MRI数据集上实现了比PINN快一个数量级的训练速度，同时保持更高的超分辨率精度。

**[PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation](priorrg_prior-guided_contrastive_pre-training_and_coarse-to-fine_decoding_for_ch.md)**

:   PriorRG 提出了一个两阶段胸部X光报告生成框架，通过先验引导的对比预训练对齐临床语境与时空视觉特征，再通过先验感知的粗到细解码逐步融合临床上下文、疾病进展和多层级视觉线索，在 MIMIC-CXR 上实现 BLEU-4 提升 3.6%、F1 提升 3.8%。

**[ProPL: Universal Semi-Supervised Ultrasound Image Segmentation via Prompt-Guided Pseudo-Labeling](propl_universal_semi-supervised_ultrasound_image_segmentation_via_prompt-guided_.md)**

:   提出 ProPL 框架，通过共享视觉编码器 + 提示引导双解码器 + 不确定性驱动伪标签校准，首次实现通用半监督超声图像分割，在 5 个器官 8 个任务上以极少标注数据（1/16）超越全监督方法 5.18% mDice。

**[Provably Minimum-Length Conformal Prediction Sets for Ordinal Classification](provably_minimum-length_conformal_prediction_sets_for_ordinal_classification.md)**

:   提出 min-CPS 及其正则化变体 min-RCPS，一种模型无关的序数保形预测方法，通过线性时间滑动窗口算法求解每个样本的最小长度预测区间，在保证覆盖率的同时平均减少 15% 的预测集大小，且提供了实例级最优性的理论保证。

**[Q-FSRU: Quantum-Augmented Frequency-Spectral Fusion for Medical Visual Question Answering](q-fsru_quantum-augmented_frequency-spectral_fusion_for_medical_visual_question_a.md)**

:   提出 Q-FSRU 模型，将医学图像和文本特征转换到频域（FFT）进行融合，并结合量子启发的检索增强生成（Quantum RAG）引入外部医学知识，在 VQA-RAD 数据集上取得 90% 准确率和 0.9541 的 ROC-AUC。

**[qa-FLoRA: Data-free Query-Adaptive Fusion of LoRAs for LLMs](qa-flora_data-free_query-adaptive_fusion_of_loras_for_llms.md)**

:   提出 qa-FLoRA，一种无需训练数据和训练过程的查询自适应 LoRA 融合方法，通过逐层计算各适配器与基座模型间的 KL 散度来动态确定融合权重，在九个多语言复合任务上显著优于静态融合和无训练基线。

**[Radiation-Preserving Selective Imaging for Pediatric Hip Dysplasia: A Cross-Modal Approach](radiation-preserving_selective_imaging_for_pediatric_hip_dysplasia_a_cross-modal.md)**

:   提出一种"超声优先、保辐射"的跨模态选择性成像策略，通过自监督预训练的冻结编码器、测量忠实的轻量头网络和共形预测校准的单侧下界，实现了在发育性髋关节发育不良（DDH）诊断中有据可依地决定何时仅用超声即可、何时需要额外的 X 光检查。

**[ReCoN-Ipsundrum: An Inspectable Recurrent Persistence Loop Agent with Affect-Coupled Cognition](recon-ipsundrum_an_inspectable_recurrent_persistence_loop_agent_with_affect-coup.md)**

:   实现ReCoN-Ipsundrum——一个可检查的智能体架构，在ReCoN感觉运动状态机上扩展了Humphrey的ipsundrum递归持续循环和可选的情感代理层，通过行为测试和因果消融实验证明：递归支撑刺激后持续性，情感耦合支撑偏好稳定性、结构化扫描和持久谨慎，并强调行为标记单独不足以归因意识。

**[Refine and Align: Confidence Calibration through Multi-Agent Interaction in VQA](refine_and_align_confidence_calibration_through_multi-agent_interaction_in_vqa.md)**

:   提出 AlignVQA，一个基于多智能体辩论的VQA置信度校准框架：专家agent生成候选答案后，通用agent进行结构化辩论（支持论据 vs 反对论据）来修正置信度；同时提出可微分的校准感知损失 AlignCal，通过最小化校准误差上界（UBCE）来训练更校准的agent，在VQARad和ScienceQA上将ECE从0.375降至0.098。

**[Rethinking Bias in Generative Data Augmentation for Medical AI: a Frequency Recalibration Approach](rethinking_bias_in_generative_data_augmentation_for_medical_ai_a_frequency_recal.md)**

:   揭示 AI 生成医学图像与真实图像之间的高频频率分布差异是生成式数据增强（GDA）不可靠的关键原因，提出 FreRec（Frequency Recalibration）方法，通过统计高频替换（SHR）和重建式高频映射（RHM）两步实现粗到细的频率分布对齐，作为即插即用的后处理模块显著提升下游医学图像分类性能。

**[Rethinking Surgical Smoke: A Smoke-Type-Aware Laparoscopic Video Desmoking Method and Dataset](rethinking_surgical_smoke_a_smoke-type-aware_laparoscopic_video_desmoking_method.md)**

:   本文首次将手术烟雾分为扩散烟（Diffusion Smoke）和环境烟（Ambient Smoke）两种类型，提出了第一个烟雾类型感知的腹腔镜视频去烟网络 STANet，包含语义软分割、粗到精解耦和双分支重建三个子网络，并构建了首个包含烟雾类型标注的大规模合成视频去烟数据集 STSVD。

**[Self-supervised Multiplex Consensus Mamba for General Image Fusion](self-supervised_multiplex_consensus_mamba_for_general_image_fusion.md)**

:   提出 SMC-Mamba 框架，通过**模态无关特征增强（MAFE）**、**多路共识跨模态 Mamba（MCCM）**和**双层自监督对比学习损失（BSCL）**，实现覆盖红外-可见光、医学、多聚焦、多曝光的通用图像融合，全面超越 SOTA。

**[SEMC: Structure-Enhanced Mixture-of-Experts Contrastive Learning for Ultrasound Standard Plane Recognition](semc_structure-enhanced_mixture-of-experts_contrastive_learning_for_ultrasound_s.md)**

:   提出 SEMC 框架，通过**语义-结构融合模块（SSFM）**对齐浅层结构线索与深层语义表征，结合**混合专家对比识别模块（MCRM）**在多层特征上进行分层对比学习，提升超声标准切面识别的细粒度判别能力，并构建了新的肝脏超声数据集 LP2025。

**[Sim4Seg: Boosting Multimodal Multi-disease Medical Diagnosis Segmentation with Region-Aware Vision-Language Similarity Masks](sim4seg_boosting_multimodal_multi-disease_medical_diagnosis_segmentation_with_re.md)**

:   提出医学诊断分割（MDS）任务并构建 M3DS 数据集，设计 Sim4Seg 框架利用 LVLM 隐藏状态的**视觉-语言相似度掩码（RVLS2M）**提示 SAM 进行分割，同时生成诊断思维链，配合测试时缩放策略在分割和诊断上全面超越基线。

**[Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)**

:   WEFT 提出了一种基于动态小波专家引导的轻量微调范式，仅需 4.52% 的可训练参数即可将大规模冻结视觉基础模型高效适配到光学遥感图像分割任务，在三个 ORSIs 数据集上超越 21 种 SOTA 方法。

**[SPA: Achieving Consensus in LLM Alignment via Self-Priority Optimization](spa_achieving_consensus_in_llm_alignment_via_self-priority_optimization.md)**

:   提出 Self-Priority Alignment（SPA），一种全无监督框架，通过字典序优化实现"可信赖优先于有用性"的严格优先级对齐——模型自生成多样响应、自评估、自改进，经双准则去噪构建偏好对，用不确定性加权 SimPO 损失微调，在多个安全基准上同时提升安全性和有用性。

**[TAlignDiff: Automatic Tooth Alignment assisted by Diffusion-based Transformation Learning](taligndiff_automatic_tooth_alignment_assisted_by_diffusion-based_transformation_.md)**

:   提出TAlignDiff框架，将基于点云的几何约束回归网络（PRN）与扩散模型辅助的变换矩阵去噪模块（DTMD）统一为一个联合训练框架，通过双向反馈机制在小样本临床数据上实现了优于现有方法的自动牙齿排列效果。

**[Towards Effective and Efficient Context-aware Nucleus Detection in Histopathology Whole Slide Images](towards_effective_and_efficient_context-aware_nucleus_detection_in_histopatholog.md)**

:   提出一种高效的上下文感知细胞核检测方法，通过聚合历史已访问滑窗的现成特征替代额外裁剪大视野图像块来提供组织上下文，同时利用跨标注策略挖掘周围未标注核样本以增强模型的上下文适应性。

**[Training-Free Policy Violation Detection via Activation-Space Whitening in LLMs](training-free_policy_violation_detection_via_activation-space_whitening_in_llms.md)**

:   将 LLM 的策略违规检测重构为激活空间中的分布外（OOD）检测问题，提出无需训练的白化方法：对合规激活拟合白化变换，用欧几里得范数作为合规分数，仅需策略文本和少量示例即可部署，在 DynaBench 上达到 86.0% F1，超越微调基线 9.1 个点、LLM-as-Judge 16 个点。

**[Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment](unleashing_the_potential_of_large_language_models_for_text-to-image_generation_t.md)**

:   提出 ARRA（Autoregressive Representation Alignment）训练框架，通过混合令牌 \<HYBNEXT\> 在训练时将外部视觉基础模型的全局表征蒸馏到自回归 LLM 的隐状态中，无需修改架构即可显著提升 LLM 的文本到图像生成质量。

**[Unsupervised Motion-Compensated Decomposition for Cardiac MRI Reconstruction via Neural Representation](unsupervised_motion-compensated_decomposition_for_cardiac_mri_reconstruction_via.md)**

:   提出 MoCo-INR，首次将隐式神经表示（INR）引入运动补偿（MoCo）框架，通过无监督方式实现心脏 MRI 的高质量动态重建，在超高加速因子（20x Cartesian / 69x Non-Cartesian）下显著优于现有无监督方法。

**[Unsupervised Multi-Parameter Inverse Solving for Reducing Ring Artifacts in 3D X-Ray CBCT](unsupervised_multi-parameter_inverse_solving_for_reducing_ring_artifacts_in_3d_x.md)**

:   提出 Riner，将 CT 环形伪影去除（RAR）建模为基于物理的多参数逆问题，通过隐式神经表示（INR）联合学习无伪影图像和探测器物理参数，实现无监督且优于有监督 SOTA 方法的 3D CBCT 重建。

**[Vascular Anatomy-aware Self-supervised Pre-training for X-ray Angiogram Analysis](vascular_anatomy-aware_self-supervised_pre-training_for_x-ray_angiogram_analysis.md)**

:   提出 VasoMIM，一个针对X射线血管造影的领域特定自监督预训练框架：通过解剖引导的掩码策略优先遮挡血管区域 + 解剖一致性损失保持重建图像的血管拓扑结构，结合构建的最大规模XA-170K预训练数据集，在4个下游任务6个数据集上全面超越通用SSL方法和医学SSL方法（包括在16.9亿图像上预训练的DINOv3）。

**[Virtual Multiplex Staining for Histological Images Using a Marker-wise Conditioned Diffusion Model](virtual_multiplex_staining_for_histological_images_using_a_marker-wise_condition.md)**

:   提出基于标记物条件扩散模型的虚拟多重染色框架，通过两阶段训练（标记物条件扩散学习+像素级微调），首次从单张H&E图像生成多达18种不同标记物的多重免疫荧光图像，在HEMIT和Orion-CRC两个公开数据集上全面超越现有方法。

**[VitalDiagnosis: AI-Driven Ecosystem for 24/7 Vital Monitoring and Chronic Disease Management](vitaldiagnosis_ai-driven_ecosystem_for_247_vital_monitoring_and_chronic_disease_.md)**

:   提出VitalDiagnosis，一个由LLM驱动的慢性病管理生态系统，通过整合可穿戴设备连续数据与多尺度LLM推理能力，建立包含异常交互式分诊和常规依从性监测的双轨框架，在协作式患者-临床医生工作流中实现从被动监测到主动参与的范式转变。

**[Voices, Faces, and Feelings: Multi-modal Emotion-Cognition Captioning for Mental Health Understanding](voices_faces_and_feelings_multi-modal_emotion-cognition_captioning_for_mental_he.md)**

:   提出情感-认知协同多模态描述（ECMC）任务和框架，通过双流BridgeNet从视频、音频、文本中提取情感和认知特征，利用LLaMA生成自然语言描述，为心理健康评估提供可解释的情感-认知画像，显著提升辅助诊断的准确性和可解释性。

**[WDT-MD: Wavelet Diffusion Transformers for Microaneurysm Detection in Fundus Images](wdt-md_wavelet_diffusion_transformers_for_microaneurysm_detection_in_fundus_imag.md)**

:   提出 WDT-MD 框架，通过噪声编码图像条件化、伪正常模式合成和小波扩散 Transformer 架构，解决眼底图像中微动脉瘤（MA）检测的三大难题：identity mapping、高假阳性和正常特征重建质量差。
