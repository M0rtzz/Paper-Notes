---
title: >-
  CVPR2025 医学图像方向 44篇论文解读
description: >-
  44篇CVPR2025 医学图像方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**📷 CVPR2025** · 共 **44** 篇

**[A Semi-Supervised Framework For Breast Ultrasound Segmentation With Training-Fre](a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)**

:   提出结合 VLM 无训练伪标签生成（外观描述 prompt 驱动 Grounding DINO + SAM）和双教师不确定性融合精炼的半监督乳腺超声分割框架，仅用 2.5% 标注数据即达到接近全监督的性能。

**[Aa-Clip Enhancing Zero-Shot Anomaly Detection Via Anomaly-Aware Clip](aa-clip_enhancing_zero-shot_anomaly_detection_via_anomaly-aware_clip.md)**

:   提出 AA-CLIP，通过两阶段训练策略（先适配文本编码器建立异常感知锚点，再对齐 patch 级视觉特征），在保留 CLIP 泛化能力的前提下增强其异常判别力，仅需极少训练样本即可在工业和医学多个数据集上达到 SOTA 零样本异常检测性能。

**[Accelerating Stroke Mri With Diffusion Probabilistic Models Through Large-Scale ](accelerating_stroke_mri_with_diffusion_probabilistic_models_through_large-scale_.md)**

:   借鉴基础模型范式，在大规模公开脑 MRI 数据上预训练扩散概率模型（DPM），再在仅 20 例中风患者数据上微调，实现数据受限场景下加速 MRI 重建，临床读者研究证实 2× 加速图像质量不劣于标准治疗。

**[Adaptation Of Weakly Supervised Localization In Histopathology By Debiasing Pred](adaptation_of_weakly_supervised_localization_in_histopathology_by_debiasing_pred.md)**

:   提出 SFDA-DeP 方法，受机器遗忘启发，通过识别并纠正源模型在目标域的预测偏差（over-predict 某些类别），解决组织病理学中弱监督定位模型跨器官/跨中心域适应时预测偏差被放大的问题。

**[Addressing Data Scarcity In 3D Trauma Detection Through Self-Supervised And Semi](addressing_data_scarcity_in_3d_trauma_detection_through_self-supervised_and_semi.md)**

:   提出两阶段标签高效学习框架：先在 1206 例无标注 CT 上用 Masked Image Modeling 自监督预训练 3D U-Net 编码器，再结合 VDETR + Vertex RPE 和 Mean Teacher 半监督学习，仅用 144 例标注数据实现腹部创伤 3D 检测 mAP@0.50 达 45.30%（+115%）。

**[Association Of Radiologic Ppfe Change With Mortality In Lung Cancer Screening Co](association_of_radiologic_ppfe_change_with_mortality_in_lung_cancer_screening_co.md)**

:   在两个大规模肺癌筛查队列（NLST 7980 例、SUMMIT 8561 例）中验证了基于深度学习自动量化的 PPFE（胸膜肺实质纤维弹性组织增生）进展与全因死亡率独立相关，提出 PPFE 纵向变化可作为筛查人群中识别高呼吸发病风险个体的影像生物标志物。

**[Automated Detection Of Malignant Lesions In The Ovary Using Deep Learning Models](automated_detection_of_malignant_lesions_in_the_ovary_using_deep_learning_models.md)**

:   使用 15 种 CNN 变体（LeNet、ResNet、VGG、Inception）在组织病理学图像上检测卵巢癌及亚型，选择 InceptionV3（ReLU）作为最优模型（平均 94.58%），并使用 LIME、SHAP、Integrated Gradients 三种 XAI 方法解释模型预测。

**[Biclip Bidirectional And Consistent Language-Image Processing For Robust Medical](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)**

:   BiCLIP 提出了一种双向一致性视觉-语言分割框架，通过双向多模态融合（BMF，让视觉特征反向精炼文本嵌入）和图像增强一致性（IAC，跨弱/强扰动正则化），在 COVID-19 CT 分割上以仅 1% 标注数据即可保持鲁棒性能，且对临床图像退化（噪声/模糊）具有容忍力。

**[Boltzmann Attention Sampling For Image Analysis With Small Objects](boltzmann_attention_sampling_for_image_analysis_with_small_objects.md)**

:   提出BoltzFormer——一种新型transformer decoder架构，通过玻尔兹曼分布动态采样稀疏注意力区域来聚焦小目标，结合退火温度调度（早期层探索、后期层利用）和PiGMA多query聚合模块，在占图像面积<0.1%的小目标分割上比SOTA提升3-12% Dice分数，同时减少一个数量级的注意力计算。

**[Cholectrack20 A Multi-Perspective Tracking Dataset For Surgical Tools](cholectrack20_a_multi-perspective_tracking_dataset_for_surgical_tools.md)**

**[Cloe Expert Consistency Learning For Missing Modality Segmentation](cloe_expert_consistency_learning_for_missing_modality_segmentation.md)**

:   提出 CLoE 框架，将缺失模态分割的鲁棒性问题重新定义为决策层专家一致性控制问题，通过全局模态专家一致性(MEC)和区域专家一致性(REC)双分支约束减少专家漂移，并用轻量门控网络将一致性分数转化为可靠性权重指导特征融合，在 BraTS 2020 和 MSD Prostate 上超越 SOTA。

**[Crosssdf 3D Reconstruction Of Thin Structures From Cross-Sections](crosssdf_3d_reconstruction_of_thin_structures_from_cross-sections.md)**

:   提出 CrossSDF，通过从 2D 截面符号距离场重建 3D SDF，结合混合编码（哈希网格 + 随机傅里叶特征）和对称差损失，首次实现对薄管状结构（如血管）的精确重建。

**[Cycleulm A Unified Label-Free Deep Learning Framework For Ultrasound Localisatio](cycleulm_a_unified_label-free_deep_learning_framework_for_ultrasound_localisatio.md)**

:   提出 CycleULM，首个统一的无标签深度学习超声定位显微(ULM)框架，通过 CycleGAN 学习 CEUS 帧到简化微泡域的物理仿真双向翻译来弥合仿真-真实域差距，实现微泡定位精度提升达40% recall、46% precision，并以18.3 fps 实现实时处理。

**[Decoding Matters Efficient Mamba-Based Decoder With Distribution-Aware Deep Supe](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)**

:   提出 Deco-Mamba，一种以解码器为核心的混合 Transformer-CNN-Mamba 架构，通过 Co-Attention Gate、Vision State Space Module 和可变形卷积精炼块增强解码器能力，并引入基于窗口化 KL 散度的分布感知深度监督策略，在 7 个医学图像分割基准上取得 SOTA 性能，同时保持适中的模型复杂度。

**[Deep Learning Based Estimation Of Blood Glucose Levels From Multidirectional Scl](deep_learning_based_estimation_of_blood_glucose_levels_from_multidirectional_scl.md)**

:   提出 ScleraGluNet，通过五方向巩膜血管图像结合多分支 CNN + MRFO 特征筛选 + Transformer 跨视图融合，实现三分类代谢状态判别（93.8% 准确率）和连续空腹血糖估计（MAE = 6.42 mg/dL），为无创血糖监测提供了新途径。

**[Developing Foundation Models For Universal Segmentation From 3D Whole-Body Posit](developing_foundation_models_for_universal_segmentation_from_3d_whole-body_posit.md)**

:   构建了最大规模 PET 分割数据集 PETWB-Seg11K（11,041 例全身 PET + 59,831 个分割掩码），并提出 SegAnyPET——基于 3D 架构 + prompt 工程的 PET 通用分割基础模型，在多中心、多示踪剂、多疾病场景下展现强零样本泛化能力。

**[Dflmoe Decentralized Federated Learning Via Mixture Of Experts For Medical Data ](dflmoe_decentralized_federated_learning_via_mixture_of_experts_for_medical_data_.md)**

:   提出 DFLMoE 在去中心化联邦学习中使用混合专家（MoE）机制处理医疗数据异质性，无需中心服务器即可在保护隐私的前提下协同训练

**[Diffusion-Based Feature Denoising And Using Nnmf For Robust Brain Tumor Classifi](diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)**

:   提出一种结合非负矩阵分解（NNMF）特征提取、统计特征筛选、轻量 CNN 分类和扩散式特征空间去噪的脑肿瘤分类框架，在保持 ~85% 干净准确率的同时，将 AutoAttack 下的鲁棒准确率从 0.47% 提升至 59.5%。

**[Equivania A Spectral Method For Rotation-Equivariant Anisotropic Image Analysis](equivania_a_spectral_method_for_rotation-equivariant_anisotropic_image_analysis.md)**

:   提出 EquivAnIA，一种基于 cake wavelet 和 ridge filter 的频谱方法，用于对图像进行旋转等变的各向异性分析，在合成和真实图像（含 CT）上展现出优于传统 angular binning 的旋转鲁棒性。

**[Evidential Learning Driven Breast Tumor Segmentation With Stage-Divided Vision-L](evidential_learning_driven_breast_tumor_segmentation_with_stage-divided_vision-l.md)**

:   提出 TextBCS 模型，通过阶段分割的视觉-语言交互模块（SVLI）和证据学习（EL）策略，利用文本提示辅助乳腺肿瘤分割，在 Duke-Breast-Cancer-MRI 数据集上 Dice 达 85.33%，超越所有对比方法。

**[Federated Modality-Specific Encoders And Partially Personalized Fusion Decoder F](federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)**

:   提出 FedMEPD 联邦学习框架，通过模态专属编码器（全局联邦）和部分个性化融合解码器，同时解决多模态 MRI 脑肿瘤分割中的模态间异质性和客户端个性化问题，在 BraTS 2018/2020 上客户端平均 mDSC 达 75.70%/75.90%。

**[Giim Graph-Based Learning Of Inter- And Intra-View Dependencies For Multi-View M](giim_graph-based_learning_of_inter-_and_intra-view_dependencies_for_multi-view_m.md)**

:   提出 GIIM，一种基于多异构图（MHG）的多视图医学图像分类框架，同时建模视图内（intra-view）和视图间（inter-view）的病灶依赖关系，在肝脏 CT、乳腺 X 线和乳腺 MRI 三种模态上均显著优于现有多视图方法，并对缺失视图具有鲁棒性。

**[Human Knowledge Integrated Multi-Modal Learning For Single Source Domain General](human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)**

:   提出 GenEval，通过域共形界（DCB）理论量化因果覆盖差距，并将人类专家知识与 MedGemma-4B 视觉语言模型结合，实现单源域泛化（SDG），在糖尿病视网膜病变分级（8 个数据集）和癫痫灶检测（2 个数据集）上大幅超越现有方法。

**[Mil-Pf Multiple Instance Learning On Precomputed Features For Mammography Classi](mil-pf_multiple_instance_learning_on_precomputed_features_for_mammography_classi.md)**

:   提出 MIL-PF 框架，利用冻结的基础视觉模型预计算特征，配合仅 ~40k 参数的轻量 MIL 聚合头，在乳腺 X 光分类任务上达到 SOTA 性能，大幅降低训练成本。

**[Multimodal Classification Of Radiation-Induced Contrast Enhancements And Tumor R](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)**

:   提出 RICE-NET，一个多模态 3D 深度学习模型，融合纵向 MRI 数据与放疗剂量分布图，用于区分胶质母细胞瘤术后放射性对比增强（RICE）与肿瘤复发，在独立测试集上达到 F1=0.92。

**[Multimodal Protein Language Models For Enzyme Kinetic Parameters From Substrate ](multimodal_protein_language_models_for_enzyme_kinetic_parameters_from_substrate_.md)**

:   提出 ERBA 适配器，将酶动力学预测建模为"底物识别→构象适应"的分阶段条件化过程，通过 MRCA 注入底物语义、G-MoE 融合活性位点3D几何、ESDA 保持 PLM 先验，在 kcat/Km/Ki 三个动力学端点上一致超越现有方法。

**[Multiscale Structure-Guided Latent Diffusion For Multimodal Mri Translation](multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)**

:   提出 MSG-LDM 框架，在潜在空间中显式解耦风格与结构信息，通过高频注入块 (HFIB)、多模态结构特征融合 (MMSF) 和多尺度结构增强 (MSSE) 提取模态不变的多尺度结构先验来引导扩散过程，解决任意模态缺失下 MRI 翻译的解剖不一致和纹理退化问题。

**[Noir Neural Operator Mapping For Implicit Representations](noir_neural_operator_mapping_for_implicit_representations.md)**

:   NOIR 将医学图像计算任务重新建模为连续函数空间之间的算子学习问题，通过隐式神经表示(INR)将离散医学信号嵌入连续函数空间，再用神经算子(NO)学习函数间的映射，实现分辨率无关的分割、形状补全、图像翻译和合成。

**[Novel Architecture Of Rpa In Oral Cancer Lesion Detection](novel_architecture_of_rpa_in_oral_cancer_lesion_detection.md)**

:   本文将 Singleton 和 Batch Processing 设计模式集成到基于 Python 的 RPA 自动化管道中，结合 EfficientNetV2B1 模型实现口腔癌病灶检测，相比 UiPath/Automation Anywhere 等传统 RPA 平台实现 60-100× 的推理加速。

**[Nyxus A Next Generation Image Feature Extraction Library For The Big Data And Ai](nyxus_a_next_generation_image_feature_extraction_library_for_the_big_data_and_ai.md)**

:   Nyxus 是一个面向大数据和 AI 时代的下一代图像特征提取库，支持 2D/3D 数据的 out-of-core 可扩展提取，覆盖 radiomics 和细胞分析两大领域共 261+ 特征，在速度上比 CellProfiler 快 3–131×、比 PyRadiomics/MITK 快数倍至数百倍。

**[Paper Title Lov3D Grounding Cognitive Prognosis Reasoning In Longitudinal 3D Bra](paper_title_lov3d_grounding_cognitive_prognosis_reasoning_in_longitudinal_3d_bra.md)**

:   LoV3D 提出一套端到端纵向 3D 脑 MRI 视觉-语言模型管线，通过结构化可验证输出设计实现解剖区域评估 + 纵向对比 + 三分类诊断推理，并利用临床加权 Verifier 驱动 DPO 训练（无需人工标注），在 ADNI 上达到 93.7% 三分类准确率且零非相邻诊断错误。

**[Prototype-Based Knowledge Guidance For Fine-Grained Structured Radiology Reporti](prototype-based_knowledge_guidance_for_fine-grained_structured_radiology_reporti.md)**

:   ProtoSR 提出从大规模自由文本放射学报告中挖掘模板对齐的原型知识库，并通过原型条件化的后期融合残差模块注入结构化报告预测，在 Rad-ReStruct 基准上实现 SOTA，尤其在细粒度属性问题 (L3) 上获得 72.1% 的相对提升。

**[Reinforcing The Weakest Links Modernizing Siena With Targeted Deep Learning Inte](reinforcing_the_weakest_links_modernizing_siena_with_targeted_deep_learning_inte.md)**

:   将深度学习模块（SynthStrip/SynthSeg）模块化替换 SIENA 管线中的经典颅骨剥离和组织分割步骤，在保留管线可解释性的前提下显著提升纵向脑萎缩（PBVC）估计的临床敏感性和鲁棒性。在 ADNI 和 PPMI 两个纵向队列上验证。

**[Residual Sodap Residual Self-Organizing Domain-Adaptive Prompting With Structura](residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)**

:   针对无任务 ID 和无数据回放的领域增量学习（DIL），提出 Residual SODAP 框架，通过 α-entmax 稀疏 prompt 选择与残差聚合、基于特征统计的伪回放蓏馏、prompt 使用模式漂移检测和不确定性加权，同时解决表示适配和分类器遗忘问题。在 DR、皮肤癌和 CORe50 上均达 SOTA。

**[Salient Frequency-Aware Paired Diffusion For Controllable Long-Tail Ct Detection](salient_frequency-aware_paired_diffusion_for_controllable_long-tail_ct_detection.md)**

:   提出 SALIENT，一个基于小波域扩散的掩码条件生成框架，通过频率感知的可解释优化目标和配对的病灶-掩码体积生成，实现长尾 CT 检测中可控、高效的合成数据增强与精度拯救。首次系统表征增强剂量-反应曲线。

**[Saw Toward A Surgical Action World Model Via Controllable And Scalable Video Gen](saw_toward_a_surgical_action_world_model_via_controllable_and_scalable_video_gen.md)**

:   提出 SAW（Surgical Action World），通过四种轻量级条件信号（语言提示、参考帧、组织功能图、工具轨迹）驱动视频扩散模型，实现可控、可扩展的手术动作视频生成，用于罕见动作增强和手术仿真。

**[Semantic Class Distribution Learning For Debiasing Semi-Supervised Medical Image](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)**

:   提出 SCDL 即插即用模块，通过学习类条件代理分布并进行双向对齐（CDBA）+ 语义锚约束（SAC），在嵌入空间显式重塑类条件特征结构，缓解半监督医学影像分割中的监督偏差和表示不平衡。

**[Semitooth A Generalizable Semi-Supervised Framework For Multi-Source Tooth Segme](semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)**

:   提出 SemiTooth 多教师多学生半监督框架，通过 Stricter Weighted-Confidence Constraint 实现多源 CBCT 牙齿分割的跨域泛化。

**[Transformer-Based Multi-Region Segmentation And Radiomic Analysis Of Hr-Pqct Ima](transformer-based_multi-region_segmentation_and_radiomic_analysis_of_hr-pqct_ima.md)**

:   首次将 SegFormer 用于 HR-pQCT 影像的多区域（骨+软组织）自动分割与放射组学分析，发现肌腱组织特征在骨质疏松分类中优于传统骨指标。

**[Ultrasoundagents Hierarchical Multi-Agent Evidence-Chain Reasoning For Breast Ul](ultrasoundagents_hierarchical_multi-agent_evidence-chain_reasoning_for_breast_ul.md)**

:   提出 UltrasoundAgents 层次化多智能体框架，通过主智能体定位病灶+子智能体识别属性+证据链推理的流程，对齐乳腺超声临床诊断工作流并实现可追溯的 BI-RADS 分级与良恶性判断。

**[Uncertainty-Aware Concept And Motion Segmentation For Semi-Supervised Angiograph](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)**

:   提出 SMART 框架，基于 SAM3 的教师-学生结构结合文本概念提示、置信度感知一致性正则化和双流时序一致性，实现 X 光冠脉造影视频的半监督血管分割。

**[Unistainnet Foundation-Model-Guided Virtual Staining Of He To Ihc](unistainnet_foundation-model-guided_virtual_staining_of_he_to_ihc.md)**

:   提出 UNIStainNet，首次将冻结病理基础模型 UNI 的稠密空间 token 作为生成器的直接条件信号，实现 H&E 到 IHC 的虚拟染色，单一统一模型同时服务四种 IHC 标记物并达到 SOTA。

**[Unleashing Video Language Models For Fine-Grained Hrct Report Generation](unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)**

:   提出 AbSteering 框架，通过异常中心化 CoT 训练和基于临床混淆异常硬负例的 DPO 优化，将通用视频语言模型（VideoLMs）高效迁移到 HRCT 报告生成任务，性能超越专用 CT 基础模型。

**[Unmasking Biases And Reliability Concerns In Convolutional Neural Networks Analy](unmasking_biases_and_reliability_concerns_in_convolutional_neural_networks_analy.md)**

:   通过从 13 个癌症病理基准数据集的图像背景区域（不含临床信息的 20×20 像素裁剪）训练 CNN，发现分类准确率远高于随机猜测（最高 93%），揭示了 CNN 可能依赖数据集偏差而非真正的病理特征来做出诊断判断。
