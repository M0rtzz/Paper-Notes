---
title: >-
  CVPR2025 355篇论文解读
description: >-
  355篇CVPR2025论文深度解读，每篇5分钟读懂核心思想。覆盖医学图像、3D视觉、图像生成、自动驾驶、多模态VLM、语义分割等32个研究领域，每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📷 CVPR2025 论文笔记

共 **355** 篇笔记，覆盖 **32** 个领域。

## 领域概览

| 领域 | 篇数 |
|:-----|-----:|
| 🏥 [医学图像](#medical_imaging) | 44 |
| 🧊 [3D视觉](#3d_vision) | 36 |
| 🎨 [图像生成](#image_generation) | 30 |
| 🚗 [自动驾驶](#autonomous_driving) | 24 |
| 🧩 [多模态VLM](#multimodal_vlm) | 23 |
| ✂️ [语义分割](#segmentation) | 22 |
| 🎬 [视频理解](#video_understanding) | 16 |
| 💬 [LLM/NLP](#llm_nlp) | 14 |
| 🤖 [机器人/具身智能](#robotics) | 14 |
| 🧑 [人体理解](#human_understanding) | 13 |
| 🖼️ [图像恢复](#image_restoration) | 11 |
| 📦 [模型压缩](#model_compression) | 11 |
| 🦾 [LLM Agent](#llm_agent) | 10 |
| 🛡️ [AI安全](#ai_safety) | 9 |
| 🎯 [目标检测](#object_detection) | 8 |
| 🔄 [自监督/表示学习](#self_supervised) | 8 |
| ⚖️ [对齐/RLHF](#llm_alignment) | 7 |
| 📚 [预训练/数据](#llm_pretraining) | 5 |
| 🛰️ [遥感](#remote_sensing) | 5 |
| 🎵 [音频/语音](#audio_speech) | 4 |
| 🔬 [可解释性](#interpretability) | 4 |
| ⚡ [LLM效率](#llm_efficiency) | 4 |
| 💡 [LLM推理](#llm_reasoning) | 3 |
| ✍️ [文本生成](#nlp_generation) | 3 |
| 👥 [社会计算](#social_computing) | 3 |
| 🔍 [信息检索/RAG](#information_retrieval) | 2 |
| 📊 [LLM评测](#llm_evaluation) | 2 |
| 📐 [优化/理论](#optimization) | 2 |
| 💻 [代码智能](#code_intelligence) | 1 |
| 🎮 [强化学习](#reinforcement_learning) | 1 |
| 📈 [时间序列](#time_series) | 1 |
| 📂 [其他](#others) | 15 |

---

## 🏥 医学图像 { #medical_imaging }

**[A Semi-Supervised Framework For Breast Ultrasound Segmentation With Training-Fre](medical_imaging/a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)**

:   提出结合 VLM 无训练伪标签生成（外观描述 prompt 驱动 Grounding DINO + SAM）和双教师不确定性融合精炼的半监督乳腺超声分割框架，仅用 2.5% 标注数据即达到接近全监督的性能。

**[Aa-Clip Enhancing Zero-Shot Anomaly Detection Via Anomaly-Aware Clip](medical_imaging/aa-clip_enhancing_zero-shot_anomaly_detection_via_anomaly-aware_clip.md)**

:   提出 AA-CLIP，通过两阶段训练策略（先适配文本编码器建立异常感知锚点，再对齐 patch 级视觉特征），在保留 CLIP 泛化能力的前提下增强其异常判别力，仅需极少训练样本即可在工业和医学多个数据集上达到 SOTA 零样本异常检测性能。

**[Accelerating Stroke Mri With Diffusion Probabilistic Models Through Large-Scale ](medical_imaging/accelerating_stroke_mri_with_diffusion_probabilistic_models_through_large-scale_.md)**

:   借鉴基础模型范式，在大规模公开脑 MRI 数据上预训练扩散概率模型（DPM），再在仅 20 例中风患者数据上微调，实现数据受限场景下加速 MRI 重建，临床读者研究证实 2× 加速图像质量不劣于标准治疗。

**[Adaptation Of Weakly Supervised Localization In Histopathology By Debiasing Pred](medical_imaging/adaptation_of_weakly_supervised_localization_in_histopathology_by_debiasing_pred.md)**

:   提出 SFDA-DeP 方法，受机器遗忘启发，通过识别并纠正源模型在目标域的预测偏差（over-predict 某些类别），解决组织病理学中弱监督定位模型跨器官/跨中心域适应时预测偏差被放大的问题。

**[Addressing Data Scarcity In 3D Trauma Detection Through Self-Supervised And Semi](medical_imaging/addressing_data_scarcity_in_3d_trauma_detection_through_self-supervised_and_semi.md)**

:   提出两阶段标签高效学习框架：先在 1206 例无标注 CT 上用 Masked Image Modeling 自监督预训练 3D U-Net 编码器，再结合 VDETR + Vertex RPE 和 Mean Teacher 半监督学习，仅用 144 例标注数据实现腹部创伤 3D 检测 mAP@0.50 达 45.30%（+115%）。

**[Association Of Radiologic Ppfe Change With Mortality In Lung Cancer Screening Co](medical_imaging/association_of_radiologic_ppfe_change_with_mortality_in_lung_cancer_screening_co.md)**

:   在两个大规模肺癌筛查队列（NLST 7980 例、SUMMIT 8561 例）中验证了基于深度学习自动量化的 PPFE（胸膜肺实质纤维弹性组织增生）进展与全因死亡率独立相关，提出 PPFE 纵向变化可作为筛查人群中识别高呼吸发病风险个体的影像生物标志物。

**[Automated Detection Of Malignant Lesions In The Ovary Using Deep Learning Models](medical_imaging/automated_detection_of_malignant_lesions_in_the_ovary_using_deep_learning_models.md)**

:   使用 15 种 CNN 变体（LeNet、ResNet、VGG、Inception）在组织病理学图像上检测卵巢癌及亚型，选择 InceptionV3（ReLU）作为最优模型（平均 94.58%），并使用 LIME、SHAP、Integrated Gradients 三种 XAI 方法解释模型预测。

**[Biclip Bidirectional And Consistent Language-Image Processing For Robust Medical](medical_imaging/biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)**

:   BiCLIP 提出了一种双向一致性视觉-语言分割框架，通过双向多模态融合（BMF，让视觉特征反向精炼文本嵌入）和图像增强一致性（IAC，跨弱/强扰动正则化），在 COVID-19 CT 分割上以仅 1% 标注数据即可保持鲁棒性能，且对临床图像退化（噪声/模糊）具有容忍力。

**[Boltzmann Attention Sampling For Image Analysis With Small Objects](medical_imaging/boltzmann_attention_sampling_for_image_analysis_with_small_objects.md)**

:   提出BoltzFormer——一种新型transformer decoder架构，通过玻尔兹曼分布动态采样稀疏注意力区域来聚焦小目标，结合退火温度调度（早期层探索、后期层利用）和PiGMA多query聚合模块，在占图像面积<0.1%的小目标分割上比SOTA提升3-12% Dice分数，同时减少一个数量级的注意力计算。

**[Cholectrack20 A Multi-Perspective Tracking Dataset For Surgical Tools](medical_imaging/cholectrack20_a_multi-perspective_tracking_dataset_for_surgical_tools.md)**

**[Cloe Expert Consistency Learning For Missing Modality Segmentation](medical_imaging/cloe_expert_consistency_learning_for_missing_modality_segmentation.md)**

:   提出 CLoE 框架，将缺失模态分割的鲁棒性问题重新定义为决策层专家一致性控制问题，通过全局模态专家一致性(MEC)和区域专家一致性(REC)双分支约束减少专家漂移，并用轻量门控网络将一致性分数转化为可靠性权重指导特征融合，在 BraTS 2020 和 MSD Prostate 上超越 SOTA。

**[Crosssdf 3D Reconstruction Of Thin Structures From Cross-Sections](medical_imaging/crosssdf_3d_reconstruction_of_thin_structures_from_cross-sections.md)**

:   提出 CrossSDF，通过从 2D 截面符号距离场重建 3D SDF，结合混合编码（哈希网格 + 随机傅里叶特征）和对称差损失，首次实现对薄管状结构（如血管）的精确重建。

**[Cycleulm A Unified Label-Free Deep Learning Framework For Ultrasound Localisatio](medical_imaging/cycleulm_a_unified_label-free_deep_learning_framework_for_ultrasound_localisatio.md)**

:   提出 CycleULM，首个统一的无标签深度学习超声定位显微(ULM)框架，通过 CycleGAN 学习 CEUS 帧到简化微泡域的物理仿真双向翻译来弥合仿真-真实域差距，实现微泡定位精度提升达40% recall、46% precision，并以18.3 fps 实现实时处理。

**[Decoding Matters Efficient Mamba-Based Decoder With Distribution-Aware Deep Supe](medical_imaging/decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)**

:   提出 Deco-Mamba，一种以解码器为核心的混合 Transformer-CNN-Mamba 架构，通过 Co-Attention Gate、Vision State Space Module 和可变形卷积精炼块增强解码器能力，并引入基于窗口化 KL 散度的分布感知深度监督策略，在 7 个医学图像分割基准上取得 SOTA 性能，同时保持适中的模型复杂度。

**[Deep Learning Based Estimation Of Blood Glucose Levels From Multidirectional Scl](medical_imaging/deep_learning_based_estimation_of_blood_glucose_levels_from_multidirectional_scl.md)**

:   提出 ScleraGluNet，通过五方向巩膜血管图像结合多分支 CNN + MRFO 特征筛选 + Transformer 跨视图融合，实现三分类代谢状态判别（93.8% 准确率）和连续空腹血糖估计（MAE = 6.42 mg/dL），为无创血糖监测提供了新途径。

**[Developing Foundation Models For Universal Segmentation From 3D Whole-Body Posit](medical_imaging/developing_foundation_models_for_universal_segmentation_from_3d_whole-body_posit.md)**

:   构建了最大规模 PET 分割数据集 PETWB-Seg11K（11,041 例全身 PET + 59,831 个分割掩码），并提出 SegAnyPET——基于 3D 架构 + prompt 工程的 PET 通用分割基础模型，在多中心、多示踪剂、多疾病场景下展现强零样本泛化能力。

**[Dflmoe Decentralized Federated Learning Via Mixture Of Experts For Medical Data ](medical_imaging/dflmoe_decentralized_federated_learning_via_mixture_of_experts_for_medical_data_.md)**

:   提出 DFLMoE 在去中心化联邦学习中使用混合专家（MoE）机制处理医疗数据异质性，无需中心服务器即可在保护隐私的前提下协同训练

**[Diffusion-Based Feature Denoising And Using Nnmf For Robust Brain Tumor Classifi](medical_imaging/diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)**

:   提出一种结合非负矩阵分解（NNMF）特征提取、统计特征筛选、轻量 CNN 分类和扩散式特征空间去噪的脑肿瘤分类框架，在保持 ~85% 干净准确率的同时，将 AutoAttack 下的鲁棒准确率从 0.47% 提升至 59.5%。

**[Equivania A Spectral Method For Rotation-Equivariant Anisotropic Image Analysis](medical_imaging/equivania_a_spectral_method_for_rotation-equivariant_anisotropic_image_analysis.md)**

:   提出 EquivAnIA，一种基于 cake wavelet 和 ridge filter 的频谱方法，用于对图像进行旋转等变的各向异性分析，在合成和真实图像（含 CT）上展现出优于传统 angular binning 的旋转鲁棒性。

**[Evidential Learning Driven Breast Tumor Segmentation With Stage-Divided Vision-L](medical_imaging/evidential_learning_driven_breast_tumor_segmentation_with_stage-divided_vision-l.md)**

:   提出 TextBCS 模型，通过阶段分割的视觉-语言交互模块（SVLI）和证据学习（EL）策略，利用文本提示辅助乳腺肿瘤分割，在 Duke-Breast-Cancer-MRI 数据集上 Dice 达 85.33%，超越所有对比方法。

**[Federated Modality-Specific Encoders And Partially Personalized Fusion Decoder F](medical_imaging/federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)**

:   提出 FedMEPD 联邦学习框架，通过模态专属编码器（全局联邦）和部分个性化融合解码器，同时解决多模态 MRI 脑肿瘤分割中的模态间异质性和客户端个性化问题，在 BraTS 2018/2020 上客户端平均 mDSC 达 75.70%/75.90%。

**[Giim Graph-Based Learning Of Inter- And Intra-View Dependencies For Multi-View M](medical_imaging/giim_graph-based_learning_of_inter-_and_intra-view_dependencies_for_multi-view_m.md)**

:   提出 GIIM，一种基于多异构图（MHG）的多视图医学图像分类框架，同时建模视图内（intra-view）和视图间（inter-view）的病灶依赖关系，在肝脏 CT、乳腺 X 线和乳腺 MRI 三种模态上均显著优于现有多视图方法，并对缺失视图具有鲁棒性。

**[Human Knowledge Integrated Multi-Modal Learning For Single Source Domain General](medical_imaging/human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)**

:   提出 GenEval，通过域共形界（DCB）理论量化因果覆盖差距，并将人类专家知识与 MedGemma-4B 视觉语言模型结合，实现单源域泛化（SDG），在糖尿病视网膜病变分级（8 个数据集）和癫痫灶检测（2 个数据集）上大幅超越现有方法。

**[Mil-Pf Multiple Instance Learning On Precomputed Features For Mammography Classi](medical_imaging/mil-pf_multiple_instance_learning_on_precomputed_features_for_mammography_classi.md)**

:   提出 MIL-PF 框架，利用冻结的基础视觉模型预计算特征，配合仅 ~40k 参数的轻量 MIL 聚合头，在乳腺 X 光分类任务上达到 SOTA 性能，大幅降低训练成本。

**[Multimodal Classification Of Radiation-Induced Contrast Enhancements And Tumor R](medical_imaging/multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)**

:   提出 RICE-NET，一个多模态 3D 深度学习模型，融合纵向 MRI 数据与放疗剂量分布图，用于区分胶质母细胞瘤术后放射性对比增强（RICE）与肿瘤复发，在独立测试集上达到 F1=0.92。

**[Multimodal Protein Language Models For Enzyme Kinetic Parameters From Substrate ](medical_imaging/multimodal_protein_language_models_for_enzyme_kinetic_parameters_from_substrate_.md)**

:   提出 ERBA 适配器，将酶动力学预测建模为"底物识别→构象适应"的分阶段条件化过程，通过 MRCA 注入底物语义、G-MoE 融合活性位点3D几何、ESDA 保持 PLM 先验，在 kcat/Km/Ki 三个动力学端点上一致超越现有方法。

**[Multiscale Structure-Guided Latent Diffusion For Multimodal Mri Translation](medical_imaging/multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)**

:   提出 MSG-LDM 框架，在潜在空间中显式解耦风格与结构信息，通过高频注入块 (HFIB)、多模态结构特征融合 (MMSF) 和多尺度结构增强 (MSSE) 提取模态不变的多尺度结构先验来引导扩散过程，解决任意模态缺失下 MRI 翻译的解剖不一致和纹理退化问题。

**[Noir Neural Operator Mapping For Implicit Representations](medical_imaging/noir_neural_operator_mapping_for_implicit_representations.md)**

:   NOIR 将医学图像计算任务重新建模为连续函数空间之间的算子学习问题，通过隐式神经表示(INR)将离散医学信号嵌入连续函数空间，再用神经算子(NO)学习函数间的映射，实现分辨率无关的分割、形状补全、图像翻译和合成。

**[Novel Architecture Of Rpa In Oral Cancer Lesion Detection](medical_imaging/novel_architecture_of_rpa_in_oral_cancer_lesion_detection.md)**

:   本文将 Singleton 和 Batch Processing 设计模式集成到基于 Python 的 RPA 自动化管道中，结合 EfficientNetV2B1 模型实现口腔癌病灶检测，相比 UiPath/Automation Anywhere 等传统 RPA 平台实现 60-100× 的推理加速。

**[Nyxus A Next Generation Image Feature Extraction Library For The Big Data And Ai](medical_imaging/nyxus_a_next_generation_image_feature_extraction_library_for_the_big_data_and_ai.md)**

:   Nyxus 是一个面向大数据和 AI 时代的下一代图像特征提取库，支持 2D/3D 数据的 out-of-core 可扩展提取，覆盖 radiomics 和细胞分析两大领域共 261+ 特征，在速度上比 CellProfiler 快 3–131×、比 PyRadiomics/MITK 快数倍至数百倍。

**[Paper Title Lov3D Grounding Cognitive Prognosis Reasoning In Longitudinal 3D Bra](medical_imaging/paper_title_lov3d_grounding_cognitive_prognosis_reasoning_in_longitudinal_3d_bra.md)**

:   LoV3D 提出一套端到端纵向 3D 脑 MRI 视觉-语言模型管线，通过结构化可验证输出设计实现解剖区域评估 + 纵向对比 + 三分类诊断推理，并利用临床加权 Verifier 驱动 DPO 训练（无需人工标注），在 ADNI 上达到 93.7% 三分类准确率且零非相邻诊断错误。

**[Prototype-Based Knowledge Guidance For Fine-Grained Structured Radiology Reporti](medical_imaging/prototype-based_knowledge_guidance_for_fine-grained_structured_radiology_reporti.md)**

:   ProtoSR 提出从大规模自由文本放射学报告中挖掘模板对齐的原型知识库，并通过原型条件化的后期融合残差模块注入结构化报告预测，在 Rad-ReStruct 基准上实现 SOTA，尤其在细粒度属性问题 (L3) 上获得 72.1% 的相对提升。

**[Reinforcing The Weakest Links Modernizing Siena With Targeted Deep Learning Inte](medical_imaging/reinforcing_the_weakest_links_modernizing_siena_with_targeted_deep_learning_inte.md)**

:   将深度学习模块（SynthStrip/SynthSeg）模块化替换 SIENA 管线中的经典颅骨剥离和组织分割步骤，在保留管线可解释性的前提下显著提升纵向脑萎缩（PBVC）估计的临床敏感性和鲁棒性。在 ADNI 和 PPMI 两个纵向队列上验证。

**[Residual Sodap Residual Self-Organizing Domain-Adaptive Prompting With Structura](medical_imaging/residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)**

:   针对无任务 ID 和无数据回放的领域增量学习（DIL），提出 Residual SODAP 框架，通过 α-entmax 稀疏 prompt 选择与残差聚合、基于特征统计的伪回放蓏馏、prompt 使用模式漂移检测和不确定性加权，同时解决表示适配和分类器遗忘问题。在 DR、皮肤癌和 CORe50 上均达 SOTA。

**[Salient Frequency-Aware Paired Diffusion For Controllable Long-Tail Ct Detection](medical_imaging/salient_frequency-aware_paired_diffusion_for_controllable_long-tail_ct_detection.md)**

:   提出 SALIENT，一个基于小波域扩散的掩码条件生成框架，通过频率感知的可解释优化目标和配对的病灶-掩码体积生成，实现长尾 CT 检测中可控、高效的合成数据增强与精度拯救。首次系统表征增强剂量-反应曲线。

**[Saw Toward A Surgical Action World Model Via Controllable And Scalable Video Gen](medical_imaging/saw_toward_a_surgical_action_world_model_via_controllable_and_scalable_video_gen.md)**

:   提出 SAW（Surgical Action World），通过四种轻量级条件信号（语言提示、参考帧、组织功能图、工具轨迹）驱动视频扩散模型，实现可控、可扩展的手术动作视频生成，用于罕见动作增强和手术仿真。

**[Semantic Class Distribution Learning For Debiasing Semi-Supervised Medical Image](medical_imaging/semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)**

:   提出 SCDL 即插即用模块，通过学习类条件代理分布并进行双向对齐（CDBA）+ 语义锚约束（SAC），在嵌入空间显式重塑类条件特征结构，缓解半监督医学影像分割中的监督偏差和表示不平衡。

**[Semitooth A Generalizable Semi-Supervised Framework For Multi-Source Tooth Segme](medical_imaging/semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)**

:   提出 SemiTooth 多教师多学生半监督框架，通过 Stricter Weighted-Confidence Constraint 实现多源 CBCT 牙齿分割的跨域泛化。

**[Transformer-Based Multi-Region Segmentation And Radiomic Analysis Of Hr-Pqct Ima](medical_imaging/transformer-based_multi-region_segmentation_and_radiomic_analysis_of_hr-pqct_ima.md)**

:   首次将 SegFormer 用于 HR-pQCT 影像的多区域（骨+软组织）自动分割与放射组学分析，发现肌腱组织特征在骨质疏松分类中优于传统骨指标。

**[Ultrasoundagents Hierarchical Multi-Agent Evidence-Chain Reasoning For Breast Ul](medical_imaging/ultrasoundagents_hierarchical_multi-agent_evidence-chain_reasoning_for_breast_ul.md)**

:   提出 UltrasoundAgents 层次化多智能体框架，通过主智能体定位病灶+子智能体识别属性+证据链推理的流程，对齐乳腺超声临床诊断工作流并实现可追溯的 BI-RADS 分级与良恶性判断。

**[Uncertainty-Aware Concept And Motion Segmentation For Semi-Supervised Angiograph](medical_imaging/uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)**

:   提出 SMART 框架，基于 SAM3 的教师-学生结构结合文本概念提示、置信度感知一致性正则化和双流时序一致性，实现 X 光冠脉造影视频的半监督血管分割。

**[Unistainnet Foundation-Model-Guided Virtual Staining Of He To Ihc](medical_imaging/unistainnet_foundation-model-guided_virtual_staining_of_he_to_ihc.md)**

:   提出 UNIStainNet，首次将冻结病理基础模型 UNI 的稠密空间 token 作为生成器的直接条件信号，实现 H&E 到 IHC 的虚拟染色，单一统一模型同时服务四种 IHC 标记物并达到 SOTA。

**[Unleashing Video Language Models For Fine-Grained Hrct Report Generation](medical_imaging/unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)**

:   提出 AbSteering 框架，通过异常中心化 CoT 训练和基于临床混淆异常硬负例的 DPO 优化，将通用视频语言模型（VideoLMs）高效迁移到 HRCT 报告生成任务，性能超越专用 CT 基础模型。

**[Unmasking Biases And Reliability Concerns In Convolutional Neural Networks Analy](medical_imaging/unmasking_biases_and_reliability_concerns_in_convolutional_neural_networks_analy.md)**

:   通过从 13 个癌症病理基准数据集的图像背景区域（不含临床信息的 20×20 像素裁剪）训练 CNN，发现分类准确率远高于随机猜测（最高 93%），揭示了 CNN 可能依赖数据集偏差而非真正的病理特征来做出诊断判断。

---

## 🧊 3D视觉 { #3d_vision }

**[3D-Grand A Million-Scale Dataset For 3D-Llms With Better Grounding And Less Hall](3d_vision/3d-grand_a_million-scale_dataset_for_3d-llms_with_better_grounding_and_less_hall.md)**

:   构建了3D-GRAND——首个百万级**密集接地**的3D场景-语言数据集（40K场景、6.2M指令），并提出3D-POPE幻觉评估基准，证明密集接地的指令微调能显著提升3D-LLM的接地能力并减少幻觉，还展示了合成数据到真实场景的迁移效果。

**[3D-Gsw 3D Gaussian Splatting For Robust Watermarking](3d_vision/3d-gsw_3d_gaussian_splatting_for_robust_watermarking.md)**

:   提出3D-GSW，首个专为3D Gaussian Splatting设计的鲁棒数字水印方法，通过频率引导致密化（FGD）移除冗余高斯并在高频区域分裂高斯来增强鲁棒性，结合梯度掩码和小波子带损失保持渲染质量，在Blender/LLFF/Mip-NeRF 360数据集上同时实现了最优的水印鲁棒性和渲染质量。

**[3D-Hgs 3D Half-Gaussian Splatting](3d_vision/3d-hgs_3d_half-gaussian_splatting.md)**

:   提出3D Half-Gaussian (3D-HGS)核函数——用一个分割平面将3D高斯分成两半，每半有独立不透明度，作为**即插即用**的重建核替换标准高斯核，在不牺牲渲染速度的前提下显著提升形状和颜色不连续处的渲染质量，在Mip-NeRF360/T&T/Deep Blending上全面超越所有SOTA方法。

**[3D-Llava Towards Generalist 3D Lmms With Omni Superpoint Transformer](3d_vision/3d-llava_towards_generalist_3d_lmms_with_omni_superpoint_transformer.md)**

:   提出3D-LLaVA，一个极简架构的通用3D大语言多模态模型，核心是**Omni Superpoint Transformer (OST)**作为多功能视觉连接器，同时充当视觉特征选择器、视觉提示编码器和分割掩码解码器，仅用点云输入就在ScanQA（92.6 CiDEr）、ScanRefer（43.3 mIoU）等5个基准上全面达到SOTA。

**[3D-Mem 3D Scene Memory For Embodied Exploration And Reasoning](3d_vision/3d-mem_3d_scene_memory_for_embodied_exploration_and_reasoning.md)**

:   提出3D-Mem——基于"记忆快照"的3D场景记忆框架，用少量精选多视角图像紧凑表示已探索区域，结合Frontier Snapshot表示未探索区域，配合VLM实现高效的具身探索与推理。

**[3D-Slnr A Super Lightweight Neural Representation For Large-Scale 3D Mapping](3d_vision/3d-slnr_a_super_lightweight_neural_representation_for_large-scale_3d_mapping.md)**

:   提出3D-SLNR，一种超轻量神经3D表示——基于锚定在点云支撑点上的带限局部SDF集合定义全局SDF，每个局部SDF仅由一个微型MLP参数化（无隐特征），通过可学习的位置/旋转/缩放适应复杂几何，配合并行查找算法和剪枝-扩展策略，以不到先前方法1/5的内存实现SOTA重建质量。

**[3D Convex Splatting Radiance Field Rendering With 3D Smooth Convexes](3d_vision/3d_convex_splatting_radiance_field_rendering_with_3d_smooth_convexes.md)**

:   用3D光滑凸体（Smooth Convex）替代高斯基元进行辐射场渲染，通过点集定义凸包+LogSumExp平滑化+自定义CUDA光栅化器，在T&T和Deep Blending上超越3DGS，且所需基元更少。

**[3D Dental Model Segmentation With Geometrical Boundary Preserving](3d_vision/3d_dental_model_segmentation_with_geometrical_boundary_preserving.md)**

:   提出 CrossTooth，通过基于曲率先验的选择性下采样（边界区域顶点密度提升 10-15%）和多视角渲染图像的跨模态边界特征融合，在 3DTeethSeg'22 公开数据集上实现 95.86% mIoU 和 82.05% boundary IoU，分别比之前 SOTA（ToothGroupNet）提升 2.3% 和 5.7%。

**[3D Gaussian Head Avatars With Expressive Dynamic Appearances By Compact Tensoria](3d_vision/3d_gaussian_head_avatars_with_expressive_dynamic_appearances_by_compact_tensoria.md)**

:   提出一种紧凑张量表示的3D高斯头部头像方法——用三平面存储中性表情的静态外观，用轻量1D特征线存储每个blendshape的动态纹理（不透明度偏移），仅需**10MB存储**即可实现300FPS实时渲染和准确的动态面部细节捕捉，在Nersemble数据集上PSNR和存储效率全面超越GA、GBS和GHA。

**[3D Gaussian Inpainting With Depth-Guided Cross-View Consistency](3d_vision/3d_gaussian_inpainting_with_depth-guided_cross-view_consistency.md)**

:   提出3DGIC，通过**深度引导的跨视角一致修复**框架实现3D高斯场景中的物体移除与修补——利用渲染深度图从其他视角发现被掩码区域中的可见背景像素来精化修补掩码，再用参考视角的2D修补结果通过3D投影约束其他视角的一致性，在SPIn-NeRF数据集上FID和LPIPS全面超越现有方法。

**[3D Student Splatting And Scooping](3d_vision/3d_student_splatting_and_scooping.md)**

:   提出SSS（Student Splatting and Scooping），用前所未有的三重创新改进3DGS范式：(1) 用**Student-t分布**替代高斯分布作为混合组件（可学习的尾部厚度，从Cauchy到Gaussian连续变化）；(2) 引入**负密度组件**（scooping减去颜色）扩展到非单调混合模型；(3) 用**SGHMC采样**替代SGD解耦参数优化，在Mip-NeRF360/T&T/Deep Blending上6/9指标取得最优，且参数效率极高——用**最少18%**的组件数即可匹配或超越3DGS。

**[3Denhancer Consistent Multi-View Diffusion For 3D Enhancement](3d_vision/3denhancer_consistent_multi-view_diffusion_for_3d_enhancement.md)**

:   提出一个基于多视图潜在扩散模型的3D增强框架，通过姿态感知编码器、多视图行注意力和近视图极线聚合模块，在保持跨视图一致性的前提下显著提升低质量3D生成结果的纹理质量。

**[3Dgut Enabling Distorted Cameras And Secondary Rays In Gaussian Splatting](3d_vision/3dgut_enabling_distorted_cameras_and_secondary_rays_in_gaussian_splatting.md)**

**[4Deform Neural Surface Deformation For Robust Shape Interpolation](3d_vision/4deform_neural_surface_deformation_for_robust_shape_interpolation.md)**

**[4Dequine Disentangling Motion And Appearance For 4D Equine Reconstruction From M](3d_vision/4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)**

:   将单目视频的4D马匹重建解耦为运动估计（AniMoFormer时空Transformer）和外观重建（EquineGS单图前馈3DGS），依托VAREN参数化模型和两个大规模合成数据集，在真实数据上达到SOTA几何+外观重建效果，且能零样本泛化到驴和斑马。

**[4Dtam Non-Rigid Tracking And Mapping Via Dynamic Surface Gaussians](3d_vision/4dtam_non-rigid_tracking_and_mapping_via_dynamic_surface_gaussians.md)**

:   本文提出了首个基于可微渲染和2D高斯表面基元的4D跟踪与建图方法（4DTAM），通过联合优化相机位姿、场景几何、外观和动态变形场，从单目RGB-D视频流实现非刚性动态场景的实时重建，并发布了全新的合成4D数据集Sim4D用于评估。

**[A Lightweight Udf Learning Framework For 3D Reconstruction Based On Local Shape ](3d_vision/a_lightweight_udf_learning_framework_for_3d_reconstruction_based_on_local_shape_.md)**

:   本文提出LoSF-UDF，一种基于局部形状函数学习无符号距离场（UDF）的轻量级框架，仅需在合成的局部点云patch上训练一次（653KB参数、0.5GB数据），即可泛化重建各种类型的3D表面，且对噪声和离群点具有鲁棒性。

**[A Unified Image-Dense Annotation Generation Model For Underwater Scenes](3d_vision/a_unified_image-dense_annotation_generation_model_for_underwater_scenes.md)**

:   本文提出TIDE，一种统一的文本到图像和密集标注生成方法，仅以文本为输入就能同时生成高度一致的水下图像、深度图和语义掩码，通过隐式布局共享（ILS）和时间自适应归一化（TAN）机制确保多模态输出的一致性，合成的SynTIDE数据集显著提升了水下深度估计和语义分割性能。

**[Activegamer Active Gaussian Mapping Through Efficient Rendering](3d_vision/activegamer_active_gaussian_mapping_through_efficient_rendering.md)**

:   提出 ActiveGAMER，首次将 3D Gaussian Splatting 用于主动建图，通过基于渲染的信息增益模块高效选择最优下一视角，结合粗到细探索、后精修和全局-局部关键帧策略，在 Replica 和 MP3D 数据集上大幅超越 NeRF-based 方法的几何精度和渲染保真度。

**[Aerialmegadepth Learning Aerial-Ground Reconstruction And View Synthesis](3d_vision/aerialmegadepth_learning_aerial-ground_reconstruction_and_view_synthesis.md)**

:   本文提出AerialMegaDepth数据集生成框架，通过将Google Earth的伪合成航空渲染与MegaDepth的真实地面图像联合配准到统一坐标系中，构建了13.2万张混合高度图像的大规模训练数据，微调DUSt3R后将地空配对的相机旋转估计准确率从5%提升到56%，同时显著改善了新视角合成质量。

**[Framevggt Frame Evidence Rolling Memory For Streaming Vggt](3d_vision/framevggt_frame_evidence_rolling_memory_for_streaming_vggt.md)**

:   提出 FrameVGGT，将流式 VGGT 的 KV 缓存从 token 级保留重组为帧级证据块保留，通过中期记忆库+稀疏锚点的双层有界内存结构，在固定内存预算下保持更连贯的几何支撑，实现长序列3D重建/深度/位姿估计的精度-内存最优权衡。

**[Hoi3Dgen Generating High-Quality Human-Object-Interactions In 3D](3d_vision/hoi3dgen_generating_high-quality_human-object-interactions_in_3d.md)**

:   提出 HOI3DGen 框架，通过MLLM自动标注高质量交互数据 + 视角条件化微调扩散模型 + 3D提升与SMPL配准，首次实现从文本精确控制接触语义的高质量3D人物交互生成，在文本一致性上超越基线4-15倍。

**[Hybrid Etfce-Grf Exact Cluster-Size Retrieval With Analytical P-Values For Voxel](3d_vision/hybrid_etfce-grf_exact_cluster-size_retrieval_with_analytical_p-values_for_voxel.md)**

:   将 eTFCE 的并查集精确聚类大小查询与 pTFCE 的解析 GRF p 值推断结合，首次在单一框架中实现精确聚类检索+无需置换检验的统计推断，速度比置换 TFCE 快 1300 倍，在全脑体素形态测量中保持严格 FWER 控制。

**[Instanthdr Single-Forward Gaussian Splatting For High Dynamic Range 3D Reconstru](3d_vision/instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)**

:   提出 InstantHDR，首个前馈式 HDR 新视角合成方法，通过几何引导的外观建模进行多曝光融合 + MetaNet 预测场景自适应色调映射器，从未标定多曝光 LDR 图像一次前向推理重建 HDR 3D 高斯，速度比优化方法快 ~700 倍。

**[Jopp-3D Joint Open Vocabulary Semantic Segmentation On Point Clouds And Panorama](3d_vision/jopp-3d_joint_open_vocabulary_semantic_segmentation_on_point_clouds_and_panorama.md)**

:   提出 JOPP-3D 框架，通过将全景图切线分解为透视图像、利用 SAM+CLIP 进行3D实例-语义对齐，首次实现对3D点云和全景图像的联合开放词汇语义分割，在 Stanford-2D-3D-s 和 ToF-360 数据集上超越现有方法。

**[Mobile-Gs Real-Time Gaussian Splatting For Mobile Devices](3d_vision/mobile-gs_real-time_gaussian_splatting_for_mobile_devices.md)**

:   提出 Mobile-GS，通过深度感知的无序渲染（消除排序瓶颈）+ 神经视角依赖增强 + 一阶SH蒸馏 + 神经向量量化 + 贡献度剪枝，首次在 Snapdragon 8 Gen 3 手机 GPU 上实现 116 FPS 实时高斯溅射渲染，存储仅 4.6MB 且视觉质量与原始 3DGS 相当。

**[Motionanymesh Physics-Grounded Articulation For Simulation-Ready Digital Twins](3d_vision/motionanymesh_physics-grounded_articulation_for_simulation-ready_digital_twins.md)**

:   提出 MotionAnyMesh，一种零样本框架，通过 SP4D 运动学先验引导 VLM 推理消除幻觉 + 物理约束轨迹优化保证无碰撞，将静态3D网格自动转化为仿真可用的铰接数字孪生，物理可执行率达 87%，是现有最好方法的近两倍。

**[Node-Rf Learning Generalized Continuous Space-Time Scene Dynamics With Neural Od](3d_vision/node-rf_learning_generalized_continuous_space-time_scene_dynamics_with_neural_od.md)**

:   提出 Node-RF，将 Neural ODE 与动态 NeRF 紧密耦合，用潜在向量的 ODE 演化建模场景连续时间动力学，实现超出训练序列的长程时序外推和跨轨迹泛化，无需光流或深度监督。

**[P-Slcr Unsupervised Point Cloud Semantic Segmentation Via Prototypes Structure L](3d_vision/p-slcr_unsupervised_point_cloud_semantic_segmentation_via_prototypes_structure_l.md)**

:   提出 P-SLCR，一种原型库驱动的无监督点云语义分割方法，通过将点分离为"一致"和"模糊"两类，用一致结构学习对齐一致点与原型 + 语义关系一致性推理约束两个原型库，在 S3DIS 上无监督达 47.1% mIoU，超越全监督 PointNet。

**[Pano360 Perspective To Panoramic Vision With Geometric Consistency](3d_vision/pano360_perspective_to_panoramic_vision_with_geometric_consistency.md)**

:   提出 Pano360，首个在3D摄影测量空间进行全景拼接的 Transformer 框架，利用预训练 VGGT 骨干获取3D感知的多视角特征对齐 + 多特征联合优化接缝检测，支持2到数百张输入图像，在弱纹理/大视差/重复模式场景下成功率达97.8%。

**[Regularizing Inr With Diffusion Prior Self-Supervised 3D Reconstruction Of Neutr](3d_vision/regularizing_inr_with_diffusion_prior_self-supervised_3d_reconstruction_of_neutr.md)**

:   提出 DINR (Diffusive INR)，将隐式神经表示 (INR/SIREN) 与预训练扩散模型先验结合，通过 proximal loss 在每个 DDIM 时间步用扩散去噪输出正则化 INR 重建，在稀疏视角中子 CT（低至 4-5 个视角）上超越 FBP、纯 INR、DD3IP 和经典 MBIR(qGGMRF) 方法。

**[Rewis3D Reconstruction Improves Weakly-Supervised Semantic Segmentation](3d_vision/rewis3d_reconstruction_improves_weakly-supervised_semantic_segmentation.md)**

:   Rewis3d 利用前馈 3D 重建（MapAnything）从 2D 视频中获取 3D 点云作为辅助监督信号，通过双 Student-Teacher 架构和加权跨模态一致性 (CMC) 损失，在仅使用稀疏标注（点/涂鸦/粗标记）的情况下将弱监督 2D 语义分割性能提升 2-7% mIoU，推理时仍为纯 2D。

**[Scope Scene-Contextualized Incremental Few-Shot 3D Segmentation](3d_vision/scope_scene-contextualized_incremental_few-shot_3d_segmentation.md)**

:   SCOPE 提出一个即插即用的背景引导原型富化框架，在基类训练后用类无关分割模型从背景区域挖掘伪实例建立 Instance Prototype Bank (IPB)，当新类别以少样本方式出现时，通过 Contextual Prototype Retrieval (CPR) 和 Attention-Based Prototype Enrichment (APE) 融合背景原型与少样本原型，在 ScanNet/S3DIS 上新类 IoU 提升最高 6.98%。

**[Spectral Defense Against Resource-Targeting Attack In 3D Gaussian Splatting](3d_vision/spectral_defense_against_resource-targeting_attack_in_3d_gaussian_splatting.md)**

:   针对 3DGS 的资源瞄准攻击（通过投毒训练图像触发高斯过度增长导致资源耗尽），提出频域防御：3D 频率滤波器通过将高斯协方差与频谱响应关联实现频率感知剪枝，2D 频谱正则化通过熵惩罚渲染图像的角向能量各向异性来抑制攻击噪声，实现高斯数量压缩 5.92×、内存减少 3.66×、速度提升 4.34×。

**[Towards Spatio-Temporal World Scene Graph Generation From Monocular Videos](3d_vision/towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)**

:   本文提出 World Scene Graph Generation (WSGG) 任务和 ActionGenome4D 数据集，将视频场景图从以帧为中心的 2D 表示升级为以世界为中心的 4D 表示，要求模型对所有物体（包括被遮挡或离开视野的不可见物体）在世界坐标系中进行 3D 定位和关系预测，并提出三种互补方法（PWG/MWAE/4DST）探索不同的不可见物体推理归纳偏置。

**[Varsplat Uncertainty-Aware 3D Gaussian Splatting For Robust Rgb-D Slam](3d_vision/varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)**

:   VarSplat 在 3DGS-SLAM 框架中为每个 Gaussian splat 学习外观方差 $\sigma^2$，通过全方差定律推导出可微分的逐像素不确定性图 $V$，并将其用于 tracking、loop detection 和 registration，在 Replica/TUM/ScanNet/ScanNet++ 四个数据集上取得了更鲁棒的位姿估计和有竞争力的重建质量。

---

## 🎨 图像生成 { #image_generation }

**[3Dtopia-Xl Scaling High-Quality 3D Asset Generation Via Primitive Diffusion](image_generation/3dtopia-xl_scaling_high-quality_3d_asset_generation_via_primitive_diffusion.md)**

:   提出基于新型原语表示PrimX和Diffusion Transformer的原生3D生成模型3DTopia-XL，能从文本或图像输入生成带有高分辨率几何、纹理和PBR材质的高质量3D资产，在质量和效率上显著超越现有方法。

**[A Comprehensive Study Of Decoder-Only Llms For Text-To-Image Generation](image_generation/a_comprehensive_study_of_decoder-only_llms_for_text-to-image_generation.md)**

:   系统研究了使用decoder-only LLM作为文本到图像扩散模型文本编码器的效果，发现直接使用最后一层embedding效果差于T5，但通过层归一化平均（layer-normalized averaging）聚合所有层的embedding可显著超越T5基线。

**[Amo Sampler Enhancing Text Rendering With Overshooting](image_generation/amo_sampler_enhancing_text_rendering_with_overshooting.md)**

:   提出AMO（Attention-Modulated Overshooting）采样器，一种无需训练的推理时增强方法，通过在rectified flow模型的采样过程中引入过冲-噪声补偿的Langevin动力学校正，并利用文本-图像交叉注意力分数自适应控制过冲强度，显著提升文本渲染的准确率，同时保持生成图像的整体质量。

**[An Image-Like Diffusion Method For Human-Object Interaction Detection](image_generation/an_image-like_diffusion_method_for_human-object_interaction_detection.md)**

**[Anidoc Animation Creation Made Easier](image_generation/anidoc_animation_creation_made_easier.md)**

**[Animer Animal Pose And Shape Estimation Using Family Aware Transformer](image_generation/animer_animal_pose_and_shape_estimation_using_family_aware_transformer.md)**

**[Any-Resolution Ai-Generated Image Detection By Spectral Learning](image_generation/any-resolution_ai-generated_image_detection_by_spectral_learning.md)**

:   提出一种基于频谱学习的 AI 生成图像检测方法，通过在频域提取分辨率不变的特征，使检测器能够在任意分辨率的输入图像上保持鲁棒性，解决了现有方法对特定分辨率依赖的问题。

**[Arbitrary-Steps Image Super-Resolution Via Diffusion Inversion](image_generation/arbitrary-steps_image_super-resolution_via_diffusion_inversion.md)**

:   本文提出InvSR，通过训练一个噪声预测网络来实现扩散反演（Diffusion Inversion），利用预训练扩散模型的图像先验进行超分辨率，支持1-5步任意步数采样，即使单步采样也能达到或超过现有SOTA方法的效果。

**[Articulated Kinematics Distillation From Video Diffusion Models](image_generation/articulated_kinematics_distillation_from_video_diffusion_models.md)**

:   本文提出AKD（Articulated Kinematics Distillation），通过将骨骼动画的low-DoF参数化与视频扩散模型的SDS蒸馏相结合，从文本驱动生成高保真3D角色动画，在3D一致性和运动质量上超过现有text-to-4D方法，并支持物理仿真进一步增强物理合理性。

**[Artifade Learning To Generate High-Quality Subject From Blemished Images](image_generation/artifade_learning_to_generate_high-quality_subject_from_blemished_images.md)**

:   本文提出ArtiFade，首个解决"瑕疵主题驱动生成"问题的方法，通过构建瑕疵-无瑕疵配对数据集、部分微调扩散模型的cross-attention权重并优化artifact-free embedding，使得现有主题驱动方法（Textual Inversion、DreamBooth）能从带水印/贴纸/对抗噪声等瑕疵的图像中生成高质量无伪影的主题图像。

**[As-Bridge A Bidirectional Generative Framework Bridging Next-Generation Astronom](image_generation/as-bridge_a_bidirectional_generative_framework_bridging_next-generation_astronom.md)**

:   提出 AS-Bridge，用双向布朗桥扩散模型建模地面 LSST 和太空 Euclid 两大天文巡天之间的随机映射关系，实现概率性跨巡天翻译与稀有事件检测（强引力透镜），并证明 epsilon-prediction 训练目标兼具重建质量和似然性优势。

**[Autopresent Designing Structured Visuals From Scratch](image_generation/autopresent_designing_structured_visuals_from_scratch.md)**

:   本文提出AutoPresent框架和SlidesBench基准，首次系统研究从自然语言指令生成演示幻灯片的任务——通过让LLM生成Python代码（而非端到端图像生成）来创建PPTX幻灯片，配合SlidesLib工具库和迭代优化，8B参数的开源模型达到接近GPT-4o的效果。

**[Beyond Convolution A Taxonomy Of Structured Operators For Learning-Based Image P](image_generation/beyond_convolution_a_taxonomy_of_structured_operators_for_learning-based_image_p.md)**

:   系统性地将学习式图像处理中卷积的替代/扩展算子组织为五大家族（分解型、自适应加权型、基自适应型、积分/核型和注意力型），并从线性、局部性、等变性、计算成本和任务适用性等多个维度进行比较分析。

**[Bigain Unified Token Compression For Joint Generation And Classification](image_generation/bigain_unified_token_compression_for_joint_generation_and_classification.md)**

:   BiGain 首次将扩散模型的 token 压缩重新定义为生成+分类的双目标优化问题，提出拉普拉斯门控 token 合并（L-GTM）和插值-外推 KV 下采样（IE-KVD）两个频率感知算子，在保持生成质量同时显著提升分类准确率（ImageNet-1K 70%合并比下 Acc +7.15%，FID -0.34）。

**[Codrawagents A Multi-Agent Dialogue Framework For Compositional Image Generation](image_generation/codrawagents_a_multi-agent_dialogue_framework_for_compositional_image_generation.md)**

:   提出 coDrawAgents，由 Interpreter、Planner、Checker、Painter 四个专家 agent 组成的交互式多智能体对话框架，通过分而治之的增量布局规划、视觉上下文感知推理和显式错误纠正，在 GenEval 上达到 0.94（SOTA）、DPG-Bench 上 85.17（SOTA）。

**[Dit-Ic Aligned Diffusion Transformer For Efficient Image Compression](image_generation/dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)**

:   DiT-IC 将预训练 T2I 扩散 Transformer 适配为单步图像压缩重建模型，在 32x 下采样的深层潜空间工作，通过方差引导重建流、自蒸馏对齐和潜变量条件引导三种对齐机制，实现 SOTA 感知质量且解码比现有扩散 codec 快 30 倍。

**[Editing Away The Evidence Diffusion-Based Image Manipulation And The Failure Mod](image_generation/editing_away_the_evidence_diffusion-based_image_manipulation_and_the_failure_mod.md)**

:   理论和实验统一分析了扩散模型编辑会"无意间"破坏鲁棒不可见水印的现象——正向加噪使水印 SNR 指数衰减，反向去噪的流形收缩效应将水印信号当作"非自然残差"消除，即使 VINE 等最先进水印在强编辑（$t^*=0.8$）下也降至接近随机猜测（~60% bit accuracy）。

**[Enhancing Image Aesthetics With Dual-Conditioned Diffusion Models Guided By Mult](image_generation/enhancing_image_aesthetics_with_dual-conditioned_diffusion_models_guided_by_mult.md)**

:   提出 DIAE，通过多模态美学感知模块（MAP）将模糊美学指令转化为 HSV/轮廓图+文本的多模态控制信号，并构建"非完美配对"数据集 IIAEData 配合双分支监督策略实现弱监督美学增强，在 LAION 和 MLLM 美学评分上达 SOTA。

**[Evotok A Unified Image Tokenizer Via Residual Latent Evolution For Visual Unders](image_generation/evotok_a_unified_image_tokenizer_via_residual_latent_evolution_for_visual_unders.md)**

:   EvoTok 提出了一种基于残差潜在演化（Residual Latent Evolution）的统一图像 tokenizer，通过在共享潜空间中级联残差向量量化，使表示从浅层的像素级细节渐进演化到深层的语义级抽象，在仅用 13M 图像训练的情况下实现了 0.43 rFID 的重建质量，并在 7/9 个理解 benchmark 和 GenEval/GenAI-Bench 上取得优异效果。

**[Fractals Made Practical Denoising Diffusion As Partitioned Iterated Function Sys](image_generation/fractals_made_practical_denoising_diffusion_as_partitioned_iterated_function_sys.md)**

:   证明 DDIM 确定性反向链是一个分区迭代函数系统（PIFS），由此推导出三个无需模型评估的可计算几何量（收缩阈值 $L_t^*$、膨胀函数 $f_t(\lambda)$、全局膨胀阈值 $\lambda^{**}$），并据此从理论上解释了四个现有的经验性设计选择（cosine offset、分辨率 logSNR shift、Min-SNR 加权、Align Your Steps）。

**[Generation Of Maximal Snake Polyominoes Using A Deep Neural Network](image_generation/generation_of_maximal_snake_polyominoes_using_a_deep_neural_network.md)**

:   将 DDPM 应用于生成最大蛇形多联骨牌，提出精简版 Structured Pixel Space Diffusion（SPS Diffusion），在训练到 14x14 正方网格的情况下泛化到 28x28 并生成有效蛇形，部分结果超越已知最大长度下界。

**[Interedit Navigating Text-Guided Multi-Human 3D Motion Editing](image_generation/interedit_navigating_text-guided_multi-human_3d_motion_editing.md)**

:   提出 InterEdit，首个文本引导的多人 3D 运动交互编辑框架，通过 Semantic-Aware Plan Token Alignment 和 Interaction-Aware Frequency Token Alignment 在扩散模型中实现语义编辑的同时保持多人之间的时空耦合关系。

**[One Model Many Budgets Elastic Latent Interfaces For Diffusion Transformers](image_generation/one_model_many_budgets_elastic_latent_interfaces_for_diffusion_transformers.md)**

:   揭示 DiT 的计算在空间 token 上均匀分配（不会把多余计算重分配到困难区域），提出 ELIT——在 DiT 中插入可变长度的 latent interface（Read/Write 交叉注意力），训练时随机丢弃尾部 latent 学出重要性排序，推理时通过调节 latent 数量实现平滑的质量-FLOPs 权衡，ImageNet 512px 上 FID 降低 53%。

**[Overcoming Visual Clutter In Vision Language Action Models Via Concept-Gated Vis](image_generation/overcoming_visual_clutter_in_vision_language_action_models_via_concept-gated_vis.md)**

:   提出 Concept-Gated Visual Distillation (CGVD)，一种无需训练的推理时框架，通过语言指令解析 → SAM3 分割 → 集合论交叉验证 → LaMa 修复的流水线，从 VLA 模型的视觉输入中选择性移除语义干扰物，在高度杂乱场景中将 π₀ 的操作成功率从 43.0% 提升至 77.5%。

**[Taming Score-Based Denoisers In Admm A Convergent Plug-And-Play Framework](image_generation/taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)**

:   提出 AC-DC 去噪器（Auto-Correction + Directional Correction + Score-Based Denoising 三阶段），解决将 score-based 扩散先验嵌入 ADMM-PnP 框架时的流形不匹配问题，并首次建立了 score-based 去噪器在 ADMM 中的收敛性理论保证，在去噪、修复、去模糊、超分辨、相位恢复、HDR 等逆问题上一致超越现有基线。

**[Trust Your Critic Robust Reward Modeling And Reinforcement Learning For Faithful](image_generation/trust_your_critic_robust_reward_modeling_and_reinforcement_learning_for_faithful.md)**

:   提出 FIRM 框架——通过"差异优先"（编辑）和"计划-打分"（生成）的数据构建流水线训练专用奖励模型（FIRM-Edit-8B / FIRM-Gen-8B），配合"Base-and-Bonus"奖励策略（CME/QMA）解决 RL 中的奖励 hacking 问题，在图像编辑和 T2I 生成任务上均取得 SOTA。

**[Unicom Unified Multimodal Modeling Via Compressed Continuous Semantic Representa](image_generation/unicom_unified_multimodal_modeling_via_compressed_continuous_semantic_representa.md)**

:   提出 UniCom，通过对 VLM 连续语义特征进行**通道维度压缩**（而非空间下采样），构建紧凑连续表示空间，用 Transfusion 架构统一多模态理解与生成，在统一模型中达到 SOTA 生成质量。

**[V-Bridge Bridging Video Generative Priors To Versatile Few-Shot Image Restoratio](image_generation/v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)**

:   将图像复原重新定义为**渐进式视频生成过程**，利用预训练视频生成模型（Wan2.2-TI2V-5B）的先验知识，仅用 1,000 个多任务训练样本（不到现有方法的 2%）即可实现竞争力的多任务图像复原。

**[Visual-Erm Reward Modeling For Visual Equivalence](image_generation/visual-erm_reward_modeling_for_visual_equivalence.md)**

:   提出 Visual-ERM，一个多模态生成式奖励模型，在视觉空间中直接评估 vision-to-code 任务的渲染质量，提供细粒度、可解释、任务无关的奖励信号，用于 RL 训练和测试时缩放。

**[When To Lock Attention Training-Free Kv Control In Video Diffusion](image_generation/when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)**

:   提出 KV-Lock，一种基于扩散幻觉检测的免训练视频编辑框架，通过动态调度 KV 缓存融合比例和 CFG 引导尺度，在保持背景一致性的同时增强前景生成质量。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[3D-Avs Lidar-Based 3D Auto-Vocabulary Segmentation](autonomous_driving/3d-avs_lidar-based_3d_auto-vocabulary_segmentation.md)**

:   提出3D-AVS，首个针对LiDAR点云的**自动词表分割**方法：无需用户指定目标类别，系统自动从图像和点云中识别场景中存在的语义实体并生成词表，再用开放词表分割器完成逐点语义分割，在nuScenes和ScanNet200上展示了生成精细语义类别的能力。

**[3D Occupancy Prediction With Low-Resolution Queries Via Prototype-Aware View Tra](autonomous_driving/3d_occupancy_prediction_with_low-resolution_queries_via_prototype-aware_view_tra.md)**

:   提出ProtoOcc，通过**原型感知视角变换**将2D图像聚类原型映射到3D体素查询空间来增强低分辨率体素的上下文信息，配合**多视角占用解码**策略从增强的体素中重建高分辨率3D占用场景，用75%更小的体素分辨率仍能达到与高分辨率方法竞争的性能（Occ3D mIoU 37.80 vs PanoOcc 38.11）。

**[A Neuro-Symbolic Framework Combining Inductive And Deductive Reasoning For Auton](autonomous_driving/a_neuro-symbolic_framework_combining_inductive_and_deductive_reasoning_for_auton.md)**

:   本文提出首个将 ASP 符号推理决策以可学习嵌入形式直接嵌入端到端规划器轨迹解码的神经-符号框架，用 LLM 动态提取场景规则、Clingo 求解器进行逻辑仲裁、可微 KBM 生成物理可行轨迹并配合神经残差修正，在 nuScenes 上 L₂ 误差 0.57m、碰撞率 0.075%、TPC 0.47m 全面超越 MomAD。

**[A Prediction-As-Perception Framework For 3D Object Detection](autonomous_driving/a_prediction-as-perception_framework_for_3d_object_detection.md)**

:   PAP 受人脑"预测性感知"启发，将上一帧轨迹预测结果作为当前帧感知模块的 query 输入替代部分随机 query，在 UniAD 上实现 AMOTA 提升 10%（0.359→0.395）、推理速度提升 15%（14→16 FPS）和训练时间缩短 14%。

**[Cawm-Mamba A Unified Model For Infrared-Visible Image Fusion And Compound Advers](autonomous_driving/cawm-mamba_a_unified_model_for_infrared-visible_image_fusion_and_compound_advers.md)**

:   CAWM-Mamba 首次提出端到端统一处理红外-可见光图像融合与复合恶劣天气（如雾+雨、雨+雪）场景的框架，通过天气感知预处理、跨模态特征交互和小波域频率-SSM 解耦多频退化，在 AWMM-100K 和标准融合数据集上全面超越 SOTA。

**[Certified Human Trajectory Prediction](autonomous_driving/certified_human_trajectory_prediction.md)**

:   首次将随机平滑（Randomized Smoothing）认证技术引入人类轨迹预测任务，通过mean/median聚合函数和扩散去噪器为轨迹预测模型提供保证性鲁棒性——即无论输入噪声如何扰动（在半径R内），输出始终保持在认证边界内。

**[Climbingcap Multi-Modal Dataset And Method For Rock Climbing In World ](autonomous_driving/climbingcap_multi-modal_dataset_and_method_for_rock_climbing_in_world_.md)**

:   构建了首个大规模攀岩运动多模态数据集 AscendMotion（412K帧，RGB+LiDAR+IMU），并提出 ClimbingCap 方法通过分离坐标解码、后处理优化和半监督训练，在世界坐标系中精确恢复攀岩者的3D运动。

**[Climbingcap Multi-Modal Dataset And Method For Rock Climbing In World Coordinate](autonomous_driving/climbingcap_multi-modal_dataset_and_method_for_rock_climbing_in_world_coordinate.md)**

:   提出首个攀岩运动多模态数据集 AscendMotion（412K 帧 RGB+LiDAR+IMU，22 名专业攀岩者，12 面攀岩墙），以及 ClimbingCap 方法通过分离坐标解码、三重后处理优化和半监督训练实现世界坐标系下的 3D 攀岩动作恢复，MPJPE 达 75.45mm。

**[Closed-Loop Supervised Fine-Tuning Of Tokenized Traffic Models](autonomous_driving/closed-loop_supervised_fine-tuning_of_tokenized_traffic_models.md)**

**[Composing Driving Worlds Through Disentangled Control For Adversarial Scenario G](autonomous_driving/composing_driving_worlds_through_disentangled_control_for_adversarial_scenario_g.md)**

:   CompoSIA 提出一种基于 Flow Matching DiT 的组合式驾驶视频生成框架，通过解耦结构（3D bbox）、身份（单参考图像）和自车动作（相机轨迹）三类控制信号的注入方式，实现精细独立控制和组合编辑，用于系统化合成对抗性驾驶场景，FVD 提升 17%，碰撞率增加 173%。

**[Cubify Anything Scaling Indoor 3D Object Detection](autonomous_driving/cubify_anything_scaling_indoor_3d_object_detection.md)**

**[Decoupledgaussian Object-Scene Decoupling For Physics-Based Interaction](autonomous_driving/decoupledgaussian_object-scene_decoupling_for_physics-based_interaction.md)**

:   将 3DGS 场景中的物体与背景解耦，使物体支持物理仿真（碰撞、抓取等），同时保持场景的高质量渲染

**[Diffusiondrive Truncated Diffusion Model For End-To-End Autonomous Driving](autonomous_driving/diffusiondrive_truncated_diffusion_model_for_end-to-end_autonomous_driving.md)**

:   本文提出DiffusionDrive，通过截断扩散策略（将去噪步骤从20步减少到2步）和级联扩散解码器，首次将扩散模型成功应用于端到端自动驾驶的实时多模态轨迹规划，在NAVSIM数据集上以88.1 PDMS刷新记录，同时保持45 FPS的实时速度。

**[Distilling Monocular Foundation Model For Fine-Grained Depth Completion](autonomous_driving/distilling_monocular_foundation_model_for_fine-grained_depth_completion.md)**

:   本文提出DMD3C，一个两阶段知识蒸馏框架，将单目深度基础模型（如Depth Anything V2）的几何知识迁移到深度补全网络，第一阶段通过合成训练数据进行预训练，第二阶段通过尺度-偏移不变损失（SSI Loss）在真实数据上微调，在KITTI深度补全排行榜上取得第一名。

**[Distilling Multi-Modal Large Language Models For Autonomous Driving](autonomous_driving/distilling_multi-modal_large_language_models_for_autonomous_driving.md)**

:   本文提出DiMA框架，通过联合训练在多模态大语言模型（MLLM）和视觉端到端规划器之间进行知识蒸馏，设计了遮蔽重建、未来预测和场景编辑三种代理任务来丰富场景表示，推理时可丢弃LLM仅用视觉规划器，在nuScenes上实现L2轨迹误差降低37%、碰撞率降低80%。

**[Driving By The Rules A Benchmark For Integrating Traffic Sign Regulations Into V](autonomous_driving/driving_by_the_rules_a_benchmark_for_integrating_traffic_sign_regulations_into_v.md)**

:   本文首次定义了将交通标志规则集成到在线向量化高精地图的任务，构建了包含10000+视频片段和18000+车道级规则的MapDR数据集，并提出模块化（VLE-MEE）和端到端（RuleVLM）两种基线方案，其中RuleVLM在整体F1指标上达到64.2%。

**[Lr-Sgs Robust Lidar-Reflectance-Guided Salient Gaussian Splatting For Self-Drivi](autonomous_driving/lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)**

:   LR-SGS 提出基于 LiDAR 反射率引导的显著高斯泼溅方法，引入结构感知的显著高斯表示（由 LiDAR 几何和反射率特征点初始化）和光照不变的反射率通道作为额外约束，在 Waymo 数据集挑战场景（复杂光照）上 PSNR 超越 OmniRe 1.18 dB。

**[M2-Occ Resilient 3D Semantic Occupancy Prediction For Autonomous Driving With In](autonomous_driving/m2-occ_resilient_3d_semantic_occupancy_prediction_for_autonomous_driving_with_in.md)**

:   M²-Occ 针对多相机输入不完整时的语义占用预测问题，提出多视角掩码重建（MMR）模块利用相邻相机重叠区域恢复缺失视角特征，以及特征记忆模块（FMM）通过类级语义原型精炼不确定体素特征，在缺失后视角设置下 IoU 提升 4.93%。

**[Mapgclr Geospatial Contrastive Learning Of Representations For Online Vectorized](autonomous_driving/mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)**

:   MapGCLR 提出地理空间对比学习方法，通过强制多次行驶中地理空间重叠区域的 BEV 特征一致性来改善在线矢量化 HD 地图构建的 BEV 编码器，在仅 5% 标注数据下实现 42% 的相对 mAP 提升。

**[O3N Omnidirectional Open-Vocabulary Occupancy Prediction](autonomous_driving/o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)**

:   O3N 首次提出纯视觉端到端的全向开放词汇占用预测框架，通过极坐标螺旋 Mamba（PsM）建模全向空间连续性、占用代价聚合（OCA）统一几何和语义监督、以及无梯度自然模态对齐（NMA）桥接像素-体素-文本模态间隙，在 QuadOcc 和 Human360Occ 上达到 SOTA。

**[Panoramic Multimodal Semantic Occupancy Prediction For Quadruped Robots](autonomous_driving/panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)**

:   首个面向四足机器人的全景多模态语义占用预测框架 VoxelHound，提出 PanoMMOcc 数据集（全景 RGB + 热成像 + 偏振 + LiDAR），通过垂直抖动补偿（VJC）和多模态信息提示融合（MIPF）模块达到 23.34% mIoU。

**[Single Pixel Image Classification Using An Ultrafast Digital Light Projector](autonomous_driving/single_pixel_image_classification_using_an_ultrafast_digital_light_projector.md)**

:   利用 microLED-on-CMOS 超快数字光投影器实现基于单像素成像（SPI）的 MNIST 图像分类，在 1.2 kfps 帧率下达到 >90% 分类精度，完全绕过图像重建直接从时序光信号分类。

**[Spectral-Geometric Neural Fields For Pose-Free Lidar View Synthesis](autonomous_driving/spectral-geometric_neural_fields_for_pose-free_lidar_view_synthesis.md)**

:   SG-NLF 提出一种无需精确位姿的 LiDAR NeRF 框架，通过混合频谱-几何表征重建平滑几何、置信度感知位姿图实现全局对齐、对抗学习增强跨帧一致性，在低频 LiDAR 场景下重建质量和位姿精度分别超越 SOTA 35.8% 和 68.8%。

**[Vird View-Invariant Representation Through Dual-Axis Transformation For Cross-Vi](autonomous_driving/vird_view-invariant_representation_through_dual-axis_transformation_for_cross-vi.md)**

:   VIRD 通过双轴变换（极坐标变换 + 上下文增强位置注意力）构建视角不变表征，实现无需方向先验的全向跨视角位姿估计，在 KITTI 上位置和方向误差分别降低 50.7% 和 76.5%。

---

## 🧩 多模态VLM { #multimodal_vlm }

**[4D Langsplat 4D Language Gaussian Splatting Via Multimodal Large Language Models](multimodal_vlm/4d_langsplat_4d_language_gaussian_splatting_via_multimodal_large_language_models.md)**

:   提出4D LangSplat，通过多模态大语言模型生成逐物体视频caption来构建4D语言场，结合状态可变形网络建模语义的时间连续演变，首次实现动态场景中时间敏感和时间无关的开放词汇查询。

**[A Closed-Form Solution For Debiasing Vision-Language Models With Utility Guarant](multimodal_vlm/a_closed-form_solution_for_debiasing_vision-language_models_with_utility_guarant.md)**

:   提出一个 training-free、data-free 的 VLM 去偏方法，通过在 cross-modal 空间中推导闭式解，实现 Pareto-optimal 的公平性与效用保持，在零样本分类、text-to-image 检索和生成三个下游任务中全面超越已有方法。

**[Active Data Curation Effectively Distills Large-Scale Multimodal Models](multimodal_vlm/active_data_curation_effectively_distills_large-scale_multimodal_models.md)**

:   提出 ACID（主动数据筛选即隐式蒸馏）和 ACED（结合显式蒸馏），证明用大模型作为参考来主动筛选训练数据是一种比传统知识蒸馏更有效的多模态模型压缩方式，两者互补结合后在 27 个零样本任务上以更少推理 FLOPs 达到 SOTA。

**[Beyond Final Answers Crystal Benchmark For Transparent Multimodal Reasoning Eval](multimodal_vlm/beyond_final_answers_crystal_benchmark_for_transparent_multimodal_reasoning_eval.md)**

:   提出 CRYSTAL benchmark（6372 实例），通过 Match F1 和 Ordered Match F1 两个指标在中间推理步骤层面评估 MLLM，揭示了普遍的 cherry-picking 行为和推理顺序混乱问题，并提出 CPR-Curriculum 训练策略改善推理质量。

**[Beyond Words Augmenting Discriminative Richness Via Diffusions In Unsupervised P](multimodal_vlm/beyond_words_augmenting_discriminative_richness_via_diffusions_in_unsupervised_p.md)**

:   提出AiR（Augmenting discriminative Richness）方法，利用LoRA微调的Stable Diffusion生成合成图像构建辅助分类器，与文本分类器互补融合，将无监督prompt learning中的文本-图像匹配扩展为图像-图像匹配，显著提升细粒度/遥感等困难数据集上的分类准确率。

**[Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves](multimodal_vlm/can_large_vision-language_models_correct_semantic_grounding_errors_by_themselves.md)**

:   系统研究了VLM在语义定位任务中的自我纠错能力，发现内在自我纠错（无外部反馈）反而损害性能（-7至-17点），但通过同一VLM作为二值验证器提供反馈的迭代纠错最多可提升8.4个百分点，揭示了反馈质量是自我纠错的关键瓶颈。

**[Coap Memory-Efficient Training With Correlation-Aware Gradient Projection](multimodal_vlm/coap_memory-efficient_training_with_correlation-aware_gradient_projection.md)**

**[Comm A Coherent Interleaved Image-Text Dataset For Multimodal Understanding And ](multimodal_vlm/comm_a_coherent_interleaved_image-text_dataset_for_multimodal_understanding_and_.md)**

**[Completion As Enhancement A Degradation-Aware Selective Image Guided Network For](multimodal_vlm/completion_as_enhancement_a_degradation-aware_selective_image_guided_network_for.md)**

:   将图像增强重构为'补全'范式，通过退化感知选择机制引导网络聚焦于需要增强的区域，避免对已清晰区域的过度处理

**[Context-Aware Multimodal Pretraining](multimodal_vlm/context-aware_multimodal_pretraining.md)**

:   本文提出LIxP（Language-Image Contextual Pretraining），通过在对比式图文预训练中引入交叉注意力上下文化机制，使视觉-语言模型在不损失零样本性能的前提下，显著提升了基于度量的few-shot适应能力（21个下游任务平均提升5%以上，样本效率提升可达4倍）。

**[Continual Learning With Vision-Language Models Via Semantic-Geometry Preservatio](multimodal_vlm/continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)**

:   提出 SeGP-CL 框架，通过对抗性锚点（DPGD）精准探测新旧任务语义边界的脆弱区域，结合跨模态几何蒸馏（ACGD）和文本语义正则化（TSGR）保护 VLM 的跨模态几何结构，在五个持续学习 benchmark 上达到 SOTA。

**[Counts Benchmarking Object Detectors And Multimodal Large Language Models Under ](multimodal_vlm/counts_benchmarking_object_detectors_and_multimodal_large_language_models_under_.md)**

:   本文构建了COUNTS——一个包含14种自然分布偏移、222K+样本和119万+标注框的大规模OOD数据集，并提出O(OD)²和OODG两个基准，系统评估了目标检测器和多模态大模型在分布偏移下的泛化能力，发现即使是GPT-4o也仅能达到56.7%的定位准确率。

**[Critic-V Vlm Critics Help Catch Vlm Errors In Multimodal Reasoning](multimodal_vlm/critic-v_vlm_critics_help_catch_vlm_errors_in_multimodal_reasoning.md)**

:   本文提出Critic-V框架，将VLM推理过程解耦为Reasoner（推理器）和Critic（评价器），通过DPO训练的Critic模型提供自然语言反馈迭代优化推理路径，在8个基准上的5个超越GPT-4V，数学推理任务提升尤为显著（MathVista +11.8%）。

**[Cropper Vision-Language Model For Image Cropping Through In-Context Learning](multimodal_vlm/cropper_vision-language_model_for_image_cropping_through_in-context_learning.md)**

:   本文提出Cropper框架，首次利用大型视觉-语言模型（VLM）的上下文学习（ICL）能力来解决图像裁剪任务，通过高效的prompt检索和基于反馈的迭代裁剪优化策略，无需任何训练即可在自由裁剪、主体感知裁剪和宽高比裁剪三种任务上大幅超越有监督SOTA方法。

**[Espire A Diagnostic Benchmark For Embodied Spatial Reasoning Of Vision-Language ](multimodal_vlm/espire_a_diagnostic_benchmark_for_embodied_spatial_reasoning_of_vision-language_.md)**

:   提出 Espire，一个基于仿真环境的具身空间推理诊断基准，将 VLM 评估分解为定位和执行两阶段，通过全生成式范式系统评估 VLM 在多种空间推理维度和粒度上的能力。

**[Forensiczip More Tokens Are Better But Not Necessary In Forensic Vision-Language](multimodal_vlm/forensiczip_more_tokens_are_better_but_not_necessary_in_forensic_vision-language.md)**

:   发现语义驱动的视觉 token 剪枝会丢弃 forensic 证据（篡改痕迹在低显著性区域），提出 ForensicZip 用 Birth-Death 最优传输量化帧间物理不连续性 + 高频先验保留取证信号，在 10% token 保留率下实现 2.97x 加速、90%+ FLOPs 降低且性能不降。

**[Hificl High-Fidelity In-Context Learning For Multimodal Tasks](multimodal_vlm/hificl_high-fidelity_in-context_learning_for_multimodal_tasks.md)**

:   通过对 attention 公式的精确分解，揭示 ICL 的效果本质上是 query-dependent 的标准自注意力输出与上下文 value 的动态混合，据此提出直接参数化"虚拟 KV 对"（低秩分解）来高保真模拟 ICL，仅 2.2M 参数即超越 MimIC/LoRA，且训练快 7.5 倍。

**[Mastering Negation Boosting Grounding Models Via Grouped Opposition-Based Learni](multimodal_vlm/mastering_negation_boosting_grounding_models_via_grouped_opposition-based_learni.md)**

:   构建首个包含正负语义描述的视觉定位数据集 D-Negation，并提出 Grouped Opposition-Based Learning (GOBL) 微调机制，通过对立语义约束显著增强 grounding 模型对否定语义的理解能力。

**[Multimodal Ocr Parse Anything From Documents](multimodal_vlm/multimodal_ocr_parse_anything_from_documents.md)**

:   提出 Multimodal OCR (MOCR) 范式，将文档中的文本和图形（图表、图标、UI 等）统一解析为结构化文本表示（包括 SVG 代码），3B 模型在 olmOCR-Bench 上达到 83.9 SOTA，图形解析超越 Gemini 3 Pro。

**[Revisiting Model Stitching In The Foundation Model Era](multimodal_vlm/revisiting_model_stitching_in_the_foundation_model_era.md)**

:   系统研究异构 Vision Foundation Model（如 CLIP、DINOv2、SigLIP 2）之间的 stitchability，发现用 Final Feature Matching 预训练 stitch layer 可实现可靠拼接，并提出 VFM Stitch Tree 架构实现多 VFM 的高效共享。

**[Spatial Reasoning Is Not A Free Lunch A Controlled Study On Llava](multimodal_vlm/spatial_reasoning_is_not_a_free_lunch_a_controlled_study_on_llava.md)**

:   通过在 LLaVA 框架中系统替换图像编码器（CLIP/SigLIP/SigLIP2/AIMv2）和引入 2D-RoPE 位置编码，发现 VLM 的空间推理能力主要由编码器的训练目标决定，指望仅靠 2D 位置结构改善空间理解是不够的。

**[Test-Time Attention Purification For Backdoored Large Vision Language Models](multimodal_vlm/test-time_attention_purification_for_backdoored_large_vision_language_models.md)**

:   CleanSight 发现 LVLM 后门攻击的机制不在像素层面而在注意力层面——触发器通过"注意力窃取"（trigger token 抢夺 text token 的注意力）来激活后门，据此提出了一种免训练、即插即用的 test-time 防御方法：通过检测跨模态注意力比例异常来识别中毒输入，再通过剪枝高注意力视觉 token 来中和后门，ASR 降至接近 0% 且几乎不影响模型性能。

**[Topo-R1 Detecting Topological Anomalies Via Vision-Language Models](multimodal_vlm/topo-r1_detecting_topological_anomalies_via_vision-language_models.md)**

:   发现现有 VLM（包括 GPT-5.2、Gemini-2.5）在拓扑异常检测上几乎为零（F1@0.5 < 1.5%），提出 Topo-R1 框架通过 SFT + GRPO（含拓扑感知复合 reward，集成 type-aware Hungarian matching + clDice）赋予 VLM 拓扑感知能力，最佳 F1@0.5 达 45.2%。

---

## ✂️ 语义分割 { #segmentation }

**[2Dmamba Efficient State Space Model For Image Representation With Applications O](segmentation/2dmamba_efficient_state_space_model_for_image_representation_with_applications_o.md)**

:   提出2DMamba，首个具有高效并行算法的**原生2D选择性状态空间模型**，通过保持2D空间连续性（而非展平为1D序列）来建模WSI中的patch间关系，在10个公共病理数据集上全面超越1D Mamba方法，并在ImageNet分类和ADE20K分割上也有提升。

**[A Distractor-Aware Memory For Visual Object Tracking With Sam2](segmentation/a_distractor-aware_memory_for_visual_object_tracking_with_sam2.md)**

:   提出SAM2.1++的干扰物感知记忆模型（DAM），将SAM2的记忆拆分为近期外观记忆（RAM，确保分割精度）和干扰物解析记忆（DRM，确保跟踪鲁棒性），通过内省式更新策略检测干扰物并自动存储锚帧，在7个基准上设立新SOTA。

**[Assessing And Learning Alignment Of Unimodal Vision And Language Model](segmentation/assessing_and_learning_alignment_of_unimodal_vision_and_language_model.md)**

:   提出 SAIL 框架——先通过 alignment probing 评估单模态视觉和语言模型的对齐潜力（发现 k-NN 聚类质量比线性可分性更重要），再用轻量级 GLU 对齐层 + Sigmoid 损失 + 多正样本策略高效对齐 DINOv2 和预训练语言模型，仅用 6% 的 CLIP 训练数据即超越 CLIP。

**[Assessing And Learning Alignment Of Unimodal Vision And Language Models](segmentation/assessing_and_learning_alignment_of_unimodal_vision_and_language_models.md)**

**[Audio-Visual Instance Segmentation](segmentation/audio-visual_instance_segmentation.md)**

**[Binwang2Hfnet Geogran-Aware Hierarchical Feature Fusion Network For Salient Obje](segmentation/binwang2hfnet_geogran-aware_hierarchical_feature_fusion_network_for_salient_obje.md)**

:   提出 G2HFNet，通过多尺度细节增强 (MDE)、双分支几何-粒度互补 (DGC)、深层语义感知 (DSP) 和局部-全局引导融合 (LGF) 四个模块，针对不同层级特征设计差异化优化策略，在三个遥感显著性检测数据集上全面超越 SOTA。

**[Condensing Action Segmentation Datasets Via Generative Network Inversion](segmentation/condensing_action_segmentation_datasets_via_generative_network_inversion.md)**

**[Continuous Locomotive Crowd Behavior Generation](segmentation/continuous_locomotive_crowd_behavior_generation.md)**

:   生成连续的人群运动行为，实现轨迹和动作的联合合成，产生自然且多样的群体运动模式

**[Cosmos Cross-Modality Self-Distillation For Vision Language Pre-Training](segmentation/cosmos_cross-modality_self-distillation_for_vision_language_pre-training.md)**

:   COSMOS 提出了一种跨模态自蒸馏框架，通过文本裁剪策略和交叉注意力模块在学生-教师结构中学习细粒度的跨模态表征，在仅使用 30M 数据预训练的情况下，在零样本检索、分类和语义分割任务上全面超越 CLIP 类基线，甚至超越在数十亿数据上训练的 OpenCLIP。

**[Crossearth-Sar A Sar-Centric And Billion-Scale Geospatial Foundation Model For D](segmentation/crossearth-sar_a_sar-centric_and_billion-scale_geospatial_foundation_model_for_d.md)**

:   提出首个十亿参数级 SAR 视觉基础模型 CrossEarth-SAR，基于物理引导的稀疏混合专家 (MoE) 架构，构建了包含 200K 图像的训练集和 22 个子基准的评估体系，在 20/22 个跨域语义分割基准上达到 SOTA。

**[Declip Decoupled Learning For Open-Vocabulary Dense Perception](segmentation/declip_decoupled_learning_for_open-vocabulary_dense_perception.md)**

:   DeCLIP 发现 CLIP 的自注意力中存在"代理 token"现象导致图像 token 无法聚合空间相关信息，提出将自注意力模块解耦为"内容"和"上下文"特征并分别用 CLIP 自蒸馏和视觉基础模型蒸馏进行优化的框架，在开放词汇目标检测和语义分割上全面超越现有方法。

**[Defmamba Deformable Visual State Space Model](segmentation/defmamba_deformable_visual_state_space_model.md)**

:   DefMamba 提出了一种基于可变形机制的视觉状态空间模型，通过可变形扫描策略动态调整扫描路径（参考点偏移 + 扫描顺序偏移），克服了现有 Visual Mamba 方法使用固定扫描顺序导致的空间结构信息丢失问题，在 ImageNet 分类、COCO 检测和 ADE20K 分割上达到 SOTA。

**[Efficient Rgb-D Scene Understanding Via Multi-Task Adaptive Learning And Cross-D](segmentation/efficient_rgb-d_scene_understanding_via_multi-task_adaptive_learning_and_cross-d.md)**

:   提出一个高效 RGB-D 多任务场景理解网络，通过改进融合编码器利用冗余特征加速推理，引入归一化聚焦通道层 (NFCL) 和上下文特征交互层 (CFIL) 进行跨维度特征引导，并设计多任务自适应损失函数动态调整任务权重，在 NYUv2/SUN RGB-D/Cityscapes 上达到 SOTA。

**[Hfp-Sam Hierarchical Frequency Prompted Sam For Efficient Marine Animal Segmenta](segmentation/hfp-sam_hierarchical_frequency_prompted_sam_for_efficient_marine_animal_segmenta.md)**

:   HFP-SAM 提出分层频率提示的 SAM 框架，通过频率引导适配器（FGA）注入海洋场景信息、频率感知点选择（FPS）自动生成高质量点提示、全视图 Mamba（FVM）高效解码，在四个海洋动物分割数据集上取得 SOTA。

**[Picosam3 Real-Time In-Sensor Region-Of-Interest Segmentation](segmentation/picosam3_real-time_in-sensor_region-of-interest_segmentation.md)**

:   PicoSAM3 是一个 1.3M 参数的超轻量可提示分割模型，通过 ROI 隐式提示编码、密集 CNN 架构（无 Transformer）、SAM3 知识蒸馏和 INT8 量化，在 COCO 上达 65.45% mIoU，并实现在 Sony IMX500 视觉传感器上 11.82ms 实时推理。

**[Prompt-Driven Lightweight Foundation Model For Instance Segmentation-Based Fault](segmentation/prompt-driven_lightweight_foundation_model_for_instance_segmentation-based_fault.md)**

:   SAM FTI-FDet 提出基于轻量 SAM 的自动提示实例分割框架，通过 Transformer 解码器式的提示生成器自动产生任务相关提示、自适应特征分发器融合多尺度特征、TinyViT backbone 降低计算开销，在货运列车故障检测数据集上达 74.6 $AP^{box}$ / 74.2 $AP^{mask}$。

**[Rdnet Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network ](segmentation/rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)**

:   RDNet 针对遥感图像中目标尺度剧烈变化的问题，提出区域比例感知的动态自适应显著性检测网络，通过动态自适应细节感知模块（DAD，根据目标区域比例选择不同大小卷积核组合）、频率匹配上下文增强模块（FCE，小波域特征交互）和区域比例感知定位模块（RPL，交叉注意力+比例引导），在 EORSSD/ORSSD/ORSI-4199 三个数据集上取得 SOTA。

**[Rsonet Region-Guided Selective Optimization Network For Rgb-T Salient Object Det](segmentation/rsonet_region-guided_selective_optimization_network_for_rgb-t_salient_object_det.md)**

:   提出区域引导选择性优化网络 RSONet，通过两阶段（区域引导+显著性生成）解决 RGB 与热红外图像中显著区域不一致问题，利用相似度分数自动选择信息更准确的模态主导后续融合。

**[Sap Segment Any 4K Panorama](segmentation/sap_segment_any_4k_panorama.md)**

:   将 360° 全景图分割重新定义为透视视频分割问题，通过沿 zigzag 轨迹分解全景图为重叠 patch 序列并微调 SAM2 的 memory 模块，配合 183K 合成 4K 全景图的大规模训练，实现零样本全景分割 +17.2 mIoU 的提升。

**[Sgma Semantic-Guided Modality-Aware Segmentation For Remote Sensing With Incompl](segmentation/sgma_semantic-guided_modality-aware_segmentation_for_remote_sensing_with_incompl.md)**

:   提出SGMA框架，通过语义引导融合(SGF)模块构建全局语义原型估计模态鲁棒性并自适应加权融合，以及模态感知采样(MAS)模块动态优先训练脆弱模态，解决遥感不完整多模态分割中的模态不平衡、类内变化和跨模态异质性三大挑战。

**[Sparrow Learning Spatial Precision And Temporal Referential Consistency In Pixel](segmentation/sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)**

:   提出SPARROW框架，通过目标特定跟踪特征(TSF)和双提示(BOX+SEG)机制，解决视频MLLM中时序引用一致性差和首帧初始化不稳定的问题，在6个基准上对3个主流视频MLLM均取得一致提升。

**[Spatio-Semantic Expert Routing Architecture With Mixture-Of-Experts For Referrin](segmentation/spatio-semantic_expert_routing_architecture_with_mixture-of-experts_for_referrin.md)**

:   提出 SERA 框架，在预训练视觉语言模型中引入轻量级表达感知的混合专家（MoE）精细化，分别在 backbone 层（SERA-Adapter）和融合层（SERA-Fusion）进行专家路由，仅更新 <1% 参数即在参考图像分割基准上达到 SOTA。

---

## 🎬 视频理解 { #video_understanding }

**[Animateanything Consistent And Controllable Animation For Video Generation](video_understanding/animateanything_consistent_and_controllable_animation_for_video_generation.md)**

:   提出两阶段可控视频生成框架：第一阶段将不同控制信号（相机轨迹、用户拖拽标注、参考视频）统一转化为逐帧光流表示，第二阶段用统一光流引导基于DiT的视频扩散模型生成最终视频，并引入频域稳定模块抑制大运动下的闪烁问题。

**[Behaviorvlm Unified Finetuning-Free Behavioral Understanding With Vision-Languag](video_understanding/behaviorvlm_unified_finetuning-free_behavioral_understanding_with_vision-languag.md)**

:   提出 BehaviorVLM，一个统一的无需微调的视觉语言框架，通过多阶段结构化推理管线同时解决动物姿态估计和行为理解两大任务，仅需 3 帧人工标注即可实现可靠的关键点追踪，并通过深度嵌入聚类 + VLM 描述 + LLM 语义合并实现可解释的多动物行为分割。

**[Beyond Single-Sample Reliable Multi-Sample Distillation For Video Understanding](video_understanding/beyond_single-sample_reliable_multi-sample_distillation_for_video_understanding.md)**

:   提出 R-MSD（Reliable Multi-Sample Distillation），通过对每个输入采样多个教师响应并结合任务自适应质量匹配，解决视频 LVLM 黑盒蒸馏中单样本教师监督不可靠的问题，4B 学生模型在 VideoMME (+1.5%)、Video-MMMU (+3.2%)、MathVerse (+3.6%) 等基准上取得一致提升。

**[Bim-Vfi Bidirectional Motion Field-Guided Frame Interpolation For Video With Non](video_understanding/bim-vfi_bidirectional_motion_field-guided_frame_interpolation_for_video_with_non.md)**

**[Bimba Selective-Scan Compression For Long-Range Video Question Answering](video_understanding/bimba_selective-scan_compression_for_long-range_video_question_answering.md)**

**[Bootstrap Your Own Views Masked Ego-Exo Modeling For Fine-Grained View-Invariant](video_understanding/bootstrap_your_own_views_masked_ego-exo_modeling_for_fine-grained_view-invariant.md)**

:   通过掩码建模在自我中心和外部视角之间学习细粒度视图不变表示，无需配对标注即可从两种视角的关联中自监督学习

**[Can Text-To-Video Generation Help Video-Language Alignment](video_understanding/can_text-to-video_generation_help_video-language_alignment.md)**

:   本文首次探索利用文本到视频生成模型产生的合成视频来改善视频语言对齐（VLA）的时序理解，提出SynViTA方法通过对齐质量加权和语义一致性正则化来有效利用合成视频，在多个VLA基准上取得了平均性能提升。

**[Coarse Correspondences Boost Spatial-Temporal Reasoning In Multimodal Language M](video_understanding/coarse_correspondences_boost_spatial-temporal_reasoning_in_multimodal_language_m.md)**

:   本文提出Coarse Correspondences，一种轻量级的training-free视觉提示方法，通过在图像帧上叠加目标跟踪得到的粗粒度实例对应关系标记，显著增强MLLM的空间时序推理能力，在ScanQA上提升+20.5%、OpenEQA上+9.7%、EgoSchema上+6.0%和R2R导航上+11%。

**[Conmo Controllable Motion Disentanglement And Recomposition For Zero-Shot Motion](video_understanding/conmo_controllable_motion_disentanglement_and_recomposition_for_zero-shot_motion.md)**

:   ConMo提出了一种零样本运动迁移框架，通过将参考视频中的复合运动解耦为独立的主体运动和背景（相机）运动，再在目标视频生成时可控地重组这些运动，实现了多主体运动迁移、语义/形状变换、主体去除、相机运动模拟等多种应用，在运动保真度和文本对齐上显著超越现有方法。

**[Context-Enhanced Memory-Refined Transformer For Online Action Detection](video_understanding/context-enhanced_memory-refined_transformer_for_online_action_detection.md)**

:   本文揭示了现有在线动作检测（OAD）方法中的训练-推理不一致问题——短时记忆帧的不均衡上下文暴露和伪未来引入的非因果信息泄漏导致学习偏向中间帧——并提出CMeRT通过近过去上下文增强编码器和基于近未来的记忆精炼解码器来解决该问题，在THUMOS'14、CrossTask和EK100上实现SOTA。

**[Fc-Track Overlap-Aware Post-Association Correction For Online Multi-Object Track](video_understanding/fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)**

:   提出 FC-Track，一个轻量级的后关联校正框架，通过基于 IoA（Intersection over Area）的外观特征过滤和重叠 tracklet 对内的相似度比较，在线纠正因目标重叠导致的检测-轨迹错误匹配，将长期身份切换比例从 36.86% 降至 29.55%，同时在 MOT17/MOT20 上保持 SOTA 性能。

**[Reasoning Over Video Evaluating How Mllms Extract Integrate And Reconstruct Spat](video_understanding/reasoning_over_video_evaluating_how_mllms_extract_integrate_and_reconstruct_spat.md)**

:   提出 VAEX-Bench 基准，首次系统评估 MLLM 的"抽象时空推理"能力——不是从单帧提取信息，而是需要跨房间/跨时间整合观察来推断全局空间布局、跨场景计数等，发现所有 SOTA 模型（包括 GPT-5.2、Gemini-3 Pro）在抽象推理上表现远低于人类。

**[Semantic Satellite Communications For Synchronized Audiovisual Reconstruction](video_understanding/semantic_satellite_communications_for_synchronized_audiovisual_reconstruction.md)**

:   提出面向卫星场景的自适应多模态语义传输系统，通过双流生成架构（视频驱动音频 / 音频驱动视频）灵活切换、动态知识库更新机制和 LLM 决策代理，在极低带宽下实现高保真音视频同步重建。

**[Vcbench A Streaming Counting Benchmark For Spatial-Temporal State Maintenance In](video_understanding/vcbench_a_streaming_counting_benchmark_for_spatial-temporal_state_maintenance_in.md)**

:   VCBench 将计数重新定位为诊断视频模型"时空状态维护"能力的最小探针，提出了覆盖物体计数（当前状态/身份追踪）和事件计数（瞬时事件/周期活动）的 8 种子类别，通过沿时间线的流式多点查询观察模型预测轨迹，在 406 个视频/4576 个查询点上评估主流模型，发现当前模型在时空状态维护上仍存在显著缺陷。

**[Video Streaming Thinking Videollms Can Watch And Think Simultaneously](video_understanding/video_streaming_thinking_videollms_can_watch_and_think_simultaneously.md)**

:   提出 Video Streaming Thinking (VST) 范式，在视频播放过程中交替执行"看"和"想"——模型边接收视频帧边生成中间推理链，将 CoT 计算摊销到预查询阶段，从而在保持实时响应（0.56s QA延迟）的同时实现 StreamingBench 79.5% 的 SOTA。

**[World2Act Latent Action Post-Training Via Skill-Compositional World Models](video_understanding/world2act_latent_action_post-training_via_skill-compositional_world_models.md)**

:   World2Act 提出了一种基于潜在空间对齐的 VLA 后训练方法：通过对比学习将 World Model 的视频动态潜表示与 VLA 的动作表示对齐（而非在像素空间监督），并引入 LLM 驱动的技能分解流水线实现任意长度视频生成，在 RoboCasa 和 LIBERO 上以 50 条合成轨迹即达到 SOTA，真实世界提升 6.7%。

---

## 💬 LLM/NLP { #llm_nlp }

**[Breaking The Low-Rank Dilemma Of Linear Attention](llm_nlp/breaking_the_low-rank_dilemma_of_linear_attention.md)**

:   从理论上揭示线性注意力性能不及 Softmax 注意力的根本原因是输出特征的低秩问题，提出秩增强线性注意力（RALA），通过增强 KV 缓存秩和输出特征秩两种互补策略，在保持线性复杂度的同时追平甚至超越 Softmax 注意力的表现。

**[Building Vision Models Upon Heat Conduction](llm_nlp/building_vision_models_upon_heat_conduction.md)**

:   提出 vHeat 视觉 backbone，将图像 patch 建模为热源，利用物理热传导方程通过 DCT/IDCT 变换实现 $O(N^{1.5})$ 复杂度的信息传播，在 ImageNet-1K 上以 3 倍吞吐量和 80% 更少 GPU 显存达到 84.0% top-1 准确率。

**[Dora Sampling And Benchmarking For 3D Shape Variational Auto-Encoders](llm_nlp/dora_sampling_and_benchmarking_for_3d_shape_variational_auto-encoders.md)**

:   提出 Dora-VAE，通过 Sharp Edge Sampling (SES) 关注几何锐边区域、Dual Cross-Attention 分别处理均匀和显著采样点，以仅 1,280 个 latent codes（8× 小于 XCube-VAE 的 10,000+）实现更优的 3D 形状重建质量，同时建立了新的 Dora-Bench 评测基准。

**[Empowering Llms To Understand And Generate Complex Vector Graphics](llm_nlp/empowering_llms_to_understand_and_generate_complex_vector_graphics.md)**

:   通过引入55个SVG语义token、构建580k条指令微调数据集SVGX-SFT，使任意LLM能准确理解和生成复杂矢量图形，在文本对齐度和美观度上超越GPT-4o和Claude，推理速度比优化方法快50-150倍。

**[Imagine And Seek Improving Composed Image Retrieval With An Imagined Proxy](llm_nlp/imagine_and_seek_improving_composed_image_retrieval_with_an_imagined_proxy.md)**

:   提出IP-CIR方法，通过大语言模型生成"想象中的目标图像描述"作为代理，将组合图像检索(CIR)转化为标准图像检索问题，在CIRR和FashionIQ等基准上达到零样本SOTA。

**[Let Samples Speak Mitigating Spurious Correlation By Exploiting The Clusterness ](llm_nlp/let_samples_speak_mitigating_spurious_correlation_by_exploiting_the_clusterness_.md)**

:   提出NSF方法，通过利用样本在特征空间中的聚类特性自动识别依赖虚假特征的样本组，无需组标注即可训练出对虚假相关性鲁棒的分类器，最差组准确度显著超越ERM基线。

**[Lost In Translation Found In Context Sign Language Translation With Contextual C](llm_nlp/lost_in_translation_found_in_context_sign_language_translation_with_contextual_c.md)**

:   通过引入背景视频描述、历史翻译和伪词汇表三种上下文线索，结合Llama3-8B的LoRA微调，实现了连续手语到文本的精确翻译，在BOBSL数据集上相比SOTA提升40%以上。

**[Robust Message Embedding Via Attention Flow-Based Steganography](llm_nlp/robust_message_embedding_via_attention_flow-based_steganography.md)**

:   首次将Transformer注意力机制与归一化流结合，通过可逆QR码转换、可逆令牌融合和注意力仿射耦合块，实现高质量高容量的隐写，在打印-拍摄等极端物理扭曲下仍能准确还原消息。

**[ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model](llm_nlp/scamo_exploring_the_scaling_law_in_autoregressive_motion_generation_model.md)**

:   首次在人类动作生成领域探索和验证缩放律，提出ScaMo系统通过Motion FSQ-VAE和文本前缀自回归变换器，发现计算预算、模型/词汇表大小和数据之间的幂律关系，为动作生成的大规模训练提供理论指导。

**[SoftShadow: Leveraging Soft Masks for Penumbra-Aware Shadow Removal](llm_nlp/softshadow_leveraging_soft_masks_for_penumbra-aware_shadow_removal.md)**

:   提出SoftShadow框架，使用连续软掩码（而非二值硬掩码）精确捕捉半影区域，结合SAM+LoRA自适应和半影形成约束损失，在SRD和LRSS数据集上达到SOTA且无需外部掩码输入。

**[Spiking Transformer With Spatial-Temporal Attention](llm_nlp/spiking_transformer_with_spatial-temporal_attention.md)**

:   将空间-时间注意力机制融入脉冲Transformer架构，通过时空解耦的注意力设计和脉冲驱动的自注意机制，在保持SNN能效优势的同时缩小与ANN的性能差距，在多个视觉基准上达到SNN SOTA。

**[Staa-Snn Spatial-Temporal Attention Aggregator For Spiking Neural Networks](llm_nlp/staa-snn_spatial-temporal_attention_aggregator_for_spiking_neural_networks.md)**

:   通过在SNN中集成全局上下文自注意(GC)、位置编码(PE)、步骤注意(SA)和时间步随机退出(TSRD)四大模块，STAA-SNN在CIFAR-10/100和ImageNet上达到97.14%/82.05%/70.40%的SNN SOTA性能。

**[Test-Time Visual In-Context Tuning](llm_nlp/test-time_visual_in-context_tuning.md)**

:   首次系统研究VICL模型在分布偏移下的鲁棒性问题，提出VICT方法利用循环一致性自监督信号在测试时进行单样本微调，在6个视觉任务和15种图像破坏下显著改进Painter等VICL模型。

**[Vires Video Instance Repainting Via Sketch And Text Guided Generation](llm_nlp/vires_video_instance_repainting_via_sketch_and_text_guided_generation.md)**

:   提出ViReS框架，通过草图和文本双重引导实现视频中特定实例的重绘，利用时序注意力和实例掩码保持背景不变和时间一致性，在多种视频编辑场景下生成高质量结果。

---

## 🤖 机器人/具身智能 { #robotics }

**[3D-Mvp 3D Multiview Pretraining For Manipulation](robotics/3d-mvp_3d_multiview_pretraining_for_manipulation.md)**

:   提出3D-MVP，将Masked Autoencoder预训练从2D扩展到3D多视角设定——在Objaverse的200K个3D物体上预训练RVT的多视角Transformer编码器，下游微调后在RLBench上平均成功率从62.9%提升到67.5%，在COLOSSEUM上显著提升对纹理、大小、光照等环境变化的鲁棒性。

**[A Data-Centric Revisit Of Pre-Trained Vision Models For Robot Learning](robotics/a_data-centric_revisit_of_pre-trained_vision_models_for_robot_learning.md)**

:   通过系统评估发现DINO/iBOT在机器人任务上优于MAE但在非物体中心(NOC)数据上性能退化，原因是丧失了物体中心表示能力。提出SlotMIM方法，通过语义瓶颈（减少原型数量促进objectness涌现）和跨视图一致性正则+slot级对比学习，使模型在NOC数据上也能学到物体中心表示，仅用241K样本即超越用>1M样本的MVP/VC-1。

**[Asap Advancing Semantic Alignment Promotes Multi-Modal Manipulation De](robotics/asap_advancing_semantic_alignment_promotes_multi-modal_manipulation_de.md)**

:   提出ASAP框架，通过大模型辅助对齐(LMA)、篡改引导交叉注意力(MGCA)和补丁篡改建模(PMM)三个核心模块，系统性地推进图文语义对齐以提升多模态篡改检测与定位性能——在DGM4基准上AUC达94.38%，文本定位F1达76.52%，显著超越现有方法。

**[Asap Advancing Semantic Alignment Promotes Multi-Modal Manipulation Detecting An](robotics/asap_advancing_semantic_alignment_promotes_multi-modal_manipulation_detecting_an.md)**

**[Chapter-Llama Efficient Chaptering In Hour-Long Videos With Llms](robotics/chapter-llama_efficient_chaptering_in_hour-long_videos_with_llms.md)**

**[Drawer Digital Reconstruction And Articulation With Environment Realism](robotics/drawer_digital_reconstruction_and_articulation_with_environment_realism.md)**

:   提出 DRAWER 框架，从静态场景视频自动构建可交互数字孪生，结合 SDF + 高斯泼溅双场景表示实现高保真渲染和精细几何，支持铰接体识别与仿真、Unreal Engine 游戏创建、以及 real-to-sim-to-real 机器人策略迁移。

**[Expert Pyramid Tuning Efficient Parameter Fine-Tuning For Expertise-Driven Task ](robotics/expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)**

:   提出 Expert Pyramid Tuning (EPT)，将计算机视觉中的多尺度特征金字塔思想引入 LoRA-based MoE，通过共享元知识子空间 + 反卷积金字塔投影机制构建不同粒度的专家，实现更高效的多任务参数微调。

**[Influence Malleability In Linearized Attention Dual Implications Of Non-Converge](robotics/influence_malleability_in_linearized_attention_dual_implications_of_non-converge.md)**

:   通过 NTK 框架揭示线性化注意力机制不会收敛到无穷宽 NTK 极限（谱放大效应使 Gram 矩阵条件数立方化，需宽度 $m = \Omega(\kappa^6)$），并引入「影响可塑性」概念量化这一非收敛的双面后果：注意力比 ReLU 网络高 6-9 倍的可塑性既增强了任务适配能力，也加剧了对抗脆弱性。

**[Language-Grounded Decoupled Action Representation For Robotic Manipulation](robotics/language-grounded_decoupled_action_representation_for_robotic_manipulation.md)**

:   提出 LaDA，将 7-DoF 机器人动作解耦为平移/旋转/夹爪三类运动原语并与语言语义建立对应，通过软标签对比学习和自适应损失加权，以 1.3B 参数在 LIBERO 上达到 93.6% 平均成功率。

**[One Token Two Fates A Unified Framework Via Vision Token Manipulation Against Ml](robotics/one_token_two_fates_a_unified_framework_via_vision_token_manipulation_against_ml.md)**

:   提出首个统一的训练无关MLLM幻觉缓解框架，围绕vision token的双重角色——增强(SVC)与抑制(CRC)——在隐表示层协同操作，在LLaVA-1.5上POPE准确率提升约2%，仅增加1.06×推理延迟。

**[Panoaffordancenet Towards Holistic Affordance Grounding In 360 Indoor Environmen](robotics/panoaffordancenet_towards_holistic_affordance_grounding_in_360_indoor_environmen.md)**

:   提出PanoAffordanceNet——首个360°全景affordance grounding框架，通过失真感知频谱调制器(DASM)处理ERP纬度依赖畸变、全球面致密化头(OSDH)恢复稀疏激活为拓扑连续区域、多层级训练目标抑制语义漂移，并构建首个全景affordance数据集360-AGD，全面超越现有方法。

**[Sapave Towards Active Perception And Manipulation In Vision-Language-Action Mode](robotics/sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)**

:   SaPaVe 提出了一种端到端的主动操作框架，通过解耦相机运动和操作动作的 action space，采用自底向上的两阶段训练策略（先学语义相机控制，再联合优化），在 200K 语义相机运动数据集上训练主动感知先验，配合 3D 几何感知模块增强视角变化下的执行鲁棒性，在真实世界任务中比 GR00T N1 和 $\pi_0$ 分别高 31.25% 和 40% 成功率。

**[Sortscrews A Dataset And Baseline For Real-Time Screw Classification](robotics/sortscrews_a_dataset_and_baseline_for_real-time_screw_classification.md)**

:   提出SortScrews数据集——一个包含560张512×512 RGB图像、覆盖6类螺丝的工业分类数据集，配套可复用的数据采集流水线，并以迁移学习的EfficientNet-B0和ResNet-18作为基线，ResNet-18在该数据集上达到96.4%验证准确率。

**[Tinynav End-To-End Tinyml For Real-Time Autonomous Navigation On Microcontroller](robotics/tinynav_end-to-end_tinyml_for_real-time_autonomous_navigation_on_microcontroller.md)**

:   在 ESP32 微控制器上部署端到端量化 CNN，仅用 23k 参数和 ToF 深度相机实现 30ms 延迟的实时自主导航。

---

## 🧑 人体理解 { #human_understanding }

**[3D Face Reconstruction From Radar Images](human_understanding/3d_face_reconstruction_from_radar_images.md)**

:   首次从毫米波雷达图像进行3D人脸重建：用物理雷达渲染器生成合成数据集训练CNN编码器估计BFM参数，再通过学习一个可微分雷达渲染器构建model-based autoencoder，在合成数据上实现2.56mm平均点距精度，并可在推理时无监督优化参数。

**[Analyzing The Synthetic-To-Real Domain Gap In 3D Hand Pose Estimation](human_understanding/analyzing_the_synthetic-to-real_domain_gap_in_3d_hand_pose_estimation.md)**

:   首次系统研究3D手势估计中合成数据到真实数据的域差距，通过可控数据合成管线分解并分析了前臂、频谱统计、手势分布、物体遮挡四个关键因素的影响，证明合理整合这些因素后纯合成数据可达到与真实数据同等的精度。

**[Anomize Better Open Vocabulary Video Anomaly Detection](human_understanding/anomize_better_open_vocabulary_video_anomaly_detection.md)**

**[Breaking The Tuning Barrier Zero-Hyperparameters Yield Multi-Corner Analysis Via](human_understanding/breaking_the_tuning_barrier_zero-hyperparameters_yield_multi-corner_analysis_via.md)**

:   用预训练的Foundation Model（TabPFN）替代传统手工先验，实现零超参数调优的电路Yield Multi-Corner Analysis：冻结backbone做in-context learning，自动跨corner迁移知识，结合自动特征选择（1152D→48D），在SRAM benchmarks上达到SOTA精度（MRE低至0.11%）且验证成本降低10倍以上。

**[Chatgarment Garment Estimation Generation And Editing Via Large Language Models](human_understanding/chatgarment_garment_estimation_generation_and_editing_via_large_language_models.md)**

**[Co-Op Correspondence-Based Novel Object Pose Estimation](human_understanding/co-op_correspondence-based_novel_object_pose_estimation.md)**

**[Conformal Prediction For Zero-Shot Models](human_understanding/conformal_prediction_for_zero-shot_models.md)**

:   将保形预测（Conformal Prediction）应用于零样本模型，为 CLIP 等模型的预测提供有理论保证的不确定性量化和校准预测集

**[L2Gtx From Local To Global Time Series Explanations](human_understanding/l2gtx_from_local_to_global_time_series_explanations.md)**

:   L2GTX 提出一种完全模型无关的时间序列分类全局解释方法，通过聚合 LOMATCE 产生的参数化时间事件原语（PEPs）构建类级全局解释，在六个基准数据集上保持稳定的全局忠实度（R²）。

**[Mm-Condchain A Programmatically Verified Benchmark For Visually Grounded Deep Co](human_understanding/mm-condchain_a_programmatically_verified_benchmark_for_visually_grounded_deep_co.md)**

:   MM-CondChain 是首个针对视觉基础深层组合推理的 MLLM 基准，通过可验证程序中间表示（VPIR）自动构建多层条件链和链式硬负样本，最强模型仅获 53.33 Path F1，揭示深层组合推理是根本挑战。

**[Nbavatar Neural Billboards Avatars With Realistic Hand-Face Interaction](human_understanding/nbavatar_neural_billboards_avatars_with_realistic_hand-face_interaction.md)**

:   NBAvatar 提出 Neural Billboard 原语——将可学习平面几何原语与神经纹理延迟渲染结合，实现手脸交互场景下的照片级真实头部 avatar 渲染，在百万像素分辨率下 LPIPS 比 Gaussian 方法降低 30%。

**[Perceive What Matters Relevance-Driven Scheduling For Multimodal Streaming Perce](human_understanding/perceive_what_matters_relevance-driven_scheduling_for_multimodal_streaming_perce.md)**

:   提出一种面向人机协作的感知调度框架，基于信息增益和计算代价的权衡来选择性激活感知模块（目标检测/姿态估计），在流式感知场景下将计算延迟降低最多 27.52%，同时 MMPose 激活召回提升 72.73%。

**[Reference-Free Image Quality Assessment For Virtual Try-On Via Human Feedback](human_understanding/reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)**

:   提出 VTON-IQA，一个无参考的虚拟试穿图像质量评估框架，通过大规模人类标注基准 VTON-QBench（62,688 张试穿图 + 431,800 条标注）和 Interleaved Cross-Attention 模块实现与人类感知对齐的图像级质量预测。

**[Team Ras In 10Th Abaw Competition Multimodal Valence And Arousal Estimation Appr](human_understanding/team_ras_in_10th_abaw_competition_multimodal_valence_and_arousal_estimation_appr.md)**

:   提出结合面部（GRADA+Transformer）、行为描述（Qwen3-VL+Mamba）和音频（WavLM）三模态的连续情感估计方法，通过 Directed Cross-Modal MoE 和 Reliability-Aware Audio-Visual 两种融合策略在 Aff-Wild2 上达到 CCC 0.6576（dev）/ 0.62（test）。

---

## 🖼️ 图像恢复 { #image_restoration }

**[A Flag Decomposition For Hierarchical Datasets](image_restoration/a_flag_decomposition_for_hierarchical_datasets.md)**

:   提出Flag Decomposition（FD）——一种保持层次结构的矩阵分解方法，将具有嵌套列层次的数据矩阵分解为Stiefel坐标表示的flag（嵌套子空间序列）、块上三角矩阵和置换矩阵，在去噪、聚类和小样本学习任务上优于SVD等标准方法。

**[A Physics-Informed Blur Learning Framework For Imaging Systems](image_restoration/a_physics-informed_blur_learning_framework_for_imaging_systems.md)**

:   提出基于物理的 PSF 学习框架，设计新型波前基（每个基仅影响单一 SFR 方向）消除梯度冲突，结合课程学习（中心→边缘），无需镜头参数即可精确估计成像系统的空间变化 PSF。

**[A Regularization-Guided Equivariant Approach For Image Restoration](image_restoration/a_regularization-guided_equivariant_approach_for_image_restoration.md)**

**[Adversarial Diffusion Compression For Real-World Image Super-Resolution](image_restoration/adversarial_diffusion_compression_for_real-world_image_super-resolution.md)**

:   提出对抗扩散压缩（ADC）框架，将一步扩散模型 OSEDiff 蒸馏为精简的扩散-GAN 混合模型，实现 73% 推理时间压缩、78% 计算量削减、74% 参数缩减，同时保持生成质量，达到 34.79 FPS 实时超分。

**[Augmenting Perceptual Super-Resolution Via Image Quality Predictors](image_restoration/augmenting_perceptual_super-resolution_via_image_quality_predictors.md)**

:   利用无参考图像质量评估（NR-IQA）模型代替人工标注，通过加权采样和直接优化两种方式提升感知超分辨率的图像质量，在无需人工数据的条件下超越依赖人工反馈的 SOTA 方法。

**[Bf-Stvsr B-Splines And Fourier---Best Friends For High Fidelity Spatia](image_restoration/bf-stvsr_b-splines_and_fourier---best_friends_for_high_fidelity_spatia.md)**

:   提出 BF-STVSR 框架，用 B-spline Mapper 建模时间运动插值、Fourier Mapper 捕获空间高频细节，无需外部光流网络即可实现连续时空视频超分辨率的 SOTA 性能。

**[Bf-Stvsr B-Splines And Fourier---Best Friends For High Fidelity Spatial-Temporal](image_restoration/bf-stvsr_b-splines_and_fourier---best_friends_for_high_fidelity_spatial-temporal.md)**

:   提出 BF-STVSR，结合 B 样条映射器（时间平滑插值）和傅里叶映射器（空间高频捕获）实现连续时空视频超分辨率，完全无需预训练光流网络（RAFT），在 GoPro 数据集上 PSNR 达 30.22dB，FLOPs 在所有方法中最低。

**[Classic Video Denoising In A Machine Learning World Robust Fast And Controllable](image_restoration/classic_video_denoising_in_a_machine_learning_world_robust_fast_and_controllable.md)**

:   重新审视经典视频去噪方法并与现代ML工具结合，实现鲁棒、快速且噪声级别可控的视频去噪

**[Polishing The Sky Wide-Field And High-Dynamic Range Interferometric Image Recons](image_restoration/polishing_the_sky_wide-field_and_high-dynamic_range_interferometric_image_recons.md)**

:   在 POLISH 框架基础上提出 POLISH+/++，通过**分块训练+拼接推理**和**arcsinh 非线性变换**两项改进，使深度学习方法首次能处理宽视场（12,960×12,960 像素）、高动态范围（~10⁶）的射电干涉成像，并展示了超分辨率对强引力透镜发现的 10× 提升潜力。

**[Towards Universal Computational Aberration Correction In Photographic Cameras A ](image_restoration/towards_universal_computational_aberration_correction_in_photographic_cameras_a_.md)**

:   扩展 OptiFusion 自动设计 120 种多样化镜头，提出 ODE 综合评估指标和大规模 benchmark，系统对比 24 种算法，发现 CNN 模型在像差校正中提供最佳速度-精度权衡，反直觉地超越 Transformer。

**[Variational Garrote For Sparse Inverse Problems](image_restoration/variational_garrote_for_sparse_inverse_problems.md)**

:   系统比较 $\ell_1$ 正则化 (LASSO) 与 Variational Garrote (VG, 概率 $\ell_0$ 近似) 在信号重采样、去噪和稀疏视角 CT 重建三种逆问题上的表现，发现 VG 在强欠定情况下（采样率低/角度稀疏）通常获得更低的泛化误差，因为 spike-and-slab 先验与真实稀疏分布更匹配。

---

## 📦 模型压缩 { #model_compression }

**[Adapter Merging With Centroid Prototype Mapping For Scalable Class-Incremental L](model_compression/adapter_merging_with_centroid_prototype_mapping_for_scalable_class-incremental_l.md)**

:   提出ACMap框架，通过将每个任务独立训练的adapter增量平均合并为单一adapter（保持O(1)推理复杂度），结合centroid prototype mapping对齐旧任务原型在新子空间中的表示，在5个基准上实现与SOTA EASE相当的精度同时推理速度快39倍。

**[Alternating Gradient Flow Utility A Unified Metric For Structural Pruning And Dy](model_compression/alternating_gradient_flow_utility_a_unified_metric_for_structural_pruning_and_dy.md)**

:   提出基于交替梯度流(AGF)的统一效用度量，将特征空间总变差作为结构化剪枝指标，并结合置信度级联路由实现离线拓扑构建与在线动态推理的解耦，在ImageNet-1K极端压缩下避免传统指标导致的结构崩溃，在ImageNet-100动态推理中以0.92x计算代价匹配全模型精度。

**[An Fpga Implementation Of Displacement Vector Search For Intra Pattern Copy In J](model_compression/an_fpga_implementation_of_displacement_vector_search_for_intra_pattern_copy_in_j.md)**

:   首次提出JPEG XS帧内模式复制(IPC)中位移向量(DV)搜索模块的FPGA架构实现，采用四级流水线设计和优化的存储组织方式，在Xilinx Artix-7上实现38.3 Mpixels/s吞吐量和277 mW功耗，为IPC实际硬件部署和ASIC转化奠定基础。

**[Arche Autoregressive Residual Compression With Hyperprior And Excitation](model_compression/arche_autoregressive_residual_compression_with_hyperprior_and_excitation.md)**

:   提出ARCHE端到端学习型图像压缩框架，在统一概率架构中整合分层Hyperprior、掩码空间自回归上下文、通道条件化和SE激励通道重校准，无需Transformer或循环组件，在Kodak上相对Ballé基线BD-Rate降低约48%，相对VVC Intra降低约5.6%，仅95M参数和222ms解码时间。

**[Autossvh Exploring Automated Frame Sampling For Efficient Self-Supervised Video H](model_compression/autossvh_exploring_automated_frame_sampling_for_efficient_self-supervised_video_h.md)**

:   提出AutoSSVH方法，通过对抗式自动帧采样网络（Grade-Net）选择最具挑战性的帧子集作为训练信号，并设计P2Set（Point-to-Set）哈希对比学习范式，实现了高效的自监督视频哈希检索，在UCF101和HMDB51上大幅超越现有方法。

**[Bhvit Binarized Hybrid Vision Transformer](model_compression/bhvit_binarized_hybrid_vision_transformer.md)**

:   针对 ViT 二值化性能严重下降的问题，提出专为二值化设计的混合 ViT 架构 BHViT，包含多尺度分组空洞卷积 token mixer、量化分解注意力矩阵二值化、shift 增强的 MLP 和正则化损失，在 ImageNet-1K 上达到 1-bit 二值化模型的 SOTA 性能。

**[Charm The Missing Piece In Vit Fine-Tuning For Image Aesthetic Assessment](model_compression/charm_the_missing_piece_in_vit_fine-tuning_for_image_aesthetic_assessment.md)**

**[Cl-Lora Continual Low-Rank Adaptation For Rehearsal-Free Class-Incremental Learn](model_compression/cl-lora_continual_low-rank_adaptation_for_rehearsal-free_class-incremental_learn.md)**

:   提出 CL-LoRA，设计双适配器架构（任务共享 + 任务特定 LoRA），结合知识蒸馏与梯度重分配以及可学习块级权重，在仅 0.3% 可训练参数下实现 SOTA 持续学习性能。

**[Coa Towards Real Image Dehazing Via Compression-And-Adaptation](model_compression/coa_towards_real_image_dehazing_via_compression-and-adaptation.md)**

:   提出压缩-适应（CoA）框架实现实际图像去雾：先在合成数据上训练大模型，然后压缩+适应到真实域，平衡性能和部署效率

**[Geochemad Benchmarking Unsupervised Geochemical Anomaly Detection For Mineral Ex](model_compression/geochemad_benchmarking_unsupervised_geochemical_anomaly_detection_for_mineral_ex.md)**

:   提出 GeoChemAD 开源基准数据集（8 个子集，覆盖多区域/多采样源/多目标元素）和 GeoChemFormer 框架，通过空间上下文自监督预训练和元素依赖建模实现无监督地球化学异常检测，在所有子集上取得最优 AUC。

**[Hiap A Multi-Granular Stochastic Auto-Pruning Framework For Vision Transformers](model_compression/hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)**

:   HiAP 提出了一种多粒度自动剪枝框架，通过在宏观（attention heads、FFN blocks）和微观（intra-head dimensions、FFN neurons）两级部署可学习 Gumbel-Sigmoid 门控，在单阶段端到端训练中自动发现最优子网络，无需手工重要性排序或后处理阈值。

---

## 🦾 LLM Agent { #llm_agent }

**[Ata Adaptive Transformation Agent For Text-Guided Subject-Position Variable Back](llm_agent/ata_adaptive_transformation_agent_for_text-guided_subject-position_variable_back.md)**

:   提出 ATA（Adaptive Transformation Agent），解决文本引导的主体位置可变背景修复任务，通过 PosAgent Block 自适应预测位移、Reverse Displacement Transform 模块和 Position Switch Embedding，在保持修复质量的同时实现主体位置的灵活调整。

**[Feature4X Bridging Any Monocular Video To 4D Agentic Ai With Versatile Gaussian ](llm_agent/feature4x_bridging_any_monocular_video_to_4d_agentic_ai_with_versatile_gaussian_.md)**

:   提出 Feature4X，一个通用框架，从任意单目视频通过动态优化策略将多种 2D 视觉基础模型（SAM2、InternVideo2 等）的功能蒸馏到统一的 4D 高斯特征场中，首次实现基于 Gaussian Splatting 的视频基础模型 4D 特征提升，支持新视角下的 segment anything、几何/外观编辑和自由形式 VQA。

**[Gui-Xplore Empowering Generalizable Gui Agents With One Exploration](llm_agent/gui-xplore_empowering_generalizable_gui_agents_with_one_exploration.md)**

:   提出 GUI-Xplore 数据集（312 个应用、32K+ QA 对、五层级任务）和 Xplore-Agent 框架（Action-aware GUI 建模 + GUI Transition Graph 推理），通过模拟"先探索再推理"的人类策略，在陌生应用上比 SOTA GUI Agent 提升约 10% StepSR。

**[Rl-Rc-Dot A Block-Level Rl Agent For Task-Aware Video Compression](llm_agent/rl-rc-dot_a_block-level_rl_agent_for_task-aware_video_compression.md)**

:   提出 RL-RC-DoT，一个基于强化学习的宏块级量化参数（QP）控制 agent，用于任务感知视频压缩。通过将 QP 选择建模为 RL 的顺序决策问题，agent 学习在给定码率约束下为任务相关区域分配更多码率，在车辆检测和 ROI 显著性编码两个任务上显著提升性能。关键优势在于推理时不需要运行下游任务模型，适合边缘设备部署。

**[Sceneassistant A Visual Feedback Agent For Open-Vocabulary 3D Scene Generation](llm_agent/sceneassistant_a_visual_feedback_agent_for_open-vocabulary_3d_scene_generation.md)**

:   提出 SceneAssistant，一个基于视觉反馈的闭环 agentic 框架，通过为 VLM 设计一套功能完备的 Action API（13个原子操作覆盖物体增删、6DoF空间操作、相机控制），让 VLM 以 ReAct 范式迭代生成开放词汇的 3D 场景，在室内（偏好率61.25%）和开放域（偏好率65.00%）场景中均大幅优于 Holodeck 和 SceneWeaver。

**[Sketchtopia A Dataset And Foundational Agents For Benchmarking Asynchronous Mult](llm_agent/sketchtopia_a_dataset_and_foundational_agents_for_benchmarking_asynchronous_mult.md)**

:   提出 Sketchtopia 大规模数据集（20K+ 游戏会话、263K 草图、916 名玩家）和三组件 Agent 框架（ActionDecider + DRAWBOT + GUESSBOT），在 Pictionary 场景下研究异步、目标驱动的多模态协作通信，引入 AAO/FRS/MATS 三个新评估指标。

**[Spiritsight Agent Advanced Gui Agent With One Look](llm_agent/spiritsight_agent_advanced_gui_agent_with_one_look.md)**

:   提出 SpiritSight，一个基于视觉的端到端 GUI agent，通过 573 万样本的多层级数据集 GUI-Lasagne 和 Universal Block Parsing (UBP) 方法解决动态高分辨率输入的定位歧义，SpiritSight-8B 在 Multimodal-Mind2Web 上非候选元素设置下 Step SR 达 52.7%，全面超越所有视觉/语言/混合方法。

**[Tango Training-Free Embodied Ai Agents For Open-World Tasks](llm_agent/tango_training-free_embodied_ai_agents_for_open-world_tasks.md)**

:   提出 TANGO，通过 LLM 的程序组合能力编排两个最小化的导航基础原语（PointGoal Navigation + 记忆驱动探索策略），无需任何任务特定训练，仅用 few-shot 示例即可在 Open-Set ObjectGoal Navigation、Multi-Modal Lifelong Navigation 和 Open Embodied QA 三个不同的具身 AI 任务上达到 SOTA，体现了"最小原语集 + LLM 组合"的通用性。

**[V-Stylist Video Stylization Via Collaboration And Reflection Of Mllm Agents](llm_agent/v-stylist_video_stylization_via_collaboration_and_reflection_of_mllm_agents.md)**

:   提出 V-Stylist，一个基于 MLLM 多 agent 协作和反思的视频风格化系统，通过 Video Parser（视频分镜）、Style Parser（风格树搜索）和 Style Artist（多轮自反思渲染）三个角色协作，在复杂转场视频和开放风格描述上实现 SOTA，整体指标超越 FRESCO 6.05%。

**[Visual Agentic Ai For Spatial Reasoning With A Dynamic Api](llm_agent/visual_agentic_ai_for_spatial_reasoning_with_a_dynamic_api.md)**

:   提出 VADAR，一种 agentic 程序合成方法用于 3D 空间推理。多个 LLM agent 协作生成 Pythonic API 并在求解过程中动态扩展新函数来解决常见子问题，克服了 VisProg/ViperGPT 等先前方法依赖静态人工定义 API 的局限。同时引入涉及多步空间定位和推理的新 benchmark，在 3D 理解任务上超越现有零样本方法。

---

## 🛡️ AI安全 { #ai_safety }

**[A Simple Data Augmentation For Feature Distribution Skewed Federated Learning](ai_safety/a_simple_data_augmentation_for_feature_distribution_skewed_federated_learning.md)**

:   提出FedRDN——一种极其简单的联邦学习数据增强方法，在训练时随机使用其他客户端的通道级均值/标准差做数据归一化（而非固定用本地统计），仅需几行代码即可显著缓解特征分布偏移问题，在多种FL方法上一致提升性能。

**[Data-Free Universal Adversarial Perturbation With Pseudo-Semantic Prior](ai_safety/data-free_universal_adversarial_perturbation_with_pseudo-semantic_prior.md)**

:   提出 PSP-UAP，一种无需训练数据的通用对抗扰动生成方法，通过从 UAP 自身提取伪语义先验、输入变换增强和样本重加权策略，在白盒平均 89.95% 愚弄率、黑盒也大幅超越现有方法，且无需任何训练数据。

**[Deal Data-Efficient Adversarial Learning For High-Quality Infrared Imaging](ai_safety/deal_data-efficient_adversarial_learning_for_high-quality_infrared_imaging.md)**

:   提出 DEAL（Data-Efficient Adversarial Learning），一种仅需 50 张清晰红外图像训练的对抗学习框架，通过动态对抗退化合成和双通道交互网络（Scale Transform + Spiking Neurons），以 0.96M 超轻量参数同时处理条纹噪声、低分辨率和低对比度三种红外退化。

**[Dede Detecting Backdoor Samples For Ssl Encoders Via Decoders](ai_safety/dede_detecting_backdoor_samples_for_ssl_encoders_via_decoders.md)**

**[Detecting Backdoor Attacks In Federated Learning Via Direction Alignment Inspect](ai_safety/detecting_backdoor_attacks_in_federated_learning_via_direction_alignment_inspect.md)**

:   提出 AlignIns 防御方法，通过双粒度方向对齐检测（全局方向 + 细粒度符号分析）识别联邦学习中的恶意模型更新，在 IID 和 non-IID 设置下均优于现有防御方法。

**[Dynamic Integration Of Task-Specific Adapters For Class Incremental Learning](ai_safety/dynamic_integration_of_task-specific_adapters_for_class_incremental_learning.md)**

:   通过动态集成任务特定适配器实现类增量学习，每个任务训练轻量适配器，推理时动态选择和组合相关适配器

**[Lyapunov Stable Graph Neural Flow](ai_safety/lyapunov_stable_graph_neural_flow.md)**

:   将 Lyapunov 稳定性理论（整数阶和分数阶）与图神经流集成，通过可学习 Lyapunov 函数和投影机制将 GNN 特征动态约束在稳定空间中，首次为图神经流提供可证明的对抗鲁棒性保证，且与对抗训练正交可叠加。

**[Neural Gate Mitigating Privacy Risks In Lvlms Via Neuron-Level Gradient Gating](ai_safety/neural_gate_mitigating_privacy_risks_in_lvlms_via_neuron-level_gradient_gating.md)**

:   Neural Gate 发现 LVLM 中隐私相关神经元具有强跨样本不一致性——仅约 10% 的神经元一致性编码隐私信号。基于此发现，提出神经元级梯度门控编辑：仅对强一致性隐私神经元施加梯度更新，在 MiniGPT 上将 Safety EtA 从 0.48 提升至 0.89，同时 Utility 保持不降。

**[Rethinking Vlms For Image Forgery Detection And Localization](ai_safety/rethinking_vlms_for_image_forgery_detection_and_localization.md)**

:   提出 IFDL-VLM，揭示 VLM 先验对伪造检测/定位几乎无益，通过将检测/定位与语言解释解耦的两阶段框架，用 ViT+SAM 专家模型做检测定位、再将定位 mask 作为辅助输入增强 VLM 训练以生成可解释文字说明。

---

## 🎯 目标检测 { #object_detection }

**[A Bias-Free Training Paradigm For More General Ai-Generated Image Detection](object_detection/a_bias-free_training_paradigm_for_more_general_ai-generated_image_detection.md)**

:   提出B-Free训练范式——通过stable diffusion的自条件重构从真实图像生成语义对齐的假图，结合inpainting内容增强，消除格式/内容/分辨率等偏差，使检测器聚焦于生成器特有的伪影痕迹，在27种生成模型（含FLUX、SD 3.5等最新模型）上泛化AUC>99%，balanced accuracy达95.2%。

**[Abra Teleporting Fine-Tuned Knowledge Across Domains For Open-Vocabulary Object ](object_detection/abra_teleporting_fine-tuned_knowledge_across_domains_for_open-vocabulary_object_.md)**

:   提出 ABRA（Aligned Basis Relocation for Adaptation），通过在权重空间中进行 SVD 分解与正交旋转对齐，将源域的类别特定检测知识"传送"到无标注数据的目标域，实现零样本跨域目标检测。

**[Any6D Model-Free 6D Pose Estimation Of Novel Objects](object_detection/any6d_model-free_6d_pose_estimation_of_novel_objects.md)**

:   提出 Any6D 框架，仅从单张 RGB-D 锚点图像即可估计未知物体的 6D 位姿和尺寸，通过 InstantMesh 3D 重建 + 朝向包围盒粗对齐 + 联合尺寸-位姿精细化，在 HO3D 上 ADD-S 达 98.7% 远超 GEDI 的 71.9%。

**[Boosting Domain Incremental Learning Selecting The Optimal Parameters Is All You](object_detection/boosting_domain_incremental_learning_selecting_the_optimal_parameters_is_all_you.md)**

:   发现在域增量学习中选择最优参数子集比微调全部参数更有效，提出参数选择策略解决域增量目标检测的灾难性遗忘

**[Dreamvideo-Omni Omni-Motion Controlled Multi-Subject Video Customization With La](object_detection/dreamvideo-omni_omni-motion_controlled_multi-subject_video_customization_with_la.md)**

:   提出 DreamVideo-Omni，通过渐进式两阶段训练范式（Omni-Motion SFT + Latent Identity Reward Feedback Learning），在统一的 DiT 框架中实现多主体定制与全运动控制（全局 bbox + 局部轨迹 + 相机运动）的协同生成。

**[Mitigating Memorization In Text-To-Image Diffusion Via Region-Aware Prompt Augme](object_detection/mitigating_memorization_in_text-to-image_diffusion_via_region-aware_prompt_augme.md)**

:   提出 RAPTA（训练时基于目标检测的区域感知 prompt 变体增强）和 ADMCD（推理时三流注意力融合的多模态复制检测），从缓解和检测两个角度端到端地应对文生图扩散模型的训练数据记忆化问题。

**[Mokus Leveraging Cross-Modal Knowledge Transfer For Knowledge-Aware Concept Cust](object_detection/mokus_leveraging_cross-modal_knowledge_transfer_for_knowledge-aware_concept_cust.md)**

:   提出 MoKus 框架，发现并利用"跨模态知识迁移"现象——在 LLM 文本编码器中更新知识会自动传递到视觉生成端——实现知识感知的概念定制，两阶段设计：先学视觉锚点表示，再秒级更新文本知识绑定。

**[Small Target Detection Based On Mask-Enhanced Attention Fusion Of Visible And In](object_detection/small_target_detection_based_on_mask-enhanced_attention_fusion_of_visible_and_in.md)**

:   提出 ESM-YOLO+，一种轻量级可见光-红外融合网络，通过 MEAF 模块（可学习空间掩码+空间注意力的像素级融合）和训练时结构表示增强（SR，推理时无开销的超分辅助监督），在 VEDAI 上达到 84.71% mAP 同时参数量仅 5.1M（减少 93.6%）。

---

## 🔄 自监督/表示学习 { #self_supervised }

**[Autossvh Exploring Automated Frame Sampling For Efficient Self-Supervised Video ](self_supervised/autossvh_exploring_automated_frame_sampling_for_efficient_self-supervised_video_.md)**

**[Boss A Best-Of-Strategies Selector As An Oracle For Deep Active Learning](self_supervised/boss_a_best-of-strategies_selector_as_an_oracle_for_deep_active_learning.md)**

:   提出BoSS——一种可扩展的主动学习oracle策略，通过集成多种选择策略生成候选批次、冻结backbone仅重训最后一层来评估性能增益，选择最优批次；在ImageNet等大规模数据集上首次展示了oracle性能，揭示SOTA主动学习策略仍有显著提升空间。

**[Chexworld Exploring Image World Modeling For Radiograph Representation Learning](self_supervised/chexworld_exploring_image_world_modeling_for_radiograph_representation_learning.md)**

**[Do Your Best And Get Enough Rest For Continual Learning](self_supervised/do_your_best_and_get_enough_rest_for_continual_learning.md)**

:   受Ebbinghaus遗忘曲线理论启发，提出View-Batch Model(VBM)——通过将batch中多个不同样本替换为同一样本的多个增强视图（replay），延长回忆间隔V倍至最优范围，同时用one-to-many KL散度自监督损失从单样本中学习更多知识（do your best），作为drop-in替代方案在多种持续学习方法上一致提升性能。

**[Escaping Platos Cave Towards The Alignment Of 3D And Text Latent Spaces](self_supervised/escaping_platos_cave_towards_the_alignment_of_3d_and_text_latent_spaces.md)**

**[Few-Shot Implicit Function Generation Via Equivariance](self_supervised/few-shot_implicit_function_generation_via_equivariance.md)**

:   通过等变性约束从少量样本生成隐式函数（NeRF/SDF），利用对称性先验减少对数据的需求

**[Representation Learning For Spatiotemporal Physical Systems](self_supervised/representation_learning_for_spatiotemporal_physical_systems.md)**

:   系统评估通用自监督方法在时空物理系统上学习物理相关表征的能力，发现在潜空间做预测的 JEPA 显著优于像素级重建的 MAE 和自回归模型，接近专用物理建模方法 DISCO。

**[Text-Phase Synergy Network With Dual Priors For Unsupervised Cross-Domain Image ](self_supervised/text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)**

:   提出 TPSNet，利用文本-相位双先验解决无监督跨域图像检索：域提示（text prior）提供比伪标签更精确的语义监督，相位特征（phase prior）实现保持语义的域不变对齐，两者通过交叉注意力协同融合。

---

## ⚖️ 对齐/RLHF { #llm_alignment }

**[Bases Of Steerable Kernels For Equivariant Cnns From 2D Rotations To The Lorentz](llm_alignment/bases_of_steerable_kernels_for_equivariant_cnns_from_2d_rotations_to_the_lorentz.md)**

:   提出一种求解可转向等变 CNN 核约束方程的替代方法，通过在不动点处求解更简单的不变性条件再"转向"到任意点，绕过了计算 Clebsch-Gordan 系数的需要，为 SO(2)、O(2)、SO(3)、O(3) 及 Lorentz 群给出了显式的核基底公式。

**[Boost Your Human Image Generation Model Via Direct Preference Optimization](llm_alignment/boost_your_human_image_generation_model_via_direct_preference_optimization.md)**

:   提出 HG-DPO，以真实人像作为 DPO 的 winning image（而非生成图像对）+ 三阶段课程学习（Easy/Normal/Hard）渐进弥合生成-真实图像分布 gap + 统计匹配损失解决色偏，FID 从 37.34 降至 29.41（-21.4%），CI-Q 0.906→0.934，win-rate 超越 Diffusion-DPO 达 99.97%。

**[Continual Sft Matches Multimodal Rlhf With Negative Supervision](llm_alignment/continual_sft_matches_multimodal_rlhf_with_negative_supervision.md)**

:   通过梯度分析发现多模态 RLHF 相比持续 SFT 的核心优势在于 rejected response 中的负监督信号，据此提出 nSFT 方法，用 LLM 从拒绝回复中提取错误信息并构造纠正性对话数据，仅用 SFT loss 就能匹配甚至超越 DPO/PPO 等 RLHF 方法，且只需 1 个模型，显存效率大幅提升。

**[Curriculum Direct Preference Optimization For Diffusion And Consistency Models](llm_alignment/curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)**

:   首次将课程学习引入 DPO 并首次将 DPO 适配到一致性模型，通过从"容易区分的偏好对"到"难以区分的偏好对"渐进训练，在文本对齐、美学和人类偏好上全面超越标准 DPO 和 DDPO，且仅需 1/10 训练数据量。

**[Jailbreaking The Non-Transferable Barrier Via Test-Time Data Disguising](llm_alignment/jailbreaking_the_non-transferable_barrier_via_test-time_data_disguising.md)**

:   提出 JailNTL，首个针对 Non-Transferable Learning (NTL) 模型的黑盒攻击方法，通过测试时数据伪装将未授权域的数据"变装"为授权域的数据，仅用 1% 授权样本即可将未授权域准确率提升最高 55.7%，无需修改模型。

**[Physmodpo Physically-Plausible Humanoid Motion With Preference Optimization](llm_alignment/physmodpo_physically-plausible_humanoid_motion_with_preference_optimization.md)**

:   提出 PhysMoDPO，将 Direct Preference Optimization 应用于文本驱动的人体运动生成，通过将全身控制器（WBC）集成到训练 pipeline 中计算基于物理的奖励来构造偏好数据，使生成运动同时满足物理约束和文本指令，并在 Unitree G1 机器人上实现零样本部署。

**[Task Preference Optimization Improving Multimodal Large Language Models With Vis](llm_alignment/task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)**

:   提出 Task Preference Optimization（TPO），通过可学习的任务 token 将视觉任务专用头（区域定位/时序定位/分割）接入 MLLM，利用视觉任务标注作为"任务偏好"反向优化 MLLM，在不损害对话能力的前提下大幅提升细粒度视觉理解，VideoChat 基线上平均提升 14.6%。

---

## 📚 预训练/数据 { #llm_pretraining }

**[3D Prior Is All You Need Cross-Task Few-Shot 2D Gaze Estimation](llm_pretraining/3d_prior_is_all_you_need_cross-task_few-shot_2d_gaze_estimation.md)**

:   提出跨任务少样本2D视线估计——利用预训练3D视线模型作为先验，通过**基于物理的可微投影模块**（6个可学习屏幕参数）将3D视线方向投影到2D屏幕坐标，仅需10张标注图像即可在未知设备上适配2D视线估计，在MPIIGaze/EVE/GazeCapture上比EFE和IVGaze提升超25%。

**[A Unified Framework For Heterogeneous Semi-Supervised Learning](llm_pretraining/a_unified_framework_for_heterogeneous_semi-supervised_learning.md)**

:   提出异构半监督学习(HSSL)新问题设定——标记数据和无标记数据来自不同分布的域，目标是训练能在两个域上都泛化的模型；通过将C类问题扩展为2C类分类（每个域的同一语义类视为不同类），结合WMA伪标签、跨域原型对齐和渐进式跨域Mixup三个组件统一解决。

**[Hsemotion Team At Abaw-10 Competition Facial Expression Recognition Valence-Arou](llm_pretraining/hsemotion_team_at_abaw-10_competition_facial_expression_recognition_valence-arou.md)**

:   HSEmotion 团队在 ABAW-10 竞赛中提出了一个轻量级 pipeline：用预训练 EfficientNet 提取面部 embedding，结合 MLP + GLA（Generalized Logit Adjustment）+ 滑窗平滑，在四项任务（EXPR/VA/AU/VD）上均大幅超过官方 baseline，其中暴力检测任务使用 ConvNeXt-T + TCN 达到 0.783 macro F1。

**[Mxnorm Reusing Mxfp Block Scales For Efficient Tensor Normalisation](llm_pretraining/mxnorm_reusing_mxfp_block_scales_for_efficient_tensor_normalisation.md)**

:   MXNorm 提出复用 MXFP 量化过程中已计算的 block absmax 来近似 RMS，将归一化与 MX 量化融合为单次统计收集操作，实现 RMSNorm 的 drop-in 替换，在 Llama 3 8B 预训练中保持训练精度的同时获得最高 2.4× 的 kernel 加速。

**[The Scene Language Representing Scenes With Programs Words And Embeddings](llm_pretraining/the_scene_language_representing_scenes_with_programs_words_and_embeddings.md)**

:   提出 Scene Language——一种用程序（P, 编码层级结构）+ 词语（W, 语义类别）+ 嵌入（Z, 视觉身份）三元组 $\Phi(s)=(W,P,Z)$ 表示视觉场景的新范式，通过 Claude 3.5 Sonnet 的 training-free 推理从文本/图像输入生成场景表示，支持传统/神经/混合渲染，在 3D/4D 场景生成质量和可控编辑上超越场景图等现有表示。

---

## 🛰️ 遥感 { #remote_sensing }

**[Dense Dispersed Structured Light For Hyperspectral 3D Imaging Of Dynamic Scenes](remote_sensing/dense_dispersed_structured_light_for_hyperspectral_3d_imaging_of_dynamic_scenes.md)**

:   提出 Dense Dispersed Structured Light（DDSL）方法，利用廉价衍射光栅薄膜（<\$20）+ 立体 RGB 相机 + RGB 投影仪，设计光谱复用 DDSL 图案大幅减少所需投影帧数，实现 6.6fps 实时高光谱 3D 成像，光谱分辨率 15.5nm FWHM，深度误差 4mm。

**[Hierarchical Dual-Change Collaborative Learning For Uav Scene Change Captioning](remote_sensing/hierarchical_dual-change_collaborative_learning_for_uav_scene_change_captioning.md)**

:   提出 UAV 场景变化描述（UAV-SCC）新任务及 HDC-CL 框架，通过动态自适应布局 Transformer 建模移动视角下的图像对重叠/非重叠区域，结合层级跨模态方向一致性校准增强视角偏移方向感知，并构建了专用基准数据集。

**[Joint And Streamwise Distributed Mimo Satellite Communications With Multi-Antenn](remote_sensing/joint_and_streamwise_distributed_mimo_satellite_communications_with_multi-antenn.md)**

:   研究多 LEO 卫星联合服务多天线地面用户的分布式 MIMO 下行通信，提出联合传输与流式传输两种模式：前者通过 WMMSE 迭代优化预编码器最大化和频谱效率，后者通过匈牙利算法的流-卫星关联减少前传开销，实现性能与前传负载的灵活权衡。

**[Metaspectra A Compact Broadband Metasurface Camera For Snapshot Hyperspectral Im](remote_sensing/metaspectra_a_compact_broadband_metasurface_camera_for_snapshot_hyperspectral_im.md)**

:   提出 MetaSpectra+，一种基于超表面-折射混合光学的紧凑多功能相机，通过双层超表面独立控制各通道色散/曝光/偏振，在约 250nm 可见光带宽内实现快照式高光谱+HDR 或高光谱+偏振联合成像，重建精度在基准数据集上达到 SOTA。

**[Think And Answer Me Benchmarking And Exploring Multi-Entity Reasoning Grounding ](remote_sensing/think_and_answer_me_benchmarking_and_exploring_multi-entity_reasoning_grounding_.md)**

:   构建遥感多实体推理定位基准 ME-RSRG（首个显式标注主体-客体角色的遥感定位数据集），提出 Entity-Aware Reasoning (EAR) 框架，结合 SFT 冷启动与实体感知奖励驱动的 GRPO 优化，实现结构化推理链输出和主-客体联合定位，Qwen2.5-VL 系列在 EAR 优化后 mAcc@0.5 提升超 10%。

---

## 🎵 音频/语音 { #audio_speech }

**[Contextual Ad Narration With Interleaved Multimodal Sequence](audio_speech/contextual_ad_narration_with_interleaved_multimodal_sequence.md)**

:   提出 Uni-AD 统一框架，以交错多模态序列（视频特征+文本+角色库+上下文）作为输入，通过视觉映射网络对齐特征 + 角色精化模块识别主要角色 + 对比损失增强上下文一致性，在 MAD-eval-Named 上达到 SOTA。

**[Crab A Unified Audio-Visual Scene Understanding Model With Explicit Cooperation](audio_speech/crab_a_unified_audio-visual_scene_understanding_model_with_explicit_cooperation.md)**

:   提出统一音视频场景理解模型 Crab，通过构建带显式推理过程的 AV-UIE 数据集（200K 样本）阐明跨任务协作关系，结合交互感知 LoRA（多头 LoRA）学习不同音视频交互模式，在多个任务上超越专用模型。

**[Distinctad Distinctive Audio Description Generation In Contexts](audio_speech/distinctad_distinctive_audio_description_generation_in_contexts.md)**

:   生成上下文中有区分度的音频描述（AD），避免生成泛化无特色的描述，通过对比学习鼓励与前后AD的差异性

**[Team Leya In 10Th Abaw Competition Multimodal Ambivalencehesitancy Recognition A](audio_speech/team_leya_in_10th_abaw_competition_multimodal_ambivalencehesitancy_recognition_a.md)**

:   本文提出面向视频级矛盾/犹豫（A/H）识别的多模态方法，整合场景（VideoMAE）、面部（EmotionEfficientNetB0）、音频（EmotionWav2Vec2.0+Mamba）和文本（EmotionDistilRoBERTa）四种模态，通过原型增强的 Transformer 融合模型实现 83.25% 平均 MF1，最终以五模型集成在测试集达到 71.43%。

---

## 🔬 可解释性 { #interpretability }

**[Differentiable Inverse Rendering With Interpretable Basis Brdfs](interpretability/differentiable_inverse_rendering_with_interpretable_basis_brdfs.md)**

:   提出基于可解释基 BRDF 的可微逆渲染方法，将材质分解为有物理意义的基函数组合，实现可解释的材质估计

**[Geometry-Guided Camera Motion Understanding In Videollms](interpretability/geometry-guided_camera_motion_understanding_in_videollms.md)**

:   提出一个从基准构建、诊断到注入的完整框架，通过 3D 基础模型（VGGT）提取相机运动线索并以结构化提示注入 VideoLLM，实现无需训练的相机运动感知增强。

**[Towards Faithful Multimodal Concept Bottleneck Models](interpretability/towards_faithful_multimodal_concept_bottleneck_models.md)**

:   提出 f-CBM，一个基于 CLIP 的忠实多模态 Concept Bottleneck Model 框架，通过可微分的 leakage 损失和 Kolmogorov-Arnold Network 预测头联合解决概念检测准确性和信息泄漏问题，在任务精度、概念检测和 leakage 三者间达到最优权衡。

**[Why Does It Look There Structured Explanations For Image Classification](interpretability/why_does_it_look_there_structured_explanations_for_image_classification.md)**

:   提出I2X框架，通过追踪训练过程中模型置信度与原型强度的协同演化，将非结构化的显著性图解释转化为结构化的解释，回答模型"为什么看那里"的问题，并可指导微调提升性能。

---

## ⚡ LLM效率 { #llm_efficiency }

**[Associative Transformer](llm_efficiency/associative_transformer.md)**

:   提出 Associative Transformer (AiT)，通过在 Transformer 中引入可学习的显式记忆模块和 Hopfield 网络进行 token 重建，以更少的参数实现优于 ViT 的分类和关系推理性能。

**[Efficient Data Driven Mixture-Of-Expert Extraction From Trained Networks](llm_efficiency/efficient_data_driven_mixture-of-expert_extraction_from_trained_networks.md)**

:   提出一种从预训练 ViT 中自动提取 MoE（Mixture-of-Experts）变体的方法：先聚类 MLP 层的输出激活模式，再据此抽取对应的子网络作为专家，无需从头训练 MoE，在 ImageNet-1k 上仅需少量微调即可恢复 98% 原始性能，同时将 FLOPs 和模型大小分别减少 36% 和 32%。

**[Kac Kolmogorov-Arnold Classifier For Continual Learning](llm_efficiency/kac_kolmogorov-arnold_classifier_for_continual_learning.md)**

:   首次将 Kolmogorov-Arnold Network (KAN) 应用于持续学习，通过将 B-spline 替换为径向基函数 (RBF) 构建分类器 KAC，仅增加 0.23M 参数即可在多种持续学习方法上获得一致且显著的性能提升（CUB200 40-step 最高 +20.70%）。

**[Language Guided Concept Bottleneck Models For Interpretable Continual Learning](llm_efficiency/language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)**

:   将语言引导的概念瓶颈模型 (CBM) 整合到持续学习中，通过 ChatGPT 生成类别概念、CLIP 编码的概念对齐模块和语义引导的原型增强策略，在 ImageNet-subset 上实现最终准确率 +3.06% 的提升，同时提供透明可解释的决策过程。

---

## 💡 LLM推理 { #llm_reasoning }

**[Cot-Vla Visual Chain-Of-Thought Reasoning For Vision-Language-Action Models](llm_reasoning/cot-vla_visual_chain-of-thought_reasoning_for_vision-language-action_models.md)**

:   提出 CoT-VLA，将视觉思维链推理引入视觉-语言-动作模型，通过两阶段推理——先预测子目标图像再生成动作序列——结合混合注意力和动作分块策略，在 LIBERO 基准上实现 81.13% 平均成功率，显著超越现有方法。

**[Interleaved-Modal Chain-Of-Thought](llm_reasoning/interleaved-modal_chain-of-thought.md)**

:   提出交错模态思维链（ICoT），在推理步骤中穿插图像区域 crop 作为视觉 rationale，通过无参数的 Attention-driven Selection（ADS）从输入图像中智能选取关键区域插入生成序列，在 Chameleon 和 Qwen2-VL 上相比现有多模态 CoT 提升高达 14%。

**[Style Evolving Along Chain-Of-Thought For Unknown-Domain Object Detection](llm_reasoning/style_evolving_along_chain-of-thought_for_unknown-domain_object_detection.md)**

:   提出 Chain-of-Thought 引导的风格演化方法（CGSE），通过词→短语→句子三级渐进式风格描述生成，结合特征解耦和类别原型聚类，在五种恶劣天气场景和 Real-to-Art 基准上实现了显著的域泛化检测性能提升。

---

## ✍️ 文本生成 { #nlp_generation }

**[Artformer Controllable Generation Of Diverse 3D Articulated Objects](nlp_generation/artformer_controllable_generation_of_diverse_3d_articulated_objects.md)**

:   提出ArtFormer框架，通过树结构参数化和条件扩散Shape Prior，从文本/图像描述生成高质量、多样化且运动学关系准确的3D关节物体，在生成质量和多样性上显著超越现有方法。

**[Dense Match Summarization For Faster Two-View Estimation](nlp_generation/dense_match_summarization_for_faster_two-view_estimation.md)**

:   通过将密集匹配聚类为代表性子集，并用9×9矩阵编码每个聚类的几何约束，在保持精度的前提下将RANSAC鲁棒估计加速10-100倍。

**[LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table](nlp_generation/lotusfilter_fast_diverse_nearest_neighbor_search_via_a_learned_cutoff_table.md)**

:   提出LotusFilter，通过预计算截取表和贪心集合过滤，以O(S+KL)线性复杂度实现多样化近邻搜索，比传统方法快50-100倍，内存仅1/40。

---

## 👥 社会计算 { #social_computing }

**[As Language Models Scale Low-Order Linear Depth Dynamics Emerge](social_computing/as_language_models_scale_low-order_linear_depth_dynamics_emerge.md)**

:   将 Transformer 的深度方向视为离散时间动力系统，发现在给定上下文内可以用仅 32 维的线性状态空间代理模型高精度预测层间灵敏度曲线（Spearman 达 0.99），而且令人惊讶的是：**模型越大，低阶线性代理越准确**——这是一条新的 scaling law。

**[As Language Models Scale Low-Order Linear Depth Dynamics Emerge V2](social_computing/as_language_models_scale_low-order_linear_depth_dynamics_emerge_v2.md)**

:   将 Transformer 的深度方向视为离散时间动力系统，发现在给定上下文内可以用仅 32 维的线性状态空间代理模型高精度预测层间灵敏度曲线（Spearman 达 0.99），而且令人惊讶的是：**模型越大，低阶线性代理越准确**——这是一条新的 scaling law。

**[Classifier-Guided Clip Distillation For Unsupervised Multi-Label Classification](social_computing/classifier-guided_clip_distillation_for_unsupervised_multi-label_classification.md)**

:   提出 Classifier-guided CLIP Distillation（CCD），通过 CAM 引导的局部视图标签聚合和 CLIP 预测去偏两项核心技术，在完全无标注的条件下达到与全监督方法持平的多标签分类性能（VOC12 上 90.1% mAP）。

---

## 🔍 信息检索/RAG { #information_retrieval }

**[Advancing Myopia To Holism Fully Contrastive Language-Image Pre-Training](information_retrieval/advancing_myopia_to_holism_fully_contrastive_language-image_pre-training.md)**

:   将CLIP从传统的一对一(image, text)对比学习升级为多对多(multi-image-embeddings, multi-texts)对比学习范式，通过VLM生成多视角多层次的描述文本、多分支视觉编码器输出多种视觉embedding，实现更全面的视觉语言对齐，在检索/分类/密集任务上大幅超越baseline。

**[Chathuman Chatting About 3D Humans With Tools](information_retrieval/chathuman_chatting_about_3d_humans_with_tools.md)**

:   提出 ChatHuman，一个基于 LLM 的语言驱动系统，通过自动选择和集成专门的 3D 人体分析工具（3D 姿态估计、形状恢复、接触检测、人物交互分析、情感识别等），利用学术论文作为工具使用说明和 RAG（检索增强生成）创建 in-context 示例以管理新工具，在工具选择准确率和整体 3D 人体任务性能上超越现有 LLM 模型。

---

## 📊 LLM评测 { #llm_evaluation }

**[Context-Cir Learning From Concepts In Text For Composed Image Retrieval](llm_evaluation/context-cir_learning_from_concepts_in_text_for_composed_image_retrieval.md)**

:   提出 ConText-CIR 框架，通过 Text Concept-Consistency 损失让文本修改中的名词短语更好地关注查询图像的相关部分，配合合成数据生成管线，在多个 CIR 基准上取得 SOTA。

**[Out Of Sight Out Of Mind Evaluating State Evolution In Video World Models](llm_evaluation/out_of_sight_out_of_mind_evaluating_state_evolution_in_video_world_models.md)**

:   StEvo-Bench 提出了一个评估视频世界模型"不可观测状态演化"能力的 benchmark——测试当物理过程不被观察时（相机移开/遮挡/关灯），世界模型能否继续正确推理状态变化，结果发现当前所有前沿模型（Veo 3、Sora 2 Pro 等）的任务成功率均低于 10%，揭示了"眼不见，心不在"的严重缺陷。

---

## 📐 优化/理论 { #optimization }

**[Automatic Joint Structured Pruning And Quantization For Efficient Neural Network](optimization/automatic_joint_structured_pruning_and_quantization_for_efficient_neural_network.md)**

:   提出 GETA 框架实现自动联合结构化剪枝和量化感知训练：量化感知依赖图（QADG）构建通用剪枝搜索空间 + 部分投影 SGD 保证逐层比特约束 + 可解释的联合学习策略，在 CNN 和 Transformer 上均达到竞争力或领先的压缩性能。

**[Scope Semantic Coreset With Orthogonal Projection Embeddings For Federated Learn](optimization/scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)**

:   SCOPE 提出了一种面向联邦学习的语义 coreset 选择框架，利用 VLM（MobileCLIP-S2）零样本提取三种标量指标（表示分数、多样性分数、边界接近度），通过服务器聚合全局共识后指导客户端进行两阶段剪枝（异常过滤+冗余消除），在 128-512× 上行带宽减少和 7.72× 加速的同时保持竞争精度。

---

## 💻 代码智能 { #code_intelligence }

**[Codepercept Code-Grounded Visual Stem Perception For Mllms](code_intelligence/codepercept_code-grounded_visual_stem_perception_for_mllms.md)**

:   通过 scaling 分析发现 STEM 视觉推理的真正瓶颈是感知而非推理，提出用可执行 Python 代码作为精确感知媒介——构建 ICC-1M 数据集（Image-Caption-Code 三元组）训练模型，在 STEM 感知基准上 CodePercept-8B 比 Qwen3-VL-8B 提升 +3.0%-12.3%。

---

## 🎮 强化学习 { #reinforcement_learning }

**[Thinking In Streaming Video](reinforcement_learning/thinking_in_streaming_video.md)**

:   提出 ThinkStream，采用 Watch-Think-Speak 范式实现流式视频的实时连续推理，通过 RCSM（推理压缩流式记忆）将推理 trace 作为紧凑语义锚点替代旧视觉 token，配合 Streaming RLVR 训练策略，在保持低延迟/低内存的同时超越现有在线视频模型。

---

## 📈 时间序列 { #time_series }

**[Competition-Aware Cpc Forecasting With Near-Market Coverage](time_series/competition-aware_cpc_forecasting_with_near-market_coverage.md)**

:   将付费搜索CPC预测重构为"部分可观测竞争下的预测"问题，通过语义邻域（Transformer嵌入）、行为邻域（DTW对齐）和地理意图三类竞争代理逼近不可观测的竞争状态，在1811个关键词×127周的Google Ads数据上显示竞争感知增强在中长期预测（6/12周）上显著优于单变量和弱上下文baseline。

---

## 📂 其他 { #others }

**[4Real-Video Learning Generalizable Photo-Realistic 4D Video Diffusion](others/4real-video_learning_generalizable_photo-realistic_4d_video_diffusion.md)**

:   提出4Real-Video，一种基于双流架构的4D视频生成框架，通过将视频token分为时间流和视角流并行处理，引入hard/soft同步层协调两流信息，约1分钟即可生成8×8的高质量时空视频网格，在视觉质量和多视角一致性上超越现有方法。

**[A2Z-10M Geometric Deep Learning With A-To-Z Brep Annotations For Ai-Assisted Cad](others/a2z-10m_geometric_deep_learning_with_a-to-z_brep_annotations_for_ai-assisted_cad.md)**

:   构建了包含100万+复杂CAD模型、超1000万多模态标注（高分辨率3D扫描、手绘3D草图、文本描述、BRep拓扑标签）的A2Z数据集，是目前最大的CAD逆向工程数据集，并基于此训练了BRep边界和角点检测的基础模型。

**[Anomalyncd Towards Novel Anomaly Class Discovery In Industrial Scenarios](others/anomalyncd_towards_novel_anomaly_class_discovery_in_industrial_scenarios.md)**

:   提出 AnomalyNCD，首个基于自监督的工业多类异常分类方法：MEBin 提取主要异常区域 → 掩码引导 ViT 聚焦弱语义异常 → 区域融合策略实现灵活的区域/图像级分类，MVTec AD 上 F1 提升 10.8%，NMI 提升 8.8%。

**[Bendfm A Taxonomy And Synthetic Cad Dataset For Manufacturability Assessment In ](others/bendfm_a_taxonomy_and_synthetic_cad_dataset_for_manufacturability_assessment_in_.md)**

:   提出一个面向板金弯曲工艺的可制造性度量分类法（按配置依赖性×可行性/复杂度两个维度划分为四象限），并构建首个包含20,000个零件（含可制造与不可制造样本）的合成数据集BenDFM，基准测试表明图结构表示（UV-Net）优于点云（PointNext），配置依赖性指标的预测更具挑战性。

**[Bounds On Agreement Between Subjective And Objective Measurements](others/bounds_on_agreement_between_subjective_and_objective_measurements.md)**

:   通过仅假设投票均值收敛于真实质量，推导出主观测试（MOS）与客观估计器之间PCC（上界）和MSE（下界）的数学界限，并提出基于二项分布的投票模型BinoVotes，使得即使在投票方差不可用时也能计算这些界限，18个主观测试数据的验证表明BinoVotes界限与全数据驱动界限高度吻合。

**[Cadcrafter Generating Computer-Aided Design Models From Unconstrained Images](others/cadcrafter_generating_computer-aided_design_models_from_unconstrained_images.md)**

**[Deconstructing The Failure Of Ideal Noise Correction A Three-Pillar Diagnosis](others/deconstructing_the_failure_of_ideal_noise_correction_a_three-pillar_diagnosis.md)**

:   通过提供完美的oracle噪声转移矩阵T，证明Forward Correction在理想条件下仍会训练崩塌（先升后降最终与无校正基线收敛），从宏观（收敛终态）、微观（梯度动力学）、信息论（噪声信道不可逆信息损失）三个层面系统诊断了失败的根本原因——这不是T估计不准的问题，而是有限样本下高容量网络的结构性缺陷。

**[Integration Of Deep Generative Anomaly Detection Algorithm In High-Speed Industr](others/integration_of_deep_generative_anomaly_detection_algorithm_in_high-speed_industr.md)**

:   基于 GAN + 残差自编码器（DRAE）的半监督异常检测框架，在制药 BFS 高速产线上实现了仅用正常样本训练、单 patch 推理 0.17ms 的实时在线质检部署，通过 Perlin 噪声增强和 Noise Loss 优化重建质量。

**[Rooftop Wind Field Reconstruction Using Sparse Sensors From Deterministic To Gen](others/rooftop_wind_field_reconstruction_using_sparse_sensors_from_deterministic_to_gen.md)**

:   提出基于风洞PIV实验数据的学习框架，系统对比Kriging插值与三种深度学习模型（UNet、ViTAE、CWGAN）在稀疏传感器屋顶风场重建任务中的表现，并结合QR分解优化传感器布局。深度学习在混合风向训练下全面优于Kriging，SSIM提升最高33.5%。

**[Sdf-Net Structure-Aware Disentangled Feature Learning For Opticall-Sar Ship Re-I](others/sdf-net_structure-aware_disentangled_feature_learning_for_opticall-sar_ship_re-i.md)**

:   提出 SDF-Net，利用船舶作为刚体的物理先验，在 ViT 中间层提取尺度不变的梯度能量统计量作为跨模态几何锚点，并在终端层将特征解耦为模态不变共享特征和模态特定特征后通过加性残差融合，实现光学-SAR 船舶重识别 SOTA。

**[Shrec A Spectral Embedding-Based Approach For Ab-Initio Reconstruction Of Helica](others/shrec_a_spectral_embedding-based_approach_for_ab-initio_reconstruction_of_helica.md)**

:   提出SHREC算法，利用谱嵌入技术从冷冻电镜二维投影图像中直接恢复螺旋分子的投影角度，无需预知螺旋对称参数（rise/twist），仅需已知轴对称群Cn，实现从头(ab-initio)螺旋结构重建。

**[Sldprtnet A Large-Scale Multimodal Dataset For Cad Generation In Language-Driven](others/sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)**

:   构建包含24.2万工业零件的大规模多模态CAD数据集SldprtNet，提供3D模型、多视图图像、参数化文本脚本和自然语言描述的完整对齐，支持语义驱动的CAD建模任务。

**[Strap-Vit Segregated Tokens With Randomized -- Transformations For Defense Again](others/strap-vit_segregated_tokens_with_randomized_--_transformations_for_defense_again.md)**

:   STRAP-ViT 提出一种无需训练的即插即用 ViT 防御模块，利用 Jensen-Shannon 散度将受对抗补丁影响的 token 从正常 token 中分离出来，再通过随机复合变换消除其对抗效应，在多种 ViT 架构和攻击方法下实现了接近干净基线 2-3% 的鲁棒精度。

**[Wear Classification Of Abrasive Flap Wheels Using A Hierarchical Deep Learning A](others/wear_classification_of_abrasive_flap_wheels_using_a_hierarchical_deep_learning_a.md)**

:   针对柔性磨料翻页轮的复杂磨损模式，提出三级层次化深度学习分类框架，将磨损评估分解为使用状态检测、磨损类型识别和严重程度评估三个子任务，使用EfficientNetV2迁移学习实现93.8%–99.3%的分类精度。

**[Zo-Sam Zero-Order Sharpness-Aware Minimization For Efficient Sparse Training](others/zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)**

:   提出 ZO-SAM，将零阶优化策略性地整合到 SAM 的扰动步骤中，仅需一次反向传播即可获得 SAM 的平坦最小值优势，在稀疏训练场景下将计算开销减半的同时提升精度和鲁棒性。
