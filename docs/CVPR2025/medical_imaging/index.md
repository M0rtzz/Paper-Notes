---
title: >-
  CVPR2025 医学图像方向 60篇论文解读
description: >-
  60篇CVPR2025 医学图像方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**📷 CVPR2025** · **60** 篇论文解读

**[A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)**

:   提出结合 VLM 无训练伪标签生成（外观描述 prompt 驱动 Grounding DINO + SAM）和双教师不确定性融合精炼的半监督乳腺超声分割框架，仅用 2.5% 标注数据即达到接近全监督的性能。

**[AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP](aa-clip_enhancing_zero-shot_anomaly_detection_via_anomaly-aware_clip.md)**

:   提出 AA-CLIP，通过两阶段训练策略（先适配文本编码器建立异常感知锚点，再对齐 patch 级视觉特征），在保留 CLIP 泛化能力的前提下增强其异常判别力，仅需极少训练样本即可在工业和医学多个数据集上达到 SOTA 零样本异常检测性能。

**[Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](accelerating_stroke_mri_with_diffusion_probabilistic_models_through_large-scale_.md)**

:   借鉴基础模型范式，在大规模公开脑 MRI 数据上预训练扩散概率模型（DPM），再在仅 20 例中风患者数据上微调，实现数据受限场景下加速 MRI 重建，临床读者研究证实 2× 加速图像质量不劣于标准治疗。

**[Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](adaptation_of_weakly_supervised_localization_in_histopathology_by_debiasing_pred.md)**

:   提出 SFDA-DeP 方法，受机器遗忘启发，通过识别并纠正源模型在目标域的预测偏差（over-predict 某些类别），解决组织病理学中弱监督定位模型跨器官/跨中心域适应时预测偏差被放大的问题。

**[Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](addressing_data_scarcity_in_3d_trauma_detection_through_self-supervised_and_semi.md)**

:   提出两阶段标签高效学习框架：先在 1206 例无标注 CT 上用 Masked Image Modeling 自监督预训练 3D U-Net 编码器，再结合 VDETR + Vertex RPE 和 Mean Teacher 半监督学习，仅用 144 例标注数据实现腹部创伤 3D 检测 mAP@0.50 达 45.30%（+115%）。

**[Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts](association_of_radiologic_ppfe_change_with_mortality_in_lung_cancer_screening_co.md)**

:   在两个大规模肺癌筛查队列（NLST 7980 例、SUMMIT 8561 例）中验证了基于深度学习自动量化的 PPFE（胸膜肺实质纤维弹性组织增生）进展与全因死亡率独立相关，提出 PPFE 纵向变化可作为筛查人群中识别高呼吸发病风险个体的影像生物标志物。

**[Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](automated_detection_of_malignant_lesions_in_the_ovary_using_deep_learning_models.md)**

:   使用 15 种 CNN 变体（LeNet、ResNet、VGG、Inception）在组织病理学图像上检测卵巢癌及亚型，选择 InceptionV3（ReLU）作为最优模型（平均 94.58%），并使用 LIME、SHAP、Integrated Gradients 三种 XAI 方法解释模型预测。

**[BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)**

:   BiCLIP 提出了一种双向一致性视觉-语言分割框架，通过双向多模态融合（BMF，让视觉特征反向精炼文本嵌入）和图像增强一致性（IAC，跨弱/强扰动正则化），在 COVID-19 CT 分割上以仅 1% 标注数据即可保持鲁棒性能，且对临床图像退化（噪声/模糊）具有容忍力。

**[Boltzmann Attention Sampling for Image Analysis with Small Objects](boltzmann_attention_sampling_for_image_analysis_with_small_objects.md)**

:   提出BoltzFormer——一种新型transformer decoder架构，通过玻尔兹曼分布动态采样稀疏注意力区域来聚焦小目标，结合退火温度调度（早期层探索、后期层利用）和PiGMA多query聚合模块，在占图像面积<0.1%的小目标分割上比SOTA提升3-12% Dice分数，同时减少一个数量级的注意力计算。

**[CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools](cholectrack20_a_multi-perspective_tracking_dataset_for_surgical_tools.md)**

**[CLoE: Expert Consistency Learning for Missing Modality Segmentation](cloe_expert_consistency_learning_for_missing_modality_segmentation.md)**

:   提出 CLoE 框架，将缺失模态分割的鲁棒性问题重新定义为决策层专家一致性控制问题，通过全局模态专家一致性(MEC)和区域专家一致性(REC)双分支约束减少专家漂移，并用轻量门控网络将一致性分数转化为可靠性权重指导特征融合，在 BraTS 2020 和 MSD Prostate 上超越 SOTA。

**[CrossSDF: 3D Reconstruction of Thin Structures From Cross-Sections](crosssdf_3d_reconstruction_of_thin_structures_from_cross-sections.md)**

:   提出 CrossSDF，通过从 2D 截面符号距离场重建 3D SDF，结合混合编码（哈希网格 + 随机傅里叶特征）和对称差损失，首次实现对薄管状结构（如血管）的精确重建。

**[CycleULM: A Unified Label-Free Deep Learning Framework for Ultrasound Localisation Microscopy](cycleulm_a_unified_label-free_deep_learning_framework_for_ultrasound_localisatio.md)**

:   提出 CycleULM，首个统一的无标签深度学习超声定位显微(ULM)框架，通过 CycleGAN 学习 CEUS 帧到简化微泡域的物理仿真双向翻译来弥合仿真-真实域差距，实现微泡定位精度提升达40% recall、46% precision，并以18.3 fps 实现实时处理。

**[Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)**

:   提出 Deco-Mamba，一种以解码器为核心的混合 Transformer-CNN-Mamba 架构，通过 Co-Attention Gate、Vision State Space Module 和可变形卷积精炼块增强解码器能力，并引入基于窗口化 KL 散度的分布感知深度监督策略，在 7 个医学图像分割基准上取得 SOTA 性能，同时保持适中的模型复杂度。

**[Deep Learning Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](deep_learning_based_estimation_of_blood_glucose_levels_from_multidirectional_scl.md)**

:   提出 ScleraGluNet，通过五方向巩膜血管图像结合多分支 CNN + MRFO 特征筛选 + Transformer 跨视图融合，实现三分类代谢状态判别（93.8% 准确率）和连续空腹血糖估计（MAE = 6.42 mg/dL），为无创血糖监测提供了新途径。

**[Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](developing_foundation_models_for_universal_segmentation_from_3d_whole-body_posit.md)**

:   构建了最大规模 PET 分割数据集 PETWB-Seg11K（11,041 例全身 PET + 59,831 个分割掩码），并提出 SegAnyPET——基于 3D 架构 + prompt 工程的 PET 通用分割基础模型，在多中心、多示踪剂、多疾病场景下展现强零样本泛化能力。

**[DFLMoE: Decentralized Federated Learning via Mixture of Experts for Medical Data](dflmoe_decentralized_federated_learning_via_mixture_of_experts_for_medical_data_.md)**

:   提出 DFLMoE 在去中心化联邦学习中使用混合专家（MoE）机制处理医疗数据异质性，无需中心服务器即可在保护隐私的前提下协同训练

**[Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)**

:   提出一种结合非负矩阵分解（NNMF）特征提取、统计特征筛选、轻量 CNN 分类和扩散式特征空间去噪的脑肿瘤分类框架，在保持 ~85% 干净准确率的同时，将 AutoAttack 下的鲁棒准确率从 0.47% 提升至 59.5%。

**[DiN: Diffusion Model for Robust Medical VQA with Semantic Noisy Labels](din_diffusion_model_for_robust_medical_vqa_with_semantic_noisy_labels.md)**

:   本文提出DiN框架，首次将扩散模型应用于医学VQA的噪声标签场景（NM-VQA），通过扩散式答案分类器从生成视角进行粗到细的答案筛选，配合噪声标签精炼模块动态修正标签，在10%语义噪声下VQA-RAD准确率达74.24%，超越SNLC的69.65%。

**[Distilled Prompt Learning for Incomplete Multimodal Survival Prediction](distilled_prompt_learning_for_incomplete_multimodal_survival_prediction.md)**

:   本文提出DisPro (Distilled Prompt Learning)，通过两阶段提示学习——UniPro蒸馏各模态知识分布 + MultiPro利用LLM从可用模态推断缺失模态——同时补偿缺失模态的特异性和共享信息，在5个TCGA生存预测数据集上取得SOTA。

**[Domain Adaptive Diabetic Retinopathy Grading with Model Absence and Flowing Data](domain_adaptive_diabetic_retinopathy_grading_with_model_absence_and_flowing_data.md)**

:   本文提出 GUES（Generative Unadversarial Examples）方法，在无法访问源模型参数和标签、目标数据以流式到达的极端在线无模型领域自适应（OMG-DA）场景下，通过 VAE 生成个性化非对抗性扰动并以显著性图作为伪监督，提升冻结源模型在目标域上的糖尿病视网膜病变（DR）分级性能。

**[EchoONE: Segmenting Multiple Echocardiography Planes in One Model](echoone_segmenting_multiple_echocardiography_planes_in_one_model.md)**

:   本文提出 EchoONE，首次用一个统一模型解决超声心动图多切面分割（MPS）问题，通过先验可组合掩码学习（PC-Mask）模块生成语义感知的稠密 prompt，并设计局部特征融合与适配（LFFA）模块将 CNN 局部特征注入 SAM 解码器，在 6 个切面上持续达到 SOTA 性能。

**[EchoWorld: Learning Motion-Aware World Models for Echocardiography Probe Guidance](echoworld_learning_motion-aware_world_models_for_echocardiography_probe_guidance.md)**

:   本文提出 EchoWorld，一种面向超声心动图探头引导的运动感知世界建模框架：先通过空间世界建模（掩码重建）和运动世界建模（探头运动与视觉变化预测）进行预训练以编码心脏解剖知识，然后在微调阶段引入运动感知注意力机制融合历史视觉-运动序列，在 10 个标准切面的引导任务上显著降低引导误差。

**[Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation](enhanced_contrastive_learning_with_multi-view_longitudinal_data_for_chest_x-ray_.md)**

:   提出 MLRG 两阶段框架，通过多视角纵向对比学习融合当前多视角图像的空间信息和历史纵向数据的时间信息进行视觉-文本预训练，并用 tokenized absence encoding 灵活处理缺失的患者先验知识，在 MIMIC-CXR 上 BLEU-4 提升 2.3%，MIMIC-ABN 上 F1 提升 5.5%。

**[Enhancing Virtual Try-On with Synthetic Pairs and Error-Aware Noise Scheduling](enhancing_virtual_try-on_with_synthetic_pairs_and_error-aware_noise_scheduling.md)**

:   本文提出通过人体图像反向提取合成服装对来增强虚拟试穿训练数据，并设计了基于错误感知噪声调度的Schrödinger Bridge精炼模型（EARSB），对已有试穿模型的生成结果进行局部纠错，在VITON-HD和DressCode上取得了SOTA效果且用户更偏好本文结果（59%）。

**[EquivAnIA: A Spectral Method for Rotation-Equivariant Anisotropic Image Analysis](equivania_a_spectral_method_for_rotation-equivariant_anisotropic_image_analysis.md)**

:   提出 EquivAnIA，一种基于 cake wavelet 和 ridge filter 的频谱方法，用于对图像进行旋转等变的各向异性分析，在合成和真实图像（含 CT）上展现出优于传统 angular binning 的旋转鲁棒性。

**[Evidential learning driven Breast Tumor Segmentation with Stage-divided Vision-Language Interaction](evidential_learning_driven_breast_tumor_segmentation_with_stage-divided_vision-l.md)**

:   提出 TextBCS 模型，通过阶段分割的视觉-语言交互模块（SVLI）和证据学习（EL）策略，利用文本提示辅助乳腺肿瘤分割，在 Duke-Breast-Cancer-MRI 数据集上 Dice 达 85.33%，超越所有对比方法。

**[Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)**

:   提出 FedMEPD 联邦学习框架，通过模态专属编码器（全局联邦）和部分个性化融合解码器，同时解决多模态 MRI 脑肿瘤分割中的模态间异质性和客户端个性化问题，在 BraTS 2018/2020 上客户端平均 mDSC 达 75.70%/75.90%。

**[Few-Shot Personalized Scanpath Prediction](few-shot_personalized_scanpath_prediction.md)**

:   提出少样本个性化扫视路径预测（FS-PSP）任务 和 Subject-Embedding Network（SE-Net），通过将主体嵌入学习与扫视路径预测解耦，仅需 1-10 张图像的注视数据即可适配新用户，在 OSIE、COCO-FreeView、COCO-Search18 三个数据集上 ScanMatch 指标超越第二名 5.9%-7.9%，且适配时间仅 3.6 秒、无需微调。

**[FFaceNeRF: Few-Shot Face Editing in Neural Radiance Fields](ffacenerf_few-shot_face_editing_in_neural_radiance_fields.md)**

:   提出 FFaceNeRF，一种基于 NeRF 的面部编辑方法，通过几何适配器（geometry adapter）+ 三平面特征注入 + 潜码混合增强（LMTA），仅需 10 张标注样本即可适配到任意自定义分割 mask 布局，实现灵活的 3D 感知面部编辑。

**[GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](giim_graph-based_learning_of_inter-_and_intra-view_dependencies_for_multi-view_m.md)**

:   提出 GIIM，一种基于多异构图（MHG）的多视图医学图像分类框架，同时建模视图内（intra-view）和视图间（inter-view）的病灶依赖关系，在肝脏 CT、乳腺 X 线和乳腺 MRI 三种模态上均显著优于现有多视图方法，并对缺失视图具有鲁棒性。

**[Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)**

:   提出 GenEval，通过域共形界（DCB）理论量化因果覆盖差距，并将人类专家知识与 MedGemma-4B 视觉语言模型结合，实现单源域泛化（SDG），在糖尿病视网膜病变分级（8 个数据集）和癫痫灶检测（2 个数据集）上大幅超越现有方法。

**[Interactive Medical Image Analysis with Concept-based Similarity Reasoning](interactive_medical_image_analysis_with_concept-based_similarity_reasoning.md)**

:   本文提出 CSR（Concept-based Similarity Reasoning）网络，通过学习概念原型在图像局部区域的相似性来进行分类推理，同时支持医生在训练和测试时从空间级和概念级两个维度进行交互式干预，在三个医学数据集上以高达 4.5% 的 F1 提升超越了现有可解释方法。

**[Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline](interactive_medical_image_segmentation_a_benchmark_dataset_and_baseline.md)**

:   本文提出 IMed-361M，一个包含 640 万张图像和 3.61 亿个 mask（平均每张 56 个）的大规模交互式医学图像分割基准数据集，覆盖 14 种成像模态和 204 个分割目标，并基于此开发了支持点击、边框、文本及组合交互的 IMIS 基线网络，在多个场景下超越现有视觉基础模型。

**[Knowledge Bridger: Towards Training-Free Missing Modality Completion](knowledge_bridger_towards_training-free_missing_modality_completion.md)**

:   本文提出 Knowledge Bridger，一个免训练的缺失模态补全框架，通过利用大型多模态模型（LMM）自动挖掘多模态知识、构建知识图谱来指导缺失模态的生成与排序，在通用场景和医学OOD场景下均超越了现有方法。

**[Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis](latent_drifting_in_diffusion_models_for_counterfactual_medical_image_synthesis.md)**

:   本文提出 Latent Drifting (LD)，通过在扩散模型的前向和反向过程中引入一个标量偏移参数 δ 来弥合预训练自然图像模型与医学图像目标分布之间的差距，显著提升了多种微调方案下的医学图像生成和反事实图像合成效果。

**[MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](mil-pf_multiple_instance_learning_on_precomputed_features_for_mammography_classi.md)**

:   提出 MIL-PF 框架，利用冻结的基础视觉模型预计算特征，配合仅 ~40k 参数的轻量 MIL 聚合头，在乳腺 X 光分类任务上达到 SOTA 性能，大幅降低训练成本。

**[MoEdit: On Learning Quantity Perception for Multi-Object Image Editing](moedit_on_learning_quantity_perception_for_multi-object_image_editing.md)**

:   提出无辅助工具的多物体图像编辑框架 MoEdit，通过 FeCom 模块补偿 CLIP 编码中物体属性的交叉混淆、QTTN 模块注入数量感知到 U-Net，实现编辑前后物体数量一致且属性互不干扰。

**[Multi-modal Vision Pre-training for Medical Image Analysis (BrainMVP)](multi-modal_vision_pre-training_for_medical_image_analysis.md)**

:   BrainMVP提出首个多模态视觉预训练范式，通过跨模态掩码重建、模态模板蒸馏和模态感知对比学习三个代理任务，在16,022例多参数脑MRI扫描(240万+图像)上预训练ViT，在六个分割和四个分类下游任务上均超越SOTA，Dice Score提升最高达14.47%。

**[Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](multi-resolution_pathology-language_pre-training_model_with_text-guided_visual_r.md)**

:   提出 MR-PLIP，首个在多分辨率（5×/10×/20×/40×）下进行病理-语言预训练的视觉语言模型，通过跨分辨率视觉-文本对齐（CVTA）和多分辨率文本引导视觉表示对齐（MRTVA），在 34M 图文对上训练后，在 26 个基准数据集上全面超越 SOTA 基础模型。

**[Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)**

:   提出 RICE-NET，一个多模态 3D 深度学习模型，融合纵向 MRI 数据与放疗剂量分布图，用于区分胶质母细胞瘤术后放射性对比增强（RICE）与肿瘤复发，在独立测试集上达到 F1=0.92。

**[Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation](multimodal_protein_language_models_for_enzyme_kinetic_parameters_from_substrate_.md)**

:   提出 ERBA 适配器，将酶动力学预测建模为"底物识别→构象适应"的分阶段条件化过程，通过 MRCA 注入底物语义、G-MoE 融合活性位点3D几何、ESDA 保持 PLM 先验，在 kcat/Km/Ki 三个动力学端点上一致超越现有方法。

**[MultiMorph: On-demand Atlas Construction](multimorph_on-demand_atlas_construction.md)**

:   本文提出MultiMorph，一种前馈式脑图谱构建模型，通过线性复杂度的GroupBlock特征共享层和Centrality Layer实现任意数量3D脑图像的单次前向传播即生成无偏群组图谱，速度比传统优化方法快100倍，且无需微调即可泛化到未见模态和人群。

**[Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)**

:   提出 MSG-LDM 框架，在潜在空间中显式解耦风格与结构信息，通过高频注入块 (HFIB)、多模态结构特征融合 (MMSF) 和多尺度结构增强 (MSSE) 提取模态不变的多尺度结构先验来引导扩散过程，解决任意模态缺失下 MRI 翻译的解剖不一致和纹理退化问题。

**[NOIR: Neural Operator Mapping for Implicit Representations](noir_neural_operator_mapping_for_implicit_representations.md)**

:   NOIR 将医学图像计算任务重新建模为连续函数空间之间的算子学习问题，通过隐式神经表示(INR)将离散医学信号嵌入连续函数空间，再用神经算子(NO)学习函数间的映射，实现分辨率无关的分割、形状补全、图像翻译和合成。

**[Novel Architecture of RPA In Oral Cancer Lesion Detection](novel_architecture_of_rpa_in_oral_cancer_lesion_detection.md)**

:   本文将 Singleton 和 Batch Processing 设计模式集成到基于 Python 的 RPA 自动化管道中，结合 EfficientNetV2B1 模型实现口腔癌病灶检测，相比 UiPath/Automation Anywhere 等传统 RPA 平台实现 60-100× 的推理加速。

**[Nyxus: A Next Generation Image Feature Extraction Library for the Big Data and AI Era](nyxus_a_next_generation_image_feature_extraction_library_for_the_big_data_and_ai.md)**

:   Nyxus 是一个面向大数据和 AI 时代的下一代图像特征提取库，支持 2D/3D 数据的 out-of-core 可扩展提取，覆盖 radiomics 和细胞分析两大领域共 261+ 特征，在速度上比 CellProfiler 快 3–131×、比 PyRadiomics/MITK 快数倍至数百倍。

**[LoV3D: Grounding Cognitive Prognosis Reasoning in Longitudinal 3D Brain MRI via Regional Volume Assessments](paper_title_lov3d_grounding_cognitive_prognosis_reasoning_in_longitudinal_3d_bra.md)**

:   LoV3D 提出一套端到端纵向 3D 脑 MRI 视觉-语言模型管线，通过结构化可验证输出设计实现解剖区域评估 + 纵向对比 + 三分类诊断推理，并利用临床加权 Verifier 驱动 DPO 训练（无需人工标注），在 ADNI 上达到 93.7% 三分类准确率且零非相邻诊断错误。

**[Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](prototype-based_knowledge_guidance_for_fine-grained_structured_radiology_reporti.md)**

:   ProtoSR 提出从大规模自由文本放射学报告中挖掘模板对齐的原型知识库，并通过原型条件化的后期融合残差模块注入结构化报告预测，在 Rad-ReStruct 基准上实现 SOTA，尤其在细粒度属性问题 (L3) 上获得 72.1% 的相对提升。

**[Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](reinforcing_the_weakest_links_modernizing_siena_with_targeted_deep_learning_inte.md)**

:   将深度学习模块（SynthStrip/SynthSeg）模块化替换 SIENA 管线中的经典颅骨剥离和组织分割步骤，在保留管线可解释性的前提下显著提升纵向脑萎缩（PBVC）估计的临床敏感性和鲁棒性。在 ADNI 和 PPMI 两个纵向队列上验证。

**[Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)**

:   针对无任务 ID 和无数据回放的领域增量学习（DIL），提出 Residual SODAP 框架，通过 α-entmax 稀疏 prompt 选择与残差聚合、基于特征统计的伪回放蓏馏、prompt 使用模式漂移检测和不确定性加权，同时解决表示适配和分类器遗忘问题。在 DR、皮肤癌和 CORe50 上均达 SOTA。

**[SALIENT: Frequency-Aware Paired Diffusion for Controllable Long-Tail CT Detection](salient_frequency-aware_paired_diffusion_for_controllable_long-tail_ct_detection.md)**

:   提出 SALIENT，一个基于小波域扩散的掩码条件生成框架，通过频率感知的可解释优化目标和配对的病灶-掩码体积生成，实现长尾 CT 检测中可控、高效的合成数据增强与精度拯救。首次系统表征增强剂量-反应曲线。

**[Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)**

:   提出 SCDL 即插即用模块，通过学习类条件代理分布并进行双向对齐（CDBA）+ 语义锚约束（SAC），在嵌入空间显式重塑类条件特征结构，缓解半监督医学影像分割中的监督偏差和表示不平衡。

**[SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)**

:   提出 SemiTooth 多教师多学生半监督框架，通过 Stricter Weighted-Confidence Constraint 实现多源 CBCT 牙齿分割的跨域泛化。

**[Transformer-Based Multi-Region Segmentation and Radiomic Analysis of HR-pQCT Imaging for Osteoporosis Classification](transformer-based_multi-region_segmentation_and_radiomic_analysis_of_hr-pqct_ima.md)**

:   首次将 SegFormer 用于 HR-pQCT 影像的多区域（骨+软组织）自动分割与放射组学分析，发现肌腱组织特征在骨质疏松分类中优于传统骨指标。

**[UltrasoundAgents: Hierarchical Multi-Agent Evidence-Chain Reasoning for Breast Ultrasound Diagnosis](ultrasoundagents_hierarchical_multi-agent_evidence-chain_reasoning_for_breast_ul.md)**

:   提出 UltrasoundAgents 层次化多智能体框架，通过主智能体定位病灶+子智能体识别属性+证据链推理的流程，对齐乳腺超声临床诊断工作流并实现可追溯的 BI-RADS 分级与良恶性判断。

**[Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)**

:   提出 SMART 框架，基于 SAM3 的教师-学生结构结合文本概念提示、置信度感知一致性正则化和双流时序一致性，实现 X 光冠脉造影视频的半监督血管分割。

**[UNIStainNet: Foundation-Model-Guided Virtual Staining of H&E to IHC](unistainnet_foundation-model-guided_virtual_staining_of_he_to_ihc.md)**

:   提出 UNIStainNet，首次将冻结病理基础模型 UNI 的稠密空间 token 作为生成器的直接条件信号，实现 H&E 到 IHC 的虚拟染色，单一统一模型同时服务四种 IHC 标记物并达到 SOTA。

**[Unleashing Video Language Models for Fine-grained HRCT Report Generation](unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)**

:   提出 AbSteering 框架，通过异常中心化 CoT 训练和基于临床混淆异常硬负例的 DPO 优化，将通用视频语言模型（VideoLMs）高效迁移到 HRCT 报告生成任务，性能超越专用 CT 基础模型。

**[Unmasking Biases and Reliability Concerns in Convolutional Neural Networks Analysis of Cancer Pathology Images](unmasking_biases_and_reliability_concerns_in_convolutional_neural_networks_analy.md)**

:   通过从 13 个癌症病理基准数据集中裁剪 20×20 像素的背景区域（不含任何临床诊断信息）训练 ResNet50/DenseNet121/InceptionV3/VGG16 四种 CNN，发现分类准确率远高于随机猜测（最高达 93%），系统性揭示了 CNN 在癌症病理分析中可能依赖数据集采集偏差（如染色协议、扫描仪差异）而非真正的病理特征进行判断。
