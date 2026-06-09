---
title: >-
  CVPR2026 医学图像方向100篇论文解读
description: >-
  100篇CVPR2026的医学图像方向论文解读，涵盖医学影像、语义分割、多模态、对抗鲁棒、少样本学习、扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2026"
  - "医学图像"
  - "论文解读"
  - "论文笔记"
  - "医学影像"
  - "语义分割"
  - "多模态"
  - "对抗鲁棒"
  - "少样本学习"
  - "扩散模型"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**📷 CVPR2026** · **100** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (24)](../../ICML2026/medical_imaging/index.md) · [🔬 ICLR2026 (19)](../../ICLR2026/medical_imaging/index.md) · [🤖 AAAI2026 (73)](../../AAAI2026/medical_imaging/index.md) · [🧠 NeurIPS2025 (75)](../../NeurIPS2025/medical_imaging/index.md) · [📹 ICCV2025 (31)](../../ICCV2025/medical_imaging/index.md) · [🧪 ICML2025 (22)](../../ICML2025/medical_imaging/index.md)

🔥 **高频主题：** 医学影像 ×35 · 语义分割 ×22 · 多模态 ×12 · 对抗鲁棒 ×8 · 少样本学习 ×6

**[A protocol for evaluating robustness to H&E staining variation in computational pathology models](a_protocol_for_evaluating_robustness_to_he_stainin.md)**

:   提出三步评估协议（选参考染色条件→表征测试集染色属性→模拟染色条件推理），系统量化306个MSI分类模型对H&E染色差异的鲁棒性，发现鲁棒性与分类性能呈弱负相关(r=-0.28)，高性能不代表高鲁棒性。

**[A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)**

:   提出面向乳腺超声（BUS）图像分割的半监督框架，利用 GPT-5 生成外观描述 + Grounding DINO + SAM 免训练生成伪标签（APPG），结合双教师框架（静态+动态）通过不确定性-熵加权融合（UEWF）和自适应不确定性引导反向对比学习（AURCL）精炼标签，仅用 2.5% 标注即接近全监督性能。

**[Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](accelerating_stroke_mri_with_diffusion_probabilist.md)**

:   受基础模型范式启发，提出一种数据高效的扩散概率模型 (DPM) 训练策略——先在大规模多对比度脑 MRI 数据（~4000 受试者）上预训练，再用极少量目标域数据（仅 20 受试者）微调，在临床中风 MRI 加速重建中达到与大规模训练相当的质量，临床读片研究证实重建质量与标准诊疗无显著差异。

**[Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning](act_like_a_pathologist_tissue-aware_whole_slide_image_reasoning.md)**

:   提出 HistoSelect 框架，模拟病理学家从粗到细的推理过程，通过组织分割→Group Sampler→Patch Selector 的三级筛选机制，基于信息瓶颈(IB)理论压缩无关视觉token，在减少约70%计算量的同时实现三个数据集上的SOTA。

**[Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](adaptation_of_weakly_supervised_localization_in_histopathology_by_debiasing_pred.md)**

:   提出 SFDA-DeP，受机器遗忘启发，将 SFDA 重新定义为"识别并纠正预测偏差"的迭代过程：对 dominant class 中高熵不确定样本执行"遗忘"操作迫使模型放弃偏向性预测，对可靠样本保持自训练，同时用像素级分类器锚定定位能力，在跨器官/跨中心病理基准上持续优于现有 SFDA 方法。

**[Adaptive Confidence Regularization for Multimodal Failure Detection](adaptive_confidence_regularization_for_multimodal_failure_detection.md)**

:   提出 ACR 框架，通过自适应置信度损失（惩罚多模态融合置信度低于单模态的"置信度退化"现象）和多模态特征交换（在特征空间合成失败样本）两个互补模块，首次系统解决多模态场景下的误分类检测问题，在四个数据集上全面超越已有方法。

**[Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](addressing_data_scarcity_in_3d_trauma_detection_through_self-supervised_and_semi.md)**

:   提出两阶段标签高效框架：先用 patch-based MIM 在1,206个无标注CT上自监督预训练3D U-Net编码器，再用VDETR+3D顶点相对位置编码做3D损伤检测，配合Mean Teacher半监督一致性正则化利用2,000个无标注体数据，仅用144个有标注样本即实现56.57% val mAP@0.50（比纯监督提升115%）。

**[From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](apex_adaptive_visual_prompting.md)**

:   提出 APEX（Adaptive Prompt EXtraction），通过从可学习 prompt 记忆中自适应检索输入特定的 visual prompt（而非为每个域固定一个 prompt），结合低频特征对比学习增强域间区分能力，显著提升医学图像分割在已见域和未见域上的泛化性能。

**[Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study](are_general-purpose_vision_models_all_we_need_for_2d_medical_image_segmentation_.md)**

:   通过在三个异构医学数据集上对 11 种架构进行标准化对比实验，证明了通用视觉模型 (GP-VMs) 在 2D 医学图像分割中可以超越大多数专用医学分割架构 (SMAs)，且 XAI 分析表明 GP-VMs 无需特定领域设计也能捕获临床相关结构。

**[Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts](association_of_radiologic_ppfe_change_with_mortali.md)**

:   在 NLST（n=7980）和 SUMMIT（n=8561）两个大规模肺癌筛查队列中，利用深度学习自动分割量化低剂量 CT 上 PPFE 的纵向变化（dPPFE），验证其与全因死亡率（HR=1.25/3.14）和呼吸系统发病率的独立关联。

**[Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](automated_detection_of_malignant_lesions_in_the_ov.md)**

:   系统对比 15 种 CNN 变体（LeNet/ResNet/VGG/Inception）在卵巢癌组织病理图像五分类上的表现，最终选出 InceptionV3-A（ReLU）达 94% 综合指标，并用 LIME/SHAP/Integrated Gradients 三种 XAI 方法做对比解释分析。

**[Benchmarking Endoscopic Surgical Image Restoration and Beyond](benchmarking_endoscopic_surgical_image_restoration_and_beyond.md)**

:   构建了首个多源真实世界内窥镜手术图像复原数据集 SurgClean（3,113张图像，覆盖去烟/去雾/去飞溅三种退化类型），在其上系统评测了22种代表性图像复原方法（12种通用+10种任务特定），揭示现有方法与临床需求间仍存在显著差距，并进一步分析了手术场景退化与自然场景退化的本质差异。

**[Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance](better_than_average_spatially-aware_aggregation_of_segmentation_uncertainty_impr.md)**

:   首次系统研究分割任务中像素级不确定性到图像级评分的聚合策略，提出融合空间结构信息（Moran's I、边缘密度、Shannon熵）的SMR聚合器和基于GMM的元聚合器，在10个数据集上证明全局平均(AVG)是次优选择，GMM-All元聚合在OoD和失败检测上表现稳健。

**[BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)**

:   提出 BiCLIP 框架，通过双向多模态融合（BMF）实现视觉信息反向精炼文本表示，并通过图像增强一致性（IAC）约束中间特征的扰动不变性，在 COVID-19 CT 分割上超越 SOTA，仅 1% 标注数据仍保持鲁棒。

**[Bridging the Skill Gap in Clinical CBCT Interpretation with CBCTRepD](bridging_the_skill_gap_in_clinical_cbct_interpreta.md)**

:   构建覆盖55种口腔疾病的7,408例大规模CBCT-报告配对数据集，开发双语口腔颌面CBCT报告生成系统CBCTRepD，通过AI生成草稿+放射科医生编辑的协作模式，在多层级临床评估中证明其可帮助初级医生达到中级水平、中级医生接近高级水平、高级医生减少遗漏。

**[Can Natural Image Autoencoders Compactly Tokenize fMRI Volumes for Long-Range Dynamics Modeling?](can_natural_image_autoencoders_compactly_tokenize_fmri_volumes_for_long-range_dy.md)**

:   提出 TABLeT，利用预训练的 2D 自然图像自编码器（DCAE）将 3D fMRI 体积压缩为仅 27 个连续 token，配合简单 Transformer 编码器实现前所未有的长时序建模（256 帧），在 UKB、HCP、ADHD-200 上多任务超越 SOTA 体素方法，且计算效率大幅提升。

**[CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection](chips_efficient_clip_adaptation_via_curvature-aware_hybrid_influence-based_data_.md)**

:   提出 CHIPS，一种基于曲率感知混合影响力的数据选择方法，在 CLIP 端点子空间中计算 Newton 风格对齐分数并结合可学习性与领域相关性权重，仅用 30% 数据即可匹配全量数据集持续预训练效果，在 17 个医学基准上达到 SOTA。

**[CLoE: Expert Consistency Learning for Missing Modality Segmentation](cloe_expert_consistency_learning_for_missing_modality_segmentation.md)**

:   提出 CLoE（Consistency Learning of Experts），将缺失模态鲁棒性问题建模为决策层面的专家一致性控制，通过模态专家一致性（MEC）和区域专家一致性（REC）双分支约束减少专家漂移，并用一致性分数驱动的门控网络实现可靠性加权融合。

**[Comparative Evaluation of Traditional Methods and Deep Learning for Brain Glioma Imaging](comparative_evaluation_of_traditional_methods_and_deep_learning_for_brain_glioma.md)**

:   一篇系统性综述论文，全面对比传统方法（阈值分割、区域生长、模糊聚类等）和深度学习方法（CNN、U-Net、SegNet 等）在脑胶质瘤 MRI 分割与分类任务上的表现，结论指出 CNN 架构在准确性和自动化程度上全面优于传统技术。

**[CRFT: Consistent-Recurrent Feature Flow Transformer for Cross-Modal Image Registration](crft_consistent-recurrent_feature_flow_transformer_for_cross-modal_image_registr.md)**

:   提出CRFT，统一的粗到精跨模态图像配准框架——在Transformer架构中学习模态无关的特征流表示，粗阶段1/8分辨率全局对应+精阶段1/2-1/4多尺度局部细化，配合迭代差异引导注意力和空间几何变换(SGT)递归精化流场捕捉微妙空间不一致，在光学/红外/SAR/多光谱等多种跨模态数据集上超越RAFT/GMFlow/LoFTR等SOTA。

**[CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)**

:   提出 CURE——一种基于误差感知课程学习的多任务训练框架，在不引入额外数据的前提下，通过动态调节采样分布重点训练困难样本，将医学 VLM 的视觉定位精度提升 +0.37 IoU，幻觉率降低 18.6%。

**[Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)**

:   提出 Deco-Mamba，一种以解码器为中心的 Transformer-CNN-Mamba 混合架构，通过 Co-Attention Gate、视觉状态空间模块（VSSM）和可变形卷积增强解码过程，同时引入基于窗口化 KL 散度的分布感知深度监督策略，在 7 个医学图像分割基准上取得 SOTA。

**[Decoupling Vision and Language: Codebook Anchored Visual Adaptation](decoupling_vision_and_language_codebook_anchored_visual_adaptation.md)**

:   提出 CRAFT，通过离散 codebook 将视觉编码器与语言模型解耦，仅微调视觉编码器即可实现领域适配，且适配后的编码器可跨 LLM 架构无缝复用，在 10 个领域基准上平均提升 13.51%。

**[Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](deep_learning_based_estimation_of_blood_glucose_levels_from_multidirectional_scl.md)**

:   提出ScleraGluNet，通过5个注视方向的巩膜血管照片，用并行CNN提取方向特异性血管特征，再经MRFO特征筛选和Transformer跨视角融合，同时完成三类代谢状态分类（93.8%准确率）和空腹血糖连续估计（MAE=6.42 mg/dL, r=0.983）。

**[Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning](deep_learningbased_assessment_of_the_relation_betw.md)**

:   在按8个独立标注者划分的全景口腔X光裁剪片上，系统对比本地学习（LL）、联邦学习（FL）和集中学习（CL）三种训练范式在第三磨牙-下颌管重叠二分类任务上的表现，验证了CL > FL > LL的性能排序（AUC分别为0.831、0.757和0.672），证明FL在保护数据隐私的前提下显著优于各站点独立训练。

**[Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](developing_foundation_models_for_universal_segmentation_from_3d_whole-body_posit.md)**

:   构建迄今最大的全身 PET 分割数据集 PETWB-Seg11K（11,041 例 3D PET + 59,831 masks），并提出 SegAnyPET——首个面向功能性 PET 影像的 3D 可提示分割基础模型，在多中心、多示踪剂、多疾病场景下实现了强零样本泛化能力。

**[Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)**

:   本文提出 NNMF+CNN+扩散防御框架用于脑肿瘤 MRI 分类：先用 NNMF 将图像分解为紧凑可解释的低秩特征，通过 AUC/Cohen's d/p-value 统计指标筛选最强判别组件，再用轻量 CNN 分类；推理时引入前向扩散加噪 + 学习去噪器的特征空间净化模块，在 AutoAttack ($L_\infty$, $\epsilon=0.10$) 下将鲁棒准确率从 0.47% 提升至 59.53%。

**[EchoAgent: Towards Reliable Echocardiography Interpretation with "Eyes", "Hands" and "Minds"](echoagent_towards_reliable_echocardiography_interpretation_with_eyeshands_and_mi.md)**

:   提出 EchoAgent，一个模拟心脏超声医师"眼-手-脑"协同工作流程的 Agent 系统，通过专业知识引擎（mind）、分层工具箱（eyes+hands）和编排推理中枢（reasoning hub）三阶段实现端到端超声心动图可靠解读，在多个基准上达到 SOTA。

**[EI: Early Intervention for Multimodal Imaging based Disease Recognition](ei_early_intervention_for_multimodal_imaging_based_disease_recognition.md)**

:   EI 提出在单模态嵌入（UIE）**之前**就注入跨模态语义引导（[INT] token），模拟临床医生"先看一个模态形成初步判断再指导另一个模态检查"的工作流程，同时设计 MoR（多种秩 LoRA + 带旁路的松弛路由器）实现参数高效的 VFM 医学域适配，在视网膜/皮肤/膝关节三个数据集上以 <9M 可训练参数超越所有全参微调和 prompt learning 基线。

**[EMAD: Evidence-Centric Grounded Multimodal Diagnosis for Alzheimer's Disease](emad_evidence-centric_grounded_multimodal_diagnosis_for_alzheimers_disease.md)**

:   提出 EMAD，一个端到端多模态视觉-语言框架，为 AD 诊断生成结构化报告，通过分层 Sentence–Evidence–Anatomy (SEA) Grounding 将每个诊断声明显式关联到临床证据和 3D 脑部解剖，并用可执行规则驱动的 GRPO 强化微调确保临床一致性。

**[EquivAnIA: A Spectral Method for Rotation-Equivariant Anisotropic Image Analysis](equivania_a_spectral_method_for_rotation-equivariant_anisotropic_image_analysis.md)**

:   提出EquivAnIA，用定向滤波器族（cake wavelets和ridge filters）在频域中做带权平均来估计图像的角度分布，替代传统angular binning方法，实现对数值旋转真正鲁棒的各向异性分析，合成图像主方向估计误差仅0.03°，CT配准误差仅0.02°。

**[Event-Level Detection of Surgical Instrument Handovers in Videos](event_level_detection_of_surgical_instrument_handovers_in_videos.md)**

:   提出面向真实手术视频中器械交接检测的时空视觉框架，结合 ViT 空间特征提取和单向 LSTM 时序建模，通过多任务学习联合预测交接事件和方向，在肾移植手术视频上达到 F1=0.84 的检测性能。

**[Every Error has Its Magnitude: Asymmetric Mistake Severity Training for Multiclass Multiple Instance Learning](every_error_has_its_magnitude_asymmetric_mistake_severity_training_for_multiclas.md)**

:   提出 PAMS（Priority-Aware Mistake Severity）方法，通过非对称严重性感知的交叉熵损失（MSCE）、语义特征混合（SFR）和非对称 Mikel's Wheel 指标，在多分类 MIL WSI 诊断中显著降低严重误诊风险。

**[Extending ZACH-ViT to Robust Medical Imaging: Corruption and Adversarial Stress Testing in Low-Data Regimes](extending_zach-vit_to_robust_medical_imaging_corruption_and_adversarial_stress_t.md)**

:   在低数据医学影像场景下，对置换不变的紧凑型 ViT 架构 ZACH-ViT 进行首次鲁棒性扩展评估。在 7 个 MedMNIST 数据集上，ZACH-ViT 在干净数据和常见损坏下均排名第一（Mean Rank 1.57），在 FGSM 下排名最佳（2.00），PGD 下排名第二（2.29）。

**[Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning](fair_lung_disease_diagnosis_from_chest_ct_via_gend.md)**

:   在 ConvNeXt-Base 骨干上构建注意力 MIL 模型，用 GRL 对抗性消除扫描表示中的性别信息，配合 focal loss（$\gamma=2$）+ 标签平滑（$\varepsilon=0.1$）、子群过采样和 5-fold 集成，在 889 例胸部 CT 四类诊断中实现均值竞赛分数 0.685±0.030，女性 macro-F1（0.691）略高于男性（0.679），验证了 GRL 能有效闭合公平性差距。

**[Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)**

:   提出 FedMEPD 框架，用模态专属编码器处理模态间异质性、滤波器级动态部分个性化解码器平衡知识共享与个性化、多锚点跨注意力校准补偿缺失模态信息，在 BraTS 2018/2020 上全面超越现有多模态联邦学习方法。

**[FedVG: Gradient-Guided Aggregation for Enhanced Federated Learning](fedvg_gradient-guided_aggregation_for_enhanced_federated_learning.md)**

:   FedVG 提出利用全局验证集上的逐层梯度范数为各客户端打分，梯度越平坦（范数越小）的客户端获得越高聚合权重，从而在高度数据异质性场景下显著提升联邦学习的泛化性能。

**[Focus-to-Perceive Representation Learning: A Cognition-Inspired Hierarchical Framework for Endoscopic Video Analysis](focus-to-perceive_representation_learning_a_cognition-inspired_hierarchical_fram.md)**

:   提出 FPRL，一个受临床认知启发的层次化自监督框架，通过先"聚焦"帧内病灶关键静态语义、再"感知"帧间上下文演化来缓解运动偏差，在 11 个内窥镜数据集上取得 SOTA。

**[Forecasting Epileptic Seizures from Contactless Camera via Cross-Species Transfer Learning](forecasting_epileptic_seizures_from_contactless_ca.md)**

:   首次系统定义基于纯视频的癫痫发作预测（forecasting）任务（用 3-10 秒发作前片段预测未来 5 秒内是否发作），提出两阶段跨物种迁移学习框架——在啮齿类+人类混合视频上自监督预训练 VideoMAE，再在极少人类癫痫视频上做少样本微调——在 2/3/4-shot 设定下平均 bacc 达 72.30%、roc_auc 达 75.58%，超越所有视频理解 baseline。

**[Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)**

:   提出 FORGE，首个专为跨站点 fMRI 脑疾病诊断设计的持续学习框架，通过结构感知 VAE 生成逼真的功能连接矩阵进行隐私保护式生成回放，结合双层知识蒸馏和层次化上下文赌博机采样策略，有效缓解灾难性遗忘。

**[GaussianPile: A Unified Sparse Gaussian Splatting Framework for Slice-based Volumetric Reconstruction](gaussianpile_a_unified_sparse_gaussian_splatting_framework_for_slice-based_volum.md)**

:   提出 GaussianPile，通过引入焦点感知的物理成像模型（Focus Gaussian），将 3D 高斯溅射从表面外观建模扩展到切片体数据重建，在超声和光片显微镜数据上实现了比 NeRF 方法快 11 倍、比体素网格储存缩小 16 倍的高质量体数据压缩与重建。

**[GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](giim_graphbased_learning_of_inter_and_intraview_de.md)**

:   提出 GIIM 框架，基于多异构图（MHG）通过四类边关系同时建模同一病灶跨期相动态变化和不同病灶间空间关联，并设计四种缺失视图填充策略，在肝脏 CT、乳腺 X 光和乳腺 MRI 三种模态上均显著优于现有方法。

**[GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification](gleam_a_multimodal_imaging_dataset_and_hamm_for_gl.md)**

:   提出首个公开三模态青光眼数据集 GLEAM（SLO 眼底图 + 环乳头 OCT + 视野偏差图，1200例，四阶段标注），以及基于 CNN 的层级注意力掩码建模框架 HAMM，通过临床启发式的多头模态门控和关系图注意力实现跨模态融合，四分类准确率达 81.08%。

**[Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)**

:   提出 GenEval，通过域共形界（DCB）量化因果覆盖差距，并将人类专家知识量化精炼后与医学 VLM（MedGemma-4B）融合，以 LoRA 微调实现单源域泛化，在 DR 分级和癫痫灶检测上显著超越基线。

**[Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset](instruction-guided_lesion_segmentation_for_chest_x-rays_with_automatically_gener.md)**

:   提出指令引导的胸部X光病变分割任务（ILS），构建了首个大规模自动生成的指令-回答数据集MIMIC-ILS（1.1M样本、192K图像、91K mask），并训练ROSALIA模型实现gIoU 71.2%和空目标准确率91.8%，远超现有通用和医学分割模型。

**[Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)**

:   发现并解决了 CLIP 在跨域少样本学习（CDFSL）中的局部特征对齐退化问题，提出基于循环一致性的 CC-CDFSL 框架，通过 T-I-T 和 I-T-I 双向循环路径和语义锚点机制改善 patch 级视觉-语言对齐，同时增强模型的可解释性。

**[Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)**

:   提出 MASS（MAsk-guided Self-Supervised learning），利用 SAM2 自动生成的类别无关 mask 作为伪标注，以 in-context 分割为 pretext task 进行自监督预训练，无需任何人工标注即可学到语义丰富、泛化性强的 3D 医学图像表征，在 few-shot 分割和冻结编码器分类上均取得优异表现。

**[LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings](lemon_a_large_endoscopic_monocular_dataset_and_foundation_model_for_perception_in.md)**

:   构建了包含 4194 个手术视频（938 小时）的大规模内窥镜数据集 LEMON，并提出基于增强知识蒸馏的自监督基础模型 LemonFM，在手术阶段识别、工具检测、动作识别和语义分割四大下游任务上全面超越现有手术基础模型。

**[LUMINA: A Multi-Vendor Mammography Benchmark with Energy Harmonization Protocol](lumina_a_multi-vendor_mammography_benchmark_with_energy_harmonization_protocol.md)**

:   提出 LUMINA 多厂商乳腺 FFDM 数据集（468 例患者、1824 张图像），附带前景像素直方图匹配的能量协调预处理方法，在诊断/BI-RADS/密度三任务上系统评估了 CNN 与 Transformer 模型。

**[MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation](medclipseg_probabilistic_vision-language_adaptation_for_data-efficient_and_gener.md)**

:   在冻结CLIP编码器的基础上，通过概率交叉模态注意力（PVL）实现图文双向交互与预测不确定性建模，配合软patch级对比损失，在16个医学分割数据集上兼顾数据效率、域泛化能力和可解释性。

**[MedGEN-Bench: Contextually Entangled Benchmark for Open-Ended Multimodal Medical Generation](medgen-bench_contextually_entangled_benchmark_for_open-ended_multimodal_medical_.md)**

:   提出 MedGEN-Bench，首个面向开放式多模态医学生成的综合基准，包含 6,422 个专家验证的图文对、6 种成像模态、16 个临床任务，配套三层评估框架，揭示了组合框架优于统一模型的跨模态一致性问题。

**[MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)**

:   MedGRPO 提出了两项关键创新解决医学视频多数据集强化学习中的训练崩溃问题：跨数据集奖励归一化（用 logistic 函数将不同难度数据集的中位表现映射到相同奖励值）和医学 LLM 评审（通过五个临床维度的比较性评分），基于 Qwen2.5-VL-7B 在 MedVidBench（532K 视频指令对）上超越 GPT-4.1 和 Gemini-2.5-Flash。

**[MedKCO: Medical Vision-Language Pretraining via Knowledge-Driven Cognitive Orchestration](medkco_medical_vision-language_pretraining_via_knowledge-driven_cognitive_orches.md)**

:   提出 MedKCO，一种知识驱动的认知编排策略用于医学视觉-语言预训练：通过分层课程（label-level 按诊断敏感度排序 + description-level 按样本代表性排序）和自步非对称对比损失，让模型从简单到复杂渐进学习，在三种医学模态的零样本和下游任务上显著超越基线。

**[Meta-learning In-Context Enables Training-Free Cross Subject Brain Decoding](meta-learning_in-context_enables_training-free_cross_subject_brain_decoding.md)**

:   提出 BrainCoDec 框架，通过两阶段层级式上下文学习（先为每个体素估计编码器参数，再跨体素聚合做功能反演），实现了无需微调即可泛化到新被试的 fMRI 视觉解码，Top-1 检索准确率从 MindEye2 的 3.9% 提升到 22.7%。

**[MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](milpf_multiple_instance_learning_on_precomputed_fe.md)**

:   将冻结的通用基础编码器（DINOv2 ViT-Giant / MedSigLIP）与仅 ~40k 参数的轻量 MIL 聚合头结合，通过预计算特征 + 双流聚合（全局均值 + 局部 Perceiver 交叉注意力），在 EMBED 等大规模乳腺 X 线分类基准上以 5-7 分钟训练达到 SOTA（AUC 0.916, Spec@Sens=0.9 达 0.762），可训练参数比基线少 35-458 倍。

**[Modeling Spatiotemporal Neural Frames for High Resolution Brain Dynamics](modeling_spatiotemporal_neural_frames_for_high_resolution_brain_dynamic.md)**

:   提出基于扩散 Transformer 的 EEG 条件 fMRI 重建框架，将脑活动建模为时空神经帧序列而非独立快照，在皮层顶点级分辨率下实现时空一致的 fMRI 重建，并通过零空间采样支持中间帧插值，下游视觉解码任务验证了功能信息的保留。

**[Momentum Memory for Knowledge Distillation in Computational Pathology](momentum_memory_for_knowledge_distillation_in_computational_pathology.md)**

:   提出 MoMKD，用动量更新的类条件记忆库替代传统 batch-local 特征对齐，实现基因组→病理切片的跨模态知识蒸馏，仅用 H&E 切片推理即可获得基因组级预测能力。

**[MRI Contrast Enhancement Kinetics World Model](mri_contrast_enhancement_kinetics_world_model.md)**

:   首次提出 MRI 造影增强动力学世界模型（MRI CEKWorld），通过时空一致性学习（STCL）在稀疏采样数据上实现从无造影 MRI 到连续高保真造影增强序列的生成，解决了内容失真和时序不连续两大难题。

**[Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)**

:   提出 RICE-NET，一个多模态 3D ResNet-18 模型，整合纵向 MRI 数据与放疗剂量分布图，用于自动区分胶质母细胞瘤术后放射诱导对比增强（RICE）与肿瘤复发，在独立测试集上达到 F1=0.92。

**[MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning](multimodalpfn_extending_prior-data_fitted_networks_for_multimodal_tabular_learni.md)**

:   提出 MMPFN，首次将预训练表格基础模型 TabPFN 扩展到多模态（表格+图像/文本）场景，通过多头门控 MLP（MGM）和交叉注意力池化器（CAP）解决非表格嵌入过压缩和 token 数量不平衡问题，在医学和通用数据集上超越 SOTA。

**[Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)**

:   提出 MSG-LDM，在潜在扩散模型中引入多尺度结构-风格解耦机制，通过高频注入、多模态结构特征融合和结构感知损失，实现缺失模态场景下保留解剖结构和精细细节的多模态 MRI 合成。

**[MUSE: Harnessing Precise and Diverse Semantics for Few-Shot Whole Slide Image Classification](muse_harnessing_precise_and_diverse_semantics_for_few-shot_whole_slide_image_cla.md)**

:   提出 MUSE 框架，通过 MoE 驱动的样本级细粒度语义增强（SFSE）和基于 LLM 知识库的随机多视角语义优化（SMMO），在少样本全切片图像分类任务上显著提升泛化能力。

**[MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)**

:   提出 MUST 框架，通过代数约束将多模态表征显式分解为模态特有和跨模态共享两部分，并用条件潜在扩散模型在模态缺失时生成特有信息，在五个 TCGA 癌症数据集上以 0.742 C-index 达到 SOTA，且在模态缺失场景下仅降约 0.4%-3.5%。

**[MuViT: Multi-Resolution Vision Transformers for Learning Across Scales in Microscopy](muvit_multi-resolution_vision_transformers_for_learning_across_scales_in_microsc.md)**

:   提出 MuViT，一种基于世界坐标 RoPE 位置编码的多分辨率 Vision Transformer，能在单一编码器中联合处理同一场景不同物理分辨率的裁剪图，在显微镜图像分割任务上显著优于单分辨率基线。

**[NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization](neuroseg_meets_dinov3_transferring_2d_self-supervised_visual_priors_to_3d_neuron.md)**

:   NeurINO 提出通过将 DINOv3 预训练的 2D 卷积核膨胀（inflate）为 3D 算子来初始化 3D 神经元分割模型，同时引入拓扑感知骨架损失（TASL）显式监督骨架级结构保真性，在四个神经影像数据集上 ESA 平均提升 2.9%、DSA 提升 2.8%、PDS 提升 3.8%。

**[Novel Architecture of RPA In Oral Cancer Lesion Detection](novel_architecture_of_rpa_in_oral_cancer_lesion_detection.md)**

:   将软件设计模式（Singleton + Batch Processing）集成到基于 EfficientNetV2B1 的口腔癌病变检测 Python 流水线中，相比传统 RPA 平台（UiPath/Automation Anywhere）实现 60-100x 推理加速（每张图 0.06s vs 2.58s），同时保持诊断准确性。

**[OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)**

:   提出 OmniFM，一个模态鲁棒且任务无关的联邦学习框架，通过频域频谱知识检索、嵌入式交叉注意力融合和前缀-后缀频谱提示三个互补组件，在一个统一的 FL pipeline 下支持分类、分割、超分辨率、VQA 和多模态融合五种医学影像任务，并在跨模态异构场景下显著超越现有基线。

**[OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation](orapo_oracle-educated_reinforcement_learning_for_data-efficient_and_factual_radi.md)**

:   提出 OraPO（Oracle-educated GRPO），在 GRPO 探索失败时注入轻量 DPO 监督将失败 rollout 转化为偏好样本，配合 FactScore 奖励实现仅用 1K 样本、3B 小模型在 CheXpert Plus 和 MIMIC-CXR 上达到放射报告生成 SOTA（F1=0.341/0.357），训练数据量比前最优减少 2-3 个数量级。

**[Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification](parameter-efficient_prompt_tuning_and_hierarchical_textual_guidance_for_few-shot.md)**

:   HIPSS 提出了两个关键创新用于少样本 WSI 分类：(1) 基于缩放和偏移特征（SSF）的参数高效 prompt 调优替代 CoOp，大幅减少可训练参数；(2) 软层次化文本引导策略无需硬过滤即可利用 VLM 的预训练知识和 WSI 的固有层次结构。在三个癌症数据集上最高提升 13.8%。

**[PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation](pgr-net_prior-guided_roi_reasoning_network_for_brain_tumor_mri_segmentation.md)**

:   PGR-Net 提出了一种显式 ROI 感知的脑肿瘤 MRI 分割网络，通过从训练集构建数据驱动的空间先验模板、层级 Top-K ROI 选择机制和窗口高斯-空间衰减引导模块（WinGS-ROI），将计算资源集中于病灶区域，仅用 8.64M 参数就在 BraTS-2019/2023 和 MSD Task01 上达到了 SOTA。

**[Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](prototypebased_knowledge_guidance_for_finegrained.md)**

:   提出 ProtoSR，通过 LLM 驱动的管道从 22.7 万篇 MIMIC-CXR 自由文本报告中挖掘模板对齐的视觉原型知识库，并设计原型条件化迟融合模块将检索到的原型证据作为 logit 残差注入层级式结构化报告模型，在 Rad-ReStruct 基准上达到 SOTA，L3 细粒度属性 F1 从 4.3 提升到 7.4（+72.1% 相对提升）。

**[RDFace: A Benchmark Dataset for Rare Disease Facial Image Analysis under Extreme Data Scarcity and Phenotype-Aware Synthetic Generation](rdface_a_benchmark_dataset_for_rare_disease_facial_image_analysis_under_extreme_.md)**

:   构建了包含 456 张儿童面部图像、覆盖 103 种罕见遗传疾病的标准化基准数据集 RDFace，并系统研究了表型感知的合成数据增强（DreamBooth/FastGAN）在超低样本罕见病诊断中的效果，DreamBooth 增强在极端低数据场景下最高可提升 13.7% 的诊断准确率。

**[Reclaiming Lost Text Layers for Source-Free Cross-Domain Few-Shot Learning](reclaiming_lost_text_layers_for_source-free_cross-domain_few-shot_learning.md)**

:   发现 CLIP 文本编码器中存在"Lost Layers"——在 Source-Free Cross-Domain Few-Shot Learning (SF-CDFSL) 中移除某些中间层反而提升性能；论文证明这些层并非冗余而是因视觉域偏移未被充分利用，提出 VtT 模型在层级和编码器级别重新利用这些信息，取得 SOTA。

**[Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](reinforcing_the_weakest_links_modernizing_siena_wi.md)**

:   将 SIENA 纵向脑萎缩管线中的经典颅骨剥离(BET2)和组织分割(FAST)模块定向替换为深度学习方案(SynthStrip/SynthSeg)，在 ADNI (N=1006) 和 PPMI (N=310) 两个大规模纵向队列上显著增强了 PBVC 与临床疾病进展的关联性(相关系数提升超 100%)，扫描顺序误差降低高达 99.1%。

**[RelativeFlow: Taming Medical Image Denoising Learning with Noisy Reference](relativeflow_taming_medical_image_denoising_learning_with_noisy_reference.md)**

:   提出 RelativeFlow，基于 flow matching 的框架，通过将绝对噪声到干净映射分解为相对更噪到噪声映射，结合一致传输约束和基于模拟的速度场，从异质噪声参考中学习统一的去噪流，突破参考偏差限制。

**[Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)**

:   提出 Residual SODAP 框架，通过 α-entmax 稀疏提示选择+残差聚合、无数据统计蒸馏+伪特征回放、提示使用模式漂移检测，以及不确定性加权多损失平衡，联合解决提示端表征适应和分类器端知识保持问题，在医学域增量学习上达到 SOTA。

**[Robust Fair Disease Diagnosis in CT Images](robust_fair_disease_diagnosis_in_ct_images.md)**

:   本文提出结合Logit调整交叉熵（处理类别不平衡）和CVaR聚合（处理人口统计公平性）的双层目标函数，在CT诊断中实现了性别平均macro F1达0.8403且公平性差距仅0.0239。

**[Robust Multi-Source Covid-19 Detection in CT Images](robust_multi-source_covid-19_detection_in_ct_images.md)**

:   提出一种多任务学习框架，在共享 EfficientNet-B7 骨干上同时训练 COVID-19 诊断头和来源医院识别头（使用 logit-adjusted 损失），推动特征提取器学习跨机构不变的表示，在多源 CT 数据集上 F1 达到 0.9098。

**[SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation](sd_fsmis_adapting_stable_diffusion_for_few_shot_medical_image_segmentation.md)**

:   提出 SD-FSMIS，一个将预训练 Stable Diffusion 适配到少样本医学图像分割的框架，通过支持-查询交互模块和视觉到文本条件转换器实现高效适配，在跨域场景中表现尤为突出。

**[Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)**

:   本文提出 SCDL（Semantic Class Distribution Learning），一个即插即用模块，通过类别分布双向对齐（CDBA）学习结构化的类条件特征分布并与可学习类代理双向对齐，结合语义锚点约束（SAC）利用标注数据引导代理学习正确语义，缓解了半监督医学图像分割中的监督偏差和特征表示偏差，在尾类器官上取得了显著提升。

**[SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)**

:   提出 SemiTooth 框架，通过多教师多学生架构和严格加权置信度约束（SWC），解决多源 CBCT 牙齿分割中的标注稀缺和跨源域间差异问题，同时构建了首个多源半监督牙齿数据集 MS3Toothset。

**[SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semisupervised_framework.md)**

:   本文提出SemiTooth框架，通过多教师-多学生架构和更严格的加权置信度约束（SWC），解决多源CBCT数据在半监督牙齿分割中的分布差异问题，在构建的MS3Toothset数据集上取得SOTA。

**[Solving a Nonlinear Blind Inverse Problem for Tagged MRI with Physics and Deep Generative Priors](solving_a_nonlinear_blind_inverse_problem_for_tagged_mri_with_physics_and_deep_g.md)**

:   提出 InvTag 框架，首次将 MR 物理前向模型与预训练扩散生成先验结合，统一解决 3D Tagged MRI 的解剖恢复、Cine 合成和运动估计三大子任务，且无需任何额外训练数据。

**[STEPH: Sparse Task Vector Mixup with Hypernetworks for Efficient Knowledge Transfer in WSI Prognosis](sparse_task_vector_mixup_wsi_prognosis.md)**

:   STEPH 提出基于任务向量混合（TVM）+ 超网络驱动稀疏聚合的模型合并方案，将多个癌种特定预后模型的知识高效融入目标癌种模型，在 13 个 TCGA 数据集上 C-Index 平均 0.6949（+5.14% vs 癌种特定学习、+2.01% vs ROUPKT），且推理仅需单模型前向传播，远低于多模型表示迁移方案。

**[SPEGC: Continual Test-Time Adaptation via Semantic-Prompt-Enhanced Graph Clustering for Medical Image Segmentation](spegc_continual_test-time_adaptation_via_semantic-prompt-enhanced_graph_clusteri.md)**

:   提出 SPEGC 框架，通过语义提示增强特征 + 可微分图聚类求解器，将原始相似度矩阵精炼为高阶结构表示，用于指导医学图像分割模型在持续变化的目标域上自适应，有效缓解误差累积与灾难性遗忘。

**[Synergistic Bleeding Region and Point Detection in Laparoscopic Surgical Videos](synergistic_bleeding_region_and_point_detection_in_laparoscopic_surgical_videos.md)**

:   构建首个腹腔镜手术出血区域+出血点标注数据集 SurgBlood，并提出基于 SAM2 的双分支双向引导在线检测器 BlooDet，通过 Mask/Point 分支协同优化实现出血区域分割与出血点定位的联合检测。

**[T-Gated Adapter: A Lightweight Temporal Adapter for Vision-Language Medical Segmentation](t-gated_adapter_a_lightweight_temporal_adapter_for_vision-language_medical_segme.md)**

:   提出轻量级时序门控适配器（T-Gated Adapter），为2D视觉语言模型CLIPSeg注入相邻切片上下文，在仅30个标注CT体积上训练即可实现平均Dice 0.704（+0.206），跨域零样本评估和CT到MRI跨模态评估中均一致提升。

**[Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)**

:   提出 Tell2Adapt 统一框架，利用视觉基础模型（BiomedParse）的泛化知识，通过上下文感知提示正则化（CAPR）生成高质量伪标签，再经视觉合理性精炼（VPR）去除解剖学不合理预测，实现跨 10 个域迁移方向、22 个解剖目标的统一无源域自适应医学图像分割。

**[The Invisible Gorilla Effect in Out-of-distribution Detection](the_invisible_gorilla_effect_in_out-of-distribution_detection.md)**

:   揭示了OOD检测中一个此前未被报告的偏差——"隐形大猩猩效应"：当OOD伪影与模型关注区域（ROI）视觉外观相似时检测性能显著更好，不相似时则大幅下降，尤其影响基于特征的OOD方法。

**[Transformer-Based Multi-Region Segmentation and Radiomic Analysis of HR-pQCT Imaging for Osteoporosis Classification](transformer-based_multi-region_segmentation_and_radiomic_analysis_of_hr-pqct_ima.md)**

:   提出基于 SegFormer 的全自动多区域 HR-pQCT 分割框架，结合影像组学特征与机器学习实现骨质疏松二分类，发现软组织（肌腱/脂肪）特征的诊断价值优于传统骨骼特征。

**[Ultrasound-CLIP: Semantic-Aware Contrastive Pre-training for Ultrasound Image-Text Understanding](ultrasound-clip_semantic-aware_contrastive_pre-training_for_ultrasound_image-tex.md)**

:   这篇论文的核心贡献不是只做了一个“超声版 CLIP”，而是围绕超声特有的解剖层级和诊断属性重新定义了图文对齐目标：先构建超声知识体系 UDT 和大规模 US-365K 数据集，再用语义软标签与属性异构图把文本里的临床关系显式注入对比学习，从而得到更像“懂超声”的视觉语言表示。

**[Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)**

:   提出 SMART 框架，基于 SAM3 的概念提示分割构建 Teacher-Student 半监督模型，结合渐进置信度正则化和双流时序一致性策略，仅用极少标注在 X 射线冠脉造影视频中实现 SOTA 血管分割。

**[UNIStainNet: Foundation-Model-Guided Virtual Staining of H&E to IHC](unistainnet_foundation-model-guided_virtual_staining_of_he_to_ihc.md)**

:   提出 UNIStainNet，首次将冻结的病理基础模型 UNI 的密集空间 token 作为 SPADE 调制信号直接注入生成器，配合错位感知损失和可学习染色嵌入，用单一模型同时生成 HER2/Ki67/ER/PR 四种 IHC 染色，在 MIST 和 BCI 基准上取得 SOTA 分布式指标。

**[Unleashing Video Language Models for Fine-grained HRCT Report Generation](unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)**

:   提出 AbSteering 两阶段框架，利用异常中心的 CoT 推理和 DPO 硬负样本对比学习，将通用 VideoLM 高效适配到 HRCT 报告生成，在临床效能指标上大幅超越专用 CT 基础模型。

**[Unlocking Multi-Site Clinical Data: A Federated Approach to Privacy-First Child Autism Behavior Analysis](unlocking_multi-site_clinical_data_a_federated_approach_to_privacy-first_child_a.md)**

:   本文提出首个面向儿童自闭症行为识别的联邦学习框架，通过 3D 骨骼抽象化（消除身份信息）+ 联邦优化（数据不出站点）的双层隐私策略，在 MMASD 数据集上用 APFL 个性化联邦方法达到 87.80% 准确率，比本地训练高 5.2%，同时满足 HIPAA/GDPR 隐私合规要求。

**[Unlocking Positive Transfer in Incrementally Learning Surgical Instruments: A Self-reflection Hierarchical Prompt Framework](unlocking_positive_transfer_in_incrementally_learning_surgical_instruments_a_sel.md)**

:   这篇论文把每个器械类的提示参数从“彼此隔离的独立 prompt”改造成“共享知识逐层拆解的树结构”，让新器械可以继承旧知识快速学会，同时让新知识反过来温和修正旧知识，从而在手术器械类增量分割中同时提升新类、常见类和旧类表现。

**[Unsupervised Domain Adaptation with Target-Only Margin Disparity Discrepancy](unsupervised_domain_adaptation_with_target-only_margin_disparity_discrepancy.md)**

:   针对 CT→CBCT 肝脏分割的无监督域自适应问题，发现经典 MDD 优化目标中存在矛盾项（源域上特征提取器被优化为最大化 $f$ 和 $f'$ 的差异），提出 Target-Only MDD 改进，去除矛盾项并在两域上统一最小化预测差异，在 2D 和 3D 实验中均取得 UDA SOTA。

**[Virtual Full-stack Scanning of Brain MRI via Imputing Any Quantised Code](virtual_full-stack_scanning_of_brain_mri_via_imputing_any_quantised_code.md)**

:   提出 CodeBrain，将脑 MRI 任意到任意模态补全问题重新表述为区域级全栈量化码预测任务，通过两阶段流程（标量量化重建 + 分级损失码预测）实现统一的缺失模态合成，超越五种 SOTA 方法。

**[Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation](weakly_supervised_teacher-student_framework_with_progressive_pseudo-mask_refinem.md)**

:   提出弱监督教师-学生框架，利用稀疏病理标注和 EMA 稳定的教师网络生成渐进式精炼伪掩码，结合置信度过滤、自适应融合和课程引导精炼策略，实现结直肠癌病理图像中腺体结构的高效分割。

**[X-WIN: Building Chest Radiograph World Model via Predictive Sensing](x-win_building_chest_radiograph_world_model_via_predictive_sensing.md)**

:   提出 X-WIN 胸片世界模型，首次将 3D CT 空间知识融入 CXR 表征学习：通过学习预测 CT 在不同旋转角度下的 2D 投影来内化 3D 解剖结构，配合亲和力引导的对比对齐和结构保持域自适应，在 6 个 CXR 基准上通过线性探测取得 SOTA。
