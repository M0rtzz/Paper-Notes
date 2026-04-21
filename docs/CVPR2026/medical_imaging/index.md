---
title: >-
  CVPR2026 医学图像方向 150篇论文解读
description: >-
  150篇CVPR2026 医学图像方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**📷 CVPR2026** · **150** 篇论文解读

**[A protocol for evaluating robustness to H&E staining variation in computational pathology models](a_protocol_for_evaluating_robustness_to_he_stainin.md)**

:   提出三步评估协议（选参考染色条件→表征测试集染色属性→模拟染色条件推理），系统量化306个MSI分类模型对H&E染色差异的鲁棒性，发现鲁棒性与分类性能呈弱负相关(r=-0.28)，高性能不代表高鲁棒性。

**[A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)**

:   提出面向乳腺超声（BUS）图像分割的半监督框架，利用 GPT-5 生成外观描述 + Grounding DINO + SAM 免训练生成伪标签（APPG），结合双教师框架（静态+动态）通过不确定性-熵加权融合（UEWF）和自适应不确定性引导反向对比学习（AURCL）精炼标签，仅用 2.5% 标注即接近全监督性能。

**[A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semisupervised_framework_for_breast_ultrasound_s.md)**

:   利用简单外观描述（"dark oval"等）驱动 Grounding DINO + SAM 免训练生成乳腺超声伪标签，再通过双教师不确定性-熵加权融合与自适应反向对比学习精炼伪标签质量，仅 2.5% 标注即达到甚至超过全监督上界。

**[Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](accelerating_stroke_mri_with_diffusion_probabilist.md)**

:   提出一种受基础模型范式启发的训练策略，先在大规模多对比度脑部 MRI 数据上预训练扩散概率模型（DPM），再用仅 20 例目标域数据微调，实现数据受限场景下与大数据集训练可比的 MRI 加速重建质量，临床盲评显示从 2× 加速数据重建的图像与标准诊疗不相上下。

**[Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](accelerating_stroke_mri_with_diffusion_probabilistic_models_through_large-scale_.md)**

:   借鉴基础模型的"预训练+微调"范式，在 ~4000 名 fastMRI 受试者（多对比度）上大规模预训练扩散概率模型（DPM），然后用极少目标域数据（20名受试者）低学习率微调，实现跨对比度、跨采集协议的 MRI 加速重建；临床中风验证中 2× 加速图像质量经神经放射科医生盲法评估 non-inferior 于标准全采样图像。

**[Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning](act_like_a_pathologist_tissue-aware_whole_slide_image_reasoning.md)**

:   提出 HistoSelect 框架，模拟病理学家从粗到细的推理过程，通过组织分割→Group Sampler→Patch Selector 的三级筛选机制，基于信息瓶颈(IB)理论压缩无关视觉token，在减少约70%计算量的同时实现三个数据集上的SOTA。

**[Active Inference for Micro-Gesture Recognition: EFE-Guided Temporal Sampling and Adaptive Learning](active_inference_for_micro-gesture_recognition_efe-guided_temporal_sampling_and_.md)**

:   提出 UAAI 框架，首次将主动推理(Active Inference)引入微手势识别，通过 EFE 引导的时间帧选择 + 空间注意力 + UMIX不确定性感知增强，在SMG数据集RGB模态上达到63.47%，大幅超越传统RGB方法。

**[Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](adaptation_of_weakly_supervised_localization_in_hi.md)**

:   提出SFDA-DeP，受机器遗忘启发将源自由域适应（SFDA）建模为迭代识别并纠正预测偏差的过程——选择性降低优势类中不确定样本的置信度、保留可靠预测、联合训练像素级分类器恢复定位判别力——在跨器官/跨中心病理基准上一致优于SFDA baselines的分类和定位性能。

**[Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](adaptation_of_weakly_supervised_localization_in_histopathology_by_debiasing_pred.md)**

:   提出 SFDA-DeP，受机器遗忘启发，将 SFDA 重新定义为"识别并纠正预测偏差"的迭代过程：对 dominant class 中高熵不确定样本执行"遗忘"操作迫使模型放弃偏向性预测，对可靠样本保持自训练，同时用像素级分类器锚定定位能力，在跨器官/跨中心病理基准上持续优于现有 SFDA 方法。

**[Adaptive Confidence Regularization for Multimodal Failure Detection](adaptive_confidence_regularization_for_multimodal_failure_detection.md)**

:   提出 ACR 框架，通过自适应置信度损失（惩罚多模态融合置信度低于单模态的"置信度退化"现象）和多模态特征交换（在特征空间合成失败样本）两个互补模块，首次系统解决多模态场景下的误分类检测问题，在四个数据集上全面超越已有方法。

**[Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](addressing_data_scarcity_in_3d_trauma_detection_th.md)**

:   在仅 206 例标注(其中 144 例用于训练)的极端稀缺条件下，通过 patch-based MIM 预训练 3D U-Net + VDETR 顶点 RPE 检测器 + 2000 例未标注数据的半监督一致性正则化，将 3D 腹部创伤检测 mAP@0.50 从 26.36% 提升至 56.57%(验证集,+115%)，冻结编码器的 7 类分类达 94.07% 准确率。

**[Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](addressing_data_scarcity_in_3d_trauma_detection_through_self-supervised_and_semi.md)**

:   提出两阶段标签高效框架：先用 patch-based MIM 在1,206个无标注CT上自监督预训练3D U-Net编码器，再用VDETR+3D顶点相对位置编码做3D损伤检测，配合Mean Teacher半监督一致性正则化利用2,000个无标注体数据，仅用144个有标注样本即实现56.57% val mAP@0.50（比纯监督提升115%）。

**[Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study](are_general-purpose_vision_models_all_we_need_for_2d_medical_image_segmentation_.md)**

:   通过统一训练协议在三个异质医学数据集上对比 11 种模型，发现通用视觉模型（GP-VMs）在标准化条件下系统性超越大多数专用医学分割架构（SMAs），挑战了"医学分割必须使用专用架构"的传统认知。

**[Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study](are_generalpurpose_vision_models_all_we_need_for_2.md)**

:   在统一训练评估协议下对比11个模型（5个专用医学分割架构SMA + 6个通用视觉模型GP-VM）在3个异构医学数据集上的表现，发现GP-VMs在所有数据集上系统性优于大多数SMAs（平均mDSC: VW-MiT 91.0% vs 最佳SMA SU-Mamba 90.5%），且Grad-CAM分析表明GP-VMs能捕获临床相关结构。

**[Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts](association_of_radiologic_ppfe_change_with_mortali.md)**

:   在两个独立大规模肺癌筛查队列中，利用深度学习自动分割量化PPFE纵向变化，首次验证其在筛查人群中的独立预后价值。

**[Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts](association_of_radiologic_ppfe_change_with_mortality_in_lung_cancer_screening_co.md)**

:   在两个大规模肺癌筛查队列（NLST n=7980, SUMMIT n=8561）上，利用深度学习自动分割 PPFE 体积并定义"进展性 PPFE"，通过 Cox 比例风险模型证明 PPFE 进展是全因死亡率的独立预测因子（NLST HR=1.25, SUMMIT HR=3.14），并与呼吸入院率、抗生素/类固醇使用等临床终点显著关联。

**[Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](automated_detection_of_malignant_lesions_in_the_ov.md)**

:   系统对比 15 种 CNN 变体（LeNet/ResNet/VGG/Inception）在卵巢癌组织病理图像五分类上的表现，最终选出 InceptionV3-A（ReLU）达 94% 综合指标，并用 LIME/SHAP/Integrated Gradients 三种 XAI 方法做对比解释分析。

**[Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](automated_detection_of_malignant_lesions_in_the_ovary_using_deep_learning_models.md)**

:   系统地比较了 LeNet/ResNet/VGG/Inception 四大CNN架构的15个变体在卵巢癌组织病理学图像分类上的表现，最终选择 InceptionV3-ReLU 作为基础模型(平均指标~94%)，并结合 LIME、SHAP、Integrated Gradients 三种 XAI 方法对分类结果进行可解释性分析。

**[Benchmarking Endoscopic Surgical Image Restoration and Beyond](benchmarking_endoscopic_surgical_image_restoration_and_beyond.md)**

:   构建了首个多源真实世界内窥镜手术图像复原数据集 SurgClean（3,113张图像，覆盖去烟/去雾/去飞溅三种退化类型），在其上系统评测了22种代表性图像复原方法（12种通用+10种任务特定），揭示现有方法与临床需求间仍存在显著差距，并进一步分析了手术场景退化与自然场景退化的本质差异。

**[Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance](better_than_average_spatially-aware_aggregation_of_segmentation_uncertainty_impr.md)**

:   首次系统研究分割任务中像素级不确定性到图像级分数的聚合策略，提出融合空间结构信息的聚合方法（基于Moran's I、Edge Density、Shannon Entropy的空间质量比SMR），以及GMM元聚合器，在10个数据集的OoD和故障检测任务上验证了空间感知聚合显著优于全局平均。

**[Beyond Pixel Simulation: Pathology Image Generation via Diagnostic Semantic Tokens and Prototype Control](beyond_pixel_simulation_pathology_image_generation_via_diagnostic_semantic_token.md)**

:   UniPath提出语义驱动的病理图像生成框架，通过多流控制（原始文本 + 从冻结病理MLLM蒸馏的诊断语义Token + 原型库形态控制）实现诊断级可控生成，Patho-FID达80.9，比第二名优51%。

**[BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)**

:   提出 BiCLIP 框架，通过双向多模态融合（BMF）实现视觉信息反向精炼文本表示，并通过图像增强一致性（IAC）约束中间特征的扰动不变性，在 COVID-19 CT 分割上超越 SOTA，仅 1% 标注数据仍保持鲁棒。

**[BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_languageimage.md)**

:   提出BiCLIP框架，通过双向多模态融合（BMF）模块让文本和视觉特征可以相互修正形成闭环，并用图像增强一致性（IAC）模块约束弱/强扰动下的中间特征一致性，在标注极度稀缺（仅1%）和图像退化（低剂量CT噪声/运动模糊）的临床场景下实现鲁棒医学图像分割。

**[Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)**

:   提出AnoPLe——一个轻量级多模态双向提示学习框架，无需手工异常描述或外部辅助模块，通过文本-视觉提示双向交互和尺度感知前缀实现少样本多类别异常检测，在MVTec-AD/VisA/Real-IAD上取得强竞争力的同时保持高效推理（~28 FPS）。

**[Bridging the Skill Gap in Clinical CBCT Interpretation with CBCTRepD](bridging_the_skill_gap_in_clinical_cbct_interpreta.md)**

:   构建覆盖55种口腔疾病的7,408例大规模CBCT-报告配对数据集，开发双语口腔颌面CBCT报告生成系统CBCTRepD，通过AI生成草稿+放射科医生编辑的协作模式，在多层级临床评估中证明其可帮助初级医生达到中级水平、中级医生接近高级水平、高级医生减少遗漏。

**[Bridging the Skill Gap in Clinical CBCT Interpretation with CBCTRepD](bridging_the_skill_gap_in_clinical_cbct_interpretation_with_cbctrepd.md)**

:   提出CBCTRepD——面向口腔颌面CBCT的双语报告生成系统，基于7408例高质量配对数据集训练，结合多层级评估框架验证其在放射科医生-AI协作工作流中对初级、中级、高级医生的分级赋能效果。

**[Can Natural Image Autoencoders Compactly Tokenize fMRI Volumes for Long-Range Dynamics Modeling?](can_natural_image_autoencoders_compactly_tokenize_fmri_volumes_for_long-range_dy.md)**

:   提出 TABLeT，利用预训练的 2D 自然图像自编码器（DCAE）将 3D fMRI 体积压缩为仅 27 个连续 token，配合简单 Transformer 编码器实现前所未有的长时序建模（256 帧），在 UKB、HCP、ADHD-200 上多任务超越 SOTA 体素方法，且计算效率大幅提升。

**[CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis](care_a_molecular-guided_foundation_model_with_adaptive_region_modeling_for_whole.md)**

:   提出 CARE，一种病理学 slide-level 基础模型，通过自适应区域生成器（ARG）将 WSI 划分为形态学相关的不规则区域（类似 NLP 中的词级 token），并结合 RNA/蛋白质表达谱的跨模态对齐进行两阶段预训练，仅用主流模型约 1/10 的数据即在 33 个下游任务上取得最优平均性能。

**[Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images](cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_.md)**

:   提出 CPNN，利用公开单细胞 RNA-seq 数据构建细胞类型原型（cell-type prototype），将 slide/patch 级基因表达建模为原型的加权组合，在基因表达估计任务上取得 SOTA 并提供可解释性。

**[CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection](chips_clip_adaptation_curvature_data_selection.md)**

:   从数据中心视角重新审视 CLIP 领域适配，提出 CHIPS，为每个图文对计算融合曲率感知牛顿对齐（忠实性）、JL sketching压缩曲率估计（可扩展性）、可学习性+领域相关性权重（保留性）三因素的效用分数，用30%数据匹配全数据集CPT、10%数据超越50%数据CPT，在17个医学+31个通用基准上达到选择SOTA。

**[CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection](chips_efficient_clip_adaptation_via_curvature-aware_hybrid_influence-based_data_.md)**

:   提出 CHIPS，一种基于曲率感知混合影响力的数据选择方法，在 CLIP 端点子空间中计算 Newton 风格对齐分数并结合可学习性与领域相关性权重，仅用 30% 数据即可匹配全量数据集持续预训练效果，在 17 个医学基准上达到 SOTA。

**[CLoE: Expert Consistency Learning for Missing Modality Segmentation](cloe_expert_consistency_learning_for_missing_modal.md)**

:   将缺失模态下的鲁棒性问题重新定义为决策级专家一致性控制，提出双分支一致性学习（全局MEC+区域REC）配合轻量门网络将一致性分数转化为模态可靠性权重，在BraTS 2020上15种缺失组合平均WT Dice达88.09%超越所有SOTA。

**[CLoE: Expert Consistency Learning for Missing Modality Segmentation](cloe_expert_consistency_learning_for_missing_modality_segmentation.md)**

:   提出 CLoE（Consistency Learning of Experts），将缺失模态鲁棒性问题建模为决策层面的专家一致性控制，通过模态专家一致性（MEC）和区域专家一致性（REC）双分支约束减少专家漂移，并用一致性分数驱动的门控网络实现可靠性加权融合。

**[CRFT: Consistent-Recurrent Feature Flow Transformer for Cross-Modal Image Registration](crft_consistent-recurrent_feature_flow_transformer_for_cross-modal_image_registr.md)**

:   提出CRFT，统一的粗到精跨模态图像配准框架——在Transformer架构中学习模态无关的特征流表示，粗阶段1/8分辨率全局对应+精阶段1/2-1/4多尺度局部细化，配合迭代差异引导注意力和空间几何变换(SGT)递归精化流场捕捉微妙空间不一致，在光学/红外/SAR/多光谱等多种跨模态数据集上超越RAFT/GMFlow/LoFTR等SOTA。

**[Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)**

:   提出 SpaHGC，一种基于多模态异构图的框架，通过构建目标切片内、跨切片和参考切片内三种子图，结合 masked graph 对比学习和跨节点双注意力机制，实现从 H&E 病理图像预测空间基因表达，在七个数据集上 PCC 指标提升 7.3%-27.1%。

**[cryoSENSE: Compressive Sensing Enables High-throughput Microscopy with Sparse and Generative Priors on the Protein Cryo-EM Image Manifold](cryosense_compressive_sensing_enables_high-throughput_microscopy_with_sparse_and.md)**

:   提出 cryoSENSE，首个冷冻电镜压缩成像的计算框架，证明蛋白质 cryo-EM 图像在稀疏先验（DCT/小波/TV）和生成先验（扩散模型）下均可从欠采样测量中高保真重建，在保持 3D 分辨率的同时实现最高 2.5× 通量提升。

**[CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)**

:   提出 CURE——一种基于误差感知课程学习的多任务训练框架，在不引入额外数据的前提下，通过动态调节采样分布重点训练困难样本，将医学 VLM 的视觉定位精度提升 +0.37 IoU，幻觉率降低 18.6%。

**[Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)**

:   提出 Deco-Mamba，一种以解码器为中心的 Transformer-CNN-Mamba 混合架构，通过 Co-Attention Gate、视觉状态空间模块（VSSM）和可变形卷积增强解码过程，同时引入基于窗口化 KL 散度的分布感知深度监督策略，在 7 个医学图像分割基准上取得 SOTA。

**[Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mambabased_decoder_with.md)**

:   提出以解码器为核心的 Deco-Mamba 网络，用 Co-Attention Gate 双向融合编解码器特征、视觉状态空间模块（VSSM）建模长程依赖、可变形卷积恢复细节，并引入窗口化分布感知 KL 散度深度监督，在 7 个医学分割基准上以中等复杂度达到 SOTA。

**[Decoupling Vision and Language: Codebook Anchored Visual Adaptation](decoupling_vision_and_language_codebook_anchored_visual_adaptation.md)**

:   提出 CRAFT，通过离散 codebook 将视觉编码器与语言模型解耦，仅微调视觉编码器即可实现领域适配，且适配后的编码器可跨 LLM 架构无缝复用，在 10 个领域基准上平均提升 13.51%。

**[Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning](deep_learning-based_assessment_of_the_relation_between_the_third_molar_and_mandi.md)**

:   在全景X光片上比较本地学习(LL)、联邦学习(FL)和集中学习(CL)三种范式对第三磨牙与下颌管重叠关系的二分类性能，发现集中学习最优(AUC 0.831)，联邦学习作为隐私保护替代方案(AUC 0.757)显著优于本地学习(AUC均值 0.672)。

**[Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](deep_learning_based_estimation_of_blood_glucose_le.md)**

:   提出ScleraGluNet多视角深度学习框架，通过五方向巩膜血管成像结合多分支CNN+MRFO特征精炼+Transformer跨视角融合，实现93.8%代谢状态三分类精度和MAE=6.42 mg/dL的空腹血糖连续估计。

**[Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](deep_learning_based_estimation_of_blood_glucose_levels_from_multidirectional_scl.md)**

:   提出ScleraGluNet，通过5个注视方向的巩膜血管照片，用并行CNN提取方向特异性血管特征，再经MRFO特征筛选和Transformer跨视角融合，同时完成三类代谢状态分类（93.8%准确率）和空腹血糖连续估计（MAE=6.42 mg/dL, r=0.983）。

**[Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning](deep_learningbased_assessment_of_the_relation_betw.md)**

:   在按8个独立标注者划分的全景口腔X光裁剪片上，系统对比本地学习（LL）、联邦学习（FL）和集中学习（CL）三种训练范式在第三磨牙-下颌管重叠二分类任务上的表现，验证了CL > FL > LL的性能排序（AUC分别为0.831、0.757和0.672），证明FL在保护数据隐私的前提下显著优于各站点独立训练。

**[Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](developing_foundation_models_for_universal_segment.md)**

:   构建了迄今最大的全身 PET 分割数据集 PETWB-Seg11K（11,041 例 3D PET + 59,831 分割掩码），并提出 SegAnyPET 基础模型，实现基于 prompt 交互的通用 PET 器官与病灶体积分割，在跨中心、跨示踪剂的零样本场景下表现优异。

**[Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](developing_foundation_models_for_universal_segmentation_from_3d_whole-body_posit.md)**

:   构建迄今最大的全身 PET 分割数据集 PETWB-Seg11K（11,041 例 3D PET + 59,831 masks），并提出 SegAnyPET——首个面向功能性 PET 影像的 3D 可提示分割基础模型，在多中心、多示踪剂、多疾病场景下实现了强零样本泛化能力。

**[Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)**

:   本文提出 NNMF+CNN+扩散防御框架用于脑肿瘤 MRI 分类：先用 NNMF 将图像分解为紧凑可解释的低秩特征，通过 AUC/Cohen's d/p-value 统计指标筛选最强判别组件，再用轻量 CNN 分类；推理时引入前向扩散加噪 + 学习去噪器的特征空间净化模块，在 AutoAttack ($L_\infty$, $\epsilon=0.10$) 下将鲁棒准确率从 0.47% 提升至 59.53%。

**[Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusionbased_feature_denoising_and_using_nnmf_fo.md)**

:   提出 NNMF 特征提取→统计特征筛选→轻量 CNN 分类→特征空间扩散净化的四阶段流水线，在干净数据上保持 85.1% 分类精度的同时，将 AutoAttack ($L_\infty$, $\epsilon=0.10$) 下的鲁棒精度从基线 0.47% 大幅提升至 59.5%。

**[EchoAgent: Towards Reliable Echocardiography Interpretation with "Eyes", "Hands" and "Minds"](echoagent_towards_reliable_echocardiography_interpretation_with_eyeshands_and_mi.md)**

:   提出 EchoAgent，一个模拟心脏超声医师"眼-手-脑"协同工作流程的 Agent 系统，通过专业知识引擎（mind）、分层工具箱（eyes+hands）和编排推理中枢（reasoning hub）三阶段实现端到端超声心动图可靠解读，在多个基准上达到 SOTA。

**[Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models](eda_arbitrary_noise_diffusion_design_space.md)**

:   提出 EDA 框架，将 EDM 的设计空间从纯高斯噪声扩展至任意噪声模式，通过多元高斯分布和多独立维纳过程驱动的 SDE 实现灵活噪声扩散，且证明噪声复杂度的提升不引入额外采样开销；仅用 5 步采样即可在 MRI 偏置场矫正、CT 金属伪影去除和自然图像阴影去除三项任务上取得媲美或优于百步 Refusion 和专用方法的效果。

**[EI: Early Intervention for Multimodal Imaging based Disease Recognition](ei_early_intervention_for_multimodal_imaging_based_disease_recognition.md)**

:   EI 提出在单模态嵌入（UIE）**之前**就注入跨模态语义引导（[INT] token），模拟临床医生"先看一个模态形成初步判断再指导另一个模态检查"的工作流程，同时设计 MoR（多种秩 LoRA + 带旁路的松弛路由器）实现参数高效的 VFM 医学域适配，在视网膜/皮肤/膝关节三个数据集上以 <9M 可训练参数超越所有全参微调和 prompt learning 基线。

**[Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models (EDA)](elucidating_the_design_space_of_arbitrary-noise-based_diffusion_models.md)**

:   提出 EDA 框架，将 EDM 的设计空间从高斯噪声扩展到任意噪声模式，通过多元高斯分布参数化协方差矩阵实现灵活的噪声扩散，在 MRI 偏置场校正、CT 金属伪影去除和自然图像阴影去除三个任务上仅用 5 步采样即达到或超越 100 步 EDM 方法和专用方法。

**[EMAD: Evidence-Centric Grounded Multimodal Diagnosis for Alzheimer's Disease](emad_evidence-centric_grounded_multimodal_diagnosis_for_alzheimers_disease.md)**

:   提出 EMAD，一个端到端多模态视觉-语言框架，为 AD 诊断生成结构化报告，通过分层 Sentence–Evidence–Anatomy (SEA) Grounding 将每个诊断声明显式关联到临床证据和 3D 脑部解剖，并用可执行规则驱动的 GRPO 强化微调确保临床一致性。

**[EquivAnIA: A Spectral Method for Rotation-Equivariant Anisotropic Image Analysis](equivania_a_spectral_method_for_rotation-equivariant_anisotropic_image_analysis.md)**

:   提出EquivAnIA，用定向滤波器族（cake wavelets和ridge filters）在频域中做带权平均来估计图像的角度分布，替代传统angular binning方法，实现对数值旋转真正鲁棒的各向异性分析，合成图像主方向估计误差仅0.03°，CT配准误差仅0.02°。

**[EquivAnIA: A Spectral Method for Rotation-Equivariant Anisotropic Image Analysis](equivania_a_spectral_method_for_rotationequivarian.md)**

:   提出EquivAnIA频谱方法，通过Cake小波和Ridge滤波器在傅里叶域计算角度能量分布，实现对数值旋转严格鲁棒的各向异性图像分析，在合成和真实图像上均远优于传统angular PSD的分箱方法。

**[Every Error has Its Magnitude: Asymmetric Mistake Severity Training for Multiclass Multiple Instance Learning](every_error_has_its_magnitude_asymmetric_mistake_severity_training_for_multiclas.md)**

:   提出 PAMS（Priority-Aware Mistake Severity）方法，通过非对称严重性感知的交叉熵损失（MSCE）、语义特征混合（SFR）和非对称 Mikel's Wheel 指标，在多分类 MIL WSI 诊断中显著降低严重误诊风险。

**[Extending ZACH-ViT to Robust Medical Imaging: Corruption and Adversarial Stress Testing in Low-Data Regimes](extending_zach-vit_to_robust_medical_imaging_corruption_and_adversarial_stress_t.md)**

:   在低数据医学影像场景下，对置换不变的紧凑型 ViT 架构 ZACH-ViT 进行首次鲁棒性扩展评估。在 7 个 MedMNIST 数据集上，ZACH-ViT 在干净数据和常见损坏下均排名第一（Mean Rank 1.57），在 FGSM 下排名最佳（2.00），PGD 下排名第二（2.29）。

**[Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning](fair_lung_disease_diagnosis_from_chest_ct_via_gend.md)**

:   在 ConvNeXt-Base 骨干上构建注意力 MIL 模型，用 GRL 对抗性消除扫描表示中的性别信息，配合 focal loss（$\gamma=2$）+ 标签平滑（$\varepsilon=0.1$）、子群过采样和 5-fold 集成，在 889 例胸部 CT 四类诊断中实现均值竞赛分数 0.685±0.030，女性 macro-F1（0.691）略高于男性（0.679），验证了 GRL 能有效闭合公平性差距。

**[Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning](fair_lung_disease_diagnosis_from_chest_ct_via_gender-adversarial_attention_multi.md)**

:   提出基于注意力 MIL 和梯度反转层（GRL）的公平性框架，从胸部 CT 体积中进行多类肺部疾病诊断，在保证诊断准确性的同时消除性别偏差。

**[Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)**

:   提出 FedMEPD 框架，用模态专属编码器处理模态间异质性、滤波器级动态部分个性化解码器平衡知识共享与个性化、多锚点跨注意力校准补偿缺失模态信息，在 BraTS 2018/2020 上全面超越现有多模态联邦学习方法。

**[Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modalityspecific_encoders_and_partially.md)**

:   提出 FedMEPD 框架，通过为每种 MRI 模态设置独立编码器（全联邦共享）+ 部分个性化的多模态融合解码器 + 多锚点跨注意力校准模块，同时解决联邦学习中模态间异质性和客户端个性化两大挑战，在 BraTS 2018/2020 上超越现有联邦方法。

**[FedVG: Gradient-Guided Aggregation for Enhanced Federated Learning](fedvg_gradient-guided_aggregation_for_enhanced_federated_learning.md)**

:   FedVG 提出利用全局验证集上的逐层梯度范数为各客户端打分，梯度越平坦（范数越小）的客户端获得越高聚合权重，从而在高度数据异质性场景下显著提升联邦学习的泛化性能。

**[Focus-to-Perceive Representation Learning: A Cognition-Inspired Hierarchical Framework for Endoscopic Video Analysis](focus-to-perceive_representation_learning_a_cognition-inspired_hierarchical_fram.md)**

:   提出 FPRL，一个受临床认知启发的层次化自监督框架，通过先"聚焦"帧内病灶关键静态语义、再"感知"帧间上下文演化来缓解运动偏差，在 11 个内窥镜数据集上取得 SOTA。

**[Forecasting Epileptic Seizures from Contactless Camera via Cross-Species Transfer Learning](forecasting_epileptic_seizures_from_contactless_ca.md)**

:   首次系统定义基于纯视频的癫痫发作预测（forecasting）任务（用 3-10 秒发作前片段预测未来 5 秒内是否发作），提出两阶段跨物种迁移学习框架——在啮齿类+人类混合视频上自监督预训练 VideoMAE，再在极少人类癫痫视频上做少样本微调——在 2/3/4-shot 设定下平均 bacc 达 72.30%、roc_auc 达 75.58%，超越所有视频理解 baseline。

**[Forecasting Epileptic Seizures from Contactless Camera via Cross-Species Transfer Learning](forecasting_epileptic_seizures_from_contactless_camera_via_cross-species_transfe.md)**

:   首次提出纯视频的癫痫发作预测任务，利用大规模啮齿动物癫痫视频进行跨物种自监督预训练，通过 VideoMAE 框架实现 3-10 秒预测窗口内 >70% 的发作预测准确率。

**[GaussianPile: A Unified Sparse Gaussian Splatting Framework for Slice-based Volumetric Reconstruction](gaussianpile_a_unified_sparse_gaussian_splatting_framework_for_slice-based_volum.md)**

:   提出 GaussianPile，通过引入焦点感知的物理成像模型（Focus Gaussian），将 3D 高斯溅射从表面外观建模扩展到切片体数据重建，在超声和光片显微镜数据上实现了比 NeRF 方法快 11 倍、比体素网格储存缩小 16 倍的高质量体数据压缩与重建。

**[GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](giim_graph-based_learning_of_inter-_and_intra-view_dependencies_for_multi-view_m.md)**

:   提出 GIIM 框架，基于多异构图（MHG）同时建模多视图医学影像中病变间的视图内（intra-view）和视图间（inter-view）依赖关系，并通过四种缺失视图表示策略实现对不完整数据的鲁棒诊断。

**[GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](giim_graphbased_learning_of_inter_and_intraview_de.md)**

:   提出 GIIM 框架，基于多异构图（MHG）通过四类边关系同时建模同一病灶跨期相动态变化和不同病灶间空间关联，并设计四种缺失视图填充策略，在肝脏 CT、乳腺 X 光和乳腺 MRI 三种模态上均显著优于现有方法。

**[GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification](gleam_a_multimodal_imaging_dataset_and_hamm_for_gl.md)**

:   提出首个公开三模态青光眼数据集 GLEAM（SLO 眼底图 + 环乳头 OCT + 视野偏差图，1200例，四阶段标注），以及基于 CNN 的层级注意力掩码建模框架 HAMM，通过临床启发式的多头模态门控和关系图注意力实现跨模态融合，四分类准确率达 81.08%。

**[GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification](gleam_a_multimodal_imaging_dataset_and_hamm_for_glaucoma_classification.md)**

:   提出首个公开的三模态青光眼数据集 GLEAM（SLO 眼底图像 + 环视盘 OCT + 视野偏差图）并设计层级注意力掩码建模框架 HAMM，通过层级注意力编码器与轻量解码器将跨模态表征学习聚焦于编码器端，实现四阶段青光眼精确分类。

**[Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)**

:   提出 GenEval，通过域共形界（DCB）量化因果覆盖差距，并将人类专家知识量化精炼后与医学 VLM（MedGemma-4B）融合，以 LoRA 微调实现单源域泛化，在 DR 分级和癫痫灶检测上显著超越基线。

**[Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](human_knowledge_integrated_multimodal_learning_for.md)**

:   提出域保形界（DCB）理论框架量化域间因果差异并定义出可优化的一致度指标SDCD，据此精炼专家知识经LoRA注入MedGemma-4B，在8个DR和2个SOZ数据集上大幅超越单源域泛化SOTA。

**[Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset](instruction-guided_lesion_segmentation_for_chest_x-rays_with_automatically_gener.md)**

:   提出指令引导的胸部X光病变分割任务（ILS），构建了首个大规模自动生成的指令-回答数据集MIMIC-ILS（1.1M样本、192K图像、91K mask），并训练ROSALIA模型实现gIoU 71.2%和空目标准确率91.8%，远超现有通用和医学分割模型。

**[Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)**

:   发现并解决了 CLIP 在跨域少样本学习（CDFSL）中的局部特征对齐退化问题，提出基于循环一致性的 CC-CDFSL 框架，通过 T-I-T 和 I-T-I 双向循环路径和语义锚点机制改善 patch 级视觉-语言对齐，同时增强模型的可解释性。

**[InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)**

:   提出 InvAD，将扩散模型异常检测从"RGB 空间去噪重建"范式转变为"潜空间加噪反演"范式，通过 DDIM 反演直接推断最终潜变量并在先验分布下度量偏差来检测异常，仅需 3 步反演即达 SOTA 性能且推理速度提升约 2 倍。

**[InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](invad_inversionbased_reconstructionfree_anomaly_de.md)**

:   提出"检测即加噪"范式取代传统"检测即去噪"——通过DDIM反转将图像映射到潜在噪声空间，仅用3步推理判断偏离先验分布的程度作为异常分数，无需重建，实现SOTA精度的同时推理速度达88 FPS（比OmiAD快2倍+）。

**[Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)**

:   提出 MASS（MAsk-guided Self-Supervised learning），利用 SAM2 自动生成的类别无关 mask 作为伪标注，以 in-context 分割为 pretext task 进行自监督预训练，无需任何人工标注即可学到语义丰富、泛化性强的 3D 医学图像表征，在 few-shot 分割和冻结编码器分类上均取得优异表现。

**[LUMINA: A Multi-Vendor Mammography Benchmark with Energy Harmonization Protocol](lumina_a_multi-vendor_mammography_benchmark_with_energy_harmonization_protocol.md)**

:   提出 LUMINA 多厂商乳腺 FFDM 数据集（468 例患者、1824 张图像），附带前景像素直方图匹配的能量协调预处理方法，在诊断/BI-RADS/密度三任务上系统评估了 CNN 与 Transformer 模型。

**[Marker-Based 3D Reconstruction of Aggregates with a Comparative Analysis of 2D and 3D Morphologies](marker-based_3d_reconstruction_of_aggregates_with_a_comparative_analysis_of_2d_a.md)**

:   提出基于标记物（marker）的低成本摄影测量方法，实现骨料颗粒的高质量 3D 重建，并通过 2D 与 3D 形态学指标的系统对比分析，揭示 2D 投影分析对真实 3D 形态的显著偏差。

**[Marker-Based 3D Reconstruction of Aggregates with a Comparative Analysis of 2D and 3D Morphologies](markerbased_3d_reconstruction_of_aggregates_with_a.md)**

:   提出一种基于标记物（marker）的低成本摄影测量方法，实现骨料（aggregate）颗粒的高质量三维重建，并通过 2D 与 3D 形态学指标的系统对比分析，揭示了仅依赖 2D 图像进行骨料形态评估的显著局限性。

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

**[MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](mil-pf_multiple_instance_learning_on_precomputed_features_for_mammography_classi.md)**

:   提出 MIL-PF，利用冻结的基础视觉编码器（DINOv2/MedSigLIP）预计算特征，再用仅约 40K 参数的轻量 MIL 头进行乳腺 X 线分类，在大规模 EMBED 数据集上达到 SOTA 性能，同时大幅降低训练成本。

**[MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](milpf_multiple_instance_learning_on_precomputed_fe.md)**

:   将冻结的通用基础编码器（DINOv2 ViT-Giant / MedSigLIP）与仅 ~40k 参数的轻量 MIL 聚合头结合，通过预计算特征 + 双流聚合（全局均值 + 局部 Perceiver 交叉注意力），在 EMBED 等大规模乳腺 X 线分类基准上以 5-7 分钟训练达到 SOTA（AUC 0.916, Spec@Sens=0.9 达 0.762），可训练参数比基线少 35-458 倍。

**[Mind the Discriminability Trap in Source-Free Cross-domain Few-shot Learning](mind_the_discriminability_trap_in_source-free_cross-domain_few-shot_learning.md)**

:   揭示了在 VLM 的跨域小样本微调中，增强视觉判别性反而损害跨模态对齐（"判别性陷阱"），提出 SVL + RA 两个即插即用模块来抑制视觉学习捷径并引导跨模态对齐，在 4 个 CDFSL 数据集和 11 个 FSL 数据集上取得 SOTA。

**[Mitigating Object Hallucination in LVLMs via Attention Imbalance Rectification](mitigating_object_hallucinations_in_lvlms_via_attention_imbalance_rectification.md)**

:   提出注意力失衡（Attention Imbalance）概念来解释 LVLM 中的对象幻觉现象，并设计轻量级解码时干预方法 AIR，通过跨模态注意力重新分配和方差约束投影正则化矫正注意力失衡，在四个 LVLM 上将幻觉率最高降低 35.1%，同时提升通用能力最高达 15.9%。

**[Modeling Spatiotemporal Neural Frames for High Resolution Brain Dynamics](modeling_spatiotemporal_neural_frames_for_high_resolution_brain_dynamic.md)**

:   提出基于扩散 Transformer 的 EEG 条件 fMRI 重建框架，将脑活动建模为时空神经帧序列而非独立快照，在皮层顶点级分辨率下实现时空一致的 fMRI 重建，并通过零空间采样支持中间帧插值，下游视觉解码任务验证了功能信息的保留。

**[MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection](moeclip_patch-specialized_experts_for_zero-shot_anomaly_detection.md)**

:   提出 MoECLIP，将 Mixture-of-Experts 引入零样本异常检测（ZSAD），通过冻结正交特征分离（FOFS）和等角紧框架（ETF）损失实现 patch 级别的动态专家路由与特化，在14个工业/医学基准上达到 SOTA。

**[Momentum Memory for Knowledge Distillation in Computational Pathology](momentum_memory_for_knowledge_distillation_in_computational_pathology.md)**

:   提出 MoMKD，用动量更新的类条件记忆库替代传统 batch-local 特征对齐，实现基因组→病理切片的跨模态知识蒸馏，仅用 H&E 切片推理即可获得基因组级预测能力。

**[MozzaVID: Mozzarella Volumetric Image Dataset](mozzavid_mozzarella_volumetric_image_dataset.md)**

:   本文发布 MozzaVID——一个基于同步辐射 X 射线 CT 的马苏里拉奶酪微结构体积图像分类数据集，包含 591-37,824 个 192³ 体积样本、25 种奶酪/149 个样本的分类目标，弥补了 3D 体积数据集在数量级和任务设计上与 2D 数据集的巨大差距，实验表明 3D 模型显著优于 2D 模型。

**[MRI Contrast Enhancement Kinetics World Model](mri_contrast_enhancement_kinetics_world_model.md)**

:   首次提出 MRI 造影增强动力学世界模型（MRI CEKWorld），通过时空一致性学习（STCL）在稀疏采样数据上实现从无造影 MRI 到连续高保真造影增强序列的生成，解决了内容失真和时序不连续两大难题。

**[Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)**

:   提出 RICE-NET，一个多模态 3D ResNet-18 模型，整合纵向 MRI 数据与放疗剂量分布图，用于自动区分胶质母细胞瘤术后放射诱导对比增强（RICE）与肿瘤复发，在独立测试集上达到 F1=0.92。

**[Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiationinduced_cont.md)**

:   提出RICE-NET，融合纵向T1加权MRI和放射治疗剂量分布图的多模态3D ResNet-18，在92例胶质母细胞瘤队列上实现F1=0.916的放射性对比增强(RICE) vs 肿瘤复发分类，消融实验揭示放疗剂量图是最关键的单模态输入(F1=0.78)。

**[Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation](multimodal_protein_language_models_for_enzyme_kine.md)**

:   提出ERBA（Enzyme-Reaction Bridging Adapter），将酶动力学参数预测重新建模为与催化机制对齐的分阶段条件化问题——先通过MRCA注入底物信息捕捉分子识别，再通过G-MoE融合活性位点3D几何信息建模构象适应，并用ESDA做分布对齐保持PLM先验——在三个动力学指标上全面超越现有SOTA。

**[Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation](multimodal_protein_language_models_for_enzyme_kinetic_parameters_from_substrate_.md)**

:   提出**ERBA(Enzyme-Reaction Bridging Adapter)**，将酶动力学参数预测重新建模为**分阶段多模态条件生成问题**——先通过MRCA注入底物信息捕获底物识别特异性，再通过G-MoE整合活性位点3D结构捕获构象适应，配合ESDA分布对齐保持PLM语义先验。

**[MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning](multimodalpfn_extending_prior-data_fitted_networks_for_multimodal_tabular_learni.md)**

:   提出 MMPFN，首次将预训练表格基础模型 TabPFN 扩展到多模态（表格+图像/文本）场景，通过多头门控 MLP（MGM）和交叉注意力池化器（CAP）解决非表格嵌入过压缩和 token 数量不平衡问题，在医学和通用数据集上超越 SOTA。

**[Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)**

:   提出 MSG-LDM，在潜在扩散模型中引入多尺度结构-风格解耦机制，通过高频注入、多模态结构特征融合和结构感知损失，实现缺失模态场景下保留解剖结构和精细细节的多模态 MRI 合成。

**[Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](multiscale_structureguided_latent_diffusion_for_mu.md)**

:   提出MSG-LDM，一个基于潜在扩散模型的多模态MRI翻译框架，通过在潜空间中显式解耦风格和结构信息，结合高频注入（HFIB）、多模态结构特征融合（MMSF）和多尺度结构增强（MSSE）模块提取模态无关的完整结构先验来引导扩散去噪，在BraTS2020和WMH数据集上超越现有方法。

**[MUSE: Harnessing Precise and Diverse Semantics for Few-Shot Whole Slide Image Classification](muse_harnessing_precise_and_diverse_semantics_for_few-shot_whole_slide_image_cla.md)**

:   提出 MUSE 框架，通过 MoE 驱动的样本级细粒度语义增强（SFSE）和基于 LLM 知识库的随机多视角语义优化（SMMO），在少样本全切片图像分类任务上显著提升泛化能力。

**[MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)**

:   提出 MUST 框架，通过代数约束将多模态表征显式分解为模态特有和跨模态共享两部分，并用条件潜在扩散模型在模态缺失时生成特有信息，在五个 TCGA 癌症数据集上以 0.742 C-index 达到 SOTA，且在模态缺失场景下仅降约 0.4%-3.5%。

**[MuViT: Multi-Resolution Vision Transformers for Learning Across Scales in Microscopy](muvit_multi-resolution_vision_transformers_for_learning_across_scales_in_microsc.md)**

:   提出 MuViT，一种基于世界坐标 RoPE 位置编码的多分辨率 Vision Transformer，能在单一编码器中联合处理同一场景不同物理分辨率的裁剪图，在显微镜图像分割任务上显著优于单分辨率基线。

**[NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization](neuroseg_meets_dinov3_transferring_2d_self-supervised_visual_priors_to_3d_neuron.md)**

:   NeurINO 提出通过将 DINOv3 预训练的 2D 卷积核膨胀（inflate）为 3D 算子来初始化 3D 神经元分割模型，同时引入拓扑感知骨架损失（TASL）显式监督骨架级结构保真性，在四个神经影像数据集上 ESA 平均提升 2.9%、DSA 提升 2.8%、PDS 提升 3.8%。

**[Novel Architecture of RPA in Oral Cancer Lesion Detection](novel_architecture_of_rpa_in_oral_cancer_lesion_de.md)**

:   本文对比了低代码 RPA 平台（UiPath、Automation Anywhere）与基于 Python 设计模式（Singleton + Batch Processing）的口腔癌检测自动化方案，后者 (OC-RPAv2) 将单图推理时间从 2.5 秒压缩到 0.06 秒，实现 60-100 倍加速。

**[Novel Architecture of RPA In Oral Cancer Lesion Detection](novel_architecture_of_rpa_in_oral_cancer_lesion_detection.md)**

:   将软件设计模式（Singleton + Batch Processing）集成到基于 EfficientNetV2B1 的口腔癌病变检测 Python 流水线中，相比传统 RPA 平台（UiPath/Automation Anywhere）实现 60-100x 推理加速（每张图 0.06s vs 2.58s），同时保持诊断准确性。

**[OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)**

:   提出 OmniFM，一个模态鲁棒且任务无关的联邦学习框架，通过频域频谱知识检索、嵌入式交叉注意力融合和前缀-后缀频谱提示三个互补组件，在一个统一的 FL pipeline 下支持分类、分割、超分辨率、VQA 和多模态融合五种医学影像任务，并在跨模态异构场景下显著超越现有基线。

**[OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation](orapo_oracle-educated_reinforcement_learning_for_data-efficient_and_factual_radi.md)**

:   提出 OraPO（Oracle-educated GRPO），在 GRPO 探索失败时注入轻量 DPO 监督将失败 rollout 转化为偏好样本，配合 FactScore 奖励实现仅用 1K 样本、3B 小模型在 CheXpert Plus 和 MIMIC-CXR 上达到放射报告生成 SOTA（F1=0.341/0.357），训练数据量比前最优减少 2-3 个数量级。

**[OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation](orapo_oracle_rl_radiology_report_generation.md)**

:   提出 OraPO, 一种结合 GRPO 和 DPO 的自适应混合 RL 框架, 用于数据高效的放射学报告生成: 通过 Zero-Reward Rate 检测动态切换 GRPO 和 DPO, 加上 FactScore-based 临床事实级奖励, 仅用 1K 样本 (对比基线 227K) 在 CheXpert Plus 和 MIMIC-CXR 上取得 SOTA 的临床 F1 (0.341/0.357).

**[Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification](parameter-efficient_prompt_tuning_and_hierarchical_textual_guidance_for_few-shot.md)**

:   HIPSS 提出了两个关键创新用于少样本 WSI 分类：(1) 基于缩放和偏移特征（SSF）的参数高效 prompt 调优替代 CoOp，大幅减少可训练参数；(2) 软层次化文本引导策略无需硬过滤即可利用 VLM 的预训练知识和 WSI 的固有层次结构。在三个癌症数据集上最高提升 13.8%。

**[PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation](pgr-net_prior-guided_roi_reasoning_network_for_brain_tumor_mri_segmentation.md)**

:   PGR-Net 提出了一种显式 ROI 感知的脑肿瘤 MRI 分割网络，通过从训练集构建数据驱动的空间先验模板、层级 Top-K ROI 选择机制和窗口高斯-空间衰减引导模块（WinGS-ROI），将计算资源集中于病灶区域，仅用 8.64M 参数就在 BraTS-2019/2023 和 MSD Task01 上达到了 SOTA。

**[Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](prototype-based_knowledge_guidance_for_fine-grained_structured_radiology_reporti.md)**

:   提出 ProtoSR，通过 LLM 从大规模自由文本放射学报告中挖掘模板对齐的视觉原型知识库，并以原型条件化残差（late fusion）方式注入结构化报告生成模型，在 Rad-ReStruct 基准上取得 SOTA，尤其显著提升细粒度属性问题的性能。

**[Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](prototypebased_knowledge_guidance_for_finegrained.md)**

:   提出 ProtoSR，通过 LLM 驱动的管道从 22.7 万篇 MIMIC-CXR 自由文本报告中挖掘模板对齐的视觉原型知识库，并设计原型条件化迟融合模块将检索到的原型证据作为 logit 残差注入层级式结构化报告模型，在 Rad-ReStruct 基准上达到 SOTA，L3 细粒度属性 F1 从 4.3 提升到 7.4（+72.1% 相对提升）。

**[RDFace: A Benchmark Dataset for Rare Disease Facial Image Analysis under Extreme Data Scarcity and Phenotype-Aware Synthetic Generation](rdface_a_benchmark_dataset_for_rare_disease_facial_image_analysis_under_extreme_.md)**

:   构建了包含 456 张儿童面部图像、覆盖 103 种罕见遗传疾病的标准化基准数据集 RDFace，并系统研究了表型感知的合成数据增强（DreamBooth/FastGAN）在超低样本罕见病诊断中的效果，DreamBooth 增强在极端低数据场景下最高可提升 13.7% 的诊断准确率。

**[ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval](recall_recalibrating_capability_degradation_for_mllm-based_composed_image_retrie.md)**

:   揭示了将生成式MLLM适配为判别式检索器时的"能力退化"现象（Capability Degradation），提出ReCALL框架通过诊断检索器盲点→利用基座MLLM的CoT推理生成纠正性三元组→分组对比精炼三阶段管线，有效恢复退化的细粒度组合推理能力，在CIRR上R@1达55.52%、FashionIQ上R@10达57.04%。

**[Reclaiming Lost Text Layers for Source-Free Cross-Domain Few-Shot Learning](reclaiming_lost_text_layers_for_source-free_cross-domain_few-shot_learning.md)**

:   发现 CLIP 文本编码器中存在"Lost Layers"——在 Source-Free Cross-Domain Few-Shot Learning (SF-CDFSL) 中移除某些中间层反而提升性能；论文证明这些层并非冗余而是因视觉域偏移未被充分利用，提出 VtT 模型在层级和编码器级别重新利用这些信息，取得 SOTA。

**[Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](reinforcing_the_weakest_links_modernizing_siena_wi.md)**

:   将 SIENA 纵向脑萎缩管线中的经典颅骨剥离(BET2)和组织分割(FAST)模块定向替换为深度学习方案(SynthStrip/SynthSeg)，在 ADNI (N=1006) 和 PPMI (N=310) 两个大规模纵向队列上显著增强了 PBVC 与临床疾病进展的关联性(相关系数提升超 100%)，扫描顺序误差降低高达 99.1%。

**[Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](reinforcing_the_weakest_links_modernizing_siena_with_targeted_deep_learning_inte.md)**

:   通过将 SIENA 脑萎缩管线中经典的颅骨剥离（BET2）和组织分割（FAST）模块替换为深度学习方案（SynthStrip、SynthSeg），在保留管线可解释性的前提下显著提升了 PBVC 估计的临床敏感度和鲁棒性。

**[Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)**

:   提出 Residual SODAP 框架，通过 α-entmax 稀疏提示选择+残差聚合、无数据统计蒸馏+伪特征回放、提示使用模式漂移检测，以及不确定性加权多损失平衡，联合解决提示端表征适应和分类器端知识保持问题，在医学域增量学习上达到 SOTA。

**[Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](residual_sodap_residual_selforganizing_domainadapt.md)**

:   提出Residual SODAP框架，在无任务ID、无数据存储的域增量学习中，联合解决表示适应（α-entmax稀疏prompt选择+残差聚合）和分类器保持（统计伪特征重放+知识蒸馏），在DR、皮肤癌和CORe50三个基准上达到SOTA。

**[Robust Multi-Source Covid-19 Detection in CT Images](robust_multi-source_covid-19_detection_in_ct_images.md)**

:   提出一种多任务学习框架，在共享 EfficientNet-B7 骨干上同时训练 COVID-19 诊断头和来源医院识别头（使用 logit-adjusted 损失），推动特征提取器学习跨机构不变的表示，在多源 CT 数据集上 F1 达到 0.9098。

**[Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)**

:   本文提出 SCDL（Semantic Class Distribution Learning），一个即插即用模块，通过类别分布双向对齐（CDBA）学习结构化的类条件特征分布并与可学习类代理双向对齐，结合语义锚点约束（SAC）利用标注数据引导代理学习正确语义，缓解了半监督医学图像分割中的监督偏差和特征表示偏差，在尾类器官上取得了显著提升。

**[SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)**

:   提出即插即用的语义类分布学习框架 SCDL，通过类分布双向对齐（CDBA）学习结构化类条件特征分布 + 语义锚约束（SAC）引导代理分布对齐真实语义，解决半监督医学分割中的监督偏差和表示不平衡，在少数类分割上取得 SOTA。

**[SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)**

:   提出 SemiTooth 框架，通过多教师多学生架构和严格加权置信度约束（SWC），解决多源 CBCT 牙齿分割中的标注稀缺和跨源域间差异问题，同时构建了首个多源半监督牙齿数据集 MS3Toothset。

**[Similarity-as-Evidence: Calibrating Overconfident VLMs for Interpretable and Label-Efficient Medical Active Learning](similarity-as-evidence_calibrating_overconfident_vlms_for_interpretable_and_labe.md)**

:   提出 Similarity-as-Evidence (SaE) 框架，将 VLM 的文本-图像相似度重新解释为 Dirichlet 证据，通过 Similarity Evidence Head (SEH) 校准过度自信的 softmax 输出，并基于 vacuity（知识空缺）和 dissonance（证据冲突）的双因子采集策略实现可解释、高效的医学主动学习，在 10 个数据集上以 20% 标注预算达到 82.57% 的 SOTA 宏平均准确率。

**[Solving a Nonlinear Blind Inverse Problem for Tagged MRI with Physics and Deep Generative Priors](solving_a_nonlinear_blind_inverse_problem_for_tagged_mri_with_physics_and_deep_g.md)**

:   提出 InvTag 框架，首次将 MR 物理前向模型与预训练扩散生成先验结合，统一解决 3D Tagged MRI 的解剖恢复、Cine 合成和运动估计三大子任务，且无需任何额外训练数据。

**[Sparse Task Vector Mixup with Hypernetworks for Efficient Knowledge Transfer in Whole-Slide Image Prognosis](sparse_task_vector_mixup_with_hypernetworks_for_efficient_knowledge_transfer_in_.md)**

:   提出 STEPH，通过任务向量混合 (Task Vector Mixup) 与超网络驱动的稀疏聚合，将多个癌种预后模型的可泛化知识高效迁移到目标癌种，在 13 个 TCGA 数据集上平均 C-Index 提升 5.14%，且无需大规模联合训练或多模型推理。

**[STEPH: Sparse Task Vector Mixup with Hypernetworks for Efficient Knowledge Transfer in WSI Prognosis](sparse_task_vector_mixup_wsi_prognosis.md)**

:   STEPH 提出基于任务向量混合（TVM）+ 超网络驱动稀疏聚合的模型合并方案，将多个癌种特定预后模型的知识高效融入目标癌种模型，在 13 个 TCGA 数据集上 C-Index 平均 0.6949（+5.14% vs 癌种特定学习、+2.01% vs ROUPKT），且推理仅需单模型前向传播，远低于多模型表示迁移方案。

**[SPEGC: Continual Test-Time Adaptation via Semantic-Prompt-Enhanced Graph Clustering for Medical Image Segmentation](spegc_continual_test-time_adaptation_via_semantic-prompt-enhanced_graph_clusteri.md)**

:   提出 SPEGC 框架，通过语义提示增强特征 + 可微分图聚类求解器，将原始相似度矩阵精炼为高阶结构表示，用于指导医学图像分割模型在持续变化的目标域上自适应，有效缓解误差累积与灾难性遗忘。

**[SVC 2026: The Second Multimodal Deception Detection Challenge and the First Domain Generalized Remote Physiological Measurement Challenge](svc_2026_the_second_multimodal_deception_detection_challenge_and_the_first_domai.md)**

:   组织SVC 2026挑战赛，包含跨域多模态欺骗检测和域泛化远程生理信号测量两个赛道，提供统一评估框架和基线模型，共22支队伍提交最终结果。

**[Synergistic Bleeding Region and Point Detection in Laparoscopic Surgical Videos](synergistic_bleeding_region_and_point_detection_in_laparoscopic_surgical_videos.md)**

:   构建首个腹腔镜手术出血区域+出血点标注数据集 SurgBlood，并提出基于 SAM2 的双分支双向引导在线检测器 BlooDet，通过 Mask/Point 分支协同优化实现出血区域分割与出血点定位的联合检测。

**[T-Gated Adapter: A Lightweight Temporal Adapter for Vision-Language Medical Segmentation](t-gated_adapter_a_lightweight_temporal_adapter_for_vision-language_medical_segme.md)**

:   提出轻量级时序门控适配器（T-Gated Adapter），为2D视觉语言模型CLIPSeg注入相邻切片上下文，在仅30个标注CT体积上训练即可实现平均Dice 0.704（+0.206），跨域零样本评估和CT到MRI跨模态评估中均一致提升。

**[Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)**

:   提出 Tell2Adapt 统一框架，利用视觉基础模型（BiomedParse）的泛化知识，通过上下文感知提示正则化（CAPR）生成高质量伪标签，再经视觉合理性精炼（VPR）去除解剖学不合理预测，实现跨 10 个域迁移方向、22 个解剖目标的统一无源域自适应医学图像分割。

**[The Invisible Gorilla Effect in Out-of-distribution Detection](the_invisible_gorilla_effect_in_out-of-distribution_detection.md)**

:   揭示了OOD检测中一个此前未被报告的偏差——"隐形大猩猩效应"：当OOD伪影与模型关注区域（ROI）视觉外观相似时检测性能显著更好，不相似时则大幅下降，尤其影响基于特征的OOD方法。

**[Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data](towards_efficient_medical_reasoning_with_minimal_fine-tuning_data.md)**

:   提出 Difficulty-Influence Quadrant (DIQ) 数据选择策略，联合考量样本难度和梯度影响力，使 VLM 语言骨干仅用 1% 精选数据即可匹配全量 SFT 性能，10% 数据则可超越全量训练。

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

**[CodeBrain: Virtual Full-stack Scanning of Brain MRI via Imputing Any Quantised Code](virtual_fullstack_scanning_of_brain_mri_via_imputi.md)**

:   CodeBrain将脑MRI多模态补全(any-to-any imputation)重新定义为区域级全栈量化码预测问题：Stage I用有限标量量化(FSQ)将完整MRI集编码为紧凑code map + 模态无关公共特征，Stage II从不完整模态预测code map(用grading loss保持量化空间平滑性)，在IXI和BraTS 2023上超越5种SOTA方法，生成的模态可接近真实数据的脑肿瘤分割性能。

**[VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)**

:   重新审视零样本异常检测（ZSAD）中文本分支的必要性，提出 VisualAD——一个纯视觉框架：在冻结 ViT 中插入两个可学习 token（anomaly/normal），配合 Spatial-Aware Cross-Attention 和 Self-Alignment Function，去掉文本编码器仍在 13 个工业+医学基准上取得 SOTA。

**[Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation](weakly_supervised_teacher-student_framework_with_progressive_pseudo-mask_refinem.md)**

:   提出弱监督教师-学生框架，利用稀疏病理标注和 EMA 稳定的教师网络生成渐进式精炼伪掩码，结合置信度过滤、自适应融合和课程引导精炼策略，实现结直肠癌病理图像中腺体结构的高效分割。

**[X-WIN: Building Chest Radiograph World Model via Predictive Sensing](x-win_building_chest_radiograph_world_model_via_predictive_sensing.md)**

:   提出 X-WIN 胸片世界模型，首次将 3D CT 空间知识融入 CXR 表征学习：通过学习预测 CT 在不同旋转角度下的 2D 投影来内化 3D 解剖结构，配合亲和力引导的对比对齐和结构保持域自适应，在 6 个 CXR 基准上通过线性探测取得 SOTA。

**[XSeg: A Large-scale X-ray Contraband Segmentation Benchmark for Real-World Security Screening](xseg_a_large-scale_x-ray_contraband_segmentation_benchmark_for_real-world_securi.md)**

:   本文构建了目前最大的 X 光违禁品分割数据集 XSeg（98,644 张图像、295,932 个实例 mask、30 个细粒度类别），并提出域特化模型 APSAM，通过 Energy-Aware Encoder 利用 X 光双能量物理特性 + Adaptive Point Generator 智能扩展用户点击提示，mIoU 达 72.83%，比 SAM 微调高 4.96%。
