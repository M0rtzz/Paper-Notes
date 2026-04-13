---
title: >-
  ICCV2025 人体理解方向 69篇论文解读
description: >-
  69篇ICCV2025 人体理解方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧑 人体理解

**📹 ICCV2025** · 共 **69** 篇

**[A Quality-Guided Mixture Of Score-Fusion Experts Framework For Human Recognition](a_quality-guided_mixture_of_score-fusion_experts_framework_for_human_recognition.md)**

:   本文提出 QME（Quality-guided Mixture of score-fusion Experts）框架，通过可学习的分数融合策略和基于模态质量的 MoE 路由机制，动态整合人脸、步态和行人重识别等多模态生物特征的相似度分数，在多个全身识别基准上实现了 SOTA 性能。

**[A Qualityguided Mixture Of Scorefusion Experts Framework For](a_qualityguided_mixture_of_scorefusion_experts_framework_for.md)**

:   提出 Quality-guided Mixture of score-fusion Experts (QME) 框架，通过质量引导的 MoE 策略对来自不同生物特征模态（人脸、步态、身体）的相似度分数进行可学习融合，配合伪质量损失和分数三元组损失，在多个全身生物特征识别基准上达到 SOTA。

**[Align Your Rhythm Generating Highly Aligned Dance Poses With Gating-Enhanced Rhy](align_your_rhythm_generating_highly_aligned_dance_poses_with_gating-enhanced_rhy.md)**

:   提出Danceba框架，通过基于相位的节奏提取（PRE）、时序门控因果注意力（TGCA）和并行Mamba运动建模（PMMM）三个核心模块，实现音乐驱动的高节奏对齐、高多样性舞蹈生成，在AIST++数据集上FIDk提升48.68%、BAS提升12%。

**[Ar-Vrm Imitating Human Motions For Visual Robot Manipulation With Analogical Rea](ar-vrm_imitating_human_motions_for_visual_robot_manipulation_with_analogical_rea.md)**

:   提出 AR-VRM，首个通过显式模仿人类手部关键点来增强视觉机器人操控的方法，采用关键点视觉语言模型预训练从大规模人类动作视频中学习动作知识，并通过类比推理(Analogical Reasoning)建立人手关键点与机器人组件的映射。

**[Avat3R Large Animatable Gaussian Reconstruction Model For Hi](avat3r_large_animatable_gaussian_reconstruction_model_for_hi.md)**

:   提出Avat3r——首个可动画的大型3D重建模型(LRM)，仅需4张输入图像即可在前馈方式下回归出高质量可驱动的3D高斯头部头像，通过整合DUSt3R位置图和Sapiens语义特征作为先验、并用简单的cross-attention建模表情动画，在Ava256和NeRSemble数据集上大幅超越现有方法。

**[Bi-Level Optimization For Self-Supervised Ai-Generated Face Detection](bi-level_optimization_for_self-supervised_ai-generated_face_detection.md)**

:   提出BLADES方法，通过双层优化（bi-level optimization）将自监督预训练与AI生成人脸检测目标显式对齐：内层优化视觉编码器学习EXIF分类/排序和人脸篡改检测等前置任务，外层优化各任务权重以提升代理检测任务性能，实现不依赖合成人脸的跨生成器泛化检测。

**[Cargait Cross-Attention Based Re-Ranking For Gait Recognition](cargait_cross-attention_based_re-ranking_for_gait_recognition.md)**

:   提出CarGait，一种基于交叉注意力的步态识别重排序方法，通过probe与候选序列之间的strip-wise交叉注意力学习细粒度的步态对应关系，将预训练单阶段模型的全局特征映射到新的判别性嵌入空间，在Gait3D、GREW和OU-MVLP三大基准上对七种步态模型均取得一致的Rank-1/5精度提升。

**[Cargait Cross Attention Based Re Ranking For Gait Recognition](cargait_cross_attention_based_re_ranking_for_gait_recognition.md)**

:   提出CarGait，一种基于交叉注意力的步态识别重排序方法，通过probe与候选序列之间的strip-wise交叉注意力学习细粒度的步态对应关系，将预训练单阶段模型的全局特征映射到新的判别性嵌入空间，在Gait3D、GREW和OU-MVLP三大基准上对七种步态模型均取得一致的Rank-1/5精度提升。

**[Cargait Crossattention Based Reranking For Gait Recognition](cargait_crossattention_based_reranking_for_gait_recognition.md)**

:   提出CarGait——基于cross-attention的步态识别重排序方法：对任意单阶段步态模型的top-K检索结果，通过probe与候选间步态条带(gait strip)的cross-attention学习细粒度pair-wise交互，生成新的条件化表征并重新计算距离进行重排序。在Gait3D/GREW/OU-MVLP三个数据集、7种基线模型上一致提升Rank-1/5准确率，推理速度6.5ms/probe远超现有重排序方法。

**[Cleanpose Category-Level Object Pose Estimation Via Causal Learning And Knowledg](cleanpose_category-level_object_pose_estimation_via_causal_learning_and_knowledg.md)**

:   首次将因果推理引入类别级物体位姿估计（COPE），通过基于前门调整的因果推理模块消除数据偏差导致的虚假关联，并利用3D基础模型ULIP-2的残差知识蒸馏提供无偏的类别语义监督，在REAL275的严格指标5°2cm上达到61.7%，超越SOTA 4.7%。

**[Combinative Matching For Geometric Shape Assembly](combinative_matching_for_geometric_shape_assembly.md)**

:   提出组合匹配（Combinative Matching）方法，同时建模互锁部件的"表面形状一致性"和"体积占用相反性"两大属性，通过等变网络学习方向对齐、形状匹配与占用匹配三个目标，大幅减少几何组装中的局部歧义。

**[Contact-Aware Refinement Of Human Pose Pseudo-Ground Truth Via Bioimpedance Sens](contact-aware_refinement_of_human_pose_pseudo-ground_truth_via_bioimpedance_sens.md)**

:   提出BioTUCH框架，通过手腕间生物阻抗传感检测自接触事件，结合视觉姿态估计器进行接触感知的3D手臂姿态优化，平均提升重建精度11.7%。

**[Controllable And Expressive One-Shot Video Head Swapping](controllable_and_expressive_one-shot_video_head_swapping.md)**

:   本文提出一个基于扩散模型的多条件可控视频头部替换框架（SwapAnyHead），通过形状无关掩码策略、发型增强策略和表情感知的3DMM驱动landmark重定向模块，实现了高保真的身份保持、无缝背景融合和精确的跨身份表情迁移与编辑。

**[Dadm Dual Alignment Of Domain And Modality For Face Anti-Spoofing](dadm_dual_alignment_of_domain_and_modality_for_face_anti-spoofing.md)**

:   提出 DADM 框架，通过互信息掩码（MIM）模块和域-模态双对齐优化策略，同时解决多模态人脸反欺骗中的域内模态不对齐和域间模态不对齐问题，在四种协议下取得 SOTA 性能。

**[Degradation-Modeled Multipath Diffusion For Tunable Metalens Photography](degradation-modeled_multipath_diffusion_for_tunable_metalens_photography.md)**

:   提出DMDiff框架，利用预训练扩散模型的自然图像先验，通过正/中/负三路径多提示扩散策略和空间变化退化感知注意力（SVDA）模块，实现毫米级超透镜相机的高保真可调图像重建，在多项指标上超越现有方法。

**[Dreamactor-M1 Holistic Expressive And Robust Human Image Animation With Hybrid G](dreamactor-m1_holistic_expressive_and_robust_human_image_animation_with_hybrid_g.md)**

:   DreamActor-M1提出基于DiT架构的人体图像动画框架，通过隐式面部表征+3D头部球体+3D身体骨架的混合控制信号实现精细面部和身体控制，结合互补外观引导和渐进式训练策略支持肖像到全身的多尺度生成。

**[Dynamic Reconstruction Of Hand-Object Interaction With Distributed Force-Aware C](dynamic_reconstruction_of_hand-object_interaction_with_distributed_force-aware_c.md)**

:   提出 ViTaM-D，一个视觉-触觉融合框架，通过新提出的分布式力感知接触表示（DF-Field）和两阶段流程（视觉动态跟踪+力感知优化），实现刚性和可变形物体的手物交互动态重建，并引入 HOT 数据集填补可变形物体手物交互的评测空白。

**[Dynfacerestore Balancing Fidelity And Quality In Diffusion-Guided Blind Face Res](dynfacerestore_balancing_fidelity_and_quality_in_diffusion-guided_blind_face_res.md)**

:   提出 DynFaceRestore，通过动态模糊等级映射（DBLM）将盲退化转化为高斯去模糊问题，结合动态起始步查找表（DSST）和区域自适应引导缩放器（DGSA），在扩散模型采样中实现保真度与感知质量的最优平衡。

**[Egoagent A Joint Predictive Agent Model In Egocentric Worlds](egoagent_a_joint_predictive_agent_model_in_egocentric_worlds.md)**

:   提出EgoAgent，一个统一的预测式智能体模型，在单个Transformer中同时学习表征第一人称视觉观测、预测未来世界状态和生成3D人体动作。

**[External Knowledge Injection For Clip-Based Class-Incremental Learning](external_knowledge_injection_for_clip-based_class-incremental_learning.md)**

:   提出 Engine（ExterNal knowledGe INjEction）框架，通过双分支注入调优（视觉分支用数据增强、文本分支用 GPT-4 生成判别性描述）和推理时后调优知识注入（成对判别特征重排序），在无需存储历史样本的条件下，在 9 个基准数据集上以 3-10% 的优势超越所有 CLIP-based 类增量学习方法。

**[Free-Merging Fourier Transform For Efficient Model Merging](free-merging_fourier_transform_for_efficient_model_merging.md)**

:   首次发现模型合并中任务干扰在频域上的表现，提出 FR-Merging 通过高通滤波去除低频干扰构建高质量合并骨干网络，并结合轻量级任务专家模块（FREE-Merging），在视觉、语言和多模态任务上实现性能-成本的最优平衡。

**[From Easy To Hard The Mir Benchmark For Progressive Interleaved Multi-Image Reas](from_easy_to_hard_the_mir_benchmark_for_progressive_interleaved_multi-image_reas.md)**

:   提出 MIR 基准，包含 22,257 个多图像交错推理问答对及五阶段推理步骤，并设计渐进式课程学习策略，从"简单到困难"逐步提升 MLLM 的多图像交错推理能力。

**[Fw-Merging Scaling Model Merging With Frank-Wolfe Optimization](fw-merging_scaling_model_merging_with_frank-wolfe_optimization.md)**

:   将模型合并形式化为约束优化问题，引入Frank-Wolfe优化启发的FW-Merging方法，通过迭代选择最相关模型并局部合并，实现在大规模黑盒模型池中的可扩展、鲁棒合并，合并20个ViT模型时超越数据感知方法Adamerging 8.39%。

**[Genm3 Generative Pretrained Multi-Path Motion Model For Text Conditional Human M](genm3_generative_pretrained_multi-path_motion_model_for_text_conditional_human_m.md)**

:   提出 GenM3 框架，通过 Multi-Expert VQ-VAE (MEVQ-VAE) 学习统一的离散运动表示，以及 Multi-path Motion Transformer (MMT) 处理模态内变异和跨模态对齐，整合 11 个运动数据集（约 220 小时），在 HumanML3D 上达到 SOTA FID 0.035。

**[Genmo A Generalist Model For Human Motion](genmo_a_generalist_model_for_human_motion.md)**

:   提出 GENMO，首个统一人体运动估计（从视频/2D 关键点恢复运动）和运动生成（从文本/音乐/关键帧合成运动）的通用模型，通过双模式训练范式（回归+扩散）在单一模型中同时实现精确估计和多样生成。

**[Gesturehydra Semantic Co-Speech Gesture Synthesis Via Hybrid Modality Diffusion ](gesturehydra_semantic_co-speech_gesture_synthesis_via_hybrid_modality_diffusion_.md)**

:   提出 GestureHYDRA，一个基于混合模态扩散 Transformer 和级联同步检索增强生成的共语手势合成系统，能够可靠地激活语义明确的手势（如数字和方向指示）。

**[Ggtalker Talking Head Systhesis With Generalizable Gaussian Priors And Identity-](ggtalker_talking_head_systhesis_with_generalizable_gaussian_priors_and_identity-.md)**

:   GGTalker 提出先验-适配两阶段训练策略，从大规模数据集学习通用的音频-表情先验和表情-视觉先验，再快速适配到特定身份，在渲染质量、3D 一致性、唇同步和训练效率上全面达到 SOTA，仅需 20 分钟适配即可生成 120 FPS 的逼真说话头视频。

**[Guiodyssey A Comprehensive Dataset For Cross-App Gui Navigation On Mobile Device](guiodyssey_a_comprehensive_dataset_for_cross-app_gui_navigation_on_mobile_device.md)**

:   提出 GUIOdyssey，首个面向移动端跨应用 GUI 导航的综合数据集（8334 episodes、212 apps、1357 app 组合），以及 OdysseyAgent——配备历史重采样模块的多模态导航智能体，在平衡性能与推理效率的同时显著提升跨应用任务表现。

**[Hcceposebf Predicting Front Back Surfaces To Construct Ultra-Dense 2D-3D Corresp](hcceposebf_predicting_front_back_surfaces_to_construct_ultra-dense_2d-3d_corresp.md)**

:   提出同时预测物体前后表面的3D坐标并在两表面间密集采样，构建超密集2D-3D对应关系，配合新颖的层级连续坐标编码（HCCE），在BOP七大核心数据集上超越现有SOTA方法。

**[Hi3Dgen High-Fidelity 3D Geometry Generation From Images Via Normal Bridging](hi3dgen_high-fidelity_3d_geometry_generation_from_images_via_normal_bridging.md)**

:   提出 Hi3DGen 框架，以法线图作为中间表示桥接 2D 图像到 3D 几何的映射，通过噪声注入回归式法线估计器（NiRNE）和法线正则化潜在扩散（NoRLD）两大核心组件，显著提升生成 3D 模型的几何细节保真度。

**[High-Resolution Spatiotemporal Modeling With Global-Local State Space Models For](high-resolution_spatiotemporal_modeling_with_global-local_state_space_models_for.md)**

:   提出 GLSMamba，首个纯 Mamba 的视频人体姿态估计框架，通过 Global Spatiotemporal Mamba（6D 选择性时空扫描 + 时空调制融合）和 Local Refinement Mamba（窗口化时空扫描）分别建模全局动态上下文和局部关键点细节，在四个基准上以线性复杂度达到 SOTA。

**[Hineus High-Fidelity Neural Surface Mitigating Low-Texture And Reflective Ambigu](hineus_high-fidelity_neural_surface_mitigating_low-texture_and_reflective_ambigu.md)**

:   提出 HiNeuS，一个统一的神经表面重建框架，通过 SDF 引导的可见性验证、平面共形正则化和渲染优先的 Eikonal 松弛三项创新，同时解决反射歧义、低纹理退化和细节保留三大核心挑战。

**[Humoto A 4D Dataset Of Mocap Human Object Interactions](humoto_a_4d_dataset_of_mocap_human_object_interactions.md)**

:   提出 HUMOTO，一个高保真 4D 人物交互数据集，包含 735 段序列（7875 秒，30fps），涵盖 63 个精确建模物体和 72 个可动部件，创新性地使用 LLM 驱动的场景脚本生成流程和多传感器捕获系统，在手部姿态精度和交互质量上显著超越现有数据集。

**[Idface Face Template Protection For Efficient And Secure Identification](idface_face_template_protection_for_efficient_and_secure_identification.md)**

:   提出 IDFace，一种基于同态加密（HE）的人脸模板保护方法，通过近等距变换（实值向量→三值向量）和空间高效编码两项技术，使 100 万加密模板的检索仅需 126ms，相比无保护仅 2× 开销。

**[Imhead A Large-Scale Implicit Morphable Model For Localized Head Modeling](imhead_a_large-scale_implicit_morphable_model_for_localized_head_modeling.md)**

:   imHead 提出首个大规模隐式 3D 头部形变模型，通过全局-局部解耦架构在 4,000 个身份的数据集上训练，实现了紧凑的隐式表示与局部面部编辑的兼顾，在重建精度和编辑灵活性上超越现有方法。

**[Intra-View And Inter-View Correlation Guided Multi-View Novel Class Discovery](intra-view_and_inter-view_correlation_guided_multi-view_novel_class_discovery.md)**

:   提出 IICMVNCD 框架，首次将新类发现（NCD）扩展到多视图设定，通过视图内矩阵分解捕捉已知/新类的分布一致性，以及视图间权重学习传递已知类的视图关系到新类，避免了对伪标签的依赖。

**[Iris Breaking Gui Complexity With Adaptive Focus And Self-Refining](iris_breaking_gui_complexity_with_adaptive_focus_and_self-refining.md)**

:   Iris 提出信息敏感裁剪（ISC）和自精炼双重学习（SRDL）两大核心创新，仅用 850K 标注数据即在多个 GUI 理解基准上达到 SOTA，性能匹敌使用 10 倍以上数据的方法，同时将处理时间从 3 秒缩短至 1 秒。

**[Kinmo Kinematic-Aware Human Motion Understanding And Generation](kinmo_kinematic-aware_human_motion_understanding_and_generation.md)**

:   提出 KinMo 框架，将人体运动分解为六大运动学组及其交互的层级可描述表示，通过自动标注管线生成细粒度文本描述，结合层级文本-运动对齐和由粗到细的运动生成策略，显著提升运动理解和细粒度运动生成能力。

**[Lay2Story Extending Diffusion Transformers For Layout-Togglable Story Generation](lay2story_extending_diffusion_transformers_for_layout-togglable_story_generation.md)**

:   Lay2Story 提出布局可切换的故事生成任务，构建了超 100 万张高分辨率图像的 Lay2Story-1M 数据集，并基于 DiT 架构设计全局-主体双分支框架，在一致性、语义相关性和美学质量上全面超越现有方法。

**[Learning Visual Proxy For Compositional Zero-Shot Learning](learning_visual_proxy_for_compositional_zero-shot_learning.md)**

:   提出 Visual Proxy（视觉代理）概念，在 CZSL 任务中首次引入文本引导的视觉类中心，并通过跨模态联合学习（CMJL）协同优化文本原型与视觉代理，在四个 CZSL 基准上达到闭世界 SOTA。

**[Lvface Progressive Cluster Optimization For Large Vision Models In Face Recognit](lvface_progressive_cluster_optimization_for_large_vision_models_in_face_recognit.md)**

:   提出 LVFace，通过渐进式聚类优化（PCO）策略解决 ViT 在大规模人脸识别中训练不稳定的问题，将训练分解为特征对齐、质心稳定和边界精炼三个阶段，在多个基准上取得 SOTA。

**[Magshield Towards Better Robustness In Sparse Inertial Motion Capture Under Magn](magshield_towards_better_robustness_in_sparse_inertial_motion_capture_under_magn.md)**

:   提出 MagShield，首个针对稀疏惯性运动捕捉系统中磁场干扰问题的方法，采用"检测-校正"两阶段策略：通过多 IMU 联合分析检测磁场扰动，再利用人体运动先验网络校正方向误差，可即插即用地增强现有稀疏 IMU 动捕系统的鲁棒性。

**[Mdd A Dataset For Text-And-Music Conditioned Duet Dance Generation](mdd_a_dataset_for_text-and-music_conditioned_duet_dance_generation.md)**

:   介绍 Multimodal DuetDance (MDD)，首个同时整合动作、音乐和文本描述的大规模专业级双人舞蹈数据集，包含 620 分钟动捕数据、15 种舞蹈类型和超过 10K 条细粒度文本标注，并提出 Text-to-Duet 和 Text-to-Dance Accompaniment 两个新任务。

**[Mixri Mixing Features Of Reference Images For Novel Object Pose Estimation](mixri_mixing_features_of_reference_images_for_novel_object_pose_estimation.md)**

:   提出 MixRI，一个仅需 12 张参考图像和 5.3M 参数的轻量级网络，通过多视角特征融合策略直接建立多参考图与查询图之间的 2D-3D 对应关系，在 BOP 挑战的 7 个核心数据集上实现了与需要数百张参考图的方法相当的位姿估计性能。

**[Monocular Facial Appearance Capture In The Wild](monocular_facial_appearance_capture_in_the_wild.md)**

:   提出一种从单目头部旋转视频重建面部外观属性（漫反射反照率、高光强度、高光粗糙度）的方法，通过提出遮挡感知的 split-sum 近似着色模型，在不对光照环境做任何简化假设的情况下实现了逼近工作室级别的面部外观捕捉质量。

**[Multi-View Gaze Target Estimation](multi-view_gaze_target_estimation.md)**

:   本文首次将注视目标估计（GTE）从单视角扩展到多视角，通过头部信息聚合（HIA）、基于不确定性的注视选择（UGS）和基于极线的场景注意力（ESA）三个模块融合多相机信息，在自建 MVGT 数据集上显著超越单视角 SOTA，并实现了单视角方法无法处理的跨视角估计。

**[Negrefine Refining Negative Label-Based Zero-Shot Ood Detection](negrefine_refining_negative_label-based_zero-shot_ood_detection.md)**

:   本文提出 NegRefine，通过 LLM 过滤负标签集中的专有名词和子类别标签，并设计多标签匹配评分函数来处理图像同时匹配分布内和负标签的情况，在 ImageNet-1K 基准上平均 AUROC 提升 1.82%、FPR95 降低 4.35%，刷新了零样本 OOD 检测 SOTA。

**[Neural Multi-View Self-Calibrated Photometric Stereo Without Photometric Stereo ](neural_multi-view_self-calibrated_photometric_stereo_without_photometric_stereo_.md)**

:   提出一种端到端的神经逆渲染框架，从多视图变化光照图像中联合恢复几何、空间变化反射率和光照参数，无需光源标定或中间光度立体线索（如法线图），超越了现有的分阶段 MVPS 方法。

**[Ngd Neural Gradient Based Deformation For Monocular Garment Reconstruction](ngd_neural_gradient_based_deformation_for_monocular_garment_reconstruction.md)**

:   提出 NGD，一种基于神经梯度的变形方法，通过将 Jacobian 场分解为帧不变的静态分量和帧相关的动态分量，结合自适应重网格化策略，从单目视频重建高保真动态纺织品几何与纹理，在宽松服装等困难场景上显著优于现有 SOTA。

**[On Large Multimodal Models As Open-World Image Classifiers](on_large_multimodal_models_as_open-world_image_classifiers.md)**

:   系统性地评估了 13 个大型多模态模型（LMM）在开放世界图像分类任务上的表现，提出包含 4 种互补指标的评估协议，揭示了 LMM 在粒度判断和细粒度区分上的系统性错误模式。

**[One-Shot Knowledge Transfer For Scalable Person Re-Identification](one-shot_knowledge_transfer_for_scalable_person_re-identification.md)**

:   提出 OSKT（One-Shot Knowledge Transfer），通过将教师模型知识精炼为"权重链"（weight chain）作为中间载体，实现一次计算即可生成任意尺寸学生模型的行人重识别模型压缩方案。

**[Openanimals Revisiting Person Re-Identification For Animals Towards Better Gener](openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)**

:   > 本文开发了 OpenAnimals 开源框架，系统回顾行人重识别方法在动物重识别中的迁移效果，提出面向动物的强基线模型 ARBase，在多个基准上大幅超越现有行人 ReID 方法。

**[Orderchain Towards General Instruct-Tuning For Stimulating The Ordinal Understan](orderchain_towards_general_instruct-tuning_for_stimulating_the_ordinal_understan.md)**

:   提出 OrderChain 提示范式，通过任务感知提示和范围优化思维链（RO-CoT）增强多模态大语言模型的序数理解能力，首次实现跨任务统一序数回归模型。

**[Posesyn Synthesizing Diverse 3D Pose Data From In-The-Wild 2D Data](posesyn_synthesizing_diverse_3d_pose_data_from_in-the-wild_2d_data.md)**

:   提出 PoseSyn 框架，通过误差提取模块（EEM）从野外 2D 姿态数据中识别目标估计器的困难样本，再通过运动合成模块（MSM）将不准确的伪标签扩展为多样化的运动序列，最终借助人体动画模型生成带有准确 3D 标注的合成训练数据，在多个真实场景基准上将 3D 姿态估计精度提升最多 14%。

**[Rapverse Coherent Vocals And Whole-Body Motion Generation From Text](rapverse_coherent_vocals_and_whole-body_motion_generation_from_text.md)**

:   构建大规模说唱数据集 RapVerse 并提出统一自回归变换器框架，首次实现从歌词文本同时生成连贯的歌声和全身3D运动。

**[Raypose Ray Bundling Diffusion For Template Views In Unseen 6D Object Pose Estim](raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)**

:   将未见物体6D位姿估计重新建模为射线对齐问题，提出物体中心的射线参数化方案，运用扩散变换器从多个已知位姿模板中推断查询图像的6D位姿。

**[Samo A Lightweight Sharpness-Aware Approach For Multi-Task Optimization With Joi](samo_a_lightweight_sharpness-aware_approach_for_multi-task_optimization_with_joi.md)**

:   提出 SAMO，一种轻量级锐度感知多任务优化方法，通过全局-局部联合扰动缓解任务梯度冲突，并利用零阶梯度近似和层级归一化大幅降低计算开销。

**[Semges Semantics-Aware Co-Speech Gesture Generation Using Semantic Coherence And](semges_semantics-aware_co-speech_gesture_generation_using_semantic_coherence_and.md)**

:   > SemGes 提出两阶段框架，通过语义一致性和语义相关性学习在全局和细粒度层面整合语义信息，生成与语音语义对齐的共语手势，在 BEAT 和 TED-Expressive 两个基准上超越现有方法。

**[Semtalk Holistic Co-Speech Motion Generation With Frame-Level Semantic Emphasis](semtalk_holistic_co-speech_motion_generation_with_frame-level_semantic_emphasis.md)**

:   > SemTalk 将共语动作分解为节奏相关的基础动作和语义感知的稀疏动作，通过学得的语义分数自适应融合两者，实现帧级语义强调的高质量全身共语动作生成。

**[Sequential Keypoint Density Estimator An Overlooked Baseline Of Skeleton-Based V](sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_v.md)**

:   > SeeKer 提出将骨架序列的联合密度在关键点级别进行自回归分解，通过预测后续关键点的条件高斯分布来检测异常人体行为，在 UBnormal 和 MSAD-HR 数据集上大幅超越现有方法。

**[Signrep Enhancing Self-Supervised Sign Representations](signrep_enhancing_self-supervised_sign_representations.md)**

:   提出 SignRep，一个可扩展的自监督手语表征学习框架，通过在 Masked Autoencoder 预训练中利用手语骨架先验、特征正则化和对抗式风格无关损失，仅用单一 RGB 模态即超越了复杂的多模态/多分支方法，在手语识别、字典检索和手语翻译三大任务上均取得 SOTA。

**[Signs As Tokens A Retrieval-Enhanced Multilingual Sign Language Generator](signs_as_tokens_a_retrieval-enhanced_multilingual_sign_language_generator.md)**

:   提出 SOKE，一种基于预训练语言模型的多语言手语生成框架，通过解耦式 tokenizer 将连续手语动作离散化为 token 序列，结合多头解码和检索增强策略，实现从文本到多语种 3D 手语 avatar 的高质量生成。

**[Switch-A-View View Selection Learned From Unlabeled In-The-Wild Videos](switch-a-view_view_selection_learned_from_unlabeled_in-the-wild_videos.md)**

:   提出 Switch-a-view 模型，通过从大规模无标注的互联网教学视频中学习视角切换模式（ego/exo），实现多视图教学视频的自动视角选择，无需显式的最佳视角标注。

**[Syncdiff Synchronized Motion Diffusion For Multi-Body Human-Object Interaction S](syncdiff_synchronized_motion_diffusion_for_multi-body_human-object_interaction_s.md)**

:   提出 SyncDiff，一个统一的多体人体-物体交互运动合成框架，通过对齐分数（alignment scores）和显式同步策略实现多体运动的精确同步，并引入频域分解来建模高频交互语义。

**[Udc-Vit A Real-World Video Dataset For Under-Display Cameras](udc-vit_a_real-world_video_dataset_for_under-display_cameras.md)**

:   提出首个真实世界屏下摄像头（UDC）视频数据集 UDC-VIT，包含 647 个视频片段共 116,460 帧，通过精心设计的双摄像头-分光器采集系统实现精确的时空对齐，并以人脸识别为核心应用场景，揭示了合成数据集在模拟真实 UDC 退化方面的不足。

**[Understanding Co-Speech Gestures In-The-Wild](understanding_co-speech_gestures_in-the-wild.md)**

:   本文提出 JEGAL——一个联合手势-语音-文本的三模态嵌入空间，通过全局短语对比损失和局部手势-词耦合损失在弱监督条件下学习共语手势表征，定义了三个新的手势理解任务和基准，超越了包括大型视觉语言模型在内的多种方法。

**[Weakly Supervised Visible-Infrared Person Re-Identification Via Heterogeneous Ex](weakly_supervised_visible-infrared_person_re-identification_via_heterogeneous_ex.md)**

:   提出首个弱监督可见光-红外行人重识别（VIReID）范式，仅使用各模态内部的身份标注（无需跨模态对应标注），通过异构专家协同一致性学习框架建立跨模态身份对应关系，性能接近全监督方法。

**[Whats Making That Sound Right Now Video-Centric Audio-Visual Localization](whats_making_that_sound_right_now_video-centric_audio-visual_localization.md)**

:   提出视频级音视频定位基准 AVATAR 和时序感知模型 TAVLO，通过高分辨率时序建模解决传统 AVL 方法忽略时间动态的问题。

**[Wir3D Visually-Informed And Geometry-Aware 3D Shape Abstraction](wir3d_visually-informed_and_geometry-aware_3d_shape_abstraction.md)**

:   > WIR3D 通过优化一组 3D Bézier 曲线参数，在 CLIP 中间层激活的空间引导下，从任意视角忠实表示 3D 形状的几何结构和视觉显著特征（包括纹理），实现稀疏但语义丰富的 3D 形状抽象。
