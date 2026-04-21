---
title: >-
  ECCV2024 人体理解方向 68篇论文解读
description: >-
  68篇ECCV2024 人体理解方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧑 人体理解

**🎞️ ECCV2024** · **68** 篇论文解读

**[3D Hand Pose Estimation in Everyday Egocentric Images](3d_hand_pose_estimation_in_everyday_egocentric_images.md)**

:   通过系统研究裁剪输入、相机内参感知位置编码(KPE)、辅助监督(手部分割+抓握标签)和多数据集联合训练这四个实践，提出WildHands系统，在仅用ResNet50和少量数据的条件下，实现了对野外第一人称图像中3D手部姿态的鲁棒估计，零样本泛化超过FrankMocap全部指标且与10倍大的HaMeR竞争。

**[3DGazeNet: Generalizing 3D Gaze Estimation with Weak-Supervision from Synthetic Views](3dgazenet_generalizing_3d_gaze_estimation_with_weak-supervision_from_synthetic_v.md)**

:   提出将视线估计重新表述为密集3D眼球网格回归，并通过从大规模野外人脸图像中自动提取伪标签+HeadGAN合成多视图进行弱监督训练，在跨域场景下比SOTA提升最多30%。

**[3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)**

:   本文提出3D空间注意力模块（3DSA），通过3D空间细分算法将特征体积划分为多个区域并为其分配基于视角的注意力权重，解决多视图3D人体姿态估计中不同视角对不同空间区域贡献不均的问题，在 CMU Panoptic Studio 数据集上达到 SOTA。

**[A Probability-guided Sampler for Neural Implicit Surface Rendering](a_probabilityguided_sampler_for_neural_implicit_surface_rend.md)**

:   提出一种概率引导的光线采样器（Probability-guided Sampler），在3D图像投影空间中建模概率密度函数来指导光线采样朝向感兴趣区域，同时设计了包含近表面和空白空间两个分量的新型表面重建损失，可作为插件集成到现有神经隐式表面渲染器中，显著提升重建精度和渲染质量。

**[A Simple Baseline for Spoken Language to Sign Language Translation with 3D Avatars](a_simple_baseline_for_spoken_language_to_sign_language_trans.md)**

:   提出首个基于3D Avatar输出的Spoken2Sign翻译基线系统，通过三步流程（字典构建→SMPLSign-X 3D手语估计→检索-连接-渲染翻译）将口语文本翻译为3D手语动画，在Phoenix-2014T上back-translation BLEU-4达25.46，同时其3D手语副产品（关键点增强和多视角理解）显著提升了手语理解任务性能。

**[AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition](adadistill_adaptive_knowledge_distillation_for_deep_face_rec.md)**

:   提出AdaDistill，将知识蒸馏概念嵌入margin penalty softmax loss中，通过基于EMA的自适应类中心（早期用sample-sample简单知识、后期用sample-center复杂知识）和困难样本感知机制，无需额外超参数即可提升轻量级人脸识别模型的判别能力，在IJB-B/C和ICCV21-MFR等挑战性基准上超越SOTA蒸馏方法。

**[ADen: Adaptive Density Representations for Sparse-view Camera Pose Estimation](aden_adaptive_density_representations_for_sparseview_camera.md)**

:   提出ADen框架，通过生成器输出多个位姿假设+判别器评分选择最佳的方式，统一了位姿回归和概率估计范式，仅需500个自适应样本即超越需要500K均匀采样的方法，同时实现实时推理。

**[Alignist: CAD-Informed Orientation Distribution Estimation by Fusing Shape and Correspondences](alignist_cad-informed_orientation_distribution_estimation_by_fusing_shape_and_co.md)**

:   提出 Alignist，首个利用 CAD 模型信息（SDF + SurfEmb 对应特征）训练隐式分布网络来推断 SO(3) 上姿态分布的方法，通过 product of experts 融合几何和特征对齐，在低数据场景下显著优于对比学习方法。

**[Audio-Driven Talking Face Generation with Stabilized Synchronization Loss](audio-driven_talking_face_generation_with_stabilized_synchronization_loss.md)**

:   提出 AVSyncNet、stabilized synchronization loss 和 silent-lip generator 三项改进，系统性地解决音频驱动说话人脸生成中 SyncNet 不稳定和嘴唇泄漏两大核心问题，在唇形同步和视觉质量上均达到 SOTA。

**[Avatar Fingerprinting for Authorized Use of Synthetic Talking-Head Videos](avatar_fingerprinting_for_authorized_use_of_synthetic_talking-head_videos.md)**

:   本文定义了"Avatar指纹识别"这一新任务——验证合成说话头视频中驱动表情的真实身份，贡献了迄今最大规模的面部重建数据集NVFAIR（161个身份），并提出基于归一化面部关键点距离和时序CNN的基线方法，通过学习与外观无关的面部运动签名实现身份验证（平均AUC 0.85），且能泛化到未见过的生成器（AUC 0.83）。

**[Bi-TTA: Bidirectional Test-Time Adapter for Remote Physiological Measurement](bi-tta_bidirectional_test-time_adapter_for_remote_physiological_measurement.md)**

:   提出 Bi-TTA 框架，首次将 Test-Time Adaptation 引入远程光电容积脉搏波 (rPPG) 任务，通过时空一致性自监督先验和前瞻-回溯双向适应策略，在推理时仅用无标注单实例数据即可完成模型域适应。

**[Bridging the Gap Between Human Motion and Action Semantics via Kinematic Phrases](bridging_the_gap_between_human_motion_and_action_semantics_via_kinematic_phrases.md)**

:   本文提出运动学短语（Kinematic Phrases, KP）作为人体运动与动作语义之间的中间表示，KP基于客观运动学事实，具有适当抽象性、可解释性和通用性，并据此构建了运动理解系统和白盒运动生成评估基准KPG。

**[Combining Generative and Geometry Priors for Wide-Angle Portrait Correction](combining_generative_and_geometry_priors_for_wide-angle_portrait_correction.md)**

:   提出结合 StyleGAN 生成式先验（用于人脸矫正）和几何对称先验（用于背景直线矫正）的双模块框架，大幅提升广角人像畸变校正的视觉质量和定量指标。

**[CoMo: Controllable Motion Generation Through Language Guided Pose Code Editing](como_controllable_motion_generation_through_language_guided_pose_code_editing.md)**

:   提出 CoMo，通过将动作序列分解为语义明确的 pose code（如"左膝微弯"），实现基于文本的可控动作生成与基于 LLM 的零样本动作编辑。

**[Cut Out the Middleman: Revisiting Pose-Based Gait Recognition](cut_out_the_middleman_revisiting_pose-based_gait_recognition.md)**

:   重新审视基于姿态的步态识别方法，提出 GaitHeat 框架，用热力图（heatmap）取代传统的骨架关键点坐标来编码人体姿态，通过改进的预处理流程和姿态引导热力图对齐模块大幅提升性能和泛化能力，使基于姿态的方法首次接近轮廓（silhouette）方法的精度。

**[Decomposed Vector-Quantized Variational Autoencoder for Human Grasp Generation](decomposed_vector-quantized_variational_autoencoder_for_human_grasp_generation.md)**

:   提出 Decomposed VQ-VAE (DVQ-VAE)，通过将手部分解为六个部分分别编码到独立码本，并设计双阶段解码策略（先姿态后位置），在四个基准数据集上质量指标相对提升约14.1%。

**[Domain Reduction Strategy for Non-Line-of-Sight Imaging](domain_reduction_strategy_for_non-line-of-sight_imaging.md)**

:   提出一种面向非视线成像（NLOS）的优化方法，通过将瞬态信号建模为逐点光传播函数的叠加，并设计由粗到细的域缩减策略剪除空白区域，在通用NLOS场景下实现约20倍加速且同时重建反射率和表面法线。

**[EgoExo-Fitness: Towards Egocentric and Exocentric Full-Body Action Understanding](egoexo-fitness_towards_egocentric_and_exocentric_full-body_action_understanding.md)**

:   提出 EgoExo-Fitness 数据集，包含同步的第一人称和第三人称健身视频，提供两级时间边界标注和创新性的可解释动作评判标注（技术关键点验证、自然语言评论、质量评分），并构建五个基准任务。

**[Event-based Head Pose Estimation: Benchmark and Method](event-based_head_pose_estimation_benchmark_and_method.md)**

:   针对事件相机头部姿态估计（HPE）领域缺乏大规模数据集和专用方法的问题，构建了两个大规模多场景事件HPE基准数据集，并提出包含事件时空融合（ESTF）和事件运动感知注意力（EMPA）两个核心模块的专用网络，在多种挑战场景下取得优异性能。

**[EvSign: Sign Language Recognition and Translation with Streaming Events](evsign_sign_language_recognition_and_translation_with_streaming_events.md)**

:   首次构建面向连续手语识别（CSLR）和手语翻译（SLT）任务的事件相机基准数据集 EvSign，并提出基于稀疏Transformer的高效框架，在仅0.34% FLOPs和44.2%参数量下达到与SOTA RGB方法可比或更优的性能。

**[Exemplar-Free Continual Representation Learning via Learnable Drift Compensation](exemplar-free_continual_representation_learning_via_learnable_drift_compensation.md)**

:   提出可学习漂移补偿(LDC)，通过训练一个前向投影器将旧特征空间映射到新特征空间，在无需存储旧样本的情况下有效补偿类原型的语义漂移，首次实现了无样本半监督持续学习。

**[Facial Affective Behavior Analysis with Instruction Tuning](facial_affective_behavior_analysis_with_instruction_tuning.md)**

:   提出首个面向面部情感行为分析（FABA）的指令微调数据集 FABA-Instruct、评测基准 FABA-Bench 以及高效 MLLM 架构 EmoLA，通过面部先验专家模块和 LoRA 适配实现了对情绪与 AU 的细粒度描述与识别。

**[FoundPose: Unseen Object Pose Estimation with Foundation Features](foundpose_unseen_object_pose_estimation_with_foundation_features.md)**

:   FoundPose 利用冻结的 DINOv2 基础模型提取 patch 描述子，通过 bag-of-words 模板检索和 kNN 匹配建立 2D-3D 对应关系，无需任何任务特定训练即可实现未见物体的 6D 位姿估计，在 BOP 基准上显著超越现有 RGB 方法。

**[FreeMotion: A Unified Framework for Number-free Text-to-Motion Synthesis](freemotion_a_unified_framework_for_number-free_text-to-motion_synthesis.md)**

:   提出FreeMotion框架，通过条件概率分解将多人运动联合分布递归拆解为单人条件运动生成，首次实现任意人数的文本驱动运动合成，并支持多人空间控制。

**[Generalizable Facial Expression Recognition](generalizable_facial_expression_recognition.md)**

:   提出 CAFE 方法，通过在固定 CLIP 人脸特征上学习 Sigmoid Mask 选取表情相关特征，配合通道分离和通道多样性损失，实现仅使用单个训练集就能在多个未见数据集上大幅超越 SOTA 表情识别方法的零样本泛化能力。

**[GraspXL: Generating Grasping Motions for Diverse Objects at Scale](graspxl_generating_grasping_motions_for_diverse_objects_at_scale.md)**

:   提出 GraspXL，一个基于强化学习的抓取动作生成框架，仅用58个物体训练即可泛化到50万+未见物体，同时支持多运动目标（抓取区域、朝向、手腕旋转、手部位置）控制和多种灵巧手平台。

**[GS-Pose: Category-Level Object Pose Estimation via Geometric and Semantic Correspondence](gs-pose_category-level_object_pose_estimation_via_geometric_and_semantic_corresp.md)**

:   提出GS-Pose方法，利用预训练视觉基础模型（DINOv2）的2D语义特征投影到3D空间，结合几何特征通过Transformer匹配网络进行类别级物体9D姿态估计，仅需10个合成3D模型训练即可在多个真实数据集上达到SOTA级别性能。

**[How Video Meetings Change Your Expression](how_video_meetings_change_your_expression.md)**

:   提出 FacET（Facial Explanations through Translations），一种基于生成式域翻译的可解释框架，通过学习解耦的面部空间特征和可解释的时空线性变换，自动发现视频会议（VC）与面对面（F2F）交流之间的细微面部表情差异模式，并支持将 VC 视频转换为 F2F 风格的"去zoom化"。

**[HPE-Li: WiFi-Enabled Lightweight Dual Selective Kernel Convolution for Human Pose Estimation](hpe-li_wifi-enabled_lightweight_dual_selective_kernel_convolution_for_human_pose.md)**

:   本文提出 HPE-Li，一种基于 WiFi 信号的轻量化人体姿态估计方法，通过创新的双选择性核注意力（SKA）机制构建多分支 CNN，能够根据输入的 WiFi CSI 数据特征动态调整感受野大小，在 MM-Fi 和 WiPose 两个基准上以极低的计算开销超越了 SOTA 方法。

**[Human Motion Forecasting in Dynamic Domain Shifts: A Homeostatic Continual Test-Time Adaptation Framework](human_motion_forecasting_in_dynamic_domain_shifts_a_homeostatic_continual_test-t.md)**

:   提出HoCoTTA框架，通过多域稳态评估和隔离参数优化策略，在持续变化的目标域中实现人体运动预测的鲁棒自适应，有效缓解了灾难性遗忘和误差累积问题。

**[HUMOS: Human Motion Model Conditioned on Body Shape](humos_human_motion_model_conditioned_on_body_shape.md)**

:   提出 HUMOS，一种基于体型条件化的人体运动生成模型，通过循环一致性损失和可微分的直觉物理/动态稳定性约束，在无配对训练数据的情况下学习体型与运动之间的相关性，生成物理可信且动态稳定的人体运动。

**[Improving Point-based Crowd Counting and Localization Based on Auxiliary Point Guidance](improving_point-based_crowd_counting_and_localization_based_on_auxiliary_point_g.md)**

:   提出辅助点引导 (APG) 策略和隐式特征插值 (IFI) 模块，通过在真值点附近显式生成辅助正负样本来稳定 point-based 人群计数方法中 proposal-target 匹配过程的不稳定性，在多个数据集上取得 SOTA。

**[Interleaving One-Class and Weakly-Supervised Models with Adaptive Thresholding for Unsupervised Video Anomaly Detection](interleaving_one-class_and_weakly-supervised_models_with_adaptive_thresholding_f.md)**

:   提出一个将加权单类分类 (wOCC) 与弱监督 (WS) 模型交替训练的无监督视频异常检测框架，通过软标签缓解训练波动、自适应阈值策略逐步优化分割阈值，无需任何人工标注即可实现接近弱监督方法的性能。

**[LaPose: Laplacian Mixture Shape Modeling for RGB-Based Category-Level Object Pose Estimation](lapose_laplacian_mixture_shape_modeling_for_rgb-based_category-level_object_pose.md)**

:   提出 LaPose 框架，通过拉普拉斯混合模型 (LMM) 建模物体形状不确定性，结合 DINOv2 通用3D流和卷积专用特征流的双流架构预测 NOCS 坐标分布，并引入尺度无关的位姿表示解决 RGB-only 场景下的固有尺度歧义，在 NOCS 数据集上取得 SOTA。

**[Large Motion Model for Unified Multi-Modal Motion Generation](large_motion_model_for_unified_multi-modal_motion_generation.md)**

:   提出 Large Motion Model (LMM)，首个以动作为中心的多模态统一动作生成基础模型，通过构建包含 10 个任务、16 个数据集、320K 序列的 MotionVerse 基准，设计支持身体部位感知的 ArtAttention 机制，以及结合随机帧率/掩码的预训练策略，实现跨任务的高质量动作生成。

**[Large Motion Model for Unified Multi-Modal Motion Generation](large_motion_model_for_unified_multimodal_motion_generation.md)**

:   LMM是首个多模态通用人体动作生成模型，统一了文本/动作/音乐/语音等10种任务、16个数据集（320K序列/1亿帧），通过身体部位感知的ArtAttention机制和可变帧率+随机遮掩的预训练策略，在多个标准benchmark上与专家模型竞争甚至超越。

**[Learning Cross-Hand Policies of High-DOF Reaching and Grasping](learning_cross-hand_policies_of_high-dof_reaching_and_grasping.md)**

:   提出一种两阶段层次化框架，通过语义关键点和交互等分面（IBS）作为手型无关的状态表示，结合Transformer策略网络和手型特定的适配模型，实现了灵巧抓取策略在不同高自由度机械手之间的零样本迁移。

**[MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation](manikin_biomechanically_accurate_neural_inverse_kinematics_for_human_motion_esti.md)**

:   本文提出MANIKIN，通过在SMPL参数模型中嵌入解剖学约束并设计基于旋转角预测的神经逆运动学求解器，从头部和手部的稀疏末端执行器姿态精确恢复全身运动，同时保证生物力学合理性和地面非穿透性。

**[MERLiN: Single-Shot Material Estimation and Relighting for Photometric Stereo](merlin_single-shot_material_estimation_and_relighting_for_photometric_stereo.md)**

:   提出单阶段注意力沙漏网络MERLiN，从单张图像联合估计空间变化BRDF参数并进行物理正确的重打光，首次利用重打光图像驱动光度立体方法实现单图法向估计，弥合了Shape from Shading与Photometric Stereo之间的鸿沟。

**[Modeling and Driving Human Body Soundfields through Acoustic Primitives](modeling_and_driving_human_body_soundfields_through_acoustic_primitives.md)**

:   提出基于声学基元(Acoustic Primitives)的人体3D声场建模与渲染框架，将多个低阶球谐声场挂载到人体骨骼关节上，在保持与SOTA可比的音质的同时，实现了15倍加速和近场声音渲染能力。

**[Motion Mamba: Efficient and Long Sequence Motion Generation](motion_mamba_efficient_and_long_sequence_motion_generation.md)**

:   本文提出 Motion Mamba，首次将选择性状态空间模型（Mamba）引入人体运动生成任务，通过层次化时序 Mamba（HTM）和双向空间 Mamba（BSM）两个核心模块，在 HumanML3D 上实现 FID 降低50%（0.473→0.281），同时推理速度提升4倍（0.217s→0.058s）。

**[Multi-Memory Matching for Unsupervised Visible-Infrared Person Re-Identification](multi-memory_matching_for_unsupervised_visible-infrared_person_re-identification.md)**

:   提出 Multi-Memory Matching（MMM）框架用于无监督可见光-红外行人重识别，通过跨模态聚类（CMC）、多记忆学习与匹配（MMLM）和软聚类级对齐损失（SCA）三个模块建立可靠的跨模态对应关系，在 SYSU-MM01 上 Rank-1 达到 61.6%，RegDB 上 Rank-1 达到 89.7%。

**[Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild](nymeria_a_massive_collection_of_multimodal_egocentric_daily_motion_in_the_wild.md)**

:   提出 Nymeria 数据集——目前最大规模的野外多模态自我中心人体日常运动数据集，包含 300 小时、264 人、50 个场景，提供全身精确动作捕捉、多设备同步多模态数据和 310.5K 句分层语言描述，并在 body tracking、motion synthesis 等任务上建立 baseline。

**[Occlusion Handling in 3D Human Pose Estimation with Perturbed Positional Encoding](occlusion_handling_in_3d_human_pose_estimation_with_perturbed_positional_encodin.md)**

:   针对人体关节遮挡导致2D骨架图边缺失、传统图拉普拉斯位置编码失效的问题，提出PerturbPE方法，利用瑞利-薛定谔微扰定理多次随机扰动并求平均来提取图拉普拉斯特征基的一致性部分作为位置编码，在完整骨架上优于MöbiusGCN，在边缺失场景下性能提升达12%。

**[One-stage Prompt-based Continual Learning](one-stage_prompt-based_continual_learning.md)**

:   提出 OS-Prompt 框架，通过直接使用 ViT 中间层 token embedding 作为 prompt query（而非额外的 query ViT 前向传播），将 Prompt-based Continual Learning 的计算成本降低约 50%，并通过 Query-Pool Regularization (QR) loss 补偿表征能力损失，在 CIFAR-100、ImageNet-R、DomainNet 上超越 CodaPrompt 约 1.4%。

**[PetFace: A Large-Scale Dataset and Benchmark for Animal Identification](petface_a_large-scale_dataset_and_benchmark_for_animal_identification.md)**

:   构建了包含13个动物科、319个品种、257,484个个体（超100万张图像）的大规模动物面部识别数据集PetFace，并建立了已见个体重识别和未见个体验证两套基准测试，为动物非侵入式自动识别提供基础设施。

**[PISR: Polarimetric Neural Implicit Surface Reconstruction for Textureless and Specular Objects](pisr_polarimetric_neural_implicit_surface_reconstruction_for_textureless_and_spe.md)**

:   提出PISR方法，利用偏振光的几何约束（偏振角与法线方位角的对应关系）直接正则化神经隐式表面形状，结合哈希网格加速和图像空间法线平滑，在无纹理和镜面物体上实现了0.5mm Chamfer距离和99.5% F-score的高精度重建，速度比此前偏振方法快4~30倍。

**[Pose-Aware Self-Supervised Learning with Viewpoint Trajectory Regularization](pose-aware_self-supervised_learning_with_viewpoint_trajectory_regularization.md)**

:   提出了一个自监督学习基准，同时评估语义分类和姿态估计能力，并设计视角轨迹正则化损失(trajectory loss)，利用相邻视角的图像三元组约束特征空间中的局部线性性，使学到的表征既保持语义分类精度又获得 emergent 的全局姿态感知能力，在域内和域外姿态估计上均提升4%。

**[PoseSOR: Human Pose Can Guide Our Attention](posesor_human_pose_can_guide_our_attention.md)**

:   本文首次将人体姿态信息引入显著目标排序(SOR)任务，通过提出姿态感知交互模块(PAI)和姿态驱动排序模块(PDR)来建模人体活动与注意力转移的关系，在复杂场景中显著提升了SOR性能并达到SOTA。

**[QUAR-VLA: Vision-Language-Action Model for Quadruped Robots](quar-vla_vision-language-action_model_for_quadruped_robots.md)**

:   首次提出四足机器人视觉-语言-动作（QUAR-VLA）范式，构建 259K episode 的多任务数据集 QUARD 和基于预训练多模态大模型的 QUART 模型，实现感知、导航、全身操作等多任务统一控制。

**[QUAR-VLA: Vision-Language-Action Model for Quadruped Robots](quarvla_visionlanguageaction_model_for_quadruped_robots.md)**

:   提出 QUAR-VLA 范式，首次将视觉、语言指令和动作生成统一到四足机器人中，构建了大规模多任务数据集 QUARD（259K episodes），训练 QUART 模型（基于 8B VLM）实现感知、导航、全身操控等多种任务，并展示了从仿真到真实的迁移能力。

**[RePOSE: 3D Human Pose Estimation via Spatio-Temporal Depth Relational Consistency](repose_3d_human_pose_estimation_via_spatio-temporal_depth_relational_consistency.md)**

:   RePOSE 提出用时空相对深度一致性损失替代传统的绝对深度监督信号，将遮挡场景下的 3D 人体姿态估计从"学习绝对深度值"转变为"学习关键点的相对深度顺序"，以极简的实现（仅需几行代码）显著提升遮挡条件下的姿态估计鲁棒性和精度。

**[SCAPE: A Simple and Strong Category-Agnostic Pose Estimator](scape_a_simple_and_strong_category-agnostic_pose_estimator.md)**

:   通过将类别无关姿态估计(CAPE)简化为纯自注意力特征匹配问题，抛弃显式相似度匹配和两阶段框架，引入全局关键点特征感知器(GKP)和关键点注意力精炼器(KAR)以提升注意力质量，在MP-100数据集上1-shot/5-shot设置下分别超越SOTA 2.2/1.3 PCK，同时减少参数量和提升推理速度。

**[Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection](self-supervised_feature_adaptation_for_3d_industrial_anomaly_detection.md)**

:   提出 LSFA（Local-to-global Self-supervised Feature Adaptation）框架，通过模态内特征紧凑性优化（IFC）和跨模态局部到全局一致性对齐（CLC）两个自监督策略对预训练特征进行任务导向适配，在 MVTec-3D AD 上取得 97.1% I-AUROC，超越 SOTA +3.4%。

**[Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection](selfsupervised_feature_adaptation_for_3d_industrial_ano.md)**

:   提出 LSFA（Local-to-global Self-supervised Feature Adaptation），通过模态内特征紧致化（IFC）和跨模态局部到全局一致性对齐（CLC）微调适配器，学习面向异常检测的任务导向表示，在 MVTec-3D AD 上达到 97.1% I-AUROC（+3.4%）。

**[SignAvatars: A Large-scale 3D Sign Language Holistic Motion Dataset and Benchmark](signavatars_a_large-scale_3d_sign_language_holistic_motion_dataset_and_benchmark.md)**

:   提出 SignAvatars，首个大规模多提示（HamNoSys/语言/单词）3D 手语全身运动数据集（70K 视频、8.34M 帧、153 名手语者），设计了带生物力学约束的自动 3D 标注流水线，并提出基于 VQ-VAE 的 SignVAE 模型作为 3D 手语生产（SLP）的首个 benchmark baseline。

**[Spectral Subsurface Scattering for Material Classification](spectral_subsurface_scattering_for_material_classification.md)**

:   提出利用Spectral Sub-Surface Scattering（S4，光谱次表面散射）进行材质分类的方法，证明了次表面散射的强光谱依赖性可以提供高度判别性的特征，并设计了一种新型成像装置通过2D投影高效获取S4测量数据，无需耗时的高光谱扫描。

**[TELA: Text to Layer-wise 3D Clothed Human Generation](tela_text_to_layer-wise_3d_clothed_human_generation.md)**

:   提出TELA方法，通过分层人体表示和渐进优化策略，从文本描述生成服装可分离的3D人体模型，支持虚拟试穿等服装编辑应用。

**[TF-FAS: Twofold-Element Fine-Grained Semantic Guidance for Generalizable Face Anti-Spoofing](tf-fas_twofold-element_fine-grained_semantic_guidance_for_generalizable_face_ant.md)**

:   本文提出TF-FAS框架，通过双重语义元素（内容元素和类别元素）的细粒度引导来增强人脸反欺骗的跨域泛化能力，其中CEDM模块探索并解耦内容相关特征，FCEM模块挖掘类别内的细粒度差异，在多个跨域FAS基准上达到SOTA。

**[Towards Unified Representation of Invariant-Specific Features in Missing Modality Face Anti-Spoofing](towards_unified_representation_of_invariant-specific_features_in_missing_modalit.md)**

:   本文提出MMA-FAS框架解决多模态人脸反欺骗中的模态缺失问题，通过模态解耦适配器从频率分解角度分离模态不变和模态特有特征，结合LBP引导的对比损失和自适应模态组合采样策略，在所有模态缺失场景下均达到SOTA。

**[U-COPE: Taking a Further Step to Universal 9D Category-Level Object Pose Estimation](u-cope_taking_a_further_step_to_universal_9d_category-level_object_pose_estimati.md)**

:   本文提出 U-COPE，首个统一处理刚性和铰接物体的类别级 9D 位姿估计框架，通过将刚性物体视为单部件铰接物体来统一问题定义，利用 Point Pair Features（PPF）独立提取各部件特征并通过通用投票策略预测关键位姿参数，在合成和真实数据集上均达到 SOTA。

**[UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues](upose3d_uncertainty-aware_3d_human_pose_estimation_with_cross-view_and_temporal_.md)**

:   提出UPose3D，一种基于不确定性感知的多视角3D人体姿态估计方法，通过Normalizing Flow建模2D关键点不确定性、可扩展的跨视角点云投影融合策略和合成数据训练的Pose Compiler模块，在无需3D标注的情况下取得OoD场景下SOTA表现，且在InD场景下与使用3D监督的方法竞争。

**[Upper-Body Hierarchical Graph for Skeleton Based Emotion Recognition in Assistive Driving](upper-body_hierarchical_graph_for_skeleton_based_emotion_recognition_in_assistiv.md)**

:   本文针对辅助驾驶场景提出 UbH-GCN，利用上半身骨骼序列构建层次化图结构（UbH-Graph）动态建模关节运动与情感的关系，并引入类别特定变化机制平衡不均衡数据分布，在 AIDE 辅助驾驶数据集上超越现有多模态方法。

**[VideoClusterNet: Self-Supervised and Adaptive Face Clustering for Videos](videoclusternet_self-supervised_and_adaptive_face_clustering_for_videos.md)**

:   VideoClusterNet 提出了一种全自监督视频人脸聚类方法：通过自蒸馏机制自适应微调通用人脸识别模型，并设计了一种基于学习损失度量的无参数聚类算法，在电影/电视剧场景中达到 SOTA。

**[Wear-Any-Way: Manipulable Virtual Try-on via Sparse Correspondence Alignment](wear-any-way_manipulable_virtual_try-on_via_sparse_correspondence_alignment.md)**

:   提出 Wear-Any-Way 框架，基于双 U-Net 扩散模型构建强基线实现高保真虚拟试穿，并通过稀疏对应对齐（Sparse Correspondence Alignment）引入点控制机制，支持用户通过点击和拖拽精确操控穿着方式（如卷袖子、开合外套、塞衣角等），在标准试穿和可操控试穿两个维度均达到 SOTA。

**[WordRobe: Text-Guided Generation of Textured 3D Garments](wordrobe_text-guided_generation_of_textured_3d_garments.md)**

:   提出 WordRobe，通过 coarse-to-fine 两阶段编码-解码框架学习 3D 服装 UDF 隐空间，利用弱监督 CLIP 映射网络实现文本驱动的 3D 服装生成与编辑，并利用 ControlNet 的 view-composited 属性在单次前向推理中生成视角一致的纹理贴图，速度比 Text2Tex 快 13 倍。

**[WordRobe: Text-Guided Generation of Textured 3D Garments](wordrobe_textguided_generation_of_textured_3d_garments.md)**

:   提出 WordRobe 框架，通过学习 3D 服装潜在空间并与 CLIP 嵌入对齐，实现文本驱动的带纹理 3D 服装网格生成，并利用 ControlNet 的单步前向推理实现高效视角一致的纹理合成。

**[WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation](worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)**

:   利用2022年FIFA世界杯体育场部署的多视角静态摄像机基础设施，构建了首个大规模多人全局3D姿态估计数据集WorldPose，包含约250万个3D姿态和超过120公里的全局轨迹，并揭示了现有全局姿态估计方法在多人场景下面临的严峻挑战。
