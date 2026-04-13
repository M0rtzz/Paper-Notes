---
title: >-
  ECCV2024 人体理解方向 52篇论文解读
description: >-
  52篇ECCV2024 人体理解方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧑 人体理解

**🎞️ ECCV2024** · 共 **52** 篇

**[3D Hand Pose Estimation In Everyday Egocentric Images](3d_hand_pose_estimation_in_everyday_egocentric_images.md)**

:   通过系统研究裁剪输入、相机内参感知位置编码(KPE)、辅助监督(手部分割+抓握标签)和多数据集联合训练这四个实践，提出WildHands系统，在仅用ResNet50和少量数据的条件下，实现了对野外第一人称图像中3D手部姿态的鲁棒估计，零样本泛化超过FrankMocap全部指标且与10倍大的HaMeR竞争。

**[3Dgazenet Generalizing 3D Gaze Estimation With Weak-Supervision From Synthetic V](3dgazenet_generalizing_3d_gaze_estimation_with_weak-supervision_from_synthetic_v.md)**

:   提出将视线估计重新表述为密集3D眼球网格回归，并通过从大规模野外人脸图像中自动提取伪标签+HeadGAN合成多视图进行弱监督训练，在跨域场景下比SOTA提升最多30%。

**[A Probabilityguided Sampler For Neural Implicit Surface Rend](a_probabilityguided_sampler_for_neural_implicit_surface_rend.md)**

:   提出一种概率引导的光线采样器（Probability-guided Sampler），在3D图像投影空间中建模概率密度函数来指导光线采样朝向感兴趣区域，同时设计了包含近表面和空白空间两个分量的新型表面重建损失，可作为插件集成到现有神经隐式表面渲染器中，显著提升重建精度和渲染质量。

**[A Simple Baseline For Spoken Language To Sign Language Trans](a_simple_baseline_for_spoken_language_to_sign_language_trans.md)**

:   提出首个基于3D Avatar输出的Spoken2Sign翻译基线系统，通过三步流程（字典构建→SMPLSign-X 3D手语估计→检索-连接-渲染翻译）将口语文本翻译为3D手语动画，在Phoenix-2014T上back-translation BLEU-4达25.46，同时其3D手语副产品（关键点增强和多视角理解）显著提升了手语理解任务性能。

**[Adadistill Adaptive Knowledge Distillation For Deep Face Rec](adadistill_adaptive_knowledge_distillation_for_deep_face_rec.md)**

:   提出AdaDistill，将知识蒸馏概念嵌入margin penalty softmax loss中，通过基于EMA的自适应类中心（早期用sample-sample简单知识、后期用sample-center复杂知识）和困难样本感知机制，无需额外超参数即可提升轻量级人脸识别模型的判别能力，在IJB-B/C和ICCV21-MFR等挑战性基准上超越SOTA蒸馏方法。

**[Aden Adaptive Density Representations For Sparseview Camera](aden_adaptive_density_representations_for_sparseview_camera.md)**

:   ADen提出生成器-判别器框架统一位姿回归和概率位姿估计：生成器输出多个6DoF位姿假设来建模多模态分布（处理对称歧义），判别器选出最佳假设，在稀疏视角位姿估计上同时实现了更高精度和更低运行时间。

**[Alignist Cad-Informed Orientation Distribution Estimation By Fusing Shape And Co](alignist_cad-informed_orientation_distribution_estimation_by_fusing_shape_and_co.md)**

:   提出 Alignist，首个利用 CAD 模型信息（SDF + SurfEmb 对应特征）训练隐式分布网络来推断 SO(3) 上姿态分布的方法，通过 product of experts 融合几何和特征对齐，在低数据场景下显著优于对比学习方法。

**[Audio-Driven Talking Face Generation With Stabilized Synchronization Loss](audio-driven_talking_face_generation_with_stabilized_synchronization_loss.md)**

:   提出 AVSyncNet、stabilized synchronization loss 和 silent-lip generator 三项改进，系统性地解决音频驱动说话人脸生成中 SyncNet 不稳定和嘴唇泄漏两大核心问题，在唇形同步和视觉质量上均达到 SOTA。

**[Bi-Tta Bidirectional Test-Time Adapter For Remote Physiological Measurement](bi-tta_bidirectional_test-time_adapter_for_remote_physiological_measurement.md)**

:   提出 Bi-TTA 框架，首次将 Test-Time Adaptation 引入远程光电容积脉搏波 (rPPG) 任务，通过时空一致性自监督先验和前瞻-回溯双向适应策略，在推理时仅用无标注单实例数据即可完成模型域适应。

**[Combining Generative And Geometry Priors For Wide-Angle Portrait Correction](combining_generative_and_geometry_priors_for_wide-angle_portrait_correction.md)**

:   提出结合 StyleGAN 生成式先验（用于人脸矫正）和几何对称先验（用于背景直线矫正）的双模块框架，大幅提升广角人像畸变校正的视觉质量和定量指标。

**[Como Controllable Motion Generation Through Language Guided Pose Code Editing](como_controllable_motion_generation_through_language_guided_pose_code_editing.md)**

:   提出 CoMo，通过将动作序列分解为语义明确的 pose code（如"左膝微弯"），实现基于文本的可控动作生成与基于 LLM 的零样本动作编辑。

**[Decomposed Vector-Quantized Variational Autoencoder For Human Grasp Generation](decomposed_vector-quantized_variational_autoencoder_for_human_grasp_generation.md)**

:   提出 Decomposed VQ-VAE (DVQ-VAE)，通过将手部分解为六个部分分别编码到独立码本，并设计双阶段解码策略（先姿态后位置），在四个基准数据集上质量指标相对提升约14.1%。

**[Domain Reduction Strategy For Non-Line-Of-Sight Imaging](domain_reduction_strategy_for_non-line-of-sight_imaging.md)**

:   提出一种面向非视线成像（NLOS）的优化方法，通过将瞬态信号建模为逐点光传播函数的叠加，并设计由粗到细的域缩减策略剪除空白区域，在通用NLOS场景下实现约20倍加速且同时重建反射率和表面法线。

**[Egoexo-Fitness Towards Egocentric And Exocentric Full-Body Action Understanding](egoexo-fitness_towards_egocentric_and_exocentric_full-body_action_understanding.md)**

:   提出 EgoExo-Fitness 数据集，包含同步的第一人称和第三人称健身视频，提供两级时间边界标注和创新性的可解释动作评判标注（技术关键点验证、自然语言评论、质量评分），并构建五个基准任务。

**[Evsign Sign Language Recognition And Translation With Streaming Events](evsign_sign_language_recognition_and_translation_with_streaming_events.md)**

:   首次构建面向连续手语识别（CSLR）和手语翻译（SLT）任务的事件相机基准数据集 EvSign，并提出基于稀疏Transformer的高效框架，在仅0.34% FLOPs和44.2%参数量下达到与SOTA RGB方法可比或更优的性能。

**[Exemplar-Free Continual Representation Learning Via Learnable Drift Compensation](exemplar-free_continual_representation_learning_via_learnable_drift_compensation.md)**

:   提出可学习漂移补偿(LDC)，通过训练一个前向投影器将旧特征空间映射到新特征空间，在无需存储旧样本的情况下有效补偿类原型的语义漂移，首次实现了无样本半监督持续学习。

**[Facial Affective Behavior Analysis With Instruction Tuning](facial_affective_behavior_analysis_with_instruction_tuning.md)**

:   提出首个面向面部情感行为分析（FABA）的指令微调数据集 FABA-Instruct、评测基准 FABA-Bench 以及高效 MLLM 架构 EmoLA，通过面部先验专家模块和 LoRA 适配实现了对情绪与 AU 的细粒度描述与识别。

**[Foundpose Unseen Object Pose Estimation With Foundation Features](foundpose_unseen_object_pose_estimation_with_foundation_features.md)**

:   FoundPose 利用冻结的 DINOv2 基础模型提取 patch 描述子，通过 bag-of-words 模板检索和 kNN 匹配建立 2D-3D 对应关系，无需任何任务特定训练即可实现未见物体的 6D 位姿估计，在 BOP 基准上显著超越现有 RGB 方法。

**[Freemotion A Unified Framework For Number-Free Text-To-Motion Synthesis](freemotion_a_unified_framework_for_number-free_text-to-motion_synthesis.md)**

:   提出FreeMotion框架，通过条件概率分解将多人运动联合分布递归拆解为单人条件运动生成，首次实现任意人数的文本驱动运动合成，并支持多人空间控制。

**[Generalizable Facial Expression Recognition](generalizable_facial_expression_recognition.md)**

:   提出 CAFE 方法，通过在固定 CLIP 人脸特征上学习 Sigmoid Mask 选取表情相关特征，配合通道分离和通道多样性损失，实现仅使用单个训练集就能在多个未见数据集上大幅超越 SOTA 表情识别方法的零样本泛化能力。

**[Graspxl Generating Grasping Motions For Diverse Objects At Scale](graspxl_generating_grasping_motions_for_diverse_objects_at_scale.md)**

:   提出 GraspXL，一个基于强化学习的抓取动作生成框架，仅用58个物体训练即可泛化到50万+未见物体，同时支持多运动目标（抓取区域、朝向、手腕旋转、手部位置）控制和多种灵巧手平台。

**[Gs-Pose Category-Level Object Pose Estimation Via Geometric And Semantic Corresp](gs-pose_category-level_object_pose_estimation_via_geometric_and_semantic_corresp.md)**

:   提出GS-Pose方法，利用预训练视觉基础模型（DINOv2）的2D语义特征投影到3D空间，结合几何特征通过Transformer匹配网络进行类别级物体9D姿态估计，仅需10个合成3D模型训练即可在多个真实数据集上达到SOTA级别性能。

**[How Video Meetings Change Your Expression](how_video_meetings_change_your_expression.md)**

:   提出 FacET（Facial Explanations through Translations），一种基于生成式域翻译的可解释框架，通过学习解耦的面部空间特征和可解释的时空线性变换，自动发现视频会议（VC）与面对面（F2F）交流之间的细微面部表情差异模式，并支持将 VC 视频转换为 F2F 风格的"去zoom化"。

**[Humos Human Motion Model Conditioned On Body Shape](humos_human_motion_model_conditioned_on_body_shape.md)**

:   提出 HUMOS，一种基于体型条件化的人体运动生成模型，通过循环一致性损失和可微分的直觉物理/动态稳定性约束，在无配对训练数据的情况下学习体型与运动之间的相关性，生成物理可信且动态稳定的人体运动。

**[Improving Point-Based Crowd Counting And Localization Based On Auxiliary Point G](improving_point-based_crowd_counting_and_localization_based_on_auxiliary_point_g.md)**

:   提出辅助点引导 (APG) 策略和隐式特征插值 (IFI) 模块，通过在真值点附近显式生成辅助正负样本来稳定 point-based 人群计数方法中 proposal-target 匹配过程的不稳定性，在多个数据集上取得 SOTA。

**[Interleaving One-Class And Weakly-Supervised Models With Adaptive Thresholding F](interleaving_one-class_and_weakly-supervised_models_with_adaptive_thresholding_f.md)**

:   提出一个将加权单类分类 (wOCC) 与弱监督 (WS) 模型交替训练的无监督视频异常检测框架，通过软标签缓解训练波动、自适应阈值策略逐步优化分割阈值，无需任何人工标注即可实现接近弱监督方法的性能。

**[Lapose Laplacian Mixture Shape Modeling For Rgb-Based Category-Level Object Pose](lapose_laplacian_mixture_shape_modeling_for_rgb-based_category-level_object_pose.md)**

:   提出 LaPose 框架，通过拉普拉斯混合模型 (LMM) 建模物体形状不确定性，结合 DINOv2 通用3D流和卷积专用特征流的双流架构预测 NOCS 坐标分布，并引入尺度无关的位姿表示解决 RGB-only 场景下的固有尺度歧义，在 NOCS 数据集上取得 SOTA。

**[Large Motion Model For Unified Multi-Modal Motion Generation](large_motion_model_for_unified_multi-modal_motion_generation.md)**

:   提出 Large Motion Model (LMM)，首个以动作为中心的多模态统一动作生成基础模型，通过构建包含 10 个任务、16 个数据集、320K 序列的 MotionVerse 基准，设计支持身体部位感知的 ArtAttention 机制，以及结合随机帧率/掩码的预训练策略，实现跨任务的高质量动作生成。

**[Large Motion Model For Unified Multimodal Motion Generation](large_motion_model_for_unified_multimodal_motion_generation.md)**

:   LMM是首个多模态通用人体动作生成模型，统一了文本/动作/音乐/语音等10种任务、16个数据集（320K序列/1亿帧），通过身体部位感知的ArtAttention机制和可变帧率+随机遮掩的预训练策略，在多个标准benchmark上与专家模型竞争甚至超越。

**[Learning Cross-Hand Policies Of High-Dof Reaching And Grasping](learning_cross-hand_policies_of_high-dof_reaching_and_grasping.md)**

:   提出一种两阶段层次化框架，通过语义关键点和交互等分面（IBS）作为手型无关的状态表示，结合Transformer策略网络和手型特定的适配模型，实现了灵巧抓取策略在不同高自由度机械手之间的零样本迁移。

**[Merlin Single-Shot Material Estimation And Relighting For Photometric Stereo](merlin_single-shot_material_estimation_and_relighting_for_photometric_stereo.md)**

:   提出单阶段注意力沙漏网络MERLiN，从单张图像联合估计空间变化BRDF参数并进行物理正确的重打光，首次利用重打光图像驱动光度立体方法实现单图法向估计，弥合了Shape from Shading与Photometric Stereo之间的鸿沟。

**[Modeling And Driving Human Body Soundfields Through Acoustic Primitives](modeling_and_driving_human_body_soundfields_through_acoustic_primitives.md)**

:   提出基于声学基元(Acoustic Primitives)的人体3D声场建模与渲染框架，将多个低阶球谐声场挂载到人体骨骼关节上，在保持与SOTA可比的音质的同时，实现了15倍加速和近场声音渲染能力。

**[Motion Mamba Efficient And Long Sequence Motion Generation](motion_mamba_efficient_and_long_sequence_motion_generation.md)**

:   本文提出 Motion Mamba，首次将选择性状态空间模型（Mamba）引入人体运动生成任务，通过层次化时序 Mamba（HTM）和双向空间 Mamba（BSM）两个核心模块，在 HumanML3D 上实现 FID 降低50%（0.473→0.281），同时推理速度提升4倍（0.217s→0.058s）。

**[Multi-Memory Matching For Unsupervised Visible-Infrared Person Re-Identification](multi-memory_matching_for_unsupervised_visible-infrared_person_re-identification.md)**

:   提出 Multi-Memory Matching（MMM）框架用于无监督可见光-红外行人重识别，通过跨模态聚类（CMC）、多记忆学习与匹配（MMLM）和软聚类级对齐损失（SCA）三个模块建立可靠的跨模态对应关系，在 SYSU-MM01 上 Rank-1 达到 61.6%，RegDB 上 Rank-1 达到 89.7%。

**[Nymeria A Massive Collection Of Multimodal Egocentric Daily Motion In The Wild](nymeria_a_massive_collection_of_multimodal_egocentric_daily_motion_in_the_wild.md)**

:   提出 Nymeria 数据集——目前最大规模的野外多模态自我中心人体日常运动数据集，包含 300 小时、264 人、50 个场景，提供全身精确动作捕捉、多设备同步多模态数据和 310.5K 句分层语言描述，并在 body tracking、motion synthesis 等任务上建立 baseline。

**[Occlusion Handling In 3D Human Pose Estimation With Perturbed Positional Encodin](occlusion_handling_in_3d_human_pose_estimation_with_perturbed_positional_encodin.md)**

:   针对人体关节遮挡导致2D骨架图边缺失、传统图拉普拉斯位置编码失效的问题，提出PerturbPE方法，利用瑞利-薛定谔微扰定理多次随机扰动并求平均来提取图拉普拉斯特征基的一致性部分作为位置编码，在完整骨架上优于MöbiusGCN，在边缺失场景下性能提升达12%。

**[One-Stage Prompt-Based Continual Learning](one-stage_prompt-based_continual_learning.md)**

:   提出 OS-Prompt 框架，通过直接使用 ViT 中间层 token embedding 作为 prompt query（而非额外的 query ViT 前向传播），将 Prompt-based Continual Learning 的计算成本降低约 50%，并通过 Query-Pool Regularization (QR) loss 补偿表征能力损失，在 CIFAR-100、ImageNet-R、DomainNet 上超越 CodaPrompt 约 1.4%。

**[Petface A Large-Scale Dataset And Benchmark For Animal Identification](petface_a_large-scale_dataset_and_benchmark_for_animal_identification.md)**

:   构建了包含13个动物科、319个品种、257,484个个体（超100万张图像）的大规模动物面部识别数据集PetFace，并建立了已见个体重识别和未见个体验证两套基准测试，为动物非侵入式自动识别提供基础设施。

**[Pisr Polarimetric Neural Implicit Surface Reconstruction For Textureless And Spe](pisr_polarimetric_neural_implicit_surface_reconstruction_for_textureless_and_spe.md)**

:   提出PISR方法，利用偏振光的几何约束（偏振角与法线方位角的对应关系）直接正则化神经隐式表面形状，结合哈希网格加速和图像空间法线平滑，在无纹理和镜面物体上实现了0.5mm Chamfer距离和99.5% F-score的高精度重建，速度比此前偏振方法快4~30倍。

**[Pose-Aware Self-Supervised Learning With Viewpoint Trajectory Regularization](pose-aware_self-supervised_learning_with_viewpoint_trajectory_regularization.md)**

:   提出了一个自监督学习基准，同时评估语义分类和姿态估计能力，并设计视角轨迹正则化损失(trajectory loss)，利用相邻视角的图像三元组约束特征空间中的局部线性性，使学到的表征既保持语义分类精度又获得 emergent 的全局姿态感知能力，在域内和域外姿态估计上均提升4%。

**[Quar-Vla Vision-Language-Action Model For Quadruped Robots](quar-vla_vision-language-action_model_for_quadruped_robots.md)**

:   首次提出四足机器人视觉-语言-动作（QUAR-VLA）范式，构建 259K episode 的多任务数据集 QUARD 和基于预训练多模态大模型的 QUART 模型，实现感知、导航、全身操作等多任务统一控制。

**[Quarvla Visionlanguageaction Model For Quadruped Robots](quarvla_visionlanguageaction_model_for_quadruped_robots.md)**

:   提出 QUAR-VLA 范式，首次将视觉、语言指令和动作生成统一到四足机器人中，构建了大规模多任务数据集 QUARD（259K episodes），训练 QUART 模型（基于 8B VLM）实现感知、导航、全身操控等多种任务，并展示了从仿真到真实的迁移能力。

**[Scape A Simple And Strong Category-Agnostic Pose Estimator](scape_a_simple_and_strong_category-agnostic_pose_estimator.md)**

:   通过将类别无关姿态估计(CAPE)简化为纯自注意力特征匹配问题，抛弃显式相似度匹配和两阶段框架，引入全局关键点特征感知器(GKP)和关键点注意力精炼器(KAR)以提升注意力质量，在MP-100数据集上1-shot/5-shot设置下分别超越SOTA 2.2/1.3 PCK，同时减少参数量和提升推理速度。

**[Self-Supervised Feature Adaptation For 3D Industrial Anomaly Detection](self-supervised_feature_adaptation_for_3d_industrial_anomaly_detection.md)**

:   提出 LSFA（Local-to-global Self-supervised Feature Adaptation）框架，通过模态内特征紧凑性优化（IFC）和跨模态局部到全局一致性对齐（CLC）两个自监督策略对预训练特征进行任务导向适配，在 MVTec-3D AD 上取得 97.1% I-AUROC，超越 SOTA +3.4%。

**[Selfsupervised Feature Adaptation For 3D Industrial Ano](selfsupervised_feature_adaptation_for_3d_industrial_ano.md)**

:   提出 LSFA（Local-to-global Self-supervised Feature Adaptation），通过模态内特征紧致化（IFC）和跨模态局部到全局一致性对齐（CLC）微调适配器，学习面向异常检测的任务导向表示，在 MVTec-3D AD 上达到 97.1% I-AUROC（+3.4%）。

**[Signavatars A Large-Scale 3D Sign Language Holistic Motion Dataset And Benchmark](signavatars_a_large-scale_3d_sign_language_holistic_motion_dataset_and_benchmark.md)**

:   提出 SignAvatars，首个大规模多提示（HamNoSys/语言/单词）3D 手语全身运动数据集（70K 视频、8.34M 帧、153 名手语者），设计了带生物力学约束的自动 3D 标注流水线，并提出基于 VQ-VAE 的 SignVAE 模型作为 3D 手语生产（SLP）的首个 benchmark baseline。

**[Upose3D Uncertainty-Aware 3D Human Pose Estimation With Cross-View And Temporal ](upose3d_uncertainty-aware_3d_human_pose_estimation_with_cross-view_and_temporal_.md)**

:   提出UPose3D，一种基于不确定性感知的多视角3D人体姿态估计方法，通过Normalizing Flow建模2D关键点不确定性、可扩展的跨视角点云投影融合策略和合成数据训练的Pose Compiler模块，在无需3D标注的情况下取得OoD场景下SOTA表现，且在InD场景下与使用3D监督的方法竞争。

**[Videoclusternet Self-Supervised And Adaptive Face Clustering For Videos](videoclusternet_self-supervised_and_adaptive_face_clustering_for_videos.md)**

:   VideoClusterNet 提出了一种全自监督视频人脸聚类方法：通过自蒸馏机制自适应微调通用人脸识别模型，并设计了一种基于学习损失度量的无参数聚类算法，在电影/电视剧场景中达到 SOTA。

**[Wear-Any-Way Manipulable Virtual Try-On Via Sparse Correspondence Alignment](wear-any-way_manipulable_virtual_try-on_via_sparse_correspondence_alignment.md)**

:   提出 Wear-Any-Way 框架，基于双 U-Net 扩散模型构建强基线实现高保真虚拟试穿，并通过稀疏对应对齐（Sparse Correspondence Alignment）引入点控制机制，支持用户通过点击和拖拽精确操控穿着方式（如卷袖子、开合外套、塞衣角等），在标准试穿和可操控试穿两个维度均达到 SOTA。

**[Wordrobe Text-Guided Generation Of Textured 3D Garments](wordrobe_text-guided_generation_of_textured_3d_garments.md)**

:   提出 WordRobe，通过 coarse-to-fine 两阶段编码-解码框架学习 3D 服装 UDF 隐空间，利用弱监督 CLIP 映射网络实现文本驱动的 3D 服装生成与编辑，并利用 ControlNet 的 view-composited 属性在单次前向推理中生成视角一致的纹理贴图，速度比 Text2Tex 快 13 倍。

**[Wordrobe Textguided Generation Of Textured 3D Garments](wordrobe_textguided_generation_of_textured_3d_garments.md)**

:   提出 WordRobe 框架，通过学习 3D 服装潜在空间并与 CLIP 嵌入对齐，实现文本驱动的带纹理 3D 服装网格生成，并利用 ControlNet 的单步前向推理实现高效视角一致的纹理合成。

**[Worldpose A World Cup Dataset For Global 3D Human Pose Estimation](worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)**

:   利用2022年FIFA世界杯体育场部署的多视角静态摄像机基础设施，构建了首个大规模多人全局3D姿态估计数据集WorldPose，包含约250万个3D姿态和超过120公里的全局轨迹，并揭示了现有全局姿态估计方法在多人场景下面临的严峻挑战。
