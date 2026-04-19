---
title: >-
  CVPR2025 人体理解方向 57篇论文解读
description: >-
  57篇CVPR2025 人体理解方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧑 人体理解

**📷 CVPR2025** · **57** 篇论文解读

**[3D Face Reconstruction From Radar Images](3d_face_reconstruction_from_radar_images.md)**

:   首次从毫米波雷达图像进行3D人脸重建：用物理雷达渲染器生成合成数据集训练CNN编码器估计BFM参数，再通过学习一个可微分雷达渲染器构建model-based autoencoder，在合成数据上实现2.56mm平均点距精度，并可在推理时无监督优化参数。

**[Analyzing the Synthetic-to-Real Domain Gap in 3D Hand Pose Estimation](analyzing_the_synthetic-to-real_domain_gap_in_3d_hand_pose_estimation.md)**

:   首次系统研究3D手势估计中合成数据到真实数据的域差距，通过可控数据合成管线分解并分析了前臂、频谱统计、手势分布、物体遮挡四个关键因素的影响，证明合理整合这些因素后纯合成数据可达到与真实数据同等的精度。

**[Anomize: Better Open Vocabulary Video Anomaly Detection](anomize_better_open_vocabulary_video_anomaly_detection.md)**

**[Breaking the Tuning Barrier: Zero-Hyperparameters Yield Multi-Corner Analysis Via Learned Priors](breaking_the_tuning_barrier_zero-hyperparameters_yield_multi-corner_analysis_via.md)**

:   用预训练的Foundation Model（TabPFN）替代传统手工先验，实现零超参数调优的电路Yield Multi-Corner Analysis：冻结backbone做in-context learning，自动跨corner迁移知识，结合自动特征选择（1152D→48D），在SRAM benchmarks上达到SOTA精度（MRE低至0.11%）且验证成本降低10倍以上。

**[ChatGarment: Garment Estimation, Generation and Editing via Large Language Models](chatgarment_garment_estimation_generation_and_editing_via_large_language_models.md)**

**[Co-op: Correspondence-Based Novel Object Pose Estimation](co-op_correspondence-based_novel_object_pose_estimation.md)**

**[Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration](collaborative_tree_search_for_enhancing_embodied_multi-agent_collaboration.md)**

:   提出 Cooperative Tree Search (CoTS) 框架，将修改版蒙特卡洛树搜索与 LLM 驱动的奖励函数结合，引导多个具身智能体进行长期战略规划和高效协作，并通过计划评估模块避免频繁计划更新带来的行为混乱，在 CWAH 和 TDW-MAT 环境上显著超越现有方法。

**[Conformal Prediction for Zero-Shot Models](conformal_prediction_for_zero-shot_models.md)**

:   将保形预测（Conformal Prediction）应用于零样本模型，为 CLIP 等模型的预测提供有理论保证的不确定性量化和校准预测集

**[ControlFace: Harnessing Facial Parametric Control for Face Rigging](controlface_harnessing_facial_parametric_control_for_face_rigging.md)**

:   提出 ControlFace，利用双分支 U-Net（FaceNet + 去噪 U-Net）结合 3DMM 渲染条件，实现无需微调即可灵活编辑人脸姿态、表情和光照，同时精确保留身份和语义细节。

**[CRISP: Object Pose and Shape Estimation with Test-Time Adaptation](crisp_object_pose_and_shape_estimation_with_test-time_adaptation.md)**

:   提出 CRISP，一个类别无关的物体姿态与形状估计 pipeline，核心创新在于基于 active shape model 的优化校正器和 correct-and-certify 自训练策略，可在测试时自适应弥合大的域差距。

**[CryptoFace: End-to-End Encrypted Face Recognition](cryptoface_end-to-end_encrypted_face_recognition.md)**

:   提出 CryptoFace，首个端到端全同态加密（FHE）人脸识别系统，通过混合浅层 patch CNN 架构（CryptoFaceNet）大幅降低乘法深度，实现比 SOTA FHE 网络快 7 倍的加密推理，同时提升验证精度。

**[Design2GarmentCode: Turning Design Concepts to Tangible Garments Through Program Synthesis](design2garmentcode_turning_design_concepts_to_tangible_garments_through_program_.md)**

:   提出 Design2GarmentCode，首个神经符号方法将多模态设计输入（文本/图像/草图）转化为参数化服装制版程序（GarmentCode DSL），实现 100% 仿真成功率和 88.67% 的用户满意度，且生成的程序可编辑、可参数化。

**[DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations](dualtalk_dual-speaker_interaction_for_3d_talking_head_conversations.md)**

:   提出 DualTalk——首个统一建模说话者和倾听者行为的多轮双人交互 3D 说话人头生成框架，配套构建了包含 50 小时、1000+ 身份的双人对话数据集。

**[Efficient Video Face Enhancement with Enhanced Spatial-Temporal Consistency](efficient_video_face_enhancement_with_enhanced_spatial-temporal_consistency.md)**

:   本文提出一种基于 3D-VQGAN 的高效盲人脸视频增强框架，通过设计空间-时间双码本记录高质量肖像特征和运动残差信息，配合边际先验正则化缓解码本崩溃问题，在 BFVR 和去闪烁任务上实现了 SOTA 效果且推理速度提升 2-140 倍。

**[EgoPressure: A Dataset for Hand Pressure and Pose Estimation in Egocentric Vision](egopressure_a_dataset_for_hand_pressure_and_pose_estimation_in_egocentric_vision.md)**

:   EgoPressure 提出首个第一人称视角的手部触觉压力和姿态数据集，包含 21 名参与者 5 小时的 RGB-D 交互数据、基于多视角优化的高保真 MANO 手部网格标注和压力传感器的真实压力映射，并建立了从 RGB 图像估计手部压力和姿态的基准模型。

**[Enhancing 3D Gaze Estimation in the Wild Using Weak Supervision with Gaze Following Labels](enhancing_3d_gaze_estimation_in_the_wild_using_weak_supervision_with_gaze_follow.md)**

:   提出一种两阶段自训练弱监督框架 ST-WSGE，利用 2D 注视跟随数据集（如 GazeFollow）生成 3D 伪标签来增强野外 3D 注视估计的泛化能力，同时设计了模态无关的 Gaze Transformer（GaT）统一处理图像和视频输入，在 Gaze360、GFIE、MPIIFaceGaze 等数据集上取得 SOTA。

**[FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video](fate_full-head_gaussian_avatar_with_textural_editing_from_monocular_video.md)**

:   提出 FATE，从单目视频重建可动画化的全头高斯化身，通过基于采样的密化策略（替代阈值分裂）、神经烘焙（将离散高斯转为连续UV纹理图以支持编辑）和通用补全框架（合成后脑外观），实现仅 49K 高斯即达到 28.37dB PSNR 的高效高质量重建。

**[Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection](forensics_adapter_adapting_clip_for_generalizable_face_forgery_detection.md)**

:   提出 Forensics Adapter，一个仅 5.7M 参数的轻量适配器网络，与冻结 CLIP 并行学习人脸伪造的融合边界特征，通过掩码边界预测+逐块对比+样本级对比三重目标实现跨数据集的高泛化性人脸伪造检测，CDF-v1 上 AUC 达 0.914。

**[FreeCloth: Free-Form Generation Enhances Challenging Clothed Human Modeling](freecloth_free-form_generation_enhances_challenging_clothed_human_modeling.md)**

:   提出 FreeCloth 混合框架，将人体表面分为"裸露/变形/生成"三类区域，对贴身衣物用 LBS 变形、对宽松服装（裙子、长裙）用无 LBS 约束的自由形态生成器建模，在 ReSynth 数据集上取得 SOTA，尤其在宽松服装场景下大幅超越现有方法。

**[FRESA: Feedforward Reconstruction of Personalized Skinned Avatars from Few Images](fresa_feedforward_reconstruction_of_personalized_skinned_avatars_from_few_images.md)**

:   提出 FRESA，通过学习一个通用着装人体先验模型，从少量图像前馈式（18秒）联合推理个性化 canonical 形状、蒙皮权重和姿态依赖变形，实现零样本泛化到手机照片的高质量可动画化 3D 人体 Avatar 重建。

**[FSboard: Over 3 Million Characters of ASL Fingerspelling Collected via Smartphones](fsboard_over_3_million_characters_of_asl_fingerspelling_collected_via_smartphone.md)**

:   发布 FSboard——迄今最大的 ASL 指拼（fingerspelling）识别数据集（320万字符、266小时视频、147位聋人签名者用智能手机自拍录制），聚焦手机文字输入场景，基线模型用 MediaPipe + ByT5 达到 11.1% CER，为指拼作为手机输入方式提供了坚实的数据基础。

**[GaussianIP: Identity-Preserving Realistic 3D Human Generation via Human-Centric Diffusion Prior](gaussianip_identity-preserving_realistic_3d_human_generation_via_human-centric_d.md)**

:   提出 GaussianIP 两阶段框架，通过自适应人体蒸馏采样（AHDS）从人体中心扩散模型高效生成身份一致的 3D 高斯人体，再通过视角一致性精炼（VCR）机制利用 mutual attention 增强面部和服饰纹理细节，在 40 分钟内完成训练并显著优于现有方法。

**[GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)**

:   GCE-Pose 提出一种"先补全再聚合"的策略，通过语义形状重建（SSR）模块将部分观测补全为完整的几何+语义 3D 表示，再通过全局上下文增强（GCE）特征融合模块将全局信息注入局部关键点特征，在 HouseCat6D 和 NOCS-REAL275 上显著超越现有方法。

**[GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities](gigahands_a_massive_annotated_dataset_of_bimanual_hand_activities.md)**

:   GigaHands 是迄今为止最大的双手活动数据集，通过设计"指令-标注"程序化采集策略和 51 相机无标记捕捉系统，收集了 34 小时、56 名被试、417 个物体的双手活动数据，包含 1.83 亿帧 RGB 图像和 84K 条详细文本标注，在文本驱动手部动作生成和动作描述任务上展示了数据规模的价值。

**[Hearing Anywhere in Any Environment](hearing_anywhere_in_any_environment.md)**

:   提出 xRIR，一个可跨房间泛化的声脉冲响应（RIR）预测统一模型，结合全景深度图的几何特征提取器和少量参考 RIR 的声学编码器，配合新构建的 AcousticRooms 数据集（260 个房间、30 万+ RIR），在已见/未见模拟环境和真实环境中均大幅超越基线方法。

**[HEIE: MLLM-Based Hierarchical Explainable AIGC Image Implausibility Evaluator](heie_mllm-based_hierarchical_explainable_aigc_image_implausibility_evaluator.md)**

:   提出HEIE——基于多模态大语言模型（MLLM）的层次化可解释AIGC图像不合理性评估器，通过CoT驱动的三位一体评估器同时输出热力图、评分和文字解释，并用自适应层次化不合理性映射器实现全局-局部缺陷的精准定位，在RichHF-18K和AbHuman数据集上达到SOTA。

**[HiPART: Hierarchical Pose AutoRegressive Transformer for Occluded 3D Human Pose Estimation](hipart_hierarchical_pose_autoregressive_transformer_for_occluded_3d_human_pose_e.md)**

:   HiPART 提出从稀疏 2D 姿态（17 关节）生成层次化稠密 2D 姿态（48→96 关节）的自回归生成方案，用丰富的骨架上下文替代复杂的时序/视觉编码器来解决遮挡问题，在单帧 3D HPE 上达到 SOTA 且超越多数多帧方法，同时参数量和计算量更小。

**[Homogeneous Dynamics Space for Heterogeneous Humans](homogeneous_dynamics_space_for_heterogeneous_humans.md)**

:   本文提出 HDyS（Homogeneous Dynamics Space），通过聚合来自生物力学和强化学习的异构人体运动数据，训练一个同构潜空间来统一不同运动学和动力学表征，实现了从运动学到动力学的高质量双向映射，并在逆动力学估计、地面反力预测等下游任务上展现了有效性。

**[HOP: Heterogeneous Topology-based Multimodal Entanglement for Co-Speech Gesture Generation](hop_heterogeneous_topology-based_multimodal_entanglement_for_co-speech_gesture_g.md)**

:   本文提出 HOP，一种基于异构拓扑的多模态纠缠方法，通过将音频作为桥梁，利用重编程模块对齐音频-文本语义、利用时空图网络对齐音频-动作节奏，实现更自然连贯的语音伴随手势生成，在 FGD、BC 和多样性指标上达到 SOTA。

**[HotSpot: Signed Distance Function Optimization with an Asymptotically Sufficient Condition](hotspot_signed_distance_function_optimization_with_an_asymptotically_sufficient_.md)**

:   提出 HotSpot，基于 screened Poisson 方程与距离的经典关系设计新的 SDF 优化损失，提供了渐近充分条件保证收敛到真正的距离函数（而非仅满足 Eikonal 的伪解），同时自然惩罚多余表面积，在复杂形状上显著优于 SAL/DiGS/StEik。

**[Improve Representation for Imbalanced Regression through Geometric Constraints](improve_representation_for_imbalanced_regression_through_geometric_constraints.md)**

:   本文首次研究深度不平衡回归（DIR）中的表征空间均匀性问题，提出包络损失（enveloping loss）和同质性损失（homogeneity loss）两种几何约束来确保回归表征在超球面上均匀分布，并设计代理驱动表征学习（SRL）框架将全局几何约束整合到mini-batch训练中，在年龄估计等多个DIR任务上达到SOTA。

**[L2GTX: From Local to Global Time Series Explanations](l2gtx_from_local_to_global_time_series_explanations.md)**

:   L2GTX 提出一种完全模型无关的时间序列分类全局解释方法，通过聚合 LOMATCE 产生的参数化时间事件原语（PEPs）构建类级全局解释，在六个基准数据集上保持稳定的全局忠实度（R²）。

**[Learning Affine Correspondences by Integrating Geometric Constraints](learning_affine_correspondences_by_integrating_geometric_constraints.md)**

:   提出一种融合稠密匹配与几何约束的仿射对应估计新框架（DenseAffine），采用两阶段解耦训练：先用 Sampson 距离损失训练稠密点匹配器，再冻结匹配器、用仿射 Sampson 距离损失训练局部仿射变换提取器，在 HPatches 匹配和 MegaDepth 位姿估计上均取得 SOTA。

**[Learning Phase Distortion with Selective State Space Models for Video Turbulence Mitigation](learning_phase_distortion_with_selective_state_space_models_for_video_turbulence.md)**

:   提出 MambaTM——首个基于 Mamba 的视频大气湍流消除网络，通过 VAE 将传统 Zernike 多项式表示的相位畸变重参数化为潜在相位畸变（LPD），用 LPD 引导 SSM 的状态转移；在保持线性复杂度和全局感受野的同时，实现了 SOTA 恢复质量和接近 2× 的推理加速（55.4 FPS vs 32.7 FPS）。

**[Learning Physics-Based Full-Body Human Reaching and Grasping from Brief Walking References](learning_physics-based_full-body_human_reaching_and_grasping_from_brief_walking_.md)**

:   仅使用约 30 秒的行走 MoCap 数据，通过将行走动作中的可迁移运动模式（浅层网络特征对齐）与运动学方法生成的抓取姿态（主动数据扩充策略）相结合，实现了物理可行、自然流畅的全身人体接近-抓取运动生成，在简单场景下抓取成功率达 99.8%。

**[Less is More: Efficient Model Merging with Binary Task Switch](less_is_more_efficient_model_merging_with_binary_task_switch.md)**

:   通过控制实验发现任务向量具有"脉冲特性"——只有幅度超过阈值的参数对任务有正贡献，据此提出T-Switch方法将任务向量二值化为激活开关、极性开关和缩放旋钮三个组件，仅需1-3%的存储空间即可实现显著优于现有基线的动态模型合并效果。

**[MagicArticulate: Make Your 3D Models Articulation-Ready](magicarticulate_make_your_3d_models_articulation-ready.md)**

:   提出 MagicArticulate 两阶段框架，第一阶段用自回归 Transformer 将骨架生成建模为序列预测任务，第二阶段用函数扩散过程结合体积测地距离先验预测蒙皮权重，搭配 33K+ 大规模 Articulation-XL 数据集，实现静态 3D 模型到可动画化资产的自动转换。

**[ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning](maniptrans_efficient_dexterous_bimanual_manipulation_transfer_via_residual_learn.md)**

:   提出 ManipTrans，两阶段残差学习框架将人手动捕数据迁移到灵巧机器手的双手操作：Stage-1 在纯手轨迹上预训练模仿模型（手腕+手指跟踪+平滑奖励），Stage-2 通过残差模块+课程学习加入物体交互约束（物体跟踪+接触力），在 OakInk-V2 上物体旋转误差仅 8.60°、双手成功率 39.5%。

**[MaRI: Material Retrieval Integration across Domains](mari_material_retrieval_integration_across_domains.md)**

:   提出 MaRI 框架，用双 DINOv2 编码器（图像 + 材质）通过对比学习构建共享嵌入空间，结合 Blender 合成数据和 ZeST 生成的真实世界材质数据，实现跨域准确的 PBR 材质检索。

**[MDP: Multidimensional Vision Model Pruning with Latency Constraint](mdp_multidimensional_vision_model_pruning_with_latency_constraint.md)**

:   MDP 提出多维度剪枝范式，将通道、注意力头、Q/K/V、嵌入维度和整个 block 等不同粒度的结构化剪枝统一建模为混合整数非线性规划(MINLP)问题，在严格延迟约束下联合求解全局最优剪枝结构，在高剪枝比下大幅超越已有方法。

**[MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities](mg-motionllm_a_unified_framework_for_motion_comprehension_and_generation_across_.md)**

:   MG-MotionLLM 提出了一个统一的多粒度动作-语言模型，通过 Motion VQ-VAE + T5 语言模型的架构和精心设计的多粒度协同预训练方案（含 28 种任务），同时支持粗粒度和细粒度的动作理解与生成，在经典任务上达到 SOTA 的同时开启了细粒度动作编辑等新应用。

**[MM-CondChain: A Programmatically Verified Benchmark for Visually Grounded Deep Compositional Reasoning](mm-condchain_a_programmatically_verified_benchmark_for_visually_grounded_deep_co.md)**

:   MM-CondChain 是首个针对视觉基础深层组合推理的 MLLM 基准，通过可验证程序中间表示（VPIR）自动构建多层条件链和链式硬负样本，最强模型仅获 53.33 Path F1，揭示深层组合推理是根本挑战。

**[MoEE: Mixture of Emotion Experts for Audio-Driven Portrait Animation](moee_mixture_of_emotion_experts_for_audio-driven_portrait_animation.md)**

:   提出情绪混合专家（MoEE）模型，为 6 种基础情绪各训练一个专家网络并通过 Soft MoE 门控组合，配合 150 小时专业情绪数据集和多模态情绪条件模块，实现对单一及复合情绪的精确、自然控制。

**[MotionMap: Representing Multimodality in Human Pose Forecasting](motionmap_representing_multimodality_in_human_pose_forecasting.md)**

:   提出MotionMap——用热力图表示运动空间分布的新范式，通过t-SNE降维+codebook实现可变数量模式预测和置信度量化，以最少采样实现最佳模式覆盖。

**[Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties](multi-sensor_object_anomaly_detection_unifying_appearance_geometry_and_internal_.md)**

:   提出 MulSen-AD，首个融合 RGB 相机、激光扫描仪和红外热成像三种传感器的工业物体异常检测数据集（15 类产品、14 种异常），并设计 MulSen-TripleAD 决策级融合基线方法，实现 96.1% AUROC，证明多传感器融合显著优于单传感器方法。

**[NBAvatar: Neural Billboards Avatars with Realistic Hand-Face Interaction](nbavatar_neural_billboards_avatars_with_realistic_hand-face_interaction.md)**

:   NBAvatar 提出 Neural Billboard 原语——将可学习平面几何原语与神经纹理延迟渲染结合，实现手脸交互场景下的照片级真实头部 avatar 渲染，在百万像素分辨率下 LPIPS 比 Gaussian 方法降低 30%。

**[One2Any: One-Reference 6D Pose Estimation for Any Object](one2any_one-reference_6d_pose_estimation_for_any_object.md)**

:   提出 One2Any，仅需单张参考图像即可估计任意新物体的 6D 位姿——用参考物体坐标（ROC，以参考相机帧为基准而非规范坐标）编码参考姿态，通过 VQVAE+U-Net 条件生成密集 ROC 图，再用 Umeyama 算法恢复位姿，在 YCB-Video 上 93.7% ADD-S AUC，推理仅 0.09 秒。

**[Perceive What Matters: Relevance-Driven Scheduling for Multimodal Streaming Perception](perceive_what_matters_relevance-driven_scheduling_for_multimodal_streaming_perce.md)**

:   提出一种面向人机协作的感知调度框架，基于信息增益和计算代价的权衡来选择性激活感知模块（目标检测/姿态估计），在流式感知场景下将计算延迟降低最多 27.52%，同时 MMPose 激活召回提升 72.73%。

**[PoseBH: Prototypical Multi-Dataset Training Beyond Human Pose Estimation](posebh_prototypical_multi-dataset_training_beyond_human_pose_estimation.md)**

:   提出 PoseBH，通过非参数关键点原型（Sinkhorn-Knopp 在线聚类）和跨类型自监督（CSS）实现人/动物/手部等不同骨骼定义数据集的统一训练，在 APT-36K 动物视频数据集上比 ViTPose++ 提升 11.2 AP，证明跨类型知识迁移的有效性。

**[Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)**

:   提出 VTON-IQA，一个无参考的虚拟试穿图像质量评估框架，通过大规模人类标注基准 VTON-QBench（62,688 张试穿图 + 431,800 条标注）和 Interleaved Cross-Attention 模块实现与人类感知对齐的图像级质量预测。

**[Removing Reflections from RAW Photos](removing_reflections_from_raw_photos.md)**

:   提出首个基于 RAW 图像的端到端去反射系统：在 XYZ 色彩空间中模拟逼真的反射（含 Fresnel/双反射/WB/曝光），训练 EfficientNet+BiFPN 基础模型分离透射/反射层，再用高斯金字塔上采样器保留高分辨率细节，利用可选的自拍相机上下文图辅助判断，PSNR 30.62dB。

**[Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents](scalable_video-to-dataset_generation_for_cross-platform_mobile_agents.md)**

:   MONDAY 框架从 YouTube 教学视频自动生成移动端导航数据集——通过 OCR 场景转换检测和 GPT-4o 的 3 步动作识别流程，以人工标注 1/17 的成本（$0.34 vs $5.76/视频）构建了覆盖 iOS/Android 双平台的 313K 标注帧，预训练后 agent 在未见的 Windows Mobile 上提升 18.11%。

**[ShowMak3r: Compositional TV Show Reconstruction](showmak3r_compositional_tv_show_reconstruction.md)**

:   提出 ShowMak3r，首个从网络视频（芭蕾/跑酷/电影片段）中进行组合式 4D 重建的完整流水线：3DGS 舞台重建 + 深度引导的 SMPL 时空定位 + 跨镜头演员匹配 + 隐式 MLP 面部表情精化，舞台 PSNR 19.65，面部 PSNR 24.34（比 ExAvatar +4.17）。

**[ShowUI: One Vision-Language-Action Model for GUI Visual Agent](showui_one_vision-language-action_model_for_gui_visual_agent.md)**

:   ShowUI 基于 Qwen2-VL-2B，通过 UI 连通图引导的视觉 token 选择减少 33% 冗余 token 并加速 1.4 倍，配合交错式视觉-语言-动作流和精选 256K 训练数据，仅 2B 参数即在零样本 ScreenSpot 上达到 75.1% 的 SOTA 精度。

**[Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach](team_ras_in_10th_abaw_competition_multimodal_valence_and_arousal_estimation_appr.md)**

:   提出结合面部（GRADA+Transformer）、行为描述（Qwen3-VL+Mamba）和音频（WavLM）三模态的连续情感估计方法，通过 Directed Cross-Modal MoE 和 Reliability-Aware Audio-Visual 两种融合策略在 Aff-Wild2 上达到 CCC 0.6576（dev）/ 0.62（test）。

**[Two is Better than One: Efficient Ensemble Defense for Robust and Compact Models](two_is_better_than_one_efficient_ensemble_defense_for_robust_and_compact_models.md)**

:   提出 EED（Efficient Ensemble Defense），从单个基础模型通过不同剪枝策略（NIS/ERM/ASE/BNSF）生成多个子模型并动态集成——在 80% 稀疏度下 CIFAR-10 PGD 鲁棒准确率 55.71%（接近未压缩基线），推理加速 1.86 倍。

**[VI3NR: Variance Informed Initialization for Implicit Neural Representations](vi3nr_variance_informed_initialization_for_implicit_neural_representations.md)**

:   推导了适用于任意激活函数的隐式神经表示（INR）初始化方法 VI3NR，将 Xavier/Kaiming 初始化推广到 Gaussian/Sinc 等非标准激活——通过控制前向和反向传播的方差一致性，用一个自由度 $\sigma_p^2$ 同时满足两个方向的稳定性，显著改善 INR 的收敛速度和重建质量。
