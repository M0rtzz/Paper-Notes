---
title: >-
  CVPR2025 人体理解方向 89篇论文解读
description: >-
  89篇CVPR2025 人体理解方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧑 人体理解

**📷 CVPR2025** · **89** 篇论文解读

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

**[NN-Former: Rethinking Graph Structure in Neural Architecture Representation](nn-former_rethinking_graph_structure_in_neural_architecture_representation.md)**

:   NN-Former 提出混合 GNN-Transformer 架构预测器，发现现有方法忽略了"兄弟节点"（共享父/子节点）的拓扑信息，通过 Adjacency-Sibling Multihead Attention (ASMA) 和 Bidirectional Graph Isomorphism FFN (BGIFFN) 在 NAS-Bench-101/201 上 Kendall's Tau 达 0.877/0.890，延迟预测 MAPE 降低 48-64%。

**[Omni-ID: Holistic Identity Representation Designed for Generative Tasks](omni-id_holistic_identity_representation_designed_for_generative_tasks.md)**

:   Omni-ID 提出了一种专为生成任务设计的全息人脸身份表征，通过 few-to-many 身份重建训练范式和多解码器目标（Masked Transformer + Flow Matching），将不定数量的输入图像编码为固定大小的结构化表征，在可控人脸生成和个性化 T2I 任务中显著超越 ArcFace 和 CLIP。

**[One2Any: One-Reference 6D Pose Estimation for Any Object](one2any_one-reference_6d_pose_estimation_for_any_object.md)**

:   提出 One2Any，仅需单张参考图像即可估计任意新物体的 6D 位姿——用参考物体坐标（ROC，以参考相机帧为基准而非规范坐标）编码参考姿态，通过 VQVAE+U-Net 条件生成密集 ROC 图，再用 Umeyama 算法恢复位姿，在 YCB-Video 上 93.7% ADD-S AUC，推理仅 0.09 秒。

**[Optimal Transport-Guided Source-Free Adaptation for Face Anti-Spoofing](optimal_transport-guided_source-free_adaptation_for_face_anti-spoofing.md)**

:   提出 OTA 框架：训练阶段学习原型表示编码源域分布，测试阶段通过最优传输(OT)在不访问源模型参数和训练数据的前提下，以 training-free 或轻量训练方式将原型迁移到目标域，同时提出 geodesic mixup 数据增强改善低数据场景的分类器学习。

**[PEACE: Empowering Geologic Map Holistic Understanding with MLLMs](peace_empowering_geologic_map_holistic_understanding_with_mllms.md)**

:   本文构建了首个地质图理解基准 GeoMap-Bench（5 种能力、25 个任务、3864 个问题），并提出 GeoMap-Agent（层级信息提取 + 领域知识注入 + 增强问答），在地质图理解上以 0.811 的整体得分大幅超越 GPT-4o 的 0.369。

**[Perceive What Matters: Relevance-Driven Scheduling for Multimodal Streaming Perception](perceive_what_matters_relevance-driven_scheduling_for_multimodal_streaming_perce.md)**

:   提出一种面向人机协作的感知调度框架，基于信息增益和计算代价的权衡来选择性激活感知模块（目标检测/姿态估计），在流式感知场景下将计算延迟降低最多 27.52%，同时 MMPose 激活召回提升 72.73%。

**[Pose Priors from Language Models](pose_priors_from_language_models.md)**

:   提出 ProsePose 框架，利用大型多模态模型 (LMM, 如 GPT-4V) 作为接触先验，从图像中提取身体部位接触约束并转化为可优化的损失函数，在无需人工接触标注的情况下改善双人交互和自接触场景的 3D 姿态估计。

**[PoseBH: Prototypical Multi-Dataset Training Beyond Human Pose Estimation](posebh_prototypical_multi-dataset_training_beyond_human_pose_estimation.md)**

:   提出 PoseBH，通过非参数关键点原型（Sinkhorn-Knopp 在线聚类）和跨类型自监督（CSS）实现人/动物/手部等不同骨骼定义数据集的统一训练，在 APT-36K 动物视频数据集上比 ViTPose++ 提升 11.2 AP，证明跨类型知识迁移的有效性。

**[Probabilistic Prompt Distribution Learning for Animal Pose Estimation](probabilistic_prompt_distribution_learning_for_animal_pose_estimation.md)**

:   提出 PPAP（Probabilistic Prompt for Animal Pose），一种基于概率提示分布学习的多物种动物姿态估计方法，通过为每个关键点构建多个可学习属性提示并建模为高斯分布，结合多样性损失和跨模态融合策略，在有监督和零样本设置下均达到 SOTA。

**[Project-Probe-Aggregate: Efficient Fine-Tuning for Group Robustness](project-probe-aggregate_efficient_fine-tuning_for_group_robustness.md)**

:   提出 PPA（Project-Probe-Aggregate）三步方法，通过投影去除类代理信息放大偏差、以组先验校正探测组标签、聚合组权重，仅需不到 0.01% 可训练参数即可在无组标注情况下提升基础模型的群组鲁棒性。

**[Quaffure: Real-Time Quasi-Static Neural Hair Simulation](quaffure_real-time_quasi-static_neural_hair_simulation.md)**

:   Quaffure 提出首个基于物理自监督的实时准静态头发仿真方法，通过将头发形变分解为刚性姿态变换和学习到的修正，使用改进的 Cosserat 弹性能量作为自监督损失训练 CNN 解码器，在消费级硬件上仅需几毫秒即可为不同发型、体型和姿态预测物理合理的头发悬垂效果。

**[Recurrent Feature Mining and Keypoint Mixup Padding for Category-Agnostic Pose Estimation](recurrent_feature_mining_and_keypoint_mixup_padding_for_category-agnostic_pose_e.md)**

:   提出 FMMP 框架，通过基于可变形注意力的循环挖掘细粒度结构感知（FGSA）特征 + 关键点 Mixup 填充策略，在类别无关姿态估计（CAPE）上大幅超越 SOTA（+3.2% PCK@0.05）。

**[Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)**

:   提出 VTON-IQA，一个无参考的虚拟试穿图像质量评估框架，通过大规模人类标注基准 VTON-QBench（62,688 张试穿图 + 431,800 条标注）和 Interleaved Cross-Attention 模块实现与人类感知对齐的图像级质量预测。

**[Remote Photoplethysmography in Real-World and Extreme Lighting Scenarios](remote_photoplethysmography_in_real-world_and_extreme_lighting_scenarios.md)**

:   提出首个面向真实户外极端光照场景的 rPPG 端到端视频 Transformer 模型，通过全局干扰共享、背景参考解耦和生物先验约束，仅基于 RGB 摄像头实现鲁棒的生理信号提取。

**[Removing Reflections from RAW Photos](removing_reflections_from_raw_photos.md)**

:   提出首个基于 RAW 图像的端到端去反射系统：在 XYZ 色彩空间中模拟逼真的反射（含 Fresnel/双反射/WB/曝光），训练 EfficientNet+BiFPN 基础模型分离透射/反射层，再用高斯金字塔上采样器保留高分辨率细节，利用可选的自拍相机上下文图辅助判断，PSNR 30.62dB。

**[RePerformer: Immersive Human-centric Volumetric Videos from Playback to Photoreal Reperformance](reperformer_immersive_human-centric_volumetric_videos_from_playback_to_photoreal.md)**

:   提出 RePerformer，一种基于 3DGS 的体积视频表示方法，通过分层解耦运动高斯和外观高斯、Morton 编码参数化以及语义感知对齐模块，统一实现高保真回放和基于新动作的逼真再表演。

**[Retrieving Semantics from the Deep: an RAG Solution for Gesture Synthesis](retrieving_semantics_from_the_deep_an_rag_solution_for_gesture_synthesis.md)**

:   RAG-Gesture 提出了一种基于检索增强生成（RAG）的手势合成框架，利用显式语言学知识从手势数据库中检索语义相关的示例动作，并通过 DDIM 反演和检索引导在推理时注入扩散模型生成过程，无需训练即可产生语义丰富且自然的共语手势。

**[RGBAvatar: Reduced Gaussian Blendshapes for Online Modeling of Head Avatars](rgbavatar_reduced_gaussian_blendshapes_for_online_modeling_of_head_avatars.md)**

:   RGBAvatar提出"精简高斯混合形状"表示，仅用20个可学习基底即可高效表征可动画头部虚拟形象，配合批量并行渲染和颜色初始化策略，首次实现在线实时（边拍边建）的头部虚拟形象重建。

**[Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents](scalable_video-to-dataset_generation_for_cross-platform_mobile_agents.md)**

:   MONDAY 框架从 YouTube 教学视频自动生成移动端导航数据集——通过 OCR 场景转换检测和 GPT-4o 的 3 步动作识别流程，以人工标注 1/17 的成本（$0.34 vs $5.76/视频）构建了覆盖 iOS/Android 双平台的 313K 标注帧，预训练后 agent 在未见的 Windows Mobile 上提升 18.11%。

**[SemGeoMo: Dynamic Contextual Human Motion Generation with Semantic and Geometric Guidance](semgeomo_dynamic_contextual_human_motion_generation_with_semantic_and_geometric_.md)**

:   提出SemGeoMo，通过LLM自动标注器提供语义引导并结合affordance-level和joint-level的层级几何引导，在两阶段框架中实现动态上下文环境下的高质量人体交互运动生成，同时输出对应文本描述。

**[SGC-Net: Stratified Granular Comparison Network for Open-Vocabulary HOI Detection](sgc-net_stratified_granular_comparison_network_for_open-vocabulary_hoi_detection.md)**

:   提出分层粒度比较网络SGC-Net，通过粒度感知对齐(GSA)模块聚合CLIP多层视觉特征，并利用层级分组比较(HGC)模块借助LLM递归生成区分性描述，解决开放词汇HOI检测中的特征粒度不足和语义混淆问题。

**[Shape My Moves: Text-Driven Shape-Aware Synthesis of Human Motions](shape_my_moves_text-driven_shape-aware_synthesis_of_human_motions.md)**

:   本文提出 ShapeMove 框架，通过 Shape-Aware FSQ-VAE 将连续体型信息注入离散量化的动作 token，并利用预训练语言模型同时预测体型参数和动作 token，实现了首个从自然语言描述端到端生成体型感知动作的方法。

**[ShowMak3r: Compositional TV Show Reconstruction](showmak3r_compositional_tv_show_reconstruction.md)**

:   提出 ShowMak3r，首个从网络视频（芭蕾/跑酷/电影片段）中进行组合式 4D 重建的完整流水线：3DGS 舞台重建 + 深度引导的 SMPL 时空定位 + 跨镜头演员匹配 + 隐式 MLP 面部表情精化，舞台 PSNR 19.65，面部 PSNR 24.34（比 ExAvatar +4.17）。

**[ShowUI: One Vision-Language-Action Model for GUI Visual Agent](showui_one_vision-language-action_model_for_gui_visual_agent.md)**

:   ShowUI 基于 Qwen2-VL-2B，通过 UI 连通图引导的视觉 token 选择减少 33% 冗余 token 并加速 1.4 倍，配合交错式视觉-语言-动作流和精选 256K 训练数据，仅 2B 参数即在零样本 ScreenSpot 上达到 75.1% 的 SOTA 精度。

**[SimMotionEdit: Text-Based Human Motion Editing with Motion Similarity Prediction](simmotionedit_text-based_human_motion_editing_with_motion_similarity_prediction.md)**

:   提出 SimMotionEdit，引入运动相似度预测作为辅助任务，配合 Condition Transformer + Diffusion Transformer 双模块架构，在 MotionFix 数据集上实现文本驱动 3D 人体动作编辑的 SOTA 性能。

**[SocialGesture: Delving into Multi-Person Gesture Understanding](socialgesture_delving_into_multi-person_gesture_understanding.md)**

:   SocialGesture 是首个专注于多人社交场景下指示性手势（pointing/showing/giving/reaching）的大规模数据集，涵盖 9889 个视频片段和 42533 个手势实例，同时提出了时序定位、分类识别和 VQA 三类基准任务，系统揭示了当前模型在多人手势理解上的严重不足。

**[SOLAMI: Social Vision-Language-Action Modeling for Immersive Interaction with 3D Autonomous Characters](solami_social_vision-language-action_modeling_for_immersive_interaction_with_3d_.md)**

:   提出 SOLAMI，首个端到端的社交视觉-语言-动作 (VLA) 建模框架，通过将语音和动作离散化为 token 并基于 decoder-only LLM 统一建模，实现用户与 3D 虚拟角色通过语音和肢体语言的沉浸式实时交互，同时构建了合成多模态社交交互数据集 SynMSI。

**[Sonic: Shifting Focus to Global Audio Perception in Portrait Animation](sonic_shifting_focus_to_global_audio_perception_in_portrait_animation.md)**

:   提出 Sonic 框架，以全局音频感知为核心范式（而非依赖视觉运动帧），通过上下文增强音频学习、运动解耦控制器和时间感知位移融合三个模块，实现了高质量、时间一致的音频驱动肖像动画生成。

**[StickMotion: Generating 3D Human Motions by Drawing a Stickman](stickmotion_generating_3d_human_motions_by_drawing_a_stickman.md)**

:   提出 StickMotion 框架，通过用户手绘的火柴人图作为细粒度动作控制条件，结合文本描述实现全局+局部的 3D 人体动作生成，并设计多条件模块（MCM）高效处理条件组合，节省用户 51.5% 的动作创意表达时间。

**[Stochastic Human Motion Prediction with Memory of Action Transition and Action Characteristic](stochastic_human_motion_prediction_with_memory_of_action_transition_and_action_c.md)**

:   本文针对动作驱动的随机人体运动预测中动作过渡不平滑和动作特征难以学习两大挑战，提出软过渡动作库（STAB）和动作特征库（ACB）两个记忆模块，配合自适应注意力调整（AAA）策略进行特征融合，在 GRAB、NTU、BABEL、HumanAct12 四个数据集上达到 SOTA。

**[Structure-Aware Correspondence Learning for Relative Pose Estimation](structure-aware_correspondence_learning_for_relative_pose_estimation.md)**

:   提出结构感知对应学习方法(SAC-Pose)，通过学习能代表物体结构的关键点，并基于图像间结构感知特征直接回归3D-3D对应关系（无需显式特征匹配），显著提升未见类别物体的相对位姿估计精度。

**[Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach](team_ras_in_10th_abaw_competition_multimodal_valence_and_arousal_estimation_appr.md)**

:   提出结合面部（GRADA+Transformer）、行为描述（Qwen3-VL+Mamba）和音频（WavLM）三模态的连续情感估计方法，通过 Directed Cross-Modal MoE 和 Reliability-Aware Audio-Visual 两种融合策略在 Aff-Wild2 上达到 CCC 0.6576（dev）/ 0.62（test）。

**[TensoFlow: Tensorial Flow-based Sampler for Inverse Rendering](tensoflow_tensorial_flow-based_sampler_for_inverse_rendering.md)**

:   提出 TensoFlow，通过张量化归一化流（Tensorial Normalizing Flow）学习空间-方向感知的重要性采样器，替代逆渲染中固定的预定义采样器（如 cosine-weighted、GGX），大幅降低渲染方程蒙特卡洛估计的方差，提升材质和光照分解质量。

**[Towards High-fidelity 3D Talking Avatar with Personalized Dynamic Texture](towards_high-fidelity_3d_talking_avatar_with_personalized_dynamic_texture.md)**

:   提出TexTalk4D数据集（100分钟扫描级8K动态纹理）和TexTalker框架，首次实现从语音同时生成面部运动和对应的动态纹理（皱纹变化），并通过基于风格锚点(style pivot)的注入策略实现解耦的运动/纹理风格控制。

**[Two by Two: Learning Multi-Task Pairwise Objects Assembly for Generalizable Robot Manipulation](two_by_two_learning_multi-task_pairwise_objects_assembly_for_generalizable_robot.md)**

:   本文提出了 2BY2 数据集——首个大规模日常成对物体组装数据集（18类任务、517对物体），并设计了一种两步式 SE(3) 位姿估计网络，利用等变特征实现多任务物体配对组装，在所有任务上达到 SOTA，并通过真实机器人实验验证了泛化能力。

**[Two is Better than One: Efficient Ensemble Defense for Robust and Compact Models](two_is_better_than_one_efficient_ensemble_defense_for_robust_and_compact_models.md)**

:   提出 EED（Efficient Ensemble Defense），从单个基础模型通过不同剪枝策略（NIS/ERM/ASE/BNSF）生成多个子模型并动态集成——在 80% 稀疏度下 CIFAR-10 PGD 鲁棒准确率 55.71%（接近未压缩基线），推理加速 1.86 倍。

**[UniHOPE: A Unified Approach for Hand-Only and Hand-Object Pose Estimation](unihope_a_unified_approach_for_hand-only_and_hand-object_pose_estimation.md)**

:   提出 UniHOPE，首个统一手部姿态估计（HPE）和手-物姿态估计（HOPE）的框架，通过物体开关器动态控制输出、抓握感知特征融合消除无关物体特征干扰，以及基于扩散模型的去遮挡生成+多层特征增强学习遮挡不变特征。

**[UniPose: A Unified Multimodal Framework for Human Pose Comprehension, Generation and Editing](unipose_a_unified_multimodal_framework_for_human_pose_comprehension_generation_a.md)**

:   UniPose 提出首个统一的多模态框架，利用 LLM 将 3D 人体姿态离散化为 pose tokens 并与文本 tokens 共享词表，通过混合视觉编码器和混合注意力机制实现了跨图像、文本和 3D SMPL 姿态的七个核心姿态任务（理解、生成和编辑）的统一建模。

**[VI3NR: Variance Informed Initialization for Implicit Neural Representations](vi3nr_variance_informed_initialization_for_implicit_neural_representations.md)**

:   推导了适用于任意激活函数的隐式神经表示（INR）初始化方法 VI3NR，将 Xavier/Kaiming 初始化推广到 Gaussian/Sinc 等非标准激活——通过控制前向和反向传播的方差一致性，用一个自由度 $\sigma_p^2$ 同时满足两个方向的稳定性，显著改善 INR 的收敛速度和重建质量。

**[VTON 360: High-Fidelity Virtual Try-On from Any Viewing Direction](vton_360_high-fidelity_virtual_try-on_from_any_viewing_direction.md)**

:   提出 VTON 360，通过将 3D 虚拟试穿重新建模为多视角一致的 2D 虚拟试穿扩展问题，结合伪 3D 姿态表示、多视角空间注意力和多视角 CLIP 嵌入三项技术，实现从任意视角的高保真虚拟试穿。

**[WildAvatar: Learning In-the-Wild 3D Avatars from the Web](wildavatar_learning_in-the-wild_3d_avatars_from_the_web.md)**

:   提出自动化标注管线和过滤协议，从 YouTube 视频中构建了 WildAvatar——一个包含 10,000+ 人体对象的大规模野外 3D avatar 创建数据集，规模比此前数据集大 10 倍以上，并在 EMDB 基准上超越现有 SMPL 标注方法。

**[X-Dyna: Expressive Dynamic Human Image Animation](x-dyna_expressive_dynamic_human_image_animation.md)**

:   X-Dyna提出了一种基于扩散模型的零样本人体图像动画管线，通过轻量级Dynamics-Adapter模块在保持外观一致性的同时生成逼真的人体和场景动态效果，并引入S-Face ControlNet实现身份解耦的面部表情迁移。

**[Zero-Shot Head Swapping in Real-World Scenarios](zero-shot_head_swapping_in_real-world_scenarios.md)**

:   提出HID（Head Injection Diffusion），一种零样本头部替换方法，通过IOMask自动生成上下文感知的编辑掩码实现无缝头身融合，并引入hair injection模块精确迁移发型细节，在包含上半身和多角度面部的真实场景中实现SOTA性能。
