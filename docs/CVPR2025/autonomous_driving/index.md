---
title: >-
  CVPR2025 自动驾驶方向 50篇论文解读
description: >-
  50篇CVPR2025 自动驾驶方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🚗 自动驾驶

**📷 CVPR2025** · **50** 篇论文解读

**[3D-AVS: LiDAR-based 3D Auto-Vocabulary Segmentation](3d-avs_lidar-based_3d_auto-vocabulary_segmentation.md)**

:   提出3D-AVS，首个针对LiDAR点云的**自动词表分割**方法：无需用户指定目标类别，系统自动从图像和点云中识别场景中存在的语义实体并生成词表，再用开放词表分割器完成逐点语义分割，在nuScenes和ScanNet200上展示了生成精细语义类别的能力。

**[ProtoOcc: 3D Occupancy Prediction with Low-Resolution Queries via Prototype-aware View Transformation](3d_occupancy_prediction_with_low-resolution_queries_via_prototype-aware_view_tra.md)**

:   提出ProtoOcc，通过**原型感知视角变换**将2D图像聚类原型映射到3D体素查询空间来增强低分辨率体素的上下文信息，配合**多视角占用解码**策略从增强的体素中重建高分辨率3D占用场景，用75%更小的体素分辨率仍能达到与高分辨率方法竞争的性能（Occ3D mIoU 37.80 vs PanoOcc 38.11）。

**[A Neuro-Symbolic Framework Combining Inductive and Deductive Reasoning for Autonomous Driving Planning](a_neuro-symbolic_framework_combining_inductive_and_deductive_reasoning_for_auton.md)**

:   本文提出首个将 ASP 符号推理决策以可学习嵌入形式直接嵌入端到端规划器轨迹解码的神经-符号框架，用 LLM 动态提取场景规则、Clingo 求解器进行逻辑仲裁、可微 KBM 生成物理可行轨迹并配合神经残差修正，在 nuScenes 上 L₂ 误差 0.57m、碰撞率 0.075%、TPC 0.47m 全面超越 MomAD。

**[PAP: A Prediction-as-Perception Framework for 3D Object Detection](a_prediction-as-perception_framework_for_3d_object_detection.md)**

:   PAP 受人脑"预测性感知"启发，将上一帧轨迹预测结果作为当前帧感知模块的 query 输入替代部分随机 query，在 UniAD 上实现 AMOTA 提升 10%（0.359→0.395）、推理速度提升 15%（14→16 FPS）和训练时间缩短 14%。

**[CAWM-Mamba: A Unified Model for Infrared-Visible Image Fusion and Compound Adverse Weather Restoration](cawm-mamba_a_unified_model_for_infrared-visible_image_fusion_and_compound_advers.md)**

:   CAWM-Mamba 首次提出端到端统一处理红外-可见光图像融合与复合恶劣天气（如雾+雨、雨+雪）场景的框架，通过天气感知预处理、跨模态特征交互和小波域频率-SSM 解耦多频退化，在 AWMM-100K 和标准融合数据集上全面超越 SOTA。

**[Certified Human Trajectory Prediction](certified_human_trajectory_prediction.md)**

:   首次将随机平滑（Randomized Smoothing）认证技术引入人类轨迹预测任务，通过mean/median聚合函数和扩散去噪器为轨迹预测模型提供保证性鲁棒性——即无论输入噪声如何扰动（在半径R内），输出始终保持在认证边界内。

**[ClimbingCap: Multi-Modal Dataset and Method for Rock Climbing in World Coordinate](climbingcap_multi-modal_dataset_and_method_for_rock_climbing_in_world_coordinate.md)**

:   提出首个攀岩运动多模态数据集 AscendMotion（412K 帧 RGB+LiDAR+IMU，22 名专业攀岩者，12 面攀岩墙），以及 ClimbingCap 方法通过分离坐标解码、三重后处理优化和半监督训练实现世界坐标系下的 3D 攀岩动作恢复，MPJPE 达 75.45mm。

**[Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models](closed-loop_supervised_fine-tuning_of_tokenized_traffic_models.md)**

**[CompoSIA: Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation](composing_driving_worlds_through_disentangled_control_for_adversarial_scenario_g.md)**

:   CompoSIA 提出一种基于 Flow Matching DiT 的组合式驾驶视频生成框架，通过解耦结构（3D bbox）、身份（单参考图像）和自车动作（相机轨迹）三类控制信号的注入方式，实现精细独立控制和组合编辑，用于系统化合成对抗性驾驶场景，FVD 提升 17%，碰撞率增加 173%。

**[Cubify Anything: Scaling Indoor 3D Object Detection](cubify_anything_scaling_indoor_3d_object_detection.md)**

**[DecoupledGaussian: Object-Scene Decoupling for Physics-Based Interaction](decoupledgaussian_object-scene_decoupling_for_physics-based_interaction.md)**

:   将 3DGS 场景中的物体与背景解耦，使物体支持物理仿真（碰撞、抓取等），同时保持场景的高质量渲染

**[DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving](diffusiondrive_truncated_diffusion_model_for_end-to-end_autonomous_driving.md)**

:   本文提出DiffusionDrive，通过截断扩散策略（将去噪步骤从20步减少到2步）和级联扩散解码器，首次将扩散模型成功应用于端到端自动驾驶的实时多模态轨迹规划，在NAVSIM数据集上以88.1 PDMS刷新记录，同时保持45 FPS的实时速度。

**[Distilling Monocular Foundation Model for Fine-grained Depth Completion](distilling_monocular_foundation_model_for_fine-grained_depth_completion.md)**

:   本文提出DMD3C，一个两阶段知识蒸馏框架，将单目深度基础模型（如Depth Anything V2）的几何知识迁移到深度补全网络，第一阶段通过合成训练数据进行预训练，第二阶段通过尺度-偏移不变损失（SSI Loss）在真实数据上微调，在KITTI深度补全排行榜上取得第一名。

**[Distilling Multi-modal Large Language Models for Autonomous Driving](distilling_multi-modal_large_language_models_for_autonomous_driving.md)**

:   本文提出DiMA框架，通过联合训练在多模态大语言模型（MLLM）和视觉端到端规划器之间进行知识蒸馏，设计了遮蔽重建、未来预测和场景编辑三种代理任务来丰富场景表示，推理时可丢弃LLM仅用视觉规划器，在nuScenes上实现L2轨迹误差降低37%、碰撞率降低80%。

**[Driving by the Rules: A Benchmark for Integrating Traffic Sign Regulations into Vectorized HD Map](driving_by_the_rules_a_benchmark_for_integrating_traffic_sign_regulations_into_v.md)**

:   本文首次定义了将交通标志规则集成到在线向量化高精地图的任务，构建了包含10000+视频片段和18000+车道级规则的MapDR数据集，并提出模块化（VLE-MEE）和端到端（RuleVLM）两种基线方案，其中RuleVLM在整体F1指标上达到64.2%。

**[DrivingSphere: Building a High-fidelity 4D World for Closed-loop Simulation](drivingsphere_building_a_high-fidelity_4d_world_for_closed-loop_simulation.md)**

:   构建基于 4D 占用网格的高保真闭环驾驶仿真框架——用 OccDreamer 从 BEV 生成静态场景占用、用 Actor Bank 组合动态物体、用 VideoDreamer 从占用条件生成多视角视频，FVD 降低 44%，物体检测 mAP 提升 33%。

**[EV-3DOD: Pushing the Temporal Boundaries of 3D Object Detection with Event Cameras](ev-3dod_pushing_the_temporal_boundaries_of_3d_object_detection_with_event_camera.md)**

:   首次将事件相机引入3D目标检测，提出 Virtual 3D Event Fusion（V3D-EF）将异步事件投影到3D体素空间与LiDAR特征融合，在帧间"盲区时间"内以100FPS持续检测物体，填补了传感器帧间~100ms的感知空白。

**[EVolSplat: Efficient Volume-based Gaussian Splatting for Urban View Synthesis](evolsplat_efficient_volume-based_gaussian_splatting_for_urban_view_synthesis.md)**

:   提出 EVolSplat，一个基于稀疏3D卷积的前馈城市场景3D高斯泼溅方法，通过全局统一体素预测高斯参数（而非像素对齐），结合遮挡感知的基于图像的渲染（IBR）着色，在 KITTI-360 上达到 23.26dB PSNR / 83.81 FPS。

**[Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation](exploring_scene_affinity_for_semi-supervised_lidar_semantic_segmentation.md)**

:   提出 AIScene 框架利用场景内一致性（点擦除策略）和场景间关联（MixPatch + InsFill 跨场景增强），在仅 1% 标注的 SemanticKITTI 上将半监督 LiDAR 分割提升 1.9 mIoU。

**[ForestLPR: LiDAR Place Recognition in Forests Attentioning Multiple BEV Density Images](forestlpr_lidar_place_recognition_in_forests_attentioning_multiple_bev_density_i.md)**

:   本文提出ForestLPR，通过将点云在不同高度切片生成多张BEV密度图，利用ViT提取局部特征后经multi-BEV交互模块自适应关注不同高度的判别性特征，实现森林环境下鲁棒的LiDAR位置识别，在多个数据集上大幅超越SOTA。

**[FreeSim: Toward Free-Viewpoint Camera Simulation in Driving Scenes](freesim_toward_free-viewpoint_camera_simulation_in_driving_scenes.md)**

:   本文提出FreeSim，通过将挑战性的偏离轨迹新视角生成问题重新表述为生成式图像增强问题，配合piece-wise高斯重建的训练数据构造和渐进式视角扩展策略，首次实现了驾驶场景中超过3米横向偏移的高质量自由视角渲染。

**[g3D-LF: Generalizable 3D-Language Feature Fields for Embodied Tasks](g3d-lf_generalizable_3d-language_feature_fields_for_embodied_tasks.md)**

:   本文提出g3D-LF，通过在约5K室内3D场景和近100万语言描述上进行多级对比学习预训练，构建了可泛化到未知环境的3D-语言特征场，在VLN（单目/全景）、零样本物体导航和情境问答四种具身任务上均取得SOTA或接近SOTA表现。

**[GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction](gaussianformer-2_probabilistic_gaussian_superposition_for_efficient_3d_occupancy.md)**

:   本文提出GaussianFormer-2，从概率视角重新诠释3D语义高斯：每个高斯表示其邻域被占用的概率分布，通过概率乘法聚合几何预测、高斯混合模型归一化语义预测，彻底消除了高斯描述空区域和相互冗余重叠的问题，以仅8.9%的高斯数量达到SOTA。

**[GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction](gaussianworld_gaussian_world_model_for_streaming_3d_occupancy_prediction.md)**

:   提出 GaussianWorld，将 3D 占用预测重新定义为以当前传感器输入为条件的 4D 占用预测问题，通过将场景演化分解为自车运动对齐、动态物体运动和新区域补全三个因素，在 3D 高斯空间中用世界模型显式建模场景变化，在 nuScenes 上不增加额外计算量的前提下将单帧方法的 mIoU 提升超过 2%。

**[Generating Multimodal Driving Scenes via Next-Scene Prediction](generating_multimodal_driving_scenes_via_next-scene_prediction.md)**

:   提出 UMGen，一个统一的多模态驾驶场景生成框架，将自车动作、地图、交通参与者和图像四种模态进行 token 化，通过帧间时序自回归（TAR）和帧内有序自回归（OAR）两阶段策略逐场景生成，同时引入动作感知地图对齐（AMA）模块保持自车运动与地图的一致性，可自主生成长达 60 秒的连贯驾驶序列。

**[Generative Gaussian Splatting for Unbounded 3D City Generation](generative_gaussian_splatting_for_unbounded_3d_city_generation.md)**

:   提出 GaussianCity，首个将 3D 高斯溅射应用于无界 3D 城市生成的框架，通过引入 BEV-Point 紧凑中间表示使显存占用与场景规模解耦（保持恒定），并设计 Point Serializer 将无序 BEV 点转为有序序列以捕获结构和上下文特征，在无人机视角和街景视角的城市生成中达到 SOTA，渲染速度比 CityDreamer（基于 NeRF）快 60 倍。

**[GLane3D: Detecting Lanes with Graph of 3D Keypoints](glane3d_detecting_lanes_with_graph_of_3d_keypoints.md)**

:   提出GLane3D，一种基于关键点的3D车道线检测方法，通过检测车道关键点并预测它们之间的有向连接构建图结构，利用PointNMS去除冗余关键点提议后用Dijkstra最短路径提取车道实例，在OpenLane和Apollo数据集上达到SOTA的F1分数且泛化能力优越。

**[Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation](helvipad_a_real-world_dataset_for_omnidirectional_stereo_depth_estimation.md)**

:   提出Helvipad——首个用于全景立体深度估计的真实世界数据集（40K帧、上下双360°相机+LiDAR），并引入极角输入和环形填充两个适配策略来改进立体匹配模型处理等距矩形投影图像，所提360-IGEV-Stereo在所有指标上达到最佳。

**[InteractionMap: Improving Online Vectorized HDMap Construction with Interaction](interactionmap_improving_online_vectorized_hdmap_construction_with_interaction.md)**

:   本文提出InteractionMap，通过点级和实例级关系嵌入、关键帧分层时序融合和几何感知分类-定位对齐三个模块，全面增强在线矢量化HD地图构建中的信息交互，在nuScenes (mAP 71.8) 和Argoverse2 (mAP 74.7) 上均取得SOTA。

**[Learning to Detect Objects from Multi-Agent LiDAR Scans without Manual Labels](learning_to_detect_objects_from_multi-agent_lidar_scans_without_manual_labels.md)**

:   提出 DOtA（Detect Objects from Multi-Agent），一种无需人工标注的多智能体 LiDAR 3D 目标检测方法：利用协作智能体内部共享的自车位姿和车身形状完成检测器初始化，再通过智能体间互补观测进行多尺度编码，解码出高低质量伪标签分别指导特征学习，实现完全无监督的高质量 3D 目标检测。

**[LiDAR-RT: Gaussian-based Ray Tracing for Dynamic LiDAR Re-Simulation](lidar-rt_gaussian-based_ray_tracing_for_dynamic_lidar_re-simulation.md)**

:   本文提出LiDAR-RT，将3D高斯原语与NVIDIA OptiX硬件加速光线追踪相结合，首次实现动态驾驶场景下实时且物理精确的LiDAR重新仿真，渲染速度达30 FPS，训练仅需2小时，远超NeRF方案的0.2 FPS和15小时。

**[LightLoc: Learning Outdoor LiDAR Localization at Light Speed](lightloc_learning_outdoor_lidar_localization_at_light_speed.md)**

:   本文提出LightLoc，通过样本分类引导 (SCG) 减少视觉相似区域的回归歧义，以及冗余样本下采样 (RSD) 剔除已学好的帧，实现大规模室外LiDAR定位训练50倍加速（1小时 vs 2天），同时达到0.83m SOTA位置精度。

**[LiMoE: Mixture of LiDAR Representation Learners from Automotive Scenes](limoe_mixture_of_lidar_representation_learners_from_automotive_scenes.md)**

:   提出 LiMoE，通过混合专家（MoE）机制融合三种互补的 LiDAR 表示（距离图/稀疏体素/原始点云），三阶段训练（图像→LiDAR 预训练 → 对比混合学习 → 语义混合监督），在 nuScenes 分割上达到 51.4% mIoU，跨域泛化到 7 个数据集。

**[LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)**

:   LR-SGS 提出基于 LiDAR 反射率引导的显著高斯泼溅方法，引入结构感知的显著高斯表示（由 LiDAR 几何和反射率特征点初始化）和光照不变的反射率通道作为额外约束，在 Waymo 数据集挑战场景（复杂光照）上 PSNR 超越 OmniRe 1.18 dB。

**[M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](m2-occ_resilient_3d_semantic_occupancy_prediction_for_autonomous_driving_with_in.md)**

:   M²-Occ 针对多相机输入不完整时的语义占用预测问题，提出多视角掩码重建（MMR）模块利用相邻相机重叠区域恢复缺失视角特征，以及特征记忆模块（FMM）通过类级语义原型精炼不确定体素特征，在缺失后视角设置下 IoU 提升 4.93%。

**[MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)**

:   MapGCLR 提出地理空间对比学习方法，通过强制多次行驶中地理空间重叠区域的 BEV 特征一致性来改善在线矢量化 HD 地图构建的 BEV 编码器，在仅 5% 标注数据下实现 42% 的相对 mAP 提升。

**[Modeling Thousands of Human Annotators for Generalizable Text-to-Image Person Re-identification](modeling_thousands_of_human_annotators_for_generalizable_text-to-image_person_re.md)**

:   提出 Human Annotator Modeling (HAM) 方法，通过对人类标注描述进行风格特征提取和聚类，用可学习提示让 MLLM 模拟数千种人类标注风格，再结合 Uniform Prototype Sampling (UPS) 进一步增加风格多样性，自动构建大规模高质量文本-图像行人 ReID 数据集，在多个基准上大幅提升了 ReID 模型的泛化能力。

**[ModeSeq: Taming Sparse Multimodal Motion Prediction with Sequential Mode Modeling](modeseq_taming_sparse_multimodal_motion_prediction_with_sequential_mode_modeling.md)**

:   提出 ModeSeq——一种将轨迹模式建模为序列的全新范式，通过逐步解码多模态轨迹（而非一次性并行解码）来显式捕捉模式间关联，并配合 Early-Match-Take-All (EMTA) 训练策略，在不依赖密集模式预测或启发式后处理的前提下，显著提升了稀疏多模态运动预测的轨迹多样性和置信度校准。

**[Multi-modal Knowledge Distillation-based Human Trajectory Forecasting](multi-modal_knowledge_distillation-based_human_trajectory_forecasting.md)**

:   本文提出首个用于行人轨迹预测的多模态知识蒸馏框架——用轨迹+人体姿态+文本描述训练全模态教师模型，将其知识蒸馏到仅用轨迹或轨迹+姿态的学生模型，在JRDB/SIT/ETH-UCY三个数据集上最高提升约13%预测精度。

**[O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)**

:   O3N 首次提出纯视觉端到端的全向开放词汇占用预测框架，通过极坐标螺旋 Mamba（PsM）建模全向空间连续性、占用代价聚合（OCA）统一几何和语义监督、以及无梯度自然模态对齐（NMA）桥接像素-体素-文本模态间隙，在 QuadOcc 和 Human360Occ 上达到 SOTA。

**[Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)**

:   首个面向四足机器人的全景多模态语义占用预测框架 VoxelHound，提出 PanoMMOcc 数据集（全景 RGB + 热成像 + 偏振 + LiDAR），通过垂直抖动补偿（VJC）和多模态信息提示融合（MIPF）模块达到 23.34% mIoU。

**[Point-to-Region Loss for Semi-Supervised Point-Based Crowd Counting](point-to-region_loss_for_semi-supervised_point-based_crowd_counting.md)**

:   发现半监督人群计数中点到点（P2P）匹配导致模型对未标注数据过度激活（通过 PSAM 梯度诊断可视化），提出点到区域（P2R）匹配——将每个 GT/伪标签点扩展为局部区域并传播置信度，在 ShanghaiTech-A 5% 标注下 MAE 69.9（前 SOTA 83.7），且比 P2P 快 68 倍。

**[GDFusion: Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction](rethinking_temporal_fusion_with_a_unified_gradient_descent_view_for_3d_semantic_.md)**

:   提出 GDFusion，将 RNN 重新解释为梯度下降步骤，统一三种时序线索（场景级/运动/几何）的融合方式，在 Occ3D 上比非时序基线提升 1.4-4.8% mIoU 同时减少 27-72% 推理内存，比 SOLOFusion 等多帧方法更高效。

**[Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation Environments](scenario_dreamer_vectorized_latent_diffusion_for_generating_driving_simulation_e.md)**

:   提出 Scenario Dreamer，将自动驾驶仿真环境生成分解为三部分：向量化潜扩散模型生成初始场景（车道+智能体）、回报条件的 CtRL-Sim 生成闭环行为、场景修补实现无界环境扩展，在 nuPlan 上 Frechet Distance 0.67（基线 SLEDGE 1.44），生成仅需 0.16 秒。

**[SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model](scenediffuser_city-scale_traffic_simulation_via_a_generative_world_model.md)**

:   提出 SceneDiffuser++，一个端到端的城市级交通仿真扩散模型，通过软裁剪（soft clipping）处理稀疏张量中的智能体出入场问题，实现 60 秒以上的行程级（trip-level）交通仿真，在 WOMD-XLMap 上达到 0.2423 综合 JS 散度。

**[Single Pixel Image Classification using an Ultrafast Digital Light Projector](single_pixel_image_classification_using_an_ultrafast_digital_light_projector.md)**

:   利用 microLED-on-CMOS 超快数字光投影器实现基于单像素成像（SPI）的 MNIST 图像分类，在 1.2 kfps 帧率下达到 >90% 分类精度，完全绕过图像重建直接从时序光信号分类。

**[SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving](solve_synergy_of_language-vision_and_end-to-end_networks_for_autonomous_driving.md)**

:   提出 SOLVE，通过共享 SQ-Former 视觉编码器实现 VLM 和端到端驾驶模型的特征级协同，用 Trajectory Chain-of-Thought（T-CoT）将 VLM 的长程轨迹作为 E2E 模型的初始化先验，在 nuScenes 上达到 0.28m 平均 L2 误差 SOTA。

**[Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](spectral-geometric_neural_fields_for_pose-free_lidar_view_synthesis.md)**

:   SG-NLF 提出一种无需精确位姿的 LiDAR NeRF 框架，通过混合频谱-几何表征重建平滑几何、置信度感知位姿图实现全局对齐、对抗学习增强跨帧一致性，在低频 LiDAR 场景下重建质量和位姿精度分别超越 SOTA 35.8% 和 68.8%。

**[UniScene: Unified Occupancy-centric Driving Scene Generation](uniscene_unified_occupancy-centric_driving_scene_generation.md)**

:   提出 UniScene，以占用网格为统一中间表示的两阶段驾驶场景生成：Occupancy Diffusion Transformer 从 BEV 布局生成语义占用，再通过高斯泼溅联合渲染语义+深度图条件化双扩散模型生成视频和 LiDAR，FVD 71.94（前 SOTA Drive-WM 122.70），下游数据增强提升 3D 检测 mAP 3.62%。

**[VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation](vird_view-invariant_representation_through_dual-axis_transformation_for_cross-vi.md)**

:   VIRD 通过双轴变换（极坐标变换 + 上下文增强位置注意力）构建视角不变表征，实现无需方向先验的全向跨视角位姿估计，在 KITTI 上位置和方向误差分别降低 50.7% 和 76.5%。
