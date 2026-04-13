---
title: >-
  CVPR2025 自动驾驶方向 24篇论文解读
description: >-
  24篇CVPR2025 自动驾驶方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🚗 自动驾驶

**📷 CVPR2025** · 共 **24** 篇

**[3D-Avs Lidar-Based 3D Auto-Vocabulary Segmentation](3d-avs_lidar-based_3d_auto-vocabulary_segmentation.md)**

:   提出3D-AVS，首个针对LiDAR点云的**自动词表分割**方法：无需用户指定目标类别，系统自动从图像和点云中识别场景中存在的语义实体并生成词表，再用开放词表分割器完成逐点语义分割，在nuScenes和ScanNet200上展示了生成精细语义类别的能力。

**[3D Occupancy Prediction With Low-Resolution Queries Via Prototype-Aware View Tra](3d_occupancy_prediction_with_low-resolution_queries_via_prototype-aware_view_tra.md)**

:   提出ProtoOcc，通过**原型感知视角变换**将2D图像聚类原型映射到3D体素查询空间来增强低分辨率体素的上下文信息，配合**多视角占用解码**策略从增强的体素中重建高分辨率3D占用场景，用75%更小的体素分辨率仍能达到与高分辨率方法竞争的性能（Occ3D mIoU 37.80 vs PanoOcc 38.11）。

**[A Neuro-Symbolic Framework Combining Inductive And Deductive Reasoning For Auton](a_neuro-symbolic_framework_combining_inductive_and_deductive_reasoning_for_auton.md)**

:   本文提出首个将 ASP 符号推理决策以可学习嵌入形式直接嵌入端到端规划器轨迹解码的神经-符号框架，用 LLM 动态提取场景规则、Clingo 求解器进行逻辑仲裁、可微 KBM 生成物理可行轨迹并配合神经残差修正，在 nuScenes 上 L₂ 误差 0.57m、碰撞率 0.075%、TPC 0.47m 全面超越 MomAD。

**[A Prediction-As-Perception Framework For 3D Object Detection](a_prediction-as-perception_framework_for_3d_object_detection.md)**

:   PAP 受人脑"预测性感知"启发，将上一帧轨迹预测结果作为当前帧感知模块的 query 输入替代部分随机 query，在 UniAD 上实现 AMOTA 提升 10%（0.359→0.395）、推理速度提升 15%（14→16 FPS）和训练时间缩短 14%。

**[Cawm-Mamba A Unified Model For Infrared-Visible Image Fusion And Compound Advers](cawm-mamba_a_unified_model_for_infrared-visible_image_fusion_and_compound_advers.md)**

:   CAWM-Mamba 首次提出端到端统一处理红外-可见光图像融合与复合恶劣天气（如雾+雨、雨+雪）场景的框架，通过天气感知预处理、跨模态特征交互和小波域频率-SSM 解耦多频退化，在 AWMM-100K 和标准融合数据集上全面超越 SOTA。

**[Certified Human Trajectory Prediction](certified_human_trajectory_prediction.md)**

:   首次将随机平滑（Randomized Smoothing）认证技术引入人类轨迹预测任务，通过mean/median聚合函数和扩散去噪器为轨迹预测模型提供保证性鲁棒性——即无论输入噪声如何扰动（在半径R内），输出始终保持在认证边界内。

**[Climbingcap Multi-Modal Dataset And Method For Rock Climbing In World ](climbingcap_multi-modal_dataset_and_method_for_rock_climbing_in_world_.md)**

:   构建了首个大规模攀岩运动多模态数据集 AscendMotion（412K帧，RGB+LiDAR+IMU），并提出 ClimbingCap 方法通过分离坐标解码、后处理优化和半监督训练，在世界坐标系中精确恢复攀岩者的3D运动。

**[Climbingcap Multi-Modal Dataset And Method For Rock Climbing In World Coordinate](climbingcap_multi-modal_dataset_and_method_for_rock_climbing_in_world_coordinate.md)**

:   提出首个攀岩运动多模态数据集 AscendMotion（412K 帧 RGB+LiDAR+IMU，22 名专业攀岩者，12 面攀岩墙），以及 ClimbingCap 方法通过分离坐标解码、三重后处理优化和半监督训练实现世界坐标系下的 3D 攀岩动作恢复，MPJPE 达 75.45mm。

**[Closed-Loop Supervised Fine-Tuning Of Tokenized Traffic Models](closed-loop_supervised_fine-tuning_of_tokenized_traffic_models.md)**

**[Composing Driving Worlds Through Disentangled Control For Adversarial Scenario G](composing_driving_worlds_through_disentangled_control_for_adversarial_scenario_g.md)**

:   CompoSIA 提出一种基于 Flow Matching DiT 的组合式驾驶视频生成框架，通过解耦结构（3D bbox）、身份（单参考图像）和自车动作（相机轨迹）三类控制信号的注入方式，实现精细独立控制和组合编辑，用于系统化合成对抗性驾驶场景，FVD 提升 17%，碰撞率增加 173%。

**[Cubify Anything Scaling Indoor 3D Object Detection](cubify_anything_scaling_indoor_3d_object_detection.md)**

**[Decoupledgaussian Object-Scene Decoupling For Physics-Based Interaction](decoupledgaussian_object-scene_decoupling_for_physics-based_interaction.md)**

:   将 3DGS 场景中的物体与背景解耦，使物体支持物理仿真（碰撞、抓取等），同时保持场景的高质量渲染

**[Diffusiondrive Truncated Diffusion Model For End-To-End Autonomous Driving](diffusiondrive_truncated_diffusion_model_for_end-to-end_autonomous_driving.md)**

:   本文提出DiffusionDrive，通过截断扩散策略（将去噪步骤从20步减少到2步）和级联扩散解码器，首次将扩散模型成功应用于端到端自动驾驶的实时多模态轨迹规划，在NAVSIM数据集上以88.1 PDMS刷新记录，同时保持45 FPS的实时速度。

**[Distilling Monocular Foundation Model For Fine-Grained Depth Completion](distilling_monocular_foundation_model_for_fine-grained_depth_completion.md)**

:   本文提出DMD3C，一个两阶段知识蒸馏框架，将单目深度基础模型（如Depth Anything V2）的几何知识迁移到深度补全网络，第一阶段通过合成训练数据进行预训练，第二阶段通过尺度-偏移不变损失（SSI Loss）在真实数据上微调，在KITTI深度补全排行榜上取得第一名。

**[Distilling Multi-Modal Large Language Models For Autonomous Driving](distilling_multi-modal_large_language_models_for_autonomous_driving.md)**

:   本文提出DiMA框架，通过联合训练在多模态大语言模型（MLLM）和视觉端到端规划器之间进行知识蒸馏，设计了遮蔽重建、未来预测和场景编辑三种代理任务来丰富场景表示，推理时可丢弃LLM仅用视觉规划器，在nuScenes上实现L2轨迹误差降低37%、碰撞率降低80%。

**[Driving By The Rules A Benchmark For Integrating Traffic Sign Regulations Into V](driving_by_the_rules_a_benchmark_for_integrating_traffic_sign_regulations_into_v.md)**

:   本文首次定义了将交通标志规则集成到在线向量化高精地图的任务，构建了包含10000+视频片段和18000+车道级规则的MapDR数据集，并提出模块化（VLE-MEE）和端到端（RuleVLM）两种基线方案，其中RuleVLM在整体F1指标上达到64.2%。

**[Lr-Sgs Robust Lidar-Reflectance-Guided Salient Gaussian Splatting For Self-Drivi](lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)**

:   LR-SGS 提出基于 LiDAR 反射率引导的显著高斯泼溅方法，引入结构感知的显著高斯表示（由 LiDAR 几何和反射率特征点初始化）和光照不变的反射率通道作为额外约束，在 Waymo 数据集挑战场景（复杂光照）上 PSNR 超越 OmniRe 1.18 dB。

**[M2-Occ Resilient 3D Semantic Occupancy Prediction For Autonomous Driving With In](m2-occ_resilient_3d_semantic_occupancy_prediction_for_autonomous_driving_with_in.md)**

:   M²-Occ 针对多相机输入不完整时的语义占用预测问题，提出多视角掩码重建（MMR）模块利用相邻相机重叠区域恢复缺失视角特征，以及特征记忆模块（FMM）通过类级语义原型精炼不确定体素特征，在缺失后视角设置下 IoU 提升 4.93%。

**[Mapgclr Geospatial Contrastive Learning Of Representations For Online Vectorized](mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)**

:   MapGCLR 提出地理空间对比学习方法，通过强制多次行驶中地理空间重叠区域的 BEV 特征一致性来改善在线矢量化 HD 地图构建的 BEV 编码器，在仅 5% 标注数据下实现 42% 的相对 mAP 提升。

**[O3N Omnidirectional Open-Vocabulary Occupancy Prediction](o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)**

:   O3N 首次提出纯视觉端到端的全向开放词汇占用预测框架，通过极坐标螺旋 Mamba（PsM）建模全向空间连续性、占用代价聚合（OCA）统一几何和语义监督、以及无梯度自然模态对齐（NMA）桥接像素-体素-文本模态间隙，在 QuadOcc 和 Human360Occ 上达到 SOTA。

**[Panoramic Multimodal Semantic Occupancy Prediction For Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)**

:   首个面向四足机器人的全景多模态语义占用预测框架 VoxelHound，提出 PanoMMOcc 数据集（全景 RGB + 热成像 + 偏振 + LiDAR），通过垂直抖动补偿（VJC）和多模态信息提示融合（MIPF）模块达到 23.34% mIoU。

**[Single Pixel Image Classification Using An Ultrafast Digital Light Projector](single_pixel_image_classification_using_an_ultrafast_digital_light_projector.md)**

:   利用 microLED-on-CMOS 超快数字光投影器实现基于单像素成像（SPI）的 MNIST 图像分类，在 1.2 kfps 帧率下达到 >90% 分类精度，完全绕过图像重建直接从时序光信号分类。

**[Spectral-Geometric Neural Fields For Pose-Free Lidar View Synthesis](spectral-geometric_neural_fields_for_pose-free_lidar_view_synthesis.md)**

:   SG-NLF 提出一种无需精确位姿的 LiDAR NeRF 框架，通过混合频谱-几何表征重建平滑几何、置信度感知位姿图实现全局对齐、对抗学习增强跨帧一致性，在低频 LiDAR 场景下重建质量和位姿精度分别超越 SOTA 35.8% 和 68.8%。

**[Vird View-Invariant Representation Through Dual-Axis Transformation For Cross-Vi](vird_view-invariant_representation_through_dual-axis_transformation_for_cross-vi.md)**

:   VIRD 通过双轴变换（极坐标变换 + 上下文增强位置注意力）构建视角不变表征，实现无需方向先验的全向跨视角位姿估计，在 KITTI 上位置和方向误差分别降低 50.7% 和 76.5%。
