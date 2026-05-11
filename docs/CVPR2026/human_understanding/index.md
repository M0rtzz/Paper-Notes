---
title: >-
  CVPR2026 人体理解方向56篇论文解读
description: >-
  56篇CVPR2026的人体理解方向论文解读，涵盖人体姿态、人脸/视线、虚拟人、多模态、对齐/RLHF、动态场景等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧑 人体理解

**📷 CVPR2026** · **56** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (1)](../../ACL2026/human_understanding/index.md) · [🔬 ICLR2026 (8)](../../ICLR2026/human_understanding/index.md) · [🤖 AAAI2026 (16)](../../AAAI2026/human_understanding/index.md) · [🧠 NeurIPS2025 (19)](../../NeurIPS2025/human_understanding/index.md) · [📹 ICCV2025 (47)](../../ICCV2025/human_understanding/index.md) · [🧪 ICML2025 (2)](../../ICML2025/human_understanding/index.md)

🔥 **高频主题：** 人体姿态 ×7 · 人脸/视线 ×5 · 虚拟人 ×4 · 多模态 ×4 · 对齐/RLHF ×3

**[A Two-Stage Dual-Modality Model for Facial Expression Recognition](a_two_stage_dual_modality_model_for_facial_expression_recognition.md)**

:   提出两阶段双模态面部表情识别框架：Stage I 通过填充感知增强和训练期 MoE 头在外部数据集上适配 DINOv2 编码器；Stage II 通过多尺度面部裁剪、Wav2Vec 2.0 音频特征提取和门控融合实现帧级音视觉表情分类，在 ABAW 2026 竞赛中取得 0.5368 Macro-F1。

**[All in One: Unifying Deepfake Detection, Tampering Localization, and Source Tracing with a Robust Landmark-Identity Watermark](all_in_one_unifying_deepfake_detection_tampering_localization_and_source_tracing.md)**

:   提出 LIDMark，首个将 deepfake 检测、篡改区域定位和源追踪统一到单一主动取证框架中的方法——通过嵌入 152 维 Landmark-Identity 水印（136D 面部关键点 + 16D 源 ID），利用内在/外在一致性实现三合一取证，PSNR/SSIM 和检测精度均超越现有方法。

**[AVATAR: Reinforcement Learning to See, Hear, and Reason Over Video](avatar_reinforcement_learning_to_see_hear_and_reason_over_video.md)**

:   提出AVATAR框架，通过离线策略训练架构（分层回放缓冲区）和时间优势塑造(TAS)策略解决GRPO在多模态视频推理中的数据低效、优势消失和均匀信用分配三大问题，在音视频理解基准上显著超越标准GRPO（OmniBench +3.7，样本效率提升5倍）。

**[Beyond the Fold: Quantifying Split-Level Noise and the Case for Leave-One-Dataset-Out AU Evaluation](beyond_the_fold_quantifying_split-level_noise_and_the_case_for_leave-one-dataset.md)**

:   揭示面部AU检测中被试独立交叉验证本身引入±0.065 F1的随机噪声（noise floor），许多声称的SOTA提升落入此噪声带内不可区分，并提出Leave-One-Dataset-Out（LODO）协议作为更稳定可靠的替代评估方案。

**[BROTHER: Behavioral Recognition Optimized Through Heterogeneous Ensemble Regularization for Ambivalence and Hesitancy](brother_behavioral_recognition_optimized_through_heterogeneous_ensemble_regulari.md)**

:   提出一个高度正则化的多模态融合管线，通过视觉(SigLip2)、音频(HuBERT)、文本(F2LLM)及统计特征四模态的异质分类器委员会，结合带训练-验证差距惩罚的 PSO 硬投票集成，实现自然场景下矛盾与犹豫（A/H）行为的鲁棒视频级识别，在 ABAW10 测试集上取得 Macro F1 = 0.7465。

**[CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation](cigpose_causal_intervention_graph_neural_network_for_whole-body_pose_estimation.md)**

:   提出因果干预图姿态估计框架 CIGPose，通过结构因果模型识别视觉上下文混杂因素，利用预测不确定性定位受混杂影响的关键点并用学习得到的上下文无关规范嵌入替换，再经层次图神经网络建模骨骼解剖约束，在 COCO-WholeBody 上达到 67.0% AP 的新 SOTA。

**[COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation](cog_confidence-aware_optimal_geometric_correspondence_for_unsupervised_single-re.md)**

:   提出 COG 框架，将跨视图对应关系建模为置信度感知的最优传输(OT)问题，通过预测逐点置信度作为传输边际约束来抑制非重叠区域和离群点，实现无监督条件下媲美有监督方法的单参考图像新物体6DoF位姿估计。

**[E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation](e-3dpsm_a_state_machine_for_event-based_egocentric_3d_human_pose_estimation.md)**

:   提出 E-3DPSM，一种基于事件相机的自我中心 3D 人体姿态状态机，将姿态估计建模为连续时间状态演化过程，通过双向 SSM 时序建模和可学习的卡尔曼式融合模块融合直接预测与增量预测，实现 80Hz 实时推理，MPJPE 降低 19%、时序稳定性提升 2.7 倍。

**[Editing Physiological Signals in Videos Using Latent Representations](editing_physiological_signals_in_videos_using_latent_representations.md)**

:   提出PhysioLatent框架，将输入面部视频编码到3D VAE潜空间，与目标心率CLIP文本嵌入融合，通过AdaLN增强的时空融合层捕捉rPPG时间相干性，结合FiLM调制解码器和微调输出层实现精确心率修改，在保持PSNR 38.96dB/SSIM 0.98的视觉质量下达到10 bpm MAE的心率调制精度。

**[Efficient Onboard Spacecraft Pose Estimation with Event Cameras and Neuromorphic Hardware](efficient_onboard_spacecraft_pose_estimation_with_event_cameras_and_neuromorphic_hardware.md)**

:   首次在 BrainChip Akida 神经形态硬件上实现端到端航天器 6-DoF 位姿估计，探索事件相机表示和量化感知训练在低功耗星载部署中的精度-效率权衡。

**[EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR](egoposeformer_v2_accurate_egocentric_human_motion_estimation_for_arvr.md)**

:   提出 EgoPoseFormer v2 (EPFv2)，通过端到端 Transformer 架构（单一全局查询 + 因果时序注意力 + 条件多视图交叉注意力）和基于不确定性蒸馏的自动标注系统，在 EgoBody3M 基准上以 0.8ms GPU 延迟实现了自我中心 3D 人体运动估计的 SOTA 精度（MPJPE 4.02cm，比前作提升 15-22%）。

**[Face Time Traveller: Travel Through Ages Without Losing Identity](face_time_traveller_travel_through_ages_without_losing_identity.md)**

:   提出 FaceTT 框架，通过面部属性感知提示词精炼、角度反演和自适应注意力控制三大模块，实现高保真、身份一致的人脸年龄变换，在多个基准上超越现有方法。

**[FlexAvatar: Learning Complete 3D Head Avatars with Partial Supervision](flexavatar_learning_complete_3d_head_avatars_with_partial_supervision.md)**

:   提出 FlexAvatar，通过引入可学习的"偏置吸收器"（bias sinks）token 统一单目和多视角数据训练，解决了驱动信号与目标视角的纠缠问题，从单张图像生成完整、高质量、可动画的 3D 头部化身。

**[A2P: From 2D Alignment to 3D Plausibility for Occlusion-Robust Two-Hand Reconstruction](from_2d_alignment_to_3d_plausibility_unifying_hete.md)**

:   解耦双手重建为 2D 结构对齐 + 3D 空间交互对齐：Stage 1 用 Fusion Alignment Encoder 隐式蒸馏 Sapiens 的关键点/分割/深度三种 2D 先验（推理时免基础模型，56fps），Stage 2 用穿透感知扩散模型 + 碰撞梯度引导将穿透姿态映射到物理合理配置——InterHand2.6M 上 MPJPE 降至 5.36mm（超 SOTA 4DHands 2.13mm），穿透体积降 7 倍。

**[FSMC-Pose: Frequency and Spatial Fusion with Multiscale Self-calibration for Cattle Mounting Pose Estimation](fsmc-pose_frequency_and_spatial_fusion_with_multiscale_selfcalibration_for_cattle.md)**

:   FSMC-Pose 提出面向牛只爬跨(mounting)姿态估计的轻量级 top-down 框架，包含频率-空间融合骨干网络 CattleMountNet（通过 SFEBlock 的小波变换+高斯滤波分离前景-背景，RABlock 的多尺度扩张卷积聚合上下文）和多尺度自校准头 SC2Head（空间-通道共校准 + 自校准分支纠正结构偏移），同时构建了首个牛只爬跨数据集 MOUNT-Cattle，在复杂群养环境中以极低计算成本(4.41 GFLOPS, 2.698M 参数)达到 89% AP。

**[FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition](fusionagent_a_multimodal_agent_with_dynamic_model_selection_for_human_recognitio.md)**

:   本文提出 FusionAgent，一个基于多模态大语言模型（MLLM）的智能体框架，用于全身生物特征识别中的动态样本级模型选择——将每个专家模型（人脸识别/步态识别/行人重识别）封装为工具，通过强化微调（RFT）让 agent 学会根据每个测试样本的特征自适应选择最优模型组合，配合新提出的 ACT 分数融合策略，显著超越现有 SOTA 融合方法。

**[HandDreamer: Zero-Shot Text to 3D Hand Model Generation](handdreamer_zero_shot_text_to_3d_hand_model_generation.md)**

:   提出 HandDreamer，首个从文本提示零样本生成 3D 手部模型的方法，通过 MANO 初始化、骨架引导扩散和校正手形损失解决 SDS 中的视图不一致和几何畸变问题。

**[HandX: Scaling Bimanual Motion and Interaction Generation](handx_scaling_bimanual_motion_and_interaction_generation.md)**

:   构建了 HandX——一个统一的双手运动生成基础设施（包含 54.2 小时运动数据 + 48.5 万条细粒度文本标注），提出解耦式自动标注策略（运动学特征提取 + LLM 推理生成描述），并基准测试了扩散和自回归两种生成范式，展示了明确的数据和模型 scaling 趋势。

**[How to Take a Memorable Picture? Empowering Users with Actionable Feedback](how_to_take_a_memorable_picture_empowering_users_with_actionable_feedback.md)**

:   定义了记忆性反馈（MemFeed）新任务，提出 MemCoach——一种 training-free 的 MLLM 激活导向方法，通过教师-学生策略将记忆性感知知识注入模型激活空间，使 MLLM 能生成提升照片记忆性的自然语言可操作建议。

**[HUM4D: A Dataset and Evaluation for Complex 4D Markerless Human Motion Capture](hum4d_markerless_motion_capture.md)**

:   提出 HUM4D 数据集，包含复杂单人和多人运动场景（快速运动、遮挡、身份交换），提供同步多视角 RGB/RGB-D 序列、精确 Vicon 标记运动捕捉真值和 SMPL/SMPL-X 参数，基准测试揭示 SOTA 无标记方法在真实条件下的显著性能退化。

**[HumanOrbit: 3D Human Reconstruction as 360° Orbit Generation](humanorbit_3d_human_reconstruction_as_360_orbit_generation.md)**

:   将单图3D人体重建转化为360°轨道视频生成问题，用仅500个3D扫描数据LoRA微调视频扩散模型（Wan 2.1）生成81帧环绕视频，再通过VGGT+Mesh Carving重建高质量纹理网格，无需位姿标注且在多视图一致性和身份保持上超越现有方法。

**[IDperturb: Enhancing Variation in Synthetic Face Generation via Angular Perturbations](idperturb_enhancing_variation_in_synthetic_face_generation_via_angular_perturbat.md)**

:   提出 IDperturb，一种在单位超球面上对身份嵌入进行角度扰动的几何采样策略，无需修改生成模型即可显著增强合成人脸数据集的类内多样性，提升下游人脸识别性能。

**[LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference](lamogen_language_to_motion_generation_through_llm-guided_symbolic_inference.md)**

:   提出 LabanLite 符号动作表示和 LaMoGen 框架，首次让 LLM 通过可解释的 Laban 符号推理自主组合动作序列，在时序精度和可控性上超越传统文本-动作联合嵌入方法。

**[LaScA: Language-Conditioned Scalable Modelling of Affective Dynamics](lasca_language-conditioned_scalable_modelling_of_affective_dynamics.md)**

:   提出 LaScA 框架，利用大语言模型生成确定性语义词典为手工制作的面部和声学特征提供语义先验，通过冻结的句子编码器生成语义嵌入并与原始特征融合，在 Aff-Wild2 和 SEWA 数据集上的情感变化预测中一致性地超越纯特征基线，并在一致性、效率和可解释性上与端到端深度模型持平或更优。

**[LASER: Layer-wise Scale Alignment for Training-Free Streaming 4D Reconstruction](laser_layer-wise_scale_alignment_for_training-free_streaming_4d_reconstruction.md)**

:   提出 LASER，一个无需重训练的框架，通过层级深度尺度对齐（Layer-wise Scale Alignment）将离线前馈重建模型（如 VGGT、π³）转换为流式系统，在 RTX A6000 上以 14 FPS、6GB 峰值显存实现千米级视频的实时流式 4D 重建。

**[LCA: Large-scale Codec Avatars - The Unreasonable Effectiveness of Large-scale Avatar Pretraining](lca_large-scale_codec_avatars_the_unreasonable_effectiveness_of_large-scale_avata.md)**

:   LCA 首次将大规模预训练/后训练范式应用于 3D 头像建模：在 100 万野外视频上预训练学习广泛的外观和几何先验，再在高质量多视图工作室数据上后训练增强精细表情和保真度，打破了泛化性与保真度的固有矛盾。

**[MatchED: Crisp Edge Detection Using End-to-End, Matching-based Supervision](matched_crisp_edge_detection_using_end-to-end_matching-based_supervision.md)**

:   MatchED 提出一种轻量（约21K参数）plug-and-play 模块，通过在训练时对预测边缘和 GT 边缘进行基于空间距离+置信度的 one-to-one 二部匹配来生成 crisp（单像素宽）边缘图，可附加到任何边缘检测器端到端训练，首次在不依赖 NMS+thinning 后处理的情况下匹配或超越标准后处理方法。

**[Miburi: Towards Expressive Interactive Gesture Synthesis](miburi_towards_expressive_interactive_gesture_synthesis.md)**

:   提出 Miburi，首个在线因果框架，通过直接利用语音-文本大模型 Moshi 的内部 token 流和二维因果 Transformer，实现实时同步的全身手势与面部表情生成。

**[MMGait: Towards Multi-Modal Gait Recognition](mmgait_multi_modal_gait_recognition.md)**

:   MMGait 构建了目前最全面的多模态步态识别基准数据集（5 种传感器、12 种模态、725 人、334K 序列），并提出全模态步态识别新任务和统一基线模型 OmniGait。

**[Mobile-VTON: High-Fidelity On-Device Virtual Try-On](mobile_vton_ondevice_virtual_tryon.md)**

:   首个全离线移动端扩散式虚拟试穿框架，基于TeacherNet-GarmentNet-TryonNet (TGT)架构，通过特征引导对抗蒸馏(FGA)将SD3.5 Large的能力迁移到415M参数的轻量学生网络，在VITON-HD和DressCode上以1024×768分辨率匹配甚至超越服务器端基线，端到端推理时间约80秒（小米17 Pro Max）。

**[MoLingo: Motion-Language Alignment for Text-to-Human Motion Generation](molingo_motion-language_alignment_for_text-to-motion_generation.md)**

:   MoLingo 通过语义对齐的运动自编码器（SAE）和多 token 交叉注意力文本条件注入，在连续潜空间上执行 masked 自回归 rectified flow，在文本到人体动作生成任务上取得了 FID、R-Precision 和用户研究的全面 SOTA。

**[OMG-Bench: A New Challenging Benchmark for Skeleton-based Online Micro Hand Gesture Recognition](omg-bench_a_new_challenging_benchmark_for_skeleton-based_online_micro_hand_gestu.md)**

:   本文构建了首个大规模公开的基于骨骼数据的在线微手势识别基准OMG-Bench（40类、13948个实例），并提出HMATr框架，通过层次化记忆库和位置感知查询实现检测-分类的端到端统一，在检测率上超越SOTA方法7.6%。

**[OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery](onlinehmr_video-based_online_world-grounded_human_mesh_recovery.md)**

:   提出 OnlineHMR，首个同时满足系统因果性、忠实性、时序一致性和高效性四项准则的在线世界坐标人体网格恢复框架，通过滑动窗口因果学习 + KV 缓存推理实现流式相机坐标 HMR，结合以人为中心的增量 SLAM 和 EMA 轨迹校正实现在线全局定位。

**[OpenFS: Multi-Hand-Capable Fingerspelling Recognition with Implicit Signing-Hand Detection and Frame-Wise Letter-Conditioned Synthesis](openfs_multi-hand-capable_fingerspelling_recognition_with_implicit_signing-hand_.md)**

:   提出 OpenFS 框架，通过双层位置编码 + 签名手聚焦损失 + 单调对齐损失实现隐式签名手检测的多手指拼识别，并设计帧级字母条件扩散生成器合成 OOV 数据，在 ChicagoFSWild/ChicagoFSWildPlus/FSNeo 三个基准上取得 SOTA，推理速度比 PoseNet 快 100 倍以上。

**[ParTY: Part-Guidance for Expressive Text-to-Motion Synthesis](party_part-guidance_for_expressive_text-to-motion_synthesis.md)**

:   提出 ParTY 框架，通过部位引导网络（Part-Guided Network）和部位感知文本对齐（Part-aware Text Grounding），在保持全身动作连贯性的同时大幅提升身体各部位的文本-动作语义对齐精度，解决了现有整体式方法与部位拆分方法之间"部位表达力 vs 全身连贯性"的根本矛盾。

**[PHASE-Net: Physics-Grounded Harmonic Attention System for Efficient Remote Photoplethysmography Measurement](phase-net_physics-grounded_harmonic_attention_system_for_efficient_remote_photop.md)**

:   从Navier-Stokes方程出发，通过严格数学推导揭示rPPG脉搏信号遵循二阶阻尼谐振子模型，其离散解形式等价于因果卷积算子，从而为TCN架构的选择提供了第一性原理依据，设计出仅0.29M参数的PHASE-Net在多个数据集上达到SOTA。

**[RAM: Recover Any 3D Human Motion in-the-Wild](ram_recover_any_3d_human_motion_in-the-wild.md)**

:   RAM 提出统一的多人 3D 运动恢复框架，集成运动感知语义跟踪器 SegFollow（基于 SAM2 + 自适应卡尔曼滤波）、记忆增强的时序人体网格恢复模块 T-HMR、轻量运动预测器和门控组合器，在 PoseTrack 和 3DPW 等基准上实现零样本跟踪稳定性和 3D 精度的 SOTA，且推理速度比之前方法快 2-3 倍。

**[Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)**

:   提出 VTON-IQA，一个无需参考图的虚拟试穿图像质量评估框架，通过构建 62,688 张试穿图像 × 431,800 条人工标注的大规模基准 VTON-QBench，以及交错式交叉注意力（ICA）模块建模服装-人物-试穿图之间的交互关系，实现与人类感知高度对齐的图像级质量预测。

**[RefTon: Reference Person Shot Assist Virtual Try-on](refton_reference_person_shot_assist_virtual_try-on.md)**

:   本文提出 RefTon，一个基于 Flux-Kontext 的人对人虚拟试穿框架，通过引入额外参考图像（其他人穿着目标服装的照片）来提供更准确的服装细节信息，同时通过两阶段训练策略和缩放位置索引机制实现了无需辅助条件（如 DensePose、分割掩码）的端到端试穿，在 VITON-HD 和 DressCode 上达到 SOTA。

**[RegFormer: Transferable Relational Grounding for Efficient Weakly-Supervised HOI Detection](regformer_transferable_relational_grounding_for_weakly-supervised_hoi_detection.md)**

:   RegFormer 提出一个轻量级关系接地 Transformer 模块，在仅图像级标注的弱监督下，通过空间接地查询和交互性感知学习，直接从图像级推理迁移到实例级 HOI 检测，无需额外训练，性能接近全监督方法。

**[ReMoGen: Real-time Human Interaction-to-Reaction Generation via Modular Learning from Diverse Data](remogen_real-time_human_interaction-to-reaction_generation_via_modular_learning_.md)**

:   提出 ReMoGen，一个模块化框架用于实时人体交互-到-反应的动作生成：利用大规模单人运动数据学习通用运动先验（冻结），通过独立训练的 Meta-Interaction 模块适配不同交互域（人-人/人-场景），并引入 Frame-wise Segment Refinement 实现逐帧低延迟在线更新（0.047s/帧），在 Inter-X 和 LINGO 数据集上全面超越 SOTA。

**[rPPG-VQA: A Video Quality Assessment Framework for Unsupervised rPPG Training](rppg_vqa_video_quality_assessment.md)**

:   rPPG-VQA 提出首个面向远程心率检测（rPPG）的视频质量评估框架，结合信号级多方法共识 SNR 和场景级 MLLM 干扰识别，配合两阶段自适应采样策略筛选野外视频构建训练集。

**[Seeing without Pixels: Perception from Camera Trajectories](seeing_without_pixels_perception_from_camera_trajectories.md)**

:   本文首次系统性地将相机位姿轨迹（6DoF pose sequence）提升为一种独立的视频感知模态，通过对比学习框架训练轻量级 Transformer 编码器 CamFormer，将相机轨迹映射到与文本对齐的联合嵌入空间，在 5 个数据集的 10 个下游任务上证明相机轨迹是既轻量又鲁棒的视频内容信号——在物理活动上甚至可以超越计算量大数千倍的视频模型。

**[Sketch2Colab: Sketch-Conditioned Multi-Human Animation via Controllable Flow Distillation](sketch2colab_sketch-conditioned_multi-human_animation_via_controllable_flow_dist.md)**

:   提出 Sketch2Colab，通过将草图驱动的扩散先验蒸馏为整流流学生网络，结合能量引导和连续时间马尔可夫链（CTMC）离散事件规划，从故事板草图生成协调的多人-物体交互 3D 动作，在 CORE4D 和 InterHuman 上实现 SOTA 约束遵从度和感知质量。

**[Stake the Points: Structure-Faithful Instance Unlearning](stake_the_points_structure-faithful_instance_unlearning.md)**

:   提出 Structguard，通过语义锚点（semantic anchors）保持遗忘过程中保留实例间的语义关系结构，避免结构性崩塌，在图像分类/人脸识别/检索三任务上平均提升 32.9%/19.3%/22.5%。

**[Talking Together: Synthesizing Co-Located 3D Conversations from Audio](talking_together_synthesizing_co-located_3d_conversations_from_audio.md)**

:   首次提出从单一混合音频流生成两个**共处同一3D空间**的对话参与者完整面部动画的方法，通过双流扩散架构（共享 U-Net + 跨注意力）、两阶段混合数据训练策略、LLM 驱动的文本-空间布局控制以及辅助眼神损失，实现自然的互视、转头和空间感知的双人对话3D动画合成。

**[Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach](team_leya_in_10th_abaw_competition_multimodal_ambi.md)**

:   提出四模态（场景 VideoMAE + 人脸 EfficientNetB0 + 音频 Wav2Vec2.0/Mamba + 文本 EmotionDistilRoBERTa）融合管线，通过原型增强 Transformer 融合模块将各模态嵌入投影到共享 128 维空间并以原型分类辅助损失正则化，在 BAH 语料的最终测试集上以 5 模型集成达到 **71.43% Macro F1**，显著超越所有单模态基线。

**[TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures](tehor_text-guided_3d_human_and_object_reconstruction_with_textures.md)**

:   TeHOR 利用文本描述作为语义引导，通过预训练扩散模型的 Score Distillation Sampling 联合优化 3D 人体和物体的几何与纹理，突破了传统方法对接触信息的依赖，实现了包括非接触交互在内的准确且语义一致的 3D 重建。

**[4DSurf: High-Fidelity Dynamic Scene Surface Reconstruction](textit4dsurf_high-fidelity_dynamic_scene_surface_reconstruction.md)**

:   本文提出 4DSurf，一个基于2D高斯泼溅的通用动态场景表面重建框架，通过引入高斯运动诱导的SDF流正则化来约束表面时序一致演化，并采用重叠分段策略处理大变形，在 Hi4D 和 CMU Panoptic 数据集上分别以 49% 和 19% 的 Chamfer 距离改进超越现有 SOTA。

**[TriLite: Efficient WSOL with Universal Visual Features and Tri-Region Disentanglement](trilite_efficient_weakly_supervised_object_localization_with_universal_visual_fe.md)**

:   仅使用冻结 DINOv2 ViT + 不到 800K 可训练参数的 TriHead 模块，通过将 patch 特征解耦为前景/背景/模糊三区域并引入对抗性背景损失，在 WSOL 上以极少参数刷新 SOTA。

**[UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos](unidex_a_robot_foundation_suite_for_universal_dexterous_hand_control_from_egocen.md)**

:   提出UniDex机器人基础套件——包含跨8种灵巧手的大规模数据集（50K+轨迹/9M帧）、功能-执行器对齐的统一动作空间（FAAS）和3D VLA策略（UniDex-VLA），在真实世界工具使用任务上达到81%平均任务进度（vs π₀的38%），并展示了空间、物体和零样本跨手泛化能力。

**[UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking](unils_end-to-end_audio-driven_avatars_for_unified_listening_and_speaking.md)**

:   提出首个端到端统一说话-倾听面部表情生成框架UniLS，通过两阶段训练范式（先学内在运动先验、再用双轨音频微调），仅需双方音频输入即可同时生成自然的说话和倾听面部动作，倾听指标提升高达44.1%。

**[Unleashing Vision-Language Semantics for Deepfake Video Detection](unleashing_vision-language_semantics_for_deepfake_video_detection.md)**

:   提出VLAForge，通过ForgePerceiver独立学习多样的伪造线索和伪造定位图，并结合身份感知的视觉-语言对齐（VLA）评分机制，释放VLM跨模态语义的潜力来增强深度伪造视频检测的判别能力，在9个数据集上全面超越现有SOTA。

**[ViBES: A Conversational Agent with Behaviorally-Intelligent 3D Virtual Body](vibes_a_conversational_agent_with_behaviorally_intelligent_3d_virtual_body.md)**

:   提出 ViBES，一个统一语言、语音和身体动作的 3D 对话代理，通过模态专家混合（MoME）架构和跨模态注意力机制，在保留预训练语音 LLM 对话能力的同时生成时间对齐的面部表情和全身动作，超越了将行为视为简单"模态翻译"的范式。

**[Vision-Language Attribute Disentanglement and Reinforcement for Lifelong Person Re-Identification](vision-language_attribute_disentanglement_and_reinforcement_for_lifelong_person_.md)**

:   VLADR 提出利用视觉-语言模型（VLM）中的细粒度属性知识来增强终身行人重识别，通过多粒度文本属性解耦（MTAD）和跨域跨模态属性强化（ICAR）两阶段训练，显式建模跨域共享的人体属性以实现高效知识转移和遗忘缓解，在抗遗忘和泛化能力上分别超越 SOTA 1.9%-2.2% 和 2.1%-2.5%。

**[WildCap: Facial Albedo Capture in the Wild via Hybrid Inverse Rendering](wildcap_facial_albedo_capture_in_the_wild_via_hybrid_inverse_rendering.md)**

:   提出 WildCap，通过混合逆渲染框架（数据驱动 SwitchLight 去光照 + 基于模型的 texel grid lighting 优化 + 扩散先验采样），从手机野外视频中重建高质量 4K 面部漫反射 albedo 贴图，大幅缩小野外捕捉与受控光照方法之间的质量差距。
