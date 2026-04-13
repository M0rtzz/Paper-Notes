---
title: >-
  ICCV2025 视频理解方向 73篇论文解读
description: >-
  73篇ICCV2025 视频理解方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎬 视频理解

**📹 ICCV2025** · 共 **73** 篇

**[4D-Bench Benchmarking Multi-Modal Large Language Models For 4D Object Understand](4d-bench_benchmarking_multi-modal_large_language_models_for_4d_object_understand.md)**

:   4D-Bench 是首个评估多模态大语言模型（MLLM）4D 物体理解能力的基准，包含 4D 物体问答和描述两大任务，揭示了即使 SOTA GPT-4o 也仅达 63% 准确率（人类基线 91%），暴露了当前 MLLM 在多视角时序推理上的显著不足。

**[4Dbench Benchmarking Multimodal Large Language Models For 4D](4dbench_benchmarking_multimodal_large_language_models_for_4d.md)**

:   提出 4D-Bench，首个评估多模态大语言模型对4D物体（具有时间演化的3D物体）理解能力的基准，包含4D物体问答（751 QA对）和4D物体描述（580物体×5标注）两大任务，发现即使SOTA的GPT-4o也仅达63%准确率（人类91%），揭示了MLLM在多视角时空理解上的巨大差距。

**[Adaptive Hyper-Graph Convolution Network For Skeleton-Based Human Action Recogni](adaptive_hyper-graph_convolution_network_for_skeleton-based_human_action_recogni.md)**

:   提出 Hyper-GCN，通过**自适应非均匀超图**替代传统二元图来建模骨骼拓扑，并引入**虚拟超关节**（hyper joints）创建虚拟连接，使多关节协同关系得以直接建模，在 NTU-60/120 和 NW-UCLA 上以最轻量的 GCN 设计实现 SOTA（base 版仅 1.1M 参数、1.63 GFLOPs）。

**[Aim Adaptive Inference Of Multi-Modal Llms Via Token Merging And Pruning](aim_adaptive_inference_of_multi-modal_llms_via_token_merging_and_pruning.md)**

:   提出一种无需训练的自适应推理方法，通过 LLM 前基于嵌入相似度的迭代式 token 合并 + LLM 层内基于 PageRank 多模态重要性的渐进式 token 剪枝，实现多模态 LLM 在 40 倍 FLOPs 减少范围内的灵活精度-效率权衡，在视频和图像理解任务上均取得优异表现。

**[Aligning Effective Tokens With Video Anomaly In Large Language Models](aligning_effective_tokens_with_video_anomaly_in_large_language_models.md)**

:   提出VA-GPT，通过空间有效Token选择（SETS）和时间有效Token生成（TETG）两个模块，在MLLM中高效对齐与视频异常相关的关键Token，实现对异常事件的精准检测、描述和时间定位。

**[Alltracker Efficient Dense Point Tracking At High Resolution](alltracker_efficient_dense_point_tracking_at_high_resolution.md)**

:   提出AllTracker，将点跟踪重新表述为多帧长程光流问题，在低分辨率网格上通过2D卷积+像素对齐时序注意力迭代优化对应估计再上采样，仅16M参数即实现SOTA准确率和高分辨率（768×1024）全像素密集跟踪，跟踪速度接近光流方法。

**[An Empirical Study Of Autoregressive Pre-Training From Videos](an_empirical_study_of_autoregressive_pre-training_from_videos.md)**

:   系统性地研究了从视频进行自回归预训练的方法（称为Toto），在超过1万亿视觉token上训练因果Transformer，发现尽管归纳偏置极少，自回归预训练在图像识别、视频分类、目标跟踪和机器人操控等多个下游任务上均具有竞争力，且展现出类似语言模型的缩放规律（但速率较慢）。

**[Attention To Trajectory Trajectory-Aware Open-Vocabulary Tracking](attention_to_trajectory_trajectory-aware_open-vocabulary_tracking.md)**

:   本文提出TRACT，一种利用轨迹级信息增强开放词汇多目标跟踪（OV-MOT）的方法，通过轨迹一致性强化（TCR）改善关联、通过轨迹特征聚合（TFA）和轨迹语义丰富（TSE）改善分类，在OV-TAO基准上显著提升了跟踪性能，尤其是分类准确率。

**[Beyond Label Semantics Language-Guided Action Anatomy For Few-Shot Action Recogn](beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)**

:   提出 Language-Guided Action Anatomy (LGA) 框架，利用大语言模型将动作标签解剖为原子级动作描述（主体-动作-对象三要素），同时在视频端通过聚类分割将帧序列划分为对应的原子动作阶段，在原子级别进行多模态融合和匹配，显著提升小样本动作识别性能。

**[Beyond The Frame Generating 360Deg Panoramic Videos From Perspective Videos](beyond_the_frame_generating_360deg_panoramic_videos_from_perspective_videos.md)**

:   提出 Argus 模型，首次实现从普通透视视频生成完整 360° 全景视频，通过相机运动模拟、视角对齐帧校准和混合解码三大几何-运动感知技术，在基于扩散模型的框架上让生成的全景视频具备空间一致性和时序连贯性。

**[Blinktrack Feature Tracking Over 80 Fps Via Events And Images](blinktrack_feature_tracking_over_80_fps_via_events_and_images.md)**

:   提出 BlinkTrack，将可微卡尔曼滤波引入学习框架，有效解决事件相机和传统相机异步数据的关联与不确定性感知融合，实现超过 80 FPS 的高帧率特征跟踪，并在遮挡场景中显著优于现有方法。

**[Breaking The Encoder Barrier For Seamless Video-Language Understanding](breaking_the_encoder_barrier_for_seamless_video-language_understanding.md)**

:   提出 ELVA，首个无编码器（encoder-free）的视频大语言模型，通过层级 token 合并、视频引导监督和混合分辨率推理机制，仅用 7M 公开视频-文本对数据即可达到与有编码器架构相当的性能，同时将 FLOPs 降低 95%、推理延迟降低 92%。

**[D3 Training-Free Ai-Generated Video Detection Using Second-Order Features](d3_training-free_ai-generated_video_detection_using_second-order_features.md)**

:   本文从牛顿力学的二阶控制系统出发，发现真实视频和 AI 生成视频在二阶时序特征（"加速度"）上存在本质差异——真实视频波动大而生成视频平坦，据此提出 D3，一种完全免训练的 AI 生成视频检测方法，仅需计算帧间特征的二阶差分标准差即可判别，在 40 个测试子集上达到 SOTA。

**[Dacon Dino For Anime Paint Bucket Colorization With Any Number Of Reference Imag](dacon_dino_for_anime_paint_bucket_colorization_with_any_number_of_reference_imag.md)**

:   提出DACoN，利用DINOv2基础模型的语义特征与U-Net的高分辨率空间特征融合，实现支持任意数量参考图像的动画线稿自动上色，在关键帧和连续帧上色任务中均超越现有方法。

**[Despite Exploring Contrastive Deep Skeleton-Pointcloud-Imu-Text Embeddings For A](despite_exploring_contrastive_deep_skeleton-pointcloud-imu-text_embeddings_for_a.md)**

:   DeSPITE 提出了一种隐私保护的多模态对比预训练模型，将 LiDAR 点云、骨架姿态、IMU 和文本四种模态对齐到统一嵌入空间，实现了跨模态匹配、检索以及人体活动识别的预训练范式。

**[Distime Distribution-Based Time Representation For Video Large Language Models](distime_distribution-based_time_representation_for_video_large_language_models.md)**

:   提出DisTime框架，通过一个可学习的时间token和基于分布的时间解码器，在Video-LLM中实现连续时间表示，配合大规模自动标注数据集InternVid-TG（125万事件），在时刻检索、密集视频描述、Grounded-VQA三类时间敏感任务上达到SOTA。

**[Dollar Fewstep Video Generation Via Distillation And Latent](dollar_fewstep_video_generation_via_distillation_and_latent.md)**

:   结合变分分数蒸馏（VSD）和一致性蒸馏实现few-step视频生成，同时提出潜空间奖励模型微调方法进一步优化生成质量，4步生成的10秒视频（128帧@12FPS）在VBench上达82.57分超越teacher模型和Gen-3/Kling等基线，1步蒸馏实现278.6倍加速。

**[Dreamrelation Relation-Centric Video Customization](dreamrelation_relation-centric_video_customization.md)**

:   提出 DreamRelation，首个关系中心的视频定制方法，通过 Relation LoRA Triplet + Hybrid Mask Training 实现关系与外观的解耦，并通过时空关系对比损失增强关系动态学习，使动物能模仿人类交互。

**[Dualreal Adaptive Joint Training For Lossless Identity-Motion Fusion In Video Cu](dualreal_adaptive_joint_training_for_lossless_identity-motion_fusion_in_video_cu.md)**

:   DualReal 首次提出身份与运动的自适应联合训练框架，通过 Dual-aware Adaptation 和 StageBlender Controller 实现两个维度的无损融合，在 CLIP-I 和 DINO-I 指标上平均提升 21.7% 和 31.8%。

**[Dynimg Key Frames With Visual Prompts Are Good Representation For Multi-Modal Vi](dynimg_key_frames_with_visual_prompts_are_good_representation_for_multi-modal_vi.md)**

:   DynImg 提出了一种新颖的视频表示方法，将非关键帧作为"时序视觉提示"叠加在关键帧下方形成动态图像，在视觉编码器内部实现细粒度时空交互（而非高层token级交互），配合4D旋转位置编码维护正确的时空序列关系，在多个视频理解基准上以更少的视觉token超越SOTA约2%。

**[Egoadapt Adaptive Multisensory Distillation And Policy Learning For Efficient Eg](egoadapt_adaptive_multisensory_distillation_and_policy_learning_for_efficient_eg.md)**

:   提出 EgoAdapt 框架，将跨模态蒸馏与策略学习联合训练，自适应选择最优模态组合，在自我中心感知任务中实现最高 89% GMACs 缩减的同时保持与 SOTA 持平甚至更优的性能。

**[Egoppg Heart Rate Estimation From Eye-Tracking Cameras In Egocentric Systems To ](egoppg_heart_rate_estimation_from_eye-tracking_cameras_in_egocentric_systems_to_.md)**

:   提出egoPPG这一新的自中心视觉任务，通过PulseFormer方法从未修改的自中心头戴设备的眼部追踪摄像头估计心率（MAE=7.67 bpm），并证明心率估计在EgoExo4D的技能水平评估下游任务中可提升14.1%的准确率。

**[Emotive Event-Guided Trajectory Modeling For 3D Motion Estimation](emotive_event-guided_trajectory_modeling_for_3d_motion_estimation.md)**

:   本文提出 EMoTive，一个基于事件相机的 3D 运动估计框架，通过 Event Kymograph 编码精细时序演化信息，并使用事件密度引导的非均匀 NURBS 参数曲线建模时空轨迹，从轨迹中导出光流和深度运动场，在自建 CarlaEvent3D 数据集和真实世界基准上取得 SOTA 性能。

**[Estimating 2D Camera Motion With Hybrid Motion Basis](estimating_2d_camera_motion_with_hybrid_motion_basis.md)**

:   提出 CamFlow，通过混合运动基（12 个物理基 + 随机噪声基）表示复杂的 2D 相机运动，揭示了多个单应性流场叠加的非线性特性，结合基于 Laplace 分布的概率损失函数，在标准和跨数据集零样本条件下均大幅超越现有单应性和 meshflow 方法。

**[Factorized Learning For Temporally Grounded Video-Language Models](factorized_learning_for_temporally_grounded_video-language_models.md)**

:   提出D2VLM框架，通过将视频理解分解为"先定位证据再基于证据生成回答"的范式，引入证据token捕捉事件级视觉语义，并设计分解式偏好优化(FPO)同时提升时序定位和文本回答能力。

**[Fine-Grained Spatiotemporal Grounding On Egocentric Videos](fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)**

:   提出 EgoMask，首个面向自我中心视频的像素级时空定位基准，包含短/中/长时视频评测集和大规模训练集 EgoMask-Train，通过系统分析揭示了自我中心与外中心视频之间的关键差异，并证明微调后模型性能可大幅提升。

**[Flow4Agent Long-Form Video Understanding Via Motion Prior From Optical Flow](flow4agent_long-form_video_understanding_via_motion_prior_from_optical_flow.md)**

:   Flow4Agent 首次将光流运动先验引入 LLM-based 视频理解，通过时域粒度优化（TGO）利用粗粒度光流聚类视频事件并用语义先验过滤冗余场景，通过运动 Token 剪枝（MTP）利用细粒度光流去除帧内静态冗余 token，在 VideoMME/MLVU/LongVideoBench 等长视频基准上取得领先表现。

**[Flowseek Optical Flow Made Easier With Depth Foundation Models And Motion Bases](flowseek_optical_flow_made_easier_with_depth_foundation_models_and_motion_bases.md)**

:   FlowSeek 将深度基础模型（Depth Anything V2）的先验知识和经典的低维运动参数化（motion bases）融入光流网络，在仅使用单张消费级 GPU 训练的条件下即可实现 SOTA 的跨数据集泛化性能。

**[Free-Moref Instantly Multiplexing Context Perception Capabilities Of Video-Mllms](free-moref_instantly_multiplexing_context_perception_capabilities_of_video-mllms.md)**

:   提出免训练方法Free-MoRef，受MoE启发将长视频token分割为多个短序列作为多参考(multi-reference)，通过MoRef注意力机制并行查询并融合统一激活值，在单卡A100上实现2×到8×更长帧输入的高效全面理解，在VideoMME/MLVU/LongVideoBench上超越专训长视频模型。

**[Fuxi-Rtm A Physics-Guided Prediction Framework With Radiative Transfer Modeling](fuxi-rtm_a_physics-guided_prediction_framework_with_radiative_transfer_modeling.md)**

:   提出 FuXi-RTM，首个将深度学习辐射传输模型 (DLRTM) 作为可微物理正则化器集成到天气预报框架中的混合物理引导体系，在 88.51% 的变量-预报时效组合上超越无约束基线。

**[General Compression Framework For Efficient Transformer Object Tracking](general_compression_framework_for_efficient_transformer_object_tracking.md)**

:   提出 CompressTracker，一个通用 Transformer 跟踪器压缩框架，通过阶段划分、替换训练和特征模仿三个递进创新，实现结构无关的高效压缩——压缩 SUTrack 后保持约 99% 精度同时加速 2.42 倍。

**[Generating Fast And Slow Scalable Parallel Video Generation With Video Interface](generating_fast_and_slow_scalable_parallel_video_generation_with_video_interface.md)**

:   提出 Video Interface Networks (VINs)，一种类似"快思考"的抽象模块，在每个扩散步中将长视频编码为固定大小的全局 token，引导 DiT 并行生成多个视频 chunk，实现高效且时序一致的长视频生成。

**[Hermes Temporal-Coherent Long-Form Understanding With Episodes And Semantics](hermes_temporal-coherent_long-form_understanding_with_episodes_and_semantics.md)**

:   提出 HERMES 框架，通过情景压缩器 (ECO) 和语义检索器 (SeTR) 两个通用模块分别捕获视频的情景记忆和语义信息，既可作为独立系统达到 SOTA，也可即插即用地增强现有视频语言模型，同时降低推理延迟达 43% 和内存消耗达 46%。

**[Hierarchical Event Memory For Accurate And Low-Latency Online Video Temporal Gro](hierarchical_event_memory_for_accurate_and_low-latency_online_video_temporal_gro.md)**

:   针对在线视频时序定位（OnVTG）任务，提出层级事件记忆机制存储不同时间尺度的历史事件信息，结合基于段树结构的事件提议和未来预测分支，在TACoS、ActivityNet Captions和MAD三大数据集上实现了SOTA的定位精度和低延迟预测。

**[Learning To Generalize Without Bias For Open-Vocabulary Action Recognition](learning_to_generalize_without_bias_for_open-vocabulary_action_recognition.md)**

:   本文提出 Open-MeDe，一个基于元学习的开放词汇动作识别框架，通过跨批次元优化模拟"已知到开放"的泛化任务，并结合高斯自集成稳定化策略，在不依赖 CLIP 正则化的情况下同时提升上下文内和上下文外场景的泛化能力。

**[Memfof High-Resolution Training For Memory-Efficient Multi-Frame Optical Flow Es](memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)**

:   MEMFOF 是首个面向显存效率的多帧光流方法，通过降低相关体积分辨率并引入高分辨率训练策略，在 1080p 推理仅需 2.09GB 显存的同时在 Spring、Sintel、KITTI 等基准上达到 SOTA 精度。

**[Mikudance Animating Character Art With Mixed Motion Dynamics](mikudance_animating_character_art_with_mixed_motion_dynamics.md)**

:   提出 MikuDance，一种基于扩散模型的角色艺术动画系统，通过 Mixed Motion Modeling（将角色运动和 3D 相机运动统一到像素空间表示）和 Mixed-Control Diffusion（在 Reference UNet 中隐式对齐角色形状/尺度与运动引导），实现了复杂角色画作的高动态动画生成。

**[Miore Var-Miore Benchmarks To Push The Boundaries Of Restoration](miore_var-miore_benchmarks_to_push_the_boundaries_of_restoration.md)**

:   提出 MIORe 和 VAR-MIORe 两个多任务运动复原基准数据集，使用 1000fps 工业级相机和专业镜头阵列采集，涵盖从极微到极端的全运动幅度谱，通过自适应帧平均机制生成一致运动模糊，为去模糊、帧插值和光流估计提供统一评估平台。

**[Mobileviclip An Efficient Video-Text Model For Mobile Devices](mobileviclip_an_efficient_video-text_model_for_mobile_devices.md)**

:   将时空结构重参数化引入高效图像-文本模型MobileCLIP，在大规模视频-文本数据集上训练，得到可在移动端运行的视频-文本模型MobileViCLIP，在零样本检索和动作识别上取得与大模型相当的性能。

**[Moment Quantization For Video Temporal Grounding](moment_quantization_for_video_temporal_grounding.md)**

:   提出 MQVTG，首次将向量量化引入视频时序定位任务，通过时刻码本和软量化将视频片段映射为离散向量，增强前景/背景的区分度，在 6 个基准上取得 SOTA。

**[Motionshot Adaptive Motion Transfer Across Arbitrary Objects For Text-To-Video G](motionshot_adaptive_motion_transfer_across_arbitrary_objects_for_text-to-video_g.md)**

:   提出 MotionShot，一个无需训练的运动迁移框架，通过高层语义对齐和低层形态对齐的两级运动对齐策略，实现在外观和结构差异显著的任意参考-目标物体对之间的高保真运动迁移。

**[Multi-Modal Multi-Platform Person Re-Identification Benchmark And Method](multi-modal_multi-platform_person_re-identification_benchmark_and_method.md)**

:   提出首个多模态多平台行人重识别基准 MP-ReID（含 RGB、红外、热成像三种模态 + 地面和无人机两种平台）和统一提示学习框架 Uni-Prompt ReID，通过模态感知、平台感知和视觉增强提示显著提升复杂场景下的 ReID 性能。

**[No More Sibling Rivalry Debiasing Human-Object Interaction Detection](no_more_sibling_rivalry_debiasing_human-object_interaction_detection.md)**

:   发现并系统分析了 HOI 检测中的"有毒兄弟"偏差问题——高度相似的 HOI 三元组在输入端和输出端相互干扰竞争，提出"对比后校准"（C2C）和"合并后拆分"（M2S）两种去偏学习目标，在 HICO-DET 上超越 baseline +9.18% mAP、超越前 SOTA +3.59%。

**[Omnihuman-1 Rethinking The Scaling-Up Of One-Stage Conditioned Human Animation M](omnihuman-1_rethinking_the_scaling-up_of_one-stage_conditioned_human_animation_m.md)**

:   提出 OmniHuman，一种基于 Diffusion Transformer 的多条件人体动画生成框架，通过混合文本/音频/姿态等运动相关条件的全条件训练策略实现数据规模化，首次实现单一模型支持任意身体比例、任意宽高比输入的音频驱动人体视频生成，在肖像和半身动画任务上均达到 SOTA。

**[Online Dense Point Tracking With Streaming Memory](online_dense_point_tracking_with_streaming_memory.md)**

:   提出 SPOT 框架，通过定制的记忆读取模块、感知记忆（sensory memory）和可见性引导的 splatting 实现在线稠密长程点跟踪，以 10× 更少参数和 2× 更快速度达到 CVO 基准上的 SOTA，在多个稀疏跟踪基准上也超越或媲美离线方法。

**[Ovg-Hq Online Video Grounding With Hybrid-Modal Queries](ovg-hq_online_video_grounding_with_hybrid-modal_queries.md)**

:   提出在线视频定位新任务 OVG-HQ，支持文本/图像/视频片段等混合模态查询，通过参数化记忆块（PMB）保留历史信息和混合蒸馏策略缓解模态不平衡，在流式视频中实时定位目标片段。

**[Prior-Flow Enhancing Primitive Panoramic Optical Flow With O](prior-flow_enhancing_primitive_panoramic_optical_flow_with_o.md)**

:   提出双分支框架 PriOr-Flow，利用正交视图的低畸变先验来补偿 ERP 全景图像极区的严重畸变，从而显著提升全景光流估计精度，在 MPFDataset 和 FlowScape 上分别降低 EPE 30.0% 和 29.6%。

**[Prior-Flow Enhancing Primitive Panoramic Optical Flow With Orthogonal View](prior-flow_enhancing_primitive_panoramic_optical_flow_with_orthogonal_view.md)**

:   提出双分支框架 PriOr-Flow，利用正交视图的低畸变先验来补偿 ERP 全景图像极区的严重畸变，从而显著提升全景光流估计精度，在 MPFDataset 和 FlowScape 上分别降低 EPE 30.0% 和 29.6%。

**[Q-Frame Query-Aware Frame Selection And Multi-Resolution Adaptation For Video-Ll](q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)**

:   提出 Q-Frame，一种无需训练的即插即用视频帧选择与多分辨率自适应框架，利用 CLIP 跨模态匹配和 Gumbel-Max 技巧实现查询感知的帧选择，使 Video-LLM 在相同计算预算下处理更多关键帧，在 MLVU、LongVideoBench、Video-MME 三个基准上显著提升性能。

**[Rainbowprompt Diversity-Enhanced Prompt-Evolving For Continual Learning](rainbowprompt_diversity-enhanced_prompt-evolving_for_continual_learning.md)**

:   提出 RainbowPrompt，通过注意力变换和任务引导对齐的提示演化机制，将多个任务特定提示整合为多样性增强的统一提示，在图像分类和视频动作识别任务上平均超越现有方法 8.23%。

**[Recammaster Camera-Controlled Generative Rendering From A Single Video](recammaster_camera-controlled_generative_rendering_from_a_single_video.md)**

:   提出 ReCamMaster，通过帧维度拼接的视频条件注入机制和 UE5 合成的多相机同步数据集，实现从单视频输入以新相机轨迹重新生成视频，显著超越现有方法。

**[Residualvit For Efficient Temporally Dense Video Encoding](residualvit_for_efficient_temporally_dense_video_encoding.md)**

:   本文提出 ResidualViT，通过类比视频压缩中的 I帧/P帧 策略，交替使用完整 ViT 和轻量残差 ViT 编码视频帧，在保持接近原始 CLIP 精度的同时，实现最高 60% 的计算成本降低和 2.5 倍推理加速。

**[Simultaneous Motion And Noise Estimation With Event Cameras](simultaneous_motion_and_noise_estimation_with_event_cameras.md)**

:   首次提出事件相机运动估计与噪声估计的联合方法，利用对比度最大化（CMax）框架中运动补偿后的局部对比度对每个事件评分，通过交替优化同时获得运动参数和信号/噪声分类，在 E-MLB 去噪基准上达到 SOTA。

**[Stiv Scalable Text And Image Conditioned Video Generation](stiv_scalable_text_and_image_conditioned_video_generation.md)**

:   本文提出 STIV，一个基于 Diffusion Transformer 的统一文本-图像条件视频生成框架，通过帧替换策略整合图像条件并引入联合图像-文本 classifier-free guidance，在单一模型中同时实现 T2V 和 TI2V 生成，8.7B 参数模型在 VBench T2V 和 I2V 上分别达到 83.1 和 90.1 的 SOTA 成绩。

**[Sweettok Semantic-Aware Spatial-Temporal Tokenizer For Compact Video Discretizat](sweettok_semantic-aware_spatial-temporal_tokenizer_for_compact_video_discretizat.md)**

:   提出 SweetTok 视频 tokenizer，通过解耦查询自编码器（DQAE）分离空间和时间信息压缩、运动增强语言码本（MLC）按词性分配码字，在仅使用 25% token 数量的情况下，rFVD 改善 42.8%，gFVD 改善 15.1%，实现压缩率与重建保真度的最佳平衡。

**[Timeexpert An Expert-Guided Video Llm For Video Temporal Grounding](timeexpert_an_expert-guided_video_llm_for_video_temporal_grounding.md)**

:   提出TimeExpert——首个基于MoE的Video-LLM框架，通过**任务感知动态门控**和**token自适应路由**将时间戳、显著性分数和文本描述路由到专门的专家，配合任务依赖辅助损失，在Dense Video Captioning、Moment Retrieval和Video Highlight Detection三类VTG任务上全面超越SOTA。

**[Toga Temporally Grounded Open-Ended Video Qa With Weak Supervision](toga_temporally_grounded_open-ended_video_qa_with_weak_supervision.md)**

:   提出TOGA——一种弱监督条件下的视觉语言模型，通过多尺度视觉语言连接器和一致性约束生成伪时序标签，在**无需任何时序标注**的情况下联合生成开放式答案与时间定位，在NExT-GQA、MSVD-QA和ActivityNet-QA上取得SOTA。

**[Towards Efficient General Feature Prediction In Masked Skeleton Modeling](towards_efficient_general_feature_prediction_in_masked_skeleton_modeling.md)**

:   提出 GFP（General Feature Prediction）框架，将掩码骨架建模的重建目标从低层关节坐标提升为多层次高层语义特征预测，配合轻量级目标生成网络和信息最大化约束，实现 6.2 倍训练加速的同时达到 SOTA 性能。

**[Towards Video Thinking Test A Holistic Benchmark For Advanced Video Reasoning An](towards_video_thinking_test_a_holistic_benchmark_for_advanced_video_reasoning_an.md)**

:   提出 Video Thinking Test (Video-TT)，一个评估视频大语言模型正确性和鲁棒性的基准，包含 1000 个 YouTube Shorts 视频和 5000 个问题，通过视觉/叙事复杂性因子和自然对抗问题揭示了当前最强模型（GPT-4o 36.6%）与人类（84.3%）之间的巨大差距。

**[Trokens Semantic-Aware Relational Trajectory Tokens For Few-Shot Action Recognit](trokens_semantic-aware_relational_trajectory_tokens_for_few-shot_action_recognit.md)**

:   提出Trokens框架，通过**语义感知的轨迹点采样**和**关系运动建模**（包含轨迹内HoD和轨迹间相对位移描述子），将点轨迹转化为语义感知的关系token，与外观特征融合后在6个few-shot动作识别基准上取得SOTA。

**[Umdatrack Unified Multi-Domain Adaptive Tracking Under Adverse Weather Condition](umdatrack_unified_multi-domain_adaptive_tracking_under_adverse_weather_condition.md)**

:   UMDATrack 提出了首个统一多域自适应跟踪框架，利用文本引导扩散模型合成少量（<2% 帧）多天气条件无标注视频，通过域定制适配器（DCA）高效迁移目标表征到不同天气域，并引入基于最优传输的目标感知置信度对齐（TCA）增强跨域定位一致性，在夜间/雾天/雨天等场景中大幅超越现有 SOTA 跟踪器。

**[Unsupervised Joint Learning Of Optical Flow And Intensity With Event Cameras](unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)**

:   提出首个基于单一网络的无监督学习框架，从事件相机数据中联合估计光流和图像亮度，核心是新推导的事件光度误差（PhE）与对比度最大化（CMax）的互补损失函数。

**[Vace All-In-One Video Creation And Editing](vace_all-in-one_video_creation_and_editing.md)**

:   本文提出VACE，一个基于Diffusion Transformer的视频生成与编辑一体化框架，通过统一的Video Condition Unit (VCU)接口和可插拔的Context Adapter结构，用单一模型覆盖参考生成、视频编辑、mask编辑等12+种视频任务，性能与任务专用模型持平。

**[Vace Allinone Video Creation And Editing](vace_allinone_video_creation_and_editing.md)**

:   提出VACE统一视频生成和编辑框架，通过Video Condition Unit（VCU）将参考图→视频生成、视频→视频编辑、mask视频编辑等多种任务的输入统一为标准接口，配合Context Adapter注入时空条件信息，单一模型在各子任务上达到专用模型水平并支持灵活的任务组合。

**[Vamba Understanding Hour-Long Videos With Hybrid Mamba-Transformers](vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)**

:   提出 Vamba —— 一种混合 Mamba-Transformer 架构的大型多模态模型，用 Mamba-2 块以线性复杂度编码视频 token、用交叉注意力更新文本 token，在单 GPU 上可处理 1024 帧视频，在小时级视频理解基准上超越所有高效 LMM 方法。

**[Videollamb Long Streaming Video Understanding With Recurrent Memory Bridges](videollamb_long_streaming_video_understanding_with_recurrent_memory_bridges.md)**

:   提出 VideoLLaMB，通过 SceneTiling 语义分段、循环记忆桥接层和记忆缓存检索机制，以线性 GPU 内存扩展实现长流式视频理解，在 4 个 VideoQA 基准上平均提升 4.2 分。

**[Videominer Iteratively Grounding Key Frames Of Hour-Long Videos Via Tree-Based G](videominer_iteratively_grounding_key_frames_of_hour-long_videos_via_tree-based_g.md)**

:   提出VideoMiner——基于强化学习的长视频理解树结构框架，通过迭代分割-描述-聚类构建层次化视频树，并提出T-GRPO（树结构Group Relative Policy Optimization）引导策略模型自适应探索关键帧，在4个长视频基准上取得SOTA，并发现T-GRPO可自发激发推理链。

**[Vmbench A Benchmark For Perception-Aligned Video Motion Generation](vmbench_a_benchmark_for_perception-aligned_video_motion_generation.md)**

:   提出 VMBench——首个面向视频运动质量评估的综合基准，包含五维感知对齐运动指标（PMM）和元信息引导的运动提示生成框架（MMPG），覆盖 969 类运动类型，在 Spearman 相关系数上比现有方法平均提升 35.3%。

**[Vpo Aligning Text-To-Video Generation Models With Prompt Optimization](vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)**

:   > 提出 VPO 框架，基于三大原则（无害、准确、有用）系统性优化视频生成的文本提示，通过原则导向的SFT和多反馈偏好优化，显著提升生成视频的安全性、对齐度和质量。

**[Vtimecot Thinking By Drawing For Video Temporal Grounding And Reasoning](vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)**

:   > 提出 VTimeCoT，一种无需训练的视觉-时间链式思维框架，通过在视频帧底部叠加可视化进度条和高亮关键片段，使多模态大模型能准确感知时间戳，在时间定位和推理问答任务上大幅超越 GPT-4o 和 Qwen2VL-7B 基线。

**[What You Have Is What You Track Adaptive And Robust Multimodal Tracking](what_you_have_is_what_you_track_adaptive_and_robust_multimodal_tracking.md)**

:   提出FlexTrack——首个系统研究**时序性不完整多模态数据**下跟踪性能的框架，通过异构MoE融合机制（HMoE）实现自适应计算复杂度，配合视频级masking训练策略，在9个基准上取得SOTA，完整模态提升2.6%，缺失模态场景提升10.2%。

**[Worldscore A Unified Evaluation Benchmark For World Generation](worldscore_a_unified_evaluation_benchmark_for_world_generation.md)**

:   提出 WorldScore —— 首个统一的世界生成评估基准，将世界生成分解为一系列"下一场景生成"任务，支持对 3D、4D、I2V 和 T2V 模型的统一评测，并涵盖 3000 个测试样本和 10 项指标。

**[Xtrack Multimodal Training Boosts Rgb-X Video Object Trackers](xtrack_multimodal_training_boosts_rgb-x_video_object_trackers.md)**

:   提出 XTrack，通过 Mixture of Modal Experts (MeME) 框架和软路由分类器，实现 RGB-D/T/E 跨模态知识共享，使推理时仅用单模态即可受益于多模态训练知识，平均精度提升 3%。
