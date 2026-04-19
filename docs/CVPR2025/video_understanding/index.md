---
title: >-
  CVPR2025 视频理解方向 45篇论文解读
description: >-
  45篇CVPR2025 视频理解方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📹 视频理解

**📷 CVPR2025** · **45** 篇论文解读

**[BehaviorVLM: Unified Finetuning-Free Behavioral Understanding with Vision-Language Reasoning](behaviorvlm_unified_finetuning-free_behavioral_understanding_with_vision-languag.md)**

:   提出 BehaviorVLM，一个统一的无需微调的视觉语言框架，通过多阶段结构化推理管线同时解决动物姿态估计和行为理解两大任务，仅需 3 帧人工标注即可实现可靠的关键点追踪，并通过深度嵌入聚类 + VLM 描述 + LLM 语义合并实现可解释的多动物行为分割。

**[Beyond Single-Sample: Reliable Multi-Sample Distillation for Video Understanding](beyond_single-sample_reliable_multi-sample_distillation_for_video_understanding.md)**

:   提出 R-MSD（Reliable Multi-Sample Distillation），通过对每个输入采样多个教师响应并结合任务自适应质量匹配，解决视频 LVLM 黑盒蒸馏中单样本教师监督不可靠的问题，4B 学生模型在 VideoMME (+1.5%)、Video-MMMU (+3.2%)、MathVerse (+3.6%) 等基准上取得一致提升。

**[BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions](bim-vfi_bidirectional_motion_field-guided_frame_interpolation_for_video_with_non.md)**

**[Bimba: Selective-Scan Compression for Long-Range Video Question Answering](bimba_selective-scan_compression_for_long-range_video_question_answering.md)**

**[Bootstrap Your Own Views: Masked Ego-Exo Modeling for Fine-Grained View-Invariant Video Representations](bootstrap_your_own_views_masked_ego-exo_modeling_for_fine-grained_view-invariant.md)**

:   通过掩码建模在自我中心和外部视角之间学习细粒度视图不变表示，无需配对标注即可从两种视角的关联中自监督学习

**[Coarse Correspondences Boost Spatial-Temporal Reasoning in Multimodal Language Models](coarse_correspondences_boost_spatial-temporal_reasoning_in_multimodal_language_m.md)**

:   本文提出Coarse Correspondences，一种轻量级的training-free视觉提示方法，通过在图像帧上叠加目标跟踪得到的粗粒度实例对应关系标记，显著增强MLLM的空间时序推理能力，在ScanQA上提升+20.5%、OpenEQA上+9.7%、EgoSchema上+6.0%和R2R导航上+11%。

**[Context-Enhanced Memory-Refined Transformer for Online Action Detection](context-enhanced_memory-refined_transformer_for_online_action_detection.md)**

:   本文揭示了现有在线动作检测（OAD）方法中的训练-推理不一致问题——短时记忆帧的不均衡上下文暴露和伪未来引入的非因果信息泄漏导致学习偏向中间帧——并提出CMeRT通过近过去上下文增强编码器和基于近未来的记忆精炼解码器来解决该问题，在THUMOS'14、CrossTask和EK100上实现SOTA。

**[Cross-modal Causal Relation Alignment for Video Question Grounding](cross-modal_causal_relation_alignment_for_video_question_grounding.md)**

:   通过因果干预消除视频问答定位（VideoQG）中的虚假跨模态关联，引入高斯平滑定位、跨模态对齐和显式因果干预三个模块，在 NextGQA 上同时提升定位（+2.2 Acc@GQA）和问答（+0.9 Acc@VQA）性能。

**[DeCafNet: Delegate and Conquer for Efficient Temporal Grounding in Long Videos](decafnet_delegate_and_conquer_for_efficient_temporal_grounding_in_long_videos.md)**

:   提出DeCafNet，通过**delegate-and-conquer双编码器策略**（轻量sidekick encoder密集提特征+生成显著性图，expert encoder仅处理top-c%关键clip），配合**DeCaf-Grounder**统一不同时序分辨率特征，在长视频时序定位任务上以**减少47% TFLOPs**的代价超越所有先前方法。

**[DivPrune: Diversity-Based Visual Token Pruning for Large Multimodal Models](divprune_diversity-based_visual_token_pruning_for_large_multimodal_models.md)**

:   将视觉token剪枝问题重新建模为**Max-Min Diversity Problem (MMDP)**，通过精确求解使保留token集合的**最小pair-wise距离最大化**，实现无需训练/校准的即插即用剪枝方案，在16个多模态基准上实现SOTA，特别是在≥80%极端剪枝率下显著优于所有基线。

**[DPFlow: Adaptive Optical Flow Estimation with a Dual-Pyramid Framework](dpflow_adaptive_optical_flow_estimation_with_a_dual-pyramid_framework.md)**

:   提出DPFlow，结合**图像金字塔**和**特征金字塔**的双金字塔循环编码器，配合纯卷积的**Cross-Gated Unit (CGU)**，仅用标准分辨率训练即可自适应泛化至8K分辨率输入，在Sintel、KITTI、Spring等基准上达到SOTA，同时发布**Kubric-NK**多分辨率光流评测数据集首次支持定量高分辨率评估。

**[DPU: Dynamic Prototype Updating for Multimodal Out-of-Distribution Detection](dpu_dynamic_prototype_updating_for_multimodal_out-of-distribution_detection.md)**

:   提出**Dynamic Prototype Updating (DPU)**框架，通过**Cohesive-Separate对比训练**建立稳健表示空间、**动态原型逼近**自适应更新类中心、**Pro-ratio差异增强**按样本到原型的距离调节多模态预测差异的放大强度，作为即插即用模块在5个数据集×9种基础OOD方法上全面提升性能，Far-OOD检测提升最高达**80%**。

**[DrVideo: Document Retrieval Based Long Video Understanding](drvideo_document_retrieval_based_long_video_understanding.md)**

:   提出DrVideo，将**长视频理解转化为长文档理解**任务：先将视频帧转为文本文档，通过**文档检索**定位关键帧并**增强信息**，再通过**Planning-Interaction双Agent循环**迭代补充缺失信息，最终以CoT方式回答问题。在EgoSchema（3分钟）、MovieChat-1K（10分钟）和Video-MME长视频分割（平均44分钟）上大幅超越现有LLM-based SOTA。

**[Dynamic Updates for Language Adaptation in Visual-Language Tracking](dynamic_updates_for_language_adaptation_in_visual-language_tracking.md)**

:   提出DUTrack，通过动态更新多模态参考信息（模板帧+语言描述）来解决视觉语言跟踪中静态参考与动态目标之间的语义不一致问题，首次让VL跟踪器在LaSOT上超越最佳纯视觉跟踪器。

**[DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding](dynfocus_dynamic_cooperative_network_empowers_llms_with_video_understanding.md)**

:   提出DynFocus，一个基于LLM的动态协作视频编码网络，通过DPE模块动态选择与问答相关的关键帧，CCE模块对关键帧用细粒度token编码（类似视锥细胞Cones）、对冗余帧用极少token粗粒度编码（类似视杆细胞Rods），在有限token预算下平衡空间细节与时序动态。

**[EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation](edcflow_exploring_temporally_dense_difference_maps_for_event-based_optical_flow_.md)**

:   提出EDCFlow，利用相邻事件帧之间时间密集的特征差分图与低分辨率代价体积的互补性，在1/4分辨率上实现高质量且轻量的事件光流估计。

**[Efficient Transfer Learning for Video-language Foundation Models](efficient_transfer_learning_for_video-language_foundation_models.md)**

:   提出多模态时空适配器MSTA，通过视觉-语言共享投影层和时空描述引导的一致性约束，以仅2-7%的可训练参数实现视频-语言基础模型向下游任务的高效迁移。

**[Ego4o: Egocentric Human Motion Capture and Understanding from Multi-Modal Input](ego4o_egocentric_human_motion_capture_and_understanding_from_multi-modal_input.md)**

:   提出Ego4o统一框架，从穿戴设备的多模态输入（1-3个IMU + 第一人称图像 + 运动描述）同时实现人体运动捕捉和运动描述生成，且两个任务可互相增强。

**[EgoLife: Towards Egocentric Life Assistant](egolife_towards_egocentric_life_assistant.md)**

:   发布EgoLife数据集（6名参与者共居一周、300小时第一人称多模态视频）和EgoLifeQA基准，提出EgoButler系统（EgoGPT + EgoRAG）探索超长上下文第一人称视觉生活助手的建设路径。

**[EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering](egotextvqa_towards_egocentric_scene-text_aware_video_question_answering.md)**

:   提出 EgoTextVQA 基准，包含 1.5K 第一人称视频和 7K 场景文字相关问答对，揭示了当前 MLLM 在以自我中心视角进行实时场景文字问答辅助时的严重不足（最佳模型 Gemini 1.5 Pro 仅约 33% 准确率）。

**[Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation](enhancing_video-llm_reasoning_via_agent-of-thoughts_distillation.md)**

:   AoTD 用 LLM agent 将复杂视频问题分解为子任务、调用专家视觉模型执行并收集中间结果作为推理链（CoT），经 LLM 质量过滤后蒸馏到 Video-LLM 中，让端到端模型同时获得准确答案和可解释的多步推理能力。

**[FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)**

:   提出 FC-Track，一个轻量级的后关联校正框架，通过基于 IoA（Intersection over Area）的外观特征过滤和重叠 tracklet 对内的相似度比较，在线纠正因目标重叠导致的检测-轨迹错误匹配，将长期身份切换比例从 36.86% 降至 29.55%，同时在 MOT17/MOT20 上保持 SOTA 性能。

**[FSBench: A Figure Skating Benchmark for Advancing Artistic Sports Understanding](fsbench_a_figure_skating_benchmark_for_advancing_artistic_sports_understanding.md)**

:   提出 FSAnno/FSBench，首个面向花样滑冰的细粒度、多模态、多层次基准数据集，覆盖从先验知识测试、单个动作识别/评估/解说到整体表演评估/解说的完整任务链，揭示了现有 LLM 在艺术体育理解上的显著不足。

**[GG-SSMs: Graph-Generating State Space Models](gg-ssms_graph-generating_state_space_models.md)**

:   提出 Graph-Generating State Space Models (GG-SSMs)，通过基于特征相似度动态构建最小生成树（MST）来替代传统 SSM 中固定的一维扫描路径，实现对高维数据中复杂非局部依赖的高效建模，在 11 个数据集上取得 SOTA 性能。

**[H-MoRe: Learning Human-centric Motion Representation for Action Analysis](h-more_learning_human-centric_motion_representation_for_action_analysis.md)**

:   提出 H-MoRe（Human-centric Motion Representation），通过骨骼约束和边界约束的联合自监督学习框架，从真实场景中学习精确的以人为中心的运动表示（world-local flows），在步态识别（CL@R1 +16.01%）、动作识别（Acc@1 +8.92%）和视频生成（FVD -67.07%）上均大幅超越传统光流方法。

**[Heterogeneous Skeleton-Based Action Representation Learning](heterogeneous_skeleton-based_action_representation_learning.md)**

:   首次研究人体骨架数据的异构性问题（不同关节数、不同坐标维度），提出通过 3D 姿态估计模块统一维度、骨架特定 prompt 统一拓扑、语义运动编码引入语义信息三大组件，结合自监督统一表示学习框架，在 NTU-60/120 和 PKU-MMD II 上取得显著提升。

**[HierarQ: Task-Aware Hierarchical Q-Former for Enhanced Video Understanding](hierarq_task-aware_hierarchical_q-former_for_enhanced_video_understanding.md)**

:   提出 HierarQ，一种任务感知的层次化 Q-Former 框架，通过双流语言引导特征调制器（实体流 + 场景流）和短/长期记忆库实现自回归式逐帧视频处理，无需帧采样即可绕过 LLM 上下文长度限制，在 10 个视频理解基准上取得 SOTA 或接近 SOTA 的性能。

**[HuMoCon: Concept Discovery for Human Motion Understanding](humocon_concept_discovery_for_human_motion_understanding.md)**

:   HuMoCon 是一个面向人体行为分析的运动-视频理解框架，其核心创新是在编码器预训练阶段通过显式的视频-运动特征对齐和基于速度重建的高频信息保持机制来发现语义化的运动概念（codebook），从而显著提升下游 LLM 的人体运动理解和推理能力。

**[HyperGLM: HyperGraph for Video Scene Graph Generation and Anticipation](hyperglm_hypergraph_for_video_scene_graph_generation_and_anticipation.md)**

:   HyperGLM 提出将实体场景图（捕捉空间关系）和程序图（建模因果时序转换）统一为超图 (HyperGraph)，并将其注入多模态 LLM 实现视频场景图的生成、预测和推理，同时发布包含 190 万帧的 VSGR 数据集支持五类任务。

**[KeyFace: Expressive Audio-Driven Facial Animation for Long Sequences via KeyFrame Interpolation](keyface_expressive_audio-driven_facial_animation_for_long_sequences_via_keyframe.md)**

:   KeyFace 提出一个两阶段扩散框架——先以低帧率生成捕捉关键表情的锚帧，再通过插值模型填充中间帧——解决了现有音频驱动面部动画方法在长序列中身份漂移和质量退化的问题，同时首次支持连续情感（valence/arousal）建模和多种非语音发声 (NSV) 的动画生成。

**[Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval](learning_audio-guided_video_representation_with_gated_attention_for_video-text_r.md)**

:   提出 AVIGATE 框架，通过门控注意力机制选择性地融合音频与视觉信息（过滤无用音频噪声），并设计自适应间距对比损失处理视频-文本之间模糊的正负关系，在多个视频-文本检索基准上取得 SOTA。

**[Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking](learning_occlusion-robust_vision_transformers_for_real-time_uav_tracking.md)**

:   提出 ORTrack 框架，通过基于空间 Cox 过程的随机遮罩来学习遮挡鲁棒的 ViT 特征表征（训练时加遮罩约束、推理时零开销），并设计自适应特征蒸馏方法将大模型压缩为轻量级学生模型 ORTrack-D，在多个无人机跟踪基准上实现 SOTA 精度与实时速度的最佳平衡。

**[LION-FS: Fast & Slow Video-Language Thinker as Online Video Assistant](lion-fs_fast_slow_video-language_thinker_as_online_video_assistant.md)**

:   提出 LION-FS 在线视频助手框架，借鉴"快思考-慢思考"认知理论，用 Fast Path（基于路由的 Token 聚合与丢弃）实现高效实时响应判断，用 Slow Path（多粒度关键帧增强）在响应生成时注入细粒度空间和交互特征，在 Ego4D/Ego-Exo4D 基准上全面超越现有方法。

**[LLAVIDAL: A Large Language Vision Model for Daily Activities of Living](llavidal_a_large_language_vision_model_for_daily_activities_of_living.md)**

:   针对日常生活活动（ADL）理解，构建了多视角多模态指令微调数据集 ADL-X，提出 LLAVIDAL 模型融合视频、3D 骨架和 HOI 线索，采用 MMPro 渐进式训练策略实现 SOTA 性能。

**[Locality-Aware Zero-Shot Human-Object Interaction Detection](locality-aware_zero-shot_human-object_interaction_detection.md)**

:   提出 LAIN 框架，通过局部适配器（LA）和交互适配器（IA）增强 CLIP 表示的局部细粒度感知和交互推理能力，在多种零样本 HOI 检测设定下达到 SOTA。

**[Localizing Events in Videos with Multimodal Queries](localizing_events_in_videos_with_multimodal_queries.md)**

:   提出 ICQ 基准和 ICQ-Highlight 数据集，首次系统研究用多模态查询（图像+文本）替代纯文本查询进行视频事件定位，并设计 3 种查询适配方法和 SUIT 代理微调策略。

**[M-LLM Based Video Frame Selection for Efficient Video Understanding](m-llm_based_video_frame_selection_for_efficient_video_understanding.md)**

:   提出一个轻量级 M-LLM 帧选择器，通过空间和时序伪标签训练，自适应地为下游视频 LLM 选取与问题最相关的帧，无需微调下游模型即可提升多个视频 QA 基准性能。

**[MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking](mambavlt_time-evolving_multimodal_state_space_model_for_vision-language_tracking.md)**

:   首个基于 Mamba 的视觉语言跟踪器 MambaVLT，利用状态空间的时间演化特性实现长时序目标信息记忆和多模态参考特征的自适应更新，在多个视觉语言跟踪基准上达到 SOTA。

**[MITracker: Multi-View Integration for Visual Object Tracking](mitracker_multi-view_integration_for_visual_object_tracking.md)**

:   提出多视角目标跟踪数据集 MVTrack（234K 帧，27 类目标）和方法 MITracker，通过将 2D 特征投影到 3D 特征体并压缩为 BEV 平面进行跨视角融合，结合空间增强注意力修正各视角跟踪结果，实现从遮挡中快速恢复跟踪。

**[MLVU: Benchmarking Multi-task Long Video Understanding](mlvu_benchmarking_multi-task_long_video_understanding.md)**

:   提出 MLVU 基准，通过9种多样化评测任务、多种视频类型和灵活的时长设置，系统评估多模态大模型在长视频理解上的能力，揭示现有模型在处理长视频时的显著不足。

**[MMVU: Measuring Expert-Level Multi-Discipline Video Understanding](mmvu_measuring_expert-level_multi-discipline_video_understanding.md)**

:   提出 MMVU 基准，包含 3,000 个专家标注的跨 27 个学科的视频理解题目，评估多模态基础模型在专业领域视频中的专家级知识推理能力，揭示即使最强模型仍显著落后于人类专家。

**[MUST: The First Dataset and Unified Framework for Multispectral UAV Single Object Tracking](must_the_first_dataset_and_unified_framework_for_multispectral_uav_single_object.md)**

:   提出首个大规模多光谱无人机单目标跟踪数据集 MUST（250 序列、43K 帧、8 光谱波段），并设计 UNTrack 统一框架融合光谱、空间、时序特征，通过非对称 Transformer 和光谱提示编码器实现高效鲁棒跟踪。

**[Reasoning over Video: Evaluating How MLLMs Extract, Integrate, and Reconstruct Spatiotemporal Evidence](reasoning_over_video_evaluating_how_mllms_extract_integrate_and_reconstruct_spat.md)**

:   提出 VAEX-Bench 基准，首次系统评估 MLLM 的"抽象时空推理"能力——不是从单帧提取信息，而是需要跨房间/跨时间整合观察来推断全局空间布局、跨场景计数等，发现所有 SOTA 模型（包括 GPT-5.2、Gemini-3 Pro）在抽象推理上表现远低于人类。

**[VCBench: A Streaming Counting Benchmark for Spatial-Temporal State Maintenance in Long Videos](vcbench_a_streaming_counting_benchmark_for_spatial-temporal_state_maintenance_in.md)**

:   VCBench 将计数重新定位为诊断视频模型"时空状态维护"能力的最小探针，提出了覆盖物体计数（当前状态/身份追踪）和事件计数（瞬时事件/周期活动）的 8 种子类别，通过沿时间线的流式多点查询观察模型预测轨迹，在 406 个视频/4576 个查询点上评估主流模型，发现当前模型在时空状态维护上仍存在显著缺陷。

**[Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously](video_streaming_thinking_videollms_can_watch_and_think_simultaneously.md)**

:   提出 Video Streaming Thinking (VST) 范式，在视频播放过程中交替执行"看"和"想"——模型边接收视频帧边生成中间推理链，将 CoT 计算摊销到预查询阶段，从而在保持实时响应（0.56s QA延迟）的同时实现 StreamingBench 79.5% 的 SOTA。
