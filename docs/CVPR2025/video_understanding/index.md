---
title: >-
  CVPR2025 视频理解方向 16篇论文解读
description: >-
  16篇CVPR2025 视频理解方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎬 视频理解

**📷 CVPR2025** · 共 **16** 篇

**[Animateanything Consistent And Controllable Animation For Video Generation](animateanything_consistent_and_controllable_animation_for_video_generation.md)**

:   提出两阶段可控视频生成框架：第一阶段将不同控制信号（相机轨迹、用户拖拽标注、参考视频）统一转化为逐帧光流表示，第二阶段用统一光流引导基于DiT的视频扩散模型生成最终视频，并引入频域稳定模块抑制大运动下的闪烁问题。

**[Behaviorvlm Unified Finetuning-Free Behavioral Understanding With Vision-Languag](behaviorvlm_unified_finetuning-free_behavioral_understanding_with_vision-languag.md)**

:   提出 BehaviorVLM，一个统一的无需微调的视觉语言框架，通过多阶段结构化推理管线同时解决动物姿态估计和行为理解两大任务，仅需 3 帧人工标注即可实现可靠的关键点追踪，并通过深度嵌入聚类 + VLM 描述 + LLM 语义合并实现可解释的多动物行为分割。

**[Beyond Single-Sample Reliable Multi-Sample Distillation For Video Understanding](beyond_single-sample_reliable_multi-sample_distillation_for_video_understanding.md)**

:   提出 R-MSD（Reliable Multi-Sample Distillation），通过对每个输入采样多个教师响应并结合任务自适应质量匹配，解决视频 LVLM 黑盒蒸馏中单样本教师监督不可靠的问题，4B 学生模型在 VideoMME (+1.5%)、Video-MMMU (+3.2%)、MathVerse (+3.6%) 等基准上取得一致提升。

**[Bim-Vfi Bidirectional Motion Field-Guided Frame Interpolation For Video With Non](bim-vfi_bidirectional_motion_field-guided_frame_interpolation_for_video_with_non.md)**

**[Bimba Selective-Scan Compression For Long-Range Video Question Answering](bimba_selective-scan_compression_for_long-range_video_question_answering.md)**

**[Bootstrap Your Own Views Masked Ego-Exo Modeling For Fine-Grained View-Invariant](bootstrap_your_own_views_masked_ego-exo_modeling_for_fine-grained_view-invariant.md)**

:   通过掩码建模在自我中心和外部视角之间学习细粒度视图不变表示，无需配对标注即可从两种视角的关联中自监督学习

**[Can Text-To-Video Generation Help Video-Language Alignment](can_text-to-video_generation_help_video-language_alignment.md)**

:   本文首次探索利用文本到视频生成模型产生的合成视频来改善视频语言对齐（VLA）的时序理解，提出SynViTA方法通过对齐质量加权和语义一致性正则化来有效利用合成视频，在多个VLA基准上取得了平均性能提升。

**[Coarse Correspondences Boost Spatial-Temporal Reasoning In Multimodal Language M](coarse_correspondences_boost_spatial-temporal_reasoning_in_multimodal_language_m.md)**

:   本文提出Coarse Correspondences，一种轻量级的training-free视觉提示方法，通过在图像帧上叠加目标跟踪得到的粗粒度实例对应关系标记，显著增强MLLM的空间时序推理能力，在ScanQA上提升+20.5%、OpenEQA上+9.7%、EgoSchema上+6.0%和R2R导航上+11%。

**[Conmo Controllable Motion Disentanglement And Recomposition For Zero-Shot Motion](conmo_controllable_motion_disentanglement_and_recomposition_for_zero-shot_motion.md)**

:   ConMo提出了一种零样本运动迁移框架，通过将参考视频中的复合运动解耦为独立的主体运动和背景（相机）运动，再在目标视频生成时可控地重组这些运动，实现了多主体运动迁移、语义/形状变换、主体去除、相机运动模拟等多种应用，在运动保真度和文本对齐上显著超越现有方法。

**[Context-Enhanced Memory-Refined Transformer For Online Action Detection](context-enhanced_memory-refined_transformer_for_online_action_detection.md)**

:   本文揭示了现有在线动作检测（OAD）方法中的训练-推理不一致问题——短时记忆帧的不均衡上下文暴露和伪未来引入的非因果信息泄漏导致学习偏向中间帧——并提出CMeRT通过近过去上下文增强编码器和基于近未来的记忆精炼解码器来解决该问题，在THUMOS'14、CrossTask和EK100上实现SOTA。

**[Fc-Track Overlap-Aware Post-Association Correction For Online Multi-Object Track](fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)**

:   提出 FC-Track，一个轻量级的后关联校正框架，通过基于 IoA（Intersection over Area）的外观特征过滤和重叠 tracklet 对内的相似度比较，在线纠正因目标重叠导致的检测-轨迹错误匹配，将长期身份切换比例从 36.86% 降至 29.55%，同时在 MOT17/MOT20 上保持 SOTA 性能。

**[Reasoning Over Video Evaluating How Mllms Extract Integrate And Reconstruct Spat](reasoning_over_video_evaluating_how_mllms_extract_integrate_and_reconstruct_spat.md)**

:   提出 VAEX-Bench 基准，首次系统评估 MLLM 的"抽象时空推理"能力——不是从单帧提取信息，而是需要跨房间/跨时间整合观察来推断全局空间布局、跨场景计数等，发现所有 SOTA 模型（包括 GPT-5.2、Gemini-3 Pro）在抽象推理上表现远低于人类。

**[Semantic Satellite Communications For Synchronized Audiovisual Reconstruction](semantic_satellite_communications_for_synchronized_audiovisual_reconstruction.md)**

:   提出面向卫星场景的自适应多模态语义传输系统，通过双流生成架构（视频驱动音频 / 音频驱动视频）灵活切换、动态知识库更新机制和 LLM 决策代理，在极低带宽下实现高保真音视频同步重建。

**[Vcbench A Streaming Counting Benchmark For Spatial-Temporal State Maintenance In](vcbench_a_streaming_counting_benchmark_for_spatial-temporal_state_maintenance_in.md)**

:   VCBench 将计数重新定位为诊断视频模型"时空状态维护"能力的最小探针，提出了覆盖物体计数（当前状态/身份追踪）和事件计数（瞬时事件/周期活动）的 8 种子类别，通过沿时间线的流式多点查询观察模型预测轨迹，在 406 个视频/4576 个查询点上评估主流模型，发现当前模型在时空状态维护上仍存在显著缺陷。

**[Video Streaming Thinking Videollms Can Watch And Think Simultaneously](video_streaming_thinking_videollms_can_watch_and_think_simultaneously.md)**

:   提出 Video Streaming Thinking (VST) 范式，在视频播放过程中交替执行"看"和"想"——模型边接收视频帧边生成中间推理链，将 CoT 计算摊销到预查询阶段，从而在保持实时响应（0.56s QA延迟）的同时实现 StreamingBench 79.5% 的 SOTA。

**[World2Act Latent Action Post-Training Via Skill-Compositional World Models](world2act_latent_action_post-training_via_skill-compositional_world_models.md)**

:   World2Act 提出了一种基于潜在空间对齐的 VLA 后训练方法：通过对比学习将 World Model 的视频动态潜表示与 VLA 的动作表示对齐（而非在像素空间监督），并引入 LLM 驱动的技能分解流水线实现任意长度视频生成，在 RoboCasa 和 LIBERO 上以 50 条合成轨迹即达到 SOTA，真实世界提升 6.7%。
