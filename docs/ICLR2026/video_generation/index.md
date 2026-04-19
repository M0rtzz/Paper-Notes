---
title: >-
  ICLR2026 视频生成方向 18篇论文解读
description: >-
  18篇ICLR2026 视频生成方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎬 视频生成

**🔬 ICLR2026** · **18** 篇论文解读

**[Arbitrary Generative Video Interpolation](arbitrary_generative_video_interpolation.md)**

:   ArbInterp 提出了一种支持任意时间戳、任意长度的生成式视频帧插值框架，通过时间戳感知旋转位置编码（TaRoPE）实现精准时间控制，并通过外观-运动解耦的条件注入策略实现长序列的无缝拼接。

**[BindWeave: Subject-Consistent Video Generation via Cross-Modal Integration](bindweave_subject-consistent_video_generation_via_cross-modal_integration.md)**

:   BindWeave 用多模态大语言模型（MLLM）替代传统的浅层融合机制来解析多主体复杂文本指令，生成主体感知的隐状态作为 DiT 的条件信号，结合 CLIP 语义特征和 VAE 细粒度外观特征，实现高保真、主体一致的视频生成。

**[Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)**

:   提出 Frame Guidance，一种无需训练的帧级引导方法，通过 latent slicing（降低 60× 显存）和 Video Latent Optimization（VLO）两个核心组件，在不修改模型的情况下实现关键帧引导、风格化和循环视频等多种可控视频生成任务。

**[Geometry-aware 4D Video Generation for Robot Manipulation](geometry-aware_4d_video_generation_for_robot_manipulation.md)**

:   本文提出几何感知的4D视频生成框架，通过跨视角点图对齐监督训练视频扩散模型，联合预测RGB和点图实现时空一致的多视角RGB-D视频，无需相机位姿输入即可在新视角下生成一致视频并用现成6DoF位姿追踪器恢复机器人末端轨迹。

**[JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization](javisdit_joint_audio-video_diffusion_transformer_with_hierarchical_spatio-tempor.md)**

:   提出 JavisDiT，基于 DiT 架构的音视频联合生成模型，通过层级化时空同步先验估计器（HiST-Sypo）实现细粒度的音视频时空对齐；同时构建了新基准 JavisBench（10K 复杂场景样本）和新评估指标 JavisScore。

**[JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation](javisdit_unified_modeling_and_optimization_for_joint_audio-video_generation.md)**

:   提出 JavisDiT++，一个面向联合音视频生成（JAVG）的简洁统一框架，通过模态特定 MoE 提升生成质量、时间对齐 RoPE 实现帧级同步、音视频 DPO 对齐人类偏好，基于 Wan2.1-1.3B 仅用约 1M 公开数据即达到 SOTA。

**[Language-guided Open-world Video Anomaly Detection under Weak Supervision](language-guided_open-world_video_anomaly_detection_under_weak_supervision.md)**

:   提出语言引导的开放世界视频异常检测范式LaGoVAD，通过将异常定义建模为随机变量并以自然语言形式输入，结合动态视频合成和对比学习正则化策略，在七个数据集上实现零样本SOTA性能。

**[Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control](learning_video_generation_for_robotic_manipulation_with_collaborative_trajectory.md)**

:   提出RoboMaster框架，通过协作轨迹（collaborative trajectory）将机器人-物体交互过程分解为前交互、交互中、后交互三阶段，结合外观和形状感知的物体嵌入，实现高质量的机器人操作视频生成。

**[LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning](lora-edit_controllable_first-frame-guided_video_editing_via_mask-aware_lora_fine.md)**

:   提出 LoRA-Edit，利用时空 mask 引导 LoRA 微调预训练 I2V 模型，实现可控的首帧引导视频编辑——mask 同时作为编辑区域指令和 LoRA 学习内容的引导信号，支持运动继承和外观控制。

**[Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](lumos-1_on_autoregressive_video_generation_with_discrete_diffusion_from_a_unifie.md)**

:   提出 Lumos-1，一个基于 LLM 架构的统一视频生成模型：通过 MM-RoPE（分布式多模态 RoPE）解决视觉时空编码问题，通过 AR-DF（自回归离散扩散强迫）解决帧间损失不均衡问题，仅用 48 GPU 训练即可在 GenEval、VBench-I2V 和 VBench-T2V 上达到竞争力水平。

**[MoSA: Motion-Coherent Human Video Generation via Structure-Appearance Decoupling](mosa_motion-coherent_human_video_generation_via_structure-appearance_decoupling.md)**

:   提出 MoSA 框架，将人体视频生成拆分为"结构生成"（3D Transformer 先生成物理合理的运动骨骼）和"外观生成"（DiT 在骨骼引导下合成视频），并设计人体感知动态控制（HADC）模块将稀疏骨骼信号扩展到整个运动区域，配合密集跟踪损失和接触约束，在 FVD、CLIPSIM 等指标上全面超越 HunyuanVideo、Wan 2.1 等 SOTA。

**[MotionStream: Real-Time Video Generation with Interactive Motion Controls](motionstream_real-time_video_generation_with_interactive_motion_controls.md)**

:   提出MotionStream——首个运动控制的实时流式视频生成系统：先训练轻量track head的双向运动控制teacher，再通过Self Forcing + DMD蒸馏为因果student，引入注意力沉降（attention sink）+滚动KV缓存（rolling KV cache）实现训练-推理分布完全匹配，单H100 GPU上480P达17FPS/29FPS（+Tiny VAE），支持无限长度恒速生成。

**[PreciseCache: Precise Feature Caching for Efficient and High-fidelity Video Generation](precisecache_precise_feature_caching_for_efficient_and_high-fidelity_video_gener.md)**

:   提出 PreciseCache——精确检测并跳过视频生成中真正冗余计算的即插即用加速框架，由 LFCache（步级，基于低频差异 LFD 度量）和 BlockCache（块级，基于输入输出差异度量）组成，在 Wan2.1-14B 等主流模型上实现平均 2.6× 加速且无明显质量损失。

**[QuantSparse: Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification](quantsparse_comprehensively_compressing_video_diffusion_transformer_with_model_q.md)**

:   本文提出 QuantSparse 框架，首次将模型量化（quantization）与注意力稀疏化（attention sparsification）协同整合用于视频扩散 Transformer 压缩，通过多尺度显著注意力蒸馏（MSAD）和二阶稀疏注意力重参数化（SSAR）解决两者朴素结合导致的"放大注意力偏移"问题，在 HunyuanVideo-13B 上以 W4A8 + 15% 注意力密度实现 3.68× 存储压缩和 1.88× 推理加速，同时几乎无损保持生成质量。

**[SIGMark: Scalable In-Generation Watermark with Blind Extraction for Video Diffusion](sigmark_scalable_in-generation_watermark_with_blind_extraction_for_video_diffusi.md)**

:   SIGMark提出首个面向现代视频扩散模型的盲水印框架，通过全局帧级伪随机编码(GF-PRC)实现恒定提取成本的可扩展盲水印，并设计分段组排序(SGO)模块应对因果3D VAE下的时序扰动，在HunyuanVideo和Wan-2.2上实现高bit精度与强鲁棒性。

**[Streaming Autoregressive Video Generation via Diagonal Distillation](streaming_autoregressive_video_generation_via_diagonal_distillation.md)**

:   提出Diagonal Distillation（DiagDistill），通过对角线去噪策略（前段多步、后段少步）和流分布匹配损失，实现流式自回归视频生成的277.3倍加速，达到31 FPS实时生成。

**[Target-Aware Video Diffusion Models](target-aware_video_diffusion_models.md)**

:   提出 target-aware 视频扩散模型，仅需一张输入图像和目标物体的分割 mask，即可生成演员与指定目标交互的视频；核心创新是引入 [TGT] 特殊 token 并设计选择性交叉注意力损失，使模型关注目标的空间位置，在目标对齐和视频质量上全面超越基线。

**[TTOM: Test-Time Optimization and Memorization for Compositional Video Generation](ttom_test-time_optimization_and_memorization_for_compositional_video_generation.md)**

:   提出 TTOM 框架，在推理时通过优化新增参数将视频生成模型的注意力与 LLM 生成的时空布局对齐，并用参数记忆机制保存历史优化上下文支持复用，在 T2V-CompBench 上相对提升 34%（CogVideoX）和 14%（Wan2.1）。
