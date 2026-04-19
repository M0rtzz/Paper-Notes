---
title: >-
  CVPR2025 视频生成方向 40篇论文解读
description: >-
  40篇CVPR2025 视频生成方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎬 视频生成

**📷 CVPR2025** · **40** 篇论文解读

**[4Real-Video: Learning Generalizable Photo-Realistic 4D Video Diffusion](4real-video_learning_generalizable_photo-realistic_4d_video_diffusion.md)**

:   提出4Real-Video，一种基于双流架构的4D视频生成框架，通过将视频token分为时间流和视角流并行处理，引入hard/soft同步层协调两流信息，约1分钟即可生成8×8的高质量时空视频网格，在视觉质量和多视角一致性上超越现有方法。

**[AnimateAnything: Consistent and Controllable Animation for Video Generation](animateanything_consistent_and_controllable_animation_for_video_generation.md)**

:   提出两阶段可控视频生成框架：第一阶段将不同控制信号（相机轨迹、用户拖拽标注、参考视频）统一转化为逐帧光流表示，第二阶段用统一光流引导基于DiT的视频扩散模型生成最终视频，并引入频域稳定模块抑制大运动下的闪烁问题。

**[Articulated Kinematics Distillation from Video Diffusion Models](articulated_kinematics_distillation_from_video_diffusion_models.md)**

:   提出 AKD，将视频扩散模型（CogVideoX-5B）的运动先验通过 SDS 蒸馏到 3D 铰链角色的关节角度参数中，结合可微分前向运动学+高斯散射渲染+物理约束，实现文本驱动的真实角色动画，VideoPhy SA 得分从 TC4D 的 0.40 提升到 0.81。

**[BF-STVSR: B-Splines and Fourier—Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution](bf-stvsr_b-splines_and_fourier---best_friends_for_high_fidelity_spatial-temporal.md)**

:   提出 BF-STVSR，结合 B 样条映射器（时间平滑插值）和傅里叶映射器（空间高频捕获）实现连续时空视频超分辨率，完全无需预训练光流网络（RAFT），在 GoPro 数据集上 PSNR 达 30.22dB，FLOPs 在所有方法中最低。

**[Can Text-to-Video Generation Help Video-Language Alignment?](can_text-to-video_generation_help_video-language_alignment.md)**

:   提出 SynViTA 框架探索文本到视频生成模型产生的合成视频能否改善视频-语言对齐（VLA），通过基于对齐质量的样本加权和语义一致性正则化解决合成视频的语义不一致和外观偏差问题，在时序挑战性任务上提升 4+ 个点。

**[ConMo: Controllable Motion Disentanglement and Recomposition for Zero-Shot Motion Transfer](conmo_controllable_motion_disentanglement_and_recomposition_for_zero-shot_motion.md)**

:   ConMo提出了一种零样本运动迁移框架，通过将参考视频中的复合运动解耦为独立的主体运动和背景（相机）运动，再在目标视频生成时可控地重组这些运动，实现了多主体运动迁移、语义/形状变换、主体去除、相机运动模拟等多种应用，在运动保真度和文本对齐上显著超越现有方法。

**[Dynamic Camera Poses and Where to Find Them](dynamic_camera_poses_and_where_to_find_them.md)**

:   提出DynPose-100K——一个包含10万个动态互联网视频及其相机位姿标注的大规模数据集，通过专用模型组合+VLM的视频过滤管线和集成最新点跟踪+动态掩码+全局BA的位姿估计管线实现。

**[DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes](dynamicscaler_seamless_and_scalable_video_generation_for_panoramic_scenes.md)**

:   DynamicScaler 提出了一个无需微调的统一框架，通过偏移移位去噪器（OSD）和全局运动引导（GMG）实现任意分辨率/宽高比的全景动态场景合成，支持常规全景和 360° 视野视频生成，同时保持恒定 VRAM 消耗。

**[Exploring Temporally-Aware Features for Point Tracking](exploring_temporally-aware_features_for_point_tracking.md)**

:   提出 Chrono，一个为点跟踪设计的时序感知特征骨干网络，通过在 DINOv2 的 Transformer 块间插入时序适配器（2D 卷积下采样 + 1D 局部时序注意力 + 2D 卷积上采样），仅通过简单的特征匹配（soft-argmax）即可在无精炼器设定下达到 SOTA 表现。

**[FADE: Frequency-Aware Diffusion Model Factorization for Video Editing](fade_frequency-aware_diffusion_model_factorization_for_video_editing.md)**

:   提出 FADE，一种免训练的视频编辑方法，通过分析 T2V 模型中各 transformer block 的频率角色（sketching vs sharpening），利用频谱引导调制在频域中分离保留与编辑内容，实现高质量的外观和运动编辑。

**[From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](from_slow_bidirectional_to_fast_autoregressive_video_diffusion_models.md)**

:   CausVid 通过非对称蒸馏将预训练的双向视频扩散 Transformer 蒸馏为因果自回归 4 步生成器，结合 ODE 初始化和 KV 缓存，实现 9.4 FPS 的流式视频生成（比 CogVideoX 快 160×），在 VBench-Long 基准上以 84.27 分排名第一。

**[Generative Inbetweening through Frame-wise Conditions-Driven Video Generation](generative_inbetweening_through_frame-wise_conditions-driven_video_generation.md)**

:   提出 FCVG，通过从两个关键帧中提取匹配线段并逐帧线性插值作为帧级条件，注入 SVD 视频生成模型，显著消解了生成式中间帧合成中前向/反向路径的模糊性，实现时序稳定的视频插帧。

**[HOIGen-1M: A Large-Scale Dataset for Human-Object Interaction Video Generation](hoigen-1m_a_large-scale_dataset_for_human-object_interaction_video_generation.md)**

:   HOIGen-1M 是首个面向人物交互 (HOI) 视频生成的百万级高质量数据集，通过高效数据筛选管线和 Mixture-of-Multimodal-Experts (MoME) 字幕策略解决了 HOI 视频数据稀缺和描述幻觉问题，并提出 CoarseHOIScore/FineHOIScore 两个评估指标来量化生成视频中交互的质量。

**[HunyuanPortrait: Implicit Condition Control for Enhanced Portrait Animation](hunyuanportrait_implicit_condition_control_for_enhanced_portrait_animation.md)**

:   HunyuanPortrait提出了首个基于Stable Video Diffusion的隐式条件肖像动画框架，通过强度感知运动编码器和ID感知多尺度适配器实现了对精细面部动态的高保真控制和强身份一致性。

**[HyperNVD: Accelerating Neural Video Decomposition via Hypernetworks](hypernvd_accelerating_neural_video_decomposition_via_hypernetworks.md)**

:   HyperNVD 提出利用超网络 (Hypernetwork) 根据 VideoMAE 编码的视频嵌入动态生成隐式神经表示 (INR) 的参数，实现跨视频的通用视频分解模型，在新视频上可比从头训练快 30+ 分钟达到相同 PSNR，同时最终性能平均提升 0.8dB。

**[Identity-Preserving Text-to-Video Generation by Frequency Decomposition](identity-preserving_text-to-video_generation_by_frequency_decomposition.md)**

:   ConsisID 提出基于频率分解的 DiT 控制方案，将人脸特征解耦为低频全局信息和高频内在身份信息，分别注入 DiT 的不同位置，实现免微调的身份保持文本到视频生成，在身份保持、文本相关性和视觉质量上全面超越现有方法。

**[IDOL: Instant Photorealistic 3D Human Creation from a Single Image](idol_instant_photorealistic_3d_human_creation_from_a_single_image.md)**

:   IDOL 通过构建包含 10 万人体的大规模多视角数据集 HuGe100K，训练基于 Transformer 的前馈模型在单张图片输入下实现即时（<1秒）的高保真可动画 3D 人体重建，在质量和泛化能力上大幅超越现有方法。

**[Improved Video VAE for Latent Video Diffusion Model](improved_video_vae_for_latent_video_diffusion_model.md)**

:   本文提出 IV-VAE，通过关键帧时序压缩架构（KTC）和组因果卷积（GCConv）解决现有视频 VAE 中图像权重初始化抑制时序压缩学习、以及因果卷积导致帧间性能不均衡的问题，在多个基准上实现 SOTA 视频重建和生成质量。

**[InterDyn: Controllable Interactive Dynamics with Video Diffusion Models](interdyn_controllable_interactive_dynamics_with_video_diffusion_models.md)**

:   InterDyn 提出将视频扩散模型作为隐式物理引擎，通过在 Stable Video Diffusion 上引入交互控制分支（ControlNet-like），从单帧图像和驱动运动信号生成物理上合理的交互动力学视频，在 Something-Something-v2 数据集上 FVD 指标超过基线 CosHand 达 77%。

**[Learning from Streaming Video with Orthogonal Gradients](learning_from_streaming_video_with_orthogonal_gradients.md)**

:   针对流式视频学习中连续帧高度相关导致梯度冗余、模型崩溃的问题，提出正交梯度优化器（Orthogonal Optimizer），通过将当前梯度投影到历史梯度的正交分量来去相关，可无缝集成到 SGD/AdamW 中，在 DoRA、VideoMAE、未来预测三个场景下均显著恢复了从打乱训练到顺序训练的性能损失。

**[Learning Temporally Consistent Video Depth from Video Diffusion Priors](learning_temporally_consistent_video_depth_from_video_diffusion_priors.md)**

:   提出 ChronoDepth——基于 Stable Video Diffusion (SVD) 的视频深度估计方法，通过在训练时为每帧独立采样噪声水平并在推理时使用无噪声前序帧作为上下文（Consistent Context-Aware Strategy），在保持空间精度的同时实现了 SOTA 的时序一致性，MFC 指标平均排名第一。

**[LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis](levitor_3d_trajectory_oriented_image-to-video_synthesis.md)**

:   LeviTor首次在image-to-video合成中引入3D物体轨迹控制，通过将物体mask用K-means聚类为少量代表点并结合深度信息作为控制信号注入SVD模型，实现了遮挡关系、前后移动和环绕等复杂3D运动的精准控制，在DAVIS上FID/FVD分别达到25.41/190.44。

**[Presto: Long Video Diffusion Generation with Segmented Cross-Attention and Content-Rich Video Data Curation](long_video_diffusion_generation_with_segmented_cross-attention_and_content-rich_.md)**

:   Presto 提出分段交叉注意力（SCA）策略，将隐状态沿时间维度分段并与对应子描述分别交叉注意力，结合精心策展的 261K 高质量长视频数据集 LongTake-HD，实现了 15 秒内容丰富且长程连贯的视频生成，在 VBench 语义得分达到 78.5%、Dynamic Degree 达到 100%。

**[LongDiff: Training-Free Long Video Generation in One Go](longdiff_training-free_long_video_generation_in_one_go.md)**

:   LongDiff 通过理论分析揭示短视频模型生成长视频时的两个关键挑战——时序位置模糊和信息稀释，并提出 Position Mapping（GROUP+SHIFT）和 Informative Frame Selection（IFS）两个简洁的时序注意力修改策略，无需训练即可让短视频模型一次性生成高质量长视频。

**[Mimir: Improving Video Diffusion Models for Precise Text Understanding](mimir_improving_video_diffusion_models_for_precise_text_understanding.md)**

:   Mimir 提出一个端到端训练框架，通过精心设计的 Token Fuser 将 decoder-only LLM（Phi-3.5）的强文本理解能力与传统 text encoder（T5）的稳定特征无损融合，显著提升视频扩散模型的文本理解精度，尤其在多物体、空间关系和时序理解上大幅领先现有方法。

**[MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling](mimo_controllable_character_video_synthesis_with_spatial_decomposed_modeling.md)**

:   MIMO 提出一种基于空间分解建模的角色视频合成框架，将 2D 视频按 3D 深度分层为人物、场景和遮挡物三个空间组件，通过解耦编码和组合解码实现了对角色身份、3D 运动和交互场景的灵活控制，在复杂运动和场景交互上显著超越先前方法。

**[Mind the Time: Temporally-Controlled Multi-Event Video Generation](mind_the_time_temporally-controlled_multi-event_video_generation.md)**

:   提出 MinT，首个支持事件时间控制的多事件视频生成器，通过 Rescaled RoPE (ReRoPE) 位置编码将事件描述绑定到特定时间段，在预训练视频 DiT 上微调实现平滑连贯的多事件视频合成。

**[MotiF: Making Text Count in Image Animation with Motion Focal Loss](motif_making_text_count_in_image_animation_with_motion_focal_loss.md)**

:   提出 Motion Focal Loss (MotiF)，通过光流生成运动热力图对扩散损失进行空间加权，引导模型关注高运动区域，显著提升 Text-Image-to-Video 生成中的文本遵循和运动质量，并构建 TI2V-Bench 评测基准。

**[Motion Modes: What Could Happen Next?](motion_modes_what_could_happen_next.md)**

:   提出 Motion Modes，一种免训练方法，通过设计四种引导能量函数探索预训练图像到视频生成器的潜在分布，从单张图像中发现物体的多种合理且多样的运动模式，同时将物体运动与相机运动解耦。

**[Motion Prompting: Controlling Video Generation with Motion Trajectories](motion_prompting_controlling_video_generation_with_motion_trajectories.md)**

:   将时空稀疏/稠密点轨迹作为"运动提示"训练ControlNet，用单一模型实现物体控制、相机控制、运动迁移、拖拽编辑等多种运动控制能力，并展现出逼真物理行为的涌现特性。

**[MotionPro: A Precise Motion Controller for Image-to-Video Generation](motionpro_a_precise_motion_controller_for_image-to-video_generation.md)**

:   提出 MotionPro，利用区域级轨迹（region-wise trajectory）和运动掩码（motion mask）双重信号，实现细粒度、可区分物体/相机运动的精确可控图像到视频生成。

**[MotionStone: Decoupled Motion Intensity Modulation with Diffusion Transformer for Image-to-Video Generation](motionstone_decoupled_motion_intensity_modulation_with_diffusion_transformer_for.md)**

:   提出 MotionStone，通过训练独立的运动强度估计器将视频运动解耦为物体运动和相机运动两个维度，并以解耦方式注入 Diffusion Transformer，实现精细的运动强度可控 I2V 生成。

**[Multi-subject Open-set Personalization in Video Generation](multi-subject_open-set_personalization_in_video_generation.md)**

:   提出 Video Alchemist，在 Diffusion Transformer 架构中内置多主体、开放集的视频个性化生成能力，支持前景物体和背景的定制，无需测试时优化。

**[Navigation World Models](navigation_world_models.md)**

:   本文提出Navigation World Model (NWM)，一个10亿参数的Conditional Diffusion Transformer (CDiT)，在多个机器人导航数据集和Ego4D无标签视频上联合训练，通过预测给定动作下的未来视觉观测来模拟导航轨迹，可用于MPC规划或对外部策略（如NoMaD）的轨迹排序，在RECON数据集上的ATE（1.13）和RPE（0.35）均显著优于现有导航策略。

**[NeuS-V: Neuro-Symbolic Evaluation of Text-to-Video Models using Formal Verification](neuro-symbolic_evaluation_of_text-to-video_models_using_formal_verification.md)**

:   提出 NeuS-V，首个用形式化验证（时序逻辑+概率模型检验）评估文本到视频（T2V）模型时序一致性的框架——将文本提示转为时序逻辑规范，用 VLM 评分原子命题，构建视频自动机后形式化验证满足概率，在 Gen-3 上与人类标注 Pearson 相关 0.71（VBench 仅 0.47）。

**[SAW: Toward a Surgical Action World Model via Controllable and Scalable Video Generation](saw_toward_a_surgical_action_world_model_via_controllable_and_scalable_video_gen.md)**

:   提出 SAW（Surgical Action World），通过四种轻量级条件信号（语言提示、参考帧、组织功能图、工具轨迹）驱动视频扩散模型，实现可控、可扩展的手术动作视频生成，用于罕见动作增强和手术仿真。

**[Semantic Satellite Communications for Synchronized Audiovisual Reconstruction](semantic_satellite_communications_for_synchronized_audiovisual_reconstruction.md)**

:   提出一种LLM驱动的自适应多模态语义卫星传输系统，通过双流生成架构（V2A/A2V）灵活切换视频驱动音频生成和音频驱动视频生成，结合动态知识库更新机制，在极低带宽卫星链路上实现高保真音视频同步重建。

**[ViReS: Video Instance Repainting via Sketch and Text Guided Generation](vires_video_instance_repainting_via_sketch_and_text_guided_generation.md)**

:   提出ViReS框架，通过草图和文本双重引导实现视频中特定实例的重绘，利用时序注意力和实例掩码保持背景不变和时间一致性，在多种视频编辑场景下生成高质量结果。

**[When to Lock Attention: Training-Free KV Control in Video Diffusion](when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)**

:   提出 KV-Lock，一种基于扩散幻觉检测的免训练视频编辑框架，通过动态调度 KV 缓存融合比例和 CFG 引导尺度，在保持背景一致性的同时增强前景生成质量。

**[World2Act: Latent Action Post-Training via Skill-Compositional World Models](world2act_latent_action_post-training_via_skill-compositional_world_models.md)**

:   World2Act 提出了一种基于潜在空间对齐的 VLA 后训练方法：通过对比学习将 World Model 的视频动态潜表示与 VLA 的动作表示对齐（而非在像素空间监督），并引入 LLM 驱动的技能分解流水线实现任意长度视频生成，在 RoboCasa 和 LIBERO 上以 50 条合成轨迹即达到 SOTA，真实世界提升 6.7%。
