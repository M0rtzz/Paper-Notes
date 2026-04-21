---
title: >-
  CVPR2026 视频生成方向 54篇论文解读
description: >-
  54篇CVPR2026 视频生成方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎬 视频生成

**📷 CVPR2026** · **54** 篇论文解读

**[ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos](activityforensics_a_comprehensive_benchmark_for_localizing_manipulated_activity_.md)**

:   首次提出活动级视频伪造定位任务和ActivityForensics大规模基准数据集（6K+伪造片段），通过grounding辅助的自动化数据构造管线制造高度逼真的活动篡改，并提出Temporal Artifact Diffuser (TADiff)基线方法，通过扩散式特征正则化放大伪造线索。

**[Anti-I2V: Safeguarding your photos from malicious image-to-video generation](anti-i2v_safeguarding_your_photos_from_malicious_image-to-video_generation.md)**

:   Anti-I2V 提出了一种针对恶意图像到视频生成的防御方法，通过在 L\*a\*b\* 和频域双空间优化扰动，并设计内部表示崩塌（IRC）和锚定（IRA）损失破坏去噪网络的语义特征传播，在 CogVideoX、DynamiCrafter 和 Open-Sora 三种不同架构上实现 SOTA 防护效果。

**[AutoCut: End-to-end Advertisement Video Editing Based on Multimodal Discretization and Controllable Generation](autocut_end-to-end_advertisement_video_editing_based_on_multimodal_discretizatio.md)**

:   AutoCut 提出了一个端到端的广告视频编辑框架，通过残差向量量化（RQVAE）将视频、音频和文本统一到共享的离散 token 空间中，在 Qwen3-8B 上进行多模态对齐和监督微调，实现了视频选择、排序、脚本生成和背景音乐选择四项任务的统一处理，在多项指标上超越 GPT-4o 基线。

**[Chain of Event-Centric Causal Thought for Physically Plausible Video Generation](chain_of_event-centric_causal_thought_for_physically_plausible_video_generation.md)**

:   将物理合理性视频生成(PPVG)建模为因果连接的事件序列，通过物理公式驱动的事件链推理将复杂物理现象分解为有序事件，再通过过渡感知的跨模态提示生成语义-视觉双重条件，引导视频扩散模型生成遵循因果演进的物理现象视频。

**[Compressed-Domain-Aware Online Video Super-Resolution](compressed-domain-aware_online_video_super-resolution.md)**

:   CDA-VSR 提出利用视频压缩域信息（运动矢量、残差图、帧类型）指导在线视频超分辨率的三个关键环节：运动矢量引导的可变形对齐实现高效精准配准、残差图门控融合抑制错配区域、帧类型感知重建自适应分配计算资源，在 REDS4 上以 93 FPS（>2倍于SOTA速度）达到最优 PSNR。

**[CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video](cubecomposer_spatio-temporal_autoregressive_4k_360_video_generation_from_perspec.md)**

:   提出 CubeComposer，将360°视频分解为 cubemap 六面表示并按时空自回归方式逐面生成，首次实现从透视视频原生生成4K（3840×1920）分辨率的360°全景视频，无需后处理超分辨率。

**[Diff4Splat: Repurposing Video Diffusion Models for Dynamic Scene Generation](diff4splat_controllable_4d_scene_generation_with_latent_dynamic_reconstruction_m.md)**

:   提出 Diff4Splat，一个前馈式框架，将视频扩散模型与可变形3D高斯场统一到端到端可训练的模型中，从单张图像在约30秒内直接生成动态4D场景表示，比优化方法快60倍。

**[DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching](disca_accelerating_video_diffusion_transformers_wi.md)**

:   DisCa 首次将可学习特征缓存与步蒸馏统一为兼容框架，用轻量神经预测器（<4% 模型参数）替代手工缓存策略，配合 Restricted MeanFlow 稳定大规模视频 DiT 蒸馏，在 HunyuanVideo 上实现 11.8× 近无损加速。

**[DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching](disca_accelerating_video_diffusion_transformers_with_distillation-compatible_lea.md)**

:   提出 DisCa，首次将**可学习特征缓存**与**步骤蒸馏**相结合，通过轻量级神经预测器替代手工缓存策略，并设计 Restricted MeanFlow 稳定大规模视频模型蒸馏，在 HunyuanVideo 上实现 11.8× 加速且几乎无质量损失。

**[FastLightGen: Fast and Light Video Generation with Fewer Steps and Parameters](fastlightgen_fast_and_light_video_generation_with_fewer_steps_and_parameters.md)**

:   FastLightGen 提出三阶段蒸馏算法，首次实现采样步数与模型大小的联合蒸馏，通过识别冗余层、动态概率剪枝和 well-guided teacher guidance 分布匹配，将 HunyuanVideo/WanX 压缩为 4 步 30% 参数剪枝的轻量生成器，实现约 35 倍加速且性能超越教师模型。

**[First Frame Is the Place to Go for Video Content Customization](first_frame_is_the_place_to_go_for_video_content_customization.md)**

:   发现视频生成模型将第一帧隐式地当作「概念记忆缓冲区」来存储和复用多个视觉实体的内在能力，提出 FFGo——仅用 20-50 个训练样本的轻量级 LoRA 适配方法，无需修改架构即可激活这一能力，实现多参考物体的视频内容定制，在用户研究中 81.2% 的情况下被评为最佳。

**[FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)**

:   提出 FlashMotion，一个三阶段训练框架，将多步轨迹可控视频生成模型蒸馏为少步版本，通过混合扩散+对抗目标微调 adapter，在少步推理下同时保持视频质量和轨迹准确性。

**[FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_fewstep_controllable_video_generation.md)**

:   提出 FlashMotion，首个实现少步（4步）轨迹可控视频生成的三阶段训练框架，通过训练轨迹适配器→蒸馏快速生成器→混合对抗+扩散微调适配器的策略，在 4 步推理下同时超越现有多步方法的视觉质量和轨迹精度，实现 47 倍加速。

**[Free-Lunch Long Video Generation via Layer-Adaptive O.O.D Correction](free-lunch_long_video_generation_via_layer-adaptive_ood_correction.md)**

:   FreeLOC 提出一种免训练的层自适应框架，通过识别视频DiT中各层对"帧级相对位置OOD"和"上下文长度OOD"两种分布外问题的敏感度差异，选择性地在敏感层应用多粒度位置重编码(VRPR)和分层稀疏注意力(TSA)，在不增加训练成本的情况下实现SOTA的长视频生成质量。

**[From Static to Dynamic: Exploring Self-supervised Image-to-Video Representation Transfer Learning](from_static_to_dynamic_exploring_self-supervised_image-to-video_representation_t.md)**

:   本文提出 Co-Settle 框架，通过在冻结的图像预训练编码器上训练一个轻量线性投影层，利用时间循环一致性损失和语义可分性约束，仅需5个epoch的自监督训练即可在8个图像基础模型上一致性提升多粒度视频下游任务性能。

**[Generative Neural Video Compression via Video Diffusion Prior](generative_neural_video_compression_via_video_diffusion_prior.md)**

:   本文提出 GNVC-VD，首个基于 DiT 的生成式神经视频压缩框架，通过将视频扩散变换器作为视频原生生成先验，在统一编解码器中实现时空潜在压缩和序列级生成精炼，在极低码率（<0.03 bpp）下大幅超越传统和学习型编解码器的感知质量，并显著减少先前生成方法中的闪烁伪影。

**[Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context](geometry-as-context_modulating_explicit_3d_in_scene-consistent_video_generation_.md)**

:   提出 Geometry-as-Context (GaC) 框架，将基于重建的场景视频生成中的不可微算子（3D重建+渲染）替换为统一的自回归视频生成模型，通过将几何信息（深度图）作为交错上下文嵌入生成序列，实现端到端训练并缓解累积误差。

**[Goal-Driven Reward by Video Diffusion Models for Reinforcement Learning](goal-driven_reward_by_video_diffusion_models_for_reinforcement_learning.md)**

:   提出 GenReward 框架，利用预训练视频扩散模型生成目标条件视频，通过视频级和帧级两层目标驱动奖励信号引导强化学习智能体，无需手工设计奖励函数即可在 Meta-World 机器人操控任务上显著超越基线。

**[Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization](identity-preserving_image-to-video_generation_via_reward-guided_optimization.md)**

:   本文提出 IPRO，通过强化学习和可微分人脸身份评分器直接优化视频扩散模型，在不修改模型架构的情况下显著提升图像到视频生成中的人脸身份一致性，在 Wan 2.2 上实现了 20%-45% 的 FaceSim 提升。

**[Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout](infinity-rope_action-controllable_infinite_video_generation_emerges_from_autoreg.md)**

:   提出 ∞-RoPE，一个训练免调的推理时框架，通过 Block-Relativistic RoPE、KV Flush 和 RoPE Cut 三个组件，将仅在5秒视频上训练的自回归视频扩散模型扩展为支持无限时长生成、精细动作控制和电影级场景切换的系统。

**[I'm a Map! Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers](interpretable_motion-attentive_maps_spatio-temporally_localizing_concepts_in_vid.md)**

:   提出IMAP(可解释运动注意力图)，通过GramCol空间定位和运动头选择时序定位两个无训练模块，从Video DiT中提取运动概念的时空显著性图，在运动定位和零样本视频语义分割上超越现有方法。

**[LAMP: Language-Assisted Motion Planning for Controllable Video Generation](lamp_language-assisted_motion_planning_for_controllable_video_generation.md)**

:   提出LAMP框架，将运动控制建模为语言到程序合成问题：设计电影摄影启发的运动DSL，训练LLM将自然语言描述转化为结构化运动程序，再确定性映射为3D对象和相机轨迹来条件化视频生成，首次实现从自然语言同时生成对象和相机运动。

**[Let Your Image Move with Your Motion! – Implicit Multi-Object Multi-Motion Transfer](let_your_image_move_with_your_motion_--_implicit_multi-object_multi-motion_trans.md)**

:   本文提出 FlexiMMT，首个支持隐式多目标多运动迁移的 I2V 框架，通过运动解耦掩码注意力机制（MDMA）约束 motion/text token 仅影响对应目标区域、差异化掩码提取机制（DMEM）从扩散注意力中推导目标掩码并渐进传播，实现了精确的组合式多目标运动迁移。

**[Lighting-grounded Video Generation with Renderer-based Agent Reasoning](lighting-grounded_video_generation_with_renderer-based_agent_reasoning.md)**

:   LiVER 提出了一种光照驱动的视频生成框架，通过渲染器Agent将文本描述转化为显式3D场景代理（包含布局、光照、相机轨迹），再利用物理渲染生成diffuse/glossy/rough GGX的场景proxy，注入视频扩散模型实现物理准确的光照效果与精确场景控制。

**[LightMover: Generative Light Movement with Color and Intensity Controls](lightmover_generative_light_movement_with_color_and_intensity_controls.md)**

:   LightMover 利用视频扩散先验，将光源编辑建模为序列到序列预测问题，通过统一的控制token表示实现光源位置、颜色和亮度的精确操控，并提出自适应token剪枝机制将控制序列长度减少41%，在光源移动和物体移动任务上均超越现有方法。

**[LinVideo: A Post-Training Framework towards O(n) Attention in Efficient Video Generation](linvideo_a_post-training_framework_towards_on_attention_in_efficient_video_gener.md)**

:   提出 LinVideo，一种无需训练数据的后训练框架，通过选择性地将视频扩散模型中的二次注意力替换为线性注意力，实现 1.43–1.71× 加速，结合蒸馏可达 15.9–20.9× 加速，同时保持生成质量。

**[LinVideo: A Post-Training Framework towards O(n) Attention in Efficient Video Generation](linvideo_linear_attention_video_generation.md)**

:   首个data-free后训练框架LinVideo，通过选择性转移自动选择最适合替换为线性注意力的层+任意时刻分布匹配(ADM)目标函数高效恢复性能，实现Wan 1.3B/14B的1.43-1.71×加速且质量无损，叠加4步蒸馏后达15.9-20.9×加速。

**[MoVieDrive: Urban Scene Synthesis with Multi-Modal Multi-View Video Diffusion Transformer](moviedrive_multimodal_multiview_video_diffusion.md)**

:   首个在统一 DiT 框架下同时生成 RGB+深度+语义三模态多视图驾驶场景视频的方法，通过模态共享层（时序+多视图时空注意力）与模态特定层（跨模态交互+投影头）的分解设计+统一布局编码器+多样化条件，在 nuScenes 上 FVD 46.8（较 CogVideoX+SyntheOcc 提升 22%），深度 AbsRel 0.110，语义 mIoU 37.5，均优于独立模型生成+估计的管线。

**[MoVieDrive: Urban Scene Synthesis with Multi-Modal Multi-View Video Diffusion Transformer](moviedrive_urban_scene_synthesis_with_multi-modal_multi-view_video_diffusion_tra.md)**

:   MoVieDrive 提出统一的多模态多视图视频扩散 Transformer，通过 modal-shared + modal-specific 的双层架构设计，在单一模型中同时生成 RGB 视频、深度图和语义图，配合多样的条件输入（文本、布局、上下文参考），在 nuScenes 上取得 FVD 46.8（SOTA），同时实现跨模态一致的高质量驾驶场景合成。

**[NeoVerse: Enhancing 4D World Model with in-the-wild Monocular Videos](neoverse_enhancing_4d_world_model_with_in-the-wild_monocular_videos.md)**

:   NeoVerse 提出了一个可扩展的 4D 世界模型，通过前馈式无位姿 4DGS 重建和在线单目退化模拟，使整个训练流程可以利用海量野外单目视频（百万级），在 4D 重建和新轨迹视频生成上均达到 SOTA。

**[NOVA: Sparse Control, Dense Synthesis for Pair-Free Video Editing](nova_sparse_control_dense_synthesis_for_pair-free_video_editing.md)**

:   提出 NOVA，首次形式化"稀疏控制、密集合成"范式用于视频编辑：稀疏分支从用户编辑的多关键帧提供语义引导，密集分支从原始视频注入运动和纹理信息；配合退化模拟训练策略实现无需配对数据的学习，在编辑保真度、运动保持和时序一致性上全面超越现有方法。

**[PAM: A Pose-Appearance-Motion Engine for Sim-to-Real HOI Video Generation](pam_a_pose-appearance-motion_engine_for_sim-to-real_hoi_video_generation.md)**

:   提出PAM——首个仅需初始/目标手部姿态和物体几何即可生成逼真手物交互视频的引擎，通过解耦姿态生成、外观生成和运动生成三阶段，在DexYCB上FVD 29.13（vs InterDyn 38.83）、MPJPE 19.37mm（vs CosHand 30.05mm），生成的合成数据还能有效增强下游手部姿态估计任务。

**[PerformRecast: Expression and Head Pose Disentanglement for Portrait Video Editing](performrecast_expression_and_head_pose_disentanglement_for_portrait_video_editin.md)**

:   PerformRecast 提出了一种基于改进 3DMM 关键点变换公式的 GAN 人像视频编辑方法，通过将表情形变加在头部旋转之前（与 FLAME 模型一致）实现表情与头部姿态的精确解耦，并引入边界对齐模块解决面部/非面部区域的拼接错位问题，在表情替换和表情增强两种模式下均显著优于现有方法。

**[Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics](phantom_physics-infused_video_generation_via_joint_modeling_of_visual_and_latent.md)**

:   提出Phantom框架，在预训练视频扩散模型（Wan2.2-TI2V）之上增加一个物理动力学分支，利用V-JEPA2提取的物理感知嵌入作为潜在物理状态，通过双向交叉注意力联合建模视觉内容和物理动力学演化，在物理一致性基准上大幅超越基线（VideoPhy PC提升50.4%），同时保持视觉质量。

**[Physical Simulator In-the-Loop Video Generation](physical_simulator_in-the-loop_video_generation.md)**

:   提出PSIVG——首个将物理模拟器嵌入视频扩散生成循环的训练-free推理时框架：从模板视频中重建4D场景和物体网格，在MPM模拟器中生成物理一致轨迹，用光流引导视频生成，并通过TTCO测试时优化保证运动物体纹理一致性，用户偏好率达82.3%。

**[Rethinking Position Embedding as a Context Controller for Multi-Reference and Multi-Shot Video Generation](rethinking_position_embedding_as_a_context_controller_for_multi-reference_and_mu.md)**

:   提出 PoCo（Position Embedding as Context Controller），通过在 RoPE 中引入额外的 SideInfo 轴编码参考实体信息，解决多参考多镜头视频生成中的"参考混淆"问题——当参考图像外观高度相似时模型无法正确关联镜头与参考。在 VACE-Wan2.1-14B 框架上实现 SOTA 的跨镜头一致性（CrossShot-FaceSim 89.35，CrossShot-DINO 92.66）。

**[SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation](seeu_seeing_the_unseen_world_via_4d_dynamics-aware_generation.md)**

:   提出 SeeU，一个 2D→4D→2D 的学习框架：从稀疏单目 2D 帧重建 4D 世界表示，在低秩表示上学习连续且物理一致的 4D 动力学（B 样条参数化 + 物理约束），最后将 4D 世界重投影回 2D 并用时空上下文感知的视频生成器补全未知区域，实现跨时间和空间的未见视觉内容生成。

**[Semantic Satellite Communications for Synchronized Audiovisual Reconstruction](semantic_satellite_communications_for_synchronized.md)**

:   本文提出一种自适应多模态语义卫星传输系统，通过双流生成架构（视频驱动音频 / 音频驱动视频）灵活切换传输优先级，结合动态知识库更新机制和LLM智能体自适应决策，在严苛带宽约束下实现高保真视听同步重建。

**[Semantic Satellite Communications for Synchronized Audiovisual Reconstruction](semantic_satellite_communications_for_synchronized_audiovisual_reconstruction.md)**

:   提出面向卫星通信的自适应多模态语义传输系统，通过双流生成架构（视频驱动音频 / 音频驱动视频）实现动态模态优先级切换，结合知识库动态更新机制和 LLM 智能决策模块，在严苛带宽约束下实现高保真音视频同步重建。

**[SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation](slvmeval_synthetic_meta_evaluation_benchmark_for_text-to-long_video_generation.md)**

:   提出SLVMEval元评估基准，通过从密集视频描述数据集合成受控退化的"高质量vs低质量"视频对（最长约3小时），测试现有T2V评估系统识别长视频质量差异的能力，发现人类在10个维度上达84.7%-96.8%准确率，而现有自动评估系统在9/10维度上落后于人类。

**[StreamDiT: Real-Time Streaming Text-to-Video Generation](streamdit_real-time_streaming_text-to-video_generation.md)**

:   StreamDiT 提出了一套完整的流式视频生成方案（包括训练、建模和蒸馏），通过在 Flow Matching 中引入带渐进去噪的移动缓冲区和混合分区训练策略，结合时变 DiT 架构和窗口注意力，以及定制化的多步蒸馏方法，使 4B 参数模型在单 GPU 上达到 512p@16FPS 的实时流式视频生成。

**[SWIFT: Sliding Window Reconstruction for Few-Shot Training-Free Generated Video Attribution](swift_sliding_window_reconstruction_for_few-shot_training-free_generated_video_a.md)**

:   SWIFT 首次定义了"少样本免训练生成视频溯源"任务，利用 3D VAE 中"多帧像素↔单帧潜变量"的时间映射特性，通过固定长度滑动窗口执行正常和损坏两次重建，用重叠帧的损失比值作为溯源信号，仅需 20 个样本即可达到 90%+ 平均溯源准确率，5 模型平均 94%。

**[SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls](switchcraft_training-free_multi-event_video_generation_with_attention_controls.md)**

:   提出 SwitchCraft，一个无需训练的多事件视频生成框架，通过 Event-Aligned Query Steering (EAQS) 将帧级注意力对齐到对应事件提示、Auto-Balance Strength Solver (ABSS) 自适应平衡引导强度，在不修改模型权重的情况下实现多事件视频的清晰时序切换和场景一致性。

**[SymphoMotion: Joint Control of Camera Motion and Object Dynamics for Coherent Video Generation](symphomotion_joint_control_of_camera_motion_and_object_dynamics_for_coherent_vid.md)**

:   提出 SymphoMotion 统一运动控制框架，通过相机轨迹控制（CTC）和物体动态控制（ODC）两个机制同时精确控制视频中的相机运动和物体3D轨迹，并构建了25K规模的真实世界联合标注数据集 RealCOD-25K。

**[TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models](tear_temporal-aware_automated_red-teaming_for_text-to-video_models.md)**

:   提出 TEAR，首个针对 T2V 模型时序维度漏洞的自动化红队测试框架，通过两阶段优化的时序感知测试生成器和迭代精炼模型，生成文本上无害但能利用时序动态触发有害视频的提示，在开源和商业 T2V 模型上达到 80%+ 的攻击成功率。

**[The Devil is in the Details: Enhancing Video Virtual Try-On via Keyframe-Driven Details Injection](the_devil_is_in_the_details_enhancing_video_virtual_try-on_via_keyframe-driven_d.md)**

:   提出 KeyTailor 框架，通过关键帧驱动的细节注入策略（服装动态增强 + 协同背景优化）在不修改 DiT 架构的前提下，大幅提升视频虚拟试穿的服装保真度与背景一致性，同时发布 15K 高清数据集 ViT-HD。

**[Training-free Motion Factorization for Compositional Video Generation](training-free_motion_factorization_for_compositional_video_generation.md)**

:   提出一个运动分解框架，将场景中多实例的运动分解为静止、刚体运动和非刚体运动三类，通过结构化运动图推理（SMR）解决 prompt 的语义歧义，通过解耦运动引导（DMG）在扩散过程中针对性地调控三类运动的生成，无需额外训练即可在 VideoCrafter-v2.0 和 CogVideoX-2B 上显著提升运动多样性和保真度。

**[U-Mind: A Unified Framework for Real-Time Multimodal Interaction with Audiovisual Generation](u-mind_a_unified_framework_for_real-time_multimodal_interaction_with_audiovisual.md)**

:   提出 U-Mind，首个统一实时全栈多模态交互系统，支持高层推理对话和指令跟随，在单一交互循环中联合生成文本、语音、动作，并渲染为逼真视频，通过排练驱动学习和文本优先解码策略兼顾推理保持与跨模态对齐。

**[UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions](uniavgen_unified_audio_and_video_generation_with_asymmetric_cross-modal_interact.md)**

:   UniAVGen 提出了一个基于对称双分支 DiT 的音视频联合生成框架，通过**非对称跨模态交互机制**和**人脸感知调制模块**实现精确的时空同步，仅用 1.3M 训练样本就在唇音同步、音色一致性和情感一致性上全面超越使用 30M 数据的竞品。

**[UniTalking: A Unified Audio-Video Framework for Talking Portrait Generation](unitalking_a_unified_audio-video_framework_for_talking_portrait_generation.md)**

:   提出 UniTalking，一个基于 MM-DiT 的端到端说话人肖像生成框架，通过双流对称架构中的联合注意力机制显式建模音视频 token 的细粒度时序对应关系，实现 SOTA 的唇音同步精度，同时支持个性化语音克隆。

**[Vanast: Virtual Try-On with Human Image Animation via Synthetic Triplet Supervision](vanast_virtual_try-on_with_human_image_animation_via_synthetic_triplet_supervisi.md)**

:   Vanast 提出一种统一框架，通过 Dual Module 架构（HAM + GTM）和三阶段合成数据构建流水线，在单阶段内同时完成服装迁移和人体动画生成，在 Internet 数据集上 PSNR 达到 17.95dB（+5.5dB vs 最佳两阶段方案），LPIPS 仅 0.237。

**[VideoCoF: Unified Video Editing with Temporal Reasoner](videocof_unified_video_editing_with_temporal_reasoner.md)**

:   提出 VideoCoF，一种受 Chain-of-Thought 启发的"看→推理→编辑"视频编辑框架，通过让视频扩散模型先预测编辑区域的推理 token（灰度高亮 latent），再生成目标视频 token，在无需用户提供 mask 的前提下实现精确的指令-区域对齐，仅用 50K 视频对训练即达到 SOTA 性能，且支持 16 倍训练长度的视频外推。

**[When Numbers Speak: Aligning Textual Numerals and Visual Instances in Text-to-Video Diffusion Models](when_numbers_speak_aligning_textual_numerals_and_visual_instances_in_text-to-vid.md)**

:   NUMINA 的核心思想是，不去重训视频扩散模型，而是在推理时先从 DiT 的注意力中提取一个“可计数的实例布局”，判断数量词和当前布局是否不一致，再对布局做保守的增删修改，并用该布局回头引导重生成，从而显著提升文本到视频模型对“两个苹果、八只鸭子”这类数量约束的遵从能力。

**[When to Lock Attention: Training-Free KV Control in Video Diffusion](when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)**

:   提出 KV-Lock，基于扩散模型幻觉检测动态调度背景 KV 缓存融合比例和 CFG 引导强度，在无需训练的前提下同时保证视频编辑的背景一致性和前景生成质量。
