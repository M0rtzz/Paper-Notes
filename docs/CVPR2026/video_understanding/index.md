---
title: >-
  CVPR2026 视频理解方向77篇论文解读
description: >-
  77篇CVPR2026的视频理解方向论文解读，涵盖目标跟踪、Agent、压缩/编码、推理、多模态、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "视频理解"
  - "论文解读"
  - "论文笔记"
  - "目标跟踪"
  - "Agent"
  - "压缩/编码"
  - "推理"
  - "多模态"
  - "对抗鲁棒"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📹 视频理解

**📷 CVPR2026** · **77** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (8)](../../ICML2026/video_understanding/index.md) · [💬 ACL2026 (11)](../../ACL2026/video_understanding/index.md) · [🔬 ICLR2026 (22)](../../ICLR2026/video_understanding/index.md) · [🤖 AAAI2026 (33)](../../AAAI2026/video_understanding/index.md) · [🧠 NeurIPS2025 (59)](../../NeurIPS2025/video_understanding/index.md) · [📹 ICCV2025 (57)](../../ICCV2025/video_understanding/index.md)

🔥 **高频主题：** 目标跟踪 ×16 · Agent ×7 · 压缩/编码 ×7 · 推理 ×6 · 多模态 ×4

**[A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning](a_multi-agent_perception-action_alliance_for_efficient_long_video_reasoning.md)**

:   提出 A4VL，一个无训练的多智能体感知-行动联盟框架，通过事件驱动视频分块、线索引导的关键帧选择和多轮智能体协商剪枝机制，在五个视频问答基准上以显著更低的推理延迟全面超越 28 个基线方法。

**[AdaSpark: Adaptive Sparsity for Efficient Long-Video Understanding](adaspark_adaptive_sparsity_for_efficient_long_video_understanding.md)**

:   提出 AdaSpark，通过 3D 时空 cube 分区和两个协同的自适应稀疏机制（cube 级注意力选择 + token 级 FFN 选择），将长视频处理 FLOPs 降低最多 57% 同时保持性能。

**[AutoGaze: Attend Before Attention — Efficient and Scalable Video Understanding via Autoregressive Gazing](autogaze_attend_before_attention_efficient_video.md)**

:   提出 AutoGaze，一个仅 3M 参数的轻量模块，通过自回归地选择最小化重建损失的多尺度 patch 集合，在 ViT 之前移除视频中的冗余信息，实现 4×~100× 的 token 压缩和最高 19× 的 ViT 加速，使 MLLM 能够扩展至 1K 帧 4K 分辨率视频并在 VideoMME 上达到 67.0%。

**[Beyond Single-Sample: Reliable Multi-Sample Distillation for Video Understanding](beyond_single-sample_reliable_multi-sample_distillation_for_video_understanding.md)**

:   揭示视频 LVLM 黑盒蒸馏中单样本 teacher 响应存在严重不可靠性（跨问题方差 σ=0.22、采样内方差 σ=0.07~0.15、格式违规 1%~10%），提出 R-MSD 框架通过多样本 teacher pool + 任务自适应匹配 + 两阶段 SFT→RL 对抗蒸馏解决该问题，4B student 在 VideoMME/Video-MMMU/WorldSense 上全面超越同规模 Qwen3-VL-4B。

**[Temporally Consistent Long-Term Memory for 3D Single Object Tracking](chronotrack_temporally_consistent_long_term_memory_for_3d_single_object_tracking.md)**

:   提出 ChronoTrack，通过紧凑的可学习记忆 token 和两个互补目标（时间一致性损失 + 记忆循环一致性损失）构建鲁棒的长程 3D 单目标跟踪框架，在多个基准上达到 SOTA 并以 42 FPS 实时运行。

**[CineSRD: Leveraging Visual, Acoustic, and Linguistic Cues for Open-World Visual Media Speaker Diarization](cinesrd_leveraging_visual_acoustic_and_linguistic_cues_for_open-world_visual_med.md)**

:   提出 CineSRD，一个免训练的多模态说话人分离框架，通过视觉锚点聚类进行说话人注册，结合音频语言模型进行说话人转换检测，解决影视作品中长视频、大量角色、音视频不同步等开放世界挑战。

**[CLCR: Cross-Level Semantic Collaborative Representation for Multimodal Learning](clcr_cross-level_semantic_collaborative_representation_for_multimodal_learning.md)**

:   提出 CLCR 框架，将每个模态特征组织为三层语义层级（浅/中/深），通过层内受控交换域（IntraCED）限制跨模态交互仅在共享子空间进行，通过层间协同聚合域（InterCAD）实现跨层自适应融合，解决多模态学习中的跨层语义不同步问题。

**[Cluster-Wise Spatio-Temporal Masking for Efficient Video-Language Pretraining](cluster-wise_spatio-temporal_masking_for_efficient_video-language_pretraining.md)**

:   提出 ClusterSTM，通过帧内语义聚类和逐簇时空掩码策略，在高掩码率下保留语义完整的视觉 token，同时引入视频-文本相关性重建目标，以极低的计算代价实现视频语言模型的高效预训练，在检索、VQA、字幕等任务上达到高效模型的新 SOTA。

**[Color When It Counts: Grayscale-Guided Online Triggering for Always-On Streaming Video Sensing](color_when_it_counts_grayscale-guided_online_triggering_for_always-on_streaming_.md)**

:   提出"灰度常开、彩色按需"新范式，通过 ColorTrigger 在灰度流上用轻量二次规划在线检测色彩冗余，仅使用 8.1% 的 RGB 帧即保持全彩基线 91.6% 的性能，实现资源受限设备的 always-on 视频感知。

**[CVA: Context-aware Video-text Alignment for Video Temporal Grounding](cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)**

:   提出 CVA（Context-aware Video-text Alignment）框架，通过 Query-aware Context Diversification（QCD）、Context-invariant Boundary Discrimination（CBD）损失和 Context-enhanced Transformer Encoder（CTE）三个协同组件，解决视频时序定位中的假阴性和背景关联问题，在 QVHighlights 上 R1@0.7 提升约 5 个点。

**[Decompose and Transfer: CoT-Prompting Enhanced Alignment for Open-Vocabulary Temporal Action Detection](decompose_and_transfer_cot-prompting_enhanced_alignment_for_open-vocabulary_temp.md)**

:   提出 Phase-wise Decomposition and Alignment (PDA) 框架，利用 LLM 的 CoT 推理能力将动作标签分解为"开始-中间-结束"三个阶段描述，通过文本引导的前景过滤和自适应阶段对齐实现细粒度动作模式迁移，在 THUMOS14 OV-TAD 上 Avg mAP 达 46.9（超越 SOTA Ti-FAD 的 41.2）。

**[DIvide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding](divide_then_ground_adapting_frame_selection_to_query_types_for_long-form_video_u.md)**

:   提出 DIG，一个免训练的帧选择框架，通过将查询分为全局查询和定位查询两类，对全局查询使用均匀采样、对定位查询使用一套专门的内容自适应帧选择+LMM奖励评分+视频精炼流水线，在三个长视频理解基准上持续超越现有方法。

**[Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering](do_you_see_what_i_am_pointing_at_gesture-based_egocentric_video_question_answeri.md)**

:   提出 EgoPointVQA 数据集和 HINT（Hand Intent Tokens）方法，通过将 3D 手部关键点编码为手意图 token 并与视觉 token 交错输入 MLLM，解决第一人称视频中基于手势指向的指示性问答任务，HINT-14B 达 68.1% 准确率超越 InternVL3-14B 5.4pp。

**[Drift-Resilient Temporal Priors for Visual Tracking](drift-resilient_temporal_priors_for_visual_tracking.md)**

:   提出 DTPTrack——一个轻量即插即用的时序建模模块，通过时序可靠性校准器（TRC）为历史帧分配可靠性分数过滤噪声，并通过时序引导合成器（TGS）将校准后的历史信息合成为动态先验 token 抑制跟踪漂移，在多个基准上达到 SOTA。

**[Dual-Agent Reinforcement Learning for Adaptive and Cost-Aware Visual-Inertial Odometry](dual-agent_reinforcement_learning_for_adaptive_and_cost-aware_visual-inertial_od.md)**

:   提出双智能体强化学习框架，通过 Select Agent（基于IMU信号决定是否启动视觉前端）和 Fusion Agent（自适应融合视觉-惯性状态）两个轻量RL策略，在不完全移除VIBA的前提下大幅降低其调用频率和计算开销，实现精度-效率-显存的更优折中。

**[EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions](egoxtreme_a_dataset_for_robust_object_pose_estimation_in_egocentric_views_under_.md)**

:   提出 EgoXtreme，首个面向极端条件下第一人称视角的大规模 6D 物体位姿估计基准数据集，涵盖严重运动模糊、动态光照和烟雾遮挡三种真实挑战，揭示了当前 SOTA 位姿估计器在这些条件下的严重失效。

**[Enhancing Accuracy of Uncertainty Estimation in Appearance-based Gaze Tracking with Probabilistic Evaluation and Calibration](enhancing_accuracy_of_uncertainty_estimation_in_appearance-based_gaze_tracking_w.md)**

:   提出一种数据高效的后验校准方法，通过等保序回归将不确定性感知视线追踪模型的预测分布与真实观测分布对齐，并引入 Coverage Probability Error (CPE) 指标替代不可靠的误差-不确定性相关性(EUC)来评估不确定性质量。

**[Envisioning the Future, One Step at a Time](envisioning_the_future_one_step_at_a_time.md)**

:   本文将开放集未来场景动态预测建模为稀疏点轨迹的逐步推理，通过自回归扩散模型实现从单张图像快速生成数千种多样化未来假设，速度比稠密模型快数个数量级。

**[Event6D: Event-based Novel Object 6D Pose Tracking](event6d_event-based_novel_object_6d_pose_tracking.md)**

:   EventTrack6D 提出事件-深度融合的 6D 位姿追踪框架，通过在任意时间戳重建强度和深度图像来弥补事件相机与深度帧率的差异，在仅合成数据训练的条件下以 120+ FPS 实现了对未见目标的鲁棒追踪。

**[FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)**

:   提出 FC-Track，一种轻量级的后关联校正框架，通过基于 IoA（Intersection over Area）的重叠感知外观特征过滤和局部不匹配重分配策略，在在线 MOT 中显式纠正由目标重叠引起的身份切换错误，将长期身份切换比例降至 29.55%。

**[FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)**

:   提出 FluxMem，一个无需训练的流式视频理解框架，通过层级化记忆设计（短期/中期/长期）和两个自适应 token 压缩模块（TAS 去时间冗余 + SDC 去空间冗余），在丢弃 60-70% 视觉 token 的同时在 StreamingBench 和 OVO-Bench 上取得新 SOTA。

**[Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding](frame2freq_spectral_adapters_for_fine-grained_video_understanding.md)**

:   提出 Frame2Freq——首个在频域进行时序建模的 PEFT 适配器族，通过 FFT 将冻结 VFM 的帧嵌入变换到频谱空间并学习频带级滤波，在五个细粒度动作识别基准上以 <10% 的可训练参数超越全量微调模型。

**[GoalForce: Teaching Video Models to Accomplish Physics-Conditioned Goals](goal_force_teaching_video_models_to_accomplish_physics-conditioned_goals.md)**

:   提出 Goal Force 框架，通过多通道物理控制信号（目标力、直接力、质量）在简单合成数据上训练视频生成模型，使其学会从目标效果逆向规划因果链，实现零样本泛化到工具使用、人-物交互等复杂现实场景。

**[Hear What Matters! Text-conditioned Selective Video-to-Audio Generation](hear_what_matters_text-conditioned_selective_video-to-audio_generation.md)**

:   SelVA 提出了文本条件的选择性视频到音频（V2A）生成任务，通过可学习的补充 token [SUP] 和自监督视频混合策略，使模型能够根据文本提示从多声源视频中仅生成用户指定的目标声音，在音频质量、语义对齐和时间同步上均超越现有方法。

**[HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering](herbench_a_benchmark_for_multi-evidence_integration_in_video_question_answering.md)**

:   HERBench 是一个专为多证据整合设计的视频问答基准，包含 26,806 个五选一问题，每题结构性地要求融合 ≥3 个时间分散的非重叠视觉线索；通过提出最小必需帧集（MRFS）指标揭示了当前 Video-LLM 的两个关键瓶颈：帧检索不足和证据融合失败。

**[HieraMamba: Video Temporal Grounding via Hierarchical Anchor-Mamba Pooling](hieramamba_video_temporal_grounding_via_hierarchical_anchor-mamba_pooling.md)**

:   HieraMamba 提出了基于 Mamba 的层次化视频时间定位架构，核心是 Anchor-MambaPooling（AMP）模块，用 Mamba 的选择性扫描将视频特征逐层压缩为多尺度锚点 token，配合 anchor-conditioned 和 segment-pooled 对比损失增强层次表示的紧凑性和判别性，在 Ego4D-NLQ、MAD 和 TACoS 上达到 SOTA。

**[How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms](how_should_video_llms_output_time.md)**

:   本文在统一框架下对比了视频时序定位的三种主流时间输出范式（文本数字生成、时间token生成、连续时间解码），发现连续分布范式在效率-精度帕累托前沿上始终表现最优。

**[LAOF: Robust Latent Action Learning with Optical Flow Constraints](laof_robust_latent_action_learning_with_optical_flow_constraints.md)**

:   提出LAOF框架，利用智能体的光流作为伪监督信号约束潜动作学习，使潜动作表示对干扰更鲁棒，在LIBERO和PROCGEN上显著超越无监督基线，且在无标签条件下匹配或超越使用1%动作标签的监督方法。

**[Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning](learning_to_assist_physics-grounded_human-human_control_via_multi-agent_reinforc.md)**

:   提出 AssistMimic，将人-人辅助交互动作的物理模仿建模为多智能体强化学习（MARL）问题，通过运动先验初始化、动态参考重定向和接触促进奖励，首次实现了力交换型辅助动作的物理仿真跟踪。

**[LensWalk: Agentic Video Understanding by Planning How You See in Videos](lenswalk_agentic_video_understanding_by_planning_how_you_see_in_videos.md)**

:   提出LensWalk，一个让LLM推理器主动控制视频观测范围和采样密度的智能体框架，通过reason-plan-observe循环实现自适应视频理解，无需微调即可在长视频基准上带来5%以上的即插即用性能提升。

**[LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding](longvideo-r1_smart_navigation_for_low-cost_long_video_understanding.md)**

:   提出 LongVideo-R1，一个配备推理能力的多模态 Agent，通过层次化视频树结构和智能导航策略，以平均仅 10.5 轮工具调用实现高效长视频问答，在精度-效率权衡上显著优于穷举式方法。

**[Mamba-VMR: Multimodal Query Augmentation via Generated Videos for Precise Temporal Grounding](mamba-vmr_multimodal_query_augmentation_via_generated_videos_for_precise_tempora.md)**

:   提出一个两阶段视频时刻检索框架：第一阶段用LLM引导字幕匹配并生成辅助短视频作为时序先验，第二阶段用多模态控制Mamba网络高效融合生成先验与长序列，在TVR数据集上超越SOTA（R@1/IoU=0.5达45.20%），同时降低计算开销。

**[MaskAdapt: Learning Flexible Motion Adaptation via Mask-Invariant Prior for Physics-Based Characters](maskadapt_learning_flexible_motion_adaptation_via_mask-invariant_prior_for_physi.md)**

:   本文提出 MaskAdapt 框架，通过两阶段残差学习范式——先训练遮蔽不变的鲁棒基础策略，再训练冻结基础控制器上的残差策略来修改目标身体部位——实现灵活精准的物理仿真人形角色动作适应。

**[MINERVA-Cultural: A Benchmark for Cultural and Multilingual Long Video Reasoning](minerva-cultural_a_benchmark_for_cultural_and_multilingual_long_video_reasoning.md)**

:   提出 MINERVA-Cultural 基准，包含 18 个语种/地区的 2400 个人工标注视频推理问题，通过证据图（evidence graph）和迭代错误隔离策略揭示当前 SOTA Video-LLM 在文化视觉感知上的严重不足（最强模型 Gemini-2.5-Pro 仅 45.07% vs 人类 95.22%）。

**[Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos](mistake_attribution_fine-grained_mistake_understanding_in_egocentric_videos.md)**

:   本文提出 Mistake Attribution (MATT) 任务，将第一人称视频中的操作错误归因到语义（违反了指令的哪个成分）、时间（不可逆转点 PNR 在哪一帧）和空间（PNR 帧中错误区域在哪里）三个维度，通过 MisEngine 数据引擎自动从已有动作数据集构建大规模错误样本，并设计统一的 Transformer 模型 MisFormer 同时完成三个归因子任务，在多个基准上超越各子任务的专用 SOTA 方法。

**[MovieRecapsQA: A Multimodal Open-Ended Video Question-Answering Benchmark](movierecapsqa_a_multimodal_open-ended_video_question-answering_benchmark.md)**

:   提出 MovieRecapsQA，一个基于电影解说视频构建的多模态开放式视频问答基准，包含 60 部电影的约 8.2K 个问题，并设计了基于原子事实 (atomic facts) 的无参考评估指标，揭示了当前 MLLM 在视觉感知而非推理上的关键瓶颈。

**[Ninja Codes: Neurally Generated Fiducial Markers for Stealthy 6-DoF Tracking](ninja_codes_neurally_generated_fiducial_markers_for_stealthy_6-dof_tracking.md)**

:   Ninja Codes 利用深度隐写术技术，通过端到端训练的编码器将任意图像转化为视觉上不显眼的基准标记，可用标准打印机打印并用RGB相机检测，实现隐蔽的6-DoF位置追踪。

**[Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking](occlusion-aware_sort_observing_occlusion_for_robust_multi-object_tracking.md)**

:   提出遮挡感知跟踪框架 OA-SORT，通过显式建模目标遮挡状态来缓解位置代价混淆和 Kalman Filter 估计不稳定问题，在 DanceTrack/SportsMOT/MOT17 上均取得 SOTA 级提升，且组件可即插即用地集成到多种跟踪器中。

**[OpenMarcie: Dataset for Multimodal Action Recognition in Industrial Environments](openmarcie_dataset_for_multimodal_action_recognition_in_industrial_environments.md)**

:   提出目前最大规模的工业场景多模态动作识别数据集 OpenMarcie，融合可穿戴传感器与视觉数据共 8 种模态、200+ 通道、37+ 小时录制，并在 HAR 分类、开放词表描述、跨模态对齐三个基准上验证了惯性+视觉融合的优越性。

**[Out of Sight, Out of Track: Adversarial Attacks on Propagation-based Multi-Object Trackers via Query State Manipulation](out_of_sight_out_of_track_adversarial_attacks_on_propagation-based_multi-object_.md)**

:   首次系统分析 Tracking-by-Query-Propagation（TBP）跟踪器的对抗脆弱性，提出 FADE 攻击框架，通过时序查询洪泛（TQF）耗尽固定查询预算和时序记忆腐蚀（TMC）破坏隐状态传播两种策略，在 MOT17/MOT20 上对 MOTR/MOTRv2/MeMOTR/Samba/CO-MOT 造成最高约 30 点 HOTA 下降和 10 倍以上身份切换。

**[Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding](question-guided_visual_compression_with_memory_feedback_for_long-term_video_unde.md)**

:   提出 QViC-MF 框架，通过问题引导的多帧视觉压缩（QMSA）和上下文记忆反馈机制，在长视频理解任务上以极少的视觉 token（每帧仅 16 个）实现了 MLVU/LVBench/VNBench 等多个基准上的 SOTA。

**[RAGTrack: Language-aware RGBT Tracking with Retrieval-Augmented Generation](ragtrack_language-aware_rgbt_tracking_with_retrieval-augmented_generation.md)**

:   首次将文本描述引入 RGBT 跟踪，提出基于检索增强生成（RAG）的框架 RAGTrack，通过多模态 Transformer 编码器、自适应 Token 融合和上下文感知推理模块，在四个 RGBT 基准上取得 SOTA。

**[Real-World Point Tracking with Verifier-Guided Pseudo-Labeling](realworld_point_tracking_with_verifierguided_pseud.md)**

:   提出一个可学习的Verifier元模型，在合成数据上训练"判断tracker预测可靠性"的能力并迁移到真实世界，通过逐帧评估6个预训练tracker的预测来选取最可靠的作为伪标签，仅用~5K真实视频即微调出在4个真实世界基准上全面SOTA的Track-On-R模型。

**[Reconstruction-Guided Slot Curriculum: Addressing Object Over-Fragmentation in Video Object-Centric Learning](reconstruction-guided_slot_curriculum_addressing_object_over-fragmentation_in_vi.md)**

:   提出 SlotCurri，一种重建引导的 slot 数量课程学习策略，从极少 slot 开始训练并仅在重建误差高的区域逐步扩展 slot 容量，配合结构感知损失和循环推理，有效解决视频物体中心学习中单一物体被多个 slot 错误拆分的过度碎片化问题，在 YouTube-VIS 上实现 +6.8 FG-ARI 提升。

**[FlexHook: Rethinking Two-Stage Referring-by-Tracking in RMOT](rethinking_two-stage_referring-by-tracking_in_referring_multi-object_tracking_ma.md)**

:   提出 FlexHook，一种新颖的两阶段 Referring-by-Tracking 框架，通过基于采样的 Conditioning Hook（C-Hook）重新定义特征构建，并用 Pairwise Correspondence Decoder（PCD）替换 CLIP 余弦相似度匹配，首次使两阶段方法全面超越当前 SOTA 的一阶段方法。

**[SAIL: Similarity-Aware Guidance and Inter-Caption Augmentation-based Learning for Weakly-Supervised Dense Video Captioning](sail_similarity-aware_guidance_and_inter-caption_augmentation-based_learning_for.md)**

:   提出 SAIL，通过跨模态相似度引导的语义感知掩码生成和 LLM 合成字幕的辅助监督，在仅有字幕标注（无时间边界）的弱监督设置下，在 ActivityNet 和 YouCook2 上实现密集视频描述和事件定位的双 SOTA。

**[SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion](savax_egotoexo_imitation_error_detection_via_scene.md)**

:   形式化 Ego→Exo 模仿错误检测任务，并提出 SAVA-X (Align–Fuse–Detect) 框架，通过自适应采样、场景自适应视角嵌入和双向交叉注意力融合三个模块联合解决时序不对齐、视频冗余和跨视角域差距三大挑战。

**[Seen-to-Scene: Keep the Seen, Generate the Unseen for Video Outpainting](seen_to_scene_keep_the_seen_generate_the_unseen_for_video_outpainting.md)**

:   提出 Seen-to-Scene，统一传播式和生成式范式的视频外推框架，通过参考帧引导的潜空间传播与视频扩散模型结合，在零样本推理中实现了超越需要输入特定适配的先进方法的时空一致性和视觉保真度。

**[SHOW3D: Capturing Scenes of 3D Hands and Objects in the Wild](show3d_capturing_scenes_of_3d_hands_and_objects_in_the_wild.md)**

:   提出首个真正野外环境下具有精确3D标注的手-物体交互数据集SHOW3D，通过设计轻便可穿戴多相机背包系统和ego-exo融合标注pipeline，采集430万帧多视角数据，手部和物体均达到亚厘米级标注精度，跨数据集实验验证其训练模型的泛化优势。

**[SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)**

:   提出SkeletonContext框架，通过跨模态上下文提示模块从预训练语言模型重建骨骼数据缺失的环境和物体上下文语义，并用关键部位解耦模块增强运动关键关节的判别力，在NTU-60/120和PKU-MMD上的零样本和广义零样本设置中达到SOTA。

**[SlotVTG: Object-Centric Adapter for Generalizable Video Temporal Grounding](slotvtg_object-centric_adapter_for_generalizable_video_temporal_grounding.md)**

:   提出SlotVTG框架，通过在MLLM解码器早期层插入轻量级Slot Adapter将视觉token分解为对象级slot表示，辅以DINOv2先验的Slot Alignment Loss引导语义一致的slot形成，显著提升视频时序定位的域外泛化性能（OOD R1@0.5最大提升+4.3），同时仅增加约0.25%的可训练参数。

**[SpikeTrack: A Spike-driven Framework for Efficient Visual Tracking](spiketrack_a_spike-driven_framework_for_efficient_visual_tracking.md)**

:   提出 SpikeTrack，首个完全符合脉冲驱动范式的 RGB 视觉跟踪框架，通过非对称时间步扩展、单向信息流和脑启发记忆检索模块（MRM），在 SNN 跟踪器中达到 SOTA 并与 ANN 跟踪器持平，同时能耗仅为 TransT 的 1/26。

**[Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning](stay_in_your_lane_role_specific_queries_with_overlap_suppression_loss_for_dense_.md)**

:   提出 ROS-DVC，通过将 DETR-based DVC 框架中的共享 query 分离为独立的 localization query 和 caption query，并设计 Overlap Suppression Loss 惩罚 query 间的时序重叠、Cross-Task Contrastive Alignment 保证跨任务语义一致性，在 YouCook2 和 ActivityNet Captions 上实现了 SOTA 的 captioning 和 localization 性能。

**[STORM: End-to-End Referring Multi-Object Tracking in Videos](storm_referring_multi_object_tracking.md)**

:   STORM 是首个端到端的多模态大语言模型框架用于指代多目标跟踪（RMOT），通过任务组合学习策略大幅减少对 RMOT 标注数据的依赖，并构建了高质量 STORM-Bench 数据集。

**[StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)**

:   提出首个注视引导的流式视频理解基准 StreamGaze，包含 8521 个 QA 对覆盖过去/现在/主动预测三类任务，通过注视轨迹-视频对齐的数据构建管线生成时空grounded的QA，揭示了当前 MLLM 在利用注视信号进行时间推理方面的巨大差距。

**[StreamingTOM: Streaming Token Compression for Efficient Video Understanding](streamingtom_streaming_token_compression_for_efficient_video_understanding.md)**

:   提出 StreamingTOM，一个无需训练的两阶段流式视频理解框架：Causal Temporal Reduction (CTR) 在 LLM 前通过因果时序选择将每帧 token 从 196 压缩到 50，Online Quantized Memory (OQM) 在 LLM 后通过 4-bit 量化和按需检索限制 kv-cache 增长，实现 15.7× 压缩比、1.2× 更低峰值显存和 2× 更快 TTFT。

**[StreamReady: Learning What to Answer and When in Long Streaming Videos](streamready_learning_what_to_answer_and_when_in_long_streaming_videos.md)**

:   提出就绪性感知的流式视频理解范式，通过可学习的 `<RDY>` token 和 Answer Readiness Score (ARS) 指标，让模型不仅回答正确，还能在证据出现的恰当时刻作答，在 9 个流式/离线视频基准上取得 SOTA。

**[SVAgent: Storyline-Guided Long Video Understanding via Cross-Modal Multi-Agent Collaboration](svagent_storyline_guided_long_video_understanding_via_cross_modal_multi_agent_collaboration.md)**

:   提出 SVAgent，一个故事线引导的跨模态多智能体框架用于长视频问答，通过渐进式构建叙事表示、DPP 证据选择、跨模态一致性验证和迭代精炼实现超越基线 5.5%-11.5% 的性能提升。

**[Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition](tcei_test_time_calibration_experience_intuition_mot.md)**

:   TCEI 框架受 Kahneman 双系统理论启发，提出直觉系统（利用近期观测对象的瞬时记忆快速推断）和经验系统（利用历史视频积累的经验校准直觉预测）相结合的测试时自适应方法，无需反向传播即可在分布偏移下显著提升多目标跟踪性能。

**[Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models](token_reduction_via_local_and_global_contexts_optimization_for_efficient_video_l.md)**

:   提出 AOT 框架，通过建立局部-全局 token anchors 并利用最优传输（Optimal Transport）在帧内和帧间两级聚合被裁剪/合并 token 的语义信息，实现 training-free 的视频 token 压缩，在裁剪 90% token 的情况下仍保留 97.6% 的原始性能。

**[TrajTok: Learning Trajectory Tokens Enhances Video Understanding](trajtok_learning_trajectory_tokens_enables_better_video_understanding.md)**

:   提出 TrajTok——一种端到端可微的轨迹 tokenizer，将视频像素隐式聚类为目标轨迹 token，取代外部分割+跟踪流水线；在从头训练 (TrajViT2)、特征适配 (TrajAdapter) 和视觉语言模型连接器 (TrajVLM) 三种场景下均取得显著提升，尤其在长视频 QA 上大幅超越 patch pooling。

**[U2Flow: Uncertainty-Aware Unsupervised Optical Flow Estimation](u2flow_uncertainty_aware_unsupervised_optical_flow_estimation.md)**

:   U2Flow是首个联合估计光流和逐像素不确定性的循环无监督框架，通过基于增强一致性的解耦不确定性学习和不确定性引导的双向光流融合，在KITTI和Sintel上实现无监督SOTA。

**[UETrack: A Unified and Efficient Framework for Single Object Tracking](uetrack_a_unified_and_efficient_framework_for_single_object_tracking.md)**

**[UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models](ufvideo_towards_unified_fine-grained_video_cooperative_understanding_with_large_.md)**

:   UFVideo 是首个统一全局、像素级和时序级三种粒度视频理解能力的 Video LLM，通过视觉-语言引导对齐策略和 SAM2 mask decoder，在单一模型内同时支持视频问答、目标引用、视频分割和时序定位，并构建了多粒度协同理解基准 UFVideo-Bench。

**[Understanding Temporal Logic Consistency in Video-Language Models through Cross-Modal Attention Discriminability](understanding_temporal_logic_consistency_in_video-language_models_through_cross-.md)**

:   本文从可解释性角度分析了视频语言模型（Video-LLMs）时间理解逻辑不一致的根本原因——跨模态注意力头无法有效区分不同时间戳的视频token——并提出 TCAS（Temporally Conditioned Attention Sharpening）方法通过优化注意力分布显著提升了时间逻辑一致性和通用时序定位性能。

**[Unified Spatiotemporal Token Compression for Video-LLMs at Ultra-Low Retention](unified_spatiotemporal_token_compression_for_video-llms_at_ultra-low_retention.md)**

:   提出统一时空token压缩方法，通过全局保留池联合评估token的贡献度和语义冗余度，并在LLM内部引入文本感知合并机制，在仅保留约2%视觉token的极端压缩下仍保留90.1%的基线性能，同时将FLOPs降至约2.6%。

**[UTPTrack: Towards Simple and Unified Token Pruning for Visual Tracking](utptrack_towards_simple_and_unified_token_pruning_for_visual_tracking.md)**

:   提出 UTPTrack，首个在 one-stream Transformer 跟踪器中**同时对搜索区域 (SR)、动态模板 (DT) 和静态模板 (ST) 三个组件进行联合 token 剪枝**的统一框架，在 RGB 和多模态/语言引导跟踪中实现 65–67% 的视觉 token 裁减，且保持 99.7%–100.5% 的基线性能。

**[VecAttention: Vector-wise Sparse Attention for Accelerating Long Context Inference](vecattention_vector-wise_sparse_attention_for_accelerating_long_context_inferenc.md)**

:   本文发现视频模型注意力图中存在强烈的"垂直向量"稀疏模式，据此提出 VecAttention 细粒度向量级稀疏注意力框架，通过 TilingSelect + minS 过滤实现高效重要向量选择，在 78%+ 稀疏度下视频理解准确率与全注意力持平，注意力计算加速 2.65 倍。

**[VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](videoarm_agentic_reasoning_over_hierarchical_memory_for_long-form_video_understa.md)**

:   VideoARM 提出了一种基于分层多模态记忆（HM3）的 Agent 推理范式，通过"观察-思考-行动-记忆"的自适应循环和粗到细的工具调用策略，在长视频理解基准上超越 SOTA 的同时将 token 消耗降低到 DVD 的 1/34。

**[VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice](videoauto-r1_video_auto_reasoning_via_thinking_once_answering_twice.md)**

:   提出 VideoAuto-R1，一个"按需推理"的视频理解框架：训练时采用"思考一次、回答两次"（answer→think→answer）范式，推理时通过首次回答的置信度决定是否启动 CoT 推理，在保持 SOTA 精度的同时将平均响应长度从 149 降至 44 token（约 3.3 倍压缩）。

**[VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning](videochatm1_collaborative_policy_planning_for_vide.md)**

:   VideoChat-M1 提出协作策略规划（CPP）范式和多智能体强化学习（MARL）训练方法，让 4 个异构 VLM agent 动态生成和更新工具调用策略来理解视频，在 LongVideoBench 上超过 Gemini 2.5 Pro 3.6%、GPT-4o 15.6%。

**[VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking](videoseek_long-horizon_video_agent_with_tool-guided_seeking.md)**

:   VideoSeek 提出一种长程视频 Agent，利用视频逻辑流主动"寻找"关键证据而非穷举解析所有帧，通过 think-act-observe 循环和多粒度工具包（overview/skim/focus），在 LVBench 上比基座模型 GPT-5 提升 10.2 个点的同时减少 93% 的帧使用量。

**[VidTAG: Temporally Aligned Video to GPS Geolocalization](vidtag_video_gps_geolocalization.md)**

:   提出 VidTAG，一个双编码器（CLIP+DINOv2）帧到GPS检索框架，通过 TempGeo 模块实现帧间时间对齐，GeoRefiner 编码器-解码器模块精炼GPS预测，在全球尺度下实现时间一致的逐帧视频地理定位。

**[VirtueBench: Evaluating Trustworthiness under Uncertainty in Long Video Understanding](virtuebench_evaluating_trustworthiness_under_uncertainty_in_long_video_understan.md)**

:   提出 VirtueBench，首个评估 VLM 在不确定性下可信度的长视频理解基准，通过为每个视频构建多级帧采样并标注可回答/不可回答的 ground truth，揭示了现有模型普遍倾向于猜测而非诚实拒绝的问题。

**[VRR-QA: Visual Relational Reasoning in Videos Beyond Explicit Cues](vrr-qa_visual_relational_reasoning_in_videos_beyond_explicit_cues.md)**

:   本文提出 VRR-QA 基准，包含 1K 精心标注的视频问答对，专门测试模型对视频中隐式视觉关系的推理能力（如屏幕外事件、跨帧因果、空间关系推断），揭示当前最强 VideoQA 模型（包括 GPT-O3）在隐式推理上的显著不足——最优模型仅达 64% 准确率，远低于人类的 83%。

**[VSI: Visual-Subtitle Integration for Keyframe Selection to Enhance Long Video Understanding](vsi_visual-subtitle_integration_for_keyframe_selection_to_enhance_long_video_un.md)**

:   VSI 提出双分支协作检索框架（视频搜索 + 字幕匹配），通过融合视觉和文本信息实现精确的关键帧定位，在文本相关任务中将搜索准确率从29.48提升至45.00，是首个跨模态关键帧检索方法。

**[Wavelet-based Frame Selection by Detecting Semantic Boundary for Long Video Understanding](wavelet-based_frame_selection_by_detecting_semantic_boundary_for_long_video_unde.md)**

:   提出 WFS-SB，一种免训练的帧选择框架，利用小波变换从查询-帧相似度信号中检测语义边界，将视频分割为语义连贯的片段后自适应分配帧预算并做多样性采样，在 VideoMME/MLVU/LongVideoBench 上大幅超越 SOTA。
