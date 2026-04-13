---
title: >-
  NeurIPS2025 视频理解方向 72篇论文解读
description: >-
  72篇NeurIPS2025 视频理解方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎬 视频理解

**🧠 NeurIPS2025** · 共 **72** 篇

**[A Little Depth Goes A Long Way The Expressive Power Of Logde](a_little_depth_goes_a_long_way_the_expressive_power_of_logde.md)**

:   本文证明了将 Transformer 的深度从常数增长到 Θ(log n) 就能解锁识别正则语言和图连通性这两类固定深度 Transformer 无法表达的问题，且深度扩展比宽度（需超多项式增长）和 CoT 步数（需超对数增长）都更高效。

**[Adavideorag Omnicontextual Adaptive Retrievalaugmented Effic](adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)**

:   提出 AdaVideoRAG，通过轻量级意图分类器将查询按难度路由到三级检索路径（无检索/朴素检索/图检索），结合全知识索引模块（caption+ASR+OCR+视觉+知识图谱）实现长视频理解的效率-精度最优平衡，在 MLVU 上为 Qwen2.5-VL-7B 带来 39.8% 提升。

**[Adversarial Locomotion And Motion Imitation For Humanoid Policy Learning](adversarial_locomotion_and_motion_imitation_for_humanoid_policy_learning.md)**

:   ALMI提出上下半身对抗训练框架：下半身策略在上半身动作干扰下学习鲁棒运动，上半身策略在下半身运动干扰下学习精确动作模仿，通过迭代对抗训练收敛到Nash均衡，实现Unitree H1-2真实机器人的稳定全身协调控制。

**[Agentic Persona Control And Task State Tracking For Realistic User Simulation In](agentic_persona_control_and_task_state_tracking_for_realistic_user_simulation_in.md)**

:   提出三 agent 协作框架用于逼真的用户模拟——User Agent（协调）+ State Tracking Agent（结构化任务状态）+ Message Attributes Generation Agent（基于 persona 和状态的行为属性控制），在餐厅点餐场景中综合仿真质量（CRRS）提升 102.6%，persona 保持度 +19.9%，行为自然度 +284.5%，且核心发现：无状态感知的行为控制导致 BVS=0（完全刚性）。

**[Cleverbirds A Multiple-Choice Benchmark For Fine-Grained Human Knowledge Tracing](cleverbirds_a_multiple-choice_benchmark_for_fine-grained_human_knowledge_tracing.md)**

:   发布 CleverBirds——超大规模细粒度视觉知识追踪基准，包含 4万+用户的 1700万+多选题交互（覆盖 10000+鸟类物种），展示了追踪细粒度视觉专家技能发展的挑战性，为 KT 方法提供了前所未有的视觉领域评测平台。

**[Cloud4D Estimating Cloud Properties At A High Spatial And Temporal Resolution](cloud4d_estimating_cloud_properties_at_a_high_spatial_and_temporal_resolution.md)**

:   首个基于地面多视角相机的学习框架，通过单应性引导的2D-to-3D Transformer重建四维（3D空间+时间）云液态水含量分布，在25m空间/5s时间分辨率下实现了相对雷达<10%的误差，比卫星观测提升了一个数量级的时空分辨率。

**[Convis-Bench Estimating Video Similarity Through Semantic Concepts](convis-bench_estimating_video_similarity_through_semantic_concepts.md)**

:   提出基于语义概念的视频相似度估计任务 ConViS 及配套 benchmark ConViS-Bench（610对视频、16领域、5概念），系统评测了10+主流模型在概念条件下的视频比较能力，揭示当前模型在时序结构和空间语境理解上的显著短板。

**[Deceptron Learned Local Inverses For Fast And Stable Physics Inversion](deceptron_learned_local_inverses_for_fast_and_stable_physics_inversion.md)**

:   提出 Deceptron 双向模块，通过学习可微分前向代理的局部逆映射并引入 Jacobian Composition Penalty (JCP)，在求解物理逆问题时将输出空间的残差拉回输入空间，实现类 Gauss-Newton 的预条件梯度更新，迭代次数大幅减少（Heat-1D 约 20 倍加速）。

**[Deltaproduct Improving State-Tracking In Linear Rnns Via Householder Products](deltaproduct_improving_state-tracking_in_linear_rnns_via_householder_products.md)**

:   提出 DeltaProduct，通过将 DeltaNet 的单步梯度下降扩展至每个 token 的多步梯度下降，使状态转移矩阵成为 $n_h$ 个广义 Householder 变换的乘积，实现了表达力与效率之间的可调平衡，显著提升了状态跟踪能力和长度外推性能。

**[Dense Sae Latents Are Features Not Bugs](dense_sae_latents_are_features_not_bugs.md)**

:   本文系统研究了稀疏自编码器(SAE)中频繁激活的"dense latents"，证明它们不是训练噪声，而是语言模型残差流中固有的密集子空间的反映，并提出了一套包含位置追踪、上下文绑定、零空间、字母、词性和PCA等六类dense latent的分类体系。

**[Disentangled Concepts Speak Louder Than Words Explainable Video Action Recogniti](disentangled_concepts_speak_louder_than_words_explainable_video_action_recogniti.md)**

:   提出DANCE框架，通过将动作解释解耦为运动动态、物体和场景三类概念，实现结构化和运动感知的可解释视频动作识别。

**[Dsas A Universal Plug-And-Play Framework For Attention Optimization In Multi-Doc](dsas_a_universal_plug-and-play_framework_for_attention_optimization_in_multi-doc.md)**

:   提出Dual-Stage Adaptive Sharpening (DSAS)，一个无需训练的即插即用注意力优化框架，通过Contextual Gate Weighting (CGW)增强关键段落对问题和目标位置的注意力、通过Reciprocal Attention Suppression (RAS)抑制关键与无关段落间的信息交换，在多文档QA上平均F1提升达4.2%。

**[Egoemotion Egocentric Vision And Physiological Signals For Emotion And Personali](egoemotion_egocentric_vision_and_physiological_signals_for_emotion_and_personali.md)**

:   提出egoEMOTION——首个结合第一人称视觉（Meta Project Aria眼镜）与生理信号的情感与人格识别数据集，涵盖43名被试、50+小时录制、16种任务，发现第一人称视觉信号（尤其眼动特征）在真实场景情感预测中优于传统生理信号。

**[Empower Words Dualground For Structured Phrase And Sentencel](empower_words_dualground_for_structured_phrase_and_sentencel.md)**

:   论文指出现有视频时序定位模型在跨模态注意力中往往过度依赖句末 [EOS] token 的全局语义、忽视词级局部信号，提出 DualGround 双分支架构，将句子级全局语义与短语级局部语义显式解耦建模，在 QVHighlights 和 Charades-STA 上实现 Moment Retrieval 与 Highlight Detection 的 SOTA。

**[Enhancing Temporal Understanding In Videollms Through Stacke](enhancing_temporal_understanding_in_videollms_through_stacke.md)**

:   提出 STAVEQ2，在 Vision Encoder 中堆叠参数高效的时序注意力模块（STA），解决现有 Video-LLM 在细粒度时序理解（如区分"从左到右拉"和"从右到左拉"）上的根本性架构缺陷，在 VITATECS/MVBench/Video-MME 上提升最高 5.5%。

**[Fastvid Dynamic Density Pruning For Fast Video Large Languag](fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)**

:   提出 FastVID，通过动态时序分割 (DySeg) + 密度空时剪枝 (STPrune) 从时间和视觉两个维度系统性消除视频 token 冗余，在 LLaVA-OneVision-7B 上剪掉 90.3% 视频 token 后仍保留 98% 精度，LLM prefill 阶段加速 7.1×。

**[Fixed-Point Rnns Interpolating From Diagonal To Dense](fixed-point_rnns_interpolating_from_diagonal_to_dense.md)**

:   提出 Fixed-Point RNN 框架，将稠密线性 RNN 参数化为对角线性 RNN 的不动点，通过迭代次数在对角（高效）与稠密（表达力强）之间动态插值，首次在状态跟踪（$A_5$/$S_5$）和拷贝任务上同时取得最优结果。

**[Force Prompting Video Generation Models Can Learn And Generalize Physics-Based C](force_prompting_video_generation_models_can_learn_and_generalize_physics-based_c.md)**

:   提出Force Prompting，将物理力（局部点力和全局风力）作为视频生成模型的控制信号，仅用~15K合成训练视频（Blender旗帜和滚球）和单日4xA100训练，即可在多样真实场景图像上展现跨物体/材质/几何的惊人泛化，包括初步的质量理解能力。

**[Foresight Adaptive Layer Reuse For Accelerated And Highquali](foresight_adaptive_layer_reuse_for_accelerated_and_highquali.md)**

:   提出 Foresight，一种训练无关的自适应层复用框架，通过动态 MSE 阈值决策在 DiT 去噪过程中哪些层可复用缓存、哪些需重新计算，在 OpenSora/Latte/CogVideoX 上实现最高 1.63× 端到端加速且保持视频质量。

**[Geodynamics A Geometric State-Space Neural Network For Understanding Brain Dynam](geodynamics_a_geometric_state-space_neural_network_for_understanding_brain_dynam.md)**

:   提出GeoDynamics，将经典状态空间模型(SSM)从欧几里得空间推广到对称正定(SPD)流形，通过加权Frechet均值聚合和正交群平移实现流形上的状态演化，在脑连接组（AD/PD/ASD早期诊断）和人体动作识别上均取得SOTA。

**[Grounding Foundational Vision Models With 3D Human Poses For Robust Action Recog](grounding_foundational_vision_models_with_3d_human_poses_for_robust_action_recog.md)**

:   提出一种融合 V-JEPA 2 视觉上下文特征与 CoMotion 3D 骨骼姿态数据的 cross-attention 多模态架构，在标准及高遮挡动作识别基准上优于单模态基线。

**[In The Eye Of Mllm Benchmarking Egocentric Video Intent Understanding With Gaze-](in_the_eye_of_mllm_benchmarking_egocentric_video_intent_understanding_with_gaze-.md)**

:   提出 EgoGazeVQA——首个利用注视（gaze）信号评估 MLLM 对第一人称视频中用户意图理解能力的基准，并设计三种 gaze-guided prompting 策略显著提升模型表现。

**[Infinipot-V Memory-Constrained Kv Cache Compression For Streaming Video Understa](infinipot-v_memory-constrained_kv_cache_compression_for_streaming_video_understa.md)**

:   提出首个无需训练、查询无关的流式视频理解框架InfiniPot-V，通过时序冗余度（TaR）和值范数（VaN）两个度量实现KV缓存的在线压缩，在固定内存约束下支持任意长度的流式视频理解。

**[Influx A Benchmark For Self-Calibration Of Dynamic Intrinsics Of Video Cameras](influx_a_benchmark_for_self-calibration_of_dynamic_intrinsics_of_video_cameras.md)**

:   提出首个包含逐帧动态相机内参真值的真实视频基准 InFlux（386 视频、143K+ 标注帧），通过镜头元数据到内参的查找表（LUT）实现精确标注，并揭示现有内参预测方法在动态内参场景下表现不佳。

**[Inst-It Boosting Instance Understanding Via Explicit Visual Prompt Instruction T](inst-it_boosting_instance_understanding_via_explicit_visual_prompt_instruction_t.md)**

:   提出Inst-IT完整方案：通过GPT-4o辅助的自动标注管线生成实例级细粒度数据，构建Inst-IT Bench评测基准和335K QA对的指令微调数据集，以持续微调范式有效提升LMM的实例级理解能力，同时增强通用图像/视频理解。

**[Kungfubot Physics-Based Humanoid Whole-Body Control For Learning Highly-Dynamic ](kungfubot_physics-based_humanoid_whole-body_control_for_learning_highly-dynamic_.md)**

:   提出 PBHC 框架，通过物理感知运动处理流水线和自适应跟踪因子的双层优化，使人形机器人（Unitree G1）学会功夫、舞蹈等高动态全身动作，跟踪误差显著优于现有方法并成功实机部署。

**[Lattice Boltzmann Model For Learning Real-World Pixel Dynamicity](lattice_boltzmann_model_for_learning_real-world_pixel_dynamicity.md)**

:   受流体力学中格子玻尔兹曼方法启发，提出 LBM（Lattice Boltzmann Model）用于在线实时像素跟踪，将视频像素建模为流体格子并通过碰撞-流式过程求解运动状态，以 18M 参数实现 SOTA 在线跟踪性能且可在边缘设备上实时运行。

**[Lemica Lexicographic Minimax Path Caching For Efficient Diffusion-Based Video Ge](lemica_lexicographic_minimax_path_caching_for_efficient_diffusion-based_video_ge.md)**

:   提出 LeMiCa，一种免训练的扩散视频生成加速框架，将缓存调度建模为有向无环图上的字典序极小极大路径优化问题，通过全局误差控制实现速度和质量的双重提升（Latte 上 2.9× 加速，Open-Sora 上 LPIPS 低至 0.05）。

**[Less Is More Local Intrinsic Dimensions Of Contextual Language Models](less_is_more_local_intrinsic_dimensions_of_contextual_language_models.md)**

:   提出利用上下文 token 嵌入的局部内在维度（Local Intrinsic Dimension, LID）来无监督监测 LLM 训练动态——维度下降预示泛化改善，维度上升预示过拟合——在对话状态跟踪、grokking、情感识别等任务上验证了这一几何信号的实用性。

**[Livestar Live Streaming Assistant For Real-World Online Video Understanding](livestar_live_streaming_assistant_for_real-world_online_video_understanding.md)**

:   提出 LiveStar，一个始终在线的直播流视频理解助手，通过 Streaming Causal Attention Masks (SCAM) 训练策略和 Streaming Verification Decoding (SVeD) 推理框架，实现自适应响应时机判断，在 OmniStar 基准上语义正确性提升 19.5%，时间偏差降低 18.1%。

**[Memtrack Evaluating Long-Term Memory And State Tracking In Multi-Platform Dynami](memtrack_evaluating_long-term_memory_and_state_tracking_in_multi-platform_dynami.md)**

:   提出 MEMTRACK 基准，评估 LLM 智能体在多平台（Slack/Linear/Git）动态环境中的长期记忆和状态追踪能力，揭示即使最强的 GPT-5 也仅达 60% 正确率。

**[Mimeqa Towards Socially-Intelligent Nonverbal Foundation Models](mimeqa_towards_socially-intelligent_nonverbal_foundation_models.md)**

:   构建首个基于哑剧视频的非语言社交推理基准 MimeQA，包含101个视频和806个QA对，覆盖三层问题层次（具象识别→场景理解→全局推理），揭示当前VideoLLMs在非语言社交理解上的严重不足（20-30% vs 人类86%）。

**[Muvr A Multi-Modal Untrimmed Video Retrieval Benchmark With Multi-Level Visual C](muvr_a_multi-modal_untrimmed_video_retrieval_benchmark_with_multi-level_visual_c.md)**

:   提出 MUVR 基准，面向长视频平台的多模态未剪辑视频检索任务，设计了以视频为中心的多模态查询格式（视频+文本+标签+掩码）和六级视觉对应匹配准则，包含 53K 视频和 1050 个查询，系统评估了检索模型和 MLLM 的局限性。

**[Neural Stochastic Flows Solver-Free Modelling And Inference For Sde Solutions](neural_stochastic_flows_solver-free_modelling_and_inference_for_sde_solutions.md)**

:   提出 Neural Stochastic Flows（NSF），通过条件归一化流直接学习 SDE 的转移分布 $p(x_t \mid x_s)$，在架构上约束满足随机流性质（恒等、Markov、Chapman-Kolmogorov），实现了无需数值求解器的单步采样，在远距时间点上加速高达两个数量级。

**[Neuropath Neurobiology-Inspired Path Tracking And Reflection For Semantically Co](neuropath_neurobiology-inspired_path_tracking_and_reflection_for_semantically_co.md)**

:   受神经生物学中海马体位置细胞导航与记忆巩固机制启发，提出 NeuroPath——一个基于语义路径追踪的 RAG 框架，通过 LLM 驱动的目标导向路径构建和后检索补全策略，在多跳问答任务上实现 recall@2 平均 16.3% 和 recall@5 平均 13.5% 的提升。

**[Open-World Drone Active Tracking With Goal-Centered Rewards](open-world_drone_active_tracking_with_goal-centered_rewards.md)**

:   提出首个开放世界无人机主动跟踪基准 DAT（24 个城市级场景、高保真动力学仿真），以及基于目标中心奖励函数和课程学习的强化学习跟踪方法 GC-VAT，在仿真器上达到约 72% 的跟踪成功率。

**[Part-Aware Bottom-Up Group Reasoning For Fine-Grained Social Interaction Detecti](part-aware_bottom-up_group_reasoning_for_fine-grained_social_interaction_detecti.md)**

:   提出一种部位感知的自底向上群组推理框架，通过姿态引导的身体部位特征增强和基于相似度的个体关联来推断社交群组和细粒度交互，在 NVI 和 Café 数据集上达到新 SOTA。

**[Pass Path-Selective State Space Model For Event-Based Recognition](pass_path-selective_state_space_model_for_event-based_recognition.md)**

:   PASS提出路径选择性事件聚合与扫描（PEAS）模块和多面选择引导（MSG）损失，利用SSM的线性复杂度和频率泛化能力，实现了从10^6到10^9事件长度的广泛分布上的事件识别，并在推理频率变化时保持性能仅下降8.62%（基线下降20.69%）。

**[Photography Perspective Composition Towards Aesthetic Perspective Recommendation](photography_perspective_composition_towards_aesthetic_perspective_recommendation.md)**

:   首次提出摄影视角构图（PPC），超越传统 2D 裁剪，通过 3D 透视变换生成"差→优"的构图过程视频，并基于人类评估训练视角质量评估模型，帮助普通用户提升摄影构图水平。

**[Pixfoundation 20 Do Video Multi-Modal Llms Use Motion In Visual Grounding](pixfoundation_20_do_video_multi-modal_llms_use_motion_in_visual_grounding.md)**

:   通过提出四项运动中心的探测技术和 MoCentric-Bench 基准，证明当前视频多模态 LLM 在像素级视觉接地任务中未能真正利用运动信息，可被静态关键帧欺骗。

**[Prefm Online Audio-Visual Event Parsing Via Predictive Future Modeling](prefm_online_audio-visual_event_parsing_via_predictive_future_modeling.md)**

:   本文首次提出在线音视频事件解析（On-AVEP）范式，通过预测性未来建模框架 PreFM，利用伪未来序列增强当前上下文理解，同时借助模态无关的知识蒸馏和焦点时间优先策略，以仅 2.7% 的参数量超越离线 SOTA 方法 +9.3 的事件级平均 F1 分数。

**[Qimeng-Neucomback Self-Evolving Translation From Ir To Assembly Code](qimeng-neucomback_self-evolving_translation_from_ir_to_assembly_code.md)**

:   提出NeuComBack基准数据集用于评估IR到汇编的神经编译任务，并设计自进化提示优化方法，通过从LLM自调试轨迹中学习来迭代改进编译提示，使正确率从44%提升到64%，且87.5%的正确程序性能超越clang-O3。

**[Radial Attention Onlog N Sparse Attention With Energy Decay For Long Video Gener](radial_attention_onlog_n_sparse_attention_with_energy_decay_for_long_video_gener.md)**

:   Radial Attention 发现了视频扩散模型中注意力分数随时空距离指数衰减的"时空能量衰减"现象，据此设计了一种 O(n log n) 复杂度的静态稀疏注意力掩码，在 HunyuanVideo/Wan2.1 等模型上实现最高 3.7× 推理加速，并通过 LoRA 微调支持 4× 更长视频生成。

**[Reinforcement Learning With Backtracking Feedback](reinforcement_learning_with_backtracking_feedback.md)**

:   提出带回溯反馈的强化学习框架 RLBF，当 agent 陷入死胡同时允许回溯到之前的状态重新探索，通过回溯信号改善信用分配，在稀疏奖励环境中显著提升探索效率。

**[Revisiting Bi-Linear State Transitions In Recurrent Neural Networks](revisiting_bi-linear_state_transitions_in_recurrent_neural_networks.md)**

:   系统性地重新审视 RNN 中的双线性状态转移（隐状态与输入的乘法交互），理论证明双线性 RNN 可模拟任意有限状态机，并展示其在去除加性项后形成了一个从对角到全结构的自然表达力层次，揭示了 Mamba 等流行线性 RNN 处于该层次最低端。

**[Sama Towards Multi-Turn Referential Grounded Video Chat With Large Language Mode](sama_towards_multi-turn_referential_grounded_video_chat_with_large_language_mode.md)**

:   提出 SAMA 框架，通过构建统一的数据集（SAMA-239K）、模型（时空上下文聚合器 + SAM）和基准（SAMA-Bench），首次实现了多轮引用式视频对话中细粒度时空理解与grounding的联合建模。

**[Scaling Rl To Long Videos](scaling_rl_to_long_videos.md)**

:   提出 LongVILA-R1 全栈框架，通过构建 104K 长视频推理数据集、两阶段 CoT-SFT + RL 训练流水线、以及高效的多模态强化学习序列并行 (MR-SP) 系统，将 VLM 的推理能力扩展到长视频（最高支持 8192 帧），在 VideoMME 上达到 65.1%/71.1%。

**[Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition](seeing_beyond_the_scene_analyzing_and_mitigating_background_bias_in_action_recog.md)**

:   系统分析动作识别模型中的背景偏差问题——模型通过背景场景而非动作本身做分类（如"游泳"被识别是因为看到泳池而非游泳动作），并提出基于因果推理的去偏方法。

**[Seeing The Arrow Of Time In Large Multimodal Models](seeing_the_arrow_of_time_in_large_multimodal_models.md)**

:   本文揭示当前大多模态模型（LMMs）对视频时间方向性（时间箭头）出人意料地不敏感——正放/倒放时答案几乎相同，提出基于 GRPO 的 ArrowRL 训练策略引入反向视频奖励来激发时间方向感知，并构建 AoTBench 基准，在多个 VQA 基准上取得显著提升（Vinoground 上相对提升 65.9%）。

**[Seeing The Wind From A Falling Leaf](seeing_the_wind_from_a_falling_leaf.md)**

:   本文提出端到端可微逆图形学框架，通过联合建模物体几何、物理属性和力表示，从视频中恢复不可见的力场（如风场），并支持基于物理的视频生成与编辑。

**[Smartwilds Multimodal Wildlife Monitoring Dataset](smartwilds_multimodal_wildlife_monitoring_dataset.md)**

:   发布SmartWilds数据集首版，包含在俄亥俄州The Wilds野生动物园同步采集的无人机影像、相机陷阱照片/视频和生物声学录音，共101GB/20K+文件，支持多模态AI在濒危物种保护和栖息地管理中的研究。

**[Stable Cinemetrics Structured Taxonomy And Evaluation For Professional Video Gen](stable_cinemetrics_structured_taxonomy_and_evaluation_for_professional_video_gen.md)**

:   提出 SCINE（Stable Cinemetrics），首个面向专业视频制作的结构化评估框架，定义了 76 个细粒度电影控制节点的分层分类体系，配合大规模专业人员评估（80+ 影视从业者、20K+ 视频、248K 标注），揭示当前最强 T2V 模型在专业控制上的显著不足。

**[Steering When Necessary Flexible Steering Large Language Models With Backtrackin](steering_when_necessary_flexible_steering_large_language_models_with_backtrackin.md)**

:   提出 FASB（Flexible Activation Steering with Backtracking）框架，通过跟踪 LLM 生成过程中的内部状态动态判断干预必要性和强度，并引入回溯机制纠正已偏离的 token，在 TruthfulQA 上 True*Info 达 80.56%、6 个多选任务平均准确率 78.8%，显著优于所有基线。

**[Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)**

**[Tapvid-360 Tracking Any Point In 360 From Narrow Field Of View Video](tapvid-360_tracking_any_point_in_360_from_narrow_field_of_view_video.md)**

:   本文提出TAPVid-360任务和数据集，要求模型在窄视野视频中跟踪查询点的3D方向（包括视野外的点），通过利用360度视频生成训练数据并微调CoTracker3实现方向预测，在视野外跟踪上远超现有方法。

**[Tempsampr1 Effective Temporal Sampling With Reinforcement Fi](tempsampr1_effective_temporal_sampling_with_reinforcement_fi.md)**

:   提出 TempSamp-R1，针对视频时序定位任务改进 GRPO 强化微调框架，通过 off-policy 时间精确引导 + 非线性软优势计算 + 混合 CoT 训练，在 Charades-STA/ActivityNet/QVHighlights 上分别提升 +2.7%/+5.3%/+3.0%。

**[The Ouroboros Of Benchmarking Reasoning Evaluation In An Era Of Saturation](the_ouroboros_of_benchmarking_reasoning_evaluation_in_an_era_of_saturation.md)**

:   本文系统分析了OpenAI、Anthropic和Google三大模型家族在52个基准上的推理能力演变，揭示了"基准饱和循环"现象——旧基准快速被超越、新基准不断涌现，质疑高基准分数是否真正反映泛化推理能力。

**[Token Bottleneck One Token To Remember Dynamics](token_bottleneck_one_token_to_remember_dynamics.md)**

:   提出Token Bottleneck（ToBo），一种自监督视觉表征学习流水线，通过将参考场景压缩为单个瓶颈token、并利用该token与极少量目标场景patch来预测后续场景，使视觉骨干网络同时学会保守编码场景信息和捕获时间动态变化。

**[Toolaugmented Spatiotemporal Reasoning For Streamlining Vide](toolaugmented_spatiotemporal_reasoning_for_streamlining_vide.md)**

:   论文为复杂 VideoQA 提出一套轻量但可扩展的 Video Toolkit，并设计 STAR（Spatiotemporal Reasoning Framework）来调度时间工具与空间工具的调用顺序，逐步定位视频关键区域，显著增强 GPT-4o 的时空推理能力，在 VideoMME 上提升 8.2%，在 LongVideoBench 上提升 4.6%。

**[Tracking And Understanding Object Transformations](tracking_and_understanding_object_transformations.md)**

:   提出 Track Any State 任务和 TubeletGraph 零样本框架，在视频中跟踪经历外观剧变的物体状态变化（如切苹果、蝴蝶从蛹中羽化），同时检测并描述这些变化。

**[Trackingworld World-Centric Monocular 3D Tracking Of Almost All Pixels](trackingworld_world-centric_monocular_3d_tracking_of_almost_all_pixels.md)**

:   提出TrackingWorld，一个从单目视频实现几乎所有像素的稠密3D跟踪的流水线，通过跟踪上采样器将稀疏2D轨迹提升为稠密轨迹、迭代跟踪所有帧中新出现的物体、以及基于优化的框架将2D轨迹提升到世界坐标系3D空间并显式分离相机运动和物体运动。

**[Two Causally Related Needles In A Video Haystack](two_causally_related_needles_in_a_video_haystack.md)**

:   提出CAUSAL2NEEDLES benchmark评估VLM的长视频双针(2-needle)因果推理能力：需要从视频两个不同位置提取因果关联的事件信息并联合推理，利用"桥接实体"迫使模型先理解结果再追溯原因，揭示即使GPT-4o在2-needle因果问题上仅达13.4%的Both准确率（vs人类79.3%）。

**[Unleashing Hour-Scale Video Training For Long Video-Language Understanding](unleashing_hour-scale_video_training_for_long_video-language_understanding.md)**

:   提出 VideoMarathon 大规模小时级视频指令跟随数据集（9700小时、330万QA对）和 Hour-LLaVA 模型，通过记忆增强模块实现 1-FPS 的小时级视频训练和推理，在多个长视频基准上达到最优。

**[Vgent Graph-Based Retrieval-Reasoning-Augmented Generation For Long Video Unders](vgent_graph-based_retrieval-reasoning-augmented_generation_for_long_video_unders.md)**

:   提出 VGEnt，一个基于图的检索-推理增强生成框架，通过构建视频知识图谱保留跨片段语义关系，并引入结构化推理步骤过滤噪声、聚合信息，在多个长视频理解基准上一致提升开源 LVLM 3.0%~5.4%，超越现有视频 RAG 方法 8.6%。

**[Video Finetuning Improves Reasoning Between Frames](video_finetuning_improves_reasoning_between_frames.md)**

:   本文通过提出视觉思维链（vCoT）方法，系统地比较了图像LLM与视频微调LLM在帧间推理能力上的差异，发现视频微调使模型隐式学会了帧间过渡推理，且这种能力可迁移到静态图像的关系推理任务中。

**[Videolucy Deep Memory Backtracking For Long Video Understanding](videolucy_deep_memory_backtracking_for_long_video_understanding.md)**

:   提出VideoLucy框架，通过层次化记忆结构和基于Agent的迭代回溯机制，模拟人类从粗到细的回忆过程，在多个长视频理解基准上大幅超越现有方法，甚至超过GPT-4o等商业模型。

**[Visual Diversity And Region-Aware Prompt Learning For Zero-Shot Hoi Detection](visual_diversity_and_region-aware_prompt_learning_for_zero-shot_hoi_detection.md)**

:   提出 VDRP 框架，通过视觉多样性感知的 prompt 学习（注入组级方差 + 高斯扰动）和区域感知的 prompt 增强（基于 LLM 生成的区域概念检索），解决零样本 HOI 检测中类内视觉多样性和类间视觉纠缠两大挑战。

**[Vmdt Decoding The Trustworthiness Of Video Foundation Models](vmdt_decoding_the_trustworthiness_of_video_foundation_models.md)**

:   提出 VMDT（Video-Modal DecodingTrust），首个统一评估 T2V 和 V2T 视频基础模型在安全、幻觉、公平、隐私和对抗鲁棒性五个维度上可信度的基准平台，涵盖 7 个 T2V 和 19 个 V2T 模型的大规模评测，揭示了模型规模与可信度之间的复杂关系。

**[Vorta Efficient Video Diffusion Via Routing Sparse Attention](vorta_efficient_video_diffusion_via_routing_sparse_attention.md)**

:   提出VORTA框架，通过桶化核心集注意力（建模长程依赖）和信号感知路由机制（自适应选择稀疏注意力分支），在不损失生成质量的前提下实现视频扩散Transformer端到端1.76×加速，并可与缓存和蒸馏方法叠加达到14.41×加速。

**[Web-Scale Collection Of Video Data For 4D Animal Reconstruction](web-scale_collection_of_video_data_for_4d_animal_reconstruction.md)**

:   提出一个全自动化的大规模视频数据采集管线，从 YouTube 挖掘并处理得到 30K 动物视频（2M帧），建立首个 4D 四足动物重建基准 Animal-in-Motion（230序列/11K帧），并提出 4D-Fauna 基线方法实现序列级优化的无模型 4D 重建。

**[When One Moment Isnt Enough Multi-Moment Retrieval With Cross-Moment Interaction](when_one_moment_isnt_enough_multi-moment_retrieval_with_cross-moment_interaction.md)**

:   提出QV-M2数据集（首个全人工标注的多时刻检索基准）和FlashMMR框架（含后验证模块），将视频时刻检索从单时刻扩展到多时刻场景，建立了多时刻检索的标准化评价体系。

**[When Thinking Drifts Evidential Grounding For Robust Video Reasoning](when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)**

:   系统揭示了CoT推理在视频理解中经常导致性能下降的"视觉思维漂移"现象，并提出Visual Evidence Reward（VER）强化学习框架，通过显式奖励与视觉证据对齐的推理链来纠正这一问题。
