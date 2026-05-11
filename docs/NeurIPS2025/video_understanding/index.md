---
title: >-
  NeurIPS2025 视频理解方向58篇论文解读
description: >-
  58篇NeurIPS2025的视频理解方向论文解读，涵盖目标跟踪、推理、对抗鲁棒、问答、LLM、压缩/编码等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📹 视频理解

**🧠 NeurIPS2025** · **58** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (11)](../../ACL2026/video_understanding/index.md) · [📷 CVPR2026 (77)](../../CVPR2026/video_understanding/index.md) · [🔬 ICLR2026 (24)](../../ICLR2026/video_understanding/index.md) · [🤖 AAAI2026 (33)](../../AAAI2026/video_understanding/index.md) · [📹 ICCV2025 (57)](../../ICCV2025/video_understanding/index.md) · [🧪 ICML2025 (7)](../../ICML2025/video_understanding/index.md)

🔥 **高频主题：** 目标跟踪 ×11 · 推理 ×5 · 对抗鲁棒 ×3 · 问答 ×3 · LLM ×3

**[A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers](a_little_depth_goes_a_long_way_the_expressive_power_of_logde.md)**

:   本文证明了将 Transformer 的深度从常数增长到 Θ(log n) 就能解锁识别正则语言和图连通性这两类固定深度 Transformer 无法表达的问题，且深度扩展比宽度（需超多项式增长）和 CoT 步数（需超对数增长）都更高效。

**[AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)**

:   提出 AdaVideoRAG，通过轻量级意图分类器将查询按难度路由到三级检索路径（无检索/朴素检索/图检索），结合全知识索引模块（caption+ASR+OCR+视觉+知识图谱）实现长视频理解的效率-精度最优平衡，在 MLVU 上为 Qwen2.5-VL-7B 带来 39.8% 提升。

**[Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning](adversarial_locomotion_and_motion_imitation_for_humanoid_policy_learning.md)**

:   ALMI提出上下半身对抗训练框架：下半身策略在上半身动作干扰下学习鲁棒运动，上半身策略在下半身运动干扰下学习精确动作模仿，通过迭代对抗训练收敛到Nash均衡，实现Unitree H1-2真实机器人的稳定全身协调控制。

**[Agentic Persona Control and Task State Tracking for Realistic User Simulation](agentic_persona_control_and_task_state_tracking_for_realistic_user_simulation_in.md)**

:   提出三 agent 协作框架用于逼真的用户模拟——User Agent（协调）+ State Tracking Agent（结构化任务状态）+ Message Attributes Generation Agent（基于 persona 和状态的行为属性控制），在餐厅点餐场景中综合仿真质量（CRRS）提升 102.6%，persona 保持度 +19.9%，行为自然度 +284.5%，且核心发现：无状态感知的行为控制导致 BVS=0（完全刚性）。

**[CleverBirds: A Multiple-Choice Benchmark for Fine-grained Human Knowledge Tracing](cleverbirds_a_multiple-choice_benchmark_for_fine-grained_human_knowledge_tracing.md)**

:   提出CleverBirds——迄今最大规模的视觉知识追踪基准，通过eBird公民科学平台收集了40000+参与者完成的1700万+多选鸟类物种识别问答（涵盖10000+物种），系统性评估了多种知识追踪与分类方法，揭示了细粒度视觉知识建模特别是预测学习者错误选择方面的核心挑战。

**[Cloud4D: Estimating Cloud Properties at a High Spatial and Temporal Resolution](cloud4d_estimating_cloud_properties_at_a_high_spatial_and_temporal_resolution.md)**

:   首个基于地面多视角相机的学习框架，通过单应性引导的2D-to-3D Transformer重建四维（3D空间+时间）云液态水含量分布，在25m空间/5s时间分辨率下实现了相对雷达<10%的误差，比卫星观测提升了一个数量级的时空分辨率。

**[ConViS-Bench: Estimating Video Similarity Through Semantic Concepts](convis-bench_estimating_video_similarity_through_semantic_concepts.md)**

:   提出基于语义概念的视频相似度估计任务 ConViS 及配套 benchmark ConViS-Bench（610对视频、16领域、5概念），系统评测了10+主流模型在概念条件下的视频比较能力，揭示当前模型在时序结构和空间语境理解上的显著短板。

**[DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products](deltaproduct_improving_state-tracking_in_linear_rnns_via_householder_products.md)**

:   提出 DeltaProduct，通过将 DeltaNet 的单步梯度下降扩展至每个 token 的多步梯度下降，使状态转移矩阵成为 $n_h$ 个广义 Householder 变换的乘积，实现了表达力与效率之间的可调平衡，显著提升了状态跟踪能力和长度外推性能。

**[Dense SAE Latents Are Features, Not Bugs](dense_sae_latents_are_features_not_bugs.md)**

:   本文系统研究了稀疏自编码器(SAE)中频繁激活的"dense latents"，证明它们不是训练噪声，而是语言模型残差流中固有的密集子空间的反映，并提出了一套包含位置追踪、上下文绑定、零空间、字母、词性和PCA等六类dense latent的分类体系。

**[Disentangled Concepts Speak Louder Than Words: Explainable Video Action Recognition](disentangled_concepts_speak_louder_than_words_explainable_video_action_recogniti.md)**

:   提出DANCE框架，通过将动作解释解耦为运动动态、物体和场景三类概念，实现结构化和运动感知的可解释视频动作识别。

**[DSAS: A Universal Plug-and-Play Framework for Attention Optimization in Multi-Document Question Answering](dsas_a_universal_plug-and-play_framework_for_attention_optimization_in_multi-doc.md)**

:   提出Dual-Stage Adaptive Sharpening (DSAS)，一个无需训练的即插即用注意力优化框架，通过Contextual Gate Weighting (CGW)增强关键段落对问题和目标位置的注意力、通过Reciprocal Attention Suppression (RAS)抑制关键与无关段落间的信息交换，在多文档QA上平均F1提升达4.2%。

**[DualGround: Structured Phrase and Sentence-Level Temporal Grounding](dualground_phrase_temporal.md)**

:   本文发现现有视频时间定位模型过度依赖 [EOS] token 的全局句子语义而忽略词级信号，提出 DualGround 双分支架构，通过句子级路径（自适应交叉注意力）和短语级路径（循环短语生成+Slot Attention）显式分离全局和局部语义，在 QVHighlights 和 Charades-STA 上实现 SOTA。

**[egoEMOTION: Egocentric Vision and Physiological Signals for Emotion and Personality Recognition in Real-World Tasks](egoemotion_egocentric_vision_and_physiological_signals_for_emotion_and_personali.md)**

:   提出egoEMOTION——首个结合第一人称视觉（Meta Project Aria眼镜）与生理信号的情感与人格识别数据集，涵盖43名被试、50+小时录制、16种任务，发现第一人称视觉信号（尤其眼动特征）在真实场景情感预测中优于传统生理信号。

**[EgoGazeVQA: Egocentric Gaze-Guided Video Question Answering Benchmark](egogazevqa_egocentric_gaze_guided_video_question_answering.md)**

:   提出 EgoGazeVQA，首个融合用户眼动注视数据的第一人称视频问答基准，通过注视引导的提示策略（文本/视觉/显著性图）显著提升 MLLM 对用户意图的理解能力，Gaze Salience Map 策略最高可将 MiniCPM-o 的准确率从35.9%提升至53.7%。

**[Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](enhancing_temporal_understanding_in_videollms_through_stacke.md)**

:   提出 STAVEQ2，在 Vision Encoder 中堆叠参数高效的时序注意力模块（STA），解决现有 Video-LLM 在细粒度时序理解（如区分"从左到右拉"和"从右到左拉"）上的根本性架构缺陷，在 VITATECS/MVBench/Video-MME 上提升最高 5.5%。

**[FastVID: Dynamic Density Pruning for Fast Video Large Language Models](fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)**

:   提出 FastVID，通过动态时序分割 (DySeg) + 密度空时剪枝 (STPrune) 从时间和视觉两个维度系统性消除视频 token 冗余，在 LLaVA-OneVision-7B 上剪掉 90.3% 视频 token 后仍保留 98% 精度，LLM prefill 阶段加速 7.1×。

**[Fixed-Point RNNs: Interpolating from Diagonal to Dense](fixed-point_rnns_interpolating_from_diagonal_to_dense.md)**

:   提出 Fixed-Point RNN 框架，将稠密线性 RNN 参数化为对角线性 RNN 的不动点，通过迭代次数在对角（高效）与稠密（表达力强）之间动态插值，首次在状态跟踪（$A_5$/$S_5$）和拷贝任务上同时取得最优结果。

**[Grounding Foundational Vision Models with 3D Human Poses for Robust Action Recognition](grounding_foundational_vision_models_with_3d_human_poses_for_robust_action_recog.md)**

:   提出一种融合 V-JEPA 2 视觉上下文特征与 CoMotion 3D 骨骼姿态数据的 cross-attention 多模态架构，在标准及高遮挡动作识别基准上优于单模态基线。

**[InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding](infinipot-v_memory-constrained_kv_cache_compression_for_streaming_video_understa.md)**

:   提出首个无需训练、查询无关的流式视频理解框架InfiniPot-V，通过时序冗余度（TaR）和值范数（VaN）两个度量实现KV缓存的在线压缩，在固定内存约束下支持任意长度的流式视频理解。

**[InFlux: A Benchmark for Self-Calibration of Dynamic Intrinsics of Video Cameras](influx_a_benchmark_for_self-calibration_of_dynamic_intrinsics_of_video_cameras.md)**

:   提出首个包含逐帧动态相机内参真值的真实视频基准 InFlux（386 视频、143K+ 标注帧），通过镜头元数据到内参的查找表（LUT）实现精确标注，并揭示现有内参预测方法在动态内参场景下表现不佳。

**[INST-IT: Boosting Instance Understanding via Explicit Visual Prompt Instruction Tuning](inst-it_boosting_instance_understanding_via_explicit_visual_prompt_instruction_t.md)**

:   提出Inst-IT完整方案：通过GPT-4o辅助的自动标注管线生成实例级细粒度数据，构建Inst-IT Bench评测基准和335K QA对的指令微调数据集，以持续微调范式有效提升LMM的实例级理解能力，同时增强通用图像/视频理解。

**[KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills](kungfubot_physics-based_humanoid_whole-body_control_for_learning_highly-dynamic_.md)**

:   提出 PBHC 框架，通过物理感知运动处理流水线和自适应跟踪因子的双层优化，使人形机器人（Unitree G1）学会功夫、舞蹈等高动态全身动作，跟踪误差显著优于现有方法并成功实机部署。

**[Lattice Boltzmann Model for Learning Real-World Pixel Dynamicity](lattice_boltzmann_model_for_learning_real-world_pixel_dynamicity.md)**

:   受流体力学中格子玻尔兹曼方法启发，提出 LBM（Lattice Boltzmann Model）用于在线实时像素跟踪，将视频像素建模为流体格子并通过碰撞-流式过程求解运动状态，以 18M 参数实现 SOTA 在线跟踪性能且可在边缘设备上实时运行。

**[Less is More: Local Intrinsic Dimensions of Contextual Language Models](less_is_more_local_intrinsic_dimensions_of_contextual_language_models.md)**

:   提出利用上下文 token 嵌入的局部内在维度（Local Intrinsic Dimension, LID）来无监督监测 LLM 训练动态——维度下降预示泛化改善，维度上升预示过拟合——在对话状态跟踪、grokking、情感识别等任务上验证了这一几何信号的实用性。

**[LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](livestar_live_streaming_assistant_for_real-world_online_video_understanding.md)**

:   提出 LiveStar，一个始终在线的直播流视频理解助手，通过 Streaming Causal Attention Masks (SCAM) 训练策略和 Streaming Verification Decoding (SVeD) 推理框架，实现自适应响应时机判断，在 OmniStar 基准上语义正确性提升 19.5%，时间偏差降低 18.1%。

**[MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments](memtrack_evaluating_long-term_memory_and_state_tracking_in_multi-platform_dynami.md)**

:   提出 MEMTRACK 基准，评估 LLM 智能体在多平台（Slack/Linear/Git）动态环境中的长期记忆和状态追踪能力，揭示即使最强的 GPT-5 也仅达 60% 正确率。

**[MimeQA: Towards Socially-Intelligent Nonverbal Foundation Models](mimeqa_towards_socially-intelligent_nonverbal_foundation_models.md)**

:   构建首个基于哑剧视频的非语言社交推理基准 MimeQA，包含101个视频和806个QA对，覆盖三层问题层次（具象识别→场景理解→全局推理），揭示当前VideoLLMs在非语言社交理解上的严重不足（20-30% vs 人类86%）。

**[MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence](muvr_a_multi-modal_untrimmed_video_retrieval_benchmark_with_multi-level_visual_c.md)**

:   提出 MUVR 基准，面向长视频平台的多模态未剪辑视频检索任务，设计了以视频为中心的多模态查询格式（视频+文本+标签+掩码）和六级视觉对应匹配准则，包含 53K 视频和 1050 个查询，系统评估了检索模型和 MLLM 的局限性。

**[Neural Stochastic Flows: Solver-Free Modelling and Inference for SDE Solutions](neural_stochastic_flows_solver-free_modelling_and_inference_for_sde_solutions.md)**

:   提出 Neural Stochastic Flows（NSF），通过条件归一化流直接学习 SDE 的转移分布 $p(x_t \mid x_s)$，在架构上约束满足随机流性质（恒等、Markov、Chapman-Kolmogorov），实现了无需数值求解器的单步采样，在远距时间点上加速高达两个数量级。

**[NeuroPath: Neurobiology-Inspired Path Tracking and Reflection for Semantically Coherent Retrieval](neuropath_neurobiology-inspired_path_tracking_and_reflection_for_semantically_co.md)**

:   受神经生物学中海马体位置细胞导航与记忆巩固机制启发，提出 NeuroPath——一个基于语义路径追踪的 RAG 框架，通过 LLM 驱动的目标导向路径构建和后检索补全策略，在多跳问答任务上实现 recall@2 平均 16.3% 和 recall@5 平均 13.5% 的提升。

**[Open-World Drone Active Tracking with Goal-Centered Rewards](open-world_drone_active_tracking_with_goal-centered_rewards.md)**

:   提出首个开放世界无人机主动跟踪基准 DAT（24 个城市级场景、高保真动力学仿真），以及基于目标中心奖励函数和课程学习的强化学习跟踪方法 GC-VAT，在仿真器上达到约 72% 的跟踪成功率。

**[PASS: Path-Selective State Space Model for Event-Based Recognition](pass_path-selective_state_space_model_for_event-based_recognition.md)**

:   PASS提出路径选择性事件聚合与扫描（PEAS）模块和多面选择引导（MSG）损失，利用SSM的线性复杂度和频率泛化能力，实现了从10^6到10^9事件长度的广泛分布上的事件识别，并在推理频率变化时保持性能仅下降8.62%（基线下降20.69%）。

**[PixFoundation 2.0: Do Video Multi-Modal LLMs Use Motion in Visual Grounding?](pixfoundation_20_do_video_multi-modal_llms_use_motion_in_visual_grounding.md)**

:   通过提出四项运动中心的探测技术和 MoCentric-Bench 基准，证明当前视频多模态 LLM 在像素级视觉接地任务中未能真正利用运动信息，可被静态关键帧欺骗。

**[PreFM: Online Audio-Visual Event Parsing via Predictive Future Modeling](prefm_online_audio-visual_event_parsing_via_predictive_future_modeling.md)**

:   本文首次提出在线音视频事件解析（On-AVEP）范式，通过预测性未来建模框架 PreFM，利用伪未来序列增强当前上下文理解，同时借助模态无关的知识蒸馏和焦点时间优先策略，以仅 2.7% 的参数量超越离线 SOTA 方法 +9.3 的事件级平均 F1 分数。

**[QiMeng-NeuComBack: Self-Evolving Translation from IR to Assembly Code](qimeng-neucomback_self-evolving_translation_from_ir_to_assembly_code.md)**

:   提出NeuComBack基准数据集用于评估IR到汇编的神经编译任务，并设计自进化提示优化方法，通过从LLM自调试轨迹中学习来迭代改进编译提示，使正确率从44%提升到64%，且87.5%的正确程序性能超越clang-O3。

**[Revisiting Bi-Linear State Transitions in Recurrent Neural Networks](revisiting_bi-linear_state_transitions_in_recurrent_neural_networks.md)**

:   系统性地重新审视 RNN 中的双线性状态转移（隐状态与输入的乘法交互），理论证明双线性 RNN 可模拟任意有限状态机，并展示其在去除加性项后形成了一个从对角到全结构的自然表达力层次，揭示了 Mamba 等流行线性 RNN 处于该层次最低端。

**[SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models](sama_towards_multi-turn_referential_grounded_video_chat_with_large_language_mode.md)**

:   提出 SAMA 框架，通过构建统一的数据集（SAMA-239K）、模型（时空上下文聚合器 + SAM）和基准（SAMA-Bench），首次实现了多轮引用式视频对话中细粒度时空理解与grounding的联合建模。

**[Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition](seeing_beyond_the_scene_analyzing_and_mitigating_background_bias_in_action_recog.md)**

:   系统分析了动作识别中背景偏差在分类模型、对比预训练模型（CLIP/SigLIP2）和视频大语言模型（VLLM）三类范式中的普遍存在，并提出两条缓解路径：分类模型通过双分支架构融合分割人体输入降低SBErr 3.78%，VLLM通过自动化prompt tuning降低SBErr 9.85%。

**[Seeing the Arrow of Time in Large Multimodal Models](seeing_the_arrow_of_time_in_large_multimodal_models.md)**

:   本文揭示当前大多模态模型（LMMs）对视频时间方向性（时间箭头）出人意料地不敏感——正放/倒放时答案几乎相同，提出基于 GRPO 的 ArrowRL 训练策略引入反向视频奖励来激发时间方向感知，并构建 AoTBench 基准，在多个 VQA 基准上取得显著提升（Vinoground 上相对提升 65.9%）。

**[SmartWilds: Multimodal Wildlife Monitoring Dataset](smartwilds_multimodal_wildlife_monitoring_dataset.md)**

:   发布首个同步多模态野生动物监测数据集SmartWilds，整合无人机影像、相机陷阱和生物声学三种模态共101GB数据，通过GPS/时间戳实现跨模态对齐，建立可重复的保护监测标准协议，填补综合性生态系统多传感器融合数据集的空白。

**[Steering When Necessary: Flexible Steering Large Language Models with Backtracking](steering_when_necessary_flexible_steering_large_language_models_with_backtrackin.md)**

:   提出 FASB（Flexible Activation Steering with Backtracking）框架，通过跟踪 LLM 生成过程中的内部状态动态判断干预必要性和强度，并引入回溯机制纠正已偏离的 token，在 TruthfulQA 上 True*Info 达 80.56%、6 个多选任务平均准确率 78.8%，显著优于所有基线。

**[Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)**

:   本文提出 PD-SSM，一种结构化稀疏参数化方法用于状态空间模型（SSM）的状态转移矩阵。核心思想是将转移矩阵分解为列 one-hot 矩阵 P 与复数对角矩阵 D 的乘积（A = PD），从而在保持与对角 SSM 相当的计算效率（Θ(LN)）的同时，获得与非结构化（稠密）SSM 等同的表达能力——单层即可模拟任意 N 状态有限状态自动机（FSA）。理论上证明了该参数化的 BIBO 稳定性和最优状态维度。实验在 FSA 模拟、多元时序分类、长序列基准和自然语言状态追踪任务中均表现优异。

**[TAPVid-360: Tracking Any Point in 360 from Narrow Field of View Video](tapvid-360_tracking_any_point_in_360_from_narrow_field_of_view_video.md)**

:   本文提出TAPVid-360任务和数据集，要求模型在窄视野视频中跟踪查询点的3D方向（包括视野外的点），通过利用360度视频生成训练数据并微调CoTracker3实现方向预测，在视野外跟踪上远超现有方法。

**[TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs](tempsampr1_effective_temporal_sampling_with_reinforcement_fi.md)**

:   提出TempSamp-R1强化微调框架，针对GRPO在视频时序定位中因搜索空间巨大而on-policy采样低效的问题，通过引入GT作为off-policy监督信号+非线性软优势估计+混合CoT训练范式，在Charades-STA/ActivityNet/QVHighlights三个基准上达到新SOTA。

**[The Ouroboros of Benchmarking: Reasoning Evaluation in an Era of Saturation](the_ouroboros_of_benchmarking_reasoning_evaluation_in_an_era_of_saturation.md)**

:   本文通过对OpenAI、Anthropic和Google三大模型家族在52个推理基准上的系统分析，揭示了一种"衔尾蛇"循环模式——旧基准被快速饱和→新基准被创建以维持区分度→新基准又被迅速饱和，由此质疑基准分数的提升是否真正代表了推理能力的泛化进步，还是仅仅反映了对特定评测集的过拟合。

**[TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning](tirex_zero-shot_forecasting_across_long_and_short_horizons_with_enhanced_in-cont.md)**

:   提出基于xLSTM的预训练时间序列预测模型TiRex，通过连续片段掩码（CPM）策略和数据增强技术，在GiftEval和Chronos-ZS两大标准基准上以仅35M参数全面超越Chronos Bolt（200M）、TimesFM（500M）等大模型，同时在短期和长期零样本预测中均达到SOTA。

**[Token Bottleneck: One Token to Remember Dynamics](token_bottleneck_one_token_to_remember_dynamics.md)**

:   提出Token Bottleneck（ToBo），一种自监督视觉表征学习流水线，通过将参考场景压缩为单个瓶颈token、并利用该token与极少量目标场景patch来预测后续场景，使视觉骨干网络同时学会保守编码场景信息和捕获时间动态变化。

**[Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task](toolaugmented_spatiotemporal_reasoning_for_streamlining_vide.md)**

:   提出了包含 22 个工具的视频工具包和 STAR（Spatiotemporal Reasoning）框架，通过时间-空间工具交替调度策略渐进式定位 3D RoI，在 VideoMME 上将 GPT-4o 提升 8.2%，同时大幅减少处理帧数和计算开销。

**[Tracking and Understanding Object Transformations](tracking_and_understanding_object_transformations.md)**

:   提出 Track Any State 任务和 TubeletGraph 零样本框架，在视频中跟踪经历外观剧变的物体状态变化（如切苹果、蝴蝶从蛹中羽化），同时检测并描述这些变化。

**[TrackingWorld: World-centric Monocular 3D Tracking of Almost All Pixels](trackingworld_world-centric_monocular_3d_tracking_of_almost_all_pixels.md)**

:   提出TrackingWorld，一个从单目视频实现几乎所有像素的稠密3D跟踪的流水线，通过跟踪上采样器将稀疏2D轨迹提升为稠密轨迹、迭代跟踪所有帧中新出现的物体、以及基于优化的框架将2D轨迹提升到世界坐标系3D空间并显式分离相机运动和物体运动。

**[Two Causally Related Needles in a Video Haystack](two_causally_related_needles_in_a_video_haystack.md)**

:   提出Causal2Needles基准（4,100个问答对），通过设计"桥接实体"将两个因果相关事件的理解绑定在一起，强制VLM必须联合检索和推理两个分散在长视频中的"针"，揭示现有最强模型在因果双针问题上的严重不足（ChatGPT-4o双针Both准确率仅13.4%）。

**[Unleashing Hour-Scale Video Training for Long Video-Language Understanding](unleashing_hour-scale_video_training_for_long_video-language_understanding.md)**

:   构建首个大规模小时级视频指令跟随数据集 VideoMarathon（9700小时、330万QA对、22种任务），并提出 Hour-LLaVA 模型，通过记忆仓库+遗忘机制+MemAug模块实现1-FPS下小时级视频的高效训练与推理，在四个长视频基准上全面领先同规模开源模型。

**[VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding](vgent_graph-based_retrieval-reasoning-augmented_generation_for_long_video_unders.md)**

:   提出 VGEnt，一个基于图的检索-推理增强生成框架，通过构建视频知识图谱保留跨片段语义关系，并引入结构化推理步骤过滤噪声、聚合信息，在多个长视频理解基准上一致提升开源 LVLM 3.0%~5.4%，超越现有视频 RAG 方法 8.6%。

**[Video Finetuning Improves Reasoning Between Frames](video_finetuning_improves_reasoning_between_frames.md)**

:   本文通过提出视觉思维链（vCoT）方法，系统地比较了图像LLM与视频微调LLM在帧间推理能力上的差异，发现视频微调使模型隐式学会了帧间过渡推理，且这种能力可迁移到静态图像的关系推理任务中。

**[VideoLucy: Deep Memory Backtracking for Long Video Understanding](videolucy_deep_memory_backtracking_for_long_video_understanding.md)**

:   提出VideoLucy框架，通过层次化记忆结构和基于Agent的迭代回溯机制，模拟人类从粗到细的回忆过程，在多个长视频理解基准上大幅超越现有方法，甚至超过GPT-4o等商业模型。

**[Web-Scale Collection of Video Data for 4D Animal Reconstruction](web-scale_collection_of_video_data_for_4d_animal_reconstruction.md)**

:   提出一个全自动化的大规模视频数据采集管线，从 YouTube 挖掘并处理得到 30K 动物视频（2M帧），建立首个 4D 四足动物重建基准 Animal-in-Motion（230序列/11K帧），并提出 4D-Fauna 基线方法实现序列级优化的无模型 4D 重建。

**[When One Moment Isn't Enough: Multi-Moment Retrieval with Cross-Moment Interactions](when_one_moment_isnt_enough_multi-moment_retrieval_with_cross-moment_interaction.md)**

:   提出QV-M2数据集（首个全人工标注的多时刻检索基准）和FlashMMR框架（含后验证模块），将视频时刻检索从单时刻扩展到多时刻场景，建立了多时刻检索的标准化评价体系。

**[When Thinking Drifts: Evidential Grounding for Robust Video Reasoning](when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)**

:   系统揭示了CoT推理在视频理解中经常导致性能下降的"视觉思维漂移"现象，并提出Visual Evidence Reward（VER）强化学习框架，通过显式奖励与视觉证据对齐的推理链来纠正这一问题。
