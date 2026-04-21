---
title: >-
  AAAI2026 目标检测方向 42篇论文解读
description: >-
  42篇AAAI2026 目标检测方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**🤖 AAAI2026** · **42** 篇论文解读

**[A Theoretical Analysis of Detecting Large Model-Generated Time Series](a_theoretical_analysis_of_detecting_large_model-generated_time_series.md)**

:   首次提出时间序列大模型（TSLM）生成内容检测理论框架，通过收缩假说（Contraction Hypothesis）揭示TSLM生成序列在递归预测下不确定性指数级衰减的本质特征，据此设计UCE检测器，在32个数据集上In-Distribution AUROC达0.855，显著超越10种文本检测baseline。

**[Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward](actor-critic_for_continuous_action_chunks_a_reinforcement_le.md)**

:   AC3 提出了一个直接学习连续动作序列（action chunk）的 actor-critic 框架，通过"仅从成功轨迹更新 actor"的非对称更新规则和基于自监督锚点的内在奖励来稳定稀疏奖励下的长时域机器人操作学习，在 BiGym 和 RLBench 的 25 个任务上取得优于现有方法的成功率。

**[AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios](aerialmind_towards_referring_multi-object_tracking_in_uav_sc.md)**

:   构建了首个面向无人机场景的大规模 Referring Multi-Object Tracking（RMOT）基准数据集 AerialMind，并提出 HawkEyeTrack（HETrack）方法，通过视觉-语言共进化融合编码器和尺度自适应上下文精炼模块，在无人机航拍场景中实现语言引导的多目标跟踪。

**[An Overall Real-Time Mechanism for Classification and Quality Evaluation of Rice](an_overall_real-time_mechanism_for_classification_and_quality_evaluation_of_rice.md)**

:   提出一个实时大米品质评估整体机制，整合改进的 YOLO-v5（品种检测）、改进的 ConvNeXt-Tiny（完整度分级）和 K-means（垩白区域量化）三个模块，在自建的六品种两万张图像数据集上实现了 99.14% mAP 和 97.89% 检测准确率。

**[Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection](beyond_boundaries_leveraging_vision_foundation_models_for_so.md)**

:   提出利用VFM（DINOv2+Grounding DINO）增强无源域自适应目标检测（SFOD）的框架，通过全局特征对齐(PGFA)、实例级原型对比学习(PIFA)和双源伪标签融合(DEPF)三个模块，在6个跨域检测基准上取得SOTA，例如Cityscapes→Foggy Cityscapes达47.1% mAP（比DRU高3.5%），Sim10k→Cityscapes达67.4% AP（比DRU高8.7%）。

**[Beyond Fact Retrieval: Episodic Memory for RAG with Generative Semantic Workspaces](beyond_fact_retrieval_episodic_memory_for_rag_with_generative_semantic_workspace.md)**

:   提出 Generative Semantic Workspace (GSW)，一种神经科学启发的生成式记忆框架，为 LLM 构建结构化的情景记忆表示，在 EpBench 上 F1 达到 0.85，同时减少 51% 的查询时上下文 token。

**[Beyond Semantic Features: Pixel-Level Mapping for Generalized AI-Generated Image Detection](beyond_semantic_features_pixel-level_mapping_for_generalized_ai-generated_image_.md)**

:   提出像素级映射（pixel-level mapping）预处理方法，通过打破像素值的单调排列来抑制低频语义偏差、增强高频生成伪影，将 AI 生成图像检测的跨模型泛化准确率提升至 98.4%。

**[Connecting the Dots: Training-Free Visual Grounding via Agentic Reasoning](connecting_the_dots_training-free_visual_grounding_via_agent.md)**

:   提出 GroundingAgent，一个完全不需要任务特定微调的视觉定位框架，通过组合预训练的开放词汇检测器（YOLO World）、MLLM（Llama-3.2-11B-Vision）和 LLM（DeepSeek-V3）进行结构化迭代推理，在 RefCOCO/+/g 上实现 65.1% 的零样本平均准确率，大幅超越之前的 zero-shot 方法。

**[Continuous Vision-Language-Action Co-Learning with Semantic-Physical Alignment for Behavioral Cloning](continuous_vision-language-action_co-learning_with_semantic-.md)**

:   提出CCoL框架，通过NeuralODE驱动的多模态连续协同学习（MCC）和双向交叉注意力的语义-物理对齐（CSA），在Behavioral Cloning中同时解决动作序列的物理不连续性和语义-物理失配问题，在三个仿真平台上平均相对提升8.0%，双臂插入任务最高达19.2%。

**[CTPD: Cross Tokenizer Preference Distillation](ctpd_cross_tokenizer_preference_distillation.md)**

:   提出 Cross-Tokenizer Preference Distillation (CTPD)，首个支持不同分词器间偏好蒸馏的统一框架，通过 Aligned Span Projection、跨分词器重要性加权和 Teacher-Anchored Reference 三项创新，在多个 benchmark 上显著超越现有方法。

**[Deep Incomplete Multi-View Clustering via Hierarchical Imputation and Alignment](deep_incomplete_multi-view_clustering_via_hierarchical_imputation_and_alignment.md)**

:   提出 DIMVC-HIA，一个集成层次化填充与双重对齐的深度不完整多视图聚类框架，先填充缺失聚类分配再填充缺失特征，在高缺失率（70%）下仍保持稳健性能。

**[Sketch-HARP: 分层自回归草图生成实现灵活笔画级绘制操控](generating_sketches_in_a_hierarchical_auto-regressive_proces.md)**

:   提出 Sketch-HARP 分层自回归草图生成框架，通过三阶段层次化过程（预测笔画嵌入→确定画布位置→生成绘制动作序列），首次实现草图绘制过程中的灵活笔画级操控，在替换/擦除/扩展等任务上显著优于 SketchEdit。

**[Ground What You See: Hallucination-Resistant MLLMs via Caption Feedback, Diversity-Aware Sampling, and Conflict Regularization](ground_what_you_see_hallucination-resistant_mllms_via_caption_feedback_diversity.md)**

:   针对多模态大模型（MLLM）在强化学习训练中产生幻觉的三大根因——视觉误解、探索多样性不足、样本冲突——分别提出 Caption Reward、奖励方差引导的样本选择、以及基于 NTK 相似度的 InfoNCE 正则化，在多个基准上显著降低幻觉率。

**[H-GAR: A Hierarchical Interaction Framework via Goal-Driven Observation-Action Refinement for Robotic Manipulation](h-gar_a_hierarchical_interaction_framework_via_goal-driven_observation-action_re.md)**

:   提出层次化目标驱动框架 H-GAR，通过先预测目标观测再合成中间观测、并利用历史动作记忆库细化粗粒度动作，实现了观测与动作的显式双向交互，在仿真和真实机器人操控任务上取得 SOTA。

**[How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts](how_many_experts_are_enough_towards_optimal_semantic_specialization_for_mixture-.md)**

:   提出MASS框架，通过基于梯度的语义漂移检测自适应扩展MoE专家池，并结合Top-p置信度路由策略，在无需超参搜索的情况下自动发现最优专家数量，同时增强专家间的语义分化。

**[Learning Procedural-aware Video Representations through State-Grounded Hierarchy Unfolding](learning_procedural-aware_video_representations_through_state-grounded_hierarchy.md)**

:   提出 Task-Step-State（TSS）三层语义框架，在传统的任务-步骤层次中引入"状态"作为视觉锚定层，并设计渐进式预训练策略（Task→Step→State→Step→Task）逐步展开 TSS 层次，在 COIN 和 CrossTask 数据集上的任务识别、步骤识别和步骤预测任务上全面超越 SOTA。

**[Leveraging Textual Compositional Reasoning for Robust Change Captioning](leveraging_textual_compositional_reasoning_for_robust_change_captioning.md)**

:   提出 CORTEX 框架，通过引入 VLM 生成的组合推理文本作为显式线索，结合图像-文本双重对齐模块（ITDA），增强纯视觉变化描述方法对物体关系和空间配置等结构化语义的理解能力。

**[Lost in Translation? A Comparative Study on the Cross-Lingual Transfer of Composite Harms](lost_in_translation_a_comparative_study_on_the_cross-lingual_transfer_of_composi.md)**

:   提出 CompositeHarm 基准，通过将对抗语法攻击（AttaQ）和语境化危害（MMSafetyBench）翻译为五种印度语言，系统研究了 LLM 安全对齐在跨语言场景下的脆弱性，发现对抗语法攻击在印度语言中攻击成功率急剧攀升。

**[Mitigating Error Accumulation in Co-Speech Motion Generation via Global Rotation Diffusion and Multi-Level Constraints](mitigating_error_accumulation_in_co-speech_motion_generation_via_global_rotation.md)**

:   提出 GlobalDiff 框架，首次在全局关节旋转空间中进行扩散生成，从根本上消除层次化前向运动学中的误差累积问题，并通过关节-骨骼-运动三层约束方案弥补全局表示丢失的结构先验，在多说话人语音驱动动作生成基准上取得 SOTA，FGD 较此前最佳方法改进 46%。

**[PASE: Leveraging the Phonological Prior of WavLM for Low-Hallucination Generative Speech Enhancement](pase_leveraging_the_phonological_prior_of_wavlm_for_low-hallucination_generative.md)**

:   提出 PASE 框架，通过去噪表示蒸馏（DRD）利用预训练 WavLM 中鲁棒的音韵先验来抑制语言幻觉，同时采用双流表示（高层音素 + 低层声学）消除声学幻觉，在感知质量和内容保真度两方面同时达到 SOTA。

**[Perceive, Act and Correct: Confidence Is Not Enough for Hyperspectral Classification](perceive_act_and_correct_confidence_is_not_enough_for_hyperspectral_classificati.md)**

:   提出 CABIN 框架，通过认知感知-行动-纠正的闭环学习机制，利用认识论不确定性（epistemic uncertainty）替代单纯的置信度来指导半监督高光谱图像分类中的样本选择与伪标签管理，在仅用 75% 标注的情况下显著超过全标注基线。

**[Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning](resource_efficient_sleep_staging_via_multi-level_masking_and_prompt_learning.md)**

:   提出 MASS (Mask-Aware Sleep Staging) 框架，通过多层级 masking 策略和层次化 prompt learning 机制，仅用 **10% 的原始 EEG 信号**即可实现可靠的睡眠分期，为资源受限的可穿戴睡眠监测系统提供方案。

**[REXO: Indoor Multi-View Radar Object Detection via 3D Bounding Box Diffusion](rexo_indoor_multi-view_radar_object_detection_via_3d_bounding_box_diffusion.md)**

:   将 DiffusionDet 的 2D BBox 扩散范式提升到 3D 雷达空间，提出 REXO 框架：通过含噪 3D BBox 的投影引导显式跨视图雷达特征关联，并引入地面约束减少扩散参数，在 HIBER 和 MMVR 两个室内雷达数据集上分别超越 SOTA +4.22 AP 和 +11.02 AP。

**[Robust Long-term Test-Time Adaptation for 3D Human Pose Estimation through Motion Discretization](robust_long-term_test-time_adaptation_for_3d_human_pose_estimation_through_motio.md)**

:   针对 3D 人体姿态估计在线测试时自适应中的误差累积问题，提出基于运动离散化（无监督聚类获得锚运动集）+ 自回放机制 + 软重置策略的解决方案，使模型能在长时间持续适应中稳健利用个人形态和习惯性运动特征，在 Ego-Exo4D 和 3DPW 上超越所有现有在线 TTA 方法。

**[SAGA: Learning Signal-Aligned Distributions for Improved Text-to-Image Generation](saga_learning_signal-aligned_distributions_for_improved_text-to-image_generation.md)**

:   提出SAGA方法，通过学习与提示词对齐的高斯分布来改进文本到图像生成模型的语义对齐，无需重新训练且支持文本和空间双条件生成，在SD 1.4和SD 3上大幅提升对齐性能（TIAM-3从8.4%提升到50.7%）。

**[SemanticVLA: Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation](semanticvla_semantic-aligned_sparsification_and_enhancement_for_efficient_roboti.md)**

:   提出 SemanticVLA 框架，通过语义引导的双视觉编码器剪枝（SD-Pruner）、语义互补层次融合（SH-Fuser）和语义条件动作耦合（SA-Coupler）三个模块，在大幅减少视觉冗余的同时增强指令-视觉-动作对齐，在 LIBERO 基准上以 97.7% 成功率超越 OpenVLA 达 21.1%，同时训练成本和推理延迟分别降低 3.0× 和 2.7×。

**[SimROD: A Simple Baseline for Raw Object Detection with Global and Local Enhancements](simrod_a_simple_baseline_for_raw_object_detection_with_global_and_local_enhancem.md)**

:   提出SimROD，一种极其轻量（仅0.003M参数）的RAW图像目标检测方法，通过全局Gamma增强（4个可学习参数）和绿色通道引导的局部增强，在多个RAW检测基准上超越了复杂的SOTA方法。

**[SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection](sm3det_a_unified_model_for_multi-modal_remote_sensing_object_detection.md)**

:   SM3Det提出了遥感领域的M2Det新任务（多模态数据集+多任务目标检测），通过网格级稀疏MoE骨干网络和动态子模块优化（DSO）机制，用单一模型同时处理SAR/光学/红外三种模态的水平/旋转框检测，显著超越各模态独立训练的三个专用模型组合。

**[T-Rex-Omni: Integrating Negative Visual Prompt in Generic Object Detection](t-rex-omni_integrating_negative_visual_prompt_in_generic_object_detection.md)**

:   提出T-Rex-Omni框架，首次将负视觉提示（negative visual prompts）系统性地引入开放集目标检测，通过训练免费的NNC模块和NNH损失，显著缩小了视觉提示和文本提示检测方法之间的性能差距，在长尾场景中表现尤为突出（LVIS-minival APr达到51.2）。

**[Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack](tackling_resource-constrained_and_data-heterogeneity_in_federated_learning_with_.md)**

:   提出FedCSPACK，一种基于余弦稀疏化参数打包和双权重聚合的个性化联邦学习方法，通过在包级别进行参数选择和共享，同时平衡了数据异质性和客户端资源约束，训练速度提升2-5倍、通信量压缩高达96%，同时模型精度提升3.34%。

**[Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection](temporal_object-aware_vision_transformer_for_few-shot_video_object_detection.md)**

:   提出一种对象感知的时序建模框架，通过选择性传播高置信度检测特征实现跨帧时序一致性，结合预训练视觉-语言编码器（OWL-ViT）和少样本检测头，在四个视频少样本检测基准上平均提升3.7%-5.3% AP。

**[Test-time Diverse Reasoning by Riemannian Activation Steering](test-time_diverse_reasoning_by_riemannian_activation_steering.md)**

:   提出 SPREAD 框架——一种无监督的测试时激活引导策略，通过在球面流形乘积上求解黎曼优化问题来最大化多条推理路径的隐藏激活张成的总体积，从而提升 Best-of-N 采样中的推理多样性和准确率，在数学推理基准上超越温度采样基线。

**[Towards Effective, Stealthy, and Persistent Backdoor Attacks Targeting Graph Foundation Models](towards_effective_stealthy_and_persistent_backdoor_attacks_targeting_graph_found.md)**

:   提出 GFM-BA，首个系统性地针对 Graph Foundation Models (GFMs) 预训练阶段的后门攻击方法，通过 label-free trigger 关联、node-adaptive trigger 生成和 persistent backdoor anchoring 三个模块，同时解决有效性、隐蔽性和持久性三大挑战。

**[Towards Scalable Web Accessibility Audit with MLLMs as Copilots](towards_scalable_web_accessibility_audit_with_mllms_as_copilots.md)**

:   提出 AAA 框架，通过 GRASP（基于图的多模态页面采样）和 MaC（MLLM 作为 Copilot）两大创新，将 WCAG-EM 标准操作化，实现可扩展的端到端网页无障碍审计。

**[TRACE: A Generalizable Drift Detector for Streaming Data-Driven Optimization](trace_a_generalizable_drift_detector_for_streaming_data-driven_optimization.md)**

:   提出TRACE，一种基于注意力序列学习的可迁移概念漂移检测器，通过统计特征标记化和双注意力编码器学习跨任务可迁移的漂移模式，能泛化到未见过的数据集，并作为即插即用模块嵌入流式数据驱动优化算法。

**[TTF-VLA: Temporal Token Fusion via Pixel-Attention Integration for Vision-Language-Action Models](ttf-vla_temporal_token_fusion_via_pixel-attention_integratio.md)**

:   TTF-VLA 提出了一种免训练的时序 Token 融合方法，通过灰度像素差异+注意力语义检测的双维度机制选择性地复用历史帧的视觉 Token，提升 VLA 模型在机器人操作任务中的推理质量，在 LIBERO 上平均提升 4.0 个百分点。

**[TubeRMC: Tube-conditioned Reconstruction with Mutual Constraints for Weakly-supervised Spatio-Temporal Video Grounding](tubermc_tube-conditioned_reconstruction_with_mutual_constraints_for_weakly-super.md)**

:   提出 TubeRMC 框架，利用文本条件化的候选 tube 生成 + 从时间/空间/时空三个维度进行 tube 条件化重建，并引入空间-时间互约束来增强弱监督时空视频定位性能。

**[VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)**

:   提出 VK-Det 框架，仅利用 VLM 的视觉知识（无需额外监督信号），通过自适应选择知识蒸馏（ASKD）+ 原型感知伪标签（PAPL）+ 综合匹配推理（SMI），在航空遥感开放词汇目标检测中达到 SOTA，甚至超越使用额外监督的方法。

**[When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets](when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)**

:   提出 CAIA 基准测试，通过加密货币市场作为天然对抗性实验室，评估 17 个 SOTA 大模型在高风险对抗环境中的 agent 能力，揭示前沿模型仅达 67.4% 准确率（GPT-5）vs 人类 80%，并发现系统性工具选择灾难。

**[When Trackers Date Fish: A Benchmark and Framework for Underwater Multiple Fish Tracking](when_trackers_date_fish_a_benchmark_and_framework_for_underwater_multiple_fish_t.md)**

:   提出 MFT25 大规模水下多鱼跟踪数据集（15 序列, 408K 标注）和 SU-T 跟踪框架（UKF + FishIoU），实现 34.1 HOTA 和 44.6 IDF1 的 SOTA 性能，并通过统计分析揭示鱼类跟踪与陆地目标跟踪的本质差异。

**[YOLO-IOD: Towards Real Time Incremental Object Detection](yolo-iod_towards_real_time_incremental_object_detection.md)**

:   首次系统性地将增量目标检测（IOD）引入 YOLO 实时框架，识别三种知识冲突类型，提出 CPR + IKS + CAKD 三模块协同解决方案，并引入更真实的 LoCo COCO 基准评估。

**[Your AI-Generated Image Detector Can Secretly Achieve SOTA Accuracy, If Calibrated](your_ai-generated_image_detector_can_secretly_achieve_sota_accuracy_if_calibrate.md)**

:   提出一种基于贝叶斯决策理论的轻量级后验校准方法，通过在模型输出logit上添加可学习标量偏移α，无需重训练即可显著提升现有AI生成图像检测器在分布偏移下的准确率。
