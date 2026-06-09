---
title: >-
  CVPR2026 机器人/具身智能方向42篇论文解读
description: >-
  42篇CVPR2026的机器人/具身智能方向论文解读，涵盖机器人、多模态、推理、导航、Agent、扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "机器人/具身智能"
  - "论文解读"
  - "论文笔记"
  - "机器人"
  - "多模态"
  - "推理"
  - "导航"
  - "Agent"
  - "扩散模型"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 机器人/具身智能

**📷 CVPR2026** · **42** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (42)](../../ICML2026/robotics/index.md) · [💬 ACL2026 (10)](../../ACL2026/robotics/index.md) · [🔬 ICLR2026 (46)](../../ICLR2026/robotics/index.md) · [🤖 AAAI2026 (30)](../../AAAI2026/robotics/index.md) · [🧠 NeurIPS2025 (75)](../../NeurIPS2025/robotics/index.md) · [📹 ICCV2025 (26)](../../ICCV2025/robotics/index.md)

🔥 **高频主题：** 机器人 ×17 · 多模态 ×12 · 推理 ×8 · 导航 ×6 · Agent ×4

**[Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)**

:   提出自适应动作分块(AAC)策略，利用动作熵作为线索在推理时动态确定最优分块大小，无需额外训练或架构修改，在RoboCasa和LIBERO等基准上持续提升GR00T N1.5和π0.5的任务成功率。

**[AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots](atomicvla_unlocking_the_potential_of_atomic_skill_learning_in_robots.md)**

:   提出AtomicVLA，在π₀基础上构建统一规划-执行框架，通过自适应Think-Act切换生成原子技能抽象，并用技能引导的MoE（SG-MoE）将动作路由到专精expert执行，LIBERO-LONG成功率从85.2%提升至95.2%（+10%），真实Franka长任务+18.3%，持续学习+21%。

**[AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention](ava_vla_improving_vision_language_action_models_with_active_visual_attention.md)**

:   从POMDP视角重新审视VLA模型的视觉处理，提出AVA-VLA框架通过循环状态和主动视觉注意力模块，根据历史上下文动态调制当前帧的视觉token重要性，在LIBERO和CALVIN等基准上达到SOTA。

**[BiPreManip: Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipatory Collaboration](bipremanip_learning_affordance-based_bimanual_preparatory_manipulation_through_a.md)**

:   提出 BiPreManip 框架，基于视觉可供性表示实现双臂预备操作：先预想主手的目标交互区域，再引导辅助手进行预备动作（如翻转瓶子使瓶盖朝向主手），在仿真和真实环境中大幅优于基线。

**[Boosting Vision-Language-Action Finetuning with Feasible Action Neighborhood Prior](boosting_vision-language-action_finetuning_with_feasible_action_neighborhood_pri.md)**

:   提出可行动作邻域（FAN）正则化器，将 VLA 模型的输出分布塑造为与物理动作容差匹配的高斯形状，在 SFT 和 RFT 两种微调范式下均显著提升成功率、泛化性和样本效率（RFT 仅需 1/3 训练步数达到 90% 成功率）。

**[Chain of World: World Model Thinking in Latent Motion (CoWVLA)](chain_of_world_world_model_thinking_in_latent_motion.md)**

:   提出CoWVLA，统一世界模型VLA和隐动作VLA的优势：通过Latent Motion Extractor将视频分解为结构隐变量和运动隐变量，VLA在隐运动空间做世界模型预测而非重建冗余像素，配合Co-Fine-tuning交替生成关键帧和动作token，LIBERO-LONG达95.2%超越π₀(85.2%)，SimplerEnv-WidowX avg 0.560超π₀(0.425)。

**[CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning](como_learning_continuous_latent_motion_from_internet_videos_for_scalable_robot_l.md)**

:   提出 CoMo，通过早期时序差分（Td）和时序对比学习（Tcl）两个机制协同解决连续隐运动学习中的捷径学习问题，从互联网视频中提取精细的连续伪动作标签，使视频数据与机器人动作在统一连续分布下联合训练，显著提升策略性能。

**[Cross-Domain Demo-to-Code via Neurosymbolic Counterfactual Reasoning](cross-domain_demo-to-code_via_neurosymbolic_counterfactual_reasoning.md)**

:   提出 NeSyCR 神经符号反事实推理框架，将视频示教抽象为符号世界模型，通过反事实状态推演检测跨域不兼容并自动修正程序步骤，在跨域 demo-to-code 任务上比最强基线 Statler 提升 31.14% 成功率。

**[CycleManip: Enabling Cyclic Task Manipulation via Effective Historical Perception and Understanding](cyclemanip_enabling_cyclic_task_manipulation_via_effective_historical_percepti.md)**

:   CycleManip 首次系统性地解决机器人循环操作任务（如摇瓶子N次），通过成本感知的历史采样策略增强历史感知，配合多任务学习辅助目标提升历史理解，以端到端模仿学习方式实现循环次数可控的操作。

**[DAWN: Pixel Motion Diffusion is What We Need for Robot Control](dawn_pixel_motion_diffusion_robot_control.md)**

:   提出 DAWN，一个两阶段全扩散的视觉语言动作框架——Motion Director（潜扩散模型）生成稠密像素运动场作为可解释的中间表示，Action Expert（扩散 Transformer 策略）将像素运动转换为可执行机器人动作；在 CALVIN 基准上取得 SOTA（平均长度 4.00），并在真实世界单臂/双臂操控中展现强泛化能力。

**[DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation](decovln_decoupling_observation_reasoning_and_correction_for_vision-and-language_.md)**

:   提出 DecoVLN 框架，将 VLN 任务中的观察、推理和纠错三个过程解耦，通过自适应记忆优化机制和基于状态-动作对的纠错微调策略，在仅使用自中心 RGB 输入的条件下实现了 R2R-CE 和 RxR-CE 上的 SOTA 性能。

**[Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols](diagnose_correct_and_learn_from_manipulation_failures_via_visual_symbols.md)**

:   提出 ViFailback 框架，利用可视化符号（箭头、准星、标签等）高效标注真实世界机器人操作失败，构建 58,128 个 VQA 对的数据集，并训练 ViFailback-8B VLM 实现失败诊断和视觉+文本纠正指导，集成到 VLA 后实现 22.2% 的任务成功率提升。

**[Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)**

:   针对MoE-LoRA方法中所有expert结构相同（统一rank）导致无法适配不同复杂度任务的问题，提出EPT：通过共享meta-knowledge子空间 + 不同kernel size的反卷积expert构建参数金字塔，配合Adaptive LoRA Pruner和对比学习Task Embedding，在GLUE上以仅0.41M参数/任务达到87.0%平均分，超越所有MoE-LoRA变体。

**[Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning](fast-thinkact_efficient_vision-language-action_reasoning_via_verbalizable_latent.md)**

:   提出 Fast-ThinkAct，通过将冗长的文本 CoT 推理（~250 token）压缩为 6 个可语言化的连续 latent token，结合 reward-guided preference distillation 和 visual trajectory alignment，实现 89.3% 推理延迟降低（9.3× faster than ThinkAct-7B）同时保持甚至超越 SOTA reasoning VLA 的性能。

**[FineCog-Nav: Integrating Fine-grained Cognitive Modules for Zero-shot Multimodal UAV Navigation](finecog_nav_fine_grained_cognitive_modules_for_zero_shot_uav_navigation.md)**

:   本文提出 FineCog-Nav，一个受人类认知启发的零样本 UAV 视觉语言导航框架，将导航分解为语言处理、感知、注意力、记忆、想象、推理和决策七个细粒度认知模块，每个模块使用中等规模基础模型驱动，无需训练即可在复杂 3D 环境中完成长程导航。

**[ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for Contact-Rich Manipulation](forcevla2_unleashing_hybrid_force-position_control_with_force_awareness_for_cont.md)**

:   提出ForceVLA2，首个在VLA框架中统一力感知(force awareness)与混合力-位置控制(hybrid force-position control)的端到端模型：通过Force-based Prompts在VLM中构建跨阶段力感知任务概念，Cross-Scale MoE自适应融合任务语义与实时交互力实现闭环力-位置调节，在5个contact-rich任务上平均成功率66%，超π₀和π₀.5分别48.0%和35.0%。

**[GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](gecosrt_geometryaware_continual_adaptation_for_rob.md)**

:   GeCo-SRT提出持续跨任务Sim-to-Real迁移范式，利用局部几何特征的域不变性和任务不变性，通过几何感知MoE模块提取可复用的几何知识并用专家引导的优先经验回放防遗忘，在4个操作任务上比基线平均提升52%成功率且仅需1/6数据。

**[GraspLDP: Towards Generalizable Grasping Policy via Latent Diffusion](graspldp_towards_generalizable_grasping_policy_via_latent_diffusion.md)**

:   提出 GraspLDP，将预训练抓取检测器的 grasp pose 先验和 graspness map 视觉线索注入潜在扩散策略框架，通过 VAE 编码的动作潜空间引导和自监督重建目标，显著提升抓取精度和泛化能力。

**[HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models](hif-vla_hindsight_insight_and_foresight_through_motion_representation_for_vision.md)**

:   提出 HiF-VLA 框架，通过运动向量（Motion Vector）作为紧凑时间原语，统一回顾（Hindsight）、洞察（Insight）和前瞻（Foresight）三种时间推理能力，实现 VLA 模型的双向时间扩展，在长时操作任务中以极低计算开销大幅超越基线。

**[IGen: Scalable Data Generation for Robot Learning from Open-World Images](igen_scalable_data_generation_for_robot_learning_from_open-world_images.md)**

:   IGen 从单张开放世界图像出发，通过3D场景重建→VLM任务规划→SE(3)动作生成→点云合成→帧渲染，自动生成大规模视觉-动作训练数据，仅用生成数据训练的策略即可完成真实世界操作。

**[Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](influence_malleability_in_linearized_attention_dual_implications_of_non-converge.md)**

:   本文揭示线性化注意力机制在 NTK 框架下不收敛至无穷宽极限，并提出"影响力可塑性"(influence malleability) 度量，证明注意力的强大能力与对抗脆弱性共享同一来源——偏离核regime的数据依赖核结构。

**[Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](lada_robotic_manipulation.md)**

:   提出 LaDA 框架，用自然语言作为语义桥梁将连续 7-DoF 动作解耦为平移/旋转/夹爪三个可解释原语，通过软标签对比学习在共享嵌入空间中对齐跨任务动作表示，仅 0.6B 参数在 LIBERO 上达 93.6% 成功率，超越 1.3B~8.5B 参数的所有基线。

**[Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)**

:   提出 TVVE 框架，通过强化学习驱动的多视角探索策略（MVEP）选择最优虚拟相机视角并在线重渲染观测，同时设计任务感知 MoE 视觉编码器（TaskMoE）解决多任务特征干扰问题，在 RLBench 18 个任务上平均成功率达 86.6%。

**[Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment](lifelong_imitation_learning_multimodal_latent_rep.md)**

:   提出终身模仿学习框架，通过多模态潜在回放（MLR）在冻结编码器的特征空间中存储和回放紧凑表示，并引入增量特征调整（IFA）机制用角距离约束维持任务间可分性，在LIBERO基准上AUC提升10-17点、遗忘降低最多65%。

**[ManipArena: Comprehensive Real-world Evaluation of Reasoning-Oriented Generalist Robot Manipulation](maniparena_comprehensive_real-world_evaluation_of_reasoning-oriented_generalist_.md)**

:   ManipArena 提出了一个标准化的真实世界机器人操作评估框架，包含 20 个推理导向任务和 10,812 条专家轨迹，通过绿幕受控环境、系统化多样性设计和分层 OOD 评估，为 VLA 模型和世界模型提供公平、可复现的评测基准。

**[MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent](mergevla_cross-skill_model_merging_toward_a_generalist_vision-language-action_ag.md)**

:   首次系统诊断 VLA 模型不可合并的两大根因（LoRA 自私参数冲突 + 动作专家自注意力导致的任务耦合），提出 MergeVLA——通过任务掩码稀疏激活 LoRA、去自注意力动作专家、无训练测试时路由，将多个单技能 VLA 专家合并为一个通用 agent，在 LIBERO 上达 90.2% 成功率，真机 SO101 达 90%。

**[PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](palm_progress-aware_policy_learning_via_affordance_reasoning_for_long-horizon_ro.md)**

:   提出 PALM，一个统一的 VLA 框架，通过结构化的细粒度可供性预测（全局/局部/空间/动态四类）作为隐式推理锚点，结合连续子任务进度估计实现无缝任务切换，在 CALVIN ABCD 上平均完成长度达 4.48（超越前 SOTA 12.5%），LIBERO-LONG 成功率 91.8%，真实世界长时域泛化测试中达到基线 2 倍以上。

**[PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments](panoaffordancenet_towards_holistic_affordance_grounding_in_360_indoor_environmen.md)**

:   提出首个面向360°全景室内环境的整体affordance定位框架PanoAffordanceNet，通过畸变感知频谱调制器(DASM)和全球面稠密化头(OSDH)系统性解决ERP几何畸变、稀疏功能区域和语义漂移问题，并构建了首个全景affordance数据集360-AGD。

**[Pixel-level Scene Understanding in One Token: Visual States Need What-is-Where Composition](pixel-level_scene_understanding_in_one_token_visual_states_need_what-is-where_co.md)**

:   本文提出CroBo，一个通过全局-局部重建目标学习视觉状态表示的自监督框架：将全局参考图压缩为单个瓶颈token，用它来重建高度遮蔽（90%）的局部裁剪视图，迫使瓶颈token编码像素级的"what-is-where"场景组成信息，在Franka Kitchen和DMC机器人策略学习benchmark上达到SOTA。

**[ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation](profocus_proactive_perception_and_focused_reasoning_in_vision-and-language_navig.md)**

:   提出 ProFocus，一个免训练的渐进式框架，通过主动感知（将全景图转为语义地图+LLM 生成针对性视觉查询）和聚焦推理（BD-MCTS 从大量历史路点中筛选 top-k 高价值候选），在 R2R 和 REVERIE 基准上达到零样本方法的 SOTA。

**[PULSE: Privileged Knowledge Transfer from Rich to Deployable Sensors for Embodied Multi-Sensory Learning](pulse_privileged_knowledge_transfer_from_rich_to_deployable_sensors_for_embodied.md)**

:   本文提出 PULSE 框架，通过冻结的特权传感器（如 EDA）教师模型向廉价可部署传感器（如 ECG、BVP、加速度计）学生模型进行知识蒸馏，引入共享-私有嵌入分解和重建防崩塌机制，在不使用 EDA 推理的情况下达到 0.994 AUROC 的压力检测性能，甚至超越使用全部传感器的模型。

**[QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)**

:   提出 QuantVLA，首个面向 Vision-Language-Action (VLA) 模型的免训练后量化框架，通过选择性量化布局和两个轻量级标定机制（注意力温度匹配 ATM 和输出头平衡 OHB），在 W4A8 精度下实现约 70% 的内存节省，同时任务成功率超过全精度基线。

**[RADAR: Closed-Loop Robotic Data Generation via Semantic Planning and Autonomous Causal Environment Reset](radar_closedloop_robotic_data_generation_via_seman.md)**

:   提出RADAR——一个完全自主的闭环机器人操作数据生成引擎，通过VLM语义规划+GNN策略执行+VQA成功评估+FSM驱动的LIFO因果逆序环境重置四个模块，仅需2-5个人工演示即可持续生成高保真操作数据，在仿真中复杂长horizon任务达到90%成功率。

**[Recurrent Reasoning with Vision-Language Models for Estimating Long-Horizon Embodied Task Progress](recurrent_reasoning_with_vision-language_models_for_estimating_long-horizon_embo.md)**

:   提出 R²VLM，通过循环推理框架逐步处理本地视频片段，维护动态更新的 CoT 记录任务分解和完成状态，结合多维 RL 奖励实现长时域具身任务进度估计的 SOTA，并支持策略学习、奖励建模、主动辅助等下游应用。

**[Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation](rethinking_camera_choice_an_empirical_study_on_fisheye_camera_properties_in_robo.md)**

:   首次系统性地对腕部鱼眼相机在机器人操作模仿学习中的特性进行实证研究，围绕空间定位、场景泛化和硬件泛化三个核心问题揭示了宽视场角的优势与局限，并提出 Random Scale Augmentation (RSA) 策略解决跨相机迁移中的尺度过拟合问题。

**[RoboAgent: Chaining Basic Capabilities for Embodied Task Planning](roboagent_chaining_basic_capabilities_for_embodied_task_planning.md)**

:   提出 RoboAgent，一种能力驱动的具身任务规划框架，用单个 VLM 同时实现调度器和 5 种基本能力（探索引导、物体定位、场景描述、动作解码、经验总结），通过三阶段训练（SFT + DAgger + 专家引导 RL）在 EB-ALFRED 和 ALFWorld 上达到 SOTA。

**[SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_active_perception_manipulation_vla_roboti.md)**

:   SaPaVe提出端到端主动操作框架，通过解耦相机动作和操作动作的底层到顶层训练策略，先用20万对语义相机控制数据学习主动感知先验，再联合优化实现主动操作，在真实世界中超越π₀和GR00T N1达31.25%成功率提升。

**[Semantic Audio-Visual Navigation in Continuous Environments](semantic_audio-visual_navigation_in_continuous_environments.md)**

:   本文提出 SAVN-CE 任务，将语义音视觉导航扩展到连续3D环境中，并设计 MAGNet（记忆增强目标描述网络），通过融合历史上下文和自运动线索实现在目标声音消失后的稳健目标推理，成功率绝对提升最高达 12.1%。

**[STRNet: Visual Navigation with Spatio-Temporal Representation through Dynamic Graph Aggregation](strnet_visual_navigation_with_spatio-temporal_representation_through_dynamic_gra.md)**

:   STRNet 提出统一的时空表征框架用于视觉导航，通过图推理模块建模帧内空间拓扑结构，结合混合时间偏移和多分辨率差分卷积建模时序动态，显著提升了目标条件导航的成功率（比 NoMaD 提升 70%）。

**[Test-time Ego-Exo-centric Adaptation for Action Anticipation via Multi-Label Prototype Growing and Dual-Clue Consistency](test-time_ego-exo-centric_adaptation_for_action_anticipation_via_multi-label_pro.md)**

:   首次提出 Test-time Ego-Exo Adaptation for Action Anticipation（TE2A3）任务，设计 DCPGN 网络通过多标签原型增长和双线索（视觉+文本）一致性，在测试时将源视角训练模型在线适配到目标视角进行动作预测，大幅超越现有 TTA 方法。

**[Towards Open Environments and Instructions: General Vision-Language Navigation via Fast-Slow Interactive Reasoning](towards_open_environments_and_instructions_general_vision-language_navigation_vi.md)**

:   针对开放环境下视觉语言导航（GSA-VLN）任务，受人类快慢认知双系统启发，提出 slow4fast-VLN 框架：快推理模块基于端到端策略网络实时导航并积累历史记忆，慢推理模块借助 LLM 反思生成结构化泛化经验，经验通过注意力融合反馈增强快推理网络，实现在未见环境和多样指令下的持续适应，在 GSA-R2R 数据集上全面超越前 SOTA（GR-DUET）。

**[Towards Training-Free Scene Text Editing](towards_training-free_scene_text_editing.md)**

:   提出TextFlow，一个免训练的场景文字编辑框架，通过在去噪早期阶段使用Flow Manifold Steering（FMS）保持风格一致性、后期阶段使用Attention Boost（AttnBoost）增强文字渲染准确性，在不需要任务特定训练的情况下达到与训练方法可比甚至更优的编辑质量。
