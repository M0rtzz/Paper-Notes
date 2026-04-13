---
title: >-
  NeurIPS2025 机器人/具身智能方向 55篇论文解读
description: >-
  55篇NeurIPS2025 机器人/具身智能方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 机器人/具身智能

**🧠 NeurIPS2025** · 共 **55** 篇

**[A Snapshot Of Influence A Local Data Attribution Framework F](a_snapshot_of_influence_a_local_data_attribution_framework_f.md)**

:   首次将数据归因（data attribution）引入在线强化学习，提出局部归因框架量化每条训练记录对策略更新的贡献，并基于此设计了迭代影响力过滤算法（IIF），在经典RL基准和LLM的RLHF上均显著提升了样本效率和最终性能。

**[Adaptive Frontier Exploration On Graphs With Applications To Network-Based Disea](adaptive_frontier_exploration_on_graphs_with_applications_to_network-based_disea.md)**

:   提出 Adaptive Frontier Exploration on Graphs (AFEG) 问题框架，设计基于 Gittins index 的策略，在图是森林时可证明最优，在实际性传播疾病检测网络上仅测试一半人口即可检出几乎全部 HIV 感染者，大幅超越贪心和 DQN 等基线。

**[Autotom Scaling Model-Based Mental Inference Via Automated Agent Modeling](autotom_scaling_model-based_mental_inference_via_automated_agent_modeling.md)**

:   AutoToM 实现完全自动化的基于模型的心智理论推理——自动提出 agent 模型（贝叶斯网络结构）并进行贝叶斯逆规划，通过推理不确定性迭代调整模型（添加心智变量/扩展时间步），在5个 ToM benchmark 上超越 SOTA LLM 和推理模型，且产生类人的置信度估计。

**[Beyond Parallelism Synergistic Computational Graph Effects In Multi-Head Attenti](beyond_parallelism_synergistic_computational_graph_effects_in_multi-head_attenti.md)**

:   将多头注意力重新建模为共享汇节点的多个前馈 DAG 系统，理论证明多头可通过跨头路径实现协同效应——降低混合时间(mixing time)并放大 minimax 保真度(fidelity)，在序列操作任务上实验验证了该效应。

**[Bridging Embodiment Gaps Deploying Vision-Language-Action Models On Soft Robots](bridging_embodiment_gaps_deploying_vision-language-action_models_on_soft_robots.md)**

:   首次在柔性连续体机械臂上部署 VLA 模型（OpenVLA-OFT 和 π₀），发现开箱即用的策略因构型不匹配完全失败，但通过针对性微调可弥合刚性-柔性的 embodiment gap，使柔性机器人在操作任务上达到与刚性 UR5 相当的成功率——证明 VLA + 柔性机器人可实现安全的人机交互。

**[C-Nav Towards Self-Evolving Continual Object Navigation In Open World](c-nav_towards_self-evolving_continual_object_navigation_in_open_world.md)**

:   提出 C-Nav 框架，通过**双路径抗遗忘**（特征蒸馏 + 特征回放）和**自适应经验选择**（LOF 异常检测选关键帧），让导航智能体在不断学习新物体类别时避免灾难性遗忘，在 4 种架构上均超越全量数据回放基线。

**[Can Agents Fix Agent Issues](can_agents_fix_agent_issues.md)**

:   AgentIssue-Bench(50个bug任务)评估SE代理解决LLM代理bug的能力，仅0.67%-4.67%解决率。

**[Cogvla Cognition-Aligned Vision-Language-Action Model Via Instruction-Driven Rou](cogvla_cognition-aligned_vision-language-action_model_via_instruction-driven_rou.md)**

:   提出 CogVLA——模仿人类多模态认知的三阶段 VLA 架构：(1) EFA-Routing 将视觉 token 压缩至 25%；(2) LFP-Routing 裁剪 50% 的 LLM 无关 token；(3) V-L-A 耦合注意力保持语义一致性——在 LIBERO 上达 97.4% 成功率，训练成本降 2.5×，推理延迟降 2.8×。

**[Coopera Continual Open-Ended Human-Robot Assistance](coopera_continual_open-ended_human-robot_assistance.md)**

:   提出 C-Nav 持续目标导航框架，通过**双路径抗遗忘机制**（特征蒸馏 + 特征回放）和**基于 LOF 的自适应经验选择**，使导航智能体在增量学习新物体类别时有效避免灾难性遗忘，在 4 种主流架构和 2 个数据集上均超越全量数据回放基线。

**[Dexflywheel A Scalable And Self-Improving Data Generation Framework For Dexterou](dexflywheel_a_scalable_and_self-improving_data_generation_framework_for_dexterou.md)**

:   提出 DexFlyWheel，一个从单个人类示教出发、通过 IL + 残差 RL + 数据增强组成的自改进循环逐步扩展数据多样性的灵巧操作数据生成框架，在 4 个任务上生成 2000+ 示教，策略平均成功率 81.9%，真实世界迁移成功率 78.3%。

**[Dynanav Dynamic Feature And Layer Selection For Efficient Visual Navigation](dynanav_dynamic_feature_and_layer_selection_for_efficient_visual_navigation.md)**

:   提出 DynaNav，通过可训练的硬特征选择器和基于贝叶斯优化的 early-exit 机制，根据场景复杂度动态调整特征与层的使用，在视觉导航中实现 2.26× FLOPs 降低、42.3% 推理时间减少，同时保持甚至提升导航性能。

**[Efficientnav Towards On-Device Object-Goal Navigation With Navigation Map Cachin](efficientnav_towards_on-device_object-goal_navigation_with_navigation_map_cachin.md)**

:   通过离散内存缓存（KV cache分组独立计算+选择性加载）、注意力驱动聚类（LLM浅层attention指导分组）和语义感知检索（CLIP+背包问题适配不同内存预算），首次在Jetson Orin上用LLaMA-3.2-11b实现零样本ObjNav，比GPT-4基线提升11.1% SR且实时延迟降低6.7×。

**[Egothinker Unveiling Egocentric Reasoning With Spatiotempora](egothinker_unveiling_egocentric_reasoning_with_spatiotempora.md)**

:   针对第一人称视频推理中“主体不可见、意图隐含、交互细粒度”的挑战，EgoThinker 提出时空 CoT 监督与两阶段训练（SFT + RFT），并构建 EgoRe-5M 大规模 egocentric QA 数据，显著提升 MLLM 在自我中心视频推理与时空定位任务上的表现。

**[Enginuity Building An Open Multi-Domain Dataset Of Complex Engineering Diagrams](enginuity_building_an_open_multi-domain_dataset_of_complex_engineering_diagrams.md)**

:   提出 Enginuity——首个大规模开放多领域工程图数据集（50K+ 标注图），涵盖层级组件关系与连接语义，旨在突破当前 AI 无法理解工程图中视觉-结构知识的瓶颈。

**[Explaining And Mitigating Crosslingual Tokenizer Inequities](explaining_and_mitigating_crosslingual_tokenizer_inequities.md)**

:   系统训练约 7000 个单语分词器覆盖 97 种语言，首次证明即使控制训练数据量、词表大小和算法后，不同语言间仍存在显著的 token premium 差异；进一步识别出词表大小和预分词策略是关键因素，并提出"最优词表大小"和 SuperBPE 两种缓解方案。

**[Falcon Fine-Grained Activation Manipulation By Contrastive Orthogonal Unalignmen](falcon_fine-grained_activation_manipulation_by_contrastive_orthogonal_unalignmen.md)**

:   提出 FALCON——基于表示引导的 LLM 遗忘框架，利用互信息进行参数选择、对比机制实现精细知识分离、梯度正交投影解决遗忘-保留冲突，在有害知识/版权/实体遗忘任务上全面超越现有方法。

**[Generalizable Domain Adaptation For Sim-And-Real Policy Co-Training](generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)**

:   提出基于不平衡最优运输（UOT）的模拟-真实策略联合训练框架，通过对观察-动作联合分布进行对齐（而非仅对齐观察边际分布），结合时间对齐采样策略处理数据不平衡，在机器人操纵任务上实现30%的OOD泛化提升。

**[Harnessing The Computation Redundancy In Vits To Boost Adversarial Transferabili](harnessing_the_computation_redundancy_in_vits_to_boost_adversarial_transferabili.md)**

:   深入挖掘 ViT 中数据级和模型级的计算冗余，提出注意力稀疏化、注意力头置换、干净 token 正则化、Ghost MoE 多样化和鲁棒化 token 五种技术，结合在线学习策略动态选择操作，在 ImageNet-1K 上以 86.9% 平均 fooling rate 大幅超越所有基线。

**[Himacon Discovering Hierarchical Manipulation Concepts From Unlabeled Multi-Moda](himacon_discovering_hierarchical_manipulation_concepts_from_unlabeled_multi-moda.md)**

:   提出自监督框架从无标注多模态机器人演示中学习层级操作概念，通过跨模态相关性网络和多时域子目标预测器组织表示，增强模仿学习策略在新物体、新障碍和新环境下的泛化能力。

**[Knolling Bot Teaching Robots The Human Notion Of Tidiness](knolling_bot_teaching_robots_the_human_notion_of_tidiness.md)**

:   提出基于 Transformer + GMM 的自监督学习框架，让机器人从 240 万组整理示范中学习"整洁"的抽象概念，以自回归方式预测物体目标位置，实现桌面物体的美观且紧凑的自动整理（knolling），并支持基于用户偏好（颜色/类别/大小）生成多样化整理方案。

**[Labutopia High-Fidelity Simulation And Hierarchical Benchmark For Scientific Emb](labutopia_high-fidelity_simulation_and_hierarchical_benchmark_for_scientific_emb.md)**

:   提出 LabUtopia——面向科学实验室的高保真仿真与层级基准套件，包含支持化学反应建模的 LabSim 仿真器、可程序化生成实验室场景的 LabScene、以及从原子操作到长程移动操纵的五级 LabBench 基准，揭示现有模仿学习方法在长程实验流程和物体泛化方面的显著瓶颈。

**[Latentguard Controllable Latent Steering For Robust Refusal Of Attacks And Relia](latentguard_controllable_latent_steering_for_robust_refusal_of_attacks_and_relia.md)**

:   提出 LatentGuard 三阶段框架，通过行为级对齐微调 + 结构化 VAE 监督潜空间 + 潜空间维度操控，实现对 LLM 拒绝行为的可解释、可控制调节，在抵御对抗攻击的同时保持对正常查询的响应能力。

**[Learning Spatial-Aware Manipulation Ordering](learning_spatial-aware_manipulation_ordering.md)**

:   提出 OrderMind 统一框架，通过空间上下文编码器和时序优先级结构化模块直接从 RGB-D 图像学习杂乱场景中物体的操作顺序，利用 VLM 蒸馏生成训练标注，在仿真和真实环境中均显著优于 VLM 基线，且支持实时推理（5.6 FPS，轻量版 21.3 FPS）。

**[Llm World Models Are Mental Output Layer Evidence Of Brittle World Model Use In ](llm_world_models_are_mental_output_layer_evidence_of_brittle_world_model_use_in_.md)**

:   借鉴认知科学的心理模型研究方法，通过滑轮系统的TikZ代码表示测试LLM的力学推理能力，发现LLM能近似估计机械优势并区分功能/非功能系统（Study 1&2），但在精细结构连接推理上完全失败（Study 3），表明LLM的"世界模型"存在但脆弱。

**[Llmscape](llmscape.md)**

:   LLMscape 是一个交互式投影映射沙盘装置，让人类参与者和多个 LLM 驱动的 AI 代理在共同的不确定环境中共同构建意义，探索人类与机器在认识论层面的共通局限。

**[Mango - Adaptable Graph Network Simulators Via Meta-Learning](mango_-_adaptable_graph_network_simulators_via_meta-learning.md)**

:   提出 MaNGO（Meta Neural Graph Operator），通过元学习和条件神经过程（CNP）学习不同物理参数下仿真任务的共享潜在结构，实现对新物理参数的快速适应，无需重新训练。

**[Manipulating Feature Visualizations With Gradient Slingshots](manipulating_feature_visualizations_with_gradient_slingshots.md)**

:   提出梯度弹弓攻击，通过利用分布外梯度轨迹操纵神经网络特征可视化结果，无需修改模型参数，揭示特征可视化作为解释性工具的脆弱性。

**[Mesatask Towards Task-Driven Tabletop Scene Generation Via 3D Spatial Reasoning](mesatask_towards_task-driven_tabletop_scene_generation_via_3d_spatial_reasoning.md)**

:   提出 MesaTask 框架，通过 Spatial Reasoning Chain 将任务描述分解为对象推理→空间关系推理→场景图构建→3D 布局，结合 10K+ 人工标注数据集和 DPO 优化，生成物理合理且任务对齐的桌面操控场景。

**[Mindforge Empowering Embodied Agents With Theory Of Mind For Lifelong Cultural L](mindforge_empowering_embodied_agents_with_theory_of_mind_for_lifelong_cultural_l.md)**

:   MindForge 为 LLM 驱动的具身智能体引入显式的心智理论（ToM）表征、自然语言通信和多组件记忆系统，使开源 LLM 智能体通过与专家协作对话（无需梯度更新）大幅提升任务完成率，在 Minecraft 中比 Voyager 多获得 3× 科技树里程碑和 2.3× 独特物品。

**[Mineanybuild Benchmarking Spatial Planning For Openworld Ai](mineanybuild_benchmarking_spatial_planning_for_openworld_ai.md)**

:   基于 Minecraft 构建空间规划基准 MineAnyBuild，要求 AI Agent 根据多模态指令生成可执行的建筑蓝图矩阵，包含 4000 个任务和 500+ 建筑/装饰资产，从空间理解、空间推理、创造力和空间常识四个维度系统评估 MLLM 的空间规划能力，揭示即便 GPT-4o 整体得分仅 41.02/100，开源模型更差。

**[Mip Against Agent Malicious Image Patches Hijacking Multimod](mip_against_agent_malicious_image_patches_hijacking_multimod.md)**

:   揭示针对多模态OS Agent的新型攻击向量——Malicious Image Patches (MIPs)：在屏幕截图中嵌入人类不可察觉的对抗性扰动图像块，当OS Agent截屏时自动触发恶意行为（如数据泄露、内存溢出），且可跨用户指令、屏幕布局和屏幕解析器泛化，甚至具备"计算机蠕虫"般的自传播潜力。

**[Mmtu A Massive Multi-Task Table Understanding And Reasoning Benchmark](mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)**

:   构建了一个包含 28,136 道问题、覆盖 25 种真实表格任务的大规模基准 MMTU，系统评估 LLM 在专业级表格理解、推理和操作方面的能力，发现即使是 GPT-5 等前沿推理模型也仅得分约 69.6%。

**[Mmwalk Towards Multi-Modal Multi-View Walking Assistance](mmwalk_towards_multi-modal_multi-view_walking_assistance.md)**

:   mmWalk 构建了首个面向视障人群步行辅助的多模态多视角数据集（CARLA 仿真器生成 62K 帧/559K 全景图 + 69K VQA 对），基准测试发现 SOTA VLM 在风险评估和导航地标识别等安全关键任务上表现不足（最优仅 55.21%），微调后在真实数据集上泛化提升 16.7%。

**[Nesypr Neurosymbolic Proceduralization For Efficient Embodied Reasoning](nesypr_neurosymbolic_proceduralization_for_efficient_embodied_reasoning.md)**

:   NeSyPr 提出了一种神经符号过程化框架，将符号规划器生成的任务计划转化为可组合的过程化表示，嵌入到语言模型的推理过程中，从而在资源受限的环境中实现高效的具身推理。

**[Operation Veja Fixing Fundamental Concepts Missing From Modern Roleplaying Train](operation_veja_fixing_fundamental_concepts_missing_from_modern_roleplaying_train.md)**

:   提出 VEJA（Values-Experiences-Judgments-Abilities）框架，通过结构化角色定义指导人工数据策化，解决现有角色扮演模型缺乏内在价值冲突推理和角色深度的系统性问题。

**[Physics Of Language Models Part 41 Architecture Design And The Magic Of Canon La](physics_of_language_models_part_41_architecture_design_and_the_magic_of_canon_la.md)**

:   通过受控合成预训练任务系统性比较语言模型架构，发现 Canon 层——一种轻量级的邻近token加权求和组件——能显著提升推理深度（2-4倍）、推理广度、知识容量等核心能力，让 NoPE 匹配 RoPE，让 GLA 匹敌 Mamba2/GDN。

**[Predicting The Performance Of Black-Box Llms Through Follow-Up Queries](predicting_the_performance_of_black-box_llms_through_follow-up_queries.md)**

:   提出 QueRE 方法，通过向黑盒LLM提出约50个后续问题（如"你对回答有信心吗？"），以"Yes"token的概率作为特征训练线性分类器，在预测模型正确性、检测对抗操纵和区分不同LLM等任务上，甚至超越需要访问模型内部状态的白盒方法。

**[Pretraining A Unified Pddl Domain From Real-World Demonstrations For Generalizab](pretraining_a_unified_pddl_domain_from_real-world_demonstrations_for_generalizab.md)**

:   UniDomain 从 12,393 个真实机器人操作视频中预训练统一的 PDDL 规划域（含 3,137 个算子和 2,875 个谓词），通过层级融合构建元域，实现零样本跨任务符号规划，比最强基线高出 58% 成功率和 160% 计划最优性。

**[Rdd Retrieval-Based Demonstration Decomposer For Planner Alignment In Long-Horiz](rdd_retrieval-based_demonstration_decomposer_for_planner_alignment_in_long-horiz.md)**

:   提出RDD（基于检索的演示分解器），通过将演示分解建模为最优分区问题，自动将长时域任务演示分解为与底层视觉运动策略训练数据对齐的子任务，从而协调层级VLA框架中高层规划器与低层策略，在RLBench上接近专家分解器的性能。

**[Redefining Experts Interpretable Decomposition Of Language Models For Toxicity M](redefining_experts_interpretable_decomposition_of_language_models_for_toxicity_m.md)**

:   提出EigenShift方法，通过对LLM最终输出层进行SVD分解，识别与毒性生成相关的特征方向（eigen-choices），并通过选择性衰减对应奇异值来实现毒性抑制——在LLaMA-2上降低58%毒性的同时仅增加3.62的困惑度，兼顾安全与流畅性。

**[Rethinking The Simulation Vs Rendering Dichotomy No Free Lunch In Spatial World ](rethinking_the_simulation_vs_rendering_dichotomy_no_free_lunch_in_spatial_world_.md)**

:   从认知神经科学视角挑战"模拟与渲染可分离"的传统观点：论证空间推理依赖于精细的感知表征而非粗粒度抽象，并指出AI空间世界模型同样需要保留丰富的感知细节——空间建模没有免费午餐。

**[Robocerebra A Large-Scale Benchmark For Long-Horizon Robotic Manipulation Evalua](robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)**

:   提出RoboCerebra长程机器人操作基准，包含1000条人类示范轨迹（平均2972步，约为现有基准的6倍），通过分层规划与执行框架和多维评估协议，系统测评VLM在规划、反思和记忆三个System 2认知维度上的能力。

**[Segmast3R Geometry Grounded Segment Matching](segmast3r_geometry_grounded_segment_matching.md)**

:   SegMASt3R 在预训练 MASt3R 3D 基础模型上添加轻量分割特征头和可微 Sinkhorn 匹配层，利用 3D 几何先验实现极端视角变化（达 180°）下的鲁棒语义段匹配，AUPRC 在 135-180° 基线上达 83.6%（vs SAM2 的 17%）。

**[Sitcom Scaling Inference-Time Compute For Vlas](sitcom_scaling_inference-time_compute_for_vlas.md)**

:   SITCOM 提出了一种受模型预测控制（MPC）启发的推理时计算框架，通过学习的动力学模型对预训练 VLA 进行多步rollout仿真并利用奖励模型选择最优轨迹，将单步 VLA 转化为鲁棒的长程规划器，在 SIMPLER 环境中将任务完成率从 48% 提升至 72%。

**[Suturebot A Precision Framework Benchmark For Autonomous End-To-End Suturing](suturebot_a_precision_framework_benchmark_for_autonomous_end-to-end_suturing.md)**

:   提出SutureBot——首个针对da Vinci手术机器人端到端自主缝合的精度导向基准与目标条件框架，发布1890条高保真演示数据集，通过点标签目标条件将针刺精度提升59%-74%，并系统评估了π0、GR00T N1、OpenVLA-OFT和多任务ACT等SOTA VLA模型。

**[T-Rex Task-Adaptive Spatial Representation Extraction For Robotic Manipulation W](t-rex_task-adaptive_spatial_representation_extraction_for_robotic_manipulation_w.md)**

:   提出T-Rex框架，根据任务复杂度动态选择最优的空间表示提取方案（点/向量/6D位姿），并设计Chain of Grounding (CoG)引导VLM逐步推理，实现无需训练的开放词汇机器人操纵。

**[Talk2Event Grounded Understanding Of Dynamic Scenes From Event Cameras](talk2event_grounded_understanding_of_dynamic_scenes_from_event_cameras.md)**

:   Talk2Event 提出首个大规模事件相机视觉定位基准（30,690 条标注表达式 + 四种定位属性），并设计 EventRefer 框架通过混合事件-属性专家（MoEE）动态融合外观/状态/观察者关系/物体间关系特征，在纯事件、纯帧和融合三种设置下均超越现有方法。

**[Task-Optimized Convolutional Recurrent Networks Align With Tactile Processing In](task-optimized_convolutional_recurrent_networks_align_with_tactile_processing_in.md)**

:   提出Encoder-Attender-Decoder（EAD）框架系统探索触觉任务优化的时序神经网络，发现卷积循环网络（ConvRNN，特别是IntersectionRNN）在触觉物体分类和啮齿类体感皮层神经对齐上均优于前馈和状态空间模型，且基于触觉特定增强的对比自监督学习能达到与监督学习相当的神经拟合，为触觉的大脑计算机制提供了首个定量刻画。

**[Thinkact Vision-Language-Action Reasoning Via Reinforced Visual Latent Planning](thinkact_vision-language-action_reasoning_via_reinforced_visual_latent_planning.md)**

:   提出ThinkAct双系统框架，通过动作对齐的视觉奖励对MLLM进行强化学习微调以激发具身推理能力，并将推理计划压缩为视觉潜在表示来指导下游动作模型，实现"先思考再行动"的VLA推理范式。

**[Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs](toward_engineering_agi_benchmarking_the_engineering_design_capabilities_of_llms.md)**

**[Towards Reliable Code-As-Policies A Neuro-Symbolic Framework For Embodied Task P](towards_reliable_code-as-policies_a_neuro-symbolic_framework_for_embodied_task_p.md)**

:   提出一种神经-符号具身任务规划框架，在 LLM 代码生成过程中引入符号验证和交互式验证，使任务成功率比 Code-as-Policies 基线提升 46.2%，可执行性达 86.8%。

**[Understanding Prompt Tuning And In-Context Learning Via Meta-Learning](understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)**

:   从贝叶斯元学习视角系统分析了提示调优（prompt tuning）的理论基础与局限性，证明了软提示可以在预训练分布内的单一目标任务上实现最优适配，但对多任务混合目标分布存在根本性限制，且软前缀能通过操纵非token空间的激活来超越最优硬token序列。

**[Vitrix-Clipin Enhancing Fine-Grained Visual Understanding In Clip Via Instructio](vitrix-clipin_enhancing_fine-grained_visual_understanding_in_clip_via_instructio.md)**

:   提出 CLIP-IN 框架，利用指令编辑数据集作为硬负样本和长描述增强 CLIP 的细粒度视觉理解能力，在 MMVP 等基准上显著提升且不损害零样本性能，集成到 MLLM 中可减少视觉幻觉。

**[Vla-Cache Efficient Vision-Language-Action Manipulation Via Adaptive Token Cachi](vla-cache_efficient_vision-language-action_manipulation_via_adaptive_token_cachi.md)**

:   提出VLA-Cache，一种免训练的VLA推理加速方法，通过跨帧识别并缓存静态视觉token的KV表示、过滤任务相关token并按层自适应调整复用比例，实现1.7倍加速且几乎不损失任务成功率。

**[Zero-Shot Embedding Drift Detection A Lightweight Defense Against Prompt Injecti](zero-shot_embedding_drift_detection_a_lightweight_defense_against_prompt_injecti.md)**

:   提出ZEDD（零样本嵌入漂移检测），通过比较良性和可疑输入在嵌入空间中的语义漂移来检测提示注入攻击，利用GMM/KDE自动确定阈值，在多种LLM架构上实现>93%的检测准确率且假阳性率<3%。
