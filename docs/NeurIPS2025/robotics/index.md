---
title: >-
  NeurIPS2025 机器人/具身智能方向57篇论文解读
description: >-
  57篇NeurIPS2025的机器人/具身智能方向论文解读，涵盖机器人、Agent、推理、多模态、LLM、导航等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 机器人/具身智能

**🧠 NeurIPS2025** · **57** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (7)](../../ACL2026/robotics/index.md) · [📷 CVPR2026 (38)](../../CVPR2026/robotics/index.md) · [🔬 ICLR2026 (47)](../../ICLR2026/robotics/index.md) · [🤖 AAAI2026 (38)](../../AAAI2026/robotics/index.md) · [📹 ICCV2025 (26)](../../ICCV2025/robotics/index.md) · [🧪 ICML2025 (16)](../../ICML2025/robotics/index.md)

🔥 **高频主题：** 机器人 ×10 · Agent ×7 · 推理 ×7 · 多模态 ×5 · LLM ×4

**[A Snapshot of Influence: A Local Data Attribution Framework for Online Reinforcement Learning](a_snapshot_of_influence_a_local_data_attribution_framework_f.md)**

:   首次将数据归因（data attribution）引入在线强化学习，提出局部归因框架量化每条训练记录对策略更新的贡献，并基于此设计了迭代影响力过滤算法（IIF），在经典RL基准和LLM的RLHF上均显著提升了样本效率和最终性能。

**[Adaptive Frontier Exploration on Graphs with Applications to Network-Based Disease Testing](adaptive_frontier_exploration_on_graphs_with_applications_to_network-based_disea.md)**

:   提出 Adaptive Frontier Exploration on Graphs (AFEG) 问题框架，设计基于 Gittins index 的策略，在图是森林时可证明最优，在实际性传播疾病检测网络上仅测试一半人口即可检出几乎全部 HIV 感染者，大幅超越贪心和 DQN 等基线。

**[AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling](autotom_scaling_model-based_mental_inference_via_automated_agent_modeling.md)**

:   AutoToM 实现完全自动化的基于模型的心智理论推理——无需人工指定 agent 模型，自动提出贝叶斯网络结构并执行贝叶斯逆规划，通过推理不确定性驱动的迭代模型调整（添加心智变量或扩展时间步），在5个ToM benchmark上以82.43%平均准确率超越GPT-4o(63.39%)、o3-mini(73.94%)等SOTA模型。

**[Beyond Parallelism: Synergistic Computational Graph Effects in Multi-Head Attention](beyond_parallelism_synergistic_computational_graph_effects_in_multi-head_attenti.md)**

:   将多头注意力重新建模为共享汇节点的多个前馈 DAG 系统，理论证明多头可通过跨头路径实现协同效应——降低混合时间(mixing time)并放大 minimax 保真度(fidelity)，在序列操作任务上实验验证了该效应。

**[Breaking the Gradient Barrier: Unveiling Large Language Models for Strategic Classification](breaking_the_gradient_barrier_unveiling_large_language_models_for_strategic_clas.md)**

:   提出 GLIM（Gradient-free Learning In-context Method），首次利用 LLM 的 In-Context Learning 机制隐式模拟策略分类中的双层优化（特征操纵 + 决策规则优化），无需微调即可在大规模数据上高效完成策略分类任务。

**[Bridging Embodiment Gaps: Deploying Vision-Language-Action Models on Soft Robots](bridging_embodiment_gaps_deploying_vision-language-action_models_on_soft_robots.md)**

:   本文首次将 VLA（Vision-Language-Action）模型部署到软体连续体机械臂 Embuddy 上，发现开箱即用的刚性机器人预训练策略因运动学和动力学差异完全失败，但通过在少量软体机器人示范数据上进行针对性微调，可以成功弥合刚性-软体之间的实体鸿沟，使软体平台在抓取和人机交互任务上达到与 UR5 刚性臂相当的任务完成率。

**[C-NAV: Towards Self-Evolving Continual Object Navigation in Open World](c-nav_towards_self-evolving_continual_object_navigation_in_open_world.md)**

:   提出 C-Nav 框架，通过**双路径抗遗忘**（特征蒸馏 + 特征回放）和**自适应经验选择**（LOF 异常检测选关键帧），让导航智能体在不断学习新物体类别时避免灾难性遗忘，在 4 种架构上均超越全量数据回放基线。

**[Can Agents Fix Agent Issues?](can_agents_fix_agent_issues.md)**

:   本文首次系统地研究了 LLM-based Agent 系统的 issue 自动修复问题——通过人工分析 201 个真实 Agent issue 构建了涵盖 6 大类 20 个子类的 Agent issue 分类体系，耗费 500 人时构建了包含 50 个可复现任务的 AgentIssue-Bench 基准，并评估发现当前最先进的软件工程 Agent（如 SWE-agent、Agentless、AutoCodeRover）在 Agent issue 上的正确修复率仅为 3.33%–12.67%，远低于它们在传统软件上的 23%–51% 修复率。

**[CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing & Sparsification](cogvla_cognition-aligned_vision-language-action_model_via_instruction-driven_rou.md)**

:   CogVLA 提出模仿人类多模态认知的三阶段VLA架构（EFA-Routing视觉聚合压缩至25% + LFP-Routing LLM内指令感知剪枝50% + V-L-A耦合注意力），在LIBERO上以97.4%成功率和2.5×训练/2.8×推理加速超越OpenVLA-OFT等SOTA方法，真实机器人任务达70.0%成功率。

**[COOPERA: Continual Open-Ended Human-Robot Assistance](coopera_continual_open_ended_human_robot_assistance.md)**

:   提出 COOPERA 框架，首次实现持续、开放式的人机协作研究，通过LLM驱动具有心理特征和长期意图的模拟人类与机器人在3D环境中多天交互，机器人通过学习人类特征和上下文意图逐步提升个性化协作能力。

**[DexFlyWheel: A Scalable Self-Improving Data Generation Framework for Dexterous Manipulation](dexflywheel_a_scalable_and_self-improving_data_generation_framework_for_dexterou.md)**

:   提出 DexFlyWheel，一个从单个人类示教出发、通过 IL + 残差 RL + 数据增强组成的自改进循环逐步扩展数据多样性的灵巧操作数据生成框架，在 4 个任务上生成 2000+ 示教，策略平均成功率 81.9%，真实世界迁移成功率 78.3%。

**[DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation](dynanav_dynamic_feature_and_layer_selection_for_efficient_visual_navigation.md)**

:   提出 DynaNav，通过可训练的硬特征选择器和基于贝叶斯优化的 early-exit 机制，根据场景复杂度动态调整特征与层的使用，在视觉导航中实现 2.26× FLOPs 降低、42.3% 推理时间减少，同时保持甚至提升导航性能。

**[EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval](efficientnav_towards_on-device_object-goal_navigation_with_navigation_map_cachin.md)**

:   通过离散内存缓存（KV cache分组独立计算+选择性加载）、注意力驱动聚类（LLM浅层attention指导分组）和语义感知检索（CLIP+背包问题适配不同内存预算），首次在Jetson Orin上用LLaMA-3.2-11b实现零样本ObjNav，比GPT-4基线提升11.1% SR且实时延迟降低6.7×。

**[EgoThinker: Unveiling Egocentric Reasoning with Spatio-Temporal CoT](egothinker_unveiling_egocentric_reasoning_with_spatiotempora.md)**

:   EgoThinker 构建了 500 万级第一人称视频 QA 数据集 EgoRe-5M（含因果 CoT 标注和手-物体精细定位数据），并通过"先 SFT 学推理、后 GRPO 练定位"的两阶段训练范式，让 7B MLLM 首次同时具备第一人称因果推理和时空精细定位能力，在 8+ 个基准上刷新 SOTA，7B 参数量在时间定位上甚至超过 72B 模型。

**[Enginuity: Building an Open Multi-Domain Dataset of Complex Engineering Diagrams](enginuity_building_an_open_multi-domain_dataset_of_complex_engineering_diagrams.md)**

:   提出 Enginuity——首个面向 AI 自动解析工程图的大规模开放多领域数据集方案，计划构建 50K+ 带有层级组件关系、空间连接和语义角色标注的汽车工程图，通过四阶段人机协同标注管线实现高质量与低成本的平衡，并定义了从符号检测到数字孪生生成的完整任务体系，为多模态大模型理解工程图中的视觉-结构知识提供了首个系统性基准资源。

**[Explaining and Mitigating Crosslingual Tokenizer Inequities](explaining_and_mitigating_crosslingual_tokenizer_inequities.md)**

:   系统训练约 7000 个单语分词器覆盖 97 种语言，首次证明即使控制训练数据量、词表大小和算法后，不同语言间仍存在显著的 token premium 差异；进一步识别出词表大小和预分词策略是关键因素，并提出"最优词表大小"和 SuperBPE 两种缓解方案。

**[FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model](falcon_fine-grained_activation_manipulation_by_contrastive_orthogonal_unalignmen.md)**

:   提出 FALCON——基于表示引导的 LLM 遗忘框架，利用互信息进行参数选择、对比机制实现精细知识分离、梯度正交投影解决遗忘-保留冲突，在有害知识/版权/实体遗忘任务上全面超越现有方法。

**[Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)**

:   提出基于不平衡最优运输（UOT）的模拟-真实策略联合训练框架，通过对观察-动作联合分布进行对齐（而非仅对齐观察边际分布），结合时间对齐采样策略处理数据不平衡，在机器人操纵任务上实现30%的OOD泛化提升。

**[Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schrödinger Bridges](grasp2grasp_vision-based_dexterous_grasp_translation_via_schrödinger_bridges.md)**

:   提出将跨手形态的视觉灵巧抓取迁移建模为 Schrödinger Bridge 问题，通过在潜空间中学习得分与流匹配（[SF]²M），并设计物理感知的最优传输代价函数（位姿/接触图/力旋量空间/雅可比可操作性），在无需配对数据的条件下实现不同机械手之间抓取意图的分布级迁移。

**[GUI-Rise: Structured Reasoning and History Summarization for GUI Navigation](gui-rise_structured_reasoning_and_history_summarization_for_gui_navigation.md)**

:   提出 GUI-Rise 框架，通过结构化推理（进度估计 + 决策推理）、动作预测和历史摘要三个子任务的联合设计，结合 GRPO 强化学习与历史摘要奖励，显著提升 GUI 导航智能体在跨域场景下的泛化能力。

**[Harnessing the Computation Redundancy in ViTs to Boost Adversarial Transferability](harnessing_the_computation_redundancy_in_vits_to_boost_adversarial_transferabili.md)**

:   深入挖掘 ViT 中数据级和模型级的计算冗余，提出注意力稀疏化、注意力头置换、干净 token 正则化、Ghost MoE 多样化和鲁棒化 token 五种技术，结合在线学习策略动态选择操作，在 ImageNet-1K 上以 86.9% 平均 fooling rate 大幅超越所有基线。

**[HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data](himacon_discovering_hierarchical_manipulation_concepts_from_unlabeled_multi-moda.md)**

:   提出自监督框架从无标注多模态机器人演示中学习层级操作概念，通过跨模态相关性网络和多时域子目标预测器组织表示，增强模仿学习策略在新物体、新障碍和新环境下的泛化能力。

**[Knolling Bot: Teaching Robots the Human Notion of Tidiness](knolling_bot_teaching_robots_the_human_notion_of_tidiness.md)**

:   将桌面物体整理（knolling）类比为 NLP 序列预测任务，用 Transformer 自回归生成每个物体的目标位置，结合 GMM 处理多解歧义，从 240 万组自动生成的示范中学习通用整洁概念，并通过输入排列顺序隐式编码用户偏好。

**[LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents](labutopia_high-fidelity_simulation_and_hierarchical_benchmark_for_scientific_emb.md)**

:   提出 LabUtopia——面向科学实验室的高保真仿真与层级基准套件，包含支持化学反应建模的 LabSim 仿真器、可程序化生成实验室场景的 LabScene、以及从原子操作到长程移动操纵的五级 LabBench 基准，揭示现有模仿学习方法在长程实验流程和物体泛化方面的显著瓶颈。

**[LatentGuard: Controllable Latent Steering for Robust Refusal of Attacks and Reliable Response Generation](latentguard_controllable_latent_steering_for_robust_refusal_of_attacks_and_relia.md)**

:   提出 LatentGuard 三阶段框架，通过行为级对齐微调 + 结构化 VAE 监督潜空间 + 潜空间维度操控，实现对 LLM 拒绝行为的可解释、可控制调节，在抵御对抗攻击的同时保持对正常查询的响应能力。

**[Learning Spatial-Aware Manipulation Ordering](learning_spatial-aware_manipulation_ordering.md)**

:   提出 OrderMind 统一框架，通过空间上下文编码器和时序优先级结构化模块直接从 RGB-D 图像学习杂乱场景中物体的操作顺序，利用 VLM 蒸馏生成训练标注，在仿真和真实环境中均显著优于 VLM 基线，且支持实时推理（5.6 FPS，轻量版 21.3 FPS）。

**[LLM World Models Are Mental: Output Layer Evidence of Brittle World Model Use in LLM Mechanical Reasoning](llm_world_models_are_mental_output_layer_evidence_of_brittle_world_model_use_in_.md)**

:   借鉴认知科学的心理模型研究方法，通过滑轮系统的TikZ代码表示测试LLM的力学推理能力，发现LLM能近似估计机械优势并区分功能/非功能系统（Study 1&2），但在精细结构连接推理上完全失败（Study 3），表明LLM的"世界模型"存在但脆弱。

**[LLMscape](llmscape.md)**

:   LLMscape 是一个投影映射沙盘交互装置，让多个独立 LLM 代理在共享的可变物理环境中接收多模态输入、相互对话和推测，探索人类与 AI 在认知不确定性下的共同意义构建过程。

**[Manipulating Feature Visualizations with Gradient Slingshots](manipulating_feature_visualizations_with_gradient_slingshots.md)**

:   提出 Gradient Slingshots（梯度弹弓）方法，通过在模型的分布外输入区域"雕刻"出导向任意目标图像的二次激活景观，使特征可视化（Feature Visualization）的梯度优化过程收敛到预设的虚假图像，同时保持模型架构、分类精度和内部特征表示基本不变，暴露了 FV 作为模型审计工具的严重脆弱性。

**[MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning](mesatask_towards_task-driven_tabletop_scene_generation_via_3d_spatial_reasoning.md)**

:   提出 MesaTask 框架，通过 Spatial Reasoning Chain 将任务描述分解为对象推理→空间关系推理→场景图构建→3D 布局，结合 10K+ 人工标注数据集和 DPO 优化，生成物理合理且任务对齐的桌面操控场景。

**[MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Cultural Learning](mindforge_empowering_embodied_agents_with_theory_of_mind_for_lifelong_cultural_l.md)**

:   MindForge 为 LLM 驱动的具身智能体引入显式的心智理论（ToM）表征、自然语言通信和多组件记忆系统，使开源 LLM 智能体通过与专家协作对话（无需梯度更新）大幅提升任务完成率，在 Minecraft 中比 Voyager 多获得 3× 科技树里程碑和 2.3× 独特物品。

**[MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents](mineanybuild_benchmarking_spatial_planning_for_openworld_ai.md)**

:   基于 Minecraft 构建空间规划基准 MineAnyBuild，要求 AI Agent 根据多模态指令生成可执行的建筑蓝图矩阵，包含 4000 个任务和 500+ 建筑/装饰资产，从空间理解、空间推理、创造力和空间常识四个维度系统评估 MLLM 的空间规划能力，揭示即便 GPT-4o 整体得分仅 41.02/100，开源模型更差。

**[MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents](mip_against_agent_malicious_image_patches_hijacking_multimod.md)**

:   揭示针对多模态OS Agent的新型对抗攻击MIP(Malicious Image Patches)：在屏幕截图中嵌入人眼不可察觉的对抗性扰动图像块(约占屏幕1/7面积)，当OS Agent截屏捕获后会输出预定义的恶意API调用序列；通过联合优化实现跨用户指令和屏幕布局的Universal泛化，攻击成功率高达100%。

**[MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark](mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)**

:   构建了一个包含 28,136 道问题、覆盖 25 种真实表格任务的大规模基准 MMTU，系统评估 LLM 在专业级表格理解、推理和操作方面的能力，发现即使是 GPT-5 等前沿推理模型也仅得分约 69.6%。

**[mmWalk: Towards Multi-modal Multi-view Walking Assistance](mmwalk_towards_multi-modal_multi-view_walking_assistance.md)**

:   mmWalk 构建了首个面向视障人群步行辅助的多模态多视角数据集（CARLA 仿真器生成 62K 帧/559K 全景图 + 69K VQA 对），基准测试发现 SOTA VLM 在风险评估和导航地标识别等安全关键任务上表现不足（最优仅 55.21%），微调后在真实数据集上泛化提升 16.7%。

**[NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning](nesypr_neurosymbolic_proceduralization_for_efficient_embodied_reasoning.md)**

:   NeSyPr提出了一种神经符号程序化框架，通过将符号规划器生成的任务计划转化为可组合的程序化表示，使紧凑的语言模型在无需外部符号引导的情况下实现高效的单步推理，类似人类的知识编译过程。

**[Operation Veja: Fixing Fundamental Concepts Missing from Modern Roleplaying Training Paradigms](operation_veja_fixing_fundamental_concepts_missing_from_modern_roleplaying_train.md)**

:   本文系统批判了现有角色扮演模型训练的四大范式（RAG、事实值设定、文学数据、合成数据）为何都无法产生有深度的角色，提出VEJA框架（Values-Experiences-Judgments-Abilities）作为角色定义和数据策化的结构化基础，在LLM评判A/B测试中VEJA指导的人工策化数据以43:28:29（胜:负:平）显著优于Gemini Pro 2.5生成的合成基线。

**[Policy Compatible Skill Incremental Learning via Lazy Learning Interface](policy_compatible_skill_incremental_learning_via_lazy_learning_interface.md)**

:   提出SIL-C框架，通过双向惰性学习接口(bilateral lazy learning interface)实现技能增量学习中的技能-策略兼容性，使增量更新的技能能直接提升下游策略性能而无需重训练或结构调整。

**[Predicting the Performance of Black-Box LLMs through Follow-Up Queries](predicting_the_performance_of_black-box_llms_through_follow-up_queries.md)**

:   提出 QueRE 方法，通过向黑盒LLM提出约50个后续问题（如"你对回答有信心吗？"），以"Yes"token的概率作为特征训练线性分类器，在预测模型正确性、检测对抗操纵和区分不同LLM等任务上，甚至超越需要访问模型内部状态的白盒方法。

**[UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Task Planning](pretraining_a_unified_pddl_domain_from_real-world_demonstrations_for_generalizab.md)**

:   UniDomain 从 12,393 个真实机器人操作视频中预训练统一的 PDDL 规划域（含 3,137 个算子和 2,875 个谓词），通过层级融合构建元域，实现零样本跨任务符号规划，比最强基线高出 58% 成功率和 160% 计划最优性。

**[RDD: Retrieval-Based Demonstration Decomposer for Planner Alignment in Long-Horizon Tasks](rdd_retrieval-based_demonstration_decomposer_for_planner_alignment_in_long-horiz.md)**

:   提出RDD（基于检索的演示分解器），通过将演示分解建模为最优分区问题，自动将长时域任务演示分解为与底层视觉运动策略训练数据对齐的子任务，从而协调层级VLA框架中高层规划器与低层策略，在RLBench上接近专家分解器的性能。

**[Redefining Experts: Interpretable Decomposition of Language Models for Toxicity Mitigation](redefining_experts_interpretable_decomposition_of_language_models_for_toxicity_m.md)**

:   提出EigenShift方法，通过对LLM最终输出层进行SVD分解，识别与毒性生成相关的特征方向（eigen-choices），并通过选择性衰减对应奇异值来实现毒性抑制——在LLaMA-2上降低58%毒性的同时仅增加3.62的困惑度，兼顾安全与流畅性。

**[Rethinking the Simulation vs. Rendering Dichotomy: No Free Lunch in Spatial World Modelling](rethinking_the_simulation_vs_rendering_dichotomy_no_free_lunch_in_spatial_world_.md)**

:   从认知神经科学视角挑战"模拟与渲染可分离"的传统观点：论证空间推理依赖于精细的感知表征而非粗粒度抽象，并指出AI空间世界模型同样需要保留丰富的感知细节——空间建模没有免费午餐。

**[RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)**

:   提出RoboCerebra长程机器人操作基准，包含1000条人类示范轨迹（平均2972步，约为现有基准的6倍），通过分层规划与执行框架和多维评估协议，系统测评VLM在规划、反思和记忆三个System 2认知维度上的能力。

**[SAFE: Multitask Failure Detection for Vision-Language-Action Models](safe_multitask_failure_detection_for_vision-language-action_models.md)**

:   SAFE 发现 VLA 模型的内部特征空间存在跨任务一致的"失败区域"，据此训练轻量 MLP/LSTM 失败检测器，配合功能保形预测（FCP）做阈值校准，在未见任务上达 78% ROC-AUC，计算开销 <1%，大幅优于 token 不确定性和一致性检测方法。

**[SegMASt3R: Geometry Grounded Segment Matching](segmast3r_geometry_grounded_segment_matching.md)**

:   SegMASt3R 在预训练 MASt3R 3D 基础模型上添加轻量分割特征头和可微 Sinkhorn 匹配层，利用 3D 几何先验实现极端视角变化（达 180°）下的鲁棒语义段匹配，AUPRC 在 135-180° 基线上达 83.6%（vs SAM2 的 17%）。

**[Spatial Understanding from Videos: Structured Prompts Meet Simulation Data](spatial_understanding_from_videos_structured_prompts_meet_simulation_data.md)**

:   提出 SpatialMind 结构化提示策略与 ScanForgeQA 合成QA数据集的双管齐下方案，在不修改VLM架构的前提下显著增强其从扫描视频进行3D空间推理的能力。

**[SutureBot: A Precision Framework & Benchmark for Autonomous End-to-End Suturing](suturebot_a_precision_framework_benchmark_for_autonomous_end-to-end_suturing.md)**

:   提出SutureBot——首个针对da Vinci手术机器人端到端自主缝合的精度导向基准与目标条件框架，发布1890条高保真演示数据集，通过点标签目标条件将针刺精度提升59%-74%，并系统评估了π0、GR00T N1、OpenVLA-OFT和多任务ACT等SOTA VLA模型。

**[Talk2Event: Grounded Understanding of Dynamic Scenes from Event Cameras](talk2event_grounded_understanding_of_dynamic_scenes_from_event_cameras.md)**

:   Talk2Event 提出首个大规模事件相机视觉定位基准（30,690 条标注表达式 + 四种定位属性），并设计 EventRefer 框架通过混合事件-属性专家（MoEE）动态融合外观/状态/观察者关系/物体间关系特征，在纯事件、纯帧和融合三种设置下均超越现有方法。

**[Task-Optimized Convolutional Recurrent Networks Align with Tactile Processing in the Rodent Brain](task-optimized_convolutional_recurrent_networks_align_with_tactile_processing_in.md)**

:   提出Encoder-Attender-Decoder（EAD）框架系统探索触觉任务优化的时序神经网络，发现卷积循环网络（ConvRNN，特别是IntersectionRNN）在触觉物体分类和啮齿类体感皮层神经对齐上均优于前馈和状态空间模型，且基于触觉特定增强的对比自监督学习能达到与监督学习相当的神经拟合，为触觉的大脑计算机制提供了首个定量刻画。

**[ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning](thinkact_vision-language-action_reasoning_via_reinforced_visual_latent_planning.md)**

:   提出ThinkAct双系统框架，通过动作对齐的视觉奖励对MLLM进行强化学习微调以激发具身推理能力，并将推理计划压缩为视觉潜在表示来指导下游动作模型，实现"先思考再行动"的VLA推理范式。

**[Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs](toward_engineering_agi_benchmarking_the_engineering_design_capabilities_of_llms.md)**

:   提出 EngDesign——首个跨 9 个工程领域（操作系统、计算机架构、控制系统、机械、结构、数字硬件、模拟电路、机器人、信号处理）的 LLM 工程设计能力基准，用仿真驱动的评估管线替代传统的问答匹配，揭示即使最强推理模型 o3 也仅达 34% 通过率。

**[Towards Reliable Code-as-Policies: A Neuro-Symbolic Framework for Embodied Task Planning](towards_reliable_code-as-policies_a_neuro-symbolic_framework_for_embodied_task_p.md)**

:   提出一种神经-符号具身任务规划框架，在 LLM 代码生成过程中引入显式的符号验证（检查前置条件是否满足）和交互式验证（主动探索获取缺失信息），使生成的代码在动态和部分可观测场景中更可靠——在 RLBench 上任务成功率从基线 38.5% 提升到 84.7%，可执行性达 86.8%。

**[Uncovering Strategic Egoism Behaviors in Large Language Models](uncovering_strategic_egoism_behaviors_in_large_language_models.md)**

:   首次形式化定义LLM中的"策略性自利"（Strategic Egoism）行为并构建SEBench基准（160个场景×6类自利维度），实验发现7个主流LLM在激励诱惑下平均69.11%的决策选择自利策略，操纵胁迫与规则规避是最常见手段，且自利倾向与毒性语言生成呈正相关。

**[Understanding Prompt Tuning and In-Context Learning via Meta-Learning](understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)**

:   从贝叶斯元学习视角系统分析了提示调优（prompt tuning）的理论基础与局限性，证明了软提示可以在预训练分布内的单一目标任务上实现最优适配，但对多任务混合目标分布存在根本性限制，且软前缀能通过操纵非token空间的激活来超越最优硬token序列。

**[VITRIX-CLIPIN: Enhancing Fine-Grained Visual Understanding in CLIP via Instruction Editing Data and Long Captions](vitrix-clipin_enhancing_fine-grained_visual_understanding_in_clip_via_instructio.md)**

:   提出 CLIP-IN 框架，利用指令编辑数据集作为硬负样本和长描述增强 CLIP 的细粒度视觉理解能力，在 MMVP 等基准上显著提升且不损害零样本性能，集成到 MLLM 中可减少视觉幻觉。

**[Zero-Shot Embedding Drift Detection: A Lightweight Defense Against Prompt Injections in LLMs](zero-shot_embedding_drift_detection_a_lightweight_defense_against_prompt_injecti.md)**

:   提出ZEDD（零样本嵌入漂移检测），通过比较良性和可疑输入在嵌入空间中的语义漂移来检测提示注入攻击，利用GMM/KDE自动确定阈值，在多种LLM架构上实现>93%的检测准确率且假阳性率<3%。
