---
title: >-
  AAAI2026 机器人/具身智能方向38篇论文解读
description: >-
  38篇AAAI2026的机器人/具身智能方向论文解读，涵盖机器人、导航、多模态、Agent、对齐/RLHF、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 机器人/具身智能

**🤖 AAAI2026** · **38** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (7)](../../ACL2026/robotics/index.md) · [📷 CVPR2026 (38)](../../CVPR2026/robotics/index.md) · [🔬 ICLR2026 (47)](../../ICLR2026/robotics/index.md) · [🧠 NeurIPS2025 (57)](../../NeurIPS2025/robotics/index.md) · [📹 ICCV2025 (26)](../../ICCV2025/robotics/index.md) · [🧪 ICML2025 (16)](../../ICML2025/robotics/index.md)

🔥 **高频主题：** 机器人 ×11 · 导航 ×6 · 多模态 ×5 · Agent ×5 · 对齐/RLHF ×5

**[10 Open Challenges Steering the Future of Vision-Language-Action Models](10_open_challenges_steering_the_future_of_vision-language-ac.md)**

:   系统梳理 VLA 模型面临的 10 大开放挑战——多模态感知、鲁棒推理、高质量训练数据、评估、跨机器人动作泛化、资源效率、全身协调、安全保障、Agent 框架、人机协作——并讨论空间理解、世界动力学建模、后训练和数据合成四大新兴趋势。

**[A Computable Game-Theoretic Framework for Multi-Agent Theory of Mind](a_computable_game-theoretic_framework_for_multi-agent_theory_of_mind.md)**

:   提出基于 Poisson 认知层次（cognitive hierarchy）的博弈论框架，通过 Gamma-Poisson 共轭贝叶斯更新实现可计算的多智能体 Theory of Mind，在避免 POMDP 不可判定性的同时支持递归式有限理性决策与在线信念修正。

**[Adaptive Theory of Mind for LLM-based Multi-Agent Coordination](adaptive_theory_of_mind_for_llm-based_multi-agent_coordination.md)**

:   提出自适应心智理论智能体(A-ToM)，将ToM阶数对齐建模为在线专家建议问题，通过FTL或Hedge算法实时估计伙伴的ToM阶数并动态调整自身推理深度，在重复矩阵博弈、网格导航和Overcooked等4类任务上实现鲁棒的零样本多智能体协作。

**[Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation](affordance-guided_coarse-to-fine_exploration_for_base_placem.md)**

:   针对开放词汇移动操控中机器人基座选位问题，提出一种零样本框架，通过构建跨模态表征（Affordance RGB + Obstacle Map+）将语义affordance线索投射到障碍物地图上，再用粗到细迭代优化平衡语义和几何约束，在5个操控任务上达到85%成功率，大幅超越几何规划器和纯VLM方法。

**[Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning](attention_as_binding_a_vector-symbolic_perspective_on_transformer_reasoning.md)**

:   本文提出将Transformer自注意力机制重新解释为向量符号架构(VSA)中的软绑定/解绑定算子——Query/Key定义角色空间、Value编码填充项、注意力权重实现可微解绑定、残差连接实现叠加——从而以代数视角统一解释LLM在符号推理中的能力与脆弱性，并提出显式绑定头、超维记忆层等VSA启发的架构改进方向。

**[Causal Inference Under Threshold Manipulation: Bayesian Mixture Modeling and Heterogeneous Treatment Effects](causal_inference_under_threshold_manipulation_bayesian_mixtu.md)**

:   提出 BMTM/HBMTM 贝叶斯混合模型框架，在消费者策略性操纵消费额以达到奖励阈值的场景下，通过将观测分布拆解为 bunching 与 non-bunching 两个子分布，准确估计阈值因果效应及跨子群的异质性处理效应。

**[Continuous Vision-Language-Action Co-Learning with Semantic-Physical Alignment for Behavioral Cloning](continuous_vision-language-action_co-learning_with_semantic-.md)**

:   提出CCoL框架，通过NeuralODE驱动的多模态连续协同学习（MCC）和双向交叉注意力的语义-物理对齐（CSA），在Behavioral Cloning中同时解决动作序列的物理不连续性和语义-物理失配问题，在三个仿真平台上平均相对提升8.0%，双臂插入任务最高达19.2%。

**[Cross Modal Fine-Grained Alignment via Granularity-Aware and Region-Uncertain Modeling](cross_modal_fine-grained_alignment_via_granularity-aware_and_region-uncertain_mo.md)**

:   提出 GRM 框架，通过模态内显著性/粒度感知适配器和基于高斯混合的区域级不确定性建模，实现鲁棒的细粒度图文对齐，在 Flickr30K 和 MS-COCO 上取得 SOTA。

**[Dexterous Manipulation Transfer via Progressive Kinematic-Dynamic Alignment](dexterous_manipulation_transfer_via_progressive_kinematic-dynamic_alignment.md)**

:   提出 PKDA 框架，通过渐进式运动学-动力学对齐，将人手操作视频自动转化为多指灵巧手的高质量操作轨迹，平均迁移成功率达 73%。

**[Do LLMs Really Struggle at NL-FOL Translation? Revealing Their Strengths via a Novel Benchmarking Strategy](do_llms_really_struggle_at_nl-fol_translation_revealing_their_strengths_via_a_no.md)**

:   本文批判性审视了现有NL到一阶逻辑(FOL)翻译的评估方法（FOLIO和MALLS），揭示其数据集与评估协议的根本缺陷，提出了一种将翻译任务分解为本体提取(OE)和逻辑翻译(LT)、并辅以"最相似选择"和"排序"子任务的新型基准测试策略，实验表明对话式LLM（o3-mini、GPT-4o-mini、Qwen3系列）展现出强大的NL-FOL翻译能力与真正的逻辑语义理解，而嵌入式模型表现显著较差。

**[EvoEmpirBench: Dynamic Spatial Reasoning with Agent-ExpVer](evoempirbench_dynamic_spatial_reasoning_with_agent-expver.md)**

:   提出 EvoEmpirBench（EEB），包含两个动态交互式 benchmark（局部可观测迷宫导航 + 消消乐），以及 Agent-ExpVer 三智能体在线学习框架（GeoLink 交互 + InsightForce 经验抽象 + TruthWeaver 知识管理），通过"经验→验证→真理归纳"的认知循环实现无参数更新的持续策略进化，使 GPT-4.1 成功率提升 5.6%、Qwen-32B 提升 29%。

**[From Passive Perception to Active Memory: A Weakly Supervised Image Manipulation Localization Framework Driven by Coarse-Grained Annotations](from_passive_perception_to_active_memory_a_weakly_supervised_image_manipulation_.md)**

:   提出 BoxPromptIML，一种基于粗粒度框标注的弱监督图像篡改定位（IML）框架，通过冻结的 SAM 教师模型将粗糙边界框转化为高质量伪掩码，结合记忆引导门控融合模块（MGFM）训练轻量级学生模型，仅需 7 秒/张的标注成本即可媲美甚至超越全监督方法。

**[From Woofs to Words: Towards Intelligent Robotic Guide Dogs with Verbal Communication](from_woofs_to_words_towards_intelligent_robotic_guide_dogs_with_verbal_communica.md)**

:   本文提出了一套面向导盲机器犬的对话系统，利用 LLM 和任务规划器实现 **计划语言化（Plan Verbalization）** 和 **场景语言化（Scene Verbalization）**，通过多轮自然语言对话辅助视障用户完成导航决策，并通过真人用户研究和仿真实验验证了系统的有效性。

**[Gaming the Answer Matcher: Examining the Impact of Text Manipulation on Automated Judgment](gaming_the_answer_matcher_examining_the_impact_of_text_manipulation_on_automated.md)**

:   本文系统性地测试了三种文本操控策略（冗长、策略性多答案嵌入、正确答案前置+矛盾）对 LLM 答案匹配评判器的影响，发现这些操控**不会提升分数甚至降低分数**，且二值评分比连续评分更鲁棒，证明答案匹配是一种对低成本文本操控具有鲁棒性的评估方法。

**[Sketch-HARP: 分层自回归草图生成实现灵活笔画级绘制操控](generating_sketches_in_a_hierarchical_auto-regressive_proces.md)**

:   提出 Sketch-HARP 分层自回归草图生成框架，通过三阶段层次化过程（预测笔画嵌入→确定画布位置→生成绘制动作序列），首次实现草图绘制过程中的灵活笔画级操控，在替换/擦除/扩展等任务上显著优于 SketchEdit。

**[GRIM: Task-Oriented Grasping with Conditioning on Generative Examples](grim_task-oriented_grasping_with_conditioning_on_generative_examples.md)**

:   本文提出 GRIM（Grasp Re-alignment via Iterative Matching），一种**免训练**的任务导向抓取（TOG）框架，通过 **retrieve–align–transfer** 流水线结合视频生成模型和多源记忆库，利用基于 DINO 特征的语义 3D 对齐实现跨物体的功能性抓取迁移，仅用 210 个记忆实例即超越了在 379K 样本上训练的 GraspMolmo。

**[H-GAR: A Hierarchical Interaction Framework via Goal-Driven Observation-Action Refinement for Robotic Manipulation](h-gar_a_hierarchical_interaction_framework_via_goal-driven_observation-action_re.md)**

:   提出层次化目标驱动框架 H-GAR，通过先预测目标观测再合成中间观测、并利用历史动作记忆库细化粗粒度动作，实现了观测与动作的显式双向交互，在仿真和真实机器人操控任务上取得 SOTA。

**[Human-Centric Open-Future Task Discovery: Formulation, Benchmark, and Scalable Tree-Based Search](human-centric_open-future_task_discovery_formulation_benchmark_and_scalable_tree.md)**

:   本文提出并形式化了**人类中心开放未来任务发现（HOTD）**问题——在人类意图并发且动态变化的场景中，发现那些在多种可能未来中都能减少人类负担的任务。同时构建了 HOTD-Bench 基准（2K+ 真实视频），并提出 **CMAST** 框架（协作多智能体搜索树），通过多智能体系统和可扩展搜索树显著超越现有 LMM 方法。

**[Human Cognitive Biases in Explanation-based Interaction: The Case of Within and Between Session Order Effect](human_cognitive_biases_in_explanation-based_interaction_the_case_of_within_and_b.md)**

:   本文通过两项大规模用户研究（总计 713 名参与者）系统评估了**顺序效应**（order effect）对解释性交互学习（XIL）的影响，发现顺序效应对用户反馈质量的影响**有限且不一致**，且仅在 session 内（而非 session 间）有显著但微弱的影响——总体结论是顺序效应不构成 XIL 实际应用的重大障碍。

**[iSeal: Encrypted Fingerprinting for Reliable LLM Ownership Verification](iseal_encrypted_fingerprinting_for_reliable_llm_ownership_verification.md)**

:   提出 iSeal——首个在模型窃取者完全控制推理过程的黑盒场景下仍能可靠验证 LLM 所有权的主动指纹方法，通过外部加密编码器 + RSC 纠错 + 相似度匹配三重机制，在 12 个 LLM、10+ 种攻击下均保持 100% 指纹成功率（FSR），而已有方法降至 0%。

**[LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward](laf-grpo_in-situ_navigation_instruction_generation_for_the_visually_impaired_via.md)**

:   提出 LaF-GRPO 框架，利用 LLM 模拟视障用户对导航指令的响应作为奖励信号，通过 GRPO 后训练 VLM 来生成更精确、更安全的视障导航指令，并构建了 27k 样本的 NIG4VI 基准数据集。

**[More Than Irrational: Modeling Belief-Biased Agents](more_than_irrational_modeling_belief-biased_agents.md)**

:   提出一种计算理性（CR）用户模型框架，将人类看似"不理性"的行为解释为在有限记忆（信念偏差）下的最优决策，通过嵌套粒子滤波（NPF）在线推断用户的潜在记忆界限参数 $\theta$ 和偏差信念状态 $\tilde{b}$，PM误差在45步内降低90%，并在辅助POMDP中展示自适应AI助手策略。

**[Neural Graph Navigation for Intelligent Subgraph Matching](neural_graph_navigation_for_intelligent_subgraph_matching.md)**

:   提出 NeuGN（Neural Graph Navigation）框架，首次将生成式神经导航集成到子图匹配的核心枚举阶段，通过 QSExtractor 提取查询图结构信号 + GGNavigator 将暴力枚举转为结构感知的候选节点优先排序，在保证完备性的同时将 First Match Steps 最高减少 98.2%。

**[PanoNav: Mapless Zero-Shot Object Navigation with Panoramic Scene Parsing and Dynamic Memory](panonav_mapless_zero-shot_object_navigation_with_panoramic_scene_parsing_and_dyn.md)**

:   提出 PanoNav，一个仅使用 RGB 图像的无地图零样本目标导航框架，通过全景场景解析（Panoramic Scene Parsing）释放 MLLM 的空间推理能力，并引入动态有界记忆队列（Dynamic Bounded Memory Queue）避免局部死锁问题。

**[Realistic Synthetic Household Data Generation at Scale](realistic_synthetic_household_data_generation_at_scale.md)**

:   提出一个基于 LLM 的双向耦合生成框架，通过人物画像驱动环境生成、环境语义引导行为生成的迭代循环过程，大规模生成包含家庭环境配置、人类行为和人机交互的合成数据集，用于训练家用机器人。

**[Recursive Visual Imagination and Adaptive Linguistic Grounding for Vision Language Navigation](recursive_visual_imagination_and_adaptive_linguistic_grounding_for_vision_langua.md)**

:   提出基于隐式场景表征（ISR）的VLN策略，通过递归视觉想象（RVI）将历史轨迹压缩为固定大小的紧凑神经网格学习高层场景先验，并通过自适应语言对齐（ALG）将指令的不同语义组件与不同网格精细匹配，在R2R-CE和ObjectNav两个连续环境导航任务上取得SOTA。

**[RENEW: Risk- and Energy-Aware Navigation in Dynamic Waterways](renew_risk-_and_energy-aware_navigation_in_dynamic_waterways.md)**

:   提出 RENEW 全局路径规划器，为水面自主航行器 (ASV) 在动态水流 (洋流) 环境中引入统一的风险感知和能量感知策略，通过自适应不可导航区域识别、最佳努力应急策略和基于约束 Delaunay 三角化的分层架构实现安全高效导航，应急碰撞测试中实现零碰撞。

**[Robust Out-of-Order Retrieval for Grid-Based Storage at Maximum Capacity](robust_out-of-order_retrieval_for_grid-based_storage_at_maximum_capacity.md)**

:   针对满载 2D 网格存储系统中检索顺序不确定的问题，提出 k-bounded perturbation 不确定性模型，证明 Θ(k) 列宽是零重定位的充要条件，并给出高效鲁棒存储求解器与贪心检索策略，当 k ≤ 0.5c 时几乎消除重定位，k 到达 c 时仍减少 50%+ 重定位。

**[SemanticVLA: Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation](semanticvla_semantic-aligned_sparsification_and_enhancement_for_efficient_roboti.md)**

:   提出 SemanticVLA 框架，通过语义引导的双视觉编码器剪枝（SD-Pruner）、语义互补层次融合（SH-Fuser）和语义条件动作耦合（SA-Coupler）三个模块，在大幅减少视觉冗余的同时增强指令-视觉-动作对齐，在 LIBERO 基准上以 97.7% 成功率超越 OpenVLA 达 21.1%，同时训练成本和推理延迟分别降低 3.0× 和 2.7×。

**[Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems](shadows_in_the_code_exploring_the_risks_and_defenses_of_llm-.md)**

:   首次系统分析 LLM 多 Agent 软件开发系统（ChatDev/MetaGPT/AgentVerse）的安全风险：提出 IMBIA 攻击框架覆盖两种威胁场景（恶意用户+良性Agent / 良性用户+恶意Agent）和 12 种恶意行为（5 大恶意软件家族），攻击成功率高达 93%（ChatDev），并设计 Adv-IMBIA 对抗性防御将 ASR 降低 40-73%。

**[SpatialActor: Exploring Disentangled Spatial Representations for Robust Robotic Manipulation](spatialactor_exploring_disentangled_spatial_representations_for_robust_robotic_m.md)**

:   提出 SpatialActor 框架，通过将语义与几何表征显式解耦，并设计语义引导几何模块（SGM）自适应融合深度噪声特征与预训练深度估计专家先验、以及空间 Transformer（SPT）编码低级空间位置线索，在 RLBench 50+ 任务上达到 87.4% 成功率（SOTA +6.0%），且在重噪声条件下比 RVT-2 高出 19.4%。

**[Theory of Mind for Explainable Human-Robot Interaction](theory_of_mind_for_explainable_human-robot_interaction.md)**

:   提出将心智理论（ToM）视为可解释AI（XAI）的一种形式，使用VXAI框架的七个评价标准系统评估现有HRI中的ToM研究，发现关键缺陷（特别是忠实度缺失），并主张将ToM整合到XAI框架中以实现用户导向的解释。

**[To Align or Not to Align: Strategic Multimodal Representation Alignment for Optimal Performance](to_align_or_not_to_align_strategic_multimodal_representation_alignment_for_optim.md)**

:   通过引入可控对比学习模块系统调节对齐强度 $\lambda$，结合偏信息分解(PID)框架量化模态间冗余-独特-协同信息结构，揭示显式对齐的效用高度依赖于数据特性：冗余主导时对齐有益，独特主导时有害，混合场景存在最优 $\lambda^*$。

**[TouchFormer: A Robust Transformer-based Framework for Multimodal Material Perception](touchformer_a_robust_transformer-based_framework_for_multimodal_material_percept.md)**

:   提出 TouchFormer，一个鲁棒的多模态融合框架，通过模态自适应门控（MAG）、模态内/模态间注意力机制和跨实例嵌入正则化（CER）三个互补模块，在视觉受损条件下实现可靠的材质感知，并在火灾场景机器人分拣实验中验证有效性。

**[Towards Reinforcement Learning from Neural Feedback: Mapping fNIRS Signals to Agent Performance](towards_reinforcement_learning_from_neural_feedback_mapping_.md)**

:   提出 NEURO-LOOP 框架，利用 fNIRS（功能性近红外光谱）脑信号作为隐式神经反馈评估 RL agent 表现，发布 25 名被试 × 3 领域 × 6 条件的 fNIRS 数据集，分类 F1 达 67%（二分类）/ 46%（多分类），跨被试 fine-tuning 分别提升 17% 和 41%，奠定 Reinforcement Learning from Neural Feedback (RLNF) 基础。

**[Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation](unintended_misalignment_from_agentic_fine-tuning_risks_and_m.md)**

:   本文揭示了在良性 Agent 数据上微调 LLM 会导致意外的安全对齐偏移（攻击成功率增加 32-38%），并提出 PING（Prefix Injection Guard）——通过迭代生成+评估自然语言前缀来引导微调后的 Agent 拒绝有害请求，平均提升拒绝率 66%（Web）和 44%（代码），同时保持任务性能（仅降 1.8%）。

**[UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories](urbannav_learning_language-guided_urban_navigation_from_web-scale_human_trajecto.md)**

:   提出 UrbanNav，利用网络规模的城市步行视频（YouTube 上 1500+ 小时、300 万条指令-轨迹-地标三元组），通过自动化标注管线和鲁棒过滤机制训练语言引导的城市导航策略，在真实世界部署中达到 83.3% 的导航成功率。

**[When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets](when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)**

:   提出 CAIA 基准测试，通过加密货币市场作为天然对抗性实验室，评估 17 个 SOTA 大模型在高风险对抗环境中的 agent 能力，揭示前沿模型仅达 67.4% 准确率（GPT-5）vs 人类 80%，并发现系统性工具选择灾难。
