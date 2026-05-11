---
title: >-
  ACL2026 LLM 推理方向37篇论文解读
description: >-
  37篇ACL2026的 LLM 推理方向论文解读，涵盖推理、LLM、对齐/RLHF、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM 推理

**💬 ACL2026** · **37** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (12)](../../CVPR2026/llm_reasoning/index.md) · [🔬 ICLR2026 (65)](../../ICLR2026/llm_reasoning/index.md) · [🤖 AAAI2026 (30)](../../AAAI2026/llm_reasoning/index.md) · [🧠 NeurIPS2025 (66)](../../NeurIPS2025/llm_reasoning/index.md) · [📹 ICCV2025 (3)](../../ICCV2025/llm_reasoning/index.md) · [🧪 ICML2025 (15)](../../ICML2025/llm_reasoning/index.md)

🔥 **高频主题：** 推理 ×21 · LLM ×7 · 对齐/RLHF ×2 · Agent ×2

**[AIM-CoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](aim-cot_active_information-driven_multimodal_chain-of-thought_for_vision-languag.md)**

:   提出 AIM-CoT 框架，通过信息觅食理论驱动的主动视觉证据选择(AVP)和基于注意力偏移的动态触发机制(DAT)，解决交错模态思维链(I-MCoT)中"看什么"和"何时看"两个核心问题。

**[Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data](budget-aware_anytime_reasoning_with_llm-synthesized_preference_data.md)**

:   本文提出了一种预算感知的任意时推理（anytime reasoning）框架和 Anytime Index 指标，用于量化 LLM 在有限 token 预算下的推理质量-效率权衡，并设计了基于 LLM 自合成偏好数据的推理时自改进方法（PDP），在规划、数学和科学 QA 任务上显著提升了中间和最终解的质量。

**[Chain-of-Thought as a Lens: Evaluating Structured Reasoning Alignment between Human Preferences and Large Language Models](chain-of-thought_as_a_lens_evaluating_structured_reasoning_alignment_between_hum.md)**

:   本文提出 Alignment Score——一种基于语义熵矩阵的语义级指标，通过比较模型生成的思维链与人类偏好参考链的中间步骤来量化推理对齐度，发现 Alignment Score 与任务准确率、可读性和连贯性高度相关，且 2-hop 推理是对齐的峰值深度。

**[Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)**

:   提出 OlymMATH，首个统一自然语言评估和形式化定理证明的奥赛级数学基准，包含350题双语（中英文）题目，涵盖OlymMATH-EASY/HARD（200题数值答案）和OlymMATH-LEAN（150题Lean 4形式化），揭示最强模型在HARD子集上仅58.4%准确率。

**[Decoupling the Effect of Chain-of-Thought Reasoning: A Human Label Variation Perspective](decoupling_the_effect_of_chain-of-thought_reasoning_a_human_label_variation_pers.md)**

:   本文通过 Cross-CoT 实验和逐步分析，揭示了 CoT 推理的"解耦机制"：最终准确率由 CoT 内容决定（99% 方差贡献），但分布排序由模型内在先验主导（>80%），说明长 CoT 是强大的决策器但弱的分布校准器。

**[Discovering a Shared Logical Subspace: Steering LLM Logical Reasoning via Alignment of Natural-Language and Symbolic Views](discovering_a_shared_logical_subspace_steering_llm_logical_reasoning_via_alignme.md)**

:   发现 LLM 内部存在一个共享的逻辑子空间，可同时对齐自然语言和符号逻辑两种推理表示，通过在推理时沿该子空间引导激活可无训练提升逻辑推理准确率最高达 11 个百分点。

**[Dissecting Failure Dynamics in Large Language Model Reasoning](dissecting_failure_dynamics_in_large_language_model_reasoning.md)**

:   通过分析 LLM 推理轨迹，发现错误集中在早期的少数关键转折点，错误发生后模型进入"认知螺旋"——局部连贯但全局错误地不断延伸；基于此提出 GUARD 框架，在熵信号检测到的高风险转折点处进行短距分支修复。

**[Do Not Step Into the Same River Twice: Learning to Reason from Trial and Error](do_not_step_into_the_same_river_twice_learning_to_reason_from_trial_and_error.md)**

:   提出 LTE (Learning to reason from Trial and Error)，通过将模型自身生成的错误答案作为提示引导额外 rollout，在不依赖外部专家的情况下有效缓解 RLVR 中的探索停滞问题。

**[DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency](dpc_training-free_text-to-sql_candidate_selection_via_dual-paradigm_consistency.md)**

:   DPC 将 Text-to-SQL 的候选选择从"在隐藏数据上猜测"转化为"在可见数据上确定性验证"：构造最小区分数据库（MDD）使冲突 SQL 产生不同结果，再用 Python/Pandas 解作为参考锚点通过跨范式一致性选择正确候选，在 BIRD 和 Spider 上超越 Self-Consistency 最高 2.2%。

**[Efficient PRM Training Data Synthesis via Formal Verification](efficient_prm_training_data_synthesis_via_formal_verification.md)**

:   本文提出 FoVer，一个利用形式化验证工具（Z3 和 Isabelle）为形式化推理任务的步骤级推理链自动标注正确性标签的框架，生成 FoVer-40K 训练集并微调 PRM，在 12 个推理基准上展示了从形式化到非形式化的迁移能力和跨任务泛化能力。

**[Efficient Process Reward Modeling via Contrastive Mutual Information](efficient_process_reward_modeling_via_contrastive_mutual_information.md)**

:   提出 CPMI（Contrastive Pointwise Mutual Information），一种高效的自动步级奖励标注方法，通过对比推理步骤对正确答案和错误答案的条件概率变化量来估计步级贡献，比 Monte Carlo 估计减少 84% 构建时间和 98% token 生成量，同时在过程级评估和数学推理基准上取得更高准确率。

**[Explicit Trait Inference for Multi-Agent Coordination](explicit_trait_inference_for_multi-agent_coordination.md)**

:   提出显式特质推理（ETI）方法，基于心理学中温暖和能力两个维度让LLM智能体推理并追踪合作伙伴的行为特征，在经济博弈中减少45-77%收益损失，在MultiAgentBench上提升3-29%任务表现。

**[Failure Modes in Multi-Hop QA: The Weakest Link Effect and the Recognition Bottleneck](failure_modes_in_multi-hop_qa_the_weakest_link_effect_and_the_recognition_bottle.md)**

:   本文提出 Multi-Focus Attention Instruction (MFAI) 作为语义探针，揭示多跳 QA 中的"最弱链效应"——多跳推理性能由最不可见证据的绝对位置决定而非事实间距离，失败主要源于识别瓶颈而非推理缺陷，且 System-2 推理模型能有效抵御位置偏差和误导性注意力线索。

**[FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents](fs-researcher_test-time_scaling_for_long-horizon_research_tasks_with_file-system.md)**

:   本文提出 FS-Researcher，一个基于文件系统的双 Agent 深度研究框架，通过 Context Builder 构建层次化知识库、Report Writer 分节撰写报告，利用持久化工作空间突破上下文窗口限制，在 DeepResearch Bench 上达到 53.94 RACE（SOTA），并展示了上下文构建计算量与报告质量的正相关测试时扩展效应。

**[GanitLLM: Difficulty-Aware Bengali Mathematical Reasoning through Curriculum-GRPO](ganitllm_difficulty-aware_bengali_mathematical_reasoning_through_curriculum-grpo.md)**

:   本文提出 GanitLLM，首个真正用孟加拉语进行推理（而非翻译或用英语推理）的数学推理模型，构建了难度标注的孟加拉语数学数据集 Ganit，并提出 Curriculum-GRPO 解决低资源语言 GRPO 训练中的冷启动问题，4B 模型在 Bn-MGSM 上提升 8 个准确率百分点，孟加拉语推理 token 从 14% 提升至 88%。

**[Generating Effective CoT Traces for Mitigating Causal Hallucination](generating_effective_cot_traces_for_mitigating_causal_hallucination.md)**

:   本文首先提出了因果幻觉率（CHR）指标来量化小型 LLM 在事件因果识别中过度预测因果关系的倾向，然后通过系统实验确定了有效 CoT 数据的两个关键标准（充分长度的语义解释+与目标模型对齐的分布），设计了一套低成本的 CoT 数据生成管线，将 Qwen2.5-1.5B 的 CHR 从 83.54% 降至 6.26%，同时提升平均准确率至 66.00%。

**[How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study](how_should_we_enhance_the_safety_of_large_reasoning_models_an_empirical_study.md)**

:   本文系统研究如何通过 SFT 增强大型推理模型（LRM）的安全性，发现直接蒸馏安全响应效果有限的根因是五种风险推理模式（尤其是"弱犹豫"），提出针对性的蒸馏策略将 PAIR 攻击成功率从 63% 降至 13%，并发现短推理链和模板推理在安全性上与长推理链表现相当。

**[JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents](jtpro_a_joint_tool-prompt_reflective_optimization_framework_for_language_agents.md)**

:   JTPRO 提出了一种无需模型微调的联合优化框架，通过反思驱动的迭代编辑同时优化全局指令和逐工具的 schema/参数描述，在大规模工具库场景下显著提升工具选择和参数填充的端到端成功率，相比 GEPA 等基线在 OSR 上提升 5%–20%。

**[Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning](know_thy_enemy_securing_llms_against_prompt_injection_via_diverse_data_synthesis.md)**

:   本文提出 InstruCoT，通过合成覆盖多种注入向量和威胁场景的多样化训练数据，并引入基于情境感知模型的三阶段指令级思维链微调，使 LLM 在面对各类提示注入攻击时能有效识别并拒绝恶意指令，在行为偏离、隐私泄露和有害输出三个维度上大幅超越现有防御方法。

**[Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners](large_reasoning_models_are_not_yet_multilingual_latent_reasoners.md)**

:   本文系统性地研究了大型推理模型（LRM）在 11 种语言上的潜在推理行为，发现潜在推理能力存在于多语言中但分布不均（高资源语言强、低资源弱），且内部推理动态趋于以英语为中心的共享路径。

**[Logical Phase Transitions: Understanding Collapse in LLM Logical Reasoning](logical_phase_transitions_understanding_collapse_in_llm_logical_reasoning.md)**

:   本文发现 LLM 逻辑推理存在"逻辑相变"现象——性能在特定复杂度阈值处突然崩塌而非平滑退化，提出逻辑复杂度度量（LoCM）来量化这一现象，并设计神经符号课程调优框架（NSCT），通过自适应神经-符号对齐和复杂度感知课程优化，在五个基准上平均提升 naive prompting +1.26 和 CoT +3.95 准确率。

**[MARCH: Evaluating the Intersection of Ambiguity Interpretation and Multi-hop Inference](march_evaluating_the_intersection_of_ambiguity_interpretation_and_multi-hop_infe.md)**

:   提出 MARCH 基准（2,209 个多跳歧义问题）和 CLARION 框架，首次系统研究歧义解析与多步推理交叉场景下的 QA 挑战，揭示现有 SOTA 模型在此类问题上的严重不足。

**[MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis](mathagent_adversarial_evolution_of_constraint_graphs_for_mathematical_reasoning_.md)**

:   提出基于约束图对抗进化的分层数据合成框架 MathAgent，将数据合成从文本生成任务重构为约束图的无监督优化问题，通过 Legislator 三Agent系统进化问题骨架再由 Executor 实例化为自然语言，仅 1K 合成样本即超越 LIMO 和 s1K 在八个数学基准上的表现。

**[Parallel Test-Time Scaling for Latent Reasoning Models](parallel_test-time_scaling_for_latent_reasoning_models.md)**

:   本文首次将并行测试时缩放（parallel TTS）引入潜在推理模型，提出两种基于不确定性理论的随机采样策略（MC-Dropout 和加性高斯噪声）以及一个步级对比训练的潜在奖励模型（LatentRM），使得在连续向量空间中进行推理的模型也能通过并行采样+聚合获得稳定的性能提升。

**[Process Reward Models Meet Planning: Generating Precise and Scalable Datasets for Step-Level Rewards](process_reward_models_meet_planning_generating_precise_and_scalable_datasets_for.md)**

:   本文提出利用规划领域定义语言（PDDL）自动生成大规模、高精度的步骤级奖励数据集，用于训练过程奖励模型（PRM），在数学和非数学推理基准上均取得显著提升。

**[ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering](recoqa_a_benchmark_for_tool-augmented_and_multi-step_reasoning_in_real_estate_qu.md)**

:   本文构建了 ReCoQA——一个包含 29,270 个房地产问答对的大规模基准，要求模型融合数据库查询和地图 API 调用进行混合多源推理，并提出层次化多 Agent 框架 HIRE-Agent 作为强基线，系统性地揭示了现有 LLM 在垂直领域复杂推理中的瓶颈。

**[Reinforced Efficient Reasoning via Semantically Diverse Exploration](reinforced_efficient_reasoning_via_semantically_diverse_exploration.md)**

:   ROSE 提出语义熵引导的 MCTS 分支策略和长度感知的段级优势估计，解决了现有 MCTS-based RLVR 方法探索多样性不足和推理效率低的问题，在多个数学推理基准上取得最优 pass@8 性能。

**[Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning](render-of-thought_rendering_textual_chain-of-thought_as_images_for_visual_latent.md)**

:   提出 Render-of-Thought（RoT），首次将文本 CoT 推理步骤渲染为图像，利用预训练视觉编码器作为语义锚点将 LLM 隐状态对齐到视觉嵌入空间，实现 3-4 倍 token 压缩和显著推理加速，同时保持推理链的可分析性。

**[Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)**

:   系统性研究了 RLVR 训练中 LLM 的熵动态，揭示正优势 token 是熵崩塌的主要驱动因素，并提出 Positive-Advantage Reweighting 方法通过动态调整正优势 token 的损失权重来有效调控模型熵。

**[Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models](scaling_test-time_compute_to_achieve_ioi_gold_medal_with_open-weight_models.md)**

:   提出 GenCluster，一个可扩展的测试时计算框架，通过大规模并行生成→行为聚类→锦标赛排名→循环提交策略，首次使开源模型 gpt-oss-120b 在 IOI 2025 上达到金牌水平（446.75/600 分）。

**[Self-Reinforcing Controllable Synthesis of Rare Relational Data via Bayesian Calibration](self-reinforcing_controllable_synthesis_of_rare_relational_data_via_bayesian_cal.md)**

:   本文提出RDDG，基于渐进式CoT的表格数据合成框架，通过核心集选择、关系挖掘和自强化反馈机制引导LLM生成高保真表格数据，在不平衡分类上平均提升2%+ Macro-F1。

**[Semantic-Aware Logical Reasoning via a Semiotic Framework](semantic-aware_logical_reasoning_via_a_semiotic_framework.md)**

:   提出 LogicAgent，一个基于格雷马斯符号方阵(Semiotic Square)的逻辑推理框架，通过多视角语义分析和反思验证，在语义复杂和逻辑复杂双重挑战下实现 SOTA 逻辑推理性能。

**[Think Outside the Policy: In-Context Steered Policy Optimization](think_outside_the_policy_in-context_steered_policy_optimization.md)**

:   提出 ICPO (In-Context Steered Policy Optimization)，利用大语言模型自身的上下文学习(ICL)能力作为隐式专家引导，在 RLVR 训练中扩展策略探索空间，无需依赖外部更强模型的推理轨迹。

**[Towards Effective In-context Cross-domain Knowledge Transfer via Domain-invariant-neurons-based Retrieval](towards_effective_in-context_cross-domain_knowledge_transfer_via_domain-invarian.md)**

:   本文提出 DIN-Retrieval，通过识别 LLM 中跨域激活极性一致的域不变神经元（DIN），构建域鲁棒的表示子空间用于检索结构兼容的跨域示例，首次证明了使用跨域 ICL 示例提升 LLM 推理性能的可行性，在数学-逻辑推理迁移上平均提升 1.8%。

**[TrigReason: Trigger-Based Collaboration between Small and Large Reasoning Models](trigreason_trigger-based_collaboration_between_small_and_large_reasoning_models.md)**

:   TrigReason 提出基于事件触发的大小推理模型协作框架，通过分析小模型三类推理风险（路径偏离、认知过载、恢复失能），设计策略引导、认知卸载和干预请求三种触发器替代逐步轮询验证，在保持 LRM 精度的同时将 1.70-4.79 倍更多推理步骤卸载给小模型，延迟降低 43.9%、API 成本降低 73.3%。

**[TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards](trojail_trajectory-level_optimization_for_multi-turn_large_language_model_jailbr.md)**

:   本文将自动化多轮越狱攻击建模为多轮强化学习问题，提出 TROJail，通过两个启发式过程奖励（过度有害惩罚和语义相关性递进）缓解结果奖励的稀疏监督问题，在多个模型和基准上显著提升攻击成功率。

**[When Is Thinking Enough? Early Exit via Sufficiency Assessment for Efficient Reasoning](when_is_thinking_enough_early_exit_via_sufficiency_assessment_for_efficient_reas.md)**

:   提出 DTSR 框架，通过检测推理过程中的"反思信号"（如 Wait、Alternatively）并在该位置让模型自我评估当前推理的"充分性"来决定是否提前终止推理，在 Qwen3 系列模型上实现 28.9%–34.9% 的推理长度缩减且几乎不损失精度。
