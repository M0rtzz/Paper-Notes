---
title: >-
  ACL2026 强化学习方向34篇论文解读
description: >-
  34篇ACL2026的强化学习方向论文解读，涵盖 LLM、强化学习、推理、Agent、对话系统、RAG等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎮 强化学习

**💬 ACL2026** · **34** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (19)](../../CVPR2026/reinforcement_learning/index.md) · [🔬 ICLR2026 (139)](../../ICLR2026/reinforcement_learning/index.md) · [🤖 AAAI2026 (71)](../../AAAI2026/reinforcement_learning/index.md) · [🧠 NeurIPS2025 (172)](../../NeurIPS2025/reinforcement_learning/index.md) · [📹 ICCV2025 (7)](../../ICCV2025/reinforcement_learning/index.md) · [🧪 ICML2025 (81)](../../ICML2025/reinforcement_learning/index.md)

🔥 **高频主题：** LLM ×10 · 强化学习 ×10 · 推理 ×8 · Agent ×3 · 对话系统 ×3

**[A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions](a_survey_of_reinforcement_learning_for_large_language_models_under_data_scarcity.md)**

:   首篇系统综述数据稀缺条件下LLM强化学习的工作，提出数据中心、训练中心、框架中心三层分类体系，覆盖数据剪枝/合成/压缩、轨迹生成/奖励工程/策略优化、以及自演化/协同演化/多智能体演化等方向。

**[Adaptive Instruction Composition for Automated LLM Red-Teaming](adaptive_instruction_composition_for_automated_llm_red-teaming.md)**

:   提出 Adaptive Instruction Composition (AIC) 框架，利用 Neural Thompson Sampling 在众包有害查询和越狱策略的组合空间中自适应地选择攻击指令，同时优化攻击成功率和多样性，在 Harmbench 上大幅超越已有方法。

**[AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation](aj-bench_benchmarking_agent-as-a-judge_for_environment-aware_evaluation.md)**

:   提出 AJ-Bench，首个系统评估 Agent-as-a-Judge 能力的基准，覆盖搜索、数据系统和 GUI 三个领域共 155 个任务和 516 条标注轨迹，实验表明 Agent-as-a-Judge 比 LLM-as-a-Judge 平均 F1 提升约 13 个百分点。

**[AttnPO: Attention-Guided Process Supervision for Efficient Reasoning](attnpo_attention-guided_process_supervision_for_efficient_reasoning.md)**

:   提出 AttnPO，一个利用模型内在注意力信号进行步级信用分配的低开销过程监督 RL 框架，通过识别 Key-Focus Heads（KFH）区分冗余和关键推理步骤，在大幅缩短推理长度的同时显著提升准确率。

**[Bootstrapping Code Translation with Weighted Multilanguage Exploration](bootstrapping_code_translation_with_weighted_multilanguage_exploration.md)**

:   BootTrans 提出了一种自举式多语言代码翻译方法，通过利用单一枢纽语言（Python）的测试用例作为跨语言验证预言，结合双池架构进行经验收集扩展训练数据，并设计语言感知加权机制动态优先处理困难的翻译方向，在 HumanEval-X 和 TransCoder-Test 上显著超越基线。

**[Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning](bridging_sft_and_rl_dynamic_policy_optimization_for_robust_reasoning.md)**

:   提出 DYPO（Dynamic Policy Optimization），通过动态难度分级将样本路由到不同优化路径——Hard样本用多教师蒸馏降低SFT偏差、Mid样本用Group Alignment Loss降低RL方差，在数学推理benchmark上平均提升4.8%，OOD任务提升13.3%。

**[CAP: Controllable Alignment Prompting for Unlearning in LLMs](cap_controllable_alignment_prompting_for_unlearning_in_llms.md)**

:   提出 CAP 框架，通过训练轻量 SLM 生成可控的提示前缀来引导冻结的 LLM 选择性遗忘目标知识，无需修改模型参数，实现了可逆、可迁移的 LLM 知识遗忘。

**[CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning](ce-gppo_coordinating_entropy_via_gradient-preserving_clipping_policy_optimizatio.md)**

:   提出 CE-GPPO 算法，通过 stop-gradient 操作重新引入 PPO 裁剪区间外低概率 token 的梯度信号，实现对策略熵的精细化协调控制，在探索-利用之间取得更好平衡。

**[ChipSeek: Optimizing Verilog Generation via EDA-Integrated Reinforcement Learning](chipseek_optimizing_verilog_generation_via_eda-integrated_reinforcement_learning.md)**

:   ChipSeek 提出了一个将 EDA 工具链直接集成到训练循环中的分层奖励 RL 框架，通过课程引导的动态策略优化（CDPO）使 LLM 能够生成同时满足功能正确性和 PPA（功耗-性能-面积）优化的 RTL 代码，在标准基准上达到 SOTA。

**[Controlling Multimodal Conversational Agents with Coverage-Enhanced Latent Actions](controlling_multimodal_conversational_agents_with_coverage-enhanced_latent_actio.md)**

:   提出为多模态对话智能体（MCA）构建紧凑的潜在动作空间来替代巨大的 token 动作空间进行 RL 微调，通过跨模态投影器和循环一致性损失利用配对图文数据和纯文本数据共同构建码本，将动作空间从 152K（词表大小）压缩到 128（码本大小），在两个对话任务上全面超越 token 级 RL 基线。

**[Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training](data_mixing_agent_learning_to_re-weight_domains_for_continual_pre-training.md)**

:   本文提出 Data Mixing Agent，首个基于模型的端到端领域重加权框架，通过在大量数据混合轨迹上使用 CQL 强化学习训练小型代理来学习可泛化的数据混合启发式，在数学推理持续预训练中平衡源领域和目标领域性能，且可泛化到未见过的源领域、目标模型和领域空间。

**[Deliberative Searcher: Improving LLM Reliability via Reinforcement Learning with Constraints](deliberative_searcher_improving_llm_reliability_via_reinforcement_learning_with_.md)**

:   本文提出 Deliberative Searcher，一个推理优先的框架，将搜索操作集成到 CoT 生成中并保持显式置信度校准，使用自适应拉格朗日乘子的约束 RL 联合优化正确性和可靠性，将 7B 模型的平均"错误-确定"率从基线的 54% 降至 2%。

**[FaithLens: Detecting and Explaining Faithfulness Hallucination](faithlens_detecting_and_explaining_faithfulness_hallucination.md)**

:   本文提出 FaithLens，一个 8B 参数的忠实性幻觉检测模型，通过高质量数据合成+三维过滤（标签正确性、解释质量、数据多样性）进行冷启动 SFT，再用基于规则的强化学习（预测正确性奖励+解释质量奖励）进一步优化，在 12 个任务上超越 GPT-5.2 和 o3，同时提供高质量的解释性输出。

**[Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)**

:   本文提出 FTRL 框架，通过五阶段自动化管线构建稳定可控的工具使用训练环境，并设计结合工具调用精度和任务完成度的可验证奖励机制，与偏好优化 RL 算法结合后，在 7B-14B 模型上实现平均超 10% 的工具使用性能提升，甚至超越最强闭源模型。

**[Frame of Reference: Addressing the Challenges of Common Ground Representation in Dialogue](frame_of_reference_addressing_the_challenges_of_common_ground_representation_in_.md)**

:   本文提出 IndiRef 基准测试，用于评估对话系统通过"关系指代"（如"昨天我们去的那个公园旁边的咖啡馆"）建立和利用持久共识（common ground）的能力，发现现有 LLM 在全上下文条件下准确率不超过 50%，并通过合成数据 + GRPO 强化学习训练将性能提升 15-20%。

**[From Passive Metric to Active Signal: The Evolving Role of Uncertainty Quantification in Large Language Models](from_passive_metric_to_active_signal_the_evolving_role_of_uncertainty_quantifica.md)**

:   本文系统综述了 LLM 中不确定性量化从"被动诊断指标"到"主动控制信号"的功能演化，覆盖三大前沿领域：高级推理（引导计算分配和自我纠正）、自主代理（驱动工具使用和信息获取的元认知决策）、以及强化学习（缓解奖励黑客并通过内在奖励实现自我改进）。

**[GeoRA: Geometry-Aware Low-Rank Adaptation for RLVR](geora_geometry-aware_low-rank_adaptation_for_rlvr.md)**

:   本文提出 GeoRA，一种专为强化学习可验证奖励（RLVR）设计的低秩适配方法，通过构建几何约束矩阵（融合谱先验和欧几里得先验）提取 RL 更新子空间的主方向进行 SVD 初始化，同时冻结残差矩阵作为结构锚，在 1.5B-32B 参数的 Qwen/Llama 模型上，数学、医学和代码 RLVR 任务中一致超越 LoRA、PiSSA、MiLoRA 等基线，且具备更强的域外泛化和更少的能力遗忘。

**[ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following](imprif_stronger_implicit_reasoning_leads_to_better_complex_instruction_following.md)**

:   ImpRIF 将复杂指令中的隐式推理结构形式化为可验证的显式推理图（ERG），基于此构建大规模单轮/多轮数据并通过 SFT+过程验证 RL 训练，使 4B-32B 模型在五个指令遵循基准上显著超越基座模型，32B 模型甚至超越部分大型商用模型。

**[Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)**

:   本文提出 LcRL 框架，通过语言耦合的 GRPO 策略优化和反一致性惩罚奖励，解决多语言 RAG 中的知识偏差和知识冲突问题，在多语言问答任务上取得显著提升。

**[LENS: Less Noise, More Voice — Reinforcement Learning for Reasoning via Instruction Purification](less_noise_more_voice_reinforcement_learning_for_reasoning_via_instruction_purif.md)**

:   LENS 发现 RLVR 中许多探索失败并非因为问题难度，而是因为 prompt 中少量（<5%）干扰 token，通过识别和删除这些 token 来提升 rollout 成功率，并将净化 rollout 的学习信号转移到原始噪声 prompt 的策略优化中，平均提升 3.88% 并加速 1.6 倍。

**[Optimizing User Profiles via Contextual Bandits for Retrieval-Augmented LLM Personalization](optimizing_user_profiles_via_contextual_bandits_for_retrieval-augmented_llm_pers.md)**

:   提出 PURPLE 框架，将检索增强 LLM 个性化中的用户画像构建问题建模为上下文老虎机问题，通过 Plackett-Luce 排序模型捕捉记录间依赖关系，以 LLM 对参考回复的 log-likelihood 作为奖励信号，直接优化检索以匹配生成质量。

**[Quality Over Clicks: Intrinsic Quality-Driven Iterative RL for Cold-Start E-Commerce Query Suggestion](quality_over_clicks_intrinsic_quality-driven_iterative_reinforcement_learning_fo.md)**

:   提出 Cold-EQS，一个面向冷启动电商场景的查询建议框架，利用可回答性、事实准确性和信息增益作为内在质量奖励，通过迭代强化学习持续优化查询建议质量，在线 chatUV 提升 6.81%。

**[ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning](rerec_reasoning-augmented_llm-based_recommendation_assistant_via_reinforcement_f.md)**

:   本文提出 ReRec，一个基于强化微调（RFT）的推荐助手框架，通过双图增强的奖励塑形提供细粒度奖励信号、推理感知的优势估计对推理步骤进行差异化监督、以及在线课程调度器动态调整训练难度，使 LLM 能处理复杂的多步推理推荐查询，在 RecBench+ 基准上显著超越现有方法。

**[Right at My Level: A Unified Multilingual Framework for Proficiency-Aware Text Simplification](right_at_my_level_a_unified_multilingual_framework_for_proficiency-aware_text_si.md)**

:   提出 Re-RIGHT 框架，通过三模块奖励（词汇覆盖率+语义保持+连贯性）的 GRPO 训练，用 4B 策略模型在英日韩中四种语言上实现按学习者熟练度等级（CEFR/JLPT/TOPIK/HSK）精确简化文本，超越 GPT-5.2 和 Gemini 2.5 等大模型。

**[RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization](rl-plus_countering_capability_boundary_collapse_of_llms_in_reinforcement_learnin.md)**

:   RL-PLUS 提出混合策略优化方法，通过多重重要性采样（MIS）解决外部数据分布不匹配问题，以及探索式优势函数（EAF）引导模型学习低概率但正确的推理路径，成功突破 RLVR 导致的能力边界坍塌，在六个数学推理基准上达到 SOTA（平均 53.4），且跨模型一致提升最高达 69.2%。

**[Savoir: Learning Social Savoir-Faire via Shapley-based Reward Attribution](savoir_learning_social_savoir-faire_via_shapley-based_reward_attribution.md)**

:   本文提出 Savoir，一个基于合作博弈论的社交 RL 框架，结合期望效用（前瞻性评估话语的战略潜力）和 Shapley 值（公理化公平信用分配）解决多轮对话中的信用分配问题，在 SOTOPIA 基准上以 7B 模型达到 SOTA 性能（Hard 设置 Goal 7.18），匹配或超越 GPT-4o 和 Claude-3.5-Sonnet，且大型推理模型（o1、DeepSeek-R1）在社交任务上系统性欠佳。

**[Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study](scaling_behaviors_of_llm_reinforcement_learning_post-training_an_empirical_study.md)**

:   首次系统研究 LLM 强化学习后训练的缩放行为，在 Qwen2.5 系列(0.5B-72B)上发现性能与训练资源之间遵循幂律关系，且学习效率随模型规模增大呈饱和趋势。

**[Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning](semantic-space_exploration_and_exploitation_in_rlvr_for_llm_reasoning.md)**

:   本文指出 RLVR 中传统的 token 级探索-利用权衡是测量方式的伪象，提出在隐状态语义空间中用 Effective Rank (ER) 和其时间导数 (ERV/ERA) 来解耦探索与利用，并据此设计 VERL 方法实现两者的同步提升，在高考数学等基准上获得高达 21.4% 的提升。

**[SpiralThinker: Latent Reasoning through an Iterative Process with Text-Latent Interleaving](spiralthinker_latent_reasoning_through_an_iterative_process_with_text-latent_int.md)**

:   本文提出 SpiralThinker，通过在潜在表示空间中进行迭代更新、并与文本推理步骤交替进行的框架实现隐式推理，引入渐进对齐目标确保潜在表示在迭代过程中保持与显式推理的一致性，在数学、逻辑和常识推理任务上超越所有潜在推理基线。

**[STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems](stride-ed_a_strategy-grounded_stepwise_reasoning_framework_for_empathetic_dialog.md)**

:   本文提出 STRIDE-ED 框架，通过构建覆盖正/中/负情绪的全面共情策略体系，设计任务对齐的多阶段认知CoT推理，结合策略感知数据精炼和SFT+PPO两阶段训练，在多个开源LLM上实现共情对话SOTA，情感准确率达57.25%，BLEU-4达4.67。

**[Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)**

:   全面综述了 LLM 时代表格问答（TQA）研究，从五个维度（表格格式、问题复杂度、答案格式、模态、领域）系统化分类任务设置，按核心挑战（表格理解、复杂查询、大输入、数据异构、知识集成）组织建模方法，覆盖 277 篇论文，并前瞻性讨论了强化学习、可解释性等新兴方向。

**[Understanding Generalization in Role-Playing Models via Information Theory](understanding_generalization_in_role-playing_models_via_information_theory.md)**

:   本文提出首个信息论框架 R-EMID 来量化角色扮演模型（RPM）在用户/角色/对话分布偏移下的性能退化，通过引入推理过程和协同进化强化学习（CoRL）实现准确估计，发现用户偏移是最大的泛化风险，且强化学习是唯一一致有效的改进方法。

**[UniCreative: Unifying Long-form Logic and Short-form Sparkle via Reference-Free Reinforcement Learning](unicreative_unifying_long-form_logic_and_short-form_sparkle_via_reference-free_r.md)**

:   本文提出 UniCreative 框架，通过自适应约束偏好优化（ACPO）和自适应标准生成式奖励模型（AC-GenRM），在无需 SFT 和参考答案的条件下统一长文本（规划→写作）和短文本（直接生成）两种创意写作模式，模型涌现出自主区分任务类型的元认知能力。

**[SCRL: What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time](what_if_consensus_lies_selective-complementary_reinforcement_learning_at_test_ti.md)**

:   本文提出 SCRL（Selective-Complementary Reinforcement Learning），一个鲁棒的测试时强化学习框架，通过选择性正伪标签（严格共识标准过滤不可靠多数）和熵门控负伪标签（首次在 TTRL 中引入负监督信号来修剪错误轨迹）缓解标签噪声放大问题，在 AIME25 上比 TTRL 提升高达 10.1 个百分点。
