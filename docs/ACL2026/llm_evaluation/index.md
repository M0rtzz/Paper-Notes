---
title: >-
  ACL2026 LLM 评测方向40篇论文解读
description: >-
  40篇ACL2026的 LLM 评测方向论文解读，涵盖 LLM、推理、对话系统、翻译、语音、信息抽取等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📊 LLM 评测

**💬 ACL2026** · **40** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (26)](../../CVPR2026/llm_evaluation/index.md) · [🔬 ICLR2026 (56)](../../ICLR2026/llm_evaluation/index.md) · [🤖 AAAI2026 (39)](../../AAAI2026/llm_evaluation/index.md) · [🧠 NeurIPS2025 (79)](../../NeurIPS2025/llm_evaluation/index.md) · [📹 ICCV2025 (27)](../../ICCV2025/llm_evaluation/index.md) · [🧪 ICML2025 (49)](../../ICML2025/llm_evaluation/index.md)

🔥 **高频主题：** LLM ×13 · 推理 ×6 · 对话系统 ×4 · 翻译 ×2 · 语音 ×2

**[Abstain-R1: Calibrated Abstention and Post-Refusal Clarification via Verifiable RL](abstain-r1_calibrated_abstention_and_post-refusal_clarification_via_verifiable_r.md)**

:   Abstain-R1 提出一种**澄清感知的 RLVR 奖励**，在不可回答查询上联合优化"明确拒答"和"拒答后给出有用澄清（指出缺失信息）"，使 3B 模型在拒答和澄清质量上接近甚至超越 DeepSeek-R1 等大模型。

**[AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)**

:   提出AnchorMem记忆框架，受普鲁斯特现象启发，将检索单元（原子事实）与生成上下文（原始交互）解耦，通过关联事件图连接碎片化记忆，在LoCoMo基准上大幅超越A-Mem、Mem0等现有记忆系统。

**[Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues](are_they_lovers_or_friends_evaluating_llms39_social_reasoning_in_english_and_kor.md)**

:   本文提出 SCRIPTS 基准，包含 1.1K 英语和韩语电影对话，通过三层概率标签（HIGHLY LIKELY / LESS LIKELY / UNLIKELY）评估 9 个 LLM 的社会关系推理能力，发现模型在英语上准确率仅 75-80%、韩语 58-69%，且 CoT 和思维模型对社会推理几乎无帮助。

**[AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage](autoreproduce_automatic_ai_experiment_reproduction_with_paper_lineage.md)**

:   AutoReproduce 提出了一个多智能体框架，通过"论文谱系"算法从引用文献中挖掘隐式领域知识，实现端到端的论文实验自动复现，在自建基准 ReproduceBench 上的代码执行率达 94.87%，性能差距仅 19.72%。

**[Beyond Reproduction: A Paired-Task Framework for Assessing LLM Comprehension and Creativity in Literary Translation](beyond_reproduction_a_paired-task_framework_for_assessing_llm_comprehension_and_.md)**

:   提出配对任务框架联合评估 LLM 的文学文本理解能力和翻译创造力，基于 11 本英文经典小说对 23 个模型进行大规模测评，发现强理解力并不能转化为人类水平的翻译创造力。

**[BizCompass: Benchmarking the Reasoning Capabilities of LLMs in Business Knowledge and Applications](bizcompass_benchmarking_the_reasoning_capabilities_of_llms_in_business_knowledge.md)**

:   本文提出 BizCompass，一个连接理论基础与实际应用的商业推理基准，覆盖金融/经济/统计/运营四个知识域和分析师/交易员/顾问三个应用角色，系统评估了开源与闭源 LLM 的商业推理能力，揭示理论知识向实际表现转化的规律。

**[Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)**

:   本文提出了一个三步评估框架（计算特征提取 + LLM-as-Judge + 人类专家验证）来系统评估六种 LLM 在唐诗生成上的能力，发现了关键的"回声室"效应：LLM 系统性地高估模仿统计模式但违反格律规则的机器生成诗歌，与人类专家判断显著偏离。

**[CAST: Achieving Stable LLM-based Text Analysis for Data Analytics](cast_achieving_stable_llm-based_text_analysis_for_data_analytics.md)**

:   提出CAST框架，通过算法提示（Algorithmic Prompting）和先思考后输出（Thinking-before-Speaking）两种机制约束LLM的潜在推理路径，显著提升文本摘要和标注任务的运行间稳定性，同时不损失输出质量。

**[Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)**

:   本文提出 TARS（Trajectory Alignment for Reasoning in Speech），一个基于强化学习的框架，通过表示对齐和行为对齐两种密集奖励信号，将语音条件下的推理轨迹与文本条件下的推理轨迹对齐，在 7B 规模模型中达到 SOTA，MRR（模态恢复率）接近甚至超过 100%。

**[Common to Whom? Regional Cultural Commonsense and LLM Bias in India](common_to_whom_regional_cultural_commonsense_and_llm_bias_in_india.md)**

:   本文构建 Indica，首个评估 LLM 次国家级文化常识的基准，聚焦印度五大区域在八个日常生活领域的文化差异，发现仅 39.4% 的问题在全部五个区域达成共识，且所有 LLM 均表现出地理偏见——过度选择中部和北部印度作为"默认"文化代表。

**[DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot Named Entity Recognition](diziner_disagreement-guided_instruction_refinement_via_pilot_annotation_simulati.md)**

:   DiZiNER 通过模拟人工标注中的"预标注"流程，利用多个异构 LLM 作为标注员、一个监督 LLM 分析模型间分歧并迭代优化任务指令，在18个NER基准上实现了14个数据集的零样本SOTA，平均提升+8.0 F1，且超越了作为监督者的GPT-5 mini。

**[Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff](do_llms_overthink_basic_math_reasoning_benchmarking_the_accuracy-efficiency_trad.md)**

:   本文提出 LLMThinkBench，一个系统性评估 LLM 基础数学推理效率的基准，引入 Overthinking Score（准确率和 token 效率的调和平均），通过动态生成的 14 个确定性数学任务评估 53 个 LLM，发现推理模型平均生成约 18× 更多 token 但有时准确率更低，且扩展推理预算呈现收益递减。

**[E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)**

:   本文提出 E2EDev，一个基于行为驱动开发（BDD）原则的端到端软件开发基准，包含 46 个真实 Web 项目、244 条细粒度需求和 703 个可执行 BDD 测试，评估发现即使最强 LLM（Claude 系列）在需求准确率上也不超过 60%，多智能体框架的复杂交互成本与性能收益不成正比。

**[Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks](enhancing_linguistic_competence_of_language_models_through_pre-training_with_lan.md)**

:   L2T 提出了一种预训练框架，将 14 种语言学习任务（字符级→篇章级）与标准 next-token prediction 混合训练，在 500M/1B 参数规模上将 BLiMP 语言能力得分提升 2-3 个百分点并加速其习得过程，同时保持通用推理性能。

**[Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language](exploring_the_capability_boundaries_of_llms_in_mastering_of_chinese_chouxiang_la.md)**

:   本文将中文互联网亚文化语言"抽象话"引入 NLP 社区，构建首个评估基准 Mouse（含翻译、表征分类、意图识别、毒性检测、含义选择、完形填空六个任务），发现 SOTA LLM 在上下文语义理解上表现尚可但在其他任务上存在明显局限。

**[Finch: Benchmarking Finance & Accounting across Spreadsheet-Centric Enterprise Workflows](finch_benchmarking_finance_amp_accounting_across_spreadsheet-centric_enterprise_.md)**

:   本文提出 Finch（FinWorkBench），一个从真实企业环境（Enron 数据集等）构建的金融会计工作流基准，包含 172 个复合工作流和 1,710 个电子表格（2700 万单元格），即使最强的 GPT 5.1 Pro 花费平均 16.8 分钟也仅通过 38.4% 的工作流，揭示了前沿 AI Agent 在真实企业场景中的严重不足。

**[From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning](from_domains_to_instances_dual-granularity_data_synthesis_for_llm_unlearning.md)**

:   本文形式化定义了领域级和实例级两种 LLM 遗忘粒度，提出 BiForget 框架——利用目标模型自身（而非外部强模型）通过种子引导合成和对抗探测两阶段生成高质量遗忘数据集，在 Harry Potter 领域将相关性提升约 20、多样性提升约 0.05 同时数据量减半。

**[HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents](higmem_a_hierarchical_and_llm-guided_memory_system_for_long-term_conversational_.md)**

:   本文提出 HiGMem，一个两层事件-对话轮记忆系统，通过让 LLM 先浏览事件摘要再预测哪些细粒度对话轮值得读取，在 LoCoMo10 基准上以少一个数量级的检索量达到了五类问题中四类的最优 F1。

**[Idiom Understanding as a Tool to Measure the Dialect Gap](idiom_understanding_as_a_tool_to_measure_the_dialect_gap.md)**

:   提出三个新的法语习语理解基准数据集（魁北克法语 QFrCoRE/QFrCoRT 和标准法语 MFrCoE），在 111 个 LLM 上评估发现 65.77% 的模型在方言习语上表现显著差于标准法语，量化了方言差距现象。

**[Language Model as Planner and Formalizer under Constraints](language_model_as_planner_and_formalizer_under_constraints.md)**

:   本文提出 CoPE 基准，通过向经典规划环境注入形式化分类的自然语言约束，揭示出仅一句约束即可将当前最强 LLM 的规划性能减半，暴露了 LLM 规划鲁棒性的严重不足。

**[LexRel: Benchmarking Legal Relation Extraction for Chinese Civil Cases](lexrel_benchmarking_legal_relation_extraction_for_chinese_civil_cases.md)**

:   构建了首个中国民事法律关系的结构化分类体系（9 大领域、265 种关系类型），并基于此提出 LexRel 基准（1,140 个专家标注样本），评估了主流 LLM 在法律关系抽取任务上的能力，发现当前模型在该任务上存在显著局限，同时证明了法律关系信息对下游法律 AI 任务的增益效果。

**[MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)**

:   本文提出 MADE——一个基于 FDA 医疗设备不良事件报告的"活"多标签文本分类基准，包含 1,154 个层次化标签和严格的时间分割，系统评估了 20+ 编码器/解码器模型在判别式微调、生成式微调和 few-shot 提示下的预测性能和不确定性量化（UQ）能力，揭示了关键权衡：小型判别式微调解码器在头到尾准确率上最优，生成式微调的 UQ 最可靠，大型推理模型提升稀有标签但 UQ 意外较弱。

**[Min-k Sampling: Decoupling Truncation from Temperature Scaling via Relative Logit Dynamics](min-k_sampling_decoupling_truncation_from_temperature_scaling_via_relative_logit.md)**

:   Min-k Sampling 通过分析排序 logit 分布的局部结构来检测"语义悬崖"（高置信候选与低质量尾部噪声的分界点），实现了严格的温度不变性截断，在极端温度下仍保持稳健的推理和创意写作质量。

**[Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding](modeling_multi-dimensional_cognitive_states_in_large_language_models_under_cogni.md)**

:   本文发现 LLM 在联合预测情感-思维风格-立场-意图四个认知维度时准确率暴跌至 5.7%（"认知拥挤"效应），通过 Gromov δ-hyperbolicity 分析证明认知状态具有层次结构，提出 HyCoLLM 框架在双曲空间中建模认知状态，8B 模型超越 GPT-4o。

**[MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models](mtr-duplexbench_towards_a_comprehensive_evaluation_of_multi-round_conversations_.md)**

:   提出 MTR-DuplexBench，一个针对全双工语音语言模型（FD-SLM）的多轮综合评估基准，通过创新的轮次分割方法解决了全双工对话中轮次边界模糊和上下文不一致的挑战，涵盖对话特性、对话质量、指令遵循和安全性四个维度，实验揭示了现有 FD-SLM 在多轮交互中性能持续衰退的问题。

**[MultiFileTest: A Multi-File-Level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms](multifiletest_a_multi-file-level_llm_unit_test_generation_benchmark_and_impact_o.md)**

:   提出 MultiFileTest，首个多文件级别 LLM 单元测试生成基准，覆盖 Python/Java/JavaScript 各 20 个项目，评估 11 个前沿 LLM 并分析手动修复和自修复机制对测试质量的影响，揭示即使最强模型也存在大量基础可执行性错误。

**[ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification](odutqa-mdc_a_task_for_open-domain_underspecified_tabular_qa_with_multi-turn_dial.md)**

:   本文提出 ODUTQA-MDC 任务和基准，首次系统研究开放域场景下用户查询模糊性的检测与多轮对话澄清问题，构建了包含 25,105 个 QA 对的大规模数据集，并设计了 MAIC-TQA 多智能体框架来完成"检测-澄清-推理"的端到端表格问答。

**[PIArena: A Platform for Prompt Injection Evaluation](piarena_a_platform_for_prompt_injection_evaluation.md)**

:   本文提出 PIArena，一个统一且可扩展的提示注入（Prompt Injection）评估平台，集成了多种 SOTA 攻击和防御方法，支持即插即用评估，并设计了基于策略的自适应攻击方法，系统性地揭示了现有防御在泛化性、自适应攻击和任务对齐场景下的关键局限。

**[ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](researchbench_benchmarking_llms_in_scientific_discovery_via_inspiration-based_ta.md)**

:   提出 ResearchBench，首个大规模评估LLM科学发现能力的基准，基于"灵感驱动假设生成"的理论分解，覆盖12个学科1386篇论文，将科学发现分解为灵感检索、假设组合、假设排序三个充分子任务，发现LLM在跨学科灵感检索上表现出色。

**[Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal Fine-grained Automatic Meeting Effectiveness Evaluation](rethinking_meeting_effectiveness_a_benchmark_and_framework_for_temporal_fine-gra.md)**

:   本文重新定义会议效率评估——提出"目标达成率/时间成本"的客观标准和时序细粒度评估范式，构建了包含 130 场会议 2,459 个标注片段的 AMI-ME 数据集，并开发了基于 LLM 的自动评估框架，在 Spearman 相关系数上达到 0.64。

**[ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)**

:   本文提出 ReTraceQA，首个面向常识推理任务的推理过程评测基准，包含 2421 条由专家标注的步骤级错误定位和错误分类标注，揭示 14-24% 的 SLM 虽给出正确答案但推理过程有误，当采用推理感知评估替代仅答案评估时，SLM 性能最多下降 25 个百分点。

**[Revisiting the Uniform Information Density Hypothesis in LLM Reasoning](revisiting_the_uniform_information_density_hypothesis_in_llm_reasoning.md)**

:   本文将心理语言学中的信息密度均匀性（UID）假说引入 LLM 推理分析，提出基于熵的步级信息密度度量框架，发现高质量推理轨迹呈现"局部均匀 + 全局非均匀"的反直觉模式，并证明该模式在 Best-of-N 采样中显著优于传统置信度/熵基线。

**[RoleConflictBench: A Benchmark of Role Conflict Scenarios for Evaluating LLMs' Contextual Sensitivity](roleconflictbench_a_benchmark_of_role_conflict_scenarios_for_evaluating_llms39_c.md)**

:   RoleConflictBench 通过构建 13,914 个角色冲突场景，利用情境紧迫性作为客观约束来评估 LLM 的上下文敏感性，揭示了模型决策被静态角色偏好主导而非响应动态情境线索的严重问题。

**[SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)**

:   本文构建 SciImpact——首个跨 19 个学科领域、涵盖 7 个影响力维度（引用、奖项、专利、媒体、代码、数据集、模型）的大规模科学影响力预测基准，包含 215,928 个对比论文对，通过多任务微调使 4B 模型超越 o4-mini 等大模型。

**[Self-Awareness before Action: Mitigating Logical Inertia via Proactive Cognitive Awareness](self-awareness_before_action_mitigating_logical_inertia_via_proactive_cognitive_.md)**

:   本文提出 SABA 推理框架，通过"先感知再行动"的范式，在做出最终决策前显式构建和审计知识状态——利用信息融合 (IF) 将叙事整合为可验证的基线状态，再通过查询驱动的结构化推理 (QSR) 递归识别和解决缺失前提——在侦探推理和通用推理基准上均取得最佳表现。

**[SessionIntentBench: A Multi-Task Inter-Session Intention-Shift Modeling Benchmark](sessionintentbench_a_multi-task_inter-session_intention-shift_modeling_benchmark.md)**

:   本文提出 SessionIntentBench，一个评估 L(V)LM 理解电商购物会话中跨步骤意图漂移能力的多任务基准，包含四个递进式子任务（意图购买似然估计、属性正则化、意图验证对比、意图演化建模），构建了 190 万条意图条目和 113 万条意图轨迹，实验表明当前 20+ 个 L(V)LM 在捕获复杂会话意图方面表现不佳。

**[Subject-level Inference for Realistic Text Anonymization Evaluation](subject-level_inference_for_realistic_text_anonymization_evaluation.md)**

:   SPIA 提出首个主体级 PII 推断评估基准（675 篇文档、1712 个主体、7040 个 PII），揭示即使 90%+ 的 PII 片段被遮蔽，主体级推断保护率可低至 33%，且聚焦单一目标主体的匿名化会导致非目标主体暴露更多。

**[Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](task-aware_llm_routing_with_multi-level_task-profile-guided_data_synthesis_for_c.md)**

:   提出多层级任务画像引导的数据合成框架解决 LLM 路由的冷启动问题，并设计 TRouter——一种将任务类型作为隐变量的路由方法，通过变分推断建模查询-成本-性能关系，在冷启动和域内设置下均实现有效路由。

**[Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context](text-to-distribution_prediction_with_quantile_tokens_and_neighbor_context.md)**

:   本文提出Quantile Token Regression方法，通过在输入序列中插入专用分位数token并结合检索到的邻居实例及其经验分布，使LLM能够预测完整的条件分布而非单一点估计，在Airbnb和StackSample数据集上相比基线降低约4个MAPE点并将预测区间收窄2倍以上。

**[Think in Latent Thoughts: A New Paradigm for Gloss-Free Sign Language Translation](think_in_latent_thoughts_a_new_paradigm_for_gloss-free_sign_language_translation.md)**

:   提出 SignThought，一种推理驱动的无注释手语翻译框架：引入可学习的潜在思维槽作为视频和文本之间的显式中间语义层，通过"先规划后定位"的双流解码器实现语义规划与视觉证据检索的解耦，在多个基准上超越现有无注释方法。
