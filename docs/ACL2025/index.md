---
title: >-
  1855 篇 ACL2025 论文解读 · 每篇 5 分钟读懂
description: >-
  1855篇ACL2025论文解读，涵盖 LLM / NLP(389篇)、多模态 VLM(132篇)、多语言/翻译(89篇)、LLM 评测(88篇)、信息检索/RAG(85篇)、对齐 / RLHF(75篇)、LLM 安全(70篇)、模型压缩(67篇)等 45个方向。每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ACL2025"
  - "AI顶会"
  - "论文解读"
  - "论文笔记"
  - "LLM / NLP"
  - "多模态 VLM"
  - "多语言/翻译"
  - "LLM 评测"
  - "信息检索/RAG"
  - "对齐 / RLHF"
  - "LLM 安全"
  - "模型压缩"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 ACL2025 论文笔记

1855篇ACL2025论文解读，涵盖 LLM / NLP(389篇)、多模态 VLM(132篇)、多语言/翻译(89篇)、LLM 评测(88篇)、信息检索/RAG(85篇)、对齐 / RLHF(75篇)、LLM 安全(70篇)、模型压缩(67篇)等 45个方向。每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。

<div class="conf-index" markdown>

---

## 💬 LLM / NLP { #llm_nlp }

**[A Large-Scale Real-World Evaluation of an LLM-Based Virtual Teaching Assistant](llm_nlp/a_large-scale_real-world_evaluation_of_llm-based_virtual_teaching_assistant.md)**

:   在韩国KAIST一门477人研究生AI编程课中部署基于RAG的LLM虚拟助教(VTA)，通过三轮问卷(472人)和3869条交互日志的纵向分析，发现VTA显著降低了学生提问心理门槛，高频用户的满意度随使用持续提升，但信任度仍低于人类助教。

**[A Modular Dataset to Demonstrate LLM Abstraction Capability](llm_nlp/a_modular_dataset_to_demonstrate_llm_abstraction_capability.md)**

:   提出ArrangementPuzzle拼图数据集并训练LLM激活值分类器，发现分类器以>80%准确率识别推理正确性，揭示LLM在中间-后层Transformer层编码了区分逻辑等价与语义等价的抽象推理概念。

**[A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models](llm_nlp/a_semantic-aware_layer-freezing_approach_to_computation-efficient_fine-tuning_of.md)**

:   通过分析LLM推理过程中潜在表征的转移轨迹（transition traces）计算各层语义偏差，结合推导的缩放律公式估计各层对降低损失的贡献，从而确定"在哪些层微调"，实现与PEFT正交的高效微调方法。

**[A Survey of Automatic Prompt Optimization with Instruction-focused Heuristic-based Search Algorithm](llm_nlp/a_survey_of_automatic_prompt_optimization_with_instruction-focused_heuristic-bas.md)**

:   系统综述 80+ 种基于启发式搜索算法的自动 Prompt 优化方法，提出五维分类体系（Where/What/What criteria/Which operators/Which algorithms）将碎片化研究统一到一个完整的分析框架下。

**[A Survey of Large Language Models in Psychotherapy: Current Landscape and Future Directions](llm_nlp/a_survey_of_large_language_models_in_psychotherapy_current_landscape_and_future_.md)**

:   首篇以 APA 三阶段（评估→诊断→治疗）概念分类法系统梳理 LLM 心理治疗研究的综述，覆盖 60+ 篇工作，从症状检测到虚拟治疗师四层面全面分析，揭示障碍覆盖、语言偏差、方法碎片化和理论整合的四重失衡。

**[A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?](llm_nlp/a_survey_of_llm-based_agents_in_medicine_how_far_are_we_from_baymax.md)**

:   系统综述 LLM-based Agent 在医学中的四层架构（Profile/临床规划/医学推理/外部能力增强）、四大应用场景和评估框架，覆盖 2022-2024 年 60 篇研究，提出四种 Agent 运作范式并识别幻觉管理、多模态整合和伦理等关键挑战。

**[A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](llm_nlp/a_survey_on_efficient_large_language.md)**

:   本文提出首个系统性的"数据高效 LLM 后训练"综述框架，将方法分为数据选择、数据质量增强、合成数据生成、数据蒸馏与压缩、自演化数据生态五大类，构建了完整的"数据价值飞轮"体系。

**[A Systematic Study of Compositional Syntactic Transformer Language Models](llm_nlp/a_systematic_study_of_compositional_syntactic_transformer_language_models.md)**

:   本文提出了一个统一框架，系统性地研究组合句法Transformer语言模型（SLM）的四个关键设计维度（树的形式、线性化策略、组合函数、子成分遮掩），涵盖了已有模型和13个新变体，并通过语言建模、句法泛化、摘要、对话和推理效率五个维度的全方位实验，得出了SLM设计的多条推荐建议。

**[A Training-free LLM-based Approach to General Chinese Character Error Correction](llm_nlp/a_training-free_llm-based_approach_to_general_chinese_character_error_correction.md)**

:   提出通用中文字符纠错任务C2EC（覆盖替换、缺失、冗余三种错误类型），通过扩展训练无关的CSC方法并结合Levenshtein距离和prompt-based LLM，使14B参数模型在不微调的条件下达到近50倍大模型的纠错性能。

**[AAD-LLM: Neural Attention-Driven Auditory Scene Understanding](llm_nlp/aad-llm_neural_attention-driven_auditory_scene_understanding.md)**

:   提出意图感知听觉场景理解（II-ASU）范式和 AAD-LLM 原型系统——通过颅内脑电（iEEG）解码听者正在关注哪个说话人，将注意力状态注入听觉 LLM，使模型在多说话人场景中生成与听者感知对齐的回答。

**[Acquisition and Application of Novel Knowledge in Large Language Models](llm_nlp/acquisition_and_application_of_novel_knowledge_in_large_language_models.md)**

:   本文提出 PermAR 框架，通过排列式语言建模赋予自回归模型双向知识获取能力，同时构建了基于知识图谱生物进化模拟的新知识数据集 NovelHuman，发现知识在句子中的位置显著影响 LLM 的知识获取效果，PermAR 在新知识注入任务上比现有方法提升 3.3%-38%。

**[AfroBench: How Good are Large Language Models on African Languages?](llm_nlp/afrobench_how_good_are_large_language_models_on_african_languages.md)**

:   提出AfroBench——覆盖64种非洲语言、15个NLP任务、22个数据集的综合评测基准，评估12个LLM发现闭源模型(GPT-4o)领先最佳开源模型(Gemma 2 27B)约12分，但所有LLM仍落后于微调基线，与英语的差距在开源模型上超过40分。

**[AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration](llm_nlp/agentdropout-dynamic-agent-elimination-for-multi-agent-collaboration.md)**

:   本文提出 AgentDropout，通过在多轮讨论中动态消除冗余 Agent（节点剪枝）和冗余通信边（边剪枝），在降低 21.6% prompt token 消耗的同时提升了 1.14 分的任务性能。

**[AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration](llm_nlp/agentdropout_dynamic_agent_elimination_for_token-efficient_and_high-performance_.md)**

:   本文提出AgentDropout，通过在每轮通信中动态删除冗余Agent节点和通信边来优化多智能体系统的通信拓扑，相比SOTA方法平均减少21.6%的prompt token消耗和18.4%的completion token消耗，同时性能提升1.14分。

**[AgentGym: Evolving Large Language Model-based Agents across Diverse Environments](llm_nlp/agentgym_evaluating_and_training_large_language_model-based_agents_across_divers.md)**

:   本文提出AgentGym框架，包含14种交互环境、89类任务、标准化轨迹数据集和评测基准，并提出AgentEvol自我进化算法，让LLM Agent通过跨环境探索和学习实现从模仿到自主进化，性能可达到SOTA模型水平。

**[AI as a Novel Ethical Agent: Exploring Moral Judgments by Large Language Models](llm_nlp/ai_as_a_novel_ethical_agent_exploring_moral_judgments_by_large_language_models.md)**

:   本文系统地探索了大语言模型作为新型伦理主体的道德判断能力，通过构建覆盖多种伦理框架的测试基准，揭示了LLM在道德推理中的偏好模式、一致性缺陷和文化偏差。

**[AIMSCheck: Leveraging LLMs for AI-Assisted Review of Modern Slavery Statements Across Jurisdictions](llm_nlp/aimscheck_modern_slavery.md)**

:   提出 AIMSCheck——一个端到端的企业现代奴隶制声明合规评估框架，将评估任务分解为句子级多标签分类、token 级 SHAP 解释和证据状态追踪三个层级，同时构建英国和加拿大两个新标注数据集，验证了在澳大利亚数据上微调的模型能有效跨司法管辖区泛化。

**[Algorithmic Fidelity of Large Language Models in Generating Synthetic German Public Opinions: A Case Study](llm_nlp/algorithmic_fidelity_german_opinion.md)**

:   基于德国纵向选举调查(GLES)的开放式问题数据，系统评估三个开源LLM（Llama2、Gemma、Mixtral）通过人口统计persona提示生成合成德国公众舆论的算法保真度，发现Llama2在亚群体代表性上表现最佳（JS距离0.28），但所有模型均表现出左倾政治偏见和群体内多样性降低的问题。

**[Alignment Drift in CEFR-prompted LLMs for Interactive Spanish Tutoring](llm_nlp/alignment_drift_in_cefr-prompted_llms_for_interactive_spanish_tutoring.md)**

:   通过 LLM 模拟师生对话实验，发现基于 CEFR 等级的 system prompting 虽然能初步约束 LLM 输出的西班牙语难度，但随着对话轮次增加，这种约束效果逐渐衰减——作者将此现象命名为"alignment drift"，表明仅靠提示工程不足以支撑长期的自适应语言教学。

**[An Empirical Study of Iterative Refinements for Non-Autoregressive Translation](llm_nlp/an_empirical_study_of_iterative_refinements_for_non-autoregressive_translation.md)**

:   本文对非自回归翻译（NAT）中的迭代精修方法进行了系统性的实证研究，比较了不同精修策略（如CMLM、DisCo、SUNDAE等）在翻译质量与推理速度之间的权衡，揭示了迭代次数、掩码比例和训练策略对最终性能的关键影响，为NAT研究提供了全面的实践指导。

**[An Empirical Study of Large Language Models for Automated Review Generation](llm_nlp/an_empirical_study_of_large_language_models_for_automated_review_generation.md)**

:   本文是一项系统性实证研究，评估了多种大语言模型在自动生成学术论文同行评审方面的能力，分析了生成评审的质量、一致性和实用性，并揭示了当前LLM在评审生成任务上的优势、不足和改进方向。

**[Analyzing and Mitigating Inconsistency in Discrete Speech Tokens for Neural Codec Language Models](llm_nlp/analyzing_and_mitigating_inconsistency_in_discrete_speech_tokens_for_neural_code.md)**

:   本文揭示了神经音频编解码器（如EnCodec）中离散表示不一致性（DRI）现象——相同音频片段在有无上下文时会被编码为不同的token序列，并提出切片一致性和扰动一致性两种约束方法，将一致性提升21-36%，在VALL-E语音生成中实现3.72% WER降低和5.68%说话人相似度提升。

**[Analyzing the Rapid Generalization of SFT via the Perspective of Attention Head Activation Patterns](llm_nlp/analyzing_the_rapid_generalization_of_sft_via_the_perspective_of_attention_head_.md)**

:   本文通过基于梯度的注意力头激活模式分析，揭示了SFT使LLM快速适应下游任务的三个关键机制：选择性激活任务特定的注意力头、复杂任务的激活模式是基础任务的线性组合、少量样本即可显著改变激活模式，并据此提出了利用基础任务数据促进复杂任务学习的实用策略。

**[APPL: A Prompt Programming Language for Harmonious Integration of Programs and Large Language Model Prompts](llm_nlp/appl_a_prompt_programming_language_for_harmonious_integration_of_programs_and_la.md)**

:   本文提出APPL——一种将LLM提示（prompt）无缝嵌入Python程序的提示编程语言，提供Python原生语法、异步并行运行时和可追溯调试模块，简化了复杂LLM工作流的开发与维护。

**[Revisiting Common Assumptions about Arabic Dialects in NLP](llm_nlp/arabic_dialects_assumptions_revisited.md)**

:   系统性检验了阿拉伯语方言 NLP 中四个被广泛接受但未被量化验证的假设，通过扩展 NADI 2024 数据集（11 个国家级方言、33 名标注者）发现这些假设过度简化了现实，56% 的方言句子跨区域有效、ADI 应建模为多标签分类任务。

**[Are Optimal Algorithms Still Optimal? Rethinking Sorting in LLM-Based Pairwise Ranking with Batching and Caching](llm_nlp/are_optimal_algorithms_still_optimal_rethinking_sorting_in_llm-based_pairwise_ra.md)**

:   本文重新审视了LLM成对排序提示（PRP）中排序算法的选择问题，提出以LLM推理调用次数（而非比较次数）为核心代价模型，发现经典最优算法Heapsort在引入批处理（batching）和缓存（caching）优化后不再最优，Quicksort在batch size≥2时推理次数减少44%，为PRP排序提供了新的最优选择。

**[Are Your LLMs Capable of Stable Reasoning?](llm_nlp/are_your_llms_capable_of_stable_reasoning.md)**

:   提出 G-Pass@k 评估指标和 LiveMathBench 动态基准，从"性能上限"和"稳定性"两个维度全面评估LLM的推理能力，揭示了当前LLM在推理一致性上存在巨大提升空间。

**[Argument Mining in the Age of Large Language Models](llm_nlp/argument_mining_in_the_age_of_large_language_models.md)**

:   本文系统性地研究了大语言模型时代下论辩挖掘（Argument Mining）任务的现状与挑战，通过全面的实验评估了LLM在论证组件识别、论证关系分类、论证质量评估等子任务上的表现，并提出了针对性的改进策略，揭示了LLM在结构化论辩理解方面的优势与不足。

**[ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving](llm_nlp/arithmattack_evaluating_robustness_of_llms_to_noisy_context_in_math_problem_solv.md)**

:   提出 ArithmAttack，通过在数学题上下文中随机插入标点符号（不改变任何单词）来测试 LLM 的鲁棒性，发现八个主流 LLM（包括 Llama3、Mistral、DeepSeek）在面对这种简单噪声时性能都显著下降。

**[Assessing and Enhancing the Causal Reasoning Abilities of Language Models via Faithful Textual Interpretation](llm_nlp/assessing_and_enhancing_the_causal_reasoning_abilities_of_language_models_via_fai.md)**

:   本文提出了一个基于忠实文本解释（Faithful Textual Interpretation, FTI）的框架，通过将因果推理任务中的变量关系忠实地转化为自然语言描述，评估并增强LLM的因果推理能力，在多个因果推理基准上取得了显著提升。

**[Assessing the Vulnerability of LLMs to Cognitive Biases in Scientific Research](llm_nlp/assessing_the_vulnerability_of_llms_to_cognitive_biases_in_scientific_research.md)**

:   本文系统性地评估了大语言模型在科学研究场景中对多种认知偏差的脆弱性，通过构建涵盖确认偏差、锚定效应、可得性偏差等的科学推理测试集，揭示了LLM在辅助科学研究时可能引入的系统性偏差风险，并提出了缓解策略。

**[ATRIE: Automating Legal Interpretation with LLMs: Retrieval, Generation, and Evaluation](llm_nlp/atrie_legal_interpretation.md)**

:   提出 ATRIE 框架，模拟法学专家的教义法学研究流程，利用 LLM 自动从案例库中检索相关信息、生成法律概念解释并评估解释质量，消除对人类法律专家的依赖。

**[Attention Speaks Volumes: Localizing and Mitigating Bias in Language Models](llm_nlp/attention_speaks_volumes_localizing_and_mitigating_bias_in_language_models.md)**

:   提出Atlas（Attention-based Targeted Layer Analysis and Scaling），通过分析注意力分数定位LLM中偏见集中的层，然后在这些层进行推理时注意力缩放干预来缓解偏见，在BBQ、CrowS-Pairs和WinoGender三个数据集、四个模型上有效降低偏见，且仅增加0.82%的困惑度。

**[Attribution Methods in NLP: Navigating a Fragmented Landscape](llm_nlp/attribution_methods_in_nlp_navigating_a_fragmented_landscape.md)**

:   本文对NLP领域的归因方法（Attribution Methods）进行了全面的综述和系统性比较，针对该领域评估标准碎片化、方法间缺乏公平对比的问题，提出了统一的评估框架，并揭示了不同归因方法在不同任务和模型架构上的适用性规律。

**[AutoExp: Automatic Experiment Design and Execution by LLMs](llm_nlp/autoexp_automatic_experiment_design_and_execution_by_llms.md)**

:   本文提出AutoExp框架，利用LLM作为智能代理自动完成NLP实验的全流程——从研究问题分析、实验方案设计、代码生成执行到结果分析解读，在多个标准NLP研究场景中展示了LLM自动化科研实验的可行性与局限性。

**[AutoGUI: Scaling GUI Grounding with Automatic Functionality Annotations from LLMs](llm_nlp/autogui_scaling_gui_grounding_with_automatic.md)**

:   提出AutoGUI自动标注管线——通过模拟交互比较UI状态变化+LLM推断元素功能+双LLM验证过滤，构建704K高质量UI功能标注数据集，标注正确率96.7%可比人类，显著提升VLM的UI grounding能力且展现数据扩展效应。

**[LLM-AT: Automatic Transmission for LLM Tiers Optimizing Cost and Accuracy](llm_nlp/automatic_transmission_for_llm_tiers_optimizing_cost_and_accuracy_in_large_langu.md)**

:   提出 LLM-AT 框架，通过 Starter（基于历史推理记录的准确率估计器选择初始 LLM 层级）→ Generator（生成回答）→ Judge（评估有效性，无效则自动升级到更高层级）的无训练迭代流程，在 MATH 上以 o1 单次推理 59.37% 的成本达到接近的准确率，在 MCQA 上以 o1 成本的 12% 实现近似性能。

**[Awes, Laws, and Flaws From Today's LLM Research](llm_nlp/awes_laws_and_flaws_from_todays_llm_research.md)**

:   对引用 GPT-3/GPT-4 的 2,054 篇 LLM 研究论文（2020-2024）进行 14 维标注与统计分析，揭示领域存在系统性方法论退化——仅 25% 论文含统计检验、伦理声明比例持续下降、LLM 评估器急增 15% 但缺乏元评估——同时用数据验证了会议强制检查清单（如 ACL 的 limitations 要求）的确有效遏制退化趋势。

**[AXIS: Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents](llm_nlp/axis_efficient_human-agent-computer_interaction_with_api-first_llm-based_agents.md)**

:   提出 AXIS 框架，通过让 LLM Agent 优先调用 API 而非模拟人类 UI 操作来完成应用任务，在 Microsoft Word 实验中将任务完成时间缩短 65-70%，认知负荷降低 38-53%，同时保持 97-98% 的准确率。

**[OPTS: Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](llm_nlp/bandit-based_prompt_design_strategy_selection_improves_prompt_optimizers.md)**

:   首次提出 prompt 设计策略的显式选择机制 OPTS，将 11 种策略（CoT、角色提示、情感提示等）建模为多臂老虎机的臂，用 Thompson 采样自动选择最合适的策略并集成到 EvoPrompt 优化器中，在 BIG-Bench Hard 的 23 个任务上用 GPT-4o mini 实现最高 50% 的性能提升。

**[Binary Classifier Optimization for Large Language Model Alignment](llm_nlp/bco_binary_classifier_alignment.md)**

:   提出 BCO（Binary Classifier Optimization），从数学上证明二元交叉熵损失是 DPO 损失的上界，使 LLM 对齐仅需"点赞/踩"二元反馈而非成对偏好数据，并通过新颖的 reward shift 技术收紧上界，在配对偏好数据集上与 DPO 持平，在真实 Likert-5 标注数据上优于 DPO 和 KTO。

**[Behavioral Analysis of Information Salience in Large Language Models](llm_nlp/behavioral_analysis_of_information_salience_in_large_language_models.md)**

:   提出一个可解释的分析框架，通过长度受控的摘要生成行为探针和 Questions Under Discussion（QUD）的可回答性追踪，系统推导并研究 LLM 内化的信息显著性（salience）概念，发现 LLM 具有层次化的一致显著性观念，但这一观念无法通过内省获取且与人类感知仅弱相关。

**[BehaviorBox: Automated Discovery of Fine-Grained Performance Differences Between Language Models](llm_nlp/behaviorbox_automated_discovery_of_fine-grained_performance_differences_between_.md)**

:   提出 BehaviorBox，利用性能感知的上下文嵌入（performance-aware contextual embeddings）自动发现两个语言模型之间细粒度的性能差异特征，如"条件语气中的'were'"或"情感句后的感叹号"等具体上下文模式。

**[Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals' Subjective Text Perceptions](llm_nlp/beyond_demographics_fine-tuning_large_language_models_to_predict_individuals_sub.md)**

:   本文系统研究了 LLM 能否通过社会人口学属性（年龄、性别、种族、教育）来预测个体标注者的主观文本感知，发现微调后的改进主要来自学习个体标注者行为而非社会人口学模式，质疑了用 LLM 模拟社会人口学差异的可行性。

**[Beyond Dialogue: A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model](llm_nlp/beyond_dialogue_a_profile-dialogue_alignment_framework_towards_general_role-play.md)**

:   提出 Beyond Dialogue 框架，通过 Profile-Dialogue 对齐消除角色扮演训练中 profile 与对话之间的偏差，并引入句子级细粒度对齐任务，使模型更好地理解和表现角色特质。

**[Beyond In-Context Learning: Aligning Long-form Generation of LLMs via Task-Inherent Attribute Guidelines](llm_nlp/beyond_in-context_learning_aligning_long-form_generation_of_large_language_model.md)**

:   从理论和实验两方面证明 ICL 示例无法充分传递任务的语言和格式属性，提出 LongGuide 算法从少量训练数据中自动学习 Metric Guideline (MG) 和 Output Constraint Guideline (OCG) 两类指导规则，在 7 个长文本生成任务上平均提升超过 5% ROUGE-L。

**[Beyond Output Matching: Bidirectional Alignment for Enhanced In-Context Learning](llm_nlp/beyond_output_matching_bidirectional_alignment_for_enhanced_in-context_learning.md)**

:   提出 Bidirectional Alignment (BiAlign)，在传统知识蒸馏仅对齐输出分布的基础上，新增输入偏好对齐——通过 ranking loss 让学生模型学习教师模型对不同 ICL 示例的偏好排序，在语言理解、推理和代码 5 个任务上一致优于基线，GSM8K 提升 20%、LogiQA 提升 18%。

**[Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs](llm_nlp/beyond_profile_from_surface-level_facts_to_deep_persona_simulation_in_llms.md)**

:   提出 CharacterBot，通过 4 个训练任务（视角重建预训练 + 选择题/生成式QA/风格迁移微调）和 CharLoRA 参数更新机制，从鲁迅 17 部杂文集中学习其语言风格和深层思想模式，在语言准确性和观点理解上显著超越各基线。

**[Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms](llm_nlp/beyond_prompt_engineering_robust_behavior_control_in_llms_via_steering_target_at.md)**

:   提出 STA（Steering Target Atoms），利用稀疏自编码器 (SAE) 将 LLM 的表示解耦为原子知识组件，通过激活幅度和频率筛选目标原子并操控，实现比提示工程更鲁棒、更精细的行为控制，在安全解毒和推理控制任务上效果优于现有 steering 方法。

**[BFS-Prover: Scalable Best-First Tree Search for LLM-Based Automatic Theorem Proving](llm_nlp/bfs-prover_scalable_best-first_tree_search_for_llm-based_automatic_theorem_provi.md)**

:   本文挑战了"自动定理证明需要复杂搜索方法（如MCTS或价值函数）"的传统认知，提出BFS-Prover系统，通过三项关键创新（数据过滤的专家迭代、基于编译器反馈的DPO、长度归一化）将简单的最佳优先搜索（BFS）扩展为高性能的定理证明器，在MiniF2F测试集上达到72.95%的SOTA成绩。

**[Bias in Language Models: Beyond Trick Tests and Towards RUTEd Evaluation](llm_nlp/bias_in_language_models_beyond_trick_tests_ruted_evaluation.md)**

:   通过对比标准偏见基准（"trick tests"）与基于真实使用场景的 RUTEd 评估，发现标准偏见基准与真实场景中的偏见表现无显著相关性，主张偏见评估应面向具体应用场景。

**[Biased LLMs Can Influence Political Decision-Making](llm_nlp/biased_llms_can_influence_political_decision-making.md)**

:   通过两项大规模交互实验（N=299），本文首次实证证明带有党派偏见的LLM可以显著影响人类的政治观点和预算分配决策，且这种影响甚至跨越党派——民主党人会被保守偏见的LLM说服，共和党人也会被自由偏见的LLM影响。

**[Big5-Chat: Shaping LLM Personalities Through Training on Human-Grounded Data](llm_nlp/big5-chat_shaping_llm_personalities_through_training_on_human-grounded_data.md)**

:   提出了 Big5-Chat 数据集（10万条对话），通过 SFT 和 DPO 训练方法将真实人类大五人格特质嵌入 LLM，效果显著优于基于提示的方法，且发现高尽责性/宜人性、低外向性/神经质的人格配置能提升模型推理能力。

**[Bilingual Zero-Shot Stance Detection](llm_nlp/bilingual_zero-shot_stance_detection.md)**

:   本文针对零样本立场检测任务的跨语言挑战，提出一种双语联合框架，通过共享语义空间的构建和跨语言知识迁移，实现在目标语言上未见过标注数据的情况下准确判断文本对特定话题的立场（支持/反对/中立）。

**[BIPro: Zero-shot Chinese Poem Generation via Block Inverse Prompting Constrained Generation Framework](llm_nlp/bipro_zero-shot_chinese_poem_generation_via_block_inverse_prompting_constrained_.md)**

:   提出 BIPro 框架，利用块生成模型（Block Generative Model）的中间文本生成能力，通过"修订（revise）"和"重写（rewrite）"两种块逆提示方法，在无需领域特定训练的情况下使弱模型 GLM-10B 在开放式传统中国诗歌生成任务中超越 GPT-4 和最佳专用系统。

**[Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](llm_nlp/bitnetcpp_efficient_edge_inference_for_ternary_llms.md)**

:   本文提出Bitnet.cpp推理系统，通过两种创新的混合精度矩阵乘法核心——基于元素级查找表的TL和基于Int2+Scale的I2_S——实现了三值LLM（如BitNet b1.58）在边缘设备上的高效无损推理，相比全精度基线加速最高6.25倍，相比低比特基线加速最高2.32倍。

**[Boosting LLM's Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning](llm_nlp/boosting_llms_molecular_structure_elucidation_with_knowledge_enhanced_tree_searc.md)**

:   提出 K-MSE（Knowledge-enhanced Molecular Structure Elucidation）框架，构建分子子结构知识库扩展 LLM 的化学结构空间覆盖，设计专用分子-光谱打分器替代 LLM 自身评估，结合蒙特卡洛树搜索（MCTS）实现测试时推理缩放，在 MolPuzzle 基准上分别将 GPT-4o-mini 和 GPT-4o 的准确率从 3.7% 和 27.8% 提升至 27.3% 和 39.8%。

**[Brevity is the soul of sustainability: Characterizing LLM response lengths](llm_nlp/brevity_is_the_soul_of_sustainability_characterizing_llm_response_lengths.md)**

:   系统研究 12 个 LLM 在 5 个数据集上的响应长度行为，发现 LLM 普遍生成远超必要的冗长回复（核心答案仅占 42%），并提出多种提示策略将响应长度缩短 25-88%、推理能耗降低 25-60%，同时保持甚至提升 ROUGE-L F1 质量。

**[BRIGHTER: BRIdging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages](llm_nlp/brighter_bridging_the_gap_in_human-annotated_textual_emotion_recognition_dataset.md)**

:   本文构建了覆盖28种语言的多标签情感标注数据集BRIGHTER，重点覆盖非洲、亚洲、东欧和拉美的低资源语言，由母语使用者标注，并在单语和跨语言情感识别任务上建立了基准实验结果。

**[Can Large Language Models Understand Internet Buzzwords Through User-Generated Content](llm_nlp/buzzword_understanding_ugc.md)**

:   本文构建了首个中文网络流行语数据集 Cheer（1127 条），并提出 Ress 方法——通过模拟儿童语言习得的六维理解技能引导 LLM 从用户生成内容中产出更准确的流行语定义，在语义准确度上平均提升 2.51%。

**[C²LEVA: Toward Comprehensive and Contamination-Free Language Model Evaluation](llm_nlp/c2leva_toward_comprehensive_and_contamination-free_language_model_evaluation.md)**

:   提出 C²LEVA，一个包含22个任务的中英双语综合评测基准，通过全自动的测试数据更新和数据保护机制来系统性地防止数据污染，在15个开源和闭源模型上验证了其有效性。

**[Automated CAD Modeling Sequence Generation from Text Descriptions via Transformer-Based Large Language Models](llm_nlp/cadllm_cad_modeling_from_text.md)**

:   本文提出了一个从文本描述自动生成 CAD 建模序列的框架，包含半自动标注流水线、双通道 Transformer 生成器 TCADGen 和 LLM 增强模块 CADLLM，最终将 CAD 命令准确率从 84% 提升到 96.6%，Chamfer Distance 从 120.99 降至 3.12。

**[Can Input Attributions Explain Inductive Reasoning in In-Context Learning?](llm_nlp/can_input_attributions_explain_inductive_reasoning_in_in-context_learning.md)**

:   设计受控的合成归纳推理任务评估 4 种输入归因方法解释 ICL 的能力，发现最简单的梯度范数常常最好，但所有方法在不同任务和模型规模上表现不一致且不稳定——ICL 的可解释性比预期更难。

**[Can Language Models Reason about Individualistic Human Values and Preferences?](llm_nlp/can_language_models_reason_about_individualistic_human_values_and_preferences.md)**

:   提出"个体价值对齐"范式，构建 IndieValueCatalog 数据集（基于世界价值观调查 WVS 的 9.3 万真人数据），评估并训练语言模型根据个人价值表达陈述推理其在新情境下的价值判断，揭示前沿 LM 仅达 55%-65% 准确率且存在显著的跨群体不公平性。

**[Can Language Models Replace Programmers for Coding? RepoCod Says 'Not Yet'](llm_nlp/can_language_models_replace_programmers_for_coding_repocod_says_not_yet.md)**

:   构建了 RepoCod——一个包含980个来自11个大型 Python 项目的复杂代码生成任务的基准，具有真实的仓库级依赖和平均314个开发者测试用例，揭示了即使最先进的 LLM 也仅能达到不到30%的 Pass@1，远未能替代程序员完成真实编码任务。

**[Can Large Language Models Accurately Generate Answer Keys for Health-related Questions?](llm_nlp/can_large_language_models_accurately_generate_answer_keys_for_health-related_que.md)**

:   本文探索使用LLM自动生成医学问答的答案关键点（information nuggets），对比多种生成方法与人类专家标注的一致性，发现提供示例+从答案中提取nugget的方法效果最好，但LLM提取原子事实的能力仍然有限，其中Llama 3.3表现最佳。

**[Can Large Language Models Address Open-Target Stance Detection?](llm_nlp/can_large_language_models_address_open-target_stance_detection.md)**

:   提出开放目标立场检测（OTSD）任务——目标在训练时未见且不作为输入提供，系统评估了 GPT、Gemini、LLaMA、Mistral 四个系列共 8 个 LLM 在目标生成和立场检测两阶段的表现，发现 LLM 整体优于现有 TSE 方法，但在目标未显式出现时表现明显下降。

**[Can LLMs Ground when they (Don't) Know: A Study on Direct and Loaded Political Questions](llm_nlp/can_llms_ground_when_they_dont_know_a_study_on_direct_and_loaded_political_quest.md)**

:   本文研究LLM在政治领域中处理直接知识问题和暗含错误预设的"loaded question"的能力，评估LLM是否能进行主动的对话基础（conversational grounding）来纠正用户的虚假信念，发现LLM在拒绝虚假预设和维护知识准确性方面存在重大缺陷。

**[Can LLMs Help Uncover Insights about LLMs? A Large-Scale, Evolving Literature Analysis of Frontier LLMs](llm_nlp/can_llms_help_uncover_insights_about_llms_a_large-scale_evolving_literature_anal.md)**

:   本文提出半自动化文献分析管线，利用LLM从arXiv论文中自动抽取实验结果构建可持续更新的LLMEvalDB数据集（18127条记录/1737篇论文），并通过该数据集复现并扩展了关于CoT和ICL提示策略在不同任务类型上有效性的关键发现。

**[Can LLMs Identify Critical Limitations within Scientific Research? A Systematic Evaluation on AI Research Papers](llm_nlp/can_llms_identify_critical_limitations_within_scientific_research_a_systematic_e.md)**

:   提出 LimitGen 基准，系统评估 LLM 识别科研论文局限性的能力，包含合成数据集（通过受控扰动创建）和人类标注数据集（ICLR 2025 评审），并通过 RAG 增强文献检索来提升 LLM 生成更具体和建设性反馈的能力。

**[Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](llm_nlp/can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)**

:   本文系统评估了 LLM 利用抽象语义表示（AMR）进行下游任务的能力，发现 AMR 增强的 prompt 在长上下文任务（如对话摘要）中显著提升 Llama 3.1 零样本性能（余弦相似度从 66% 提升至 76%），但在短上下文任务中通常会降低性能。

**[Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](llm_nlp/can_llms_interpret_leverage_amrs.md)**

:   系统评估LLM解释和利用抽象语义表示(AMR)的能力，发现AMR增强提示在长上下文任务（如对话摘要）中可显著提升性能（Llama3.1零样本余弦相似度从66%提升至76%），但在短上下文任务中通常会降低表现。

**[Can LLMs Reason About Program Semantics? A Comprehensive Evaluation of LLMs on Formal Specification Inference](llm_nlp/can_llms_reason_about_program_semantics_a_comprehensive_evaluation_of_llms_on_fo.md)**

:   提出 FormalBench 基准，通过形式化程序规格（formal specifications）推断任务系统评估LLM的程序语义推理能力，发现LLM在简单控制流上表现良好但在循环等复杂结构上挣扎，并设计了自修复提示（self-repair prompts）将成功率提升25%。

**[Can LLMs Simulate L2-English Dialogue? An Information-Theoretic Analysis of L1-Dependent Biases](llm_nlp/can_llms_simulate_l2-english_dialogue_an_information-theoretic_analysis_of_l1-de.md)**

:   本文评估LLM模拟非母语英语使用者（L2学习者）对话的能力，通过信息论和分布密度度量分析LLM生成的L2英语是否能复现人类L2学习者的母语依赖性偏差（如时态一致性错误、回避行为等），发现现代LLM确实能复现部分L1依赖模式。

**[Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](llm_nlp/can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)**

:   本文提出了一种基于可训练 EMG 适配器模块的方法，将无声肌电图（EMG）信号映射到大语言模型（LLM）的输入嵌入空间，在闭合词汇无声 EMG 转文本任务中实现了 0.49 的词错误率（WER），仅需 6 分钟训练数据即比专用模型提升约 20%。

**[Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks](llm_nlp/can_we_further_elicit_reasoning_in_llms_critic-guided_planning_with_retrieval-au.md)**

:   本文提出CR-Planner框架，通过微调的批判模型（critic model）引导推理和检索过程的规划，结合蒙特卡洛树搜索（MCTS）来训练critic，在竞赛编程、定理驱动的数学推理和复杂领域检索问题上显著优于基线方法。

**[Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method](llm_nlp/can_we_retrieve_everything_all_at_once_arm_an_alignment-oriented_llm-based_retri.md)**

:   提出 ARM（Alignment-oriented Retrieval Method），通过在 LLM 解码过程中融合信息对齐（N-gram 约束解码）、结构对齐（MIP 求解器推理数据对象间关系）和自验证聚合三个模块，实现"一次检索所有"所需数据对象，在 Bird 和 OTT-QA 数据集上显著优于标准 RAG（最高 +5.2pt）和 agentic RAG/ReAct（最高 +19.3pt）。

**[Can You Share Your Story? Modeling Clients' Metacognition and Openness for LLM Therapist Evaluation](llm_nlp/can_you_share_your_story_modeling_clients_metacognition_and_openness_for_llm_the.md)**

:   本文提出 MindVoyager 框架，通过构建具有动态元认知和开放性的客户模拟器来评估 LLM 心理治疗师的探索能力，解决了现有评估方法中客户模拟器过于"配合"的问题。

**[Catching Shortcuts: A Framework for Evaluating Shortcuts in Large Language Models](llm_nlp/catching_shortcuts_a_framework_for_evaluating_shortcuts_in_large_language_models.md)**

:   本文提出了一个系统化的框架来检测和评估大语言模型中的捷径学习（shortcut learning）现象，通过构造对比测试集和诊断指标，揭示LLM在多个NLP任务上依赖虚假相关而非真正理解语义的问题。

**[CER: Confidence Enhanced Reasoning in LLMs](llm_nlp/cer_confidence_enhanced_reasoning.md)**

:   提出置信度增强推理框架 CER——在 CoT 推理的每个中间步骤中量化关键 token（数学任务的数值/开放域的专有名词）的置信度，用步间置信度乘积评估整条推理链的可靠性，用置信度加权聚合替代简单多数投票，在数学和开放域任务上比 self-consistency 分别提升最高 7.4% 和 5.8%。

**[Enhancing Character-Level Understanding in LLMs through Token Internal Structure Learning](llm_nlp/character_level_understanding.md)**

:   提出 TIPA（Token Internal Position Awareness）方法，通过在 tokenizer 词汇表上进行逆序字符预测训练，增强 LLM 对 token 内部字符结构和位置的感知能力，显著提升中文拼写纠错等字符级任务的表现。

**[Cheaper and Better Diffusion Language Model via Task-Specific Training](llm_nlp/cheaper_and_better_diffusion_language_model_via_task-specific_training.md)**

:   本文提出通过任务特定的训练策略优化扩散语言模型，在保持生成质量的同时大幅降低训练和推理成本，使扩散模型在文本生成任务上更具实用性。

**[ChronoSense: Exploring Temporal Understanding in Large Language Models with Time Intervals of Events](llm_nlp/chronosense_exploring_temporal_understanding_in_large_language_models_with_time_.md)**

:   提出 ChronoSense 基准，首次完整覆盖 Allen 区间代数全部 13 种时间关系并加入 3 类时间算术任务，通过对 7 个 LLM 在 0-shot / few-shot / CoT 下的系统评估，揭示模型时间理解能力普遍薄弱且严重依赖预训练记忆。

**[Circuit Compositions: Exploring Modular Structures in Transformer-Based Language Models](llm_nlp/circuit_compositions_modular_structures.md)**

:   通过在 PCFG SET 数据集上识别 10 个组合性字符串编辑操作的电路（circuits），研究 Transformer 中功能相关电路之间的模块化关系，发现功能相似的电路具有显著的节点重叠和跨任务忠实度，且电路可以通过集合运算（并集）组合以表示超出单个电路能力的更复杂功能。

**[Circuit Stability Characterizes Language Model Generalization](llm_nlp/circuit_stability_characterizes_language_model_generalization.md)**

:   本文提出"电路稳定性"作为评估语言模型泛化能力的新方法，通过数学形式化定义软电路和电路等价性，在算术推理、布尔表达式和体育理解三个案例研究中证明电路稳定性可以预测和刻画泛化行为。

**[Classifying Unreliable Narrators with Large Language Models](llm_nlp/classifying_unreliable_narrators.md)**

:   借鉴文学叙事学理论，定义三种不同层次的不可靠叙事者（intra-narrational / inter-narrational / inter-textual），构建专家标注数据集 TUNa，系统评估 LLM 在分类不可靠叙事者任务上的表现。

**[Clue Guided Re-Assessment to Improve Reasoning in Large Language Models](llm_nlp/clue_guided_re-assessment_to_improve_reasoning_in_large_language_models.md)**

:   本文提出"线索引导的重新评估"（Clue Guided Re-Assessment）方法，通过在LLM推理过程中提取关键线索并引导模型对初始推理进行反思和修正，显著提升了多步骤推理任务的准确率。

**[CodeTool: Enhancing Programmatic Tool Invocation of LLMs via Process Supervision](llm_nlp/codetool_enhancing_programmatic_tool_invocation_of_llms_via_process_supervision.md)**

:   提出 CodeTool，一种逐步代码生成框架，通过即时奖励（On-the-spot Reward）和潜在奖励（Latent Reward）两种过程奖励机制引导 LLM 选择最优的工具调用路径，在 StableToolBench 和 RestBench-TMDB 上显著超越现有方法。

**[CogniBench: A Legal-inspired Framework and Dataset for Assessing Cognitive Faithfulness of Large Language Models](llm_nlp/cognibench_cognitive_faithfulness.md)**

:   借鉴法律领域间接证据认定标准，提出分层评估框架和 CogniBench 数据集，首次系统性地定义和评估 LLM 在认知性陈述（推理、评价、解释）中的忠实度问题，并训练 CogniDet 检测器实现事实与认知幻觉的同时检测。

**[CogSteer: Cognition-Inspired Selective Layer Intervention for Efficiently Steering Large Language Models](llm_nlp/cogsteer_cognition-inspired_selective_layer_intervention_for_efficiently_steerin.md)**

:   利用认知科学中的眼动数据分析 LLM 各层行为，发现中间层与人类注视相关性最高且最适合语义干预，提出 CogSteer 框架——仅微调最优单层（约 3% 参数）即可达到或超过全层微调的效果，在 GLUE/毒性控制任务上有效。

**[Enough Coin Flips Can Make LLMs Act Bayesian](llm_nlp/coin_flips_bayesian.md)**

:   通过抛硬币这一受控随机过程，系统研究LLM是否在in-context learning中执行贝叶斯推理，发现LLM通常具有偏置先验，但随着上下文证据增加会以近似贝叶斯更新的方式修正后验估计，偏差主要源于校准不良的先验而非错误的更新机制。

**[Collaborative Performance Prediction for Large Language Models](llm_nlp/collaborative_performance_prediction_for_large_language_models.md)**

:   本文提出协同性能预测框架CPP，利用多个LLM在多个任务上的历史性能数据及模型/任务的设计因素进行协同过滤式预测，突破了传统Scaling Law仅限单模型族预测的限制，能跨模型族准确预测LLM的下游性能。

**[Combining the Best of Both Worlds: A Method for Hybrid NMT and LLM Translation](llm_nlp/combining_the_best_of_both_worlds_a_method_for_hybrid_nmt_and_llm_translation.md)**

:   提出基于源句特征的NMT与LLM混合翻译调度策略（PPLT与JDM），在保持翻译质量最优的同时将LLM调用比例降至约25-30%，大幅减少计算开销。

**[Comparing Large Language Models in Extracting Subjective Information from Political News](llm_nlp/comparing_large_language_models_in_extracting_subjective_information_from_politi.md)**

:   本文系统比较了多种大语言模型在政治新闻中提取主观信息（情感倾向、立场、偏见、框架效应等）的能力，发现不同LLM在不同维度的主观信息提取上表现差异显著，并揭示了LLM自身的政治偏见对提取结果的影响。

**[Comparing Linguistic Acceptability Judgments of Autoregressive Language Models](llm_nlp/comparing_linguistic_acceptability_judgments_of_autoregressive_language_models.md)**

:   本文比较了多种自回归语言模型（GPT系列、Llama系列等）在语言可接受性判断任务上的表现，通过系统实验揭示了模型规模、训练数据和架构对语法判断能力的影响，并探讨了模型的语法知识是否与人类语言直觉一致。

**[Revisiting Compositional Generalization Capability of Large Language Models Considering Instruction Following Ability](llm_nlp/compositional_generalization_instruction.md)**

:   提出 Ordered CommonGen 基准，通过要求 LLM 按指定顺序生成包含所有概念的句子，同时评估组合泛化与指令遵循能力，在 36 个 LLM 上发现即使最强模型也仅能达到约 75% 的有序覆盖率。

**[Computation Mechanism Behind LLM Position Generalization](llm_nlp/computation_mechanism_behind_llm_position_generalization.md)**

:   揭示 LLM 注意力 logit 学习了位置相关性和语义重要性的近似算术加法解耦（$W_{i,j} \approx f(\mathbf{q}, i-j) + g(\mathbf{q}, \mathbf{k})$，线性相关 0.959），发现了使这种解耦成立的中间表示模式，并用此解释了 LLM 的位置排列容忍性和长度泛化能力。

**[How Humans and LLMs Organize Conceptual Knowledge: Exploring Subordinate Categories in Italian](llm_nlp/conceptual_knowledge_org.md)**

:   通过构建首个意大利语下位类别心理语言学数据集（187 个基本类别），系统对比了人类和 LLM 在下位概念层级上的类别组织结构，发现两者的对齐度较低但在不同语义领域存在显著差异。

**[Concreteness Versus Abstractness: A Selectivity Analysis in LLMs](llm_nlp/concreteness_versus_abstractness_a_selectivity_analysis_in_llms.md)**

:   本文探究大语言模型中具体概念（如"苹果"）和抽象概念（如"自由"）的处理差异，通过选择性分析（selectivity analysis）发现LLM内部存在对具体性/抽象性具有选择性响应的神经元群体，揭示了LLM的语义表示与人类认知理论中的"具体性效应"（concreteness effect）之间的有趣对应。

**[Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis and Refinement](llm_nlp/condor_enhance_llm_alignment_with_knowledge-driven_data_synthesis_and_refinement.md)**

:   Condor 提出了一个两阶段合成数据生成框架，通过 World Knowledge Tree 构建多样化标签驱动的问题生成、再用 Self-Reflection Refinement 迭代优化回答质量，仅用 20K 合成样本即可让基座模型在对话对齐任务上超越同等规模的对手，并在最大 72B 模型上验证了迭代自我改进的有效性。

**[Conformity in Large Language Models](llm_nlp/conformity_in_large_language_models.md)**

:   将心理学中的 Asch 从众实验范式迁移到 LLM 上，系统研究了 LLM 的从众行为（conformity），发现所有模型都会受多数意见影响改变答案，且不确定性越高越容易从众，并提出 Devil's Advocate 和 Question Distillation 两种干预方法有效缓解从众效应。

**[ConsistencyChecker: Tree-based Evaluation of LLM Generalization Capabilities](llm_nlp/consistencychecker_tree_evaluation.md)**

:   ConsistencyChecker 提出基于自一致性树（self-consistency tree）的无参考 LLM 评估框架，通过构建可逆变换的树状多步路径（如多语言往返翻译、代码等价重写），量化模型在迭代变换中的语义/功能保持能力，动态生成 benchmark 从根源消除数据泄露，且与 WMT 2024 权威排名的相关性 r > 0.7，证明无需配对数据即可可靠评估 LLM 泛化能力。

**[Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models](llm_nlp/contrastive_perplexity_controlled_gen.md)**

:   提出基于原型对比困惑度（Contrastive Perplexity, CP）的框架，通过构造语义相似但毒性属性不同的正负样本对，在困惑度空间中进行对比学习来微调 LLM，实现显著的毒性降低（Mistral-7b 毒性从 33.1% 降至 4.3%）且几乎不影响下游任务性能。

**[Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering](llm_nlp/contrastive_prompting_embeddings.md)**

:   提出对比提示（Contrastive Prompting, CP）方法，通过构造辅助提示编码句子的非核心信息，在推理时将正常提示与辅助提示的隐层激活值做"语义减法"，过滤停用词等无关语义，使 LLM 句子嵌入更聚焦核心语义，即插即用地一致提升 PromptEOL/CoT/Knowledge 等多种提示方法在 STS 和分类任务上的表现。

**[Controlling Politeness in Multi-Turn Dialogues Through Pre-Phrase Augmentation](llm_nlp/controlling_politeness_in_multi-turn_dialogues_through_pre-phrase_augmentation.md)**

:   本文提出一种基于前缀短语增强（Pre-Phrase Augmentation）的方法，通过在对话生成过程中自动添加礼貌性调控前缀，实现多轮对话中礼貌程度的细粒度控制，同时保持对话内容的连贯性和信息完整性。

**[Conversational Quality Assessment: A Large-Scale Corpus and Comprehensive Study](llm_nlp/conversational_quality_assessment_a_large-scale_corpus_and_comprehensive_study.md)**

:   本文构建了一个大规模多维度对话质量评估语料库，涵盖流畅性、一致性、信息量、参与度等多个质量维度，并基于此语料库对现有对话评估方法进行了全面的基准测试和分析。

**[Convert Language Model into a Value-based Strategic Planner](llm_nlp/convert_language_model_into_a_value-based_strategic_planner.md)**

:   提出 straQ* 框架，将 LLM 的 next-token prediction 转化为 next-strategy prediction，用 Bellman 方程训练 LLM 作为策略级 Q 网络，在情感支持对话（ESC）中根据长期回报规划最优支持策略，以即插即用的轻量规划器引导对话 LLM 生成高质量回复。

**[Cool-Fusion: Fuse Large Language Models without Training](llm_nlp/cool-fusion_fuse_large_language_models_without_training.md)**

:   提出 Cool-Fusion，一种无需任何训练即可融合异构 LLM 的方法，通过在文本段粒度上让多个模型互相评估和重排序生成内容，在 GSM8K 上相对最强源模型提升 17.4% 准确率。

**[COSMIC: Generalized Refusal Direction Identification in LLM Activations](llm_nlp/cosmic_generalized_refusal_direction_identification_in_llm_activations.md)**

:   提出 COSMIC 框架，利用余弦相似度在激活空间中自动选择拒绝引导方向，完全不依赖模型输出 token 或预定义拒绝模板，在标准设置下匹配已有方法性能，并首次在对抗性完全拒绝和弱对齐模型中成功提取有效的拒绝方向。

**[Cross-Modal Alignment for LLM-Enhanced Spoken Language Understanding](llm_nlp/cross-modal_alignment_for_llm-enhanced_spoken_language_understanding.md)**

:   本文提出一种跨模态对齐框架，通过将语音表示与LLM的文本语义空间显式对齐，实现了LLM增强的口语理解（SLU），在意图识别和槽位填充任务上取得SOTA性能。

**[Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts](llm_nlp/cross_model_transferability_sv.md)**

:   提出 L-Cross Modulation 方法，通过简单线性变换将一个 LLM 的概念方向向量（steering vectors）迁移到另一个 LLM 中实现行为控制，发现三个关键结论：(1) 跨模型 SV 迁移有效；(2) 不同概念共享同一变换矩阵；(3) 小模型的 SV 可以控制大模型（弱到强迁移）。

**[Cuckoo: An IE Free Rider Hatched by Massive Nutrition in LLM's Nest](llm_nlp/cuckoo_an_ie_free_rider_hatched_by_massive_nutrition_in_llms_nest.md)**

:   本文提出 Next Tokens Extraction (NTE) 范式，将 LLM 预训练数据中的下一个 token 预测转化为 BIO 标注的抽取任务，利用 C4 和 TuluV3 共 1.026 亿实例预训练 RoBERTa 标注器（Cuckoo），在少样本信息抽取任务上全面超越现有 IE 预训练模型。

**[Cultural Learning-Based Culture Adaptation of Language Models](llm_nlp/cultural_learning-based_culture_adaptation_of_language_models.md)**

:   提出 CLCA 框架，借鉴文化学习理论，通过模拟社会交互生成文化适配对话数据，结合意图理解进行多任务训练，在 World Values Survey 上显著提升多种 LLM 的文化价值观对齐。

**[Culture is Not Trivia: Sociocultural Theory for Cultural NLP](llm_nlp/culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)**

:   本文从社会文化语言学理论出发，指出当前文化 NLP 的方法论局限（粗粒度国家边界、静态基准、缺乏统一文化定义），论证文化是动态建构的过程而非静态知识，并提出"本地化"作为更可行的研究框架。

**[DeAL: Decoding-time Alignment for Large Language Models](llm_nlp/deal_decoding_time_alignment.md)**

:   DeAL 将 LLM 对齐问题重新形式化为解码时的启发式搜索问题，在推理阶段利用可定制的奖励函数（包括程序化约束和参数化 reward model）引导 token 选择，实现了灵活的多目标对齐且可与 RLHF 互补叠加。

**[When People are Floods: Analyzing Dehumanizing Metaphors in Immigration Discourse with Large Language Models](llm_nlp/dehumanizing_metaphors_immigration.md)**

:   提出结合 LLM 词级隐喻检测与 SBERT 篇章级语义关联的计算框架，在 40 万条美国移民推文上揭示保守派更多使用去人化隐喻、但生物类隐喻对自由派的用户互动效应更强的复杂图景。

**[Déjà Vu? Decoding Repeated Reading from Eye Movements](llm_nlp/deja_vu_decoding_repeated_reading_from_eye_movements.md)**

:   首次提出从眼动模式自动判断读者是否曾经阅读过某文本的预测任务，通过基于特征的 XGBoost 和神经网络 RoBERTEye 模型，在单次试验中达到 ~70% 准确率、配对试验中达到 ~91% 准确率，并引入 E-Z Reader 认知模型生成的合成扫视路径作为辅助参考信号来增强预测。

**[DenseLoRA: Dense Low-Rank Adaptation of Large Language Models](llm_nlp/denselora_dense_low-rank_adaptation_of_large_language_models.md)**

:   本文提出DenseLoRA，通过引入跨层共享的Encoder-Decoder进行隐藏表示的压缩与重建，用一个稠密的小型低秩矩阵替代LoRA中两个冗余的低秩矩阵来进行适配，仅用0.01%可训练参数在LLaMA3-8B上达到83.8%准确率，超越了LoRA用0.70%参数达到的80.8%。

**[Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models](llm_nlp/deontological_keyword_bias.md)**

:   本文揭示LLM存在"义务论关键词偏见"(DKB)——当提示中包含"must"、"ought to"等模态义务表达时，模型会将超过90%的常识场景误判为义务，并提出基于少样本示例与推理提示的去偏策略。

**[Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training](llm_nlp/derta_decoupled_refusal.md)**

:   发现标准安全微调数据存在"拒绝位置偏差"——模型只学会在回答开头拒绝，中途发现不安全时无法中断。提出 DeRTa（Decoupled Refusal Training），通过"有害前缀+安全拒绝"的 MLE 训练和在每个位置模拟"从有害到安全"转换的 RTO 训练，让 LLM 能在回答的任何位置感知到不安全时拒绝，在六种攻击场景下超越 GPT-4 和 LLaMA3-Instruct。

**[Detecting Referring Expressions in Visually Grounded Dialogue with Autoregressive Language Models](llm_nlp/detecting_referring_expressions_in_visually_grounded_dialogue_with_autoregressiv.md)**

:   本文将视觉对话中的指称表达检测建模为自回归 token 预测任务，通过对 Llama 3.1-8B 进行参数高效微调 (QLoRA)，证明仅使用文本上下文即可有效检测视觉对话中的 mention span，在 AGOS 和 PhotoBook 数据集上 F1 达 0.90 和 0.94。

**[DICE-Bench: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues](llm_nlp/dice-bench_evaluating_the_tool-use_capabilities_of_large_language_models_in_mult.md)**

:   提出 DICE-Bench，一个面向多轮多方对话场景的函数调用评测基准，包含 1607 条高质量对话实例和量化信息分散度的 DICE-Score 指标，揭示当前 LLM 在复杂对话中工具调用能力的不足。

**[DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models](llm_nlp/difflm_controllable_synthetic_data_generation_via_diffusion_language_models.md)**

:   DiffLM 提出基于 VAE + 潜在扩散 + 冻结 LLM 解码器的可控数据合成框架，通过在潜在空间引入扩散过程来精确建模真实数据分布，并以 soft prompt 方式将分布信息注入 LLM，在表格、代码和工具三类结构化数据上合成质量超越真实数据 2%-7%。

**[Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence In Large Language Models](llm_nlp/direct_confidence_alignment_aligning_verbalized_confidence_with_internal_confide.md)**

:   提出 Direct Confidence Alignment (DCA)，利用 DPO 将 LLM 的文字表达置信度（verbalized confidence）与内部 token 概率置信度（internal confidence）对齐，提升模型置信度表达的一致性和透明度。

**[DiSCo: Device-Server Collaborative LLM-Based Text Streaming Services](llm_nlp/disco_device-server_collaborative_llm-based_text_streaming_services.md)**

:   提出 DiSCo，一个端-云协同的 LLM 推理调度器，通过成本感知的请求分发和 token 级迁移机制，在成本约束下优化用户的首 token 延迟 (TTFT) 和 token 间延迟 (TBT)。

**[Disentangling Memory and Reasoning Ability in Large Language Models](llm_nlp/disentangle_memory_reasoning.md)**

:   提出将 LLM 的推理过程显式分解为"记忆回忆"和"逻辑推理"两个步骤——引入 `<memory>` 和 `<reason>` 两个可学习特殊 token 标记每步是知识回忆还是逻辑推理，用双 LLM 框架生成训练数据后 LoRA 微调，在 StrategyQA/CommonsenseQA/TruthfulQA 上提升性能并增强可解释性，8B 模型在 TruthfulQA 上超越 GPT-4o。

**[Diversity-oriented Data Augmentation with Large Language Models](llm_nlp/diversity_data_augmentation.md)**

:   提出 DoAug 框架，通过 SFT+DPO 微调 LLM 释义器并结合核心集选择与多样性采样，在保持语义一致性的同时显著提升增强数据集的多样性，在 12 个数据集上平均性能提升 10.52%，超出次优基线 3.76 个百分点。

**[Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs](llm_nlp/do_language_models_mirror_human_confidence_exploring_psychological_insights_to_a.md)**

:   从心理学过度自信理论出发，揭示 LLM 的置信度估计对任务难度不敏感且会受角色扮演偏见影响（如专家角色过度自信、女性/亚裔角色低自信但实际准确率不变），提出 Answer-Free Confidence Estimation（AFCE）方法将信心估计与答案生成解耦，在高难度任务上将 GPT-4o 的 ECE 降低 58.4%。

**[Do Language Models Understand Honorific Systems in Javanese?](llm_nlp/do_language_models_understand_honorific_systems_in_javanese.md)**

:   构建首个爪哇语敬语语料库 Unggah-Ungguh（4,024 句，覆盖四个敬语层级），通过分类/风格转换/跨语言翻译/对话生成四个任务系统评估 LLM 对爪哇语敬语系统的理解能力，发现即使最强闭源模型（GPT-4o）的零样本分类准确率也仅 53.5%，且普遍偏向特定敬语层级。

**[Do Language Models Understand the Cognitive Tasks Given to Them? Investigations with the N-Back Paradigm](llm_nlp/do_language_models_understand_the_cognitive_tasks_given_to_them_investigations_w.md)**

:   通过 N-back 范式系统分析多个 LLM 的认知任务表现，发现性能低下的主因是任务理解不足和任务集维持失败，而非工作记忆容量限制，最佳模型（Llama 3.1 70b）在课程学习辅助下甚至能完成 10-back 任务（准确率 84.75%）。

**[Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?](llm_nlp/do_large_language_models_perform_latent_multi-hop_reasoning_without_exploiting_s.md)**

:   构建无快捷方式的评估数据集 SOCRATES，系统评估 41 个 LLM 在潜在多跳推理中的真实能力，发现模型在国家类桥接实体上可达 80% 组合率，但年份类仅约 5%。

**[Do LLMs Give Psychometrically Plausible Responses in Educational Assessments?](llm_nlp/do_llms_give_psychometrically_plausible_responses_in_educational_assessments.md)**

:   从心理测量学（经典测试理论 CTT 和项目反应理论 IRT）的角度评估 18 个指令微调 LLM 在教育评估中的"类人性"，发现即使经过温度缩放校准，LLM 的响应分布与人类仍有本质差异——大模型过度自信，且无法预测人类被干扰项吸引的模式，零样本 LLM 不适合替代人类进行测试预试验。

**[Does Time Have Its Place? Temporal Heads Where Language Models Recall Time-specific Information](llm_nlp/does_time_have_its_place_temporal_heads_where_language_models_recall_time-specif.md)**

:   通过 EAP-IG 电路分析在 Llama-2/Qwen/Phi-3 中发现了专门处理时间条件知识的"时间头"（Temporal Heads），消融这些头只降低时间知识准确率（降 3-9%）而不影响时间无关知识和通用 QA，并展示了通过注入时间头激活值实现选择性时间知识编辑的可能性。

**[Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting](llm_nlp/dolphin_document_image_parsing_via_heterogeneous_anchor_prompting.md)**

:   提出 Dolphin，一个轻量级（322M）的文档图像解析模型，采用"先分析后解析"（analyze-then-parse）两阶段范式——先进行页面级布局分析生成阅读顺序的元素序列，再利用异构锚点提示（heterogeneous anchor prompting）并行解析各元素内容，以仅 322M 参数在页面级和元素级解析任务上超越 7B+ 模型和商业系统。

**[Dynamic Knowledge Integration for Evidence-Driven Counter-Argument Generation with Large Language Models](llm_nlp/dynamic_knowledge_integration_for_evidence-driven_counter-argument_generation_wi.md)**

:   提出动态网络知识检索框架来增强 LLM 的反驳论证生成质量，构建了长度适中的新评估数据集（150对），并用 LLM-as-a-Judge 评估方法取代传统参考度量，实验证明外部知识集成显著提升了生成质量的相关性、说服力和事实性。

**[Dynamic Parallel Tree Search for Efficient LLM Reasoning](llm_nlp/dynamic_parallel_tree_search_for_efficient_llm_reasoning.md)**

:   提出DPTS（Dynamic Parallel Tree Search）框架，通过并行流水线（Parallelism Streamline）解决树搜索中路径频繁切换难以并行化的问题，通过搜索与转换机制（Search and Transition Mechanism）的早停和深度搜索策略减少对低置信度路径的冗余探索，在Qwen-2.5和Llama-3上实现2-4倍推理加速，同时保持或超越MCTS等现有算法的推理准确率。

**[ECLM: Entity Level Language Model for Spoken Language Understanding with Chain of Intent](llm_nlp/eclm_entity_level_language_model_spoken_language_understanding.md)**

:   提出 ECLM 框架，将 LLM 应用于多意图口语理解：通过将 token 级槽填充转化为实体识别任务解决序列对齐问题，引入"意图链"（Chain of Intent）实现逐步多意图识别，在 MixATIS 和 MixSNIPS 上大幅超越 SOTA 基线。

**[EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models](llm_nlp/editext_diffusion_text_editing.md)**

:   提出 EdiText，一种基于嵌入扩散模型的可控文本编辑方法，结合 SDEdit 粗粒度编辑和 self-conditioning 细粒度编辑，实现从轻微修改到大幅改写的多尺度文本编辑控制。

**[Educators' Perceptions of Large Language Models as Tutors: Comparing Human and AI Tutors in a Blind Text-only Setting](llm_nlp/educators_perceptions_of_large_language_models_as_tutors_comparing_human_and_ai_.md)**

:   本文通过盲测实验让有教学经验的人类标注者对比评估 LLM 辅导员与人类辅导员在小学数学应用题场景下的表现，在参与度、共情能力、支架式教学和简洁性四个维度上 LLM 均被评为优于人类辅导员，其中共情维度最突出——80% 的标注者更偏好 LLM。

**[Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](llm_nlp/efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)**

:   提出 EnsembleLoRA——一种面向多数据集微调的高效集成方法，利用一阶 Taylor 近似快速估计任务亲和度将数据集分组，为每组训练一个 adapter 后加权组合，在 10 个 SuperGLUE 任务上以仅 9% 额外计算代价将 QLoRA 的平均测试准确率提升 10%。

**[ELI-Why: Evaluating the Pedagogical Utility of Language Model Explanations](llm_nlp/eli-why_evaluating_the_pedagogical_utility_of_language_model_explanations.md)**

:   构建了包含 13.4K "Why" 问题的 ELI-Why 基准，通过两项人类研究发现 GPT-4 生成的面向不同教育水平的解释仅 50% 能匹配目标年级（人工策划达 79%），且对学习者信息需求的满足度比人类答案低 20%。

**[Enabling LLM Knowledge Analysis via Extensive Materialization](llm_nlp/enabling_llm_knowledge_analysis_via_extensive_materialization.md)**

:   本文提出通过递归查询和结果整合将 LLM 的事实知识大规模物化为知识库的方法论，构建了包含 1.01 亿三元组、290 万实体的 GPTKB，首次全面分析了 GPT-4o-mini 知识的规模、准确性、偏见、时效性和一致性。

**[Enhancing Input-Label Mapping in In-Context Learning with Contrastive Decoding](llm_nlp/enhancing_input-label_mapping_in_in-context_learning_with_contrastive_decoding.md)**

**[Enhancing Spoken Discourse Modeling in Language Models Using Gestural Cues](llm_nlp/enhancing_spoken_discourse_modeling_in_language_models_using_gestural_cues.md)**

:   提出将手势动作序列（3D 人体运动数据）通过 VQ-VAE 编码为离散 gesture token，再经特征对齐映射到语言模型输入空间，用于增强口语篇章建模；在三类篇章标记（话语连接词、量词、立场标记）的文本填充任务上验证了手势信息对口语篇章理解的互补价值。

**[Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge](llm_nlp/enhancing_transformation_from_natural_language_to_signal_temporal_logic_using_ll.md)**

:   提出 STL-DivEn 数据集（16K样本）和 KGST（知识引导的 STL 转换）框架，通过"生成-精炼"两阶段流程将自然语言转换为信号时序逻辑（STL），在 STL Formula Accuracy 上达到 0.5587，显著超过 GPT-4（0.4733）和 DeepSeek（0.4790）。

**[EnigmaToM: Improve LLMs' Theory-of-Mind Reasoning Capabilities with Neural Knowledge Base of Entity States](llm_nlp/enigmatom_improve_llms_theory-of-mind_reasoning_capabilities_with_neural_knowled.md)**

:   提出 EnigmaToM 神经符号框架，通过构建实体状态的神经知识库（Enigma）来生成空间场景图用于信念追踪，并结合心理学启发的迭代遮蔽机制实现准确的视角切换，在 ToMi、HiToM 和 FANToM 三个基准上显著提升了 LLM 的心智理论推理能力，尤其在高阶推理场景中表现突出。

**[Growing Through Experience: Scaling Episodic Grounding in Language Models](llm_nlp/episodic_grounding_experience.md)**

:   提出一个 weak-to-strong episodic grounding 框架，利用 MCTS 收集结构化经验数据，通过行为比率蒸馏将小模型的 episodic grounding 能力迁移到大模型，结合 DPO 优化实现从成功和失败经验中学习，在物理规划任务上超越 GPT-4o 等 SOTA 模型 3.45%。

**[Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection](llm_nlp/erm_prompt_optimization_memory.md)**

:   提出 ERM 方法，通过指导性元提示生成带详细解题过程的 exemplar 来增强 feedback 质量，并引入 Feedback Memory 和 Exemplar Factory 两种长期记忆机制来高效存储和复用历史反馈与示例，在多个任务上以约一半的优化步数超越了 SOTA prompt 优化方法。

**[EscapeBench: Towards Advancing Creative Intelligence of Language Model Agents](llm_nlp/escapebench_creative_agent.md)**

:   本文推出 EscapeBench——基于密室逃脱游戏的 LLM Agent 创意智能评测基准（36 个场景、3 个难度），揭示当前模型在创造性工具使用和隐式目标推断上的严重不足，并提出 EscapeAgent（Foresight + Reflection）将提示依赖降低近 50%。

**[Evaluating Implicit Bias in Large Language Models by Attacking from a Psychometric Perspective](llm_nlp/evaluating_implicit_bias_in_large_language_models_by_attacking_from_a_psychometr.md)**

:   借鉴认知与社会心理学中的三种心理测量学原理（目标转移、认知协调、模仿学习），设计 Disguise/Deception/Teaching 三类攻击方法来诱发 LLM 的隐式偏见，构建了双语基准 BUMBLE（12.7K 条目覆盖 9 类偏见），揭示所有主流 LLM 均存在可被激发的系统性隐式偏见。

**[Evaluating Language Models as Synthetic Data Generators](llm_nlp/evaluating_lms_synthetic_data_gen.md)**

:   提出 AgoraBench 基准，系统评估 6 个 LLM 在 3 个领域×3 种数据生成方式下的数据生成能力，通过训练 99 个学生模型发现：LLM 的数据生成能力与问题求解能力不直接相关，GPT-4o 在实例生成上最强而 Claude-3.5-Sonnet 在质量增强上最强。

**[ExpeTrans: LLMs Are Experiential Transfer Learners](llm_nlp/expetrans_llms_are_experiential_transfer_learners.md)**

:   ExpeTrans 提出了一个自主经验迁移框架，让 LLM 模仿人类认知智能，将已有源任务的解题经验自动迁移到新遇到的目标任务上，在 13 个数据集上有效提升了 LLM 性能，无需为每个新任务人工收集经验。

**[Explain-then-Process: Using Grammar Prompting to Enhance Grammatical Acceptability Judgments](llm_nlp/explain-then-process_using_grammar_prompting_to_enhance_grammatical_acceptabilit.md)**

:   提出 grammar prompting 的 explain-then-process 范式——先让 LLM 生成目标语法现象的解释，再将该解释作为上下文反馈给目标模型（LLM 或 SLM）辅助最小对语法判断。在英语 BLiMP、中文 SLING、俄语 RuBLiMP 三个跨语言基准上显著提升准确率，SLM 搭配 GP+CoT 将 LLM-SLM 平均差距从 13.0pp 缩小到 5.8pp（缩小 56%）。

**[ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models](llm_nlp/explica_evaluating_explicit_causal_reasoning_in_large_language_models.md)**

:   提出 ExpliCa 数据集（4800 条问题，含因果和时间连接词），首次整合因果和时间关系评估并配以众包人类评分，发现即使顶级模型准确率也难超 0.80，且模型系统性地将时间关系误判为因果关系。

**[Exploring Explanations Improves the Robustness of In-Context Learning](llm_nlp/exploring_explanations_improves_the_robustness_of_in-context_learning.md)**

:   提出 X²-ICL 框架，通过在上下文学习的示例中为所有可能的标签（而非仅观测标签）生成解释推理路径，系统性地探索隐变量推理空间，从而显著提升 ICL 在分布外（OOD）数据上的鲁棒性——在 5 个 LLM 上的 8 个 OOD 数据集中，X²-ICL 在 6-8 个上超越 ICL 和 X-ICL。

**[Exploring Graph Representations of Logical Forms for Language Modeling](llm_nlp/exploring_graph_representations_of_logical_forms_for_language_modeling.md)**

:   提出 GFoLDS，一种在 DMRS 逻辑形式图表示上预训练的图 Transformer 语言模型，并提出"语言知识催化假说"(LKCH)：逻辑形式语言模型几乎立刻学会基础语言现象，进而加速复杂模式学习，在相同数据量下大幅超越 BERT。

**[HiCUPID: Exploring the Potential of LLMs as Personalized Assistants](llm_nlp/exploring_the_potential_of_llms_as.md)**

:   提出HiCUPID——首个全面满足个性化AI助手五大需求（用户信息遵循/隐含信息理解/多信息推理/长上下文建模/主动性回复）的开源基准，含1500用户×40个对话+QA对+Llama-3.2自动评估模型。

**[Factual Knowledge in Language Models: Robustness and Anomalies under Simple Temporal Context Variations](llm_nlp/factual_knowledge_in_language_models_robustness_and_anomalies_under_simple_tempo.md)**

:   发布 TimeStress 数据集（521K 陈述，2003 条时间事实），评估 18 个 LLM 在时间上下文变化下的事实知识鲁棒性，发现最好的模型仅对 11% 的事实实现完美鲁棒，且存在人类不会犯的关键错误。

**[FoodTaxo: Generating Food Taxonomies with Large Language Models](llm_nlp/foodtaxo_generating_food_taxonomies_with_large_language_models.md)**

:   提出 FoodTaxo，基于 Llama-3 的迭代自底向上分类法生成与补全算法，利用 CoT 提示 + RAG 检索 + NLI 验证三阶段流程，从已知叶节点概念出发逐步构建层次化 taxonomy；在五个基准数据集上与 TacoPrompt 等 SOTA 方法竞争，同时通过 reference-free 指标和消融实验揭示了非叶节点放置这一根本性瓶颈。

**[From Data to Knowledge: Evaluating How Efficiently Language Models Learn Facts](llm_nlp/from_data_to_knowledge_evaluating_how_efficiently_language_models_learn_facts.md)**

:   首次直接研究事实在预训练数据中出现频次与 LLM 能否回忆该事实之间的关系，提出两种样本效率指标，发现不同架构/规模的模型在高频事实上表现相似但在低频事实上差异显著——低频事实的学习能力是区分模型样本效率的关键。

**[From Neurons to Semantics: Evaluating Cross-Linguistic Alignment Capabilities of Large Language Models via Neurons Alignment](llm_nlp/from_neurons_to_semantics_evaluating_cross-linguistic_alignment_capabilities_of_.md)**

:   提出基于神经元激活状态的跨语言对齐评估框架 NeuronXA，利用 FFN 层神经元状态作为语言的内在表征来衡量多语言 LLM 的跨语言对齐能力，仅需 100 对平行句子即可实现与下游任务性能 0.9556 的皮尔逊相关。

**[From Selection to Generation: A Survey of LLM-based Active Learning](llm_nlp/from_selection_to_generation_a_survey.md)**

:   首篇系统综述 LLM 时代主动学习（Active Learning）的全景图谱，提出以 Querying（选择 + 生成）× Annotation（人工 + LLM + 混合）为两轴的分类体系，完整梳理了 LLM 如何在五步 AL 循环的每个环节中替代或增强传统方法，并拓展到 ICL、SFT、RLHF、知识蒸馏等四大 LLM 学习范式。

**[Game Development as Human-LLM Interaction](llm_nlp/game_development_as_human-llm_interaction.md)**

:   本文提出 Chat Game Engine（ChatGE），一个基于 LLM 的对话式游戏引擎，通过自然语言交互让用户无需编程知识即可开发定制游戏，并设计了数据合成流水线和三阶段渐进式训练策略将对话模型转化为游戏引擎。

**[GAMEBoT: Transparent Assessment of LLM Reasoning in Games](llm_nlp/gamebot_transparent_assessment_of_llm_reasoning_in_games.md)**

:   本文提出 GAMEBoT，一个基于游戏的 LLM 推理评测平台，通过将游戏中的复杂推理分解为预定义的模块化子问题，结合基于规则的 ground truth 验证，实现对 17 个主流 LLM 的透明化推理能力评估。

**[Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models](llm_nlp/gapo_multi_objective_alignment.md)**

:   提出GAPO，一种基于梯度自适应缩放的多目标策略优化方法，利用多梯度下降算法(MGDA)结合梯度归一化，平衡LLM在帮助性和无害性等冲突目标间的权衡，并通过P-GAPO支持用户偏好驱动的Pareto前沿生成。

**[GAPO: Learning Preferential Prompt through Generative Adversarial Policy Optimization](llm_nlp/gapo_preferential_prompt.md)**

:   提出 GAPO（Generative Adversarial Policy Optimization）框架，将 GAN 的对抗训练机制与 PPO 结合，使用 encoder-only 奖励模型替代传统 decoder-only 架构，通过"Preferential Prompt"（修改 prompt 中的约束而非 response）的新范式来增强 LLM 对细粒度约束的理解和遵循能力，在 IFEval 和产品描述生成任务上大幅超越 DPO/KTO/SimPO 等基线。

**[Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models](llm_nlp/generative_psycholexical_approach_for_constructing_value.md)**

:   提出生成式心理词汇方法（GPLA），自动化构建面向LLM的五因素价值体系（社会责任、冒险性、规则遵循、自我效能、理性），在结构效度、安全预测和价值对齐上优于经典Schwartz人类价值体系。

**[Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models](llm_nlp/genetic_instruct_scaling_up_synthetic_generation_of_coding_instructions_for_larg.md)**

:   提出 Genetic-Instruct 算法，借鉴进化算法的交叉和变异操作，从 512 个种子指令扩展生成 750 万+高质量编码指令，使用 Instructor-LLM/Coder-LLM/Judge-LLM 三角色流水线，训练后的模型在代码生成基准上超越 Self-Instruct 和 Evol-Instruct。

**[GenKnowSub: Improving Modularity and Reusability of LLMs through General Knowledge Subtraction](llm_nlp/genknowsub_improving_modularity_and_reusability_of_llms_through_general_knowledg.md)**

:   提出 GenKnowSub（通用知识减法），通过在 Wikipedia 语料上训练通用知识 LoRA 并从任务特定 LoRA 中减去它（$LoRA_{res}^i = LoRA_{ts}^i - LoRA_g$），得到更纯净的残差模块；结合 Arrow 路由算法动态选择最相关的模块，在 Phi-3 上零样本迁移平均准确率提升 1.6%，跨语言场景提升更大（德语+3.9%，法语+3.6%）。

**[Towards Geo-Culturally Grounded LLM Generations](llm_nlp/geocultural_grounded_llm.md)**

:   本文系统评估了知识库增强（KB grounding）和搜索增强（search grounding）两种RAG策略对LLM文化感知能力的影响，发现搜索增强显著提升了命题性文化知识但加剧了刻板印象风险，且两种策略均未能改善人类评估中的文化流利度。

**[Geometric Signatures of Compositionality Across a Language Model's Lifetime](llm_nlp/geometric_compositionality_lifetime.md)**

:   通过将数据集的组合性程度与语言模型表示的非线性内在维度(I_d)和线性有效维度(d)联系起来，揭示了一个形式-意义二分：非线性 I_d 编码有意义的组合语义复杂度，而线性 d 编码表面词形复杂度；该对应关系在训练过程中随语言能力涌现而建立。

**[GIFT-SW: Gaussian Noise Injected Fine-Tuning of Salient Weights for LLMs](llm_nlp/gift-sw_gaussian_noise_injected_fine-tuning_of_salient_weights_for_llms.md)**

:   提出 GIFT-SW，一种新型参数高效微调方法：仅更新权重矩阵中的"显著列"(salient columns)，同时对非显著列注入高斯噪声，在相同计算预算下超越全参微调和 LoRA/DoRA 等现代 PEFT 方法。

**[Efficient Universal Goal Hijacking with Semantics-guided Prompt Organization](llm_nlp/goal_hijacking_attack.md)**

:   本文提出POUGH方法，通过高效的渐进式优化算法和两种语义引导的提示组织策略（采样策略+排序策略），实现了对LLM的高效通用目标劫持攻击，在四个开源LLM和十种恶意目标响应上平均攻击成功率达93.41%。

**[What Makes a Good Natural Language Prompt?](llm_nlp/good_natural_language_prompt.md)**

:   通过元分析150+篇prompting文献，提出包含6个维度21个属性的以属性为中心的prompt质量评估框架，并通过推理任务实验发现：单属性增强常常优于多属性组合，且在属性增强数据上微调可进一步提升模型推理能力。

**[GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](llm_nlp/gorp_continual_gradient_projection.md)**

:   GORP 提出将全秩参数和 LoRA 低秩参数的梯度统一投影到低秩梯度子空间中联合更新，利用 Adam 一阶矩隐式构建跨任务共享梯度空间来缓解灾难性遗忘，在 T5 和 LLaMA2 上持续学习性能接近多任务联合训练上界。

**[GradOT: Training-free Gradient-preserving Offsite-tuning for Large Language Models](llm_nlp/gradot_offsite_tuning.md)**

:   从优化理论角度首次系统分析 Offsite-tuning 问题，提出梯度保持压缩分数（GCS），并基于此设计了 GradOT 方法，对 MHA 使用动态秩分解（DRD）、对 MLP 使用选择性通道剪枝（SCP），在免训练条件下同时实现性能保持和隐私保护。

**[Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning](llm_nlp/graph_counselor_multiagent_graphrag.md)**

:   Graph Counselor 提出了一个多智能体协作的 GraphRAG 推理框架，通过 Planning/Thought/Execution 三个 Agent 自适应提取图结构信息，并引入多视角自反思机制纠正推理偏差，在多个图推理任务上超越现有方法。

**[Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?](llm_nlp/graph_descriptive_order_llm.md)**

:   首次系统研究了图描述顺序（BFS、DFS、PageRank、PPR）对LLM解决图推理问题的影响，发现有序描述显著优于随机描述，且不同任务偏好不同的排列策略。

**[HFT: Half Fine-Tuning for Large Language Models](llm_nlp/hft_half_fine-tuning_for_large_language_models.md)**

:   本文提出Half Fine-Tuning (HFT)，在每轮微调中随机冻结一半参数、只更新另一半，不改变模型架构的情况下显著缓解灾难性遗忘问题，同时在下游任务上取得与FFT相当甚至更好的性能，并减少约30%的训练时间。

**[Hierarchical Retrieval with Evidence Curation for Open-Domain Financial QA](llm_nlp/hierarchical_retrieval_with_evidence_curation_for_open-domain_financial_question.md)**

:   HiREC 提出分层检索与证据策展框架，先检索相关文档再从中选取段落，并通过过滤无关段落 + 自动生成补充查询来补全缺失信息，在包含 14.5 万篇 SEC 文档的 LOFin 基准上相比最优 RAG 基线提升 13%+ 答案准确率。

**[How LLMs Comprehend Temporal Meaning in Narratives: A Case Study in Cognitive Evaluation of LLMs](llm_nlp/how_llms_comprehend_temporal_meaning_in_narratives_a_case_study_in_cognitive_eva.md)**

:   构建 Expert-in-the-Loop 探测管线，通过真值判断/词语补全/开放式因果提问三组认知语言学实验（16篇叙事×30种prompt变体×7个LLM），系统评估LLM对叙事中语法体貌（perfective vs imperfective）的理解能力，发现LLM在非原型体貌条件下准确率仅18%（人类71%），且缺乏远距因果推理能力。

**[How Numerical Precision Affects Arithmetical Reasoning Capabilities of LLMs](llm_nlp/how_numerical_precision_affects_arithmetical_reasoning_capabilities_of_llms.md)**

:   从电路复杂度理论出发，严格证明低精度（如 int4/int8）Transformer 在迭代加法和整数乘法上需要超多项式规模才能求解，而标准精度（float32）Transformer 仅需常数深度+多项式宽度即可高效求解三类算术任务，并在 LLaMA-3.1-8B 上实验验证了精度对算术能力的关键影响。

**[How to Enable Effective Cooperation Between Humans and NLP Models: A Survey of Principles, Formalizations, and Beyond](llm_nlp/human_nlp_cooperation_survey.md)**

:   首次系统综述人-模型合作（Human-Model Cooperation）的原则、形式化分类和开放挑战，提出基于"谁做最终决策"的三类合作范式分类法（序列/分诊/联合合作），为每种范式梳理角色框架和方法路线。

**[HumT DumT: Measuring and Controlling Human-like Language in LLMs](llm_nlp/humt_dumt_measuring_and_controlling_human-like_language_in_llms.md)**

:   提出基于 GPT-2 对数概率比的文本人类化语气度量 HumT 及其社会感知泛化版 SocioT，在 40 万+偏好样本上发现用户普遍偏好更低人类化的 LLM 输出且人类化语气与社交亲近（r=0.87）、低地位（r=-0.80）、女性化（r=0.47）强相关，进而通过仅 500 对偏好数据的 DPO 微调（DumT）有效降低人类化程度而不损模型性能。

**[HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation](llm_nlp/hygenar_an_llm-driven_hybrid_genetic_algorithm_for_few-shot_grammar_generation.md)**

:   构建 540 个挑战的文法生成数据集，设计 6 种评测指标，提出基于 LLM 驱动的混合遗传算法 HyGenar，显著提升 LLM 从少量示例生成 BNF 文法的能力。

**[Enhancing the Rule Learning Ability of Large Language Model Agent through Induction, Deduction, and Abduction](llm_nlp/idea_enhancing_the_rule_learning_ability_of_large_language_model_agent_through_i.md)**

:   提出 RULEARN benchmark（300 个手工交互式文字环境谜题，涵盖三类场景）和 IDEA 框架（溯因假设生成→演绎计划验证→归纳反馈修正的迭代循环），在 GPT-4o 上达到 50.33% 成功率（+7% vs ReAct baseline），但仍远低于人类 63.33%，细粒度人类评估揭示了 LLM 在假设修正阶段的根本瓶颈。

**[If Eleanor Rigby Had Met ChatGPT: A Study on Loneliness in a Post-LLM World](llm_nlp/if_eleanor_rigby_had_met_chatgpt_a_study_on_loneliness_in_a_post-llm_world.md)**

:   对 79,951 条 ChatGPT 对话（WildChat 数据集）进行定性和定量分析，研究孤独用户如何使用 LLM 服务，发现孤独用户对话更长（12 vs 5 轮）且 37% 在寻求建议/倾听，但 ChatGPT 在自杀意念等严重场景中回应不当，且孤独对话的有毒内容高达 55%（主语料 20%），其中女性被攻击概率是男性的 22 倍。

**[The Impossibility of Fair LLMs](llm_nlp/impossibility_fair_llms.md)**

:   系统分析了四种主流技术公平性框架（FTU、多方公平、群体公平/公平表示、公平组合）在通用LLM场景下均存在固有不可克服挑战，论证了严格意义上的公平LLM在原理层面不可行，并提出了三条务实的前进方向。

**[Improve Language Model and Brain Alignment via Associative Memory](llm_nlp/improve_language_model_and_brain_alignment_via_associative_memory.md)**

:   通过模拟联想记忆对文本进行数据增强，以及对 LLM 进行联想记忆指令微调，本文证明两种方式均能显著提升语言模型与人脑在语音理解任务中的对齐程度，尤其在内侧颞叶等联想记忆相关脑区。

**[Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization](llm_nlp/improving_contextual_faithfulness_of_large_language_models_via_retrieval_heads-i.md)**

:   本文发现LLM中的"检索头"（retrieval heads）与长文本问答的上下文忠实度高度相关，据此提出Rhio框架：通过遮蔽检索头生成不忠实样本、引入控制令牌进行忠实度感知调优、再利用对比解码增强忠实度，在7B和13B模型上均超越GPT-4o。

**[Improving Preference Extraction In LLMs By Identifying Latent Knowledge Through Classifying Probes](llm_nlp/improving_preference_extraction_in_llms_by_identifying_latent_knowledge_through_.md)**

:   本文提出使用线性分类探针（classifying probes）结合对比对（contrast pairs）来提取LLM的隐含偏好判断，在LLM-as-Judge任务中持续优于传统的生成式评估方法，且监督探针甚至超越微调评估器，同时保持类似的计算成本。

**[InductionBench: LLMs Fail in the Simplest Complexity Class](llm_nlp/inductionbench_llms_fail_in_the_simplest_complexity_class.md)**

:   本文提出 InductionBench，一个基于子正则函数层次（subregular hierarchy）的归纳推理基准，揭示即使是最强的 LLM（如 o3-mini）也难以掌握最简单复杂度类的归纳推理任务，暴露了当前 LLM 在从观测数据中归纳规则方面的根本缺陷。

**[InfiniSST: Simultaneous Translation of Unbounded Speech with Large Language Model](llm_nlp/infinisst_simultaneous_translation_of_unbounded_speech_with_large_language_model.md)**

:   提出 InfiniSST，将无界流式语音同声传译建模为 LLM 多轮对话任务，结合鲁棒片段训练数据构造、多延迟增强策略和 Λ-shaped KV cache 管理，在 MuST-C En-Es/De/Zh 三个方向上将计算感知延迟降低 0.5-1 秒而不损失翻译质量。

**[Information Locality as an Inductive Bias for Neural Language Models](llm_nlp/information_locality_as_an_inductive_bias_for_neural_language_models.md)**

:   本文提出 $m$-local entropy 这一信息论度量来量化语言的局部不确定性，通过在扰动自然语言和概率有限状态自动机（PFSA）定义的语言上的实验，证明了具有更高 $m$-local entropy 的语言对 Transformer 和 LSTM 语言模型来说更难学习，揭示了神经语言模型像人类一样对语言的局部统计结构高度敏感。

**[Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs](llm_nlp/input_dependent_soft_prompting.md)**

:   提出 ID-SPAM，通过在输入 token 嵌入上施加可学习自注意力层并经瓶颈 MLP 生成**输入依赖**的软提示，仅在单层 Transformer 输入端拼接即可超越多种 Soft Prompt 基线，且具备优秀的零样本跨任务/跨领域迁移能力。

**[INTERACT: Enabling Interactive, Question-Driven Learning in Large Language Models](llm_nlp/interact_enabling_interactive_question-driven_learning_in_large_language_models.md)**

:   提出INTERACT框架，通过模拟师生对话让"学生"LLM通过主动提问向"教师"LLM学习新概念，在1,347个未见过的上下文上实验证明交互式学习最高可提升25%的理解准确率，且仅需5轮对话即可匹配静态学习基线。

**[Interactive and Expressive Code-Augmented Planning with Large Language Models](llm_nlp/interactive_and_expressive_code-augmented_planning_with_large_language_models.md)**

:   本文提出REPL-Plan，通过让LLM与扩展的REPL（Read-Eval-Print Loop）交互，实现既能充分利用代码表达力又能动态纠错和处理模糊子问题的自顶向下规划方法，在ALFWorld、WebShop和真实网页导航任务中取得强劲表现。

**[Internal and External Impacts of Natural Language Processing Papers](llm_nlp/internal_and_external_impacts_of_natural_language_processing_papers.md)**

:   从内部（学术引用）和外部（专利、媒体、政策文档）两个维度系统分析 1979-2024 年 ACL/EMNLP/NAACL 论文的影响力，发现语言建模主题影响力最广，伦理公平主题在政策文档中影响力突出但学术引用较低，且多维外部影响力可高效预测内部高被引论文。

**[Investigating Context-Faithfulness in Large Language Models: The Roles of Memory Strength and Evidence Style](llm_nlp/investigating_context-faithfulness_in_large_language_models_the_roles_of_memory_.md)**

:   通过测量 LLM 对同一问题不同释义的回答一致性来量化"记忆强度"，发现 LLM 对外部证据的接受度与记忆强度高度负相关，且改写式证据比重复或详细证据更有效。

**[IPO: Your Language Model is Secretly a Preference Classifier](llm_nlp/ipo_your_language_model_is_secretly_a_preference_classifier.md)**

:   提出隐式偏好优化（IPO），利用生成式LLM自身作为偏好分类器（通过"Yes/No"token的概率），替代外部奖励模型来获取偏好信号，从而实现低成本的自对齐训练。

**[Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs](llm_nlp/is_it_just_semantics_a_case_study_of_discourse_particle_understanding_in_llms.md)**

:   以英语多义语气词 "just" 为案例，通过专家构造数据集和电影字幕标注数据，用两种元语言实验（few-shot 语义标注和成对比较）系统评估 LLM 对语气词细粒度语义的理解能力，发现模型能区分大类（形容词、时间义）但无法充分捕捉语气词的微妙语义差异（排他、轻描淡写、无因、强调义）。

**[JoPA: Explaining Large Language Model's Generation via Joint Prompt Attribution](llm_nlp/jopa_explaining_large_language_models_generation_via_joint_prompt_attribution.md)**

:   提出 JoPA（Joint Prompt Attribution）框架，将 LLM 生成任务的 prompt 归因建模为组合优化问题，用概率搜索算法高效寻找对输出有因果影响的输入 token 组合，解决了现有方法忽略 token 间协同效应的问题。

**[Just a Scratch: Enhancing LLM Capabilities for Self-Harm Detection through Intent Refinement](llm_nlp/just_a_scratch_enhancing_llm_capabilities_for_self-harm_detection_through_intent.md)**

:   提出 SHINES 数据集和 CESM-100 表情符号矩阵，通过区分社交媒体帖子中自伤表达的"随意提及"和"严重意图"，结合表情符号语境解读和多任务微调，将 LLM 自伤检测 F1 从 0.74（zero-shot）提升至 0.88（多任务+CESM-100），同时生成可解释的预测理由。

**[KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan](llm_nlp/kazmmlu_evaluating_language_models_on_kazakh_russian_and_regional_knowledge_of_k.md)**

:   提出 KazMMLU，首个专为哈萨克斯坦设计的 MMLU 风格双语（哈萨克语+俄语）评估基准，包含 23,000 道来自真实教育材料的多选题，覆盖 STEM、人文、社会科学等多学科多教育层次，评估了 27 个多语言 LLM，揭示了当前模型在哈萨克语上的显著不足。

**[Knockout LLM Assessment: Using Large Language Models for Evaluations through Iterative Pairwise Comparisons](llm_nlp/knockout_llm_assessment_using_large_language_models_for_evaluations_through_iter.md)**

:   提出 Knockout Assessment——基于淘汰赛制的迭代成对比较 LLM-as-a-Judge 方法，通过多轮锦标赛让回答之间反复对比以建立全局排名视角，在科学考试评分和机器翻译评估上比个体评估方法平均提升 0.07 Pearson 相关系数。

**[Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations](llm_nlp/knowledge_boundary_crosslingual.md)**

:   通过探测 LLM 内部表示，揭示知识边界认知在多语言间呈线性结构，提出 training-free 对齐方法实现跨语言知识边界感知迁移，并发现"弱到强泛化"现象。

**[Knowledge Boundary of Large Language Models: A Survey](llm_nlp/knowledge_boundary_survey.md)**

:   提出LLM知识边界的形式化定义框架——三层嵌套边界（Outward⊂Parametric⊂Universal）和四类知识分类（PAK/PSK/MSU/MAU），围绕"为何/如何识别/如何缓解"三个问题系统综述相关研究。

**[Language-Codec: Bridging Discrete Codec Representations and Speech Language Models](llm_nlp/language_codec_bridging_discrete_codec_speech_language_models.md)**

:   提出 Language-Codec，通过掩码通道残差向量量化（MCRVQ）机制和改进的傅里叶变换解码器，弥合离散编解码器表示与下游语音语言模型之间的鸿沟，仅用4个码本通道即实现高质量音频重建。

**[Language Model Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions](llm_nlp/language_model_fine-tuning_on_scaled_survey_data_for_predicting_distributions_of.md)**

:   提出直接在大规模公众意见调查数据（SubPOP，含 3362 道题目、70K 子群体-响应对）上微调 LLM，使其预测不同人口统计子群体的意见分布，相比 prompt engineering 基线将 Wasserstein 距离降低 32-46%，且泛化到未见过的调查和子群体。

**[Language Models Grow Less Humanlike beyond Phase Transition](llm_nlp/language_models_grow_less_humanlike_beyond_phase_transition.md)**

:   本文发现语言模型在预训练过程中与人类阅读行为的对齐（PPP）经历先升后降的拐点现象，通过关联和因果实验证明此拐点由预训练中的相变（specialized attention heads 的快速涌现）导致，且相变并非直接产生有害的注意力模式，而是改变了模型后续的学习动态使其持续偏离人类。

**[Large Language Models are Good Relational Learners](llm_nlp/large_language_models_are_good_relational_learners.md)**

:   提出 Rel-LLM 框架，利用 GNN 编码器从关系数据库中提取结构化子图表示，将其作为软提示注入冻结的 LLM，在 RelBench 基准上实现了关系深度学习（RDL）任务的 SOTA 性能，并支持零样本预测。

**[Large Language Models for Predictive Analysis: How Far Are They?](llm_nlp/large_language_models_for_predictive_analysis_how_far_are_they.md)**

:   提出 PredictiQ 基准——首个系统评估 LLM 预测分析能力的综合框架，整合 8 个领域 44 个真实数据集和 1130 条专家设计的查询，从文本分析、代码生成、文本-代码对齐三个维度七个方面评估 12 个主流 LLM，揭示即使最强的 GPT4O3Mini 在深度分析（2.91/4）和数据预处理（51%缺失）上仍存在显著不足。

**[Large Language Models in Bioinformatics: A Survey](llm_nlp/large_language_models_in_bioinformatics_a_survey.md)**

:   本文系统综述了大语言模型在生物信息学四大领域（DNA/基因组、RNA、蛋白质、单细胞分析）的应用进展，涵盖 30+ 代表性模型的架构、任务和数据集，并讨论了数据稀缺、计算复杂度、跨组学整合等核心挑战和未来方向。

**[LazyReview: A Dataset for Uncovering Lazy Thinking in NLP Peer Reviews](llm_nlp/lazyreview_peer_review.md)**

:   构建了首个包含 500 条专家标注 + 1276 条银标注的 NLP 同行评审"懒惰思维"细粒度分类数据集 LazyReview，通过三轮迭代标注协议和正例增强将标注一致性翻倍，并证明在该数据集上指令微调 LLM 可将检测性能提升 10-20 个百分点，最终的控制实验表明懒惰思维反馈能显著改善评审质量。

**[Learning from Litigation: Graphs and LLMs for Retrieval and Reasoning in eDiscovery](llm_nlp/learning_from_litigation_graphs_and_llms_for_retrieval_and_reasoning_in_ediscove.md)**

:   本文提出DISCOG（DISCOvery Graph）系统，将知识图谱与LLM驱动推理相结合用于法律电子取证（eDiscovery）中的文档检索和分类，在平衡和不平衡数据集上均超越强基线，实际部署中将诉讼文档审查成本降低约98%。

**[Length Controlled Generation for Black-box LLMs](llm_nlp/length_controlled_generation_for_black-box_llms.md)**

:   提出基于 Metropolis-Hastings 算法的迭代采样框架，结合重要性采样加速策略，在**不修改模型参数**的前提下实现黑盒 LLM 的精确长度控制，在 Llama3.1 上达到**100%**的长度控制成功率，最多仅需 5 次迭代，且不损害生成质量。

**[LESA: Learnable LLM Layer Scaling-Up](llm_nlp/lesa_learnable_llm_layer_scaling-up.md)**

:   提出 LESA，一种基于 SVD 发现层间潜在模式并通过神经网络预测中间层参数的可学习深度扩展方法，相比启发式层复制方法获得更好的初始化和更快的收敛速度，训练成本降低一半以上。

**[Leveraging Human Production-Interpretation Asymmetries to Test LLM Cognitive Plausibility](llm_nlp/leveraging_human_production-interpretation_asymmetries_to_test_llm_cognitive_pla.md)**

:   本文利用人类在隐含因果动词（implicit causality verbs）上"代词产出"与"代词解读"之间的已知不对称现象作为测试平台，系统评估了指令微调LLM是否能复现这种人类认知不对称性，发现模型大小和元语言提示的选择是决定性因素。

**[Leveraging In-Context Learning for Political Bias Testing of LLMs](llm_nlp/leveraging_in-context_learning_for_political_bias_testing_of_llms.md)**

:   本文提出"问卷建模"（Questionnaire Modeling, QM）这一新探测任务，利用人类调查数据作为上下文示例来改善LLM政治偏见检测的稳定性，发现指令微调可以改变偏见方向，且更大的模型能更有效地利用上下文示例并展现更小的偏见分数。

**[Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora](llm_nlp/leveraging_large_language_models_to_measure_gender_representation_bias_in_gender.md)**

:   提出利用LLM的语境理解能力来检测和量化有语法性别语言（如西班牙语、巴伦西亚语）训练语料中的性别表征偏差（representation bias），发现严重的男性主导不平衡，并验证了通过反向偏差数据进行持续预训练可有效缓解模型输出偏差。

**[ReCall: Library-Like Behavior In Language Models is Enhanced by Self-Referencing Causal Cycles](llm_nlp/library-like_behavior_in_language_models_is_enhanced_by_self-referencing_causal_.md)**

:   提出"自引用因果循环"（ReCall）概念，揭示 LLM 预训练数据中自然存在的重复 token 序列如何形成循环引用，使自回归模型能够绕过单向因果限制、克服逆向诅咒（reversal curse），并据此设计了两步 ReCall-aware prompting 策略。

**[On the Limit of Language Models as Planning Formalizers](llm_nlp/limit_llm_planning_formalizer.md)**

:   系统评估"LLM-as-Formalizer"方法论的极限——首次要求 LLM 生成完整 PDDL 表示（而非部分），从不同自然度的文本描述中形式化规划领域，发现最强模型（GPT-4o/o3-mini/DeepSeek-R1）可有效形式化超越直接规划，但描述越自然性能越低，弱模型卡在语法错误而强模型面临语义错误。

**[LlamaDuo: LLMOps Pipeline for Seamless Migration from Service LLMs to Small-Scale Local LLMs](llm_nlp/llamaduo_llmops_pipeline_for_seamless_migration_from_service_llms_to_small-scale.md)**

:   提出 LlamaDuo 自动化 LLMOps 流水线，通过服务 LLM 生成合成数据迭代微调小模型，使 2B-8B 本地模型在特定下游任务上逼近甚至匹敌 GPT-4o 等大模型性能，且长期部署成本显著降低。

**[LLM as Effective Streaming Processor: Bridging Streaming-Batch Mismatches with Group Position Encoding](llm_nlp/llm_as_effective_streaming_processor_bridging_streaming-batch_mismatches_with_gr.md)**

:   系统性地识别并量化了 batch-trained LLM 适配流式场景的三种不匹配（输入注意力 / 输出注意力 / 位置 ID），发现仅输入注意力不匹配才是关键瓶颈（+2.20 BLEU），据此提出组位置编码（Group Position Encoding）——源/目标各自维护连续位置 ID 即可，无需昂贵的 KV cache 重编码，在机器翻译和 ASR 两种跨模态任务上均超越专用流式架构。

**[LLM Braces: Straightening Out LLM Predictions with Relevant Sub-Updates](llm_nlp/llm_braces_straightening.md)**

:   LLMBraces 通过计算 FFN 层中各 value 向量与输入的相关性得分，动态调节子更新（sub-update）的贡献权重，用极少参数（比 LoRA 少 75%）同时提升模型预测精度和实现可控文本生成。

**[LLM as a Broken Telephone: Iterative Generation Distorts Information](llm_nlp/llm_broken_telephone.md)**

:   以翻译为测试床模拟 LLM 的"传话游戏"，发现信息在 100 次迭代翻译后严重失真——一辆卡车司机的罚款新闻经 100 轮英泰互译后变成"小汽车获得赔偿后发生爆炸"，而中间语言的选择、链条复杂度和解码温度是失真速度的关键调控因素。

**[LLM×MapReduce: Simplified Long-Sequence Processing using Large Language Models](llm_nlp/llm_mapreduce_simplified_long_sequence_processing.md)**

:   提出 LLM×MapReduce，一个无需训练的分治框架，通过结构化信息协议和上下文内置信度校准机制解决长文本分块后的跨块依赖和跨块冲突问题，使 8K 上下文的 LLM 能有效处理超过 100K 甚至 1280K tokens 的长文本，性能超越 GPT-4 等长上下文模型。

**[LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs?](llm_nlp/llm_meets_scene_graph_can_large_language_models_understand_and_generate_scene_gr.md)**

:   提出 TSG Bench 基准，系统评估 11 个 LLM 在场景图理解和生成任务上的能力，揭示 LLM 在场景图生成（尤其是多动作分解）方面存在显著瓶颈。

**[Wait, that's not an option: LLMs Robustness with Incorrect Multiple-Choice Options](llm_nlp/llm_robustness_incorrect_mcq.md)**

:   提出"反思判断力"（Reflective Judgment）概念来衡量 LLM 在所有选项都错误的选择题中拒绝选择的能力，发现对齐后的模型（GPT-4o 等）往往盲目服从指令选择错误选项，而基座模型反而表现更好，且该能力随模型规模增大而涌现。

**[LLM-Powered Test Case Generation for Detecting Bugs in Plausible Programs](llm_nlp/llm_test_case_gen_bugs.md)**

:   本文提出TrickCatcher，利用LLM生成程序变体和测试输入生成器，结合diversity-driven差分测试来检测通过现有测试套件但仍含隐蔽bug的"plausible programs"，在Recall/Precision/F1上分别达到SOTA的1.80×/2.65×/1.66×。

**[LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks](llm_nlp/llm_vs_human_judges_study.md)**

:   构建包含20个NLP数据集（7万+实例）的 Judge-Bench 基准，系统评估11个LLM作为评判者与人类标注的一致性，发现模型在不同任务/属性/标注者专业水平上表现差异巨大，建议部署前必须针对特定任务做人类标注验证。

**[LLMs can Perform Multi-Dimensional Analytic Writing Assessments](llm_nlp/llm_writing_assessment.md)**

:   利用 L2 研究生文献综述语料库，系统评估了 LLM 在多维分析写作评估（评分+评论）上的能力，并提出可解释的反馈质量评估框架 ProEval。

**[LLMs Can Be Easily Confused by Instructional Distractions](llm_nlp/llms_can_be_easily_confused_by_instructional_distractions.md)**

:   本文发现 LLM 在处理"输入本身也像指令"的场景时会被严重误导（指令干扰），提出 DIM-Bench 基准系统评估该问题，实验证明包括 GPT-4o 在内的主流 LLM 均显著受影响，且现有提示策略无法根本解决。

**[LLMs Know Their Vulnerabilities: Uncover Safety Gaps through Natural Distribution Shifts](llm_nlp/llms_know_their_vulnerabilities_uncover_safety_gaps_through_natural_distribution.md)**

:   提出 ActorBreaker 多轮攻击方法，基于 Latour 的行动者网络理论，利用与有害内容语义相关的良性 prompt（自然分布偏移）绕过安全机制，在 HarmBench 上达到 SOTA 攻击成功率，揭示了预训练数据与安全训练数据之间的语义覆盖差距。

**[LLMs + Persona-Plug = Personalized LLMs](llm_nlp/llms_persona-plug_personalized_llms.md)**

:   提出 PPlug 模型，通过轻量级插件式用户嵌入器将用户历史行为压缩为单一个性化嵌入，以 plug-and-play 方式引导 LLM 生成个性化输出，在 LaMP 基准上显著超越检索式和微调式基线，最高提升 35.8%。

**[Language Models, Graph Searching, and Supervision Adulteration: When More Supervision is Less and How to Make More More](llm_nlp/lm_graph_search_supervision.md)**

:   本文证明了 path-star 图搜索任务在 decoder-only LM 上的失败并非 next-token prediction 范式的根本缺陷，而是由"监督污染"（supervision adulteration）导致的——过量的 teacher-forcing 监督信号诱导模型学到 Clever Hans Cheat 捷径，阻碍了子任务分解；通过 token masking、ranking-into-the-future、scratchpad、树形拓扑等六种正交方法均可使任务可学。

**[Locate-and-Focus: Enhancing Terminology Translation in Speech Language Models](llm_nlp/locateandfocus_enhancing_terminology_translation_in_speech.md)**

:   提出Locate-and-Focus方法用于语音LLM的术语翻译：先用滑动窗口检索定位语音中包含术语的片段，再通过音频替换和Tag Cue引导模型聚焦翻译知识，在英中/英德方向上术语翻译成功率大幅提升。

**[Logical Forms Complement Probability in Understanding Language Model (and Human) Performance](llm_nlp/logical_forms_complement_probability_in_understanding_language_model_and_human_p.md)**

:   系统研究 LLM 在命题逻辑和模态逻辑推理上的能力，发现除了输入概率（perplexity）外，逻辑形式（modality、argument form）是预测 LLM 表现的重要互补因素，并通过人类行为数据对比揭示人机推理的异同。

**[LongDPO: Unlock Better Long-form Generation Abilities for LLMs via Critique-augmented Stepwise Information](llm_nlp/longdpo_unlock_better_long-form_generation_abilities_for_llms_via_critique-augme.md)**

:   提出 LongDPO，通过 MCTS 收集步级偏好对、全局记忆池维护事实一致性、critique 增强低质量候选，再用步级 DPO 进行细粒度优化，在 LongBench-Write 上显著提升长文本生成质量，同时保持通用能力。

**[Lost in Literalism: How Supervised Training Shapes Translationese in LLMs](llm_nlp/lost_in_literalism_how_supervised_training_shapes_translationese_in_llms.md)**

:   本文系统研究了大语言模型在机器翻译中产生翻译腔（translationese）的现象，揭示了监督微调（SFT）数据中的翻译腔偏差是导致LLM翻译不自然的根本原因，并提出了通过润色训练参考译文和过滤不自然训练实例来缓解翻译腔的方法。

**[LR²Bench: Evaluating Long-chain Reflective Reasoning Capabilities of Large Language Models via Constraint Satisfaction Problems](llm_nlp/lr2bench_evaluating_long-chain_reflective_reasoning_capabilities_of_large_langua.md)**

:   提出 LR²Bench 基准，通过六类约束满足问题（CSP）系统评测 LLM 的长链反思推理能力，发现即使最先进的推理模型如 DeepSeek-R1 和 o1-preview 的平均 Exact Match 也仅为 20.0% 和 23.6%，揭示了当前模型在反思推理方面的巨大改进空间。

**[Mapping 1,000+ Language Models via the Log-Likelihood Vector](llm_nlp/mapping_1000_models_loglikelihood.md)**

:   提出用对数似然向量（log-likelihood vector）将 1000+ 语言模型映射到一个统一空间，证明向量间欧氏距离近似 KL 散度，可实现模型聚类可视化、基准性能预测（r=0.96）和数据泄漏检测。

**[MAPS: Motivation-Aware Personalized Search via LLM-Driven Consultation Alignment](llm_nlp/maps_personalized_search.md)**

:   首次建模电商搜索中的"搜索动机"——用户在搜索前的咨询行为蕴含的真实需求，提出MAPS框架融合LLM语义、MoAE池化和双重对齐机制，在真实商业数据上HR@10提升24.4%（从0.5685到0.7071）。

**[Masking in Multi-hop QA: How LMs Perform with Context Permutation](llm_nlp/masking_in_multi-hop_qa_an_analysis_of_how_language_models_perform_with_context_.md)**

:   通过系统性的文档排列实验和注意力权重分析，揭示因果掩码是 decoder-only LLM 在多跳问答中的结构性瓶颈，并证明将因果掩码替换为 prefix mask 可显著提升性能和鲁棒性。

**[MasRouter: Learning to Route LLMs for Multi-Agent Systems](llm_nlp/masrouter_learning_to_route_llms_for_multi-agent_systems.md)**

:   首次提出多智能体系统路由（MASR）问题，设计 MasRouter 级联控制器网络，依次决定协作模式、角色分配和 LLM 路由，在保持高性能的同时将 MAS 的推理成本降低最高 52%，实现效果与效率的平衡。

**[MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion](llm_nlp/mathfusion_instruction_fusion.md)**

:   提出 MathFusion 框架，通过三种问题融合策略（顺序/并行/条件融合）将数学问题两两合成新问题，仅用 45K 额外合成数据就在多个基准上实现平均 18 个百分点的数学推理提升。

**[Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes](llm_nlp/mathneuro_math_reasoning_isolation.md)**

:   提出 MathNeuro，一种仅需前向传播的计算高效方法，通过过滤掉对通用语言任务同样重要的参数来定位 LLM 中数学推理专属的参数，剪枝这些参数可删除数学能力，缩放这些参数可提升 4-35% 的数学性能。

**[Meaning Beyond Truth Conditions: Evaluating Discourse Level Understanding via Anaphora Accessibility](llm_nlp/meaning-beyond-truth-conditions-anaphora-accessibility.md)**

:   本文提出自然语言理解能力的三层次层级体系（词汇/句子/篇章），以回指可及性（anaphora accessibility）作为篇章级理解的诊断任务，通过动态语义学启发的评估数据集系统考察了 LLM 在全称量词、否定和析取三种语言结构下的篇章理解能力。

**[MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](llm_nlp/membench_towards_more_comprehensive_evaluation_on_the_memory_of_llm-based_agents.md)**

:   构建了首个同时覆盖参与/观察两种场景、事实/反思两种记忆层次、准确性/召回/容量/效率四种指标的 LLM Agent 记忆能力评估基准 MemBench，在 7 种记忆机制上的评测显示简单的 RetrievalMemory 在大规模记忆（100K token）下表现最佳（准确率 0.833），而复杂机制（MemGPT、GenerativeAgent）未展现优势。

**[MEraser: An Effective Fingerprint Erasure Approach for Large Language Models](llm_nlp/meraser_fingerprint_erasure.md)**

:   提出 MEraser（Mismatched Eraser），通过两阶段微调策略（错配数据擦除 + 干净数据恢复）以不到 1000 条样本完全移除 LLM 中基于后门的指纹水印，同时保持模型性能，并首创可迁移的 LoRA 擦除适配器。

**[MergePrint: Merge-Resistant Fingerprints for Robust Black-box Ownership Verification of Large Language Models](llm_nlp/mergeprint_fingerprint_ownership.md)**

:   提出 MergePrint，首个针对模型合并（model merging）场景的 LLM 黑盒指纹验证方法，通过伪合并模型模拟合并行为并两阶段优化（输入优化 + 参数优化），使嵌入的指纹在合并后仍可被检测，实现高效、无害、抗篡改的所有权验证。

**[MExGen: Multi-Level Explanations for Generative Language Models](llm_nlp/mexgen_multi_level_explanations.md)**

:   提出MExGen框架，通过scalarizer将生成模型的文本输出映射为实数值、多粒度语言分割和线性复杂度归因算法（C-LIME/L-SHAP），为上下文驱动的文本生成（摘要、QA）提供比PartitionSHAP和LLM自解释更忠实的输入归因解释。

**[MHA2MLA: Towards Economical Inference by Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](llm_nlp/mha2mla_deepseek_latent_attention.md)**

:   MHA2MLA 首次提出将已训练好的 MHA 模型高效迁移到 DeepSeek 的 MLA 架构的方法，通过贡献度感知的 partial-RoPE 移除和联合 SVD 低秩近似，仅用 0.6%-1% 的训练数据即可恢复性能，将 Llama2-7B 的 KV cache 压缩 92.19% 且 LongBench 性能仅下降 1%。

**[Mind the (Belief) Gap: Group Identity in the World of LLMs](llm_nlp/mind_the_belief_gap_group_identity_in_the_world_of_llms.md)**

:   通过多智能体 LLM 框架模拟信念一致性（Belief Congruence）理论，发现 LLM 表现出比人类更强的信念一致性偏向，导致错误信息传播增加和学习能力受损，并提出三种基于社会心理学的缓解策略。

**[Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy](llm_nlp/mind_your_tone_investigating_how_prompt_politeness_affects_llm_accuracy_short_pa.md)**

:   本文系统研究了提示语的礼貌程度对LLM回答准确率的影响，通过构建5种语气梯度（从"非常礼貌"到"非常粗鲁"）的250条多选题提示并在ChatGPT 4o上测试，发现与直觉相反——粗鲁提示的准确率（84.8%）显著高于礼貌提示（80.8%）。

**[MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments](llm_nlp/mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)**

:   本文提出 MIRAGE 评估框架，通过八个精心设计的剧本杀场景和四项指标（信任倾向 TII、线索调查 CIC、交互能力 ICI、剧本合规 SCI）系统评估 LLM 在复杂社交互动环境中的表现，发现即使 GPT-4 也在这些场景中面临严峻挑战。

**[Mitigate Position Bias in LLMs via Scaling a Single Hidden States Channel](llm_nlp/mitigate_position_bias_in_large_language_models_via_scaling_a_single_dimension.md)**

:   发现 LLM 隐状态中存在编码绝对位置信息的特定通道（positional hidden states），通过缩放这单一通道即可缓解"lost in the middle"位置偏差，在多文档 QA 基准上提升高达 15.2%，且不影响模型其他能力。

**[Mixtures of In-Context Learners](llm_nlp/mixtures_of_in-context_learners.md)**

:   提出 MoICL 方法，将 ICL 的 demonstration 集合划分为多个子集（专家），通过可学习的权重函数融合各专家的 next-token 分布，在不修改 LLM 参数的前提下显著提升 ICL 的准确率、鲁棒性和效率。

**[Comparing Moral Values in Western English-speaking Societies and LLMs with Word Associations](llm_nlp/moral_values_western.md)**

:   提出基于词语联想（word association）而非直接提问的 LLM 道德评估框架，构建人类和 LLM 的全局道德网络（GMN），发现两者在正面道德维度上高度一致，但 LLM 在负面道德概念上系统性地更抽象、更少情感化和具体性。

**[Multi-Prompting Decoder Helps Better Language Understanding](llm_nlp/multi-prompting_decoder_helps_better_language_understanding.md)**

:   提出 Multi-Prompting Decoder（MPD）框架，通过多提示查询 PLM 获取多组隐状态和类别分数，结合最优传输匹配和校准解码策略，在 MaaS（模型即服务）场景下的 few-shot 分类任务上显著超越现有方法。

**[Multi-Attribute Steering of Language Models via Targeted Intervention](llm_nlp/multi_attribute_steering.md)**

:   提出 MAT-Steer，通过属性感知的 token 级 gating 机制和正交性约束，实现推理时对 LLM 多属性（如真实性、毒性、偏见）的同时精准干预，在 QA 和生成任务上全面超越现有 ITI 和微调方法。

**[Which of These Best Describes Multiple Choice Evaluation with LLMs?](llm_nlp/multiple_choice_eval.md)**

:   系统论证 MCQA 作为 LLM 标准评估格式存在三大类问题：(1) 格式缺陷——无法测试生成/主观性、不匹配 LLM 真实使用场景、不能充分测试知识深度；(2) 数据集缺陷——泄露、不可回答、捷径和饱和；(3) 模型行为问题——鲁棒性差、选项偏置和不忠实解释。借鉴教育测试学提出 Constructed Response、Explanation MCQA、IRT 分析等系统化修复方案。

**[Natural Language Processing in Support of Evidence-based Medicine: A Scoping Review](llm_nlp/natural_language_processing_in_support_of_evidence-based_medicine_a_scoping_revi.md)**

:   基于 PRISMA 指南对 129 篇研究（2019-2024）进行范围综述，以 EBM 五步流程（Ask-Acquire-Appraise-Apply-Assess）为组织框架，全面梳理了 NLP 技术在循证医学中的应用现状、技术演进路径与未来方向。

**[NeKo: Cross-Modality Post-Recognition Error Correction with Tasks-Guided Mixture-of-Experts Language Model](llm_nlp/neko_cross-modality_post-recognition_error_correction_with_tasks-guided_mixture-.md)**

:   提出 NeKo，一种基于任务引导 Mixture-of-Experts (MoE) 的多任务后识别纠错语言模型，在 ASR、语音翻译、OCR 等多个跨模态纠错任务上达到 SOTA，零样本场景下超越 GPT-3.5 和 Claude-3.5 Sonnet。

**[Neural Topic Modeling with Large Language Models in the Loop](llm_nlp/neural_topic_modeling_with_large_language_models_in_the_loop.md)**

:   提出LLM-ITL框架，将LLM以"in-the-loop"方式集成到神经主题模型（NTM）训练中，通过基于最优传输的主题对齐目标和置信度加权机制，在保持文档表示质量和计算效率的同时显著提升主题可解释性。

**[NewsInterview: a Dataset and a Playground to Evaluate LLMs' Grounding Gap via Informational Interviews](llm_nlp/newsinterview_a_dataset_and_a_playground_to_evaluate_llms_grounding_gap_via_info.md)**

:   构建了 4 万条新闻采访对话数据集，发现 LLM 在采访场景中缺乏 acknowledgement（少 50%）和话题转换能力（少 30%），并设计了含说服机制的模拟博弈环境（NewsInterview），证明最优 LLM（gpt-4o）也仅能提取 50.4% 的信息项。

**[Not Quite Sherlock Holmes: Language Model Predictions Do Not Reliably Differentiate Impossible from Improbable Events](llm_nlp/not_quite_sherlock_holmes_language_model_predictions_do_not_reliably_differentia.md)**

:   本文通过精心设计的最小对实验，揭示语言模型无法可靠区分"不可能事件"和"不太可能但可能的事件"——在对抗性条件下（可能句含不相关词、不可能句含相关词），包括 Llama 3、Gemma 2、Mistral NeMo 在内的所有 35 个模型均表现低于随机水平。

**[Nudging: Inference-time Alignment of LLMs via Guided Decoding](llm_nlp/nudging_inference_time_alignment.md)**

:   提出 Nudging，一种免训练的推理时对齐算法，利用小型对齐模型在基础模型不确定时注入少量"nudging tokens"来引导输出，用 7-14 倍小的模型就能达到甚至超过大型对齐模型的性能。

**[OLMoTrace: Tracing Language Model Outputs Back to Trillions of Training Tokens](llm_nlp/olmotrace_tracing_language_model_outputs_back_to_trillions_of_training_tokens.md)**

:   提出OLMoTrace——首个能在实时（平均4.5秒）将语言模型输出逐字追溯到其完整多万亿token训练数据的系统，基于扩展的infini-gram引擎通过后缀数组索引实现高效精确匹配，支持事实核查、创意溯源和数学能力追踪等应用场景。

**[On Entity Identification in Language Models](llm_nlp/on_entity_identification_in_language_models.md)**

:   提出基于聚类的评估框架（Purity/Inverse Purity）分析 LLM 内部表示中的实体区分能力，发现实体信息在早期层（~归一化位置 0.2）的 20 维子空间中达到线性可分（F1~0.9），且不同大模型收敛到结构同构的实体编码——为"LLM 从纯文本训练中涌现离散知识结构"提供了系统性证据。

**[On the Acquisition of Shared Grammatical Representations in Bilingual Language Models](llm_nlp/on_the_acquisition_of_shared_grammatical_representations_in_bilingual_language_m.md)**

:   本文通过训练受控的小型双语语言模型，使用心理语言学中的结构启动方法（structural priming）研究跨语言语法表征的共享机制，发现跨语言结构启动效应在语言对之间呈不对称性，且对类型学距离较远的语言对（如英语-希腊语）明显减弱。

**[On the Mutual Influence of Gender and Occupation in LLM Representations](llm_nlp/on_the_mutual_influence_of_gender_and_occupation_in_llm_representations.md)**

:   通过在 LLM 嵌入空间中近似性别方向（gender direction），系统研究了名字的性别表征与职业上下文之间的双向影响：职业上下文会偏移名字的性别表征，而名字的性别表征反过来影响 LLM 在职业预测任务中的偏差行为，但二者的相关性仅为中等强度。

**[On the Risk of Evidence Pollution for Malicious Social Text Detection in the Era of LLMs](llm_nlp/on_the_risk_of_evidence_pollution_for_malicious_social_text_detection_in_the_era.md)**

:   本文系统研究了LLM时代下恶意社交文本检测中的"证据污染"风险，提出13种污染方法和3种防御策略，发现LLM生成的虚假证据可导致检测器性能下降高达14.4%，且现有防御策略面临实际部署挑战。

**[Open-Set Living Need Prediction with Large Language Models](llm_nlp/open-set_living_need_prediction_with_large_language_models.md)**

:   提出 PIGEON 系统，将生活服务平台上的用户需求预测从封闭集分类重新定义为开放集生成问题，通过 GNN 行为嵌入检索历史记录辅助 LLM 预测、马斯洛需求层次引导精化、以及微调文本嵌入模型实现灵活需求到服务的召回，在美团真实数据上平均提升 19.37%。

**[OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models](llm_nlp/opencoder_the_open_cookbook_for_top-tier_code_large_language_models.md)**

:   提出OpenCoder，一个完全开源的代码大语言模型（含1.5B和8B版本），不仅性能达到顶级水平，更作为"open cookbook"开放了可复现的数据处理流水线、预训练数据集、消融实验和训练协议，为代码智能研究提供基础设施。

**[P3: Prompts Promote Prompting](llm_nlp/p3_prompts_promote_prompting.md)**

:   本文提出 P3 框架，首次同时优化系统提示（system prompt）和用户提示（user prompt），通过离线迭代优化生成高质量的提示模板，再用这些模板进行在线 query-dependent 提示优化，在 Arena-Hard、AlpacaEval、GSM8K、GPQA 等通用和推理任务上均超过仅优化单侧提示的方法。

**[Palm: A Culturally Inclusive and Linguistically Diverse Dataset for Arabic LLMs](llm_nlp/palm_a_culturally_inclusive_and_linguistically_diverse_dataset_for_arabic_llms.md)**

:   由 44 名阿拉伯世界研究者历时一年社区驱动构建的 Palm 数据集，涵盖全部 22 个阿拉伯国家、20 个文化主题、10 种方言，共 17,411 条人工创建的指令对，用于评估和提升 LLM 的阿拉伯文化和方言能力。

**[Enhancing Open-Domain Task-Solving Capability of LLMs via Autonomous Tool Integration from GitHub](llm_nlp/paper_2312_17294.md)**

:   提出OpenAgent系统，通过Search→Setup→Apply→Store四阶段流程自主从GitHub搜索、配置、使用和存储仓库作为工具，解决LLM在金融、化学、生物等专业领域的开放域任务，平均成功率69.4%。

**[Past Meets Present: Creating Historical Analogy with Large Language Models](llm_nlp/past_meets_present_creating_historical_analogy_with_large_language_models.md)**

:   本文首次定义了"历史类比获取"任务，系统探索了基于LLM的检索和生成方法，并提出自反思机制来减轻LLM生成历史类比时的幻觉和刻板印象问题，通过人工评估和自动多维评估验证了LLM在历史类比方面的潜力。

**[Perspective Transition of Large Language Models for Solving Subjective Tasks](llm_nlp/perspective_transition_of_large_language_models_for_solving_subjective_tasks.md)**

:   提出 RPT（Reasoning through Perspective Transition），通过在同一 prompt 中让 LLM 依次探索直接/角色扮演/第三人称三种视角、按置信度排序、选最优视角推理，在 12 个主观任务、4 个模型（GPT-4/GPT-3.5/Llama-3/Qwen-2）上均超越固定视角与集成基线，GPT-3.5 上平均提升 +4.56 点。

**[Pitfalls of Scale: Investigating the Inverse Task of Redefinition in Large Language Models](llm_nlp/pitfalls_of_scale_investigating_the_inverse_task_of_redefinition_in_large_langua.md)**

:   通过重新定义物理/数学常量和计量单位（如"令 π=500"），系统研究 LLM 在逆缩放任务中的表现，发现模型规模越大越倾向于锚定记忆中的原始值而拒绝遵循 prompt 的重新定义，且错误信心（拒绝弃权而给出错误答案）随规模上升。

**[PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](llm_nlp/plangenllms_planning_survey.md)**

:   首篇基于经典规划理论 (Kartam & Wilkins 1990) 提出六维评估框架（完整性、可执行性、最优性、表示、泛化性、效率）的 LLM 规划能力综述，系统梳理了从任务分解到搜索算法的基础范式，并指出多智能体规划、幻觉、人类偏好对齐等关键未解决方向。

**[Planning-Driven Programming: A Large Language Model Programming Workflow](llm_nlp/planning-driven_programming_a_large_language_model_programming_workflow.md)**

:   提出 LPW（LLM Programming Workflow），通过"方案生成→计划验证→代码实现→基于计划验证的精准调试"的两阶段工作流，显著提升 LLM 代码生成准确率，在 GPT-4o 上实现 HumanEval 98.2%、MBPP 84.8%、LiveCode 59.3% 的新 SOTA。

**[PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](llm_nlp/plugin_finetuning_bridge.md)**

:   提出 PiFi 框架，将 LLM 的单层冻结参数插入到 SLM 中并微调，以极低计算开销显著提升 SLM 在 NLU 和 NLG 任务上的性能。

**[KoGEM: Polishing Every Facet of the GEM: Testing Linguistic Competence of LLMs and Humans in Korean](llm_nlp/polishing_every_facet_of_the_gem.md)**

:   提出 KoGEM（韩语语法评估基准），包含 1,524 道基于理论语言学分类的多选题，覆盖音韵/形态/句法/语义/规范 5 大类 16 子类，零样本评估 27 个 LLM 并与人类对比，揭示 LLM 在需要经验知识的语言子类（如发音规则、音韵变化）上远逊人类，而显式补充经验知识（发音文本、语素分解）后可大幅提升。

**[Only a Little to the Left: A Theory-grounded Measure of Political Bias in LLMs](llm_nlp/political_bias_theory_grounded.md)**

:   本文用政治科学中经过验证的 World Values Survey (WVS) 替代缺乏科学基础的 Political Compass Test (PCT)，设计 30 种提示变体在 11 个开源/商业 LLM 上收集 88,110 条开放式回复并训练立场分类器自动标注，发现指令微调模型普遍偏左但偏见度量对提示高度敏感，PCT 会夸大特定模型（如 GPT-3.5）的政治偏见。

**[Pragmatics in the Era of Large Language Models: A Survey on Datasets, Evaluation, Opportunities and Challenges](llm_nlp/pragmatics_survey.md)**

:   系统综述 58 篇文献中评估 NLP 模型语用能力的资源，按语用现象（上下文/指示语、隐含义/预设、言语行为、话语连贯、社会语用）分类，梳理任务设计（MCQ/QA/NLI/参照游戏等）和数据构建方法（自底向上/自顶向下），揭示当前评估的核心差距（英语中心偏置、单模态局限、细粒度评估不足），为 LLM 时代的语用评估提供路线图。

**[PRAISE: Enhancing Product Descriptions with LLM-Driven Structured Insights](llm_nlp/praise_enhancing_product_descriptions_with_llm-driven_structured_insights.md)**

:   提出 PRAISE，一个 4 步 LLM pipeline（属性提取 → 跨产品对比 → 语义分组 → 结构化呈现），使用 Gemini 2.0 Flash 从 Amazon 产品描述中自动生成结构化洞察。在 90 个产品 × 9 个类别上验证，多步 pipeline 显著优于单次生成；效果与产品主观性高度相关（Arts&Crafts F1=0.82 vs Books F1=0.36），每产品仅需 $2R+1$ 次 API 调用。

**[Pre³: Enabling Deterministic Pushdown Automata for Faster Structured LLM Generation](llm_nlp/pre3_deterministic_pda_structured_gen.md)**

:   提出 Pre³，将 LR(1) 文法转化为确定性下推自动机（DPDA），通过预计算前缀条件边消除运行时非确定性探索，实现结构化 LLM 生成的显著加速——每 token 耗时降低最高 40%，吞吐提升最高 36%。

**[Prediction Hubs are Context-Informed Frequent Tokens in LLMs](llm_nlp/prediction_hubs_are_context-informed_frequent_tokens_in_llms.md)**

:   本文首次在自回归LLM中系统分析hubness现象，从理论上证明LLM预测中使用的概率距离不受距离集中效应影响，实证发现预测hub是上下文调制的高频token（属于"良性hub"），但用欧氏距离比较LLM表示时会产生有害的nuisance hub。

**[Probabilistic Aggregation and Targeted Embedding Optimization for Collective Moral Reasoning](llm_nlp/probabilistic_aggregation_and_targeted_embedding_optimization_for_collective_mor.md)**

:   提出一种双阶段框架：先用截断正态分布EM算法将多个LLM的连续道德评分聚合为集体共识概率，再对偏离共识的模型进行道德理论token级嵌入优化，使其与集体意见对齐，实现多LLM间一致的道德推理。

**[Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning](llm_nlp/problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)**

:   提出基于问题求解逻辑（Problem-Solving Logic）的课程式 ICL 策略，通过分析问题的求解步骤结构来选择和排序 demonstration examples，有效提升 LLM 的复杂推理能力。

**[ProgCo: Program Helps Self-Correction of Large Language Models](llm_nlp/progco_program_helps_self-correction_of_large_language_models.md)**

:   ProgCo 提出用 LLM 自动生成并执行验证伪程序（ProgVe）来检查自身回答的正确性，再通过对回答和验证程序的双重反思与修正（ProgRe）来实现可靠的自我纠正，在指令遵循和数学推理任务上显著提升了纠正成功率。

**[Psycholinguistic Word Features: A New Approach for the Evaluation of LLMs Alignment with Humans](llm_nlp/psycholinguistic_word_features_a_new_approach_for_the_evaluation_of_llms_alignme.md)**

:   首次系统提出使用心理语言学词汇规范（Glasgow 5,553词 × 7特征 + Lancaster 39,707词 × 6感知模态，共13种词汇特征）评估LLM与人类对齐，发现GPT-4o在Glasgow情感/概念特征上相关性较高，但所有模型在Lancaster感知觉特征上对齐极差，定量揭示LLM缺乏具身认知的根本局限。

**[Aligning Large Language Models with Implicit Preferences from User-Generated Content](llm_nlp/pugc_align_implicit_pref_ugc.md)**

:   提出 PUGC 框架，利用非标注用户生成内容（UGC）中的隐式人类偏好来生成偏好数据——将 UGC 转化为查询+参考文本，以此评分模型生成的响应，用 DPO 实现可扩展的领域特定对齐，在 Alpaca Eval 2 上基于 Mistral-7B 达到 35.93% 长度控制胜率 SOTA。

**[Quantifying Semantic Emergence in Language Models](llm_nlp/quantifying_semantic_emergence_in_language_models.md)**

:   提出了 Information Emergence (IE) 这一基于信息论的定量指标，通过比较 Transformer 各层中宏观（序列级）与微观（token级）的互信息差异，量化 LLM 从 token 中提取语义的能力。

**[Ranking Unraveled: Recipes for LLM Rankings in Head-to-Head AI Combat](llm_nlp/ranking_unraveled_recipes_for_llm_rankings_in_head-to-head_ai_combat.md)**

:   系统性地评估四种排名算法（Elo、Bradley-Terry、Glicko、Markov Chain）在LLM头对头评估中的表现，定义三条核心排名准则（传递性、预测准确率、超参数敏感性），发现广泛使用的 Elo 排名在稳定性和一致性方面存在严重缺陷，推荐 Glicko 用于大规模不均匀数据集、Bradley-Terry 用于小型可控数据集。

**[Re-TASK: Revisiting LLM Tasks from Capability, Skill, and Knowledge Perspectives](llm_nlp/re-task_revisiting_llm_tasks_from_capability_skill_and_knowledge_perspectives.md)**

:   借鉴 Bloom 分类学和知识空间理论，提出 Re-TASK 框架将 LLM 任务从"能力项-技能-知识"三层视角进行重新审视，并设计 Re-TASK prompting 策略通过针对性的知识注入和技能适配来增强 CoT 在领域任务上的表现，在法律任务上最高提升 45%。

**[Reason from Future: Reverse Thought Chain Enhances LLM Reasoning](llm_nlp/reason_from_future_reverse_thought_chain_enhances_llm_reasoning.md)**

:   提出 Reason from Future（RFF）推理范式，通过交替进行逆向推理（从目标向前分解）和正向推理（从当前状态向目标逼近）实现双向推理，在 Game of 24、GSM8K、MATH-500 等基准上显著超越 CoT、ToT、CR 等方法，同时大幅减少搜索空间。

**[Recent Advances in Speech Language Models: A Survey](llm_nlp/recent_advances_in_speech_language_models_a_survey.md)**

:   首篇 Speech Language Models (SpeechLMs) 综合综述，系统梳理从"ASR+LLM+TTS"级联架构到端到端语音语言模型的演进，提出按三大组件（speech tokenizer / language model / vocoder）和训练方案分类的分类体系，覆盖下游能力、评估指标、挑战与未来方向。

**[Reconsidering LLM Uncertainty Estimation Methods in the Wild](llm_nlp/reconsidering_llm_uncertainty_estimation_methods_in_the_wild.md)**

:   本文系统性地考察了19种LLM不确定性估计方法在实际部署中面临的四大挑战（阈值选择敏感性、查询变换鲁棒性、长文本生成适用性、多分数集成策略），揭示了现有方法在真实场景下的显著局限并提出了集成策略作为实用改进方向。

**[Recurrent Knowledge Identification and Fusion for Language Model Continual Learning](llm_nlp/recurrent_kif_continual_learning.md)**

:   提出Recurrent-KIF持续学习框架，通过内外循环迭代机制动态估计参数重要性分布，利用基于重要性的二值掩码进行知识融合，有效缓解灾难性遗忘并促进知识迁移。

**[Red-Teaming LLM Multi-Agent Systems via Communication Attacks](llm_nlp/red-teaming_llm_multi-agent_systems_via_communication_attacks.md)**

:   提出 Agent-in-the-Middle (AiTM) 攻击，通过拦截和篡改 LLM 多智能体系统中的 agent 间通信消息（而非直接修改 agent 本身），利用一个带反思机制的对抗性 agent 生成上下文感知的恶意指令，在多种框架/通信结构/真实应用上均取得 40%~100% 的攻击成功率。

**[Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models](llm_nlp/refining_salience-aware_sparse_fine-tuning_strategies_for_language_models.md)**

:   首次系统评估了 8 种显著性度量在稀疏参数高效微调（SPEFT）中的效果，发现简单的梯度基方法配合静态掩码即可一致性地超越 LoRA，挑战了"PEFT 需要复杂设计"的常见认知。

**[Representation Bending for Large Language Model Safety](llm_nlp/repbend_representation_bending_safety.md)**

:   提出 RepBend，将 activation steering 的核心思想（安全/不安全表示的向量差异）引入 LoRA 微调的损失函数设计，通过"弯曲"模型的表示空间使安全和不安全状态在潜在空间中远离彼此，在多种越狱攻击基准上实现高达 95% 的攻击成功率降低，且对模型通用能力影响极小。

**[Representations of Fact, Fiction and Forecast in Large Language Models: Epistemics and Attitudes](llm_nlp/representations_of_fact_fiction_and_forecast_in_large_language_models_epistemics.md)**

:   通过受控故事任务评估 8 个开源 LLM 对认识情态（may/must、know/believe/doubt）的语义知识，发现 LLM 在生成恰当认知表达方面表现有限且不鲁棒——必然性（must）优于可能性（may），事实陈述优于信念陈述。

**[RetroLLM: Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation](llm_nlp/retrollm_empowering_large_language_models_to_retrieve_fine-grained_evidence_with.md)**

:   提出RetroLLM统一框架，将检索和生成集成为单一自回归解码过程，通过层级FM-Index约束和前瞻式受限解码，使LLM能直接从语料库中生成细粒度证据，同时显著减少token消耗。

**[Reversal of Thought: Enhancing Large Language Models with Preference-Guided Reverse Reasoning Warm-up](llm_nlp/reversal_of_thought_enhancing_large_language.md)**

:   提出 Reversal of Thought (RoT)，一个即插即用的推理框架，通过偏好引导的逆向推理预热策略，让 LLM 从示例中反向生成"LLM 口味"的最优 prompt，再通过认知偏好管理器自动区分已知/未知任务，在多种推理任务上超越 CoT/ToT/GoT 等基线。

**[Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?](llm_nlp/revisiting_epistemic_markers_in_confidence_estimation_can_markers_accurately_ref.md)**

:   本文定义了"标记置信度"（marker confidence）概念来衡量 LLM 使用认知标记（如"fairly certain"）时的实际准确率，通过 7 个模型和 7 个数据集的系统实验发现：认知标记在分布内场景表现稳定，但在分布外场景下极不可靠。

**[Revisiting Uncertainty Quantification Evaluation in Language Models: Spurious Interactions with Response Length Bias Results](llm_nlp/revisiting_uncertainty_quantification_evaluation_in_language_models_spurious_int.md)**

:   本文发现语言模型不确定性量化（UQ）评估中存在严重的长度偏差问题——UQ 方法和正确性度量函数都受响应长度偏差影响，二者的"互偏差"系统性地扭曲了 AUROC 排名，并在理论和实证上证明了这一点，同时发现 LLM-as-a-Judge 方法是最不受长度偏差影响的评估选择。

**[RiOT: Efficient Prompt Refinement with Residual Optimization Tree](llm_nlp/riot_efficient_prompt_refinement_with_residual_optimization_tree.md)**

:   提出 Residual Optimization Tree（RiOT），一种自动 prompt 优化框架，通过树结构管理优化过程、基于困惑度的节点选择增强多样性、以及文本残差连接缓解语义漂移问题。

**[Robust Utility-Preserving Text Anonymization Based on Large Language Models](llm_nlp/robust_utility-preserving_text_anonymization_based_on_large_language_models.md)**

:   提出RUPTA框架，通过隐私评估器、效用评估器和优化器三个LLM组件协同工作，迭代编辑文本以实现防御LLM重识别攻击的同时保留下游任务效用，并通过DPO蒸馏将匿名化能力迁移到轻量模型。

**[RoCoFT: Efficient Finetuning of Large Language Models with Row-Column Updates](llm_nlp/rocoft_efficient_finetuning_of_large_language_models_with_row-column_updates.md)**

:   提出 RoCoFT，一种极简的参数高效微调方法：仅更新 Transformer 权重矩阵中少量行或列的参数，在 GLUE、QA、摘要生成和常识/数学推理等任务上达到与 LoRA 等 SOTA PEFT 方法相当的精度，同时更省内存和计算，并通过 Neural Tangent Kernel 理论解释了其有效性。

**[Safer or Luckier? LLMs as Safety Evaluators Are Not Robust to Artifacts](llm_nlp/safer_or_luckier_llms_as_safety_evaluators_are_not_robust_to_artifacts.md)**

:   系统评估了11个LLM裁判在安全领域的鲁棒性，发现道歉前缀等表面文本特征（artifact）可将评估偏好扭曲高达98%，提出基于jury的多模型聚合方案但仍未完全解决该问题。

**[Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition](llm_nlp/sample-efficient_human_evaluation_of_large_language_models_via_maximum_discrepan.md)**

:   本文提出基于最大差异 (MAD) 竞争原则的高效人工评测方法，通过自动选择最能区分 LLM 差异的指令子集来大幅减少人工标注量，用仅 280 条对比即可恢复大规模评测的排名结果。

**[Beware of Your Po! Measuring and Mitigating AI Safety Risks in Role-Play Fine-Tuning of LLMs](llm_nlp/sarft_roleplay_safety.md)**

:   首次系统评估了角色扮演微调（role-play fine-tuning）对 LLM 安全性的影响，发现安全退化程度与角色特质（特别是反派角色）正相关，并提出 SaRFT 框架，通过隐式奖励函数自适应识别对不同角色有害的训练数据子集，配合 KL 散度正则化实现角色表现力与安全性的 Pareto 最优平衡。

**[SConU: Selective Conformal Uncertainty in Large Language Models](llm_nlp/sconu_selective_conformal_uncertainty_in_large_language_models.md)**

:   SConU 首次在 LLM 的保形不确定性框架中引入显著性检验，通过构建两种保形 p-value 来识别并过滤违反可交换性假设的不确定性数据异常点，从而在单域和跨域 QA 场景中实现对错误覆盖率（miscoverage rate）的严格管理。

**[SCoP: Evaluating the Comprehension Process of Large Language Models from a Cognitive View](llm_nlp/scop_evaluating_the_comprehension_process_of_large_language_models_from_a_cognit.md)**

:   SCoP 从认知科学视角将 LLM 的文档理解过程分解为五种递进技能（定位、推断、连接、组织、选择），构建了包含 4682 个样本的测试集来评估理解"过程"而非仅仅"答案"，发现 LLM 普遍在局部理解（~94%）上远强于全局理解（~31%），且即使答案正确其理解过程也可能是错误的。

**[SCULPT: Systematic Tuning of Long Prompts](llm_nlp/sculpt_systematic_tuning_of_long_prompts.md)**

:   提出 SCULPT 框架，将长提示词优化建模为层次化树结构的迭代修正问题，通过 Critic-Actor 框架对提示词进行结构化的反思和操作级修改，在保持长提示词信息完整性的同时显著提升 LLM 任务性能，并具备对抗扰动的鲁棒性。

**[SEE: Strategic Exploration and Exploitation for Cohesive In-Context Prompt Optimization](llm_nlp/see_strategic_exploration_exploitation_prompt_optimization.md)**

:   SEE 是首个将指令（instruction）和示例（examples）作为整体进行联合优化的 prompt 优化框架，采用元启发式优化原则设计四阶段探索-利用策略，配合五种 LLM 算子的自适应选择，在 35 个基准任务上大幅超越 9 种 SOTA 方法。

**[Stepwise Reasoning Disruption Attack of LLMs](llm_nlp/seed_stepwise_reasoning_disruption_attack.md)**

:   提出 SEED（Stepwise rEasoning Error Disruption）攻击方法，通过在 LLM 的推理链前几步中巧妙注入细微错误（如微调计算数字），让模型在后续推理中自然传播错误得出错误答案，兼容零样本/少样本设置，GPT-4o 检测率低至 0.8%，揭示了 LLM 逐步推理过程的严重安全漏洞。

**[Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models](llm_nlp/segment_level_diffusion.md)**

:   提出段落级扩散（Segment-Level Diffusion, SLD），将长文本输出切分为多个段落（如句子/对话轮次），对每个段落的潜在表示进行扩散建模，结合对比学习和对抗训练增强表示鲁棒性，在摘要、故事生成、对话生成等任务上实现了比现有扩散模型更好的长文本生成质量。

**[Self-Instructed Derived Prompt Generation Meets In-Context Learning: Unlocking New Potential of Black-Box LLMs](llm_nlp/self-instructed_derived_prompt_generation_meets_in-context_learning_unlocking_ne.md)**

:   提出一种自指导强化学习框架来训练"派生提示生成模型"，并将派生提示-响应对作为上下文学习（ICL）示例来增强原始提示的查询，在不修改黑盒 LLM（如 GPT-4）参数的情况下显著提升响应质量。

**[Self-Training Elicits Concise Reasoning in Large Language Models](llm_nlp/self-training_elicits_concise_reasoning_in_large_language_models.md)**

:   发现 LLM 输出分布中天然包含简洁推理路径，提出 FS-BoN（Few-shot 条件化 + Best-of-N 采样）自训练框架，从模型自身分布中筛选短且正确的推理样本进行微调，在 GSM8K 和 MATH 上跨 5 个模型族实现平均 30% token 缩减且不损准确率，效率为先前方法 Rational Metareasoning 的 2.4 倍。

**[Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching](llm_nlp/self-tuning_instructing_llms_to_effectively_acquire_new_knowledge_through_self-t.md)**

:   受费曼学习法启发，提出 Self-Tuning 框架，通过记忆-理解-自省三层自教学策略，显著提升 LLM 从新文档中有效获取和回忆知识的能力。

**[SelfElicit: Your Language Model Secretly Knows Where is the Relevant Evidence](llm_nlp/selfelicit_evidence_highlighting.md)**

:   SelfElicit 发现 LLM 深层注意力分数天然具有定位上下文中关键证据的能力（即使模型回答错误时也是如此），据此提出一种推理时的上下文增强方法：仅需生成一个额外 token 即可自动识别并高亮关键证据句，引导模型给出更准确的回答。

**[Semantic Exploration with Adaptive Gating for Efficient Problem Solving with Language Models](llm_nlp/semantic_exploration_adaptive_gating.md)**

:   针对 LLM 树搜索推理中"简单题也做复杂搜索"和"语义重复路径反复扩展"两大浪费问题，提出 SEAG 框架：先用 entropy 门控决定是否启动树搜索，再用语义聚类合并等价推理步骤，最终在准确率平均提升 4.3% 的同时仅需 RAP 31% 的推理开销。

**[Embracing Imperfection: Simulating Students with Diverse Cognitive Levels Using LLM-based Agents](llm_nlp/simulating_diverse_students.md)**

:   针对 LLM 难以模拟低水平学生犯错行为的问题，提出基于知识图谱认知原型的 training-free 框架，通过认知状态建模 → 行为预测 → beam search 自精炼三阶段生成逼真的学生解答，在 Student_100 数据集上模拟准确率提升 100%。

**[SkillAggregation: Reference-free LLM-Dependent Aggregation](llm_nlp/skillaggregation_reference-free_llm-dependent_aggregation.md)**

:   本文提出SkillAggregation方法，通过学习上下文相关的LLM评判者技能权重并利用后验估计进行推理，在无需参考标签的情况下有效聚合多个LLM评判者的预测，在多个任务上超越了现有聚合方法。

**[SkillVerse: Assessing and Enhancing LLMs with Tree Evaluation](llm_nlp/skillverse_tree_eval.md)**

:   提出SkillVerse——一种无监督的树结构LLM诊断框架，通过将LLM-as-Judge的评价反馈组织为层次化的技能树（dendrogram），在任意粒度上揭示模型能力的优劣势，并进一步用于选择更优的few-shot示例（ICL提升25%）和预测未知场景下的模型弱点（55%成功率，比无信息基线高22%）。

**[SocialEval: Evaluating Social Intelligence of Large Language Models](llm_nlp/socialeval_evaluating_social_intelligence_of_large_language_models.md)**

:   提出 SocialEval —— 一个基于叙事脚本的双语社会智能基准，通过手工构建 153 个"世界树"将社交互动建模为目标条件 MDP，整合结果导向的目标达成评估（GAE）和过程导向的人际能力评估（IAE），系统评测 LLM 在多回合社交场景中的社会智能及其与人类的差距。

**[SongComposer: A Large Language Model for Lyric and Melody Generation in Song Composition](llm_nlp/songcomposer_llm_lyric_melody_generation.md)**

:   SongComposer是首个能够同时生成歌词和旋律的音乐专用大语言模型，通过词级对齐的元组格式、基于音乐知识的标量音高初始化、以及渐进式结构感知训练（motif→独立全曲→短语级配对），在歌词配旋律、旋律配歌词、歌曲续写和文本生成歌曲等任务上全面超越GPT-4。

**[SQLong: Enhanced NL2SQL for Longer Contexts with LLMs](llm_nlp/sqlong_enhanced_nl2sql_for_longer_contexts_with_llms.md)**

:   提出 SQLong，一种面向长上下文场景的 NL2SQL 数据增强框架，通过向训练数据中注入采样自其他数据库的合成 CREATE TABLE 语句来扩展上下文长度，使微调后的 LLM 在大规模 Schema 场景下显著提升 SQL 生成准确率。

**[SR-LLM: Rethinking the Structured Representation in Large Language Model](llm_nlp/sr-llm_rethinking_the_structured_representation_in_large_language_model.md)**

:   提出 SR-LLM 框架，通过两种设置（training-free 的自然语言描述转换和 training-dependent 的混合数据微调）将结构化表示（AMR、PST、FOL）有效集成到 LLM 中，在 PAWS 等下游任务上分别取得 3.17% 和 12.38% 的提升，首次证明结构化表示能实质性增强 LLM 推理能力。

**[Steering off Course: Reliability Challenges in Steering Language Models](llm_nlp/steering_off_course_reliability_challenges_in_steering_language_models.md)**

:   本文系统性地评估了三种主流的语言模型引导方法（DoLa、功能向量、任务向量）在多达36个模型上的泛化性，发现这些方法存在严重的脆弱性和高方差问题，并揭示了其底层假设的根本缺陷。

**[STEM-PoM: Evaluating Language Models Math-Symbol Reasoning in Document Parsing](llm_nlp/stem-pom_evaluating_language_models_math-symbol_reasoning_in_document_parsing.md)**

:   提出 STEM-PoM 基准数据集（2K+ 数学符号实例），将 Part-of-Math Tagging 与文档解析结合，系统评估 LLM 对数学符号上下文多义性的分类能力，并证明符号分类能力的提升可迁移增强下游数学推理表现。

**[Stress-testing Machine Generated Text Detection: Shifting Language Models Writing Style to Fool Detectors](llm_nlp/stress-testing_machine_generated_text_detection_shifting_language_models_writing.md)**

:   通过 DPO 微调将 LLM 的写作风格对齐到人类文本的语言特征分布，生成更难被检测的机器文本，揭示了现有 MGT 检测器对浅层语言线索的过度依赖。

**[Structural Reasoning Improves Molecular Understanding of LLM](llm_nlp/structural_reasoning_improves_molecular_understanding_of_llm.md)**

:   提出 Molecular Structural Reasoning (MSR) 框架，通过显式融入分子的六种关键结构信息（分子式、最长碳链、芳环、环化合物、官能团、手性中心）作为推理中间步骤，显著提升 LLM 在分子理解任务上的表现。

**[SynapticRAG: Enhancing Temporal Memory Retrieval in Large Language Models through Synaptic Mechanisms](llm_nlp/synapticrag_enhancing_temporal_memory_retrieval_in_large_language_models_through.md)**

:   提出 SynapticRAG，借鉴神经科学中突触传播和漏积分发放（LIF）模型，将时序关联触发与语义相似度融合，在对话记忆检索任务上较 SOTA 提升最高 14.66%。

**[Synergizing Unsupervised Episode Detection with LLMs for Large-Scale News Events](llm_nlp/synergizing_unsupervised_episode_detection_with_llms_for_large-scale_news_events.md)**

:   本文提出 EpiMine，一种无监督的 episode 检测框架，通过判别性词项共现驱动的文章分割与 LLM 协同，从新闻语料中检测关键事件下的 episode（子事件片段），在三个真实数据集上平均提升 59.2%。

**[Systematic Generalization in Language Models Scales with Information Entropy](llm_nlp/systematic_generalization_in_language_models_scales_with_information_entropy.md)**

:   证明语言模型的系统泛化能力与训练数据中成分分布的信息熵正相关——高熵训练分布下即使没有内置组合先验的标准 seq2seq 模型也能实现强系统泛化。

**[T5Score: A Methodology for Automatically Assessing the Quality of LLM Generated Multi-Document Topic Sets](llm_nlp/t5score_a_methodology_for_automatically_assessing_the_quality_of_llm_generated_m.md)**

:   提出 T5Score 方法论，将 LLM 生成的自由文本主题集(FT-topics)的质量分解为五个可量化维度（可解释性、主题覆盖、文档覆盖、非重叠性、内部排序），通过简单标注任务实现高标注者一致性，并验证 LLM 可作为自动评估器替代人工。

**[TableLoRA: Low-rank Adaptation on Table Structure Understanding for Large Language Models](llm_nlp/table_lora_structure_understanding.md)**

:   TableLoRA 提出面向表格任务的专用 LoRA 模块，通过特殊 token 编码器改善表格序列化，并用 2D LoRA 编码单元格的行列位置信息，在参数高效微调设置下相比 vanilla LoRA 在 HiTab 上提升 5.9%，弥合了 LoRA 与全量微调之间 40.56% 的性能差距。

**[TaxoAdapt: Aligning LLM-Based Multidimensional Taxonomy Construction to Evolving Research Corpora](llm_nlp/taxoadapt_aligning_llm-based_multidimensional_taxonomy_construction_to_evolving_.md)**

:   提出 TaxoAdapt 框架，通过层次分类驱动的深度/宽度扩展和分类感知聚类，将 LLM 生成的多维度分类体系动态对齐到特定科学语料库，在粒度保持和兄弟节点一致性上分别超越最优基线 26.51% 和 50.41%。

**[Team Anotheroption at SemEval-2025 Task 8: Bridging the Gap Between Open-Source and Proprietary LLMs in Table QA](llm_nlp/team_anotheroption_at_semeval-2025_task_8_bridging_the_gap_between_open-source_a.md)**

:   提出一种面向表格问答的多模型协同管道系统，整合 Text-to-SQL、Text-to-Code（Pandas）、端到端语义理解三条路径，通过 RAG 检索增强上下文 + Llama 3.3-70B 作为 Orchestrator 仲裁最终答案，在 SemEval-2025 Task 8 的开源赛道中以 80% 准确率排名 13/38，开发集上开源组合（88%）显著超越 GPT-4o 单模型（74%）。

**[TESS 2: A Large-Scale Generalist Diffusion Language Model](llm_nlp/tess_2_a_large-scale_generalist_diffusion_language_model.md)**

:   提出 TESS 2，首个从已有自回归模型适配而来的大规模通用指令遵循扩散语言模型，通过 UL2 masking + label shifting + 双向注意力的适配训练方案 + reward guidance 推理引导，在 QA 和指令遵循任务上匹配甚至超越同等 AR 模型。

**[TestCase-Eval: A Systematic Evaluation of Fault Coverage and Exposure](llm_nlp/testcase_eval_llm_test_gen.md)**

:   提出TestCase-Eval基准，包含500道Codeforces竞赛题和10万份人类提交代码，通过Fault Coverage和Fault Exposure两个任务系统评估19个LLM在算法问题测试用例生成方面的能力，发现最强模型Qwen3-32B仅达43.8%暴露率，远低于人类专家的93.3%。

**[The Nature of NLP: Analyzing Contributions in NLP Papers](llm_nlp/the_nature_of_nlp_analyzing_contributions_in_nlp_papers.md)**

:   提出 NLP 论文贡献的分类体系（知识/工件 × 8 子类），构建 ~2k 人工标注数据集 NLPContributions，训练 SciBERT 自动识别贡献声明，并对 ~29k 篇 ACL Anthology 论文做 50 年纵向趋势分析，揭示 NLP 研究从语言学导向转向方法/模型主导、近年又重拾人文与语言关注的演化轨迹。

**[The Role of Deductive and Inductive Reasoning in Large Language Models](llm_nlp/the_role_of_deductive_and_inductive_reasoning_in_large_language_models.md)**

:   本文提出 DID（De-In-Ductive）框架，通过动态结合演绎推理和归纳推理来增强 LLM 的推理能力，利用 Littlestone 维度和信息熵组成的双指标复杂度评估系统来指导问题分解策略，在 AIW 基准上达到 70.3% 准确率（超过 ToT 的 62.2%），同时保持更低的计算成本。

**[A Theory of Response Sampling in LLMs: Part Descriptive and Part Prescriptive](llm_nlp/theory_of_llm_sampling.md)**

:   提出并验证了LLM的响应采样理论——采样过程同时受描述性成分(统计规范)和规范性成分(隐式理想值)双重驱动，导致样本系统性地偏离统计平均值向理想值方向偏移，这种偏差在15个模型、500个概念上具有统计显著性，且模型越大偏差越强。

**[Theory of Mind in Large Language Models: Assessment and Enhancement](llm_nlp/theory_of_mind_llm.md)**

:   系统综述了 LLM 的心智理论（ToM）能力的评估基准（10+ story-based benchmarks）和增强策略（prompt-only 和 fine-tuning 两类方法），指出当前 LLM 在 ToM 推理上仍有显著不足，并提出未来方向。

**[TigerLLM - A Family of Bangla Large Language Models](llm_nlp/tigerllm_-_a_family_of_bangla_large_language_models.md)**

:   针对孟加拉语（全球第5大语言）的 LLM 严重不足问题，构建高质量教科书语料 Bangla-TextBook（10M token）和原生指令数据 Bangla-Instruct（100K），训练的 TigerLLM 家族在六项基准上超越所有开源替代方案并胜过 GPT-3.5。

**[To Code or not to Code? Adaptive Tool Integration for Math Language Models via Expectation-Maximization](llm_nlp/to_code_or_not_to_code_adaptive_tool_integration_for_math_language_models_via_ex.md)**

:   提出基于EM框架的AutoCode方法，让数学LLM自主决定何时使用代码工具辅助推理，通过E-step引导探索高潜力代码触发决策+M-step离线RL优化，7B模型在MATH500上提升11%+。

**[The Impact of Token Granularity on the Predictive Power of Language Model Surprisal](llm_nlp/token_granularity_impact.md)**

:   系统研究子词 token 粒度（词表大小 256~128K）对 LM surprisal 预测人类阅读时间能力的影响，发现 ~8K 词表的中等粒度在自然阅读时间预测上最优（甚至优于 GPT-2），而粗粒度 token 在花园路径句法效应上更敏感，揭示认知建模的最优分词粒度并非 NLP 通用标准。

**[Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs](llm_nlp/token_prepending_training_free.md)**

:   提出 Token Prepending (TP) 技术，通过在每层将解码得到的句子嵌入前置到句子开头，使因果注意力机制下的早期 token 也能感知完整句子信息，无需训练即可显著提升 LLM 的句子嵌入质量。

**[Turning Trash into Treasure: Accelerating Inference of Large Language Models with Token Recycling](llm_nlp/token_recycling.md)**

:   提出 Token Recycling——一种无需额外训练的投机解码方法，将解码过程中被拒绝的候选 token 存入轻量邻接矩阵，通过 BFS 算法构建 draft tree 并用 tree attention 验证，仅需 <2MB 存储即在所有规模 LLM 上实现约 2 倍加速。

**[ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models](llm_nlp/toolcoder_code_empowered_tool_learning.md)**

:   提出 ToolCoder 框架，将工具学习重新定义为代码生成任务，借鉴软件工程原则（需求分析→模块化设计→实现执行→错误调试→代码复用）让 LLM 通过生成和执行 Python 代码来完成多步工具调用，在 RestBench 和 API-Bank 上全面超越 ReAct、CodeAct 等基线方法。

**[ToolSpectrum: Towards Personalized Tool Utilization for Large Language Models](llm_nlp/toolspectrum_towards_personalized_tool_utilization_for_large_language_models.md)**

:   提出ToolSpectrum基准，首次定义并评估LLM的个性化工具使用能力——根据用户画像和环境因素选择最合适的工具，实验表明个性化显著提升用户体验，但现有LLM在联合推理用户和环境因素时能力有限。

**[Towards Enhanced Immersion and Agency for LLM-based Interactive Drama](llm_nlp/towards_enhanced_immersion_and_agency_for_llm-based_interactive_drama.md)**

:   提出沉浸感-代理感（Immersion-Agency）范式来理解LLM交互式戏剧，并设计Playwriting-guided Generation和Plot-based Reflection两种方法分别增强故事生成质量和玩家对故事的影响力。

**[Towards Harmonized Uncertainty Estimation for Large Language Models](llm_nlp/towards_harmonized_uncertainty_estimation_for_large_language_models.md)**

:   提出 CUE 框架，通过训练一个与目标 LLM 性能对齐的轻量级分类器（Corrector）来校正现有不确定性估计方法的分数，在指示性、精确-召回平衡和校准三个维度上实现协调一致的改进，最高提升达 60%。

**[Training-free LLM Merging for Multi-task Learning](llm_nlp/training-free_llm_merging_for_multi-task_learning.md)**

:   提出Hi-Merging，一种**层级迭代式无训练模型合并**方法，通过模型级和层级的剪枝与缩放操作，结合贡献度分析来识别和解决参数冲突，将不同任务/语言的专用LLM合并为统一的多任务模型，在大多数场景下超越混合数据微调的基线。

**[Training Language Model to Critique for Better Refinement](llm_nlp/training_language_model_to_critique_for_better_refinement.md)**

:   提出 Refinement-oriented Critique Optimization（RCO），以"批判效用"（Critique Utility, CU）——即批判导致的精炼改善比例——作为奖励信号训练 critic 模型，通过 DPO 变体的 MSE 目标函数优化，无需直接评估批判质量；在对话生成、摘要、问答、数学推理、代码生成 5 个任务上，RCO 训练的 7B/13B critic 模型在 CU 和 RQS 指标上显著超过 70B 基线模型和 DPCO 方法。

**[Transforming Podcast Preview Generation: From Expert Models to LLM-Based Systems](llm_nlp/transforming_podcast_preview_generation_from_expert_models_to_llm-based_systems.md)**

:   Spotify 提出用 LLM（Gemini 1.5 Pro）替代传统多模型特征工程流水线来生成播客预览片段，在离线人工评估和线上 A/B 测试中均显著优于传统系统，用户互动时长提升 4.6%，处理效率提升 5 倍。

**[TRATES: Trait-Specific Rubric-Assisted Cross-Prompt Essay Scoring](llm_nlp/trates_trait-specific_rubric-assisted_cross-prompt_essay_scoring.md)**

:   提出 TRATES 框架，重新定义 LLM 在自动作文评分中的角色——从直接评分者转变为**特质特征生成器与提取器**，通过 LLM 将评分标准(rubric)自动转化为评估问题(子特质)，结合通用写作质量特征和提示特定特征训练回归模型，在 ASAP 数据集 8 个特质上全部达到 SOTA，且首次在 ELLIPSE 数据集上建立跨提示特质评分基线。

**[TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues](llm_nlp/tremu_towards_neuro-symbolic_temporal_reasoning_for_llm-agents_with_memory_in_mu.md)**

:   提出TReMu框架，通过时间感知记忆化（时间线摘要）和神经符号时间推理（LLM生成Python代码执行时间计算），将GPT-4o在多会话对话时间推理基准上的准确率从29.83%提升到77.67%。

**[TrimLLM: Progressive Layer Dropping for Domain-Specific LLMs](llm_nlp/trimllm_layer_dropping.md)**

:   提出TrimLLM，基于层级专业化（layer-wise specialization）现象，在领域微调过程中渐进式丢弃对目标领域不重要的层，在50-60%压缩率下无精度损失且获得2.1-5.7倍推理加速，且不依赖专用硬件。

**[UAQFact: Evaluating Factual Knowledge Utilization of LLMs on Unanswerable Questions](llm_nlp/uaqfact_evaluating_factual_knowledge_utilization_of_llms_on_unanswerable_questio.md)**

:   提出双语不可回答问题数据集UAQFact（13,970题），每个问题附带知识图谱事实知识，定义三个评估任务分别衡量LLM区分不可回答问题（UAQ）与可回答问题（ABQ）、利用内部/外部事实知识处理UAQ的能力，实验揭示即使LLM已存储相关知识也难以有效利用。

**[Un-considering Contextual Information: Assessing LLMs' Understanding of Indexical Elements](llm_nlp/un-considering_contextual_information_assessing_llms_understanding_of_indexical_.md)**

:   首次系统评估 LLM 对英语指示词（I/you/here/tomorrow）的理解能力，构建 1600 条 2×2 因素设计的评测集，揭示 LLM 在 you/here/tomorrow 上严重依赖无关上下文信息而非语法规则，且引号对不同指示词的影响方向截然相反。

**[Uncertainty Unveiled: Can Exposure to More In-context Examples Mitigate Uncertainty for Large Language Models?](llm_nlp/uncertainty_unveiled_can_exposure_to_more_in-context_examples_mitigate_uncertain.md)**

:   本文系统研究了长上下文 ICL 中增加示例数量对 LLM 预测不确定性的影响，通过不确定性分解揭示性能提升主要源于认知不确定性（EU）的降低，并从残差流投影角度解释了不确定性减少的内部机制。

**[Understanding and Meeting Practitioner Needs When Measuring Representational Harms Caused by LLM-Based Systems](llm_nlp/understanding_and_meeting_practitioner_needs_when_measuring_representational_har.md)**

:   通过对 12 位负责评估 LLM 系统表征性伤害（representational harms）的从业者进行半结构化访谈，发现公开可用的测量工具普遍无法满足实践者需求——要么因效度/特异性不足而"不好用"（not useful），要么因组织/制度壁垒而"用不了"（not used），并基于测量理论和实用测量框架提出系统性改进建议。

**[Understanding Silent Data Corruption in LLM Training](llm_nlp/understanding_silent_data_corruption_in_llm_training.md)**

:   本文首次系统研究了真实世界静默数据损坏（SDC）对LLM训练的影响，通过将不健康节点与健康节点配对并引入同步机制，在子模块计算、单步梯度、累积训练三个层面揭示了SDC的特征和影响模式。

**[Understanding the Dark Side of LLMs' Intrinsic Self-Correction](llm_nlp/understanding_the_dark_side_of_llms_intrinsic_self-correction.md)**

:   本文系统研究了 LLM 内在自我纠错（intrinsic self-correction）的失败现象，提出三种可解释性方法揭示失败原因——简单任务中的答案动摇和提示偏差、复杂任务中的类人认知偏差，并提出问题重复和少样本 SFT 两种简单有效的缓解策略。

**[Understanding the Repeat Curse in Large Language Models from a Feature Perspective](llm_nlp/understanding_the_repeat_curse_in_large_language_models_from_a_feature_perspecti.md)**

:   从机制可解释性角度研究 LLM 重复生成问题（Repeat Curse），用 Sparse Autoencoder 提取单语义特征，定位中间层和最终层的"重复特征"，激活它们可诱导重复、关闭它们可缓解重复且不损害模型性能。

**[Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](llm_nlp/unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)**

:   本文首次系统揭示了基于Schwartz价值观对齐的LLM存在非预期安全风险——特定价值维度与特定安全风险类别显著相关，并从心理学视角解释了这些关联的根源，进而提出通过提示抑制相关价值来降低有害行为的缓解策略。

**[ScaleQuest: Unleashing LLM Reasoning Capability via Scalable Question Synthesis from Scratch](llm_nlp/unleashing_llm_reasoning_capability_via_scalable.md)**

:   提出 ScaleQuest，通过 Question Fine-Tuning (QFT) + Question Preference Optimization (QPO) 两阶段训练将 7B 解题模型变为出题模型，从零合成 100 万高质量数学问题-解答对，在四个基准上全面超越所有开源数据集，且数据量扩展至 1M 时性能持续提升未见饱和。

**[Unlocking Recursive Thinking of LLMs: Alignment via Refinement](llm_nlp/unlocking_recursive_thinking_of_llms_alignment_via_refinement.md)**

:   提出 AvR（Alignment via Refinement）两阶段框架，通过细化感知奖励（refinement-aware reward）和差分学习，让 LLM 学会"批评→改进"的递归思维能力，仅用 10k 数据即在 AlpacaEval 2 上将 LLaMA-3-8B-Instruct 的胜率提升超 26 个百分点。

**[UnSeenTimeQA: Time-Sensitive Question-Answering Beyond LLMs' Memorization](llm_nlp/unseentimeqa_time-sensitive_question-answering_beyond_llms_memorization.md)**

:   提出UnSeenTimeQA——一个基于合成事实（而非真实世界事件）的时间敏感问答基准，通过避免网络可搜索的查询来消除数据污染风险，设计了三类时间推理问题来评估LLM的真实时序推理能力，发现LLM在长程事件依赖和并行事件推理上表现较差。

**[Unveiling Dual Quality in Product Reviews: An NLP-Based Approach](llm_nlp/unveiling_dual_quality_in_product_reviews_an_nlp-based_approach.md)**

:   提出面向产品评论的"双重质量"自动检测任务，通过迭代式主动学习构建首个波兰语DQ数据集（1,957条评论），系统对比SetFit、Transformer编码器和LLM三类方法，发现语言专用编码器与带指令的LLM性能相当（DQ F1 ≈ 80-83%），并验证了跨语言迁移能力。

**[Value Portrait: Assessing Language Models' Values through Psychometrically and Ecologically Valid Items](llm_nlp/value_portrait_assessing_language_models_values_through_psychometrically_and_eco.md)**

:   提出Value Portrait基准，通过心理测量学验证（每个测试项与真实人类价值观得分的相关性）和生态效度设计（基于真实用户-LLM交互场景），系统评估44个LLM的价值取向，发现模型普遍优先仁慈、安全和自我导向价值，同时揭露了模型对不同人口群体的价值认知偏见。

**[Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning](llm_nlp/veracity_bias_llm_hidden_beliefs.md)**

:   揭示了 LLM 在推理任务中存在"真实性偏见"（Veracity Bias）——尽管显式对齐反对刻板印象，LLM 仍系统性地将正确答案归因于特定种族群体（归因偏差），并对相同解答因"作者"种族不同给出不同评价（评估偏差），在数学、编程、常识推理和写作任务中普遍存在。

**[Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System](llm_nlp/virsci_multi_agent_idea_gen.md)**

:   提出 VirSci 多 agent 系统，用真实科学家数据构建虚拟科研生态，通过 5 步协作流程和创新的组间+组内讨论机制生成科学 idea，在新颖性和潜在影响力上显著超越单 agent 系统。

**[WarriorCoder: Learning from Expert Battles to Augment Code Large Language Models](llm_nlp/warriorcoder_learning_from_expert_battles_to_augment_code_large_language_models.md)**

:   提出 WarriorCoder，通过构建多个专家代码 LLM 之间的竞技场（arena），让攻击者用自身擅长的领域挑战防御者，由裁判评估后用胜者回答训练目标模型，从而无需依赖专有模型或预存数据集即可从零生成高质量、高多样性的代码训练数据，实现 SOTA 性能。

**[What Happened in LLM Layers when Trained for Fast vs. Slow Thinking: A Gradient Perspective](llm_nlp/what_happened_in_llms_layers_when_trained_for_fast_vs_slow_thinking_a_gradient_p.md)**

:   本文从梯度的视角系统地研究了 LLM 各层在训练快思考（无/简短 CoT）vs 慢思考（详细 CoT）数据时的行为差异，发现慢思考训练导致梯度在各层间更均匀稳定，而快思考导致更大的梯度和更剧烈的层间波动，且慢思考的梯度模式能区分正确与不相关的推理路径。

**[When Large Language Models Meet Speech: A Survey on Integration Approaches](llm_nlp/when_large_language_models_meet_speech_a_survey_on_integration_approaches.md)**

:   系统综述语音与大语言模型的集成方法，将现有工作分为文本级、隐表示级、音频token级三大类，覆盖 ASR/S2TT/S2ST/TTS 等应用场景，并给出各方法的优劣对比与未来挑战。

**[Which Demographics Do LLMs Default to During Annotation?](llm_nlp/which_demographics_do_llms_default_to_during_annotation.md)**

:   通过对比 LLM 在无人口统计信息(N)、有人口统计(SD)、安慰剂信息(P)三种 prompt 条件下的标注行为，揭示 LLM 在主观标注任务(冒犯性/礼貌性)中默认更接近白人、年轻、高学历群体的标注模式，且人口统计 prompting 确实产生了比安慰剂信息更系统性的影响。

**[Why Not Act on What You Know? Unleashing Safety Potential of LLMs via Self-Aware Guard Enhancement](llm_nlp/why_not_act_on_what_you_know_unleashing_safety_potential_of_llms_via_self-aware_.md)**

:   发现 LLM 作为判别器时能准确识别越狱请求、但作为生成器时却仍产出有害内容的"判别-生成安全差距"，提出免训练的 SAGE（Self-Aware Guard Enhancement）策略，通过判别分析模块和判别响应模块将模型自身的安全鉴别能力桥接到生成行为，在 6 个模型上达到平均 99% 的防御成功率。

**[Why Prompt Design Matters and Works: A Complexity Analysis of Prompt Search Space in LLMs](llm_nlp/why_prompt_design_matters_and_works_a_complexity_analysis_of_prompt_search_space.md)**

:   从理论角度分析 prompt 在 LLM 推理中的作用机制——证明 prompt 充当"选择器"从隐藏状态中提取任务相关信息并定义答案空间中的轨迹，分析了最优 prompt 搜索空间的复杂度，并通过实验验证了最优 prompt 搜索可带来 50%+ 的推理性能提升。

**[Why Safeguarded Ships Run Aground? Aligned Large Language Models' Safety Mechanisms Tend to Be Anchored in The Template Region](llm_nlp/why_safeguarded_ships_run_aground_aligned_large_language_models_safety_mechanism.md)**

:   揭示了安全对齐LLM的一个普遍现象：安全机制过度锚定在chat template区域（TASA），导致越狱攻击可通过干扰template区域的信息处理来绕过安全防线，并提出通过将安全探针从template区域迁移到生成阶段来缓解该漏洞。

**[Zero-Shot Belief: A Hard Problem for LLMs](llm_nlp/zero-shot_belief_a_hard_problem_for_llms.md)**

:   本文提出统一式（Unified）和混合式（Hybrid）两种零样本框架用于源-目标信念预测（source-and-target belief prediction），混合方法使用微调 DeBERTa 做事件检测 + LLM 做信念标注，在 FactBank 上以 72.0% Full F1 刷新 SOTA，同时首次报告嵌套信念指标（Nested F1 仅 25.3%），揭示该子任务对当前所有 LLM 仍是极大挑战。

---

## 🧩 多模态 VLM { #multimodal_vlm }

**[A Parameter-Efficient and Fine-Grained Prompt Learning for Vision-Language Models](multimodal_vlm/a_parameter-efficient_and_fine-grained_prompt_learning_for_vision-language_model.md)**

:   本文提出 DoPL（Detail-oriented Prompt Learning）方法，通过低熵信息集中理论发现文本-视觉兴趣共享 token，并以此构建对齐权重增强文本和视觉提示，仅用 0.25M（0.12%）可训练参数即实现细粒度多模态语义对齐，在六个基准上超越全参数微调方法。

**[Activating Distributed Visual Region within LLMs for Efficient and Effective Vision-Language Training and Inference](multimodal_vlm/activating_distributed_visual_region_within_llms_for_efficient_and_effective_vis.md)**

:   本文发现 LLM 中存在类似人脑视觉皮层的"视觉区域"——稀疏且均匀分布的层集合，仅更新 25% 的层即可保留 99% 的视觉性能，同时保持甚至提升语言能力，并据此提出了高效的视觉区域定向训练和剪枝范式。

**[Activation Steering Decoding: Mitigating Hallucination in Large Vision-Language Models through Bidirectional Hidden State Intervention](multimodal_vlm/activation_steering_decoding_mitigating_hallucination_in_large_vision-language_m.md)**

:   本文提出 ASD（Activation Steering Decoding），一种无需训练的推理时幻觉缓解方法，通过在 LVLM 的中间隐藏状态中识别幻觉方向模式，利用双向引导和对比解码来抑制幻觉输出，同时保持模型在通用视觉理解任务上的性能。

**[AdamMeme: Adaptively Probe the Reasoning Capacity of Multimodal Large Language Models on Harmfulness](multimodal_vlm/adammeme_adaptively_probe_the_reasoning_capacity_of_multimodal_large_language_mo.md)**

:   提出AdamMeme——一个基于多智能体协作的自适应评估框架，通过迭代生成更具挑战性的meme样本来探测多模态大语言模型(mLLM)在有害内容理解上的推理能力和特定弱点。

**[Adaptive Linguistic Prompting (ALP) Enhances Phishing Webpage Detection in Multimodal Large Language Models](multimodal_vlm/adaptive_linguistic_prompting_alp_enhances_phishing_webpage_detection_in_multimo.md)**

:   提出 Adaptive Linguistic Prompting (ALP)，一种 8-shot 结构化提示方法，引导多模态 LLM 从 HTML 文本、截图和 URL 三个维度联合推理，检测钓鱼网页，在 GPT-4o 上组合分析达到 F1=0.93，超过传统零样本基线。

**[Can LLMs Deceive CLIP? Benchmarking Adversarial Compositionality of Pre-trained Multimodal Representation via Text Updates](multimodal_vlm/adversarial_compositionality_clip.md)**

:   提出MAC基准和diversity-promoting自训练方法，通过让LLM生成欺骗性文本来系统暴露CLIP等预训练多模态表征的组合性漏洞，在图像/视频/音频三个模态上均显著超越已有方法。

**[Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents](multimodal_vlm/agent_rewardbench.md)**

:   本文提出Agent-RewardBench，首个评估多模态LLM作为agent奖励模型能力的基准，覆盖感知/规划/安全三个维度和7个真实场景，包含1,136条高质量step-level样本，实验揭示即使最强模型GPT-4o也仅达61.4%准确率，且强模型在安全维度反而表现更差。

**[AGRI-CM3: A Chinese Massive Multi-Modal Multi-Level Benchmark for Agricultural Understanding](multimodal_vlm/agri-cm3_a_chinese_massive_multi-modal_multi-level_benchmark_for_agricultural_un.md)**

:   本文提出AGRI-CM3，一个面向农业领域的中文大规模多模态多层次评测基准，涵盖作物识别、病虫害诊断、农事操作理解等多个农业子任务，用于系统评估VLM在农业垂直领域的能力。

**[AkaCE: A Multimodal Multi-party Dataset for Emotion Recognition in Movie Dialogues](multimodal_vlm/akan_cinematic_emotions_ace_a_multimodal_multi-party_dataset_for_emotion_recogni.md)**

:   构建 AkaCE——首个非洲语言多模态对话情感识别数据集，覆盖阿坎语（加纳主要语言，约 2000 万使用者），含 385 段对话 6162 条发言（音频+视觉+文本三模态）、308 名说话人（性别平衡 155男/153女），并提供首个非洲语言词级韵律突出标注。

**[Aligning VLM Assistants with Personalized Situated Cognition](multimodal_vlm/aligning_vlm_assistants_with_personalized_situated.md)**

:   基于社会学"角色集合"(Role-Set) 概念刻画用户多样性，提出 PCogAlign 框架，通过认知感知的动作导向奖励模型来为 VLM 助手生成个性化回复，使不同角色的用户在相同视觉场景下获得最适合自身需求的建议。

**[AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](multimodal_vlm/alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)**

:   提出 AlignMMBench，首个面向中文视觉上下文的多模态对齐评测基准，涵盖 3 大类 13 项任务、1054 张图像和 4978 个 QA 对（含单轮/多轮对话），并训练了基于 ChatGLM3-6B 的评估器 CritiqueVLM，其评估一致性超过 GPT-4。

**[Answering Complex Geographic Questions by Adaptive Reasoning with Visual Context and External Commonsense Knowledge](multimodal_vlm/answering_complex_geographic_questions_by_adaptive_reasoning_with_visual_context.md)**

:   本文提出一种面向复杂地理问题的自适应推理框架，结合视觉上下文（如地图、卫星图像）和外部常识知识库进行多步推理，根据问题复杂度动态选择推理路径，在地理VQA任务上显著超越直接端到端回答的方法。

**[Aria-UI: Visual Grounding for GUI Instructions](multimodal_vlm/aria-ui_visual_grounding_for_gui_instructions.md)**

:   提出 Aria-UI，一个专为 GUI 视觉定位设计的纯视觉多模态模型，通过可扩展的指令合成数据管线和文本-图像交错的动作历史机制，在离线和在线 Agent 基准上均达到 SOTA，包括 AndroidWorld 第1名（44.8%）和 OSWorld 第3名（15.2%）。

**[Attacking Vision-Language Computer Agents via Pop-ups](multimodal_vlm/attacking_vl_agents_popups.md)**

:   系统性设计了一套对抗性弹窗攻击方法来攻击基于视觉语言模型的计算机操控 agent，在 OSWorld 和 VisualWebArena 上平均攻击成功率达 86%，任务成功率下降 47%，基础防御手段几乎无效。

**[AVG-LLaVA: An Efficient Large Multimodal Model with Adaptive Visual Granularity](multimodal_vlm/avg-llava_an_efficient_large_multimodal_model_with_adaptive_visual_granularity.md)**

:   在 LLaVA-NeXT 上增加视觉粒度缩放器（空间金字塔池化获取多级粒度 token）和视觉粒度路由器（基于图像+指令自适应选粒度），并提出 RGLF 训练范式用 LMM 自身的生成概率作为反馈来训练路由器，在 11 个基准上实现"减少 token 反而提升性能"的效果。

**[Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning](multimodal_vlm/benchmarking_and_improving_large_vision-language_models_for_fundamental_visual_g.md)**

:   本文构建了一个系统性评测基准来评估大型视觉语言模型（LVLM）在基础视觉图结构理解与推理上的能力，发现现有模型在此类任务上表现欠佳，并提出了针对性的改进方法。

**[Enhancing Multimodal Continual Instruction Tuning with BranchLoRA](multimodal_vlm/branchlora_continual_instruction.md)**

:   针对多模态持续指令微调(MCIT)中MoELoRA的参数低效和灾难性遗忘问题，提出BranchLoRA——一种非对称架构，共享矩阵A捕获跨任务通用模式、多路矩阵B编码任务特有知识，配合灵活调参-冻结机制和任务特定路由器，在CoIN benchmark上以更少参数大幅超越前SOTA MoELoRA（ACC: 44.20 vs 37.13, BWT: -20.98 vs -25.91）。

**[Burn After Reading: Do Multimodal Large Language Models Truly Capture Order of Events in Image Sequences?](multimodal_vlm/burn_after_reading_do_multimodal_large_language_models_truly_capture_order_of_ev.md)**

:   提出 TempVS 基准测试，系统评估 38 个 MLLM 在图像序列中对多事件时序关系的 grounding 和推理能力，揭示 SOTA 模型与人类之间存在巨大性能差距。

**[Can MLLMs Understand the Deep Implication Behind Chinese Images?](multimodal_vlm/can_mllms_understand_the_deep_implication_behind_chinese_images.md)**

:   提出 CII-Bench（Chinese Image Implication Understanding Benchmark），包含698张中国互联网/传统文化图像及800道选择题，系统评测MLLM对中文图像深层含义的高阶理解能力，发现最佳模型准确率仅64.4%，远低于人类平均78.2%，且模型在中国传统文化领域表现最差。

**[Can Multimodal Foundation Models Understand Schematic Diagrams? An Empirical Study on Information-Seeking QA over Scientific Papers](multimodal_vlm/can_multimodal_foundation_models_understand_schematic_diagrams_an_empirical_stud.md)**

:   本文提出 MISS-QA，首个专门评估多模态基础模型理解科学论文中示意图能力的基准，包含 1,500 个专家标注样本，揭示了当前最强模型与人类专家之间存在显著性能差距。

**[Can Vision-Language Models Evaluate Handwritten Math?](multimodal_vlm/can_vision-language_models_evaluate_handwritten_math.md)**

:   本文提出FERMAT基准，通过609道人工策划的7-12年级数学题及其2200+份手写错误解答（覆盖计算、概念、符号、格式四类错误），系统评估9个VLM在手写数学内容的错误检测、定位和纠正能力，发现Gemini-1.5-Pro达到最高纠错率77%，但所有模型在处理手写内容时仍面临显著挑战。

**[Can Vision Language Models Understand Mimed Actions?](multimodal_vlm/can_vision_language_models_understand_mimed_actions.md)**

:   提出 Mime 基准（86 个哑剧动作 × 10 种变体 = 860 个样本），通过动作捕捉 + 3D 渲染构建可控评测，发现人类在各种扰动下保持近 100% 准确率而最强 VLM 仅 52.3%（多选）/ 19.8%（自由回答），揭示 VLM 严重依赖场景上下文线索而非动作本身。

**[MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs](multimodal_vlm/cant_see_the_forest_for_the.md)**

:   提出 MMSafeAware，首个同时评估"不安全内容识别"和"过度敏感"的多模态安全意识基准，包含 1,500 个跨 29 种安全场景的图文对，评估 9 个 MLLM 发现所有模型都存在安全与有用性的严重权衡——GPT-4V 将 36.1% 的不安全输入误判为安全，同时将 59.9% 的安全输入误判为不安全；三种改进方法均无法根本解决问题。

**[Centurio: On Drivers of Multilingual Ability of Large Vision-Language Model](multimodal_vlm/centurio_multilingual_vlm.md)**

:   系统研究多语言LVLM训练策略中训练语言数量、语言数据分布和多语言OCR三个维度，发现可同时训练100种语言且仅需25-50%非英语数据，据此训练出覆盖100语言的Centurio模型达到SOTA。

**[Chart-based Reasoning: Transferring Capabilities from LLMs to VLMs](multimodal_vlm/chart-based_reasoning_transferring_capabilities_from_llms_to_vlms.md)**

:   本文提出一种将LLM的推理能力迁移到VLM的方法，通过改进图表表示预训练、构造大规模合成推理数据集和多任务微调，使5B参数的PaLI-3在ChartQA上超越10倍大的模型。

**[ChartCoder: Advancing Multimodal Large Language Model for Chart-to-Code Generation](multimodal_vlm/chartcoder_chart_to_code.md)**

:   提出首个专用chart-to-code MLLM（ChartCoder），以Code LLM为语言骨干+160K大规模图表-代码数据集+Snippet-of-Thought逐步推理方法，7B模型在三个基准上超越所有开源MLLM，接近GPT-4o水平。

**[Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation](multimodal_vlm/code_guided_text_rich_image.md)**

:   提出CoSyn框架，利用纯文本LLM的代码生成能力自动创建40万张文本丰富图像（图表、文档、图表等）+270万条指令微调数据，训练的7B VLM在7个基准上达到SOTA，超越GPT-4V和Gemini 1.5 Flash。

**[COLING-UniA at SciVQA 2025: Few-Shot Example Retrieval and Confidence-Informed Ensembling for Multimodal Large Language Models](multimodal_vlm/coling-unia_at_scivqa_2025_few-shot_example_retrieval_and_confidence-informed_en.md)**

:   本文提出了一种基于多模态大模型（MLLM）集成的科学图表视觉问答系统，通过 few-shot 示例检索策略和置信度感知的模型选择机制，在 SciVQA 2025 共享任务中获得第三名（平均 F1 = 85.12）。

**[Con Instruction: Universal Jailbreaking of Multimodal Large Language Models via Non-Textual Modalities](multimodal_vlm/con_instruction_universal_jailbreaking_of_multimodal_large_language_models_via_n.md)**

:   本文提出 Con Instruction 方法，通过优化对抗性图像或音频使其在嵌入空间中与目标恶意指令对齐，实现无需文本输入即可越狱多模态大语言模型（MLLM），在 LLaVA-v1.5 上达到 86.6% 的攻击成功率，并提出了 ARC 评估框架来同时衡量攻击响应的质量和相关性。

**[Insight Over Sight: Exploring the Vision-Knowledge Conflicts in Multimodal LLMs](multimodal_vlm/conflictvis_vision_knowledge_conflict.md)**

:   首次系统探索 MLLM 中常识级别的视觉-知识冲突问题，提出自动化框架构建 ConflictVis 基准（374 图 + 1122 QA），发现 MLLM 在约 20% 的冲突场景中过度依赖参数化知识（尤其是 Yes-No 和动作类问题），并提出 Focus-on-Vision 提示策略进行缓解。

**[CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relations?](multimodal_vlm/cordial_can_multimodal_large_language_models_effectively_understand_coherence_re.md)**

:   本文提出CORDIAL，首个用连贯关系（Coherence Relations）评估MLLM多模态话语分析能力的基准，涵盖灾难管理、社交媒体和在线文章3个话语领域的不同粒度连贯关系，实验发现即使Gemini 1.5 Pro和GPT-4o也无法匹配简单的CLIP分类器基线，揭示了MLLM在语用理解方面的根本不足。

**[Coreference as an Indicator of Context Scope in Multimodal Narrative](multimodal_vlm/coreference_as_an_indicator_of_context_scope_in_multimodal_narrative.md)**

:   本文揭示了大型多模态语言模型在视觉叙事（visual storytelling）任务中的共指表达（coreferential expressions）分布与人类存在显著差异——人类能变化丰富地交错引用不同实体并保持跨文本-图像的一致性，而机器在追踪混合引用方面能力不足，并提出了一系列量化共指模式特征的指标。

**[CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World](multimodal_vlm/craftext_benchmark_advancing_instruction_following_in_complex_multimodal_open-en.md)**

:   提出 CrafText，一个基于 Craftax 开放世界环境的多模态指令跟随基准，包含 3,924 条指令和 3,423 个独特词汇，覆盖定位、条件、建造和成就四类任务，并设计双重评估协议测试智能体的语言泛化和目标泛化能力。

**[Cultivating Game Sense for Yourself: Making VLMs Gaming Experts](multimodal_vlm/cultivating_gaming_sense_for_yourself_making_vlms_gaming_experts.md)**

:   本文提出了 GameSense 框架，将 VLM 从游戏的直接控制者提升为高级开发者，通过让 VLM 自主观察和开发特定任务的"游戏感"执行模块（从规则到神经网络），首次实现了在动作游戏、射击游戏和休闲游戏等多种类型中的流畅游玩。

**[DALR: Dual-level Alignment Learning for Multimodal Sentence Representation Learning](multimodal_vlm/dalr_dual-level_alignment_learning_for_multimodal_sentence_representation_learni.md)**

:   提出 DALR 框架，通过跨模态一致性学习 + 模态内排序蒸馏的双层对齐策略，解决多模态句子表示中的跨模态不对齐偏差（CMB）和模态内语义分歧（ISD）问题，在 STS 和 TR 任务上取得 SOTA。

**[Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation](multimodal_vlm/do_vision-language_models_have_internal_world_models_towards_an_atomic_evaluatio.md)**

:   提出基于认知科学的双阶段框架（感知+预测），构建 WM-ABench 大规模基准（23 维度、6 模拟器、10 万+实例），通过 660 组实验系统揭示 15 个 SOTA VLM 在基本世界建模能力上的严重不足。

**[Donate or Create? Comparing Data Collection Strategies for Emotion-labeled Multimodal Social Media Posts](multimodal_vlm/donate_or_create_comparing_data_collection.md)**

:   本文系统比较了三种收集作者标注情感数据的策略（创建、捐赠、近期帖子），发现研究创建的数据在文本长度、情感原型性和图文关系上与真实数据存在显著差异，但创建数据仍可有效训练泛化模型，不过真实数据对准确评估模型效果不可或缺。

**[Don't Miss the Forest for the Trees: Attentional Vision Calibration for Large Vision Language Models](multimodal_vlm/dont_miss_the_forest_for_the_trees_attentional_vision_calibration_for_large_visi.md)**

:   发现 LVLM 中存在"blind token"现象——少量语义无关的图像 token 吸引了不成比例的注意力权重，并提出 AvisC 方法通过测试时对比解码重新校准 blind token 影响，有效减轻视觉幻觉。

**[EffiVLM-Bench: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models](multimodal_vlm/effivlm_bench_vlm_acceleration.md)**

:   提出 EffiVLM-Bench 统一评估框架，从性能、泛化性、忠实度和效率四个维度系统评估 LVLM 免训练加速方法（token 压缩 + 参数压缩），覆盖 3 个前沿模型和 17 个基准任务，揭示各方法在不同压缩率下的 Pareto 最优权衡。

**[Enhance Multimodal Consistency and Coherence for Text-Image Plan Generation](multimodal_vlm/enhance_multimodal_consistency_and_coherence_for_text-image_plan_generation.md)**

:   本文提出一种自回归文本-图像计划生成框架（MPlanner），通过四阶段迭代——文本草拟、图像编辑、视觉信息提取、文本精炼——有效提升多模态计划中视觉步骤的连贯性和文本-图像的一致性。

**[Error-driven Data-efficient Large Multimodal Model Tuning](multimodal_vlm/error-driven_data-efficient_large_multimodal_model_tuning.md)**

:   提出一种错误驱动的数据高效微调框架，通过教师模型分析学生模型的错误推理步骤并识别缺失技能，从外部数据集检索针对性训练样本进行微调，无需任务特定数据即可实现平均 7.01% 的性能提升。

**[Evaluating Multimodal Language Models as Visual Assistants for Visually Impaired Users](multimodal_vlm/evaluating_multimodal_language_models_as_visual_assistants_for_visually_impaired.md)**

:   通过用户调查确定视障人群对 AI 视觉助手的核心需求与挑战，设计涵盖图像描述、多语言VQA、光学盲文识别、视频物体识别、视频问答五大用户中心任务的评估框架，系统评测 12 个 MLLM，揭示当前模型在文化理解、多语言支持、盲文阅读、辅助设备识别和幻觉控制方面的显著不足。

**[Evaluating Visual and Cultural Interpretation: The K-Viscuit Benchmark with Human-VLM Collaboration](multimodal_vlm/evaluating_visual_and_cultural_interpretation_the_k-viscuit_benchmark_with_human.md)**

:   本文提出了一种半自动化的文化 VLM 基准构建框架，通过人-VLM 协作生成多选 VQA 样本，并以此构建了聚焦韩国文化的 K-Viscuit 数据集（657 题），揭示了开源与闭源 VLM 在文化理解上的显著差距。

**[Exploring Compositional Generalization of Multimodal LLMs for Medical Imaging](multimodal_vlm/exploring_compositional_generalization_of_multimodal_llms_for_medical_imaging.md)**

:   提出 Med-MAT 数据集（106个医学数据集、53个子集），通过 MAT-Triplet（Modality-Anatomical area-Task）分解医学影像属性，首次系统验证了多模态大模型在医学影像上存在组合泛化（Compositional Generalization）现象，并证明组合泛化是多任务训练泛化增益的关键驱动因素。

**[Exploring How Generative MLLMs Perceive More Than CLIP with the Same Vision Encoder](multimodal_vlm/exploring_how_generative_mllms_perceive_more.md)**

:   系统探究为何生成式多模态LLM（如LLaVA）使用与CLIP相同的视觉编码器却能在视觉推理任务上大幅超越CLIP，发现patch token、位置编码和prompt加权是关键因素。

**[FIHA: Autonomous Fine-grained Hallucination Evaluation in Vision-Language Models with Davidson Scene Graphs](multimodal_vlm/fiha_autonomous_hallucination_evaluation_in_vision-language_models_with_davidson.md)**

:   本文提出 FIHA，一个无需 LLM 和人工标注的自动化细粒度幻觉评估框架，通过从图像和描述中提取实体、属性和关系生成 Q&A 对，并引入 Davidson 场景图（DSG）建模问题间的依赖关系，构建了 FIHA-v1 基准，全面评估了主流大视觉语言模型的幻觉水平。

**[Filter-And-Refine: A MLLM Based Cascade System for Industrial-Scale Video Content Moderation](multimodal_vlm/filter-and-refine_a_mllm_based_cascade_system_for_industrial-scale_video_content.md)**

:   TikTok提出一种基于MLLM的两阶段级联内容审核系统（Router-Ranker），通过轻量级嵌入检索路由器过滤97.5%的合规流量，仅将高风险视频送入微调后的LLaVA进行精细分类，F1提升66.5%的同时部署成本降至直接全量部署的1.5%。

**[Finding Needles in Images: Can Multi-modal LLMs Locate Fine Details?](multimodal_vlm/finding_needles_in_images_can_multi-modal_llms_locate_fine_details.md)**

:   本文提出 NiM 基准数据集，系统评估多模态大模型（MLLM）在复杂文档中定位细粒度信息的能力，并设计 Spot-IT 方法通过智能补丁选择和高斯注意力机制显著提升模型在细节抽取任务上的表现。

**[FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation](multimodal_vlm/finmme_benchmark_dataset_for_financial_multi-modal_reasoning_evaluation.md)**

:   构建了一个包含 11,000+ 高质量金融多模态样本的评估基准 FinMME，涵盖 18 个金融领域和 10 种图表类型，提出了融合幻觉惩罚和领域归一化的 FinScore 评估体系，实验表明即使 GPT-4o 也仅得 47 分，揭示了 MLLM 在金融领域的显著不足。

**[FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation](multimodal_vlm/flagevalmm_a_flexible_framework_for_comprehensive_multimodal_model_evaluation.md)**

:   提出 FlagEvalMM，一个开源的多模态模型评估框架，通过将模型推理与评估过程解耦的架构设计，统一支持视觉语言理解（VQA）、文生图/文生视频生成和图文检索等多种多模态任务的评估。

**[GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art](multimodal_vlm/godbench_a_benchmark_for_multimodal_large_language_models_in_video_comment_art.md)**

:   GODBench 构建了首个系统性评估多模态大模型（MLLM）视频弹幕/评论创作能力的基准，定义了5个创意维度和25个子类别，并提出受物理波纹传播启发的 Ripple of Thought（RoT）多步推理框架来增强模型的创意生成能力。

**[Harnessing PDF Data for Improving Japanese Large Multimodal Models](multimodal_vlm/harnessing_pdf_data_for_improving_japanese_large_multimodal_models.md)**

:   提出一套全自动 PDF 数据提取管道，从日语 PDF 中提取图文对并生成指令数据，通过持续微调 LLaVA1.5 框架显著提升日语多模态模型性能，在 Heron-Bench 上实现 2.1%~13.8% 的提升。

**[Hidden in Plain Sight: Evaluation of the Deception Detection Capabilities of LLMs in Multimodal Settings](multimodal_vlm/hidden_in_plain_sight_evaluation_of_the_deception_detection_capabilities_of_llms.md)**

:   本文系统评估了LLM和多模态大模型在文本、视频、音频等多种模态上的欺骗检测能力，发现微调后的LLM在文本欺骗检测上达到SOTA，但多模态模型在利用跨模态线索方面仍存在显著不足。

**[HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model](multimodal_vlm/hidellava_hierarchical_decoupling_for_continual_instruction.md)**

:   通过 CKA 分析发现 MLLM 顶层学任务特异信息而其余层学通用知识，提出 HiDe-LLaVA：顶层 LoRA 做 MoE 式任务特异扩展（双模态锚点匹配）+ 其余层 LoRA 做均匀融合，在新构建的无信息泄露基准 UCIT 上比最佳基线提升 5.8%。

**[Hierarchical Safety Realignment: Lightweight Restoration of Safety in Pruned Large Vision-Language Models](multimodal_vlm/hierarchical_safety_realignment_lightweight_restoration_of_safety_in_pruned_larg.md)**

:   提出层次化安全重对齐方法HSR，通过先识别安全关键注意力头、再在这些头中定位并恢复被剪枝的安全关键神经元，以极低参数开销（万分之几）显著恢复被剪枝LVLM丢失的安全性能。

**[HotelMatch-LLM: Joint Multi-Task Training of Small and Large Language Models for Efficient Multimodal Hotel Retrieval](multimodal_vlm/hotelmatch_llm_retrieval.md)**

:   提出 HotelMatch-LLM，用 SLM 编码 query + LLM 编码酒店文档的非对称架构，配合三目标多任务优化（检索对齐 + MLM地理预测 + 视觉设施识别）和 patch 级 mean pooling 多图处理，在旅行领域多模态检索任务上显著超过 MARVEL/VISTA 等 SOTA。

**[HSCR: Hierarchical Self-Contrastive Rewarding for Aligning Medical Vision Language Models](multimodal_vlm/hscr_hierarchical_self-contrastive_rewarding_for_aligning_medical_vision_languag.md)**

:   提出层级自对比奖励方法 HSCR，通过视觉 token dropout 暴露模型内在的模态失对齐（misalignment），自动生成高质量偏好数据，并结合显式/隐式多层级偏好优化，仅用2000条训练样本即显著提升医学VLM的零样本性能和可信度。

**[I See What You Mean: Co-Speech Gestures for Reference Resolution in Multimodal Dialogue](multimodal_vlm/i_see_what_you_mean_co-speech_gestures_for_reference_resolution_in_multimodal_di.md)**

:   提出自监督预训练方法学习表征性共语手势（co-speech iconic gestures）的嵌入表示，将骨骼动作 grounded 到语言中，在面对面对话的指称消解任务上证明手势与语音的互补性——手势+语音准确率 31% 远超单独语音 24% 或手势 19%。

**[Improving Medical Large Vision-Language Models with Abnormal-Aware Feedback](multimodal_vlm/improving_medical_large_vision-language_models_with_abnormal-aware_feedback.md)**

:   提出 UMed-LVLM，通过 Abnormal-Aware Instruction Tuning 和 Abnormal-Aware Rewarding（包含 Relevance Reward、Abnormal Localization Reward、Vision Relevance Reward）训练策略增强医学 LVLM 的异常区域定位能力，在 MAU 数据集上比基线提升 58%，并展现出优秀的跨模态和 OOD 泛化能力。

**[Improving MLLM's Document Image Machine Translation via Synchronously Self-reviewing Its OCR Proficiency](multimodal_vlm/improving_mllms_document_image_machine_translation_via_synchronously_self-review.md)**

:   本文提出 Synchronously Self-Reviewing (SSR) 范式，通过在文档图像翻译过程中让 MLLM 先生成 OCR 文本再生成翻译文本，利用"双语认知优势"缓解微调导致的灾难性遗忘，同时提升 OCR 和文档图像机器翻译（DIMT）的性能。

**[iNews: A Multimodal Dataset for Modeling Personalized Affective Responses to News](multimodal_vlm/inews_a_multimodal_dataset_for_modeling_personalized_affective_responses_to_news.md)**

:   构建了一个包含 291 位英国标注者对 2,899 条 Facebook 多模态新闻帖子的个性化情感标注数据集 iNews，标注者特征（人口统计、人格、媒体信任等）可解释 15.2% 的标注方差，结合 persona 信息的 LLM 零样本预测准确率提升最高 7%。

**[Inference Compute-Optimal Video Vision Language Models](multimodal_vlm/inference_compute_optimal_video_vlm.md)**

:   首次系统性研究视频VLM推理计算预算的最优分配问题：在固定推理FLOPs下，通过大规模训练扫描（~100k A100小时）和add-interact参数化建模（$R^2$=0.98），确定语言模型大小 $x_N$、帧数 $x_T$ 和每帧视觉token数 $x_V$ 三个维度的最优权衡策略。

**[InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model](multimodal_vlm/internlm-xcomposer25-reward_a_simple_yet_effective_multi-modal_reward_model.md)**

:   基于InternLM-XComposer2.5构建判别式多模态奖励模型IXC-2.5-Reward，通过精心构建跨文本/图像/视频的多领域偏好数据集训练，在多模态奖励基准VL-RewardBench上以70.0% Macro Acc超越GPT-4o（62.4%），并展示了RL训练、Best-of-N测试时缩放和数据清洗三大应用。

**[Jailbreak Large Vision-Language Models Through Multi-Modal Linkage](multimodal_vlm/jailbreak_large_vision-language_models_through_multi-modal_linkage.md)**

:   提出多模态链接（MML）攻击框架，通过跨模态加密-解密机制和"邪恶对齐"策略，以极高成功率（GPT-4o上达99%+）越狱当前最先进的视觉语言模型。

**[JARVIS-VLA: Post-Training Large-Scale Vision Language Models to Play Visual Games](multimodal_vlm/jarvis-vla_post-training_large-scale_vision_language_models_to_play_visual_games.md)**

:   提出ActVLP训练范式，在动作模仿学习之前增加视觉语言后训练阶段（世界知识、视觉对齐、空间定位），构建首个能在Minecraft中执行1000+原子任务的VLA模型JARVIS-VLA，相比最佳基线提升40%。

**[Judging the Judges: Can Large Vision-Language Models Fairly Evaluate Chart Comprehension and Reasoning?](multimodal_vlm/judging_the_judges_can_large_vision-language_models_fairly_evaluate_chart_compre.md)**

:   系统评估了 13 个开源小型 LVLM（≤9B 参数）作为图表理解和推理任务的评判者，发现部分开源模型（如 LLaVA-Critic-7B）可达到接近 GPT-4 水平的评判能力（约 80% 一致率），但位置偏差和长度偏差等问题仍然普遍存在。

**[LogicQA: Logical Anomaly Detection with Vision Language Model Generated Questions](multimodal_vlm/logicqa_logical_anomaly_detection_with_vision_language_model_generated_questions.md)**

:   提出 LogicQA 框架，利用预训练 VLM 自动生成异常相关问题并通过问答投票机制检测逻辑异常，在无需训练、无需标注的少样本设置下达到 SOTA 性能，同时提供自然语言的异常原因解释。

**[LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating](multimodal_vlm/longdocurl_multimodal_long_doc.md)**

:   提出 LongDocURL 基准，覆盖理解/数值推理/跨元素定位三大任务类别共 20 个子任务，包含 2325 个高质量 QA 对、覆盖 33000+ 页文档，系统评估 26 种模型配置暴露了当前 LVLM 在长文档理解上的关键性能差距。

**[MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference](multimodal_vlm/madakv_adaptive_modality-perception_kv_cache_eviction_for_efficient_multimodal_l.md)**

:   本文提出MadaKV，一种模态感知的KV缓存逐出策略，通过模态偏好自适应（MPA）和层级压缩补偿（HCC）两个组件，在保持多模态长上下文任务性能的同时，显著降低KV缓存内存占用（80-95%）和解码延迟（1.3-1.5倍加速）。

**[MAGIC-VQA: Multimodal and Grounded Inference with Commonsense Knowledge for Visual Question Answering](multimodal_vlm/magic-vqa_multimodal_and_grounded_inference_with_commonsense_knowledge_for_visua.md)**

:   提出MAGIC-VQA框架，通过三阶段流程（显式知识检索→按类型后处理→GNN隐式增强）将外部常识知识系统地注入LVLM，在ScienceQA、TextVQA、MMMU等基准上实现即插即用的常识推理增强，仅需0.33M可训练参数。

**[MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale](multimodal_vlm/mammoth_vl_multimodal_reasoning.md)**

:   提出一种可扩展、低成本的方法，仅使用开源模型构建含 1200 万条富含中间推理过程 (CoT) 的多模态指令微调数据集 MAmmoTH-VL-Instruct，训练的 MAmmoTH-VL-8B 在推理基准上达到 SOTA（MathVerse +8.1%, MMMU-Pro +7%, MuirBench +13.3%）。

**[MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning](multimodal_vlm/mathcoder-vl_bridging_vision_and_code_for_enhanced_multimodal_mathematical_reaso.md)**

:   提出利用代码作为跨模态对齐的监督信号，构建860万图像-代码对数据集ImgCode-8.6M和300万多模态数学指令微调数据集MM-MathInstruct-3M，训练的MathCoder-VL在开源模型中达到多模态数学推理SOTA，在几何问题上超越GPT-4o和Claude 3.5 Sonnet。

**[Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search](multimodal_vlm/mcts_video_captioning_eval.md)**

:   提出AutoCaption框架，利用蒙特卡洛树搜索(MCTS)自动迭代生成细粒度视频描述关键点（平均122个/视频），构建MCTS-VCB基准评估20+个MLLM的视频描述能力，并证明生成的数据可用于微调显著提升模型性能。

**[MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval](multimodal_vlm/megapairs_massive_data_synthesis_for_universal_multimodal_retrieval.md)**

:   提出 MegaPairs 数据合成方法，利用异构 KNN 三元组从开放域图像语料中挖掘相关图像对，结合 VLM/LLM 生成检索指令，合成 2600 万多模态训练实例，训练的 MMRet 模型仅用 0.5M 数据即超越使用 36.7M 数据的 MagicLens（70× 数据效率），在 4 个 CIR 基准和 MMEB 36 个数据集上达到 SOTA。

**[MEIT: Multimodal Electrocardiogram Instruction Tuning on Large Language Models for Report Generation](multimodal_vlm/meit_multimodal_electrocardiogram_instruction_tuning_on_large_language_models_fo.md)**

:   提出 MEIT 框架，通过多模态指令微调将 ECG 信号与 LLM 对齐，利用轻量级拼接融合策略（无需额外参数）在 LLM 的自注意力层中注入 ECG 嵌入，实现自动 ECG 报告生成，并建立涵盖质量评估、零样本迁移、噪声鲁棒性和专家对齐四项任务的综合基准。

**[MIRA: Empowering One-Touch AI Services on Smartphones with MLLM-based Instruction Recommendation](multimodal_vlm/mira_empowering_one-touch_ai_services_on_smartphones_with_mllm-based_instruction.md)**

:   提出 MIRA 框架，通过结构化推理、模板增强推理和前缀树约束解码，让用户在智能手机上长按文本或图片即可获得上下文相关的 AI 服务指令推荐，在 7B 模型上超越 GPT-4V（F1: 0.9121 vs 0.879），token 使用量仅为 1/7。

**[MIRe: Enhancing Multimodal Queries Representation via Fusion-Free Modality Interaction](multimodal_vlm/mire_enhancing_multimodal_queries_representation_via_fusion-free_modality_intera.md)**

:   提出MIRe框架，通过"无融合模态交互"（fusion-free modality interaction）在视觉-文本对齐阶段避免直接融合文本特征，利用查询引导注意力池化模块让文本嵌入引导视觉信息提取但不将文本信号反馈回视觉表示，有效缓解多模态检索中的文本主导问题，在四个基准上取得零样本SOTA。

**[Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucination in Multimodal LLMs](multimodal_vlm/mixture_of_decoding_an_attention-inspired_adaptive_decoding_strategy_to_mitigate.md)**

:   提出了 Mixture of Decoding (MoD)，通过 JS 散度衡量模型对图像 token 注意力的正确性，在注意力正确时采用互补解码放大关键信息，注意力错误时采用对比解码抑制误导信息，从而自适应地缓解多模态大模型的幻觉问题。

**[MMBoundary: Advancing MLLM Knowledge Boundary Awareness through Reasoning Step Confidence Calibration](multimodal_vlm/mmboundary_reasoning_step_confidence.md)**

:   提出 MMBoundary 框架，通过在推理链的每一步插入自然语言置信度表述（而非只在最终回答后给置信度），结合文本+跨模态的自奖励信号估计置信度，并用 SFT+RL 两阶段训练实现步级置信度校准，平均降低 7.5% 校准误差并提升 8.3% 任务准确率。

**[MMINA: Benchmarking Multihop Multimodal Internet Agents](multimodal_vlm/mmina_benchmarking_multihop_multimodal_internet_agents.md)**

:   提出MMInA基准，包含1,050个人工编写的多跳多模态网页任务（覆盖14个真实动态网站，平均2.85跳），并设计逐跳评估协议和记忆增强方法，揭示当前最强Agent（GPT-4V仅21.8%任务成功率）在多跳网页导航上与人类（96.3%）的巨大差距。

**[MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark](multimodal_vlm/mmmu_pro_robust_benchmark.md)**

:   在 MMMU 基础上通过三步加固（过滤纯文本可解题目、扩展选项至 10 个、引入 Vision-only 输入）构建更鲁棒的 MMMU-Pro 基准，所有模型性能下降 16.8%~26.9%，揭示当前多模态模型远未实现真正的跨模态理解。

**[MMSciBench: Benchmarking Language Models on Chinese Multimodal Scientific Problems](multimodal_vlm/mmscibench_benchmarking_language_models_on_chinese_multimodal_scientific_problem.md)**

:   提出 MMSciBench，一个包含 4,482 道中文高中数学和物理题目的多模态科学推理基准，涵盖选择题和问答题、纯文本和图文配对两种模态，并带有人工标注难度等级和三级知识分类体系；评估显示最强模型 Gemini 1.5 Pro 002 仅达 63.77% 准确率，且在图文题上性能大幅下降（36.28 个百分点差距）。

**[MultiMM: Cultural Bias Matters — Cross-Cultural Benchmark for Multimodal Metaphors](multimodal_vlm/multimm_cultural_metaphor.md)**

:   提出MultiMM——首个跨文化多模态隐喻数据集，包含8461个中英文广告图文对及细粒度标注，并设计SEMD模型融合情感特征增强隐喻检测。

**[Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](multimodal_vlm/multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)**

:   提出 TikTalkCoref，首个面向中文社交媒体对话的多模态共指消解数据集（基于抖音短视频），并构建了包含文本共指消解、视觉人物追踪和跨模态对齐三个模块的 pipeline benchmark。

**[NegVQA: Can Vision Language Models Understand Negation?](multimodal_vlm/negvqa_can_vision_language_models_understand_negation.md)**

:   提出 NegVQA 基准（7,379 道二选一 VQA 题），系统评估 20 个 VLM 对否定句的理解能力，发现所有模型在否定问题上性能大幅下降（平均 29.7%），并揭示"U 型"缩放趋势。

**[OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval](multimodal_vlm/omgm_orchestrate_multiple_granularities_and_modalities_for_efficient_multimodal_.md)**

:   提出OMGM——一个面向知识密集型视觉问答(KB-VQA)的多模态RAG系统，通过粗到细三步检索策略协调查询与知识库在不同粒度和模态间的匹配，在InfoSeek和E-VQA上取得SOTA检索性能和极具竞争力的问答结果。

**[OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference](multimodal_vlm/omnialign-v_towards_enhanced_alignment_of_mllms_with_human_preference.md)**

:   构建了 OmniAlign-V（200K 高质量多模态 SFT 数据集）和 MM-AlignBench 评测基准，通过多样化图片来源、开放式问题设计和多样化回答格式，显著提升开源 MLLM 的人类偏好对齐能力，使 LLaVA-Next-32B 经 SFT+DPO 后超越 Qwen2VL-72B。

**[A Survey on Patent Analysis: From NLP to Multimodal AI](multimodal_vlm/patent_analysis_survey.md)**

:   系统综述了 NLP 和多模态 AI 在专利分析四大核心任务（分类、检索、质量分析、生成）中的应用，提出基于专利生命周期的分类体系，揭示了从 Word2Vec+LSTM 到 BERT/GPT 再到多模态模型的方法演进趋势及重要研究空白。

**[Performance Gap in Entity Knowledge Extraction Across Modalities in Vision Language Models](multimodal_vlm/performance_gap_in_entity_knowledge_extraction_across_modalities_in_vision_langu.md)**

:   系统性地揭示了视觉语言模型（VLM）在视觉 vs 文本表征下实体知识提取的显著性能差距（最高达 18%），通过机制可解释性工具发现图像 token 的关键信息流发生在模型中间层很深处，导致后续事实推理的层数不足。

**[Progressive Multimodal Reasoning via Active Retrieval](multimodal_vlm/progressive_multimodal_reasoning_via_active_retrieval.md)**

:   本文提出AR-MCTS框架，将主动检索（Active Retrieval）与蒙特卡洛树搜索（MCTS）结合，在多步多模态推理的每一步动态检索关键知识来替代传统beam search采样，自动生成逐步推理标注以渐进式对齐过程奖励模型（PRM），在MathVista、We-Math和GAOKAO-MM上显著提升了多种MLLM的推理性能。

**[PunchBench: Benchmarking MLLMs in Multimodal Punchline Comprehension](multimodal_vlm/punchbench_mllm_punchline.md)**

:   本文提出PunchBench，一个包含6,000个图文对和54,000个问答对的多模态幽默/讽刺理解基准，通过同义/反义标题生成消除语言捷径，同时提出Simple-to-Complex Chain-of-Question (SC-CoQ)策略，在所有模型和问题格式上一致性提升punchline理解能力。

**[R-VLM: Region-Aware Vision Language Model for Precise GUI Grounding](multimodal_vlm/r-vlm_region-aware_vision_language_model_for_precise_gui_grounding.md)**

:   提出R-VLM，将传统目标检测中的区域提议（region proposal）和IoU感知损失引入VLM的GUI元素定位，通过两阶段放大推理和IoU加权交叉熵损失，在ScreenSpot和AgentStudio上平均提升13%的grounding准确率。

**[RATE-Nav: Region-Aware Termination Enhancement for Zero-shot Object Navigation with Vision-Language Models](multimodal_vlm/rate-nav_region-aware_termination_enhancement_for_zero-shot_object_navigation_wi.md)**

:   提出 RATE-Nav，一种基于边际效用理论的零样本目标导航方法，通过几何预测区域分割和基于区域的探索率估计，结合 VLM 的宏观环境感知能力智能判断是否终止当前区域的探索，在 HM3D 上达到 67.8% 成功率和 31.3% SPL，在 MP3D 上比先前零样本方法提升约 10%。

**[Redundancy Principles for MLLMs Benchmarks](multimodal_vlm/redundancy_principles_for_mllms_benchmarks.md)**

:   本文从维度冗余、实例冗余和跨基准冗余三个层面系统量化了当前MLLM评测基准中的冗余现象，提出了基于性能排名相关性的冗余分析框架，为未来基准设计提供了原则性指导。

**[RedundancyLens: Revealing and Exploiting Visual Token Processing Redundancy for Efficient Decoder-Only MLLMs](multimodal_vlm/redundancylens_revealing_and_exploiting_visual_token_processing_redundancy_for_e.md)**

:   提出 RedundancyLens 框架，系统揭示了 decoder-only MLLM 中视觉 token 在自注意力和 FFN 操作上存在大量结构化、聚簇式冗余，并利用这一发现实现免训练推理加速，与现有 token 压缩方法正交且可组合。

**[ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models](multimodal_vlm/reefknot_a_comprehensive_benchmark_for_relation_hallucination_evaluation_analysi.md)**

:   提出首个系统性评估多模态大模型**关系级幻觉**的综合基准 Reefknot（含 2 万+ 样本、三种任务），并基于置信度熵检测提出 Detect-then-Calibrate 缓解策略，平均降低幻觉率 9.75%。

**[Response Wide Shut: Surprising Observations in Basic Vision Language Model Capabilities](multimodal_vlm/response_wide_shut_surprising_observations_in_basic_vision_language_model_capabi.md)**

:   通过在VLM的三个中间特征空间（视觉编码器、VL投影层、语言解码器）上训练线性探针，系统揭示了一个反直觉的现象：对于大多数视觉任务，视觉编码器和VL投影层其实保留了充分的视觉信息，真正的瓶颈在于语言解码器的响应空间——信息在从投影层传递到最终文本输出的过程中大量丢失。

**[Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models](multimodal_vlm/retrieval_visual_contrastive_decoding_to_mitigate_object_hallucinations_in_large.md)**

:   提出 RVCD（Retrieval Visual Contrastive Decoding），通过检索 AI 生成的单概念显式图像构建正/负 logit 集合，在解码阶段抑制 LVLM 的物体幻觉（Object Hallucination），无需额外训练即可显著优于现有解码方法。

**[Scalable Vision Language Model Training via High Quality Data Curation](multimodal_vlm/scalable_vision_language_model_training_via_high_quality_data_curation.md)**

:   提出 SAIL-VL 系列开源视觉语言模型（2B/8B），核心贡献在于：构建了3亿规模最高质量的 SAIL-Caption 数据集，首次揭示了VLM预训练中的数据量对数缩放定律（655B token实验），并通过课程式三阶段SFT将缩放曲线从对数提升至近线性，在18个基准上达到SOTA。

**[SciVer: Evaluating Foundation Models for Multimodal Scientific Claim Verification](multimodal_vlm/sciver_evaluating_foundation_models_for_multimodal_scientific_claim_verification.md)**

:   SciVer 是首个面向多模态科学文献声称验证的基准数据集，包含 3000 个专家标注样本覆盖 1113 篇 CS 论文，设计了直接/并行/序列/分析四种推理子任务，评估 21 个基础模型后发现最强模型 o4-mini（77.7%）与人类专家（93.8%）仍有 16% 的显著差距。

**[SemEval-2025 Task 1: AdMIRe -- Advancing Multimodal Idiomaticity Representation](multimodal_vlm/semeval-2025_task_1_admire_--_advancing_multimodal_idiomaticity_representation.md)**

:   设计了 SemEval-2025 AdMIRe 共享任务——通过图像排序和图像序列补全两个子任务，在多模态（文本+图像）和多语言（英语+巴西葡萄牙语）场景下评估模型对习语表达的理解能力，最佳系统通过混合专家和多查询平滑策略达到了接近人类水平的表现。

**[SingaKids: A Multilingual Multimodal Dialogic Tutor for Language Learning](multimodal_vlm/singakids_a_multilingual_multimodal_dialogic_tutor_for_language_learning.md)**

:   提出 SingaKids 系统，一个面向小学生的多语言多模态对话式语言学习辅导系统，通过图像描述任务整合稠密图像字幕、多语言对话、语音理解和儿童友好语音生成，支持英语、中文、马来语和泰米尔语四种语言的互动学习。

**[Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Translation](multimodal_vlm/single-to-mix_modality_alignment_with_multimodal_large_language_model_for_docume.md)**

:   本文提出 M4Doc，一种基于"单模态到混合模态对齐"的文档图像机器翻译框架，在训练阶段利用多模态大语言模型（MLLM）的视觉-文本联合表示来增强轻量级图像编码器，推理时丢弃 MLLM 以保持高效，在跨域泛化和复杂文档场景中取得了显著的翻译质量提升。

**[Sharper and Faster mean Better: Towards More Efficient Vision-Language Model for Hour-scale Long Video Understanding](multimodal_vlm/sophia_efficient_long_video.md)**

:   提出Sophia模型处理小时级长视频：通过Shot-adaptive Frame Pruning（基于镜头分割的两阶段帧剪枝）精准选择查询相关帧，结合O(N)复杂度的Hierarchical Attention替代全注意力，在8个长视频benchmark中6个SOTA，且注意力FLOPs仅为InternVL2的1/8.5。

**[SpaRE: Enhancing Spatial Reasoning in Vision-Language Models with Synthetic Data](multimodal_vlm/spare_enhancing_spatial_reasoning_in_vision-language_models_with_synthetic_data.md)**

:   本文发现现有VLM数据集中空间关系数据严重匮乏（前17%的关系占据90%以上样本），提出从DOCCI、Localized Narratives和PixMo-Cap等超详细图像描述数据集中，利用LLM自动提取45.5万样本（340万QA对）的空间推理合成数据，微调后的SpaRE模型在What's Up基准上实现最高49%的性能提升，同时不损害通用VL能力。

**[Can Multimodal Large Language Models Understand Spatial Relations?](multimodal_vlm/spatialmqa_mllm_spatial_relations.md)**

:   提出 SpatialMQA 基准，以多选题形式评估 MLLM 的空间关系推理能力，发现 SOTA 模型仅达 48.14% 准确率，远低于人类 98.40%。

**[Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues](multimodal_vlm/speaking_beyond_language.md)**

:   提出 VENUS——首个大规模多模态对话数据集（89,459 段对话、14,910 小时），包含时间对齐的文本、3D 面部表情和肢体语言标注；基于该数据集开发 MARS 多模态语言模型，通过 VQ-VAE 将非语言线索离散化后与文本统一建模，实现对话中文本与非语言动作的联合理解和生成。

**[SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation](multimodal_vlm/sphere_unveiling_spatial_blind_spots_in.md)**

:   提出 SPHERE 三层级空间推理评估框架（单技能→多技能→推理），基于 MS COCO 人工标注 2285 个 QA 对，发现 GPT-4o（67.9%）与人类（93.0%）差距 25%，尤其在距离判断、视角切换和物理推理上表现严重不足。

**[Symmetrical Visual Contrastive Optimization: Aligning Vision-Language Models with Minimal Contrastive Images](multimodal_vlm/symmetrical_visual_contrastive_optimization_aligning_visionlanguage.md)**

:   提出 S-VCO（对称视觉对比优化），一种新的 VLM 微调目标，通过对称地对齐/拒绝匹配/矛盾的图像-文本对来增强视觉依赖，配合最小视觉对比数据集 MVC，在幻觉检测上减少 22%，视觉依赖任务上显著提升。

**[Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](multimodal_vlm/table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)**

:   提出 TableEval 基准（3017 张表格，5 种格式），系统比较了文本 LLM 和多模态 LLM 在科学 vs. 非科学表格理解任务上的表现，发现模型对表格模态（图像/文本）保持鲁棒但在科学表格上性能显著下降。

**[Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions](multimodal_vlm/teaching_vlm_ask_ambiguity.md)**

:   提出 ClearVQA 基准和自动化数据生成管线，让 VLM 学会在遇到歧义视觉问题时主动提出澄清问题而非强行作答，通过三类歧义分类（引用歧义、属性歧义、关系歧义）系统化交互式 VQA，实验证明训练后 VLM 能显著提升歧义识别和澄清质量，获 ACL 2025 SAC Highlight Award。

**[The Role of Visual Modality in Multimodal Mathematical Reasoning: Challenges and Insights](multimodal_vlm/the_role_of_visual_modality_in_multimodal_mathematical_reasoning_challenges_and_.md)**

:   系统性揭示了现有多模态数学推理模型对视觉信息的利用极其有限——打乱或移除训练图像对模型性能影响甚微——并提出 HC-M3D 基准来真正测试视觉依赖性，发现主流模型无法识别图像中的细微差异。

**[TheoremExplainAgent: Towards Video-based Multimodal Explanations for LLM Theorem Understanding](multimodal_vlm/theorem_explain_agent.md)**

:   提出 TheoremExplainAgent，一个双 Agent 系统（Planner + Coder），通过 Manim 动画脚本自动生成长达 10 分钟的定理讲解视频，配套 TheoremExplainBench（240 个 STEM 定理 × 5 维评估指标），证明 agentic planning 是长视频生成的关键，且视觉解释能暴露文本评估无法发现的推理缺陷。

**[Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?](multimodal_vlm/token_pruning_in_multimodal_large_language_models_are_we_solving_the_right_probl.md)**

:   通过大规模基准实验揭示了当前MLLM视觉token剪枝方法的多个根本性问题：精心设计的剪枝策略（FastV、SparseVLM）在多数基准上甚至不如随机选择和池化等朴素方法，原因在于注意力评分的位置偏差、对语言信息的误用、重要性与冗余性的失衡以及评估指标的不可靠。

**[Transferring Textual Preferences to Vision-Language Understanding through Model Merging](multimodal_vlm/transferring_textual_preferences_to_vision-language_understanding_through_model_.md)**

:   提出一种免训练方法，通过模型参数合并（model merging）将纯文本奖励模型（RM）的偏好能力迁移到大视觉语言模型（LVLM）中，构建视觉语言奖励模型（VLRM），在多个多模态评估基准上超越LVLM直接评分和纯文本RM。

**[Mitigating Visual Forgetting via Take-along Visual Conditioning for Multi-modal Long CoT Reasoning](multimodal_vlm/tvc_mitigating_visual_forgetting.md)**

:   发现 MLLM 在长链 CoT 推理中存在严重的视觉遗忘现象——推理过半后移除图像仅导致 ~2% 的准确率下降，表明模型过度依赖自生成文本而忽视视觉证据。提出 TVC (Take-along Visual Conditioning) 策略，在训练阶段通过动态视觉重确认 (DVR) 注入图像回顾机制，推理阶段通过周期性视觉校准 (PVC) 压缩并重注入视觉 token，在 5 个数学推理基准上平均超越 SOTA 3.4 分（43.4 vs 40.0）。

**[Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models](multimodal_vlm/unsolvable_problem_detection.md)**

:   提出 Unsolvable Problem Detection (UPD) 任务，通过三类不可解问题（缺失答案、不兼容选项、图文不匹配）系统评估大型多模态模型在面对无法回答的 MCQA 问题时是否能正确拒绝作答，揭示了现有 benchmark 无法衡量的可信度维度。

**[Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension](multimodal_vlm/unveiling_cultural_blind_spots_analyzing_the_limitations_of_mllms_in_procedural_.md)**

:   提出 CAPTex 基准，通过跨 7 个国家/语言的文化程序性文本理解任务（步骤排序、选择题、对话推理等），系统揭示了多语言大模型在文化特定程序性文本理解上的盲区和局限。

**[Unveiling the Lack of LVLM Robustness to Fundamental Visual Variations: Why and Path Forward](multimodal_vlm/unveiling_the_lack_of_lvlm_robustness_to_fundamental_visual_variations_why_and_p.md)**

:   提出 V2R-Bench 基准框架系统评估 21 个 LVLM 对位置/尺度/方向/上下文四种基本视觉变化的鲁棒性，揭示了即使先进模型在简单视觉任务上也存在显著脆弱性，并通过组件级分析证明这些漏洞根源在于多模态对齐不足和流水线架构的误差累积，而非数据不足。

**[Value-Spectrum: Quantifying Preferences of Vision-Language Models via Value Decomposition](multimodal_vlm/value_spectrum_vlm_pref.md)**

:   提出 Value-Spectrum 基准，通过 50K+ 社交媒体短视频截图和 Schwartz 价值理论框架，系统评估 VLM 的内在价值偏好及角色扮演时的偏好适配能力。

**[VF-Eval: Evaluating Multimodal LLMs for Generating Feedback on AIGC Videos](multimodal_vlm/vf_eval_aigc_video_feedback.md)**

:   提出VF-Eval基准，通过一致性验证、错误感知、错误类型检测、推理评估四大任务系统评估13个MLLM为AIGC视频提供反馈的能力，发现即使GPT-4.1也难以在所有任务上表现一致，揭示了AIGC视频理解的挑战性。

**[ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding](multimodal_vlm/vigil3d_a_linguistically_diverse_dataset_for_3d_visual_grounding.md)**

:   提出 ViGiL3D——一个语言多样性诊断数据集和自动化分析框架，用于评估 3D 视觉定位（3DVG）方法在否定、粗粒度指代、共指消解等多种语言现象上的表现，揭示现有方法在分布外提示上性能显著下降（最高达 20+ 点）。

**[Vision-Language Models Struggle to Align Entities across Modalities](multimodal_vlm/vision-language_models_struggle_to_align_entities_across_modalities.md)**

:   提出 MATE 基准（5,500 个问答实例），通过合成 3D 场景的跨模态属性检索任务系统评估 VLM 的实体链接能力，发现即使最强闭源模型仍落后人类约 15 个百分点，且性能随场景物体数量增加急剧下降——根源在于跨模态特征绑定而非单模态感知。

**[Visual Evidence Prompting Mitigates Hallucinations in Large Vision-Language Models](multimodal_vlm/visual_evidence_prompting.md)**

:   提出Visual Evidence Prompting (VEP)，利用小型视觉专家模型（目标检测器、场景图生成器）的输出作为文本化"视觉证据"输入LVLM，无需训练即可在11个LVLM上显著降低幻觉——LLaVA-1.5在POPE上提升7.2%、Claude 3上提升12.1%。

**[VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search](multimodal_vlm/visuothink_empowering_lvlm_reasoning_with_multimodal_tree_search.md)**

:   本文提出VisuoThink框架，通过视觉-文本交织推理和预测性前瞻树搜索，在推理过程中动态整合视觉辅助信息并探索多条推理路径，无需微调即可在几何和空间推理任务上实现SOTA性能（Geomverse-109上Accuracy@1最高达48.5%，相比最优基线提升21.8%）。

**[VLM2-Bench: A Closer Look at How Well VLMs Implicitly Link Explicit Matching Visual Cues](multimodal_vlm/vlm2-bench_a_closer_look_at_how_well_vlms_implicitly_link_explicit_matching_visu.md)**

:   本文提出VLM2-Bench，一个专门评估视觉语言模型（VLM）跨图像/帧"视觉线索关联"能力的基准，涵盖通用线索、物体中心线索和人物中心线索3大类9个子任务共3000+测试样本，发现即使最先进的商业模型在该任务上也落后人类30%以上，揭示了VLM在基础视觉匹配能力上的重大差距。

**[VLMInferSlow: Evaluating the Efficiency Robustness of Large Vision-Language Models as a Service](multimodal_vlm/vlminferslow_evaluating_the_efficiency_robustness_of.md)**

:   首次在黑盒设置下研究 VLM 的效率鲁棒性，提出 VLMInferSlow 方法，通过零阶优化搜索对抗性图像扰动，迫使 VLM 生成更长序列，将计算成本最高增加 128.47%，揭示了 VLM 在 MLaaS 部署场景下的效率安全隐患。

**[VLSBench: Unveiling Visual Leakage in Multimodal Safety](multimodal_vlm/vlsbench_unveiling_visual_leakage_in_multimodal_safety.md)**

:   揭示现有多模态安全基准中存在的视觉安全信息泄露（VSIL）问题——图像中的危险内容已在文本查询中暴露，导致模型仅凭文本即可拒绝，从而使安全评估不可靠；为此构建了无泄露的VLSBench基准（2.2k图文对），发现多模态对齐在无VSIL场景中显著优于纯文本对齐。

**[VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism](multimodal_vlm/vrest_tree_search_vlm_reasoning.md)**

:   首次将蒙特卡洛树搜索(MCTS)引入多模态CoT推理，配合无需额外模型的多模态自奖励机制系统性探索推理空间，在三个视觉数学推理基准上实现SOTA并验证了多模态测试时缩放定律。

**[Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains](multimodal_vlm/weaving_context_across_images_improving_vision-language_models_through_focus-cen.md)**

:   提出 Focus-Centric Visual Chain 多图推理范式，通过问题分解和逐步聚焦关键视觉信息实现跨图推理，并构建 VISC-150K 数据集，在七个多图基准上实现 2-3% 的一致性提升。

**[We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?](multimodal_vlm/wemath_knowledge_reasoning.md)**

:   本文提出We-Math基准，包含6.5K视觉数学问题和67个层次化知识概念，通过将复合问题分解为子问题引入四维评估指标（知识不足IK、泛化不足IG、完全掌握CM、机械记忆RM），首次从知识掌握角度系统评估LMM的数学推理过程而非仅关注最终结果。

**[WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts](multimodal_vlm/wikimixqa_a_multimodal_benchmark_for_question_answering_over_tables_and_charts.md)**

:   提出 WikiMixQA 基准，包含 1,000 道需要跨表格和图表进行多模态推理的多选题，评估 12 个 VLLM 后发现闭源模型在提供精确上下文时准确率约 70%，但需从长文档检索时性能骤降，开源模型最高仅 27%，揭示了当前视觉语言模型在长上下文多模态文档理解上的严重不足。

---

## 🌐 多语言/翻译 { #multilingual_mt }

**[A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs](multilingual_mt/a_case_study_of_cross-lingual_zero-shot_generalization_for_classical_languages_i.md)**

:   系统评估 LLM 在三种古典语言（梵语、古希腊语、拉丁语）上的零样本跨语言泛化能力，涵盖 NER、机器翻译和问答三个 NLU 任务，同时贡献 1501 对梵语问答数据集并验证 RAG 策略的有效性，揭示模型规模是跨语言泛化的决定性因素。

**[Accessible Machine Translation Evaluation For Low-Resource Languages](multilingual_mt/accessible_machine_translation_evaluation_for_low-resource_languages.md)**

:   本文针对低资源语言的机器翻译评估困境，提出了一套不依赖高质量参考翻译和大规模标注数据的可访问评估框架，使得资源匮乏的语言可以有效评估翻译质量。

**[Alleviating Distribution Shift in Synthetic Data for Machine Translation Quality Estimation](multilingual_mt/alleviating_distribution_shift_in_synthetic_data_for_machine_translation_quality.md)**

:   提出 DCSQE 框架，通过约束波束搜索生成更真实的合成翻译、利用独立的标注模型纠正标签偏差、以及 SPCE 算法将 token 级标签聚合为短语级标签，有效缓解合成 QE 数据的分布偏移问题，在有监督和无监督设置下均超越 CometKiwi 等 SOTA 基线。

**[An Expanded Massive Multilingual Dataset for High-Performance Language Technologies (HPLT)](multilingual_mt/an_expanded_massive_multilingual_dataset_for_high-performance_language_technolog.md)**

:   本文介绍 HPLT v2，一个从 4.5 PB 的 Internet Archive 和 Common Crawl 数据中提取的大规模多语言数据集，包含覆盖 193 种语言的 8 万亿 token 单语数据和覆盖 51 种语言的 3.8 亿句对平行数据，并通过改进的数据处理管线显著提升了数据质量。

**[Are Rules Meant to be Broken? Understanding Multilingual Moral Reasoning as a Computational Pipeline with UniMoral](multilingual_mt/are_rules_meant_to_be_broken_understanding_multilingual_moral_reasoning_as_a_com.md)**

:   提出UniMoral——一个跨6种语言的统一道德推理数据集，将道德推理建模为包含行为预测、道德类型分类、因素归因和后果生成的计算流水线，对3个LLM的基准测试揭示隐式道德语境能增强模型道德推理能力但仍需专门化方法。

**[AskQE: Question Answering as Automatic Evaluation for Machine Translation](multilingual_mt/askqe_question_answering_as_automatic_evaluation_for_machine_translation.md)**

:   提出 AskQE——基于问答的机器翻译质量估计框架，通过对源文本生成问题、分别在源文本和回译输出上回答、对比答案差异来检测翻译错误，帮助不懂目标语言的用户判断翻译是否可接受，在 BioMQM 数据集上 Kendall's τ 相关和决策准确率均优于现有 QE 指标。

**[7 Points to Tsinghua but 10 Points to 清华? Assessing Agentic Large Language Models in Multilingual National Bias](multilingual_mt/assessing_agentic_large_language_models_in_multilingual_national_bias.md)**

:   首次系统研究LLM作为多语言智能建议agent在推理型决策任务中的国籍偏见，通过大学申请/旅行/搬迁三类场景+Thurstone比较法量化GPT-3.5/GPT-4/Claude Sonnet在6种语言下的评分偏差，发现"本地语言偏见"（local language bias）普遍存在，且CoT推理在非英语语言中反而加剧偏见。

**[Beyond N-Grams: Rethinking Evaluation Metrics and Strategies for Multilingual Abstractive Summarization](multilingual_mt/beyond_n-grams_rethinking_evaluation_metrics_and_strategies_for_multilingual_abs.md)**

:   系统评估了 n-gram 和神经网络评估指标在 8 种语言（4 个形态类型族）上与人类判断的相关性，发现 n-gram 指标在高融合语言（阿拉伯语、希伯来语）上与人类判断负相关，而专门训练的神经指标 COMET 在所有语言类型上一致优于其他方法。

**[Blessing of Multilinguality: A Systematic Analysis of Multilingual In-Context Learning](multilingual_mt/blessing_of_multilinguality_a_systematic_analysis_of_multilingual_in-context_lea.md)**

:   系统分析多语言 ICL 策略，发现在 prompt 中混合多种高资源语言（HRL）的 demonstrations 一致性优于纯英文 demonstrations，尤其在低资源语言（LRL）上提升显著（Llama3.1 上 LRL 平均准确率提升 8.9~12.6%），甚至仅在 prompt 中加入不相关的非英语句子也能带来可测量的增益，揭示了"多语言暴露本身即有效"的现象。

**[Bridging the Language Gaps in Large Language Models with Inference-Time Cross-Lingual Intervention](multilingual_mt/bridging_the_language_gaps_in_large_language_models_with_inference-time_cross-li.md)**

:   本文提出INCLINE（Inference-Time Cross-Lingual Intervention），一种无需微调的推理时框架，通过学习对齐矩阵将低性能语言的内部表示变换到高性能语言的表示空间，在9个基准和5个LLM上显著提升多语言性能。

**[CC-Tuning: A Cross-Lingual Connection Mechanism for Improving Joint Multilingual Supervised Fine-Tuning](multilingual_mt/cc-tuning_a_cross-lingual_connection_mechanism_for_improving_joint_multilingual_.md)**

:   本文提出 CC-Tuning，一种在隐空间层面显式建立跨语言连接的多语言微调范式，通过融合英语和非英语输入的前馈激活来提升非英语语言的能力，并在推理时利用 Transform Matrix 模拟跨语言连接。

**[CCHall: A Novel Benchmark for Joint Cross-Lingual and Cross-Modal Hallucinations Detection in Large Language Models](multilingual_mt/cchall_a_novel_benchmark_for_joint_cross-lingual_and_cross-modal_hallucinations_.md)**

:   提出首个**联合跨语言与跨模态**幻觉检测基准 CCHall，覆盖 9 种语言和 4 类多模态数据集，系统评估 6 款主流 MLLM 在联合场景下的幻觉表现，揭示当前模型在该联合场景中 F1 比单独跨模态低 10.9、比单独跨语言低 3.4，且提出多语提示和外部工具辅助两条缓解路径。

**[CLIX: Cross-Lingual Explanations of Idiomatic Expressions](multilingual_mt/clix_cross-lingual_explanations_of_idiomatic_expressions.md)**

:   提出跨语言习语解释任务 CLIX，构建了包含英语习语及其西班牙语/德语解释的数据集，系统评估了 seq2seq 模型和 LLM 在该任务上的表现，发现 GPT-3.5 Turbo 的 pipeline 策略（先英文解释再翻译）配合 few-shot 效果最佳，人工评估流畅度和准确度高达 4.7+/5。

**[Code-Switching Curriculum Learning for Multilingual Transfer in LLMs](multilingual_mt/code-switching_curriculum_learning_for_multilingual_transfer_in_llms.md)**

:   本文受人类第二语言习得中语码转换现象的启发，提出 CSCL（Code-Switching Curriculum Learning）框架，通过"token 级 CS → 句子级 CS → 单语语料"的渐进式课程训练策略来增强 LLM 的跨语言迁移能力，在韩语、日语、印尼语等目标语言上显著优于单语持续预训练方法。

**[Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding](multilingual_mt/code-switching_red-teaming_llm_evaluation_for_safety_and_multilingual_understand.md)**

:   本文提出 CSRT（Code-Switching Red-Teaming）框架，利用日常生活中常见的语码转换（code-switching）现象来构造多语言混合的红队攻击查询，成功在 10 个主流 LLM 上发现了严重的安全漏洞，攻击成功率比标准英语攻击高出 46.7%，揭示了当前 LLM 安全对齐在多语言场景下的脆弱性。

**[Comparative Analysis of Multilingual Hate Speech Detection](multilingual_mt/comparative_analysis_of_multilingual_hate_speech_detection.md)**

:   本文系统比较了多种LLM和预训练语言模型在多语言仇恨言论检测任务上的表现，揭示了跨语言迁移的关键瓶颈，并提出针对低资源语言的增强策略。

**[Context Augmented Token-Level Post-Editing for Human Interpreting](multilingual_mt/context_augmented_token-level_post-editing_for_human_interpreting.md)**

:   本文提出一种上下文增强的 token 级后编辑方法，利用对话上下文信息对人工口译的 ASR 转录进行细粒度纠错，在保持口译流畅性的同时显著提升转录质量。

**[COSMMIC: Comment-Sensitive Multimodal Multilingual Indian Corpus](multilingual_mt/cosmmic_commentsensitive_multimodal_multilingual_indian_corpus.md)**

:   构建首个面向印度语言的评论感知多模态多语言数据集COSMMIC——覆盖9种印度语言、4,959篇文章-图像对、24,484条读者评论，提出评论过滤（IndicBERT）和图像分类（CLIP）增强方案，用GPT-4和LLaMA3建立摘要和标题生成基准。

**[Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs](multilingual_mt/cross-lingual_auto_evaluation_for_assessing_multilingual_llms.md)**

:   提出 CIA (Cross Lingual Auto Evaluation) Suite，一个跨语言 LLM 评估框架，包含评估模型 Hercule 和人工标注测试集 Recon，通过利用英语参考答案对非英语语言的 LLM 响应进行评分，8B 模型在多语言评估上超越了 GPT-4o 等闭源大模型。

**[Cross-Lingual Optimization for Language Transfer in Large Language Models](multilingual_mt/cross-lingual_optimization_for_language_transfer_in_large_language_models.md)**

:   提出 Cross-Lingual Optimization (CLO)，通过修改 DPO 损失函数实现跨语言偏好优化——给目标语言输入时偏好目标语言回复、给英语输入时偏好英语回复——在 5 个模型 × 6 种语言上一致超越 SFT，低资源语言中仅 3,200 样本的 CLO 即超越 6,400 样本的 SFT。

**[Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning](multilingual_mt/cross-lingual_representation_alignment_through_contrastive_image-caption_tuning.md)**

:   探索了一种无需平行语料的跨语言表示对齐方法——通过多语言图像-文本描述的对比学习（类 CLIP），让不同语言的文本表示在共享视觉空间中隐式对齐，并证明即使是编码器预训练中未见过的语言（如 Quechua）也能通过这种方式被纳入对齐体系。

**[Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon](multilingual_mt/cross-lingual_transfer_of_cultural_knowledge_an_asymmetric_phenomenon.md)**

:   通过构建可解释的实验框架，研究 LLM 语言适应过程中文化知识的跨语言迁移现象，发现高资源语言（中文、韩语）与英语之间存在双向迁移，而低资源语言（藏语、蒙古语）则呈现不对称迁移——知识主要从低资源语言流向英语，反向流动有限，并提出频率假说加以解释。

**[Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual LLMs: An Extensive Investigation](multilingual_mt/cross-lingual_transfer_of_debiasing_and_detoxification_in_multilingual_llms_an_e.md)**

:   在 7 个 LLM 和 20 种语言上系统研究了英语去偏见/去毒化微调的跨语言迁移效果，发现 SFT 有效去偏见、DPO 有效去毒化，但迁移到非英语语言时普遍伴随语言生成能力下降（语言一致性、流畅度、多样性均受损），迁移效果可由预训练数据中目标语言的数据量预测。

**[Cross-Lingual Generalization and Compression: From Language-Specific to Shared Neurons](multilingual_mt/cross_lingual_neurons_compression.md)**

:   本文通过追踪多语言语言模型预训练过程中的检查点，发现模型从语言特定表示逐渐压缩为跨语言共享表示：中间层的语言识别能力下降、语义概念的"专家神经元"跨语言对齐，操控从西班牙语数据提取的概念神经元后模型反而生成语义相关的英语文本。

**[Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](multilingual_mt/crosslingual_pitfalls.md)**

:   提出一种基于 beam search 和 LLM 模拟的自动化方法，高效生成双语问题对以暴露多语言 LLM 在目标语言上的跨语言性能缺陷，构建了覆盖 16 种语言的 6000+ 样本数据集，揭示即使 GPT-4o 也有超 30% 的跨语言准确率下降。

**[CruxEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution](multilingual_mt/cruxeval-x_a_benchmark_for_multilingual_code_reasoning_understanding_and_executi.md)**

:   提出 CruxEval-X，一个覆盖 19 种编程语言的多语言代码推理基准，通过全自动的测试引导翻译流水线从 Python 版 CruxEval 扩展而来，包含 12,660 个题目和 19K 测试用例，对 24 个 LLM 的评估揭示了编程语言间的相关性以及单语言训练模型的跨语言泛化能力。

**[CulFiT: A Fine-grained Cultural-aware LLM Training Paradigm via Multilingual Critique Data Synthesis](multilingual_mt/culfit_a_fine-grained_cultural-aware_llm_training_paradigm_via_multilingual_crit.md)**

:   CulFiT 提出了一种文化感知的 LLM 训练范式，通过多语言批评数据合成和细粒度奖励建模来增强模型对不同文化的敏感度和包容性，在多个文化理解基准上达到了开源模型的最优性能。

**[Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries](multilingual_mt/dictionaries_to_the_rescue_cross-lingual_vocabulary_transfer_for_low-resource_la.md)**

:   本文提出一种基于双语词典的跨语言词汇迁移方法，利用BPE分词器"删除子词后回退到更短子词"的特性，通过迭代删除-重分词-对齐的过程最大化目标语言子词的映射覆盖率，在低资源语言上显著优于依赖单语语料或平行语料的现有方法。

**[Disentangling Language and Culture for Evaluating Multilingual Large Language Models](multilingual_mt/disentangle_language_culture.md)**

:   提出 Dual Evaluation Framework，将多语言 LLM 评估沿"语言媒介"和"文化语境"两个维度解耦，发现"文化-语言协同"(Cultural-Linguistic Synergy) 现象——模型在文化语境与提问语言对齐时表现更好，并通过 FFN 神经元激活分析从可解释性角度给出解释。

**[Edit Once, Update Everywhere: A Simple Framework for Cross-Lingual Knowledge Synchronization in LLMs](multilingual_mt/edit_once_update_everywhere_a_simple_framework_for_cross-lingual_knowledge_synch.md)**

:   提出 X-KDE 框架，通过跨语言编辑指令微调（XE-IT）+ 目标语言偏好优化（TL-PO）实现"编辑一种语言、所有语言同步更新"的跨语言知识民主化，在 Bi-ZsRE 和 MzsRE 基准上平均提升 +8.19%，跨语言场景下显著超越所有现有方法。

**[EXECUTE: A Multilingual Benchmark for LLM Token Understanding](multilingual_mt/execute_a_multilingual_benchmark_for_llm_token_understanding.md)**

:   扩展字符理解基准 CUTE 到 8 种语言和多种文字系统，提出 EXECUTE 框架，发现 LLM 在不同语言的字符/词/子字符级别表现差异巨大，且意外发现 LLM 对越不熟悉的语言反而在 token 理解任务上表现越好。

**[Exploring In-context Example Generation for Machine Translation](multilingual_mt/exploring_in-context_example_generation_for_machine_translation.md)**

:   提出DAT(Demonstration Augmentation for Translation)——在**无需任何外部资源**的情况下，让LLM自动生成与用户查询相关且多样的源-目标句对作为in-context示例，在5个低资源语言翻译任务上超越zero-shot和固定示例的few-shot基线。

**[Exploring In-Image Machine Translation with Real-World Background](multilingual_mt/exploring_in-image_machine_translation_with_real-world_background.md)**

:   提出 DebackX 模型，通过将图像分离为背景和文字图像分别处理，首次解决了真实复杂背景下的图像内机器翻译 (IIMT) 任务，在翻译质量和视觉效果上均优于现有方法。

**[Language Fusion for Parameter-Efficient Cross-lingual Transfer (FLARE)](multilingual_mt/flare_crosslingual_lora.md)**

:   FLARE 在 LoRA 适配器的低秩瓶颈中通过轻量线性/非线性变换融合源语言（英语）和目标语言的逐层表示，无需额外参数即可实现参数高效的跨语言迁移，在 Llama 3.1 上 QA 精确匹配提升 4.9%。

**[GrammaMT: Improving Machine Translation with Grammar-Informed In-Context Learning](multilingual_mt/grammamt_improving_machine_translation_with_grammar-informed_in-context_learning.md)**

:   提出 GrammaMT，利用语素间注释文本 (Interlinear Glossed Text, IGT) 的语法信息来增强 LLM 的 few-shot 机器翻译，在濒危语言上平均提升 12+ BLEU，在中高资源语言上也有一致改进。

**[Group then Scale: Dynamic Mixture-of-Experts Multilingual Language Model](multilingual_mt/group_then_scale_dynamic_mixture-of-experts_multilingual_language_model.md)**

:   提出 DMoE——基于参数偏差的动态语言分组 + 选择性 MoE 层扩展方法，通过仅 10 步微调量化语言间相似性，将相似语言分组共享同一 expert，只在参数偏差大的层（语言特定层）扩展为 MoE 层，在 18~128 种语言上 PPL 比持续预训练降低 11.4%，用 3.6 倍少的参数超越 X-ELM 9.6%。

**[Hierarchical Level-Wise News Article Clustering via Multilingual Matryoshka Embeddings](multilingual_mt/hierarchical_news_clustering.md)**

:   提出利用多语言Matryoshka嵌入实现层级化新闻聚类的方法：嵌入的不同维度子集对应不同粒度的语义相似性（主题→话题→事件），配合改进的层级凝聚聚类算法，在SemEval 2022 Task 8上达到SOTA（Pearson ρ=0.816）。

**[Implicit Cross-Lingual Rewarding for Efficient Multilingual Preference Alignment](multilingual_mt/implicit_cross-lingual_rewarding_for_efficient_multilingual_preference_alignment.md)**

:   本文提出利用已对齐的英文 DPO 模型中的隐式奖励信号，通过跨语言指令-响应配对标注偏好关系，结合迭代 DPO 训练实现高效的多语言偏好对齐，在 X-AlpacaEval 上平均 Win Rate 提升 12.72%。

**[Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models](multilingual_mt/just_go_parallel_improving_the_multilingual_capabilities_of_large_language_model.md)**

:   系统研究在 decoder-only LLM 训练中加入平行数据对多语言能力的影响：平行数据放在训练末期效果最好且显著优于等量单语数据；LLM 无法自动泛化到训练方向的反向翻译（reversal curse）。

**[KnowCoder-X: Boosting Multilingual Information Extraction via Code](multilingual_mt/knowcoder-x_boosting_multilingual_information_extraction_via_code.md)**

:   提出 KnowCoder-X，通过统一的 Python 类表示多语言 IE schema，并引入 IE 跨语言对齐指令微调阶段（含高质量 ParallelNER 数据集），在 64 个 IE 基准上大幅提升跨语言信息抽取性能。

**[LACA: Improving Cross-lingual Aspect-Based Sentiment Analysis with LLM Data Augmentation](multilingual_mt/laca_crosslingual_absa.md)**

:   提出 LACA 框架，利用 LLM 为目标语言生成高质量伪标注数据（而非依赖机器翻译），在六种语言上显著提升跨语言 ABSA 性能，在 mBERT 和 XLM-R 上分别平均超过前 SOTA 1.50% 和 2.62%。

**[LangMark: A Multilingual Dataset for Automatic Post-Editing](multilingual_mt/langmark_a_multilingual_dataset_for_automatic_post-editing.md)**

:   发布 LangMark——一个包含 206,983 个三元组、覆盖英语到七种语言的大规模多语言自动后编辑（APE）数据集，并证明 LLM 配合 few-shot prompting 能有效改善专有 NMT 引擎的输出质量。

**[LangSAMP: Language-Script Aware Multilingual Pretraining](multilingual_mt/langsamp_multilingual_pretraining.md)**

:   提出 LangSAMP 方法，在多语言预训练中将语言和文字系统 (script) embedding 添加到 Transformer 输出端（而非输入端），使模型主干学到更语言中立的表示，在 500+ 语言的零样本跨语言迁移中一致优于基线。

**[LEMONADE: A Large Multilingual Expert-Annotated Abstractive Event Dataset for the Real World](multilingual_mt/lemonade_a_large_multilingual_expert-annotated_abstractive_event_dataset_for_the.md)**

:   发布 Lemonade——基于 ACLED 冲突数据的大规模多语言专家标注事件数据集（39,786 事件，20 种语言，171 个国家，10,707 实体），提出 Abstractive Event Extraction (AEE) 新任务范式，事件参数不限于文本 span 而是归一化为数值/类别/实体，配套 Zest 零样本实体链接系统在 AEL 子任务上 F1=45.7% 大幅超越 baseline 的 23.7%。

**[Less, but Better: Efficient Multilingual Expansion for LLMs via Layer-wise Mixture-of-Experts](multilingual_mt/less_but_better_efficient_multilingual_expansion.md)**

:   分析 LLM 不同层间的跨语言表征相似度，提出 LayerMoE 按层分配不同数量的新语言专家（高相似层少分配、低相似层多分配），用 60% 更少的专家参数超越 SOTA，并通过在高相似层添加路由分类器进一步缓解灾难性遗忘。

**[LexGen: Domain-aware Multilingual Lexicon Generation](multilingual_mt/lexgen_domain-aware_multilingual_lexicon_generation.md)**

:   本文提出LexGen框架，通过在预训练多语言翻译模型的decoder中引入可学习的"领域路由"（Domain Routing）层，实现领域特定和领域通用知识的动态融合，在6种印度语言、8个领域的词典生成任务上超越了NLLB和BLICEr等基线。

**[LLMs Can Achieve High-quality Simultaneous Machine Translation as Efficiently as Offline](multilingual_mt/llms_can_achieve_high-quality_simultaneous_machine_translation_as_efficiently_as.md)**

:   本文提出了一种新范式，通过将源语言和目标语言 token 按延迟要求重排为交错序列来构造 SFT 数据，使 LLM 能够像离线翻译一样高效地完成高质量同声传译（SiMT），在多个基准上达到 SOTA 性能，同时保持离线翻译的原有能力。

**[Lost in Multilinguality: Dissecting Cross-lingual Factual Inconsistency in Transformer Language Models](multilingual_mt/lost_in_multilinguality_dissecting_cross-lingual_factual_inconsistency_in_transf.md)**

:   用机制可解释性方法解剖多语言 LLM 的跨语言事实不一致问题，发现模型在大多数层中以语言无关的概念空间处理知识，但在最后几层的"语言转换"过程中失败导致不一致，提出线性快捷方法绕过最后层以提升一致性和准确率。

**[Read it in Two Steps: Translating Extremely Low-Resource Languages with Code-Augmented Grammar Books](multilingual_mt/low_resource_translation.md)**

:   将语法书辅助的极低资源翻译分解为**语法规则检索**和**规则应用**两步，提出 Rule-by-Rule 检索策略和代码格式语法规则表示，在壮语翻译上端到端提升 13.1% BLEU。

**[M-MAD: Multidimensional Multi-Agent Debate for Advanced Machine Translation Evaluation](multilingual_mt/m-mad_multidimensional_multi-agent_debate_for_advanced_machine_translation_evalu.md)**

:   提出 M-MAD 框架，将 MQM 评估标准解耦为独立维度（准确性、流畅性、风格、术语），在每个维度内进行多智能体正反方辩论，最后由裁判智能体综合各维度结果，在 segment 级别显著超越已有 LLM-as-a-judge 方法，甚至用 GPT-4o mini 就能媲美 SOTA 有参考自动指标。

**[M2rc-Eval: Massively Multilingual Repository-level Code Completion Evaluation](multilingual_mt/m2rc-eval_massively_multilingual_repository-level_code_completion_evaluation.md)**

:   提出覆盖18种编程语言的大规模多语言仓库级代码补全基准 M2rc-Eval，配合基于 AST 的桶级和语义级细粒度标注，并构建 M2rc-Instruct 指令语料以提升模型性能。

**[M3FinMeeting: A Multilingual, Multi-Sector, and Multi-Task Financial Meeting Understanding Evaluation Dataset](multilingual_mt/m3finmeeting_a_multilingual_multi-sector_and_multi-task_financial_meeting_unders.md)**

:   构建了 M3FinMeeting——首个面向金融会议的多语言（中英日）、多行业、多任务评测基准，包含 600 场真实金融会议的摘要、QA 对抽取和问答三项任务，揭示了当前最先进 LLM 在金融会议理解上仍有显著提升空间。

**[M-RewardBench: Evaluating Reward Models in Multilingual Settings](multilingual_mt/m_rewardbench.md)**

:   构建首个多语言奖励模型评估基准 M-RewardBench（23种 typologically 多样语言、2.87K 偏好实例，覆盖 Chat/Safety/Reasoning/Translation 四类能力），系统评估多种 RM 后发现英语与非英语性能存在显著差距，且 RM 偏好可在语言间发生实质性漂移。

**[Machine Translation Models are Zero-Shot Detectors of Translation Direction](multilingual_mt/machine_translation_models_are_zero-shot_detectors_of_translation_direction.md)**

:   提出一种基于 NMT 模型翻译概率的无监督翻译方向检测方法：若 $p(\text{translation}|\text{original}) > p(\text{original}|\text{translation})$，则可零样本判断平行文本的原始翻译方向，NMT 翻译的文档级检测准确率达 96%。

**[Marco-Bench-MIF: On Multilingual Instruction-Following Capability of Large Language Models](multilingual_mt/marco_bench_multilingual_if.md)**

:   将英文IFEval基准扩展到30种语言并进行文化本地化，揭示LLM在多语言指令遵循中高/低资源语言间25-35%的准确率差距，以及机器翻译数据低估模型性能7-22%。

**[MaXIFE: Multilingual and Cross-lingual Instruction Following Evaluation](multilingual_mt/maxife_multilingual_and_cross-lingual_instruction_following_evaluation.md)**

:   提出 MaXIFE 评测基准，覆盖 23 种语言的 1667 个可验证指令跟随任务，结合基于规则和基于模型的双重评估策略，系统评测 LLM 在多语言和跨语言场景下的指令遵循能力，填补了该方向的评测空白。

**[Memorization Inheritance in Sequence-Level Knowledge Distillation for Neural Machine Translation](multilingual_mt/memorization_inheritance_seqkd.md)**

:   本文首次系统研究了序列级知识蒸馏（SeqKD）中教师模型的记忆行为如何传递给学生模型，发现学生模型虽未直接接触原始训练数据，但其提取式记忆率比基线模型高 57%，幻觉率也增加，并提出 Adaptive-SeqKD 通过在高质量子集上微调教师来缓解这些问题。

**[Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](multilingual_mt/mid_layer_crosslingual_alignment.md)**

:   通过大规模分析 1000+ 语言对（35 种语言、1190 个方向）发现 LLM **中间层**具有最强跨语言语义对齐潜力，提出在任务微调中交替优化中间层对比对齐损失，在槽填充（F1 +1.5）、机器翻译（COMET +1.1）和 JSON 生成三大任务上显著提升跨语言迁移，且对未见语言和不同域数据均有效；分别训练的对齐与任务 LoRA 模块可通过权重平均合并使用。

**[MiLiC-Eval: Benchmarking Multilingual LLMs for China's Minority Languages](multilingual_mt/milic-eval_benchmarking_multilingual_llms_for_chinas_minority_languages.md)**

:   构建了首个面向中国少数民族语言（藏语、维吾尔语、哈萨克语、蒙古语）的标准化LLM评估基准MiLiC-Eval，包含9类任务2.4万实例，揭示了当前LLM在非主流书写系统上的严重不足。

**[Modular Sentence Encoders: Separating Language Specialization from Cross-Lingual Alignment](multilingual_mt/modular_sentence_encoders.md)**

:   本文提出模块化多语言句子编码器训练方案：先训练语言特定模块（embedding + 语言适配器 + 句子编码适配器）缓解多语言诅咒，再训练跨语言对齐适配器同时使用平行和释义数据解决不同跨语言任务间的性能权衡，在 4 个任务和 23 种语言上全面优于单体模型训练。

**[mOSCAR: A Large-scale Multilingual and Multimodal Document-level Corpus](multilingual_mt/moscar_a_large-scale_multilingual_and_multimodal_document-level_corpus.md)**

:   提出 mOSCAR——首个大规模多语言多模态文档级语料库（163种语言、303M文档、200B tokens、1.15B图片），从 Common Crawl 中提取交错的图文文档，并证明在此数据上训练的多语言 mLLM 能获得显著的 few-shot 学习提升。

**[Delving into Multilingual Ethical Bias: The MSQAD with Statistical Hypothesis Tests for Large Language Models](multilingual_mt/msqad_multilingual_ethical_bias.md)**

:   提出多语言敏感问答数据集MSQAD（基于Human Rights Watch 17个人权话题、6种语言），通过McNemar检验和PERMANOVA检验两种统计假设检验，系统证明LLM在不同语言下回答相同敏感问题时存在显著伦理偏差：中文/印地语拒绝率最高而西/德语最易生成不当回答，且该偏差在7个LLM中普遍存在。

**[Has Machine Translation Evaluation Achieved Human Parity?](multilingual_mt/mt_eval_human_parity.md)**

:   首次将人类基线引入 WMT Metrics Shared Task 的排名，发现最先进的自动指标经常与人类评估者排名持平甚至更高，但论证了现在声称"人类对等"为时尚早，并讨论了衡量 MT 评估进步的根本困难。

**[MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering](multilingual_mt/mtvqa_benchmarking_multilingual_text-centric_visual_question_answering.md)**

:   构建了 MTVQA——首个覆盖 9 种语言的多语言文本中心视觉问答基准，通过人类专家标注解决翻译方法的"视觉-文本不对齐"问题，评估显示最佳 MLLM（InternVL-2.5，32.2%）与人类表现（79.7%）差距巨大，揭示了多语言文本理解的严峻挑战。

**[Multi-perspective Alignment for Increasing Naturalness in Neural Machine Translation](multilingual_mt/multi-perspective_alignment_for_increasing_naturalness_in_neural_machine_transla.md)**

:   提出多视角对齐框架 (Multi-perspective Alignment)，同时奖励翻译自然度和内容保留，通过翻译体分类器和 COMET 的联合奖励信号对 NMT 模型进行强化学习微调，使译文词汇更丰富且不损失翻译准确度。

**[Multilingual Encoder Knows More Than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages](multilingual_mt/multilingual_encoder_knows_more_than_you_realize_shared_weights_pretraining_for_.md)**

:   提出编码器-解码器权重共享框架，通过交替复用编码器权重层和随机初始化层构建解码器，将多语言编码器CINO高效扩展为seq2seq模型XLM-SWCM，在藏维哈蒙四种极低资源语言上以不到0.5B参数大幅超越mBART和13B LLaMA。

**[Do Large Language Models Have an English Accent? Evaluating and Improving the Naturalness of Multilingual LLMs](multilingual_mt/multilingual_llm_english_accent.md)**

:   本文揭示多语言 LLM 在非英语语言生成中存在"英语口音"——词汇和句法上偏向英语模式，提出了基于 JSD（词汇分布）和 WL 图核+MMD（句法依赖树）的语料级自然度指标，并通过 DPO 对齐方法有效提升目标语言的自然度。

**[Data Quality Issues in Multilingual Speech Datasets: The Need for Sociolinguistic Awareness and Proactive Language Planning](multilingual_mt/multilingual_speech_data_quality.md)**

:   对三大公开多语言语音数据集（Common Voice 17.0、FLEURS、VoxPopuli）进行覆盖 40+ 种语言的系统质量审计，将问题分为可程序化修复的"微观问题"和需语言学介入的"宏观问题"，发现低制度化语言面临的宏观问题尤为严重，并提出融入社会语言学意识的 5 步数据集创建指南。

**[NameTag 3: A Tool and a Service for Multilingual/Multitagset NER](multilingual_mt/nametag_3_a_tool_and_a_service_for_multilingualmultitagset_ner.md)**

:   本文介绍 NameTag 3，一个开源的多语言、多数据集、多标签集命名实体识别工具和云服务，基于微调的预训练语言模型，单个 355M 参数模型在 15 种语言的 21 个测试集上达到 SOTA，同时比 DeepSeek-R1 等 LLM 快 10,000 倍以上。

**[Probing LLMs for Multilingual Discourse Generalization Through a Unified Label Set](multilingual_mt/probing_llms_for_multilingual_discourse_generalization_through_a_unified_label_s.md)**

:   本文提出首个跨框架、跨语言的统一篇章关系标签集（17类），并通过对23个LLM的注意力探针实验，证明多语言LLM能够在中间层编码跨语言可迁移的篇章级表征，且多语言训练和模型规模共同提升泛化能力。

**[Registering Source Tokens to Target Language Spaces in Multilingual Neural Machine Translation](multilingual_mt/registering_source_tokens_to_target_language_spaces_in_multilingual_neural_machi.md)**

:   提出 Registering 方法：在源语言和目标语言 token 之间插入一组目标语言标记（registers），通过修改注意力掩码使目标生成仅依赖 registers 的激活，彻底解决多语言翻译中的 off-target 问题，使小模型 MITRE-913M 超越 NLLB-3.3B。

**[Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-Lingual Transfer](multilingual_mt/semantic_aware_linear_transfer_by_recycling_pre-trained_language_models_for_cros.md)**

:   提出 SALT（Semantic Aware Linear Transfer），通过为每个非共享词表 token 基于语义相似的共享 token 对构建独立的最小二乘变换矩阵，将目标语言 PLM 的丰富嵌入表示迁移到英语中心 LLM 的嵌入空间，在下游任务、持续预训练收敛速度和跨语言理解上均优于现有方法。

**[SeqPO-SiMT: Sequential Policy Optimization for Simultaneous Machine Translation](multilingual_mt/seqpo-simt_sequential_policy_optimization_for_simultaneous_machine_translation.md)**

:   将同步机器翻译（SiMT）建模为多步序列决策问题，提出 SeqPO-SiMT 策略优化框架，融合翻译质量和延迟的奖励信号，在 7B LLM 上实现 SiMT 性能媲美离线翻译的强模型。

**[ShifCon: Enhancing Non-Dominant Language Capabilities with a Shift-based Multilingual Contrastive Framework](multilingual_mt/shifcon_nondominant_language.md)**

:   提出 ShifCon 框架，通过将非优势语言的表示 shift 到优势语言子空间以获取更丰富的模型知识，再 shift 回原语言子空间进行生成，结合多语言对比学习，显著提升低资源语言的表现。

**[SIFT-50M: A Large-Scale Multilingual Dataset for Speech Instruction Fine-Tuning](multilingual_mt/sift-50m_a_large-scale_multilingual_dataset_for_speech_instruction_fine-tuning.md)**

:   本文构建了 SIFT-50M，一个包含 5000 万条样本、覆盖 5 种语言的语音指令微调数据集，利用 LLM 和专家模型从公开语音语料中自动生成多样化的语音理解与可控语音生成指令，并训练出 SIFT-LLM 在指令跟随基准上超越现有语音文本 LLM。

**[Statement-Tuning Enables Efficient Cross-lingual Generalization in Encoder-only Models](multilingual_mt/statement-tuning_enables_efficient_cross-lingual_generalization_in_encoder-only_.md)**

:   将 Statement-Tuning 方法扩展到多语言场景，证明仅 276M 参数的 mDeBERTa 编码器模型通过多语言 Statement-Tuning 微调后，能在未见任务和未见语言上实现跨语言零样本泛化，在多个 NLU 任务上匹敌甚至超越 70B+ 参数的生成式 LLM。

**[Team ACK at SemEval-2025 Task 2: Beyond Word-for-Word Machine Translation for English-Korean Pairs](multilingual_mt/team_ack_at_semeval-2025_task_2_beyond_word-for-word_machine_translation_for_eng.md)**

:   本文在 SemEval-2025 Task 2 中系统评估了 13 个模型（LLM + 传统 MT）在英韩实体密集文本翻译上的表现，通过自动指标和双语人工评估揭示了 LLM 虽优于传统 MT 但在需要文化适应的实体翻译上仍普遍失败，并构建了翻译错误分类体系。

**[The Esethu Framework: Reimagining Sustainable Dataset Governance and Curation for Low-Resource Languages](multilingual_mt/the_esethu_framework_reimagining_sustainable_dataset_governance_and_curation_for.md)**

:   提出 Esethu 框架——一种社区驱动的可持续数据治理方案，通过创新的社区中心化许可证实现数据收益的循环再投资，并以 isiXhosa 语音数据集 ViXSD 作为概念验证。

**[The Hidden Space of Safety: Understanding Preference-Tuned LLMs in Multilingual Contexts](multilingual_mt/the_hidden_space_of_safety_understanding_preference-tuned_llms_in_multilingual_c.md)**

:   本文系统分析了偏好调优（RLHF/DPO 等）对 LLM 内部表示空间在多语言场景下的影响，发现对齐机制在英语上能有效分离有害/无害内容的隐空间表示，但在印地语、中文、德语等非英语语言上效果显著退化，揭示了当前对齐方法存在严重的单语偏差问题。

**[THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation](multilingual_mt/thor-moe_hierarchical_task-guided_and_context-responsive_routing_for_neural_mach.md)**

:   提出THOR-MoE框架，通过层级任务引导路由（自动预测领域/语言并生成混合任务表示来选任务级专家子集）和上下文响应路由（将全局上下文注入token表示以辅助专家选择），在多领域和多语言翻译中以更少激活参数获得显著性能提升。

**[Towards Global AI Inclusivity: A Large-Scale Multilingual Terminology Dataset (GIST)](multilingual_mt/towards_global_ai_inclusivity_a_large-scale_multilingual_terminology_dataset_gis.md)**

:   构建首个大规模多语言 AI 术语数据集 GIST（约 5K 术语、5 种语言），采用 LLM 抽取 + 人工众包翻译 + LLM 选择的混合框架，并通过 prompting 后翻译优化方法在 BLEU/COMET 等指标上一致提升机器翻译中 AI 术语的翻译质量。

**[Trans-Zero: Self-Play Incentivizes Large Language Models for Multilingual Translation](multilingual_mt/trans-zero_self-play_incentivizes_large_language_models_for_multilingual_transla.md)**

:   提出 Trans-Zero 自博弈框架，仅使用单语数据，通过遗传蒙特卡洛树搜索（G-MCTS）在多语言翻译过程中探索语义一致的候选翻译，结合偏好优化实现无平行数据的多语言翻译训练，性能可媲美大规模监督微调方法。

**[Translation and Fusion Improves Zero-shot Cross-lingual Information Extraction](multilingual_mt/translation_and_fusion_improves_cross-lingual_information_extraction.md)**

:   提出 TransFusion 框架，通过在推理时先将低资源语言文本翻译为英语、在英语上进行信息抽取标注、再用融合模型将英语标注与原文结合来生成最终预测，在50种语言的零样本跨语言IE任务上显著优于基线（MasakhaNER2平均F1从47.9提升到62.4）。

**[Did Translation Models Get More Robust Without Anyone Even Noticing?](multilingual_mt/translation_robustness.md)**

:   通过合成噪声和社交媒体文本实验发现，近年大规模预训练翻译模型（如 TowerInstruct 13B、GPT-3.5）在未使用任何专门鲁棒性训练技术的情况下，对多种字符级噪声的鲁棒性远超传统 NMT 模型（OPUS），且源端纠错+LLM 翻译的组合可进一步超越 GPT-3.5。

**[Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu](multilingual_mt/understanding_in-context_machine_translation_for_low-resource_languages_a_case_s.md)**

:   系统研究了LLM上下文机器翻译中各类语言资源（词典、平行语料、语法书、CoT提示）对翻译质量的影响，以满语为案例发现高质量词典和检索的平行例句最有价值而语法书几乎无用，并通过字符加密实验证明LLM主要依赖上下文学习能力而非先验知识，最后展示了利用上下文翻译生成合成平行数据来训练传统NMT模型的有效性。

**[Unveiling the Power of Source: Source-based Minimum Bayes Risk Decoding for Neural Machine Translation](multilingual_mt/unveiling_the_power_of_source_source-based_minimum_bayes_risk_decoding_for_neura.md)**

:   提出 source-based MBR (sMBR) 解码方法，利用释义/回译生成的准源端句子作为"支持假设"，结合无参考 QE 指标作为效用函数，首次在 MBR 解码中完全依赖源端信息，在经典和 LLM 两种 NMT 设置下均优于 QE reranking 和标准 MBR 解码。

**[Watching the Watchers: Exposing Gender Disparities in Machine Translation Quality Estimation](multilingual_mt/watching_the_watchers_exposing_gender_disparities_in_machine_translation_quality.md)**

:   > 本文系统揭示了机器翻译质量评估 (QE) 指标中的性别偏差：在源语言性别模糊时阳性形式得分高于阴性形式，在有上下文线索时阴性形式的错误率更高，且偏差会通过数据过滤和质量感知解码传播到下游 MT 系统。

**[X-WebAgentBench: A Multilingual Interactive Web Benchmark for Evaluating Global Agentic System](multilingual_mt/x-webagentbench_a_multilingual_interactive_web_benchmark_for_evaluating_global_a.md)**

:   提出 X-WebAgentBench——一个多语言交互式网页基准，用于评估语言智能体在多种语言下的规划和交互能力，并评测了多种 LLM 及跨语言对齐方法，发现即使 GPT-4o 结合跨语言技术也无法达到令人满意的结果。

**[ZIPA: A Family of Efficient Models for Multilingual Phone Recognition](multilingual_mt/zipa_a_family_of_efficient_models_for_multilingual_phone_recognition.md)**

:   提出 Zipa 系列高效语音模型，基于 Zipformer 骨干和 IpaPack++（17,132 小时多语言标注数据），在多语言音素识别上达到 SOTA，64M 参数模型即超越现有 300M 模型，并通过噪声学生训练在 4000+ 种语言上进一步提升。

---

## 📊 LLM 评测 { #llm_evaluation }

**[A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates](llm_evaluation/a_conformal_risk_control_framework_for_granular_word_assessment_and_uncertainty_.md)**

:   提出基于 conformal risk control 框架对 CLIPScore 进行细粒度词级错误检测和不确定性校准，通过简单的注意力掩码采样生成分数分布，在保持模型无关性的同时提供形式化的风险控制保证。

**[MisMatched: A Benchmark for Scientific Natural Language Inference](llm_evaluation/a_mismatched_benchmark_for_scientific_natural_language_inference.md)**

:   引入 MisMatched——首个覆盖非 CS 领域（心理学、工程、公共卫生）的科学 NLI 评估基准，包含 2,700 对人工标注句子对，最佳 SLM 基线（SciBERT）Macro F1 仅 78.17%，最佳 LLM 基线（Phi-3）仅 57.16%，并证明训练时加入隐式关系句子对可提升模型性能。

**[AbGen: Evaluating Large Language Models in Ablation Study Design and Evaluation for Scientific Research](llm_evaluation/abgen_evaluating_large_language_models_in.md)**

:   提出 AbGen——首个评估 LLM 设计消融实验能力的基准（1500 条专家标注数据来自 807 篇 NLP 论文），发现最强 LLM (DeepSeek-R1) 与人类专家差距 14.4%，且 LLM-as-Judge 评分与人类评估严重不一致。

**[Access Denied Inc: The First Benchmark Environment for Sensitivity Awareness](llm_evaluation/access_denied_inc_the_first_benchmark_environment_for_sensitivity_awareness.md)**

:   首次形式化定义 LLM "敏感性感知"（Sensitivity Awareness）概念——评估 LLM 能否根据基于角色的访问控制（RBAC）规则决定信息是否可以提供——并构建自动化评估基准 Access Denied Inc，在 7 个主流 LLM 上发现即使数据高度结构化且规则极简，最佳模型 Grok-2 仍有 18.28% 的泄露率。

**[Ad-hoc Concept Forming in the Game Codenames as a Means for Evaluating Large Language Models](llm_evaluation/ad-hoc_concept_forming_in_the_game_codenames_as_a_means_for_evaluating_large_lan.md)**

:   将桌游Codenames实现为LLM评测基准，LLM同时扮演线索给出者(Spymaster)和猜测者(Field Operative)，在13种不同难度实验中与确定性对手对战，14个模型中最佳(o3-mini)胜率仅49%，揭示了LLM在词汇关联、策略选择和纠错能力上的显著局限。

**[AD-LLM: Benchmarking Large Language Models for Anomaly Detection](llm_evaluation/ad-llm_benchmarking_large_language_models_for_anomaly_detection.md)**

:   提出首个LLM异常检测基准AD-LLM，系统评估LLM在零样本检测、数据增强和无监督模型选择三个核心任务中的能力，发现GPT-4o零样本检测在多数数据集上超越传统训练方法，合成数据对灵活表示的检测器有效但对几何假设模型有害，推理型LLM模型选择接近最优但解释缺乏数据集针对性。

**[AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents](llm_evaluation/androidlab_autonomous_agent.md)**

:   提出AndroidLab——一个系统性的Android智能体评测与训练框架，包含统一的操作环境、138个任务的可复现基准测试和94.3K步骤的指令数据集，通过微调将开源LLM成功率从4.59%提升至21.50%。

**[AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge](llm_evaluation/antileakbench_preventing_data_contamination_by_automatically_constructing_benchm.md)**

:   提出 AntiLeakBench 自动化反泄露基准框架，通过追踪 Wikidata 知识更新历史识别 LLM 截止时间后的新知识，自动构建单跳/多跳 QA 测试样本（附真实 Wikipedia 支撑文档），确保知识级严格无污染，12 个 LLM 的大规模实验证实截止后性能普遍下降（EM 跌幅显著）验证了框架有效性。

**[Atomic Calibration of LLMs in Long-Form Generations](llm_evaluation/atomic_calibration_of_llms_in_long-form_generations.md)**

:   系统研究长文本生成中的原子级校准(atomic calibration)，将置信度获取方法分为判别式和生成式两类，发现两者互补且提出基于置信度一致性的融合策略，揭示了模型在生成过程中置信度变化的有趣模式。

**[Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](llm_evaluation/batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)**

:   提出 Batayan——首个全面的菲律宾语 LLM 评测基准，覆盖理解/推理/生成三大能力的 8 个任务（含 3 个全新菲律宾语任务），由母语者翻译和标注确保语言真实性，评测 50+ 开源和商用 LLM 后发现菲律宾语表现显著落后于英语，显式菲律宾语支持和模型规模的提升均能带来明显增益。

**[BelarusianGLUE: Towards a Natural Language Understanding Benchmark for Belarusian](llm_evaluation/belarusian_glue.md)**

:   为白俄罗斯语（Belarusian，东斯拉夫语族）构建了首个NLU benchmark——BelarusianGLUE，包含5个任务约15K条实例，系统评估了BERT系列和LLM的表现，发现简单任务（情感分析）接近人类水平但难任务（Winograd）仍有显著差距，且最优模型类型因任务而异。

**[Benchmarking LLMs and LLM-based Agents in Practical Vulnerability Detection for Code Repositories](llm_evaluation/benchmarking_llms_and_llm-based_agents_in_practical_vulnerability_detection_for_.md)**

:   提出 JitVul 基准，将每个函数与其引入漏洞和修复漏洞的commit关联，基于879个CVE覆盖91种漏洞类型，系统评估了LLM和ReAct Agent在仓库级别漏洞检测中的能力，发现ReAct Agent优于纯LLM但两者均有较大改进空间。

**[Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph](llm_evaluation/benchmarking_uncertainty_quantification_methods_for_large_language_models_with_l.md)**

:   构建了 LM-Polygraph 不确定性量化（UQ）基准，实现了30+种SOTA方法，在11个文本生成任务上系统评估了UQ和置信度归一化技术的效果，为LLM幻觉检测提供了统一的评测框架。

**[BESSTIE: A Benchmark for Sentiment and Sarcasm Classification for Varieties of English](llm_evaluation/besstie_a_benchmark_for_sentiment_and_sarcasm_classification_for_varieties_of_en.md)**

:   构建 BESSTIE，首个针对英语变体（澳大利亚/印度/英国英语）的情感分析和讽刺检测标注基准，通过 9 个微调 LLM 评估发现模型在印度英语（外圈变体）上表现显著差于内圈变体，跨变体泛化能力也有限。

**[Beyond One-Size-Fits-All: Tailored Benchmarks for Efficient Evaluation](llm_evaluation/beyond_one-size-fits-all_tailored_benchmarks_for_efficient_evaluation.md)**

:   提出 TailoredBench 方法，为每个待评估的目标模型**自适应构建定制化核心集**（Native-coreset），而非使用所有模型共享的静态子集，通过自适应源模型选择、可扩展 K-Medoids 聚类和校准估计策略，在仅需 20-40 个样本的推理预算下将准确率估计的 MAE 平均降低 **31.4%**。

**[Browsing Lost Unformed Recollections: A Benchmark for Tip-of-the-Tongue Search and Reasoning](llm_evaluation/browsing_lost_unformed_recollections_a_benchmark_for_tip-of-the-tongue_search_an.md)**

:   > 提出 BLUR（Browsing Lost Unformed Recollections），一个包含 573 道真实"话到嘴边"(tip-of-the-tongue) 已知物品搜索与推理问题的基准数据集，人类准确率 98%，而最佳 AI 系统仅约 56%，揭示了当前 AI 在工具使用和多跳推理上的巨大差距。

**[CalibraEval: Calibrating Prediction Distribution to Mitigate Selection Bias in LLMs-as-Judges](llm_evaluation/calibraeval_calibrating_prediction_distribution_to_mitigate_selection_bias_in_ll.md)**

:   提出 CalibraEval，一种无标签的推理时去偏方法，通过将去偏问题形式化为优化任务，利用非参数保序算法（NOA）学习校准函数，将 LLM 评判器的观测概率分布映射到无偏分布，有效缓解 LLM-as-Judge 中的选择偏差。

**[Improving the Calibration of Confidence Scores in Text Generation Using the Output Distribution's Characteristics](llm_evaluation/calibration_confidence_text_gen.md)**

:   针对文本生成中多个有效输出导致传统置信度指标失效的问题，提出两种任务无关的置信度度量——"比率"（头部vs中部概率比）和"尾部稀薄度"（分布尾部薄厚），仅依赖模型输出概率即可改善 BART/Flan-T5 在摘要、翻译、问答任务上的置信度校准。

**[Can External Validation Tools Improve Annotation Quality for LLM-as-a-Judge?](llm_evaluation/can_external_validation_tools_improve_annotation_quality_for_llm-as-a-judge.md)**

:   提出 Evaluation Agent，一个工具增强的 LLM-as-a-Judge 框架，通过集成网络搜索（事实核查）、代码执行和数学验证工具，在长文本事实验证上将与人类一致性从 63% 提升到 81%，在编程评估上从 31% 提升到 71%，且对无关领域几乎无退化。

**[CFBench: A Comprehensive Constraints-Following Benchmark for LLMs](llm_evaluation/cfbench_a_comprehensive_constraints-following_benchmark_for_llms.md)**

:   提出 CFBench——一个包含 1000 条精标样本、覆盖 200+ 真实场景和 50+ NLP 任务的中文大规模约束遵循基准，系统性地定义了 10 大类 25+ 子类的约束分类体系，并设计结合约束满足率（CSR）、指令满足率（ISR）和需求优先级满足率（PSR）的多维评估框架，揭示当前顶级 LLM 在约束遵循方面仍存在显著提升空间。

**[ChatBench: From Static Benchmarks to Human-AI Evaluation](llm_evaluation/chatbench_from_static_benchmarks_to_human-ai_evaluation.md)**

:   通过用户实验将 MMLU 静态基准转换为用户-AI 对话，构建 ChatBench 数据集（396 道题、7336 段对话），发现 AI-alone 准确率无法预测 user-AI 准确率，并训练用户模拟器使相关性提升 22-26 个百分点，为可扩展的交互式评估奠基。

**[CodeMEnv: Benchmarking Large Language Models on Code Migration](llm_evaluation/codemenv_benchmarking_large_language_models_on_code_migration.md)**

:   提出 CodeMEnv，首个系统评估 LLM 跨环境代码迁移能力的基准，包含 922 个样本、19 个 Python/Java 包、3 个层次化任务（定位不兼容函数→描述变更→迁移代码），9 个 LLM 的平均 Pass@1 仅 26.50%，GPT-4o 最高 43.84%，揭示 LLM 更熟悉新版本函数且存在版本推理逻辑不一致问题。

**[Com2: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models](llm_evaluation/com2_causal_commonsense.md)**

:   提出 Com2，一个基于因果事件图和因果理论（干预/反事实）构建的复杂常识推理基准，包含 2500 道主题和 1254 道侦探故事题目，揭示 LLM 在推理深度与广度上的显著不足。

**[CoPrUS: Consistency Preserving Utterance Synthesis Towards More Realistic Benchmark](llm_evaluation/coprus_consistency_preserving_utterance_synthesis_towards_more_realistic_benchma.md)**

:   本文提出 CoPrUS 框架，一种面向对话基准构建的一致性保持话语合成方法，通过在生成对话数据时显式维护人设、知识和对话历史的一致性约束，生成比现有方法更真实的对话基准数据。

**[CoV-Eval: Can You Really Trust Code Copilots? Evaluating Large Language Models from a Code Security Perspective](llm_evaluation/cov_eval_evaluating_llms_from_code_security_perspective.md)**

:   提出 CoV-Eval，首个多任务代码漏洞评估基准，涵盖代码补全、漏洞修复、漏洞检测和漏洞分类四个维度，并开发 VC-Judge 漏洞判断模型替代传统静态分析工具，对 20 个 LLM 进行全面评估，发现虽然多数 LLM 能检测漏洞代码，但仍倾向生成不安全代码且漏洞修复能力有限。

**[CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding](llm_evaluation/culemo_cultural_lenses_on_emotion_-_benchmarking_llms_for_cross-cultural_emotion.md)**

:   提出 CuLEmo，首个评估文化感知情感预测的多语言基准数据集，涵盖 6 种语言/文化（阿姆哈拉语、阿拉伯语、英语、德语、印地语、西班牙语），通过 400 个文化相关场景评估 LLM 的跨文化情感理解能力，发现情感表达在不同文化间存在显著差异且 LLM 表现参差不齐。

**[CulturalBench: A Robust, Diverse, and Challenging Cultural Benchmark by Human-AI CulturalTeaming](llm_evaluation/culturalbench_a_robust_diverse_and_challenging_cultural_benchmark_by_human-ai_cu.md)**

:   通过 Human-AI CulturalTeaming（人机协作红队测试）流水线构建 CulturalBench，包含 1,696 个人类撰写并经五人独立验证的文化知识问题，覆盖 45 个全球地区和 17 个主题。CulturalBench-Hard（True/False格式）对最强模型（OpenAI o1）也仅 61.5%，远低于人类的 92.4%，揭示了模型在多答案问题上的模式寻求倾向和跨区域文化知识的不均衡表现。

**[EcomScriptBench: A Multi-task Benchmark for E-commerce Script Planning via Step-wise Intention-Driven Product Association](llm_evaluation/ecomscriptbench.md)**

:   定义电商脚本规划（EcomScript）任务并构建首个大规模基准 EcomScriptBench（60 万脚本 + 240 万产品），通过购买意图桥接动作步骤与产品搜索的语义鸿沟，揭示当前 LLM 在该任务上的显著不足。

**[EditInspector: A Benchmark for Evaluation of Text-Guided Image Edits](llm_evaluation/editinspector_a_benchmark_for_evaluation_of_text-guided_image_edits.md)**

:   本文提出 EditInspector，一个基于人工标注的多维度文本引导图像编辑评估基准，覆盖编辑准确性、伪影检测、视觉质量、场景融合度、常识一致性和变化描述六个维度，揭示了当前 VLM 在全面评估编辑质量方面的不足，并提出在伪影检测和差异描述生成上超越 SOTA 的两种新方法。

**[EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework](llm_evaluation/educationq_evaluating_llms_teaching_capabilities_through_multi-agent_dialogue_fr.md)**

:   提出 EducationQ 多智能体对话框架，通过模拟真实课堂中教师-学生的形成性评估交互来评估 LLM 的教学能力，发现教学效果与模型规模或通用推理能力不呈线性关系，Llama 3.1 70B 在教学中表现最优。

**[ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](llm_evaluation/elaboration_competitive_programming.md)**

:   提出首个全面评估**人类-LLM协作竞赛编程**的基准ELABORATION，通过覆盖编程全流程的人类反馈分类体系和8320题精标注数据集，揭示LLM独立解题能力有限（困难题仅3.4% Pass@1），但人类反馈（尤其编码阶段的专家反馈）可带来平均9.3%的显著提升。

**[EvoWiki: Evaluating LLMs on Evolving Knowledge](llm_evaluation/evowiki_evaluating_llms_on_evolving_knowledge.md)**

:   提出 EvoWiki，一个可自动更新的动态评估基准，将知识分为稳定 (stable)、演化 (evolved) 和未知 (uncharted) 三级，系统评估 LLM 对演化知识的利用能力，发现 RAG 和持续学习 (CL) 结合使用具有协同效应。

**[Exposing Numeracy Gaps: A Benchmark to Evaluate Fundamental Numerical Abilities in Large Language Models](llm_evaluation/exposing_numeracy_gaps_a_benchmark_to_evaluate_fundamental_numerical_abilities_i.md)**

:   提出 NumericBench 综合基准，通过 6 类数据集评估 LLM 的 6 种基本数值能力（数字识别、算术运算、上下文检索、比较、汇总、逻辑推理），发现包括 GPT-4o、DeepSeek-V3 在内的 SOTA 模型在简单数值任务上仍表现极差，并深入分析了 5 种根因。

**[FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging](llm_evaluation/financereasoning_benchmarking_financial_numerical_reasoning_more.md)**

:   提出 FinanceReasoning benchmark，通过重标注公开数据集、构建 3,133 个 Python 金融函数库和新增 908 道专家标注难题，在可信度、全面性和挑战性三个维度提升金融数值推理评估能力。

**[From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions](llm_evaluation/from_tools_to_teammates_evaluating_llms_in_multi-session_coding_interactions.md)**

:   提出 MemoryCode 合成多会话数据集评估 LLM 在长期交互中追踪和执行编码指令的能力，发现即使 GPT-4o 在提供完整对话历史时准确率也下降 67%，揭示了当前 LLM 在前瞻性记忆和信息整合上的根本局限。

**[GRACE: A Granular Benchmark for Evaluating Model Calibration Against Human Calibration](llm_evaluation/grace_a_granular_benchmark_for_evaluating_model_calibration_against_human_calibr.md)**

:   提出GRACE基准，通过渐进式增量问答和真人vs模型竞赛收集1749个数据点，以人类校准表现为参照评估LLM校准能力，并引入CalScore指标发现：虽然人类准确率可能低于模型，但人类在校准方面普遍优于SOTA模型——模型在不确定时过度自信、在正确时又信心不足。

**[GuessArena: Guess Who I Am? A Self-Adaptive Framework for Evaluating LLMs in Domain-Specific Knowledge and Reasoning](llm_evaluation/guessarena_guess_who_i_am_a.md)**

:   提出 GuessArena，一种基于"猜猜我是谁"博弈游戏的自适应 LLM 评估框架，通过领域知识建模和多轮交互推理，在五个垂直行业中有效区分模型的领域知识和推理能力。

**[HalluLens: LLM Hallucination Benchmark](llm_evaluation/hallulens_llm_hallucination_benchmark.md)**

:   提出了 HalluLens 幻觉基准，明确区分幻觉与事实性，建立了外在幻觉（与训练数据不一致）和内在幻觉（与输入上下文不一致）的清晰分类体系，引入三个动态可重生成的外在幻觉评估任务，并全面分析了现有基准的局限性。

**[HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs in Commonsense Reasoning](llm_evaluation/hellaswag-pro_a_large-scale_bilingual_benchmark_for_evaluating_the_robustness_of.md)**

:   构建首个大规模双语（中英）LLM 常识推理鲁棒性评估基准 HellaSwag-Pro，通过 7 种推理形式变体对 1,600 道原始题生成 11,200 道变体题，在 41 个 LLM 上的系统评估表明所有模型在常识推理鲁棒性上远未达标——否定变换平均准确率仅 9.01%，人机差距显著。

**[Help Me Write a Story: Evaluating LLMs' Ability to Generate Writing Feedback](llm_evaluation/help_write_story_feedback.md)**

:   本文定义了"LLM 生成写作反馈"这一新任务，构建了包含 1,300 个带有受控写作缺陷的故事数据集（StoryFeedback，共 83K 对故事-反馈），通过自动指标和人工评估系统地测试了 8 个 LLM 在反馈的具体性、正确性、问题检测和正面评价适当性四个维度的表现，发现模型能给出具体且基本正确的反馈，但常常抓不住最大的写作问题，且不善于判断何时该给正面评价。

**[HomeBench: Evaluating LLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multiple Devices](llm_evaluation/homebench_evaluating_llms_in_smart_homes_with_valid_and_invalid_instructions_acr.md)**

:   本文提出HomeBench，首个同时包含合法与非法指令、覆盖单设备和多设备场景的智能家居LLM评估基准，发现即使是GPT-4o在包含非法指令的多设备场景中成功率也仅为0.0%。

**[How Far are LLMs from Being Our Digital Twins? A Benchmark for Persona-Based Behavior Chain Simulation](llm_evaluation/how_far_are_llms_from_being_our_digital_twins_a_benchmark_for_persona-based_beha.md)**

:   本文提出BehaviorChain基准，首次评估LLM模拟连续人类行为的能力，包含1001个人格画像下的15846个行为样本，发现即使是最先进的模型在连续行为模拟任务上仍表现不佳。

**[HPSS: Heuristic Prompting Strategy Search for LLM Evaluators](llm_evaluation/hpss_heuristic_prompting_strategy_search_for_llm_evaluators.md)**

:   整合 8 个影响 LLM 评估提示效果的关键因子（评分尺度、ICL 示例、评估标准、参考答案、CoT、AutoCoT、度量指标、组件顺序），提出基于遗传算法的启发式提示策略搜索方法 HPSS，在 12,960 种组合空间中高效找到最优提示策略，仅用基线 5% 的生成成本即超越 G-Eval 和 CloserLook。

**[Influences on LLM Calibration: A Study of Response Agreement, Loss Functions, and Prompt Styles](llm_evaluation/influences_on_llm_calibration_a_study_of_response_agreement_loss_functions_and_p.md)**

:   本文系统研究影响 LLM 校准（calibration）的三大因素——多模型响应一致性、损失函数选择和 prompt 风格，提出 Calib-n 框架通过训练辅助模型聚合多个 LLM 的响应来估计置信度，发现响应一致性和 focal loss 能显著改善校准性能。

**[JuStRank: Benchmarking LLM Judges for System Ranking](llm_evaluation/justrank_llm_judge_system_ranking.md)**

:   首次大规模研究 LLM 判官在系统排名任务中的表现，提出 JuStRank 基准，收集 48 个判官对 63 个系统的 150 万条评分，揭示实例级判断能力与系统级排名能力之间存在显著差距，并发现判官的"果断性"（decisiveness）和"系统特异性偏见"两个可量化的系统级行为特征。

**[KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding](llm_evaluation/kitab-bench_a_comprehensive_multi-domain_benchmark_for_arabic_ocr_and_document_u.md)**

:   KITAB-Bench 是一个涵盖 9 大领域 36 个子领域共 8,809 个样本的综合性阿拉伯语 OCR 基准，评估结果显示现代视觉语言模型（如 GPT-4o、Gemini）在字符错误率上平均超过传统 OCR 方法 60%，但在 PDF-to-Markdown 转换中最优模型仅达到 65% 准确率，凸显了阿拉伯语文档理解的巨大挑战。

**[KRISTEVA: Close Reading as a Novel Task for Benchmarking Interpretive Reasoning](llm_evaluation/kristeva_close_reading_as_a_novel_task_for_benchmarking_interpretive_reasoning.md)**

:   本文提出 KRISTEVA，首个评估 LLM 细读（close reading）能力的基准，包含 1331 道从大学课堂数据中构建的多选题，覆盖风格特征提取、上下文检索、特征-上下文多跳推理三个递进难度层次，19 个 SOTA LLM 在 11 个任务中的 10 个上仍落后于人类专家。

**[La Leaderboard: A Large Language Model Leaderboard for Spanish Varieties and Languages of Spain and Latin America](llm_evaluation/la_leaderboard_spanish.md)**

:   构建首个面向西班牙和拉丁美洲语言的开源LLM排行榜，整合66个数据集覆盖西班牙语、加泰罗尼亚语、巴斯克语、加利西亚语，评估50个模型并分析训练策略、算力与性能的关系。

**[Language Complexity Measurement as a Noisy Zero-Shot Proxy for Evaluating LLM Performance](llm_evaluation/language_complexity_measurement_as_a_noisy_zero-shot_proxy_for_evaluating_llm_pe.md)**

:   利用语言复杂度计算任务（LIX 可读性指标和平均依存距离 ADD）作为 LLM 通用能力的零样本代理评估方法，在瑞典语论文上测试 6 个模型，发现 LIX 误差与 MMLU 分数呈强负相关（$r=-0.875$, $p=0.026$），表明结构分析能力可作为模型通用能力的廉价近似指标。

**[Language Model Probabilities are Not Calibrated in Numeric Contexts](llm_evaluation/language_model_probabilities_are_not_calibrated_in_numeric_contexts.md)**

:   系统研究了语言模型在数值上下文中的概率校准问题，发现即使在简单场景（如从袋中取弹珠）下，包括 GPT-4o 在内的所有测试模型均严重校准不良，存在基于词序、词频和词标识的系统性偏差（如某些模型总选第一个选项，其他模型总选第二个），指令微调加剧了模式崩塌。

**[MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset](llm_evaluation/mars_benchmarking_the_metaphysical_reasoning_abilities_of_language_models_with_a.md)**

:   本文提出了 **Metaphysical Reasoning（形而上推理）** 的形式化定义，将分布变化下的推理分解为三步判别过程，并构建了首个大规模评估基准 Mars（355K 标注数据），实验表明 20+ 语言模型在该任务上表现均不理想，揭示了 LLM 在理解事件组成要素变化及其因果效应方面的显著短板。

**[McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](llm_evaluation/mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)**

:   提出首个多任务中文偏见评估基准 McBE，包含 4,077 条偏见评估实例（BEI），覆盖 12 种偏见类别和 82 个子类别，通过 5 种评估任务（偏好计算/子类别分类/场景选择/偏见分析/偏见评分）多角度量化 LLM 中的中文偏见，并揭示"参数越大偏见越强"的传统结论可能源于单任务评估的局限性。

**[MDBench: A Synthetic Multi-Document Reasoning Benchmark Generated with Knowledge Guidance](llm_evaluation/mdbench_a_synthetic_multi-document_reasoning_benchmark_generated_with_knowledge_.md)**

:   提出 MDBench，一个通过「结构化知识→LLM 辅助增强→自然文本生成」管线合成的多文档推理 QA 基准，可控地注入跨文档依赖，对前沿 LLM 构成显著挑战（最佳模型 EM 仅~60%）。

**[Mis-prompt: Benchmarking Large Language Models for Proactive Error Handling](llm_evaluation/mis-prompt_benchmarking_large_language_models_for_proactive_error_handling.md)**

:   提出 Mis-prompt 基准，包含 4 项评估任务、14 类错误分类体系和 14,969 条数据集，系统研究 LLM 在无显式错误处理指令时的**主动**纠错能力，发现当前 LLM 主动纠错能力严重不足，SFT 可显著提升。

**[MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark](llm_evaluation/mmlu-cf_a_contamination-free_multi-task_language_understanding_benchmark.md)**

:   提出 MMLU-CF，一个包含 20,000 道题的无数据污染多任务语言理解基准，通过从更广泛的来源收集数据并应用三条去污染规则（改写题目、打乱选项、随机替换选项）来避免无意和恶意的数据泄露，最强模型 GPT-4o 在该基准上仅获得 73.4%（MMLU 上为 88.0%）。

**[Movie101v2: Improved Movie Narration Benchmark](llm_evaluation/movie101v2_improved_movie_narration_benchmark.md)**

:   提出 Movie101v2 大规模双语电影叙事基准（203 部电影、46K 中英文视频-叙事对），将自动电影叙事拆解为 L1 视觉事实描述 → L2 情节叙述 → L3 可部署 AD 三阶段渐进目标，设计基于 LLM 的分级评估框架，系统基线测试多种 LVLM 并深入分析视觉感知与文本生成的核心瓶颈。

**[Navigating Rifts in Human-LLM Grounding: Study and Benchmark](llm_evaluation/navigating_rifts_in_human-llm_grounding_study_and_benchmark.md)**

:   系统研究人与 LLM 对话中的 grounding（建立共识）失败问题，发现 LLM 主动澄清的频率仅为人类的 1/3、主动追问的频率仅为 1/16，提出 Rifts 基准（约 1.8K 任务）评测 LLM 的 grounding 能力，并通过 grounding forecaster 实现初步干预。

**[NorEval: A Norwegian Language Understanding and Generation Evaluation Benchmark](llm_evaluation/noreval_a_norwegian_language_understanding_and_generation_evaluation_benchmark.md)**

:   本文提出 NorEval，一个包含 24 个人工创建数据集、覆盖 9 类任务的挪威语综合评测套件，系统地评测了 19 个开源挪威语语言模型在语言理解和生成上的能力，发现模型在常识推理、真实性和指令遵循上仍显著落后于人类。

**[ONEBench to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities](llm_evaluation/onebench_to_test_them_all_sample-level_benchmarking_over_open-ended_capabilities.md)**

:   ONEBench提出了一种新的基准评测范式：将多个评测数据集的样本合并为统一数据池，通过Plackett-Luce排名聚合算法在样本级别进行模型比较，支持异构指标聚合、不完整数据处理和个性化能力探测。

**[Pap2Pat: Benchmarking Outline-Guided Long-Text Patent Generation with Patent-Paper Pairs](llm_evaluation/pap2pat_benchmarking_outline-guided_long-text_patent_generation_with_patent-pape.md)**

:   构建了包含 1.8k 专利-论文配对的 Pap2Pat 基准，提出基于大纲的分块专利描述生成方法 COPGen，并设计了基于 NLI 的事实性/覆盖率/风格评估指标，系统评测了当前 LLM 在超长专利文档生成上的能力与不足。

**[PapersPlease: A Benchmark for Evaluating Motivational Values of Large Language Models Based on ERG Theory](llm_evaluation/papersplease_a_benchmark_for_evaluating_motivational_values_of_large_language_mo.md)**

:   提出PapersPlease基准，包含3700个基于ERG动机理论构建的道德困境场景，让LLM扮演移民检查官决定是否放行，揭示了6个LLM在动机价值优先级上的显著差异以及对边缘化身份群体的偏见。

**[PATCH: Psychometrics-Assisted Benchmarking of LLMs Against Human Populations](llm_evaluation/patch_psychometrics-assisted_benchmarking_of_large_language_models_against_human.md)**

:   提出 PATCH 框架，将心理测量学中的项目反应理论（IRT 3PL/2PL 模型）引入 LLM 基准测试，在 TIMSS 2011 八年级数学测试（88 道题、56 个国家/地区）上对比 GPT-4V、Gemini-Pro-Vision、Qwen-VL 与人类群体的能力值，发现 IRT 能力估计与简单准确率排名显著不同，GPT-4V 与韩国/新加坡/中国台北学生处于同一排名区间；同时发布 4 个高质量数据集（TIMSS 2011 & 2008 数学/科学/物理）。

**[PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning](llm_evaluation/physreason_a_comprehensive_benchmark_towards_physics-based_reasoning.md)**

:   提出 PhysReason 基准，包含 1200 道物理题（平均 8.1 步解题），设计了答案级和步骤级两层自动评估框架 PSAS，揭示顶尖模型（Deepseek-R1、o3-mini）在物理推理上准确率不足 60%，并识别出四大推理瓶颈。

**[READoc: A Unified Benchmark for Realistic Document Structured Extraction](llm_evaluation/readoc_a_unified_benchmark_for_realistic_document_structured_extraction.md)**

:   READoc 提出了首个将文档结构化提取（DSE）定义为端到端 PDF 到 Markdown 转换的统一基准，包含 3,576 篇来自 arXiv/GitHub/Zenodo 的真实文档和三模块评估套件（标准化+分割+评分），首次揭示了当前 DSE 系统与真实场景需求之间的差距。

**[RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis](llm_evaluation/realhitbench_a_comprehensive_realistic_hierarchical_table_benchmark_for_evaluati.md)**

:   提出 RealHiTBench——首个全面评估 LLM 对复杂层次化表格理解能力的基准，包含 708 张来自 13 个平台、24 个领域的真实复杂表格和 3,752 道题目，定义了 5 种复杂结构类型和 5 大任务类型，并提出基于树结构的 TreeThinker 推理管线显著提升模型对层次化表头的理解能力。

**[Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](llm_evaluation/retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)**

:   提出ToolRet——首个大规模工具检索基准（7.6k检索任务、43k工具），揭示现有强IR模型在工具检索任务上表现不佳（最强模型nDCG@10仅33.83），并贡献超20万训练实例的ToolRet-train数据集，显著提升IR模型的工具检索能力和端到端工具使用任务通过率。

**[Revisiting 3D LLM Benchmarks: Are We Really Testing 3D Capabilities?](llm_evaluation/revisiting_3d_llm_benchmarks_are_we_really_testing_3d_capabilities.md)**

:   揭示了 3D LLM 评测中的"2D-Cheating"问题——将点云渲染为图像后，2D VLM 在部分基准上超越 3D SOTA，说明这些基准未能有效评估真正的 3D 理解能力，并据此提出了有效 3D 评测的设计原则。

**[Right Answer, Wrong Score: Uncovering the Inconsistencies of LLM Evaluation in Multiple-Choice QA](llm_evaluation/right_answer_wrong_score_uncovering_the_inconsistencies_of_llm_evaluation_in_mul.md)**

:   系统揭示了LLM在多选问答(MCQA)评估中的不一致性——不同评估策略(RegEx/Logprobs/xFinder)和提示设置(约束/自由生成)组合会导致模型性能报告产生显著差异，且即使是SOTA的LLM-based答案提取器也无法可靠识别推理矛盾，呼吁建立标准化评估协议。

**[RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios](llm_evaluation/rulearena_rule_guided_reasoning.md)**

:   提出 RuleArena——一个基于航空行李费、NBA交易规则、税务法规三个真实场景的benchmark，用于评估LLM遵循复杂自然语言规则进行推理的能力；实验发现即使最强模型（o1-preview）在最难任务上准确率也不足50%，暴露了LLM在规则召回、规则区分和数学计算三方面的系统性缺陷。

**[SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models' Knowledge of Indian Culture](llm_evaluation/sanskriti_a_comprehensive_benchmark_for_evaluating_language_models_knowledge_of_.md)**

:   构建了覆盖印度全部 36 个行政区域、16 类文化属性、21,853 道 MCQ 的大规模文化知识基准 SANSKRITI，在 11 个 LLM/SLM/ILM 上的零样本评测揭示模型文化知识存在严重的地域和属性不均衡。

**[SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](llm_evaluation/seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)**

:   提出 SeedBench——首个面向种子科学（种子育种）的多任务 LLM 评测基准，包含 2,264 道专家验证题目，覆盖基因信息检索、基因功能调控和品种选育三大育种流程，对 26 个 LLM 进行系统评估，揭示了当前 LLM 与真实育种需求之间的显著差距。

**[skLEP: A Slovak General Language Understanding Benchmark](llm_evaluation/sklep_a_slovak_general_language_understanding_benchmark.md)**

:   本文提出 skLEP，首个面向斯洛伐克语的综合性自然语言理解基准，包含 9 个多层级任务（词级、句对级、文档级），并对斯洛伐克语专有模型、多语言模型和英语模型进行了系统评测，发现 mDeBERTaV3 在平均性能上超越了所有斯洛伐克专有模型。

**[Something's Fishy In The Data Lake: A Critical Re-evaluation of Table Union Search Benchmarks](llm_evaluation/somethings_fishy_in_the_data_lake_a_critical_re-evaluation_of_table_union_search.md)**

:   系统性分析了主流表联合搜索 (Table Union Search, TUS) 基准测试的三大结构性缺陷——过度重叠、语义简单、真值噪声，揭示简单的词袋 (BoW) 和预训练嵌入基线就能在这些基准上达到或超越复杂 SOTA 方法的效果，调研结论指出当前基准无法有效评估语义理解能力。

**[StrucText-Eval: Evaluating LLM's Reasoning on Structure-Rich Text](llm_evaluation/structext_eval.md)**

:   提出 StrucText-Eval——通过自动生成语义无关的结构化文本样本，覆盖 8 种结构化语言和 29 个任务共 5,800 个样本，以可控的嵌套深度和宽度调节难度，揭示最强开源 LLM 在困难集上仅 45.8% 而人类达 92.6%，系统性暴露了 LLM 在纯结构推理上的严重短板。

**[StructFlowBench: A Structured Flow Benchmark for Multi-turn Instruction Following](llm_evaluation/structflowbench_a_structured_flow_benchmark_for_multi-turn_instruction_following.md)**

:   提出 StructFlowBench，一个融入结构流建模的多轮指令遵循基准测试，定义了六种基本的轮间关系（跟进、精炼、回忆、总结、扩展、不相关），建立了双层约束评估体系（轮内约束 + 轮间结构约束），系统评估了 13 个主流 LLM 在多轮对话结构理解上的能力。

**[SwiLTra-Bench: The Swiss Legal Translation Benchmark](llm_evaluation/swiltra-bench_the_swiss_legal_translation_benchmark.md)**

:   构建了 SwiLTra-Bench——一个包含超过 18 万对齐瑞士法律翻译对的大规模多语言基准（覆盖法律、判例摘要、新闻稿，涵盖德法意罗英五种语言），系统评估了前沿 LLM 和微调开源 SLM 在法律翻译上的表现，并提出 SwiLTra-Judge 自动评估方法。

**[TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining](llm_evaluation/tic-lm_a_web-scale_benchmark_for_time-continual_llm_pretraining.md)**

:   本文提出TiC-LM，一个基于114个月Common Crawl数据（2.9T tokens）的大规模时间持续学习基准，通过150+实验系统评估了优化器、数据回放和正则化方法在持续预训练场景下的表现，发现自回归学习率调度结合固定比例数据回放可以在仅2.6倍计算量下接近从头训练的性能。

**[Towards Dynamic Theory of Mind: Evaluating LLM Adaptation to Temporal Evolution of Human States](llm_evaluation/towards_dynamic_theory_of_mind_evaluating_llm_adaptation_to_temporal_evolution_o.md)**

:   提出 DynToM 基准，通过 1,100 个社会情境中 5,500 个时序关联场景和 78,100 道题目，评估 LLM 追踪人类心理状态时序演化的能力，揭示模型平均落后人类 44.7%。

**[Towards Objective Fine-tuning: How LLMs' Prior Knowledge Causes Potential Poor Calibration?](llm_evaluation/towards_objective_fine-tuning_how_llms_prior_knowledge_causes_potential_poor_cal.md)**

:   揭示LLM的先验知识在微调过程中会导致校准退化（已知数据引发过度自信，未知数据反而有利于校准），提出CogCalib认知感知校准框架，在训练中根据知识偏差动态应用不同学习策略，在保持任务性能的同时平均降低57%的ECE。

**[TripCraft: A Benchmark for Spatio-Temporally Fine Grained Travel Planning](llm_evaluation/tripcraft_a_benchmark_for_spatio-temporally_fine_grained_travel_planning.md)**

:   提出TripCraft——一个整合真实世界时空约束（公共交通、活动可用性、用户画像等）的旅行规划基准数据集，配套五个连续评估指标，系统评估LLM生成行程的质量，在参数指导设置下将餐饮时间分数从61%提升至80%。

**[TripTailor: A Real-World Benchmark for Personalized Travel Planning](llm_evaluation/triptailor_a_real-world_benchmark_for_personalized_travel_planning.md)**

:   提出 TripTailor，一个基于真实数据的大规模旅行规划 benchmark，包含 40 个城市的 50 万+ POI 和近 4000 条真实行程，并引入可行性、合理性和个性化三维评估框架，发现最先进 LLM 生成的行程不到 10% 能达到人类水平。

**[TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages](llm_evaluation/tumlu_a_unified_and_native_language_understanding_benchmark_for_turkic_languages.md)**

:   提出 TUMLU 和 TUMLU-mini，首个面向突厥语系 9 种语言的原生多任务语言理解基准，包含 38,139 道中高中学科多选题，覆盖拉丁/西里尔/阿拉伯三种文字系统，系统评估了 13 个开源与闭源 LLM，揭示了文字系统、语言资源量和 CoT 对模型性能的差异化影响。

**[VoxEval: Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models](llm_evaluation/voxeval_benchmarking_the_knowledge_understanding_capabilities_of_end-to-end_spok.md)**

:   提出 VoxEval，首个支持端到端纯语音输入-输出评估的 SpeechQA 基准，涵盖 56 个学科、26 种输入音频条件，系统揭示了当前端到端语音大模型在知识理解和数学推理方面的严重不足。

**[WebWalker: Benchmarking LLMs in Web Traversal](llm_evaluation/webwalker_benchmarking_llms_in_web_traversal.md)**

:   提出 WebWalkerQA 基准评测 LLM 在网页深层遍历中的信息获取能力，并设计 WebWalker 多智能体框架通过 Explore-Critic 范式模仿人类网页导航行为，结合 RAG 的横向-纵向集成显著提升了复杂问答性能。

**[Where Are We? Evaluating LLM Performance on African Languages](llm_evaluation/where_are_we_evaluating_llm_performance_on_african_languages.md)**

:   构建了覆盖517种非洲语言、30个数据集、16类任务的 Sahara 基准，系统评估24个LLM在非洲语言上的表现，揭示语言政策驱动的数据不平等如何直接影响模型效果。

**[WiCkeD: A Simple Method to Make Multiple Choice Benchmarks More Challenging](llm_evaluation/wicked_a_simple_method_to_make_multiple_choice_benchmarks_more_challenging.md)**

:   提出 WiCkeD 方法，通过随机将多选题的一个选项替换为"以上都不对"来增加现有基准难度，使18个 LLM 的平均性能下降12.1个百分点，且链式思维推理也无法弥补这一下降。

**[WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](llm_evaluation/wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)**

:   提出首个面向极端天气影响理解的LLM评估基准WXImpactBench，包含四阶段数据构建流水线和两个评估任务（多标签分类与排序问答），系统性评估了多个LLM在气候适应领域的能力。

**[YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering](llm_evaluation/yescieval_llm_judge_science.md)**

:   提出YESciEval框架，结合九维细粒度评估准则和SFT+RL对齐策略来缓解LLM评估者的乐观偏差(optimism bias)，在科学问答场景下构建鲁棒的开源LLM-as-a-Judge系统，无需人类标注和闭源模型。

---

## 🔍 信息检索/RAG { #information_retrieval }

**[A Reality Check on Context Utilisation for Retrieval-Augmented Generation](information_retrieval/a_reality_check_on_context_utilisation_for_retrieval-augmented_generation.md)**

:   提出DRUID真实世界事实验证数据集和ACU评估指标，揭示合成数据集（CounterFact、ConflictQA）夸大了上下文特征的影响，导致对LLM上下文利用能力的过度乐观评估，呼吁使用真实检索数据研究RAG。

**[A Text is Worth Several Tokens: Text Embedding from LLMs Secretly Aligns Well with The Key Tokens](information_retrieval/a_text_is_worth_several_tokens_text_embedding_from_llms_secretly_aligns_well_wit.md)**

:   揭示 LLM 文本嵌入的有趣现象：将嵌入向量通过解码层映射回词表空间后，解码概率最高的 token 与输入文本的关键词高度对齐；进一步通过谱分析发现这一现象主要受第一主成分控制，并据此提出一种简洁的稀疏检索方法，达到原密集检索 80%+ 的效果。

**[Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps](information_retrieval/accelerating_adaptive_retrieval_augmented_generation_via_instruction-driven_repr.md)**

:   提出 IDR²，一种模型无关的自适应RAG加速框架，通过消除多轮检索间重叠文档的冗余表示并利用检索内容指导并行解码，实现端到端约2倍加速且不损失生成质量。

**[AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark](information_retrieval/air-bench_automated_heterogeneous_information_retrieval_benchmark.md)**

:   提出AIR-Bench——首个利用LLM自动生成测试数据的异构IR基准，覆盖2个任务（QA/长文档）、9个领域、13种语言共69个数据集，三阶段质量控制管线确保生成数据与人工标注高度一致，解决了传统IR基准领域覆盖有限和更新成本高的问题。

**[Any Information Is Just Worth One Single Screenshot: Unifying Search With Visualized Information Retrieval](information_retrieval/any_information_is_just_worth_one_single_screenshot_unifying_search_with_visuali.md)**

:   本文正式定义了可视化信息检索（Vis-IR）范式——将多模态信息统一渲染为截图（Screenshot）进行检索，构建了包含1300万截图的VIRA数据集、UniSE检索模型家族和MVRB基准测试，为统一搜索引擎奠定基础。

**[Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models](information_retrieval/astute_rag_knowledge_conflicts.md)**

:   Astute RAG 提出了一种对不完美检索具有鲁棒性的 RAG 方法，通过自适应生成 LLM 内部知识作为补充、带有来源标注的知识整合、以及基于可靠性的答案生成三个步骤，在 Gemini 和 Claude 上显著优于现有鲁棒 RAG 方法，且是唯一在最坏情况下（检索全部无用）不劣于无 RAG 基线的方法。

**[Atomic LLM: A Fine-Grained Information Retrieval Evaluation Benchmark for Language Models](information_retrieval/atomic_llm_a_fine-grained_information_retrieval_evaluation_benchmark_for_languag.md)**

:   本文提出Atomic LLM基准，通过将信息检索评估分解为原子级别（Atomic Level）的事实检索任务，从事实精确性、来源归因、粒度覆盖等多个维度对LLM的信息检索能力进行细粒度评估，揭示了现有LLM在精确事实提取方面的系统性不足。

**[Automatic Benchmark Generation from Scientific Papers via Retrieval-Augmented LLMs](information_retrieval/automatic_benchmark_generation_from_scientific_papers_via_retrieval-augmented_ll.md)**

:   本文提出一种基于检索增强LLM的自动基准生成方法，从科学论文中自动抽取可测试的知识点并生成高质量的评估题目，在NLP、机器学习、生物信息学等领域验证了方法的有效性，为领域特定LLM评估基准的快速构建提供了新范式。

**[Beyond True or False: Retrieval-Augmented Hierarchical Analysis of Nuanced Claims](information_retrieval/beyond_true_or_false_retrieval-augmented_hierarchical_analysis_of_nuanced_claims.md)**

:   提出 ClaimSpect 框架，将复杂声明自动分解为层次化的方面（aspect）树，并通过区分性检索从语料库中发现各方面的支持/中立/反对观点及其共识程度。

**[CART: A Generative Cross-Modal Retrieval Framework with Coarse-To-Fine Semantic Modeling](information_retrieval/cart_a_generative_cross-modal_retrieval_framework_with_coarse-to-fine_semantic_m.md)**

**[CoIR: A Comprehensive Benchmark for Code Information Retrieval Models](information_retrieval/coir_a_comprehensive_benchmark_for_code_information_retrieval_models.md)**

:   提出 CoIR，首个全面的代码信息检索基准，包含 10 个数据集、覆盖 4 大类 8 个子任务和 14 种编程语言，揭示了即使是 SOTA 检索模型在代码检索中也表现不佳，并指出许多模型已在现有排行榜上过拟合。

**[Collapse of Dense Retrievers: Short, Early, and Literal Biases Outranking Factual Evidence](information_retrieval/collapse_dense_retrievers.md)**

:   本文首次系统研究稠密检索器中多种启发式偏见（简短偏见、前置偏见、字面偏见、重复偏见）的个体和组合效应，发现当多种偏见叠加时，检索器选择包含答案的文档的概率低于10%，且这些偏见可被利用来操纵RAG系统，导致34%的性能下降。

**[ComRAG: Retrieval-Augmented Generation with Dynamic Vector Stores for Real-time Community Question Answering in Industry](information_retrieval/comrag_retrieval-augmented_generation_with_dynamic_vector_stores_for_real-time_c.md)**

:   提出ComRAG——一个面向工业实时社区问答的检索增强生成框架，通过**静态知识向量库+高/低质量动态QA向量库**的三库架构和**质心记忆机制**，在三个CQA数据集上获得向量相似度最高25.9%的提升，同时降低延迟8.7%-23.3%。

**[Contradiction Detection in RAG-Based Chatbots](information_retrieval/contradiction_detection_in_rag-based_chatbots.md)**

:   本文针对 RAG 对话系统中检索文档与生成回答之间的矛盾问题，提出了一种多粒度矛盾检测框架，能够识别显式矛盾、隐式矛盾和部分矛盾，并提供可解释的矛盾定位。

**[CoRe-MMRAG: Cross-Source Knowledge Reconciliation for Multimodal RAG](information_retrieval/core_mmrag_knowledge_reconciliation.md)**

:   CoRe-MMRAG 提出了一个端到端多模态 RAG 框架，通过四阶段流水线（参数知识生成→视觉-文本联合重排序→外部知识生成→内外知识整合）解决参数知识-检索知识不一致(PRKI)和视觉-文本知识不一致(VTKI)两个问题，在 InfoSeek 和 Encyclopedic-VQA 上分别提升 5.6% 和 9.3%。

**[Cross-Lingual Relevance Transfer for Document Retrieval](information_retrieval/cross-lingual_relevance_transfer_for_document_retrieval.md)**

:   本文提出一种跨语言相关性迁移方法，通过在高资源语言（如英语）上训练的检索模型将相关性判断能力迁移到低资源语言，在多个跨语言文档检索基准上显著超越现有方法。

**[Divide-Then-Align: Honest Alignment based on the Knowledge Boundary of RAG](information_retrieval/divide_then_align_rag_knowledge_boundary.md)**

:   DTA 提出将 RAG 查询按参数知识边界和检索知识边界划分为四个象限，对"两者都不知道"的查询构造偏好数据用 DPO 训练模型回答"我不知道"，解决了 RAFT 模型即使在检索完全噪声时也强行生成答案的问题，在准确率和适当弃权之间实现了有效平衡。

**[Don't Reinvent the Wheel: Efficient Instruction-Following Text Embedding based on Guided Space Transformation](information_retrieval/dont_reinvent_the_wheel_efficient_instruction-following_text_embedding_based_on_.md)**

:   提出 GSTransform 框架，通过轻量级空间变换将预计算的通用嵌入实时适配到用户指令指定的语义空间，避免每次新指令都重新编码全部语料，在 9 个数据集上平均得分 66.01（SOTA 基线 55.31），同时实现 6~300 倍实时加速。

**[DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination](information_retrieval/drag_distilling_rag_slm.md)**

:   DRAG 提出了一种从大模型向小模型蒸馏 RAG 能力的框架：用大模型（如 GPT-4o）为给定问题生成证据和知识图谱三元组，经排序过滤后作为结构化上下文输入给小模型（2B-9B），无需微调即可将小模型在 ARC-C 上提升高达 27.7%，同时显著减少幻觉。

**[Drama: Diverse Augmentation from Large Language Models to Smaller Dense Retrievers](information_retrieval/drama_diverse_augmentation_from_large_language_models_to_smaller_dense_retriever.md)**

:   提出 Drama 框架，系统性地探索多种基于 LLM 的数据增强策略（裁剪句+合成查询+LLM 重排序偏好）与 LLM 剪枝 backbone 的结合，在单阶段对比学习中训练出 0.1B-1B 参数的小型检索器，在 BEIR 上以 0.3B 参数匹配 1B 参数的 Gecko，且具备强多语言和长上下文能力。

**[Empaths at SemEval-2025 Task 11: Retrieval-Augmented Approach to Perceived Emotions Prediction](information_retrieval/empaths_at_semeval-2025_task_11_retrieval-augmented_approach_to_perceived_emotio.md)**

:   提出 EmoRAG 系统，用检索增强生成（RAG）管道结合多 LLM 集成聚合，在 SemEval-2025 Task 11 多标签情感检测任务上无需额外训练即在 28 种语言中取得有竞争力的结果，平均 F1-micro 0.638。

**[Enhancing Lexicon-Based Text Embeddings with Large Language Models](information_retrieval/enhancing_lexicon-based_text_embeddings_with_large_language_models.md)**

:   提出 LENS 框架，首次将 LLM 用于通用词汇级文本嵌入（lexicon-based embedding），通过 token 嵌入聚类解决 LLM 词表冗余问题、引入双向注意力克服因果 LLM 的限制，在 MTEB 上超越同数据训练的稠密嵌入，且与稠密嵌入结合后在 BEIR 上达到 SOTA。

**[Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models](information_retrieval/evaluation_of_attribution_bias_in_generator-aware_retrieval-augmented_large_lang.md)**

:   定义并研究 RAG 中 LLM 对作者身份信息的归因敏感性和偏差，通过反事实评估发现告知 LLM 文档作者身份可显著改变归因质量 3-18%，且 LLM 存在对人类作者身份的归因偏差。

**[EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation](information_retrieval/exit_context-aware_extractive_compression_for_enhancing_retrieval-augmented_gene.md)**

:   提出 EXIT——一种抽取式上下文压缩框架，通过上下文感知的句子级二分类并行选取与查询相关的句子，在 QA 准确率和推理延迟上同时优于现有的抽生式和抽取式压缩方法。

**[FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation](information_retrieval/faithfulrag_fact_level_conflict.md)**

:   发现现有忠实 RAG 方法通过强制抑制参数知识来实现上下文忠实，但这增加了误解上下文的风险（不忠实错误减少 6.65% 的同时错误匹配增加 6.42%）。提出 FaithfulRAG，通过事实级冲突识别（自事实挖掘）和冲突推理（自思考模块）解决知识冲突，在 FaithEval/SQuAD/MuSiQue/RealtimeQA 上超越最强基线 8-9 个百分点。

**[FlashBack: Efficient Retrieval-Augmented Language Modeling for Fast Inference](information_retrieval/flashbackefficient_retrieval-augmented_language_modeling_for_long_context_infere.md)**

:   针对检索增强语言模型(RALM)中因检索内容前置(prepending)导致 KV cache 反复重算的推理效率问题，提出 FlashBack，将检索内容后置(appending)以保留输入的 KV cache，并用 Marking Token + LoRA 微调适配新的上下文模式，在 Llama 2-7B 上实现最高 4 倍推理加速且 perplexity 持平。

**[FlexRAG: A Flexible and Comprehensive Framework for Retrieval-Augmented Generation](information_retrieval/flexrag_a_flexible_and_comprehensive_framework_for_retrieval-augmented_generatio.md)**

:   提出 FlexRAG，一个面向研究和原型开发的开源 RAG 框架，支持文本、多模态和 Web 检索三种模式，通过内存映射和异步处理实现比同类框架（FlashRAG）低一个数量级的资源开销。

**[From Ambiguity to Accuracy: The Transformative Effect of Coreference Resolution on RAG Systems](information_retrieval/from_ambiguity_to_accuracy_the_transformative_effect_of_coreference_resolution_o.md)**

:   本文系统研究了共指消解（coreference resolution）对 RAG 系统中文档检索和问答生成两阶段的影响，发现共指消解能一致性提升检索性能（尤其 mean pooling 模型受益最大），在 QA 任务中小模型的性能提升显著大于大模型，甚至使小模型达到大模型的基线水平。

**[GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis](information_retrieval/gainrag_preference_alignment.md)**

:   发现 RAG 中检索器优化的"相关性"与 LLM 实际需要的"增益"存在系统性偏差——含正确答案的段落仍有近 50% 概率导致错误生成，而间接相关段落反而更有效。提出 GainRAG，通过对比解码困惑度定义"增益"信号，训练轻量选择器在检索器和 LLM 之间做增益导向的段落筛选，在 6 个 QA 数据集上全面超越 Standard RAG 和 Rerank 基线。

**[GaRAGe: A Benchmark with Grounding Annotations for RAG Evaluation](information_retrieval/garage_a_benchmark_with_grounding_annotations_for_rag_evaluation.md)**

:   GaRAGe 是一个包含 2366 个问题和超过 35K 条人工标注 grounding 段落的 RAG 基准，通过细粒度的 grounding 相关性标注，系统评估 LLM 在 RAG 场景下识别相关信息、拒绝回答和归因引用的能力。

**[Controllable and Reliable Knowledge-Intensive Task-Oriented Conversational Agents with Declarative Genie Worksheets](information_retrieval/genie_worksheets_tod_agent.md)**

:   Genie 提出了一个可编程的知识密集型任务导向对话框架，通过声明式 Worksheet 规范定义 Agent 策略，将 LLM 限制在语义解析和回复生成两个角色，由算法化运行时系统强制执行策略，实现从 21.8% 到 82.8% 的真实任务完成率提升。

**[Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](information_retrieval/gor_rag_long_context_summary.md)**

:   提出 Graph of Records（GoR），将 LLM 历史响应与检索文本块构建为图结构，用 GNN 学习节点间的语义和逻辑关联，配合 BERTScore 自监督训练目标，在四个长文本全局摘要数据集上比检索基线提升 8-19%（ROUGE 指标）。

**[GRAF: Graph Retrieval Augmented by Facts for Romanian Legal Multi-Choice Question Answering](information_retrieval/graf_graph_retrieval_augmented_by_facts_for_romanian_legal_multi-choice_question.md)**

:   提出GRAF算法，结合法律知识图谱（Law-RoG）和图注意力网络进行罗马尼亚语法律多选题问答，同时开源了首个罗马尼亚语法律MCQA数据集JuRO（10,836题）和法律语料库CROL。

**[Gumbel Reranking: Differentiable End-to-End Reranker Optimization](information_retrieval/gumbel_reranking.md)**

:   将 RAG 系统中的重排序过程重新建模为文档级 Top-k 注意力掩码问题，利用 Gumbel 技巧和松弛 Top-k 采样实现端到端可微优化，直接最小化最终语言建模损失，在 HotpotQA 上 Recall@5 提升 10.4%。

**[HASH-RAG: Bridging Deep Hashing with Retriever for Efficient, Fine Retrieval and Augmented Generation](information_retrieval/hash-rag_bridging_deep_hashing_with_retriever_for_efficient_fine_retrieval_and_a.md)**

:   Hash-RAG 将深度哈希技术系统集成到 RAG 框架中，实现了仅需传统方法 10% 检索时间的高效检索，并通过 Prompt-Guided Chunk-to-Context（PGCC）模块在保持效率的同时提升了生成质量。

**[Health-LLM: Personalized Retrieval-Augmented Disease Prediction System](information_retrieval/health-llm_personalized_retrieval-augmented_disease_prediction_system.md)**

:   提出 Health-LLM 框架，通过 LLM + Llama Index 从健康报告中提取特征评分、RAG 增强医学知识检索、CAAFE 自动特征工程结合 XGBoost 分类器，在 IMCS-21 中文远程医疗数据集上实现 Accuracy 0.833、F1 0.762 的疾病预测性能，大幅超越 GPT-4 few-shot+RAG (Acc 0.68) 和 fine-tuned LLaMA-2-13B (Acc 0.73)。

**[HELIOS: Harmonizing Early Fusion, Late Fusion, and LLM Reasoning for Multi-Granular Table-Text Retrieval](information_retrieval/helios_harmonizing_early_fusion_late_fusion_and_llm_reasoning_for_multi-granular.md)**

:   提出 HELIOS 三阶段图检索框架（边级早期融合 → 节点级晚期融合 → 星图级 LLM 精化），通过多粒度协调统一解决表格-文本检索中的检索单元粒度、查询依赖关系发现和高级推理三大挑战，在 OTT-QA 上实现 42.6% Answer Recall 提升。

**[Hierarchical Document Refinement for Long-context Retrieval-augmented Generation](information_retrieval/hierarchical_document_refinement_for_long-context_retrieval-augmented_generation.md)**

:   提出 LongRefiner，一个即插即用的长文档精炼系统，通过双层查询分析、层次化文档结构化和自适应精炼三个步骤，在 7 个 QA 数据集上以**仅 1/10 的 token 预算**实现了优于全文输入的性能，同时延迟仅为最佳基线的 1/10。

**[HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation](information_retrieval/hoh_a_dynamic_benchmark_for_evaluating_the_impact_of_outdated_information_on_ret.md)**

:   本文提出 HoH，首个专门评估过时信息对 RAG 系统影响的大规模动态基准，包含 96,124 个 QA 对和 219,463 篇文档，揭示了过时信息对 RAG 性能和安全性的严重危害。

**[HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases](information_retrieval/hybgrag_hybrid_rag_skb.md)**

:   提出 HybGRAG 方法，通过检索器库（Retriever Bank）同时利用文本和关系信息，配合 Critic 模块的自反思迭代纠正问题路由错误，在半结构化知识库上的混合问答任务中 Hit@1 平均提升 51%。

**[Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion](information_retrieval/hypothetical_documents_or_knowledge_leakage_rethinking_llm-based_query_expansion.md)**

:   质疑 LLM-based 查询扩展（HyDE/Query2doc）的性能提升是否来自"假设性文档生成"，发现性能增益仅在 LLM 生成的文档包含与 gold evidence 语义一致的句子时才一致出现，揭示了 benchmark 中可能存在的知识泄露问题。

**[Investigating Language Preference of Multilingual RAG Systems](information_retrieval/investigating_language_preference_of_multilingual_rag_systems.md)**

:   系统研究多语言 RAG 系统在检索和生成两个阶段的语言偏好问题，提出 MLRS 指标量化检索器对特定语言的偏好程度，揭示检索器偏好高资源语言和查询语言、生成器偏好查询语言和拉丁字母语言的现象，并设计 DKM-RAG 框架通过融合翻译段落与模型内部知识有效缓解偏好问题。

**[Investigating the Robustness of Retrieval-Augmented Generation at the Query Level](information_retrieval/investigating_the_robustness_of_retrieval-augmented_generation_at_the_query_leve.md)**

:   提出首个查询级 RAG 鲁棒性模块化分析框架，通过 5 种扰动类型 × 4 种检索器 × 3 种 LLM × 3 个数据集共 1092+ 实验，揭示 dense 与 sparse 检索器对不同扰动类型的互补鲁棒性，并给出可操作的工程建议。

**[KnowShiftQA: How Robust are RAG Systems when Textbook Knowledge Shifts in K-12 Education?](information_retrieval/knowshiftqa_rag_knowledge_shifts.md)**

:   构建了 KnowShiftQA 数据集（3,005 道题，覆盖 5 个学科），通过假设性知识更新模拟教科书与 LLM 参数知识的差异，系统评估 RAG 系统面对知识偏移时的鲁棒性，发现现有 RAG 系统在知识偏移下性能下降 22-27%。

**[LDIR: Low-Dimensional Dense and Interpretable Text Embeddings with Relative Representations](information_retrieval/ldir_low-dimensional_dense_and_interpretable_text_embeddings_with_relative_repre.md)**

:   提出 LDIR 方法，通过最远点采样选取锚文本（anchor texts），计算待编码文本与各锚文本的语义相关度，构建低维（≤500 维）、稠密且可解释的文本嵌入，性能接近黑盒模型并显著优于已有可解释嵌入方法。

**[Re-ranking Using Large Language Models for Mitigating Exposure to Harmful Content on Social Media Platforms](information_retrieval/llm_reranking_harmful_content.md)**

:   提出基于 LLM 的成对偏好重排序方法，在零样本和少样本设置下对社交媒体推荐序列中的有害内容进行降级排序，显著优于 Perspective API 和 OpenAI Moderation API 等工业级分类器，同时引入 PP-k 和 EWN 两个新评估指标。

**[Logical Consistency is Vital: Neural-Symbolic Information Retrieval for Negative-Constraint Queries](information_retrieval/logical_consistency_is_vital_neural-symbolic_information_retrieval_for_negative-.md)**

:   提出 NS-IR，通过将自然语言查询和文档转换为一阶逻辑（FOL），利用逻辑对齐和连接词约束两项技术优化稠密检索嵌入，显著提升了负约束查询等复杂逻辑检索场景的性能。

**[MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation](information_retrieval/main-rag_multi-agent_filtering_retrieval-augmented_generation.md)**

:   提出 MAIN-RAG，一个无需训练的多 Agent RAG 过滤框架，通过 Predictor→Judge→Final-Predictor 三个 LLM Agent 协作评估检索文档的相关性，并设计自适应阈值（基于分数均值和标准差）动态过滤噪声文档，在 4 个 QA 基准上实现 2-11% 的准确率提升。

**[Maximal Matching Matters: Preventing Representation Collapse for Robust Cross-Modal Retrieval](information_retrieval/maximal_matching_matters_preventing_representation_collapse_for_robust_cross-mod.md)**

:   提出 MaxMatch 方法，通过基于匈牙利算法的最大配对相似度和两个新损失函数，解决集合嵌入方法中的稀疏监督和集合坍塌问题，在 MS-COCO 和 Flickr30k 上取得 SOTA 性能。

**[MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation](information_retrieval/memerag_a_multilingual_end-to-end_meta-evaluation_benchmark_for_retrieval_augmen.md)**

:   构建首个原生多语言 RAG 元评估基准 MEMERAG，覆盖 5 种语言，通过流程图引导的标注达到高标注者一致性，用于评估和比较多语言 RAG 自动评估器。

**[Mitigating Lost-in-Retrieval Problems in RAG Multi-Hop QA](information_retrieval/mitigating_lost-in-retrieval_problems_in_retrieval_augmented_multi-hop_question_.md)**

:   本文识别 RAG 多跳问答中的"检索丢失"（lost-in-retrieval）问题——子问题分解后后续子问题因缺少关键实体导致检索性能骤降，提出 ChainRAG 框架通过构建句子图 + 渐进式检索 + 子问题重写（补全缺失实体）形成完整推理链，在 MuSiQue、2Wiki、HotpotQA 三个数据集上一致超越基线。

**[MoC: Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System](information_retrieval/moc_mixtures_of_text_chunking_learners_for_retrieval-augmented_generation_system.md)**

:   提出 Boundary Clarity 和 Chunk Stickiness 两个直接量化分块质量的指标，以及基于粒度感知混合专家（MoC）的文本分块框架，通过正则表达式引导的轻量化分块策略在 RAG 系统中取得优于传统方法和大模型直接分块的性能。

**[MT-RAIG: Novel Benchmark and Evaluation Framework for Retrieval-Augmented Insight Generation over Multiple Tables](information_retrieval/mt-raig_novel_benchmark_and_evaluation_framework_for_retrieval-augmented_insight.md)**

:   提出MT-RAIG Bench——首个面向多表格检索增强洞察生成的大规模基准，以及MT-RAIG Eval——基于分解的细粒度自动评估框架，实验表明即使前沿LLM在多表格推理上仍表现不佳（忠实度仅约40%，完整度约60%）。

**[Multilingual Retrieval Augmented Generation for Culturally-Sensitive Tasks: A Benchmark for Cross-lingual Robustness](information_retrieval/multilingual_retrieval_augmented_generation_for_culturally-sensitive_tasks_a_ben.md)**

:   构建了 BordIRLines 基准数据集，包含 49 种语言的领土争端查询及配对的 Wikipedia 检索文档，系统评估了多语言 RAG 环境下的跨语言鲁棒性，发现**检索多语言文档能比仅检索同语言文档更好地提高响应一致性并减少地缘政治偏差**。

**[NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering](information_retrieval/neusym_rag_pdf_qa.md)**

:   NeuSym-RAG 提出了一个混合神经-符号检索框架，将 PDF 文档通过多视角分块解析同时存入关系数据库和向量库，LLM Agent 通过可执行动作（SQL 查询 + 向量检索 + 查看图片等）迭代式交互检索，在学术论文 QA 上比经典 RAG 提升 17.3%。

**[On Synthetic Data Strategies for Domain-Specific Generative Retrieval](information_retrieval/on_synthetic_data_strategies_for_domain-specific_generative_retrieval.md)**

:   本文系统研究了针对领域特定语料库训练生成式检索模型的合成数据策略，提出多粒度查询生成、约束条件查询和基于硬负样本的偏好学习方法，显著提升检索性能。

**[Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval](information_retrieval/optimized_text_embedding_models_and_benchmarks_for_amharic_passage_retrieval.md)**

:   针对低资源、形态丰富的阿姆哈拉语（Amharic），提出基于预训练 Amharic BERT/RoBERTa 的稠密检索模型和 ColBERT 晚期交互模型，在参数量远小于多语言基线的情况下大幅提升段落检索效果，并建立了该语言首个系统性检索基准。

**[Pandora's Box or Aladdin's Lamp: A Comprehensive Analysis Revealing the Role of RAG Noise in Large Language Models](information_retrieval/pandora_box_rag_noise.md)**

:   本文从语言学视角定义了 RAG 系统中的 7 种噪声类型，构建了 NoiserBench 综合评测框架，通过 8 个 LLM 的大规模实验发现噪声可分为有害噪声（反事实、支持性、拼写）和有益噪声（语义、数据类型、非法句子），其中有益噪声反而能提升模型准确率 1-3%。

**[Parenting: Optimizing Knowledge Selection of Retrieval-Augmented Language Models with Parameter Decoupling and Tailored Tuning](information_retrieval/parenting_optimizing_knowledge_selection_of_retrievalaugmented.md)**

:   受人脑功能分区启发，提出 Parenting 框架，通过解耦并定位 LLM 参数空间中与"上下文遵循"(adherence)和"噪声鲁棒"(robustness)相关的子空间，并为不同子空间设计定制化微调策略，实现两种能力的平衡提升。

**[PRISM: A Framework for Producing Interpretable Political Bias Embeddings with Political-Aware Cross-Encoder](information_retrieval/prism_political_bias_embeddings.md)**

:   提出 PRISM 框架，首次将政治偏见嵌入建模为可解释任务：自动从弱标注新闻语料中挖掘争议性话题及左/右偏见指标作为嵌入维度，再用政治感知交叉编码器为文章在每个话题维度上打分，生成稀疏且语义透明的政治偏见嵌入向量，在 NewsSpectrum 分类准确率达 86.1%（领先 POLITICS 34.8%），同时支持多样化检索。

**[Uncovering Visual-Semantic Psycholinguistic Properties from the Distributional Structure of Text Embedding Space](information_retrieval/psycholinguistic_visual_semantic.md)**

:   提出Neighborhood Stability Measure (NSM)——一种无监督、无分布假设的方法，通过量化文本嵌入空间中邻域的稳定性来估计词语的可意象性(imageability)和具体性(concreteness)，仅使用文本模态即可超越依赖多模态或生成模型的已有方法。

**[RAEmoLLM: Retrieval Augmented LLMs for Cross-Domain Misinformation Detection Using In-Context Learning Based on Emotional Information](information_retrieval/raemollm_retrieval_augmented_llms_for_cross-domain_misinformation_detection_usin.md)**

:   提出 RAEmoLLM，首个基于情感信息检索的 RAG 框架，利用情感 LLM 的隐式嵌入构建检索数据库，为跨域虚假信息检测提供情感相关的 few-shot 示例，在三个基准上最高分别提升 15.64%、31.18% 和 15.73%（对比其他 few-shot 方法），无需微调。

**[RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework](information_retrieval/rageval_scenario_specific_rag_evaluation_dataset_generation_framework.md)**

:   RAGEval 提出了一个基于 schema 的自动化评估数据集生成框架，能够针对不同垂直领域（金融、法律、医疗等）自动生成高质量的文档-问题-答案-参考四元组，并引入完整性（Completeness）、幻觉（Hallucination）和无关性（Irrelevance）三个新评估指标来严格评估 RAG 系统的事实准确性。

**[RARE: Retrieval-Augmented Reasoning Enhancement for Large Language Models](information_retrieval/rare_retrieval_augmented_reasoning.md)**

:   提出 RARE，在 rStar 的 MCTS 推理框架中引入两个检索增强动作（A6: 基于原始问题生成搜索查询并检索，A7: 对子问题进行检索并重新回答），并用检索增强的事实性评分器（RAFS）替代原始判别器，使 LLaMA 3.1 在医学和常识推理任务上达到甚至超越 GPT-4o 的水平。

**[REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark](information_retrieval/real-mm-rag_a_real-world_multi-modal_retrieval_benchmark.md)**

:   提出 REAL-MM-RAG 多模态文档检索基准，定义了真实世界检索基准的四大关键属性（多模态文档、增强难度、真实 RAG 查询、准确标注），引入多级查询改写鲁棒性评估，并通过针对性训练集（改写数据集+金融表格数据集）实现 SOTA 检索性能。

**[Redundancy, Isotropy and Intrinsic Dimensionality of Prompt-Based Text Embeddings](information_retrieval/redundancy_isotropy_and_intrinsic_dimensionality_of_prompt-based_text_embeddings.md)**

:   系统研究了基于Prompt的文本嵌入模型（如gte-Qwen2、E5-mistral等）在后处理降维下的性能鲁棒性，发现分类/聚类任务仅保留原始维度的0.5%即可基本保持性能，并通过内在维度（ID）和各向同性（IsoScore）两个指标定量解释了不同任务Prompt产生的嵌入冗余度差异。

**[REFIND at SemEval-2025 Task 3: Retrieval-Augmented Factuality Hallucination Detection in Large Language Models](information_retrieval/refind_at_semeval-2025_task_3_retrieval-augmented_factuality_hallucination_detec.md)**

:   提出 REFIND 框架，通过计算每个 token 在有无检索文档条件下的生成概率之比（Context Sensitivity Ratio, CSR），实现对 LLM 输出中幻觉片段的高效检测，在 SemEval-2025 Task 3 的 9 种语言上显著超越基线。

**[Removal of Hallucination on Hallucination: Debate-Augmented RAG](information_retrieval/removal_of_hallucination_on_hallucination_debate-augmented_rag.md)**

:   DRAG（Debate-Augmented RAG）提出在RAG系统的检索和生成两个阶段均引入多智能体辩论（MAD）机制，通过正反方辩论+裁判仲裁的结构化流程，消除因错误检索导致的"幻觉传递幻觉"问题，在6个QA基准上显著提升事实准确性。

**[Reranking-based Generation for Unbiased Perspective Summarization](information_retrieval/reranking-based_generation_for_unbiased_perspective_summarization.md)**

:   针对政治视角摘要任务，构建了受控测试集验证现有评估指标的可靠性，发现 LLM-based 指标远优于传统指标，并证明基于重排序（Reranking）的方法及在重排序数据上的 DPO 训练能显著提升摘要的覆盖性和忠实性。

**[SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model](information_retrieval/saferag_benchmarking_security_in_retrieval-augmented_generation_of_large_languag.md)**

:   提出首个中文 RAG 安全评估基准 SafeRAG，设计四种能绕过现有检索器、过滤器和生成器防御的新型攻击任务（银噪声、上下文间冲突、软广告、白色拒绝服务），在 14 种 RAG 组件上系统评估安全漏洞，揭示即使最先进的 RAG 系统也对这些攻击高度脆弱。

**[SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](information_retrieval/seakr_self-aware_knowledge_retrieval_for_adaptive_retrieval_augmented_generation.md)**

:   SeaKR 利用 LLM 内部隐藏层的自感知不确定性（通过多次采样 EOS token 隐藏表示的 Gram 行列式度量）来自适应地决定何时检索、如何重排检索结果、以及选择何种推理策略，在复合 QA 上 F1 比 DRAGIN 提升 6%，比 IRCoT 提升 9.5%。

**[Semantic Outlier Removal with Embedding Models and LLMs](information_retrieval/semantic_outlier_removal_with_embedding_models_and_llms.md)**

:   提出 SORE（Semantic Outlier Removal），一种基于多语言句子嵌入和近似近邻搜索的文本清洗方法，通过元数据嵌入识别核心内容，并标记与预定义离群类别匹配或严重偏离核心内容的文本片段，在接近 LLM 水平的精度下实现了极低的计算成本，已在生产环境中每天处理数百万文档。

**[SetR: Shifting from Ranking to Set Selection for Retrieval Augmented Generation](information_retrieval/setr_set_selection_rag.md)**

:   提出 SetR，将 RAG 中的文档排序范式转变为集合选择范式，通过 CoT 推理识别查询的信息需求并选择最优文档集合，在使用更少文档（平均 2.91 个 vs 5 个）的同时显著提升多跳问答性能。

**[SGIC: A Self-Guided Iterative Calibration Framework for RAG](information_retrieval/sgic_a_self-guided_iterative_calibration_framework_for_rag.md)**

:   SGIC 利用 LLM 的 token 级不确定性分数（文档相关性不确定性 + 答案置信度不确定性）作为自校准的引导信号，通过迭代将前一轮答案及其不确定性分数注入提示中触发上下文推理，在 HotpotQA 上将 Llama2-7B 的 EM 从 69.1% 提升到 77.2%（+8.1%），对 GPT-4o 也有 2.8% 的提升。

**[Sticking to the Mean: Detecting Sticky Tokens in Text Embedding Models](information_retrieval/sticking_to_the_mean_detecting_sticky_tokens_in_text_embedding_models.md)**

:   本文系统研究了文本嵌入模型中的"粘性token"现象——某些异常token被反复插入句子后会将句子相似度拉向固定值，提出了高效检测方法 STD 并在 14 个模型家族的 40 个检查点中发现 868 个粘性token，揭示了高达 50% 的下游任务性能下降。

**[The Distracting Effect: Understanding Irrelevant Passages in RAG](information_retrieval/the_distracting_effect_understanding_irrelevant_passages_in_rag.md)**

:   本文提出了一个形式化的段落干扰效应（Distracting Effect）度量方法，并开发了多种获取高干扰段落的技术（包括偏斜检索和分类合成），证明了该度量跨LLM的鲁棒性，最终通过用高干扰段落微调LLM，在问答准确率上实现了最高7.5%的提升。

**[Toward Structured Knowledge Reasoning: Contrastive Retrieval-Augmented Generation on Experience](information_retrieval/toward_structured_knowledge_reasoning_contrastive_retrieval-augmented_generation.md)**

:   本文提出 CoRE 框架，通过蒙特卡洛树搜索（MCTS）构建包含成功和失败经验的记忆库，并在推理时通过对比式上下文学习（Contrastive ICL）检索正负样例来增强 LLM 对结构化数据（表格、数据库）的推理能力，在 Text-to-SQL 和 TableQA 上分别平均提升 3.44% 和 4.24%。

**[Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](information_retrieval/towards_adaptive_memory-based_optimization_for_enhanced_retrieval-augmented_gene.md)**

:   提出 Amber 框架，通过 Agent 协作式记忆更新器、自适应信息收集器和多粒度内容过滤器三个组件协同工作，在迭代式 RAG 范式中提升开放域问答的检索效率和答案质量。

**[Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings](information_retrieval/towards_storage-efficient_visual_document_retrieval_an_empirical_study_on_reduci.md)**

:   系统性研究了视觉文档检索（VDR）中 patch 级别嵌入的压缩策略，发现 pruning 在 VDR 中本质不适用（简单随机剪枝反而最优），而 token merging 结合微调可在仅保留 2.8% 存储量时维持 94.6% 的检索性能（Light-ColPali/ColQwen2）。

**[Typed-RAG: Type-Aware Decomposition of Non-Factoid Questions for Retrieval-Augmented Generation](information_retrieval/typed-rag_type-aware_decomposition_of_non-factoid_questions_for_retrieval-augmen.md)**

:   提出 Typed-RAG 框架，通过对非事实性问题（NFQ）进行类型感知的分解，将复杂的多方面问题拆解为单方面子查询，针对不同问题类型（辩论、经验、比较等）设计差异化的检索与生成策略，显著提升了 RAG 在 NFQA 中的表现。

**[Unanswerability Evaluation for Retrieval Augmented Generation](information_retrieval/unanswerability_evaluation_for_retrieval_augmented_generation.md)**

:   UAEval4RAG 提出了一个针对 RAG 系统处理不可回答查询的全面评估框架，定义了六类不可回答类别，能够基于任意知识库自动合成测试数据并评估系统的拒绝能力，实验揭示没有单一配置能在所有数据集上同时最优化可回答和不可回答查询的表现。

**[VISA: Retrieval Augmented Generation with Visual Source Attribution](information_retrieval/visa_retrieval_augmented_generation_with_visual_source_attribution.md)**

:   VISA 提出了一种基于视觉来源归因的 RAG 方法，利用大型视觉语言模型（VLM）在检索到的文档截图中用 bounding box 高亮支持生成答案的精确区域，并构建了 Wiki-VISA 和 Paper-VISA 两个数据集验证其有效性。

**[VoxRAG: A Step Toward Transcription-Free RAG Systems in Spoken Question Answering](information_retrieval/voxrag_a_step_toward_transcription-free_rag_systems_in_spoken_question_answering.md)**

:   提出 VoxRAG，一个模块化的语音到语音检索增强生成系统，使用 CLAP 音频嵌入绕过转录直接从语音查询检索语义相关的音频片段，在播客问答场景中验证了无转录语音检索的可行性，Recall@10 在 somewhat relevant 片段上达到 0.60。

**[When Claims Evolve: Evaluating and Enhancing the Robustness of Embedding Models Against Misinformation Edits](information_retrieval/when_claims_evolve_evaluating_and_enhancing_the_robustness_of_embedding_models_a.md)**

:   提出一个扰动框架来系统评估句子嵌入模型在处理经过编辑的虚假信息声明时的鲁棒性，发现标准嵌入模型性能显著下降，并通过知识蒸馏和声明规范化两种缓解方法将领域内鲁棒性提升最高 17 个百分点、跨域泛化提升 10 个百分点。

**[When Should Dense Retrievers Be Updated in Evolving Corpora? Detecting Out-of-Distribution Corpora Using GradNormIR](information_retrieval/when_should_dense_retrievers_be_updated_in_evolving_corpora_detecting_out-of-dis.md)**

:   提出GradNormIR方法，利用梯度范数在无需查询的情况下无监督检测语料库是否对dense retriever构成分布外(OOD)，从而判断何时需要更新检索器，保障动态语料库场景下的检索鲁棒性。

---

## ⚖️ 对齐 / RLHF { #llm_alignment }

**[AGD: Adversarial Game Defense Against Jailbreak Attacks in Large Language Models](llm_alignment/agd_adversarial_game_defense_against_jailbreak_attacks_in_large_language_models.md)**

:   本文提出AGD（Adversarial Game Defense），一种基于对抗博弈的LLM越狱攻击防御方法，通过动态调整模型内部表示在有用性和无害性之间取得平衡，利用IQR异常检测、双层优化博弈和专家模型采样三个阶段显著提升LLM安全性。

**[AgentAlign: Navigating Safety Alignment in the Shift from Informative to Agentic LLMs](llm_alignment/agentalign_navigating_safety_alignment_in_the_shift_from_informative_to_agentic_.md)**

:   本文提出 AgentAlign 框架，利用抽象行为链作为中介，在模拟环境中合成高质量的 agent 安全对齐数据（有害+良性），通过 SFT 使三类开源模型的 agent 安全性提升35.8%-79.5%，同时保持甚至提升了任务能力。

**[AgentRM: Enhancing Agent Generalization with Reward Modeling](llm_alignment/agentrm_enhancing_agent_generalization_with_reward_modeling.md)**

:   提出 AgentRM，一个可泛化的奖励模型，通过显式/隐式/LLM-as-Judge 三种方式构建，用测试时搜索（Best-of-N / Beam Search）引导策略模型，在 9 个 Agent 任务上平均提升 8.8 分并超越最佳通用 Agent 4.0 分。

**[Aligning to What? Limits to RLHF Based Alignment](llm_alignment/aligning_to_what_limits_to_rlhf_based_alignment.md)**

:   本文通过系统实验发现RLHF（包括DPO、ORPO、RLOO等方法）在减少LLM隐性种族偏见方面基本无效，且SFT在RLHF之前进行会"固化"模型偏见，揭示了当前对齐技术在处理模糊目标（如消除偏见）方面的根本局限。

**[AMoPO: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models](llm_alignment/amopo_adaptive_multi-objective_preference_optimization_without_reward_models_and.md)**

:   提出AMoPO框架，通过将生成空间建模为高斯分布实现维度感知的自适应权重分配，在不依赖奖励模型和参考模型的情况下完成多目标偏好对齐，在HelpSteer2数据集上超越SOTA 28.5%，并在7B/14B/32B模型上验证了缩放能力。

**[ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning](llm_alignment/aspo_adaptive_sentence-level_preference_optimization_for_fine-grained_multimodal.md)**

:   将 DPO 的偏好优化粒度从回复级细化到句子级，通过图文相似度和文本困惑度两个维度动态计算每个句子的自适应奖励权重，在 LLaVA-1.5-7B/13B 和 InstructBLIP-13B 上分别带来平均 2.57/2.87/1.98 分提升，同时显著降低幻觉率。

**[Atyaephyra at SemEval-2025 Task 4: Low-Rank Negative Preference Optimization](llm_alignment/atyaephyra_at_semeval-2025_task_4_low-rank_negative_preference_optimization.md)**

:   在 SemEval 2025 LLM 遗忘共享任务中，将负偏好优化 (NPO) 与低秩适配 (LoRA) 结合，利用 LoRA 的结构特性零开销获取原始模型分布来计算 KL 散度正则化，显著稳定了遗忘过程并超越了任务基线。

**[AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs](llm_alignment/automixalign_adaptive_data_mixing.md)**

:   AutoMixAlign 提出了一种理论驱动的多任务偏好优化数据混合方法：先训练各任务的 specialist model 确定最优 loss 基线，再通过 minimax 优化自适应调整数据混合比例，优先处理 excess loss（与 specialist 的差距）最大的任务，在 helpfulness/harmlessness/reasoning 多任务 DPO 中平均提升 9.42%。

**[Beyond Similarity: A Gradient-based Graph Method for Instruction Tuning Data Selection](llm_alignment/beyond_similarity_a_gradient-based_graph_method_for_instruction_tuning_data_sele.md)**

:   本文提出 G2IS（Gradient-based Graph Instruction Selection），通过构建基于梯度的指令图来建模指令数据之间的联合分布和相互依赖关系，结合梯度遍历算法进行数据选择，仅用 1% 的数据即可超越全数据指令调优的效果。

**[Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs](llm_alignment/beyond_surface-level_patterns_an_essence-driven_defense_framework_against_jailbr.md)**

:   提出 EDDF，一种基于"攻击本质"而非表面模式的越狱防御框架：离线提取已知攻击的本质策略存入向量数据库，在线时对新查询做本质抽象+检索+细粒度判断，将攻击成功率降低至少 20% 且误报率仅 2.18%。

**[Beyond the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models](llm_alignment/beyond_the_tip_of_efficiency_uncovering_the_submerged_threats_of_jailbreak_attac.md)**

:   系统评估 13 个 SOTA 小语言模型（<4B参数）在 5 种越狱攻击下的安全性，发现 SLM 虽能抵御直接攻击但在越狱攻击下显著比大模型脆弱，进一步分析了架构压缩、量化和知识蒸馏等 SLM 技术对安全性的影响。

**[Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data](llm_alignment/boosting_vulnerability_detection_of_llms_via_curriculum_preference_optimization_.md)**

:   提出 ReVD 框架，通过双向漏洞推理数据合成 + 三元组 SFT（同时学习漏洞代码/修复代码/代码差异的推理）+ 课程化在线偏好优化（COPO），将 LLM 的漏洞检测准确率提升 12-23%，在 PrimeVul 和 SVEN 上达到 SOTA。

**[Breaking the Ceiling: Exploring the Potential of Jailbreak Attacks through Expanding Strategy Space](llm_alignment/breaking_the_ceiling_exploring_the_potential_of_jailbreak_attacks_through_expand.md)**

:   基于精细化可能性模型 (ELM) 将越狱策略分解为四类可独立进化的组件（角色/内容支撑/语境/沟通技巧），提出 CL-GSO 遗传算法在组件级进行交叉与变异，将策略空间从既有方法的 40 种扩展到 839 种，在 Claude-3.5 上实现 96% 攻击成功率（此前方法最高仅 4%），同时提出基于意图一致性的评估机制，准确率达 96.5% 超越专用安全模型。

**[Call for Rigor in Reporting Quality of Instruction Tuning Data](llm_alignment/call_for_rigor_in_reporting_quality_of_instruction_tuning_data.md)**

:   通过系统性的 16 种超参数组合实验，揭示了指令微调数据质量评估中的严重问题——研究者对训练超参数的任意选择可以导致完全相反的「数据 A 优于数据 B」的结论，呼吁在报告数据质量时必须采用经过验证的超参数设置。

**[Chain-of-Jailbreak Attack for Image Generation Models via Editing Step by Step](llm_alignment/chain-of-jailbreak_attack_for_image_generation_models_via_editing_step_by_step.md)**

:   提出 Chain-of-Jailbreak（CoJ）攻击，将无法直接绕过安全护栏的恶意 query 分解为多步编辑子 query（删然后插、插然后删、改然后改回），在 GPT-4V/4o/Gemini 上达到 60%+ 越狱成功率；同时提出 Think-Twice Prompting 防御，拦截 95%+ 的 CoJ 攻击。

**[Cheems: A Practical Guidance for Building and Evaluating Chinese Reward Models from Scratch](llm_alignment/cheems_chinese_reward_models.md)**

:   为弥补中文 Reward Model 资源的空白，本文构建了 CheemsBench（首个大规模中文 RM 评测基准）和 CheemsPreference（首个大规模中文偏好数据集），通过人机协作标注 + 远程监督过滤策略训练的 CheemsRM 在中文场景显著超越现有所有开源 RM。

**[Constitutional Classifiers: Defending Against Universal Jailbreaks Across Thousands of Hours of Red Teaming](llm_alignment/constitutional_classifiers_defending_against_universal_jailbreaks_across_thousan.md)**

:   Anthropic提出"宪法分类器"（Constitutional Classifiers），通过从自然语言安全规则（宪法）出发生成合成训练数据来训练输入/输出安全分类器，在超过3000小时的红队测试中未被发现通用越狱攻击，同时仅增加0.38%的误拒率和23.7%的推理开销。

**[Curiosity-Driven Reinforcement Learning from Human Feedback](llm_alignment/curiosity_driven_rlhf.md)**

:   CD-RLHF 将好奇心驱动探索（curiosity-driven RL）引入 RLHF，通过前向动力学模型的预测误差作为内在奖励，结合 top-k 门控过滤与 reward whitening，在不损失对齐质量的前提下大幅提升 LLM 输出多样性（Llama-3.2-1B 上 Diversity 提升 40.26%，EAD 提升 8.92%）。

**[Debate, Reflect, and Distill: Multi-Agent Feedback with Tree-Structured Preference Optimization for Efficient Language Model Enhancement](llm_alignment/debate_reflect_and_distill_multi-agent_feedback_with_tree-structured_preference_.md)**

:   提出 D&R 框架，让小模型（student）与多个大模型（teacher）进行多轮辩论并收集自我反思和教师反馈，然后将辩论日志组织为偏好树做 Tree-structured DPO (T-DPO) 蒸馏，在 MMLU Pro 和 MATH 上平均提升 14.18 分，且推理效率优于基线。

**[DiffPO: Diffusion Alignment with Direct Preference Optimization](llm_alignment/diffpo_diffusion_alignment.md)**

:   提出 DiffPO，将 LLM 对齐重新建模为句子级扩散去噪过程，通过 parallel decoding 实现高效推理时对齐，作为即插即用模块可增强任意底座模型的对齐质量。

**[Don't Say No: Jailbreaking LLM by Suppressing Refusal](llm_alignment/dont_say_no_jailbreaking_llm_by_suppressing_refusal.md)**

:   本文提出 DSN（Don't Say No）攻击方法，通过分析现有越狱攻击中目标损失函数的缺陷，引入余弦衰减调度和拒绝抑制两种改进策略，在多个 LLM 上实现了超越现有方法的攻击成功率（ASR），并展示了对未见数据集和黑盒模型的强迁移性。

**[Dynamic Scaling of Unit Tests for Code Reward Modeling](llm_alignment/dynamic_scaling_of_unit_tests_for_code_reward_modeling.md)**

:   本文发现扩展LLM生成的单元测试数量可以持续提升代码奖励信号质量（尤其对困难问题效果更好），据此训练了轻量级单元测试生成模型CodeRM-8B并实现动态缩放策略，在多个代码生成基准上取得显著提升。

**[Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent](llm_alignment/expectation_confirmation_preference_optimization_for_multi-turn_conversational_r.md)**

:   提出 ECPO（Expectation Confirmation Preference Optimization），首个面向 LLM 对话推荐 Agent 的多轮偏好优化方法——基于心理学期望确认理论（ECT）显式建模用户满意度在多轮对话中的演变，通过前向期望确认定位不满意根因 + 后向期望推导重写回复构建 turn-level 偏好对，配合 AILO 用户模拟器，在 3 个数据集上显著优于现有 MTPO 方法。

**[Federated Data-Efficient Instruction Tuning for Large Language Models](llm_alignment/federated_data-efficient_instruction_tuning_for_large_language_models.md)**

:   提出 FedHDS（Federated Hierarchical Data Selection），通过 intra-client 和 inter-client 两级层次化数据选择消除联邦学习中客户端内部和跨客户端的数据冗余，结合多层 Transformer 特征融合提升 coreset 质量；仅用不到 1.5% 的数据，在 Rouge-L 上相对 SOTA 全数据联邦基线平均提升 10.72%，训练效率提升最高达 48.8 倍。

**[Finding the Sweet Spot: Preference Data Construction for Scaling Preference Optimization](llm_alignment/finding_the_sweet_spot_preference_data_construction_for_scaling_preference_optim.md)**

:   发现传统的 DPO 偏好数据构建策略（max-min）在增加采样量时性能反而下降，通过基于奖励分布的系统性探索发现 rejected 响应应选在 μ−2σ 而非最小值，据此提出了一种随采样量增加而持续提升的偏好数据构建方法。

**[Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization](llm_alignment/fine-grained_video_dubbing_duration_alignment_with_segment_supervised_preference.md)**

:   提出 Segment Supervised Preference Optimization (SSPO)，将视频配音中译文与源语音的时长对齐问题建模为段级偏好优化，通过逐句采样+细粒度 DPO 损失实现每行对话的时长一致性，同时维持翻译质量和输出格式。

**[FocalPO: Enhancing Preference Optimizing by Focusing on Correct Preference Rankings](llm_alignment/focalpo_enhancing_preference_optimizing_by_focusing_on_correct_preference_rankin.md)**

:   本文提出 FocalPO，一种 DPO 变体，通过引入受 Focal Loss 启发的调制因子来降低错误排序对的权重、优先强化模型已正确排序的偏好对的理解，在 AlpacaEval 2.0 等基准上超越 DPO 及其变体。

**[Focused-DPO: Enhancing Code Generation Through Focused Preference Optimization on Error-Prone Points](llm_alignment/focused-dpo_enhancing_code_generation_through_focused_preference_optimization_on.md)**

:   发现代码生成模型的错误高度集中在特定"错误易发点"（error-prone points），前缀/后缀几乎不变而中间段决定正确性，提出 Focused-DPO：通过 PageRank 在代码-测试二部图上排序定位关键中间段，并在 DPO 损失中对该段加权放大（$w_{focused}=2$），仅用 5000 样本即可在 HumanEval+ 上提升 4.41%、LiveCodeBench-Hard 上相对提升 42.86%。

**[HAF-RM: A Hybrid Alignment Framework for Reward Model Training](llm_alignment/haf-rm_a_hybrid_alignment_framework_for_reward_model_training.md)**

:   提出混合对齐框架 HaF-RM，在奖励模型训练中保留策略层（policy layer），通过同时优化序列级奖励损失和 token 级策略损失来共同监督共享的内部偏好模型，在 5 个数据集上一致性超越标准 Baseline 和 DPO 方法。

**[HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States](llm_alignment/hiddendetect_detecting_jailbreak_attacks_against_multimodal_large_language_model.md)**

:   提出 HiddenDetect，一种免训练（tuning-free）的基于内部激活状态的安全检测框架：通过监控 LVLM 推理时隐藏状态中的拒绝语义信号来检测越狱攻击，在多个模型和多模态基准上 AUROC 大幅超越现有方法。

**[Understanding Impact of Human Feedback via Influence Functions](llm_alignment/influence_functions_rlhf.md)**

:   首次将影响函数应用于 RLHF 奖励模型的反馈数据审计，结合 OPORP 向量压缩实现 2.5 倍加速，在偏差检测上超越 GPT-4o（AUC 0.8 vs 0.747），并从 Anthropic-HH 数据集中发现 47% 的错标样本。

**[Internal Value Alignment in Large Language Models through Controlled Value Vector Activation](llm_alignment/internal_value_alignment_in_large_language_models_through_controlled_value_vecto.md)**

:   提出 ConVA（Controlled Value Vector Activation）框架，通过上下文控制的数据集精准识别 LLM 隐空间中的价值向量，并用门控最小扰动机制在推理时激活目标价值，在 Schwartz 10 种基本价值上实现平均 29.6% 的控制成功率提升，同时保持 97%+ 的文本流畅度和通用能力。

**[Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process](llm_alignment/intuitive_fine_tuning_simplifying_alignment_into_single_process.md)**

:   通过 MDP 框架统一分析 SFT 和偏好优化（PO），发现 SFT 是 PO 的特例但偏好估计和转移优化均不充分，提出 IFT（Intuitive Fine-Tuning）通过时序残差连接让模型在不需要偏好数据的情况下获得接近甚至超越 SFT+PO 流水线的对齐效果。

**[IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization](llm_alignment/iopo_input_output_preference.md)**

:   提出 IOPO（Input-Output Preference Optimization），在传统 DPO 仅优化输出偏好的基础上，引入输入偏好建模——让模型学习"给定回复 y，哪个指令 x 更匹配"，从而增强对复杂多约束指令的细粒度感知能力；同时构建了包含 120K 训练数据、1K 评测数据、覆盖 5 大类 26 个约束维度的 Trace 基准。

**[Jailbreaking? One Step Is Enough!](llm_alignment/jailbreaking_one_step_is_enough.md)**

:   本文提出REDA（Reverse Embedded Defense Attack）方法，将攻击意图伪装为"防御"有害内容的任务，通过反转攻击视角+ICL示例引导+请求意图削弱，实现一步生成、跨模型通用的高成功率越狱攻击。

**[JailbreakRadar: Comprehensive Assessment of Jailbreak Attacks Against LLMs](llm_alignment/jailbreakradar_comprehensive_assessment_jailbreak_attacks.md)**

:   首个覆盖自动和非自动越狱攻击的统一全面评估框架：收集17种代表性越狱攻击，建立六类攻击分类体系，在9个对齐LLM×8种防御策略下进行大规模系统评测，揭示启发式攻击"高ASR但低实用性"的关键洞察。

**[JsonTuning: Towards Generalizable, Robust, and Controllable Instruction Tuning](llm_alignment/jsontuning_towards_generalizable_robust_and_controllable_instruction_tuning.md)**

:   提出 JsonTuning——将指令微调的输入输出从自然语言文本替换为 JSON 结构化格式，通过显式表示任务元素、关系和输出约束（JSON Schema），在 7 个预训练模型和 6 类任务上一致超越传统 TextTuning，平均性能从 26.78 提升到 30.88，同时显著增强鲁棒性和可控性。

**[LLMs Caught in the Crossfire: Malware Requests and Jailbreak Challenges](llm_alignment/llms_caught_in_the_crossfire_malware_requests_and_jailbreak_challenges.md)**

:   构建 MalwareBench 基准（320 个手工恶意代码需求 × 11 种黑盒越狱方法 = 3520 个 prompt），系统评测 29 个 LLM 在恶意代码生成场景下的安全性，发现越狱攻击将平均拒绝率从 60.93% 降至 39.92%，且模型参数量与防御能力并非正比关系。

**[LPOI: Listwise Preference Optimization for Vision Language Models](llm_alignment/lpoi_listwise_preference_optimization_for_vision_language_models.md)**

:   本文提出 LPOI，首个面向 VLM 的目标感知列表级偏好优化方法——通过识别并遮挡图像中的关键目标，在正样本和负样本之间插值生成渐进式遮挡序列，训练模型按目标可见度排序，从而在无需额外标注的情况下有效降低幻觉，在 MMHalBench、AMBER 和 Object HalBench 上超越现有偏好优化方法。

**[LSSF: Safety Alignment via Low-Rank Safety Subspace Fusion](llm_alignment/lssf_safety_subspace.md)**

:   LSSF 提出 LLM 的安全信息存在于低秩子空间中的假设，通过 SVD 提取安全对齐模型的主成分，利用安全奇异值熵自适应确定每层的保留秩，最终将提取的安全主成分线性融合到微调后的模型中，无需额外训练即可恢复因微调而退化的安全对齐，同时保持下游任务性能。

**[M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs](llm_alignment/m2s_multiturn_to_singleturn_jailbreak_in.md)**

:   提出 M2S 框架，通过三种简单的格式转换方法（Hyphenize/Numberize/Pythonize）将多轮人类越狱对话压缩为单轮 prompt，不仅保持甚至超越原始多轮攻击效果（ASR 高达 95.9%，比多轮提升最多 17.5%），同时 token 使用量减半以上。

**[Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric](llm_alignment/measuring_data_diversity_for_instruction_tuning_a_systematic_analysis_and_a_reli.md)**

:   系统分析 11 种现有多样性度量方法的局限性，提出 NovelSum——一种同时考虑样本间差异和信息密度的数据多样性指标，与指令微调性能达到 0.97 相关性。

**[MPO: Multilingual Safety Alignment via Reward Gap Optimization](llm_alignment/mpo_multilingual_safety_alignment.md)**

:   MPO 发现 LLM 在主导语言（英文）和目标语言间的隐式 Reward Gap 与安全性能强相关，提出直接最小化两者 Reward Gap 差异来将主导语言的安全对齐能力迁移到多语言，在三个模型上显著降低了低资源语言的攻击成功率且不损害通用能力。

**[MTSA: Multi-Turn Safety Alignment for LLMs through Multi-Round Red-Teaming](llm_alignment/mtsa_multi-turn_safety_alignment_for_llms_through_multi-round_red-teaming.md)**

:   提出MTSA框架，通过思维引导的多轮红队攻击学习和基于未来奖励的多轮强化学习算法，在对抗迭代优化中同时提升红队模型的攻击能力和目标模型的安全防御能力，在多个安全基准上达到SOTA，且不损失模型通用性能。

**[Mutual-Taught for Co-adapting Policy and Reward Models](llm_alignment/mutual_taught_policy_reward.md)**

:   Mutual-Taught 提出了一种基于 EM 算法的自训练框架，在偏好优化过程中同时迭代更新 policy model 和 reward model：E-step 用当前 RM 优化 PM，M-step 用 PM 更新前后的输出差异构建伪偏好对来更新 RM，解决了分布偏移导致的 reward hacking 问题，8B 模型在 AlpacaEval-2 达到 54.1% LC win rate。

**[Optimal Transport-Based Token Weighting for Enhanced Preference Optimization](llm_alignment/otpo_token_weighting.md)**

:   OTPO 利用无平衡最优传输（UOT）在 chosen/rejected 回复的 token 表示之间计算语义对齐权重，使偏好优化聚焦于关键差异 token 而非均等对待所有 token，在 AlpacaEval2 上将 DPO 的 LC WR 从 48.14% 提升至 55.84%，并将 DPO/SimPO/SamPO/LDDPO 统一为 token 加权的特例。

**[Whose Boat Does it Float? Improving Personalization in Preference Tuning via Inferred User Personas](llm_alignment/personalized_preference_opt.md)**

:   提出基于**溯因推理（abductive reasoning）**的偏好个性化框架：通过 **Persona Inference (PI)** 推断偏好数据 chosen/rejected 回答背后的用户画像，再用画像增强的偏好数据进行 **Persona Tailoring (PT)** 训练，使 LLM 能根据用户画像生成个性化回答，在对话、问答、教育三个领域均大幅提升个性化适配能力。

**[PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference](llm_alignment/pku-saferlhf_towards_multi-level_safety_alignment_for_llms_with_human_preference.md)**

:   发布 PKU-SafeRLHF 大规模安全偏好数据集，包含 44.6k 精炼 prompt、265k 带安全元标签的 QA 对和 166.8k 偏好数据，首次引入 19 种危害类别和 3 级严重程度标注，并训练了严重程度敏感的审核模型（93% 准确率）和基于该数据的 SafeRLHF 对齐 pipeline。

**[PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models](llm_alignment/prmbench_a_fine-grained_and_challenging_benchmark_for_process-level_reward_model.md)**

:   本文提出PRMBench，一个包含6,216个精心设计问题和83,456个步骤级标签的基准，从简洁性（Simplicity）、合理性（Soundness）和敏感性（Sensitivity）三个维度系统评估过程级奖励模型（PRM）的细粒度错误检测能力，实验揭示了现有15个PRM的显著不足。

**[Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](llm_alignment/probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)**

:   > PCPO 在偏好对选择阶段引入 token 级概率一致性指标，选出答案正确且推理过程与错误回答最"相似"的配对进行 DPO 训练，让模型聚焦关键推理差异，在多个数学推理 benchmark 上一致超越 IRPO/ScPO。

**[QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language](llm_alignment/queryattack_jailbreaking_aligned_large_language_models_using_structured_non-natu.md)**

:   提出 QueryAttack，将恶意自然语言查询分解为三个语义组件（内容、修饰符、类别）并填入编程语言模板（SQL/URL/Python/Java/C++ 等 9 种），结合 ICL 引导目标 LLM 直接用自然语言回复有害内容，无需解密步骤，在 GPT-4o 上 Ensemble 配置达到 96.35% ASR，且提出的跨语言 CoT 防御可将 ASR 降低最多 64%。

**[Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking](llm_alignment/red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)**

:   提出 Red Queen Attack——首个基于 Theory of Mind（ToM）构建多轮对话场景并隐藏恶意意图的越狱攻击方法，生成 56K 多轮隐蔽攻击数据，在 GPT-4o 上达到 87.6% ASR；同时提出 Red Queen Guard 防御策略，通过多轮 DPO 数据训练将 ASR 降至 <1%，同时不影响通用基准性能。

**[Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization](llm_alignment/rethinking_reward_model_evaluation_through_the_lens_of_reward_overoptimization.md)**

:   本文从奖励过优化（reward overoptimization）的视角重新审视奖励模型评估方法，发现现有基准与下游策略性能相关性弱，并提出了三条构建可靠评估基准的关键准则：最小化正负样本的非正确性差异、使用多次比较覆盖广泛响应范围、以及从多样模型中采样响应。

**[Rethinking Table Instruction Tuning](llm_alignment/rethinking_table_instruction_tuning.md)**

:   系统消融表格指令微调中被忽视的超参数选择（学习率、数据量、epoch），揭示现有表格 LLM 因学习率过大（2e-5）导致通用能力严重退化（MMLU 降 14 分、AI2ARC 降 21 分），提出仅需 13 个数据集各 200 条（共 2600 条）+ 学习率 1e-6 + 2 epoch 微调 LLaMA 3.1 8B Instruct 即可构建 TAMA，在 13 个表格任务上匹配/超越 GPT-3.5 和 GPT-4，同时完整保持通用能力。

**[Retrieval-Augmented Fine-Tuning With Preference Optimization For Visual Program Generation](llm_alignment/retrieval-augmented_fine-tuning_with_preference_optimization_for_visual_program_.md)**

:   本文针对工业视觉编程语言（Ladder Diagram）的自动生成任务，提出了一种两阶段训练策略：先通过检索增强微调（Retrieval-Augmented Fine-Tuning）利用子程序复用特性，再通过基于图编辑操作构造偏好对的 DPO 训练进一步提升准确性，在真实 LD 数据上将程序级准确率提升超过 10%。

**[Reverse Preference Optimization for Complex Instruction Following](llm_alignment/reverse_preference_optimization_for_complex_instruction_following.md)**

:   提出反向偏好优化（RPO），通过动态反转指令中未满足的约束将任意回复转化为"完美"chosen 样本，消除多约束偏好对中的噪声，在多轮复杂指令遵循任务上显著超越 DPO 基线。

**[Towards Reward Fairness in RLHF: From a Resource Allocation Perspective](llm_alignment/reward_fairness_rlhf.md)**

:   将 RLHF 中的长度偏差、类别偏差、社会偏差等多种奖励偏差统一定义为"奖励不公平"问题，借鉴资源分配理论提出 Fairness Regularization 和 Fairness Coefficient 两种偏差无关方法，分别应用于奖励模型训练和策略模型训练，在不针对特定偏差设计的前提下同时缓解多种偏差并提升对齐质量。

**[Reward Generalization in RLHF: A Topological Perspective](llm_alignment/reward_generalization_in_rlhf_a_topological_perspective.md)**

:   从信息拓扑的角度系统刻画 RLHF 中 reward 信息的流动——宏观层面将 RLHF 建模为自编码过程，微观层面提出 Induced Bayesian Network (IBN) 分析偏好数据拓扑对 reward 泛化的影响，进而提出树结构偏好数据方法，在 HH-RLHF/GSM-8K/DialogSum 三个任务上平均 65% win rate 超越链式 baseline。

**[Rewrite to Jailbreak: Discover Learnable and Transferable Implicit Harmfulness Instruction](llm_alignment/rewrite_to_jailbreak_discover_learnable_and_transferable_implicit_harmfulness_in.md)**

:   提出 R2J（Rewrite to Jailbreak），一种可学习、可迁移的黑盒越狱方法——通过迭代训练 attacker LLM 学习改写有害指令（仅改措辞不改意图），相比 GCG/AutoDAN 等方法攻击成功率提高 20%+，且无额外前缀/后缀，更隐蔽且跨模型可迁移。

**[RISE: Subtle Errors in Reasoning: Preference Learning via Error-injected Self-editing](llm_alignment/rise_error_inject_preference.md)**

:   RISE 发现 LLM 约 75% 的数学错误是微妙的步内错误（数字替换、操作数交换、步骤遗漏），通过让 LLM 自编辑向正确解注入预定义微妙错误来构造高质量难负样本，配合错误感知 DPO 训练，仅用 4.5K 样本在 GSM8K 提升 3.0%、MATH 提升 7.9%，并泛化到逻辑推理和代码生成。

**[Robust Preference Optimization via Dynamic Target Margins](llm_alignment/robust_preference_optimization_via_dynamic_target_margins.md)**

:   本文提出 γ-PO，一种通过在偏好对级别动态调整目标奖励边际的方式来增强 DPO 鲁棒性的即插即用方法，在 AlpacaEval2 和 Arena-Hard 上平均提升 4.4%。

**[RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation](llm_alignment/rpo_retrieval_preference_optimization_for_robust_retrieval-augmented_generation.md)**

:   提出 Retrieval Preference Optimization (RPO)，一种专为 RAG 设计的轻量级偏好对齐方法，通过将检索质量评估隐式地集成到生成过程中，使 LLM 能够自适应地在参数知识和检索知识之间做出选择，无需额外组件即可缓解知识冲突导致的幻觉问题。

**[Safety Alignment via Constrained Knowledge Unlearning](llm_alignment/safety_alignment_via_constrained_knowledge_unlearning.md)**

:   本文提出 Constrained Knowledge Unlearning (CKU)，通过在 MLP 层中定位有用知识神经元并在遗忘过程中保护其梯度来删除有害知识，在不损害模型通用能力的前提下显著提升 LLM 的安全性。

**[SDPO: Segment-Level Direct Preference Optimization for Social Agents](llm_alignment/sdpo_segment-level_direct_preference_optimization_for_social_agents.md)**

:   SDPO 提出在多轮社交对话中以"片段"（segment）为粒度进行偏好优化，通过动态定位错误轮次、从错误点前的历史重新采样正样本、选取等长的关键片段对进行训练，既降低了会话级 DPO 的训练噪声，又通过等长约束严格消除了分区函数 $Z$，在 SOTOPIA 基准上超越了 GPT-4o 和所有 DPO 变体。

**[SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings](llm_alignment/sea_lowresource_safety_alignment_for_multimodal.md)**

:   提出 SEA 框架，通过梯度优化生成合成模态 embedding（不需要真实图像/视频/音频），仅用文本安全数据就能实现多模态 LLM 的安全对齐，在单张 RTX3090 上 24 秒即可合成高质量 embedding，同时发布了视频和音频安全基准 VA-SafetyBench。

**[SQL Injection Jailbreak: A Structural Disaster of Large Language Models](llm_alignment/sql_injection_jailbreak_a_structural_disaster_of_large_language_models.md)**

:   提出 SQL Injection Jailbreak（SIJ），一种利用 LLM 提示构造方式中的结构性漏洞进行越狱的新方法，在开源模型上实现近 100% 攻击成功率，在闭源模型上平均超过 85%，并提出 Self-Reminder-Key 防御方案。

**[SynthesizeMe! Inducing Persona-Guided Prompts for Personalized Reward Models in LLMs](llm_alignment/synthesizeme_persona_prompts.md)**

:   提出 SynthesizeMe 方法，通过从用户有限的成对偏好交互中自动推理-合成用户画像（persona），构建可解释、可迁移的个性化 prompt，在 PersonalRewardBench 上显著提升个性化偏好预测准确率。

**[T-REG: Preference Optimization with Token-Level Reward Regularization](llm_alignment/t-reg_preference_optimization_with_token-level_reward_regularization.md)**

:   T-REG 提出了一种 token 级奖励正则化方法，利用 LLM 的对比提示自生成 token 级奖励信号，将其作为弱监督来引导 DPO 隐式学习到的 token 级奖励分配，在 Alpaca Eval 2 和 Arena-Hard 上分别超过 DPO 最多 3.8% 和 4.4%。

**[TableDreamer: Progressive and Weakness-Guided Data Synthesis from Scratch for Table Instruction Tuning](llm_alignment/tabledreamer_progressive_and_weakness-guided_data_synthesis_from_scratch_for_tab.md)**

:   提出 TableDreamer 两阶段数据合成框架：第一阶段从零合成多样化表格及种子指令数据，第二阶段通过弱点引导的迭代输入空间探索（在三个方向上演化数据，并用 LLM-as-Judge 筛选模型表现差的样本作为下一轮种子），仅用 27K GPT-4o 合成数据即将 Llama3.1-8B 的平均准确率提升 11.62%，超越使用 80K-100K 数据的所有基线方法。

**[Teaching an Old LLM Secure Coding: Localized Preference Optimization on Distilled Preferences](llm_alignment/teaching_an_old_llm_secure_coding.md)**

:   提出 DiSCo（从前沿 LLM 蒸馏的安全代码偏好数据集，10K 实例覆盖 431 种 CWE）和 LPO（局部偏好优化算法，仅在安全相关 token 上传播损失），在四个安全编码基准上减少 19-40% 的安全问题，同时提升 3-10% 的代码质量。

**[Tempest: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search](llm_alignment/tempest_autonomous_multi-turn_jailbreaking_of_large_language_models_with_tree_se.md)**

:   本文提出 Tempest（论文早期版本称 Siege），一个基于广度优先树搜索的多轮对抗框架，通过追踪目标 LLM 的部分合规信息并将其重新注入后续查询，在 JailbreakBench 上对 GPT-3.5-turbo 达到 100%、GPT-4 达到 97% 的攻击成功率，且需要的查询数远少于 Crescendo/GOAT 等基线。

**[Think&Cite: Improving Attributed Text Generation with Self-Guided Tree Search and Progress Reward Modeling](llm_alignment/think_cite_attributed_text_gen.md)**

:   将归因文本生成（带引用的文本生成）建模为多步推理问题，提出自引导蒙特卡洛树搜索（SG-MCTS）结合进度奖励建模（PRM），通过多路径搜索+中间状态反思+生成/归因双维度进度奖励，在 ALCE 基准三个数据集上显著超越所有基线。

**[A Troublemaker with Contagious Jailbreak Makes Chaos in Honest Towns](llm_alignment/tmcht_contagious_jailbreak_multiagent.md)**

:   提出TMCHT（大规模多智能体多拓扑文本攻击评估框架）和ARCJ（对抗性复制传染越狱）方法——通过优化检索后缀提高毒性样本被检索概率+优化复制后缀使毒性信息具有自我复制传染能力，解决了多智能体系统中单智能体攻击方法面临的"毒性消散"问题。

**[Upcycling Instruction Tuning from Dense to Mixture-of-Experts via Parameter Merging](llm_alignment/upcycling_instruction_tuning_from_dense_to_mixture-of-experts_via_parameter_merg.md)**

:   本文提出UpIT (Upcycling Instruction Tuning)，利用密集模型指令微调过程中的中间checkpoint作为专业化专家，通过遗传算法扩展专家数量和路由预优化，实现数据高效且灵活的dense-to-MoE转换。

**[World Modeling Makes a Better Planner: Dual Preference Optimization for Embodied Task Planning](llm_alignment/world_modeling_makes_a_better_planner_dual_preference_optimization_for_embodied_.md)**

:   提出 Dual Preference Optimization (D²PO) 框架，通过联合优化状态预测（世界建模）和动作选择两个目标的偏好学习，使视觉语言模型在具身任务规划中同时学会"理解世界动态"和"做出更好决策"，7B 模型大幅超越 GPT-4o。

---

## 🔒 LLM 安全 { #llm_safety }

**[A Statistical and Multi-Perspective Revisiting of the Membership Inference Attack in Large Language Models](llm_safety/a_statistical_and_multi-perspective_revisiting_of_the_membership_inference_attac.md)**

:   本文通过数千次实验从统计视角全面重新审视 LLM 中的成员推断攻击（MIA），从数据分割方式、模型规模、领域特性、文本特征、嵌入可分性和解码动态六个维度分析 MIA 性能的不一致性，揭示了阈值泛化、文本长度/相似性影响、嵌入层涌现变化等此前被忽视的发现。

**[AGrail: A Lifelong Agent Guardrail with Effective and Adaptive Safety Detection](llm_safety/agrail_a_lifelong_agent_guardrail_with_effective_and_adaptive_safety_detection.md)**

:   提出 AGrail，一个终身学习的 LLM Agent 安全护栏框架，通过双 LLM 协作（Analyzer + Executor）和记忆模块，在测试时自适应地生成和优化安全检查策略，有效防御任务特定风险和系统性风险。

**[Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filtering](llm_safety/aligning_large_language_models_to_follow_instructions_and_hallucinate_less_via_e.md)**

:   提出NOVA框架，通过内部一致性探测(ICP)衡量LLM对指令的熟悉度+语义等价识别(SEI)衡量LLM对目标回复的熟悉度，筛选出知识对齐的高质量指令数据，仅用5%数据微调LLaMA-3-8B即可在BioGEN上提升8.6分、FollowRAG上提升7.2分，同时保持指令遵循能力。

**[Alleviating Hallucinations from Knowledge Misalignment in Large Language Models via Selective Abstention Learning](llm_safety/alleviating_hallucinations_from_knowledge_misalignment_in_large_language_models_.md)**

:   本文针对LLM中因知识错位（模型参数知识与事实不一致）导致的幻觉问题，提出选择性弃权学习（Selective Abstention Learning）方法，让模型在遇到知识边界外的问题时学会主动拒绝回答而非编造内容，从而减少幻觉。

**[Answer When Needed, Forget When Not: Language Models Pretend to Forget via In-Context Knowledge Unlearning](llm_safety/answer_when_needed_forget_when_not_language_models_pretend_to_forget_via_in-cont.md)**

:   提出"上下文知识遗忘"方法，通过引入特殊的遗忘 token `<<UNL>>...<</UNL>>` 使 LLM 在推理时根据上下文选择性遗忘特定知识，在 TOFU/AGE/RWKU 上达到 95% 遗忘准确率且保留 80% 无关知识，深入的内部分析发现 LLM 并未真正删除知识而是在最后一层"假装遗忘"。

**[Are the Hidden States Hiding Something? Testing the Limits of Factuality-Encoding Capabilities in LLMs](llm_safety/are_the_hidden_states_hiding_something_testing_the_limits_of_factuality-encoding.md)**

:   本文挑战了"LLM隐藏状态能编码事实真假信息"这一先前结论，通过构建更真实、更具挑战性的数据集（基于困惑度的负样本采样和基于QA的LLM生成数据集），发现先前方法在更贴近实际场景的数据上泛化能力有限，为LLM事实性评估研究提供了更严格的评估基准和实践指导。

**[Automated Explanation Generation and Hallucination Detection for Heritage Image Retrieval](llm_safety/automated_explanation_generation_and_hallucination_detection_for_heritage_image_.md)**

:   本文针对文化遗产图像检索任务，提出了一个结合自动解释生成和幻觉检测的框架，利用视觉语言模型为检索结果生成可解释的文本描述，同时通过领域知识约束的幻觉检测机制确保描述的事实准确性，在多个文化遗产数据集上验证了方法的有效性。

**[Bias in the Mirror: Are LLMs' Opinions Robust to Their Own Adversarial Attacks](llm_safety/bias_in_the_mirror_are_llms_opinions_robust_to_their_own_adversarial_attacks.md)**

:   本文提出一种新颖的"自辩论"范式，让同一个LLM的两个实例分别扮演正方和反方进行辩论，试图说服一个中立版本的模型，以此评估LLM内在偏见的鲁棒性——偏见是否容易被动摇，以及模型是否容易被自身的对抗性论证带偏。

**[CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks](llm_safety/cavgan_unifying_jailbreak_and_defense_of_llms_via_generative_adversarial_attacks.md)**

:   提出 CAVGAN 框架，利用生成对抗网络在 LLM 内部表示空间中同时学习越狱攻击（生成器）和安全防御（判别器），首次将攻防统一到同一框架中实现"攻防共进"，攻击成功率平均 88.85%，防御成功率平均 84.17%。

**[Chinese SimpleQA: A Chinese Factuality Evaluation for Large Language Models](llm_safety/chinese_simpleqa_a_chinese_factuality_evaluation_for_large_language_models.md)**

:   提出 Chinese SimpleQA——首个全面的中文事实性评估基准，包含 3000 个高质量短问答（覆盖 6 大主题、99 个子主题），评估 41 个 LLM 后发现仅 o1-preview（63.8%）和 Doubao-pro-32k（61.9%）能通过，并系统揭示了"大模型更好"、"RAG缩小差距"、"对齐降低事实性"等关键洞察。

**[CLIPErase: Efficient Unlearning of Visual-Textual Associations in CLIP](llm_safety/cliperase_efficient_unlearning_of_visual-textual_associations_in_clip.md)**

:   提出 CLIPErase，一种专为 CLIP 多模态模型设计的机器遗忘框架，通过遗忘模块、保留模块和一致性模块三部分协同，选择性地移除特定视觉-文本关联，同时保持模型在保留数据上的性能。

**[ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty](llm_safety/comparisonqa_evaluating_factuality_robustness_of_llms_through_knowledge_frequenc.md)**

:   构建 ComparisonQA 基准（283K 配对问题），通过让高频和低频实体共享同一抽象问题实现受控对比，结合正确性和不确定性的两轮评估方法发现 LLM（包括 GPT-4o）对低频知识的鲁棒性极差。

**[Core: Robust Factual Precision with Informative Sub-Claim Identification](llm_safety/core_robust_factual_precision_with_informative_sub-claim_identification.md)**

:   本文提出 Core 框架，通过识别和过滤信息性子声明（informative sub-claims）来实现鲁棒的事实精度（factual precision）评估，解决了现有方法因无信息声明的稀释效应而导致评估不准确的问题。

**[Correcting Hallucinations in News Summaries: Exploration of Self-Correcting LLM Methods with External Knowledge](llm_safety/correcting_hallucinations_in_news_summaries_exploration_of_self-correcting_llm_m.md)**

:   系统性地探究了两种自纠正方法（CoVE 和 RARR）在新闻摘要幻觉纠正中的表现，比较了三种搜索引擎、多种检索设置和提示策略，发现 Bing 搜索片段 + RARR（few-shot）组合效果最佳，且 G-Eval 与人类评估高度一致。

**[Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence](llm_safety/cracking_hallucination_vhd.md)**

:   提出 VHD 指标量化每个注意力头输出对视觉输入的敏感程度，发现仅少数注意力头对视觉信息高度敏感而模型过度依赖语言先验是导致幻觉的关键因素，进而设计 VHR 免训练方法逐层自适应增强视觉感知头的贡献（$\alpha=2$），在 CHAIR 上将 LLaVA-1.5 的 CHAIR$_S$ 从 49.68 降至 33.32，且几乎无额外推理开销。

**[Defense Against Prompt Injection Attack by Leveraging Attack Techniques](llm_safety/defense_prompt_injection.md)**

:   本文提出一种"以攻为防"的 prompt injection 防御策略：将已有的攻击技术（ignore、escape、fake completion）反转用于防御，在被注入的数据内容后追加 shield prompt + 原始指令，使 LLM 忽略注入指令而执行原始指令，在多种攻击场景下将 ASR 降至接近零。

**[ReDial: Assessing Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks](llm_safety/dialect_fairness_robustness.md)**

:   本文构建了首个高质量人工标注的标准英语-AAVE平行推理基准ReDial（1216对），系统评估LLM在方言输入下的公平性与鲁棒性，发现几乎所有主流模型在AAVE查询上性能显著下降超过10%。

**[ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models](llm_safety/elba-bench_an_efficient_learning_backdoor_attacks_benchmark_for_large_language_m.md)**

:   建立了 ELBA-Bench——一个涵盖 12 种攻击方法、18 个数据集和 12 个 LLM 的综合后门攻击基准，系统评估 PEFT 和无微调两种范式下 LLM 后门攻击的有效性和隐蔽性。

**[Ensemble Watermarks for Large Language Models](llm_safety/ensemble_watermarks_llm.md)**

:   提出集成水印方法，将文体特征（藏头词 acrostic + 感觉运动词 sensorimotor norms）与已有红绿水印组合，在 paraphrasing 攻击后三特征集成检测率达 95%，而单独红绿水印仅 49%。

**[Estimating Privacy Leakage of Augmented Contextual Knowledge in Language Models](llm_safety/estimating_privacy_leakage_of_augmented_contextual_knowledge_in_language_models.md)**

:   本文提出context influence指标，基于差分隐私框架量化语言模型在解码时对增强上下文知识的隐私泄露程度，并系统分析了模型大小、上下文大小、生成位置等因素对隐私泄露的影响。

**[Exploring Forgetting in Large Language Model Pre-Training](llm_safety/exploring_forgetting_in_large_language_model_pre-training.md)**

:   系统性地探索了 LLM 预训练阶段的灾难性遗忘问题，提出了基于实体记忆的新指标（M_ex、M_in）替代传统 PPL 来检测遗忘，并验证了周期性高强度 memory replay 策略在缓解预训练遗忘中的有效性。

**[Fairness through Difference Awareness: Measuring Desired Group Discrimination in LLMs](llm_safety/fairness_difference_awareness.md)**

:   挑战当前LLM公平性评估中"差异无视"(difference unawareness)的主导范式，提出DiffAware和CtxtAware两个指标和包含8个场景16K问题的基准套件，证明在法律、文化、伤害评估等场景中模型应当区分群体差异，而现有去偏方法反而损害了这种必要的差异感知能力。

**[Faithful and Robust LLM-Driven Theorem Proving for NLI Explanations](llm_safety/faithful_and_robust_llm-driven_theorem_proving_for_nli_explanations.md)**

:   本文研究 LLM 与定理证明器（TP）的交互架构，提出四种策略来缓解自动形式化中的语义信息损失、语法错误、证明构造不足和反馈解读困难等问题，在 e-SNLI、QASC 和 WorldTree 三个数据集上分别实现了形式化精度 +18.46%/+34.2%/+39.77% 和解释质量 +29.5%/+51.5%/+41.25% 的显著提升。

**[From Misleading Queries to Accurate Answers: A Three-Stage Fine-Tuning Method for LLMs](llm_safety/from_misleading_queries_to_accurate_answers_a_three-stage_fine-tuning_method_for.md)**

:   提出三阶段微调方法（误导检测->查询纠正->准确回答）增强 LLM 处理含误导信息输入的能力，在误导检测和 QA 任务上显著提升准确率，同时减少幻觉生成。

**[From Trade-off to Synergy: A Versatile Symbiotic Watermarking Framework for Large Language Models](llm_safety/from_tradeoff_to_synergy_a_versatile.md)**

:   提出SymMark共生水印框架，融合logits-based和sampling-based两类水印方法（串行/并行/混合三种策略），通过token熵和语义熵自适应选择水印策略，在可检测性、鲁棒性、文本质量和安全性上实现SOTA。

**[Hallucination Detox: Sensitivity Dropout (SenD) for Large Language Model Training](llm_safety/hallucination_detox_send.md)**

:   提出Sensitivity Dropout (SenD)训练协议，通过识别并确定性丢弃训练过程中波动最大的嵌入索引（Sensitive Embedding Indices），减少LLM训练中幻觉的振荡行为，同时提出高效EigenScore近似方法(EES)实现2倍加速。

**[HALoGEN: Fantastic LLM Hallucinations and Where to Find Them](llm_safety/halogen_hallucinations.md)**

:   提出 HALoGEN——覆盖 9 个领域（含编程、科学引用、摘要等）的 10,923 条 prompt 的大规模幻觉评测框架，配套原子级自动验证器，在 14 个 LLM 的约 150,000 条生成上系统性评估幻觉，发现即使最佳模型也可能有高达 86% 的原子事实存在幻觉，并提出 Type A/B/C 三类错误分类法。

**[HD-NDEs: Neural Differential Equations for Hallucination Detection in LLMs](llm_safety/hd-ndes_neural_differential_equations_for_hallucination_detection_in_llms.md)**

:   本文首次将神经微分方程（Neural DEs）应用于LLM幻觉检测，通过对隐空间中token激活的连续轨迹建模来系统评估陈述的真实性，在True-False数据集上AUC-ROC超过SOTA 14%以上。

**[How Does Response Length Affect Long-Form Factuality](llm_safety/how_does_response_length_affect_long-form_factuality.md)**

:   本文系统研究了LLM响应长度与事实精确度的关系，提出高效的双层事实性评估框架Bafe（与人类注释89.31%一致），确认了长度偏差的存在，并通过排除错误传播和长上下文假说，证明"事实耗竭"是事实性下降的主要原因。

**[Improved Unbiased Watermark for Large Language Models](llm_safety/improved_unbiased_watermark_for_large_language.md)**

:   提出 MCmark，一族基于多通道（Multi-Channel）的无偏水印算法，通过将词表分割为 $l$ 个段并在选中段内提升 token 概率来嵌入统计信号，在保持 LLM 原始输出分布的同时，可检测性比现有无偏水印提升超 10%。

**[Ewe: Improving Factuality with Explicit Working Memory](llm_safety/improving_factuality_with_explicit_working_memory.md)**

:   提出 Ewe（Explicit Working mEmory），在 LLM 解码过程中引入由多个 KV cache 单元组成的显式工作记忆，实时接收检索知识反馈和事实核查反馈，检测到错误时删除错误句子并用更新后的记忆重新生成，在 4 个事实性长文本生成基准上将 VeriScore F1 提升 2–6 分且不损失回答有用性。

**[Improving Fairness of Large Language Models in Multi-document Summarization](llm_safety/improving_fairness_of_large_language_models_in_multi-document_summarization.md)**

:   提出 FairPO（Fair Preference Optimization），通过扰动式偏好对生成和公平感知偏好调优，同时优化多文档摘要中的摘要级和语料级公平性。

**[Improving Model Factuality with Fine-grained Critique-based Evaluator](llm_safety/improving_model_factuality_with_fine-grained_critique-based_evaluator.md)**

:   训练细粒度的事实性评估器 FenCE，通过在公开数据集上增强文本批评（critique）和多工具获取的多样化源文档来提升评估准确率，并利用 FenCE 对生成器响应进行修订和评分以构建偏好训练数据，使 Llama2-7B/Llama3-8B 在 FActScore 上分别提升 16.86%/14.45%。

**[Can Indirect Prompt Injection Attacks Be Detected and Removed?](llm_safety/indirect_prompt_injection_detection.md)**

:   本文系统研究间接 prompt injection 攻击的检测与移除：构建评估基准，发现现有检测模型对间接攻击表现不佳但专门训练的模型可达 99% 准确率，提出分割移除和抽取移除两种方法，并将检测+移除组合为过滤管道，有效降低间接 prompt injection 的攻击成功率。

**[Beyond Facts: Evaluating Intent Hallucination in Large Language Models](llm_safety/intent_hallucination_eval.md)**

:   本文提出"意图幻觉"（Intent Hallucination）概念——LLM 在处理复杂多条件查询时遗漏或误解部分意图约束导致的偏离用户意图的生成，构建 FaithQA 基准（20,068 题）和 Constraint Score 评估指标，实验表明意图幻觉在 SOTA 模型中普遍存在且随查询复杂度增加而加剧。

**[Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs](llm_safety/lacuna_inc_at_semeval-2025_task_4_lora-enhanced_influence-based_unlearning_for_l.md)**

:   提出 LIBU（LoRA 增强的影响函数遗忘算法），分两阶段实现 LLM 机器遗忘：Phase 1 用对角 Fisher 信息矩阵加权的影响函数更新参数精准遗忘，Phase 2 用 Sophia 二阶优化器稳定化训练，在 SemEval-2025 Task 4 的 OLMo-7B 上达到 0.283 遗忘率同时维持 0.469 MMLU 准确率。

**[Language Models Can Subtly Deceive Without Lying: A Case Study on Strategic Phrasing](llm_safety/language_models_can_subtly_deceive_without_lying_a_case_study_on_strategic_phras.md)**

:   构建了一个立法环境测试平台（LobbyLens），研究 LLM 是否能通过策略性措辞（strategic phrasing）——即不说谎但有意操纵表达方式——来隐藏修正案中对特定公司的利益导向，发现 LLM 经过 re-planning 可使欺骗率提升最多 40 个百分点。

**[Learning Auxiliary Tasks Improves Reference-Free Hallucination Detection in Open-Domain Long-Form Generation](llm_safety/learning_auxiliary_tasks_improves_reference-free_hallucination_detection_in_open.md)**

:   系统性地研究了开放域长文本生成中的无参考幻觉检测问题，发现 LLM 内部状态（概率/熵）不足以可靠区分事实与幻觉内容，并提出 RATE-FT（Rationale and Auxiliary Task Enhanced Fine-Tuning），通过引入推理解释和辅助 QA 任务增强微调，在 LongFact 上比普通微调提升 3% 以上。

**[Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?](llm_safety/llm_watermark_distillation_robustness.md)**

:   本文首次系统研究 LLM 水印在防止未授权知识蒸馏中的鲁棒性，提出三种水印去除攻击（无目标/有目标释义 + 推理时水印中和），发现有目标释义和水印中和可以彻底去除继承的水印，其中水印中和在保持知识迁移效率的同时实现零额外训练开销的水印去除。

**[Fine-grained Hallucination Detection and Mitigation in Long-form Question Answering](llm_safety/localizing_and_mitigating_errors_in_long-form_question_answering.md)**

:   本文构建了首个包含 span 级别错误标注的 LFQA 幻觉数据集 HaluQuestQA（698 QA 对，4.7k 错误标注，5 种错误类型），训练了一个自动反馈模型来检测不完整信息的错误 span 并生成解释，最后提出 Error-informed Refinement 方法利用反馈信号精炼答案，将幻觉减少约 3%，且人类评估中 84% 的用户更偏好精炼后的答案。

**[Mamba Knockout for Unraveling Factual Information Flow](llm_safety/mamba_knockout_for_unraveling_factual_information_flow.md)**

:   将 Transformer 上的 Attention Knockout 可解释性方法迁移至 Mamba-1 和 Mamba-2，揭示了 SSM 模型中事实信息的流动模式——发现 Mamba 与 Transformer 共享"主语 token 在中后层向最后 token 传递关键信息"的普遍模式，但在首 token 偏置和关系 token 依赖等方面存在架构特异性差异。

**[Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models](llm_safety/manu_modality_aware_unlearning.md)**

:   提出 MANU——首个模态感知的 MLLM 遗忘框架，通过四种互补的神经元重要性函数（绝对/频率/方差/RMS）识别跨模态纠缠的知识载体神经元，选择性剪枝 top-α% 神经元实现多模态和纯文本输入下的均衡遗忘，无需任何梯度更新。

**[MEGen: Generative Backdoor into Large Language Models via Model Editing](llm_safety/megen_generative_backdoor_into_large_language_models_via_model_editing.md)**

:   提出 MEGen，一种基于模型编辑的生成式后门攻击方法，能够仅通过少量样本修改少量局部参数，在 LLM 中注入生成式后门，使模型在触发时自由输出预设的危险内容。

**[Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models](llm_safety/merge_hijacking_backdoor_attacks_to_model_merging_of_large_language_models.md)**

:   提出 Merge Hijacking——首个针对 LLM 模型合并的后门攻击方法，攻击者仅需上传一个恶意模型，当受害者将其与任意干净模型合并时，生成的合并模型继承后门并在所有任务上保持攻击有效性和正常性能，且对现有防御方法具有鲁棒性。

**[Unveiling Privacy Risks in LLM Agent Memory](llm_safety/mextra_agent_memory_privacy.md)**

:   本文系统研究了 LLM Agent 记忆模块的隐私风险，提出 MEXTRA 黑盒记忆提取攻击，通过精心设计的定位-对齐攻击 prompt 和自动化多样 prompt 生成方法，在医疗和网购两种 Agent 上成功提取大量私人查询记录。

**[MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models](llm_safety/mmunlearner_reformulating_multimodal_machine_unlearning_in_the_era_of_multimodal.md)**

:   本文重新定义了多模态大语言模型（MLLM）时代的机器遗忘任务——仅擦除与特定实体关联的视觉模式而保留文本知识，并提出几何约束梯度上升方法MMUnlearner，通过权重显著性图选择性更新参数，在MLLMU-Bench和CLEAR两大基准上全面超越GA和NPO等基线。

**[Monitoring Decoding: Mitigating Hallucination via Evaluating the Factuality of Partial Response during Generation](llm_safety/monitoring_decoding_mitigating_hallucination_via_evaluating_the_factuality_of_pa.md)**

:   提出 Monitoring Decoding (MD) 框架，在生成过程中动态监控部分响应的事实性，通过监控函数识别易产生幻觉的 token 并利用树搜索策略选择性地修正这些关键 token，从而在保持效率的同时显著提升事实准确性。

**[MorphMark: Flexible Adaptive Watermarking for Large Language Models](llm_safety/morphmark_adaptive_watermarking.md)**

:   MorphMark 通过多目标权衡分析框架揭示了绿表概率 P_G 在水印效果与文本质量之间的关键作用，并据此提出自适应调整水印强度 r 的方法——当 P_G 高时增强水印、P_G 低时减弱水印，实现了在不依赖额外模型训练的前提下同时提升水印可检测性和文本质量。

**[On-Policy Self-Alignment with Fine-grained Knowledge Feedback for Hallucination Mitigation](llm_safety/on-policy_self-alignment_with_fine-grained_knowledge_feedback_for_hallucination_.md)**

:   提出 RLFH（Reinforcement Learning for Hallucination），一种在策略（on-policy）自对齐方法，让 LLM 自己作为评判者，将回复分解为原子事实并进行真实性和信息量评估，生成 token 级别的密集奖励信号，通过在线 PPO 优化来有效缓解幻觉问题。

**[Opt-Out: Investigating Entity-Level Unlearning for Large Language Models via Optimal Transport](llm_safety/opt-out_investigating_entity-level_unlearning_for_large_language_models_via_opti.md)**

:   提出 Opt-Out，一种基于最优传输理论的实体级 LLM 遗忘方法，利用 Sliced Wasserstein Distance 正则化参数偏移实现精细遗忘；同时构建首个实体级遗忘数据集 ELUDe（20 目标实体 + 144 邻居实体，15K+ forget / 90K+ retain QA 对），在 Llama-3.1-8B 和 Phi-3.5 上全面超越现有方法。

**[PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative Prompts](llm_safety/pig_privacy_jailbreak.md)**

:   提出 PIG 框架，通过识别隐私查询中的 PII 实体类型、构建隐私上下文示例、并利用三种基于梯度的迭代优化策略更新上下文，实现对 LLM 的高效隐私越狱攻击，在白盒和黑盒模型上均达到 SOTA。

**[Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models](llm_safety/private_memorization_editing_turning_memorization_into_a_defense_to_strengthen_d.md)**

:   提出 PME（Private Memorization Editing），将 LLM 的记忆化特性从安全弱点转化为防御手段，通过编辑 Feed Forward 层参数来移除已记忆的个人身份信息（PII），实现无需重训的隐私保护。

**[Real-time Factuality Assessment from Adversarial Feedback](llm_safety/real-time_factuality_assessment_from_adversarial_feedback.md)**

:   本文揭示了现有事实性评估数据集存在"数据泄漏"问题（LLM 因预训练记忆而轻松识别旧虚假信息），提出了一个基于 RAG 检测器对抗反馈的迭代改写流水线来生成真正具有挑战性的实时虚假新闻变体，使 GPT-4o RAG 检测器的 ROC-AUC 绝对下降 17.5%。

**[ReLearn: Unlearning via Learning for Large Language Models](llm_safety/relearn_unlearning_via_learning_for_large_language_models.md)**

:   ReLearn提出用"正向学习"替代传统的"逆向优化"来实现LLM知识遗忘，通过数据增强和微调pipeline使模型在遗忘目标知识的同时保持语言生成质量和流畅性，并设计了包含KFR、KRR和LS三个指标的综合评估框架。

**[REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space](llm_safety/revs_unlearning_sensitive_information_in_language_models_via_rank_editing_in_the.md)**

:   提出 REVS，一种无梯度的模型编辑方法，通过在 FF2 层中定位与敏感 token 关联最强的神经元，将其投影到词汇空间后迭代降低目标 token 排名，在 SSN/Email/URL 三类敏感数据上 Unlearning Score 显著超越 6 种基线（89.58 vs 36.98），同时通用能力几乎零损（MMLU 61.05→60.87），且对 Logit-Lens 和 Delta 提取攻击高度鲁棒。

**[Robust Data Watermarking in Language Models by Injecting Fictitious Knowledge](llm_safety/robust_data_watermarking_in_language_models_by_injecting_fictitious_knowledge.md)**

:   提出一种基于虚构知识（Fictitious Knowledge）的数据水印方法，通过在训练数据中注入虚构但合理的实体及其属性描述，实现对 LLM 训练数据所有权的可追溯验证，水印抗数据预处理过滤且支持黑盒 QA 验证。

**[SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models](llm_safety/saferoute_adaptive_model_selection_for_efficient_and_accurate_safety_guardrails_.md)**

:   提出 SafeRoute，一个二分类路由器，根据输入难度自适应地在小型和大型安全护栏模型之间选择，仅对约5%的"困难"样本使用大模型，在保持安全检测精度的同时大幅降低计算开销。

**[SEUF: Is Unlearning One Expert Enough for Mixture-of-Experts LLMs?](llm_safety/seuf_is_unlearning_one_expert_enough_for_mixture-of-experts_llms.md)**

:   SEUF 首次揭示现有 LLM 遗忘方法在 MoE 模型上严重失效（效用下降 35%+），根因是遗忘过程导致路由器的专家选择漂移形成"捷径"——本该遗忘的目标专家被绕过而无辜专家被破坏，并提出通过专家归因定位目标专家+路由器锚定损失固定选择的框架，仅更新 0.06% 参数即可同时提升遗忘质量和模型效用。

**[Stochastic Chameleons: Irrelevant Context Hallucinations Reveal Class-Based (Mis)Generalization in LLMs](llm_safety/stochastic_chameleons_irrelevant_context_hallucinations_reveal_class-based_misge.md)**

:   通过行为分析和机械可解释性实验揭示 LLM 无关上下文幻觉的内部机制：模型在底层构建抽象类别表示（如"语言"），然后两条竞争电路（query-based vs context-based）争夺特征选择权，相对激活强度决定正确泛化还是产生幻觉。

**[TIP of the Iceberg: Task-in-Prompt Adversarial Attacks on LLMs](llm_safety/tip_iceberg_adversarial_attacks.md)**

:   本文提出 Task-in-Prompt (TIP) 攻击——一类通过在 prompt 中嵌入序列到序列任务（如密码解码、谜语、代码执行）来间接生成违禁内容的新型越狱攻击类别，并构建 PHRYGE benchmark 系统评估，证明该攻击可成功绕过 GPT-4o、LLaMA 3.2 等六种 SOTA LLM 的安全防护。

**[Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach](llm_safety/towards_context-robust_llms_a_gated_representation_fine-tuning_approach.md)**

:   提出 Grft（Gated Representation Fine-Tuning），一种轻量级即插即用的门控表示微调方法，仅需不到 200 个训练样本和模型 0.0004% 的参数，即可让 LLM 在面对矛盾、无用的外部上下文时表现出类似人类的鲁棒认知行为。

**[Towards Effective Extraction and Evaluation of Factual Claims](llm_safety/towards_effective_extraction_and_evaluation_of_factual_claims.md)**

:   提出了一个用于评估事实声明抽取质量的标准化框架（包含覆盖率和去语境化等指标），并开发了Claimify——一个能在高置信度下处理歧义并抽取声明的LLM方法，在该框架下显著优于已有方法。

**[TreeCut: A Synthetic Unanswerable Math Word Problem Dataset for LLM Hallucination Evaluation](llm_safety/treecut_a_synthetic_unanswerable_math_word_problem_dataset_for_llm_hallucination.md)**

:   提出 TreeCut，一种基于树结构的合成数据集生成方法，通过在树路径上移除必要条件边来系统性生成无穷多的不可回答数学应用题，用以评估 LLM 在面对不可解问题时的幻觉行为。

**[Truth Knows No Language: Evaluating Truthfulness Beyond English](llm_safety/truth_knows_no_language_evaluating_truthfulness_beyond_english.md)**

:   构建首个专业翻译的多语言 TruthfulQA 基准（巴斯克语、加泰罗尼亚语、加利西亚语、西班牙语），发现 LLM 的跨语言真实性差异小于预期，且 LLM-as-a-Judge 比多选题指标更贴合人类判断。

**[The Tug of War Within: Mitigating the Fairness-Privacy Conflicts in Large Language Models](llm_safety/tug_of_war_fairness_privacy.md)**

:   发现 LLM 通过 SFT 增强隐私意识会显著降低公平性意识（trade-off），提出无训练方法 SPIN（抑制公平-隐私耦合神经元），基于信息论解耦两种意识，在 Qwen2-7B 上同时提升公平性 12.2% 和隐私意识 14.0%。

**[UAlign: Leveraging Uncertainty Estimations for Factuality Alignment on Large Language Models](llm_safety/ualign_leveraging_uncertainty_estimations_for_factuality_alignment_on_large_lang.md)**

:   提出 UAlign 框架，利用置信度分数和语义熵两种不确定性估计来显式建模 LLM 知识边界，并将其作为输入特征融入 PPO 对齐训练，使模型自信回答已知问题、坚定拒绝未知问题，在多个知识 QA 数据集上显著提升可靠性与泛化性。

**[Unveiling and Addressing Pseudo Forgetting in Large Language Models](llm_safety/unveiling_and_addressing_pseudo_forgetting_in_large_language_models.md)**

:   揭示 LLM 持续学习中的"伪遗忘"现象：性能下降并非因为模型丧失了旧任务能力，而是指令无法正确激活已有能力。通过归因分析证明遗忘模型的指令依赖度降低，并提出基于 Rationale-Guidance Difficulty（RGD）的动态数据回放框架 RGD-R 来缓解伪遗忘。

**[When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations](llm_safety/when_backdoors_speak_understanding_llm_backdoor_attacks_through_model-generated_.md)**

:   本文首次从自然语言解释的角度研究 LLM 后门攻击，发现后门模型对干净输入生成逻辑连贯的解释，但对中毒输入生成多样且逻辑有缺陷的解释；进一步通过 token 级和句子级分析揭示中毒样本的预测语义仅在最后几层才出现，且注意力从输入上下文转移到新生成的 token。

**[Which Retain Set Matters for LLM Unlearning? A Case Study on Entity Unlearning](llm_safety/which_retain_set_matters_for_llm_unlearning_a_case_study_on_entity_unlearning.md)**

:   系统研究实体遗忘中 retain set 的选择问题，提出 Syntactically Similar Neighbor Set，发现句法相似性（而非领域/实体相似性）才是遗忘过程中知识退化的主要驱动因素，用句法相似的 retain set 做正则化可同时最优保护所有类型的邻居知识。

**[ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging](llm_safety/zjuklab_at_semeval-2025_task_4_unlearning_via_model_merging.md)**

:   在 SemEval-2025 Task 4（LLM 敏感内容遗忘）中获得第二名，核心思路是训练两个互补模型（一个过度遗忘、一个遗忘不足），通过 TIES-Merging 合并得到平衡遗忘的模型，本地实验达到近乎完美的 MIA 分数 0.501。

---

## 📦 模型压缩 { #model_compression }

**[500xCompressor: Generalized Prompt Compression for Large Language Models](model_compression/500xcompressor_generalized_prompt_compression_for_large_language_models.md)**

:   提出 500xCompressor，将最多约 500 个自然语言 token 压缩为最少 1 个特殊 token 的 KV 值，实现 6x 到 480x 的压缩比，仅增加约 0.25% 的参数，LLM 在压缩后保留 62.26%-72.89% 的原始能力，显著超越 ICAE 基线。

**[Accurate KV Cache Quantization with Outlier Tokens Tracing](model_compression/accurate_kv_cache_quantization_with_outlier_tokens_tracing.md)**

:   发现 KV Cache 的 outlier channel 中存在少量异常 token 偏离先前假设的均匀分布，提出 OTT（Outlier Tokens Tracing）方法，在量化过程中动态追踪并排除这些 token，在 2-bit 量化下实现 6.4x 内存压缩和 2.3x 吞吐提升，同时显著提高精度。

**[AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation](model_compression/aligndistil_token_level_alignment.md)**

:   AlignDistil 证明了 RLHF 目标函数与 token 级蒸馏过程的理论等价性，并据此设计了一种简单的蒸馏方法：用 DPO 模型和反向 DPO 模型的 logit 分布线性组合构造教师分布，配合 token 自适应外推机制实现 token 级奖励优化，在 AlpacaEval 2.0、MT-Bench 和 Arena-Hard 上优于现有方法且收敛更快。

**[APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs](model_compression/apb_distributed_long_context.md)**

:   APB 提出了一种分布式长上下文推理框架，通过在序列并行框架中引入本地 KV cache 压缩和跨 GPU 传递压缩上下文块的机制，在不损失任务性能的前提下实现了相比 FlashAttn/RingAttn/StarAttn 分别高达 9.2x/4.2x/1.6x 的 prefill 加速。

**[Assigning Distinct Roles to Quantized and Low-Rank Matrices Toward Optimal Weight Decomposition](model_compression/assigning_distinct_roles_to_quantized_and_low-rank_matrices_toward_optimal_weigh.md)**

:   提出ODLRI (Outlier-Driven Low-Rank Initialization)，为联合量化+低秩优化(Q+LR)框架中的低秩分量赋予明确角色——捕获激活异常值敏感权重，使量化分量处理更平滑的残差，在Llama2/3和Mistral的2-bit极端量化场景下持续降低困惑度和提升零样本精度。

**[Basic Reading Distillation](model_compression/basic_reading_distillation.md)**

:   本文提出基础阅读蒸馏（BRD），通过让教师LLM在通用语料上生成基础阅读行为数据（包括NER和问答），训练小型学生模型模仿这些行为，使564M参数的小模型在不接触下游任务数据的情况下就能在多种NLP任务上达到或超过20倍大的教师模型性能。

**[BeamLoRA: Beam-Constraint Low-Rank Adaptation](model_compression/beamlora_beam_constraint_lora.md)**

:   BeamLoRA 发现 LoRA 模块中不同 rank 的重要性存在显著差异且随训练动态演变，受 beam search 启发，提出在训练过程中动态评估 rank 重要性、剪枝不重要的 rank 并将参数空间扩展给重要 rank，在固定总 rank 下提升性能，在三个基座模型的 12 个数据集上持续优于 LoRA 及其变体。

**[Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation](model_compression/beyond_logits_aligning_feature_dynamics_for_effective_knowledge_distillation.md)**

:   本文提出一种超越 logit 匹配的知识蒸馏方法，通过对齐教师和学生模型在训练过程中的特征变化动态（而非静态特征快照），实现更有效的知识转移，显著提升了 NLP 任务上的蒸馏效果。

**[Beyond Text Compression: Evaluating Tokenizers Across Scales](model_compression/beyond_text_compression_tokenizers.md)**

:   本文系统评估了 6 种 tokenizer 在 350M 和 2.7B 参数模型上的影响，发现 tokenizer 选择对英文任务影响极小但对多语言任务（如机器翻译）有显著且跨尺度一致的影响，并提出了基于 Zipf 定律的新型内在评估指标，比文本压缩率能更好地预测多语言场景下的下游性能。

**["Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization](model_compression/bf16_or_death_quantization_tradeoffs.md)**

:   这是迄今最全面的 LLM 量化实证研究，在 Llama-3.1 全系列（8B/70B/405B）上对 FP8/INT8/INT4 进行了超过 50 万次评估，发现 FP8 几乎无损、INT8 仅降 1-3%、INT4 出奇地有竞争力，并给出了不同部署场景的量化格式选择建议。

**[BlockPruner: Fine-grained Pruning for Large Language Models](model_compression/blockpruner_fine-grained_pruning_for_large_language_models.md)**

:   提出 BlockPruner，将 Transformer 层分解为 MHA 和 MLP 两个最小残差块，基于困惑度评估块重要性并通过迭代搜索进行细粒度剪枝，实现比层级剪枝更优的压缩效果。

**[BrainECHO: Semantic Brain Signal Decoding through Vector-Quantized Spectrogram Reconstruction for Whisper-Enhanced Text Generation](model_compression/brainecho_semantic_brain_signal_decoding_through_vector-quantized_spectrogram_re.md)**

:   提出 BrainECHO 三阶段框架（自编码—对齐—微调），通过向量量化离散表示将脑信号映射到 Mel 频谱图空间，再借助 Whisper 完成非侵入式脑信号到文本的高质量解码。

**[CAMI: A Counselor Agent Supporting Motivational Interviewing through State Inference and Topic Exploration](model_compression/cami_a_counselor_agent_supporting_motivational_interviewing_through_state_infere.md)**

:   本文提出CAMI（Counselor Agent for Motivational Interviewing），一个基于动机式访谈（MI）原则的咨询Agent，通过STAR框架（状态推断、话题探索、回复生成）来引导来访者产生改变谈话（change talk），在自动化和人工评估中均优于现有方法。

**[Capture the Key in Reasoning to Enhance CoT Distillation Generalization](model_compression/capture_key_cot_distillation.md)**

:   提出 EDIT（mistakE-Driven key reasonIng step distillaTion），通过构造正确/错误配对的 dual CoTs 数据，利用最小编辑距离算法定位关键推理步骤，并以 token 级细粒度损失函数引导小模型聚焦学习这些关键步骤，而非简单模仿教师的推理形式。

**[CFSP: An Efficient Structured Pruning Framework for LLMs with Coarse-to-Fine Activation Information](model_compression/cfsp_an_efficient_structured_pruning_framework_for_llms_with_coarse-to-fine_acti.md)**

:   本文提出CFSP框架，利用粗粒度（block间）和细粒度（block内）的激活信息作为重要性准则指导LLM的结构化剪枝，仅需一次前向传播即可完成剪枝，在多个模型和稀疏度预算上优于现有方法。

**[ClaimPKG: Enhancing Claim Verification via Pseudo-Subgraph Generation with Lightweight Specialized LLM](model_compression/claimpkg_enhancing_claim_verification_via_pseudo-subgraph_generation_with_lightw.md)**

:   提出 ClaimPKG 框架，通过轻量级专用 LLM 将文本声明转换为伪子图表示，再从知识图谱中检索相关子图作为证据，最终由通用 LLM 进行推理验证，在 FactKG 数据集上比 SOTA 高出 9%-12% 准确率。

**[Compact and Compressible Representations for LLMs Using Structured Sparse Decomposition](model_compression/compact_and_compressible_representations_for_llms_using_structured_sparse_decom.md)**

:   本文提出一种结构化稀疏分解方法，将LLM权重矩阵分解为低秩部分和结构化稀疏部分的组合，实现高压缩比的同时保持模型性能，使大模型在资源受限环境下高效部署成为可能。

**[Compression in Transformer Language Models Has a Surprising Relationship with Performance](model_compression/compression_in_transformer_language_models_has_a_surprising_relationship_with_pe.md)**

:   本文从信息论角度研究Transformer语言模型中压缩（权重的可压缩性）与模型性能之间的关系，发现了一个反直觉的现象：在一定范围内，更容易被压缩的模型反而具有更好的泛化性能，这与最小描述长度（MDL）原理的预测一致。

**[DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression](model_compression/dac_prompt_compression.md)**

:   DAC 提出动态注意力感知的 prompt 压缩方法，通过融合信息熵和注意力分数作为 token 重要性度量，并动态感知压缩过程中的熵偏移来进行细粒度压缩，在 LongBench 上比 SOTA 方法提升平均 1.33 分。

**[Data Laundering: Artificially Boosting Benchmark Results through Knowledge Distillation](model_compression/data_laundering_artificially_boosting_benchmark_results_through_knowledge_distil.md)**

:   本文揭示了知识蒸馏可被滥用来人为提高基准测试分数的漏洞——通过"数据洗白"（Data Laundering）方法，将教师模型在测试集上学到的知识通过看似合法的中间训练步骤隐蔽地传递给学生模型，使一个2层BERT即可在GPQA上达到73.94%（接近OpenAI o1的77.30%），而该模型并未真正学会推理。

**[DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking](model_compression/deepsolution_boosting_complex_engineering_solution_design_via_tree-based_explora.md)**

:   提出面向复杂工程方案设计的新基准 SolutionBench 和新系统 SolutionRAG，通过树搜索探索+双视角思维（设计-审查交替）在 RAG 框架下逐步生成满足多约束的可靠工程方案，在 8 个工程领域达到 SOTA。

**[Direct Behavior Optimization: Unlocking the Potential of Lightweight LLMs](model_compression/direct_behavior_optimization_unlocking_the_potential_of_lightweight_llms.md)**

:   提出 DeBoP 范式，将轻量级 LLM（LwLLM）的行为优化转化为对离散执行序列的优化，通过无梯度蒙特卡洛树搜索（MCTS）自动寻找最优 demonstration，使 LLaMA3-8B 在多数任务上超越 GPT-3.5 并减少约 60% 计算时间。

**[Disentangling the Roles of Representation and Selection in Data Pruning](model_compression/disentangling_the_roles_of_representation_and_selection_in_data_pruning.md)**

:   本文将数据剪枝（data pruning）系统性地拆解为"数据表示"和"选择算法"两个独立维度，通过理论分析和大规模实验发现：表示质量（尤其是训练梯度）对剪枝效果起决定性作用，而不同选择算法在不同场景下各有优劣，且常常偏离其设计目标。

**[DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization](model_compression/drpruning_robust_pruning.md)**

:   DRPruning 将分布稳健优化（DRO）引入 LLM 结构化剪枝，通过 scaling law 预测各领域最终 loss 作为参考、动态调整训练数据分布来平衡剪枝后各领域性能，在单语和多语设置下分别以 -5.59% PPL 和 +2.95% 下游任务的提升超越 Sheared LLaMA。

**[EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models](model_compression/eac_moe_expert_aware_compression.md)**

:   EAC-MoE 深入分析 MoE 模型的专家选择特性，提出两个互补模块——量化时通过逐层校准路由器缓解 expert-shift 问题（QESC），推理时基于专家选择频率动态剪枝不重要专家（PESF），在 4 个 MoE 模型上实现显著的内存压缩和推理加速且精度损失极小。

**[Efficient Long Context Language Model Retrieval with Compression](model_compression/efficient_long_context_language_model_retrieval_with_compression.md)**

:   提出 CoLoR（Compression for Long context Retrieval），通过偏好优化和长度正则化联合训练段落压缩模型，在保持长上下文语言模型检索性能提升 6% 的同时将上下文长度压缩 1.91 倍。

**[EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](model_compression/efficientqat.md)**

:   EfficientQAT 提出两阶段 QAT 框架——先逐块训练所有参数（Block-AP）提供良好初始化，再端到端训练量化参数（E2E-QP）捕获跨块交互，在单张 A100 上 41 小时完成 Llama-2-70B 的 2-bit 量化，精度仅降 3 点。

**[Explaining Puzzle Solutions in Natural Language: An Exploratory Study on 6×6 Sudoku](model_compression/explaining_puzzle_solutions_in_natural_language_an_exploratory_study_on_6x6_sudo.md)**

:   评估五个LLM在求解和解释6×6数独谜题上的能力，发现即使o1-preview能解出65%的题目，其推理解释在忠实性、清晰度和教育价值方面仍严重不足。

**[FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Large Language Models](model_compression/fedex_lora_federated_exact_aggregation.md)**

:   FedEx-LoRA 发现联邦学习中独立平均 LoRA 的 A 和 B 矩阵会导致不精确的全局更新（"乘积的均值≠均值的乘积"），通过在冻结权重矩阵中加入残差误差项实现精确聚合，在多个推理和 NLU 任务上一致优于 FedIT 和 FFA-LoRA。

**[Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching](model_compression/flipping_kd_small_to_large.md)**

:   本文提出"反向知识蒸馏"范式——让 LLM 从微调过的小模型学习文本匹配的领域专家知识，通过将 decoder-only LLM 重新解释为 encoder-decoder 架构（用 LoRA 的压缩矩阵做 encoder）并设计 Margin-aware Contrastive Loss 来对齐表示相似度。

**[A Silver Bullet or a Compromise for Full Attention? A Comprehensive Study of Gist Token-based Context Compression](model_compression/gist_token_context_compression.md)**

:   对基于 Gist Token 的上下文压缩方法进行全面系统研究，发现细粒度 KV Cache 架构在 RAG/QA 等任务上接近无损，但在精确回忆任务上存在明显差距，并识别出三种关键失败模式和两种有效改进策略。

**[GSQ-Tuning: Group-Shared Exponents Integer in Fully Quantized Training for LLMs On-Device Fine-tuning](model_compression/gsq-tuning_group-shared_exponents_integer_in_fully_quantized_training_for_llms_o.md)**

:   GSQ-Tuning 提出了一种基于"组共享指数整数"（Group-Shared Exponents Integer）格式的全量化微调框架，在推理和训练中完全消除浮点运算，结合 LoRA 适配器实现了精度接近 BF16 微调、内存降低 1.85x、功耗降低 5x、芯片面积缩小 11x 的边端 LLM 微调方案。

**[IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](model_compression/iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)**

:   发现不同规模 LLM 的注意力矩阵具有高度相似性，提出 IAM 框架——在 prefill 阶段建立小模型与大模型注意力头之间的余弦相似度映射，decode 阶段用小模型的注意力矩阵替代大模型映射层的注意力计算，实现 KV cache 减少 22% 和推理加速 11%，且与现有 KV cache 压缩方法正交。

**[L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](model_compression/l4q_parameter_efficient_quantization_aware_finetuning.md)**

:   提出 L4Q，将量化感知训练 (QAT) 与 LoRA 深度整合：先合并权重与LoRA参数再统一量化，通过定制反向传播路径消除权重梯度存储开销，实现联合优化量化与微调参数，在4-bit和3-bit量化下显著超越现有方法。

**[Language Models Resist Alignment: Evidence From Data Compression](model_compression/language_models_resist_alignment.md)**

:   本文从压缩理论视角提出LLM的"弹性"(elasticity)概念，证明模型在受到微调扰动时压缩率变化与数据集大小成反比——因为预训练数据远大于对齐数据，对齐效果被优先"遗忘"，这从信息论角度根本性地解释了为什么LLM对齐如此脆弱。

**[Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders](model_compression/language_specific_features.md)**

:   利用 Sparse Autoencoders (SAEs) 分析多语言 LLM 的内部表示，发现存在强烈的语言特定 SAE features，这些 features 不仅与语言特有 token 相关还与语言上下文相关，消融它们只影响对应语言能力，且多个语言 features 之间存在协同效应；进一步利用这些 features 增强 steering vectors 实现对生成语言的精确控制。

**[Towards the Law of Capacity Gap in Distilling Language Models](model_compression/law_of_capacity_gap_distilling_language_models.md)**

:   揭示了语言模型蒸馏中的"容量差距定律"——最优教师模型的参数量与学生模型成线性关系（约 2.5 倍），将 LLM 蒸馏中的"不可能三角"转化为可解问题，并据此成功蒸馏出 3B 的 MiniMA 模型。

**[Limited-Resource Adapters Are Regularizers, Not Linguists](model_compression/limited-resource_adapters_are_regularizers_not_linguists.md)**

:   本文将 adapter souping（权重平均）与交叉注意力微调结合用于低资源克里奥尔语机器翻译，发现虽然方法带来了显著提升（最高 +8 BLEU），但语言关联性与 adapter 性能无有意义的协变关系——随机初始化的未训练 adapter 表现同样优秀，表明 adapter 在此设定下的作用本质是**参数正则化而非语言信息迁移**。

**[LLMSR@XLLM25: Less is More: Enhancing Structured Multi-Agent Reasoning via Quality-Guided Distillation](model_compression/llmsrxllm25_less_is_more_enhancing_structured_multi-agent_reasoning_via_quality-.md)**

:   本文提出 Less is More 框架，在仅有 24 个标注样本的极端低资源条件下，通过逆向提示归纳、GPT-4o 增强的检索式推理合成和双阶段奖励引导过滤三个阶段，蒸馏出高质量的结构化推理数据来微调 LLaMA3-8B 多智能体系统，在 XLLM@ACL2025 共享任务中获得第三名。

**[LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation](model_compression/longred_mitigating_short-text_degradation_of_long-context_large_language_models_.md)**

:   本文系统分析了长上下文LLM在短文本任务上性能退化的两个原因（分布漂移和灾难性遗忘），并提出LongReD方法，通过短文本蒸馏和短到长蒸馏两个训练目标来最小化扩展模型与原始模型之间的分布差异，在保持长文本建模能力的同时将短文本性能保留至原始模型的99.4%。

**[Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation](model_compression/magnet_multi-turn_tool-use_data_synthesis_and_distillation_via_graph_translation.md)**

:   提出 Magnet 框架，基于函数依赖图的随机游走和节点操作（Insert/Merge/Split）构建高质量多轮 Function Calling 训练轨迹，结合基于提示的上下文蒸馏生成正负对比轨迹进行 SFT + mDPO 训练，使 14B 模型 Magnet-14B-mDPO 在 BFCL-v3 上达到 68.01（排名第 4），在多轮场景上大幅超越教师模型 Gemini-1.5-pro-002。

**[MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts](model_compression/moqae_mixed_precision_kv_cache.md)**

:   MoQAE 创造性地将不同量化比特宽度配置视为 MoE 中的"专家"，通过轻量路由器学习每个 chunk 的最优量化策略，结合路由冻结和路由共享机制，在几乎不损失精度的情况下大幅减少长上下文推理的 KV cache 内存。

**[mPLUG-DocOwl2: High-resolution Compressing for OCR-free Multi-page Document Understanding](model_compression/mplug_docowl2_doc_compress.md)**

:   提出布局感知的High-resolution DocCompressor模块，用全局低分辨率视觉特征作为query、子图特征作为key/value进行分组交叉注意力，将每张高分辨率文档图片从数千tokens压缩至324 tokens，配合三阶段训练框架在多页文档理解上达到SOTA且First Token Latency降低50%以上。

**[One QuantLLM for ALL: Fine-tuning Quantized LLMs Once for Efficient Deployments](model_compression/one_quantllm_for_all_fine-tuning_quantized_llms_once_for_efficient_deployments.md)**

**[Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging](model_compression/osrm_lora_merging_orthogonal.md)**

:   OSRM 发现 LoRA 模型合并失败的根因是参数与数据分布的交互干扰（而非仅仅是参数冲突），提出在微调前通过数据协方差矩阵的特征分解来初始化 LoRA 矩阵 A，使其子空间与其他任务的数据分布正交，从而在合并时最小化跨任务干扰，在 8 个数据集、5 个模型上显著提升合并性能。

**[Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models](model_compression/outlier-safe_pre-training_for_robust_4-bit_quantization_of_large_language_models.md)**

:   OSP（Outlier-Safe Pre-Training）框架通过三项创新——Muon 优化器（消除特权基方向）、Single-Scale RMSNorm（防止通道放大）和可学习嵌入投影层（重分布嵌入层激活），在预训练阶段主动防止异常值形成，训练的 1.4B 模型在 1T tokens 上实现近零超额峰度（0.04 vs 标准模型的 1818.56），在激进4-bit量化下平均分 35.7（Adam 为 26.5），仅 2% 训练开销。

**[C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](model_compression/parameter-efficient_fine-tuning_via_circular_convolution.md)**

:   提出 C3A 方法用循环卷积算子替代 LoRA 的低秩矩阵分解实现参数高效微调，核心优势是矩阵秩与参数量解耦——可用少量参数实现高秩适配，同时通过 FFT 保持与 LoRA 相当的计算和内存效率，在多种微调任务上一致超越 LoRA 及其变体。

**[Pre-training Distillation for Large Language Models: A Design Space Exploration](model_compression/pre-training_distillation_for_large_language_models_a_design_space_exploration.md)**

:   系统性地探索大语言模型预训练蒸馏（Pre-training Distillation）的设计空间，从 logits 处理、损失函数选择、scaling law 和 offline/online logits 四个维度进行广泛实验，找到更优配置并得出有价值的结论。

**[Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation](model_compression/prompt_distill_teacher_student.md)**

:   提出CanDist框架，借鉴人类面对不确定性时的"模糊规避"心理，引导LLM输出多个候选标签而非单一标签(候选标注)，再通过分布精炼(Distribution Refinery)策略蒸馏到小语言模型(SLM)获得最终标注，从理论到实验证明候选标注蒸馏优于单一标注。

**[PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models](model_compression/ptq161_low_bit_quantization.md)**

:   提出 PTQ1.61，首个将 LLM 权重有效压缩到真正 sub-2-bit（1.61-bit）的后训练量化方法，通过一维结构化掩码（仅增加 0.0002-bit 开销）、分块缩放因子优化和量化预处理三项技术实现 SOTA 性能。

**[Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis](model_compression/quaff_quantized_peft.md)**

:   本文提出 Outlier Spatial Stability Hypothesis (OSSH)——微调期间激活异常通道的空间位置保持稳定——并基于此设计了 Quaff 框架，通过目标动量缩放仅处理少量不变的异常通道，实现 1.73× 延迟降低和 30% 内存节省，同时在 GPQA 上精度还提升了 0.6%。

**[Quantification of Large Language Model Distillation](model_compression/quantification_of_large_language_model_distillation.md)**

:   本文提出了两种互补的LLM蒸馏量化方法——身份一致性评估（ICE）和响应相似性评估（RSE），通过越狱攻击挖掘模型身份信息泄露和多粒度响应相似性来衡量模型的蒸馏程度，发现大多数知名LLM（除Claude、Doubao和Gemini外）都表现出较高的蒸馏程度。

**[Revisiting LoRA through the Lens of Parameter Redundancy: Spectral Encoding Helps](model_compression/revisiting_lora_through_the_lens_of_parameter_redundancy_spectral_encoding_helps.md)**

:   本文系统研究了 LoRA 微调中的参数冗余问题，发现降低密度冗余不会损害表达能力（稀疏性质），并提出 SeLoRA——利用频谱变换（Fourier/Wavelet）从稀疏频谱子空间重参数化 LoRA 矩阵，以更少参数实现更优性能，且可即插即用地集成到多种 LoRA 变体中。

**[Spectra 1.1: Scaling Laws and Efficient Inference for Ternary Language Models](model_compression/scaling_laws_and_efficient_inference_for_ternary_language_models.md)**

:   本文系统研究三值语言模型（TriLM）的缩放规律，发现 TriLM 从增加训练数据中获益远大于增加参数量，基于此训练了在 1.2T token 上预训练的 Spectra-1.1 模型族（1B/2B/3B），并提出 1.6-bit 和 2-bit 权重打包方案及 TriRun GPU 内核，实现最高 8 倍的推理加速。

**[Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing](model_compression/sci-lora_mixture_of_scientific_loras_for_cross-domain_lay_paraphrasing.md)**

:   提出 Sci-LoRA——一种混合多领域 LoRA 的框架，通过对比学习训练文本编码器+动态权重生成器+LoRA 融合模块，在无需领域标签的情况下实现跨12个学科领域的科学文本通俗化改写，在5个数据集10个指标上超越 SOTA。

**[SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation](model_compression/scope_optimizing_key-value_cache_compression_in_long-context_generation.md)**

:   提出 SCOPE 框架，针对长上下文生成任务分别优化 prefill 和 decoding 阶段的 KV 缓存压缩策略——prefill 阶段保留完整缓存以维持理解能力，decoding 阶段采用滑动窗口选择 heavy hitters，并通过 adaptive 和 discontinuous 策略进一步优化内存和传输效率。

**[Mitigating Selection Bias with Node Pruning and Auxiliary Options](model_compression/selection_bias_node_pruning.md)**

:   提出 Bias Node Pruning (BNP) 和 Auxiliary Option Injection (AOI) 两种互补方法，通过定位并剪除模型输出层中 0.002% 的偏差参数（白盒）与注入"I don't know"辅助选项（黑盒通用），从内外两端同时缓解 LLM 在多选题中的选择偏差，同时提出分布级偏差度量 CKLD，组合方法在 Llama-3 上将 ARC-Challenge 准确率从 52.3% 提升至 65.3%。

**[Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs](model_compression/sparse_logit_sampling_accelerating_knowledge_distillation_in_llms.md)**

:   证明了朴素的 Top-K 稀疏知识蒸馏会产生有偏估计，提出基于重要性采样的 Random Sampling Knowledge Distillation (RSKD) 方法，提供无偏梯度估计，仅需存储极度稀疏的 logits，训练开销仅比交叉熵增加不到 10%，性能与全量蒸馏持平。

**[State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](model_compression/state_offset_tuning_ssm_peft.md)**

:   针对 SSM（如 Mamba）提出 State-offset Tuning，一种新的"状态基"PEFT 方法家族，通过在每个时间步直接注入可训练的状态偏移量 $h'$ 替代 Prefix-Tuning 的虚拟 token，解决了 prompt-based 方法在 SSM 上表达能力受限的问题，在更少参数量下持续优于 LoRA 和 Prefix-Tuning。

**[STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning](model_compression/stun_moe_pruning.md)**

:   STUN 提出"先结构化后非结构化"的两阶段 MoE 剪枝范式：第一阶段利用路由权重的行为相似性聚类冗余专家，以 $O(1)$ GPU 前向传播完成专家级剪枝；第二阶段在剩余专家内做非结构化权重剪枝，两者协同在 480B Snowflake Arctic 上以 40% 稀疏度几乎无性能损失。

**[TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering](model_compression/tada_training-free_recipe_for_decoding_with_adaptive_kv_cache_compression_and_me.md)**

:   提出 TaDA——无需训练的 KV cache 压缩方法，通过对 K/V 激活做 head 维度均值中心化后量化偏差（而非原始激活），自动消除离群值问题，配合逐层自适应量化精度搜索，将 KV cache 压缩至原始 16 位的 27% 同时保持接近基线的精度。

**[TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition](model_compression/teamlora_boosting_low-rank_adaptation_with_expert_collaboration_and_competition.md)**

:   提出 TeamLoRA，通过非对称协作模块（共享A矩阵+多个专家B矩阵的"插件式"组织）和基于Shapley值的竞争模块来优化Multi-LoRA架构，在多任务学习中实现了更好的效果-效率平衡——训练时间比MoELoRA减少30%，推理速度提升40%，同时性能更优。

**[Trans-PEFT: Transferable Parameter-Efficient Fine-Tuning on Evolving Base Models](model_compression/trans_peft_transferable.md)**

:   Trans-PEFT 发现基座模型更新（如 Qwen2→Qwen2.5）主要改变 FFN 层的任务知识存储而较少影响 Attention 层的任务模式，据此提出层内知识掩码和跨层知识丢弃两种策略，使在旧版本上训练的 PEFT 模块可直接迁移到新版本而不需重新微调，性能提升可达 30%。

**[UniICL: An Efficient ICL Framework Unifying Compression, Selection, and Generation](model_compression/uniicl_icl_framework.md)**

:   提出 UniICL 框架，用**一个冻结的 LLM** 同时完成 demonstration 压缩（compress→virtual tokens）、demonstration 选择（基于压缩后的 virtual token 相似度排序）和最终响应生成三个任务，仅需 17M 可训练参数（projection layer + learnable embedding），配合 Demonstration Bank 缓存机制避免重复压缩，实现 12× 压缩率下从 4-shot 扩展到 64-shot ICL（24GB 显存内），在多个 out-of-domain 数据集上超越 AutoCompressor、ICAE、LLMLingua 等基线。

**[UniQuanF: Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models](model_compression/uniquanf_unified_quantization.md)**

:   UniQuanF 统一了均匀量化（UQ,表现力弱但优化性强）和二进制编码量化（BCQ,表现力强但优化性差）的优势，通过统一初始化、局部周期映射和统一定理，实现无额外部署开销的高精度 LLM 量化，在 GSM8K 上提升最高 4.60%。

**[Wanda++: Pruning Large Language Models via Regional Gradients](model_compression/wanda_pruning_large_language_models_via_regional_gradients.md)**

:   提出 Wanda++——基于 decoder block 级别区域梯度的轻量级 LLM 剪枝框架，通过区域梯度评分（RGS）改进剪枝准则 + 区域优化（RO）最小化稠密/稀疏块输出差异，在 2:4 稀疏下 WikiText 困惑度较 Wanda 最高降低 32%，单 H100 GPU 10 分钟内完成 7B 模型剪枝。

**[Who Taught You That? Tracing Teachers in Model Distillation](model_compression/who_taught_you_that_tracing_teachers_in_model_distillation.md)**

:   本文提出"教师模型归因"新问题：给定一个蒸馏后的学生模型，能否从候选教师中识别出其训练教师？发现 n-gram 相似度和困惑度不可靠，但词性（PoS）句法模板能提供有效的教师识别信号。

---

## 🦾 LLM Agent { #llm_agent }

**[Agentic Knowledgeable Self-Awareness](llm_agent/agentic_knowledgeable_self-awareness.md)**

:   本文提出 KnowSelf，一种数据驱动方法，通过在 agent 自探索轨迹上标注特殊 token 来标识不同思维情境（快速思考/慢速思考/知识思考），经两阶段训练（SFT + RPO）使 agent 模型学会自主判断何时需要调用外部知识，以最小知识消耗代价达到最优规划效果。

**[Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools](llm_agent/agentic_reasoning_tools.md)**

:   Agentic Reasoning 提出了一个将 Web 搜索、代码执行和知识图谱记忆（Mind-Map）三种 Agent 工具集成到 LLM 推理过程中的框架，在 DeepSeek-R1 上将 Humanity's Last Exam 准确率从 9.4% 提升到 23.8%（+14.4%），GPQA 从 71.5% 到 81.2%，接近 OpenAI Deep Research 水平。

**[Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems](llm_agent/agentic_reward_modeling_integrating_human_preferences_with_verifiable_correctnes.md)**

:   本文提出Agentic Reward Modeling范式和RewardAgent实现，将传统基于人类偏好的奖励模型与来自事实性验证和指令遵循验证的可验证正确性信号进行整合，通过Router-验证Agent-Judger三模块架构显著提升奖励模型的可靠性。

**[Agents Under Siege: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks](llm_agent/agents_under_siege_breaking_pragmatic_multi-agent_llm_systems_with_optimized_pro.md)**

:   本文首次系统研究了在带宽约束、延迟和安全机制的现实多智能体LLM系统中的对抗攻击问题，提出基于最大流最小费用的拓扑优化和排列不变蒙骗损失（PIEL）的攻击方法，在多个LLM架构上实现了高达7倍于传统攻击的成功率。

**[An Empirical Study on LLM-based Agents for Automated Bug Fixing](llm_agent/an_empirical_study_on_llm-based_agents_for_automated_bug_fixing.md)**

:   本文对SWE-bench Verified上排名前六的LLM-based bug修复系统进行了系统性分析，从整体修复能力、故障定位准确率和缺陷复现有效性三个维度揭示了当前Agent系统的能力边界和改进方向。

**[AndroidGen: Building an Android Language Agent under Data Scarcity](llm_agent/androidgen_agent_data_scarcity.md)**

:   提出 AndroidGen 框架，通过经验检索（ExpSearch）、反思规划（ReflectPlan）、自动校验（AutoCheck）和步骤级评判（StepCritic）四个模块，在高质量训练数据稀缺的条件下增强LLM的Android操作能力，并通过自动生成轨迹数据训练出无需人工标注的开源移动端agent。

**[Auto-TA: Towards Scalable Automated Thematic Analysis (TA) via Multi-Agent Large Language Models with Reinforcement Learning](llm_agent/auto-ta_towards_scalable_automated_thematic_analysis_ta_via_multi-agent_large_la.md)**

:   提出一个基于多智能体 LLM 的全自动主题分析（Thematic Analysis）流水线，通过专业角色分工和可选的 RLHF 微调，实现对临床叙事的端到端主题提取，消除了人工编码和全文审阅的需求。

**[Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](llm_agent/bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)**

:   提出 Bel Esprit 多 Agent 对话框架，通过 Mentalist（需求澄清）→ Builder（管线构建）→ Inspector（验证）→ Matchmaker（模型分配）四步协作，将用户模糊的自然语言需求自动转化为多模型 AI 管线图，在 441 条管线数据上达到 25.2% EM 和 37.0 GED（GPT-4o Builder）。

**[Beyond Numeric Rewards: In-Context Dueling Bandits with LLM Agents](llm_agent/beyond_numeric_rewards_in-context_dueling_bandits_with_llm_agents.md)**

:   系统评估了 LLM 在 Dueling Bandits（偏好反馈强化学习）中的零样本上下文决策能力，发现 GPT-4 Turbo 在弱遗憾（weak regret）上表现出色但强遗憾（strong regret）存在差距，进而提出 LEAD 框架（LLM with Enhanced Algorithmic Dueling），通过将经典 DB 算法与 LLM 智能体细粒度自适应融合来同时获得理论保证和鲁棒性。

**[BookWorld: From Novels to Interactive Agent Societies for Story Creation](llm_agent/bookworld_from_novels_to_interactive_agent_societies_for_story_creation.md)**

:   BookWorld是首个基于小说的多智能体社会模拟系统，通过从源书籍中提取角色数据和世界观信息来构建交互式虚拟世界，让小说角色在其中自主行动和互动，进而生成创意故事，在75.36%的对比中超越了之前的故事生成方法。

**[Browsing Like Human: A Multimodal Web Agent with Experiential Fast-and-Slow Thinking](llm_agent/browsing_like_human_a_multimodal_web_agent_with_experiential_fast-and-slow_think.md)**

:   本文提出WebExperT框架，模拟人类"快思考与慢思考"的规划模式，并通过从失败中反思的经验学习机制不断改进决策，在Mind2Web基准上取得了监督和无监督设置下的优异表现。

**[Can a Single Model Master Both Multi-turn Conversations and Tool Use? CoALM: A Unified Conversational Agentic Language Model](llm_agent/can_a_single_model_master_both_multi-turn_conversations_and_tool_use_coalm_a_uni.md)**

:   本文提出CoALM（Conversational Agentic Language Model），通过构建融合多轮ReAct推理和复杂API调用的多任务训练数据CoALM-IT，训练出同时擅长传统任务型对话（TOD）和语言Agent（LA）工具调用的统一模型，在MultiWOZ、BFCL V3和API-Bank三个基准上超越GPT-4o等专用模型。

**[Caution for the Environment: Multimodal LLM Agents are Susceptible to Environmental Distractions](llm_agent/caution_environment_gui_agent_distractions.md)**

:   本文首次系统研究了多模态 GUI Agent 对环境干扰（弹窗广告、推荐内容等）的脆弱性，在无恶意攻击的自然场景下，即使最强的 MLLM（包括GPT-4o）也有 20-40% 的概率被环境中的无关内容分散注意力而执行偏离用户目标的操作。

**[Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration](llm_agent/dpt_agent_dual_process.md)**

:   提出 DPT-Agent，首个将双过程理论（Dual Process Theory）系统化地融入语言智能体框架的方法——用有限状态机(FSM)+code-as-policy 作为快速直觉的 System 1，用心智理论(ToM)+异步反思的 LLM 作为慢速深思的 System 2，首次实现了自主的实时同步人机协作（在 Overcooked 困难版中）。

**[EMULATE: A Multi-Agent Framework for Determining the Veracity of Atomic Claims by Emulating Human Actions](llm_agent/emulate_a_multi-agent_framework_for_determining_the_veracity_of_atomic_claims_by.md)**

:   提出 EMULATE 多智能体事实核查框架，通过 7 个专职 LLM agent 模拟人类验证声明的完整行为链（搜索→排序→内容评估→证据充分性判断→分类），在三个事实核查 benchmark 上的 Macro-F1 和 Weighted-F1 均超越现有方法。

**[Enhancing LLM Agent Safety via Causal Influence Prompting](llm_agent/enhancing_llm_agent_safety_via_causal_influence_prompting.md)**

:   提出 CIP（Causal Influence Prompting），利用因果影响图（CID）结构化表征 LLM Agent 的决策因果关系，通过初始化 CID→CID 引导交互→迭代更新 CID 三步流程，有效增强 Agent 在代码执行和移动设备控制任务中的安全性。

**[Explorer: Scaling Exploration-Driven Web Trajectory Synthesis for Multimodal Web Agents](llm_agent/explorer_scaling_exploration-driven_web_trajectory_synthesis_for_multimodal_web_.md)**

:   提出 Explorer——一个可扩展的多智能体 pipeline，通过自主网页探索和逐步精炼来合成大规模多模态 web 轨迹数据集（94K 成功轨迹，49K+ URL，720K 截图），训练的 Explorer-7B 在 Mind2Web-Live、MiniWob++ 等 benchmark 上达到甚至超过 GPT-4 水平。

**[FACT-AUDIT: An Adaptive Multi-Agent Framework for Dynamic Fact-Checking Evaluation of Large Language Models](llm_agent/fact_audit_factcheck.md)**

:   提出FACT-AUDIT——一个基于重要性采样和多智能体协作的自适应动态事实核查评估框架，通过动态生成测试数据、迭代探测模型弱点、并同时评估verdict预测和justification质量，全面审计LLM的事实核查能力边界。

**[GeAR: Graph-enhanced Agent for Retrieval-augmented Generation](llm_agent/gear_graph-enhanced_agent_for_retrieval-augmented_generation.md)**

:   GeAR 通过图扩展机制（SyncGE）增强传统检索器的多跳发现能力，并结合 Gist Memory 代理框架实现多步检索推理，在 MuSiQue 等多跳QA数据集上超过现有SOTA 10%+，同时消耗更少的 token 和迭代次数。

**[GUI Agents: A Survey](llm_agent/gui_agents_a_survey.md)**

:   本文对基于大型基础模型的图形用户界面（GUI）代理进行了全面综述，提出了涵盖感知、推理、规划、执行四大能力的统一分析框架，系统梳理了 GUI Agent 的基准测试、评估指标、架构设计和训练方法，并讨论了关键挑战和未来方向。

**[GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](llm_agent/gui_explorer_autonomous.md)**

:   提出 GUI-explorer，一个无需训练的 GUI agent，通过自主探索收集功能感知的交互轨迹，并以无监督方式从状态转换三元组中挖掘 transition-aware 知识，在 SPA-Bench 和 AndroidWorld 上分别达到 53.7% 和 47.4% 的任务成功率。

**[GUICourse: From General Vision Language Model to Versatile GUI Agent](llm_agent/guicourse_from_general_vision_language_model_to_versatile_gui_agent.md)**

:   本文提出GUICourse——一套用于从通用视觉语言模型（VLM）训练多功能GUI代理的数据集系列（GUIEnv/GUIAct/GUIChat），通过两阶段训练流程先增强OCR和定位能力、再注入GUI知识，使得3.1B参数的小模型也能在网页和手机GUI导航任务上取得有效表现。

**[GuideBench: Benchmarking Domain-Oriented Guideline Following for LLM Agents](llm_agent/guidebench_guideline_following.md)**

:   提出 GuideBench 基准测试，系统评估 LLM 在领域导向指南遵循方面的能力，覆盖 7 个任务类别共 1272 个实例，从规则遵循、规则更新鲁棒性和人类偏好对齐三个维度评估 18 个 LLM，发现当前模型在复杂领域规则遵循上仍有较大提升空间。

**[Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement](llm_agent/gödel_agent_a_self-referential_agent_framework_for_recursive_self-improvement.md)**

:   提出 Gödel Agent，一种受 Gödel Machine 启发的自引用 Agent 框架，能通过 Python monkey patching 在运行时读取和修改自身代码（包括修改自身的修改逻辑），实现递归自改进，在 DROP/MGSM/MMLU/GPQA 上超越手工设计和元学习优化的 Agent。

**[iAgent: LLM Agent as a Shield between User and Recommender Systems](llm_agent/iagent_llm_agent_as_a_shield_between_user_and_recommender_systems.md)**

:   提出用户-Agent-平台三层范式，在用户和推荐系统之间插入 LLM Agent 作为保护层，通过指令解析、知识获取、重排序和动态用户画像实现个性化推荐，在四个数据集上平均提升 16.6%，同时有效缓解回音室效应和低活跃用户的不公平问题。

**[LegalAgentBench: Evaluating LLM Agents in Legal Domain](llm_agent/legalagentbench_evaluating_llm_agents_in_legal_domain.md)**

:   提出 LegalAgentBench，一个面向中国法律领域的 LLM Agent 综合评测基准，包含 17 个真实语料库、37 个工具和 300 个覆盖多跳推理与写作的任务，通过关键词匹配和过程进度率实现细粒度评估。

**[Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models](llm_agent/llm_agent_image_classification.md)**

:   提出 Conditional Concept Bottleneck Models (CoCoBMs) 和 LLM-driven Concept Agent 框架，通过类别条件化的概念评分机制和基于环境反馈的动态概念库优化，在 6 个数据集上提升分类准确率 6% 的同时将可解释性提升约 30%。

**[LLM Agents Making Agent Tools](llm_agent/llm_agents_making_agent_tools.md)**

:   本文提出ToolMaker，一个自主将GitHub代码仓库转化为LLM兼容工具的代理框架，给定一个仓库URL和任务描述即可自动安装依赖、生成调用代码并通过闭环自修复机制调试，在涵盖多个领域15个复杂任务的新基准上正确实现了80%的任务，大幅超越了现有软件工程代理。

**[LocAgent: Graph-Guided LLM Agents for Code Localization](llm_agent/locagent_graph-guided_llm_agents_for_code_localization.md)**

:   LocAgent 将代码库解析为有向异构图（含 contain/import/invoke/inherit 四种关系），并设计统一工具（SearchEntity/TraverseGraph/RetrieveEntity）引导 LLM Agent 进行多跳推理，实现高精度代码定位，在文件级达到 92.7% 准确率，同时通过微调开源模型将成本降低 86%。

**[MAM: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis via Role-Specialized Collaboration](llm_agent/mam_modular_multi-agent_framework_for_multi-modal_medical_diagnosis_via_role-spe.md)**

:   提出模块化多智能体框架 MAM，将医学诊断过程分解为全科医生、专科团队、放射科医生、医学助手和主任医师五个角色，通过角色专业化协作实现多模态（文本/图像/音频/视频）医学诊断，在多个公开数据集上比基线模型提升 18%~365%。

**[Adaptive Tool Use in Large Language Models with Meta-Cognition Trigger](llm_agent/meco_metacognition_tool_use.md)**

:   提出 MeCo（Meta-Cognition Trigger），通过表示工程从 LLM 内部提取"元认知信号"——模型对自身能力的自我评估——来自适应决定是否需要调用外部工具，无需微调且计算开销极小，在多个骨干模型和基准上显著改善工具使用决策的准确性。

**[MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis](llm_agent/meddxagent_a_unified_modular_agent_framework_for_explainable_automatic_different.md)**

:   提出 MEDDxAgent 框架，通过中央编排器 DDxDriver 协调病史采集模拟器、知识检索智能体和诊断策略智能体三个模块进行迭代式鉴别诊断（DDx），在交互式诊断场景下实现超过 10% 的准确率提升，同时提供完整的推理可解释性。

**[METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](llm_agent/metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)**

:   提出 METAL，一个基于 VLM 的多智能体框架，将图表生成任务（chart-to-code）分解为生成、视觉评审、代码评审和修订四个专门化智能体的迭代协作，在 ChartMIMIC 基准上比现有最佳方法提升 5.2% F1，并展现了测试时缩放（test-time scaling）现象。

**[MetaSynth: Meta-Prompting-Driven Agentic Scaffolds for Diverse Synthetic Data Generation](llm_agent/metasynth_meta-prompting-driven_agentic_scaffolds_for_diverse_synthetic_data_gen.md)**

:   提出 MetaSynth，利用元提示（meta-prompting）驱动的多智能体协作框架生成高多样性的合成数据，仅用 25M tokens 合成数据（不混合真实数据）就能将 Mistral-7B 成功适配到金融和生物医学领域，分别提升 4.08% 和 13.75%，同时不损害通用能力。

**[MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection](llm_agent/mind_a_multi-agent_framework_for_zero-shot_harmful_meme_detection.md)**

:   提出 MIND 框架，通过相似样本检索、双向洞察推导和多智能体辩论三个阶段实现零样本有害梗图（meme）检测，无需标注数据即可在三个数据集上超越现有零样本方法，并在不同模型架构和规模上展现强泛化性。

**[A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems](llm_agent/multi_agent_dialect_bias_privacy_qa.md)**

:   构建 Dialect Agent（方言翻译+审查）与 Privacy Policy Agent（领域回答）的双 Agent 迭代协作框架，通过注入方言语言学知识的提示工程，在无需重训练的前提下同时提升隐私政策 QA 的整体准确率和跨方言公平性。

**[MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents](llm_agent/multiagentbench_evaluating_the_collaboration_and_competition_of_llm_agents.md)**

:   提出 MultiAgentBench 基准测试和 MARBLE 框架，系统评估 LLM 多智能体系统在协作与竞争场景中的表现，包含 6 种交互场景（研究、Minecraft、数据库、编程、讨价还价、狼人杀），引入基于里程碑的 KPI 指标和协调评分，发现 gpt-4o-mini 整体任务分最高、图结构协调协议在研究场景中表现最佳、认知规划可提升里程碑达成率 3%。

**[Multiple LLM Agents Debate for Equitable Cultural Alignment](llm_agent/multiple_llm_agents_debate_for_equitable.md)**

:   提出 Multi-Agent Debate 框架，让两个 LLM agent 围绕文化场景进行辩论并由 judge LLM 仲裁，在 NormAd-eti 基准上显著提升文化适应准确率和跨文化群体公平性，使 7-9B 小模型达到 27B 模型的性能水平。

**[NexusSum: Hierarchical LLM Agents for Long-Form Narrative Summarization](llm_agent/nexussum_narrative_summarization.md)**

:   提出 NexusSum，一个三阶段多Agent LLM框架（对话转描述→层次摘要→迭代压缩），无需微调即可处理书籍/电影/电视剧等长叙事文本的摘要生成，在 BookSum 上 BERTScore 提升达 30%。

**[OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents](llm_agent/os-kairos_adaptive_interaction_for_mllm-powered_gui_agents.md)**

:   提出 OS-Kairos，通过协作探测框架标注每步置信度分数并微调进基座模型，使 GUI Agent 能在每步预测置信度、自主决定执行或请求人类干预，在复杂场景下任务成功率 (TSR) 从 OS-Atlas-Pro-7B 的 14.29% 提升到 88.20%，在 AITZ 和 Meta-GUI 基准上也有 24~87% 的绝对提升。

**[OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use](llm_agent/os_agents_a_survey_on_mllm-based_agents_for_general_computing_devices_use.md)**

:   全面综述了基于多模态大语言模型的操作系统 Agent（OS Agents），系统梳理了其基础概念（环境/观察/动作空间）、核心能力（理解/规划/动作落地）、构建方法（基础模型+Agent框架）和评估体系，涵盖 30+ 基础模型和 20+ Agent 框架的分类对比。

**[OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis](llm_agent/os_genesis_gui_agent_trajectory.md)**

:   提出 OS-Genesis，一种交互驱动的 GUI Agent 轨迹合成 pipeline，通过先让 agent 在环境中探索交互再反向推导任务（Reverse Task Synthesis），结合轨迹奖励模型 (TRM) 过滤质量，生成高质量多样化的训练轨迹，在 AndroidWorld 上性能接近翻倍。

**[PaSa: An LLM Agent for Comprehensive Academic Paper Search](llm_agent/pasa_an_llm_agent_for_comprehensive_academic_paper_search.md)**

:   PaSa 是一个基于 LLM 的学术论文搜索智能体，通过自主调用搜索工具、阅读论文和导航引用网络来实现全面准确的学术文献检索，经 RL 训练后在真实场景中大幅超越 Google Scholar 和 GPT-4o。

**[Play2Prompt: Zero-shot Tool Instruction Optimization for LLM Agents via Tool Play](llm_agent/play2prompt_zero-shot_tool_instruction_optimization_for_llm_agents_via_tool_play.md)**

:   提出 Play2Prompt，通过让 LLM 自主"玩"工具（试探输入输出行为）来零样本地生成工具使用示例和优化工具文档，无需任何标注数据即可显著提升 LLM agent 的工具调用能力。

**[R2D2: Remembering, Replaying and Dynamic Decision Making with a Reflective Agentic Memory](llm_agent/r2d2_reflective_agentic_memory.md)**

:   R2D2 提出了一个结合 Remember（经验回放缓冲区 + A* 搜索导航）和 Reflect（错误反思 + 反思记忆存储）两范式的 Web Agent 框架，将 Web 导航从 Unknown MDP 转化为 Known MDP，在 WebArena 上导航错误减少 50%，任务完成率提升 3 倍，超越 SOTA 17%。

**[REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](llm_agent/repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)**

:   本文提出 REPRO-Bench，一个包含 112 个社会科学论文实例的基准，用于评估 AI Agent 自动化评估论文可重复性的能力；现有最佳 Agent 准确率仅 21.4%（低于随机猜测的 25%），作者进一步开发的 REPRO-Agent 将准确率提升至 36.6%（71% 相对提升）。

**[REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science?](llm_agent/repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_social_science_.md)**

:   提出 REPRO-Bench，包含 112 个社会科学论文的可复现性评估任务，发现现有 AI Agent（最高准确率仅 21.4%）远不足以自动化该流程，并据此开发 REPRO-Agent 将准确率提升至 36.6%。

**[Select, Read, and Write: A Multi-Agent Framework of Full-Text-based Related Work Generation](llm_agent/select_read_and_write_a_multi-agent_framework_of_full-text-based_related_work_ge.md)**

:   提出 Select-Read-Write 三 Agent 协同框架，通过图感知的阅读顺序决策和共享工作记忆机制，实现基于论文全文（而非摘要）的 Related Work 自动生成，在 Llama3-8B / Claude-3-Haiku / GPT-4o 三个基座模型上均取得一致提升，Citation Graph 策略效果最优。

**[Self-Taught Agentic Long-Context Understanding](llm_agent/self_taught_agentic_long_ctx.md)**

:   提出 AgenticLU 框架，通过 Chain-of-Clarifications (CoC) 工作流让 LLM 自主生成澄清问题并检索相关上下文，再通过 SFT+DPO 两阶段微调将树搜索路径蒸馏到模型中，使 8B 模型在 128K 长上下文 QA 任务上大幅超越基线。

**[SMART: Self-Aware Agent for Tool Overuse Mitigation](llm_agent/smart_self-aware_agent_for_tool_overuse_mitigation.md)**

:   揭示 LLM Agent 中的"工具过度使用"现象（≥30% 的工具调用是不必要的），提出 SMART 元认知范式，通过在推理步骤中显式标注"知识驱动 vs 工具依赖"来训练 Agent 的自感知能力，7B 模型匹配 GPT-4o 水平。

**["sudo rm -rf agentic_security" | SUDO: Screen-based Universal Detox2tox Offense](llm_agent/sudo_rm_-rf_agentic_security.md)**

:   提出SUDO攻击框架，通过Detox2tox三阶段流水线将恶意请求伪装为无害指令再恢复攻击载荷，配合基于检查清单反馈的动态迭代优化，系统性攻破Claude CUA、MANUS等计算机使用Agent的安全防护，最高达41.33%攻击成功率。

**[SynWorld: Virtual Scenario Synthesis for Agentic Action Knowledge Refinement](llm_agent/synworld_agentic_action_knowledge.md)**

:   SynWorld 提出让 Agent 在合成的虚拟场景中通过蒙特卡洛树搜索（MCTS）来探索和优化动作知识（工具描述和工作流），使 Agent 能够自主适应新环境的工具使用，在 ToolBench 上比 ReAct 基线提升约 9 个百分点。

**[Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning](llm_agent/table_critic_multi_agent.md)**

:   提出 Table-Critic 多智能体框架，通过 Judge-Critic-Refiner-Curator 四个专门化 Agent 的协作批评与迭代精化，配合自进化模板树累积批评知识，在 WikiTQ 和 TabFact 上分别实现 73.7% 和 91.7% 的准确率，大幅超越现有方法。

**[The Behavior Gap: Evaluating Zero-shot LLM Agents in Complex Task-Oriented Dialogs](llm_agent/the_behavior_gap_evaluating_zero-shot_llm_agents_in_complex_task-oriented_dialog.md)**

:   提出综合评估框架量化 LLM agent 与人类专家在任务导向对话中的"行为差距"，从 dialog acts、工具使用、知识利用三个维度系统诊断行为偏差，发现行为差距与任务复杂度高度相关（$r=0.963$），通过行为注入缩小差距可平均提升 24.3% 性能。

**[Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models](llm_agent/theorem-of-thought_a_multi-agent_framework_for_abductive_deductive_and_inductive.md)**

:   提出 Theorem-of-Thought (ToTh) 框架，通过三个并行智能体分别模拟溯因、演绎和归纳推理，将推理轨迹构建为形式推理图并利用 NLI 校准的贝叶斯置信传播选出最连贯推理链，在符号和数值推理上一致优于 CoT、Self-Consistency 和 CoT-Decoding。

**[ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](llm_agent/toolhop_multi_hop_tool_use.md)**

:   提出 ToolHop——一个包含 995 个多跳查询和 3912 个本地可执行工具的基准数据集，通过"查询驱动"的数据构建方式（先有查询再造工具）确保工具间有真实依赖关系和可验证答案，评测 14 个 LLM 发现最强的 GPT-4o 准确率仅 49%，揭示了不同模型家族在工具使用上的显著策略差异。

---

## 💡 LLM 推理 { #llm_reasoning }

**[An Efficient and Precise Training Data Construction Framework for Process-Supervised Reward Model in Mathematical Reasoning](llm_reasoning/an_efficient_and_precise_training_data_construction_framework_for_process-superv.md)**

:   本文提出EpicPRM框架，通过基于困惑度（perplexity）的Monte Carlo估计量化每个推理步骤的贡献度，并使用自适应二分搜索高效定位首个错误步骤，构建了高质量的过程监督数据集Epic50k（仅50k标注步骤），训练出的PRM性能可媲美甚至超越在PRM800k上训练的模型。

**[Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework](llm_reasoning/aristotle_logical_reasoning.md)**

:   提出 Aristotle 逻辑推理框架，将符号表达式和逻辑规则全面融入 Decompose-Search-Resolve 的每个阶段，通过逻辑分解器、搜索路由器和消解器三大组件实现逻辑完备的推理，在多个逻辑推理基准上以 GPT-4 平均提升 4.5%、GPT-4o 平均提升 5.4% 超越 SOTA。

**[Beyond the Answer: Advancing Multi-Hop QA with Fine-Grained Graph Reasoning and Evaluation](llm_reasoning/beyond_the_answer_advancing_multi-hop_qa_with_fine-grained_graph_reasoning_and_e.md)**

:   本文针对多跳问答（Multi-hop QA）任务中推理过程不透明、评估粒度粗糙的问题，提出了一种基于细粒度图推理的框架，通过构建推理图来显式建模证据链，并引入细粒度评估指标来衡量推理过程的质量，而非仅关注最终答案的正确性。

**[BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving](llm_reasoning/bpp-search_enhancing_tree_of_thought_reasoning_for_mathematical_modeling_problem.md)**

:   提出 BPP-Search 算法，将 Beam Search、过程奖励模型 (PRM) 和 Pairwise Preference 机制整合到 Tree-of-Thought 框架中，用于运筹学数学建模问题的自动求解，在 StructuredOR 等数据集上以更少的推理步骤显著超越 CoT/SC/ToT 基线。

**[Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?](llm_reasoning/can_large_language_models_detect_errors_in_long_chain-of-thought_reasoning.md)**

:   本文提出DeltaBench——首个系统评估o1类模型长CoT推理质量和现有LLM/PRM错误检测能力的基准数据集，通过对1,236个样本的精细人工标注，揭示了o1类模型约27%推理冗余、67.8%反思无效，以及最强critic模型GPT-4-turbo-128k也仅达F1=40.8%的令人警醒的现状。

**[Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective](llm_reasoning/chain_of_reasoning_unified_math.md)**

:   提出 Chain-of-Reasoning（CoR）框架，将自然语言推理（NLR）、算法推理（AR）和符号推理（SR）三种范式统一在一个推理链中，通过渐进范式训练（PPT）策略让 7B 模型（CoR-Math-7B）在零样本下超越 GPT-4o 41% 的定理证明准确率，在 MATH 基准上超过 RL 方法 15%。

**[ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations](llm_reasoning/clozemath_improving_mathematical_reasoning_in_language_models_by_learning_to_fil.md)**

:   ClozeMath 提出了一种受人类完形填空学习启发的微调策略，通过掩码数学解答中的方程式并训练模型预测它们（text-infilling目标），与标准语言模型目标联合训练，在GSM8K和MATH上显著超越了强基线Masked Thought，并在推理时间扩展和鲁棒性测试中表现出更好的泛化能力。

**[Commonsense Abductive Reasoning using Knowledge from Multiple Sources](llm_reasoning/commonsense_abductive_reasoning_using_knowledge_from_multiple_sources.md)**

:   本文提出一种融合多源知识（知识图谱、预训练语言模型、规则库）的常识溯因推理方法，通过联合利用结构化和非结构化知识来生成更准确、更可解释的最佳解释，在溯因推理基准上取得了显著提升。

**[Complex Reasoning with Natural Language Contexts and Background Knowledge](llm_reasoning/complex_reasoning_with_natural_language_contexts_and_background_knowledge.md)**

:   本文提出一种将自然语言上下文与结构化背景知识相结合的复杂推理框架，通过知识图谱检索增强和上下文感知的推理链生成，显著提升LLM在需要外部知识支撑的多步推理任务上的表现。

**[CoT-based Synthesizer: Enhancing LLM Performance through Answer Synthesis](llm_reasoning/cot-based_synthesizer_enhancing_llm_performance_through_answer_synthesis.md)**

:   提出 CoT-based Synthesizer——一种新的推理扩展策略，通过利用 CoT 推理分析多个候选回答的互补信息来合成更优的最终答案，即便所有候选回答都是错误的也能综合出正确答案，在 MATH500 上对 Llama3-8B 提升 11.8%、对 GPT-4o 提升 10.3%。

**[CoT-ICL Lab: A Synthetic Framework for Studying Chain-of-Thought Learning from In-Context Demonstrations](llm_reasoning/cot-icl_lab_a_synthetic_framework_for_studying_chain-of-thought_learning_from_in.md)**

:   提出 CoT-ICL Lab 框架，通过解耦因果结构（DAG）和 token 处理函数（MLP）生成可控的合成 token 化数据集，系统研究了 CoT 对 ICL 的加速效应、模型深度的关键作用、以及 Transformer 嵌入与注意力图对底层推理结构的学习机制。

**[CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought](llm_reasoning/cot-uq_improving_response-wise_uncertainty_quantification_in_llms_with_chain-of-.md)**

:   针对 LLM 在推理任务中过度自信的问题，提出 CoT-UQ 框架，将 CoT 推理步骤中的关键词提取和重要性评分整合到不确定性量化过程中，在逻辑和数学推理任务上 AUROC 平均提升 5.9%。

**[CoT-Valve: Length-Compressible Chain-of-Thought Tuning](llm_reasoning/cot-valve_length-compressible_chain-of-thought_tuning.md)**

:   本文提出CoT-Valve，一种通过在参数空间中识别"长度控制方向"（以LoRA实现）来弹性控制推理链长度的方法，仅训练一次即可生成从长到短不同长度的推理路径，在QwQ-32B-Preview上将GSM8K推理链从741压缩至225 tokens且准确率仅降0.15%（95.07%→94.92%）。

**[Critic-CoT: Boosting the Reasoning Abilities of Large Language Model via Chain-of-Thoughts Critic](llm_reasoning/critic-cot_boosting_the_reasoning_abilities_of_large_language_model_via_chain-of.md)**

:   提出 Critic-CoT 框架，通过逐步 Chain-of-Thought 批判范式和无需人工标注的弱监督数据自动构建，将 LLM 的自我批判从 System-1 式直觉判断推向 System-2 式慎重逐步分析；两阶段训练（GPT-4 蒸馏 + 自我批判）使 Llama-3-70B-Instruct 在 GSM8K 从 89.6% 提升至 95.4%，MATH500 从 50.4% 提升至 68.4%，并发现批判能力与任务求解能力可以相互增强。

**[Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs](llm_reasoning/dcot_diverse_cot_refinement.md)**

:   提出 Diverse Chain of Thought (DCoT) 训练方法，通过在单次推理中生成多条串行推理链实现"推理内自修正"（within-inference refinement），在 1.3B–70B 模型上均超越标准 CoT 基线，尤其在大输出空间任务（数值/抽取型）上提升显著。

**[DeFine: Decision-Making with Analogical Reasoning over Factor Profiles](llm_reasoning/define_decision-making_with_analogical_reasoning_over_factor_profiles.md)**

:   提出 DeFine 框架，从财报电话会议等复杂场景的语音转录文本中构建概率因子画像(factor profile)，结合 Bradley-Terry 模型识别关键因子并通过因子画像间的 KL 散度做类比推理，用于辅助 LLM 在不确定性下做投资决策，准确率和 F1 均超越基线。

**[Dynamic and Generalizable Process Reward Modeling (DG-PRM)](llm_reasoning/dgprm_dynamic_process_reward.md)**

:   提出DG-PRM框架，通过构建层次化奖励树动态存储和选择多维评估标准，结合Pareto支配估计识别多目标下的正负样本对，实现动态、可泛化的过程奖励建模。

**[DRT: Deep Reasoning Translation via Long Chain-of-Thought](llm_reasoning/drt_deep_reasoning_translation_via_long_chain-of-thought.md)**

:   将长 CoT 推理引入机器翻译，构建多智能体框架（翻译器→顾问→评估器）迭代精炼含比喻/隐喻的文学翻译，合成 22K 长思维翻译训练样本，训练的 DRT-14B 在文学翻译上超越 QwQ-32B 和 DeepSeek-R1-Distill-32B 等大模型。

**[Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning](llm_reasoning/enhancing_chain-of-thought_reasoning_with_critical_representation_fine-tuning.md)**

:   提出 CRFT 方法，通过信息流分析自动识别 Transformer 各层中对推理输出影响最大的"关键表征"，并在低秩线性子空间中对这些表征进行有监督优化，在仅使用模型 0.016% 参数的情况下，将 LLaMA-2-7B 在 GSM8K 上的准确率提升了 18.2%。

**[Enhancing Mathematical Reasoning in LLMs by Stepwise Correction](llm_reasoning/enhancing_mathematical_reasoning_in_llms_by_stepwise_correction.md)**

:   本文提出StepCo（Stepwise Correction），一种迭代式"验证-修正"框架：利用过程监督验证器（PSV）逐步定位LLM推理路径中的首个错误步骤并触发LLM修正，在8个数学推理基准上以GPT-4o为后端取得94.1%平均准确率，超越Best-of-10方法+2.4个点，同时减少77.8%的token消耗。

**[Enhancing Retrieval Systems with Inference-Time Logical Reasoning](llm_reasoning/enhancing_retrieval_systems_with_inference-time_logical_reasoning.md)**

:   提出推理时逻辑推理框架（ITLR），利用 LLM 将自然语言查询转换为逻辑表达式（AND/OR/NOT），然后基于模糊逻辑对各子项的 cosine similarity 分数进行组合，在合成数据和 NFCorpus/SciFact/ArguAna 三个真实数据集上一致性超越传统 dense retrieval 和 BRIGHT baseline，尤其在含否定的复杂查询上提升显著。

**[Entropy-based Exploration Conduction for Multi-step Reasoning](llm_reasoning/entropy-based_exploration_conduction_for_multi-step_reasoning.md)**

:   提出 Entro-duction 方法，通过监控 LLM 推理过程中输出的熵和方差熵变化来动态调整探索深度，使用 $\epsilon$-greedy 策略选择加深、扩展或停止三种探索行为，在避免冗余推理的同时提升推理准确率。

**[FineReason: Evaluating and Improving LLMs' Deliberate Reasoning through Reflective Puzzle Solving](llm_reasoning/finereason_evaluating_and_improving_llms_deliberate_reasoning_through_reflective.md)**

:   提出 FineReason——一个基于逻辑谜题的推理基准，通过"状态检查"（判断当前状态是否可解）和"状态转换"（决定下一步操作）两个任务，对LLM的审慎推理能力（反思、回溯、纠错）进行原子级粒度评估，并证明在谜题数据上的训练可迁移提升数学推理能力（GSM8K 提升 5.1%）。

**[Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering](llm_reasoning/glore_long_cot_representation.md)**

:   从表示空间角度发现 LLM 将长 CoT 推理编码为一种与普通 CoT 明确区分的通用能力，提出 GLoRE（General Long CoT Reasoning via Representation Engineering）——通过对比推理模式注入和领域特定表示调整来解锁长 CoT 能力，无需训练即可在域内和跨域场景下超越 SFT 方法。

**[Improve Vision Language Model Chain-of-thought Reasoning](llm_reasoning/improve_vlm_cot_reasoning.md)**

:   通过(1)从GPT-4o蒸馏193K多任务CoT推理数据进行SFT，(2)利用模型自生成的推理链构建正负样本对进行DPO强化学习，显著提升VLM的链式推理能力，CoT预测平均+11.7%，同时直接回答也提升+7.3%。

**[Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions](llm_reasoning/improving_chain-of-thought_reasoning_via_quasi-symbolic_abstractions.md)**

:   本文提出QuaSAR（Quasi-Symbolic Abstract Reasoning），一种CoT变体方法，通过引导LLM先对问题进行符号化抽象（提取变量/谓词）、再用半形式化表示重构问题、最后基于准符号推理链求解，在GPT-4o上相比CoT提高最多8%准确率，并显著增强了对对抗性变体（选项打乱、数值替换）的鲁棒性。

**[Large Language and Reasoning Models are Shallow Disjunctive Reasoners](llm_reasoning/large_language_and_reasoning_models_are_shallow_disjunctive_reasoners.md)**

:   本文利用合成的空间与时间推理基准（STaR）评估LLM和LRM在需要组合多条推理路径的析取规则推理任务上的系统性泛化能力，发现即使是o3-mini等推理模型也只能处理单路径推理，在多路径析取推理场景中性能急剧下降。

**[Local Look-Ahead Guidance via Verifier-in-the-Loop for Automated Theorem Proving](llm_reasoning/local_look-ahead_guidance_via_verifier-in-the-loop_for_automated_theorem_proving.md)**

:   提出 LeanListener，在自动定理证明(ATP)中引入 verifier-in-the-loop 设计，利用 Lean 验证器在每步提供中间反馈（子目标数变化）而非仅轨迹级奖励，通过在线 GRPO 训练使 ReProver 的 tactic 有效率和证明率均获提升，证明速度快 20%。

**[LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning](llm_reasoning/logicpro_program_guided_reasoning.md)**

:   提出 LogicPro 数据合成方法，利用 LeetCode 算法题和 Python 代码解作为逻辑源，通过"问题生成→代码中间变量提取→程序引导推理生成"三步流水线，从 2360 道算法题合成 540K 高质量文本推理数据，在 BBH27、LogicBench、DROP 等多个 OOD 基准上显著超越现有推理数据集。

**[Marco-o1 v2: Towards Widening The Distillation Bottleneck for Reasoning Models](llm_reasoning/marco-o1_v2_towards_widening_the_distillation_bottleneck_for_reasoning_models.md)**

:   揭示了直接蒸馏大推理模型（如 DeepSeek-R1）的长 CoT 数据到小模型时的「形式化长时间思考」瓶颈，提出基于 MCTS 从头构造树状 CoT 数据并结合思维长度平衡、细粒度 DPO 和联合训练目标来缓解该问题。

**[Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning](llm_reasoning/mclm_multilingual_test_time_scaling.md)**

:   提出 MCLM（55 语言的竞赛级数学基准），发现三种 test-time scaling 方法（ORM/PRM/Budget Forcing）在英语上提升显著（如 AIME +20 分），但在其他语言上平均仅提升 1.94 分，表明 test-time scaling 的多语言泛化能力严重不足。

**[MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification](llm_reasoning/mm-verify_enhancing_multimodal_reasoning_with_chain-of-thought_verification.md)**

:   本文提出MM-Verifier和MM-Reasoner两个模型，通过模拟搜索+拒绝采样合成长链CoT验证数据、以及文本蒸馏合成多模态推理数据，仅7B参数即在MathVista上达到65.3%准确率超越GPT-4o（63.8%）和人类表现（60.3%）。

**[On Generalization across Measurement Systems: LLMs Entail More Test-Time Compute for Underrepresented Cultures](llm_reasoning/on_generalization_across_measurement_systems_llms_entail_more_test-time_compute_.md)**

:   系统研究 LLM 跨度量系统（货币、长度、重量）的泛化能力，发现模型默认使用训练数据中的主导度量（如美元、公制），对非主导度量查询准确率显著下降；CoT 推理可弥补但推理成本增加高达 300%，对欠代表文化用户构成系统性不公平。

**[One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL](llm_reasoning/one_missing_piece_for_open-source_reasoning_models_a_dataset_to_mitigate_cold-st.md)**

:   提出 Long CoT Collection——一个由短链式思维 LLM（如 GPT-4o）标注的 100K 长链式推理数据集，通过从 o1 提取推理流程作为引导，使短 CoT LLM 也能生成长 CoT 数据，从而解决强化学习中的冷启动问题，训练在该数据上初始化的模型在后续 RL 中获得 2-3 倍的性能提升。

**[PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation](llm_reasoning/pcot_persuasion-augmented_chain_of_thought_for_detecting_fake_news_and_social_me.md)**

:   将心理学中"学会识别说服技巧可提升真伪判断力"的发现迁移到 LLM，提出两阶段零样本 PCoT 方法：第一阶段识别并分析六类说服策略，第二阶段将分析上下文融入虚假信息检测，在 5 个 LLM × 5 数据集上 F1 平均提升 15%。

**[ProcessBench: Identifying Process Errors in Mathematical Reasoning](llm_reasoning/processbench_identifying_process_errors_in_mathematical_reasoning.md)**

:   本文提出ProcessBench基准（3,400个测试用例，以竞赛/奥赛级数学题为主），评估PRM和critic模型定位数学推理中最早错误步骤的能力，发现现有PRM无法泛化到超出GSM8K/MATH的难题，而通用LLM（如QwQ-32B-Preview）作为critic的表现可与GPT-4o媲美。

**[Ranked Voting based Self-Consistency of Large Language Models](llm_reasoning/ranked_voting_based_self-consistency_of_large_language_models.md)**

:   将 Self-Consistency 的多数投票升级为排序投票，让 LLM 每次推理生成多个候选答案的偏好排序而非单一答案，用三种排序投票方法（IRV/BCV/MRRV）聚合多次推理的排序信息，在 6 个数据集上一致超越传统 SC，最高提升 12.46%。

**[Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory](llm_reasoning/rethinking_the_role_of_prompting_strategies_in_llm_test-time_scaling_a_perspecti.md)**

:   本文在 6 个 LLM × 8 种 prompting 策略 × 6 个 benchmark 上系统实验发现，随着 majority voting 采样次数增加，简单的 CoT 始终超越复杂 prompting 策略；并从概率论角度给出理论证明，提出 $O(1)$ 复杂度的 scaling 性能预测方法和两种改进策略。

**[Revisiting Self-Consistency from Dynamic Distributional Alignment Perspective on Answer Aggregation](llm_reasoning/revisiting_self-consistency_from_dynamic_distributional_alignment_perspective_on.md)**

:   将 Self-Consistency 重新理解为采样分布与真实答案分布的动态对齐问题，揭示温度不仅控制采样随机性还直接塑造真实答案分布，据此提出置信度驱动的三阶段动态温度调节机制（FSD 阈值理论推导），在 10 个模型 × GSM8K/MATH 上零训练开销同时提升平均和最佳性能。

**[Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities?](llm_reasoning/revisiting_the_test-time_scaling_of_o1-like_models_do_they_truly_possess_test-ti.md)**

:   系统性地揭示了 QwQ/DeepSeek-R1/LIMO 等 o1-like 模型在测试时并不具备真正的顺序扩展 (sequential scaling) 能力——更长的 CoT 并不带来更高准确率，根因是自我修正 (self-revision) 能力不足——并据此提出 Shortest Majority Vote 并行扩展方法显著超越传统多数投票。

**[RSVP: Reasoning Segmentation via Visual Prompting and Multi-modal Chain-of-Thought](llm_reasoning/rsvp_reasoning_segmentation_via_visual_prompting_and_multi-modal_chain-of-though.md)**

:   本文提出 RSVP 框架，通过两阶段结构（推理驱动定位 + 分割精炼）将多模态大模型的推理能力与视觉分割相统一，利用多模态思维链视觉提示在 ReasonSeg 上超越 SOTA 达 +6.5 gIoU / +9.2 cIoU，零样本 SegInW 达到 49.7 mAP。

**[Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification](llm_reasoning/safe_math_reasoning.md)**

:   提出 Safe 框架，首次利用 Lean 4 形式化语言对 LLM 数学推理的每一步进行回顾性逐步验证，通过自动形式化+自动定理证明检测幻觉，并与前瞻性 PRM 分数融合，在多个数学数据集上取得 SOTA，同时发布包含 30,809 条形式化声明的 FormalStep 基准。

**[Self-Correction is More than Refinement: A Learning Framework for Visual and Language Reasoning Tasks](llm_reasoning/self-correction_is_more_than_refinement_a_learning_framework_for_visual_and_lang.md)**

:   提出 Self-Correction Learning (SCL)，通过将 VLM 自身产生的自纠正数据（成功和失败的纠正样本）分类为偏好/非偏好对，利用 DPO 进行偏好微调，从根本上提升模型直接生成正确答案的能力，而非仅仅依赖推理时的迭代修正。

**[Self-Error-Instruct: Generalizing from Errors for LLMs Mathematical Reasoning](llm_reasoning/self-error-instruct_generalizing_from_errors_for_llms_mathematical_reasoning.md)**

:   提出 Self-Error-Instruct (SEI) 框架，通过分析目标模型在数学推理中的错误案例，使用 GPT-4o 提取错误关键词并聚类得到错误类型，然后针对每种错误类型采用 self-instruct 方式合成训练数据，迭代微调模型以系统性地补齐数学推理能力短板。

**[SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs](llm_reasoning/softcot_soft_chain_of_thought.md)**

:   提出 SoftCoT，用一个冻结的小型辅助模型（如 LLaMA-3.2-1B）生成实例特定的"软思维 token"（连续隐状态），通过可训练的投影模块映射到主 LLM 的表示空间作为推理前缀，实现参数高效的连续空间 CoT 推理，避免了全模型微调导致的灾难性遗忘问题。

**[STRICTA: Structured Reasoning in Critical Text Assessment for Peer Review and Beyond](llm_reasoning/stricta_structured_reasoning_peer_review.md)**

:   提出 STRICTA 框架，将专家文本评估（如论文审稿）建模为基于结构因果模型（SCM）的逐步推理图，收集 40+ 生物医学专家对 22 篇论文的 4000+ 推理步骤数据，揭示先验知识差异是评审分歧的主因、写作风格对终审影响过大，并发现 LLM 在人工监督下可有效辅助结构化评估。

**[Is That Your Final Answer? Test-Time Scaling Improves Selective Question Answering](llm_reasoning/test_time_scaling_selective_qa.md)**

:   首次评估 test-time scaling 模型（DeepSeek R1、S1）在选择性问答（允许拒绝回答）场景下的表现，发现增加推理计算不仅提高正确率还提升对正确答案的置信度，提出基于置信度阈值的选择函数和 "Jeopardy Odds" 效用函数来评估非零错误惩罚下的 test-time scaling 性能。

**[ThinkGuard: Deliberative Slow Thinking Leads to Cautious Guardrails](llm_reasoning/thinkguard_deliberative_slow_thinking_leads_to_cautious_guardrails.md)**

:   通过从 GPT-4o/DeepSeek-R1 蒸馏结构化批判（安全标签+详细推理理由），微调护栏模型实现"慢思考"式安全判断，在 4 个安全 benchmark 上达到最高平均 F1（75.5%）和 AUPRC（79.5%），相比 LLaMA Guard 3 准确率提升 16.1%、宏 F1 提升 27.0%。

**[Towards Better Chain-of-Thought: A Reflection on Effectiveness and Faithfulness](llm_reasoning/towards_better_chain-of-thought_a_reflection_on_effectiveness_and_faithfulness.md)**

:   从有效性和忠实性两个视角系统分析 CoT 的表现模式：发现问题难度、信息增益和信息流单调性决定 CoT 有效性，并揭示不忠实 CoT 的机制——模型在预测答案时从问题中召回了 CoT 遗漏的正确信息。在此基础上提出 QUIRE 算法，同时提升 CoT 的有效性（+2.4%）和忠实性（+5.6%）。

**[Towards Safety Reasoning in LLMs: AI-agentic Deliberation for Policy-embedded CoT Data Creation](llm_reasoning/towards_safety_reasoning_in_llms_ai-agentic_deliberation_for_policy-embedded_cot.md)**

:   提出 AIDsafe 多智能体迭代审议框架，自动生成嵌入安全策略的高质量 CoT 数据，微调后的模型在安全泛化和越狱鲁棒性上显著优于传统安全训练，同时引入 ear-whisperer agent 解决 DPO 偏好数据中 selected/rejected 难以区分的问题。

**[TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning](llm_reasoning/tract_regression_cot.md)**

:   提出 TRACT，一种两阶段回归感知微调方法，将 CoT 推理与回归损失（squared error）结合，用于提升 LLM-as-a-Judge 场景中的数值评分精度，显著优于仅用交叉熵训练或仅用回归损失的现有方案。

**[Training Turn-by-Turn Verifiers for Dialogue Tutoring Agents: The Curious Case of LLMs as Your Coding Tutors](llm_reasoning/training_turn-by-turn_verifiers_for_dialogue_tutoring_agents_the_curious_case_of.md)**

:   提出 **Traver**（Trace-and-Verify）agent 工作流，通过**知识追踪**显式估计学生知识状态 + **逐轮验证器**（turn-by-turn verifier）对候选辅导话语打分选优，并设计 **Dict** 自动评估协议（模拟学生 + 代码生成测试），在编程辅导场景中将学生 Pass 率从 38.7% 提升至 43.7%（相对提升 106.5%），显著超越 Vanilla Instruct、Self-Refine 和 TreeInstruct。

**[Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning](llm_reasoning/unveiling_the_key_factors_for_distilling_chain-of-thought_reasoning.md)**

:   系统研究影响 CoT 蒸馏的三大因素（粒度、格式、教师模型），发现 SLM 与粒度呈非单调关系、格式影响较小、强教师不总是更好。

---

## 🎵 音频/语音 { #audio_speech }

**[Finding A Voice: Exploring the Potential of African American Dialect and Voice Generation for Chatbots](audio_speech/aae_voice_chatbot.md)**

:   对文本和语音两种模态下将非裔美式英语（AAE）融入聊天机器人进行系统研究，发现文本AAE反而损害用户体验，但配合非裔口音的语音机器人受到AAE使用者青睐，揭示了语言个性化中模态选择的关键作用。

**[Acoustic Individual Identification of White-Faced Capuchin Monkeys Using Joint Multi-Species Embeddings](audio_speech/acoustic_individual_identification_of_white-faced_capuchin_monkeys_using_joint_m.md)**

:   本文探索利用来自鸟类和人类的跨物种声学预训练嵌入来识别白面卷尾猴的个体叫声，发现联合多物种表征能进一步提升识别性能，为数据极度稀缺的野生动物个体识别提供了新的迁移学习范式。

**[Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment](audio_speech/advancing_zero-shot_text-to-speech_intelligibility_across_diverse_domains_via_pr.md)**

:   提出INTP（Intelligibility Preference Speech Dataset）数据集和面向多种TTS架构的DPO扩展方法，通过偏好对齐显著提升零样本TTS系统在绕口令、重复词、中英混合、跨语言等挑战场景下的可懂度，并验证了弱模型到强模型的泛化能力。

**[AI4Reading: Chinese Audiobook Interpretation System Based on Multi-Agent Collaboration](audio_speech/ai4reading_chinese_audiobook_interpretation_system_based_on_multi-agent_collabor.md)**

:   提出 AI4Reading，一个基于 11 个专业化 LLM Agent 协作的中文有声书解读系统，通过主题分析、案例扩展、编辑润色、口语化改写和整合修订等阶段自动生成解读稿，并用 TTS 合成音频，在解读脚本质量（简洁性、完整性、准确性、连贯性）上超过专业人工解读平台樊登读书。

**[Amplifying Trans and Nonbinary Voices: A Community-Centred Harm Taxonomy for LLMs](audio_speech/amplifying_trans_and_nonbinary_voices_a_community-centred_harm_taxonomy_for_llms.md)**

:   本文采用社区导向（community-centred）的研究方法，通过与跨性别和非二元性别（Trans and Nonbinary, TNB）群体的深入合作，构建了一套专门针对LLM输出中对TNB群体伤害的分类体系（harm taxonomy），揭示了现有LLM安全评估未覆盖的TNB特有伤害类型。

**[ATRI: Mitigating Multilingual Audio Text Retrieval Inconsistencies by Reducing Data Distribution Errors](audio_speech/atri_mitigating_multilingual_audio_text_retrieval_inconsistencies_by_reducing_da.md)**

:   从理论上分析多语言音频文本检索（ML-ATR）中跨语言不一致性的根本原因是训练数据分布误差，并提出 1-to-K 对比学习（KCL）和音频-英语共锚对比学习（CACL）两种策略来减少该误差，在召回率和一致性上达到 SOTA。

**[Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models](audio_speech/audio_dialogue_benchmark.md)**

:   本文提出 ADU-Bench，一个包含 4 个子数据集（通用对话、技能、多语言、歧义处理）共 20,000+ 开放式音频对话的综合基准，系统评估 16 个大型音频语言模型（LALM）在音频对话理解上的能力，揭示现有模型在数学公式理解、角色扮演、多语言和语音歧义处理上的显著不足。

**[Analyzing and Mitigating Inconsistency in Discrete Audio Tokens for Neural Codec Language Models](audio_speech/audio_token_consistency.md)**

:   本文揭示并量化分析了神经音频编解码器中的离散表示不一致性（DRI）问题——相同音频片段因上下文不同被编码为不同离散token序列，提出切片一致性和扰动一致性两种约束方法，将一致性平均提升21-36%，并在VALL-E语音生成中将WER降低3.72%。

**[Autoregressive Speech Synthesis without Vector Quantization](audio_speech/autoregressive_speech_synthesis_without_vq.md)**

:   MELLE 提出了一种基于连续 mel-spectrogram 帧的自回归语言模型 TTS 方法，通过回归损失 + 变分推断采样模块 + spectrogram flux loss 直接预测连续频谱帧，避免了向量量化带来的保真度损失和采样鲁棒性问题，单阶段模型即可达到与人类水平相当的语音合成质量。

**[Chain-Talker: Chain Understanding and Rendering for Empathetic Conversational Speech Synthesis](audio_speech/chain-talker_chain_understanding_and_rendering_for_empathetic_conversational_spe.md)**

:   提出 Chain-Talker，通过三阶段链式建模（情感理解→语义理解→共情渲染）实现可解释的共情对话语音合成，并开发 CSS-EmCap 自动标注管道为对话语音生成情感描述。

**[CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages](audio_speech/clamp_3_universal_music_information_retrieval_across_unaligned_modalities_and_un.md)**

:   提出 CLaMP 3 统一框架，通过对比学习将乐谱、演奏信号、音频录音与多语言文本对齐到共享表示空间，在无配对训练数据的模态间实现跨模态检索，并展现出对未见语言的强泛化能力。

**[Contextual Biasing with the Knowledgeable External Language Model for End-to-End Speech Recognition](audio_speech/contextual_biasing_with_the_knowledgeable_external_language_model_for_end-to-end.md)**

:   本文提出利用知识增强的外部语言模型（KELM）进行上下文偏置，在端到端语音识别中通过动态融合外部领域知识和偏置词表，大幅提升稀有词和专有名词的识别准确率。

**[ControlSpeech: Towards Simultaneous and Independent Zero-shot Speaker Cloning and Zero-shot Language Style Control](audio_speech/controlspeech_zero_shot.md)**

:   ControlSpeech 是首个同时实现零样本音色克隆和零样本语言风格控制的TTS系统，通过离散编解码器空间中的解耦表示和风格混合语义密度（SMSD）模块解决了风格控制中的多对多问题。

**[Dialectal Coverage and Generalization in Arabic Speech Recognition](audio_speech/dialectal_coverage_and_generalization_in_arabic_speech_recognition.md)**

:   系统研究阿拉伯语方言覆盖对 ASR 性能的影响，通过多方言预训练和联合微调扩展 ArTST 模型覆盖 17 个阿拉伯国家的语音变体，并探索了代码切换场景下的多语言优化策略。

**[Different Speech Translation Models Encode and Translate Speaker Gender Differently](audio_speech/different_speech_translation_models_encode_and_translate_speaker_gender_differen.md)**

:   通过注意力探针分析不同架构的语音翻译模型如何编码说话人性别信息，发现传统编码器-解码器模型能较好保留性别信息，而新型 speech+MT 架构的适配器会显著擦除性别信息，导致翻译中出现更严重的阳性默认偏差。

**[Distilling an End-to-End Voice Assistant Without Instruction Training Data](audio_speech/distilling_an_end-to-end_voice_assistant_without_instruction_training_data.md)**

:   提出DiVA（Distilled Voice Assistant），通过将文本LLM对转录文本的响应作为自监督信号进行跨模态蒸馏，无需任何语音指令训练数据即可训练端到端语音LLM——仅用3.5k小时ASR数据就泛化到口语问答、分类和翻译任务，且在用户偏好测试中以72%胜率碾压Qwen 2 Audio（使用100倍以上训练计算量）。

**[DNCASR: End-to-End Training for Speaker-Attributed ASR](audio_speech/dncasr_end-to-end_training_for_speaker-attributed_asr.md)**

:   提出 DNCASR，一种端到端可训练的说话人归因 ASR 系统，通过链接神经聚类解码器和 ASR 解码器，联合训练生成带说话人标识的转录文本，在 AMI 会议数据上实现 cpWER 9.0% 的相对降低。

**[Does Your Voice Assistant Remember? Analyzing Conversational Context Recall and Utilization in Voice Interaction Models](audio_speech/does_your_voice_assistant_remember_analyzing_conversational_context_recall_and_u.md)**

:   系统性评估开源语音交互模型的对话历史回忆能力，提出 ContextDialog 基准，发现这些模型在回忆过去语音信息方面远弱于文本模型，且 RAG 方法也难以有效弥补这一差距。

**[Double Entendre: Robust Audio-Based AI-Generated Lyrics Detection via Multi-View Fusion](audio_speech/double_entendre_robust_audio-based_ai-generated_lyrics_detection_via_multi-view_.md)**

:   提出 DE-detect，一个仅以音频为输入的多视角晚期融合管线，通过结合自动转录歌词的文本特征和语音模型提取的歌词相关音频特征，实现了对 AI 生成歌词的鲁棒检测，在域内外均优于单模态方法。

**[Eta-WavLM: Efficient Speaker Identity Removal in Self-Supervised Speech Representations Using a Simple Linear Equation](audio_speech/eta-wavlm_efficient_speaker_identity_removal_in_self-supervised_speech_represent.md)**

:   提出 Eta-WavLM，通过简单的线性方程将 WavLM 自监督语音表示分解为说话人相关和说话人无关两个分量，无需复杂训练即可生成高质量的说话人解耦表示，在语音转换任务上全面超越现有方法。

**[GigaSpeech 2: An Evolving, Large-Scale and Multi-domain ASR Corpus for Low-Resource Languages](audio_speech/gigaspeech2_low_resource_asr.md)**

:   GigaSpeech 2 构建了一个约 30,000 小时的大规模低资源语言（泰语、印尼语、越南语）ASR 语料库，通过自动化爬取-转录-精炼管线从无标注 YouTube 视频生成高质量伪标签，训练的模型仅用 10% 参数量即可将 WER 比 Whisper large-v3 降低 25%-40%。

**[Investigating and Enhancing Vision-Audio Capability in Omnimodal Large Language Models](audio_speech/investigating_and_enhancing_vision-audio_capability_in_omnimodal_large_language_.md)**

:   发现当前全模态大语言模型（OLLMs）在视觉-音频任务上显著弱于视觉-文本任务，原因在于视觉与音频模态之间缺乏直接对齐，并提出 Self-KD（自知识蒸馏）方法，利用 OLLM 自身的视觉-文本组件作为教师来增强视觉-音频能力。

**[Mind the Gap! Static and Interactive Evaluations of Large Audio Models](audio_speech/mind_the_gap_static_and_interactive_evaluations_of_large_audio_models.md)**

:   本文通过收集 484 名参与者的 7,500 次交互评估数据，首次系统比较了大型音频模型（LAM）的静态基准和交互式评估表现，发现两者之间存在显著差距（$R^2=0.30$），并揭示了用户对 LAM 的真实使用场景和偏好。

**[MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens](audio_speech/mms-llama_efficient_llm-based_audio-visual_speech_recognition_with_minimal_multi.md)**

:   提出 MMS-LLaMA，通过早期音视频融合、动态查询分配的 AV Q-Former 和语速预测器三个模块，将多模态语音 token 压缩至每秒仅 3.5 个，在 LRS3 上以 0.72% WER 达到 SOTA 的同时减少 86% token 用量和 35.7% FLOPs。

**[OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation](audio_speech/omniflatten_an_end-to-end_gpt_model_for_seamless_voice_conversation.md)**

:   提出 OmniFlatten——基于 Qwen2-0.5B 的端到端全双工语音对话模型，通过三阶段渐进式后训练（模态对齐→半双工→全双工对话学习）和统一的 flatten 操作，在不修改 GPT 架构的前提下实现了低延迟的自然全双工语音交互，turn-taking 响应时间仅 193ms，显著优于 Moshi 的 553ms。

**[On the Robust Approximation of ASR Metrics](audio_speech/on_the_robust_approximation_of_asr_metrics.md)**

:   提出一种无需真实标签的 ASR 性能指标近似方法，利用多模态统一 embedding 空间中的语音-文本相似度和高质量代理模型的 proxy metrics，训练回归模型预测 WER/CER，在 40+ 模型和 14 个数据集上绝对误差控制在个位数以内，超过最新基线 50% 以上。

**[Predicting Turn-Taking and Backchannel in Human-Machine Conversations Using Linguistic, Acoustic, and Visual Signals](audio_speech/predicting_turn-taking_and_backchannel_in_human-machine_conversations_using_ling.md)**

:   提出首个融合语言、声学和视觉三模态信号预测对话中轮换（turn-taking）和反馈通道（backchannel）动作的端到端框架，并构建了包含 210+ 小时的 MM-F2F 面对面对话数据集，turn-taking F1 提升 10%，backchannel F1 提升 33%。

**[Soundwave: Less is More for Speech-Text Alignment in LLMs](audio_speech/soundwave_less_is_more_for_speech-text_alignment_in_llms.md)**

:   提出 Soundwave 模型，通过高效训练策略和新颖架构解决语音和文本之间的表示空间差距与序列长度不一致问题，仅用五十分之一的训练数据即超越 Qwen2-Audio 在语音翻译和 AIR-Bench 语音任务上的表现。

**[Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens](audio_speech/spark-tts_an_efficient_llm-based_text-to-speech_model_with_single-stream_decoupl.md)**

:   提出 Spark-TTS，基于新型单流语音编解码器 BiCodec 和 Qwen2.5 LLM 的高效 TTS 系统，通过将语音解耦为低码率语义 token 和固定长度全局 token，实现零样本语音克隆和从粗到细的属性控制，在 Seed-TTS-eval 上达到 SOTA 可懂度。

**[Sparsify: Learning Sparsity for Effective and Efficient Music Performance Question Answering](audio_speech/sparsify_music_avqa.md)**

:   Sparsify 提出三层稀疏化策略（稀疏掩码+自适应稀疏合并+关键子集选择）用于音乐表演视听问答（Music AVQA），在 MUSIC-AVQA 和 v2.0 两个 benchmark 上达到 SOTA（81.75%/81.30%），训练时间减少 28.32%，25% 数据即保持 74% 的全量性能。

**[SpeechIQ: Speech-Agentic Intelligence Quotient Across Cognitive Levels in Voice Understanding by Large Language Models](audio_speech/speechiq_speechagentic_intelligence_quotient_across_cognitive.md)**

:   提出 SpeechIQ，一个基于 Bloom 认知分类学的层次化语音理解评估框架，从 Remember（WER）、Understand（语义相似度）、Apply（QA 准确率）三个层次综合评估语音 LLM 的智能水平，发现级联 ASR+LLM 系统在同规模下优于端到端多模态模型。

**[SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models](audio_speech/speechweave_diverse_multilingual_synthetic_text_audio_data_generation_pipeline_f.md)**

:   SpeechWeave 提出了一套端到端的合成语音数据生成流水线，通过关键词采样提升文本多样性、在生成时即完成文本归一化（准确率达97%）、并利用跨语言语音克隆实现说话人标准化，生成的数据在多样性上比直接提示LLM高出10-48%，并能有效提升下游TTS模型性能。

**[T2A-Feedback: Improving Basic Capabilities of Text-to-Audio Generation via Fine-grained AI Feedback](audio_speech/t2a_feedback_audio_gen.md)**

:   提出三个细粒度 AI 音频评分管线（事件出现/事件顺序/声学和谐质量）替代人工标注构建大规模音频偏好数据集 T2A-FeedBack（41K提示+249K音频），用偏好调优增强 TTA 模型的基础能力，在简单（AudioCaps）和复杂（T2A-EpicBench）场景下都显著提升多事件音频生成质量。

**[In-the-wild Audio Spatialization with Flexible Text-guided Localization](audio_speech/tas_audio_spatialization.md)**

:   提出 TAS（Text-guided Audio Spatialization）框架，用灵活的文本提示（3D 空间位置描述或声源间相对位置描述）引导潜在扩散模型将单声道音频转换为双耳音频，构建了 376K 样本的 SpatialTAS 数据集，在模拟和真实录制数据上均超越现有方法，并基于 Llama-3.1-8B 开发了空间语义一致性评估模型。

**[TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis](audio_speech/tcsinger_2_customizable_multilingual_zero-shot_singing_voice_synthesis.md)**

:   提出 TCSinger 2，一个多任务多语言零样本歌声合成模型，通过模糊边界编码器、对比学习音频编码器和基于 Flow 的自定义 Transformer（含 Cus-MOE），实现基于歌声/语音/文本提示的风格迁移与多层级风格控制。

**[Towards Reliable Large Audio Language Model](audio_speech/towards_reliable_large_audio_language_model.md)**

:   本文首次系统研究大型音频语言模型（LALM）的可靠性问题，提出训练无关方法（IDK/MCoT/Task Agent）和训练方法（基于模型特定 IDK 数据集的 LoRA SFT），并设计 Reliability Gain Index（RGI）指标来评估可靠性提升效果，发现"知道说不知道"是可跨音频模态迁移的元能力。

**[UniCodec: Unified Audio Codec with Single Domain-Adaptive Codebook](audio_speech/unicodec_unified_audio_codec_with_single_domain-adaptive_codebook.md)**

:   UniCodec 提出了一种使用单个域自适应码本的统一音频编解码器，通过分区域码本和域混合专家（MoE）策略，在语音、音乐和声音三个域上均实现卓越的重建和语义表示性能。

**[WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models](audio_speech/wavrag_audio-integrated_retrieval_augmented_generation_for_spoken_dialogue_model.md)**

:   提出 WavRAG，首个端到端原生支持音频的检索增强生成框架，通过 WavRetriever 实现音频-文本混合知识库的统一检索，并结合 CoT 推理增强口语对话模型的上下文能力，在保持与 SOTA 文本 RAG 可比性能的同时实现约 10 倍加速。

**[Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Audio-Language Models](audio_speech/who_can_withstand_chat-audio_attacks_an_evaluation_benchmark_for_large_audio-lan.md)**

:   提出 Chat-Audio Attacks (CAA) 基准，包含四类通用对抗音频攻击（内容攻击、情感攻击、显式噪声攻击、隐式噪声攻击），通过三种评估方法系统评测六个 SOTA 大型音频语言模型的鲁棒性，发现 GPT-4o 表现最优但所有模型均存在显著脆弱性。

**[Zero-Shot Text-to-Speech for Vietnamese](audio_speech/zero-shot_text-to-speech_for_vietnamese.md)**

:   针对越南语零样本TTS缺乏高质量长音频数据集的问题，构建了941小时的PhoAudiobook数据集，并在VALL-E、VoiceCraft和XTTS-v2三个SOTA零样本TTS模型上进行系统实验，证明PhoAudiobook显著提升了模型性能，其中XTTS-v2在长句合成上全面超越基线viXTTS，而VALL-E和VoiceCraft在短句合成上更具鲁棒性。

---

## 📚 预训练 { #llm_pretraining }

**[Adversarial Tokenization](llm_pretraining/adversarial_tokenization.md)**

:   本文发现 LLM 管线中 BPE tokenizer 只使用唯一一种分词方式，但同一字符串存在指数级多种合法分词；通过对抗性地选择非标准分词方案，可以在不改变原始文本的情况下绕过安全对齐，攻击成功率与现有 SOTA 文本级攻击方法相当。

**[AsyncLM: Efficient and Adaptive Async Pre-training of Language Models](llm_pretraining/asynclm_efficient_and_adaptive_async_pre-training_of_language_models.md)**

:   本文提出AsyncLM，一种高效的异步预训练框架，通过自适应梯度补偿和动态批量调度策略解决异步分布式训练中的梯度过期问题，在保持与同步训练相当的模型质量的同时，将大规模语言模型预训练的吞吐量提升了1.4-1.8倍。

**[AutoDS: Autonomous Data Selection with Zero-shot Generative Classifiers for Mathematical Texts](llm_pretraining/autonomous_data_selection_with_zero-shot_generative_classifiers_for_mathematical.md)**

:   提出 AutoDS——用基座语言模型自身作为零样本生成分类器，通过 YES/NO token 的 logits 计算连续 LM-Score 来自动评估数学文本质量，筛选高质量语料做持续预训练，在 MATH/GSM8K/BBH 上实现约 2 倍 token 效率提升。

**[Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases](llm_pretraining/between_circuits_chomsky.md)**

:   提出在自然语言预训练前先在形式语言上进行"pre-pretraining"，发现具有层级依赖结构的形式语言（如 k-Shuffle Dyck）能为 Transformer 提供有效的归纳偏置，使 1B 参数模型以 33% 更少的 token 达到相同的语言建模损失。

**[Chinese Grammatical Error Correction With Pre-trained Models and Linguistic Clues](llm_pretraining/chinese_grammatical_error_correction_with_pre-trained_models_and_linguistic_clue.md)**

:   本文提出一种融合预训练语言模型和多层级语言学线索（拼音、字形、句法依存）的中文语法纠错方法，通过显式注入语言学先验知识提升纠错模型对中文特有错误类型的识别和修正能力。

**[CritiQ: Mining Data Quality Criteria from Human Preferences](llm_pretraining/critiq_mining_data_quality_criteria_from_human_preferences.md)**

:   CritiQ 提出了一种基于 Agent 协作的数据质量标准自动挖掘方法，仅需约 30 个人类偏好标注对即可自动发现可解释的数据质量标准，并训练评分器进行高效数据选择，在代码、数学和逻辑领域的实验中显著提升了 Llama 3.1 的下游性能。

**[Data-Constrained Synthesis of Training Data for De-Identification](llm_pretraining/data-constrained_synthesis_of_training_data_for_de-identification.md)**

:   本文系统研究了在数据受限条件下，如何利用领域适应的LLM生成合成临床文本，并通过机器标注训练NER模型进行个人身份信息（PII）检测，发现机器标注器的质量而非生成模型的规模是决定合成数据效用的关键因素。

**[Data Caricatures: On the Representation of African American Language in Pretraining Corpora](llm_pretraining/data_caricatures_on_the_representation_of_african_american_language_in_pretraini.md)**

:   结合定量实验、人工判断和定性分析，系统评估了 12 个开源预训练语料库中非裔美国人语言（AAL）的数量与质量：发现 AAL 仅占 0.007%–0.18% 的文档（远低于人口比例），C4 中 28.9% 的 AAL 文本被判为不适合 LLM 生成、24.5% 强化有害刻板印象，且 16 种自动过滤器中有 13 种系统性地偏向保留白人主流英语（WME）而非 AAL。

**[Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](llm_pretraining/data_whisperer_data_selection.md)**

:   Data Whisperer 提出一种无需训练的注意力加权 few-shot ICL 数据选择方法，利用预训练模型自身的 ICL 能力和注意力分数为训练样本打分，仅用 10% 数据即可超越全量微调性能，同时比现有方法快 7-20 倍。

**[DavIR: Data Selection via Implicit Reward for Large Language Models](llm_pretraining/davir_data_selection_via_implicit_reward_for_large_language_models.md)**

:   提出 DavIR 数据选择方法，通过对基座模型与参考模型的损失差进行**参考模型损失归一化**（而非 token 数归一化），有效消除 RHO 目标中的序列长度依赖，使仅 **6%** 的 Alpaca 数据集（3K/52K）训练出的模型优于全量数据训练模型，同时将归一化思想推广到 DPO 得到 DavIR-DPO，在 AlpacaEval 上提升 Zephyr 8% 的对齐性能。

**[Diversity Explains Inference Scaling Laws: Through a Case Study of Minimum Bayes Risk Decoding](llm_pretraining/diversity_explains_inference_scaling_laws_through_a_case_study_of_minimum_bayes_.md)**

:   从 bias-diversity 分解的理论视角重新解释 MBR 解码：质量估计误差 MSE = Bias - Diversity，增加 diversity（伪参考的多样性）是提升 MBR 性能的关键；进一步通过信息论扩展到一般推理方法，揭示 diversity 是推理 scaling law（增加采样提升性能但边际递减）的理论根源，并在机器翻译、摘要、图像描述任务上实证验证。

**[An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling](llm_pretraining/dual_stage_curriculum_learning_sequence_labeling.md)**

:   提出面向序列标注任务的双阶段课程学习（DCL）框架，通过数据级与模型级两阶段由易到难训练策略，配合基于贝叶斯不确定性的 token 级动态难度度量和 Root 函数训练调度器，在 CWS、POS、NER 三类任务上实现性能提升与训练加速超 27% 的双重收益。

**[Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation](llm_pretraining/emergent_abilities_continued_pt.md)**

:   揭示了持续预训练（CPT）进行语言适应时，混入英文数据对保留模型上下文学习（ICL）能力和下游涌现能力至关重要——尽管不影响验证困惑度；并提出课程学习和 EMA 权重平均作为替代方案。

**[FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling](llm_pretraining/fr_spec_speculative_sampling.md)**

:   提出FR-Spec框架，通过基于词频的词表空间压缩优化投机采样的draft候选选择，将LM Head计算开销降低75%，在保持输出分布不变的前提下实现EAGLE-2之上额外1.12×加速。

**[How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training](llm_pretraining/how_do_llms_acquire_new_knowledge_a_knowledge_circuits_perspective_on_continual_.md)**

:   从知识电路(knowledge circuit)演化视角研究LLM持续预训练中的新知识获取机制，在GPT-2/Llama/Phi三个架构上发现：(1)与已有知识相关的新知识更容易获取；(2)知识电路经历"形成→优化"的明显相变；(3)电路演化遵循"中深层先建立提取功能→浅层后丰富知识表示"的深到浅模式。

**[Improving Continual Pre-training Through Seamless Data Packing](llm_pretraining/improving_continual_pre-training_through_seamless_data_packing.md)**

:   提出 Seamless Packing (SP) 数据打包策略，通过两阶段方法——滑动窗口处理长文本 + FFD 算法打包短文本——在持续预训练中保持上下文连续性、最小化截断和填充，在 99% 的实验设置中超越基线方法。

**[Inconsistent Tokenizations Cause Language Models to be Perplexed by Japanese Grammar](llm_pretraining/inconsistent_tokenizations_cause_language_models_to_be_perplexed_by_japanese_gra.md)**

:   揭示了 tokenizer 的不一致分词是导致 LLM 无法遵守日语"第一人称心理谓词限制"等细微语法规则的根本原因——当限制测试句子为一致分词时，Llama 3 的困惑度差异可改善28倍。

**[Incorporating Domain Knowledge into Materials Tokenization](llm_pretraining/incorporating_domain_knowledge_into_materials_tokenization.md)**

:   提出 MATTER——一种面向材料科学的领域感知分词框架，通过训练材料概念检测器 MatDetector 并将检测结果注入分词的合并排序中，避免领域术语碎片化，在生成和分类任务上分别平均提升 4% 和 2%。

**[InSerter: Speech Instruction Following with Unsupervised Interleaved Pre-training](llm_pretraining/inserter_speech_instruction.md)**

:   提出 InSerter（交错语音-文本预训练）方法，通过 TTS 将大规模文本语料合成为交错的语音-文本序列进行预训练，大幅提升 SpeechLLM 的语音指令遵循能力，并构建首个全面的语音指令遵循基准 SpeechInstructBench。

**[Large Vocabulary Size Improves Large Language Models](llm_pretraining/large_vocabulary_size_improves_large_language_models.md)**

:   实验证明更大的 subword 词汇表大小 (vocabulary size) 能持续提升 LLM 在下游任务上的性能，并提出了一种简洁的词汇表替换方法 (Swap & Insert) 用于持续训练场景下切换到更合适的词汇表。

**[LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models](llm_pretraining/leancode_understanding_models_better_for_code_simplification_of_pre-trained_larg.md)**

:   本文提出LeanCode，一种基于上下文感知注意力分数的代码简化方法，利用CLS注意力（分类任务）和编码器-解码器注意力（生成任务）来衡量token重要性，在代码搜索和代码摘要任务上分别比SOTA方法DietCode/SlimCode提升最高60%和29%，同时减少高达40.9%的推理时间。

**[Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning](llm_pretraining/making_llms_better_many-to-many_speech-to-text_translators_with_curriculum_learn.md)**

:   提出 LLM-SRT，将语音到文本翻译（S2TT）任务转化为语音识别与翻译联合任务（SRT），并通过三阶段课程学习策略（ASR→SMT→SRT）有效利用 LLM 的机器翻译能力，在低资源场景（每种语言不到 10 小时数据）下实现 15×14 语言对的 SOTA 多对多语音翻译性能。

**[Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](llm_pretraining/metarater_a_multidimensional_data_selection_method.md)**

:   提出Meta-rater多维数据选择框架，定义PRRC四个质量维度（专业性/可读性/推理性/清洁度），通过proxy模型回归学习多个质量分数的最优加权组合，使1.3B模型训练收敛速度翻倍、下游任务提升3.23%。

**[Model Performance-Guided Evaluation Data Selection for Effective Prompt Optimization](llm_pretraining/model_performance-guided_evaluation_data_selection_for_effective_prompt_optimiza.md)**

:   提出 IPOMP——一种两阶段评估数据选择方法，第一阶段通过语义聚类和边界分析选取多样化样本，第二阶段利用提示优化过程中的实时模型性能迭代替换冗余样本，在 BIG-bench 和 LIAR 上将提示优化效果提升 1.6%-3.1%，稳定性提升 50%+，额外开销不到 1%。

**[Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset](llm_pretraining/nemotron_cc_pretraining_data.md)**

:   Nemotron-CC 通过分类器集成提升高质量 token 召回、合成数据改写扩展唯一 token 数量、对高质量数据取消启发式过滤三大策略，从 Common Crawl 构建了 6.3T token 的长周期预训练数据集（含 4.4T 唯一真实 token + 1.9T 合成 token），在 15T token 训练场景下使 8B 模型 MMLU 达 70.3，超越同规模训练的 Llama 3.1 8B 的 65.3。

**[Optimizing Pre-Training Data Mixtures with Mixtures of Data Expert Models](llm_pretraining/optimizing_pre-training_data_mixtures_with_mixtures_of_data_expert_models.md)**

:   提出Mixture of Data Experts (MDE)方法，通过在各数据域上独立训练专家模型并用混合权重进行概率级集成，高效近似不同数据混合比下的语言模型损失，大幅提升预训练数据混合比例的搜索效率和预测精度。

**[Pre-Training Curriculum for Multi-Token Prediction in Language Models](llm_pretraining/pre-training_curriculum_for_multi-token_prediction_in_language_models.md)**

:   针对小语言模型（SLM）难以直接受益于多 token 预测（MTP）目标的问题，提出前向/反向课程学习策略——前向课程（NTP→MTP）使 SLM 在保持自推测解码加速的同时提升生成质量，反向课程（MTP→NTP）在 NTP 性能上更优但失去推理加速优势。

**[Retrofitting Large Language Models with Dynamic Tokenization](llm_pretraining/retrofitting_large_language_models_with_dynamic_tokenization.md)**

:   本文提出对已有语言模型进行动态分词改造（dynamic tokenization），通过受 BPE 启发的子词合并算法动态决定 token 边界，结合预训练的嵌入预测超网络在线计算合并后 token 的嵌入向量，在 encoder 模型上实现平均 >20% 的序列长度缩减且性能仅下降不到 2%，在 decoder 模型上也实现了最高 17% 的序列缩减。

**[SCAR: Data Selection via Style Consistency-Aware Response Ranking for Efficient Instruction-Tuning](llm_pretraining/scar_style_consistency_data_selection.md)**

:   SCAR 识别出回复的"语言形式"和"指令惊奇度"是影响 LLM 指令微调效果的两个关键风格因素，并提出基于风格一致性的排序方法自动选择高质量训练数据，仅用 0.7% 的原始数据就能让微调后的 LLM 匹配甚至超越全数据集训练的性能。

**[Second Language (Arabic) Acquisition of LLMs via Progressive Vocabulary Expansion](llm_pretraining/second_language_arabic_acquisition_of_llms_via_progressive_vocabulary_expansion.md)**

:   受人类第二语言习得启发，提出渐进式词表扩展（Progressive Vocabulary Expansion）方法，通过分阶段指数增长地扩展阿拉伯语子词到 LLaMA2 词表中，在保留原模型英语知识的同时高效适配阿拉伯语，构建出 AraLLaMA 7B/13B 模型。

**[Splintering Nonconcatenative Languages for Better Tokenization](llm_pretraining/splintering_nonconcatenative_languages_for_better_tokenization.md)**

:   提出 Splinter，一种预分词步骤，通过迭代剪除模板字符将非拼接性语言（希伯来语、阿拉伯语、马来语）的词重排为线性形式，使标准 BPE/UnigramLM 能发现形态学上有意义的连续片段，在内在指标和希伯来语下游任务上均优于原始分词。

**[Stealing Training Data from Large Language Models in Decentralized Training through Activation Inversion Attack](llm_pretraining/stealing_training_data_from_large_language_models_in_decentralized_training_thro.md)**

:   提出 Activation Inversion Attack（AIA），首次系统揭示去中心化训练（流水线并行）中恶意阶段可通过截获中间激活值高效重构训练数据，在 Bloom-7B1 微调场景下可精确恢复 62% 的私人邮件和接近 100% 的生日信息。

**[Synthesizing Post-Training Data for LLMs through Multi-Agent Simulation](llm_pretraining/synthesizing_post-training_data_for_llms_through_multi-agent_simulation.md)**

:   本文提出 MATRIX 多智能体模拟器和 MATRIX-Gen 场景驱动指令生成器，通过模拟真实社会场景来合成高质量的 LLM 后训练数据，仅用 20K 条合成数据训练的 Llama-3-8B 在 AlpacaEval 2 和 Arena-Hard 上超过了使用超过 10M 数据训练的 Meta 官方 Llama-3-8B-Instruct。

**[TokAlign: Efficient Vocabulary Adaptation via Token Alignment](llm_pretraining/tokalign_vocab_adaptation.md)**

:   提出 TokAlign，基于 Token 共现信息学习两个词表之间的一对一映射矩阵，高效替换 LLM 的词表，实现跨语言知识迁移和跨模型 token 级蒸馏。

**[Tokenization is Sensitive to Language Variation](llm_pretraining/tokenization_is_sensitive_to_language_variation.md)**

:   系统研究了 BPE tokenizer 的三个关键设计选择（拟合语料、pre-tokenizer、词表大小）对语言变体鲁棒性任务和敏感性任务下游性能的差异化影响，并提出基于 logistic regression 的 task-aware tokenizer 评估指标，显著优于 Rényi efficiency 等 task-agnostic 指标。

**[Towards Effective and Efficient Continual Pre-training of Large Language Models](llm_pretraining/towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)**

:   系统性地研究了对Llama-3 (8B)进行持续预训练的数据策略，通过主题级数据混合、困惑度课程学习和高质量合成科学QA数据三大策略，仅用100B token就显著增强了中文能力（C-Eval +8.81）和科学推理能力（MATH +12.00），同时有效保持了原始英语能力。

**[Training Dynamics Underlying Language Model Scaling Laws: Loss Deceleration and Zero-Sum Learning](llm_pretraining/training_dynamics_underlying_language_model_scaling_laws_loss_deceleration_and_z.md)**

:   发现语言模型训练中存在 loss deceleration（损失减速）现象——损失曲线在 log-log 空间呈分段线性，根因是 zero-sum learning（ZSL）：per-token 梯度系统性对立导致破坏性干涉，将一部分样本的改善抵消另一部分的恶化；scale up 通过降低减速触发损失 $L_d$ 和提升减速后斜率 $r_d$ 来缓解 ZSL，为突破 scaling law 瓶颈提供了可直接干预的机制。

**[Unsupervised Morphological Tree Tokenizer](llm_pretraining/unsupervised_morphological_tree_tokenizer.md)**

:   提出 TreeTok，一种基于无监督神经形态结构归纳的分词器，通过 MorphOverriding 机制和自监督目标学习字符级树结构，以自顶向下词表匹配方式进行分词，在形态分割和语言建模任务上均优于 BPE/WordPiece。

**[Velocitune: A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training](llm_pretraining/velocitune_a_velocity-based_dynamic_domain_reweighting_method_for_continual_pre-.md)**

:   提出 Velocitune 框架，通过学习速度（learning velocity）动态调整持续预训练中各数据域的采样权重——优先加大学习较慢的域的权重，并利用 scaling law 低成本估计目标损失，在数学/代码推理和系统命令生成任务上显著优于静态混合基线。

---

## ⚡ LLM 效率 { #llm_efficiency }

**[A Drop-In Solution for On-the-Fly Adaptation of Speculative Decoding in Large Language Models](llm_efficiency/a_drop-in_solution_for_on-the-fly_adaptation_of_speculative_decoding_in_large_la.md)**

:   本文提出一种即插即用的推测解码自适应方案，能够在推理过程中动态调整草稿模型的推测窗口大小γ（以及可能的草稿模型选择），从而在不同输入分布下最大化推测解码的端到端加速比。

**[Accelerating Speculative Decoding via Efficient Context-Aware Draft Generation](llm_efficiency/accelerating_speculative_decoding_via_efficient_context-aware_draft_generation.md)**

:   本文提出了一种上下文感知的高效草稿生成策略来加速推测解码（Speculative Decoding），通过让草稿模型根据当前上下文动态调整生成质量，在保持输出一致性的前提下显著提升 LLM 推理吞吐量。

**[LaMPE: Length-aware Multi-grained Positional Encoding for Adaptive Long-context Scaling Without Training](llm_efficiency/adaptive_grouped_pe_context_window.md)**

:   提出 LaMPE（Length-aware Multi-grained Positional Encoding），通过 **参数化 scaled sigmoid 函数** 自适应确定最优映射长度，并设计 **三区域多粒度注意力机制**（head 精细局部 + middle 线性归一化压缩 + tail 恢复长程依赖），实现无训练即插即用的 LLM 上下文窗口外推，在五大长上下文基准上全面超越现有方法。

**[Boosting Long-Context Information Seeking via Query-Guided Activation Refilling](llm_efficiency/boosting_long-context_information_seeking_via_query-guided_activation_refilling.md)**

:   本文提出ACRE（Activation Refilling）方法，通过构建双层KV缓存架构——L1层紧凑捕获全局信息、L2层提供局部详细信息——并利用输入查询动态从L2向L1补充相关条目，实现长上下文信息检索任务的高效处理，在性能和效率上均有显著提升。

**[CLaSp: In-Context Layer Skip for Self-Speculative Decoding](llm_efficiency/clasp_self_speculative_decoding.md)**

:   CLaSp 提出一种无需训练的自推测解码方法，通过动态规划算法在每个验证步骤后根据上下文动态调整跳层策略，利用上一次验证的完整隐状态作为目标来选择最优跳层集合，在 LLaMA3 系列上实现 1.3-1.7× 加速且不改变生成分布。

**[CNNSum: Exploring Long-Context Summarization with Large Language Models in Chinese Novels](llm_efficiency/cnnsum_exploring_long-context_summarization_with_large_language_models_in_chines.md)**

:   构建了 CNNSum——基于中文小说的多尺度长文本摘要基准（695 样本，16k-128k tokens），通过人工标注确保质量，系统测评了 20+ 个 LLM，发现高级 LLM 倾向生成主观评述导致摘要模糊、小模型性价比更高、Base 版微调效果优于 Chat 版，且用短文本数据微调即可显著提升长文本摘要能力。

**[Consistency-Preserving Contrastive Decoding for Faithful Document-Grounded Dialogue](llm_efficiency/consistency-preserving_contrastive_decoding_for_faithful_document-grounded_dial.md)**

:   本文提出一种一致性保持的对比解码（Consistency-Preserving Contrastive Decoding, CPCD）方法，通过在解码阶段对比有文档条件和无文档条件的生成分布，增强文档基础对话系统对源文档的忠实性，同时保持回复的流畅性和对话一致性。

**[Giraffe: Design Choices for Extending the Context Length of Visual Language Models](llm_efficiency/design_choices_for_extending_the_context_length_of_visual_language_models.md)**

:   系统性地探索了将现有视觉语言模型（VLM）的上下文窗口扩展到128K的设计空间，从数据配方、位置编码扩展到上下文利用三个维度提出最佳实践，并提出 M-RoPE++ 和混合分辨率训练两项技术，构建的 Giraffe 模型在长上下文 VLM 中达 SOTA。

**[Distance between Relevant Information Pieces Causes Bias in Long-Context LLMs](llm_efficiency/distance_between_relevant_information_pieces_causes_bias_in_long-context_llms.md)**

:   本文提出 LongPiBench 基准，首次系统研究当长上下文中存在多个相关信息片段时，LLM 对信息片段间距离（spacing）的敏感性，揭示了当前模型虽已基本克服"中间丢失"问题，但在相关信息间距变化时仍存在显著的位置偏差。

**[DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts](llm_efficiency/dive_moe_reconstruction.md)**

:   提出 DIVE，一种将 Dense LLM 重构为 MoE 架构的方法，核心洞察是不同领域的校准数据集会让结构化剪枝产生不同的剪枝结果，利用这种多样性构建领域特异的专家，配合高效的两阶段重训练（router dense训练 + expert LoRA稀疏训练），在仅调不到 1% 参数的情况下实现优于现有剪枝和 MoE 重构方法的效果。

**[Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models](llm_efficiency/dynamic_chunking_and_selection_for_reading_comprehension_of_ultra-long_context_i.md)**

:   提出 Dynamic Chunking and Selection (DCS)，通过基于语义相似度的动态分块和问题感知分类器的块选择，解决长文本固定分块导致的语义断裂问题，在 12 个长文本 QA 数据集上以 Llama3 为基座实现 single-hop 平均 35.50（+28.6%）和 multi-hop 平均 29.07（+20.0%）的提升，且在 256k token 输入下保持鲁棒。

**[Efficient Many-Shot In-Context Learning with Dynamic Block-Sparse Attention](llm_efficiency/efficient_many-shot_in-context_learning_with_dynamic_block-sparse_attention.md)**

:   提出 Dynamic Block-Sparse Attention (DBSA)，一种无需训练的推理框架，通过结构化块稀疏注意力编码和动态检索 KV 缓存，在多示例上下文学习中实现接近微调的推理延迟，同时保持 >95% 的最佳方法准确率。

**[Entailment-Preserving First-order Logic Representations in Natural Language Entailment](llm_efficiency/entailment-preserving_first-order_logic_representations_in_natural_language_enta.md)**

:   形式化定义了蕴含保持一阶逻辑表示（EPF）任务及无参考评价指标（EPR系列），提出迭代learning-to-rank训练方法，通过BRIO损失优化T5模型的NL→FOL翻译，使其生成的FOL表示能被自动定理证明器验证蕴含关系，在三个数据集上EPR提升1.8-2.7%、EPR@16提升17.4-20.6%。

**[FocusLLM: Precise Understanding of Long Context by Dynamic Condensing](llm_efficiency/focusllm_precise_understanding_of_long_context_by_dynamic_condensing.md)**

:   提出FocusLLM框架，通过将长文本分块并为每块注入动态提示（dynamic prompt），用可训练的候选token浓缩各块的关键信息，再通过并行解码机制聚合到本地上下文中生成下一个token，仅用8K训练长度和0.5B训练预算即可扩展LLaMA-2到400K上下文，在LongBench和∞-Bench上超越所有基线。

**[FUEL: Unveiling Environmental Impacts of Large Language Model Serving: A Functional Unit View](llm_efficiency/fuel_unveiling_environmental_impacts_of_llm_serving.md)**

:   提出 FUEL 框架，首次引入生命周期评估中的"功能单元"（Functional Unit）概念作为标准化比较基准，在统一的质量、性能和工作负载约束下评估不同 LLM 服务配置的碳排放，通过模型大小、量化策略和硬件选择三个案例研究揭示了多个反直觉的绿色 AI 洞察。

**[GigaChat Family: Efficient Russian Language Modeling Through Mixture of Experts Architecture](llm_efficiency/gigachat_family_efficient_russian_language_modeling_through_mixture_of_experts_a.md)**

:   介绍 GigaChat 系列——首个从头为俄语设计并预训练的 MoE 架构 LLM 家族，包含 20B 总参数/3.3B 激活参数的基座和指令微调模型，在俄语 benchmark 上达到同规模 SOTA，训练速度是同量级 dense 模型的 2 倍，推理延迟降低 40%。

**[KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding](llm_efficiency/kv_latent_cache_reduction.md)**

:   KV-Latent 通过直接缩减预训练模型中 Key/Value 注意力头的维度（将 KV 向量映射到低维隐空间），配合两阶段微调策略和频率感知的 RoPE 修改，仅用不到 1% 预训练量的额外训练就实现 KV Cache 50-87% 的压缩，同时基本保持模型性能。

**[LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs](llm_efficiency/ladm_long_context_data.md)**

:   LADM提出了一种基于注意力机制的长上下文训练数据选择框架，通过训练一个小型Long Attention Calculator来计算span间的注意力依赖分数（PFS → AFS → CDS），从大规模语料中高效筛选具有强长程依赖的高质量样本用于持续预训练，仅用1B tokens即可显著提升LLM的长上下文能力。

**[Literary Evidence Retrieval via Long-Context Language Models](llm_efficiency/literary_evidence_retrieval_via_long-context_language_models.md)**

:   将 RELiC 数据集改造为长上下文文学证据检索 benchmark（292 个高质量样本），要求模型在完整小说文本（45k-125k tokens）中为文学评论找到缺失引用；Gemini Pro 2.5 以 62.5% 准确率首次超越人类专家（55%），但最佳开源模型 DeepSeek-R1 仅 29.1%，揭示了闭源/开源模型在解释性推理上的巨大鸿沟。

**[LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks](llm_efficiency/longbench_v2_towards_deeper_understanding_and_reasoning_on_realistic_long-contex.md)**

:   LongBench v2 是一个包含 503 道高难度多选题的长上下文评测基准，上下文长度从 8k 到 2M 词，覆盖六大任务类型，人类专家在 15 分钟限制下仅达到 53.7% 准确率，而最强直接回答模型（GPT-4o 2024-08）仅 50.1%，推理模型 o1-preview 达到 57.7%，凸显了推理时计算扩展对长上下文深层理解的重要性。

**[LongReward: Improving Long-context Large Language Models with AI Feedback](llm_efficiency/longreward_improving_long-context_large_language_models_with_ai_feedback.md)**

:   提出 LongReward，利用现成 LLM 从帮助性、逻辑性、忠实性和完整性四个维度为长上下文模型回复自动打分，结合 DPO 离线强化学习显著提升长上下文 SFT 模型的多维能力。

**[LongSafety: Evaluating Long-Context Safety of Large Language Models](llm_efficiency/longsafety_evaluating_long-context_safety_of_large_language_models.md)**

:   提出LongSafety——首个专门针对开放式长上下文任务的LLM安全评估基准，包含7类安全问题和6种任务类型共1,543个测试用例，揭示大多数模型安全率低于55%，且短上下文安全能力无法迁移到长上下文场景。

**[What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs](llm_efficiency/many_shot_attacks_long_context.md)**

:   系统分析 Many-Shot Jailbreaking（MSJ）攻击的关键因素，发现上下文长度是攻击成功的决定性因素，而内容的有害性、主题、格式几乎不重要——即使重复安全内容、随机无意义文本（Lorem Ipsum）都能在长上下文下突破模型安全对齐。

**[Mitigating Posterior Salience Attenuation in Long-Context LLMs with Positional Contrastive Decoding](llm_efficiency/mitigating_posterior_salience_attenuation_in_long-context_llms_with_positional_c.md)**

:   发现长上下文LLM中的后验显著性衰减（PSA）现象——gold token的显著性随上下文增长而下降但仍保持高排名，由此提出无需训练的位置对比解码（PCD）方法，通过对比长距离感知注意力和局部感知注意力的logits来放大长距离信号，在多个长上下文基准上取得SOTA。

**[Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](llm_efficiency/native_sparse_attention.md)**

:   DeepSeek提出NSA——一种可原生训练的分层稀疏注意力机制，通过压缩token、选择token和滑动窗口三条并行注意力路径实现高效长上下文建模，在27B参数模型上预训练后性能全面匹配甚至超越Full Attention，同时在64k序列上实现显著加速。

**[On Many-Shot In-Context Learning for Long-Context Evaluation](llm_efficiency/on_many-shot_in-context_learning_for_long-context_evaluation.md)**

:   深入研究 many-shot ICL 用于长上下文语言模型评估，提出 Sample Learning Ratio 指标区分 SSL 和 ASL 任务，构建 ManyICLBench 基准全面评测 12 个 LCLM。

**[Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models](llm_efficiency/ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)**

:   提出 Ref-Long benchmark，从"引用定位"（给定 key 识别哪些文档引用了它并返回索引）这一被忽视的维度评估长上下文模型，包含 3 个子集（合成→真实）共 4300 个任务；发现即使 GPT-4o 在 Multi-Hard-24K 上 ExAcc 仅 19%，远低于人类 92%，且 prompt 工程和专项微调均无法根本解决该问题。

**[RefreshKV: Updating Small KV Cache During Long-form Generation](llm_efficiency/refreshkv_updating_small_kv_cache_during_long-form_generation.md)**

:   提出RefreshKV推理方法，通过在生成过程中周期性地在全KV缓存注意力和小KV缓存注意力之间交替，并基于全注意力步的注意力模式动态更新小KV缓存，在不永久丢弃任何token的前提下，实现与驱逐式方法相当的加速且大幅提升长文本生成任务性能。

**[SAM Decoding: Speculative Decoding via Suffix Automaton](llm_efficiency/sam_decoding_speculative_decoding_via_suffix_automaton.md)**

:   提出SAM-Decoding，利用后缀自动机（Suffix Automaton）对通用文本语料和当前文本序列进行最长后缀匹配来高效生成推测解码的草稿，平均O(1)时间复杂度，在Spec-Bench上比现有检索式方法快18%+，并可与EAGLE-2等方法互补组合进一步提速3.28%-11.13%。

**[Scaling Context, Not Parameters: Training a Compact 7B Language Model for Efficient Long-Context Processing](llm_efficiency/scaling_context_not_parameters_training_a_compact_7b_language_model_for_efficien.md)**

:   提出 MegaBeam-Mistral-7B，一个支持 512K token 上下文长度的 7B 语言模型，通过四阶段渐进式训练、RoPE theta 调优、bfloat16 精度修复和 XLA 编译器内存优化等工程实践，使紧凑型模型在长上下文任务上达到甚至超越大参数模型（如 Llama-3.1-70B、GPT-4）的性能。

**[SEAL: Scaling to Emphasize Attention for Long-Context Retrieval](llm_efficiency/seal_scaling_to_emphasize_attention_for_long-context_retrieval.md)**

:   SEAL 通过发现特定注意力头/通道对长上下文检索有正/负影响的现象，设计了头级和通道级可学习缩放因子，仅用50个合成样本微调即可大幅提升LLM长上下文检索性能，且缩放因子可离线合并至模型权重实现零推理开销。

**[Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models](llm_efficiency/sliding_windows_full_ranking.md)**

:   本文系统研究了长上下文LLM在段落排序中的应用，提出用 full ranking（一次性排序所有段落）替代传统滑动窗口策略，并设计了多轮滑动窗口标签构造方法和重要性感知损失函数来微调 full ranking 模型，在效率提升约30-65%的同时实现了排序效果的全面超越。

**[Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](llm_efficiency/smarter_better_faster_longer_a_modern_bidirectional_encoder_for_fast_memory_effi.md)**

:   提出 ModernBERT，将现代 LLM 架构优化（RoPE、GeGLU、交替局部/全局注意力、unpadding）系统性地引入 encoder-only 模型，在 2T token 上训练并原生支持 8192 上下文长度，在分类和检索任务上全面超越 BERT/RoBERTa/DeBERTaV3，同时推理速度和显存效率大幅领先。

**[SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers](llm_efficiency/spindlekv_layered_kv_cache.md)**

:   SpindleKV 提出分层处理 KV cache 压缩的策略——深层使用注意力驱动的 token eviction（利用稀疏注意力），浅层使用基于相似性学习的 codebook 替换（利用 token 间高相似度），并解决了 GQA 兼容性问题，实现 50% KV cache 缩减而不损失性能。

**[Squeezed Attention: Accelerating Long Context Length LLM Inference](llm_efficiency/squeezed_attention_accelerating_long_context_length_llm_inference.md)**

:   提出 Squeezed Attention，通过离线 K-means 聚类压缩固定上下文的 Key 向量，在推理时用质心匹配预测重要 Key 并仅对其计算精确注意力，实现 3.1 倍 KV 预算削减且无精度损失，预填充和生成阶段均获得超过 4 倍加速。

**[Tetris: Optimal Draft Token Selection for Batch Speculative Decoding](llm_efficiency/tetris_optimal_draft_token_selection_for_batch_speculative_decoding.md)**

:   Tetris 提出了一种在批量推测解码场景下，跨请求动态选择最优草稿token的方法，通过贪心选择累积接受概率最高的token来最大化有限计算资源下的推理吞吐量。

**[How to Train Long-Context Language Models (Effectively)](llm_efficiency/train_long_context_effectively.md)**

:   本文系统研究如何通过持续预训练和 SFT 有效训练长上下文语言模型，提出数据配比、训练长度缩放、评估协议等一系列关键设计，最终训练出的 ProLong-8B 仅用 Llama-3.1 **5%** 的长上下文训练数据即在 128K 长度上达到同规模 SOTA。

**[What are the Essential Factors in Crafting Effective Long Context Multi-Hop Instruction Datasets? Insights and Best Practices](llm_efficiency/what_are_the_essential_factors_in_crafting_effective_long_context_multi-hop_inst.md)**

:   提出多智能体交互式多跳生成（MIMG）框架，通过质量验证、单跳问题生成、多问题采样和多跳合并四个模块，系统性地合成高质量长上下文多跳指令数据，训练后模型平均提升7.54%，甚至超越更大规模人工标注数据集。

---

## 📖 NLP 理解 { #nlp_understanding }

**[A Comprehensive Graph Framework for Question Answering with Mode-Seeking Preference Alignment](nlp_understanding/a_comprehensive_graph_framework_for_question_answering_with_mode-seeking_prefere.md)**

:   提出GraphMPA框架，通过构建基于通用相似度度量的层次化文档图实现全局文档理解，并引入mode-seeking偏好优化替代传统DPO实现更精准的人类偏好对齐，在6个QA数据集上全面超越现有RAG方法。

**[A Variational Approach for Mitigating Entity Bias in Relation Extraction](nlp_understanding/a_variational_approach_for_mitigating_entity_bias_in_relation_extraction.md)**

:   提出基于变分信息瓶颈（VIB）的实体去偏方法，将实体token映射为高斯分布以选择性压缩实体特定信息、保留上下文语义，在通用/金融/生物医学三个领域的关系抽取数据集上均取得SOTA，特别是在OOD场景下BioRED提升5.3个F1点。

**[Active LLMs for Multi-hop Question Answering](nlp_understanding/active_llms_for_multi-hop_question_answering.md)**

:   本文提出一种主动式大语言模型框架，通过让LLM主动决定何时需要检索外部信息、何时可以直接推理，从而在多跳问答任务中实现更高效、更准确的推理过程。

**[Adapting Psycholinguistic Research for LLMs: Gender-Inclusive Language in a Coreference Context](nlp_understanding/adapting_psycholinguistic_research_for_llms_gender-inclusive_language_in_a_coref.md)**

:   将 Tibblin et al. (2023) 的心理语言学实验从法语适配到英语和德语 LLM，通过测量共指词概率和生成内容分析发现：英语 LLM 基本保持先行词-共指词性别一致但 they 单数几乎不被使用且存在底层男性偏见；德语 Leo Mistral 7B 的男性偏见更强烈（压倒所有 8 种包容策略），但包容策略仍能增加女性/中性性别的出现概率，与心理语言学人类实验结果一致。

**[Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification](nlp_understanding/analyzing_political_bias_in_llms_via_target-oriented_sentiment_classification.md)**

:   提出基于目标导向情感分类(TSC)的LLM政治偏差分析框架，通过在450个政治句子中替换1319位政治家名字并用7个模型在6种语言中预测情感，定义了基于熵的不一致性指标来量化偏差，发现LLM对左翼和中间派有正面偏见、对极右翼有负面偏见，且更大模型偏差更强更一致。

**[Automatic Generation of Inference Making Questions for Reading Comprehension Assessments](nlp_understanding/automatic_generation_of_inference_making_questions_for_reading_comprehension_ass.md)**

:   开发了一套阅读理解推理题分类法（代词桥接/文本连接/填补空白），用 GPT-4o few-shot 提示自动生成针对特定推理类型的多项选择题；93.8% 的题目质量合格，但仅 42.6% 准确匹配目标推理类型，说明 LLM 在精确推理能力控制上仍有不足。

**[BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering](nlp_understanding/belle_a_bi-level_multi-agent_reasoning_framework_for_multi-hop_question_answerin.md)**

:   提出 BELLE 双层多智能体辩论框架，先将多跳问题分类为四种类型，再通过双层辩论机制（第一层正反方辩论 + 第二层快/慢辩论者监督）动态规划 CoT、单步检索、迭代检索等算子的组合方案，实现面向问题类型的自适应多跳推理。

**[BookCoref: Coreference Resolution at Book Scale](nlp_understanding/bookcoref_book_scale.md)**

:   提出首个书级别共指消解基准BookCoref，通过角色链接+LLM过滤+窗口扩展的自动标注管线，在50本完整小说上生成高质量银标注数据，平均文档长度超过20万tokens。

**[BQA: Body Language Question Answering Dataset for Video Large Language Models](nlp_understanding/bqa_body_language_question_answering_dataset_for_video_large_language_models.md)**

:   基于BoLD数据集通过四步半自动流水线构建了BQA——一个包含7,632个短视频的肢体语言情感识别多选QA基准，评估发现最强VideoLLM（GPT-4o/Gemini）准确率仅约60%远低于人类的85%，同时揭示了模型对面部表情的过度依赖以及针对特定种族群体的显著偏见。

**[CaLMQA: Exploring Culturally Specific Long-Form Question Answering across 23 Languages](nlp_understanding/calmqa_cultural_multilingual_qa.md)**

:   构建了首个多语言长文本问答数据集 CaLMQA（51.7K 问题，23 种语言），通过无翻译方式收集文化特异性问题，发现 LLM 回答文化特异性问题的事实性（45-52%）显著低于文化无关问题（64-71%），低资源语言表现尤其差。

**[Can LLMs Reliably Simulate Real Students' Abilities in Mathematics and Reading Comprehension?](nlp_understanding/can_llms_reliably_simulate_real_students_abilities_in_mathematics_and_reading_co.md)**

:   利用项目反应理论(IRT)将11个LLM与真实学生放在同一能力量表上评估，发现在无引导情况下强模型远超学生平均水平，而"扮演某年级学生"的提示虽能改变表现，但**没有任何模型-提示组合**能在所有学科和年级上可靠模拟平均学生。

**[Disambiguate First, Parse Later: Generating Interpretations for Ambiguity Resolution in Semantic Parsing](nlp_understanding/disambiguate_first_parse_later_generating_interpretations_for_ambiguity_resoluti.md)**

:   提出"先消歧、后解析"的模块化方法，利用LLM生成默认解释并训练专门的infilling模型补全缺失解释，将歧义自然语言问题转化为多个明确解释后再分别进行SQL解析。

**[Dynamic Order Template Prediction for Generative Aspect-Based Sentiment Analysis](nlp_understanding/dot_absa_template.md)**

:   本文提出 Dynamic Order Template (DOT) 方法，将 ABSA 情感四元组生成分为两阶段——先预测元组数量并生成初始模板，再基于动态模板生成具体情感元组，在 9 个 ABSA 数据集上实现 SOTA 同时推理时间比 MvP 减少 7 倍。

**[Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering](nlp_understanding/embqa_embedding_odqa.md)**

:   EmbQA 提出嵌入级 ODQA 框架，用轻量线性层和无监督对比学习优化查询表示实现段落重排序，并引入基于序统计量的探索性嵌入扩展候选答案多样性，配合熵选择机制自动选答，在 4 个 ODQA 数据集上以更低计算成本超越 SuRe 等 prompt 级方法。

**[End-to-End Dialog Neural Coreference Resolution: Balancing Efficiency and Accuracy in Large-Scale Systems](nlp_understanding/end-to-end_dialog_neural_coreference_resolution_balancing_efficiency_and_accurac.md)**

:   提出一个端到端神经共指消解系统，通过结合上下文嵌入、层次化注意力机制和优化策略（剪枝/量化），在OntoNotes等基准数据集上实现效率与准确率的平衡，SpanBERT达到87.3 F1。

**[Towards a More Generalized Approach in Open Relation Extraction](nlp_understanding/generalized_open_relation_extract.md)**

:   提出 MixORE 框架，在更通用的 Open Relation Extraction 设定下（无标注数据同时包含已知和新颖关系，且不做长尾或预分割假设），通过 Semantic Autoencoder 检测新关系 + 开放世界半监督联合学习，在 FewRel/TACRED/Re-TACRED 上全面超越 SOTA。

**[Generating Diverse Training Samples for Relation Extraction with Large Language Models](nlp_understanding/generating_diverse_training_samples_for_relation_extraction_with_large_language_.md)**

:   研究如何用LLM为关系抽取（RE）生成高质量且多样化的训练样本，提出基于ICL的逐条生成策略和基于DPO的多样性微调方法，生成的训练数据可有效提升few-shot RE模型性能。

**[In the LLM Era, Word Sense Induction Remains Unsolved](nlp_understanding/in_the_llm_era_word_sense_induction_remains_unsolved.md)**

:   本文系统评估了 LLM 时代的词义归纳（WSI）任务，在控制更严谨的 SemCor 衍生评估集上发现，包括 LLM 方法在内的所有无监督方法都无法超越简单的"每个词只有一个义项"基线，而结合 Wiktionary 的半监督方法超越了前 SOTA 3.3%，说明 WSI 仍远未被解决。

**[iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering](nlp_understanding/iquest_an_iterative_question-guided_framework_for_knowledge_base_question_answer.md)**

:   iQUEST 提出迭代式子问题引导框架，在每一步推理中动态生成当前可解答的子问题以维持推理方向，并结合 GNN 聚合二跳邻居语义信息实现"前瞻性"实体探索，在 CWQ、WebQSP、WebQuestions、GrailQA 四个基准上取得 SOTA 或接近 SOTA 的性能，且无需微调 LLM。

**[Multi-Hop Reasoning for Question Answering with Hyperbolic Representations](nlp_understanding/multi-hop_reasoning_for_question_answering_with_hyperbolic_representations.md)**

:   通过在 T5 编码器-解码器模型中简单插入一层 Poincaré 双曲层，以最小改动将欧几里得嵌入映射到双曲空间进行多跳推理，实验在四个数据集上一致性地超越欧几里得对照组，并证明了基于 δ-hyperbolicity 初始化曲率的有效性以及双曲空间在层次结构更强的数据集上优势更明显。

**[On Synthesizing Data for Context Attribution in Question Answering](nlp_understanding/on_synthesizing_data_for_context_attribution_in_question_answering.md)**

:   本文提出 SynQA，一种基于"给定上下文句子→生成 QA 对"的合成数据策略，用于训练小模型完成上下文归因任务（即为 QA 系统的回答找到支撑证据句），在多个 QA 任务和跨域场景中显著优于零样本推理和 LLM 集成方法。

**[QQSUM: A Novel Task and Model of Quantitative Query-Focused Summarization for Review-based Product Question Answering](nlp_understanding/qqsum_a_novel_task_and_model_of_quantitative_query-focused_summarization_for_rev.md)**

:   提出 QQSUM 任务和 QQSUM-RAG 框架，通过 KP 导向检索与聚类、Next-KP-Generation 训练策略，从产品评论中生成包含多元观点及其流行度量化的 Key Point 摘要，解决传统 PQA 只输出单一视角答案的问题。

**[Recursive Question Understanding for Complex Question Answering over Heterogeneous Personal Data](nlp_understanding/recursive_question_understanding_for_complex_question_answering_over_heterogeneo.md)**

:   提出 ReQAP 方法，通过递归问题分解构建可执行算子树，在结构化+非结构化的异构个人数据上实现复杂问答，支持端侧轻量部署。

**[ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision](nlp_understanding/rescore_multihop_qa.md)**

:   提出 ReSCORE，利用 LLM 生成的文档-问题相关性（relevance）和文档-答案一致性（consistency）的联合概率作为伪标签，在迭代 RAG 框架中无监督训练 dense retriever，在三个多跳 QA 数据集上达到 SOTA。

**[Rethinking Semantic Parsing for Large Language Models: Enhancing LLM Performance with Semantic Hints](nlp_understanding/rethinking_semantic_parsing_for_large_language_models_enhancing_llm_performance_.md)**

:   针对"语义解析结果直接输入LLM反而降低性能"这一反直觉现象，提出SENSE——一种在prompt中嵌入语义提示（而非显式解析结果）的零样本方法，在GLUE理解任务、机器翻译、复述和简化等生成任务上一致性地提升了LLM表现。

**[RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](nlp_understanding/rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)**

:   提出 RISE——结合 RAG 与自迭代训练的多跳问答框架，通过问题分解、检索阅读、自我批判三个动作的自我探索循环，迭代生成训练数据并多目标优化模型，在 2Wiki/HotpotQA/MuSiQue 上超越 GPT-3.5 和所有 8B 级基线。

**[Self-Critique Guided Iterative Reasoning for Multi-hop Question Answering](nlp_understanding/self-critique_guided_iterative_reasoning_for_multi-hop_question_answering.md)**

:   提出 SiGIR 框架，通过端到端训练使模型具备迭代问题分解、检索、推理和自我评估能力，在推理阶段利用自我批评反馈引导 iteration-level beam search 选择最优推理路径，在三个多跳 QA 数据集上平均超越 SOTA 8.6%。

**[Sentiment Reasoning for Healthcare](nlp_understanding/sentiment_reasoning_for_healthcare.md)**

:   提出"情感推理"（Sentiment Reasoning）新任务，要求模型在预测医疗对话情感标签的同时生成解释理据，并构建了覆盖五种语言的 30K 样本多模态情感分析数据集，通过理据增强训练在分类准确率和 macro-F1 上均提升约 2%。

**[SynGraph: A Dynamic Graph-LLM Synthesis Framework for Sparse Streaming User Sentiment Analysis](nlp_understanding/syngraph_a_dynamic_graph-llm_synthesis_framework_for_sparse_streaming_user_senti.md)**

:   提出SynGraph框架，在连续时间动态图上将稀疏用户分为mid-tail/long-tail/extreme三类，针对不同稀疏程度利用LLM合成增强数据（结合局部-全局图理解、高阶关系和画像生成），有效缓解流式评论情感分析中的数据稀疏问题。

**[A Variational Approach for Mitigating Entity Bias in Relation Extraction](nlp_understanding/variational_approach_mitigating_entity_bias_relation_extraction.md)**

:   提出将变分信息瓶颈(VIB)应用于关系抽取中的实体去偏，通过将实体映射到概率分布 $\mathcal{N}(\mu, \sigma)$ 来压缩实体特定信息同时保留任务相关特征，方差 $\sigma^2$ 可量化模型对实体vs上下文的依赖程度，在TACRED、REFinD、BioRED三个域的ID和OOD设置上均达到SOTA。

---

## 💻 代码智能 { #code_intelligence }

**[LongCodeU: Benchmarking Long-Context Language Models on Long Code Understanding](code_intelligence/benchmarking_long-context_language_models_on_long_code_understanding.md)**

:   提出 LongCodeU 基准，从代码单元感知、单元内理解、单元间关系理解和长文档理解四个维度设计 8 个任务，评估 9 个长上下文语言模型在真实仓库级长代码上的理解能力，揭示 32K token 是当前 LCLM 长代码理解的实际上限。

**[Beyond Sequences: Two-dimensional Representation and Dependency Encoding for Code Generation](code_intelligence/beyond_sequences_two-dimensional_representation_and_dependency_encoding_for_code.md)**

:   本文提出超越传统一维序列表示的二维代码表示方法，通过显式编码代码的结构依赖关系（如语法树结构和变量依赖），显著提升了代码生成的准确性和结构正确性。

**[CoCo-Bench: A Comprehensive Code Benchmark for Multi-task Large Language Model Evaluation](code_intelligence/coco-bench_a_comprehensive_code_benchmark_for_multi-task_large_language_model_ev.md)**

:   提出 CoCo-Bench（Comprehensive Code Benchmark），一个覆盖代码理解、代码生成、代码修改和代码审查四个维度的综合代码基准，支持多编程语言和多难度等级，通过严格的人工审核确保数据质量，揭示了现有 LLM 在代码能力上的不均衡表现。

**[CodeDPO: Aligning Code Models with Self Generated and Verified Source Code](code_intelligence/codedpo_code_alignment.md)**

:   提出 CodeDPO，通过 PageRank 启发的自验证评分机制从自生成代码中构造高质量偏好对（93K 正确性 + 21K 效率），DPO 训练后在 8 个代码模型上 HumanEval 平均提升 10+ 分，同时提升代码执行效率 1.25-1.45×。

**[CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation](code_intelligence/codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod.md)**

:   提出 CodeIF，第一个系统性评估 LLM 在代码生成中指令遵循能力的基准，含 8 大类 50 个细粒度约束指令、4 种新评估指标，并对 35 个 SOTA 模型进行全面评估。

**[CodeReviewQA: The Code Review Comprehension Assessment for Large Language Models](code_intelligence/codereviewqa_the_code_review_comprehension_assessment_for_large_language_models.md)**

:   提出 CodeReviewQA 基准，将代码审查自动修正（ACR）任务分解为三个中间推理步骤——变更类型识别（CTR）、变更定位（CL）、解决方案识别（SI），各自设计为不同难度的多选题探测，在 900 个人工验证的高质量样例（9 种语言）上评测 72 个 LLM，揭示了模型在代码审查理解中的具体弱点。

**[CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System](code_intelligence/compileagent_automated_real-world_repo-level_compilation_with_tool-integrated_ll.md)**

:   提出 CompileAgent，首个面向仓库级代码编译的 LLM Agent 框架，集成五种专用工具和流程化 Agent 策略，在 100 个 C/C++ 真实项目的 CompileAgentBench 上将编译成功率最高提升 71%，平均每个项目仅需 $0.22。

**[CoRet: Improved Retriever for Code Editing](code_intelligence/coret_improved_retriever_for_code_editing.md)**

:   提出 CoRet，一个面向代码编辑任务的稠密检索模型，通过整合代码语义、仓库文件层级结构和调用图依赖关系，并使用针对仓库级检索设计的对数似然损失函数，在 SWE-bench 和 Long Code Arena 上比现有模型的 Recall 至少提升 15 个百分点。

**[DARS: Dynamic Action Re-Sampling to Enhance Coding Agent Performance by Adaptive Tree Traversal](code_intelligence/dars_dynamic_action_re-sampling_to_enhance_coding_agent_performance_by_adaptive_.md)**

:   本文提出 DARS（动态动作重采样），一种针对编程智能体的推理时计算扩展方法，在智能体做出次优决策的关键节点上动态分支并尝试替代动作，在 SWE-Bench Lite 上以 Claude 3.5 Sonnet V2 实现 55% 的 pass@k 和 47% 的 pass@1，超越当时开源 SOTA 框架。

**[DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](code_intelligence/dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)**

:   提出 DynaCode，一个动态复杂度感知的代码生成基准，通过将代码问题按圈复杂度分类并用调用图（Call Graph）组合嵌套，生成约 1.89 亿个唯一问题，有效缓解数据污染并系统评估 LLM 在不同复杂度下的代码生成能力。

**[ETF: An Entity Tracing Framework for Hallucination Detection in Code Summaries](code_intelligence/etf_an_entity_tracing_framework_for_hallucination_detection_in_code_summaries.md)**

:   提出 Entity Tracing Framework (ETF)，一种通过静态程序分析提取代码实体、再用 LLM 验证这些实体在生成摘要中是否被正确描述的幻觉检测框架，配合首创的 CodeSumEval 数据集（~10K样本），在代码摘要幻觉检测上达到 73% F1。

**[ExploraCoder: Advancing Code Generation for Multiple Unseen APIs via Planning and Chained Exploration](code_intelligence/exploracoder_advancing_code_generation_for_multiple_unseen_apis_via_planning_and.md)**

:   提出无需额外训练的 ExploraCoder 框架，通过任务规划将复杂多 API 编程问题分解为子任务，再通过链式 API 探索（CoAE）逐步实验并积累正确的 API 用法经验，在多 API 不可见库基准上 pass@10 绝对提升最高 17.28%。

**[FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation](code_intelligence/feabench_repo_code_gen.md)**

:   提出 FEA-Bench——首个评估 LLM 在仓库级代码库中实现新特性（Feature Implementation）能力的基准，包含来自 83 个 GitHub 仓库的 1401 个任务实例，每个实例配有单元测试。最强模型 DeepSeek-R1 仅解决约 10% 的任务，揭示了仓库级增量开发对当前 LLM 的巨大挑战。

**[GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding](code_intelligence/galla_graph_aligned_large_language_models.md)**

:   提出 GALLa，通过 GNN 编码代码的 AST/DFG 结构图并用跨模态适配器对齐到 LLM 嵌入空间，在微调时作为辅助任务注入代码结构信息，推理时丢弃 GNN 和 adapter 实现零额外开销，在 5 个代码任务 × 7 个基线 LLM（350M-14B）上持续提升。

**[GiFT: Gibbs Fine-Tuning for Code Generation](code_intelligence/gift_gibbs_fine_tuning_code_gen.md)**

:   提出 Gibbs Fine-Tuning（GiFT），受 Gibbs 采样启发，通过"代码→描述→代码"的迭代翻译从边际分布而非条件分布中采样自生成代码，结合困惑度引导的长尾数据选择，在 APPS+/MBPP+/CodeInsight 上比标准自训练提升最高 9.8%。

**[MLDebugging: Towards Benchmarking Code Debugging Across Multi-Library Scenarios](code_intelligence/mldebugging_towards_benchmarking_code_debugging_across_multi-library_scenarios.md)**

:   本文提出 MLDebugging——首个面向**多库 Python 代码调试**的综合基准，涵盖 126 个 Python 库和 7 种 bug 类型（共 1175 个样本），系统评估主流开源和闭源 LLM 在多库调试场景下的能力，发现当前 LLM 在此任务上仍有很大提升空间。

**[OASIS: Order-Augmented Strategy for Improved Code Search](code_intelligence/oasis_order-augmented_strategy_for_improved_code_search.md)**

:   提出OASIS方法，通过为负样本对引入基于序的相似度标签来捕捉代码语义中的细微差异，结合InfoNCE和CoSENT双重损失函数训练代码嵌入模型，在CoSQA、AdvTest和CodeSearchNet三个基准的NL2Code和Code2Code搜索任务上全面超越现有SOTA。

**[Personality-Guided Code Generation Using Large Language Models](code_intelligence/personality_guided_code_gen.md)**

:   用 GPT-4o 为每个编程任务动态生成适配的 MBTI 人格类型和详细描述，再让目标 LLM 以该人格角色扮演程序员生成代码，在 7 个 LLM × 4 个数据集的 28 个组合中 23 个取得 pass rate 提升，最高达 12.9%，关键因素是人格多样性而非某个特定人格。

**[Program Synthesis Benchmark for Visual Programming in XLogoOnline Environment](code_intelligence/program_synthesis_benchmark_for_visual_programming_in_xlogoonline_environment.md)**

:   本文基于 XLogoOnline 视觉编程环境构建了一个需要空间规划、编程和逻辑推理等多技能组合的程序合成基准，发现 GPT-4V 仅能解决 20% 的任务，但通过 8 万+合成数据微调加上模拟器驱动的课程学习，Llama3-8B 大幅超越了 GPT-4V 和 Llama3-70B。

**[ReflectionCoder: Learning from Reflection Sequence for Enhanced One-off Code Generation](code_intelligence/reflectioncoder_learning_from_reflection_sequence_for_enhanced_one-off_code_gene.md)**

:   ReflectionCoder通过构建整合编译器反馈的"反思序列"（reflection sequence）数据，结合反思自蒸馏和动态掩码蒸馏两种训练策略，使模型在一次性代码生成中达到SOTA性能，无需运行时的多轮调试。

**[Rethinking Repetition Problems of LLMs in Code Generation](code_intelligence/rethinking_repetition_problems_of_llms_in_code_generation.md)**

:   本文重新定义了代码生成中的重复问题，区分出比内容重复更普遍且更难处理的"结构性重复"，并提出基于语法规则的重复惩罚解码方法RPG（Repetition Penalization based on Grammar），在新构建的CodeRepetEval和标准基准上大幅缓解重复问题。

**[Revisit Self-Debugging with Self-Generated Tests for Code Generation](code_intelligence/revisit_self-debugging_with_self-generated_tests_for_code_generation.md)**

:   系统性地研究了使用 LLM 自生成测试进行自调试（self-debugging）的效果，发现基于后执行信息的自调试在基础编程问题上反而降低性能（因自生成测试偏差），但基于执行中间状态（in-execution）的自调试可有效规避该偏差，在基础和竞赛题上均有提升。

**[SceneGenAgent: Precise Industrial Scene Generation with Coding Agent](code_intelligence/scenegenagent_precise_industrial_scene_generation_with_coding_agent.md)**

:   提出 SceneGenAgent，一个基于 LLM 的代码生成 Agent，通过结构化布局规划、布局验证和迭代优化流程，利用 C# 代码精确生成工业场景，在真实工业任务上达到 81% 成功率，并构建 SceneInstruct 数据集使开源 LLM 接近 GPT-4o 水平。

**[SHARE: An SLM-based Hierarchical Action CorREction Assistant for Text-to-SQL](code_intelligence/share_text_to_sql_correction.md)**

:   提出 SHARE 框架，用三个 <8B 参数的专用小语言模型（SLM）组成顺序管道，将声明式 SQL 转换为可暴露推理路径的步进动作轨迹，再分阶段修正 schema 链接错误与逻辑推理错误，以极低成本实现 LLM 的 Text-to-SQL 自纠正。

**[STaR-SQL: Self-Taught Reasoner for Text-to-SQL](code_intelligence/star-sql_self-taught_reasoner_for_text-to-sql.md)**

:   将 Text-to-SQL 任务重新定义为推理驱动的过程，通过 STaR（Self-Taught Reasoner）自举方法让 LLM 学习生成逐步推理来辅助 SQL 生成，并集成 ORM 验证器进行 best-of-N 采样，在 Spider 基准上达到 86.6% 执行准确率。

**[TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs](code_intelligence/texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)**

:   提出TeXpert——首个系统评估LLM从自然语言指令生成科学文档LaTeX代码能力的多难度级别基准，包含440个高质量样本（Simple/Average/Hard三级），在9个开闭源LLM上的评估揭示了LaTeX生成是LLM的显著短板（Hard任务准确率普遍低于17.5%），逻辑错误和格式错误是主要瓶颈。

**[Tree-of-Code: A Tree-Structured Exploring Framework for End-to-End Code Generation](code_intelligence/tree-of-code_a_tree-structured_exploring_framework_for_end-to-end_code_generatio.md)**

:   提出 Tree-of-Code（ToC）框架，通过树结构组织端到端的完整代码程序（CodeProgram）节点，结合基于执行结果的反思机制和提示/模型随机探索策略，在无需标注数据的零样本设置下，以不到 1/4 的交互轮次实现了比 CodeAct 高近 20% 的复杂任务准确率。

**[Tree-of-Evolution: Tree-Structured Instruction Evolution for Code Generation in Large Language Models](code_intelligence/tree_of_evolution_code_gen.md)**

:   提出Tree-of-Evolution (ToE)——一种树结构代码指令合成框架，通过多路径进化和质量驱动优化克服Code Evol-Instruct和OSS-Instruct的单向合成与随机生成限制，仅用75K合成数据微调base model即可达到或超越Qwen2.5-Coder-Instruct（数百万样本微调）的性能。

**[UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench](code_intelligence/utboost_rigorous_evaluation_of_coding_agents_on_swe-bench.md)**

:   本文提出 UTBoost 框架，通过基于 LLM 的测试用例生成器 UTGenerator 和改进的解析器来增强 SWE-Bench 的测试用例覆盖率，发现 36 个测试不充分的实例和 345 个被错误标记为通过的补丁，导致 SWE-Bench Lite 排行榜 40.9% 和 SWE-Bench Verified 24.4% 的排名发生变化。

---

## 👥 社会计算 { #social_computing }

**[A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models](social_computing/a_survey_on_proactive_defense_strategies_against_misinformation_in_large_languag.md)**

:   提出从被动检测到主动防御的范式转换，构建知识可信度、推理可靠性、输入鲁棒性"三支柱"框架，将 127 种防御技术系统映射到三支柱中，元分析 48 项基准研究表明主动防御相比传统方法提升 42-63%，同时识别了计算开销和跨域泛化的非平凡权衡。

**[BanStereoSet: A Dataset to Measure Stereotypical Social Biases in LLMs for Bangla](social_computing/banstereoset_a_dataset_to_measure_stereotypical_social_biases_in_llms_for_bangla.md)**

:   构建 BanStereoSet，一个包含 1194 条填空式样本、覆盖 9 类偏见（种族/性别/宗教/职业/美貌/年龄/种姓/地区等）的孟加拉语刻板印象偏见数据集，用于评估多语言 LLM 在孟加拉语中的社会偏见，发现 GPT-4o 偏见最高，Mistral 最低。

**[Beyond Negative Stereotypes -- Non-Negative Abusive Utterances about Identity Groups and Their Semantic Variants](social_computing/beyond_negative_stereotypes_--_non-negative_abusive_utterances_about_identity_gr.md)**

:   本文研究了一种被忽视的仇恨言论类型——表面上不包含负面刻板印象但实际上针对身份群体的滥用性表达，系统分析了这类"非负面滥用话语"的语义变体，并评估了现有检测模型的处理能力。

**[BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models](social_computing/biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)**

:   提出 BiasGuard，通过显式推理公平性规范来检测 LLM 输出偏见：第一阶段用教师模型生成推理轨迹做 SFT 初始化，第二阶段用 DPO 强化推理质量，在 5 个数据集上超越分类器和 LLM-as-Judge 方法且降低过度公平误判。

**[Can Community Notes Replace Professional Fact-Checkers?](social_computing/can_community_notes_replace_professional_fact-checkers.md)**

:   大规模分析 Twitter/X 社区笔记 66.4 万条，发现社区笔记对专业事实核查的依赖是此前报告的 5 倍（≥5-7%），涉及阴谋论/虚假叙事的内容引用事实核查来源的概率是其他内容的 2 倍，证明高质量社区审核与专业事实核查深度交织、不可替代。

**[Conspiracy Theories and Where to Find Them on TikTok](social_computing/conspiracy_theories_and_where_to_find_them_on_tiktok.md)**

:   首个TikTok阴谋论系统性分析：通过官方API收集美国150万条长视频，利用标签富集和远程监督识别阴谋论内容（每月约1000条新视频），评估TikTok创作者激励计划的影响，并测试开源LLM（Llama3、Mistral、Gemma）在基于音频转录的阴谋论检测上的效果（精确率高达96%但整体水平与微调RoBERTa相当）。

**[Culture Matters in Toxic Language Detection in Persian](social_computing/culture_matters_in_toxic_language_detection_in_persian.md)**

:   本文系统比较了多种方法（微调、数据增强、零样本/少样本学习、跨语言迁移学习）在波斯语有害语言检测中的效果，揭示了文化相似性是决定跨语言迁移学习成功与否的关键因素——来自文化相近国家的语言数据带来更好的迁移效果。

**[Detection of Human and Machine-Authored Fake News in Urdu](social_computing/detection_of_human_and_machine-authored_fake_news_in_urdu.md)**

:   本文提出了乌尔都语四分类假新闻检测任务（人类假/人类真/机器假/机器真），构建了首个乌尔都语机器生成新闻数据集，并提出层次化检测方法将四分类分解为机器文本检测和假新闻检测两个子任务，在域内和跨域设置中均优于基线。

**[Explicit vs. Implicit: Investigating Social Bias in Large Language Models through Self-Reflection](social_computing/explicit_vs_implicit_investigating_social_bias_in_large_language_models_through_.md)**

:   借鉴社会心理学中隐式联想测验（IAT）和自我报告评估（SRA），提出自反思评估框架系统研究 LLM 的显式和隐式偏见，发现 LLM 与人类一样存在显式-隐式偏见不一致——显式偏见轻微但隐式偏见强烈，且模型越大/对齐训练越多，这种不一致越严重。

**[Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language](social_computing/exploring_gender_bias_in_large_language_models_an_in-depth_dive_into_the_german_.md)**

:   本文针对德语场景构建了五个性别偏见评测数据集，并在八个多语言 LLM 上进行系统评估，揭示了德语特有的性别偏见挑战——包括阳性职业名词的歧义解读和看似中性的名词对性别感知的影响。

**[Exploring Multimodal Challenges in Toxic Chinese Detection: Taxonomy, Benchmark, and Findings](social_computing/exploring_multimodal_challenges_in_toxic_chinese_detection_taxonomy_benchmark_an.md)**

:   这篇工作把中文毒性文本中的“形、音、义混合扰动”系统化为 3 类 8 种策略，构建了大规模扰动基准 CNTP，并证明当前中美主流 LLM 在这类中文多模态毒性检测上都明显不稳，而小样本 ICL / SFT 虽能抬高检出率，却容易把正常内容一起误杀。

**[Exploring the Impact of Instruction-Tuning on LLMs' Susceptibility to Misinformation](social_computing/exploring_the_impact_of_instruction-tuning_on_llms_susceptibility_to_misinformat.md)**

:   首次系统研究指令微调如何影响 LLM 对虚假信息的易感性，发现指令微调使模型从偏信 assistant-role 转变为偏信 user-role，当虚假信息以独立 user-turn 呈现时易感性最高，揭示了指令微调的"副作用"。

**[FairSteer: Inference Time Debiasing for LLMs with Dynamic Activation Steering](social_computing/fairsteer_inference_time_debiasing_for_llms_with_dynamic_activation_steering.md)**

:   提出 FairSteer，一种推理时去偏框架，通过轻量线性分类器检测激活中的偏见信号，再用对比 prompt 对计算的去偏转向向量（DSV）动态调整隐藏层激活，无需重训即可在多任务上有效缓解 LLM 的社会偏见。

**[GG-BBQ: German Gender Bias Benchmark for Question Answering](social_computing/gg-bbq_german_gender_bias_benchmark_for_question_answering.md)**

:   将英语BBQ偏见基准数据集的性别子集翻译为德语，经人工审校后创建GG-BBQ德语性别偏见评估基准，揭示了机器翻译在性别偏见评估数据集构建中的局限性，并评估了多个德语LLM的偏见表现。

**[HateDay: Insights from a Global Hate Speech Dataset Representative of a Day on Twitter](social_computing/hateday_global_hate_speech.md)**

:   HateDay 构建了首个全球代表性仇恨言论数据集——24 万条随机采样的 Twitter 推文覆盖 8 种语言和 4 个英语国家，揭示了学术数据集大幅高估了检测模型在真实场景中的表现，尤其对非欧洲语言检测能力极差。

**[How does Misinformation Affect Large Language Model Behaviors and Preferences?](social_computing/how_does_misinformation_affect_large_language.md)**

:   构建了目前最大的误信息评估基准 MisBench（1034 万条误信息），从知识冲突类型和文本风格两个维度系统分析 LLM 对误信息的行为和偏好，并提出 RtD 方法结合外部知识源提升误信息检测能力。

**[ImpliHateVid: Implicit Hate Speech Detection in Videos](social_computing/implihatevid_video_hate.md)**

:   首次提出视频中隐性仇恨言论检测任务，构建2009个视频的ImpliHateVid数据集，并设计两阶段对比学习框架融合文本、图像、音频三模态特征。

**[Is LLM an Overconfident Judge? Unveiling the Capabilities of LLMs in Detecting Offensive Language with Annotation Disagreement](social_computing/is_llm_an_overconfident_judge_unveiling_the_capabilities_of_llms_in_detecting_of.md)**

:   系统评估了多个 LLM 在攻击性语言检测中面对标注分歧时的表现，发现 LLM 在标注者高度一致的样本上表现优异（GPT-4o F1 85.24%）但在低一致度样本上骤降至 57.06%，且模型对不确定样本表现出严重的过度自信；进一步通过 few-shot 和指令微调实验证明，在训练中引入分歧样本可同时提升检测准确率和人-AI 对齐度。

**[K/DA: Automated Data Generation Pipeline for Detoxifying Implicitly Offensive Language in Korean](social_computing/kda_automated_data_generation_pipeline_for_detoxifying_implicitly_offensive_lang.md)**

:   本文提出 K/DA，一个自动化的韩语攻击性语言配对数据生成管线，通过 RAG 从在线社区检索时下流行的俚语来增强中性句子生成毒性版本，配合两阶段过滤（配对一致性 + 隐性攻击性），生成了 7.5K 高质量中性-毒性配对数据集，训练的去毒化模型优于基于人工标注和翻译数据集训练的模型。

**[Synergizing LLMs with Global Label Propagation for Multimodal Fake News Detection](social_computing/llm_label_propagation.md)**

:   提出 GLPN-LLM 框架，通过 mask-based 全局标签传播机制有效整合 LLM 生成的伪标签，解决了 LLM 伪标签直接组合效果不佳的问题，在 Twitter/PHEME/Weibo 三个数据集上全面超越 SOTA。

**[Evaluation of LLM Vulnerabilities to Being Misused for Personalized Disinformation Generation](social_computing/llm_personalized_disinformation.md)**

:   系统评估了 6 个主流 LLM 生成个性化虚假信息的能力，发现大多数 LLM 能生成高质量个性化虚假新闻，且个性化请求反而降低了安全过滤器的触发率（相当于一种 jailbreak），同时轻微降低了机器生成文本的可检测性。

**[MDiT-Bench: Evaluating the Dual-Implicit Toxicity in Large Multimodal Models](social_computing/mdit-bench_evaluating_the_dual-implicit_toxicity_in_large_multimodal_models.md)**

:   提出"双模态隐式毒性"(dual-implicit toxicity)概念——仅当结合图文两个模态时才能被识别的偏见与歧视，构建了包含317K问题、12类23子类的MDIT-Bench基准，并通过长上下文越狱揭示了主流多模态大模型中大量可被激活的隐藏毒性。

**[Measuring Social Biases in Masked Language Models by Proxy of Prediction Quality](social_computing/measuring_social_biases_in_masked_language_models_by_proxy_of_prediction_quality.md)**

:   提出了注意力加权的预测质量代理度量 Δpa 和 CRRA，在迭代掩码实验（IME）下评估 MLM 的社会偏见，并引入模型比较函数 BSRT 来估计重训练引入的偏见，发现所提方法比 CSPS、AUL、AULA 等现有方法更准确、更敏感。

**[Silencing Empowerment, Allowing Bigotry: Auditing the Moderation of Hate Speech on Twitch](social_computing/silencing_empowerment_allowing_bigotry_auditing_the_moderation_of_hate_speech_on.md)**

:   对 Twitch 平台的自动化内容审核工具 AutoMod 进行大规模审计，发送超过 10.7 万条消息，发现 AutoMod 在最严格设置下仅能标记 22% 的仇恨内容，高度依赖侮辱性词汇作为检测信号，同时错误屏蔽高达 89.5% 的教育性/赋权性内容。

**[STATE ToxiCN: A Benchmark for Span-level Target-Aware Toxicity Extraction in Chinese Hate Speech Detection](social_computing/state_toxicn_a_benchmark_for_span-level_target-aware_toxicity_extraction_in_chin.md)**

:   构建了首个中文 span 级仇恨言论检测数据集 STATE ToxiCN（8029 条帖子、9533 个四元组标注），提出 Target-Argument-Hateful-Group 四元组标注体系，并首次建立了中文仇恨俚语标注词典（830 条），系统评估了多种 LLM 在 span 级中文仇恨言论检测上的能力。

**[taz2024full: Analysing German Newspapers for Gender Bias and Discrimination across Decades](social_computing/taz2024full_analysing_german_newspapers_for_gender_bias_and_discrimination_acros.md)**

:   构建迄今最大的公开德语新闻语料库 taz2024full（180万+篇文章，1980-2024），并适配actor级语篇分析管线至德语，揭示四十余年间新闻报道中持续存在的性别表征失衡与情感偏差。

**[Translate With Care: Addressing Gender Bias, Neutrality, and Reasoning in Large Language Model Translations](social_computing/translate_with_care_addressing_gender_bias_neutrality_and_reasoning_in_large_lan.md)**

:   提出 Translate-with-Care (TWC) 数据集（3,950 条跨 6 种无性别语言的翻译挑战），系统揭示 GPT-4、Google Translate 等模型在无性别→有性别语言翻译中的性别偏见和推理错误，并通过微调 mBART-50 在偏见消除和翻译准确率上大幅超越闭源 LLM。

---

## ✍️ 文本生成 { #nlp_generation }

**[A Representation Level Analysis of NMT Model Robustness to Grammatical Errors](nlp_generation/a_representation_level_analysis_of_nmt_model_robustness_to_grammatical_errors.md)**

:   从表示层面系统分析 NMT 编码器如何处理语法错误——发现编码器先在浅层"检测"错误（GED 探测 F1 上升），再在深层"纠正"错误（CKA 距离下降），并提出 Robustness Heads 概念识别出参与纠正的具体注意力头，在 4 个模型×5 个语言方向上验证了该"检测→纠正"两阶段机制。

**[Abstractive Snippet Generation](nlp_generation/abstractive_snippet_generation.md)**

:   本文提出了一种面向搜索引擎的抽象化片段生成方法，通过查询感知的摘要生成技术，为搜索结果页面生成比传统抽取式片段更简洁、信息量更大的文本摘要，显著提升用户搜索体验。

**[An Empirical Study of Many-to-Many Summarization with Large Language Models](nlp_generation/an_empirical_study_of_manytomany_summarization.md)**

:   首次系统研究LLM在多对多摘要（M2MS）任务上的表现，整合8个数据集构建涵盖5个领域6种语言的47.8K样本基准，评测18个LLM发现零样本LLM可媲美微调传统模型，指令微调后显著超越，但事实性问题仍是关键瓶颈。

**[ATGen: A Framework for Active Text Generation](nlp_generation/atgen_a_framework_for_active_text_generation.md)**

:   提出ATGen——首个系统化的NLG主动学习框架，集成SOTA AL策略、人工/LLM标注界面、PEFT高效训练和vLLM推理优化，在TriviaQA/GSM8K等4个NLG任务上验证主动学习可将标注成本降低2-4倍。

**[Balancing Diversity and Risk in LLM Sampling: How to Select Your Method and Parameter for Open-Ended Text Generation](nlp_generation/balancing_diversity_and_risk_in_llm_sampling_how_to_select_your_method_and_param.md)**

:   本文提出了一种基于上下文保持前缀树（CP-Trie）的系统性评估框架，通过不依赖概率和参数调优的指标来评估截断采样方法在多样性与风险之间的内在适应能力，并为实际应用中的参数选择提供指导。

**[CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation](nlp_generation/cocolex_legal_text_gen.md)**

:   提出 CoCoLex，一种无需训练的解码策略，利用解码过程中隐状态与上下文 token 隐状态的欧氏距离构造复制分布，并通过基于预测熵的置信度分数动态平衡"从上下文复制"与"自由生成"的比例，在五个法律基准上一致提升忠实性和正确性，尤其在长文本生成任务中效果突出。

**[Context-Aware Hierarchical Merging for Long Document Summarization](nlp_generation/context-aware_hierarchical_merging_for_long_document_summarization.md)**

:   提出上下文感知的层次合并（CAHM）方法，通过在层次合并摘要过程中引入源文档的相关上下文（抽取/检索/引用三种方式），有效缓解 LLM 在超长文档（>100K tokens）摘要中的幻觉问题。

**[Decomposed Opinion Summarization with Verified Aspect-Aware Modules](nlp_generation/decomposed_opinion_summarization_with_verified_aspect-aware_modules.md)**

:   本文将观点摘要（opinion summarization）任务分解为三个可逐步验证的模块——方面识别、观点汇总、元评论合成，通过 LLM 零样本提示实现领域无关的模块化处理，在科研论文、商业评论和产品评论三个领域生成了更可追溯、更全面的摘要。

**[Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems](nlp_generation/dehumanizing_machines_anthropomorphic.md)**

:   通过文献综述和众包研究，系统整理出 21 类干预措施来降低文本生成系统输出的拟人化程度，提出包含干预类型、目标行为、操作化方式和负面影响四个维度的概念框架，为去拟人化研究提供最全面的基础设施。

**[Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport](nlp_generation/doc_level_mbr_optimal_transport.md)**

:   提出 MBR-OT，将最优传输（Wasserstein距离）引入最小贝叶斯风险（MBR）解码，实现用句子级效用函数评估文档级输出质量，在文档级机器翻译、文本简化和密集图像描述任务上显著优于标准 MBR 解码。

**[DTCRS: Dynamic Tree Construction for Recursive Summarization](nlp_generation/dtcrs_dynamic_tree_construction_for_recursive_summarization.md)**

:   提出 DTCRS 方法，根据文档结构和查询语义动态构建摘要树，通过问题分解和子问题引导聚类减少冗余摘要节点，在三个 QA 数据集上显著优于静态摘要树方法 RAPTOR。

**[Enhancing Text Editing for Grammatical Error Correction: Arabic as a Case Study](nlp_generation/enhancing_text_editing_for_grammatical_error_correction_arabic_as_a_case_study.md)**

:   本文提出一种无需语言特定编辑集的通用文本编辑方法（SWEET），通过数据驱动的编辑标签自动提取和压缩策略，首次成功将文本编辑范式应用于阿拉伯语语法纠错，在多个基准上达到SOTA且推理速度提升6倍以上。

**[Multi-document Summarization through Multi-document Event Relation Graph Reasoning in LLMs](nlp_generation/event_graph_bias_mitigation_summarization.md)**

:   构建多文档事件关系图（包含四类文档内事件关系、跨文档事件共指、事件级道德观点），通过图文本化和图提示微调两种策略将偏见信息注入 LLM，生成去偏见的中立化摘要，在内容保留和偏见消除上均优于基线。

**[gec-metrics: A Unified Library for Grammatical Error Correction Evaluation](nlp_generation/gec-metrics_a_unified_library_for_grammatical_error_correction_evaluation.md)**

:   提出 gec-metrics 统一库，将 10 种语法纠错 (GEC) 评估指标整合到统一接口中，并提供元评估功能，解决了现有 GEC 评估实现碎片化、不可复现、难以扩展的问题。

**[IMPARA-GED: Grammatical Error Detection is Boosting Reference-free Grammatical Error Quality Estimator](nlp_generation/impara-ged_grammatical_error_detection_is_boosting_reference-free_grammatical_er.md)**

:   在 IMPARA 的质量估计器构建之前，增加一步语法错误检测（GED）预训练，同时去掉失效的相似度估计器，使无参考 GEC 评估在 SEEDA 上达到句子级最高相关性。

**[Odysseus Navigates the Sirens' Song: Dynamic Focus Decoding for Factual and Diverse Open-Ended Text Generation](nlp_generation/odysseus_dynamic_focus_decoding.md)**

:   提出动态聚焦解码（DFD），通过追踪 LLM 各层间分布差异（KL 散度）来识别知识密集型解码步骤，自适应调整温度——知识密集步用低温保事实性，非知识密集步用高温促多样性——在七个数据集上同时提升事实性和多样性。

**[Personalized Text Generation with Contrastive Activation Steering](nlp_generation/personalized_text_generation_with_contrastive_activation_steering.md)**

:   提出 StyleVector——一个无需训练的个性化文本生成框架，通过对比用户真实响应与模型生成的无风格响应之间的隐层激活差异来提取"风格向量"，在推理时通过简单的线性激活干预引导 LLM 生成符合用户写作风格的文本，在 LaMP 和 LongLaMP 基准上实现 8% 的相对提升，同时将存储需求降低至 PEFT 方法的 1/1700。

**[PerSphere: A Comprehensive Framework for Multi-Faceted Perspective Retrieval and Summarization](nlp_generation/persphere_a_comprehensive_framework_for_multi-faceted_perspective_retrieval_and_.md)**

:   > 提出 PerSphere 基准数据集和 MURS（Multi-faceted perspective retrieval and summarization）任务，旨在从文档集中检索并全面总结争议性问题的多面向观点，并提出分层多智能体总结系统 HierSphere 来缓解长上下文和观点提取的挑战。

**[Rethinking Evaluation Metrics for Grammatical Error Correction: Why Use a Different Evaluation Process than Human?](nlp_generation/rethinking_evaluation_metrics_for_grammatical_error_correction_why_use_a_differe.md)**

:   本文指出当前 GEC 自动评估与人工评估在"从句级分数到系统排名"的聚合流程上存在根本差异——人工评估用句级两两比较+TrueSkill 排名算法，而自动评估用平均绝对分数+排序——并通过在自动评估中也采用 TrueSkill 聚合来弥补这一差距，在 SEEDA 基准上大幅提升多数指标与人工评估的相关性，甚至使 BERT 级指标超越 GPT-4。

**[TagRouter: Learning Route to LLMs through Tags for Open-Domain Text Generation Tasks](nlp_generation/tagrouter_learning_route_to_llms_through_tags_for_open-domain_text_generation_ta.md)**

:   这篇论文提出 TagRouter，用一个小型标签生成器把开放域文本生成请求先压缩成一组语义标签，再基于标签统计每个候选 LLM 的相对优势并进行路由，从而在不重新训练路由器的前提下，把多模型系统的接受率做得比单个大模型更高，同时显著降低推理成本。

**[Tell, Don't Show: Leveraging Language Models' Abstractive Retellings to Model Literary Themes](nlp_generation/tell_dont_show_leveraging_language_models_abstractive_retellings_to_model_litera.md)**

:   提出 Retell 方法：利用小型 LM 对文学段落进行抽象复述（abstractive retelling），

**[Theme-Explanation Structure for Table Summarization Using Large Language Models](nlp_generation/theme-explanation_structure_for_table_summarization_using_large_language_models_.md)**

:   提出 Tabular-TX 管线，通过多步 CoT 推理实现深度表格理解、记者角色 prompt 生成清晰句子、并将输出结构化为 Theme（主题状语）+ Explanation（解释谓语）的格式，在韩语行政表格摘要基准上不依赖微调即实现 ROUGE-1 0.51 的最佳性能，显著超越微调和纯 ICL 方法。

**[Towards Better Open-Ended Text Generation: A Multicriteria Evaluation Framework](nlp_generation/towards_better_open-ended_text_generation_a_multicriteria_evaluation_framework.md)**

:   针对开放式文本生成中多指标（coherence/diversity/perplexity）之间的权衡问题，提出三种互补的多准则评估方法——Extended Bradley-Terry 模型（序数排名）、Union-Free Generic Depth（允许不可比性的偏序）和 Q*Text（基数评估综合指标），在6个 LLM × 59种解码策略 × 180万+生成文本上验证，发现中等超参配置普遍优于极端配置，小模型+合理解码策略可匹敌大模型。

**[Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing](nlp_generation/unveiling_attractor_cycles_in_large_language_models_a_dynamical_systems_view_of_.md)**

:   本文从动力系统理论出发，发现LLM在连续释义（successive paraphrasing）过程中输出会收敛至稳定的2-周期吸引子循环，而非探索广阔的释义空间，揭示了LLM生成能力的固有局限性。

**[What Is That Talk About? A Video-to-Text Summarization Dataset for Scientific Presentations](nlp_generation/video_text_summarization.md)**

:   提出VISTA数据集——18,599个AI会议演讲视频与论文摘要配对，并引入plan-based摘要框架，通过生成中间问题序列引导科学视频的结构化摘要生成，显著提升事实一致性。

**[Writing Like the Best: Exemplar-Based Expository Text Generation](nlp_generation/writing_like_best_exemplar.md)**

:   定义"基于范例的说明文生成"新任务——给定一篇关于源主题的范例文本，生成关于目标主题的说明文，提出 Recurrent Plan-then-Adapt（RePA）框架，通过逐段模仿规划+检索增强自适应生成+双记忆机制，在 Wikipedia/RoleEE/USNews 三个数据集上显著优于 GPT-4 和 o1 基线。

---

## 🕸️ 图学习 { #graph_learning }

**[A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning](graph_learning/a_generative_adaptive_replay_continual_learning_model_for_temporal_knowledge_gra.md)**

:   本文提出深度生成自适应重放（DGAR）方法，利用预训练扩散模型生成历史实体分布表示、通过增强历史分布与当前分布的共同特征来缓解分布冲突，并设计逐层自适应重放机制整合历史与当前知识，在时序知识图谱推理的持续学习场景中显著缓解了灾难性遗忘问题。

**[A Mutual Information Perspective on Knowledge Graph Embedding](graph_learning/a_mutual_information_perspective_on_knowledge_graph_embedding.md)**

:   本文提出基于互信息最大化的知识图谱嵌入（KGE）框架，通过最大化三元组不同组成部分之间的互信息来提升实体和关系的语义表示能力，在复杂关系模式（1-N、N-1等）上取得一致性能提升。

**[Agent Steerable Search for Knowledge Graph Question Answering](graph_learning/agent_steerable_search_for_knowledge_graph_question_answering.md)**

:   本文提出一种基于智能体的可控知识图谱搜索框架，让LLM Agent能够根据问题类型和推理需求动态调整图搜索策略（如搜索深度、方向、剪枝规则），实现对知识图谱问答过程的精细控制。

**[Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning](graph_learning/beyond_completion_a_foundation_model_for_general_knowledge_graph_reasoning.md)**

:   提出 MERRY，一个统一处理 KG 内（零样本 KGC）和 KG 外（KGQA）推理任务的知识图谱基础模型，通过多视角条件消息传递 (CMP) 融合文本和结构信息，在 28 个数据集上超越现有方法。

**[Can Graph Neural Networks Learn Language with Extremely Weak Text Supervision?](graph_learning/can_graph_neural_networks_learn_language.md)**

:   本文提出Morpher，一种多模态提示学习范式，在极弱文本监督（仅几个token的标签名）下，通过同时学习图提示和文本提示将预训练GNN嵌入到LLM的语义空间中，实现跨任务、跨领域的图分类迁移以及首个CLIP风格的GNN零样本分类原型。

**[Croppable Knowledge Graph Embedding](graph_learning/croppable_knowledge_graph_embedding.md)**

:   提出 MED 框架训练"可裁剪"知识图谱嵌入——一次训练同时优化 64 个不同维度的子模型（共享嵌入前缀），通过互学习、进化改进和动态损失权重，各维度子模型直接裁剪使用即超越独立训练和蒸馏方法，训练速度快 10 倍。

**[Cross-Document Contextual Coreference Resolution in Knowledge Graphs](graph_learning/cross-document_contextual_coreference_resolution_in_knowledge_graphs.md)**

:   提出基于知识图谱的跨文档共指消解方法，通过动态链接机制将文本实体提及与知识图谱节点关联，结合上下文嵌入和图消息传递推理提升跨文档实体识别的精度和召回率，在多个基准数据集上超越传统方法。

**[Disentangled Multi-span Evolutionary Network against Temporal Knowledge Graph Reasoning](graph_learning/disentangled_multi-span_evolutionary_network_against_temporal_knowledge_graph_re.md)**

:   提出 DiMNet，通过多跨度演化策略和跨时间解耦机制，分离节点语义的活跃/稳定特征，显著提升时序知识图谱（TKG）外推推理性能，在四个基准数据集上取得 SOTA。

**[Extending Complex Logical Queries on Uncertain Knowledge Graphs](graph_learning/extending_complex_logical_queries_uncertain_knowledge_graphs.md)**

:   本文提出"软查询"形式化框架，将复杂逻辑查询扩展到不确定知识图谱（带置信度值），并设计 SRC 方法结合前向推理和后向校准来高效回答软查询，理论证明误差不会灾难性级联。

**[Fast-and-Frugal Text-Graph Transformers are Effective Link Predictors](graph_learning/fast-and-frugal_text-graph_transformers_are_effective_link_predictors.md)**

:   提出 Fast-and-Frugal Text-Graph (FnF-TG) Transformer，通过 Transformer 的自注意力机制统一编码文本描述和图结构（ego-graph），在归纳链接预测任务上以小 BERT 模型超越了使用大 BERT+MPNN 的 SOTA，同时首次扩展到完全归纳设置（关系也可归纳）。

**[FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering](graph_learning/fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_.md)**

:   提出 FiDeLiS 框架，通过 Path-RAG 预选候选集缩小搜索空间 + 演绎验证beam search (DVBS) 逐步构建并验证推理路径，在无需训练的情况下提升 LLM 在知识图谱问答中的准确性和可解释性。

**[GraphCheck: Breaking Long-Term Text Barriers with Extracted Knowledge Graph-Powered Fact-Checking](graph_learning/graphcheck_breaking_long-term_text_barriers_with_extracted_knowledge_graph-power.md)**

:   GraphCheck 提出一种图增强的事实核查框架，利用 LLM 从文档和声明中提取知识图谱三元组，通过 GNN 编码图结构并作为 soft prompt 注入冻结的 LLM 验证器，在单次推理调用中实现细粒度事实核查，在7个基准上平均提升 7.1%，且在医学领域表现出强泛化能力。

**[GraphNarrator: Generating Textual Explanations for Graph Neural Networks](graph_learning/graphnarrator.md)**

:   GraphNarrator是首个为图神经网络生成自然语言解释的方法，通过显著性图语言化生成伪标签、信息论指标驱动的专家迭代自改进、以及知识蒸馏训练端到端解释器，实现了忠实、简洁且人类友好的GNN决策解释。

**[Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering](graph_learning/kg_llm_trustworthy_qa.md)**

:   提出开放域知识图谱问答基准 OKGQA 及其扰动变体 OKGQA-P，通过统一的图引导检索-生成框架系统性地验证了 KG 增强可以有效降低 LLM 幻觉率（FActScore 提升约 20 个百分点），子图检索在各类查询上表现最优且对 KG 噪声具有鲁棒性。

**[Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation (K-RagRec)](graph_learning/kg_rag_recommendation.md)**

:   提出K-RagRec框架，通过从知识图谱中检索多跳子图为LLM推荐系统提供结构化、可靠的外部知识，结合基于流行度的选择性检索策略和GNN编码器，有效缓解LLM推荐中的幻觉和知识缺失问题。

**[M3HG: Multimodal, Multi-scale, and Multi-type Node Heterogeneous Graph for Emotion Cause Triplet Extraction in Conversations](graph_learning/m3hg_multimodal_multi-scale_and_multi-type_node_heterogeneous_graph_for_emotion_.md)**

:   提出 M3HG 模型，通过构建多模态多类型节点异构图来显式建模对话中的情感与原因上下文，并在句间和句内两个尺度上融合语义信息，实现多模态对话中情感-原因三元组的端到端提取。同时构建了首个中文多场景 MECTEC 数据集 MECAD。

**[mRAKL: Multilingual Retrieval-Augmented Knowledge Graph Construction for Low-Resourced Languages](graph_learning/mrakl_multilingual_retrieval-augmented_knowledge_graph_construction_for_low-reso.md)**

:   将多语言知识图谱构建（mKGC）重新定义为 QA 任务，提出基于 RAG 的 mRAKL 系统，利用非结构化单语数据作为检索源来克服低资源语言中结构化数据匮乏的困难，在 Tigrinya 和 Amharic 两种低资源语言上显著超越已有方法。

**[Multimodal Transformers are Hierarchical Modal-wise Heterogeneous Graphs](graph_learning/multimodal_transformers_are_hierarchical_modal-wise_heterogeneous_graphs.md)**

:   从图论视角证明了多模态 Transformer（MulTs）本质上是层次化模态异质图（HMHG），并基于此提出 GsiT 模型，通过 Interlaced Mask 机制实现仅 1/3 参数的 All-Modal-In-One 融合，同时性能显著超越传统 MulTs。

**[Ontology-Guided Reverse Thinking Makes Large Language Models Stronger on Knowledge Graph Question Answering](graph_learning/ontology-guided_reverse_thinking_makes_large_language_models_stronger_on_knowled.md)**

:   提出 ORT（Ontology-Guided Reverse Thinking），利用知识图谱本体结构从目标逆向构建标签推理路径，引导知识检索，显著提升 LLM 的知识图谱问答能力。

**[Can LLMs Evaluate Complex Attribution in QA? Automatic Benchmarking using Knowledge Graphs](graph_learning/paper_2401_14640.md)**

:   提出 CAQA 基准，利用知识图谱自动生成包含四类归因类别（支持、部分支持、矛盾、无关）与四种推理复杂度的大规模问答归因评估数据集（161K 样本），系统评测 25 种自动归因评估器，揭示"部分支持"识别与复杂推理场景为当前评估器的核心瓶颈。

**[Predicate-Conditional Conformalized Answer Sets for Knowledge Graph Embeddings](graph_learning/predicate-conditional_conformalized_answer_sets_for_knowledge_graph_embeddings.md)**

:   提出 CondKGCP——基于谓词条件的 conformal prediction 方法用于知识图谱嵌入的不确定性量化，通过合并相似谓词增大校准集+双重校准（score+rank）减小预测集大小，在保证谓词级条件覆盖率的同时输出更紧凑的答案集，在多个KGE基准上优于5个baseline。

**[RSCF: Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph](graph_learning/rscf_relationsemantics_consistent_filter_for_entity.md)**

:   提出 RSCF 插件式 KGE 方法，通过共享仿射变换 + 根植实体变换 + 归一化三特征确保"语义相似的关系产生相似的实体变换"（关系语义一致性），在距离模型和张量分解模型上均显著超越 SOTA，并从理论和实验上验证了一致性保持率。

**[SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation](graph_learning/simgrag_leveraging_similar_subgraphs_for_knowledge_graphs_driven_retrieval-augme.md)**

:   提出 SimGRAG 方法，通过"查询→模式图→子图"两阶段对齐策略，利用 LLM 将查询转化为图模式，再用图语义距离（GSD）度量在知识图谱中高效检索语义最相似的子图，实现即插即用的 KG 驱动 RAG，在问答和事实验证任务上超越所有现有方法。

**[The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering](graph_learning/the_role_of_exploration_modules_in_small_language_models_for_knowledge_graph_que.md)**

:   本文系统性地诊断了小语言模型（SLM，0.5B–8B）在 Think-on-Graph 知识图谱问答框架中失效的根因——**探索阶段（exploration）而非推理阶段是性能瓶颈**，并证明用零样本、即插即用的轻量级段落检索模块（SentenceBERT/GTR，仅~110M参数）替代 SLM 进行 KG 遍历，即可在 CWQ 和 WebQSP 两个基准上带来一致且显著的 EM 提升。

---

## 🔬 可解释性 { #interpretability }

**[A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability](interpretability/a_dual-perspective_nlg_meta-evaluation_framework_with_automatic_benchmark_and_be.md)**

:   提出一个双视角 NLG 元评估框架，将传统的人-指标相关性分解为全局视角（序数分类，判断粗粒度质量等级）和局部视角（相邻对比，区分细粒度质量差异），并通过自动化基准构建方法避免人工标注和数据污染，在 16 个 LLM 评估器上实验发现 Qwen-2.5-72B 全局最优、DeepSeek-V3 局部最优。

**[An Empirical Study of Mechanistic Interpretability Approaches for Factual Recall](interpretability/an_empirical_study_of_mechanistic_interpretability_approaches_for_factual_recall.md)**

:   本文系统性地比较了多种机理可解释性方法（因果追踪、激活修补、探针分析等）在定位和解释LLM事实回忆机制方面的表现，揭示了不同方法的一致性、分歧点和各自的适用场景。

**[Around the World in 24 Hours: Probing LLM Knowledge of Time and Place](interpretability/around_the_world_in_24_hours_probing_llm_knowledge_of_time_and_place.md)**

:   本文提出 GeoTemp 数据集（320k 提示，覆盖 289 个城市和 37 个时区），首次评估 LLM 联合时间和空间推理的能力，发现模型能独立处理时间计算和地理知识，但在需要结合两者时性能急剧下降。

**[Bias Attribution in Filipino Language Models: Extending a Bias Interpretability Metric for Application on Agglutinative Languages](interpretability/bias_attribution_in_filipino_language_models_extending_a_bias_interpretability_m.md)**

:   将信息论偏见归因分数指标扩展到黏着语（菲律宾语），通过对子词分数取均值来处理复杂词素结构，在 4 个多语言 PLM 上揭示菲律宾语模型的偏见由实体类主题词（人物/物品/关系）驱动，与英语中动作类主题词（犯罪/性行为）形成鲜明对比。

**[CLEME2.0: Towards Interpretable Evaluation by Disentangling Edits for Grammatical Error Correction](interpretability/cleme2_gec_evaluation.md)**

:   本文提出 CLEME2.0，一种可解释的 GEC 参考评估指标，通过将编辑解耦为四类（正确纠正 TP、错误纠正 FPne、欠纠正 FN、过纠正 FPun）并结合编辑加权技术，在 GJG15 和 SEEDA 两个人工评判数据集上达到了与人工判断最高相关性的 SOTA 结果。

**[Cracking Factual Knowledge: A Comprehensive Analysis of Degenerate Knowledge Neurons in Large Language Models](interpretability/degenerate_knowledge_neurons.md)**

:   本文从结构和功能双重角度重新定义了LLM中的退化知识神经元（DKN），提出神经拓扑聚类方法获取任意数量和结构的DKN，并通过34个实验揭示了DKN与LLM鲁棒性、可进化性和复杂性的内在关联。

**[EXPERT: An Explainable Image Captioning Evaluation Metric with Structured Explanations](interpretability/expert_an_explainable_image_captioning_evaluation_metric_with_structured_explana.md)**

:   本文提出 EXPERT，一种基于 VLM 微调的无参考图像描述评估指标，通过构建大规模结构化解释数据集并设计两阶段评估模板，在多个基准数据集上达到 SOTA 的同时，提供基于流畅度、相关性、描述性三个维度的高质量结构化解释。

**[IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory](interpretability/irt_router_multi_llm.md)**

:   IRT-Router 借鉴心理测量学的项目反应理论（IRT），将 LLM 视为"考生"、query 视为"考题"，学习多维能力向量和难度/区分度参数实现可解释的多 LLM 路由，在 OOD 场景下达 87%+ 准确率且成本仅为 GPT-4o 的 1/30。

**[Llama See, Llama Do: A Mechanistic Perspective on Contextual Entrainment and Distraction in LLMs](interpretability/llama_see_llama_do_entrainment.md)**

:   本文发现并定义了"上下文夹带"(contextual entrainment)现象——LLM会对上下文中出现过的任意token赋予更高概率，并通过可微掩码方法定位了负责该现象的entrainment heads，关闭这些头后可显著抑制干扰效应。

**[Mechanistic Interpretability of Emotion Inference in Large Language Models](interpretability/mechanistic_interpretability_of_emotion_inference_in_large_language_models.md)**

:   通过 probing、activation patching 和 generation steering 三种机制可解释性技术，发现 LLM 的情感表征功能性地定位于中间层的 MHSA 单元，并基于认知评估理论（appraisal theory）证明这些表征具有心理学合理性，成功通过干预评估概念（如 self-agency、pleasantness）引导情感输出。

**[Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability](interpretability/normalized_aopc_faithfulness_metrics.md)**

:   本文揭示了广泛使用的 AOPC（扰动曲线下面积）忠实度指标在跨模型比较时会产生误导性结论（因为不同模型的 AOPC 上下界差异巨大），提出 Normalized AOPC (NAOPC) 通过 min-max 归一化消除模型间的不可比性，实验表明归一化可以根本性地改变模型忠实度排名。

**[Enhancing Automated Interpretability with Output-Centric Feature Descriptions](interpretability/output_centric_interpretability.md)**

:   提出基于输出的特征描述方法（VocabProj和TokenChange），弥补了现有自动化可解释性管线仅依赖输入激活样本的局限，结合输入-输出双视角的集成方法在两类评估中均取得最优表现。

**[Position-aware Automatic Circuit Discovery](interpretability/position-aware_automatic_circuit_discovery.md)**

:   提出位置感知的边归因修补方法（PEAP）和数据集 Schema 机制，解决了自动电路发现中忽略位置信息导致的抵消效应和重要性高估问题，实现了更小且更忠实的电路发现。

**[Probing Subphonemes in Morphology Models](interpretability/probing_subphonemes_in_morphology_models.md)**

:   本文提出了一种语言无关的探测方法，研究在形态学变形任务上训练的 Transformer 模型如何隐式习得音韵特征，发现局部特征（如末辅音清化）在音素嵌入中编码良好，而长距离依赖（如元音和谐）在编码器层的上下文化表示中更显著。

**[Probing the Geometry of Truth: Consistency and Generalization of Truth Directions](interpretability/probing_the_geometry_of_truth_consistency_and_generalization_of_truth_directions.md)**

:   系统性研究LLM内部"真值方向"(truth direction)的一致性与泛化能力，发现只有能力较强的模型才稳定展现一致的真值方向，且基于简单原子陈述训练的真实性探针可泛化至逻辑变换、问答任务和上下文知识场景。

**[Reasoning Circuits in Language Models: A Mechanistic Interpretation of Syllogistic Inference](interpretability/reasoning_circuits_in_language_models_a_mechanistic_interpretation_of_syllogisti.md)**

:   用机械可解释性技术（激活补丁 + Logit Lens + 电路消融）发现语言模型中实现三段论推理的完整电路：三阶段机制——长归纳偏差→中间项抑制（h11.10）→传递项移动，该电路在符号输入上既充分又必要，可迁移到自然语言输入，且跨 GPT-2/Pythia/LLaMA/Qwen 四种架构存在兼容模式。

**[Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety](interpretability/safety_is_not_only_about_refusal_reasoning-enhanced_fine-tuning_for_interpretabl.md)**

:   提出 Rational 框架，通过推理增强微调让 LLM 在回答前进行显式的安全推理（分析意图、伦理和潜在危害），而非依赖僵硬的拒绝启发式，在保持有用性的同时显著提升对推理层面对抗攻击的鲁棒性。

**[Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers](interpretability/separating_tongue_from_thought_activation_patching_reveals_language-agnostic_con.md)**

:   通过激活修补实验，首次提供了因果性证据证明大语言模型内部存在与语言解耦的概念表示——模型先确定输出语言，再确定概念，并且跨语言平均的概念表示不仅不损害翻译能力，反而能提升翻译准确率。

**[Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis](interpretability/shortcut_neuron_eval.md)**

:   提出通过对比分析和因果分析定位污染模型中的"捷径神经元"（shortcut neurons），并通过 activation patching 抑制这些神经元，实现更可信的 LLM 评估，与 MixEval 的 Spearman 相关系数超过 0.95。

**[The Anatomy of Evidence: An Investigation Into Explainable ICD Coding](interpretability/the_anatomy_of_evidence_an_investigation_into_explainable_icd_coding.md)**

:   本文对 MDACE 数据集和当前可解释 ICD 编码系统进行了深入的应用导向分析，揭示了人工标注证据与代码描述的重叠规律、证据在文档中的分布特征，并提出了新的匹配度量来评估模型解释的实用性。

**[Towards Explainable Temporal Reasoning in Large Language Models: A Structure-Aware Generative Framework](interpretability/towards_explainable_temporal_reasoning_in_large_language_models_a_structure-awar.md)**

:   提出 GETER 框架，通过轻量级 Structure-Text Adapter 将时序知识图谱的结构信息注入 LLM，使模型在时序推理任务中既能给出准确预测又能生成可解释的推理说明。

---

## 🏥 医学图像 { #medical_imaging }

**[A Modular Approach for Clinical SLMs Driven by Synthetic Data with Pre-Instruction Tuning, Model Merging, and Clinical-Tasks Alignment](medical_imaging/a_modular_approach_for_clinical_slms_driven_by_synthetic_data_with_pre-instructi.md)**

:   本文提出一种将小型语言模型（SLM）高效适配为临床领域模型的模块化框架，包含领域专家预指令微调（在医学语料上训练多个专家模型）、模型合并（将多个专家合并为统一的 MediPhi）、以及基于 250 万条合成指令（MediFlow）的临床任务对齐，最终 3.8B 参数的 MediPhi 在多项临床任务上超越 GPT-4。

**[A Retrieval-Based Approach to Medical Procedure Matching in Romanian](medical_imaging/a_retrieval-based_approach_to_medical_procedure_matching_in_romanian.md)**

:   将罗马尼亚语医疗程序名称匹配建模为检索问题而非分类问题，在 39,097 个标准条目（50% 仅有单样本）的极端长尾场景下，对比 BM25 稀疏检索与 mE5/RoBERT/BioClinicalBERT 三种密集嵌入，通过度量学习微调后 mE5 达到 85.2% Acc@1，真实部署中医生验证 94.7% 准确率且比人工快 1200 倍。

**[Automated Structured Radiology Report Generation](medical_imaging/automated_structured_radiology_report_generation.md)**

:   提出结构化放射学报告生成（SRRG）新任务，利用LLM将自由文本报告重构为标准化格式，同时引入55标签的SRR-BERT疾病分类模型和F1-SRR-BERT评估指标，解决传统报告生成中风格多样导致的生成与评估困难。

**[The Impact of Auxiliary Patient Data on Automated Chest X-Ray Report Generation and How to Incorporate It](medical_imaging/auxiliary_patient_data_xray.md)**

:   本文研究如何将急诊科患者数据（生命体征、药物、分诊信息等）整合到多模态语言模型中用于自动胸部X光报告生成，提出将异构表格数据、文本和图像转化为统一嵌入的方法，在MIMIC-CXR + MIMIC-IV-ED数据集上显著提升了报告的诊断准确性，超越了包括CXRMate-RRG24在内的多个基准模型。

**[CheXalign: Preference Fine-tuning in Chest X-ray Interpretation Models without Human Feedback](medical_imaging/chexalign_preference_finetuning.md)**

:   CheXalign 提出了一种无需放射科医生反馈的自动化偏好数据生成管线，利用公开数据集中的参考报告和基于参考的评估指标（如 GREEN、BERTScore）构造偏好对，通过 DPO 等直接对齐算法对胸部X光报告生成模型进行偏好微调，在 MIMIC-CXR 上取得 SOTA CheXbert 分数。

**[Aligning AI Research with the Needs of Clinical Coding Workflows: Eight Recommendations Based on US Data Analysis and Critical Review](medical_imaging/clinical_coding_eight_recommendations.md)**

:   这篇 position paper 通过对 MIMIC 数据集和现有自动化临床编码研究的深入分析，指出当前评估方法（如仅关注前50个高频编码、使用不恰当指标）与真实临床场景严重脱节，并提出八条具体建议来改进评估方法和研究方向。

**[CSTRL: Context-Driven Sequential Transfer Learning for Abstractive Radiology Report Summarization](medical_imaging/cstrl_context-driven_sequential_transfer_learning_for_abstractive_radiology_repo.md)**

:   提出 CSTRL，一种基于顺序迁移学习的放射学报告摘要生成方法，通过优化的间隔句生成（GSG）预训练、Fisher 矩阵正则化防止灾难性遗忘，并结合知识蒸馏实现模型压缩，在 MIMIC-CXR 和 Open-I 数据集上大幅超越现有方法。

**[KokoroChat: A Japanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors](medical_imaging/kokorochat_a_japanese_psychological_counseling_dialogue.md)**

:   提出 KokoroChat，一个通过训练有素的咨询师角色扮演收集的日语心理咨询对话数据集，包含 6,589 段长对话及详细的客户反馈评分，用于提升 LLM 的心理咨询回复生成和对话评估能力。

**[ANGEL: Learning from Negative Samples in Biomedical Generative Entity Linking](medical_imaging/learning_from_negative_samples_in_biomedical_generative_entity_linking.md)**

:   提出 ANGEL 框架，首次在生成式生物医学实体链接（BioEL）中引入负样本训练，通过两阶段策略（正样本训练 + 负样本感知的偏好优化）显著提升模型区分表面形式相似但语义不同的实体的能力，在五个基准数据集上平均 top-1 准确率提升 1.7%。

**[MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA](medical_imaging/medbiorag_semantic_search_and_retrieval-augmented_generation_for_biomedical_lite.md)**

:   MedBioRAG 提出了一种结合语义搜索、文档检索和微调 LLM 的检索增强生成框架，在生物医学问答的文本检索、封闭式 QA 和长文本 QA 三类任务上全面超越 GPT-4o 基线和此前 SOTA。

**[MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA](medical_imaging/medbiorag_semantic_search_and_retrieval-augmented_generation_with_large_language.md)**

:   MedBioRAG 提出了一个集成语义搜索、文档检索和微调LLM的检索增强生成框架，用于生物医学问答任务，在文本检索（NFCorpus、TREC-COVID）、封闭式问答（MedQA、PubMedQA、BioASQ）和长文本问答四个维度的多个基准上均超越了先前SOTA和GPT-4o基线模型。

**[Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking](medical_imaging/mitigating_confounding_in_speech-based_dementia_detection_through_weight_masking.md)**

:   针对基于语音转录文本的痴呆检测任务中的性别混淆偏差问题，提出 Extended Confounding Filter（ECF）和 Dual Filter（DF）两种无需额外训练模块的权重掩码方法，通过追踪微调过程中的权重变化来定位性别关联参数并将其置零，在多种分布偏移场景下保持痴呆检测性能的同时显著降低性别间的假阳性率差异和统计均等性差距。

**[MultiMed: Multilingual Medical Speech Recognition via Attention Encoder Decoder](medical_imaging/multimed_multilingual_medical_speech_recognition_via_attention_encoder_decoder.md)**

:   发布 MultiMed——首个多语言医学 ASR 数据集（150小时，5种语言，10种录制场景，16种口音），配套小到大规模的端到端 Whisper 模型基线，首次系统研究医学领域的多语言 ASR：单语 vs 多语微调、AED vs Hybrid 架构对比，发现多语联合训练在小模型上有收益但大模型上可能退化。

**[Online Iterative Self-Alignment for Radiology Report Generation](medical_imaging/oisa_radiology_report_gen.md)**

:   提出在线迭代自对齐（OISA）方法：通过自生成→自评估→自对齐→自迭代的四阶段循环，利用多目标偏好优化（MODPO）让轻量级 RRG 模型在无需外部大模型或人工标注的条件下，持续提升放射学报告质量，在 MIMIC-CXR 和 IU-Xray 上达到 SOTA。

**[Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications](medical_imaging/omni_rag_medical.md)**

:   本文提出了 MedOmniKB 医学多源知识库和 Source Planning Optimisation (SPO) 方法，通过让专家模型探索多源检索计划并训练小模型学习源对齐，显著提升了医学多源检索规划能力，使 7B 小模型超越 72B 大模型。

**[One Size Fits None: Rethinking Fairness in Medical AI](medical_imaging/one_size_fits_none_rethinking_fairness_in_medical_ai.md)**

:   本文在三个多模态医学预测任务（ICU死亡率、移植物失败、急诊分诊）上进行子群体性能分析，揭示聚合指标掩盖的群体间性能差异，主张将公平性与透明度紧密结合，通过常规化的子群体报告推动负责任的医学AI部署。

**[Pattern Recognition or Medical Knowledge? The Problem with Multiple-Choice Questions in Medicine](medical_imaging/pattern_recognition_or_medical_knowledge_the_problem_with_multiple-choice_questi.md)**

:   本文通过构建围绕虚构器官"Glianorex"的医学选择题基准，揭示LLM在医学MCQ测试中主要依赖模式识别和答题策略而非真正的临床推理能力——模型在完全虚构的医学知识上平均得分64%，而医生仅得27%。

**[Radar: Enhancing Radiology Report Generation with Supplementary Knowledge Injection](medical_imaging/radar_radiology_report_gen.md)**

:   提出 Radar 框架，通过区分 LLM 已掌握的可信内部知识和需要外部补充的知识，系统性地融合两种知识源以生成更准确的放射学报告。

**[RedactX: An LLM-Powered Framework for Automatic Clinical Data De-Identification](medical_imaging/redactor_an_llm-powered_framework_for_automatic_clinical_data_de-identification.md)**

:   提出 RedactX——一个全自动、多模态的临床数据去标识化框架，结合 LLM 多轮抽取、规则处理和检索式再词汇化，在 i2b2 数据集上实现了与专用商业系统可比的 F1（0.9646），同时优化了 token 使用效率。

**[SECRET: Semi-supervised Clinical Trial Document Similarity Search](medical_imaging/secret_semi-supervised_clinical_trial_document_similarity_search.md)**

:   提出 SECRET，一种半监督临床试验协议相似性搜索方法，通过将临床试验文档转换为 Q/A 对表示，并结合局部（Q/A 级）和全局（试验级）对比学习来生成嵌入，在完整试验搜索的 recall@1 上相对最佳基线提升 78%。

---

## ✏️ 知识编辑 { #knowledge_editing }

**[A General Knowledge Injection Framework for ICD Coding](knowledge_editing/a_general_knowledge_injection_framework_for_icd_coding.md)**

:   > 本文提出 GKI-ICD，一个通用的知识注入框架，通过指南合成和多任务学习机制，无需额外网络模块即可同时整合 ICD Description、Synonym 和 Hierarchy 三种知识，在 MIMIC-III 基准上取得 SOTA 性能。

**[ToxEdit: Adaptive Detoxification Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing](knowledge_editing/adaptive_detoxification_safeguarding_general_capabilities_of_llms_through_toxici.md)**

:   提出 ToxEdit——毒性感知的知识编辑方法，在 LLM 前向传播早期层用 SVM 分类器检测有害隐藏状态，通过路由机制将有害输入导向编辑后的 FFN 副本、无害输入走原始 FFN，在 LLaMA3-8B/LLaMA2-7B/Mistral-7B 上同时实现了近 98% 去毒成功率和 95% 指令遵从保留（DL 指标），解决了知识编辑去毒中"去毒 vs 过度编辑"的核心矛盾。

**[BMIKE-53: Investigating Cross-Lingual Knowledge Editing with In-Context Learning](knowledge_editing/bmike-53_investigating_cross-lingual_knowledge_editing_with_in-context_learning.md)**

:   提出 BMIKE-53 —— 覆盖 53 种语言、整合 zsRE/CounterFact/WikiFactDiff 三个知识编辑数据集的跨语言基准，系统评估 zero-shot 到 8-shot 的上下文知识编辑方法，发现文字系统（拉丁 vs 非拉丁）比语言家族更能决定跨语言编辑效果，且 metric-specific 示例策略显著优于混合示例。

**[ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains](knowledge_editing/chainedit_propagating_ripple_effects_in_llm.md)**

:   提出 ChainEdit 框架，通过将知识图谱中挖掘的逻辑规则与 LLM 内在逻辑推理能力对齐，实现知识编辑时的链式更新，将逻辑泛化准确率从约 20% 提升至 58-65%。

**[CKnowEdit: A New Chinese Knowledge Editing Dataset for Linguistics, Facts, and Logic Error Correction in LLMs](knowledge_editing/cknowedit_chinese_knowledge_editing_dataset_llms.md)**

:   构建首个面向中文语言特性的知识编辑数据集 CKnowEdit，涵盖语言学（拼音/古诗/文言/成语/谚语）、事实（历史地理）和逻辑陷阱（谐音/推理/文字游戏）三大类共 1,854 条样本，系统评估五种主流知识编辑方法在四个中文 LLM 上的表现，揭示中文独有的编辑难题。

**[CompKe: Complex Question Answering under Knowledge Editing](knowledge_editing/compke_complex_question_answering_under_knowledge_editing.md)**

:   提出CompKe基准——包含11,924个复杂问题——用于评估知识编辑方法在涉及**一对多关系、逻辑操作（交集/并集）和条件确认**的复杂推理场景下的表现，揭示现有方法在复杂问答上的显著不足。

**[Context-Robust Knowledge Editing for Language Models](knowledge_editing/context-robust_knowledge_editing_for_language_models.md)**

:   发现现有知识编辑方法在前缀上下文存在时大幅失败（编辑成功率从 90.9% 降至 69.1%），提出 CHED 基准评估上下文鲁棒性，并设计 CoRE 方法通过多样化前缀上下文 + 跨前缀隐藏状态方差正则化来增强编辑的上下文鲁棒性，在保持模型通用能力的同时显著缩小有/无上下文的性能差距。

**[DocMEdit: Towards Document-Level Model Editing](knowledge_editing/docmedit_towards_document-level_model_editing.md)**

:   首次提出文档级模型编辑任务，构建包含 37,990 条数据、105,652 个编辑事实的 DocMEdit 基准，揭示现有编辑方法在长上下文、多事实并行编辑场景下的严重不足。

**[Efficient Knowledge Editing via Minimal Precomputation](knowledge_editing/efficient_knowledge_editing.md)**

:   证明了 MEMIT/ROME/EMMET 等知识编辑方法的预计算步骤（缓存 4400 万隐向量）可以减少到理论最小值的 2-10 倍（不到原来的 0.3%），将预计算时间从数十小时降到几分钟，且编辑性能基本无损。

**[Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning](knowledge_editing/memorizing_is_not_enough_deep_knowledge_injection_through_reasoning.md)**

:   提出四层知识注入框架（记忆→检索→推理→关联），构建 DeepKnowledge 合成测试平台，系统性揭示了知识注入各层级的关键因素：重复学习实现记忆、表达多样性实现检索、显式推理模式实现深度推理和关联，为 LLM 知识更新提供了完整的方法-层级映射。

**[Mitigating Negative Interference in Multilingual Sequential Knowledge Editing through Null-Space Constraints](knowledge_editing/mitigating_negative_interference_in_multilingual_sequential_knowledge_editing_th.md)**

:   本文提出 LangEdit 框架，通过将每种语言的参数更新投影到先前编辑语言的零空间上，实现多语言序列知识编辑中不同语言更新之间的数学隔离，有效缓解负干扰并保持多语言泛化能力。

**[Neuron-Level Sequential Editing for Large Language Models](knowledge_editing/neuron-level_sequential_editing_for_large_language_models.md)**

:   提出NSE方法用于LLM的序列化模型编辑，通过权重回退（weights rewinding）防止模型崩溃、基于激活值的神经元级选择性权重更新缓解模型遗忘、以及迭代多层编辑提高大规模知识更新的成功率。

**[REP: Keys to Robust Edits — From Theoretical Insights to Practical Advances](knowledge_editing/rep_robust_knowledge_editing.md)**

:   揭示locate-and-edit知识编辑方法中语义键的根本缺陷——内部表示无法同时满足鲁棒性和特异性，提出REP模块通过对比学习解耦编辑键，在鲁棒性测试上提升最高66.4%。

**[Revealing the Deceptiveness of Knowledge Editing: A Mechanistic Analysis of Superficial Editing](knowledge_editing/revealing_the_deceptiveness_of_knowledge_editing_a_mechanistic_analysis_of_super.md)**

:   本文定义了"表面编辑"（superficial editing）现象——经过知识编辑的模型在常规提示下表现良好，但在特制攻击探针下会回退到原始知识——并通过机制分析揭示了早期层残差流和后期层特定注意力头是导致该现象的两个关键因素。

**[SAKE: Steering Activations for Knowledge Editing](knowledge_editing/sake_steering_activations_for_knowledge_editing.md)**

:   SAKE 提出将知识编辑建模为激活空间中的分布映射问题，通过对编辑事实生成改述和逻辑蕴含的提示集合来构建源/目标激活分布，再用最优传输的线性映射替换激活向量，实现比 ROME/MEMIT 等方法更鲁棒的事实编辑，在逻辑蕴含泛化和上下文鲁棒性上显著领先。

**[ScEdit: Script-based Assessment of Knowledge Editing](knowledge_editing/scedit_script-based_assessment_of_knowledge_editing.md)**

:   提出 ScEdit，一个基于脚本（Script）的知识编辑评估基准，将传统的"What"类事实回忆评估扩展为"How"类程序性推理评估，同时引入 token 级和文本级双层评估体系，揭示了现有知识编辑方法在实际应用场景中的显著不足。

**[Structure-aware Domain Knowledge Injection for Large Language Models](knowledge_editing/structure-aware_domain_knowledge_injection_for_large_language_models.md)**

:   提出 StructTuning 方法，通过自动提取训练语料的知识分类结构，设计结构感知的持续预训练（SCPT）和结构感知的监督微调（SSFT）两阶段策略，仅用传统方法5%的数据量即可达到100%的领域知识注入效果。

**[The Mirage of Model Editing: Revisiting Evaluation in the Wild](knowledge_editing/the_mirage_of_model_editing_revisiting_evaluation_in_the_wild.md)**

:   本文揭示了模型编辑领域评估实践中的系统性缺陷——先前方法报告的近完美成功率（~96.8%）在真实应用场景下骤降至 38.5%，根本原因是测试中使用 teacher forcing 泄露了真值信息，并提出 QAEdit 基准和 WILD 评估框架来推动更可靠的评估。

**[Towards a Principled Evaluation of Knowledge Editors](knowledge_editing/towards_a_principled_evaluation_of_knowledge_editors.md)**

:   本文系统性地揭示了知识编辑评估中评估方法、评估指标和编辑批量大小的选择会显著影响编辑器排名，并通过与LM Evaluation Harness集成来评估编辑对模型整体能力的副作用。

---

## 🗣️ 对话系统 { #dialogue }

**[DEMO: Reframing Dialogue Interaction with Fine-grained Element Modeling](dialogue/demo_reframing_dialogue_interaction_with_fine-grained_element_modeling.md)**

:   本文提出对话元素建模（Dialogue Element Modeling）这一新任务，系统定义了对话生命周期中从"前奏"到"尾声"的全面元素体系，构建了包含元素感知和对话智能体交互两大能力的DEMO benchmark，并通过模仿学习训练DEMO agent在域内外任务上均表现优异。

**[Dialogue Systems for Emotional Support via Value Reinforcement](dialogue/dialogue_systems_for_emotional_support_via_value_reinforcement.md)**

:   提出 ES-VR，首个将人类价值观强化融入情感支持对话系统的方法，通过目标价值检测器和参考生成器（均在 Reddit 数据上训练），结合 SFT + DPO 两阶段训练，使支持者模型不仅能缓解求助者的负面情绪，还能探索和强化其积极价值观，实现更深层的内在转变。

**[Dynamic Label Name Refinement for Few-Shot Dialogue Intent Classification](dialogue/dynamic_label_name_refinement_for_few-shot_dialogue_intent_classification.md)**

:   提出动态标签名称精炼方法，在检索式 ICL 意图分类中，利用 LLM 根据检索到的示例动态生成更具区分性的意图标签名称（如 "Verify PAN" → "Verify PAN card details"），有效降低语义相似意图间的混淆，在 6 个数据集上一致提升 2.07%-7.51% 准确率。

**[Enabling Chatbots with Eyes and Ears: An Immersive Multimodal Conversation System](dialogue/enabling_chatbots_with_eyes_and_ears_an_immersive_multimodal_conversation_system.md)**

:   本文提出赋予聊天机器人"眼睛和耳朵"的沉浸式多模态对话系统，构建了融合视觉与听觉的多会话多方对话数据集 M3C，并设计了包含对话模块和多模态记忆检索模块的对话模型，实现了多说话者共享视听体验的动态长期对话。

**[Enhancing Goal-oriented Proactive Dialogue Systems via Consistency Reflection and Correction](dialogue/enhancing_goal-oriented_proactive_dialogue_systems_via_consistency_reflection_an.md)**

:   提出模型无关的两阶段 CRC 框架（一致性反思 + 一致性纠正），通过先让模型反思生成回复与对话上下文之间的不一致之处、再据此纠正回复，显著提升了目标导向主动对话系统中生成回复与对话上下文的一致性。

**[EnSToM: Enhancing Dialogue Systems with Entropy-Scaled Steering Vectors for Topic Maintenance](dialogue/enstom_enhancing_dialogue_systems_with_entropy-scaled_steering_vectors_for_topic.md)**

:   提出 EnSToM，一种基于熵缩放转向向量的轻量级方法，通过利用 LLM 内部层级熵分布差异来动态调整转向强度，在不修改模型参数的情况下提升任务导向对话系统的主题维持能力。

**[Know You First and Be You Better: Modeling Human-Like User Simulators via Implicit Profiles](dialogue/know_you_first_and_be_you_better_modeling_human-like_user_simulators_via_implici.md)**

:   本文提出 USP（User Simulator with Implicit Profiles）框架，通过从人机对话中提取隐式用户画像，并结合条件监督微调和基于循环一致性的强化学习，在真实性、一致性和多样性三个维度上显著超越基线方法，语义相似度和风格相似度分别提升约 34% 和 43%。

**[Know Your Mistakes: Towards Preventing Overreliance on Task-Oriented Conversational AI Through Accountability Modeling](dialogue/know_your_mistakes_towards_preventing_overreliance_on_task-oriented_conversation.md)**

:   本文提出面向任务型对话系统的 Accountability Model，在 LLM 中加入额外的 accountability head 作为二分类器预测对话状态中各 slot 的概率，从而检测并自校正假阳性和假阴性错误，在 MultiWOZ 上将 JGA 从 64.34 提升到 70.51（↑9.6%），达到 SOTA。

**[Exploring Persona Sentiment Sensitivity in Personalized Dialogue Generation](dialogue/persona_sentiment_dialogue.md)**

:   大规模分析发现 LLM 生成的个性化对话质量对人物画像的情感极性高度敏感——负面画像导致过度强调人设引发矛盾，正面画像则选择性融入人设产生更高质量对话——基于此提出结合轮次生成、画像排序和情感感知提示的改进方法。

**[PersonaLens: A Benchmark for Personalization Evaluation in Conversational AI Assistants](dialogue/personalens_a_benchmark_for_personalization_evaluation_in_conversational_ai_assi.md)**

:   提出 PersonaLens，一个面向任务导向型 AI 助手个性化能力的综合评测基准，包含 1500 个丰富用户画像、20 个领域 111 个任务、用户模拟 Agent 和 Judge Agent，通过大规模自动化评估揭示当前 LLM 助手在个性化方面的显著不足。

**[ReflectDiffu: Reflect between Emotion-intent Contagion and Mimicry for Empathetic Response Generation via a RL-Diffusion Framework](dialogue/reflectdiffu_empathetic_response.md)**

:   提出轻量级共情对话框架 ReflectDiffu，融合情感传染（捕捉情绪）、意图二次机制（Exploring-Sampling-Correcting将情绪映射为行动意图）和扩散模型生成，在相关性、可控性和信息量上全面超越现有基线和 Llama-3.1-8B。

**[SHARE: Shared Memory-Aware Open-Domain Long-Term Dialogue Dataset Constructed from Movie Script](dialogue/share_shared_memory-aware_open-domain_long-term_dialogue_dataset_constructed_fro.md)**

:   提出了基于电影剧本构建的长期对话数据集 SHARE，首次引入「共享记忆」概念，并设计了 EPISODE 对话框架来管理个人信息、个人事件和共享记忆，使长期对话更具亲密感和参与度。

**[Single- vs. Dual-Prompt Dialogue Generation with LLMs for Job Interviews in Human Resources](dialogue/single-_vs_dual-prompt_dialogue_generation_with_llms_for_job_interviews_in_human.md)**

:   本文系统比较了使用 LLM 生成求职面试对话的两种策略——单提示（一次性生成完整对话）和双提示（两个 agent 分别扮演面试官和候选人轮流对话），发现双提示方法生成的对话在自然度上的胜率是单提示的 2-10 倍，但 token 成本增加约 6 倍。

**[Sparse Rewards Can Self-Train Dialogue Agents](dialogue/sparse_rewards_can_self-train_dialogue_agents.md)**

:   提出 JOSH（Juxtaposed Outcomes for Simulation Harvesting）自对齐算法，让 LLM 对话 Agent 通过稀疏奖励的模拟环境自主提升性能，无需外部人类反馈，并构建了 ToolWOZ 稀疏奖励工具调用模拟环境加以验证。

**[UniConv: Unifying Retrieval and Response Generation for Large Language Models in Conversations](dialogue/uniconv_retrieval_response_gen.md)**

:   探索如何将对话场景中的稠密检索和响应生成统一到单个 LLM 中，通过三个联合训练目标（对话检索 + 响应生成 + 上下文识别指令）和数据差异缓解机制，在五个对话搜索数据集上实现检索和生成的相互促进，超越分离式基线。

**[When Harry Meets Superman: The Role of The Interlocutor in Persona-Based Dialogue Generation](dialogue/when_harry_meets_superman_the_role_of_the_interlocutor_in_persona-based_dialogue.md)**

:   系统性地研究了人设对话生成中**对话者（interlocutor）信息**对目标说话人生成质量的影响，通过遮蔽/揭示对话者信息的评估框架发现：模型能有效适应对话者人设、对陌生对话者泛化能力弱于陌生话题，且零样本设置下LLM倾向于"复制粘贴"人设细节。

**[Wizard of Shopping: Target-Oriented E-commerce Dialogue Generation with Decision Tree Branching](dialogue/wizard_of_shopping_target-oriented_e-commerce_dialogue_generation_with_decision_.md)**

:   本文提出 TRACER 方法，利用决策树模型规划对话路径，引导两个 LLM Agent（顾客和卖家）生成自然且有目标导向的电商购物对话，并发布了包含 3600 条对话的 Wizard of Shopping (WoS) 数据集，在对话查询生成和商品排序两个下游任务上验证了数据集的有效性。

---

## 🔎 AIGC 检测 { #aigc_detection }

**[A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI](aigc_detection/a_rose_by_any_other_name_llm-generated_explanations_are_good_proxies_for_human_e.md)**

:   提出用 LLM 生成的 NLI 解释替代昂贵的人工解释来近似人工判断分布（HJD），实验表明在提供人工标签引导的条件下，LLM 生成的解释与人工解释在 KL 散度、JSD 等指标上效果相当，并可推广到无人工解释的数据集（MNLI）和域外测试集（ANLI）。

**[Are We in the AI-Generated Text World Already? Quantifying and Monitoring AIGT on Social Media](aigc_detection/aigt_social_media_monitoring.md)**

:   首次大规模量化社交媒体上 AI 生成文本(AIGT)的占比变化——收集 Medium/Quora/Reddit 上 240 万帖子，构建 AIGTBench 训练最佳检测器 OSM-Det，发现 2022-2024 年间 Medium 和 Quora 的 AIGT 占比从~2% 飙升至~37-39%，而 Reddit 仅从 1.3% 增至 2.5%。

**[An Empirical Study on Detecting AI-Generated Text in Financial Reports](aigc_detection/an_empirical_study_on_detecting_ai-generated_text_in_financial_reports.md)**

:   本文针对金融报告这一高监管领域，系统评估了多种AI生成文本检测方法（统计特征、神经网络分类器、水印检测等）在识别金融文档中AI生成内容方面的表现，揭示了领域特异性对检测效果的显著影响。

**[People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text](aigc_detection/chatgpt_user_ai_text_detection.md)**

:   通过 1,740 条标注实验发现，经常使用 LLM 进行写作任务的人类标注者可以极高精度（5人投票仅错 1/300）检测 AI 生成文本，即使面对改写和人性化逃逸策略也显著优于大多数自动检测器。

**[ChemActor: Enhancing Automated Extraction of Chemical Synthesis Actions with LLM-Generated Data](aigc_detection/chemactor_enhancing_automated_extraction_of_chemical_synthesis_actions_with_llm-.md)**

:   本文提出 ChemActor，一个经过完全微调的 LLM 化学执行器，通过序列化 LLM 生成数据框架和分布散度数据筛选模块来解决化学合成动作提取中的数据稀缺问题，在 R2D 和 D2A 任务上超越基线模型 10%。

**[Cognitive Framework for Detecting AI-Generated Fiction](aigc_detection/cognitive_framework_for_detecting_ai-generated_fiction.md)**

:   本文提出一种基于认知语言学特征的AI生成小说/虚构文本检测框架，通过建模人类创意写作中的认知模式（如叙事节奏、情感弧线、隐喻密度）来区分人类和AI创作的虚构文本，在长文本场景下显著优于现有检测方法。

**[Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training](aigc_detection/greater_adversarial_mgt_detection.md)**

:   提出 GREATER 对抗训练框架，同步训练对抗攻击器（Greater-A）和 MGT 检测器（Greater-D），对抗器通过代理模型梯度识别关键 token 并在嵌入空间扰动生成对抗样本，检测器从课程式对抗样本中学习泛化防御，在 16 种攻击下 ASR 降至 5.53%（SOTA 为 6.20%），攻击效率比 SOTA 快 4 倍。

**[HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring](aigc_detection/haco-det_a_study_towards_fine-grained_machine-generated_text_detection_under_hum.md)**

:   提出面向人机协作写作场景的细粒度机器生成文本（MGT）检测基准 HACo-Det，通过多轮局部改写流水线自动构建带词级归属标注的 11,200 篇人机共创文本，并将七种主流检测器改造为词级序列标注模式进行系统评估，揭示当前方法在细粒度检测上的巨大改进空间。

**[KatFishNet: Detecting LLM-Generated Korean Text through Linguistic Feature Analysis](aigc_detection/katfishnet_detecting_llm-generated_korean_text_through_linguistic_feature_analys.md)**

:   本文构建了首个韩语 LLM 生成文本检测基准 KatFish（涵盖三种文体、四种 LLM），通过分析词间距、词性多样性和逗号使用三类韩语语言学特征，提出 KatFishNet 检测方法，在 OOD（未见过的 LLM）设置下平均 AUROC 比最佳基线高 19.78%。

**[Learning to Rewrite: Generalized LLM-Generated Text Detection](aigc_detection/learning_to_rewrite_generalized_llm-generated_text_detection.md)**

:   提出Learning2Rewrite（L2R）框架，通过微调LLM的改写模型来放大人写文本和AI生成文本在改写编辑距离上的差异，从而实现跨领域高度泛化的AI文本检测——在21个独立领域上平均AUROC达0.9009，域外测试超越RAIDAR达4.67%、超越直接分类微调达51.35%。

**[Comparing LLM-generated and human-authored news text using formal syntactic theory](aigc_detection/llm_vs_human_formal_syntax.md)**

:   首次使用 **HPSG 形式句法理论**（通过英语资源语法 ERG）从句法构式（298 种）、词汇类型（1398 种）和词法规则（100 种）三个层级系统比较 6 个 LLM 与人类 NYT 新闻写作的语法差异，发现 LLM 在语法特征上是人类作者的 **"均值化"投影**——人类个体作者间的语法差异反而大于任何人类与 LLM 的差异，而 LLM 之间几乎无差别。

**[Low-Perplexity LLM-Generated Sequences and Where To Find Them](aigc_detection/low-perplexity_llm-generated_sequences_and_where_to_find_them.md)**

:   提出系统化 pipeline 分析 LLM 生成的低困惑度序列（token 预测概率 ≥0.9）并追溯到训练数据来源，发现 30-60% 的低困惑度片段无法匹配训练数据，将可匹配片段分为四种记忆行为类别。

**[MultiSocial: Multilingual Benchmark of Machine-Generated Text Detection of Social-Media Texts](aigc_detection/multisocial_mgt_detection.md)**

:   构建了首个覆盖 22 种语言、5 个社交媒体平台、7 个 LLM 生成器的大规模机器生成文本检测基准 MultiSocial（47.2 万文本），实验表明 fine-tuned 检测器（Llama-3-8B/Mistral-7B, AUC ROC 0.977）在社交媒体文本上表现优异，且训练平台选择对跨平台泛化影响显著。

**[Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction](aigc_detection/reliably_bounding_false_positives_a_zero-shot_machine-generated_text_detection_f.md)**

:   提出基于多尺度保形预测（MCP）的零样本机器生成文本检测框架，通过文本长度感知的分组分位数计算，在严格约束假阳性率（FPR）上界的同时显著提升检测性能，并构建了覆盖15个领域、22个LLM的大规模双语基准数据集RealDet。

**[Who Writes What: Unveiling the Impact of Author Roles on AI-generated Text Detection](aigc_detection/who_writes_what_ai_detection.md)**

:   揭示作者的社会语言学属性（性别、CEFR水平、学科领域、语言环境）会系统性地影响AI生成文本检测器的准确率，其中语言水平和语言环境的偏差最为显著且一致，提出了基于多因素WLS+ANOVA的偏差量化框架。

---

## 🛡️ AI 安全 { #ai_safety }

**[Building a Long Text Privacy Policy Corpus with Multi-Class Labels](ai_safety/building_a_long_text_privacy_policy_corpus_with_multi-class_labels.md)**

:   本文构建了一个包含149家公司隐私政策的多维度标注语料库（64个标注维度），涵盖欧盟和美国隐私法规中的争议条款和法律规则，并使用当前大语言模型建立了分类基准。

**[CENTAUR: Bridging the Impossible Trinity of Privacy, Efficiency, and Performance in Privacy-Preserving Transformer Inference](ai_safety/centaur_bridging_the_impossible_trinity_of.md)**

:   提出 Centaur 框架，融合随机置换矩阵和安全多方计算（SMPC）来打破隐私保护 Transformer 推理（PPTI）中的"不可能三角"——同时实现强隐私保护、5-30x 加速和明文级别推理精度。

**[Crafting Privacy-Preserving Adversarial Examples: A Defense Against Membership Inference](ai_safety/crafting_privacy-preserving_adversarial_examples_a_defense_against_membership_inf.md)**

:   本文提出一种通过构造隐私保护型对抗样本来防御成员推理攻击（MIA）的方法，在模型预测输出中注入精心设计的扰动，使攻击者无法判断某条数据是否属于训练集，同时保持模型对正常用户的服务质量。

**[FairI Tales: Evaluation of Fairness in Indian Contexts with a Focus on Bias and Stereotypes](ai_safety/fairi_tales_evaluation_of_fairness_in_indian_contexts_with_a_focus_on_bias_and_s.md)**

:   本文提出 Indic-Bias，首个面向印度多元社会的大规模 LLM 公平性基准，通过 20,000 个人工验证的场景模板在三大评估任务上测试 14 个 LLM，揭示模型对达利特等边缘化群体存在严重负面偏见，且超过 70% 的情况下会强化刻板印象。

**[Gender Inclusivity Fairness Index (GIFI): A Multilevel Framework](ai_safety/gifi_gender_fairness.md)**

:   提出 GIFI（Gender Inclusivity Fairness Index），一个涵盖代词识别、情感中立性、毒性、反事实公平性、刻板印象关联、职业公平性和数学推理一致性七个维度的多层次评估框架，在 22 个主流 LLM 上系统量化二元与非二元性别的公平性，揭示新代词在无提示时完全缺席、"she" 过度矫正等深层偏见模式。

**[Multi-task Adversarial Attacks against Black-box Model with Few-shot Queries](ai_safety/multi-task_adversarial_attacks_against_black-box_model_with_few-shot_queries.md)**

:   提出 CEMA（Cluster and Ensemble Multi-task Text Adversarial Attack）方法，通过训练"深层替代模型"将复杂的多任务黑盒攻击转化为单任务文本分类攻击，仅需约 100 次查询即可同时攻击分类、翻译、摘要、文生图等多种任务，并在 ChatGPT-4o、百度翻译、Stable Diffusion 等商用模型上验证了有效性。

**[PrivaCI-Bench: Evaluating Privacy with Contextual Integrity and Legal Compliance](ai_safety/privacibench_evaluating_privacy_with_contextual_integrity.md)**

:   提出 PrivaCI-Bench，基于 Contextual Integrity 理论构建了目前最大的上下文隐私评估基准（154K 实例），涵盖真实法院案例、隐私政策和 EU AI Act 合规检查器合成数据，评估 LLM 在 HIPAA/GDPR/AI Act 下的法律合规能力。

**[Quantifying Misattribution Unfairness in Authorship Attribution](ai_safety/quantifying_misattribution_unfairness_in_authorship_attribution.md)**

:   本文提出MAUI_k指标量化作者归因系统中"错误归因不公平性"——某些作者系统性地更容易被误判为可疑作者，并发现这种不公平与作者嵌入在向量空间中距质心的距离高度相关。

**[Robust and Minimally Invasive Watermarking for EaaS](ai_safety/robust_and_minimally_invasive_watermarking_for_eaas.md)**

:   提出 ESpeW（Embedding-Specific Watermark），一种嵌入特异性水印方法，通过在每个嵌入向量的不同位置注入独特水印，实现对 Embeddings as a Service (EaaS) 的鲁棒版权保护，抵抗各种水印移除攻击且对嵌入质量的影响小于 1%。

**[Sandcastles in the Storm: Revisiting Watermarking Impossibility](ai_safety/sandcastles_watermarking_impossibility.md)**

:   本文通过大规模实验和人类评估挑战了 "Watermarks in the Sand" (WITS) 的理论不可能性结论：证明随机游走攻击的两个关键假设在实践中不成立——混合(mixing)速度极慢（100% 的攻击文本仍可追溯原始来源）且质量预言机(quality oracle)不可靠（仅 77% 准确率），自动攻击仅 26% 成功率，人类质量审核后降至 10%。

**[SpeechFake: A Large-Scale Multilingual Speech Deepfake Dataset Incorporating Cutting-Edge Generation Methods](ai_safety/speechfake_a_largescale_multilingual_speech_deepfake.md)**

:   构建 SpeechFake 大规模语音深伪数据集，包含 300 万+深伪样本、3000+ 小时音频、40 种生成工具和 46 种语言，并通过基线实验系统分析了生成方法、语言多样性和说话人变化对检测性能的影响。

**[Towards Fairness Assessment of Dutch Hate Speech Detection](ai_safety/towards_fairness_assessment_of_dutch_hate_speech_detection.md)**

:   本文系统评估了荷兰语仇恨言论检测模型的反事实公平性，提出四种反事实数据生成方法（LLMdef、LLMlist、SLL、MGS），并通过在 BERTje 模型上微调验证了反事实数据增强对模型性能和公平性的改进效果。

**[Efficiently Identifying Watermarked Segments in Mixed-Source Texts](ai_safety/watermark_segment_detection.md)**

:   提出两种高效方法（Geometric Cover Detector 和 Adaptive Online Locator）用于在长文混合来源文本中检测和精确定位水印片段，时间复杂度从 O(n²) 降至 O(n log n)，在三种主流水印技术上均显著优于 baseline。

**[WET: Overcoming Paraphrasing Vulnerabilities in Embeddings-as-a-Service with Linear Transformation Watermark](ai_safety/wet_eaas_watermark.md)**

:   揭示了现有 EaaS 嵌入水印（EmbMarker/WARDEN）可被改写攻击绕过，提出 WET（线性变换水印），通过秘密循环矩阵对嵌入做线性变换注入水印，理论和实验证明其对改写攻击具有鲁棒性，验证 AUC 接近 100%。

---

## 🩺 医疗 NLP { #medical_nlp }

**[Adaptive-VP: A Framework for LLM-Based Virtual Patients that Adapts to Trainees' Dialogue to Facilitate Nurse Communication Training](medical_nlp/adaptive-vp_a_framework_for_llm-based_virtual_patients_that_adapts_to_trainees_d.md)**

:   提出 Adaptive-VP 框架，利用 LLM 构建可根据护理学员沟通质量动态调整行为的虚拟病人（VP），通过多 Agent 评估→动态适应→对话生成→安全监控的四模块管线，在 28 名护理专家的 between-subjects 实验中显著提升了 VP 交互的感知真实感（角色保真度 $\eta_p^2 = 0.151$，对话真实感 $\eta_p^2 = 0.254$）。

**[AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset](medical_nlp/afrimed_qa_pan_african.md)**

:   构建首个大规模泛非洲医学问答基准 AfriMed-QA（15,275 题，16 国 60+ 医学院校、32 个专科），系统评估 30 个 LLM 并发现非洲医疗场景下存在显著的地域性能差距和生物医学模型反不如通用模型的反直觉现象。

**[Are LLMs Effective Psychological Assessors? Leveraging Adaptive RAG for Interpretable Mental Health Screening through Psychometric Practice](medical_nlp/are_llms_effective_psychological_assessors_leveraging_adaptive_rag_for_interpret.md)**

:   本文提出了一种基于问卷引导的心理健康筛查框架，通过自适应RAG从用户Reddit帖子中检索相关内容，再用LLM代为填写标准化心理量表（如BDI-II），在无需训练数据的情况下匹配或超越有监督方法的性能，同时提供了临床可解释的评估结果。

**[ArgHiTZ at ArchEHR-QA 2025: A Two-Step Divide and Conquer Approach to Patient Question Answering for Top Factuality](medical_nlp/arghitz_at_archehr-qa_2025_a_two-step_divide_and_conquer_approach_to_patient_que.md)**

:   在 ArchEHR-QA 2025 共享任务中提出两阶段"分治"方法：先用重排序模型从电子健康记录中提取关键句子，再用小型医学 LLM 生成回复，在不使用外部知识的情况下取得事实性排名第一、总分第 8/30 的成绩。

**[Improving Automatic Evaluation of LLMs in Biomedical Relation Extraction via LLMs-as-the-Judge](medical_nlp/biore_llm_judge_evaluation.md)**

:   本文首次系统研究了 LLM-as-Judge 在生物医学关系抽取评估中的表现，发现其准确率通常低于 50%，并提出结构化输出格式（JSON）和域适应技术来提升约 15% 的评估准确率。

**[CliniDial: A Naturally Occurring Multimodal Dialogue Dataset for Team Reflection in Action During Clinical Operation](medical_nlp/clinidial_a_naturally_occurring_multimodal_dialogue_dataset_for_team_reflection_.md)**

:   构建了 CliniDial 数据集，收集自模拟临床手术中的自然对话，包含音频转录、双角度视频和患者生理信号等多模态数据，标注了团队反思行为编码，揭示了现有 LLM 在处理标签不均衡、自然对话交互和领域多模态数据方面的显著不足。

**[Enhancing Medical Dialogue Generation through Knowledge Refinement and Dynamic Prompt Adjustment](medical_nlp/enhancing_medical_dialogue_generation_through_knowledge_refinement_and_dynamic_p.md)**

:   提出 MedRef，一种融合知识精炼机制和动态 Prompt 调整策略的医学对话系统，通过隐变量过滤无关知识图谱三元组、实体-行为联合预测、以及三元组过滤器和示例选择器动态构建系统 Prompt，在 MedDG 和 KaMed 两个基准上取得 SOTA 性能。

**[Evaluation of LLMs in Medical Text Summarization: The Role of Vocabulary Adaptation in High OOV Settings](medical_nlp/evaluation_of_llms_in_medical_text_summarization_the_role_of_vocabulary_adaptati.md)**

:   系统性基准研究发现 LLM 在高 OOV（词汇外词）和高新颖性医学文本摘要场景下性能显著下降，并通过多种词汇适配策略（MEDVOC、MEDVOC-LLM、ScafFix）证明即使 Llama-3.1（128K 词汇量）仍受过度分片问题困扰，词汇适配可带来显著改善。

**[LLMs Can Simulate Standardized Patients via Agent Coevolution](medical_nlp/evopatient_standardized_patient.md)**

:   EvoPatient 提出了一个多智能体协同进化框架，通过患者 Agent 和医生 Agent 之间的自主模拟对话，让 LLM 无需人工监督即可学会模拟标准化病人（SP），在需求对齐度上超过现有推理方法 10%+。

**[Follow-up Question Generation for Enhanced Patient-Provider Conversations](medical_nlp/follow-up_question_generation_for_enhanced_patient-provider_conversations.md)**

:   提出 FollowupQ 多智能体框架，结合 EHR 推理、鉴别诊断和消息澄清三类 Agent，为异步医患对话自动生成个性化追问列表，在真实和半合成数据集上分别比基线提升 17% 和 5% 的 RIM 分数，将医生需要额外发送的信息收集消息减少 34%。

**[ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents](medical_nlp/reflectool_clinical_agent.md)**

:   ReflecTool 提出了一个反思感知的工具增强临床 Agent 框架，通过优化阶段积累成功轨迹和工具级经验，推理阶段检索相似案例并用验证器改进工具使用，在涵盖 18 个任务的 ClinicalAgent Bench 上超越纯 LLM 10+ 分、超越已有 Agent 方法 3 分。

**[Query-driven Document-level Scientific Evidence Extraction from Biomedical Studies](medical_nlp/urca_biomedical_evidence_extraction.md)**

:   本文提出 URCA（Uniform Retrieval Clustered Augmentation）框架，通过均匀检索+聚类+知识提取的 RAG 流程，从 RCT 研究全文中自动提取与临床问题相关的科学证据结论，在新构建的 CochraneForest 数据集上比最佳基线提升了 8.81% F1。

**[VITAL: A New Dataset for Benchmarking Pluralistic Alignment in Healthcare](medical_nlp/vital_pluralistic_alignment_healthcare.md)**

:   本文构建了首个面向医疗健康领域的多元化对齐（pluralistic alignment）基准数据集 VITAL，包含 13.1K 价值观情境和 5.4K 多选题，并通过对 8 个 LLM 的广泛评估表明，现有多元化对齐技术（尤其是 ModPlural）在医疗场景下表现不佳，简单的 prompting 反而效果更好。

---

## 🔗 因果推理 { #causal_inference }

**[Causal Graph based Event Reasoning using Semantic Relation Experts](causal_inference/causal_graph_based_event_reasoning_using_semantic_relation_experts.md)**

:   提出基于四类语义关系专家（时间、篇章、条件、常识）多轮协作讨论的因果事件图生成框架，在零样本设置下于事件预测、事件预报等多个下游任务上取得与微调模型竞争的结果，并提供可解释的因果事件链。

**[CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation](causal_inference/causalrag_integrating_causal_graphs_into_retrieval-augmented_generation.md)**

:   提出 CausalRAG，将因果图集成到 RAG 的检索过程中——从文档构建文本图并识别因果关系，在查询时通过因果路径发现和因果摘要生成来检索上下文，在文档问答中显著提升上下文精度（92.86%）和检索召回率。

**[CoA-Reasoning: Explorations on Counterfactual Analysis in Physical Reasoning of LVLMs](causal_inference/coa-reasoning_explorations_on_counterfactual_analysis_in_physical_reasoning_of_l.md)**

:   本文提出CoA-Reasoning框架，通过构造反事实场景来系统性地评估和增强大型视觉语言模型（LVLMs）在物理世界推理中的因果理解能力，揭示了现有模型在反事实物理推理上的显著不足。

**[Counterfactual-Consistency Prompting for Relative Temporal Understanding in Large Language Models](causal_inference/counterfactual-consistency_prompting_for_relative_temporal_understanding_in_larg.md)**

:   本文提出了一种反事实一致性提示（Counterfactual-Consistency Prompting）方法，通过生成反事实问题并施加集体约束来解决大语言模型在时序推理中的不一致性问题，在多个时序理解数据集上取得了显著改进。

**[Counterfactual Explanations for Aspect-Based Sentiment Analysis](causal_inference/counterfactual_explanations_for_aspect-based_sentiment_analysis.md)**

:   本文提出一种为方面级情感分析（ABSA）生成反事实解释的方法，通过找到能翻转特定方面情感极性的最小文本修改，为 ABSA 模型的预测提供直观的因果解释。

**[FitCF: A Framework for Automatic Feature Importance-guided Counterfactual Example Generation](causal_inference/fitcf_a_framework_for_automatic_feature_importance-guided_counterfactual_example.md)**

:   提出 FitCF 框架，利用 BERT 特征归因方法（LIME/IG/SHAP等）提取重要词来引导 LLM 在 zero-shot 下生成反事实样本（ZeroCF），再经标签翻转验证筛选后作为 few-shot 示例，在新闻分类和情感分析任务上一致性超越 Polyjuice、BAE、FIZLE 三种基线。

**[IRIS: An Iterative and Integrated Framework for Verifiable Causal Discovery](causal_inference/iris_an_iterative_and_integrated_framework.md)**

:   提出 IRIS 框架——仅需一组初始变量名作为输入，即可自动检索文档、提取变量值构建结构化数据、通过混合因果发现（GES 统计算法 + LLM 因果关系验证）构建因果图，并通过缺失变量提议组件迭代扩展变量集合，放松了传统方法的无环和因果充分性假设，在 Cancer、Diabetes、Obesity、ADNI、Insurance 等 6 个数据集上 F1 全面超越 0-shot/CoT/RAG 基线。

**[Leveraging Variation Theory in Counterfactual Data Augmentation for Optimized Active Learning](causal_inference/leveraging_variation_theory_in_counterfactual_data_augmentation_for_optimized_ac.md)**

:   本文将变异理论(Variation Theory)引入反事实数据增强(CDA)框架，通过保留神经符号模式的方式使用LLM生成反事实样本，并结合三级过滤流水线筛选高质量数据，用于优化主动学习中的少样本文本分类，在多个数据集上取得显著F1提升。

**[On the Reliability of Large Language Models for Causal Discovery](causal_inference/llm_causal_discovery_reliability.md)**

:   利用开源 LLM（OLMo、BLOOM）可访问的预训练语料库，实证验证了"因果鹦鹉"假说——LLM 识别因果关系的能力与预训练数据中该关系的出现频率高度相关（Spearman r=0.9），且错误因果关系的存在和上下文变化都会显著影响预测可靠性。

**[Reasoning is All You Need for Video Generalization: A Counterfactual Benchmark with Sub-question Evaluation](causal_inference/reasoning_is_all_you_need_for_video_generalization_a_counterfactual_benchmark_wi.md)**

:   提出 COVER（COunterfactual VidEo Reasoning），一个多维度视频反事实推理 benchmark，将评估任务按抽象-具体和感知-认知两个维度分为四象限共 13 类任务，并通过将复杂问题分解为子问题（必要条件）来揭示——子问题准确率与反事实推理能力强相关，提升推理能力是改善视频理解鲁棒性的关键。

---

## 📹 视频理解 { #video_understanding }

**[A Thousand Words Paint a Picture: Multimodal Goal Tracking for Grounded Social Intelligence](video_understanding/a_thousand_words_paint_a_picture_multimodal_goal_tracking_for_grounded_social_in.md)**

:   本文提出了一个多模态目标追踪框架，通过结合视觉与语言线索来推理社交场景中参与者的隐含目标，从而提升模型对社交情境的理解能力（即"落地的社会智能"）。

**[Addressing Blind Guessing: Calibration of Selection Bias in Multiple-Choice Question Answering by Video Language Models](video_understanding/addressing_blind_guessing_calibration_of_selection_bias_in_multiple-choice_quest.md)**

:   首次系统性研究视频语言模型（VLM）在多选题回答中的选项选择偏差问题，通过任务分解分析偏差来源，提出BOLD后处理校准技术，在减少偏差的同时提升模型性能。

**[Attention-Seeker: Dynamic Self-Attention Scoring for Unsupervised Key-Frame Extraction](video_understanding/attention-seeker_dynamic_self-attention_scoring_for_unsupervised_key-frame_extra.md)**

:   本文提出Attention-Seeker方法，通过动态地分析Transformer模型中自注意力层的注意力得分分布，无需任何监督信号即可从视频中提取最具代表性的关键帧，在多个视频摘要基准数据集上超越了现有的无监督方法。

**[From Teacher to Student: Tracking Memorization Through Model Distillation](video_understanding/from_teacher_to_student_tracking_memorization_through_model_distillation.md)**

:   系统研究了知识蒸馏（KD）对大语言模型记忆化行为的影响，发现蒸馏不仅能压缩模型，还能显著降低对训练数据的逐字记忆风险——其中反向 KL 蒸馏（RKLD/MiniLLM）将记忆化比例从 SFT 的 65.4% 降至最低 6.0%。

**[Generative Frame Sampler for Long Video Understanding](video_understanding/generative_frame_sampler_for_long_video_understanding.md)**

:   提出 GenS，一个基于 VideoLLM 的生成式帧采样模块，用自然语言输出question-aware的相关帧时间段和置信度分数，作为即插即用模块在 LongVideoBench/MLVU/HourVideo 上为多种 VideoLLM 带来 2-4 个点的一致提升。

**[ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs](video_understanding/icr_probe_tracking_hidden_state_dynamics_for_reliable_hallucination_detection_in.md)**

:   提出 ICR Score（Information Contribution to Residual Stream），通过测量 MHSA 和 FFN 模块对隐状态更新的贡献一致性来量化残差流动态，构建仅 16K 参数的 ICR Probe，在 4 个数据集 × 3 个 LLM 上幻觉检测 AUROC 全面超越基线。

**[Improving Dialogue State Tracking through Combinatorial Search for In-Context Examples](video_understanding/improving_dialogue_state_tracking_through_combinatorial_search_for_in-context_ex.md)**

:   提出 CombiSearch 方法，通过组合式评分为对话状态追踪（DST）选择最优 in-context 示例组合，在仅用 5% 训练数据的情况下超越所有使用 100% 数据的 baseline，理想设置下 JGA 上界比传统方法高 12%。

**[RAVEN: Robust Advertisement Video Violation Temporal Grounding via Reinforcement Reasoning](video_understanding/raven_robust_advertisement_video_violation_temporal_grounding_via_reinforcement_.md)**

:   本文提出RAVEN框架，将课程强化学习与多模态LLM结合，通过分层奖励机制和渐进式训练策略，实现广告视频违规内容的精确时序定位和类别预测，无需显式推理标注数据即可激发涌现推理能力。

**[Sparse-to-Dense: A Free Lunch for Lossless Acceleration of Video Understanding in LLMs](video_understanding/sparse-to-dense_a_free_lunch_for_lossless_acceleration_of_video_understanding_in.md)**

:   基于Video-LLM中注意力分数的稀疏性观察，提出Sparse-to-Dense (StD)解码策略，用top-K稀疏注意力模型作为draft model快速生成候选token，再用全注意力模型并行验证，实现最高1.94倍的无损加速，且无需额外训练或架构修改。

---

## 🎨 图像生成 { #image_generation }

**[A Unified Agentic Framework for Evaluating Conditional Image Generation](image_generation/a_unified_agentic_framework_for_evaluating_conditional_image_generation.md)**

:   提出 CIGEval，一个基于大型多模态模型（LMM）的统一 Agent 评估框架，通过工具集成（Grounding、Highlight、Difference、Scene Graph）和分而治之的评估策略，在 7 种条件图像生成任务上达到与人类标注者相当的相关性（0.4625 vs 人类间 0.47），且仅用 2.3K 训练数据微调 7B 模型即超越 GPT-4o 版 SOTA。

**[D-GEN: Automatic Distractor Generation and Evaluation for Reliable Assessment of Generative Models](image_generation/d-gen_automatic_distractor_generation_and_evaluation_for_reliable_assessment_of_.md)**

:   提出 D-GEN——首个开源干扰项生成模型（LLaMA微调，8B/70B），自动将开放式评测题转为多选题格式，配套排名对齐+熵分析两种评估方法验证干扰项质量，在 MMLU 上 Spearman's ρ=0.99 保持模型排名一致性。

**[Planning with Diffusion Models for Target-Oriented Dialogue Systems](image_generation/difftod_diffusion_dialogue_planning.md)**

:   DiffTOD 将对话规划建模为轨迹生成问题，利用掩码扩散语言模型实现非顺序对话规划，并设计三种引导机制（词级/语义级/搜索级）灵活控制对话朝目标推进，在谈判/推荐/闲聊三种场景上显著超越基线。

**[FlashAudio: Rectified Flows for Fast and High-Fidelity Text-to-Audio Generation](image_generation/flashaudio_rectified_flow_tta.md)**

:   将整流流（Rectified Flow）引入文本转音频生成，通过双焦采样器优化时间步分布、不混溶流减少数据-噪声总距离、锚定优化修正 CFG 引导误差，实现单步生成 FAD=1.49 超越百步扩散模型，生成速度达实时 400 倍。

**[Generating Pedagogically Meaningful Visuals for Math Word Problems: A New Benchmark and Analysis of Text-to-Image Models](image_generation/generating_pedagogically_meaningful_visuals_for_math_word_problems_a_new_benchma.md)**

:   Math2Visual 提出了一个从数学应用题（MWP）文本描述自动生成教学可视化图像的框架，定义了基于教师访谈的视觉语言和设计空间，构建了 1,903 张标注数据集，并评估和微调了多个 TTI 模型，揭示了当前模型在数学关系表示上的关键不足。

**[Multimodal Pragmatic Jailbreak on Text-to-image Models](image_generation/multimodal_pragmatic_jailbreak_on_text-to-image_models.md)**

:   提出"多模态语用越狱"（Multimodal Pragmatic Jailbreak）新型攻击方式，通过让T2I模型生成包含视觉文字的图像，使得图像内容和文字内容单独看都安全但组合后产生不安全内容，揭示了所有测试模型（包括DALL·E 3）均受此攻击影响。

**[OZSpeech: One-step Zero-shot Speech Synthesis with Learned-Prior-Conditioned Flow Matching](image_generation/ozspeech_one-step_zero-shot_speech_synthesis_with_learned-prior-conditioned_flow.md)**

:   提出OZSpeech，首个将最优传输条件流匹配(OT-CFM)与学习先验分布相结合实现单步采样的零样本TTS系统，在内容准确性(WER)、推理速度和模型大小上均大幅领先现有方法。

**[R-VC: Rhythm Controllable and Efficient Zero-Shot Voice Conversion via Shortcut Flow Matching](image_generation/rvc_rhythm_voice_conversion.md)**

:   R-VC 是首个实现节奏可控的零样本语音转换系统，通过 Mask Transformer 时长模型建模目标说话人的节奏风格，结合 Shortcut Flow Matching 的 DiT 解码器实现仅 2 步采样的高效高质量语音生成，在 LibriSpeech 上 WER 3.51、说话人相似度 0.930。

---

## 👥 多智能体 { #multi_agent }

**[Beyond Frameworks: Unpacking Collaboration Strategies in Multi-Agent Systems](multi_agent/beyond_frameworks_multi_agent_collaboration.md)**

:   本文系统化地将多智能体协作分解为四个维度（治理模式、参与控制、交互模式、上下文管理），通过两个上下文依赖任务的大量实验证明：集中治理+指导者控制参与+有序交互+指导者摘要的组合最优，在保持甚至提升准确率的同时减少高达 93% 的 token 消耗。

**[CoMet: Metaphor-Driven Covert Communication for Multi-Agent Language Games](multi_agent/comet_metaphor-driven_covert_communication_for_multi-agent_language_games.md)**

:   本文提出 CoMet 框架，通过整合基于假设检验的隐喻推理器和自改进式隐喻生成器，使 LLM 智能体能在多智能体语言博弈中运用隐喻进行隐蔽通信和语义规避，在 Undercover 和 Adversarial Taboo 两个游戏中显著提升了智能体的策略沟通能力（胜率从 0.20 提升至 0.70）。

**[CortexDebate: Debating Sparsely and Equally for Multi-Agent Debate](multi_agent/cortexdebate_debating_sparsely_and_equally_for_multi-agent_debate.md)**

:   提出 CortexDebate，一种受人脑皮层工作机制启发的多智能体辩论方法，通过构建稀疏动态辩论图和基于 McKinsey 信任公式的评估模块（MDM），同时解决了现有 MAD 方法中"输入上下文过长"和"过度自信导致不平等辩论"两大核心问题。

**[DocAgent: A Multi-Agent System for Automated Code Documentation Generation](multi_agent/docagent_a_multi-agent_system_for_automated_code_documentation_generation.md)**

:   提出 DocAgent，一个基于拓扑依赖排序的多智能体代码文档生成系统，通过 Reader-Searcher-Writer-Verifier 协作流程增量构建上下文，在完整性、实用性和真实性三个维度上显著优于 FIM 和 Chat 基线。

**[GETReason: Enhancing Image Context Extraction through Hierarchical Multi-Agent Reasoning](multi_agent/getreason_enhancing_image_context_extraction_through_hierarchical_multi-agent_re.md)**

:   提出 GETReason，一个层级化多智能体框架，通过将公共事件图像的上下文提取分解为地理空间、时间和事件三个子任务，并由专门化的 Agent 协作完成，实现比现有方法更准确的图像上下文推理。

**[Multi-Agent Collaboration via Cross-Team Orchestration](multi_agent/multi-agent_collaboration_via_cross-team_orchestration.md)**

:   提出 Cross-Team Orchestration (Croto)，一个可扩展的多团队协作框架，通过将多个独立 agent 团队组织起来进行跨团队交互，利用层次化分组 (Hierarchy Partitioning) 和贪心聚合 (Greedy Aggregation) 机制将各团队的多样化解决方案融合为更优结果。

**[Preventing Rogue Agents Improves Multi-Agent Collaboration](multi_agent/preventing_rogue_agents_improves_multi-agent_collaboration.md)**

:   提出一种通过实时监控 Agent 不确定性来检测"失控 Agent"（rogue agent）并进行干预的框架，在自建的 WhoDunitEnv 多智能体协作环境以及代码生成和资源可持续性任务上分别取得高达 17.4%、2.5% 和 20% 的性能提升。

**[Voting or Consensus? Decision-Making in Multi-Agent Debate](multi_agent/voting_or_consensus_decision-making_in_multi-agent_debate.md)**

:   系统性对比了多智能体辩论中 7 种决策协议（投票 vs 共识），发现共识协议在知识任务上提升 2.8%、投票协议在推理任务上提升 13.2%，并提出 AAD 和 CI 两种增强答案多样性的新方法，分别带来 3.3% 和 7.4% 的性能提升。

---

## 🎮 强化学习 { #reinforcement_learning }

**[Align-SLM: Textless Spoken Language Models with Reinforcement Learning from AI Feedback](reinforcement_learning/align-slm_textless_spoken_language_models_with_reinforcement_learning_from_ai_fe.md)**

:   本文提出 Align-SLM 框架，首次将偏好优化（DPO + RLAIF）应用于纯语音语言模型（无文本注入），通过 LLM 自动评估生成的语音续写质量构建偏好数据，结合课程学习迭代提升 SLM 的语义理解能力，在 ZeroSpeech 和 StoryCloze 等基准上达到 SLM 的 SOTA。

**[Bypass Back-propagation: Optimization-based Structural Pruning for Large Language Models via Policy Gradient](reinforcement_learning/bypass_back-propagation_optimization-based_structural_pruning_for_large_language.md)**

:   本文提出一种基于策略梯度的LLM结构化剪枝方法，通过在概率空间中学习伯努利剪枝掩码来直接优化剪枝模型的损失函数，全程无需对LLM本身进行反向传播，仅需前向推理即可完成剪枝优化。

**[An Efficient Task-Oriented Dialogue Policy: Evolutionary Reinforcement Learning Injected by Elite Individuals](reinforcement_learning/eierl_dialogue_policy.md)**

:   首次将进化强化学习（ERL）应用于任务导向对话策略任务，提出 EIERL 方法结合 EA 的全局探索与 DRL 的局部优化，并通过精英个体注入（EII）机制解决 EA 在自然语言大搜索空间中进化缓慢的问题，在 4 个数据集上实现了更高效的探索-利用平衡。

**[Learning to Generate Structured Output with Schema Reinforcement Learning](reinforcement_learning/learning_to_generate_structured_output_with_schema_reinforcement_learning.md)**

:   提出 SchemaBench 基准（约4万条 JSON schema）和 Schema Reinforcement Learning (SRL) 训练框架，通过细粒度 schema 验证器提供密集奖励信号，结合 Thoughts of Structure (ToS) 推理机制，将 LLM 的复杂 JSON 生成准确率提升高达16%，同时不损害通用推理能力。

**[LLM-Enhanced Self-Evolving Reinforcement Learning for Multi-Step E-Commerce Payment Fraud Risk Detection](reinforcement_learning/llm-enhanced_self-evolving_reinforcement_learning_for_multi-step_e-commerce_paym.md)**

:   将电商支付欺诈检测建模为多步 MDP，用 LLM（Mixtral/LLaMA/Gemma）通过进化算法自动生成和优化 RL 奖励函数，在 eBay 真实交易数据上比人工设计奖励函数和传统 SL 基线显著提升 dollar-wise precision。

**[MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning](reinforcement_learning/maporl_multi-agent_post-co-training_for_collaborative_large_language_models_with.md)**

:   提出 MAPoRL——一种基于多智能体强化学习的后训练范式，通过让多个 LLM 在辩论框架中共同训练（co-training），配合验证器评分和协作激励机制，显著提升多 LLM 协作的效果，并展现出跨任务的泛化能力。

**[Prompt-based Personality Profiling: Reinforcement Learning for Relevance Filtering](reinforcement_learning/prompt-based_personality_profiling_reinforcement_learning_for_relevance_filterin.md)**

:   提出RL-Profiler方法，用强化学习训练一个帖子相关性过滤器（SelNet），从用户Profile的大量帖子中筛选出与人格特征相关的少量帖子，再交给LLM零样本预测人格，在大幅减少上下文长度的同时保持接近使用全部帖子的预测效果。

**[TreeRL: LLM Reinforcement Learning with On-Policy Tree Search](reinforcement_learning/treerl_tree_search_rl.md)**

:   提出 TreeRL，将基于熵引导的树搜索（EPTree）直接集成到 LLM 的 on-policy 强化学习训练中，通过在高不确定性 token 处分叉来扩展推理路径多样性，并利用树结构提供的全局+局部优势作为过程监督信号，在数学和代码推理任务上超过传统的多链采样 RL。

---

## 🎁 推荐系统 { #recommender }

**[Beyond Single Labels: Improving Conversational Recommendation through LLM-Powered Data Augmentation](recommender/beyond_single_labels_improving_conversational_recommendation_through_llm-powered.md)**

:   针对对话推荐系统中的假阴性问题（用户可能喜欢的item被错误标记为负样本），提出基于LLM的数据增强框架，通过语义检索+相关性打分生成合成标签，再通过两阶段训练策略平衡语义相关性和协同信息。

**[Laser: Bi-Tuning with Collaborative Information for Controllable LLM-Based Sequential Recommendation](recommender/bi-tuning_with_collaborative_information_for_controllable_llm-based_sequential_r.md)**

:   本文提出Laser框架，通过在LLM输入的前缀和后缀分别插入可训练虚拟token（Bi-Tuning），将用户-物品协同信息注入冻结的LLM，并设计基于MoE的M-Former来捕获不同类型用户的差异化特征，实现参数高效的序列推荐。

**[CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems](recommender/cove_compressed_vocabulary_expansion_makes_better_llm-based_recommender_systems.md)**

:   提出 CoVE 框架，通过扩展 LLM 词表为每个物品分配唯一 token ID 和嵌入，将序列推荐任务转化为 next-token prediction，相比现有方法推荐准确率提升最高 62%，推理速度提升约 100 倍，并通过哈希嵌入压缩解决大规模场景的内存问题。

**[GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion](recommender/gram_generative_recommendation.md)**

:   提出 GRAM 生成式推荐框架，通过**语义到词汇翻译**将隐式物品层次/协同关系编码到 LLM 词汇空间，并用**多粒度迟融合**独立编码不同粒度提示再在解码端融合，在四个基准上 Recall@5 提升 11.5–16.0%、NDCG@5 提升 5.3–13.6%。

**[KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models](recommender/kerl_knowledge-enhanced_personalized_recipe_recommendation_using_large_language_.md)**

:   提出 KERL 统一食品推荐系统，结合 FoodKG 知识图谱和 Phi-3-mini 多 LoRA 微调，实现个性化食谱推荐（F1=0.973）、食谱生成和微量营养素估算三个功能，大幅超越基线 LLM 和传统嵌入方法。

**[LOTUS: A Leaderboard for Detailed Image Captioning from Quality to Societal Bias and User Preferences](recommender/lotus_a_leaderboard_for_detailed_image_captioning_from_quality_to_societal_bias_.md)**

:   提出 LOTUS 排行榜，从描述质量（对齐性、描述性、语言复杂度）、副作用（幻觉、有害性）和社会偏见（性别、肤色）三个维度统一评估大型视觉语言模型的详细图像描述能力，并支持基于用户偏好的定制化评估。

**[RecLM: Recommendation Instruction Tuning](recommender/reclm_recommendation_instruction_tuning.md)**

:   提出 RecLM，一个模型无关的推荐指令微调框架，通过两轮对话式指令微调将协同过滤信号注入 LLM 生成的用户/商品画像，再用 RLHF（PPO）精炼画像质量，在 MIND/Netflix/工业数据集上作为即插即用组件为 BiasMF/NCF/LightGCN/SGL/SimGCL 一致带来提升，尤其在冷启动场景效果显著。

---

## 🤖 机器人/具身智能 { #robotics }

**[CHEER-Ekman: Fine-grained Embodied Emotion Classification](robotics/cheer-ekman_fine-grained_embodied_emotion_classification.md)**

:   本文提出CHEER-Ekman数据集，将CHEER数据集的二元具身情感标注扩展为Ekman六类基础情绪，并采用基于LLM的自动Best-Worst Scaling（BWS）技术实现无需任务特定训练的细粒度情感分类，性能超越有监督BERT。

**[Rolling the DICE on Idiomaticity: How LLMs Fail to Grasp Context](robotics/dice_idiomaticity.md)**

:   提出 DICE 数据集（2066 句，402 个习语），通过严格控制习语形式一致的对比评测，揭示 LLM 在需要上下文理解才能消歧习语（字面 vs 比喻义）时存在系统性缺陷。

**[Do Emotions Really Affect Argument Convincingness? A Dynamic Approach with LLM-based Manipulation Checks](robotics/do_emotions_really_affect_argument_convincingness_a_dynamic_approach_with_llm-ba.md)**

:   提出一种受心理学操控检验启发的动态框架，利用LLM调节论证的情感强度，系统考察情感对论证说服力的因果影响，发现超过半数情况下人类的说服力判断不受情感变化影响，而当情感有影响时更多是增强而非削弱说服力。

**[DRAE: Dynamic Retrieval-Augmented Expert Networks for Lifelong Learning and Task Adaptation in Robotics](robotics/drae_dynamic_retrieval-augmented_expert_networks_for_lifelong_learning_and_task_.md)**

:   提出 DRAE 框架，整合动态 MoE 路由、参数化 RAG（P-RAG）、三层认知控制架构（ReflexNet-SchemaPlanner-HyperOptima）和 DPMM 终身知识保留，在机器人操作和自动驾驶任务上平均成功率达 82.5%，有效缓解灾难性遗忘。

**[Task-aware MoILE: Hierarchical-Task-Aware Multi-modal Mixture of Incremental LoRA Experts for Embodied Continual Learning](robotics/hierarchical-task-aware_multi-modal_mixture_of_incremental_lora_experts_for_embo.md)**

:   提出层次化具身持续学习设置（HEC），将 agent 学习分为高层指令和低层动作两级，并设计 Task-aware MoILE 方法——通过跨模态聚类识别任务、双路由器选择 LoRA 专家、SVD 正交训练保留旧知识，在 5 种增量学习场景中遗忘率降至 3.37%（vs 前 SOTA 7.44%）。

**[SELF-PERCEPT: Introspection Improves LLMs' Detection of Multi-Person Mental Manipulation in Conversations](robotics/self_percept_manipulation_detection.md)**

:   提出 SELF-PERCEPT 两阶段 prompting 框架，借鉴心理学自我知觉理论（Self-Perception Theory），引导 LLM 先观察对话参与者的行为线索再推断内在态度，显著提升多人多轮对话中心理操纵的检测效果。

**[Vulnerability of LLMs to Vertically Aligned Text Manipulations](robotics/vulnerability_of_llms_to_vertically_aligned_text_manipulations.md)**

:   本文系统揭示了LLM对垂直排列文本输入的严重脆弱性：仅将少量关键词垂直排列即可导致文本分类准确率下降25-45个百分点，CoT推理无法缓解此问题，但精心设计的few-shot learning可有效恢复性能。

---

## 📈 时间序列 { #time_series }

**[ANRE: Analogical Replay for Temporal Knowledge Graph Forecasting](time_series/anre_analogical_replay_for_temporal_knowledge_graph_forecasting.md)**

:   本文提出ANRE（Analogical Replay）方法，通过检索历史知识图谱快照中与当前查询结构类似的"类比事件"，将其作为推理线索辅助时序知识图谱的未来事件预测，在多个基准数据集上取得了显著的性能提升。

**[Context-Aware Sentiment Forecasting via LLM-based Multi-Perspective Role-Playing Agents](time_series/context_aware_sentiment_forecasting_agents.md)**

:   提出一个基于 LLM 的多视角角色扮演框架（MPR），通过主观 Agent 模拟用户发帖、客观 Agent（微调的"心理学家"LLM）审查行为一致性，以迭代纠正的方式预测社交媒体用户对实时事件的未来情感反应，在宏观和微观层面均大幅超越传统方法。

**[CTPD: Cross-Modal Temporal Pattern Discovery for Enhanced Multimodal Electronic Health Records Analysis](time_series/ctpd_cross-modal_temporal_pattern_discovery_for_enhanced_multimodal_electronic_h.md)**

:   提出 CTPD 框架，利用 Slot Attention 从多模态 EHR 数据（不规则时间序列+临床笔记）中发现跨模态共享的时序原型模式，通过 TP-NCE 对比损失对齐两模态的时序语义，在 MIMIC-III 的死亡率预测和表型分类任务上取得 SOTA。

**[G2S: A General-to-Specific Learning Framework for Temporal Knowledge Graph Forecasting with Large Language Models](time_series/g2s_a_general-to-specific_learning_framework_for_temporal_knowledge_graph_foreca.md)**

:   提出 G2S 框架，将时序知识图谱（TKG）预测中的通用模式（时序结构规律）与场景信息（具体实体/关系）解耦，先在匿名化时序结构上学习通用模式，再注入场景信息，有效提升 LLM 在 TKG 预测中的泛化能力。

**[LETS-C: Leveraging Text Embedding for Time Series Classification](time_series/lets-c_leveraging_text_embedding_for_time_series_classification.md)**

:   提出 LETS-C——将时间序列数字化为文本字符串后用 text embedding 模型编码，与原始时间序列元素级相加融合后送入轻量 CNN+MLP 分类头，在 UEA 10 个多变量时间序列数据集上以仅 14.5% 的可训练参数量超越 OneFitsAll（GPT-2 微调）等 27 个 baseline 达到 SOTA。

**[Revisiting LLMs as Zero-Shot Time-Series Forecasters: Small Noise Can Break Large Models](time_series/revisiting_llms_as_zero-shot_time_series_forecasters_small_noise_can_break_large.md)**

:   本文系统评估了 LLM 作为零样本时间序列预测器的有效性，发现 LLM 对输入噪声极度敏感——即使少量噪声也会使性能大幅下降，甚至不如简单的领域专用模型（如 DLinear），建议未来应聚焦于对 LLM 进行微调以更好地处理数值序列。

**[Time-MQA: Time Series Multi-Task Question Answering with Context Enhancement](time_series/time-mqa_time_series_multi-task_question_answering_with_context_enhancement.md)**

:   提出Time-MQA框架和TSQA数据集（~200k QA对），将时间序列的预测、填补、异常检测、分类和开放式推理问答统一到自然语言问答范式下，通过持续预训练LLM使其具备时间序列理解和推理能力。

---

## 🔄 自监督/表示学习 { #self_supervised }

**[Improving Low-Resource Morphological Inflection via Self-Supervised Objectives](self_supervised/improving_low-resource_morphological_inflection_via_self-supervised_objectives.md)**

:   系统探索 13 种自监督辅助目标（自编码、CMLM、T5-style 等）在极低资源形态变化任务中的效果，发现无标注数据极少时自编码最优，数据增多后字符级 MLM 更好，按形态素边界采样掩码是最有前景的方向。

**[Contrastive Learning on LLM Back Generation Treebank for Cross-domain Constituency Parsing](self_supervised/llm_back_gen_treebank.md)**

:   提出 LLM 反向生成 (LLM Back Generation) 方法，将不完整的跨领域句法树作为输入让 LLM 补全缺失词生成 treebank，并设计 span 级别对比学习预训练策略，实现跨领域成分句法分析的 SOTA 性能。

**[Magnet: Augmenting Generative Decoders with Representation Learning and Infilling Capabilities](self_supervised/magnet_augmenting_generative_decoders_with_representation_learning_and_infilling.md)**

:   提出 Magnet 方法，通过混合注意力机制（双向+因果）和三个自监督目标（掩码预测+对比学习+缺失片段生成），将纯解码器 LLM 同时增强为文本编码器和填充模型，在 token 级和句子级表示学习任务上超越 LLM2Vec 等专用方法，同时避免了双向化带来的严重文本重复问题。

**[QAEncoder: Towards Aligned Representation Learning in Question Answering Systems](self_supervised/qaencoder_aligned_representation.md)**

:   提出 QAEncoder，一种免训练方法通过蒙特卡洛采样估计文档对应查询的期望嵌入作为文档表示的代理，配合文档指纹保持区分性，在 BEIR 上将 bge-large 从 58.5 提升到 61.8 NDCG@10，零额外存储和延迟开销。

**[SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction](self_supervised/shubert_self-supervised_sign_language_representation_learning_via_multi-stream_c.md)**

:   提出 SHuBERT（Sign Hidden-Unit BERT），将语音自监督学习模型 HuBERT 的 masked cluster prediction 范式迁移到手语视频——对手部、面部、身体姿态四个流分别聚类并同时预测 masked 帧的聚类标签，在约 984 小时 ASL 视频上预训练后，在翻译/孤立识别/指拼检测多任务上达到公开数据 SOTA。

**[WhiSPA: Semantically and Psychologically Aligned Whisper with Self-Supervised Contrastive and Student-Teacher Learning](self_supervised/whispa_semantically_and_psychologically_aligned_whisper_with_self-supervised_con.md)**

:   提出 WhiSPA，通过对比学习将 Whisper 音频编码器的潜在空间与 SBERT 语义表征和心理学维度（情感、人格）对齐，消除语音处理中对额外文本 LM 的依赖，在心理学评估任务上误差降低 73-84%。

---

## 🧬 计算生物 { #computational_biology }

**[Align-Pro: Align Protein Representations Through Multi-Modal Learning](computational_biology/align-pro_align_protein_representations_through_multi-modal_learning.md)**

:   Align-Pro通过多模态对比学习框架，将蛋白质的序列、结构和功能描述三种模态的表示对齐到统一的嵌入空间中，从而实现跨模态的蛋白质检索、分类和功能预测。

**[Concept Bottleneck Language Models For Protein Design](computational_biology/concept_bottleneck_language_models_for_protein_design.md)**

:   本文将概念瓶颈模型（Concept Bottleneck Model）的可解释性设计理念引入蛋白质语言模型，通过中间层的生物学概念作为瓶颈，实现既能设计功能性蛋白质序列又能提供人类可理解的设计理由的蛋白质生成系统。

**[A Survey on Foundation Language Models for Single-cell Biology](computational_biology/foundation_lm_single_cell_survey.md)**

:   首篇从语言建模视角系统综述单细胞生物学基础语言模型，将现有工作划分为PLM（从头预训练）和LLM（利用已有大模型）两大类，全面分析tokenization策略、预训练/微调范式以及下游任务体系，并指出当前领域在数据质量、统一评测和scaling law方面的核心挑战。

**[Enhancing Safe and Controllable Protein Generation via Knowledge Preference Optimization](computational_biology/kpo_protein_safety.md)**

:   提出KPO框架，通过构建蛋白质安全知识图谱(PSKG)并结合加权图剪枝策略识别"相似但安全"的蛋白质对，用DPO微调蛋白质语言模型使其远离有害序列空间，同时保持功能性。

**[Retrieve to Explain: Evidence-driven Predictions for Explainable Drug Target Identification](computational_biology/retrieve_to_explain_drug_target_identification.md)**

:   提出 R2E (Retrieve to Explain)，一种基于检索的架构，通过从文献语料库中检索证据来评分和排序所有候选答案，并利用 Shapley 值将预测忠实地归因到支撑证据，在药物靶点识别任务上超越了遗传学基线和 GPT-4 基线。

---

## ✂️ 语义分割 { #segmentation }

**[BERT-like Models for Slavic Morpheme Segmentation](segmentation/bert-like_models_for_slavic_morpheme_segmentation.md)**

:   本文探索使用 BERT 类预训练语言模型来完成斯拉夫语系语言的形态素分割任务，通过将形态素分割建模为序列标注问题，在多个斯拉夫语言上取得了优于传统方法的结果。

**[DEF-DTS: Deductive Reasoning for Open-domain Dialogue Topic Segmentation](segmentation/def-dts_deductive_reasoning_for_open-domain_dialogue_topic_segmentation.md)**

:   提出 DEF-DTS，一种基于 LLM 多步演绎推理的对话话题分割方法——通过双向上下文摘要 → 话语意图分类（5 类） → 演绎话题转移判断三步 pipeline，在 TIAGE、SuperDialseg、Dialseg711 三个数据集上取得无监督/prompt 方法 SOTA，在 Dialseg711 上超越监督方法。

**[InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning](segmentation/instructpart_task-oriented_part_segmentation_with_instruction_reasoning.md)**

:   提出 InstructPart，首个将任务导向指令与部件级分割结合的真实世界 benchmark——2400 张图像、48 类物体、44 类部件、9600 条人工标注的任务指令，评估发现当前 VLM 在指令驱动的部件分割上严重不足，基于 LISA+DINOv2 的 baseline 微调后性能提升约 100%。

**[Pixel-Level Reasoning Segmentation via Multi-turn Conversations](segmentation/pixel-level_reasoning_segmentation_via_multi-turn_conversations.md)**

:   提出像素级推理分割 (Pixel-level RS) 新任务，通过多轮对话逐步理解用户意图实现细粒度分割，构建了包含 24k 对话轮次的 PRIST 数据集，并设计 MIRAS 框架在分割精度和推理能力上均超越现有基线。

---

## 🖼️ 图像恢复 { #image_restoration }

**[A Self-Denoising Model for Robust Few-Shot Relation Extraction](image_restoration/a_self-denoising_model_for_robust_few-shot_relation_extraction.md)**

:   本文针对少样本关系抽取中支持集标签噪声问题，提出自去噪模型（SDM），通过标签校正模块和关系分类模块的协同训练，自动修正噪声标签并实现更鲁棒的关系预测，即使在无噪声场景下也显著超越基线。

**[DiffuseDef: Improved Robustness to Adversarial Attacks via Iterative Denoising](image_restoration/diffusedef_adversarial_defense.md)**

:   DiffuseDef 在编码器与分类器之间插入一个扩散去噪层，训练时学习预测隐状态噪声，推理时对隐表示加噪→迭代去噪→集成平均，以即插即用的方式大幅提升文本分类模型在黑盒和白盒对抗攻击下的鲁棒性。

**[PreP-OCR: A Complete Pipeline for Document Image Restoration and Enhanced OCR Accuracy](image_restoration/prep-ocr_a_complete_pipeline_for_document_image_restoration_and_enhanced_ocr_acc.md)**

:   提出 PreP-OCR 两阶段流水线：先用合成退化数据训练的 ResShift 模型修复历史文档图像（多方向 patch 提取+中值融合），再用 ByT5 做 OCR 后语义纠错，在 13,831 页真实历史文档上降低 CER 63.9-70.3%。

---

## 🧑 人体理解 { #human_understanding }

**[Beyond Surface Simplicity: Revealing Hidden Reasoning Attributes for Precise Commonsense Diagnosis](human_understanding/beyond_surface_simplicity_revealing_hidden_reasoning_attributes_for_precise_comm.md)**

:   本文揭示了常识推理基准中表面简单但实际隐含复杂推理属性的问题，提出了一种基于隐藏推理属性的细粒度诊断框架，能够更精确地分析和评估模型的常识推理能力。

**[TransBench: Breaking Barriers for Transferable Graphical User Interface Agents in Dynamic Digital Environments](human_understanding/transbench_breaking_barriers_for_transferable_graphical_user_interface_agents_in.md)**

:   提出首个系统评估 GUI Agent **迁移性**（跨版本/跨平台/跨应用）的 benchmark TransBench，涵盖 81 个中文 App、1459 张截图、22K+ 标注指令，实验表明在旧版本上微调可有效迁移到新版本和其他平台，而跨平台迁移中 Android 数据的泛化性最强。

---

## 🎯 目标检测 { #object_detection }

**[Anchored Answers: Unravelling Positional Bias in GPT-2's Multiple-Choice Questions](object_detection/anchored_answers_unravelling_positional_bias_in_gpt-2s_multiple-choice_questions.md)**

:   首次从失败案例角度对GPT-2系列在MCQ中的"锚定偏差"（始终选A）进行机械分析，通过Logit Lens定位到MLP中存储"A"偏好的特定值向量，用极简干预（更新值向量）将MCQ准确率平均提升70%+。

**[Weed Out, Then Harvest: Dual Low-Rank Adaptation is an Effective Noisy Label Detector for Noise-Robust Learning](object_detection/weed_out_then_harvest_dual_low-rank_adaptation_is_an_effective_noisy_label_detec.md)**

:   提出Delora框架，通过引入clean LoRA和noisy LoRA双模块构建噪声标签检测器，将样本选择与模型训练解耦，打破传统"小损失"方法中样本选择与训练互相影响的恶性循环。

---

## 🎬 视频生成 { #video_generation }

**[Q2E: Query-to-Event Decomposition for Zero-Shot Multilingual Text-to-Video Retrieval](video_generation/q2e_query-to-event_decomposition_for_zero-shot_multilingual_text-to-video_retrie.md)**

:   Q2E 提出了一种零样本的查询到事件分解方法，利用 LLM 和 VLM 的参数化世界知识将简单查询分解为前因/当前/后果事件，并结合视频的视觉描述和语音转录，通过逆熵融合排序实现 SOTA 的多语言文本到视频检索性能。

**[VidCapBench: A Comprehensive Benchmark of Video Captioning for Controllable Text-to-Video Generation](video_generation/vidcapbench_a_comprehensive_benchmark_of_video_captioning_for_controllable_text-.md)**

:   提出 VidCapBench，首个专为可控文生视频（T2V）设计的视频描述评估 benchmark，从美学/内容/运动/物理规律四个维度评估 caption 质量，643 个视频+10,644 个 QA 对，实验证明 VidCapBench 分数与 T2V 生成质量高度正相关。

---

## 🧊 3D 视觉 { #3d_vision }

**[Slamming: Training a Speech Language Model on One GPU in a Day](3d_vision/slamming_training_a_speech_language_model_on_one_gpu_in_a_day.md)**

:   提出 Slam 训练配方，通过系统化的模型初始化、架构选择、合成数据、偏好优化等环节优化，在单张 A5000 GPU 上 24 小时内训练出性能媲美大规模 SLM 的语音语言模型。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[Embracing Large Language Models in Traffic Flow Forecasting](autonomous_driving/embracing_large_language_models_in_traffic_flow_forecasting.md)**

:   提出 LEAF 框架，用图分支（pair-wise关系）和超图分支（non-pair-wise关系）的双分支预测器生成候选预测，再用冻结的 LLM 作为选择器（判别而非生成）挑选最优预测，通过 ranking loss 反馈优化预测器，在 PEMS 数据集上取得 SOTA。

---

## 📐 优化/理论 { #optimization }

**[ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting](optimization/scalebio_bilevel_data_reweighting.md)**

:   ScaleBiO 提出基于罚函数重构的全一阶双层优化算法，首次将双层优化应用于 30B+ 参数 LLM 的数据源重加权，在 Qwen-2.5-32B 上实现 GSM8K +9%、MATH +5.8% 的提升。

---

## 📡 信号/通信 { #signal_comm }

**[WirelessMathBench: A Mathematical Modeling Benchmark for LLMs in Wireless Communications](signal_comm/wirelessmathbench_a_mathematical_modeling_benchmark_for_llms_in_wireless_communi.md)**

:   本文提出WirelessMathBench，一个包含587道题目的无线通信数学建模基准，从40篇前沿论文中提取，系统评估LLM在领域特定数学推导上的能力，揭示即使最强的DeepSeek-R1平均准确率也仅38.05%，完整公式推导仅7.83%。

---

## 📂 其他 { #others }

**[Towards Robust ESG Analysis Against Greenwashing Risks: A3CG](others/a3cg_esg_greenwashing.md)**

:   提出 A3CG 数据集和方面-行动分析任务（从可持续性声明中提取方面及其行动类型：已实施/计划中/不确定），通过跨类别泛化设置评估 NLP 方法抵御漂绿风险的鲁棒性，发现监督学习（GRACE F1=47.51）优于 LLM（Claude 3.5 F1=42.03）但泛化效率更差。

**[A Dual-Mind Framework for Strategic and Expressive Negotiation Agent](others/a_dual-mind_framework_for_strategic_and_expressive_negotiation_agent.md)**

:   本文受人类认知双过程理论启发，提出双心智谈判智能体框架（DMNA），将直觉模块（快速策略规划，基于 MCTS+DPO 训练）和审慎模块（慢速表达优化，基于多面反思机制）相结合，在谈判任务上取得了最优性能。

**[Barec: A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment](others/a_large_and_balanced_corpus_for_fine-grained_arabic_readability_assessment.md)**

:   构建 Barec——首个大规模、平衡、细粒度的阿拉伯语可读性评估语料库（69K+ 句子、100 万+词、19 个等级），由 6 名专业教育者标注，并基准测试了 4 种阿拉伯语 BERT 模型 × 4 种输入变体 × 5 种损失函数，发现形态学分词输入 D3Tok 配合回归损失在 QWK 上达到 84.0%。

**[A Little Human Data Goes A Long Way](others/a_little_human_data_goes_a_long_way.md)**

:   通过在8个事实验证和问答数据集上的大规模实验，证明了在合成数据中混入极少量人工标注数据（甚至仅125个样本）就能显著提升模型性能，替换最后10%的人工数据会导致性能严重下降，且200条人工数据的增益需要数量级更多的合成数据才能匹配。

**[A Measure of the System Dependence of Automated Metrics](others/a_measure_of_the_system_dependence_of_automated_metrics.md)**

:   揭示 MT 自动评估指标 "尺子因被测物不同而改变长度" 的系统依赖性问题，提出 SysDep 度量来量化不同翻译系统被指标高估/低估的程度。

**[A Multi-Persona Framework for Argument Quality Assessment](others/a_multi-persona_framework_for_argument_quality_assessment.md)**

:   本文提出 MPAQ 框架，通过大语言模型模拟多个不同评估者视角（persona），对论点进行多角度质量评估，并设计粗到细的评分策略（先整数再小数），在 IBM-Rank-30k 和 IBM-ArgQ-5.3kArgs 数据集上显著超越现有基线，同时提供了可解释的多视角评估理由。

**[A New Formulation of Zipf's Meaning-Frequency Law through Contextual Diversity](others/a_new_formulation_of_zipfs_meaning-frequency_law_through_contextual_diversity.md)**

:   本文提出将 Zipf 语义-频率定律重新形式化为词频与上下文多样性之间的幂律关系，通过语言模型生成的上下文化词向量方向来量化词义数量，发现该定律在小规模语言模型中不可观测，且自回归 LM 需要比掩码 LM 多得多的参数才能呈现该定律。

**[A Practical Approach for Building Production-Grade Conversational Agents with Workflow Graphs](others/a_practical_approach_for_building_production-grade_conversational_agents_with_wo.md)**

:   提出基于有向无环图(DAG)的工作流框架，通过将LLM agent的复杂业务约束分解到图中不同状态节点，并结合响应掩码微调策略，构建满足生产级要求的电商对话代理，在任务准确率和格式遵循方面均大幅超越GPT-4o基线。

**[SSUF: A Semi-supervised Scalable Unified Framework for E-commerce Query Classification](others/a_semi-supervised_scalable_unified_framework_for_e-commerce_query_classification.md)**

:   提出电商查询分类统一框架 SSUF，通过三个可插拔模块——标签增强（BERT 语义编码标签）、知识增强（LLM 世界知识 + 后验点击 + 半监督标签生成）、结构增强（共现/语义/层级三图融合 GCN）——解决短查询信息不足和"马太效应"恶性循环问题，在 JD.COM 意图分类和品类分类任务上 Macro F1 分别达到 49.46 和 41.22（均超 SMGCN 等 SOTA），已上线服务带来显著商业价值。

**[A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior](others/a_spatio-temporal_point_process_for_fine-grained_modeling_of_reading_behavior.md)**

:   本文提出基于时空标记点过程（marked spatio-temporal point process）的阅读行为概率模型，联合建模注视何时发生、落在哪里、持续多久，发现 Hawkes 过程配合读者特定效应和方向偏移能显著提升扫视预测，但 surprisal 等语言学预测因子仅带来微弱改善——这对现有 surprisal 理论提出了质疑。

**[AceCoder: Acing Coder RL via Automated Test-Case Synthesis](others/acecoder_acing_coder_rl_via_automated.md)**

:   构建 AceCode-87K（87K 编码题 + 138 万自动合成测试用例），训练代码专用 Reward Model（7B 超越 340B Nemotron），Best-of-N 提升 Llama-3.1-8B 平均 8.9 分，R1 风格从 base 直接 RL 仅 80 步 HumanEval+ 提升 22.5%。

**[Achieving Certification-by-Design Through Model-Driven Development](others/achieving_certification-by-design_through_model-driven_development.md)**

:   本文提出了一种"认证驱动设计"（Certification-by-Design）的模型驱动开发方法，通过在 NLP 系统的设计阶段就嵌入安全认证要求，使得最终系统能够自动满足相关行业标准和法规，减少事后认证的高昂成本。

**[ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Clause Retrieval](others/acord_an_expert-annotated_retrieval_dataset_for_legal_contract_drafting.md)**

:   构建首个面向合同起草的专家标注条款检索基准ACORD（114查询、126K+对、1-5星评分），评估20种检索方法发现BM25+GPT-4o pointwise重排序最优（NDCG@5=76.9%），但高质量条款精度极低（5星precision@5仅17.2%），揭示模型距真实律师需求的巨大差距。

**[ACT: Knowledgeable Agents to Design and Perform Complex Tasks](others/act_knowledgeable_agents_to_design_and_perform_complex_tasks.md)**

:   本文提出 ACT 框架，让多个 LLM 智能体通过协作设计任务并相互获取结构化知识描述，使每个智能体不仅理解自身任务还了解他人的工作方式，从而在创意写作和工具使用等复杂任务上显著超越现有多智能体方法。

**[Adaptive Feature-based Low Rank Plus Sparse Decomposition for Subspace Clustering](others/adaptive_feature-based_low_rank_plus_sparse_decomposition_for_subspace_clusterin.md)**

:   本文提出一种自适应特征驱动的低秩加稀疏矩阵分解方法，通过在特征空间中自适应学习低秩和稀疏成分的权重，解决子空间聚类中噪声鲁棒性和特征判别力不足的问题。

**[Adaptive Retrieval without Self-Knowledge? Bringing Uncertainty Back Home](others/adaptive_retrieval_without_self-knowledge_bringing_uncertainty_back_home.md)**

:   对 35 种自适应检索方法（含 8 种最新方法和 27 种不确定性估计方法）进行了全面评测，发现经典的不确定性估计技术在效率和自知能力方面往往优于复杂的专用流水线，同时保持相当的 QA 性能。

**[Advancing Sequential Numerical Prediction in Autoregressive Models](others/advancing_sequential_numerical_prediction_in_autoregressive_models.md)**

:   提出Numerical Token Integrity Loss (NTIL)——一种双层级数值预测损失函数，在token级别用指数位置加权的EMD替代交叉熵以保持数值有序性，在序列级别通过可微数值构造进行整体数值偏差惩罚，在目标检测、文字检测、数学推理和时钟识别等任务上显著提升自回归模型的数值预测精度。

**[AIDE: Attribute-Guided Multi-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](others/aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)**

:   提出AIDE框架，通过"属性引导+Persona增强+残差连接"的多跳数据扩展机制，从仅10个种子样本生成约3K条高质量任务特定训练数据，微调Mistral-7B后在zero-shot上平均超越人工标注数据微调6%、超越Evol-Instruct等SOTA方法30%。

**[ALGEN: Few-Shot Inversion Attacks on Textual Embeddings via Cross-Model Alignment](others/algen_few-shot_inversion_attacks_on_textual_embeddings_via_cross-model_alignment.md)**

:   本文提出ALGEN，一种少样本文本嵌入反转攻击方法，通过将受害者的嵌入空间与攻击者的嵌入空间进行线性对齐，再利用训练好的嵌入到文本生成器重建原始文本，仅需1个泄露样本即可发起部分成功的攻击，1000个样本时Rouge-L达45.75。

**[Aligned but Blind: Alignment Increases Implicit Bias by Reducing Awareness of Race](others/aligned_but_blind_implicit_bias.md)**

:   揭示对齐训练的"种族盲视"副作用：对齐使 LLM 在歧义上下文中不再将 black/white 表征为种族概念，安全护栏因此无法激活，导致隐式偏见从 64.1% 飙升至 91.4%；反直觉地，在早期层注入种族感知激活（而非遗忘）可将隐式偏见从 97.3% 降至 42.4%。

**[AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment](others/ambik_dataset_of_ambiguous_tasks_in_kitchen_environment.md)**

:   提出 AmbiK，一个专门用于厨房环境中歧义指令检测的纯文本数据集，包含 1000 对歧义/非歧义指令，按三种歧义类型（用户偏好/常识/安全）分类标注，并评估了多种基于 conformal prediction 的歧义检测方法，发现现有方法在该基准上表现很差。

**[An Analysis of Datasets, Metrics and Models in Keyphrase Generation](others/an_analysis_of_datasets_metrics_and_models_in_keyphrase_generation.md)**

:   对关键短语生成（keyphrase generation）领域50+篇论文进行系统性分析，揭示了基准数据集高度相似、评估指标计算不一致导致性能被高估等关键问题，并发布了一个强力PLM-based模型以促进未来研究。

**[AnalyticKWS: Towards Exemplar-Free Analytic Class Incremental Learning for Small-footprint Keyword Spotting](others/analytickws_towards_exemplar-free_analytic_class_incremental_learning_for_small-.md)**

:   提出 AnalyticKWS，一种无需存储历史样本的关键词检测增量学习方法，通过冻结特征提取器 + 递归最小二乘解析解更新分类器，在 GSC 和 SC-100 数据集上超过了所有基于样本回放的方法，且训练时间和内存开销极低。

**[Anything Goes? A Crosslinguistic Study of (Im)possible Language Learning in LMs](others/anything_goes_a_crosslinguistic_study_of_impossible_language_learning_in_lms.md)**

:   在12种语言上训练GPT-2 small，系统性测试语言模型是否能区分可能语言(自然语言)与不可能语言(打乱词序等)，发现LM展现出部分类人的学习偏向但并非完美——能在单语言内区分但无法跨语言完全分离，而名词短语词序实验中泛化测试(而非困惑度)能反映类型学偏好。

**[Are Any-to-Any Models More Consistent Across Modality Transfers Than Specialists?](others/are_any-to-any_models_more_consistent_across_modality_transfers_than_specialists.md)**

:   本文提出 ACON 数据集和三种一致性评估标准（循环一致性、前向等变性、共轭等变性），发现当前 any-to-any 模型在逐点评估中并不比专用模型组合更具跨模态一致性，但通过多编辑操作的分布式分析可以观察到弱一致性。

**[Are Bias Evaluation Methods Biased?](others/are_bias_evaluation_methods_biased.md)**

:   严格控制变量后比较三种主流偏见评估方法（结构化问答 BBQ、LLM-as-a-Judge、情感分析），发现不同方法对同一组 LLM 产生显著不同的偏见排名——偏见评估方法本身就是有偏的，企业不应依赖单一偏见基准来选择模型。

**[ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search](others/arise_risk_adaptive_search.md)**

:   提出 ARise 框架，将贝叶斯风险评估与动态 RAG 集成到蒙特卡洛树搜索中，解决知识增强推理中的错误传播和验证瓶颈问题，在多跳QA任务上平均准确率超 SOTA KAR 方法 23.10%，超 RAG-equipped 推理模型（DeepSeek-R1）25.37%。

**[ASPERA: A Simulated Environment to Evaluate Planning for Complex Action Execution](others/aspera_a_simulated_environment_to_evaluate_planning_for_complex_action_execution.md)**

:   提出 ASPERA 框架及 Asper-Bench 基准，用于评估 LLM 在自定义助手库约束下执行复杂多步动作规划（程序生成）的能力，揭示了相比自由代码生成，基于自定义 API 库的程序生成对 LLM 构成显著更大的挑战。

**[Attention Entropy is a Key Factor for Parallel Context Encoding](others/attention_entropy_parallel_encoding.md)**

:   发现并行上下文编码导致 query token 的注意力熵异常升高是性能下降的关键因素，并提出 Attention Sink 共享前缀和 Selective Attention 两种免微调方法有效缓解该问题。

**[Autalic: A Dataset for Anti-Autistic Ableist Language In Context](others/autalic_a_dataset_for_anti-autistic_ableist_language_in_context.md)**

:   提出 Autalic——首个专注于上下文中反自闭症残障歧视语言检测的数据集，包含 2,400 条 Reddit 句子及上下文标注，由神经多样性背景的专家标注，实验揭示当前 LLM（包括 DeepSeek、Llama3、Gemma2、Mistral）在识别反自闭症歧视语言时与人类判断严重不一致（平均 Cohen's Kappa 仅 0.091），凸显该任务的困难性。

**[AutoMixer: Checkpoint Artifacts as Automatic Data Mixers](others/automixer_checkpoint_artifacts_as_automatic_data_mixers.md)**

:   提出 AutoMixer 框架，利用训练过程中保存的检查点模型作为"数据混合器"，通过聚合多检查点的一阶影响函数近似来重新分组和加权训练数据，在八个推理基准上取得最高 1.93% 的性能提升。

**[Balancing the Budget: Understanding Trade-offs Between Supervised and Preference-Based Finetuning](others/balancing_the_budget_understanding_trade-offs_between_supervised_and_preference-.md)**

:   系统研究了在固定数据标注预算下，如何在监督微调（SFT）和偏好微调（PFT/DPO）两个阶段之间最优分配资源，发现低数据量时纯 SFT 最优，高预算时组合使用效果最佳，且仅将 <10% 预算分配给 SFT 就能解决 DPO 的冷启动问题并带来 15-20% 的数学推理提升。

**[Battling against Tough Resister: Strategy Planning with Adversarial Game for Non-collaborative Dialogues](others/battling_against_tough_resister_strategy_planning_with_adversarial_game_for_non-.md)**

:   本文提出一种基于对抗博弈的策略规划框架，用于处理非合作对话（如说服、谈判）中面对顽固对手时的策略选择问题，通过建模对话双方的对抗动态来生成更有效的说服策略。

**[Behavioural vs. Representational Systematicity in End-to-End Models: An Opinionated Survey](others/behavioural_vs_representational_systematicity_in_end-to-end_models_an_opinionate.md)**

:   这篇观点性综述区分了行为系统性（模型能否正确泛化到新组合）和表征系统性（模型内部表征是否结构化），用 Hadley 的弱/准/强三级分类审视了语言和视觉领域的主流基准，发现大多数现有基准仅测试弱或准系统性，并呼吁通过机械可解释性方法弥补行为与表征评估的鸿沟。

**[Better Embeddings with Coupled Adam](others/better_embeddings_with_coupled_adam.md)**

:   从理论上证明 Adam 优化器的逐 token 二阶矩是导致 LLM 词嵌入各向异性（均值偏移）的根因，提出 Coupled Adam——对嵌入层的二阶矩取词汇平均——消除了各向异性问题，并在大规模实验中提升了嵌入质量和下游性能。

**[Beyond Position: the emergence of wavelet-like properties in Transformers](others/beyond_position_the_emergence_of_wavelet-like_properties_in_transformers.md)**

:   通过频率分析和小波分解，揭示了使用 RoPE 位置编码的 Transformer 模型中注意力头自发涌现出类小波（wavelet-like）的多分辨率处理特性，以弥补 RoPE 在位置精度和频率分辨率之间的固有权衡。

**[BIG-Bench Extra Hard](others/big-bench_extra_hard.md)**

:   为应对 BIG-Bench Hard 被前沿模型饱和的问题，Google DeepMind 推出 BIG-Bench Extra Hard (BBEH)，用 23 个更难的任务替换 BBH 中的对应任务，最强通用模型仅达 9.8%（调和平均）、最强推理模型达 44.8%，揭示了 LLM 在通用推理上的巨大差距。

**[Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation](others/bone_soups_multi_objective_gen.md)**

:   提出 Bone Soup 模型合并方法，通过先构造"骨架奖励"（多目标奖励的组合）训练骨架模型、再用对称循环矩阵映射确定合并系数，解决了 Rewarded Soup 中单目标模型合并的次优性问题，在三个多目标生成任务上实现更好的 Pareto 前沿和可控性。

**[Bregman Conditional Random Fields: Sequence Labeling with Parallelizable Inference](others/bregman_conditional_random_fields_sequence_labeling_with_parallelizable_inferenc.md)**

:   提出 Bregman CRF (Bcrf)，一种基于均值正则化（mean regularization）的新型序列标注判别模型，使用迭代 Bregman 投影实现可并行化的推理算法，替代传统 CRF 中固有顺序的 Viterbi/Forward 算法，在 POS/NER/分词任务上性能与标准 CRF 持平但更快，且在有禁止标签转移约束的场景下优于 Mean Field 方法。

**[Building Better: Avoiding Pitfalls in Developing Language Resources when Data is Scarce](others/building_better_avoiding_pitfalls_in_developing_language_resources_when_data_is_.md)**

:   通过对 81 名低资源语言 NLP 研究者和标注者的调查，揭示了低资源语言数据构建中的质量问题（数据不自然、文化失当）和伦理问题（标注者劳动被剥削、署名不公），并提出六条改进建议。

**[Byte Latent Transformer: Patches Scale Better Than Tokens](others/byte_latent_transformer.md)**

:   提出 Byte Latent Transformer (BLT)，一种无分词器的字节级 LLM 架构，通过基于熵的动态分组将字节聚合为可变长度 patch，首次在 8B 规模上匹配 token-based 模型性能，同时解锁了通过同时增大 patch 和模型尺寸来提升推理效率的新 scaling 维度。

**[CADReview: Automatically Reviewing CAD Programs with Error Detection and Correction](others/cadreview_automatically_reviewing_cad_programs_with_error_detection_and_correcti.md)**

:   提出 CAD 程序审查任务及 ReCAD 框架，基于参考图像自动检测 CAD 程序中的错误并生成修正反馈，构建了包含 20K+ 样本（8 类错误）的 CADReview 数据集。

**[Can Third-parties Read Our Emotions?](others/can_third_parties_read_our_emotions.md)**

:   本文通过人类被试实验，系统比较了第三方标注者（人类标注者和LLM）与第一方（作者自标注）在情感识别任务中的对齐程度，发现第三方标注与作者真实情感之间存在显著差距，LLM虽优于人类标注者，但仍表现不佳；人口统计学相似性可提升标注质量。

**[Can Uniform Meaning Representation Help GPT-4 Translate from Indigenous Languages?](others/can_uniform_meaning_representation_help_gpt-4_translate_from_indigenous_language.md)**

:   探索将统一意义表示（UMR）语义图纳入 GPT-4 提示中，翻译三种原住民语言（纳瓦霍语、阿拉帕霍语、库卡马语），发现在大多数情况下 UMR 的加入带来统计显著的性能提升。

**[Capacity Matters: A Proof-of-Concept for Transformer Memorization on Real-World Data](others/capacity_matters_a_proof-of-concept_for_transformer_memorization_on_real-world_d.md)**

:   本文以SNOMED医学知识图谱为数据源，系统研究了decoder-only Transformer在结构化数据上的记忆容量，发现嵌入维度是决定学习速度和容量的主要因素，而增加层数收效甚微，Softmax激活函数表现最稳定。

**[Causal Estimation of Tokenisation Bias](others/causal_tokenisation_bias.md)**

:   本文首次将 tokeniser 选择对语言模型输出的影响定义为"分词偏差"(tokenisation bias)，并利用因果推断中的断点回归设计(RDD)来量化这一效应——发现当一个 subword 被纳入词表时，其对应字符串的概率最高可提升 17 倍（小模型），揭示分词是语言建模中一个被低估的关键设计选择。

**[Cautious Next Token Prediction](others/cautious_next_token_prediction.md)**

:   提出 Cautious Next Token Prediction (CNTP)，一种无需训练的自适应解码策略：在模型预测熵较高（不确定）时采样多条候选路径至标点处，选择困惑度最低的路径作为最终续写，从而在不牺牲多样性的前提下显著提升准确性。

**[ChartLens: Fine-Grained Visual Attribution in Charts](others/chartlens_fine-grained_visual_attribution_in_charts.md)**

:   提出图表的事后细粒度视觉归因（Post-Hoc Fine-grained Visual Attribution）任务，设计 ChartLens 算法利用分割技术标记图表元素并通过 Set-of-Marks 提示多模态 LLM 进行精确归因，同时构建 ChartVA-Eval 基准，在三类图表上取得 26-66% 的 F1 提升。

**[ChildMandarin: A Comprehensive Mandarin Speech Dataset for Young Children Aged 3-5](others/childmandarin_a_comprehensive_mandarin_speech_dataset_for_young_children_aged_3-.md)**

:   提出 ChildMandarin，一个面向 3-5 岁幼儿的普通话语音数据集，包含 397 名说话人、41.25 小时语音、覆盖中国 22 个省级行政区，并在 ASR 和说话人验证任务上提供了全面的基线评估。

**[ChuLo: Chunk-Level Key Information Representation for Long Document Understanding](others/chulo_chunk-level_key_information_representation_for_long_document_understanding.md)**

:   ChuLo 的核心不是单纯把长文档切小，而是先在全文范围内找出最关键的语义短语，再把这些关键信息重新注入每个 chunk 的表示里，从而在只用紧凑块表示的前提下，同时保住全局语义和细粒度 token 信息。

**[CiteEval: Principle-Driven Citation Evaluation for Source Attribution](others/citeeval_principle-driven_citation_evaluation_for_source_attribution.md)**

:   本文提出 CiteEval，一个基于原则驱动的引用评估框架，通过考虑完整检索上下文、超越检索的多种上下文以及细粒度评价标准，构建了 CiteBench 基准和 CiteEval-Auto 自动指标，在引用质量评估上显著优于基于 NLI 的现有方法。

**[CLaC at SemEval-2025 Task 6: A Multi-Architecture Approach for Corporate Environmental Promise Verification](others/clac_at_semeval-2025_task_6_a_multi-architecture_approach_for_corporate_environm.md)**

:   本文针对SemEval-2025 Task 6（PromiseEval）的企业ESG报告承诺验证任务，探索了三种递进的模型架构：ESG-BERT基线、语言特征增强版、以及融合注意力池化和多目标学习的联合子任务模型，最终以0.5268的私榜分数略超基线（0.5227），验证了语言特征工程和多任务学习在ESG承诺验证中的有效性。

**[CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model](others/coachme_sport_instruction.md)**

:   提出 CoachMe，通过对比学习者动作与参考动作的差异（时间+物理两个维度），自动生成运动特异性的教练指导文本，在花样滑冰和拳击上分别超过 GPT-4o 31.6% 和 58.3%（G-Eval）。

**[CoAM: Corpus of All-Type Multiword Expressions](others/coam_corpus_of_all-type_multiword_expressions.md)**

:   构建了一个高质量的全类型多词表达(MWE)识别数据集 CoAM（1.3K句），通过多步质量保障流程解决了现有数据集标注不一致的问题，并发现微调大语言模型在 MWE 识别任务上显著优于此前的 SOTA 方法 MWEasWSD。

**[Code-Switching and Syntax: A Large-Scale Experiment](others/code-switching_and_syntax_a_large-scale_experiment.md)**

:   本文通过大规模、多语言、跨现象的实验，首次系统性验证了"句法信息足以解释语码转换（code-switching）模式"这一语言学共识，仅用句法特征就能达到与双语人类相当的判断准确率，且学到的句法模式可泛化到未见过的语言对。

**[CoLA: Collaborative Low-Rank Adaptation](others/cola_collaborative_low-rank_adaptation.md)**

:   提出 CoLA，一种灵活的 LoRA 架构，打破矩阵 A 和 B 之间的固定数量约束（#A=M, #B=N），并设计三种协作策略（全协作/随机协作/启发式协作），结合扩展的 PiSSA 初始化，在低样本场景下显著优于现有 PEFT 方法。

**[ComfyUI-Copilot: An Intelligent Assistant for Automated Workflow Development](others/comfyui-copilot_an_intelligent_assistant_for_automated_workflow_development.md)**

:   提出 ComfyUI-Copilot，一个基于 LLM 的层次化多 agent 框架，作为 ComfyUI 插件提供智能节点/模型推荐和一键式工作流构建，覆盖 7K 节点、62K 模型和 9K 工作流的知识库，在线服务 22 国的 19K 用户并处理了 85K+ 查询。

**[Commonsense Reasoning in Arab Culture](others/commonsense_arab_culture.md)**

:   提出 ArabCulture 数据集（3482 个 MSA 问题，覆盖 13 个阿拉伯国家/4 个区域/54 个文化子领域），系统评估多个 LLM 的阿拉伯文化常识推理能力，发现即使 GPT-4o 也仅达 90%、大部分模型在 40-80% 之间，揭示了 LLM 在非西方文化理解上的显著不足。

**[Completing A Systematic Review in Hours instead of Months with Interactive AI Agents](others/completing_a_systematic_review_in_hours.md)**

:   提出 InsightAgent，一个以人为中心的交互式多 Agent 系统，通过语义聚类分区、多 agent 并行阅读和实时用户交互，将医学系统综述的撰写时间从数月缩短到约 1.5 小时，达到人类撰写质量的 79.7%。

**[ConceptCarve: Dynamic Realization of Evidence](others/conceptcarve_dynamic_realization_of_evidence.md)**

:   提出 ConceptCarve 框架，通过 LLM 与传统检索器的交互式协作，动态构建概念树来表征证据在特定社区中的实现形式，解决了证据检索中的推理鸿沟和领域敏感性两大挑战。

**[ConECT Dataset: Overcoming Data Scarcity in Context-Aware E-Commerce MT](others/conect_dataset_overcoming_data_scarcity_in_context-aware_e-commerce_mt.md)**

:   构建了 ConECT——首个捷克-波兰电商多模态翻译数据集（11,400 句对 + 产品图片 + 类目路径），通过 VLM 端到端翻译、NMT+类目路径前缀、NMT+图像描述前缀三条技术路线的系统对比，发现结构化类目上下文能稳定提升翻译质量（COMET +0.005），而合成图像描述以级联方式注入反而严重损害翻译性能（COMET 暴跌 0.11+）。

**[CONFETTI: Conversational Function-Calling Evaluation Through Turn-Level Interactions](others/confetti_conversational_function-calling_evaluation_through_turn-level_interacti.md)**

:   CONFETTI 提出了一个面向多轮对话场景的函数调用（function-calling）评测基准，包含 109 段人类模拟对话、313 个用户轮次和 86 个 API，通过 off-policy turn-level 评估和 dialog act 标注系统性地测试 LLM 在复杂对话场景下的工具调用能力，发现即便最强模型（Nova Pro）也仅达 40% 左右，链式调用更是普遍短板。

**[ConSim: Measuring Concept-Based Explanations' Effectiveness with Automated Simulatability](others/consim_measuring_concept-based_explanations_effectiveness_with_automated_simulat.md)**

:   ConSim 提出用 LLM 作为"模拟器"来自动评估概念解释（concept-based explanation）的有效性——通过检测 LLM 能否仅凭解释来预测被解释模型的输出，从而同时衡量概念空间的质量和解释的可理解性，实现了可扩展、一致且全面的解释方法评估。

**[Consistent Client Simulation for Motivational Interviewing-based Counseling](others/consistent_client_simulation_for_motivational_interviewing-based_counseling.md)**

:   提出一种面向动机性访谈（MI）心理咨询的一致性客户模拟框架，通过状态转换、行动选择、信息选择和回复生成四个模块，确保模拟客户的行为与其预设的画像（动机、信念、改变计划、配合度）保持一致，在自动和专家评估中均优于基线方法。

**[Consultant Decoding: Yet Another Synergistic Mechanism](others/consultant_decoding_yet_another_synergistic_mechanism.md)**

:   提出 Consultant Decoding (CD)，一种基于目标模型负对数似然（NLL）验证 draft token 的新型协同解码机制，相比传统 Speculative Decoding 的似然比验证方法，能大幅提升接受率、降低大模型调用频率，同时保持甚至超越目标模型的生成质量。

**[Contextual Experience Replay for Self-Improvement of Language Agents](others/contextual_experience_replay_for_self-improvement_of_language_agents.md)**

:   CER（Contextual Experience Replay）提出了一个无需训练的语言 agent 自我改进框架，通过将过去的交互经验累积并合成为动态记忆缓冲区，在推理时让 agent 检索相关知识来增强新任务的决策，在 WebArena 上相对 GPT-4o baseline 提升了 51.0% 的成功率。

**[Cooperating and Competing Through Natural Language](others/cooperating_and_competing_through_natural_language.md)**

:   本文研究 LLM 智能体在自然语言交互环境中的合作与竞争行为，通过设计多人博弈场景，分析语言策略（如说服、欺骗、协商）如何影响博弈结果，揭示了 LLM 在社会交互中的策略涌现能力和局限性。

**[CORAL: Learning Consistent Representations across Multi-step Training with Lighter Speculative Drafter](others/coral_speculative_drafting.md)**

:   CORAL 通过跨步表示对齐（CSRA）改进多步训练中 draft 模型的特征一致性，并用权重分组机制压缩大词表 LM head 的推理延迟，在 LLaMA3/Qwen2.5 上实现 2.50-4.07× 加速，超越 EAGLE-2 和 HASS。

**[Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the Limits of Embedding Space Capacity](others/cramming_tokens_embedding_capacity.md)**

:   通过逐样本优化方法将文本压缩到可训练的 [mem] 向量中，发现 Llama-3.1-8B 可以将 1568 个 token 无损压缩到单个输入向量中，揭示了现有方法（约 x10 压缩比）与实际可达极限（x1500+）之间存在两个数量级的差距。

**[DAPE V2: Process Attention Score as Feature Map for Length Extrapolation](others/dape_v2_process_attention_score_as_feature_map_for_length_extrapolation.md)**

:   本文将 Transformer 的注意力分数视为特征图，通过在注意力分数上施加卷积操作（而非简单的 key-query 点积）来显著提升 Transformer 在长序列上的长度外推能力，将外推问题转化为经典的图像特征处理问题。

**[Decoding Knowledge Attribution in Mixture-of-Experts: A Framework of Basic-Refinement Collaboration and Efficiency Analysis](others/decoding_knowledge_attribution_in_mixture-of-experts_a_framework_of_basic-refine.md)**

:   提出跨层级知识归因算法，系统解析 MoE 模型中共享专家与路由专家的"基础-精炼"协作框架，揭示 MoE 相比稠密模型实现 31% 更高的逐层效率，并通过语义驱动路由机制（注意力头-专家相关性 r=0.68）和专家阻断实验验证了架构深度对鲁棒性的决定性影响。

**[Decoding Reading Goals from Eye Movements](others/decoding_reading_goals_from_eye_movements.md)**

:   本文首次提出从眼动轨迹中解码读者阅读目标（信息检索 vs. 普通阅读）的任务，通过 12 种模型的系统比较发现基于 Transformer 的扫视路径+语言建模方案（RoBERTa-Eye-F）最优，可在阅读早期即实现高精度实时预测。

**[DeepRTL2: A Versatile Model for RTL-Related Tasks](others/deeprtl2_a_versatile_model_for_rtl-related_tasks.md)**

:   DeepRTL2是首个统一处理RTL（寄存器传输级）相关生成任务与嵌入任务的LLM，通过精心构建的数据集和GRIT训练策略，在代码生成、代码理解、自然语言代码搜索、功能等价检查和性能预测五大任务上达到SOTA。

**[Detecting Sockpuppetry on Wikipedia Using Meta-Learning](others/detecting_sockpuppetry_on_wikipedia_using_meta-learning.md)**

:   本文将元学习（meta-learning）应用于维基百科的恶意马甲账号（sockpuppet）检测任务，通过跨多个马甲组的训练优化模型的快速适应能力，在数据稀缺场景下显著提升了检测精度，并发布了新的马甲调查数据集。

**[Developmentally-plausible Working Memory Shapes a Critical Period for Language Acquisition](others/developmentally-plausible_working_memory_shapes_a_critical_period_for_language_a.md)**

:   受"Less-is-More"假说启发，本文提出 DynamicLimit-Exp 方法，将人类工作记忆在关键期内的指数增长特征集成到语言模型训练中（通过动态调节 ALiBi 斜率），在 Child-Directed Speech 数据上训练的 GPT-2 模型在句法评估中显著优于无记忆约束和静态约束的基线。

**[Digital Gatekeepers: Google's Role in Curating Hashtags and Subreddits](others/digital_gatekeepers_googles_role_in_curating_hashtags_and_subreddits.md)**

:   本文通过对比 Google 搜索引擎返回的 hashtag 和 subreddit 结果与 Reddit 和 Twitter/X 的非采样数据，揭示了 Google 的算法系统性地压制了与色情、阴谋论、广告和加密货币相关的内容，同时推广高参与度内容，从而充当"数字守门人"影响公众话语。

**[Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction](others/distractor_gen_multiple_choice.md)**

:   本文提出了一个通过成对排序器预测学生选择倾向、再利用DPO训练干扰项生成器的三步流水线，使生成的多选题干扰项更具有迷惑性和区分度。

**[Do not Abstain! Identify and Solve the Uncertainty](others/do_not_abstain_identify_and_solve_the_uncertainty.md)**

:   本文提出ConfuseBench基准和基于inquiry answer唯一性判断不确定性来源的方法，并通过InteractDPO在策略训练中动态生成偏好对来提升inquiry质量，使LLM能主动识别并解决不确定性而非简单回避。

**[Dolphin: Moving Towards Closed-loop Auto-research through Thinking, Practice, and Feedback](others/dolphin_moving_towards_closed-loop_auto-research_through_thinking_practice_and_f.md)**

:   > 提出 Dolphin，一个闭环自动科研框架，包含"想法生成→实验验证→结果反馈"三阶段循环，通过任务属性引导的论文排序和异常回溯引导的调试流程，在 3D 分类等任务上自动提出并验证了接近人类设计 SOTA 的方法。

**[DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning](others/domix_an_efficient_framework_for_exploiting.md)**

:   提出 DoMIX，将各领域知识用独立 LoRA 模块存储后通过对角初始化的 bridge 矩阵在微调时灵活组合利用，在持续领域适应预训练场景下减少 58% 预训练时间和 87% GPU 内存，同时性能超越 SOTA。

**[Principled Content Selection to Generate Diverse and Personalized Multi-Document Summaries](others/dpp_diverse_multidoc_summary.md)**

:   提出将多文档摘要解耦为关键点抽取→DPP多样性选择→重写三步流水线，通过行列式点过程（DPP）进行原则性内容选择，显著提升LLM多文档摘要的源文档覆盖率。

**[DREsS: Dataset for Rubric-based Essay Scoring on EFL Writing](others/dress_dataset_rubric_based_essay_scoring_efl_writing.md)**

:   发布DREsS大规模标准化评分准则数据集，包含三个子集（DREsS_New真实课堂数据1.7K + DREsS_Std标准化历史数据集6.5K + DREsS_CASE增强数据40.1K），提出基于腐蚀的作文增强策略CASE，将BERT基线的QWK分数从0.471提升至0.685（提升45.44%）。

**[DRS: Deep Question Reformulation With Structured Output](others/drs_deep_question_reformulation_with_structured_output.md)**

:   提出 DRS（Deep Question Reformulation with Structured Output），一种零样本方法，通过实体驱动的 DFS 搜索 + 结构化输出约束，将 GPT-3.5 的问题重构准确率从 23.03% 提升至 70.42%，使 LLM 能有效帮助用户将不可回答的问题转化为可回答的形式。

**[Divide-Then-Aggregate: An Efficient Tool Learning Method via Parallel Tool Invocation](others/dta_llama_parallel_tool_invocation.md)**

:   提出 DTA-Llama，将传统树搜索的串行工具调用路径转换为有向无环图（DAG）结构实现并行调用，设计 Process/Thread 推理框架使 LLM 在每轮中可分解任务并并行执行多个工具，在 StableToolBench 上使 Llama2-7B 达到 GPT-3.5 Parallel Function Calling 的水平。

**[Efficient OpAmp Adaptation for Zoom Attention to Golden Contexts](others/efficient_opamp_adaptation_for_zoom_attention_to_golden_contexts.md)**

:   受运算放大器（OpAmp）电路启发，提出 OpAmp Adaptation 方法通过 adapter 高效改造预训练 Transformer 的注意力机制，在噪声上下文场景下让 LLM 更精准聚焦于 golden document，Qwen2.5-OpAmp-72B 在多个噪声上下文基准上超越 DeepSeek-V3 和 GPT-4o。

**[Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction](others/enhancing_conversational_agents_with_theory_of_mind_aligning_beliefs_desires_and.md)**

:   本文探索了从开源 LLM（LLaMA）内部表征中提取心智理论（ToM）相关信息的可行性，并利用 BDI（信念-愿望-意图）框架操纵这些表征来生成更符合人类社交认知的对话回复，ToM 对齐后的模型在 3B 和 8B 上分别达到 67% 和 63% 的胜率。

**[Enhancing Transformers for Generalizable First-Order Logical Entailment](others/enhancing_fol_entailment.md)**

:   系统性研究 Transformer 在一阶逻辑蕴涵任务中的泛化推理能力，揭示了查询语法、token 嵌入和 Transformer 架构（特别是位置编码）的影响，并提出 TEGA（Transformer Encoder with Guided Attention）在相对位置编码设定下显著提升逻辑推理性能。

**[Enhancing Marker Scoring Accuracy through Ordinal Confidence Modelling in Educational Assessments](others/enhancing_marker_scoring_accuracy_through_ordinal_confidence_modelling_in_educat.md)**

:   本文提出了一种基于核加权序数分类交叉熵（KWOCCE）的置信度建模方法，通过利用 CEFR 等级的序数结构和分数分箱策略，实现最高 47% 评分在 100% CEFR 一致性下释放，99% 在 ≥95% 一致性下释放，显著优于无置信度过滤时的约 92%。

**[Enhancing the Comprehensibility of Text Explanations via Unsupervised Concept Discovery](others/enhancing_the_comprehensibility_of_text_explanations_via_unsupervised_concept_di.md)**

:   提出 ECO-Concept 框架，通过 slot attention 机制自动提取文本概念，并利用 LLM 作为人类代理评估概念的可理解性，用可理解性反馈损失指导模型微调，在无概念标注的情况下实现了兼具高分类精度和人类可理解性的概念解释。

**[Entailed Between the Lines: Incorporating Implication into NLI](others/entailed_between_the_lines_incorporating_implication_into_nli.md)**

:   形式化定义"隐含蕴涵"（implied entailment）任务，将传统NLI的三分类扩展为四分类（隐式蕴涵/显式蕴涵/中立/矛盾），构建包含10K前提和40K假设的INLI数据集，实验表明微调后的模型能有效识别隐含蕴涵并跨领域泛化。

**[Entity Framing and Role Portrayal in the News](others/entity_framing_and_role_portrayal_in_the_news.md)**

:   本文构建了一个包含 5 种语言、1378 篇新闻文章、5800+ 实体标注的多语言层次化实体框架语料库，提出含 22 种精细角色的叙事角色分类体系（主角 / 反派 / 无辜者三大框架下），并在微调多语言 Transformer 和 LLM 层次零样本学习上建立了基准。

**[Entropy-UID: A Method for Optimizing Information Density](others/entropy-uid_a_method_for_optimizing_information_density.md)**

:   提出 Entropy-UID 方法，在自回归语言模型的解码过程中联合最小化熵和 surprisal 的加权组合，以实现信息密度的均匀分布。在 WikiText-2、OpenWebText 和 WMT 数据集上，该方法实现了最低的熵标准差（≈2.8）和稳定的 surprisal（≈5.7），优于单目标优化策略。

**[EpiCoDe: Boosting Model Performance Beyond Training with Extrapolation and Contrastive Decoding](others/epicode_boosting_model_performance_beyond_training_with_extrapolation_and_contra.md)**

:   提出 EpiCoDe，一种结合模型外推（Model Extrapolation）和对比解码（Contrastive Decoding）的无训练方法，在数据稀缺场景中通过参数空间外推和推理时logit差异对比来提升微调模型性能，并从logit误差角度给出了理论分析框架。

**[EpMAN: Episodic Memory AttentioN for Generalizing to Longer Contexts](others/epman_episodic_memory_attention_for_generalizing_to_longer_contexts.md)**

:   提出 EpMAN 方法，通过情景记忆模块估计上下文块的相对相关性，用该相关性重新加权解码器的自注意力（differentiating attention），配合噪声训练和注意力范围扩展策略，在 16k-256k 上下文长度范围内实现了比长上下文 LLM 和 RAG 更强且更鲁棒的表现。

**[ERU-KG: Efficient Reference-aligned Unsupervised Keyphrase Generation](others/eru-kg_efficient_reference-aligned_unsupervised_keyphrase_generation.md)**

:   ERU-KG 提出了一个由信息量模块和短语性模块组成的无监督关键短语生成框架，通过参考文本（查询、引用上下文、标题）在词项级别学习信息量估计，在关键短语生成基准上超越所有无监督基线并达到有监督模型 89% 的性能，同时推理速度最快。

**[Evaluating Design Decisions for Dual Encoder-based Entity Disambiguation](others/evaluating_design_decisions_for_dual_encoder-based_entity_disambiguation.md)**

:   系统评估了 Dual Encoder 在实体消歧（ED）任务中的关键设计选择（损失函数、相似度度量、标签语义化格式、负采样策略），并基于最优设计构建了 VerbalizED 系统，在 ZELDA 基准上达到了新的 SOTA，同时探索了一种迭代预测策略来利用已消歧的邻居实体改进困难样本。

**[Evaluating the Evaluation of Diversity in Commonsense Generation](others/evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)**

:   对常识生成（GCR）任务中的12种多样性评估指标进行系统元评估，发现基于形式（n-gram）的指标在低质量生成上严重高估多样性，而基于内容（句子嵌入）的指标与人类判断一致性更高，推荐未来 GCR 研究使用 VS-Embed 或 Chamfer Distance 等内容级指标。

**[Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection](others/explaining_matters_leveraging_definitions_and_semantic_expansion_for_sexism_dete.md)**

:   针对在线性别歧视检测中的数据稀疏和细粒度分类歧义问题，提出两种基于prompt的数据增强技术——定义驱动数据增强（DDA）利用类别定义生成语义对齐的合成样本，上下文语义扩展（CSE）通过分析模型错误的语义特征丰富训练数据——并结合 Mistral-7B 回退集成策略，在 EDOS 数据集上实现全任务 SOTA。

**[Explicit and Implicit Data Augmentation for Social Event Detection](others/explicit_and_implicit_data_augmentation_for_social_event_detection.md)**

:   本文提出SED-Aug，一个结合显式（LLM文本增强）和隐式（特征空间扰动）的双重数据增强框架用于社交事件检测，在Twitter2012和Twitter2018上分别超越最优基线17.67%和15.57%的平均F1。

**[Model Extrapolation Expedites Alignment](others/expo_model_extrapolation.md)**

:   基于"对齐训练仅产生微小参数变化"的观察，提出ExPO方法——通过放大SFT→DPO的参数变化方向（$\theta_2 = \theta_1 + \alpha\Delta\theta$），在零额外训练开销下提升对齐性能，使仅训练20%步骤的DPO模型超越完整训练的版本。

**[FastDraft: How to Train Your Draft](others/fastdraft_how_to_train_your_draft.md)**

:   提出 FastDraft，一套高效的 draft 模型预训练与对齐流程，可在24小时内用单节点8卡训练出约50M参数的 draft 模型，配合 Speculative Decoding 实现最高3倍内存带宽加速和2倍实际推理加速。

**[FastMCTS: A Simple Sampling Strategy for Data Synthesis](others/fastmcts_a_simple_sampling_strategy_for_data_synthesis.md)**

:   FastMCTS 提出了一种受 MCTS 启发的轻量级推理数据合成策略，通过自适应停留策略、动态探索和保留模拟三个改进，比拒绝采样多生成 30% 以上的正确推理路径，训练出的模型在多个数学基准上平均提升 3.9%。

**[FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning](others/fcmr_robust_evaluation_of_financial_cross-modal_multi-hop_reasoning.md)**

:   构建了金融领域跨模态多跳推理基准 FCMR，包含文本、表格和图表三种模态，分 Easy/Medium/Hard 三个难度等级，最强模型 Claude 3.5 Sonnet 在 Hard 级别仅达 30.4% 准确率，揭示了 MLLM 在信息检索阶段的关键瓶颈。

**[FEAT: A Preference Feedback Dataset through a Cost-Effective Auto-Generation and Labeling Framework for English AI Tutoring](others/feat_a_preference_feedback_dataset_through_a_cost-effective_auto-generation_and_.md)**

:   提出 FEAT 框架，通过 LLM 自动生成和标注教师反馈偏好数据集用于英语辅导系统，发现仅混入 5-10% 人工标注数据就能超越 100% 人工数据的排序性能。

**[Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients](others/federated_lora_heterogeneous.md)**

:   提出 LoRA-A2 框架，通过交替冻结 LoRA 的 A/B 模块与自适应秩选择策略，同时解决联邦学习中 LoRA 聚合不一致和通信开销大的双重难题。

**[Behind Closed Words: Creating and Investigating the forePLay Annotated Dataset for Polish Erotic Discourse](others/foreplay_polish_erotic_detection.md)**

:   构建了首个波兰语色情内容检测数据集 forePLay（24,768 句，5 类标签），提出涵盖模糊性、暴力和社会不可接受行为的多维标注体系，评估发现专用波兰语模型显著优于多语言模型，且 Transformer 编码器模型在不平衡类别处理上表现最强。

**[FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](others/fractal_fine-grained_scoring_from_aggregate_text_labels.md)**

:   提出 FRACTAL 方法，将回复级别（response-level）的聚合标签分解为句子级别（sentence-level）的伪标签，利用多实例学习（MIL）和标签比例学习（LLP）技术结合先验信息（文档-句子余弦相似度）训练句子级评分模型，覆盖检索、问答、摘要和数学推理四类任务。

**[Frictional Agent Alignment Framework: Slow Down and Don't Break Things](others/frictional_agent_alignment.md)**

:   提出摩擦对齐框架 FAAF（Frictional Agent Alignment Framework），通过双策略（frictive state policy + intervention policy）目标函数，训练 LLM 在协作对话中识别信念冲突并生成促进反思与审议的"摩擦"干预，超越 DPO/IPO/PPO 等对齐方法。

**[From Lists to Emojis: How Format Bias Affects Model Alignment](others/from_lists_to_emojis_how_format_bias_affects_model_alignment.md)**

:   本文系统研究了 RLHF 中偏好模型（包括人类评估者、GPT-4 和开源模型）对粗体、列表、emoji 等格式模式的偏好偏差，展示了不到 1% 的偏差数据即可显著注入偏差，并提出了双头奖励模型的去偏方法。

**[From Real to Synthetic: Synthesizing Millions of Diversified and Complicated User Instructions with Attributed Grounding](others/from_real_to_synthetic_synthesizing_millions_of_diversified_and_complicated_user.md)**

:   本文提出"归因式接地"（Attributed Grounding）框架，通过自顶向下的用户归因和自底向上的基于网络文档的指令合成，构建了一个包含 100 万条多样且复杂指令的 SynthQuestions 数据集，训练模型在多个通用基准上达到领先表现。

**[GA-S3: Comprehensive Social Network Simulation with Group Agents](others/ga-s3_comprehensive_social_network_simulation_with_group_agents.md)**

:   提出基于"群体智能体"（Group Agent）的社交网络模拟系统 GA-S3，将具有相似行为的个体聚合为群体代理，通过层次化生成、马尔可夫网络推理和行为模块实现大规模社交网络的高效精确模拟。

**[GeAR: Generation Augmented Retrieval](others/gear_generation_augmented_retrieval.md)**

:   GeAR 在传统 bi-encoder 检索框架上引入融合编码器和文本解码器，通过生成任务增强检索模型对文档内部细粒度语义的理解能力，同时不增加全局检索的计算开销。

**[Generating Synthetic Relational Tabular Data via Structural Causal Models](others/generating_synthetic_relational_tabular_data_via_structural_causal_models.md)**

:   本文扩展了 TabPFN 的基于结构因果模型（SCM）的合成数据生成方法，提出了一个能够生成多表关联（relational）合成表格数据的框架，通过耦合节点和隐因果关系实现跨表依赖建模。

**[GeNRe: A French Gender-Neutral Rewriting System Using Collective Nouns](others/genre_a_french_gender-neutral_rewriting_system_using_collective_nouns.md)**

:   GeNRe 是首个法语性别中性重写系统，利用集体名词（collective nouns）替代阳性泛指（masculine generics），提出规则系统、微调模型和指令模型三种方案，其中规则系统和 Claude 3 Opus + 词典方案效果最好。

**[GPT-4 as a Homework Tutor can Improve Student Engagement and Learning Outcomes](others/gpt-4_as_a_homework_tutor_can_improve_student_engagement_and_learning_outcomes.md)**

:   在意大利高中进行了为期 8 周的随机对照试验（RCT），用 GPT-4 替代传统英语作业作为互动辅导工具，发现 GPT-4 组学生在参与度（有趣性、资源充分性显著提升）和特定条件下的学习增益（三年级 Cohen's d=0.603）方面有所改善，仅需教师提供作业目标和描述即可实施，幻觉率低于 1%，且所有在校学生均表示希望继续使用。

**[Graph-guided Cross-composition Feature Disentanglement for Compositional Zero-shot Learning](others/graph-guided_cross-composition_feature_disentanglement_for_compositional_zero-sh.md)**

:   DCDA 提出图引导的跨组合特征解耦方案，通过双适配器（L-Adapter 用于文本端 GNN 特征聚合、V-Adapter 用于视觉端跨注意力解耦）注入冻结 CLIP，在组合零样本学习任务上显著超越现有方法。

**[Graph-Structured Trajectory Extraction from Travelogues](others/graph-structured_trajectory_extraction_from_travelogues.md)**

:   提出"访问顺序图"（Visiting Order Graph）来统一表示旅行轨迹中的地理包含层级关系和时序转移关系，构建了覆盖 100 篇日语游记的 ATD-VSO 基准数据集（3354 个地理实体、3369 条关系），并通过基线实验发现地理包含关系预测（F1=0.355）是核心瓶颈，为该领域指明了地理知识融合的关键方向。

**[Graphically Speaking: Unmasking Abuse in Social Media with Conversation Insights](others/graphically_speaking_unmasking_abuse_in_social_media_with_conversation_insights.md)**

:   提出一种基于图注意力网络（GAT）的上下文感知滥用语言检测框架，将 Reddit 对话建模为图结构（节点=评论，边=回复关系），利用基于 Reddit 界面渲染逻辑的 affordance-based 图裁剪策略保留关键上下文，3 层 GAT 模型达到 F1=0.7624，显著优于无上下文基线和扁平化上下文方法，在上下文敏感样本上提升尤为明显（+4.75%）。

**[Guidelines for Fine-grained Sentence-level Arabic Readability Annotation](others/guidelines_for_fine-grained_sentence-level_arabic_readability_annotation.md)**

:   本文提出了 BAREC 语料库及其标注指南，这是一个拥有 69K+ 句子、覆盖 19 个可读性等级的大规模阿拉伯语句子级可读性评估资源，并在此基础上建立了自动可读性评估的基准模型。

**[Hanging in the Balance: Pivotal Moments in Crisis Counseling Conversations](others/hanging_in_the_balance_pivotal_moments_in_crisis_counseling_conversations.md)**

:   本文提出了一种无监督方法来检测对话中的"关键时刻"（pivotal moments）——即下一步回应可能极大影响对话结局的节点，并在危机心理咨询场景中验证了该方法的有效性。

**[Hard Negative Mining for Domain-Specific Retrieval in Enterprise Systems](others/hard_negative_mining_for_domain-specific_retrieval_in_enterprise_systems.md)**

:   本文提出了一种面向企业级领域特定检索的可扩展硬负样本挖掘框架，通过融合多种嵌入模型、PCA 降维和双语义条件筛选来动态选择高质量硬负样本，在内部云服务数据集和公开基准上均取得了显著提升。

**[HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference](others/hata_trainable_and_hardware-efficient_hash-aware_top-k_attention_for_scalable_la.md)**

:   HATA 提出了一种将 learning-to-hash 技术集成到 top-k 注意力机制的方法，通过将查询和键映射为二进制哈希码来获取相对 qk 分数排序（而非绝对分数估计），在保持模型精度的同时实现了相对全注意力最高 7.2 倍的加速。

**[HelpSteer3: Human-Annotated Feedback and Edit Data to Empower Inference-Time Scaling](others/helpsteer3_human-annotated_feedback_and_edit_data_to_empower_inference-time_scal.md)**

:   NVIDIA 发布 HelpSteer3 数据集（7000+标注员、80+国家），训练专用的 Feedback 和 Edit 模型，在推理时通过"初始响应→反馈→编辑"循环实现开放域通用任务的推理时扩展，基于 Llama 3 系列 70B 模型在 Arena Hard 上达到 92.7 分，超越 OpenAI o1-preview (90.4) 和 DeepSeek R1 (92.3)。

**[Hierarchical Attention Generates Better Proofs](others/hierarchical_attention_generates_better_proofs.md)**

:   提出 Hierarchical Attention 正则化方法，通过建立五层语义层次结构来引导 LLM 的注意力机制，使其与数学推理的自然信息流对齐，在 miniF2F 和 ProofNet 上分别提升证明成功率 2.05% 和 1.69%，同时降低证明复杂度 23.81% 和 16.50%。

**[Hierarchical Bracketing Encodings for Dependency Parsing as Tagging](others/hierarchical_bracketing_dep_parsing.md)**

:   提出层次化括号编码家族用于依存句法分析的序列标注范式，证明现有4-bit编码是该家族的非最优特例，推导出仅需12个标签的最优编码，并将其推广到处理任意非投射性。

**[Hierarchical Memory Organization for Wikipedia Generation](others/hierarchical_memory_wikipedia_gen.md)**

:   提出 Memory Organization-based Generation（MOG）框架，从网页文档中提取细粒度记忆单元（factoid），通过递归聚类-摘要算法组织为层次化 Wikipedia 大纲结构，使每个章节都有直接的记忆支撑，在 FreshWiki 和 WikiStart 数据集上信息量、引用率和可验证性全面超越 RAG 和 STORM 基线。

**[Counterspeech the Ultimate Shield! Multi-Conditioned Counterspeech Generation through Attributed Prefix Learning](others/hippro_counterspeech_gen.md)**

:   提出 HiPPrO 两阶段框架用于多条件反仇恨言论生成——第一阶段通过层次化前缀学习在多个属性（策略+情感）空间中优化反言论生成，第二阶段用无参考无奖励的偏好优化提升建设性，策略一致性提升 ~38%，ROUGE 指标提升 2-3%。

**[How to Mitigate Overfitting in Weak-to-Strong Generalization?](others/how_to_mitigate_overfitting_in_weak-to-strong_generalization.md)**

:   提出两阶段训练框架解决弱到强泛化中的过拟合问题：第一阶段通过基于不确定性的过滤提高弱监督信号质量，第二阶段用已微调的强模型为被丢弃的难题重新生成答案以恢复问题质量，在 GSM8k 和 MATH 上将 PGR 从 7.19% 提升到 120.50%。

**[Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback](others/hybrid_preferences_learning_to_route_instances_for_human_vs_ai_feedback.md)**

:   本文提出HyPER（混合偏好路由器），通过训练性能预测模型来智能决定每个标注实例应由人类还是AI完成偏好标注，在RewardBench上比纯人类或纯AI标注分别提升7-13%，同时大幅降低标注成本。

**[Enhancing Hyperbole and Metaphor Detection with Their Bidirectional Dynamic Interaction and Emotion Knowledge](others/hyperbole_metaphor_detection.md)**

:   提出 EmoBi 框架，通过情感分析→情感引导的域映射→双向动态交互三阶段 prompting 流程，利用 LLM 挖掘夸张和隐喻背后的情感线索及二者的互促关系，在四个数据集上大幅超越 SoTA（TroFi 上夸张检测 F1 提升 28.1%，HYPO-L 上隐喻检测 F1 提升 23.1%）。

**[I0T: Embedding Standardization Method Towards Zero Modality Gap](others/i0t_embedding_standardization_method_towards_zero_modality_gap.md)**

:   提出 I0T 框架，通过发现并消除 CLIP 中图像/文本编码器各自学到的模态特异性特征（表现为归一化嵌入中的峰值激活），将模态差距降低至接近零，同时保持甚至提升下游任务性能，并提出了比 CLIPScore 更具可解释性的自动评估指标 I0T-Score。

**[If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?](others/if_attention_serves_as_a_cognitive_model_of_human_memory_retrieval_what_is_the_p.md)**

:   本文探究 Transformer Grammar（TG）的注意力机制能否作为人类记忆检索的认知模型，通过 Normalized Attention Entropy（NAE）将模型与人类阅读时间关联，发现基于句法结构的注意力比基于 token 的注意力更能解释人类句子处理行为，且两者提供独立互补的贡献。

**[Predicting Implicit Arguments in Procedural Video Instructions](others/implicit_arguments_video_instructions.md)**

:   提出 Implicit-VidSRL 数据集与 iSRL-Qwen2-VL 模型，针对过程性视频指令中省略的隐含论元（食材成分）进行预测，通过 SRL 框架将多步指令分解为 {verb, what, where/with} 三元组，在银标数据上微调后在隐含论元 F1 上超越 GPT-4o 达 17%。

**[Implicit Reasoning in Transformers is Reasoning through Shortcuts](others/implicit_reasoning_in_transformers_is_reasoning_through_shortcuts.md)**

:   本文通过在受控的多步数学推理数据集上从零训练 GPT-2，系统研究了语言模型的隐式推理机制，发现隐式推理本质上是基于模式匹配的捷径学习——在固定模式数据上能泛化，在非固定模式数据上则过拟合，这一发现在 SOTA 大模型上同样成立。

**[Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate](others/improve_rule_retrieval_and_reasoning_with_self-induction_and_relevance_reestimat.md)**

:   针对规则检索中查询（具体实例化事实）与规则（抽象变量形式）之间的语义鸿沟，提出 SIAR（自归纳增强检索）和 R3（规则相关性重评估）两种方法，通过将查询映射到规则语义空间并重新评估规则相关性，显著提升规则检索和下游推理性能。

**[Improving Language and Modality Transfer in Translation by Character-level Modeling](others/improving_language_and_modality_transfer_in.md)**

:   提出基于字符级编码器 charSONAR 的跨语言跨模态翻译方法，通过 teacher-student 训练获得字符级文本编码器，再用轻量适配器连接 1000+ 语言的 CTC ASR 模型（MMS），在 75 语言文本翻译和 33 语言语音翻译上实现 SOTA，零资源低资源场景表现尤其突出。

**[In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents](others/in_prospect_and_retrospect_reflective_memory_management_for_long-term_personaliz.md)**

:   本文提出 Reflective Memory Management (RMM) 机制，通过前瞻性反思（多粒度记忆摘要）和回顾性反思（强化学习驱动的在线检索优化）两个方向的结合，为长期个性化对话系统构建了高效的记忆管理框架，在 LongMemEval 上准确率提升超过 10%。

**[Inducing Lexicons of In-Group Language with Socio-Temporal Context](others/inducing_lexicons_of_in-group_language_with_socio-temporal_context.md)**

:   提出 LISTN（Lexicon Induction with Socio-Temporal Nuance）框架，利用动态词嵌入和用户嵌入联合建模社区语言的社会结构和时间演化，在反女性在线社区（manosphere）的群体内词汇归纳任务上达到 0.77 的平均精度，显著超越现有方法。

**[Inferring Functionality of Attention Heads from their Parameters](others/inferring_functionality_of_attention_heads_from_their_parameters.md)**

:   提出MAPS框架，通过将注意力头参数投影到词汇空间构建token映射矩阵$M$，无需任何推理或训练即可推断注意力头实现的功能，在6个LLM上验证了20种关系操作的映射准确性，并开发自动化pipeline发现了大量此前未被识别的注意力头功能。

**[Infogen: Generating Complex Statistical Infographics from Documents](others/infogen_generating_complex_statistical_infographics_from_documents.md)**

:   提出Infogen框架，将文本文档转化为复杂统计信息图（多子图组合），采用两阶段设计——先用微调LLM生成结构化中间元数据，再用LLM代码生成器和反馈模块迭代生成最终信息图代码。

**[INJONGO: A Multicultural Intent Detection and Slot-filling Dataset for 16 African Languages](others/injongo_a_multicultural_intent_detection_and_slot-filling_dataset_for_16_african.md)**

:   本文引入 Injongo——一个覆盖 16 种非洲语言的多文化意图检测与槽位填充基准数据集，由母语者在银行、旅行、家庭、餐饮等领域原创生成，实验揭示 LLM 在非洲语言上的槽位填充表现极差（GPT-4o 仅 26 F1），意图检测也明显落后于微调基线。

**[Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking](others/inner_thinking_transformer_leveraging_dynamic_depth_scaling_to_foster_adaptive_i.md)**

:   提出 Inner Thinking Transformer (ITT)，通过自适应 token 路由和残差思维连接，在不增加参数的情况下为关键 token 动态分配更多计算步骤，实现隐式深度推理，162M 参数即可达到 466M Transformer 96.5% 的性能。

**[InspireDebate: Multi-Dimensional Evaluation-Guided Reasoning for Debating](others/inspiredebate_multidim_evaluation_debating.md)**

:   提出双组件框架：InspireScore（融合4个主观维度+2个客观维度的辩论评估系统）和 InspireDebate（通过CoT-SFT + 多维DPO + Web-RAG 三阶段优化的辩论框架），评估系统与专家判断相关性提高 44%，辩论性能超越基线 57%。

**[Instruction-Tuning Data Synthesis from Scratch via Web Reconstruction](others/instruction-tuning_data_synthesis_from_scratch_via_web_reconstruction.md)**

:   提出 Web Reconstruction (WebR)，一种从原始网页文档全自动合成高质量指令微调数据的框架，通过"Web作为指令"和"Web作为回复"双视角范式，无需人工标注即可生成优于现有SOTA的IT数据。

**[Inter-Passage Verification for Multi-evidence Multi-answer QA](others/inter-passage_verification_for_multi-evidence_multi-answer_qa.md)**

:   本文提出 RI²VER 框架解决多答案问答（multi-answer QA）问题——先通过独立阅读大量检索段落生成高召回但含噪声的候选答案集，再通过段落间验证（生成验证问题 → 收集额外证据 → 跨段落综合验证）过滤错误答案，在 QAMPARI 和 RoMQA 上平均 F1 提升 11.17%。

**[Interlocking-free Selective Rationalization Through Genetic-based Learning](others/interlocking-free_selective_rationalization_through_genetic-based_learning.md)**

:   本文提出 GenSPP，首个完全消除 interlocking 问题的选择性合理化框架，通过遗传算法对生成器和预测器进行分离优化，在合成数据集和仇恨言论检测任务上显著提升了高亮质量（Hl-F1 提升 6.5%–10.3%），同时保持了可比的分类性能。

**[Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process](others/intuitive_fine_tuning.md)**

**[IRIS: Interactive Research Ideation System for Accelerating Scientific Discovery](others/iris_interactive_research_ideation_system_for_accelerating_scientific_discovery.md)**

:   提出 IRIS，一个开源的交互式研究构思系统，通过蒙特卡洛树搜索（MCTS）扩展测试时计算、细粒度反馈机制和基于查询的文献综合，实现人机协作的科学假设生成。

**[Is Linguistically-Motivated Data Augmentation Worth It?](others/is_linguistically-motivated_data_augmentation_worth_it.md)**

:   系统比较语言学驱动和非语言学（随机扰动）数据增强策略在两种低资源语言上的效果，发现语言学方法仅在生成的样本接近训练数据分布时才有优势，否则可能有害。

**[It's Not a Walk in the Park! Challenges of Idiom Translation in Speech-to-text Systems](others/its_not_a_walk_in_the_park_challenges_of_idiom_translation_in_speech-to-text_sys.md)**

:   本文首次系统比较了语音到文本翻译（SLT）、文本机器翻译（MT）和大语言模型（LLM）在习语翻译任务上的表现，发现 SLT 系统在处理习语时性能大幅下降，即便在编码器高层仍倾向于字面翻译，而 MT 和 LLM 对习语的处理能力明显更优。

**[Knowledge Tracing in Programming Education Integrating Students' Questions](others/knowledge_tracing_in_programming_education_integrating_students_questions.md)**

:   本文提出 SQKT（Students' Question-based Knowledge Tracing）模型，首次将学生提问和自动提取的技能信息整合到知识追踪中，用于预测编程教育中学生对后续编程题的完成情况，域内实验 AUC 提升高达 33.1%。

**[KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding](others/kodcode_a_diverse_challenging_and_verifiable_synthetic_dataset_for_coding.md)**

:   KodCode 提出一套三阶段合成数据管线（编程题目合成→解决方案+单元测试自验证→后训练数据合成），构建了 447K 经过验证的编程 question-solution-test 三元组，微调后的模型在 HumanEval、MBPP、BigCodeBench、LiveCodeBench 等基准上超越 Qwen2.5-Coder-32B-Instruct 和 DeepSeek-R1-Distill-Llama-70B。

**[LADDER: Language-Driven Slice Discovery and Error Rectification in Vision Classifiers](others/ladder_language-driven_slice_discovery_and_error_rectification_in_vision_classif.md)**

:   提出 LADDER 框架，利用 LLM 的推理能力和潜在领域知识，通过分析文本（图像描述/医学报告/元数据）自动发现视觉分类器中的系统性偏差切片（error slices），并通过伪标签生成和属性重平衡实现无需标注的多偏差缓解。

**[LAQuer: Localized Attribution Queries in Content-grounded Generation](others/laquer_localized_attribution.md)**

:   提出 Localized Attribution Queries (LAQuer) 任务——将生成文本中用户选定的片段精确定位到源文档的对应片段，实现比句子级归因更精细、比子句级归因更用户导向的溯源，在多文档摘要和长文本问答上显著减少了归因文本长度。

**[LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models](others/latim_measuring_latent_token-to-token_interactions_in_mamba_models.md)**

:   提出 LaTIM，一种针对 Mamba-1 和 Mamba-2 的 token 级分解方法，将 SSM 的隐式计算重构为类似 Transformer 注意力的 token-to-token 贡献矩阵，实现对 Mamba 模型的细粒度可解释性分析。

**[Learning to Align Multi-Faceted Evaluation: A Unified and Robust Framework](others/learning_to_align_multi-faceted_evaluation_a_unified_and_robust_framework.md)**

:   提出 ARJudge 评估框架，通过微调 Analyzer 自适应生成评估标准并执行文本+代码双驱动分析，配合无需微调的 Refiner 综合判断，在多个评估基准上超越现有微调评估器，尤其在指令遵循评估上通过代码驱动分析提升高达 11.1%。

**[Learning to Reason from Feedback at Test-Time](others/learning_to_reason_from_feedback_at_test-time.md)**

:   本文提出FTTT（Feedback at Test-Time Training）范式，将LLM在推理阶段对环境反馈的利用形式化为一个优化问题，并设计了可学习的测试时优化器OpTune，在四个推理数据集上实现了优于现有反馈利用方法的可扩展性和性能。

**[LegalReasoner: Step-wised Verification-Correction for Legal Judgment Reasoning](others/legalreasoner_step-wised_verification-correction_for_legal_judgment_reasoning.md)**

:   本文提出LegalReasoner框架，通过争议焦点识别、分步推理、过程验证器逐步校验以及专家设计的归因纠正策略来增强法律判决预测的可靠性，配合新发布的58130条香港法院案例LegalHK数据集，在LLAMA-3.1-70B上将与法院判决的一致率从72.37%提升至80.27%。

**[Length-Induced Embedding Collapse in PLM-based Models](others/length-induced_embedding_collapse_in_plm-based_models.md)**

:   发现并严格证明了 PLM 文本嵌入模型中的"长度坍缩"现象——长文本嵌入趋于聚集，源于 self-attention 作为低通滤波器随文本长度增加而滤波率增强，高频信息被过度抑制；提出 TempScale 方法通过降低 attention 温度来缓解长短文本嵌入分布差异，在 MTEB 上提升 0.94%、LongEmbed 上提升 1.10%。

**[Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation](others/leveraging_unit_language_guidance_to_advance_speech_modeling_in_textless_speech-.md)**

:   本文提出"单元语言"（unit language）的概念，通过n-gram语言建模将语音离散单元构造为类文本表示，并利用多任务学习引导无文本语音到语音翻译（S2ST）模型的训练过程，同时提出任务提示建模来缓解源端和目标端单元语言同时使用时的冲突，在VoxPopuli四语数据集上取得显著提升。

**[Limited Generalizability in Argument Mining: State-Of-The-Art Models Learn Datasets, Not Arguments](others/limited_generalizability_in_argument_mining_state-of-the-art_models_learn_datase.md)**

:   对 4 种 Transformer 模型在 17 个英语句子级论辩挖掘数据集上进行首次大规模跨数据集泛化评估，发现 SOTA 模型主要学到了数据集特有的词汇模式而非论辩的结构性信号，泛化能力远低于基准表现，但任务相关预训练和联合数据训练可部分缓解这一问题。

**[Literature Meets Data: A Synergistic Approach to Hypothesis Generation](others/literature_meets_data_hypothesis.md)**

:   提出首个将文献驱动和数据驱动假设生成进行协同整合的方法，通过 Refinement 和 Union 两种策略让 LLM 从论文摘要和观测数据中联合生成更具泛化性的假设，在五个社会科学分类任务的 OOD 数据集上比纯数据驱动方法平均提升 3.37%，并首次通过人类实验证明 LLM 生成的假设能显著改善人类决策准确率（+7.44% / +14.19%）。

**[LoGU: Long-form Generation with Uncertainty Expressions](others/logu_longform_gen_uncertainty.md)**

:   定义"长文本不确定性生成"（LoGU）任务，识别不确定性抑制和不确定性错位两个子挑战，提出基于分解的数据构造框架和 SFT+DPO 两阶段训练流水线，使 LLM 在长文本生成中对不确定事实显式表达不确定性，在三个数据集上将 Llama3-8B 的事实准确率从 51.9% 提升到 71.6%，错误声明数从 20.4 降到 5.81。

**[Low-Rank Interconnected Adaptation across Layers](others/low-rank_interconnected_adaptation_across_layers.md)**

:   提出 Lily（Low-rank Interconnected Adaptation across Layers），通过将 LoRA 的 A/B 适配器跨层解耦并互联共享，配合数据依赖的路由机制，在相同或更少参数下实现高秩权重更新，在多模态、多架构、多规模场景中均优于 LoRA。

**[MaCP: Minimal yet Mighty Adaptation via Hierarchical Cosine Projection](others/macp_minimal_yet_mighty_adaptation_via_hierarchical_cosine_projection.md)**

:   本文提出 MaCP——一种基于离散余弦变换（DCT）的参数高效微调方法，通过将权重变化投影到余弦频域并分层选择最关键的频率分量，在极低参数量（比 LoRA 少 99.7%）下实现了优于或媲美现有 PEFT 方法的性能。

**[Making FETCH! Happen: Finding Emergent Dog Whistles Through Common Habitats](others/making_fetch_happen_finding_emergent_dog_whistles_through_common_habitats.md)**

:   提出 FETCH! 基准和 EarShot 系统，用于在大规模社交媒体语料库中发现新兴的"狗哨"（dog whistle，即具有双重含义的编码表达），利用向量数据库和 LLM 的结合实现了比现有方法高 2-20 个 F-score 百分点的提升。

**[Map&Make: Schema Guided Text to Table Generation](others/mapmake_schema_guided_text_to_table_generation.md)**

:   提出 Map&Make 方法，先将非结构化文本"解剖"为命题原子语句（Map阶段），再从中推导表格模式并填充数据（Make阶段），在 Rotowire 和 Livesum 两个场景上显著提升了文本转表格的质量和可解释性。

**[Mapping the Podcast Ecosystem with the Structured Podcast Research Corpus](others/mapping_the_podcast_ecosystem_with_the_structured_podcast_research_corpus.md)**

:   构建并发布了 SPoRC——一个包含 110 万集播客转录的大规模数据集（含元数据、推断的说话者角色和 37 万集的音频特征），并通过话题分析、嘉宾共现网络分析和 George Floyd 事件响应性分析，首次全面刻画了播客生态系统的内容、结构和响应性。

**[MapQaTor: An Extensible Framework for Efficient Annotation of Map-Based QA Datasets](others/mapqator_an_extensible_framework_for_efficient_annotation_of_map-based_qa_datase.md)**

:   提出 MapQaTor——一个可扩展的开源 Web 框架，通过集成多种地图 API（Google Maps、OpenStreetMap 等），将地理空间 QA 数据集的标注速度提升至少 30 倍，同时通过 API 响应缓存确保数据可复现性。

**[MDCure: A Scalable Pipeline for Multi-Document Instruction-Following](others/mdcure_a_scalable_pipeline_for_multi-document_instruction-following.md)**

:   提出 MDCure 框架，通过两阶段流程（生成+过滤）自动构建高质量的多文档指令数据，并训练 MDCureRM 多目标奖励模型进行数据过滤，使微调后的 LLM（最高 70B）在多文档和长上下文任务上相比基线提升高达 75.1%，且实现跨任务、跨领域的强泛化能力。

**[Meaning Beyond Truth Conditions: Evaluating Discourse Level Understanding via Anaphora Accessibility](others/meaning_beyond_truth_conditions_evaluating_discourse_level_understanding_via_ana.md)**

:   本文提出语义理解能力的层级框架（词汇/句子/话语），构建了基于照应可及性（anaphora accessibility）的评估数据集，发现 LLM 在某些结构上与人类一致但在其他结构上存在系统性分歧——LLM 依赖词汇线索而非结构化抽象。

**[Measuring the Effect of Transcription Noise on Downstream Language Understanding Tasks](others/measuring_the_effect_of_transcription_noise_on_downstream_language_understanding.md)**

:   提出ENDow框架，首次系统化地分析ASR转录噪声对下游NLU任务的影响，通过可配置的pipeline评估不同噪声强度和类型下任务模型的行为，发现命名实体是最关键的词类型，且模型能容忍一定程度的噪声。

**[Memorization: A Close Look at Books](others/memorization_a_close_look_at_books.md)**

:   系统研究 Llama 3 系列模型对完整书籍的记忆化程度，发现书籍提取率与其流行度（训练数据重复度代理）高度正相关，并通过 LoRA 微调揭示指令微调的抗反刍缓解措施仅涉及极少量集中在底层 transformer block 的权重变化。

**[Meta-Learning Neural Mechanisms rather than Bayesian Priors](others/meta-learning_neural_mechanisms_rather_than_bayesian_priors.md)**

:   挑战了"元学习在神经网络中蒸馏贝叶斯简单性先验"的主流观点，通过形式语言实验证明元学习实际上是在模型中植入有用的**神经机制**（如计数器），而非学习简单性偏好。

**[Meta-Reflection: A Feedback-Free Reflection Learning Framework](others/meta-reflection_a_feedback-free_reflection_learning_framework.md)**

:   提出 Meta-Reflection 框架，通过可学习的元反思代码本（codebook）存储和检索反思性洞察，使 LLM 在推理时无需外部反馈和多轮迭代，仅需单次前向传播即可利用历史反思经验来改善输出质量，在编程、数学推理和电商意图检测任务上均取得显著提升。

**[MEXMA: Token-level Objectives Improve Sentence Representations](others/mexma_token-level_objectives_improve_sentence_representations.md)**

:   提出 MEXMA，一种结合句子级和 token 级目标的跨语言句子编码器训练方法：用一种语言的句子表示去预测另一种语言的被掩码 token，同时让句子和 token 的梯度都直接更新编码器，在双文本挖掘和多项下游任务上超越 SONAR 和 LaBSE。

**[Micro-Act: Mitigate Knowledge Conflict in QA via Actionable Self-Reasoning](others/micro_act_knowledge_conflict_reasoning.md)**

:   提出 Micro-Act 框架，通过层次化动作空间（导航/功能/桥接动作）和自适应粒度分解，让 LLM 自动感知上下文复杂度并逐层拆解知识对比，在 5 个知识冲突基准上全面超越 SOTA，同时在无冲突场景下也保持鲁棒。

**[MindRef: Mimicking Human Memory for Hierarchical Reference Retrieval with Fine-Grained Location Awareness](others/mindref_mimicking_human_memory_hierarchical_reference_retrieval.md)**

:   提出 MindRef 框架，模拟人类先回忆文档标题再定位具体段落的两阶段记忆模式，通过 Trie 和 FM-Index 约束解码让 LLM 独立召回参考段落，无需额外检索模型或预分段。

**[Minimal Pair-Based Evaluation of Code-Switching](others/minimal_pair-based_evaluation_of_code-switching.md)**

:   本文提出基于最小对（minimal pairs）的代码切换（code-switching, CS）评估方法，为11个语言对各收集最多1000个最小对，发现双语者和大规模LLM都倾向于自然出现的CS句子，且模型越大偏好越一致，封闭类词的操纵产生最大概率差异。

**[MIR: Methodology Inspiration Retrieval for Scientific Research Problems](others/mir_methodology_inspiration_retrieval_for_scientific_research_problems.md)**

:   本文定义了方法论启发检索（Methodology Inspiration Retrieval, MIR）新任务——为给定的科研问题检索能提供方法论启发的论文，构建了Methodology Adjacency Graph (MAG)来捕获方法论传承关系，在Recall@3上提升+5.4、mAP上提升+7.8，结合LLM重排序再获+4.5/+4.8的额外提升。

**[Mitigating Shortcut Learning with InterpoLated Learning](others/mitigating_shortcut_learning_with_interpolated_learning.md)**

:   提出 InterpoLated Learning (InterpoLL)，通过将多数样本的表示与同类少数样本的表示进行插值，削弱模型对虚假关联（shortcut）的依赖，显著提升少数样本上的泛化能力。

**[Mixture of Small and Large Models for Chinese Spelling Check](others/mixture_of_small_and_large_models_for_chinese_spelling_check.md)**

:   本文提出在Beam Search解码阶段动态混合小模型（fine-tuned BERT）和大语言模型（LLM）的概率分布来进行中文拼写纠错，无需微调LLM即可兼顾小模型的精确纠错和LLM的流畅性，在多个CSC数据集上达到SOTA。

**[MockConf: A Student Interpretation Dataset: Analysis, Word- and Span-level Alignment and Baselines](others/mockconf_a_student_interpretation_dataset_analysis_word-_and_span-level_alignmen.md)**

:   本文构建了 MockConf——一个以捷克语为中心的**学生同声传译数据集**（7 小时，5 种欧洲语言），提供人工标注的 span 级和 word 级对齐，同时发布了专用标注工具 InterAlign，并建立了自动对齐的基线和评估指标体系。

**[MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](others/more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)**

:   提出 MoRE (Mixture of Low-Rank Experts)，将 LoRA 中的不同秩视为不同专家，通过自适应秩选择器为每个任务动态选择最合适的秩，配合对比学习优化的任务嵌入和平衡数据采样策略，使用单个 LoRA 模块实现高效的多任务微调。

**[MOSAIC: Multiple Observers Spotting AI Content](others/mosaic_multiple_observers_spotting_ai_content.md)**

:   基于信息论中的通用压缩原理，提出 MOSAIC——多 LLM 集成的 AI 生成文本检测方法，通过 Blahut-Arimoto 算法为多个 detector LLM 计算最优组合权重，构建混合分布作为观察者，比较文本的实际 surprisal 与混合模型的期望交叉熵差异来判断是否为 AI 生成，在多个域/语言/生成器上鲁棒优于单模型和双模型（如 Binoculars）方法。

**[Multi-Facet Blending for Faceted Query-by-Example Retrieval](others/multi-facet_blending_for_faceted_query-by-example_retrieval.md)**

:   > 提出 FaBle（Multi-Facet Blending）数据增强方法，通过对文档进行面向分解（decomposition）、面向生成（generation）、面向重组（recomposition）三阶段，仅用 1K 文档合成出面向条件的训练三元组，在数据稀缺条件下显著提升分面 QBE 检索效果，特别是在最具挑战性的 method 分面上超越了使用 130 万+ 数据训练的强基线。

**[Multi-Hop Question Generation via Dual-Perspective Keyword Guidance](others/multi-hop_question_generation_via_dual-perspective_keyword_guidance.md)**

:   定义了双视角关键词——问题关键词（捕捉提问者意图）和文档关键词（反映 QA 对相关内容），并提出 DPKG 框架，通过扩展 Transformer 编码器和两个答案感知解码器，将关键词无缝集成到多跳问题生成过程中。

**["My life is miserable, have to sign 500 autographs everyday": Exposing Humblebragging, the Brags in Disguise](others/my_life_is_miserable_have_to_sign_500_autographs_everyday_exposing_humblebraggin.md)**

:   首次将 humblebragging（谦虚式自夸）检测引入计算语言学领域，提出了一个4元组形式化定义，构建了 HB-24 合成数据集，并在 ML/DL/LLM 上进行了全面基准评估，GPT-4o 在 zero-shot+定义 设定下达到 0.88 F1，超越人类标注者。

**[Narrative Media Framing in Political Discourse](others/narrative_media_framing_in_political_discourse.md)**

:   将叙事学理论与媒体框架分析相结合，提出了包含角色（英雄/反派/受害者）、冲突/解决、文化故事三个结构化组件的叙事框架分析体系，在气候变化和 COVID-19 两个领域验证了该框架的有效性和可迁移性。

**[Unifying Continuous and Discrete Text Diffusion with Non-simultaneous Diffusion Processes](others/neodiff_unified_text_diffusion.md)**

:   提出 NeoDiff，通过引入"外在时间"（句子级扩散进度）和"内在时间"（token 级扩散进度）的双时间框架，利用 Poisson 过程为每个 token 独立分配细粒度噪声水平，并用上下文感知的时间预测器自适应调节去噪进度，统一了离散和连续文本扩散模型的理论框架，在机器翻译、复述、文本简化等多个任务上超越现有扩散基线。

**[Neural Parameter Search for Slimmer Fine-Tuned Models and Better Transfer](others/neural_parameter_search_for_slimmer_fine-tuned_models_and_better_transfer.md)**

:   提出Neural Parameter Search (NPS)，通过在task vector的低秩子空间中搜索最优权重系数来提升微调模型的剪枝效率，在知识迁移（+1.5%）、模型融合（+2.1%）和压缩（40%效率提升）三个场景下均取得显著改进。

**[Neuron Empirical Gradient: Discovering and Quantifying Neurons' Global Linear Controllability](others/neuron_empirical_gradient_discovering_and_quantifying_neurons_global_linear_cont.md)**

:   揭示了预训练语言模型 FF 层神经元激活值与模型输出之间存在全局线性关系，提出了神经元经验梯度（NEG）来量化这种线性关系，并设计了高效估算方法 NeurGrad，最终通过技能神经元探测实验证明 NEG 能有效表征多种语言技能。

**[On Support Samples of Next Word Prediction](others/on_support_samples_of_next_word_prediction.md)**

:   基于表示定理（representer theorem），研究语言模型下一词预测中训练样本的角色，发现两类支持样本（促进预测和抑制预测），并证明支持样本是样本的内在属性（训练前即可预测），而非支持样本对表示学习至关重要。

**[One for All: Update Parameterized Knowledge Across Multiple Models with Once Edit](others/one_for_all_update_parameterized_knowledge_across_multiple_models_with_once_edit.md)**

:   提出 OnceEdit，通过编辑一个轻量级插件模型并利用异构模型集成技术将编辑后的知识迁移到多个 LLM，实现"一次编辑，多模型更新"，在 ZsRE 和 Counterfact 数据集上显著超越现有方法。

**[Optimizing Decomposition for Optimal Claim Verification](others/optimizing_decomposition_for_optimal_claim_verification.md)**

:   提出动态分解（Dynamic Decomposition）框架，通过强化学习从验证器反馈中学习分解策略，将声明（claim）分解为验证器偏好的原子性粒度，弥合分解器与验证器之间的性能差距。

**[Partial Colexifications Improve Concept Embeddings](others/partial_colexifications_improve_concept_embeddings.md)**

:   首次将部分共词化（affix/overlap colexification）引入概念嵌入训练，在语义相似性建模、语义变化预测和词语联想预测三个任务上均优于仅使用完全共词化的基线。

**[Towards Better Evaluation for Generated Patent Claims](others/patclaimeval_patent_evaluation.md)**

:   提出首个专利权利要求评估基准 Patent-CE（1228 个专家标注的比较评估数据点）和专用评估方法 PatClaimEval（基于 Longformer + 对比学习变体），在特征完整性、概念清晰度、术语一致性、逻辑连接和整体质量五个维度上与人类专家评估的相关性全面超越 13 种现有指标（包括 G-Eval-4），整体质量维度的 Spearman 提升 58%。

**[Persistent Homology of Topic Networks for the Prediction of Reader Curiosity](others/persistent_homology_of_topic_networks_for_the_prediction_of_reader_curiosity.md)**

:   > 将文本的主题网络结构用持续同调 (Persistent Homology) 量化为拓扑空洞（连通分量、环、空腔），以此作为"信息空白"的代理变量来预测读者好奇心，在《饥饿游戏》小说上实现了 73% 的解释偏差（vs 基线 30%）。

**[Persona Dynamics: Unveiling the Impact of Personality Traits on Agents in Text-Based Games](others/persona_dynamics_unveiling_the_impact_of_persona_traits_on_agents_in_text-based_.md)**

:   提出 PANDA 方法，将人类人格特质（Big Five + Dark Triad 共8种）投射到文本游戏智能体的策略学习中，通过人格分类器引导 Q 值调整，发现高开放性（Openness）人格在冒险类文本游戏中表现显著优于其他人格类型。

**[PersonaBench: Evaluating AI Models on Understanding Personal Information through Accessing (Synthetic) Private User Data](others/personabench_evaluating_ai_models_on_understanding_personal_information_through_.md)**

:   提出 PersonaBench 基准及配套的合成私有数据生成管线，系统评估 AI 模型通过 RAG 从模拟用户数据中提取个人信息的能力，揭示当前方案的严重不足。

**[Personalized Generation In Large Model Era: A Survey](others/personalized_generation_in_large_model_era_a_survey.md)**

:   首篇跨模态个性化生成（PGen）系统综述，提出统一的用户中心视角将 NLP/CV/IR 社区的研究纳入同一框架，覆盖文本/图像/视频/音频/3D/跨模态六大模态。

**[All That Glitters is Not Novel: Plagiarism in AI Generated Research](others/plagiarism_ai_generated_research.md)**

:   在对自主科研 Agent（如 AI Scientist）生成的研究文档进行专家审查后发现，24% 的文档是"智能剽窃"——方法论与已有工作一一对应但不引用原始来源，且现有剽窃检测工具无法识别这种"改头换面"的抄袭。

**[PopAlign: Diversifying Contrasting Patterns for a More Comprehensive Alignment](others/popalign_diversifying_contrasting_patterns_for_a_more_comprehensive_alignment.md)**

:   提出PopAlign框架，从Prompt、Model、Pipeline三个层面构建六种多样化对比策略（包括创新的Elicitive Contrast），无需额外人工标注即可合成高质量偏好数据，实现更全面的LLM对齐。

**[Predicting Through Generation: Why Generation Is Better for Prediction](others/predicting_through_generation_why_generation_is_better_for_prediction.md)**

:   本文从信息论角度证明了token级生成比pooled表示保留更多互信息，提出PredGen框架通过scheduled sampling和task adapter解决生成式预测中的exposure bias和格式不匹配问题，并设计了Writer-Director Alignment Loss统一生成与预测目标。

**[Principled Understanding of Generalization for Generative Transformer Models in Arithmetic Reasoning Tasks](others/principled_generalization_arithmetic.md)**

:   建立首个统一理论框架来理解 Transformer 在算术任务（加法/乘法/模运算）上的泛化行为——从任务性质（平移不变性）和位置编码类型（APE/RPE）的交互出发，解释了之前困扰领域的多个泛化谜题（如加法能泛化但乘法不能，模100能泛化但模101不能），实验验证了理论预测。

**[Substance over Style: Evaluating Proactive Conversational Coaching Agents](others/proactive_conversational_coaching.md)**

:   通过健康教练领域的专家访谈和用户研究（31 名参与者、155 段对话），系统评估了五种不同对话风格（Directive、Interrogative、Facilitative）的 LLM 教练 Agent，发现用户高度重视核心功能性（substance）而对缺乏功能性时的风格修饰（style）持负面态度，同时揭示了用户第一人称评价与专家/LLM 第三方评价之间的显著不一致。

**[ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering](others/proxann_topic_model_eval.md)**

:   提出面向实际使用场景的主题模型评估协议ProxAnn，结合可扩展的人类评估流程和LLM代理标注者，发现最佳LLM代理在统计上与人类标注者不可区分，可作为自动化评估的合理替代。

**[PVP: An Image Dataset for Personalized Visual Persuasion with Persuasion Strategies, Viewer Characteristics, and Persuasiveness Ratings](others/pvp_an_image_dataset_for_personalized.md)**

:   构建了首个将图像说服策略与 2,521 位标注者心理特征（人格/价值观/道德基础）关联的大规模数据集 PVP（28,454 张图像、596 条行为消息、9 种说服策略），并在"个性化说服图像生成"和"说服力自动评估"两个基准任务上验证了心理特征对提升说服效果的关键作用。

**[QG-SMS: Enhancing Test Item Analysis via Student Modeling and Simulation](others/qg-sms_enhancing_test_item_analysis_via_student_modeling_and_simulation.md)**

:   QG-SMS 提出用单个 LLM 模拟不同理解水平的学生群体，通过学生画像生成、表现预测和分析三步流程，弥补了现有 LLM 评估器在考后分析维度（题目难度、区分度、干扰项效率）上的严重不足，在多个数据集上实现了最高一致性准确率。

**[QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning](others/qualispeech_a_speech_quality_assessment_dataset_with_natural_language_reasoning_.md)**

:   本文提出 QualiSpeech，首个包含 11 个维度标注和详细自然语言推理描述的语音质量评估数据集，以及配套的评测基准，证明了微调后的听觉 LLM 能生成关于噪声和失真的详细描述，并展示了推理增强质量评估的潜力。

**[Quantifying Lexical Semantic Shift via Unbalanced Optimal Transport](others/quantifying_lexical_semantic_shift_via_unbalanced_optimal_transport.md)**

:   将Unbalanced Optimal Transport（UOT）应用于上下文化词嵌入集合，提出Sense Usage Shift（SUS）指标在每个用法实例级别量化语义变化，统一解决实例级变化检测、词级变化幅度量化和词义扩展/缩小判定三项任务。

**[Rank, Chunk, and Expand: Lineage-Oriented Reasoning for Taxonomy Expansion](others/rank_chunk_and_expand_lineage-oriented_reasoning_for_taxonomy_expansion.md)**

:   LORex 提出了一个即插即用的分类体系扩展框架，结合判别式排序器 TEMPORA（基于欧拉路径的分类路径语言化）和迭代式 LLM 推理（语义过滤→父节点检索→路径验证），无需微调 LLM，在 4 个基准上实现了 12% 的准确率提升和 5% 的 Wu&P 提升。

**[Rationales Are Not Silver Bullets: Measuring the Impact of Rationales on Model Performance and Reliability](others/rationales_are_not_silver_bullets_measuring_the_impact_of_rationales_on_model_pe.md)**

:   本文通过对 18 个数据集、7 类任务的系统实验，发现在训练数据中加入 rationale（推理过程）并非总是有益——rationale 有时会削弱模型性能，但可以提升模型可靠性（校准度），且性能和可靠性的改善呈线性相关，两者都受任务固有难度驱动。

**[Re-identification of De-identified Documents with Autoregressive Infilling](others/reidentification_deidentified.md)**

:   提出一种基于 RAG 的去标识化文档重标识方法：先用稀疏+稠密检索找到相关背景文档，再用自回归填充模型推断被遮蔽的个人标识信息，在三个数据集上恢复了高达 80% 的被遮蔽文本。

**[Identifying Reliable Evaluation Metrics for Scientific Text Revision](others/reliable_eval_metrics_scientific.md)**

:   系统分析了传统相似度指标（ROUGE、BERTScore 等）在科学文本修订评估中的局限性，发现它们与编辑距离强相关且惩罚深度修改，提出结合 LLM-as-Judge 和任务特定跨域指标的混合评估方法，在与人类判断的对齐度上显著优于单一指标。

**[RePanda: Pandas-powered Tabular Verification and Reasoning](others/repanda_pandas-powered_tabular_verification_and_reasoning.md)**

:   提出 RePanda，将自然语言声明翻译为可执行的 pandas 查询来实现表格事实验证，在 TabFact 上达到 84.09% 准确率，在 OOD 的 WikiFact 上无需额外微调达 84.72%，同时以仅 7B 参数的模型逼近 671B DeepSeek-Chat 的零样本性能，并扩展至表格问答任务取得 75.1% 准确率。

**[Research Borderlands: Analysing Writing Across Research Cultures](others/research_borderlands_analysing_writing_across_research_cultures.md)**

:   通过访谈跨学科研究者构建学术写作文化规范框架（结构/风格/修辞/引用四类），并用计算指标量化11个CS社区的写作差异，揭示LLM在跨社区写作改编时存在严重的"同质化"倾向。

**[Retrospective Learning from Interactions](others/retrospective_learning_from_interactions.md)**

:   提出 ReSpect 方法，让多模态 LLM 通过回顾性地解码用户在多轮交互中的隐式反馈信号来自我改进，无需任何外部标注，在数千次人机交互中将任务完成率从 31% 提升至 82%。

**[Revisiting Weak-to-Strong Generalization: Reverse KL vs. Forward KL](others/revisiting_weak-to-strong_generalization_in_theory_and_practice_reverse_kl_vs_fo.md)**

:   在 Weak-to-Strong Generalization (W2SG) 框架中，提出用 reverse KL 替代 forward KL 作为损失函数——理论证明 reverse KL 的 mode-seeking 特性可保证强模型超过弱监督者至少"分歧量"的幅度，实验在 GPT-2/Pythia/Qwen2.5 系列上验证 reverse KL/CE 在 12/12 设置中超越 forward KL 且噪声鲁棒性更好。

**[RMoA: Optimizing Mixture-of-Agents through Diversity Maximization and Residual Compensation](others/rmoa_optimizing_mixture-of-agents_through_diversity_maximization_and_residual_co.md)**

:   受 ResNet 残差学习启发，提出 RMoA 框架，通过嵌入式多样性贪心选择、残差提取/聚合智能体和自适应终止机制来优化多智能体协作架构，在降低计算开销的同时实现 SOTA 性能。

**[RoToR: Towards More Reliable Responses for Order-Invariant Inputs](others/rotor_towards_more_reliable_responses_for_order-invariant_inputs.md)**

:   提出 RoToR，一种基于全局排序和循环位置编码分配的零样本顺序不变语言模型，通过最小化位置 ID 修改来实现稳定的顺序不变性，并设计选择路由（Selective Routing）机制自适应处理混合输入类型。

**[(RSA)²: A Rhetorical-Strategy-Aware Rational Speech Act Framework for Figurative Language Understanding](others/rsatexttwosuperior_a_rhetorical-strategy-aware_rational_speech_act_framework_for.md)**

:   本文提出 (RSA)² 框架，通过在概率语用学的 RSA 框架中显式建模说话者的修辞策略（如反讽、夸张），使 LLM 能够在不建模说话者动机的情况下正确理解非字面义，在反讽理解数据集 PragMega+ 上达到 SOTA。

**[Rubrik's Cube: Testing a New Rubric for Evaluating Explanations on the CUBE Dataset](others/rubriks_cube_testing_a_new_rubric_for_evaluating_explanations_on_the_cube_datase.md)**

:   提出 Rubrik——受教育评估启发的解释质量评价量规，基于三层嵌套类型体系（Commentary⊆Justification⊆Argument）+ 8 维质量维度，配套 CUBE 数据集（26K 条由人类和 6 个 LLM 生成的解释），发现 LLM 解释低质主因是缺乏简洁性而非连贯性。

**[S2WTM: Spherical Sliced-Wasserstein Autoencoder for Topic Modeling](others/s2wtm_spherical_sliced-wasserstein_autoencoder_for_topic_modeling.md)**

:   提出 S2WTM，一种基于球面切片 Wasserstein 自编码器的主题模型，在超球面潜空间上对齐聚合后验与先验分布，有效避免 VAE 的后验坍塌问题，同时在主题连贯性和多样性上超越现有 SOTA。

**[S3 - Semantic Signal Separation](others/s3_-_semantic_signal_separation.md)**

:   S3将主题建模概念化为发现语义空间中独立语义轴的过程，利用独立成分分析（ICA）分解文档嵌入矩阵，无需预处理即可产生高度连贯且多样化的主题，同时是最快的上下文主题模型（平均比BERTopic快4.5倍）。

**[SDD: Self-Degraded Defense against Malicious Fine-tuning](others/sdd_self-degraded_defense_against_malicious_fine-tuning.md)**

:   SDD通过训练LLM对有害指令生成高质量但无关的良性回复来实现防御：当攻击者进行恶意微调时，模型的通用能力会显著下降，从而无法有效执行恶意指令。

**[Segment-Based Attention Masking for GPTs](others/segment-based_attention_masking_for_gpts.md)**

:   MAS（Masked Attention by Segment）在预训练 GPT 模型的 prefill 阶段将因果注意力掩码替换为按段（segment）的双向注意力——同一段内的 token 可以互相 attend，生成阶段仍保持因果掩码——通过 LoRA 微调即可在 8 个常识推理任务上一致提升性能（Llama-3-8B 平均 +1.8%，Llama-3.2-3B +3.3%），无额外计算开销。

**[Self-Foveate: Enhancing Diversity and Difficulty of Synthesized Instructions from Unsupervised Text via Multi-Level Foveation](others/self-foveate_enhancing_diversity_and_difficulty_of_synthesized_instructions_from.md)**

:   提出 Self-Foveate 方法，受人类视觉注视机制启发，通过"微观-散射-宏观"三级注视策略，从无监督文本中系统性提取多粒度信息，合成具有更高多样性和难度的指令数据，用于 LLM 的指令微调。

**[SEOE: A Scalable and Reliable Semantic Evaluation Framework for Open Domain Event Detection](others/seoe_semantic_eval.md)**

:   针对开放域事件检测（ODED）评估的两大痛点——有限 benchmark 缺乏真实世界代表性、token 级匹配指标无法捕捉语义相似性——提出 SEOE 框架，构建包含 564 种事件类型覆盖 7 大领域的可扩展 benchmark，并引入基于 LLM 的语义 F1 评估指标。

**[Sightation Counts: Leveraging Sighted User Feedback in Building a BLV-aligned Dataset of Diagram Descriptions](others/sightation_counts_leveraging_sighted_user_feedback_in_building_a_blv-aligned_dat.md)**

:   提出让视力正常者「评估」而非「生成」VLM 的图表描述，构建了首个经 BLV 专业教育者验证的 5k 图表 / 137k 样本多任务数据集 Sightation，偏好微调 2B 模型后在 BLV 有用性评分上平均提升 1.67σ。

**[Sleepless Nights, Sugary Days: Creating Synthetic Users with Health Conditions for Realistic Coaching Agent Interactions](others/sleepless_nights_sugary_days_creating_synthetic_users_with_health_conditions_for.md)**

:   提出一个端到端框架，基于真实人口学、健康/生活方式和行为/心理特征数据生成有健康状况的合成用户（涵盖睡眠和糖尿病管理），用于评估健康教练Agent的交互质量，并通过人类专家评估验证其显著优于通用合成用户。

**[SoRFT: Issue Resolving with Subtask-oriented Reinforced Fine-Tuning](others/sorft_issue_resolving_with_subtask-oriented_reinforced_fine-tuning.md)**

:   提出 SoRFT（Subtask-oriented Reinforced Fine-Tuning），将 GitHub Issue 解决任务分解为文件定位、函数定位、行定位和代码编辑四个子任务，通过拒绝采样SFT + 基于规则的PPO强化学习两阶段训练，显著提升开源LLM在 SWE-Bench 上的 Issue 解决能力。

**[SOTOPIA-Ω: Dynamic Strategy Injection Learning and Social Instruction Following Evaluation for Social Agents](others/sotopia-ensuremathomega_dynamic_strategy_injection_learning_and_social_instructi.md)**

:   本文提出 SOTOPIA-Ω 框架，通过将谈判理论中的多步推理策略和直接策略动态注入到专家 agent 中，自动构建高质量社交对话训练语料，并定义了"社交指令跟随（S-IF）"这一新概念和两个评估指标，在 7B 模型上超越了 GPT-4 专家 agent 的社交目标达成能力。

**[SPOT: Bridging Natural Language and Geospatial Search for Investigative Journalists](others/spot_bridging_natural_language_and_geospatial_search_for_investigative_journalis.md)**

:   提出 SPOT 系统，通过微调 LLaMA 3 将自然语言场景描述转换为 YAML 查询，结合语义标签捆绑机制实现对 OpenStreetMap 数据的可靠自然语言访问，服务于调查新闻的地理定位验证。

**[Spotting Out-of-Character Behavior: Atomic-Level Evaluation of Persona Fidelity in Open-Ended Generation](others/spotting_out-of-character_behavior_atomic-level_evaluation_of_persona_fidelity_i.md)**

:   提出原子级（句子级）评估框架，通过三个指标（ACC_atom、IC_atom、RC_atom）细粒度检测大语言模型在开放式文本生成中的角色偏离（Out-of-Character）行为，弥补了传统整体评分方法无法捕捉长文本中微妙人格不一致的问题。

**[Statistical Deficiency for Task Inclusion Estimation](others/statistical_deficiency_task_inclusion.md)**

:   基于统计缺陷性（statistical deficiency）理论，提出一种理论驱动的任务包含关系（task inclusion）定义与度量框架，以信息充分性（information sufficiency, IS）作为可计算代理指标，通过比较微调模型的中间层表征来估计任务间的包含程度，并在合成数据和真实NLP任务上成功重建了经典NLP pipeline的层次关系。

**[STRICTA: Structured Reasoning in Critical Text Assessment for Peer Review and Beyond](others/stricta_structured_reasoning_in_critical_text_assessment_for_peer_review_and_bey.md)**

:   提出 STRICTA 框架，基于结构因果模型（SCM）将文本评审建模为显式的逐步推理图（workflow），在生物医学论文评审中收集 40+ 位专家的 4000+ 推理步骤数据集，发现先验知识差异是专家分歧主因、写作风格对最终评审有因果影响，LLM 存在错误传播但人类监督可有效缓解。

**[Subword Models Struggle with Word Learning, but Surprisal Hides It](others/subword_models_struggle_with_word_learning_but_surprisal_hides_it.md)**

:   本文通过心理语言学的词汇判断任务（lexical decision task）揭示了使用子词（BPE）分词的语言模型在单词学习方面存在严重缺陷，而基于字符级分词的模型能轻松完成该任务；当使用 surprisal（在语境中的出乎意料程度）来评估时，这一差距被掩盖了。

**[SudoLM: Learning Access Control of Parametric Knowledge with Authorization Alignment](others/sudolm_authorization_alignment.md)**

:   SudoLM 提出了一种 LLM 参数化知识访问控制框架，通过"SUDO key"机制让授权用户解锁受限知识（如医学领域知识），未授权用户则只能访问公开知识，用 DPO 的 authorization alignment 在一个模型内实现了传统需要多版本模型才能完成的分级访问控制。

**[Synergistic Weak-Strong Collaboration by Aligning Preferences](others/synergistic_weak-strong_collaboration_by_aligning_preferences.md)**

:   本文提出 CoWest 框架，通过让专业化的弱模型（如 LLaMA3-8B）生成初始草稿，再由通用强模型（如 GPT-4）精炼，并利用协作反馈通过 DPO 微调弱模型以对齐强模型偏好，在反事实推理、医学和伦理三个领域显著超越单模型和已有协作方法。

**[Synthia: Novel Concept Design with Affordance Composition](others/synthia_novel_concept_design_with_affordance_composition.md)**

:   Synthia 提出了一种基于 affordance（功能可供性）组合的新颖概念设计框架，通过层次化概念本体、affordance 采样策略和课程学习微调 T2I 模型，生成既视觉新颖又功能连贯的创新设计。

**[TabXEval: Why this is a Bad Table? An eXhaustive Rubric for Table Evaluation](others/tabxeval_why_this_is_a_bad_table_an_exhaustive_rubric_for_table_evaluation.md)**

:   TabXEval 提出了一种基于结构化评分规则（rubric）的两阶段表格评估框架——先通过 TabAlign 对齐参考表和生成表的结构，再通过 TabCompare 进行语义和语法层面的细粒度比较，同时构建了多领域基准 TabXBench。

**[TACLR: A Scalable and Efficient Retrieval-Based Method for Industrial Product Attribute Value Identification](others/taclr_a_scalable_and_efficient_retrieval-based_method_for_industrial_product_att.md)**

:   TACLR 提出了首个基于检索范式的产品属性值识别（PAVI）方法，通过分类感知对比学习和自适应推理机制，在处理隐含值、OOD 值和归一化输出方面全面超越分类和生成方法，并已成功部署在闲鱼（Xianyu）平台。

**[Tag-Evol: Achieving Efficient Instruction Evolving via Tag Injection](others/tag-evol_achieving_efficient_instruction_evolving_via_tag_injection.md)**

:   Tag-Evol 提出了一种基于知识标签注入的指令进化框架，通过构建多步细粒度标签池和预算控制注入机制，无需迭代即可生成不同难度的高质量进化指令数据，在多任务多骨干上显著优于 Evol-Instruct。

**[TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](others/targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)**

:   TARGA 提出了一种针对性的合成数据生成框架，无需任何人工标注即可为知识库问答（KBQA）动态生成高相关性的合成示例用于上下文学习，仅用 7B 模型即在 GrailQA（+7.7 F1）和 KBQA-Agent（+12.2 F1）上大幅超越所有非微调方法。

**[Task-Informed Anti-Curriculum by Masking Improves Downstream Performance on Text](others/task-informed_anti-curriculum_by_masking_improves_downstream_performance_on_text.md)**

:   TIACBM 提出了一种任务感知的反课程掩码微调策略：利用下游任务知识（如情感极性、词性标签）决定哪些 token 被掩码，并采用周期衰减的掩码率，在情感分析、文本分类和作者归属三个任务上均取得统计显著的性能提升。

**[Temporal Reasoning for Timeline Summarisation in Social Media](others/temporal_reasoning_for_timeline_summarisation_in_social_media.md)**

:   本文提出通过构建新的叙事时序推理数据集 NarrativeReason 来增强 LLM 的时序推理能力，并通过知识蒸馏框架将时序推理知识迁移到小模型中，同时训练其完成时间线摘要任务，在跨域心理健康摘要任务上取得最优效果并显著减少幻觉。

**[TestNUC: Enhancing Test-Time Computing Approaches and Scaling through Neighboring Unlabeled Data Consistency](others/testnuc_enhancing_test-time_computing_approaches_and_scaling_through_neighboring.md)**

:   TestNUC 提出了一种线性扩展的测试时推理增强方法，通过检索测试样本的近邻无标注数据，让 LLM 同时预测测试样本及其邻居，再通过加权多数投票聚合，稳定提升分类准确率。

**[The AI Gap: How Socioeconomic Status Affects Language Technology Interactions](others/the_ai_gap_how_socioeconomic_status_affects_language_technology_interactions.md)**

:   通过对1000名不同社会经济地位(SES)用户的大规模调查和6482条真实LLM prompts的分析，揭示了高低SES群体在语言技术使用频率、交互方式和话题选择上存在显著系统性差异，呼吁开发更具包容性的NLP技术以缩小AI鸿沟。

**[The Harmonic Structure of Information Contours](others/the_harmonic_structure_of_information_contours.md)**

:   提出 Harmonic Surprisal (HS) 假说——文本中 surprisal 曲线呈周期性波动且周期与语篇结构（EDU/句子/段落）对齐，用带时间缩放的谐波回归检验，在 6 种语言上发现一致的周期模式，精化了经典的 Uniform Information Density 假说。

**[The Hidden Attention of Mamba Models](others/the_hidden_attention_of_mamba_models.md)**

:   揭示了Mamba（选择性状态空间模型S6）可以被重新表述为一种隐式的因果自注意力机制，并基于此提出了适用于Mamba模型的注意力可视化和可解释性方法（Attention Rollout和Mamba-Attribution），证明其可解释性指标与Transformer相当。

**[The Knowledge Microscope: Features as Better Analytical Lenses than Neurons](others/the_knowledge_microscope_features_as_better_analytical_lenses_than_neurons.md)**

:   本文通过系统实验验证了 SAE（稀疏自编码器）分解出的特征（features）在知识表达影响力、可解释性、单义性（monosemanticity）三个维度上全面优于传统神经元（neurons）作为分析单元，并提出首个基于 feature 的模型编辑方法 FeatureEdit，在隐私知识擦除任务上大幅超越神经元方法。

**[The Noisy Path from Source to Citation: Measuring How Scholars Engage with Past Research](others/the_noisy_path_from_source_to_citation_measuring_how_scholars_engage_with_past_r.md)**

:   构建大规模计算流水线量化学术引用的忠实度（fidelity），分析 1300 万引用句对揭示了影响引用忠实度的关键因素，并通过准因果实验证实了"电话效应"——低忠实度中间引用会导致后续引用进一步失真。

**[The Time Scale of Redundancy between Prosody and Linguistic Context](others/the_time_scale_of_redundancy_between_prosody_and_linguistic_context.md)**

:   本文系统研究了韵律特征（如音高、响度、时长等）与语言上下文之间冗余性的时间尺度，发现韵律与过去上下文的冗余性跨越较长时间尺度（3-8个词），而与未来上下文的冗余性仅限于短时间尺度（1-2个词），揭示了韵律在语音交流中帮助整合过去信息和预测即将出现的词汇的双重作用。

**[Theorem Prover as a Judge for Synthetic Data Generation](others/theorem_prover_as_a_judge_for_synthetic_data_generation.md)**

:   提出 TP-as-a-Judge 框架，利用 Lean 定理证明器验证 LLM 生成的中间推理步骤，结合迭代自动形式化和基于定理证明器反馈的强化学习（RLTPF），仅用 3,508 个样本就在多个数学推理基准上取得了显著提升。

**[Theoretical Guarantees for Minimum Bayes Risk Decoding](others/theoretical_guarantees_for_minimum_bayes_risk_decoding.md)**

:   本文首次为最小贝叶斯风险（MBR）解码提供了严格的理论收敛性保证，证明了在参考假设集大小为 $n$ 时，MBR 解码以 $O(n^{-1/2})$ 的速率逼近最优解，并与 MAP 解码进行了理论比较，表明 MBR 在多种场景下收敛更快。

**[Learning to Reason Over Time: Timeline Self-Reflection for Temporal Reasoning](others/tiser_timeline_self_reflection_temporal.md)**

:   提出 TISER 框架，通过"推理→时间线构建→自反思→答案生成"四阶段管道实现LLM时间推理的test-time scaling，配合合成推理轨迹数据微调，让 7B 开源模型在多个时间推理基准上超越 GPT-4，在TGQA等任务上达到 SOTA。

**[Tokenisation is NP-Complete](others/tokenisation_is_np-complete.md)**

:   证明了分词问题（tokenisation）的两种变体——直接分词和自底向上分词——都是 NP 完全的，通过从 max-2-SAT 问题多项式时间归约实现，这意味着不可能找到高效的最优分词算法，BPE 等近似方法是合理选择。

**[Towards Comprehensive Argument Analysis in Education: Dataset, Tasks, and Method](others/towards_comprehensive_argument_analysis_in_education_dataset_tasks_and_method.md)**

:   本文针对中文高中议论文，提出包含纵向（论证关系）和横向（话语关系）两个维度共 14 种细粒度论证关系类型的标注方案，并在论证成分检测、关系预测和自动评分三个任务上建立了全面的 benchmark。

**[Towards Style Alignment in Cross-Cultural Translation](others/towards_style_alignment_in_cross-cultural_translation.md)**

:   本文首次将"风格对齐"定义为跨文化翻译的核心目标，系统揭示了 LLM 翻译中的风格中性化偏差和英语中心偏差，并提出 RASTA 方法在嵌入空间中学习文化对齐映射来检索风格匹配的少样本示例，在不降低翻译质量的前提下将风格对齐度提升最高 56%。

**[Towards Text-Image Interleaved Retrieval](others/towards_text-image_interleaved_retrieval.md)**

:   定义文本-图像交错检索（TIIR）新任务，构建基于 wikiHow 的首个 TIIR 基准数据集（155K 文档、7654 测试对），并提出 Matryoshka Multimodal Embedder（MME）通过多粒度视觉 token 压缩解决 MLLM 中视觉 token 过多导致的效率和语义偏差问题，大幅提升检索性能。

**[Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis](others/tree-of-debate_multi-persona_debate_trees_elicit_critical_thinking_for_scientifi.md)**

:   提出Tree-of-Debate (ToD)框架，将科学论文转化为LLM persona进行树结构化辩论，通过自我审议、迭代检索和主持人引导的层级子话题扩展，生成细粒度、上下文化的论文对比摘要，在领域专家评估中显著优于基线方法。

**[TROVE: A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Classification](others/trove_a_challenge_for_finegrained_text.md)**

:   提出TROVE文本溯源挑战，将目标文本中每个句子追溯到源文档中的具体源句，并分类其细粒度关系（引用、压缩、推理等），覆盖多文档和长文档场景。

**[Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](others/tuna_temporal_understanding.md)**

:   Tuna 构建了 1000 个时间密集短视频的细粒度多维标注数据集，配套字幕评测（事件拆分→匹配→关系分类）和时序问答两个任务，系统性地暴露了当前视频 LMM 在动态时序理解上的弱点。

**[Understanding Common Ground Misalignment in Goal-Oriented Dialog: A Case-Study with Ubuntu Chat Logs](others/understanding_common_ground_misalignment_in_goal-oriented_dialog_a_case-study_wi.md)**

:   本文通过在 Ubuntu IRC 技术支持对话中标注"对话摩擦"（conversational friction），实证揭示了共识基础（common ground）的失配与任务成功率之间的显著关联，并发现 LLM 能识别显式的对话摩擦但难以处理需要语用或领域推理的隐式摩擦。

**[Understanding Cross-Domain Adaptation in Low-Resource Topic Modeling](others/understanding_cross-domain_adaptation_in_low-resource_topic_modeling.md)**

:   首次将领域自适应形式化引入低资源主题建模，推导有限样本泛化上界指导方法设计，提出 DALTA 框架通过共享编码器、领域专用解码器和对抗对齐实现跨领域主题知识的选择性迁移。

**[Uni-Retrieval: A Multi-Style Retrieval Framework for STEM's Education](others/uni-retrieval_a_multi-style_retrieval_framework_for_stems_education.md)**

:   本文提出面向 STEM 教育场景的多风格多模态检索任务和数据集 SER（24,000+ 查询对），以及基于 Prompt Bank 的轻量检索模型 Uni-Retrieval，通过原型学习提取查询风格特征并动态选择提示向量来增强不同风格（文本、草图、艺术、低分辨率、语音）的检索性能，在 STEM 教育检索和传统检索数据集上均超越已有方法。

**[Unifying Language Agent Algorithms with Graph-based Orchestration Engine for Reproducible Agent Research](others/unifying_language_agent_algorithms_with_graph-based_orchestration_engine_for_rep.md)**

:   提出 AGORA 框架，通过 DAG 图编排引擎将 CoT、ReAct、ToT、RAP 等 10 种主流 Agent 推理算法统一为可插拔的 Operator 模块，在数学推理和多模态任务上系统比较后发现：简单的 CoT 方法在准确率和成本效益上往往优于复杂算法，而一句提示语改动就能带来 90% 的性能飞跃。

**[Unique Hard Attention: A Tale of Two Sides](others/unique_hard_attention_a_tale_of_two_sides.md)**

:   本文证明在有限精度transformer中，leftmost unique hard attention (UHA)严格弱于rightmost UHA，前者等价于线性时序逻辑片段LTL[$\Diamond^-$]（即部分有序有限自动机），并与soft attention transformer表达能力等价，从而精确刻画了注意力方向性对transformer表达力的影响。

**[Unlocking Speech Instruction Data Potential with Query Rewriting](others/unlocking_speech_instruction_data_potential_with_query_rewriting.md)**

:   提出基于多LLM知识融合的查询重写框架与多智能体标注验证方法，将超出TTS词汇范围的文本指令重写为适合语音合成的形式，使语音指令数据可用率从72%提升至93%，为端到端大型语音语言模型(LSLM)构建高质量语音指令数据集。

**[USDC: A Dataset of User Stance and Dogmatism in Long Conversations](others/usdc_a_dataset_of_underlineuser_underlinestance_and_underlinedogmatism_in_long_u.md)**

:   构建 USDC——首个用户级长对话立场和教条主义数据集，764 个多用户 Reddit 对话（22 子版块），用 {Mistral Large, GPT-4} × {zero/one/few-shot} 共 6 设置多数投票标注立场(5级)+教条程度(4级)，并用 7 个 SLM 微调/指令微调建立基线。

**[Using Shapley Interactions to Understand How Models Use Structure](others/using_shapley_interactions_to_understand_how_models_use_structure.md)**

:   利用Shapley Taylor交互指数（STII）跨模态（文本+语音）系统分析语言模型如何通过非线性交互编码句法结构、非组合语义和语音协同发音，发现自回归模型在句法编码上显著优于遮蔽模型。

**[Using Source-Side Confidence Estimation for Reliable Translation into Unfamiliar Languages](others/using_source-side_confidence_estimation_for_reliable_translation_into_unfamiliar.md)**

:   提出一种基于梯度归因的源端置信度估计方法，通过测量输出序列对源端嵌入的敏感度来识别可能误译的源端词汇，无需词对齐，在误译检测任务上显著优于传统对齐方法。

**[Value Residual Learning](others/value_residual_learning.md)**

:   提出 ResFormer 和 SVFormer，通过在注意力机制中引入第一层 Value 向量到后续层的残差连接，增强初始 token 级信息在深层网络中的传播，以比标准 Transformer 少 16.11% 的参数和 20.3% 的训练数据达到同等性能，SVFormer 还能减少近一半 KV 缓存。

**[VAQUUM: Are Vague Quantifiers Grounded in Visual Data?](others/vaquum_are_vague_quantifiers_grounded_in_visual_data.md)**

:   本文发布了VAQUUM数据集（20,300条人类评分，1,089张图片），系统评估视觉语言模型在模糊量词（few/many等）使用上与人类的一致性，发现VLM像人类一样受物体数量影响，但不同评估范式下模型表现差异大，表明判断和生成模糊量词依赖不同认知过程。

**[Verbosity-Aware Rationale Reduction: Effective Reduction of Redundant Rationale](others/verbosity-aware_rationale_reduction_effective_reduction_of_redundant_rationale_v.md)**

:   提出 VARR 框架，以句子为单位并利用基于似然度的"冗余度（verbosity）"标准识别和移除推理路径中的冗余句子，在多种推理任务上平均提升 7.71% 准确率同时减少 19.87% 的 token 生成。

**[Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction](others/visual_cues_enhance_predictive_turn-taking_for_two-party_human_interaction.md)**

:   提出 MM-VAP 多模态预测性话轮转换模型，将面部表情、头部姿态和注视方向等视觉线索引入语音预测模型，在视频会议语料上将 hold/shift 预测准确率从 79% 提升至 84%。

**[Well Begun is Half Done: Low-resource Preference Alignment by Weak-to-Strong Decoding](others/well_begun_is_half_done_low-resource_preference_alignment_by_weak-to-strong_deco.md)**

:   提出 Weak-to-Strong Decoding (WSD) 框架，利用一个小型对齐模型为大型基座模型起草对齐的开头，再由大模型续写，以低资源方式实现偏好对齐且不产生 alignment tax。

**[What is Stigma Attributed to? A Theory-Grounded, Expert-Annotated Interview Corpus for Demystifying Mental-Health Stigma](others/what_is_stigma_attributed_to_a_theory-grounded_expert-annotated_interview_corpus.md)**

:   基于归因理论构建了一个专家标注的心理健康污名访谈语料库（4,141 条片段、684 名参与者），涵盖 7 种细粒度污名类型和社会文化背景信息，并以此为基准评测了多种 SOTA 神经模型在污名检测任务上的表现与挑战。

**[What Matters in Evaluating Book-Length Stories? A Systematic Study of Long Story Evaluation](others/what_matters_in_evaluating_book-length_stories_a_systematic_study_of_long_story_.md)**

:   本文系统研究了书籍级长篇故事（>100K tokens）的自动评估问题，构建了首个大规模长篇故事评估基准LongStoryEval（600本新出版小说、340K条读者评论），提出分层评价标准体系，比较三种评估策略的有效性，并训练了专用评估模型NovelCritique-8B，在与人类评分的对齐度上超越GPT-4o。

**[When to Speak, When to Abstain: Contrastive Decoding with Abstention](others/when_to_speak_when_to_abstain.md)**

:   提出 CDA（Contrastive Decoding with Abstention），一种免训练解码方法，通过熵校准的不确定性估计让 LLM 在参数/上下文知识可用时生成正确回答、在两者都不可靠时主动弃权，覆盖全部四种知识可用性场景。

**[Why Are Positional Encodings Nonessential for Deep Autoregressive Transformers? Revisiting a Petroglyph](others/why_are_positional_encodings_nonessential_for_deep_autoregressive_transformers_r.md)**

:   重新阐释并溯源一个 pre-LLM 时代已知但被遗忘的结论——多层自回归 Transformer 语言模型无需显式位置编码即可区分排列序列，因为级联的（排列不变的）集合处理器在因果掩码下集体展现出完全位置敏感性；同时反思了 LLM 时代的知识断层和引用偏差。

**[Words of Warmth: Trust and Sociability Norms for over 26k English Words](others/words_of_warmth_trust_and_sociability_norms_for_over_26k_english_words.md)**

:   通过严格的众包标注流程构建了首个大规模词汇-温暖（Warmth）、信任（Trust）和社交性（Sociability）关联词典（覆盖 26k+ 英语单词），并通过儿童词汇习得分析和社交媒体刻板印象案例研究，展示了该资源在社会认知研究中的广泛价值。

**[X-Turing: Towards an Enhanced and Efficient Turing Test for Long-Term Dialogue Agents](others/xturing_enhanced_turing_test.md)**

:   提出 X-Turing 框架，通过引入 burst 对话模式和伪对话生成技术来增强和高效化图灵测试，能够评估 LLM 在长期对话中的人类模仿能力，发现 LLM 随着对话轮次增加表现显著下降。

**[You need to MIMIC to get FAME: Solving Meeting Transcript Scarcity with Multi-Agent Conversations](others/you_need_to_mimic_to_get_fame_solving_meeting_transcript_scarcity_with_a_multi-a.md)**

:   提出 MIMIC 框架，通过多智能体辩论模拟生成合成会议转录，构建了包含 800 场会议的 FAME 数据集（500 英语 + 300 德语），并设计了基于心理学的行为真实性评估框架。

**[Your Model is Overconfident, and Other Lies We Tell Ourselves](others/your_model_is_overconfident_and_other_lies_we_tell_ourselves.md)**

:   通过对 29 个模型在 ChaosNLI 和 DynaSent 数据集上的全面分析，揭示了标注者分歧、训练动态、模型置信度等数据复杂度指标之间存在相关性但非线性非单调的关系，挑战了"模型不确定性 ≈ 人类分歧"这一常见假设。

**[Zero-Shot Conversational Stance Detection: Dataset and Approaches](others/zero-shot_conversational_stance_detection_dataset_and_approaches.md)**

:   构建了首个零样本多轮多方对话立场检测数据集 ZS-CSD（280 个目标、17,063 条对话样本），并提出 SITPCL 模型，结合说话者交互编码器与目标感知原型对比学习，在零样本对话立场检测中取得 SOTA（F1-macro 43.81%）。

</div>