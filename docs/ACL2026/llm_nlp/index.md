---
title: >-
  ACL2026 LLM / NLP方向36篇论文解读
description: >-
  36篇ACL2026的 LLM / NLP 方向论文解读，涵盖 LLM、Agent、少样本学习、推理、人脸/视线、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM / NLP

**💬 ACL2026** · **36** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (9)](../../CVPR2026/llm_nlp/index.md) · [🔬 ICLR2026 (39)](../../ICLR2026/llm_nlp/index.md) · [🤖 AAAI2026 (38)](../../AAAI2026/llm_nlp/index.md) · [🧠 NeurIPS2025 (53)](../../NeurIPS2025/llm_nlp/index.md) · [📹 ICCV2025 (8)](../../ICCV2025/llm_nlp/index.md) · [🧪 ICML2025 (28)](../../ICML2025/llm_nlp/index.md)

🔥 **高频主题：** LLM ×16 · Agent ×4 · 少样本学习 ×2 · 推理 ×2 · 人脸/视线 ×2

**[A Study of LLMs' Preferences for Libraries and Programming Languages](a_study_of_llms39_preferences_for_libraries_and_programming_languages.md)**

:   首次系统研究8个LLM在代码生成中对库和编程语言的偏好行为，发现LLM严重偏好NumPy等流行库（45%的使用不必要）和Python语言（58%的高性能任务仍选Python），且自然语言推荐与实际代码选择不一致。

**[Adam's Law: Textual Frequency Law on Large Language Models](adam39s_law_textual_frequency_law_on_large_language_models.md)**

:   本文提出"文本频率定律"（TFL），发现当语义相同时，使用更高频率的文本表达来提示或微调LLM能获得更好效果，并设计了频率蒸馏和课程训练策略来进一步利用该规律。

**[AlphaContext: An Evolutionary Tree-based Psychometric Context Generator for Creativity Assessment](alphacontext_an_evolutionary_tree-based_psychometric_context_generator_for_creat.md)**

:   提出 AlphaContext，一个基于进化树的心理测量情境生成器，通过 HyperTree 大纲规划、MCTS 逐句生成、MAP-Elites 多样性优化和评估引导迭代精炼四个模块，自动生成用于创造力评估的高质量长文本情境，在 7 个评估维度上平均超越竞争方法 8%。

**[An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal](an_existence_proof_for_neural_language_models_that_can_explain_garden-path_effec.md)**

:   通过在花园路径句上微调神经语言模型，证明了存在一个神经 LM 能够通过惊奇度（surprisal）同时解释花园路径效应和自然阅读时间，为惊奇度理论提供了存在性证明。

**[Are Emotion and Rhetoric Neurons in LLM? Neuron Recognition and Adaptive Masking for Emotion-Rhetoric Prediction Steering](are_emotion_and_rhetoric_neurons_in_llm_neuron_recognition_and_adaptive_masking_.md)**

:   系统研究LLM中情感和修辞神经元的表征机制及其内在关联，提出结合多维筛选的神经元识别框架和自适应遮蔽验证方法，实现了情感/修辞预测的定向诱导和修辞神经元辅助情感识别。

**[Automatic Combination of Sample Selection Strategies for Few-Shot Learning](automatic_combination_of_sample_selection_strategies_for_few-shot_learning.md)**

:   本文提出 ACSESS 方法，通过前向选择、后向选择和 Datamodels 三种机制自动识别互补的样本选择策略并加权组合，在 23 种策略、5 个 ICL 模型和 3 种梯度少样本学习方法、6 个文本和 8 个图像数据集上验证了组合策略一致优于单一策略和 ICL 专用基线。

**[ChatHLS: Towards Systematic Design Automation and Optimization for High-Level Synthesis](chathls_towards_systematic_design_automation_and_optimization_for_high-level_syn.md)**

:   ChatHLS 提出了一个多智能体 HLS 设计框架，通过 HLSTuner（QoR 感知推理优化指令选择）和 HLSFixer（分层反馈增强的调试框架）两个核心组件，结合自进化错误用例扩展机制（VODA），在 HLS-C 生成成功率和硬件性能优化上显著超越基线。

**[CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models](costomcausal-oriented_steering_for_intrinsic_theory-of-mind_alignment_in_large_l.md)**

:   提出 CoSToM 框架，先用因果追踪定位 LLM 中编码心智理论（ToM）特征的关键层（发现主要在早期层），再通过激活转向在这些层上进行轻量级对齐，使 LLM 在谈判和说服对话中显著提升社会推理质量——从"知道但不会用"变为"知道且会用"。

**[Detoxification for LLM from Dataset Itself](detoxification_for_llm_from_dataset_itself.md)**

:   本文提出 HSPD（层次化语义保留去毒）流水线，通过 SoCD（软对比解码）引导 LLM 定位并重写原始语料中的有毒片段，同时保留语义，生成可直接替换原始数据用于微调的去毒语料——在 GPT2-XL 上将毒性概率从 0.42 降至 0.18，在 LLaMA2-7B、OPT-6.7B 和 Falcon-7B 上也取得了最优去毒效果。

**[DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot NER](diziner_disagreement-guided_instruction_refinement_via_pilot_annotation_simulati.md)**

:   DiZiNER 模拟人类试标注流程：多个异构 LLM 独立标注同一文本，分析模型间分歧来迭代精炼任务指令，在 18 个 NER 基准中的 14 个上达到零样本 SOTA，平均 F1 提升 +8.0，且超越其监督模型 GPT-5 mini。

**[Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models](don39t_adapt_small_language_models_for_tools_adapt_tool_schemas_to_the_models.md)**

:   本文提出 PA-Tool，一种无训练的工具 Schema 优化方法，利用从数据污染检测中借鉴的"尖锐度"（peakedness）信号识别模型预训练中熟悉的命名模式，通过重命名工具组件来对齐小语言模型的内化知识，在 MetaTool 和 RoTBench 上实现最高 17% 的提升，Schema 不对齐错误减少 80%。

**[EvoSpark: Endogenous Interactive Agent Societies for Unified Long-Horizon Narrative Evolution](evospark_endogenous_interactive_agent_societies_for_unified_long-horizon_narrati.md)**

:   EvoSpark 提出一个支持长程叙事演化的多智能体框架，通过分层递归记忆（RSB 做社会认知代谢）、生成式场面调度（GMS 做角色-地点-情节对齐）和涌现角色锚定协议（ECGP 将 LLM 幻觉转化为持久角色）三重设计解决社会记忆堆叠和叙事-空间失谐问题。

**[Expect the Unexpected? Testing the Surprisal of Salient Entities](expect_the_unexpected_testing_the_surprisal_of_salient_entities.md)**

:   本文研究全局显著实体（discourse-level salient entities）与惊异度（surprisal）的关系，通过 70K+ 手工标注的实体提及和新颖的最小对提示方法，发现全局显著实体本身更出人意料（更高 surprisal），但它们系统性地降低周围内容的 surprisal，且该效应随体裁变化——话题连贯性高的文本中效应最强。

**[FastDiSS: Few-step Match Many-step Diffusion Language Model on Sequence-to-Sequence Generation](fastdiss_few-step_match_many-step_diffusion_language_model_on_sequence-to-sequen.md)**

:   本文分析了连续扩散语言模型在少步采样时自条件化信号的不匹配和训练饱和两个瓶颈，提出FastDiSS框架通过自条件化扰动（SCP）和模型感知噪声缩放（MANS）来改善鲁棒性，在6个基准上实现4×-400×加速同时保持质量。

**[Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)**

:   本文提出 Foresight Policy Optimization（FoPO），通过在策略优化中引入对手建模的前瞻修正项，使 LLM 能够显式预见对手行为并据此调整自身策略，在合作（Cooperative RSA）和竞争（Competitive Taboo）两类博弈任务上显著提升策略推理能力，并在跨域 γ-Bench 上取得一致性提升。

**[From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models](from_static_inference_to_dynamic_interaction_a_survey_of_streaming_large_languag.md)**

:   本文首次系统综述流式大语言模型（Streaming LLMs），提出基于数据流和交互并发性的统一定义，将现有方法分为三级递进分类——输出流式（Output-streaming）、顺序流式（Sequential-streaming）和并发流式（Concurrent-streaming），覆盖文本、语音和视频流式场景的方法论和应用。

**[GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning](grass_gradient-based_adaptive_layer-wise_importance_sampling_for_memory-efficien.md)**

:   提出 GRASS 框架，使用均值梯度范数（MGN）作为任务感知和训练阶段感知的层重要性指标，自适应地采样和更新模型层子集进行微调，配合层级优化器状态卸载机制，在平均准确率提升最高 4.38 分的同时减少最高 19.97% 的内存使用。

**[HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction](hcre_llm-based_hierarchical_classification_for_cross-document_relation_extractio.md)**

:   提出 HCRE 模型，通过构建层次化关系树将跨文档关系抽取从大规模关系集的直接分类转化为逐层层次化分类，并设计预测-验证推理策略缓解层间错误传播，在 CodRED 数据集上显著超越 SLM 和 LLM 基线。

**[How Do Answer Tokens Read Reasoning Traces? Self-Reading Patterns in Thinking LLMs](how_do_answer_tokens_read_reasoning_traces_self-reading_patterns_in_thinking_llm.md)**

:   本文发现推理 LLM（如 DeepSeek-R1）在定量推理中存在"良性自读"模式——答案 token 对推理痕迹的注意力呈现前移漂移（沿推理链逐步推进）和语义锚点集中（反复回顾关键步骤），且此模式与正确性强相关；基于此提出 SRQ（自读质量）驱动的免训练激活引导方法，在多个基准上提升准确率最高 2.6%。

**[It's High Time: A Survey of Temporal Question Answering](it39s_high_time_a_survey_of_temporal_question_answering.md)**

:   本文提供了时序问答（TQA）的全面综述，提出了基于语料时间性、问题时间性和模型时间能力三个维度的统一分析框架，系统梳理了从规则管道到 Transformer/LLM 时代的 TQA 方法演进、基准数据集和评估策略，并识别了未来挑战。

**[Iterative Formalization and Planning in Partially Observable Environments](iterative_formalization_and_planning_in_partially_observable_environments.md)**

:   提出 PDDLego+ 框架，让 LLM 在部分可观测环境中迭代地生成和修正 PDDL（规划领域定义语言）表示，通过双层错误修复循环（solver error + simulation error）实现无需微调、无需示例的有效规划。

**[Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models](lost_in_the_prompt_order_revealing_the_limitations_of_causal_attention_in_langua.md)**

:   本文深入研究了大语言模型在多选题问答中对提示组件顺序的敏感性，通过系统性实验排除了训练偏差和记忆衰退假说，揭示了因果注意力掩码是导致 QOC（问题-选项-上下文）顺序性能大幅下降的根本机制。

**[Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness](masked_by_consensus_disentangling_privileged_knowledge_in_llm_correctness.md)**

:   本文通过对比自探针（使用模型自身隐藏状态）和外部探针（使用其他模型隐藏状态）预测正确性的能力，发现"模型间一致性"是掩盖特权知识的关键混淆因子，在消除一致性后揭示了领域特异性的特权知识：事实性任务中存在但数学推理中不存在。

**[Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)**

:   提出 MALMAS，一个记忆增强的 LLM 多智能体系统用于表格数据自动特征生成，通过六个专职 Agent 分工探索不同特征空间维度 + 三级记忆机制（过程/反馈/概念）实现跨轮迭代优化，在 16 个分类和 7 个回归数据集上超越现有基线。

**[MulDimIF: A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models](muldimif_a_multi-dimensional_constraint_framework_for_evaluating_and_improving_i.md)**

:   提出 MulDimIF 多维约束框架，从约束模式（3种）、约束类别（4类13子类）和约束难度（4级）三个维度系统评估 LLM 的指令遵循能力，并通过 GRPO 训练显著提升模型性能，发现改进主要源自注意力模块的参数更新。

**[Not All Animals Are Equal: Metaphorical Framing through Source Domains and Semantic Frames](not_all_animals_are_equal_metaphorical_framing_through_source_domains_and_semant.md)**

:   本文提出首个结合 FrameNet 语义框架和概念隐喻理论（CMT）源域的计算框架 ConceptFrameMet，通过 RoBERTa 多任务模型检测隐喻并预测其语义框架和源域，配合对数似然比统计方法发现话语中显著的隐喻模式，揭示了自由派和保守派在移民话语中使用相同源域但选择不同语义框架来传达截然不同的联想。

**[One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization](one_persona_many_cues_different_results_how_sociodemographic_cues_impact_llm_per.md)**

:   本文系统比较了 6 种常用的人物画像提示方式（姓名/显式提及/对话历史各两种变体）在 7 个 LLM 和 4 个任务上的效果，发现虽然平均响应跨提示方式高度相关，但不同提示方式产生的人物画像间差异显著不同，过于显式的提示导致更强的个性化偏差，警示不应基于单一提示方式得出偏差结论。

**[Please Refuse to Answer Me: Mitigating Over-Refusal in LLMs via Adaptive Contrastive Decoding](please_refuse_to_answer_me_mitigating_over-refusal_in_large_language_models_via_.md)**

:   本文提出 AdaCD（自适应对比解码），通过比较极端安全提示下和无提示下的 token 分布差异提取拒绝 token 分布，再根据一致性比率动态决定增强或抑制拒绝行为，在降低过度拒绝 10.35% 的同时提升恶意查询拒绝率 0.13%。

**[Revisiting Non-Verbatim Memorization in Large Language Models: The Role of Entity Surface Forms](revisiting_non-verbatim_memorization_in_large_language_models_the_role_of_entity.md)**

:   本文通过构建 RedirectQA 数据集（利用 Wikipedia 重定向信息将同一实体关联到多种表面形式），系统研究了 LLM 的非逐字记忆如何受实体命名变体的影响，发现事实记忆既非纯粹依赖特定表面形式也非完全表面无关，且实体级频率在表面频率之外仍有独立贡献。

**[Route to Rome Attack: Directing LLM Routers to Expensive Models via Adversarial Suffixes](route_to_rome_attack_directing_llm_routers_to_expensive_models_via_adversarial_s.md)**

:   本文提出 R2A（Route to Rome Attack），通过在黑盒设置下构建混合集成代理路由器并优化通用对抗后缀，将 LLM 路由器的路由决策从廉价弱模型导向昂贵强模型——在 7 个开源路由器和 2 个商用路由器（GPT-5-Auto、OpenRouter）上平均攻击成功率提升 49%，推理成本增加 2.7-2.9 倍。

**[Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models](style_amnesia_investigating_speaking_style_degradation_and_mitigation_in_multi-t.md)**

:   发现口语语言模型（SLMs）在多轮对话中无法维持初始指定的说话风格（情感、口音、音量、语速），称之为"风格遗忘"现象，并通过注意力分析揭示其成因（注意力衰减），提出显式回忆过程作为缓解手段。

**[The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models](the_model_agreed_but_didn39t_learn_diagnosing_surface_compliance_in_large_langua.md)**

:   提出 SA-MCQ 诊断框架揭示知识编辑中的"表面合规"现象——编辑器在标准基准上达到高分但并未真正覆写内部信念，模型在判别式自评中会回退到原始参数记忆，递归编辑还会累积表征残留导致认知不稳定。

**[Think in Sentences: Explicit Sentence Boundaries Enhance Language Model's Capabilities](think_in_sentences_explicit_sentence_boundaries_enhance_language_model39s_capabi.md)**

:   本文提出在 LLM 输入中的句子边界处插入分隔符标记，通过 ICL 和 SFT 两种方式实现"逐句思考"的推理范式，在 7B 到 600B 模型上取得一致提升（GSM8k +7.7%，DROP +12.5%），且几乎不增加额外计算开销。

**[Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration](towards_robust_real-world_spreadsheet_understanding_with_multi-agent_multi-forma.md)**

:   提出 SpreadsheetAgent，一种两阶段多智能体框架，通过代码执行、视觉和 LaTeX 三种格式的渐进式区域读取与交叉验证，在不超出 LLM 上下文限制的前提下实现鲁棒的真实世界电子表格理解。

**[Why Did Apple Fall: Evaluating Curiosity in Large Language Models](why_did_apple_fall_evaluating_curiosity_in_large_language_models.md)**

:   本文提出首个系统评估 LLM 好奇心行为的心理学启发框架，结合问卷自评和行为实验发现 LLM 展现出好奇心般的行为模式但并非内在特质，并设计好奇心驱动的提问管道证明模拟好奇行为可提升下游推理性能。

**[XtraGPT: Context-Aware and Controllable Academic Paper Revision via Human-AI Collaboration](xtragpt_context-aware_and_controllable_academic_paper_revision_via_human-ai_coll.md)**

:   本文提出 XtraGPT——首个面向学术论文修改的开源 LLM 套件（1.5B-14B），通过在 7,000 篇顶会论文和 140,000 个标准引导的指令-修改对上微调，实现上下文感知的段落级可控修改，7B 版本匹配 GPT-4o-mini，14B 版本超越 GPT-4o-mini，人类评估显示修改后论文预测评分平均提升 0.65 分。
