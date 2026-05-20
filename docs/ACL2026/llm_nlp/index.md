---
title: >-
  ACL2026 LLM / NLP方向50篇论文解读
description: >-
  50篇ACL2026的 LLM / NLP 方向论文解读，涵盖 LLM、Agent、扩散模型、推理、少样本学习、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ACL2026"
  - "LLM / NLP"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "Agent"
  - "扩散模型"
  - "推理"
  - "少样本学习"
  - "对齐/RLHF"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM / NLP

**💬 ACL2026** · **50** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (4)](../../ICML2026/llm_nlp/index.md) · [📷 CVPR2026 (9)](../../CVPR2026/llm_nlp/index.md) · [🔬 ICLR2026 (35)](../../ICLR2026/llm_nlp/index.md) · [🤖 AAAI2026 (32)](../../AAAI2026/llm_nlp/index.md) · [🧠 NeurIPS2025 (50)](../../NeurIPS2025/llm_nlp/index.md) · [📹 ICCV2025 (6)](../../ICCV2025/llm_nlp/index.md)

🔥 **高频主题：** LLM ×16 · Agent ×3 · 扩散模型 ×3 · 推理 ×3 · 少样本学习 ×2

**[A Study of LLMs' Preferences for Libraries and Programming Languages](a_study_of_llms39_preferences_for_libraries_and_programming_languages.md)**

:   首次系统研究8个LLM在代码生成中对库和编程语言的偏好行为，发现LLM严重偏好NumPy等流行库（45%的使用不必要）和Python语言（58%的高性能任务仍选Python），且自然语言推荐与实际代码选择不一致。

**[Adam's Law: Textual Frequency Law on Large Language Models](adam39s_law_textual_frequency_law_on_large_language_models.md)**

:   本文提出"文本频率定律"（TFL），发现当语义相同时，使用更高频率的文本表达来提示或微调LLM能获得更好效果，并设计了频率蒸馏和课程训练策略来进一步利用该规律。

**[AlphaContext: An Evolutionary Tree-based Psychometric Context Generator for Creativity Assessment](alphacontext_an_evolutionary_tree-based_psychometric_context_generator_for_creat.md)**

:   提出 AlphaContext，一个基于进化树的心理测量情境生成器，通过 HyperTree 大纲规划、MCTS 逐句生成、MAP-Elites 多样性优化和评估引导迭代精炼四个模块，自动生成用于创造力评估的高质量长文本情境，在 7 个评估维度上平均超越竞争方法 8%。

**[An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal](an_existence_proof_for_neural_language_models_that_can_explain_garden-path_effec.md)**

:   通过在花园路径句上微调神经语言模型，证明了存在一个神经 LM 能够通过惊奇度（surprisal）同时解释花园路径效应和自然阅读时间，为惊奇度理论提供了存在性证明。

**[Automatic Combination of Sample Selection Strategies for Few-Shot Learning](automatic_combination_of_sample_selection_strategies_for_few-shot_learning.md)**

:   本文提出 ACSESS 方法，通过前向选择、后向选择和 Datamodels 三种机制自动识别互补的样本选择策略并加权组合，在 23 种策略、5 个 ICL 模型和 3 种梯度少样本学习方法、6 个文本和 8 个图像数据集上验证了组合策略一致优于单一策略和 ICL 专用基线。

**[Big AI is Accelerating the Metacrisis: What Can We Do?](big_ai_is_accelerating_the_metacrisis_what_can_we_do.md)**

:   Steven Bird 在这篇 ACL 2026 立场论文里论证："Big AI"（少数巨头驱动的工业化 LLM 工程）正在同时加速 3 大相互纠缠的危机——**生态危机 / 意义危机 / 语言危机**——而 ACL 作为最大 LLM 研究发表方，必须从"个人合规"转向"职业共同体集体行动"，并提出 7 项面向 ACL 的具体改革建议（重申公共利益优先、抵御 corporate capture、保护批判性 NLP、设立 NLP policy track 等）。

**[C-World: A Computer Use Agent Environment Creator](c-world_a_computer_use_agent_environment_creator.md)**

:   作者将"agent 环境"形式化为 Action / Task / Transition / Reward 四元组并实现为 C-World：用 5,571 个真实 MCP 工具 + 自动任务合成 + state controller 扰动 + 双信号 reward 提供高保真评测，又用一个"World Engine"在无 live API 下模拟工具响应实现可规模化训练；评测 9 个前沿 LLM 发现"规划普遍强、执行普遍弱"，仅用 1,170 条 C-World 轨迹微调即可超过用 119k 样本训练的 baseline。

**[Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future](can_ai_be_a_good_peer_reviewer_a_survey_of_peer_review_process_evaluation_and_th.md)**

:   作者系统综述了 LLM 时代 AI 辅助 peer review 全流程的方法：把"review 生成"分为 fine-tuning / agent / RL / 生成增强 四大范式，把"after-review"分为 rebuttal / meta-review / paper revision 三类，再给出"human / reference-based / LLM-based / aspect-oriented"四象限评测分类法，最后从 novelty、自动评测、跨域、多模态、伦理 6 个方向讨论未来。

**[CAST: Achieving Stable LLM-based Text Analysis for Data Analytics](cast_achieving_stable_llm-based_text_analysis_for_data_analytics.md)**

:   提出CAST框架，通过算法提示（Algorithmic Prompting）和先思考后输出（Thinking-before-Speaking）两种机制约束LLM的潜在推理路径，显著提升文本摘要和标注任务的运行间稳定性，同时不损失输出质量。

**[Characterizing the Expressivity of Local Attention in Transformers](characterizing_the_expressivity_of_local_attention_in_transformers.md)**

:   作者用线性时序逻辑（LTL）作为统一刻画工具，严格证明 global-only Transformer ↔ $\mathrm{LTL}[\mathrm{P}]$、$k$-local-only ↔ $\mathrm{LTL}[\mathrm{Y}^{\leq k}]$、global+local 混合 ↔ $\mathrm{LTL}[\mathrm{P}, \mathrm{Y}^{\leq k}]$，并由此证明 **local 与 global 表达力互不包含**、混合严格更强、**1-local 是 local 家族里表达力最强**，最后在合成正则语言和 WikiText-2 上经验验证理论预测。

**[CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models](costomcausal-oriented_steering_for_intrinsic_theory-of-mind_alignment_in_large_l.md)**

:   提出 CoSToM 框架，先用因果追踪定位 LLM 中编码心智理论（ToM）特征的关键层（发现主要在早期层），再通过激活转向在这些层上进行轻量级对齐，使 LLM 在谈判和说服对话中显著提升社会推理质量——从"知道但不会用"变为"知道且会用"。

**[DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)**

:   提出 DeCoVec（Decoding Space based Task Vector），一个无训练、非侵入式的框架，通过对比 few-shot 和 zero-shot prompt 的输出 logit 分布差异构建解码空间中的任务向量，注入解码过程引导生成，在 TruthfulQA、Math-500 和 AQUA-RAT 上比标准 few-shot 基线平均提升高达 5.50 准确率。

**[Dual Alignment Between Language Model Layers and Human Sentence Processing](dual_alignment_between_language_model_layers_and_human_sentence_processing.md)**

:   作者用 logit-lens 把 GPT-2/Pythia/OPT 共 19 个 LM 的每一层都解出"内部 surprisal"，发现一个反直觉的"双重对齐"：在自然阅读语料上**浅层**的 surprisal 最像人；但在 garden-path / NPS / NPZ / RC / Attachment 等**句法挑战句**上反而**深层**才像人，对应人类"shallow 默认 + 困难时切换到 deep 重分析"的双机制阅读模型——并由此提出用浅深层 surprisal 差（KL/JS）作为"层间预测更新量"来当 reading-time 的补充特征。

**[EVE: A Domain-Specific LLM Framework for Earth Intelligence](eve_a_domain-specific_llm_framework_for_earth_intelligence.md)**

:   本文提出 EVE——欧空局 (ESA) Φ-lab 主导的首个面向 Earth Observation / Earth Sciences 的开源端到端 LLM 框架，包含 24B 领域适配的 EVE-Instruct（基于 Mistral Small 3.2 + 10.7B 合成 token 的 IFT/CPT 交替微调 + 10 个 checkpoint 融合）、首批 5693 条人工 EO 评测 benchmark、RAG + 幻觉检测 pipeline，已在 6 个月 pilot 中服务 350 位用户。

**[EvoSpark: Endogenous Interactive Agent Societies for Unified Long-Horizon Narrative Evolution](evospark_endogenous_interactive_agent_societies_for_unified_long-horizon_narrati.md)**

:   EvoSpark 提出一个支持长程叙事演化的多智能体框架，通过分层递归记忆（RSB 做社会认知代谢）、生成式场面调度（GMS 做角色-地点-情节对齐）和涌现角色锚定协议（ECGP 将 LLM 幻觉转化为持久角色）三重设计解决社会记忆堆叠和叙事-空间失谐问题。

**[Expect the Unexpected? Testing the Surprisal of Salient Entities](expect_the_unexpected_testing_the_surprisal_of_salient_entities.md)**

:   本文研究全局显著实体（discourse-level salient entities）与惊异度（surprisal）的关系，通过 70K+ 手工标注的实体提及和新颖的最小对提示方法，发现全局显著实体本身更出人意料（更高 surprisal），但它们系统性地降低周围内容的 surprisal，且该效应随体裁变化——话题连贯性高的文本中效应最强。

**[FastDiSS: Few-step Match Many-step Diffusion Language Model on Sequence-to-Sequence Generation](fastdiss_few-step_match_many-step_diffusion_language_model_on_sequence-to-sequen.md)**

:   本文分析了连续扩散语言模型在少步采样时自条件化信号的不匹配和训练饱和两个瓶颈，提出FastDiSS框架通过自条件化扰动（SCP）和模型感知噪声缩放（MANS）来改善鲁棒性，在6个基准上实现4×-400×加速同时保持质量。

**[Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)**

:   本文提出 Foresight Policy Optimization（FoPO），通过在策略优化中引入对手建模的前瞻修正项，使 LLM 能够显式预见对手行为并据此调整自身策略，在合作（Cooperative RSA）和竞争（Competitive Taboo）两类博弈任务上显著提升策略推理能力，并在跨域 γ-Bench 上取得一致性提升。

**[From Fallback to Frontline: When Can LLMs be Superior Annotators of Human Perspectives?](from_fallback_to_frontline_when_can_llms_be_superior_annotators_of_human_perspec.md)**

:   本文把"perspective-taking (PT)"这个长期被视为人类专属的主观标注任务重新形式化为「对潜在群体均值 $f^*(x,g)$ 的统计估计问题」，用偏差-方差-相关性三项分解证明 LLM 在低预算 / 群体宽泛 / out-group 场景下不只是廉价替代品，而是**比 in-group 人类标注者更优**的估计器，并发现"打开 reasoning 反而变差"的 reasoning paradox。

**[From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models](from_static_inference_to_dynamic_interaction_a_survey_of_streaming_large_languag.md)**

:   本文首次系统综述流式大语言模型（Streaming LLMs），提出基于数据流和交互并发性的统一定义，将现有方法分为三级递进分类——输出流式（Output-streaming）、顺序流式（Sequential-streaming）和并发流式（Concurrent-streaming），覆盖文本、语音和视频流式场景的方法论和应用。

**[Generative Floor Plan Design with LLMs via Reinforcement Learning with Verifiable Rewards](generative_floor_plan_design_with_llms_via_reinforcement_learning_with_verifiabl.md)**

:   作者把 RPLAN 中 8 万套真实公寓户型转成 JSON 多边形格式，用 Llama-3.3-70B-Instruct 做两阶段训练（SFT + GRPO with verifiable rewards：connectivity + total-area 奖励，重叠/解析失败硬置零），让 LLM 输出能同时满足 bubble diagram 拓扑约束和数值面积约束的 CAD-ready 户型，在 8 房间任务上 Compatibility 相比 HouseDiffusion 下降 94%（2.5 → 0.15）。

**[Generative Interfaces for Language Models](generative_interfaces_for_language_models.md)**

:   本文提出 **Generative Interfaces (GenUI)**，让 LLM 不再用单一聊天框回复用户，而是基于"交互流图 + 有限状态机"的结构化中间表示和"自适应奖励驱动的迭代精化"，在线生成一个为查询量身定制的可交互 Web 界面，在 100 条 UIX prompt 上相比 Claude 3.7 聊天 UI 取得 **84%** 的总体偏好胜率。

**[GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning](grass_gradient-based_adaptive_layer-wise_importance_sampling_for_memory-efficien.md)**

:   提出 GRASS 框架，使用均值梯度范数（MGN）作为任务感知和训练阶段感知的层重要性指标，自适应地采样和更新模型层子集进行微调，配合层级优化器状态卸载机制，在平均准确率提升最高 4.38 分的同时减少最高 19.97% 的内存使用。

**[Hot-Start from Pixels: Low-Resolution Visual Tokens for Chinese Language Modeling](hot-start_from_pixels_low-resolution_visual_tokens_for_chinese_language_modeling.md)**

:   把中文字符渲染成 $8\times 8$ 灰度小图喂给 GPT-2 风格解码器做 next-character 预测，最终精度（39.21%）追平 index-based 基线（39.10%），并且在训练早期（0.4% 数据时）就把基线翻一倍以上——展示了"视觉结构"作为汉字建模的天然 hot-start 先验。

**[SteerEval: How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities](how_controllable_are_large_language_models_a_unified_evaluation_across_behaviora.md)**

:   SteerEval 把 LLM 可控性按 Marr 的三层分析框架拆成 L1（表达什么）/L2（怎么表达）/L3（具体落到哪个词），覆盖 Personality、Sentiment、Language Features 三个域共 7560 个 paired sample，系统揭示了"细粒度上现有 steering 方法普遍崩溃"这一关键缺口。

**[How Do Answer Tokens Read Reasoning Traces? Self-Reading Patterns in Thinking LLMs](how_do_answer_tokens_read_reasoning_traces_self-reading_patterns_in_thinking_llm.md)**

:   本文发现推理 LLM（如 DeepSeek-R1）在定量推理中存在"良性自读"模式——答案 token 对推理痕迹的注意力呈现前移漂移（沿推理链逐步推进）和语义锚点集中（反复回顾关键步骤），且此模式与正确性强相关；基于此提出 SRQ（自读质量）驱动的免训练激活引导方法，在多个基准上提升准确率最高 2.6%。

**[Identifying the Periodicity of Information in Natural Language](identifying_the_periodicity_of_information_in_natural_language.md)**

:   本文把信号处理领域的 AutoPeriod 周期检测算法搬到 token-surprisal 序列上，提出 APS（AutoPeriod of Surprisal）能在单文档级别**直接检测**出自然语言信息密度的周期（如 "每 53 个 token 一个周期"），发现人类文本中约 11% 的文档存在严格周期，且 LLM 生成文本的周期性比人类强 2 倍（30% vs 14.8%），为 UID 理论提供了直接证据并给 AI 文本检测提供了可解释特征。

**[Iterative Formalization and Planning in Partially Observable Environments](iterative_formalization_and_planning_in_partially_observable_environments.md)**

:   提出 PDDLego+ 框架，让 LLM 在部分可观测环境中迭代地生成和修正 PDDL（规划领域定义语言）表示，通过双层错误修复循环（solver error + simulation error）实现无需微调、无需示例的有效规划。

**[Leveraging Pretrained Language Models as Energy Functions for Glauber Dynamics Text Diffusion](leveraging_pretrained_language_models_as_energy_functions_for_glauber_dynamics_t.md)**

:   本文用统计物理中的 Glauber 动力学构造离散文本扩散：把预训练 UL2 模型当成"能量函数 / 噪声分布"，每一步把 mask infilling 当 Markov 转移核，训出的 Glauber-UL2 在生成 perplexity 上**首次匹配 GPT-2-M/L 同尺寸 AR 模型**，并在 Sudoku/Zebra 等搜索规划任务上击败 MDLM，同 compute 下 best-of-N 也优于 AR。

**[Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness](masked_by_consensus_disentangling_privileged_knowledge_in_llm_correctness.md)**

:   本文通过对比自探针（使用模型自身隐藏状态）和外部探针（使用其他模型隐藏状态）预测正确性的能力，发现"模型间一致性"是掩盖特权知识的关键混淆因子，在消除一致性后揭示了领域特异性的特权知识：事实性任务中存在但数学推理中不存在。

**[Min-k Sampling: Decoupling Truncation from Temperature Scaling via Relative Logit Dynamics](min-k_sampling_decoupling_truncation_from_temperature_scaling_via_relative_logit.md)**

:   Min-k Sampling 通过分析排序 logit 分布的局部结构来检测"语义悬崖"（高置信候选与低质量尾部噪声的分界点），实现了严格的温度不变性截断，在极端温度下仍保持稳健的推理和创意写作质量。

**[Mind the Gap: How Elicitation Protocols Shape the Stated-Revealed Preference Gap in Language Models](mind_the_gap_how_elicitation_protocols_shape_the_stated-revealed_preference_gap_.md)**

:   作者在 LitmusValues / AIRiskDilemmas 框架上把 forced-choice 升级为 expanded-choice（允许 "Equal Preference" 和 "Depends" 中立选项），系统评测 24 个 LLM 发现：在 stated 侧允许中立可以把 SvR Spearman 相关 ρ 从 ~0.2 大幅提升到 ~0.7（因为过滤掉了模型本来就没立场的弱信号），但 revealed 侧也允许中立反而把 ρ 打到接近 0 甚至负值（很多模型在情境化场景里几乎全选 Depends/Equal），同时验证基于 stated ranking 的 system prompt steering 在 16-value 大集合下普遍不可靠——结论是 **SvR gap 高度依赖 elicitation protocol，偏好评测必须显式建模"无定见"状态**。

**[Model-Agnostic Meta Learning for Class Imbalance Adaptation](model-agnostic_meta_learning_for_class_imbalance_adaptation.md)**

:   本文提出 HAMR（Hardness-Aware Meta-Resample），一个统一的元学习框架，通过双层优化动态估计实例级权重优先处理真正困难的样本，配合邻域感知重采样机制将训练焦点放在困难样本及其语义邻居上，在 6 个不平衡 NLP 数据集上持续超越强基线。

**[MoRI: Learning Motivation-Grounded Reasoning for Scientific Ideation in Large Language Models](mori_learning_motivation-grounded_reasoning_for_scientific_ideation_in_large_lan.md)**

:   把科学 ideation 显式建模为「context → motivation → reasoning → method」的两阶段条件推理任务，在 SFT 冷启动基础上用 GRPO + 一对新型可验证奖励（**熵感知信息增益 EAIG** + **对比语义增益 CSG**）训练 14B 模型，让其在 ICLR/NeurIPS 留出测试集上同时超越 GPT-4o、Claude-3.5-Sonnet 与 AI-Scientist-V2 等 agentic 框架。

**[MulDimIF: A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models](muldimif_a_multi-dimensional_constraint_framework_for_evaluating_and_improving_i.md)**

:   提出 MulDimIF 多维约束框架，从约束模式（3种）、约束类别（4类13子类）和约束难度（4级）三个维度系统评估 LLM 的指令遵循能力，并通过 GRPO 训练显著提升模型性能，发现改进主要源自注意力模块的参数更新。

**[Not All Animals Are Equal: Metaphorical Framing through Source Domains and Semantic Frames](not_all_animals_are_equal_metaphorical_framing_through_source_domains_and_semant.md)**

:   本文提出首个结合 FrameNet 语义框架和概念隐喻理论（CMT）源域的计算框架 ConceptFrameMet，通过 RoBERTa 多任务模型检测隐喻并预测其语义框架和源域，配合对数似然比统计方法发现话语中显著的隐喻模式，揭示了自由派和保守派在移民话语中使用相同源域但选择不同语义框架来传达截然不同的联想。

**[Nürnberg NLP at PsyDefDetect: Multi-Axis Voter Ensembles for Psychological Defence Mechanism Classification](nürnberg_nlp_at_psydefdetect_multi-axis_voter_ensembles_for_psychological_defenc.md)**

:   这篇 BioNLP 2026 PsyDefDetect 共享任务系统论文把心理防御机制分类看成一个边界模糊、标注一致性有限的问题，用 9 个跨粒度、跨训练方式、跨基座模型的投票器做集成，在隐藏测试集上取得 F1=.420，并在 21 支注册队伍中排名第一。

**[One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization](one_persona_many_cues_different_results_how_sociodemographic_cues_impact_llm_per.md)**

:   本文系统比较了 6 种常用的人物画像提示方式（姓名/显式提及/对话历史各两种变体）在 7 个 LLM 和 4 个任务上的效果，发现虽然平均响应跨提示方式高度相关，但不同提示方式产生的人物画像间差异显著不同，过于显式的提示导致更强的个性化偏差，警示不应基于单一提示方式得出偏差结论。

**[Overcoming Copyright Barriers in Corpus Distribution Through Non-Reversible Hashing](overcoming_copyright_barriers_in_corpus_distribution_through_non-reversible_hash.md)**

:   本文提出 novelshare：把受版权保护文本的 token 变成截断后的非可逆哈希，并只公开哈希序列与研究者自有标注，使拥有合法原文的用户可以在轻微版本差异下重新对齐标注，在近版本小说上达到 98.7% 到 99.79% 的 token 正确对齐率。

**[Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design](solver-independent_automated_problem_formulation_via_llms_for_high-cost_simulati.md)**

:   本文提出 APF（Automated Problem Formulation），一种与求解器无关的框架，利用 LLM 将工程师的自然语言设计需求转化为可执行的数学优化模型，通过创新的数据生成和测试实例标注管线克服高成本仿真场景下无法使用求解器反馈筛选数据的困难，在天线设计任务上显著优于现有方法。

**[Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models](style_amnesia_investigating_speaking_style_degradation_and_mitigation_in_multi-t.md)**

:   发现口语语言模型（SLMs）在多轮对话中无法维持初始指定的说话风格（情感、口音、音量、语速），称之为"风格遗忘"现象，并通过注意力分析揭示其成因（注意力衰减），提出显式回忆过程作为缓解手段。

**[Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning](synthetic_eggs_in_many_baskets_the_impact_of_synthetic_data_diversity_on_llm_fin.md)**

:   这篇论文系统比较了单一模型、多模型和人类数据作为监督微调来源时对 Llama 模型的影响，发现多源合成数据能缓解分布坍缩和自偏好，但合成数据也可能在保留输出质量的同时削弱安全护栏，且风险会随源模型规模和混合方式发生复杂变化。

**[Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context](text-to-distribution_prediction_with_quantile_tokens_and_neighbor_context.md)**

:   本文提出Quantile Token Regression方法，通过在输入序列中插入专用分位数token并结合检索到的邻居实例及其经验分布，使LLM能够预测完整的条件分布而非单一点估计，在Airbnb和StackSample数据集上相比基线降低约4个MAPE点并将预测区间收窄2倍以上。

**[The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models](the_model_agreed_but_didn39t_learn_diagnosing_surface_compliance_in_large_langua.md)**

:   提出 SA-MCQ 诊断框架揭示知识编辑中的"表面合规"现象——编辑器在标准基准上达到高分但并未真正覆写内部信念，模型在判别式自评中会回退到原始参数记忆，递归编辑还会累积表征残留导致认知不稳定。

**[Think in Sentences: Explicit Sentence Boundaries Enhance Language Model's Capabilities](think_in_sentences_explicit_sentence_boundaries_enhance_language_model39s_capabi.md)**

:   本文提出在 LLM 输入中的句子边界处插入分隔符标记，通过 ICL 和 SFT 两种方式实现"逐句思考"的推理范式，在 7B 到 600B 模型上取得一致提升（GSM8k +7.7%，DROP +12.5%），且几乎不增加额外计算开销。

**[UCS: Estimating Unseen Coverage for Improved In-Context Learning](ucs_estimating_unseen_coverage_for_improved_in-context_learning.md)**

:   本文提出 UCS（Unseen Coverage Selection），一种基于 Smoothed Good-Turing 估计器的无训练子集级覆盖率先验，通过估计候选示例集中未观测到的潜在聚类数量来正则化现有 ICL 示例选择方法，在意图分类和推理任务上提升 2-6% 准确率。

**[Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection](understanding_structured_financial_data_with_llms_a_case_study_on_fraud_detectio.md)**

:   本文提出 FinFRE-RAG，一种两阶段框架，通过重要性引导的特征降维将高维表格交易数据序列化为自然语言，并结合标签感知的检索增强上下文学习，使开源 LLM 在金融欺诈检测上的 F1/MCC 大幅提升，缩小了与专用表格分类器的性能差距。

**[Unlocking the Potential of Diffusion Language Models through Template Infilling](unlocking_the_potential_of_diffusion_language_models_through_template_infilling.md)**

:   本文提出 Template Infilling，把扩散语言模型的生成条件从单一前缀改成分布在整段输出中的结构锚点，并用动态片段分配给复杂推理留出空间，从而在数学推理、代码生成和全局规划任务上显著稳定并提升并行生成质量。

**[VOYAGER: A Training Free Approach for Generating Diverse Datasets using LLMs](voyager_a_training_free_approach_for_generating_diverse_datasets_using_llms.md)**

:   Voyager 是一个无需训练的 LLM 数据生成算法，用 DPP 维护多样 anchor 和 explorer，并用 textual gradients 迭代改写提示词，从而在创意写作和推理数据生成中显著提高 Vendi 多样性且基本不牺牲质量。

**[Why Did Apple Fall: Evaluating Curiosity in Large Language Models](why_did_apple_fall_evaluating_curiosity_in_large_language_models.md)**

:   本文提出首个系统评估 LLM 好奇心行为的心理学启发框架，结合问卷自评和行为实验发现 LLM 展现出好奇心般的行为模式但并非内在特质，并设计好奇心驱动的提问管道证明模拟好奇行为可提升下游推理性能。
