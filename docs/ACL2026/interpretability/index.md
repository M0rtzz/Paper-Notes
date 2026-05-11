---
title: >-
  ACL2026 可解释性方向34篇论文解读
description: >-
  34篇ACL2026的可解释性方向论文解读，涵盖 LLM、推理、多模态、对齐/RLHF、模型压缩、知识蒸馏等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔬 可解释性

**💬 ACL2026** · **34** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (28)](../../CVPR2026/interpretability/index.md) · [🔬 ICLR2026 (56)](../../ICLR2026/interpretability/index.md) · [🤖 AAAI2026 (37)](../../AAAI2026/interpretability/index.md) · [🧠 NeurIPS2025 (86)](../../NeurIPS2025/interpretability/index.md) · [📹 ICCV2025 (10)](../../ICCV2025/interpretability/index.md) · [🧪 ICML2025 (33)](../../ICML2025/interpretability/index.md)

🔥 **高频主题：** LLM ×11 · 推理 ×6

**[A Structured Clustering Approach for Inducing Media Narratives](a_structured_clustering_approach_for_inducing_media_narratives.md)**

:   提出一个从大规模新闻语料中自动归纳媒体叙事模式的框架，通过联合建模事件因果链和角色（英雄/威胁/受害者）信息，使用角色约束的聚类算法将叙事链组织成语义连贯的叙事模式，在移民和枪支控制两个领域生成了可解释且与框架理论一致的叙事模式。

**[Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)**

:   构建大规模Post-hoc Self-Consistency Bank（PSCB，85K决策×428K解释），量化LLM答案与其解释之间的特征归因差距，并通过DPO优化在不损害准确率的前提下提升解释的归因一致性。

**[ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding](chemvlr_prioritizing_reasoning_in_perception_for_chemical_vision-language_unders.md)**

:   提出 ChemVLR，首个化学领域推理型 VLM，通过跨模态逆向工程策略构建 760K 推理数据集，结合持续预训练-SFT-RL 三阶段训练流程，在分子识别和反应预测任务上显著超越专有模型和领域专家 VLM。

**[Context-Value-Action Architecture for Value-Driven Large Language Model Agents](context-value-action_architecture_for_value-driven_large_language_model_agents.md)**

:   提出 CVA（Context-Value-Action）架构，基于 S-O-R 心理学模型和 Schwartz 价值理论，通过训练在真实人类数据上的 Value Verifier 解耦行为生成与认知推理，有效缓解 LLM 智能体的行为极化问题，在超过 110 万真实交互轨迹的 CVABench 上显著优于基线。

**[Curing "Miracle Steps" in LLM Mathematical Reasoning with Rubric Rewards](curing_miracle_steps_in_llm_mathematical_reasoning_with_rubric_rewards.md)**

:   本文发现当前 LLM 数学推理中存在大量"Miracle Steps"——推理链中凭空跳跃到正确答案的现象，并提出 Rubric Reward Model (RRM)，一种基于问题特定评分标准的过程奖励函数，在 RL 训练中显著减少 Miracle Steps 71% 并将 AIME2024 的 Verified Pass@1024 从 26.7% 提升至 62.6%。

**[Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations](do_llms_know_tool_irrelevance_demystifying_structural_alignment_bias_in_tool_inv.md)**

:   发现并形式化了 LLM 工具调用中的"结构对齐偏差"——当查询属性可以有效映射到工具参数时（即使工具功能与用户目标无关），LLM 仍倾向调用该工具。构建 SABEval 数据集解耦结构对齐和语义相关性，用对比注意力归因揭示内部存在语义检查和结构匹配两条竞争路径，提出再平衡策略实现 80% 的相对错误减少。

**[Evian: Towards Explainable Visual Instruction-tuning Data Auditing](evian_towards_explainable_visual_instruction-tuning_data_auditing.md)**

:   提出"分解-再评估"（Decomposition-then-Evaluation）范式和 EVIAN 框架，将视觉指令微调数据的回答分解为视觉描述、主观推理和事实声明三个组件，沿图文一致性、逻辑连贯性和事实准确性三个正交维度评估，发现用其筛选的少量高质量数据训练的模型优于大规模数据集训练的模型。

**[Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models](experiments_or_outcomes_probing_scientific_feasibility_in_large_language_models.md)**

:   构建控制知识框架系统研究LLM在科学可行性评估中如何利用实验描述和结果证据，发现提供结果证据比实验描述更可靠，部分实验信息常导致性能低于仅用参数知识的基线，揭示了LLM推理的脆弱性。

**[Forest Before Trees: Latent Superposition for Efficient Visual Reasoning](forest_before_trees_latent_superposition_for_efficient_visual_reasoning.md)**

:   本文提出 Laser，通过动态窗口对齐学习（DWAL）在潜在空间中进行视觉推理，使模型在推理过程中维持未来语义的"概率叠加态"而非逐 token 精确预测，实现"先全局后局部"的认知层次，在 6 个基准上以仅 6 个推理 token（减少 97%+）达到潜在推理方法的 SOTA，超越 Monet 平均 5.03%。

**[From Signal Degradation to Computation Collapse: Uncovering the Two Failure Modes of LLM Quantization](from_signal_degradation_to_computation_collapse_uncovering_the_two_failure_modes.md)**

:   本文通过系统的机械可解释性分析，揭示LLM量化存在两种质性不同的失败模式：4-bit的信号退化（Signal Degradation，计算模式完整但精度受损，可局部修复）和2-bit的计算崩溃（Computation Collapse，关键组件功能性破坏，需结构重建）。

**[HistLens: Mapping Idea Change across Concepts and Corpora](histlens_mapping_idea_change_across_concepts_and_corpora.md)**

:   提出 HistLens 框架，基于稀疏自编码器（SAE）将概念表示分解为可解释的语义基向量，在共享坐标系中追踪多概念、多语料的历时演化轨迹，支持隐式概念计算，为数字人文和概念史研究提供可量化、可比较的分析工具。

**[IDEA: An Interpretable and Editable Decision-Making Framework for LLMs via Verbal-to-Numeric Calibration](idea_an_interpretable_and_editable_decision-making_framework_for_llms_via_verbal.md)**

:   提出 IDEA 框架，将 LLM 的决策知识提取为语义因子上的可解释参数化模型，通过 EM 算法联合学习语言概率表达到数值的映射和决策参数，实现了可校准、可编辑、可解释的 LLM 决策，在五个数据集上以 Qwen-3-32B (78.6%) 超越 DeepSeek R1 (68.1%) 和 GPT-5.2 (77.9%)。

**[Interpretability from the Ground Up](interpretability_from_the_ground_up_stakeholder-centric_design_of_automated_scor.md)**

:   本文从教育评估利益相关者需求出发提出 FGTI 四原则（忠实、扎根、可追溯、可互换），开发 AnalyticScore 三阶段框架实现可解释自动评分，在 ASAP-SAS 上平均 QWK 仅比不可解释 SOTA 低 0.06。

**[Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation](interpretable_traces_unexpected_outcomes_investigating_the_disconnect_in_trace-b.md)**

:   通过规则化问题分解方法构建可验证的中间推理链数据集，揭示 CoT 推理链的语义正确性与最终答案准确率不可靠地相关（正确链仅 28% 导致正确答案），且最可解释的推理链并非最提升性能的——冗长的 R1 链性能最优但用户评为最不可解释。

**[LePREC: Reasoning as Classification over Structured Factors for Assessing Relevance of Legal Issues](leprec_reasoning_as_classification_over_structured_factors_for_assessing_relevan.md)**

:   本文提出 LePREC，一种受法律专业人士启发的神经-符号框架，通过 LLM 生成推理问答对将非结构化法律文本转化为结构化特征，再利用稀疏线性模型进行相关性分类，在 769 个马来西亚合同法案例构建的 LIC 数据集上相比 GPT-4o 等 LLM 基线提升 30–40%。

**[LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines](llm-guided_semantic_bootstrapping_for_interpretable_text_classification_with_tse.md)**

:   本文提出 LLM 引导的语义引导框架，通过 LLM 生成子意图和三阶段课程式合成数据训练非否定 Tsetlin Machine（NTM），提取高置信度符号特征注入真实数据，使标准 TM 在保持完全可解释性的同时逼近 BERT 的分类性能。

**[NOSE: Neural Olfactory-Semantic Embedding with Tri-Modal Orthogonal Contrastive Learning](nose_neural_olfactory-semantic_embedding_with_tri-modal_orthogonal_contrastive_l.md)**

:   提出 NOSE 三模态嗅觉表示学习框架，以分子为枢纽通过正交注入机制对齐分子结构、受体序列和自然语言描述三个模态，配合 LLM 驱动的弱正样本策略缓解描述稀疏问题，在 11 个下游任务上达到 SOTA 并展现优秀的零样本泛化能力。

**[PV-SQL: Synergizing Database Probing and Rule-based Verification for Text-to-SQL Agents](pv-sql_synergizing_database_probing_and_rule-based_verification_for_text-to-sql_.md)**

:   本文提出 PV-SQL，一个 Agent 式 Text-to-SQL 框架，通过 Probe（迭代生成探测查询发现数据库值格式/列语义/表关系）和 Verify（基于模式匹配提取可验证约束并构建检查清单）两个互补组件，在 BIRD 基准上比最佳基线高 5% 执行准确率和 20.8% 有效效率分。

**[Reasoning Fails Where Step Flow Breaks](reasoning_fails_where_step_flow_breaks.md)**

:   提出 Step-Saliency 诊断工具发现大推理模型中两种深度相关的信息流失败模式（Shallow Lock-in 和 Deep Decay），并设计 StepFlow 测试时干预方法在不重训练的情况下修复信息传播、提升推理准确率。

**[Revitalizing Black-Box Interpretability: Actionable Interpretability for LLMs via Proxy Models](revitalizing_black-box_interpretability_actionable_interpretability_for_llms_via.md)**

:   本文提出一种基于代理模型的黑盒可解释性框架，利用廉价小模型近似昂贵大模型的局部决策边界来生成 LIME/SHAP 解释，通过统计筛选-应用（screen-and-apply）机制确保可靠性，代理解释在保持超过 90% 保真度的同时将成本降低 88.2%，并成功用于 Prompt 压缩和中毒样本移除等下游优化任务。

**[Rhetorical Questions in LLM Representations: A Linear Probing Study](rhetorical_questions_in_llm_representations_a_linear_probing_study.md)**

:   通过线性探针分析 LLM 内部如何表征反问句，发现反问句在表征空间中是线性可分的且可跨数据集迁移，但不同数据集学到的探针方向并不一致——反问句由多个异构的线性方向编码，而非单一统一维度。

**[Similarity-Distance-Magnitude Activations](similarity-distance-magnitude_activations.md)**

:   本文提出 SDM（Similarity-Distance-Magnitude）激活函数作为 softmax 的更鲁棒替代，通过将正确预测的深度匹配（Similarity）、到训练分布的距离（Distance）和决策边界距离（Magnitude）三个认知维度解耦并整合为新的激活 $\text{sdm}(\mathbf{z}')_i = (2+q)^{d \cdot z'_i} / \sum_c (2+q)^{d \cdot z'_c}$，并在此基础上构建 SDM 估计器进行选择性分类，在协变量偏移和分布外输入下比现有校准方法更鲁棒。

**[SITE: Soft Head Selection for Injecting ICL-Derived Task Embeddings](soft_head_selection_for_injecting_icl-derived_task_embeddings.md)**

:   SITE 提出了一种基于梯度优化的软注意力头选择方法，通过识别任务相关的注意力头来有效注入 ICL 衍生的任务嵌入，在 12 个 LLM（4B-70B）上显著超越 ICL 和现有嵌入方法，同时用远少于 PEFT 的可训练参数达到可比性能。

**[SPENCE: A Syntactic Probe for Detecting Contamination in NL2SQL Benchmarks](spence_a_syntactic_probe_for_detecting_contamination_in_nl2sql_benchmarks.md)**

:   SPENCE 通过对 NL2SQL 基准查询进行系统性句法改写并测量执行准确率随句法距离的衰减程度，检测和量化 LLM 在 NL2SQL 基准上的数据污染行为，发现越老的基准（如 Spider）污染信号越强，而较新的 BIRD 基准几乎不受影响。

**[StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference](structkv_preserving_the_structural_skeleton_for_scalable_long-context_inference.md)**

:   本文提出 StructKV，一个结构感知的 KV Cache 压缩框架，通过全局入度中心性（Global In-Degree Centrality）跨层累积注意力模式识别全局信息枢纽，动态枢纽层检测（Dynamic Pivot Detection）自适应定位最优压缩层，以及结构传播与解耦（Structural Propagation & Decoupling）分离计算预算和存储预算，在 LongBench 和 RULER 上以 60% prefill + 10% KV 实现了接近全上下文的性能。

**[Style over Story: Measuring LLM Narrative Preferences via Structured Selection](style_over_story_measuring_llm_narrative_preferences_via_structured_selection.md)**

:   本文设计了一种基于约束选择的实验范式来测量 LLM 的叙事偏好，使用叙事学理论构建的 200 个约束库让 6 个 LLM 在不同指令类型下进行选择，发现模型系统性地优先选择"风格"（Style）而非"事件"（Event）、"角色"（Character）和"场景"（Setting）等内容元素。

**[TabReX: Tabular Referenceless eXplainable Evaluation](tabrex_tabular_referenceless_explainable_evaluation.md)**

:   提出 TabReX，一种基于图推理的无参考表格生成评估框架，将源文本和生成表格转化为知识图谱三元组并对齐，计算可解释的属性驱动分数，在人类判断相关性上大幅超越现有方法；同时构建 TabReX-Bench 大规模基准。

**[The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination](the_reasoning_trap_how_enhancing_llm_reasoning_amplifies_tool_hallucination.md)**

:   系统性揭示了"推理陷阱"悖论：增强LLM推理能力（无论通过RL、蒸馏还是可切换推理模式）会系统性地放大工具幻觉，且这一效应与推理本身而非RL训练相关联，现有缓解策略（提示工程、DPO）面临不可避免的可靠性-能力权衡。

**[ThreadSumm: Summarization of Nested Discourse Threads Using Tree of Thoughts](threadsumm_summarization_of_nested_discourse_threads_using_tree_of_thoughts.md)**

:   本文提出 ThreadSumm，一个多阶段 LLM 管道框架，将嵌套话语线程摘要建模为层次推理问题——先提取方面和原子内容单元进行内容规划，再通过句子排序构建线程感知序列，最后用 Tree of Thoughts 搜索生成和评分多个段落候选，在 Reddit/StackExchange 数据集上优于基线。

**[To Trust or Not to Trust: Attention-Based Trust Management for LLM Multi-Agent Systems](to_trust_or_not_to_trust_attention-based_trust_management_for_llm_multi-agent_sy.md)**

:   本文为 LLM 多智能体系统（LLM-MAS）提出首个全面的"可信度"定义（基于 Grice 合作原则的六个正交维度），发现 LLM 的注意力模式可区分不同类型的可信度违规，据此设计了轻量级的 A-Trust 评估方法和端到端的信任管理系统（TMS），在多种攻击下将恶意消息检测率提升至 77-90%。

**[Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures](towards_intrinsic_interpretability_of_large_language_modelsa_survey_of_design_pr.md)**

:   系统综述了 LLM 内在可解释性的最新进展，将现有方法分为五大设计范式（功能透明性、概念对齐、表征可分解性、显式模块化、潜在稀疏归纳），并讨论了开放挑战和未来方向。

**[Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)**

:   本文系统研究LLM在文本生成过程中回忆关系知识的内部机制，发现注意力头对残差流的逐头贡献（$\Delta_{att,h}$）是线性关系分类的最强特征（准确率达91%），并提出HeadScore和TokenScore两种探针归因方法来分解预测到注意力头和源token级别，揭示了探针精度与关系特异性、实体连通度及探针信号集中度之间的明确相关性。

**[Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation](understanding_new-knowledge-induced_factual_hallucinations_in_llms_analysis_and_.md)**

:   本文通过受控合成数据集 Biography-Reasoning 系统分析了 SFT 阶段学习新知识导致的事实幻觉现象，发现幻觉的根本机制是模型对关键实体的注意力被削弱，并提出 KnownPatch——在训练末期注入少量已知知识来恢复注意力模式，有效缓解幻觉。

**[Understanding or Memorizing? A Case Study of German Definite Articles in Language Models](understanding_or_memorizing_a_case_study_of_german_definite_articles_in_language.md)**

:   本文利用 Gradiend 梯度可解释性方法研究语言模型预测德语定冠词（der/die/das/den/dem/des）时是基于抽象语法规则还是表层记忆，发现模型至少部分依赖记忆化关联而非严格的规则编码。
