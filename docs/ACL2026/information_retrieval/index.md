---
title: >-
  ACL2026 信息检索/RAG方向43篇论文解读
description: >-
  43篇ACL2026的信息检索/RAG 方向论文解读，涵盖 RAG、LLM、问答、对抗鲁棒、推理、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 信息检索/RAG

**💬 ACL2026** · **43** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (7)](../../CVPR2026/information_retrieval/index.md) · [🔬 ICLR2026 (33)](../../ICLR2026/information_retrieval/index.md) · [🤖 AAAI2026 (28)](../../AAAI2026/information_retrieval/index.md) · [🧠 NeurIPS2025 (30)](../../NeurIPS2025/information_retrieval/index.md) · [📹 ICCV2025 (8)](../../ICCV2025/information_retrieval/index.md) · [🧪 ICML2025 (5)](../../ICML2025/information_retrieval/index.md)

🔥 **高频主题：** RAG ×16 · LLM ×5 · 问答 ×4 · 对抗鲁棒 ×3 · 推理 ×3

**[All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG](all_languages_matter_understanding_and_mitigating_language_bias_in_multilingual_.md)**

:   系统揭示多语言 RAG 系统在重排序阶段存在严重的语言偏差（偏好英语和查询语言），提出 LAURA 框架通过下游生成质量驱动的监督信号对齐重排序器，有效缓解偏差并提升生成性能。

**[An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs](an_iterative_utility_judgment_framework_inspired_by_philosophical_relevance_via_.md)**

:   受Schutz哲学相关性理论启发，提出ITEM迭代效用判断框架，通过让RAG中的三个组件（相关性排序、效用判断、答案生成）动态交互增强，在检索、效用判断和QA任务上均优于基线。

**[Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring](bayesian_active_learning_with_gaussian_processes_guided_by_llm_relevance_scoring.md)**

:   提出 BAGEL，一个基于高斯过程（GP）的贝叶斯主动学习框架，在有限 LLM 预算下通过探索-利用平衡策略传播稀疏 LLM 相关性信号，实现全局嵌入空间的段落检索，显著超越传统 LLM 重排序方法。

**[Beyond Black-Box Interventions: Latent Probing for Faithful Retrieval-Augmented Generation](beyond_black-box_interventions_latent_probing_for_faithful_retrieval-augmented_g.md)**

:   提出 ProbeRAG，通过发现 LLM 隐空间中冲突/对齐知识的线性可分性，设计三阶段框架（细粒度知识剪枝→隐空间冲突探测→冲突感知注意力），从模型内部机制解决 RAG 忠实性问题。

**[Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation](beyond_explicit_refusals_soft-failure_attacks_on_retrieval-augmented_generation.md)**

:   形式化定义 RAG 系统的"软失败"威胁（生成流畅但无信息量的回答），提出 DEJA 黑箱进化攻击框架，通过对抗性文档诱导模型利用安全对齐机制产生模棱两可的回答，SASR 超过 79% 且高度隐蔽。

**[CarO: Chain-of-Analogy Reasoning Optimization for Robust Content Moderation](caro_chain-of-analogy_reasoning_optimization_for_robust_content_moderation.md)**

:   提出 CarO（Chain-of-Analogy Reasoning Optimization），一个两阶段训练框架，通过 RAG 引导生成类比推理链 + SFT + 定制 DPO 优化，使 LLM 在推理时自主生成类比参考案例进行内容审核，在模糊审核基准上 F1 平均提升 24.9%，显著超越推理模型（DeepSeek R1）和专用审核模型（LLaMA Guard）。

**[ChAIRO: Contextual Hierarchical Analogical Induction and Reasoning Optimization for LLMs](chairo_contextual_hierarchical_analogical_induction_and_reasoning_optimization_f.md)**

:   提出 ChAIRO，一个上下文层次化类比归纳与推理优化框架，通过三阶段 pipeline（类比案例生成→规则归纳→规则注入微调）让 LLM 在内容审核中自主生成类比案例并归纳显式审核规则，比单实例规则生成提升 F1 4.5%，比静态 RAG 提升 2.3%。

**[ChunQiuTR: Time-Keyed Temporal Retrieval in Classical Chinese Annals](chunqiutr_time-keyed_temporal_retrieval_in_classical_chinese_annals.md)**

:   提出 ChunQiuTR，首个基于非格里历的时间键检索基准，从《春秋》及其注疏传统中构建，并设计了 CTD（历法时间双编码器），通过傅里叶绝对历法上下文和相对偏移偏置实现时间感知检索，显著优于纯语义基线。

**[CodePromptZip: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs](codepromptzip_code-specific_prompt_compression_for_retrieval-augmented_generatio.md)**

:   提出 CodePromptZip，首个面向代码的提示压缩框架，通过类型感知优先级排序构建训练数据并训练带 copy 机制的小模型压缩器，在三个编码任务上分别比最佳基线提升 23.4%、28.7% 和 8.7%。

**[Conjecture and Inquiry: Quantifying Software Performance Requirements via Interactive Retrieval-Augmented Preference Elicitation](conjecture_and_inquiry_quantifying_software_performance_requirements_via_interac.md)**

:   提出IRAP方法，通过交互式检索增强偏好获取（Interactive Retrieval-Augmented Preference Elicitation）将自然语言描述的软件性能需求量化为数学函数，在4个真实数据集上相比10种SOTA方法取得最高40倍的性能提升，且仅需5轮交互。

**[Context Attribution with Multi-Armed Bandit Optimization](context_attribution_with_multi-armed_bandit_optimization.md)**

:   本文提出 CAMAB，将 RAG 中的上下文归因（识别哪些上下文片段对生成答案有贡献）建模为组合多臂赌博机（CMAB）问题，使用线性 Thompson 采样自适应地探索上下文子集空间，在 HotpotQA、CNN/DM、TyDi QA 上比 SHAP 和 ContextCite 减少最多 30% 的模型查询次数同时匹配或超越归因质量。

**[CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering](counterrefine_answer-conditioned_counterevidence_retrieval_for_inference-time_kn.md)**

:   本文提出 CounterRefine，一个轻量级推理时修复层：先用标准 RAG 产生初步答案，再通过答案条件化的反证检索收集支持/反对证据，最后通过受限的 KEEP/REVISE 决策和确定性验证修复错误答案，在 SimpleQA 上将 GPT-5 的正确率从 67.3% 提升至 73.1%。

**[CRAFT: Training-Free Cascaded Retrieval for Tabular QA](craft_training-free_cascaded_retrieval_for_tabular_qa.md)**

:   本文提出 CRAFT，一个无需数据集特定训练的三阶段级联表格检索框架（SPLADE 稀疏过滤 → 语义 mini-table 排序 → 神经重排序），通过 Gemini 生成的表格标题和描述增强表格表示，在 NQ-Tables 上达到 SOTA（R@1 49.84），在 OTT-QA 上展现强零样本泛化能力，且对查询改写具有显著鲁棒性。

**[CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge](curate_continual_unlearning_in_real_time_with_ensured_preservation_of_llm_knowle.md)**

:   CURaTE 提出一种基于句子嵌入匹配的行为遗忘框架：预部署时训练一个通用的遗忘嵌入器（不使用任何遗忘集），部署后实时将新遗忘请求嵌入存入数据库，推理时通过余弦相似度决定是回答还是拒绝，完全不修改 LLM 权重从而实现近乎完美的知识保留。

**[Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game](detecting_rag_extraction_attack_via_dual-path_runtime_integrity_game.md)**

:   提出 CanaryRAG，一个受软件安全中栈金丝雀启发的 RAG 运行时防御机制，通过在检索块中注入非语义金丝雀 token 并设计双路径完整性博弈（目标路径不应泄露金丝雀 + Oracle 路径应能引出金丝雀），实时检测知识库提取攻击，在不影响任务性能和推理延迟的前提下实现最强防护。

**[Domain-Specific Data Generation Framework for RAG Adaptation](domain-specific_data_generation_framework_for_rag_adaptation.md)**

:   本文提出 RAGen，一个可扩展的模块化数据生成框架，通过文档级概念提取、多块证据组装和 Bloom 分类学引导的问题生成，自动合成领域特定的 QAC（问题-答案-上下文）数据，支持嵌入模型对比微调和 LLM 监督微调，在三个领域数据集上显著优于 AutoRAG 和 LlamaIndex 基线。

**[DQA: Diagnostic Question Answering for IT Support](dqa_diagnostic_question_answering_for_it_support.md)**

:   本文提出DQA框架，通过维护持久化的诊断状态和在根因层面聚合检索证据（而非逐文档处理），实现企业IT支持场景下的系统化故障排查，成功率从基线41.3%提升至78.7%，平均轮次从8.4降至3.9。

**[End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)**

:   本文提出 MHGPO（Multi-Agent Heterogeneous Group Policy Optimization），一种无需 critic 的多智能体 RL 方法，通过异构组相对优势估计和反向奖励传播，在三智能体搜索系统（Rewriter→Reranker→Answerer）中实现端到端优化，捕获隐式跨智能体依赖和跨轨迹关联，在 HotpotQA 等多跳 QA 基准上显著优于 MAPPO 和 GRPO 基线。

**[Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization](enhancing_llm-based_search_agents_via_contribution_weighted_group_relative_polic.md)**

:   CW-GRPO 将过程监督重新定义为"优势重分配"：用 LLM 判断器评估每轮搜索的检索有用性和推理正确性，计算贡献分数来缩放基于结果的优势，实现轮级别信用分配而不引入不稳定的价值函数，在 Qwen3-8B 上超越标准 GRPO 5.0%。

**[Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)**

:   本文发现多语言 RAG 系统中"英语偏好"主要是评估基准中结构性先验（gold 证据集中于英语、文化先验）的伪影而非模型固有偏差，提出去偏语言偏好指标 DeLP 揭示检索器实际偏好单语对齐，并基于此设计 DELTA 查询增强框架，在多语言 RAG 上一致超越英语枢轴策略。

**[FAITH: Factuality Alignment through Integrating Trustworthiness and Honestness](faith_factuality_alignment_through_integrating_trustworthiness_and_honestness.md)**

:   本文提出FAITH框架，通过将LLM的不确定性信号（一致性+语义熵）映射到自然语言描述的知识状态象限（可信度×诚实度），设计考虑不确定性的细粒度奖励函数进行PPO训练，再用RAG模块纠正潜在错误，系统性提升LLM的事实准确性。

**[Feedback Adaptation for Retrieval-Augmented Generation](feedback_adaptation_for_retrieval-augmented_generation.md)**

:   本文提出"反馈适应"作为RAG系统的新问题设定——研究纠正性反馈多快、多有效地传播到未来查询，定义了纠正延迟和反馈后性能两个评估轴，并提出PatchRAG作为免训练的推理时反馈整合方案，实现即时纠正和强泛化。

**[FLARE: Task-Agnostic Embedding Model Evaluation via Normalizing Flows](flare_task-agnostic_embedding_model_evaluation_through_a_normalization_process.md)**

:   提出FLARE框架，利用正则化流（Normalizing Flows）进行无标签的文本嵌入模型评估，通过直接从对数似然估计信息充分性来避免基于距离的密度估计在高维空间中的崩溃，在11个数据集上与有监督基准的Spearman $\rho$ 达0.90。

**[From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)**

:   本文提出AuthGR，首个将文档权威性系统性整合到生成式检索中的框架，通过VLM多模态权威评分、三阶段渐进式训练（CPT→SFT→GRPO）和混合集成部署管线，在Naver商业搜索引擎的大规模A/B测试中验证了显著的用户参与度提升。

**[How Retrieved Context Shapes Internal Representations in RAG](how_retrieved_context_shapes_internal_representations_in_rag.md)**

:   本文从隐藏表示的角度系统分析 RAG 中检索文档如何影响 LLM 内部状态，发现了五个关键模式：随机文档引发大表示漂移并触发拒绝行为、相关文档主要确认而非改变参数化知识、单个相关文档能锚定多文档场景中的表示、后层逐步强调参数化知识从而限制检索证据的影响、以及 LLM 在早期层就能区分随机文档但到最后层仍无法可靠区分干扰文档和相关文档。

**[Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy](hybrid-vector_retrieval_for_visually_rich_documents_combining_single-vector_effi.md)**

:   HEAVEN 提出了一种即插即用的两阶段混合向量框架，通过视觉摘要页（VS-Pages）加速单向量粗检索 + 基于词性的查询 token 过滤减少多向量重排序计算，在四个基准上保持 99.87% 的多向量 Recall@1 同时减少 99.82% 的每查询 FLOPs。

**[Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches](is_agentic_rag_worth_it_an_experimental_comparison_of_rag_approaches.md)**

:   本文在四个数据集上从用户意图处理、查询重写、文档精炼和底层 LLM 选择四个维度系统对比了 Enhanced RAG 和 Agentic RAG，发现两者各有优势——Agentic RAG 在意图路由和查询重写上更灵活，Enhanced RAG 在文档重排上更有效，而 Agentic RAG 的成本高达 3.3 倍。

**[KoCo-Bench: Can Large Language Models Leverage Domain Knowledge in Software Development?](koco-bench_can_large_language_models_leverage_domain_knowledge_in_software_devel.md)**

:   KoCo-Bench 提出首个包含显式领域知识语料库的代码基准，覆盖 6 个新兴领域（RL、Agent、RAG 等）的 11 个框架和 25 个项目，评估 LLM 从知识语料库中获取和应用领域知识进行代码生成和知识理解的能力，揭示即使最强 coding agent Claude Code 也仅达 34.2%。

**[MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits](mab-dqa_addressing_query_aspect_importance_in_document_question_answering_with_m.md)**

:   提出 MAB-DQA 框架，将复杂查询分解为多个方面子查询，用多臂老虎机机制（Thompson Sampling）动态评估各方面的重要性并重新分配检索预算，显著提升多模态文档问答的检索精度和回答准确率。

**[MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](mass-rag_multi-agent_synthesis_retrieval-augmented_generation.md)**

:   本文提出 MASS-RAG，一个免训练的多 Agent 综合 RAG 框架，通过 Summarizer/Extractor/Reasoner 三个专门化过滤 Agent 从互补视角处理检索文档，再通过 Synthesis Agent 整合多视角证据或候选答案，在四个基准上持续超越强基线。

**[Multi-Faceted Self-Consistent Preference Alignment for Query Rewriting in Conversational Search](multi-faceted_self-consistent_preference_alignment_for_query_rewriting_in_conver.md)**

:   本文提出 MSPA-CQR，通过从改写、检索、回复三个维度构建自一致性偏好数据，并使用前缀引导的多维度 DPO 优化来训练查询改写模型，在分布内外场景均显著超越现有方法。

**[RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding](racer_retrieval-augmented_contextual_rapid_speculative_decoding.md)**

:   RACER 提出了一种无需训练的推测解码方法，将基于检索的精确模式匹配与基于 logits 的未来预测统一起来，通过 copy-logit 策略构建 Logits Tree、LRU 驱逐的 AC 自动机构建 Retrieval Tree，在多个基准上实现了超过 2 倍的推理加速。

**[ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)**

:   ReasonEmbed 提出三项技术创新——ReMixer 非平凡合成数据方法（82K 高质量样本）、Redapter 自适应推理强度加权训练和多骨干实现——在 BRIGHT 基准上以 38.1 的 nDCG@10 显著超越所有现有文本嵌入模型约 10 个点。

**[Region-R1: Reinforcing Query-Side Region Cropping for Multi-Modal Re-Ranking](region-r1_reinforcing_query-side_region_cropping_for_multi-modal_re-ranking.md)**

:   本文提出 Region-R1，将多模态重排序中的查询图像区域裁剪建模为决策问题，通过强化学习（r-GRPO）学习何时以及如何裁剪查询图像中与问题相关的区域，在 E-VQA 和 InfoSeek 上将 CondRecall@1 分别提升 20% 和 8%。

**[RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion](reposhapley_shapley-enhanced_context_filtering_for_repository-level_code_complet.md)**

:   提出 RepoShapley，一种基于 Shapley 值的联盟感知上下文过滤框架，通过估计检索代码片段在组合中的交互贡献来决定保留/丢弃，显著提升仓库级代码补全质量。

**[Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval](sculpting_the_vector_space_towards_efficient_multi-vector_visual_document_retrie.md)**

:   本文提出 Prune-then-Merge，一个两阶段的免训练多向量文档压缩框架——先通过自适应注意力剪枝移除低信息 patch，再对剩余高信号 patch 进行层次聚类合并，在 29 个 VDR 数据集上将近无损压缩范围从 50-60% 扩展到 60-70%，并在 80%+ 高压缩率下显著优于单阶段方法。

**[Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)**

:   揭示 RAG 系统对检索文档排列顺序高度敏感的问题，提出 Stable-RAG：通过对文档排列产生的隐状态做谱聚类识别主导推理模式，再用 DPO 对齐将幻觉输出引导向正确答案，在三个 QA 数据集上实现准确率和推理一致性的双重提升。

**[TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](taxpraben_a_scalable_benchmark_for_structured_evaluation_of_llms_in_chinese_real.md)**

:   本文提出 TaxPraBen，首个面向中国税务实践的 LLM 评测基准，包含 14 个数据集共 7.3K 样本，覆盖税务风险防控、稽查分析和税务筹划三大真实场景，并设计了"结构化解析—字段对齐提取—数值与文本匹配"的可扩展评估范式，评测 19 个 LLM 后发现闭源大模型和中文优化模型表现更优，而税务领域微调模型 YaYi2 改进有限。

**[To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs](to_lie_or_not_to_lie_investigating_the_biased_spread_of_global_lies_by_llms.md)**

:   本文提出 GlobalLies——一个包含 440 个虚假信息生成模板和 6,867 个实体的多语言平行数据集（8 种语言、195 个国家），揭示了 LLM 在传播虚假信息时存在系统性的国家和语言偏差：对低 HDI 国家的虚假信息生成率显著更高（统计相关 $\rho=-0.355$, $p=5\times10^{-7}$），低资源语言的合规率高出英语 30% 以上，且现有安全分类器和 RAG 防护措施提供不均匀的保护。

**[TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG](tpa_next_token_probability_attribution_for_detecting_hallucinations_in_rag.md)**

:   本文提出 TPA 框架，通过数学方法将 LLM 每个 token 的生成概率精确分解为七个来源（Query、RAG Context、Past Token、Self Token、FFN、Final LayerNorm、Initial Embedding）的贡献，结合词性标注聚合特征，实现 RAG 场景下的 SOTA 幻觉检测。

**[Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection](understanding_structured_financial_data_with_llms_a_case_study_on_fraud_detectio.md)**

:   本文提出 FinFRE-RAG，一种两阶段框架，通过重要性引导的特征降维将高维表格交易数据序列化为自然语言，并结合标签感知的检索增强上下文学习，使开源 LLM 在金融欺诈检测上的 F1/MCC 大幅提升，缩小了与专用表格分类器的性能差距。

**[VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG](videostir_understanding_long_videos_via_spatio-temporally_structured_and_intent-.md)**

:   VideoStir 提出了一种结构化且意图感知的长视频 RAG 框架，通过将视频建模为时空图进行多跳 clip 检索 + 训练意图相关性评分器进行帧级筛选，在不依赖辅助文本工具的前提下达到了与 SOTA 长视频 RAG 方法可比的性能。

**[Why These Documents? Explainable Generative Retrieval with Hierarchical Category Paths](why_these_documents_explainable_generative_retrieval_with_hierarchical_category_.md)**

:   提出 HyPE 框架，在生成式检索中通过先生成层级类别路径（如 "Government >> Government by cities"）再解码文档标识符，为检索结果提供查询相关的可解释路径，同时提升检索准确率。
