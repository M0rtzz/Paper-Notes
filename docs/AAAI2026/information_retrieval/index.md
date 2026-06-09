---
title: >-
  AAAI2026 信息检索/RAG方向27篇论文解读
description: >-
  27篇AAAI2026的信息检索/RAG 方向论文解读，涵盖 RAG、推理、多模态、LLM、Agent、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "AAAI2026"
  - "信息检索/RAG"
  - "论文解读"
  - "论文笔记"
  - "RAG"
  - "推理"
  - "多模态"
  - "LLM"
  - "Agent"
  - "对齐/RLHF"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 信息检索/RAG

**🤖 AAAI2026** · **27** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (22)](../../ICML2026/information_retrieval/index.md) · [💬 ACL2026 (74)](../../ACL2026/information_retrieval/index.md) · [📷 CVPR2026 (9)](../../CVPR2026/information_retrieval/index.md) · [🔬 ICLR2026 (33)](../../ICLR2026/information_retrieval/index.md) · [🧠 NeurIPS2025 (30)](../../NeurIPS2025/information_retrieval/index.md) · [📹 ICCV2025 (7)](../../ICCV2025/information_retrieval/index.md)

🔥 **高频主题：** RAG ×7 · 推理 ×4 · 多模态 ×4 · LLM ×3 · Agent ×3

**["As Eastern Powers, I Will Veto." : An Investigation of Nation-Level Bias of Large Language Models in International Relations](as_eastern_powers_i_will_veto_an_investigation_of_nation-level_bias_of_large_lan.md)**

:   系统性地研究 LLM 在国际关系领域的国家级偏见，基于联合国安理会真实数据设计三种偏见测试（直接问答、关联测试、投票模拟），揭示偏见的多维性——随模型和评知上下文变化，并提出 RAG+Reflexion 去偏框架。

**[Beyond Perplexity: Let the Reader Select Retrieval Summaries via Spectrum Projection Score](beyond_perplexity_let_the_reader_select_retrieval_summaries_via_spectrum_project.md)**

:   提出 Spectrum Projection Score (SPS) 这一无需训练的指标，通过衡量摘要 token 嵌入与 reader LLM 主子空间的对齐程度来评估检索摘要质量，替代传统困惑度指标。结合 xCompress 推理时控制器，在 5 个 QA 数据集上显著优于基于困惑度的方法（HotpotQA EM +3.6）。

**[Cog-RAG: Cognitive-Inspired Dual-Hypergraph with Theme Alignment Retrieval-Augmented Generation](cog-rag_cognitive-inspired_dual-hypergraph_with_theme_alignment_retrieval-augmen.md)**

:   提出 Cog-RAG，用主题超图和实体超图构建双超图索引，模拟人类"自顶向下"的认知过程进行两阶段检索（先主题后细节），实现从全局语义到局部信息的对齐生成。

**[ComLQ: Benchmarking Complex Logical Queries in Information Retrieval](comlq_benchmarking_complex_logical_queries_in_information_retrieval.md)**

:   构建了首个面向复杂逻辑查询的信息检索基准 ComLQ（含合取、析取、否定等 14 种查询类型），并提出子图引导的 LLM 数据合成方法和否定一致性评估指标 LSNC，揭示现有检索器在逻辑推理尤其是否定建模上的严重不足。

**[ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning](comorag_a_cognitive-inspired_memory-organized_rag_for_stateful_long_narrative_re.md)**

:   受人脑前额叶皮层元认知调控机制启发，提出 ComoRAG 框架，通过动态记忆工作空间和迭代探测查询实现有状态的多步推理，在长篇叙事理解（200K+ tokens）任务上显著超越现有 RAG 方法。

**[ConvMix: A Mixed-Criteria Data Augmentation Framework for Conversational Dense Retrieval](convmix_a_mixed-criteria_data_augmentation_framework_for_conversational_dense_re.md)**

:   提出 ConvMix 混合准则数据增强框架，从查询和文档双方向用 LLM 进行可扩展的相关性标注增强，并通过聚类多样性选择和 Fisher 信息近分布监督筛选，系统性提升对话式稠密检索性能。

**[Do Retrieval Augmented Language Models Know When They Don't Know?](do_retrieval_augmented_language_models_know_when_they_dont_know.md)**

:   系统分析RAG模型的拒绝校准问题，发现RALM在检索文档全部不相关时过度拒绝率超过55%（即使模型内部知识足够回答），提出结合不确定性估计和拒绝感知微调的机制来平衡拒绝与回答质量。

**[Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs](does_less_hallucination_mean_less_creativity_an_empirical_investigation_in_llms.md)**

:   系统研究三种幻觉缓解方法（CoVe、DoLa、RAG）对LLM创造力的影响，发现它们对发散性创造力有截然相反的效果——CoVe增强、DoLa抑制、RAG无影响——而收敛性创造力基本不受影响，这一规律跨模型家族和参数规模一致成立。

**[Exposing the Cracks: Vulnerabilities of Retrieval-Augmented LLM-Based Machine Translation](exposing_the_cracks_vulnerabilities_of_retrieval-augmented_llm-based_machine_tra.md)**

:   开发受控噪声注入框架系统评估检索增强翻译（REAL-MT），引入Fidelity和CAR两个新指标，在10语言对×4种噪声类型上揭示模型即使面对矛盾上下文仍盲目采纳（CAR保持65-78%），大推理模型（LRM）反而更脆弱（会"合理化"错误上下文），且噪声鲁棒性与干净上下文利用率存在根本性trade-off。

**[HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment](himo-clip_modeling_semantic_hierarchy_and_monotonicity_in_vi.md)**

:   提出 HiMo-CLIP，通过对文本嵌入做 batch 内 PCA 分解（HiDe）提取多粒度语义成分，配合双分支单调性感知对比损失（MoLo），在不修改编码器的前提下让 CLIP 学会"文本越完整、对齐分数越高"的语义单调性，在长文本检索上显著超越现有方法。

**[Knowledge Completes the Vision: A Multimodal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning](knowledge_completes_the_vision_a_multimodal_entity-aware_retrieval-augmented_gen.md)**

:   本文提出MERGE，首个面向新闻图像描述的多模态实体感知RAG框架，通过构建实体中心多模态知识库（EMKB）、假设描述引导的多模态对齐（HCMA）和检索驱动的多模态知识集成（RMKI）三大组件，在GoodNews上CIDEr提升+6.84、F1提升+4.14，并在未见过的Visual News上实现CIDEr +20.17的强泛化。

**[Magnitude Matters: A Superior Class of Similarity Metrics for Holistic Semantic Understanding](magnitude_matters_a_superior_class_of_similarity_metrics_for_holistic_semantic_u.md)**

:   提出两种无参数、幅度感知的向量相似度度量——Overlap Similarity (OS) 和 Hyperbolic Tangent Similarity (HTS)，在 4 个句子嵌入模型和 8 个 NLP 基准上，对分类任务（释义、推理）的 MSE 显著低于 Cosine Similarity 和 Dot Product，且无需任何额外训练开销。

**[Mem-PAL: Towards Memory-based Personalized Dialogue Assistants for Long-term User-Agent Interaction](mem-pal_towards_memory-based_personalized_dialogue_assistants_for_long-term_user.md)**

:   提出H2Memory四层分层异构记忆结构（日志图/背景记忆/主题大纲/原则），通过PAL-Set数据集（100用户×8.4个月交互）验证，在需求重述和方案建议任务上将BLEU-1从13.59提升至26.67。

**[Multimodal DeepResearcher: Generating Text-Chart Interleaved Reports From Scratch with Agentic Framework](multimodal_deepresearcher_generating_text-chart_interleaved_.md)**

:   提出 Multimodal DeepResearcher，一个四阶段 Agent 框架从零生成图文交替研究报告：通过形式化可视化描述（FDV）让 LLM 学习和生成多样化图表，结合 Actor-Critic 迭代精炼机制（LLM生成D3.js代码→浏览器渲染→多模态LLM评审），在自建 MultimodalReportBench 上达到 82% 整体胜率（Claude 3.7），人类评估 100% 胜率。

**[N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs](n2n-gqa_noise-to-narrative_for_graph-based_table-text_question_answering_using_l.md)**

:   提出 N2N-GQA——首个用于开放域混合表格-文本问答的零样本框架，核心思路是将检索到的嘈杂文档构建为动态证据图（文档为节点、TF-IDF共享词为边），通过图中心性剪枝识别"桥接文档"连接多跳推理链，在 OTT-QA 上比 Vanilla RAG 提升 +39.6 EM（从 8.0 到 48.8），零样本即接近微调系统 CORE (49.0 EM)。

**[Neighbor-aware Instance Refining with Noisy Labels for Cross-Modal Retrieval](neighbor-aware_instance_refining_with_noisy_labels_for_cross-modal_retrieval.md)**

:   提出 NIRNL 框架，通过跨模态边距保持（CMP）增强样本区分度，并利用邻域感知实例精炼（NIR）将训练数据三分为纯净/困难/噪声子集，分别定制不同优化策略，统一了鲁棒学习、标签校准和实例选择三种范式，在高噪声率下实现了 SOTA 跨模态检索性能。

**[OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval](opera_a_reinforcement_learning--enhanced_orchestrated_planner-executor_architect.md)**

:   提出 OPERA 框架，通过 Goal Planning Module 和 Reason-Execute Module 的分层架构，结合专为多 agent 设计的 MAPGRPO 训练算法，大幅提升 reasoning-oriented multi-hop retrieval 性能。

**[Positional Bias in Multimodal Embedding Models: Do They Favor the Beginning, the Middle, or the End?](positional_bias_in_multimodal_embedding_models_do_they_favor_the_beginning_the_m.md)**

:   本文首次系统研究多模态表示模型中的位置偏差现象，发现文本编码器倾向于偏好输入开头，而图像编码器在开头和结尾均表现偏好，并通过大量控制实验揭示该偏差源于位置编码方案、训练损失、上下文重要性和图文对训练的多因素共同作用。

**[PRECISE: Reducing the Bias of LLM Evaluations Using Prediction-Powered Ranking Estimation](precise_reducing_the_bias_of_llm_evaluations_using_prediction-powered_ranking_es.md)**

:   将Prediction-Powered Inference（PPI）框架扩展到子实例级别的排序指标（如Precision@K），通过仅30-100条人工标注+大量LLM评判结果获得无偏的排序指标估计，计算复杂度从 $O(2^{|C|})$ 降至 $O(2^K)$，在印度电商搜索场景中成功指导LLM查询改写系统上线。

**[PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning](prime_planning_and_retrieval-integrated_memory_for_enhanced_reasoning.md)**

:   受双系统认知理论启发，提出PRIME多Agent推理框架——Quick Thinking Agent（System 1）快速生成直觉答案，Reflection Agent评估可信度，不确定时触发System 2的6个专门化Agent（规划/搜索/阅读/假设/整合/决策）进行深度知识检索推理，使开源LLaMA 3在医学/多跳QA上接近GPT-4o性能。

**[RAGFort: Dual-Path Defense Against Proprietary Knowledge Base Extraction in Retrieval-Augmented Generation](ragfort_dual-path_defense_against_proprietary_knowledge_base_extraction_in_retri.md)**

:   提出 RAGFort，首个系统性防御 RAG 知识库抽取攻击的双路径框架，通过对比重索引（inter-class）隔离主题间边界和约束级联生成（intra-class）抑制敏感内容输出，在安全性上将知识恢复率降低至无保护的 0.51×，同时保持回答质量。

**[REAP: Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-Hop Question Answering](reap_enhancing_rag_with_recursive_evaluation_and_adaptive_planning_for_multi-hop.md)**

:   提出 REAP 双模块迭代框架，通过子任务规划器 (SP) 维护全局视角动态指导推理轨迹，事实提取器 (FE) 从检索内容中提取结构化事实和潜在线索，两者递归协作解决多跳问答。在 4 个基准上以 Llama-3.1-8B 显著超越所有基线（HotpotQA F1 68.0 vs 次优 63.4）。

**[ReFeed: Retrieval Feedback-Guided Dataset Construction for Style-Aware Query Rewriting](refeed_retrieval_feedback-guided_dataset_construction_for_style-aware_query_rewr.md)**

:   提出一个检索反馈驱动的数据集生成框架，通过识别检索失败case、LLM风格化改写、重检索验证三步闭环，自动构建高质量的风格感知查询改写数据集，为训练检索对齐的改写模型提供数据基础。

**[RRRA: Resampling and Reranking through a Retriever Adapter](rrra_resampling_and_reranking_through_a_retriever_adapter.md)**

:   提出RRRA框架，通过在Bi-Encoder上添加轻量级可学习适配器来建模每个候选文档的假阴性概率，并将其同时用于训练时的负样本重采样和推理时的重排序，在NQ/TQ/MS MARCO上持续超越SimANS/TriSampler等强基线。

**[SR-KI: Scalable and Real-Time Knowledge Integration into LLMs via Supervised Attention](sr-ki_scalable_and_real-time_knowledge_integration_into_llms_via_supervised_atte.md)**

:   提出SR-KI框架，通过两阶段训练（检索层定位 + 注意力监督损失）实现结构化知识库向LLM KV缓存的高效注入，在单块A100 40GB GPU上支持最多40K知识库条目的注入，且通过top-100压缩实现高达99.75%的压缩率，同时保持88%以上的平均Recall@10检索性能。

**[Towards Inference-Time Scaling for Continuous Space Reasoning](towards_inference-time_scaling_for_continuous_space_reasoning.md)**

:   首次系统研究离散文本推理中的inference-time scaling技术能否迁移到连续潜空间推理模型（COCONUT），发现dropout采样能生成多样推理路径（Pass@32达44.43%），但PRM/ORM仅带来不足2.3%提升，根因在于连续思维表示缺乏区分正误推理的几何归纳偏置。

**[When Small Models Are Right for Wrong Reasons: Process Verification for Trustworthy Agents](when_small_models_are_right_for_wrong_reasons_process_verification_for_trustwort.md)**

:   通过分析 10,734 条推理轨迹揭示小型语言模型（7-9B）存在严重的"答对但理由错"（RWR）现象——50-69% 的正确答案包含根本性推理缺陷；提出推理完整性评分（RIS）作为过程级指标，发现 RAG 能有效改善推理质量而元认知干预反而有害，并蒸馏出快速分类器（0.86 F1, 100× 加速）用于实时部署。
