---
title: >-
  ACL2025 NLP理解方向 26篇论文解读
description: >-
  26篇ACL2025 NLP理解方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📖 NLP理解

**💬 ACL2025** · **26** 篇论文解读

**[A Comprehensive Graph Framework for Question Answering with Mode-Seeking Preference Alignment](a_comprehensive_graph_framework_for_question_answering_with_mode-seeking_prefere.md)**

:   提出GraphMPA框架，通过构建基于通用相似度度量的层次化文档图实现全局文档理解，并引入mode-seeking偏好优化替代传统DPO实现更精准的人类偏好对齐，在6个QA数据集上全面超越现有RAG方法。

**[A Variational Approach for Mitigating Entity Bias in Relation Extraction](a_variational_approach_for_mitigating_entity_bias_in_relation_extraction.md)**

:   提出基于变分信息瓶颈（VIB）的实体去偏方法，将实体token映射为高斯分布以选择性压缩实体特定信息、保留上下文语义，在通用/金融/生物医学三个领域的关系抽取数据集上均取得SOTA，特别是在OOD场景下BioRED提升5.3个F1点。

**[Adapting Psycholinguistic Research for LLMs: Gender-Inclusive Language in a Coreference Context](adapting_psycholinguistic_research_for_llms_gender-inclusive_language_in_a_coref.md)**

:   将 Tibblin et al. (2023) 的心理语言学实验从法语适配到英语和德语 LLM，通过测量共指词概率和生成内容分析发现：英语 LLM 基本保持先行词-共指词性别一致但 they 单数几乎不被使用且存在底层男性偏见；德语 Leo Mistral 7B 的男性偏见更强烈（压倒所有 8 种包容策略），但包容策略仍能增加女性/中性性别的出现概率，与心理语言学人类实验结果一致。

**[Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification](analyzing_political_bias_in_llms_via_target-oriented_sentiment_classification.md)**

:   提出基于目标导向情感分类(TSC)的LLM政治偏差分析框架，通过在450个政治句子中替换1319位政治家名字并用7个模型在6种语言中预测情感，定义了基于熵的不一致性指标来量化偏差，发现LLM对左翼和中间派有正面偏见、对极右翼有负面偏见，且更大模型偏差更强更一致。

**[Automatic Generation of Inference Making Questions for Reading Comprehension Assessments](automatic_generation_of_inference_making_questions_for_reading_comprehension_ass.md)**

:   开发了一套阅读理解推理题分类法（代词桥接/文本连接/填补空白），用 GPT-4o few-shot 提示自动生成针对特定推理类型的多项选择题；93.8% 的题目质量合格，但仅 42.6% 准确匹配目标推理类型，说明 LLM 在精确推理能力控制上仍有不足。

**[BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering](belle_a_bi-level_multi-agent_reasoning_framework_for_multi-hop_question_answerin.md)**

:   提出 BELLE 双层多智能体辩论框架，先将多跳问题分类为四种类型，再通过双层辩论机制（第一层正反方辩论 + 第二层快/慢辩论者监督）动态规划 CoT、单步检索、迭代检索等算子的组合方案，实现面向问题类型的自适应多跳推理。

**[BookCoref: Coreference Resolution at Book Scale](bookcoref_book_scale.md)**

:   提出首个书级别共指消解基准BookCoref，通过角色链接+LLM过滤+窗口扩展的自动标注管线，在50本完整小说上生成高质量银标注数据，平均文档长度超过20万tokens。

**[CaLMQA: Exploring Culturally Specific Long-Form Question Answering across 23 Languages](calmqa_cultural_multilingual_qa.md)**

:   构建了首个多语言长文本问答数据集 CaLMQA（51.7K 问题，23 种语言），通过无翻译方式收集文化特异性问题，发现 LLM 回答文化特异性问题的事实性（45-52%）显著低于文化无关问题（64-71%），低资源语言表现尤其差。

**[Can LLMs Reliably Simulate Real Students' Abilities in Mathematics and Reading Comprehension?](can_llms_reliably_simulate_real_students_abilities_in_mathematics_and_reading_co.md)**

:   利用项目反应理论(IRT)将11个LLM与真实学生放在同一能力量表上评估，发现在无引导情况下强模型远超学生平均水平，而"扮演某年级学生"的提示虽能改变表现，但**没有任何模型-提示组合**能在所有学科和年级上可靠模拟平均学生。

**[Disambiguate First, Parse Later: Generating Interpretations for Ambiguity Resolution in Semantic Parsing](disambiguate_first_parse_later_generating_interpretations_for_ambiguity_resoluti.md)**

:   提出"先消歧、后解析"的模块化方法，利用LLM生成默认解释并训练专门的infilling模型补全缺失解释，将歧义自然语言问题转化为多个明确解释后再分别进行SQL解析。

**[Dynamic Order Template Prediction for Generative Aspect-Based Sentiment Analysis](dot_absa_template.md)**

:   本文提出 Dynamic Order Template (DOT) 方法，将 ABSA 情感四元组生成分为两阶段——先预测元组数量并生成初始模板，再基于动态模板生成具体情感元组，在 9 个 ABSA 数据集上实现 SOTA 同时推理时间比 MvP 减少 7 倍。

**[Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering](embqa_embedding_odqa.md)**

:   EmbQA 提出嵌入级 ODQA 框架，用轻量线性层和无监督对比学习优化查询表示实现段落重排序，并引入基于序统计量的探索性嵌入扩展候选答案多样性，配合熵选择机制自动选答，在 4 个 ODQA 数据集上以更低计算成本超越 SuRe 等 prompt 级方法。

**[End-to-End Dialog Neural Coreference Resolution: Balancing Efficiency and Accuracy in Large-Scale Systems](end-to-end_dialog_neural_coreference_resolution_balancing_efficiency_and_accurac.md)**

:   提出一个端到端神经共指消解系统，通过结合上下文嵌入、层次化注意力机制和优化策略（剪枝/量化），在OntoNotes等基准数据集上实现效率与准确率的平衡，SpanBERT达到87.3 F1。

**[Towards a More Generalized Approach in Open Relation Extraction](generalized_open_relation_extract.md)**

:   提出 MixORE 框架，在更通用的 Open Relation Extraction 设定下（无标注数据同时包含已知和新颖关系，且不做长尾或预分割假设），通过 Semantic Autoencoder 检测新关系 + 开放世界半监督联合学习，在 FewRel/TACRED/Re-TACRED 上全面超越 SOTA。

**[Generating Diverse Training Samples for Relation Extraction with Large Language Models](generating_diverse_training_samples_for_relation_extraction_with_large_language_.md)**

:   研究如何用LLM为关系抽取（RE）生成高质量且多样化的训练样本，提出基于ICL的逐条生成策略和基于DPO的多样性微调方法，生成的训练数据可有效提升few-shot RE模型性能。

**[Hierarchical Retrieval with Evidence Curation for Open-Domain Financial QA](hierarchical_retrieval_with_evidence_curation_for_open-domain_financial_question.md)**

:   HiREC 提出分层检索与证据策展框架，先检索相关文档再从中选取段落，并通过过滤无关段落 + 自动生成补充查询来补全缺失信息，在包含 14.5 万篇 SEC 文档的 LOFin 基准上相比最优 RAG 基线提升 13%+ 答案准确率。

**[iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering](iquest_an_iterative_question-guided_framework_for_knowledge_base_question_answer.md)**

:   iQUEST 提出迭代式子问题引导框架，在每一步推理中动态生成当前可解答的子问题以维持推理方向，并结合 GNN 聚合二跳邻居语义信息实现"前瞻性"实体探索，在 CWQ、WebQSP、WebQuestions、GrailQA 四个基准上取得 SOTA 或接近 SOTA 的性能，且无需微调 LLM。

**[Multi-Hop Reasoning for Question Answering with Hyperbolic Representations](multi-hop_reasoning_for_question_answering_with_hyperbolic_representations.md)**

:   通过在 T5 编码器-解码器模型中简单插入一层 Poincaré 双曲层，以最小改动将欧几里得嵌入映射到双曲空间进行多跳推理，实验在四个数据集上一致性地超越欧几里得对照组，并证明了基于 δ-hyperbolicity 初始化曲率的有效性以及双曲空间在层次结构更强的数据集上优势更明显。

**[NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering](neusym_rag_pdf_qa.md)**

:   NeuSym-RAG 提出了一个混合神经-符号检索框架，将 PDF 文档通过多视角分块解析同时存入关系数据库和向量库，LLM Agent 通过可执行动作（SQL 查询 + 向量检索 + 查看图片等）迭代式交互检索，在学术论文 QA 上比经典 RAG 提升 17.3%。

**[QQSUM: A Novel Task and Model of Quantitative Query-Focused Summarization for Review-based Product Question Answering](qqsum_a_novel_task_and_model_of_quantitative_query-focused_summarization_for_rev.md)**

:   提出 QQSUM 任务和 QQSUM-RAG 框架，通过 KP 导向检索与聚类、Next-KP-Generation 训练策略，从产品评论中生成包含多元观点及其流行度量化的 Key Point 摘要，解决传统 PQA 只输出单一视角答案的问题。

**[Recursive Question Understanding for Complex Question Answering over Heterogeneous Personal Data](recursive_question_understanding_for_complex_question_answering_over_heterogeneo.md)**

:   提出 ReQAP 方法，通过递归问题分解构建可执行算子树，在结构化+非结构化的异构个人数据上实现复杂问答，支持端侧轻量部署。

**[ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision](rescore_multihop_qa.md)**

:   提出 ReSCORE，利用 LLM 生成的文档-问题相关性（relevance）和文档-答案一致性（consistency）的联合概率作为伪标签，在迭代 RAG 框架中无监督训练 dense retriever，在三个多跳 QA 数据集上达到 SOTA。

**[Rethinking Semantic Parsing for Large Language Models: Enhancing LLM Performance with Semantic Hints](rethinking_semantic_parsing_for_large_language_models_enhancing_llm_performance_.md)**

:   针对"语义解析结果直接输入LLM反而降低性能"这一反直觉现象，提出SENSE——一种在prompt中嵌入语义提示（而非显式解析结果）的零样本方法，在GLUE理解任务、机器翻译、复述和简化等生成任务上一致性地提升了LLM表现。

**[Self-Critique Guided Iterative Reasoning for Multi-hop Question Answering](self-critique_guided_iterative_reasoning_for_multi-hop_question_answering.md)**

:   提出 SiGIR 框架，通过端到端训练使模型具备迭代问题分解、检索、推理和自我评估能力，在推理阶段利用自我批评反馈引导 iteration-level beam search 选择最优推理路径，在三个多跳 QA 数据集上平均超越 SOTA 8.6%。

**[Sentiment Reasoning for Healthcare](sentiment_reasoning_for_healthcare.md)**

:   提出"情感推理"（Sentiment Reasoning）新任务，要求模型在预测医疗对话情感标签的同时生成解释理据，并构建了覆盖五种语言的 30K 样本多模态情感分析数据集，通过理据增强训练在分类准确率和 macro-F1 上均提升约 2%。

**[SynGraph: A Dynamic Graph-LLM Synthesis Framework for Sparse Streaming User Sentiment Analysis](syngraph_a_dynamic_graph-llm_synthesis_framework_for_sparse_streaming_user_senti.md)**

:   提出SynGraph框架，在连续时间动态图上将稀疏用户分为mid-tail/long-tail/extreme三类，针对不同稀疏程度利用LLM合成增强数据（结合局部-全局图理解、高阶关系和画像生成），有效缓解流式评论情感分析中的数据稀疏问题。
