---
title: >-
  ACL2025 图学习方向 23篇论文解读
description: >-
  23篇ACL2025 图学习方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🕸️ 图学习

**💬 ACL2025** · 共 **23** 篇

**[Beyond Completion A Foundation Model For General Knowledge Graph Reasoning](beyond_completion_a_foundation_model_for_general_knowledge_graph_reasoning.md)**

:   提出 MERRY，一个统一处理 KG 内（零样本 KGC）和 KG 外（KGQA）推理任务的知识图谱基础模型，通过多视角条件消息传递 (CMP) 融合文本和结构信息，在 28 个数据集上超越现有方法。

**[Can Graph Neural Networks Learn Language](can_graph_neural_networks_learn_language.md)**

:   提出 Morpher，首个图-文多模态 prompt learning 范式——在冻结 GNN 和 LLM 参数的前提下，同时学习图 prompt 和文本 prompt + 跨模态投影器，用极弱文本监督（仅类别名几个词）将 GNN 表征对齐到 LLM 语义空间，首次实现 GNN 的 CLIP 式零样本图分类。

**[Claimpkg Enhancing Claim Verification Via Pseudo-Subgraph Generation With Lightw](claimpkg_enhancing_claim_verification_via_pseudo-subgraph_generation_with_lightw.md)**

:   提出 ClaimPKG 框架，通过轻量级专用 LLM 将文本声明转换为伪子图表示，再从知识图谱中检索相关子图作为证据，最终由通用 LLM 进行推理验证，在 FactKG 数据集上比 SOTA 高出 9%-12% 准确率。

**[Croppable Knowledge Graph Embedding](croppable_knowledge_graph_embedding.md)**

:   提出 MED 框架训练"可裁剪"知识图谱嵌入——一次训练同时优化 64 个不同维度的子模型（共享嵌入前缀），通过互学习、进化改进和动态损失权重，各维度子模型直接裁剪使用即超越独立训练和蒸馏方法，训练速度快 10 倍。

**[Cross-Document Contextual Coreference Resolution In Knowledge Graphs](cross-document_contextual_coreference_resolution_in_knowledge_graphs.md)**

:   提出基于知识图谱的跨文档共指消解方法，通过动态链接机制将文本实体提及与知识图谱节点关联，结合上下文嵌入和图消息传递推理提升跨文档实体识别的精度和召回率，在多个基准数据集上超越传统方法。

**[Disentangled Multi-Span Evolutionary Network Against Temporal Knowledge Graph Re](disentangled_multi-span_evolutionary_network_against_temporal_knowledge_graph_re.md)**

:   提出 DiMNet，通过多跨度演化策略和跨时间解耦机制，分离节点语义的活跃/稳定特征，显著提升时序知识图谱（TKG）外推推理性能，在四个基准数据集上取得 SOTA。

**[Extending Complex Logical Queries Uncertain Knowledge Graphs](extending_complex_logical_queries_uncertain_knowledge_graphs.md)**

:   提出在不确定知识图谱（Uncertain KG）上进行软查询（Soft Query）的新问题设定 SQUK，结合必要性（necessity）和重要性（importance）扩展一阶逻辑查询语义，并设计带校准的神经符号推理方法 SRC，避免前向推理中的级联错误。

**[Fast-And-Frugal Text-Graph Transformers Are Effective Link Predictors](fast-and-frugal_text-graph_transformers_are_effective_link_predictors.md)**

:   提出 Fast-and-Frugal Text-Graph (FnF-TG) Transformer，通过 Transformer 的自注意力机制统一编码文本描述和图结构（ego-graph），在归纳链接预测任务上以小 BERT 模型超越了使用大 BERT+MPNN 的 SOTA，同时首次扩展到完全归纳设置（关系也可归纳）。

**[Fidelis Faithful Reasoning In Large Language Model For Knowledge Graph Question ](fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_.md)**

:   提出 FiDeLiS 框架，通过 Path-RAG 预选候选集缩小搜索空间 + 演绎验证beam search (DVBS) 逐步构建并验证推理路径，在无需训练的情况下提升 LLM 在知识图谱问答中的准确性和可解释性。

**[Graphcheck Breaking Long-Term Text Barriers With Extracted Knowledge Graph-Power](graphcheck_breaking_long-term_text_barriers_with_extracted_knowledge_graph-power.md)**

:   GraphCheck 提出一种图增强的事实核查框架，利用 LLM 从文档和声明中提取知识图谱三元组，通过 GNN 编码图结构并作为 soft prompt 注入冻结的 LLM 验证器，在单次推理调用中实现细粒度事实核查，在7个基准上平均提升 7.1%，且在医学领域表现出强泛化能力。

**[Graphnarrator](graphnarrator.md)**

:   提出GraphNarrator——首个为图神经网络生成自然语言解释的方法，通过将显著性图解释"语言化"为文本段落、用Expert Iteration迭代优化伪标签质量、最终蒸馏到端到端解释器模型，在三个数据集上生成的解释在忠实度、简洁性和人类偏好上均优于GPT-4o零样本解释。

**[Kg Llm Trustworthy Qa](kg_llm_trustworthy_qa.md)**

:   提出开放域知识图谱问答基准 OKGQA 及其扰动变体 OKGQA-P，通过统一的图引导检索-生成框架系统性地验证了 KG 增强可以有效降低 LLM 幻觉率（FActScore 提升约 20 个百分点），子图检索在各类查询上表现最优且对 KG 噪声具有鲁棒性。

**[Kg Rag Recommendation](kg_rag_recommendation.md)**

:   提出 K-RagRec 框架，将知识图谱（KG）中的结构化关系信息引入 LLM 推荐系统的 RAG 流程——从 KG 中检索高质量的结构化实体关系信息来增强推荐生成，解决纯文本 RAG 忽略结构关系和引入噪声的问题。

**[M3Hg Multimodal Multi-Scale And Multi-Type Node Heterogeneous Graph For Emotion ](m3hg_multimodal_multi-scale_and_multi-type_node_heterogeneous_graph_for_emotion_.md)**

:   提出 M3HG 模型，通过构建多模态多类型节点异构图来显式建模对话中的情感与原因上下文，并在句间和句内两个尺度上融合语义信息，实现多模态对话中情感-原因三元组的端到端提取。同时构建了首个中文多场景 MECTEC 数据集 MECAD。

**[Mrakl Multilingual Retrieval-Augmented Knowledge Graph Construction For Low-Reso](mrakl_multilingual_retrieval-augmented_knowledge_graph_construction_for_low-reso.md)**

:   将多语言知识图谱构建（mKGC）重新定义为 QA 任务，提出基于 RAG 的 mRAKL 系统，利用非结构化单语数据作为检索源来克服低资源语言中结构化数据匮乏的困难，在 Tigrinya 和 Amharic 两种低资源语言上显著超越已有方法。

**[Multimodal Transformers Are Hierarchical Modal-Wise Heterogeneous Graphs](multimodal_transformers_are_hierarchical_modal-wise_heterogeneous_graphs.md)**

:   从图论视角证明了多模态 Transformer（MulTs）本质上是层次化模态异质图（HMHG），并基于此提出 GsiT 模型，通过 Interlaced Mask 机制实现仅 1/3 参数的 All-Modal-In-One 融合，同时性能显著超越传统 MulTs。

**[Ontology-Guided Reverse Thinking Makes Large Language Models Stronger On Knowled](ontology-guided_reverse_thinking_makes_large_language_models_stronger_on_knowled.md)**

:   提出 ORT（Ontology-Guided Reverse Thinking），利用知识图谱本体结构从目标逆向构建标签推理路径，引导知识检索，显著提升 LLM 的知识图谱问答能力。

**[Paper 2401 14640](paper_2401_14640.md)**

:   提出 CAQA 基准，利用知识图谱自动生成包含四类归因类别（支持、部分支持、矛盾、无关）和四种推理复杂度的大规模问答归因评估数据集（161K 样本），系统性地评测了 25 种自动归因评估器的能力。

**[Predicate-Conditional Conformalized Answer Sets For Knowledge Graph Embeddings](predicate-conditional_conformalized_answer_sets_for_knowledge_graph_embeddings.md)**

:   提出 CondKGCP——基于谓词条件的 conformal prediction 方法用于知识图谱嵌入的不确定性量化，通过合并相似谓词增大校准集+双重校准（score+rank）减小预测集大小，在保证谓词级条件覆盖率的同时输出更紧凑的答案集，在多个KGE基准上优于5个baseline。

**[Rscf Relationsemantics Consistent Filter For Entity](rscf_relationsemantics_consistent_filter_for_entity.md)**

:   提出 RSCF 插件式 KGE 方法，通过共享仿射变换 + 根植实体变换 + 归一化三特征确保"语义相似的关系产生相似的实体变换"（关系语义一致性），在距离模型和张量分解模型上均显著超越 SOTA，并从理论和实验上验证了一致性保持率。

**[Simgrag Leveraging Similar Subgraphs For Knowledge Graphs Driven Retrieval-Augme](simgrag_leveraging_similar_subgraphs_for_knowledge_graphs_driven_retrieval-augme.md)**

:   提出 SimGRAG 方法，通过"查询→模式图→子图"两阶段对齐策略，利用 LLM 将查询转化为图模式，再用图语义距离（GSD）度量在知识图谱中高效检索语义最相似的子图，实现即插即用的 KG 驱动 RAG，在问答和事实验证任务上超越所有现有方法。

**[The Role Of Exploration Modules In Small Language Models For Knowledge Graph Comp](the_role_of_exploration_modules_in_small_language_models_for_knowledge_graph_comp.md)**

:   本文发现小语言模型（SLM）在知识图谱问答中的性能瓶颈在于图探索（exploration）阶段，提出用轻量级检索模块（SentenceBERT/GTR）替代 SLM 自身进行知识图谱遍历，显著提升了 SLM 在 KGQA 任务上的表现。

**[The Role Of Exploration Modules In Small Language Models For Knowledge Graph Que](the_role_of_exploration_modules_in_small_language_models_for_knowledge_graph_que.md)**

:   本文发现小语言模型（SLM）在知识图谱问答（KGQA）中使用Think-on-Graph框架时，性能瓶颈在于KG探索阶段而非推理阶段，通过引入轻量级段落检索模块（SentenceBERT/GTR，~110M参数）替代SLM自身进行KG遍历，可显著提升SLM在KGQA上的表现。
