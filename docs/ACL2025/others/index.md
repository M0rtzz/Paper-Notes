---
title: >-
  ACL2025 其他方向287篇论文解读
description: >-
  287篇ACL2025的其他方向论文解读，涵盖推理、对话系统、对齐/RLHF、对抗鲁棒、Agent、语音等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "其他"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "对话系统"
  - "对齐/RLHF"
  - "对抗鲁棒"
  - "Agent"
  - "语音"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📂 其他

**💬 ACL2025** · **287** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (91)](../../ICML2026/others/index.md) · [💬 ACL2026 (1)](../../ACL2026/others/index.md) · [📷 CVPR2026 (35)](../../CVPR2026/others/index.md) · [🔬 ICLR2026 (74)](../../ICLR2026/others/index.md) · [🤖 AAAI2026 (122)](../../AAAI2026/others/index.md) · [🧠 NeurIPS2025 (142)](../../NeurIPS2025/others/index.md)

🔥 **高频主题：** 推理 ×16 · 对话系统 ×12 · 对齐/RLHF ×12 · 对抗鲁棒 ×7 · Agent ×7

**[Towards Robust ESG Analysis Against Greenwashing Risks: A3CG](a3cg_esg_greenwashing.md)**

:   提出 A3CG 数据集和方面-行动分析任务（从可持续性声明中提取方面及其行动类型：已实施/计划中/不确定），通过跨类别泛化设置评估 NLP 方法抵御漂绿风险的鲁棒性，发现监督学习（GRACE F1=47.51）优于 LLM（Claude 3.5 F1=42.03）但泛化效率更差。

**[A Dual-Mind Framework for Strategic and Expressive Negotiation Agent](a_dual-mind_framework_for_strategic_and_expressive_negotiation_agent.md)**

:   本文受人类认知双过程理论启发，提出双心智谈判智能体框架（DMNA），将直觉模块（快速策略规划，基于 MCTS+DPO 训练）和审慎模块（慢速表达优化，基于多面反思机制）相结合，在谈判任务上取得了最优性能。

**[Barec: A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment](a_large_and_balanced_corpus_for_fine-grained_arabic_readability_assessment.md)**

:   构建 Barec——首个大规模、平衡、细粒度的阿拉伯语可读性评估语料库（69K+ 句子、100 万+词、19 个等级），由 6 名专业教育者标注，并基准测试了 4 种阿拉伯语 BERT 模型 × 4 种输入变体 × 5 种损失函数，发现形态学分词输入 D3Tok 配合回归损失在 QWK 上达到 84.0%。

**[A Little Human Data Goes A Long Way](a_little_human_data_goes_a_long_way.md)**

:   通过在8个事实验证和问答数据集上的大规模实验，证明了在合成数据中混入极少量人工标注数据（甚至仅125个样本）就能显著提升模型性能，替换最后10%的人工数据会导致性能严重下降，且200条人工数据的增益需要数量级更多的合成数据才能匹配。

**[A Measure of the System Dependence of Automated Metrics](a_measure_of_the_system_dependence_of_automated_metrics.md)**

:   揭示 MT 自动评估指标 "尺子因被测物不同而改变长度" 的系统依赖性问题，提出 SysDep 度量来量化不同翻译系统被指标高估/低估的程度。

**[A Multi-Persona Framework for Argument Quality Assessment](a_multi-persona_framework_for_argument_quality_assessment.md)**

:   本文提出 MPAQ 框架，通过大语言模型模拟多个不同评估者视角（persona），对论点进行多角度质量评估，并设计粗到细的评分策略（先整数再小数），在 IBM-Rank-30k 和 IBM-ArgQ-5.3kArgs 数据集上显著超越现有基线，同时提供了可解释的多视角评估理由。

**[A New Formulation of Zipf's Meaning-Frequency Law through Contextual Diversity](a_new_formulation_of_zipfs_meaning-frequency_law_through_contextual_diversity.md)**

:   本文提出将 Zipf 语义-频率定律重新形式化为词频与上下文多样性之间的幂律关系，通过语言模型生成的上下文化词向量方向来量化词义数量，发现该定律在小规模语言模型中不可观测，且自回归 LM 需要比掩码 LM 多得多的参数才能呈现该定律。

**[A Practical Approach for Building Production-Grade Conversational Agents with Workflow Graphs](a_practical_approach_for_building_production-grade_conversational_agents_with_wo.md)**

:   提出基于有向无环图(DAG)的工作流框架，通过将LLM agent的复杂业务约束分解到图中不同状态节点，并结合响应掩码微调策略，构建满足生产级要求的电商对话代理，在任务准确率和格式遵循方面均大幅超越GPT-4o基线。

**[SSUF: A Semi-supervised Scalable Unified Framework for E-commerce Query Classification](a_semi-supervised_scalable_unified_framework_for_e-commerce_query_classification.md)**

:   提出电商查询分类统一框架 SSUF，通过三个可插拔模块——标签增强（BERT 语义编码标签）、知识增强（LLM 世界知识 + 后验点击 + 半监督标签生成）、结构增强（共现/语义/层级三图融合 GCN）——解决短查询信息不足和"马太效应"恶性循环问题，在 JD.COM 意图分类和品类分类任务上 Macro F1 分别达到 49.46 和 41.22（均超 SMGCN 等 SOTA），已上线服务带来显著商业价值。

**[A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior](a_spatio-temporal_point_process_for_fine-grained_modeling_of_reading_behavior.md)**

:   本文提出基于时空标记点过程（marked spatio-temporal point process）的阅读行为概率模型，联合建模注视何时发生、落在哪里、持续多久，发现 Hawkes 过程配合读者特定效应和方向偏移能显著提升扫视预测，但 surprisal 等语言学预测因子仅带来微弱改善——这对现有 surprisal 理论提出了质疑。

**[AceCoder: Acing Coder RL via Automated Test-Case Synthesis](acecoder_acing_coder_rl_via_automated.md)**

:   构建 AceCode-87K（87K 编码题 + 138 万自动合成测试用例），训练代码专用 Reward Model（7B 超越 340B Nemotron），Best-of-N 提升 Llama-3.1-8B 平均 8.9 分，R1 风格从 base 直接 RL 仅 80 步 HumanEval+ 提升 22.5%。

**[Achieving Certification-by-Design Through Model-Driven Development](achieving_certification-by-design_through_model-driven_development.md)**

:   本文提出了一种"认证驱动设计"（Certification-by-Design）的模型驱动开发方法，通过在 NLP 系统的设计阶段就嵌入安全认证要求，使得最终系统能够自动满足相关行业标准和法规，减少事后认证的高昂成本。

**[ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Clause Retrieval](acord_an_expert-annotated_retrieval_dataset_for_legal_contract_drafting.md)**

:   构建首个面向合同起草的专家标注条款检索基准ACORD（114查询、126K+对、1-5星评分），评估20种检索方法发现BM25+GPT-4o pointwise重排序最优（NDCG@5=76.9%），但高质量条款精度极低（5星precision@5仅17.2%），揭示模型距真实律师需求的巨大差距。

**[ACT: Knowledgeable Agents to Design and Perform Complex Tasks](act_knowledgeable_agents_to_design_and_perform_complex_tasks.md)**

:   本文提出 ACT 框架，让多个 LLM 智能体通过协作设计任务并相互获取结构化知识描述，使每个智能体不仅理解自身任务还了解他人的工作方式，从而在创意写作和工具使用等复杂任务上显著超越现有多智能体方法。

**[Adaptive Feature-based Low Rank Plus Sparse Decomposition for Subspace Clustering](adaptive_feature-based_low_rank_plus_sparse_decomposition_for_subspace_clusterin.md)**

:   本文提出一种自适应特征驱动的低秩加稀疏矩阵分解方法，通过在特征空间中自适应学习低秩和稀疏成分的权重，解决子空间聚类中噪声鲁棒性和特征判别力不足的问题。

**[Adaptive Retrieval without Self-Knowledge? Bringing Uncertainty Back Home](adaptive_retrieval_without_self-knowledge_bringing_uncertainty_back_home.md)**

:   对 35 种自适应检索方法（含 8 种最新方法和 27 种不确定性估计方法）进行了全面评测，发现经典的不确定性估计技术在效率和自知能力方面往往优于复杂的专用流水线，同时保持相当的 QA 性能。

**[Advancing Sequential Numerical Prediction in Autoregressive Models](advancing_sequential_numerical_prediction_in_autoregressive_models.md)**

:   提出Numerical Token Integrity Loss (NTIL)——一种双层级数值预测损失函数，在token级别用指数位置加权的EMD替代交叉熵以保持数值有序性，在序列级别通过可微数值构造进行整体数值偏差惩罚，在目标检测、文字检测、数学推理和时钟识别等任务上显著提升自回归模型的数值预测精度。

**[AIDE: Attribute-Guided Multi-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)**

:   提出AIDE框架，通过"属性引导+Persona增强+残差连接"的多跳数据扩展机制，从仅10个种子样本生成约3K条高质量任务特定训练数据，微调Mistral-7B后在zero-shot上平均超越人工标注数据微调6%、超越Evol-Instruct等SOTA方法30%。

**[ALGEN: Few-Shot Inversion Attacks on Textual Embeddings via Cross-Model Alignment](algen_few-shot_inversion_attacks_on_textual_embeddings_via_cross-model_alignment.md)**

:   本文提出ALGEN，一种少样本文本嵌入反转攻击方法，通过将受害者的嵌入空间与攻击者的嵌入空间进行线性对齐，再利用训练好的嵌入到文本生成器重建原始文本，仅需1个泄露样本即可发起部分成功的攻击，1000个样本时Rouge-L达45.75。

**[Aligned but Blind: Alignment Increases Implicit Bias by Reducing Awareness of Race](aligned_but_blind_implicit_bias.md)**

:   揭示对齐训练的"种族盲视"副作用：对齐使 LLM 在歧义上下文中不再将 black/white 表征为种族概念，安全护栏因此无法激活，导致隐式偏见从 64.1% 飙升至 91.4%；反直觉地，在早期层注入种族感知激活（而非遗忘）可将隐式偏见从 97.3% 降至 42.4%。

**[AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment](ambik_dataset_of_ambiguous_tasks_in_kitchen_environment.md)**

:   提出 AmbiK，一个专门用于厨房环境中歧义指令检测的纯文本数据集，包含 1000 对歧义/非歧义指令，按三种歧义类型（用户偏好/常识/安全）分类标注，并评估了多种基于 conformal prediction 的歧义检测方法，发现现有方法在该基准上表现很差。

**[An Analysis of Datasets, Metrics and Models in Keyphrase Generation](an_analysis_of_datasets_metrics_and_models_in_keyphrase_generation.md)**

:   对关键短语生成（keyphrase generation）领域50+篇论文进行系统性分析，揭示了基准数据集高度相似、评估指标计算不一致导致性能被高估等关键问题，并发布了一个强力PLM-based模型以促进未来研究。

**[AnalyticKWS: Towards Exemplar-Free Analytic Class Incremental Learning for Small-footprint Keyword Spotting](analytickws_towards_exemplar-free_analytic_class_incremental_learning_for_small-.md)**

:   提出 AnalyticKWS，一种无需存储历史样本的关键词检测增量学习方法，通过冻结特征提取器 + 递归最小二乘解析解更新分类器，在 GSC 和 SC-100 数据集上超过了所有基于样本回放的方法，且训练时间和内存开销极低。

**[Anything Goes? A Crosslinguistic Study of (Im)possible Language Learning in LMs](anything_goes_a_crosslinguistic_study_of_impossible_language_learning_in_lms.md)**

:   在12种语言上训练GPT-2 small，系统性测试语言模型是否能区分可能语言(自然语言)与不可能语言(打乱词序等)，发现LM展现出部分类人的学习偏向但并非完美——能在单语言内区分但无法跨语言完全分离，而名词短语词序实验中泛化测试(而非困惑度)能反映类型学偏好。

**[Are Any-to-Any Models More Consistent Across Modality Transfers Than Specialists?](are_any-to-any_models_more_consistent_across_modality_transfers_than_specialists.md)**

:   本文提出 ACON 数据集和三种一致性评估标准（循环一致性、前向等变性、共轭等变性），发现当前 any-to-any 模型在逐点评估中并不比专用模型组合更具跨模态一致性，但通过多编辑操作的分布式分析可以观察到弱一致性。

**[Are Bias Evaluation Methods Biased?](are_bias_evaluation_methods_biased.md)**

:   严格控制变量后比较三种主流偏见评估方法（结构化问答 BBQ、LLM-as-a-Judge、情感分析），发现不同方法对同一组 LLM 产生显著不同的偏见排名——偏见评估方法本身就是有偏的，企业不应依赖单一偏见基准来选择模型。

**[ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search](arise_risk_adaptive_search.md)**

:   提出 ARise 框架，将贝叶斯风险评估与动态 RAG 集成到蒙特卡洛树搜索中，解决知识增强推理中的错误传播和验证瓶颈问题，在多跳QA任务上平均准确率超 SOTA KAR 方法 23.10%，超 RAG-equipped 推理模型（DeepSeek-R1）25.37%。

**[ASPERA: A Simulated Environment to Evaluate Planning for Complex Action Execution](aspera_a_simulated_environment_to_evaluate_planning_for_complex_action_execution.md)**

:   提出 ASPERA 框架及 Asper-Bench 基准，用于评估 LLM 在自定义助手库约束下执行复杂多步动作规划（程序生成）的能力，揭示了相比自由代码生成，基于自定义 API 库的程序生成对 LLM 构成显著更大的挑战。

**[Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)**

:   发现并行上下文编码导致 query token 的注意力熵异常升高是性能下降的关键因素，并提出 Attention Sink 共享前缀和 Selective Attention 两种免微调方法有效缓解该问题。

**[Autalic: A Dataset for Anti-Autistic Ableist Language In Context](autalic_a_dataset_for_anti-autistic_ableist_language_in_context.md)**

:   提出 Autalic——首个专注于上下文中反自闭症残障歧视语言检测的数据集，包含 2,400 条 Reddit 句子及上下文标注，由神经多样性背景的专家标注，实验揭示当前 LLM（包括 DeepSeek、Llama3、Gemma2、Mistral）在识别反自闭症歧视语言时与人类判断严重不一致（平均 Cohen's Kappa 仅 0.091），凸显该任务的困难性。

**[AutoMixer: Checkpoint Artifacts as Automatic Data Mixers](automixer_checkpoint_artifacts_as_automatic_data_mixers.md)**

:   提出 AutoMixer 框架，利用训练过程中保存的检查点模型作为"数据混合器"，通过聚合多检查点的一阶影响函数近似来重新分组和加权训练数据，在八个推理基准上取得最高 1.93% 的性能提升。

**[Balancing the Budget: Understanding Trade-offs Between Supervised and Preference-Based Finetuning](balancing_the_budget_understanding_trade-offs_between_supervised_and_preference-.md)**

:   系统研究了在固定数据标注预算下，如何在监督微调（SFT）和偏好微调（PFT/DPO）两个阶段之间最优分配资源，发现低数据量时纯 SFT 最优，高预算时组合使用效果最佳，且仅将 <10% 预算分配给 SFT 就能解决 DPO 的冷启动问题并带来 15-20% 的数学推理提升。

**[Battling against Tough Resister: Strategy Planning with Adversarial Game for Non-collaborative Dialogues](battling_against_tough_resister_strategy_planning_with_adversarial_game_for_non-.md)**

:   本文提出一种基于对抗博弈的策略规划框架，用于处理非合作对话（如说服、谈判）中面对顽固对手时的策略选择问题，通过建模对话双方的对抗动态来生成更有效的说服策略。

**[Behavioural vs. Representational Systematicity in End-to-End Models: An Opinionated Survey](behavioural_vs_representational_systematicity_in_end-to-end_models_an_opinionate.md)**

:   这篇观点性综述区分了行为系统性（模型能否正确泛化到新组合）和表征系统性（模型内部表征是否结构化），用 Hadley 的弱/准/强三级分类审视了语言和视觉领域的主流基准，发现大多数现有基准仅测试弱或准系统性，并呼吁通过机械可解释性方法弥补行为与表征评估的鸿沟。

**[Better Embeddings with Coupled Adam](better_embeddings_with_coupled_adam.md)**

:   从理论上证明 Adam 优化器的逐 token 二阶矩是导致 LLM 词嵌入各向异性（均值偏移）的根因，提出 Coupled Adam——对嵌入层的二阶矩取词汇平均——消除了各向异性问题，并在大规模实验中提升了嵌入质量和下游性能。

**[Beyond Position: the emergence of wavelet-like properties in Transformers](beyond_position_the_emergence_of_wavelet-like_properties_in_transformers.md)**

:   通过频率分析和小波分解，揭示了使用 RoPE 位置编码的 Transformer 模型中注意力头自发涌现出类小波（wavelet-like）的多分辨率处理特性，以弥补 RoPE 在位置精度和频率分辨率之间的固有权衡。

**[BIG-Bench Extra Hard](big-bench_extra_hard.md)**

:   为应对 BIG-Bench Hard 被前沿模型饱和的问题，Google DeepMind 推出 BIG-Bench Extra Hard (BBEH)，用 23 个更难的任务替换 BBH 中的对应任务，最强通用模型仅达 9.8%（调和平均）、最强推理模型达 44.8%，揭示了 LLM 在通用推理上的巨大差距。

**[Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation](bone_soups_multi_objective_gen.md)**

:   提出 Bone Soup 模型合并方法，通过先构造"骨架奖励"（多目标奖励的组合）训练骨架模型、再用对称循环矩阵映射确定合并系数，解决了 Rewarded Soup 中单目标模型合并的次优性问题，在三个多目标生成任务上实现更好的 Pareto 前沿和可控性。

**[Bregman Conditional Random Fields: Sequence Labeling with Parallelizable Inference](bregman_conditional_random_fields_sequence_labeling_with_parallelizable_inferenc.md)**

:   提出 Bregman CRF (Bcrf)，一种基于均值正则化（mean regularization）的新型序列标注判别模型，使用迭代 Bregman 投影实现可并行化的推理算法，替代传统 CRF 中固有顺序的 Viterbi/Forward 算法，在 POS/NER/分词任务上性能与标准 CRF 持平但更快，且在有禁止标签转移约束的场景下优于 Mean Field 方法。

**[Building Better: Avoiding Pitfalls in Developing Language Resources when Data is Scarce](building_better_avoiding_pitfalls_in_developing_language_resources_when_data_is_.md)**

:   通过对 81 名低资源语言 NLP 研究者和标注者的调查，揭示了低资源语言数据构建中的质量问题（数据不自然、文化失当）和伦理问题（标注者劳动被剥削、署名不公），并提出六条改进建议。

**[Byte Latent Transformer: Patches Scale Better Than Tokens](byte_latent_transformer.md)**

:   提出 Byte Latent Transformer (BLT)，一种无分词器的字节级 LLM 架构，通过基于熵的动态分组将字节聚合为可变长度 patch，首次在 8B 规模上匹配 token-based 模型性能，同时解锁了通过同时增大 patch 和模型尺寸来提升推理效率的新 scaling 维度。

**[CADReview: Automatically Reviewing CAD Programs with Error Detection and Correction](cadreview_automatically_reviewing_cad_programs_with_error_detection_and_correcti.md)**

:   提出 CAD 程序审查任务及 ReCAD 框架，基于参考图像自动检测 CAD 程序中的错误并生成修正反馈，构建了包含 20K+ 样本（8 类错误）的 CADReview 数据集。

**[Can Third-parties Read Our Emotions?](can_third_parties_read_our_emotions.md)**

:   本文通过人类被试实验，系统比较了第三方标注者（人类标注者和LLM）与第一方（作者自标注）在情感识别任务中的对齐程度，发现第三方标注与作者真实情感之间存在显著差距，LLM虽优于人类标注者，但仍表现不佳；人口统计学相似性可提升标注质量。

**[Can Uniform Meaning Representation Help GPT-4 Translate from Indigenous Languages?](can_uniform_meaning_representation_help_gpt-4_translate_from_indigenous_language.md)**

:   探索将统一意义表示（UMR）语义图纳入 GPT-4 提示中，翻译三种原住民语言（纳瓦霍语、阿拉帕霍语、库卡马语），发现在大多数情况下 UMR 的加入带来统计显著的性能提升。

**[Capacity Matters: A Proof-of-Concept for Transformer Memorization on Real-World Data](capacity_matters_a_proof-of-concept_for_transformer_memorization_on_real-world_d.md)**

:   本文以SNOMED医学知识图谱为数据源，系统研究了decoder-only Transformer在结构化数据上的记忆容量，发现嵌入维度是决定学习速度和容量的主要因素，而增加层数收效甚微，Softmax激活函数表现最稳定。

**[Causal Estimation of Tokenisation Bias](causal_tokenisation_bias.md)**

:   本文首次将 tokeniser 选择对语言模型输出的影响定义为"分词偏差"(tokenisation bias)，并利用因果推断中的断点回归设计(RDD)来量化这一效应——发现当一个 subword 被纳入词表时，其对应字符串的概率最高可提升 17 倍（小模型），揭示分词是语言建模中一个被低估的关键设计选择。

**[Cautious Next Token Prediction](cautious_next_token_prediction.md)**

:   提出 Cautious Next Token Prediction (CNTP)，一种无需训练的自适应解码策略：在模型预测熵较高（不确定）时采样多条候选路径至标点处，选择困惑度最低的路径作为最终续写，从而在不牺牲多样性的前提下显著提升准确性。

**[ChartLens: Fine-Grained Visual Attribution in Charts](chartlens_fine-grained_visual_attribution_in_charts.md)**

:   提出图表的事后细粒度视觉归因（Post-Hoc Fine-grained Visual Attribution）任务，设计 ChartLens 算法利用分割技术标记图表元素并通过 Set-of-Marks 提示多模态 LLM 进行精确归因，同时构建 ChartVA-Eval 基准，在三类图表上取得 26-66% 的 F1 提升。

**[ChildMandarin: A Comprehensive Mandarin Speech Dataset for Young Children Aged 3-5](childmandarin_a_comprehensive_mandarin_speech_dataset_for_young_children_aged_3-.md)**

:   提出 ChildMandarin，一个面向 3-5 岁幼儿的普通话语音数据集，包含 397 名说话人、41.25 小时语音、覆盖中国 22 个省级行政区，并在 ASR 和说话人验证任务上提供了全面的基线评估。

**[ChuLo: Chunk-Level Key Information Representation for Long Document Understanding](chulo_chunk-level_key_information_representation_for_long_document_understanding.md)**

:   ChuLo 的核心不是单纯把长文档切小，而是先在全文范围内找出最关键的语义短语，再把这些关键信息重新注入每个 chunk 的表示里，从而在只用紧凑块表示的前提下，同时保住全局语义和细粒度 token 信息。

**[CiteEval: Principle-Driven Citation Evaluation for Source Attribution](citeeval_principle-driven_citation_evaluation_for_source_attribution.md)**

:   本文提出 CiteEval，一个基于原则驱动的引用评估框架，通过考虑完整检索上下文、超越检索的多种上下文以及细粒度评价标准，构建了 CiteBench 基准和 CiteEval-Auto 自动指标，在引用质量评估上显著优于基于 NLI 的现有方法。

**[CLaC at SemEval-2025 Task 6: A Multi-Architecture Approach for Corporate Environmental Promise Verification](clac_at_semeval-2025_task_6_a_multi-architecture_approach_for_corporate_environm.md)**

:   本文针对SemEval-2025 Task 6（PromiseEval）的企业ESG报告承诺验证任务，探索了三种递进的模型架构：ESG-BERT基线、语言特征增强版、以及融合注意力池化和多目标学习的联合子任务模型，最终以0.5268的私榜分数略超基线（0.5227），验证了语言特征工程和多任务学习在ESG承诺验证中的有效性。

**[CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model](coachme_sport_instruction.md)**

:   提出 CoachMe，通过对比学习者动作与参考动作的差异（时间+物理两个维度），自动生成运动特异性的教练指导文本，在花样滑冰和拳击上分别超过 GPT-4o 31.6% 和 58.3%（G-Eval）。

**[CoAM: Corpus of All-Type Multiword Expressions](coam_corpus_of_all-type_multiword_expressions.md)**

:   构建了一个高质量的全类型多词表达(MWE)识别数据集 CoAM（1.3K句），通过多步质量保障流程解决了现有数据集标注不一致的问题，并发现微调大语言模型在 MWE 识别任务上显著优于此前的 SOTA 方法 MWEasWSD。

**[Code-Switching and Syntax: A Large-Scale Experiment](code-switching_and_syntax_a_large-scale_experiment.md)**

:   本文通过大规模、多语言、跨现象的实验，首次系统性验证了"句法信息足以解释语码转换（code-switching）模式"这一语言学共识，仅用句法特征就能达到与双语人类相当的判断准确率，且学到的句法模式可泛化到未见过的语言对。

**[CoLA: Collaborative Low-Rank Adaptation](cola_collaborative_low-rank_adaptation.md)**

:   提出 CoLA，一种灵活的 LoRA 架构，打破矩阵 A 和 B 之间的固定数量约束（#A=M, #B=N），并设计三种协作策略（全协作/随机协作/启发式协作），结合扩展的 PiSSA 初始化，在低样本场景下显著优于现有 PEFT 方法。

**[ComfyUI-Copilot: An Intelligent Assistant for Automated Workflow Development](comfyui-copilot_an_intelligent_assistant_for_automated_workflow_development.md)**

:   提出 ComfyUI-Copilot，一个基于 LLM 的层次化多 agent 框架，作为 ComfyUI 插件提供智能节点/模型推荐和一键式工作流构建，覆盖 7K 节点、62K 模型和 9K 工作流的知识库，在线服务 22 国的 19K 用户并处理了 85K+ 查询。

**[Commonsense Reasoning in Arab Culture](commonsense_arab_culture.md)**

:   提出 ArabCulture 数据集（3482 个 MSA 问题，覆盖 13 个阿拉伯国家/4 个区域/54 个文化子领域），系统评估多个 LLM 的阿拉伯文化常识推理能力，发现即使 GPT-4o 也仅达 90%、大部分模型在 40-80% 之间，揭示了 LLM 在非西方文化理解上的显著不足。

**[Completing A Systematic Review in Hours instead of Months with Interactive AI Agents](completing_a_systematic_review_in_hours.md)**

:   提出 InsightAgent，一个以人为中心的交互式多 Agent 系统，通过语义聚类分区、多 agent 并行阅读和实时用户交互，将医学系统综述的撰写时间从数月缩短到约 1.5 小时，达到人类撰写质量的 79.7%。

**[ConceptCarve: Dynamic Realization of Evidence](conceptcarve_dynamic_realization_of_evidence.md)**

:   提出 ConceptCarve 框架，通过 LLM 与传统检索器的交互式协作，动态构建概念树来表征证据在特定社区中的实现形式，解决了证据检索中的推理鸿沟和领域敏感性两大挑战。

**[ConECT Dataset: Overcoming Data Scarcity in Context-Aware E-Commerce MT](conect_dataset_overcoming_data_scarcity_in_context-aware_e-commerce_mt.md)**

:   构建了 ConECT——首个捷克-波兰电商多模态翻译数据集（11,400 句对 + 产品图片 + 类目路径），通过 VLM 端到端翻译、NMT+类目路径前缀、NMT+图像描述前缀三条技术路线的系统对比，发现结构化类目上下文能稳定提升翻译质量（COMET +0.005），而合成图像描述以级联方式注入反而严重损害翻译性能（COMET 暴跌 0.11+）。

**[CONFETTI: Conversational Function-Calling Evaluation Through Turn-Level Interactions](confetti_conversational_function-calling_evaluation_through_turn-level_interacti.md)**

:   CONFETTI 提出了一个面向多轮对话场景的函数调用（function-calling）评测基准，包含 109 段人类模拟对话、313 个用户轮次和 86 个 API，通过 off-policy turn-level 评估和 dialog act 标注系统性地测试 LLM 在复杂对话场景下的工具调用能力，发现即便最强模型（Nova Pro）也仅达 40% 左右，链式调用更是普遍短板。

**[ConSim: Measuring Concept-Based Explanations' Effectiveness with Automated Simulatability](consim_measuring_concept-based_explanations_effectiveness_with_automated_simulat.md)**

:   ConSim 提出用 LLM 作为"模拟器"来自动评估概念解释（concept-based explanation）的有效性——通过检测 LLM 能否仅凭解释来预测被解释模型的输出，从而同时衡量概念空间的质量和解释的可理解性，实现了可扩展、一致且全面的解释方法评估。

**[Consistent Client Simulation for Motivational Interviewing-based Counseling](consistent_client_simulation_for_motivational_interviewing-based_counseling.md)**

:   提出一种面向动机性访谈（MI）心理咨询的一致性客户模拟框架，通过状态转换、行动选择、信息选择和回复生成四个模块，确保模拟客户的行为与其预设的画像（动机、信念、改变计划、配合度）保持一致，在自动和专家评估中均优于基线方法。

**[Consultant Decoding: Yet Another Synergistic Mechanism](consultant_decoding_yet_another_synergistic_mechanism.md)**

:   提出 Consultant Decoding (CD)，一种基于目标模型负对数似然（NLL）验证 draft token 的新型协同解码机制，相比传统 Speculative Decoding 的似然比验证方法，能大幅提升接受率、降低大模型调用频率，同时保持甚至超越目标模型的生成质量。

**[Contextual Experience Replay for Self-Improvement of Language Agents](contextual_experience_replay_for_self-improvement_of_language_agents.md)**

:   CER（Contextual Experience Replay）提出了一个无需训练的语言 agent 自我改进框架，通过将过去的交互经验累积并合成为动态记忆缓冲区，在推理时让 agent 检索相关知识来增强新任务的决策，在 WebArena 上相对 GPT-4o baseline 提升了 51.0% 的成功率。

**[Cooperating and Competing Through Natural Language](cooperating_and_competing_through_natural_language.md)**

:   本文研究 LLM 智能体在自然语言交互环境中的合作与竞争行为，通过设计多人博弈场景，分析语言策略（如说服、欺骗、协商）如何影响博弈结果，揭示了 LLM 在社会交互中的策略涌现能力和局限性。

**[CORAL: Learning Consistent Representations across Multi-step Training with Lighter Speculative Drafter](coral_speculative_drafting.md)**

:   CORAL 通过跨步表示对齐（CSRA）改进多步训练中 draft 模型的特征一致性，并用权重分组机制压缩大词表 LM head 的推理延迟，在 LLaMA3/Qwen2.5 上实现 2.50-4.07× 加速，超越 EAGLE-2 和 HASS。

**[Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the Limits of Embedding Space Capacity](cramming_tokens_embedding_capacity.md)**

:   通过逐样本优化方法将文本压缩到可训练的 [mem] 向量中，发现 Llama-3.1-8B 可以将 1568 个 token 无损压缩到单个输入向量中，揭示了现有方法（约 x10 压缩比）与实际可达极限（x1500+）之间存在两个数量级的差距。

**[DAPE V2: Process Attention Score as Feature Map for Length Extrapolation](dape_v2_process_attention_score_as_feature_map_for_length_extrapolation.md)**

:   本文将 Transformer 的注意力分数视为特征图，通过在注意力分数上施加卷积操作（而非简单的 key-query 点积）来显著提升 Transformer 在长序列上的长度外推能力，将外推问题转化为经典的图像特征处理问题。

**[Decoding Knowledge Attribution in Mixture-of-Experts: A Framework of Basic-Refinement Collaboration and Efficiency Analysis](decoding_knowledge_attribution_in_mixture-of-experts_a_framework_of_basic-refine.md)**

:   提出跨层级知识归因算法，系统解析 MoE 模型中共享专家与路由专家的"基础-精炼"协作框架，揭示 MoE 相比稠密模型实现 31% 更高的逐层效率，并通过语义驱动路由机制（注意力头-专家相关性 r=0.68）和专家阻断实验验证了架构深度对鲁棒性的决定性影响。

**[Decoding Reading Goals from Eye Movements](decoding_reading_goals_from_eye_movements.md)**

:   本文首次提出从眼动轨迹中解码读者阅读目标（信息检索 vs. 普通阅读）的任务，通过 12 种模型的系统比较发现基于 Transformer 的扫视路径+语言建模方案（RoBERTa-Eye-F）最优，可在阅读早期即实现高精度实时预测。

**[DeepRTL2: A Versatile Model for RTL-Related Tasks](deeprtl2_a_versatile_model_for_rtl-related_tasks.md)**

:   DeepRTL2是首个统一处理RTL（寄存器传输级）相关生成任务与嵌入任务的LLM，通过精心构建的数据集和GRIT训练策略，在代码生成、代码理解、自然语言代码搜索、功能等价检查和性能预测五大任务上达到SOTA。

**[Detecting Sockpuppetry on Wikipedia Using Meta-Learning](detecting_sockpuppetry_on_wikipedia_using_meta-learning.md)**

:   本文将元学习（meta-learning）应用于维基百科的恶意马甲账号（sockpuppet）检测任务，通过跨多个马甲组的训练优化模型的快速适应能力，在数据稀缺场景下显著提升了检测精度，并发布了新的马甲调查数据集。

**[Developmentally-plausible Working Memory Shapes a Critical Period for Language Acquisition](developmentally-plausible_working_memory_shapes_a_critical_period_for_language_a.md)**

:   受"Less-is-More"假说启发，本文提出 DynamicLimit-Exp 方法，将人类工作记忆在关键期内的指数增长特征集成到语言模型训练中（通过动态调节 ALiBi 斜率），在 Child-Directed Speech 数据上训练的 GPT-2 模型在句法评估中显著优于无记忆约束和静态约束的基线。

**[Digital Gatekeepers: Google's Role in Curating Hashtags and Subreddits](digital_gatekeepers_googles_role_in_curating_hashtags_and_subreddits.md)**

:   本文通过对比 Google 搜索引擎返回的 hashtag 和 subreddit 结果与 Reddit 和 Twitter/X 的非采样数据，揭示了 Google 的算法系统性地压制了与色情、阴谋论、广告和加密货币相关的内容，同时推广高参与度内容，从而充当"数字守门人"影响公众话语。

**[Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction](distractor_gen_multiple_choice.md)**

:   本文提出了一个通过成对排序器预测学生选择倾向、再利用DPO训练干扰项生成器的三步流水线，使生成的多选题干扰项更具有迷惑性和区分度。

**[Do not Abstain! Identify and Solve the Uncertainty](do_not_abstain_identify_and_solve_the_uncertainty.md)**

:   本文提出ConfuseBench基准和基于inquiry answer唯一性判断不确定性来源的方法，并通过InteractDPO在策略训练中动态生成偏好对来提升inquiry质量，使LLM能主动识别并解决不确定性而非简单回避。

**[Dolphin: Moving Towards Closed-loop Auto-research through Thinking, Practice, and Feedback](dolphin_moving_towards_closed-loop_auto-research_through_thinking_practice_and_f.md)**

:   > 提出 Dolphin，一个闭环自动科研框架，包含"想法生成→实验验证→结果反馈"三阶段循环，通过任务属性引导的论文排序和异常回溯引导的调试流程，在 3D 分类等任务上自动提出并验证了接近人类设计 SOTA 的方法。

**[DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning](domix_an_efficient_framework_for_exploiting.md)**

:   提出 DoMIX，将各领域知识用独立 LoRA 模块存储后通过对角初始化的 bridge 矩阵在微调时灵活组合利用，在持续领域适应预训练场景下减少 58% 预训练时间和 87% GPU 内存，同时性能超越 SOTA。

**[Principled Content Selection to Generate Diverse and Personalized Multi-Document Summaries](dpp_diverse_multidoc_summary.md)**

:   提出将多文档摘要解耦为关键点抽取→DPP多样性选择→重写三步流水线，通过行列式点过程（DPP）进行原则性内容选择，显著提升LLM多文档摘要的源文档覆盖率。

**[DREsS: Dataset for Rubric-based Essay Scoring on EFL Writing](dress_dataset_rubric_based_essay_scoring_efl_writing.md)**

:   发布DREsS大规模标准化评分准则数据集，包含三个子集（DREsS_New真实课堂数据1.7K + DREsS_Std标准化历史数据集6.5K + DREsS_CASE增强数据40.1K），提出基于腐蚀的作文增强策略CASE，将BERT基线的QWK分数从0.471提升至0.685（提升45.44%）。

**[DRS: Deep Question Reformulation With Structured Output](drs_deep_question_reformulation_with_structured_output.md)**

:   提出 DRS（Deep Question Reformulation with Structured Output），一种零样本方法，通过实体驱动的 DFS 搜索 + 结构化输出约束，将 GPT-3.5 的问题重构准确率从 23.03% 提升至 70.42%，使 LLM 能有效帮助用户将不可回答的问题转化为可回答的形式。

**[Divide-Then-Aggregate: An Efficient Tool Learning Method via Parallel Tool Invocation](dta_llama_parallel_tool_invocation.md)**

:   提出 DTA-Llama，将传统树搜索的串行工具调用路径转换为有向无环图（DAG）结构实现并行调用，设计 Process/Thread 推理框架使 LLM 在每轮中可分解任务并并行执行多个工具，在 StableToolBench 上使 Llama2-7B 达到 GPT-3.5 Parallel Function Calling 的水平。

**[Efficient OpAmp Adaptation for Zoom Attention to Golden Contexts](efficient_opamp_adaptation_for_zoom_attention_to_golden_contexts.md)**

:   受运算放大器（OpAmp）电路启发，提出 OpAmp Adaptation 方法通过 adapter 高效改造预训练 Transformer 的注意力机制，在噪声上下文场景下让 LLM 更精准聚焦于 golden document，Qwen2.5-OpAmp-72B 在多个噪声上下文基准上超越 DeepSeek-V3 和 GPT-4o。

**[Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction](enhancing_conversational_agents_with_theory_of_mind_aligning_beliefs_desires_and.md)**

:   本文探索了从开源 LLM（LLaMA）内部表征中提取心智理论（ToM）相关信息的可行性，并利用 BDI（信念-愿望-意图）框架操纵这些表征来生成更符合人类社交认知的对话回复，ToM 对齐后的模型在 3B 和 8B 上分别达到 67% 和 63% 的胜率。

**[Enhancing Transformers for Generalizable First-Order Logical Entailment](enhancing_fol_entailment.md)**

:   系统性研究 Transformer 在一阶逻辑蕴涵任务中的泛化推理能力，揭示了查询语法、token 嵌入和 Transformer 架构（特别是位置编码）的影响，并提出 TEGA（Transformer Encoder with Guided Attention）在相对位置编码设定下显著提升逻辑推理性能。

**[Enhancing Marker Scoring Accuracy through Ordinal Confidence Modelling in Educational Assessments](enhancing_marker_scoring_accuracy_through_ordinal_confidence_modelling_in_educat.md)**

:   本文提出了一种基于核加权序数分类交叉熵（KWOCCE）的置信度建模方法，通过利用 CEFR 等级的序数结构和分数分箱策略，实现最高 47% 评分在 100% CEFR 一致性下释放，99% 在 ≥95% 一致性下释放，显著优于无置信度过滤时的约 92%。

**[Enhancing the Comprehensibility of Text Explanations via Unsupervised Concept Discovery](enhancing_the_comprehensibility_of_text_explanations_via_unsupervised_concept_di.md)**

:   提出 ECO-Concept 框架，通过 slot attention 机制自动提取文本概念，并利用 LLM 作为人类代理评估概念的可理解性，用可理解性反馈损失指导模型微调，在无概念标注的情况下实现了兼具高分类精度和人类可理解性的概念解释。

**[Entailed Between the Lines: Incorporating Implication into NLI](entailed_between_the_lines_incorporating_implication_into_nli.md)**

:   形式化定义"隐含蕴涵"（implied entailment）任务，将传统NLI的三分类扩展为四分类（隐式蕴涵/显式蕴涵/中立/矛盾），构建包含10K前提和40K假设的INLI数据集，实验表明微调后的模型能有效识别隐含蕴涵并跨领域泛化。

**[Entity Framing and Role Portrayal in the News](entity_framing_and_role_portrayal_in_the_news.md)**

:   本文构建了一个包含 5 种语言、1378 篇新闻文章、5800+ 实体标注的多语言层次化实体框架语料库，提出含 22 种精细角色的叙事角色分类体系（主角 / 反派 / 无辜者三大框架下），并在微调多语言 Transformer 和 LLM 层次零样本学习上建立了基准。

**[Entropy-UID: A Method for Optimizing Information Density](entropy-uid_a_method_for_optimizing_information_density.md)**

:   提出 Entropy-UID 方法，在自回归语言模型的解码过程中联合最小化熵和 surprisal 的加权组合，以实现信息密度的均匀分布。在 WikiText-2、OpenWebText 和 WMT 数据集上，该方法实现了最低的熵标准差（≈2.8）和稳定的 surprisal（≈5.7），优于单目标优化策略。

**[EpiCoDe: Boosting Model Performance Beyond Training with Extrapolation and Contrastive Decoding](epicode_boosting_model_performance_beyond_training_with_extrapolation_and_contra.md)**

:   提出 EpiCoDe，一种结合模型外推（Model Extrapolation）和对比解码（Contrastive Decoding）的无训练方法，在数据稀缺场景中通过参数空间外推和推理时logit差异对比来提升微调模型性能，并从logit误差角度给出了理论分析框架。

**[EpMAN: Episodic Memory AttentioN for Generalizing to Longer Contexts](epman_episodic_memory_attention_for_generalizing_to_longer_contexts.md)**

:   提出 EpMAN 方法，通过情景记忆模块估计上下文块的相对相关性，用该相关性重新加权解码器的自注意力（differentiating attention），配合噪声训练和注意力范围扩展策略，在 16k-256k 上下文长度范围内实现了比长上下文 LLM 和 RAG 更强且更鲁棒的表现。

**[ERU-KG: Efficient Reference-aligned Unsupervised Keyphrase Generation](eru-kg_efficient_reference-aligned_unsupervised_keyphrase_generation.md)**

:   ERU-KG 提出了一个由信息量模块和短语性模块组成的无监督关键短语生成框架，通过参考文本（查询、引用上下文、标题）在词项级别学习信息量估计，在关键短语生成基准上超越所有无监督基线并达到有监督模型 89% 的性能，同时推理速度最快。

**[Evaluating Design Decisions for Dual Encoder-based Entity Disambiguation](evaluating_design_decisions_for_dual_encoder-based_entity_disambiguation.md)**

:   系统评估了 Dual Encoder 在实体消歧（ED）任务中的关键设计选择（损失函数、相似度度量、标签语义化格式、负采样策略），并基于最优设计构建了 VerbalizED 系统，在 ZELDA 基准上达到了新的 SOTA，同时探索了一种迭代预测策略来利用已消歧的邻居实体改进困难样本。

**[Evaluating the Evaluation of Diversity in Commonsense Generation](evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)**

:   对常识生成（GCR）任务中的12种多样性评估指标进行系统元评估，发现基于形式（n-gram）的指标在低质量生成上严重高估多样性，而基于内容（句子嵌入）的指标与人类判断一致性更高，推荐未来 GCR 研究使用 VS-Embed 或 Chamfer Distance 等内容级指标。

**[Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection](explaining_matters_leveraging_definitions_and_semantic_expansion_for_sexism_dete.md)**

:   针对在线性别歧视检测中的数据稀疏和细粒度分类歧义问题，提出两种基于prompt的数据增强技术——定义驱动数据增强（DDA）利用类别定义生成语义对齐的合成样本，上下文语义扩展（CSE）通过分析模型错误的语义特征丰富训练数据——并结合 Mistral-7B 回退集成策略，在 EDOS 数据集上实现全任务 SOTA。

**[Explicit and Implicit Data Augmentation for Social Event Detection](explicit_and_implicit_data_augmentation_for_social_event_detection.md)**

:   本文提出SED-Aug，一个结合显式（LLM文本增强）和隐式（特征空间扰动）的双重数据增强框架用于社交事件检测，在Twitter2012和Twitter2018上分别超越最优基线17.67%和15.57%的平均F1。

**[Model Extrapolation Expedites Alignment](expo_model_extrapolation.md)**

:   基于"对齐训练仅产生微小参数变化"的观察，提出ExPO方法——通过放大SFT→DPO的参数变化方向（$\theta_2 = \theta_1 + \alpha\Delta\theta$），在零额外训练开销下提升对齐性能，使仅训练20%步骤的DPO模型超越完整训练的版本。

**[FastDraft: How to Train Your Draft](fastdraft_how_to_train_your_draft.md)**

:   提出 FastDraft，一套高效的 draft 模型预训练与对齐流程，可在24小时内用单节点8卡训练出约50M参数的 draft 模型，配合 Speculative Decoding 实现最高3倍内存带宽加速和2倍实际推理加速。

**[FastMCTS: A Simple Sampling Strategy for Data Synthesis](fastmcts_a_simple_sampling_strategy_for_data_synthesis.md)**

:   FastMCTS 提出了一种受 MCTS 启发的轻量级推理数据合成策略，通过自适应停留策略、动态探索和保留模拟三个改进，比拒绝采样多生成 30% 以上的正确推理路径，训练出的模型在多个数学基准上平均提升 3.9%。

**[FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning](fcmr_robust_evaluation_of_financial_cross-modal_multi-hop_reasoning.md)**

:   构建了金融领域跨模态多跳推理基准 FCMR，包含文本、表格和图表三种模态，分 Easy/Medium/Hard 三个难度等级，最强模型 Claude 3.5 Sonnet 在 Hard 级别仅达 30.4% 准确率，揭示了 MLLM 在信息检索阶段的关键瓶颈。

**[FEAT: A Preference Feedback Dataset through a Cost-Effective Auto-Generation and Labeling Framework for English AI Tutoring](feat_a_preference_feedback_dataset_through_a_cost-effective_auto-generation_and_.md)**

:   提出 FEAT 框架，通过 LLM 自动生成和标注教师反馈偏好数据集用于英语辅导系统，发现仅混入 5-10% 人工标注数据就能超越 100% 人工数据的排序性能。

**[Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients](federated_lora_heterogeneous.md)**

:   提出 LoRA-A2 框架，通过交替冻结 LoRA 的 A/B 模块与自适应秩选择策略，同时解决联邦学习中 LoRA 聚合不一致和通信开销大的双重难题。

**[Behind Closed Words: Creating and Investigating the forePLay Annotated Dataset for Polish Erotic Discourse](foreplay_polish_erotic_detection.md)**

:   构建了首个波兰语色情内容检测数据集 forePLay（24,768 句，5 类标签），提出涵盖模糊性、暴力和社会不可接受行为的多维标注体系，评估发现专用波兰语模型显著优于多语言模型，且 Transformer 编码器模型在不平衡类别处理上表现最强。

**[FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](fractal_fine-grained_scoring_from_aggregate_text_labels.md)**

:   提出 FRACTAL 方法，将回复级别（response-level）的聚合标签分解为句子级别（sentence-level）的伪标签，利用多实例学习（MIL）和标签比例学习（LLP）技术结合先验信息（文档-句子余弦相似度）训练句子级评分模型，覆盖检索、问答、摘要和数学推理四类任务。

**[Frictional Agent Alignment Framework: Slow Down and Don't Break Things](frictional_agent_alignment.md)**

:   提出摩擦对齐框架 FAAF（Frictional Agent Alignment Framework），通过双策略（frictive state policy + intervention policy）目标函数，训练 LLM 在协作对话中识别信念冲突并生成促进反思与审议的"摩擦"干预，超越 DPO/IPO/PPO 等对齐方法。

**[From Lists to Emojis: How Format Bias Affects Model Alignment](from_lists_to_emojis_how_format_bias_affects_model_alignment.md)**

:   本文系统研究了 RLHF 中偏好模型（包括人类评估者、GPT-4 和开源模型）对粗体、列表、emoji 等格式模式的偏好偏差，展示了不到 1% 的偏差数据即可显著注入偏差，并提出了双头奖励模型的去偏方法。

**[From Real to Synthetic: Synthesizing Millions of Diversified and Complicated User Instructions with Attributed Grounding](from_real_to_synthetic_synthesizing_millions_of_diversified_and_complicated_user.md)**

:   本文提出"归因式接地"（Attributed Grounding）框架，通过自顶向下的用户归因和自底向上的基于网络文档的指令合成，构建了一个包含 100 万条多样且复杂指令的 SynthQuestions 数据集，训练模型在多个通用基准上达到领先表现。

**[GA-S3: Comprehensive Social Network Simulation with Group Agents](ga-s3_comprehensive_social_network_simulation_with_group_agents.md)**

:   提出基于"群体智能体"（Group Agent）的社交网络模拟系统 GA-S3，将具有相似行为的个体聚合为群体代理，通过层次化生成、马尔可夫网络推理和行为模块实现大规模社交网络的高效精确模拟。

**[GeAR: Generation Augmented Retrieval](gear_generation_augmented_retrieval.md)**

:   GeAR 在传统 bi-encoder 检索框架上引入融合编码器和文本解码器，通过生成任务增强检索模型对文档内部细粒度语义的理解能力，同时不增加全局检索的计算开销。

**[Generating Synthetic Relational Tabular Data via Structural Causal Models](generating_synthetic_relational_tabular_data_via_structural_causal_models.md)**

:   本文扩展了 TabPFN 的基于结构因果模型（SCM）的合成数据生成方法，提出了一个能够生成多表关联（relational）合成表格数据的框架，通过耦合节点和隐因果关系实现跨表依赖建模。

**[GeNRe: A French Gender-Neutral Rewriting System Using Collective Nouns](genre_a_french_gender-neutral_rewriting_system_using_collective_nouns.md)**

:   GeNRe 是首个法语性别中性重写系统，利用集体名词（collective nouns）替代阳性泛指（masculine generics），提出规则系统、微调模型和指令模型三种方案，其中规则系统和 Claude 3 Opus + 词典方案效果最好。

**[GPT-4 as a Homework Tutor can Improve Student Engagement and Learning Outcomes](gpt-4_as_a_homework_tutor_can_improve_student_engagement_and_learning_outcomes.md)**

:   在意大利高中进行了为期 8 周的随机对照试验（RCT），用 GPT-4 替代传统英语作业作为互动辅导工具，发现 GPT-4 组学生在参与度（有趣性、资源充分性显著提升）和特定条件下的学习增益（三年级 Cohen's d=0.603）方面有所改善，仅需教师提供作业目标和描述即可实施，幻觉率低于 1%，且所有在校学生均表示希望继续使用。

**[Graph-guided Cross-composition Feature Disentanglement for Compositional Zero-shot Learning](graph-guided_cross-composition_feature_disentanglement_for_compositional_zero-sh.md)**

:   DCDA 提出图引导的跨组合特征解耦方案，通过双适配器（L-Adapter 用于文本端 GNN 特征聚合、V-Adapter 用于视觉端跨注意力解耦）注入冻结 CLIP，在组合零样本学习任务上显著超越现有方法。

**[Graph-Structured Trajectory Extraction from Travelogues](graph-structured_trajectory_extraction_from_travelogues.md)**

:   提出"访问顺序图"（Visiting Order Graph）来统一表示旅行轨迹中的地理包含层级关系和时序转移关系，构建了覆盖 100 篇日语游记的 ATD-VSO 基准数据集（3354 个地理实体、3369 条关系），并通过基线实验发现地理包含关系预测（F1=0.355）是核心瓶颈，为该领域指明了地理知识融合的关键方向。

**[Graphically Speaking: Unmasking Abuse in Social Media with Conversation Insights](graphically_speaking_unmasking_abuse_in_social_media_with_conversation_insights.md)**

:   提出一种基于图注意力网络（GAT）的上下文感知滥用语言检测框架，将 Reddit 对话建模为图结构（节点=评论，边=回复关系），利用基于 Reddit 界面渲染逻辑的 affordance-based 图裁剪策略保留关键上下文，3 层 GAT 模型达到 F1=0.7624，显著优于无上下文基线和扁平化上下文方法，在上下文敏感样本上提升尤为明显（+4.75%）。

**[Guidelines for Fine-grained Sentence-level Arabic Readability Annotation](guidelines_for_fine-grained_sentence-level_arabic_readability_annotation.md)**

:   本文提出了 BAREC 语料库及其标注指南，这是一个拥有 69K+ 句子、覆盖 19 个可读性等级的大规模阿拉伯语句子级可读性评估资源，并在此基础上建立了自动可读性评估的基准模型。

**[Hanging in the Balance: Pivotal Moments in Crisis Counseling Conversations](hanging_in_the_balance_pivotal_moments_in_crisis_counseling_conversations.md)**

:   本文提出了一种无监督方法来检测对话中的"关键时刻"（pivotal moments）——即下一步回应可能极大影响对话结局的节点，并在危机心理咨询场景中验证了该方法的有效性。

**[Hard Negative Mining for Domain-Specific Retrieval in Enterprise Systems](hard_negative_mining_for_domain-specific_retrieval_in_enterprise_systems.md)**

:   本文提出了一种面向企业级领域特定检索的可扩展硬负样本挖掘框架，通过融合多种嵌入模型、PCA 降维和双语义条件筛选来动态选择高质量硬负样本，在内部云服务数据集和公开基准上均取得了显著提升。

**[HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference](hata_trainable_and_hardware-efficient_hash-aware_top-k_attention_for_scalable_la.md)**

:   HATA 提出了一种将 learning-to-hash 技术集成到 top-k 注意力机制的方法，通过将查询和键映射为二进制哈希码来获取相对 qk 分数排序（而非绝对分数估计），在保持模型精度的同时实现了相对全注意力最高 7.2 倍的加速。

**[HelpSteer3: Human-Annotated Feedback and Edit Data to Empower Inference-Time Scaling](helpsteer3_human-annotated_feedback_and_edit_data_to_empower_inference-time_scal.md)**

:   NVIDIA 发布 HelpSteer3 数据集（7000+标注员、80+国家），训练专用的 Feedback 和 Edit 模型，在推理时通过"初始响应→反馈→编辑"循环实现开放域通用任务的推理时扩展，基于 Llama 3 系列 70B 模型在 Arena Hard 上达到 92.7 分，超越 OpenAI o1-preview (90.4) 和 DeepSeek R1 (92.3)。

**[Hierarchical Attention Generates Better Proofs](hierarchical_attention_generates_better_proofs.md)**

:   提出 Hierarchical Attention 正则化方法，通过建立五层语义层次结构来引导 LLM 的注意力机制，使其与数学推理的自然信息流对齐，在 miniF2F 和 ProofNet 上分别提升证明成功率 2.05% 和 1.69%，同时降低证明复杂度 23.81% 和 16.50%。

**[Hierarchical Bracketing Encodings for Dependency Parsing as Tagging](hierarchical_bracketing_dep_parsing.md)**

:   提出层次化括号编码家族用于依存句法分析的序列标注范式，证明现有4-bit编码是该家族的非最优特例，推导出仅需12个标签的最优编码，并将其推广到处理任意非投射性。

**[Hierarchical Memory Organization for Wikipedia Generation](hierarchical_memory_wikipedia_gen.md)**

:   提出 Memory Organization-based Generation（MOG）框架，从网页文档中提取细粒度记忆单元（factoid），通过递归聚类-摘要算法组织为层次化 Wikipedia 大纲结构，使每个章节都有直接的记忆支撑，在 FreshWiki 和 WikiStart 数据集上信息量、引用率和可验证性全面超越 RAG 和 STORM 基线。

**[Counterspeech the Ultimate Shield! Multi-Conditioned Counterspeech Generation through Attributed Prefix Learning](hippro_counterspeech_gen.md)**

:   提出 HiPPrO 两阶段框架用于多条件反仇恨言论生成——第一阶段通过层次化前缀学习在多个属性（策略+情感）空间中优化反言论生成，第二阶段用无参考无奖励的偏好优化提升建设性，策略一致性提升 ~38%，ROUGE 指标提升 2-3%。

**[How to Mitigate Overfitting in Weak-to-Strong Generalization?](how_to_mitigate_overfitting_in_weak-to-strong_generalization.md)**

:   提出两阶段训练框架解决弱到强泛化中的过拟合问题：第一阶段通过基于不确定性的过滤提高弱监督信号质量，第二阶段用已微调的强模型为被丢弃的难题重新生成答案以恢复问题质量，在 GSM8k 和 MATH 上将 PGR 从 7.19% 提升到 120.50%。

**[Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback](hybrid_preferences_learning_to_route_instances_for_human_vs_ai_feedback.md)**

:   本文提出HyPER（混合偏好路由器），通过训练性能预测模型来智能决定每个标注实例应由人类还是AI完成偏好标注，在RewardBench上比纯人类或纯AI标注分别提升7-13%，同时大幅降低标注成本。

**[Enhancing Hyperbole and Metaphor Detection with Their Bidirectional Dynamic Interaction and Emotion Knowledge](hyperbole_metaphor_detection.md)**

:   提出 EmoBi 框架，通过情感分析→情感引导的域映射→双向动态交互三阶段 prompting 流程，利用 LLM 挖掘夸张和隐喻背后的情感线索及二者的互促关系，在四个数据集上大幅超越 SoTA（TroFi 上夸张检测 F1 提升 28.1%，HYPO-L 上隐喻检测 F1 提升 23.1%）。

**[I0T: Embedding Standardization Method Towards Zero Modality Gap](i0t_embedding_standardization_method_towards_zero_modality_gap.md)**

:   提出 I0T 框架，通过发现并消除 CLIP 中图像/文本编码器各自学到的模态特异性特征（表现为归一化嵌入中的峰值激活），将模态差距降低至接近零，同时保持甚至提升下游任务性能，并提出了比 CLIPScore 更具可解释性的自动评估指标 I0T-Score。

**[If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?](if_attention_serves_as_a_cognitive_model_of_human_memory_retrieval_what_is_the_p.md)**

:   本文探究 Transformer Grammar（TG）的注意力机制能否作为人类记忆检索的认知模型，通过 Normalized Attention Entropy（NAE）将模型与人类阅读时间关联，发现基于句法结构的注意力比基于 token 的注意力更能解释人类句子处理行为，且两者提供独立互补的贡献。

**[Predicting Implicit Arguments in Procedural Video Instructions](implicit_arguments_video_instructions.md)**

:   提出 Implicit-VidSRL 数据集与 iSRL-Qwen2-VL 模型，针对过程性视频指令中省略的隐含论元（食材成分）进行预测，通过 SRL 框架将多步指令分解为 {verb, what, where/with} 三元组，在银标数据上微调后在隐含论元 F1 上超越 GPT-4o 达 17%。

**[Implicit Reasoning in Transformers is Reasoning through Shortcuts](implicit_reasoning_in_transformers_is_reasoning_through_shortcuts.md)**

:   本文通过在受控的多步数学推理数据集上从零训练 GPT-2，系统研究了语言模型的隐式推理机制，发现隐式推理本质上是基于模式匹配的捷径学习——在固定模式数据上能泛化，在非固定模式数据上则过拟合，这一发现在 SOTA 大模型上同样成立。

**[Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate](improve_rule_retrieval_and_reasoning_with_self-induction_and_relevance_reestimat.md)**

:   针对规则检索中查询（具体实例化事实）与规则（抽象变量形式）之间的语义鸿沟，提出 SIAR（自归纳增强检索）和 R3（规则相关性重评估）两种方法，通过将查询映射到规则语义空间并重新评估规则相关性，显著提升规则检索和下游推理性能。

**[Improving Language and Modality Transfer in Translation by Character-level Modeling](improving_language_and_modality_transfer_in.md)**

:   提出基于字符级编码器 charSONAR 的跨语言跨模态翻译方法，通过 teacher-student 训练获得字符级文本编码器，再用轻量适配器连接 1000+ 语言的 CTC ASR 模型（MMS），在 75 语言文本翻译和 33 语言语音翻译上实现 SOTA，零资源低资源场景表现尤其突出。

**[In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents](in_prospect_and_retrospect_reflective_memory_management_for_long-term_personaliz.md)**

:   本文提出 Reflective Memory Management (RMM) 机制，通过前瞻性反思（多粒度记忆摘要）和回顾性反思（强化学习驱动的在线检索优化）两个方向的结合，为长期个性化对话系统构建了高效的记忆管理框架，在 LongMemEval 上准确率提升超过 10%。

**[Inducing Lexicons of In-Group Language with Socio-Temporal Context](inducing_lexicons_of_in-group_language_with_socio-temporal_context.md)**

:   提出 LISTN（Lexicon Induction with Socio-Temporal Nuance）框架，利用动态词嵌入和用户嵌入联合建模社区语言的社会结构和时间演化，在反女性在线社区（manosphere）的群体内词汇归纳任务上达到 0.77 的平均精度，显著超越现有方法。

**[Inferring Functionality of Attention Heads from their Parameters](inferring_functionality_of_attention_heads_from_their_parameters.md)**

:   提出MAPS框架，通过将注意力头参数投影到词汇空间构建token映射矩阵$M$，无需任何推理或训练即可推断注意力头实现的功能，在6个LLM上验证了20种关系操作的映射准确性，并开发自动化pipeline发现了大量此前未被识别的注意力头功能。

**[Infogen: Generating Complex Statistical Infographics from Documents](infogen_generating_complex_statistical_infographics_from_documents.md)**

:   提出Infogen框架，将文本文档转化为复杂统计信息图（多子图组合），采用两阶段设计——先用微调LLM生成结构化中间元数据，再用LLM代码生成器和反馈模块迭代生成最终信息图代码。

**[INJONGO: A Multicultural Intent Detection and Slot-filling Dataset for 16 African Languages](injongo_a_multicultural_intent_detection_and_slot-filling_dataset_for_16_african.md)**

:   本文引入 Injongo——一个覆盖 16 种非洲语言的多文化意图检测与槽位填充基准数据集，由母语者在银行、旅行、家庭、餐饮等领域原创生成，实验揭示 LLM 在非洲语言上的槽位填充表现极差（GPT-4o 仅 26 F1），意图检测也明显落后于微调基线。

**[Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking](inner_thinking_transformer_leveraging_dynamic_depth_scaling_to_foster_adaptive_i.md)**

:   提出 Inner Thinking Transformer (ITT)，通过自适应 token 路由和残差思维连接，在不增加参数的情况下为关键 token 动态分配更多计算步骤，实现隐式深度推理，162M 参数即可达到 466M Transformer 96.5% 的性能。

**[InspireDebate: Multi-Dimensional Evaluation-Guided Reasoning for Debating](inspiredebate_multidim_evaluation_debating.md)**

:   提出双组件框架：InspireScore（融合4个主观维度+2个客观维度的辩论评估系统）和 InspireDebate（通过CoT-SFT + 多维DPO + Web-RAG 三阶段优化的辩论框架），评估系统与专家判断相关性提高 44%，辩论性能超越基线 57%。

**[Instruction-Tuning Data Synthesis from Scratch via Web Reconstruction](instruction-tuning_data_synthesis_from_scratch_via_web_reconstruction.md)**

:   提出 Web Reconstruction (WebR)，一种从原始网页文档全自动合成高质量指令微调数据的框架，通过"Web作为指令"和"Web作为回复"双视角范式，无需人工标注即可生成优于现有SOTA的IT数据。

**[Inter-Passage Verification for Multi-evidence Multi-answer QA](inter-passage_verification_for_multi-evidence_multi-answer_qa.md)**

:   本文提出 RI²VER 框架解决多答案问答（multi-answer QA）问题——先通过独立阅读大量检索段落生成高召回但含噪声的候选答案集，再通过段落间验证（生成验证问题 → 收集额外证据 → 跨段落综合验证）过滤错误答案，在 QAMPARI 和 RoMQA 上平均 F1 提升 11.17%。

**[Interlocking-free Selective Rationalization Through Genetic-based Learning](interlocking-free_selective_rationalization_through_genetic-based_learning.md)**

:   本文提出 GenSPP，首个完全消除 interlocking 问题的选择性合理化框架，通过遗传算法对生成器和预测器进行分离优化，在合成数据集和仇恨言论检测任务上显著提升了高亮质量（Hl-F1 提升 6.5%–10.3%），同时保持了可比的分类性能。

**[Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process](intuitive_fine_tuning.md)**

**[IRIS: Interactive Research Ideation System for Accelerating Scientific Discovery](iris_interactive_research_ideation_system_for_accelerating_scientific_discovery.md)**

:   提出 IRIS，一个开源的交互式研究构思系统，通过蒙特卡洛树搜索（MCTS）扩展测试时计算、细粒度反馈机制和基于查询的文献综合，实现人机协作的科学假设生成。

**[Is Linguistically-Motivated Data Augmentation Worth It?](is_linguistically-motivated_data_augmentation_worth_it.md)**

:   系统比较语言学驱动和非语言学（随机扰动）数据增强策略在两种低资源语言上的效果，发现语言学方法仅在生成的样本接近训练数据分布时才有优势，否则可能有害。

**[It's Not a Walk in the Park! Challenges of Idiom Translation in Speech-to-text Systems](its_not_a_walk_in_the_park_challenges_of_idiom_translation_in_speech-to-text_sys.md)**

:   本文首次系统比较了语音到文本翻译（SLT）、文本机器翻译（MT）和大语言模型（LLM）在习语翻译任务上的表现，发现 SLT 系统在处理习语时性能大幅下降，即便在编码器高层仍倾向于字面翻译，而 MT 和 LLM 对习语的处理能力明显更优。

**[Knowledge Tracing in Programming Education Integrating Students' Questions](knowledge_tracing_in_programming_education_integrating_students_questions.md)**

:   本文提出 SQKT（Students' Question-based Knowledge Tracing）模型，首次将学生提问和自动提取的技能信息整合到知识追踪中，用于预测编程教育中学生对后续编程题的完成情况，域内实验 AUC 提升高达 33.1%。

**[KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding](kodcode_a_diverse_challenging_and_verifiable_synthetic_dataset_for_coding.md)**

:   KodCode 提出一套三阶段合成数据管线（编程题目合成→解决方案+单元测试自验证→后训练数据合成），构建了 447K 经过验证的编程 question-solution-test 三元组，微调后的模型在 HumanEval、MBPP、BigCodeBench、LiveCodeBench 等基准上超越 Qwen2.5-Coder-32B-Instruct 和 DeepSeek-R1-Distill-Llama-70B。

**[LADDER: Language-Driven Slice Discovery and Error Rectification in Vision Classifiers](ladder_language-driven_slice_discovery_and_error_rectification_in_vision_classif.md)**

:   提出 LADDER 框架，利用 LLM 的推理能力和潜在领域知识，通过分析文本（图像描述/医学报告/元数据）自动发现视觉分类器中的系统性偏差切片（error slices），并通过伪标签生成和属性重平衡实现无需标注的多偏差缓解。

**[LAQuer: Localized Attribution Queries in Content-grounded Generation](laquer_localized_attribution.md)**

:   提出 Localized Attribution Queries (LAQuer) 任务——将生成文本中用户选定的片段精确定位到源文档的对应片段，实现比句子级归因更精细、比子句级归因更用户导向的溯源，在多文档摘要和长文本问答上显著减少了归因文本长度。

**[LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models](latim_measuring_latent_token-to-token_interactions_in_mamba_models.md)**

:   提出 LaTIM，一种针对 Mamba-1 和 Mamba-2 的 token 级分解方法，将 SSM 的隐式计算重构为类似 Transformer 注意力的 token-to-token 贡献矩阵，实现对 Mamba 模型的细粒度可解释性分析。

**[Learning to Align Multi-Faceted Evaluation: A Unified and Robust Framework](learning_to_align_multi-faceted_evaluation_a_unified_and_robust_framework.md)**

:   提出 ARJudge 评估框架，通过微调 Analyzer 自适应生成评估标准并执行文本+代码双驱动分析，配合无需微调的 Refiner 综合判断，在多个评估基准上超越现有微调评估器，尤其在指令遵循评估上通过代码驱动分析提升高达 11.1%。

**[Learning to Reason from Feedback at Test-Time](learning_to_reason_from_feedback_at_test-time.md)**

:   本文提出FTTT（Feedback at Test-Time Training）范式，将LLM在推理阶段对环境反馈的利用形式化为一个优化问题，并设计了可学习的测试时优化器OpTune，在四个推理数据集上实现了优于现有反馈利用方法的可扩展性和性能。

**[LegalReasoner: Step-wised Verification-Correction for Legal Judgment Reasoning](legalreasoner_step-wised_verification-correction_for_legal_judgment_reasoning.md)**

:   本文提出LegalReasoner框架，通过争议焦点识别、分步推理、过程验证器逐步校验以及专家设计的归因纠正策略来增强法律判决预测的可靠性，配合新发布的58130条香港法院案例LegalHK数据集，在LLAMA-3.1-70B上将与法院判决的一致率从72.37%提升至80.27%。

**[Length-Induced Embedding Collapse in PLM-based Models](length-induced_embedding_collapse_in_plm-based_models.md)**

:   发现并严格证明了 PLM 文本嵌入模型中的"长度坍缩"现象——长文本嵌入趋于聚集，源于 self-attention 作为低通滤波器随文本长度增加而滤波率增强，高频信息被过度抑制；提出 TempScale 方法通过降低 attention 温度来缓解长短文本嵌入分布差异，在 MTEB 上提升 0.94%、LongEmbed 上提升 1.10%。

**[Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation](leveraging_unit_language_guidance_to_advance_speech_modeling_in_textless_speech-.md)**

:   本文提出"单元语言"（unit language）的概念，通过n-gram语言建模将语音离散单元构造为类文本表示，并利用多任务学习引导无文本语音到语音翻译（S2ST）模型的训练过程，同时提出任务提示建模来缓解源端和目标端单元语言同时使用时的冲突，在VoxPopuli四语数据集上取得显著提升。

**[Limited Generalizability in Argument Mining: State-Of-The-Art Models Learn Datasets, Not Arguments](limited_generalizability_in_argument_mining_state-of-the-art_models_learn_datase.md)**

:   对 4 种 Transformer 模型在 17 个英语句子级论辩挖掘数据集上进行首次大规模跨数据集泛化评估，发现 SOTA 模型主要学到了数据集特有的词汇模式而非论辩的结构性信号，泛化能力远低于基准表现，但任务相关预训练和联合数据训练可部分缓解这一问题。

**[Literature Meets Data: A Synergistic Approach to Hypothesis Generation](literature_meets_data_hypothesis.md)**

:   提出首个将文献驱动和数据驱动假设生成进行协同整合的方法，通过 Refinement 和 Union 两种策略让 LLM 从论文摘要和观测数据中联合生成更具泛化性的假设，在五个社会科学分类任务的 OOD 数据集上比纯数据驱动方法平均提升 3.37%，并首次通过人类实验证明 LLM 生成的假设能显著改善人类决策准确率（+7.44% / +14.19%）。

**[LoGU: Long-form Generation with Uncertainty Expressions](logu_longform_gen_uncertainty.md)**

:   定义"长文本不确定性生成"（LoGU）任务，识别不确定性抑制和不确定性错位两个子挑战，提出基于分解的数据构造框架和 SFT+DPO 两阶段训练流水线，使 LLM 在长文本生成中对不确定事实显式表达不确定性，在三个数据集上将 Llama3-8B 的事实准确率从 51.9% 提升到 71.6%，错误声明数从 20.4 降到 5.81。

**[Low-Rank Interconnected Adaptation across Layers](low-rank_interconnected_adaptation_across_layers.md)**

:   提出 Lily（Low-rank Interconnected Adaptation across Layers），通过将 LoRA 的 A/B 适配器跨层解耦并互联共享，配合数据依赖的路由机制，在相同或更少参数下实现高秩权重更新，在多模态、多架构、多规模场景中均优于 LoRA。

**[MaCP: Minimal yet Mighty Adaptation via Hierarchical Cosine Projection](macp_minimal_yet_mighty_adaptation_via_hierarchical_cosine_projection.md)**

:   本文提出 MaCP——一种基于离散余弦变换（DCT）的参数高效微调方法，通过将权重变化投影到余弦频域并分层选择最关键的频率分量，在极低参数量（比 LoRA 少 99.7%）下实现了优于或媲美现有 PEFT 方法的性能。

**[Making FETCH! Happen: Finding Emergent Dog Whistles Through Common Habitats](making_fetch_happen_finding_emergent_dog_whistles_through_common_habitats.md)**

:   提出 FETCH! 基准和 EarShot 系统，用于在大规模社交媒体语料库中发现新兴的"狗哨"（dog whistle，即具有双重含义的编码表达），利用向量数据库和 LLM 的结合实现了比现有方法高 2-20 个 F-score 百分点的提升。

**[Map&Make: Schema Guided Text to Table Generation](mapmake_schema_guided_text_to_table_generation.md)**

:   提出 Map&Make 方法，先将非结构化文本"解剖"为命题原子语句（Map阶段），再从中推导表格模式并填充数据（Make阶段），在 Rotowire 和 Livesum 两个场景上显著提升了文本转表格的质量和可解释性。

**[Mapping the Podcast Ecosystem with the Structured Podcast Research Corpus](mapping_the_podcast_ecosystem_with_the_structured_podcast_research_corpus.md)**

:   构建并发布了 SPoRC——一个包含 110 万集播客转录的大规模数据集（含元数据、推断的说话者角色和 37 万集的音频特征），并通过话题分析、嘉宾共现网络分析和 George Floyd 事件响应性分析，首次全面刻画了播客生态系统的内容、结构和响应性。

**[MapQaTor: An Extensible Framework for Efficient Annotation of Map-Based QA Datasets](mapqator_an_extensible_framework_for_efficient_annotation_of_map-based_qa_datase.md)**

:   提出 MapQaTor——一个可扩展的开源 Web 框架，通过集成多种地图 API（Google Maps、OpenStreetMap 等），将地理空间 QA 数据集的标注速度提升至少 30 倍，同时通过 API 响应缓存确保数据可复现性。

**[MDCure: A Scalable Pipeline for Multi-Document Instruction-Following](mdcure_a_scalable_pipeline_for_multi-document_instruction-following.md)**

:   提出 MDCure 框架，通过两阶段流程（生成+过滤）自动构建高质量的多文档指令数据，并训练 MDCureRM 多目标奖励模型进行数据过滤，使微调后的 LLM（最高 70B）在多文档和长上下文任务上相比基线提升高达 75.1%，且实现跨任务、跨领域的强泛化能力。

**[Meaning Beyond Truth Conditions: Evaluating Discourse Level Understanding via Anaphora Accessibility](meaning_beyond_truth_conditions_evaluating_discourse_level_understanding_via_ana.md)**

:   本文提出语义理解能力的层级框架（词汇/句子/话语），构建了基于照应可及性（anaphora accessibility）的评估数据集，发现 LLM 在某些结构上与人类一致但在其他结构上存在系统性分歧——LLM 依赖词汇线索而非结构化抽象。

**[Measuring the Effect of Transcription Noise on Downstream Language Understanding Tasks](measuring_the_effect_of_transcription_noise_on_downstream_language_understanding.md)**

:   提出ENDow框架，首次系统化地分析ASR转录噪声对下游NLU任务的影响，通过可配置的pipeline评估不同噪声强度和类型下任务模型的行为，发现命名实体是最关键的词类型，且模型能容忍一定程度的噪声。

**[Memorization: A Close Look at Books](memorization_a_close_look_at_books.md)**

:   系统研究 Llama 3 系列模型对完整书籍的记忆化程度，发现书籍提取率与其流行度（训练数据重复度代理）高度正相关，并通过 LoRA 微调揭示指令微调的抗反刍缓解措施仅涉及极少量集中在底层 transformer block 的权重变化。

**[Meta-Learning Neural Mechanisms rather than Bayesian Priors](meta-learning_neural_mechanisms_rather_than_bayesian_priors.md)**

:   挑战了"元学习在神经网络中蒸馏贝叶斯简单性先验"的主流观点，通过形式语言实验证明元学习实际上是在模型中植入有用的**神经机制**（如计数器），而非学习简单性偏好。

**[Meta-Reflection: A Feedback-Free Reflection Learning Framework](meta-reflection_a_feedback-free_reflection_learning_framework.md)**

:   提出 Meta-Reflection 框架，通过可学习的元反思代码本（codebook）存储和检索反思性洞察，使 LLM 在推理时无需外部反馈和多轮迭代，仅需单次前向传播即可利用历史反思经验来改善输出质量，在编程、数学推理和电商意图检测任务上均取得显著提升。

**[MEXMA: Token-level Objectives Improve Sentence Representations](mexma_token-level_objectives_improve_sentence_representations.md)**

:   提出 MEXMA，一种结合句子级和 token 级目标的跨语言句子编码器训练方法：用一种语言的句子表示去预测另一种语言的被掩码 token，同时让句子和 token 的梯度都直接更新编码器，在双文本挖掘和多项下游任务上超越 SONAR 和 LaBSE。

**[Micro-Act: Mitigate Knowledge Conflict in QA via Actionable Self-Reasoning](micro_act_knowledge_conflict_reasoning.md)**

:   提出 Micro-Act 框架，通过层次化动作空间（导航/功能/桥接动作）和自适应粒度分解，让 LLM 自动感知上下文复杂度并逐层拆解知识对比，在 5 个知识冲突基准上全面超越 SOTA，同时在无冲突场景下也保持鲁棒。

**[MindRef: Mimicking Human Memory for Hierarchical Reference Retrieval with Fine-Grained Location Awareness](mindref_mimicking_human_memory_hierarchical_reference_retrieval.md)**

:   提出 MindRef 框架，模拟人类先回忆文档标题再定位具体段落的两阶段记忆模式，通过 Trie 和 FM-Index 约束解码让 LLM 独立召回参考段落，无需额外检索模型或预分段。

**[Minimal Pair-Based Evaluation of Code-Switching](minimal_pair-based_evaluation_of_code-switching.md)**

:   本文提出基于最小对（minimal pairs）的代码切换（code-switching, CS）评估方法，为11个语言对各收集最多1000个最小对，发现双语者和大规模LLM都倾向于自然出现的CS句子，且模型越大偏好越一致，封闭类词的操纵产生最大概率差异。

**[MIR: Methodology Inspiration Retrieval for Scientific Research Problems](mir_methodology_inspiration_retrieval_for_scientific_research_problems.md)**

:   本文定义了方法论启发检索（Methodology Inspiration Retrieval, MIR）新任务——为给定的科研问题检索能提供方法论启发的论文，构建了Methodology Adjacency Graph (MAG)来捕获方法论传承关系，在Recall@3上提升+5.4、mAP上提升+7.8，结合LLM重排序再获+4.5/+4.8的额外提升。

**[Mitigating Shortcut Learning with InterpoLated Learning](mitigating_shortcut_learning_with_interpolated_learning.md)**

:   提出 InterpoLated Learning (InterpoLL)，通过将多数样本的表示与同类少数样本的表示进行插值，削弱模型对虚假关联（shortcut）的依赖，显著提升少数样本上的泛化能力。

**[Mixture of Small and Large Models for Chinese Spelling Check](mixture_of_small_and_large_models_for_chinese_spelling_check.md)**

:   本文提出在Beam Search解码阶段动态混合小模型（fine-tuned BERT）和大语言模型（LLM）的概率分布来进行中文拼写纠错，无需微调LLM即可兼顾小模型的精确纠错和LLM的流畅性，在多个CSC数据集上达到SOTA。

**[MockConf: A Student Interpretation Dataset: Analysis, Word- and Span-level Alignment and Baselines](mockconf_a_student_interpretation_dataset_analysis_word-_and_span-level_alignmen.md)**

:   本文构建了 MockConf——一个以捷克语为中心的**学生同声传译数据集**（7 小时，5 种欧洲语言），提供人工标注的 span 级和 word 级对齐，同时发布了专用标注工具 InterAlign，并建立了自动对齐的基线和评估指标体系。

**[MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)**

:   提出 MoRE (Mixture of Low-Rank Experts)，将 LoRA 中的不同秩视为不同专家，通过自适应秩选择器为每个任务动态选择最合适的秩，配合对比学习优化的任务嵌入和平衡数据采样策略，使用单个 LoRA 模块实现高效的多任务微调。

**[MOSAIC: Multiple Observers Spotting AI Content](mosaic_multiple_observers_spotting_ai_content.md)**

:   基于信息论中的通用压缩原理，提出 MOSAIC——多 LLM 集成的 AI 生成文本检测方法，通过 Blahut-Arimoto 算法为多个 detector LLM 计算最优组合权重，构建混合分布作为观察者，比较文本的实际 surprisal 与混合模型的期望交叉熵差异来判断是否为 AI 生成，在多个域/语言/生成器上鲁棒优于单模型和双模型（如 Binoculars）方法。

**[Multi-Facet Blending for Faceted Query-by-Example Retrieval](multi-facet_blending_for_faceted_query-by-example_retrieval.md)**

:   > 提出 FaBle（Multi-Facet Blending）数据增强方法，通过对文档进行面向分解（decomposition）、面向生成（generation）、面向重组（recomposition）三阶段，仅用 1K 文档合成出面向条件的训练三元组，在数据稀缺条件下显著提升分面 QBE 检索效果，特别是在最具挑战性的 method 分面上超越了使用 130 万+ 数据训练的强基线。

**[Multi-Hop Question Generation via Dual-Perspective Keyword Guidance](multi-hop_question_generation_via_dual-perspective_keyword_guidance.md)**

:   定义了双视角关键词——问题关键词（捕捉提问者意图）和文档关键词（反映 QA 对相关内容），并提出 DPKG 框架，通过扩展 Transformer 编码器和两个答案感知解码器，将关键词无缝集成到多跳问题生成过程中。

**["My life is miserable, have to sign 500 autographs everyday": Exposing Humblebragging, the Brags in Disguise](my_life_is_miserable_have_to_sign_500_autographs_everyday_exposing_humblebraggin.md)**

:   首次将 humblebragging（谦虚式自夸）检测引入计算语言学领域，提出了一个4元组形式化定义，构建了 HB-24 合成数据集，并在 ML/DL/LLM 上进行了全面基准评估，GPT-4o 在 zero-shot+定义 设定下达到 0.88 F1，超越人类标注者。

**[Narrative Media Framing in Political Discourse](narrative_media_framing_in_political_discourse.md)**

:   将叙事学理论与媒体框架分析相结合，提出了包含角色（英雄/反派/受害者）、冲突/解决、文化故事三个结构化组件的叙事框架分析体系，在气候变化和 COVID-19 两个领域验证了该框架的有效性和可迁移性。

**[Unifying Continuous and Discrete Text Diffusion with Non-simultaneous Diffusion Processes](neodiff_unified_text_diffusion.md)**

:   提出 NeoDiff，通过引入"外在时间"（句子级扩散进度）和"内在时间"（token 级扩散进度）的双时间框架，利用 Poisson 过程为每个 token 独立分配细粒度噪声水平，并用上下文感知的时间预测器自适应调节去噪进度，统一了离散和连续文本扩散模型的理论框架，在机器翻译、复述、文本简化等多个任务上超越现有扩散基线。

**[Neural Parameter Search for Slimmer Fine-Tuned Models and Better Transfer](neural_parameter_search_for_slimmer_fine-tuned_models_and_better_transfer.md)**

:   提出Neural Parameter Search (NPS)，通过在task vector的低秩子空间中搜索最优权重系数来提升微调模型的剪枝效率，在知识迁移（+1.5%）、模型融合（+2.1%）和压缩（40%效率提升）三个场景下均取得显著改进。

**[Neuron Empirical Gradient: Discovering and Quantifying Neurons' Global Linear Controllability](neuron_empirical_gradient_discovering_and_quantifying_neurons_global_linear_cont.md)**

:   揭示了预训练语言模型 FF 层神经元激活值与模型输出之间存在全局线性关系，提出了神经元经验梯度（NEG）来量化这种线性关系，并设计了高效估算方法 NeurGrad，最终通过技能神经元探测实验证明 NEG 能有效表征多种语言技能。

**[On Support Samples of Next Word Prediction](on_support_samples_of_next_word_prediction.md)**

:   基于表示定理（representer theorem），研究语言模型下一词预测中训练样本的角色，发现两类支持样本（促进预测和抑制预测），并证明支持样本是样本的内在属性（训练前即可预测），而非支持样本对表示学习至关重要。

**[One for All: Update Parameterized Knowledge Across Multiple Models with Once Edit](one_for_all_update_parameterized_knowledge_across_multiple_models_with_once_edit.md)**

:   提出 OnceEdit，通过编辑一个轻量级插件模型并利用异构模型集成技术将编辑后的知识迁移到多个 LLM，实现"一次编辑，多模型更新"，在 ZsRE 和 Counterfact 数据集上显著超越现有方法。

**[Optimizing Decomposition for Optimal Claim Verification](optimizing_decomposition_for_optimal_claim_verification.md)**

:   提出动态分解（Dynamic Decomposition）框架，通过强化学习从验证器反馈中学习分解策略，将声明（claim）分解为验证器偏好的原子性粒度，弥合分解器与验证器之间的性能差距。

**[Partial Colexifications Improve Concept Embeddings](partial_colexifications_improve_concept_embeddings.md)**

:   首次将部分共词化（affix/overlap colexification）引入概念嵌入训练，在语义相似性建模、语义变化预测和词语联想预测三个任务上均优于仅使用完全共词化的基线。

**[Towards Better Evaluation for Generated Patent Claims](patclaimeval_patent_evaluation.md)**

:   提出首个专利权利要求评估基准 Patent-CE（1228 个专家标注的比较评估数据点）和专用评估方法 PatClaimEval（基于 Longformer + 对比学习变体），在特征完整性、概念清晰度、术语一致性、逻辑连接和整体质量五个维度上与人类专家评估的相关性全面超越 13 种现有指标（包括 G-Eval-4），整体质量维度的 Spearman 提升 58%。

**[Persistent Homology of Topic Networks for the Prediction of Reader Curiosity](persistent_homology_of_topic_networks_for_the_prediction_of_reader_curiosity.md)**

:   > 将文本的主题网络结构用持续同调 (Persistent Homology) 量化为拓扑空洞（连通分量、环、空腔），以此作为"信息空白"的代理变量来预测读者好奇心，在《饥饿游戏》小说上实现了 73% 的解释偏差（vs 基线 30%）。

**[Persona Dynamics: Unveiling the Impact of Personality Traits on Agents in Text-Based Games](persona_dynamics_unveiling_the_impact_of_persona_traits_on_agents_in_text-based_.md)**

:   提出 PANDA 方法，将人类人格特质（Big Five + Dark Triad 共8种）投射到文本游戏智能体的策略学习中，通过人格分类器引导 Q 值调整，发现高开放性（Openness）人格在冒险类文本游戏中表现显著优于其他人格类型。

**[PersonaBench: Evaluating AI Models on Understanding Personal Information through Accessing (Synthetic) Private User Data](personabench_evaluating_ai_models_on_understanding_personal_information_through_.md)**

:   提出 PersonaBench 基准及配套的合成私有数据生成管线，系统评估 AI 模型通过 RAG 从模拟用户数据中提取个人信息的能力，揭示当前方案的严重不足。

**[Personalized Generation In Large Model Era: A Survey](personalized_generation_in_large_model_era_a_survey.md)**

:   首篇跨模态个性化生成（PGen）系统综述，提出统一的用户中心视角将 NLP/CV/IR 社区的研究纳入同一框架，覆盖文本/图像/视频/音频/3D/跨模态六大模态。

**[All That Glitters is Not Novel: Plagiarism in AI Generated Research](plagiarism_ai_generated_research.md)**

:   在对自主科研 Agent（如 AI Scientist）生成的研究文档进行专家审查后发现，24% 的文档是"智能剽窃"——方法论与已有工作一一对应但不引用原始来源，且现有剽窃检测工具无法识别这种"改头换面"的抄袭。

**[PopAlign: Diversifying Contrasting Patterns for a More Comprehensive Alignment](popalign_diversifying_contrasting_patterns_for_a_more_comprehensive_alignment.md)**

:   提出PopAlign框架，从Prompt、Model、Pipeline三个层面构建六种多样化对比策略（包括创新的Elicitive Contrast），无需额外人工标注即可合成高质量偏好数据，实现更全面的LLM对齐。

**[Predicting Through Generation: Why Generation Is Better for Prediction](predicting_through_generation_why_generation_is_better_for_prediction.md)**

:   本文从信息论角度证明了token级生成比pooled表示保留更多互信息，提出PredGen框架通过scheduled sampling和task adapter解决生成式预测中的exposure bias和格式不匹配问题，并设计了Writer-Director Alignment Loss统一生成与预测目标。

**[Principled Understanding of Generalization for Generative Transformer Models in Arithmetic Reasoning Tasks](principled_generalization_arithmetic.md)**

:   建立首个统一理论框架来理解 Transformer 在算术任务（加法/乘法/模运算）上的泛化行为——从任务性质（平移不变性）和位置编码类型（APE/RPE）的交互出发，解释了之前困扰领域的多个泛化谜题（如加法能泛化但乘法不能，模100能泛化但模101不能），实验验证了理论预测。

**[Substance over Style: Evaluating Proactive Conversational Coaching Agents](proactive_conversational_coaching.md)**

:   通过健康教练领域的专家访谈和用户研究（31 名参与者、155 段对话），系统评估了五种不同对话风格（Directive、Interrogative、Facilitative）的 LLM 教练 Agent，发现用户高度重视核心功能性（substance）而对缺乏功能性时的风格修饰（style）持负面态度，同时揭示了用户第一人称评价与专家/LLM 第三方评价之间的显著不一致。

**[ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering](proxann_topic_model_eval.md)**

:   提出面向实际使用场景的主题模型评估协议ProxAnn，结合可扩展的人类评估流程和LLM代理标注者，发现最佳LLM代理在统计上与人类标注者不可区分，可作为自动化评估的合理替代。

**[PVP: An Image Dataset for Personalized Visual Persuasion with Persuasion Strategies, Viewer Characteristics, and Persuasiveness Ratings](pvp_an_image_dataset_for_personalized.md)**

:   构建了首个将图像说服策略与 2,521 位标注者心理特征（人格/价值观/道德基础）关联的大规模数据集 PVP（28,454 张图像、596 条行为消息、9 种说服策略），并在"个性化说服图像生成"和"说服力自动评估"两个基准任务上验证了心理特征对提升说服效果的关键作用。

**[QG-SMS: Enhancing Test Item Analysis via Student Modeling and Simulation](qg-sms_enhancing_test_item_analysis_via_student_modeling_and_simulation.md)**

:   QG-SMS 提出用单个 LLM 模拟不同理解水平的学生群体，通过学生画像生成、表现预测和分析三步流程，弥补了现有 LLM 评估器在考后分析维度（题目难度、区分度、干扰项效率）上的严重不足，在多个数据集上实现了最高一致性准确率。

**[QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning](qualispeech_a_speech_quality_assessment_dataset_with_natural_language_reasoning_.md)**

:   本文提出 QualiSpeech，首个包含 11 个维度标注和详细自然语言推理描述的语音质量评估数据集，以及配套的评测基准，证明了微调后的听觉 LLM 能生成关于噪声和失真的详细描述，并展示了推理增强质量评估的潜力。

**[Quantifying Lexical Semantic Shift via Unbalanced Optimal Transport](quantifying_lexical_semantic_shift_via_unbalanced_optimal_transport.md)**

:   将Unbalanced Optimal Transport（UOT）应用于上下文化词嵌入集合，提出Sense Usage Shift（SUS）指标在每个用法实例级别量化语义变化，统一解决实例级变化检测、词级变化幅度量化和词义扩展/缩小判定三项任务。

**[Rank, Chunk, and Expand: Lineage-Oriented Reasoning for Taxonomy Expansion](rank_chunk_and_expand_lineage-oriented_reasoning_for_taxonomy_expansion.md)**

:   LORex 提出了一个即插即用的分类体系扩展框架，结合判别式排序器 TEMPORA（基于欧拉路径的分类路径语言化）和迭代式 LLM 推理（语义过滤→父节点检索→路径验证），无需微调 LLM，在 4 个基准上实现了 12% 的准确率提升和 5% 的 Wu&P 提升。

**[Rationales Are Not Silver Bullets: Measuring the Impact of Rationales on Model Performance and Reliability](rationales_are_not_silver_bullets_measuring_the_impact_of_rationales_on_model_pe.md)**

:   本文通过对 18 个数据集、7 类任务的系统实验，发现在训练数据中加入 rationale（推理过程）并非总是有益——rationale 有时会削弱模型性能，但可以提升模型可靠性（校准度），且性能和可靠性的改善呈线性相关，两者都受任务固有难度驱动。

**[Re-identification of De-identified Documents with Autoregressive Infilling](reidentification_deidentified.md)**

:   提出一种基于 RAG 的去标识化文档重标识方法：先用稀疏+稠密检索找到相关背景文档，再用自回归填充模型推断被遮蔽的个人标识信息，在三个数据集上恢复了高达 80% 的被遮蔽文本。

**[Identifying Reliable Evaluation Metrics for Scientific Text Revision](reliable_eval_metrics_scientific.md)**

:   系统分析了传统相似度指标（ROUGE、BERTScore 等）在科学文本修订评估中的局限性，发现它们与编辑距离强相关且惩罚深度修改，提出结合 LLM-as-Judge 和任务特定跨域指标的混合评估方法，在与人类判断的对齐度上显著优于单一指标。

**[RePanda: Pandas-powered Tabular Verification and Reasoning](repanda_pandas-powered_tabular_verification_and_reasoning.md)**

:   提出 RePanda，将自然语言声明翻译为可执行的 pandas 查询来实现表格事实验证，在 TabFact 上达到 84.09% 准确率，在 OOD 的 WikiFact 上无需额外微调达 84.72%，同时以仅 7B 参数的模型逼近 671B DeepSeek-Chat 的零样本性能，并扩展至表格问答任务取得 75.1% 准确率。

**[Research Borderlands: Analysing Writing Across Research Cultures](research_borderlands_analysing_writing_across_research_cultures.md)**

:   通过访谈跨学科研究者构建学术写作文化规范框架（结构/风格/修辞/引用四类），并用计算指标量化11个CS社区的写作差异，揭示LLM在跨社区写作改编时存在严重的"同质化"倾向。

**[Retrospective Learning from Interactions](retrospective_learning_from_interactions.md)**

:   提出 ReSpect 方法，让多模态 LLM 通过回顾性地解码用户在多轮交互中的隐式反馈信号来自我改进，无需任何外部标注，在数千次人机交互中将任务完成率从 31% 提升至 82%。

**[Revisiting Weak-to-Strong Generalization: Reverse KL vs. Forward KL](revisiting_weak-to-strong_generalization_in_theory_and_practice_reverse_kl_vs_fo.md)**

:   在 Weak-to-Strong Generalization (W2SG) 框架中，提出用 reverse KL 替代 forward KL 作为损失函数——理论证明 reverse KL 的 mode-seeking 特性可保证强模型超过弱监督者至少"分歧量"的幅度，实验在 GPT-2/Pythia/Qwen2.5 系列上验证 reverse KL/CE 在 12/12 设置中超越 forward KL 且噪声鲁棒性更好。

**[RMoA: Optimizing Mixture-of-Agents through Diversity Maximization and Residual Compensation](rmoa_optimizing_mixture-of-agents_through_diversity_maximization_and_residual_co.md)**

:   受 ResNet 残差学习启发，提出 RMoA 框架，通过嵌入式多样性贪心选择、残差提取/聚合智能体和自适应终止机制来优化多智能体协作架构，在降低计算开销的同时实现 SOTA 性能。

**[RoToR: Towards More Reliable Responses for Order-Invariant Inputs](rotor_towards_more_reliable_responses_for_order-invariant_inputs.md)**

:   提出 RoToR，一种基于全局排序和循环位置编码分配的零样本顺序不变语言模型，通过最小化位置 ID 修改来实现稳定的顺序不变性，并设计选择路由（Selective Routing）机制自适应处理混合输入类型。

**[(RSA)²: A Rhetorical-Strategy-Aware Rational Speech Act Framework for Figurative Language Understanding](rsatexttwosuperior_a_rhetorical-strategy-aware_rational_speech_act_framework_for.md)**

:   本文提出 (RSA)² 框架，通过在概率语用学的 RSA 框架中显式建模说话者的修辞策略（如反讽、夸张），使 LLM 能够在不建模说话者动机的情况下正确理解非字面义，在反讽理解数据集 PragMega+ 上达到 SOTA。

**[Rubrik's Cube: Testing a New Rubric for Evaluating Explanations on the CUBE Dataset](rubriks_cube_testing_a_new_rubric_for_evaluating_explanations_on_the_cube_datase.md)**

:   提出 Rubrik——受教育评估启发的解释质量评价量规，基于三层嵌套类型体系（Commentary⊆Justification⊆Argument）+ 8 维质量维度，配套 CUBE 数据集（26K 条由人类和 6 个 LLM 生成的解释），发现 LLM 解释低质主因是缺乏简洁性而非连贯性。

**[S2WTM: Spherical Sliced-Wasserstein Autoencoder for Topic Modeling](s2wtm_spherical_sliced-wasserstein_autoencoder_for_topic_modeling.md)**

:   提出 S2WTM，一种基于球面切片 Wasserstein 自编码器的主题模型，在超球面潜空间上对齐聚合后验与先验分布，有效避免 VAE 的后验坍塌问题，同时在主题连贯性和多样性上超越现有 SOTA。

**[S3 - Semantic Signal Separation](s3_-_semantic_signal_separation.md)**

:   S3将主题建模概念化为发现语义空间中独立语义轴的过程，利用独立成分分析（ICA）分解文档嵌入矩阵，无需预处理即可产生高度连贯且多样化的主题，同时是最快的上下文主题模型（平均比BERTopic快4.5倍）。

**[SDD: Self-Degraded Defense against Malicious Fine-tuning](sdd_self-degraded_defense_against_malicious_fine-tuning.md)**

:   SDD通过训练LLM对有害指令生成高质量但无关的良性回复来实现防御：当攻击者进行恶意微调时，模型的通用能力会显著下降，从而无法有效执行恶意指令。

**[Segment-Based Attention Masking for GPTs](segment-based_attention_masking_for_gpts.md)**

:   MAS（Masked Attention by Segment）在预训练 GPT 模型的 prefill 阶段将因果注意力掩码替换为按段（segment）的双向注意力——同一段内的 token 可以互相 attend，生成阶段仍保持因果掩码——通过 LoRA 微调即可在 8 个常识推理任务上一致提升性能（Llama-3-8B 平均 +1.8%，Llama-3.2-3B +3.3%），无额外计算开销。

**[Self-Foveate: Enhancing Diversity and Difficulty of Synthesized Instructions from Unsupervised Text via Multi-Level Foveation](self-foveate_enhancing_diversity_and_difficulty_of_synthesized_instructions_from.md)**

:   提出 Self-Foveate 方法，受人类视觉注视机制启发，通过"微观-散射-宏观"三级注视策略，从无监督文本中系统性提取多粒度信息，合成具有更高多样性和难度的指令数据，用于 LLM 的指令微调。

**[SEOE: A Scalable and Reliable Semantic Evaluation Framework for Open Domain Event Detection](seoe_semantic_eval.md)**

:   针对开放域事件检测（ODED）评估的两大痛点——有限 benchmark 缺乏真实世界代表性、token 级匹配指标无法捕捉语义相似性——提出 SEOE 框架，构建包含 564 种事件类型覆盖 7 大领域的可扩展 benchmark，并引入基于 LLM 的语义 F1 评估指标。

**[Sightation Counts: Leveraging Sighted User Feedback in Building a BLV-aligned Dataset of Diagram Descriptions](sightation_counts_leveraging_sighted_user_feedback_in_building_a_blv-aligned_dat.md)**

:   提出让视力正常者「评估」而非「生成」VLM 的图表描述，构建了首个经 BLV 专业教育者验证的 5k 图表 / 137k 样本多任务数据集 Sightation，偏好微调 2B 模型后在 BLV 有用性评分上平均提升 1.67σ。

**[Sleepless Nights, Sugary Days: Creating Synthetic Users with Health Conditions for Realistic Coaching Agent Interactions](sleepless_nights_sugary_days_creating_synthetic_users_with_health_conditions_for.md)**

:   提出一个端到端框架，基于真实人口学、健康/生活方式和行为/心理特征数据生成有健康状况的合成用户（涵盖睡眠和糖尿病管理），用于评估健康教练Agent的交互质量，并通过人类专家评估验证其显著优于通用合成用户。

**[SoRFT: Issue Resolving with Subtask-oriented Reinforced Fine-Tuning](sorft_issue_resolving_with_subtask-oriented_reinforced_fine-tuning.md)**

:   提出 SoRFT（Subtask-oriented Reinforced Fine-Tuning），将 GitHub Issue 解决任务分解为文件定位、函数定位、行定位和代码编辑四个子任务，通过拒绝采样SFT + 基于规则的PPO强化学习两阶段训练，显著提升开源LLM在 SWE-Bench 上的 Issue 解决能力。

**[SOTOPIA-Ω: Dynamic Strategy Injection Learning and Social Instruction Following Evaluation for Social Agents](sotopia-ensuremathomega_dynamic_strategy_injection_learning_and_social_instructi.md)**

:   本文提出 SOTOPIA-Ω 框架，通过将谈判理论中的多步推理策略和直接策略动态注入到专家 agent 中，自动构建高质量社交对话训练语料，并定义了"社交指令跟随（S-IF）"这一新概念和两个评估指标，在 7B 模型上超越了 GPT-4 专家 agent 的社交目标达成能力。

**[SPOT: Bridging Natural Language and Geospatial Search for Investigative Journalists](spot_bridging_natural_language_and_geospatial_search_for_investigative_journalis.md)**

:   提出 SPOT 系统，通过微调 LLaMA 3 将自然语言场景描述转换为 YAML 查询，结合语义标签捆绑机制实现对 OpenStreetMap 数据的可靠自然语言访问，服务于调查新闻的地理定位验证。

**[Spotting Out-of-Character Behavior: Atomic-Level Evaluation of Persona Fidelity in Open-Ended Generation](spotting_out-of-character_behavior_atomic-level_evaluation_of_persona_fidelity_i.md)**

:   提出原子级（句子级）评估框架，通过三个指标（ACC_atom、IC_atom、RC_atom）细粒度检测大语言模型在开放式文本生成中的角色偏离（Out-of-Character）行为，弥补了传统整体评分方法无法捕捉长文本中微妙人格不一致的问题。

**[Statistical Deficiency for Task Inclusion Estimation](statistical_deficiency_task_inclusion.md)**

:   基于统计缺陷性（statistical deficiency）理论，提出一种理论驱动的任务包含关系（task inclusion）定义与度量框架，以信息充分性（information sufficiency, IS）作为可计算代理指标，通过比较微调模型的中间层表征来估计任务间的包含程度，并在合成数据和真实NLP任务上成功重建了经典NLP pipeline的层次关系。

**[STRICTA: Structured Reasoning in Critical Text Assessment for Peer Review and Beyond](stricta_structured_reasoning_in_critical_text_assessment_for_peer_review_and_bey.md)**

:   提出 STRICTA 框架，基于结构因果模型（SCM）将文本评审建模为显式的逐步推理图（workflow），在生物医学论文评审中收集 40+ 位专家的 4000+ 推理步骤数据集，发现先验知识差异是专家分歧主因、写作风格对最终评审有因果影响，LLM 存在错误传播但人类监督可有效缓解。

**[Subword Models Struggle with Word Learning, but Surprisal Hides It](subword_models_struggle_with_word_learning_but_surprisal_hides_it.md)**

:   本文通过心理语言学的词汇判断任务（lexical decision task）揭示了使用子词（BPE）分词的语言模型在单词学习方面存在严重缺陷，而基于字符级分词的模型能轻松完成该任务；当使用 surprisal（在语境中的出乎意料程度）来评估时，这一差距被掩盖了。

**[SudoLM: Learning Access Control of Parametric Knowledge with Authorization Alignment](sudolm_authorization_alignment.md)**

:   SudoLM 提出了一种 LLM 参数化知识访问控制框架，通过"SUDO key"机制让授权用户解锁受限知识（如医学领域知识），未授权用户则只能访问公开知识，用 DPO 的 authorization alignment 在一个模型内实现了传统需要多版本模型才能完成的分级访问控制。

**[Synergistic Weak-Strong Collaboration by Aligning Preferences](synergistic_weak-strong_collaboration_by_aligning_preferences.md)**

:   本文提出 CoWest 框架，通过让专业化的弱模型（如 LLaMA3-8B）生成初始草稿，再由通用强模型（如 GPT-4）精炼，并利用协作反馈通过 DPO 微调弱模型以对齐强模型偏好，在反事实推理、医学和伦理三个领域显著超越单模型和已有协作方法。

**[Synthia: Novel Concept Design with Affordance Composition](synthia_novel_concept_design_with_affordance_composition.md)**

:   Synthia 提出了一种基于 affordance（功能可供性）组合的新颖概念设计框架，通过层次化概念本体、affordance 采样策略和课程学习微调 T2I 模型，生成既视觉新颖又功能连贯的创新设计。

**[TabXEval: Why this is a Bad Table? An eXhaustive Rubric for Table Evaluation](tabxeval_why_this_is_a_bad_table_an_exhaustive_rubric_for_table_evaluation.md)**

:   TabXEval 提出了一种基于结构化评分规则（rubric）的两阶段表格评估框架——先通过 TabAlign 对齐参考表和生成表的结构，再通过 TabCompare 进行语义和语法层面的细粒度比较，同时构建了多领域基准 TabXBench。

**[TACLR: A Scalable and Efficient Retrieval-Based Method for Industrial Product Attribute Value Identification](taclr_a_scalable_and_efficient_retrieval-based_method_for_industrial_product_att.md)**

:   TACLR 提出了首个基于检索范式的产品属性值识别（PAVI）方法，通过分类感知对比学习和自适应推理机制，在处理隐含值、OOD 值和归一化输出方面全面超越分类和生成方法，并已成功部署在闲鱼（Xianyu）平台。

**[Tag-Evol: Achieving Efficient Instruction Evolving via Tag Injection](tag-evol_achieving_efficient_instruction_evolving_via_tag_injection.md)**

:   Tag-Evol 提出了一种基于知识标签注入的指令进化框架，通过构建多步细粒度标签池和预算控制注入机制，无需迭代即可生成不同难度的高质量进化指令数据，在多任务多骨干上显著优于 Evol-Instruct。

**[TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)**

:   TARGA 提出了一种针对性的合成数据生成框架，无需任何人工标注即可为知识库问答（KBQA）动态生成高相关性的合成示例用于上下文学习，仅用 7B 模型即在 GrailQA（+7.7 F1）和 KBQA-Agent（+12.2 F1）上大幅超越所有非微调方法。

**[Task-Informed Anti-Curriculum by Masking Improves Downstream Performance on Text](task-informed_anti-curriculum_by_masking_improves_downstream_performance_on_text.md)**

:   TIACBM 提出了一种任务感知的反课程掩码微调策略：利用下游任务知识（如情感极性、词性标签）决定哪些 token 被掩码，并采用周期衰减的掩码率，在情感分析、文本分类和作者归属三个任务上均取得统计显著的性能提升。

**[Temporal Reasoning for Timeline Summarisation in Social Media](temporal_reasoning_for_timeline_summarisation_in_social_media.md)**

:   本文提出通过构建新的叙事时序推理数据集 NarrativeReason 来增强 LLM 的时序推理能力，并通过知识蒸馏框架将时序推理知识迁移到小模型中，同时训练其完成时间线摘要任务，在跨域心理健康摘要任务上取得最优效果并显著减少幻觉。

**[TestNUC: Enhancing Test-Time Computing Approaches and Scaling through Neighboring Unlabeled Data Consistency](testnuc_enhancing_test-time_computing_approaches_and_scaling_through_neighboring.md)**

:   TestNUC 提出了一种线性扩展的测试时推理增强方法，通过检索测试样本的近邻无标注数据，让 LLM 同时预测测试样本及其邻居，再通过加权多数投票聚合，稳定提升分类准确率。

**[The AI Gap: How Socioeconomic Status Affects Language Technology Interactions](the_ai_gap_how_socioeconomic_status_affects_language_technology_interactions.md)**

:   通过对1000名不同社会经济地位(SES)用户的大规模调查和6482条真实LLM prompts的分析，揭示了高低SES群体在语言技术使用频率、交互方式和话题选择上存在显著系统性差异，呼吁开发更具包容性的NLP技术以缩小AI鸿沟。

**[The Harmonic Structure of Information Contours](the_harmonic_structure_of_information_contours.md)**

:   提出 Harmonic Surprisal (HS) 假说——文本中 surprisal 曲线呈周期性波动且周期与语篇结构（EDU/句子/段落）对齐，用带时间缩放的谐波回归检验，在 6 种语言上发现一致的周期模式，精化了经典的 Uniform Information Density 假说。

**[The Hidden Attention of Mamba Models](the_hidden_attention_of_mamba_models.md)**

:   揭示了Mamba（选择性状态空间模型S6）可以被重新表述为一种隐式的因果自注意力机制，并基于此提出了适用于Mamba模型的注意力可视化和可解释性方法（Attention Rollout和Mamba-Attribution），证明其可解释性指标与Transformer相当。

**[The Knowledge Microscope: Features as Better Analytical Lenses than Neurons](the_knowledge_microscope_features_as_better_analytical_lenses_than_neurons.md)**

:   本文通过系统实验验证了 SAE（稀疏自编码器）分解出的特征（features）在知识表达影响力、可解释性、单义性（monosemanticity）三个维度上全面优于传统神经元（neurons）作为分析单元，并提出首个基于 feature 的模型编辑方法 FeatureEdit，在隐私知识擦除任务上大幅超越神经元方法。

**[The Noisy Path from Source to Citation: Measuring How Scholars Engage with Past Research](the_noisy_path_from_source_to_citation_measuring_how_scholars_engage_with_past_r.md)**

:   构建大规模计算流水线量化学术引用的忠实度（fidelity），分析 1300 万引用句对揭示了影响引用忠实度的关键因素，并通过准因果实验证实了"电话效应"——低忠实度中间引用会导致后续引用进一步失真。

**[The Time Scale of Redundancy between Prosody and Linguistic Context](the_time_scale_of_redundancy_between_prosody_and_linguistic_context.md)**

:   本文系统研究了韵律特征（如音高、响度、时长等）与语言上下文之间冗余性的时间尺度，发现韵律与过去上下文的冗余性跨越较长时间尺度（3-8个词），而与未来上下文的冗余性仅限于短时间尺度（1-2个词），揭示了韵律在语音交流中帮助整合过去信息和预测即将出现的词汇的双重作用。

**[Theorem Prover as a Judge for Synthetic Data Generation](theorem_prover_as_a_judge_for_synthetic_data_generation.md)**

:   提出 TP-as-a-Judge 框架，利用 Lean 定理证明器验证 LLM 生成的中间推理步骤，结合迭代自动形式化和基于定理证明器反馈的强化学习（RLTPF），仅用 3,508 个样本就在多个数学推理基准上取得了显著提升。

**[Theoretical Guarantees for Minimum Bayes Risk Decoding](theoretical_guarantees_for_minimum_bayes_risk_decoding.md)**

:   本文首次为最小贝叶斯风险（MBR）解码提供了严格的理论收敛性保证，证明了在参考假设集大小为 $n$ 时，MBR 解码以 $O(n^{-1/2})$ 的速率逼近最优解，并与 MAP 解码进行了理论比较，表明 MBR 在多种场景下收敛更快。

**[Learning to Reason Over Time: Timeline Self-Reflection for Temporal Reasoning](tiser_timeline_self_reflection_temporal.md)**

:   提出 TISER 框架，通过"推理→时间线构建→自反思→答案生成"四阶段管道实现LLM时间推理的test-time scaling，配合合成推理轨迹数据微调，让 7B 开源模型在多个时间推理基准上超越 GPT-4，在TGQA等任务上达到 SOTA。

**[Tokenisation is NP-Complete](tokenisation_is_np-complete.md)**

:   证明了分词问题（tokenisation）的两种变体——直接分词和自底向上分词——都是 NP 完全的，通过从 max-2-SAT 问题多项式时间归约实现，这意味着不可能找到高效的最优分词算法，BPE 等近似方法是合理选择。

**[Towards Comprehensive Argument Analysis in Education: Dataset, Tasks, and Method](towards_comprehensive_argument_analysis_in_education_dataset_tasks_and_method.md)**

:   本文针对中文高中议论文，提出包含纵向（论证关系）和横向（话语关系）两个维度共 14 种细粒度论证关系类型的标注方案，并在论证成分检测、关系预测和自动评分三个任务上建立了全面的 benchmark。

**[Towards Style Alignment in Cross-Cultural Translation](towards_style_alignment_in_cross-cultural_translation.md)**

:   本文首次将"风格对齐"定义为跨文化翻译的核心目标，系统揭示了 LLM 翻译中的风格中性化偏差和英语中心偏差，并提出 RASTA 方法在嵌入空间中学习文化对齐映射来检索风格匹配的少样本示例，在不降低翻译质量的前提下将风格对齐度提升最高 56%。

**[Towards Text-Image Interleaved Retrieval](towards_text-image_interleaved_retrieval.md)**

:   定义文本-图像交错检索（TIIR）新任务，构建基于 wikiHow 的首个 TIIR 基准数据集（155K 文档、7654 测试对），并提出 Matryoshka Multimodal Embedder（MME）通过多粒度视觉 token 压缩解决 MLLM 中视觉 token 过多导致的效率和语义偏差问题，大幅提升检索性能。

**[Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis](tree-of-debate_multi-persona_debate_trees_elicit_critical_thinking_for_scientifi.md)**

:   提出Tree-of-Debate (ToD)框架，将科学论文转化为LLM persona进行树结构化辩论，通过自我审议、迭代检索和主持人引导的层级子话题扩展，生成细粒度、上下文化的论文对比摘要，在领域专家评估中显著优于基线方法。

**[TROVE: A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Classification](trove_a_challenge_for_finegrained_text.md)**

:   提出TROVE文本溯源挑战，将目标文本中每个句子追溯到源文档中的具体源句，并分类其细粒度关系（引用、压缩、推理等），覆盖多文档和长文档场景。

**[Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)**

:   Tuna 构建了 1000 个时间密集短视频的细粒度多维标注数据集，配套字幕评测（事件拆分→匹配→关系分类）和时序问答两个任务，系统性地暴露了当前视频 LMM 在动态时序理解上的弱点。

**[Understanding Common Ground Misalignment in Goal-Oriented Dialog: A Case-Study with Ubuntu Chat Logs](understanding_common_ground_misalignment_in_goal-oriented_dialog_a_case-study_wi.md)**

:   本文通过在 Ubuntu IRC 技术支持对话中标注"对话摩擦"（conversational friction），实证揭示了共识基础（common ground）的失配与任务成功率之间的显著关联，并发现 LLM 能识别显式的对话摩擦但难以处理需要语用或领域推理的隐式摩擦。

**[Understanding Cross-Domain Adaptation in Low-Resource Topic Modeling](understanding_cross-domain_adaptation_in_low-resource_topic_modeling.md)**

:   首次将领域自适应形式化引入低资源主题建模，推导有限样本泛化上界指导方法设计，提出 DALTA 框架通过共享编码器、领域专用解码器和对抗对齐实现跨领域主题知识的选择性迁移。

**[Uni-Retrieval: A Multi-Style Retrieval Framework for STEM's Education](uni-retrieval_a_multi-style_retrieval_framework_for_stems_education.md)**

:   本文提出面向 STEM 教育场景的多风格多模态检索任务和数据集 SER（24,000+ 查询对），以及基于 Prompt Bank 的轻量检索模型 Uni-Retrieval，通过原型学习提取查询风格特征并动态选择提示向量来增强不同风格（文本、草图、艺术、低分辨率、语音）的检索性能，在 STEM 教育检索和传统检索数据集上均超越已有方法。

**[Unifying Language Agent Algorithms with Graph-based Orchestration Engine for Reproducible Agent Research](unifying_language_agent_algorithms_with_graph-based_orchestration_engine_for_rep.md)**

:   提出 AGORA 框架，通过 DAG 图编排引擎将 CoT、ReAct、ToT、RAP 等 10 种主流 Agent 推理算法统一为可插拔的 Operator 模块，在数学推理和多模态任务上系统比较后发现：简单的 CoT 方法在准确率和成本效益上往往优于复杂算法，而一句提示语改动就能带来 90% 的性能飞跃。

**[Unique Hard Attention: A Tale of Two Sides](unique_hard_attention_a_tale_of_two_sides.md)**

:   本文证明在有限精度transformer中，leftmost unique hard attention (UHA)严格弱于rightmost UHA，前者等价于线性时序逻辑片段LTL[$\Diamond^-$]（即部分有序有限自动机），并与soft attention transformer表达能力等价，从而精确刻画了注意力方向性对transformer表达力的影响。

**[Unlocking Speech Instruction Data Potential with Query Rewriting](unlocking_speech_instruction_data_potential_with_query_rewriting.md)**

:   提出基于多LLM知识融合的查询重写框架与多智能体标注验证方法，将超出TTS词汇范围的文本指令重写为适合语音合成的形式，使语音指令数据可用率从72%提升至93%，为端到端大型语音语言模型(LSLM)构建高质量语音指令数据集。

**[USDC: A Dataset of User Stance and Dogmatism in Long Conversations](usdc_a_dataset_of_underlineuser_underlinestance_and_underlinedogmatism_in_long_u.md)**

:   构建 USDC——首个用户级长对话立场和教条主义数据集，764 个多用户 Reddit 对话（22 子版块），用 {Mistral Large, GPT-4} × {zero/one/few-shot} 共 6 设置多数投票标注立场(5级)+教条程度(4级)，并用 7 个 SLM 微调/指令微调建立基线。

**[Using Shapley Interactions to Understand How Models Use Structure](using_shapley_interactions_to_understand_how_models_use_structure.md)**

:   利用Shapley Taylor交互指数（STII）跨模态（文本+语音）系统分析语言模型如何通过非线性交互编码句法结构、非组合语义和语音协同发音，发现自回归模型在句法编码上显著优于遮蔽模型。

**[Using Source-Side Confidence Estimation for Reliable Translation into Unfamiliar Languages](using_source-side_confidence_estimation_for_reliable_translation_into_unfamiliar.md)**

:   提出一种基于梯度归因的源端置信度估计方法，通过测量输出序列对源端嵌入的敏感度来识别可能误译的源端词汇，无需词对齐，在误译检测任务上显著优于传统对齐方法。

**[Value Residual Learning](value_residual_learning.md)**

:   提出 ResFormer 和 SVFormer，通过在注意力机制中引入第一层 Value 向量到后续层的残差连接，增强初始 token 级信息在深层网络中的传播，以比标准 Transformer 少 16.11% 的参数和 20.3% 的训练数据达到同等性能，SVFormer 还能减少近一半 KV 缓存。

**[VAQUUM: Are Vague Quantifiers Grounded in Visual Data?](vaquum_are_vague_quantifiers_grounded_in_visual_data.md)**

:   本文发布了VAQUUM数据集（20,300条人类评分，1,089张图片），系统评估视觉语言模型在模糊量词（few/many等）使用上与人类的一致性，发现VLM像人类一样受物体数量影响，但不同评估范式下模型表现差异大，表明判断和生成模糊量词依赖不同认知过程。

**[Verbosity-Aware Rationale Reduction: Effective Reduction of Redundant Rationale](verbosity-aware_rationale_reduction_effective_reduction_of_redundant_rationale_v.md)**

:   提出 VARR 框架，以句子为单位并利用基于似然度的"冗余度（verbosity）"标准识别和移除推理路径中的冗余句子，在多种推理任务上平均提升 7.71% 准确率同时减少 19.87% 的 token 生成。

**[Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction](visual_cues_enhance_predictive_turn-taking_for_two-party_human_interaction.md)**

:   提出 MM-VAP 多模态预测性话轮转换模型，将面部表情、头部姿态和注视方向等视觉线索引入语音预测模型，在视频会议语料上将 hold/shift 预测准确率从 79% 提升至 84%。

**[Well Begun is Half Done: Low-resource Preference Alignment by Weak-to-Strong Decoding](well_begun_is_half_done_low-resource_preference_alignment_by_weak-to-strong_deco.md)**

:   提出 Weak-to-Strong Decoding (WSD) 框架，利用一个小型对齐模型为大型基座模型起草对齐的开头，再由大模型续写，以低资源方式实现偏好对齐且不产生 alignment tax。

**[What is Stigma Attributed to? A Theory-Grounded, Expert-Annotated Interview Corpus for Demystifying Mental-Health Stigma](what_is_stigma_attributed_to_a_theory-grounded_expert-annotated_interview_corpus.md)**

:   基于归因理论构建了一个专家标注的心理健康污名访谈语料库（4,141 条片段、684 名参与者），涵盖 7 种细粒度污名类型和社会文化背景信息，并以此为基准评测了多种 SOTA 神经模型在污名检测任务上的表现与挑战。

**[What Matters in Evaluating Book-Length Stories? A Systematic Study of Long Story Evaluation](what_matters_in_evaluating_book-length_stories_a_systematic_study_of_long_story_.md)**

:   本文系统研究了书籍级长篇故事（>100K tokens）的自动评估问题，构建了首个大规模长篇故事评估基准LongStoryEval（600本新出版小说、340K条读者评论），提出分层评价标准体系，比较三种评估策略的有效性，并训练了专用评估模型NovelCritique-8B，在与人类评分的对齐度上超越GPT-4o。

**[When to Speak, When to Abstain: Contrastive Decoding with Abstention](when_to_speak_when_to_abstain.md)**

:   提出 CDA（Contrastive Decoding with Abstention），一种免训练解码方法，通过熵校准的不确定性估计让 LLM 在参数/上下文知识可用时生成正确回答、在两者都不可靠时主动弃权，覆盖全部四种知识可用性场景。

**[Why Are Positional Encodings Nonessential for Deep Autoregressive Transformers? Revisiting a Petroglyph](why_are_positional_encodings_nonessential_for_deep_autoregressive_transformers_r.md)**

:   重新阐释并溯源一个 pre-LLM 时代已知但被遗忘的结论——多层自回归 Transformer 语言模型无需显式位置编码即可区分排列序列，因为级联的（排列不变的）集合处理器在因果掩码下集体展现出完全位置敏感性；同时反思了 LLM 时代的知识断层和引用偏差。

**[Words of Warmth: Trust and Sociability Norms for over 26k English Words](words_of_warmth_trust_and_sociability_norms_for_over_26k_english_words.md)**

:   通过严格的众包标注流程构建了首个大规模词汇-温暖（Warmth）、信任（Trust）和社交性（Sociability）关联词典（覆盖 26k+ 英语单词），并通过儿童词汇习得分析和社交媒体刻板印象案例研究，展示了该资源在社会认知研究中的广泛价值。

**[X-Turing: Towards an Enhanced and Efficient Turing Test for Long-Term Dialogue Agents](xturing_enhanced_turing_test.md)**

:   提出 X-Turing 框架，通过引入 burst 对话模式和伪对话生成技术来增强和高效化图灵测试，能够评估 LLM 在长期对话中的人类模仿能力，发现 LLM 随着对话轮次增加表现显著下降。

**[You need to MIMIC to get FAME: Solving Meeting Transcript Scarcity with Multi-Agent Conversations](you_need_to_mimic_to_get_fame_solving_meeting_transcript_scarcity_with_a_multi-a.md)**

:   提出 MIMIC 框架，通过多智能体辩论模拟生成合成会议转录，构建了包含 800 场会议的 FAME 数据集（500 英语 + 300 德语），并设计了基于心理学的行为真实性评估框架。

**[Your Model is Overconfident, and Other Lies We Tell Ourselves](your_model_is_overconfident_and_other_lies_we_tell_ourselves.md)**

:   通过对 29 个模型在 ChaosNLI 和 DynaSent 数据集上的全面分析，揭示了标注者分歧、训练动态、模型置信度等数据复杂度指标之间存在相关性但非线性非单调的关系，挑战了"模型不确定性 ≈ 人类分歧"这一常见假设。

**[Zero-Shot Conversational Stance Detection: Dataset and Approaches](zero-shot_conversational_stance_detection_dataset_and_approaches.md)**

:   构建了首个零样本多轮多方对话立场检测数据集 ZS-CSD（280 个目标、17,063 条对话样本），并提出 SITPCL 模型，结合说话者交互编码器与目标感知原型对比学习，在零样本对话立场检测中取得 SOTA（F1-macro 43.81%）。
