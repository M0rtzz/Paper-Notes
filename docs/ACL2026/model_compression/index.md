---
title: >-
  ACL2026 模型压缩方向42篇论文解读
description: >-
  42篇ACL2026的模型压缩方向论文解读，涵盖 LLM、模型压缩、Agent、压缩/编码、推理、个性化生成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📦 模型压缩

**💬 ACL2026** · **42** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (51)](../../CVPR2026/model_compression/index.md) · [🔬 ICLR2026 (91)](../../ICLR2026/model_compression/index.md) · [🤖 AAAI2026 (54)](../../AAAI2026/model_compression/index.md) · [🧠 NeurIPS2025 (137)](../../NeurIPS2025/model_compression/index.md) · [📹 ICCV2025 (48)](../../ICCV2025/model_compression/index.md) · [🧪 ICML2025 (70)](../../ICML2025/model_compression/index.md)

🔥 **高频主题：** LLM ×9 · 模型压缩 ×3 · Agent ×3 · 压缩/编码 ×3 · 推理 ×3

**[A Computational Method for Measuring "Open Codes" in Qualitative Analysis](a_computational_method_for_measuring_34open_codes34_in_qualitative_analysis.md)**

:   提出一种基于理论的计算方法，通过LLM增强的代码合并算法和四个无需ground truth的指标（Coverage, Overlap, Novelty, Divergence），系统评估人类和AI在归纳定性编码中的表现。

**[A Layer-wise Analysis of Supervised Fine-Tuning](a_layer-wise_analysis_of_supervised_fine-tuning.md)**

:   通过信息论、几何和优化三个视角对 1B-32B 模型的 SFT 进行逐层分析，发现指令跟随能力集中在中间层（20%-80%），而非均匀分布，据此提出 Mid-Block Efficient Tuning 策略，选择性更新中间层，在 GSM8K 上比标准 LoRA 提升高达 10.2%。

**[Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)**

:   提出ASL（Adaptive Selection Layer），通过监控token注意力分数排名的方差来自适应确定KV缓存剪枝的层位置，在困难任务上显著优于固定层选择方法，同时保持无需训练。

**[Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis](analytical_ffn-to-moe_restructuring_via_activation_pattern_analysis.md)**

:   提出一种分析式后训练框架，通过神经元激活模式分析将dense FFN快速重构为sparse MoE——区分高频共享专家和低频路由专家，并从激活统计量构建路由器，仅需2k样本微调即可实现1.17×加速。

**[arXiv2Table: Toward Realistic Benchmarking and Evaluation for LLM-Based Literature-Review Table Generation](arxiv2table_toward_realistic_benchmarking_and_evaluation_for_llm-based_literatur.md)**

:   提出 arXiv2Table 基准（1,957 张表、7,158 篇论文），通过引入干扰论文、模式无关的用户需求和基于 QA 的无标注评估框架，实现更真实的 LLM 文献综述表格生成评估，并提出迭代批处理生成方法。

**[Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference](calibrated_speculative_decoding_frequency-guided_candidate_selection_for_efficie.md)**

:   CSD 提出一种训练免的推测解码增强框架，通过在线校正记忆（OCM）记录高频拒绝模式提供救援候选，再用语义一致性门控（SCG）基于概率比验证候选可靠性，将推测解码的吞吐量提升至最高 2.33×，同时在 HumanEval 和 MATH500 上甚至提升了准确率。

**[CBRS: Cognitive Blood Request System with Bilingual Dataset and Dual-Layer Filtering](cbrs_cognitive_blood_request_system_with_bilingual_dataset_and_dual-layer_filter.md)**

:   CBRS 提出一个多平台框架，通过双层过滤架构（轻量分类器 + LLM）从社交媒体消息流中高效检测并解析血液捐献请求，构建了首个包含 11K 条孟加拉语-英语-转写孟加拉语的血液捐献请求数据集，LoRA 微调的 Llama-3.2-3B 在解析任务上达到 92% 零样本准确率。

**[ChemAmp: Amplified Chemistry Tools via Composable Agents](chemamp_amplified_chemistry_tools_via_composable_agents.md)**

:   提出"工具放大"新范式（区别于传统的工具编排），通过 ChemAmp 框架将化学专用工具（UniMol2、Chemformer等）作为可组合积木块动态构建任务专用超级智能体，在分子设计、反应预测等四个核心化学任务上超越专用模型和通用LLM，同时推理token成本减少94%。

**[CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents](clag_adaptive_memory_organization_via_agent-driven_clustering_for_small_language.md)**

:   本文提出 CLAG，一种基于聚类的 Agent 记忆框架，通过 SLM 驱动的路由将记忆组织到语义一致的聚类中，在聚类内部进行局部进化更新，并通过两阶段检索过滤噪声，在多个 QA 数据集上显著优于全局记忆池基线。

**[Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)**

:   本文提出组合引导 token，通过自蒸馏将行为指令压缩为输入空间的嵌入向量，并训练专用组合 token <and> 来捕获"组合"的通用概念，在未见过的行为组合、未见过的行为以及未见过的组合数量上均展现强泛化能力。

**[DeepPrune: Parallel Scaling without Inter-Trace Redundancy](deepprune_parallel_scaling_without_inter-trace_redundancy.md)**

:   本文提出 DeepPrune，通过训练专门的判断模型从部分推理轨迹预测答案等价性，结合在线贪心聚类算法动态剪枝冗余的并行 CoT 路径，在保持竞争准确率（3 个百分点以内）的同时减少 65.73%-88.50% 的 token 消耗。

**[Efficient Learned Data Compression via Dual-Stream Feature Decoupling](efficient_learned_data_compression_via_dual-stream_feature_decoupling.md)**

:   本文提出FADE框架，通过双流多尺度解耦器将微观句法和宏观语义特征分离到并行浅层流中处理（取代深层串行堆叠），结合层次化门控精炼器和并发流并行流水线，在压缩率和吞吐量上同时达到SOTA。

**[Enabling Agents to Communicate Entirely in Latent Space](enabling_agents_to_communicate_entirely_in_latent_space.md)**

:   本文提出 Interlat，一个让 LLM 智能体完全在潜空间中通信的框架——发送方直接传递最后一层隐状态作为"思维"的表示，接收方通过通信适配器解释这些潜空间消息，并通过潜空间推理进一步压缩到仅 8 个 token 同时保持竞争性能，实现高达 24× 的通信加速。

**[Establishing a Scale for Kullback–Leibler Divergence in Language Models Across Various Settings](establishing_a_scale_for_kullback-leibler_divergence_in_language_models_across_v.md)**

:   本文利用对数似然向量将不同架构的语言模型嵌入统一空间，系统测量了预训练、模型规模、随机种子、量化、微调和层间等多种设置下的 KL 散度特征尺度，并发现预训练轨迹在对数似然空间中呈亚扩散行为——尽管权重空间持续漂移，模型输出分布早期即趋于稳定。

**[FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration](fastkv_decoupling_of_context_reduction_and_kv_cache_compression_for_prefill-deco.md)**

:   本文提出 FastKV，通过将上下文缩减（prefill 阶段的 Token-Selective Propagation）与 KV 缓存压缩（decoding 阶段的层级 KV 保留）解耦，在 LLaMA-3.1-8B-Instruct 上实现 prefill 1.82× 和 decoding 2.87× 加速，同时在 LongBench 上精度下降控制在 1% 以内。

**[Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation](find_your_optimal_teacher_personalized_data_synthesis_via_router-guided_multi-te.md)**

:   提出 PerSyn（Personalized data Synthesis），通过"先路由再生成"范式让路由器为每个 prompt 分配最优教师模型，综合考虑学生可学习性和教师响应质量，比传统"先生成再选择"范式高效且效果更好，在指令微调和数学推理两个场景中一致超越所有基线。

**[MAGEO: From Experience to Skill — Multi-Agent Generative Engine Optimization via Reusable Strategy Learning](from_experience_to_skill_multi-agent_generative_engine_optimization_via_reusable.md)**

:   本文将生成引擎优化（GEO）从逐实例启发式优化重构为策略学习问题，提出 MAGEO 多智能体框架——执行层由偏好/规划/编辑/评估四个智能体协作，学习层将验证有效的编辑模式蒸馏为可复用的引擎特定策略技能，并引入 Twin Branch 因果评估协议和 DSV-CF 双轴指标，在三个主流引擎上显著优于启发式基线。

**[From Weights to Activations: Is Steering the Next Frontier of Adaptation?](from_weights_to_activations_is_steering_the_next_frontier_of_adaptation.md)**

:   本文系统性地论证 steering（推理时激活空间干预）应被视为一种独立的模型适配范式，提出八项功能性评估标准对比 steering 与微调、PEFT、提示工程等传统方法，将 steering 定位为基于激活空间的局部可逆行为修改方法，具有计算高效、数据高效和可逆性等独特优势。

**[HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference](heterocache_a_dynamic_retrieval_approach_to_heterogeneous_kv_cache_compression_f.md)**

:   本文提出 HeteroCache，一种免训练的动态 KV 缓存压缩框架，基于注意力头的时间异质性（稳定头 vs 漂移头）和层内冗余性（相似头聚类），实施细粒度的角色分配策略——为漂移头分配更大缓存预算，用代表头稀疏监控注意力漂移触发异步按需检索，在 224K 上下文下实现 3 倍解码加速。

**[IMPACT: Importance-Aware Activation Space Reconstruction](impact_importance-aware_activation_space_reconstruction.md)**

:   提出 IMPACT 框架，将 LLM 低秩压缩从最小化权重重构误差转向最小化重要性加权的激活重构误差，通过将梯度信息融入激活协方差矩阵推导出闭式最优解，实现在保持精度的同时最高减少 55.4% 的模型体积。

**[CadLLM: Improving the Throughput of Diffusion-based LLMs via Training-Free Confidence-Aware Calibration](improving_the_throughput_of_diffusion-based_large_language_models_via_a_training.md)**

:   提出 CadLLM，一种免训练的自适应推理加速方法，利用扩散语言模型（dLLM）的 token 解码置信度信号动态调整块大小、步数、词表采样范围和提交阈值四个维度，在 LLaDA 和 DREAM 上实现 1.1-2.28× 的吞吐量提升且保持竞争性准确率。

**[JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew](judgemenot_personalizing_large_language_models_to_emulate_judicial_reasoning_in_.md)**

:   提出了一个 synthetic-organic 监督管线，将法官的原始判决文书转化为推理指令微调数据，通过 CLM→指令微调的 Chain-of-LoRA 策略实现对个体法官推理风格的高保真模拟，在希伯来语低资源场景下生成内容与真实法官不可区分。

**[LLM Prompt Duel Optimizer: Efficient Label-Free Prompt Optimization](llm_prompt_duel_optimizer_efficient_label-free_prompt_optimization.md)**

:   将无标签提示优化形式化为决斗老虎机（dueling bandit）问题，提出 Prompt Duel Optimizer (PDO)，通过 Double Thompson Sampling 高效选择信息量最大的提示对进行比较，结合 top-performer 变异策略扩展搜索空间，在 BBH 和 MS MARCO 上以更少的 judge 调用次数找到更强提示。

**[LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging](lora_on_the_go_instance-level_dynamic_lora_selection_and_merging.md)**

:   提出 LoGo（LoRA on the Go），一个免训练的框架，通过单次前向传播提取 LoRA 激活信号（范数或熵），在实例级别动态选择和合并最相关的 LoRA 适配器，无需标注数据或额外训练即可实现跨任务泛化。

**[MAESTRO: Meta-learning Adaptive Estimation of Scalarization Trade-offs for Reward Optimization](maestro_meta-learning_adaptive_estimation_of_scalarization_trade-offs_for_reward.md)**

:   本文提出 MAESTRO，将 GRPO 中的奖励标量化重新定义为上下文老虎机问题，通过轻量级 Conductor 网络利用模型末层隐藏状态自适应地为每个 prompt-response 对选择奖励权重，在七个开放域基准上一致超越静态奖励和单一奖励基线。

**[Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation](mem2evolve_towards_self-evolving_agents_via_co-evolutionary_capability_expansion.md)**

:   本文提出 Mem²Evolve，一种通过双记忆机制（资产记忆 + 经验记忆）实现能力扩展与经验蒸馏协同进化的自进化 Agent 框架，在 6 类任务 8 个基准上平均 Pass@1 达 70.24%，分别超过纯经验进化和纯能力进化的最强基线 11.80% 和 6.46%。

**[Mem^p: Exploring Agent Procedural Memory](memp_exploring_agent_procedural_memory.md)**

:   本文提出 Mem^p 框架，系统性地研究如何为 LLM Agent 构建可学习、可更新、终身演化的程序性记忆——通过将过去的任务轨迹蒸馏为细粒度的分步指令和高层脚本抽象，并配合动态更新机制（添加/验证/反思/淘汰），在 TravelPlanner 和 ALFWorld 上实现了成功率持续提升和执行步数大幅减少。

**[Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models](meta-tool_efficient_few-shot_tool_adaptation_for_small_language_models.md)**

:   通过在四个基准上系统对比超网络 LoRA 适应 vs 精心设计的 few-shot 提示，发现 2.28 亿参数的超网络提供零增益——few-shot 示例贡献 +21.5%、文档编码贡献 +5.0%、超网络贡献 0%，3B 模型配合良好提示可达 GPT-5 平均性能的 79.7% 且延迟低 10 倍。

**[Model Internal Sleuthing: Finding Lexical Identity and Inflectional Features in Modern Language Models](model_internal_sleuthing_finding_lexical_identity_and_inflectional_features_in_m.md)**

:   本文系统地对 25 个 Transformer 语言模型（从 BERT Base 到 Qwen2.5-7B）进行探针分析，发现词汇同一性（lexeme）在早期层线性可解码但随深度衰减，而屈折特征（inflection）在所有层中保持稳定可读，且占据紧凑可控的子空间。

**[No-Worse Context-Aware Decoding: Preventing Neutral Regression in Context-Conditioned Generation](no-worse_context-aware_decoding_preventing_neutral_regression_in_context-conditi.md)**

:   提出 NWCAD，一种解码时适配器，通过两阶段门控机制在上下文无信息时精确回退到无上下文解码（防止中性退化），在上下文有帮助时利用上下文进行修正，兼顾"无害"与"有效"两个目标。

**[Polynomial Expansion Rank Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions](polynomial_expansion_rank_adaptation_enhancing_low-rank_fine-tuning_with_high-or.md)**

:   本文提出 PERA（Polynomial Expansion Rank Adaptation），通过在低秩因子的参数空间中引入结构化多项式展开（平方项和交叉项），将 LoRA 的线性适配空间扩展为多项式流形，在不增加秩或推理开销的前提下显著提升权重更新的表达能力，在常识推理和 NLU 任务上一致优于 LoRA/DoRA/HiRA 等方法。

**[Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty](reason_only_when_needed_efficient_generative_reward_modeling_via_model-internal_.md)**

:   提出 E-GRM 框架，利用模型并行解码的收敛行为估计不确定性，仅在必要时触发 CoT 推理，并通过混合损失训练的判别式评分器精细评估推理路径质量，在多个奖励模型基准上实现 SOTA 同时降低 62% 推理延迟。

**[SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)**

:   SAMoRA 通过语义感知路由器和任务自适应缩放机制，解决了现有 MoE-LoRA 方法中路由不精确和权重融合缺乏灵活性的问题，在多任务基准上以最少可训练参数（0.15%）达到 SOTA。

**[SCURank: Ranking Multiple Candidate Summaries with Summary Content Units for Enhanced Summarization](scurank_ranking_multiple_candidate_summaries_with_summary_content_units_for_enha.md)**

:   本文提出 SCURank，一种基于摘要内容单元（SCU）的排序框架，通过提取 SCU、跨摘要聚类估计信息重要性、按信息丰富度评分来排序候选摘要，替代不稳定的 LLM 直接排序和粗粒度的 ROUGE 排序，在多 LLM 蒸馏场景中配合 BRIO 对比学习显著提升了蒸馏模型的摘要性能。

**[SeLaR: Selective Latent Reasoning in Large Language Models](selar_selective_latent_reasoning_in_large_language_models.md)**

:   本文提出 SeLaR，一种轻量级无训练框架，通过熵门控机制仅在模型不确定的"探索步"激活软嵌入潜在推理、在高置信的"确定步"保持离散解码，并引入熵感知对比正则化防止软嵌入向主导 token 坍缩，在五个推理基准上一致超越标准 CoT 和 SOTA 无训练方法。

**[Supplement Generation Training for Enhancing Agentic Task Performance](supplement_generation_training_for_enhancing_agentic_task_performance.md)**

:   SGT（Supplement Generation Training）训练一个小型 LLM（1.7B）生成逐实例的补充文本（推理线索、摘要、错误提醒等），附加到输入后让冻结的大型 Actor 模型更有效地解决任务，在 5 个基准上平均提升 21%，无需修改大模型参数。

**[Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs](task-stratified_knowledge_scaling_laws_for_post-training_quantized_large_languag.md)**

:   本文建立了首个面向后训练量化（PTQ）的任务分层知识缩放定律，将 LLM 能力分为记忆/应用/推理三层，统一建模模型大小、位宽、组大小和校准集大小四个因素，在 293 种 PTQ 配置上验证，揭示推理对精度敏感、应用随规模提升、记忆对校准敏感的差异化规律。

**[Training-Free Test-Time Contrastive Learning for Large Language Models](training-free_test-time_contrastive_learning_for_large_language_models.md)**

:   本文提出 TF-TTCL，一种无需梯度更新的测试时对比学习框架，通过"探索-反思-引导"循环让冻结的 LLM 在线自我改进——用多智能体角色扮演生成多样推理轨迹，从正负样本对比中蒸馏文本规则存入记忆库，推理时检索相关规则引导生成。

**[UKP_Psycontrol at SemEval-2026 Task 2: Modeling Valence and Arousal Dynamics from Text](ukp_psycontrol_at_semeval-2026_task_2_modeling_valence_and_arousal_dynamics_from.md)**

:   UKP_Psycontrol 在 SemEval-2026 Task 2 上取得双项第一，通过结合 LLM 提示、Ising 交互的 MaxEnt 模型和神经回归模型，发现 LLM 擅长捕捉静态情感信号而短期情感变化更多由近期数值轨迹而非文本语义解释。

**[Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment](which_reasoning_trajectories_teach_students_to_reason_better_a_simple_metric_of_.md)**

:   提出 Rank-Surprisal Ratio (RSR) 指标，通过联合衡量推理轨迹对学生模型的"信息量"和"对齐度"来评估训练数据适配性，在 5 个学生模型和 11 个教师模型的组合中与训练后性能达到平均 0.86 的 Spearman 相关性，并成功应用于轨迹选择和教师选择。

**[WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling](wisca_a_lightweight_model_transition_method_to_improve_llm_training_via_weight_s.md)**

:   本文提出等价模型理论和 WISCA 权重缩放策略，通过在训练中动态调整 Transformer 注意力层的 $W_q/W_k$ 和 $W_v/W_o$ 权重使其 L1 范数相等（保持模型输出不变），将优化引导至更平坦的损失最小值区域，在 GQA 架构上实现平均 5.6% 的零样本评估提升和 2.12% 的训练困惑度降低。

**[YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents](yield_a_large-scale_dataset_and_evaluation_framework_for_information_elicitation.md)**

:   提出信息引出代理（IEA）作为新的对话范式，发布了首个大规模（2,281 段对话，26M token）人与人信息引出对话数据集 YIELD，将信息引出形式化为有限视野 POMDP，并设计了专门的评估指标（Conformity、Progression、TLR），实验表明在 YIELD 上微调能显著提升 LLM 与真实引出行为的对齐。
