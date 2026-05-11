---
title: >-
  ICLR2026 LLM / NLP方向39篇论文解读
description: >-
  39篇ICLR2026的 LLM / NLP 方向论文解读，涵盖 LLM、推理、扩散模型、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM / NLP

**🔬 ICLR2026** · **39** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (36)](../../ACL2026/llm_nlp/index.md) · [📷 CVPR2026 (9)](../../CVPR2026/llm_nlp/index.md) · [🤖 AAAI2026 (38)](../../AAAI2026/llm_nlp/index.md) · [🧠 NeurIPS2025 (53)](../../NeurIPS2025/llm_nlp/index.md) · [📹 ICCV2025 (8)](../../ICCV2025/llm_nlp/index.md) · [🧪 ICML2025 (28)](../../ICML2025/llm_nlp/index.md)

🔥 **高频主题：** LLM ×11 · 推理 ×5 · 扩散模型 ×5 · Agent ×2

**[AP-OOD: Attention Pooling for Out-of-Distribution Detection](ap-ood_attention_pooling_for_out-of-distribution_detection.md)**

:   提出AP-OOD，将Mahalanobis距离的均值池化替换为可学习的注意力池化，解决了均值池化丢失token级异常信息的问题，在文本OOD检测中将XSUM摘要的FPR95从27.84%降至4.67%，支持无监督到半监督的平滑过渡。

**[AssetFormer: Modular 3D Assets Generation with Autoregressive Transformer](assetformer_modular_3d_assets_generation_with_autoregressive_transformer.md)**

:   提出 AssetFormer，基于 Llama 架构的自回归 Transformer，将模块化 3D 资产（由 primitive 序列组成）建模为离散 token 序列，通过 DFS/BFS 图遍历重排序和联合词汇表解码实现从文本描述生成可直接用于游戏引擎的模块化 3D 资产。

**[BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning](bots_a_unified_framework_for_bayesian_online_task_selection_in_llm_reinforcement.md)**

:   提出 BOTS——一个基于贝叶斯推断的在线任务选择统一框架，在 LLM 强化微调中通过融合显式证据（直接评估的历史通过率）和隐式证据（利用参考模型插值推断的未评估任务难度），配合 Thompson 采样实现探索-利用平衡，在数学/代码/逻辑任务上以仅 0.2% 的额外开销带来最高 50% 的训练加速。

**[Compositional-ARC: Assessing Systematic Generalization in Abstract Spatial Reasoning](compositional-arc_assessing_systematic_generalization_in_abstract_spatial_reason.md)**

:   提出 Compositional-ARC 数据集评估模型在抽象空间推理中的系统性泛化能力——从已知基础几何变换（如平移、旋转）泛化到未见过的变换组合。一个仅 5.7M 参数的 MLC 训练的 encoder-decoder 模型在系统性任务上达到 78.26%，与 ARC Prize 2024 冠军的 8B 模型+TTT 持平，远超 GPT-4o、o3-mini 等（<3%）。

**[d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching](d2cache_accelerating_diffusion-based_llms_via_dual_adaptive_caching.md)**

:   提出 d²Cache，一种面向 Diffusion-based LLM（dLLM）的无训练近似 KV 缓存框架，通过确定性先验引导的 masked token 选择 + 注意力感知的非 mask token 选择两阶段策略，实现 4.1× 推理加速同时提升生成质量。

**[DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas](dreamon_diffusion_language_models_for_code_infilling_beyond_fixed-size_canvas.md)**

:   DreamOn 通过引入 [expand] 和 [delete] 两个特殊状态解决了扩散语言模型（DLM）的固定长度生成限制，无需架构修改即可实现变长代码填充，在 HumanEval-Infilling 上比扩散基线平均提升 26.4%，达到与 SOTA 自回归模型持平的水平。

**[ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework](ellmob_event-driven_human_mobility_generation_with_self-aligned_language_models.md)**

:   提出 ELLMob 框架，基于认知心理学的模糊痕迹理论（FTT），通过提取并迭代对齐"习惯 gist"和"事件 gist"来调和用户日常模式与社会事件约束之间的竞争，实现事件驱动的可解释轨迹生成。

**[Enhancing Persona Following at Decoding Time via Dynamic Importance-Guided Token Estimation for Role-Playing Agents](enhancing_persona_following_at_decoding_time_via_dynamic_importance-guided_token.md)**

:   提出 Persona Dynamic Decoding (PDD) 框架，通过条件互信息动态估计人格属性的场景依赖重要性，并将重要性分数整合到多目标奖励引导解码中，实现无需微调的推理时人格跟随。

**[Evaluating Text Creativity across Diverse Domains: A Dataset and Large Language Model Evaluator](evaluating_text_creativity_across_diverse_domains_a_dataset_and_large_language_m.md)**

:   提出基于上下文感知的成对比较框架来评估文本创造力，构建了包含 100K+ 人类级别和 1M+ 合成数据的 CreataSet 数据集，训练出 CrEval 评估器，在与人类判断的对齐度上超越 GPT-4o 达 18.7%。

**[Fine-Grained Activation Steering: Steering Less, Achieving More](fine-grained_activation_steering_steering_less_achieving_more.md)**

:   AUSteer 发现块级激活转向（steering）本质上是异质的——不同维度控制不同 token 分布，混合转向既放大有益信号也放大有害信号。提出原子单元（AU）级细粒度转向：用激活动量定位判别性维度，自适应调节转向强度，仅转向 ≤100 个维度即大幅超越转向数千维度的 SOTA 方法。

**[First is Not Really Better Than Last: Evaluating Layer Choice and Aggregation Strategies in Language Model Data Influence Estimation](first_is_not_really_better_than_last_evaluating_layer_choice_and_aggregation_str.md)**

:   通过理论和实验证明先前工作所推崇的"第一层（embedding）最适合做 influence estimation"的结论是不可靠的，发现中间 attention 层才是更好的估计层，并提出 Rank 和 Vote 两种新的跨层聚合策略以及 Noise Detection Rate (NDR) proxy 指标，显著改善了 LLM 中有害训练样本的检测效果。

**[From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning](from_assumptions_to_actions_turning_llm_reasoning_into_uncertainty-aware_plannin.md)**

:   提出 PCE（Planner-Composer-Evaluator）框架，将 LLM 推理链中隐含的环境假设显式提取并组织为决策树，通过似然度-增益-成本评分实现不确定性感知的行动选择，大幅减少多智能体协作中的通信开销。

**[FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model](fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m.md)**

:   提出 FS-DFM（Few-Step Discrete Flow-Matching），通过步数感知训练和累积标量更新规则，将离散 flow-matching 语言模型的采样步数从 1024 步降低到 8 步，实现 128 倍加速，同时保持相当的困惑度和生成质量。

**[Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition](function_induction_and_task_generalization_an_interpretability_study_with_off-by.md)**

:   通过 off-by-one addition（如 1+1=3, 2+2=5）这一反事实任务，利用 path patching 发现大语言模型内部存在 **function induction** 机制——一种超越 token 级别 pattern matching、在函数级别进行归纳推理的注意力头电路，并证明该机制可跨任务复用。

**[GASP: Guided Asymmetric Self-Play For Coding LLMs](gasp_guided_asymmetric_self-play_for_coding_llms.md)**

:   提出GASP框架，在非对称自博弈中引入"goalpost"（硬目标题）引导教师生成有针对性的训练问题，通过lemma（简化变体）→lift（加难变体）的课程结构逐步逼近困难目标，在LiveCodeBench上超越无引导自博弈2.5%且解决了所有baseline无法解决的难题。

**[How Catastrophic is Your LLM? Certifying Risk in Conversation](how_catastrophic_is_your_llm_certifying_risk_in_conversation.md)**

:   提出 C3LLM（Certification of Catastrophic risks in multi-turn Conversation for LLMs），首个为多轮 LLM 对话中灾难性风险提供统计认证的框架：用语义相似度图上的 Markov 过程建模对话分布，定义 3 种对话采样策略 + 增强层，使用 Clopper-Pearson 95% 置信区间认证模型产生有害输出的概率界——发现最差模型风险下界高达 72%。

**[How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use](how_far_are_llms_from_professional_poker_players_revisiting_game-theoretic_reaso.md)**

:   系统分析了 LLM 在扑克中的三大推理缺陷（启发式推理、事实误解、知行差距），提出 ToolPoker 框架——首个面向不完全信息博弈的工具集成 LLM 推理系统，通过外部 CFR solver 提供博弈论最优的行动指导，使 7B 模型在 Limit Hold'em 中逼近 Nash 均衡。

**[Is the Reversal Curse a Binding Problem? Uncovering Limitations of Transformers from a Basic Generalization Failure](is_the_reversal_curse_a_binding_problem_uncovering_limitations_of_transformers_f.md)**

:   提出反转诅咒（Reversal Curse）是认知科学中"绑定问题"在Transformer中的表现——源于概念表示的不一致性和纠缠性，并首次设计出基于JEPA和记忆层的架构真正突破反转诅咒（非绕过）。

**[KVComm: Enabling Efficient LLM Communication through Selective KV Sharing](kvcomm_enabling_efficient_llm_communication_through_selective_kv_sharing.md)**

:   提出 KVComm 框架通过选择性共享 KV pairs 实现 LLM 间高效通信，发现 hidden states 存在"信息集中偏差"使其不适合跨模型传递，设计基于注意力重要性 + 高斯先验的层选择策略，仅传输 30% 层即可超越大多数 baseline。

**[LLEMA: Evolutionary Search with LLMs for Multi-Objective Materials Discovery](llema_evolutionary_search_with_llms_for_multi-objective_material_design.md)**

:   提出 LLEMA 框架，将 LLM 的科学知识与化学规则引导的进化搜索和记忆驱动的迭代优化相结合，在 14 个多目标材料发现任务上实现了更高的命中率、稳定性和 Pareto 前沿质量。

**[Meta-RL Induces Exploration in Language Agents](meta-rl_induces_exploration_in_language_agents.md)**

:   提出 LaMer 框架，将元强化学习（Meta-RL）引入 LLM agent 训练，通过跨 episode 的奖励优化和基于反思的上下文策略适应，使语言智能体学会主动探索环境，在 Sokoban/MineSweeper/Webshop 上分别获得 11%/14%/19% 的绝对性能提升。

**[Near-Optimal Online Deployment and Routing for Streaming LLMs](near-optimal_online_deployment_and_routing_for_streaming_llms.md)**

:   首次形式化 LLM 流式在线部署+路由联合问题：新模型持续出现、旧模型可能过时，在并发部署上限 $M_{\max}$ 和成本预算约束下，提出 StageRoute 分层算法，证明 $\tilde{\mathcal{O}}(T^{2/3})$ 遗憾界并给出匹配下界，达到近最优。

**[Neural Synchrony Between Socially Interacting Language Models](neural_synchrony_between_socially_interacting_language_models.md)**

:   首次研究社会交互中 LLM 间的神经同步现象：通过训练仿射变换预测交互伙伴的未来表征，定义 $SyncR^2$ 指标量化同步强度，发现该同步依赖于社会参与和时间邻近性，且与 LLM 的社会行为表现高度相关（Pearson $r$ = 0.88-0.99），呼应了人类脑间同步（IBS）的神经科学发现。

**[Optimas: Optimizing Compound AI Systems with Globally Aligned Local Rewards](optimas_optimizing_compound_ai_systems_with_globally_aligned_local_rewards.md)**

:   提出 Optimas 框架，为复合 AI 系统中每个组件维护一个与全局奖励对齐的局部奖励函数（LRF），使异构组件（prompt、模型参数、超参数、模型选择）可独立优化，在五个真实系统上平均提升 11.92%。

**[Predicting LLM Reasoning Performance with Small Proxy Models](predicting_llm_reasoning_performance_with_small_proxy_models.md)**

:   提出 rBridge 方法，通过结合前沿模型推理轨迹 (reasoning trace) 的 NLL 评估与 token 级任务对齐权重，使 ≤1B 的小模型能有效预测 13B-32B 大模型的推理性能，数据排序计算成本降低 100 倍以上。

**[PT2-LLM: Post-Training Ternarization for Large Language Models](pt2-llm_post-training_ternarization_for_large_language_models.md)**

:   提出 PT2-LLM，首个针对 LLM 的后训练三值化框架，通过非对称三值量化器（含迭代三值拟合和激活感知网格对齐）与结构相似性重排序策略，在 1.58-bit 下实现优于 2-bit PTQ 方法的性能。

**[ConflictScope: Generative Value Conflicts Reveal LLM Priorities](quamo_quaternion_motions_for_vision-based_3d_human_kinematics_capture.md)**

:   提出ConflictScope——自动化价值冲突场景生成与评估流水线：给定任意价值集合，自动生成价值对之间的冲突场景，通过模拟用户的开放式交互（而非选择题）评估LLM的价值优先级排序；发现模型在开放式评估中从"保护性价值"（如无害性）显著转向"个人价值"（如用户自主性），系统提示可使对齐目标排序提升14%。

**[Rethinking Code Similarity for Automated Algorithm Design with LLMs](rethinking_code_similarity_for_automated_algorithm_design_with_llms.md)**

:   提出 BehaveSim，一种基于"问题求解轨迹"（PSTrajs）和动态时间规整（DTW）的算法相似度度量方法，从执行行为层面而非语法或输出层面衡量算法差异，集成到 FunSearch/EoH 等 LLM-AAD 框架后显著提升性能。

**[Rethinking Uncertainty Estimation in LLMs: A Principled Single-Sequence Measure](rethinking_uncertainty_estimation_in_llms_a_principled_single-sequence_measure.md)**

:   从 proper scoring rules 框架出发，证明最高概率输出序列的负对数似然（MSP）是理论上合理的不确定性度量，并提出 G-NLL——仅用一次贪心解码就能逼近该度量，在多个场景下匹配或超越需要多次采样的 SOTA 方法。

**[Statistical Advantage of Softmax Attention: Insights from Single-Location Regression](statistical_advantage_of_softmax_attention_insights_from_single-location_regress.md)**

:   通过提出"单位置回归"(Single-Location Regression, SLR) 理论框架，结合统计物理中的 order parameter 方法，在高维极限下严格证明了 softmax attention 在种群层面达到 Bayes 风险而线性 attention 本质上无法做到，并在有限样本情形下证实 softmax 始终优于线性 attention，为 softmax 在检索任务中的优势提供了首个原理性解释。

**[Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding](stopping_computation_for_converged_tokens_in_masked_diffusion-lm_decoding.md)**

:   提出 SureLock，当 Masked Diffusion LM 中已 unmask 的 token 后验分布稳定后永久锁定该位置（跳过 Q 投影和 FFN，缓存 KV），将每步注意力计算从 $O(N^2d)$ 降为 $O(MNd)$，在 LLaDA-8B 上减少 30-50% FLOPs 且不损生成质量。

**[The Lattice Representation Hypothesis of Large Language Models](the_lattice_representation_hypothesis_of_large_language_models.md)**

:   提出 LLM 的**格表示假说 (Lattice Representation Hypothesis)**：通过将线性表示假说与形式概念分析 (FCA) 统一，证明 LLM 嵌入空间中的属性方向通过半空间交集隐式编码了一个**概念格 (concept lattice)**，从而实现了连续几何与符号抽象之间的桥接。

**[The Path of Least Resistance: Guiding LLM Reasoning Trajectories for Efficient Consistency](the_path_of_least_resistance_guiding_llm_reasoning_trajectories_for_efficient_co.md)**

:   提出 PoLR（Path of Least Resistance），首个利用推理前缀一致性的推理时方法，通过聚类短前缀并仅扩展主导聚类来实现 Self-Consistency 的高效替代，可减少高达 60% token 使用和 50% 延迟。

**[Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)**

:   揭示了掩码扩散语言模型（MDLM）中的"启动漏洞"（priming vulnerability）——在去噪中间步骤注入肯定性 token 可绕过安全防线，并提出 Recovery Alignment（RA）方法训练模型从被污染的中间状态恢复到安全响应。

**[Trapped by simplicity: When Transformers fail to learn from noisy features](trapped_by_simplicity_when_transformers_fail_to_learn_from_noisy_features.md)**

:   研究表明 Transformer 在从含特征噪声的数据中学习布尔函数时会失败——其简单性偏好（倾向学习低敏感度函数）导致模型被困在比目标函数更简单的最优噪声预测器上，无法恢复真实的无噪声目标函数。

**[Unsupervised Evaluation of Multi-Turn Objective-Driven Interactions](unsupervised_evaluation_of_multi-turn_objective-driven_interactions.md)**

:   提出三种**无监督**指标——LLM 引导聚类（目标识别）、基于微调完成模型的交互完整性检测、响应树（LLM 不确定性量化）——用于评估多轮目标驱动对话，无需标注数据或 LLM-as-a-judge，仅用 8B 模型即可匹配/超越 70B judge 的性能。

**[WebDevJudge: Evaluating (M)LLMs as Critiques for Web Development Quality](webdevjudge_mllm_web_development.md)**

:   构建 WebDevJudge 元评估基准，系统评估 LLM/MLLM 及智能体工作流在 Web 开发质量评估任务上作为裁判的能力，发现当前最强模型与人类专家之间仍存在约15%的一致率差距，并揭示了功能等价识别失败和可行性验证薄弱两大根本瓶颈。

**[Weight Decay may matter more than μP for Learning Rate Transfer in Practice](weight_decay_may_matter_more_than_mup_for_learning_rate_transfer_in_practice.md)**

:   本文通过大规模实证分析表明，μP 的核心对齐假设仅在训练初期短暂成立，实际训练中是独立权重衰减（independent weight decay）而非 μP 在正确稳定跨宽度的特征学习动态，μP 的实际益处可被解释为一种隐式学习率预热。

**[When Stability Fails: Hidden Failure Modes of LLMs in Data-Constrained Scientific Decision-Making](when_stability_fails_hidden_failure_modes_of_llms_in_data-constrained_scientific.md)**

:   通过控制性行为评估框架，揭示 LLM 在数据约束的科学决策任务中的四种隐藏失败模式：高稳定性≠正确性、prompt 措辞敏感性、放宽阈值下的过度选择、以及幻觉产生无效标识符。
