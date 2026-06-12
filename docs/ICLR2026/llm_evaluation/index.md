---
title: >-
  ICLR2026 LLM评测论文汇总 · 29篇论文解读
description: >-
  29篇ICLR2026的 LLM 评测方向论文解读，涵盖 LLM、扩散模型、推理、多模态、对抗鲁棒、目标跟踪等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "LLM 评测"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "扩散模型"
  - "推理"
  - "多模态"
  - "对抗鲁棒"
  - "目标跟踪"
item_list:
  - u: "adablock-dllm_semantic-aware_diffusion_llm_inference_via_adaptive_block_size/"
    t: "AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size"
  - u: "anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni/"
    t: "AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning"
  - u: "aside_architectural_separation_of_instructions_and_data_in_language_models/"
    t: "ASIDE: Architectural Separation of Instructions and Data in Language Models"
  - u: "astabench_benchmarking_ai_agents/"
    t: "AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite"
  - u: "benchmarking_overton_pluralism_in_llms/"
    t: "Benchmarking Overton Pluralism in LLMs"
  - u: "biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation/"
    t: "BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation"
  - u: "can_vision_language_models_assess_graphic_design_aesthetics_a_benchmark_evaluati/"
    t: "Can Vision–Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective"
  - u: "can_you_hear_me_now_a_benchmark_for_long-range_graph_propagation_and_beyond/"
    t: "Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation and Beyond"
  - u: "dare-bench_evaluating_modeling_and_instruction_fidelity_of_llms_in_data_science/"
    t: "DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science"
  - u: "doubly-robust_llm-as-a-judge_externally_valid_estimation_with_imperfect_personas/"
    t: "Doubly-Robust LLM-as-a-Judge: Externally Valid Estimation with Imperfect Personas"
  - u: "enabling_fine-grained_operating_points_for_black-box_llms/"
    t: "Enabling Fine-Grained Operating Points for Black-Box LLMs"
  - u: "guidedsampling_steering_llms_towards_diverse_candidate_solutions_at_inference-ti/"
    t: "GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time"
  - u: "how_reliable_is_language_model_micro-benchmarking/"
    t: "How Reliable is Language Model Micro-Benchmarking?"
  - u: "human-llm_collaborative_feature_engineering_for_tabular_data/"
    t: "Human-LLM Collaborative Feature Engineering for Tabular Learning"
  - u: "in-context_learning_for_pure_exploration/"
    t: "In-Context Learning for Pure Exploration"
  - u: "in-context_learning_of_temporal_point_processes_with_foundation_inference_models/"
    t: "In-Context Learning of Temporal Point Processes with Foundation Inference Models"
  - u: "log_probability_tracking_of_llm_apis/"
    t: "Log Probability Tracking of LLM APIs"
  - u: "multi-llm_adaptive_conformal_inference_for_reliable_llm_responses/"
    t: "Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses"
  - u: "preference_leakage_a_contamination_problem_in_llm-as-a-judge/"
    t: "Preference Leakage: A Contamination Problem in LLM-as-a-judge"
  - u: "prompt_and_parameter_co-optimization_for_large_language_models/"
    t: "Prompt and Parameter Co-Optimization for Large Language Models"
  - u: "rankllm_weighted_ranking_of_llms_by_quantifying_question_difficulty/"
    t: "RankLLM: Weighted Ranking of LLMs by Quantifying Question Difficulty"
  - u: "same_content_different_representations_a_controlled_study_for_t/"
    t: "Same Content, Different Representations: A Controlled Study for Table QA"
  - u: "simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_agents/"
    t: "SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home Agents"
  - u: "subliminal_signals_in_preference_labels/"
    t: "Subliminal Signals in Preference Labels"
  - u: "talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis/"
    t: "Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis"
  - u: "truthfulness_despite_weak_supervision_evaluating_and_training_llms_using_peer_pr/"
    t: "Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction"
  - u: "unpacking_human_preference_for_llms_demographically_aware_evaluation_of_long-fo/"
    t: "Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework"
  - u: "vcache_verified_semantic_prompt_caching/"
    t: "vCache: Verified Semantic Prompt Caching"
  - u: "when_to_ensemble_identifying_token-level_points_for_stable_and_fast_llm_ensembli/"
    t: "When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling"
item_total: 29
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📊 LLM 评测

**🔬 ICLR2026** · **29** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (24)](../../ICML2026/llm_evaluation/index.md) · [💬 ACL2026 (91)](../../ACL2026/llm_evaluation/index.md) · [🤖 AAAI2026 (16)](../../AAAI2026/llm_evaluation/index.md) · [🧠 NeurIPS2025 (38)](../../NeurIPS2025/llm_evaluation/index.md) · [📹 ICCV2025 (27)](../../ICCV2025/llm_evaluation/index.md) · [🧪 ICML2025 (22)](../../ICML2025/llm_evaluation/index.md)

🔥 **高频主题：** LLM ×9

**[AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size](adablock-dllm_semantic-aware_diffusion_llm_inference_via_adaptive_block_size.md)**

:   通过统计分析扩散语言模型（dLLM）去噪过程中 token 置信度的动态变化，发现"波动带"（Volatility Band）区域编码了文本的局部语义结构，进而提出 AdaBlock-dLLM——一个无训练、即插即用的自适应块大小调度器，让半自回归解码的块边界与语义步骤自然对齐，在相同吞吐量下最高提升 5.3% 准确率。

**[AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)**

:   构建首个面向麻醉学推理的综合数据集套件AnesSuite，包含评测基准AnesBench（7972道三级认知难度双语选择题）和三组训练数据集（AnesCorpus/AnesQA/AnesR1），基于此训练的Morpheus模型通过SFT+GRPO让7B模型追平14B基线，同时揭示了当前最强LLM在复杂临床推理（System 2）上的显著瓶颈。

**[ASIDE: Architectural Separation of Instructions and Data in Language Models](aside_architectural_separation_of_instructions_and_data_in_language_models.md)**

:   提出 ASIDE，一种在 token embedding 层面通过正交旋转区分指令和数据的架构级改造，仅需修改前向传播并在标准指令微调数据上训练，即可显著提升指令-数据分离度和 prompt injection 鲁棒性，无需任何安全专项训练。

**[AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite](astabench_benchmarking_ai_agents.md)**

:   AI2 团队针对现有科研 Agent 基准的 5 大方法学缺陷，构建了首个覆盖科学研究全流程的 Agent 评估套件 AstaBench，包含 4 大类 11 个子基准共 2400+ 问题，配备基于 Semantic Scholar 的生产级可控搜索工具和 9 类科研优化 Asta Agent 基线，对 57 个 Agent（22 类）进行了迄今最大规模的系统评估，发现尽管在文献检索等单项任务上取得了进展，AI 在端到端科学研究辅助方面仍远未达标。

**[Benchmarking Overton Pluralism in LLMs](benchmarking_overton_pluralism_in_llms.md)**

:   提出 OvertonBench 框架，通过大规模人类研究（1208名美国代表性参与者、60个主观问题、8个LLM）将 Overton 多元主义形式化为集合覆盖度指标 OvertonScore，发现当前所有模型得分仅 0.35–0.41（理论上限为 1.0），并构建了与人类判断高度相关（ρ=0.88）的自动化评测工具。

**[BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)**

:   提出 BiasScope，一个完全由 LLM 驱动的迭代式框架，能自动、大规模地发现 LLM-as-a-Judge 中的潜在未知偏差，并基于此构建了更具挑战性的 JudgeBench-Pro 基准，在其上即使强大的 LLM 评估器错误率也超过 50%。

**[Can Vision–Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective](can_vision_language_models_assess_graphic_design_aesthetics_a_benchmark_evaluati.md)**

:   提出 AesEval-Bench，首个系统性评估 VLM 图形设计美学评估能力的 benchmark（4维度×12指标×3任务），发现现有 VLM（含推理增强型）在设计美学上表现有限，并通过 human-guided VLM labeling + indicator-grounded reasoning 构建训练数据，微调 7B 模型在精确定位任务上超过 GPT-5。

**[Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation and Beyond](can_you_hear_me_now_a_benchmark_for_long-range_graph_propagation_and_beyond.md)**

:   本文提出 ECHO 基准，包含 3 个合成任务和 2 个基于密度泛函理论（DFT）的真实化学任务，要求图神经网络在 17–40 跳范围内有效传播信息，系统评估了 11 种 GNN 架构的长程传播能力。

**[DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science](dare-bench_evaluating_modeling_and_instruction_fidelity_of_llms_in_data_science.md)**

:   DARE-bench 是一个面向数据科学任务的大规模可验证基准，包含 6300 个 Kaggle 衍生任务，支持 ML 建模和指令遵循两类评估，提供训练集支持 SFT 和 RL——SFT 将 Qwen3-32B 提升 1.83×，RL 将 Qwen3-4B 提升 8× 以上。

**[Doubly-Robust LLM-as-a-Judge: Externally Valid Estimation with Imperfect Personas](doubly-robust_llm-as-a-judge_externally_valid_estimation_with_imperfect_personas.md)**

:   提出一种 doubly-robust 估计框架，将不完美的 LLM persona 评分与存在采样偏差的人工评分相结合，在协变量偏移和选择偏差同时存在时仍能产生统计有效的 GenAI 系统质量估计。

**[Enabling Fine-Grained Operating Points for Black-Box LLMs](enabling_fine-grained_operating_points_for_black-box_llms.md)**

:   发现黑盒 LLM 的语言化概率仅输出 16-23 个唯一值（低基数问题），导致 PR/ROC 曲线粗糙无法精细调优；通过注入参数化噪声和可选的 MLP 校正，将唯一值从 16 个提升到 20,000+，在仅需 1-2 次 API 调用的条件下达到 20 次采样的性能。

**[GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time](guidedsampling_steering_llms_towards_diverse_candidate_solutions_at_inference-ti.md)**

:   提出 GuidedSampling 推理算法，将重复采样（RS）的隐式探索和生成过程显式解耦为两阶段：先迭代生成多样化的解题概念/定理，再基于各概念分别生成候选解。在 pass@50 上平均提升约 21.6%，微调后 pass@5 提升约 9.7%。

**[How Reliable is Language Model Micro-Benchmarking?](how_reliable_is_language_model_micro-benchmarking.md)**

:   提出 Minimum Detectable Ability Difference (MDAD) 元评估指标，系统揭示了 micro-benchmark 在极小规模下无法可靠区分性能差距小的模型对，且当样本量达到 ~250 时随机采样与精心设计的 micro-benchmark 方法表现相当。

**[Human-LLM Collaborative Feature Engineering for Tabular Learning](human-llm_collaborative_feature_engineering_for_tabular_data.md)**

:   提出一个人-LLM协作特征工程框架，将LLM的特征操作提议与选择过程解耦，通过贝叶斯神经网络建模操作效用和不确定性来指导选择，并选择性地引入人类偏好反馈，在18个表格数据集上平均错误率降低8.96%~11.23%。

**[In-Context Learning for Pure Exploration](in-context_learning_for_pure_exploration.md)**

:   提出 ICPE（In-Context Pure Exploration），一种结合监督学习和强化学习的上下文学习框架，使用 Transformer 从经验中直接学习探索策略，在主动序列假设检验/纯探索问题中实现接近最优的实例自适应算法性能，无需显式建模信息结构。

**[In-Context Learning of Temporal Point Processes with Foundation Inference Models](in-context_learning_of_temporal_point_processes_with_foundation_inference_models.md)**

:   提出 FIM-PP——首个面向标记时间点过程（MTPP）的基础推断模型，在 72K 合成点过程（1440 万事件）上预训练 Transformer 来上下文推断条件强度函数，零样本即可匹配专用模型数小时训练的性能，微调几分钟后在四个真实数据集的多事件预测上全面刷新 SOTA。

**[Log Probability Tracking of LLM APIs](log_probability_tracking_of_llm_apis.md)**

:   提出 Logprob Tracking (LT) 方法，仅用单token输入和单token输出的log概率即可检测LLM API的微小变更（如单步微调），灵敏度比现有方法高2-3个数量级，成本低1000倍。

**[Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses](multi-llm_adaptive_conformal_inference_for_reliable_llm_responses.md)**

:   提出 MACI（Multi-LLM Adaptive Conformal Inference），通过**累积乘积型 conformity score** + **多 LLM 集成**的 factuality 评分 + **组条件校准**，在严格保证用户指定错误率的同时，显著提升 LLM 回复中事实性声明的保留率。

**[Preference Leakage: A Contamination Problem in LLM-as-a-judge](preference_leakage_a_contamination_problem_in_llm-as-a-judge.md)**

:   首次定义并系统研究 LLM-as-a-Judge 中的 **偏好泄漏 (Preference Leakage)** 问题——当合成数据生成器 $M_G$ 与评估器 $M_J$ 存在关联（同模型/继承/同家族）时，评委会对"相关学生模型"产生系统性偏好，同模型场景下 PLS 高达 28.7%（Arena-Hard），且该偏差比自中心偏差更隐蔽、更难检测。

**[Prompt and Parameter Co-Optimization for Large Language Models](prompt_and_parameter_co-optimization_for_large_language_models.md)**

:   提出 MetaTuner 框架，通过共享 meta encoder 同时生成 prompt 和 LoRA 参数，将离散 prompt 优化与连续参数微调统一为端到端可优化的联合框架，在数学推理和问答任务上大幅超越单独优化的方法。

**[RankLLM: Weighted Ranking of LLMs by Quantifying Question Difficulty](rankllm_weighted_ranking_of_llms_by_quantifying_question_difficulty.md)**

:   提出 RankLLM，一个基于有向二部图双向分数传播的非参数化框架，联合估计题目难度和模型能力，实现难度感知的 LLM 排名，与人类判断达到 90% 一致性。

**[Same Content, Different Representations: A Controlled Study for Table QA](same_content_different_representations_a_controlled_study_for_t.md)**

:   首个控制变量研究：在保持表格内容完全相同的条件下变换表示形式（结构化 vs 半结构化），系统评估 NL2SQL、LLM、混合三类方法在不同表格大小/模式质量/查询复杂度下的鲁棒性，发现表示形式是影响 Table QA 性能的一阶因素。

**[SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home Agents](simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_agents.md)**

:   SimuHome 是一个基于 Matter 协议的高保真智能家居仿真器和 600 集评估基准，支持环境变量动态变化和时间加速调度评估，揭示了工作流调度是当前 LLM 代理最持久的挑战。

**[Subliminal Signals in Preference Labels](subliminal_signals_in_preference_labels.md)**

:   证明偏好标签可以作为隐蔽通信通道：即使学生模型生成的是语义无关的数字序列，有偏见的裁判模型仅通过二值偏好标签就能向学生模型传递潜意识行为特征，且这种传递在迭代对齐中会增强。

**[Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis.md)**

:   提出TED(Talk, Evaluate, Diagnose)框架，通过通用可复用的expert/non-expert persona模板实现用户感知的动态Agent评估、grading notes+LLM-as-judge+MaxProgressRate@k等新指标进行细粒度效率评估、自动化错误发现和聚类提供可操作的改进反馈，在τ²-bench和ToolSandbox上揭示新的Agent性能洞察。

**[Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction](truthfulness_despite_weak_supervision_evaluating_and_training_llms_using_peer_pr.md)**

:   提出将博弈论中的 Peer Prediction 机制应用于 LLM 评估和训练，通过衡量参与者答案的互预测性来区分诚实与欺骗回答，无需真值标签即可实现诚实性激励，展现出惊人的"逆向缩放"特性——专家越弱反而越能抵抗强模型的欺骗。

**[Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework](unpacking_human_preference_for_llms_demographically_aware_evaluation_of_long-fo.md)**

:   提出 HUMAINE 框架，通过 23,404 名人口统计分层参与者对 28 个 SOTA 模型进行多维度（5 维）、多轮对话的人类偏好评估，用层次贝叶斯 BTD 模型揭示年龄是偏好异质性的最大驱动因素（平均排名偏移 ±2.8），证明单一聚合排行榜不足以反映不同人群的真实偏好。

**[vCache: Verified Semantic Prompt Caching](vcache_verified_semantic_prompt_caching.md)**

:   提出 vCache——首个具有**用户定义错误率保证**的语义缓存系统，通过在线学习为每个缓存嵌入独立估计最优相似度阈值，无需预训练即可在满足正确性约束下实现最高 12.5× 缓存命中率提升和 26× 错误率降低。

**[When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling](when_to_ensemble_identifying_token-level_points_for_stable_and_fast_llm_ensembli.md)**

:   提出 SAFE（Stable And Fast LLM Ensembling），通过 Generate-Verify-Ensemble 循环在 token 级别选择性地集成多个异构分词器 LLM，解决长序列生成中分词不匹配导致的 OOV-like 污染问题，仅在不到 1% 的 token 上集成即可提升效果，MATH500 上将 UniTE 从 59.6% 提升到 77.4%。
