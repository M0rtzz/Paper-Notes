---
title: >-
  ICML2026 LLM评测论文汇总 · 25篇论文解读
description: >-
  25篇ICML2026的 LLM 评测方向论文解读，涵盖 LLM、强化学习、Agent、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "LLM 评测"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "强化学习"
  - "Agent"
  - "推理"
item_list:
  - u: "agent_world_model_infinity_synthetic_environments_for_agentic_reinforcement_lear/"
    t: "Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning"
  - u: "agzo_activation-guided_zeroth-order_optimization_for_llm_fine-tuning/"
    t: "AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning"
  - u: "bespoke_benchmark_for_search-augmented_large_language_model_personalization_via_/"
    t: "BESPOKE: Benchmark for Search-Augmented Large Language Model Personalization via Diagnostic Feedback"
  - u: "beyond_log_likelihood_probability-based_objectives_for_supervised_fine-tuning_ac/"
    t: "Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum"
  - u: "beyond_trajectory-level_attribution_graph-based_credit_assignment_for_agentic_re/"
    t: "Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for Agentic Reinforcement Learning"
  - u: "capbencher_give_your_llm_benchmark_a_built-in_alarm_for_test-set_overfitting/"
    t: "CapBencher: Give Your LLM Benchmark a Built-in Alarm for Test-Set Overfitting"
  - u: "decompose_structure_and_repair_a_neuro-symbolic_framework_for_autoformalization_/"
    t: "Decompose, Structure, and Repair: A Neuro-Symbolic Framework for Autoformalization via Operator Trees"
  - u: "dei_diversity_in_evolutionary_inference_for_quality-diversity_search/"
    t: "DEI: Diversity in Evolutionary Inference for Quality-Diversity Search"
  - u: "discovering_ordinary_differential_equations_with_llm-based_qualitative_and_quant/"
    t: "Discovering Ordinary Differential Equations with LLM-Based Qualitative and Quantitative Evaluation"
  - u: "from_human-level_ai_tales_to_ai_leveling_human_scales/"
    t: "From Human-Level AI Tales to AI Leveling Human Scales"
  - u: "geometry-aware_decoding_with_wasserstein-regularized_truncation_and_mass_penalti/"
    t: "Top-W: Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for LLMs"
  - u: "hiper_hierarchical_reinforcement_learning_with_explicit_credit_assignment_for_la/"
    t: "HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents"
  - u: "investigating_advanced_reasoning_of_large_language_models_via_black-box_environm/"
    t: "Investigating Advanced Reasoning of Large Language Models via Black-Box Environment Interaction"
  - u: "margin-adaptive_confidence_ranking_for_reliable_llm_judgement/"
    t: "Margin-Adaptive Confidence Ranking for Reliable LLM Judgement"
  - u: "multi2_hierarchical_multi-agent_decision-making_with_llm-based_agents_in_interac/"
    t: "Multi$^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments"
  - u: "on_effectiveness_and_efficiency_of_agentic_tool-calling_and_rl_training/"
    t: "On Effectiveness and Efficiency of Agentic Tool-calling and RL Training"
  - u: "politicsbench_benchmarking_political_values_in_large_language_models_with_multi-/"
    t: "PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Stage Roleplay"
  - u: "real_regression-aware_reinforcement_learning_for_llm-as-a-judge/"
    t: "REAL：把回归感知奖励塞进 RL，让 LLM-as-a-Judge 学会\"差一分也是差\""
  - u: "reasoning_is_not_free_robust_adaptive_cost-efficient_routing_for_llm-as-a-judge/"
    t: "Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge"
  - u: "resolution_diagnostics_for_paired_llm_evaluation/"
    t: "Resolution Diagnostics for Paired LLM Evaluation"
  - u: "spherical_steering_geometry-aware_activation_rotation_for_language_models/"
    t: "Spherical Steering: Geometry-Aware Activation Rotation for Language Models"
  - u: "toward_training_superintelligent_software_agents_through_self-play_swe-rl/"
    t: "Toward Training Superintelligent Software Agents through Self-Play SWE-RL"
  - u: "when_ai_benchmarks_plateau_a_systematic_study_of_benchmark_saturation/"
    t: "When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation"
  - u: "who_can_we_trust_llm-as-a-jury_for_comparative_assessment/"
    t: "Who can we trust? LLM-as-a-jury for Comparative Assessment"
  - u: "whose_alignment_comparing_llm_process_alignment_across_diverse_organizational_de/"
    t: "Whose Alignment? Comparing LLM Process Alignment Across Diverse Organizational Decision Contexts"
item_total: 25
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📊 LLM 评测

**🧪 ICML2026** · **25** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (92)](../../ACL2026/llm_evaluation/index.md) · [🔬 ICLR2026 (29)](../../ICLR2026/llm_evaluation/index.md) · [🤖 AAAI2026 (16)](../../AAAI2026/llm_evaluation/index.md) · [🧠 NeurIPS2025 (38)](../../NeurIPS2025/llm_evaluation/index.md) · [📹 ICCV2025 (27)](../../ICCV2025/llm_evaluation/index.md) · [🧪 ICML2025 (22)](../../ICML2025/llm_evaluation/index.md)

🔥 **高频主题：** LLM ×14 · 强化学习 ×3 · Agent ×2 · 推理 ×2

**[Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning](agent_world_model_infinity_synthetic_environments_for_agentic_reinforcement_lear.md)**

:   本文提出 Agent World Model，一条从场景、任务、数据库、MCP 工具接口到验证器的全合成流水线，生成 1000 个可执行数据库驱动环境，并用它们训练工具调用 Agent，在 BFCLv3、$\tau^2$-bench 和 MCP-Universe 上取得更强的域外泛化。

**[AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning](agzo_activation-guided_zeroth-order_optimization_for_llm_fine-tuning.md)**

:   AGZO 发现线性层梯度行空间受前向激活子空间约束，并据此在零阶微调中只沿激活引导的低秩方向扰动参数，从而在几乎保持 MeZO 级别显存占用的同时提升梯度对齐和下游任务性能。

**[BESPOKE: Benchmark for Search-Augmented Large Language Model Personalization via Diagnostic Feedback](bespoke_benchmark_for_search-augmented_large_language_model_personalization_via_.md)**

:   提出 Bespoke 基准，通过 30 名标注者 3 周的真实聊天+搜索历史收集 2,870 个会话，构建包含细粒度偏好评分与诊断反馈的评测框架，系统评估搜索增强 LLM 的个性化能力，发现当前模型在所有配置下平均得分均不超过 60，个性化瓶颈在于历史推理而非生成。

**[Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum](beyond_log_likelihood_probability-based_objectives_for_supervised_fine-tuning_ac.md)**

:   本文系统研究了 SFT 中概率类目标函数的行为规律，发现标准 NLL 并非普适最优：在模型先验强的任务上先验倾向（prior-leaning）目标如 $-p$ 显著优于 NLL（最高提升 16%），而在先验弱的任务上 NLL 仍然占优，揭示了由模型能力连续谱（model-capability continuum）主导的目标函数选择原则。

**[Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for Agentic Reinforcement Learning](beyond_trajectory-level_attribution_graph-based_credit_assignment_for_agentic_re.md)**

:   提出 GraphGPO，将所有 rollout 轨迹聚合为统一的状态转移图，利用图上全局最短路径信息为每一步计算基于距离的 advantage，实现比轨迹级归因更精细的信用分配，在 ALFWorld、WebShop、Sokoban 上显著超越 GRPO 和 GiGPO。

**[CapBencher: Give Your LLM Benchmark a Built-in Alarm for Test-Set Overfitting](capbencher_give_your_llm_benchmark_a_built-in_alarm_for_test-set_overfitting.md)**

:   CapBencher 通过为每道题注入随机性（生成多个逻辑正确答案并随机选一个作为标准答案），将 benchmark 的 Bayes 准确率降到可控水平（如 50%），从而在公开发布 benchmark 的同时实现数据污染的黑盒统计检测——任何准确率显著超过 Bayes 上界的模型都被判定为存在污染。

**[Decompose, Structure, and Repair: A Neuro-Symbolic Framework for Autoformalization via Operator Trees](decompose_structure_and_repair_a_neuro-symbolic_framework_for_autoformalization_.md)**

:   本文提出 DSR（Decompose-Structure-Repair）神经符号框架，把自然语言定理形式化拆解为「分解 NL 成分 → 联合生成 FL 成分与算子树（OPT） → 基于子树定位的层级化修复」三阶段，在 ProverBench / ProofNet / PRIME 上以 7B 模型刷新 SOTA，并配套发布 156 题的研究生级 Lean 4 基准 PRIME。

**[DEI: Diversity in Evolutionary Inference for Quality-Diversity Search](dei_diversity_in_evolutionary_inference_for_quality-diversity_search.md)**

:   本文提出 DEI，把多个**不同家族的 LLM**当作 Quality-Diversity 搜索里的异构变异算子分布到不同节点，用全异步 gossip 互相广播每轮 champion 形成跨模型对抗压力，在 Core War 程序合成任务上以等总算力换来比单节点 +124% 的 QD-Score 与 +28% 的 archive coverage。

**[Discovering Ordinary Differential Equations with LLM-Based Qualitative and Quantitative Evaluation](discovering_ordinary_differential_equations_with_llm-based_qualitative_and_quant.md)**

:   DoLQ 在 LLM 符号回归的搜索循环里插入一个 "Scientist Agent"，对候选项同时做定性（物理合理性）+ 定量（消融式 MSE 贡献）评估，把 LLM-SR 那种 "低误差但项数臃肿、物理上荒谬" 的方程逼到既数值精确又结构紧凑。

**[From Human-Level AI Tales to AI Leveling Human Scales](from_human-level_ai_tales_to_ai_leveling_human_scales.md)**

:   本文用 LLM 当人口外推器，把 18 个能力维度按"全世界人口正确率"对数刻度 $L=-\log_B p_W$ 校准，并发现 Volume / Attention 维度真实 base $B \gg 10$、Comprehension 维度 $B \approx 1$，揭示现行 AI 与人类的比较其实严重失调。

**[Top-W: Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for LLMs](geometry-aware_decoding_with_wasserstein-regularized_truncation_and_mass_penalti.md)**

:   Top-W 把 next-token 截断写成"考虑 token embedding 几何的 Wasserstein-熵-质量"三项最小化问题，理论证明最优解要么是单 token、要么是按 $f(i)+\lambda\log p_i$ 排序的前缀，工程实现只是 $O(n\log n)$ 的扫描；在 GSM8K、GPQA、AlpacaEval、MT-Bench 上 15 个 (T, model) 组合多数胜出，高温下 GSM8K 比 Top-H 最多再提 33.7%。

**[HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents](hiper_hierarchical_reinforcement_learning_with_explicit_credit_assignment_for_la.md)**

:   HiPER 把 LLM agent 的扁平 RL 改造成"高层规划子目标 + 低层执行原子动作"的两级 Plan-Execute 结构，并配套提出 Hierarchical Advantage Estimation (HAE) 把 GAE 沿子目标段切开做有界差分耦合的优势估计，在 ALFWorld / WebShop 上分别拿到 97.4% / 83.3% 成功率（Qwen2.5-7B），相对最强基线 GiGPO 提 +6.6% 与 +8.3%。

**[Investigating Advanced Reasoning of Large Language Models via Black-Box Environment Interaction](investigating_advanced_reasoning_of_large_language_models_via_black-box_environm.md)**

:   本文提出「黑盒环境交互」作为评估 LLM 集成式推理（演绎+归纳+溯因）的新范式，构建含 6 类任务 96 个环境的 ORACLE 基准，benchmark 19 个 LLM 后发现：即便最强的 o3 也只能在简单环境拿 70% 准确率、难环境跌到 40%，且所有 LLM 都缺乏「根据反馈自适应优化探索策略」的高层规划能力。

**[Margin-Adaptive Confidence Ranking for Reliable LLM Judgement](margin-adaptive_confidence_ranking_for_reliable_llm_judgement.md)**

:   本文针对 LLM-as-a-judge 中"置信度高就一定靠谱"这一常被违反的单调性假设，提出用一个小 MLP 把多组 in-context 预测概率映射成置信度，并通过 margin-based ranking loss + PAC-Bayes 泛化界推导出一个 margin 自适应训练策略，使学到的置信度在四个数据集与六个 judge 模型上都获得更低的 ranking loss、更高的 AUROC，并显著提升 fixed-sequence 测试的目标一致性达成率。

**[Multi$^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments](multi2_hierarchical_multi-agent_decision-making_with_llm-based_agents_in_interac.md)**

:   本文提出 Multi$^2$ 框架，把 LLM agent 的"规划"与"执行"显式拆成 System 1（SFT 训练的子目标规划器）和 System 2（offline-to-online RL 训练的原子动作执行器），通过角色专属 LoRA 适配器和带策略锚定/KL 正则的训练目标，在 ScienceWorld、ALFWorld、TextCraft 三个长时序交互环境中显著缓解了 objective drift 并提升了 token 效率。

**[On Effectiveness and Efficiency of Agentic Tool-calling and RL Training](on_effectiveness_and_efficiency_of_agentic_tool-calling_and_rl_training.md)**

:   作者从「评测有效性」和「训练效率」两条主线系统审视 LLM 工具调用：一方面用 BFCL 作为案例证明随机种子、多轮模板、思考历史、系统提示等"小细节"能让排行榜分数大幅漂移，使跨论文比较不可靠；另一方面定位 RL（GRPO）训练中 rollout 和 policy update 两个阶段的浪费，提出"在线预 rollout 过滤 + 最大方差 rollout 下采样"两件套，在单轮/多轮工具调用上实现 1.7× 和 2.6× 端到端加速且性能不降。

**[PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Stage Roleplay](politicsbench_benchmarking_political_values_in_large_language_models_with_multi-.md)**

:   PoliticsBench 是基于**多阶段角色扮演**的新型基准——通过 20 个政治情景和 4 阶段交互评测 LLM 的政治价值观表达，发现 7 个主流 LLM 都呈左倾（19-39 分），唯有 Grok 右倾（-22.7）但波动性最大；**情景提示比直接提问更能激发模型的价值观维度**（特征激活 +0.48、承诺度 +1.39）。

**[REAL：把回归感知奖励塞进 RL，让 LLM-as-a-Judge 学会"差一分也是差"](real_regression-aware_reinforcement_learning_for_llm-as-a-judge.md)**

:   针对 LLM 充当评分器（LLM-as-a-Judge）时 RL 用 0/1 二值奖励忽视序数结构的本质缺陷，作者把 RAFT 的"期望值预测 + 平方误差"塞进 RL 目标，因为奖励此时显式依赖策略参数，所以改用广义策略梯度——它干净地拆成"CoT 探索项 + 预测精修项"两部分；在 8B–32B 多基座上相对 SFT/标准 RL 全面胜出，Qwen3-32B 上 Pearson/Spearman 相对 SFT 提 8.4/7.2 点。

**[Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge](reasoning_is_not_free_robust_adaptive_cost-efficient_routing_for_llm-as-a-judge.md)**

:   RACER 把"对每个 query 决定要不要调用 reasoning 模式做 judge"建模为带 KL 不确定集的分布鲁棒约束优化问题，用 primal-dual 算法解出 OOD 下仍满足 cost 预算的最优路由策略，并首次给出 LLM 路由器策略的 linear convergence 理论保证。

**[Resolution Diagnostics for Paired LLM Evaluation](resolution_diagnostics_for_paired_llm_evaluation.md)**

:   本文把 LLM 排行榜上"A 比 B 高 0.X pp"的排名当作配对假设检验问题,通过反演 level-α / power-(1-β) 检验定义"分辨率比" $q=N/N^\star$,并证明常用计算器把单臂 Cohen-$h$ 公式乘 $(1-\rho)$ 这种捷径在小效应下会系统性低估所需样本量一倍,实测发现 Open LLM Leaderboard v1 有 11/40 对、MMLU-Pro top-10 相邻对有 4/9 在 $(\alpha,1-\beta)=(0.05,0.8)$ 下根本"分辨不出来",再叠加多重比较、真实学科聚类、anytime-valid 后这个数还会涨到 6/9。

**[Spherical Steering: Geometry-Aware Activation Rotation for Language Models](spherical_steering_geometry-aware_activation_rotation_for_language_models.md)**

:   本文提出 Spherical Steering：在 LLM 隐藏状态的单位超球面上，沿测地线把激活向量旋转到由对比样本估计出的"真实性方向"，而不是像传统 activation addition 那样做线性加法，从而在保持激活幅值（norm）的同时显著提升 TruthfulQA / COPA / StoryCloze 等基准的多选准确率（+10% 量级），且不损伤开放式生成质量。

**[Toward Training Superintelligent Software Agents through Self-Play SWE-RL](toward_training_superintelligent_software_agents_through_self-play_swe-rl.md)**

:   本文提出 Self-play SWE-RL (SSR)，让同一个 LLM 在沙箱化代码仓里既扮演"造 bug 的 proposer"又扮演"修 bug 的 solver"，仅以 Docker 镜像为输入、用一致性校验和 solve-rate 作奖励做联合 RL，在 SWE-bench Verified 与 SWE-Bench Pro 上分别自我提升 +10.4 / +7.8 分，并稳定优于使用人类标注 issue + 测试套件的"human-data"基线。

**[When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation](when_ai_benchmarks_plateau_a_systematic_study_of_benchmark_saturation.md)**

:   这篇论文把 AI benchmark 饱和定义为前沿模型之间失去可靠区分度，提出基于 leaderboard 不确定性的 saturation index，并分析 60 个文本 LLM benchmark，发现近一半已高饱和，年龄和测试集规模比私有测试集、开放式输出或模板多样性更能解释饱和。

**[Who can we trust? LLM-as-a-jury for Comparative Assessment](who_can_we_trust_llm-as-a-jury_for_comparative_assessment.md)**

:   这篇论文指出多个 LLM 评审在成对比较中可靠性差异很大，提出带评审判别参数的 BT-σ 模型，在没有人工校准标签的情况下同时学习候选输出排名和每个 LLM judge 的可靠性，从而比简单平均和标准 Bradley-Terry 聚合更接近人类排序。

**[Whose Alignment? Comparing LLM Process Alignment Across Diverse Organizational Decision Contexts](whose_alignment_comparing_llm_process_alignment_across_diverse_organizational_de.md)**

:   这篇论文提出 CALM 来评估 LLM 是否按组织真实决策过程而不只是输出结果对齐，并通过 ECHR 法律裁判与 German Credit 信贷决策的对比说明：在规范稳定的领域过程对齐能预测准确率，而在价值争议领域，高过程对齐既难实现也未必应该追求。
