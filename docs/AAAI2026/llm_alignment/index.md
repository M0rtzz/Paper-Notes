---
title: >-
  AAAI2026 对齐/RLHF方向 19篇论文解读
description: >-
  19篇AAAI2026 对齐/RLHF方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐/RLHF

**🤖 AAAI2026** · **19** 篇论文解读

**[Align to Structure: Aligning Large Language Models with Structural Information](align_to_structure_aligning_large_language_models_with_struc.md)**

:   提出 Structural Alignment 方法，通过将语言学篇章结构框架（表层文本结构评分 + 基于RST的篇章motif分类器）融入PPO强化学习训练，并设计基于篇章motif的密集奖励机制，使LLM生成更连贯、更具人类写作风格的长文本，在论文写作和长文档摘要任务上均优于标准RLHF模型。

**[AlignTree: Efficient Defense Against LLM Jailbreak Attacks](aligntree_efficient_defense_against_llm_jailbreak_attacks.md)**

:   AlignTree 利用 LLM 内部激活特征（线性 refusal direction + 非线性 SVM 信号）训练轻量级随机森林分类器，在几乎不增加计算开销的情况下高效检测越狱攻击，实现了 SOTA 的攻击成功率（ASR）降低效果。

**[AMaPO: Adaptive Margin-attached Preference Optimization for Language Model Alignment](amapo_adaptive_margin-attached_preference_optimization_for_l.md)**

:   提出AMaPO算法，通过实例级自适应margin（结合Z-normalization和指数缩放）动态调节梯度幅度，解决DPO等离线偏好优化方法中对已正确排序样本过拟合、对错误排序样本欠拟合的核心矛盾，显著提升排序准确率和下游对齐性能。

**[BiasJailbreak: Analyzing Ethical Biases and Jailbreak Vulnerabilities in Large Language Models](biasjailbreakanalyzing_ethical_biases_and_jailbreak_vulnerabilities_in_large_lan.md)**

:   揭示LLM安全对齐中引入的伦理偏见可被反向利用作为越狱攻击向量——边缘化群体关键词的越狱成功率比优势群体高出20%，并提出基于提示词的轻量防御方法BiasDefense。

**[DeCoRL: Decoupling Reasoning Chains via Parallel Sub-Step Generation and Cascaded Reinforcement for Interpretable and Scalable RLHF](decorl_decoupling_reasoning_chains_via_parallel_sub-step_gen.md)**

:   DeCoRL 将 CoT 推理从单体顺序处理转变为"交响乐团式"的模块化并行协作——9 个专用子模型（解析/语义/实体/事实核查/风格/质量/计算/验证/整合）并行生成推理子步骤，通过双重奖励归因（本地质量+贡献度）+ 级联 DRPO 优化协调，在 RM-Bench 上达到 80.8%（超越所有基线），同时实现 3.8 倍推理加速和 22.7% 的可解释性提升。

**[Differentiated Directional Intervention: A Framework for Evading LLM Safety Alignment](differentiated_directional_intervention_a_framework_for_evading_llm_safety_align.md)**

:   将 LLM 安全对齐的内部表征从传统的"单一拒绝方向"解构为功能独立的"危害检测方向"和"拒绝执行方向"，在此基础上提出 DBDI 框架，分别用自适应投影消除和直接引导两种策略精准干预两个方向，在 Llama-2 上实现 97.88% 的越狱成功率。

**[EASE: Practical and Efficient Safety Alignment for Small Language Models](ease_practical_and_efficient_safety_alignment_for_small_language_models.md)**

:   提出 EASE——面向边缘部署小语言模型（SLM）的安全对齐框架，通过两阶段设计解决"浅层拒绝不够安全 vs 深度推理太贵"的矛盾：第一阶段从大型推理模型蒸馏安全推理能力到 SLM，第二阶段用选择性推理激活（仅对脆弱语义区域的对抗查询启用推理，良性查询直接响应），越狱攻击成功率降低 17%（vs 浅层对齐）同时推理开销降低 90%（vs 全推理）。

**[Enhancing Uncertainty Estimation in LLMs with Expectation of Aggregated Internal States](enhancing_uncertainty_estimation_in_llms_with_expectation_of_aggregated_internal.md)**

:   提出EAGLE方法，通过聚合LLM多个中间层隐藏状态的logits并计算置信度分布的期望值来估计不确定性，无需训练额外参数，在多个数据集和模型上ECE从12.6%降至3.2%，AUROC从59.0%提升至61.6%。

**[EPO: Diverse and Realistic Protein Ensemble Generation via Energy Preference Optimization](epo_diverse_and_realistic_protein_ensemble_generation_via_energy_preference_opti.md)**

:   提出EPO（Energy Preference Optimization），将反向SDE采样与listwise能量排序偏好优化结合，用能量信号对齐预训练蛋白质生成器与目标Boltzmann分布，在Tetrapeptides/ATLAS/Fast-Folding三个基准9个指标上达到SOTA，完全消除了昂贵的分子动力学（MD）模拟需求。

**[Exploring the Effects of Alignment on Numerical Bias in Large Language Models](exploring_the_effects_of_alignment_on_numerical_bias_in_large_language_models.md)**

:   系统揭示了LLM对齐过程（指令调优+偏好调优）是LLM评估器产生数值偏差的根本原因，并验证分数范围调整是最有效的缓解策略。

**[From Classification to Ranking: Enhancing LLM Reasoning for MBTI Personality Detection](from_classification_to_ranking_enhancing_llm_reasoning_capabilities_for_mbti_per.md)**

:   将MBTI人格检测从传统的四维二分类重构为listwise排序任务，通过SFT冷启动+GRPO强化学习（NDCG+维度相似度双奖励），在Kaggle和PANDORA数据集上以7B模型达到SOTA。

**[Importance-Aware Data Selection for Efficient LLM Instruction Tuning](importance-aware_data_selection_for_efficient_llm_instruction_tuning.md)**

:   提出MIWV（Model Instruction Weakness Value）指标，通过比较LLM在有/无one-shot ICL示例下的损失差来衡量每条指令数据对模型能力提升的重要性，在Alpaca数据集上仅用1%（520条）数据即全面超越全量52002条的微调效果。

**[Margin-aware Preference Optimization for Aligning Diffusion Models without Reference](margin-aware_preference_optimization_for_aligning_diffusion_models_without_refer.md)**

:   提出 MaPO（Margin-aware Preference Optimization），一种无需参考模型的偏好对齐方法，通过直接优化 Bradley-Terry 模型下偏好/非偏好输出的似然 margin 来对齐 T2I 扩散模型，在风格适配、安全生成、通用偏好对齐等 5 个领域均超越 DPO 和专用方法。

**[MetaGDPO: Alleviating Catastrophic Forgetting with Metacognitive Knowledge through Group Direct Preference Optimization](metagdpo_alleviating_catastrophic_forgetting_with_metacognitive_knowledge_throug.md)**

:   提出MetaGDPO方法，从数据侧（基于元认知知识的5K数据构建MetaKL）和训练侧（GDPO——将GRPO的在线采样替换为大模型离线response group的DPO变体）两方面缓解小模型（<8B）在推理能力蒸馏中的灾难性遗忘问题。

**[On the Exponential Convergence for Offline RLHF with Pairwise Comparisons](on_the_exponential_convergence_for_offline_rlhf_with_pairwise_comparisons.md)**

:   在离线RLHF的成对比较设定下，提出RL-LOW算法实现了simple regret的指数收敛 $\exp(-\Omega(n/H))$，并首次导出实例依赖下界证明该速率在指数意义上是最优的。

**[Reducing the Scope of Language Models](reducing_the_scope_of_language_models.md)**

:   系统评估 LLM "范围限制"（scoping）方法——让部署在特定用途的 LLM 只响应域内查询、拒绝所有域外请求。在 3 个模型家族×多种任务上比较 prompting / SFT / DPO / 探针 / Circuit Breakers (CB)，发现 SFT 在高数据多样性下最强、CB 在低多样性下最强、分层组合 (SFT→CB) 保留两者优势——关键发现是范围限制的可行性高度依赖训练数据多样性。

**[Rethinking Direct Preference Optimization in Diffusion Models](rethinking_direct_preference_optimization_in_diffusion_models.md)**

:   提出两个正交改进增强扩散模型偏好优化：(1) 稳定参考模型更新策略放松冻结参考模型并通过正则化鼓励探索；(2) 时间步感知训练策略缓解跨时间步奖励尺度不平衡。二者可嵌入多种偏好优化算法，在人类偏好评估基准上提升SOTA。

**[SafeNlidb: A Privacy-Preserving Safety Alignment Framework for LLM-based Natural Language Database Interfaces](safenlidb_a_privacy-preserving_safety_alignment_framework_for_llm-based_natural_.md)**

:   提出SafeNlidb框架，通过安全感知数据合成管线和交替偏好优化策略，实现LLM驱动的自然语言数据库接口（NLIDB）在安全推理与SQL生成之间的联合优化，有效防御隐式推理攻击下的隐私泄露。

**[When Human Preferences Flip: An Instance-Dependent Robust Loss for RLHF](when_human_preferences_flip_an_instance-dependent_robust_loss_for_rlhf.md)**

:   针对人类偏好标注中普遍存在的"偏好翻转"问题，提出 FA-DPO（Flipping-Aware DPO），将标注过程建模为"真实意图 + 实例依赖翻转概率"两阶段，通过修正 BT 模型损失和迭代优化翻转估计模块，在多种噪声场景下显著提升对齐鲁棒性，实例依赖翻转率高时比 DPO 提升 16.7%。
