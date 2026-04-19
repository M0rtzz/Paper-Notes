---
title: >-
  ICML2025 LLM/NLP方向 18篇论文解读
description: >-
  18篇ICML2025 LLM/NLP方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM/NLP

**🧪 ICML2025** · **18** 篇论文解读

**[Adaptive Multi-prompt Contrastive Network for Few-shot Out-of-distribution Detection](adaptive_multi-prompt_contrastive_network_for_few-shot_out-of-distribution_detec.md)**

:   提出 AMCN（Adaptive Multi-prompt Contrastive Network），通过生成三类自适应文本 prompt（可学习 ID prompt、标签固定 OOD prompt、标签自适应 OOD prompt）并结合类别自适应阈值，在仅有少量 ID 标注样本的条件下实现高质量 OOD 检测，显著超越现有 few-shot OOD 检测方法。

**[B-score: Detecting biases in large language models using response history](b-score_detecting_biases_in_large_language_models_using_response_history.md)**

:   提出B-score指标，通过比较LLM在单轮(single-turn)与多轮(multi-turn)对话中的回答概率差异来检测偏见，发现LLM在多轮对话中能"自我去偏"，并利用B-score提升答案验证准确率。

**[BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute](best-route_adaptive_llm_routing_with_test-time_optimal_compute.md)**

:   提出 BEST-Route（Best-of-n Enhanced Sampling and Test-time Route Optimization），在传统查询路由的基础上引入 best-of-n 采样策略，使路由器不仅选择模型，还自适应决定采样数量 n，通过小模型多次采样+选优替代大模型单次调用，在不到 1% 性能损失下降低高达 60% 的推理成本。

**[Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence](beyond_induction_heads_in-context_meta_learning_induces_multi-phase_circuit_emer.md)**

:   本文通过设计 In-Context Meta-Learning (ICML) 实验环境，揭示了 Transformer 在获得上下文元学习能力的训练过程中，内部电路经历了三个截然不同的阶段性涌现（Bigram → Label Attention → Chunk Example），而非 induction head 研究中观察到的单阶段跃变，从而为理解 ICL 的深层机制提供了新视角。

**[Binary Hypothesis Testing for Softmax Models and Leverage Score Models](binary_hypothesis_testing_for_softmax_models_and_leverage_score_models.md)**

:   从理论角度研究Softmax模型和Leverage Score模型的二元假设检验问题，建立了在能量约束下区分两个参数化模型所需的查询次数的紧界，与理解LLM不同能力域的区分性问题相关。

**[Build Agent Advocates, Not Platform Agents](build_agent_advocates_not_platform_agents.md)**

:   Position paper，指出LMA（语言模型代理）若被平台公司控制将成为加剧监控、锁定和注意力操控的"platform agents"，提出应发展用户控制的"agent advocates"来保护个人自主权，并给出三大干预措施：开放模型/算力、互操作性标准、市场监管。

**[Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models](emergent_symbolic_mechanisms_support_abstract_reasoning_in_large_language_models.md)**

:   本文通过因果分析、表征分析和注意力分析等方法，在13个开源LLM中识别出支持抽象推理的三阶段涌现符号架构——符号抽象头将输入token转化为抽象变量、符号归纳头在抽象变量层面进行序列归纳、检索头根据预测的抽象变量检索对应值来完成下一token预测。

**[Interchangeable Token Embeddings for Extendable Vocabulary and Alpha-Equivalence](interchangeable_token_embeddings_for_extendable_vocabulary_and_alpha-equivalence.md)**

:   提出双部分 token 嵌入策略（共享可学习部分 + 随机区分部分），使语言模型能在训练后泛化到更大词表，并对 alpha-等价变换具有天然鲁棒性。

**[LaRoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation](la_rosa_enhancing_llm_efficiency_via_layerwise_rotated_sparse_activation.md)**

:   LaRoSA 提出了一种无需训练的激活稀疏化方法，通过逐层正交旋转矩阵将输入激活变换到更适合稀疏化的空间，并结合 Top-K 选择实现一致的模型级稀疏度和可靠的推理加速。

**[LASER: Attention with Exponential Transformation](laser_attention_with_exponential_transformation.md)**

:   通过分析注意力机制中 softmax 的梯度反向传播瓶颈，提出 LASER 注意力——在指数变换的 Value 空间中做注意力计算（即对 exp(V) 做 attention 再取 log），从而获得更大的 Jacobian 信号，改善参数学习效率。

**[On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding](on_expressive_power_of_looped_transformers_theoretical_analysis_and_enhancement_.md)**

:   本文首次建立了 Looped Transformer 关于循环次数和目标函数连续性模的逼近速率理论，揭示了循环架构特有的逼近误差来源（上下文连续性与 token 连续性），并提出 Timestep-Modulated Looped Transformer (TMLT) 通过时间步编码消除该限制，在推理、上下文学习和语言建模任务上取得一致提升。

**[Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective](product_of_experts_with_llms_boosting_performance_on_arc_is_a_matter_of_perspect.md)**

:   将 LLM 同时用作候选解生成器和评分器，通过基于 DFS 的搜索算法生成高概率候选解，再利用多视角增强下的 Product of Experts (PoE) 打分选出最优答案，在 ARC-AGI 公开评估集上以 71.6% 的准确率达到开源 SOTA，超越人类平均水平（60.2%），且单任务推理成本仅约 $0.02。

**[QuEst: Enhancing Estimates of Quantile-Based Distributional Measures Using Model Predictions](quest_enhancing_estimates_of_quantile-based_distributional_measures_using_model_.md)**

:   提出 QuEst 框架，将少量高质量观测数据与大量模型预测（imputed）数据相结合，对分位数相关的分布度量（QBDM）给出更精确的点估计和严格的置信区间，覆盖 CVaR、Interval-VaR 等经典指标。

**[Regress, Don't Guess — A Regression-like Loss on Number Tokens for Language Models](regress_dont_guess_--_a_regression-like_loss_on_number_tokens_for_language_model.md)**

:   提出 Number Token Loss (NTL)，一种纯 token 级别的回归式损失函数，通过最小化数值 token 之间的 $L_p$ 范数或 Wasserstein 距离，为 LLM 注入数值邻近性归纳偏置。

**[Taming Knowledge Conflicts in Language Models](taming_knowledge_conflicts_in_language_models.md)**

:   揭示了语言模型注意力头中"上下文信息与参数记忆的叠加"（CP Superposition）现象，提出 JuICE（Just Run Twice）方法，通过双次推理的注意力干预策略，在不微调的前提下灵活引导模型偏向参数知识或上下文知识，在 11 个数据集 × 6 种模型架构上达到 SOTA。

**[The Lock-in Hypothesis: Stagnation by Algorithm](the_lock-in_hypothesis_stagnation_by_algorithm.md)**

:   本文提出并形式化了"锁定假说"（Lock-in Hypothesis）：LLM 训练与部署过程中形成的人类-AI 反馈循环会固化用户的现有信念，导致群体观点多样性不可逆地丧失，甚至锁定在错误信念上。

**[Theoretical Limitations of Ensembles in the Age of Overparameterization](theoretical_limitations_of_ensembles_in_the_age_of_overparameterization.md)**

:   在过参数化条件下，无限集成模型与单个无穷宽模型逐点等价，集成方差不再反映传统贝叶斯不确定性而是衡量增加模型容量的预期效果，从理论上解释了深度集成相比大模型无本质泛化优势的经验观察。

**[Towards Universal Offline Black-Box Optimization via Learning Language Model Embeddings](towards_universal_offline_black-box_optimization_via_learning_language_model_emb.md)**

:   提出UniSO框架，将不同类型和维度的优化变量统一编码为JSON字符串后输入语言模型，通过token预测（UniSO-T）和数值回归（UniSO-N）两种建模范式训练通用回归器，并通过元数据引导的对比学习和Lipschitz平滑正则化改善嵌入空间质量，实现了跨域跨维度的通用离线黑盒优化。
