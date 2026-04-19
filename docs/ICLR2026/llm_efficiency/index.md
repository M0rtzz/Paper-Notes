---
title: >-
  ICLR2026 LLM效率方向 22篇论文解读
description: >-
  22篇ICLR2026 LLM效率方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM效率

**🔬 ICLR2026** · **22** 篇论文解读

**[Did You Check the Right Pocket? Cost-Sensitive Store Routing for Memory-Augmented Agents](did_you_check_the_right_pocket_cost-sensitive_store_routing_for_memory-augmented.md)**

:   将记忆增强 Agent 的多存储检索形式化为代价敏感的存储路由问题（store routing），证明选择性检索相比全量检索可在减少 62% context token 的同时提升 QA 准确率（86% vs 81%），并提出基于语义信号的启发式路由基线。

**[DND: Boosting Large Language Models with Dynamic Nested Depth](dnd_boosting_large_language_models_with_dynamic_nested_depth.md)**

:   DND在Transformer层末端通过路由器选出关键token，将其回送同一层进行额外处理（嵌套深度），配合路由控制损失和阈值控制方案实现精确稳定的token选择，以极少的参数增加（<0.1M）在Qwen3-1.7B和Qwen3-30B-A3B上分别获得1.88%和0.87%的平均性能提升。

**[EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models](evoengineer_mastering_automated_cuda_kernel_code_evolution_with_large_language_m.md)**

:   提出 EvoEngineer，首个系统化的 LLM-based 代码演化框架，将代码演化分解为 traverse technique（含两层设计：solution guiding + prompt engineering）和 population management 两个正交组件，在 91 个真实 CUDA kernel 上实现最高 2.72× 中位加速比和 69.8% 代码有效率，在性能和正确性两个维度上超越现有方法。

**[Expert Divergence Learning for MoE-based Language Models](expert_divergence_learning_for_moe-based_language_models.md)**

:   解决 MoE 训练中的专家同质化问题，通过最大化不同数据域之间路由分布的 Jensen-Shannon 散度，鼓励不同域激活不同专家子集，在 15B-A1.5B 模型上提升专家特化程度和语言建模性能。

**[Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws](fast_catch-up_late_switching_optimal_batch_size_scheduling_via_functional_scalin.md)**

:   通过 Functional Scaling Law 框架理论推导出 batch size scheduling 的最优策略——对困难任务，最优策略是训练大部分时间用小 batch，仅在最后阶段切换到大 batch（late switching）；并揭示了 fast catch-up 效应——切换后 loss 迅速追上全程大 batch 的轨迹，在 1.1B 参数 1T token 的 LLM 预训练中验证了该原则。

**[LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](lycheedecode_accelerating_long-context_llm_inference_via_hybrid-head_sparse_deco.md)**

:   提出 LycheeDecode，通过将注意力头细粒度分为少量 retrieval heads（负责全注意力选关键 token）和大量 sparse heads（复用选出的 token 做稀疏计算），并用 HardKuma 分布端到端学习头类型，在 128K 上下文下实现 2.7× 加速且性能不降。

**[MVAR: Visual Autoregressive Modeling with Scale and Spatial Markovian Conditioning](mvar_visual_autoregressive_modeling_with_scale_and_spatial_markovian_conditionin.md)**

:   提出 MVAR（Markovian Visual AutoRegressive），通过引入尺度 Markov 假设（仅依赖相邻尺度而非所有前序尺度）和空间 Markov 注意力（限制邻域大小 k），将 VAR 模型的注意力计算复杂度从 $\mathcal{O}(N^2)$ 降至 $\mathcal{O}(Nk)$，在 ImageNet 256×256 上实现同等或更优性能的同时，推理显存降低 3.0-4.2×，且仅需 8 张 RTX 4090 即可训练。

**[One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning](one-prompt_strikes_back_sparse_mixture_of_experts_for_prompt-based_continual_lea.md)**

:   提出 SMoPE 框架，将单个共享 prompt 组织为稀疏 MoE 结构中的多个 prompt expert，通过 prompt-attention score aggregation 实现动态稀疏激活，在保持高参数效率的同时显著缓解知识干扰，在多个持续学习 benchmark 上达到 SOTA。

**[Polynomial, trigonometric, and tropical activations](polynomial_trigonometric_and_tropical_activations.md)**

:   系统探索基于正交基（Hermite多项式、Fourier三角基）和热带化（tropicalization）的可学习激活函数族，通过方差保持初始化解决多项式激活的梯度爆炸/消失问题，在GPT-2和ConvNeXt上成功替代GELU实现有效训练。

**[Q-RAG: Long Context Multi-Step Retrieval via Value-Based Embedder Training](q_rag_long_context_multi_step_retrieval.md)**

:   将多步检索建模为 MDP，用基于值的 RL（soft Q-learning）微调 **embedder 而非 LLM**，Q 函数设计为状态嵌入和动作嵌入的内积（理论证明为万能近似器），结合 RoPE 相对位置编码实现时序推理，在单卡 A100 上训练 12 小时，4K 训练泛化到 1M+ token 上下文，RULER 基准达到近乎完美的 NIAH 性能。

**[RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training](race_attention_a_strictly_linear-time_attention_for_long-sequence_training.md)**

:   提出 RACE Attention——用幂次角核替代 softmax 并通过可微 LSH 草图近似注意力输出，实现严格线性时间复杂度，支持单 GPU 处理 1200 万 token、单 CPU 处理 7500 万 token，在多种任务上匹配或超越 softmax 精度。

**[Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective](randomization_boosts_kv_caching_learning_balances_query_load_a_joint_perspective.md)**

:   提出首个KV缓存感知负载均衡统一数学模型，设计随机化叶节点淘汰算法RLT(O(log n)竞争比)和基于学习的贪心路由LBGR，在多LLM服务场景下将延迟降低最高11.96×、TTFT降低14.06×。

**[Rethinking Benign Relearning: Syntax as the Hidden Driver of Unlearning Failures](rethinking_benign_relearning_syntax_as_the_hidden_driver_of_unlearning_failures.md)**

:   揭示 LLM 机器遗忘中"良性重学习"现象的真正驱动因素是句法相似性而非主题相关性，并提出句法多样化策略（paraphrase forget set），有效抑制重学习、加速遗忘并缓解遗忘效果与模型效用之间的 trade-off。

**[Rethinking Uncertainty Estimation in LLMs: A Principled Single-Sequence Measure](rethinking_uncertainty_estimation_in_llms_a_principled_single-sequence_measure.md)**

:   从 proper scoring rules 框架出发，证明最高概率输出序列的负对数似然（MSP）是理论上合理的不确定性度量，并提出 G-NLL——仅用一次贪心解码就能逼近该度量，在多个场景下匹配或超越需要多次采样的 SOTA 方法。

**[Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling](semantic_parallelism_redefining_efficient_moe_inference_via_model-data_co-schedu.md)**

:   提出语义并行(Semantic Parallelism)范式，通过预测token-expert路由路径并协同调度模型放置与数据分发，大幅削减MoE推理中专家并行的all-to-all通信开销，在Attention-DP场景下吞吐提升最高2.78×，Attention-TP场景下延迟降低最高24.9%。

**[Steering Language Models with Weight Arithmetic](steering_language_models_with_weight_arithmetic.md)**

:   提出对比式权重引导（Contrastive Weight Steering），通过对正/负行为微调模型的权重差来提取行为方向向量，直接修改模型权重实现行为控制，在谄媚性、恶意性和拒绝性实验中比激活引导（Activation Steering）具有更好的泛化能力和一致性。

**[SwingArena: Adversarial Programming Arena for Long-context GitHub Issue Solving](swingarena_competitive_programming_arena_for_long-context_github_issue_solving.md)**

:   提出SwingArena对抗性评测框架，让两个LLM在真实GitHub issue上交替扮演补丁提交者和测试审查者，通过仓库原生CI流水线（编译/lint/回归测试）端到端验证，在C++/Python/Rust/Go四语言400个实例上揭示了模型在"激进补丁生成"与"防御性质量保证"间的行为分化。

**[Token-level Data Selection for Safe LLM Fine-tuning](token-level_data_selection_for_safe_llm_fine-tuning.md)**

:   提出 TOSS（Token-level data Selection for Safe LLM fine-tuning），首个 token 级别的数据选择框架,通过安全退化模型和效用导向模型之间的损失差评估每个 token 的安全风险，实现比样本级方法更优的安全-效用权衡。

**[Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models](understanding_and_improving_length_generalization_in_hierarchical_sparse_attenti.md)**

:   系统解剖基于 chunk 的稀疏注意力架构，识别出三个关键设计原则（非线性 Chunk Encoder + CLS token、Bypassing Residual Path、训练时强制选择稀疏性），将 4K 上下文训练的模型成功外推到 3200 万 token。

**[Universe Routing: Why Self-Evolving Agents Need Epistemic Control](universe_routing_why_self-evolving_agents_need_epistemic_control.md)**

:   将自主Agent在链式推理中容易混淆认识论框架（如频率主义vs贝叶斯）的问题形式化为"宇宙路由"，训练一个465M参数的轻量路由器将问题分类到7个互斥信念空间后分发给专用求解器，证明硬路由比软MoE快7倍且精度相同，模块化架构配合rehearsal可实现零遗忘的持续学习。

**[When Does Divide and Conquer Work for Long Context LLM? A Noise Decomposition Framework](when_does_divide_and_conquer_work_for_long_context_llm_a_noise_decomposition_fra.md)**

:   提出理论框架将长上下文任务失败分解为三类噪声（任务噪声/模型噪声/聚合器噪声），证明当模型噪声超线性增长时弱模型+分块处理可超越强模型单次处理，并给出快速估计最优 chunk size 的方法（3-5 个样本即可）。

**[xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity](xlstm_scaling_laws_competitive_performance_with_linear_time-complexity.md)**

:   系统对比 xLSTM 与 Transformer 的 scaling law，证明 xLSTM 在训练损失-算力 Pareto 前沿、过训练 regime 和推理速度上全面优于同规模 Transformer，且优势随上下文长度增大而增长。
