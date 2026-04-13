---
title: >-
  ICLR2026 代码智能方向 21篇论文解读
description: >-
  21篇ICLR2026 代码智能方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💻 代码智能

**🔬 ICLR2026** · 共 **21** 篇

**[Ambig-Swe Interactive Agents To Overcome Underspecificity In Software Engineerin](ambig-swe_interactive_agents_to_overcome_underspecificity_in_software_engineerin.md)**

:   构建 Ambig-SWE（基于 SWE-Bench Verified 的欠指定变体），系统评估 LLM 编程 agent 在三个维度上的交互能力——检测欠指定、提出澄清问题、利用交互信息——发现交互可将欠指定场景下的解决率提升最高 74%，但模型默认非交互行为且难以区分指定充分/不足的指令。

**[Breaking The Sft Plateau Multimodal Structured Reinforcement Learning For Chart-](breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)**

:   针对图表到代码生成任务中SFT的性能瓶颈问题，提出多模态结构化强化学习（MSRL），通过文本+视觉双层奖励函数和两阶段RL策略，在ChartMimic和ReachQA上分别提升6.2%和9.9%的高层指标，达到开源SOTA并媲美GPT-4o。

**[Card Towards Conditional Design Of Multi-Agent Topological Structures](card_towards_conditional_design_of_multi-agent_topological_structures.md)**

:   CARD提出了一种条件图生成框架(Conditional Agentic Graph Designer)，通过条件变分图编码器和环境感知优化，根据模型能力、工具可用性和知识源变化等动态环境信号自适应地设计多Agent通信拓扑结构，在HumanEval、MATH和MMLU上一致超越静态和基于提示的基线方法。

**[Diablo Diagonal Blocks Are Sufficient For Finetuning](diablo_diagonal_blocks_are_sufficient_for_finetuning.md)**

:   提出 DiaBlo，仅微调权重矩阵的对角块作为参数高效微调方法：避免了 LoRA 低秩矩阵乘积的优化难题，zero 初始化即可稳定收敛，GPU 友好的 batched 矩阵乘法实现，理论证明在参数预算相同时表达力严格优于 LoRA，在常识推理/算术推理/代码生成/安全对齐上全面优于 LoRA 及其变体。

**[Dro-Instructzero Distributionally Robust Prompt Optimization For Instruction Fol](dro-instructzero_distributionally_robust_prompt_optimization_for_instruction_fol.md)**

:   将分布鲁棒优化（DRO）引入贝叶斯优化框架以实现零样本指令优化，使优化后的指令在分布偏移和对抗性评估条件下仍保持可靠性能。

**[Dro-Instructzero Distributionally Robust Prompt Optimization For Large Language ](dro-instructzero_distributionally_robust_prompt_optimization_for_large_language_.md)**

:   将分布鲁棒优化（DRO）引入 InstructZero 的贝叶斯优化框架，通过在 f-divergence 球定义的模糊集上最大化最坏情况期望效用，使自动搜索得到的 prompt 在分布偏移下仍能保持可靠性能。

**[Execution-Grounded Credit Assignment For Grpo In Code Generation](execution-grounded_credit_assignment_for_grpo_in_code_generation.md)**

:   提出 EGCA（Execution-Grounded Credit Assignment），通过执行追踪定位程序中最早的语义偏差位置，将 GRPO 的梯度集中到因果 token span 上，解决代码生成中粗粒度信用分配问题，在 HumanEval 上达到 82.1% pass@1。

**[Improving Code Localization With Repository Memory](improving_code_localization_with_repository_memory.md)**

:   通过利用代码仓库的 commit 历史构建情景记忆（过去 commit）和语义记忆（活跃代码功能摘要），增强语言代理的代码定位能力，在 SWE-bench 上取得显著提升。

**[Imse Intrinsic Mixture Of Spectral Experts Fine-Tuning For Test-Time Adaptation](imse_intrinsic_mixture_of_spectral_experts_fine-tuning_for_test-time_adaptation.md)**

:   提出 IMSE——将预训练 ViT 线性层通过 SVD 分解为"谱专家"，仅微调奇异值实现极端参数高效的测试时适应，并通过多样性最大化损失和域感知谱码检索机制，在 TTA/CTTA/渐进 CTTA 三种场景下达到 SOTA。

**[Inference-Time Safety For Code Llms Via Retrieval-Augmented Revision](inference-time_safety_for_code_llms_via_retrieval-augmented_revision.md)**

:   提出SOSecure方法，在LLM生成代码后通过检索Stack Overflow安全讨论作为上下文引导模型推理时修正潜在漏洞，无需重训练即可适应新的安全实践，在多个数据集上减少漏洞且不引入新的安全问题。

**[Innogym Benchmarking The Innovation Potential Of Ai Agents](innogym_benchmarking_the_innovation_potential_of_ai_agents.md)**

:   提出InnoGym框架，首次从"创新性"维度系统评估AI Agent——引入Performance Gain（性能增益）和Novelty（方法论新颖性）双指标，在18个真实工程/科研任务上发现当前Agent能产生新颖方案但执行鲁棒性不足，无法将创意转化为性能提升（平均归一化增益为负）。

**[Kv Cache Transform Coding For Compact Storage In Llm Inference](kv_cache_transform_coding_for_compact_storage_in_llm_inference.md)**

:   提出 KVTC，一种借鉴经典媒体压缩技术（PCA 特征去相关 + 自适应量化 + 熵编码）的 KV 缓存压缩方法，在 Llama 3、Mistral NeMo、R1-Qwen 2.5 等模型上实现最高 20× 压缩（特定场景下 40×+），优于 token 驱逐、量化、SVD 等基线方法。

**[Learning To Reason Without External Rewards](learning_to_reason_without_external_rewards.md)**

:   提出 Intuitor，一种用模型自身置信度（self-certainty，即输出分布与均匀分布的 KL 散度）替代外部可验证奖励的 RLIF 方法，在数学推理上匹配 GRPO 性能，同时在代码生成等域外任务上展现更好的泛化能力。

**[Mathfimer Enhancing Mathematical Reasoning By Expanding Reasoning Steps Through ](mathfimer_enhancing_mathematical_reasoning_by_expanding_reasoning_steps_through_.md)**

:   借鉴代码补全中的 Fill-in-the-Middle (FIM) 范式，训练一个专门的步骤扩展模型 MathFimer-7B，在已有数学解题链中插入更细粒度的中间推理步骤，从而系统性提升下游模型的数学推理能力。

**[Paper2Code Automating Code Generation From Scientific Papers In Machine Learning](paper2code_automating_code_generation_from_scientific_papers_in_machine_learning.md)**

:   提出 PaperCoder——一个多智能体 LLM 框架，通过规划（Planning）、分析（Analysis）、生成（Coding）三阶段流水线，将机器学习论文自动转化为可运行的代码仓库，其中 88% 的生成仓库被论文作者评为最佳，且在 PaperBench 基准上大幅超越基线。

**[Sharing State Between Prompts And Programs](sharing_state_between_prompts_and_programs.md)**

:   提出共享程序状态（shared program state）抽象，让 prompt 直接读写程序变量、操作堆对象和控制程序流程，实现为 Nightjar 系统（Python + prompt 混合编程），在保持或提升准确率（+4-19%）的同时减少 39.6% 代码量。

**[Shieldedcode Learning Robust Representations For Virtual Machine Protected Code](shieldedcode_learning_robust_representations_for_virtual_machine_protected_code.md)**

:   提出 ShieldedCode——首个保护感知的代码表征学习框架，通过层次依赖建模（指令内/前序/跨指令三层）和联合功能感知+保护感知对比学习，使 LLM 能够生成、比较和推理虚拟机保护代码，在 VM 代码生成（Pass@1 26.95% vs. GPT-4o 22.58%）和二进制相似性检测上均超越现有方法。

**[Supervised Reinforcement Learning From Expert Trajectories To Step-Wise Reasonin](supervised_reinforcement_learning_from_expert_trajectories_to_step-wise_reasonin.md)**

:   提出 Supervised Reinforcement Learning (SRL)，将问题求解重新建模为逐步动作生成过程，通过基于序列相似度的密集奖励信号，使小模型能够从专家轨迹中学习原本 SFT 和 RLVR 都无法解决的困难推理问题。

**[The Limits Of Long-Context Reasoning In Automated Bug Fixing](the_limits_of_long-context_reasoning_in_automated_bug_fixing.md)**

:   系统评估当前 LLM 在长上下文代码调试中的能力极限，发现 agentic 工作流的成功来自任务分解而非长上下文推理（成功轨迹仅消耗 20-30K token），64K token 单次补丁生成中性能急剧下降（GPT-5-nano 0%），揭示名义上下文长度与实际可用上下文能力之间的显著差距。

**[Training Large Language Models To Reason In Parallel With Global Forking Tokens](training_large_language_models_to_reason_in_parallel_with_global_forking_tokens.md)**

:   提出 Set Supervised Fine-Tuning (SSFT)，通过二分图匹配将全局分叉令牌 (global forking tokens) 与多样推理轨迹对齐，使 LLM 能从单个控制令牌全局引导不同推理模式，在数学推理和代码生成任务上显著优于标准 SFT 和 GRPO。

**[Training Large Language Models To Reason In Parallel With Global Reflection](training_large_language_models_to_reason_in_parallel_with_global_reflection.md)**

:   提出 Set Supervised Fine-Tuning (SSFT)，通过引入全局分叉 token 和基于二部匹配的集合损失，训练 LLM 从单个控制 token 触发多样且正确的推理模式，在 Pass@1 和 Cons@k 上均超越标准 SFT+GRPO。
