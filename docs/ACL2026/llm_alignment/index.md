---
title: >-
  ACL2026 对齐 / RLHF方向11篇论文解读
description: >-
  11篇ACL2026的对齐 / RLHF 方向论文解读，涵盖 LLM、Agent、对抗鲁棒、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐 / RLHF

**💬 ACL2026** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (10)](../../CVPR2026/llm_alignment/index.md) · [🔬 ICLR2026 (42)](../../ICLR2026/llm_alignment/index.md) · [🤖 AAAI2026 (20)](../../AAAI2026/llm_alignment/index.md) · [🧠 NeurIPS2025 (53)](../../NeurIPS2025/llm_alignment/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/llm_alignment/index.md) · [🧪 ICML2025 (27)](../../ICML2025/llm_alignment/index.md)

🔥 **高频主题：** LLM ×3 · Agent ×2 · 对抗鲁棒 ×2 · 对齐/RLHF ×2

**[Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling](aligning_agents_via_planning_a_benchmark_for_trajectory-level_reward_modeling.md)**

:   提出 Plan-RewardBench，一个面向复杂工具增强场景的轨迹级偏好基准，用于评估奖励模型在多步规划、工具使用和错误恢复等场景下区分优劣智能体轨迹的能力。

**[Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs](beyond_marginal_distributions_a_framework_to_evaluate_the_representativeness_of_.md)**

:   本文提出了一种超越边际分布的 LLM 代表性评估框架，通过同时考察边际响应分布和跨问题相关结构来评估人口统计对齐模型，发现虽然微调和 persona prompting 能改善边际分布的近似度，但两者都无法忠实再现人类价值观调查中的多变量相关模式。

**[ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training](consistrm_improving_generative_reward_models_via_consistency-aware_self-training.md)**

:   ConsistRM 提出基于一致性感知的自训练框架，通过时序一致性伪标签（融合在线状态和历史记忆的偏好一致性）和语义一致性批评奖励（衡量多次生成批评的语义相似度）两个模块，在无需人工标注的条件下将生成式奖励模型的五个基准平均性能提升 1.5%，同时显著缓解了位置偏差问题。

**[Into the Gray Zone: Domain Contexts Can Blur LLM Safety Boundaries](into_the_gray_zone_domain_contexts_can_blur_llm_safety_boundaries.md)**

:   本文发现领域特定上下文（如化学论文）会选择性放松 LLM 对相关有害知识的防护（纵向解锁），而安全研究上下文会触发跨所有有害类别的广泛防护放松（通用解锁），据此提出 Jargon 攻击框架，在包括 GPT-5.2、Claude-4.5 在内的七个前沿模型上实现超 93% 的攻击成功率。

**[Reward Modeling for Scientific Writing Evaluation](reward_modeling_for_scientific_writing_evaluation.md)**

:   本文提出 SciRM 和 SciRM-Ref 两个针对科学写作评估的开源奖励模型，通过两阶段强化学习（GRPO）分别优化评估偏好和推理能力，实现了在多种科学写作任务上的细粒度多方面评估，并能泛化到未见过的评估任务和标准。

**[Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors](robust_tool_use_via_fission-grpo_learning_to_recover_from_execution_errors.md)**

:   提出 Fission-GRPO，在 RL 训练循环中将工具执行错误动态转化为在线策略修正训练实例：通过学习的错误模拟器生成诊断反馈并重采样恢复轨迹，将 Qwen3-8B 的错误恢复率提升 5.7%，整体准确率从 42.75% 提升至 46.75%。

**[SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging](safemerge_preserving_safety_alignment_in_fine-tuned_large_language_models_via_se.md)**

:   本文提出 SafeMERGE，一种轻量级后微调框架，通过余弦相似度检测偏离安全行为的微调层，仅将这些层与安全模型的对应层合并，在四个 LLM 上显著降低有害输出同时保持甚至提升任务性能。

**[SFTMix: Elevating Language Model Instruction Tuning with Mixup Recipe](sftmix_elevating_language_model_instruction_tuning_with_mixup_recipe.md)**

:   本文提出 SFTMix，一种基于 Mixup 的指令微调方法，通过训练动态将 SFT 数据集分为高置信度和低置信度子集，在隐表示空间对两者进行线性插值并施加 Mixup 正则化，在不依赖高质量数据集的情况下，跨 LLM 家族和数据集规模一致性地提升指令遵循能力。

**[STAR-Teaming: A Strategy-Response Multiplex Network Approach to Automated LLM Red Teaming](star-teaming_a_strategy-response_multiplex_network_approach_to_automated_llm_red.md)**

:   本文提出 STAR-Teaming，一种基于策略-响应多路复用网络（Multiplex Network）的自动化红队测试框架，通过将攻击策略选择建模为逆 Ising 问题的概率优化，在 HarmBench 上达到平均 74.5% 的攻击成功率，比最强基线高 13.5%，同时显著降低计算开销。

**[Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms](towards_bridging_the_reward-generation_gap_in_direct_alignment_algorithms.md)**

:   本文识别了直接对齐算法（DAAs）中的"奖励-生成鸿沟"——训练目标与自回归解码动态之间的不匹配，提出 POET（Prefix-Oriented Equal-length Training），通过将偏好响应对截断为较短者长度来隐式约束 token 级 MDP 在所有时间步上收敛，在 AlpacaEval 2 上最高提升 11.8 个百分点。

**[TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense](trajguard_streaming_hidden-state_trajectory_detection_for_decoding-time_jailbrea.md)**

:   本文提出 TrajGuard，一种无需训练的解码时越狱防御框架，通过滑动窗口聚合关键层隐藏状态轨迹实时量化风险，仅在风险持续超过阈值时触发轻量级语义裁判，在 12 种越狱攻击上实现 95% 平均防御率，检测延迟仅 5.2ms/token，误报率低于 1.5%。
