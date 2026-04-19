---
title: >-
  ICLR2026 LLM安全方向 11篇论文解读
description: >-
  11篇ICLR2026 LLM安全方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔒 LLM安全

**🔬 ICLR2026** · **11** 篇论文解读

**[Enhancing Hallucination Detection through Noise Injection](enhancing_hallucination_detection_through_noise_injection.md)**

:   在 LLM 中间层的 MLP 激活中注入均匀噪声来近似贝叶斯后验，捕获认知不确定性（epistemic uncertainty），与采样温度捕获的偶然不确定性（aleatoric uncertainty）互补，将 GSM8K 上的幻觉检测 AUROC 从 71.56 提升到 76.14。

**[Gaussian Certified Unlearning in High Dimensions: A Hypothesis Testing Approach](gaussian_certified_unlearning.md)**

:   提出 $(\phi,\varepsilon)$-Gaussian certifiability——基于假设检验 trade-off 函数的高维机器遗忘隐私框架，严格证明在高维比例体系 ($p \sim n$) 下单步 Newton 更新 + 校准高斯噪声即可同时满足隐私 (GPAR) 和精度 (GED→0) 要求，推翻了 Zou et al. (2025) "至少需两步 Newton" 的结论，并从理论上揭示旧 $\varepsilon$-certifiability 与噪声添加机制不兼容的根本原因。

**[How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use](how_far_are_llms_from_professional_poker_players_revisiting_game-theoretic_reaso.md)**

:   系统分析了 LLM 在扑克中的三大推理缺陷（启发式推理、事实误解、知行差距），提出 ToolPoker 框架——首个面向不完全信息博弈的工具集成 LLM 推理系统，通过外部 CFR solver 提供博弈论最优的行动指导，使 7B 模型在 Limit Hold'em 中逼近 Nash 均衡。

**[LH-Deception: Simulating and Understanding LLM Deceptive Behaviors in Long-Horizon Interactions](lh-deception_simulating_and_understanding_llm_deceptive_behaviors_in_long-horizo.md)**

:   提出首个面向长时域交互的 LLM 欺骗行为仿真框架 LH-Deception，采用执行者-监督者-审计者三角色多智能体架构，结合社会科学理论驱动的概率事件系统，在 11 个前沿模型上系统量化了欺骗频率、严重性、类型分布及其对信任关系的侵蚀效应，揭示了静态单轮评估完全无法捕捉的"欺骗链"涌现现象。

**[Lifelong Learning with Behavior Consolidation for Vehicle Routing](lifelong_learning_with_behavior_consolidation_for_vehicle_routing.md)**

:   提出 LLR-BC 框架，在神经 VRP 求解器的终身学习场景中，通过决策步骤级经验缓冲、置信度感知加权（CaEW）和反向 KL 散度行为巩固（DsBC），在分布与规模同时变化的任务序列上将平均性能差距（AP）降低一个数量级，同时保持学新任务的可塑性并提升零样本泛化。

**[Perturbation-Induced Linearization: Constructing Unlearnable Data with Solely Linear Classifiers](perturbation-induced_linearization_constructing_unlearnable_data_with_solely_lin.md)**

:   提出PIL方法，仅使用无偏置线性分类器作为代理模型生成不可学习扰动，通过诱导深度模型线性化来阻止其学习语义特征，比现有方法快100倍以上（CIFAR-10上不到1分钟GPU时间）。

**[Redirection for Erasing Memory (REM): Towards a Universal Unlearning Method for Corrupted Data](redirection_for_erasing_memory_rem_towards_a_universal_unlearning_method_for_cor.md)**

:   本文提出损坏数据遗忘任务的二维分类框架（发现率 × 统计规律性），揭示了现有遗忘方法各自仅在特定区域有效的局限，并提出 REM（重定向以擦除记忆）方法，通过将损坏数据重定向到新增的专用网络容量再丢弃，首次在整个二维任务空间中实现强劲且一致的遗忘性能。

**[Self-Destructive Language Model](self-destructive_language_model.md)**

:   提出 Seam，通过耦合良性和有害数据的优化轨迹（使梯度方向相反），将 LLM 转变为"自毁模型"——在有害微调时自动触发灾难性性能崩溃，创造攻击者的两难困境：低强度攻击无效，高强度攻击导致模型报废。

**[Understanding Sensitivity of Differential Attention through the Lens of Adversarial Robustness](understanding_sensitivity_of_differential_attention_through_the_lens_of_adversar.md)**

:   首次从对抗鲁棒性角度分析 Differential Attention（DA）机制，揭示其减法结构在抑制噪声的同时会通过负梯度对齐放大对抗扰动敏感度，发现"脆弱性原理"——DA 在干净样本上提升判别力但在对抗攻击下更脆弱，且存在深度依赖的鲁棒性交叉效应。

**[Unlearning Evaluation through Subset Statistical Independence](unlearning_evaluation_through_subset_statistical_independence.md)**

:   提出 Split-half Dependence Evaluation (SDE)，利用 HSIC 统计独立性检验在子集级别评估机器遗忘效果，无需重训模型或辅助分类器。

**[VeriTrail: Closed-Domain Hallucination Detection with Traceability](veritrail_closed-domain_hallucination_detection_with_traceability.md)**

:   提出 VeriTrail，首个面向多步生成（MGS）过程的闭域幻觉检测方法，通过将生成过程建模为 DAG 并沿图逐层验证 claim，实现了幻觉检测+溯源（provenance）+错误定位（error localization）的完整可追溯性，在两个新数据集上显著优于所有基线。
