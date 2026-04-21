---
title: >-
  [论文解读] Not All Explanations for Deep Learning Phenomena Are Equally Valuable
description: >-
  [ICML 2025][深度学习现象] 本文是一篇 position paper，主张深度学习中的"反直觉现象"（如 double descent、grokking、lottery ticket）在实际场景中很少出现，研究者不应追求对它们的孤立解释，而应将其作为检验和完善更广泛深度学习理论的实验场。
tags:
  - ICML 2025
  - 深度学习现象
  - 科学方法论
  - Double Descent
  - Grokking
  - Lottery Ticket Hypothesis
---

# Not All Explanations for Deep Learning Phenomena Are Equally Valuable

**会议**: ICML 2025  
**arXiv**: [2506.23286](https://arxiv.org/abs/2506.23286)  
**代码**: 无  
**领域**: 推荐系统  
**关键词**: 深度学习现象, 科学方法论, Double Descent, Grokking, Lottery Ticket Hypothesis

## 一句话总结

本文是一篇 position paper，主张深度学习中的"反直觉现象"（如 double descent、grokking、lottery ticket）在实际场景中很少出现，研究者不应追求对它们的孤立解释，而应将其作为检验和完善更广泛深度学习理论的实验场。

## 研究背景与动机

近年来，深度学习领域出现了一系列令人惊讶的经验现象：double descent（双下降）、grokking（延迟泛化）、lottery ticket hypothesis（彩票假说）等。这些现象看似违背了我们对神经网络行为的传统认知，因此引发了大量研究关注——三篇原始论文的总引用量已超过 7200 次，顶会中每年有上百篇论文提及它们。

然而，作者指出一个核心矛盾：**这些现象在实际深度学习应用中几乎不会出现**。Double descent 在使用恰当正则化时消失；grokking 仅在小型算法数据集上观察到，在大规模任务中无法复现；lottery ticket 虽然理论上存在，但至今无法在训练前有效识别。这意味着大量研究资源被投入到了对"边缘案例"（edge cases）的孤立解释上，而这些解释往往是 **narrow ad hoc hypotheses**——窄的事后假说，不具备泛化能力。

本文的核心 idea 是：深度学习现象的真正价值不在于"解决"或"解释"它们本身，而在于将它们作为**极端测试场景**，用来打磨我们对深度学习核心原理（如泛化、优化、稀疏性）的**广义解释理论**（broad explanatory theories）。

## 方法详解

### 整体框架

本文不是一篇提出新算法的工作，而是一篇方法论的 position paper。其核心论证框架分为三个层次：

1. **实证分析**：逐一审视 double descent、grokking 和 lottery ticket hypothesis 的实际相关性
2. **理论区分**：区分"窄事后假说"与"广义解释理论"两种研究范式
3. **实践建议**：提出面向未来研究的具体行动指南

### 关键设计

1. **"窄事后假说" vs "广义解释理论"的区分**: 作者构造了一个故意荒谬的反例来阐明这一区别——用"网络参数中质数的数量"来"解释" double descent 和 grokking。这个"理论"在数据上确实能追踪测试性能的变化，但显然不具备任何泛化价值。这说明**一个在特定现象上准确的解释，未必对更广泛的领域有用**。作者认为，真正有价值的研究应该是：利用现象来修正或检验我们对 bias-variance tradeoff、优化动力学、模型稀疏性等核心概念的理解。

2. **社会技术实用主义（Sociotechnical Pragmatism）框架**: 作者引入 Watson et al. (2024) 的框架，主张深度学习研究的价值应以其**下游影响**来衡量——这里的"影响"既包括技术进步，也包括社会层面的考量。在这个框架下，知识的价值取决于其实用性，"没有实际影响的理论不过是一个形式化练习"。作者发现，96% 的 ML 论文都声称以性能和泛化为目标，超过 50% 明确表示关注实际应用。

3. **科学方法论的倡导**: 作者主张深度学习现象研究应更严格地遵循科学方法，包括：假说驱动的研究、负结果报告、可证伪性、预注册（preregistration）、元研究与复现。这些做法在自然科学中已有悠久传统，但在深度学习领域尚未充分应用。

### 三大现象的具体分析

| 现象 | 实际不相关性 | 广义理论价值 |
|------|-------------|-------------|
| Double Descent | 在适当正则化下消失；不出现在 LLM/ViT 的 scaling 分析中 | 推动了对 bias-variance tradeoff、benign overfitting、memorization 的重新理解 |
| Grokking | 仅限小型算法数据集；大数据集上效果减弱；人为放大初始化可诱导 | 推动了对学习动力学、lazy-to-feature learning、Softmax 数值不稳定性的研究 |
| Lottery Ticket | 训练前无法高效识别；稀疏性优势在现代硬件上难以实现 | 影响了对剪枝、量化、参数高效微调的理解 |

## 实验关键数据

### 主实验

本文作为 position paper，不包含传统意义上的实验。但提供了以下量化证据：

| 指标 | 数据 |
|------|------|
| 三篇原始论文总引用量 | 7,272 次 (截至 2025.06) |
| NeurIPS 2024 主会相关论文 | 149 篇 |
| ICML 2024 主会相关论文 | 132 篇 |
| ICLR 2024 主会相关论文 | 108 篇 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| "质数参数"假说 | 与测试性能高度相关 | 说明"解释得准"不等于"有用" |
| Double descent + 正则化 | 现象消失 | 说明实际训练中不会出现 |
| Grokking + 大数据集 | 效果显著减弱 | 说明仅限边缘场景 |

### 关键发现

- 深度学习现象在实际应用中不构成实质性挑战，不应按"问题→解决方案"的模式研究
- 存在大量"窄事后假说"式的研究，它们虽然在特定设置下正确，但对推动领域进步贡献有限
- 这些现象的真正价值在于：提供了计算成本低、知识门槛低的极端设置，可用于检验和打磨我们对核心深度学习原理的理解

## 亮点与洞察

- **"质数参数"的反例构造**非常精彩，一个看似荒谬但在数据上"成立"的理论，直观展示了"准确的解释"和"有用的解释"之间的鸿沟
- 深度学习现象研究具有独特优势：计算资源需求低、知识门槛低、更接近科学探索而非 SOTA 竞争、理论与实验交叉的良好试验田
- 对整个社区的研究导向有反思价值：当一个现象不是实际问题时，追求"解决"它可能是在错误的方向上用力

## 局限与展望

- 作为 position paper，核心论点的评估具有较大主观性——如何量化"一个解释的广义理论价值"仍然模糊
- 对三个现象的分析虽有代表性，但不够全面（如未讨论 neural scaling laws、emergence 等更新的现象）
- 文中承认判断哪些理论有"预期效用"是困难的，但给出的实操指南仍偏抽象
- 未充分讨论"边缘现象"可能在未来随着新架构/任务的出现变得"不再边缘"的可能性

## 相关工作与启发

- 与 Schaeffer et al. (2024) 关于 LLM "emergent abilities" 的批判性分析思路一致
- 与 Karl et al. (2024) 关于深度学习研究中负结果报告的呼吁互补
- 启发思考：在推荐系统等应用领域，很多"反直觉现象"的研究是否也存在类似问题——解释了一个仅在极端设置下出现的行为，但对实际系统改进无帮助？

## 评分

- 新颖性: ⭐⭐⭐⭐ 观点清晰有洞察，但 position paper 本身不提出新方法
- 实验充分度: ⭐⭐⭐ 作为 position paper，定量证据以文献计量为主，"质数参数"反例构造巧妙
- 写作质量: ⭐⭐⭐⭐⭐ 论述逻辑严密，层层递进，图表清晰
- 价值: ⭐⭐⭐⭐ 对社区研究导向有重要反思价值，但实操落地仍有距离

<!-- RELATED:START -->

## 相关论文

- [LCRON: Learning Cascade Ranking as One Network](learning_cascade_ranking_as_one_network.md)
- [SIMPLEMIX: Frustratingly Simple Mixing of Off- and On-policy Data in Language Model Preference Learning](simplemix_frustratingly_simple_mixing_of_off-_and_on-policy_data_in_language_mod.md)
- [Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning](../../NeurIPS2025/recommender/transformer_copilot_learning_from_the_mistake_log_in_llm_fine-tuning.md)
- [MultiTab: A Scalable Foundation for Multitask Learning on Tabular Data](../../AAAI2026/recommender/multitab_a_scalable_foundation_for_multitask_learning_on_tabular_data.md)
- [Probabilistic Hash Embeddings for Online Learning of Categorical Features](../../AAAI2026/recommender/probabilistic_hash_embeddings_for_online_learning_of_categorical_features.md)

<!-- RELATED:END -->
