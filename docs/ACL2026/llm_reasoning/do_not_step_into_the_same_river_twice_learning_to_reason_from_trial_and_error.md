---
title: >-
  [论文解读] Do Not Step Into the Same River Twice: Learning to Reason from Trial and Error
description: >-
  [ACL 2026][LLM推理][探索停滞] 提出 LTE (Learning to reason from Trial and Error)，通过将模型自身生成的错误答案作为提示引导额外 rollout，在不依赖外部专家的情况下有效缓解 RLVR 中的探索停滞问题。
tags:
  - ACL 2026
  - LLM推理
  - 探索停滞
  - 试错学习
  - 强化学习
  - 提示引导探索
  - 数学推理
---

# Do Not Step Into the Same River Twice: Learning to Reason from Trial and Error

**会议**: ACL 2026  
**arXiv**: [2510.26109](https://arxiv.org/abs/2510.26109)  
**代码**: [GitHub](https://github.com/JamyDon/LTE)  
**领域**: LLM Reasoning / Reinforcement Learning  
**关键词**: 探索停滞, 试错学习, 强化学习, 提示引导探索, 数学推理

## 一句话总结

提出 LTE (Learning to reason from Trial and Error)，通过将模型自身生成的错误答案作为提示引导额外 rollout，在不依赖外部专家的情况下有效缓解 RLVR 中的探索停滞问题。

## 研究背景与动机

**领域现状**：RLVR（基于可验证奖励的强化学习）是提升 LLM 推理能力的核心范式，GRPO 是其事实标准算法。然而，RLVR 训练中存在严重的**探索停滞**(exploration stagnation)问题——当训练样本对模型过于困难时，所有 rollout 都无法生成正确答案，导致模型从这些样本中获得零梯度信号，永远无法突破自身能力上限。

**现有痛点**：已有方法试图通过引入外部引导来突破探索停滞：(1) 利用人工标注的标准解题过程，成本高且不可扩展；(2) 使用更强 LRM 生成的推理链（如 LUFFY），但更强模型并非总是可获取（如训练旗舰模型时）。这些方法更像是"治标不治本"的权宜之计。

**核心矛盾**：在标准 GRPO 的 0/1 奖励设定下，当所有 $G$ 个 rollout 都失败（none-pass samples）时，所有 group-relative advantage 退化为零（$\hat{A}_{i,t}=0$），梯度为零，策略完全无法从这些样本中学习。简单增加 rollout 数量（GRPOExtra）效果甚微，因为在相同策略分布下重复采样并不能带来本质突破。

**本文目标**：设计一个完全不依赖外部专家引导的方法来缓解 RLVR 中的探索停滞。

**核心idea**：类比人类学习——当学生被告知之前犯过的错误后，会避开这些错误，从而更有可能找到正确答案。LTE 收集模型在初始 rollout 中生成的错误答案，将其作为"不要重蹈覆辙"的提示，引导模型在额外 rollout 中探索不同的解题路径。

## 方法详解

### 整体框架

LTE 的核心流程：对每个训练 prompt，先执行 $G$ 个标准 rollout。如果全部失败（none-pass），则根据失败模式选择合适的提示模板，执行 $G$ 个额外的提示引导 rollout。如果额外 rollout 中有正确答案，则随机替换初始失败的 rollout，并通过混合策略 GRPO 进行策略更新。整个过程仅依赖模型自身的行为信息。

### 关键设计

1. **提示引导的额外 Rollout(Hinted Extra Rollouts)**:

    - 功能：为 none-pass 样本提供基于自身错误的引导性提示，增加生成正确答案的概率
    - 核心思路：根据初始 rollout 的失败模式分三种情况处理：(a) **全部截断**(all-truncated)：所有回复因过长被截断，提示模型"简洁思考"；(b) **部分截断**(some-truncated)：提取未截断回复中的错误答案作为提示，同时要求简洁；(c) **无截断**(none-truncated)：纯粹提取错误答案集合 $\{a_1, a_2, ...\}$ 作为提示。模型被指示不要明确提及或使用这些提示，以保持推理链的"干净"
    - 设计动机：错误答案缩小了解空间，告诉模型"这些都是错的，别再走这条路"，类似人类在试错后的学习过程

2. **混合策略优化(Mixed-Policy Optimization)**:

    - 功能：将提示引导生成的正确答案以 off-policy 方式纳入策略更新
    - 核心思路：如果额外 rollout 中有 $G'$ 个正确答案 $\{o'_1, ..., o'_{G'}\}$，随机替换初始失败 rollout 中的等量样本。由于正确答案是在提示条件下生成的，定义提示策略 $\hat{\pi}(\cdot|\cdot) = \pi_{\mathrm{old}}(\cdot|\mathcal{H}_q, \cdot)$，重要性采样比率调整为 $\hat{r}'_{i,t}(\theta) = \frac{\pi_\theta(o'_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o'_{i,t}|\mathcal{H}_q, q, o_{i,<t})}$
    - 设计动机：提示引导轨迹来自不同的输入条件（带提示 vs 不带提示），需要以 off-policy 方式处理，使用正则化重要性采样 $f(r) = r/(r+\gamma)$ 保证稳定性

3. **熵控制增强变体(LTE†)**:

    - 功能：通过添加熵损失进一步促进探索
    - 核心思路：在 LTE 基础上增加熵正则化项，鼓励策略保持多样性。这使得 LTE† 能与 DAPO、LUFFY 等使用熵控制的方法进行公平比较
    - 设计动机：LTE 本身通过提示引导隐式地促进探索，但显式的熵控制可以进一步释放模型的探索潜力

### 损失函数 / 训练策略

LTE 使用混合策略 GRPO 目标函数，包含 on-policy 部分（标准 GRPO 的 clip 损失）和 off-policy 部分（对提示引导轨迹使用正则化重要性采样）。所有 rollout 共同计算 group-relative advantage。训练使用 verl 框架，每个 prompt 采样 8 个 rollout，温度 1.0，最大回复长度 8192，训练 500 步。

## 实验关键数据

### 主实验 (Pass@1)

| 模型 | 方法 | MATH-500 | AIME24 | AIME25 | Avg. |
|------|------|----------|--------|--------|------|
| Qwen3-8B-Base | GRPO | 77.70 | 22.08 | 15.83 | 43.99 |
| Qwen3-8B-Base | GRPOExtra | 75.75 | 22.29 | 14.79 | 41.72 |
| Qwen3-8B-Base | LTE | 78.85 | 30.62 | 27.29 | 49.01 |
| Qwen3-8B-Base | LUFFY | 80.20 | 29.17 | 21.25 | 47.15 |
| Qwen3-8B-Base | LTE† | 79.80 | 30.83 | 25.83 | 49.56 |

### Pass@k (探索上限)

| 模型 | 方法 | AIME24 | AIME25 | Avg. |
|------|------|--------|--------|------|
| Qwen3-8B-Base | GRPO | 43.33 | 30.00 | 56.82 |
| Qwen3-8B-Base | LTE | 70.00 | 46.67 | 66.78 |
| Qwen3-8B-Base | LUFFY | 60.00 | 33.33 | 61.55 |
| Qwen3-8B-Base | LTE† | 66.67 | 50.00 | 67.58 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| GRPOExtra vs LTE | -7.29 Pass@1, -10.04 Pass@k | 无提示的额外 rollout 效果极差 |
| LTE vs LUFFY (无熵控制) | +1.86 Pass@1 | 不依赖外部专家仍优于 LUFFY |
| LTE† vs LUFFY (有熵控制) | +2.41 Pass@1, +2.15 Pass@k | 加熵控制后优势更明显 |

### 关键发现
- LTE 在训练过程中持续减少 none-pass 样本数量，有效缓解了探索停滞
- LTE 维持更高的长尾熵值，鼓励测试时的深度思考（更长的回复长度）
- 简单增加 rollout 数量(GRPOExtra)甚至可能比标准 GRPO 更差，验证了"量变不能带来质变"
- LTE† 在 Pass@k 指标上尤其突出，说明 LTE 有效拓展了探索上限

## 亮点与洞察
- **"试错学习"的优雅类比**：将人类通过错误反馈改进的学习方式迁移到 RL 训练中，直觉简洁且有效
- **零外部依赖**：完全利用模型自身的行为信息（错误答案）作为引导，实际部署无门槛
- **三种失败模式的差异化处理**：根据截断情况分别设计提示模板，体现了对问题的深入理解
- **Pass@k 的巨大提升**：特别是在 AIME24 上从 43.33 提升到 70.00(Pass@k)，证明了 LTE 显著拓展了模型的探索上限
- **与 LUFFY 的有力对比**：不使用任何外部专家轨迹的 LTE† 竟然超越了依赖更强模型的 LUFFY

## 局限与展望
- 额外 rollout 引入了约 2 倍的采样开销（对 none-pass 样本），训练效率有待优化
- 仅在数学推理任务上验证，其他推理领域（代码、逻辑）的适用性未知
- 提示模板是手工设计的，自动化的提示生成策略可能带来进一步提升
- 部分基线（ReLIFT、EvoCoT）在 Qwen3 上表现不佳，可能存在模型兼容性问题
- 未来可探索更动态的提示策略（如利用更丰富的错误模式信息而非仅错误答案）

## 相关工作与启发
- **vs LUFFY**：LUFFY 使用更强 LRM 的推理轨迹作为 off-policy 样本，LTE 仅用自身错误答案作为提示，在两个 LM 上均超越 LUFFY
- **vs GRPOExtra**：简单增加 rollout 数量在 Qwen3-8B-Base 上甚至比 GRPO 更差(-2.27 Avg)，证明了提示引导的必要性
- **vs DAPO**：DAPO 通过 clip-higher 等技术改进优化算法本身，LTE 通过改进采样策略增强探索，两者正交且可互补
- **vs EvoCoT**：EvoCoT 使用 ground truth 答案作为提示，LTE 使用自身错误答案作为"反面教材"，思路完全不同且效果更好

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 利用模型自身错误作为探索引导的想法简洁而新颖，"不要踏入同一条河"的隐喻贴切
- 实验充分度: ⭐⭐⭐⭐ 多模型多基准评估，含 Pass@1 和 Pass@k，训练动态分析全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题定义精确，算法伪代码完整
- 价值: ⭐⭐⭐⭐⭐ 提供了一种零外部依赖的探索停滞解决方案，实用性极强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners](large_reasoning_models_are_not_yet_multilingual_latent_reasoners.md)
- [\[ICLR 2026\] From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics](../../ICLR2026/llm_reasoning/from_abstract_to_contextual_what_llms_still_cannot_do_in_math_word_problem_solvi.md)
- [\[ACL 2026\] ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering](recoqa_a_benchmark_for_tool-augmented_and_multi-step_reasoning_in_real_estate_qu.md)
- [\[ACL 2025\] Self-Error-Instruct: Generalizing from Errors for LLMs Mathematical Reasoning](../../ACL2025/llm_reasoning/self-error-instruct_generalizing_from_errors_for_llms_mathematical_reasoning.md)
- [\[ACL 2026\] Process Reward Models Meet Planning: Generating Precise and Scalable Datasets for Step-Level Rewards](process_reward_models_meet_planning_generating_precise_and_scalable_datasets_for.md)

</div>

<!-- RELATED:END -->
