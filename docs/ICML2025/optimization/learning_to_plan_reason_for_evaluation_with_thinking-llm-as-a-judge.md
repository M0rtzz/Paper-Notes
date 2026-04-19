---
title: >-
  [论文解读] Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge
description: >-
  [ICML 2025][优化][LLM-as-a-Judge] 提出 EvalPlanner，通过将 LLM-as-a-Judge 的推理过程解耦为"评估计划生成"和"计划执行"两个阶段，并在自训练循环中用 DPO 迭代优化计划与执行的偏好对，在 RewardBench 上以仅 22K 合成偏好对达到 93.9 的生成式奖励模型新 SOTA。
tags:
  - ICML 2025
  - 优化
  - LLM-as-a-Judge
  - 偏好优化
  - 链式思维
  - 评估规划
  - 自训练
---

# Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge

**会议**: ICML 2025  
**arXiv**: [2501.18099](https://arxiv.org/abs/2501.18099)  
**代码**: 无  
**领域**: 优化  
**关键词**: LLM-as-a-Judge, 偏好优化, 链式思维, 评估规划, 自训练

## 一句话总结

提出 EvalPlanner，通过将 LLM-as-a-Judge 的推理过程解耦为"评估计划生成"和"计划执行"两个阶段，并在自训练循环中用 DPO 迭代优化计划与执行的偏好对，在 RewardBench 上以仅 22K 合成偏好对达到 93.9 的生成式奖励模型新 SOTA。

## 研究背景与动机

LLM-as-a-Judge 范式用 LLM 自身作为评估器来替代昂贵的人工评估，通过生成链式思维（CoT）来提供评估理由。这类模型也可作为生成式奖励模型，在迭代偏好优化和自改进训练中发挥关键作用。

然而，现有方法面临两个核心问题：

**缺乏人工标注的评估 CoT**：人类偏好标注数据通常只包含最终判断，不包含推理过程，导致有效推理链的结构和组成成分研究不足。

**评估推理被约束为手工设计的组件**：先前工作通常将 CoT 限制为预定义的评估准则列表、参考答案或验证问题，无法自适应地处理不同类型任务（如作文 vs. 数学题的评估标准截然不同）。

**规划与推理纠缠**：现有方法将"确定评估标准"和"执行评估"混合在同一个生成过程中，缺乏清晰的阶段分离。

这些局限导致评估模型在面对多样化、复杂指令时泛化能力不足。EvalPlanner 的核心动机是：**评估本质上是规划+推理问题**——先确定评估方案（plan），再按方案逐步执行评估（execute），最后给出判断（verdict）。

## 方法详解

### 整体框架

EvalPlanner 将 Thinking-LLM-as-a-Judge 的 CoT 显式解耦为三个组件：

1. **评估计划 $z$**：仅基于输入指令 $x$ 生成的评估方案（不看候选回答），规定了评估的"食谱"——需要检查哪些维度、使用什么标准等。
2. **计划执行 $e$**：在给定计划 $z$、指令 $x$ 和回答对 $(a, b)$ 的条件下，逐步执行计划中的每个评估步骤，分析两个回答的质量。
3. **最终判断 $y$**：基于执行结果输出哪个回答更好。

形式化地，生成判断 $y$ 的过程建模为：

$$p_\theta(y|x,a,b) = \sum_{z \in \mathcal{P}} \sum_{e \in \mathcal{E}} p_\theta(y|e,z,x,a,b) \cdot p_\theta(e|z,x,a,b) \cdot p_\theta(z|x)$$

关键设计选择：**计划生成仅条件于指令 $x$，不依赖回答对**，确保计划只描述评估方案而非执行实际评估，从而实现阶段分离。

### 关键设计

#### 1. 合成训练数据生成

由于缺乏人类标注的 CoT，EvalPlanner 完全基于合成数据训练：

- **提示选择**：从 WildChat（通用指令跟随）和 MATH（数学推理）中选取指令。
- **回答对构造**：
    - 通用任务：将原始指令修改为"噪声指令"，对原始/噪声指令分别生成回答，构成 chosen/rejected 对。
    - 数学任务：采样多个解，正确答案为 chosen、错误答案为 rejected。
- **计划生成**：使用通用、无约束的 plan 生成提示，让种子模型（如 Llama-3.1-70B-Instruct）根据输入指令自由生成评估计划，不预设任何组件结构。
- **执行生成**：给定计划和回答对，再次由种子模型执行计划生成评估推理和判断。

#### 2. 偏好对构建

对每条指令：
- 采样 $|\mathcal{P}|=5$ 个计划
- 每个计划采样 $|\mathcal{E}|=8$ 次执行（每种回答顺序 4 次，共处理 $(a,b)$ 和 $(b,a)$ 两种顺序以消除位置偏差）
- 总共产生 $2 \times 5 \times 8 = 80$ 个 CoT

正确性判定：若 (plan, execution, verdict) 三元组导出正确判断，则为 chosen；否则为 rejected。对每个计划，将所有正确/错误执行配对，构建偏好训练数据。

#### 3. 无约束计划 vs. 约束计划

核心创新之一：EvalPlanner 使用通用的无约束 plan 生成提示，让模型自主决定评估维度和方法，而非预定义"必须评估准则列表"或"必须生成验证问题"。实验表明无约束计划在所有设置中均优于约束计划。

#### 4. 规划与执行的解耦优势

- **执行忠实性**：执行阶段被约束为遵循计划，提高一致性。
- **数据多样性**：同一指令可采样多种计划，每种计划又可采样多种执行，训练数据在规划和执行两个维度上多样化。

### 损失函数 / 训练策略

EvalPlanner 采用三阶段自训练循环：

**阶段 1：SFT（$\mathcal{M}_1^{\text{SFT}}$）**
- 从种子模型 $\mathcal{M}_0$ 出发
- 在 5K 指令上，每条随机选一个 chosen CoT 做监督微调
- 目的是让模型学会 plan + execution + verdict 的输出格式

**阶段 2：第一轮 DPO（$\mathcal{M}_1^{\text{DPO}}$）**
- 从 $\mathcal{M}_1^{\text{SFT}}$ 初始化
- 在 $\mathcal{D}_1$（5K 指令产生的偏好对）上做 DPO
- 模型学习对比正确与错误的 (plan, execution) 组合

**阶段 3：第二轮 DPO（$\mathcal{M}_2^{\text{DPO}}$）**
- 从 $\mathcal{M}_1^{\text{DPO}}$ 初始化
- 使用新的 17K 指令子集，从 $\mathcal{M}_1^{\text{DPO}}$ 自身采样生成新的 CoT 偏好对 $\mathcal{D}_2$
- 再做一轮 DPO

关键训练参数：
- 最大训练步数 1K，每 100 步存一次 checkpoint，基于验证集做 early stopping
- 采样温度 0.8，top_p 0.95
- 验证集：WildChat 和 MATH 各 150 条，双向排列共 600 条
- 推理温度 0，最大生成 2048 tokens

迭代 DPO 的优势：第二轮使用更新模型产生的、更高质量的 CoT 数据训练，比一次性用所有数据训练效果更好。

## 实验关键数据

### 主实验

**RewardBench 结果（Table 1）**：

| 模型 | 偏好对数 | Overall | Chat | Chat-Hard | Safety | Reasoning |
|------|---------|---------|------|-----------|--------|-----------|
| Llama-3.1-70B-Instruct | - | 84.0 | 97.2 | 70.2 | 82.8 | 86.0 |
| GPT-4o | - | 86.7 | 96.1 | 76.1 | 88.1 | 86.6 |
| Self-Taught Evaluator | 20K | 90.0 | 96.9 | 85.1 | 89.6 | 88.4 |
| Skywork-Critic-70B | 80K | 93.3 | 96.6 | 87.9 | 93.1 | 95.5 |
| **EvalPlanner (3.1-70B)** | **22K** | **93.9** | **97.5** | **89.4** | **93.0** | **95.5** |
| **EvalPlanner (3.3-70B)** | **22K** | **93.8** | **97.7** | **89.5** | **91.7** | **96.1** |

**PPE 结果（Table 2）**：

| 模型 | PPE Overall | PPE Preference | PPE Correctness Overall |
|------|-------------|----------------|------------------------|
| GPT-4o | 62.3 | 67.1 | 57.6 |
| DeepSeek-GRM-27B (237K) | 62.2 | 64.7 | 59.8 |
| **EvalPlanner (3.3-70B, 22K)** | **67.9** | **65.6** | **70.2** |

### 消融实验

**迭代 DPO 的效果（Table 4）**：

| 配置 | 偏好对数 | Accuracy | 说明 |
|------|---------|----------|------|
| 1 轮 DPO | 5K | 92.3 | 基线 |
| 1 轮 DPO | 22K | 92.5 | 数据翻倍仅 +0.2 |
| 2 轮 DPO（迭代） | 5K+17K | **93.9** | 迭代优化显著 +1.6 |

**FollowBenchEval 多约束评估（Table 5）**：

| 模型 | Overall | L1 | L2 | L3 | L4 | L5 |
|------|---------|-----|-----|-----|-----|-----|
| Skywork-Critic-70B | 52.2 | 63.8 | 57.1 | 48.7 | 46.2 | 48.5 |
| **EvalPlanner (3.3-70B)** | **65.4** | **72.3** | **73.8** | **66.7** | **61.5** | **57.6** |

**RM-Bench 鲁棒性（Table 6）**：

| 模型 | Overall | Easy | Normal | Hard |
|------|---------|------|--------|------|
| Skywork-Critic-70B | 74.1 | 76.3 | 72.9 | 73.1 |
| **EvalPlanner (3.3-70B)** | **82.1** | **81.1** | **80.8** | **84.3** |

### 关键发现

1. **数据高效**：仅 5K 合成偏好对即可达到 92.3（已接近 SOTA），22K 合成数据超越使用 80K-680K 人工标注数据的先前方法。
2. **迭代优化关键**：两轮 DPO 比一轮 DPO+更多数据效果好，说明用更新模型生成的新数据比历史数据更有价值。
3. **小模型也受益**：8B 版本的 EvalPlanner 在 RewardBench 上达到 83.0，接近 Llama-3.1-70B-Instruct 的 84.0 和 Claude-3.5-Sonnet 的 84.2。
4. **多约束评估优势显著**：在 FollowBenchEval 上超越 Skywork-Critic 13 个点，说明显式规划对需要逐项检查的复杂约束任务特别有效。
5. **无约束计划优于约束计划**：通用 plan 提示比预定义"准则列表"或"验证问题"形式泛化更好。
6. **鲁棒性突出**：在 RM-Bench Hard 子集上，EvalPlanner 不降反升（84.3 vs. Overall 82.1），而其他模型在 Hard 上普遍下降。

## 亮点与洞察

1. **"评估即规划+推理"** 的问题定义非常精准——将评估任务类比为先写评审方案再执行评审，符合人类专家的评审思维模式。
2. **规划不看答案** 是一个巧妙的设计约束：仅基于指令生成评估计划，确保计划的通用性和客观性，避免了计划被特定回答内容污染。
3. **合成数据驱动** 的自训练循环极具实用价值：完全不需要人工标注的推理链，仅需少量 (指令, chosen, rejected) 三元组即可引导。
4. **位置偏差消除** 通过双向排列回答对的方式，简单有效。
5. **迭代 DPO > 大批量 DPO** 的发现对偏好优化领域有普遍启示意义：用更新模型生成训练数据的"on-policy"特性比单纯增加数据量更重要。

## 局限与展望

1. **仅验证了两轮迭代**：更多轮次的迭代 DPO 是否持续改进尚未探索。
2. **计划与执行仍为串行生成**：两阶段增加了推理延迟和 token 消耗（最大 2048 tokens），对效率有影响。
3. **仅有 pairwise 评估**：未扩展到 pointwise 评分或 ranking 多个回答的场景。
4. **JudgeBench 上未超越 Skywork-Critic**：在需要深度领域知识的高难度对比中仍有提升空间。
5. **计划质量评估缺乏**：目前仅通过最终判断正确性间接评估计划质量，缺乏对计划本身的直接评价指标。
6. **可扩展到更强的种子模型**：当前仅测试了 Llama 70B 级别，GPT-4o 或更大模型作为种子可能带来更大提升。

## 相关工作与启发

- **Self-Taught Evaluators (Wang et al., 2024c)**：同样使用合成数据和自训练，但将评估 CoT 约束为"准则列表"形式，EvalPlanner 的无约束计划和显式解耦提供了更好的灵活性。
- **Skywork-Critic (Shiwen et al., 2024)**：使用 80K 人工标注数据训练的强基线，EvalPlanner 用更少的合成数据即超越之。
- **Chain-of-Verification (Dhuliawala et al., 2023)**：将推理约束为验证问题列表，EvalPlanner 的消融表明无约束形式更优。
- **DeepSeek-GRM (Liu et al., 2025)**：在 PPE 上的强基线，使用 MetaRM 投票策略，EvalPlanner 在 PPE Correctness 上大幅超越。
- **对研究启发**：plan-then-execute 的范式可推广到其他需要结构化推理的任务，如代码审查、文档评估、对话质量评判等。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 计划-执行解耦和无约束计划生成是新颖贡献 |
| 技术深度 | 4 | 形式化完整，迭代自训练设计精细 |
| 实验充分度 | 5 | 5 个 benchmark，全面消融，跨种子模型验证 |
| 实用性 | 4 | 完全合成数据驱动，可直接复现 |
| 写作质量 | 4 | 结构清晰，动机阐述有说服力 |
| **总分** | **4.2** | 扎实的系统性工作，对 LLM-as-a-Judge 领域有明确推进 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization](../../ICLR2026/optimization/frontierco_real-world_and_large-scale_evaluation_of_machine_learning_solvers_for.md)
- [ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting](../../ACL2025/optimization/scalebio_bilevel_data_reweighting.md)
- [DartQuant: Efficient Rotational Distribution Calibration for LLM Quantization](../../NeurIPS2025/optimization/dartquant_efficient_rotational_distribution_calibration_for_llm_quantization.md)
- [MeCeFO: Enhancing LLM Training Robustness via Fault-Tolerant Optimization](../../NeurIPS2025/optimization/mecefo_enhancing_llm_training_robustness_via_fault-tolerant_optimization.md)
- [A Generalization Result for Convergence in Learning-to-Optimize](a_generalization_result_for_convergence_in_learning-to-optimize.md)

<!-- RELATED:END -->
