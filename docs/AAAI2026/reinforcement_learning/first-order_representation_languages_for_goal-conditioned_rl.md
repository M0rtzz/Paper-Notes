---
title: >-
  [论文解读] First-Order Representation Languages for Goal-Conditioned RL
description: >-
  [AAAI 2026][强化学习][目标条件强化学习] 本文研究一阶关系语言在目标条件强化学习（goal-conditioned RL）和泛化规划中的应用，提出将目标表示为原子集合的子集或提升版本，结合 HER 自动创建由简到难的目标课程，在大规模稀疏奖励规划问题上成功学习到泛化策略。
tags:
  - "AAAI 2026"
  - "强化学习"
  - "目标条件强化学习"
  - "一阶表示语言"
  - "Hindsight Experience Replay"
  - "泛化规划"
  - "课程学习"
---

# First-Order Representation Languages for Goal-Conditioned RL

**会议**: AAAI 2026  
**arXiv**: [2512.19355](https://arxiv.org/abs/2512.19355)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 目标条件强化学习, 一阶表示语言, Hindsight Experience Replay, 泛化规划, 课程学习

## 一句话总结

本文研究一阶关系语言在目标条件强化学习（goal-conditioned RL）和泛化规划中的应用，提出将目标表示为原子集合的子集或提升版本，结合 HER 自动创建由简到难的目标课程，在大规模稀疏奖励规划问题上成功学习到泛化策略。

## 研究背景与动机

**领域现状**：一阶关系语言在 MDP 规划和强化学习中有两个主要用途：（1）以紧凑形式描述 MDP；（2）表示和学习具有泛化能力的策略——这些策略不依赖于特定实例或状态空间。目标条件 RL 是其中一个重要方向，智能体需要学习一个可以到达任意目标状态的策略。

**现有痛点**：当训练实例规模较大且目标无法通过随机探索到达时，目标条件 RL 面临严峻挑战。Hindsight Experience Replay（HER）通过将失败轨迹重新标记为"到达了实际到达的状态"来缓解这一问题，但标准的 HER 在状态和目标用命题表示时，泛化能力有限——它难以从小实例的经验推广到大实例。

**核心矛盾**：泛化目标条件策略需要处理变化的状态空间和目标空间，标准的向量表示难以捕捉关系结构。一阶表示虽然更有表达力，但如何有效利用一阶表示来加速目标条件 RL 的学习是一个未解决的问题。

**本文目标**：研究当状态和目标用原子集合（sets of atoms）表示时，目标条件 RL 能否获得进一步的性能提升。

**切入角度**：作者提出三种目标表示方式：（1）目标为完整状态；（2）目标为原始目标的子集；（3）目标为子目标的提升版本（lifted version）。后两种表示通过自动创建更简单的目标来形成从易到难的课程，加速学习。

**核心 idea**：利用一阶原子表示的结构性，将复杂的目标自动分解为更简单的子目标，形成课程学习机制，使得在大规模稀疏奖励问题上可以有效学习泛化策略。

## 方法详解

### 整体框架

输入为以一阶原子集合表示的 MDP 状态和目标。智能体通过 HER 风格的训练学习目标条件策略。关键创新在于目标的表示和重标记方式——从完整状态目标到子目标再到提升子目标，逐步增加泛化能力和课程学习效果。

### 关键设计

1. **目标作为原子子集（Goal as Subgoals）**:

    - 功能：将复杂目标分解为更简单的子目标，降低学习难度。
    - 核心思路：原始目标是一组原子的合取（conjunction），如 $\{on(A,B), on(B,C), clear(A)\}$。将其分解为子集，如 $\{on(A,B)\}$，作为更简单的中间目标。在 HER 重标记时，不仅可以用实际到达的完整状态作为新目标，还可以用其满足的任何原子子集作为新目标，大幅增加有效训练数据。
    - 设计动机：在积木世界等组合问题中，完整目标可能涉及众多约束，从随机状态到完整目标的距离很大。子目标分解自然形成从简单到复杂的课程。

2. **提升子目标（Lifted Subgoals）**:

    - 功能：进一步增强泛化能力，使策略可以跨对象泛化。
    - 核心思路：将子目标中的具体对象替换为变量，如 $\{on(A,B)\}$ 提升为 $\{on(X,Y)\}$。这使得"把任何块放到另一个块上"成为一个通用目标，而非"把A放到B上"。提升后的子目标可以被任何满足该模式的状态匹配，进一步增加了 HER 重标记的有效训练数据量。
    - 设计动机：在泛化规划中，关键挑战是从小实例推广到大实例。提升目标消除了对象特定性，使得策略学到的是关系模式而非特定对象之间的操作。

3. **自动课程生成机制**:

    - 功能：从简单目标开始训练，逐步增加复杂度。
    - 核心思路：训练初期主要重标记为简单的单原子或二原子子目标，策略先学会完成简单任务。随着训练进行，逐步增加子目标的复杂度，直到能完成完整的原始目标。这个课程是自动产生的——通过 HER 重标记机制和子目标分解的嵌套结构自然形成。
    - 设计动机：稀疏奖励的大规模规划问题中，直接学习完整目标几乎不可能成功。课程学习通过梯度渐进的方式降低了探索难度。

### 损失函数 / 训练策略

采用标准的目标条件 RL 训练框架（如 DQN 或 PPO 的目标条件版本），使用稀疏二值奖励（达到目标为+1，否则为0）。HER 重标记策略扩展为多级重标记——同时使用完整状态、子集和提升子集作为虚拟目标。

## 实验关键数据

### 主实验

在多个规划域（如 Blocksworld、Logistics 等）的大实例上评估。

| 方法 | 大实例成功率 | 泛化能力 | 数据效率 | 说明 |
|------|------------|---------|---------|------|
| 标准 RL | 极低 | 无 | 差 | 随机探索无法到达目标 |
| HER（完整目标） | 中等 | 有限 | 中等 | 标准HER基线 |
| HER + 子集目标 | 高 | 良好 | 好 | 子目标分解效果显著 |
| HER + 提升子目标 | 最高 | 最强 | 最好 | 跨对象泛化+课程效果最佳 |

### 消融实验

| 配置 | 成功率 | 说明 |
|------|--------|------|
| 完整目标 (baseline) | 低 | 目标太复杂难以学习 |
| 子集目标 | 中高 | 自动课程有效 |
| 提升子目标 | 最高 | 泛化+课程双重增益 |
| 随机课程 | 低 | 结构化课程远优于随机 |

### 关键发现

- 子集目标和提升子目标版本成功地在大规模规划实例上学到了泛化策略，而标准方法完全失败——这证明了一阶表示在目标条件 RL 中的实际价值。
- 自动课程生成是成功的关键——从简单子目标开始训练避免了稀疏奖励探索的瓶颈。
- 提升目标的优势在于跨对象泛化——相同的关系模式可以应用于任何具体对象实例。
- 论文也诚实地讨论了方法的局限性和改进方向。

## 亮点与洞察

- **原子子集作为自动课程**的思路非常自然而优雅——不需要人工设计课程，目标的组合结构本身就定义了从简单到复杂的层次。
- **一阶提升的泛化增益**证明了符号 AI 和深度 RL 结合的价值——关系结构的显式建模确实能带来命题表示无法获得的泛化能力。
- HER 在一阶表示下获得了"免费"的数据增强效果——同一条轨迹可以被重标记为更多的有效训练样本。

## 局限与展望

- 方法假设状态和目标可以用原子集合精确表示，不直接适用于连续或像素级观测。
- 提升目标需要对象类型系统的支持，在类型不明确的域中可能不适用。
- 学到的策略可能在非常大的实例上仍有泛化限制。
- 可以与图神经网络结合，用 GNN 直接处理一阶关系结构，增强表征能力。

## 相关工作与启发

- **vs 标准 HER**: 标准 HER 在命题空间中重标记，本文在一阶原子空间中重标记，获得了结构化的课程效应。
- **vs STRIPS 规划器**: 经典规划器可以精确求解但不能学习和泛化。本文的方法结合了规划的结构性和 RL 的学习能力。
- **vs 课程学习方法**: 大多数课程学习需要人工设计课程。本文通过目标分解自动产生课程，更加通用。

## 评分

- 新颖性: ⭐⭐⭐⭐ 一阶表示+HER+自动课程的组合方案新颖
- 实验充分度: ⭐⭐⭐⭐ 多个域上验证了三种变体的效果和局限
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，递进式阐述三种方法
- 价值: ⭐⭐⭐⭐ 为符号AI与深度RL的结合提供了有效范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Do It for HER: First-Order Temporal Logic Reward Specification in Reinforcement Learning](do_it_for_her_first-order_temporal_logic_reward_specification_in_reinforcement_l.md)
- [\[ICML 2026\] Latent Representation Alignment for Offline Goal-Conditioned Reinforcement Learning](../../ICML2026/reinforcement_learning/latent_representation_alignment_for_offline_goal-conditioned_reinforcement_learn.md)
- [\[ICML 2026\] Compositional Transduction with Latent Analogies for Offline Goal-Conditioned Reinforcement Learning](../../ICML2026/reinforcement_learning/compositional_transduction_with_latent_analogies_for_offline_goal-conditioned_re.md)
- [\[AAAI 2026\] ReGal: A First Look at PPO-based Legal AI for Judgment Prediction and Summarization in India](regal_a_first_look_at_ppo-based_legal_ai_for_judgment_prediction_and_summarizati.md)
- [\[ICML 2026\] FAB: A First-Order AB-based Gradient Algorithm for Distributed Bilevel Optimization over Time-Varying Directed Graphs](../../ICML2026/reinforcement_learning/fab_a_first-order_ab-based_gradient_algorithm_for_distributed_bilevel_optimizati.md)

</div>

<!-- RELATED:END -->
