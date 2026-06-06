---
title: >-
  [论文解读] AgentRM: Enhancing Agent Generalization with Reward Modeling
description: >-
  [ACL 2025][LLM对齐][agent] 提出 AgentRM，一个可泛化的奖励模型，通过显式/隐式/LLM-as-Judge 三种方式构建，用测试时搜索（Best-of-N / Beam Search）引导策略模型…
tags:
  - "ACL 2025"
  - "LLM对齐"
  - "agent"
  - "reward model"
  - "generalization"
  - "test-time search"
  - "MCTS"
  - "Best-of-N"
---

# AgentRM: Enhancing Agent Generalization with Reward Modeling

**会议**: ACL 2025  
**arXiv**: [2502.18407](https://arxiv.org/abs/2502.18407)  
**代码**: -  
**领域**: LLM Alignment  
**关键词**: agent, reward model, generalization, test-time search, MCTS, Best-of-N  

## 一句话总结

提出 AgentRM，一个可泛化的奖励模型，通过显式/隐式/LLM-as-Judge 三种方式构建，用测试时搜索（Best-of-N / Beam Search）引导策略模型，在 9 个 Agent 任务上平均提升 8.8 分并超越最佳通用 Agent 4.0 分。

## 研究背景与动机

- **现有问题**：基于 LLM 的 Agent 在训练时见过的任务上表现良好，但在未见任务上泛化能力差。已有工作通过扩大训练任务多样性来微调策略模型，但微调策略模型会增大已见动作 token 的概率同时降低未见动作的概率，导致在 held-out 任务上性能下降。
- **关键发现**：微调奖励模型比直接微调策略模型更鲁棒——在单个任务上微调策略模型仅改善该任务而降低其他任务性能（对角线正值），而微调奖励模型在单个任务上即可提升大部分未见任务的性能。
- **原因分析**：奖励函数的回归训练目标本质上对动作 token 具体分布不敏感，因此不会像策略微调那样过度偏向训练任务的动作空间。
- **本文方案**：提出 AgentRM，系统研究三种奖励建模方法，并在测试时通过 Best-of-N 采样和步级 Beam Search 引导策略模型做出更好的决策。

## 方法详解

### 整体框架

AgentRM 的流程分四步：(1) **行为克隆**：在专家轨迹上 SFT 得到初始策略模型；(2) **搜索树构建**：用 SFT 策略模型在训练任务环境中构建 MCTS 搜索树；(3) **奖励模型训练**：从搜索树中提取状态-奖励对训练泛化奖励模型；(4) **测试时搜索**：在未见任务上用奖励模型引导策略模型（Best-of-N 或 Beam Search）。

### 关键设计

- **显式奖励建模 (Explicit RM)**：使用 MCTS 启发式搜索构建搜索树，通过蒙特卡洛模拟估计每个状态的 Q 值 $V(s_t)$，训练语言模型 + 值头最小化 MSE 损失 $\mathcal{L}(\theta) = \frac{1}{N}\sum_{t=1}^{N}(\hat{V}(s_t) - V(s_t))^2$。搜索树使用 UCB 选择节点、动作合并减少冗余探索、模拟节点缓存加速搜索。
- **隐式奖励建模 (Implicit RM)**：基于 DPO 范式，从训练于结果奖励的模型中导出过程奖励 $r_\theta^t = \beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{ref}(a_t|s_t)}$，无需显式标注步级奖励。
- **步级 Beam Search**：每步采样 $W_1 \times W_2$ 个候选动作，用奖励模型评分后保留 top-$W_1$，对每个保留状态再扩展 $W_2$ 个动作，迭代直到终止。

### 损失函数

- Explicit RM：MSE 损失学习 Q 值。
- Implicit RM：MSE 损失拟合环境提供的进度标量奖励。
- 策略模型 SFT：标准自回归交叉熵损失。

## 实验

### 主实验结果（与通用 Agent 对比，LLaMA-3-8B 策略模型）

| 方法 | Web | Embodied | Text Game | Tool | 总体 |
|------|-----|----------|-----------|------|------|
| GPT-4o | 57.7 | 73.6 | 59.9 | 49.7 | 65.9 |
| AgentGym | 68.5* | 62.2* | 28.5 | 55.3* | 59.3* |
| Greedy Search | 57.8 | 50.6 | 37.4 | 56.6 | 52.7 |
| Best-of-5 (Explicit RM) | 62.4 | 62.7 | 47.8 | 68.7 | **61.5** |
| Beam Search (Explicit RM) | 64.4 | 65.1 | 47.5 | 64.0 | **63.3** |

*\* 表示训练时见过的任务*

### 消融实验

| 分析维度 | 关键发现 |
|----------|----------|
| 三种 RM 对比 | Explicit RM 最优（+8.8），Implicit RM 次之（+2.0），LLM-as-Judge 反而下降（-0.6） |
| 鲁棒性测试 | 对 Alfworld 进行 5 种扰动后，AgentGym 下降 25.6、Agent-FLAN 下降 30.3，而 AgentRM 仅下降 2.1，标准差最低 |
| 弱到强泛化 | 用 LLaMA-3-8B 采样训练的 RM 直接应用于 LLaMA-3-70B，提升 12.6 分 |
| 训练数据缩放 | 仅 4K 状态即可超越 LLM-as-Judge（57.6 vs 52.1），性能随数据增长呈对数线性增长 |
| 状态表示消融 | 主要依赖动作 token，同时去除思考和观察 token 性能下降 3.2 分 |

### 关键发现

1. Explicit RM 在所有设置下一致最优，Beam Search 进一步带来提升（总体 63.3 vs Best-of-5 的 61.5）。
2. AgentRM 展现出显著的弱到强泛化能力——用弱模型（8B）采样训练的 RM 对强模型（70B）带来更大提升（+12.6 vs +8.8）。
3. 已有通用 Agent（AgentGym、Agent-FLAN）存在严重过拟合——简单动作扰动就导致性能暴跌（最多-30.3），而 AgentRM 保持稳定。
4. 在专用任务上，AgentRM + Beam Search 超越最佳专用 Agent（QLASS）11.4 分。

## 亮点

- 揭示了"微调奖励模型比微调策略模型对泛化更鲁棒"这一核心洞察，有清晰的实验可视化支持。
- 弱到强泛化的发现具有实践价值——用小模型的经验提升大模型的决策质量。
- 系统比较三种奖励建模范式在 Agent 场景下的有效性，填补了该领域的空白。
- 扰动测试揭示已有通用 Agent 实际上在"记忆"而非"理解"任务。

## 局限性

- MCTS 搜索树构建需要与环境交互，对于无法重置的真实环境不适用。
- 训练数据仅来自 3 个 held-in 任务（Webshop、Alfworld、Sciworld），任务多样性有限。
- Implicit RM 和 LLM-as-Judge 的改进幅度较小，这两种方法在 Agent 场景的潜力可能需要更多研究。
- 奖励模型仅用 LLaMA-3-8B 训练，在更大规模 RM 上的收益未探索。
- Beam Search 的计算成本随 $W_1 \times W_2$ 远高于 Best-of-N，实际部署需权衡。

## 相关工作

- **专用 Agent**：SPIN、NAT、ETO、StepAgent、QLASS——每个任务单独训练，缺乏跨任务泛化。
- **通用 Agent**：Agent-FLAN、AgentGym、AgentGen——通过多任务微调策略模型但仍过拟合。
- **奖励建模**：Process Reward Model（数学推理领域成熟）、DPO 隐式奖励、LLM-as-Judge——首次系统应用于 Agent 任务。
- **测试时搜索**：Best-of-N、Beam Search、MCTS——从数学推理到 Agent 决策的迁移。

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 实用性 | 5 |
| 实验充分性 | 5 |
| 写作质量 | 4 |
| 总评 | 4.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Reward Generalization in RLHF: A Topological Perspective](reward_generalization_in_rlhf_a_topological_perspective.md)
- [\[ACL 2025\] Dynamic Scaling of Unit Tests for Code Reward Modeling](dynamic_scaling_of_unit_tests_for_code_reward_modeling.md)
- [\[ACL 2025\] Think&Cite: Improving Attributed Text Generation with Self-Guided Tree Search and Progress Reward Modeling](think_cite_attributed_text_gen.md)
- [\[ACL 2025\] Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent](expectation_confirmation_preference_optimization_for_multi-turn_conversational_r.md)
- [\[ACL 2025\] FocalPO: Enhancing Preference Optimizing by Focusing on Correct Preference Rankings](focalpo_enhancing_preference_optimizing_by_focusing_on_correct_preference_rankin.md)

</div>

<!-- RELATED:END -->
