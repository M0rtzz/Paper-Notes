---
title: >-
  [论文解读] ReForm: Reflective Autoformalization with Prospective Bounded Sequence Optimization
description: >-
  [ICLR 2026][LLM推理][autoformalization] 提出 ReForm，一种反思式自动形式化范式，将自然语言数学问题转为 Lean 形式声明的过程从一次生成转变为"生成 → 语义自验证 → 修正"的迭代循环，并设计 PBSO 算法优化异构奖励信号，在四个基准上比最强基线平均提升 22.6 个百分点。
tags:
  - ICLR 2026
  - LLM推理
  - autoformalization
  - Lean
  - semantic consistency
  - self-correction
  - reinforcement-learning
  - heterogeneous reward
  - PBSO
---

# ReForm: Reflective Autoformalization with Prospective Bounded Sequence Optimization

**会议**: ICLR 2026  
**arXiv**: [2510.24592](https://arxiv.org/abs/2510.24592)  
**代码**: [GitHub](https://github.com/RUCAIBox/ReForm)（附模型和基准）  
**领域**: llm_reasoning  
**关键词**: autoformalization, Lean, semantic consistency, self-correction, reinforcement-learning, heterogeneous reward, PBSO

## 一句话总结

提出 ReForm，一种反思式自动形式化范式，将自然语言数学问题转为 Lean 形式声明的过程从一次生成转变为"生成 → 语义自验证 → 修正"的迭代循环，并设计 PBSO 算法优化异构奖励信号，在四个基准上比最强基线平均提升 22.6 个百分点。

## 研究背景与动机

自动形式化（Autoformalization）是将自然语言数学问题翻译为可机器验证的形式声明（如 Lean 语言），是形式化数学推理的关键瓶颈。当前存在核心矛盾：

**语法正确 ≠ 语义正确**：LLM 能生成通过 Lean 编译器验证的语法正确声明，但经常无法忠实保留原始问题的语义意图（量词范围误解、隐含约束遗漏、边界情况错误等）

**一次生成范式的局限**：现有方法（包括 Goedel-V2、Kimina）将形式化视为简单翻译任务，缺乏自我反思和迭代纠错机制

**人类专家也犯错**：miniF2F 中 16.4%、ProofNet 中 38.5% 的人工形式化声明包含语义错误，说明问题本身极具挑战性

核心思想：模仿人类专家的"审查-修正"迭代过程，让模型能够在生成过程中发现并纠正自己的语义错误。

## 方法详解

### 整体框架

ReForm 将自动形式化重新定义为一个反思式迭代过程，交织形式声明生成和语义自验证：

给定自然语言问题 $Q$，在第 $t$ 次迭代中：
1. **形式化生成**：$S_t = \pi(Q, \mathcal{H}_t)$，其中 $\mathcal{H}_t = \{(S_1, C_1), \ldots, (S_{t-1}, C_{t-1})\}$ 为历史
2. **语义自验证**：$C_t = \pi(Q, \mathcal{H}_t, S_t)$，评估 $S_t$ 相对于 $Q$ 的语义一致性
3. 循环直到验证通过

关键在于：整个过程实现为**单次自回归生成**，无需多次模型调用。

### 关键设计：异构奖励机制

设计两类互补奖励信号：

**任务奖励（终端）**：
$$r_{\text{task}}(Q, \text{Ans}) = \begin{cases} 1 & \text{if PassesLean(Ans)} \land \text{IsConsistent}(Q, \text{Ans}) \\ 0 & \text{otherwise} \end{cases}$$

**辅助奖励（中间步）**：
$$r_{\text{aux}}^t(Q, S_t, C_t) = \begin{cases} 1 & \text{if IsFaithfulCritique}(Q, S_t, C_t) \\ 0 & \text{otherwise} \end{cases}$$

辅助奖励评估每一步 critique 是否准确诊断了语义关系，惩罚假阳性、假阴性和过早终止。

### Prospective Bounded Sequence Optimization (PBSO)

核心贡献：解决异构奖励在不同序列位置的优化问题。

**前瞻有界回报**：对轨迹中的每一步计算未来累积奖励：
$$G_t = \text{clip}(r_t + \gamma \cdot G_{t+1}, r_{\min}, r_{\max})$$

其中 $\gamma \in (0,1]$ 为折扣因子，$G_{T+1} = 0$，clip 操作将回报限制在奖励函数范围内，避免无界累积导致梯度不稳定。

**位置特定优势函数**：对 $N$ 条采样轨迹，通过联合归一化计算每个迭代步的优势：
$$\hat{A}_t^j = \frac{G_t^j - \text{mean}(\mathcal{G})}{\text{std}(\mathcal{G})}, \quad \mathcal{G} = \bigcup_{j=1}^N \{G_t^j : t=1,...,T_j+1\}$$

同一轨迹内不同迭代步可获得不同优势值，实现细粒度信用分配。最终使用标准 GRPO 策略更新。

### 损失函数

标准 GRPO 损失配合位置特定优势 $\hat{A}_t^j$，同时优化形式化准确性和自验证质量。

## 实验关键数据

### 主实验

| 模型 | miniF2F sem | ProofNet sem | Putnam sem | AIME2025 sem | AVG sem |
|------|-----------|------------|----------|------------|---------|
| GPT-5 | 66.0 | 44.6 | 45.8 | 13.3 | 42.4 |
| Goedel-V2-8B | 81.1 | 47.3 | 42.9 | 26.7 | 49.5 |
| Goedel-V2-32B | 82.0 | 50.5 | 41.4 | 26.7 | 50.1 |
| **ReForm-8B** | **87.7** | **65.6** | **57.3** | **46.7** | **64.3** |
| **ReForm-32B** | **91.4** | **70.4** | **62.3** | **66.7** | **72.7** |

ReForm-8B 平均语义一致性比 Goedel-V2-8B 提升 **+14.8pp**，甚至超过 4 倍大的 Goedel-V2-32B (+14.2pp)。ReForm-32B 平均语义一致性达到 **72.7%**，比最强基线提升 **+22.6pp**。

在难度更高的数据集上提升更显著：ProofNet +19.9pp, PutnamBench +20.9pp, AIME2025 +40.0pp（32B）。

### 消融实验

| 变体 | miniF2F | ProofNet | Putnam | AIME25 |
|------|---------|----------|--------|--------|
| ReForm（完整） | 87.7 | 65.6 | 57.3 | 46.7 |
| w/o clip | 84.0 | 59.6 | 48.9 | 26.7 |
| w/o $r_{\text{aux}}$ | 87.7 | 65.6 | 52.1 | 40.0 |
| w/o RL | 85.2 | 62.3 | 49.4 | 30.0 |
| One-pass | 82.7 | 59.1 | 40.8 | 16.7 |

关键发现：
- **移除 clip** 导致严重退化（AIME25 从 46.7 降至 26.7），确认有界回报对异构奖励优化至关重要
- **辅助奖励** $r_{\text{aux}}$ 在复杂问题上影响更大（Putnam -5.2, AIME25 -6.7）
- **One-pass vs ReForm**：差距随问题难度增加而扩大（AIME25 差距 30pp），验证了反思范式对困难问题的必要性

### 关键发现

1. **训练动态**：响应长度从 2,300 自然增长到 4,800 token（无长度奖励），模型自发学会更深入的自检行为
2. **ConsistencyCheck 基准**：859 个专家标注项表明，前沿 LLM 作为评测者的准确率约 85.8%；但 ReForm 的提升 (+22.6pp) 远超评测噪声
3. **人类专家也犯错**：miniF2F 16.4%、ProofNet 38.5% 的人工形式化包含语义错误

## 亮点与洞察

1. **范式转变**：从一次翻译到迭代反思、从单一终端奖励到异构奖励是两个独立但互补的创新
2. **参数效率惊人**：8B 模型超过 32B 基线，说明反思架构创新的价值超越了参数规模
3. **PBSO 的通用性**：前瞻有界回报不仅适用于形式化任务，对其他多目标序列决策问题也有启发
4. **训练稳定性**：奖励曲线平滑上升且置信带收窄，验证了 PBSO 对异构目标的有效平衡
5. **揭示评测局限**：ConsistencyCheck 基准的构建不仅验证了评测可靠性，还量化了形式化挑战

## 局限性

1. 训练数据来自多个开源来源，虽已去重但数据质量仍可能影响上界
2. 推理时的 token 消耗增长 2.1 倍，对推理效率有一定影响
3. 评测依赖 LLM-as-judge（准确率 85.8%），存在约 14% 的评判噪声
4. PBSO 引入了折扣因子 $\gamma$ 和 clip 范围等额外超参数

## 相关工作与启发

- **与 Goedel/Kimina 的关系**：它们通过高质量数据提升语义一致性，但仍是一次生成；ReForm 在方法论层面引入反思循环
- **与通用 RL for LLM 的关系**：GRPO、DAPO 等方法仅用终端奖励，不监督中间步；PBSO 的前瞻有界回报为中间步提供了显式监督信号
- **对自动定理证明的启发**：如果形式化本身能更准确，下游 ATP 的性能上界也会相应提升

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 反思范式 + PBSO 的双重创新，解决了形式化中的核心语义问题
- **实用性**: ⭐⭐⭐⭐ — 形式化任务偏专业化，但方法思想有广泛迁移潜力
- **实验完整度**: ⭐⭐⭐⭐⭐ — 四个基准、全面消融、训练动态分析、评测可靠性验证
- **写作质量**: ⭐⭐⭐⭐⭐ — 动机清晰，方法推导严谨，实验分析深入
- **综合评分**: ⭐⭐⭐⭐⭐ — 在形式化领域实现了质的飞跃，方法论贡献具有普适价值

<!-- RELATED:START -->

## 相关论文

- [DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)
- [Adaptive Social Learning via Mode Policy Optimization for Language Agents](adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)
- [FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning](fastgrpo_accelerating_policy_optimization_via_concurrency-aware_speculative_deco.md)
- [AbbIE: Autoregressive Block-Based Iterative Encoder for Efficient Sequence Modeling](../../NeurIPS2025/llm_reasoning/abbie_autoregressive_block-based_iterative_encoder_for_efficient_sequence_modeli.md)
- [SAPO: Self-Adaptive Process Optimization Makes Small Reasoners Stronger](../../AAAI2026/llm_reasoning/sapo_self-adaptive_process_optimization_makes_small_reasoners_stronger.md)

<!-- RELATED:END -->
