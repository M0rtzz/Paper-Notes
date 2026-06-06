---
title: >-
  [论文解读] Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for Agentic Reinforcement Learning
description: >-
  [ICML 2026][LLM评测][信用分配] 提出 GraphGPO，将所有 rollout 轨迹聚合为统一的状态转移图，利用图上全局最短路径信息为每一步计算基于距离的 advantage，实现比轨迹级归因更精细的信用分配…
tags:
  - "ICML 2026"
  - "LLM评测"
  - "信用分配"
  - "图结构策略优化"
  - "多轮智能体任务"
  - "状态转移图"
  - "无评论家RL"
---

# Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for Agentic Reinforcement Learning

**会议**: ICML 2026  
**arXiv**: [2605.26684](https://arxiv.org/abs/2605.26684)  
**代码**: [https://github.com/langfengQ/verl-agent/tree/master/recipe/GraphGPO](https://github.com/langfengQ/verl-agent/tree/master/recipe/GraphGPO)  
**领域**: 强化学习  
**关键词**: 信用分配, 图结构策略优化, 多轮智能体任务, 状态转移图, 无评论家RL  

## 一句话总结

提出 GraphGPO，将所有 rollout 轨迹聚合为统一的状态转移图，利用图上全局最短路径信息为每一步计算基于距离的 advantage，实现比轨迹级归因更精细的信用分配，在 ALFWorld、WebShop、Sokoban 上显著超越 GRPO 和 GiGPO。

## 研究背景与动机

**领域现状**：基于组的强化学习方法（如 GRPO）在 LLM 后训练中取得了巨大成功，其核心优势是抛弃了资源密集的 critic 模型，仅依赖可验证奖励和组内统计量估计 advantage。近期多项工作已将 GRPO 扩展到多轮智能体任务。

**现有痛点**：GRPO 及其变体的信用分配本质上依赖轨迹级归因——成功轨迹中所有步骤都获得正信用，失败轨迹中所有步骤都被惩罚。然而在多轮任务中，这种归因存在严重的不对齐：失败轨迹中约 22% 的步骤实际上在推进任务目标，而成功轨迹中约 65% 的步骤并未有效推进任务。冗余步骤被错误奖励，有价值的失败步骤被错误惩罚。

**核心矛盾**：轨迹级别的成功/失败信号粒度太粗，无法反映中间步骤对任务目标的真实贡献。即便 GiGPO 引入了步级 advantage，其步级奖励 $R^S = \lambda^{T-i} R(\boldsymbol{\tau})$ 仍然依赖最终轨迹结果 $R(\boldsymbol{\tau})$，未能真正脱离轨迹级归因。

**本文目标**：设计一种完全基于全局状态结构的步级信用分配方法，无需额外 critic 模型，且不引入显著计算开销。

**切入角度**：如果将所有 rollout 轨迹中的状态合并为一张有向图，就能利用图的连通性判断每个状态距目标有多远，从而为每一步分配基于"距离缩减"的奖励——这完全不依赖该步所在轨迹的最终结果。

**核心 idea**：将全部 rollout 轨迹聚合为统一状态转移图，用最短路径距离定义步级奖励，用图上同源边的组内统计量计算 advantage。

## 方法详解

### 整体框架

GraphGPO 的 pipeline 分三步：(1) 将同一 task 的 $M$ 条 rollout 轨迹聚合为一张有向状态转移图 $\mathcal{G} = (\mathcal{S}, \mathcal{E})$；(2) 在图上用 Dijkstra 算法计算每个状态到目标状态 $s_{\text{succ}}$ 的最短距离 $d(s)$；(3) 基于距离为每条边计算图级步级奖励和 advantage，最终结合轨迹级 advantage 进行 PPO 风格的策略优化。

### 关键设计

1. **聚合状态转移图**:

   将 $M$ 条轨迹中所有状态作为节点、所有转移作为有向边，相同状态合并为同一节点。节点集 $\mathcal{S} = \bigcup_{m,t} \{s_t^m\}$，边 $(s, \boldsymbol{a}, s', c(s,\boldsymbol{a})) \in \mathcal{E}$ 表示在状态 $s$ 执行动作 $\boldsymbol{a}$ 转移到 $s'$ 并产生代价 $c$。这使得不同轨迹间的状态共享和路径交叉关系得以显式表达——例如失败轨迹的前半段可能通过共享状态连接到成功轨迹的后半段。

2. **基于最短路径的步级奖励**:

   对每个状态 $s$，用递归定义计算到目标的最短距离：$d(s) = \min_{(s,a,s',c) \in \mathcal{E}} (c(s,\boldsymbol{a}) + d(s'))$，其中 $d(s_{\text{succ}})=0$，不可达状态 $d(s)=d_{\max}+1$。然后定义图级步级奖励 $R^G(s, \boldsymbol{a}, s') = r_{\text{succ}} \cdot \omega^{d(s') + c(s,\boldsymbol{a})}$，其中 $\omega \in (0,1)$ 是距离折扣因子。这意味着距目标越近的转移获得越高奖励，与轨迹最终成败完全无关。

3. **图级 advantage 估计与组合优化**:

   将同一起始状态 $s$ 的所有出边分为一组 $G^G(s)$，在组内计算标准化 advantage $A^G = (R^G - \mu) / \sigma$。当组内只有一条边时 $A^G = 0$，因此需要结合轨迹级 advantage：$A(s,\boldsymbol{a},s') = \beta^G A^G + \beta^E A^E(\boldsymbol{\tau})$。最终用 PPO clipped objective 加 KL 惩罚进行策略更新。作者证明了图级 advantage 具有单调性（距离缩减越多 advantage 越大）和方差缩减性质（条件方差不超过轨迹级反馈）。

## 实验关键数据

| 基准 | 模型 | GRPO | GiGPO | GraphGPO | 提升 (vs GRPO) |
|------|------|------|-------|----------|------------|
| ALFWorld | Qwen2.5-1.5B | 77.86% | 90.88% | **92.71%** | +14.85% |
| ALFWorld | Qwen2.5-7B | 83.33% | 94.27% | **95.31%** | +11.98% |
| WebShop (Succ.) | Qwen2.5-1.5B | 71.35% | 73.83% | **78.65%** | +7.30% |
| WebShop (Succ.) | Qwen2.5-7B | 75.00% | 78.38% | **80.31%** | +5.31% |
| Sokoban 6×6 | Qwen2.5-VL-3B | 67.1% | 76.92% | **86.98%** | +19.88% |

| 消融/特性 | 结论 |
|----------|------|
| 去除 $A^E$ | 两方法均下降，但 GraphGPO 在 Sokoban 仍超 GiGPO 20.57% |
| 动态采样 (+DS) | GraphGPO + DS 在 ALFWorld 达 **98.43%**，WebShop 达 **85.68%** |
| 计算开销 | 图构建 0.108s + advantage 计算 0.025s，仅占每轮总时间的 0.04% |
| 训练动态 | 训练前期收敛速度显著更快，尤其在成功率低时信号更有效 |

## 亮点与洞察

- **对失败轨迹的价值挖掘**：通过图结构，失败轨迹中的有效步骤可以获得正 advantage（因为它们确实缩短了到目标的距离），这是传统轨迹级归因无法做到的。
- **冗余/循环行为的天然惩罚**：在图中形成环路的步骤必然增加距离（$d(s_{41}) > d(s_2)$），自然获得更低 advantage，无需额外的惩罚机制。
- **几乎零额外开销**：仅需在每个训练迭代额外执行一次 Dijkstra 最短路径搜索，复杂度 $O((|\mathcal{V}|+|\mathcal{E}|) \log |\mathcal{V}|)$，耗时 0.133s vs 总耗时 291s。
- **理论保证**：证明了 advantage 单调性（Proposition 4.1）和条件方差缩减（Proposition 4.2），为方法的有效性提供了分析支撑。

## 局限性 / 可改进方向

- **确定性环境假设**：图上的状态合并要求环境是确定性的（同一状态执行同一动作得到相同后继），在随机环境中状态合并的有效性可能大幅下降。
- **状态定义依赖人工设计**：需要定义什么构成"相同状态"（论文中使用环境观测的确定性部分），对于开放域任务（如自由文本对话）中状态等价性判断可能困难。
- **代价函数 $c(s,\boldsymbol{a})$ 简化为 1**：实验中所有转移代价统一设为 1，未探索非均匀代价（如工具调用的真实时间/金钱代价）的效果。
- **仅在单轮训练迭代内构图**：每次迭代的图仅基于当前 rollout 数据，未跨迭代积累历史经验。

## 相关工作与启发

- **GRPO** (Shao et al., 2024)：组级 RL 的基础，GraphGPO 保留了其无 critic 的核心优势。
- **GiGPO** (Feng et al., 2025b)：引入步级分组但仍依赖轨迹结果，GraphGPO 通过图结构彻底解耦。
- **PPO** (Schulman et al., 2017)：GraphGPO 的策略优化目标沿用 PPO clipped objective 框架。
- 启发：图结构视角为 RL 信用分配提供了一条新路径，可以考虑将类似思路应用到代码生成、数学推理等思维链场景中。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将轨迹聚合为状态转移图用于信用分配的思路新颖且直觉清晰
- 实验充分度: ⭐⭐⭐⭐ — 涵盖文本和视觉两类智能体任务，消融完整，开销分析到位
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，图示直观，理论和实验配合紧密
- 价值: ⭐⭐⭐⭐ — 为 LLM agent 的 RL 训练提供了实用且低成本的信用分配改进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents](hiper_hierarchical_reinforcement_learning_with_explicit_credit_assignment_for_la.md)
- [\[ICML 2026\] Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning](agent_world_model_infinity_synthetic_environments_for_agentic_reinforcement_lear.md)
- [\[ICML 2026\] On Effectiveness and Efficiency of Agentic Tool-calling and RL Training](on_effectiveness_and_efficiency_of_agentic_tool-calling_and_rl_training.md)
- [\[ICML 2026\] From Human-Level AI Tales to AI Leveling Human Scales](from_human-level_ai_tales_to_ai_leveling_human_scales.md)
- [\[ACL 2026\] AgentEval: DAG-Structured Step-Level Evaluation for Agentic Workflows with Error Propagation Tracking](../../ACL2026/llm_evaluation/agenteval_dag-structured_step-level_evaluation_for_agentic_workflows_with_error_.md)

</div>

<!-- RELATED:END -->
