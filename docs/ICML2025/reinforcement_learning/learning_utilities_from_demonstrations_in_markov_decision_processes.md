---
title: >-
  [论文解读] Learning Utilities from Demonstrations in Markov Decision Processes
description: >-
  [ICML 2025][效用学习] 本文提出 Utility Learning (UL) 问题，通过从演示中推断智能体的效用函数来捕捉其风险态度，设计了两个可证明高效的算法并分析了样本复杂度和可辨识性。
tags:
  - ICML 2025
  - 效用学习
  - 风险敏感行为
  - 逆强化学习
  - 效用函数
  - MDP
---

# Learning Utilities from Demonstrations in Markov Decision Processes

**会议**: ICML 2025  
**arXiv**: [2409.17355](https://arxiv.org/abs/2409.17355)  
**代码**: 无  
**领域**: Reinforcement Learning / Inverse RL  
**关键词**: 效用学习, 风险敏感行为, 逆强化学习, 效用函数, MDP

## 一句话总结
本文提出 Utility Learning (UL) 问题，通过从演示中推断智能体的效用函数来捕捉其风险态度，设计了两个可证明高效的算法并分析了样本复杂度和可辨识性。

## 研究背景与动机
**领域现状**: 逆强化学习 (IRL) 从专家演示推断奖励函数，是理解和模仿行为的核心方法。然而，经典 IRL 假设智能体是风险中性的 (最大化期望累积奖励)。

**现有痛点**: 人类行为普遍表现出风险敏感性——面对随机性时会表现出风险规避或风险追求。风险中性假设不仅引入模型误差，还无法直接捕捉智能体的风险态度。

**核心矛盾**: IRL 的建模能力不足以表达风险偏好——即使完美恢复奖励函数，也无法区分风险规避者和风险追求者。

**本文切入角度**: 引入效用函数 $u$ 来显式表示风险态度，将其学习定义为 Utility Learning (UL) 问题。

**核心 idea**: 在 MDP 中定义效用函数作为风险态度的编码，分析其部分可辨识性，并设计有限样本算法进行推断。

## 方法详解

### 整体框架
输入：MDP 环境结构 + 智能体的演示轨迹 → 效用学习算法 → 输出效用函数（编码风险态度）

### 关键设计

1. **效用函数建模**:

    - 效用函数 $u: \mathbb{R} \to \mathbb{R}$ 作用于累积奖励，捕捉风险态度
    - 凹函数 → 风险规避，凸函数 → 风险追求，线性 → 风险中性
    - 设计动机：经济学中的期望效用理论 (von Neumann-Morgenstern) 为风险建模提供坚实理论基础

2. **部分可辨识性分析**:

    - 分析在什么条件下效用函数可从演示中唯一确定
    - 建立不可辨识性的精确刻画：存在等价类
    - 设计动机：理解问题的理论边界对算法设计至关重要

3. **两个高效算法**:

    - 算法基于有限样本保证
    - 分析样本复杂度：给定精度要求需要多少演示轨迹
    - 设计动机：提供可证明的有限样本效率保证，而非仅渐近一致性

### 损失函数 / 训练策略
- 通过最大似然或矩匹配从演示中估计效用函数
- 利用 MDP 结构约束搜索空间

## 实验关键数据

### 主实验
| 场景 | 指标 | 效果 | 说明 |
|------|------|------|------|
| 风险规避智能体 | 效用函数恢复 | 正确识别凹形效用 | 验证模型能区分风险态度 |
| 风险追求智能体 | 效用函数恢复 | 正确识别凸形效用 | 与风险规避形成对比 |
| 风险中性智能体 | 效用函数恢复 | 正确识别线性效用 | 退化为标准 IRL |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 不同样本量 | 恢复误差 | 随样本增加单调下降，符合理论预测 |
| 不同 MDP 结构 | 可辨识性 | 某些简单 MDP 下效用不可辨识 |

### 关键发现
- 效用函数的可辨识性强烈依赖 MDP 的结构（状态空间大小、随机性程度）
- 两个算法在有限数据条件下即可有效区分风险规避和风险追求行为
- 标准 IRL 方法在风险敏感行为数据上产生显著的模型误设

## 亮点与洞察
- **问题定义新颖**: 将风险态度显式建模为效用函数并定义 UL 问题，扩展了 IRL 的表达能力
- **理论严谨**: 提供了可辨识性分析和样本复杂度保证
- **连接经济学与 RL**: 将期望效用理论引入 MDP/IRL 框架

## 局限与展望
- 实验为 proof-of-concept 级别，缺乏大规模/实际场景验证
- 效用函数假设为已知参数族，非参数设置更具挑战性
- 未讨论时变风险态度（人的风险偏好可能随上下文变化）

## 相关工作与启发
- 经典 IRL (Abbeel & Ng 2004, MaxEnt IRL) 假设风险中性
- Risk-sensitive RL (CVaR-RL, 鲁棒 MDP) 从智能体训练角度处理风险
- 行为经济学中的前景理论 (Kahneman & Tversky) 提供更丰富的风险建模方式
- 启发：从演示中学习的不仅是"做什么"，还包括"如何面对不确定性"

## 评分
- 新颖性: ⭐⭐⭐⭐ 效用学习问题的形式化定义和可辨识性分析是新颖的理论贡献
- 实验充分度: ⭐⭐⭐ 理论驱动的工作，实验以概念验证为主
- 写作质量: ⭐⭐⭐⭐ 数学表述严谨清晰
- 价值: ⭐⭐⭐⭐ 为 IRL/LfD 领域引入重要的建模维度

<!-- RELATED:START -->

## 相关论文

- [A Generalized Bisimulation Metric of State Similarity between Markov Decision Processes: From Theoretical Propositions to Applications](../../NeurIPS2025/reinforcement_learning/a_generalized_bisimulation_metric_of_state_similarity_betwee.md)
- [Solving Zero-Sum Convex Markov Games](solving_zero-sum_convex_markov_games.md)
- [Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making](counterfactual_effect_decomposition_in_multi-agent_sequential_decision_making.md)
- [Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning](divide_and_conquer_grounding_llms_as_efficient_decision-making_agents_via_offlin.md)
- [SkillMimic: Learning Basketball Interaction Skills from Demonstrations](../../CVPR2025/reinforcement_learning/skillmimic_learning_basketball_interaction_skills_from_demonstrations.md)

<!-- RELATED:END -->
