---
title: >-
  [论文解读] Sequential Multi-Agent Dynamic Algorithm Configuration
description: >-
  [NeurIPS 2025][动态算法配置] 提出 Seq-MADAC 框架，将多超参数动态配置建模为上下文顺序多智能体 MDP，通过顺序优势分解网络（SADN）利用参数间的固有依赖关系，在多目标优化算法配置上超越现有 MARL 方法。
tags:
  - NeurIPS 2025
  - 动态算法配置
  - 多智能体强化学习
  - 顺序依赖
  - 优势分解
  - 超参数优化
---

# Sequential Multi-Agent Dynamic Algorithm Configuration

**会议**: NeurIPS 2025  
**arXiv**: [2510.23535](https://arxiv.org/abs/2510.23535)  
**代码**: https://github.com/lamda-bbo/seq-madac  
**领域**: 强化学习  
**关键词**: 动态算法配置, 多智能体强化学习, 顺序依赖, 优势分解, 超参数优化

## 一句话总结
提出 Seq-MADAC 框架，将多超参数动态配置建模为上下文顺序多智能体 MDP，通过顺序优势分解网络（SADN）利用参数间的固有依赖关系，在多目标优化算法配置上超越现有 MARL 方法。

## 研究背景与动机

**领域现状**：动态算法配置（DAC）通过 RL 学习在算法执行过程中动态调整超参数，已证明优于静态配置。多智能体 DAC（MADAC）进一步处理多个异构超参数

**现有痛点**：许多算法的超参数存在固有的顺序依赖（如先确定算子类型，再设置算子参数），但 MADAC 假设参数独立。CANDID DAC 尝试用 SAQL 解决但存在信用分配和收敛性问题

**核心矛盾**：参数间的依赖关系使得联合动作空间存在大量非法组合，忽略依赖导致低效探索

**核心 idea**：将问题建模为顺序 MMDP，后续智能体能观察前序智能体的动作，并通过优势分解实现差异化信用分配

## 方法详解

### 整体框架
Seq-MADAC 将参数配置建模为顺序决策过程。每个时间步，智能体按预定顺序依次决策，第 i 个智能体可观察到前 i-1 个智能体的动作。

### 关键设计

1. **顺序优势分解网络 (SADN)**：

    - 全局优势函数分解为各智能体的个体优势之和：$A(s,\boldsymbol{a}) = \sum_{i=1}^n A_i(s, \boldsymbol{a}_{1:i-1}, a_i)$
    - 每个智能体维护独立网络建模 $A_i$，通过全局优势的反向传播隐式更新
    - 满足 IGM 原则：$\arg\max_{\boldsymbol{a}} Q(s,\boldsymbol{a})$ 等价于顺序选择各维度最优动作

2. **独立更新机制**：

    - 全局优势通过一步 TD 误差更新：$A(s,\boldsymbol{a}) \leftarrow A(s,\boldsymbol{a}) + \alpha[r + \gamma V(s') - V(s) - A(s,\boldsymbol{a})]$
    - 各智能体网络独立更新，避免了 ACE 的链式误差累积和 SAQL 的干扰问题

3. **上下文顺序 MMDP 建模**：将问题实例分布纳入 MDP 定义，支持跨问题泛化

## 实验关键数据

### 主实验 — MOEA/D 多目标优化 (IGD↑越大越好)

| 问题 | 维度 | SADN | ACE | SAQL | VDN | MAPPO |
|------|------|------|-----|------|-----|-------|
| DTLZ2 | 6 | **4.593e-2** | 3.809e-2 | 3.851e-2 | 3.950e-2 | 3.906e-2 |
| WFG4 | 6 | **5.729e-2** | 4.537e-2 | 4.626e-2 | 4.817e-2 | 4.639e-2 |
| WFG6 | 9 | **6.641e-2** | 4.884e-2 | 5.111e-2 | 5.558e-2 | 6.194e-2 |

### 消融实验 — Seq-Sigmoid-Robust

| 设置 | SADN | ACE | SAQL |
|------|------|-----|------|
| 5维-1随机 | 最稳定 | 性能衰退 | 低性能 |
| 10维-2随机 | 最稳定 | 严重衰退 | 低性能 |

### 关键发现
- 训练集上 SADN 在 0/9/0 的显著性检验中全胜
- ACE 在多实例环境下因链式更新中断而衰退
- 正确顺序 vs 逆序实验证实了建模依赖关系的重要性

## 亮点与洞察
- IGM 原则的顺序扩展有理论保证，将联合动作优化分解为顺序贪心
- 优势分解比 Q 值分解更灵活——解决了 VDN/QMIX 中加性/单调性假设的局限

## 局限与展望
- 需要预先指定参数顺序，自动学习顺序将更有价值
- 实验集中在 MOEA/D，对更多算法的验证不足

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次在 DAC 中建模参数顺序依赖
- 实验充分度: ⭐⭐⭐⭐ 合成+真实任务，全面比较
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰
- 价值: ⭐⭐⭐⭐ 实用且有理论基础

<!-- RELATED:START -->

## 相关论文

- [Graph-Supported Dynamic Algorithm Configuration for Multi-Objective Combinatorial Optimization](../../ICML2025/reinforcement_learning/graph-supported_dynamic_algorithm_configuration_for_multi-objective_combinatoria.md)
- [Extending NGU to Multi-Agent RL: A Preliminary Study](extending_ngu_to_multi-agent_rl_a_preliminary_study.md)
- [Robust and Diverse Multi-Agent Learning via Rational Policy Gradient](robust_and_diverse_multi-agent_learning_via_rational_policy_gradient.md)
- [Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models](communicating_plans_not_percepts_scalable_multi-agent_coordination_with_embodied.md)

<!-- RELATED:END -->
