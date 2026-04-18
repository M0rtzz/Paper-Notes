---
title: >-
  [论文解读] Routing, Cascades, and User Choice for LLMs
description: >-
  [ICLR 2026][LLM routing] 将LLM路由建模为provider-user Stackelberg博弈，证明最优路由几乎总是静态无级联的阈值规则，揭示质量/成本排序不一致时的用户-提供商不对齐，以及低流失惩罚下provider被激励通过throttling延迟来降低成本但损害用户效用。
tags:
  - ICLR 2026
  - LLM routing
  - cascading
  - Stackelberg game
  - user-provider misalignment
  - throttling
---

# Routing, Cascades, and User Choice for LLMs

**会议**: ICLR 2026  
**arXiv**: [2602.09902](https://arxiv.org/abs/2602.09902)  
**代码**: 无  
**领域**: llm_alignment  
**关键词**: LLM routing, cascading, Stackelberg game, user-provider misalignment, throttling

## 一句话总结

将LLM路由建模为provider-user Stackelberg博弈，证明最优路由几乎总是静态无级联的阈值规则，揭示质量/成本排序不一致时的用户-提供商不对齐，以及低流失惩罚下provider被激励通过throttling延迟来降低成本但损害用户效用。

## 研究背景与动机

**领域现状**：LLM提供商通过路由和级联策略在异构模型之间分配用户任务来平衡质量、延迟和成本。GPT-5已明确采用路由，在"高效模型"和"深度推理模型"之间切换。

**现有痛点**：现有路由算法（Ding et al., 2024; Dekoninck et al., 2025）聚焦于估计LLM性能并优化质量-延迟-成本权衡，但将用户的响应行为视为外生变量。然而，LLM的提示式接口意味着模型失败后用户可能反复交互，产生反复推理成本。

**核心矛盾**：单次优化成本可能在用户行为层面带来反效果。用户可能因延迟放弃任务甚至取消订阅，视任务的价值和模型的延迟而定。优化单次成本可能"省小钱亏大钱"（penny-wise but welfare-foolish）。

**本文方案**：形式化一个双层Stackelberg博弈——provider选择路由策略（初始模型+级联概率），用户基于观察到的策略决定放弃概率。通过完全刻画用户最优响应并简化provider问题，推导出简洁的阈值规则。

## 方法详解

### 整体框架

考虑单个provider拥有两个模型 $M_1$（标准）和 $M_2$（推理），满足 $t_1 < t_2$, $c_1 < c_2$, $0 < p_1 < p_2 < 1$。Provider选择路由策略 $(i, s)$：初始模型 $i$ 和级联概率 $s$。用户选择放弃概率 $q$。

定义用户单次净值 $\xi_i := Vp_i - t_i$，当 $\xi_i > 0$ 时模型为value-dominated，否则为latency-dominated。

用户效用为成功值减去累积延迟：

$$U_i(s, q) = V \cdot S_i(s, q) - L_i(s, q)$$

Provider成本为服务成本加用户放弃惩罚：

$$J_i(s, q) = C_i(s, q) + P(1 - S_i(s, q))$$

### 关键设计1: 用户最优响应的完全刻画

**功能**：推导用户在给定provider策略下的最优放弃策略。

**核心思路**（Theorem 1-2）：
- 若路由到 $M_2$：$q^* = \mathbb{1}\{\xi_2 < 0\}$（纯阈值规则）
- 若路由到 $M_1$ 且 $\xi_1, \xi_2$ 同号：用户行为静态（均value-dominated则 $q^*=0$；均latency-dominated则 $q^*=1$）
- 若 $\xi_1 < 0 < \xi_2$：存在阈值 $s_0 = -\xi_1/(\xi_2/p_2 - \xi_1)$，当 $s > s_0$ 时用户留下，否则放弃
- 若 $\xi_1 > 0 > \xi_2$：存在 $s_L, s_H$ 两个阈值，$s \leq s_L$ 时留下，$s \geq s_H$ 时放弃，中间区域存在混合策略

**设计动机**：用户行为仅在模型价值差异化时受路由策略影响。当两模型同质化时，路由对用户决策无影响。

### 关键设计2: Provider最优路由的简化

**功能**：将provider优化问题简化为单变量问题并推导闭式解。

**核心思路**（Theorem 3-5）：
- **同号情形**（Theorem 3）：最优策略总是路由到单一模型且无级联。$\xi_1, \xi_2 > 0$ 时按cost-of-pass $c_i/p_i$ 选择；$\xi_1, \xi_2 < 0$ 时取决于惩罚 $P$ 与增量cost-of-pass的比较
- **差异化情形**（Theorem 4-5）：几乎所有regime中最优策略仍为静态 $(i^*, s^*) \in \{(1,0), (1,1), (2,0)\}$，级联仅在狭窄区域中最优

**设计动机**：级联在未差异化模型间增加成本和方差却无收益。仅当两模型净值不同且特定参数范围时，级联才有价值。

### 不对齐与Throttling分析

**Provider-用户不对齐**（Proposition 1）：当provider的cost-of-pass排序与用户的效用排序不一致时，misalignment gap $\Delta_U > 0$，即provider的成本最优策略损害用户效用。

**Throttling风险**（Proposition 2）：当用户流失惩罚 $P \leq \min\{c_1/p_1, c_2/p_2\}$ 时，provider被激励人为增加延迟 $\hat{t}_i > Vp_i$ 使两模型变为latency-dominated，鼓励用户放弃以降低服务成本。此时用户效用被最大化损害。

## 实验关键数据

### 主实验：Provider最优策略的区域划分

| $\xi_1, \xi_2$ 状态 | 用户行为 | Provider最优策略 | 级联是否有效 |
|---------------------|---------|-----------------|-------------|
| 均value-dominated | 静态留下 | 按 $c_i/p_i$ 路由，无级联 | 无效 |
| 均latency-dominated | 静态放弃 | 取决于 $P$ vs $(c_2-c_1)/(p_2-p_1)$ | 无效 |
| $\xi_1 < 0 < \xi_2$ | 依赖级联概率 | 几乎总路由到 $M_1$，除非cost-of-pass差距大 | 仅特定条件 |
| $\xi_1 > 0 > \xi_2$ | 三段式响应 | 静态为主，极窄区间有混合 | 极少 |

### 消融实验：Throttling收益

| 配置 | 效果 | 条件 |
|-----|-----|------|
| $P < \min\{c_1/p_1, c_2/p_2\}$ | Throttling有利于provider | 用户流失惩罚低 |
| $P > \min\{c_1/p_1, c_2/p_2\}$ | Throttling反而增加provider成本 | 用户流失惩罚高 |
| Throttling收益面积 | 线性于 $P$ | 用户退订可防止throttling |

### 关键发现

- 最优路由在绝大多数参数区域退化为简单阈值规则，级联的价值极为有限
- 用户行为仅在两模型差异化时受路由策略影响
- 当用户和provider的模型排序不一致时，不对齐不可避免
- 防止throttling的关键是确保用户放弃的代价（流失惩罚）足够高——用户应有"退订权"

## 亮点与洞察

- 首次将LLM路由问题从纯工程优化提升到博弈论框架，考虑用户反应行为
- 理论结果具有高度实践指导性：级联rarely optimal这一结论对GPT-5等路由系统设计有直接意义
- Throttling分析揭示了LLM订阅模式中provider的道德风险，有政策含义
- 论文本身使用LLM辅助完成（附录A详细记录），构成meta层面的自洽验证

## 局限与展望

- 仅分析两个模型的情况，实际部署可能涉及更多模型的路由
- 假设用户能观察provider的级联策略且采用平稳放弃策略，实际中路由策略对用户不透明
- 每次pass的成功概率假定为i.i.d.，忽略了用户反馈对后续尝试的影响
- 固定订阅制框架，未考虑按调用计费的API定价模式
- 缺乏实证实验验证理论预测

## 相关工作与启发

- **FrugalGPT**（Chen et al., 2023）和**RouteLLM**（Ong et al., 2025）：聚焦路由算法设计，本文从博弈论角度补充了用户行为维度
- **Cost-of-Pass**（Mahmood 2024; Erol et al., 2025）：本文直接使用cost-of-pass概念作为路由决策的核心指标
- 对LLM订阅服务设计的启发：应允许用户自选模型（opt-out routing）来防止throttling

## 评分

- 新颖性: ⭐⭐⭐⭐ 博弈论视角分析LLM路由非常新颖，填补了用户行为建模的空白
- 实验充分度: ⭐⭐⭐ 纯理论工作，有定理证明和可视化但无实证实验
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，Figure 1的guideline总结极为实用
- 价值: ⭐⭐⭐⭐ 对LLM服务定价和路由策略设计有直接实践指导，throttling分析具有政策意义
