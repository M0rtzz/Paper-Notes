---
title: >-
  [论文解读] Computational Hardness of Reinforcement Learning with Partial $q^\pi$-Realizability
description: >-
  [NeurIPS 2025][强化学习][计算复杂度] 引入"部分 $q^\pi$-可实现性"概念，证明在此设定下使用贪心策略集时学习近优策略是 NP-hard 的，使用 softmax 策略集时在 rETH 假设下需要指数时间，弥合了 $q^*$-可实现性和 $q^\pi$-可实现性之间的理论空白。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "计算复杂度"
  - "线性函数逼近"
  - "可实现性"
  - "NP-hard"
---

# Computational Hardness of Reinforcement Learning with Partial $q^\pi$-Realizability

**会议**: NeurIPS 2025  
**arXiv**: [2510.21888](https://arxiv.org/abs/2510.21888)  
**代码**: 待确认  
**领域**: 强化学习  
**关键词**: 强化学习, 计算复杂度, 线性函数逼近, 可实现性, NP-hard

## 一句话总结

引入"部分 $q^\pi$-可实现性"概念，证明在此设定下使用贪心策略集时学习近优策略是 NP-hard 的，使用 softmax 策略集时在 rETH 假设下需要指数时间，弥合了 $q^*$-可实现性和 $q^\pi$-可实现性之间的理论空白。

## 研究背景与动机

在使用线性函数逼近的强化学习中，两个极端的可实现性假设已被广泛研究：

**$q^*$-可实现性**（较弱假设）：仅假设最优值函数线性可实现。已知在此设定下学习具有准多项式甚至指数级计算复杂度下界

**$q^\pi$-可实现性**（较强假设）：假设所有策略的值函数都线性可实现。在生成模型访问下可实现高效计算

这产生了两个核心问题：
- **Q1**：在 $q^\pi$-可实现性设定下限制策略类 $\Pi$，是否仍能获得正面结果？
- **Q2**：扩展 $q^*$-可实现性的策略类到 $\{π^*\} \subsetneq \Pi$，能否打破计算困难性？

## 方法详解

### 整体框架

**部分 $q^\pi$-可实现性**（Definition 3.1）：给定策略集 $\Pi \subset \mathcal{A}^{\mathcal{S}}$ 和特征向量 $\phi: \mathcal{S} \times \mathcal{A} \to \mathbb{R}^d$，MDP 在 $\Pi$ 下部分 $q^\pi$-可实现，如果对所有 $\pi \in \Pi$，存在 $\theta_h \in \mathbb{R}^d$ 使得：

$$q_h^\pi(s_h, a_h) = \langle \phi(s_h, a_h), \theta_h \rangle \quad \forall (s_h, a_h) \in \mathcal{S}_h \times \mathcal{A}$$

**学习目标**（$\epsilon$-最优）：找到策略 $\pi$ 使得

$$\max_{\hat{\pi} \in \Pi} v^{\hat{\pi}}(s_1) - v^{\pi}(s_1) \leq \epsilon$$

### 关键设计

**贪心策略集 $\Pi^g$**（Definition 3.3）：

由参数化特征向量 $\phi': \mathcal{S} \times \mathcal{A} \to \mathbb{R}^{d'}$ 和权重 $\theta' \in \mathbb{R}^{d'}$ 定义：

$$\pi_{\theta'}^g(s_h) := \arg\max_{a \in \mathcal{A}} \langle \phi'(s_h, a), \theta' \rangle$$

策略集 $\Pi^g = \{\pi_{\theta'}^g \mid \theta' \in \mathbb{R}^{d'}\}$。

**Softmax 策略集 $\Pi^{sm}$**（Definition 3.4）：

$$\pi_{\theta'}(a | s_h) = \frac{e^{\phi'(s_h, a)^\top \theta'}}{\sum_{i=1}^{\kappa} e^{\phi'(s_h, a_i)^\top \theta'}}$$

策略集 $\Pi^{sm} = \{\pi_{\theta'} \mid \theta' \in \mathbb{R}^{d'}\}$。

**规约证明的核心思路**（两步法）：

**Step 1：MDP 多项式构造**。从 $\delta$-Max-3SAT 实例 $\varphi$（$n$ 个变量，$k$ 个子句）构造 MDP $M_\varphi$：
- **状态**：$n$-元组 $\{-1, 0, 1\}^n$，初始状态全为 $-1$，二叉树结构共 $2^{n+1}-1$ 个状态
- **动作**：$\mathcal{A} = \{0, 1\}$（对应变量赋值 False/True）
- **转移**：确定性，在第 $h$ 步将状态第 $h$ 个分量设为选择的动作值
- **奖励**：仅在终止状态给出，为满足子句比例 $R(s_H) = |\mathcal{C}_{true}(s_H)| / |\mathcal{C}|$

**Step 2：算法等价性**。证明如果 $\mathcal{A}_{rl}$ 在 $M_\varphi$ 上返回 $\epsilon$-最优策略，则可构造 $\mathcal{A}_{sat}$ 解决 $\delta$-Max-3SAT。

### 损失函数

本文为纯理论工作，不涉及训练损失函数。核心是通过计算复杂度规约建立不可能性结果。

## 实验关键数据

### 主实验

本文为理论贡献，主要定理为：

| 定理 | 策略集 | 复杂性假设 | 结论 |
|------|--------|-----------|------|
| Theorem 3.1 | 贪心 $\Pi^g$ | P ≠ NP | gLinear-$\kappa$-RL 是 NP-hard（$\epsilon \leq 0.05$） |
| Theorem 3.2 | Softmax $\Pi^{sm}$ | rETH | sLinear-$\kappa$-RL 需要 $\exp(o(d^{1/3}/\text{polylog}(d^{1/3})))$ 时间 |

**复杂性谱系定位**：

| 可实现性设定 | 假设强度 | 计算复杂度 |
|-------------|---------|-----------|
| $q^*$-可实现性 | 最弱 | 指数下界（已知） |
| **部分 $q^\pi$-可实现性** | **中间** | **NP-hard / 指数下界（本文）** |
| $q^\pi$-可实现性 | 最强 | 多项式可解（生成模型） |

### 消融实验

**MDP 构造参数分析**：

- PSP 特征向量 $\phi' \in \mathbb{R}^n$ 的构造：第 $h$ 个分量对应变量 $x_h$，动作 True 设为 $+1$，False 设为 $-1$
- 可实现性特征向量 $\phi \in \mathbb{R}^d$ 的构造复杂度：$O(n^3)$（依赖于总子句集大小）
- 整个规约在多项式时间内完成

**$\delta$-Max-3SAT 实例分析**（Example 4.1）：以 $\varphi: (x_1 \vee \bar{x}_2 \vee x_3) \wedge (\bar{x}_1 \vee x_2 \vee \bar{x}_3)$ 为例，构造含15个状态的 MDP，除 $(0,1,0)$ 和 $(1,0,1)$ 外所有终态奖励为1，这两个终态奖励为 $1/2$。

### 关键发现

1. **不可能性持续存在**：即使策略类 $\Pi$ 扩展到包含无限多个线性可实现策略，部分 $q^\pi$-可实现性下求解仍然计算不可行
2. **微妙悖论**：$\Pi^g \subset \Pi^{sm}$ 高概率成立，但贪心策略集仅需 P ≠ NP 假设即可证明硬度，softmax 需更强的 rETH 假设
3. **对 Q1 和 Q2 的部分否定回答**：限制策略类不能保证计算效率（Q1），扩展策略类不能打破硬度（Q2）
4. **与 Agnostic RL 的联系**：硬度结果也暗示了线性函数逼近下 Agnostic RL 的内在计算困难

## 亮点与洞察

- **理论定位精准**：部分 $q^\pi$-可实现性优雅地弥合了两个极端可实现性假设之间的空白，提供了更实际的建模框架
- **规约设计巧妙**：将 Max-3SAT 的变量赋值自然映射为 MDP 的动作序列，二叉树 MDP 结构简洁清晰
- **一般化条件**（Remark 4.2）：给出了将硬度结果推广到任意策略集的充分条件三元组：(i) 策略多项式可参数化，(ii) 特征向量多项式可构造，(iii) 规约保持问题求解等价性
- **实际意义**：硬度结果不应被视为纯负面——它指出需要额外结构假设（如策略/特征约束）才能获得高效算法

## 局限性

1. 策略参数化特征 $\phi'$ 与可实现性特征 $\phi$ 使用不同向量，统一特征的情况（$\phi' = \phi$）尚未解决
2. 硬度结果基于最坏情况分析，实际 RL 问题可能具有使其更容易求解的结构
3. 仅考虑确定性转移 MDP，随机转移下的结论可能不同
4. 未提供正面算法结果或高效求解的充分条件
5. 生成模型（generative model）是最强交互形式，更弱的交互模式下结论未知

## 相关工作与启发

- **与 KLL+ (2023) 的关系**：扩展了 $q^*$/$v^*$-可实现性下的指数硬度结果到中间假设地带
- **与 YHAY+ (2022) 的对比**：完全 $q^\pi$-可实现性下可用生成模型高效求解，但部分可实现性即使有生成模型也不行——展示了小假设放松导致的巨大计算复杂度跃迁
- **与 Agnostic RL 的联系**：部分可实现性等价于给定策略类的 agnostic 学习，本文结果暗示 agnostic RL 在线性逼近下的内在困难
- **启发方向**：研究统一特征情况（$\phi' = \phi$）的复杂度；探索使部分可实现性易解的额外结构条件

## 评分

- ⭐ 创新性：4/5 — 部分可实现性概念的提出和精确复杂度刻画填补了重要理论空白
- ⭐ 实用性：2/5 — 负面理论结果对算法设计有指导意义但无直接应用
- ⭐ 实验充分度：2/5 — 纯理论工作，以示例和证明替代实验
- ⭐ 写作质量：4/5 — 定义和定理表述严谨，证明思路清晰，但符号较重

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Partial Action Replacement: Tackling Distribution Shift in Offline MARL](../../AAAI2026/reinforcement_learning/partial_action_replacement_tackling_distribution_shift_in_offline_marl.md)
- [\[NeurIPS 2025\] Hybrid Latent Reasoning via Reinforcement Learning](hybrid_latent_reasoning_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Reinforcement Learning with Action Chunking](reinforcement_learning_with_action_chunking.md)
- [\[NeurIPS 2025\] Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning](adaptive_neighborhoodconstrained_q_learning_for_offline_rein.md)
- [\[NeurIPS 2025\] Learning to Clean: Reinforcement Learning for Noisy Label Correction](learning_to_clean_reinforcement_learning_for_noisy_label_correction.md)

</div>

<!-- RELATED:END -->
