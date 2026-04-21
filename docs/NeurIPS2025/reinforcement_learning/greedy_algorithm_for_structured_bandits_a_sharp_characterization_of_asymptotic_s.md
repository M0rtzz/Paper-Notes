---
title: >-
  [论文解读] Greedy Algorithm for Structured Bandits: A Sharp Characterization of Asymptotic Success / Failure
description: >-
  [NeurIPS 2025][structured bandits] 本文对结构化 bandit 问题中的贪心算法（Greedy）进行了完整的理论刻画，提出 self-identifiability 作为贪心算法能否获得 sublinear regret 的充要条件，并将结论推广到上下文 bandit 及一般交互决策框架 DMSO。
tags:
  - NeurIPS 2025
  - structured bandits
  - greedy algorithm
  - regret
  - self-identifiability
  - contextual bandits
---

# Greedy Algorithm for Structured Bandits: A Sharp Characterization of Asymptotic Success / Failure

**会议**: NeurIPS 2025  
**arXiv**: [2503.04010](https://arxiv.org/abs/2503.04010)  
**代码**: 无  
**领域**: reinforcement_learning  
**关键词**: structured bandits, greedy algorithm, regret, self-identifiability, contextual bandits

## 一句话总结
本文对结构化 bandit 问题中的贪心算法（Greedy）进行了完整的理论刻画，提出 self-identifiability 作为贪心算法能否获得 sublinear regret 的充要条件，并将结论推广到上下文 bandit 及一般交互决策框架 DMSO。

## 研究背景与动机

**领域现状**：Multi-armed bandit 是序贯决策的基本框架，传统理论强调探索（exploration）是获得低 regret 的必要手段。结构化 bandit 引入已知的奖励结构（如线性、Lipschitz），通过结构约束减少探索需求。

**现有痛点**：尽管探索理论成熟，纯贪心算法（Greedy，仅利用不探索）在实践中被广泛使用——因为探索在人类交互系统中成本高、不公平、可能违反用户激励。但理论上何时贪心可行、何时必败缺乏统一刻画。

**核心矛盾**：先前工作只在少数特定结构上给出贪心的成功/失败例子（如线性 contextual bandit 需上下文多样性），缺乏对任意有限奖励结构的一般性理论。

**本文目标** 对任意有限奖励结构，给出贪心算法渐近成功（sublinear regret）vs 失败（linear regret）的完整"iff"刻画。

**切入角度**：作者发现"部分可辨识性"——self-identifiability——是关键：若固定次优臂的期望奖励就能识别出它是次优的，则贪心成功；否则存在 decoy 使贪心永久陷入。

**核心 idea**：Self-identifiability 是贪心算法在结构化 bandit 中获得 sublinear regret 的充要条件。

## 方法详解

### 整体框架

考虑有限动作集 $\mathcal{A}$、有限上下文集 $\mathcal{X}$、有限奖励函数类 $\mathcal{F}$ 的结构化 bandit。每轮 $t$ 到达上下文 $x_t$，算法选臂 $a_t$，获得奖励 $r_t \sim \mathcal{N}(f^*(x_t,a_t), 1)$。贪心算法在每轮做最小二乘回归后取最优臂：

$$f_t = \arg\min_{f \in \mathcal{F}} \sum_{s \in [t]} (f(x_s,a_s) - r_s)^2, \quad a_t = \arg\max_{a} f_t(x_t,a)$$

### 关键设计

1. **Self-Identifiability（自可辨识性）**:

    - 功能：刻画 Greedy 成功的充分必要条件
    - 核心思路：对 $f^*$ 的每个次优臂 $a$，若所有满足 $f(a) = f^*(a)$ 的 $f \in \mathcal{F}$ 都认为 $a$ 是次优的，则称 $a$ 是 self-identifiable 的。若所有次优臂都有此性质，则实例是 self-identifiable 的
    - 设计动机：Greedy 只观察它选择的臂的奖励，若该臂的期望奖励一旦被准确估计就能判断它不是最优的，则 Greedy 自然会"摆脱"它

2. **Decoy（诱饵函数）**:

    - 功能：刻画 Greedy 失败的机制
    - 核心思路：$f_{\text{dec}} \in \mathcal{F}$ 是 $f^*$ 的 decoy 若其最优臂 $a_{\text{dec}}$ 对 $f^*$ 是次优的，且 $f_{\text{dec}}(a_{\text{dec}}) = f^*(a_{\text{dec}})$
    - 核心等价：$f^*$ 没有 decoy $\Leftrightarrow$ 问题实例是 self-identifiable 的

3. **Function-Gap 参数化**:

    - 定义 $\Gamma(f^*, \mathcal{F}) = \min_{f \neq f^*} \min_{a: f(a) \neq f^*(a)} |f^*(a) - f(a)|$
    - 正面结果的 regret 上界和负面结果的失败概率下界都用 $\Gamma$ 参数化

### 核心定理

**Theorem 3.3（StructuredMAB）**：(a) 若 self-identifiable，则 $\mathbb{E}[R(t)] \leq T_0 + (K/\Gamma)^2 \cdot O(\log t)$；(b) 若有 decoy，以概率 $\geq e^{-O(K/\Gamma^2)}$ Greedy 永远选 decoy 臂。

**Theorem 4.3（StructuredCB）**：推广到上下文 bandit，regret 上界 $(|\mathcal{X}|K/\Gamma)^2/p_0 \cdot O(\log t)$。

**Theorem 5.3（DMSO）**：MLE-based Greedy + KL model-gap，实现同样的充要刻画。

### 证明思路

**正面方向**：次优臂被选 $O(K/\Gamma^2)$ 次后，经验均值以高概率接近真值，由 self-identifiability 判定为次优，MSE 论证保证不再被选。

**负面方向**：构造独立事件 $E_1$（warm-up 误导）和 $E_2$（decoy 臂均值永远接近），$E_1 \cap E_2$ 保证永久陷入。

## 实验关键数据

### 正面/负面实例汇总

| 奖励结构 | Greedy 表现 | 解释 |
|----------|------------|------|
| 线性 bandit | 几乎所有实例失败 | 存在 decoy |
| Lipschitz bandit | 几乎所有实例失败 | 存在 decoy |
| 多项式 bandit | 几乎所有实例失败 | 存在 decoy |
| 线性 contextual + 多样上下文 | 成功 | self-identifiability 成立 |
| 线性 contextual + 低维上下文 | 可能失败 | 缺乏多样性 |
| Lipschitz contextual | 几乎所有实例失败 | 与线性 contextual 迥异 |

### 无穷函数类推广（Theorem 6.2）

| 设定 | 正面结果 | 负面结果 |
|------|---------|---------|
| eps-self-identifiable | 对数 regret $(K/\varepsilon)^2 O(\log t)$ | — |
| decoy 在 interior 中 | — | 正概率 $\geq e^{-O(K^2/\varepsilon^2)}$ 永久陷入 |

### 关键发现
- Greedy 在大多数连续奖励结构中失败是常态
- Self-identifiability 不仅让 Greedy 成功，且让所有温和非退化算法都成功——问题本身简单
- 无穷函数类的完整 iff 刻画仍是开放问题

## 亮点与洞察
- **Self-identifiability 概念**：将"何时不需要探索"形式化为全新结构性概念
- **Decoy 机制**：揭示 Greedy 失败的统一机制——结构性永久陷入
- **DMSO 推广**：MLE + KL 散度，将结论推广到 RL 等复杂设定
- **实践指导**：大规模实验中 Greedy 的成功可被 self-identifiability 解释

## 局限与展望
- $(K/\Gamma)^2$ 常数可能很大，实际收敛慢
- 失败概率 $p_{\text{dec}}$ 可能很小
- 无穷函数类完整 iff 刻画仍开放
- 不保证计算效率

## 相关工作与启发
- **vs Bastani et al. (2021)**：他们线性 contextual + 多样上下文有 near-optimal regret，本文更一般但常数更弱
- **vs Banihashem et al. (2023)**：他们刻画无结构 K-armed bandit 的 Greedy 失败概率，本文推广到任意结构
- **vs Foster et al. (2021)**：本文在 DMSO 框架下完成 Greedy 分析首次刻画

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次给出任意有限奖励结构下 Greedy 的完整 iff 刻画
- 实验充分度: ⭐⭐⭐ 纯理论工作，应用实例分析详尽
- 写作质量: ⭐⭐⭐⭐⭐ 概念清晰、定理精确、证明层层递进
- 价值: ⭐⭐⭐⭐⭐ 解决了 bandit 理论中长期开放的基本问题

<!-- RELATED:START -->

## 相关论文

- [Prompt Tuning Decision Transformers with Structured and Scalable Bandits](prompt_tuning_decision_transformers_with_structured_and_scalable_bandits.md)
- [Exploration via Feature Perturbation in Contextual Bandits](exploration_via_feature_perturbation_in_contextual_bandits.md)
- [Learning in Stackelberg Mean Field Games: A Non-Asymptotic Analysis](learning_in_stackelberg_mean_field_games_a_non-asymptotic_analysis.md)
- [Sequential Multi-Agent Dynamic Algorithm Configuration](sequential_multi-agent_dynamic_algorithm_configuration.md)
- [Tractable Multinomial Logit Contextual Bandits with Non-Linear Utilities](tractable_multinomial_logit_contextual_bandits_with_non-linear_utilities.md)

<!-- RELATED:END -->
