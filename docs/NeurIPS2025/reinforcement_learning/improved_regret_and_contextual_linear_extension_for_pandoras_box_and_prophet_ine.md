---
title: >-
  [论文解读] Improved Regret and Contextual Linear Extension for Pandora's Box and Prophet Inequality
description: >-
  [NeurIPS 2025][Pandora's Box] 本文针对在线 Pandora's Box 问题提出新算法，将 regret 从 $\widetilde{O}(n\sqrt{T})$ 改进到 $\widetilde{O}(\sqrt{nT})$（匹配下界），并首次提出 contextual linear 扩展实现 $\widetilde{O}(nd\sqrt{T})$ regret。
tags:
  - NeurIPS 2025
  - Pandora's Box
  - prophet inequality
  - online learning
  - regret bound
  - contextual linear bandits
---

# Improved Regret and Contextual Linear Extension for Pandora's Box and Prophet Inequality

**会议**: NeurIPS 2025  
**arXiv**: [2505.18828](https://arxiv.org/abs/2505.18828)  
**代码**: 无  
**领域**: reinforcement_learning  
**关键词**: Pandora's Box, prophet inequality, online learning, regret bound, contextual linear bandits

## 一句话总结
本文针对在线 Pandora's Box 问题提出新算法，将 regret 从 $\widetilde{O}(n\sqrt{T})$ 改进到 $\widetilde{O}(\sqrt{nT})$（匹配下界），并首次提出 contextual linear 扩展实现 $\widetilde{O}(nd\sqrt{T})$ regret。

## 研究背景与动机

**领域现状**：Pandora's Box——$n$ 个盒子各有成本和未知奖励分布，需决定开启顺序和停止时机。Weitzman (1978) 给出已知分布下的最优策略。

**现有痛点**：在线设定中 Agarwal et al. (2024) 达到 $\widetilde{O}(Un\sqrt{T})$（$U$ 可达 $n$），下界 $\Omega(\sqrt{nT})$，差距达 $n$ 倍。

**核心矛盾**：先前方法以固定量移动概率质量，TV 距离界太松；$\text{Term}_{t,i}$ 被粗糙地用 $U \cdot \text{TV}$ 约束。

**本文目标** 闭合 non-contextual 的 $n$ 差距；建立 contextual linear 模型获得 $\sqrt{T}$ 型 regret。

**切入角度**：Bernstein 型 DKW 不等式自适应调整概率质量 + 效用函数导数精细分析。

**核心 idea**：自适应乐观分布 + 两区域导数分析 = minimax 最优 regret。

## 方法详解

### 整体框架

每轮按阈值向量 $\sigma_t$ 运行 Weitzman 策略。核心是从样本构建乐观分布 $\hat{\mathcal{E}}_t$ 计算 $\sigma_t$。

### 关键设计

1. **自适应乐观分布（Bernstein-type）**:

    - 经验 CDF 加 Bernstein 置信区间向下移动：$F_{\hat{\mathcal{E}}}(x) = \max\{0, F_\mathcal{E}(x) - \sqrt{2F_\mathcal{E}(x)(1-F_\mathcal{E}(x))L/m} - L/m\}$
    - 与固定移动 $\sim 1/\sqrt{m}$ 不同，尾部区域自动少移动

2. **效用函数导数两区域分析**:

    - $\widetilde{R}_i(\sigma_t; z)$ 关于 $z$ 是 1-Lipschitz 且单调的
    - 大 $z$ 区域：$\sqrt{F(1-F)}$ 在 $z$ 大时自动收缩
    - 小 $z$ 区域：导数约束 $\partial_z \widetilde{R}_i \leq \prod_{j<i} F_{D_j}(z)/Q_{t,i}$
    - Cauchy-Schwarz + telescoping sum 得到 $\widetilde{O}(\sqrt{\sum_i Q_{t,i}/m_{t,i}})$ per-round regret

3. **Contextual Linear 扩展**:

    - 期望奖励 $\mu_{t,i} = \theta_i^\top x_{t,i}$，噪声固定
    - Ridge regression + value-optimistic 经验分布：$\hat{z} = \min\{1, v - \text{LCB} + \text{UCB}\}$
    - Regret 分解为乐观重加权 + value shift + 经验 vs 真实分布三项

### 核心结果

| 问题 | 本文 | 先前最优 | 下界 |
|------|------|---------|------|
| Non-contextual Pandora's Box | $\widetilde{O}(\sqrt{nT})$ | $\widetilde{O}(Un\sqrt{T})$ | $\Omega(\sqrt{nT})$ |
| Contextual Linear PB | $\widetilde{O}(nd\sqrt{T})$ | $\widetilde{O}(nT^{5/6})$ | 开放 |
| Non-contextual Prophet | $\widetilde{O}(\sqrt{nT})$ | $\widetilde{O}(n\sqrt{T})$ | $\Omega(\sqrt{T})$ |

## 实验关键数据

### 技术对比

| 技术 | Agarwal et al. | 本文 |
|------|---------------|------|
| 乐观分布 | 固定质量 $\sim 1/\sqrt{m}$ | Bernstein-type 自适应 |
| Regret 分解 | $U \cdot \text{TV}$ | 两区域导数分析 |
| 效用函数 | 未用 Lipschitz | 1-Lipschitz + 单调 |
| 结果 | $\widetilde{O}(Un\sqrt{T})$ | $\widetilde{O}(\sqrt{nT})$ |

### 关键发现
- Bernstein 型构造在 CDF 集中性好的区域少移动质量是核心改进
- 1-Lipschitz + 单调性避免了 $U$ 因子
- Contextual 中 $n$ 线性依赖可能不可避免
- Prophet Inequality $\widetilde{O}(\sqrt{T})$ 可达性仍开放

## 亮点与洞察
- **自适应乐观分布**最精妙：尾部自动收缩
- **两区域导数分析**利用 Pandora 结构的 telescoping sum
- **Value-optimistic 去偏**处理 contextual 自然优雅
- 可迁移到需要学全分布的 bandit 问题

## 局限与展望
- Contextual Linear 中 regret 对 $n$ 的线性依赖是否可改进？即使所有盒子共享 $\theta$ 也可能不可避免（需为每个盒子学独立噪声分布）
- Prophet Inequality 的 minimax 最优 regret 仍未确定——$\widetilde{O}(\sqrt{nT})$ 上界 vs $\Omega(\sqrt{T})$ 下界间有 $\sqrt{n}$ gap
- Jin et al. (2024) 表明 Prophet 的最优样本复杂度 $\widetilde{O}(1/\epsilon^2)$ 与 $n$ 无关，暗示 $\widetilde{O}(\sqrt{T})$ 可能可达
- 缺乏实验验证算法在有限 $T$ 下的实际表现
- 所有结果假设奖励分布独立，实际应用中可能存在相关性
- 当 $d$ 较大（$d = \Omega(T^{1/3})$）时 contextual 结果不优于先前方法

### 算法伪代码概述

核心循环：每轮 $t$ 对每个盒子 $i$ 构建乐观分布 $\hat{\mathcal{E}}_{t,i}$ → 计算阈值 $\sigma_{t,i}$ → 运行 Weitzman 算法 → 观察打开盒子的奖励 → 更新计数器 $m_{t+1,i}$。整体框架遵循 optimism in the face of uncertainty。

## 相关工作与启发
- **vs Agarwal et al. (2024)**：固定质量导致 $U$ 因子，本文自适应构造消除
- **vs Atsidakou et al. (2024)**：contextual $T^{5/6}$，本文 $\sqrt{T}$
- **vs Gatmiry et al. (2024)**：下界 $\Omega(\sqrt{nT})$，本文匹配
- **vs Guo et al. (2021)**：他们用类似 Bernstein 构造做 PAC 保证，本文将其适配到 regret 分析中，分析方式本质不同
- **vs Weitzman (1978)**：经典最优策略需已知分布，本文学习分布同时保持最优性

### 算法复杂度

每轮需要为每个盒子构建乐观分布并求解 threshold，计算量为 $O(n \cdot m_{\max})$，其中 $m_{\max}$ 是最大样本数。与 Agarwal et al. 的算法计算量相当，改进纯在统计效率上。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Bernstein 乐观分布和两区域分析全新，contextual 扩展首创
- 实验充分度: ⭐⭐⭐ 纯理论，匹配下界是最强验证
- 写作质量: ⭐⭐⭐⭐ 技术概览清晰但公式密集
- 价值: ⭐⭐⭐⭐⭐ 闭合重要理论差距，开辟 contextual Pandora's Box 新方向

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Improved Regret Bounds for GP-UCB in Bayesian Optimization](improved_regret_bounds_for_gaussian_process_upper_confidence_bound_in_bayesian_o.md)
- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](thompson_sampling_for_multi-objective_linear_contextual_bandit.md)
- [\[NeurIPS 2025\] Tractable Multinomial Logit Contextual Bandits with Non-Linear Utilities](tractable_multinomial_logit_contextual_bandits_with_non-linear_utilities.md)
- [\[NeurIPS 2025\] Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update](generalized_linear_bandits_almost_optimal_regret_with_one-pass_update.md)
- [\[NeurIPS 2025\] Exploration via Feature Perturbation in Contextual Bandits](exploration_via_feature_perturbation_in_contextual_bandits.md)

<!-- RELATED:END -->
