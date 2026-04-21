---
title: >-
  [论文解读] Learning-Augmented Algorithms for MTS with Bandit Access to Multiple Predictors
description: >-
  [ICML2025][Metrical Task Systems] 在度量任务系统(MTS)中，当算法仅能以 bandit 方式（每步只查询一个启发式且需连续查询 $m$ 步才能观测状态）访问 $\ell$ 个启发式时，本文给出了 regret 为 $O(\text{OPT}^{2/3})$ 的算法，并证明该界是紧的。
tags:
  - ICML2025
  - Metrical Task Systems
  - 学习增强算法
  - Bandit反馈
  - 启发式组合
  - 竞争比
  - Regret界
---

# Learning-Augmented Algorithms for MTS with Bandit Access to Multiple Predictors

**会议**: ICML2025  
**arXiv**: [2506.05479](https://arxiv.org/abs/2506.05479)  
**代码**: 无  
**领域**: 其他/在线算法  
**关键词**: Metrical Task Systems, 学习增强算法, Bandit反馈, 启发式组合, 竞争比, Regret界

## 一句话总结

在度量任务系统(MTS)中，当算法仅能以 bandit 方式（每步只查询一个启发式且需连续查询 $m$ 步才能观测状态）访问 $\ell$ 个启发式时，本文给出了 regret 为 $O(\text{OPT}^{2/3})$ 的算法，并证明该界是紧的。

## 研究背景与动机

**现状**：度量任务系统 (MTS) 是在线算法中最通用的框架之一，能建模缓存、$k$-server、滑雪租赁、凸体追逐等经典问题。在学习增强算法范式下，"组合启发式"(Combining Heuristics) 是一项核心技术——给定 $\ell$ 个启发式 $H_1, \dots, H_\ell$（各自可能针对不同输入类型设计），希望构建一个在线算法，性能接近事后最优启发式。

**已有结果**：在 full-feedback 设定下（每步可查询所有启发式的状态），Blum & Burch (2000) 使用 HEDGE 算法实现 $(1+\epsilon)$-competitive。Antoniadis et al. (2023a) 研究了一种"类 bandit"设定，但要求启发式主动报告移动代价。

**痛点**：
1. 查询所有启发式代价高昂，特别是使用重量级 ML 模型时
2. 在真正的 bandit 设定中，每步只能查询一个启发式，且启发式 $H_i$ 在时刻 $t$ 的代价 $c_t(s_t^i) + d(s_{t-1}^i, s_t^i)$ 既包含服务代价也包含移动代价——除非连续 $m$ 步查询同一启发式，否则无法估计其代价
3. 经典 bandit 方法（如 EXP3）无法直接应用，因为缺乏无偏损失估计器

**核心 idea**：设计一种交替探索-利用的算法，在探索阶段通过连续 $m$ 步查询同一启发式获取代价信息，同时利用"非正常步"(improper steps) 和 MTS 式舍入控制切换代价，实现 regret 依赖于最优启发式代价 $\text{OPT}_{\leq 0}$ 而非时间长度 $T$。

## 方法详解

### 问题形式化

给定度量空间 $(M, d)$、直径 $D = \max_{s,s'} d(s,s')$、$\ell$ 个启发式。算法具有 **$m$-delayed bandit access**：
- 每步只能查询一个启发式 $H_i$
- 仅当 $H_i$ 在 $t-m+2, \dots, t$ 连续被查询时，才能获得其状态 $s_t^i$
- 算法保留对输入实例（代价函数 $c_t$）的完全访问

启发式代价：$f_t(i) = c_t(s_t^i) + d(s_{t-1}^i, s_t^i) \in [0, 2D]$

### 算法框架：探索-利用交替

算法核心是经典 MAB 探索-利用策略的 MTS 改编版，内部使用 HEDGE/SHARE 作为全信息学习算法 $\bar{A}$：

1. **初始化**：采样 $\beta_t \sim \text{Bernoulli}(\epsilon)$ 控制探索概率，$e_t \sim U(\{1,\dots,\ell\})$ 选择探索目标
2. **利用步 (Exploitation)**：跟随内部分布 $x_t$ 采样的启发式行动
3. **探索触发**：当 $\beta_t = 1$ 时，进入 $m$ 步的引导阶段（bootstrapping），连续查询 $H_{e_{t+m}}$
4. **探索步 (Exploration)**：在第 $t+m$ 步获得 $f_{t+m}(e_{t+m})$，构造损失向量更新分布

### 关键设计

**损失向量构造**：在探索步获得代价后，构造归一化损失向量

$$g_{t+m}^{e_{t+m}}(i) = \begin{cases} f_{t+m}(e_{t+m}) / 2D & \text{if } i = e_{t+m} \\ 0 & \text{otherwise} \end{cases}$$

保证 $g_{t+m} \in [0,1]^\ell$，满足 HEDGE 输入要求。

**MTS 式舍入 (Proposition 2.2)**：利用 Earth Mover Distance 控制分布切换代价，避免独立随机采样导致的 $O(T)$ 切换代价：

$$\mathbb{E}\left[\sum_t c_t(s_t^{i_t}) + d(s_{t-1}^{i_{t-1}}, s_t^{i_t})\right] \leq \sum_t f_t^T x_t + \frac{D}{2}\|x_{t-1} - x_t\|_1$$

**非正常步 (Improper Steps)**：在引导阶段和探索步，算法不跟随任何启发式，而是从已知的最近利用状态出发执行贪心步。这是实现 regret 依赖 $\text{OPT}_{\leq 0}$ 而非 $T$ 的关键。

**HEDGE 稳定性利用 (Property 2.3)**：分布变化量被损失控制

$$\|x_{t-1} - x_t\|_1 \leq \eta \cdot g_{t-1}^T x_{t-1}$$

用于约束切换代价。

### 参数选择

探索率 $\epsilon$ 和学习率 $\eta$ 联合优化，取 $\epsilon = \Theta(\text{OPT}_{\leq 0}^{-1/3})$，在探索代价、学习精度、切换代价之间取平衡。

## 理论结果

| 定理 | 设定 | 结果 | 说明 |
|------|------|------|------|
| Thm 1.1 | 静态基准 ($k=0$) | $\mathbb{E}[\text{ALG}] \leq \text{OPT}_{\leq 0} + O(\text{OPT}_{\leq 0}^{2/3})$ | 竞争比趋近 1 |
| Thm 1.2 | 动态基准 ($k$ 次切换) | $\mathbb{E}[\text{ALG}] \leq \text{OPT}_{\leq k} + \tilde{O}(k^{1/3} \cdot \text{OPT}_{\leq k}^{2/3})$ | 允许基准切换启发式 |
| Thm 1.3 | 下界 ($m=2$) | $\mathbb{E}[\text{ALG}] \geq \text{OPT}_{\leq 0} + \tilde{\Omega}(\text{OPT}_{\leq 0}^{2/3})$ | 基于 Dekel et al. 构造，证明上界紧 |

**参数依赖性**：Regret 的精确缩放为 $(Dk\ell \ln \ell)^{1/3} m^{2/3} \cdot \text{OPT}^{2/3}$，其中 $(Dk\ell)^{1/3}$ 部分被证明几乎最优。

**与 Memory-Bounded Bandits 的桥接**：算法可适配 $m$-memory bounded bandit 设定，获得 $O(T^{2/3})$ regret，稍微改进了 Arora et al. (2012) 对 $m$ 的依赖。

## 亮点与洞察

- **紧界**：上界 $O(\text{OPT}^{2/3})$ 与下界 $\tilde{\Omega}(\text{OPT}^{2/3})$ 匹配（忽略对数因子），给出了该问题的本质复杂度
- **Regret 依赖 OPT 而非 T**：当最优启发式表现很好（$\text{OPT} \ll T$）时，界远优于经典 bandit 的 $O(T^{2/3})$
- **非正常步的必要性**：即使 $m=2$（最小延迟），improper steps 也是获得 OPT-dependent regret 的必要条件，揭示了 MTS 与标准 bandit 问题的本质差异
- **鲁棒化应用**：将经典 $\rho$-competitive 在线算法加入启发式集合后，组合算法在最坏情况下仍保持 $(1+o(1))\rho$-competitive，同时在好实例上可利用 ML 预测
- **下界构造的技巧**：经典 Dekel et al. 下界无法直接使用，因为 MTS 算法具有非正常动作和 1-step look-ahead 两个额外优势，需要精心改造

## 局限与展望

- **纯理论工作**：缺少实验验证，未在具体 MTS 实例（如缓存、k-server）上评估实际性能
- **参数需预知 OPT**：探索率 $\epsilon$ 依赖于未知的 $\text{OPT}_{\leq 0}$，虽可用 doubling trick 解决，但增加了常数因子
- **对数间隙**：上下界之间存在对数因子差距，是否可消除尚不清楚
- **$D, \ell, m$ 为常数假设**：主定理假设这些参数为常数；当 $\ell$ 或 $m$ 很大时，实际性能可能退化显著
- **自适应对手**：仅考虑 oblivious adversary，对 adaptive adversary 的推广未探讨

## 相关工作与启发

- **Arora et al. (2012)**：Memory-Bounded Adversaries 的 bandit 学习，$O(\mu T^{2/3})$ regret，本文在 MTS 设定下改进了对 $m$ 的依赖
- **Dekel et al. (2013)**：Bandits with Switching Costs 的 $\Omega(T^{2/3})$ 下界构造，本文下界证明的技术基础
- **Antoniadis et al. (2023a)**：MTS 动态启发式组合，bandit-like（但更简单的）设定下的 $(1+\epsilon)$-competitive 算法
- **Blum & Burch (2000)**：Full-feedback 下 MTS 启发式组合的经典结果，使用 HEDGE 实现 $(1+\epsilon)$-competitive
- **Lykouris & Vassilvitskii (2021)**：学习增强算法的开山之作，提出鲁棒化框架

## 评分
- 新颖性: ⭐⭐⭐⭐ — Bandit access 下 MTS 启发式组合是新设定，improper steps 的必要性是新颖洞察
- 实验充分度: ⭐⭐ — 纯理论，无实验
- 写作质量: ⭐⭐⭐⭐ — 问题动机清晰，技术脉络连贯，与相关工作对比充分
- 价值: ⭐⭐⭐⭐ — 给出紧界，解决了自然而重要的开放问题

<!-- RELATED:START -->

## 相关论文

- [Fixed-Confidence Multiple Change Point Identification under Bandit Feedback](fixed-confidence_multiple_change_point_identification_under_bandit_feedback.md)
- [Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/others/learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [Learning-Augmented Online Bipartite Fractional Matching](../../NeurIPS2025/others/learning-augmented_online_bipartite_fractional_matching.md)
- [Near-Optimal Consistency-Robustness Trade-Offs for Learning-Augmented Online Knapsack Problems](near-optimal_consistency-robustness_trade-offs_for_learning-augmented_online_kna.md)
- [Learning-Augmented Hierarchical Clustering](learning-augmented_hierarchical_clustering.md)

<!-- RELATED:END -->
