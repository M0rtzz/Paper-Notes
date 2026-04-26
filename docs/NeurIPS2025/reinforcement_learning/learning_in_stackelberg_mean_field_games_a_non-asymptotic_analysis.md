---
title: >-
  [论文解读] Learning in Stackelberg Mean Field Games: A Non-Asymptotic Analysis
description: >-
  [NeurIPS 2025][Stackelberg博弈] 提出首个具有非渐近收敛保证的单循环Actor-Critic算法AC-SMFG，用于求解Stackelberg平均场博弈（SMFG），收敛速率达到 $\widetilde{\mathcal{O}}(k^{-1/2})$。
tags:
  - NeurIPS 2025
  - Stackelberg博弈
  - 平均场博弈
  - Actor-Critic
  - 非渐近收敛
  - 双层优化
---

# Learning in Stackelberg Mean Field Games: A Non-Asymptotic Analysis

**会议**: NeurIPS 2025  
**arXiv**: [2509.15392](https://arxiv.org/abs/2509.15392)  
**代码**: 有（补充材料中提供）  
**领域**: 强化学习  
**关键词**: Stackelberg博弈, 平均场博弈, Actor-Critic, 非渐近收敛, 双层优化

## 一句话总结

提出首个具有非渐近收敛保证的单循环Actor-Critic算法AC-SMFG，用于求解Stackelberg平均场博弈（SMFG），收敛速率达到 $\widetilde{\mathcal{O}}(k^{-1/2})$。

## 研究背景与动机

**领域现状**: 平均场博弈（MFG）为无限理性智能体间的策略交互提供了建模框架，广泛应用于资源分配、电信和电力系统优化。Stackelberg MFG进一步引入层次结构，一个leader智能体影响follower群体。

**现有痛点**: 现有SMFG算法存在三大问题：(i) 对leader和follower目标做过强的独立性假设；(ii) 嵌套循环结构导致样本效率低下；(iii) 缺乏有限时间收敛保证，只有渐近收敛。

**核心矛盾**: SMFG本质上是双层优化——leader的目标依赖于follower群体的均衡响应，而该响应反过来又受leader策略影响。这种耦合使得算法设计和分析极其困难。

**本文目标**: 设计一个简单实用且具有非渐近收敛保证的单循环算法来求解SMFG。

**切入角度**: 将SMFG视为双层优化问题，通过多时间尺度分析协调leader策略、follower策略和平均场的更新。

**核心 idea**: 利用"梯度对齐"条件放松leader-follower独立假设，结合PL条件处理非凸下层问题，实现比标准双层优化更优的收敛速率。

## 方法详解

### 整体框架

AC-SMFG是一个单循环Actor-Critic算法。SMFG建模为元组 $({\cal S},{\cal A},{\cal B},{\cal P},r_f,r_l,\gamma)$，leader和follower各自维护策略参数 $\omega$ 和 $\theta$，使用softmax参数化：

$$\phi_\omega(b|s) = \frac{\exp(\omega(s,b))}{\sum_{b'}\exp(\omega(s,b'))}, \quad \pi_\theta(a|s) = \frac{\exp(\theta(s,a))}{\sum_{a'}\exp(\theta(s,a'))}$$

### 关键设计

**模块1: Leader策略更新（最慢时间尺度）**
- 功能：更新leader策略参数 $\omega$ 以最大化Stackelberg目标 $\Phi(\phi) = J_l(\phi, \mu^*(\phi))$
- 核心公式：$\omega_{k+1} = \omega_k + \zeta_k \nabla_\omega \log\phi_{\omega_k}(b_k|s_k)(r_l(s_k,b_k,\hat{\mu}_k) + \gamma \hat{V}_{l,k}(s_{k+1}) - \hat{V}_{l,k}(s_k))$
- 设计动机：使用策略梯度方法，步长 $\zeta_k$ 最小以确保在follower和平均场收敛后再调整leader

**模块2: Follower策略更新（中间时间尺度）**
- 功能：更新follower策略参数 $\theta$ 以最大化正则化累积奖励 $J_f(\pi,\phi,\mu)$
- 核心公式：类似形式的策略梯度，步长 $\alpha_k > \zeta_k$
- 设计动机：follower需要比leader更快收敛以近似最优响应

**模块3: 平均场更新（快时间尺度）**
- 功能：跟踪follower群体的聚合行为 $\mu$
- 核心公式：$\hat{\mu}_{k+1} = \Pi_{\Delta_{\cal S}}(\hat{\mu}_k + \xi_k(e_{\bar{s}_k} - \hat{\mu}_k))$
- 设计动机：通过第二条样本轨迹估计平稳分布

**模块4: 值函数（Critic）更新（最快时间尺度）**
- 功能：通过TD学习估计leader和follower的值函数
- 步长 $\beta_k$ 最大，确保值函数估计最先收敛

### 损失函数/训练策略

- 采用四时间尺度设计：$\zeta_k \leq \xi_k \leq \alpha_k \leq \beta_k$，步长均为 $\frac{c}{(k+1)^{1/2}}$
- 使用带正则化权重 $\tau$ 的熵正则化确保Assumption 4（收缩条件）成立
- 每次迭代仅需两个样本（一个用于占据测度，一个用于平稳分布）

## 实验关键数据

### 主实验

| 环境 | AC-SMFG | OneByOne | CuiKoeppl | ADAGE |
|------|---------|----------|-----------|-------|
| Market Entrance (sell skew) | **最快收敛** | 慢收敛 | 振荡/次优 | 中等 |
| Shop Positioning ($c=0.05$) | **Leader最快** | 较慢 | 较慢 | Follower不稳定 |
| Equilibrium Pricing ($i=5$) | **显著优于其他SMFG** | 收敛差 | 收敛差 | 略优(函数逼近) |

### 消融实验

| 分析项 | 结论 |
|--------|------|
| 不同leader配置 | AC-SMFG在各配置下均能收敛到合适MFE |
| 连续状态/动作空间 | 通过函数逼近（高斯平均场假设）兼容连续空间 |
| 收敛速率对比 | $\widetilde{\mathcal{O}}(k^{-1/2})$ 优于hong2023two的 $\widetilde{\mathcal{O}}(k^{-2/5})$ |

### 关键发现

1. AC-SMFG在所有环境中均展示了显著更快的leader收敛速度和更平滑的收敛轨迹
2. follower行为越受leader影响的环境，AC-SMFG的提升越显著
3. 连续空间实验表明AC-SMFG与函数逼近（神经网络策略+高斯平均场）兼容良好

## 亮点与洞察

1. **首个非渐近保证**: 这是SMFG领域首个具有有限时间/有限样本复杂度保证的算法
2. **超越双层优化**: 在更困难的设置中（下层非凸）实现了比标准双层优化（下层强凸）更好的收敛速率
3. **技术创新**: 开发了处理PL条件（而非强凸）下层的分析技巧，可推广到更一般的双层优化
4. **梯度对齐条件**: 提出比现有"leader-follower独立"假设更弱的假设，允许更广泛的耦合情形

## 局限与展望

1. 依赖足够大的正则化权重 $\tau$，无法求解原始非正则化问题的均衡（这是整个正则化MFG领域的共性问题）
2. 收敛到驻点而非全局最优，因为复合函数 $\Phi$ 不满足梯度支配条件
3. follower同质性假设限制了实际应用场景
4. MFG作为 $N$ 智能体马尔可夫博弈的渐近近似，在follower数量较少时误差可能不可忽略

## 相关工作与启发

- **与cui2024learning对比**: 后者是嵌套循环+虚拟博弈方法，仅有渐近收敛；本文是单循环+非渐近保证
- **与hong2023two对比**: 标准双层优化方法，在下层强凸下达到 $\widetilde{\mathcal{O}}(k^{-2/5})$；本文在更弱假设下达到 $\widetilde{\mathcal{O}}(k^{-1/2})$
- **启发**: PL条件下双层优化的分析技巧可推广到更一般的分层决策问题

## 评分

⭐⭐⭐⭐ (4/5)

理论贡献扎实，首次为SMFG提供非渐近保证，技术创新（PL条件处理）有独立价值。实验场景充分展示了方法优势。局限性在于正则化假设和同质性假设限制了直接应用范围。

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Solving Continuous Mean Field Games: Deep Reinforcement Learning for Non-Stationary Dynamics](solving_continuous_mean_field_games_deep_reinforcement_learning_for_non-stationa.md)
- [\[NeurIPS 2025\] Last Iterate Convergence in Monotone Mean Field Games](last_iterate_convergence_in_monotone_mean_field_games.md)
- [\[NeurIPS 2025\] Non-convex Entropic Mean-Field Optimization via Best Response Flow](non-convex_entropic_mean-field_optimization_via_best_response_flow.md)
- [\[NeurIPS 2025\] Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)
- [\[NeurIPS 2025\] Scalable Neural Incentive Design with Parameterized Mean-Field Approximation](scalable_neural_incentive_design_with_parameterized_mean-field_approximation.md)

<!-- RELATED:END -->
