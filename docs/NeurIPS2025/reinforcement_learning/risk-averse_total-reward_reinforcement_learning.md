---
title: >-
  [论文解读] Risk-Averse Total-Reward Reinforcement Learning
description: >-
  [NeurIPS 2025][风险规避RL] 提出了面向无折扣总奖励准则(TRC)的风险规避Q-learning算法（ERM-TRC和EVaR-TRC），利用ERM的可引出性(elicitability)将Bellman算子转化为随机梯度下降形式，并证明了算法的收敛保证。
tags:
  - NeurIPS 2025
  - 风险规避RL
  - 总奖励准则
  - Q-learning
  - 熵风险度量(ERM)
  - 熵风险值(EVaR)
---

# Risk-Averse Total-Reward Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2506.21683](https://arxiv.org/abs/2506.21683)  
**代码**: [有](https://github.com/suxh2019/ERM_EVaR_Q)  
**领域**: 强化学习 / 风险规避  
**关键词**: 风险规避RL, 总奖励准则, Q-learning, 熵风险度量(ERM), 熵风险值(EVaR)

## 一句话总结

提出了面向无折扣总奖励准则(TRC)的风险规避Q-learning算法（ERM-TRC和EVaR-TRC），利用ERM的可引出性(elicitability)将Bellman算子转化为随机梯度下降形式，并证明了算法的收敛保证。

## 研究背景与动机

风险规避强化学习在自动驾驶、机器人手术、医疗和金融等高风险应用中至关重要。现有风险规避RL方法主要关注**折扣无限时间horizon**目标，但许多实际任务（如机器人、游戏）并没有自然的折扣理由——它们有吸收终止状态，未来奖励不应被折扣。

**总奖励准则(TRC)** 是一种不折扣未来奖励的目标函数，推广了随机最短路径和最长路径问题。TRC的核心挑战在于：

**分布估计困难**: 风险规避RL需要评估回报的完整分布，而非仅期望值

**Bellman算子非压缩**: 风险规避TRC的Bellman算子可能不是压缩映射，不能直接套用传统model-free方法

**有界性条件**: 需要额外的有界条件来保证收敛，且风险参数β过大时值函数可能无界

现有model-based方法（如线性规划方法）虽然有效，但需要完整的转移概率，无法扩展到大规模问题。本文首次提出了面向TRC的model-free Q-learning算法。

## 方法详解

### 整体框架

本文设计了两个Q-learning算法：
- **ERM-TRC Q-learning**: 面向熵风险度量(ERM)目标
- **EVaR-TRC Q-learning**: 面向熵风险值(EVaR)目标，通过将EVaR分解为一系列ERM问题来求解

### 关键设计

#### 1. 基于可引出性的ERM Bellman算子

**核心思路**: 利用ERM的可引出性(elicitability)，将Bellman算子重新定义为一个回归问题的解：

$$\hat{B}_\beta q(s,a) = \arg\min_{y \in \mathbb{R}} \mathbb{E}^{a,s}[\ell_\beta(r(s,a,\tilde{s}_1) + \max_{a'} q(\tilde{s}_1,a',\beta) - y)]$$

其中损失函数为指数损失：$\ell_\beta(z) = \beta^{-1}(\exp(-\beta z) - 1) + z$

**设计动机**: 标准Q-learning可以看作二次损失函数的随机梯度下降。类比地，ERM Q-learning沿指数损失函数$\ell_\beta$的导数进行梯度下降，使得算法可以在model-free设置下直接从样本估计ERM值函数。

#### 2. ERM-TRC Q-learning的更新规则

Q值更新遵循指数损失梯度：

$$\tilde{q}_{i+1}(s,a,\beta) = \tilde{q}_i(s,a,\beta) - \tilde{\eta}_i \cdot (\exp(-\beta \cdot \tilde{z}_i(\beta)) - 1)$$

其中TD残差 $\tilde{z}_i(\beta) = r(s,a,s'_i) + \max_{a'} \tilde{q}_i(s'_i,a',\beta) - \tilde{q}_i(s,a,\beta)$

**有界条件**: 当TD残差超出$[z_{\min}, z_{\max}]$范围时，说明β值过大导致q值无界，算法返回$-\infty$。

#### 3. EVaR-TRC Q-learning

EVaR不具有动态一致性，无法直接使用Bellman方程。本文将EVaR优化问题分解为一系列离散化β值上的ERM问题：

$$(\pi^*, \beta^*) \in \arg\max_{(\pi,\beta) \in \Pi \times \mathcal{B}(\beta_0,\delta)} h(\pi, \beta)$$

通过构造有限β值集合$\mathcal{B}(\beta_0, \delta)$，对每个β运行ERM Q-learning，最后选择使$h(\pi,\beta)$最大的策略，得到$\delta$-最优EVaR策略。

### 收敛性分析

**定理4.2 (ERM-TRC收敛)**: 在标准假设（每个状态动作对无限次访问、步长条件$\sum \eta_i = \infty, \sum \eta_i^2 < \infty$）和TD残差有界条件下，ERM Q-learning几乎必然收敛到最优值函数的固定点。

证明的三个关键技术差异：
1. 不依赖压缩映射性质，而是利用Bellman算子的**单调性**
2. 需要额外的有界性条件
3. 指数损失函数的强凸性和Lipschitz连续性提供了收敛保证（强凸常数$l=\beta\exp(-\beta z_{\max})$，Lipschitz常数$L=\beta\exp(-\beta z_{\min})$）

## 实验关键数据

### 主实验：Cliff Walking环境

实验在两个表格域验证：Cliff Walking (CW)和Gambler's Ruin (GR)。

| 环境 | α (风险水平) | 回报均值 | 回报标准差 | 回报范围 |
|------|------------|---------|-----------|---------|
| CW | α=0.2 | 1.92 | 0.228 | (0, 2] |
| CW | α=0.6 | -0.074 | 0.228 | (-1, 2] |

α=0.2时（更风险规避），agent完全避免跌落悬崖，回报几乎全在(0,2]；α=0.6时agent可能跌落悬崖导致负回报。

### 收敛实验（消融）

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 6个随机种子，α=0.2 | EVaR标准差=0.015 | 算法在不同种子间收敛到相似解 |
| CW域，~20000样本 | EVaR差值→0 | Q-learning收敛到LP最优值 |
| GR域，~20000样本 | EVaR差值→0 | 同样收敛到LP最优值 |

### 关键发现

1. **策略可视化**: 不同α值产生可解释的不同策略——α小时agent绕行远离悬崖，α大时agent更冒险走近路
2. **收敛稳定性**: 6个随机种子的标准差仅0.015，说明算法稳定可靠
3. **与LP基准一致**: Q-learning在约20000个样本后收敛到与线性规划方法相同的最优值

## 亮点与洞察

1. **理论贡献突出**: 首个面向无折扣总奖励准则的model-free风险规避RL算法，填补了重要理论空白
2. **可引出性的巧妙利用**: 将ERM的可引出性转化为随机梯度下降框架，使model-free学习成为可能
3. **有界检测机制**: TD残差越界时返回$-\infty$的设计巧妙地检测了β过大导致的发散情况
4. **EVaR的离散化处理**: 将不满足动态一致性的EVaR问题优雅地分解为一系列ERM子问题

## 局限与展望

1. **仅限表格设置**: 当前算法和分析局限于表格表示，未涉及函数近似（如深度神经网络）
2. **参数选择依赖先验**: $z_{\min}$/$z_{\max}$的选择需要平衡——过小可能找不到好解，过大则发散检测过慢
3. **β₀的选择**: EVaR算法需要选择初始β₀，可能需要多次Q-learning运行才能确定合适值
4. **扩展到连续动作空间**: 未讨论如何扩展到连续状态/动作空间

## 相关工作与启发

- 与Hau等人(2024)的折扣ERM Q-learning相比，本文处理了更具挑战性的无折扣设置
- 可引出性+SGD的思路可能推广到其他可引出风险度量（如CVaR的近似）
- 未来可结合DQN等深度RL方法，将算法扩展到大规模问题

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (首个TRC model-free风险规避RL算法)
- 实验充分度: ⭐⭐⭐ (仅表格域验证)
- 写作质量: ⭐⭐⭐⭐ (理论严谨，结构清晰)
- 价值: ⭐⭐⭐⭐ (重要理论基础工作)

<!-- RELATED:START -->

## 相关论文

- [Risk-Averse Constrained Reinforcement Learning with Optimized Certainty Equivalents](risk-averse_constrained_reinforcement_learning_with_optimized_certainty_equivale.md)
- [Reward-Aware Proto-Representations in Reinforcement Learning](reward-aware_proto-representations_in_reinforcement_learning.md)
- [Maximum Total Correlation Reinforcement Learning](../../ICML2025/reinforcement_learning/maximum_total_correlation_reinforcement_learning.md)
- [DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning](discover_automated_curricula_for_sparse-reward_reinforcement_learning.md)
- [Finite-Sample Analysis of Policy Evaluation for Robust Average Reward Reinforcement Learning](finite-sample_analysis_of_policy_evaluation_for_robust_average_reward_reinforcem.md)

<!-- RELATED:END -->
