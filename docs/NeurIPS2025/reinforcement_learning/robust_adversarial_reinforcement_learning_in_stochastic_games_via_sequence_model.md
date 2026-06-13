---
title: >-
  [论文解读] Robust Adversarial Reinforcement Learning in Stochastic Games via Sequence Modeling
description: >-
  [NeurIPS 2025][强化学习][对抗鲁棒性] 提出CART（Conservative Adversarially Robust Decision Transformer），首个在随机博弈中增强Decision Transformer对抗鲁棒性的方法，通过阶段博弈建模和NashQ值估计解决ARDT在随机状态转移下的过度乐观问题，实现更准确的极小极大值估计和更优的最差情况回报。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "对抗鲁棒性"
  - "Transformer"
  - "随机博弈"
  - "NashQ"
  - "期望分位数回归"
---

# Robust Adversarial Reinforcement Learning in Stochastic Games via Sequence Modeling

**会议**: NeurIPS 2025  
**arXiv**: [2510.11877](https://arxiv.org/abs/2510.11877)  
**代码**: 暂无  
**领域**: 强化学习  
**关键词**: 对抗鲁棒性, Decision Transformer, 随机博弈, NashQ, 期望分位数回归

## 一句话总结

提出CART（Conservative Adversarially Robust Decision Transformer），首个在随机博弈中增强Decision Transformer对抗鲁棒性的方法，通过阶段博弈建模和NashQ值估计解决ARDT在随机状态转移下的过度乐观问题，实现更准确的极小极大值估计和更优的最差情况回报。

## 研究背景与动机

Decision Transformer（DT）将强化学习重新定义为基于序列建模的条件生成问题，通过以目标回报为条件生成动作。在对抗环境中，ARDT通过以极小极大回报（而非return-to-go）为条件来学习最坏情况感知策略。然而，ARDT存在一个根本缺陷：

**ARDT假设状态转移是确定性的**。在随机博弈中，状态转移是概率性的，ARDT的极小极大回报计算：
$$Q_{\text{ARDT}}(s_t, a_t) = \min_{\bar{a}_t} \max_{a_{t+1}} \min_{\bar{a}_{t+1}} \cdots \hat{R}(\tau_{t:H})$$
直接在轨迹上操作，忽略了到达高回报子博弈的概率。这导致ARDT被罕见但高回报的轨迹误导，产生过度乐观的值估计。

**具体例子**：在一个随机博弈中，选择动作 $a_0$ 有90%概率到达低回报状态、10%概率到达高回报状态。ARDT可能被那10%的罕见高回报轨迹误导，高估 $a_0$ 的价值，而忽视更稳健的替代动作。实验表明ARDT在此场景下回报仅5.7，而CART达到8.0。

## 方法详解

### 整体框架

CART的核心思想是在NashQ值计算中显式考虑状态转移的随机性。将每个时间步的主角-对手交互建模为阶段博弈（stage game），其支付函数定义为后续状态的期望最大值，从而整合转移概率。

### 关键设计

1. **阶段博弈建模与支付函数**：在每个阶段，主角选择动作 $a$，对手在观察到 $a$ 后选择 $\bar{a}$。支付函数通过额外的状态值函数 $V$ 考虑转移随机性：
$$\bar{Q}(s, a, \bar{a}) = \mathbb{E}_{s' \sim T(\cdot|s,a)}[r + V(s')]$$
其中 $V(s') = \max_{a'} Q(s', a')$。这意味着支付函数对所有可能的后续状态求期望，而非仅从单条轨迹估计。

2. **NashQ条件值**：作为DT训练的条件变量 $z$，CART使用：
$$Q_{\text{CART}}(s, a) = \min_{\bar{a}} \bar{Q}(s, a, \bar{a})$$
这是阶段博弈的解，同时考虑了对抗鲁棒性和转移随机性。与ARDT的关键区别在于：ARDT直接在轨迹return上做min-max操作，而CART通过显式的 $V$ 函数整合状态转移概率。

3. **Expectile Regression近似Min/Max**：直接在数据中遍历所有动作来计算 $\min_{\bar{a}}$ 或 $\max_a$ 效率低下。CART使用期望分位数回归（ER）联合进行Q-learning和极值近似：

    - 学习支付函数 $\bar{Q}$：最小化TD误差 $\mathcal{L}(\bar{Q}) = \mathbb{E}[\bar{Q}(s,a,\bar{a}) - V(s') - r]$
    - 估计NashQ值：$\mathcal{L}(Q) = \mathbb{E}[L_{\text{ER}}^{\alpha \to 0}(Q(s,a) - \bar{Q}(s,a,\bar{a}))]$（$\alpha \to 0$ 近似 $\min$）
    - 估计最优状态价值：$\mathcal{L}(V) = \mathbb{E}[L_{\text{ER}}^{\alpha \to 1}(V(s') - Q(s',a'))]$（$\alpha \to 1$ 近似 $\max$）

### 损失函数 / 训练策略

训练分两个阶段：
1. **NashQ值估计**：交替优化 $\bar{Q}$、$Q$、$V$ 三个损失函数直到收敛。终端状态值用MSE初始化
2. **DT训练**：使用收敛的 $Q_{\text{CART}}$ 作为条件值 $z$，按标准DT损失训练：
$$\mathcal{L}_{\text{DT}}(\theta) = -\mathbb{E}[\log \pi_\theta(a_t | \tau_{0:t-1}, s_t, z)]$$
推理时设置高目标回报 $z$ 来引导鲁棒策略的生成。

## 实验关键数据

### 主实验

在合成随机博弈中对比最差情况回报（数据由均匀随机策略收集，测试时对抗最优对手）：

| 方法 | 两阶段博弈回报 | 5个变体游戏平均回报 | 过度乐观风险 |
|------|-------------|-------------------|-------------|
| DT | 6.0 | 低 | 高（不考虑对抗） |
| ARDT | 5.7 | 中等 | 高（忽略转移概率） |
| **CART** | **8.0** | **最高** | **低** |

### 消融实验

| 博弈变体 | DT | ARDT | CART | 说明 |
|---------|-----|------|------|------|
| 原始两阶段 | 6.0 | 5.7 | **8.0** | ARDT被10%高回报误导 |
| 罕见回报=100 | 低 | 更差 | **稳定** | 极端罕见高回报放大ARDT的过度乐观 |
| 转移概率20/80 | 中 | 中 | **最优** | 改变转移比例验证鲁棒性 |
| 三阶段博弈 | 低 | 低 | **最优** | 更长的决策链验证可扩展性 |
| 混合变体 | 低 | 低 | **最优** | 综合多种随机性 |

### 关键发现

- CART在所有随机博弈变体中始终获得最高最差情况回报，且方差最低
- ARDT在目标回报增大时鲁棒性反而下降——因为更高的目标回报引导模型追求罕见高回报轨迹
- 引入显式的 $V$ 函数是解决随机转移下对抗鲁棒性的关键
- Expectile Regression有效近似了min/max操作，避免了穷举搜索

## 亮点与洞察

- 精准识别了ARDT在随机博弈中的理论缺陷：忽略转移概率导致值函数过度乐观
- 将Nash Q-Learning的思想优雅地融入Decision Transformer框架，通过轨迹重标注实现
- Expectile Regression作为min/max的可微近似是一个通用技巧，使得Q-learning和极值优化可以联合高效进行

## 局限与展望

- 实验仅在合成短视野随机博弈上进行，未在Poker等更复杂的多智能体竞争环境中验证
- 离线设定限制了数据覆盖——如果行为策略未充分探索关键状态，NashQ估计可能不准确
- Expectile Regression中 $\alpha$ 参数的选择对min/max近似质量有影响，但论文未详细讨论调参策略
- 仅考虑两玩家零和博弈，扩展到多玩家或一般和博弈需要更多理论支撑

## 相关工作与启发

- **Decision Transformer (DT)**：将RL建模为条件序列生成，本文基于此框架
- **ARDT**：首次将对抗鲁棒性引入DT，但限于确定性转移，CART直接扩展了ARDT
- **Nash Q-Learning**：经典的随机博弈求解方法，CART借鉴了其阶段博弈和NashQ值概念
- **IQL (Implicit Q-Learning)**：使用expectile regression进行离线Q-learning，CART的值函数学习直接受其启发
- **启发**：在对抗RL中，显式建模环境随机性比仅依赖最坏情况回报更重要

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次解决DT在随机博弈中的对抗鲁棒性问题
- **实验充分度**: ⭐⭐⭐ 仅合成博弈实验，缺乏大规模或现实场景验证
- **写作质量**: ⭐⭐⭐⭐ 问题定义清晰，动机充分，方法部分数学推导清楚
- **价值**: ⭐⭐⭐⭐ 为DT在多智能体随机环境中的应用提供了理论基础和实用方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] A Near-optimal, Scalable and Parallelizable Framework for Stochastic Bandits Robust to Adversarial Corruptions and Beyond](a_nearoptimal_scalable_and_parallelizable_framework_for_stoc.md)
- [\[NeurIPS 2025\] Emergent World Beliefs: Exploring Transformers in Stochastic Games](emergent_world_beliefs_exploring_transformers_in_stochastic_games.md)
- [\[ICLR 2026\] Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](../../ICLR2026/reinforcement_learning/robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)
- [\[NeurIPS 2025\] Incremental Sequence Classification with Temporal Consistency](incremental_sequence_classification_with_temporal_consistency.md)
- [\[NeurIPS 2025\] Confounding Robust Deep Reinforcement Learning: A Causal Approach](confounding_robust_deep_reinforcement_learning_a_causal_approach.md)

</div>

<!-- RELATED:END -->
