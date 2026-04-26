---
title: >-
  [论文解读] Heterogeneous Data Game: Characterizing the Model Competition Across Multiple Data Sources
description: >-
  [ICML 2025][数据异构性] 本文提出了异构数据博弈（HD-Game）框架，用博弈论分析多个ML模型提供商在异构数据源上的竞争行为，揭示了三种纯策略纳什均衡（PNE）模式——不存在、同质化和异质化——并给出了各类均衡存在的充分/必要条件。
tags:
  - ICML 2025
  - 数据异构性
  - 纳什均衡
  - 博弈论
  - ML市场竞争
  - 选择模型
---

# Heterogeneous Data Game: Characterizing the Model Competition Across Multiple Data Sources

**会议**: ICML 2025  
**arXiv**: [2505.07688](https://arxiv.org/abs/2505.07688)  
**代码**: 无  
**领域**: 强化学习 / 博弈论  
**关键词**: 数据异构性, 纳什均衡, 博弈论, ML市场竞争, 选择模型

## 一句话总结
本文提出了异构数据博弈（HD-Game）框架，用博弈论分析多个ML模型提供商在异构数据源上的竞争行为，揭示了三种纯策略纳什均衡（PNE）模式——不存在、同质化和异质化——并给出了各类均衡存在的充分/必要条件。

## 研究背景与动机

**领域现状**：现实ML市场中，数据通常来自多个异构来源（如不同医院的患者数据），而市场上存在多个ML模型提供商相互竞争。现有工作大多关注单一模型如何在异构数据上表现稳健（如不变学习、分布鲁棒优化等）。

**现有痛点**：已有的ML竞争分析工作（如Ben-Porat & Tennenholtz 2017; Jagadeesan et al. 2023）主要假设单一数据分布，忽略了数据源之间的异构性。然而现实中不同数据源有不同的分布特征，这直接影响模型提供商的策略选择。

**核心矛盾**：数据异构性与市场竞争的交互作用尚不清楚——当多个提供商在异构数据市场中竞争时，均衡策略是什么样的？是所有人训练一个"万能模型"还是各自专注于某些数据源？

**本文目标**：建立异构数据下的多ML提供商竞争的博弈论框架，分析纯策略纳什均衡的存在性和具体形式。

**切入角度**：将ML模型竞争类比为竞争性选址模型（competitive location models），用Mahalanobis距离衡量模型对各数据源的损失，用两种选择模型（proximity和probability）刻画数据源选择行为。

**核心 idea**：用两种数据源选择模型（确定性选最优 vs. logit概率选择）分别分析垄断、双寡头和多提供商场景下的纳什均衡结构，揭示数据异构性、温度参数和提供商数量如何共同决定市场均衡类型。

## 方法详解

### 整体框架
考虑K个数据源（各有权重 $w_k$、真实参数 $\theta_k$ 和协方差矩阵 $\Sigma_k$），N个模型提供商各选择一个模型参数 $\hat{\theta}_n$。模型在数据源k上的损失定义为Mahalanobis距离的平方 $\ell_{n,k} = (\hat{\theta}_n - \theta_k)^\top \Sigma_k (\hat{\theta}_n - \theta_k)$。每个数据源根据选择模型分配给各提供商，提供商的效用是其获得的加权市场份额之和。

### 关键设计

1. **两种数据源选择模型**:

    - 功能：刻画数据源如何在多个ML模型中做出选择
    - 核心思路：Proximity选择模型（PROX）直接选损失最小的模型（确定性）；Probability选择模型（PROP）用logit softmax以温度参数 $t$ 控制噪声选择，$t \to 0$ 退化为PROX，$t \to \infty$ 变为均匀随机
    - 设计动机：两种模型分别对应理性决策和有限理性决策，覆盖了现实中的不同选择行为

2. **均衡策略集合的刻画（Proposition 4.1）**:

    - 功能：限定均衡中每个玩家可能选择的策略空间
    - 核心思路：证明任何PNE中的策略必属于集合 $\vartheta = \{\bar{\theta}(\boldsymbol{q}) : \boldsymbol{q} \in \Delta_K\}$，其中 $\bar{\theta}(\boldsymbol{q})$ 是各数据源真实参数的加权最优。也就是说，每个提供商的最优策略等价于在某个数据源权重分配下最小化加权损失
    - 设计动机：将高维连续策略空间降维到K维单纯形上的搜索，使理论分析可行

3. **三类均衡的完整分类**:

    - 功能：给出不同市场结构下均衡的存在性和形式
    - 核心思路：在PROX下双寡头PNE存在当且仅当 $w_1 \geq 0.5$（有主导数据源），且PNE必然是异质化的（两个提供商都选主导源的真实参数）；在PROP下双寡头PNE必须是同质化的（都选加权最优 $\hat{\theta}^M$），存在的充分必要条件是温度 $t \geq \underline{t}$。多提供商时，PROX产生异质化PNE（提供商按数据源权重比例分配），PROP则在高温下产生同质化PNE、低温下产生异质化PNE
    - 设计动机：全面理解不同市场参数如何影响竞争结果，为监管政策提供理论基础

### 损失函数 / 训练策略
本文为纯理论工作，不涉及模型训练。分析基于Mahalanobis距离定义的损失函数，该损失在线性模型下等价于MSE。

## 实验关键数据

### 主实验
本文通过合成实验验证理论结果。设置K=2数据源、D=2维参数空间，随机生成10个博弈配置。

| 实验设置 | 指标 | 结果 |
|---------|------|------|
| 同质化PNE阈值温度 $\underline{t}$ | $\underline{t}/(2\ell_{max})$ | 约0.1-0.2，远小于理论上界1.0 |
| 异质化PNE最大温度 | t值 | 随N增大趋势下降 |
| 两类PNE共存 | 是否共存 | 部分配置下共存，大N时不太可能 |

### 消融实验

| 配置 | 关键发现 | 说明 |
|------|---------|------|
| N递增 (2→30) | 同质PNE阈值增大后趋稳 | 与 $\underline{t} \leq 2\ell_{max}$ 一致 |
| t递增 | 先出现异质PNE，后出现同质PNE | 验证了温度决定均衡类型 |
| 数据异构度变化 | $\ell_{max}$ 越大，阈值 $\underline{t}$ 越高 | 异构性越强越需要高噪声才有同质均衡 |

### 关键发现
- 在PROX下，双寡头市场只有当最大数据源权重 ≥0.5 时才存在PNE，且提供商必然"扎堆"于主导数据源
- 在PROP下，温度参数 $t$ 是决定均衡类型的关键：低温促进异质化（模型多样性），高温促进同质化（万能模型）
- 提供商数量越多，越容易出现PNE，且异质化PNE中提供商按数据源权重比例分配
- 两类PNE可以在同一博弈中共存（Example 5.2验证了N=8, K=2的情况）

## 亮点与洞察
- **博弈视角的新颖性**：首次将数据异构性引入ML竞争分析，揭示了"万能模型"和"专精模型"两种市场模式的博弈论基础。这个视角可以迁移到联邦学习中分析不同参与方的策略选择
- **政策启示**：理论结果直接指导监管——若要促进模型多样性，应鼓励数据源理性选择（降低"温度"）；若某些小数据源被忽视，可以通过增加提供商数量或激励措施来平衡
- **Mahalanobis距离框架**：同时捕获concept shift和covariate shift，且适用于线性探测（linear probing）场景，使理论结果有更广泛的适用性

## 局限与展望
- 理论分析限于线性模型和Mahalanobis距离，对深度神经网络的适用性未验证
- 假设数据源权重和分布参数已知，现实中这些通常是不确定的
- 只分析了纯策略纳什均衡，混合策略均衡可能在PNE不存在时提供更多洞察
- 合成实验仅用K=2、D=2的小规模设置，缺少真实数据上的验证
- 未考虑动态博弈（如多轮竞争中策略调整）和信息不对称情形

## 相关工作与启发
- **vs 传统ML竞争分析 (Jagadeesan et al. 2023)**：他们假设单一数据分布下的竞争，本文扩展到异构数据源，发现了全新的均衡结构
- **vs 竞争性选址模型 (Hotelling 1929)**：经典模型局限于低维空间和均匀距离度量，本文引入源特定距离度量和高维策略空间，挑战更大但也更贴近ML实际
- **vs 分布鲁棒优化 (Duchi & Namkoong 2021)**：DRO关注单一模型的鲁棒性，本文关注多个竞争模型的均衡行为，两者可以结合——竞争中每个提供商可以用DRO来选择自己的策略

## 评分
- 新颖性: ⭐⭐⭐⭐ 将数据异构性引入ML竞争框架是全新视角，但仅限于线性模型
- 实验充分度: ⭐⭐⭐ 合成实验验证了理论，但缺少真实场景的实证
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨清晰，符号系统统一
- 价值: ⭐⭐⭐⭐ 对理解ML市场竞争和制定监管政策有重要参考意义

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Automatic Reward Shaping from Confounded Offline Data](automatic_reward_shaping_from_confounded_offline_data.md)
- [\[ICML 2025\] The Impact of On-Policy Parallelized Data Collection on Deep Reinforcement Learning Networks](the_impact_of_on-policy_parallelized_data_collection_on_deep_reinforcement_learn.md)
- [\[ICML 2025\] Zero-Shot Generalization of Vision-Based RL Without Data Augmentation](zero-shot_generalization_of_vision-based_rl_without_data_augmentation.md)
- [\[ICML 2025\] Leveraging Skills from Unlabeled Prior Data for Efficient Online Exploration](leveraging_skills_from_unlabeled_prior_data_for_efficient_online_exploration.md)
- [\[NeurIPS 2025\] NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation](../../NeurIPS2025/reinforcement_learning/noisyrollout_reinforcing_visual_reasoning_with_data_augmenta.md)

<!-- RELATED:END -->
