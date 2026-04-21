---
title: >-
  [论文解读] Latent Wasserstein Adversarial Imitation Learning
description: >-
  [ICLR 2026][Wasserstein距离] 提出LWAIL方法，用ICVF从少量随机数据学习动态感知的潜空间表示，将Wasserstein距离的"地面度量"从欧氏距离升级为潜空间距离，仅用单条状态轨迹即可达到专家级模仿性能。
tags:
  - ICLR 2026
  - Wasserstein距离
  - ICVF
  - 动态感知嵌入
  - 状态观测模仿
  - 少样本
---

# Latent Wasserstein Adversarial Imitation Learning

**会议**: ICLR 2026  
**arXiv**: [2603.05440](https://arxiv.org/abs/2603.05440)  
**代码**: [GitHub](https://github.com/JackyYang258/LWAIL)  
**领域**: 模仿学习/强化学习  
**关键词**: Wasserstein距离, ICVF, 动态感知嵌入, 状态观测模仿, 少样本

## 一句话总结
提出LWAIL方法，用ICVF从少量随机数据学习动态感知的潜空间表示，将Wasserstein距离的"地面度量"从欧氏距离升级为潜空间距离，仅用单条状态轨迹即可达到专家级模仿性能。

## 研究背景与动机

**领域现状**：模仿学习从专家演示中学习策略。对抗式模仿学习(AIL)通过匹配智能体与专家的状态分布来学习，f-散度和Wasserstein距离是两种主要分布度量。

**现有痛点**：(1) f-散度需要分布支撑集重叠，当使用低质量非专家数据时难以满足；(2) 基于KR对偶的Wasserstein方法虽然更鲁棒，但1-Lipschitz约束隐含假设了欧氏距离作为地面度量——这无法捕捉环境动态。例如，状态B虽然在欧氏空间离专家状态C更近，但如果B无法到达C，它就不如更远的A有价值。

**核心矛盾**：Wasserstein距离需要一个好的地面度量来衡量状态间距离，但欧氏距离忽略了转移动态——距离近的状态可能在环境中完全不可达。

**切入角度**：用ICVF（意图条件值函数）从少量随机状态数据学到动态感知的嵌入空间 $\phi(s)$，在此空间中欧氏距离自然捕捉可达性关系。

**核心 idea**：用ICVF学到的动态感知潜空间替换Wasserstein AIL中的欧氏空间。

## 方法详解

### 整体框架
两阶段：(1) 预训练：用1%的随机状态数据训练ICVF获得 $\phi(s)$；(2) 在线模仿：在 $\phi$ 空间中做Wasserstein对抗模仿学习。

### 关键设计

1. **ICVF预训练**:

    - 功能：从随机状态转移数据学习状态嵌入 $\phi_\theta(s)$
    - 核心思路：$V_\theta(s, s_+, z) = \phi_\theta(s)^T T_\theta(z) \psi_\theta(s_+)$，用IQL离线RL训练。$\phi(s)$ 编码了状态的可达性结构。
    - 设计动机：Theorem 3.1证明状态对占据概率 $d_{ss}^{\pi_z}(s,s')$ 近似是 $\phi(s)$ 的线性组合，意味着 $\phi$ 空间天然适合做Wasserstein状态匹配。

2. **潜空间Wasserstein AIL**:

    - 功能：在 $\phi$ 空间中匹配智能体和专家的状态对分布
    - 核心思路：$\min_\pi \max_{\|f\|_L \leq 1} (\mathbb{E}_{d_{ss}^\pi}[f(\phi(s), \phi(s'))] - \mathbb{E}_{d_{ss}^E}[f(\phi(s), \phi(s'))])$
    - 设计动机：在 $\phi$ 空间中1-Lipschitz约束对应的欧氏距离现在是动态感知的

3. **奖励设计**:

    - 功能：用判别器输出构造RL奖励 $r(s,s') = \sigma(-f(\phi(s), \phi(s')))$
    - 设计动机：sigmoid归一化到[0,1]稳定下游TD3训练

## 实验关键数据

### 主实验
MuJoCo环境，单条状态轨迹（无动作）：

| 环境 | LWAIL | WDAIL | GAIfO | IQlearn | 专家得分 |
|------|-------|-------|-------|---------|---------|
| Hopper | ~专家 | 低 | 中 | 中 | 113.23 |
| HalfCheetah | ~专家 | 低 | 低 | 中 | 88.42 |
| Walker2D | ~专家 | 低 | 中 | 低 | 106.84 |
| Ant | ~专家 | 低 | 低 | 低 | 116.97 |

### 消融实验

| 配置 | 性能 | 说明 |
|------|------|------|
| LWAIL (完整) | 最优 | ICVF嵌入 + Wasserstein |
| 无ICVF (欧氏距离) | 显著下降 | 验证嵌入的重要性 |
| 不同随机数据量 | 1%即够 | 数据效率极高 |

### 关键发现
- ICVF只需1%的在线数据量的随机转移数据就能学到有效嵌入
- t-SNE可视化清晰显示：ICVF嵌入空间中状态按动态关系组织（奖励高的状态聚在一起），而原始空间不具备此性质
- 在Maze2D上，LWAIL甚至超越了用真实稀疏奖励的TD3——因为ICVF提供了更密集的奖励信号

## 亮点与洞察
- **地面度量的重要性**：点出了KR对偶Wasserstein方法中被广泛忽视的问题——1-Lipschitz约束将度量锁定为欧氏距离。这个insight对整个Wasserstein IL社区有价值。
- **随机数据的惊人价值**：仅用随机策略收集的1%状态转移数据，就能学到足够好的动态感知嵌入。这意味着"垃圾数据"在正确利用后也有大价值。
- **理论与实践的优美结合**：Theorem 3.1为 $\phi$ 空间的Wasserstein匹配提供了理论依据。

## 局限与展望
- ICVF的乘法分解结构 $V = \phi^T T \psi$ 是否限制了表达力？
- 仅在连续控制MuJoCo上验证，高维观测（图像）场景未探索
- 单条专家轨迹的实验设置较极端，5-10条轨迹下的表现未报告
- 需要环境交互收集随机数据，纯离线设置下ICVF的效果待验证

## 相关工作与启发
- **vs WDAIL/IQlearn**: 共享KR对偶框架但忽略了地面度量问题，LWAIL用ICVF直接解决
- **vs SMODICE**: 用f-散度需要分布覆盖假设，LWAIL用Wasserstein更鲁棒
- **vs 基于primal Wasserstein的方法**: primal形式避免了度量问题但引入了其他复杂性

## 评分
- 新颖性: ⭐⭐⭐⭐ 动态感知地面度量的思路简洁有洞察力
- 实验充分度: ⭐⭐⭐⭐ 多环境、多baseline、消融充分
- 写作质量: ⭐⭐⭐⭐ 问题动机阐述清晰，理论与实验结合好
- 价值: ⭐⭐⭐⭐ 对Wasserstein IL方法有直接改进价值

<!-- RELATED:START -->

## 相关论文

- [On Discovering Algorithms for Adversarial Imitation Learning](on_discovering_algorithms_for_adversarial_imitation_learning.md)
- [Model Predictive Adversarial Imitation Learning for Planning from Observation](model_predictive_adversarial_imitation_learning_for_planning_from_observation.md)
- [Near-Optimal Second-Order Guarantees for Model-Based Adversarial Imitation Learning](near-optimal_second-order_guarantees_for_model-based_adversarial_imitation_learn.md)
- [Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment](../../CVPR2026/reinforcement_learning/lifelong_imitation_learning_with_multimodal_latent_replay_and_incremental_adjust.md)
- [Learning to Generate Unit Test via Adversarial Reinforcement Learning](learning_to_generate_unit_test_via_adversarial_reinforcement_learning.md)

<!-- RELATED:END -->
