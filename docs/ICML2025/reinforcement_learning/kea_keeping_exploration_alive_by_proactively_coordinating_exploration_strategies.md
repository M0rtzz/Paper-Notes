---
title: >-
  [论文解读] KEA: Keeping Exploration Alive by Proactively Coordinating Exploration Strategies
description: >-
  [ICML2025][强化学习][探索策略协调] 提出 KEA 方法，通过引入标准智能体与新颖性增强智能体的动态切换机制，主动协调不同探索策略，解决 SAC 与新颖性探索结合时因策略交互导致的冗余采样和低效探索问题。 核心问题：SAC 在稀疏奖励环境中结合新颖性探索方法（如 RND）时，两种探索策略之间存在复杂交互…
tags:
  - "ICML2025"
  - "强化学习"
  - "探索策略协调"
  - "稀疏奖励"
  - "Soft Actor-Critic"
  - "新颖性探索"
  - "RND"
  - "NovelD"
---

# KEA: Keeping Exploration Alive by Proactively Coordinating Exploration Strategies

**会议**: ICML2025  
**arXiv**: [2503.18234](https://arxiv.org/abs/2503.18234)  
**代码**: 待确认  
**领域**: 强化学习  
**关键词**: 探索策略协调, 稀疏奖励, Soft Actor-Critic, 新颖性探索, RND, NovelD

## 一句话总结

提出 KEA 方法，通过引入标准智能体与新颖性增强智能体的动态切换机制，主动协调不同探索策略，解决 SAC 与新颖性探索结合时因策略交互导致的冗余采样和低效探索问题。

## 研究背景与动机

**核心问题：** SAC 在稀疏奖励环境中结合新颖性探索方法（如 RND）时，两种探索策略之间存在复杂交互，导致探索效率低下。

具体来说，这种交互呈现循环模式：

**阶段 T1**：高内在奖励驱动智能体反复访问特定区域，动作分布集中在少数方向

**阶段 T2**：该区域内在奖励衰减后，策略熵增大，引入更多随机性，增加探索新方向的概率

**阶段 T3**：发现未访问区域后，智能体又被高内在奖励吸引，重新聚焦于新发现区域

这种"访问→衰减→随机→发现→再访问"的循环依赖自然的策略转换，导致：

- 大量冗余经验收集（重复访问已知高新颖性区域）
- 发现新状态的延迟
- 整体学习效率下降

**现有方法的不足：** 奖励塑形可能导致行为偏离真实目标；好奇心/新颖性方法（ICM、RND、NovelD）虽然鼓励探索，但缺乏对不同探索策略之间交互的显式协调机制。

## 方法详解

### 整体架构

KEA 集成两个智能体并通过切换机制协调：

- **新颖性增强智能体 $\mathcal{A}^{\text{N}}$**：使用 SAC + 内在奖励（RND/NovelD），负责在已探索区域内寻找相对新颖的状态
- **标准智能体 $\mathcal{A}^{\text{S}}$**：纯 SAC（仅外在奖励），保持高随机性用于探索未知区域
- **切换机制 $\psi$**：基于状态新颖性动态选择使用哪个智能体的策略

### 新颖性探索策略

修改 SAC 的 Soft Bellman 更新目标，同时考虑外在和内在奖励：

$$y_Q = (\beta^{\text{ext}} r^{\text{ext}} + \beta^{\text{int}} r^{\text{int}}) + \gamma \left( \min_{\theta'_{1,2}} Q_{\theta'_i}(s', a') - \alpha \log \pi^{\text{SAC}}(\cdot|s') \right)$$

其中内在奖励由 RND 计算：

$$r_t^{\text{int}} = \| \hat{f}(s_t; \theta) - f(s_t) \|^2$$

$f$ 为随机初始化的目标网络，$\hat{f}$ 为通过梯度下降训练的预测器网络。

### 标准智能体的设计

$\mathcal{A}^{\text{S}}$ 的关键设计：**延迟学习** — 在获得外在奖励之前，将损失权重设为零，冻结梯度更新。这使得策略保持高动作方差（接近初始随机策略），从而在切换到该智能体时能提供互补的随机探索。获得外在奖励后恢复正常训练。

两个智能体共享统一的 replay buffer，利用 off-policy 特性收集多样化数据。

### 切换机制

基于当前状态的内在奖励决定切换：

$$\pi(s_t) = \psi(r_t^{\text{int}}, \pi^{\text{N}}(s_t), \pi^{\text{S}}(s_t))$$

$$\psi = \begin{cases} \pi^{\text{S}}(s_t), & \text{if } r_t^{\text{int}} > \sigma \\ \pi^{\text{N}}(s_t), & \text{otherwise} \end{cases}$$

- **内在奖励 > 阈值 $\sigma$**：状态高度新颖（接近边界），切换到 $\mathcal{A}^{\text{S}}$ 进行随机探索，增加进入未知区域的概率
- **内在奖励 ≤ 阈值 $\sigma$**：状态在已知区域，切换到 $\mathcal{A}^{\text{N}}$ 利用新颖性引导走向更新颖的区域

## 实验关键数据

### 2D 导航任务（稀疏奖励，300K 步）

| 方法 | 平均回报 |
|------|---------|
| SAC | 0.0 ± 0.0 |
| RND-SAC | 0.235 ± 0.184 |
| **KEA-RND-SAC** | **0.403 ± 0.042** |
| NovelD-SAC | 0.607 ± 0.042 |
| **KEA-NovelD-SAC** | 0.604 ± 0.051（收敛快 24%） |

KEA-RND-SAC 相比 RND-SAC 提升 **70%+**，且方差显著降低。

### DeepSea 硬探索基准（100K episode）

| 方法 | N=10 | N=20 | N=24 | N=30 |
|------|------|------|------|------|
| SAC | 0.98 | 0.00 | 0.00 | 0.00 |
| RND-SAC | 0.99 | 0.89 | 0.67 | 0.35 |
| SOFE-DQN | 0.97 | 0.70 | 0.65 | 0.42 |
| **KEA-RND-SAC** | **0.99** | **0.92** | **0.81** | **0.54** |

在高难度 N=30 时，KEA 达到 0.54，超越所有基线。

### DeepMind Control Suite 稀疏奖励（500K 步）

| 方法 | Walker Run | Cheetah Run | Reacher Hard |
|------|-----------|-------------|--------------|
| SAC | 0.0 | 0.0 | 715.17 |
| RND-SAC | 287.65 | 512.02 | 790.32 |
| **KEA-RND-SAC** | **629.74** | **773.76** | **874.61** |
| NovelD-SAC | 553.26 | 647.29 | 860.40 |
| **KEA-NovelD-SAC** | **706.47** | **734.67** | 837.12 |

KEA-RND-SAC 在 Walker Run 上提升 **119%**，Cheetah Run 提升 **51%**。

### 切换阈值敏感性分析

| 阈值 $\sigma$ | 回报 | $\mathcal{A}^{\text{S}}$ 使用率 |
|---------|------|------|
| 0.50 | 0.358 | 24% |
| 1.00 | **0.407** | 14% |
| 1.50 | 0.334 | 8% |

所有阈值配置均优于基线 RND-SAC（0.235），方法对超参数不敏感。

### Off-Policy 方法泛化

- **SAC/SQL**（随机策略与新颖性探索耦合）→ KEA 提升显著
- **DQN-P**（Q值比例采样）→ KEA 有改善
- **DQN**（ε-greedy 独立于 Q 值）→ KEA 无明显效果（探索策略不交互）

## 亮点与洞察

1. **问题分析深入**：清晰刻画了"新颖性探索 ↔ 随机性探索"之间的循环交互问题，用 2D 可视化直观展示
2. **设计简洁有效**：仅需额外一个标准智能体 + 简单阈值切换，无需复杂的元学习或层次结构
3. **即插即用**：可与任意新颖性方法（RND、NovelD 等）结合，兼容多种 off-policy 算法
4. **共享 buffer 提效**：两个智能体共享 replay buffer，充分利用 off-policy 特性，不浪费数据
5. **泛化性分析透彻**：明确指出 KEA 的适用条件——基础探索策略需与新颖性探索产生耦合

## 局限与展望

1. **仅适用于 off-policy 方法**：标准智能体需要共享经验，无法直接扩展到 on-policy（PPO 等）
2. **阈值 $\sigma$ 为固定超参数**：虽然对阈值不敏感，但自适应阈值（如基于内在奖励分布的分位数）可能进一步提升性能
3. **仅验证了 RND/NovelD 两种内在奖励**：是否适用于 ICM 等基于预测误差的方法未验证
4. **标准智能体的"延迟学习"设计**：在外在奖励极其稀疏时，$\mathcal{A}^{\text{S}}$ 可能长期保持纯随机，影响后期利用效率
5. **实验规模有限**：未在高维观察空间（如像素输入）或更复杂环境（如 MuJoCo 复杂机器人操作）中验证
6. **计算开销**：维护两个独立智能体增加了参数量和训练时间，未充分讨论效率权衡

## 相关工作与启发

- **RND** (Burda et al., 2018)：基于随机网络蒸馏的内在奖励，KEA 的核心组件
- **NovelD** (Zhang et al., 2021)：结合新颖性差异与计数奖励，KEA 兼容的另一探索方法
- **NGU** (Badia et al., 2020)：结合 episodic 记忆与终身新颖性的探索策略
- **DeRL** (Schäfer et al., 2021)：解耦表示用于探索的方法，实验基线
- **SOFE** (Castanyer et al., 2024)：另一探索增强方法，在 DeepSea 上的基线

## 评分

- 新颖性: ⭐⭐⭐ — 问题定义清晰但方法核心（双智能体+阈值切换）相对直观
- 实验充分度: ⭐⭐⭐⭐ — 多环境多基线，含泛化分析和敏感性分析，但缺少高维场景
- 写作质量: ⭐⭐⭐⭐ — 问题可视化好，行文清晰，结构完整
- 价值: ⭐⭐⭐ — 解决了SAC+新颖性探索的实际痛点，但适用范围限于off-policy方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Adaptively Coordinating with Novel Partners via Learned Latent Strategies](../../NeurIPS2025/reinforcement_learning/adaptively_coordinating_with_novel_partners_via_learned_latent_strategies.md)
- [\[ICML 2025\] Leveraging Skills from Unlabeled Prior Data for Efficient Online Exploration](leveraging_skills_from_unlabeled_prior_data_for_efficient_online_exploration.md)
- [\[ICML 2025\] Controlling Underestimation Bias in Constrained Reinforcement Learning for Safe Exploration](controlling_underestimation_bias_in_constrained_reinforcement_learning_for_safe_.md)
- [\[NeurIPS 2025\] Exploration via Feature Perturbation in Contextual Bandits](../../NeurIPS2025/reinforcement_learning/exploration_via_feature_perturbation_in_contextual_bandits.md)
- [\[ICML 2025\] Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration](enhancing_cooperative_multi-agent_reinforcement_learning_with_state_modelling_an.md)

</div>

<!-- RELATED:END -->
