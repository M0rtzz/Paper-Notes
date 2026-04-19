---
title: >-
  [论文解读] Bootstrap Off-policy with World Model
description: >-
  [NeurIPS 2025][基于模型的强化学习] 提出 BOOM 框架，通过 bootstrap 对齐回路将在线规划器的高质量动作蒸馏到策略网络，使用 likelihood-free 的前向 KL 散度和软 Q 加权机制，有效缓解规划器与策略之间的 actor divergence 问题，在高维连续控制任务上取得 SOTA。
tags:
  - NeurIPS 2025
  - 基于模型的强化学习
  - online planning
  - 离策略学习
  - actor divergence
  - 世界模型
  - MPPI
---

# Bootstrap Off-policy with World Model

**会议**: NeurIPS 2025  
**arXiv**: [2511.00423](https://arxiv.org/abs/2511.00423)  
**代码**: [molumitu/BOOM_MBRL](https://github.com/molumitu/BOOM_MBRL)  
**领域**: reinforcement_learning  
**关键词**: 基于模型的强化学习, online planning, 离策略学习, actor divergence, 世界模型, MPPI

## 一句话总结

提出 BOOM 框架，通过 bootstrap 对齐回路将在线规划器的高质量动作蒸馏到策略网络，使用 likelihood-free 的前向 KL 散度和软 Q 加权机制，有效缓解规划器与策略之间的 actor divergence 问题，在高维连续控制任务上取得 SOTA。

## 研究背景与动机

- **在线规划提升 RL 性能**：基于学习的世界模型进行前瞻搜索（如 MPPI），可以生成比纯策略网络更高质量的动作，广泛应用于 model-based RL
- **actor divergence 问题不可避免**：训练数据由规划器增强策略 β=π+MPPI 收集，但策略网络 π 与行为策略 β 的分布存在本质差异
- **价值函数学习偏差**：Q 函数在行为策略 β 的分布上训练，对 π 实际访问但 β 很少覆盖的区域容易产生过高估计
- **策略更新不可靠**：基于有偏 Q 值的策略梯度更新会误导策略优化方向，导致训练不稳定和性能退化
- **规划器输出不可参数化**：MPPI 等基于采样的规划器产生的动作分布是非参数的，其似然函数不可解析计算，传统 KL 散度方法无法直接使用
- **现有方法局限**：TD-MPC2 忽视 actor divergence；BMPC 纯模仿规划器但缺乏价值引导，历史动作质量参差不齐时效果受限

## 方法详解

### 整体框架

BOOM 由三个紧密耦合的组件组成——策略网络、在线规划器和世界模型，形成一个 bootstrap 回路：策略为规划器提供初始化解，规划器通过模型预测优化细化动作后反过来指导策略对齐。世界模型以 TD-MPC2 风格联合训练 encoder $h$、动力学模型 $f$、奖励预测器 $R$ 和价值函数 $Q$，同时服务于规划器的轨迹模拟和策略的价值估计。

### 关键设计一：Likelihood-Free 对齐损失

- **功能**：让策略 π 对齐规划器 β 的动作分布，将规划器的高质量行为蒸馏到策略网络中
- **核心思路**：采用前向 KL 散度 $\text{KL}(\beta \| \pi)$ 而非反向 KL，展开后第一项关于 β 的熵与 π 无关可丢弃，最终损失 $\mathcal{L}_{\text{align}} = -\mathbb{E}[\log \pi(a|s)]$ 只需要计算 π 的对数似然，完全不需要 β 的似然
- **设计动机**：规划器 MPPI 经过加权重采样后的动作分布是非参数的、似然不可计算的，传统反向 KL 需要 $\beta(a|s)$ 不可行；前向 KL 天然规避了这一问题，提供了简洁有效的对齐机制

### 关键设计二：软 Q 加权机制

- **功能**：在对齐损失中引入基于 Q 值的软权重，优先对齐高回报动作
- **核心思路**：对 replay buffer 中的每个转移计算 $w_i = \exp(Q_i/\tau) / \sum_j \exp(Q_j/\tau)$，加权对齐损失变为 $\sum_i w_i [-\log \pi(a_i|s_i)]$，使策略更关注高价值经验
- **设计动机**：replay buffer 中存储了不同时间步规划器产生的历史动作，质量参差不齐（早期规划器较弱）；Q 值加权类似规划器本身的动作选择原则（按价值加权选动作），确保策略从最有价值的经验中学习

### 关键设计三：Bootstrap 策略目标

- **功能**：将对齐损失与标准 Q 最大化损失结合为统一的策略优化目标
- **核心思路**：$\mathcal{L}_{\text{policy}} = -\sum[Q(s, \pi(s)) + \lambda_{\text{align}} \cdot \mathcal{L}_{\text{align}}]$，其中 $\lambda_{\text{align}} = \dim(\mathcal{A})/1000$（DMC）或 $\dim(\mathcal{A})/50$（Humanoid-Bench）
- **设计动机**：纯 Q 最大化受 actor divergence 导致的值偏差影响；纯模仿规划器则丧失了值函数驱动的优化能力；两者结合兼得 Q 值引导和行为对齐的优势

## 损失函数与训练

- **世界模型损失**：联合训练 encoder、dynamics、reward、Q-value，$\mathcal{L}_{\text{model}} = \sum \gamma^t [\|f(z_t, a_t) - \text{sg}(h(s'_t))\|_2^2 + \text{CE}(R_t, r_t) + \text{CE}(Q_t, q_t)]$
- **策略损失**：$\mathcal{L}_{\text{policy}} = -\sum [Q(s, \pi(s)) + \lambda_{\text{align}} \cdot \sum_i w_i (-\log \pi(a_i|s_i))]$
- **训练流程**：先用随机动作 warmup 预训练世界模型，然后迭代：规划器收集数据 → 训练世界模型 → bootstrap 策略更新

## 实验

### 表1：DMC Suite 高维控制任务（总平均回报 TAR，3 seeds）

| 任务 | SAC | DreamerV3 (10M) | TD-MPC2 | BMPC | **BOOM** |
|------|-----|-----------------|---------|------|----------|
| Humanoid-stand | 9.0 | 717.0 | 913.3 | 947.9 | **962.1** |
| Humanoid-walk | 173.8 | 755.6 | 884.8 | 935.1 | **936.1** |
| Humanoid-run | 1.6 | 353.5 | 316.2 | 531.2 | **582.8** |
| Dog-stand | 197.6 | 35.4 | 936.4 | 971.3 | **986.8** |
| Dog-walk | 24.7 | 9.1 | 885.0 | 942.9 | **965.4** |
| Dog-trot | 67.1 | 8.4 | 884.4 | 911.3 | **947.9** |
| Dog-run | 16.5 | 4.3 | 427.0 | 673.7 | **820.7** |
| **平均** | 58.8 | 269.0 | 745.6 | 835.8 | **877.7** |

BOOM 在所有 7 个高维 DMC 任务上均取得最优，平均 TAR 877.7 较 BMPC 提升 +5.0%，较 TD-MPC2 提升 +17.7%。Dog-run 上领先次优 +21.8%。

### 表2：Humanoid Bench 任务（总平均回报 TAR，3 seeds）

| 任务 | SAC | DreamerV3 (10M) | TD-MPC2 | BMPC | **BOOM** |
|------|-----|-----------------|---------|------|----------|
| H1hand-stand | 74.1 | 845.4 | 728.7 | 780.0 | **926.1** |
| H1hand-walk | 27.0 | 744.0 | 644.2 | 672.6 | **935.4** |
| H1hand-run | 14.1 | 622.4 | 66.1 | 236.0 | **682.2** |
| H1hand-sit | 268.4 | 699.1 | 693.7 | 688.2 | **918.1** |
| H1hand-slide | 19.0 | 367.6 | 141.3 | 440.1 | **926.1** |
| H1hand-pole | 122.5 | 577.4 | 207.5 | 739.9 | **930.5** |
| H1hand-hurdle | 12.9 | 135.7 | 59.0 | 197.1 | **435.6** |
| **平均** | 68.5 | 555.6 | 338.8 | 511.7 | **820.6** |

BOOM 在 H-Bench 全部 7 个任务上均为最优，平均 TAR 820.6 较 DreamerV3 (10M) 提升 +47.7%，较 BMPC 提升 +60.5%。H1hand-slide 上提升 +110.5%，H1hand-hurdle 上提升 +121.0%。

### 消融实验

- **对齐度量**：前向 KL（likelihood-free）优于反向 KL（需近似 β 的似然），验证了避免似然估计的必要性
- **Q 加权机制**：去掉 Q 加权改用均匀加权后性能和收敛速度均下降，说明价值引导对处理历史动作质量差异至关重要
- **对齐系数 $\lambda_{\text{align}}$**：在 0.1× 到 10× 默认值范围内性能稳定，对超参鲁棒

## 亮点

- 清晰定义了 planning + off-policy RL 中 actor divergence 的两个后果（值偏差和策略误导），问题分析深入
- Likelihood-free 的前向 KL 对齐避免了规划器非参数分布似然不可计算的难题，方法简洁优雅
- 提供了理论保证：bootstrap 对齐控制 return gap（Theorem 1）和 Q-value gap（Theorem 2）
- 在 14 个高维任务上全面 SOTA，部分任务提升超过 100%，说服力强
- 实现简单，在 TD-MPC2 基础上只需增加一个对齐损失项

## 局限性

- 仍依赖 MPPI 规划器的质量，如果 world model 学习不好，规划器产生的对齐目标本身就有问题
- 前向 KL（mode-covering）倾向于让策略过度覆盖规划器分布，可能导致策略过于保守
- 对齐系数 $\lambda_{\text{align}}$ 在 DMC 和 H-Bench 上使用不同公式，需要根据任务特性调整
- 仅在连续控制 locomotion 任务上验证，缺乏操作（manipulation）、导航等其他类型任务的评估
- 缺少与最新 imagination-driven 方法（如 DreamerV3 的改进版）以及 offline RL 方法的对比

## 相关工作

- **Planning-driven MBRL**：TD-MPC/TD-MPC2 联合学习世界模型和策略，用 MPPI 规划收集数据；BMPC 在此基础上加入规划器动作的模仿和重标注
- **Imagination-driven MBRL**：DreamerV3 通过世界模型内部 rollout 训练策略，不依赖在线规划但样本效率受限
- **Off-policy RL**：SAC 等方法面临从 replay buffer 学习时的分布偏移问题，与本文讨论的 actor divergence 本质相关
- **行为克隆与策略蒸馏**：本文的对齐损失可视为从规划器到策略的在线蒸馏，与 offline RL 中的 behavior cloning 正则化思路类似

## 评分

- 新颖性: ⭐⭐⭐⭐ — 问题定义清晰，likelihood-free 对齐方案简洁有效，有理论支撑
- 实验充分度: ⭐⭐⭐⭐ — 14 个高维任务全面 SOTA，消融完整，但任务类型偏单一
- 写作质量: ⭐⭐⭐⭐ — 问题动机推导流畅，理论分析与实验互相印证
- 价值: ⭐⭐⭐⭐ — 为 planning + off-policy RL 的 actor divergence 问题提供了实用解决方案

<!-- RELATED:START -->

## 相关论文

- [Bootstrap Off-policy with World Model (BOOM)](bootstrap_off-policy_with_world_model.md)
- [Foundation Models as World Models: A Foundational Study in Text-Based GridWorlds](foundation_models_as_world_models_a_foundational_study_in_text-based_gridworlds.md)
- [Learning Interactive World Model for Object-Centric Reinforcement Learning](learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models](communicating_plans_not_percepts_scalable_multi-agent_coordination_with_embodied.md)
- [The Physical Basis of Prediction: World Model Formation in Neural Organoids via an LLM-Generated Curriculum](the_physical_basis_of_prediction_world_model_formation_in_neural_organoids_via_a.md)

<!-- RELATED:END -->
