---
title: >-
  [论文解读] Continual Reinforcement Learning by Planning with Online World Models
description: >-
  [ICML 2025 (Spotlight)][持续强化学习] 提出 FTL Online Agent (OA)，通过在线学习的 Follow-The-Leader 浅层世界模型 + 模型预测控制（MPC）规划来实现持续强化学习，该世界模型从构造上免疫灾难性遗忘，且具有理论遗憾界保证 $\mathcal{O}(\sqrt{K^2 D \log(T)})$，在专门设计的 Continual Bench 上全面超越基于深度世界模型的方法。
tags:
  - ICML 2025 (Spotlight)
  - 持续强化学习
  - 在线世界模型
  - 灾难性遗忘
  - Follow-The-Leader
  - 模型预测控制
---

# Continual Reinforcement Learning by Planning with Online World Models

**会议**: ICML 2025 (Spotlight)  
**arXiv**: [2507.09177](https://arxiv.org/abs/2507.09177)  
**代码**: 无  
**领域**: 强化学习 / 持续学习  
**关键词**: 持续强化学习, 在线世界模型, 灾难性遗忘, Follow-The-Leader, 模型预测控制

## 一句话总结

提出 FTL Online Agent (OA)，通过在线学习的 Follow-The-Leader 浅层世界模型 + 模型预测控制（MPC）规划来实现持续强化学习，该世界模型从构造上免疫灾难性遗忘，且具有理论遗憾界保证 $\mathcal{O}(\sqrt{K^2 D \log(T)})$，在专门设计的 Continual Bench 上全面超越基于深度世界模型的方法。

## 研究背景与动机

**领域现状**：持续强化学习（Continual RL, CRL）是一个自然且重要的设定——智能体需要在序列地呈现的多个任务上不断进化和学习。这与人类的终身学习能力密切相关，也是实际机器人部署的必要能力。

**现有痛点**：CRL 面临的最大障碍是**灾难性遗忘（catastrophic forgetting）**——当学习新任务时，智能体会遗忘之前任务的解法。当前主流方法要么使用深度神经网络作为世界模型（如 DreamerV3），配合各种持续学习技巧（如 EWC、经验回放）来缓解遗忘；要么使用独立的模型/头部来隔离不同任务。这些方法要么效果有限，要么扩展性差。

**核心矛盾**：深度世界模型的参数化本身就是遗忘的根源——用梯度下降更新参数来拟合新任务时，旧任务的信息不可避免地被覆盖。各种"打补丁"式的持续学习技术只是在缓解而非根本解决问题。能否设计一种**从构造上（by construction）就不会遗忘**的世界模型？

**本文目标**：设计一种天然免疫灾难性遗忘的世界模型用于持续 RL。核心要求：（1）学到的世界动力学永不遗忘；（2）能在其上有效规划来解决任意奖励函数指定的任务；（3）增量更新，计算高效。

**切入角度**：使用 Follow-The-Leader（FTL）在线学习算法来维护浅层（非深度）世界模型。FTL 模型的更新是对所有历史数据做全局最优拟合（而非梯度下降的局部更新），因此从数学上保证不会遗忘旧数据。在此模型上用模型预测控制（MPC）来规划执行。

**核心 idea**：用 FTL 在线学习算法构建的浅层世界模型天然免疫遗忘（因为每次更新都拟合全部历史数据），配合 MPC 规划器来解决 CRL 中的序列任务，形成 FTL Online Agent (OA)。

## 方法详解

### 整体框架

```
序列任务流：Task 1 → Task 2 → ... → Task K
                    ↓
      [FTL 在线世界模型] —— 增量更新，拟合全部历史数据
                    ↓
      [MPC 规划器] —— 给定任务奖励函数，搜索最优动作序列
                    ↓
                环境执行
```

- 世界模型只学习**环境动力学**（状态转移），与任务奖励解耦
- 任务切换时，只需改变 MPC 的奖励函数，世界模型不变
- 世界模型增量更新，无需重新训练

### 关键设计

1. **Follow-The-Leader (FTL) 浅层世界模型**:

    - **功能**：在线维护一个浅层模型（如线性模型或核方法），用来预测环境的状态转移 $s_{t+1} = f(s_t, a_t)$
    - **核心思路**：FTL 是在线学习中的经典算法。在每个时步 $t$，模型参数选择为所有历史数据上的最优拟合：
  
    $\theta_t = \arg\min_\theta \sum_{i=1}^{t} \ell(\theta; s_i, a_i, s_{i+1})$
   
   即每次更新都重新对**全部历史数据**做全局最优，而不是像梯度下降那样在新数据上局部更新
    - **设计动机**：这是免疫遗忘的数学保证来源——因为更新目标始终包含所有历史数据，旧数据的信息永远不会被丢弃。对比深度模型的梯度下降更新，后者本质上是在新数据上的局部调整，必然会冲淡旧信息

2. **增量更新的高效实现**:

    - **功能**：虽然 FTL 的定义是"对全部历史重新拟合"，但通过选择合适的模型类（如线性模型），可以实现增量更新
    - **核心思路**：对于线性模型 $\theta_t = A_t^{-1} b_t$，只需维护统计量 $A_t$ 和 $b_t$ 的增量更新：
  
    $A_t = A_{t-1} + \phi(s_t, a_t)\phi(s_t, a_t)^\top, \quad b_t = b_{t-1} + \phi(s_t, a_t) s_{t+1}$
   
   其中 $\phi$ 是特征映射。每次更新只需 $O(D^2)$ 的计算，不需要重新遍历历史数据
    - **设计动机**：使得算法在实践中可行——虽然理论上等价于对全部历史重新拟合，实际上只需 $O(1)$ 的增量计算

3. **模型预测控制（MPC）规划器**:

    - **功能**：给定任务的奖励函数 $r(\cdot)$，在 FTL 世界模型上搜索使期望累积奖励最大的动作序列
    - **核心思路**：在每个决策时步，用世界模型 roll out 多条模拟轨迹，评估不同动作序列的累积奖励，选择最优的第一步动作执行：
  
    $a_t^* = \arg\max_{a_{t:t+H}} \sum_{h=0}^{H} r(\hat{s}_{t+h}, a_{t+h})$
   
   其中 $\hat{s}_{t+h}$ 是世界模型的预测状态，$H$ 是规划视野
    - **设计动机**：MPC 的优势在于它完全基于当前最新的世界模型进行规划，不需要额外训练策略网络。任务切换时只需改变奖励函数 $r$，规划器自动适应——这实现了真正的"零遗忘"任务切换

4. **理论遗憾界**:

    - **功能**：证明 OA 在温和假设下具有次线性遗憾界
    - **核心思路**：在 $K$ 个任务、$T$ 个总时步、特征维度 $D$ 的设定下：
  
    $\text{Regret}(T) = \mathcal{O}(\sqrt{K^2 D \log(T)})$
   
    - **设计动机**：理论保证了 OA 的持续学习性能——遗憾随时间次线性增长，意味着平均每步的性能损失趋近于零

### 损失函数 / 训练策略

- FTL 世界模型使用最小二乘损失拟合状态转移
- MPC 规划器使用 Cross-Entropy Method (CEM) 或 Random Shooting 进行动作搜索
- 整个系统无需梯度下降训练策略网络——世界模型增量更新，规划器在线搜索

## 实验关键数据

### 主实验（Continual Bench）

本文设计了 Continual Bench，一个专门用于评估 CRL 的环境套件。

| 方法 | 前向迁移 | 遗忘率 | 总体 CRL 得分 | 说明 |
|------|---------|-------|--------------|------|
| DreamerV3 (naive) | 中等 | 高 | 较低 | 深度世界模型，无抗遗忘措施 |
| DreamerV3 + EWC | 中等 | 中等 | 中等 | 加弹性权重巩固 |
| DreamerV3 + PackNet | 中等 | 低 | 中等 | 参数隔离 |
| DreamerV3 + 经验回放 | 中等 | 中低 | 中等 | 旧数据回放 |
| **OA (本文)** | **高** | **零** | **最高** | FTL 浅层模型 + MPC |

### 消融实验

| 配置 | 遗忘率 | 新任务学习效率 | 说明 |
|------|-------|--------------|------|
| OA (完整) | 零遗忘 | 高效 | FTL + MPC 完整方案 |
| 替换为深度模型 + FTL | 非零遗忘 | 高效 | 深度模型即使用 FTL 理论也有近似误差 |
| 替换为梯度下降更新 | 显著遗忘 | 高效 | 证明 FTL 更新方式是关键 |
| 固定世界模型（不更新） | 零遗忘 | 差 | 无法学习新动力学 |
| MPC 视野 $H$ 变化 | 不变 | 随 $H$ 变化 | 更长视野在复杂任务上更好 |

### 关键发现

1. **OA 实现了真正的零遗忘**：在所有实验中，OA 切换到新任务后，旧任务的性能完全不下降——这是所有深度世界模型方法（即使配合 EWC、PackNet 等）都无法做到的
2. **浅层模型的竞争力**：尽管 OA 使用的是浅层模型（不如 DreamerV3 等深度模型表达能力强），但由于零遗忘的优势，在整体 CRL 得分上反而更高
3. **增量更新的高效性**：OA 的每步更新时间远低于深度模型的训练时间，计算效率显著更高
4. **FTL 更新 vs 梯度下降**：即使在同一浅层模型上，FTL 更新方式也显著优于梯度下降更新，前者无遗忘而后者有明显遗忘

## 亮点与洞察

- **从构造上解决遗忘**：不是用"打补丁"的方式（EWC、经验回放等）缓解遗忘，而是从根本上设计了不会遗忘的学习算法——这是一种范式上的转变
- **模型与策略的解耦**：世界模型只学动力学（与任务无关），策略由 MPC 在线生成（与世界模型解耦）。任务切换只需换奖励函数，优雅且高效
- **理论与实践的统一**：既有严格的遗憾界理论保证，又在实际 CRL 环境中展现了优势——这在 CRL 领域少见
- **ICML Spotlight**：作为 Spotlight 论文，说明了社区对"从根本上解决遗忘"这一方向的认可

## 局限与展望

1. **表达能力限制**：浅层（线性）世界模型的表达能力有限，在环境动力学高度非线性的场景下可能不够强。如何在保持不遗忘的同时提升模型表达能力是关键挑战
2. **特征工程依赖**：浅层模型需要好的特征映射 $\phi$，这在复杂高维环境中不易获得
3. **MPC 的计算开销**：每步推理都需要在世界模型上做多条轨迹的 roll out 和搜索，实时性可能受限
4. **Continual Bench 的复杂度**：所设计的 CRL 基准虽然专门且合理，但相比 Atari 等标准 RL 基准，环境复杂度仍有差距
5. **与更多 CRL 方法的对比**：如 Progressive Neural Networks、CLEAR 等方法的缺失使得对比不够全面

## 相关工作与启发

- **DreamerV3**：深度世界模型的代表，GLIDER 将其作为主要 baseline
- **EWC / PackNet / 经验回放**：经典的持续学习技术，本文作为对比展示了"打补丁"策略的局限性
- **Follow-The-Leader / Online Learning**：经典在线学习理论，本文将其引入 CRL
- **Model Predictive Control**：经典控制方法，与在线世界模型结合形成"零遗忘"决策
- **启发**：在 CRL 中，"不遗忘"可能比"学得快"更重要——一个永不遗忘的简单模型可能优于一个表达力强但易遗忘的复杂模型

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从构造上解决遗忘是范式性创新，FTL + MPC 的组合在 CRL 中是全新的
- 实验充分度: ⭐⭐⭐⭐ 自建 Continual Bench + 多种 baseline 对比，但环境复杂度可进一步提升
- 写作质量: ⭐⭐⭐⭐ 理论和实验兼顾，Spotlight 论文质量
- 价值: ⭐⭐⭐⭐⭐ 为 CRL 提出了根本性的新思路，具有长远影响

<!-- RELATED:START -->

## 相关论文

- [Reward-free World Models for Online Imitation Learning](reward-free_world_models_for_online_imitation_learning.md)
- [Online Pre-Training for Offline-to-Online Reinforcement Learning](online_pre-training_for_offline-to-online_reinforcement_learning.md)
- [Position: Lifetime Tuning is Incompatible with Continual Reinforcement Learning](position_lifetime_tuning_is_incompatible_with_continual_reinforcement_learning.md)
- [Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn](mitigating_plasticity_loss_in_continual_reinforcement_learning_by_reducing_churn.md)
- [Temporal-Difference Variational Continual Learning](../../NeurIPS2025/reinforcement_learning/temporal-difference_variational_continual_learning.md)

<!-- RELATED:END -->
