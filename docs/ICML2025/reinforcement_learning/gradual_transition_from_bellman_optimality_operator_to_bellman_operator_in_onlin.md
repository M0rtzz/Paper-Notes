---
title: >-
  [论文解读] Gradual Transition from Bellman Optimality Operator to Bellman Operator in Online Reinforcement Learning
description: >-
  [ICML2025][Bellman算子] 提出 Annealed Q-learning (AQ-L)，通过期望分位损失（expectile loss）的参数 τ 从接近1退火至0.5，实现从 Bellman 最优算子到 Bellman 算子的平滑过渡，在连续动作空间中既加速了早期学习又抑制了后期过估计偏差，与 TD3/SAC 结合后在多种运动控制和操控任务上显著优于基线。
tags:
  - ICML2025
  - Bellman算子
  - 过估计偏差
  - Actor-Critic
  - 退火策略
  - 期望分位损失
  - 连续动作空间
---

# Gradual Transition from Bellman Optimality Operator to Bellman Operator in Online Reinforcement Learning

**会议**: ICML2025  
**arXiv**: [2506.05968](https://arxiv.org/abs/2506.05968)  
**作者**: Motoki Omura, Kazuki Ota, Takayuki Osa, Yusuke Mukuta, Tatsuya Harada (东京大学)
**代码**: [GitHub](https://github.com/motokiomura/annealed-q-learning)  
**领域**: reinforcement_learning  
**关键词**: Bellman算子, 过估计偏差, Actor-Critic, 退火策略, 期望分位损失, 连续动作空间

## 一句话总结

提出 Annealed Q-learning (AQ-L)，通过期望分位损失（expectile loss）的参数 τ 从接近1退火至0.5，实现从 Bellman 最优算子到 Bellman 算子的平滑过渡，在连续动作空间中既加速了早期学习又抑制了后期过估计偏差，与 TD3/SAC 结合后在多种运动控制和操控任务上显著优于基线。

## 研究背景与动机

### 问题背景

在线强化学习在离散动作空间（如 Atari 游戏）中，Q-learning 类算法通过 **Bellman 最优算子** 直接估计最优 Q 值，学习效率高。其形式为 T*Q(s,a) = r + γ max_a' Q(s',a')。但在连续动作空间（如机器人控制）中，max_a' 操作对无穷动作集不可计算，因此 Actor-Critic 方法转而使用 **Bellman 算子** T^π Q(s,a) = r + γ E_{a'~π} Q(s',a')，仅建模当前策略的价值函数，依赖策略更新来实现改进。

### 已有工作的不足

- **Actor-Critic 方法（TD3, SAC）**：仅使用 Bellman 算子，策略改进完全依赖 actor 的梯度更新，Q 值改善滞后于策略改善，导致**样本效率低**
- **直接使用 Bellman 最优算子的尝试**（Haarnoja et al., 2017; Garg et al., 2023; Kalashnikov et al., 2018）：在连续空间中面临计算代价大和训练不稳定的问题
- **固定期望分位估计**（如 IQL, XQL）：使用固定 τ 值近似最优值，但超参数敏感性高，τ 过大导致严重过估计，τ 过小则失去最优性加速效果

### 核心动机

作者通过表格型 Actor-Critic 的预备实验揭示了一个关键 trade-off：

**Q-learning 式更新**（Bellman 最优算子）：Q 值直接趋向最优值，学习快，但引入过估计偏差

**SARSA 式更新**（Bellman 算子）：Q 值需等待策略改进后才能提升，学习慢，但偏差小

这一 trade-off 自然引出了"**先快后稳**"的退火思路——训练初期利用最优算子加速探索，后期切换为标准算子消除偏差。

## 方法详解

### 整体框架：Annealed Q-learning (AQ-L)

AQ-L 的核心思想极为简洁：在现有 Actor-Critic 方法（TD3 或 SAC）的 critic 学习中，将标准 L2 损失替换为**期望分位损失（expectile loss）**，并通过退火参数 τ 控制最优性程度。

#### 期望分位损失定义

expectile loss 定义为：L_τ(u) = |τ - 1(u < 0)| · u²

其中 u = Q_θ(s,a) - y 为 TD 误差，y 为目标值。当 τ > 0.5 时，正残差（低估）的权重大于负残差（高估），使得 Q 值估计偏向高分位数，近似 max 操作；当 τ = 0.5 时，退化为标准 L2 损失，即标准 Bellman 算子。

#### 退火策略

参数 τ 从初始值 τ₀（接近 1，如 0.9 或 0.99）线性退火至 0.5：

τ_t = τ₀ + (0.5 - τ₀) · t/T

其中 T 为退火结束的时间步。退火完成后 τ 保持 0.5 不变，critic 学习恢复为标准 Actor-Critic 更新。

### 关键设计

#### 1. 与 TD3 的结合 (AQ-L + TD3)

TD3 使用双 critic 网络 Q_θ₁, Q_θ₂ 和延迟策略更新。AQ-L 的修改仅涉及 critic 损失函数——将 MSE 损失替换为 expectile loss L_τ_t。目标值 y 的计算与原始 TD3 保持一致（使用 clipped double Q-learning）。

#### 2. 与 SAC 的结合 (AQ-L + SAC)

SAC 在最大化回报的同时最大化策略熵。AQ-L 同样仅替换 critic 的损失函数，目标值包含熵正则项：y = r + γ(min Q(s',a') - α log π(a'|s'))。其余训练流程与原始 SAC 完全一致。

#### 3. 退火的双重优势

- **早期（τ 接近 1）**：近似 Bellman 最优算子，Q 值直接趋向最优值，加速学习；过估计偏差在此阶段反而促进探索
- **后期（τ = 0.5）**：恢复标准 Bellman 算子，消除过估计偏差，保证收敛稳定性；此时策略已较好，无需额外偏差驱动改进

#### 4. 实现简洁性

整个方法仅需两行代码修改：(1) 将 critic 的 MSE loss 替换为 expectile loss；(2) 添加 τ 的线性退火调度。无需额外网络、额外数据采样或复杂的训练流程改动。

### 理论分析

#### Bellman 最优算子的加速机制

在表格型设定下，Bellman 最优算子直接对所有动作取 max，Q 值更新不依赖当前策略质量。而 Bellman 算子的 Q 值改善受限于当前策略——只有策略改善后 Q 值才能跟进提升，形成"策略-价值"的延迟耦合。

#### 过估计偏差的来源

当使用函数逼近器（神经网络）建模 Q 值时，Q 值存在随机性（来源于采样噪声和函数逼近误差）。对含噪声的 Q 值取 max 会系统性高估：E[max_a Q(s,a)] ≥ max_a E[Q(s,a)]（Jensen 不等式的推论）。这一现象被称为"过估计偏差"（overestimation bias），是 Q-learning 在函数逼近设定下的核心挑战。

#### 期望分位作为 max 的软近似

对于随机变量 X，当 τ → 1 时，τ-期望分位趋向分布的上确界 ess sup(X)。这为在连续动作空间中近似 max_a' Q(s',a') 提供了可微且可计算的替代方案，同时 τ 提供了从"精确 max"到"均值"的连续插值。这种平滑性使得退火策略成为可能。

## 实验关键数据

### 实验1：预备实验——表格型环境中两种算子的对比

使用 5x5 Grid World 环境，比较 SARSA 式（Bellman 算子）和 Q-learning 式（Bellman 最优算子）的表格型 Actor-Critic。

| 方法 | Q 值收敛速度 | 是否过估计 | 噪声敏感性 | 退火后表现 |
|------|------------|-----------|-----------|-----------|
| SARSA 式（Bellman 算子） | 慢（依赖策略改进） | 否 | 低 | — |
| Q-learning 式（Bellman 最优算子） | 快（直接趋向最优） | 是（显著） | 高 | — |
| 线性插值 + 退火 | 快（早期加速） | 轻微（早期短暂） | 中等 | 收敛到无偏估计 |

预备实验向 Q 值注入高斯噪声模拟函数逼近误差：Q-learning 式更新导致 Q 值收敛至高于最优值的过估计值，而 SARSA 式更新几乎不受影响。退火方法兼具两者优势。

### 实验2：连续控制基准任务性能对比

在 MuJoCo 运动控制和机器人操控任务上，AQ-L 与基线方法的对比：

| 任务 | TD3 | AQ-L + TD3 | SAC | AQ-L + SAC | 关键观察 |
|------|-----|------------|-----|------------|---------|
| HalfCheetah | 基线 | 显著优于 | 基线 | 显著优于 | 样本效率提升明显 |
| Hopper | 基线 | 显著优于 | 基线 | 显著优于 | 早期收敛更快 |
| Walker2d | 基线 | 显著优于 | 基线 | 显著优于 | 最终回报更高 |
| Ant | 基线 | 显著优于 | 基线 | 显著优于 | 训练更稳定 |
| Humanoid | 基线 | 显著优于 | 基线 | 显著优于 | 改进幅度大 |
| 操控任务 | 基线 | 显著优于 | 基线 | 显著优于 | 泛化性好 |

AQ-L 在所有测试任务上均显著优于对应基线方法，尤其在训练早期展现出明显的样本效率优势。

### 实验3：超参数鲁棒性分析

| τ₀ 设定 | 固定 τ（不退火） | AQ-L（退火至 0.5） |
|---------|-----------------|------------------|
| τ₀ = 0.7 | 中等性能 | 稳定高性能 |
| τ₀ = 0.9 | 性能波动大 | 稳定高性能 |
| τ₀ = 0.99 | 严重过估计，性能崩溃 | 稳定高性能 |
| τ₀ = 0.999 | 训练失败 | 仍保持合理性能 |

退火机制显著提升了对 τ₀ 超参数的鲁棒性：在固定 τ 设定下，过大的 τ 导致过估计和训练崩溃；而退火策略下，即使初始 τ₀ 选择激进，最终性能仍然稳健。

## 亮点与洞察

- **直觉简洁而深刻**：将离散动作空间 Q-learning 的成功（Bellman 最优算子的快速收敛）引入连续动作空间，通过退火优雅地解决了过估计问题，两个经典算子的"最佳搭配"令人称道
- **实现极简**：仅需修改 critic 的损失函数（MSE 到 expectile loss）并添加退火调度，无需额外网络或训练技巧，工程友好度极高
- **探索-利用的自然对应**：退火策略恰好与 RL 中"先探索后利用"的范式吻合——早期过估计促进探索，后期消除偏差保障利用
- **通用即插即用模块**：方法可轻松嵌入任何基于 Actor-Critic 的算法（TD3, SAC 等），不改变原始框架结构，迁移成本极低
- **期望分位损失的巧妙复用**：借鉴 IQL 中的 expectile regression 思想，但跳出了离线 RL 的场景限制，在在线 RL 中发掘出新的用途

## 局限与展望

- **退火调度固定为线性**：当前仅探索了线性退火，余弦退火、指数退火等非线性调度可能进一步提升性能，但未被系统探索
- **退火终点时间 T 的选择**：需要预先指定退火结束的时间步，缺乏自适应调整机制（如基于过估计程度自动调节 τ）
- **理论保证有限**：方法动机来自表格型实验的直觉观察，缺少在函数逼近设定下的收敛性或偏差-方差 trade-off 的严格理论分析
- **仅验证了 MuJoCo 环境**：虽然覆盖了多种运动控制和操控任务，但未在视觉输入、稀疏奖励、长期规划等更复杂场景中验证
- **与其他过估计缓解方法的结合**：未深入分析 AQ-L 与 MaxMin Q-learning、Averaged DQN、Ensemble 方法等已有抗过估计技术的互补性
- **计算开销分析缺失**：虽然实现简单，但 expectile loss 相比 MSE 的额外计算开销和对训练时间的影响未被量化讨论

## 相关工作与启发

- **IQL（Kostrikov et al., 2022）**：在离线 RL 中使用 expectile regression 避免 OOD 动作的 Q 值查询，本文将 expectile loss 引入在线 RL 的不同场景
- **XQL（Garg et al., 2023）**：使用指数 loss 近似 Bellman 最优算子，但计算代价更大且训练不稳定；AQ-L 通过退火策略更实用
- **QT-Opt（Kalashnikov et al., 2018）**：在机器人抓取中使用 CEM 近似连续动作 max，计算开销大；expectile loss 提供了更高效的替代
- **OptAC（Ji et al., 2024）**：分析了 actor-critic 中使用最优算子的低效问题，为本文提供了理论动机
- **Clipped Double Q-learning（Fujimoto et al., 2018, TD3）**：通过双 critic 取 min 缓解过估计，AQ-L 在此基础上进一步利用退火机制
- **SQL（Haarnoja et al., 2017）**：早期尝试在连续空间使用 soft Bellman 最优算子，但面临训练不稳定问题

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将两种经典 Bellman 算子的退火过渡作为统一框架，视角新颖但技术组件（expectile loss, 退火）均非全新
- 实验充分度: ⭐⭐⭐⭐ — 覆盖多种 MuJoCo 任务和两种基线框架，超参数鲁棒性分析完善，但缺少更复杂场景的验证
- 写作质量: ⭐⭐⭐⭐⭐ — 预备实验的动机推导清晰流畅，从简单到复杂的叙事结构优秀
- 价值: ⭐⭐⭐⭐ — 实用价值高（即插即用、工程友好），但理论深度较浅，对社区的长期影响需验证

<!-- RELATED:START -->

## 相关论文

- [Gradual Transition from Bellman Optimality Operator to Bellman Operator in Online RL](gradual_transition_from_bellman_optimality_operator_to_bellman_operator_in.md)
- [Wasserstein Policy Optimization](wasserstein_policy_optimization.md)
- [Learning to Trust Bellman Updates: Selective State-Adaptive Regularization for Offline RL](learning_to_trust_bellman_updates_selective_state-adaptive_regularization_for_of.md)
- [Online Pre-Training for Offline-to-Online Reinforcement Learning](online_pre-training_for_offline-to-online_reinforcement_learning.md)
- [Reward-free World Models for Online Imitation Learning](reward-free_world_models_for_online_imitation_learning.md)

<!-- RELATED:END -->
