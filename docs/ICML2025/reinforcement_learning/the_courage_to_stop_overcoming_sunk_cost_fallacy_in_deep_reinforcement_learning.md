---
title: >-
  [论文解读] LEAST: The Courage to Stop — Overcoming Sunk Cost Fallacy in Deep RL
description: >-
  [ICML 2025][强化学习][沉没成本谬误] 提出 Learn to Stop（LEAST），一种轻量级自适应 episode 提前终止机制：维护最近 K 个 episode 的 Q 值和梯度幅值缓冲区，用步级中位数构造质量阈值 $\epsilon_i$ 和学习潜力权重 $\omega_i$…
tags:
  - "ICML 2025"
  - "强化学习"
  - "沉没成本谬误"
  - "提前终止"
  - "Q值阈值"
  - "梯度统计"
  - "Replay Buffer"
---

# LEAST: The Courage to Stop — Overcoming Sunk Cost Fallacy in Deep RL

**会议**: ICML 2025  
**arXiv**: [2506.13672](https://arxiv.org/abs/2506.13672)  
**代码**: 无  
**领域**: 强化学习 / 样本效率  
**关键词**: 沉没成本谬误, 提前终止, Q值阈值, 梯度统计, Replay Buffer

## 一句话总结

提出 Learn to Stop（LEAST），一种轻量级自适应 episode 提前终止机制：维护最近 K 个 episode 的 Q 值和梯度幅值缓冲区，用步级中位数构造质量阈值 $\epsilon_i$ 和学习潜力权重 $\omega_i$，当当前 Q 值低于 $\omega_i \times \epsilon_i$ 时终止并重置；在 MuJoCo 四任务上为 TD3/SAC/REDQ 均带来显著提升（归一化分数从 0.65 提升到 0.70+），DMC 视觉 RL 的 Finger Turn Hard 任务收敛速度加快约 30%。

## 研究背景与动机

**领域现状**：Off-policy 深度 RL 依赖 replay buffer 重用历史经验以提升样本效率。然而，传统 RL 框架强制 agent 跑完每个 episode，即使轨迹已陷入次优区域。

**现有痛点**：当 agent 陷入低质量轨迹时，(1) 继续交互浪费环境交互预算；(2) 低质量转移"污染"replay buffer，导致训练数据中无信息量的样本占比过高（实验显示可达 30-40%）；(3) 参数敏感的优先采样方法（如 PER）无法根本解决数据源头的质量问题。

**核心矛盾**：传统 RL 缺乏自主停止机制，agent 无法判断"继续跑下去是否值得"，类似于人类的沉没成本谬误——因为已经投入了步数而不愿放弃已注定低回报的轨迹。

**本文目标** 赋予 agent 基于历史统计的自适应终止能力，在检测到当前轨迹质量低于历史中位数且学习价值不高时主动终止并重置。

**切入角度**：利用 Q 值衡量轨迹质量、梯度幅值衡量学习潜力，两者结合构造双准则停止阈值。

**核心 idea**：教 agent 学会"及时止损"，用已有的 Q 值和梯度信息零成本判断何时放弃当前 episode。

## 方法详解

### 整体框架

LEAST 是一个即插即用模块，可嵌入任何 off-policy RL 算法。在每个 episode 的每一步，LEAST 执行：(1) 计算当前 $(s_i, a_i)$ 的 Q 值 $\hat{Q}_i$ 和梯度幅值 $G_i$；(2) 将其与历史缓冲区的步级中位数比较；(3) 若低于自适应阈值则终止 episode 并重置环境；(4) 通过熵感知机制动态调整缓冲区大小，通过噪声调度帮助逃离次优策略。

### 关键设计

1. **Q值质量阈值（步级中位数）**:

    - 功能：为 episode 内每一步 $i$ 独立构造质量参考线
    - 核心思路：维护二维缓冲区 $\mathcal{B}_Q \in \mathbb{R}^{K \times L}$（K 个最近 episode × L 最大步数），每步的阈值为 $\epsilon_i = \text{Median}(\mathcal{B}_Q[:, i])$。当 $\hat{Q}_i < \epsilon_i$ 时，说明当前轨迹回报预期低于历史中位数，触发终止
    - 设计动机：不同步数的 Q 值尺度和行为需求不同，需要步级独立阈值而非全局统一阈值；使用中位数而非均值避免异常值导致阈值震荡（实验验证中位数比均值更稳定）

2. **梯度学习潜力权重**:

    - 功能：调制 Q 值阈值，平衡"质量"与"学习价值"
    - 核心思路：维护梯度缓冲区 $\mathcal{B}_G$，计算动态权重 $\omega_i = \frac{\text{Median}(\mathcal{B}_G[:, i])}{G_i}$。当 $\omega_i < 1$（当前梯度 > 历史中位数）说明状态新颖、值得探索，降低阈值鼓励继续；当 $\omega_i > 1$ 则收紧阈值。完整停止准则：$\hat{Q}_i < \omega_i \times \epsilon_i$（$\epsilon_i \geq 0$时）或 $\hat{Q}_i < \omega_i^{-1} \times \epsilon_i$（$\epsilon_i < 0$时）
    - 设计动机：仅靠 Q 值停止会错过有价值的探索机会——低 Q 值但高梯度意味着 agent 正在学习新知识，不应终止

3. **熵感知动态缓冲区 + 噪声调度**:

    - 功能：自适应调整统计窗口大小和探索噪声
    - 核心思路：每 $c$ 步计算 $\mathcal{B}_Q$ 的熵 $H_t$，若 $H_t > (1+\gamma) \times \bar{H}$ 则扩展缓冲区以获得更平滑的统计估计。同时，若 agent 频繁在早期被终止（说明策略可能卡在次优），通过 sigmoid 调度增大探索噪声 $\sigma$，帮助逃离当前行为模式
    - 设计动机：不稳定策略导致相邻轨迹差异大，需要更大窗口来稳定中位数估计；频繁早停说明策略需要更多随机性来发现新行为

### 损失函数 / 训练策略

LEAST 不引入额外的训练损失或网络参数。它仅利用 RL 训练中已有的 Q 值和 TD 损失（作为梯度幅值的代理），因此计算开销几乎为零。终止时将已收集的部分轨迹正常存入 replay buffer，然后重置环境开始新 episode。

## 实验关键数据

### 主实验（MuJoCo-v4，5 seeds）

| 算法 | Ant | Walker2d | HalfCheetah | Humanoid | 归一化均分 |
|------|-----|----------|-------------|----------|-----------|
| TD3 | ~4500 | ~4000 | ~11000 | ~5000 | 0.58 |
| TD3+LEAST | **~6800** | **~5200** | **~12000** | **~5800** | **0.70** |
| SAC | ~5200 | ~4500 | ~11500 | ~5200 | 0.63 |
| SAC+LEAST | **~6500** | **~5500** | **~12500** | **~5800** | **0.71** |
| REDQ | ~6000 | ~5000 | ~12000 | ~5500 | 0.68 |
| REDQ+LEAST | **~6800** | **~5500** | **~12500** | **~6000** | **0.73** |

### 视觉 RL（DeepMind Control Suite，DrQv2 骨干）

| 方法 | Finger Turn Hard | Quadruped Run | 归一化均分 |
|------|-----------------|---------------|-----------|
| DrQv2 | ~700 | ~900 | 0.724 |
| CURL | ~720 | ~880 | 0.71 |
| A-LIX | ~740 | ~910 | 0.73 |
| DrQv2+LEAST | ~780 | ~953 | 0.756 |
| TACO | **~800** | **~960** | **0.78** |

LEAST 在 Finger Turn Hard 上收敛速度比 DrQv2 快约 30%，归一化分数接近 TACO。

### 消融实验

| 配置 | 归一化均分 | 说明 |
|------|-----------|------|
| 仅 Q 值阈值（无 $\omega$） | 0.65 | 缺少学习潜力调制，过于激进地终止 |
| Q 值 + $\omega$ 调制 | 0.68 | 双准则明显优于单准则 |
| + 动态缓冲区 + 噪声调度 | **0.70** | 完整 LEAST 最稳定 |
| 均值代替中位数 | 0.62 | 异常值导致阈值不稳定 |

| 超参数 | 最优范围 | 敏感度 |
|--------|---------|--------|
| $\omega$ 权重系数 | [0.3, 0.6] | 中等（SAC 比 TD3 更敏感） |
| 噪声上界 $\bar{\sigma}$ | TD3: [0.25, 0.35], SAC: [0.15, 0.25] | 低 |
| 启动时间 | MuJoCo: 10-20% 训练, DMC: 5-15% | 低 |
| 初始缓冲区大小 | 250 episodes | 中等 |
| 熵溢出率 $\gamma$ | [0, 0.1] | 低 |

### 关键发现

- LEAST 对三类 off-policy 算法（确定策略 TD3、随机策略 SAC、集成 REDQ）均一致有效
- Replay buffer 分析：LEAST 显著减少低 Q 值 + 低 loss 的"白区"样本（无信息转移），增加高 Q 值 + 高 loss 的"黑区"样本（高质量学习信号）
- 样本效率：TD3+LEAST 达到 vanilla TD3 最终性能所需步数平均减少 30-50%
- 在视觉 RL 中无需额外网络即可接近 TACO 等需要表征学习修改的方法
- $\omega \in [0.3, 0.6]$ 在 TD3 和 SAC 上均鲁棒；SAC 因随机策略本身具备多样性，对噪声调度不敏感

## 亮点与洞察

- **"沉没成本谬误"在 RL 中的类比**极具启发性——传统 RL 强制跑完 episode 的设计确实类似于"买了电影票所以看完烂片"的非理性行为
- **零额外计算成本**——仅利用训练中已计算的 Q 值和 TD 损失，不需多余网络或训练
- **中位数 vs 均值的选择**有实践价值——在强化学习这种高方差场景中，中位数对异常轨迹的鲁棒性远优于均值
- **与好奇心驱动互补**——好奇心鼓励探索新状态，LEAST 鼓励放弃已熟悉的低质量轨迹，两者正交可组合

## 局限与展望

- 终止阈值虽不敏感但仍有多个超参数（$\omega$ 系数、缓冲区大小、启动时间等），缺乏自动化选择方法
- 仅在 off-policy 算法上验证，对 on-policy 方法（PPO 等）的适用性未讨论
- 在极稀疏奖励环境中 Q 值信号本身不可靠，LEAST 的终止判断可能失效
- 重置后 agent 可能因策略未变而重新陷入同一次优轨迹——噪声调度仅部分缓解
- LEAST 可能增大分数方差（box plot 显示更大的离散度），训练稳定性有待改善
- Replay buffer 中高 loss 但低 Q 值的"蓝区"样本未被有效处理

## 相关工作与启发

- **vs PER（优先经验回放）**：PER 在采样端过滤低质量转移，LEAST 在产生端就避免生成低质量转移，两者互补
- **vs ICM（好奇心驱动）**：ICM 鼓励探索新状态，LEAST 鼓励放弃旧的坏状态，方向互补
- **vs DroQ / REDQ**：这些方法通过更好的 Q 函数拟合提升效率，LEAST 通过更好的数据收集策略提升效率，正交组合有效
- **vs DrQv2 / TACO**：视觉 RL 方法靠编码器改进提升效率，LEAST 靠数据策略改进达到接近效果，成本更低

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 沉没成本谬误在 RL 中的类比新颖，方法简洁优雅
- 实验充分度: ⭐⭐⭐⭐ MuJoCo 4 任务 × 4 算法 + DMC 4 任务 + 详尽消融，覆盖全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、类比精准、论证层层递进，开头引语增加趣味性
- 价值: ⭐⭐⭐⭐ 即插即用的通用改进，对 off-policy RL 实践有直接贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Adaptive Cooperative Transmission Design for URLLC via Deep RL](../../NeurIPS2025/reinforcement_learning/adaptive_cooperative_transmission_design_for_ultra-reliable_low-latency_communic.md)
- [\[NeurIPS 2025\] Deep RL Needs Deep Behavior Analysis: Exploring Implicit Planning by Model-Free Agents](../../NeurIPS2025/reinforcement_learning/deep_rl_needs_deep_behavior_analysis_exploring_implicit_planning_by_model-free_a.md)
- [\[NeurIPS 2025\] Counteractive RL: Rethinking Core Principles for Efficient and Scalable Deep Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/counteractive_rl_rethinking_core_principles_for_efficient_and_scalable_deep_rein.md)
- [\[ICML 2025\] Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning](network_sparsity_unlocks_the_scaling_potential_of_deep_reinforcement_learning.md)
- [\[ACL 2026\] EvoCoT: Overcoming the Exploration Bottleneck in Reinforcement Learning for LLMs](../../ACL2026/reinforcement_learning/evocot_overcoming_the_exploration_bottleneck_in_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
