---
title: >-
  [论文解读] DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning
description: >-
  [NeurIPS 2025][强化学习][sparse reward] 提出 DISCOVER，一种面向稀疏奖励长视野 RL 的目标选择策略，通过同时平衡可达性（achievability）、新颖性（novelty）和相关性（relevance）来生成指向目标任务的课程，理论上证明达到目标的步数与目标距离线性相关（而非搜索空间体积），在高维导航和操作任务中显著超越先前 SOTA 探索策略。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "sparse reward"
  - "goal selection"
  - "exploration-exploitation"
  - "UCB"
  - "curriculum learning"
  - "goal-conditioned RL"
---

# DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2505.19850](https://arxiv.org/abs/2505.19850)  
**代码**: [已开源](https://github.com/LeanderDiazBone/discover)  
**领域**: 强化学习  
**关键词**: sparse reward, goal selection, exploration-exploitation, UCB, curriculum learning, goal-conditioned RL

## 一句话总结
提出 DISCOVER，一种面向稀疏奖励长视野 RL 的目标选择策略，通过同时平衡可达性（achievability）、新颖性（novelty）和相关性（relevance）来生成指向目标任务的课程，理论上证明达到目标的步数与目标距离线性相关（而非搜索空间体积），在高维导航和操作任务中显著超越先前 SOTA 探索策略。

## 研究背景与动机

**稀疏奖励是 RL 的核心难题**：agent 只在任务完成时获得奖励，需要深度探索（deep exploration）——这在长视野高维空间中几乎不可能靠随机探索实现。

**Goal-conditioned RL 的探索困境**：
   - 可以通过 HER 重标记获得样外目标的学习信号
   - 但目标选择（goal selection）面临自身的探索-利用权衡：选太难的目标学不到东西，选太简单的没有新信息，选不相关的浪费样本

**已有方法的偏向**：
   - **HER**：总是选最终目标——太难，学不到
   - **MEGA**：选已达到的最稀有目标——均匀探索所有方向，在高维空间指数增长
   - **DISCERN (uniform)**：随机选已达到目标——同样无方向性
   - 这些方法都缺少**相关性**维度——不考虑中间目标是否有助于最终目标

## 方法详解

### 核心思想：Goal Utility = Achievability + Novelty + Relevance

DISCOVER 在每个 episode 从已达到目标集 $\mathcal{G}_{\text{ach}}$ 中选择目标 $g_t$：

$$g_t = \arg\max_{g \in \mathcal{G}_{\text{ach}}} \underbrace{V(s_0, g) + \alpha_t \sigma(s_0, g)}_{\text{Achievability + Novelty}} + \underbrace{\beta_t [V(g, g^\star) + \alpha_t \sigma(g, g^\star)]}_{\text{Relevance}}$$

**三原则解读**：
- $V(s_0, g)$ 高 → 策略能到达 $g$（**可达性**）
- $\sigma(s_0, g)$ 高 → 对 $g$ 的可达性不确定（**新颖性**，鼓励探索未知区域）
- $V(g, g^\star)$ 高 → $g$ 离最终目标 $g^\star$ 近（**相关性**，提供方向）
- $\sigma(g, g^\star)$ 高 → 对 $g \to g^\star$ 的价值不确定（**相关性的探索**，防止过早收敛到错误方向）

### 不确定性估计
使用 **critic ensemble**（多个 Q 网络）估计 $V$ 的均值和方差 $\sigma^2$，完全从 agent 自身经验 bootstrap，不需要任何先验信息。

### 自适应参数调整
自动调节 $\alpha_t$ 以维持目标达成率 $p^\star \approx 50\%$：

$$\alpha_{t+1} = \Pi_{[0,1]}(\alpha_t + \eta(p_t - p^\star))$$

$p_t$ 是近 $k_{\text{adapt}}$ 个 episode 的目标达成率。太低 → 降低 $\alpha$（选更容易的）；太高 → 增加 $\alpha$（选更新颖的）。

### 理论保证（简化设定）

在线性特征空间 + 高斯噪声假设下，证明达到目标所需的 episode 数：

$$N \leq \tilde{O}\left(\frac{Dd^2}{\kappa^3}\right)$$

- $D = -V^\star(s_0, g^\star)$：起点到目标的最优距离
- $d$：特征空间维度
- $\kappa$：可达集扩展速率

**关键意义**：bound 只依赖距离 $D$（一维），不依赖目标空间 $\mathcal{G}$ 的体积。先前方法（如 MEGA）的 bound 依赖 $|\mathcal{G}|$，在高维空间指数爆炸。DISCOVER 通过方向性避免了维度诅咒。

### 算法流程

每个 episode 执行：
1. **SelectGoal**：从 $\mathcal{G}_{\text{ach}}$ 按 DISCOVER 目标函数选择 $g_t$
2. **Rollout**：用 $\pi(g_t)$ 执行至 $g_t$ 达成
3. 达成后进入随机探索直到 episode 结束
4. 更新 replay buffer
5. 用 off-policy RL (TD3) + HER relabeling 更新参数

## 实验关键数据

### 高维 Pointmaze（达到 10% 成功率所需步数，M steps）

| 维度 | HER | MEGA | Ach.+Nov. | DISCOVER |
|---|---|---|---|---|
| 2 | ∞ | 4.8M | 5.2M | **2.9M** |
| 3 | ∞ | ∞ | ∞ | **3.1M** |
| 4 | ∞ | ∞ | ∞ | **7.4M** |
| 5 | ∞ | ∞ | ∞ | **5.4M** |
| 6 | ∞ | ∞ | ∞ | **18.7M** |

∞ 表示 50M 步内未达到。3 维以上只有 DISCOVER 能成功——因为无方向的探索在高维空间中指数爆炸，而 DISCOVER 沿目标方向"隧穿"。

### Antmaze + Arm 环境（成功率，取训练后期峰值）

| 环境 / 方法 | HER | MEGA | DISCERN | Ach.+Nov. | **DISCOVER** |
|---|---|---|---|---|---|
| Antmaze (simple) | ~40% | ~80% | ~70% | ~75% | **~90%** |
| Antmaze (hard) | 0% | ~20% | ~15% | ~25% | **~60%** |
| Arm (simple) | ~50% | ~60% | ~55% | ~55% | **~85%** |
| Arm (hard) | 0% | 0% | 0% | 0% | **~30%** |

在困难配置下优势最明显——其他方法要么性能很低（antmaze hard），要么完全不收敛（arm hard）。

### 消融实验关键结论
- **去掉 Relevance**（= Ach.+Nov.）：高维 pointmaze 全部失败，antmaze hard 从 60% 降到 25%
- **去掉 Novelty**（只用 Ach.+Rel.）：容易过早收敛到错误方向
- **去掉 Achievability**（只用 Nov.+Rel.）：选择不可达目标，学不到有效经验
- 三个组件缺一不可

### 利用先验知识
- 手设距离先验（如 L2 距离）或预训练 critic 可边际加速探索
- 但 DISCOVER 的 bootstrap 版本已经足够强，先验的收益有限

### 标准 RL 方法完全失败
TD3 + curiosity、TD3 + RND、TD3 + count-based 等非 goal-conditioned 方法在 simple 环境上都无法达到目标——深度探索需要子目标分解。

## 亮点与洞察

- **"方向感"是关键**：DISCOVER 的核心 insight——光有可达性和新颖性不够，还需要知道"往哪走"。这种方向感完全从 critic 的 bootstrap 估计中获得，无需任何 oracle
- **探索的维度诅咒解决方案**：undirected 探索的复杂度是 $(R/\epsilon)^m$（维度指数），DISCOVER 将其压缩为 $O(D)$（一维距离）。这在理论和实验中都得到验证
- **UCB 的新应用**：将多臂赌博机的 UCB 思想推广到目标选择——每个可达目标是一个"臂"，reward 是该目标对达到最终任务的贡献
- **自适应难度控制**：通过维持约 50% 的目标达成率自动平衡 exploration-exploitation，简单有效

## 局限与展望
- 依赖 critic ensemble 做不确定性估计，对大型 critic 网络增加计算开销
- 理论保证基于线性特征假设，与深度网络的 bootstrapping 非平稳性有差距
- 目标只能从已达到的目标中**选择**，无法**生成**新目标——限制了在连续空间中的精细控制
- 当前聚焦单一目标任务，扩展到多目标分布需要进一步设计
- 仅用 TD3 作为 backbone，未验证与 model-based 方法（如 Dreamer）的组合

## 相关工作与启发
- **vs MEGA (Pitis et al.)**：MEGA 选最稀有的可达目标（= 最大化目标分布熵），但完全不考虑方向。DISCOVER 添加 relevance 项使其在高维任务中从失败变为成功
- **vs HER**：HER 只选最终目标，等于 exploitation-only。DISCOVER 在困难任务中比 HER 好几个数量级
- **与 Test-Time RL (TTRL) 的联系**：DISCOVER 可被视为成功的 test-time RL，通过数百万步的自主探索解决初始不可解的目标
- **对 LLM 的启示**：论文提到方向性探索原则可用于编程/数学等复杂目标空间的 LLM self-improvement，值得关注

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Achievability + Novelty + Relevance 的统一目标函数是优雅的贡献；与 UCB 的理论连接深刻
- 实验充分度: ⭐⭐⭐⭐⭐ 3 环境 ×2 难度、高维扫描（2-6维）、5 baseline、详细消融和可视化，10 seeds
- 写作质量: ⭐⭐⭐⭐⭐ 理论和实验的叙事线流畅，Figure 2 的直觉展示和 Figure 4 的目标选择可视化极为清晰
- 价值: ⭐⭐⭐⭐⭐ 解决了稀疏奖励 RL 的核心探索难题，理论保证 + 强实验证据，方法简洁通用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning Interestingness in Automated Mathematical Theory Formation](learning_interestingness_in_automated_mathematical_theory_formation.md)
- [\[ICLR 2026\] Thermodynamics of Reinforcement Learning Curricula](../../ICLR2026/reinforcement_learning/thermodynamics_of_reinforcement_learning_curricula.md)
- [\[NeurIPS 2025\] Risk-Averse Total-Reward Reinforcement Learning](risk-averse_total-reward_reinforcement_learning.md)
- [\[NeurIPS 2025\] Reward-Aware Proto-Representations in Reinforcement Learning](reward-aware_proto-representations_in_reinforcement_learning.md)
- [\[ICLR 2026\] ARM-FM: Automated Reward Machines via Foundation Models for Compositional Reinforcement Learning](../../ICLR2026/reinforcement_learning/arm-fm_automated_reward_machines_via_foundation_models_for_compositional_reinfor.md)

</div>

<!-- RELATED:END -->
