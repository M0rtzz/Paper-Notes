---
title: >-
  [论文解读] Strict Subgoal Execution: Reliable Long-Horizon Planning in Hierarchical Reinforcement Learning
description: >-
  [ICLR 2026][强化学习][分层RL] 提出 SSE（Strict Subgoal Execution）框架，通过**前沿经验回放（FER）** 严格区分子目标到达成功与失败，配合解耦探索策略和失败感知路径优化，在每个高层步骤内强制完成子目标到达，显著减少高层决策步数并提升长时程任务成功率。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "分层RL"
  - "子目标执行"
  - "图规划"
  - "前沿经验回放"
  - "长时程任务"
---

# Strict Subgoal Execution: Reliable Long-Horizon Planning in Hierarchical Reinforcement Learning

**会议**: ICLR 2026  
**arXiv**: [2506.21039](https://arxiv.org/abs/2506.21039)  
**代码**: [https://github.com/Jaebak1996/SSE](https://github.com/Jaebak1996/SSE)  
**领域**: 分层强化学习 / 目标条件 RL  
**关键词**: 分层RL, 子目标执行, 图规划, 前沿经验回放, 长时程任务

## 一句话总结

提出 SSE（Strict Subgoal Execution）框架，通过**前沿经验回放（FER）** 严格区分子目标到达成功与失败，配合解耦探索策略和失败感知路径优化，在每个高层步骤内强制完成子目标到达，显著减少高层决策步数并提升长时程任务成功率。

## 研究背景与动机

- **长时程目标条件任务的挑战**：目标遥远、奖励稀疏，探索困难
- **HER 在高层的问题**：传统图-层次 RL 对高层策略使用 HER（Hindsight Experience Replay），将失败轨迹中的中间状态当作虚拟子目标。这导致：
    - 高层策略反复选择不可达子目标
    - 高层轨迹过长，信用分配困难
    - 同一子目标产生高度不一致的转移
- **核心思路**：不是让高层策略反复尝试不可达子目标，而是严格执行——成功就继续，失败就立即终止。

## 方法详解

### 整体框架

SSE 包含三个核心组件：前沿经验回放（FER）、解耦探索策略、失败感知路径优化。

### 1. 前沿经验回放（FER）

FER 将高层经验分为三种类型：

$$\mathcal{B}_F^h = \begin{cases} (s_t, g, \tilde{g}_t, \sum_{j=t}^{t'-1} r_j, s_{t'}) & \text{(成功)} \\ (s_t, g, \tilde{g}_t, 0, s_T) & \text{(失败终止)} \\ (s_t, g, \text{wp}_{\text{final}}, \sum_{j=t}^{t_{\text{wp}}-1} r_j, s_{t_{\text{wp}}}) & \text{(部分成功)} \end{cases}$$

- **成功**：低层策略成功到达子目标，记录完整回报
- **失败终止**：子目标不可达（$\|\phi(s_{t'}) - \tilde{g}_t\| \geq \lambda$），回报为 0，下一状态为终止态 $s_T$，立即截断回合
- **部分成功**：失败时记录最后成功到达的路标点 $\text{wp}_{\text{final}}$

### 2. 解耦探索策略

分为利用策略 $\pi^h$ 和探索策略 $\pi^{\text{exp}}$：

利用策略（$\epsilon$-greedy）：
$$\pi^h(\tilde{g}_t | s_t, g) = \begin{cases} \arg\max_{\tilde{g}} Q^h(s_t, \tilde{g}, g) & \text{概率 } 1-\epsilon \\ \text{Uniform}(\mathcal{G}) & \text{概率 } \epsilon \end{cases}$$

探索策略：
$$\pi^{\text{exp}}(\tilde{g}_t | s_t, g) = \begin{cases} g & \text{概率 } 1/3 \\ \tilde{g}_{\max,t} & \text{概率 } 1/3 \\ \tilde{g}_{\text{novel}} \sim \text{Uniform}(V_{\text{novel}}) & \text{概率 } 1/3 \end{cases}$$

两种策略按比例 $\eta : (1-\eta)$ 混合使用。

### 3. 失败感知路径优化

在图 $G = (V, E)$ 上调整边代价，使 Dijkstra 规避失败频繁区域：

$$\tilde{d}(v_1 \to v_2) = d(v_1 \to v_2) \times \max(1, c_{\text{dist}} \cdot \text{ratio}_{\text{fail}}(v_2))$$

其中 $\text{ratio}_{\text{fail}}$ 为目标节点的失败比例。

### 两种实现变体

- **SSE (Grid)**：基于网格的离散化方法，适用于 2D/3D 目标空间
- **SSE (Model)**：基于神经网络的方法，可扩展到高维目标空间

## 实验结果

### 主实验：9 个长时程任务（5 seeds）

| 环境 | HIRO | HRAC | HIGL | DHRL | BEAG | NGTE | SSE |
|------|------|------|------|------|------|------|-----|
| U-maze | ✗ | ✗ | △ | △ | ✓ | ✓ | **✓✓** |
| π-maze | ✗ | ✗ | △ | △ | ✓ | ✓ | **✓✓** |
| AntMazeComplex | ✗ | ✗ | ✗ | △ | ✓ | △ | **✓✓** |
| AntMazeBottleneck | ✗ | ✗ | ✗ | ✗ | △ | ✗ | **✓✓** |
| AntKeyChest | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓✓** |
| AntDoubleKeyChest | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓✓** |

（✓✓=高成功率快收敛, ✓=成功, △=部分成功, ✗=失败）

### 消融实验（AntDoubleKeyChest）

| 变体 | 表现 |
|------|------|
| SSE 完整 | ✓（约 3M 步解决） |
| SSE + FPS（替换网格为 FPS） | 成功但收敛慢 |
| SSE + HER（替换 FER 为 HER） | **完全失败** |
| SSE 无 $\mathcal{B}_F^h$（无 FER） | **完全失败** |
| SSE 无 $\pi^{\text{exp}}$（无探索策略） | 显著退化 |
| SSE 无路径优化 | 适度退化 |

### 关键发现

- FER 是核心——移除 FER 或使用 HER 均导致彻底失败
- SSE 使智能体仅需 **单个高层步骤** 即可到达地图上任意可达位置
- AntDoubleKeyChest 仅需 **3 个高层步骤** 完成整个任务（拿两把钥匙+到达终点）

## 亮点与洞察

1. **每步必达的思想**：强制子目标必须被到达才继续，根本性地减少了高层决策步数
2. **FER 的精巧设计**：将经验细致分为成功/失败/部分成功三类，定位可达边界
3. **无需课程学习**：SSE 自动发现子目标的正确执行序列
4. **计算效率优势**：失败时立即终止回合，避免长无效轨迹，实际迭代更快

## 局限性

- 引入新超参数（$\eta$, $c_{\text{dist}}$, $d_\mathcal{G}$），但消融研究表明有效范围稳定
- 假设目标空间 $\mathcal{G}$ 已知——这在许多环境中是标准假设
- 网格方法仅适用于低维目标空间
- 主要在固定目标设置下评估

## 相关工作

- **目标条件 RL**：HER、UVFA、优先目标采样
- **图规划 HRL**：HIGL、DHRL、BEAG、NGTE
- **前沿探索**：与传统前沿探索（边界状态）不同，SSE 的"前沿"指成功/失败边界

## 评分

- **创新性**: ⭐⭐⭐⭐ — 严格子目标执行思想简单却高效
- **技术深度**: ⭐⭐⭐⭐ — FER 设计精巧，三种经验类型的划分有充分动机
- **实验充分性**: ⭐⭐⭐⭐⭐ — 9 个环境，全面对比 7 个基线，消融详尽
- **实用价值**: ⭐⭐⭐⭐ — 在复杂长时程任务上显著优于现有方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Long-Horizon Model-Based Offline Reinforcement Learning Without Explicit Conservatism](../../ICML2026/reinforcement_learning/long-horizon_model-based_offline_reinforcement_learning_without_explicit_conserv.md)
- [\[NeurIPS 2025\] Reinforcement Learning for Long-Horizon Multi-Turn Search Agents](../../NeurIPS2025/reinforcement_learning/reinforcement_learning_for_long-horizon_multi-turn_search_agents.md)
- [\[ICLR 2026\] LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)
- [\[ICLR 2026\] Model Predictive Adversarial Imitation Learning for Planning from Observation](model_predictive_adversarial_imitation_learning_for_planning_from_observation.md)
- [\[AAAI 2026\] ManiLong-Shot: Interaction-Aware One-Shot Imitation Learning for Long-Horizon Manipulation](../../AAAI2026/reinforcement_learning/manilong-shot_interaction-aware_one-shot_imitation_learning_for_long-horizon_man.md)

</div>

<!-- RELATED:END -->
