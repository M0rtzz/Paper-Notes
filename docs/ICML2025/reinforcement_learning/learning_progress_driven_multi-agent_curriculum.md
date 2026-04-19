---
title: >-
  [论文解读] Learning Progress Driven Multi-Agent Curriculum
description: >-
  [ICML2025][多智能体强化学习] 提出 SPMARL，以基于 TD 误差的学习进度（而非回报）驱动智能体数量的自适应课程分布，解决多智能体稀疏奖励任务中回报估计高方差与信用分配困难两大问题。
tags:
  - ICML2025
  - 多智能体强化学习
  - 课程学习
  - 学习进度
  - TD误差
  - 自步学习
---

# Learning Progress Driven Multi-Agent Curriculum

**会议**: ICML2025  
**arXiv**: [2205.10016](https://arxiv.org/abs/2205.10016)  
**代码**: [GitHub](https://github.com/wenshuaizhao/spmarl)  
**领域**: 多智能体课程学习 / 强化学习  
**关键词**: 多智能体强化学习, 课程学习, 学习进度, TD误差, 自步学习

## 一句话总结

提出 SPMARL，以基于 TD 误差的学习进度（而非回报）驱动智能体数量的自适应课程分布，解决多智能体稀疏奖励任务中回报估计高方差与信用分配困难两大问题。

## 研究背景与动机

在多智能体强化学习（MARL）中，稀疏奖励导致探索极其困难。课程学习（CRL）是一种有效策略：通过先在简单任务上训练、再逐步过渡到目标任务来缓解探索难题。**智能体数量**是多智能体场景中天然的课程变量。

现有方法的局限：

- **手工线性课程**：DyMA-CL、EPC、VACL 等方法以预设的线性方式（少→多或多→少）调整智能体数量，缺乏自适应能力
- 直接将单智能体自步学习 SPRL 扩展到 MARL（即 SPRLM）虽能自适应选择任务，但存在**两个缺陷**：
    1. **回报估计高方差**：稀疏奖励下每个 episode 只能获得一次回报估计，方差极大
    2. **信用分配恶化**：在许多任务中，增加智能体数量会自然带来更高回报（如 Simple-Spread 中 20 个智能体随机移动即可覆盖大量地标），基于回报的课程会偏向"看似简单实则无学习价值"的高智能体数任务

## 方法详解

### 问题形式化

任务建模为 Dec-POMDP $\langle \mathcal{S}, \{\mathcal{O}^i\}, \{\mathcal{A}^i\}, r, \mathcal{P}, \gamma \rangle$，其中 $n$ 个智能体共享全局奖励。通过上下文强化学习（Contextual RL）引入上下文 $\mathbf{c}$（即智能体数量）参数化不同难度的 MDP。

### SPRLM：直接扩展 SPRL 到多智能体

SPRLM 将 SPRL 的约束优化框架应用于智能体数量控制：

$$\min_{\nu} D_{\mathrm{KL}}(p(\mathbf{c}|\nu) \| \mu(\mathbf{c}))$$

满足约束：(1) 期望回报 $\mathbb{E}_{p(\mathbf{c}|\nu)}[J(\theta, \mathbf{c})] \geq V_{\mathrm{LB}}$；(2) 前后分布的 KL 散度 $D_{\mathrm{KL}}(p(\mathbf{c}|\nu_k) \| p(\mathbf{c}|\nu_{k+1})) \leq \epsilon$。

采用**两阶段优化**：

- **阶段 1**：当期望性能低于阈值 $V_{\mathrm{LB}}$ 时，通过重要性采样最大化回报
- **阶段 2**：性能达标后，最小化当前分布与目标分布的 KL 散度，逐步收敛到目标任务

### SPMARL：以学习进度替代回报

核心改进：用 **TD 误差（值函数损失）** 替代 episode 回报作为课程优化目标。定义学习进度：

$$\mathrm{LP}(c) = \frac{1}{2} \mathbb{E}_{s, \mathbf{a} \sim \pi(\mathbf{a}|s,\mathbf{c})} [\| R(s, \mathbf{a}) - V(s) \|^2]$$

其中 $R(s,\mathbf{a})$ 是折扣回报，$V(s)$ 是值函数估计。阶段 1 的优化目标变为：

$$\max_{\nu_{k+1}} \frac{1}{M} \sum_{i=1}^{M} \frac{p(\mathbf{c}_i|\nu_{k+1})}{p(\mathbf{c}_i|\nu_k)} \mathrm{LP}_\theta(\mathbf{c}_i)$$

**为什么 TD 误差有效：**

1. **低方差**：TD 误差在每个状态转移上都可计算，而非仅 episode 结束时才有信号
2. **自然反映策略改进**：值损失大 → 策略仍在显著变化 → 任务对当前策略有学习价值；值损失趋零 → 策略已收敛 → 该难度任务已无更多可学内容
3. **缓解信用分配问题**：不直接追求高回报的任务，而是追求"最能推动策略提升"的任务

阶段 2 保持不变——仍用性能阈值 $V_{\mathrm{LB}}$ 判断何时向目标分布收敛。

## 实验关键数据

在三个稀疏奖励基准上评估，目标任务均有明确的智能体数量：

| 基准任务 | 目标智能体数 | 奖励设计 | SPMARL 表现 |
|---|---|---|---|
| MPE Simple-Spread | 8 | 覆盖≥4地标才有奖励 | 收敛最快，回报最高 |
| XOR Matrix Game | 20 | 所有玩家选不同动作才得分 | 最快收敛到最优 |
| SMACv2 Protoss 5v5 | 5 | 胜+1 / 负-1 | 领先或持平 |
| SMACv2 Protoss 6v6 | 6 | 胜+1 / 负-1 | 显著领先 |
| SMACv2 Protoss 7v7 | 7 | 胜+1 / 负-1 | 显著领先 |
| SMACv2 Protoss 8v8 | 8 | 胜+1 / 负-1 | 显著领先 |

关键对比结果：

- **W/O teacher**（无课程直接训练）：在所有任务上完全失败，奖励为零
- **Linear**（线性课程）：Simple-Spread 完全失败；XOR 最终收敛但不稳定
- **ALPGMM**：训练回报最高但偏向过多智能体，评估性能不佳
- **VACL**：无法保证收敛到目标分布，多项任务失败
- **SPRLM**：多数任务优于启发式基线，但 SMACv2 极端奖励下表现不稳定
- **SPMARL**：在所有任务上一致性最佳或持平

目标估计方差对比（SMACv2 Protoss 7v7 / 8v8）：SPMARL 的 TD 误差标准差远低于 SPRLM 的 episode 回报标准差，验证了"低方差带来更稳定课程"的假设。

## 亮点与洞察

1. **问题发现精准**：明确指出基于回报的 ACRL 在 MARL 中的两个根本缺陷（高方差 + 信用分配），并非简单套用单智能体方法
2. **方法优雅简洁**：仅改变课程目标从回报到 TD 误差，不改变整体优化框架，改动最小但效果显著
3. **直觉类比**：TD 误差驱动的任务选择类似于 Prioritized Experience Replay（PER），但在任务层面而非样本层面做优先级排序
4. **实验覆盖全面**：三个不同类型的基准（覆盖任务 / 协调任务 / 对抗任务），均为严格稀疏奖励设定
5. **课程可视化有价值**：展示了不同方法生成的课程分布轨迹，直观说明 SPMARL 为何更高效

## 局限与展望

1. **KL 约束的必要性未探讨**：作者自己指出，当目标从回报换为学习进度时，SPRL 原有的 KL 约束可能不再必要，这是重要的未完成工作
2. **仅控制智能体数量**：上下文变量仅为智能体数量，未同时控制环境参数（如地图大小、敌方强度等）
3. **底层算法绑定 MAPPO**：所有实验仅在 MAPPO 上验证，未测试 QMIX、MADDPG 等其他 MARL 算法
4. **超参数敏感性**：$V_{\mathrm{LB}}$、$\epsilon$ 等阈值需手动设定，不同任务可能需要不同的值
5. **可扩展性未验证**：测试的最大智能体数为 20，更大规模时性能如何未知

## 相关工作与启发

- **SPRL** (Klink et al., 2021)：自步强化学习，本工作的直接基础
- **VACL** (Chen et al., 2021)：变分自动课程学习，NeurIPS 2021，处理多智能体课程
- **DyMA-CL** (Wang et al., 2020)：动态多智能体课程学习，手工设计智能体数量递增
- **ALPGMM** (Portelas et al., 2020)：基于学习进度的课程方法，但仍用回报差衡量进度
- **PER** (Schaul et al., 2015)：优先经验回放，TD 误差驱动的样本优先级，SPMARL 将此思想提升到任务级别

## 评分

- 新颖性: ⭐⭐⭐⭐ — 问题发现新颖，解决方案简洁有效
- 实验充分度: ⭐⭐⭐⭐ — 三种基准覆盖不同场景，消融充分
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，公式推导连贯
- 价值: ⭐⭐⭐⭐ — 为多智能体课程学习提供了即插即用的改进方案

<!-- RELATED:START -->

## 相关论文

- [Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration](enhancing_cooperative_multi-agent_reinforcement_learning_with_state_modelling_an.md)
- [Robust and Diverse Multi-Agent Learning via Rational Policy Gradient](../../NeurIPS2025/reinforcement_learning/robust_and_diverse_multi-agent_learning_via_rational_policy_gradient.md)
- [Sequential Multi-Agent Dynamic Algorithm Configuration](../../NeurIPS2025/reinforcement_learning/sequential_multi-agent_dynamic_algorithm_configuration.md)
- [Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [Learning to Incentivize in Repeated Principal-Agent Problems with Adversarial Agent Arrivals](learning_to_incentivize_in_repeated_principal-agent_problems_with_adversarial_ag.md)

<!-- RELATED:END -->
