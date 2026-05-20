---
title: >-
  [论文解读] Enhancing Control Policy Smoothness by Aligning Actions with Predictions from Preceding States
description: >-
  [AAAI 2026][强化学习] 提出 **ASAP（Action Smoothing by Aligning Actions with Predictions from Preceding States）**，一种基于**转移诱导相似状态定义**的强化学习动作平滑方法…
tags:
  - "AAAI 2026"
  - "强化学习"
  - "动作平滑"
  - "Lipschitz约束"
  - "策略平滑性"
  - "机器人控制"
---

# Enhancing Control Policy Smoothness by Aligning Actions with Predictions from Preceding States

**会议**: AAAI 2026  
**arXiv**: [2601.18479](https://arxiv.org/abs/2601.18479)  
**代码**: [https://github.com/AIRLABkhu/ASAP](https://github.com/AIRLABkhu/ASAP)  
**领域**: 其他  
**关键词**: 强化学习, 动作平滑, Lipschitz约束, 策略平滑性, 机器人控制

## 一句话总结

提出 **ASAP（Action Smoothing by Aligning Actions with Predictions from Preceding States）**，一种基于**转移诱导相似状态定义**的强化学习动作平滑方法，通过空间约束（对齐前一状态的预测动作）和时间约束（惩罚二阶动作差异）有效抑制高频动作振荡，在 Gymnasium 和 Isaac-Lab 环境中优于现有方法。

## 研究背景与动机

### 问题场景
深度强化学习（DRL）在连续控制任务中取得了显著成功，但策略输出的**高频动作振荡**是将学到的策略部署到真实世界的重大障碍：
- **硬件磨损**：高频振荡导致机械部件寿命大幅缩短
- **安全隐患**：不平滑的动作可能导致用户体验差或安全问题
- **根本原因**：actor 网络对微小状态扰动过度敏感，产生大幅动作偏差

### 现有方法的局限

现有方法分为两大类：

**架构方法（Architectural Methods）**：
- 如 Spectral Normalization、LipsNet 等，通过修改网络结构满足 Lipschitz 约束
- **缺点**：推理时引入额外计算开销，且在不同环境间性能波动大

**损失惩罚方法（Loss Penalty Methods）**：
- 如 CAPS、L2C2、Grad-CAPS，在策略损失中加入平滑正则项
- **关键问题**：需要定义"相似状态"来强制动作一致性
    - **CAPS**：从当前状态周围的**固定高斯分布**采样相似状态 → 启发式定义，不反映真实状态分布
    - **L2C2**：自适应边界但仍基于**合成构造** → 不匹配实际环境动力学
- **后果**：相似状态不准确导致理论保证失效和性能下降

### 核心动机
关键洞察：**从同一前驱状态转移出来的状态应当是"相似"的**。这种基于转移函数的相似状态定义：
1. 仅使用环境反馈和实际收集的数据
2. 天然反映系统动力学
3. 可以被证明形成有界邻域

## 方法详解

### 整体框架

ASAP 的策略总损失：
$$J_{\pi_\phi}^{\text{ASAP}} = J_{\pi_\phi} + \lambda_S L_S + \lambda_P L_P + \lambda_T L_T$$

包含：标准 RL actor 损失 + 空间约束项 + 预测器训练项 + 时间约束项。

### 关键设计

#### 1. **转移诱导相似状态（Definition 3）**

- **核心定义**：给定状态 $s_t$，其相似状态分布定义为从前驱状态 $s_{t-1}$ 转移出的所有可能下一状态分布：
$$\text{sim}(s_t) = P(\cdot | s_{t-1})$$
- **有界邻域保证（Lemma 1）**：在转移函数关于噪声满足局部 Lipschitz 连续（Assumption 1）且噪声有界（Assumption 2）的条件下，任意两个相似状态的距离有上界：
$$d_S(s_t^{(1)}, s_t^{(2)}) \leq 2K_\xi(s_{t-1}, a_{t-1}) \sigma_\xi$$
- **设计动机**：
    - 与 CAPS/L2C2 不同，不需要合成状态，完全基于实际收集的转移数据
    - 保证了相似状态区域的有界性，为施加局部 Lipschitz 约束提供了理论基础
    - 分布与真实转移核 $P_{\text{real}}(\cdot | s_{t-1})$ 完全一致

#### 2. **空间平滑项（Spatial Loss）**

- **复合函数 Lipschitz 约束（Theorem 1）**：$f \circ T$ 在噪声空间上是局部 Lipschitz 连续的，常数为 $K_{\text{comp}} = K \cdot K_\xi$
- **推导出的损失**：
$$L_S = \|\pi_\phi(s_t) - \text{stopgrad}(\pi_P(s_{t-1}))\|_2^2$$
  即最小化当前动作 $\pi_\phi(s_t)$ 与前一状态的**预测下一动作** $\pi_P(s_{t-1})$ 之间的差异。
- **预测器损失**：
$$L_P = \|\pi_P(s_{t-1}) - \text{stopgrad}(\pi_\phi(s_t))\|_2^2$$
  训练预测器模仿实际策略输出。使用 stopgrad 分离两个损失，分配不同的强度参数。

#### 3. **时间平滑项（Temporal Loss）**

直接采用 Grad-CAPS 的二阶差分惩罚：
$$L_T = \left\|\frac{a_{t+1} - 2a_t + a_{t-1}}{\tanh(a_{t+1} - a_{t-1}) + \epsilon}\right\|_2^2$$

- 惩罚动作的二阶变化（"加速度"），而非一阶变化
- 分母 $\tanh(\cdot)$ 提供对动作尺度的自适应归一化
- 二阶差分比一阶差分更灵活：允许平稳的动作变化，同时抑制高频振荡

#### 4. **预测器实现**

- 在 actor 的 MLP 上增加一个 prediction head（与 action head 共享底层）
- action head 用 $L_S$ 训练，prediction head 用 $L_P$ 训练
- 解决"移动目标"问题：使用更多并行环境收集数据 + 减小 $\lambda_P$ 权重

### 损失函数 / 训练策略

- **兼容多种 RL 算法**：$J_{\pi_\phi}$ 可以是 PPO 或 SAC 的标准 actor 损失
- **超参数**：$\lambda_S$, $\lambda_P$, $\lambda_T$ 分别控制空间、预测器、时间约束的强度
- **推理零开销**：训练时增加计算，推理时与基线完全相同（不修改网络结构）

## 实验关键数据

### 实验设置
- **Gymnasium**：LunarLander, Pendulum, Reacher, Ant, Hopper, Walker（6个环境）
- **Isaac-Lab**：Reach-Franka, Lift-Cube-Franka, Repose-Cube-Allegro, Anymal-Velocity-Rough（4个高保真机器人环境）
- **指标**：Cumulative Return (re, ↑) 和 Smoothness Score (sm, ↓)

### 主实验

**PPO 设置（Table 2, 选择关键环境）**：

| 方法 | Hopper re↑ | Hopper sm↓ | Walker re↑ | Walker sm↓ |
|------|-----------|-----------|-----------|-----------|
| PPO Base | 2902 | 1.709 | 2654 | 1.764 |
| CAPS | 2362 | 0.281 | 2179 | 0.565 |
| L2C2 | 2345 | 1.344 | 2014 | 1.686 |
| GRAD | 2737 | 0.193 | 1967 | 0.342 |
| **ASAP** | **2691** | **0.179** | **3128** | **0.345** |

ASAP 在 Hopper 上 sm 降低 **89.5%**，Walker 上 sm 降低 **80.4%**。

**SAC 设置（Table 3, 选择关键环境）**：

| 方法 | Hopper re↑ | Hopper sm↓ | Walker re↑ | Walker sm↓ |
|------|-----------|-----------|-----------|-----------|
| SAC Base | 3349 | 0.856 | 4476 | 0.823 |
| CAPS | 3413 | 0.793 | 4320 | 0.815 |
| GRAD | 3190 | 0.588 | 4339 | 0.612 |
| **ASAP** | **3448** | **0.498** | **4665** | **0.578** |

**Isaac-Lab（Table 4）**：

| 任务 | PPO Base re | PPO Base sm | ASAP re | ASAP sm |
|------|------------|------------|---------|---------|
| Reach-Franka | 0.380 | 0.959 | **0.525** | **0.658** |
| Lift-Cube-Franka | **136.1** | 2.315 | 134.0 | **0.926** |
| Anymal-Velocity-Rough | **16.69** | 3.502 | 16.09 | **2.861** |

在高保真机器人任务中也展现了一致的平滑性提升。

### 消融实验

**转移诱导相似状态有效性验证（Table 1）**：

| 环境 | SAC Base sm | 预测器微调后 sm | 改善 |
|------|-----------|---------------|------|
| Hopper | 0.857 | 0.712 | -16.9% |
| Walker | 0.836 | 0.715 | -14.4% |
| LunarLander | 0.296 | 0.227 | -23.3% |

仅通过预测器微调就能一致降低 sm，验证了转移诱导相似状态的有效性。

**空间项来源对比**：

| 空间项 $L_S$ | 时间项 $L_T$ | Hopper re | Hopper sm | Walker re | Walker sm |
|-------------|-------------|-----------|-----------|-----------|-----------|
| — (无) | GRAD | 2963 | 0.241 | 2659 | 0.541 |
| CAPS | GRAD | 2264 | 0.201 | 2303 | 0.467 |
| L2C2 | GRAD | 2925 | 0.227 | 2500 | 0.519 |
| ASAP (ours) | GRAD | 2691 | 0.179 | 3128 | 0.345 |

ASAP 的空间项在保持 re 的同时实现最低 sm。

**与架构方法结合（Table 5）**：

| 方法 | Walker re | Walker sm |
|------|-----------|-----------|
| LipsNet | 3942 | 0.915 |
| LipsNet + CAPS | 3464 | 0.665 |
| LipsNet + ASAP | **4475** | **0.485** |

ASAP 可以与架构方法叠加使用，进一步提升效果。

### 关键发现

1. **PPO 平均 sm 降低 43.3%**，SAC 平均 sm 降低 27.9%，且 re 保持稳定或提升。
2. **ASAP 在 5/6（PPO）和 4/6（SAC）环境中取得最佳平滑度**。
3. **与 Grad-CAPS 互补**：ASAP 的空间约束 + Grad 的时间约束配合效果最好。
4. **在 Isaac-Lab 高保真环境中同样有效**，验证了向真实机器人场景迁移的潜力。

## 亮点与洞察

1. **理论基础扎实**：从 Lipschitz 连续性理论出发，严格证明了转移诱导相似状态形成有界邻域，推导出复合函数 Lipschitz 约束，再化简为可优化的损失函数。
2. **与现有方法清晰对比**：通过 Figure 1 的对比图直观展示了 CAPS（固定高斯）、L2C2（自适应但合成）和 ASAP（基于真实转移分布）的本质差异。
3. **推理零开销**：作为损失惩罚方法，只在训练时增加计算，推理时与基线相同，适合部署。
4. **模块化设计**：空间项和时间项可独立使用，也可与架构方法组合。

## 局限与展望

1. **Hopper 环境中 re 略有下降**：可能是在需要快速动作变化的区域过度平滑了，需要自适应调节平滑强度。
2. **假设要求**：转移函数关于噪声的 Lipschitz 连续性在大多数物理场景中成立，但在某些极端场景可能不满足。
3. **预测器的训练稳定性**：on-policy 方法（PPO）中，快速变化的策略分布可能导致预测器滞后。
4. **超参数选择**：$\lambda_S$, $\lambda_P$, $\lambda_T$ 的调节需要环境特定调优。
5. **仅评估了 PPO 和 SAC**：未测试 TD3、DDPG 等其他连续控制算法。

## 相关工作与启发

- **CAPS（Mysore et al. 2021）**：首次将 Lipschitz 约束分解为时间和空间两个维度，是本文的主要对比对象。
- **L2C2（Kobayashi 2022）**：引入局部 Lipschitz 连续性概念，但相似状态定义仍基于合成构造。
- **Grad-CAPS（Lee et al. 2024）**：提出二阶差分时间惩罚，被 ASAP 直接采用。
- **启发**：当"正确的定义"是方法的核心时，从系统动力学出发定义概念（而非人为构造）往往更有效。这种"让环境告诉你答案"的思路具有通用价值。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 转移诱导相似状态的定义是核心贡献，理论推导完整
- 实验充分度: ⭐⭐⭐⭐⭐ — 6个Gymnasium + 4个Isaac-Lab环境，PPO/SAC双算法，消融全面
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，但符号略多
- 价值: ⭐⭐⭐⭐ — 对sim-to-real转移有直接意义，方法实用且开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] A Switching Framework for Online Interval Scheduling with Predictions](a_switching_framework_for_online_interval_scheduling_with_pr.md)
- [\[ICLR 2026\] Enhancing Generative Auto-bidding with Offline Reward Evaluation and Policy Search](../../ICLR2026/others/enhancing_generative_auto_bidding.md)
- [\[AAAI 2026\] Deadline-Aware, Energy-Efficient Control of Domestic Immersion Hot Water Heaters](deadline-aware_energy-efficient_control_of_domestic_immersion_hot_water_heater.md)
- [\[AAAI 2026\] Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit](area-optimal_control_strategies_for_heterogeneous_multi-agen.md)
- [\[AAAI 2026\] From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback](from_sequential_to_recursive_enhancing_decision-focused_learning_with_bidirectio.md)

</div>

<!-- RELATED:END -->
