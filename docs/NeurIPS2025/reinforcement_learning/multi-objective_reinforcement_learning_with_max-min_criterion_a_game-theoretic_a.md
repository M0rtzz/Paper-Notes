---
title: >-
  [论文解读] Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach
description: >-
  [NeurIPS 2025][多目标强化学习] 将熵正则化的 max-min 多目标强化学习重新建模为两人零和正则连续博弈，提出 ERAM/ARAM 算法，通过镜像下降实现闭式权重更新和全局 last-iterate 收敛，在多种 MORL 环境中显著超越基线。
tags:
  - NeurIPS 2025
  - 多目标强化学习
  - 最大最小公平性
  - 博弈论
  - 镜像下降
  - last-iterate收敛
---

# Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach

**会议**: NeurIPS 2025  
**arXiv**: [2510.20235](https://arxiv.org/abs/2510.20235)  
**代码**: [GitHub](https://github.com/whbyeon/ERAM-ARAM)  
**领域**: 强化学习  
**关键词**: 多目标强化学习, 最大最小公平性, 博弈论, 镜像下降, last-iterate收敛

## 一句话总结

将熵正则化的 max-min 多目标强化学习重新建模为两人零和正则连续博弈，提出 ERAM/ARAM 算法，通过镜像下降实现闭式权重更新和全局 last-iterate 收敛，在多种 MORL 环境中显著超越基线。

## 研究背景与动机

1. **领域现状**: 多目标强化学习 (MORL) 在自动驾驶、资源分配等场景有广泛需求。基于效用函数的方法是主流，常用加权和作为效用函数。

2. **现有痛点**: 加权和效用函数不适合公平性场景。max-min 准则（$\max_\pi \min_{k} V_k^\pi$）更自然但优化困难：min 操作不可微，且无法直接用标准 RL 方法求解。已有方法 [Park et al.] 计算代价高、仅保证 average-iterate 收敛、需要大量内存存储 Q 网络副本。

3. **核心矛盾**: max-min MORL 的 min 非线性使得标准 Bellman operator 不适用；直接优化导致 one-hot 权重向量在最差维度间剧烈振荡，只能保证 average-iterate 收敛。

4. **本文目标**: 设计一个高效、理论可证明收敛的 max-min MORL 算法，具有 last-iterate 收敛保证且计算高效。

5. **切入角度**: 利用 max-min = min-max 的等式（在熵正则化下成立），将问题转化为两人零和博弈的 Nash 均衡求解，并引入对 adversary 的熵正则化来稳定训练。

6. **核心 idea**: 通过对权重向量 $w$ 添加熵正则化 $H(w)$，既可获得 $w$ 更新的闭式解 (softmax)，又避免权重振荡从而实现 last-iterate 收敛。

## 方法详解

### 整体框架

将 $\max_\pi \min_k V_{k,\tau}^\pi$ 转化为两人零和博弈 $\mathcal{RG}$：Learner 学策略 $\pi_\theta$（最大化），Adversary 选权重 $w \in \Delta^K$（最小化）。效用函数为 $u = \langle w, \mathbf{V}_\tau^{\pi_\theta} \rangle - \tau_w H(w)$。

### 关键设计

**1. 两人零和正则博弈建模 (Theorem 3.1)**

- **功能**: 将 max-min MORL 等价为求解正则博弈的 Nash 均衡
- **核心思路**: 证明 $\max_\pi \min_w \langle w, \mathbf{V}_\tau^\pi \rangle = \min_w \max_\pi \langle w, \mathbf{V}_\tau^\pi \rangle$（在熵正则化下成立），且 Nash 均衡的策略部分即为 max-min MORL 的解
- **设计动机**: 一旦转化为零和博弈，可利用成熟的博弈论学习方法（镜像下降）

**2. ERAM 算法：闭式双方更新**

- **功能**: 单循环高效算法
- **核心思路**: 
    - Learner: 使用自然策略梯度 (NPG) 更新 $\theta$，softmax 策略下有闭式：$\pi_{\theta_{t+1}}(a|s) = \frac{1}{Z} (\pi_{\theta_t}(a|s))^\alpha \exp(\frac{1-\alpha}{\tau} Q_{w_t,\tau}^{\pi_{\theta_t}}(s,a))$
    - Adversary: 使用修改的镜像下降更新 $w$，带 $-H(w)$ 正则化，有闭式：$w_{t+1} = \text{softmax}(-\frac{1-\beta}{\tau_w}\mathbf{V}^{\pi_{\theta_t}} + \beta \log w_t)$
- **设计动机**: 选择 KL 散度作为 Bregman 散度 + 负熵正则化，使得 $w$ 更新有解析解且 $\beta \in (0,1)$ 永远成立

**3. ARAM：自适应正则化增强**

- **功能**: 更好的多维度联合优化
- **核心思路**: 将 $H(w)$（等价于到均匀分布的 KL 距离）替换为到动态参考向量 $c$ 的 KL 距离 $-D_{KL}(w\|c)$，其中 $c_i = \text{softmax}(\mathbb{E}[r_i(s,a) \cdot r_{i'}(s,a)])$，$i'$ 为上一轮最差目标
- **设计动机**: ERAM 中 $w$ 趋向均匀分布可能忽略关键差目标；ARAM 更关注与最差目标相关的维度但非单纯聚焦最差

### 损失函数 / 训练策略

- 策略侧：PPO (深度 RL) 或闭式 NPG (tabular)
- 权重侧：闭式 softmax 更新
- 理论保证：last-iterate 指数收敛 $\rho(\eta,\lambda)^t$，迭代复杂度 $O(\frac{1}{\epsilon^2}\log\frac{1}{\epsilon_{acc}})$

## 实验关键数据

### 主实验

**交通信号控制**

| 环境 | ARAM | ERAM | Park et al. | GGF-PPO | GGF-DQN | Avg-DQN |
|------|------|------|-------------|---------|---------|---------|
| Base-4 | **-1160** | -1387 | -1681 | -1731 | -1838 | -2774 |
| Asym-4 | **-2696** | -2732 | -3510 | -3501 | -3053 | -4245 |
| Asym-16 | **-15043** | -17334 | -23663 | -21663 | -17792 | -27499 |

### 消融实验

| 指标 | ERAM | Park et al. |
|------|------|-------------|
| 收敛类型 | Last-iterate ✓ | Average-iterate |
| 模型参数 (Base-4) | 13,704 | 274,084 |
| 参数减少比例 | — | **95%** |
| 训练时间 Base-4 (min) | **111±2.6** | 346±14 |
| 训练时间 Asym-16 (min) | **356±27** | 1125±95 |

### 关键发现

- ARAM 在所有环境中一致最优，ERAM 次优但架构更简单
- 相比 Park et al.，参数量减少约 **95%**，训练时间减少约 **3 倍**
- ERAM 展现 last-iterate 收敛（值单调趋近最优），Park et al. 展现振荡行为
- 16 目标场景 (Asym-16) 中优势更明显，说明方法在高维目标空间中更具可扩展性

## 亮点与洞察

- 理论与实践高度统一：last-iterate 收敛理论直接指导了实际算法设计
- 熵正则化一石二鸟：既解决策略不确定性问题，又为 $w$ 更新提供闭式解
- 将 MORL 公平性问题优雅地转化为博弈论框架，使大量博弈论工具可直接复用
- ARAM 的自适应参考向量是介于"均匀关注所有目标 (ERAM)"和"只关注最差目标 (GGF-PPO)"之间的优雅折中

## 局限与展望

- ARAM 的理论分析留作未来工作，目前仅有 ERAM 的收敛证明
- tabular 理论限制：收敛证明依赖 softmax 策略参数化的闭式 NPG，深度 RL 场景的理论保证未建立
- $\tau_w$ 的下界条件较保守（$\tau_w \geq O(K/\tau(1-\gamma)^4)$），实际中可能需要更精细的调参
- 仅考虑了 min 作为效用函数，其他非线性效用（如 Nash 社会福利函数）的扩展未讨论

## 相关工作与启发

- Park et al. [2024] 的凸重建方法是直接前驱，本文通过博弈论视角大幅简化
- 与分布鲁棒 RL 的联系：$w \in \Delta^K$ 可视为有限不确定集上的分布，熵正则化对应内化的奖励不确定性
- GGF-PPO 是本方法的特例（$\tau_w = 0$, 无 Bregman 约束 → one-hot $w$）

## 评分

- **新颖性**: ⭐⭐⭐⭐ 博弈论重建虽非全新，但闭式 $w$ 更新和 last-iterate 收敛证明是重要贡献
- **实验充分度**: ⭐⭐⭐⭐ 涵盖 tabular、交通信号控制等多个环境，复杂度对比充分
- **写作质量**: ⭐⭐⭐⭐ 定理-算法-实验流程清晰，理论部分严谨
- **价值**: ⭐⭐⭐⭐ 为公平性 RL 提供了目前最高效且有理论保证的算法

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] TRiCo: Triadic Game-Theoretic Co-Training for Robust Semi-Supervised Learning](trico_triadic_game-theoretic_co-training_for_robust_semi-supervised_learning.md)
- [\[NeurIPS 2025\] Solving Neural Min-Max Games: The Role of Architecture, Initialization & Dynamics](solving_neural_min-max_games_the_role_of_architecture_initialization_dynamics.md)
- [\[NeurIPS 2025\] VolleyBots: A Testbed for Multi-Drone Volleyball Game Combining Motion Control and Strategic Play](volleybots_a_testbed_for_multi-drone_volleyball_game_combining_motion_control_an.md)
- [\[NeurIPS 2025\] Confounding Robust Deep Reinforcement Learning: A Causal Approach](confounding_robust_deep_reinforcement_learning_a_causal_approach.md)
- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](thompson_sampling_for_multi-objective_linear_contextual_bandit.md)

<!-- RELATED:END -->
