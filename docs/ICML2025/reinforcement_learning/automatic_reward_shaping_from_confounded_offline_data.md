---
title: >-
  [论文解读] Automatic Reward Shaping from Confounded Offline Data
description: >-
  [ICML 2025][reward shaping] 提出首个理论上有保障的数据驱动方法，从含未观测混淆因子的离线数据中自动学习基于势的奖励整形函数 (PBRS)，通过因果贝尔曼最优方程上界最优状态值作为势函数，并证明所得 Q-UCB Shaping 算法在伪次优状态-动作对上享有比 vanilla Q-UCB 更优的 gap-dependent regret bound。
tags:
  - ICML 2025
  - reward shaping
  - confounded MDP
  - causal inference
  - PBRS
  - offline RL
---

# Automatic Reward Shaping from Confounded Offline Data

**会议**: ICML 2025  
**作者**: Mingxuan Li, Junzhe Zhang, Elias Bareinboim (Columbia University)  
**arXiv**: [2505.11478](https://arxiv.org/abs/2505.11478)  
**代码**: 无  
**领域**: 强化学习, 因果推理  
**关键词**: reward shaping, confounded MDP, causal inference, PBRS, offline RL  

## 一句话总结

提出首个理论上有保障的数据驱动方法，从含未观测混淆因子的离线数据中自动学习基于势的奖励整形函数 (PBRS)，通过因果贝尔曼最优方程上界最优状态值作为势函数，并证明所得 Q-UCB Shaping 算法在伪次优状态-动作对上享有比 vanilla Q-UCB 更优的 gap-dependent regret bound。

## 研究背景与动机

**领域现状**：基于势的奖励整形 (PBRS) 是加速 RL 学习的有效技术——通过在原始奖励上叠加状态势差信号，引导智能体更快找到高回报状态，同时保证最优策略不变。Ng et al. (1999) 指出最优状态值函数 $V_h^*(s)$ 是理想的势函数候选。

**现有痛点**：实践中势函数设计要么依赖领域专家手工构造（昂贵、易误导），要么从离线数据中用标准 off-policy 方法估计。但当离线数据来源于不可控的行为策略时，未观测的混淆变量（如示教者观察到但学习者看不到的环境状态）导致标准 off-policy 估计产生系统偏差——甚至可能得到"被稳定惩罚"的错误势信号。

**核心矛盾**：在含混淆偏差的 MDP (CMDP) 中，转移函数 $\mathcal{T}$ 和奖励函数 $\mathcal{R}$ 从观测数据中不可点辨识（无论样本量多大）。直接用行为数据估计的值函数可能严重偏离真实最优值，导致整形信号误导智能体。

**本文目标** (1) 如何从混淆离线数据中稳健地构造势函数？(2) 使用这些势函数的在线学习器能获得怎样的理论保证？

**切入角度**：借鉴因果推理中部分辨识 (partial identification) 的思路——虽然无法精确辨识 $\mathcal{T}$ 和 $\mathcal{R}$，但可以用 Manski bounds 对它们进行上界估计。只要势函数上界了最优状态值（保守乐观条件），就能同时保证策略不变性和加速探索。

**核心 idea**：用因果偏序关系对含混淆的离线数据推导最优状态值的上界，作为 PBRS 的势函数，实现自动化奖励整形。

## 方法详解

### 整体框架

整个方法分为两阶段流水线：
1. **离线阶段**：从可能含混淆偏差的多个离线数据集中，通过 Causal Bellman Optimal Equation 计算最优状态值的因果上界 $\bar{V}_h(s)$，作为势函数 $\phi_h(s)$
2. **在线阶段**：将势函数代入改进的 Q-UCB 算法（Q-UCB Shaping），在线交互学习最优策略

### 关键设计

1. **Confounded MDP 建模**:

    - 功能：形式化未观测混淆存在的 MDP
    - 核心思路：CMDP 在标准 MDP 基础上引入外生噪声 $U_h$，同时影响行为策略 $\beta_h(S_h, U_h)$、奖励 $r_h(S_h, X_h, U_h)$ 和下一状态 $\tau_h(S_h, X_h, U_h)$。在因果图中表现为动作-奖励和动作-下一状态之间的双向箭头。
    - 设计动机：标准 MDP 假设行为数据满足"无未观测混淆"条件，但实际中示教者可能利用学习者看不到的信息做决策（如机器人行走中的身体稳定性），导致观测数据中行为、奖励和转移之间存在虚假关联。

2. **Causal Bellman Optimal Equation（因果贝尔曼最优方程）**:

    - 功能：从混淆离线数据中上界最优介入策略的状态值
    - 核心思路：对转移和奖励分别应用部分辨识 bounds：$\mathcal{T}_h(s,x,s') \leq \tilde{T}_h(s,x,s') P_h(x|s) + P_h(\neg x|s)$，$\mathcal{R}_h(s,x) \leq \tilde{R}_h(s,x) P_h(x|s) + b \cdot P_h(\neg x|s)$。将这些 bounds 代入标准贝尔曼最优方程，得到递推上界：$\bar{V}_h(s) = \max_x [P_h(x|s)(\tilde{R}_h + \mathbb{E}_{\tilde{T}}[\bar{V}_{h+1}]) + P_h(\neg x|s)(b + \max_{s'} \bar{V}_{h+1}(s'))]$。其中 $P_h(\neg x|s)$ 项的含义是：对于行为数据中未选择的动作，乐观地假设它会获得最大可能回报 $b$ 并转移到最优下一状态。
    - 设计动机：标准值迭代在混淆数据上会给出错误估计（如 Walking Robot 例子中，能力型示教者的数据导致稳定/不稳定状态值相同，而差劲示教者甚至惩罚稳定状态）。因果上界方法回避了不可辨识问题，仅需上界成立即可。当有多个数据集时，取每个状态的最小上界（Corollary 3.2）得到最紧估计。

3. **Q-UCB Shaping 算法**:

    - 功能：利用因果上界势函数的在线 model-free 学习
    - 核心思路：在 Q-UCB 基础上做三处关键修改：(1) Q 值零初始化（而非 $H$ 初始化）；(2) UCB bonus 项依赖势函数最大值 $\phi_m = \max_s \phi(s)$：$b_t = c\sqrt{H\phi_m^2\iota/t}$；(3) 学习更新使用整形后的奖励 $r'_h = y_h - \phi_h(s_h) + \phi_{h+1}(s_{h+1})$，并用 $\min(\max(Q, 0), \phi_m)$ 代替原始的 $[0, H]$ 裁剪。
    - 设计动机：保守乐观条件（$V^*(s) \leq \phi(s) \leq H$）使得零初始化成为可能——势函数本身提供了足够的探索信号，无需传统的乐观初始化。这也使得裁剪范围从 $[0,H]$ 缩小到 $[0, \phi_m]$，减少了过度探索。

### 损失函数 / 训练策略

离线阶段使用改进的值迭代（Algo. 2），从 $h=H$ 逆向更新，跳过未访问状态-动作对，最后跨数据集取最小值。在线阶段使用自适应学习率 $\alpha_t = (H+1)/(H+t)$ 的 Q-learning 更新。

## 实验关键数据

### 主实验

在四个 Windy MiniGrid 环境中对比，风向作为未观测混淆因子：

| 环境 | Q-UCB (No Shaping) | Shaping+Min Value | Shaping+Avg Value | **Shaping+Causal Bound (Ours)** |
|------|--------|--------|--------|--------|
| Empty World | 发散 | 收敛 | 收敛 | **收敛** (与 baseline 持平) |
| LavaCross Easy | 发散 | 缓慢/不稳 | 缓慢/不稳 | **最快收敛 + 最低 regret** |
| LavaCross Hard | 发散 | 误导 | 误导 | **唯一正确收敛** |
| LavaCross Maze | 发散 | 误导/陷阱 | 误导/陷阱 | **唯一找到正确策略** |

### 消融实验

| 实验 | 结果 |
|------|------|
| 最优策略一致性检验 | Ours 在所有环境 100% 找到最优介入策略 |
| vs BCQ (深度 offline RL) | BCQ 在混淆数据上最优率仅 14.9%~40.1%，远低于 Ours |
| 不同行为策略质量 | 能力好/差/随机的三种示教者数据融合后，仅 causal bound 正确整合信息 |
| Walking Robot 例子 | 因果上界 $\phi(L=0,F=0)=9.0, \phi(L=0,F=1)=9.5$ 正确上界了真实最优值 $V^*(L=0,F=0)=5.0, V^*(L=0,F=1)=5.5$，且保持了"稳定优于不稳定"的序关系 |

### 关键发现

- 使用混淆值函数做整形不仅无法加速，反而会误导智能体——在 LavaCross Hard 中，混淆值甚至引导智能体走入岩浆区
- 因果上界方法的核心优势在于：即使无法精确估计值函数，上界关系足以提供正确的势函数序关系
- Regret 分析揭示：对于伪次优状态-动作对 $\text{Sub}_\Delta$，regret 从 $O(H^6/\Delta)$ 改进到 $O(H^5/\Delta)$，与 SOTA Q-learning 变体（Xu et al. 2021）匹配

## 亮点与洞察

- 因果推理 × 奖励整形 的巧妙结合：将不可辨识问题转化为"只需上界"的弱要求，完美匹配 PBRS 对势函数的需求——势函数不需要精确等于最优值，只需保持正确的序关系
- 保守乐观条件 (Conservative Optimism) 是连接因果上界与 regret 改进的关键桥梁——上界性质保证 Q 值非负，从而允许零初始化替代传统的 $H$ 初始化
- Walking Robot 例子极具教学价值：直观展示了混淆偏差如何颠覆标准 off-policy 估计

## 局限性

- 仅在表格型 (tabular) 设定下证明理论保证，未扩展到函数逼近场景
- 实验环境规模较小（MiniGrid），未在高维/连续动作空间验证
- 因果上界的紧致性依赖于 $P(\neg x|s)$ 的大小——当行为策略几乎总选某个动作时上界紧致，但当行为策略接近均匀时上界可能过于宽松
- 假设奖励有已知上界 $b$，这在某些场景中可能不现实

## 相关工作与启发

- **vs Gupta et al. (2022, NeurIPS)**：他们证明了 PBRS 在 model-based 学习器中的加速保证，但假设势函数已知。本文首次给出 model-free 学习器的 gap-dependent regret bound，且势函数从数据自动学习
- **vs Zhang & Bareinboim (2024)**：他们研究混淆 MDP 中的 off-policy evaluation bounds，本文将类似的因果 bounds 应用到完全不同的下游任务（奖励整形 + 在线学习）
- **vs Ball et al. (2023) / Song et al. (2023)**：这些 hybrid RL 工作用离线数据预热在线学习，但假设 NUC 条件成立。本文首次处理离线数据含混淆偏差的情况

## 评分

| 维度 | 分数 | 理由 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 因果推理 × PBRS 的新颖交叉，保守乐观条件的提出极具洞察力 |
| 技术深度 | ⭐⭐⭐⭐⭐ | 因果贝尔曼方程推导严密，regret 分析完整，收敛证明（定常 CMDP 的不动点唯一性）精巧 |
| 实验充分度 | ⭐⭐⭐ | 环境规模偏小，缺少连续控制和大规模环境，但 Walking Robot 分析很细致 |
| 写作质量 | ⭐⭐⭐⭐ | 动机清晰，例子贯穿全文，proof sketch 简洁 |
| 实用性 | ⭐⭐⭐ | 目前限于 tabular 设定，实用性有限，但为混淆数据利用开辟新方向 |

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Action-Dependent Optimality-Preserving Reward Shaping (ADOPS)](action-dependent_optimality-preserving_reward_shaping.md)
- [\[ICML 2025\] Heterogeneous Data Game: Characterizing the Model Competition Across Multiple Data Sources](heterogeneous_data_game_characterizing_the_model_competition_across_multiple_dat.md)
- [\[ICLR 2026\] MVR: Multi-view Video Reward Shaping for Reinforcement Learning](../../ICLR2026/reinforcement_learning/mvr_multi-view_video_reward_shaping_for_reinforcement_learning.md)
- [\[ICML 2025\] The Impact of On-Policy Parallelized Data Collection on Deep Reinforcement Learning Networks](the_impact_of_on-policy_parallelized_data_collection_on_deep_reinforcement_learn.md)
- [\[ICML 2025\] Robust Offline Reinforcement Learning with Linearly Structured f-Divergence Regularization](robust_offline_reinforcement_learning_with_linearly_structured_f-divergence_regu.md)

<!-- RELATED:END -->
