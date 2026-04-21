---
title: >-
  [论文解读] Distributionally Robust Online Markov Game with Linear Function Approximation
description: >-
  [AAAI 2026][人体理解][Markov Game] 本文研究具有线性函数近似的在线分布鲁棒马尔可夫博弈，首次识别了该设定下的学习困难性，并提出 DR-CCE-LSI 算法，在特定特征映射条件下实现了关于特征维度 $d$ 的极小极大最优样本复杂度。
tags:
  - AAAI 2026
  - 人体理解
  - Markov Game
  - Distributional Robustness
  - Linear Function Approximation
  - Coarse Correlated Equilibrium
  - Sample Complexity
---

# Distributionally Robust Online Markov Game with Linear Function Approximation

**会议**: AAAI 2026  
**arXiv**: [2511.07831](https://arxiv.org/abs/2511.07831)  
**代码**: 无  
**领域**: 多智能体强化学习/鲁棒博弈  
**关键词**: Markov Game, Distributional Robustness, Linear Function Approximation, Coarse Correlated Equilibrium, Sample Complexity  

## 一句话总结

本文研究具有线性函数近似的在线分布鲁棒马尔可夫博弈，首次识别了该设定下的学习困难性，并提出 DR-CCE-LSI 算法，在特定特征映射条件下实现了关于特征维度 $d$ 的极小极大最优样本复杂度。

## 研究背景与动机

### 领域现状

多智能体强化学习（MARL）在博弈论框架下被广泛研究，马尔可夫博弈（Markov Game）是核心模型。近年来，线性函数近似被引入以处理大状态空间问题。在单智能体鲁棒 RL 和表格型鲁棒博弈方面已有一些进展。

### 现有痛点

**sim-to-real gap 在 MARL 中更严重**：均衡解对环境微小扰动高度敏感

**在线设定研究不足**：现有鲁棒马尔可夫博弈算法主要面向生成模型或离线设定

**线性近似+鲁棒性尚未结合**：在线鲁棒一般和博弈的线性函数近似设定完全未被探索

### 核心矛盾

**支撑偏移问题**（support shift）：在线收集的样本可能无法覆盖最坏情况环境下的所有轨迹，导致算法在未观测状态上表现不优于随机猜测。这是鲁棒在线学习的根本障碍。

### 本文目标

在具有线性函数近似、$d$-矩形不确定性集的一般和博弈中，设计可证明样本高效的算法来找到近似鲁棒粗相关均衡（CCE）。

### 切入角度

1. 首先通过构造反例证明在线学习的不可能性（下界为 $\Omega(\sigma \cdot HK)$）
2. 引入**消失最小值假设**来规避支撑偏移问题
3. 设计具有智能体特定奖励项的最小二乘值迭代算法

### 核心 idea

**在采用消失最小值假设后，利用 $d$-矩形结构保持线性性，通过分坐标岭回归和智能体特定探索奖励实现样本高效的鲁棒均衡学习**。

## 方法详解

### 整体框架

DR-CCE-LSI（Distributionally Robust CCE Least Square Iteration）算法在每个 episode $k$ 中：
1. 每个玩家 $i$ 利用历史数据通过岭回归估计鲁棒动作值函数 $Q_{i,h}^k$
2. 对估计的 $Q$ 值矩阵求解 $n$ 人矩阵博弈的 CCE（通过 Find-CCE 子程序）
3. 使用 CCE 策略与环境交互，收集新数据
4. 更新特征协方差矩阵 $\Lambda_h^{k+1}$

### 关键设计一：$d$-矩形不确定性集

**功能**：在线性 MDP 结构下定义合适的鲁棒不确定性集。

**核心思路**：假设转移核 $P_h(s'|s, \boldsymbol{a}) = \langle \phi_{s\boldsymbol{a}}, \mu_h^0(s') \rangle$ 具有线性结构。$d$-矩形不确定性集将扰动施加在 $\mu_h$ 的每个坐标上：
$$\mathcal{U}_{TV}^{\sigma_i}(\mu_h^0) : \bigotimes_{j \in [d]} \{\mu_{h,j} : D_{TV}(\mu_{h,j} \| \mu_{h,j}^0) \leq \sigma_i\}$$

**设计动机**：$d$-矩形结构确保鲁棒动作值函数保持线性形式 $Q_{i,h}^{\pi,\sigma}(s,\boldsymbol{a}) = \langle \phi_{s\boldsymbol{a}}, w_{i,h} \rangle + \text{bonus}$，避免了一般函数近似中需要的完备性假设。

### 关键设计二：消失最小值假设与等价优化

**功能**：通过假设 $\min_s V_{i,h}^{\pi,\sigma}(s) = 0$ 来避免支撑偏移问题。

**核心思路**：在该假设下，鲁棒优化问题可以化简为：
$$\inf_{P \in \mathcal{U}_{TV}^{\sigma_i}(P_{h,s,\boldsymbol{a}}^0)} \mathbb{E}_P[V] = \sigma_i \mathbb{E}_{\tilde{P}_{h,s,\boldsymbol{a}}}[V]$$

这保证了最坏情况转移核的支撑不超出名义核的支撑范围（命题 4.3 的第 (12) 式）。

**设计动机**：该假设实际上等价于在马尔可夫博弈中添加一个吸收失败状态 $s_f$（奖励为 0），这在很多实际场景中是自然的（如游戏中玩家可能在每一步失败）。

### 关键设计三：智能体特定奖励项

**功能**：设计探索奖励引导策略充分探索。

**核心思路**：每个玩家 $i$ 的奖励项为：
$$\Gamma_{h,k}^i(s,\boldsymbol{a}) = \beta_i \sum_{j=1}^d \sqrt{\phi_j(s,\boldsymbol{a}) \mathbf{1}_j^T (\Lambda_h^k)^{-1} \mathbf{1}_j \phi_j(s,\boldsymbol{a})}$$

其中 $\beta_i = \min\{H, 1/\sigma_i\} \sqrt{c_\beta n d \log(ndHK/\delta)}$。

**设计动机**：不同于非鲁棒设定的单一 UCB 奖励，鲁棒设定需要 $d$ 个分坐标的置信上界。每个坐标对应一个独立的岭回归任务，目标是 $[V_{i,h+1}^k(s_{h+1}^\tau)]_{\alpha_j}$。同时，$\beta_i$ 包含 $\min\{H, 1/\sigma_i\}$ 反映了每个玩家各自的鲁棒性偏好。

### 关键设计四：Find-CCE 子程序

**功能**：解决 CCE 对 $Q$ 值矩阵不稳定（非 Lipschitz 连续）的问题。

**核心思路**：先在 $Q$ 值函数类的 $\epsilon$-覆盖集中找到最近的代理博弈矩阵 $\tilde{Q}$，再对 $\tilde{Q}$ 求精确 CCE。这确保了对于覆盖集中的任意两个近邻 $Q$ 值，使用同一个 CCE 时，值函数的偏差可以被控制在 $2\epsilon$ 以内。

**设计动机**：直接对 $Q$ 函数求 CCE 会导致值函数类的覆盖数爆炸（引理 4.4 的反例说明微小 $Q$ 值扰动可导致 CCE 值差异为 1）。通过"先离散化再求均衡"的策略解决这一技术困难。

### 损失函数

遗憾定义为所有玩家中最大单方面偏离收益的总和：
$$\text{Regret}(K) = \max_{i \in [n]} \sum_{k=1}^K [V_{i,1}^{\star, \pi_{-i}^k, \sigma}(s_1^k) - V_{i,1}^{\pi^k, \sigma}(s_1^k)]$$

## 实验关键数据

### 主实验

在 5 状态、2 玩家、$H=3$ 的模拟博弈中（结构如论文 Figure 1 所示），比较 DR-CCE-LSI 与非鲁棒的 NQOVI：

| 不确定性水平 $\rho$ | DR-CCE-LSI（本文） | NQOVI |
|---------------------|-------------------|-------|
| 0.0（无偏移） | 略低（牺牲最优性换鲁棒性） | 更优 |
| 0.1 | 接近 | 开始下降 |
| 0.2+ | **显著更优** | 性能急剧退化 |

### 理论对比

| 设定 | 方法 | 上界 | 下界 |
|------|------|------|------|
| 非鲁棒单智能体 | he2023 | $\sqrt{d^2H^3K}$ | $\sqrt{d^2H^3K}$ |
| 非鲁棒多玩家 | cisneros2023 | $\sqrt{d^3H^5K}$ | $\sqrt{d^2H^3K}$ |
| **鲁棒多玩家（本文）** | DR-CCE-LSI | $dH\min\{H,1/\sigma\}\sqrt{K}$ | $dH^{1/2}\min\{H,1/\sigma\}\sqrt{K}$ |

### 关键发现

1. 本文上界在 $d$ 和 $K$ 上与单智能体鲁棒设定匹配，达到了**特征维度 $d$ 上的极小极大最优**
2. 遗憾界中 $\min\{H, 1/\min\{\sigma_i\}\}$ 体现了鲁棒性代价的刻画
3. 当环境扰动敏感度高时，DR-CCE-LSI 的优势更加明显
4. 不可能性定理（定理 4.1）证明无额外假设时遗憾为 $\Omega(\sigma \cdot HK)$（线性增长，不可学习）

## 亮点与洞察

1. **首个理论结果**：在线鲁棒线性马尔可夫博弈的首个样本高效算法
2. **不可能性+可行性的完整图景**：先证明下界不可能，再通过恰当假设（消失最小值）使问题可解
3. **有趣的"风险偏好一致性"观察**：玩家间不一致的鲁棒性水平（$\sigma_i$ 差异大）会导致样本效率下降，暗示合作学习中需要共同的风险意识
4. **技术贡献**：Find-CCE 子程序解决了 CCE 对 $Q$ 值不稳定的根本技术困难

## 局限与展望

1. 上下界在 $H$ 上仍有 $H^{1/2}$ 的差距，可能通过方差加权岭回归改进
2. 消失最小值假设虽然合理，但限制了适用范围
3. 仅考虑 TV 散度的 $d$-矩形不确定性集，$\chi^2$ 和 KL 散度的扩展未探索
4. 特征映射需满足非退化和正定性条件（推论 5.3），对特征设计有额外要求
5. 实验仅在小规模模拟环境中验证，缺少更复杂环境的实证评估
6. 去中心化学习设定下的鲁棒博弈仍为开放问题

## 相关工作与启发

1. **Liu et al. (2024)**：单智能体在线鲁棒线性 MDP，上界 $O(dH\min\{1/\sigma, H\}\sqrt{K})$，本文将其推广到多玩家
2. **Cisneros et al. (2023)**：非鲁棒在线线性博弈的 NQOVI 算法，上界 $O(\sqrt{d^3H^5K})$
3. **Jiao et al. (2024)**：生成模型设定下鲁棒博弈的极小极大最优算法
4. **启发**：鲁棒性在博弈设定中的额外挑战（均衡不稳定性+支撑偏移）提供了丰富的理论研究空间

## 评分

⭐⭐⭐⭐ (4/5)

**优势**：问题新颖，理论分析完整（不可能性+可行性+极小极大性），技术贡献扎实。

**不足**：$H$ 依赖的差距尚未弥合，实验过于简单，消失最小值假设的实际验证不足。

<!-- RELATED:START -->

## 相关论文

- [Function Spaces Without Kernels: Learning Compact Hilbert Space Representations](../../ICLR2026/human_understanding/function_spaces_without_kernels_learning_compact_hilbert_space_representations.md)
- [HiLoMix: Robust High- and Low-Frequency Graph Learning Framework for Mixing Address Association](hilomix_robust_high-_and_low-frequency_graph_learning_framework_for_mixing_addre.md)
- [An Efficient, Provably Optimal Algorithm for the 0-1 Loss Linear Classification Problem](../../ICLR2026/human_understanding/an_efficient_provably_optimal_algorithm_for_the_0-1_loss_linear_classification_p.md)
- [Mixing Expert Knowledge: Bring Human Thoughts Back to the Game of Go](../../NeurIPS2025/human_understanding/mixing_expert_knowledge_bring_human_thoughts_back_to_the_game_of_go.md)
- [REA-RL: Reflection-Aware Online Reinforcement Learning for Efficient Reasoning](../../ICLR2026/human_understanding/rea-rl_reflection-aware_online_reinforcement_learning_for_efficient_reasoning.md)

<!-- RELATED:END -->
