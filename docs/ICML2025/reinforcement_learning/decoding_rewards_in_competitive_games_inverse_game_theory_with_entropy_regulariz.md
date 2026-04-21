---
title: >-
  [论文解读] Decoding Rewards in Competitive Games: Inverse Game Theory with Entropy Regularization
description: >-
  [ICML2025][逆博弈论] 提出基于熵正则化的零和博弈逆问题统一框架，利用 Quantal Response Equilibrium (QRE) 在线性假设下建立奖励函数的可辨识性条件，并给出从观测动作恢复奖励函数的置信集构造算法，附带 $\mathcal{O}(T^{-1/2})$ 收敛速率保证。
tags:
  - ICML2025
  - 逆博弈论
  - 零和博弈
  - 熵正则化
  - Quantal Response Equilibrium
  - Markov博弈
  - 奖励恢复
  - 可辨识性
---

# Decoding Rewards in Competitive Games: Inverse Game Theory with Entropy Regularization

**会议**: ICML2025  
**arXiv**: [2601.12707](https://arxiv.org/abs/2601.12707)  
**代码**: 无  
**领域**: 逆博弈论 / 逆强化学习  
**关键词**: 逆博弈论, 零和博弈, 熵正则化, Quantal Response Equilibrium, Markov博弈, 奖励恢复, 可辨识性

## 一句话总结
提出基于熵正则化的零和博弈逆问题统一框架，利用 Quantal Response Equilibrium (QRE) 在线性假设下建立奖励函数的可辨识性条件，并给出从观测动作恢复奖励函数的置信集构造算法，附带 $\mathcal{O}(T^{-1/2})$ 收敛速率保证。

## 研究背景与动机

**逆强化学习 (IRL)** 旨在从智能体的观测行为推断驱动其决策的奖励函数。经典 IRL 聚焦单智能体场景，而在**竞争性博弈**（如零和博弈）中，智能体策略不仅取决于自身奖励，还依赖对手策略，使得问题更加复杂。

现有方法面临三大核心挑战：

**逆问题病态性**：多个奖励函数可能导致相同的均衡策略，需识别整个可行奖励集而非单一解

**离线数据覆盖不足**：观测策略可能无法遍历整个状态-动作空间，导致奖励恢复不鲁棒

**动态博弈复杂性**：在 Markov 博弈中，策略随时间演变，增加了可辨识性和估计的难度

应用场景涵盖经济市场定价分析、网络安全攻防策略推断、交通物流竞争路由等。

## 方法详解

### 1. 熵正则化零和矩阵博弈

考虑两人零和矩阵博弈 $(\mathcal{A}, \mathcal{B}, Q)$，其中 $|\mathcal{A}|=m$, $|\mathcal{B}|=n$。引入熵正则化后的 minimax 目标为：

$$\max_{\mu} \min_{\nu} \mu^\top Q \nu + \eta^{-1}\mathcal{H}(\mu) - \eta^{-1}\mathcal{H}(\nu)$$

其中 $\eta > 0$ 为正则化参数，$\mathcal{H}(\pi) = -\sum_i \pi_i \log(\pi_i)$ 为 Shannon 熵。该问题的解称为 **Quantal Response Equilibrium (QRE)**，满足 softmax 形式的不动点方程：

$$\mu^*(a) = \frac{e^{\eta Q(a,\cdot)\nu^*}}{\sum_{a' \in \mathcal{A}} e^{\eta Q(a',\cdot)\nu^*}}, \quad \nu^*(b) = \frac{e^{-\eta Q(\cdot,b)^\top \mu^*}}{\sum_{b' \in \mathcal{B}} e^{-\eta Q(\cdot,b')^\top \mu^*}}$$

### 2. 线性参数化下的可辨识性

**线性假设（Assumption 2.1）**：存在特征函数 $\phi: \mathcal{A} \times \mathcal{B} \to \mathbb{R}^d$ 和参数 $\theta^* \in \mathbb{R}^d$，使得 $Q(a,b) = \langle \phi(a,b), \theta^* \rangle$。

将 QRE 不动点方程取对数化简后，得到线性系统：

$$\begin{bmatrix} A(\nu^*) \\ B(\mu^*) \end{bmatrix} \theta = \begin{bmatrix} c(\mu^*) \\ d(\nu^*) \end{bmatrix}$$

**强可辨识性条件（Proposition 2.2）**：$\theta^*$ 唯一可辨识当且仅当上述系数矩阵满秩，即 $\text{rank}\left(\begin{bmatrix} A(\nu^*) \\ B(\mu^*) \end{bmatrix}\right) = d$。

### 3. 两步估计算法

**Step 1**：用频率估计器从 $N$ 个 i.i.d. 样本估计 QRE：$\hat{\mu}(a) = \frac{1}{N}\sum_{k=1}^N \mathbf{1}_{\{a^k=a\}}$

**Step 2**：通过最小二乘法估计参数：$\hat{\theta} = \arg\min_\theta \left\| \begin{bmatrix} A(\hat{\nu}) \\ B(\hat{\mu}) \end{bmatrix} \theta - \begin{bmatrix} c(\hat{\mu}) \\ d(\hat{\nu}) \end{bmatrix} \right\|^2$

**有限样本误差界（Theorem 2.4）**：以概率 $\geq 1-\delta$，

$$\|\hat{Q} - Q\|_F^2 \lesssim \mathcal{O}\left(\frac{m^2 + n^2 + (m+n)\log(1/\delta)}{N}\right)$$

### 4. 部分可辨识与置信集

当秩条件不满足时，构造置信集 $\hat{\Theta}$ 替代点估计，选取合适阈值 $\kappa$ 使得 $\Theta \subseteq \hat{\Theta}$，Hausdorff 距离 $d_H(\Theta, \hat{\Theta}_N) \lesssim \sqrt{\kappa}$，以 $\mathcal{O}(N^{-1/2})$ 速率收敛（Theorem 2.7）。

### 5. 推广至 Markov 博弈

将框架推广至具有状态空间 $\mathcal{S}$、时间步 $H$、折扣因子 $\gamma$ 的两人零和 Markov 博弈。在**线性 MDP 假设**下，算法包含四步：
- 频率估计 QRE → 构造 Q 函数置信集 → 岭回归估计转移核 → 用 Bellman 方程恢复奖励

**样本复杂度（Theorem 3.12）**：以概率 $\geq 1-3\delta$，

$$D(\mathcal{R}, \hat{\mathcal{R}}) \lesssim \frac{1}{\sqrt{T}}\left(\sqrt{S(m+n)\log\frac{HS}{\delta}}\left(\sqrt{S(m+n)} + \log T\right) + \left(\sqrt{Sd} + \sqrt{d\log T}\right)\log(mn)\right)$$

### 6. MLE 替代频率估计

为放松"所有状态均被充分访问"假设，引入线性参数化 QRE（Assumption 3.13），用 MLE 进行策略估计。MLE 的 Hellinger 距离收敛速率为 $\mathcal{O}(1/T)$，最终奖励集误差达 $\mathcal{O}(T^{-1/2})$。

## 实验关键数据

### 矩阵博弈实验

| 设置 | 参数维度 $d$ | 动作空间 $m \times n$ | 可辨识性 | 样本范围 |
|------|-------------|---------------------|---------|---------|
| Setup I | 2 | 4×6 | 强可辨识 | $10^3$–$10^6$ |
| Setup II | 6 | 6×6 | 部分可辨识 | $10^3$–$10^6$ |

- **Setup I**（强可辨识）：$\|\hat{\theta} - \theta^*\|$、$\|\hat{Q} - Q^*\|_F$、$\text{TV}(\hat{\mu},\mu^*) + \text{TV}(\hat{\nu},\nu^*)$ 均以~$\mathcal{O}(N^{-1/2})$ 收敛，与理论一致
- **Setup II**（部分可辨识）：参数和奖励不收敛至真值，但 QRE 误差仍收敛至零

### Markov 博弈实验（$m=n=5$, $S=4$, $H=6$, $\eta=0.5$）

| 样本量 | 奖励误差（均值±95%CI） | QRE 误差（均值±95%CI） |
|-------|----------------------|---------------------|
| 10,000 | 2.46 ± 0.16 | 7.08×10⁻³ ± 4.61×10⁻⁴ |
| 20,000 | 1.90 ± 0.10 | 5.11×10⁻³ ± 3.11×10⁻⁴ |
| 50,000 | 1.56 ± 0.07 | 3.28×10⁻³ ± 1.70×10⁻⁴ |
| 100,000 | 1.44 ± 0.05 | 2.41×10⁻³ ± 1.41×10⁻⁴ |

关键观察：即使奖励函数估计误差较大，恢复的 QRE 仍能高度一致真实 QRE，验证了方法的统计一致性。

## 亮点与洞察

1. **首次为竞争性博弈逆问题建立完整的可辨识性理论**：给出强可辨识的充要条件（秩条件），并处理部分可辨识情形
2. **统一框架覆盖静态与动态**：同一理论体系同时处理矩阵博弈和 Markov 博弈
3. **置信集而非点估计**：当逆问题存在多解时，构造包含整个可行集的置信集，比强加唯一解更合理
4. **MLE 扩展**：放宽了频率估计器要求所有状态被充分覆盖的强假设，使方法更具实用性
5. **$\mathcal{O}(T^{-1/2})$ 最优收敛速率**：所有理论保证均达到经典经验风险极小化的最优统计速率

## 局限与展望

1. **线性参数化假设较强**：现实博弈的奖励/转移函数往往是非线性的，限制了方法的表达能力
2. **仅限零和博弈**：未覆盖一般和博弈（general-sum game）或多人博弈
3. **实验规模有限**：状态空间 $S=4$、动作空间 $m=n=5$，大规模场景未验证
4. **完全信息假设**：未考虑部分可观测博弈（POSG）
5. **离线设定**：未探索与在线学习/探索策略的结合
6. **计算复杂度未讨论**：置信集构造在高维参数空间中可能代价高昂

## 相关工作与启发

- **逆强化学习**：继承 Maximum Entropy IRL (Ziebart et al., 2008) 的熵正则化思路，拓展到博弈论
- **零和 Markov 博弈**：与 Cen et al. (2021, 2022)、Xie et al. (2023) 等正向学习算法互补
- **逆博弈论**：扩展了 Lin et al. (2014)、Yu et al. (2019) 的工作，首次给出完整的样本复杂度分析
- **启发**：QRE 的 softmax 结构是连接熵正则化 RL 与博弈论的关键桥梁，可能启发更广泛的多智能体 IRL 研究

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次系统性地将熵正则化 IRL 的可辨识性理论拓展至竞争性博弈，理论贡献显著
- 实验充分度: ⭐⭐⭐ — 理论验证充分但规模较小，缺乏真实任务实验
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，从矩阵博弈到 Markov 博弈层层递进，理论推导严谨
- 价值: ⭐⭐⭐⭐ — 为竞争性环境下的奖励推断提供了坚实的理论基础

<!-- RELATED:START -->

## 相关论文

- [Learning to Trust Bellman Updates: Selective State-Adaptive Regularization for Offline RL](learning_to_trust_bellman_updates_selective_state-adaptive_regularization_for_of.md)
- [Robust Offline Reinforcement Learning with Linearly Structured f-Divergence Regularization](robust_offline_reinforcement_learning_with_linearly_structured_f-divergence_regu.md)
- [Convergence Theorems for Entropy-Regularized and Distributional Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/convergence_theorems_for_entropy-regularized_and_distributional_reinforcement_le.md)
- [A Theory of Multi-Agent Generative Flow Networks](../../NeurIPS2025/reinforcement_learning/a_theory_of_multi-agent_generative_flow_networks.md)
- [Learning Interestingness in Automated Mathematical Theory Formation](../../NeurIPS2025/reinforcement_learning/learning_interestingness_in_automated_mathematical_theory_formation.md)

<!-- RELATED:END -->
