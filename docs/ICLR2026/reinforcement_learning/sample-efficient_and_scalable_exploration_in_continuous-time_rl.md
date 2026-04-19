---
title: >-
  [论文解读] Sample-efficient and Scalable Exploration in Continuous-Time RL
description: >-
   提出 COMBRL 算法，通过最大化外在奖励与模型认知不确定性的加权和，在连续时间模型基 RL 中实现可扩展且样本高效的探索，并具有次线性后悔理论保证。
tags:

---

# Sample-efficient and Scalable Exploration in Continuous-Time RL

## 元信息
- **会议**: ICLR 2026
- **arXiv**: [2510.24482](https://arxiv.org/abs/2510.24482)
- **代码**: [https://go.klem.nz/combrl](https://go.klem.nz/combrl)
- **领域**: reinforcement_learning
- **关键词**: continuous-time RL, model-based RL, optimistic exploration, epistemic uncertainty, Gaussian processes, Bayesian neural networks

## 一句话总结
提出 COMBRL 算法，通过最大化外在奖励与模型认知不确定性的加权和，在连续时间模型基 RL 中实现可扩展且样本高效的探索，并具有次线性后悔理论保证。

## 研究背景与动机
- 大多数 RL 算法假设离散时间动态，但真实世界控制系统（机器人、生物过程）天然由 ODE 描述。离散化可能遮蔽关键时间行为并限制控制灵活性。
- 先前连续时间 MBRL 方法（如 OCORL）通过联合优化策略和可信动力学来实现乐观探索，但计算代价很高，需对 plausible dynamics 集合做耦合优化，输入维度从 $d_u$ 升至 $d_u + d_x$，无法扩展到高维系统。
- 先前方法依赖外在奖励信号，无法处理无监督 RL / 系统辨识等场景。
- 核心问题：如何在连续时间 ODE 框架下设计既可扩展、样本高效又具理论保证的探索机制？

## 方法详解

### 整体框架
COMBRL（Continuous-time Optimistic Model-Based RL）在 episode 式连续时间设置中交替进行两步：
1. **模型学习**：用概率模型（GP 或 BNN）学习 ODE $\bm{f}^*$，获得均值预测 $\bm{\mu}_n(\bm{z})$ 和认知不确定性 $\bm{\sigma}_n(\bm{z})$
2. **策略选择**：最大化奖励+不确定性目标

### 关键设计：乐观规划目标
在每个 episode $n$，策略 $\bm{\pi}_n$ 通过最大化以下目标选出：

$$\bm{\pi}_n = \arg\max_{\bm{\pi} \in \Pi} \int_0^T \frac{r(\bm{x}'(s), \bm{u}(s)) + \lambda_n \cdot \|\bm{\sigma}_{n-1}(\bm{x}'(s), \bm{u}(s))\|}{1 + \lambda_n} ds$$

其中 $\lambda_n$ 控制外在奖励与认知不确定性之间的权衡：
- $\lambda_n = 0$：贪心策略（纯利用）
- $0 < \lambda_n < \infty$：平衡探索与利用
- $\lambda_n \to \infty$：纯无监督探索

### $\lambda_n$ 的选择策略
- **静态**：固定超参，通过网格搜索调优
- **退火调度**：$\lambda_n = \lambda_0 \cdot (1 - n/N)$，前期多探索后期多利用
- **自动调优**：基于互信息增益自适应选择

### 与先前方法的关键区别
COMBRL 无需对可信动力学集合做联合优化，只需选取 $\mathcal{M}_{n-1} \cap \mathcal{F}$ 中任意模型（实践中用均值模型 $\bm{\mu}_n$），通过内在奖励实现乐观探索。这避免了 OCORL 的重参数化技巧导致的维度膨胀。

### 测量选择策略（MSS）
连续时间 RL 中，agent 还需决定何时观测和控制系统。MSS $S = (S_n)_{n \geq 1}$ 指定每个 episode 中的测量时间点，影响数据收集质量和后悔界。

### 理论保证
**定理 1（后悔界）**：在 Lipschitz 连续性、亚高斯噪声、well-calibrated 模型假设下，COMBRL 的累积后悔满足：

$$R_N \leq \mathcal{O}\left(\sqrt{\mathcal{I}_N^3(\bm{f}^*, S) \cdot N}\right)$$

其中 $\mathcal{I}_N$ 为模型复杂度，对 RBF 核和等距 MSS，$\mathcal{I}_N$ 以 $\text{polylog}(N)$ 增长，保证次线性后悔。

**定理 2（无监督样本复杂度）**：纯内在探索（$\lambda_n \to \infty$）下，最大认知不确定性以 $\mathcal{O}(\sqrt{\mathcal{I}_N^3 / N})$ 速率衰减。

## 实验关键数据

### 主实验：GP 动力学下的学习效果

| 环境 | 方法 | 渐近性能 | 计算时间比 |
|------|------|----------|-----------|
| Pendulum | Mean (λ=0) | 次优 | 1× |
| Pendulum | PETS | 中等 | ~1× |
| Pendulum | OCORL | 最优级 | ~3× |
| Pendulum | **COMBRL** | **最优级** | **~1×** |
| MountainCar | Mean (λ=0) | 次优 | 1× |
| MountainCar | **COMBRL** | **最优** | ~1× |

> COMBRL 在性能上匹配或超越 OCORL，同时计算成本仅为其约 1/3。

### 消融实验：内在奖励的效果

| 环境 | Mean (λ=0) | PETS | COMBRL (auto λ) | 性能提升 |
|------|-----------|------|-----------------|---------|
| Reacher (easy) | ~基线 | 中等 | 最优 | 显著 |
| Finger (spin) | ~基线 | 中等 | 最优 | 显著 |
| Cartpole (balance) | ~基线 | 接近 | 最优 | 中等 |
| Hopper (stand) | ~基线 | 中等 | 最优 | 显著 |

> COMBRL 在稀疏奖励或欠驱动任务中获得最大性能增益，在高维域中也有一致提升。自动调优 $\lambda_n$ 有效。

### 关键发现
1. COMBRL 在所有测试环境中一致优于 greedy baseline 和 PETS
2. 与 OCORL 性能相当，但计算开销仅约 1/3
3. 无监督学到的模型可迁移到未见下游任务
4. 自动 $\lambda_n$ 调优与最佳手调超参性能接近

## 亮点与洞察
- **统一框架**：单一标量 $\lambda_n$ 优雅地统一了有监督和无监督 RL 设置
- **可扩展性**：避免了对可信动力学集合的优化，可用 BNN 等神经网络模型
- **理论完备**：同时提供有监督后悔界和无监督样本复杂度界
- **MSS 的显式依赖**：首次明确了测量策略对连续时间 RL 性能的影响

## 局限性
- 理论分析依赖 RKHS 平滑性假设和 well-calibrated 模型假设，实际中 BNN 可能不完全满足
- 目前实验仅在中等维度任务验证（最高到 DMC 环境），超高维（如像素输入）的效果需验证
- $\lambda_n$ 的最优选择策略仍需进一步探索，自动调优方法的理论保证有限

## 相关工作
- **连续时间 MBRL**: OCORL (Treven et al., 2023) 提供理论保证但不可扩展；Yildiz et al. (2021) 贪心方法无探索
- **内在动机/无监督 RL**: Sekar et al. (2020), Pathak et al. (2019), Sukhija et al. (2023) 均为离散时间
- **离散时间对应**: Sukhija et al. (2025b) 研究离散时间版本，COMBRL 关注连续时间的不同理论和实验要求

## 评分
- 新颖性: ⭐⭐⭐⭐ — 将奖励+不确定性乐观探索统一到连续时间设置，同时处理有/无监督场景
- 理论深度: ⭐⭐⭐⭐ — 次线性后悔和样本复杂度双重保证
- 实验充分性: ⭐⭐⭐⭐ — 多环境对比、消融、自动调优验证
- 实用价值: ⭐⭐⭐⭐ — 计算高效，适用于连续时间物理控制系统

<!-- RELATED:START -->

## 相关论文

- [WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)
- [Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning](continuous-time_value_iteration_for_multi-agent_reinforcement_learning.md)
- [Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form](safe_continuous-time_multi-agent_reinforcement_learning_via_epigraph_form.md)
- [Spectral Bellman Method: Unifying Representation and Exploration in RL](spectral_bellman_method_unifying_representation_and_exploration_in_rl.md)
- [Ensemble++: Scalable Exploration via Ensemble](../../NeurIPS2025/reinforcement_learning/scalable_exploration_via_ensemble.md)

<!-- RELATED:END -->
