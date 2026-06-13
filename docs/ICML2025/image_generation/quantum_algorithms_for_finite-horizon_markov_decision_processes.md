---
title: >-
  [论文解读] Quantum Algorithms for Finite-horizon Markov Decision Processes
description: >-
  [ICML2025][图像生成][量子算法] 提出四种量子值迭代算法（QVI-1/2/3/4），在精确动力学和生成模型两种设定下，对有限时域时变MDP实现了状态空间 $S$、动作空间 $A$、误差 $\epsilon$ 和时域 $H$ 多维度的量子加速，并证明了渐近最优的量子下界。 领域现状 领域现状：马尔可夫决策过程（MD…
tags:
  - "ICML2025"
  - "图像生成"
  - "量子算法"
  - "有限时域MDP"
  - "量子值迭代"
  - "量子加速"
  - "样本复杂度下界"
---

# Quantum Algorithms for Finite-horizon Markov Decision Processes

**会议**: ICML2025  
**arXiv**: [2508.05712](https://arxiv.org/abs/2508.05712)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 量子算法, 有限时域MDP, 量子值迭代, 量子加速, 样本复杂度下界

## 一句话总结

提出四种量子值迭代算法（QVI-1/2/3/4），在精确动力学和生成模型两种设定下，对有限时域时变MDP实现了状态空间 $S$、动作空间 $A$、误差 $\epsilon$ 和时域 $H$ 多维度的量子加速，并证明了渐近最优的量子下界。

## 研究背景与动机

### 领域现状

**领域现状**：马尔可夫决策过程（MDP）是强化学习的核心数学框架，但当状态空间或动作空间规模巨大时面临严重的"维度灾难"。量子计算在无结构搜索（Grover）、优化等任务上已展现出相对经典算法的显著加速。

**现有工作的不足**：

### 现有痛点

**现有痛点**：此前量子RL算法（Wang et al. 2021; Cherrat et al. 2023）仅针对**无限时域、时不变**MDP设计，无法处理值函数随时间变化的有限时域场景

### 核心矛盾

**核心矛盾**：Wiedemann et al. (2022) 虽尝试有限时域，但样本复杂度关于 $S$ 呈指数增长

### 解决思路

**解决思路**：Naguleswaran et al. (2006) 的量子行走方法仅适用于确定性最短路径问题，无法推广到一般MDP

**本文目标**：在有限时域、时变MDP的**精确动力学设定**和**生成模型设定**下，设计量子算法以突破经典复杂度下界。

## 方法详解

### 问题建模

有限时域时变MDP定义为五元组 $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \{P_h\}, \{r_h\}, H)$，其中 $S = |\mathcal{S}|$，$A = |\mathcal{A}|$，$H$ 为时间步长。值函数满足 Bellman 递推：

$$V_h^*(s) = \max_{a \in \mathcal{A}} \left\{ r_h(s,a) + \sum_{s'} P_h(s'|s,a) V_{h+1}^*(s') \right\}$$

### 设定一：精确动力学（Section 3）

**QVI-1 — 动作空间 $A$ 上的二次加速**

- 核心思想：在经典值迭代的 Bellman 递推中，取最大值需遍历所有 $A$ 个动作。QVI-1 用**量子最大搜索**（Dürr & Høyer 1999）替换此步骤，将 $O(A)$ 降至 $O(\sqrt{A})$
- 量子查询复杂度：$\tilde{O}(S^2 \sqrt{A} H)$（经典为 $O(S^2 A H)$）
- 输出精确最优策略 $\pi^* $ 和 $V_0^*$，成功概率 $\geq 1 - \delta$

**QVI-2 — 状态空间 $S$ 上的额外加速**

- 动机：许多问题（如围棋）状态空间远大于动作空间，$O(S^2)$ 仍是瓶颈
- 核心创新：提出**QMEBO（Quantum Mean Estimation with Binary Oracles）**子程序，用 $O(\sqrt{S}/\epsilon)$ 次查询估计 $P_{h|s,a}^T V$，对比经典的 $O(S)$
- QMEBO的关键步骤：将二进制编码的概率分布 $B_p$ 转换为振幅编码的酉算子 $\hat{U}_p$，制备量子态后用振幅估计提取均值
- 量子查询复杂度：$\tilde{O}\left(\frac{S^{1.5}\sqrt{A} H^3}{\epsilon}\right)$
- 输出 $\epsilon$-最优策略，满足 $V_h^* - \epsilon \leq \hat{V}_h \leq V_h^{\hat{\pi}} \leq V_h^*$

### 设定二：生成模型（Section 4）

当环境动力学未知但可通过生成模型采样时：

**QVI-3 — 基于 Hoeffding 型量子均值估计**

- 将经典 RandomizedFiniteHorizonVI（Sidford et al. 2023）中的经典采样替换为量子均值估计 QME1，同时用量子最大搜索加速动作空间遍历
- 量子样本复杂度：$\tilde{O}\left(\frac{S\sqrt{A} H^3}{\epsilon}\right)$

**QVI-4 — 基于 Bernstein 型量子均值估计**

- 利用 QME2 的方差自适应特性和单调性技巧实现更紧的估计
- 量子样本复杂度：$\tilde{O}\left(\frac{SA H^{2.5}}{\epsilon}\right)$
- 虽然在 $A$ 上不如 QVI-3，但在 $H$ 上有更好的依赖关系

## 理论结果（关键数据）

### 精确动力学设定复杂度对比

| 目标 | 经典上界 | 经典下界 | 量子上界（本文） |
|------|---------|---------|----------------|
| 精确 $\pi^*, V_0^*$ | $O(S^2 A H)$ | $\Omega(S^2 A)$ | $\tilde{O}(S^2 \sqrt{A} H)$ — **QVI-1** |
| $\epsilon$-最优 $\pi^*, \{V_h^*\}$ | $O(S^2 A H)$ | $\Omega(S^2 A)$ | $\tilde{O}(S^{1.5}\sqrt{A}H^3/\epsilon)$ — **QVI-2** |

### 生成模型设定复杂度对比

| 目标 | 经典上界 | 经典下界 | 量子上界（本文） | 量子下界（本文） |
|------|---------|---------|----------------|----------------|
| $\epsilon$-最优 $\{Q_h^*\}$ | $O(SAH^4/\epsilon^2)$ | $\Omega(SAH^3/\epsilon^2)$ | $\tilde{O}(SAH^{2.5}/\epsilon)$ — **QVI-4** | $\Omega(SAH^{1.5}/\epsilon)$ |
| $\epsilon$-最优 $\pi^*, \{V_h^*\}$ | $O(SAH^4/\epsilon^2)$ | $\Omega(SAH^3/\epsilon^2)$ | $\tilde{O}(S\sqrt{A}H^3/\epsilon)$ — **QVI-3** | $\Omega(S\sqrt{A}H^{1.5}/\epsilon)$ |

**关键观察**：当 $H$ 为常数时，QVI-3 和 QVI-4 的量子上界与量子下界匹配（至多差对数因子），**证明了渐近最优性**。

### 加速倍数汇总

- **QVI-1**：$A$ 上二次加速（$A \to \sqrt{A}$）
- **QVI-2**：$A$ + $S$ 双重加速（$S^2 \to S^{1.5}$，$A \to \sqrt{A}$）
- **QVI-3**：$A$ 二次 + $\epsilon$ 二次 + $H$ 改进（$\epsilon^2 \to \epsilon$，$A \to \sqrt{A}$）
- **QVI-4**：$\epsilon$ 二次 + $H$ 改进（$\epsilon^2 \to \epsilon$，$H^4 \to H^{2.5}$）

## 亮点与洞察

1. **QMEBO 子程序是核心技术贡献**：首次解决了从二进制编码（binary oracle）进行量子均值估计的问题，弥补了现有 QME 仅支持振幅编码的限制
2. **量子下界的建立**：不仅给出上界算法，还证明了生成模型设定下的量子下界，说明 QVI-3/4 在常数 $H$ 下不可能被本质改进
3. **经典新下界**：作为副产品，推导出经典设定下获取 Q 值的新下界 $\Omega(SAH^3/\epsilon^2)$
4. **单调性技巧的推广**：将 Sidford et al. (2018) 用于无限时域的单调性技巧推广到有限时域+量子设定

## 局限与展望

1. **量子硬件要求高**：算法依赖容错量子计算机和 QRAM，短期内难以实际部署
2. **$H$ 依赖的 gap**：量子上界与下界在 $H$ 上仍有差距（如 QVI-3 的 $H^3$ vs 下界 $H^{1.5}$），closing this gap 是开放问题
3. **仅考虑确定性策略**：未涉及随机策略、连续状态空间或函数逼近等现代 RL 设定
4. **无实验验证**：纯理论工作，缺乏量子模拟器或真实量子硬件上的实验

## 相关工作与启发

- **无限时域量子 MDP**：Wang et al. (2021) 的近似极小极大最优量子算法是最直接的前导工作
- **量子搜索/估计基础**：Grover 搜索、Dürr-Høyer 最大搜索、Montanaro QME 是本文的量子子程序来源
- **经典有限时域 MDP**：Li et al. (2020)、Sidford et al. (2018, 2023) 的经典算法和下界是本文对标的基线

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首个有限时域时变MDP的完整量子算法族+匹配下界
- 实验充分度: ⭐⭐ — 纯理论无实验
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，两大设定×四种算法的组织逻辑性强
- 价值: ⭐⭐⭐⭐ — 量子RL理论的重要推进，但实际影响需等量子硬件成熟

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Flow Matching Neural Processes](../../NeurIPS2025/image_generation/flow_matching_neural_processes.md)
- [\[NeurIPS 2025\] Recurrent Memory for Online Interdomain Gaussian Processes](../../NeurIPS2025/image_generation/recurrent_memory_for_online_interdomain_gaussian_processes.md)
- [\[NeurIPS 2025\] ALE-Bench: A Benchmark for Long-Horizon Objective-Driven Algorithm Engineering](../../NeurIPS2025/image_generation/alebench_a_benchmark_for_longhorizon_objectivedriven_algorit.md)
- [\[CVPR 2025\] Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models](../../CVPR2025/image_generation/finite_difference_flow_optimization_for_rl_post-training_of_text-to-image_models.md)
- [\[NeurIPS 2025\] Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms](../../NeurIPS2025/image_generation/fast_solvers_for_discrete_diffusion_models_theory_and_applications_of_high-order.md)

</div>

<!-- RELATED:END -->
