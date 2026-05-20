---
title: >-
  [论文解读] Sampling from Binary Quadratic Distributions via Stochastic Localization
description: >-
  [ICML2025][随机局部化] 首次将随机局部化 (Stochastic Localization, SL) 框架应用于一般二元二次分布 (BQD) 采样，证明经过足够SL迭代后后验分布几乎处处满足 Poincaré 不等式，从而保证离散 MCMC 采样器多项式时间混合…
tags:
  - "ICML2025"
  - "随机局部化"
  - "二元二次分布"
  - "离散MCMC"
  - "Poincaré不等式"
  - "组合优化"
  - "Gibbs采样"
---

# Sampling from Binary Quadratic Distributions via Stochastic Localization

**会议**: ICML2025  
**arXiv**: [2505.19438](https://arxiv.org/abs/2505.19438)  
**代码**: 待确认  
**领域**: 其他/采样  
**关键词**: 随机局部化, 二元二次分布, 离散MCMC, Poincaré不等式, 组合优化, Gibbs采样

## 一句话总结

首次将随机局部化 (Stochastic Localization, SL) 框架应用于一般二元二次分布 (BQD) 采样，证明经过足够SL迭代后后验分布几乎处处满足 Poincaré 不等式，从而保证离散 MCMC 采样器多项式时间混合，并在 QUBO 组合优化问题上验证了一致的采样效率提升。

## 研究背景与动机

**问题定义**：从二元二次分布 (BQD) 中采样：

$$\nu(x) \propto e^{-\frac{\beta}{2}\langle x, Wx \rangle + \langle x, b \rangle}, \quad x \in \{-1, 1\}^N$$

其中 $\beta$ 是逆温度参数。这类分布在统计物理（Ising 模型、自旋系统）和组合优化（QUBO 问题）中广泛出现。

**核心挑战**：
- 离散状态空间的复杂依赖性使得采样极其困难
- 连续域中 SL 的理论保证（如 log-concavity 对偶性）无法直接迁移到离散设置
- 已有离散 SL 工作仅针对特定模型（如 SK 模型），缺乏通用框架

**研究问题**：SL 能否像在连续设置中那样，通过构造易于采样的后验分布，降低一般 BQD 的采样难度？

## 方法详解

### 框架：SL + 离散 MCMC

核心思路是将困难的直接采样分解为一系列更简单的后验分布采样。基于观测过程：

$$Y_t^\alpha = \alpha(t) X + \sigma B_t, \quad t \in [0, T_{\text{gen}})$$

其中 $X \sim \nu$，$B_t$ 是标准布朗运动。对应的 SDE 为：

$$dY_t^\alpha = \dot{\alpha}(t) u_t^\alpha(Y_t^\alpha) dt + \sigma dB_t$$

**算法流程（Algorithm 1）**：
1. 初始化 $\tilde{Y}_0 \sim \mathcal{N}(0, t_0 \sigma^2 I)$
2. 在每个 SL 迭代 $i$ 中：
    - 用离散 MCMC 采样器从后验 $\mathbb{P}(X|\tilde{Y}_i)$ 中采 $n$ 个样本
    - 估计后验期望 $\tilde{U}_i^\alpha = \frac{1}{n}\sum_{j=1}^n x_j^i$
    - 更新 $\tilde{Y}_{i+1} \sim \mathcal{N}(\tilde{Y}_i + w_i \tilde{U}_i^\alpha, \sigma^2 \delta_i \mathbf{I})$
3. 输出 $\tilde{Y}_T / \alpha(t_T)$

### 关键设计：后验分布结构

对于 BQD，后验分布具有显式形式：

$$q_t^\alpha(x|y) \propto e^{-\frac{\beta}{2}\langle x, Wx \rangle + \langle x, b + \frac{\alpha(t) Y_t}{\sigma^2 t} \rangle}$$

**关键洞察**：由于 $x \in \{-1,1\}^N$，高斯似然中的二次项 $\langle x,x \rangle = 1$ 消失，后验与原分布仅在线性项（外部场 $h_t$）上不同：

$$h_t = b + \frac{\alpha(t) Y_t}{\sigma^2 t}$$

### 核心理论：Poincaré 不等式

**Theorem 3.1**：当 $\frac{\alpha(t)}{\sigma\sqrt{t}} \to +\infty$ 时，外部场 $|h_t| \to \infty$，即后验分布趋近于独立 Bernoulli 乘积分布——极易采样。

**Condition 4.1（强外部场条件）**：要求 $|h| \geq 2\beta \sup_{i \in [N]} \sum_{k \neq i} |W_{ik}|$

在此条件下，对不同 MCMC 采样器建立了 Poincaré 不等式：

| 采样器类型 | 谱隙 $\gamma_{\text{gap}}$ |
|---|---|
| Glauber 动力学 | $1 - \frac{|h|}{e^{3|h|/4} + e^{-3|h|/4}}$ |
| 经典 Metropolis 链 | $1 - 2|h|e^{-|h|}$ |
| 单点梯度 MH | $1 - \frac{|h|}{(e^{|h|/4} + e^{-|h|/4})^2}$ |
| DULA（全站点更新） | $1 - \frac{4|h|N}{(e^{|h|/4} + e^{-|h|/4})^2}$ |

所有情况下，$\gamma_{\text{gap}}$ 随 $|h|$ 增大趋向 1，保证多项式时间混合。

**Corollary 4.4（Chernoff 误差界）**：Monte Carlo 估计器的误差随样本量 $n$ 指数衰减：

$$P_q\left[\left|\frac{1}{n}\sum_{i=1}^n X_i - \mathbb{E}_{\nu_{\beta,h}}[X]\right| \geq \varepsilon\right] \leq C \cdot e^{-\frac{n\varepsilon^2 \gamma_{\text{gap}}}{c}}$$

## 实验关键数据

在三类 QUBO 组合优化问题（MIS, MaxCut, MaxClique）上，用 GWG、PAS、DMALA 三种 MCMC 采样器分别测试原始版本与 SL 增强版本，共 14 个数据集。

| 任务 | 采样器 | 原始 | SL 增强 | 提升 |
|---|---|---|---|---|
| MIS ER-0.05 | PAS | 104.375 | **104.531** | +0.15% |
| MIS ER-0.25 | GWG | 27.813 | **28.000** | +0.67% |
| MaxClique RB | PAS | 87.544% | **87.649%** | +0.12% |
| MaxCut BA-512 | PAS | 100.883% | **100.928%** | +0.045% |
| MaxCut ER-1024 | GWG | 100.098% | **100.101%** | +0.003% |

**关键发现**：
- SL 变体在所有 14 个数据集、所有 3 种采样器上均实现一致提升或持平
- SL-PAS 整体表现最优
- 消融实验显示"指数衰减 MCMC 步数分配 + 均匀时间离散化"策略最优，与理论预测一致

## 亮点与洞察

1. **理论突破**：首次为一般 BQD 上的 SL 框架建立了严格的 Poincaré 不等式保证，不依赖分布的特殊结构
2. **物理直觉优美**：外部场 $|h_t| \to \infty$ 使交互项可忽略，后验退化为独立 Bernoulli——类似物理中强外场压制自旋关联
3. **广泛适用性**：理论覆盖 Glauber 动力学和 MH 两大类离散 MCMC 方法，包括梯度信息增强的变体
4. **即插即用**：SL 是通用包装器，可增强任何现有离散 MCMC 采样器而无需修改其内部结构
5. **理论-实验一致**：指数衰减分配策略的优越性，完美验证了"后期采样更容易"的理论预测

## 局限与展望

1. **仅限二元变量**：当前框架限于 $x \in \{-1,1\}^N$，未推广到多值离散变量
2. **仅限二次分布**：无法处理未知形式分布（如深度能量模型），需高效二阶 Taylor 近似
3. **整体收敛理论缺失**：仅保证每步后验采样的混合，未建立 SL 整体收敛速率理论
4. **实验提升幅度有限**：基线采样器已接近最优，SL 提升通常 <1%，实际价值需权衡额外计算成本
5. **强外场条件的实际可达性**：Condition 4.1 在实践中何时准确满足，依赖于问题结构

## 相关工作与启发

- **SLIPS** (Grenioux et al., 2024)：连续域 SL 采样框架，本文将其理论扩展到离散域
- **GWG/PAS/DMALA**：三种代表性梯度信息离散 MCMC 采样器（Grathwohl 2021, Sun 2021, Zhang 2022）
- **SK模型SL** (El Alaoui et al., 2022)：针对特定模型的 SL 应用，本文推广到一般 BQD
- **DISCS** (Goshvadi et al., 2024)：离散采样基准测试框架

**启发**：SL 的"将困难采样分解为简单子问题"思路与扩散模型的核心思想一致——是否可以用类似理论工具分析离散扩散模型的收敛性？

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次为一般 BQD 上的 SL 建立完整理论保证
- 实验充分度: ⭐⭐⭐⭐ — 14个数据集、3种采样器、详尽消融，但提升幅度较小
- 写作质量: ⭐⭐⭐⭐ — 物理直觉→严格理论→实验验证的逻辑链清晰完整
- 价值: ⭐⭐⭐⭐ — 理论贡献显著，但实际提升有限，框架扩展性待验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Scalable Equilibrium Sampling with Sequential Boltzmann Generators](scalable_equilibrium_sampling_with_sequential_boltzmann_generators.md)
- [\[ICML 2025\] Principled Algorithms for Optimizing Generalized Metrics in Binary Classification](principled_algorithms_for_optimizing_generalized_metrics_in_binary_classificatio.md)
- [\[AAAI 2026\] Provably Data-Driven Projection Method for Quadratic Programming](../../AAAI2026/others/provably_data-driven_projection_method_for_quadratic_programming.md)
- [\[NeurIPS 2025\] Sample-Adaptivity Tradeoff in On-Demand Sampling](../../NeurIPS2025/others/sample-adaptivity_tradeoff_in_on-demand_sampling.md)
- [\[NeurIPS 2025\] Robust Sampling for Active Statistical Inference](../../NeurIPS2025/others/robust_sampling_for_active_statistical_inference.md)

</div>

<!-- RELATED:END -->
