---
title: >-
  [论文解读] Set Smoothness Unlocks Clarke Hyper-stationarity in Bilevel Optimization
description: >-
  [NeurIPS 2025][优化][双层优化] 本文提出"集合光滑性"(set smoothness)这一新的结构性质，证明它在非凸-PŁ双层优化中自然成立，并据此揭示超目标函数隐藏的弱凸/弱凹结构，首次建立了非光滑超目标函数Clarke稳定点的可计算性保证。
tags:
  - NeurIPS 2025
  - 优化
  - 双层优化
  - 集合光滑性
  - Clarke次梯度
  - 弱凸性
  - 零阶方法
---

# Set Smoothness Unlocks Clarke Hyper-stationarity in Bilevel Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2506.04587](https://arxiv.org/abs/2506.04587)  
**代码**: 暂无  
**领域**: 优化  
**关键词**: 双层优化, 集合光滑性, Clarke次梯度, 弱凸性, 零阶方法

## 一句话总结

本文提出"集合光滑性"(set smoothness)这一新的结构性质，证明它在非凸-PŁ双层优化中自然成立，并据此揭示超目标函数隐藏的弱凸/弱凹结构，首次建立了非光滑超目标函数Clarke稳定点的可计算性保证。

## 研究背景与动机

双层优化(BLO)建模层次化决策问题：领导者(上层)优化自身目标，同时约束跟随者(下层)在给定上层决策下求解自身优化问题。双层优化广泛应用于超参数优化、Stackelberg博弈、强化学习、拦截博弈等领域。

标准做法是将BLO转化为单层问题：定义**超目标函数** $\varphi_o(\mathbf{x}) = \min_{\mathbf{y} \in \mathcal{S}(\mathbf{x})} F(\mathbf{x}, \mathbf{y})$（乐观）或 $\varphi_p(\mathbf{x}) = \max_{\mathbf{y} \in \mathcal{S}(\mathbf{x})} F(\mathbf{x}, \mathbf{y})$（悲观），其中 $\mathcal{S}(\mathbf{x})$ 为下层解集。当下层强凸时，$\mathcal{S}(\mathbf{x})$ 是单点，$\varphi$ 光滑，现有方法可以找到其稳定点。但当下层有多个解时，$\varphi$ 通常**非光滑**，现有方法面临两个困境：

1. 要么需要不切实际的假设（如惩罚模型 $\sigma F + f$ 对 $\mathbf{y}$ 满足PŁ条件）来保证光滑性
2. 要么只能保证找到较弱的Goldstein稳定点，而非Clarke稳定点

**核心问题：在下层多解的一般设定下，能否计算更强的Clarke超稳定点？**

困难链条：下层多解 → 超目标非光滑 → 一般Lipschitz函数的Clarke稳定点不可计算 → 需要发现新的可验证结构条件。

## 方法详解

### 整体框架

技术路线：(1) 引入集合光滑性概念 → (2) 证明它蕴含超目标函数的弱凸/弱凹性 → (3) 证明在非凸-PŁ设定下，下层解映射满足集合光滑性 → (4) 利用弱凸性+零阶方法计算Clarke稳定点。

### 关键设计

1. **集合光滑性 (Definition 3)**: 对于集值映射 $\mathcal{Y}: \mathbb{R}^m \rightrightarrows \mathbb{R}^n$，称其为 $L$-smooth，如果对任意 $\mathbf{x}_1, \mathbf{x}_2$ 和 $\theta \in [0,1]$，对任意中间点处的解 $\mathbf{y} \in \mathcal{Y}(\theta\mathbf{x}_1 + (1-\theta)\mathbf{x}_2)$，存在 $\mathbf{y}_1 \in \mathcal{Y}(\mathbf{x}_1), \mathbf{y}_2 \in \mathcal{Y}(\mathbf{x}_2)$ 使得：

    - 凸组合逼近：$\|\theta\mathbf{y}_1 + (1-\theta)\mathbf{y}_2 - \mathbf{y}\| \leq \frac{L}{2}\theta(1-\theta)\|\mathbf{x}_1 - \mathbf{x}_2\|^2$
    - 一致分支选择：$\|\mathbf{y}_1 - \mathbf{y}_2\|^2 \leq L\|\mathbf{x}_1 - \mathbf{x}_2\|^2$

   条件(4)是实值函数光滑性向集值映射的自然推广——凸组合的近似误差是二阶的。条件(5)防止"跨分支配对"的平凡满足（Example 1给出了反例）。整体来看，集合光滑性等价于 $\mathcal{Y}(\theta\mathbf{x}_1 + (1-\theta)\mathbf{x}_2) \subseteq \theta\mathcal{Y}(\mathbf{x}_1) + (1-\theta)\mathcal{Y}(\mathbf{x}_2) + O(\|\mathbf{x}_1 - \mathbf{x}_2\|^2)\mathbb{B}$。

2. **集合光滑性蕴含弱凸性 (Theorem 1)**: 如果 $\mathcal{Y}$ 是 $L_\mathcal{Y}$-smooth，$g$ 是 $M_g$-Lipschitz且 $L_g$-smooth，则 $\phi(\mathbf{x}) = \max_{\mathbf{y} \in \mathcal{Y}(\mathbf{x})} g(\mathbf{x}, \mathbf{y})$ 是 $\rho$-弱凸的，其中 $\rho = M_g L_\mathcal{Y} + L_g(1 + L_\mathcal{Y})$。这是本文的核心结构性结果——将参数化优化问题的值函数正则性归结为约束集映射的光滑性。

3. **误差界蕴含集合光滑性 (Theorem 2)**: 在标准假设下（$f$ 光滑、二阶导Lipschitz、解集非空闭凸、满足误差界 $\text{dist}(\mathbf{y}, \mathcal{S}(\mathbf{x})) \leq \tau\|\nabla_\mathbf{y} f(\mathbf{x}, \mathbf{y})\|$），下层解映射 $\mathcal{S}$ 是 $L_\mathcal{S}$-smooth的。

   证明的核心技巧是**残差回填**(residual backfilling)：朴素地将 $\mathbf{y}$ 投影到端点纤维 $\mathcal{S}(\mathbf{x}_i)$ 得到 $\bar{\mathbf{y}}_i$，其凸组合 $\bar{\mathbf{y}}^\theta$ 可能与 $\mathbf{y}$ 有一阶误差（因为它们在不同的"分支"上）。修正方法：先将 $\bar{\mathbf{y}}^\theta$ 投影到中间纤维得到 $\hat{\mathbf{y}}$，然后用残差 $\mathbf{y} - \hat{\mathbf{y}}$ 修正端点选择：$\mathbf{y}_i = \Pi_{\mathcal{S}(\mathbf{x}_i)}(\bar{\mathbf{y}}_i + (\mathbf{y} - \hat{\mathbf{y}}))$。这消除了一阶分支不匹配，只留下二阶余项。

4. **超目标弱凸/弱凹性 (Theorem 3)**: 结合Theorem 1和2，在Assumptions 1-2下，乐观超目标 $\varphi_o$ 是 $\rho$-弱凹的，悲观超目标 $\varphi_p$ 是 $\rho$-弱凸的。

### 算法与收敛保证

采用不精确零阶方法(IZOM, Algorithm 1)：通过有限差分近似超目标函数值的方向梯度。子程序 $\mathcal{A}$ 近似评估 $\varphi(\mathbf{x})$（通过求解内层优化）。

**Theorem 4**: 对于悲观BLO，设置步长 $\eta = \Theta(m^{-1/2}T^{-1/2})$，有 $\mathbb{E}[\|\nabla\varphi_{p,\gamma}(\bar{\mathbf{x}})\|^2] = O(\sqrt{m}/\sqrt{T})$。对于乐观BLO，$\mathbb{E}[\text{dist}(\mathbf{0}, \cup_{\mathbf{z} \in \mathbb{B}(\bar{\mathbf{x}}, \delta)} \partial\varphi_o(\mathbf{z}))^2] = O(\sqrt{m}/\sqrt{T})$，$\delta = O(T^{-1/4})$。

## 实验关键数据

本文以理论贡献为主，无标准数值实验表格。关键理论结果对比如下：

### 方法对比

| 方法 | 稳定性保证 | 是否需要下层强凸 | 是否需要惩罚模型PŁ |
|---|---|---|---|
| 隐式梯度下降 | $\|\nabla\varphi\| \leq \epsilon$ | 是 | N/A |
| Kwon et al. 惩罚法 | $\|\nabla\varphi\| \leq \epsilon$ | 否 | 是 |
| Chen et al. (Goldstein) | Goldstein稳定 | 否 | 否 |
| **本文 (Clarke)** | **Clarke稳定** | **否** | **否** |

### 收敛复杂度

| 设定 | 复杂度 | 稳定性类型 |
|---|---|---|
| 悲观BLO | $O(\sqrt{m}\Delta_p / \sqrt{T})$ | Clarke (via Moreau envelope) |
| 乐观BLO | $O(\sqrt{m}\Delta_o / \sqrt{T})$ | $(ε, O(T^{-1/4}))$-Clarke |

### 关键发现

- Clarke稳定性严格强于Goldstein稳定性：存在凸Lipschitz函数在某点是Goldstein稳定但远非Clarke稳定
- 下层约束($\mathbf{y} \in \mathcal{Y}$)可能破坏弱凸性（Example 3展示了一个具体反例），而上层约束不影响
- 集合光滑性不一定需要误差界条件（Example 2），存在更广泛的充分条件待发现

## 亮点与洞察

- "集合光滑性"是一个优美且具有独立价值的概念，将经典的实值函数光滑性推广到集值映射，为参数化优化问题的值函数分析提供了新工具
- 残差回填的证明技巧很巧妙——通过"先投影再修正"消除分支不匹配带来的一阶误差
- 乐观设定下（弱凹情形）没有现成的Moreau envelope方法，作者发展了基于Brøndsted-Rockafellar型近似的新收敛分析
- 统一了乐观和悲观BLO的分析框架

## 局限与展望

- 零阶方法的复杂度中有 $\sqrt{m}$ 因子（$m$ 为上层维度），对高维问题不够理想
- 下层约束情形（$\mathbf{y} \in \mathcal{Y}$）下弱凸性可能失效，需要新的结构条件
- 尚未设计利用弱凸/弱凹结构的更快算法（如近端类方法）
- 集合光滑性在 minimax 优化、集值优化等其他领域的应用有待探索

## 相关工作与启发

- 与Chen et al. (2023, 2024)的无下层强凸BLO工作形成互补：他们只保证Goldstein稳定性，本文提升到Clarke稳定性
- 集合光滑性与变分分析中的经典概念（Lipschitz连续、proto-differentiability等）相关但更强，提供了更精细的结构信息
- 弱凸优化的Moreau envelope技术(Davis & Drusvyatskiy, 2019)是本文算法分析的基石

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 集合光滑性概念新颖，首次证明非光滑超目标的Clarke稳定性可计算
- 实验充分度: ⭐⭐⭐ 纯理论贡献，缺乏数值验证
- 写作质量: ⭐⭐⭐⭐ 逻辑链清晰（概念→结构性质→算法→保证），但数学密度高
- 价值: ⭐⭐⭐⭐⭐ 解决了双层优化中的根本理论问题，集合光滑性概念有广泛应用前景

<!-- RELATED:START -->

## 相关论文

- [Problem-Parameter-Free Decentralized Bilevel Optimization](problem-parameter-free_decentralized_bilevel_optimization.md)
- [An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)
- [Learning Theory for Kernel Bilevel Optimization](learning_theory_for_kernel_bilevel_optimization.md)
- [A Single-Loop First-Order Algorithm for Linearly Constrained Bilevel Optimization](a_single-loop_first-order_algorithm_for_linearly_constrained_bilevel_optimizatio.md)
- [Escaping Saddle Points without Lipschitz Smoothness: The Power of Nonlinear Preconditioning](escaping_saddle_points_without_lipschitz_smoothness_the_power_of_nonlinear_preco.md)

<!-- RELATED:END -->
