---
title: >-
  [论文解读] Can Adaptive Gradient Methods Converge under Heavy-Tailed Noise? A Case Study of AdaGrad
description: >-
  [ICML 2026][优化/理论][自适应梯度方法] 首次证明 AdaGrad 在重尾噪声（$p \in (4/3, 2]$）下无需任何算法修改即可收敛，同时给出算法依赖的下界表明 AdaGrad 无法达到 minimax 最优速率…
tags:
  - "ICML 2026"
  - "优化/理论"
  - "自适应梯度方法"
  - "重尾噪声"
  - "AdaGrad"
  - "非凸优化"
  - "收敛速率"
---

# Can Adaptive Gradient Methods Converge under Heavy-Tailed Noise? A Case Study of AdaGrad

**会议**: ICML 2026  
**arXiv**: [2605.18694](https://arxiv.org/abs/2605.18694)  
**代码**: 无  
**领域**: 优化  
**关键词**: 自适应梯度方法, 重尾噪声, AdaGrad, 非凸优化, 收敛速率  

## 一句话总结
首次证明 AdaGrad 在重尾噪声（$p \in (4/3, 2]$）下无需任何算法修改即可收敛，同时给出算法依赖的下界表明 AdaGrad 无法达到 minimax 最优速率，并证明 AdaGrad-Norm 在有界目标函数假设下可获得更快的 $O(1/T^{(p-1)/(2p)})$ 速率。

## 研究背景与动机

**领域现状**：现代机器学习任务（特别是训练 Transformer 等注意力模型）的优化过程中，梯度噪声普遍呈现重尾分布——即噪声仅有有限的 $p$ 阶矩（$p \in (1, 2]$），而非经典的有限方差假设（$p = 2$）。为应对这一挑战，已有两类方法被证明可保证收敛：一是基于梯度裁剪的 Clipped SGD，二是基于梯度归一化的 NSGD(M)。然而，这两类方法均需额外的算法修改（裁剪阈值或归一化操作）。

**现有痛点**：实践中，自适应梯度方法（AdaGrad、Adam、AdamW 等）在重尾噪声场景下表现良好，但已有理论无法解释这一现象。唯一相关的理论工作 Chezhegov et al. (2025) 存在三个关键缺陷：(1) 主要结果针对"延迟"变体而非标准算法；(2) 需要梯度裁剪等额外修改；(3) 需要目标函数有界这一更强假设以及尾指数 $p$ 的先验知识。

**核心矛盾**：自适应梯度方法通过动态调节步长来隐式处理梯度规模差异，这种自适应机制在经典有限方差假设下并未体现出超越 SGD 的理论优势（两者同为 $\tilde{O}(1/T^{1/4})$）。那么，这种自适应性能否自然地应对重尾噪声？

**本文目标**：以 AdaGrad 为案例，回答"自适应梯度方法能否在无需任何修改的情况下，于重尾非凸优化中收敛"这一核心问题。

**切入角度**：作者注意到 AdaGrad 的逐坐标步长 $\gamma / (\lambda + \sqrt{v_t})$ 天然地根据梯度历史累积量进行缩放——当某坐标噪声大时，$v_t$ 大，步长自动缩小。这种隐式的"噪声自适应"可能足以控制重尾噪声。

**核心 idea**：通过推广 proxy stepsize 技巧（引入自由参数 $\boldsymbol{c}$ 替代固定的 $\boldsymbol{\sigma}$），首次证明 AdaGrad 的自适应步长机制本身就足以保证重尾噪声下的收敛，无需裁剪或归一化。

## 方法详解

### 整体框架
本文是一项纯理论工作，不提出新算法，而是分析标准 AdaGrad 和 AdaGrad-Norm 两个已有算法在重尾噪声下的收敛行为。分析框架包含三部分：(1) AdaGrad 的上界分析（证明收敛）；(2) AdaGrad 的算法依赖下界（证明有根本局限）；(3) AdaGrad-Norm 在更强假设下的改进上界。

标准 AdaGrad 的更新规则为：$v_t = v_{t-1} + g_t^2$，$x_{t+1} = x_t - \frac{\gamma}{\lambda + \sqrt{v_t}} g_t$，其中所有运算均为逐坐标进行。AdaGrad-Norm 则用全局标量 $v_t = v_{t-1} + \|g_t\|_2^2$ 替代逐坐标累积。

### 关键设计

1. **推广的 Proxy Stepsize 技巧（上界证明核心）**:

    - 功能：解决 AdaGrad 步长 $\gamma/(\lambda + \sqrt{v_t})$ 与随机梯度 $g_t$ 的统计耦合问题，使分析可进行条件期望运算
    - 核心思路：设定可预测（$\mathcal{F}_{t-1}$-可测）的代理步长 $w_t = v_{t-1} + (\nabla f(x_t))^2 + c^2$，其中 $c$ 为自由参数。关键创新在于选取 $c_i = \sigma_i T^{1/2 - 1/\bar{p}} / D_{T,i}^{1/2 - 1/\bar{p}}$，而非直接令 $c = \sigma$。当 $p = 2$ 时 $c$ 自然退化为 $\sigma$，恢复经典分析
    - 设计动机：若简单取 $c = \sigma$，最优速率仅为 $\tilde{O}(1/T^{(2p-3)/(2p)})$；而精心选择的 $c$ 可将速率改善至 $\tilde{O}(1/T^{(3p-4)/(4p)})$，这是一个严格的提升

2. **算法依赖的下界构造**:

    - 功能：证明 AdaGrad 存在无法逾越的收敛速率下限，且该下限随输入学习率 $\gamma$ 变化
    - 核心思路：在一维情形（$d=1$）下，针对给定的 $\gamma$，构造特定的目标函数 $f$ 和随机梯度预言机，使得以常数概率 $f'(x_t) \geq \Omega(\epsilon)$ 对所有 $t \in [T]$ 成立（当 $T$ 小于与 $\gamma$ 相关的阈值时）。下界为 $\Omega\big(\frac{\Delta^2/\gamma^2 + \gamma^2 L^2 \ln^2(\gamma L/\epsilon)}{\epsilon^2} + \frac{(\cdots)\sigma^{p/(p-1)}}{\epsilon^{(3p-2)/(p-1)}}\big)$
    - 设计动机：已有的 AdaGrad 下界（Jiang et al., 2025）不含学习率 $\gamma$ 的依赖，因此无法反映算法配置对复杂度的影响。新下界显式捕获 $\gamma$ 的作用，揭示 AdaGrad 无法通过调参达到 minimax 最优

3. **AdaGrad-Norm 的有界目标函数加速分析**:

    - 功能：证明 AdaGrad-Norm 在额外假设 $\sup f < +\infty$ 下可获得对所有 $p \in (1, 2]$ 均非平凡的 $O(1/T^{(p-1)/(2p)})$ 速率
    - 核心思路：利用 AdaGrad-Norm 的标量步长特性，将单步进展不等式中的 $(f(x_t) - f_\star)/\gamma_t$ 求和后用有界性条件控制为 $\Delta_\star / \gamma_T$；再通过 Lemma 4.7 将 $\mathbb{E}[\sqrt{v_T}]$ 分解为噪声项 $\|\sigma\|_p T^{1/p}$ 和梯度项 $\mathbb{E}[\sqrt{u_T}]$，最终用 AM-GM 不等式吸收递归项
    - 设计动机：标准 AdaGrad 的逐坐标步长阻止了上述推导，因此这一更快速率是 AdaGrad-Norm 特有的结构优势

## 实验关键数据

### 主要理论结果对比

| 算法 | 速率 | 适用 $p$ 范围 | 需要先验信息 | 额外假设 |
|------|------|--------------|-------------|---------|
| Clipped SGD | $O(1/T^{(p-1)/(3p-2)})$ (minimax 最优) | $(1, 2]$ | 需要 $p$ | 无 |
| NSGD(M) (有先验) | $O(1/T^{(p-1)/(3p-2)})$ | $(1, 2]$ | 需要 $p$ | 无 |
| NSGD(M) (无先验) | $O(1/T^{(p-1)/(2p)})$ | $(1, 2]$ | 不需要 | 无 |
| **AdaGrad (本文 Thm 3.1)** | $\tilde{O}(1/T^{(3p-4)/(4p)})$ | $(4/3, 2]$ | 不需要 | 无 |
| **AdaGrad-Norm (本文 Thm 4.2)** | $O(1/T^{(p-1)/(2p)})$ | $(1, 2]$ | 不需要 | 目标有界 |
| **AdaGrad-Norm (本文 Thm C.1)** | $\tilde{O}(1/T^{(3p-4)/(4p)})$ | $(4/3, 2]$ | 不需要 | 无 |

### 不同尾指数 $p$ 下的收敛速率对比

| 尾指数 $p$ | Minimax 最优 | AdaGrad (本文) | AdaGrad-Norm (有界, 本文) | NSGD(M) (无先验) |
|-----------|-------------|---------------|--------------------------|-----------------|
| $2.0$ | $O(1/T^{1/4})$ | $\tilde{O}(1/T^{1/4})$ | $O(1/T^{1/4})$ | $O(1/T^{1/4})$ |
| $1.5$ | $O(1/T^{1/5})$ | $\tilde{O}(1/T^{1/12})$ | $O(1/T^{1/6})$ | $O(1/T^{1/6})$ |
| $4/3$ | — | 平凡（临界点） | $O(1/T^{1/8})$ | $O(1/T^{1/8})$ |
| $1.2$ | $O(1/T^{1/8})$ | 平凡 | $O(1/T^{1/12})$ | $O(1/T^{1/12})$ |

### 关键发现
- AdaGrad 在 $p = 2$ 时恢复经典 $\tilde{O}(1/T^{1/4})$ 速率，与已有结果一致；但当 $p \leq 4/3$ 时速率退化为平凡，说明自适应步长的隐式噪声控制能力有限
- AdaGrad-Norm 在有界目标函数假设下可匹配 NSGD(M) 的无先验最优速率 $O(1/T^{(p-1)/(2p)})$，且对所有 $p \in (1, 2]$ 均非平凡
- 算法依赖下界显示，即使知道 $\Delta$ 和 $L$，AdaGrad 仍至少需要额外的 polylog 因子，这是对 Jiang et al. (2025) 下界的严格改进
- 本文是首个理论上证明自适应梯度方法在重尾噪声下优于 SGD 的工作

## 亮点与洞察
- **推广 proxy stepsize 的自由参数技巧**：将经典分析中固定为 $\sigma$ 的参数替换为可在证明末尾最优化的自由量 $c$，这一看似简单的推广带来了速率的实质改善。该技巧具有一般性，有望推广到 Adam/AdamW 等更复杂的自适应方法分析中
- **算法依赖下界的方法论贡献**：构造随学习率 $\gamma$ 参数化的困难实例，使下界显式依赖于算法配置，这比传统的 minimax 下界提供了更精细的信息。这种方法论可直接用于其他自适应优化器的理论分析
- **自适应性的"免费午餐"**：AdaGrad 无需知道尾指数 $p$ 就能自动适配到最大可用的 $p$ 值，同时适应噪声水平 $\|\sigma\|_1$ — 这是首次同时实现双重自适应性的理论保证

## 局限与展望
- 上界在 $p \leq 4/3$ 时退化为平凡，这一空白尚不清楚是分析瓶颈还是 AdaGrad 的本质限制
- 算法依赖下界在 $\epsilon$ 的指数上可能不紧，上下界之间仍有差距
- 仅分析了 AdaGrad 和 AdaGrad-Norm，未覆盖实践中更常用的 Adam/AdamW；将分析推广到这些方法是重要的后续方向
- AdaGrad-Norm 的更快速率需要目标函数有界的额外假设，该条件在深度学习中是否合理值得探讨
- 未考虑更现实的广义光滑性条件（generalized smoothness），这是理论与实践之间的另一个重要鸿沟

## 相关工作与启发
- 与 Clipped SGD 和 NSGD(M) 的对比揭示了自适应方法在重尾噪声下的独特理论地位：不需要裁剪/归一化等显式机制，但代价是无法达到 minimax 最优
- Ward et al. (2019) 的 AdaGrad-Norm 分析框架和 Liu (2026) 的在线凸优化重尾分析为本文的技术路线提供了关键灵感
- Hübler et al. (2025) 和 Liu & Zhou (2025) 关于 NSGD(M) 的"无需先验知识"速率 $O(1/T^{(p-1)/(2p)})$ 构成了自然的对比基准

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Clipping Improves Adam-Norm and AdaGrad-Norm when the Noise Is Heavy-Tailed](../../ICML2025/optimization/clipping_improves_adam-norm_and_adagrad-norm_when_the_noise_is_heavy-tailed.md)
- [\[ICML 2026\] Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time](bregman_meets_lévy_stochastic_mirror_descent_with_heavy-tailed_noise_in_continuo.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[ICML 2026\] On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach](on_the_interaction_of_batch_noise_adaptivity_and_compression_under_l_0l_1-smooth.md)
- [\[ICML 2026\] Mirror Descent Under Generalized Smoothness](mirror_descent_under_generalized_smoothness.md)

</div>

<!-- RELATED:END -->
