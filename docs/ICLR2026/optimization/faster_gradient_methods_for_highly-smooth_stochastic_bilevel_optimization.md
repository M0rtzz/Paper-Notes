---
title: >-
  [论文解读] Faster Gradient Methods for Highly-Smooth Stochastic Bilevel Optimization
description: >-
  [ICLR2026][优化][bilevel optimization] 通过将 F2SA 方法重新解释为前向差分近似 hyper-gradient，提出利用高阶有限差分的 F2SA-p 方法族，在高阶光滑条件下将随机双层优化的 SFO 复杂度从 $\tilde{\mathcal{O}}(\epsilon^{-6})$ 改进至 $\tilde{\mathcal{O}}(p\epsilon^{-4-2/p})$，并证明了 $\Omega(\epsilon^{-4})$ 下界表明该方法在 $p$ 足够大时近乎最优。
tags:
  - ICLR2026
  - 优化
  - bilevel optimization
  - stochastic optimization
  - finite difference
  - hyper-gradient estimation
  - complexity lower bound
---

# Faster Gradient Methods for Highly-Smooth Stochastic Bilevel Optimization

**会议**: ICLR2026  
**arXiv**: [2509.02937](https://arxiv.org/abs/2509.02937)  
**代码**: [GitHub](https://github.com/TrueNobility303/F2BA)  
**领域**: optimization  
**关键词**: bilevel optimization, stochastic optimization, finite difference, hyper-gradient estimation, complexity lower bound

## 一句话总结
通过将 F2SA 方法重新解释为前向差分近似 hyper-gradient，提出利用高阶有限差分的 F2SA-p 方法族，在高阶光滑条件下将随机双层优化的 SFO 复杂度从 $\tilde{\mathcal{O}}(\epsilon^{-6})$ 改进至 $\tilde{\mathcal{O}}(p\epsilon^{-4-2/p})$，并证明了 $\Omega(\epsilon^{-4})$ 下界表明该方法在 $p$ 足够大时近乎最优。

## 背景与动机
双层优化 (bilevel optimization) 在元学习、超参数调优、对抗训练、强化学习等任务中广泛出现，其目标为 $\min_{\bm{x}} f(\bm{x}, \bm{y}^*(\bm{x}))$，其中 $\bm{y}^*(\bm{x}) = \arg\min_{\bm{y}} g(\bm{x}, \bm{y})$。现有方法主要分两类：

1. **基于 Hessian-向量积 (HVP) 的方法**：如 BSA、stocBiO，需要随机 Hessian 预言机，假设更强
2. **纯一阶方法**：如 F2SA，仅需随机梯度预言机，假设更弱，实践中可扩展到 32B 规模 LLM 训练

F2SA 在标准 SGD 假设下取得了 $\tilde{\mathcal{O}}(\epsilon^{-6})$ 的复杂度上界，但与单层优化的 $\Omega(\epsilon^{-4})$ 下界仍有显著差距。问题在于：**纯一阶方法能否在双层优化中达到最优复杂度？**

## 核心问题
在上层函数非凸、下层函数强凸的随机双层优化设定下，能否利用目标函数的高阶光滑性质，突破现有纯一阶方法的 $\tilde{\mathcal{O}}(\epsilon^{-6})$ 复杂度瓶颈，逼近 $\Omega(\epsilon^{-4})$ 的下界？

## 方法详解

### 关键洞察：F2SA 即前向差分
作者首先揭示了 F2SA 与有限差分之间的深层联系。定义扰动下层问题 $g_\nu(\bm{x}, \bm{y}) = \nu f(\bm{x}, \bm{y}) + g(\bm{x}, \bm{y})$ 及其最优值函数 $\ell_\nu(\bm{x})$，可以证明：

$$\frac{\partial^2}{\partial \nu \partial \bm{x}} \ell_\nu(\bm{x})\big|_{\nu=0} = \nabla \varphi(\bm{x})$$

F2SA 的惩罚函数梯度本质上是用**前向差分** $(\ell_\nu - \ell_0)/\nu$ 来近似 $\partial \ell_\nu / \partial \nu|_{\nu=0}$，仅有一阶误差保证 $\mathcal{O}(\nu)$。

### F2SA-p：高阶有限差分
既然前向差分只是诸多差分格式中最简单的一种，自然可以用更高阶的有限差分来降低近似误差。根据数值分析中的经典结果 (Lemma 3.1)，$p$ 阶有限差分可以实现 $\mathcal{O}(\nu^p)$ 的近似精度：

- **$p=1$（前向差分）**：$\alpha_0=-1, \alpha_1=1$，即原始 F2SA
- **$p=2$（中心差分）**：$\alpha_{-1}=-1/2, \alpha_1=1/2$，对应对称惩罚问题 F2SA-2

对于一般的偶数 $p$，算法结构为双层循环：

1. **内层循环**：对每个 $j = -p/2, \ldots, p/2$，运行 $K$ 步 SGD 求解扰动下层问题 $g_{j\nu}(\bm{x}, \cdot)$，近似 $\bm{y}_{j\nu}^*(\bm{x})$
2. **外层循环**：利用有限差分系数 $\{\alpha_j\}$ 线性组合各扰动点的梯度信息构建 hyper-gradient 估计 $\Phi_t$，然后执行归一化梯度下降步 $\bm{x}_{t+1} = \bm{x}_t - \eta_x \Phi_t / \|\Phi_t\|$

### 高阶光滑性假设
方法的改进依赖于下层变量 $\bm{y}$ 方向上的高阶光滑性（Assumption 2.5），即要求 $f, g$ 关于 $\bm{y}$ 的高阶导数满足 Lipschitz 连续。注意这比要求关于 $(\bm{x}, \bm{y})$ 联合高阶光滑更弱。softmax 函数是天然满足任意阶光滑性的经典例子，因此许多涉及逻辑回归的超参数调优问题（如 data hyper-cleaning、learn-to-regularize）天然满足此条件。

### 复杂度分析
通过高维 Faà di Bruno 公式证明 $\ell_\nu(\bm{x})$ 关于 $\nu$ 的高阶导数的 Lipschitz 连续性 (Lemma 3.2)，从而保证有限差分的 $\mathcal{O}(\nu^p)$ 误差。主定理表明：

$$\text{SFO 复杂度} = \tilde{\mathcal{O}}\left(p \kappa^{9+2/p} \epsilon^{-4-2/p}\right)$$

当 $p = \Omega(\log\epsilon^{-1}/\log\log\epsilon^{-1})$ 时，复杂度简化为 $\tilde{\mathcal{O}}(\kappa^9 \epsilon^{-4})$，匹配基于 HVP 方法在随机 Hessian 假设下的最优复杂度。

### 归一化梯度下降
与原始 F2SA 不同，外层循环使用归一化梯度步 $\bm{x}_{t+1} = \bm{x}_t - \eta_x \Phi_t / \|\Phi_t\|$。归一化可以控制 $\bm{y}_{j\nu}^*(\bm{x}_t)$ 的变化幅度，简化内层分析。作者认为标准梯度步也可通过更复杂的分析得到同样保证。

### 偶数 p vs 奇数 p
偶数 $p$ 时中心差分的 $\alpha_0 = 0$，仅需 $p$ 个扰动点；奇数 $p$ 需要 $p+1$ 个点。因此 F2SA-2 实际上优于 F2SA（$p=1$）：同样求解 2 个下层问题，但误差阶更高。即使不满足二阶光滑也不会比 F2SA 差，仅退化为一阶误差。

### 下界证明
通过构造一个完全可分的双层实例（$f(\bm{x}, \bm{y}) \equiv f_{\bm{U}}(\bm{x})$，$g(\bm{x}, y) = \mu y^2/2$），将双层问题约化为单层问题，从而继承 Arjevani et al. (2023) 的 $\Omega(\epsilon^{-4})$ 下界。该构造巧妙地自动满足所有高阶光滑性条件，避免了先前工作（Dagréou et al. 2024, Kwon et al. 2024a）构造中违反光滑性假设的问题。

## 实验关键数据
在 20 Newsgroups 数据集上进行 learn-to-regularize 逻辑回归实验（18000 样本，130107 维特征，20 类），该问题天然满足任意阶高阶光滑。

- 比较方法：F2SA-p（$p \in \{2,3,5,8,10\}$）vs F2SA vs stocBiO vs MRBO vs VRBO
- 内循环 $K=10$，外循环 $T=1000$
- F2SA-p 系列随 $p$ 增大收敛速度逐步改善，验证了理论预测
- 附录中在 5 层 ReLU MLP 上的实验表明方法在非光滑非凸问题上也有潜力

## 亮点
1. **优雅的理论洞察**：将 F2SA 重新解释为有限差分，这一视角自然地导出了算法改进方向
2. **几乎免费的改进**：F2SA-2 只需求解 2 个下层问题（与 F2SA 相同），但获得二阶误差保证
3. **完备的理论体系**：上界 $\tilde{\mathcal{O}}(p\epsilon^{-4-2/p})$ + 下界 $\Omega(\epsilon^{-4})$，证明高阶光滑区域近乎最优
4. **更紧的已知界**：$p=1$ 时改进系数 $\kappa$，$p=2$ 时修正了先前工作的 Hessian 收敛界
5. **假设更弱**：仅需 $\bm{y}$ 方向高阶光滑，不需联合高阶光滑或随机 Hessian 假设

## 局限与展望
1. 当 $p$ 较小（特别是 $p=1$）时上下界仍有差距，即使 $p=1$ 的最优复杂度仍是开放问题
2. 条件数 $\kappa$ 的依赖存在 $\Omega(\kappa^9)$ 的上下界差距
3. 实验仅在凸下层问题上验证，非凸-非凸结构化双层优化的推广尚待研究
4. 未与方差缩减或动量技术结合，有进一步加速空间
5. 高阶光滑性假设限制了适用范围，深度网络中通常不满足

## 与相关工作的对比

| 方法 | 光滑阶 | 复杂度 | 需要 HVP |
|------|--------|--------|----------|
| BSA | 1 阶 | $\tilde{\mathcal{O}}(\epsilon^{-6})$ SFO + $\tilde{\mathcal{O}}(\epsilon^{-4})$ HVP | 是 |
| stocBiO | 1 阶 | $\tilde{\mathcal{O}}(\epsilon^{-4})$ | 是 |
| F2SA | 1 阶 | $\tilde{\mathcal{O}}(\kappa^{12}\epsilon^{-6})$ | 否 |
| **F2SA-p** | **$p$ 阶** | **$\tilde{\mathcal{O}}(p\kappa^{9+2/p}\epsilon^{-4-2/p})$** | **否** |
| 下界 | $p$ 阶 | $\Omega(\epsilon^{-4})$ | — |

相比 Chayti & Jaggi (2024) 仅在元学习和对称近似中建立的有限差分联系，本文推广到一般双层优化和任意阶有限差分。相比 Huang et al. (2025) 需要联合高阶光滑，本文仅需 $\bm{y}$ 方向高阶光滑。

## 启发与关联
- **有限差分视角的力量**：将算法动机从惩罚函数转化为有限差分，为双层优化方法设计提供了全新透镜
- **可推广至其他问题**：有限差分改进思路可能适用于其他涉及隐函数定理的优化问题（如 minimax、compositional optimization）
- **实践指导**：F2SA-2 作为默认替代 F2SA 的方案，几乎无额外代价但理论更优

## 评分
- 新颖性: ⭐⭐⭐⭐ — 有限差分视角新颖且自然，算法族设计优雅
- 实验充分度: ⭐⭐⭐ — 实验相对简单，仅凸下层逻辑回归，大规模实验不足
- 写作质量: ⭐⭐⭐⭐⭐ — 论文结构清晰，从洞察到算法到理论的逻辑链条完整
- 价值: ⭐⭐⭐⭐ — 在双层优化理论中建立了重要的新结果，缩小了上下界差距

<!-- RELATED:START -->

## 相关论文

- [Nonlinearly Preconditioned Gradient Methods: Momentum and Stochastic Analysis](../../NeurIPS2025/optimization/nonlinearly_preconditioned_gradient_methods_momentum_and_stochastic_analysis.md)
- [Converge Faster, Talk Less: Hessian-Informed Federated Zeroth-Order Optimization](converge_faster_talk_less_hessian-informed_federated_zeroth-order_optimization.md)
- [Improved Last-Iterate Convergence of Shuffling Gradient Methods for Nonsmooth Convex Optimization](../../ICML2025/optimization/improved_last-iterate_convergence_of_shuffling_gradient_methods_for_nonsmooth_co.md)
- [Efficient Curvature-Aware Hypergradient Approximation for Bilevel Optimization](../../ICML2025/optimization/efficient_curvature-aware_hypergradient_approximation_for_bilevel_optimization.md)
- [ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting](../../ACL2025/optimization/scalebio_bilevel_data_reweighting.md)

<!-- RELATED:END -->
