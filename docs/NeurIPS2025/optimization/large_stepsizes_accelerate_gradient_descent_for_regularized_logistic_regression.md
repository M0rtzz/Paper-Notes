---
title: >-
  [论文解读] Large Stepsizes Accelerate Gradient Descent for Regularized Logistic Regression
description: >-
  [NeurIPS 2025][优化][gradient descent] 证明了在线性可分数据上对 $\ell_2$ 正则化逻辑回归使用大步长 GD（进入 Edge of Stability 区间），可将步复杂度从经典的 $\widetilde{O}(\kappa)$ 加速到 $\widetilde{O}(\sqrt{\kappa})$，在小正则化下匹配 Nesterov 动量的加速率。
tags:
  - NeurIPS 2025
  - 优化
  - gradient descent
  - large stepsizes
  - edge of stability
  - logistic regression
  - acceleration
  - condition number
---

# Large Stepsizes Accelerate Gradient Descent for Regularized Logistic Regression

**会议**: NeurIPS 2025  
**arXiv**: [2506.02336](https://arxiv.org/abs/2506.02336)  
**代码**: 无  
**领域**: optimization  
**关键词**: gradient descent, large stepsizes, edge of stability, logistic regression, acceleration, condition number

## 一句话总结
证明了在线性可分数据上对 $\ell_2$ 正则化逻辑回归使用大步长 GD（进入 Edge of Stability 区间），可将步复杂度从经典的 $\widetilde{O}(\kappa)$ 加速到 $\widetilde{O}(\sqrt{\kappa})$，在小正则化下匹配 Nesterov 动量的加速率。

## 研究背景与动机
**领域现状**：$\ell_2$ 正则化逻辑回归是经典强凸优化问题。标准 GD 以小步长单调递减目标函数，步复杂度为 $O(\kappa \ln(1/\varepsilon))$，已知可被 Nesterov 动量加速到 $O(\sqrt{\kappa} \ln(1/\varepsilon))$。

**现有工作**：Wu et al. (2024) 证明了无正则化逻辑回归中大步长 GD 可加速，但依赖最小值在无穷远处的特殊结构。对有限极小值的强凸问题，大步长是否仍有加速效应未知。

**核心矛盾**：(a) 正则化后极小值有限，GD 过大步长会导致不稳定发散；(b) 先前结果仅对优化误差 $\varepsilon < 1/n$ 有效，但统计误差通常 $\gg 1/n$。

**切入角度**：分析 GD 在 Edge of Stability (EoS) 区间的两阶段行为——先非单调振荡，后转入稳定阶段指数收敛。

## 方法详解

### 问题设定

最小化 $\ell_2$ 正则化逻辑损失：

$$\widetilde{\mathcal{L}}(w) = \frac{1}{n}\sum_{i=1}^n \ln(1 + \exp(-y_i x_i^\top w)) + \frac{\lambda}{2}\|w\|^2$$

数据满足 Assumption 1：$\|x_i\| \le 1$，$y_i x_i^\top w^* \ge \gamma > 0$（线性可分，间隔 $\gamma$）。

条件数 $\kappa = \Theta(1/\lambda)$，因光滑参数 $\approx 1+\lambda$，强凸参数 $= \lambda$。

### Theorem 1：小正则化下的收敛（匹配 Nesterov）

**条件**：$\lambda \le \gamma^2 / (C_1 n \ln n)$，步长 $\eta \le \min\{\gamma/\sqrt{C_1\lambda},\ \gamma^2/(C_1 n \lambda)\}$。

**两阶段行为**：
1. **EoS 阶段**（前 $\tau$ 步）：$\tau = O(\max\{\eta, n, n\ln n / \eta\}/\gamma^2)$ 步内目标函数非单调振荡
2. **稳定阶段**（$t \ge \tau$）：指数收敛 $\widetilde{\mathcal{L}}(w_t) - \min \widetilde{\mathcal{L}} \le C_3 e^{-\lambda\eta(t-\tau)}$

**Corollary 2**：取最大允许步长 $\eta = \gamma/\sqrt{C_1\lambda}$（当 $\lambda \lesssim \gamma^2/n^2$），步复杂度为

$$t = O\left(\frac{\ln(1/\varepsilon)}{\gamma\sqrt{\lambda}}\right) = O\left(\frac{\ln(1/\varepsilon)\sqrt{\kappa}}{\gamma}\right)$$

**匹配 Nesterov 加速**，且仅用恒定步长 GD！

### Theorem 3：稳定区间的下界

构造二维数据集证明在稳定区间（$\widetilde{\mathcal{L}}$ 单调递减）下，步复杂度至少为 $\Omega(\ln(1/\varepsilon)/(\lambda \ln^2(1/\lambda)))$，即 $\Omega(\kappa)$ 量级——说明加速必须来自 EoS 区间。

### Theorem 4：一般正则化下的改进

**条件**：$\lambda \le \gamma^2/C_1$，$\eta \le (\gamma^2/(C_1\lambda))^{1/3}$。

**步复杂度**（Corollary 5）：

$$t = O\left(\frac{\ln(1/\varepsilon)}{(\gamma\lambda)^{2/3}}\right)$$

虽不匹配 Nesterov $O(\sqrt{\kappa})$，但仍优于经典 $O(\kappa)$。此结果对任意 $\lambda$ 成立，无需 $\lambda \lesssim 1/(n\ln n)$ 的限制。

### 技术核心：EoS 阶段界（Lemma 1）

$$\frac{1}{t}\sum_{k=0}^{t-1} \mathcal{L}(w_k) \le O\left(\frac{\eta^2 + \ln^2(e + \gamma^2\min\{\eta t, 1/\lambda\})}{\gamma^2\min\{\eta t, 1/\lambda\}}\right)$$

$$\|w_t\| \le O\left(\frac{\eta + \ln(e + \gamma^2\min\{\eta t, 1/\lambda\})}{\gamma}\right)$$

利用 logistic loss 的 self-boundedness $\|\nabla^2\mathcal{L}(w)\| \le \mathcal{L}(w)$ 和强凸正则项的交互控制相变。

### 统计学习场景（Section 3）

对满足 Assumption 2（bounded separable distribution）的数据 + $\lambda = \Theta(1/n)$，各算法达到 population risk $\widetilde{O}(1/n)$ 所需步数：

| 算法 | 步数 | $\lambda$ | $\eta$ |
|------|------|-----------|--------|
| GD（无正则化，早停） | $O(n)$ | 0 | $\Theta(1)$ |
| GD（小步长） | $O(n\ln n)$ | $1/n$ | 1 |
| **GD（大步长）** | $O(n^{2/3}\ln n)$ | $1/n$ | $\Theta(n^{1/3})$ |
| Nesterov 动量 | $O(n^{1/2}\ln n)$ | $1/n$ | 1 |
| 自适应 GD | $O(1/\gamma^2)$ | 0 | $\Theta(\ln n)$ |

**大步长 + 正则化的组合**将 GD 步数从 $O(n)$ 降至 $O(n^{2/3})$，提供了首个大步长在统计不确定性下仍有加速效应的证据。

### 收敛临界步长（Section 4, 1D 结果）

在额外数据假设下，推导出局部收敛的临界步长阈值为 $\Theta(1/(\lambda\ln(1/\lambda)))$：
- 小于此的恒因子：GD 局部/全局收敛
- 大于此的恒因子：对几乎所有初始化 GD 发散

## 实验关键数据

Figure 2 展示二维可分数据集上不同步长的 GD 行为：

| 步长 | 行为 | 步数（达 $10^{-6}$ 误差） |
|------|------|-------------------------|
| $\eta = 1$（小） | 稳定区间，单调下降 | ≈3000 |
| $\eta = 10$（中） | EoS 区间，先振荡后收敛 | ≈500 |
| $\eta = 50$（大） | EoS 区间，更快收敛 | ≈200 |
| $\eta = 100$（过大） | 发散 | ∞ |

Sharpness（Hessian 最大特征值）在 EoS 阶段振荡在 $2/\eta$ 附近，与 Cohen et al. (2020) 的观察一致。

## 亮点
1. **仅用恒定步长 GD 即匹配 Nesterov 加速**——无需动量、无需知道 $\varepsilon$，方法极简
2. 首次在强凸有限极小值问题上证明 EoS 区间的全局收敛和加速效应
3. 阶段式分析（EoS → 稳定）框架清晰，可推广到其他问题
4. 统计场景下的加速结果弥合了纯优化理论与学习理论的鸿沟

## 局限与展望
1. Theorem 1 仅适用于小正则化 $\lambda \lesssim 1/(n\ln n)$，Theorem 4 虽放宽但未匹配 $\sqrt{\kappa}$
2. 严格依赖线性可分假设，非可分数据 Meng et al. (2024) 已构造反例
3. 临界步长结论仅限 1D，高维推广是开放问题
4. 缺少大规模数值实验验证理论预测的紧致性
5. Corollary 2 和 5 在重叠区域给出不可比较的界，暗示分析可进一步改进

## 与相关工作的对比
- **vs. Wu et al. (2024)**：后者研究无正则化（$\lambda=0$, 极小在 $\infty$），本文将大步长加速推广到有正则化的有限极小值场景
- **vs. Altschuler & Parrilo (2025) Silver stepsize**：后者达到 $\widetilde{O}(\kappa^{0.79})$（任意光滑强凸），本文在更窄问题类上达到 $\widetilde{O}(\kappa^{0.5})$ 且用恒定步长
- **vs. Nesterov 动量**：同样 $\sqrt{\kappa}$ 加速，但本文无额外动量项、步长不依赖 $\varepsilon$
- **vs. Zhang et al. (2025) 自适应 GD**：后者优化步数更少但 population risk 对 $\gamma$ 依赖更差

## 启发与关联
- EoS 现象在深度学习中普遍存在（Cohen et al. 2020），本文为理解其加速机制提供了严格理论基础
- 大步长 + 正则化的协同效应暗示实际训练中应同时调大 lr 和加正则化
- 两阶段分析框架可望推广至其他损失函数（hinge loss、cross-entropy）

## 评分
- ⭐ 新颖性: 5/5 — "恒定大步长 = Nesterov 加速"的发现非常令人意外
- ⭐ 实验充分度: 2/5 — 主要依赖理论和低维 toy 示例，实验不够丰富
- ⭐ 写作质量: 4/5 — 理论叙述清晰，阶段式证明结构好
- ⭐ 综合价值: 4/5 — 核心优化理论贡献突出，但适用范围有限

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Benefits of Early Stopping in Gradient Descent for Overparameterized Logistic Regression](../../ICML2025/optimization/benefits_of_early_stopping_in_gradient_descent_for_overparameterized_logistic_re.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)
- [\[NeurIPS 2025\] Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies](natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)
- [\[NeurIPS 2025\] Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study](do_neural_networks_need_gradient_descent_to_generalize_a_theoretical_study.md)

<!-- RELATED:END -->
