---
title: >-
  [论文解读] Generalization and Robustness of the Tilted Empirical Risk
description: >-
  [ICML2025][优化][tilted empirical risk] 本文为负倾斜参数(γ<0)下的 Tilted Empirical Risk (TER) 提供了系统性的泛化误差上下界和鲁棒性保证，在损失函数无界但具有有界 (1+ε) 阶矩条件下，通过均匀方法和信息论方法建立了 $O(n^{-\epsilon/(1+\epsilon)})$ 的收敛速率，并给出了数据驱动的倾斜参数选择方案。
tags:
  - ICML2025
  - 优化
  - tilted empirical risk
  - 泛化误差
  - 鲁棒性
  - 信息论界
  - 无界损失函数
  - KL正则化
---

# Generalization and Robustness of the Tilted Empirical Risk

**会议**: ICML2025  
**arXiv**: [2409.19431](https://arxiv.org/abs/2409.19431)  
**代码**: 无  
**领域**: optimization  
**关键词**: tilted empirical risk, 泛化误差, 鲁棒性, 信息论界, 无界损失函数, KL正则化

## 一句话总结

本文为负倾斜参数(γ<0)下的 Tilted Empirical Risk (TER) 提供了系统性的泛化误差上下界和鲁棒性保证，在损失函数无界但具有有界 (1+ε) 阶矩条件下，通过均匀方法和信息论方法建立了 $O(n^{-\epsilon/(1+\epsilon)})$ 的收敛速率，并给出了数据驱动的倾斜参数选择方案。

## 研究背景与动机

- **Tilted Empirical Risk (TER)** 由 Li et al. (2021) 提出，是一种非线性风险度量：$\hat{R}_\gamma(h,S) = \frac{1}{\gamma}\log\left(\frac{1}{n}\sum_{i=1}^n \exp(\gamma \ell(h,Z_i))\right)$，当 γ→0 时退化为标准经验风险
- 实证研究已证明 TER 在 **负倾斜(γ<0)** 下能有效抵抗离群点/噪声标签，在 **正倾斜(γ>0)** 下能处理类别不平衡和公平性约束
- **理论缺口**：已有泛化分析要么限于有界损失函数(Lee et al. 2020)，要么假设损失及其导数均有界(Wang et al. 2023)，无法涵盖回归等常见的无界损失场景
- 现有无界损失下线性经验风险的泛化界(Cortes et al. 2019)收敛速率为 $O(\log(n) \cdot n^{-\epsilon/(1+\epsilon)})$，本文旨在去掉 log(n) 因子并推广到 TER

## 方法详解

### 整体框架

本文将 **倾斜泛化误差** 定义为总体风险与 TER 之差：

$$\text{gen}_\gamma(h,S) := R(h,\mu) - \hat{R}_\gamma(h,S)$$

核心思路是将其分解为三部分分别界定：(1) 总体风险与倾斜总体风险之差 $I_1$；(2) 非线性泛化误差 $\hat{\text{gen}}_\gamma$；(3) 分布偏移项。

### 关键设计一：均匀界 (Section 3.1)

**假设**：损失的 $(1+\epsilon)$ 阶矩一致有界，即 $\mathbb{E}_\mu[\ell^{1+\epsilon}(h,Z)] \le \kappa_u^{1+\epsilon}$，$\epsilon \in (0,1]$。

- **Proposition 3.2**（$I_1$ 的界）：$0 \le R(h,\mu) - R_\gamma(h,\mu^{\otimes n}) \le |\gamma|^\epsilon \kappa_u^{1+\epsilon}$
- **Proposition 3.3/3.4**：利用 Bernstein 不等式和对数函数性质，导出倾斜泛化误差的上下界
- **Theorem 3.5**（主定理）：对有限假设空间，以概率 ≥ 1−δ：

$$\sup_{h \in \mathcal{H}} |\text{gen}_\gamma(h,S)| \le \frac{2e^{|\gamma|\kappa_u}}{(1-\zeta)|\gamma|}\sqrt{\frac{2^\epsilon |\gamma|^{1+\epsilon}\kappa_u^{1+\epsilon} B(\delta)}{n}} + \frac{4e^{|\gamma|\kappa_u}B(\delta)}{3n|\gamma|(1-\zeta)} + |\gamma|^\epsilon \kappa_u^{1+\epsilon}$$

其中 $B(\delta) = \log(\text{card}(\mathcal{H})) + \log(2/\delta)$。当 $\gamma \asymp n^{-1/(1+\epsilon)}$ 时收敛速率为 $O(n^{-\epsilon/(1+\epsilon)})$。

### 关键设计二：信息论界 (Section 3.2)

放松均匀假设为依赖学习算法的矩条件 (Assumption 3.8)：

- **Theorem 3.11**：期望倾斜泛化误差的绝对值满足：

$$|\bar{\text{gen}}_\gamma(H,S)| \le D(\gamma)\sqrt{\frac{2\kappa_t^{1+\epsilon}|\gamma|^{1+\epsilon} I(H;S)}{n}} + |\gamma|^\epsilon \kappa_t^{1+\epsilon}$$

其中 $D(\gamma) = \frac{e^{|\gamma|\kappa_t}}{|\gamma|(1-\zeta)}$，$I(H;S)$ 为假设与训练集之间的互信息。

### 关键设计三：分布偏移下的鲁棒性 (Section 4)

建模噪声/离群引入的训练分布 $\tilde{\mu}$ 与真实分布 $\mu$ 的偏移：

- **Proposition 4.2**：倾斜总体风险在两个分布间的差异可用全变差距离 $\mathbb{TV}(\mu, \tilde{\mu})$ 界定
- **Theorem 4.3**：分布偏移下绝对倾斜泛化误差额外增加项 $\frac{\mathbb{TV}(\mu,\tilde{\mu})}{\gamma^2} \cdot \frac{e^{|\gamma|\kappa_u} - e^{|\gamma|\kappa_s}}{\kappa_u - \kappa_s}$
- **核心洞察**：与 ERM 不同，TER 在负倾斜下即使对无界损失也能用 TV 距离给出有限界，而 ERM 则需要 KL 散度且可能发散

### 关键设计四：KL 正则化 TERM (Section 6)

- 最优解为 **倾斜 Gibbs 后验**：$P_{H|S}^\gamma \propto \pi_H \cdot \left(\frac{1}{n}\sum_i e^{\gamma \ell(H,Z_i)}\right)^{-\alpha/\gamma}$
- 当 γ→0 退化为标准 Gibbs 后验
- **Theorem 6.3**：在 $\gamma = O(1/n)$ 时可达 $O(n^{-\epsilon})$ 收敛速率，比未正则化版本更快

### 数据驱动的倾斜参数选择 (Section 5)

通过最小化 Theorem 4.3 上界中关于 γ 的项来选择最优倾斜：

$$\gamma_{\text{data}} = \arg\min_{\gamma < 0}\left[|\gamma|^\epsilon \kappa_u^{1+\epsilon} + \frac{\mathbb{TV}(\mu,\tilde{\mu})}{\gamma^2}\cdot \frac{e^{|\gamma|\kappa_u} - e^{|\gamma|\kappa_s}}{\kappa_u - \kappa_s}\right]$$

## 实验关键数据

### 逻辑回归 + 高斯离群点 (Table 1, n=1000)

| 离群比例 ρ | γ*(网格搜索) | R(TERM) | R(ERM) | γ_data | R(数据驱动) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.1% | -0.53 | 0.00±0.000 | 0.05±0.001 | -1.40 | 0.00±0.000 |
| 17.6% | -2.98 | 0.15±0.004 | 0.22±0.001 | -4.91 | 0.16±0.003 |
| 35.0% | -3.86 | 0.16±0.004 | 0.30±0.002 | -3.33 | 0.20±0.002 |
| 52.5% | -2.10 | 0.11±0.001 | 0.28±0.002 | -1.93 | 0.14±0.001 |
| 70.0% | -1.23 | 0.14±0.002 | 0.18±0.000 | -2.28 | 0.15±0.002 |

### 逻辑回归 + Pareto 离群点 (Table 2, n=1000)

| 离群比例 ρ | γ*(网格搜索) | R(TERM) | R(ERM) | γ_data | R(数据驱动) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.1% | -1.40 | 0.00±0.000 | 0.03±0.001 | -0.70 | 0.01±0.000 |
| 17.58% | -3.33 | 0.00±0.003 | 0.02±0.000 | -0.88 | 0.01±0.000 |
| 35.05% | -1.05 | 0.00±0.000 | 0.01±0.002 | -0.70 | 0.01±0.000 |
| 52.53% | -1.05 | 0.01±0.000 | 0.01±0.002 | -1.06 | 0.01±0.001 |
| 70.0% | -0.88 | 0.00±0.002 | 0.02±0.001 | -0.70 | 0.01±0.000 |

### 关键发现

- **TERM 显著优于 ERM**：尤其在高离群比例下，ERM 的总体风险远高于 TERM（如 Gaussian ρ=35% 时 ERM=0.30 vs TERM=0.16）
- **数据驱动选择有效**：γ_data 的表现接近网格搜索最优 γ*，且方差更小
- **Pareto 离群更有利**：无界二阶矩的 Pareto 噪声下 TERM 和数据驱动方法的优势更为明显
- **鲁棒性-泛化权衡**：|γ| 过大增强鲁棒性但削弱泛化，存在最优点

## 亮点与洞察

1. **首次为无界损失下 TER 建立完整泛化理论**：填补了 TER 实证成功与理论保证之间的空白
2. **收敛速率改进**：去掉了 Cortes et al. (2019) 中的 log(n) 因子，达到 $O(n^{-\epsilon/(1+\epsilon)})$
3. **鲁棒性的理论刻画**：首次证明负倾斜 TER 在 TV 距离度量下对分布偏移具有有限界的鲁棒性，而 ERM 无法做到
4. **倾斜 Gibbs 后验**是新概念：KL 正则化 TERM 的最优解，γ→0 退化为标准 Gibbs 后验，提供更快的 $O(n^{-\epsilon})$ 收敛速率
5. **数据驱动倾斜选择**实用性强：基于理论上界优化即可获得接近最优的 γ

## 局限与展望

1. **仅分析负倾斜 γ<0**：正倾斜（处理类不平衡/公平性）的泛化分析未涉及
2. **实验设置简单**：仅在逻辑回归/线性回归上验证，缺少深度学习等复杂场景的实验
3. **有限假设空间假设**：均匀界需要有限 card(H)，虽可通过 ε-net / VC维推广，但实际松紧度未探讨
4. **互信息 I(H;S) 需有界**：信息论界的实用性依赖于能否有效估计/控制互信息
5. **TV 距离在高维中估计困难**：数据驱动方法需估计 TV 距离和矩参数，实际操作有挑战
6. **与 DRO 等鲁棒优化方法缺乏比较**：未与分布鲁棒优化等主流鲁棒方法做实验对比

## 相关工作与启发

- **Li et al. (2021, 2023a)**：TER 的提出和应用（类不平衡、离群/噪声、公平性）
- **Cortes et al. (2019)**：无界损失的 VC 维泛化界，收敛 $O(\log(n)n^{-\epsilon/(1+\epsilon)})$，本文改进
- **Xu & Raginsky (2017)**：信息论泛化界框架，本文将其推广到非线性经验风险
- **Haddouche & Guedj (2022)**：有界二阶矩下 PAC-Bayes 界，本文提供互补视角
- **Behnamnia et al. (2025)**：TER 在 off-policy 学习中的并行工作
- **启发**：TER 的理论优势（鲁棒性保证）可能对对抗训练、联邦鲁棒学习等场景产生指导

## 评分

- 新颖性: ⭐⭐⭐⭐ (首次为无界损失 TER 建立完整泛化/鲁棒性理论)
- 实验充分度: ⭐⭐ (实验过于简单，仅线性模型)
- 写作质量: ⭐⭐⭐⭐ (理论推导严谨，结构清晰)
- 价值: ⭐⭐⭐⭐ (为 TERM 的理论基础提供重要贡献，但实用影响待验证)

<!-- RELATED:START -->

## 相关论文

- [Tilted Sharpness-Aware Minimization](tilted_sharpness-aware_minimization.md)
- [A Generalization Result for Convergence in Learning-to-Optimize](a_generalization_result_for_convergence_in_learning-to-optimize.md)
- [Leveraging Perturbation Robustness to Enhance Out-of-Distribution Detection](../../CVPR2025/optimization/leveraging_perturbation_robustness_to_enhance_out-of-distribution_detection.md)
- [MeCeFO: Enhancing LLM Training Robustness via Fault-Tolerant Optimization](../../NeurIPS2025/optimization/mecefo_enhancing_llm_training_robustness_via_fault-tolerant_optimization.md)
- [FedSWA: Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging](fedswa_improving_generalization_in_federated_learning_with_highly_heterogeneous_.md)

<!-- RELATED:END -->
