---
title: >-
  [论文解读] The Price of Robustness: Stable Classifiers Need Overparameterization
description: >-
  [ICLR 2026][过参数化] 建立了不连续分类器的稳定性-泛化界，证明了分类任务中的"鲁棒性代价定律"：任何参数量 $p \approx n$ 的插值分类器必然不稳定，实现高稳定性需要 $p \approx nd$ 量级的过参数化。
tags:
  - "ICLR 2026"
  - "过参数化"
  - "鲁棒性"
  - "稳定性"
  - "分类器"
  - "泛化界"
  - "margin"
---

# The Price of Robustness: Stable Classifiers Need Overparameterization

**会议**: ICLR 2026  
**arXiv**: [2603.02806](https://arxiv.org/abs/2603.02806)  
**代码**: 无  
**领域**: 学习理论 / 泛化理论  
**关键词**: 过参数化, 鲁棒性, 稳定性, 分类器, 泛化界, margin

## 一句话总结

建立了不连续分类器的稳定性-泛化界，证明了分类任务中的"鲁棒性代价定律"：任何参数量 $p \approx n$ 的插值分类器必然不稳定，实现高稳定性需要 $p \approx nd$ 量级的过参数化。

## 研究背景与动机

- **过参数化的悖论**：经典学习理论认为参数越多过拟合越严重，但现代神经网络在过参数化时反而泛化更好（双下降现象）
- **Bubeck & Sellke 2021 的鲁棒性定律**：证明了回归场景中 Lipschitz 连续函数的平滑性-过参数化权衡。但该结果依赖 Lipschitz 假设，不适用于分类器（离散输出，天然不连续）
- **经验发现**：大规模研究（Jiang et al. 2019）显示 40+ 复杂度度量中，与泛化最一致相关的是 **margin**（到决策边界的距离），而非范数类度量
- **核心问题**：如何为不连续分类器建立稳定性驱动的泛化理论？

## 方法详解

### 1. 类稳定性定义

**定义 1**（Margin 和类稳定性）：对分类器 $f: \mathcal{X} \to \{-1, 1\}$：

无符号 margin：
$$h_f(x) := |d_f(x)| = \inf\{\|x - z\|_2 : f(z) \neq f(x), z \in \mathcal{X}\}$$

类稳定性（期望 margin）：
$$S(f) := \mathbb{E}[h_f]$$

这衡量了在数据分布下分类器预测对输入扰动的平均鲁棒性。

### 2. 等周不等式假设

假设数据分布 $\mu$ 满足 $c$-等周性（isoperimetry）：对任意有界 $L$-Lipschitz 函数 $f$ 和 $t \geq 0$：

$$\mathbb{P}(|f(x) - \mathbb{E}[f]| \geq t) \leq 2 e^{-\frac{dt^2}{2cL^2}}$$

高斯分布和球面均匀分布等满足此条件。根据流形假设，$d$ 可解释为内在流形维度。

### 3. 有限函数类的 Rademacher 界

**定理 4**：假设 $\min_{f \in \mathcal{F}} S(f) > S > 0$ 且 $\log|\mathcal{F}| \geq n$：

$$\mathcal{R}_{n,\mu}(\mathcal{F}) \leq K_1 \max\left\{\frac{1}{\sqrt{n}}, \frac{\sqrt{c}}{S} \cdot \frac{\log|\mathcal{F}|}{n\sqrt{d}}\right\}$$

在正则性条件下可改进为：

$$\mathcal{R}_{n,\mu}(\mathcal{F}) \leq K_2 \max\left\{\frac{1}{\sqrt{n}}, \frac{\sqrt{c}}{S}\sqrt{\frac{\log|\mathcal{F}|}{nd}}, 2\exp\left(-\frac{dS^2}{8c}\right)\right\}$$

**关键洞察**：$1/S$ 出现在 $\sqrt{\log|\mathcal{F}|}$ 前面——稳定性降低了模型类的有效复杂度。

### 4. 分类的鲁棒性定律

**推论 6**：令 $p := \log|\mathcal{F}| \geq n$。在适当条件下，以高概率：

$$\hat{R}_{\text{0-1}}(f) \leq R^* - \varepsilon \implies S(f) < \max\left\{\frac{3K}{\varepsilon}\sqrt{\frac{c\log|\mathcal{F}|}{nd}}, \sqrt{\frac{8c}{d}\log\frac{6K}{\varepsilon}}\right\}$$

**含义**：
- 参数量 $p \approx n$ 时，任何插值分类器必然不稳定
- 要同时实现低训练误差和高稳定性，需要 $p \approx nd$ 的过参数化

### 5. 无限函数类的扩展

引入**归一化 co-stability**：对分类器 $f = \text{sgn} \circ g_w$ 中得分函数的输出 margin 进行归一化。结合 $g_w$ 在参数和输入上的 Lipschitz 连续性，推导对应的泛化界（定理 13）和鲁棒性定律（推论 15）。

## 实验结果

### MNIST 实验

| 网络宽度 | 测试准确率 | 类稳定性 $S(f)$ | 谱范数 |
|---------|-----------|----------------|--------|
| 小 | 较低 | 低 | 无规律 |
| 中 | 中等 | 中 | 无规律 |
| 大 | 高 | **高** | 无规律 |

### CIFAR-10 实验

| 网络宽度 | 测试准确率 | 归一化 co-stability | 谱范数 |
|---------|-----------|-------------------|--------|
| 窄 | ~70% | 低 | 变化不一致 |
| 宽 | ~85% | **高** | 变化不一致 |
| 更宽 | ~90% | **更高** | 变化不一致 |

### 关键发现

- 稳定性和归一化 co-stability 随网络宽度**单调增加**
- 稳定性与测试性能呈**正相关**
- 传统范数度量（谱范数等）与泛化**无系统性关联**
- 验证了理论预测：过参数化→高稳定性→好泛化

## 亮点与洞察

1. **扩展到不连续函数**：首次将鲁棒性定律从 Lipschitz 回归推广到不连续分类器
2. **0-1 损失的直接分析**：不需要 Lipschitz 损失假设
3. **解释过参数化**：过参数化不是过拟合的源头，而是实现鲁棒性的必要条件
4. **适用广泛**：覆盖量化神经网络、脉冲神经网络等天然不连续模型
5. **Transformer 的特殊意义**：self-attention 通常不是 Lipschitz 连续的，本框架比 Lipschitz 框架更适用

## 局限性

- 等周假设对某些数据分布可能不成立
- 有限函数类到无限类的推广需要额外的参数 Lipschitz 假设
- 泛化界可能仍然是 vacuous 的（与 Nagarajan & Kolter 2021 的批评一致）
- 理论维度 $d$ 的实际值难以精确估计（外在维度 vs 内在维度）
- 未建立与优化动力学（如 implicit bias）的直接联系

## 相关工作

- **鲁棒性定律**：Bubeck & Sellke 2021（Lipschitz 回归）
- **Margin 泛化界**：Bartlett et al. 2017（谱归一化 margin）
- **算法稳定性**：Bousquet & Elisseeff 2002
- **双下降**：Belkin et al. 2019

## 评分

- **创新性**: ⭐⭐⭐⭐ — 将鲁棒性定律推广到分类的自然且重要的扩展
- **技术深度**: ⭐⭐⭐⭐⭐ — 证明技巧精巧，从 Lipschitz 代理到 signed distance 表示
- **实验充分性**: ⭐⭐⭐ — MNIST 和 CIFAR-10 验证，定性为主
- **实用价值**: ⭐⭐⭐⭐ — 为理解过参数化泛化提供理论支撑

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Evaluating GFlowNet from Partial Episodes for Stable and Flexible Policy-Based Training](evaluating_gflownet_from_partial_episodes_for_stable_and_flexible_policy-based_t.md)
- [\[ICLR 2026\] Fast and Stable Riemannian Metrics on SPD Manifolds via Cholesky Product Geometry](fast_and_stable_riemannian_metrics_on_spd_manifolds_via_cholesky_product_geometr.md)
- [\[ICLR 2026\] LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models](lipnext_scaling_up_lipschitz-based_certified_robustness_to_billion-parameter_mod.md)
- [\[ICLR 2026\] Key and Value Weights Are Probably All You Need: On the Necessity of the Query, Key, and Value Weight Triplet in Self-Attention](key_and_value_weights_are_probably_all_you_need_on_the_necessity_of_the_query_ke.md)
- [\[ICML 2026\] Private and Stable Test-Time Adaptation with Differential Privacy](../../ICML2026/others/private_and_stable_test-time_adaptation_with_differential_privacy.md)

</div>

<!-- RELATED:END -->
