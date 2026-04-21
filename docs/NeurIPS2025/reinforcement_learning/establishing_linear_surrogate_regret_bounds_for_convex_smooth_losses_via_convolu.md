---
title: >-
  [论文解读] Establishing Linear Surrogate Regret Bounds for Convex Smooth Losses via Convolutional Fenchel–Young Losses
description: >-
  [NeurIPS 2025][surrogate regret bound] 通过构造基于卷积负熵（convolutional negentropy）的 Fenchel–Young 损失，首次证明凸且光滑的代理损失可以同时拥有线性代理遗憾界，打破了此前社区认为光滑性与线性遗憾率不可兼得的固有认知。
tags:
  - NeurIPS 2025
  - surrogate regret bound
  - convex smooth loss
  - Fenchel–Young loss
  - infimal convolution
  - discrete prediction
---

# Establishing Linear Surrogate Regret Bounds for Convex Smooth Losses via Convolutional Fenchel–Young Losses

**会议**: NeurIPS 2025  
**arXiv**: [2505.09432](https://arxiv.org/abs/2505.09432)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: surrogate regret bound, convex smooth loss, Fenchel–Young loss, infimal convolution, discrete prediction

## 一句话总结

通过构造基于卷积负熵（convolutional negentropy）的 Fenchel–Young 损失，首次证明凸且光滑的代理损失可以同时拥有线性代理遗憾界，打破了此前社区认为光滑性与线性遗憾率不可兼得的固有认知。

## 研究背景与动机

在机器学习中，目标损失（如 0-1 损失）通常是离散的、难以直接优化的，因此需要设计代理损失（surrogate loss）作为替代。好的代理损失应同时满足三个性质：**凸性**（保证优化可行）、**光滑性**（加速优化与估计收敛）和**线性遗憾率**（无损地将代理遗憾转化为目标遗憾）。

然而，此前的研究表明这三者似乎不可兼得：
- **合页损失（hinge loss）**：凸但非光滑 → 拥有线性遗憾率
- **逻辑损失（logistic loss）**：凸且光滑 → 仅有平方根遗憾率 $\psi(r) = O(\sqrt{r})$
- **Sigmoid 损失**：光滑但非凸 → 拥有线性遗憾率

poly_regret 等工作进一步证明，局部强凸且光滑的损失**必然**只能有平方根遗憾率。本文的核心贡献在于绕过这一限制，构造出一类既凸又光滑、同时享有线性遗憾率的代理损失。

## 方法详解

### 整体框架

本文基于 **Fenchel–Young 损失**框架进行构造。Fenchel–Young 损失由广义负熵 $\Omega$ 生成：$L_\Omega(\boldsymbol{\theta}, \boldsymbol{p}) = \Omega(\boldsymbol{p}) + \Omega^*(\boldsymbol{\theta}) - \langle\boldsymbol{\theta}, \boldsymbol{p}\rangle$，天然具备凸性。核心思路分三步：

1. 选择一个强凸的**基础负熵** $\Omega$（如 Shannon 负熵）
2. 将目标损失 $\ell$ 的结构通过**负贝叶斯风险** $T(\boldsymbol{p}) = -\min_{t} \langle \boldsymbol{p}, \boldsymbol{\ell}^{\rho}(t)\rangle$ 编码到负熵中，构造 $\Omega_T = \Omega + T$
3. 利用 infimal convolution 结构推导凸光滑代理损失及配套预测链接

### 关键设计

**卷积负熵与 infimal convolution**：卷积负熵 $\Omega_T = \Omega + T$ 的共轭具有 infimal convolution 形式：

$$\Omega_T^*(\boldsymbol{\theta}) = \inf_{\boldsymbol{\pi} \in \Delta^N} \Omega^*(\boldsymbol{\theta} + \mathcal{L}^{\rho}\boldsymbol{\pi})$$

其中 $\mathcal{L}^{\rho}$ 为损失矩阵。这是方法成功的**核心**——使得代理遗憾可进行可加分解，实现线性下界。

**$\boldsymbol{\pi}$-argmax 链接**：不同于标准 argmax，定义 $\Pi(\boldsymbol{\theta}) = \arg\min\{\Omega^*(\boldsymbol{\theta} + \mathcal{L}^{\rho}\boldsymbol{\pi}):\boldsymbol{\pi}\in\Delta^N\}$，预测取 $\pi_t$ 的最大分量对应的类别。该链接通过优化信息扭曲原始分数，融入目标损失结构。

**凸光滑性保证**：当 $\Omega$ 强凸时，$L_{\Omega_T}(\cdot,y)$ 在 $\mathbb{R}^d$ 上凸且光滑。关键在于该损失**不是局部强凸的**，从而绕过了已知的平方根遗憾下界。

### 损失函数 / 训练策略

**卷积 Fenchel–Young 损失**：
$$L_{\Omega_T}(\boldsymbol{\theta}, y) = \min_{\boldsymbol{\pi}\in\Delta^N}\Omega^*(\boldsymbol{\theta}+\mathcal{L}^{\rho}\boldsymbol{\pi}) + \Omega_T(\boldsymbol{\rho}(y)) - \langle\boldsymbol{\theta},\boldsymbol{\rho}(y)\rangle$$

**线性遗憾界（Theorem 13）**：$\text{Regret}_\ell(\varphi(\boldsymbol{\theta}),\boldsymbol{\eta}) \leq N \cdot \text{Regret}_{L_{\Omega_T}}(\boldsymbol{\theta},\boldsymbol{\eta})$

**常数改进（Corollary 16）**：利用低维分解将常数从 $N$ 降至 $\text{affdim}(\mathcal{L}^{\rho})+1$，多标签分类中从 $2^d$ 降至 $d+1$。

**梯度计算**：通过包络定理，$\nabla L_{\Omega_T}(\boldsymbol{\theta},y)=\nabla\Omega^*(\boldsymbol{\theta}+\mathcal{L}^{\rho}\boldsymbol{\pi})-\boldsymbol{\rho}(y)$，归结为求解 $\Pi(\boldsymbol{\theta})$。多类分类实例中，Algorithm 1 以 $O(K\ln K)$ 排序求解。

**Fisher 一致概率估计器（Theorem 17）**：$\nabla\Omega^*(\boldsymbol{\theta}^*+\mathcal{L}^{\rho}\boldsymbol{\pi})$ 给出 $\mathbb{E}[\boldsymbol{\rho}(y)]$ 的一致估计，可用于不确定性量化和拒绝分类。

## 实验关键数据

### 主实验（损失性质对比）

| 损失类型 | 凸性 | 光滑性 | 遗憾率 | 概率估计 |
|---------|------|--------|--------|---------|
| Hinge loss | ✓ | ✗ | $O(r)$ 线性 | ✗ |
| Logistic loss | ✓ | ✓ | $O(\sqrt{r})$ | ✓ |
| Sigmoid loss | ✗ | ✓ | $O(r)$ 线性 | ✗ |
| Polyhedral loss | ✓ | ✗ | $O(r)$ 线性 | ✗ |
| **Conv. FY loss（本文）** | **✓** | **✓** | **$O(r)$ 线性** | **✓** |

### 计算复杂度对比（多类分类）

| 方法 | 训练时梯度计算 | 测试时预测 |
|------|--------------|-----------|
| Cross-entropy（标准 FY） | $O(K)$ softmax | $O(K)$ argmax |
| Sparsemax（FY） | $O(K\ln K)$ 投影 | $O(K)$ argmax |
| **Conv. FY（本文）** | **$O(K\ln K)$ Algorithm 1** | **$O(K)$ argmax** |

### 消融实验

- **随机化链接**：使用 $\Pr[\tilde{\varphi}(\boldsymbol{\theta})=t]=\pi_t(\boldsymbol{\theta})$ 可将遗憾常数从 $N$ 改进至 **1**（维度无关）
- **低维分解 vs. 标准分解**：多标签分类用二进制编码后，优化维度从 $2^d$ 降至 $d$
- **$\boldsymbol{\pi}_{\log}$-argmax 等价性**：多类分类中与标准 argmax 等价（Proposition 19），测试零开销

### 关键发现

1. **凸光滑+线性遗憾可共存**，关键在于损失不局部强凸
2. 训练开销仅多一个排序步骤 $O(K\ln K)$，测试时与标准分类器完全一致
3. 天然提供概率估计器，打破了"线性遗憾与概率估计不兼容"的猜想

## 亮点与洞察

- **精妙的凸分析**：利用 infimal convolution 将"加法结构"从负熵空间传递到损失空间，使遗憾可加分解成立
- **不违反已知下界**：poly_regret 的下界要求局部强凸，而本文损失在边界不强凸（因 $T$ 不可微），巧妙避开限制
- **统一框架**：适用于任意离散目标损失——多类、多标签、top-k、结构化预测等

## 局限与展望

- **纯理论工作，缺乏实证实验**：实际训练中该损失与交叉熵的收敛速度对比尚未验证
- 高维结构化预测中求解 $\Omega_T^*$ 可能仍有较高计算开销
- 线性遗憾界常数在某些场景下可能较大，影响实际效用
- 是否可在低噪声假设下获得 fast rates 仍为开放问题

## 相关工作与启发

- **Polyhedral losses**（Mao et al.）：凸非光滑线性遗憾损失，本文可视为其"光滑化"对偶
- **Fenchel–Young losses**（Blondel et al.）：本文在此框架引入卷积结构，扩展了适用范围
- **SELF 分解**：为结构化预测提供低维标签编码，本文用其降低遗憾常数
- 对在线学习/bandit 问题，该损失可结合 OCO 框架实现高效在线预测

## 评分

- **新颖性**: 9/10 — 首次解决凸光滑+线性遗憾的长期开放问题
- **理论深度**: 10/10 — 凸分析工具运用精湛
- **实用性**: 5/10 — 纯理论，缺乏实证
- **清晰度**: 8/10 — 数学密集但逻辑连贯

<!-- RELATED:START -->

## 相关论文

- [Non-stationary Online Learning for Curved Losses: Improved Dynamic Regret via Mixability](../../ICML2025/reinforcement_learning/non-stationary_online_learning_for_curved_losses_improved_dynamic_regret_via_mix.md)
- [Improved Regret Bounds for GP-UCB in Bayesian Optimization](improved_regret_bounds_for_gaussian_process_upper_confidence_bound_in_bayesian_o.md)
- [Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update](generalized_linear_bandits_almost_optimal_regret_with_one-pass_update.md)
- [Improved Regret and Contextual Linear Extension for Pandora's Box and Prophet Inequality](improved_regret_and_contextual_linear_extension_for_pandoras_box_and_prophet_ine.md)
- [Dynamic Regret Reduces to Kernelized Static Regret](dynamic_regret_reduces_to_kernelized_static_regret.md)

<!-- RELATED:END -->
