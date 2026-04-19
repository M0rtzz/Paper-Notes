---
title: >-
  [论文解读] Gradient-Variation Online Adaptivity for Accelerated Optimization with Hölder Smoothness
description: >-
  [NeurIPS 2025][online learning] 在 Hölder 光滑函数类上实现梯度变差自适应的在线学习算法，其 regret 在光滑和非光滑极端之间平滑插值；通过在线到批量转换，首次为强凸优化提供在光滑情形下加速、非光滑情形下近优的通用方法。
tags:
  - NeurIPS 2025
  - online learning
  - Hölder smoothness
  - gradient variation
  - acceleration
  - universality
  - online-to-batch conversion
---

# Gradient-Variation Online Adaptivity for Accelerated Optimization with Hölder Smoothness

**会议**: NeurIPS 2025  
**arXiv**: [2511.02276](https://arxiv.org/abs/2511.02276)  
**作者**: Yuheng Zhao, Yu-Hu Yan, Kfir Yehuda Levy, Peng Zhao (Nanjing University / Technion)
**代码**: 待确认  
**领域**: reinforcement_learning  
**关键词**: online learning, Hölder smoothness, gradient variation, acceleration, universality, online-to-batch conversion

## 一句话总结

在 Hölder 光滑函数类上实现梯度变差自适应的在线学习算法，其 regret 在光滑和非光滑极端之间平滑插值；通过在线到批量转换，首次为强凸优化提供在光滑情形下加速、非光滑情形下近优的通用方法。

## 背景与动机

- 光滑性在加速优化和梯度变差 regret 最小化中至关重要
- **Hölder 光滑**统一了光滑与非光滑：$(L_\nu, \nu)$-Hölder smooth 满足 $\|\nabla\ell(\mathbf{x}) - \nabla\ell(\mathbf{y})\| \leq L_\nu \|\mathbf{x} - \mathbf{y}\|^\nu$
    - $\nu = 1$：标准光滑；$\nu = 0$：Lipschitz 连续
- 凸情形的通用方法已有解决（Kavis et al., 2019），但**强凸设置下实现光滑加速 + 非光滑近优**仍为开放问题
- 核心思路：利用在线学习的梯度变差自适应性，通过 online-to-batch 转换获得离线优化加速

## 核心问题

1. 能否在 Hölder 光滑假设下实现梯度变差 regret，在光滑和非光滑极端间平滑插值？
2. 能否由此获得离线（强）凸优化的**通用方法**——不需知道光滑参数即自适应加速？

## 方法详解

### Hölder 光滑的"近似光滑"引理（Lemma 1）

$(L_\nu, \nu)$-Hölder smooth 函数满足：对任意 $\delta > 0$，令 $L = \delta^{(\nu-1)/(1+\nu)} L_\nu^{2/(1+\nu)}$：

$$\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\|^2 \leq L^2 \|\mathbf{x} - \mathbf{y}\|^2 + 4L\delta$$

将 Hölder 光滑转化为标准光滑 + 常数扰动形式，是后续分析的关键桥梁。

### 凸情形：Optimistic OGD + AdaGrad 步长

使用乐观在线梯度下降，维护两个决策序列：

$$\mathbf{x}_t = \Pi_\mathcal{X}[\hat{\mathbf{x}}_t - \eta_t M_t], \quad \hat{\mathbf{x}}_{t+1} = \Pi_\mathcal{X}[\hat{\mathbf{x}}_t - \eta_t \nabla f_t(\mathbf{x}_t)]$$

关键创新——AdaGrad 式步长实现**隐式裁剪**，无需知道光滑参数：

$$\eta_{t+1} = \frac{D}{2\sqrt{A_t}}, \quad A_t = \|\nabla f_1(\mathbf{x}_1)\|^2 + \sum_{s=2}^{t} \|\nabla f_s(\mathbf{x}_s) - M_s\|^2$$

**虚拟裁剪原理**：$\eta_{t+1}$ 单调递减，最终自动变小于 $1/L$。在自动裁剪前，$\sqrt{A_\tau}$ 本身很小，未消去的梯度变差项可控。

### 凸 Hölder regret（Theorem 1）

$$\text{Reg}_T \leq O\left(D\sqrt{V_T} + L_\nu D^{1+\nu} T^{(1-\nu)/2} + D\|\nabla f_1(\mathbf{x}_1)\|\right)$$

其中梯度变差 $V_T = \sum_{t=2}^T \sup_\mathbf{x} \|\nabla f_t(\mathbf{x}) - \nabla f_{t-1}(\mathbf{x})\|^2$。

验证极端情形：
- $\nu = 1$（光滑）：恢复最优 $O(D\sqrt{V_T} + LD^2)$
- $\nu = 0$（Lipschitz）：恢复最优 $O(GD\sqrt{T})$
- **不需要 $L_\nu$ 和 $\nu$ 的知识**（强通用）

### 强凸情形的通用方法

**三步方案**：
1. 强凸梯度变差 regret：$O\big(\frac{1}{\lambda}\log V_T + \frac{1}{\lambda}L_\nu^2 (\log T)^{(1-\nu)/(1+\nu)}\big)$
2. **检测机制**：基于 guess-and-check 过程，自适应判断函数接近光滑还是非光滑
3. 精心设计的 O2B 转换

### Stabilized Online-to-Batch Conversion

将在线 regret 转化为离线收敛率：

$$\mathbb{E}[\ell(\bar{\mathbf{x}}_T)] - \ell(\mathbf{x}_\star) \leq \frac{\mathbb{E}[\text{Reg}_T^\alpha]}{\alpha_{1:T}}$$

通过设 $\alpha_t = t$ 并利用梯度变差得到 $O(1)$ 加权 regret，最终实现 $O(1/T^2)$ 收敛。

## 实验关键数据

### 凸 + 随机优化收敛率对比

| 方法 | 收敛率 | 通用性 |
|------|--------|--------|
| Kavis et al., 2019 | $O(L/T^2 + \sigma/\sqrt{T})$ smooth; $O(1/\sqrt{T})$ Lipschitz | 弱通用 |
| Rodomanov et al., 2024 | $O(L_\nu / T^{(1+3\nu)/2} + \sigma/\sqrt{T})$ | 强通用 |
| **本文 Theorem 2** | $O(L_\nu / T^{(1+3\nu)/2} + \sigma/\sqrt{T})$ | **强通用** |

### 强凸优化收敛率对比

| 方法 | 光滑收敛率 | 非光滑收敛率 | 通用性 |
|------|-----------|------------|--------|
| Levy, 2017 | $O(\exp(-T/\kappa) \cdot T/\kappa)$（非加速） | $\tilde{O}(1/\lambda T)$ | 弱（非加速） |
| **本文 Theorem 4** | $O(\exp(-T/6\sqrt{\kappa}))$（**加速**） | $\tilde{O}(1/\lambda T)$ | **弱通用** |

**首次**在强凸通用方法中实现光滑加速收敛。

### 在线学习 regret 对比

| 设置 | 光滑最优 | Lipschitz 最优 | 本文（Hölder） |
|------|---------|---------------|----------------|
| 凸 | $O(\sqrt{V_T} + LD^2)$ | $O(GD\sqrt{T})$ | $O(D\sqrt{V_T} + L_\nu D^{1+\nu} T^{(1-\nu)/2})$ |
| 强凸 | $O(\frac{1}{\lambda}\log V_T)$ | $O(\frac{1}{\lambda}\log T)$ | $O(\frac{1}{\lambda}\log V_T + \frac{L_\nu^2}{\lambda}(\log T)^{(1-\nu)/(1+\nu)})$ |

### 通用性定义（Definition 1）

| 类型 | 说明 |
|------|------|
| 弱通用 | 同时自适应光滑和非光滑（Lipschitz） |
| 强通用 | 同时自适应 $(L_\nu, \nu)$-Hölder smooth，$\nu \in [0,1]$ |

## 亮点

- ⭐⭐⭐⭐ 首次在强凸通用优化中实现加速率，解决了长期开放问题
- ⭐⭐⭐⭐ Hölder "近似光滑"引理优雅地统一了分析框架
- ⭐⭐⭐⭐ AdaGrad 步长的虚拟裁剪技巧——无需光滑参数即可自适应
- ⭐⭐⭐ 凸情形首次实现不依赖 $L$ 的最优梯度变差 regret
- 建立了梯度变差在线自适应性与离线加速优化间的清晰桥梁

## 局限与展望

- 强凸仅实现弱通用（光滑 vs 非光滑），尚未扩展到完整 Hölder 参数的强通用
- 检测机制增加复杂性，实际实现效率待验证
- 缺乏数值实验
- 是否可去除检测步骤，直接通过在线自适应实现强凸强通用是开放问题
- 与深度学习优化器的联系值得进一步探索

## ⭐ 推荐指数：⭐⭐⭐⭐⭐

理论优化领域的重要进展，将在线学习自适应性转化为离线优化加速。Hölder 光滑的统一视角和虚拟裁剪技巧具有广泛技术启发性，对优化理论社区具有深远影响。

<!-- RELATED:START -->

## 相关论文

- [Online Optimization for Offline Safe Reinforcement Learning](online_optimization_for_offline_safe_reinforcement_learning.md)
- [Robust and Diverse Multi-Agent Learning via Rational Policy Gradient](robust_and_diverse_multi-agent_learning_via_rational_policy_gradient.md)
- [On the Global Optimality of Policy Gradient Methods in General Utility Reinforcement Learning](on_the_global_optimality_of_policy_gradient_methods_in_general_utility_reinforce.md)
- [Bandit and Delayed Feedback in Online Structured Prediction](bandit_and_delayed_feedback_in_online_structured_prediction.md)
- [Trust Region Reward Optimization and Proximal Inverse Reward Optimization Algorithm](trust_region_reward_optimization_and_proximal_inverse_reward_optimization_algori.md)

<!-- RELATED:END -->
