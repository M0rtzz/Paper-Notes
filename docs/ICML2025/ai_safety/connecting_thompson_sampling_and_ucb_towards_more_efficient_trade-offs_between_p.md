---
title: >-
  [论文解读] Connecting Thompson Sampling and UCB: Towards More Efficient Trade-offs Between Privacy and Regret
description: >-
  [ICML2025][AI安全][Thompson Sampling] 提出 DP-TS-UCB 算法，通过限制高斯采样次数并复用最大模型值，在 Thompson Sampling 和 UCB 之间建立连接，实现 $\tilde{O}(T^{0.25(1-\alpha)})$-GDP 隐私保证和 $O(K\ln^{\alpha+1}(T)/\Delta)$ 遗憾上界的参数化权衡。
tags:
  - ICML2025
  - AI安全
  - Thompson Sampling
  - UCB
  - 差分隐私
  - 高斯差分隐私(GDP)
  - 随机多臂老虎机
  - 隐私-遗憾权衡
  - anti-concentration bounds
---

# Connecting Thompson Sampling and UCB: Towards More Efficient Trade-offs Between Privacy and Regret

**会议**: ICML2025  
**arXiv**: [2505.02383](https://arxiv.org/abs/2505.02383)  
**代码**: 无  
**领域**: AI安全 / 差分隐私与在线学习  
**关键词**: Thompson Sampling, UCB, 差分隐私, 高斯差分隐私(GDP), 随机多臂老虎机, 隐私-遗憾权衡, anti-concentration bounds

## 一句话总结

提出 DP-TS-UCB 算法，通过限制高斯采样次数并复用最大模型值，在 Thompson Sampling 和 UCB 之间建立连接，实现 $\tilde{O}(T^{0.25(1-\alpha)})$-GDP 隐私保证和 $O(K\ln^{\alpha+1}(T)/\Delta)$ 遗憾上界的参数化权衡。


## 研究背景与动机

- **隐私多臂老虎机**：在线学习中学习者需利用历史数据做决策，但隐私约束要求限制信息泄露量。
- **现有方法的 GDP 松弛**：TS-Gaussian（Agrawal & Goyal, 2017）本身满足 $O(\sqrt{T})$-GDP，但每轮对每个臂都采样高斯模型导致隐私预算浪费严重。
- **核心洞察**：(1) 高斯分布仅在臂被拉取时才变化，中间轮次的采样是多余的隐私消耗；(2) 使用 arm-specific epoch 结构限制每个观测仅用一次，进一步减少隐私损失；(3) $\phi$ 个采样的最大值在概率意义上等价于 UCB 的乐观上界。

## 方法详解

### 整体框架

DP-TS-UCB 是一个两阶段算法：
1. **强制 TS-Gaussian 阶段**（$h_i \geq 1$）：从高斯分布 $\mathcal{N}(\hat{\mu}_{i,n_i}, \ln^\alpha(T)/n_i)$ 采样至多 $\phi$ 个模型值。
2. **可选 UCB 阶段**（$h_i = 0$）：复用 $\phi$ 个模型值中的最大值 $\text{MAX}_i = \max_{h \in [\phi]} \theta_{i,n_i}^{(h)}$ 作为乐观估计。

### 关键设计

- **采样预算**：$\phi = c_0 T^{0.5(1-\alpha)} \ln^{0.5(3-\alpha)}(T)$，由权衡参数 $\alpha \in [0,1]$ 和时间范围 $T$ 决定。
- **Lemma 4.1（充分探索保证）**：$\phi$ 个采样的最大值 $\max_{h \in [\phi]} \theta_{i,s}^{(h)} \geq \mu_i$ 的概率至少为 $1 - O(1/T)$，等价于 UCB 式的乐观性。
- **Arm-specific epoch 结构**：第 $r$ 个 epoch 使用最近 $2^r$ 个观测更新经验均值，确保每个观测仅被使用一次。

### 隐私分析

- 每轮 TS-Gaussian 阶段的隐私为 $\sqrt{1/\ln^\alpha(T)}$-GDP。
- $\phi$ 轮组合后为 $\sqrt{\phi/\ln^\alpha(T)}$-GDP (Theorem 4.7)。
- UCB 阶段为后处理，不增加隐私损失 (Theorem 4.8)。
- **总体 (Theorem 4.4)**：$\sqrt{2c_0 T^{0.5(1-\alpha)} \ln^{1.5(1-\alpha)}(T)}$-GDP。

## 实验关键数据

### 隐私-遗憾权衡总结

| 算法 | 遗憾上界 | GDP 保证 |
|---|---|---|
| TS-Gaussian | $O(K\ln(T\Delta^2)/\Delta)$ | $O(T^{0.5})$ |
| M-TS-Gaussian (调参 $b,c=\ln^\alpha T$) | $O(K\ln^\alpha(T)\ln(T\Delta^2)/\Delta)$ | $O(T^{0.5}/\ln^\alpha T)$ |
| M-TS-Gaussian (调参 $b,c=T^\gamma$) | $O(KT^\gamma\ln(T\Delta^2)/\Delta)$ | $O(T^{0.5-\gamma})$ |
| **DP-TS-UCB ($\alpha=0$)** | $O(K\ln(T^{1.5}\Delta^2)/\Delta)$ | $\tilde{O}(T^{0.25})$ |
| **DP-TS-UCB ($\alpha=1$)** | $O(K\ln(T\Delta^2)\ln T/\Delta)$ | $O(1)$ |

### 关键发现

- $\alpha=0$ 时：遗憾接近最优（仅多 $\log^{0.5}$ 因子），GDP 从 $O(\sqrt{T})$ 降至 $\tilde{O}(T^{0.25})$，显著改善。
- $\alpha=1$ 时：实现常数 GDP，遗憾仅多一个 $\ln T$ 因子，意味着增加时间范围不增加隐私成本。
- 与 M-TS-Gaussian 相比，在相同遗憾量级下 GDP 改善幅度更大——M-TS-G 最多只能将 $\sqrt{T}$ 缩减为 $\sqrt{T}/\text{polylog}$，而 DP-TS-UCB 直接降至 $T^{0.25}$ 量级。
- (ε,δ)-DP 转换 (Theorem 4.6)：通过 Theorem 2.4 的原始-对偶转换，$\phi$ 越小 $\delta(\varepsilon)$ 越小。

## 亮点与洞察

1. **TS-UCB 连接的理论价值**：揭示了 $\phi$ 个高斯采样的最大值等价于 UCB 的乐观上界，桥接了随机探索与确定性探索的理论。
2. **平滑可调的隐私-遗憾权衡**：单一参数 $\alpha$ 控制从最优遗憾到常数隐私的连续谱。
3. **实用的 epoch 结构**：arm-specific epoch 结构不仅是分析工具，更是减少隐私损失的核心机制。
4. **GDP 框架的优势**：高斯差分隐私的组合定理使分析更紧致、更自然。
5. **最坏情况遗憾**：DP-TS-UCB 的最坏情况遗憾为 $O(\sqrt{KT}\ln^{0.5(1+\alpha)}(T))$，接近 $O(\sqrt{KT\ln K})$ 的最优界。
6. **后处理保持隐私**：UCB 阶段使用已有采样的最大值，属于后处理，不增加隐私成本——这是算法设计的关键技巧。

### Worst-case 遗憾详细对比

| 算法 | Worst-case 遗憾 | GDP |
|---|---|---|
| TS-Gaussian | $O(\sqrt{KT\ln K})$ | $O(\sqrt{T})$ |
| DP-TS-UCB ($\alpha=0$) | $O(\sqrt{KT\ln T})$ | $\tilde{O}(T^{0.25})$ |
| DP-TS-UCB ($\alpha=1$) | $O(\sqrt{KT}\ln T)$ | $O(1)$ |

## 局限与展望

- 仅考虑随机多臂老虎机，未推广到线性/上下文老虎机或对抗环境。
- 未提供新的下界结果，现有上界与已知下界之间可能仍有间隙。
- $\phi$ 的选择依赖于 $T$，实际中 $T$ 可能未知，需要 doubling trick 等技术。
- 实验部分较弱，主要以理论分析为主，未在实际数据集上验证。
- 对 $[0,1]$ 支撑的奖励分布假设在实践中可能受限。
- 未讨论与 local DP 或 shuffle DP 的关系。

## 相关工作与启发

- **TS-Gaussian (Agrawal & Goyal, 2017)**、**UCB1 (Auer et al., 2002)**：本文的两个基石算法。
- **Ou et al. (2024)**：首次将 TS-Gaussian 与 GDP 联系，证明其为 $O(\sqrt{T})$-GDP。本文在其基础上通过限制采样次数将 GDP 降至 $T^{0.25}$ 量级。
- **Sajed & Sheffet (2019)、Azize & Basu (2022)**：开创 epoch 结构用于隐私在线学习。
- **Hu et al. (2021)、Hu & Hegde (2022)**：设计最优 $(\varepsilon,0)$-DP 多臂老虎机算法。
- **下界参考**：Shariff & Sheffet (2018) 的 $\Omega(\sum \ln(T)/\Delta_i + K\ln(T)/\varepsilon)$ 与本文上界兼容。
- **启发**：TS 采样最大值 = UCB 上界的洞察可推广到更广泛的贝叶斯优化场景。

## 评分

- 新颖性: ⭐⭐⭐⭐ — TS与UCB之间的联系洞察具有独立研究价值
- 实验充分度: ⭐⭐⭐ — 以理论为主，实验验证有限
- 写作质量: ⭐⭐⭐⭐ — 表述清晰，proof sketch 简洁
- 价值: ⭐⭐⭐⭐ — 为隐私在线学习提供了更优的privacy-regret trade-off

<!-- RELATED:START -->

## 相关论文

- [Empirical Privacy Variance](empirical_privacy_variance.md)
- [Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off](clients_collaborate_flexible_differentially_private_federated_learning_with_guar.md)
- [Activation Space Interventions Can Be Transferred Between Large Language Models](activation_space_interventions_can_be_transferred_between_large_language_models.md)
- [On Differential Privacy for Adaptively Solving Search Problems via Sketching](on_differential_privacy_for_adaptively_solving_search_problems_via_sketching.md)
- [Cape: Context-Aware Prompt Perturbation Mechanism with Differential Privacy](cape_context-aware_prompt_perturbation_mechanism_with_differential_privacy.md)

<!-- RELATED:END -->
