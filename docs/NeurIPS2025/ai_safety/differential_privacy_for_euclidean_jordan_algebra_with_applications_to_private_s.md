---
title: >-
  [论文解读] Differential Privacy for Euclidean Jordan Algebra with Applications to Private Symmetric Cone Programming
description: >-
  [NEURIPS2025][AI安全][differential privacy] 提出了基于 Euclidean Jordan Algebra (EJA) 的通用 Gaussian 隐私机制，并在此基础上设计了首个差分隐私的 Symmetric Cone Programming (SCP) 求解算法…
tags:
  - "NEURIPS2025"
  - "AI安全"
  - "differential privacy"
  - "Euclidean Jordan Algebra"
  - "Symmetric Cone Programming"
  - "Semidefinite Programming"
  - "Gaussian Mechanism"
---

# Differential Privacy for Euclidean Jordan Algebra with Applications to Private Symmetric Cone Programming

**会议**: NEURIPS2025  
**arXiv**: [2509.16915](https://arxiv.org/abs/2509.16915)  
**代码**: 无  
**领域**: AI安全  
**关键词**: differential privacy, Euclidean Jordan Algebra, Symmetric Cone Programming, Semidefinite Programming, Gaussian Mechanism

## 一句话总结

提出了基于 Euclidean Jordan Algebra (EJA) 的通用 Gaussian 隐私机制，并在此基础上设计了首个差分隐私的 Symmetric Cone Programming (SCP) 求解算法，解决了 Hsu et al. (ICALP 2014) 提出的关于差分隐私半定规划的重要开放问题。

## 背景与动机

差分隐私 (Differential Privacy, DP) 是数据隐私保护的事实标准，在机器学习中应用广泛。现有的隐私算法大多针对具体问题逐案设计，或者通过 DP-SGD 等通用方法实现。然而，在凸优化领域，已有的 DP 算法主要局限于**线性规划 (LP)** 的场景。

Symmetric Cone Programming (SCP) 是一类非常广泛的凸优化问题，涵盖了线性规划 (LP)、二阶锥规划 (SOCP) 和半定规划 (SDP)。SCP 在机器学习中有大量应用，包括支持向量机、矩阵补全、鲁棒均值/协方差估计、实验设计和稀疏 PCA 等。尽管 SCP 的算法设计已经很成熟，但其差分隐私算法几乎空白。特别是，**设计差分隐私的 SDP 算法** 被 Hsu et al. (2014) 提出为一个重要的开放问题，十余年未被解决。

核心挑战在于：SCP 的优化变量（如对称矩阵）具有丰富的代数和几何结构，传统逐元素加噪的 Laplace 机制无法捕捉这些结构（如谱结构），导致隐私-效用权衡很差。

## 核心问题

1. 如何为输出在 Euclidean Jordan Algebra 中的函数设计通用的差分隐私机制？特别是当灵敏度用 $\ell_1$、$\ell_2$、$\ell_\infty$ 范数度量时。
2. 如何在不同的隐私设置（高灵敏度约束隐私、低灵敏度约束/标量/目标隐私）下，设计差分隐私的 SCP 求解算法？
3. 能否解决差分隐私 SDP 这一长期悬而未决的开放问题？

## 方法详解

### 1. Euclidean Jordan Algebra 上的通用 Gaussian 机制

EJA 是一类有限维向量空间，配备了 Jordan 乘积和内积。它统一了 $\mathbb{R}^k$、实对称矩阵空间等重要结构。每个 EJA 元素都有谱分解 $x = \sum_{i=1}^r \lambda_i q_i$（类似矩阵的特征值分解），其中 $r$ 是 EJA 的秩，$k$ 是维度。

**核心思路**：利用 EJA 到 $\mathbb{R}^k$ 的等距同构 $\phi$，将 Gaussian 噪声从 $\mathbb{R}^k$ 映射回 EJA。

- 设 $f: \mathcal{D} \to \mathcal{J}$ 的 $\ell_2$ 灵敏度为 $\Delta_2$
- 生成 $\nu \sim \mathcal{N}(0, \sigma^2 I_k)$，其中 $\sigma = \Delta_2 \sqrt{2\log(1.25/\delta)} / \epsilon$
- 输出 $f(D) + \phi^{-1}(\nu)$

由于 $\phi$ 保持内积和 $\ell_2$ 范数，隐私性直接由标准 Gaussian 机制保证。对于 $\ell_1$ 和 $\ell_\infty$ 灵敏度，通过范数不等式进行归约。

**关键洞察**：不能只对特征值加噪而忽略 Jordan 框架——仅扰动谱不足以保证差分隐私，还需要随机化基底（Jordan frame）。这解释了为什么需要在全 $k$ 维空间加噪，而非仅在 $r$ 维谱空间加噪。

### 2. 高灵敏度约束隐私下的 SCP 求解

在此设置中，两个相邻数据库的 SCP 实例可能完全增加或删除一个约束。算法基于 **Dense Multiplicative Weights Update (MWU)** 框架：

- 在约束空间上运行 MWU，输出满足大部分约束的解
- Oracle 通过 **Exponential Mechanism** 实现隐私保护
- 对于 Covering SDP，利用 $\gamma$-net 对正半定锥的极端射线进行离散化，将不可数的可行解空间控制在 $\exp(r)$ 量级
- 结果：至多 $s = \Omega(r/\epsilon \cdot \text{polylog})$ 个约束被违反，违反量为 $O(\text{OPT})$

### 3. 低灵敏度约束隐私下的 SCP 求解

在此设置中，相邻数据库的约束矩阵在 $\ell_\infty$ 范数下接近。算法（Algorithm 1）在变量空间上运行 MWU：

- 维护分布元素 $x^t \in \mathcal{K}$，初始化为 $e/r$
- 每轮通过 Exponential Mechanism 找到近似最大违反约束
- 对返回的约束元素施加 **Generic Gaussian Mechanism** 加噪
- 用加噪后的损失进行 MWU 更新：$x^{t+1} \propto \exp(-\sum_i \eta \hat\ell^i)$
- 输出所有迭代的均值 $\bar{x} = \frac{1}{T}\sum_t x^t$

约束违反界为：

$$\alpha = \widetilde{O}\left(\frac{\Delta_\infty^{1/2} r^{1/4} k^{1/4}}{\epsilon^{1/2}} \cdot \text{polylog}(r, 1/\beta, 1/\delta)\right)$$

相比 LP 的结果，多出了 $k^{1/4}$ 因子，来自于需要在完整 $k$ 维空间加 Gaussian 噪声（而 LP 只需逐元素加 Laplace 噪声）。

### 4. 标量隐私与目标隐私

作为低灵敏度框架的变体，文章还处理了右端向量 $b$ 和目标函数 $c$ 的隐私保护设置，算法结构类似。

## 实验关键数据

本文为纯理论工作，未包含数值实验。主要结果均为定理形式的隐私和效用保证：

| 设置 | 约束违反量 $\alpha$ | 隐私保证 |
|------|---------------------|----------|
| 高灵敏度约束隐私 (Covering SCP) | $O(\text{OPT})$，至多 $\Omega(r/\epsilon \cdot \text{polylog})$ 个约束 | $\epsilon$-DP |
| 低灵敏度约束隐私 | $\widetilde{O}(\Delta_\infty^{1/2} r^{1/4} k^{1/4} / \epsilon^{1/2})$ | $(\epsilon,\delta)$-DP |
| 标量/目标隐私 | 类似低灵敏度约束结果 | $(\epsilon,\delta)$-DP |

## 亮点

1. **解决重要开放问题**：首次给出差分隐私的 SDP 算法，解决了 Hsu et al. (ICALP 2014) 提出的十余年未解的开放问题
2. **统一框架**：通过 EJA 统一了 LP、SOCP、SDP 的差分隐私处理，而非逐案设计
3. **深刻的不可能性洞察**：仅扰动特征值不足以保证隐私、$\ell_1$ 范数 Laplace 机制在 EJA 中失效——这些负面结果非常有指导意义
4. **Noisy MWU 框架**：将隐私噪声注入 MWU 的 oracle 中形成 noisy MWU，可推广到其他需要不同误差保证的场景
5. **$\gamma$-net 技术**：对正半定锥和一般 EJA 中 primitive idempotent 射线进行 $\gamma$-net 量化，这一技术可能有独立价值

## 局限与展望

1. **隐私分析工具较基础**：仅使用了经典的 Gaussian 机制和 Advanced Composition，未利用 Rényi DP 或 Moments Accountant 等更紧的分析工具
2. **约束不完全满足**：在高灵敏度设置下，输出解会违反少量约束；在低灵敏度设置下有 $\alpha$ 量级的约束松弛。能否精确满足所有约束仍是开放问题
3. **$k^{1/4}$ 额外因子**：低灵敏度约束隐私结果比 LP 多一个 $k^{1/4}$ 因子，来源于必须在全维空间加噪。能否通过更精细的噪声设计缩小此差距是有趣的方向
4. **无实验验证**：纯理论工作，缺少在实际 ML 任务（如鲁棒估计、实验设计）上的数值验证
5. **仅处理简单 EJA**：Covering SCP 结果限于简单 EJA，对一般 EJA（直和分解）的处理有待扩展

## 与相关工作的对比

| 工作 | 范围 | 灵敏度度量 | 噪声机制 |
|------|------|-----------|----------|
| Hsu et al. (2014) | LP | 多种设置 | Laplace / Exponential |
| Nguyen & Silberstein (2024) | LP | 标量隐私 | Laplace（精确约束满足） |
| Benvenuti et al. (2024/2025) | LP | 多种设置 | 机会约束/重构法 |
| **本文** | **SCP (含 LP/SOCP/SDP)** | **$\ell_1/\ell_2/\ell_\infty$** | **Generic Gaussian on EJA + Exponential** |

本文的关键区别在于：(1) 将范围从 LP 大幅扩展到 SCP；(2) 引入谱感知的灵敏度度量而非逐元素度量；(3) 通过 EJA 的代数结构提供统一处理。

## 启发与关联

- **EJA 框架的可迁移性**：EJA 上的 Gaussian 机制可推广到其他需要对矩阵/锥约束数据进行隐私保护的场景，如联邦学习中的协方差矩阵聚合
- **Noisy Oracle 范式**：将隐私噪声注入优化 oracle 而非最终解的思路，可能启发其他隐私优化问题的算法设计
- **DP + 凸优化的交叉方向**：本文打开了 DP-SDP 的大门，后续可深入到具体应用（鲁棒估计、实验设计、稀疏 PCA）并追求更紧的界

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (解决十余年开放问题，框架高度原创)
- 实验充分度: ⭐⭐ (纯理论工作，无实验)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，动机阐述充分)
- 价值: ⭐⭐⭐⭐⭐ (开创 DP-SCP 方向，影响深远)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Sequentially Auditing Differential Privacy](sequentially_auditing_differential_privacy.md)
- [\[NeurIPS 2025\] Differentially Private High-dimensional Variable Selection via Integer Programming](differentially_private_high-dimensional_variable_selection_via_integer_programmi.md)
- [\[NeurIPS 2025\] Spectral Perturbation Bounds for Low-Rank Approximation with Applications to Privacy](spectral_perturbation_bounds_for_low-rank_approximation_with_applications_to_pri.md)
- [\[NeurIPS 2025\] Mitigating Privacy-Utility Trade-off in Decentralized Federated Learning via f-Differential Privacy](mitigating_privacy-utility_trade-off_in_decentralized_federated_learning_via_f-d.md)
- [\[NeurIPS 2025\] Multi-Class Support Vector Machine with Differential Privacy](multi-class_support_vector_machine_with_differential_privacy.md)

</div>

<!-- RELATED:END -->
