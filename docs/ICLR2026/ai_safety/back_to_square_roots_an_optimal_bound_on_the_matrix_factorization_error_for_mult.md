---
title: >-
  [论文解读] Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch Differentially Private SGD
description: >-
  [ICLR 2026][AI安全][differential privacy] 提出 Banded Inverse Square Root (BISR) 矩阵分解方法，通过对逆相关矩阵（而非相关矩阵本身）施加带状结构，首次在多轮参与差分隐私 SGD 中实现渐近最优的分解误差界…
tags:
  - "ICLR 2026"
  - "AI安全"
  - "differential privacy"
  - "matrix factorization"
  - "DP-SGD"
  - "multi-epoch participation"
  - "banded factorization"
  - "optimal error bounds"
---

# Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch Differentially Private SGD

**会议**: ICLR 2026  
**arXiv**: [2505.12128](https://arxiv.org/abs/2505.12128)  
**代码**: 无（使用 jax-privacy 库进行基线比较）  
**领域**: AI安全 / 差分隐私  
**关键词**: differential privacy, matrix factorization, DP-SGD, multi-epoch participation, banded factorization, optimal error bounds

## 一句话总结
提出 Banded Inverse Square Root (BISR) 矩阵分解方法，通过对逆相关矩阵（而非相关矩阵本身）施加带状结构，首次在多轮参与差分隐私 SGD 中实现渐近最优的分解误差界，并配套低存储优化变体 BandInvMF。

## 研究背景与动机
**领域现状**：矩阵分解机制（Matrix Factorization Mechanism）是差分隐私训练中通过注入相关噪声来提升模型效用的重要方法，已被 Google 用于生产级 on-device 语言模型训练。

**现有痛点**：在多轮训练（multi-epoch）中，同一数据点被多次使用，需要刻画分解误差与参与次数的关系。但现有上下界之间存在显著差距——Banded Square Root (BSR) 的误差界中对带宽 $p$ 的依赖是隐式的，无法判断其是否最优。

**核心矛盾**：理论上不清楚多轮参与下分解误差的最优增长率是什么，实践中缺少既高效又有显式误差刻画的分解方法。

**本文目标** 给出多轮参与下矩阵分解误差的紧界（tight bound），并提供一个计算高效、理论最优的显式分解方法。

**切入角度**：不是像 BSR 那样让相关矩阵 $C$ 带状化，而是让 $C^{-1}$ 带状化——这一视角转换带来了显式误差刻画和高效实现的双重优势。

**核心 idea**：在逆相关矩阵上施加带状结构，使噪声注入可通过卷积高效实现，同时获得关于带宽的显式最优误差界。

## 方法详解

### 整体框架
矩阵分解机制把要发布的工作负载矩阵 $A$ 拆成 $A = BC$，在私有端注入相关噪声后得到无偏估计 $\widehat{AX} = B(CX + Z)$，整套方法的好坏全压在分解误差 $\mathcal{E}(B,C)$ 上——它由 $\|B\|_F$ 与 $C$ 的列灵敏度共同决定。本文不在相关矩阵 $C$ 上做文章，而是反过来让 $C^{-1}$ 带状化：先给出兼具显式最优误差界与卷积式高效实现的 BISR 分解，再用配套的下界证明它在多轮参与下渐近最优，最后给出更省存储的数值优化变体 BandInvMF。

### 关键设计

**1. BISR 分解：让逆相关矩阵带状，把噪声注入变成卷积。**

已有的 BSR 直接对相关矩阵 $C$ 截断成带状，结果是误差里对带宽 $p$ 的依赖被埋在闭式解内部、看不出是否最优。BISR 换了个落点：从工作负载 $A$ 出发先取其正定平方根 $C$（满足 $C^2 = A$），再去截断它的逆 $C^{-1}$ 为 $p$-带状矩阵，最后重新求逆得到 $C^p$，从而给出分解 $A = B^p C^p$（Definition 1）。关键好处在于，$C^{-1}$ 带状意味着每一步私有更新只需把当前梯度与 $p$ 个系数做一次卷积，可直接用 FFT 加速，而不需要像 BandMF 那样在线求解优化问题。这一步看似只是把"截断对象"从 $C$ 换成 $C^{-1}$，却同时打开了显式误差刻画和廉价实现两条路。

**2. 匹配的下界：钉死多轮参与下误差的最优增长率。**

要谈"最优"，先得知道任何分解都逃不掉的误差下限是多少。Theorem 3 用概率方法构造参与向量并界定其范数，给出与参与次数 $k$、矩阵规模 $n$ 相关的紧下界：在无 weight decay（$\alpha = 1$）时为 $\Omega(\sqrt{k}\log n + k)$，在带 weight decay（$\alpha < 1$）时收紧为 $\Omega_\alpha(\sqrt{k})$。这条下界本身就是本文的核心理论贡献——它第一次明确了多轮参与下分解误差关于 $\sqrt{k}$ 的"平方根"标度，也正是标题"Back to Square Roots"的来历。

**3. BISR 上界与最优带宽：上下界对齐证渐近最优。**

有了下界，剩下的就是证 BISR 能贴着它。Theorem 4 给出 BISR 误差关于带宽 $p$、参与次数 $k$、矩阵大小 $n$ 和分离参数 $b$ 的显式上界——正因为带状落在 $C^{-1}$ 上，这个上界是写得出来的，而不像 BSR 那样隐式。把带宽取到最优的 $p^* = O(b \log b)$ 后，上界在阶的意义上正好匹配 Theorem 3 的下界，于是 BISR 被证明在多轮参与场景下渐近最优。这就闭合了 BSR 遗留的上下界差距。

**4. BandInvMF：放弃闭式解、改用数值优化的省存储变体。**

BISR 的系数来自平方根的闭式构造，在有限规模下不一定是字面最优。BandInvMF 保留"逆矩阵带状 + Toeplitz"这套结构，但把系数交给数值优化去拟合而非套闭式公式，并以 BISR 的系数作初始化，约 20 步即收敛。收益是即使只用单条带宽，误差也能从普通分解的 $O(\sqrt{n})$ 压到 $O(n^{1/4})$，在低存储 regime 下显著改善，同时仍只需缓存少量系数。

**5. 算法实现：$p$ 个缓冲向量跑完整条训练。**

落到 Algorithm 1，整套机制的在线开销就是一个长度 $p$ 的噪声缓冲区：第 $i$ 步的私有梯度按 $\hat{x}_i = x_i + \zeta \sum_{t=0}^{\min(p,i)-1} c_t Z_{i-t}$ 注入相关噪声，即把最近 $p$ 个噪声样本用系数 $c_t$ 加权叠加。因为只触碰梯度的线性变换，它天然兼容 momentum 与 weight decay，存储和计算成本都远低于需要在线解优化的 BandMF。

## 实验关键数据

### 表1：CIFAR-10 测试精度（(9, 10⁻⁵)-DP，10 epochs）

| 方法 | Epoch 1 | Epoch 5 | Epoch 10 |
|------|---------|---------|----------|
| DP-SGD (Amp.) | 12.7±2.2 | 39.8±1.2 | 44.6±0.7 |
| BSR (Amp.) | 28.3±0.7 | 48.0±2.0 | 49.8±0.3 |
| **BISR (Amp.)** | **32.3±0.7** | **52.8±2.0** | **61.8±0.3** |
| Band-MF (Amp.) | 27.7±2.0 | 46.8±0.8 | 50.0±0.4 |
| Band-Inv-MF (Amp.) | 23.6±2.8 | 48.6±1.0 | 57.4±1.2 |
| DP-SGD (Non-Amp.) | 19.5±3.0 | 37.7±1.2 | 39.0±0.7 |
| **BISR (Non-Amp.)** | **31.8±1.5** | **51.1±1.0** | **56.2±0.2** |

### 表2：RMSE 比较（矩阵分解误差，n=16384）

| 方法 | k=4, α=1,β=0 | k=16, α=1,β=0 | k=16, α=1,β=0.9 |
|------|---------------|----------------|------------------|
| BSR | 与 BISR 相当 | 明显差于 BISR | 差于 BISR |
| BLT | 与 BISR 相当 | 与 BISR 相当 | 仅支持 prefix-sum |
| BandMF | 略优（小矩阵） | 略优但不可扩展 | 计算成本过高 |
| **BISR** | **最优或接近最优** | **显著优于 BSR** | **一致性最佳** |

> BISR 在 k=16 高参与次数下优势尤为明显；BandMF 虽 RMSE 略低但不可扩展至 n>4096

## 亮点与洞察
- **视角转换的力量**：从"让 $C$ 带状化"转为"让 $C^{-1}$ 带状化"，获得了显式误差刻画——这种看似微小的结构改变带来了理论突破。
- **理论与实践闭环**：BISR 同时实现了理论最优性（上下界匹配）和实践竞争力（与 BLT/BandMF 精度相当），且实现极其简单（卷积操作）。
- **低存储 regime 的洞察**：RMSE 更低不等于模型精度更高——Band-Inv-MF 的 RMSE 优于 BISR，但两者训练精度相近，说明分解误差与模型效用的关系非简单单调。
- **实用性突出**：仅需 $p$ 个系数的卷积，存储和计算成本远低于需要求解优化问题的 BandMF。

## 局限性
1. **渐近最优 ≠ 有限规模最优**：BISR 在渐近意义上最优，但有限矩阵大小下 BandMF 等数值优化方法仍可能略优。
2. **常数因子未优化**：上下界匹配在阶（order）的意义上，常数项的差距尚未完全消除。
3. **RMSE-精度脱节**：更低的分解 RMSE 不一定转化为更高的模型精度，特别是在使用 amplification by subsampling 时。
4. **BLT 比较受限**：BLT 仅实现了 prefix-sum 矩阵，无法在 momentum/weight decay 设置下对比。

## 补充实验：IMDB 情感分析（BERT-base）
- 在 (9, 10⁻⁵)-DP 下微调 BERT-base，BISR 在 amplified 和 non-amplified 设置下均优于 BSR 和 Band-MF
- BISR (Amplified) 10 epoch 后显著领先 DP-SGD，体现矩阵分解机制的优势
- 低存储 regime 下 Band-Inv-MF 与 BISR 精度接近，但 BISR 无需优化求解

## 相关工作
- **矩阵分解机制**：Choquette-Choo et al. (2023a) 定义了多轮参与下的最优分解问题；BLT (Dvijotham et al., 2024) 提供了 buffer-based 方法；BandMF (McKenna, 2025) 通过数值优化求解最优带状分解。
- **平方根分解**：Henzinger et al. (2024) 提出，Kalinin & Lampert (2024) 扩展为 BSR 并建立了首个上下界，但带宽依赖不显式。
- **隐私会计**：本文使用 MCMC accountant (Choquette-Choo et al., 2024b) 和 bins-and-balls 子采样 (Chua et al., 2025) 进行隐私分析。
- **联邦学习中的 MF**：Zhang et al. (2025) 和 Bienstock et al. (2025) 将矩阵分解扩展到联邦学习场景。

## 评分
- **创新性**: ★★★★☆ — 逆矩阵带状化的视角转换优雅且有效，配套的理论紧界具有重要贡献。
- **实用性**: ★★★★☆ — 实现简单高效，卷积操作可并行化，已有 JAX 实现。
- **理论深度**: ★★★★★ — 闭合了多轮参与分解误差的理论差距，上下界渐近匹配。
- **实验充分性**: ★★★★☆ — RMSE 和训练精度双重评估，覆盖多种优化器设置和数据集，但大规模 LLM 实验缺失。
- **表达清晰度**: ★★★★☆ — 数学推导严谨，算法描述清晰，Figure 1 的可视化很直观。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Unified Privacy Guarantees for Decentralized Learning via Matrix Factorization](unified_privacy_guarantees_for_decentralized_learning_via_matrix_factorization.md)
- [\[ICLR 2026\] Skirting Additive Error Barriers for Private Turnstile Streams](skirting_additive_error_barriers_for_private_turnstile_streams.md)
- [\[AAAI 2026\] An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses](../../AAAI2026/ai_safety/an_improved_privacy_and_utility_analysis_of_differentially_p.md)
- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](../../NeurIPS2025/ai_safety/differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)
- [\[ICML 2026\] PRISM: Gauge-Invariant Tangent-Space Differentially Private LoRA](../../ICML2026/ai_safety/prism_gauge-invariant_tangent-space_differentially_private_lora.md)

</div>

<!-- RELATED:END -->
