---
title: >-
  [论文解读] Statistical and Computational Guarantees of Kernel Max-Sliced Wasserstein Distances
description: >-
  [ICML2025][优化][Wasserstein距离] 本文为 Kernel Max-Sliced (KMS) Wasserstein 距离提供了**尖锐的有限样本统计保证**（无维度依赖、收敛速率 $n^{-1/(2p)}$）和**计算保证**（证明精确计算是 NP-hard 后提出高效的半定松弛 SDR 及一阶算法），并在高维两样本检验、人体活动检测和生成建模上验证了优越性能。
tags:
  - ICML2025
  - 优化
  - Wasserstein距离
  - 核方法
  - 半定松弛
  - 样本复杂度
  - 非参数检验
  - 维度灾难
---

# Statistical and Computational Guarantees of Kernel Max-Sliced Wasserstein Distances

**会议**: ICML2025  
**arXiv**: [2405.15441](https://arxiv.org/abs/2405.15441)  
**代码**: 未公开  
**领域**: 优化 / 最优传输  
**关键词**: Wasserstein距离, 核方法, 半定松弛, 样本复杂度, 非参数检验, 维度灾难

## 一句话总结

本文为 Kernel Max-Sliced (KMS) Wasserstein 距离提供了**尖锐的有限样本统计保证**（无维度依赖、收敛速率 $n^{-1/(2p)}$）和**计算保证**（证明精确计算是 NP-hard 后提出高效的半定松弛 SDR 及一阶算法），并在高维两样本检验、人体活动检测和生成建模上验证了优越性能。

## 研究背景与动机

- **维度灾难**：经典 Wasserstein 距离的样本复杂度随数据维度 $d$ 指数增长，限制了高维应用。
- **降维策略**：Sliced Wasserstein 取随机一维投影的均值，信息量有限；Max-Sliced (MS) Wasserstein 寻找最优**线性**投影方向，但对非线性结构的数据区分力不足。
- **KMS Wasserstein 的提出**：Wang et al. (2022a) 将投影函数从线性扩展到 RKHS 球中的最优**非线性**映射，即 Kernel Max-Sliced Wasserstein 距离。当核取点积核时退化为 MS Wasserstein；取高斯核时函数类具有万能逼近性质。
- **已有不足**：(1) 统计收敛率要求投影 Poincaré 不等式等难以验证的正则条件；(2) 计算仅有梯度下降的局部最优，质量无理论保证且对初始化敏感。

## 方法详解

### 核心定义

给定 RKHS $\mathcal{H}$ 及对应核 $K$，KMS $p$-Wasserstein 距离定义为：

$$\mathcal{KMS}_p(\mu,\nu) = \max_{f\in\mathcal{H},\|f\|_{\mathcal{H}}\le 1} W_p(f_\#\mu, f_\#\nu)$$

其中 $f_\#\mu$ 为 $\mu$ 经 $f:\mathcal{B}\to\mathbb{R}$ 的推前测度。利用再生性质 $f(x)=\langle f, K_x\rangle_{\mathcal{H}}$，可将 KMS 等价改写为将数据先经特征映射 $\Phi(x)=K_x$ 嵌入 $\mathcal{H}$，然后在希尔伯特空间上做 MS Wasserstein：

$$\mathcal{KMS}_p(\mu,\nu) = \mathcal{MS}_p(\Phi_\#\mu, \Phi_\#\nu)$$

### 统计保证（Theorem 3.2）

**条件**：仅需核满足 $\sqrt{K(x,x)}\le A$（高斯核自然满足）。

- **单样本**：$\Pr[\mathcal{KMS}_p(\hat\mu_n, \mu) \le \frac{1}{2}\Delta(n,\alpha)] \ge 1-\alpha$
- **双样本**：$\Pr[\mathcal{KMS}_p(\hat\mu_n, \hat\nu_n) \le \mathcal{KMS}_p(\mu,\nu) + \Delta(n,\alpha)] \ge 1-\alpha$

其中临界值 $\Delta(n,\alpha) = 4A(C+4\sqrt{\log(2/\alpha)})^{1/p} \cdot n^{-1/(2p)}$。

**关键特性**：收敛速率 $O(n^{-1/(2p)})$ **无维度依赖**，且在最坏情况下是最优的（一维点积核时与经典 $W_p$ 一致）。

### 计算保证

#### NP-hardness（Theorem 4.2）

利用表示定理将 KMS 的函数优化转化为有限维问题（式9）：

$$\max_{\omega\in\mathbb{R}^{2n},\|\omega\|_2=1} \min_{\pi\in\Gamma_n} \sum_{i,j}\pi_{i,j}(M_{i,j}^T\omega)^2$$

证明该 max-min 问题在最坏情况下可规约到 fair-PCA / fair beamforming 问题，从而为 NP-hard。

#### 半定松弛 SDR

令 $S=\omega\omega^T$，放松秩-1约束得到 SDP：

$$\max_{S\in\mathcal{S}_{2n}} F(S), \quad F(S)=\min_{\pi\in\Gamma_n}\sum_{i,j}\pi_{i,j}\langle M_{i,j}M_{i,j}^T, S\rangle$$

#### 一阶不精确镜像上升算法（Algorithm 1）

- 用 Katyusha 动量随机梯度法解内层 OT 问题得到近似 $\hat\pi$，构造超梯度 $v(S_k)=\sum_{i,j}\hat\pi_{i,j}M_{i,j}M_{i,j}^T$
- 在谱面体 $\mathcal{S}_{2n}$ 上做 von Neumann 熵镜像上升，更新有闭式解（矩阵指数）
- **复杂度**（Theorem 4.4）：$\tilde{O}(n^2\delta^{-2}\cdot\max(n,\delta^{-1}))$，远优于内点法的 $\tilde{O}(n^{6.5})$

#### 秩约界与秩缩减（Theorem 4.5 & 4.6）

- SDR 存在最优解秩至多为 $k=1+\lfloor\sqrt{2n+9/4}-3/2\rfloor$，远小于平凡上界 $2n$
- 设计了基于 Hungarian + 零空间搜索的秩缩减算法，将近似解投影到低秩空间，复杂度 $O(n^5)$
- 松弛间隙（Theorem 4.7）：$\varepsilon n^{-4}\cdot\text{Optval(SDR)}\le\text{Optval(KMS)}\le\text{Optval(SDR)}$

## 实验关键数据

### 计算效率与解质量（Fig. 3）

| 方法 | 计算时间 | 解质量（KMS $W_2$ 估计值） |
|------|---------|------------------------|
| SDR-Efficient（本文） | 中等，大规模优势明显 | 均值最大、方差最小 |
| BCD（Wang et al. 2022a） | 小规模略快，大规模超时 | 依赖初始化，方差大 |
| SDR-IPM（内点法） | 小规模即极慢 | 与 SDR-Efficient 一致 |

测试场景：100维合成高斯、MNIST、CIFAR-10。

### 高维两样本检验（Fig. 4）

| 场景 | 最优方法 | KMS 表现 |
|------|---------|---------|
| 高斯协方差偏移 | MS（线性最优） | 次优 |
| 高斯混合分布偏移 | **KMS** | 最优 |
| MNIST 分布丰度变化 | **KMS** | 最优 |
| CIFAR-10 分布丰度变化 | **KMS** | 最优 |

### 人体活动检测（Table 1）

| 方法 | 检测延迟均值 | 标准差 |
|------|-----------|-------|
| **KMS** | **11.4** | **5.56** |
| GSW | 12.9 | 6.4 |
| Sinkhorn Div | 16.5 | 4.4 |
| SW | 17.2 | 8.7 |
| MS | 17.8 | 9.2 |
| MMD | 50.6 | 39.5 |

### 生成建模 FID 分数（Fig. 6）

| 方法 | FID ↓ |
|------|------|
| **KMS Wasserstein** | **105.98** |
| MMD | 113.08 |
| Sinkhorn Div | 114.36 |
| SW | 113.92 |
| MS | 115.21 |
| GSW | 128.07 |

## 亮点与洞察

1. **统计-计算双轨保证**：在同一框架下同时给出尖锐的统计收敛率和高效计算方案，这在 OT 变体文献中罕见。
2. **无维度依赖**：收敛率 $O(n^{-1/(2p)})$ 仅依赖样本量，完全避免维度灾难，且条件（核有界）极其温和。
3. **NP-hard + 高质量松弛**：先证明精确计算的不可行性，再给出带理论保证的凸松弛，包括秩约界和松弛间隙。
4. **实用一阶算法**：复杂度从内点法的 $\tilde{O}(n^{6.5})$ 降至 $\tilde{O}(n^2\delta^{-3})$，使得中等规模问题可解。
5. **非线性投影的自适应性**：KMS 在包含非线性结构的场景（混合分布、图像数据）中显著优于线性投影的 MS。

## 局限与展望

- **松弛间隙较保守**：Theorem 4.7 给出的比值含 $n^{-4}$ 因子，理论保证偏悲观，实际表现好得多。
- **秩缩减代价**：$O(n^5)$ 的秩缩减算法在大规模时仍是瓶颈。
- **仅处理 $p=2$**：计算保证（SDR 及算法）仅针对 2-Wasserstein，未覆盖 $p=1$ 等情形。
- **核与带宽选择**：实验统一用高斯核 + 中位数带宽启发式，对核选择的理论指导缺失。
- **高维实验规模有限**：MNIST/CIFAR-10 上样本量受限于计算预算（1小时），未测试更大规模。
- **与 MMD 的检验力对比**：虽然 KMS 更灵活，但 MMD 计算 $O(n^2)$ 远更高效，实际应用中的性价比需权衡。

## 相关工作与启发

- **MS Wasserstein**：Deshpande et al. (2019) 的线性投影版本，KMS 是其非线性推广
- **Boedihardjo (2025)**：为 MS Wasserstein 给出了无维度尖锐界，本文将类似思路扩展到核空间
- **Kernel OT**：Zhang et al. (2019) 考虑核映射推前 Wasserstein，但受维度灾难影响无法获得尖锐速率
- **SDR for OT**：Xie and Xie (2021) 首次对 MS $W_1$ 用 SDR 但缺乏理论保证和高效算法
- **Kernel PCA 联系**（Remark 2.7）：当 $\nu=\delta_0, p=2$ 时 KMS 退化为 Kernel PCA，可视为双样本的 Kernel PCA 推广

## 评分

- 新颖性: ⭐⭐⭐⭐ — NP-hard 证明 + SDR 秩约界 + 一阶算法是全新贡献
- 实验充分度: ⭐⭐⭐⭐ — 合成+真实数据覆盖检验/检测/生成三个任务，对比充分
- 写作质量: ⭐⭐⭐⭐⭐ — 理论-算法-实验结构清晰，定理陈述精确
- 价值: ⭐⭐⭐⭐ — 为核最优传输的理论与计算奠定坚实基础

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Understanding the Statistical Accuracy-Communication Trade-off in Personalized Federated Learning with Minimax Guarantees](understanding_the_statistical_accuracy-communication_trade-off_in_personalized_f.md)
- [\[NeurIPS 2025\] Wasserstein Transfer Learning](../../NeurIPS2025/optimization/wasserstein_transfer_learning.md)
- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](../../NeurIPS2025/optimization/learning_theory_for_kernel_bilevel_optimization.md)
- [\[ICML 2025\] Subspace Optimization for Large Language Models with Convergence Guarantees](subspace_optimization_for_large_language_models_with_convergence_guarantees.md)
- [\[NeurIPS 2025\] Kernel Learning with Adversarial Features: Numerical Efficiency and Adaptive Regularization](../../NeurIPS2025/optimization/kernel_learning_with_adversarial_features_numerical_efficiency_and_adaptive_regu.md)

<!-- RELATED:END -->
