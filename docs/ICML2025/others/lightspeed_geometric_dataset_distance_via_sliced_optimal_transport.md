---
title: >-
  [论文解读] Lightspeed Geometric Dataset Distance via Sliced Optimal Transport
description: >-
  [ICML2025][Sliced Optimal Transport] 提出 s-OTDD（sliced optimal transport dataset distance），通过 Moment Transform Projection（MTP）将标签分布映射为标量，实现近线性复杂度的数据集距离计算…
tags:
  - "ICML2025"
  - "Sliced Optimal Transport"
  - "Dataset Distance"
  - "Moment Transform Projection"
  - "Wasserstein Distance"
  - "迁移学习"
---

# Lightspeed Geometric Dataset Distance via Sliced Optimal Transport

**会议**: ICML2025  
**arXiv**: [2501.18901](https://arxiv.org/abs/2501.18901)  
**代码**: [hainn2803/s-OTDD](https://github.com/hainn2803/s-OTDD)  
**领域**: 最优传输 / 数据集距离  
**关键词**: Sliced Optimal Transport, Dataset Distance, Moment Transform Projection, Wasserstein Distance, Transfer Learning

## 一句话总结

提出 s-OTDD（sliced optimal transport dataset distance），通过 Moment Transform Projection（MTP）将标签分布映射为标量，实现近线性复杂度的数据集距离计算，速度远超 OTDD 且性能相当。

## 研究背景与动机

数据集距离度量在迁移学习、领域适应、数据增强等任务中至关重要。已有方法中，OTDD（Optimal Transport Dataset Distance）基于层次化最优传输框架，先用内层 OT 计算标签距离，再用外层 OT 计算数据集距离，理论性质好但计算复杂度高：

- **OTDD（精确版）**：时间复杂度 $\mathcal{O}(n^3 \log n + c^2(n_{\max}^3 \log n_{\max} + d))$，空间 $\mathcal{O}(n^2 + c^2)$
- **WTE**（Wasserstein Task Embedding）：需要 MDS + Wasserstein embedding，开销大且不是有效度量
- **CHSW**：需要先用 MDS 嵌入标签，仍有二次类别数依赖

核心痛点：当数据量 $n > 30000$ 或特征维度 $d$ 较高时，上述方法因内存限制崩溃。作者希望设计一种**无需训练、无需嵌入**的数据集距离，实现近线性计算复杂度且不依赖类别数。

## 方法详解

### 核心思路：将数据点投影到一维

s-OTDD 的关键挑战在于：数据点 $(x, y)$ 包含特征 $x$ 和标签 $y$，标签被表示为特征空间上的条件分布 $q_y(X) = q(X|Y=y)$。如何将"分布"投影到标量？

### 1. Moment Transform Projection (MTP)

MTP 是本文核心创新，分两步将标签分布 $q_y$ 映射为标量：

**Step 1 — 特征投影**：用投影函数 $\mathcal{FP}_\theta: \mathcal{X} \to \mathbb{R}$（如 Radon 变换 $\theta^\top x$）将高维分布投影到一维。

**Step 2 — 缩放矩（Scaled Moment）**：对一维投影分布计算第 $\lambda$ 阶缩放矩：

$$\mathcal{SM}_\lambda(\mu) = \int_{\mathbb{R}} \frac{x^\lambda}{\lambda!} f_\mu(x) dx$$

用 $\lambda!$ 做缩放防止高阶矩数值爆炸。经验分布下：$\mathcal{MTP}_{\lambda,\theta}(q_y) = \frac{1}{n_y} \sum_{i: y_i=y} \frac{(\theta^\top x_i)^\lambda}{\lambda!}$。

**单射性保证**：当矩生成函数存在时，MTP 在 $\Lambda = \mathbb{N}$（无穷阶）下可单射；有限阶情况下，若 Hankel 矩阵正定且矩有界（$|m_{\theta,\mu,\lambda}| < CD^\lambda \lambda!$），同样可单射。

### 2. 数据点投影 (Data Point Projection)

将特征投影和多个 MTP 组合：

$$\mathcal{DP}^k_{\psi,\theta,\lambda,\phi}(x, q_y) = \psi^{(1)} \mathcal{FP}_\theta(x) + \sum_{i=1}^{k} \psi^{(i+1)} \mathcal{MTP}_{\lambda^{(i)}, \phi}(q_y)$$

其中 $\psi \in \mathbb{S}^k$ 为单位超球上的权重向量，$k$ 为使用的矩阶数。实验中取 $k=5$。

### 3. s-OTDD 定义

$$\text{s-OTDD}_p^p(\mathcal{D}_1, \mathcal{D}_2) = \mathbb{E}_{(\psi,\theta,\lambda,\phi)} \left[ W_p^p \left( \mathcal{DP}^k \sharp P_{\mathcal{D}_1},\ \mathcal{DP}^k \sharp P_{\mathcal{D}_2} \right) \right]$$

期望对随机投影参数取，一维 Wasserstein 有闭式解（排序即可），Monte Carlo 估计用 $L$ 组投影近似。

### 计算复杂度对比

| 方法 | 时间复杂度 | 空间复杂度 | 类别依赖 |
|------|-----------|-----------|---------|
| OTDD (Exact) | $\mathcal{O}(n^3 \log n + c^2 n_{\max}^3)$ | $\mathcal{O}(n^2 + c^2)$ | 是 |
| OTDD (Gaussian) | $\mathcal{O}(n^3 \log n + c^2 d^3)$ | $\mathcal{O}(n^2 + c^2)$ | 是 |
| **s-OTDD** | $\mathcal{O}(L(n \log n + dn))$ | $\mathcal{O}(L(d+n))$ | **否** |

s-OTDD 不依赖类别数 $c$，对不均衡数据友好，且支持分布式计算（每个数据集可独立预计算投影）。

## 实验关键数据

### 1. 与 OTDD 的相关性（MNIST / CIFAR10 子集对比）

| 方法 | MNIST Spearman $\rho$ | CIFAR10 Spearman $\rho$ |
|------|----------------------|------------------------|
| OTDD (Gaussian) | ~0.90 | ~0.85 |
| WTE | ~0.88 | ~0.82 |
| CHSW (10K proj) | ~0.75 | ~0.70 |
| **s-OTDD (10K proj)** | **~0.90** | **~0.85** |

s-OTDD 与 OTDD (Exact) 高度相关，与 Gaussian 近似和 WTE 性能相当。

### 2. 计算时间

数据量达 30K 以上时，OTDD/WTE/CHSW 因内存不足崩溃，s-OTDD 可顺畅运行至全量数据集（60K MNIST、50K CIFAR10）。从 MNIST 到 CIFAR10（维度增大），s-OTDD 运行时间增长远小于其他方法。

### 3. 迁移学习相关性

| 实验 | 方法 | Spearman $\rho$ | Pearson $r$ |
|------|------|----------------|------------|
| NIST 图像 | OTDD (Exact) | 0.40 | — |
| NIST 图像 | **s-OTDD** | **0.40** | — |
| 文本数据集 | OTDD (Exact) | 较高 | 0.48 |
| 文本数据集 | **s-OTDD** | 略低 | **0.48** |
| Split Tiny-ImageNet 224×224 | **s-OTDD** | 强相关 | 强相关 |
| Split Tiny-ImageNet 224×224 | OTDD | 不可行 | 不可行 |

在 224×224 分辨率的 Tiny-ImageNet 上，OTDD 因计算量过大无法运行，s-OTDD 仍可工作。

### 4. 数据增强（Tiny-ImageNet → CIFAR10）

| 方法 | 样本量 | 投影数 | Spearman $\rho$ | 耗时 |
|------|--------|--------|----------------|------|
| OTDD (Exact) | 5K | — | -0.70 | ~74×10³s |
| **s-OTDD** | **50K** | 100K | **-0.74** | **~53×10³s** |

s-OTDD 处理 10 倍数据量，速度更快，相关性更高。

## 亮点与洞察

1. **MTP 的设计精巧**：用缩放矩将分布映射为标量，$\lambda!$ 缩放防止数值爆炸，理论上保证单射性（基于 Hamburger 矩问题）
2. **完全解耦类别数**：复杂度不依赖 $c$，对类别数多或不均衡数据特别友好
3. **支持分布式 / 联邦学习**：每个数据集可独立预计算投影后的逆 CDF，交换极少数据即可得距离
4. **Monte Carlo 近似误差 $\mathcal{O}(L^{-1/2})$**：收敛快，10K 投影已足够
5. **s-OTDD 是合法度量**：在 $\mathcal{P}(\mathcal{X} \times \mathcal{P}(\mathcal{X}))$ 上满足度量公理

## 局限与展望

1. **矩阶数 $k$ 选择**：实验取 $k=5$，过高时可能出现数值溢出，缺少自适应策略
2. **投影类型选择**：Radon 变换 vs 卷积投影的效果因数据类型而异，需人工选择
3. **非线性结构捕捉有限**：线性投影可能丢失数据的非线性流形结构
4. **梯度流未探索**：论文提到未来工作之一是理解 s-OTDD 的梯度流性质，目前不支持可微优化
5. **仅验证了图像和文本**：缺少图、时序等其他模态的实验

## 相关工作与启发

- **OTDD** (Alvarez-Melis & Fusi, 2020)：层次化 OT 框架的数据集距离，本文加速版本
- **WTE** (Liu et al., 2025)：MDS + Wasserstein embedding，高效但非度量
- **CHSW** (Bonet et al., 2024)：Cartan-Hadamard 流形上的 SW，需 MDS 预处理
- **Sliced Wasserstein** (Bonneel et al., 2015)：s-OTDD 的理论基石，随机投影 + 一维 OT 闭式解

## 评分

- 新颖性: ⭐⭐⭐⭐ — MTP 将"分布→标量"的投影设计新颖且理论完备
- 实验充分度: ⭐⭐⭐⭐ — 覆盖图像/文本/大规模实验，含消融和计算时间分析
- 写作质量: ⭐⭐⭐⭐ — 层次清晰，数学严谨
- 价值: ⭐⭐⭐⭐ — 解决了 OTDD 的核心可扩展性瓶颈，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport](../../NeurIPS2025/others/bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)
- [\[ICML 2025\] Theoretical Performance Guarantees for Partial Domain Adaptation via Partial Optimal Transport](theoretical_performance_guarantees_for_partial_domain_adaptation_via_partial_opt.md)
- [\[ICML 2025\] Hierarchical Refinement: Optimal Transport to Infinity and Beyond](hierarchical_refinement_optimal_transport_to_infinity_and_beyond.md)
- [\[ECCV 2024\] Mahalanobis Distance-Based Multi-View Optimal Transport for Multi-View Crowd Localization](../../ECCV2024/others/mahalanobis_distance-based_multi-view_optimal_transport_for_multi-view_crowd_loc.md)
- [\[NeurIPS 2025\] Estimation of Stochastic Optimal Transport Maps](../../NeurIPS2025/others/estimation_of_stochastic_optimal_transport_maps.md)

</div>

<!-- RELATED:END -->
