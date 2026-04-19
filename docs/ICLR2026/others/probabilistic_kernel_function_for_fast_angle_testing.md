---
title: >-
  [论文解读] Probabilistic Kernel Function for Fast Angle Testing
description: >-
  [ICLR 2026][近似最近邻搜索] 本文研究高维欧氏空间中的角度测试问题，提出两个基于参考角度的确定性概率核函数 $K_S^1$ 和 $K_S^2$，分别用于角度比较和角度阈值判断，无需高斯分布的渐近假设即可获得理论保证，并将其应用于近似最近邻搜索（ANNS），在 HNSW 图上实现 2.5×–3× 的 QPS 加速。
tags:
  - ICLR 2026
  - 近似最近邻搜索
  - 概率核函数
  - 角度测试
  - 随机投影
  - 相似性搜索
---

# Probabilistic Kernel Function for Fast Angle Testing

**会议**: ICLR 2026  
**arXiv**: [2505.20274](https://arxiv.org/abs/2505.20274)  
**代码**: [GitHub](https://github.com/KejingLu-810/KS)  
**领域**: 其他  
**关键词**: 近似最近邻搜索, 概率核函数, 角度测试, 随机投影, 相似性搜索

## 一句话总结

本文研究高维欧氏空间中的角度测试问题，提出两个基于参考角度的确定性概率核函数 $K_S^1$ 和 $K_S^2$，分别用于角度比较和角度阈值判断，无需高斯分布的渐近假设即可获得理论保证，并将其应用于近似最近邻搜索（ANNS），在 HNSW 图上实现 2.5×–3× 的 QPS 加速。

## 研究背景与动机

向量相似性搜索是机器学习、数据挖掘和信息检索中的核心问题。在高维空间中，$\ell_2$ 范数、余弦相似度和内积是最常用的相似性度量，这些度量通常可以归结为归一化向量之间角度的计算。在许多实际场景中，我们并不关心角度的精确值，而是关注角度的大小关系（角度比较）或是否超过阈值（角度阈值判断），这被称为**角度测试**。

现有工作（如 CEOs, PEOs）使用高斯分布生成投影向量，并依赖于 Lemma 1.3（Pham, 2021）建立角度与投影值之间的关系。然而该引理有一个重大理论缺陷：关系式依赖于投影向量数量 $m \to \infty$ 的渐近假设，在 $m$ 有限时无法给出精确保证。本文的出发点正是克服这一限制。

作者的两个关键观察：(1) 高斯分布并非本质需要，决定估计精度的关键因素是**参考角度**，即查询向量与最近投影向量之间的角度；(2) 通过引入随机旋转矩阵，参考角度变得可预测。

## 方法详解

### 整体框架

本文提出两个概率核函数，分别对应两个角度测试问题：
- **$K_S^1(\mathbf{q}, \mathbf{v})$**：用于角度比较（Problem 1.1），判断哪个数据向量与查询更相似
- **$K_S^2(\mathbf{q}, \mathbf{v})$**：用于角度阈值判断（Problem 1.2），判断角度是否小于给定阈值

核心思想是利用确定性的投影向量结构和参考角度信息，取代高斯分布的随机投影。

### 关键设计

1. **核函数 $K_S^1$（角度比较）**: 定义为 $K_S^1(\mathbf{q}, \mathbf{v}) = \langle \mathbf{v}, Z_{HS}(\mathbf{q}) \rangle$，其中 $H$ 是随机旋转矩阵，$Z_{HS}(\mathbf{q})$ 是 $\mathbf{q}$ 在旋转后的投影向量集合 $HS$ 中的参考向量（内积最大的向量）。Lemma 4.2 证明了其条件 CDF 可用正则化不完全 Beta 函数精确表示：$F_{K_S^1}(x | A_S(\mathbf{q}) = \cos\psi) = I_t\left(\frac{d-2}{2}, \frac{d-2}{2}\right)$，且概率保证随参考角度 $\psi$ 减小而增强。设计动机：摆脱渐近假设，建立确定性的概率关系。

2. **核函数 $K_S^2$（角度阈值）**: 定义为 $K_S^2(\mathbf{q}, \mathbf{v}) = \langle H\mathbf{q}, Z_S(H\mathbf{v}) \rangle / A_S(H\mathbf{v})$，通过除以参考角度的余弦值 $A_S(H\mathbf{v})$ 进行归一化。Lemma 4.3 证明 $K_S^2$ 是角度敏感函数（angle-sensitive），满足 Problem 1.2 的概率保证。设计动机：利用参考角度信息进行更精确的阈值判断，现有方法（PEOs）未使用此信息。

3. **投影向量配置**: 提出两种确定性结构取代高斯随机投影：

    - **对踵对结构 (Algorithm 1)**: 在每个子空间中生成 $m/2$ 个随机点及其对踵点，简单高效
    - **多交叉多面体结构 (Algorithm 2)**: 利用交叉多面体（cross-polytope）的最优覆盖性质，通过随机旋转生成多个交叉多面体，选择覆盖效果最好的配置
    - 引入多层级（$L$ 个子空间）的乘积量化技术，虚拟生成 $m^L$ 个投影向量，显著减小参考角度

4. **KS2 路由测试**: 将 $K_S^2$ 应用于相似性图的路由机制，得到简明的测试不等式：$\sum_{i=1}^L \mathbf{q}_i^\top \mathbf{u}_{e[i]}^i \geq A_S(\mathbf{e}) \cdot \frac{\|\mathbf{w}\|^2/2 - \tau - \mathbf{v}^\top\mathbf{q}}{\|\mathbf{e}\|}$。相比 PEOs 测试，不等式更简单（评估更快）、存储常数更少（索引更小）。

### 损失函数 / 训练策略

本文不涉及神经网络训练。核心优化目标是最大化覆盖效果 $J(S) = \mathbb{E}_{\mathbf{v} \in U(\mathbb{S}^{d-1})}[A_S(\mathbf{v})]$，即投影向量配置 $S$ 对单位球面的平均覆盖质量。通过在均匀采样的 $N$ 个点上计算 $\tilde{J}(S,N)$ 近似评估。

## 实验关键数据

### 主实验（ANNS 性能对比）

| 数据集 | 指标 (QPS@Recall=90%) | HNSW+KS2 | HNSW+PEOs | HNSW | 本方法加速比 |
|--------|----------------------|----------|-----------|------|-----------|
| SIFT | QPS | 最优 | 次优 | 基线 | **2.5×–3× vs HNSW** |
| GloVe1M | QPS | 最优 | 次优 | 基线 | **10%–30% vs PEOs** |
| GloVe2M | QPS | 最优 | 次优 | 基线 | 1.1–1.3× vs PEOs |
| Tiny | QPS | 最优 | 次优 | 基线 | 显著优于 HNSW 和 ScaNN |
| GIST | QPS | 最优 | 次优 | 基线 | 显著优于 HNSW |

### 消融实验（KS1 vs CEOs, $k$-MIPS, $k=10$, $m=2048$）

| 配置 | Word Probe@1K | GloVe1M Probe@1K | 说明 |
|------|-------------|-----------------|------|
| CEOs(2048) | 90.203% | 24.166% | 高斯随机投影基线 |
| KS1($S_{sym}$) | 90.265% | 24.456% | 对踵对结构，小幅提升 |
| KS1($S_{pol}$) | 90.678% | 24.556% | 多交叉多面体，最优 |

### 关键发现

- 参考角度越小，核函数精度越高——这是贯穿全文的核心洞察
- 高斯分布不是投影向量的最优选择，确定性结构（尤其是交叉多面体）能提供更小的参考角度
- 当子空间维度 $d' = d/L \approx 16$ 时，HNSW+KS2 达到最佳搜索性能
- KS2 的索引尺寸比 PEOs 小约 5%，且测试不等式更简单，评估更快
- KS1 对 CEOs 的改进较小（约 0.8%），因为差异仅在投影向量分布上

## 亮点与洞察

- **理论贡献**: 建立了不依赖渐近假设的确定性概率关系（Eq. 4），是对现有理论（Lemma 1.3）的根本性改进
- **实践影响大**: HNSW+KS2 在 ANNS 上实现了 2.5×–3× 的加速，具有直接的实际应用价值
- **高斯分布次优性**: 理论和实验均表明高斯投影不是角度测试的最优选择，这挑战了随机投影领域的常见假设
- **SIMD 友好**: KS2 测试可通过 SIMD 并行处理 16 条边，实现高效的硬件级优化

## 局限与展望

- KS1 对 CEOs 的改进较为有限，实际增益主要体现在 KS2 的图搜索加速上
- 最优覆盖问题（best covering problem）在一般情况下仍然开放，$m > d+3$ 时无已知最优解
- 仅在 CPU 上进行实验，GPU 加速场景下的性能有待验证
- 理论分析主要针对欧氏空间和余弦相似度，对其他度量空间的推广尚未讨论

## 相关工作与启发

- CEOs（Pham, 2021）：本文 $K_S^1$ 可视为 CEOs 的推广，用确定性参考角度取代高斯渐近关系
- PEOs（Lu et al., 2024）：$K_S^2$ 在 PEOs 基础上增加了参考角度归一化，简化了测试不等式
- Falconn（Andoni et al., 2015）：使用交叉多面体进行 LSH，本文借鉴了其结构思想但用于不同目的
- 启发：参考角度作为控制精度的关键因素，可能在其他随机投影应用中（如维度约减、草图算法）也有类似作用

## 评分

- 新颖性: ⭐⭐⭐⭐ 参考角度视角新颖，确定性关系取代渐近关系是重要理论进步
- 实验充分度: ⭐⭐⭐⭐ 6 个数据集全面覆盖，与 SOTA 对比充分，消融实验完整
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，但符号较多，部分符号定义分散在正文和附录中
- 价值: ⭐⭐⭐⭐ 对 ANNS 领域有实质贡献，2.5×–3× 加速具有直接应用价值

<!-- RELATED:START -->

## 相关论文

- [Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation](learning_adaptive_distribution_alignment_with_neural_characteristic_function_for.md)
- [An Information-Theoretic Framework For Optimizing Experimental Design To Distinguish Probabilistic Neural Codes](an_information-theoretic_framework_for_optimizing_experimental_design_to_disting.md)
- [Predicting Kernel Regression Learning Curves from Only Raw Data Statistics](predicting_kernel_regression_learning_curves_from_only_raw_data_statistics.md)
- [Fast and Stable Riemannian Metrics on SPD Manifolds via Cholesky Product Geometry](fast_and_stable_riemannian_metrics_on_spd_manifolds_via_cholesky_product_geometr.md)
- [Characterizing and Optimizing the Spatial Kernel of Multi Resolution Hash Encodings](characterizing_and_optimizing_the_spatial_kernel_of_multi_resolution_hash_encodi.md)

<!-- RELATED:END -->
