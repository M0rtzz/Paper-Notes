---
title: >-
  [论文解读] On the Variability of Concept Activation Vectors
description: >-
  [AAAI 2026][Concept Activation Vectors] 对 TCAV 方法中概念激活向量（CAV）的变异性进行首次理论分析，证明 CAV 的方差以 $O(1/N)$ 速率衰减（$N$ 为随机样本数），而 TCAV 分数的方差因"边界点"保持 $O(1)$…
tags:
  - "AAAI 2026"
  - "Concept Activation Vectors"
  - "TCAV"
  - "方差分析"
  - "渐近正态性"
  - "可解释AI稳定性"
---

# On the Variability of Concept Activation Vectors

**会议**: AAAI 2026  
**arXiv**: [2509.24058](https://arxiv.org/abs/2509.24058)  
**代码**: 待发布  
**领域**: 其他  
**关键词**: Concept Activation Vectors, TCAV, 方差分析, 渐近正态性, 可解释AI稳定性

## 一句话总结

对 TCAV 方法中概念激活向量（CAV）的变异性进行首次理论分析，证明 CAV 的方差以 $O(1/N)$ 速率衰减（$N$ 为随机样本数），而 TCAV 分数的方差因"边界点"保持 $O(1)$，需通过多次运行平均以 $O(1/s)$ 降低。

## 研究背景与动机

### 领域现状

**领域现状**：TCAV（Testing with Concept Activation Vectors）是概念可解释性的核心方法之一，通过训练线性分类器分离概念嵌入和随机嵌入来获取概念方向向量 CAV，然后计算模型预测对该方向的敏感度。

**现有痛点**：TCAV 依赖随机采样构建参考集，导致每次运行结果可能不同。Kim et al. 建议多次运行取平均，但未量化需要多少次运行、多少样本才能获得稳定结果。

**核心问题**：在固定计算预算下，是做一次大样本运行更好，还是多次小样本运行取平均更好？目前缺乏理论指导。

**切入角度**：借助不平衡逻辑回归的渐近理论，分析 CAV 估计量在随机样本数趋于无穷时的收敛行为。

## 方法详解

### 理论框架

将 CAV 的计算形式化为不平衡逻辑回归问题：概念样本数固定，随机样本数 $N \to \infty$。在此极限下分析 CAV 估计量 $\hat{\beta}_N$ 的渐近性质。

### 关键理论结果

1. **定理 1：CAV 的渐近正态性**:
    - 内容：在"包围均值"假设下，$\sqrt{N}(\hat{\beta}_N - \beta_0) \Rightarrow \mathcal{N}(0, \Sigma)$
    - 推论：CAV 的方差 $\text{tr}(\text{Cov}(\hat{\beta}_N)) = O(1/N)$
    - 意义：增加随机样本数可以有效稳定 CAV 方向估计
    - 证明思路：对损失函数梯度在最优点做 Taylor 展开，利用大数定律（Hessian 收敛）和中心极限定理（Score 收敛）结合 Slutsky 定理

2. **推论 1：敏感度分数的方差**:
    - 内容：$\sqrt{N}(S(\mathbf{x}, \beta_N) - S(\mathbf{x}, \beta_0)) \xrightarrow{D} \mathcal{N}(0, V(\mathbf{x}))$
    - 意义：敏感度分数的方差也以 $O(1/N)$ 衰减

3. **TCAV 分数方差的意外发现**:
    - 内容：TCAV 分数的方差**不随 $N$** 衰减，保持 $O(1)$
    - 原因：TCAV 是对敏感度分数取阈值后计数，"边界点"（敏感度接近 0 的样本）的分类对 CAV 微小变化高度敏感，贡献恒定方差
    - 解决：多次运行平均，$\text{Var}(T_{\text{multi}}) = O(1/s)$

### 实践建议

- 稳定 **TCAV 分数**：使用多次独立运行（大 $s$），每次样本数可以较小
- 稳定 **CAV 方向**（用于下游如偏差消除）：增大每次运行的样本数 $N$
- 没有万能设置：最优分配取决于具体方法和实现

## 实验关键数据

### 跨模态验证


### 主实验

| 数据类型 | 数据集 | 模型 | CAV方差 $\propto 1/N$? | TCAV方差稳定? |
|----------|--------|------|----------------------|--------------|
| 图像 | ImageNet + Broden | ResNet | ✅ | ✅ |
| 表格 | UCI Adult | 2层MLP | ✅ | ✅ |
| 文本 | IMDB | CNN分类器 | ✅ | ✅ |

### CAV 方差衰减验证


### 消融实验

| $N$ (随机样本数) | CAV 方差 (trace) 大致量级 |
|-----------------|-------------------------|
| 10 | ~$10^{-1}$ |
| 50 | ~$10^{-2}$ |
| 100 | ~$10^{-2.5}$ |
| 200 | ~$10^{-3}$ |

在所有三个领域中，经验方差与理论预测的 $1/N$ 衰减率一致。

### 多次运行平均

| $s$ (运行次数) | TCAV 方差 |
|---------------|-----------|
| 2 | ~0.01 |
| 5 | ~0.004 |
| 10 | ~0.002 |
| 20 | ~0.001 |

方差以 $1/s$ 速率下降，符合 Conjecture 1。

### 关键发现
- CAV 方差衰减与分类器类型无关——逻辑回归、SVM、均值差分法都展现相同的 $O(1/N)$ 行为
- TCAV 方差不降反稳的原因是"边界点"效应——这些样本的敏感度接近 0，阈值化后对 CAV 微小变化过度敏感
- 计算预算分配 tradeoff：多次运行比单次大样本对 TCAV 更高效，但对 CAV 方向则相反

## 亮点与洞察
- **"边界点"导致 TCAV 方差不收敛**：这个发现出人意料——即使 CAV 本身变得极其精确，TCAV 分数仍可能不稳定。根源在于阈值化（indicator function）的不连续性，这是一个影响所有基于阈值的统计量的普遍问题
- **从 LIME 到 TCAV 的理论分析路线**：借鉴了 Garreau & Mardaoui (2021) 对 LIME 稳定性的分析范式，建立了 XAI 方法可靠性分析的理论框架
- **实用价值高**：给出了具体的计算预算分配建议，对实际使用 TCAV 的研究者有直接指导意义

## 局限与展望
- 理论分析假设优化器完美收敛，实际中求解器收敛不完全可能引入额外方差
- "包围均值"假设（Assumption 1）虽然通常成立，但未给出可验证的充分条件
- 未分析非线性概念边界（如 CAR、Concept Gradient 等方法）的方差行为
- Conjecture 1 未给出正式证明，依赖独立性假设

## 相关工作与启发
- **vs LIME 稳定性分析 (Garreau & Mardaoui 2021)**：LIME 的不稳定性来自采样数不足，方差也以 $1/N$ 衰减；TCAV 的不稳定性更深层——即使 CAV 收敛了 TCAV 分数仍不稳定
- **vs Adversarial CAV (Soni et al. 2020)**：他们通过对抗扰动提高 CAV 鲁棒性，本文从理论角度分析并提出更简单的多次运行方案

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次对 TCAV 方差做理论分析，"边界点"发现有洞察
- 实验充分度: ⭐⭐⭐⭐ 三种数据模态验证理论预测
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，实用建议明确
- 价值: ⭐⭐⭐⭐ 对 XAI 方法可靠性的理论理解有重要贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Autonomous Concept Drift Threshold Determination](autonomous_concept_drift_threshold_determination.md)
- [\[ACL 2025\] Partial Colexifications Improve Concept Embeddings](../../ACL2025/others/partial_colexifications_improve_concept_embeddings.md)
- [\[NeurIPS 2025\] FACE: Faithful Automatic Concept Extraction](../../NeurIPS2025/others/face_faithful_automatic_concept_extraction.md)
- [\[ICML 2025\] Synthesizing Images on Perceptual Boundaries of ANNs for Uncovering and Manipulating Human Perceptual Variability](../../ICML2025/others/synthesizing_images_on_perceptual_boundaries_of_anns_for_uncovering_and_manipula.md)
- [\[ICML 2026\] Polaris: Coupled Orbital Polar Embeddings for Hierarchical Concept Learning](../../ICML2026/others/polaris_coupled_orbital_polar_embeddings_for_hierarchical_concept_learning.md)

</div>

<!-- RELATED:END -->
