---
title: >-
  [论文解读] Generalization Bounds for Semi-supervised Matrix Completion with Distributional Side Information
description: >-
  [AAAI 2026][半监督矩阵补全] 提出首个半监督矩阵补全学习范式：假设采样分布 $P$ 和真实矩阵 $G$ 共享低秩子空间，给定大量未标注数据 $M$ 和少量标注数据 $N$，证明泛化误差可分解为 $\tilde{O}(\sqrt{nd/M}) + \tilde{O}(\sqrt{dr/N})$ 两个独立项，在 Douban 和 MovieLens 数据集上显著优于仅用显式反馈的基线。
tags:
  - AAAI 2026
  - 半监督矩阵补全
  - 泛化界
  - 低秩子空间
  - 隐式反馈
  - 显式反馈
  - 核范数约束
---

# Generalization Bounds for Semi-supervised Matrix Completion with Distributional Side Information

**会议**: AAAI 2026  
**arXiv**: [2511.13049](https://arxiv.org/abs/2511.13049)  
**领域**: 矩阵补全理论 / 推荐系统  
**关键词**: 半监督矩阵补全, 泛化界, 低秩子空间, 隐式反馈, 显式反馈, 核范数约束

## 一句话总结

提出首个半监督矩阵补全学习范式：假设采样分布 $P$ 和真实矩阵 $G$ 共享低秩子空间，给定大量未标注数据 $M$ 和少量标注数据 $N$，证明泛化误差可分解为 $\tilde{O}(\sqrt{nd/M}) + \tilde{O}(\sqrt{dr/N})$ 两个独立项，在 Douban 和 MovieLens 数据集上显著优于仅用显式反馈的基线。

## 研究背景与动机

**领域现状**：矩阵补全是推荐系统的核心技术，从有限的观测条目恢复完整的用户-物品评分矩阵。经典理论（Candes & Recht 等）证明了核范数最小化在均匀采样下能精确恢复秩 $r$ 矩阵，前提是观测数 $\tilde{O}(nr)$。归纳矩阵补全（IMC）引入边信息矩阵 $X, Y$ 进一步降低样本复杂度。

**现有痛点**：
(1) **所有理论结果都假设观测全部带标签**——每个观测到的条目 $(i,j)$ 都有对应的评分标签，但实际中大量交互（浏览/点击/购买等"隐式反馈"）不带评分；
(2) **IMC 假设边信息已知**——实际中边信息矩阵需要从其他数据（社交图、分子结构等）估计，而非直接可用；
(3) **隐式与显式反馈的关系缺乏理论刻画**——大量实践表明隐式反馈包含评分预测的有用信息，但缺乏形式化的学习理论支撑。

**本文切入角度**：假设采样分布 $P$（隐式反馈的来源）与真实矩阵 $G$（显式评分）共享一个低秩子空间，用大量未标注样本估计共享子空间，用少量标注样本在此子空间上进行矩阵恢复。

## 方法详解

### 整体框架

DAMC（Distributionally Aware Matrix Completion）算法分两步：

1. **步骤 1——子空间估计**：构造未标注数据的经验分布矩阵 $\mathcal{H} = \frac{1}{M}\sum_{o=1}^M \mathbf{1}_{i_o, j_o}$，对其进行秩 $d$ 的截断 SVD 得到 $U, V$，构造边信息 $X = \sqrt{m/d} \cdot U$，$Y = \sqrt{n/d} \cdot V$

2. **步骤 2——矩阵恢复**：固定 $X, Y$，通过核范数约束的经验风险最小化求解核心矩阵 $\underline{M}$：$\min_{\|\underline{M}\|_* \leq \mathcal{M}} \frac{1}{N}\sum_{o=1}^N l((X\underline{M}Y^\top)_{\xi_o}, \tilde{G}_o)$

### 关键设计

1. **共享低秩子空间假设（Assumption 2）**

    - **功能**：形式化隐式反馈与显式反馈之间的关系
    - **核心思路**：假设 $P = U^* \Sigma^* [V^*]^\top$（秩 $d$），且 $G = U^* \underline{M}^{*-} [V^*]^\top$（秩 $r \leq d$），即两个矩阵的行/列空间由相同的子空间张成
    - **设计动机**：在推荐系统中，用户的交互模式（隐式反馈）和评分偏好（显式反馈）受相同的潜在因子驱动。这是最简单的形式化，但对于理论分析足够用

2. **泛化界的误差分解定理（Theorem 1）**

    - **功能**：证明半监督矩阵补全的泛化误差可分解为两个独立的误差项
    - **核心结论**：在满足有界性、Lipschitz连续性、共享子空间、非相干性、均匀边际等假设下，泛化误差界为：
        - $\tilde{O}(\sqrt{(m+n)r/M})$：子空间估计误差，由未标注样本 $M$ 控制
        - $\tilde{O}(\sqrt{dr/N})$：矩阵恢复误差，由标注样本 $N$ 控制
        - 高阶交叉项 $\tilde{O}(\sqrt{\Gamma(m+n)r/(MN)})$：在温和条件下可被吸收
    - **关键意义**：标注数据的样本复杂度仅为 $\tilde{O}(dr)$，远小于不使用边信息时的 $\tilde{O}((m+n)r)$——即便每行/列的标注样本数趋于零，只要未标注样本充足，仍可准确恢复

### 损失函数 / 训练策略

实际实现中将核范数约束转化为 Lagrangian 形式：$\min \frac{1}{N}\sum l((XAB^\top Y^\top)_{\xi_o}, \tilde{G}_o) + \lambda_\mathcal{M}[\|A\|_{Fr}^2 + \|B\|_{Fr}^2]$，利用 $\|A\|_{Fr}^2 + \|B\|_{Fr}^2$ 等价核范数的经典引理。实数据实验中将 SVD 替换为非线性自编码器以处理数据非线性。

## 实验关键数据

### 合成数据实验——误差分解验证

在 $200 \times 200$ 矩阵（$d=r=4$）上，$M \in [10000, 100000]$，$N \in [50, 1000]$，30 次独立运行。泛化误差与分解估计（$\text{GAP}(M, 1000) + \text{GAP}(100000, N)$）高度相关，验证了两种误差确实可加性独立组合。

### 真实数据实验——RMSE 对比

| 数据集 | 方法 | p=0.0 | p=0.5 | p=0.9 | p=0.95 |
|--------|------|-------|-------|-------|--------|
| ML-100K | UserKNN | 1.012 | 1.038 | 1.168 | 1.246 |
| | IGMC | 0.928 | 0.982 | 1.117 | 1.148 |
| | Soft Impute | 0.918 | 0.962 | 1.419 | 1.823 |
| | **DAMC** | **0.907** | **0.936** | **1.006** | **1.046** |
| Douban | UserKNN | 0.795 | 0.829 | 0.964 | 0.984 |
| | **DAMC** | **0.718** | **0.738** | **0.832** | **0.858** |

*p 表示移除标签的比例，p=0.9 意味着仅保留 10% 的显式评分*

### 关键发现

- DAMC 在高标签移除率（p=0.9, 0.95）下优势更加明显——Soft Impute 在 p=0.9 时 RMSE 暴涨至 1.419 而 DAMC 仅 1.006
- 未标注样本（隐式反馈）确实包含估计真实矩阵行列子空间的有用信息
- 合成实验中误差分解估计与真实泛化误差的散点图几乎落在对角线上，强烈支持理论预测

## 亮点与洞察

1. **首次形式化隐式-显式反馈关系**：共享低秩子空间假设虽简单但有效，为推荐系统双反馈联合建模提供了理论根基
2. **误差分解结论优美**：两种数据源的误差加性独立，给出了标注/未标注数据量的精确权衡关系
3. **实际意义深远**：理论支持了"少量评分 + 大量交互"即可构建高质量推荐系统的实践经验
4. **与 MNAR（非随机缺失）等相关设定的清晰对比**：详细讨论了与其他非均匀采样理论的区别

## 局限与展望

1. 假设 7（采样概率均匀上界）较强，在实际推荐场景中热门物品的采样概率远高于冷门物品
2. 需要指定子空间维度 $d$——实际中 $d$ 未知，选择不当会影响效果
3. 仅验证 RMSE 指标，未评估排序指标（NDCG、Hit Rate 等推荐系统常用指标）
4. 非线性自编码器替代 SVD 的做法缺少理论保证

## 相关工作与启发

- 半监督矩阵补全的学习范式可推广到药物交互预测、化学工程等其他矩阵补全应用
- 将采样分布视为可利用的信号而非需要去除的偏差，是一种富有启发性的新视角
- 理论框架可以与协同过滤中的 Graph Neural Network 方法（如 IGMC）结合，同时利用结构信息和分布信息

## 评分

⭐⭐⭐⭐

- **新颖性** ⭐⭐⭐⭐⭐：首次在矩阵补全中引入半监督范式并给出严格泛化界
- **实验充分度** ⭐⭐⭐：合成实验精心设计但真实实验较单薄
- **写作质量** ⭐⭐⭐⭐：数学推导严谨清晰，假设讨论充分
- **价值** ⭐⭐⭐⭐：为推荐系统隐式/显式反馈联合利用提供了坚实的理论基础

<!-- RELATED:START -->

## 相关论文

- [Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling](semi-supervised_synthetic_data_generation_with_fine-grained_relevance_control_fo.md)
- [FreqRec: Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation](exploiting_inter-session_information_with_frequency-enhanced_dual-path_networks_.md)
- [PAC-Bayes Bounds for Multivariate Linear Regression and Linear Autoencoders](../../NeurIPS2025/recommender/pac-bayes_bounds_for_multivariate_linear_regression_and_linear_autoencoders.md)
- [Adaptive Elicitation of Latent Information Using Natural Language](../../ICML2025/recommender/adaptive_elicitation_of_latent_information_using_natural_language.md)
- [Recommendations with Sparse Comparison Data: Provably Fast Convergence for Nonconvex Matrix Factorization](../../ICML2025/recommender/recommendations_with_sparse_comparison_data_provably_fast_convergence_for_noncon.md)

<!-- RELATED:END -->
