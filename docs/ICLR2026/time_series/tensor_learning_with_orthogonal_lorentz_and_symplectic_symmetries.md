---
title: >-
  [论文解读] Tensor learning with orthogonal, Lorentz, and symplectic symmetries
description: >-
  [ICLR 2026][时间序列][等变学习] 本文给出了关于正交群 $O(d)$、不定正交群（含 Lorentz 群）和辛群 $Sp(d)$ 对张量对角作用下的等变多项式函数的完整参数化刻画，并将其应用于设计可学习的稀疏向量恢复算法，在多种数据生成假设下超越了已有的 sum-of-squares 谱方法。
tags:
  - ICLR 2026
  - 时间序列
  - 等变学习
  - 张量函数
  - 正交群
  - Lorentz群
  - 辛群
  - 稀疏向量恢复
---

# Tensor learning with orthogonal, Lorentz, and symplectic symmetries

**会议**: ICLR 2026  
**arXiv**: [2406.01552](https://arxiv.org/abs/2406.01552)  
**代码**: [https://github.com/WilsonGregory/TensorPolynomials](https://github.com/WilsonGregory/TensorPolynomials)  
**领域**: time_series  
**关键词**: 等变学习, 张量函数, 正交群, Lorentz群, 辛群, 稀疏向量恢复

## 一句话总结

本文给出了关于正交群 $O(d)$、不定正交群（含 Lorentz 群）和辛群 $Sp(d)$ 对张量对角作用下的等变多项式函数的完整参数化刻画，并将其应用于设计可学习的稀疏向量恢复算法，在多种数据生成假设下超越了已有的 sum-of-squares 谱方法。

## 研究背景与动机

在机器学习中引入对称性约束已成为主流趋势——图神经网络、几何深度学习和 AI for Science 领域都在利用等变/不变结构来改善泛化性和采样效率。现有工作（如 Villar et al. NeurIPS 2021）已经研究了向量输入到张量输出的 $O(d)$-等变函数构造，但还没有统一处理高阶张量输入、不同奇偶性以及更广泛李群（如 Lorentz 群和辛群）的理论框架。

稀疏向量恢复（planted sparse vector problem）是理论计算机科学中被广泛研究的问题。Hopkins et al. (2016) 和 Mao & Wein (2022) 提出了基于 sum-of-squares 的谱方法来恢复稀疏向量，但这些方法仅在特定假设（如恒等协方差矩阵）下有理论保证。现实场景中的数据分布更加多样化，需要更灵活的算法。

本文的核心动机：能否从理论出发构建等变机器学习模型，使之既尊重底层对称性、又能通过数据学习到比手工设计更好的算法？

## 方法详解

### 整体框架

论文工作分两部分：理论刻画（Section 3-4）和应用验证（Section 5）。理论部分给出了等变张量多项式函数的完整参数化表示，应用部分利用该参数化构建可学习的等变神经网络来解决稀疏向量恢复问题。

### 关键设计

1. **$O(d)$-等变多项式函数的参数化（Theorem 1）**：对从多个张量输入到张量输出的 $O(d)$-等变多项式函数 $f$，证明了它可以写成张量积和 $k$-收缩的线性组合形式，其中系数张量 $c$ 必须是 $O(d)$-迷向张量。关键引理（Lemma 1）进一步表明，所有 $O(d)$-迷向张量都可以由 Kronecker delta $\delta$ 和 Levi-Civita 符号 $\epsilon$ 构造。这为实际构建等变模型提供了完整的"构建块"。

2. **向量输入特化推论（Corollary 1）**：当输入限制为向量（1-张量）时，$O(d)$-等变函数可以简洁地表示为：输出等于所有输入向量对和 Kronecker delta 之间的外积排列的线性组合，系数仅依赖于输入向量之间的内积。这一形式非常适合实际实现——系数函数  $q_{t,\sigma,J}$ 可以用 MLP 近似。

3. **推广到其他群（Theorem 2 & Corollary 2）**：通过复化（complexification）和 Zariski 稠密紧子群上的 Haar 测度平均技术，将结果推广到不定正交群 $O(s,d-s)$（包含 Lorentz 群）和辛群 $Sp(d)$。关键修改在于：用相应群保持的双线性形式的度量张量 $\theta_G$ 替代 Kronecker delta；用群的内积 $\langle \cdot, \cdot \rangle_G$ 替代欧几里得内积。

4. **SparseVectorHunter（SVH）模型**：利用 Corollary 1 的参数化，设计等变模型学习一个 $d \times d$ 对称矩阵 $h$，然后取其最大特征值对应特征向量作为稀疏向量估计。具体地，$h$ 由所有输入行向量的对称外积组合而成，每个组合的系数由一个以所有向量对内积为输入的 MLP 决定。

5. **SVH-Diag 简化模型**：仅使用对角项（$a_i \otimes a_i$），系数仅依赖各向量的范数平方 $\|a_\ell\|^2$。参数更少，在对角协方差设定下表现更好。

### 损失函数 / 训练策略

- 损失函数：$1 - \langle \hat{v}, v_0 \rangle^2$，即预测向量与真实稀疏向量的内积平方的补
- 训练集 5000 个样本，验证集和测试集各 500 个
- 使用 Adam 优化器，学习率指数衰减（0.999/epoch），批量大小 100
- SVH 约 99K 参数，SVH-Diag 约 59K 参数，非等变基线（BL）约 1.33M 参数
- 在单张 RTX 6000 Ada GPU 上训练，共 18 小时
- Early stopping：验证误差连续 20 个 epoch 不改善则停止

## 实验关键数据

### 主实验

实验设置：$n=100$, $d=5$, $\epsilon=0.25$，四种稀疏向量采样方式（AR, BG, CBG, BR）× 三种协方差矩阵（Identity, Diagonal, Random）。评估指标为 $\langle v_0, \hat{v} \rangle^2$（越接近 1 越好）。

| 采样方式 | 协方差 | SOS-I | SOS-II | BL | SVH-Diag | SVH |
|---------|--------|-------|--------|------|----------|-----|
| BR | Random | 0.526 | 0.526 | 0.923 | 0.437 | **0.957** |
| BR | Diagonal | 0.334 | 0.334 | 0.864 | **0.588** | 0.903 |
| BR | Identity | 0.524 | 0.524 | 0.845 | 0.317 | **0.889** |
| CBG | Random | 0.412 | 0.412 | 0.239 | 0.372 | **0.935** |
| A/R | Random | 0.610 | 0.610 | 0.241 | 0.493 | **0.938** |
| BG | Identity | **0.962** | **0.962** | 0.196 | 0.908 | 0.342 |

### 消融实验

| 配置 | 关键表现 | 说明 |
|------|---------|------|
| SVH (全内积) | Random 协方差最优 | 利用全部成对信息优势 |
| SVH-Diag (仅范数) | Diagonal 协方差最优 | 参数更少，匹配数据结构时表现佳 |
| BL (非等变) | 训练集过拟合严重 | 1.33M 参数但泛化差 |
| SOS-I / SOS-II | Identity 协方差最优 | 有理论保证的手工设计方法 |

### 关键发现

- **等变性大幅改善泛化**：非等变基线 BL 在训练集上常常达到最好的拟合，但测试集表现远不如等变模型，生动验证了"对称性改善泛化"的理论预测
- **学习方法在无理论保证区域优于 SOS**：在非恒等协方差（对角、随机）的场景，SVH/SVH-Diag显著超越 SOS 方法，这些场景目前没有理论分析
- **格局分明**：SVH 在 Random 协方差下最优，SVH-Diag 在 Diagonal 下最优，SOS 在 Identity 下最优；唯一例外是 Bernoulli-Rademacher 采样，SVH 全面胜出

## 亮点与洞察

- 非常优雅的理论-实践结合范例：先建立数学刻画，再导出可计算的架构
- Corollary 1 将高度抽象的等变函数参数化变得可实际编程——系数仅依赖标量不变量（内积），通过 MLP 学习
- 与 neural algorithmic reasoning 理念一致：ML 模型结构对齐已知算法策略
- 推广到 Lorentz 和辛群的路径非常清晰：只需替换度量张量和双线性形式
- 实验规模虽小（$n=100, d=5$），但足以揭示等变性的价值

## 局限与展望

- 实验仅在小规模设定（$n=100, d=5$）验证，未测试更大规模
- 输入为高阶张量或混合奇偶性时，实现效率会下降
- 未处理输入张量之间额外的排列不变性（如 $S_n$-不变性），这与图同构问题相关
- SVH 模型中每对向量 $(i,j)$ 都有独立的系数函数，可扩展性有限
- 缺乏与其他等变架构（如 EGNN、SE(3)-Transformers）的对比

## 相关工作与启发

- Villar et al. (2021) "Scalars are universal" 是直接前置工作，本文将其从向量推广到张量
- Hopkins et al. (2016) sum-of-squares 方法提供了基线算法
- 与 Kunisky, Moore & Wein 的 tensor cumulants 工作形成互补：后者侧重对称张量
- Stone-Weierstrass 定理保证了多项式等变函数可以逼近连续等变函数
- 启发：这种"理论驱动架构设计"的思路可应用于更多物理/数学对称性问题

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [SCRAPL: Scattering Transform with Random Paths for Machine Learning](scrapl_scattering_transform_with_random_paths_for_machine_learning.md)
- [Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning](towards_generalizable_pde_dynamics_forecasting_via_physics-guided_invariant_lear.md)
- [GTM: A General Time-series Model for Enhanced Representation Learning](gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-series.md)
- [PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection](paano_patch-based_representation_learning_for_time-series_anomaly_detection.md)
- [Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)

<!-- RELATED:END -->
