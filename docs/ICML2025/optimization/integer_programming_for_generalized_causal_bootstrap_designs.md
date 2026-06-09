---
title: >-
  [论文解读] Integer Programming for Generalized Causal Bootstrap Designs
description: >-
  [ICML2025][优化/理论][因果 bootstrap] 提出基于整数规划（IP）数值求解最不利 copula 的方法，将因果 bootstrap 的设计不确定性量化从"完全随机化 + 均值差估计量"推广到任意已知概率分配与线性/二次处理估计量，并证明渐近有效性。
tags:
  - "ICML2025"
  - "优化/理论"
  - "因果 bootstrap"
  - "整数规划"
  - "设计不确定性"
  - "方差上界"
  - "最不利 copula"
  - "有限总体推断"
---

# Integer Programming for Generalized Causal Bootstrap Designs

**会议**: ICML2025  
**arXiv**: [2410.21464](https://arxiv.org/abs/2410.21464)  
**代码**: 未公开（文中提及开源求解器）  
**领域**: 优化 / 因果推断 / 实验设计  
**关键词**: 因果 bootstrap, 整数规划, 设计不确定性, 方差上界, 最不利 copula, 有限总体推断

## 一句话总结

提出基于整数规划（IP）数值求解最不利 copula 的方法，将因果 bootstrap 的设计不确定性量化从"完全随机化 + 均值差估计量"推广到任意已知概率分配与线性/二次处理估计量，并证明渐近有效性。

## 研究背景与动机

在实验因果推断中，不确定性来源分为两类：

**抽样不确定性（sampling uncertainty）**：样本是从超总体中抽取时产生的随机性，经典 bootstrap 和 Wald 型置信区间主要估计这一来源。

**设计不确定性（design uncertainty）**：有限总体模型下，随机分配处理/对照带来的随机性。每个个体只能观测到一种潜在结果（处理或对照），因此统计量在不同随机化下会波动。

在地理实验（如国家级 A/B 测试）等场景中，样本量小（几十到几百）、处理效果异质性大、且样本几乎覆盖全部总体，此时抽样不确定性方法往往**过于保守**。Neyman 方差分解中，方差项 $S_\tau^2$（个体级处理效果方差）依赖于未观测的联合分布，无法直接估计。

**现有局限**：

- Aronow et al. (2014) 和 Imbens & Menzel (2021) 的因果 bootstrap 方法仅适用于**完全随机化分配 + 均值差估计量**，利用 isotone copula（排序配对）的解析解。
- 对于匹配配对设计、Horvitz-Thompson 估计量、doubly-robust 估计量等非标准组合，isotone copula 不再是已知的最不利分布，缺乏通用方法。

**本文动机**：通过整数规划**数值求解**最不利 copula，绕过对解析解的依赖，从而推广到更多估计量和分配机制。

## 方法详解

### 核心框架

目标是找到最大化估计量方差的联合潜在结果分布 $F_{0,1}$：

$$\max_{Y_i(0), Y_i(1) \in \mathcal{Y}^2} \mathbf{Var}_Z[\hat{\tau}] \quad \text{s.t.} \quad F_{0,1} \in \mathcal{C}$$

其中 $\mathcal{C}$ 是对联合分布的约束集合，确保与观测边际分布一致。

### 整数规划建模

**决策变量**：引入指示变量 $X_{ik}^{(a)} := \mathbb{I}(Y_i(a) = y_k)$，其中 $a \in \{0,1\}$ 为处理状态，$i$ 为个体，$y_k$ 为离散结果值。

**目标函数**：方差可表示为二次型 $\mathbf{Var}_Z[\hat{\tau}] = \mathbf{X}^T \mathbf{Q} \mathbf{X}$，其中 $\mathbf{Q} = \mathbf{Y}^T \mathbf{\Sigma}_{ZZ} \mathbf{Y}$，$\mathbf{\Sigma}_{ZZ}$ 为处理分配的协方差矩阵。由于 $X$ 为二值变量，二次项可线性化，转化为标准整数线性规划。

**约束条件**（以完全随机化为例）：

- **(a) 观测锁定**：$X_{ik}^{(Z_i)} = 1$ 当且仅当 $Y_i^{obs} = y_k$，观测到的潜在结果固定。
- **(b) 支撑约束**（可选）：$X_{ik}^{(a)} = 0$ 若 $y_k \notin \text{supp}(F_a)$，限制潜在结果只取已观测值。
- **(c) 唯一匹配**：$\sum_k X_{ik}^{(a)} = 1$，每个个体在每种处理下只取一个值。
- **(d) 边际匹配**：$\sum_i X_{ik}^{(a)} (Z_i/N_1 - (1-Z_i)/N_0) = 0$，确保观测与缺失的边际分布一致。约束 (d) 可能不可行，引入松弛参数 $\epsilon$。

**Lemma 2.1**：当 $\epsilon \geq 1/\min(N_0, N_1)$ 时，整数规划保证可行。

### 估计量推广

**线性处理估计量**：形如 $\hat{\tau} = \sum_i Z_i a_i + b_i$，包括 Horvitz-Thompson 估计量。方差为 $\sum_{i,j} a_i a_j \mathbf{Cov}[Z_i, Z_j]$，只需将 $\mathbf{Q}$ 替换为 $\mathbf{Q}' = \mathbf{Y}^T \mathbf{U}^T \mathbf{\Sigma}_{ZZ} \mathbf{U} \mathbf{Y}$。

**二次处理估计量**：形如 $\hat{\tau} = \sum_i b_i + \sum_j Z_i Z_j a_{ij}$，包括线性回归对处理变量和协变量的拟合。方差涉及四阶矩 $\mathbf{Cov}[Z_i Z_j, Z_k Z_l]$。

**Doubly-Robust 估计量**：利用样本外拟合的协变量预测函数，将问题转化为对残差 $Y_i' = Y_i - Z_i \hat{f}_1(W_i) - (1-Z_i)\hat{f}_0(W_i)$ 应用 Horvitz-Thompson 估计量。

### 分配机制推广

**一般已知概率分配**：将约束 (d) 推广为基于倾向性得分加权的边际匹配：

$$(-1)^b \sum_i X_{ik}^{(a)} \left(\frac{Z_i}{P_i} - \frac{1-Z_i}{1-P_i}\right) \leq \epsilon N$$

**条件非混杂分配**：利用协变量 $W_i$ 的条件边际分布匹配，添加更细粒度约束以收紧置信区间。

### 渐近有效性

**Theorem 4.1**：对无混杂分配，$\mathbb{P}(V^* \geq \mathbf{Var}_Z[\hat{\tau}]) \geq 1 - \beta$，其中 $\beta = 8\exp(-\epsilon^2 N \tilde{P}/4) + O(\text{Cov}(Z_i,Z_j))$，随 $N$ 增大趋于 0。

**Theorem 4.3**：对有界混杂的个体分配，当松弛 $\epsilon \geq \delta$（混杂度）时同样渐近有效。

## 实验关键数据

使用 IMF 公开的 2017–2019 年 GDP 数据（前 50 国），模拟国家级地理实验。

### 主要结果（Table 1 - A/A 测试，95% CI 宽度）

| 协变量 | 方法 | 完全随机覆盖率 | 匹配配对覆盖率 | 完全随机 CI | 匹配配对 CI |
|--------|------|--------------|--------------|-----------|-----------|
| 无 | Sampling Bootstrap | 90% | 100% | 3967 | 4110 |
| 无 | Conservative Var. | 98% | 100% | 3991 | 1152 |
| 无 | Isotone Copula | 87% | **0%** | 3348 | 44 |
| 无 | **Opt. Causal Boot.** | 87% | **100%** | 3348 | **2233** |
| 2018 GDP | Sampling Bootstrap | 91% | 100% | 141 | 142 |
| 2018 GDP | **Opt. Causal Boot.** | 90% | **100%** | **133** | **102** |

### 核心发现

1. **完全随机化下**：本文方法与 isotone copula 完全一致，验证了 IP 求解正确性。
2. **匹配配对下**：isotone copula 覆盖率降至 **0%**（方差估计为零），本文方法覆盖率 **100%**，CI 宽度合理。
3. **CI 宽度**：相比 sampling bootstrap，本文方法在所有实验中 CI 宽度缩减 **≥10%**，差值估计量+乘法效果场景下缩减达 **50%**。
4. 加入前期 GDP 协变量后，CI 宽度从约 3000+ 大幅降至 ~130，体现协变量调整的巨大收益。

### 可扩展性

- 匹配配对设计：50 个单位平均 671 秒（IP 求解）
- 完全随机化设计：LP 松弛仅需 22 秒
- 计算复杂度随单位数增长，适用于数百规模的数据集

## 亮点与洞察

1. **方法论突破**：用整数规划替代解析求解最不利 copula，解锁了因果 bootstrap 对非标准设计（匹配配对等）和非标准估计量（doubly-robust、线性回归等）的适用性。
2. **Isotone copula 的失效案例**：匹配配对设计下 isotone copula 覆盖率为 0% 是极好的反例——配对良好时反事实填补会导致方差退化为零。
3. **理论完备性**：覆盖了无混杂、条件无混杂、有界混杂三类分配机制的渐近有效性证明。
4. **实用价值**：地理实验（国家/城市级 A/B 测试）是工业界常见场景，本文直接解决了该场景的不确定性量化痛点。

## 局限与展望

1. **计算可扩展性**：整数规划的求解时间随单位数快速增长，目前仅适用于数百规模的小样本，大规模实验需要近似算法。
2. **覆盖率不足**：在完全随机化+重尾数据下，所有 bootstrap 方法的覆盖率低于名义水平（87-91% vs 95%），边际分布收敛慢是根本原因。
3. **未覆盖一般混杂分配**：仅证明了有界混杂的个体分配，一般混杂机制的渐近有效性留作未来工作。
4. **离散化假设**：连续结果需离散化为观测值支撑，当缺失潜在结果的支撑超出观测范围时，上界不再保证有效。
5. **协变量维度**：条件边际匹配在高维/连续协变量下收敛更慢，实际改进可能有限。

## 相关工作与启发

- **Neyman (1923)**：有限总体因果推断的奠基工作，方差分解的起点
- **Aronow et al. (2014)**：证明 isotone copula 是完全随机化+DIM 下的方差最大化 copula
- **Imbens & Menzel (2021)**：提出因果 bootstrap 重采样方法，本文的直接前身
- **Harshaw et al. (2021)**：通过凸优化求二次型方差界，但限于二次形式
- **Ji et al. (2023)**：用最优传输估计联合分布泛函的界，对偶方法但不显式求解耦合
- **启发**：将因果推断中的不可识别问题转化为数学规划问题是一种通用思路，可推广到其他反事实估计任务

## 评分

- 新颖性: ⭐⭐⭐⭐ — 整数规划求最不利 copula 的思路新颖，解锁了之前不可处理的设计×估计量组合
- 实验充分度: ⭐⭐⭐ — IMF GDP 数据模拟设计合理，但仅一个数据集，缺少更多实际应用场景验证
- 写作质量: ⭐⭐⭐⭐ — 从简单到一般的展开清晰，理论证明完整
- 价值: ⭐⭐⭐⭐ — 对地理实验等小样本因果推断场景有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Provably Data-Driven Lagrangian Relaxation for Mixed Integer Linear Programming](../../ICML2026/optimization/provably_data-driven_lagrangian_relaxation_for_mixed_integer_linear_programming.md)
- [\[ICLR 2026\] Constraint Matters: Multi-Modal Representation for Reducing Mixed-Integer Linear programming](../../ICLR2026/optimization/constraint_matters_multi-modal_representation_for_reducing_mixed-integer_linear_.md)
- [\[ICML 2025\] Sparse Causal Discovery with Generative Intervention for Unsupervised Graph Domain Adaptation](sparse_causal_discovery_with_generative_intervention_for_unsupervised_graph_doma.md)
- [\[ICML 2025\] SDP-CROWN: Efficient Bound Propagation for Neural Network Verification with Tightness of Semidefinite Programming](sdp-crown_efficient_bound_propagation_for_neural_network_verification_with_tight.md)
- [\[NeurIPS 2025\] Conformal Prediction for Causal Effects of Continuous Treatments](../../NeurIPS2025/optimization/conformal_prediction_for_causal_effects_of_continuous_treatments.md)

</div>

<!-- RELATED:END -->
