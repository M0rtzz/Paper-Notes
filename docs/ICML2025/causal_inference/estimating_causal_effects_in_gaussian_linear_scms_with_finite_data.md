---
title: >-
  [论文解读] Estimating Causal Effects in Gaussian Linear SCMs with Finite Data
description: >-
  [ICML 2025][因果推理][结构因果模型] 提出 Centralized Gaussian Linear SCM (CGL-SCM)，通过将外生变量标准化为 $\mathcal{N}(0,1)$ 大幅减少参数量，并设计基于 EM 的估计算法，在有限观测数据下准确恢复可识别的因果效应。
tags:
  - "ICML 2025"
  - "因果推理"
  - "结构因果模型"
  - "高斯线性SCM"
  - "EM算法"
  - "因果效应估计"
  - "潜在混杂因子"
---

# Estimating Causal Effects in Gaussian Linear SCMs with Finite Data

**会议**: ICML 2025  
**arXiv**: [2601.04673](https://arxiv.org/abs/2601.04673)  
**代码**: 无  
**领域**: 因果推断  
**关键词**: 结构因果模型, 高斯线性SCM, EM算法, 因果效应估计, 潜在混杂因子

## 一句话总结

提出 Centralized Gaussian Linear SCM (CGL-SCM)，通过将外生变量标准化为 $\mathcal{N}(0,1)$ 大幅减少参数量，并设计基于 EM 的估计算法，在有限观测数据下准确恢复可识别的因果效应。

## 研究背景与动机

从观测数据中估计因果效应是因果推断的核心挑战，尤其当存在潜在混杂因子 (latent confounders) 时。现有工作主要分为两大流派：

**非参数方法**：以 Pearl 的 do-calculus 为代表，在理论上解决了 L2（干预）和 L3（反事实）查询的可识别性问题，但通常假设无限数据或不涉及具体参数估计。

**参数方法**：计量经济学和统计学中常用线性结构因果模型 (Linear SCM)，但已有工作多假设无限数据、已知部分分布参数、或 Markovian 假设（排除潜在混杂）。

**核心痛点**：Gaussian Linear SCM (GL-SCM) 虽然分析友好，但在建模观测和潜在混杂时参数过多 (overparameterization)，导致有限数据下参数估计不可行。外生变量 $\mathbf{U'} \sim \mathcal{N}(\boldsymbol{\mu_{U'}}, \boldsymbol{\Sigma^2})$ 引入了均值和方差共 $2|\mathbf{U}|$ 个额外参数，加上边权和偏置，参数总量远超观测数据所能约束的信息量。

本文的目标：在已知因果图的前提下，仅用有限观测样本，估计 GL-SCM 中可识别的因果效应。

## 方法详解

### 整体框架

本文提出了两步走的技术路线：

1. **模型简化**：定义 CGL-SCM 子类，证明其与 GL-SCM 在因果效应可识别性上等价
2. **参数估计**：设计 EM 算法从有限数据学习 CGL-SCM 参数，进而计算因果效应

核心洞察在于：因果效应的可识别性意味着所有共享相同观测分布 $P(\mathbf{X})$ 和因果图 $G$ 的 SCM 对于可识别查询 $Q$ 会给出相同结果。因此，不需要恢复真实的数据生成模型，只需找到一个与观测分布匹配的 CGL-SCM 即可。

### 关键设计

#### 1. Centralized Gaussian Linear SCM (CGL-SCM)

CGL-SCM 通过两个关键约束简化参数空间：

- 外生混杂变量：$\mathbf{U} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$（标准正态）
- 外生非混杂变量：$\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Psi^2})$（零均值）

内生变量的结构方程为：

$$X_i = \sum_{X_j \in Pa^o(X_i)} \alpha_{ji} X_j + \sum_{U_k \in Pa^u(X_i)} \alpha_{ki} U_k + \mu_i + \varepsilon_i$$

相比 GL-SCM，CGL-SCM 将外生变量的均值和方差信息"吸收"进边权 $\alpha_{ki}$ 和偏置 $\mu_i$ 中，实现 $\alpha_{ki} = \alpha'_{ki} \cdot \sigma^2_{U'_k}$，$\mu_i = \sum \alpha'_{ki} \mu_{U'_k} + \mu'_i + \mu_{\varepsilon'_i}$。

**等价性定理 (Theorem 2.3)**：对任意 GL-SCM $M'$，存在具有相同因果图的 CGL-SCM $M$，使得 $P^{M'}(\mathbf{X}) = P^{M}(\mathbf{X})$。

**可识别性传递定理 (Theorem 2.4)**：若查询 $Q$ 在因果图 $G$ 中可识别，则 $P^{M'}(Q) = P^{M}(Q)$。这意味着在 CGL-SCM 上估计的因果效应与原始 GL-SCM 完全一致。

#### 2. 向量化表示

为使 EM 算法可行，将 CGL-SCM 转换为矩阵形式。定义：

- **边权矩阵** $T$：$t_{ij} = \alpha_{ij}$ 若 $X_i \to X_j$，否则为 0
- **路径聚合矩阵** $B$：$B = I + \sum_{i=1}^{d} T^i$，其中 $d$ 为图中最长路径长度
- **外生影响矩阵** $C$：$c_{ij}$ 表示 $U_i$ 对 $X_j$ 的直接影响

最终的向量化形式：

$$\mathbf{X} = B^T \boldsymbol{\mu} + B^T C^T \mathbf{U} + B^T \boldsymbol{\varepsilon}$$

#### 3. 图结构约束的掩码机制

直接应用 EM 可能产生违反图结构的参数。论文定义了掩码矩阵 $B_m$ 和 $C_m$：

- $B_m$：由 $A = \sum_{i=1}^d T_m^i$ 计算，$b^m_{ij} = 1$ 当 $a_{ij} > 0$（即存在有向路径）
- $C_m$：$(i,j)$ 元素为 1 当且仅当 $U_i \to X_j$ 存在

梯度更新时通过掩码强制 $B \leftarrow B + \eta \nabla_B l \cdot B_m$，确保无边处的权重始终为零。

#### 4. 从聚合矩阵 B 恢复边权矩阵 T（Algorithm 2: CGL-Edge）

矩阵 $B$ 编码的是变量间的总影响（所有路径之和），需要还原为单边权重 $T$。对每个节点 $X_i$，按拓扑序遍历其直接后继 $\bar{X_i}$，递推计算：

$$t_{ik} = b_{ik} - \sum_{\tau_i(X_j) < \tau_i(X_k)} t_{ij} \cdot b_{jk}$$

这本质上是从总效应中剥离间接路径的贡献，逐步提取直接因果边的权重。

### 损失函数 / 训练策略

#### EM 算法 (Algorithm 1: CGL-Go)

**E-step**：计算潜在变量的后验分布 $\mathbf{U}^i | \mathbf{x}^i$：

$$\boldsymbol{\mu}_{\mathbf{U}^i | \mathbf{x}^i} = CB \big((CB)^T CB + B^T B\big)^{-1} (\mathbf{x}^i - B^T \boldsymbol{\mu})$$

$$\Sigma_{\mathbf{U}^i | \mathbf{x}^i} = \mathbf{I} - CB \big((CB)^T CB + B^T B\big)^{-1} B^T C^T$$

**M-step**：最大化期望对数似然：

$$\max_{B,C,\mu} -n \log |B^T B| - \sum_{i=1}^N \mathbb{E}_{\mathbf{U}^i|\mathbf{x}^i} \Big[ (\mathbf{x}^i - B^T \boldsymbol{\mu} - B^T C^T \mathbf{U}^i)^T (B^T B)^{-1} (\mathbf{x}^i - B^T \boldsymbol{\mu} - B^T C^T \mathbf{U}^i) \Big]$$

由于 $B$ 和 $C$ 无闭式解，使用梯度上升配合掩码约束进行优化。$\boldsymbol{\mu}$ 有闭式更新：

$$\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^N \Big( (B^T)^{-1} \mathbf{x}^i - C^T \boldsymbol{\mu}_{\mathbf{U}^i | \mathbf{x}^i} \Big)$$

**关键策略**：$B$ 和 $C$ 交替优化，每轮内对所有样本做梯度更新直到收敛，再更新另一个矩阵，最后更新 $\boldsymbol{\mu}$，整体迭代至收敛。

## 实验关键数据

### 主实验

在两个经典因果图（Frontdoor 和 Napkin）上生成合成数据，样本量 10,000，比较原始与估计的干预分布。

| 因果图 | 干预查询 | 原始分布 | 估计分布 | 均值误差 |
|--------|----------|----------|----------|----------|
| Frontdoor | $P(X_3 \| do(X_2=1))$ | $\mathcal{N}(1.1, 1.09)$ | $\mathcal{N}(1.1018, 1.069)$ | 0.0018 |
| Frontdoor | $P(X_3 \| do(X_1=1))$ | $\mathcal{N}(0.74, 1.9)$ | $\mathcal{N}(0.7391, 1.881)$ | 0.0009 |
| Napkin | $P(X_4 \| do(X_3=1))$ | $\mathcal{N}(0.3, 1.16)$ | $\mathcal{N}(0.3051, 1.1692)$ | 0.0051 |
| Napkin | $P(X_4 \| do(X_1=1))$ | $\mathcal{N}(-1.068, 2.3248)$ | $\mathcal{N}(-0.9721, 2.3274)$ | 0.0959 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| GL-SCM 直接估计 | 不可行 | 参数过多，有限数据下无法识别 |
| CGL-SCM + EM | 均值误差 < 0.1 | 标准化后参数减少，估计可行 |
| 掩码约束启用 | 保持图结构 | 消除不存在边的虚假权重 |
| 掩码约束关闭 | 图结构破坏 | 优化后出现不应存在的边 |

### 关键发现

1. **高精度恢复**：Frontdoor 图上干预分布的均值估计误差小于 0.002，方差误差小于 0.02
2. **Napkin 图更具挑战性**：$P(X_4|do(X_1=1))$ 的均值误差达到 0.096，因为该查询涉及更复杂的路径和更多潜在变量
3. **CGL-SCM 等价性得到验证**：尽管简化了参数空间，学到的模型仍能准确恢复因果分布

## 亮点与洞察

1. **简洁而深刻的理论贡献**：CGL-SCM 的等价性证明非常优雅——只需将外生变量的均值和方差信息"吸收"进结构方程的系数中，即可在不损失任何因果信息的前提下大幅减少参数
2. **实用的方法论**：结合向量化表示、掩码梯度、EM 算法三个工具，将理论上的可识别性问题转化为实际可求解的优化问题
3. **"不需要恢复真实模型"的洞察**：利用因果效应可识别性的定义，只需找到观测分布一致的模型即可，避免了参数不可识别的根本困难

## 局限与展望

1. **仅限高斯线性模型**：无法处理非线性关系或非高斯噪声，适用范围有限
2. **需要已知因果图**：实际中因果图往往未知，需要与因果发现方法结合
3. **实验规模较小**：仅在 3-4 个变量的经典小图上验证，缺乏大规模因果图上的可扩展性分析
4. **收敛性分析缺失**：未给出 EM 算法的收敛速率或有限样本误差界
5. **未探索非可识别查询**：作者在结论中提到可为非可识别查询建立有限数据下的界，但未做
6. **与机器学习方法缺乏对比**：未与 Double Machine Learning 等现有有限数据因果估计方法比较

## 相关工作与启发

- **非参数因果推断**：Pearl do-calculus、Tian & Pearl (2002) 的 c-component 分解、Shpitser & Pearl (2007) 的反事实可检验性
- **线性 SCM 可识别性**：Brito & Pearl (2002) 的图准则、Kumor et al. (2019, 2020) 的 instrumental/auxiliary cutsets
- **有限数据因果估计**：Jung et al. (2021) 的 Double Machine Learning 方法，但限于非参数设定
- **启发**：将参数化假设与非参数可识别性理论结合，是有限数据下因果推断的重要方向；标准化技巧可推广到其他参数族

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 3 | CGL-SCM 的提出有一定新意，但技术上并非大突破 |
| 理论深度 | 4 | 等价性证明严谨，EM推导完整 |
| 实验充分性 | 2 | 仅两个小型合成图，缺乏真实数据和可扩展性实验 |
| 写作质量 | 3 | 结构清晰但篇幅较短，部分细节在附录中 |
| 实用价值 | 3 | 对高斯线性场景有用，但适用范围窄 |
| **总分** | **3.0** | 理论贡献扎实但实验验证不足 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models](../../ICLR2026/causal_inference/distributional_equivalence_in_linear_non-gaussian_latent-variable_cyclic_causal_.md)
- [\[ICML 2025\] Isolated Causal Effects of Natural Language](isolated_causal_effects_of_natural_language.md)
- [\[NeurIPS 2025\] Transferring Causal Effects using Proxies](../../NeurIPS2025/causal_inference/transferring_causal_effects_using_proxies.md)
- [\[NeurIPS 2025\] An Analysis of Causal Effect Estimation Using Outcome Invariant Data Augmentation](../../NeurIPS2025/causal_inference/an_analysis_of_causal_effect_estimation_using_outcome_invariant_data_augmentatio.md)
- [\[ICML 2025\] Causal Abstraction Inference under Lossy Representations](causal_abstraction_inference_under_lossy_representations.md)

</div>

<!-- RELATED:END -->
