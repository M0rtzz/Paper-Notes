---
title: >-
  [论文解读] Runtime Analysis of Evolutionary NAS for Multiclass Classification
description: >-
  [ICML2025][进化神经架构搜索] 首次对进化神经架构搜索(ENAS)在多类分类问题上进行运行时理论分析，证明 one-bit 和 bit-wise 变异的 (1+1)-ENAS 算法均以 $O(rM\ln rM)$ 期望运行时找到最优架构，说明简单的 one-bit 变异即可与复杂的 bit-wise 变异媲美。
tags:
  - "ICML2025"
  - "进化神经架构搜索"
  - "运行时分析"
  - "多类分类"
  - "(1+1)-EA"
  - "适应度函数"
---

# Runtime Analysis of Evolutionary NAS for Multiclass Classification

**会议**: ICML2025  
**arXiv**: [2506.06019](https://arxiv.org/abs/2506.06019)  
**代码**: 无  
**领域**: NAS理论 / 进化计算理论  
**关键词**: 进化神经架构搜索, 运行时分析, 多类分类, (1+1)-EA, 适应度函数

## 一句话总结

首次对进化神经架构搜索(ENAS)在多类分类问题上进行运行时理论分析，证明 one-bit 和 bit-wise 变异的 (1+1)-ENAS 算法均以 $O(rM\ln rM)$ 期望运行时找到最优架构，说明简单的 one-bit 变异即可与复杂的 bit-wise 变异媲美。

## 研究背景与动机

进化神经架构搜索(ENAS)利用进化算法自动设计深度神经网络架构，在实际任务中表现优异。然而，ENAS 的**理论研究严重滞后**于实践：

- **运行时分析**是进化算法理论的核心课题，但 ENAS 的运行时分析几乎空白
- 核心困难在于：ENAS 的适应度需通过训练神经网络获得，难以用数学公式表达
- 此前仅有 Fischer et al.(2023/2024) 针对固定架构的分类问题构建过适应度函数，Lv et al.(2024b) 首次为 ENAS 建立了适应度函数，但仅限**二分类**
- 实际应用（图像识别、语音识别、医学诊断等）普遍是**多类分类**，亟需多类分类场景的理论分析
- 已有搜索空间设计也无法支持多类分类和 bit-wise 变异的分析

本文首次将 ENAS 运行时分析从二分类推广到多类分类，填补了理论空白。

## 方法详解

### 1. 神经架构建模

将 $M$ 类分类任务分解为 $M-1$ 个二分类器（cell）：

- 每个 cell 包含 $l$ 个 block 和一个 OR 神经元
- 三种 block 类型：**A-type**（线段决策区域）、**B-type**（扇形决策区域）、**C-type**（三角形决策区域）
- $M-1$ 个 cell 的输出经隐藏层 $M$ 个神经元汇聚，再通过 Softmax 得到各类概率

架构编码为 $M-1$ 个三元组：

$$\boldsymbol{x} = \{(n_A^1, n_B^1, n_C^1), \ldots, (n_A^{M-1}, n_B^{M-1}, n_C^{M-1})\}$$

### 2. 多类分类基准问题 Mcc

在单位圆 $S$ 上定义 $M$ 类分类问题：

- 单位圆均分为 $n = 2rM$ 个扇形（$r \geq 2$）
- 每类包含 $r$ 个线段区域(segment)、$r$ 个扇形区域(sector)、$r$ 个三角形区域(triangle)
- 这三类区域分别对应半空间、无界多面体、有界多面体的代表结构

### 3. 适应度函数

数学推导了闭式适应度函数：

$$\mathcal{F}(\boldsymbol{x}) = \frac{Ar_{tri} \cdot (\mathbb{I}_x + 2r) + Ar_{seg} \cdot (\mathbb{J}_x + 2r - \epsilon_x)}{\pi}$$

其中：

| 符号 | 含义 |
|------|------|
| $\mathbb{I}_x$ | 类 1 到 $M-1$ 的正确分类三角形数 |
| $\mathbb{J}_x$ | 类 1 到 $M-1$ 的正确分类线段数（含扇形内线段） |
| $\epsilon_x$ | 第 $M$ 类中被第 $M-1$ 个 cell 错误分类的线段数 |
| $Ar_{tri}$, $Ar_{seg}$ | 单个三角形、线段的面积 |

适应度由 $\mathbb{I}_x$ 项主导（因 $Ar_{tri} > (n-2r) \cdot Ar_{seg}$），类似于进化计算中经典的 OneMax 函数。

### 4. (1+1)-ENAS 算法

采用**两层变异**策略：

- **外层变异**：选择哪些 cell 进行变异
    - **One-bit**：随机选一个 cell
    - **Bit-wise**：每个 cell 以概率 $1/(M-1)$ 独立选中
- **内层变异**：对选中 cell 执行具体操作
    - **Local**：对每个选中 cell 变异一次
    - **Global**：变异 $K$ 次，$K \sim \text{Pois}(1)$
- 变异操作（Definition 2.1）：随机选 block 类型，执行 Addition / Deletion / Modification

## 理论结果

### 运行时上下界

| 算法 | 上界 | 下界 |
|------|------|------|
| (1+1)-ENAS$_{\text{onebit}}$ | $O(rM\ln(rM))$ | $\Omega(rM\ln M)$ |
| (1+1)-ENAS$_{\text{bitwise}}$ | $O(rM\ln(rM))$ | $\Omega(rM\ln M)$ |

**核心发现**：两种变异策略的运行时上下界**几乎一致**，差距仅在 $\ln r$ 因子。

### 证明框架

将优化过程分为两阶段：

| 阶段 | 目标 | 距离函数 | 漂移下界 |
|------|------|----------|----------|
| Phase 1 | $\mathbb{I}_x = N = 2r(M-1)$ | $V_1(x) = N - \mathbb{I}_x$ | $\geq (V_1/N) \cdot 2/9$ |
| Phase 2 | $\mathbb{J}_x - \epsilon_x = N$ | $V_2(x) = N - (\mathbb{J}_x - \epsilon_x)$ | $\geq (V_2/N) \cdot 1/9$ |

两阶段均应用**乘法漂移定理**(Multiplicative Drift Theorem)得 $O(N\ln N)$，合计 $O(rM\ln(rM))$。

### 搜索空间划分

基于 $\mathbb{I}_x$ 和 $\mathbb{J}_x - \epsilon_x$ 的值，将搜索空间 $\mathcal{S}$ 二级划分：

- 一级划分：$\mathcal{S}_0 <_{\mathcal{F}} \mathcal{S}_1 <_{\mathcal{F}} \cdots <_{\mathcal{F}} \mathcal{S}_{n-2r}$
- 二级划分：$\mathcal{S}_i^0 <_{\mathcal{F}} \mathcal{S}_i^1 <_{\mathcal{F}} \cdots <_{\mathcal{F}} \mathcal{S}_i^{n-2r}$

此层次划分保证了适应度的单调性，使漂移分析得以进行。

## 亮点与洞察

1. **首个多类分类 ENAS 运行时分析**：从二分类推广到 $M$ 类，具有开创性意义
2. **简单变异即够用**：理论证明 one-bit 变异与 bit-wise 变异运行时相当，挑战了"复杂变异更优"的直觉——这对 ENAS 算法设计有直接指导意义
3. **可数学化的 ENAS 适应度函数**：绕过了黑盒训练过程，通过几何性质直接刻画分类精度，为后续 ENAS 理论分析提供了基准工具
4. **两层搜索空间设计**：cell 层 + block 层的编码方式与 ENAS 社区主流做法一致，增强了理论结果的实用性
5. **漂移分析技术的扩展应用**：将乘法漂移分析从标准 EA 推广到 ENAS 的两层搜索空间

## 局限与展望

1. **问题过于理想化**：Mcc 基准中数据区域均匀分布在单位圆上，远离真实多类分类数据分布
2. **仅分析 (1+1) 模式**：实际 ENAS 使用种群、交叉等更复杂策略，(1+1) 框架难以反映实际算法行为
3. **假设最优参数可达**：分析中假设每个架构的参数均能达到最优，跳过了训练过程的复杂性
4. **二维输入限制**：输入固定为二维，无法刻画高维特征空间中的架构搜索
5. **最优架构的实际意义有限**：Mcc 的最优架构是一组 block 计数约束，与真实 NAS 中的架构质量差异较大
6. **缺少与实际 ENAS 算法的对比实验**：仅有理论验证性实验，未在标准 NAS benchmark 上测试

## 相关工作与启发

- **经典 EA 运行时分析**：OneMax、LeadingOnes 上的 (1+1)-EA 分析 (Droste et al., 2002; Witt, 2013)
- **NAS 分类适应度函数**：Fischer et al.(2023/2024) 的超平面/弯曲超平面方法
- **二分类 ENAS 分析**：Lv et al.(2024b) 的 polytope-based 决策边界方法，本文直接扩展于此
- **启发**：可尝试将此框架推广到更大种群、交叉算子、或动态搜索空间设定

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个多类分类 ENAS 运行时分析，理论贡献明确
- 实验充分度: ⭐⭐⭐ — 有验证性实验支持理论，但缺少实际 NAS 场景对比
- 写作质量: ⭐⭐⭐⭐ — 证明框架清晰，符号体系一致
- 价值: ⭐⭐⭐ — 理论意义突出但实际影响有限，适用范围受理想化假设约束

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Theoretical and Empirical Analysis of Lehmer Codes to Search Permutation Spaces with Evolutionary Algorithms](../../AAAI2026/others/theoretical_and_empirical_analysis_of_lehmer_codes_to_search_permutation_spaces_.md)
- [\[NeurIPS 2025\] Evolutionary Prediction Games](../../NeurIPS2025/others/evolutionary_prediction_games.md)
- [\[ICML 2025\] The Price of Freedom: Exploring Expressivity and Runtime Tradeoffs in Equivariant Networks](the_price_of_freedom_exploring_expressivity_and_runtime_tradeoffs_in_equivariant.md)
- [\[ICML 2025\] Continuous-Time Analysis of Heavy Ball Momentum in Min-Max Games](continuous-time_analysis_of_heavy_ball_momentum_in_min-max_games.md)
- [\[AAAI 2026\] Improved Runtime Guarantees for the SPEA2 Multi-Objective Optimizer](../../AAAI2026/others/improved_runtime_guarantees_for_the_spea2_multi-objective_optimizer.md)

</div>

<!-- RELATED:END -->
