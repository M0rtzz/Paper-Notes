---
title: >-
  [论文解读] Hierarchical Refinement: Optimal Transport to Infinity and Beyond
description: >-
  [ICML 2025][最优传输] 提出 Hierarchical Refinement (HiRef) 算法，通过递归求解低秩最优传输子问题来动态构建多尺度数据分区，以对数线性时间和线性空间复杂度获得完整的双射 Monge 映射，将最优传输扩展到百万级数据集。
tags:
  - "ICML 2025"
  - "最优传输"
  - "低秩分解"
  - "层次细化"
  - "Monge映射"
  - "大规模对齐"
---

# Hierarchical Refinement: Optimal Transport to Infinity and Beyond

**会议**: ICML 2025 Oral  
**arXiv**: [2503.03025](https://arxiv.org/abs/2503.03025)  
**代码**: 无  
**领域**: 最优传输 / 算法优化  
**关键词**: 最优传输, 低秩分解, 层次细化, Monge映射, 大规模对齐

## 一句话总结
提出 Hierarchical Refinement (HiRef) 算法，通过递归求解低秩最优传输子问题来动态构建多尺度数据分区，以对数线性时间和线性空间复杂度获得完整的双射 Monge 映射，将最优传输扩展到百万级数据集。

## 研究背景与动机

**领域现状**：最优传输（OT）在机器学习中广泛用于数据对齐，Sinkhorn 算法是最常用的求解器，具有 $O(n^2 \log n)$ 时间和 $O(n^2)$ 空间复杂度。低秩 OT 将空间复杂度降到 $O(nr)$，但无法产生一对一的点对应关系。

**现有痛点**：(1) Sinkhorn 的二次空间复杂度限制其处理超过 ~$10^4$ 个点的数据集；(2) 低秩 OT 虽然高效但只产生分辨率有限的近似耦合，无法恢复精确的双射映射；(3) Mini-batch OT 引入系统性偏差，且不计算全局对齐；(4) 神经 OT 方法在恢复忠实映射方面存在局限。

**核心矛盾**：全秩 OT 能产生精确的 Monge 映射但计算代价是二次的，低秩 OT 计算高效但无法提供一对一对应——需要一种兼具两者优势的方法。

**本文目标**：设计一种算法，在保持低秩 OT 的线性空间复杂度的同时，恢复出完整的双射 Monge 映射，并扩展到百万级数据。

**切入角度**：作者发现低秩 OT 的最优因子矩阵 $(Q^*, R^*)$ 具有一个关键不变量：它们将每个源点与其在 Monge 映射下的像分配到同一个簇中。这意味着可以利用低秩 OT 来正确地"共聚类"源-目标点对。

**核心 idea**：递归地用低秩 OT 将数据分成越来越细的共聚类，直到每个簇只包含一个点对，从而恢复完整的 Monge 映射——效果等价于全秩 OT，但复杂度仅为对数线性。

## 方法详解

### 整体框架
HiRef 采用自顶向下的多尺度策略。初始时整个源数据集 $\mathsf{X}$ 和目标数据集 $\mathsf{Y}$ 构成一个粗粒度共聚类。在每一层，对每个共聚类求解一个低秩 OT 子问题将其进一步细分为 $r$ 个更小的共聚类。经过 $\kappa = \log_r n$ 层的递归细化，每个共聚类恰好包含一个 $(x_i, T(x_i))$ 点对，即恢复了 Monge 映射。

### 关键设计

1. **共聚类不变量 (Proposition 3.1)**:

    - 功能：为层次细化提供理论保证——每一层的低秩 OT 都能正确地将 Monge 对分到同一个簇
    - 核心思路：证明当最优低秩因子 $(Q^*, R^*)$ 对应硬聚类时，源点的聚类标签 $q^*(x)$ 等于其 Monge 像的聚类标签 $r^*(T^*(x))$。关键条件是代价矩阵满足"严格 $r$-Monge 可分性"，此时最优因子可分解为对称形式 $Q = P^\dagger R$（$P^\dagger$ 是 Monge 耦合）
    - 设计动机：这个理论结果是整个算法正确性的基石——如果低秩 OT 不能正确地共聚类 Monge 对，递归细化就无法收敛到正确的映射

2. **秩退火调度 (Rank-Annealing Schedule)**:

    - 功能：控制每层的细化程度，平衡计算效率和内存约束
    - 核心思路：选择因子序列 $(r_1, r_2, \ldots, r_\kappa)$ 使得 $\prod r_i = n$。秩与熵正则化参数 $\epsilon$ 呈反比关系，因此从小秩到大秩的递进等价于从高 $\epsilon$ 到低 $\epsilon$ 的退火。通过动态规划在 $O(r_{\max} \kappa n)$ 时间内找到最优调度，最小化累积子问题总秩
    - 设计动机：秩的选择直接影响复杂度和正确性。均匀的秩约束 $g = 1_r/r$ 确保均匀分割；二元调度 ($r=2$) 自动满足硬分区条件

3. **层次块耦合与成本递减 (Proposition 3.4)**:

    - 功能：保证每层细化都能改善传输成本
    - 核心思路：定义尺度 $t$ 的隐式块耦合 $P^{(t)}$，证明相邻层的成本差 $\Delta_{t,t+1} \geq 0$ 且 $\Delta_{t,t+1} \leq \|\nabla c\|_\infty \cdot \text{mean\_diam}(\Gamma_t)$。这意味着细化总是降低传输成本，且改善程度受当前簇直径的上界约束
    - 设计动机：为算法的收敛性和解质量提供理论保障

### 损失函数 / 训练策略
HiRef 是一个纯算法（非学习方法），不涉及训练。核心优化目标是每一层的低秩 OT 子问题 $\min_{Q,R} \langle C, Q \text{diag}(1/g) R^\top \rangle_F$，约束 $g$ 为均匀分布。

## 实验关键数据

### 主实验

| 数据集 | HiRef 成本 | Sinkhorn 成本 | 备注 |
|--------|-----------|--------------|------|
| Half-Moon (1K) | 最优/接近 | 接近 | 两者相当，但 HiRef 输出恰好 1024 个非零项 |
| Half-Moon (1M) | 14.2 | **无法运行** | HiRef 扩展到 $2^{20}$ 数据规模 |
| MOSTA E12-13.5 (51K) | **14.35** | - | Sinkhorn 超出内存 |
| MOSTA E15-16.5 (121K) | **12.79** | - | 仅 HiRef 和低秩/mini-batch 可运行 |

### 消融实验

| 方法对比 | 成本质量 | 可扩展性 | 说明 |
|---------|---------|---------|------|
| HiRef vs Sinkhorn | 相当或更优 | HiRef >>  | HiRef 可到百万，Sinkhorn 限于 ~16K |
| HiRef vs LOT/FRLC (r=40) | HiRef 更优 | 相当 | 细化步骤降低了低秩 OT 的粗糙成本 |
| HiRef vs Mini-batch | HiRef 更优 | 相当 | HiRef 无偏差，mini-batch 有系统偏差 |
| HiRef vs MOP | HiRef 远优 | 相当 | MOP 在 Checkerboard 上成本高 2 倍 |

### 关键发现
- HiRef 输出严格的双射映射（耦合矩阵恰好 $n$ 个非零项），而 Sinkhorn/ProgOT 输出 62-68 万个非零项——占据了稠密矩阵的大部分空间
- 在单细胞转录组数据上，HiRef 的传输成本在所有时间点上都低于其他方法，包括作为子程序的 FRLC
- 运行时间呈线性增长，而 Sinkhorn 呈二次增长；成功扩展到 >100 万个点

## 亮点与洞察
- **理论与算法的精巧融合**：Proposition 3.1 的共聚类不变量将低秩 OT 的代数结构转化为可递归利用的分治保证，设计非常优雅
- **秩-熵的对偶性洞察**：低秩 ↔ 高正则化的对应关系，使得层次细化在概念上等价于 ε-退火，但在执行上更加结构化且节省内存
- 在转录组学等实际大规模应用中，HiRef 首次实现了 >10 万点的全秩对齐，填补了低秩近似和精确求解之间的空白

## 局限与展望
- 理论保证依赖"严格 $r$-Monge 可分性"假设，实际数据未必满足；算法在实践中仍有效但缺少一般性证明
- 当代价矩阵不具备低秩分解时（$d = O(n)$），退化为 $O(n^2)$ 复杂度
- 均匀秩约束 $g = 1_r/r$ 迫使每一层均匀分割，可能不适用于高度不均匀的数据分布
- 当前仅处理等大小数据集（$|X| = |Y|$），非双射场景需要扩展

## 相关工作与启发
- **vs Sinkhorn**: Sinkhorn 是 $O(n^2)$ 的金标准但不可扩展；HiRef 用对数线性时间达到相当或更优的传输成本
- **vs 低秩 OT (Scetbon et al.)**: 低秩 OT 是 HiRef 的子程序；HiRef 通过层次细化超越了低秩的有限分辨率
- **vs Gerber & Maggioni (2017)**: 同为多尺度 OT，但 MOP 需要预定义的多尺度分区且依赖环境空间网格；HiRef 自动构建分区且完全数据驱动

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 共聚类不变量的理论发现是精彩的结构性洞察，将低秩和全秩 OT 完美桥接
- 实验充分度: ⭐⭐⭐⭐ 合成数据+转录组学，覆盖从小到百万级数据，与多种基线对比充分
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，图表说明力强
- 价值: ⭐⭐⭐⭐⭐ 首次将全秩 OT 扩展到百万点级别，对计算生物学等领域有重大实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Theoretical Performance Guarantees for Partial Domain Adaptation via Partial Optimal Transport](theoretical_performance_guarantees_for_partial_domain_adaptation_via_partial_opt.md)
- [\[ICML 2025\] Lightspeed Geometric Dataset Distance via Sliced Optimal Transport](lightspeed_geometric_dataset_distance_via_sliced_optimal_transport.md)
- [\[ACL 2025\] Quantifying Lexical Semantic Shift via Unbalanced Optimal Transport](../../ACL2025/others/quantifying_lexical_semantic_shift_via_unbalanced_optimal_transport.md)
- [\[NeurIPS 2025\] Estimation of Stochastic Optimal Transport Maps](../../NeurIPS2025/others/estimation_of_stochastic_optimal_transport_maps.md)
- [\[ICCV 2025\] LaCoOT: Layer Collapse through Optimal Transport](../../ICCV2025/others/lacoot_layer_collapse_through_optimal_transport.md)

</div>

<!-- RELATED:END -->
