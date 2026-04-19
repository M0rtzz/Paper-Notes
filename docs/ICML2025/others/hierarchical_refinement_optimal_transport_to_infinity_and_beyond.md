---
title: >-
  [论文解读] Hierarchical Refinement: Optimal Transport to Infinity and Beyond
description: >-
  [ICML 2025][optimal transport] 提出 Hierarchical Refinement (HiRef) 算法，通过递归求解低秩 OT 子问题动态构建多尺度数据划分，最终恢复双射 Monge 映射，实现 $O(n\log n)$ 时间和 $O(n)$ 空间复杂度，将全秩 OT 扩展到百万级数据。
tags:
  - ICML 2025
  - optimal transport
  - Monge Map
  - Low-rank OT
  - Hierarchical Algorithm
  - Scalability
---

# Hierarchical Refinement: Optimal Transport to Infinity and Beyond

**会议**: ICML 2025  
**arXiv**: [2503.03025](https://arxiv.org/abs/2503.03025)  
**代码**: 有  
**领域**: 最优传输  
**关键词**: optimal transport, Monge Map, Low-rank OT, Hierarchical Algorithm, Scalability

## 一句话总结
提出 Hierarchical Refinement (HiRef) 算法，通过递归求解低秩 OT 子问题动态构建多尺度数据划分，最终恢复双射 Monge 映射，实现 $O(n\log n)$ 时间和 $O(n)$ 空间复杂度，将全秩 OT 扩展到百万级数据。

## 研究背景与动机
**领域现状**：Sinkhorn 算法解决了 OT 的 $O(n^3)$ 问题但仍有 $O(n^2)$ 时空复杂度；低秩 OT 降至线性但无法给出一一对应。

**现有痛点**：现代数据规模达百万级，Sinkhorn 二次复杂度不可行；mini-batch OT 有固有偏差；神经 OT 映射质量不稳定。

**核心矛盾**：如何在保持线性复杂度的同时恢复全秩双射映射？

**切入角度**：证明低秩 OT 的因子矩阵 (Q,R) 会将 Monge 对共聚类（Proposition 3.1），据此递归细化。

**核心idea**：低秩 OT 的最优共聚类保证 $\mathsf{q}^*(x) = \mathsf{r}^*(T^*(x))$，递归到单点即得 Monge 映射。

## 方法详解

### 整体框架
初始将两个数据集视为单一簇对 → 在每个簇对上求解 rank-$r$ 低秩 OT → 根据因子矩阵分裂为 $r$ 个子簇对 → 递归直到每个簇对包含单个点 → 输出双射映射。

### 关键设计

1. **核心理论 (Proposition 3.1)**:

    - 在严格 $r$-Monge 可分条件下，低秩 OT 的最优因子共聚类 Monge 对
    - 即 $\mathsf{q}^*(x) = \mathsf{r}^*(T^*(x))$，源点和其 Monge 像被分到同一标签
    - 保证每步细化都正确保留 Monge 配对关系

2. **Rank Annealing Schedule**:

    - 层级深度 $\kappa$，每层使用 rank $r_t$，有效 rank 为 $\rho_t = \prod_{s=1}^t r_s$
    - 当 $r=2$ 时，最优 Q,R 自动是硬划分（Lemma B.5）
    - 序列 $(r_1,...,r_\kappa)$ 可适配不同数据规模

3. **代价递减保证 (Proposition 3.4)**:

    - 每步细化的传输代价非增：$0 \leq \Delta_{t,t+1} \leq \|\nabla c\|_\infty \cdot \text{mean\_diam}(\Gamma_t)$
    - 簇对直径随细化层级递减

### 损失函数 / 训练策略
使用 uniform 内边际约束 $\mathbf{g} = (1/r)\mathbf{1}_r$ 强制均匀分裂，在每个子问题上调用低秩 OT 求解器。

## 实验关键数据

### 主实验
| 数据集规模 | 方法 | Primal Cost | 时间 | 空间 |
|-----------|------|------------|------|------|
| $10^5$ | Sinkhorn | 基准 | 数小时 | $O(n^2)$ |
| $10^5$ | HiRef | ≈Sinkhorn | 分钟级 | $O(n)$ |
| $10^6$ | HiRef | 可计算 | 小时级 | $O(n)$ |
| $10^6$ | Sinkhorn | 不可行 | - | 内存溢出 |

### 消融实验
| 配置 | 效果 | 说明 |
|------|------|------|
| HiRef (rank=2) | 最稳定 | 自动硬划分 |
| HiRef (rank=4) | 更快收敛 | 每步分裂更多 |
| Mini-batch OT | 有偏差 | 局部近似全局耦合 |

### 关键发现
- HiRef 在高维空间中常常匹配甚至超过 Sinkhorn 的 primal cost
- 首次将 OT 扩展到百万级数据，无需 mini-batch
- 自动构建多尺度划分，无需预先给定层次结构

## 亮点与洞察
- Proposition 3.1 是极优雅的理论结果：低秩 OT 天然进行了"正确"的共聚类
- 算法设计简洁但理论基础深厚，连接了低秩OT、聚类和分治策略
- 可用于大规模生成模型训练中的 OT 匹配（Flow Matching等）

## 局限与展望
- 严格 $r$-Monge 可分条件在实际数据中可能不完全满足
- 实际使用的低秩 OT 求解器是近似的，可能影响共聚类质量
- 对非均匀分布的数据可能需要调整 rank schedule

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 理论贡献突出
- 实验充分度: ⭐⭐⭐⭐ 百万级实验有说服力
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨清晰
- 价值: ⭐⭐⭐⭐⭐ 对 OT 领域有重要推动

<!-- RELATED:START -->

## 相关论文

- [LaCoOT: Layer Collapse through Optimal Transport](../../ICCV2025/others/lacoot_layer_collapse_through_optimal_transport.md)
- [Theoretical Performance Guarantees for Partial Domain Adaptation via Partial Optimal Transport](theoretical_performance_guarantees_for_partial_domain_adaptation_via_partial_opt.md)
- [Lightspeed Geometric Dataset Distance via Sliced Optimal Transport](lightspeed_geometric_dataset_distance_via_sliced_optimal_transport.md)
- [Quantifying Lexical Semantic Shift via Unbalanced Optimal Transport](../../ACL2025/others/quantifying_lexical_semantic_shift_via_unbalanced_optimal_transport.md)
- [To Each Metric Its Decoding: Post-Hoc Optimal Decision Rules of Probabilistic Hierarchical Classifiers](to_each_metric_its_decoding_post-hoc_optimal_decision_rules_of_probabilistic_hie.md)

<!-- RELATED:END -->
