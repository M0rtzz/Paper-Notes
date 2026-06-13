---
title: >-
  [论文解读] Improving the Effective Receptive Field of Message-Passing Neural Networks
description: >-
  [ICML 2025][MPNN] 本文形式化了 MPNN 中有效感受野（ERF）的概念，证明节点贡献随距离指数衰减（二项式分布），并提出 IM-MPNN 架构通过多尺度图粗化和跨尺度信息交织来扩展 ERF，在 LRGB 等长程依赖基准上显著提升。 领域现状：MPNN 通过局部消息传递更新节点表示，但受 over-squas…
tags:
  - "ICML 2025"
  - "MPNN"
  - "Over-squashing"
  - "Effective Receptive Field"
  - "Multiscale"
  - "Graph Coarsening"
---

# Improving the Effective Receptive Field of Message-Passing Neural Networks

**会议**: ICML 2025  
**arXiv**: [2505.23185](https://arxiv.org/abs/2505.23185)  
**代码**: [https://github.com/BGU-CS-VIL/IM-MPNN](https://github.com/BGU-CS-VIL/IM-MPNN)  
**领域**: LLM评测  
**关键词**: MPNN, Over-squashing, Effective Receptive Field, Multiscale, Graph Coarsening

## 一句话总结
本文形式化了 MPNN 中有效感受野（ERF）的概念，证明节点贡献随距离指数衰减（二项式分布），并提出 IM-MPNN 架构通过多尺度图粗化和跨尺度信息交织来扩展 ERF，在 LRGB 等长程依赖基准上显著提升。

## 研究背景与动机
**领域现状**：MPNN 通过局部消息传递更新节点表示，但受 over-squashing 限制，远距离信息难以有效聚合。

**现有痛点**：图重连方法（SDRF等）和 Transformer 方法增加了密集通信的计算复杂度；大多方法受限于原始图分辨率。

**核心矛盾**：在不显著增加计算量的前提下，如何让 MPNN 有效捕获长程依赖？

**切入角度**：类比 CNN 中的多尺度方法，通过图粗化在粗糙尺度上用少量层实现长程通信。

**核心idea**：线性图上 $\ell$ 层后节点贡献服从 $B(\ell, 1/2)$ 二项分布，即指数衰减；粗糙尺度等效于扩大 $\kappa$ 值。

## 方法详解

### 整体框架
输入图 → 编码 → Graclus 粗化 $S$ 次得到多尺度图 → 在 $S+1$ 个尺度上并行 MP → Scale-mix 交织信息 → 重复 $L$ 次 → Unpool + 拼接 → 任务头。

### 关键设计

1. **ERF 理论分析**:

    - 线性图上 $\ell$ 层均匀权重卷积后中心节点的特征为：$x_0^\ell = \sum_{i=0}^\ell \binom{\ell}{i} v_{2i-\ell}$
    - 归一化后为二项分布 $B(\ell, 1/2)$，由 Hoeffding 不等式可得尾部贡献指数衰减
    - 连续扩散 PDE 分析得到高斯核衰减：$x(p,t) \propto \exp(-\|p-p_0\|^2 / 4\kappa t)$

2. **多尺度处理**:

    - 粗糙尺度 Laplacian 等效于更大 $\kappa$（粗化因子 2 → $\kappa$ 乘以 4）
    - 粗糙尺度上 1 hop 等效于原始图上约 4 hop，且信息不被过度压缩

3. **Scale-Mix 层**:

    - 每个节点从父节点（更粗）和子节点（更细）接收信息
    - 不同尺度使用独立参数的 MPNN backbone

### 损失函数 / 训练策略
任务相关标准损失，可搭配 GCN/GINE/GatedGCN 等不同 backbone。

## 实验关键数据

### 主实验

| 数据集 | 指标 | IM-GatedGCN | GatedGCN | 提升 |
|--------|------|------------|----------|------|
| Peptides-func | AP↑ | 0.684 | 0.653 | +3.1% |
| Peptides-struct | MAE↓ | 0.244 | 0.256 | -4.7% |
| PascalVOC-SP | F1↑ | 0.397 | 0.367 | +3.0% |

### 消融实验

| 配置 | Peptides-func AP | 说明 |
|------|-----------------|------|
| GCN baseline | 0.594 | 无多尺度 |
| IM-GCN S=1 | 0.623 | 1层粗化 |
| IM-GCN S=2 | 0.645 | 2层粗化 |
| IM-GCN S=3 | 0.659 | 3层粗化 |

### 关键发现
- 增加尺度数 $S$ 持续提升性能（图 1 可视化），说明多尺度有效扩展 ERF
- IM-MPNN 是通用框架，可提升不同 MPNN backbone 性能
- 计算效率优于全连接 Transformer 方法

## 亮点与洞察
- ERF 的理论分析清晰优雅：从离散二项分布到连续扩散方程的双重视角
- 多尺度方法从 CNN 迁移到 GNN 的思路自然但验证充分
- 框架通用性强，可作为任何 MPNN 的即插即用增强

## 局限与展望
- Graclus 粗化是固定的预处理，不可学习
- 对非常异构的图拓扑，粗化质量可能不一致
- 未与最新 Graph Transformer 方法做全面对比

## 评分
- 新颖性: ⭐⭐⭐⭐ ERF分析新颖，方法增量性
- 实验充分度: ⭐⭐⭐⭐ LRGB基准全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论动机清晰
- 价值: ⭐⭐⭐⭐ 对GNN长程依赖问题有实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Permutation Equivariant Neural Networks for Symmetric Tensors](permutation_equivariant_neural_networks_for_symmetric_tensors.md)
- [\[ICML 2025\] Improved Exploration in GFlowNets via Enhanced Epistemic Neural Networks](improved_exploration_in_gflownets_via_enhanced_epistemic_neural_networks.md)
- [\[ICLR 2026\] Improving Set Function Approximation with Quasi-Arithmetic Neural Networks](../../ICLR2026/others/improving_set_function_approximation_with_quasi-arithmetic_neural_networks.md)
- [\[ICML 2025\] Truly Self-Improving Agents Require Intrinsic Metacognitive Learning](truly_self-improving_agents_require_intrinsic_metacognitive_learning.md)
- [\[NeurIPS 2025\] Depth-Bounds for Neural Networks via the Braid Arrangement](../../NeurIPS2025/others/depth-bounds_for_neural_networks_via_the_braid_arrangement.md)

</div>

<!-- RELATED:END -->
