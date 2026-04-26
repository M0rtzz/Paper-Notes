---
title: >-
  [论文解读] On Measuring Long-Range Interactions in Graph Neural Networks
description: >-
  [ICML 2025][图学习][长距离交互] 首次从第一性原理出发形式化定义图任务中的"长距离交互"，推导出唯一满足四条公理的 range measure $\hat{\rho}_u = \mathbb{E}_{v \sim I_u}[d_G(u,v)]$，通过合成实验验证其有效性后，用该度量揭示 LRGB 基准中的 peptides 任务实际上是短距离的。
tags:
  - ICML 2025
  - 图学习
  - 长距离交互
  - 图神经网络
  - 消息传递
  - LRGB 基准
  - 过压缩
---

# On Measuring Long-Range Interactions in Graph Neural Networks

**会议**: ICML 2025  
**arXiv**: [2506.05971](https://arxiv.org/abs/2506.05971)  
**代码**: [range-measure](https://github.com/BenGutteridge/range-measure)  
**领域**: 图学习  
**关键词**: 长距离交互, GNN Range Measure, 消息传递, LRGB 基准, 过压缩

## 一句话总结

首次从第一性原理出发形式化定义图任务中的"长距离交互"，推导出唯一满足四条公理的 range measure $\hat{\rho}_u = \mathbb{E}_{v \sim I_u}[d_G(u,v)]$，通过合成实验验证其有效性后，用该度量揭示 LRGB 基准中的 peptides 任务实际上是短距离的。

## 研究背景与动机

**领域现状**：GNN 研究中"长距离"是核心话题——over-smoothing、over-squashing、graph rewiring 等大量工作都围绕如何让 GNN 捕获远距离依赖展开。LRGB（Long Range Graph Benchmark）是验证长距离能力的标准基准。

**现有痛点**：(1) "长距离"概念模糊，现有基准仅通过图规模/直径和领域直觉来判定任务是否长距离，缺乏理论支撑；(2) 合成长距离任务过于简化，简单的重布线就能转为短距离任务；(3) LRGB 对超参数高度敏感，其任务是否真正"长距离"受到质疑；(4) 现有的少数度量（如 problem radius、influence score）要么过于粗糙，要么缺乏公理化推导。

**核心矛盾**：缺乏一个理论上有根基的、可量化的度量来回答"一个任务/模型到底有多长距离？"。没有这个度量，就无法可靠地评估架构改进是否真正解决了长距离问题。

**本文目标** (1) 给出长距离交互的形式化定义；(2) 推导一族满足公理性质的 range measure；(3) 用该度量重新审视 LRGB 基准。

**切入角度**：作者从线性任务的公理化出发——一个好的 range measure 应满足局部性、单位交互、可加性、齐次性四条性质。从这些公理可以唯一推导出 range 的形式。

**核心 idea**：用"距离 × 影响力"的加权期望作为 range 的唯一公理化度量，并将其从线性任务推广到非线性 GNN（通过 Jacobian/Hessian 近似）。

## 方法详解

### 整体框架

给定一个图 $G$、一个图上的任务 $\mathbf{F}$（将节点特征映射到输出）、以及图上的距离度量 $d_G$，输出每个节点 $u$ 的 range 值 $\hat{\rho}_u$，再聚合为图级/数据集级 range。对于节点级任务用 Jacobian（一阶），对图级任务用 Hessian（二阶）来获取节点间的成对交互强度。

### 关键设计

1. **节点级 Range Measure（Jacobian 基）**:

    - 功能：度量节点级任务中每个节点的交互范围
    - 核心思路：定义四条公理——局部性（$\rho_u$ 只依赖 $u$ 收到的交互）、单位交互（单一交互的 range 等于距离）、可加性（不相交交互的 range 可加）、齐次性（缩放交互强度等比缩放 range）。从这四条公理唯一推导出：$\rho_u(\mathbf{L}) = \sum_v |\mathbf{L}_{uv}| \cdot d_G(u,v)$。归一化版本将 Property 2+3 替换为加权平均性质，得到：$\hat{\rho}_u = \frac{1}{\sum_v |\mathbf{L}_{uv}|} \sum_v |\mathbf{L}_{uv}| \cdot d_G(u,v)$，即影响力分布 $I_u$ 下距离的期望
    - 设计动机：归一化版本避免了"大量短距离交互导致高 range"的问题，使得 range 反映的是平均交互距离而非总交互量

2. **推广到非线性 GNN**:

    - 功能：将线性任务的 range 推广到任意可微映射
    - 核心思路：对非线性映射 $\mathbf{F}$ 使用 Jacobian $\frac{\partial \mathbf{F}_u}{\partial \mathbf{x}_v}$ 作为交互强度的度量。影响力分布定义为 $I_u(v) = \frac{1}{N_u}\sum_{\alpha,\beta}|\frac{\partial \mathbf{F}_u^\alpha}{\partial \mathbf{x}_v^\beta}|$，range 为 $\hat{\rho}_u = \mathbb{E}_{v \sim I_u}[d_G(u,v)]$。支持多输入/输出通道，且概率视角允许快速随机近似
    - 设计动机：Jacobian 是最佳线性近似，自然继承了线性情况下的唯一性保证

3. **图级 Range Measure（Hessian 基）**:

    - 功能：度量图级任务中的节点间交互范围
    - 核心思路：图级任务的 Jacobian 只是向量，无法捕获成对交互。因此使用 Hessian（Taylor 展开二阶项）：$\hat{\eta}_u = \mathbb{E}_{v \sim J_u}[d_G(u,v)]$，其中 $J_u(v) \propto \sum_{\alpha,\beta,\gamma}|\frac{\partial^2 \mathbf{y}^\gamma}{\partial \mathbf{x}_u^\alpha \partial \mathbf{x}_v^\beta}|$
    - 设计动机：二阶导数天然捕获节点间的特征混合程度，与 Giovanni et al. (2024) 提出的 mixing measure 一致

### Range 粒度层级

定义了节点级 → 图级 → 数据集级的聚合层级（逐级取平均），支持 transductive 和 inductive 设置。

## 实验关键数据

### 合成实验：Range 随任务变化

| 任务类型 | $k$=1 range | $k$=5 range | 增长趋势 |
|---------|-------------|-------------|---------|
| $k$-Dirac | ~1.0 | ~5.0 | 线性增长 |
| $k$-Rectangle | ~0.5 | ~2.5 | 线性增长 |
| $k$-Power | ~0.5 | ~1.3 | 亚线性增长 |

### LRGB 基准评估

| 模型 | vocsuperpixels (F1↑) | range (hop) | peptides-func (AP↑) | range (hop) | peptides-struct (MAE↓) | range (hop) |
|------|---------------------|-------------|---------------------|-------------|----------------------|-------------|
| GCN | 低 | ~2 | 好 | ~1 | 好 | ~1 |
| GINE | 中 | ~2.5 | 好 | ~1 | 好 | ~1 |
| GatedGCN | 中 | ~3 | 好 | ~1 | 好 | ~1 |
| GCN+VN | 中高 | ~5 | 中 | ~3 | 中 | ~3 |
| GPS | **高** | **~10** | 差 | ~6 | 差 | ~6 |
| GT | 高 | ~10 | 差 | ~10 | 差 | ~10 |

### 关键发现

- **vocsuperpixels 确实是长距离任务**：model range 与性能正相关——GPS（range~10）表现最好，纯 MPNN（range~2-3）最差。更高的 range 带来更好的性能
- **peptides 任务并非真正长距离**：model range 与性能**负相关**——MPNN 们 range~1 却表现最好，GPS/GT 的高 range 反而有害。这挑战了 LRGB 的基本假设
- 训练过程中，MPNN 的 range 在最初几个 epoch 快速增长后稳定，说明模型先学局部再尝试远距离
- Peptides-struct 上 GCN 至今仍是 SOTA，进一步证实该任务本质上是局部的

## 亮点与洞察

- **公理化推导的优雅性**：从四条直观的公理出发，唯一确定了 range measure 的函数形式，不需要任何启发式选择。这种推导方式在 GNN 理论中并不常见，值得借鉴
- **"影响力 × 距离"的概率解释**：归一化 range 本质上是"在影响力分布下距离的期望"，这不仅概念清晰，还允许通过随机采样进行高效近似，使该度量在大规模图上也可用
- **挑战了 LRGB 的权威性**：peptides 任务被广泛用于验证长距离架构的优越性，但本文的度量显示 MPNN 在 range~1 时就能达到最佳性能。这意味着很多声称"解决了长距离问题"的架构可能只是在做更好的局部特征提取

## 局限与展望

- Hessian 基 range 的计算成本很高，论文在 LRGB 实验中实际采用 Jacobian 基的近似，两者之间的定量关系需要更多验证
- Range measure 基于 Jacobian/Hessian 的局部线性近似，对高度非线性的深层 GNN（如带 ReLU 的 GNN），局部近似的质量可能不稳定
- 论文主要分析了 LRGB 中的三个任务，对更多真实世界数据集（特别是分子属性预测、蛋白质结构）的分析将更有说服力
- 该度量目前只能对"已训练好的模型"计算 range，无法在训练前预测一个架构的 range 能力

## 相关工作与启发

- **vs Alon & Yahav (2020) 的 problem radius**：problem radius 是任务的固有属性（解所需的最少跳数），而本文的 range 可以同时度量任务和模型，且有严格的公理化基础
- **vs Di Giovanni et al. (2023) 的 over-squashing 分析**：他们用成对节点的灵敏度分析来研究信息传播瓶颈，本文将此扩展为一族系统化的度量
- **vs Liang et al. (2025)**：他们也提出了基于影响力的度量来构建新的长距离基准，本文指出他们的度量是本文框架的特殊实例

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次从第一性原理公理化推导 GNN 的 range measure，方法论贡献突出
- 实验充分度: ⭐⭐⭐⭐ 合成实验验证充分，LRGB 分析深入，但缺少更多真实任务的分析
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰优雅，图表精心设计，整体可读性很高
- 价值: ⭐⭐⭐⭐⭐ 为 GNN 社区提供了核心基础工具，并挑战了主流基准的假设，影响深远

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](../../NeurIPS2025/graph_learning/sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)
- [\[ICLR 2026\] Improving Long-Range Interactions in Graph Neural Simulators via Hamiltonian Dynamics](../../ICLR2026/graph_learning/improving_long-range_interactions_in_graph_neural_simulators_via_hamiltonian_dyn.md)
- [\[ICML 2025\] A Cognac Shot To Forget Bad Memories: Corrective Unlearning for Graph Neural Networks](a_cognac_shot_to_forget_bad_memories_corrective_unlearning_for_graph_neural_netw.md)
- [\[ICLR 2026\] Are We Measuring Oversmoothing in Graph Neural Networks Correctly?](../../ICLR2026/graph_learning/are_we_measuring_oversmoothing_in_graph_neural_networks_correctly.md)
- [\[ICML 2025\] Hyperbolic-PDE GNN: Spectral Graph Neural Networks in the Perspective of A System of Hyperbolic Partial Differential Equations](hyperbolic-pde_gnn_spectral_graph_neural_networks_in_the_perspective_of_a_system.md)

<!-- RELATED:END -->
