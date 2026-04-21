---
title: >-
  [论文解读] Tractable Weighted First-Order Model Counting with Bounded Treewidth Binary Evidence
description: >-
  [AAAI 2026][加权一阶模型计数] 提出一种在域大小上多项式时间的算法，计算带有有界树宽二元证据的 $\text{FO}^2$ 和 $\text{C}^2$ 片段的加权一阶模型计数（WFOMC），并解决了有界树宽有界度图上的稳定座位安排开放问题。
tags:
  - AAAI 2026
  - 加权一阶模型计数
  - 二元证据
  - 树宽
  - 提升推理
  - 组合计数
---

# Tractable Weighted First-Order Model Counting with Bounded Treewidth Binary Evidence

**会议**: AAAI 2026  
**arXiv**: [2511.09174](https://arxiv.org/abs/2511.09174)  
**代码**: 无  
**领域**: 人工智能 / 逻辑推理 (AI / Logic & Reasoning)  
**关键词**: 加权一阶模型计数, 二元证据, 树宽, 提升推理, 组合计数

## 一句话总结

提出一种在域大小上多项式时间的算法，计算带有有界树宽二元证据的 $\text{FO}^2$ 和 $\text{C}^2$ 片段的加权一阶模型计数（WFOMC），并解决了有界树宽有界度图上的稳定座位安排开放问题。

## 研究背景与动机

1. **领域现状**: 加权一阶模型计数（WFOMC）要求计算给定一阶逻辑语句在指定域上所有模型的加权和，是统计关系学习的基础问题（Markov逻辑网络、概率逻辑程序、概率数据库等都可归约到 WFOMC）。已知双变量片段 $\text{FO}^2$（带计数量词 $\text{C}^2$）在无证据时是 domain-liftable 的（多项式时间）。
2. **现有痛点**: 对 WFOMC 施加二元证据（固定某些二元谓词的真值）后，即使对原本 domain-liftable 的 $\text{FO}^2$ 也变成 #P-hard。直觉上，二元证据打破了域元素的对称性——无证据时元素可按 1-type 分类，同类元素不可区分；二元证据使每对元素行为不同。
3. **核心矛盾**: 二元证据是实际应用必需的（图结构、社交网络等），但其引入使得高效提升推理无法进行。
4. **本文目标**: 在二元证据的 Gaifman 图具有有界树宽的限制下，为 $\text{FO}^2$ 和 $\text{C}^2$ 提供多项式时间的 WFOMC 算法。
5. **切入角度**: 在 Gaifman 图的 nice tree decomposition 上做动态规划，利用已有的 1-type 配置技术处理对称元素，仅对树分解中的 bag 内元素特殊处理。
6. **核心 idea**: 利用 Gaifman 图的有界树宽结构，在树分解上动态规划计算 WFOMC，将二元证据的影响局部化。

## 方法详解

### 整体框架

给定 $\text{UFO}^2$ 语句 $\Psi$、域 $\Delta$、权重函数 $(w, \bar{w})$、一元证据 $\mathcal{U}$ 和有界树宽二元证据 $\mathcal{E}$，先构建 $\mathcal{E}$ 的 Gaifman 图的 nice tree decomposition $T(V_T, E_T)$，然后在树上自底向上递归计算部分模型的加权和 $f(u, \tau, \boldsymbol{\zeta})$。

### 关键设计

1. **递归框架与部分模型**:
    - **功能**: 定义树分解节点上的递归计算目标
    - **核心思路**: 对树中每个节点 $u$，定义 $f(u, \tau, \boldsymbol{\zeta})$ 为满足条件的部分模型权重和，其中 $\tau$ 是 bag $B_u$ 中元素的 1-type 赋值，$\boldsymbol{\zeta}$ 是 bag 外元素 $S_u$ 的 1-type 配置。最终 $\text{WFOMC} = \sum_{\boldsymbol{\zeta}} f(root, \top, \boldsymbol{\zeta})$
    - **设计动机**: tree decomposition 保证 bag 外元素与 bag 的交互仅通过 bag 内元素中介——bag 外元素之间可按 1-type 配置对称处理（保留多项式时间），bag 内有限数量元素逐个处理

2. **四种节点类型的递归**:
    - **功能**: 对叶节点、引入节点、遗忘节点、合并节点分别定义递归
    - **核心思路**:
     - 叶节点: $f(u, \top, \mathbf{0}) = 1$
     - 引入节点: 新元素 $a$ 不与 $S_u$ 相连（Gaifman 图性质），2-table 权重可用 $r_{\tau_a, C_i}$ 批量计算
     - 遗忘节点: 枚举离开元素的 1-type 和与 bag 中每个元素的 2-table
     - 合并节点: $S_{v_1}$ 和 $S_{v_2}$ 不相连，其 2-table 权重可按 1-type 配置批量计算
    - **设计动机**: 利用 nice tree decomposition 的结构保证每步只处理一个元素的变化，计算可控

3. **扩展到 $\text{FO}^2$、$\text{C}^2$ 和非对称权重**:
    - **功能**: 将核心算法从 $\text{UFO}^2$ 推广到更丰富的逻辑片段
    - **核心思路**: 利用已有的模块化归约（modular transformation），将 $\text{FO}^2$ 和带计数量词/基数约束的 $\text{C}^2$ 归约到 $\text{UFO}^2$。对非对称权重，将涉及的元素对加入 Gaifman 图
    - **设计动机**: 模块化归约保持域大小不变，因此多项式时间复杂度得以保留

### 损失函数 / 训练策略

本文是理论/算法工作，无训练过程。时间复杂度为 $O(kn \cdot p^{k+3} \cdot q^{k+1} \cdot k \cdot n^{2p})$，其中 $n$ 为域大小，$k$ 为树宽，$p$、$q$ 为 1-type 和 2-table 数量（对固定语句为常数），因此关于 $n$ 是多项式的。

## 实验关键数据

### 主实验

| 问题 | 域大小 | TD-WFOMC | d4 | Forclift | Crane2 |
|------|--------|----------|-----|----------|--------|
| Friends & Smokers | 60 | ~0.01s | ~100s | ~100s | ~10s |
| Friends & Smokers | 120 | ~0.1s | timeout | timeout | timeout |
| WS (ring + cliques) | 60 | ~1s | timeout | — | — |

### 消融实验

| 配置 | 关键观察 | 说明 |
|------|---------|------|
| 固定域=60, 变clique大小 | 运行时间随 clique 大小增长 | clique 大小影响树宽 |
| WS简化模型 vs 完整模型 | 简化模型更快 | 简化模型无基数约束 |
| Ring vs Clique 起始图 | 两者都高效 | 只要树宽有界即可 |

### 关键发现

- TD-WFOMC 在域大小扩展时保持高效，其他求解器在域较大时超时
- 算法对 clique 大小（影响树宽）敏感，但对域大小的扩展性极佳
- 解决了开放问题：有界树宽有界度图上固定类数的稳定座位安排问题的计数是多项式时间可解的（包括 $2 \times n$ 网格桌）

## 亮点与洞察

- 突破了二元证据导致 #P-hard 的壁垒——有界树宽是一个自然且实用的限制条件
- 与 Courcelle 定理不可比较：后者要求所有非一元谓词全部解释，本文允许二元谓词保持未解释；但本文限于双变量片段
- 稳定座位安排的应用展示了 WFOMC 框架在组合计数中的通用性
- 算法框架清晰，理论贡献扎实

## 局限与展望

- 仅适用于 Gaifman 图有界树宽的二元证据，对树宽较大的稠密图不适用
- 对固定语句而言 $p$ 和 $q$ 是常数，但实际中 1-type 和 2-table 数量可能很大
- 实验仅比较了两个问题，未涉及更多实际应用场景
- 关于 $n$ 的多项式次数 $2p$ 可能较高，实际可扩展性有待更多验证
- 对超过两个变量的片段（$\text{FO}^3$）已知不可提升，是理论上的硬限制

## 相关工作与启发

- **vs BMF 方法 (WFOMC-binary-evidence-bmf)**: BMF 将二元证据转换为一元证据，但要求布尔矩阵秩低，实际中往往不满足；本文的树宽限制更自然
- **vs Courcelle 定理**: Courcelle 处理单调二阶逻辑但要求完全解释图，本文处理 $\text{FO}^2$ 但允许未解释谓词
- **vs d4 / Forclift**: 命题级求解器 d4 在域扩大时指数爆炸；Forclift 支持提升推理但无法高效处理二元证据
- **vs Crane2**: 性能较好但不支持完整 $\text{FO}^2$ 语句

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 填补了有界树宽二元证据下 WFOMC 的理论空白，解决了开放问题
- 实验充分度: ⭐⭐⭐ 实验场景较少，主要验证扩展性，缺少实际应用 benchmark
- 写作质量: ⭐⭐⭐⭐ 数学严谨，但对非该领域读者门槛较高
- 价值: ⭐⭐⭐⭐ 理论贡献显著，对统计关系学习和组合计数有重要推动作用

<!-- RELATED:START -->

## 相关论文

- [Variance Computation for Weighted Model Counting with Knowledge Compilation Approach](variance_computation_for_weighted_model_counting_with_knowledge_compilation_appr.md)
- [Model Counting for Dependency Quantified Boolean Formulas](model_counting_for_dependency_quantified_boolean_formulas.md)
- [Higher-Order Responsibility](higher-order_responsibility.md)
- [DW-DGAT: Dynamically Weighted Dual Graph Attention Network for Neurodegenerative Disease Diagnosis](dw-dgat_dynamically_weighted_dual_graph_attention_network_for_neurodegenerative_.md)
- [Enhancing Transformers for Generalizable First-Order Logical Entailment](../../ACL2025/others/enhancing_fol_entailment.md)

<!-- RELATED:END -->
