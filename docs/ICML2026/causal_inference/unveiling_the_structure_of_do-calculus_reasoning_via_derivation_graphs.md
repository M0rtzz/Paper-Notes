---
title: >-
  [论文解读] Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs
description: >-
  [ICML 2026][因果推理][因果推断] 通过引入推导图（derivation graphs）显式表示 do-演算规则的所有等价变换——揭示因果表达式空间的结构，并证明最多 4 步规则应用可达任意等价表达式。
tags:
  - "ICML 2026"
  - "因果推理"
  - "因果推断"
  - "做演算"
  - "等价性"
  - "推导图"
  - "可识别性"
---

# Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs

**会议**: ICML 2026  
**arXiv**: [2606.03719](https://arxiv.org/abs/2606.03719)  
**代码**: https://gricad-gitlab.univ-grenoble-alpes.fr/yvernesc/do-calculus-derivation-graphs  
**领域**: 因果推断  
**关键词**: 因果推断, 做演算, 等价性, 推导图, 可识别性

## 一句话总结
通过引入推导图（derivation graphs）显式表示 do-演算规则的所有等价变换——揭示因果表达式空间的结构，并证明最多 4 步规则应用可达任意等价表达式。

## 研究背景与动机

**领域现状**：Pearl 的 do-演算通过三条图形重写规则（R1、R2、R3）实现观测和干预概率的递归变换，广泛应用于因果推断中的可识别性问题。

**现有痛点**：尽管 do-演算理论上完备，但从某个起始表达式到另一个等价表达式的路径空间极其复杂——同一因果效应可能有指数级等价表达式，不同表达式的统计性质差异显著，但如何系统探索和选择最优表达式缺乏理论指导。

**核心矛盾**：do-演算虽完备，但其规则间的组合顺序关系、等价表达式的完整刻画，以及多个有效估计量的系统导出都未被充分表征。

**本文目标**：（1）显式刻画 do-演算的全部等价变换；（2）揭示规则间的对易性；（3）为实验设计和统计效率优化提供原则性方法。

**切入角度**：将 do-演算的逐步应用过程视为图论中的节点转移，构建推导图使所有等价表达式及其关系一目了然。

**核心 idea**：用有向图结构化组织 do-演算规则应用序列，通过图的拓扑性质揭示规则的对易性和等价表达式的指数级增长规律。

## 方法详解

### 整体框架
do-演算的三条重写规则（R1/R2/R3）原本是分散的局部变换，从一个因果表达式手工推导到等价表达式既无全局视野也缺乏停机保证。本文把这个推导过程整体搬进一张图里：每个因果表达式当作顶点，每次合法的规则应用当作一条边，于是"哪些表达式互相等价、最少几步可达、有几个可选估计量"全都变成图上的可枚举问题。

### 关键设计

**1. 推导图表示：把规则应用变成图上的边**

do-演算的痛点在于规则是局部的、表达式空间是隐式的，无法一眼看出全局等价结构。本文构造无向推导图 $D[G] = (V_D, E_D)$ 来显式化它：顶点是所有形如 $P(y \mid \text{do}(x), w)$ 的表达式，其中 $(Y, X, W)$ 是因果图顶点集 $V_G$ 的两两不相交子集且 $Y \neq \emptyset$；只要两个表达式之间存在一次合法的 do-演算规则应用，就在它们之间连一条边。这样一来，因果图中互相等价的表达式恰好落在推导图的同一个连通分量里，原本要逐个手工验证的等价关系被转成"是否同属一个连通分量"的图查询，所有等价变换都能被显式枚举。

**2. 规则对易性与 4 步可达界：约束路径长度**

如果规则可以任意长地组合，等价性判定和路径搜索的代价无法控制。本文通过分析三条规则的对易关系给出了硬上界：R1（观测的插入/删除）与其他规则可交换，而 R2（行动与观测的互换）和 R3（行动的插入/删除）之间存在特定的顺序依赖。基于这些对易性质，作者证明任意两个等价表达式之间总能通过至多 2 次 R2 和 2 次 R3（合计 $\le 4$ 步）相互到达，不需要更长的规则序列。配套地引入一个图形判据来直接判定等价性——对待比较的两个查询 $Q_1$、$Q_2$ 检查相应的 d-分离条件即可，无需真的去搜路径。这把等价性验证从一个开放式搜索问题压成了常数步数内可判定的问题。

**3. 多估计量导出：一个识别公式扩展成一族**

ID 算法只返回单个识别公式，但同一因果效应的不同等价写法在统计性质上差别很大。本文以 ID 算法的输出表达式为起点，在推导图上做 BFS/DFS 遍历，把它所在等价类中的其他表达式全部展开，每个等价表达式都对应一个有效但不同的估计量。这些估计量的方差和计算复杂度取决于各自涉及的条件化变量和求和维度：后门公式 $P(y \mid z)$ 只需对 1 个变量求和，而前门公式要对指数级数量的变量求和，两者的方差与计算开销因此相差悬殊。于是在数据有限、或某些干预在实际中不可行的场景（如生物、医学实验）里，这一族估计量提供了不同的成本-精度权衡，可以按需挑选。

## 实验关键数据

### 主实验

| 因果图规模 | 变量数 | 推导图顶点数 | 等价表达式数 | 最大规则应用数 |
|-----------|--------|-------------|------------|-------------|
| 空图（无边） | 3 | 27 | 27 | 2 |
| 链图 A→B→C | 3 | 9 | 9 | ≤4 |
| 分叉图 A←B→C | 3 | 6 | 6 | ≤4 |
| 对撞器 A→B←C | 3 | 18 | 18 | ≤4 |

### 等价表达式数量

| 表达式类型 | 起始表达式 | 等价数 | 所需最大规则应用 | 去除规则 R1 后数量 |
|-----------|-----------|-------------|---------------|--------------|
| 干预查询 | $P(a\|\text{do}(b,c))$ | 12 | 4 | 4 |
| 混合表达式 | $P(b\|\text{do}(a),c)$ | 8 | 3 | 3 |
| 观测查询 | $P(a\|b,c)$ | 3 | 2 | 2 |

## 亮点与洞察
- **对易性发现**：证明 $\text{R1} \circ \text{R2} = \text{R2} \circ \text{R1}$ 而 $\text{R2} \circ \text{R3}$ 在特定图形条件下可交换，为 do-演算提供新的代数结构。
- **最小 4 步界**：从直觉上看需要任意多步的规则组合实际上被严格限制在 4 步以内——简化等价性判定的计算复杂度。
- **实验设计新视角**：不同的等价表达式对应不同方差的估计量；为生物实验、医学干预等成本受限场景提供原则性选择方法。

## 局限与展望
- 计算复杂度——推导图顶点数随变量数指数增长，大规模因果图的完整推导图难以枚举。
- 估计量方差分析不完整——论文展示了不同估计量形式的差异但未给出方差的闭形式解析。
- 可识别性的扩展——现有方法限于已识别的因果效应，对部分识别的推导图刻画缺失。

## 相关工作与启发
- **vs ID 算法**：ID 算法返回单个识别公式；本文通过推导图揭示等价类内所有表达式，提供结构化替代方案。
- **vs 调整公式理论**：后门调整理论和前门公式都是本文等价类的特例，推导图统一了这些方法。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐  首次系统刻画 do-演算规则间的对易性和等价表达式的完整图论结构。
- 实验充分度: ⭐⭐⭐⭐  提供图论刻画和 Jupyter notebook 重现，实验例子清晰但样本量有限。
- 写作质量: ⭐⭐⭐⭐⭐  论文逻辑清晰，定义严格，符号一致。
- 价值: ⭐⭐⭐⭐⭐  对因果推断理论深化，并为实验设计提供实用指导。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](../../NeurIPS2025/causal_inference/practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)
- [\[NeurIPS 2025\] Do-PFN: In-Context Learning for Causal Effect Estimation](../../NeurIPS2025/causal_inference/do-pfn_in-context_learning_for_causal_effect_estimation.md)
- [\[NeurIPS 2025\] Characterization and Learning of Causal Graphs from Hard Interventions](../../NeurIPS2025/causal_inference/characterization_and_learning_of_causal_graphs_from_hard_interventions.md)
- [\[ICML 2026\] Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping](harnessing_reasoning_trajectories_for_hallucination_detection_via_answer-agreeme.md)
- [\[AAAI 2026\] Sparse Additive Model Pruning for Order-Based Causal Structure Learning](../../AAAI2026/causal_inference/sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)

</div>

<!-- RELATED:END -->
