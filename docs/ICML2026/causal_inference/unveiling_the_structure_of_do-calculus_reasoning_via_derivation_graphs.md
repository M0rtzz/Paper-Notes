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
推导图是无向图结构 $D[G] = (V_D, E_D)$——顶点代表所有形如 $P(y | \text{do}(x), w)$ 的因果表达式，边表示单次有效的 do-演算规则应用。流程：（1）定义推导图顶点和边；（2）分析规则间对易性；（3）证明最少规则应用数；（4）从等价表达式导出多个有效估计量。

### 关键设计

1. **推导图的图论表示**:

    - 功能：将因果表达式空间和 do-演算规则应用系统化为图论对象，实现所有等价变换的显式枚举和查询。
    - 核心思路：定义顶点为 $P(y | \text{do}(x), w)$ 形式的所有表达式，其中 $(Y, X, W)$ 为 $V_G$ 的不相交子集且 $Y \neq \emptyset$；定义边为两表达式间单步有效的 do-演算规则。图的连通分量对应因果图中的等价类。
    - 设计动机：三条 do-演算规则本质是表达式的局部变换，用图表示使全局等价结构显式化，避免逐个手工推导等价式。

2. **规则对易性与最小操作演算**:

    - 功能：证明任意两等价表达式间可通过最多 2 次 R2 和 2 次 R3 应用（共 ≤4 步）达到，无需任意长的规则序列。
    - 核心思路：通过图论分析揭示 R1（观测插入/删除）、R2（行动/观测交换）、R3（行动插入/删除）间的对易关系。关键发现：R1 与其他规则可交换，R2 和 R3 间存在特定顺序依赖。引入图形判据判定两表达式等价性：对 $Q_1$ 和 $Q_2$，检查相应 d-分离条件。
    - 设计动机：规则顺序复杂性阻碍实际应用，证明最小 4 步界限大幅简化等价性验证和路径搜索。

3. **多估计量导出与实验设计优化**:

    - 功能：从 ID 算法输出的单个识别公式出发，通过遍历等价表达式集找到多个不同的、统计性质各异的有效估计量。
    - 核心思路：使用推导图对图遍历（BFS/DFS）从 ID 算法的输出表达式扩展到其等价类中的其他表达式。每个等价表达式对应一个不同的估计量，其方差和计算复杂度因涉及的条件化变量、求和维度而异。后门公式 $P(y|z)$ 仅涉及 1 个变量求和，而前门公式涉及指数级变量，导致方差和计算复杂度大幅不同。
    - 设计动机：在数据有限或部分干预不可行的场景（生物、医学实验）下，多个有效估计量提供不同的成本-精度权衡。

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

- [\[AAAI 2026\] Skill Path: Unveiling Language Skills from Circuit Graphs](../../AAAI2026/causal_inference/skill_path_unveiling_language_skills_from_circuit_graphs.md)
- [\[ICML 2026\] Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping](harnessing_reasoning_trajectories_for_hallucination_detection_via_answer-agreeme.md)
- [\[NeurIPS 2025\] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](../../NeurIPS2025/causal_inference/practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)
- [\[AAAI 2026\] Sparse Additive Model Pruning for Order-Based Causal Structure Learning](../../AAAI2026/causal_inference/sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)
- [\[AAAI 2026\] CaDyT: Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis](../../AAAI2026/causal_inference/causal_structure_learning_for_dynamical_systems_with_theoretical_score_analysis.md)

</div>

<!-- RELATED:END -->
