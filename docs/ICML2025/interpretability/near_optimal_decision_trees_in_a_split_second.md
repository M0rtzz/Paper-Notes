---
title: >-
  [论文解读] Near-Optimal Decision Trees in a SPLIT Second
description: >-
  [ICML2025][决策树优化] 提出 SPLIT 算法族，通过在决策树根部附近做全局最优搜索、叶节点附近用贪心策略的混合方案，实现比全局最优方法快 100 倍以上且精度几乎无损的决策树构建。
tags:
  - ICML2025
  - 决策树优化
  - 分支定界
  - 动态规划
  - 贪心搜索
  - Rashomon集
  - 可解释性
  - 稀疏性
---

# Near-Optimal Decision Trees in a SPLIT Second

**会议**: ICML2025  
**arXiv**: [2502.15988](https://arxiv.org/abs/2502.15988)  
**代码**: 待确认  
**领域**: 决策树 / 可解释机器学习  
**关键词**: 决策树优化, 分支定界, 动态规划, 贪心搜索, Rashomon集, 可解释性, 稀疏性

## 一句话总结

提出 SPLIT 算法族，通过在决策树根部附近做全局最优搜索、叶节点附近用贪心策略的混合方案，实现比全局最优方法快 100 倍以上且精度几乎无损的决策树构建。

## 研究背景与动机

决策树是可解释机器学习的核心模型。传统方法两极分化：

- **贪心方法**（CART、C4.5）：速度快（线性复杂度），但无最优性保证，准确率和稀疏性均不理想
- **全局最优方法**（GOSDT、MurTree、DL8.5）：使用动态规划+分支定界求解，可证明最优，但面对高维特征或深层树时扩展性极差

实证研究表明，贪心与最优树之间平均存在 11-22 个百分点的准确率差距，个别数据集差距高达 10 个百分点。理想方案应兼具最优方法的精度和贪心方法的速度。

**核心观察**：作者通过分析 Rashomon 集（近最优树集合）发现，**越靠近叶节点的分裂越倾向于贪心分裂**。这意味着在叶节点附近进行全局优化收益极小，而根部优化至关重要。这一发现构成了 SPLIT 算法的理论基础。

## 方法详解

### 问题形式化

给定数据集 $D = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$，其中 $\mathbf{x}_i \in \{0,1\}^K$，目标是求解：

$$\mathcal{L}^*(D, d, \lambda) = \min_{T \in \mathcal{T}} \sum_{i=1}^{|D|} \frac{1}{N} \left( \ell(T(\mathbf{x}_i), y_i) + \lambda S(T) \right) \quad \text{s.t. depth}(T) \leq d$$

其中 $S(T)$ 为叶节点数（稀疏性惩罚），$\lambda$ 为正则化系数。

### SPLIT 算法（Algorithm 2）

SPLIT 接收一个**前瞻深度** $d_l$ 参数，将搜索分为两个层次：

1. **根部到前瞻深度**（$d_l$ 层）：使用分支定界 + 动态规划全局优化
2. **前瞻深度到叶节点**（$d - d_l$ 层）：使用贪心信息增益分裂

递归优化方程为：

$$\mathcal{L}(D, d', \lambda) = \begin{cases} \min\left\{\lambda + \frac{|D^-|}{N},\; \min_{f \in \mathcal{F}} \left\{ L(T_g(D(f), d', \lambda)) + L(T_g(D(\bar{f}), d', \lambda)) \right\} \right\} & d' = d - d_l \\ \min\left\{\lambda + \frac{|D^-|}{N},\; \min_{f \in \mathcal{F}} \left\{ \mathcal{L}(D(f), d'-1, \lambda) + \mathcal{L}(D(\bar{f}), d'-1, \lambda) \right\} \right\} & d' > d - d_l \end{cases}$$

关键创新在于 `get_bounds` 过程：在前瞻深度处，子问题的上下界直接设为贪心树的损失（而非宽松界），从而大幅剪枝。求解完前缀树后，还可用最优子树替换贪心子树进行**后处理优化**。

### LicketySPLIT（Algorithm 3）

多项式时间变体。递归地以前瞻深度 $d_l = 1$ 调用 SPLIT：先找当前最优分裂（假设后续全部贪心），然后对左右子树递归调用 LicketySPLIT，复杂度为 $O(nk^2 d^2)$，远优于全局最优的 NP-hard 复杂度。

### RESPLIT：Rashomon 集近似

利用 SPLIT 作为子程序，先生成一组前缀树，再在每个叶节点处调用 TreeFARMS 计算浅层近最优子树集合，从而快速近似完整 Rashomon 集。

## 实验关键数据

### 速度对比（SPLIT vs GOSDT）

| 数据集 | GOSDT 时间 | SPLIT 时间 | LicketySPLIT 时间 | 加速比 |
|--------|-----------|-----------|-------------------|--------|
| Bike | ~1000s | ~10s | ~1s | **100-1000×** |
| COMPAS | ~1s | ~1s | ~1s | 相当 |
| Netherlands | ~1s | ~1s | ~1s | 相当 |

- 在稀疏性惩罚 $\lambda$ 较小（树较复杂）时，SPLIT 比 GOSDT 快 2-3 个数量级
- 在 $\lambda$ 较大（超稀疏树）时，所有方法均在亚秒级完成

### Rashomon 集计算加速

| 数据集 | 完整计算(s) | RESPLIT(s) | 加速比 | 变量重要性相关系数 $\tau$ |
|--------|------------|-----------|--------|------------------------|
| COMPAS | 152 | 18 | 8.4× | 1.000 |
| Spambase | 2659 | 154 | 17.3× | 0.930 |
| Netherlands | 4255 | 216 | 19.7× | 0.932 |
| HELOC | 5564 | 337 | 16.5× | 0.979 |
| HIV | 9273 | 388 | 23.9× | 0.959 |
| Bike | 14330 | 194 | 73.9× | 0.999 |

RESPLIT 实现 10-74 倍加速，且变量重要性排序与完整 Rashomon 集高度一致（Pearson 相关 ≥0.93）。

### 前沿特性

在测试损失-稀疏性-训练时间三维空间中，SPLIT 和 LicketySPLIT 始终位于 Pareto 前沿的理想区域（低损失、低稀疏度、短时间），优于 CART、MAP-Tree、Thompson Sampling 等方法。

## 亮点与洞察

1. **简洁而深刻的观察**：近最优树在叶节点附近的分裂几乎总是贪心的，这一实证发现直接驱动了算法设计
2. **理论保证完备**：
    - SPLIT 复杂度 $O(n(d-d_l)k^{d_l+1} + nk^{d-d_l})$，在最优前瞻深度 $d_l = (d-1)/2$ 时相比全局最优节省 $O(k^{(d-1)/2} \cdot (d/2)!)$ 倍
    - LicketySPLIT 达到多项式时间 $O(nk^2d^2)$
    - 证明 SPLIT 可任意优于纯贪心方法（Theorem 6.5）
3. **实用性强**：即插即用地修改现有 GOSDT 框架，仅需改变初始化界的方式
4. **Rashomon 集应用**：首次将大规模 Rashomon 集计算扩展到更深、更宽特征空间

## 局限与展望

1. **仅支持二值特征**：需要预先离散化连续特征（使用 threshold guessing），离散化本身引入次优性
2. **仅针对分类任务**：未扩展到回归决策树
3. **前瞻深度需手动设定**：虽然理论给出最优值 $d_l = (d-1)/2$，但实际中可能需要针对数据调优
4. **后处理步骤仍调用全局最优求解器**：在极端情况下后处理可能成为瓶颈
5. **多类别问题**：论文仅讨论二分类，虽然声称可平凡扩展，但未提供多类别实验

## 相关工作与启发

- **GOSDT**（gosdt_guesses）：当前最快全局最优决策树方法，SPLIT 直接在其基础上改造
- **TreeFARMS**：Rashomon 集计算框架，RESPLIT 作为其加速前端
- **MurTree / DL8.5**：其他分支定界方法，SPLIT 的思想可迁移
- **Top-k 方法**：类似混合思路但不优化稀疏性，且未利用"叶节点附近贪心更好"的洞察

## 评分

- 新颖性: ⭐⭐⭐⭐ — 核心观察简洁有力，混合策略设计优雅
- 实验充分度: ⭐⭐⭐⭐⭐ — 多数据集、多指标、多基线全面对比，含 Rashomon 集实验
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，理论与实验紧密结合
- 价值: ⭐⭐⭐⭐ — 在可解释 ML 社区有实际影响力，弥合贪心与最优方法的鸿沟

<!-- RELATED:START -->

## 相关论文

- [Leveraging Predictive Equivalence in Decision Trees](leveraging_predictive_equivalence_in_decision_trees.md)
- [Empowering Decision Trees via Shape Function Branching](../../NeurIPS2025/interpretability/empowering_decision_trees_via_shape_function_branching.md)
- [How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis](../../NeurIPS2025/interpretability/toward_explainable_offline_rl_analyzing_representations_in_intrinsically_motivat.md)
- [ValuePilot: A Two-Phase Framework for Value-Driven Decision-Making](../../NeurIPS2025/interpretability/valuepilot_a_two-phase_framework_for_value-driven_decision-making.md)
- [ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees](../../AAAI2026/interpretability/shapbpt_image_feature_attributions_using_data-aware_binary_partition_trees.md)

<!-- RELATED:END -->
