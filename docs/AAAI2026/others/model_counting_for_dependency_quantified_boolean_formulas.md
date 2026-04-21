---
title: >-
  [论文解读] Model Counting for Dependency Quantified Boolean Formulas
description: >-
  [AAAI 2026][DQBF] 本文首次研究了依赖量化布尔公式（DQBF）的模型计数问题，证明了即使仅含两个存在量词变量的 #2-DQBF 就已是 #EXP-完全的，并基于 BDD 符号可达性技术实现了一个实用的 2-DQBF 模型计数器 sharp2DQR，在大依赖集上显著优于基于展开的基线方法。
tags:
  - AAAI 2026
  - DQBF
  - 模型计数
  - "#EXP-完全"
  - BDD
  - 符号可达性
---

# Model Counting for Dependency Quantified Boolean Formulas

**会议**: AAAI 2026  
**arXiv**: [2511.07337](https://arxiv.org/abs/2511.07337)  
**代码**: [GitHub](https://github.com/Sat-DQBF/sharp2DQR)  
**领域**: 理论计算机科学 / 形式化方法  
**关键词**: DQBF, 模型计数, #EXP-完全, BDD, 符号可达性

## 一句话总结

本文首次研究了依赖量化布尔公式（DQBF）的模型计数问题，证明了即使仅含两个存在量词变量的 #2-DQBF 就已是 #EXP-完全的，并基于 BDD 符号可达性技术实现了一个实用的 2-DQBF 模型计数器 sharp2DQR，在大依赖集上显著优于基于展开的基线方法。

## 研究背景与动机

依赖量化布尔公式（DQBF）是 QBF 的推广，通过显式指定每个存在量词变量依赖于哪些全称量词变量（而非依赖线性量词顺序），在硬件验证、程序综合等领域有广泛应用。DQBF 的可满足性问题是 NEXP-完全的，近年来人们已经开发了多个 DQBF 求解器，并在 QBF 评测中设立了专门的 DQBF 赛道。

然而，除了可满足性判定，许多综合与验证任务还需要知道**解的数量**——例如，Skolem 函数数量异常多可能意味着规约允许了非预期行为。虽然 QBF 和布尔综合的模型计数已有研究，但 DQBF 的模型计数问题（#DQBF）此前从未被探讨。这个问题格外困难：首先判定 DQBF 是否有解就已经是 NEXP-完全的；其次，DQBF 允许任意的、可能不可比的依赖集，使得现有的 #QBF 技术（依赖线性量词顺序）无法直接适用。

本文的核心思路是：利用 $k$-DQBF 与 $k$-SAT 之间的紧密对应关系（最近的研究发现 $k$-DQBF 是 $k$-SAT 的紧凑表示），将经典的 #SAT 复杂度结果"提升"到 DQBF 设定，并开发基于符号可达性的实用计数算法。

## 方法详解

### 整体框架

本文工作分为两大部分：（1）理论方面，证明 #2-DQBF 的 #EXP-完全性及其在一阶模型计数中的应用；（2）算法方面，基于 BDD 符号可达性提出 2-DQBF 模型计数算法。

### 关键设计

1. **Poly-monious 规约**:

    - 功能：定义一种新的多项式时间规约，用于建立 #EXP-困难性。
    - 核心思路：给定函数 $F$ 和 $G$，poly-monious 规约是一个多项式时间图灵机 $M$，在输入 $w$ 上输出 $t$ 个串 $v_1, \ldots, v_t$，使得 $F(w) = p(G(v_1), \ldots, G(v_t))$，其中 $p$ 是一个多项式。它比单调规约更强，但比一般的多项式时间图灵规约更弱。
    - 设计动机：#EXP 中的函数可能输出双指数大的数（需要指数多位表示），因此标准的多项式时间图灵规约不适合建立 #EXP-困难性。Poly-monious 规约在力度上恰好平衡了单调规约和图灵规约。

2. **#2-DQBF 的 #EXP-完全性证明**:

    - 功能：证明核心定理——仅含两个存在量词变量的 DQBF 模型计数就已 #EXP-完全。
    - 核心思路：利用投影（projection）技术和 Lemma 3，将任意 DQBF 实例 $\Psi$ 多项式时间规约到两个 2-DQBF 实例 $\Phi_1, \Phi_2$，使得 $\#\Psi = \#\Phi_1 - \#\Phi_2$。关键依赖于从 #SAT 到 #2-SAT 的规约（来自 Bannach 等人的工作）实际上是投影规约，可与 DQBF 展开的紧凑表示组合。显式规约的时间复杂度为 $O(k^2|\psi|)$。
    - 设计动机：类比 Valiant 的经典定理（#2-SAT 是 #P-完全的，尽管 2-SAT 在多项式时间内可解），在 DQBF 层面建立类似的复杂度跳跃。

3. **一阶模型计数的应用**:

    - 功能：证明一阶模型计数（FOMC）即使限制到域大小为 2 且基础逻辑是 PSPACE-可判定片段，仍然是 #EXP-完全的。
    - 核心思路：从 uniform 2-DQBF 到 FOMC 的规约利用谓词 $U$ 表示布尔值，谓词 $S$ 表示 Skolem 函数。证明 $\#\Phi$ 恰好是域 $\{1,2\}$ 上 FO 句子模型数的一半。
    - 设计动机：解释了为何可扩展的一阶模型计数器在十多年的研究中仍然难以实现。

4. **2-DQBF 模型计数算法**:

    - 功能：设计一个实用的 #2-DQBF 计数器 sharp2DQR。
    - 核心思路：(a) 将 2-DQBF 的矩阵解释为展开公式蕴含图的紧凑编码；(b) 通过 BDD 符号可达性构造蕴含图的传递闭包 $\varphi_{tr}$；(c) 将蕴含图分解为弱连通分量，独立计数每个分量；(d) 在每个分量内，枚举 $y_1$ 的 Skolem 函数，对每个函数将问题简化为 1-DQBF 计数（可高效求解）。
    - 设计动机：直接枚举 Skolem 函数不可行（可能双指数多），展开为命题公式会导致指数级膨胀。分量分解思想类似于命题模型计数中的组件分解技术，大幅降低了实际计算量。

### 损失函数 / 训练策略

本文为理论和算法工作，不涉及训练。算法关键策略包括：
- 使用 ABC 的 IC3/reach 命令进行 BDD 可达性计算
- 使用 CUDD 包进行 BDD 操作
- 非支撑变量预处理：通过 Lemma 10 高效计算非支撑变量数量，避免枚举
- 候选 Skolem 函数枚举中使用传递闭包信息进行剪枝（Eq. 3），排除不可能的赋值组合

## 实验关键数据

### 主实验

实验在三类基准上评估 sharp2DQR 与基线 Exp+ganak（展开 + 命题计数器）：

| 实例类型 | 依赖集大小 | sharp2DQR 表现 | Exp+ganak 表现 | 说明 |
|----------|-----------|---------------|---------------|------|
| PEC_opt (370实例) | 10-50 | **显著更好** | 瓶颈在展开 | 大依赖集时展开代价太高 |
| PEC_small (192实例) | 3-10 | 稍弱 | 较好 | 小实例 BDD 操作开销较大 |
| 2_colorability | 2-127 bits | **最高127 bits** | 最高12 bits | Exp+ganak 无法处理超过12 bits |

### 消融实验

| 比较对象 | 域大小上限 | 说明 |
|---------|-----------|------|
| sharp2DQR | $2^{127}$ | 可处理超过 $2^{2^{64}}$ 个 Skolem 函数 |
| Exp+ganak (cryptominisat) | 12 bits | 展开规模指数增长 |
| Exp+ganak (z3) | 12 bits | 性能与 cryptominisat 类似 |
| WFOMC (FOMC工具) | 域大小 4 | 远低于 sharp2DQR |

### 关键发现

- sharp2DQR 在大依赖集（10-50个变量）上大幅优于展开方法，因为展开规模是依赖集大小的指数
- 在小依赖集上，展开方法反而更好，因为 BDD 操作本身有一定开销
- 在独立集计数实例上，Exp+ganak 比 sharp2DQR 表现更好，说明两种方法有互补性
- 2-可着色性实例中，sharp2DQR 可处理 127 位图（$2^{127}$ 个节点），而 WFOMC 仅能处理域大小 4

## 亮点与洞察

- **理论贡献突出**：建立了 DQBF 模型计数的复杂度全景——#2-DQBF 就已是 #EXP-完全的，与 Valiant 的 #2-SAT 是 #P-完全形成完美类比
- **Poly-monious 规约**是一个有潜力的新工具，填补了单调规约和图灵规约之间的空白
- **FOMC 的新下界**解释了实际中可扩展 FO 模型计数器难以构建的根本原因
- **分量分解**思想在 DQBF 设定中的应用是自然而有效的

## 局限与展望

- 当前算法仅适用于 2-DQBF，推广到 3-DQBF 及一般 DQBF 是重要的下一步
- 在小依赖集实例上性能不如展开方法，BDD 操作的开销有优化空间
- 独立集计数等特定结构问题上，展开方法更优，说明算法的适用范围有待扩展
- FOMC 应用目前限于二元关系，能否推广到高阶关系值得探索

## 相关工作与启发

本文工作建立在 DQBF 可满足性求解（PEDANT、HQS 等求解器）、符号模型检查（IC3/PDR 算法）、命题模型计数（Ganak、#SAT）以及一阶模型计数（WFOMC）等多个方向之上。分量分解思想借鉴了命题模型计数中的成功经验。这项工作开辟了 DQBF 模型计数这一全新研究方向，为形式化验证和统计关系学习提供了理论基础。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Variance Computation for Weighted Model Counting with Knowledge Compilation Approach](variance_computation_for_weighted_model_counting_with_knowledge_compilation_appr.md)
- [Tractable Weighted First-Order Model Counting with Bounded Treewidth Binary Evidence](tractable_weighted_first-order_model_counting_with_bounded_treewidth_binary_evid.md)
- [From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm](from_decision_trees_to_boolean_logic_a_fast_and_unified_shap_algorithm.md)
- [How Hard is it to Explain Preferences Using Few Boolean Attributes?](how_hard_is_it_to_explain_preferences_using_few_boolean_attributes.md)
- [The Counting Power of Transformers](../../ICLR2026/others/the_counting_power_of_transformers.md)

<!-- RELATED:END -->
