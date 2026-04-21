---
title: >-
  [论文解读] Non-Clashing Teaching in Graphs: Algorithms, Complexity, and Bounds
description: >-
  [ICLR 2026][非冲突教学] 研究图中闭邻域概念类的非冲突教学问题，提供精确匹配的算法上下界（N-NCTD⁺ 的 $2^{\mathcal{O}(|E|)}$ 紧界）、对 treedepth/vertex cover 参数化的 FPT 算法（含首个负面标签 FPT 结果），以及平面图和单位正方形图的组合上界，全面推进了非冲突教学的计算与组合理解。
tags:
  - ICLR 2026
  - 非冲突教学
  - 图概念类
  - FPT算法
  - 教学维度
  - 组合复杂度
  - 参数化复杂度
---

# Non-Clashing Teaching in Graphs: Algorithms, Complexity, and Bounds

**会议**: ICLR 2026  
**arXiv**: [2602.00657](https://arxiv.org/abs/2602.00657)

**领域**: 学习理论/机器教学  
**关键词**: 非冲突教学, 图概念类, FPT算法, 教学维度, 组合复杂度, 参数化复杂度

## 一句话总结

研究图中闭邻域概念类的非冲突教学问题，提供精确匹配的算法上下界（N-NCTD⁺ 的 $2^{\mathcal{O}(|E|)}$ 紧界）、对 treedepth/vertex cover 参数化的 FPT 算法（含首个负面标签 FPT 结果），以及平面图和单位正方形图的组合上界，全面推进了非冲突教学的计算与组合理解。

## 研究背景与动机

**领域现状**：非冲突教学（Non-Clashing Teaching）是目前已知最高效的满足 Goldman-Mathias 反共谋基准的批量机器教学模型。给定概念类 $\mathcal{C}$，教师为每个概念 $C \in \mathcal{C}$ 分配教学集 $T(C)$，要求：对任意两个不同概念 $C, C'$，$T(C) \cup T(C')$ 中至少有一个样本仅与其中一个概念一致（即"区分"它们）。非冲突教学维度 $\text{NCTD}(\mathcal{C})$ 为满足此条件的最小教学集大小。

**现有痛点**：
   - 此前对"图中球"概念类的 FPT 算法仅对 vertex integrity 参数化成立，而 treedepth 等更通用参数下的可解性未知
   - 允许负面标签（负例）参与教学时，**无任何 FPT 结果存在**——这是一个显著的理论空白
   - 先前 NCTD 的算法下界仅为 $2^{o(\sqrt{|V|})}$（来自 [KSZ19] 对开邻域的归约），与已知上界 $2^{\mathcal{O}(|V| \cdot k \cdot \log|V|)}$ 之间差距巨大

**切入角度**：选择闭邻域（closed neighborhoods，即 radius-1 球）作为概念类——任何有限二值概念类 $\mathcal{C} \subseteq 2^V$ 都可等价地表示为某个图 $G$ 中的一组闭邻域 $\{N[x_C] \mid C \in \mathcal{C}\}$（通过构造 $V(G) = V \cup \{x_C\}$，概念顶点构成团，$x_C$ 与 $v \in C$ 相邻），因此结果具有**最大通用性**。

**关键问题定义**：
   - **N-NCTD**: 给定图 $G$、闭邻域集合 $\mathcal{B}$、整数 $k$，判断 $\text{NCTD}(\mathcal{B}) \leq k$？
   - **N-NCTD⁺**: 同上但限制教学集只用正面标签，判断 $\text{NCTD}^+(\mathcal{B}) \leq k$？

**理论意义**：VC 维与 NCTD 的关系是核心开放问题（是否存在 $\text{NCTD}(\mathcal{C}) > \text{VCD}(\mathcal{C})$？），闭邻域的研究可能为回答此问题提供关键线索。

**广泛应用**：机器教学在样本压缩、逆强化学习、训练数据安全、人机交互等领域均有应用，理论基础的推进对这些方向都有潜在影响。

## 方法详解

### 核心贡献一：改进的算法上下界

**N-NCTD 下界（Theorem 2）**：
从 3-SAT 归约。给定 $n$ 变量 $m$ 子句的 3-SAT 实例 $\varphi$，构造图 $G$：

- 每个变量 $x_i$ 对应变量顶点 $v_i$，两个文字顶点 $t_i, f_i$
- 每个子句 $C_j$ 对应子句顶点 $c_j$
- 引入虚拟变量顶点 $v_0$、辅助集合 $\mathcal{V}_i = \{v_i^0, \ldots, v_i^4\}$ 和特殊顶点 $v_i^\star$
- 子句/变量顶点构成团

关键引理（Lemma 1）：4 个 pairwise false twins 的闭邻域在 NCTM size-1 下，必有一个教学集恰为 $\{u_i\}$（鸽巢原理：4 个顶点产生 6 对需区分，每个非自身顶点最多区分 1 对）。

结论：除非 ETH 失败，N-NCTD 不可能在 $2^{o(f(k) \cdot |V(G)|)}$ 时间内求解（$|V(G)| = \mathcal{O}(n+m)$），**大幅改进**此前 $2^{o(\sqrt{|V|})}$ 的下界。

**N-NCTD⁺ 精确匹配界**：
- **下界（Theorem 3）**：类似归约证明 $2^{o(f(k) \cdot (|V|+|E|))}$ 不可能
- **上界（Theorem 4）**：穷举所有正面教学映射——每个 $v$ 的正面教学集 $T(N[v]) \subseteq N[v]$，共 $2^{d(v)+1}$ 种选择，总计 $2^{\sum(d(v)+1)} = 2^{\mathcal{O}(|E|)}$ 种映射
- **精确匹配**：上下界在 $2^{\Theta(|E|)}$ 处吻合！

### 核心贡献二：FPT 算法

**N-NCTD⁺ 对 treedepth 参数化 FPT（Theorem 5）**：

核心思路：自底向上剪枝树深分解 $\mathcal{T}$。

- **Reduction Rule 1**: 设 $X \subseteq V(G)$，$A = \{A_1, \ldots, A_\ell\}$ 为 $G-X$ 的连通分量子集，$\max|A_i| = t$。若 $\ell > (|X|+t) \cdot 2^{(|X|+t)^2} \cdot 2^{2t+|X|+1}$，则删除某个特定分量。
- **安全性（Lemma 6）**：鸽巢原理保证存在 3 个自同构分量 $A_P, A_Q, A_R$，其邻接结构/教学集（含正面约束）完全"相同"→可将 $A_P$ 中的教学集元素替换为 $A_Q/A_R$ 中的对应副本→$A_P$ 可安全删除。
- **算法流程**：从叶节点层层往上剪枝，每层剪枝后节点数被 $g_j(\text{td}(G))$ 界定→最终图大小为 $f(\text{td}(G))$→暴力求解。

**N-NCTD 对 vertex cover 参数化 FPT（Theorem 7）**：

这是**首个**允许负面标签的 FPT 结果。技术路线：核化算法（kernelization）。

- **Lemma 8**: $\text{NCTD}(\mathcal{B}) \leq 2^{|X|+1} + |X|$（$X$ 为 vertex cover），给出解大小上界
- **Reduction Rule 2**: 若独立集中某等价类有 $q + 2k + 1$ 个 pairwise false twins 的闭邻域在 $\mathcal{B}$ 中，则删除一个（Lemma 9 via 鸽巢原理）
- **Reduction Rule 3**: 若独立集中两个 false twins $u, v$ 的闭邻域都不在 $\mathcal{B}$ 中，则删除 $v$
- 穷举应用后图的顶点数被 $2^{|X|}(2^{2^{|X|}+|X|} + 2^{|X|+2} + 2|X|) + |X|$ 界定→核心大小仅依赖 $|X|$

### 核心贡献三：组合上界

| 图类 | NCTD⁺ | NCTD |
|------|-------|------|
| 平面图 | $\leq 7$（Thm 12） | $\leq 5$（Thm 13） |
| 单位正方形图 | $\leq 4$（Thm 14） | — |

**平面图证明思路（Theorem 12, NCTD⁺ ≤ 7）**：
- 度 $\leq 6$ 的顶点：$T(N[v]) := N[v]$
- 度 $\geq 7$ 的顶点：选 3 个邻居放入 $T(N[v])$→由 $K_{3,3}$ 禁止子图性质，至多 1 个其他顶点 $u$ 也邻接这 3 个→最多再补 4 个顶点→$|T| \leq 7$

**单位正方形图证明（Theorem 14, NCTD⁺ ≤ 4）**：
几何论证——每个顶点 $v$ 的闭邻域被最小包围矩形 $R(v)$ 包含→$T(N[v])$ 取最左/最右/最上/最下的方块→若 $R(u) \neq R(v)$，必有某方向极值方块区分两者。

### 关键设计

- **闭邻域等价表示**：$V(G) = V \cup \{x_C | C \in \mathcal{C}\}$，概念顶点构成团，$x_C \sim v \iff v \in C$，则 $\mathcal{B} = \{N[x_C]\}$ 等价表示 $\mathcal{C}$
- **false twins 分析**：贯穿所有归约规则的核心工具——相同开邻域的顶点对教学映射有强约束
- **鸽巢原理的巧妙应用**：从 Lemma 1（4 twins → 必有自选）到 Lemma 9（大量 twins → 两个教学集"相同"）

## 主要理论结果

### 算法复杂度全景

| 问题 | 参数 | 本文结果 | 先前最佳 |
|------|------|---------|---------|
| N-NCTD | — | 下界 $2^{o(f(k)\cdot|V|)}$ 不可能 | $2^{o(\sqrt{|V|})}$ [KSZ19] |
| N-NCTD⁺ | — | **精确** $2^{\Theta(|E|)}$ | 无下界 |
| N-NCTD⁺ | treedepth | **FPT** | vertex integrity [GKM+25] |
| N-NCTD | vertex cover | **FPT** | 无（首个负标签 FPT） |

### 组合上界

| 图类 | NCTD⁺ 上界 | NCTD 上界 | VCD |
|------|-----------|----------|-----|
| 平面图 | ≤ 7 | ≤ 5 | ≤ 4 |
| 单位正方形图 | ≤ 4 | — | ≤ 4 |
| 树/环/仙人掌图 | 已知最优 [CCM+24] | — | — |

## 关键发现

- N-NCTD⁺ 的指数时间复杂度在 $2^{\Theta(|E|)}$ 处精确确定——这是非冲突教学领域首个精确匹配的算法界
- treedepth 严格推广了 vertex integrity 参数→FPT 结果覆盖了更广的图类
- 允许负面标签使问题"非局部化"（教学集可含不在闭邻域中的顶点），导致证明显著复杂化，但核化仍可行
- 平面图的 NCTD ≤ 5 可能高于 VCD ≤ 4，暗示 $\text{NCTD} > \text{VCD}$ 的反例可能存在于平面图中

## 亮点与洞察

- **精确匹配的指数算法界**：$2^{\Theta(|E|)}$ 表明问题在此参数下完全"关闭"，无进一步改进空间
- **任意有限概念类 = 闭邻域**：结果的通用性极强，不仅限于图论语境
- **首个负面标签 FPT**：突破了此前所有 FPT 仅限正面教学的瓶颈
- **学习理论 + 图论 + 参数化复杂度**的交叉融合——优美的跨领域理论贡献
- 鸽巢原理在归约规则安全性证明中的多层次运用，技术层面极为精巧

## 局限性

- 纯理论工作，无实验验证——所有结果为数学定理，不涉及实际教学场景测试
- FPT 算法的可计算函数 $f$ 可能增长极快（多重指数），实际可解规模有限
- treedepth 参数化的 FPT 仅限正面变体（N-NCTD⁺），负面变体对 treedepth 的复杂性未知
- 平面图 NCTD ≤ 5 的紧性未确定——是否存在平面图使 NCTD = 5？
- treewidth 参数化的复杂性（预计 W[1]-hard）尚未证明
- 核化后的核心大小为多重指数级→实际应用中可能不实用

## 相关工作对比

### vs. Chalopin et al. [COLT 2024]（CCM+24）
CCM+24 研究图中所有球的正面非冲突教学：(1) Strict Non-Clash 对 vertex cover 参数化 FPT，但时间为 $2^{2^{\mathcal{O}(\text{vc})}}$；(2) 给出树/环/仙人掌图等特殊图类的最优 NCTM。本文聚焦闭邻域（radius-1 球）获得了**更强结果**：对更通用参数 treedepth 的 FPT、首个负面标签 FPT 以及更广图类的组合上界。

### vs. Ganian et al. [ICLR 2025]（GKM+25）
GKM+25 证明 Non-Clash 对 vertex integrity FPT 且对 feedback vertex number + pathwidth + $k$ 联合参数化 W[1]-hard，给出了几乎完整的复杂度图景。本文在以下方面推进：(1) treedepth（比 vertex integrity 更通用）下仍为 FPT；(2) 精确匹配的 $2^{\Theta(|E|)}$ 算法界（GKM+25 的上界含额外 $\log$ 因子）；(3) 首次处理含负面标签的变体。

### vs. Kirkpatrick et al. [ALT 2019]（KSZ19）
KSZ19 引入非冲突教学模型并证明开邻域的 NCTD 判定为 NP 完全。本文的下界 $2^{o(|V|)}$ 大幅改进了 KSZ19 隐含的 $2^{o(\sqrt{|V|})}$ 下界。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 精确匹配算法界 + 首个负面标签 FPT + treedepth FPT 均为全新结果
- 理论深度: ⭐⭐⭐⭐⭐ 纯理论工作，证明技术精妙（多层鸽巢 + 自同构分量剪枝 + 核化）
- 写作质量: ⭐⭐⭐⭐⭐ 数学严谨，结构清晰，图示（Fig 1-8）辅助理解极佳
- 影响力: ⭐⭐⭐⭐ 对机器教学和参数化复杂度的基础理论有重要推进，但受众较窄

<!-- RELATED:START -->

## 相关论文

- [Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms](accessible_realistic_and_fair_evaluation_of_positive-unlabeled_learning_algorith.md)
- [Tight Lower Bounds and Improved Convergence in Performative Prediction](../../NeurIPS2025/llm_evaluation/tight_lower_bounds_and_improved_convergence_in_performative_prediction.md)
- [Cross-regularization: Adaptive Model Complexity through Validation Gradients](../../ICML2025/llm_evaluation/cross-regularization_adaptive_model_complexity_through_validation_gradients.md)
- [A Unified Framework for Provably Efficient Algorithms to Estimate Shapley Values](../../NeurIPS2025/llm_evaluation/a_unified_framework_for_provably_efficient_algorithms_to_estimate_shapley_values.md)
- [EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework](../../ACL2025/llm_evaluation/educationq_evaluating_llms_teaching_capabilities_through_multi-agent_dialogue_fr.md)

<!-- RELATED:END -->
