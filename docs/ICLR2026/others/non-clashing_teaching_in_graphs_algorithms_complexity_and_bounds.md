---
title: >-
  [论文解读] Non-Clashing Teaching in Graphs: Algorithms, Complexity, and Bounds
description: >-
  [ICLR 2026][非冲突教学] 研究图中闭邻域概念类的非冲突教学问题，提供精确匹配的算法上下界（N-NCTD⁺ 的 $2^{\mathcal{O}(|E|)}$ 紧界）、对 treedepth/vertex cover 参数化的 FPT 算法（含首个负面标签 FPT 结果）…
tags:
  - "ICLR 2026"
  - "非冲突教学"
  - "图概念类"
  - "FPT算法"
  - "教学维度"
  - "组合复杂度"
  - "参数化复杂度"
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

### 整体框架

本文围绕闭邻域概念类 $\mathcal{B} = \{N[v]\}$ 把非冲突教学拆成三条平行的理论线：先用 3-SAT 归约配合穷举给出 N-NCTD 与 N-NCTD⁺ 的精确指数时间界，再用树深剪枝和核化分别得到 treedepth、vertex cover 下的 FPT 算法，最后用图论结构论证平面图与单位正方形图的常数组合上界。三条线背后反复依赖同一套工具——闭邻域的等价表示、false twins 分析和分层的鸽巢原理。

### 关键设计

**1. 闭邻域等价表示：把任意概念类装进图。**

所有结果建立在一个统一的编码上：任何有限二值概念类 $\mathcal{C} \subseteq 2^V$ 都能写成某个图里的一组闭邻域。构造 $V(G) = V \cup \{x_C \mid C \in \mathcal{C}\}$，让概念顶点 $\{x_C\}$ 两两相连构成团，并令 $x_C \sim v \iff v \in C$，则 $\mathcal{B} = \{N[x_C]\}$ 就等价表示了 $\mathcal{C}$。正因为这一步无损，针对闭邻域（radius-1 球）证得的上下界对一切有限概念类都成立，结果的通用性远超图论本身。贯穿后续所有归约的核心抽象是 false twins——开邻域相同的顶点对，它们对教学映射施加强约束；而不同强度的鸽巢原理则负责把"twins 太多"翻译成"可以安全删一个顶点"。

**2. 改进的算法上下界：把 N-NCTD⁺ 钉死在 $2^{\Theta(|E|)}$。**

N-NCTD 的下界（Theorem 2）来自 3-SAT 归约：给定 $n$ 变量 $m$ 子句的实例 $\varphi$，构造的图 $G$ 为每个变量配变量顶点 $v_i$ 与两个文字顶点 $t_i, f_i$、为每个子句配顶点 $c_j$，再加虚拟变量顶点 $v_0$、辅助集合 $\mathcal{V}_i = \{v_i^0, \ldots, v_i^4\}$ 和特殊顶点 $v_i^\star$，子句与变量顶点各自成团，使 $|V(G)| = \mathcal{O}(n+m)$。关键引理（Lemma 1）正是鸽巢原理的第一次出场：4 个 pairwise false twins 的闭邻域产生 6 对需区分、而每个非自身顶点最多区分 1 对，于是在 size-1 的 NCTM 下必有某个教学集恰为 $\{u_i\}$，把 SAT 的真值赋值锁进教学映射。由此除非 ETH 失败，N-NCTD 无法在 $2^{o(f(k)\cdot|V(G)|)}$ 内求解，大幅改进了此前隐含的 $2^{o(\sqrt{|V|})}$ 下界。对只用正面标签的 N-NCTD⁺，类似归约给出 $2^{o(f(k)\cdot(|V|+|E|))}$ 的下界（Theorem 3），而上界（Theorem 4）只需穷举所有正面教学映射：每个 $v$ 的 $T(N[v]) \subseteq N[v]$ 有 $2^{d(v)+1}$ 种选择，总计 $2^{\sum(d(v)+1)} = 2^{\mathcal{O}(|E|)}$。上下界在 $2^{\Theta(|E|)}$ 处严丝合缝，这是非冲突教学领域首个精确匹配的指数算法界，意味着问题在该参数上已无改进空间。

**3. FPT 算法：treedepth 剪枝与 vertex cover 核化。**

N-NCTD⁺ 对 treedepth 参数化的 FPT（Theorem 5）走自底向上剪枝树深分解 $\mathcal{T}$ 的路线。Reduction Rule 1 取 $X \subseteq V(G)$、令 $A = \{A_1, \ldots, A_\ell\}$ 为 $G-X$ 的连通分量子集、$\max|A_i| = t$，一旦 $\ell > (|X|+t)\cdot 2^{(|X|+t)^2}\cdot 2^{2t+|X|+1}$ 就删掉一个特定分量。安全性（Lemma 6）再次靠鸽巢原理：分量数一旦超过这个阈值，必存在 3 个自同构分量 $A_P, A_Q, A_R$，其邻接结构与教学集（含正面约束）完全"相同"，于是 $A_P$ 里的教学元素可替换成 $A_Q/A_R$ 中的对应副本，$A_P$ 被安全删除。从叶节点层层往上剪，每层后节点数被 $g_j(\text{td}(G))$ 界定，最终图缩到 $f(\text{td}(G))$ 大小后暴力求解。N-NCTD 对 vertex cover 参数化的 FPT（Theorem 7）是首个允许负面标签的结果，技术路线改为核化：Lemma 8 先给出解大小上界 $\text{NCTD}(\mathcal{B}) \leq 2^{|X|+1} + |X|$（$X$ 为 vertex cover）；Reduction Rule 2 在独立集某等价类含 $q + 2k + 1$ 个 pairwise false twins 的闭邻域在 $\mathcal{B}$ 中时删一个（Lemma 9，仍是鸽巢）；Reduction Rule 3 在两个 false twins $u, v$ 的闭邻域都不在 $\mathcal{B}$ 中时删 $v$。穷举应用后顶点数被 $2^{|X|}(2^{2^{|X|}+|X|} + 2^{|X|+2} + 2|X|) + |X|$ 界定，核心大小只依赖 $|X|$。负面标签让教学集可以含闭邻域之外的顶点、问题变得"非局部",证明因此显著复杂，但核化依旧可行。

**4. 组合上界：用图论结构换常数教学集。**

最后是几个图类上的常数上界，归纳如下表。

| 图类 | NCTD⁺ | NCTD |
|------|-------|------|
| 平面图 | $\leq 7$（Thm 12） | $\leq 5$（Thm 13） |
| 单位正方形图 | $\leq 4$（Thm 14） | — |

平面图的 NCTD⁺ ≤ 7（Theorem 12）按度数分类处理：度 $\leq 6$ 的顶点直接取 $T(N[v]) := N[v]$；度 $\geq 7$ 的顶点先选 3 个邻居放进 $T(N[v])$，再借 $K_{3,3}$ 禁止子图性质保证至多 1 个其他顶点 $u$ 也邻接这 3 个，最多补 4 个顶点即得 $|T| \leq 7$。单位正方形图的 NCTD⁺ ≤ 4（Theorem 14）则是纯几何论证：每个顶点 $v$ 的闭邻域被最小包围矩形 $R(v)$ 包含，$T(N[v])$ 取最左、最右、最上、最下四个方块，只要 $R(u) \neq R(v)$ 就必有某方向的极值方块区分两者。值得注意的是平面图的 NCTD ≤ 5 可能高于 VCD ≤ 4，暗示 $\text{NCTD} > \text{VCD}$ 的反例或许就藏在平面图里。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Distributed Algorithms for Euclidean Clustering](distributed_algorithms_for_euclidean_clustering.md)
- [\[ICLR 2026\] Deterministic Bounds and Random Estimates of Metric Tensors on Neuromanifolds](deterministic_bounds_and_random_estimates_of_metric_tensors_on_neuromanifolds.md)
- [\[NeurIPS 2025\] The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets](../../NeurIPS2025/others/the_cost_of_robustness_tighter_bounds_on_parameter_complexity_for_robust_memoriz.md)
- [\[ICML 2025\] Improved Generalization Bounds for Transductive Learning by Transductive Local Complexity and Its Applications](../../ICML2025/others/improved_generalization_bounds_for_transductive_learning_by_transductive_local_c.md)
- [\[ICLR 2026\] The Hot Mess of AI: How Does Misalignment Scale With Model Intelligence and Task Complexity?](the_hot_mess_of_ai_how_does_misalignment_scale_with_model_intelligence_and_task_.md)

</div>

<!-- RELATED:END -->
