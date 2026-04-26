---
title: >-
  [论文解读] How Hard Is It to Rig a Tournament When Few Players Can Beat or Be Beaten by the Favorite?
description: >-
  [AAAI 2026][tournament fixing] 本文提出两个新的结构化参数——目标选手在锦标赛有向图中的入度 $k$ 和出度 $\ell$——用于分析锦标赛赛程操纵问题 (TFP)，证明 TFP 在以这两个参数为参数时均是 FPT 的，其中入度参数化的算法设计涉及复杂的结构分析和颜色编码技术。
tags:
  - AAAI 2026
  - tournament fixing
  - knockout tournament
  - parameterized complexity
  - FPT
  - color coding
---

# How Hard Is It to Rig a Tournament When Few Players Can Beat or Be Beaten by the Favorite?

**会议**: AAAI 2026  
**arXiv**: [2601.08530](https://arxiv.org/abs/2601.08530)  
**代码**: 无  
**领域**: 算法博弈论 / 参数化复杂度 / 竞赛操纵  
**关键词**: tournament fixing, knockout tournament, parameterized complexity, FPT, color coding

## 一句话总结

本文提出两个新的结构化参数——目标选手在锦标赛有向图中的入度 $k$ 和出度 $\ell$——用于分析锦标赛赛程操纵问题 (TFP)，证明 TFP 在以这两个参数为参数时均是 FPT 的，其中入度参数化的算法设计涉及复杂的结构分析和颜色编码技术。

## 研究背景与动机

淘汰赛（knockout tournament）是最广泛使用的竞赛形式之一：每轮选手配对比赛，失败者淘汰，获胜者晋级，直到产生唯一冠军。FIFA 世界杯和 NCAA 篮球锦标赛都采用这种赛制。除了体育，淘汰赛也被用于选举、组织决策等场景。

**核心问题**：给定一个锦标赛有向图 $D$（编码所有选手之间的胜负关系），以及一个目标选手 $v^*$，是否存在一种赛程安排（seeding）使得 $v^*$ 一定获胜？这就是锦标赛赛程操纵问题（TFP）。TFP 已被证明是 NP-hard 的。

**已有参数化结果**：当以反馈弧集数 (fas) 或反馈顶点集数 (fvs) 为参数时，TFP 是 FPT 的。这些是全局结构参数，衡量锦标赛有向图距离无环图有多远。

**本文的新视角**：如果 $v^*$ 不在任何环中，TFP 就可以在多项式时间内解决。这启发了两个新的局部参数——$k$（能击败 $v^*$ 的选手数，即入度）和 $\ell$（$v^*$ 能击败的选手数，即出度）。这两个参数以 $v^*$ 为中心，直觉清晰、易于计算，且在入度/出度小但 fas/fvs 大时仍可能高效求解。

## 方法详解

### 整体框架

本文分为两部分：出度参数化（相对简单）和入度参数化（核心技术贡献）。出度参数化通过一个简洁的观察直接得到 FPT 结果。入度参数化则需要三步走：结构分析 → 中间结构（获胜见证森林 WWF） → 基于颜色编码的 FPT 算法。

### 关键设计

1. **出度参数化（Theorem 1, 简单情形）**:

    - 功能：证明 TFP 在以 $v^*$ 的出度 $\ell$ 为参数时是 FPT 的
    - 核心思路：冠军需要赢 $\log n$ 场比赛，但 $v^*$ 只能赢 $\ell$ 个对手。如果 $\ell < \log n$ 则必为 No-instance；否则 $n \leq 2^\ell$，直接用 $2^n \cdot n^{O(1)}$ 时间的精确算法即可在 FPT 时间内解决
    - 运行时间：$2^{2^\ell} \cdot n^{O(1)}$
    - 下界（Theorem 2）：在 ETH 假设下，不存在 $2^{2^{\ell/c}} \cdot n^{O(1)}$ 时间算法（$c > 1$），说明上述简单算法几乎最优

2. **二项树形结构与 LBA（关键等价转换）**:

    - 功能：将 TFP 转化为寻找以 $v^*$ 为根的标记生成二项树形结构（LBA）
    - 核心等价（Proposition 1, Williams 2010）：存在使 $v^*$ 获胜的赛程，当且仅当锦标赛 $D$ 存在以 $v^*$ 为根的 LBA
    - 设计动机：LBA 精确对应淘汰赛中的胜者树结构——根是冠军，每个节点的子树中的根是该节点击败的对手。这个等价使得 TFP 从组合赛程问题转化为子图搜索问题

3. **Nice Seeding 的结构性质（入度参数化的基石）**:

    - 功能：证明如果 TFP 是 Yes-instance 且 $k < \log n$，则必然存在一个"nice"的获胜赛程
    - Nice 的定义（Definition 3）：每轮要么淘汰至少一个 $v^*$ 的入邻居（能击败 $v^*$ 的选手），要么此时已无入邻居存活
    - 关键推论（Corollary 1）：所有能击败 $v^*$ 的选手必须在前 $k$ 轮内被全部淘汰
    - 证明方法：从任意获胜赛程出发，通过"修复"非 nice 轮次的迭代过程构造 nice 赛程——将非 nice 轮中的败者重新组织成一个子淘汰赛，与主淘汰赛合并
    - 设计动机：这大大限制了获胜赛程的结构，使得后续的算法设计可以聚焦于包含 $v^*$ 入邻居的小规模子结构

4. **获胜见证森林 WWF（核心中间结构）**:

    - 功能：定义 WWF 并证明其存在等价于 TFP 是 Yes-instance（当 $k \cdot 2^k < n$）
    - WWF 定义（Definition 4）：由 $k$ 棵大小为 $2^k$ 的顶点不相交的 LBA 组成的森林，所有入邻居都在其中，每棵树的根在 $N_{\text{out}}(v^*) \cup \{v^*\}$ 中
    - 等价性（Lemma 4）：利用 Lemma 3 证明正向（从获胜赛程构造 WWF），利用合并 LBA 的过程证明反向（从 WWF 构造获胜赛程）
    - 设计动机：WWF 将全局的 LBA 存在性问题分解为寻找 $k$ 棵小树的局部问题，为颜色编码技术的应用创造条件

5. **颜色编码算法（Algorithm 1）**:

    - 功能：用随机化算法高效检测 WWF 的存在
    - 步骤一（着色）：确定性地给 $N_{\text{in}}(v^*)$ 的 $k$ 个顶点分配 $1, \ldots, k$ 的颜色；其余顶点从 $\{k+1, \ldots, k \cdot 2^k\}$ 中随机均匀采样颜色
    - 步骤二（检测彩色 WWF）：构造辅助图 $F'$（$k$ 棵 UBA 加一个虚拟根 $f$）和 $D'$（从 $D$ 删除指向 $v^*$ 的弧，加虚拟顶点 $d$），将问题归约为着色子图同构问题
    - 运行时间分析：WWF 被正确着色的概率至少为 $e^{-t}$（$t = k \cdot 2^k - k$），子图同构求解用时 $2^t \cdot n^{O(1)}$，迭代 $e^t$ 次使成功概率达 $1 - 1/e$，总时间 $(2e)^t \cdot n^{O(1)}$
    - 可去随机化：用典型的 splitter 构造替换随机着色

### 运行时间总结

| 参数化 | 运行时间 | 备注 |
|--------|---------|------|
| 出度 $\ell$ | $2^{2^\ell} \cdot n^{O(1)}$ | 几乎匹配 ETH 下界 |
| 入度 $k$ | $(2e)^{k \cdot 2^k - k} \cdot n^{O(1)}$ | 可去随机化为 $(2e)^{t+o(t)} \cdot n^{O(1)}$ |

## 实验关键数据

本文是纯理论的参数化算法工作，没有实验。

### 理论结果对比

| 参数化方式 | 已知结果 | 本文结果 | 关系 |
|-----------|---------|---------|------|
| fas 数 $p$ | $p^{O(p)} \cdot n^{O(1)}$ FPT | — | $k, \ell$ 可以在 $p$ 小时很大 |
| fvs 数 $q$ | $q^{O(q)} \cdot n^{O(1)}$ FPT | — | $q \leq p$，但 $k, \ell$ 与 $q$ 无单调关系 |
| 出度 $\ell$ | 新参数 | $2^{2^\ell} \cdot n^{O(1)}$ FPT + ETH 下界 | 互补视角 |
| 入度 $k$ | 新参数 | $(2e)^{k \cdot 2^k} \cdot n^{O(1)}$ FPT | 核心技术贡献 |
| 子集 fas/fvs | 未知 | 仍为开放问题 | $\leq k$ 和 $\leq \ell$ |

### 参数层级关系

| 不等关系 | 说明 |
|---------|------|
| subset fas $\leq$ fas | 子集参数不大于全局参数 |
| subset fvs $\leq$ fvs | 同上 |
| subset fas $\leq k$ | 子集 fas 不大于入度 |
| subset fvs $\leq \ell$ | 子集 fvs 不大于出度 |
| $k, \ell$ 与 fas/fvs | 无单调关系 |

### 关键发现

- 入度 $k$ 可以在 fas 和 fvs 都很大时仍然很小，因此入度参数化提供了真正新的高效可解区域
- Nice seeding 的存在性是整个入度参数化分析的基石——它将"全局淘汰赛结构"约束为"入邻居在前 $k$ 轮被淘汰"的局部结构
- WWF 的等价性将存在性问题从全尺寸 LBA（$n$ 个节点）压缩到 $k$ 棵小树（各 $2^k$ 个节点），使颜色编码成为可能

## 亮点与洞察

- 问题的动机极其自然：从"目标选手能被多少人击败"这一直觉出发，引出全新的参数化视角
- Nice seeding 的构造性证明（Lemma 2）非常精巧：通过将非 nice 轮中的败者单独组成子赛事再合并回来，渐进修复赛程结构
- WWF 是一个非常漂亮的中间结构——它既够小（仅 $k \cdot 2^k$ 个节点）使颜色编码可行，又保留了足够的信息使其存在性等价于原问题
- 出度参数化的简洁性（"能赢的人太少就不可能赢"）与入度参数化的技术深度形成鲜明对比，展示了同一问题的两面

## 局限与展望

- 入度参数化的运行时间中 $k \cdot 2^k$ 出现在指数上，实际可用性局限于 $k$ 非常小的情况
- 子集 fas 和子集 fvs 的参数化复杂度仍是开放问题（甚至不知道当这些值为 1 时 TFP 是否 NP-hard）
- 只考虑了确定性胜负关系（tournament digraph），没有处理概率性比赛结果
- 锦标赛结构限制为完全二叉树（选手数为 $2^c$），实际赛事中常有 bye（轮空）等机制
- 颜色编码的去随机化虽然理论上可行，但增加了 $\log n$ 因子，实践中可能影响效率

## 相关工作与启发

- Williams (2010) 的 LBA 等价定理是整个领域的基石，本文在此基础上进一步分析了 LBA 的精细结构
- Ramanujan & Szeider (2017) 和 Gupta et al. (2018) 的 fas 参数化 FPT 算法是直接前驱，本文的新参数提供了与之不可比较的新视角
- Zehavi & Zehavi (2023) 的 fvs 参数化结果是目前最强的全局参数结果（因为 $q \leq p$）
- 颜色编码（Alon et al., 1995）在子图同构问题中的成功应用，在此得到了新颖而非平凡的扩展

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 提出全新的局部参数视角，入度参数化的结构分析和 WWF 概念极具原创性
- 实验充分度: ⭐⭐⭐ — 纯理论工作，定理和证明完整，但缺少实验验证算法的实际性能
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，从简单的出度到复杂的入度层层推进，图示辅助理解
- 价值: ⭐⭐⭐⭐ — 显著拓展了 TFP 的参数化理解，但实际算法可用性受限于参数大小

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] How Hard is it to Explain Preferences Using Few Boolean Attributes?](how_hard_is_it_to_explain_preferences_using_few_boolean_attributes.md)
- [\[AAAI 2026\] Align When They Want, Complement When They Need! Human-Centered Ensembles for Adaptive Human-AI Collaboration](align_when_they_want_complement_when_they_need_human-centere.md)
- [\[AAAI 2026\] How Wide and How Deep? Mitigating Over-Squashing of GNNs via Channel Capacity Constrained Estimation](how_wide_and_how_deep_mitigating_over-squashing_of_gnns_via_channel_capacity_con.md)
- [\[AAAI 2026\] How to Marginalize in Causal Structure Learning?](how_to_marginalize_in_causal_structure_learning.md)
- [\[CVPR 2026\] What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely F₁](../../CVPR2026/others/what_is_the_optimal_ranking_score_between_precision_and_recall_we_can_always_fin.md)

<!-- RELATED:END -->
