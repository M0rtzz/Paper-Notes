---
title: >-
  [论文解读] Data Complexity of Querying Description Logic Knowledge Bases under Cost-Based Semantics
description: >-
  [AAAI 2026][描述逻辑] 系统研究加权描述逻辑知识库在代价语义下的查询应答的数据复杂度，证明最优代价语义在$\Delta_2^p$内可解，并给出一个令人惊喜的正面结果：在DL-Lite$_{\text{bool}}^{\mathcal{H}}$本体和固定代价界限下，实例查询的确定回答和合取查询的可能回答可通过一阶重写实现最低数据复杂度（AC$^0$）。
tags:
  - AAAI 2026
  - 描述逻辑
  - 代价语义
  - 不一致性容忍
  - 本体查询应答
  - 计算复杂度
---

# Data Complexity of Querying Description Logic Knowledge Bases under Cost-Based Semantics

**会议**: AAAI 2026  
**arXiv**: [2511.07095](https://arxiv.org/abs/2511.07095)  
**代码**: 无  
**领域**: 其他（知识表示与推理/描述逻辑）  
**关键词**: 描述逻辑, 代价语义, 不一致性容忍, 本体查询应答, 计算复杂度

## 一句话总结

系统研究加权描述逻辑知识库在代价语义下的查询应答的数据复杂度，证明最优代价语义在$\Delta_2^p$内可解，并给出一个令人惊喜的正面结果：在DL-Lite$_{\text{bool}}^{\mathcal{H}}$本体和固定代价界限下，实例查询的确定回答和合取查询的可能回答可通过一阶重写实现最低数据复杂度（AC$^0$）。

## 研究背景与动机

### 本体中介查询应答（OMQA）

OMQA是知识表示领域的核心任务：给定本体（TBox）$\mathcal{T}$、数据集（ABox）$\mathcal{A}$和查询$q(\vec{x})$，找出所有"确定回答"（在知识库$(T,A)$的所有模型中都成立的元组）。描述逻辑（DL）是最常用的本体形式化语言。

### 不一致性问题

当知识库不一致时（$\mathcal{T}$和$\mathcal{A}$矛盾），经典语义下**任何**查询答案都平凡成立（前提矛盾推不出任何有意义结论），OMQA退化。

### 现有不一致性容忍方法的局限

**基于修复的语义**（如AR语义）考虑ABox的极大一致子集，但假设TBox完全可靠、不一致仅来自ABox错误。实际中TBox公理也可能有例外（"软"约束）。

### 代价语义：一种量化方法

Bienvenu, Bourgaux & Jean (2024) 提出**代价语义**：
- 为每个公理和断言赋予权重（惩罚值），权重$\infty$代表"必须满足"
- 每个解释的**代价**基于违反的公理/断言的权重之和
- 定义四种查询回答语义：有界代价确定/可能、最优代价确定/可能

### 本文动机

前期工作存在两个重要空白：

**最优代价确定语义**（最有用的语义）没有已知的非平凡上界
2. 考虑的DL没有逆角色和角色包含，不覆盖DL-Lite家族（OMQA中最常用的DL）

## 方法详解

### 整体框架

本文的核心贡献是一系列**复杂度结果**，覆盖从轻量级DL-Lite$_{\text{core}}$到表达力丰富的$\mathcal{ALCHIO}$的多种描述逻辑。方法论上分三部分：上界证明、下界证明、一阶可重写性的正面结果。

### 关键设计

#### 1. **代价语义的形式化**

**加权知识库（WKB）**：$\mathcal{K}_\omega = (\mathcal{T}, \mathcal{A})_\omega$，其中$\omega: \mathcal{T} \cup \mathcal{A} \mapsto \mathbb{N}_{>0} \cup \{\infty\}$。

**解释的代价**：
$$\omega(\mathcal{I}) = \sum_{\tau \in \mathcal{T}} \omega(\tau)|vio_\tau(\mathcal{I})| + \sum_{\alpha \in vio_\mathcal{A}(\mathcal{I})} \omega(\alpha)$$

TBox公理的代价按违反次数（元素个数）累计，ABox断言的代价按是否违反计算。

**四种查询回答语义**：
- $k$-代价有界确定：在所有代价 ≤ $k$的解释中都成立
- $k$-代价有界可能：在某个代价 ≤ $k$的解释中成立
- 最优代价确定：在所有最优代价解释中都成立
- 最优代价可能：在某个最优代价解释中成立

#### 2. **一般上界：商构造与小解释性质**（Section 4）

**核心技术**：给定满足（或不满足）查询$q$的解释$\mathcal{I}$，构造另一个解释$\mathcal{J}$，使得：
- 保持与$\mathcal{I}$相同的查询满足性
- 代价不超过$\mathcal{I}$
- 域的大小关于ABox大小$|\mathcal{A}|$多项式有界，且**独立于代价$k$**

**关键困难**：保持"查询不满足"（用于CQA$_c$）比保持"查询满足"困难得多。前者需要**参数化商构造**（Lemma 6），引入"保护集"$\mathcal{V} \subseteq \mathcal{T}$来控制哪些违反被精确保持。

**迭代构造过程**：
1. 初始化$\mathcal{V}_0 = \emptyset$
2. 若$\omega(\mathcal{J}_{\mathcal{V}_i}) > k$，找到比$\mathcal{I}$多出的违反对应的公理$\tau$
3. 将$\tau$加入$\mathcal{V}_{i+1}$，重新构造
4. 至多$|\mathcal{T}|$步后收敛

**定理**：CQA$_p$和CQA$_c$对$\mathcal{ALCHIO}$分别在NP和coNP内；最优代价语义在$\Delta_2^p$内（通过二分搜索找最优代价）。

#### 3. **下界：NP/coNP硬度在$k=1$时即成立**（Section 5）

**对$\mathcal{EL}_\bot$**（Theorem 3）：通过3-SAT归约。构造WKB使得唯一权重为1的公理$\mathsf{Bool} \sqsubseteq \mathsf{True}$编码了布尔变量的赋值，公式可满足当且仅当WKB是1-可满足的。关键改进：将前期工作的$k \geq 3$改进到$k \geq 1$。

**对DL-Lite$_{\text{core}}$**（Theorem 5）：通过3-着色问题归约，使用6个角色名$\{r_1,r_2,g_1,g_2,b_1,b_2\}$编码三种颜色，不相容性公理确保相邻节点不同色。

**最优代价语义**（Theorem 6）：$\Delta_2^p$硬度通过"词典序最大可满足赋值"问题归约，关键利用二进制编码的指数大权重实现词典序比较。

#### 4. **惊喜结果：DL-Lite家族的AC$^0$可重写性**（Section 6）

**定理8**（最具技术挑战性）：对任意固定$k \geq 1$，DL-Lite$_{\text{bool}}^{\mathcal{H}}$下的CQA$^k_p$和IQA$^k_c$在AC$^0$内。

**方法**：一阶查询重写——给定加权TBox $\mathcal{T}_{\omega_\mathcal{T}}$、查询$q$和代价$k$，构造一阶查询$q'$使得对任意加权ABox $\mathcal{A}_{\omega_\mathcal{A}}$：

$$(\mathcal{T}, \mathcal{A})_\omega \models^k_p q \text{ iff } \mathcal{I}_{\mathcal{A}_{\omega_\mathcal{A}}} \models q'$$

**核心技术——极小解释性质**：

**ABox类型**：扩展1-type概念，加入基数信息$\exists^{>k}\mathsf{r}$（角色$r$有超过$k$个不同的$r$-后继），捕捉代价约束下的结构信息。

**稀有类型**：ABox类型$t$的实例数 ≤ 2k时为"稀有"。非稀有类型必存在无违反的解释方式（否则超过$k$次违反导致代价超$k$）。

**核心（core）**：
$$\mathsf{core}(\mathcal{A}) = \mathsf{pc}(\mathcal{A}) \cup \{b \mid a \leadsto_i b, a \in \mathsf{pc}(\mathcal{A}), i \leq k+1\}$$
其中$\mathsf{pc}(\mathcal{A})$包含稀有类型的个体和查询中的个体。邻域探索深度为$k+1$（因为每层至多引入一个违反）。

**核心大小**为常数（关于ABox大小），因此所有相关违反和查询满足信息都集中在常数大小的子域中，可被有限数量的一阶子查询枚举。

**策略（Strategy）**：重写查询$q'$是所有p-策略（或c-策略）对应子查询的析取。每个策略$(M, \Gamma, \mathcal{V}_\nu)$指定一个核心大小的解释、其个体部分和允许的ABox违反。子查询检查剩余个体的ABox类型是否"安全"（可在不引入违反的情况下解释）。

### 关键公式总结

| 推理任务 | 固定$k$ | 变化$k$ | 最优代价 |
|---------|---------|---------|---------|
| $\mathcal{EL}_\bot$ / $\mathcal{ALCHIO}$ | NP/coNP-完全 | NP/coNP-完全 | $\Delta_2^p$-完全 |
| DL-Lite$_{\text{core}}$ / DL-Lite$_{\text{bool}}^{\mathcal{H}}$ | **AC$^0$** (CQA$_p^k$, IQA$_c^k$) / coNP (CQA$_c^k$) | NP/coNP-完全 | $\Delta_2^p$-完全 |

## 实验关键数据

### 主要理论结果（Table 1）

| 推理任务 | DL-Lite$_{\text{core}}$固定$k$ | DL-Lite$_{\text{core}}$变化$k$ | $\mathcal{EL}_\bot$固定$k$ |
|---------|------|------|------|
| BCS$^k$ / IQA$_p^k$ | AC$^0$ (上界) | NP-完全 | NP-完全 ($k \geq 1$) |
| IQA$_c^k$ | AC$^0$ (上界) | coNP-完全 | coNP-完全 ($k \geq 1$) |
| CQA$_c^k$ | coNP-完全 ($k \geq 1$) | coNP-完全 | coNP-完全 ($k \geq 1$) |
| 最优代价确定/可能 | $\Delta_2^p$-完全 | - | $\Delta_2^p$-完全 |

### 消融/关键对比

| 对比维度 | 本文 vs 前期工作(BBJ 2024) |
|---------|------------------------|
| 最优代价确定上界 | **$\Delta_2^p$**（首次非平凡上界） | 无上界 |
| NP/coNP硬度的$k$值 | **$k \geq 1$** | $k \geq 3$ |
| DL覆盖范围 | 包含DL-Lite家族 | 仅$\mathcal{EL}_\bot$到$\mathcal{ALCO}$ |
| DL-Lite固定$k$的正面结果 | **AC$^0$** | 无 |

### 关键发现

1. **最优代价语义的精确复杂度为$\Delta_2^p$**——这个结果适用于从DL-Lite$_{\text{core}}$到$\mathcal{ALCHIO}$的广泛DL范围
2. **固定代价下DL-Lite的可处理性是惊喜结果**：CQA$_p^k$和IQA$_c^k$的数据复杂度与经典DL-Lite查询应答相同（AC$^0$）
3. **NP硬度在$k=1$即成立**：大幅改进了前期的$k \geq 3$下界
4. **CQA$_c^k$对DL-Lite$_{\text{core}}$是coNP-完全的**——即使CQA$_p^k$和IQA$_c^k$可处理

## 亮点与洞察

1. **代价独立的小解释性质**是核心技术突破：通过迭代商构造，域大小的多项式界限不依赖代价$k$，这是获得NP/coNP上界（而非更高）的关键
2. **AC$^0$结果的极小解释性质设计精巧**："稀有类型"的概念将无限多的ABox配置压缩为常数个类别，ABox类型中引入$\exists^{>k}\mathsf{r}$信息精确捕捉了代价约束下的结构
3. **一阶可重写性意味着实际可实施**：理论上可以将代价语义下的查询应答编译为标准SQL查询在数据库上执行
4. **下界的改进（$k \geq 1$）**：证明了不可处理性并非高代价值才出现的伪象

## 局限与展望

1. **AC$^0$结果不覆盖CQA$_c^k$**（被证明是coNP-完全的），限制了正面结果的适用范围
2. **$\Delta_2^p$硬度依赖二进制编码权重**——一元编码时复杂度可能较低
3. **未处理功能约束或数量限制的DL**（如$\mathcal{ALCQI}$），这需要完全不同的方法
4. **未提供重写的实际实现**，AC$^0$的理论结果到高效实现之间仍有差距
5. **DL-Lite$_\mathcal{R}$的处理仍正在进行**（含负角色包含）

## 相关工作与启发

- 代价语义统一了经典OMQA和基于加权修复的AR语义
- 与Lutz & Manière (2024) 的限界描述技术联系紧密（迭代商构造的灵感来源）
- 启发：在不一致性处理中，量化方法（权重/代价）比定性方法（简单的consistent/inconsistent）能提供更丰富的语义，同时保持可处理性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (AC$^0$结果超出预期，商构造技术有深度)
- 实验充分度: ⭐⭐⭐ (纯理论工作，无实现/实验，但复杂度landscape完整)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，但技术密度高，对非专家不友好)
- 价值: ⭐⭐⭐⭐ (对描述逻辑和知识表示领域有重要理论贡献)

<!-- RELATED:START -->

## 相关论文

- [Description Logics with Two Types of Definite Descriptions: Complexity, Expressiveness, and Automated Deduction](description_logics_with_two_types_of_definite_descriptions_complexity_expressive.md)
- [Cost-Free Neutrality for the River Method](cost-free_neutrality_for_the_river_method.md)
- [From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm](from_decision_trees_to_boolean_logic_a_fast_and_unified_shap_algorithm.md)
- [The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets](../../NeurIPS2025/others/the_cost_of_robustness_tighter_bounds_on_parameter_complexity_for_robust_memoriz.md)
- [ShortageSim: Simulating Drug Shortages under Information Asymmetry](shortagesim_simulating_drug_shortages_under_information_asymmetry.md)

<!-- RELATED:END -->
