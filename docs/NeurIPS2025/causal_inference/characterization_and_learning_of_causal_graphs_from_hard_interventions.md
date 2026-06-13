---
title: >-
  [论文解读] Characterization and Learning of Causal Graphs from Hard Interventions
description: >-
  [NeurIPS 2025][因果推理][hard intervention] 首次系统分析硬干预（hard interventions）在含隐变量因果发现中的理论优势，提出广义do-演算（4条规则）和孪生增强MAG图表示，给出 $\mathcal{I}$-Markov 等价类的充要图条件，并设计可证明正确的FCI变体学习算法；实验表明硬干预比软干预将等价类缩小37-57%。
tags:
  - "NeurIPS 2025"
  - "因果推理"
  - "hard intervention"
  - "causal discovery"
  - "Markov equivalence class"
  - "twin augmented MAG"
  - "do-calculus"
  - "latent variables"
---

# Characterization and Learning of Causal Graphs from Hard Interventions

**会议**: NeurIPS 2025  
**arXiv**: [2505.01037](https://arxiv.org/abs/2505.01037)  
**作者**: Zihan Zhou, Muhammad Qasim Elahi, Murat Kocaoglu (Johns Hopkins / Purdue)  
**代码**: 待确认  
**领域**: 音频语音  
**关键词**: hard intervention, causal discovery, Markov equivalence class, twin augmented MAG, do-calculus, latent variables

## 一句话总结

首次系统分析硬干预（hard interventions）在含隐变量因果发现中的理论优势，提出广义do-演算（4条规则）和孪生增强MAG图表示，给出 $\mathcal{I}$-Markov 等价类的充要图条件，并设计可证明正确的FCI变体学习算法；实验表明硬干预比软干预将等价类缩小37-57%。

## 研究背景与动机

**领域现状**：因果发现旨在从观测和实验数据中推断因果结构，核心工具是条件独立性（CI）约束和d-分离准则。现有干预性因果发现方法主要处理软干预（soft intervention），即改变变量的条件分布但保留因果图结构不变。

**现有痛点**：软干预不移除被干预变量的入边，因此不会打破 inducing path。当存在隐变量时，大量不同因果图在软干预下具有相同的分布族，Markov 等价类（MEC）过大，无法进一步缩小搜索空间。

**核心矛盾**：硬干预（$\text{do}(X=x)$）直接将变量设定为常数并移除所有入边，能够非局部地改变 d-分离关系。例如对图 $\{X \to Z \to Y, Z \leftrightarrow Y\}$ 做硬干预 $\text{do}(Z)$ 后，inducing path $\langle X,Z,Y\rangle$ 被打断，$X$ 和 $Y$ 变为独立；而软干预无法做到这一点。但硬干预的理论框架和学习算法此前尚未建立。

**本文目标**：
   - 硬干预比软干预到底多提供了哪些分布约束？
   - 如何刻画硬干预下含隐变量因果图的 $\mathcal{I}$-Markov 等价类？
   - 如何设计算法从多个硬干预数据集中学习因果结构？

**切入角度**：从 Pearl 的 do-calculus 的逆向（converse）出发——当两个干预分布不等时，反推图中必须存在的结构约束。利用 F-node 增强图将干预效应显式编码为图结构，从而统一分析。

**核心 idea**：硬干预移除入边后因果图变稀疏，产生更多 do-invariance 约束；孪生增强 MAG 完整捕捉这些新约束，使等价类显著缩小。

## 方法详解

### 整体框架

三层递进的理论贡献：

1. **广义 do-演算**：将 Pearl 原始 3 条规则推广到任意两个硬干预集之间的比较（Proposition 3.2，4 条规则）
2. **$\mathcal{I}$-Markov 等价类刻画**：通过孪生增强 MAG（Twin Augmented MAG）和 $\mathcal{I}$-增强 MAG 给出充要图条件（Theorem 4.7, Proposition 5.2）
3. **学习算法**：基于 FCI 框架的 3 阶段算法，包含 4 条新定向规则（Algorithm 1），证明了 soundness

### 关键设计

#### 1. 广义 do-演算（Proposition 3.2）

- **功能**：将 Pearl 的 3 条 do-规则推广为可在任意两个硬干预集 $\mathbf{I}, \mathbf{J}$ 之间比较分布的 4 条规则
- **核心思路**：定义对称差 $\mathbf{K} = \mathbf{I} \Delta \mathbf{J}$，进而分解为多个子集（$\mathbf{K_I}, \mathbf{K_J}, \mathbf{R_I}, \mathbf{R_J}, \mathbf{W_I}, \mathbf{W_J}$）。每条规则给出：当特定图变换（删除入边/出边）后满足 d-分离条件时，$P_\mathbf{I}(\mathbf{y}|\mathbf{w})$ 与 $P_\mathbf{J}(\mathbf{y}|\mathbf{w})$ 相等
    - Rule 1（CI）：单个干预分布内的条件独立性
    - Rule 2（do-see）：两个干预分布在条件化对称差变量后相等
    - Rule 3（do-do）：两个干预分布的边际相等
    - Rule 4（混合）：统一 do-see 和 do-do 的一般形式
- **关键洞察**：硬干预使被干预变量的入边消失，图变稀疏，产生更多 d-分离关系，从而产生比软干预更多的约束

#### 2. 孪生增强 MAG（Definition 4.5, Theorem 4.7）

- **功能**：构造一种图结构，使得判断两个因果图是否 $\mathcal{I}$-Markov 等价只需检查标准的图性质（骨架 + 非屏蔽碰撞子 + 判别路径）
- **构造过程**：
  1. 对干预对 $(\mathbf{I}, \mathbf{J})$，创建变量的两份副本 $\mathbf{V}^{(\mathbf{I})}, \mathbf{V}^{(\mathbf{J})}$
  2. 在各份副本中保留对应干预图 $\mathcal{D}_{\overline{\mathbf{I}}}, \mathcal{D}_{\overline{\mathbf{J}}}$ 的边
  3. 添加辅助节点 $F$ 并指向对称差 $\mathbf{K}$ 中的所有变量
  4. 取 MAG 后添加对称化边得到 Twin Augmented MAG
- **核心定理（Theorem 4.7）**：两图 $\mathcal{I}$-Markov 等价 $\iff$ 对所有干预对 $(\mathbf{I}, \mathbf{J})$，对应的 Twin Augmented MAG 具有相同骨架、相同非屏蔽碰撞子、相同判别路径碰撞性质
- **Lemma 4.6**：证明 Twin Augmented MAG 是合法的 MAG（保持祖先性和极大性）

#### 3. $\mathcal{I}$-增强 MAG（Definition 5.1）

- **动机**：$k$ 个干预目标产生 $\binom{k}{2}$ 个 Twin Augmented MAG，需要更紧凑的表示
- **做法**：每个干预 $\mathbf{I}$ 对应一个 $\mathcal{I}$-增强 MAG，融合所有包含 $\mathbf{I}$ 的 Twin MAG 中 $\mathbf{V}^{(\mathbf{I})}$ 域的信息。最终输出 $k$ 个图组成的元组，而非 $\binom{k}{2}$ 个
- **Proposition 5.2**：$\mathcal{I}$-增强 MAG 与 Twin Augmented MAG 在等价类刻画上完全等价

#### 4. 学习算法（Algorithm 1）

3 阶段 FCI 变体：

- **Phase I（初始化）**：对每个 $\mathbf{I} \in \mathcal{I}$，创建变量副本 $\mathbf{V}^{(\mathbf{I})}$ 并初始化为完全图（circle edges），创建 F 节点并连接到所有变量
- **Phase II（骨架学习）**：Algorithm 3 通过条件独立性检测确定分离集。非 F 节点间检测 $P_\mathbf{I}(y|\mathbf{w},x) = P_\mathbf{I}(y|\mathbf{w})$；F 节点与变量间检测 $P_\mathbf{I}(y|\mathbf{w}) = P_\mathbf{J}(y|\mathbf{w})$
- **Phase III（定向）**：先用 Rule 0 定向非屏蔽碰撞子，再反复应用 7 条 FCI 规则 + 4 条新规则：
    - **Rule 8**（F-node）：F 节点的所有边都朝外定向
    - **Rule 9**（干预节点）：$X \in \mathbf{I}$ 且 $X, Y$ 相邻 → $X \to Y$（因为 $Y$ 必为 $X$ 的后代）
    - **Rule 10**（骨架一致性）：若 $X \to Y$ 在一个域确定，则在其他域也将 $Y$ 端标记为箭头（祖先关系不可被硬干预逆转）
    - **Rule 11**（inducing path）：若 $\mathbf{J} = \mathbf{I} \cup \{X\}$ 且 $F$ 与 $Y$ 相邻，则 $X \to Y$

### 理论保证

- **Theorem 6.3（正确性）**：在 h-faithfulness 假设下，Algorithm 1 输出的 $\mathcal{I}$-增强图中，每个邻接关系和定向标记都在所有 $\mathcal{I}$-Markov 等价图中一致成立（soundness）
- 完备性（completeness）留作开放问题

## 实验结果

### 实验 1：穷举所有 ADMG

对小规模变量（$n = 2,3,4$）穷举所有 ADMG，比较硬干预与软干预下 $\mathcal{I}$-MEC 大小：

| $n$ | 硬干预 MEC | 软干预 MEC | 比值 | ADMG 总数 |
|-----|-----------|-----------|------|----------|
| 2（随机）| 2.03 | 2.93 | 0.69 | 6 |
| 3（随机）| 19.50 | 30.57 | 0.64 | 200 |
| 4（随机）| 677.13 | 1218.83 | 0.56 | 34,752 |
| 2（完全）| 2.37 | 3.67 | 0.65 | 6 |
| 3（完全）| 14.03 | 24.70 | 0.57 | 200 |
| 4（完全）| 721.37 | 1529.57 | 0.47 | 34,752 |

**核心发现**：硬干预的 MEC 约为软干预的 47-69%。随着变量数增加，比值持续下降——说明硬干预在更复杂的图中优势更大。

### 实验 2：采样 ADMG（$n=5$）

$n=5$ 时 ADMG 总数达 29,983,744，无法穷举。使用 Hoeffding 不等式控制估计误差（$\epsilon=0.01$, 置信度 99%），每个设定采样 23,025 个 ADMG：

- 硬干预的 $\mathbb{E}_\mathcal{S}^{hard}$ 显著低于软干预的 $\mathbb{E}_\mathcal{S}^{soft}$
- 趋势与穷举实验一致：硬干预能更高效地缩小等价类

## 优缺点分析

### 优点

1. **理论贡献扎实**：首次完整刻画硬干预下含隐变量因果图的 $\mathcal{I}$-Markov 等价类，Theorem 4.7 给出充要条件
2. **统一框架**：广义 do-演算的 4 条规则涵盖了硬干预间所有可能的分布比较，软干预可视为特例
3. **紧凑表示**：$\mathcal{I}$-增强 MAG 将 $\binom{k}{2}$ 个对图压缩为 $k$ 个，大幅降低复杂度
4. **4 条新定向规则**设计精巧，充分利用了硬干预的独特性质（入边移除、祖先关系不变性、inducing path 打断）
5. **算法具有可证明的正确性**（soundness）

### 局限

1. **完备性未证明**：Algorithm 1 只证明了 soundness，是否能恢复所有可辨识的边尚为开放问题
2. **计算复杂度高**：骨架学习需要测试所有可能的条件集，最坏情况下指数级复杂度
3. **仅针对硬干预**：实际应用中（尤其生物学），精确的硬干预可能难以实现；计算机系统（如微服务架构）中更易实现
4. **实验规模较小**：由于 ADMG 数量超指数增长，实验仅限 $n \leq 5$；大规模图的表现有待验证
5. **未提供有限样本保证**：理论结果基于总体层面的独立性，未分析估计误差对算法的影响

## 相关工作对比

| 方法 | 干预类型 | 隐变量 | 等价类刻画 | 学习算法 |
|------|---------|--------|-----------|---------|
| Hauser & Bühlmann (2012) | 硬/软 | ✗ | ✓ | ✓ |
| Yang et al. (2018) | 硬/软 | ✗ | ✓（相同刻画） | ✓ |
| Kocaoglu et al. (2019) | 软 | ✓ | ✓（增强 MAG） | ✓ |
| Jaber et al. (2020) | 未知软 | ✓ | ✓（$\Psi$-MEC） | ✓ |
| **本文** | **硬** | **✓** | **✓（孪生增强 MAG）** | **✓** |

本文填补了"硬干预 + 隐变量"组合的理论空白。与最接近的 Kocaoglu et al. (2019) 相比，本文利用硬干预移除入边的特性产生更强约束，等价类缩小 30-50%+。

## 读后感

这篇论文是因果发现理论的重要进展。核心洞察非常自然——硬干预比软干预信息量更大——但将这一直觉转化为严谨的等价类刻画需要大量技术工作。孪生增强 MAG 的构造巧妙地将硬干预的"非局部效应"编码为辅助节点 $F$ 的邻接关系，使得标准的 MAG 比较方法可以直接复用。

从实用角度看，这项工作对计算机系统（如微服务因果分析、A/B 测试）中的因果发现最有价值，因为这类系统中硬干预是可行的。生物学领域的适用性相对受限。完备性和有限样本分析是明显的后续方向。

## 实验关键数据

### 主实验（枚举所有ADMG）

| 变量数 | 硬干预MEC | 软干预MEC | 比率(硬/软) |
|-------|---------|---------|----------|
| n=2 | 2.03 | 2.93 | 0.69 |
| n=3 | 19.50 | 30.57 | 0.64 |
| n=4(完全DAG) | 721.37 | 1529.57 | **0.47** |
| n=5(采样) | 0.028 | 0.061 | 0.46 |

### 消融：混杂密度的影响（n=5）

| 双向边密度ρ | 硬/软比率 |
|----------|---------|
| ρ=0.1（低混杂） | 0.804 |
| ρ=0.5（中混杂） | 0.459 |
| ρ=0.9（高混杂） | **0.365** |

### 关键发现
- **硬干预优势随图规模增长**：n=2时缩减31% → n=4时缩减53%
- **混杂越多，硬干预优势越明显**：高混杂(ρ=0.9)时MEC缩小63.5%
- **非局部效应是关键**：硬干预通过移除入边打断了远距离变量间的诱导路径

## 亮点与洞察
- **硬干预的非局部信息量首次被量化**：不仅是理论上更强，实验给出了明确的缩减比率
- **广义do-演算的Rule 4统一了极端情况**：为混合硬/软干预的未来工作奠定基础
- **孪生增强MAG的对称化策略**：用可观察量表达不可观察约束，使基于成对分布的学习可行

## 局限与展望
- 仅证明算法正确性（soundness），完备性（completeness）未证明
- h-faithful假设在实际数据中难以验证
- 多干预目标时计算复杂度爆炸（$\binom{k}{2}$ 对图构造）
- 大规模图（n>6）的实验缺失

## 相关工作与启发
- **vs 软干预因果发现（GIES等）**：硬干预提供更多信息，MEC更小
- **vs FCI算法**：本文扩展FCI处理硬干预特有的约束

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统化硬干预因果发现的理论框架
- 实验充分度: ⭐⭐⭐ 枚举+采样实验，但缺少真实数据和有限样本分析
- 写作质量: ⭐⭐⭐⭐ 理论严谨，定义和定理体系完整
- 价值: ⭐⭐⭐⭐ 为因果发现提供了新的理论工具和算法基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Cyclic Counterfactuals under Shift–Scale Interventions](cyclic_counterfactuals_under_shift-scale_interventions.md)
- [\[NeurIPS 2025\] It's Hard to Be Normal: The Impact of Noise on Structure-agnostic Estimation](its_hard_to_be_normal_the_impact_of_noise_on_structure-agnostic_estimation.md)
- [\[ACL 2025\] CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation](../../ACL2025/causal_inference/causalrag_integrating_causal_graphs_into_retrieval-augmented_generation.md)
- [\[NeurIPS 2025\] Do-PFN: In-Context Learning for Causal Effect Estimation](do-pfn_in-context_learning_for_causal_effect_estimation.md)
- [\[NeurIPS 2025\] Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization](bi-level_decision-focused_causal_learning_for_large-scale_marketing_optimization.md)

</div>

<!-- RELATED:END -->
