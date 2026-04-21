---
title: >-
  [论文解读] Tokenisation is NP-Complete
description: >-
   证明了分词问题（tokenisation）的两种变体——直接分词和自底向上分词——都是 NP 完全的，通过从 max-2-SAT 问题多项式时间归约实现，这意味着不可能找到高效的最优分词算法，BPE 等近似方法是合理选择。
tags:

---

# Tokenisation is NP-Complete

- **会议**: ACL 2025
- **arXiv**: [2412.15210](https://arxiv.org/abs/2412.15210)
- **代码**: 未提供
- **领域**: NLP理解 / 计算语言学 / 理论
- **关键词**: Tokenisation, NP-Completeness, BPE, Compression, Max-2-SAT Reduction

## 一句话总结

证明了分词问题（tokenisation）的两种变体——直接分词和自底向上分词——都是 NP 完全的，通过从 max-2-SAT 问题多项式时间归约实现，这意味着不可能找到高效的最优分词算法，BPE 等近似方法是合理选择。

## 研究背景与动机

- **现有痛点**：分词（tokenisation）是 NLP 的第一步也是最底层步骤，BPE 和 UnigramLM 是最常用的分词算法，但它们都是贪心/启发式方法，不保证找到最优解（即最大化压缩的词表）。研究者一直不清楚是否存在高效的精确算法来替代这些近似方法。坏的分词选择会导致下游任务性能下降（如 GPT-4 无法正确计算 strawberry 中 r 的个数）。

- **核心矛盾**：分词的优化目标（最大化压缩）清晰明确，但在给定词表大小 $K$ 的约束下寻找最优词表的计算复杂度一直未被确定。如果是 P 问题则应开发精确算法替代 BPE；如果是 NP 难则说明近似算法不可避免。这一基础理论问题直接影响分词算法的研究方向。

- **本文要解决**：(1) 证明直接分词问题（找最优词表使压缩最大化）是 NP 完全的；(2) 证明自底向上分词问题（找最优 merge 操作序列）也是 NP 完全的；(3) 讨论理论结果对实践的意义。

- **切入角度**：作者将分词问题形式化为判定问题——"是否存在一个词表/merge 序列使数据集被压缩到不超过 $\delta$ 个符号"——然后从已知 NP 难问题 max-2-SAT 构造多项式时间归约。关键洞察：max-2-SAT 中变量的 True/False 赋值可映射为"选择哪个 subword 放入词表"的选择。

## 方法详解

### 整体框架

本文是纯理论工作。核心是两个 NP 完全性证明，每个包含两部分：(1) 证明问题在 NP 中（给定解可以多项式时间验证）；(2) 证明问题是 NP 难的（从 max-2-SAT 归约）。

### 关键设计

1. **分词问题的形式化定义**：

    - 功能：严格定义分词的两种变体
    - 核心思路：分词器定义为三元组 $\langle \mathcal{S}, \text{tok}, \text{detok}\rangle$，其中 $\mathcal{S}$ 是词表（包含所有原始字符加 $K$ 个额外 subword）。**直接分词**是给定词表后找最短 subword 串（用动态规划可在 $O(|c|^2)$ 内完成）；**自底向上分词**是通过一系列 merge 操作逐步合并相邻 subword（即 BPE 的工作方式）。优化目标均为最大化压缩 $\mathfrak{G}(s)=-|s|$。
    - 设计动机：区分这两种变体很重要，因为它们对应不同的实际分词算法（UnigramLM vs BPE），且技术上需要不同的归约构造

2. **直接分词的 NP 完全性证明（Reduction 1）**：

    - 功能：证明寻找最优词表是 NP 难的
    - 核心思路：给定 max-2-SAT 实例（$J$ 个变量、$I$ 个子句、阈值 $\psi$），构造字母表 $\Sigma = \{\circledcirc\} \cup \{x_j^T, x_j^F\}_{j=1}^J$ 和三组字符串 $\mathcal{D}_1, \mathcal{D}_2, \mathcal{D}_3$。$\mathcal{D}_1$ 的重复次数 $f$ 极大迫使最优词表必须选择形如 $\circledcirc x_j^T \circledcirc$ 或 $\circledcirc x_j^F \circledcirc$ 的 subword；$\mathcal{D}_2$ 进一步迫使对每个变量只能选其中之一（对应 True/False 赋值）；$\mathcal{D}_3$ 编码 SAT 子句使得满足更多子句的赋值对应更好的压缩。词表大小 $K=J$，压缩目标 $\delta = (4f+3f')J + 5I - 2\psi$。
    - 设计动机：归约的核心技巧是用数据重复次数（$f \gg f' \gg 1$ 的量级关系）创建"优先级层次"，确保最优解必须按特定形式选择 subword

3. **自底向上分词的 NP 完全性证明（Reduction 2）**：

    - 功能：证明寻找最优 merge 序列是 NP 难的
    - 核心思路：类似思路但更复杂，需引入额外字符 $\otimes$ 和五组字符串 $\mathcal{D}_1$-$\mathcal{D}_5$。由于 merge 操作的顺序和左到右应用规则带来额外约束，需更精细的构造确保 merge 选择对应 SAT 赋值。$K=8J$（需更多 merge 操作）。
    - 设计动机：自底向上分词更接近实际使用的 BPE，其 NP 完全性意义更大

### NP 成员性证明

- **直接分词**：给定词表 $\mathcal{S}$ 作为证书，用 PathPiece 方法在 $O(|c|^2)$ 内验证，多项式完成
- **自底向上分词**：给定 merge 序列作为证书，逐个应用 merge 每个 $O(|c|)$，总时间 $O(|\mathcal{D}||c||m|)$

## 实验关键数据

### 理论结果总结

| 分词变体 | NP 完全性 | 归约来源 | 词表大小参数 | 关键构造 |
|---------|----------|---------|-------------|---------|
| 直接分词 | **是** | max-2-SAT | $K=J$ | 3 组字符串 + 2 频率参数 |
| 自底向上分词 | **是** | max-2-SAT | $K=8J$ | 5 组字符串 + 4 频率参数 |
| 直接分词（无重复数据集） | **是** | §6.3 变体 | — | 字母表扩展 |
| 自底向上分词（无重复数据集） | **是** | §6.3 变体 | — | 字母表扩展 |

### 实际分词算法复杂度对比

| 算法 | 是否最优 | 运行时间 | 说明 |
|------|---------|---------|------|
| BPE (贪心) | 否 | $O(N \cdot K)$ | 自底向上，每步选最频繁对 |
| UnigramLM | 否 | 迭代优化 | 直接分词，EM 近似 |
| PathPiece | 给定词表最优 | $O(\|c\|^2)$ | 直接分词的应用阶段 |
| 最优词表搜索 | 是 | **NP 难** | 本文证明 |

### 关键发现

- **BPE 等贪心算法的理论必要性被证实**：不存在多项式时间精确最优分词算法（除非 P=NP），BPE 和 UnigramLM 作为近似算法是合理的研究方向
- **并发工作一致性**：Kozma & Voderholzer (2024) 独立证明了自底向上分词的 APX-hardness（更强结论），两项独立工作结论一致增加可信度
- **实践意义**：对分词器压缩、多语言公平性等应用，应聚焦改进近似算法质量而非追求精确解

## 亮点与洞察

- **基础理论贡献**：在 NLP 最基础的分词步骤上给出了计算复杂度的确定性回答，为分词算法研究画定了理论边界。这类"为什么近似解够用"的理论解释在实验驱动的 NLP 领域尤为珍贵
- **归约构造的技巧**：用数据重复次数的量级差异（$f \gg f' \gg 1$）创建"优先级层次"是一个很巧的技巧，使最优解结构被严格约束，可在其他归约中借鉴
- **两种变体的统一处理**：同时证明直接分词和自底向上分词的 NP 完全性，覆盖了 UnigramLM 和 BPE 两大主流范式

## 局限性

- 仅考虑压缩作为优化目标，未涉及 unigram log-probability 或 Rényi efficiency 等其他目标的复杂度
- 证明依赖 $K$ 作为输入的一部分；若 $K$ 固定为常数（如 32K），问题可能变得可解
- 未讨论近似比——虽然精确解不可行，但 BPE 的近似质量如何仍是开放问题
- 纯理论工作，没有实验验证（如比较 BPE 与最优解在小规模实例上的差距）

## 相关工作

- **vs BPE (Sennrich et al., 2016)**：BPE 是自底向上分词的贪心实现，本文证明了它所近似的问题本身是 NP 难的
- **vs UnigramLM (Kudo, 2018)**：UnigramLM 对应直接分词，使用 EM 近似，本文证明精确求解是 NP 难的
- **vs Kozma & Voderholzer (2024)**：并发工作证明了自底向上分词的 APX-hardness（更强），两项独立工作互相验证
- **vs Geh et al. (2024)**：已证明基于语言模型的分词函数的 NP 完全性，本文扩展到基于压缩的设定

## 评分

- **新颖性**: 9/10 — 首次证明分词优化的 NP 完全性，填补重要理论空白
- **技术深度**: 9/10 — 归约构造精巧，证明严谨，覆盖多种变体
- **实验充分度**: 4/10 — 纯理论工作，缺乏实验验证
- **清晰度**: 7/10 — 数学符号较多但逻辑清晰，对非理论读者门槛较高
- **总分**: 8/10

<!-- RELATED:START -->

## 相关论文

- [Causal Estimation of Tokenisation Bias](causal_tokenisation_bias.md)
- [Jigsaw++: Imagining Complete Shape Priors for Object Reassembly](../../ICCV2025/others/jigsaw_imagining_complete_shape_priors_for_object_reassembly.md)
- [The Limitations and Power of NP-Oracle-Based Functional Synthesis Techniques](../../AAAI2026/others/the_limitations_and_power_of_np-oracle-based_functional_synthesis_techniques.md)
- [Rank, Chunk, and Expand: Lineage-Oriented Reasoning for Taxonomy Expansion](rank_chunk_and_expand_lineage-oriented_reasoning_for_taxonomy_expansion.md)
- [Enhancing the Comprehensibility of Text Explanations via Unsupervised Concept Discovery](enhancing_the_comprehensibility_of_text_explanations_via_unsupervised_concept_di.md)

<!-- RELATED:END -->
