---
title: >-
  [论文解读] Strassen Attention, Split VC Dimension and Compositionality in Transformers
description: >-
  [NeurIPS 2025][LLM/NLP][Transformer] 提出 Splitting VC 维度理论工具证明了单层 softmax Transformer（即使无限精度）在组合推理任务上的根本限制，并设计了具有亚立方时间复杂度的 Strassen 注意力机制来突破这些限制。
tags:
  - NeurIPS 2025
  - LLM/NLP
  - Transformer
  - 注意力机制
  - 组合推理
  - VC 维度
  - 复杂度下界
---

# Strassen Attention, Split VC Dimension and Compositionality in Transformers

**会议**: NeurIPS 2025  
**arXiv**: [2501.19215](https://arxiv.org/abs/2501.19215)  
**代码**: 未公开  
**领域**: llm_nlp  
**关键词**: Transformer 理论, 注意力机制, 组合推理, VC 维度, 复杂度下界  

## 一句话总结

提出 Splitting VC 维度理论工具证明了单层 softmax Transformer（即使无限精度）在组合推理任务上的根本限制，并设计了具有亚立方时间复杂度的 Strassen 注意力机制来突破这些限制。

## 研究背景与动机

Transformer 架构在许多 NLP 任务上表现出色，但在需要组合推理（compositionality）的任务上往往表现不佳。组合推理要求模型将知识块组合起来解决新任务，被认为是语言系统性泛化的核心能力。

现有理论工作的不足：
- Hahn (2020) 使用 hardmax 注意力的限制证明，但 hardmax 与实际 softmax 差距较大
- Sanford et al. (2023) 和 Peng et al. (2024) 基于通信复杂度的方法，**仅适用于有限精度**（亚线性位数）的 Transformer
- 这引发一个关键问题：是否使用更高精度算术就能绕过这些限制？

此外，已有的高阶注意力机制（如三阶注意力）虽理论上能解决这些问题，但时间复杂度为 $O(n^3)$，扩展性差。

本文的核心动机是：
1. 建立对**任意精度**（包括无限精度）softmax Transformer 的理论限制
2. 设计一个**更高效**的注意力机制来突破这些限制

## 方法详解

### 整体框架

本文有两大贡献线：**理论下界**（通过 Splitting VC 维度）和**新注意力机制**（Strassen attention）。

### 关键设计一：Splitting VC 维度

核心思想是将布尔函数 $f: \Sigma^n \to \{0,1\}$ 的 $n$ 个参数**分裂**为两部分——一部分视为输入、另一部分视为参数——从而将函数看作假设类，再计算该假设类的 VC 维度。

给定位置集合 $A \subseteq \{1,\ldots,n\}$，构造布尔矩阵 $M_f^A$：
- 行索引为 $w^1 \in \Sigma^A$（输入部分）
- 列索引为 $w^2 \in \Sigma^B$（参数部分），$B = [n] \setminus A$
- 矩阵元素 $M_A^f(w^1, w^2) = f(w^1 \oplus w^2)$

**定义**：$\text{split-VC}(f)$ 为在所有可能的分割 $A$ 上，$M_f^A$ 列集的 VC 维度的最大值。

### 关键设计二：主定理

**定理 3.2**：设 $T$ 为单层标准注意力 Transformer（可使用**实数无限精度**运算），其嵌入维度 $d$、注意力头数 $H$、输出 MLP 大小 $L$。若 $T$ 能计算函数 $f$，则：

$$\text{size}(T) = \max\{d, H, L\} \geq \text{split-VC}(f)^{\Omega(1)}$$

证明核心思路：
1. 将注意力输出在辅助 token 位置分解为 $\frac{\alpha^{(h)}(w^1) + \beta^{(h)}(w^2) + \gamma^{(h)}}{\lambda^{(h)}(w^1) + \mu^{(h)}(w^2) + \nu^{(h)}}$ 的分数形式
2. 利用 Goldberg-Jerrum 定理对参数化函数类的 VC 维度上界
3. 这一分解只依赖注意力的求和结构，与精度无关

### 关键设计三：三个任务的下界

| 任务 | 定义 | split-VC 下界 |
|------|------|--------------|
| 函数复合 | 给定 $g,h:[n]\to[n]$，计算 $h(g(x))$ | $\geq n$ |
| Match3 | 判断是否存在 $j,k$ 使得 $p_i + p_j + p_k \equiv 0 \pmod{m}$ | $\geq n/2$ |
| 二元关系复合 | 计算 $(B \circ A)_{ij} = \bigvee_k (A_{ik} \wedge B_{kj})$ | $\geq \sqrt{n}$ |

这些都意味着单层标准注意力 Transformer（$\text{size}(T) = n^{o(1)}$）**不可能**解决这些任务。

### 关键设计四：Strassen 注意力

受 Strassen 矩阵乘法的启发，Strassen 注意力的计算公式为：

$$a_i = \sum_{j,k} a_{ijk}(v_j \odot v_k)$$

$$a_{ijk} = \text{Softmax}_{j,k}\left(\frac{f_i g_j + g_j h_k + h_k f_i}{\sqrt{d}}\right)$$

其中 $f_i = W^f x_i$，$g_j = W^g x_j$，$h_k = W^h x_k$。

关键优势：注意力分数使用**成对点积和**而非三坐标乘积，可分解为常数个 $n \times n$ 矩阵乘法。

**定理 4.1**：Strassen 注意力层的时间复杂度为 $O(n^\omega \cdot d)$，其中 $\omega < 2.372$ 是矩阵乘法指数。相比三阶注意力的 $O(n^3)$，这是显著的效率提升。

### 损失函数 / 训练策略

本文主要是理论工作。实验部分使用 AdamW 优化器，学习率 $10^{-3}$，在合成数据上训练。对于每种注意力机制，使用相同的训练配置进行公平比较。

## 实验关键数据

### 主实验

在四个组合推理任务上比较不同注意力机制的表现：

| 注意力类型 | 函数复合 | Match3 | 二元关系复合 | 商二元关系复合 | 时间复杂度 |
|-----------|---------|--------|------------|--------------|-----------|
| 标准注意力 | ✗ 失败 | ✗ 失败 | ✗ 失败 | ✗ 失败 | $O(n^2 d)$ |
| 三角注意力 | ✗ 失败 | — | ✓ 成功 | ✗ 失败 | $O(n^{3/2} d)$* |
| 三阶注意力 | ✓ 成功 | ✓ 成功 | ✓ 成功 | ✓ 成功 | $O(n^3 d)$ |
| **Strassen 注意力** | **✓ 成功** | **✓ 成功** | **✓ 成功** | **✓ 成功** | $O(n^\omega d)$ |

*三角注意力仅在输入已结构化为矩阵时为 $O(n^{3/2})$

### 消融实验

- **训练效率比较**：Strassen 注意力在训练时间和计算资源方面优于三阶注意力，因为其避免了 $O(n^3)$ 操作
- **嵌入维度影响**：实验验证了理论预测——标准注意力在固定小维度下无法学习这些任务，即使增加训练时间也不行
- **商二元关系复合任务**：这是专门设计的新任务，能将 Strassen 注意力与标准+三角注意力的混合体区分开

### 关键发现

1. 标准注意力在所有四个任务上完全失败，验证了理论下界
2. Strassen 注意力在所有任务上都能收敛到理论解，且效率优于三阶注意力
3. 三角注意力只能解决输入天然为矩阵结构的二元关系复合任务
4. 商二元关系复合任务成功区分了 Strassen 注意力与标准+三角注意力的混合体

## 亮点与洞察

1. **精度无关的下界**：首次证明即使 Transformer 可以做无限精度算术运算，单层标准注意力仍然解不了某些组合任务——这是一个真正的**表达能力**限制，而非精度限制
2. **优雅的理论工具**：Splitting VC 维度概念简洁而强大，将组合优化问题与学习理论中的经典 VC 维度桥接起来
3. **Strassen 矩阵乘法的跨领域灵感**：将 70 年代的矩阵乘法算法思想（Strassen 1969）巧妙地引入注意力机制设计
4. **理论与实验的一致性**：所有理论预测都在实验中得到了精确验证

## 局限与展望

1. **仅限单层分析**：理论结果仅适用于单层 Transformer；多层设置下的表达能力分析仍是开放问题
2. **实验仅在合成数据上**：未在真实 NLP 任务上验证 Strassen 注意力的实际效用
3. **实现复杂度**：虽然理论时间复杂度更优，但快速矩阵乘法算法的常数因子可能使实际加速有限
4. **与现代 Transformer 架构的差距**：未考虑层归一化、多层交互等实际组件的影响
5. **Strassen 注意力的可训练性**：虽然实验展示了收敛性，但大规模训练稳定性未知

## 相关工作与启发

- **与 Sanford et al. (2023) 的关系**：扩展了其 Match3 任务的下界到无限精度设场景
- **与 Peng et al. (2024) 的关系**：同样研究函数复合，但克服了通信复杂度方法对精度的依赖
- **与 Bergen et al. (2021) 的三角注意力**：实验展示了 Strassen 能解决三角注意力无法处理的任务
- **启发**：Splitting VC 维度方法可能适用于分析更广泛的神经网络架构的表达能力限制

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — Splitting VC 维度和 Strassen 注意力都是全新贡献
- **理论深度**: ⭐⭐⭐⭐⭐ — 证明严谨且揭示了深刻的结构性限制
- **实验充分度**: ⭐⭐⭐⭐ — 合成实验充分验证理论，但缺少真实任务评估
- **写作质量**: ⭐⭐⭐⭐⭐ — 清晰的组织和精确的数学陈述
- **实用价值**: ⭐⭐⭐ — 目前更偏理论，Strassen 注意力的实际应用前景需进一步探索
- **综合**: ⭐⭐⭐⭐⭐ (9/10) — 顶级理论工作，优雅地连接了复杂度理论与 Transformer 架构设计

<!-- RELATED:START -->

## 相关论文

- [CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers](cat_circular-convolutional_attention_for_sub-quadratic_transformers.md)
- [Spectral Conditioning of Attention Improves Transformer Performance](spectral_conditioning_of_attention_improves_transformer_performance.md)
- [Vision Transformers are Circulant Attention Learners](../../AAAI2026/llm_nlp/vision_transformers_are_circulant_attention_learners.md)
- [Linear Transformers Implicitly Discover Unified Numerical Algorithms](linear_transformers_implicitly_discover_unified_numerical_algorithms.md)
- [Rethinking Spiking Self-Attention Mechanism: Implementing a-XNOR Similarity Calculation in Spiking Transformers](../../CVPR2025/llm_nlp/rethinking_spiking_self-attention_mechanism_implementing_a-xnor_similarity_calcu.md)

<!-- RELATED:END -->
