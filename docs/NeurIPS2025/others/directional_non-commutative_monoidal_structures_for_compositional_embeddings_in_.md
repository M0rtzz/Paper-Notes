---
title: >-
  [论文解读] Directional Non-Commutative Monoidal Structures for Compositional Embeddings in Machine Learning
description: >-
  [NEURIPS2025][non-commutative algebra] 提出一种基于方向性非交换幺半群算子的代数框架，为多维组合嵌入提供统一数学基础，将 SSM 递归、Transformer 自注意力和 RoPE 位置编码统一为特例。
tags:
  - NEURIPS2025
  - non-commutative algebra
  - compositional embeddings
  - monoidal structure
  - interchange law
  - positional encoding
  - SSM
  - Transformer
---

# Directional Non-Commutative Monoidal Structures for Compositional Embeddings in Machine Learning

**会议**: NEURIPS2025  
**arXiv**: [2505.15507](https://arxiv.org/abs/2505.15507)  
**代码**: 无（纯理论工作）  
**领域**: others  
**关键词**: non-commutative algebra, compositional embeddings, monoidal structure, interchange law, positional encoding, SSM, Transformer  

## 一句话总结
提出一种基于方向性非交换幺半群算子的代数框架，为多维组合嵌入提供统一数学基础，将 SSM 递归、Transformer 自注意力和 RoPE 位置编码统一为特例。

## 背景与动机
- 很多结构化数据（序列、图像、视频）具有沿多个维度的层次化组合性质，但现有代数工具主要针对一维组合（如自由群、非交换半群）
- 二维及更高维的组合缺乏公认的代数框架：一个 2D 数组既可以按行组合也可以按列组合，这种多路径组合不适配传统的一维代数体系
- 现有 ML 架构（Transformer 位置编码、SSM 递归）虽然隐式实现了某种组合，但缺乏严格的代数基础和统一视角
- 将 token 表示为矩阵而非向量会破坏注意力机制的核心假设，矩阵表示与向量学习架构之间存在张力

## 核心问题
**如何设计一个既兼容向量学习架构、又能沿多个轴自然支持组合的代数结构？**

## 方法详解

### 一维情形
每个元素为元组 $({\bf a}, A)$，其中 ${\bf a} \in \mathbb{R}^n$，$A \in GL(n)$。组合算子定义为：

$$({\bf a}, A) \circ ({\bf b}, B) := ({\bf a} + A{\bf b}, AB)$$

该运算满足结合律但不满足交换律，本质上是仿射变换群的半直积。

### 多轴推广
$D$ 维的元素表示为 ${\bf x} = ({\bf a}, R_1^{n_1}, R_2^{n_2}, \ldots, R_D^{n_D})$，其中 $R_i \in GL(n)$ 为第 $i$ 轴的变换矩阵，$n_i \in \mathbb{Z}$ 为该轴上的位置/范围。沿第 $k$ 轴的组合定义为：

$${\bf x} \circ_k {\bf y} = ({\bf a} + R_k^{n_k} {\bf b},\; R_1^{n_1}, \ldots, R_k^{n_k + m_k}, \ldots, R_D^{n_D})$$

### 四大核心性质
1. **轴特定组合算子**：每个轴 $i$ 有独立的组合算子 $\circ_i$
2. **沿每轴结合律**：$(x \circ_i y) \circ_i z = x \circ_i (y \circ_i z)$
3. **全局交换律（interchange law）**：$(x \circ_i y) \circ_j (z \circ_i w) = (x \circ_j z) \circ_i (y \circ_j w)$，当且仅当 $R_i R_j = R_j R_i$
4. **单轴非交换性**：$x \circ_i y \neq y \circ_i x$，保留方向/顺序信息

### 非交换自注意力机制
定义从位置 $q$ 到 $p$ 的相对变换 $T_{p,q} = R_q R_{q+1} \cdots R_{p-1}$，将其应用到 key 和 value：

$$\tilde{K}_{p,q} = T_{p,q} K_q, \quad \tilde{V}_{p,q} = T_{p,q} V_q$$

注意力权重和输出按标准方式计算，但位置信息以乘法方式（而非加法）编码进去。

### 多维推广
对于 $D$ 维数据点，相对变换为 $T_{p,q} = \prod_{i=1}^D R_i^{(n_{p,i} - n_{q,i})}$，因 $R_i$ 两两可交换故顺序无关。

### 统一 SSM 与 Transformer
- **SSM 递归**：$y_k = \sum_{i \le k} C_k (\prod_{j=i}^{k-1} A_j) B_i x_i$
- **Vanilla Transformer**：$y_k = \sum_{i \le k} \alpha_{ik} V_i$
- **本框架**：$y_k = \sum_{i \le k} \alpha_{ik} (\prod_{j=i}^{k-1} R_j) V_i$

SSM 是注意力权重退化为隐式（均匀）且交互受限为递归结构的特例；标准 Transformer 是变换矩阵退化为单位阵的特例。

### m-表示：平移不变的组合嵌入
通过滑动窗口 + 旋转变换构造局部有序但全局平移不变的表示。窗口嵌入为 $s_k = \sum_{i=1}^m R^{i-1} a_{k+i-1}$，取分块范数后求和得到 m-representation $v$，仅对内容变化敏感而对全局平移不变。

### RoPE 作为特例
当所有 $R_i = R$（固定的块对角旋转矩阵）时，$T_{p,q} = R^{(p-q)}$ 仅依赖相对位置，恰好恢复 Rotary Position Embedding。二维情形取 $R_x, R_y$ 两个旋转矩阵即得 2D RoPE。

## 实验关键数据
**本文为纯理论工作，不包含任何实验。** 作者明确表示将实证验证留给后续工作。

## 亮点
- **统一性极强**：用一个代数框架将 SSM、Transformer 自注意力、RoPE、仿射变换等看似不同的范式统一为特例
- **数学基础扎实**：严格证明了结合律、交换律、逆元等代数性质
- **多维扩展自然**：从 1D 到 2D/3D 的推广不是特设的，而是代数结构自然推导的结果
- **计算效率友好**：当 $R_i$ 参数化为 $2 \times 2$ 块旋转矩阵时，乘法退化为角度相加，可高效并行扫描

## 局限与展望
- **最大短板：无实验验证**，不清楚该代数结构在实践中是否真的带来性能提升
- $R_i \in GL(n)$ 在高维时计算和存储开销大，但块对角旋转参数化可缓解
- 交换律要求 $R_i R_j = R_j R_i$，限制了变换矩阵的表达能力（只能用可交换的矩阵族）
- 框架的实际训练稳定性未知
- 对边界条件和退化情形（如 $R$ 接近奇异矩阵）的讨论不足

## 与相关工作的对比

| 方法 | 组合方式 | 多维 | 代数保证 |
|------|---------|------|---------|
| RoPE | 旋转矩阵乘法 | 可扩展但受限 | 本框架特例 |
| 矩阵空间模型 (Rudolph 2010) | 矩阵乘法 | 仅 1D | 非交换半群 |
| S4/Mamba (SSM) | 状态递归 | 仅 1D (S4ND 除外) | 本框架特例 |
| Transformer 自注意力 | 加权求和 | 需位置编码 | 本框架特例 |
| **本文** | **轴特定非交换幺半群** | **自然多维** | **结合律 + 交换律** |

## 启发与关联
- 为设计新架构提供了理论指导：可以有意识地选择不同的 $R_i$ 参数化来获得特定的归纳偏置
- 将"位置编码"从 ad-hoc 设计提升到有代数保证的框架层面
- 多维交换律暗示图像/视频模型中行组合和列组合的结果应当一致，这是一个可测试的设计原则
- 可能启发新的高效并行算法：结合律保证了前缀扫描的可行性
- m-representation 的平移不变性 + 局部有序性组合对内容匹配任务有潜在价值
- 轴特定的拼接操作 $\oplus_k$ 为多模态融合提供了代数化的拼接/对齐原语

## 评分
- 新颖性: ⭐⭐⭐⭐⭐（首次提出此类多维非交换组合代数框架）
- 实验充分度: ⭐（无任何实验）
- 写作质量: ⭐⭐⭐⭐（理论推导清晰，但部分符号较重）
- 价值: ⭐⭐⭐（理论贡献有意义，但缺乏实验大幅削弱了实际影响力）

<!-- RELATED:START -->

## 相关论文

- [Learning non-equilibrium diffusions with Schrödinger bridges: from exactly solvable to simulation-free](learning_non-equilibrium_diffusions_with_schrödinger_bridges_from_exactly_solvab.md)
- [Directional Sheaf Hypergraph Networks: Unifying Learning on Directed and Undirected Hypergraphs](../../ICLR2026/others/directional_sheaf_hypergraph_networks_unifying_learning_on_directed_and_undirect.md)
- [Graph-guided Cross-composition Feature Disentanglement for Compositional Zero-shot Learning](../../ACL2025/others/graph-guided_cross-composition_feature_disentanglement_for_compositional_zero-sh.md)
- [Better Embeddings with Coupled Adam](../../ACL2025/others/better_embeddings_with_coupled_adam.md)
- [SAD Neural Networks: Divergent Gradient Flows and Asymptotic Optimality via o-minimal Structures](sad_neural_networks_divergent_gradient_flows_and_asymptotic_optimality_via_o-min.md)

<!-- RELATED:END -->
