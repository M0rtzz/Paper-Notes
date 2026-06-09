---
title: >-
  [论文解读] Equivariance by Contrast: Identifiable Equivariant Embeddings from Unlabeled Finite Group Actions
description: >-
  [NeurIPS 2025][等变表示] 提出 Equivariance by Contrast (EbC)，一种仅用编码器的方法，从观测对 $(\mathbf{y}, g \cdot \mathbf{y})$ 中联合学习等变嵌入空间和隐式群表示，使有限群作用在潜空间中对应可逆线性映射，并提供可辨识性理论保证…
tags:
  - "NeurIPS 2025"
  - "等变表示"
  - "对比学习"
  - "非线性ICA"
  - "群表示"
  - "可辨识性"
---

# Equivariance by Contrast: Identifiable Equivariant Embeddings from Unlabeled Finite Group Actions

**会议**: NeurIPS 2025  
**arXiv**: [2510.21706](https://arxiv.org/abs/2510.21706)  
**代码**: 未提供  
**领域**: 表示学习 / 群论  
**关键词**: 等变表示, 对比学习, 非线性ICA, 群表示, 可辨识性  

## 一句话总结

提出 Equivariance by Contrast (EbC)，一种仅用编码器的方法，从观测对 $(\mathbf{y}, g \cdot \mathbf{y})$ 中联合学习等变嵌入空间和隐式群表示，使有限群作用在潜空间中对应可逆线性映射，并提供可辨识性理论保证。

## 背景与动机

- 许多真实推理问题中，观测之间的关系由结构化变换控制：计算机视觉中的旋转/平移、生物学中的基因敲除、神经科学中的感觉刺激
- 目标是学习等变嵌入：在嵌入空间中，群作用对应线性变换 $\mathbf{x}' = \mathbf{R}(g)\mathbf{x}$
- 非线性ICA为此问题提供理论基础，但需要额外结构假设使问题可解
- 已有方法的局限：CARE 限于正交表示，STL 允许非线性等变关系，NFT 需要学习生成模型
- 需要一种方法：无需生成模型、无需群特异性归纳偏置、面向一般线性群表示的编码器方法

## 核心问题

如何从配对观测 $(\mathbf{y}, g \cdot \mathbf{y})$（群元素 $g$ 未知）中学习编码器 $\phi$ 和群表示 $\mathbf{R}'$，使得 $\phi(g \cdot \mathbf{y}) = \mathbf{R}'(g)\phi(\mathbf{y})$ 成立，且具有可辨识性保证。

## 方法详解

### 数据假设与问题设定

数据以批次形式给出，每批包含 $n+1$ 对样本 $\{(\mathbf{y}_i, \mathbf{y}'_i)\}$，同一批内所有对经历相同群作用 $g$。数据生成过程为：

$$\mathbf{y}_i = \mathbf{f}(\mathbf{x}_i), \quad \mathbf{y}'_i = \mathbf{f}(\mathbf{R}(g)\mathbf{x}_i)$$

其中 $\mathbf{f}$ 为未知非线性混合函数，$\mathbf{R}: G \to \text{GL}(d, \mathbb{R})$ 为群的线性表示。

### 隐式群表示估计

使用最小二乘回归从编码后的样本对中估计群表示矩阵：

$$\hat{\mathbf{R}}(\mathbf{X}, \mathbf{X}') = \arg\min_{\mathbf{R} \in \text{GL}(d)} \|\mathbf{X}' - \mathbf{X}\mathbf{R}^\top\|_F^2 = (\mathbf{X}^\top\mathbf{X})^{-1}(\mathbf{X}^\top\mathbf{X}')$$

其中 $n$ 个样本对用于估计 $\hat{\mathbf{R}}$，剩余1对用于查询。

### 对比学习目标

训练目标函数结合了InfoNCE损失与群结构：

$$p_\phi(\mathbf{y}' \mid \mathbf{y}, \mathbf{Y}, \mathbf{Y}', S) = \frac{\exp(-\|\mathbf{u}_\phi(\mathbf{y}, \mathbf{Y}, \mathbf{Y}') - \phi(\mathbf{y}')\|^2)}{\sum_{\mathbf{y}'' \in S} \exp(-\|\mathbf{u}_\phi(\mathbf{y}, \mathbf{Y}, \mathbf{Y}') - \phi(\mathbf{y}'')\|^2)}$$

其中 $\mathbf{u}_\phi$ 是从上下文对推断群作用并应用到查询样本的操作：

$$\mathbf{u}_\phi(\mathbf{y}, \mathbf{Y}, \mathbf{Y}') = \hat{\mathbf{R}}(\phi(\mathbf{Y}), \phi(\mathbf{Y}'))\phi(\mathbf{y})$$

最终优化：$\min_\phi \mathcal{L}[\phi] = -\mathbb{E}[\log p_\phi(\mathbf{y}' \mid \mathbf{y}, \mathbf{Y}, \mathbf{Y}', S)]$

### 内容-风格分离

将嵌入空间分为等变子空间（$n$ 维）和不变子空间（$m$ 维），通过约束群表示矩阵的结构：

$$\hat{\mathbf{R}}_{n+m}' = \begin{pmatrix} \hat{\mathbf{R}}_n & \mathbf{0} \\ \mathbf{0} & \mathbf{I}_m \end{pmatrix}$$

### 可辨识性保证

**定理1**（群表示可辨识性）：在数据多样性条件下，定义 $\mathbf{h} := \phi \circ \mathbf{f}$：

- **(a)** 恢复原始向量空间至线性不确定性：$\mathbf{h}(\mathbf{x}) = \mathbf{L}\mathbf{x}$，$\mathbf{L} \in \text{GL}(d)$
- **(b)** 恢复群表示至共轭形式：$\hat{\mathbf{R}}(\mathbf{h}(\mathbf{X}), \mathbf{h}(g\mathbf{X})) = \mathbf{L}\mathbf{R}(g)\mathbf{L}^{-1}$

**推论**（等变性）：$\mathbf{h}(g\mathbf{x}) = g\mathbf{h}(\mathbf{x})$，即编码器严格保持等变性。

## 实验关键数据

### 合成数据与图像数据综合结果

| 群 $G$ | 数据 | $R^2(x)$ | $R^2(G)$ | Acc(C) | Acc(G,5) |
|--------|------|----------|----------|--------|----------|
| $SO_3$ — InfoNCE | non-linear | 0.0 | 0.0 | 98.9 | — |
| $SO_3$ — **EbC** | non-linear | **99.7** | **99.7** | 99.1 | — |
| $O_3$ — InfoNCE | non-linear | 0.0 | 0.0 | 99.1 | — |
| $O_3$ — **EbC** | non-linear | **99.8** | **99.7** | 99.2 | — |
| $GL_3$ — InfoNCE | non-linear | 0.1 | 0.0 | 98.5 | — |
| $GL_3$ — **EbC** | non-linear | **99.8** | **99.7** | 98.5 | — |
| $R_m \times \mathbb{Z}_n^2$ — InfoNCE | idSprites | — | — | 99.97 | 0.36 |
| $R_m \times \mathbb{Z}_n^2$ — **EbC** | idSprites | — | — | 74.04 | **99.91** |

关键发现：
- EbC 在所有合成群上均达到 $R^2 > 99\%$ 的潜空间恢复和群表示恢复质量
- InfoNCE/LDS/SLDS 基线完全无法恢复群结构（$R^2 \approx 0$），仅能学到不变表示
- 在 idSprites 上存在内容-群结构的权衡：群结构恢复99.91%但内容分类降至74%
- 线性基线 EbC(lin.) 在非线性混合下严重退化（$R^2(x) \approx 60$-$70\%$）

### 模型鲁棒性

- 嵌入维度过大时：群结构kNN准确率保持 >99%，内容分类稳定 >80%
- 维度误设（真实3维）：Acc(G) 在正确维度 $d=3$ 处有清晰峰值，可作为超参数选择依据
- 混合层数增加到4层时性能仍保持良好，之后开始退化

## 亮点

- ⭐ 首个仅用编码器实现通用线性群表示学习的方法（无需生成模型/群特异归纳偏置），涵盖非Abel群 $O(n), GL(n)$
- ⭐ 隐式群表示设计精巧：通过最小二乘直接从嵌入对中估计群矩阵，编码器与群表示通过单一 $\phi$ 统一定义
- ⭐ 理论完备：从非线性ICA的判别形式出发证明线性可辨识性和群表示可辨识性
- Acc(G) 指标可在无真实潜变量的情况下使用，提供了实际可用的模型选择准则

## 局限与展望

- 在 idSprites 上存在内容分类准确率下降（74%），表明内容-风格分离仍有改进空间
- 理论假设要求数据满足"充分多样性"条件，对小数据场景的适用性需要验证
- 目前仅验证了有限群，连续群（如连续 $SO(3)$）的扩展未涉及
- 数据生成过程不包含噪声（定理要求精确群作用），带噪声的鲁棒性仅在附录初步探索
- 实际视觉数据（超越 idSprites）上的广泛评估留作未来工作

## 与相关工作的对比

| 方法 | 类型 | 群表示 | 需要生成模型 | 可辨识性 |
|------|------|--------|------------|---------|
| CARE | 编码器 | 正交（超球面） | 否 | 有限 |
| STL | 编码器 | 非线性等变 | 否 | 否 |
| NFT | 生成+编码 | 一般线性 | 是 | 是 |
| **EbC（本文）** | **仅编码器** | **一般线性（GL）** | **否** | **是（线性不确定性）** |

## 启发与关联

- 最小二乘估计群矩阵的方法在已知配对数据的场景下非常实用，可广泛应用于物理仿真、机器人学等领域
- "对比学习 + 群结构"的组合为自监督表示学习提供了新的正则化手段
- 超参数选择可通过 Acc(G) 的峰值来确定嵌入维度，这一策略对任何需要选择潜空间维度的方法都有参考价值
- 与非线性ICA可辨识性理论的联系为等变表示学习提供了坚实的理论支撑

## 评分

- ⭐ 新颖性: 9/10 — 从对比学习中涌现等变性和群表示的思路新颖，理论贡献突出
- ⭐ 实验充分度: 7/10 — 合成数据验证充分，但视觉数据仅限 idSprites，缺乏更多真实场景
- ⭐ 写作质量: 8/10 — 理论推导清晰，符号一致，图示直观
- ⭐ 价值: 8/10 — 为等变表示学习提供了新范式，理论意义大于当前实际应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Identifiable Equivariant Networks are Layerwise Equivariant](../../ICML2026/others/identifiable_equivariant_networks_are_layerwise_equivariant.md)
- [\[NeurIPS 2025\] Faithful Group Shapley Value](faithful_group_shapley_value.md)
- [\[NeurIPS 2025\] On Universality Classes of Equivariant Networks](on_universality_classes_of_equivariant_networks.md)
- [\[NeurIPS 2025\] Finite-Time Analysis of Stochastic Nonconvex Nonsmooth Optimization on the Riemannian Manifolds](finite-time_analysis_of_stochastic_nonconvex_nonsmooth_optimization_on_the_riema.md)
- [\[NeurIPS 2025\] Learning (Approximately) Equivariant Networks via Constrained Optimization](learning_approximately_equivariant_networks_via_constrained_optimization.md)

</div>

<!-- RELATED:END -->
