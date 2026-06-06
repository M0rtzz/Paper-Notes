---
title: >-
  [论文解读] On Universality Classes of Equivariant Networks
description: >-
  [NeurIPS 2025][等变神经网络] 本文证明等变神经网络的分离能力（区分对称等价输入的能力）不足以完全刻画其表达能力——具有相同分离能力的模型可能拥有不同的逼近能力，并给出了浅层不变网络通用性类的完整刻画及失败的充分条件。
tags:
  - "NeurIPS 2025"
  - "等变神经网络"
  - "万有逼近"
  - "分离能力"
  - "对称群"
  - "通用性类"
---

# On Universality Classes of Equivariant Networks

**会议**: NeurIPS 2025  
**arXiv**: [2506.02293](https://arxiv.org/abs/2506.02293)  
**代码**: 暂无  
**领域**: 深度学习理论 / 等变神经网络  
**关键词**: 等变神经网络, 万有逼近, 分离能力, 对称群, 通用性类

## 一句话总结

本文证明等变神经网络的分离能力（区分对称等价输入的能力）不足以完全刻画其表达能力——具有相同分离能力的模型可能拥有不同的逼近能力，并给出了浅层不变网络通用性类的完整刻画及失败的充分条件。

## 研究背景与动机

等变神经网络通过将对称性编码进架构（如置换等变的 GNN、旋转等变的卷积网络）获得了显著的经验成功。目前关于其表达能力的研究主要集中在**分离能力**——即区分在群作用意义下不等价的输入的能力。在图学习中，这通常通过 Weisfeiler-Leman (WL) 测试层次来形式化。

然而，分离能力只回答了"能不能区分不同输入"的问题，却没有回答"能不能逼近所有尊重对称性的函数"（即**万有性/通用性**）。直觉上，能区分所有不同输入是逼近任意函数的必要条件，但不一定是充分条件。

作者提出了一个关键问题：**分离能力是否是比较等变网络表达能力的完备代理？** 答案是否定的。他们构造了明确的例子：PointNet 和宽度为 1 的 CNN 虽然具有**相同的分离能力**（都能分离所有置换不等价的输入），但它们的逼近能力严格不同——CNN(filter size 1) $\subsetneq$ PointNet $\subsetneq$ 正则表示网络。这意味着分离能力作为表达能力的代理是不完备的。

## 方法详解

### 整体框架

核心理论工具是将浅层不变网络表示为满足特定微分约束的函数，然后通过分析这些微分约束的结构来刻画通用性类。由于等变网络通过投影到平凡表示可化为不变网络（Remark 11），分析不变情形就足以在等变情形达到相应的结论。

### 关键设计

1. **通用性类的微分算子刻画（Theorem 13）**: 核心定理建立了浅层不变网络的通用性类与微分约束之间的等价关系。对于层空间 $M \subseteq \text{Aff}_G(V, W)$ 和 $N \subseteq \text{Aff}_G(W, \mathbb{R})$，定义基映射 $\phi_i: \mathbb{R}^X \to \mathbb{R}^m$（从等变层结构提取）。则连续函数 $f \in \mathcal{U}_\sigma(M, N)$ 当且仅当 $P(\partial_1, \ldots, \partial_d) f = 0$ 对所有在基映射行空间上消失的多项式 $P$ 成立。直觉上，网络层的等变结构通过限制信息流的方向，在函数空间中引入了不可避免的微分约束。

2. **通用性失败的充分条件（Theorem 14 & 15）**: 如果存在方向 $c_\alpha \in \ker(\phi_\alpha^\top)$ 使得

$$D_{c_1} \cdots D_{c_\ell} f \neq 0$$

   则 $f \notin \mathcal{U}_\sigma(M, N)$，即网络无法逼近 $f$。这个条件的含义是：如果函数 $f$ 在某些"网络看不到的方向"上具有非零混合偏导数，那么网络就无法逼近它。Theorem 15 进一步处理了 $n=3$ 的特殊情况。

3. **分离相同但逼近不同的具体构造（Proposition 16）**: 通过上述理论证明了三个严格包含关系：

$$\mathcal{U}_\sigma(C^1, I) \subsetneq \mathcal{U}_\sigma(\mathbb{R}^n, \mathbb{R}^n, \mathbb{R}) \subsetneq \mathcal{U}_\sigma(\mathbb{R}^n, \mathbb{R}^{S_n}, \mathbb{R})$$

   具体证明策略：
    - **CNN(width 1) 无法逼近** $(x_1 + \cdots + x_n)^n$：因为 $\partial_n \cdots \partial_1 (x_1 + \cdots + x_n)^n = n! \neq 0$
    - **PointNet 可以逼近** $(x_1 + \cdots + x_n)^n$：通过 $f(x_i, x_1+\cdots+x_n)$ 的求和形式
    - **PointNet 无法逼近** $x_1 \cdots x_n$：构造方向向量验证 Theorem 14 的条件
    - **正则表示网络可以逼近** $x_1 \cdots x_n$：由 Theorem 6 (Ravanbakhsh) 保证

4. **正群论条件下的通用性（Theorem 18）**: 如果隐藏表示来自群 $G$ 的**正规子群** $H$ 的陪集空间 $\mathbb{R}^{G/H}$，则对应的浅层网络是分离约束通用的。但对置换群 $S_n$（$n \geq 5$）来说，唯一非平凡正规子群是交替群 $A_n$，其陪集只有 2 个元素，隐藏表示太小而无用。阿贝尔群的所有子群都是正规的，因此循环 CNN 具有分离约束通用性。

### 损失函数 / 训练策略

本文是纯理论工作，不涉及训练。主要使用数学证明（Stone-Weierstrass 定理的变体、微分算子理论、群表示论）建立结果。

## 实验关键数据

### 理论结果汇总

| 架构 | 对称群 | 分离能力 | 逼近 $(x_1+\cdots+x_n)^n$ | 逼近 $x_1 \cdots x_n$ | 分离约束通用 |
|------|--------|----------|---------------------------|------------------------|-------------|
| CNN (filter=1) | $\mathbb{Z}_n$ / $S_n$ | 最大 | ✗ | ✗ | ✗ |
| PointNet (浅层) | $S_n$ | 最大 | ✓ | ✗ | ✗ |
| PointNet (深度3) | $S_n$ | 最大 | ✓ | ✓ | ✓ (Segol & Lipman) |
| 正则表示 $\mathbb{R}^{S_n}$ | $S_n$ | 最大 | ✓ | ✓ | ✓ (Ravanbakhsh) |
| 循环 CNN (任意宽度) | $\mathbb{Z}_n$ | 最大 | ✓ | - | ✓ (Theorem 18) |

### 影响通用性的结构因素

| 因素 | 对通用性的影响 | 说明 |
|------|--------------|------|
| 分离能力 | 必要非充分 | 相同分离 ≠ 相同逼近 |
| 深度 | 可补救浅层失败 | PointNet 深度3恢复通用性 |
| 隐藏表示类型 | 关键 | 正则表示 > PointNet表示 > CNN(1)表示 |
| 群结构（正规子群） | 决定浅层能否通用 | 阿贝尔群好，置换群差 |

### 关键发现

1. **分离 ≠ 通用**: 三类网络分离能力完全相同，但逼近能力严格分层，这是该领域的核心新发现
2. **深度的作用超越参数效率**: 在等变网络中，增加深度不仅是减少参数，更是从根本上扩展可逼近的函数类
3. **群结构决定浅层极限**: 对于置换群（最重要的对称群之一），浅层网络的通用性存在本质障碍
4. **微分约束视角**: 等变层的结构通过微分算子约束了可逼近函数的"光滑方向"，提供了全新的分析工具

## 亮点与洞察

- 颠覆了图学习社区将"分离能力"等同于"表达能力"的隐含假设
- 理论贡献深刻：Theorem 13 将神经网络逼近问题转化为微分算子零化问题，建立了代数/分析之间的桥梁
- Proposition 16 的证明策略（对不同 $n$ 值使用不同定理）展示了问题的微妙性
- 正规子群条件的出现揭示了群结构（不仅是群大小）对网络表达能力的深层影响

## 局限与展望

- 仅分析浅层网络（深度 ≤ 2），深层等变网络的通用性分析仍然开放
- 未涉及深度如何与分离交互（如 IGN 中深度增加可提升分离能力的情况）
- 缺乏实验验证理论预测（纯数学工作）
- 对不满足正规子群条件的情况（如 $S_n$），未给出正面的充分条件

## 相关工作与启发

- 推广了 Ravanbakhsh (2020) 的正则表示通用性结果到更广泛的表示族
- 与经典逼近论（Cybenko, Hornik 的万有逼近定理）在等变设定下的对应关系清晰
- 启发：设计等变网络时，不仅要关注能区分什么（分离），还要关注能逼近什么（通用性）——两者需要不同的架构选择

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统性地将通用性与分离在等变网络中解耦，核心发现改变了领域认知
- 实验充分度: ⭐⭐⭐ 纯理论工作无实验，但理论结果本身足够有力
- 写作质量: ⭐⭐⭐⭐ 数学符号密集但定义清晰，例子选择得当（PointNet, CNN 读者直觉友好）
- 价值: ⭐⭐⭐⭐⭐ 对等变深度学习理论的基础性贡献，将影响后续架构设计的理论分析范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning (Approximately) Equivariant Networks via Constrained Optimization](learning_approximately_equivariant_networks_via_constrained_optimization.md)
- [\[ICML 2026\] Identifiable Equivariant Networks are Layerwise Equivariant](../../ICML2026/others/identifiable_equivariant_networks_are_layerwise_equivariant.md)
- [\[ICML 2025\] Permutation Equivariant Neural Networks for Symmetric Tensors](../../ICML2025/others/permutation_equivariant_neural_networks_for_symmetric_tensors.md)
- [\[ICML 2025\] The Price of Freedom: Exploring Expressivity and Runtime Tradeoffs in Equivariant Networks](../../ICML2025/others/the_price_of_freedom_exploring_expressivity_and_runtime_tradeoffs_in_equivariant.md)
- [\[NeurIPS 2025\] Equivariance by Contrast: Identifiable Equivariant Embeddings from Unlabeled Finite Group Actions](equivariance_by_contrast_identifiable_equivariant_embeddings_from_unlabeled_fini.md)

</div>

<!-- RELATED:END -->
