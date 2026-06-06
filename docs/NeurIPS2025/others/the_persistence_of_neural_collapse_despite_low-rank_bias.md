---
title: >-
  [论文解读] The Persistence of Neural Collapse Despite Low-Rank Bias
description: >-
  [NeurIPS 2025][神经坍缩] 本文从理论上证明了深度神经坍缩（DNC）在深层无约束特征模型中由于 L2 正则化引起的低秩偏差而全局次优，同时首次解释了 DNC 在实践中持续出现的原因——其解空间维度随网络宽度增长快于低秩解。
tags:
  - "NeurIPS 2025"
  - "神经坍缩"
  - "低秩偏差"
  - "深度无约束特征模型"
  - "损失曲面"
  - "Schatten准范数"
---

# The Persistence of Neural Collapse Despite Low-Rank Bias

**会议**: NeurIPS 2025  
**arXiv**: [2410.23169](https://arxiv.org/abs/2410.23169)  
**代码**: 无  
**领域**: 深度学习理论  
**关键词**: 神经坍缩, 低秩偏差, 深度无约束特征模型, 损失曲面, Schatten准范数

## 一句话总结

本文从理论上证明了深度神经坍缩（DNC）在深层无约束特征模型中由于 L2 正则化引起的低秩偏差而全局次优，同时首次解释了 DNC 在实践中持续出现的原因——其解空间维度随网络宽度增长快于低秩解。

## 研究背景与动机

神经坍缩（Neural Collapse, NC）是深度神经网络分类训练后期发现的结构化几何现象：最后一层特征坍缩到类均值，类均值形成单纯形等角紧框架（simplex ETF），权重与特征对齐。NC 不仅出现在最后一层，在早期层也有类似结构，称为深度神经坍缩（DNC）。

**已有理论结果**：在单层无约束特征模型（UFM）中，NC 已被证明是全局最优解，且损失曲面是严格鞍函数（只有全局最优和非退化鞍点）。

**核心矛盾**：Sukenik et al. (2024) 证明了在 ReLU + MSE 损失的深层 UFM 中，DNC 并非全局最优，因为 L2 正则化诱导的低秩偏差使得更低秩的解可以获得更低的损失。但他们没有分析 DNC 或低秩解是否为局部最优，也没有解释为什么 DNC 虽然次优但在实践中频繁出现。

**本文的切入角度**：使用交叉熵（CE）损失 + 线性层的深层 UFM 进行系统分析。线性层便于分析，而 UFM 的无约束特征假设补偿了线性层缺乏的表达能力。本文的目标是：(1) 全面刻画低秩偏差如何影响损失曲面，(2) 首次解释 DNC 的经验持续性。

## 方法详解

### 整体框架

考虑 K 类分类，每类 n 个样本。深层 UFM 的损失函数为：
$$\mathcal{L}(H_1, W_1, ..., W_L) = g(Z) + \sum_{l=1}^{L} \frac{1}{2}\lambda \|W_l\|_F^2 + \frac{1}{2}\lambda \|H_1\|_F^2$$
其中 g(Z) 是交叉熵损失，Z 是 logit 矩阵。关键观察是正则化项等价于 Schatten 2/L 准范数：
$$\frac{1}{2}L\lambda \|X\|_{S_{2/L}}^{2/L} = \min \{\text{正则化项}\}$$
当 L 增大时，Schatten 2/L 准范数趋近于矩阵的秩，因此深层网络内在地偏好低秩解。

### 关键设计

1. **DNC 全局次优定理（Theorem 1）**：对于深层线性 UFM，若 K≥4 且 L≥3（或 K≥6 且 L=2），则没有 DNC 结构的解能成为全局最优。证明方法是构造一个具有块对角结构的低秩 logit 矩阵（每个 2×2 块为 [[1,-1],[-1,1]]），在等比例条件下比 DNC 解（秩 K）获得更低损失。这揭示了单层 UFM 的最优结构无法推广到深层模型。

2. **一般高秩次优定理（Theorem 2）**：对于任意固定结构，如果其秩高于拟合数据所需的最低秩，当层数 L 足够大时，该结构必然次优。关键概念是"对角优势矩阵"（diagonally superior matrix）：每个样本的正确类得分最高。这类矩阵可以通过适当缩放实现任意小的拟合损失。

3. **全局最优的低秩性质（Theorem 3）**：当正则化满足 $\lambda_L = o(L^{-1})$ 时，全局最优解 $Z_L^*$ 中至多 $q_K$（对角优势矩阵的最小秩，≤2）个奇异值不为零或不按指数速率衰减。这意味着最优解近似低秩，大多数奇异值随 L 指数衰减。DNC 的秩为 K-1，与最优秩 ≤2 之间存在巨大差距。

4. **DNC 持续性的解释（Theorems 4-5）**：

    - **Theorem 4**：当正则化 λ 足够小时，DNC 解是临界点且 Hessian 无负特征值（局部最小或退化鞍点）。这与单层 UFM 的严格鞍性质形成对比。
    - **Theorem 5**：DNC 解的参数空间维度 $D_{DNC}$ 与低秩解的参数空间维度 $D_{Z^*}$ 之比 R(d) 是关于宽度 d 的单调递增函数，从 <1 趋向 (K-1)/r > 1。当 d 增大时，DNC 在损失曲面中占据的"体积"越来越大，超过低秩解。

5. **ReLU 扩展（Theorem 7）**：在 ReLU UFM 中，当 K≥10 且 L≥5（或 K≥16 且 L=4）时，DNC 同样全局次优。证明通过证明线性模型的 DNC 损失是 ReLU 模型 DNC 损失的下界。

### 损失函数 / 训练策略

使用交叉熵损失 + L2 正则化（权重衰减），适用于所有参数（包括UFM框架下的特征矩阵 H₁）。理论分析关注正则化参数 λ 和层数 L 如何塑造损失曲面的全局和局部结构。

## 实验关键数据

### 主实验：深层线性 UFM

| 实验设置 | DNC 解损失 | 低秩解损失 | 观察 |
|---------|-----------|-----------|------|
| L=2, d=70, K=10, λ=2⁻¹⁰ | 较高 | 更低 | 低秩解确实优于 DNC（验证 Theorem 1） |
| 收敛后 logit 矩阵 | Simplex ETF | 块对角结构 | 两种不同的收敛结构 |

### 消融实验：宽度和正则化对 DNC 概率的影响

| 参数变化 | DNC 出现概率 | 说明 |
|---------|-------------|------|
| d 从小到大 | 从 ~0% 到 ~100% | 验证 Theorem 5：宽网络更容易 DNC |
| λ 从大到小 | 概率增加 | 小正则化下 DNC 和低秩解损失接近 |
| MNIST, L=3, 线性头 | 低秩解优于 DNC 界 | 4 个非零奇异值 |
| CIFAR-10, L=3, 线性头 | 低秩解优于 DNC 界 | 3 个非零奇异值 |
| CIFAR-10, 标准正则化 | 低秩结构出现 | logit 矩阵不形成单纯形 |

### 关键发现

- 即使在真实网络（ResNet-20 + 全连接头）上，低秩偏差也普遍存在
- 使用标准权重衰减（非 UFM 风格正则化）时，低秩结构仍然出现
- ReLU 激活下低秩偏差同样存在，线性模型有效捕获了关键현象
- 网络宽度是影响 DNC 出现概率的关键因素——宽网络 DNC 区域的"体积"指数增长
- Hessian 分析表明 DNC 是局部吸引子（正半定 Hessian），解释了梯度下降收敛到 DNC 的现象

## 亮点与洞察

- **首次完整解释 DNC 持续性**：不仅证明 DNC 次优，还解释了为什么它仍频繁出现——解空间维度差异
- **深层与单层的本质区别**：单层 UFM 是严格鞍函数，深层 UFM 出现退化鞍/局部最小→ 根本改变了优化景观的几何
- **低秩偏差的量化刻画**：不仅定性说明低秩更好，还精确刻画了奇异值衰减速率和对角优势矩阵的最小秩
- **理论与实验紧密结合**：每个理论结果都有对应的数值验证

## 局限与展望

- 模型聚焦理论分析，未探讨下游性能（泛化、鲁棒性）
- UFM 假设网络过参数化，在欠参数化场景下可能不适用
- 未考虑初始化规模、批大小、优化器等实际因素与低秩偏差的交互
- Theorem 5 仅给出维度比较的启发式论证，未严格证明维度优势→收敛概率优势
- 对不平衡数据集和超大类别数的分析还不够深入

## 相关工作与启发

本文在 NC 研究领域推进了几个重要理论前沿。(1) 将 Sukenik et al. 的 MSE + ReLU 结果扩展到 CE + 线性和 CE + ReLU 设定。(2) 首次分析了 DNC 在损失曲面上的局部性质（Hessian 分析）。(3) 将低秩偏差与矩阵补全文献中的 Schatten 准范数理论联系起来。对于基金会模型中宽度、深度、正则化等超参数对内部表示结构的影响提供了新的理论洞察。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次完整解释 DNC 次优但持续出现的矛盾现象
- 实验充分度: ⭐⭐⭐⭐ 理论结果验证充分，但实验规模较小
- 写作质量: ⭐⭐⭐⭐⭐ 理论展开层层递进，定理陈述精确
- 价值: ⭐⭐⭐⭐ 深化了对深度学习训练动态的理论理解

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CoLA: Collaborative Low-Rank Adaptation](../../ACL2025/others/cola_collaborative_low-rank_adaptation.md)
- [\[ACL 2025\] Low-Rank Interconnected Adaptation across Layers](../../ACL2025/others/low-rank_interconnected_adaptation_across_layers.md)
- [\[CVPR 2025\] Detecting Out-of-Distribution through the Lens of Neural Collapse](../../CVPR2025/others/detecting_out-of-distribution_through_the_lens_of_neural_collapse.md)
- [\[ICLR 2026\] Consistent Low-Rank Approximation](../../ICLR2026/others/consistent_low-rank_approximation.md)
- [\[NeurIPS 2025\] Neural Collapse in Cumulative Link Models for Ordinal Regression: An Analysis with Unconstrained Feature Model](neural_collapse_in_cumulative_link_models_for_ordinal_regression_an_analysis_wit.md)

</div>

<!-- RELATED:END -->
