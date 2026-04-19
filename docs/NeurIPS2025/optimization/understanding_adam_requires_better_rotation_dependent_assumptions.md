---
title: >-
  [论文解读] Understanding Adam Requires Better Rotation Dependent Assumptions
description: >-
  [NeurIPS 2025][优化][Adam优化器] 本文通过系统的实验研究揭示了 Adam 优化器对参数空间坐标基底的强依赖性，证明现有旋转不变的理论假设不足以解释 Adam 的优越性，并发现层更新的正交性是预测 Adam 在不同基底下性能的有力指标。
tags:
  - NeurIPS 2025
  - 优化
  - Adam优化器
  - 旋转等变性
  - 参数空间旋转
  - 正交性
  - 自适应优化
---

# Understanding Adam Requires Better Rotation Dependent Assumptions

**会议**: NeurIPS 2025  
**arXiv**: [2410.19964](https://arxiv.org/abs/2410.19964)  
**代码**: 暂无  
**领域**: 优化  
**关键词**: Adam优化器, 旋转等变性, 参数空间旋转, 正交性, 自适应优化

## 一句话总结

本文通过系统的实验研究揭示了 Adam 优化器对参数空间坐标基底的强依赖性，证明现有旋转不变的理论假设不足以解释 Adam 的优越性，并发现层更新的正交性是预测 Adam 在不同基底下性能的有力指标。

## 研究背景与动机

尽管 Adam 是训练 Transformer 模型的标准优化器，但其相对于 SGD 的优势缺乏全面的理论解释。SGD 具有旋转等变性——如果参数空间被旋转，SGD 的优化轨迹也会相应旋转；而 Adam 由于其逐元素除法操作，不具备这一性质。

现有文献中用于分析 Adam 的理论假设大多是**旋转不变**的（如有界梯度方差、Lipschitz 光滑性等），这意味着这些分析对任何基底给出相同的收敛保证，无法解释 Adam 为什么在标准基底下表现更好。作者提出了一个核心问题：**参数空间的基底选择到底如何影响 Adam 的性能？** 以及 **现有的旋转依赖假设能否解释这些影响？**

此外，近期 SOAP 和 Muon 等优化器通过在旋转后的参数空间中优化取得了实际提升，但这些旋转的设计更多依赖直觉而非系统理论，进一步凸显了理解旋转与 Adam 关系的必要性。

## 方法详解

### 整体框架

本文的方法论核心不是提出新算法，而是设计了一套系统的实验框架来研究 Adam 在旋转参数空间中的行为。具体来说，训练时仍在标准基底中进行前向和反向传播，但将梯度旋转到目标空间后让 Adam 更新，再将更新旋转回标准空间应用于参数。

### 关键设计

1. **多尺度旋转范围研究**: 作者定义了四种不同范围的随机旋转——全局旋转（整个参数空间）、层级旋转（每层子空间内）、输入端旋转（同一输入神经元的权重）和输出端旋转（同一输出神经元的权重）。实验发现旋转范围越广，Adam 性能下降越严重。对 GPT-2(124M)，全局旋转导致 16% 的训练减速；对 ViT/S(22M)，减速高达 96%。而输出端旋转几乎不影响甚至略微提升性能，说明 Adam 在输出神经元内的自适应性较弱。

2. **SVD 结构化旋转**: 受 GaLore 启发，作者将每层梯度矩阵 $\mathbf{G}$ 做 SVD 分解 $\mathbf{G} = \mathbf{U}\mathbf{S}\mathbf{V}^\top$，并在 $\mathbf{U}^\top \mathbf{G} \mathbf{V}$ 旋转空间中运行 Adam。结果表明，这种结构化旋转不仅不损害性能，反而**显著提升**了 GPT-2 的训练效果，证明存在比标准基底更优的坐标系。

3. **层更新正交性指标**: 为了找到一个能预测 Adam 性能的旋转依赖量，作者测量了每层权重更新 $\mathbf{A} = \mathbf{R}^\top \mathbf{M}_t^{(\mathbf{R})} / (\sqrt{\mathbf{V}_t^{(\mathbf{R})}} + \epsilon)$ 的奇异值变异系数（CV）。CV 越小意味着更新越接近缩放正交矩阵。实验发现 CV 与 Adam 性能高度一致：SVD 旋转 → 最低 CV → 最佳性能；全局随机旋转 → 最高 CV → 最差性能。

### 现有旋转依赖假设的检验

作者系统检验了三类文献中使用的旋转依赖假设：

- **$L_\infty$ 梯度有界**: 在全局旋转下 $\tilde{C}$ 显著减小（预示更好性能），但 Adam 性能实际下降，说明这一假设方向**相反**。
- **Hessian 块对角性**: 虽然标准基底下 Hessian 确实呈近似块对角结构，但量化分析发现对角块外的元素虽然数值小，却因维度更高在梯度变化中起**主导作用**，严格块对角近似是过度简化。
- **$(1,1)$-范数 / $L_\infty$ 光滑性**: 在全局和 SVD 旋转下与性能相关，但在输出端旋转下失效（性能略有提升但范数下降），说明不完全可靠。

## 实验关键数据

### 主实验

| 模型 | 旋转类型 | 训练减速 | 性能变化 |
|------|---------|---------|---------|
| GPT-2 (124M) | 全局旋转 | ~16% | 显著下降 |
| GPT-2 (124M) | 层级旋转 | ~8% | 中等下降 |
| GPT-2 (124M) | 输出端旋转 | ~0% | 略有提升 |
| GPT-2 (124M) | SVD旋转 | — | 显著提升 |
| ViT/S (22M) | 全局旋转 | ~96% | 严重下降 |
| ViT/S (22M) | 输出端旋转 | ~0% | 无变化 |
| ResNet-50 | 全局旋转 | 极小 | 几乎无影响 |

### 消融实验

| 度量指标 | 全局旋转 | SVD旋转 | 输出端旋转 | 与Adam性能一致性 |
|---------|---------|---------|-----------|----------------|
| $L_\infty$ 梯度界 | 减小(↓) | — | — | ❌ 不一致 |
| $(1,1)$-范数 | 增大(↑) | 减小(↓) | 增大(↑) | ⚠️ 部分一致 |
| 更新正交性 CV | 高(差) | 低(好) | 中(正常) | ✅ 完全一致 |

### 关键发现

- ResNet 对旋转不敏感，这可能解释了 SGD 在 ResNet 上并不逊色于 Adam 的现象。
- 全局旋转使 Adam 的二阶矩分布更集中，意味着有效学习率差异减小、自适应性减弱。
- SVD 旋转每 250 步更新一次就能带来可观提升，且在更新时刻 CV 出现明显下降。
- 不同参数block类型对旋转的敏感度不同，Appendix C 进一步分析了 Transformer 中 K/Q/V 投影层的独立影响。

## 亮点与洞察

- **核心发现极具启发性**: 将 Adam 性能与基底选择的关系系统化，揭示了一个被忽视的重要维度。
- **正交性指标与 Muon 优化器呼应**: Muon 通过近似正交化梯度更新取得优异性能，本文从 Adam 的角度提供了额外的支持证据。
- **实验设计精巧**: 四种不同尺度的旋转形成了从局部到全局的连续谱，系统性很强。

## 局限与展望

- 本文主要是实验性工作，缺乏为什么正交性更新更好的**严格理论分析**。
- SVD 旋转的目的是提供洞察而非实用优化器，其计算开销限制了直接应用。
- 未能解释 Adam 在 SVD 旋转下为何产生更正交的更新。
- ResNet 上旋转敏感性低的原因需要更多证据确认。

## 相关工作与启发

- 与 Muon 和 SOAP 优化器密切相关：它们通过旋转提升性能，本文提供了理论理解的基础。Muon 通过 Nesterov 迭代近似正交化梯度矩阵，SOAP 在 Adam 上应用 Shampoo 风格旋转。
- 为设计新的旋转感知优化器指明了方向：好的基底应使层更新更正交。
- 挑战了块对角 Hessian 假设，对后续理论工作有重要提示。
- Bernstein (2025) 从线性层角度论证正交化更新可以控制特征缩放尺度，减少对归一化层的依赖。
- 与 GaLore 的联系：GaLore 用低秩 SVD 压缩优化器状态，本文的全秩 SVD 旋转可看作其自然推广。
- 为 Adam 的收敛分析指明了新方向——需要旋转依赖的理论框架而非传统旋转不变假设。

## 评分

- 新颖性: ⭐⭐⭐⭐ 系统研究Adam与坐标基底的关系，角度新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型、多旋转类型、多假设检验，实验非常全面
- 写作质量: ⭐⭐⭐⭐ 叙述清晰，图表丰富，结论有说服力
- 价值: ⭐⭐⭐⭐ 为理解和改进自适应优化器提供重要实验基础

<!-- RELATED:START -->

## 相关论文

- [Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [Exploring Landscapes for Better Minima along Valleys](exploring_landscapes_for_better_minima_along_valleys.md)
- [In Search of Adam's Secret Sauce](in_search_of_adams_secret_sauce.md)
- [Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry](projecting_assumptions_the_duality_between_sparse_autoencoders_and_concept_geome.md)
- [The Rich and the Simple: On the Implicit Bias of Adam and SGD](the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)

<!-- RELATED:END -->
