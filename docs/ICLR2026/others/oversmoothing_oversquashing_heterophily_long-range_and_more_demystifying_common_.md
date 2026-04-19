---
title: >-
  [论文解读] Oversmoothing, Oversquashing, Heterophily, Long-Range, and More: Demystifying Common Beliefs in Graph Machine Learning
description: >-
   本文系统梳理了图机器学习领域围绕 oversmoothing、oversquashing、同质/异质性和长程依赖的九个常见误区，通过简洁反例逐一反驳，将"oversquashing"拆分为**计算瓶颈**和**拓扑瓶颈**两个独立概念，厘清了领域中广泛存在的概念混淆。
tags:

---

# Oversmoothing, Oversquashing, Heterophily, Long-Range, and More: Demystifying Common Beliefs in Graph Machine Learning

- **会议**: ICLR2026
- **arXiv**: [2505.15547](https://arxiv.org/abs/2505.15547)
- **代码**: [GitHub](https://github.com/AdrianArnaiz/demystifyingGraphML)
- **领域**: 图神经网络 / 图机器学习理论
- **关键词**: oversmoothing, oversquashing, heterophily, long-range dependencies, message-passing, GNN

## 一句话总结

本文系统梳理了图机器学习领域围绕 oversmoothing、oversquashing、同质/异质性和长程依赖的九个常见误区，通过简洁反例逐一反驳，将"oversquashing"拆分为**计算瓶颈**和**拓扑瓶颈**两个独立概念，厘清了领域中广泛存在的概念混淆。

## 研究背景与动机

**领域现状**: 图上消息传递（message-passing）深度学习经历快速发展，研究者关注其固有局限——oversmoothing（OSM，节点表征趋同）、oversquashing（OSQ，信息压缩损失）、同质/异质性对分类的影响、以及长程信息传播等议题。

**痛点**: 由于领域进展速度过快，大量"常被接受的信念"（common beliefs）在未经充分验证的情况下被广泛传播和引用，造成研究者概念混淆、问题定义模糊，严重阻碍了有针对性的进一步研究。

**核心矛盾**: 文献中将 oversmoothing 视为性能下降的根本原因、将 oversquashing 与拓扑瓶颈直接等同、将异质性等同于"困难"等论断，并非普遍成立，却几乎成为公认结论。

**目标**: 明确指出这些信念的局限性，通过简明反例予以反驳，让研究者能够区分并精准定义待解决的问题。

**切入角度**: 不批驳具体工作，而是将文献中散布的含混论断归纳为九条 common beliefs，逐条用数学定义+反例的方式"去神秘化"。

**核心 idea**: "Oversquashing"应当被拆解为**计算瓶颈**（computational bottleneck，源于计算树的指数膨胀）和**拓扑瓶颈**（topological bottleneck，源于图连通性）两个独立问题，它们可独立存在也可互不相关。

## 方法详解

### 整体框架

本文是一篇**综述+分析+反驳**型工作，围绕三大主题（OSM、同质/异质性、OSQ）的九个 belief 展开：

| 主题 | Beliefs |
|------|---------|
| OSM | 1. OSM 导致性能下降；2. OSM 是所有 DGN 的固有属性 |
| 同质/异质 | 3. 同质好、异质差；4. 长程传播在异质图上评估；5. 不同类别意味着不同特征 |
| OSQ | 6. OSQ=拓扑瓶颈；7. OSQ=计算瓶颈；8. OSQ 对长程任务有害；9. 拓扑瓶颈与长程问题关联 |

### 关键设计

#### 1. Oversmoothing 并非普遍存在也非性能下降根因

**功能**: 证明 OSM 是否发生、如何度量都高度依赖于架构与超参的选择。

**核心思路**: 用 Dirichlet Energy（DE）和 Rayleigh Quotient（RQ）两种指标度量 OSM：

$$\mathrm{DE}(\mathbf{H}^\ell) = \mathrm{Tr}((\mathbf{H}^\ell)^T \mathbf{L} \mathbf{H}^\ell), \quad \mathrm{RQ} = \frac{\mathrm{Tr}((\mathbf{H}^\ell)^T \mathbf{L} \mathbf{H}^\ell)}{\|\mathbf{H}^\ell\|_F^2}$$

实验表明：(i) GIN 的 DE 在标准设置下**爆炸**而非坍缩；(ii) 将权重矩阵乘以 2（$AX(2W)$）即可逆转 DE 的趋势；(iii) DE 和 RQ 在同一模型上经常给出矛盾结论。因此 OSM 既不普遍、也不唯一定义。

**设计动机**: 即使 DE 下降，节点嵌入的**类别可分性**可能保持甚至提升——不同类先各自坍缩到不同点（"有益平滑"阶段），真正的性能下降更多由梯度消失和过拟合导致。

#### 2. 将 Oversquashing 拆分为计算瓶颈与拓扑瓶颈

**功能**: 给出计算瓶颈的严格定义，证明其与拓扑瓶颈可独立存在。

**核心思路**: 定义**计算瓶颈**为计算树的多重集大小 $|\mathcal{M}_v^K|$：

$$\mathcal{M}_v^K := \mathcal{M}_v^{K-1} \uplus \left\{\biguplus_{u \in \mathcal{M}_v^{K-1}} \mathcal{N}_u\right\}$$

它随层数指数增长，独立于图是否存在拓扑瓶颈。反例：(a) 网格图无拓扑瓶颈但有严重计算瓶颈；(b) 存在拓扑瓶颈的哑铃图在少量层时计算瓶颈很轻。

**设计动机**: 现有 rewiring 方法改善了拓扑瓶颈，但加边加节点反而会**恶化**计算瓶颈；message filtering 方法反之可以在不修改图结构的情况下减小计算瓶颈。

#### 3. 同质/异质性与任务的解耦

**功能**: 证明高异质不等于"难"、长程传播任务不必然出现在异质图上。

**核心思路**: (a) 完全异质二部图中，节点度数不同足以让 1 层 sum-based DGN 完美分类；(b) 高度同质图上，若任务是判断节点到特定节点距离是否>5，则需要长程传播——但该图是同质的。因此 **长程任务 ⊥ 异质性**。

### 损失函数

本文为分析型工作，不提出新模型。实验中复现了标准的 DGN 训练，使用交叉熵损失进行节点分类。

## 实验关键数据

### 主实验：OSM 指标在不同架构下的表现（Cora, 50 seeds）

| 架构 | W: DE 趋势 | 2W: DE 趋势 | W: RQ 趋势 | 2W: RQ 趋势 |
|------|-----------|-------------|-----------|-------------|
| GCN | 坍缩→0 | 爆炸↑ | 平稳 | 平稳 |
| GAT | 坍缩→0 | 爆炸↑ | 线性衰减 | 稳定 |
| SAGE | 坍缩→0 | 爆炸↑ | 线性衰减 | 部分稳定 |
| GIN | 爆炸↑ | 爆炸↑↑ | 线性衰减 | 部分稳定 |

> 结论：同一模型在两个指标下可表现出完全矛盾的趋势。OSM 的观测高度依赖度量选择和超参。

### 消融实验：有/无偏置下性能退化与 DE 对比

| 配置 | 深度增加时 DE | 深度增加时准确率 |
|------|------------|--------------|
| GCN (有 bias) | 坍缩 | 下降 |
| GCN (无 bias) | **不坍缩** | 同样下降 |

> 说明性能下降不能归因于 OSM——去除 bias 消除了 DE 坍缩但准确率仍下降，真正原因是梯度消失/过拟合。

### 关键发现

1. **OSM 非普遍现象**: 换聚合函数（GIN）、微调权重缩放（2W）或换度量指标（DE vs RQ）都会改变结论
2. **计算瓶颈 ≠ 拓扑瓶颈**: 网格图无拓扑瓶颈但对角节点间 effective resistance 线性增长，仍有严重计算瓶颈；graph rewiring 改善拓扑但可能恶化计算瓶颈
3. **异质性 ≠ 困难性**: 完全异质图存在 1 层 DGN 完美分类的情况；高同质图也可能需要长程传播

## 亮点与洞察

- 首次将"oversquashing"**显式拆分**为计算瓶颈与拓扑瓶颈两个独立概念，具有高度启发性
- 反例简洁且形式化，如二部图完美分类、网格图无拓扑瓶颈但计算瓶颈严重，易于记忆和传播
- 九条 belief 的梳理方式像"debug 清单"，可作为图 ML 研究者的参考手册
- 呼吁社区在论述中更精确地使用术语、避免过度泛化

## 局限性

- 主要是概念澄清和反例驱动，未提出新的解决方案或模型
- 大部分反例基于人工构造的小图，实际大规模图上的情况更复杂
- 面向图 ML 社区的"元研究"，对其他领域的直接实用价值有限
- 部分 belief 的"反驳"本质是"不总是成立"，而非"从不成立"——边界条件仍需深入探讨

## 相关工作与启发

- **Oversmoothing**: Cai & Wang (2020) 提出 Dirichlet Energy 度量；Roth & Liebig (2024) 分析 Rayleigh Quotient；Zhang et al. (2025) 指出未训练网络的 OSM 不反映训练后行为
- **Oversquashing**: Topping et al. (2022) 连接到曲率和 Jacobian 灵敏度；Errica et al. (2025) 提出 message filtering 减少计算瓶颈
- **Heterophily**: Ma et al. (2022) 提出新指标关联异质性与可区分度；Platonov et al. (2023) 构建更严格的异质性基准
- **启发**: 未来工作应明确区分度量对象（DE vs RQ vs 可分性）、问题类型（计算 vs 拓扑）、以及任务与图属性的关系

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**: ⭐⭐⭐⭐ — 将分散的困惑系统化为九条 belief 并给出形式化反驳，conceptual contribution 很强
- **实用性**: ⭐⭐⭐ — 概念澄清价值大但无新方法
- **写作**: ⭐⭐⭐⭐⭐ — 逻辑清晰，反例精炼，表格总结一目了然
- **影响力**: ⭐⭐⭐⭐ — 对图 ML 社区的术语使用和问题定义具有长期影响

<!-- RELATED:START -->

## 相关论文

- [Learning Structure-Semantic Evolution Trajectories for Graph Domain Adaptation](learning_structure-semantic_evolution_trajectories_for_graph_domain_adaptation.md)
- [Directional Non-Commutative Monoidal Structures for Compositional Embeddings in Machine Learning](../../NeurIPS2025/others/directional_non-commutative_monoidal_structures_for_compositional_embeddings_in_.md)
- [Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation](learning_adaptive_distribution_alignment_with_neural_characteristic_function_for.md)
- [Compositional Diffusion with Guided Search for Long-Horizon Planning](compositional_diffusion_long_horizon_planning.md)
- [Revisiting Sharpness-Aware Minimization: A More Faithful and Effective Implementation](revisiting_sharpness-aware_minimization_a_more_faithful_and_effective_implementa.md)

<!-- RELATED:END -->
