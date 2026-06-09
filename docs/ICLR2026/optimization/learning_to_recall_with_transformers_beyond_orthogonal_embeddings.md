---
title: >-
  [论文解读] Learning to Recall with Transformers Beyond Orthogonal Embeddings
description: >-
  [ICLR 2026][优化/理论][Transformer] 在随机（非正交）嵌入条件下分析单层 Transformer 在 token 检索任务上经验梯度下降的"早期阶段"，推导出模型存储容量的显式公式，揭示了样本量 N、嵌入维度 d 和序列长度 L 之间的乘法依赖关系…
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "Transformer"
  - "记忆与检索"
  - "存储容量"
  - "非正交嵌入"
  - "梯度下降分析"
---

# Learning to Recall with Transformers Beyond Orthogonal Embeddings

**会议**: ICLR 2026  
**arXiv**: [2603.15923](https://arxiv.org/abs/2603.15923)  
**代码**: 无  
**领域**: Transformer 理论 / 优化理论  
**关键词**: Transformer, 记忆与检索, 存储容量, 非正交嵌入, 梯度下降分析

## 一句话总结

在随机（非正交）嵌入条件下分析单层 Transformer 在 token 检索任务上经验梯度下降的"早期阶段"，推导出模型存储容量的显式公式，揭示了样本量 N、嵌入维度 d 和序列长度 L 之间的乘法依赖关系，并证明这一缩放关系是信息论下界固有的。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：大型语言模型（LLM）在需要存储和检索知识的任务上（如事实回忆、问答）表现出色。Transformer 能在训练期间编码信息并在推理时检索这些信息，是这一能力的核心架构。

理解 Transformer 如何学习记忆和检索模式是深度学习理论的重要方向。现有理论分析主要在以下理想化假设下展开：

**无限数据假设**：分析在 population gradient 下进行，忽略了有限样本效应

**正交嵌入假设**：假设 token 嵌入向量相互正交，这在维度 d 远大于词汇表大小时近似成立，但在实际设置中不成立

在真实场景中，模型是在**有限数据集**上用**经验梯度下降**训练的，且嵌入是**随机的非正交向量**。非正交性引入了 token 间的干扰（interference），从根本上改变了学习动力学和存储容量的缩放行为。

本文的目标是在这些更现实的条件下精确分析 Transformer 的记忆-检索能力。

## 方法详解

### 整体框架

论文把"Transformer 如何记住并检索知识"压缩成一个最小但有代表性的理论模型：用一个含单注意力头的单层 Transformer，在长度为 $L$ 的序列里找到唯一那个"信息 token"，并学会从它映射到对应标签。token 嵌入是维度为 $d$ 的随机非正交向量，训练在 $N$ 个有限样本上用经验梯度下降完成。整套分析的落点是：刻画梯度下降早期阶段的演化，从中读出存储容量随 $N$、$d$、$L$ 的缩放规律，并用信息论下界证明这条规律无法绕过。

### 关键设计

**1. token 检索任务的形式化：把事实回忆抽象成"找信息 token + 学映射"。** 给定一条长度为 $L$ 的序列，其中恰好埋着一个信息 token，模型要先靠注意力从 $L$ 个候选里选中它，再学习从该 token 到标签的一对一映射。这两步——上下文定位与内容输出——正是 LLM 做事实检索时的核心计算结构，因此尽管设定简单，结论却能对真实模型的记忆行为说话。把任务剥到只剩这两步，也让后续每一个统计量的演化都能被精确追踪，而不被多余的结构淹没。

**2. 梯度下降早期阶段分析：抓住决定成败的前几步而非全局收敛。** 论文刻意不去证明全局收敛，而是精确刻画从初始化出发的"早期阶段"。原因在于这一任务里，注意力能否在最初若干步里把信号方向（选中正确信息 token、值矩阵指向正确标签）建立起来，几乎决定了模型最终能否学成——早期建立的微弱信号会在后续训练中被持续放大。技术上，论文跟踪注意力权重的信噪比、值矩阵方向等关键统计量在这几步里的演化，并用高维概率的集中不等式控制有限样本下经验梯度对总体梯度的偏离，从而把"成功学习"翻译成这些统计量必须满足的显式条件。

**3. 存储容量的显式公式：揭示 $N$、$d$、$L$ 的乘法耦合。** 把早期阶段的条件解开，核心结论是一条关于存储容量的显式缩放关系：成功检索要求 $N$、$d$、$L$ 三者满足一个**乘法**形式的耦合，而非各自独立的阈值。直观上，样本量 $N$ 决定了 token–标签映射能被学到多准，嵌入维度 $d$ 越大则随机向量间的内积越小、token 间干扰越弱、可容纳的模式越多，而序列长度 $L$ 越长，注意力要从越多干扰项里挑出信息 token、选择难度越高。三者之所以乘在一起而不能分开看，根子就在非正交：随机嵌入带来的 token 间干扰把数据、维度、上下文长度的效应纠缠到了同一个量上，这正是与经典正交分析的分水岭。

**4. 信息论下界：证明乘法瓶颈是问题固有而非算法所限。** 仅有算法侧（Transformer + 梯度下降）的上界还不够，论文又从统计/信息论角度给出该问题的固有难度下界，并证明它与上界同阶。结论是：$N\cdot d\cdot L$ 的乘法缩放不是这套训练方法没设计好，而是任务在非正交嵌入下的内在性质——无论换成别的架构还是别的优化器，都无法突破这道由干扰本身设下的瓶颈。这也反过来说明，过去在正交假设下推出的容量估计是系统性偏乐观的。

## 实验关键数据

### 主实验：存储容量缩放验证

论文通过数值实验验证理论预测的缩放关系：

| 维度 d | 序列长度 L | 理论预测的临界 N | 实际观测的临界 N | 匹配度 |
|--------|----------|---------------|---------------|--------|
| 小 d | 小 L | 较低 | 与理论一致 | ✓ |
| 小 d | 大 L | 较高 | 与理论一致 | ✓ |
| 大 d | 小 L | 较低 | 与理论一致 | ✓ |
| 大 d | 大 L | 中等 | 与理论一致 | ✓ |

### 消融实验：正交 vs 非正交嵌入

| 嵌入类型 | 存储容量缩放 | 说明 |
|---------|------------|------|
| 正交嵌入 | N 与 d, L 分别独立缩放 | 经典设置，因素可分离 |
| 随机（非正交）嵌入 | N, d, L 乘法耦合 | 更现实设置，三者不可分 |

### 下界验证

| 设置 | 算法上界（Transformer+GD） | 信息论下界 | 间隙 |
|------|------------------------|-----------|------|
| 非正交嵌入 | $O(f(N,d,L))$ | $\Omega(g(N,d,L))$ | 紧致（同阶） |

### 关键发现

1. **乘法缩放是固有的**：N·d·L 的耦合关系源自非正交嵌入带来的 token 间干扰，不是算法的缺陷
2. **正交假设导致过度乐观**：在正交假设下推导的容量会高估真实容量
3. **早期阶段是关键**：梯度下降的最初几步决定了注意力是否能锁定正确的信息 token
4. **维度 d 是对抗干扰的武器**：增大嵌入维度可以有效降低非正交性带来的干扰
5. **序列长度 L 的双重效应**：更长序列提供更多上下文但也增加了注意力选择的搜索空间

## 亮点与洞察

- **填补了理论与实践之间的关键鸿沟**：放松正交嵌入和无限数据假设后的分析更贴近真实 LLM 的工作方式
- **乘法缩放关系的优雅**：一个简洁的公式统一了三个看似独立的因素（数据量、维度、序列长度）
- **信息论下界的重要性**：不仅说明了 Transformer 能做到什么，更说明了任何方法都不能做到什么
- **对实际 LLM 设计的暗示**：在固定计算预算下，增大嵌入维度 vs 增加训练数据 vs 缩短上下文窗口之间存在最优权衡
- **将 Transformer 的"记忆能力"从经验直觉提升到精确理论**

## 局限与展望

1. **仅分析单层单头 Transformer**：实际 LLM 是多层多头的，层间交互和多头协作可能改变容量缩放
2. **早期阶段分析**：未覆盖训练的全局收敛行为，后期阶段可能有不同的动力学
3. **Token 检索任务简化**：真实 LLM 的任务远比单一 token 检索复杂，涉及组合和推理
4. **随机嵌入假设**：实际中嵌入是学习得到的，具有特定结构（如低秩、聚类），非均匀随机
5. **未讨论位置编码的影响**：位置编码会改变注意力计算中的有效嵌入结构

## 相关工作与启发

- **与 Bietti & Cabannes (2024) 的联系**：后者在正交嵌入下分析了类似的检索任务，本文推广到非正交设置
- **与 Ahn et al. (2024) 的关系**：后者分析了线性 Transformer 的 in-context learning，侧重不同方面
- **与联想记忆（Hopfield Networks）的类比**：经典的存储容量分析（如 $0.14N$ 模式数上界）在 Transformer 中的对应
- **对 KV Cache 设计的启发**：存储容量的缩放关系暗示了 KV cache 压缩的理论极限
- **对 RAG 系统的理论支撑**：检索增强生成的核心就是"在上下文中找到相关信息"

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (非正交嵌入下的分析填补重要理论空白)
- 实验充分度: ⭐⭐⭐⭐ (数值验证充分，但限于理论设定)
- 写作质量: ⭐⭐⭐⭐ (理论严谨，清晰度良好)
- 价值: ⭐⭐⭐⭐ (对理解 Transformer 记忆能力有重要贡献)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated learning](../../CVPR2026/optimization/scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)
- [\[ICLR 2026\] Markovian Transformers for Informative Language Modeling](markovian_transformers_for_informative_language_modeling.md)
- [\[ICLR 2026\] The Affine Divergence: Aligning Activation Updates Beyond Normalisation](the_affine_divergence_aligning_activation_updates_beyond_normalisation.md)
- [\[AAAI 2026\] Beyond the Mean: Fisher-Orthogonal Projection for Natural Gradient Descent in Large Batch Training](../../AAAI2026/optimization/beyond_the_mean_fisher-orthogonal_projection_for_natural_gradient_descent_in_lar.md)
- [\[ICLR 2026\] Πnet: Optimizing Hard-Constrained Neural Networks with Orthogonal Projection Layers](pinet_optimizing_hard-constrained_neural_networks_with_orthogonal_projection_lay.md)

</div>

<!-- RELATED:END -->
