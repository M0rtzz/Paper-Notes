---
title: >-
  [论文解读] Learning to Recall with Transformers Beyond Orthogonal Embeddings
description: >-
  [ICLR 2026][优化][Transformer] 在随机（非正交）嵌入条件下分析单层 Transformer 在 token 检索任务上经验梯度下降的"早期阶段"，推导出模型存储容量的显式公式，揭示了样本量 N、嵌入维度 d 和序列长度 L 之间的乘法依赖关系，并证明这一缩放关系是信息论下界固有的。
tags:
  - ICLR 2026
  - 优化
  - Transformer
  - 记忆与检索
  - 存储容量
  - 非正交嵌入
  - 梯度下降分析
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

大型语言模型（LLM）在需要存储和检索知识的任务上（如事实回忆、问答）表现出色。Transformer 能在训练期间编码信息并在推理时检索这些信息，是这一能力的核心架构。

理解 Transformer 如何学习记忆和检索模式是深度学习理论的重要方向。现有理论分析主要在以下理想化假设下展开：

**无限数据假设**：分析在 population gradient 下进行，忽略了有限样本效应

**正交嵌入假设**：假设 token 嵌入向量相互正交，这在维度 d 远大于词汇表大小时近似成立，但在实际设置中不成立

在真实场景中，模型是在**有限数据集**上用**经验梯度下降**训练的，且嵌入是**随机的非正交向量**。非正交性引入了 token 间的干扰（interference），从根本上改变了学习动力学和存储容量的缩放行为。

本文的目标是在这些更现实的条件下精确分析 Transformer 的记忆-检索能力。

## 方法详解

### 整体框架

论文设置了一个简洁但有代表性的理论模型：
- **架构**：单层 Transformer（含一个注意力头）
- **任务**：token 检索——在长度为 L 的序列中找到一个信息 token，并学习从 token 到标签的一对一映射
- **嵌入**：随机嵌入向量（非正交），维度为 d
- **训练**：有限样本 N 上的经验梯度下降

### 关键设计

1. **Token 检索任务的形式化**:

   给定长度为 L 的 token 序列，其中恰好有一个"信息 token"（informative token），模型需要：
   - 通过注意力机制识别并选择这个信息 token
   - 学习从该 token 到对应标签的映射

   这个任务捕捉了 LLM 中事实检索的核心计算结构：在上下文中找到相关信息并正确输出。

2. **梯度下降早期阶段分析**:

   论文不追求全局收敛结果，而是精确刻画梯度下降的"早期阶段"（early phase）——即从初始化开始的前若干步迭代中模型的演化。这一阶段通常决定了模型是否能成功学习，因为在早期阶段建立的"信号方向"在后续训练中会被放大。

   通过跟踪关键统计量（如注意力权重的信噪比、值矩阵的方向）在早期阶段的演化，论文推导出模型成功学习所需的条件。

3. **存储容量的显式公式**:

   核心结果是一个关于存储容量的显式缩放关系，揭示了三个关键量之间的**乘法依赖**：

   成功检索所需条件大致为：$N \cdot d \cdot L$ 满足特定的缩放关系

   具体来说：
   - **样本量 N**：需要足够多的样本来学习 token-标签映射
   - **嵌入维度 d**：更高维度减少 token 间干扰，提高存储容量
   - **序列长度 L**：更长序列增加了注意力选择的难度

   三者之间的乘法耦合是非正交嵌入的直接后果——在正交嵌入下，这些因素可以独立分析。

4. **信息论下界**:

   论文不仅从上方（算法角度）分析了存储容量，还从下方（统计/信息论角度）给出了该问题的固有难度下界。

   结果表明：N、d、L 之间的乘法缩放关系不是算法的局限，而是问题本身的**内在性质**。任何方法——无论是 Transformer 还是其他架构——在非正交嵌入下都无法绕过这一缩放瓶颈。

### 理论工具

- **随机矩阵理论**：分析非正交随机嵌入矩阵的谱性质和 token 间干扰
- **高维概率**：处理有限样本下经验梯度的集中不等式
- **信息论**：建立统计下界，证明缩放关系的最优性
- **动力系统分析**：跟踪梯度下降迭代中关键统计量的演化

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

## 相关论文

- [SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning](../../CVPR2026/optimization/scope_semantic_coreset_with_orthogonal_projection.md)
- [Markovian Transformers for Informative Language Modeling](markovian_transformers_for_informative_language_modeling.md)
- [The Affine Divergence: Aligning Activation Updates Beyond Normalisation](the_affine_divergence_aligning_activation_updates_beyond_normalisation.md)
- [Beyond the Mean: Fisher-Orthogonal Projection for Natural Gradient Descent in Large Batch Training](../../AAAI2026/optimization/beyond_the_mean_fisher-orthogonal_projection_for_natural_gradient_descent_in_lar.md)
- [Πnet: Optimizing Hard-Constrained Neural Networks with Orthogonal Projection Layers](pinet_optimizing_hard-constrained_neural_networks_with_orthogonal_projection_lay.md)

<!-- RELATED:END -->
