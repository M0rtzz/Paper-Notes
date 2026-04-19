---
title: >-
  [论文解读] Generative Graph Pattern Machine
description: >-
  [NeurIPS 2025][图学习][graph pre-training] 提出 Generative Graph Pattern Machine (G2PM)，一种完全无消息传递的生成式 Transformer 图预训练框架：通过随机游走将图实例（节点/边/图）tokenize 为子结构序列，以 Masked Substructure Modeling 目标进行自监督预训练，在节点/链接/图分类及跨域迁移任务上全面超越现有图预训练方法，并展现出类似 NLP/CV 的模型和数据双重扩展性。
tags:
  - NeurIPS 2025
  - 图学习
  - graph pre-training
  - Transformer
  - substructure tokenization
  - masked substructure modeling
  - random walk
  - message-passing-free
---

# Generative Graph Pattern Machine

**会议**: NeurIPS 2025  
**arXiv**: [2505.16130](https://arxiv.org/abs/2505.16130)  
**作者**: Zehong Wang, Zheyuan Zhang, Tianyi Ma, Chuxu Zhang, Yanfang Ye (University of Notre Dame, University of Connecticut)
**代码**: https://github.com/Zehong-Wang/G2PM (有)  
**领域**: 图学习 / 图预训练 / Transformer  
**关键词**: graph pre-training, generative Transformer, substructure tokenization, masked substructure modeling, random walk, message-passing-free

## 一句话总结

提出 Generative Graph Pattern Machine (G2PM)，一种完全无消息传递的生成式 Transformer 图预训练框架：通过随机游走将图实例（节点/边/图）tokenize 为子结构序列，以 Masked Substructure Modeling 目标进行自监督预训练，在节点/链接/图分类及跨域迁移任务上全面超越现有图预训练方法，并展现出类似 NLP/CV 的模型和数据双重扩展性。

## 研究背景与动机

Transformer 已在 NLP（LLM）和 CV（ViT）中通过生成式预训练取得巨大成功，但这一范式在图领域尚未实现类似的突破，严重制约了图基础模型的发展。现有图学习主要依赖**消息传递 GNN（MPNN）**，存在以下根本瓶颈：

**表达能力受限**：受制于 1-WL 测试的上界

**过平滑（over-smoothing）**：层数增加时节点表示趋同

**过压缩（over-squashing）**：信息在瓶颈边上被压缩丢失

**长程依赖建模不足**：消息传递的局部性限制了全局信息流

**可扩展性差**：增大模型或数据反而无法带来持续的性能提升

此外，已有图预训练研究主要采用**对比学习**，但其学习泛化语义表示的能力弱于生成式目标（如 BERT、MAE 已充分说明）。

将生成式 Transformer 预训练迁移到图领域面临三大独特挑战：
- **缺乏序列结构**：图不具备文本的线性或图像的网格结构
- **语义粒度低**：图的节点/边是低级语义单元，不同于词/图像块的高级语义
- **序列长度**：如果以节点为 token，序列长度随节点数线性增长，Transformer 的二次复杂度不可承受

## 方法详解

### 核心思想：子结构序列化

G2PM 的核心洞察是：**有意义的图子结构（substructure）**是图的语义构建单元（如社交网络中的三角形表示稳定关系，分子图中的苯环编码化学稳定性）。因此可以将任意图实例表示为**子结构序列**，类似于文本的词序列或图像的 patch 序列。

### 随机游走 Tokenizer

为避免子图同构匹配（NP-complete）的高计算成本，G2PM 采用**随机游走**实时采样子结构模式：

- 对每个图实例采样 $k$ 条无偏随机游走，每条长度为 $L$
- 转移概率：$P(v_{i+1}|v_0,...,v_i) = \mathbb{1}[(v_i,v_{i+1}) \in \mathcal{E}] / D(v_i)$
- 每条游走对应一个子结构模式（如 [1,2,3,1,4,3] 对应菱形结构）

子结构编码方式：将游走中的节点特征序列 $w = [\mathbf{x}_1, ..., \mathbf{x}_m]$ 输入一个**小型 Transformer 编码器** $f$ 生成子结构嵌入 $\mathbf{p} = f(w)$。

**关键设计决策**：
- **无位置编码**：实验发现位置嵌入无一致收益，且引入立方复杂度，因此完全省略
- **任务无关**：节点任务从目标节点出发游走，边任务从端点出发，图任务从随机节点出发
- **消除消息传递瓶颈**：随机游走天然捕获长程依赖，表达能力超越 1-WL，且不受过平滑/过压缩影响

### G2PM 主干网络

采用标准 Transformer 架构作为编码器和解码器的骨干。输入为子结构 token 序列 $\mathbf{P} = [\mathbf{p}_1, ..., \mathbf{p}_n]$，通过多头自注意力和前馈网络进行逐层编码。默认配置为：

- 隐藏维度 768，注意力头数 12
- 编码器 3 层，解码器 1 层
- 参数约 21.3M（编码器）+ 7.1M（解码器）

### 预训练目标：Masked Substructure Modeling (MSM)

类似 BERT/MAE 的掩码建模思想，但操作单元是**子结构**而非词或 patch：

1. 从图中采样子结构序列 $\mathbf{P} = [\mathbf{p}_1, ..., \mathbf{p}_n]$
2. 随机掩码一部分子结构（默认高比例掩码），保留可见子结构 $\mathbf{P}_{vis}$
3. 可见子结构进入编码器生成上下文表示 $\mathbf{H}_{vis}$
4. 在掩码位置插入可学习的 [MASK] token，拼接后送入解码器重建

**重建目标**：不是重建低级节点特征，而是重建由 **EMA 编码器**生成的高级语义嵌入：

$$\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} \text{is\_masked}(\mathbf{p}_i) \cdot \|\mathbf{r}_i - \text{sg}[\hat{\mathbf{p}}_i]\|_2^2$$

其中 $\hat{\mathbf{p}}_i = f_{\text{EMA}}(w_i)$ 是 EMA 编码器的输出，sg 为停梯度算子。

### 数据增强策略

采用**混合增强**策略，每个实例同时应用一种特征级和一种结构级增强：
- **特征级**：特征掩码（置零部分特征）、节点掩码（掩码整个节点）
- **结构级**：子结构腐蚀（丢弃游走中的节点）、子结构注入（用随机节点替换）

### 下游适配

预训练后丢弃解码器，在编码器上接线性预测头。对子结构表示做均值池化后进行最终预测。

## 设计空间探索（关键消融实验）

论文进行了系统的设计空间分析，产出多条重要洞察：

| 设计维度 | 最佳选择 | 洞察 |
|---------|---------|------|
| 架构风格 | MAE-style（稀疏编码器+轻量解码器） | SimMIM-style 表现更差，稀疏性鼓励编码器学习更强的上下文表示 |
| 模型维度 | 768（64.5M 参数） | 性能随维度持续提升，呈现良好的扩展规律 |
| 编码器/解码器层数 | 编码器 3 层 / 解码器 1 层 | 更深解码器反而导致过拟合 |
| 子结构编码器 | Transformer | 优于 GRU、GIN、Mean Pooling |
| 位置编码 | 不使用 | 无一致收益且计算代价高 |
| 掩码比例 | 高比例（~0.7-0.8） | 高掩码减少冗余游走，提供更挑战的学习信号 |
| 掩码 token | 可学习 | 优于固定/随机/采样 token |
| 重建目标 | EMA 语义嵌入 | 优于低级特征（均值池化/拼接） |
| 损失函数 | L2 loss | 优于 L1，对离群值更敏感有助于捕捉细微语义 |
| EMA 动量 | α=0.99, 每 10 步更新 | α=0（无 EMA）导致训练崩溃 |
| 增强策略 | 混合增强 | 单一增强效果有限，多样性至关重要 |

## 实验结果

### 节点分类（同质图）

在 7 个数据集上评估（Pubmed 到 ogbn-products），G2PM 在所有自监督方法中表现最佳，平均排名 2.0（仅次于有监督 GPM）：
- ogbn-arxiv：72.31%（60M 参数），持续随参数量提升
- ogbn-products：80.56%（显著超越 GraphMAE2 的 79.33%）

### 模型/数据扩展性

- **模型扩展**：ogbn-arxiv 上从 0.4M 到 64.5M 参数，性能从 68.2% 持续提升到 72.3%，而 GraphMAE/BGRL 在约 3M 参数后即停滞甚至下降
- **数据扩展**：增加预训练数据持续提升性能，而对比方法在小数据比例即达峰值后下降

### 图分类

在 7 个数据集（5 分子图 + 2 社交网络）上以平均排名 1.0 全面领先：
- ogbg-HIV：78.7%（vs GraphMAE 77.8%）
- IMDB-B：83.0%（vs GraphMAE 75.5%，巨大提升）
- 无预训练版本平均排名 8.3，说明预训练对图级任务至关重要

### 链接预测

在 Cora/Pubmed/ogbl-Collab 三个数据集上均优于 GCN 和 GraphMAE。

### 跨域迁移

展现出优秀的跨域迁移能力：
- Arxiv → Products：81.3%（正向迁移 +0.7%）
- Arxiv → HIV：76.8%
- HIV → Arxiv：72.6%（正向迁移 +0.3%）
- HIV → PCBA：77.9%（正向迁移 +2.3%）

跨域预训练在 Arxiv+FB15K237+ChemBL 上联合预训练后，在三个目标上均接近或超越图基础模型（OFA、GFT）。

## 理论洞察：子结构依赖关系

论文从三个角度分析了 MSM 有效性的结构基础：

1. **层次依赖**：小子结构（如三角形）是大结构（如团）的构建单元，推断可双向进行
2. **功能增强**：某些子结构统计共现（如高密度三角形与高阶团的关联）
3. **功能排斥**：某些子结构互相排斥（如密集社区与星形枢纽结构不兼容）

## 局限性与未来方向

- 当前使用**无序**子结构序列，适合掩码预测；未来可探索有序序列以支持**next-token prediction**，进一步提升扩展性
- 随机游走作为在线 tokenizer 缺乏可学习性；未来可设计**自适应可学习子结构 tokenizer**
- 异质图上（如 Actor 数据集），增加感受野未必带来提升，暗示某些图的关键信号是局部的

## 个人思考

1. **方法论意义重大**：G2PM 证明图学习可以完全脱离消息传递范式，用纯 Transformer + 随机游走 tokenization + 生成式预训练实现 SOTA，这为图基础模型指明了新方向
2. **扩展性是核心卖点**：这是第一个在图领域展现类似 NLP/CV 的模型和数据扩展规律的自监督方法，GraphMAE/BGRL 在 3M 参数后即停滞下降，而 G2PM 在 60M 仍在上升
3. **设计空间探索非常全面**：消融实验揭示的多条洞察（如 MAE 优于 SimMIM、高掩码比例更好、无需位置编码）对后续工作具有重要指导价值
4. **跨域迁移能力**值得关注：通过学习可迁移的子结构模式，G2PM 可以在域和任务边界之间正迁移，这是消息传递方法难以做到的
5. **与 GPM 的关系**：G2PM 可视为 GPM（同组前作，ViT 风格子结构建模）的生成式预训练扩展，将对比/监督学习升级为生成式自监督学习

<!-- RELATED:START -->

## 相关论文

- [Beyond Message Passing: Neural Graph Pattern Machine](../../ICML2025/graph_learning/beyond_message_passing_neural_graph_pattern_machine.md)
- [Relieving the Over-Aggregating Effect in Graph Transformers](relieving_the_over-aggregating_effect_in_graph_transformers.md)
- [Unifying and Enhancing Graph Transformers via a Hierarchical Mask Framework](unifying_and_enhancing_graph_transformers_via_a_hierarchical_mask_framework.md)
- [P-DRUM: Post-hoc Descriptor-based Residual Uncertainty Modeling for Machine Learning Potentials](p-drum_post-hoc_descriptor-based_residual_uncertainty_modeling_for_machine_learn.md)
- [From Sequence to Structure: Uncovering Substructure Reasoning in Transformers](from_sequence_to_structure_uncovering_substructure_reasoning_in_transformers.md)

<!-- RELATED:END -->
