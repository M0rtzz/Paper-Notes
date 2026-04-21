---
title: >-
  [论文解读] Can Graph Neural Networks Learn Language with Extremely Weak Text Supervision?
description: >-
  [ACL 2025][图学习][图神经网络] 本文提出Morpher，一种多模态提示学习范式，在极弱文本监督（仅几个token的标签名）下，通过同时学习图提示和文本提示将预训练GNN嵌入到LLM的语义空间中，实现跨任务、跨领域的图分类迁移以及首个CLIP风格的GNN零样本分类原型。
tags:
  - ACL 2025
  - 图学习
  - 图神经网络
  - 多模态提示学习
  - 图文对齐
  - 少样本学习
  - 零样本分类
---

# Can Graph Neural Networks Learn Language with Extremely Weak Text Supervision?

**会议**: ACL 2025  
**arXiv**: [2412.08174](https://arxiv.org/abs/2412.08174)  
**代码**: [Violet24K/Morpher](https://github.com/Violet24K/Morpher)  
**领域**: graph_learning  
**关键词**: 图神经网络, 多模态提示学习, 图文对齐, 少样本学习, 零样本分类  

## 一句话总结

本文提出Morpher，一种多模态提示学习范式，在极弱文本监督（仅几个token的标签名）下，通过同时学习图提示和文本提示将预训练GNN嵌入到LLM的语义空间中，实现跨任务、跨领域的图分类迁移以及首个CLIP风格的GNN零样本分类原型。

---

## 研究背景与动机

**研究背景：** CLIP通过图文联合预训练构建了高质量的视觉-语言对齐模型，但将此范式扩展到图数据面临重大挑战。图数据天然稀缺，文本监督极弱（标签名仅几个token），任务涉及节点/边/图三个层级，且相同图结构在不同领域可能有截然不同的语义。

**现有方法的局限性：** (1) 图文联合预训练仅在分子领域和文本属性图上可行，通用图数据上因数据不足而不可行；(2) 现有图提示方法（如GPF）在实际应用中存在交叉连接淹没原始图结构的问题，导致训练不稳定；(3) 仅对单一模态做提示会限制另一模态的调整灵活性。

**核心动机：** 利用LLM编码器已有的高质量语义空间，通过双模态提示学习在冻结GNN和LLM参数的条件下，将图嵌入对齐到语义空间。

---

## 方法详解

### 整体框架

Morpher由三个可学习组件构成：图提示 $\mathbf{P}_\theta^g$、文本提示 $\mathbf{P}_\theta^t$ 和跨模态投影器 $\text{Proj}_\theta$。GNN和LLM参数完全冻结，仅通过提示和投影器实现对齐。

### 关键设计

**1. 改进的图提示设计：** 分析了现有图提示（Sun et al., 2023）的根本问题——由于prompt token初始化接近零向量，sigmoid值接近0.5，导致交叉连接过密，prompt图的特征淹没原始图。解决方案：将交叉连接数量约束为不超过原始图边数 $n_e$，每个节点最多连接 $\lfloor n_e/a \rfloor$ 个prompt token，并使用余弦相似度替代sigmoid计算连接权重。

**2. 跨模态投影器：** 使用tanh激活的线性层将图嵌入空间映射到文本嵌入空间：
$$\widetilde{\mathbf{v}} = \text{Proj}_\theta(\mathbf{v}) := \tanh(\mathbf{W}\mathbf{v} + \mathbf{b}) \in \mathbb{R}^{1 \times d_t}$$

**3. 文本嵌入归一化：** 考虑到少量标签文本可能语义接近，先减去均值 $\mu$ 再L2归一化，分离语义相近的类别嵌入。

### 损失函数

采用batch内对比损失进行图-文对齐训练：

$$\mathcal{L}_{G \rightarrow T} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\mathbf{z}_i^{\mathcal{G}} \cdot \mathbf{z}_i^t / \tau)}{\sum_{j=1}^{B} \exp(\mathbf{z}_i^{\mathcal{G}} \cdot \mathbf{z}_j^t / \tau)}$$

推理时通过计算图嵌入与各类别文本嵌入的余弦相似度进行分类。

---

## 实验

### 主实验结果（Few-shot图分类）

| 训练方式 | GNN预训练 | MUTAG | ENZYMES | PROTEINS | MSRC_21C |
|----------|-----------|-------|---------|----------|----------|
| Supervised | N/A+GCN | 66.00 | 16.67 | 65.89 | 38.85 |
| Pre-train+FT | GraphCL+GCN | 70.00 | 17.91 | 65.89 | 40.00 |
| Graph Prompt | GPF+GCN | 64.67 | 17.02 | 63.50 | 43.46 |
| **Morpher** | **GCN+LLaMA** | **75.33** | **22.39** | **68.32** | **50.86** |

### 消融实验

| 组件 | 效果 |
|------|------|
| 无图提示改进 | 训练不稳定，部分数据集无法收敛 |
| 无文本提示 | 性能下降2-5%，单模态调整灵活性不足 |
| 无跨模态投影器 | 维度不匹配无法训练 |
| 原始GPF图提示 | 交叉连接过密导致性能退化 |

### 关键发现

- **极弱监督有效：** 仅用类名（几个token）作为文本监督，Morpher即可显著提升GNN分类性能
- **跨领域迁移：** 在分子→社交网络等跨领域设置下，Morpher仍保持竞争力
- **零样本原型：** 首次在GNN上实现CLIP风格的零样本分类——将图嵌入投射到文本空间后，可直接用未见过的类名进行分类
- **图提示问题诊断：** 揭示了现有图提示中交叉连接淹没原始图信息的根本原因，并提出了有效的修复方案

---

## 亮点

- 首个在极弱文本监督下的图-文多模态提示学习框架，GNN和LLM参数完全冻结
- 深入分析并修复了现有图提示设计中交叉连接过密的核心缺陷
- 实现了首个GNN的CLIP风格零样本分类原型，展示了图模型语言理解的可能性
- 在少样本、多任务和跨领域三种设置下均表现优异

## 局限性

- 依赖预训练GNN和LLM的质量，两者的预训练领域偏差可能影响结果
- 跨模态投影器仅使用简单的线性层+tanh，表达能力有限
- 零样本分类仅在有限场景下验证，泛化能力有待进一步评估
- 实验主要在中小规模图数据集上进行，未在大规模工业级图上验证

## 相关工作

- **CLIP与视觉-语言对齐：** Radford et al. (2021)的CLIP框架是本文灵感来源
- **图提示学习：** GPF (Sun et al., 2023) 首创图提示概念，本文发现并修复了其设计缺陷
- **图自监督预训练：** GraphCL (You et al., 2020), GCC (Qiu et al., 2020) 等提供预训练GNN
- **多模态提示学习：** CoCoOp (Zhou et al., 2022), MaPLe (Khattak et al., 2023) 在视觉-语言中使用双模态提示

---

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 总评 | 7.5/10 |

<!-- RELATED:START -->

## 相关论文

- [GraphNarrator: Generating Textual Explanations for Graph Neural Networks](graphnarrator.md)
- [Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering](kg_llm_trustworthy_qa.md)
- [Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs](../../NeurIPS2025/graph_learning/dynamic_bundling_with_large_language_models_for_zero-shot_inference_on_text-attr.md)
- [Over-squashing in Spatiotemporal Graph Neural Networks](../../NeurIPS2025/graph_learning/over-squashing_in_spatiotemporal_graph_neural_networks.md)
- [Graph Neural Networks for Interferometer Simulations](../../NeurIPS2025/graph_learning/graph_neural_networks_for_interferometer_simulations.md)

<!-- RELATED:END -->
