---
title: >-
  [论文解读] The Structure of Relation Decoding Linear Operators in Large Language Models
description: >-
  [NeurIPS 2025 Spotlight][模型压缩][线性关系解码] 揭示 Transformer 语言模型中的线性关系解码器（LRE）并非编码细粒度关系，而是提取共享的粗粒度语义属性（如"国家"、"性别"），并利用阶-3 张量网络将大量关系解码矩阵压缩数个数量级。 Hernandez et al. (2023) 发…
tags:
  - "NeurIPS 2025 Spotlight"
  - "模型压缩"
  - "线性关系解码"
  - "张量网络"
  - "知识压缩"
  - "可解释性"
  - "语义属性"
---

# The Structure of Relation Decoding Linear Operators in Large Language Models

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.26543](https://arxiv.org/abs/2510.26543)  
**代码**: [GitHub](https://bit.ly/structure-of-relations)  
**领域**: 模型压缩  
**关键词**: 线性关系解码, 张量网络, 知识压缩, 可解释性, 语义属性

## 一句话总结

揭示 Transformer 语言模型中的线性关系解码器（LRE）并非编码细粒度关系，而是提取共享的粗粒度语义属性（如"国家"、"性别"），并利用阶-3 张量网络将大量关系解码矩阵压缩数个数量级。

## 研究背景与动机

Hernandez et al. (2023) 发现 Transformer 语言模型中的关系知识（如"巴黎→法国"）可以被一个简单的仿射变换 $f_R(\mathbf{v}_S) = W_R\mathbf{v}_S + \mathbf{b}_R$ 近似。每个关系 $R$ 对应一个 $d \times d$ 的线性关系嵌入（LRE）矩阵 $W_R$。

但先前工作只分析单个关系，留下了关键问题：

**这些关系解码器之间是否存在共享结构？** 一个 LRE 矩阵在 GPT-J（$d=4096$）中就有超过 1600 万参数，100 个关系就是 16 亿参数——是否存在更紧凑的共享表示？

**如果可以压缩，为什么能压缩？** 冗余的来源是语义重叠还是更深层的结构规律？

**压缩后的模型能否泛化到未见关系？**

这些问题不仅关乎可解释性（理解模型如何组织知识），也关乎效率（用更少的参数表示大量关系）。

## 方法详解

### 整体框架

研究分三个层次递进：
1. 用张量网络压缩关系解码矩阵的集合 → 证明可压缩性
2. 用交叉评估协议揭示压缩性的来源 → 发现属性级编码
3. 测试张量网络对未见关系的泛化能力 → 探索共享表示的极限

### 关键设计

1. **张量网络压缩**: 将 $n$ 个 $d \times d$ 的关系矩阵堆叠为一个阶-3 张量 $\mathbb{R}^{d \times d \times n}$，然后用两种张量网络架构进行压缩：

    - **SimpleOrder3Network**: 核心是一个小型阶-3 张量 $T^0 \in \mathbb{R}^{d_{s'} \times d_{r'} \times d_{o'}}$，通过三个投影矩阵 $P^1, P^2, P^3$ 连接到原始嵌入空间。参数量 $N_{\text{Simple}} = (d \cdot d_{s'} + d \cdot d_{r'} + d \cdot d_{o'}) + d_{s'} \cdot d_{r'} \cdot d_{o'}$，远小于原始的 $n \cdot d^2$。
    - **TriangleTensorNetwork**: 核心由三个互连的阶-3 张量组成，提供更强的表达能力。对于给定的关系嵌入 $\mathbf{v}_r$，通过张量缩并（contraction）生成对应的 LRE 矩阵。
   
   训练方式是端到端的：固定 LLM 参数，仅训练张量网络，损失函数为对象 token 的交叉熵。关键发现：**不加额外关系编码器的线性张量网络反而比加了非线性编码器的版本效果更好**，说明线性结构足以高效压缩。

2. **交叉评估协议（Cross-Evaluation Protocol）**: 对于 $k$ 个关系和对应的 $k$ 个解码器 $\{f_j\}$，构建 $k \times k$ 的忠实度矩阵 $F_{j,l} = \text{faithfulness}(R_l, f_j)$，即把关系 $j$ 的解码器应用到关系 $l$ 的主语上。如果两个关系语义相近（如"特征性别"和"职业性别"），则交叉评估分数应较高。这一矩阵可视为关系间的经验相似度核，揭示了明显的**块状结构**。

3. **属性提取器假说**: 交叉评估的核心发现是：LRE 矩阵本质上不是关系解码器，而是**属性提取器**。例如，"首都所在国家"、"食物来源国家"、"地标所在国家"的解码器可以互换使用，因为它们都在提取"国家"这一粗粒度属性。甚至 "职业性别" 解码器可以在 "特征性别" 上取得更好的忠实度。这解释了为什么压缩如此有效：大量看似不同的关系实际共享一小组属性提取模式。

### 损失函数 / 训练策略

- 端到端训练，固定 LLM 参数，优化张量网络参数。
- 损失函数：$\mathcal{L}_{\mathcal{R}}(T_{s,r,o}) = \sum_{R \in \mathcal{R}} \sum_{(S,O) \in R} CE(\mathbb{1}_O, L_{\text{head}}(T^R_{s,o}(\mathbf{v}_S)))$
- 使用 SGD 优化器，在关系-主语-对象三元组上训练至收敛。

## 实验关键数据

### 主实验 — 压缩效果

| 方法 | 参数量 | 平均忠实度 | 对比 |
|------|--------|-----------|------|
| 原始 47 个 LRE 矩阵 | ~788M | 0.41 | 基线（Jacobian近似） |
| 逐关系低秩近似 | ~80M | ~0.35 | 独立压缩，无共享 |
| SimpleOrder3Network | <1M | >0.41 | **压缩 788x，忠实度反而更高** |
| TriangleTensorNetwork | <1M | >0.41 | 类似，内部结构更丰富 |
| 张量网络+额外编码器 | 稍多 | 略低 | 非线性编码器反而有害 |

### 泛化实验 — 数学数据集

| 配置 | 训练集忠实度 | 测试集忠实度 | 说明 |
|------|------------|------------|------|
| 张量网络（75%训练/25%测试） | 0.992 (±0.012) | 0.96 (±0.031) | 近乎完美的泛化 |
| 多数类基线 | — | 0.30 | 对比参考 |
| 随机关系嵌入 | 完美记忆 | ~0 | 确认依赖语义信息 |
| 随机主语/对象嵌入 | ~0 | ~0 | 确认依赖实体表示 |

### 消融实验

| 消融条件 | 效果 | 说明 |
|---------|------|------|
| 去除语义相似关系 | 仍可大幅压缩 | 压缩不仅来自语义重叠 |
| 使用非线性关系编码器 | 无改善 | 线性结构已足够 |
| 扩展数据集（79关系） | 交叉评估块结构更明显 | 进一步证实属性假说 |
| GPT-J / Llama 3.1 8B / GPT-NeoX-20B | 结论一致 | 跨模型鲁棒性 |

### 关键发现

- 47 个关系的 788M 参数可被 <1M 参数的张量网络等效替代，压缩率约 **800 倍**。
- 交叉评估矩阵的块结构清晰对应语义属性类（国家、性别、反义词等），甚至存在跨类的句法相似性（如"首字母"解码器对"最高级形容词"也有效）。
- 在密集且结构化的算术关系数据集上，张量网络能完美泛化到未见关系（忠实度 0.96+）。

## 亮点与洞察

- **"属性提取器而非关系解码器"** 这一发现改变了对 LRE 的理解：模型不是存储了数千个独立的关系映射，而是通过少数几种属性提取模式的组合来解码关系。
- 张量网络作为压缩工具的成功暗示了**可能将 LoRA 矩阵集合也进行类似的张量网络压缩**。
- 交叉评估协议是一种独立于嵌入相似度的语义相似性度量方法，可广泛应用于其他场景。

## 局限与展望

- 仅在较小的 LLM 上验证（6B、8B、20B参数），大型模型/指令微调模型上的适用性未知。
- 关系数据集覆盖有限（47-79 个关系），人类知识中的关系空间远大于此。
- 在通用语言关系上泛化能力有限——仅对语义足够接近的关系才能成功泛化。
- 属性提取器假说目前主要是经验性的，缺乏在模型内部机制层面的验证。

## 相关工作与启发

- 与 Chanin et al. (2024) 的 Linear Relational Concept（LRC）互补：LRC 用 LRE 的伪逆做分类，本文揭示 LRE 之间的共享结构。
- 张量网络在物理学中用于量子态表示，本文将其引入 NLP 知识表示，跨学科应用值得关注。
- 对 Mixture-of-Experts (MoE) 架构的压缩也有启示：多个专家可能共享底层属性结构。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 交叉评估协议和属性提取器假说是原创且深刻的洞察
- 实验充分度: ⭐⭐⭐⭐ 三个模型、多个数据集、充分的消融和泛化实验
- 写作质量: ⭐⭐⭐⭐ 概念清晰，渐进式呈现，图表有效但部分过于密集
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 知识结构的理解有重要推动，兼具可解释性和压缩价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] A Simple Linear Patch Revives Layer-Pruned Large Language Models](a_simple_linear_patch_revives_layerpruned_large_language_mod.md)
- [\[NeurIPS 2025\] Correlation Dimension of Auto-Regressive Large Language Models](correlation_dimension_of_auto-regressive_large_language_models.md)
- [\[NeurIPS 2025\] Restoring Pruned Large Language Models via Lost Component Compensation](restoring_pruned_large_language_models_via_lost_component_compensation.md)
- [\[NeurIPS 2025\] PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models](permllm_learnable_channel_permutation_for_nm_sparse_large_language_models.md)
- [\[NeurIPS 2025\] LayerIF: Estimating Layer Quality for Large Language Models using Influence Functions](layerif_estimating_layer_quality_for_large_language_models_using_influence_funct.md)

</div>

<!-- RELATED:END -->
