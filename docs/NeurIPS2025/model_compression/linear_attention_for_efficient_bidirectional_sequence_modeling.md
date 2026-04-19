---
title: >-
  [论文解读] Linear Attention for Efficient Bidirectional Sequence Modeling
description: >-
  [NeurIPS 2025][模型压缩][线性注意力] 提出 Lion 框架，首次系统性地将线性 Transformer 扩展到双向序列建模，统一了完整线性注意力、双向 RNN 和分块并行三种等价表示形式，训练速度比 SSM 快 10 倍且匹配 softmax Transformer 性能。
tags:
  - NeurIPS 2025
  - 模型压缩
  - 线性注意力
  - 双向序列建模
  - 状态空间模型
  - 高效推理
  - 双向RNN
---

# Linear Attention for Efficient Bidirectional Sequence Modeling

**会议**: NeurIPS 2025  
**arXiv**: [2502.16249](https://arxiv.org/abs/2502.16249)  
**代码**: [GitHub](https://github.com/LIONS-EPFL/Lion)  
**领域**: 模型压缩 / 高效序列建模  
**关键词**: 线性注意力, 双向序列建模, 状态空间模型, 高效推理, 双向RNN

## 一句话总结

提出 Lion 框架，首次系统性地将线性 Transformer 扩展到双向序列建模，统一了完整线性注意力、双向 RNN 和分块并行三种等价表示形式，训练速度比 SSM 快 10 倍且匹配 softmax Transformer 性能。

## 研究背景与动机

1. **领域现状**: 线性 Transformer 和状态空间模型 (SSM) 已成为 softmax Transformer 在因果序列建模中的高效替代方案，可通过矩阵乘法并行训练并以 RNN 形式高效推理。

2. **现有痛点**: 尽管在因果任务中成功，但线性 Transformer 在双向序列建模中仍缺乏统一框架。现有双向 SSM（如 Bi-Mamba、Vim）大多简单将因果形式在前向和后向方向分别应用（如双扫描），未能利用双向建模的天然先验——训练和推理时整个序列均可用。

3. **核心矛盾**: 双向 SSM 由于依赖分块训练保证数值稳定性（避免衰减因子累积乘积溢出/下溢），训练速度远慢于 softmax Transformer。而直接将两个线性 Transformer 输出naive相加会导致"双重计数"和注意力不平衡问题。

4. **本文目标**: 如何构建一个通用框架，让广泛类别的线性 Transformer 都能高效地应用于双向序列建模。

5. **切入角度**: 从因果线性注意力的衰减掩码出发，定义双向对称掩码 $\mathbf{M}_{ij}$ 使其等于位置 $i$ 和 $j$ 之间所有衰减因子的乘积，从而自然推导出完整注意力、RNN 和分块并行三种等价形式。

6. **核心 idea**: 因果线性 Transformer 的衰减掩码可以自然推广为对称的双向掩码，通过下三角/上三角分解实现等价的双向 RNN，且无需分块即可稳定训练。

## 方法详解

### 整体框架

Lion 框架为双向线性 Transformer 提供三种理论等价的表示：
- **Full Linear Attention**: 最大训练速度，直接计算 $\mathbf{Y} = \text{scale}(\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M})\mathbf{V}$
- **Bidirectional RNN**: 最高推理效率，前向和反向各运行一次 RNN 再合并
- **Chunkwise Parallel**: 平衡速度与内存

### 关键设计

**1. 双向掩码构造**

- **功能**: 将因果掩码推广到双向，编码位置间的相对距离信息
- **核心思路**: 定义 $\mathbf{M}_{ij}$ 为位置 $i$ 和 $j$ 之间所有衰减因子 $\lambda_k$ 的乘积。对于选择性衰减有 $\mathbf{M}_{ij} = \prod_{k=\min(i,j)+1}^{\max(i,j)} \lambda_k$；固定衰减有 $\mathbf{M}_{ij} = \lambda^{|i-j|}$；无衰减则 $\mathbf{M}_{ij} = 1$。将其分解为 $\mathbf{M} = \mathbf{M}^F + \mathbf{M}^B - \mathbf{I}$，其中 $\mathbf{M}^F$ 为下三角、$\mathbf{M}^B$ 为上三角。
- **设计动机**: 因果设置中掩码 $\mathbf{M}^C_{ij} = \lambda_{j+1}\lambda_{j+2}\cdots\lambda_i$ 编码了相对位置信息，双向场景自然推广为对称形式。

**2. 平衡双向 RNN 合并**

- **功能**: 避免前向和后向 RNN 输出naive相加导致的双重计数和注意力不平衡
- **核心思路**: 将注意力矩阵 $\mathbf{A} = \mathbf{Q}\mathbf{K}^\top$ 分解为 $\mathbf{A}^F$（下三角，对角取半）和 $\mathbf{A}^B$（上三角，对角取半），同样分解掩码和缩放因子。最终输出为 $\mathbf{Y} = (\mathbf{C}^F + \mathbf{C}^B)^{-1}(\mathbf{Y}^F + \mathbf{Y}^B)$，其中 $\mathbf{Y}^F$ 和 $\mathbf{Y}^B$ 分别由前向和反向 RNN 计算。反向部分通过翻转序列后复用前向 RNN 即可。
- **设计动机**: 简单相加导致 $\mathbf{Y} = ((\mathbf{I} + \mathbf{1}) \odot \mathbf{QK}^\top)\mathbf{V}$，对角被计算两次导致不平衡。

**3. 三种 Lion 变体**

- **功能**: 覆盖不同衰减类型
- **核心思路**: Lion-lit（无衰减 $\lambda_i = 1$，双向 Vanilla Linear Transformer）、Lion-d（固定可学习衰减 $\lambda = \sigma(a)$，双向 RetNet）、Lion-s（选择性衰减 $\lambda_i = \sigma(\mathbf{W}\mathbf{x}_i + b)$，双向 GRFA/Mamba2 风格）。
- **设计动机**: 表 1 展示了十余种因果线性 Transformer 均可通过 Lion 映射到双向形式，三种变体覆盖了标量/对角衰减的代表性情况。

### 损失函数 / 训练策略

- 训练时使用 Full Linear Attention 形式以获得最大速度（与 softmax Transformer 可比）
- 推理时可选择 RNN 形式（最高内存效率）、完整注意力（最快速度）或分块形式（平衡）
- 使用 shifted normalized SiLU 激活函数 $\phi(\mathbf{x}) = \frac{\text{SiLU}(\mathbf{x}) + 0.5}{\|\text{SiLU}(\mathbf{x}) + 0.5\|}$
- 直接替换 DeiT / BERT 中的注意力层，不修改其他超参数

## 实验关键数据

### 主实验

**ImageNet-1K 图像分类（Small 规模）**

| 模型 | 参数量 | Top-1 Acc (%) | 训练时间倍数 ↓ |
|------|--------|--------------|--------------|
| DeiT | 22M | 79.8 | ×1 |
| Hydra | 22M | 78.6 | ×2.50 |
| Vim | 26M | 80.3 | ×14.95 |
| **Lion-s♮** | 22M | **80.5** | **×1.00** |
| Lion-d | 22M | 79.8 | ×0.97 |
| Lion-lit | 22M | 78.9 | ×0.76 |

**ImageNet-1K 图像分类（Base 规模）**

| 模型 | 参数量 | Top-1 Acc (%) | 训练时间倍数 ↓ |
|------|--------|--------------|--------------|
| DeiT | 86M | 81.8 | ×1 |
| Hydra | 91M | 81.0 | ×2.51 |
| Vim | 98M | 81.9 | ×14.63 |
| **Lion-s♮** | 86M | **82.0** | **×1.01** |

### 消融实验

| 组件 | Top-1 Acc (Small) |
|------|------------------|
| 无衰减 (Lion-lit) | 78.9 |
| 固定衰减 (Lion-d) | 79.8 |
| 选择性衰减 (Lion-s) | 79.6 |
| Lion-s + 多扫描 (Lion-s♮) | 80.5 |
| Naive 前后向相加（不平衡） | 性能显著下降 |

### 关键发现

- Lion 在训练速度上与 DeiT 持平，但比 Vim 快约 **15 倍**、比 Hydra 快约 **2.5 倍**
- Lion 在 Base 规模上匹配甚至超过 softmax Transformer（DeiT 81.8% vs Lion-s♮ 82.0%）
- RNN 形式推理时内存复杂度为 $O(d^2)$，不随序列长度增长
- 在 MLM 任务（C4 数据集）上，Lion-s 也达到与 BERT 可比的性能

## 亮点与洞察

- **统一性极强**: 一个框架覆盖了 LinAtt、RetNet、Mamba-2、GLA、HGRN-2、xLSTM、DeltaNet 等十余种线性 Transformer 的双向扩展
- 证明了 Full Linear Attention 无需分块即可数值稳定训练（因双向场景中所有衰减因子已知，可用 cumsum 在 log 空间高效计算）
- 训练-推理解耦：训练时用高速的完整注意力，推理时切换到低内存的 RNN

## 局限与展望

- 目前主要关注标量/对角衰减（$TC^0$ 类），非对角衰减（如 DeltaNet）的双向扩展仅在附录讨论
- 视觉任务需要多扫描策略（Lion-s♮）弥补缺少显式位置编码的问题，增加实现复杂度
- 在 LRA 长程依赖基准上的评估有限

## 相关工作与启发

- RetNet、Mamba-2、GLA 等因果线性 Transformer 均可通过 Lion 扩展到双向
- Hydra 和 Vim 作为现有双向 SSM 的代表，其"双扫描"方案被 Lion 证明是次优的
- 启发：双向建模的核心优势是"整个序列可用"，应充分利用这一先验而非简单复制因果形式

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次为线性 Transformer 建立系统性的双向扩展框架，理论推导严谨
- **实验充分度**: ⭐⭐⭐⭐ 覆盖图像分类和 MLM 两大双向任务，但缺少更多下游应用评估
- **写作质量**: ⭐⭐⭐⭐⭐ 结构清晰，数学推导详尽，表 1 的统一映射表非常有价值
- **价值**: ⭐⭐⭐⭐⭐ 为高效双向建模提供了理论基础和实用工具，训练速度与 softmax Transformer 持平是重大突破

<!-- RELATED:START -->

## 相关论文

- [RAT: Bridging RNN Efficiency and Attention Accuracy via Chunk-based Sequence Modeling](rat_bridging_rnn_efficiency_and_attention_accuracy_via_chunk-based_sequence_mode.md)
- [LALIC: Linear Attention Modeling for Learned Image Compression](../../CVPR2025/model_compression/linear_attention_modeling_for_learned_image_compression.md)
- [A*-Thought: Efficient Reasoning via Bidirectional Compression for Low-Resource Settings](a-thought_efficient_reasoning_via_bidirectional_compression_for_low-resource_set.md)
- [Infrequent Exploration in Linear Bandits](infrequent_exploration_in_linear_bandits.md)
- [FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning](../../ICCV2025/model_compression/fastvar_linear_visual_autoregressive_modeling_via_cached_token_pruning.md)

<!-- RELATED:END -->
