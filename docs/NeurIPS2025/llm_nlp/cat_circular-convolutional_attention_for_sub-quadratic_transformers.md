---
title: >-
  [论文解读] CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers
description: >-
  [NeurIPS 2025][LLM/NLP][循环卷积] CAT 将标准自注意力中的 $N \times N$ 注意力矩阵替换为一个由 $N$ 维向量生成的循环矩阵（circulant matrix），利用 FFT 实现 $O(N \log N)$ 复杂度的注意力计算…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "循环卷积"
  - "FFT注意力"
  - "次二次复杂度"
  - "softmax保持"
  - "EIT框架"
---

# CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers

**会议**: NeurIPS 2025  
**arXiv**: [2504.06704](https://arxiv.org/abs/2504.06704)  
**代码**: 无（作者未公开）  
**领域**: LLM/NLP / Transformer 效率  
**关键词**: 循环卷积, FFT注意力, 次二次复杂度, softmax保持, EIT框架

## 一句话总结

CAT 将标准自注意力中的 $N \times N$ 注意力矩阵替换为一个由 $N$ 维向量生成的循环矩阵（circulant matrix），利用 FFT 实现 $O(N \log N)$ 复杂度的注意力计算，在严格保持 softmax 行归一化结构的前提下，在 ImageNet-1k（avg pool 下 CLIP-L 准确率 0.694 vs 标准注意力 0.646）和 WikiText-103 masked LM（PPL 8.32 vs 9.82）上匹配或超越标准注意力。

## 研究背景与动机

**领域现状**：Transformer 的标准自注意力机制复杂度为 $O(N^2)$，制约了长序列任务的可扩展性。现有高效注意力方案主要有两条路线：(1) Linear Transformer（如 Performer）用核函数近似 softmax，复杂度降至 $O(N)$ 但牺牲了 softmax 结构，训练不稳定；(2) 稀疏注意力（如 BigBird、Longformer）只在局部 token 子集上应用 softmax，丢失全局上下文覆盖。

**现有痛点**：Linear Transformer 的核近似经常导致数值不稳定和精度下降；稀疏方法需要精心调节块大小、稀疏模式等超参数，且不同序列长度需要重新调优；Mamba 等 SSM 方法完全放弃注意力机制，与现有 Transformer 训练栈和推理优化（FlashAttention、KV Cache）不兼容。

**核心矛盾**：所有现有方法都在"降低复杂度"和"保持 softmax 注意力的表示能力/兼容性"之间做取舍——要么放弃 softmax 换取效率，要么保留 softmax 但引入依赖序列长度的超参数。

**本文目标** 找到一种方法，既严格保持全局 softmax 行归一化（与标准注意力数学形式一致），又实现低于 $O(N^2)$ 的计算复杂度，且不引入额外超参数。

**切入角度**：作者观察到注意力矩阵的每一行本质上是对所有 token 的加权，而循环矩阵（circulant matrix）天然满足"每行是相同权重集合的循环移位"这一结构——且循环矩阵的乘法可以通过 FFT 在 $O(N \log N)$ 内完成。

**核心 idea**：将 $Q K^\top$ 生成的 $N \times N$ 注意力矩阵替换为由单向量 softmax 后生成的循环矩阵，利用 FFT 实现次二次复杂度的全局注意力。

## 方法详解

### 整体框架

CAT 作为标准多头自注意力的 drop-in 替换。输入 $\mathbf{X} \in \mathbb{R}^{N \times D}$，标准注意力需要 $\mathbf{W_Q}, \mathbf{W_K}, \mathbf{W_V}$ 三个投影矩阵（共 $3D^2$ 参数），CAT 仅需 $\mathbf{W_A} \in \mathbb{R}^{D \times 1}$（注意力投影）和 $\mathbf{W_V} \in \mathbb{R}^{D \times D}$（值投影），共 $(D+H)D$ 参数（$H$ 为头数）。

### 关键设计

1. **循环注意力矩阵构造**:

    - 功能：将 $N \times N$ 注意力权重矩阵替换为由 $N$ 维向量生成的循环矩阵
    - 核心思路：用单一投影 $\mathbf{Z} = \mathbf{X} \mathbf{W_A} \in \mathbb{R}^{N \times 1}$ 得到注意力得分向量，对其做行 softmax 得到 $\mathbf{Z}^\star = \text{softmax}(\mathbf{Z})$。然后构建循环矩阵 $\text{circ}(\mathbf{Z}^\star)$，其第 $i$ 行是 $\mathbf{Z}^\star$ 的第 $i$ 次循环移位。关键性质：$\text{softmax}(\text{circ}(\mathbf{Z})) \equiv \text{circ}(\text{softmax}(\mathbf{Z}))$，循环矩阵与 softmax 可交换，所以每行仍然是合法的 softmax 归一化权重
    - 设计动机：循环矩阵是"保持 softmax 行归一化的最简结构"——每行共享相同权重集合，只是位置不同。这同时赋予了模型显式的相对位置编码能力，因为注意力权重只依赖 token 间的相对偏移

2. **FFT 加速计算**:

    - 功能：利用循环卷积定理将 $O(N^2)$ 的矩阵乘法加速为 $O(N \log N)$
    - 核心思路：循环矩阵乘向量等价于循环卷积。根据卷积定理，$\text{circ}(\mathbf{Z}^\star) \mathbf{V} = \text{IFFT}[\text{FFT}(\mathbf{Z}^\star) \odot \text{FFT}(\mathbf{V})]$，其中 $\odot$ 为逐元素 Hadamard 乘积。FFT/IFFT 的复杂度为 $O(N \log N)$，Hadamard 乘积为 $O(N)$，总复杂度 $O(N \log N)$
    - 设计动机：这是唯一不近似、不截断、完全精确的次二次注意力计算方式——循环矩阵乘法与 FFT 在浮点精度内完全等价

3. **CAT-Alter 混合架构**:

    - 功能：在网络中交替使用 CAT 层和标准注意力层
    - 核心思路：将一半注意力层替换为 CAT，另一半保留标准注意力，交替排列。CAT 层提供 $O(N \log N)$ 的高效计算和循环移位正则化，标准注意力层保留完全灵活的全局交互能力
    - 设计动机：纯 CAT 在需要高度灵活全局交互的任务（如 causal LM with token pooling）中偶尔不如标准注意力，混合方案兼得两者优势。类似于 Jamba 等 Transformer/SSM 混合架构的思路

### 损失函数 / 训练策略

CAT 不改变训练损失——在视觉任务中使用标准交叉熵分类损失，在语言建模中使用标准交叉熵 LM 损失。训练时直接替换注意力层，使用相同超参数。ImageNet-1k 训练 50 epochs（4 × V100），WikiText-103 训练 50 epochs，均使用 AdamW。Causal LM 场景下需要对 $\mathbf{Z}$ 做移位以保持因果性，但这会将复杂度退化回 $O(N^2)$。

## 实验关键数据

### 主实验

| 任务 / 模型 | Pool/LM类型 | 方法 | 可学习参数 | 复杂度 | 指标 |
|------------|-----------|------|----------|--------|------|
| ImageNet CLIP-L | avg pool | Attention | $3D^2$ | $O(N^2)$ | Acc 0.646 |
| ImageNet CLIP-L | avg pool | **CAT** | $(D+H)D$ | $O(N\log N)$ | **Acc 0.694** (+4.8%) |
| ImageNet CLIP-L | avg pool | CAT-Alter | $(2D+H/2)D$ | $O(N^2)$ | Acc 0.681 |
| ImageNet CLIP-L | token | Attention | $3D^2$ | $O(N^2)$ | Acc 0.574 |
| ImageNet CLIP-L | token | CAT-Alter | $(2D+H/2)D$ | $O(N^2)$ | **Acc 0.593** |
| WikiText-103 GPT-2 | masked | Attention | $3D^2$ | $O(N^2)$ | PPL 9.82 |
| WikiText-103 GPT-2 | masked | **CAT** | $(D+H)D$ | $O(N\log N)$ | **PPL 8.32** (-15.3%) |
| WikiText-103 GPT-2 | causal | Attention | $3D^2$ | $O(N^2)$ | PPL 27.84 |
| WikiText-103 GPT-2 | causal | CAT-Alter | $(2D+H/2)D$ | $O(N^2)$ | PPL 27.68 |

### 消融实验

Q/K/V 参数化方式对比（CLIP-L, avg pool）：

| 循环卷积变体 | 可学习参数 | 复杂度 | Acc ↑ |
|------------|----------|--------|------|
| qkv (Averaged-Key) | $3D^2$ | $O(N\log N)$ | 0.696 |
| **qv (CAT, 默认)** | $(D+H)D$ | $O(N\log N)$ | **0.694** |
| q only | $(N+H)D$ | $O(N\log N)$ | 0.637 |
| v only | $(N+D)D$ | $O(N\log N)$ | 0.625 |
| 标准 Attention | $3D^2$ | $O(N^2)$ | 0.646 |

### 关键发现

- **avg pooling 下 CAT 大幅领先标准注意力**：CLIP-L 上提升 4.8%（0.694 vs 0.646），说明循环卷积的移位结构在全局 token 混合场景下特别有效
- **masked LM 场景增益惊人**：GPT-2 上 PPL 从 9.82 降至 8.32（-15.3%），Transformer-XL 上从 13.94 降至 10.28（-26.3%），暗示循环结构对 mask 预测有天然优势
- **qv 合并是最佳 trade-off**：与保留完整 qkv（仅多 0.2%）精度相当，但参数量从 $3D^2$ 降至 $(D+H)D$，减少约 2/3
- **causal 场景是弱点**：因果约束迫使 CAT 退回 $O(N^2)$，且精度略低于标准注意力，需要 CAT-Alter 混合方案补救
- **Linear Attention 训练崩溃**：在 CLIP-L 上多次出现 NaN loss，佐证了 softmax 保持的重要性

## 亮点与洞察

- **循环矩阵与 softmax 的交换律**是全文最精妙的理论洞察：$\text{softmax}(\text{circ}(\mathbf{Z})) = \text{circ}(\text{softmax}(\mathbf{Z}))$，这意味着只需对 $N$ 维向量做 softmax 就能得到 $N \times N$ 的合法注意力矩阵，一步到位解决了"保持 softmax"和"降低复杂度"的矛盾
- **"注意力系数从 $N^2$ 降到 $N$"的视角**：标准注意力需要运行时生成 $N^2$ 个系数，CAT 只需生成 $N$ 个然后循环移位。这不仅降低计算量，还可能起到正则化作用，避免过拟合
- **对 ViT avg pooling 趋势的呼应**：近期研究发现 avg pooling 在 ViT 中经常优于 class token pooling，CAT 恰好在 avg pooling 场景下表现最强，暗示循环结构与全局 token 混合有深层亲和性

## 局限与展望

- **causal LM 无法保持次二次复杂度**：因果 mask 破坏循环结构，使 CAT 退化为 $O(N^2)$。这是最大限制——GPT 式生成模型无法从中受益
- **实际加速有限**：FFT 实现在 $N=256$ 时开销与理论增益抵消，gather-based 实现仍为 $O(N^2)$（虽然有 ~10% 加速）。需要定制 GPU kernel 或更长序列才能体现 $O(N \log N)$ 优势
- **所有 token 共享相同注意力权重集合**：循环矩阵的每行只是权重的移位，不能学习真正 token-specific 的注意力模式。这在需要高度异质化注意力的任务中可能是瓶颈
- **未在超长序列任务上验证**：ImageNet ViT 的 $N=256$，WikiText 的 $N=256$，都很短。CAT 的理论优势需要在 $N > 10000$ 的长文本/长视频任务上验证
- **无代码开源**：可复现性受限

## 相关工作与启发

- **vs Performer / Linear Attention**: Performer 用随机特征近似 softmax 实现 $O(N)$，但破坏了 softmax 结构导致训练不稳定。CAT 严格保持 softmax 且更稳定，代价是复杂度为 $O(N \log N)$ 而非 $O(N)$
- **vs BigBird / Longformer**: 稀疏方法保持 softmax 但仅在 token 子集上，丢失全局覆盖且需要调参。CAT 覆盖所有 token 对且零超参数
- **vs Mamba (SSM)**: Mamba 完全放弃注意力，与 Transformer 推理栈不兼容。CAT 是 drop-in 替换，与 FlashAttention、KV Cache 等完全兼容。有趣的是 CAT 的 qv 合并设计与 Mamba 的输入驱动控制有概念相似性
- **与 Jamba 等混合架构的呼应**：CAT-Alter 的"交替混合"策略与 Jamba（SSM + Attention 混合）异曲同工，都说明单一机制难以在所有场景最优

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 循环矩阵替代注意力矩阵的想法优雅且理论基础扎实，EIT框架定义有指导意义
- 实验充分度: ⭐⭐⭐ 任务覆盖视觉和语言，但序列长度太短，缺少长序列任务验证和误差棒
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，消融设计合理，但部分概念重复叙述略冗长
- 价值: ⭐⭐⭐⭐ 为高效注意力设计提供了新思路，但causal场景的限制降低了对LLM的实际适用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Strassen Attention, Split VC Dimension and Compositionality in Transformers](strassen_attention_split_vc_dimension_and_compositionality_in_transformers.md)
- [\[ACL 2026\] Characterizing the Expressivity of Local Attention in Transformers](../../ACL2026/llm_nlp/characterizing_the_expressivity_of_local_attention_in_transformers.md)
- [\[AAAI 2026\] Vision Transformers are Circulant Attention Learners](../../AAAI2026/llm_nlp/vision_transformers_are_circulant_attention_learners.md)
- [\[NeurIPS 2025\] msf-CNN: Patch-based Multi-Stage Fusion with Convolutional Neural Networks for TinyML](msf-cnn_patch-based_multi-stage_fusion_with_convolutional_neural_networks_for_ti.md)
- [\[NeurIPS 2025\] Linear Transformers Implicitly Discover Unified Numerical Algorithms](linear_transformers_implicitly_discover_unified_numerical_algorithms.md)

</div>

<!-- RELATED:END -->
