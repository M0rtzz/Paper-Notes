---
title: >-
  [论文解读] MonarchAttention: Zero-Shot Conversion to Fast, Hardware-Aware Structured Attention
description: >-
  [NeurIPS 2025][LLM/NLP][注意力机制] 提出 MonarchAttention，利用 Monarch 矩阵的结构化特性，通过 softmax 变分形式的交替优化，实现 $\Theta(N\sqrt{N}d)$ 复杂度的注意力近似，无需额外训练即可零样本替换预训练 Transformer 的注意力层，同时在 GPU 上相比 FlashAttention-2 实现 1.4×–8.2× 的加速。
tags:
  - NeurIPS 2025
  - LLM/NLP
  - 注意力机制
  - Monarch matrices
  - structured matrices
  - hardware-aware
  - zero-shot conversion
---

# MonarchAttention: Zero-Shot Conversion to Fast, Hardware-Aware Structured Attention

**会议**: NeurIPS 2025  
**arXiv**: [2505.18698](https://arxiv.org/abs/2505.18698)  
**代码**: [GitHub](https://github.com/cjyaras/monarch-attention)  
**领域**: llm_nlp  
**关键词**: efficient attention, Monarch matrices, sub-quadratic attention, structured matrices, hardware-aware, zero-shot conversion

## 一句话总结

提出 MonarchAttention，利用 Monarch 矩阵的结构化特性，通过 softmax 变分形式的交替优化，实现 $\Theta(N\sqrt{N}d)$ 复杂度的注意力近似，无需额外训练即可零样本替换预训练 Transformer 的注意力层，同时在 GPU 上相比 FlashAttention-2 实现 1.4×–8.2× 的加速。

## 研究背景与动机

Transformer 的核心注意力机制具有 $\Theta(N^2 d)$ 的二次时间复杂度，这是长序列训练和推理的关键瓶颈。现有的亚二次注意力方法主要分为：

**低秩方法**（线性注意力、Performer 等）：硬件友好但不适合作为预训练模型的直接替换，因为注意力矩阵常呈强对角线结构

**稀疏方法**（LSH、固定稀疏掩码）：数据依赖的稀疏支持难以高效实现于 GPU

**低秩+稀疏**：组合方法虽然准确度提升，但开销较大

这些方法要么需要从头训练/微调（不可迁移），要么在实际 GPU 上无法获得加速（理论复杂度与实际性能差距大）。MonarchAttention 的核心动机是同时实现**可迁移性**（零样本替换）和**硬件效率**（利用 GPU 张量核心）。

## 方法详解

### 整体框架

MonarchAttention 的目标是找到一个 Monarch 矩阵 $\mathbf{M} \in \mathbb{R}^{N \times N}$ 使得 $\mathbf{M} \approx \text{softmax}(\mathbf{Q}\mathbf{K}^\top)$，然后高效计算 $\mathbf{O} = \mathbf{M}\mathbf{V}$。

**Monarch 矩阵**的定义：给定 $N = m \times b$（通常取 $m = b = \sqrt{N}$），Monarch 矩阵 $\mathbf{M} = \mathbf{P}^\top \mathbf{B}$，其中 $\mathbf{P}$ 是转置置换矩阵，$\mathbf{B}$ 是块秩一矩阵。每个块 $\mathbf{B}_{jk} = \mathbf{L}_{jk}\mathbf{R}_{kj}^\top$，存储只需 $\Theta(N\sqrt{N})$ 空间，矩阵乘法只需 $\Theta(N\sqrt{N}d)$ 操作。

### 关键设计

**Softmax 变分形式**：利用 softmax 的变分定义：

$$\text{softmax}(\mathbf{z}) = \arg\max_{\mathbf{a} \in \Delta^N} \langle \mathbf{a}, \mathbf{z} \rangle + H(\mathbf{a})$$

其中 $H(\mathbf{a}) = -\sum_i \mathbf{a}_i \log \mathbf{a}_i$ 是 Shannon 熵。将注意力矩阵的计算重构为优化问题：

$$\sigma(\mathbf{Q}\mathbf{K}^\top) = \arg\max_{\mathbf{A} \in \Delta^{N \times N}} f(\mathbf{A}; \mathbf{Q}, \mathbf{K}) := \langle \mathbf{A}, \mathbf{Q}\mathbf{K}^\top \rangle + H(\mathbf{A})$$

**低维结构利用**：当 $\mathbf{A}$ 具有 Monarch 结构时，目标函数可分解为多个独立的小规模子问题：

$$f(\mathbf{P}^\top \mathbf{B}; \mathbf{Q}, \mathbf{K}) = \sum_{j,k} f(\mathbf{B}_{jk}; \tilde{\mathbf{Q}}_j, \mathbf{K}_k)$$

由于块秩一结构，熵项也可分离计算，每个子问题只需 $\Theta((m+b)d)$ 操作。

**交替最大化**：固定 $\mathbf{L}$ 时目标对 $\mathbf{R}$ 为凹函数（反之亦然），因此可通过 KKT 条件得到闭式更新：

$$\mathbf{R}^{(t)} = \text{softmax}_i(\mathbf{Z}_R^{(t)}), \quad \mathbf{L}^{(t)} = \text{softmax}_k(\mathbf{Z}_L^{(t)})$$

初始化 $\mathbf{L}_{jkl}^{(0)} = \delta_{kl}$（单位矩阵），经过 $T$ 步交替优化后得到 $\mathbf{M}^{(T)} \approx \sigma(\mathbf{Q}\mathbf{K}^\top)$。

### 损失函数 / 训练策略

MonarchAttention 是一种**推理时**的优化方法，不涉及训练损失函数。其核心是通过变分目标的交替最大化来近似注意力矩阵。

**IO 优化实现**：不在 HBM 中物化 $\mathbf{L}$、$\mathbf{R}$，只维护状态变量 $\alpha_R, \alpha_L, c_R, c_L$（额外 $\Theta(Nd)$ 内存）。所有中间值仅在片上 SRAM 中物化，实现类似 FlashAttention 的 IO 节省，但有效序列长度为 $\sqrt{N}$，消除了沿序列长度的分块需要，最优 IO 复杂度 $\Theta(Nd)$，优于 FlashAttention 的最坏情况 $O(N^2 d^2 / S)$。

## 实验关键数据

### 主实验

| 任务 | 模型 | 方法 | FLOPs 减少 | 性能损失 |
|------|------|------|-----------|---------|
| 图像分类 (ImageNet) | ViT-B (87M) | MonarchAttention | 80% | Top-5 accuracy 仅降 5% |
| 图像分类 (ImageNet) | ViT-B | MonarchAttention | 50% | 性能持平 |
| 问答 (SQuAD) | RoBERTa-B (125M) | MonarchAttention | 60% | F1 仅降 10 分 |
| 问答 (SQuAD) | RoBERTa-B | MonarchAttention | 35% | 性能持平 |
| 摘要 (BookSum) | BART-B (139M) | MonarchAttention (N=8192) | vs softmax N=2048 类似 FLOPs | ROUGE-1 +0.75, ROUGE-L +0.5 |

**图像生成 (DiT-XL 675M，ImageNet)**：

| 替换层 | 方法 | FLOPs (×10⁹) | FID ↓ | sFID ↓ |
|--------|------|-------------|-------|--------|
| 全部 | Nyströmformer | 3.30 | 5.97 | 13.47 |
| 全部 | MonarchAttention | 3.44 | **2.82** | **5.09** |
| 前半部分 | Nyströmformer | 5.88 | 8.17 | 19.01 |
| 前半部分 | MonarchAttention | 5.95 | **0.39** | **0.66** |
| 后半部分 | Nyströmformer | 5.88 | 6.76 | 13.58 |
| 后半部分 | MonarchAttention | 5.95 | **1.98** | **3.36** |

### 消融实验

**速度基准测试 (NVIDIA A40 GPU)**：

| 序列长度 N | MonarchAttention vs FlashAttention-2 加速比 |
|-----------|-------------------------------------------|
| 256 | 1.4× |
| 4096 | 4.5× |
| 16384 | 8.2× |

不同迭代步数 $T$ 对 ViT 精度的影响：$T \in \{1, 2, 3\}$ 越大精度越高但 FLOPs 越多，$T=1$ 已能实现良好的精度-效率权衡。

### 关键发现

1. MonarchAttention 在所有任务上显著优于低秩基线（Performer、Nyströmformer），尤其在图像生成任务上差距巨大
2. 在摘要任务中，MonarchAttention 实现了比 softmax 注意力更好的 ROUGE vs FLOPs 权衡（通过高效处理更长序列）
3. 替换 DiT 前半部分层时 FID 仅 0.39，表明前层的注意力更易被 Monarch 矩阵近似
4. 短序列（N=256）也能获得 1.4× 加速，这得益于全融合 Triton kernel 实现

## 亮点与洞察

1. **变分形式的巧妙利用**：将 softmax 近似问题转化为约束优化，避免了直接计算 $N \times N$ 注意力矩阵
2. **块秩一结构的熵分离性**：Monarch 矩阵的块秩一性质使得熵项可分离计算，从 $\Theta(mb)$ 降至 $\Theta(m+b)$，是理论上的关键贡献
3. **硬件感知设计**：Monarch 矩阵通过批量密集矩阵乘法（batched matmuls）利用 GPU 张量核心，而非 FFT 类算法的不友好内存访问模式
4. **IO 复杂度严格优于 FlashAttention**：在最坏情况下 $\Theta(Nd)$ vs $O(N^2 d^2 / S)$

## 局限与展望

1. **不适用于自回归生成**：无法用于 decoder 的逐 token 生成，因为不存在完整的注意力矩阵可供近似
2. **因果掩码整合困难**：如何将因果掩码高效融入 MonarchAttention 尚不明确
3. **均匀块大小分配**：Monarch 矩阵对每个块分配相同参数，但注意力矩阵不同区域的复杂度可能不同
4. **可扩展性**：对于极长序列，$\sqrt{N}d > S$（SRAM 大小）时仍需沿序列分块

## 相关工作与启发

- **FlashAttention**：MonarchAttention 可视为 FlashAttention 的块稀疏版推广，每个优化步骤可写成类 FlashAttention 计算
- **Monarch Mixer**：虽然也用 Monarch 矩阵做 token 混合，但非数据依赖，MonarchAttention 保留了注意力的数据依赖性
- **$\alpha$-entmax**：比 softmax 产生更稀疏的注意力矩阵，可能更适合块秩一近似，是有前景的未来方向
- 该方法可用于扩散语言模型（非自回归）和预填充阶段的加速

## 评分

- ⭐⭐⭐⭐ **创新性**：将 softmax 变分形式与 Monarch 结构化矩阵结合，实现亚二次注意力近似的思路非常优雅
- ⭐⭐⭐⭐ **实验充分性**：覆盖视觉（ViT、DiT）、语言（RoBERTa、BART）多种架构和任务，说服力强
- ⭐⭐⭐⭐⭐ **实用价值**：零样本替换 + 实际 GPU 加速 + 开源代码，工程价值极高
- ⭐⭐⭐ **局限性**：不适用于当前最主流的自回归 LLM 推理场景，限制了影响力

**总评**: ⭐⭐⭐⭐ (4/5) — 理论漂亮、实验扎实、工程实用的高质量工作。核心局限在于无法用于自回归解码，但对 prefill、编码器模型和扩散模型仍有重要价值。

<!-- RELATED:START -->

## 相关论文

- [Zero-Shot Belief: A Hard Problem for LLMs](../../ACL2025/llm_nlp/zero-shot_belief_a_hard_problem_for_llms.md)
- [Spectral Conditioning of Attention Improves Transformer Performance](spectral_conditioning_of_attention_improves_transformer_performance.md)
- [Strassen Attention, Split VC Dimension and Compositionality in Transformers](strassen_attention_split_vc_dimension_and_compositionality_in_transformers.md)
- [Unifying Attention Heads and Task Vectors via Hidden State Geometry in In-Context Learning](unifying_attention_heads_and_task_vectors_via_hidden_state_geometry_in_in-contex.md)
- [BIPro: Zero-shot Chinese Poem Generation via Block Inverse Prompting Constrained Generation Framework](../../ACL2025/llm_nlp/bipro_zero-shot_chinese_poem_generation_via_block_inverse_prompting_constrained_.md)

<!-- RELATED:END -->
