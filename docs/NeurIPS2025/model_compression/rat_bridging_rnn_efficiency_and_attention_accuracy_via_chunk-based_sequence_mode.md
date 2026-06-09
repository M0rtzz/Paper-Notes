---
title: >-
  [论文解读] RAT: Bridging RNN Efficiency and Attention Accuracy via Chunk-based Sequence Modeling
description: >-
  [NeurIPS 2025][模型压缩][高效序列建模] 提出 RAT（Recurrence And aTtention），一种基于 Chunk 的中间架构——在 Chunk 内使用线性 RNN 建模局部依赖、Chunk 间使用 softmax 注意力实现全局访问…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "高效序列建模"
  - "Chunk-based架构"
  - "注意力机制"
  - "长上下文"
  - "线性复杂度"
---

# RAT: Bridging RNN Efficiency and Attention Accuracy via Chunk-based Sequence Modeling

**会议**: NeurIPS 2025  
**arXiv**: [2507.04416](https://arxiv.org/abs/2507.04416)  
**代码**: [GitHub](https://github.com/CLAIRE-Labo/RAT)  
**领域**: 模型压缩  
**关键词**: 高效序列建模, Chunk-based架构, RNN-Attention混合, 长上下文, 线性复杂度

## 一句话总结

提出 RAT（Recurrence And aTtention），一种基于 Chunk 的中间架构——在 Chunk 内使用线性 RNN 建模局部依赖、Chunk 间使用 softmax 注意力实现全局访问。L=16 时单层解码速度提升 9 倍、最大吞吐量提升 10 倍，且性能与标准注意力持平；与滑动窗口注意力交替使用的混合变体在几乎所有 benchmark 上最优。

## 研究背景与动机

Transformer 依赖全 self-attention 导致训练和推理的计算量随序列长度二次增长，限制了长上下文场景的可扩展性。近年来涌现了大量高效替代方案，包括 Mamba、线性注意力、状态空间模型等，但它们的核心问题在于：

**RNN 类模型的本质缺陷**：将整个序列历史压缩到固定大小的整体状态中，随序列增长不可避免地出现**记忆退化**（memory degradation）。这意味着远距离的精确信息检索变得困难，尤其在噪声上下文中。

**注意力的过度分配**：标准注意力对每个 token 保留全访问权限，但大量局部依赖关系实际上可以用更轻量的方式处理，注意力在局部上下文中被"大材小用"了。

**中间设计的缺失**：RNN（L=T，完全压缩）和注意力（L=1，零压缩）是两个极端，缺少一种能在效率和精度之间灵活插值的中间架构。

RAT 的核心洞察是：**部分压缩 + 全局直接访问**。Chunk 内用 RNN 短序列压缩不会有明显信息损失，Chunk 间用 softmax 注意力保证远距离信息可直接检索。通过调整 Chunk 大小 $L$，RAT 在 RNN 和注意力之间形成一个连续谱。

## 方法详解

### 整体框架

给定长度为 $T$ 的序列，RAT 将其分为 $C = T/L$ 个 Chunk，每个 Chunk 包含 $L$ 个 token。Chunk 内部用线性 RNN（EMA 门控）递归地压缩 key 和 value，得到 Chunk 级表示；Chunk 间用标准 softmax 注意力对这些 Chunk 级表示做全局交互。整个过程如下：

（1）输入 token 在 Chunk 内递归更新 → （2）取每个 Chunk 的末状态作为 Chunk 级 KV → （3）当前 query 对所有 Chunk 级 KV 做注意力 → （4）输出门控

### 关键设计

1. **Chunk 内线性 RNN（Intra-chunk Recurrence）**：采用简单的线性递归形式，对 value 和 key 分别做 EMA 式门控聚合：

$$\tilde{v}_{c,l} = g_{c,l} \odot \tilde{v}_{c,l-1} + (1 - g_{c,l}) \odot v_{c,l}$$

$$\tilde{k}_{c,l} = g_{c,l} \odot \tilde{k}_{c,l-1} + (1 - g_{c,l}) \odot k_{c,l}$$

其中 $g_{c,l}$ 是逐维度的遗忘门（sigmoid 激活），通过输入的线性投影计算得到。设计动机：Chunk 长度短（如 L=16），短序列 RNN 不会有记忆退化问题，且比注意力更高效。选择最简单的线性 RNN 形式是为了突出核心思想，但框架兼容更复杂的 RNN 变体。

2. **Chunk 间 softmax 注意力（Inter-chunk Attention）**：每个位置 $(c,l)$ 的 query 对所有前续 Chunk 的末状态 $\tilde{K}_{:,-1}$ 和当前 Chunk 的累积状态 $\tilde{k}_{c,l}$ 计算注意力：

$$y_{c,l} = f([q_{c,l}\tilde{K}_{:,-1}^\top; q_{c,l}\tilde{k}_{c,l}^\top])[\tilde{V}_{:,-1}; \tilde{v}_{c,l}]$$

最后通过输出门 $z_{c,l}$ 调制结果。Chunk 间注意力操作在长度 $C$ 的序列上进行（而非原始长度 $T$），FLOPs 减少了 $L$ 倍。

3. **参数分配与位置编码**：

    - **参数共享**：总参数预算 $4D^2$。实验发现将更多参数分配给 RNN 门控（而非注意力 QK）效果更好。最终方案是共享 Q/K 投影，省出的参数给门控。虽然 QK 共享，但遗忘门在逐维度上产生不同的门控 key $\tilde{k}$，因此不会退化为单头注意力。
    - **Chunk 级 RoPE**：位置编码基于 Chunk 索引（而非原始 token 位置），RNN 本身已编码 Chunk 内位置信息。这还改善了长度泛化能力，因为需要编码的位置数（Chunk 数）远小于序列长度。

4. **混合架构 RAT-SWA**：RAT 与滑动窗口注意力（SWA，窗口大小 1024）交替使用。两者高度互补——SWA 把大部分计算集中在固定窗口内处理局部交互，RAT 则将注意力保留给全局访问，用轻量 RNN 更高效地处理局部。

### 损失函数 / 训练策略

在 FineWeb-Edu 数据集上从头预训练 1.3B 模型，100B token，学习率 8e-4 余弦衰减至 1e-6，全局 batch size 2M token，4K 上下文窗口。训练实现不依赖自定义 CUDA/Triton 内核：Chunk 内递归用 PyTorch associative scan（scan 深度从 $O(\log T)$ 降至 $O(\log L)$），Chunk 间注意力用 PyTorch flex attention（支持自定义 mask 和返回 softmax 分母）。因果 mask 问题通过 online softmax 分解为两项分别计算后合并。

## 实验关键数据

### 主实验 — 1.3B 模型效率与性能

| 模型 | 最大吞吐量 (tok/s) | Val PPL | CSR Avg Acc | LongBench SQA F1 | LongBench Summ R-L | LongBench Code |
|------|-------------------|---------|-------------|-------------------|--------------------|----|
| Attention | 3,052 | 7.61 | 56.9 | 18.2 | 19.5 | 23.9 |
| RNN | — | 7.82 | 55.8 | — | — | — |
| RAT(L=16) | **31,170** (10.2×) | 7.67 | 56.7 | **19.6** | **20.2** | 17.4 |
| Attention-SWA | 4,605 | 7.61 | 57.1 | 17.4 | 19.4 | 21.7 |
| **RAT(L=16)-SWA** | **13,582** (4.4×) | **7.57** | **58.0** | 18.8 | 19.5 | **28.2** |

### 消融实验（200M 模型, Book 数据集）

| 配置 | PPL (4K) | PPL (32K) | 说明 |
|------|----------|-----------|------|
| 更多参数给 RNN 门控 | **13.42** | 14.05 | 最优配置 |
| 更多参数给 QK | 13.82 (+0.40) | 14.52 (+0.47) | 门控更重要 |
| 原始 RoPE (token 位置) | 13.52 (+0.10) | 14.35 (+0.30) | Chunk 级 RoPE 更好 |
| Chunk 级 RoPE | **13.42** | **14.05** | 尤其长序列改善明显 |

### 扩展性：7B/13B 模型吞吐量

| 模型规模 | RAT(L=16) | Attention | 加速比 |
|----------|-----------|-----------|--------|
| 1.3B | 31,170 | 3,152 | 10.2× |
| 7B | 10,103 | 983 | 10.3× |
| 13B | 5,749 | 534 | **10.8×** |

### 关键发现

- **效率**：RAT(L=16) 训练 100K 序列比注意力快 7 倍，4K 位置生成快 9 倍，最大吞吐量高 10 倍。且模型越大加速比越高（13B 达 10.8×），因为注意力在大模型中 GPU 利用率更低。
- **性能中间态**：预训练 PPL 上，RAT 精确位于 RNN 和注意力之间——Attention 7.61，RAT(L=16) 7.67，RNN 7.82。增大 Chunk 从 L=16→256，PPL 平滑增大。
- **长上下文优势**：在 LongBench 的 QA 和摘要任务上，RAT(L=16) 多项超越全注意力（例如 NarrativeQA 14.5 vs 12.3，QA Avg 19.6 vs 18.2），因为 Chunk 间注意力避免了 RNN 的长程记忆退化。
- **混合架构最优**：RAT(L=16)-SWA 在 commonsense reasoning（+1）、代码补全（+4）、困难 QA（+4）、摘要（+1）上全面超越所有变体，同时保持 ~4× 吞吐提升。

## 亮点与洞察

1. **Chunk 大小 L 作为连续旋钮**：L=1 → Attention，L=T → RNN，中间值提供效率-精度的平滑权衡。这一设计极为优雅，将两类看似不相关的架构统一在同一框架下。
2. **"注意力在局部被浪费"**：这个洞察被 RAT-SWA 的实验强力验证——用 RNN 处理局部、注意力处理全局的组合不仅更快，而且更准。
3. **无自定义内核**：纯 PyTorch 实现（associative scan + flex attention），降低了工程门槛，且兼容张量并行和上下文并行。
4. **KV 缓存减少**：RAT 仅需存储 Chunk 级 KV（比全注意力少 16 倍），显著减少 OOM 风险。

## 局限与展望

- Chunk 边界处可能存在信息不连续——当一个语义单元跨越两个 Chunk 时，RNN 只能部分捕捉。
- 仅使用最简单的线性 RNN（EMA 门控），未探索更强的 RNN 变体（如非线性 RNN、2D 递归）。
- 短序列（<4K）时训练速度略慢于注意力，因为 flex attention 在少量 Chunk 上 GPU 并行不充分。
- 目前验证规模限于 1.3B（主实验），7B/13B 仅报告了吞吐量，缺少对应的精度数据。

## 相关工作与启发

- 与 Mamba/Mamba2/GatedDeltaNet 等固定状态模型的对比中，RAT 的记忆容量随序列增长（Chunk 数增加），这是本质区别。
- RAT-SWA 的"全局用注意力、局部用 RNN"思路与 Samba、Griffin 等混合模型一致，但 RAT 的实现更简洁。
- Chunk 级 RoPE 的位置编码策略对多层级架构的位置编码设计有参考价值。

## 评分

- **新颖性**: ⭐⭐⭐⭐ RNN+Attention 的 Chunk 融合架构简洁有效，L 作为连续插值参数的设计优雅
- **实验充分度**: ⭐⭐⭐⭐⭐ 7 项短上下文 reasoning + 11 项 LongBench + 4 项 SFT + 9 项检索合成任务，覆盖极全
- **写作质量**: ⭐⭐⭐⭐⭐ 动机推导清晰，从 RNN/Attention 对比自然引出 Chunk 设计
- **价值**: ⭐⭐⭐⭐⭐ 10× 吞吐提升 + 无自定义内核 + 精度持平，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Linear Attention for Efficient Bidirectional Sequence Modeling](linear_attention_for_efficient_bidirectional_sequence_modeling.md)
- [\[NeurIPS 2025\] Adaptive Prediction-Powered AutoEval with Reliability and Efficiency Guarantees](adaptive_predictionpowered_autoeval_with_reliability_and_eff.md)
- [\[NeurIPS 2025\] Order-Level Attention Similarity Across Language Models: A Latent Commonality](order-level_attention_similarity_across_language_models_a_latent_commonality.md)
- [\[CVPR 2025\] LALIC: Linear Attention Modeling for Learned Image Compression](../../CVPR2025/model_compression/linear_attention_modeling_for_learned_image_compression.md)
- [\[NeurIPS 2025\] Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry for Long-Context LLMs](homogeneous_keys_heterogeneous_values_exploiting_local_kv_cache_asymmetry_for_lo.md)

</div>

<!-- RELATED:END -->
