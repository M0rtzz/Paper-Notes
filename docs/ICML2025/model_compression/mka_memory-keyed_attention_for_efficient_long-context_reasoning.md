---
title: >-
  [论文解读] MKA: Memory-Keyed Attention for Efficient Long-Context Reasoning
description: >-
  [ICML2025][模型压缩][KV cache 压缩] 提出 Memory-Keyed Attention (MKA)，将 KV 缓存组织为三级分层记忆（局部/会话/长期），通过可学习路由门动态分配注意力；加速版 FastMKA 在注意力计算前融合记忆源，实现训练吞吐量达 MLA 的 5 倍、解码延迟降至 MLA 的 54%，perplexity 仅损失约 1%。
tags:
  - "ICML2025"
  - "模型压缩"
  - "KV cache 压缩"
  - "长上下文注意力"
  - "分层记忆"
  - "动态路由"
  - "高效推理"
---

# MKA: Memory-Keyed Attention for Efficient Long-Context Reasoning

**会议**: ICML2025  
**arXiv**: [2603.20586](https://arxiv.org/abs/2603.20586)  
**作者**: Dong Liu, Yanxuan Yu, Ben Lengerich, Ying Nian Wu (UCLA, Columbia, UW-Madison)
**代码**: 未公开  
**领域**: 模型压缩  
**关键词**: KV cache 压缩, 长上下文注意力, 分层记忆, 动态路由, 高效推理

## 一句话总结

提出 Memory-Keyed Attention (MKA)，将 KV 缓存组织为三级分层记忆（局部/会话/长期），通过可学习路由门动态分配注意力；加速版 FastMKA 在注意力计算前融合记忆源，实现训练吞吐量达 MLA 的 5 倍、解码延迟降至 MLA 的 54%，perplexity 仅损失约 1%。

## 研究背景与动机

长上下文 LLM（128K–1M tokens）的核心瓶颈在于 KV cache 的内存占用与访问延迟：

- **内存开销巨大**：LLaMA-7B 在 32K 上下文下 KV cache 约占 15.8 GB（A800 GPU），KV 读取耗时 11.3 ms，占推理延迟 50% 以上。
- **现有方案的不足**：
    - MQA/GQA：共享 KV head 减少冗余，但表示能力受限。
    - MLA（DeepSeek）：低秩分解压缩 KV，但不区分不同时间尺度的记忆，缺乏对异构记忆源的灵活调度。
    - Token 驱逐方法（DynamicKV、PyramidKV）：不可逆丢弃信息。
- **核心观察**：不同 query token 对近期/中期/远期上下文的依赖程度不同，**静态统一的 KV 缓存策略是次优的**。需要一种能按"记忆时间尺度"动态路由注意力的机制。

## 方法详解

### 1. 三级分层记忆架构

灵感来自计算机存储层次结构（SRAM → HBM → DRAM），MKA 将注意力记忆分为三级：

| 层级 | 名称 | 功能 | 类比 |
|------|------|------|------|
| L1 | Local Memory | 当前窗口 token 的标准因果注意力 | SRAM（片上） |
| L2 | Session Memory | 因果前缀摘要（低秩汇总或 EMA） | HBM |
| L3 | Long-term Memory | 基于向量化哈希的历史记忆库检索 | DRAM |

- **L2 的因果性保证**：$M_2[t] = \text{Summary}(X_{[:,1:t,:]})$，仅利用过去 token 的摘要，避免信息泄露。
- **L3 的检索机制**：通过语义分块 + 向量化哈希索引，从历史 attention block 中召回 $R \ll T$ 个最相关块，实现摊销次线性复杂度。

### 2. 动态路由门

对每个 query token，学习路由权重 $\lambda = \text{softmax}(\text{MLP}(q)) \in \mathbb{R}^{B \times S \times 3}$，用于加权融合三级记忆的注意力输出：

$$O_h = \sum_{\ell=1}^{3} \lambda_\ell \odot a_\ell$$

其中 $a_\ell = \text{softmax}(q_h \cdot k_\ell^{h\top}) \cdot v_\ell^h$ 是第 $\ell$ 级记忆的注意力输出。路由权重逐 token、逐层动态计算。

### 3. FastMKA（Route-Fused MKA）

MKA 需要对三级记忆分别做注意力计算（3 次 attention），开销较大。FastMKA 的核心改进是**先融合、再做单次注意力**：

$$X_{\text{fused}} = \sum_{\ell=1}^{3} \lambda_\ell \odot L_\ell$$

$$K, V = X_{\text{fused}} W_k, \quad X_{\text{fused}} W_v$$

然后用融合后的 KV 做一次标准因果注意力。这样：
- 内核启动数从 9 降至 3（Table 11 数据）
- 只需一条注意力路径，兼容标准 Transformer 管线
- 缓存的是**融合后的路由 KV**，而非原始 token KV，进一步节省带宽

### 4. Block-MKA 与数值稳定性

- **分块计算**：将 Q/K/V 划分为 $T = N/B$ 个 block，在 L1（SRAM）中做局部 softmax，L2（HBM）存中间结果，L3（DRAM）做哈希召回。
- **在线 max-shift**：递推更新全局最大值 $\mu^{(\ell)}$，用 $\exp(\mu^{(\ell-1)} - \mu^{(\ell)})$ 对历史累积量做修正，保证低精度/长序列下的数值稳定性（与 FlashAttention 的 scan trick 一致）。

### 5. 理论复杂度

总运行时间：

$$\mathcal{O}(BTd + BRd) \quad \text{with } R \ll T$$

其中 $B$ 为 block size，$T = N/B$，$R$ 为 L3 召回块数。相比全注意力 $\mathcal{O}(N^2 d)$ 为次二次复杂度。

## 实验设置与主要结果

### 实验设置

- **模型**：Qwen2.5-7B/14B（GQA）、Llama 3.1-8B（GQA）、DeepSeek-V3（MLA）
- **数据**：WikiText-2（train 36,718 句）；LongBench、RULER 长上下文 benchmark
- **硬件**：NVIDIA A800 80GB；7B 单卡，14B 用 4-8 卡 TP
- **序列长度**：4K–256K tokens
- **训练**：1 epoch fine-tune，AdamW，bf16 + FlashAttention-2

### 核心结果

**Table 1：Qwen2.5-7B, 16K 上下文**

| 方法 | PPL ↓ | 训练时间 (s) ↓ | 解码 (ms/tok) ↓ |
|------|-------|----------------|-----------------|
| MHA | 3.31 | 6234.7 | 21.4 |
| GQA | 3.28 | 5012.4 | 18.6 |
| MLA | 3.22 | 4456.9 | 12.8 |
| **FastMKA** | **3.26** | **1248.3** | **8.4** |

FastMKA 训练时间仅为 MLA 的 28%，解码延迟为 MLA 的 66%，PPL 仅高 0.04。

**Table 2：训练吞吐量 (tokens/s)**

| 方法 | 4K | 32K | 128K | 256K |
|------|-----|------|------|------|
| MLA | 468 | 342 | 212 | 148 |
| FastMKA | 1847 | 1453 | 1032 | 742 |
| **加速比** | **3.94×** | **4.25×** | **4.87×** | **5.01×** |

加速比随序列长度增长而增大，与次二次复杂度的理论预期一致。

**Table 3：解码延迟 (ms/tok)**

| 方法 | 4K | 32K | 128K | 256K |
|------|-----|------|------|------|
| MLA | 8.7 | 16.4 | 32.7 | 48.9 |
| FastMKA | 6.2 | 10.3 | 18.4 | 26.3 |
| **加速比** | **1.40×** | **1.59×** | **1.78×** | **1.86×** |

**Table 5：KV Cache 内存（128K, Qwen2.5-7B）**

| 方法 | KV Cache (GB) | HBM BW (GB/s) | 利用率 |
|------|---------------|---------------|--------|
| MHA | 18.7 | 1240 | 78.2% |
| MLA | 8.9 | 1087 | 68.5% |
| FastMKA | **6.2** | **1324** | **83.5%** |

FastMKA KV cache 比 MHA 减少 66.8%，且因融合 KV 张量的连续内存访问模式，HBM 带宽利用率反而更高。

**跨模型泛化（Table 6, 32K）**

| 模型 | 方法 | PPL | 训练 tok/s | 解码 ms/tok |
|------|------|-----|-----------|-------------|
| Qwen2.5-14B | MLA | 3.06 | 184 | 21.8 |
| Qwen2.5-14B | FastMKA | 3.10 | 642 | 13.6 |
| Llama 3.1-8B | MLA | 3.13 | 294 | 17.9 |
| Llama 3.1-8B | FastMKA | 3.17 | 1078 | 11.2 |
| DeepSeek-V3 | MLA | 3.08 | – | 18.4 |
| DeepSeek-V3 | FastMKA | 3.11 | – | 12.7 |

**长上下文 Benchmark**

- LongBench (128K)：FastMKA 平均 54.5 vs MLA 55.0，差距仅 0.5 分
- RULER Passkey (128K)：FastMKA 73.4% vs MLA 74.8%，差距 1.4%

## 局限与展望

1. **PPL 略有损失**：FastMKA 用效率换精度，PPL 始终略高于 MLA（约 1-2%）。对精度要求极高的场景（如代码生成），这种 trade-off 可能不可接受。
2. **实验规模有限**：仅在 WikiText-2 上 fine-tune 1 epoch，未在大规模预训练中验证。7B/14B 的结论能否推广到 70B+ 模型存疑。
3. **L3 长期记忆的实际效果有限**：从消融实验看，L3 的贡献相对较小，且依赖外部哈希索引结构，增加工程复杂度。缓存内容结束后似乎未提供完整的 L3 消融数据。
4. **训练成本**：虽然吞吐快，但路由 MLP 的引入增加了参数量和梯度计算，论文未详细讨论总 FLOPs 对比。
5. **基准偏弱**：未与近期更强的 KV 压缩方法（如 KIVI、Gear、SnapKV）做对比。

## 相关工作与启发

- **MLA (DeepSeek-V2)**：低秩分解压缩 KV，是本文最直接的对比基线。FastMKA 在此基础上引入分层路由。
- **FlashAttention**：提供了 IO-aware tiled softmax 的基础，MKA 的 Block-MKA 算法直接借鉴其 online softmax 技巧。
- **Transformer-XL / Compressive Transformer**：早期分层记忆工作，但难以扩展到 LLM 规模。
- **PERK**：将长上下文存入模型权重而非 KV cache，思路互补。
- **路由 Transformer / MoE**：路由思想的来源，但本文将路由应用于记忆层级选择而非 FFN 专家选择。

## 个人点评

**优点**：
- 分层记忆 + 动态路由的设计直觉清晰，类比计算机存储层次有说服力
- FastMKA 的"先融合再注意力"思路简洁优雅，工程友好
- 实验覆盖了多模型、多序列长度，趋势一致
- 效率提升显著（5× 训练加速、1.8× 解码加速），对实际部署有吸引力

**不足**：
- 本质上 FastMKA 的融合操作可能丢失了分层记忆的细粒度信息，论文对此缺乏分析
- 仅 1 epoch fine-tune 这种设置对 PPL 对比的公平性存疑
- 缓存文本被截断，无法看到完整的消融实验（Table 9 仅出现标题）

## 评分
- 新颖性: ⭐⭐⭐⭐ (分层记忆+路由门的组合有新意，FastMKA 的融合大幅简化计算)
- 实验充分度: ⭐⭐⭐⭐ (多模型多长度覆盖好，但缺少更强 baseline 对比和完整消融)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，理论推导完整，伪代码详尽)
- 价值: ⭐⭐⭐⭐ (5× 训练加速在长上下文部署中有实际价值，但需更大规模预训练验证)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models](lacache_ladder-shaped_kv_caching_for_efficient_long-context_modeling_of_large_la.md)
- [\[ICML 2025\] Core Context Aware Transformers for Long Context Language Modeling](core_context_aware_transformers_for_long_context_language_modeling.md)
- [\[ICML 2026\] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](../../ICML2026/model_compression/token_sparse_attention_efficient_long-context_inference_with_interleaved_token_s.md)
- [\[ACL 2025\] Efficient Long Context Language Model Retrieval with Compression](../../ACL2025/model_compression/efficient_long_context_language_model_retrieval_with_compression.md)
- [\[ICML 2025\] ParallelComp: Parallel Long-Context Compressor for Length Extrapolation](parallelcomp_parallel_long-context_compressor_for_length_extrapolation.md)

</div>

<!-- RELATED:END -->
