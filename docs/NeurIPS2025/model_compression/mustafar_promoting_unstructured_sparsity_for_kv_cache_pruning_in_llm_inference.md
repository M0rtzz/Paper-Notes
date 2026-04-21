---
title: >-
  [论文解读] MUSTAFAR: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference
description: >-
  [NeurIPS 2025][模型压缩][KV缓存压缩] 提出 MUSTAFAR 框架，系统性地证明了非结构化稀疏性在 KV 缓存剪枝中的优越性（Key 和 Value 均可达 70% 稀疏度且不损精度），并设计了基于 bitmap 的稀疏格式和自定义注意力内核，实现了端到端推理吞吐量 2.23 倍加速。
tags:
  - NeurIPS 2025
  - 模型压缩
  - KV缓存压缩
  - 非结构化稀疏
  - 大语言模型推理
  - 注意力加速
  - 稀疏格式
---

# MUSTAFAR: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference

**会议**: NeurIPS 2025  
**arXiv**: [2505.22913](https://arxiv.org/abs/2505.22913)  
**代码**: [GitHub](https://github.com/dhjoo98/mustafar)  
**领域**: 模型压缩  
**关键词**: KV缓存压缩, 非结构化稀疏, 大语言模型推理, 注意力加速, 稀疏格式

## 一句话总结

提出 MUSTAFAR 框架，系统性地证明了非结构化稀疏性在 KV 缓存剪枝中的优越性（Key 和 Value 均可达 70% 稀疏度且不损精度），并设计了基于 bitmap 的稀疏格式和自定义注意力内核，实现了端到端推理吞吐量 2.23 倍加速。

## 研究背景与动机

随着大语言模型处理越来越长的序列，KV 缓存的内存开销已成为扩展上下文长度的关键瓶颈。现有压缩技术包括量化、低秩近似、token 驱逐和结构化剪枝，但在剪枝方面存在以下核心问题：

**结构化剪枝的局限性**：此前的 KV 缓存剪枝工作（如 ThinK）局限于结构化模式（按通道移除），这严重限制了可达到的稀疏度。ThinK 在 Key 缓存上仅能达到约 50% 的结构化稀疏度，对 Value 缓存更是只能承受 30% 的稀疏度。

**Value 缓存难以压缩**：Value 缓存的激活分布相对均匀，没有明显的通道级异常值，使得结构化剪枝极易导致精度崩塌。

**非结构化稀疏的计算挑战**：虽然非结构化稀疏理论上能保留更多重要元素，但由于稀疏模式不规则，难以在 GPU 上高效利用，此前缺乏合适的稀疏计算方案。

MUSTAFAR 的核心洞察是：放弃对稀疏模式的任何约束（即采用非结构化稀疏），可以在更高稀疏度下保持模型精度。关键在于解决两个问题——找到合适的剪枝策略，以及设计能高效利用非结构化稀疏的计算内核。

## 方法详解

### 整体框架

MUSTAFAR 由两部分组成（如 Figure 1）：(1) 绿色部分——剪枝算法，探索 Key/Value 缓存的最优剪枝策略；(2) 粉色部分——自定义稀疏注意力内核，基于 bitmap 稀疏格式实现对压缩后 KV 缓存的高效计算。

### 关键设计

1. **Key 缓存剪枝策略**：Key 缓存展现出明显的通道级异常值（某些通道的数值远大于其他通道）。基于此观察，采用 **per-token magnitude-based pruning**（按 token 方向、基于幅值的剪枝），这样可以有效保留每个 token 中的异常值通道元素。剪枝得分公式为：

$$S = |K| \odot \text{broadcast}\left(\sum_{t=T}^{T+31} |Q_t|\right)$$

其中 output-aware 方法用后续 32 个 query 的 L1 累积与 key 元素相乘得到剪枝得分。但实验发现，简单的 per-token 幅值剪枝已经能略超结构化剪枝甚至接近 output-aware 方法的效果。

2. **Value 缓存剪枝策略**：Value 缓存的分布更均匀，没有通道级异常值。作者系统比较了 (per-channel/per-token) × (magnitude/output-aware) 四种组合。关键发现是：对于 Value 缓存，**per-token magnitude-based pruning 等价于 per-token output-aware pruning**，因为注意力计算 $\text{AttentionScore} \times \text{Value}$ 中，同一 token 的所有 Value 元素被同一个注意力分数相乘，因此幅值本身就反映了对输出的贡献。最终统一采用 per-token magnitude-based pruning 处理 Key 和 Value 缓存。

3. **Bitmap 稀疏格式与自定义注意力内核**：

    - **稀疏格式**：扩展 Coruscant 的 bitmap 格式，将剪枝后的 KV 缓存按 $1 \times 64$ 列分块，每块用 64 位 bitmap 表示非零位置，加上 tile offset 定位起始非零元素，实现最大压缩比。
    - **SpMV 内核**：解码阶段的注意力操作（Query × Key^T 和 Attention Score × Value）本质上是内存受限的矩阵-向量乘法。自定义 CUDA 内核采用 "load-as-compressed, compute-as-dense" 范式——从全局内存加载压缩数据到寄存器，解压到共享内存，然后执行分块稠密计算。
    - **混合计算**：解码阶段注意力被重构为两部分：压缩 KV 缓存的 SpMV + 局部窗口（最近 32 个 token）的稠密 MV，通过 online softmax 组合两部分结果。

### 损失函数 / 训练策略

MUSTAFAR 是一种无需训练（training-free）的方法。剪枝和压缩在运行时完成：prefill 阶段生成的 KV 缓存在 decode 开始前被剪枝和压缩；decode 阶段新生成的 KV 缓存在局部窗口内保持稠密，超出窗口后再剪枝压缩。与 FlashAttention prefill 完全兼容。

## 实验关键数据

### 主实验（LongBench）

| 模型 / 配置 | K稀疏度 | V稀疏度 | LongBench 均分 | 对比基线 |
|-------------|---------|---------|---------------|---------|
| Llama-3-8B Dense | 0% | 0% | 43.19 | — |
| ThinK (结构化) | 50% | 0% | 38.53 | 结构化 SOTA |
| **MUSTAFAR** | **50%** | **0%** | **42.84** | +4.31 vs ThinK |
| ThinK (结构化) | 70% | 0% | 26.55 | 精度崩塌 |
| **MUSTAFAR** | **70%** | **0%** | **41.55** | +14.98 vs ThinK |
| **MUSTAFAR** | **70%** | **70%** | **40.96** | 仍优于ThinK@50% |
| Mistral-7B Dense | 0% | 0% | 42.65 | — |
| **MUSTAFAR** | **70%** | **70%** | **40.95** | 仅降1.70 |

### 消融实验（Value 缓存剪枝策略对比）

| 剪枝方式 | V稀疏度=50% | V稀疏度=70% | 说明 |
|----------|------------|------------|------|
| ThinK 结构化 | 38.45 | 30.60 | 结构化在 V 缓存上效果极差 |
| Per-channel magnitude | 42.50 | 41.69 | 通道方向幅值 |
| Per-channel output-aware | 42.84 | 42.67 | 加入注意力分数加权 |
| **Per-token magnitude** | **43.04** | **42.78** | **最简单最好，无需额外计算** |

### 关键发现

- **非结构化 vs 结构化**：在 70% Key 稀疏度下，非结构化剪枝（41.55）比结构化 ThinK（26.55）高出 15 分，后者已完全崩塌。
- **KV 双缓存同时剪枝**：Key 和 Value 同时 70% 稀疏（40.96）仍优于 ThinK 仅 Key 50% 结构化（38.53），这是此前不可想象的。
- **与正交方法兼容**：MUSTAFAR 可与 KIVI 量化联合使用（如 K70%+V70%+4bit 量化），也可与 H2O token 驱逐联合，进一步压缩。
- **端到端加速**：KV 缓存压缩至稠密推理的 45%，吞吐量提升最高 2.23 倍。SpMV 内核的加速完全覆盖了运行时剪枝和压缩的开销。

## 亮点与洞察

1. **打破结构化剪枝的思维定式**：此前 KV 缓存剪枝领域默认需要结构化模式，本文令人信服地证明了"放弃约束反而更好"这一反直觉结论。
2. **Key vs Value 缓存的差异化分析**：Key 缓存有通道异常值（适合 per-token 保留异常值），Value 缓存分布均匀（per-token magnitude 即 output-aware），这种细致分析为后续工作提供了重要参考。
3. **系统级解决方案**：不仅提出剪枝算法，还封装了完整的格式设计和内核实现，使非结构化稀疏真正可落地。

## 局限与展望

- 当前仅在 7B-8B 模型上验证，缺少在更大规模模型（70B+）上的实验。
- Bitmap 稀疏格式的 overhead 在极短序列时可能不划算，短上下文场景的收益有限。
- 自定义 CUDA 内核目前仅支持特定 GPU 架构（RTX 6000 ADA），移植到其他硬件需要额外工作。
- 局部窗口大小（32 token）是固定超参数，未探索自适应窗口大小的可能性。

## 相关工作与启发

- 与 KIVI（量化）和 H2O（token 驱逐）的兼容性实验表明，多种压缩技术可以叠加使用，这为构建"压缩栈"提供了思路。
- Bitmap 稀疏格式借鉴自 Coruscant，后者用于 LLM 权重投影层的 SpMM，本文将其扩展到注意力计算的 SpMV 场景。
- Per-token magnitude pruning 的简洁性令人印象深刻——最简单的方法往往最鲁棒，这值得在其他压缩场景中验证。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 非结构化稀疏用于 KV 缓存的系统性探索是首次，bitmap 格式+自定义内核的组合有工程创新
- **实验充分度**: ⭐⭐⭐⭐ 三个模型（Llama-2/3, Mistral）、LongBench 全任务、正交方法兼容性验证，但缺少更大模型
- **写作质量**: ⭐⭐⭐⭐ 逻辑清晰，Figure 1 提供了很好的全局视角
- **价值**: ⭐⭐⭐⭐⭐ 直接解决 LLM 推理的核心瓶颈，方法简单有效可落地

<!-- RELATED:START -->

## 相关论文

- [ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [Inference-Time Hyper-Scaling with KV Cache Compression](inference-time_hyper-scaling_with_kv_cache_compression.md)
- [KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments](keydiff_key_similarity-based_kv_cache_eviction_for_long-context_llm_inference_in.md)
- [RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](../../ICML2025/model_compression/rocketkv_accelerating_long-context_llm_inference_via_two-stage_kv_cache_compress.md)
- [Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning](twilight_adaptive_attention_sparsity_with_hierarchical_top-p_pruning.md)

<!-- RELATED:END -->
