---
title: >-
  [论文解读] Compress, Gather, and Recompute: REFORMing Long-Context Processing in Transformers
description: >-
  [NEURIPS2025][KV cache compression] 提出 REFORM 推理框架，通过"压缩—检索—重算"三阶段流水线高效处理超长上下文（百万级 token），在 RULER 和 BABILong 上相比最强基线分别提升 52% 和 34%，同时降低 30% 推理时间和 5% 峰值显存。
tags:
  - NEURIPS2025
  - KV cache compression
  - long-context inference
  - token retrieval
  - early exit
  - on-demand recomputation
---

# Compress, Gather, and Recompute: REFORMing Long-Context Processing in Transformers

**会议**: NEURIPS2025  
**arXiv**: [2506.01215](https://arxiv.org/abs/2506.01215)  
**代码**: 待确认  
**领域**: model_compression  
**关键词**: KV cache compression, long-context inference, token retrieval, early exit, on-demand recomputation

## 一句话总结

提出 REFORM 推理框架，通过"压缩—检索—重算"三阶段流水线高效处理超长上下文（百万级 token），在 RULER 和 BABILong 上相比最强基线分别提升 52% 和 34%，同时降低 30% 推理时间和 5% 峰值显存。

## 背景与动机

大语言模型在实际场景中越来越多地面临超长上下文处理需求（终身用户对话、仓库级代码理解、多模态交错序列），但 Transformer 的二次复杂度和有限的预训练窗口使直接处理百万级 token 不可行。现有方法分为两大类：

1. **循环压缩方法**（StreamingLLM、H2O、TOVA、InfiniPot）：将输入分块迭代处理，通过驱逐或压缩 KV cache 控制内存。优点是内存占用低，但在压缩过程中丢失关键信息，导致"遗忘"问题，检索精度大幅下降。
2. **随机访问方法**（InfLLM、ReAttention）：保留完整 KV cache 并按需检索相关部分。可灵活访问历史上下文，但需巨大显存（通常需要 CPU offloading），延迟显著增加，且灵活性并不一定转化为高检索性能。

两类方法各有缺陷，需要一种兼顾效率与精确检索能力的新方案。

## 核心问题

如何在有限的计算和内存预算下，对超出预训练窗口长度的极长序列实现高精度的信息检索和生成？关键矛盾在于：压缩省内存但丢信息，完整缓存保信息但太耗资源。

## 方法详解

REFORM（REcurrent chunked Forwarding with On-demand cache RecoMputation）采用两阶段推理流水线：

### 阶段一：循环分块前向 + Embedding 提取

1. **分块处理**：将长输入切分为固定大小的块（实验中使用 32k token），逐块送入模型。
2. **渐进式 KV cache 压缩**：每处理完一个块，按照 H2O 的 attention-based token eviction 策略压缩 KV cache，仅保留"重击者"（high attention score token）。压缩后重新分配 position ID 使其连续，从而突破预训练窗口限制。
3. **跨层上下文 Embedding 构建**：在压缩过程中，从选定的中间层和注意力头提取 QKV 状态，拼接构造轻量级的逐 token 检索 Embedding。关键发现是 QKV 状态的余弦相似度检索效果优于常用的 attention score，且维度更小（160 vs 5120）。
4. **Early Exit 策略**：最优检索头集中在中低层（深度 < 70%），因此无需将输入前向传播到上层，提前退出可节省计算和内存。

**Embedding 头选择**：通过合成任务（pattern matching + multi-hop QA）评估所有注意力头的检索性能（MNR 指标），选取 4 个最优头（2 个 pattern matching + 2 个 multi-hop QA）。将选中头的 embedding 各自 L2 归一化后拼接，等效于独立计算各头余弦相似度后取平均。

### 阶段二：按需 Cache 重算

1. **相关 token 识别**：利用存储的 cross-layer context embedding，计算 query token（输入末尾部分）与所有历史 token 的余弦相似度，对 query 维度做 max-pooling 得到逐 token 分数，再用 129-token 窗口 max-pooling 平滑以保持上下文连续性。
2. **Gather**：选取得分最高的 token（Mistral-Nemo 用 8k，其他模型用 16k），始终保留首尾各 256 个 token。
3. **Recompute**：将选中 token 的原始 embedding 重新送入完整模型前向传播，重建高保真 KV cache 用于生成。

这种"压缩用于检索、重算用于生成"的分离设计是 REFORM 的核心创新：压缩阶段不直接用于生成，而是仅服务于轻量级检索；生成所需的 KV cache 通过精选 token 的完整重算获得，避免了压缩导致的信息损失。

## 实验关键数据

### Needle-in-a-Haystack

Qwen2.5-7B-Instruct 在所有深度和长度（至 1M token）均达到 100% 检索准确率。

### RULER & BABILong（1M token，Mistral-Nemo）

| 方法 | RULER 1M | BABILong 1M |
|------|----------|-------------|
| InfLLM（最强基线） | 23.3 | 9.6 |
| InfiniPot | 12.0 | 8.8 |
| **REFORM** | **75.5** | **48.8** |

相比 InfLLM 提升超过 52 和 34 个百分点。Qwen2.5-7B 上 REFORM 在 1M 达到 75.1 / 58.8，同样大幅领先。

### ∞-Bench（Mistral-Nemo）

REFORM 平均 50.2%，InfLLM 37.6%，InfiniPot 24.0%。在 R.KV 子任务上 REFORM 达 88.2%（InfLLM 仅 1.0%）。

### RepoEval 代码补全（Qwen2.5-Coder-1.5B API-level）

REFORM 65.3% ES，InfLLM 61.8%，InfiniPot 59.4%。

### 多模态 MM-NIAH（Pixtral-12B）

REFORM 57.5% 平均，TOVA 52.0%，InfiniPot 53.0%。

### 效率分析（256k token，单卡 H100）

| 方法 | 推理时间 (s) | 峰值显存 (GB) |
|------|-------------|--------------|
| InfLLM | 129.14 | 51.62 |
| H2O | 41.33 | 37.85 |
| **REFORM** | **27.24** | **35.00** |

相比 InfLLM 时间降 80%、显存降 32%；相比 InfiniPot 时间降 33%、显存降 5%。

### 与 RAG 对比（RULER 300k NIAH）

REFORM 在所有四类 needle 任务上均超过 BM25 和 Dense RAG，组合使用可进一步小幅提升。

## 亮点

- **"压缩—检索—重算"三阶段解耦设计**，巧妙结合循环压缩的效率和随机访问的精确性，避免了两类方法各自的根本缺陷。
- **QKV 状态余弦相似度优于 attention score** 的发现为检索提供了更紧凑高效的 embedding，且跨层组合进一步提升效果。
- **Early Exit** 利用最优头集中在中低层的特性，自然地与 embedding 提取结合，减少无用计算。
- **模态无关**：架构级操作使其可直接用于多模态模型（Pixtral），无需修改。
- 实验覆盖极为全面：合成检索、真实 NLU、代码补全、多模态、与 RAG 对比、消融实验一应俱全。

## 局限与展望

- Gather 阶段的 token 选择依赖 embedding 质量，极端情况下可能遗漏关键 token。
- 当前实现中 token eviction 需额外计算 attention score（Flash Attention 不输出 attention weights），存在冗余计算；集成到 Flash Attention kernel 可进一步提速。
- 压缩组件直接复用 H2O，未针对 embedding 构建做专门优化，有提升空间。
- 未在音频、视频等更多模态上验证。
- 头选择基于短合成数据（8k token），虽然实验表明可泛化至百万级，但理论保证不足。

## 与相关工作的对比

| 维度 | StreamingLLM / H2O / TOVA / InfiniPot | InfLLM | REFORM |
|------|---------------------------------------|--------|--------|
| 核心思路 | 循环 KV cache 压缩 | 全 KV cache + CPU offloading 随机访问 | 压缩仅做检索 + 选中 token 重算 |
| 信息保留 | 压缩丢失严重 | 完整保留 | 选择性高保真重算 |
| 内存 | 低 | 非常高 | 低（early exit + 小 embedding） |
| 延迟 | 低 | 高（CPU↔GPU 传输） | 最低 |
| 检索精度 | 低 | 中 | 高 |
| 是否需改模型 | 否 | 否 | 否 |

与 RAG 的区别：REFORM 的检索 embedding 是在完整上下文条件下构建的（避免 context fragmentation），且无需外部检索模型，天然支持多模态。

## 启发与关联

- 核心 insight "压缩不用于生成，而用于检索；生成用重算"可推广到其他场景：例如视频理解中对帧做轻量 embedding 保存再按需解码。
- QKV 状态作为检索 embedding 的发现可用于改进 retrieval head 相关工作。
- Early exit + 检索头选择的组合策略可启发推测解码（speculative decoding）中的分层调度。
- 与 SnapKV、HOMER 等 post-hoc KV 压缩互补：REFORM 的 embedding 构建可与更先进的压缩方法结合。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 三阶段解耦和 QKV embedding 检索是有意义的新贡献
- 实验充分度: ⭐⭐⭐⭐⭐ — 合成/真实/代码/多模态/RAG 对比/消融/效率，极为全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表丰富
- 价值: ⭐⭐⭐⭐⭐ — 实用性强，性能碾压基线，即插即用无需训练

<!-- RELATED:START -->

## 相关论文

- [SEAL: Scaling to Emphasize Attention for Long-Context Retrieval](../../ACL2025/information_retrieval/seal_scaling_to_emphasize_attention_for_long-context_retrieval.md)
- [Hierarchical Document Refinement for Long-context Retrieval-augmented Generation](../../ACL2025/information_retrieval/hierarchical_document_refinement_for_long-context_retrieval-augmented_generation.md)
- [RAPID: Long-Context Inference with Retrieval-Augmented Speculative Decoding](../../ICML2025/information_retrieval/rapid_long-context_inference_with_retrieval-augmented_speculative_decoding.md)
- [Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](../../ACL2025/information_retrieval/gor_rag_long_context_summary.md)
- [The Atlas of In-Context Learning: How Attention Heads Shape In-Context Retrieval Augmentation](the_atlas_of_in-context_learning_how_attention_heads_shape_in-context_retrieval_.md)

<!-- RELATED:END -->
