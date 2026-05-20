---
title: >-
  [论文解读] HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding
description: >-
  [ACL 2026][视频理解][流式视频] 本文提出 HERMES，基于对 MLLM 解码器层级注意力偏好的机制性分析，将 KV 缓存概念化为层级记忆框架（浅层=感觉记忆、中层=工作记忆、深层=长期记忆），实现免训练的高效流式视频理解，在减少 68% 视频 token 的条件下仍保持或提升准确率…
tags:
  - "ACL 2026"
  - "视频理解"
  - "流式视频"
  - "KV缓存管理"
  - "层级记忆"
  - "实时响应"
  - "免训练"
---

# HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding

**会议**: ACL 2026  
**arXiv**: [2601.14724](https://arxiv.org/abs/2601.14724)  
**代码**: [GitHub](https://github.com/haowei-freesky/HERMES)  
**领域**: 视频理解 / 流式推理  
**关键词**: 流式视频, KV缓存管理, 层级记忆, 实时响应, 免训练

## 一句话总结

本文提出 HERMES，基于对 MLLM 解码器层级注意力偏好的机制性分析，将 KV 缓存概念化为层级记忆框架（浅层=感觉记忆、中层=工作记忆、深层=长期记忆），实现免训练的高效流式视频理解，在减少 68% 视频 token 的条件下仍保持或提升准确率，TTFT 延迟仅 <30ms，比前 SOTA 快 10 倍。

## 研究背景与动机

**领域现状**：MLLM 在离线视频理解上取得显著进展，但扩展到流式视频输入仍面临挑战——需要同时维持理解性能、实时响应和低 GPU 内存开销。现有流式方法分为外部记忆（将视频内容存为描述或补丁，查询时检索）和内部记忆（直接在 KV 缓存中管理）。

**现有痛点**：(1) 外部记忆方法在查询到达时需要检索和多模态预填充，延迟高且缺乏端到端连贯性；(2) ReKV 和 LiveVLM 等缓存方法将视频段卸载到 CPU/磁盘，查询时需额外检索操作，延迟仍然显著；(3) 现有方法使用粗粒度的淘汰策略（如 FIFO 统一应用于所有层），忽略了不同层的注意力偏好差异。

**核心矛盾**：KV 缓存天然是模型内在的潜在记忆，适合流式场景的免训练管理，但现有方法未利用层间注意力模式的差异——不同层对视频信息的"记忆方式"不同。

**本文目标**：设计一种基于层级注意力分析的 KV 缓存管理方法，无需训练即可插入现有 MLLM，实现真正的实时流式视频问答。

**切入角度**：对 LLaVA-OV-7B 的 28 层解码器进行注意力可视化分析，发现三种截然不同的层级记忆模式。

**核心 idea**：浅层展现强烈的近因偏好（感觉记忆），用指数衰减管理；深层关注帧级"锚点 token"（长期记忆），用注意力权重管理；中层在两者间过渡（工作记忆），用插值管理。加上跨层平滑和位置重索引确保一致性。

## 方法详解

### 整体框架

HERMES 包含三个组件：(1) **层级 KV 缓存管理**——根据层级类型（浅/中/深）使用不同的 token 重要性评分和淘汰策略；(2) **跨层记忆平滑**——防止独立层级淘汰导致的跨层不一致性；(3) **位置重索引**——淘汰后重新映射位置编码以保持连续性。推理时直接复用压缩后的 KV 缓存，用户提问时无需任何额外计算。

### 关键设计

1. **层级 KV 缓存管理**:

    - 功能：根据不同层的注意力特性实施差异化的 token 淘汰策略
    - 核心思路：浅层用指数遗忘曲线 $S_i^l = \alpha_i^l \cdot e^{-k\Delta t_i}$ 评估 token 重要性（越新越重要）；深层用注意力权重 $S_i^l = \alpha_i^l \cdot W_i^l$（基于伪查询的注意力）；中层用层依赖权重 $\omega_l$ 插值近因分数和注意力分数 $S_i^l = (1-\omega_l) A_i^l + \omega_l R_i^l$
    - 设计动机：注意力可视化清晰表明不同层有不同的记忆功能——统一的 FIFO 或注意力淘汰策略无法同时满足所有层的需求

2. **跨层记忆平滑（Cross-Layer Memory Smoothing）**:

    - 功能：防止各层独立淘汰导致同一 token 在不同层的存留不一致
    - 核心思路：相邻层间共享部分淘汰决策，确保同一视频 token 在多层中的保留/淘汰具有一定一致性
    - 设计动机：独立层级管理会导致视觉记忆碎片化——同一帧的信息在某些层保留、另一些层被淘汰，破坏了端到端推理的连贯性

3. **位置重索引（Position Re-Indexing）**:

    - 功能：淘汰 token 后重新映射位置编码，保持 RoPE 的正确性
    - 核心思路：每次淘汰后将保留 token 的位置重索引为连续 $[0, |M|)$，避免位置编码不连续带来的注意力计算异常
    - 设计动机：直接删除中间 token 会导致位置编码跳跃，影响基于位置的注意力机制

### 损失函数 / 训练策略

完全免训练方法。基于 Ebbinghaus 遗忘曲线理论和认知心理学的层级记忆模型设计。使用通用引导提示（generic guidance prompt）作为伪查询来计算深层注意力权重。

## 实验关键数据

### 主实验

**流式视频基准（LLaVA-OV-7B）**

| 方法 | StreamingBench | EgoSchema | MVBench | Video-MME | 平均 |
|------|---------------|-----------|---------|-----------|------|
| 全量（无压缩） | 53.2 | 58.1 | 69.3 | 61.8 | 60.6 |
| ReKV | 51.8 | 55.2 | 67.1 | 59.4 | 58.4 |
| StreamMem | 52.1 | 56.8 | 68.5 | 60.1 | 59.4 |
| **HERMES** | **59.3** | **58.9** | **69.8** | **62.4** | **62.6** |

### 消融实验

**效率对比（单 A800 GPU）**

| 方法 | TTFT（ms） | GPU 内存 | token 减少 |
|------|----------|---------|-----------|
| 全量 | ~3000+ | 线性增长 | 0% |
| ReKV | ~1500 | 需 CPU 内存 | ~50% |
| **HERMES** | **<30** | **恒定** | **68%** |

### 关键发现

- HERMES 在减少 68% video token 的条件下，在流式基准上反而提升 11.4%——证明冗余 token 的去除实际改善了推理质量
- TTFT < 30ms 且 GPU 内存恒定，随输入帧数增加无 OOM 风险——查询到达时零额外计算
- 层级记忆模型在多种 MLLM 上泛化——不仅限于 LLaVA-OV
- 浅层注意力的近因偏好符合 Ebbinghaus 遗忘曲线，深层注意力的锚点模式间隔恰好等于单帧 token 数（196）

## 亮点与洞察

- 从认知心理学借鉴的层级记忆概念与 Transformer 层级的注意力模式精确对应——这不仅是类比，而是有定量注意力分析支持的发现
- 零额外延迟的设计对实时应用至关重要——ReKV 等方法虽然减少了存储但查询时仍需检索
- 免训练+即插即用的特性使其可直接应用于现有 MLLM，降低了实用门槛

## 局限与展望

- 层级边界（浅/中/深）的划分依赖于特定模型的分析，不同架构可能需要重新确定
- 伪查询替代真实用户查询可能在特定场景下产生偏差
- 仅在视频流式场景验证，对文本流式或多模态流式的适用性未探索
- 指数遗忘率 $k$ 和插值参数需要手动设定

## 相关工作与启发

- **vs ReKV/LiveVLM**: 需要 CPU 卸载+检索操作，延迟高；HERMES 直接复用 GPU 上的 KV 缓存
- **vs StreamMem**: 利用聊天模板 token 引导压缩但缺乏细粒度管理；HERMES 基于层级注意力分析实现精细管理
- **vs StreamingLLM**: attention sink 机制保留初始 token 但忽略层间差异；HERMES 利用层级特化实现更智能的淘汰

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 层级记忆概念化和基于注意力分析的差异化管理策略非常新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 多个流式基准+效率分析+注意力可视化+消融
- 写作质量: ⭐⭐⭐⭐⭐ 从机制分析到方法设计的逻辑链非常清晰
- 价值: ⭐⭐⭐⭐⭐ 实时流式视频理解的实用解决方案，TTFT 10 倍加速

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](../../CVPR2026/video_understanding/fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)
- [\[NeurIPS 2025\] InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding](../../NeurIPS2025/video_understanding/infinipot-v_memory-constrained_kv_cache_compression_for_streaming_video_understa.md)
- [\[ACL 2026\] Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding](response-g1_explicit_scene_graph_modeling_for_proactive_streaming_video_understa.md)
- [\[CVPR 2026\] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](../../CVPR2026/video_understanding/videoarm_agentic_reasoning_over_hierarchical_memory_for_long-form_video_understa.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](../../CVPR2026/video_understanding/streamingtom_streaming_token_compression_for_efficient_video_understanding.md)

</div>

<!-- RELATED:END -->
