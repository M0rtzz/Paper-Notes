---
title: >-
  [论文解读] StreamingTOM: Streaming Token Compression for Efficient Video Understanding
description: >-
  [CVPR2026][视频理解][视频理解] 提出 StreamingTOM，一个无需训练的两阶段流式视频理解框架：Causal Temporal Reduction (CTR) 在 LLM 前通过因果时序选择将每帧 token 从 196 压缩到 50，Online Quantized Memory (OQM) 在 LLM 后通过 4-bit 量化和按需检索限制 kv-cache 增长，实现 15.7× 压缩比、1.2× 更低峰值显存和 2× 更快 TTFT。
tags:
  - CVPR2026
  - 视频理解
  - token compression
  - 量化
  - training-free
  - causal inference
---

# StreamingTOM: Streaming Token Compression for Efficient Video Understanding

**会议**: CVPR2026  
**arXiv**: [2510.18269](https://arxiv.org/abs/2510.18269)  
**代码**: [yige24/StreamingTOM](https://yige24.github.io/StreamingTOM)  
**领域**: 视频理解 / 流式视频问答 / Token压缩  
**关键词**: streaming video understanding, token compression, kv-cache quantization, training-free, causal inference

## 一句话总结

提出 StreamingTOM，一个无需训练的两阶段流式视频理解框架：Causal Temporal Reduction (CTR) 在 LLM 前通过因果时序选择将每帧 token 从 196 压缩到 50，Online Quantized Memory (OQM) 在 LLM 后通过 4-bit 量化和按需检索限制 kv-cache 增长，实现 15.7× 压缩比、1.2× 更低峰值显存和 2× 更快 TTFT。

## 背景与动机

1. **流式视频的双重约束**：与离线处理不同，流式视频 VLM 面临因果性（无法访问未来帧）和累积性（token 随时间无界增长）两大约束，使得 token 压缩从可选优化变为必要前提。
2. **kv-cache 无界增长**：以 LLaVA-OV-7B 为例，1 小时视频在 0.5 fps 下 kv-cache 达 18.8 GB，远超典型 GPU 显存容量，无法维持实时推理。
3. **现有方法仅管理 post-LLM**：当前训练无关的流式方法（ReKV、LiveVLM、StreamMem）仅对 LLM 之后的 kv-cache 进行驱逐/压缩，无法降低 pre-LLM prefill 的 $O(tNLd^2)$ 计算开销。
4. **离线压缩违反因果性**：成熟的离线 token 合并/剪枝方法（ToMe、DyCoke、HoliTom）依赖全局/双向注意力和未来帧信息，无法直接用于流式场景。
5. **训练方法成本高**：训练式流式方法（Flash-VStream、Dispider）需要针对特定模型的昂贵重训练，难以跨骨干网络迁移。
6. **pre-LLM 因果压缩空白**：据作者所知，此前没有训练无关的流式方法在 LLM 之前执行严格因果的 token 削减，留下了重要的效率空间。

## 方法详解

### 整体框架：两阶段流水线

StreamingTOM = OQM₁₆→₄ ∘ CTR_{N→G}，以 **group 抽象**（每帧固定 G=50 个 token 的帧对齐组）作为两阶段间的接口：

- **视觉流水线**：视觉编码器提取特征 → CTR 压缩 → 写入在线记忆
- **查询流水线**：用户问题驱动解码器 → OQM 检索相关 group → 4-bit 反量化 → 高效生成

### Stage 1: Causal Temporal Reduction (CTR)

CTR 遵循三个设计原则：严格因果（2 帧窗口）、单遍处理、固定每帧预算 G。

1. **时序相似度计算**：对相邻帧 $t$ 和 $t{-}1$ 的同位置 token 计算余弦相似度 $s_t^{(i)}$，衡量跨帧冗余。
2. **空间显著性**：复用视觉编码器的注意力分数 $\alpha_t^{(i)}$ 作为零成本副产品，通过 chunked attention 避免显存峰值。
3. **静态/动态分类**：以阈值 $\tau_c = 0.9$ 将 token 分为静态集 $\mathcal{S}_t$（高相似度，冗余）和动态集 $\mathcal{D}_t$（低相似度，新信息）。
4. **自适应预算分配**：按照静态/动态比例将 G 个名额分配为 $k_s$ 和 $k_d$，内容变化大时倾斜给动态 token。
5. **双路径处理**：
    - 动态路径：按显著性选 top-$k_d$ token（保留关键新信息）
    - 静态路径：密度聚类合并为 $k_s$ 个代表 token（去除冗余）
6. **复杂度**：每帧 $O(N + G^2)$，状态仅需前一帧特征 $O(Nd)$，不随流长度增长。

### Stage 2: Online Quantized Memory (OQM)

OQM 解决 CTR 之后 kv-cache 仍线性增长的问题：

1. **增量 group 量化**：每个 group 独立量化为 4-bit（per-head, per-channel 的 scale/offset），同时存储代表性 key $\bar{\mathbf{k}}_t$。
2. **检索-反量化范式**：查询时用 decoder state 与所有 group 的代表 key 计算余弦相似度，选 top-k 个最相关 group，仅对选中 group 做 4-bit → FP16 反量化。
3. **有界活跃显存**：总存储 $O(T \cdot G \cdot d / 4)$ 保留完整历史，活跃 kv 仅 $O(k \cdot G \cdot d)$（$k \ll T$），解码延迟不随流长度增长。

### 压缩比

综合 CTR 和 OQM：压缩比 = $4N/G = 4 \times 196/50 \approx 15.7\times$。

## 实验关键数据

### 离线长视频评测（LLaVA-OV-7B backbone）

| 方法 | VideoMME Overall | MLVU | EgoSchema | Avg |
|---|---|---|---|---|
| LLaVA-OV-7B (offline baseline) | 58.4 | 64.7 | 60.1 | 61.0 |
| +LiveVLM (training-free SOTA) | 57.3 | 66.3 | 59.0 | 60.9 |
| +StreamMem | 59.4 | 66.9 | 63.0 | 63.1 |
| **+StreamingTOM (ours)** | **59.9** | **67.9** | **63.7** | **63.8** |

### 在线流式评测（RVS benchmark，28GB 显存限制）

| 方法 | RVS-Ego Acc/Score | RVS-Movie Acc/Score | Avg Acc/Score |
|---|---|---|---|
| Flash-VStream (训练式) | 57.0 / 4.0 | 53.1 / 3.3 | 55.0 / 3.6 |
| StreamMem | 57.6 / 3.8 | 52.7 / 3.4 | 55.2 / 3.6 |
| **StreamingTOM** | **58.3 / 3.9** | **53.2 / 3.5** | **55.8 / 3.7** |

### 效率指标

- **kv-cache 压缩比**：15.7×
- **峰值显存**：相比 LiveVLM 降低 1.2×
- **TTFT**：相比 LiveVLM 加速 2×
- **1 小时视频 kv-cache**：18.8 GB → 1.2 GB
- **显存增长**：16-512 帧仅从 16.0 GB → 16.7 GB（亚线性）
- **吞吐量**：长序列稳定在约 20 tokens/s

### 消融实验

| Token数 | 量化位数 | 压缩比 | VideoMME Overall |
|---|---|---|---|
| 40 | 4-bit | 5.1% | 58.9 |
| **50** | **4-bit** | **6.4%** | **59.9** |
| 60 | 4-bit | 7.7% | 59.3 |
| 50 | 2-bit | 3.2% | 58.5 |

- 50 token 是最优平衡点：过少（40）丢失关键细节，过多（60）在固定显存下减少时序覆盖
- 4-bit 量化优于 2-bit，精度-压缩比最优

## 亮点

1. **首创因果 pre-LLM token 压缩**：填补了训练无关流式方法中 pre-LLM 压缩空白，将 prefill 复杂度从 $O(tNLd^2)$ 降至 $O(tGLd^2)$。
2. **Group 抽象设计优雅**：固定大小的帧对齐 group 同时服务于 CTR 输出和 OQM 存储/检索，保证时序一致性和可预测延迟。
3. **完全即插即用**：无需训练，可直接应用于 LLaVA-OV 等不同骨干网络。
4. **实际部署友好**：单张 A6000 即可运行，batch-agnostic，显存增长亚线性。
5. **双阶段互补**：CTR 降计算、OQM 降显存，两者缺一不可，组合效果远超单阶段。

## 局限与展望

1. **固定 G 可能非最优**：所有帧使用相同的 50 token 预算，对信息密度差异大的帧（关键帧 vs 静态帧）不够灵活。
2. **仅验证单一骨干**：实验主要基于 LLaVA-OV-7B，未在更大模型（如 72B）或其他架构上验证。
3. **2 帧窗口限制**：CTR 的因果窗口仅看相邻两帧，对缓慢渐变场景可能累积误差。
4. **代表性 key 的检索质量**：OQM 用均值 key 做检索，可能对细粒度时序推理不够精确。
5. **未评估多模态音频流**：仅考虑视觉流，实际流式应用通常伴随音频流。

## 与相关工作的对比

| 维度 | StreamingTOM | LiveVLM/StreamMem | DyCoke/HoliTom | Flash-VStream |
|---|---|---|---|---|
| Pre-LLM 压缩 | ✅ CTR | ❌ | ✅ (非因果) | ✅ (需训练) |
| Post-LLM 管理 | ✅ OQM 4-bit | ✅ kv-cache 驱逐 | ❌ | ✅ (需训练) |
| 因果约束 | ✅ 严格 | ✅ | ❌ 需未来帧 | ✅ |
| 训练需求 | 无 | 无 | 无 | 需重训练 |
| 压缩比 | 15.7× | ~4× | ~4× | N/A |

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次在训练无关流式方法中引入因果 pre-LLM token 压缩，group 抽象统一两阶段
- 实验充分度: ⭐⭐⭐⭐ — 覆盖离线/在线两类 benchmark，效率分析详尽，消融完整
- 写作质量: ⭐⭐⭐⭐⭐ — 问题定义清晰，公式推导严谨，pipeline 图直观
- 价值: ⭐⭐⭐⭐ — 解决流式视频 VLM 实际部署的显存瓶颈，即插即用实用性强

<!-- RELATED:START -->

## 相关论文

- [FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)
- [Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding](question-guided_visual_compression_with_memory_feedback_for_long-term_video_unde.md)
- [Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing](attend_before_attention_efficient_and_scalable_video_understanding_via_autoregre.md)
- [FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](../../ICLR2026/video_understanding/floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)
- [Unified Spatiotemporal Token Compression for Video-LLMs at Ultra-Low Retention](unified_spatiotemporal_token_compression_for_video-llms_at_ultra-low_retention.md)

<!-- RELATED:END -->
