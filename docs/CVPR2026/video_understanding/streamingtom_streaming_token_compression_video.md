---
title: >-
  [论文解读] StreamingTOM: Streaming Token Compression for Efficient Video Understanding
description: >-
  [CVPR 2026][视频理解][流式视频理解] 首个同时解决流式视频VLM中pre-LLM prefill和post-LLM KV-cache两个效率瓶颈的免训练框架，实现15.7倍压缩和有界活跃内存。
tags:
  - CVPR 2026
  - 视频理解
  - 流式视频理解
  - token压缩
  - KV-cache优化
  - 因果时序缩减
  - 4-bit量化记忆
---

# StreamingTOM: Streaming Token Compression for Efficient Video Understanding

**会议**: CVPR 2026  
**arXiv**: [2510.18269](https://arxiv.org/abs/2510.18269)  
**代码**: [项目页](https://yige24.github.io/StreamingTOM)  
**领域**: 多模态VLM / 视频理解  
**关键词**: 流式视频理解, token压缩, KV-cache优化, 因果时序缩减, 4-bit量化记忆  

## 一句话总结

首个同时解决流式视频VLM中pre-LLM prefill和post-LLM KV-cache两个效率瓶颈的免训练框架，实现15.7倍压缩和有界活跃内存。

## 研究背景与动机

**流式视频理解与离线处理有根本区别**：面临两个独特约束——(1) 因果性：只能看到已有帧，无法利用未来帧；(2) 累积性：token数量随时间无界增长，内存和延迟不断恶化。以LLaVA-OV-7B为例，处理1小时视频的KV-cache达18.8GB，远超GPU容量。

**现有免训练方法只管post-LLM的KV-cache**（如eviction策略），但完全忽略了pre-LLM的prefill开销——每帧所有 $N$ 个视觉token都需经过完整transformer前向传播，这是延迟主要来源。更关键的是，现有离线token压缩方法需利用全局/未来帧信息，**违反了流式场景的因果约束**。

**因此，因果约束下的pre-LLM token缩减与post-LLM内存管理的联合优化是未被探索的关键问题**。核心洞察：有效的流式压缩必须在LLM之前、在严格因果约束下进行——post-LLM方法无法减少已产生的prefill计算。

## 方法详解

### 整体框架

两阶段免训练框架：Stage 1 因果时序缩减（CTR）处理pre-LLM瓶颈——将每帧 $N$ 个token缩减为固定预算 $G$ 个；Stage 2 在线量化记忆（OQM）处理post-LLM瓶颈——将KV-cache以4-bit存储并按需检索。两者通过帧对齐的group抽象协调。

### 关键设计

1. **因果时序缩减（CTR）**:

    - 功能：在严格因果约束下将每帧视觉token从 $N$ 压缩到固定预算 $G$
    - 核心思路：只用相邻两帧窗口，通过余弦相似度将token分为静态/动态集合，按比例分配预算。静态token用DPC聚类合并，动态token按注意力显著性选择
    - 设计动机：固定预算 $G$ 保证可预测延迟；自适应分配让静止场景多压缩、运动场景多保留

2. **在线量化记忆（OQM）**:

    - 功能：将post-LLM的KV-cache以4-bit格式存储，按需检索反量化
    - 核心思路：保留帧对齐group结构（每组 $G$ 个token对应一帧），查询时检索最相关的 $k$ 个group反量化为FP16参与注意力计算。活跃KV-cache有上界，不随视频长度增长
    - 设计动机：4-bit量化降存储4倍；group级检索保持时序完整性，避免token碎片化

3. **统一压缩比分析**:

    - 功能：量化端到端压缩效果
    - 核心思路：prefill从 $O(TNLd^2)$ 降到 $O(TGLd^2)$，存储从 $O(TN \cdot d \cdot 16)$ bit降到 $O(TG \cdot d \cdot 4)$ bit，组合压缩比 $4N/G \approx 15.7\times$（$N=196, G=50$）
    - 设计动机：预算 $G$ 同时控制计算和存储，实现双重压缩

### 损失函数 / 训练策略

完全免训练方法，可直接应用于现有VLM（如LLaVA-OV-7B）。

## 实验关键数据

### 主实验

| 指标 | StreamingTOM | LiveVLM (之前SOTA) | 提升 |
|------|-------------|-------------------|------|
| KV-cache压缩比 | 15.7× | - | - |
| 峰值内存 | - | - | 1.2×更低 |
| TTFT | - | - | 2×更快 |
| 离线平均准确率 | 63.8% | ~61% | +2.8% |
| RVS准确率 | 55.8% | ~54% | +1.8% |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅CTR | 内存未控 | prefill加速但KV-cache仍无界 |
| 仅OQM | 延迟未改 | 存储压缩但prefill不变 |
| CTR+OQM | 双优 | 两阶段缺一不可 |
| 不同预算G | G=50最优 | G过低损精度，G过高压缩不足 |

### 关键发现

- 1小时视频KV-cache从18.8GB降到1.2GB，有界增长使无限长视频流理论上可行
- CTR的双路处理关键：纯聚类或纯选择都不如混合策略
- 在离线和流式基准上同时达到免训练方法SOTA

## 亮点与洞察

- 核心贡献在于识别"pre-LLM和post-LLM是两个独立瓶颈需分别解决"这一洞察。帧对齐group抽象连接两阶段，使token缩减和存储优化解耦但协调——设计优雅且有实际意义。

## 局限与展望

- 4-bit量化在极端精度要求场景可能引入质量损失
- 基于相邻帧余弦相似度的分类在快速运动场景可能遗漏关键变化
- 固定预算G不随内容复杂度自适应
- 未在训练式方法上验证CTR/OQM作为即插即用模块的效果

## 相关工作与启发

- **vs LiveVLM**: 只做KV-cache管理（post-LLM），StreamingTOM首次在pre-LLM层面也做优化
- **vs FastV/TokenPacker**: 面向单图像/离线视频，需全局信息，不满足流式因果约束

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次同时解决双层效率瓶颈，因果token缩减+4-bit量化记忆结合新颖
- 实验充分度: ⭐⭐⭐⭐ 离线和流式基准均SOTA，效率指标全面
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，数学推导完整
- 价值: ⭐⭐⭐⭐⭐ 解决流式视频VLM实际部署核心痛点

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)
- [\[CVPR 2026\] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)
- [\[CVPR 2026\] AutoGaze: Attend Before Attention — Efficient and Scalable Video Understanding via Autoregressive Gazing](autogaze_attend_before_attention_efficient_video.md)
- [\[ICLR 2026\] FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](../../ICLR2026/video_understanding/floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)
- [\[CVPR 2026\] Unified Spatiotemporal Token Compression for Video-LLMs at Ultra-Low Retention](unified_spatiotemporal_token_compression_for_video-llms_at_ultra-low_retention.md)

<!-- RELATED:END -->
