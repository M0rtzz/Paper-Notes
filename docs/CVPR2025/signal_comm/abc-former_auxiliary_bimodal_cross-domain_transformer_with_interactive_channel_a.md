---
title: >-
  [论文解读] ABC-Former: Auxiliary Bimodal Cross-domain Transformer with Interactive Channel Attention
description: >-
  [CVPR 2025][白平衡][跨域Transformer] 提出辅助双模态跨域Transformer和交互通道注意力用于sRGB图像白平衡矫正
tags:
  - CVPR 2025
  - 白平衡
  - 跨域Transformer
  - 通道注意力
  - 色温矫正
---

# ABC-Former: Auxiliary Bimodal Cross-domain Transformer with Interactive Channel Attention

**会议**: CVPR 2025  
**arXiv**: 待公开  
**代码**: 待确认  
**领域**: 图像处理 / 白平衡  
**关键词**: 白平衡, 双模态, 跨域Transformer, 通道注意力, 色彩矫正

## 一句话总结
提出 ABC-Former，利用辅助双模态信息（全局色温和局部色彩）通过跨域 Transformer 和交互通道注意力实现高质量的 sRGB 图像白平衡矫正。

## 研究背景与动机
**领域现状**：白平衡矫正旨在消除不准确色温导致的图像色偏，是相机 ISP 和后处理的核心环节。

**现有痛点**：现有方法要么只做全局色彩调整（忽略混合光照场景），要么局限于局部处理（丢失全局色温信息），效果有限。

**本文目标** 同时利用全局色温和局部色彩信息进行白平衡矫正。

**核心 idea**：用辅助双模态（全局+局部）信息通过跨域 Transformer 融合来增强白平衡矫正。

## 方法详解

### 关键设计
1. **双模态编码**：分别提取全局色温特征和局部色彩分布特征。
2. **跨域 Transformer**：在全局和局部特征域之间进行交叉注意力交互。
3. **交互通道注意力**：在通道级别自适应融合两种模态的信息。

## 实验关键数据

### 关键发现
- 在混合光照场景下显著优于仅全局或仅局部的方法
- 色温估计精度和色偏消除效果领先

## 亮点与洞察
- 全局-局部双模态融合自然地解决了混合光照场景的白平衡问题

## 局限与展望
- 对极端色温场景的泛化性有待验证
- 可以探索与相机 RAW 域白平衡的结合

<!-- RELATED:START -->

## 相关论文

- [FASA: Frequency-Aware Sparse Attention](../../ICLR2026/signal_comm/fasa_frequency-aware_sparse_attention.md)
- [Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization](../../ICML2025/signal_comm/fourier_position_embedding_enhancing_attentions_periodic_extension_for_length_ge.md)
- [Text-Guided Channel Perturbation and Pretrained Knowledge Integration for Unified Multi-Modality Image Fusion](../../AAAI2026/signal_comm/text-guided_channel_perturbation_and_pretrained_knowledge_integration_for_unifie.md)
- [Neural Video Compression with Context Modulation](neural_video_compression_with_context_modulation.md)
- [Continuous Space-Time Video Resampling with Invertible Motion Steganography](continuous_space-time_video_resampling_with_invertible_motion_steganography.md)

<!-- RELATED:END -->
