---
title: >-
  [论文解读] Continuous Space-Time Video Resampling with Invertible Motion Steganography
description: >-
  [CVPR 2025][视频处理][可逆隐写] 提出可逆运动隐写术实现连续时空视频重采样，在下采样中隐藏运动信息用于高质量上采样
tags:
  - CVPR 2025
  - 视频重采样
  - 可逆网络
  - 运动隐写
  - 时空超分辨率
---

# Continuous Space-Time Video Resampling with Invertible Motion Steganography

**会议**: CVPR 2025  
**arXiv**: 待公开  
**代码**: 待确认  
**领域**: 视频处理  
**关键词**: 时空重采样, 可逆隐写, 运动保持, 视频超分辨率, 灵活采样率

## 一句话总结
提出基于可逆运动隐写术的连续时空视频重采样方法，在时空下采样过程中将运动信息隐藏在低分辨率帧中，上采样时恢复隐藏的运动细节，实现灵活的时空重采样。

## 研究背景与动机
**领域现状**：视频时空重采样需要同时进行空间和时间维度的下采样/上采样，在视频压缩、传输和显示适配中有重要应用。

**现有痛点**：时间下采样丢失运动信息导致上采样时运动模糊；空间和时间重采样率通常固定不灵活。

**本文目标** 在下采样中保留运动信息，并支持灵活的时空重采样率。

**核心 idea**：用可逆网络在下采样帧中隐藏运动信息，上采样时通过逆变换恢复。

## 方法详解

### 关键设计
1. **可逆运动编码**：将帧间运动信息通过可逆网络编码到下采样帧的不可察觉修改中。
2. **连续时空表示**：支持任意（非整数）的时空重采样率。
3. **隐写质量约束**：确保隐藏信息不影响下采样帧的视觉质量。

## 实验关键数据

### 关键发现
- 运动恢复质量显著优于无信息隐藏的插帧方法
- 支持连续的非整数重采样率
- 下采样帧的视觉质量几乎不受影响

## 亮点与洞察
- 将隐写术的思想创新性地应用于视频处理，实现信息的隐式传递
- 可逆网络保证了信息的无损往返

## 局限与展望
- 可逆网络的计算开销较大
- 在高压缩率传输后隐藏信息的鲁棒性有待验证

<!-- RELATED:START -->

## 相关论文

- [Neural Video Compression with Context Modulation](neural_video_compression_with_context_modulation.md)
- [Angular Steering: Behavior Control via Rotation in Activation Space](../../NeurIPS2025/signal_comm/angular_steering_behavior_control_via_rotation_in_activation_space.md)
- [AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation](../../CVPR2026/signal_comm/actta_rethinking_test-time_adaptation_via_dynamic_activation.md)
- [ABC-Former: Auxiliary Bimodal Cross-domain Transformer with Interactive Channel Attention](abc-former_auxiliary_bimodal_cross-domain_transformer_with_interactive_channel_a.md)
- [Radio Frequency Ray Tracing with Neural Object Representation for Enhanced RF Modeling](radio_frequency_ray_tracing_with_neural_object_representation_for_enhanced_rf_mo.md)

<!-- RELATED:END -->
