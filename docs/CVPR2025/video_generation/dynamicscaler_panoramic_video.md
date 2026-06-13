---
title: >-
  [论文解读] DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes
description: >-
  [CVPR 2025][视频生成][全景视频生成] 提出 DynamicScaler，一个无需训练的统一框架，通过偏移移位去噪器和全局运动引导，实现任意分辨率/宽高比的全景动态场景生成，支持 360° 视场、长时长和可循环视频。 领域现状：沉浸式 AR/VR 应用需要高质量全景场景合成，但视频扩散模型受限于固定分辨率和宽高比…
tags:
  - "CVPR 2025"
  - "视频生成"
  - "全景视频生成"
  - "偏移移位去噪"
  - "360度全景"
  - "无训练"
  - "可扩展"
---

# DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes

**会议**: CVPR 2025  
**arXiv**: [2412.11100](https://arxiv.org/abs/2412.11100)  
**代码**: [https://dynamic-scaler.pages.dev/new](https://dynamic-scaler.pages.dev/new)  
**领域**: 视频生成 / 全景生成  
**关键词**: 全景视频生成, 偏移移位去噪, 360度全景, 无训练, 可扩展

## 一句话总结

提出 DynamicScaler，一个无需训练的统一框架，通过偏移移位去噪器和全局运动引导，实现任意分辨率/宽高比的全景动态场景生成，支持 360° 视场、长时长和可循环视频。

## 研究背景与动机

**领域现状**：沉浸式 AR/VR 应用需要高质量全景场景合成，但视频扩散模型受限于固定分辨率和宽高比。

**现有痛点**：MultiDiffusion 等方法用重叠窗口生成全景图但计算开销大；360DVD 在等距柱状投影空间微调导致分辨率低和插值伪影；4K4DGen 的固定窗口限制运动范围。

**核心矛盾**：需要在保持运动连贯性的同时实现空间可扩展性，且不能因窗口边界产生接缝。

**本文目标**：无需微调，生成任意分辨率/宽高比的全景动态场景，确保空间和时序连贯性。

**核心 idea**：通过每步偏移去噪窗口位置，在步间创建"重叠"区域，同步整个全景的去噪过程。

## 方法详解

### 整体框架

两阶段管线：低分辨率阶段建立粗运动结构（OSD + 可选全景投影），上采样阶段用更多移位窗口 + GMG 生成高分辨率精细全景。

### 关键设计

1. **偏移移位去噪器（OSD）**:

    - 功能：实现无缝全景视频去噪
    - 核心思路：在每个去噪步骤中，将去噪窗口在垂直和水平方向偏移，步间的"重叠"同步了内容和运动。水平方向将全景视为环形（左右边界连通），确保 360° 无缝过渡
    - 设计动机：避免重叠窗口的高计算开销，通过步间偏移隐式创建重叠效果

2. **全局运动引导（GMG）**:

    - 功能：确保高分辨率生成的全局运动一致性
    - 核心思路：先生成低分辨率视频捕捉整体运动结构，上采样后加噪作为高分辨率生成的初始化，引导高分辨率阶段保持全局运动的同时细化局部细节
    - 设计动机：早期去噪步构建布局时 OSD 的同步效果还未累积足够，需要全局先验引导

3. **时序偏移移位与循环生成**:

    - 功能：生成超长时长和可循环的视频
    - 核心思路：将 OSD 机制扩展到时序维度——将长视频分成帧片段窗口，在步间偏移片段窗口位置。循环模式下将首尾帧视为连通，窗口可跨越首尾边界
    - 设计动机：突破视频扩散模型的帧数限制（通常 16 帧），实现持续运动

### 损失函数 / 训练策略

完全无训练方法，基于现有视频扩散模型（如 I2V、T2V）进行推理时修改。VRAM 消耗恒定，不随输出分辨率增长。

## 实验关键数据

### 主实验

| 指标 | DynamicScaler | 360DVD |
|------|--------------|--------|
| CLIP-Score | **0.302** | 0.293 |
| Image Quality | **0.583** | 0.436 |
| Dynamic Degree | **0.783** | 0.412 |
| Motion Smoothness | **0.963** | 0.917 |
| Q-Align(V) | **0.613** | 0.532 |

### 关键发现

- 恒定 VRAM 消耗使得可以生成任意大分辨率
- 可从 16 帧扩展到 80+ 帧，画面质量保持一致
- 环形边界处理确保 360° 无缝

## 亮点与洞察

- OSD 的核心思路极其简洁——只是在步间移位窗口
- 统一框架覆盖普通全景、360° 全景、长视频、循环视频
- 完全无训练，可即插即用于现有视频扩散模型

## 局限与展望

- 复杂运动模式可能因窗口间信息传递不足而不协调
- 360° 模式在极地区域存在重叠处理的额外复杂性
- 生成质量仍受底层视频扩散模型能力限制

## 评分

- 新颖性：8/10 — OSD 简洁优雅
- 技术深度：7/10 — 方法简单但有效
- 实验充分度：7/10 — 定量指标有限
- 写作质量：7/10 — 结构清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling](spatiotemporal_skip_guidance_for_enhanced_video_diffusion_sampling.md)
- [\[CVPR 2025\] VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step](videoscene_distilling_video_diffusion_model_to_generate_3d_scenes_in_one_step.md)
- [\[CVPR 2025\] SketchVideo: Sketch-Based Video Generation and Editing](sketchvideo_sketch-based_video_generation_and_editing.md)
- [\[CVPR 2025\] Pathways on the Image Manifold: Image Editing via Video Generation](pathways_on_the_image_manifold_image_editing_via_video_generation.md)
- [\[CVPR 2025\] Tora: Trajectory-Oriented Diffusion Transformer for Video Generation](tora_trajectory-oriented_diffusion_transformer_for_video_generation.md)

</div>

<!-- RELATED:END -->
