---
title: >-
  [论文解读] VACE: All-in-One Video Creation and Editing
description: >-
  [ICCV 2025][统一框架] 提出VACE统一视频创建和编辑框架，通过Video Condition Unit(VCU)将文本/图像/视频/掩码统一为条件输入，结合Context Adapter注入任务概念到DiT模型，首次在单一视频DiT中同时支持参考生成、视频编辑、掩码编辑及其自由组合。
tags:
  - ICCV 2025
  - 统一框架
  - DiT
  - Video Condition Unit
  - 可控生成
  - 视频编辑
---

# VACE: All-in-One Video Creation and Editing

**会议**: ICCV 2025  
**arXiv**: [2503.07598](https://arxiv.org/abs/2503.07598)  
**代码**: [Project](https://ali-vilab.github.io/VACE-Page/)  
**领域**: 视频生成 / 视频编辑  
**关键词**: 统一框架, DiT, Video Condition Unit, 可控生成, 视频编辑  

## 一句话总结

提出VACE统一视频创建和编辑框架，通过Video Condition Unit(VCU)将文本/图像/视频/掩码统一为条件输入，结合Context Adapter注入任务概念到DiT模型，首次在单一视频DiT中同时支持参考生成、视频编辑、掩码编辑及其自由组合。

## 研究背景与动机

- **领域现状**：图像领域已有ACE/OmniGen等统一生成编辑框架，但视频领域因时空一致性要求更高，仍以单任务单模型为主。
- **现有痛点**：视频任务繁多（参考生成、风格转换、inpainting等），每个任务独立模型部署成本高；多任务链式组合（如长视频编辑）难以实现。
- **核心矛盾**：统一多种视频任务输入模态 vs 保持时空一致性。
- **本文目标**：构建一个模型覆盖尽可能多的视频生成和编辑任务。
- **切入角度**：设计统一输入接口(VCU) + 概念解耦策略 + 可插拔Context Adapter。
- **核心 idea**：将所有视频任务的条件统一为(文本, 帧序列, 掩码序列)三元组，通过概念解耦和Adapter注入实现多任务统一。

## 方法详解

### 整体框架

基于预训练T2V DiT模型，输入VCU=[T;F;M]三元组。F和M通过概念解耦分为reactive（待改）和inactive（保留）帧，分别经VAE编码后与noisy video token融合，通过全参微调或Context Adapter训练。

### 关键设计

**设计1：Video Condition Unit (VCU)**
- **功能**：将T2V/R2V/V2V/MV2V四类基础任务及其组合统一为(T,F,M)三元组。
- **核心思路**：T是文本；F是帧序列（参考图/控制信号/编辑视频/空帧）；M是二值掩码序列（1=需生成，0=保留）。不同任务通过F和M的不同组合表示。
- **设计动机**：避免为每类任务设计独立接口，组合自由度极高。

**设计2：概念解耦策略**
- **功能**：将F按M分为reactive帧（$F_c=F \times M$）和inactive帧（$F_k=F \times (1-M)$）。
- **核心思路**：显式分离"需要改变的像素"和"需要保留的像素"，确保模型理解编辑vs参考的区别。分别经VAE编码。
- **设计动机**：自然视频和控制信号（深度/pose等）分布不同，混合处理会阻碍收敛。

**设计3：Context Adapter Tuning**
- **功能**：可选的轻量训练策略，冻结DiT主干，仅训练Context Embedder和Context Blocks。
- **核心思路**：从DiT复制部分Transformer Block形成Context Blocks，处理context token并加回主分支，类似Res-Tuning。
- **设计动机**：避免全参微调，收敛更快，与基础模型可插拔。

### 损失函数/训练策略

标准diffusion去噪损失，全参微调或Context Adapter两种策略，随机drop不同条件实现多任务训练。

## 实验关键数据

### 主实验

**VACE-Benchmark 12任务自动评估（部分，归一化平均分）**

| 任务类型 | 对比方法 | VACE(Ours) |
|---------|---------|------------|
| I2V | CogVideoX-I2V: 73.66 | **74.38** |
| Inpaint | ProPainter: 70.15 | 竞争力 |
| Depth Control | 专用模型 | 可比 |
| Pose Control | 专用模型 | 可比 |

### 消融实验

| 配置 | 效果 |
|------|------|
| 无概念解耦 | 编辑和参考任务混淆 |
| 全参微调 vs Adapter | 全参略优但Adapter收敛更快 |
| 无随机条件drop | 任务泛化性下降 |

### 关键发现

1. 统一模型在各子任务上达到与专用模型可比的性能，验证了统一框架的可行性。
2. 任务组合（如参考+inpainting）在专用模型不可能实现但VACE原生支持。
3. 人类评估中VACE在temporal consistency维度显著优于多数baseline。

## 亮点与洞察

1. VCU设计极其优雅，用最少的形式化统一了最多的任务类型。
2. 概念解耦是关键——显式分离编辑和参考信息大幅提升收敛和质量。
3. 首个视频领域的All-in-One生成编辑模型，开创性工作。

## 局限与展望

1. 长视频场景的时间一致性仍有提升空间。
2. 控制信号（深度/pose等）的类型扩展需要额外训练数据。
3. 用户研究规模有限。

## 相关工作与启发

- ACE/OmniGen在图像领域实现了统一生成编辑，VACE将此范式推广到视频。
- 启发：模态统一的关键不是复杂的架构而是优雅的输入接口设计。

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | ★★★★★ |
| 实用性 | ★★★★★ |
| 实验充分性 | ★★★★☆ |
| 写作清晰度 | ★★★★☆ |

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] MagicDrive-V2: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control](magicdrive-v2_high-resolution_long_video_generation_for_autonomous_driving_with_.md)
- [\[ICCV 2025\] OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human Animation Models](omnihuman-1_rethinking_the_scaling-up_of_one-stage_conditioned_human_animation_m.md)
- [\[CVPR 2025\] Visual Prompting for One-Shot Controllable Video Editing Without Inversion](../../CVPR2025/video_generation/visual_prompting_for_one-shot_controllable_video_editing_without_inversion.md)
- [\[ICCV 2025\] AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction](aid_adapting_image2video_diffusion_models_for_instruction-guided_video_predictio.md)
- [\[CVPR 2025\] SketchVideo: Sketch-Based Video Generation and Editing](../../CVPR2025/video_generation/sketchvideo_sketch-based_video_generation_and_editing.md)

<!-- RELATED:END -->
