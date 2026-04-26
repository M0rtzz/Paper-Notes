---
title: >-
  [论文解读] SnapGen-V: Generating a Five-Second Video within Five Seconds on a Mobile Device
description: >-
  [CVPR 2025][图像生成][移动端部署] SnapGen-V 提出了一套完整的移动端视频扩散模型加速框架，通过剪枝高效空间骨干网络、延迟-内存联合架构搜索确定时序层设计、以及专用的对抗微调将去噪步数降至 4 步，最终以 0.6B 参数在 iPhone 16 上 5 秒内生成 5 秒视频，是首个在移动设备上实现实时文本到视频生成的工作。
tags:
  - CVPR 2025
  - 图像生成
  - 移动端部署
  - 视频扩散模型
  - 架构搜索
  - 对抗蒸馏
  - 模型加速
---

# SnapGen-V: Generating a Five-Second Video within Five Seconds on a Mobile Device

**会议**: CVPR 2025  
**arXiv**: [2412.10494](https://arxiv.org/abs/2412.10494)  
**代码**: [https://snap-research.github.io/snapgen-v/](https://snap-research.github.io/snapgen-v/)  
**领域**: 图像生成 / 视频生成  
**关键词**: 移动端部署, 视频扩散模型, 架构搜索, 对抗蒸馏, 模型加速

## 一句话总结

SnapGen-V 提出了一套完整的移动端视频扩散模型加速框架，通过剪枝高效空间骨干网络、延迟-内存联合架构搜索确定时序层设计、以及专用的对抗微调将去噪步数降至 4 步，最终以 0.6B 参数在 iPhone 16 上 5 秒内生成 5 秒视频，是首个在移动设备上实现实时文本到视频生成的工作。

## 研究背景与动机

**领域现状**：视频扩散模型（如 CogVideoX、Sora）在生成质量上取得了惊人进展，但参数量巨大（数十亿）且生成速度极慢（A100 GPU 上需数分钟），只能部署在云端服务器。

**现有痛点**：(1) 视频模型比图像模型需要更多参数来建模运动，且时空分辨率带来的 token 数量远超图像；(2) 现有所有视频扩散模型（包括最小的 AnimateDiff 和 SVD）在移动设备上都会 OOM；(3) 图像扩散模型的加速方法不能直接迁移到视频领域，因为时序建模层的设计空间和计算特性完全不同。

**核心矛盾**：内容创作者需要在移动端即时生成视频，但视频扩散模型的计算和内存需求远超移动芯片能力。

**本文目标**：设计首个可在移动设备上实时运行的文本到视频扩散模型。

**切入角度**：从三个维度系统化压缩——空间架构（剪枝图像骨干）、时序架构（搜索最优时序层组合）、采样步数（对抗蒸馏到 4 步）。

**核心 idea**：从预训练图像模型出发剪枝获得高效空间骨干，然后在 6 种时序层候选（Conv1D/3D、SelfAttn1D/3D、CrossAttn1D/3D）中进行移动端延迟和内存约束下的进化搜索确定最优时序设计，最后通过图像-视频联合对抗微调实现 4 步生成。

## 方法详解

### 整体框架

三阶段框架：(1) 从 SD v1.5 剪枝获得 2.5× 压缩的高效 UNet 图像骨干；(2) 对时序层类型、位置和数量进行进化架构搜索，找到 Pareto 最优的移动端时空 UNet；(3) 用专为视频设计的对抗微调将搜索出的模型从 25 步蒸馏到 4 步，同时消除 CFG 引导。

### 关键设计

1. **移动端延迟-内存联合架构搜索**:

    - 功能：在 6 种时序层候选中搜索最优组合，满足移动端 OOM 和延迟约束
    - 核心思路：先构建查找表，记录每种时序层在不同时空分辨率下的延迟和内存占用。排除 OOM 状态后，冻结空间骨干，仅搜索时序层的类型（Conv/Attn, 1D/3D）、放置位置（哪个分辨率层级）和数量。候选架构用预计算视频潜变量训练 20K 步后在 VBench 上评估。进化搜索获得速度-质量 Pareto 最优解
    - 设计动机：不同时序层在不同分辨率下的计算特性差异巨大——3D 注意力在低分辨率划算但高分辨率不可行，1D 注意力计算量随空间分辨率线性增长而非二次，需要混合搭配

2. **图像-视频联合对抗微调**:

    - 功能：将去噪步数从 25 步降至 4 步，同时保持视频质量
    - 核心思路：生成器用预训练的文本到视频模型初始化。判别器的骨干使用 UNet 编码器（冻结），在每个下采样块后添加时空判别器头（空间 ResBlock + 时序自注意力），仅更新这些头的参数。关键创新是支持图像-视频混合训练——判别器头同时处理图像和视频数据，图像数据帮助保持纹理质量，视频数据确保时间一致性。使用 Rectified Flow 框架，在 4 个固定时间步上训练
    - 设计动机：紧凑模型（0.6B）的轨迹冗余度低于大模型，直接应用现有蒸馏方法效果不佳；对抗损失提供的分布级监督比逐点损失更适合少步生成

3. **VAE 解码器压缩**:

    - 功能：消除 VAE 解码瓶颈
    - 核心思路：冻结编码器，分别对时序和空间解码器进行剪枝。原始时序解码器 23,100ms → 压缩后 210ms，空间解码器 4100ms → 330ms，实现 50× 加速且质量损失可忽略
    - 设计动机：移动端 VAE 解码也是重要瓶颈

### 损失函数 / 训练策略

三阶段训练：(1) 图像骨干剪枝使用知识蒸馏；(2) 时序层搜索后进行 100K 步图像-视频联合训练，使用 Flow Matching 损失；(3) 对抗微调使用生成器的 Rectified Flow 损失 + 判别器的 hinge loss，固定 4 个推理步骤训练。时间步从 logit-normal 分布采样。

## 实验关键数据

### 主实验 — 速度与质量对比

| 模型 | 参数量 | 步数 | A100 (s) | iPhone (s) | VBench ↑ |
|------|-------|------|---------|-----------|---------|
| CogVideoX-2B | 1.6B | 50 | 54.09 | ✗ | 80.91 |
| AnimateDiff-V2 | 1.2B | 25 | 9.04 | ✗ | 80.27 |
| AnimateDiffLCM | 1.2B | 4 | 1.77 | ✗ | 79.42 |
| OpenSora-v1.2 | 1.2B | 30 | 31.00 | ✗ | 79.76 |
| SnapGen-V | **0.6B** | **4** | **0.47** | **4.12** | **81.14** |

### 消融实验

| 组件 | VBench ↑ | iPhone 延迟 |
|------|---------|-----------|
| 无时序搜索（全 1D Attn） | 79.5 | 5.8s |
| 搜索后架构 | 80.8 | 4.5s |
| + 对抗微调 (4步) | 81.14 | 4.12s |

### 关键发现

- SnapGen-V 是**唯一能在移动设备上运行**的视频扩散模型，其他所有模型（包括参数最少的 AnimateDiff）都会 OOM
- 在 A100 上 SnapGen-V 仅需 **0.47 秒**（4 步），比 CogVideoX 快 115 倍，同时 VBench 得分更高
- 架构搜索发现：高分辨率层级应使用 1D 时序注意力或 Conv，低分辨率层级可使用更强的 3D 注意力——这种混合搭配是手工设计难以发现的
- 图像-视频联合对抗微调比纯视频微调的 VBench 提升 0.5+，说明图像数据对保持纹理质量至关重要

## 亮点与洞察

- **"首个移动端视频扩散模型"**具有里程碑意义，证明了轻量化视频生成的可行性
- **架构搜索的思路值得借鉴**——不同时序层在不同分辨率下的效率特性差异巨大，手工设计很难找到最优组合
- **判别器设计巧妙**——复用预训练 UNet 编码器作为冻结的特征提取器，仅训练轻量头部

## 局限与展望

- 基于 UNet 架构，未探索 DiT 的移动端可行性（DiT 的 token 数二次复杂度使其更难部署）
- 当前仅支持一次性生成完整视频，不支持流式/实时连续生成
- 0.6B 模型在复杂场景（多物体交互、精细人脸）的生成质量可能不如大模型
- iPhone 16 PM 的 4.12 秒已接近实时但仍有优化空间

## 相关工作与启发

- **vs CogVideoX/OpenSora**：这些大模型追求极致质量但只能跑在 GPU 服务器上；SnapGen-V 证明了移动端的质量-效率 tradeoff 是可接受的
- **vs AnimateDiff**：AnimateDiff 虽然较小但 1.2B 参数+25 步仍无法在移动端运行；SnapGen-V 通过系统化压缩实现了质的飞跃
- **对图像扩散的启示**：对抗微调中图像-视频联合训练的判别器设计可推广到其他多模态蒸馏场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个移动端视频扩散模型，三阶段加速框架系统完整
- 实验充分度: ⭐⭐⭐⭐⭐ VBench 定量对比全面，移动端延迟数据真实可信
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，各阶段逻辑连贯
- 价值: ⭐⭐⭐⭐⭐ 开创性工作，对视频生成的民主化部署有重大意义

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] REDUCIO! Generating 1K Video within 16 Seconds using Extremely Compressed Motion Latents](../../ICCV2025/image_generation/reducio_generating_1k_video_within_16_seconds_using_extremely_compressed_motion_.md)
- [\[CVPR 2025\] ShowHowTo: Generating Scene-Conditioned Step-by-Step Visual Instructions](showhowto_generating_scene-conditioned_step-by-step_visual_instructions.md)
- [\[CVPR 2025\] Articulated Kinematics Distillation from Video Diffusion Models](articulated_kinematics_distillation_from_video_diffusion_models.md)
- [\[CVPR 2025\] MobilePortrait: Real-Time One-Shot Neural Head Avatars on Mobile Devices](mobileportrait_real-time_one-shot_neural_head_avatars_on_mobile_devices.md)
- [\[CVPR 2025\] DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models](difflocks_generating_3d_hair_from_a_single_image_using_diffusion_models.md)

<!-- RELATED:END -->
