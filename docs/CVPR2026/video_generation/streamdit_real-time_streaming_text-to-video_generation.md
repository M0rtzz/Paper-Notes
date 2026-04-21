---
title: >-
  [论文解读] StreamDiT: Real-Time Streaming Text-to-Video Generation
description: >-
  [CVPR 2026][流式视频生成] StreamDiT 提出了一套完整的流式视频生成方案（包括训练、建模和蒸馏），通过在 Flow Matching 中引入带渐进去噪的移动缓冲区和混合分区训练策略，结合时变 DiT 架构和窗口注意力，以及定制化的多步蒸馏方法，使 4B 参数模型在单 GPU 上达到 512p@16FPS 的实时流式视频生成。
tags:
  - CVPR 2026
  - 流式视频生成
  - Transformer
  - 实时推理
  - 采样蒸馏
  - Flow Matching
---

# StreamDiT: Real-Time Streaming Text-to-Video Generation

**会议**: CVPR 2026  
**arXiv**: [2507.03745](https://arxiv.org/abs/2507.03745)  
**代码**: https://cumulo-autumn.github.io/StreamDiT/ (项目页)  
**领域**: 扩散模型 / 视频生成  
**关键词**: 流式视频生成, 扩散Transformer, 实时推理, 采样蒸馏, Flow Matching

## 一句话总结
StreamDiT 提出了一套完整的流式视频生成方案（包括训练、建模和蒸馏），通过在 Flow Matching 中引入带渐进去噪的移动缓冲区和混合分区训练策略，结合时变 DiT 架构和窗口注意力，以及定制化的多步蒸馏方法，使 4B 参数模型在单 GPU 上达到 512p@16FPS 的实时流式视频生成。

## 研究背景与动机

1. **领域现状**：当前顶尖的文本到视频（T2V）模型（如 MovieGen、Hunyuan、Step-Video）基于 Diffusion Transformer（DiT）架构，使用双向注意力，能生成高质量短视频。但它们只能离线生成固定长度的短片段，无法支持交互式和实时应用。

2. **现有痛点**：
    - 增加视频长度代价极高——Transformer 对序列长度有二次复杂度
    - 自回归（AR）方式可以生成长视频但使用因果注意力，质量远不如双向注意力
    - 现有无训练的流式方法（StreamDiffusion、FIFO-Diffusion）缺少训练支持，质量受限
    - 采样蒸馏方法（步骤蒸馏、一致性蒸馏）无法直接应用于流式去噪的非标准设置

3. **核心矛盾**：低延迟（流式输出）、高吞吐量（批量处理）、高质量（双向注意力）三者难以兼得。AR 有低延迟但质量差；双向扩散有高质量但无法流式输出。

4. **本文目标** 设计一套可训练、可蒸馏的流式视频生成完整方案，同时兼顾质量和实时性。

5. **切入角度**：受 FIFO-Diffusion 的对角去噪启发，将缓冲区内帧分配不同噪声等级，但通过可训练方案和混合分区策略来弥补质量差距。

6. **核心 idea**：通过统一的帧分区方案将均匀噪声和渐进对角噪声作为特例纳入同一框架，用混合训练提升一致性，用定制化多步蒸馏实现实时推理。

## 方法详解

### 整体框架
输入文本 prompt，通过 StreamDiT 模型以帧缓冲区的方式持续生成视频帧。缓冲区包含 $B$ 帧，每帧有不同噪声等级，经过去噪后干净帧从缓冲区弹出输出，新噪声帧推入。整体 pipeline 分三层：(1) Buffered Flow Matching 训练框架；(2) 时变 DiT + 窗口注意力的高效模型架构；(3) 定制化多步蒸馏实现实时性。

### 关键设计

1. **Buffered Flow Matching（缓冲区流匹配）**:

    - 功能：将标准 Flow Matching 扩展为支持流式生成的训练框架
    - 核心思路：标准 FM 对所有帧施加相同时间步 $t$，StreamDiT 改为对缓冲区内帧分配一组单调递增的时间步 $\tau = [\tau_1, ..., \tau_B]$。训练样本构造为 $\mathbf{X}_\tau^i = \tau \circ \mathbf{X}_1^i + (1-(1-\sigma_{min})\tau) \circ \mathbf{X}_0$，推理时缓冲区沿帧维度滑动，干净帧弹出、噪声帧推入，实现流式输出
    - 设计动机：直接在 Flow Matching 框架内引入流式机制，使训练和推理一致，避免无训练方法的质量损失

2. **统一分区方案（Partitioning Scheme）**:

    - 功能：用一个通用框架统一不同的噪声分配策略
    - 核心思路：缓冲区被分为 $K$ 个参考帧和 $N$ 个块（chunk），每个块有 $c$ 帧和 $s$ 个微步。总帧数 $B = K + N \times c$，总去噪步 $T = s \times N$。当 $c=B, s=1$ 时退化为均匀噪声（标准 T2V）；$c=1, s=1$ 时退化为对角噪声（FIFO-Diffusion）。混合训练在不同分区方案之间交替，防止过拟合并增强内容一致性
    - 设计动机：单一分区方案容易过拟合，混合训练能学到更泛化的去噪能力。实验证明混合所有 chunk size（1,2,4,8,16）效果最好

3. **时变 DiT 架构（Time-Varying DiT）**:

    - 功能：让模型能处理缓冲区内不同帧的不同噪声等级
    - 核心思路：修改标准 adaLN DiT，要求时间嵌入在帧维度上是可分离的。将 latent 张量 reshape 为 $[F,H,W]$，沿第一维应用不同的时间嵌入来控制 scale 和 shift 调制。同时用窗口注意力替换全注意力——将 3D latent 划分为非重叠窗口 $[F_w, H_w, W_w]$，每隔一层交替移位实现全局信息传播，计算量仅为全注意力的 $\frac{F_w H_w W_w}{FHW}$
    - 设计动机：帧级时间嵌入是 StreamDiT 训练方案的必要条件；窗口注意力大幅降低计算量，是实现实时推理的关键

4. **定制化多步蒸馏**:

    - 功能：将采样步数从 128 步降至 8 步且去除 CFG，实现实时推理
    - 核心思路：选定分区方案 $c=2, s=16, N=8$，教师模型有 $s \times N = 128$ 步。FM 轨迹被分为 $N$ 段，在每段内独立进行步数蒸馏。同时进行步数蒸馏和引导蒸馏——将教师的多步 CFG 推理蒸馏为学生的单步无条件前向传播。蒸馏后微步 $s$ 从 16 降到 1，总步数仅 8 步
    - 设计动机：标准蒸馏方法（步骤蒸馏、一致性蒸馏）无法直接应用于流式去噪的非标准设置，需要按分区方案的段落结构进行蒸馏

### 训练策略
三阶段训练：(1) 任务学习——3K 高质量视频、大学习率 $1e{-4}$，适配流式任务；(2) 任务泛化——2.6M 预训练视频、小学习率 $1e{-5}$，提升泛化；(3) 质量微调——高质量数据、小学习率精调。每阶段 128 H100 GPU 训练 10K 迭代。蒸馏在 64 H100 上进行 10K 迭代。

## 实验关键数据

### 主实验（VBench 质量指标）

| 方法 | 主题一致性 | 背景一致性 | 时序闪烁 | 运动平滑 | 动态程度 | 美学质量 | 质量分 |
|------|-----------|-----------|---------|---------|---------|---------|-------|
| ReuseDiffuse | 0.9501 | 0.9615 | 0.9838 | 0.9912 | 0.2900 | 0.5993 | 0.8019 |
| FIFO-Diffusion | 0.9412 | 0.9576 | 0.9796 | 0.9889 | 0.3094 | 0.6088 | 0.7981 |
| StreamDiT (teacher) | 0.9622 | 0.9625 | 0.9671 | 0.9861 | **0.5240** | 0.6026 | **0.8185** |
| StreamDiT (distill) | 0.9491 | 0.9555 | 0.9649 | 0.9831 | **0.7040** | 0.5940 | 0.8163 |

### 消融实验（混合训练效果）

| Chunk size 组合 | 质量分 | 说明 |
|----------------|-------|------|
| [1] | 0.8129 | 仅对角噪声（Progressive AR Diffusion） |
| [1,2] | 0.8100 | 混合 2 种 |
| [1,2,4] | 0.8080 | 混合 3 种 |
| [1,2,4,8] | 0.8076 | 混合 4 种 |
| [1,2,4,8,16] | **0.8144** | 全混合，效果最好 |

### 关键发现
- StreamDiT 在质量分和人评（4 个维度全胜）上均超越 ReuseDiffuse 和 FIFO-Diffusion
- 基线方法虽然有更高的时序一致性和运动平滑度，但实际生成内容更加静态（动态程度极低 0.29-0.31 vs StreamDiT 的 0.52-0.70）
- 蒸馏模型与教师模型质量非常接近（0.8163 vs 0.8185），但步数从 128 降至 8
- 混合所有 chunk size 的训练方案效果最优，即使推理时只用 chunk size 1
- 实时性能：蒸馏模型在单 H100 上 482ms 生成 2 帧 latent（8 帧视频），达到 16 FPS

## 亮点与洞察
- **统一分区方案的优雅设计**：用 $(K, N, c, s)$ 四个参数统一了从标准扩散到对角扩散的所有方案，将不同方法纳入同一框架，这种抽象方式极其简洁
- **混合训练策略的意外效果**：混合所有 chunk size（包括非流式的 chunk=16）反而提升了流式生成（chunk=1）的质量，说明多任务训练的正则化效应
- **段落化蒸馏**：将 FM 轨迹按分区段落切分后独立蒸馏的思路，可以迁移到其他非标准采样路径的蒸馏场景

## 局限与展望
- 4B 参数的模型容量有限，部分生成视频存在 artifact（作者验证 30B 模型质量显著提升）
- 短上下文长度限制——画面外物体重新出现时外观可能改变
- 窗口注意力虽然高效但可能损失全局一致性
- 未来方向：结合 KV cache 扩展上下文、扩展到更大模型、提升分辨率

## 相关工作与启发
- **vs FIFO-Diffusion**: FIFO 是无训练的对角去噪方法，StreamDiT 通过训练方案和混合策略大幅提升质量
- **vs Self-Forcing**: Self-Forcing 是 AR 视频扩散的方案，每次生成一帧延迟低但质量受限；StreamDiT 用双向注意力+流式实现质量与延迟的平衡
- **vs StreamingT2V**: 使用短期和长期记忆块的 AR 方案，StreamDiT 的统一分区框架更优雅

## 评分
- 新颖性: ⭐⭐⭐⭐ 统一分区方案和定制化蒸馏有原创性，系统性设计完整
- 实验充分度: ⭐⭐⭐⭐ VBench 定量+人评+消融+多种应用展示，较为全面
- 写作质量: ⭐⭐⭐⭐⭐ 框架层次清晰，公式推导严谨，图示精美
- 价值: ⭐⭐⭐⭐⭐ 首次实现实时流式视频生成，在交互式应用场景中有重要价值

<!-- RELATED:START -->

## 相关论文

- [U-Mind: A Unified Framework for Real-Time Multimodal Interaction with Audiovisual Generation](u-mind_a_unified_framework_for_real-time_multimodal_interaction_with_audiovisual.md)
- [Teller: Real-Time Streaming Audio-Driven Portrait Animation with Autoregressive Motion Generation](../../CVPR2025/video_generation/teller_real-time_streaming_audio-driven_portrait_animation_with_autoregressive_m.md)
- [MotionStream: Real-Time Video Generation with Interactive Motion Controls](../../ICLR2026/video_generation/motionstream_real-time_video_generation_with_interactive_motion_controls.md)
- [PAM: A Pose-Appearance-Motion Engine for Sim-to-Real HOI Video Generation](pam_a_pose-appearance-motion_engine_for_sim-to-real_hoi_video_generation.md)
- [Streaming Autoregressive Video Generation via Diagonal Distillation](../../ICLR2026/video_generation/streaming_autoregressive_video_generation_via_diagonal_distillation.md)

<!-- RELATED:END -->
