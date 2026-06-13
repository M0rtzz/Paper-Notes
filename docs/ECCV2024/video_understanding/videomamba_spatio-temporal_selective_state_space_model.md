---
title: >-
  [论文解读] VideoMamba: Spatio-Temporal Selective State Space Model
description: >-
  [ECCV 2024][视频理解][State Space Model] 提出基于纯 Mamba 架构的视频识别模型 VideoMamba（KAIST 版），通过设计时空前向-后向 SSM（Spatio-Temporal Forward and Backward SSM）来有效处理视频中非序列空间信息与序列时间信息的复杂交互，以线性复杂度实现了与 Transformer 竞争的性能。
tags:
  - "ECCV 2024"
  - "视频理解"
  - "State Space Model"
  - "Mamba"
  - "时空建模"
  - "视频识别"
  - "双向扫描"
---

# VideoMamba: Spatio-Temporal Selective State Space Model

**会议**: ECCV 2024  
**arXiv**: [2407.08476](https://arxiv.org/abs/2407.08476)  
**代码**: [https://github.com/jinyjelly/VideoMamba](https://github.com/jinyjelly/VideoMamba)  
**领域**: 视频理解  
**关键词**: State Space Model, Mamba, 时空建模, 视频识别, 双向扫描

## 一句话总结

提出基于纯 Mamba 架构的视频识别模型 VideoMamba（KAIST 版），通过设计时空前向-后向 SSM（Spatio-Temporal Forward and Backward SSM）来有效处理视频中非序列空间信息与序列时间信息的复杂交互，以线性复杂度实现了与 Transformer 竞争的性能。

## 研究背景与动机

**领域现状**：Transformer 在视频识别中表现优异，但自注意力的二次复杂度在处理多帧视频时成为严重瓶颈，尤其在资源受限环境下。

**现有痛点**：传统 CNN 使用 3D 卷积或时空分解卷积，虽然效率较高但捕捉长程依赖能力有限；纯 Transformer 架构虽能建模长程依赖，但二次复杂度随序列长度增长不可接受。

**核心矛盾**：视频数据中空间信息是非序列的（如某帧中人的位置和姿态），而时间信息是序列的（如人的动作随帧变化），如何在 1D 序列模型中有效表达这种时空交互关系。

**本文目标**：探索纯 Mamba 架构在视频识别任务中的适配方案，特别是解决视频 token 的时空双向扫描方向问题。

**切入角度**：从后向扫描方向的选择入手，系统研究空间反转、时间反转、时空反转三种策略对模型性能的影响。

**核心 idea**：通过时空前向-后向 SSM（Spatio-Temporal Forward and Backward SSM），使后向扫描对所有 token 进行完全时空反转，让前向与后向的 token 顺序形成互补。

## 方法详解

### 整体框架

VideoMamba（KAIST 版）采用纯 Mamba 编码器架构。输入视频 $V \in \mathbb{R}^{T \times H \times W \times C}$ 首先通过 Video Tokenizer（3D 卷积，tubelet 大小 $s_t \times s_h \times s_w = 2 \times 16 \times 16$）映射为 $n_t \cdot n_h \cdot n_w$ 个 video token $z_i \in \mathbb{R}^d$（$d=384$）。加入位置编码后，前置 class token，送入 $L=24$ 层 VideoMamba 编码器。最终 class token 经归一化和单层 MLP 输出分类结果。

### 关键设计

1. **Video Tokenizer**：使用 3D 卷积从不重叠的 tubelet 中提取 token。关键的初始化策略是从预训练 2D 卷积膨胀到 3D 卷积——在时间轴上扩展权重张量并取平均：

    $n_t = \lfloor T/s_t \rfloor, \quad n_h = \lfloor H/s_h \rfloor, \quad n_w = \lfloor W/s_w \rfloor$

   这种 inflation 策略使模型能有效利用 ImageNet 预训练权重。

2. **位置编码（Positional Embedding）**：SSM 原本不需要位置编码（其递归特性隐含了位置信息），但考虑到视频的时空特性，本文系统比较了多种位置编码方案。通过消融实验确定 **时间维度扩展（Temporal Expanding）** 初始化最优——将图像预训练的位置编码 $P_{image} \in \mathbb{R}^{n_h \cdot n_w \times d}$ 沿时间轴复制 $n_t$ 次：

   | 位置编码方式 | SSV2 | HMDB |
   |------------|------|------|
   | 无位置编码 | 63.2% | 48.7% |
   | Sinusoidal | 63.3% | 47.5% |
   | Learned (随机初始化) | 63.4% | 47.9% |
   | Learned (空间插值) | 63.6% | 49.4% |
   | Learned (嵌入维度插值) | 63.6% | 51.5% |
   | **Learned (时间扩展)** | **63.7%** | **58.9%** |

   时间扩展方式在 HMDB 上领先第二名 7.4%，证明从图像模型继承空间位置信息并合理扩展到时间维度至关重要。

3. **时空前向-后向 SSM（Spatio-Temporal Forward and Backward SSM）**：这是本文最核心的贡献。为处理视频中非序列空间与序列时间信息的交互，设计了三种后向扫描方向：

    - **时空反转（Spatio-temporal reversal）**：完全反转所有 $n_t \cdot n_h \cdot n_w$ 个 token 的顺序，等效于将视频各帧纵向拼接成一张长图后再反转。前向和后向的 token 顺序完全互补。
    - **空间反转（Spatial reversal）**：仅反转每帧内的 $n_h \cdot n_w$ 个 token，保持时间轴顺序不变。保留了清晰的时间流。
    - **时间反转（Temporal reversal）**：保持帧内空间 token 顺序，仅反转帧的时间序列。提供反向事件进程而不改变帧的空间完整性。

   实验表明 **时空反转** 效果最优（SSV2: 64.7%, HMDB: 55.2%），因为前向与后向扫描提供了最大程度的 token 顺序互补。空间反转效果最差，因为大部分 token 的相对位置在前后向路径中保持不变。

4. **Delta 参数分析**：Mamba 中的 $\Delta$ 参数起门控作用——大 $\Delta$ 表示忽略隐藏状态、强调当前输入，小 $\Delta$ 表示忽略当前输入。可视化分析表明：

    - 浅层：$\Delta$ 值普遍较高，模型先理解整体场景
    - 深层：$\Delta$ 值降低并聚焦于关键运动区域（如骑自行车场景中的手部、跳水场景中的运动员），有效过滤静态背景
   
   这证明 VideoMamba 通过 $\Delta$ 实现了高效的时空推理。

5. **时间一致性依赖分析**：通过重排输入帧验证模型对时间顺序的依赖：

   | 重排策略 | HMDB Top-1 |
   |---------|-----------|
   | Interleaved（交错, 最大扰动） | 51.3% |
   | Pairwise（成对交换） | 53.5% |
   | Block-wise（块交换） | 56.5% |
   | **Sequential（原始顺序）** | **58.9%** |

   时间扰动越严重，性能下降越明显，证明模型确实在利用时间顺序进行推理。

### 损失函数 / 训练策略

- **优化器**：AdamW，学习率 3e-4，cosine decay 调度 + 线性 warmup
- **训练策略**：K400 训练 30 epoch，SSV2 训练 35 epoch，HMDB 训练 50 epoch，batch size 64
- **数据增强**：Label Smoothing、RandAugment、Random Erasing
- **初始化**：ImageNet-1K 预训练权重初始化 backbone
- **推理**：多视角（crops）推理取平均得分

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 (VideoMamba) | 对比方法 | 说明 |
|--------|------|------|------|------|
| HMDB51 (IN-1K 初始化) | Top-1 | 59.3% (32f) | VideoSwin-T 54.4% | +4.9%, 参数更少 |
| HMDB51 (K400 初始化) | Top-1 | 68.6% (16f) | Mamba-ND 59.0% | +9.6% |
| HMDB51 (K400 初始化) | Top-1 | 75.7% (32f) | VideoSwin-T(K400) 69.9% | +5.8% |
| SSV2 (IN-1K 初始化) | Top-1 | 64.2% (32f) | VideoSwin-T 52.3% | +11.9% |
| K400 (IN-1K 初始化) | Top-1 | 77.7% (32f) | VideoSwin-T 78.8% | 接近, GFLOPs 更低 |

### 效率对比

| 模型 | GFLOPs | 参数量 | 显存 |
|------|--------|-------|------|
| VideoMamba (16f) | 34 G | 26.3M | 显著低于 Transformer |
| VideoSwin-T (32f) | 88 G | 27.8M | - |
| VideoMAE-S (16f) | 57 G | 22.0M | - |
| TimeSformer (8f) | 196 G | 121.4M | - |

### 消融实验

| 配置 | SSV2 | HMDB | 说明 |
|------|------|------|------|
| 空间反转 | 61.9% | 43.3% | 最差，互补性不足 |
| 时间反转 | 63.3% | 52.9% | 中等 |
| **时空反转** | **64.7%** | **55.2%** | **最优，完全互补** |
| 帧数 8f → 16f → 32f | 61.0→63.7→64.2 | 52.7→58.9→59.3 | 更多帧持续提升 |
| 嵌入维度 192 → 384 | 54.6→63.7 | 56.5→68.6 | 更大维度显著提升 |

### 关键发现

- 时空反转是最优后向扫描策略，前向与后向 token 顺序的互补性是关键
- 位置编码对视频 SSM 至关重要（HMDB 上引入可学习位置编码提升 10.2%），且初始化方式影响巨大
- VideoMamba 真正依赖时间顺序进行推理，而非简单地将视频当作图像集合
- $\Delta$ 参数的可视化揭示了 SSM 从浅层全局感知到深层局部聚焦的渐进推理模式
- 在 GFLOPs 仅为 VideoSwin-T 的 39% 的情况下，SSV2 上领先 11.9%

## 亮点与洞察

1. **系统性的扫描方向研究**：三种后向扫描策略的比较提供了清晰的设计指导——时空完全反转最优，因为它最大化了双向扫描的互补性
2. **$\Delta$ 可视化分析**：首次深入分析了视频 SSM 中 $\Delta$ 参数的行为，揭示了从全局理解到局部聚焦的层级推理模式
3. **位置编码的系统探索**：为 SSM-based 视频模型的位置编码设计提供了全面的基准和指导
4. **时间一致性实验**：通过帧重排实验严格验证了模型对时间顺序的依赖，而非简单的外观识别

## 局限与展望

1. 在 K400 上性能（77.7%）略低于 VideoSwin-T（78.8%），场景相关的任务上纯 SSM 可能不如局部注意力
2. 仅使用 ImageNet-1K 预训练，未探索更大规模预训练（如 IN-21K）的效果
3. 固定的 tubelet 大小（$2 \times 16 \times 16$）可能不适合所有视频分辨率和帧率
4. 未探索 SSM 与注意力机制的混合架构，可能会进一步提升性能
5. 模型规模较小（仅 26M 参数），更大模型的可扩展性问题有待研究

## 相关工作与启发

- **与 OpenGVLab VideoMamba 的关系**：两篇同名论文，本文来自 KAIST，侧重于扫描方向的系统研究和 $\Delta$ 分析；OpenGVLab 版本侧重于可扩展性和长视频理解
- **Vision Mamba (Vim)**：本文在 Vim 的双向扫描基础上扩展到时空维度
- **S4ND**：早期在视频中使用 SSM 的尝试，但缺乏输入依赖的选择机制，性能受限
- **启发**：$\Delta$ 作为注意力权重的类比，可能启发新的可解释性工具；双向扫描中互补性的重要性可推广到其他序列建模任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 时空前向-后向 SSM 的设计清晰，三种后向扫描的系统比较有价值
- 实验充分度: ⭐⭐⭐⭐ 消融实验全面（扫描方向、位置编码、帧数、维度），Delta 分析深入
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题定义明确，可视化分析直观
- 价值: ⭐⭐⭐⭐ 为视频 SSM 的设计提供了系统性指导，但整体影响力可能被同名 OpenGVLab 版本分流

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] VideoMamba: State Space Model for Efficient Video Understanding](videomamba_state_space_model_for_efficient_video_understanding.md)
- [\[NeurIPS 2025\] PASS: Path-Selective State Space Model for Event-Based Recognition](../../NeurIPS2025/video_understanding/pass_path-selective_state_space_model_for_event-based_recognition.md)
- [\[ECCV 2024\] PiTe: Pixel-Temporal Alignment for Large Video-Language Model](pite_pixel-temporal_alignment_for_large_video-language_model.md)
- [\[CVPR 2025\] MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking](../../CVPR2025/video_understanding/mambavlt_time-evolving_multimodal_state_space_model_for_vision-language_tracking.md)
- [\[ECCV 2024\] SAFNet: Selective Alignment Fusion Network for Efficient HDR Imaging](safnet_selective_alignment_fusion_network_for_efficient_hdr_imaging.md)

</div>

<!-- RELATED:END -->
