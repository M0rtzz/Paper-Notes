---
title: >-
  [论文解读] VP-SAM: Taming Segment Anything Model for Video Polyp Segmentation via Disentanglement and Spatio-Temporal Side Network
description: >-
  [ECCV 2024][图像分割][视频息肉分割] 本文提出 VP-SAM，通过语义解耦适配器（SDA）利用傅里叶频谱的幅度信息帮助 SAM 区分低对比度的息肉与背景，同时设计时空侧网络（STSN）为 SAM 注入视频帧间时序信息，在 SUN-SEG、CVC-612 和 CVC-300 等数据集上达到 SOTA。
tags:
  - ECCV 2024
  - 图像分割
  - 视频息肉分割
  - SAM适配
  - 语义解耦
  - 时空建模
  - 频域分析
---

# VP-SAM: Taming Segment Anything Model for Video Polyp Segmentation via Disentanglement and Spatio-Temporal Side Network

**会议**: ECCV 2024  
**PDF**: [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03403.pdf)
**代码**: https://github.com/zhixue-fang/VPSAM (有)  
**领域**: 分割  
**关键词**: 视频息肉分割, SAM适配, 语义解耦, 时空建模, 频域分析

## 一句话总结

本文提出 VP-SAM，通过语义解耦适配器（SDA）利用傅里叶频谱的幅度信息帮助 SAM 区分低对比度的息肉与背景，同时设计时空侧网络（STSN）为 SAM 注入视频帧间时序信息，在 SUN-SEG、CVC-612 和 CVC-300 等数据集上达到 SOTA。

## 研究背景与动机

**领域现状**：视频息肉分割（Video Polyp Segmentation, VPS）是肠镜辅助诊断的关键任务，旨在从结肠镜视频中逐帧分割出息肉区域。近年来，Segment Anything Model（SAM）作为通用分割基础模型展示了强大的泛化能力，成为各种下游分割任务的适配基座。

**现有痛点**：(1) 息肉与周围黏膜组织的颜色和纹理高度相似，对比度极低，SAM 在自然图像上训练的特征难以有效区分这两者；(2) SAM 是图像级模型，缺乏处理视频序列时序信息的能力——息肉在连续帧间的大小、位置、形状变化剧烈（由于镜头运动和肠道蠕动），单帧处理无法利用时序一致性；(3) 现有的 SAM 适配方法（如 SAM-Adapter、Medical SAM Adapter）主要为静态图片设计，未考虑视频的时空动态。

**核心矛盾**：SAM 强大的通用表示能力与息肉分割的特殊需求之间存在鸿沟——SAM 擅长分割高对比度目标但对低对比度医学目标力不从心，SAM 处理单帧但 VPS 是时序任务。如何在不破坏 SAM 预训练知识的前提下弥补这两个鸿沟？

**本文目标** (1) 如何帮助 SAM 在低对比度场景中解耦前景（息肉）和背景（黏膜）？(2) 如何为 SAM 注入视频时序信息实现帧间跟踪？

**切入角度**：作者观察到息肉和背景虽然在空间域中外观相似，但在频域中表现出不同的频率特征——息肉区域通常具有与背景不同的纹理频率分布。利用傅里叶变换的幅度谱可以放大这种差异。同时，通过设计一个与 SAM 并行的轻量侧网络来处理时序信息，可以在不修改 SAM 主干的情况下注入时空感知能力。

**核心 idea**：用频域幅度信息解耦低对比度前背景 + 时空侧网络注入帧间信息，双管齐下让 SAM 适配视频息肉分割任务。

## 方法详解

### 整体框架

VP-SAM 以冻结的 SAM 编码器（ViT-B/H）为主干，在其基础上添加两个可训练的模块：(1) 语义解耦适配器（SDA），插入到 SAM 编码器的每个 Transformer 层中，利用频域信息增强前背景区分能力；(2) 时空侧网络（STSN），与 SAM 编码器并行运行，接受多帧输入提取时序特征，并在多个尺度上与 SAM 特征融合。输入是连续的内窥镜视频帧，输出是每帧的息肉分割掩码。

### 关键设计

1. **语义解耦适配器（Semantic Disentanglement Adapter, SDA）**:

    - 功能：利用傅里叶频谱的幅度信息帮助 SAM 区分视觉相似的息肉和背景
    - 核心思路：对 SAM 编码器中间层的特征图进行 2D FFT 变换，提取幅度谱 $|F(u,v)| = \sqrt{Re^2 + Im^2}$。幅度谱编码了不同频率成分的能量分布——息肉区域特有的纹理模式对应特定的频率分量。将幅度谱通过一个轻量的 MLP 映射回特征空间，得到频域增强特征 $f_{freq}$，然后与原始空间域特征进行逐元素相加融合：$f_{out} = f_{spatial} + \gamma \cdot f_{freq}$，其中 $\gamma$ 是可学习的缩放参数
    - 设计动机：在空间域中，息肉和背景的颜色/亮度高度相似（低对比度），但它们的纹理模式不同——息肉表面通常较光滑而周围黏膜有褶皱纹理。傅里叶幅度谱能显式捕获这种纹理差异，为 SAM 提供了额外的区分线索

2. **时空侧网络（Spatio-Temporal Side Network, STSN）**:

    - 功能：在不修改 SAM 主干的前提下，为分割模型注入视频帧间的时序信息
    - 核心思路：STSN 是一个独立的轻量编码器，接受当前帧及其相邻帧（如前后各 2 帧）作为输入。首先用共享的 CNN 提取每帧的空间特征，然后通过时序注意力模块建模帧间关系：$A_{t} = \text{softmax}(Q_t K_{1:T}^T / \sqrt{d}) V_{1:T}$，捕获息肉在连续帧间的运动轨迹和形变模式。STSN 的输出在多个分辨率上通过跳跃连接与 SAM 编码器的中间特征融合
    - 设计动机：SAM 是单帧模型，无法感知息肉在视频中的运动状态（出现、消失、形变）。通过并行的侧网络注入时序信息，既保持了 SAM 预训练知识的完整性，又赋予了模型时序感知能力。"侧网络"设计比微调 SAM 主干更安全，避免了灾难性遗忘

3. **多尺度特征融合解码器（Multi-Scale Fusion Decoder）**:

    - 功能：整合 SAM 特征和 STSN 时序特征，生成精细的分割掩码
    - 核心思路：在四个分辨率尺度上，将 SAM 编码器的中间层特征与 STSN 对应尺度的时序特征通过注意力门控机制融合。门控权重由两路特征的相关性决定：$g = \sigma(W_s f_{SAM} + W_t f_{STSN})$，$f_{fused} = g \odot f_{SAM} + (1-g) \odot f_{STSN}$。融合后的多尺度特征通过渐进式上采样解码为最终分割掩码
    - 设计动机：不同尺度捕获不同信息——低分辨率特征提供全局上下文和息肉位置信息，高分辨率特征提供边界细节。门控融合让模型自适应决定在每个位置更信任 SAM 特征还是时序特征

### 损失函数 / 训练策略

使用 BCE 损失 + Dice 损失的加权组合：$L = L_{BCE} + L_{Dice}$。训练时冻结 SAM 编码器主干（ViT 权重），仅训练 SDA、STSN 和解码器。使用 AdamW 优化器，学习率 1e-4，余弦退火调度。训练数据使用连续视频帧片段（5帧为一组）。

## 实验关键数据

### 主实验

| 数据集 | 指标 | VP-SAM | PNS+ | SANet | Polyp-PVT | 提升 |
|--------|------|--------|------|-------|-----------|------|
| SUN-SEG (Easy) | Dice ↑ | **88.3** | 84.7 | 83.2 | 85.1 | +3.2 |
| SUN-SEG (Hard) | Dice ↑ | **80.1** | 74.6 | 72.8 | 75.9 | +4.2 |
| CVC-612 | Dice ↑ | **92.7** | 89.3 | 88.1 | 90.5 | +2.2 |
| CVC-300 | Dice ↑ | **91.5** | 88.0 | 87.2 | 89.1 | +2.4 |
| SUN-SEG (Hard) | IoU ↑ | **72.8** | 66.3 | 64.1 | 68.2 | +4.6 |

### 消融实验

| 配置 | SUN-SEG Easy Dice | SUN-SEG Hard Dice | 说明 |
|------|-------------------|-------------------|------|
| VP-SAM (完整) | **88.3** | **80.1** | 完整模型 |
| w/o SDA | 85.6 | 76.5 | SDA 在困难场景贡献大 |
| w/o STSN (单帧) | 86.1 | 75.8 | 时序信息对困难场景至关重要 |
| w/o 频域 (仅空域 adapter) | 86.8 | 77.3 | 频域信息额外贡献约 2.8 |
| SDA + STSN w/o 门控融合 | 87.2 | 78.4 | 门控融合贡献约 1.7 |
| SAM-ViT-B 基底 | 88.3 | 80.1 | 标准配置 |
| SAM-ViT-H 基底 | 89.1 | 81.5 | 更大模型进一步提升 |

### 关键发现
- 在"困难"子集上的提升（+4.2 Dice）远大于"简单"子集（+3.2），说明 SDA 和 STSN 在低对比度和大形变场景中效果显著
- STSN 对困难场景的贡献（-4.3 Dice when removed）大于 SDA（-3.6），说明时序信息在处理息肉运动和遮挡时更关键
- 频域分析（幅度谱）比纯空域 adapter 多贡献约 2.8 Dice，验证了频域信息对低对比度目标区分的有效性

## 亮点与洞察
- **频域解耦前背景**的思路非常巧妙：当空间域中前背景不可分时，转换到频域可能找到新的区分线索。这个思路可以迁移到其他低对比度分割任务，如皮肤病变分割、视网膜血管分割等
- **侧网络架构**在适配基础模型时兼顾了"不破坏预训练知识"和"注入新能力"两个需求，是一种比 LoRA/Adapter 更灵活的适配范式——适合需要注入全新模态信息（如时序）的场景
- 在简单和困难子集上的差异化分析提供了对方法有效性的深入理解

## 局限与展望
- STSN 使用固定的帧数（5帧），对于长序列中的远程时序依赖（如息肉消失后重新出现）可能不够
- 仅在息肉分割上验证，未在其他医学视频分割任务（如手术器械分割、超声视频分割）上测试泛化性
- 侧网络增加了参数量和推理时间，虽然论文声称"轻量"，但具体的 FPS 数据未报告
- 可以引入 SAM 2（支持视频理解）作为基底替代 SAM，或许能进一步提升时序处理能力

## 相关工作与启发
- **vs PNS+ (Polyp-NeoNet-Seg+)**: PNS+ 是专为息肉设计的视频分割方法，使用光流进行帧间对齐。VP-SAM 通过 STSN 的时序注意力替代了显式光流计算，更鲁棒且无需额外的光流网络
- **vs SAM-Adapter**: SAM-Adapter 只在空间域插入 adapter，VP-SAM 额外加入了频域解耦和时序建模，更全面地适配了 VPS 任务的特殊需求
- **vs Medical SAM Adapter**: Medical SAM Adapter 为静态医学图像设计，VP-SAM 专门针对视频场景引入了 STSN，扩展了适配范围

## 评分
- 新颖性: ⭐⭐⭐⭐ 频域解耦 + 时空侧网络的组合设计新颖，频域解耦思路尤其有启发性
- 实验充分度: ⭐⭐⭐⭐ 多数据集验证，消融详细，easy/hard 分析有深度
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，方法图示易懂
- 价值: ⭐⭐⭐⭐ 为 SAM 适配医学视频分割提供了有效方案，频域解耦思路可广泛迁移

<!-- RELATED:START -->

## 相关论文

- [SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Model](../../AAAI2026/segmentation/saq-sam_semantically-aligned_quantization_for_segment_anything_model.md)
- [SAM-DAQ: Segment Anything Model with Depth-guided Adaptive Queries for RGB-D Video Salient Object Detection](../../AAAI2026/segmentation/sam-daq_segment_anything_model_with_depth-guided_adaptive_queries_for_rgb-d_vide.md)
- [Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustment](long-tail_temporal_action_segmentation_with_group-wise_temporal_logit_adjustment.md)
- [Segment and Matte Anything in a Unified Model (SAMA)](../../AAAI2026/segmentation/segment_and_matte_anything_in_a_unified_model.md)
- [E-SAM: Training-Free Segment Every Entity Model](../../ICCV2025/segmentation/e-sam_training-free_segment_every_entity_model.md)

<!-- RELATED:END -->
