---
title: >-
  [论文解读] EdgeTAM: On-Device Track Anything Model
description: >-
  [CVPR 2025][图像分割][SAM 2] EdgeTAM 通过详细的延迟分析发现 SAM 2 的瓶颈在记忆注意力而非图像编码器，提出 2D Spatial Perceiver 将帧级记忆从 64×64 维压缩到 ~500 个 token（保留空间结构），配合两阶段知识蒸馏，在 iPhone 15 Pro Max 上实现 16 FPS 的实时 Track Anything。
tags:
  - CVPR 2025
  - 图像分割
  - SAM 2
  - 端侧部署
  - 记忆压缩
  - 2D Spatial Perceiver
  - 知识蒸馏
---

# EdgeTAM: On-Device Track Anything Model

**会议**: CVPR 2025  
**arXiv**: [2501.07256](https://arxiv.org/abs/2501.07256)  
**代码**: 有（Meta Reality Labs）  
**领域**: 分割 / 视频理解  
**关键词**: SAM 2, 端侧部署, 记忆压缩, 2D Spatial Perceiver, 知识蒸馏

## 一句话总结

EdgeTAM 通过详细的延迟分析发现 SAM 2 的瓶颈在记忆注意力而非图像编码器，提出 2D Spatial Perceiver 将帧级记忆从 64×64 维压缩到 ~500 个 token（保留空间结构），配合两阶段知识蒸馏，在 iPhone 15 Pro Max 上实现 16 FPS 的实时 Track Anything。

## 研究背景与动机

**领域现状**：SAM 2 在 SAM 基础上引入记忆库机制，将能力从图像扩展到视频。通过记忆编码器将历史帧编码并存入记忆库，再通过记忆注意力块将当前帧特征与记忆特征融合，实现跨帧跟踪分割。SAM 2 在视频目标分割 (VOS) 和提示式视频分割 (PVS) 上取得了卓越性能，成为该领域的基础模型。

**现有痛点**：SAM 2 最小变体在 iPhone 15 Pro Max 上仅约 1 FPS，无法用于端侧实时应用。现有针对 SAM 的效率优化工作（EdgeSAM、EfficientViT-SAM 等）全部聚焦于压缩图像编码器，因为在 SAM v1 中 mask decoder 极其轻量。但这些方法直接应用到 SAM 2 上效果不佳。

**核心矛盾**：作者通过细致的 iPhone 延迟基准测试发现，即使将图像编码器替换为 ViT-Tiny 或 RepViT 等极轻量骨干，整体延迟改善有限。根本原因在于 SAM 2 新引入的**记忆注意力模块**才是真正的延迟瓶颈。每个记忆特征图大小为 $64 \times 64 \times 64$，T 帧记忆在交叉注意力中形成 $O(T \cdot C \cdot H^2 \cdot W^2)$ 的计算复杂度——这是一个巨大的矩阵乘法，移动设备有限的并行能力无法高效执行。

**本文目标** 如何压缩记忆特征以降低记忆注意力的计算成本，同时不损失视频分割的精度？

**切入角度**：视频天然具有信息冗余——连续帧之间大量内容重复。因此，帧级记忆的密集存储是可以被压缩的。关键在于如何压缩——朴素的空间池化会严重损失精度，因为视频分割是密集预测任务，需要保留空间结构信息。

**核心 idea**：用 2D Spatial Perceiver 在保留空间结构的前提下压缩帧级记忆，将记忆注意力加速 8 倍。

## 方法详解

### 整体框架

EdgeTAM 保持 SAM 2 的元架构不变，主要改动三处：(1) 将图像编码器替换为 RepViT-M1；(2) 将记忆注意力块从 4 个减少到 2 个；(3) 在记忆注意力之前插入 **2D Spatial Perceiver** 模块压缩每帧的记忆特征。此外，采用两阶段知识蒸馏流水线从 SAM 2 教师模型迁移知识到轻量学生模型。

### 关键设计

1. **2D Spatial Perceiver (核心创新)**:

    - 功能：将密集的帧级记忆 $M_t \in \mathbb{R}^{C \times H \times W}$（4096 个 token）压缩为 $\sim$500 个 token，同时保留空间结构
    - 核心思路：由两组可学习查询构成——

        **Global Perceiver**：$N_g$ 个全局查询，每个查询全局注意所有记忆 token，输出 $N_g$ 个向量作为帧级概要。查询之间有一定冗余但可以动态分布到图像任意位置

        **2D Spatial Perceiver**：$N_l$ 个局部查询，用 window partition 将记忆特征图切成 $N_l$ 个不重叠的 patch，每个查询仅注意对应 patch 内的 token，输出有明确的空间位置。位置编码从输入移到输出端（使用 2D-RoPE），保持空间结构

        两组输出沿空间维度展平后拼接，替代原始记忆 token 参与记忆注意力。记忆注意力复杂度从 $O(TCH^2W^2)$ 降为 $O(TCHW(N_g + N_l))$，加速约 $T$ 倍

    - 设计动机：纯 Global Perceiver 会丢弃空间结构信息，导致密集预测任务性能大幅下降。2D Spatial Perceiver 通过局部窗口实现了空间信息的显式保留

2. **两阶段知识蒸馏 (Distillation Pipeline)**:

    - 功能：利用 SAM 2 教师模型提升轻量学生模型的精度，无推理开销
    - 核心思路：

        **阶段 1 - 图像分割预训练**：在 SA-1B 上用任务损失 + 图像编码器特征对齐（MSE loss）训练。$\mathcal{L}_{sam} = \mathcal{L}_{task} + \gamma \mathcal{L}_{img}(F_{16}^t, F_{16}^s)$

        **阶段 2 - 视频分割训练**：在 SA-V 等数据上，除了图像编码器对齐外，额外对齐记忆注意力输出特征。$\mathcal{L}_{sam2} = \mathcal{L}_{task} + \alpha \mathcal{L}_{img} + \beta \mathcal{L}_{mem}(F_M^t, F_M^s)$

    - 设计动机：阶段 1 的编码器蒸馏帮助学生学习更好的表征；阶段 2 的记忆蒸馏让学生的记忆模块也能获得教师的监督信号，弥补压缩带来的信息损失

3. **渐进式长序列微调 (Progressive Fine-tuning)**:

    - 功能：提升长视频场景下的跟踪稳定性
    - 核心思路：先用 8 帧训练，再用 16 帧微调，最后用 32 帧微调。每次延长序列但保持记忆库大小不变。后两阶段冻结图像编码器且不蒸馏
    - 设计动机：EdgeTAM 比 SAM 2 占用更少显存，有条件用更长的训练序列。SAM 2.1 也用了类似策略

### 损失函数 / 训练策略

- 任务损失：dice loss（权重 20）+ focal loss（权重 1）+ IoU loss（权重 1）+ 遮挡预测 BCE loss（阶段 2）
- 蒸馏损失：图像编码器 MSE（权重 1）+ 记忆注意力输出 MSE（权重 1）
- 教师模型：SAM2-HieraB+；学生骨干：RepViT-M1
- 默认配置：2 个记忆注意力块，Global/2D Spatial Perceiver 各 256 个查询

## 实验关键数据

### 主实验

视频目标分割 (VOS) $\mathcal{J}\&\mathcal{F}$：

| 方法 | DAVIS 2017 | MOSE | SA-V val | SA-V test | iPhone FPS |
|------|-----------|------|---------|---------|-----------|
| SAM 2-B+ | 90.9 | 75.8 | 73.6 | 74.1 | 0.7 |
| SAM 2.1-B+ | 90.2 | 76.6 | 76.8 | 77.0 | 0.7 |
| Cutie-base+ | 88.1 | 71.7 | 61.3 | 62.8 | - |
| XMem | 86.0 | 59.6 | 60.1 | 62.3 | - |
| **EdgeTAM** | **87.7** | **70.0** | **72.3** | **71.7** | **15.7** |

Segment Anything (SA-23 基准, 1-click mIoU)：

| 方法 | SA-23 All | SA-23 Image | SA-23 Video | iPhone FPS |
|------|----------|------------|------------|-----------|
| SAM 2 | 61.4 | 63.1 | 59.1 | 1.3 |
| SAM 2.1 | 61.9 | 63.3 | 60.1 | 1.3 |
| **EdgeTAM** | **55.5** | **56.0** | **54.8** | **40.4** |

### 消融实验

2D Spatial Perceiver 消融（RepViT-M1, 2 blocks）：

| 配置 | DAVIS | MOSE | SA-V val | iPhone FPS |
|------|-------|------|---------|-----------|
| 无压缩（baseline） | 86.2 | 66.1 | 71.4 | 2.5 |
| 空间池化 4× | 83.1 | 60.2 | 64.5 | 11.3 |
| Global Perceiver only | 84.5 | 63.8 | 67.2 | 14.8 |
| **2D Spatial Perceiver** | **86.8** | **67.0** | **72.3** | **15.7** |

蒸馏消融（SA-V val/test $\mathcal{J}\&\mathcal{F}$）：

| 配置 | SA-V val | SA-V test | 说明 |
|------|---------|---------|------|
| 无蒸馏 | 71.0 | 68.4 | 基线 |
| + 图像编码器蒸馏 | 71.8 | 70.2 | +0.8/+1.8 |
| + 记忆蒸馏 | **72.3** | **71.7** | +1.3/+3.3 |

### 关键发现

- 记忆注意力是延迟瓶颈：将记忆注意力块从 4 减到 2 几乎线性减少解码延迟；块内移除交叉注意力速度提升最显著
- 空间池化导致严重性能下降（SA-V val -6.9），而 2D Spatial Perceiver 不仅恢复而且超越了基线（+0.9），说明保留空间结构至关重要
- 知识蒸馏在 SA-V test 上带来 +3.3 的提升，阶段 2 的记忆蒸馏贡献了额外的 +0.5/+1.5
- EdgeTAM 在 A100 上达到 150.9 FPS，在 iPhone 上 15.7 FPS，是首个在移动设备上实时运行的统一分割-跟踪模型
- 渐进式长序列微调（8→16→32 帧）带来额外 ~1 点提升

## 亮点与洞察

- **深入的延迟分析改变了优化方向**：揭示了 "SAM 2 瓶颈在记忆注意力而非图像编码器" 这一非直觉的观察，纠正了前人盲目压缩编码器的做法。这种先做 profiling 再做优化的思路值得借鉴
- **2D Spatial Perceiver 的精巧设计**：全局查询捕获帧级概要，局部查询保留空间结构，两者互补。局部查询用 window partition 实现，输出添加 2D-RoPE 位置编码，设计简洁且即插即用
- **蒸馏扩展到视频域**：不仅对齐图像编码器特征，还对齐记忆注意力输出，让记忆模块也能从教师获得监督。这是首次将视频分割的蒸馏扩展到记忆模块

## 局限与展望

- SA-23 图像分割基准上与 SAM 2 差距仍较大（55.5 vs 61.4），说明 RepViT-M1 骨干的表达能力有限
- 记忆库大小（7 帧 + 16 指针）是固定的，未探索自适应记忆管理策略
- 仅验证了 RepViT-M1 和 ViT-Tiny 两种骨干，更多高效架构（如 MobileViT、FastViT）未探索
- 在视频帧率很高的场景下，2D Spatial Perceiver 的压缩比可能不够（帧间冗余更大时可以压缩更多）
- 蒸馏需要教师模型的前向传播，增加了训练成本

## 相关工作与启发

- **vs EdgeSAM/EfficientViT-SAM**: 这些方法仅压缩 SAM v1 的图像编码器；EdgeTAM 额外解决了 SAM 2 的记忆注意力瓶颈，是 SAM 2 高效化方向的首个完整方案
- **vs Cutie/XMem**: 传统 VOS 方法虽然效率尚可，但在 SA-V 等大规模多粒度数据集上精度远不如 SAM 2 系列。EdgeTAM 在保持 SAM 2 级精度的同时实现了端侧部署
- **vs Perceiver/Perceiver IO**: 原版 Perceiver 压缩丢弃空间结构，不适合密集预测。2D Spatial Perceiver 通过局部-全局混合查询解决了这个问题

## 评分

- 新颖性: ⭐⭐⭐⭐ 2D Spatial Perceiver 是工程与设计的好结合，延迟分析揭示了重要洞察
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 PVS/SA/VOS 三类任务共 20+ 数据集，iPhone/A100 双平台延迟测试
- 写作质量: ⭐⭐⭐⭐ 延迟分析图表清晰，方法动机逻辑链完整
- 价值: ⭐⭐⭐⭐⭐ 首个端侧实时 Track Anything 模型，对移动端 AR/MR 应用有直接价值

<!-- RELATED:START -->

## 相关论文

- [SAM2-LOVE: Segment Anything Model 2 in Language-Aided Audio-Visual Scenes](sam2-love_segment_anything_model_2_in_language-aided_audio-visual_scenes.md)
- [InfoSAM: Fine-Tuning the Segment Anything Model from An Information-Theoretic Perspective](../../ICML2025/segmentation/infosam_fine-tuning_the_segment_anything_model_from_an_information-theoretic_per.md)
- [OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation](../../ICCV2025/segmentation/omnisam_omnidirectional_segment_anything_model_for_uda_in_panoramic_semantic_seg.md)
- [Segment and Matte Anything in a Unified Model (SAMA)](../../AAAI2026/segmentation/segment_and_matte_anything_in_a_unified_model.md)
- [SmartEraser: Remove Anything from Images using Masked-Region Guidance](smarteraser_remove_anything_from_images_using_masked-region_guidance.md)

<!-- RELATED:END -->
