---
title: >-
  [论文解读] Fine-grained Spatiotemporal Grounding on Egocentric Videos
description: >-
  [ICCV 2025][视频理解][egocentric video] 提出 EgoMask，首个面向自我中心视频的像素级时空定位基准，包含短/中/长时视频评测集和大规模训练集 EgoMask-Train，通过系统分析揭示了自我中心与外中心视频之间的关键差异，并证明微调后模型性能可大幅提升。
tags:
  - ICCV 2025
  - 视频理解
  - egocentric video
  - spatiotemporal grounding
  - 图像分割
  - benchmark
---

# Fine-grained Spatiotemporal Grounding on Egocentric Videos

**会议**: ICCV 2025  
**arXiv**: [2508.00518](https://arxiv.org/abs/2508.00518)  
**代码**: [https://github.com/LaVi-Lab/EgoMask](https://github.com/LaVi-Lab/EgoMask)  
**领域**: video_understanding  
**关键词**: egocentric video, spatiotemporal grounding, pixel-level segmentation, benchmark, video understanding

## 一句话总结

提出 EgoMask，首个面向自我中心视频的像素级时空定位基准，包含短/中/长时视频评测集和大规模训练集 EgoMask-Train，通过系统分析揭示了自我中心与外中心视频之间的关键差异，并证明微调后模型性能可大幅提升。

## 研究背景与动机

时空视频定位（Spatiotemporal Video Grounding）旨在根据文本查询在视频中定位目标实体。现有研究主要集中在外中心（exocentric）视频上，自我中心（egocentric）视频虽在 AR 和机器人等领域日益重要，但像素级时空定位仍未被充分探索。

核心差异的定量分析：自我中心视频中的实体呈现——
- **总出现时长更短**：仅 21.56%（vs 外中心 77-94%）
- **连续轨迹更稀疏**：单次轨迹仅占视频 1.33%（vs 外中心 65-90%），消失时长为出现的 6 倍
- **目标更小**：mask 面积仅 1.20%（vs 外中心 5%+）
- **位置偏移更大**：相邻帧 mask IoU 仅 14.96%（vs 外中心 50%+）

现有数据集（EgoTracks 仅有 bbox、RefEgo 仅含短视频）无法支撑像素级评测，亟需新基准。

## 方法详解

### 整体框架

本文的贡献核心是数据集构建而非模型设计。设计了一个自动标注流水线，分两部分：1) 像素级 mask 生成；2) 指代表达式生成。以此构建评测基准 EgoMask 和训练数据集 EgoMask-Train。

### 关键设计

1. **像素级 Mask 生成**：基于 EgoTracks 提供的 bbox 标注，利用 SAM2 进行视频级对象分割。仅选取包含目标对象的片段进行标注，以第一帧的 bbox 作为 SAM2 的 box prompt。后处理时仅保留与 bbox 标注重叠的区域，最大限度减少幻觉错误。

2. **指代表达式生成**：采用两种策略确保多样性——1) 直接提示 GPT-4o 生成短/长描述（选取 3 个目标最清晰的帧，画框后让模型生成）；2) 先让 GPT-4o 生成元数据（视觉属性、世界知识、物体功能等），再用预定义模板组合成表达式。最终所有标注经人工校验。

3. **多时长分层评测设计**：

    - EgoMask-Short（<1分钟，200 video，400 expr.）：基于 RefEgo 采样，人工标注 mask 和精化表达式
    - EgoMask-Medium（1-3分钟，100 video，200 expr.）：从标注好的长视频中随机截取
    - EgoMask-Long（>3分钟，15 video，100 expr.）：基于 EgoTracks 验证集，用流水线生成后人工精化
    - EgoMask-Train：2,624 视频，9,592 个对象，47,968 个表达式

### 损失函数 / 训练策略

微调方案针对两个 SOTA 模型：
- **Sa2VA-4B (+FT)**：在 EgoMask-Train + 3 个外中心视频分割数据集上微调，8 张 A100，约 10 小时，AdamW lr=4e-6，batch size=16
- **VideoLISA-3.8B (+FT)**：80% EgoMask-Train + 20% 原始训练数据，4 张 A100，20 epochs，AdamW lr=3e-5，batch size=16，每 epoch 500 步，约 12 小时

### 评测指标设计

提出 4 个指标：
- **T_recall**：预测帧占目标帧的比例（时间定位能力）
- **IoU_all**：所有帧上的平均 IoU（传统指标 J）
- **IoU_gold**：仅在目标帧上计算平均 IoU
- **IoU_gold_pred**：在所有目标帧和预测帧上计算 IoU（惩罚背景帧上的幻觉预测）

## 实验关键数据

### 主实验

EgoMask 基准上的结果（IoU_gold_pred）：

| 方法 | Short | Medium | Long |
|------|-------|--------|------|
| Grounded-SAM2 | 49.95 | 25.73 | 24.80 |
| Sa2VA-26B | 37.30 | 25.83 | 12.96 |
| Sa2VA-4B | 29.00 | 17.02 | 8.11 |
| Sa2VA-4B (+FT) | 30.97 (+1.97) | 18.52 (+1.50) | 8.24 (+0.13) |
| VideoLISA-3.8B | 17.85 | 6.48 | 5.15 |
| VideoLISA-3.8B (+FT) | **23.36 (+5.51)** | **9.98 (+3.50)** | **7.16 (+2.01)** |

外中心基准上微调前后对比（验证不损害原有能力）：

| 方法 | Ref-Davis | Mevis | ReasonVOS |
|------|-----------|-------|-----------|
| VideoLISA-3.8B | 65.82 | 49.20 | 42.41 |
| VideoLISA-3.8B (+FT) | 65.60 (-0.22) | 49.20 (+0.00) | 44.18 (+1.77) |
| Sa2VA-4B | 69.75 | 50.01 | 42.35 |
| Sa2VA-4B (+FT) | 69.97 (+0.22) | 55.55 (+5.54) | 45.54 (+3.19) |

### 消融实验

SAM2 初始化状态的影响（Grounded-SAM2）：

| EgoMask 子集 | 最高置信度初始化 IoU_gold_pred | 朴素初始化 IoU_gold_pred | 差距 |
|------------|---------------------------|----------------------|------|
| Short | 49.95 | 40.42 | -9.53 |
| Medium | 25.73 | 15.11 | -10.62 |
| Long | 24.80 | 11.65 | -13.15 |

Sa2VA 关键帧有效性影响：当前 5 帧不包含目标对象时，性能暴跌至接近 0%。

### 关键发现

- 所有 SOTA 模型在自我中心视频上表现不佳：Short 子集最佳仅 ~50% IoU_gold_pred，Medium/Long 不到 30%
- 微调效果显著：VideoLISA 平均相对提升 41.30%，同时保留外中心基准性能
- EgoMask-Train 数据与外中心数据互补——微调后在 ReasonVOS 反而提升 1.77%
- SAM2 初始化状态至关重要：错误初始化导致性能暴跌 10-18%
- IoU_all 指标在长视频中因背景帧占大比例而误导性强（Sa2VA 的 IoU_all 随视频变长反而增高），IoU_gold_pred 更能反映真实性能
- 推理速度：VideoLISA（逐帧分割）仅 0.42 FPS，SAM2-based 方法至少 3.17 FPS

## 亮点与洞察

- **系统性差异分析**：首次定量揭示自我中心 vs 外中心视频的 4 个关键维度差异，为后续建模提供指导
- **全自动标注流水线**：SAM2 + GPT-4o 的组合大幅降低标注成本，仅需人工精化而非从零标注
- **多时长分层设计**：Short/Medium/Long 三档覆盖了从秒级到分钟级的实际应用场景
- **评测指标创新**：IoU_gold_pred 惩罚幻觉预测，比传统 IoU_all 更适合稀疏目标场景
- 数据集具有真正的补充价值——微调后外中心性能不降反升

## 局限与展望

- 当前模型设计未针对自我中心视频特性进行专门优化
- Sa2VA 受限于输入 token 数量，关键帧选择不灵活——长视频中目标可能不在前 5 帧出现
- 训练集定位标注为 1 FPS（vs EgoTracks 原始 5 FPS），可能遗漏快速运动细节
- Long 子集仅 15 个视频、100 个表达式，规模偏小
- 可探索方向：增强长视频理解能力、优化帧选择策略以捕获目标实体

## 相关工作与启发

- 与 RVOS 数据集（Ref-DAVIS、MeViS、ReasonVOS）相比，EgoMask 首次覆盖自我中心视频且支持多时长评测
- Sa2VA 在外中心基准上优于 Grounded-SAM2，但在自我中心基准上反而劣于后者——说明端到端模型未充分挖掘预训练定位模型在自我中心场景的潜力
- 帧选择策略是提升 SAM2-based 方法的关键方向

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个像素级自我中心时空定位基准，填补重要空白
- **实验充分度**: ⭐⭐⭐⭐ 多模型对比 + 微调验证 + 详细分析 + 多指标评测
- **写作质量**: ⭐⭐⭐⭐ 分析透彻，统计数据详实，可视化丰富
- **价值**: ⭐⭐⭐⭐⭐ 数据集和分析对自我中心视频理解社区有重大推动意义

<!-- RELATED:START -->

## 相关论文

- [Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos](../../CVPR2026/video_understanding/mistake_attribution_fine-grained_mistake_understanding_in_egocentric_videos.md)
- [Fine-Grained Captioning of Long Videos through Scene Graph Consolidation](../../ICML2025/video_understanding/fine-grained_captioning_of_long_videos_through_scene_graph_consolidation.md)
- [VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization](videominer_iteratively_grounding_key_frames_of_hour-long_videos_via_tree-based_g.md)
- [Beyond the Frame: Generating 360° Panoramic Videos from Perspective Videos](beyond_the_frame_generating_360deg_panoramic_videos_from_perspective_videos.md)
- [Moment Quantization for Video Temporal Grounding](moment_quantization_for_video_temporal_grounding.md)

<!-- RELATED:END -->
