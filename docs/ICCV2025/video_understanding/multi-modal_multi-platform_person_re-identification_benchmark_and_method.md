---
title: >-
  [论文解读] Multi-modal Multi-platform Person Re-Identification: Benchmark and Method
description: >-
  [ICCV 2025][视频理解][行人重识别] 提出首个多模态多平台行人重识别基准 MP-ReID（含 RGB、红外、热成像三种模态 + 地面和无人机两种平台）和统一提示学习框架 Uni-Prompt ReID，通过模态感知、平台感知和视觉增强提示显著提升复杂场景下的 ReID 性能。 传统 ReID 研究主要局限于单一模…
tags:
  - "ICCV 2025"
  - "视频理解"
  - "行人重识别"
  - "多模态"
  - "多平台"
  - "提示学习"
  - "CLIP"
---

# Multi-modal Multi-platform Person Re-Identification: Benchmark and Method

**会议**: ICCV 2025  
**arXiv**: [2503.17096](https://arxiv.org/abs/2503.17096)  
**代码**: [GitHub](https://mp-reid.github.io/)  
**领域**: 视频理解  
**关键词**: 行人重识别, 多模态, 多平台, 提示学习, CLIP

## 一句话总结

提出首个多模态多平台行人重识别基准 MP-ReID（含 RGB、红外、热成像三种模态 + 地面和无人机两种平台）和统一提示学习框架 Uni-Prompt ReID，通过模态感知、平台感知和视觉增强提示显著提升复杂场景下的 ReID 性能。

## 研究背景与动机

传统 ReID 研究主要局限于**单一模态（RGB）+ 固定摄像头**的设置，无法应对真实城市环境中日益复杂的多传感器部署。考虑一个 24/7 城市行人监控系统：

- **地面 RGB 摄像头**：白天场景
- **红外/热成像传感器**：夜间或恶劣光照
- **无人机 (UAV)**：动态追踪、灵活视角

这种多模态 + 多平台配置面临三重挑战：(1) 模态差异（RGB vs 红外 vs 热成像的外观鸿沟）、(2) 平台差异（地面 vs 航拍的视角、分辨率差异）、(3) 两者同时存在时的极端困难条件。

**现有数据集的不足**：
- 跨模态数据集（SYSU-MM01、LLCM）仅覆盖 RGB + 红外，且只有地面摄像头
- UAV 数据集（AG-ReID）仅覆盖 RGB 模态
- **没有任何数据集同时包含多模态 + 多平台**

这一关键空白促使作者构建 MP-ReID 并设计相应的统一学习框架。

## 方法详解

### 整体框架

Uni-Prompt ReID 基于 CLIP 视觉-语言模型，通过精心设计的多部分文本提示进行微调学习。框架包含三类可学习提示和一个视觉增强网络，用于将图像特征融入文本提示空间。

### 关键设计

1. **MP-ReID 数据集构建**

   跨越 3 种模态 × 2 种平台：
    - 地面 RGB（6 台海康威视 1920×1080 全彩相机）
    - 地面红外（6 台红外夜视模式相机）
    - UAV RGB（DJI Mavic 3T，3840×2160）
    - UAV 热成像（DJI Mavic 3T 热感相机，640×512）

   数据规模：1930 个身份、136,156 个标注框、14 台相机、总视频时长超 13 小时。UAV 在 5m/7m/10m 三种高度以 30-80° 角度采集。所有数据经过面部马赛克处理并删除原始素材以保护隐私。

2. **Uni-Prompt 多部分文本提示**

   文本提示由三部分串联组成：

    $t_i(a) = X_1(a) \cdots X_M(a) \; P_1(a) \cdots P_R(a) \; M_1(a) \cdots M_B(a), \text{person}_i$

    - **Specific ReID Prompt** ($X$)：编码个体特有信息（身份级别）
    - **Modality-Aware Prompt** ($M$)：捕获模态特定细节（RGB vs 红外 vs 热成像）
    - **Platform-Aware Prompt** ($P$)：融入平台特定上下文（地面 vs 航拍）

3. **视觉增强网络 (Visual-Enhanced Network)**

   设计轻量级神经网络 $g_\theta(\cdot)$ 将图像特征 $a$ 映射为上下文向量：

    $\sigma = (\sigma_X, \sigma_P, \sigma_M) = g_\theta(a)$

   然后加到对应提示上：$S_m(a) = [S]_m + \sigma_S$

   直觉：红外图像的视觉特征天然包含模态线索，可引导模态感知提示向红外方向学习。

### 损失函数 / 训练策略

**两阶段训练**：

- 阶段一：冻结模态和平台提示，用 CLIP-ReID 的 $\mathcal{L}_{i2t} + \mathcal{L}_{t2i}$ 学习 Specific ReID Prompt
- 阶段二：冻结 ReID Prompt，用模态级和平台级对比损失学习其余提示

$$\mathcal{L}_{\text{Uni-Prompt}} = \mathcal{L}_{mi2t} + \mathcal{L}_{mt2i} + \mathcal{L}_{pi2t} + \mathcal{L}_{pt2i}$$

其中每项均为对比学习损失（InfoNCE 形式），分别对模态标签和平台标签进行对齐。数据增强包括随机擦除 (p=0.5)、随机水平翻转和随机裁剪。

## 实验关键数据

### 主实验

**MP-ReID 基准三类设置平均结果**

| 方法 | 跨平台 Rank-1 | 跨模态 Rank-1 | 跨模态+平台 Rank-1 | 平均 Rank-1 | 平均 mAP |
|------|-------------|-------------|-----------------|-----------|---------|
| CAJ | 40.36 | 45.34 | 10.62 | 32.11 | 21.51 |
| CAJ+ | 47.60 | 58.16 | 21.51 | 42.42 | 30.61 |
| AGW | 53.68 | 51.88 | 19.21 | 41.59 | 30.56 |
| DEEN | 60.05 | 69.59 | 27.59 | 52.41 | 39.33 |
| OTLA-ReID | 73.24 | 68.12 | 29.31 | 56.89 | 43.03 |
| **Uni-Prompt** | **78.77** | **72.26** | **43.16** | **64.73** | **58.45** |

平均 Rank-1 提升 **+7.87%**，mAP 提升 **+15.42%**。在最困难的跨模态+跨平台设置中提升最为显著（+13.85% Rank-1）。

### 消融实验

| 配置 | 跨平台 R1 | 跨模态 R1 | 跨模态+平台 R1 | 平均 R1 | 平均 mAP |
|------|---------|---------|--------------|--------|---------|
| Base (ReID Prompt) | 77.01 | 61.11 | 28.40 | 55.51 | 47.98 |
| +Modality-Aware | 77.18 | 67.34 | 31.57 | 58.70 | 51.67 |
| +Platform-Aware | 78.62 | 70.31 | 40.66 | 63.20 | 57.48 |
| +Visual-Enhanced (Full) | **78.77** | **72.26** | **43.16** | **64.73** | **58.45** |

### 关键发现

- **跨模态+跨平台是最难的设置**：现有方法在此设置下性能急剧下降（CAJ 仅 10.62% Rank-1），而 Uni-Prompt 达到 43.16%，说明同时处理两类差异确实需要专门设计。
- **模态感知提示**主要在跨模态设置中有效（+6.23% Rank-1），对跨平台影响小。
- **平台感知提示**在跨模态+平台设置中贡献最大（+9.09% Rank-1），是解决最困难场景的关键。
- **视觉增强网络**在各设置中均有边际提升，跨模态+平台设置提升 2.50%，说明视觉线索对引导提示学习有辅助作用。
- 现有基线方法仅在地面 RGB 数据可用的单一差异设置下尚可接受，一旦涉及 UAV + 多模态就显著退化。

## 亮点与洞察

- **首个多模态多平台 ReID 基准**填补了关键空白——1930 身份、14 台相机、3 种模态、2 种平台，数据规模和多样性兼具。
- **统一提示学习框架**优雅地将模态和平台信息分解为独立的可学习提示，避免了复杂的特征融合网络。
- 两阶段训练策略（先学身份提示 → 再学模态/平台提示）类似于课程学习，确保模型先建立身份概念再学习跨域对齐。
- 隐私保护措施完善：面部马赛克、原始素材删除、伦理委员会审批、公告告知。

## 局限与展望

- 数据集规模受限于多模态多平台采集的高成本（1930 身份 vs MSMT17 的 4101 身份）。
- 仅在一个数据集上评估，未验证在其他数据集上的迁移性。
- 未涉及穿戴设备平台和事件相机模态，作者鼓励未来补充。
- 视觉增强网络的设计较简单（轻量级线性映射），更复杂的适配器可能进一步提升效果。
- UAV 热成像相机分辨率偏低（640×512），导致 YOLOX 跟踪性能不佳，需大量人工标注。

## 相关工作与启发

- CLIP-ReID 和 DAPrompt 为本工作中提示学习的基础；CoCoOp 启发了视觉-条件化提示的设计。
- SYSU-MM01、LLCM 等跨模态数据集局限于 RGB+红外 + 地面相机；AG-ReID 覆盖航拍但仅 RGB。
- MP-ReID 的多平台设计对智慧城市和公共安全场景有直接应用价值。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 数据集的多模态多平台设计是主要贡献，方法基于已有提示学习框架的渐进式扩展
- **实验充分度**: ⭐⭐⭐⭐ 12 种实验设置 + 详细消融 + 10 次重复取平均，设计严谨
- **写作质量**: ⭐⭐⭐⭐ 数据集描述详尽，方法叙述清晰
- **价值**: ⭐⭐⭐⭐⭐ 数据集和基准对行人重识别社区有重要推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence](../../NeurIPS2025/video_understanding/muvr_a_multi-modal_untrimmed_video_retrieval_benchmark_with_multi-level_visual_c.md)
- [\[ICCV 2025\] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning](aim_adaptive_inference_multimodal_llms_token_merging_pruning.md)
- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](4dbench_benchmarking_multimodal_large_language_models_for_4d.md)
- [\[ICCV 2025\] DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding](dynimg_key_frames_with_visual_prompts_are_good_representation_for_multi-modal_vi.md)
- [\[ICCV 2025\] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs](q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)

</div>

<!-- RELATED:END -->
