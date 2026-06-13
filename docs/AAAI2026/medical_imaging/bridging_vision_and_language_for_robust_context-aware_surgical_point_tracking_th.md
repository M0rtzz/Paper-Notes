---
title: >-
  [论文解读] Bridging Vision and Language for Robust Context-Aware Surgical Point Tracking: The VL-SurgPT Dataset and Benchmark
description: >-
  [AAAI 2026][医学图像][手术点追踪] 提出首个大规模多模态手术点追踪数据集 VL-SurgPT，结合视觉坐标与文本状态描述，并设计文本引导追踪方法 TG-SurgPT，通过语义信息显著提升复杂手术场景下的追踪精度和鲁棒性。
tags:
  - "AAAI 2026"
  - "医学图像"
  - "手术点追踪"
  - "视觉-语言多模态"
  - "数据集"
  - "文本引导"
  - "手术视频分析"
---

# Bridging Vision and Language for Robust Context-Aware Surgical Point Tracking: The VL-SurgPT Dataset and Benchmark

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.12026](https://arxiv.org/abs/2511.12026)  
**代码**: [有](https://szupc.github.io/VL-SurgPT/)  
**领域**: 医学图像  
**关键词**: 手术点追踪, 视觉-语言多模态, 数据集, 文本引导, 手术视频分析

## 一句话总结

提出首个大规模多模态手术点追踪数据集 VL-SurgPT，结合视觉坐标与文本状态描述，并设计文本引导追踪方法 TG-SurgPT，通过语义信息显著提升复杂手术场景下的追踪精度和鲁棒性。

## 研究背景与动机

手术环境中的精确点追踪对计算机辅助手术至关重要，可支撑运动理解和场景感知等应用。然而手术场景引入了极端的视觉挑战：

- **电凝烟雾**：密集烟雾遮挡追踪目标
- **镜面反射**：湿润解剖表面产生强反光
- **器械遮挡**：手术器械动态遮挡组织
- **组织形变**：非刚性组织在手术操作中持续变形

**现有数据集的局限**：

**缺乏语义上下文**：TAP-Vid、SurgT、STIR 等数据集仅提供几何坐标，无法解释追踪失败的原因

**单模态局限**：所有现有数据集都是纯视觉标注，无文本描述

**覆盖不全**：多数数据集仅关注组织追踪或器械追踪中的一种

**离体设置**：许多数据集基于离体（ex vivo）设置，无法反映真实手术复杂性

核心假设：**引入语义文本描述可以为追踪提供纯视觉方法无法获取的上下文引导，从而在视觉特征不可靠的情况下显著提升追踪鲁棒性。**

## 方法详解

### 整体框架

本文贡献包含两大部分：VL-SurgPT 数据集构建和 TG-SurgPT 追踪方法。

#### VL-SurgPT 数据集

数据来源于深圳人民医院的达芬奇 Xi 手术系统，涵盖胃癌根治术、直肠癌根治术等复杂手术，共 20 台手术约 33 小时视频，筛选后 115 分钟用于标注。

**数据采集流程**：

1. 术中使用 ICG（吲哚菁绿）荧光染料标记追踪点，UV 模式下获取高精度坐标作为 ground truth
2. 切换至白光模式执行正常手术操作 5-10 秒
3. 再次激活 UV 模式记录终点坐标

**标注内容**（每个点包含）：

- 2D 坐标 $(x, y)$ 或 "null"（不可见时）
- 点类型：组织（Tissue）或器械（Instrument）
- **点状态文本描述**：Clear View / Pulled / Reflection / Smoke Obscuration / Instrument Obscuration / Tissue Obscuration / Out of View / External Occlusion / Self-occlusion
- 器械特有标签：器械类型（7 种）、实例 ID

**数据规模**：

- 组织追踪子集：754 个视频片段，180.5k 帧，7,117 标注帧，1,862 轨迹，17,171 可见点
- 器械追踪子集：154 个视频片段，26,490 帧，1,108 标注帧，7 种器械
- 覆盖 5 种挑战场景：组织形变、器械遮挡、相机抖动、表面反射、电凝烟雾

### 关键设计：TG-SurgPT

TG-SurgPT 基于 Track-On 框架扩展，通过双分支架构整合视觉和文本模态：

**文本分支**：对每个追踪点编码两类语义属性（点类型 + 点状态），通过冻结的 CLIP Text Encoder 生成文本特征 $F_t \in \mathbb{R}^{2 \times 512}$。

**视觉分支**：冻结的 Track-On 模型处理查询点和视频帧，输出查询点特征 $F_q$、当前帧稠密特征 $F_h$、粗匹配位置 $F_p$ 和初始坐标 $(X_N, Y_N)$。

**属性预测头（Attributes Pred Head）**：

- 融合 $F_p$ 和 $F_h$ 通过多尺度相关性模块得到 $F_{p\text{-}h}$
- 以 $F_q$ 为 Query，$F_{p\text{-}h}$ 为 Key/Value 进行多尺度可变形注意力
- 两个并行分类头：Point Type Head（2×N）和 Point Status Head（7×N 或 4×N）
- **关键**：推理时使用模型自身预测的状态标签代替 GT 文本，实现全自动文本引导

**文本引导注意力（Text-guided Attention）**：

- 文本特征 $F_t$ 通过线性变换投影到与 $F_q$ 同维
- 标准交叉注意力：$F_q$ 为 Query，$F_t$ 为 Key/Value → 融合表示 $F_{t\text{-}q}$
- $F_{t\text{-}q}$ 与 $F_{p\text{-}h}$ 通过可变形注意力生成精修偏移 $(a_N, b_N)$
- 最终预测：$(\hat{X}_N, \hat{Y}_N) = (X_N + a_N, Y_N + b_N)$

### 损失函数 / 训练策略

总损失包含三项：

$$\mathcal{L} = \sum_{t \in \mathcal{T}} \left( \underbrace{\mathcal{H}_\delta(\hat{\mathbf{p}}_t - \mathbf{p}_t)}_{\mathcal{L}_p} + \underbrace{\|\Delta^2 \hat{\mathbf{p}}_t\|_1}_{\mathcal{L}_s} + \underbrace{\mathcal{L}_{CE}(\hat{\mathbf{s}}_t, \mathbf{s}_t)}_{\mathcal{L}_{text}} \right)$$

- **点距离损失** $\mathcal{L}_p$：Huber 损失衡量预测与 GT 坐标距离
- **轨迹平滑损失** $\mathcal{L}_s$：最小化预测轨迹的二阶差分，促进时序平滑
- **文本分类损失** $\mathcal{L}_{text}$：交叉熵损失监督状态标签分类

训练特点：可使用**稀疏标注的真实手术数据**训练，无需依赖密集合成数据（如 MOVi-F），得益于文本标注的补充监督。

## 实验关键数据

### 主实验

8 种 SOTA 追踪方法 + TG-SurgPT 在 VL-SurgPT 上的对比：

**组织追踪子集：**

| 方法 | AJ↑ | $<\delta_{avg}^x$↑ | OA↑ | EPE↓ |
|------|-----|-----|-----|------|
| RAFT | 27.81 | 30.37 | 85.81 | 99.73 |
| BootsTAP | 56.93 | 62.77 | 87.87 | 23.52 |
| MFTIQ | 61.52 | 63.44 | 87.80 | 19.81 |
| Track-On | 58.55 | 66.27 | 88.81 | 13.79 |
| **TG-SurgPT** | **62.88** | **67.77** | **91.04** | **11.02** |

**器械追踪子集：**

| 方法 | AJ↑ | $<\delta_{avg}^x$↑ | OA↑ | EPE↓ |
|------|-----|-----|-----|------|
| Track-On | 46.97 | 59.18 | 85.07 | 41.67 |
| **TG-SurgPT** | **49.52** | **62.94** | **89.79** | **39.14** |

器械追踪改进更显著：AJ +5.4%，位置精度 +6.4%。推理速度 9.72 fps，接近临床实时要求。

### 消融实验

| 微调 | 片段长度 | 文本 | 组织AJ↑ | 组织EPE↓ | 器械AJ↑ | 器械OA↑ |
|------|---------|------|---------|----------|---------|---------|
| ✗ | - | ✗ | 58.55 | 13.79 | 46.97 | 85.07 |
| ✓ | short | ✗ | 61.09 | 12.82 | 47.48 | 86.43 |
| ✓ | short | ✓ | **62.88** | **11.02** | **49.52** | **89.79** |

- 微调提升 AJ +4.3%（组织），文本引导在此基础上再提升 +2.9%
- 短片段（31帧）优于长片段（181帧），因短窗口减少噪声
- 文本引导对器械追踪 OA 提升 +3.9%，对抗遮挡效果尤为突出

### 关键发现

- **场景难度层级**：组织形变最难，电凝烟雾反而最容易（烟雾产生独特的时空模式，注意力机制可有效利用）
- **文本引导优势在困难场景中更大**：在器械遮挡和相机抖动场景中，TG-SurgPT 的改进最为显著
- **视觉状态预测精度**：Clear View（78.5%）和 Pulled（85.1%）分类精度高，Out of View（48.1%）最难预测

## 亮点与洞察

1. **首创视觉-语言手术追踪数据集**：VL-SurgPT 是第一个同时提供坐标和语义状态描述的手术追踪数据集，填补了重要空白
2. **推理时自动化**：训练用 GT 文本描述，推理时用模型自己预测的状态标签，无需人工标注即可实现文本引导
3. **ICG 荧光标记获取 GT**：巧妙利用手术中已有的 ICG 标记技术获取高精度 ground truth，贴合临床实际
4. **稀疏标注训练**：不需要密集合成数据，直接在真实手术数据上训练，降低了部署门槛

## 局限与展望

- **手术类型有限**：仅覆盖胃肠道手术（达芬奇 Xi 系统），需扩展到其他手术领域
- **推理速度不足**：9.72 fps 尚未达到临床实时要求（通常需要 ≥25 fps）
- **点状态预测精度有限**：Out of View 仅 48.1% 准确率，遮挡消歧能力仍需提升
- **仅含 2D 追踪**：未涉及 3D 重建或深度信息，限制了在 3D 手术导航中的应用

## 相关工作与启发

- **Track-On**：TG-SurgPT 的视觉骨干，平衡了效率和性能（10.85 fps）
- **SurgMotion**：最新的手术追踪方法，但仍为纯视觉单模态
- **CLIP**：冻结的 CLIP 文本编码器用于语义特征提取，证明预训练 VL 模型可有效迁移到手术领域
- 文本引导追踪的思路可推广到其他需要解释追踪失败原因的应用场景

## 评分

- **创新性**: ★★★★★ — 首次将语言引入手术点追踪，数据集和方法都具有开创性
- **实验充分度**: ★★★★★ — 8 种基线对比、详细场景分析、完整消融
- **写作质量**: ★★★★☆ — 结构清晰，数据集构建流程描述详尽
- **实用性**: ★★★★☆ — 数据集价值极高，但推理速度仍需优化

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools](../../CVPR2025/medical_imaging/cholectrack20_a_multi-perspective_tracking_dataset_for_surgical_tools.md)
- [\[AAAI 2026\] Rethinking Surgical Smoke: A Smoke-Type-Aware Laparoscopic Video Desmoking Method and Dataset](rethinking_surgical_smoke_a_smoke-type-aware_laparoscopic_video_desmoking_method.md)
- [\[AAAI 2026\] DiA-gnostic VLVAE: Disentangled Alignment-Constrained Vision Language Variational AutoEncoder for Robust Radiology Reporting with Missing Modalities](dia-gnostic_vlvae_disentangled_alignment-constrained_vision_language_variational.md)
- [\[AAAI 2026\] Sim4Seg: Boosting Multimodal Multi-disease Medical Diagnosis Segmentation with Region-Aware Vision-Language Similarity Masks](sim4seg_boosting_multimodal_multi-disease_medical_diagnosis_segmentation_with_re.md)
- [\[AAAI 2026\] Towards Effective and Efficient Context-aware Nucleus Detection in Histopathology Whole Slide Images](towards_effective_and_efficient_context-aware_nucleus_detection_in_histopatholog.md)

</div>

<!-- RELATED:END -->
