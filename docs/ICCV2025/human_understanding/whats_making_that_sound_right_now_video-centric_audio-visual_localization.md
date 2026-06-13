---
title: >-
  [论文解读] What's Making That Sound Right Now? Video-centric Audio-Visual Localization
description: >-
  [人体理解] 提出视频级音视频定位基准 AVATAR 和时序感知模型 TAVLO，通过高分辨率时序建模解决传统 AVL 方法忽略时间动态的问题。 音视频定位（AVL）旨在在视觉场景中识别发出声音的物体。现有研究存在两个关键局限： 图像级关联的局限：现有基准（如 Flickr-SoundNet、VGGSS）采用图像级标注方式—…
tags:
  - "人体理解"
---

# What's Making That Sound Right Now? Video-centric Audio-Visual Localization

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2507.04667](https://arxiv.org/abs/2507.04667)
- **代码**: [项目页面](https://hahyeon610.github.io/Video-centric_Audio_Visual_Localization/)
- **领域**: 人体理解
- **关键词**: 音视频定位, 时序建模, 视频级标注, 多场景评估, 自监督学习

## 一句话总结

提出视频级音视频定位基准 AVATAR 和时序感知模型 TAVLO，通过高分辨率时序建模解决传统 AVL 方法忽略时间动态的问题。

## 研究背景与动机

音视频定位（AVL）旨在在视觉场景中识别发出声音的物体。现有研究存在两个关键局限：

**图像级关联的局限**：现有基准（如 Flickr-SoundNet、VGGSS）采用图像级标注方式——标注者观看完整视频后仅标注单帧中的发声物体，将该帧视为整个视频的代表。这导致现有方法仅处理单帧输入，完全忽略时间动态。在真实场景中需要追踪移动的声源并处理动态变化，时空建模至关重要。

**过度简化的假设**：现有基准假设发声物体始终可见且通常仅涉及单个声源。然而真实场景常包含多个同时发声源、发声物体可能在画面外等复杂情况。部分研究通过构建不匹配的音频-图像负样本对或合成混合音频来部分解决，但仍不够全面。

## 方法详解

### 整体框架

本文提出两个核心贡献：(1) AVATAR 基准数据集；(2) TAVLO 时序感知 AVL 模型。

### AVATAR 基准

AVATAR 基于 VGGSound 构建，采用半自动标注管线，包含 5000 个视频、24266 帧、80 个类别。关键创新在于引入四种评估场景：

- **Single-sound**：帧中仅一个实例发出声音，评估一对一音视频对应
- **Mixed-sound**：多个同时发声源，需要区分和关联声音与正确视觉源
- **Multi-entity**：多个视觉相似物体中仅一个发声，需要时空推理区分
- **Off-screen**：发声物体在画面外，评估模型避免假阳性的能力

半自动标注管线包含三个阶段：
1. 候选视频选取：从 VGGSound 获取约 70k 原始视频，经分辨率、帧率、时长等过滤后得到 39k 视频
2. 自动片段和帧采样：基于 RMS 能量检测音频活跃区域，通过拉普拉斯滤波器选择最清晰帧
3. 模型驱动标注：YoloV8 检测 + CAV-MAE 音频分类 + SAM 实例分割 + 人工验证

### TAVLO 模型

TAVLO 显式融合时间信息进行时空 AVL。核心架构包括：

**模态特定特征编码**：

视觉编码器 $f_v$（ResNet-18）提取帧级特征 $\mathbf{V} = f_v(V) \in \mathbb{R}^{T \times H \times W \times D_f}$。音频编码器 $f_a$ 使用矩形 2D CNN 核，核大小设计为 $K_w = \lfloor W_a / T \rfloor, K_h = H_a$，确保每个音频段与对应视觉帧对齐，输出 $\mathbf{A} = f_a(A) \in \mathbb{R}^{T \times D_f}$。

**位置编码**：定义空间位置编码 $\text{Pos}_s \in \mathbb{R}^{T \times H \times W \times D_s}$ 和时间位置编码 $\text{Pos}_t \in \mathbb{R}^{T \times D_t}$：

$$\tilde{\mathbf{V}} = [\mathbf{V} + \text{Pos}_s; \text{Pos}_t] \in \mathbb{R}^{T \times H \times W \times D}$$
$$\tilde{\mathbf{A}} = [\mathbf{A}; \text{Pos}_t] \in \mathbb{R}^{T \times D}$$

空间编码与视觉特征逐元素相加，时间编码跨两个模态拼接，最终维度 $D = D_f + D_t$。

**AST 注意力模块**：将音视频特征拼接为 $\mathbf{Z}^0 = [\tilde{\mathbf{A}}; \tilde{\mathbf{V}}] \in \mathbb{R}^{T \times (1+H \cdot W) \times D}$，采用分解注意力策略避免直接对展平视频特征计算自注意力的二次复杂度：

- **空间注意力**：在 $1 + H \cdot W$ 维度上执行多头自注意力，捕获单帧内音视频跨模态交互
- **时间注意力**：在 $T$ 维度上执行多头自注意力，建模跨帧时间依赖

### 训练目标

基于 EZ-VSL 的跨模态多实例对比学习损失进行修改：(1) 引入时间分量在帧级定义视觉 bag；(2) 对负样本 bag 使用均值相似度替代最大值，减少噪声实例主导损失：

$$\mathcal{L}_{a \rightarrow v} = -\mathbb{E}_{t,i}\left[\log \frac{\exp(\text{p}_i^t)}{\exp(\text{p}_i^t) + \sum_{j \neq i}^B \exp(\text{n}_{ij}^t)}\right]$$

最终损失为双向对齐：$\mathcal{L} = \mathcal{L}_{a \rightarrow v} + \mathcal{L}_{v \rightarrow a}$。

## 实验

### 主实验结果

| 方法 | Single-sound CIoU(%) | Single-sound AUC(%) | Mixed-sound CIoU(%) | Multi-entity CIoU(%) | Off-screen TN(%) |
|------|---------------------|--------------------|--------------------|---------------------|-----------------|
| SLAVC(144k) | 9.07 | 10.60 | 6.31 | 6.41 | 96.46 |
| EZ-VSL(10k) | 9.66 | 11.07 | 8.16 | 6.87 | 96.91 |
| EZ-VSL(144k) | 10.92 | 12.22 | 6.97 | 5.80 | 96.47 |
| SSL-TIE(144k) | 13.10 | 14.23 | 5.19 | 5.50 | 90.82 |
| **TAVLO(10k)** | **13.42** | **14.08** | **14.13** | **12.08** | 91.18 |

### Cross-event 场景下的鲁棒性

| 方法 | Total CIoU(%) | Cross-event CIoU(%) | Δ |
|------|--------------|--------------------|----|
| EZ-VSL(full) | 10.50 | 5.26 | -5.24 |
| SSL-TIE(144k) | 10.39 | 5.03 | -5.36 |
| **TAVLO(10k)** | **13.37** | **13.04** | **-0.33** |

### 关键发现

1. TAVLO 仅用 10k 训练数据即超越使用 144k 数据训练的基线方法
2. 在 Mixed-sound 和 Multi-entity 场景中优势最为显著，CIoU 提升约 6-7 个百分点
3. Cross-event 场景下，基线方法 CIoU 下降 3-5 个百分点，而 TAVLO 仅下降 0.33，表明时序建模对动态声源追踪至关重要
4. 定性分析表明 TAVLO 能正确区分多面鼓场景中的实际发声鼓和画面外转画面内说话场景

## 亮点与洞察

1. **问题定义精准**：首次系统性地将 AVL 扩展到视频维度，定义了四种全面的评估场景
2. **分解注意力设计精妙**：AST 模块通过空间-时间分解注意力，以线性时间处理视频特征，避免二次复杂度
3. **数据效率高**：仅需 10k 训练样本即可超越 144k 训练的基线，体现时序建模带来的归纳偏置优势
4. **半自动标注管线实用**：结合模型辅助和人工验证，平衡了标注质量和效率

## 局限性

1. 假设音视频至少存在部分对齐，画面外声音的独立定位仍是开放问题
2. 基准未为每种场景提供特定优化策略
3. Off-screen 评估中阈值选择对结果影响较大

## 相关工作

- **AVL 方法**：EZ-VSL、SSL-TIE、SLAVC 等基于自监督学习的方法，DMT 半监督方法
- **AVL 基准**：Flickr-SoundNet、VGGSS、AVSBench 等，均缺乏视频级时序标注
- **视频理解**：时空注意力分解策略（TimeSformer、ViViT）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 视频级 AVL 基准和时序感知模型是领域首创
- **技术深度**: ⭐⭐⭐⭐ — 时序编码和 AST 注意力设计合理
- **实验充分性**: ⭐⭐⭐⭐ — 四种场景全面评估，Cross-event 分析有说服力
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机明确

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Sequential Keypoint Density Estimator: An Overlooked Baseline of Skeleton-Based Video Anomaly Detection](sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_v.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)
- [\[ICCV 2025\] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [\[ICCV 2025\] SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning](semges_semantics-aware_co-speech_gesture_generation_using_semantic_coherence_and.md)
- [\[CVPR 2025\] ControlFace: Harnessing Facial Parametric Control for Face Rigging](../../CVPR2025/human_understanding/controlface_harnessing_facial_parametric_control_for_face_rigging.md)

</div>

<!-- RELATED:END -->
