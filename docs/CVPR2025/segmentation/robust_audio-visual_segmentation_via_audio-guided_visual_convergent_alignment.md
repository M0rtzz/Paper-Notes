---
title: >-
  [论文解读] Robust Audio-Visual Segmentation via Audio-Guided Visual Convergent Alignment
description: >-
  [CVPR 2025][图像分割][音视频分割] 本文通过音频引导的模态对齐（AMA）和不确定性估计（UE）两个模块，解决音视频分割中视觉相似物体的错误关联和发声状态频繁变化导致的过/欠分割问题，在AVS-Semantic上提升4.2%。
tags:
  - CVPR 2025
  - 图像分割
  - 音视频分割
  - 模态对齐
  - 不确定性估计
  - 对比学习
  - 语义分组
---

# Robust Audio-Visual Segmentation via Audio-Guided Visual Convergent Alignment

**会议**: CVPR 2025  
**arXiv**: [2503.12847](https://arxiv.org/abs/2503.12847)  
**代码**: 无  
**领域**: Segmentation  
**关键词**: 音视频分割, 模态对齐, 不确定性估计, 对比学习, 语义分组

## 一句话总结

本文通过音频引导的模态对齐（AMA）和不确定性估计（UE）两个模块，解决音视频分割中视觉相似物体的错误关联和发声状态频繁变化导致的过/欠分割问题，在AVS-Semantic上提升4.2%。

## 研究背景与动机

音视频分割（AVS）旨在根据音频线索定位和分割视频中的发声物体。现有方法主要关注时空多模态建模，但忽略了两个关键挑战：

- **空间歧义**：场景中视觉相似但声学不同的物体（如两只外观相同但只有一只在叫的狗）靠近时，全局注意力机制难以区分发声和静默物体，导致过分割
- **时间歧义**：物体的发声状态频繁变化（如狗在不同时刻交替叫/不叫），现有时序建模倾向于平滑预测，忽略突变点

作者对AVSS-V2数据集进行统计分析发现，33.3%的采样子集中存在大量上述挑战性帧，说明这是一个普遍且重要的问题。

## 方法详解

### 整体框架

框架由视觉编码器（L个block）、音频编码器和mask解码器组成。在每个视觉编码器block后插入AMA模块进行音频引导的模态对齐。所有帧的融合特征经过时序注意力层后，同时输入mask预测头和不确定性估计头，用不确定性图调整最终预测置信度。

### 关键设计

**1. 音频引导模态对齐（AMA）：语义分组 + 声音引导合并 + 对比学习**

- **功能**：将模型注意力聚焦于与音频相关的视觉区域，区分发声和静默物体
- **核心思路**：首先用DPC-KNN密度峰值聚类将视觉特征分为 $P$ 个语义组；然后在组内进行音频-视觉交叉注意力，根据对音频的响应度加权合并组内特征为紧凑表示（高响应特征被放大，低响应被抑制）；通过多层迭代，发声区域被逐步增强。最后用InfoNCE对比损失拉远正样本（高音频响应）和负样本（低音频响应）的特征距离
- **设计动机**：全局注意力会导致注意力分散到所有视觉相似区域。通过分组限制交互范围+音频引导的特征竞争，强制模型区分发声/静默物体，而非依赖GT mask（如stepstone方法）

**2. 不确定性估计（UE）模块**

- **功能**：识别因发声状态频繁变化导致的高不确定性区域，降低这些区域的预测置信度
- **核心思路**：融合特征经时序注意力后，分两路输出：mask预测logits $m$ 和不确定性logits $\alpha$（基于Dirichlet分布）。最终预测通过加权整合不确定性图和mask概率图生成
- **设计动机**：时序建模天然倾向于生成平滑预测，在状态转换帧会产生高不确定性。通过显式估计不确定性并降低转换帧的置信度，避免在突变点产生错误的过分割

**3. 多层级紧凑表示更新**

- **功能**：通过多层transformer decoder迭代更新紧凑表示，逐步收敛高层语义
- **核心思路**：$G_l \leftarrow G_l + \text{softmax}(G_l f_{v_l}^T / \sqrt{D_l} + S) f_{v_l}$，将相关性分数 $S$ 加入注意力权重，确保对音频响应高的token贡献更大。更新后的紧凑表示被映射回视觉特征图进入下一层
- **设计动机**：单次对齐不够充分，多层迭代使发声区域被逐步增强而静默区域逐步减弱

### 损失函数

$$\mathcal{L} = \mathcal{L}_{\text{seg}} + \lambda \mathcal{L}_{\text{cst}} + \mu \mathcal{L}_{\text{unc}}$$

分割损失包含交叉熵、Dice和IoU损失；对比损失 $\mathcal{L}_{\text{cst}}$ 为InfoNCE；不确定性损失 $\mathcal{L}_{\text{unc}}$ 基于Dirichlet分布的KL散度。

## 实验关键数据

### 主实验：AVS基准对比

| 方法 | AVS-Object $\mathcal{J\&F}$↑ | AVS-Semantic $\mathcal{J\&F}_\beta$↑ | VPO-MSMI $\mathcal{J\&F}_\beta$↑ |
|------|---------------------------|----------------------------------|-------------------------------|
| TPAVI | 78.7 | 29.7 | - |
| CATR | 81.4 | 37.6 | - |
| 前SOTA | - | ~42.0 | ~35.0 |
| **Ours** | **84.1** | **+4.2 vs SOTA** | **+11.5 vs SOTA** |

### 消融实验：各模块贡献

| 组件 | AVS-Semantic $\mathcal{J\&F}_\beta$ |
|------|----------------------------------|
| Baseline | 37.6 |
| + 语义分组 | 39.8 |
| + 声音引导合并 | 41.2 |
| + 对比学习 | 42.8 |
| + 不确定性估计 | **44.2** |

### 关键发现

- AMA模块在VPO-MSMI（包含多个同时发声源的高难场景）上提升最大（+11.5%），证明其处理复杂场景的能力
- 不确定性估计在状态频繁转换帧上效果显著，有效减少了过分割
- 使用音频响应度自动构建正负样本优于使用GT mask构建，因为后者无法让模型学习每个物体的独特声学响应
- 分组数 $P$ 对性能有影响，过少无法区分相似物体，过多增加噪声

## 亮点与洞察

1. **从"全局对齐"到"分组竞争"的范式转变**：通过语义分组+音频引导的特征竞争替代全局注意力，更精确地定位发声源
2. **不确定性估计处理状态转换**是一个优雅的解决方案——不是试图精确预测转换帧，而是承认不确定性并降低置信度
3. 自动构建对比学习的正负样本（基于音频响应度）避免了对GT mask的依赖

## 局限与展望

- 分组策略依赖DPC-KNN聚类，计算开销和组数选择需要调优
- 对极端噪声或静默视频的处理能力有待验证
- 不确定性估计基于Dirichlet分布的假设可能在某些分布下不成立
- 未来可探索无需预设组数的自适应分组策略

## 相关工作与启发

- **与TPAVI/CATR的关系**：这些方法用全局注意力进行音视频对齐，本文用分组竞争+对比学习替代
- **与BAVS的关系**：BAVS用帧级输入避免过分割，本文通过不确定性估计在段级输入下解决同一问题
- **启发**：在多模态对齐任务中，"区分性对齐"（分组+竞争）比"无差别对齐"（全局注意力）更有效

## 评分

⭐⭐⭐⭐

系统性地分析并解决了AVS的两个核心挑战，AMA和UE模块设计合理且互补。在多个benchmark上全面SOTA。技术亮点在于音频驱动的分组竞争机制和不确定性感知预测。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics](dynamic_derivation_and_elimination_audio_visual_segmentation_with_enhanced_audio.md)
- [\[ICCV 2025\] Implicit Counterfactual Learning for Audio-Visual Segmentation](../../ICCV2025/segmentation/implicit_counterfactual_learning_for_audio-visual_segmentation.md)
- [\[CVPR 2025\] SAM2-LOVE: Segment Anything Model 2 in Language-Aided Audio-Visual Scenes](sam2-love_segment_anything_model_2_in_language-aided_audio-visual_scenes.md)
- [\[CVPR 2025\] Revisiting Audio-Visual Segmentation with Vision-Centric Transformer](revisiting_audio-visual_segmentation_with_vision-centric_transformer.md)
- [\[CVPR 2025\] Audio-Visual Instance Segmentation](audio-visual_instance_segmentation.md)

<!-- RELATED:END -->
