---
title: >-
  [论文解读] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos
description: >-
  [CVPR 2026][医学图像][半监督分割] 提出 SMART 框架，基于 SAM3 的概念提示分割构建 Teacher-Student 半监督模型，结合渐进置信度正则化和双流时序一致性策略，仅用极少标注在 X 射线冠脉造影视频中实现 SOTA 血管分割。
tags:
  - CVPR 2026
  - 医学图像
  - 半监督分割
  - 冠脉造影
  - SAM3
  - 时序一致性
  - 光流
---

# Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos

**会议**: CVPR 2026  
**arXiv**: [2603.00881](https://arxiv.org/abs/2603.00881)  
**代码**: [GitHub](https://github.com/qimingfan10/SMART)  
**领域**: 医学图像  
**关键词**: 半监督分割, 冠脉造影, SAM3, 时序一致性, 光流

## 一句话总结

提出 SMART 框架，基于 SAM3 的概念提示分割构建 Teacher-Student 半监督模型，结合渐进置信度正则化和双流时序一致性策略，仅用极少标注在 X 射线冠脉造影视频中实现 SOTA 血管分割。

## 研究背景与动机

冠状动脉疾病 (CAD) 是全球主要死因，X 射线冠脉造影 (XCA) 是临床"金标准"诊断工具。精确的冠脉分割对自动化诊断至关重要，但面临以下挑战：

**标注稀缺**：临床场景中标注数据获取极其昂贵耗时，大量数据无标注

**XCA 特性难点**：边界模糊、辐射对比度不一致、复杂运动模式、低信噪比

**现有 SSL 方法局限**：
   - 依赖几何提示或特征提示的 SAM 方法在跨机构场景下泛化能力差
   - 直接应用 SAM3 到 XCA 序列会忽略时序依赖，导致分割时序不一致
   - 教师模型在低质量区域的预测不可靠（低准确率、高方差）

## 方法详解

### 整体框架

SMART 采用两阶段训练：
1. **文本驱动分割微调**：在标注数据上微调教师 SAM3，利用文本概念提示适配医学领域
2. **运动感知半监督学习**：冻结教师，引导学生模型在无标注数据上学习。推理时仅用学生模型。

### 关键设计

1. **SAM3 概念提示微调 (TPT)**：不依赖几何提示（点/框），而是利用 SAM3 独特的文本概念提示能力。微调 SAM3 的图像编码器、文本编码器和检测器，保持其他组件冻结。损失函数为：
   $$\mathcal{L}_{\text{ft}} = \lambda_1 \mathcal{L}_{\text{Dice}} + \lambda_2 \mathcal{L}_{\text{Bce}}$$
   核心优势：文本概念提示理解视觉结构的语义，跨机构泛化能力远优于几何提示。

2. **渐进置信度感知一致性正则化 (CCR)**：对教师模型注入 $N=8$ 次噪声扰动 $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$ 获取 $N$ 组预测，计算集成均值 $\bar{\mathbf{P}}$ 和不确定性权重 $\boldsymbol{\mathcal{U}}$。一致性损失对不确定区域给予更高权重：
   $$\mathcal{L}_{\text{conf}} = \frac{\sum_{x,y} \mathcal{D}(x,y) \mathcal{U}(x,y)}{\sum_{x,y} \mathcal{U}(x,y) + N\eta} + \frac{\beta}{N} \sum_{x,y} \mathcal{U}(x,y)$$
   其中 $\mathcal{D}(x,y) = (\sigma(S(x,y)) - \sigma(\bar{P}(x,y)))^2$ 为学生与教师集成预测的一致性距离。设计动机：低对比度区域教师预测不可靠，需自适应调节监督强度。

3. **双流时序一致性 (DSTC)**：使用 SEA-RAFT 计算前向、后向光流 $\mathbf{F}_{t \to t+1}$ 和 $\mathbf{F}_{t+1 \to t}$，双向 Mask Warping 确保分割的时序一致性：
   $$\mathcal{L}_{\text{opti}} = \frac{1}{2N} \sum_{x,y} \Big[\big(\mathbf{S}_t - \mathcal{W}(\mathbf{S}_{t+1}, \mathbf{F}_{t \to t+1})\big)^2 + \big(\mathbf{S}_{t+1} - \mathcal{W}(\mathbf{S}_t, \mathbf{F}_{t+1 \to t})\big)^2\Big]$$
   额外引入 Flow Coherence Loss $\mathcal{L}_{\text{coh}}$ 惩罚边界点偏离血管主体运动，区分前景/背景。

### 损失函数 / 训练策略

- 总损失：$\mathcal{L}_{\text{all}} = \lambda_{\text{Dice}} \mathcal{L}_{\text{Dice}} + \lambda_{\text{Bce}} \mathcal{L}_{\text{Bce}} + \lambda_{\text{conf}} \mathcal{L}_{\text{conf}} + \lambda_{\text{opti}} \mathcal{L}_{\text{opti}} + \lambda_{\text{coh}} \mathcal{L}_{\text{coh}}$
- 权重：$\lambda_{\text{Dice}}=0.5, \lambda_{\text{Bce}}=0.5, \lambda_{\text{conf}}=0.5, \lambda_{\text{opti}}=0.3, \lambda_{\text{coh}}=0.2$
- AdamW 优化器，lr=1e-4，weight decay=0.01，batch size=4，6k 迭代
- 教师/学生采用不对称数据增强：教师用强增强（旋转±15°，噪声 σ=0.03），学生用弱增强

## 实验关键数据

### 主实验

在 XCAV（111 视频）和 CAVSA（1061 视频）数据集上，仅用 16 个标注视频：

| 方法 | XCAV DSC ↑ | XCAV clDice ↑ | CAVSA DSC ↑ | CAVSA clDice ↑ |
|------|-----------|--------------|-----------|--------------|
| UNet (监督) | 70.80 | 69.24 | 64.19 | 70.27 |
| Denver | 73.30 | 70.40 | 76.53 | 79.17 |
| CPC-SAM | 77.90 | 79.15 | 77.90 | 78.28 |
| **SMART (Ours)** | **84.39** | **83.01** | **91.00** | **97.73** |

仅用 14% 标注视频，SMART 在 XCAV 上超越次优方法 CPC-SAM 6.49% DSC；在 CAVSA 上仅用 1.5% 标注数据即提升 13.1% DSC。

### 消融实验

| 配置 | XCAV DSC ↑ | CAVSA DSC ↑ | 说明 |
|------|-----------|-----------|------|
| TPT + CCR（无 DSTC） | 82.38 | 78.87 | 缺少时序一致性 |
| TPT + DSTC（无 CCR） | 76.24 | 47.77 | 不可靠伪标签严重影响 |
| CCR + DSTC（无 TPT） | 76.71 | 25.82 | 文本概念提示对 SAM3 适配至关重要 |
| **TPT + CCR + DSTC** | **84.39** | **91.00** | 三组件缺一不可 |

### 关键发现

- **CCR 是核心**：去掉 CCR 后 CAVSA DSC 暴降 43.23%，说明不正则化教师输出对分割影响极大
- **DSTC 提升空间连通性**：clDice 提升约 39%，有效减少断裂/过分割
- **噪声扰动次数 N=8 最佳**：从 N=2 到 N=8，clDice 从 81.82% 提升到 83.01%
- 文本概念提示 vs 点提示：概念提示在跨机构泛化上明显更优，CADICA 数据集上视觉对比显著

## 亮点与洞察

- **文本概念提示的医学适配**：利用 SAM3 的语义理解能力替代几何提示，解决跨机构域差异问题
- 渐进置信度正则化同时做到"加权高不确定区域"和"集成多噪声预测"，双重增强鲁棒性
- 双流光流设计（前向+后向）缓解了单向光流的确认偏差
- 极少标注下的惊人性能：16 个视频标注+每个仅 1-2 帧，就能达到远超监督方法的效果

## 局限与展望

- XCA 视频帧数有限，长序列场景下时序建模能力未知
- SAM3 本身的计算开销较大，实时性可能无法满足术中需求
- 仅在冠脉造影数据上验证，扩展到其他血管造影场景需进一步实验
- 未探索 SAM3 不同规模变体的影响

## 相关工作与启发

- 与 MedSAM2/KnowSAM 等基于几何提示的方法不同，SMART 利用文本语义消除了对特定点/框的依赖
- 置信度正则化思想可推广到其他教师-学生框架中处理不可靠伪标签
- SEA-RAFT 光流 + Mask Warping 的组合在医学视频中证明有效

## 评分

- 新颖性: ⭐⭐⭐⭐ — SAM3 概念提示在医学半监督中的首次成功应用
- 实验充分度: ⭐⭐⭐⭐⭐ — 三个数据集、完整消融、跨机构泛化验证
- 写作质量: ⭐⭐⭐⭐ — 方法描述清晰，各组件动机充分
- 价值: ⭐⭐⭐⭐ — 标注效率极高，临床应用前景好

<!-- RELATED:START -->

## 相关论文

- [Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [Synergistic Bleeding Region and Point Detection in Laparoscopic Surgical Videos](synergistic_bleeding_region_and_point_detection_in_laparoscopic_surgical_videos.md)
- [A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance](better_than_average_spatially-aware_aggregation_of_segmentation_uncertainty_impr.md)

<!-- RELATED:END -->
