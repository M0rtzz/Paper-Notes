---
title: >-
  [论文解读] Show and Segment: Universal Medical Image Segmentation via In-Context Learning
description: >-
  [CVPR 2025][医学图像][医学图像分割] 提出Iris框架，通过轻量级任务编码模块从参考图像-标签对中提取任务嵌入来指导目标图像分割，无需微调即可适应新任务，在12个数据集上达到或超越任务特定模型性能，在7个未见数据集上展示出优秀的泛化能力。 领域现状：医学图像分割方法分四类——任务特定模型（如nnUNet）效果好…
tags:
  - "CVPR 2025"
  - "医学图像"
  - "医学图像分割"
  - "上下文学习"
  - "通用分割"
  - "任务编码"
  - "少样本分割"
---

# Show and Segment: Universal Medical Image Segmentation via In-Context Learning

**会议**: CVPR 2025  
**arXiv**: [2503.19359](https://arxiv.org/abs/2503.19359)  
**代码**: 无  
**领域**: 医学图像  
**关键词**: 医学图像分割, 上下文学习, 通用分割, 任务编码, 少样本分割

## 一句话总结

提出Iris框架，通过轻量级任务编码模块从参考图像-标签对中提取任务嵌入来指导目标图像分割，无需微调即可适应新任务，在12个数据集上达到或超越任务特定模型性能，在7个未见数据集上展示出优秀的泛化能力。

## 研究背景与动机

**领域现状**：医学图像分割方法分四类——任务特定模型（如nnUNet）效果好但不可扩展；多任务通用模型（如UniSeg）可处理多种任务但遇到未见类别需微调；SAM类交互式模型需要多次人工提示；现有上下文学习方法（UniverSeg、Tyche）性能不如专用模型且计算效率低。

**现有痛点**：(1) 任务特定模型无法处理未见类别；(2) 通用模型仍需微调；(3) SAM需要多次交互尤其3D结构；(4) 现有ICL方法每次推理重新编码上下文，效率低且性能差。

**核心矛盾**：灵活性与性能的矛盾——能适应新任务的方法性能不如专用模型，性能好的方法不能泛化。

**本文目标**：(1) 在训练分布上达到任务特定模型性能；(2) 在未见类别/域外数据上保持强泛化能力；(3) 实现高效的自动化推理。

**切入角度**：将任务编码与推理解耦——先从参考对中提取紧凑的任务嵌入（一次性开销），然后在任意数量的查询图像上复用。

**核心 idea**：设计轻量级任务编码模块，将参考图像-标签对中的前景和上下文信息分别编码为任务嵌入token，通过交叉注意力指导3D查询图像分割，支持多类别一次前向推理。

## 方法详解

### 整体框架

Iris由三部分组成：3D UNet编码器（从头训练）、任务编码模块、掩码解码模块。任务编码模块从参考图像-标签对中提取任务嵌入，解码模块利用任务嵌入通过双向交叉注意力指导查询图像生成分割掩码。

### 关键设计

1. **前景特征编码**:

    - 功能：提取参考标签对应区域的精确特征
    - 核心思路：将编码器特征$\mathbf{F}_s$上采样到原始分辨率，与高分辨率二值掩码$\mathbf{y}_s$逐元素相乘后池化，得到前景嵌入$\mathbf{T}_f \in \mathbb{R}^{1 \times C}$。使用高分辨率掩码而非下采样掩码，因为医学图像中很多结构仅占少量体素
    - 设计动机：下采样masking会丢失细边界和小结构信息，上采样后再masking保证精确的ROI特征提取

2. **上下文特征编码**:

    - 功能：捕获全局上下文信息补充前景编码
    - 核心思路：PixelShuffle将特征展开到高分辨率同时减少通道数，与掩码拼接后经卷积和PixelUnshuffle回到原分辨率。融合特征与$m$个可学习查询token通过交叉/自注意力交互，产生上下文嵌入$\mathbf{T}_c \in \mathbb{R}^{m \times C}$。最终$\mathbf{T} = [\mathbf{T}_f; \mathbf{T}_c]$
    - 设计动机：PixelShuffle实现内存高效的高分辨率特征-掩码融合

3. **掩码解码模块**:

    - 功能：利用任务嵌入指导多类分割
    - 核心思路：多类别任务嵌入拼接为$\mathbf{T} \in \mathbb{R}^{K(m+1) \times C}$，通过双向交叉注意力与查询特征交互，单次前向输出$K$类分割
    - 设计动机：比逐类别推理效率提升$K$倍

### 损失函数 / 训练策略

Dice + 交叉熵组合损失。Episode训练模拟ICL场景。Lamb优化器，lr=$2\times10^{-3}$，80K iterations，batch size 32，体积$128^3$。数据增强包括随机裁剪、仿射变换、强度调整和随机扰动。

## 实验关键数据

### 主实验

| 方法类别 | 方法 | 平均Dice(%) |
|---------|------|------------|
| 任务特定 | nnUNet | 83.18 |
| 多任务通用 | Multi-Talent | 84.47 |
| 交互式 | SAM-Med3D | 68.42 |
| ICL | Tyche-IS | 61.20 |
| **ICL** | **Iris** | **84.52** |

### 消融实验

| 数据集(OOD) | nnUNet-gen | UniSeg | Tyche | Iris |
|------------|-----------|--------|-------|------|
| ACDC | 82.06 | 84.98 | 74.91 | **86.45** |
| SegTHOR | 76.92 | 78.56 | 56.75 | **82.77** |
| MSD Pancreas(未见) | — | — | 11.97 | **28.28** |
| Pelvic(未见) | — | — | 61.92 | **69.03** |

### 关键发现

- Iris首次在分布内达到/超越任务特定和多任务通用模型（84.52% vs 84.47%）
- 3D架构至关重要——现有2D ICL方法大幅落后（61.20% vs 84.52%）
- 对象级上下文检索优于图像级检索
- 任务嵌入可自动揭示跨数据集的解剖学关系

## 亮点与洞察

- **任务编码与推理解耦**是核心设计——任务嵌入提取一次即可复用，比每次重新编码参考高效得多
- **高分辨率前景编码**针对医学图像小结构至关重要
- **多类单次前向**通过任务嵌入拼接实现，效率提升显著

## 局限与展望

- 未见类别分割能力仍不如在该类别上训练的nnUNet
- 参考图像选择对性能有显著影响，最优选择仍是开放问题
- 3D UNet从头训练，未利用预训练视觉基础模型

## 相关工作与启发

- **vs nnUNet**: 分布内几乎持平但域外泛化更强，无需重训
- **vs UniverSeg**: 3D架构+任务解耦+多类单次推理全面超越
- **vs SAM-Med3D**: Iris通过参考对自动定义任务，更适合自动化

## 评分

- 新颖性: ⭐⭐⭐⭐ 任务编码解耦推理设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 19个数据集、四类方法全面对比
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰、图表精美
- 价值: ⭐⭐⭐⭐⭐ 首个在分布内匹配专用模型的ICL方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] I-MedSAM: Implicit Medical Image Segmentation with Segment Anything](../../ECCV2024/medical_imaging/i-medsam_implicit_medical_image_segmentation_with_segment_anything.md)
- [\[CVPR 2025\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[CVPR 2025\] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation?](are_general-purpose_vision_models_all_we_need_for_2d_medical_image_segmentation_.md)
- [\[CVPR 2025\] vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation](vesselfm_a_foundation_model_for_universal_3d_blood_vessel_segmentation.md)
- [\[CVPR 2025\] Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline](interactive_medical_image_segmentation_a_benchmark_dataset_and_baseline.md)

</div>

<!-- RELATED:END -->
