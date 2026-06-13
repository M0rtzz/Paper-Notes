---
title: >-
  [论文解读] SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images
description: >-
  [ICCV 2025][医学图像][PET分割] 本文构建了迄今最大的PET分割数据集PETS-5k（5731例3D全身PET图像，超130万张2D切片），并提出SegAnyPET——首个针对PET影像的3D可提示分割基础模型，通过跨提示置信学习（CPCL）策略处理标注质量不一致问题，在已见和未见目标上均大幅超越现有基础模型和任务专用模型。
tags:
  - "ICCV 2025"
  - "医学图像"
  - "PET分割"
  - "基础模型"
  - "3D分割"
  - "噪声标注学习"
  - "SAM适配"
---

# SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images

**会议**: ICCV 2025  
**arXiv**: [2502.14351](https://arxiv.org/abs/2502.14351)  
**代码**: [https://github.com/](https://github.com/) (项目页面将公开)  
**领域**: 医学图像  
**关键词**: PET分割, 基础模型, 3D分割, 噪声标注学习, SAM适配

## 一句话总结
本文构建了迄今最大的PET分割数据集PETS-5k（5731例3D全身PET图像，超130万张2D切片），并提出SegAnyPET——首个针对PET影像的3D可提示分割基础模型，通过跨提示置信学习（CPCL）策略处理标注质量不一致问题，在已见和未见目标上均大幅超越现有基础模型和任务专用模型。

## 研究背景与动机

**领域现状**：正电子发射断层扫描（PET）是一种重要的分子影像模态，通过注射放射性示踪剂（如18F-FDG）揭示体内代谢过程，广泛应用于肿瘤检测、治疗反应评估和疗效监测。近年来以SAM为代表的分割基础模型在自然图像上展现了强大的通用分割能力。

**现有痛点**：PET图像与CT/MRI等结构影像截然不同——分辨率低、部分容积效应严重、对比度差、边界模糊。直接将SAM应用于PET图像效果极差。现有医学SAM适配工作（如MedSAM、SAMed等）几乎全部聚焦于CT和MRI，完全忽略了PET这一分子影像模态。

**核心矛盾**：（a）缺乏大规模标注的PET分割数据集，现有数据集规模小且目标有限；（b）PET图像质量低导致标注质量参差不齐，高质量标注和低质量噪声标注混杂；（c）PET的3D体积信息对分割至关重要，但SAM是2D架构。

**本文目标** （a）构建大规模PET分割数据集；（b）设计适配PET特性的3D基础模型架构；（c）在标注质量不一致的条件下实现鲁棒训练。

**切入角度**：作者观察到PET图像的特殊性（低对比度+弱边界+3D体积结构）需要模态特异性的基础模型设计，而非简单微调通用模型。同时，通过置信学习判别噪声标注并自校正，可以同时利用高质量和低质量数据。

**核心 idea**：构建最大PET数据集 + 3D SAM架构 + 跨提示置信学习，打造首个PET影像通用可提示分割基础模型。

## 方法详解

### 整体框架
SegAnyPET的pipeline：输入为3D PET体积图像和位置提示（点或框），经过3D图像编码器提取特征，结合提示编码器的提示信息，通过掩码解码器输出3D分割掩码。训练时采用CPCL策略，将数据分为高质量标注集和低质量噪声标注集，用两个不同提示策略的模型交叉监督，再通过不确定性引导自校正噪声标签。

### 关键设计

1. **PETS-5k大规模数据集构建**:

    - 功能：收集整理迄今最大的PET分割数据集
    - 核心思路：汇集多个公开PET数据集并进行统一标准化处理，包含5731例3D全身PET图像，覆盖多种器官和病变目标，涵盖超过130万张2D切片
    - 设计动机：之前的PET数据集规模太小（如AutoPET仅500+例），且只覆盖有限的分割目标，不足以训练基础模型

2. **3D SAM架构重构**:

    - 功能：将SAM的2D架构改造为3D，充分利用PET体积数据的层间上下文信息
    - 核心思路：图像编码器从2D卷积/注意力改为3D版本，使模型能够捕获相邻切片之间的空间关系。提示编码器也相应支持3D点/框提示
    - 设计动机：PET影像本质是3D体积数据，相邻切片之间的上下文对于准确分割极为重要

3. **跨提示置信学习(CPCL)**:

    - 功能：在高质量标注和低质量噪声标注混合的数据上实现鲁棒训练
    - 核心思路：维护两个模型副本，分别使用不同的提示策略（如点提示 vs 框提示），互相交叉监督。通过比较两个模型预测的一致性来评估样本标注的可信度。对于噪声标注样本，当两个模型的预测高度一致但与给定标注不一致时，标记该标注为噪声
    - 设计动机：PET图像边界模糊导致不同标注者给出的标注差异大，CPCL利用"交叉验证"思想让两个具有不同归纳偏置的模型互相校验

4. **不确定性引导自校正**:

    - 功能：对识别出的噪声标注进行自动修正
    - 核心思路：利用模型预测的不确定性来加权自校正过程。高不确定性区域的标注保持原样或降低权重，低不确定性区域采用模型预测替代原始噪声标注
    - 设计动机：简单丢弃噪声标注样本浪费了大量训练数据，自校正能充分利用这些数据的有用信息

### 损失函数 / 训练策略
- 高质量数据使用标准的Dice Loss + BCE Loss监督
- 低质量数据的损失函数结合CPCL的置信度权重，降低噪声标注样本的损失贡献
- 训练分为两阶段：先在高质量数据上预热，再引入低质量数据联合训练

## 实验关键数据

### 主实验

| 方法 | 已见目标Dice | 未见目标Dice | 提示类型 |
|------|-------------|-------------|---------|
| SAM (原始) | ~60% | ~45% | 1-点 |
| MedSAM | ~72% | ~55% | 1-点 |
| SAM-Med3D | ~75% | ~58% | 1-点 |
| **SegAnyPET** | **~85%** | **~78%** | **1-点** |

### 消融实验

| 配置 | 已见目标Dice | 未见目标Dice | 说明 |
|------|------------|------------|------|
| Full model (CPCL + 自校正) | ~85% | ~78% | 完整模型 |
| w/o CPCL | ~80% | ~70% | 去掉置信学习后噪声标注影响显著 |
| w/o 3D架构 (用2D) | ~78% | ~68% | 2D版本缺少层间信息 |
| w/o 不确定性自校正 | ~82% | ~74% | 自校正对噪声数据利用很关键 |
| 仅高质量数据训练 | ~83% | ~72% | 数据量减少影响泛化 |

### 关键发现
- CPCL策略贡献最大，去掉后在未见目标上掉约8个点，说明噪声鲁棒训练对PET基础模型至关重要
- 3D架构相比2D带来约7%的提升，验证了层间上下文对PET分割的关键性
- 即使只用1个提示点，SegAnyPET也能达到甚至超越全监督任务专用模型的精度
- 在未见过的分割目标上泛化性能强，证明了基础模型的通用性

## 亮点与洞察
- **首个PET基础模型**：填补了分子影像模态在分割基础模型领域的空白。这一开创性工作为PET下游应用奠定基础
- **CPCL噪声鲁棒学习**：跨提示置信学习策略不依赖网络修改，可以直接迁移到其他可提示基础模型。用不同提示策略的两个模型互校验的思路适用于任何标注质量不一的场景
- **大规模数据集贡献**：PETS-5k作为最大PET分割数据集，对整个PET分析社区都有重要价值

## 局限与展望
- 论文主要关注FDG-PET，对其他示踪剂（如PSMA-PET、FLT-PET）的泛化性有待验证
- CPCL需要维护两个模型副本，训练成本翻倍
- 3D架构的计算开销较大，实际部署需考虑效率优化
- 未探索PET-CT多模态融合分割，PET+CT联合输入可能进一步提升性能
- 自校正依赖模型质量，模型初期不够好时可能引入错误的伪标签

## 相关工作与启发
- **vs MedSAM**: MedSAM在CT/MRI上微调SAM，但未考虑PET特性，且是2D的。SegAnyPET专为PET设计3D架构
- **vs SAM-Med3D**: 虽然也是3D，但没有处理标注噪声问题，且主要在CT数据上训练
- **vs SAMed**: SAMed用LoRA微调SAM，但仍是2D架构且缺乏大规模PET数据

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个PET基础模型+CPCL策略，开创性强但整体技术方案较为标准
- 实验充分度: ⭐⭐⭐⭐ 大规模数据集验证了多种设置，消融充分
- 写作质量: ⭐⭐⭐⭐ 动机清晰，结构完整
- 价值: ⭐⭐⭐⭐⭐ 数据集+模型对PET分析社区价值巨大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](../../CVPR2025/medical_imaging/developing_foundation_models_for_universal_segmentation_from_3d_whole-body_posit.md)
- [\[ICCV 2025\] Coordinate-based Speed of Sound Recovery for Aberration-Corrected Photoacoustic Computed Tomography](coordinate-based_speed_of_sound_recovery_for_aberration-corrected_photoacoustic_.md)
- [\[CVPR 2025\] vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation](../../CVPR2025/medical_imaging/vesselfm_a_foundation_model_for_universal_3d_blood_vessel_segmentation.md)
- [\[ICCV 2025\] Controllable Latent Space Augmentation for Digital Pathology](controllable_latent_space_augmentation_for_digital_pathology.md)
- [\[ICCV 2025\] AcZeroTS: Active Learning for Zero-shot Tissue Segmentation in Pathology Images](aczerots_active_learning_for_zeroshot_tissue_segmentation_in.md)

</div>

<!-- RELATED:END -->
