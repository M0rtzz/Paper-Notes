---
title: >-
  [论文解读] Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation
description: >-
  [CVPR 2025][医学图像][医学图像合成] Siamese-Diffusion 提出双组件扩散模型（Image-Diffusion + Mask-Diffusion）生成合成医学图像-掩码对用于数据增强，训练时通过噪声一致性损失（参数空间约束）让 Mask-Diffusion 学习图像扩散的形态先验，推理时仅用 Mask-Diffusion 保持多样性，在 Polyps 数据集上 mDice 提升 3.6%。
tags:
  - CVPR 2025
  - 医学图像
  - 医学图像合成
  - 扩散数据增强
  - 噪声一致性
  - 分割
  - 形态保真
---

# Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation

**会议**: CVPR 2025  
**arXiv**: [2505.06068](https://arxiv.org/abs/2505.06068)  
**代码**: https://github.com/Qiukunpeng/Siamese-Diffusion  
**领域**: 医学图像  
**关键词**: 医学图像合成、扩散数据增强、噪声一致性、分割、形态保真

## 一句话总结

Siamese-Diffusion 提出双组件扩散模型（Image-Diffusion + Mask-Diffusion）生成合成医学图像-掩码对用于数据增强，训练时通过噪声一致性损失（参数空间约束）让 Mask-Diffusion 学习图像扩散的形态先验，推理时仅用 Mask-Diffusion 保持多样性，在 Polyps 数据集上 mDice 提升 3.6%。

## 研究背景与动机

1. **领域现状**：医学图像标注成本极高，数据增强是缓解标注不足的关键手段。最近扩散模型被用于合成医学图像，但生成的图像-掩码配对常出现形态不匹配。
2. **现有痛点**：仅从掩码条件化生成图像时，合成图像的解剖结构可能与掩码不一致——特别是在小目标和复杂边界场景下。
3. **核心矛盾**：强条件化（如直接用GT掩码生成）保证了一致性但牺牲了多样性；弱条件化保持多样性但形态不匹配。
4. **本文目标**：在保持多样性的同时提升形态保真度。
5. **切入角度**：用Image-Diffusion 作为"教师"在训练时指导 Mask-Diffusion 学习形态约束，推理时教师不参与。
6. **核心 idea**：训练时双组件+噪声一致性损失；推理时单组件保持多样性。

## 方法详解

### 整体框架

训练：图像→Image-Diffusion 学习图像分布 + 掩码→Mask-Diffusion 学习掩码分布 → 噪声一致性损失约束两者在参数空间对齐 + Dense Hint Input 提供先验。推理：仅 Mask-Diffusion 生成图像-掩码对。

### 关键设计

1. **噪声一致性损失**：在参数空间（而非输出空间）约束 Mask-Diffusion 的噪声预测与 Image-Diffusion 一致
2. **Dense Hint Input (DHI)**：替代标准 Hint Input，提供更密集的条件先验
3. **Online-Augmentation**：单步采样模块实现在线数据增强

### 损失函数 / 训练策略

扩散去噪损失 + 噪声一致性损失（参数空间 L2）。结构和纹理分阶段学习。

## 实验关键数据

### 主实验

| 数据集 | 模型 | 基线 mDice | +Siamese-Diffusion | 提升 |
|--------|------|-----------|-------------------|------|
| Polyps | SANet | 57.8% | **61.4%** | +3.6% |
| ISIC2018 | UNet | 80.2% | **81.7%** | +1.5% |

### 关键发现

- 噪声一致性损失在参数空间约束比输出空间更稳定
- 推理时不使用 Image-Diffusion 保证了生成多样性不受损

## 亮点与洞察

- **训练/推理解耦设计**：训练时用教师约束，推理时去掉教师——巧妙平衡保真度和多样性
- **在线增强的实用性**：单步采样使得增强可以嵌入训练循环

## 局限与展望

- 仅在标准解剖结构上验证，精细结构（手指、面部表情）效果未知
- 提升幅度相对温和（1.5-3.6%），临床验证有待开展

## 相关工作与启发

- **vs ControlNet**: 直接条件化生成缺乏形态约束。Siamese-Diffusion 通过噪声一致性弥补
- **vs 传统增强**: 旋转/翻转等几何增强多样性有限，扩散增强提供语义级多样化

## 评分

- 新颖性: ⭐⭐⭐⭐ 双组件+噪声一致性的设计有新意
- 实验充分度: ⭐⭐⭐ 两个数据集偏少
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 医学图像增强的实用方案

<!-- RELATED:START -->

## 相关论文

- [Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis](latent_drifting_in_diffusion_models_for_counterfactual_medical_image_synthesis.md)
- [BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline](interactive_medical_image_segmentation_a_benchmark_dataset_and_baseline.md)
- [Show and Segment: Universal Medical Image Segmentation via In-Context Learning](show_and_segment_universal_medical_image_segmentation_via_in-context_learning.md)
- [Revisiting MAE Pre-Training for 3D Medical Image Segmentation](revisiting_mae_pre-training_for_3d_medical_image_segmentation.md)

<!-- RELATED:END -->
