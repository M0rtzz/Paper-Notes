---
title: >-
  [论文解读] CustAny: Customizing Anything from A Single Example
description: >-
  [CVPR 2025][图像生成][零样本定制化] 提出 CustAny，一种零样本通用物体定制化框架，通过 DINOv2+MAE 双编码器提取全面身份特征、全局+局部双层级注入、以及对比学习身份解耦，从单张参考图生成保持身份的多样化图像，在通用/人脸/试穿三个领域全面超越专用方法。
tags:
  - CVPR 2025
  - 图像生成
  - 零样本定制化
  - 通用身份保持
  - DINOv2+MAE特征
  - 双层级注入
  - 身份解耦
---

# CustAny: Customizing Anything from A Single Example

**会议**: CVPR 2025  
**arXiv**: [2406.11643](https://arxiv.org/abs/2406.11643)  
**代码**: https://github.com/LingjieKong-fdu/CustAny  
**领域**: 扩散模型 / 图像生成  
**关键词**: 零样本定制化, 通用身份保持, DINOv2+MAE特征, 双层级注入, 身份解耦

## 一句话总结
提出 CustAny，一种零样本通用物体定制化框架，通过 DINOv2+MAE 双编码器提取全面身份特征、全局+局部双层级注入、以及对比学习身份解耦，从单张参考图生成保持身份的多样化图像，在通用/人脸/试穿三个领域全面超越专用方法。

## 研究背景与动机

**领域现状**：图像定制化（customization）从单张参考图生成保持身份的新图像。现有方法要么是领域特定的（PhotoMaker 专注人脸，MagicClothing 专注服装），要么是通用但身份保真度不足的（IP-Adapter）。

**现有痛点**：(1) IP-Adapter 仅用 CLIP 图像特征，丢失细粒度身份信息；(2) DINOv2 特征细节丰富但对颜色不敏感（对比学习训练的弊端）；(3) 现有方法在注入身份特征时抑制了文本可编辑性——身份和非身份信息（姿态、方向等）耦合导致生成多样性差。

**核心矛盾**：需要同时保持身份细节（颜色、纹理、形状）和保留文本编辑能力（改变姿态、背景等），但身份特征往往与非身份元素纠缠。

**本文目标** 用单张参考图实现跨所有物体类别的高保真度身份定制化，同时保持文本编辑能力和生成多样性。

**切入角度**：构建 MC-IDC 数据集（315K 图像，10K+ 类别）提供大规模身份多样性训练；DINOv2（细节）+ MAE（颜色）双编码器互补；全局+局部双层级注入；对比学习身份解耦。

**核心 idea**：DINOv2+MAE 双编码器互补提取完整身份特征，全局/局部双层级注入 + 对比学习解耦身份与非身份信息，实现通用零样本定制化。

## 方法详解

### 关键设计

1. **DINOv2+MAE 双编码器**：DINOv2 提供细节丰富的语义特征但对颜色不敏感（对比学习中颜色增广导致的不变性），MAE 通过重建训练保留了颜色信息。拼接两者 patch token 互补。

2. **双层级 ID 注入**：全局级——将 class token 融合到文本嵌入中。局部级——在 UNet upblock 中添加额外交叉注意力模块注入 patch 级特征，提供细粒度空间身份信息。

3. **ID-aware 解耦**：对比损失分离身份信息和非身份信息（姿态、运动、方向）。同一身份不同姿态的特征应该身份部分相似、非身份部分不同。这保护了文本对非身份属性的编辑能力。

## 实验关键数据

### 主实验

| 领域 | 方法 | FID↓ | CLIP-i↑ | DINO-i↑ |
|------|------|------|---------|---------|
| 通用 | IP-Adapter | 70.32 | 77.18 | 44.94 |
| 通用 | **CustAny** | **47.09** | **82.16** | **65.13** |
| 人脸 | PhotoMaker | 106.35 | 71.80 | 44.62 |
| 人脸 | **CustAny** | **86.40** | **79.60** | **57.44** |
| 试穿 | MagicClothing | 126.09 | 76.53 | 29.10 |
| 试穿 | **CustAny** | **50.65** | **83.82** | **66.24** |

在所有三个领域全面超越通用和专用方法。DINO-i 提升 20+ 点。

### 关键发现
- DINOv2+MAE 互补在颜色保真度上显著优于单一编码器
- ID 解耦使文本编辑能力保持——可以改变姿态/背景同时保持身份
- 跨领域泛化能力强——一个模型处理通用/人脸/试穿三个领域

## 局限与展望
- 基于 SD1.5，分辨率上限受限
- MC-IDC 数据集中部分类别用数据增强替代多视角数据

## 评分
- 新颖性: ⭐⭐⭐⭐ 双编码器互补和 ID 解耦设计有创新
- 实验充分度: ⭐⭐⭐⭐⭐ 三个领域+多种基线+消融，非常全面
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐⭐ 通用定制化能覆盖多个下游任务，实用价值高

<!-- RELATED:START -->

## 相关论文

- [DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness](dexgrasp_anything_towards_universal_robotic_dexterous_grasping_with_physics_awar.md)
- [ScribbleLight: Single Image Indoor Relighting with Scribbles](scribblelight_single_image_indoor_relighting_with_scribbles.md)
- [Orient Anything V2: Unifying Orientation and Rotation Understanding](../../NeurIPS2025/image_generation/orient_anything_v2_unifying_orientation_and_rotation_understanding.md)
- [Exploring Semantic-constrained Adversarial Example with Instruction Uncertainty Reduction](../../NeurIPS2025/image_generation/exploring_semantic-constrained_adversarial_example_with_instruction_uncertainty_.md)
- [ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models](ice_intrinsic_concept_extraction_from_a_single_image_via_diffusion_models.md)

<!-- RELATED:END -->
