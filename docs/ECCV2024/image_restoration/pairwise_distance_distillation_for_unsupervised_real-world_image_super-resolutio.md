---
title: >-
  [论文解读] Pairwise Distance Distillation for Unsupervised Real-World Image Super-Resolution
description: >-
  [ECCV 2024][图像恢复][真实世界超分辨率] 提出成对距离蒸馏框架，通过蒸馏专用模型和通用模型之间的内部和模型间距离关系，实现无监督真实世界图像超分辨率的退化自适应。
tags:
  - ECCV 2024
  - 图像恢复
  - 真实世界超分辨率
  - 无监督学习
  - 知识蒸馏
  - 成对距离
  - 退化自适应
---

# Pairwise Distance Distillation for Unsupervised Real-World Image Super-Resolution

**会议**: ECCV 2024  
**arXiv**: [2407.07302](https://arxiv.org/abs/2407.07302)  
**代码**: 有  
**领域**: Image Restoration  
**关键词**: 真实世界超分辨率, 无监督学习, 知识蒸馏, 成对距离, 退化自适应

## 一句话总结

提出成对距离蒸馏框架，通过蒸馏专用模型和通用模型之间的内部和模型间距离关系，实现无监督真实世界图像超分辨率的退化自适应。

## 研究背景与动机

单图超分辨率（SISR）是计算机视觉中的经典问题。标准方法使用已知的下采样核（如bicubic）从高分辨率图像构建低分辨率训练对，然而在真实世界场景中，低分辨率图像的退化过程是未知的，且远比简单的下采样复杂（包含噪声、模糊、压缩伪影等）。这就是**真实世界超分辨率（Real-World SR, RWSR）**问题。

当前处理RWSR的主流方法分为两类：(1) **盲SR方法**：通过复杂的合成退化增强来训练一个适用于各种退化的通用模型（如Real-ESRGAN），但为了泛化牺牲了在特定退化上的性能；(2) **退化估计方法**：先估计退化参数再进行SR，但退化估计本身是困难的。

核心矛盾在于：通用模型追求广泛的退化覆盖但在特定退化上不够好；专用模型在已知退化上表现优秀但无法处理未知退化。本文提出了一个新颖的视角——通过知识蒸馏，让一个针对合成退化训练的专用模型自适应到目标真实世界退化，同时参考一个预训练的通用模型。

## 方法详解

### 整体框架

成对距离蒸馏框架包含三个参与者：(1) 一个在合成退化上训练的**专用模型（Specialized Model）**作为学生模型；(2) 一个预训练的**通用模型（Generalized Model）**作为辅助教师；(3) 学生模型通过蒸馏两种距离关系来适应目标退化——模型内距离（intra-model distance）和模型间距离（inter-model distance）。

### 关键设计

1. **模型内距离蒸馏（Intra-model Distance Distillation）**:
    - 功能：保持特征空间中的相对结构关系
    - 核心思路：对于同一退化的多个低分辨率图像，计算模型输出特征之间的成对距离矩阵。通过蒸馏通用模型的距离矩阵到专用模型，使专用模型学到通用模型的特征组织结构。这比直接蒸馏绝对特征值更灵活，因为允许了特征空间的仿射变换
    - 设计动机：在无配对数据的情况下，无法直接监督SR输出；但可以利用特征间的相对关系作为间接监督信号

2. **模型间距离蒸馏（Inter-model Distance Distillation）**:
    - 功能：对齐专用模型和通用模型的输出关系
    - 核心思路：对于同一输入，分别计算专用模型和通用模型的输出，然后蒸馏两者之间的距离关系。使专用模型在保持其合成退化知识的同时，学习通用模型在真实世界退化上的处理策略
    - 设计动机：通用模型虽然在特定退化上不够精确，但它包含了关于真实退化的有用先验知识

3. **退化自适应训练策略**:
    - 功能：在无标注的真实世界数据上进行自适应
    - 核心思路：使用目标域的无标注低分辨率图像进行自适应训练。专用模型通过蒸馏损失在真实世界数据上微调，逐步适应目标退化。训练过程中冻结通用模型参数，只更新专用模型
    - 设计动机：真实世界的LR图像虽然没有配对的HR标签，但包含了退化模式的信息，可以通过蒸馏间接利用

### 损失函数 / 训练策略

- 模型内距离损失：最小化专用模型和通用模型在同一批次样本上的成对距离矩阵之差
- 模型间距离损失：约束专用模型和通用模型对同一输入的输出距离
- 可选的感知损失和对抗损失用于进一步提升视觉质量
- 自适应训练使用较小学习率，防止过度偏离预训练知识

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| RealSR | PSNR ↑ | SOTA | Real-ESRGAN | +0.5-1.0dB |
| RealSR | LPIPS ↓ | SOTA | BSRGAN | 显著降低 |
| DRealSR | PSNR ↑ | SOTA | 多个基线 | +0.3-0.8dB |
| 真实世界图像 | 视觉质量 | 最优 | 通用模型 | 更清晰自然 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅模型内距离 | 部分提升 | 学到特征结构但缺少跨模型信息 |
| 仅模型间距离 | 部分提升 | 学到模型对齐但特征结构弱 |
| 两者结合 | 最优 | 互补关系明显 |
| 不同通用模型 | 鲁棒 | 对通用模型选择不敏感 |

### 关键发现

- 成对距离蒸馏比绝对特征蒸馏更适合无监督场景
- 模型内和模型间距离提供了互补的监督信号
- 专用模型通过自适应可以在特定退化上超越通用模型
- 方法对不同类型的退化（噪声、模糊、JPEG压缩）均有效

## 亮点与洞察

- 从蒸馏角度解决无监督RWSR是新颖的视角
- 成对距离作为无监督信号的思想具有通用性
- 专用+通用模型的组合利用了两种模型的互补优势
- 方法不需要退化估计或复杂的合成增强

## 局限与展望

- 需要一个预训练的通用模型作为教师，方法效果部分依赖教师质量
- 成对距离蒸馏需要在每个批次内计算距离矩阵，增加了计算开销
- 对于极端或罕见的退化类型，如果通用模型也处理不好，蒸馏效果可能受限
- 可以探索在线更新教师模型的策略

## 相关工作与启发

- **Real-ESRGAN**: 通用RWSR方法的代表，通过复杂合成退化增强训练
- **CycleSR / DASR**: 无监督方法，使用域适应或循环一致性
- **知识蒸馏**: FitNet、CRD等工作提供了蒸馏的理论基础
- 启发：成对距离蒸馏的思想可以推广到其他无监督图像恢复任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 成对距离蒸馏的视角新颖，将蒸馏引入无监督SR
- 实验充分度: ⭐⭐⭐⭐ 多数据集评估和消融实验充分
- 写作质量: ⭐⭐⭐ 论文逻辑清晰，但某些技术细节可以更详细
- 价值: ⭐⭐⭐ 对真实世界SR有实际贡献，但适用场景相对特定

<!-- RELATED:START -->

## 相关论文

- [AdcSR: Adversarial Diffusion Compression for Real-World Image Super-Resolution](../../CVPR2025/image_restoration/adversarial_diffusion_compression_for_real-world_image_super-resolution.md)
- [Towards Real-world Event-guided Low-light Video Enhancement and Deblurring](towards_real-world_event-guided_low-light_video_enhancement_and_deblurring.md)
- [DenoiSplit: A Method for Joint Microscopy Image Splitting and Unsupervised Denoising](denoisplit_a_method_for_joint_microscopy_image_splitting_and_unsupervised_denois.md)
- [Rethinking Image Super-Resolution from Training Data Perspectives](rethinking_image_super-resolution_from_training_data_perspectives.md)
- [Overcoming Distribution Mismatch in Quantizing Image Super-Resolution Networks](overcoming_distribution_mismatch_in_quantizing_image_super-resolution_networks.md)

<!-- RELATED:END -->
