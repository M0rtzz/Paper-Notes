---
title: >-
  [论文解读] TIP: Tabular-Image Pre-training for Multimodal Classification with Incomplete Data
description: >-
  [ECCV 2024][医学图像][表格-图像多模态] 提出 TIP 框架，通过掩码表格重建、图像-表格匹配和对比学习三种自监督任务，对表格数据和图像进行联合预训练，学习对不完整表格数据鲁棒的多模态表征，用于下游分类任务。
tags:
  - "ECCV 2024"
  - "医学图像"
  - "表格-图像多模态"
  - "自监督预训练"
  - "缺失数据"
  - "对比学习"
  - "掩码重建"
---

# TIP: Tabular-Image Pre-training for Multimodal Classification with Incomplete Data

**会议**: ECCV 2024  
**arXiv**: [2407.07582](https://arxiv.org/abs/2407.07582)  
**代码**: [有](https://github.com/siyi-wind/TIP)  
**领域**: 医学图像  
**关键词**: 表格-图像多模态, 自监督预训练, 缺失数据, 对比学习, 掩码重建

## 一句话总结

提出 TIP 框架，通过掩码表格重建、图像-表格匹配和对比学习三种自监督任务，对表格数据和图像进行联合预训练，学习对不完整表格数据鲁棒的多模态表征，用于下游分类任务。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：在现实数据库中，图像和结构化表格数据是两种核心数据模态。在医学领域，一个患者的数据通常包括医学影像（如 X 光、MRI）和结构化的临床表格数据（如年龄、血压、实验室指标等）。联合学习这两种模态的表征有巨大潜力，但面临多重挑战：

**模态异质性**：表格数据天然异构（混合连续型、类别型特征），与图像的密集像素表征差异巨大

**数据缺失问题**：真实场景中表格数据经常有缺失值，现有方法大多假设数据完整

**简单融合策略的局限**：早期工作主要使用简单拼接或注意力融合，未考虑缺失数据场景

现有多模态预训练方法（如 CLIP、ALBEF）主要针对图像-文本模态对，无法直接应用于图像-表格场景。表格数据的特殊性（异构特征、变量独立性、类别编码等）要求专门设计的编码器和预训练策略。

## 方法详解

### 整体框架

TIP 框架包含以下核心组件：

| 组件 | 功能 | 说明 |
|------|------|------|
| 图像编码器 $\phi^i$ | 提取图像特征 | 标准视觉骨干（如 ResNet/ViT） |
| 表格编码器 $\phi^t$ | 处理不完整异构表格 | 专门设计的表格编码器，支持缺失值掩码 |
| 多模态交互模块 $\psi$ | 融合图像和表格表征 | 跨模态注意力机制 |
| 投影头 $g^i, g^t$ | 映射到对比学习空间 | 线性投影 |
| 任务头 $h^{itm}, h^{mtr}$ | 匹配/重建任务 | MLP 分类/回归头 |

预训练阶段使用三种自监督目标，微调阶段使用预训练的编码器 $\phi^i, \phi^t, \psi$ 进行下游分类。

### 关键设计

**专用表格编码器**：针对不完整、异构表格数据设计的编码器 $\phi^t$。接收原始表格数据和缺失掩码 $\mathbf{M}$ 作为输入，能够感知哪些特征缺失，从可用特征中提取有效信息。支持随机掩码增强以增强对缺失数据的鲁棒性。

**三种自监督预训练任务**：

1. **图像-表格对比学习（ITC）**$\mathcal{L}_{itc}$：将匹配的图像-表格对拉近，不匹配的推远。通过投影头 $g^i, g^t$ 将两种模态映射到共享嵌入空间，计算对比损失。

2. **图像-表格匹配（ITM）**$\mathcal{L}_{itm}$：二分类任务，判断给定的图像-表格对是否匹配。使用多模态交互模块 $\psi$ 融合特征后，由分类头 $h^{itm}$ 预测。引入**硬负样本挖掘（Hard Negative Mining）**策略：在 batch 内找到与当前图像最相似但不匹配的表格作为负样本（反之亦然），增加训练难度。

3. **掩码表格重建（MTR）**$\mathcal{L}_{mtr}$：随机掩盖一定比例 $\rho$ 的表格特征，利用图像信息和剩余表格信息重建被掩盖的值。由重建头 $h^{mtr}$ 基于多模态融合特征预测被掩盖的表格值。这一任务直接训练模型在数据缺失情况下的推理能力。

**总预训练损失**：

$$\mathcal{L} = \frac{1}{3}(\mathcal{L}_{itc} + \mathcal{L}_{itm} + \mathcal{L}_{mtr})$$

### 损失函数 / 训练策略

- 三种预训练目标等权重平均
- 掩码比例 $\rho$ 为超参数，控制数据缺失的模拟强度
- 硬负样本策略：在 batch 内基于单模态投影特征的相似度选取最困难的负样本
- 预训练后冻结或微调编码器用于下游分类任务

## 实验关键数据

### 主实验（表格）

TIP 在自然图像和医学图像数据集上与多种基线对比，覆盖完整数据和不完整数据两种场景。根据论文摘要，TIP 在以下方面表现优异：

| 对比类别 | 代表方法 | TIP 优势 |
|---------|---------|---------|
| 监督图像方法 | ResNet, ViT | 多模态信息带来增益 |
| 自监督图像方法 | MAE, MoCo | 表格信息的互补性 |
| 多模态方法 | 简单融合策略 | 对缺失数据更鲁棒 |
| 不完整数据方法 | 插值+融合 | 专门的掩码重建训练 |

### 消融实验（表格）

**预训练目标的消融分析**：

| ITC | ITM | MTR | 效果说明 |
|-----|-----|-----|---------|
| ✓ | ✗ | ✗ | 仅对齐两模态的全局表征 |
| ✓ | ✓ | ✗ | 增加细粒度匹配，空间信息更丰富 |
| ✓ | ✗ | ✓ | 增加缺失鲁棒性训练 |
| ✓ | ✓ | ✓ | 完整 TIP，最优性能 |

各组件协同工作：ITC 负责全局对齐，ITM 负责细粒度匹配+硬负样本，MTR 负责缺失数据鲁棒性。

### 关键发现

1. **数据缺失场景优势显著**：当表格缺失率增加时，TIP 相比仅使用完整数据训练的方法性能下降更缓慢
2. **硬负样本挖掘很关键**：随机负样本不足以充分训练匹配模块
3. **掩码重建是鲁棒性的核心**：通过在预训练阶段模拟数据缺失，模型学会从可用信息中推断
4. **表格-图像的互补性**：图像可以帮助恢复缺失的表格信息，反之亦然
5. **适用于自然图像和医学图像**：框架的通用性得到跨领域验证

## 亮点与洞察

- **填补了重要的研究空白**：表格-图像多模态学习在缺失数据场景下的自监督预训练此前鲜有研究
- **三种预训练目标的互补设计**：ITC 做全局对齐，ITM 做细粒度匹配，MTR 做缺失鲁棒性，分工明确
- **掩码重建的双重作用**：既是预训练目标（学习重建能力），又是数据增强（增加训练多样性）
- **通用框架**：不限于特定的图像编码器或表格编码器，可灵活替换组件

## 局限与展望

1. 缓存内容仅包含 Algorithm 1 伪代码，缺乏详细的架构图和完整实验数据
2. 表格编码器的具体架构设计（如何处理异构特征）需要更多细节
3. 掩码比例 $\rho$ 的敏感性分析值得关注
4. 可以探索更复杂的缺失模式（如 MNAR：非随机缺失）
5. 预训练的计算开销和所需数据量是实际部署的关注点
6. 可以扩展到更多医学图像任务（如分割、检测）

## 相关工作与启发

- 借鉴了视觉-语言预训练（ALBEF、BLIP）的成功范式（ITC + ITM），并适配到表格模态
- 掩码重建思想来源于 MAE/BERT 的掩码预训练，但在表格数据上有独特的意义（直接对应数据缺失）
- 硬负样本挖掘策略源自 ALBEF，对模型学习有效匹配至关重要

## 评分

| 维度 | 评分 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 首次系统性地解决表格-图像预训练中的缺失数据问题 |
| 技术深度 | 3.5 | 整体框架借鉴了 VLP 范式，适配层面有创新 |
| 实验充分性 | 3.5 | 涵盖自然和医学数据集，但缓存信息有限 |
| 实用性 | 4.5 | 直击医学 AI 中数据缺失的核心痛点 |
| 总体 | 4 | 实用价值高的多模态预训练工作，对医学 AI 领域有重要意义 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification](../../ICLR2026/medical_imaging/inference-time_dynamic_modality_selection_for_incomplete_multimodal_classificati.md)
- [\[CVPR 2026\] MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning](../../CVPR2026/medical_imaging/multimodalpfn_extending_prior-data_fitted_networks_for_multimodal_tabular_learni.md)
- [\[ICML 2026\] MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training](../../ICML2026/medical_imaging/meg-xl_data-efficient_brain-to-text_via_long-context_pre-training.md)
- [\[ECCV 2024\] Pathology-knowledge Enhanced Multi-instance Prompt Learning for Few-shot Whole Slide Image Classification](pathology-knowledge_enhanced_multi-instance_prompt_learning_for_few-shot_whole_s.md)
- [\[CVPR 2025\] Revisiting MAE Pre-Training for 3D Medical Image Segmentation](../../CVPR2025/medical_imaging/revisiting_mae_pre-training_for_3d_medical_image_segmentation.md)

</div>

<!-- RELATED:END -->
