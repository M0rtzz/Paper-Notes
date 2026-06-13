---
title: >-
  [论文解读] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study
description: >-
  [CVPR 2026][医学图像][图像分割] 通过在三个异构医学数据集上对 11 种架构进行标准化对比实验，证明了通用视觉模型 (GP-VMs) 在 2D 医学图像分割中可以超越大多数专用医学分割架构 (SMAs)，且 XAI 分析表明 GP-VMs 无需特定领域设计也能捕获临床相关结构。 医学图像分割 (MIS) 是计算…
tags:
  - "CVPR 2026"
  - "医学图像"
  - "图像分割"
  - "general-purpose vision models"
  - "empirical study"
  - "benchmarking"
  - "Grad-CAM"
---

# Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study

**会议**: CVPR 2026  
**arXiv**: [2603.13044](https://arxiv.org/abs/2603.13044)  
**代码**: [GitHub](https://github.com/)  
**领域**: 医学图像分割  
**关键词**: medical image segmentation, general-purpose vision models, empirical study, benchmarking, Grad-CAM

## 一句话总结

通过在三个异构医学数据集上对 11 种架构进行标准化对比实验，证明了通用视觉模型 (GP-VMs) 在 2D 医学图像分割中可以超越大多数专用医学分割架构 (SMAs)，且 XAI 分析表明 GP-VMs 无需特定领域设计也能捕获临床相关结构。

## 研究背景与动机

医学图像分割 (MIS) 是计算机辅助诊断和临床决策支持系统的基础组件。过去十年中，大量针对医学影像特定挑战（低对比度、小解剖结构、有限标注数据）的专用架构不断涌现，如 HiFormer、MISSFormer、U-KAN、Swin-UMamba 等。与此同时，通用视觉模型 (GP-VMs) 在自然图像上取得了显著进展。

然而，现有研究缺乏 **标准化的受控实验** 来公平比较两类模型：不同论文使用不同的数据集、预处理流水线、增强策略和评估流程，导致性能差异可能来自实验设计而非架构本身的优势。

本文的核心问题是：**医学分割任务是否真的需要专用架构，还是通用视觉模型就能达到甚至超越其性能？**

## 方法详解

### 整体框架

这篇论文不提新架构，而是搭了一个标准化基准测试框架来回答一个问题：2D 医学图像分割到底需不需要专用架构。做法是在统一的预处理、训练和评估协议下，横跨三个成像模态各异的医学数据集，对 11 种架构（5 个专用医学分割架构 SMA + 6 个通用视觉模型 GP-VM）做一次公平的 head-to-head 对比，再用 Grad-CAM 看看两类模型关注的区域是否临床相关。

### 关键设计

**1. 模型选择：让两类架构在可比规模上同台竞技**

要公平回答"专用还是通用"，前提是被比的模型在规模和可见度上对等。专用医学分割架构（SMA）这一侧选了 U-Net（31M, CNN）、HiFormer-B（26M, CNN-ViT 混合）、MISSFormer（42M, Transformer）、Swin-UMamba（60M, 状态空间混合）、U-KAN-L（25M, CNN+KAN）；通用视觉模型（GP-VM）这一侧选了 SegFormer-B3、SegNeXt-L、VWFormer（MiT-B3 与 ConvNeXt-S 两种 backbone）、InternImage-T+UPerHead、TransNeXt-Tiny+UPerHead（均在 47–58M 量级）。挑选标准是架构多样性、计算规模可比、学术可见度和代码可用性，避免拿过时或失配的模型凑数。

**2. 异构数据集：覆盖差异最大的成像场景**

为了让结论不被单一模态绑架，三个数据集刻意拉开差异：ISIC'18 是 RGB 皮肤镜、二分类病变分割、3565 张；BKAI-IGH NeoPolyp Small 是 RGB 内窥镜、三分类息肉分割、945 张；CAMUS 是灰度超声心动图、四分类心脏区域分割、1996 张。模态（RGB/灰度）、任务（二/三/四分类）、数据量都不同，能检验结论的跨场景稳健性。

**3. 标准化训练协议：把"实验设计"这个混淆因子摁住**

不同论文的性能差异往往来自预处理和训练设置而非架构本身，所以这里强制所有模型走同一套协议：ImageNet 预训练编码器、$512 \times 512$ 输入、AdamW + REX 调度器、batch size 8，学习率在 $\{10^{-4}, 5 \times 10^{-5}, 10^{-5}\}$ 内搜索，5 折交叉验证、150 epoch、统一早停。数据增强按数据集定制但对所有模型一致。这样一来，最终性能差异才能干净地归因到架构本身。

### 损失函数 / 训练策略

- ISIC'18 使用二元交叉熵损失
- 多分类任务 (NeoPolyp, CAMUS) 使用交叉熵损失
- 混合精度训练 (除 SegNeXt 因不稳定外)
- 评估指标：mIoU、mDSC、mRec、mPrec（不含背景类，全局微平均）
- 使用 Grad-CAM 可视化进行可解释性分析

## 实验关键数据

### 主实验

| 模型 | 类型 | NeoPolyp mDSC | CAMUS mDSC | ISIC'18 mDSC | 平均 mDSC |
|------|------|:---:|:---:|:---:|:---:|
| VW-MiT | GP-VM | 89.7 | 91.4 | 91.8 | **91.0** |
| VW-Conv | GP-VM | 89.6 | 91.3 | 91.8 | **90.9** |
| TransNeXt | GP-VM | 89.4 | 91.7 | 91.7 | **90.9** |
| InternImage | GP-VM | 89.6 | 91.2 | 91.5 | **90.8** |
| SegFormer | GP-VM | 89.1 | 91.4 | 91.7 | **90.7** |
| SegNeXt | GP-VM | 89.2 | 91.3 | 91.5 | **90.7** |
| SU-Mamba | SMA | 88.9 | 91.3 | 91.3 | 90.5 |
| HiFormer | SMA | 84.6 | 90.8 | 91.0 | 88.8 |
| U-KAN | SMA | 82.5 | 90.5 | 90.6 | 87.9 |
| MISSFormer | SMA | 82.9 | 90.4 | 90.3 | 87.9 |
| U-Net | SMA | 83.3 | 89.1 | 89.3 | 87.2 |

### 消融实验 / 关键对比

| 指标维度 | GP-VMs 表现 | SMAs 表现 | 差距分析 |
|----------|:---:|:---:|------|
| NeoPolyp 非肿瘤息肉(C1) | 62.4-66.1% | 34.9-59.2% | GP-VMs 领先最大（7+ pp） |
| CAMUS 左心室壁(C2) | 87.7-88.4% | 85.6-88.3% | 差距较小（≈1-2 pp） |
| ISIC'18 整体 | 91.5-91.8% | 89.3-91.3% | GP-VMs 稳定优势 |
| Grad-CAM 可解释性 | 捕获临床相关区域 | 类似表现 | GP-VMs 无需特定设计也有良好关注 |

### 关键发现

- **GP-VMs 全面领先**：按平均 mDSC 排名，前 6 名全部是 GP-VMs
- **性能差异与数据集相关**：NeoPolyp 上差距最大（GP-VM vs SMA 可达 7+ pp），CAMUS 上差距较小（≈1-2 pp）
- **SU-Mamba 是最强 SMA**：在 SMA 中稳定占优，但仍略低于最佳 GP-VMs
- **GP-VMs 的 XAI 表现良好**：Grad-CAM 分析显示 GP-VMs 能在没有领域特定设计的情况下捕获临床相关结构

## 亮点与洞察

- **实验设计的严谨性**：在相同预处理、增强、优化设置下对比，消除了实验设计混淆因素，这在 MIS 领域相当稀缺
- **实用启示**：在引入新的专用架构之前，应先系统评估现有 GP-VMs 的性能
- **资源节约视角**：当 GP-VMs 已经具备竞争力时，研究精力可转向数据整理、训练协议优化和 OOD 泛化评估

## 局限与展望

- 仅覆盖三个 2D 数据集，无法完全代表临床影像的多样性（如 3D、CT、MRI 等）
- 部分模型参数量不完全可比（如 U-KAN 无 ImageNet 预训练）
- 未评估 OOD 泛化能力（如在 Kvasir-SEG 上测试 NeoPolyp 训练模型）
- 未涉及 3D 医学图像分割和半监督/少样本场景

## 相关工作与启发

- **SAM 在医学影像中的应用**：SAM 已被适配到医学场景，但本文指出即使不用 SAM 级别的基础模型，标准 GP-VMs + fine-tuning 就很有效
- **与 Moglia et al. (2026) 综述对比**：该综述也发现通用模型表现优异，但依赖了不同论文的原始结果，本文通过标准化协议提供了更可靠的证据
- 对医学影像社区的启发：资源有限时优先考虑成熟的 GP-VMs（如 InternImage、VWFormer），而非从零开始设计专用架构

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 总体评价 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation?](../../CVPR2025/medical_imaging/are_general-purpose_vision_models_all_we_need_for_2d_medical_image_segmentation_.md)
- [\[CVPR 2026\] SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation](sd_fsmis_adapting_stable_diffusion_for_few_shot_medical_image_segmentation.md)
- [\[CVPR 2026\] Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)
- [\[CVPR 2026\] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)
- [\[CVPR 2026\] T-Gated Adapter: A Lightweight Temporal Adapter for Vision-Language Medical Segmentation](t-gated_adapter_a_lightweight_temporal_adapter_for_vision-language_medical_segme.md)

</div>

<!-- RELATED:END -->
