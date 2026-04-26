---
title: >-
  [论文解读] M-Net: MRI Brain Tumor Sequential Segmentation Network via Mesh-Cast
description: >-
  [ICCV 2025][医学图像][MRI脑肿瘤分割] M-Net 将 MRI 相邻切片间的空间连续性重新理解为"类时序"数据，提出 Mesh-Cast 机制将任意序列模型（LSTM、Transformer、Mamba SSM 等）无缝集成到通道和时序信息处理中，配合两阶段顺序训练策略（TPS），在 BraTS2019 和 BraTS2023 上取得了 SOTA 分割性能。
tags:
  - ICCV 2025
  - 医学图像
  - MRI脑肿瘤分割
  - 序列建模
  - Mesh-Cast机制
  - 时空相关性
  - 两阶段训练
---

# M-Net: MRI Brain Tumor Sequential Segmentation Network via Mesh-Cast

**会议**: ICCV 2025  
**arXiv**: [2507.20582](https://arxiv.org/abs/2507.20582)  
**代码**: 无  
**领域**: 医学图像  
**关键词**: MRI脑肿瘤分割、序列建模、Mesh-Cast机制、时空相关性、两阶段训练

## 一句话总结

M-Net 将 MRI 相邻切片间的空间连续性重新理解为"类时序"数据，提出 Mesh-Cast 机制将任意序列模型（LSTM、Transformer、Mamba SSM 等）无缝集成到通道和时序信息处理中，配合两阶段顺序训练策略（TPS），在 BraTS2019 和 BraTS2023 上取得了 SOTA 分割性能。

## 研究背景与动机

**领域现状**：脑肿瘤 MRI 分割对疾病诊断和治疗规划至关重要。深度学习方法从 UNet 开始不断演进，包括结合注意力机制的 CANet、MIRAU-Net，基于 KAN 的 UKAN，以及 TransUNet、Swin UNETR 等混合架构。此外 Mamba SSM 相关方法如 Mamba UNet 也在医学图像分割中展现了优异表现。

**现有痛点**：现有方法大多将 MRI 切片独立处理（2D 方法）或直接使用 3D 卷积处理整个体积。2D 方法无法利用相邻切片间的空间连续性，导致分割连续性差；3D 方法虽然能捕获体积信息，但计算代价极高，在实际部署中受限。尽管已有工作将语言模型中的序列模块引入视觉领域，但这些序列模块通常仅处理单张图像内的 patch 序列，并未充分利用 MRI 切片间的"类时序"空间相关性。

**核心矛盾**：MRI 相邻切片间存在明显的空间连续性（病变区域的大小和位置随切片连续变化），这种结构性信息类似于视频帧间的时序关系。然而在 2D 切片处理框架下，这种跨切片的"类时序"信息是不可观测的，导致 2D 方法的性能天花板明显低于 3D 方法。

**本文目标**：设计一种既能利用 MRI 切片间的"类时序"空间相关性，又保持 2D 方法计算效率的分割框架。

**切入角度**：将 MRI 切片序列类比为视频帧序列，将序列建模问题引入 MRI 切片分割。关键创新在于设计了一种通用的"网格传播"（Mesh-Cast）机制，能在时序维度和通道维度上灵活地嵌入任意序列处理算法。

**核心 idea**：通过 Mesh-Cast 机制在时序和通道两个维度上交替传播序列信息，让 2D 切片方法也能捕获 3D 体积上下文信息，同时保持计算效率。

## 方法详解

### 整体框架

M-Net 采用经典的编码器-解码器结构加跳跃连接。输入是一组多模态 MRI 序列 $X = \{x_1, x_2, \ldots, x_T\}$，其中每个切片 $x_t \in \mathbb{R}^{H \times W \times C}$。每层包含 Vision Sequential Module（处理单帧内的空间信息）和 Mesh-Cast Sequential Module（处理跨帧的时序与通道信息）。输出是每个切片的分割掩码。

### 关键设计

1. **Mesh-Cast Sequential Module**:

    - 功能：核心组件，在时序和通道维度上交替进行序列建模，捕获 MRI 切片间的"类时序"空间相关性
    - 核心思路：给定特征序列 $X_{in}$，首先在时序维度上将通道 $C$ 作为 batch 维度，让序列模型感知 $T$ 帧间的时序相关性；然后通过 Mesh-Cast Forward 进行维度交换（$C$ 和 $T$ 互换），使序列模型转而在通道维度上建模，捕获不同模态（T1、T1c、T2、FLAIR）间的特征相关性；最后 Mesh-Cast Backward 将维度还原。当堆叠多层时，使用 SE（Squeeze-and-Excitation）层级注意力机制加权各层输出
    - 设计动机：单纯的时序建模只能捕获空间位置变化，而通道维度上的建模能利用不同 MRI 模态间的互补信息。Mesh-Cast 的维度交换机制让同一个序列模型同时服务于两个维度，且可以灵活替换为 LSTM、Transformer、Mamba SSM 等任意序列算法

2. **两阶段顺序训练策略（TPS, Two-Phase Sequential）**:

    - 功能：增强模型的泛化能力和鲁棒性
    - 核心思路：第一阶段，对输入序列进行帧级 Shuffle，将来自不同序列/位置的切片随机组合成新序列进行训练，让模型学习跨序列的通用特征模式；第二阶段，恢复原始有序序列进行微调，让模型学习真实的时序依赖关系。第一阶段的 Shuffle 提供数据多样性并加速收敛，第二阶段的有序训练提供精细的序列相关性学习
    - 设计动机：直接用有序序列训练容易过拟合于特定序列模式，先 shuffle 再有序的两阶段策略能让模型先学到通用的解剖结构特征，再专注于序列特定的上下文依赖。实验表明 shuffle-then-order 的顺序明显优于 order-then-shuffle

3. **Vision Sequential Module**:

    - 功能：在单帧内提取空间序列特征
    - 核心思路：采用 Cross-Scan 方式将 2D 图像沿四个方向（左→右、右→左、上→下、下→上）序列化，然后分别通过序列模块提取各方向的空间相关性，最后合并四个方向的特征。与 Mesh-Cast Sequential Module 共享相同的序列模块接口
    - 设计动机：确保模型不仅捕获跨切片的时序信息，也充分提取单张切片内的空间特征

### 损失函数 / 训练策略

使用 BCE Loss 和 Dice Loss 的组合损失函数：$L_{joint} = \sum_{i=1}^{3} (\lambda L_{Dice}^i + (1-\lambda) L_{BCE}^i)$。将多类分割任务转化为多通道单类分割，分别计算 WT、TC、ET 三个区域的损失。Dice Loss 专门处理类别不平衡问题，BCE Loss 提供像素级错误计算。

## 实验关键数据

### 主实验

在 BraTS 2019 和 BraTS 2023 上与 12 种 SOTA 方法对比（格式：BraTS2019/BraTS2023）：

| 方法 | 年份 | FLOPs | Dice-WT↑ | Dice-TC↑ | Dice-ET↑ | Haus95-WT↓ |
|------|------|-------|----------|----------|----------|-----------|
| UNet | 2015 | 321G | 87.36/90.71 | 88.59/93.05 | 90.69/93.36 | 1.358/1.186 |
| Swin UNETR | 2022 | 137G | 88.16/91.11 | 88.85/93.20 | 90.86/93.42 | 1.308/1.163 |
| Mamba UNet | 2024 | 72G | 88.21/91.03 | 90.11/93.32 | 90.86/93.31 | 1.306/1.173 |
| nnUNet | 2021 | 82G | 87.81/90.34 | 90.23/92.74 | 90.96/92.37 | 1.297/1.210 |
| **M-Net** | 本文 | **91G** | **88.38/91.33** | **90.52/93.55** | **91.43/93.42** | **1.287/1.153** |

### 消融实验

不同序列模型和训练策略在 BraTS 2019 上的消融：

| 配置 | Dice-WT | Dice-TC | Dice-ET | 说明 |
|------|---------|---------|---------|------|
| Backbone (Slices) | 87.17 | 89.29 | 90.41 | 无序列模块基线 |
| Mamba SSM (Slices) | 88.05 | 90.21 | 90.65 | 仅用切片输入 |
| Mamba SSM (TPS) | **88.38** | **90.52** | **91.43** | 完整 TPS 策略 |
| M-Net (T only) | 87.86 | 89.28 | 90.93 | 仅时序建模 |
| M-Net (T+C, TPS) | **88.38** | **90.52** | **91.43** | 时序+通道完整模型 |
| M-Net (T+C, Ordered) | 88.05 | 90.21 | 90.65 | 仅有序训练 |
| M-Net (T+C, Shuffled) | 88.07 | 90.32 | 91.05 | 仅 Shuffle 训练 |

### 关键发现

- **Mamba SSM 效果最佳**：在所有可选序列模型中，Mamba SSM 以最少的额外 FLOPs（91.29G vs 基线 72.44G）取得了最好的性能
- **通道建模不可或缺**：仅做时序建模（T only）相比同时做时序和通道建模（T+C）在 TC 上低 1.24%，证明通道维度的跨模态信息建模至关重要
- **TPS 策略有效**：shuffle-then-order 两阶段策略比任何单阶段策略都更优，且优于 order-then-shuffle 的逆序策略
- **推理效率优秀**：M-Net 推理仅需 15 分钟，而 nnUNet 需要 97 分钟（仅为其 16%）

## 亮点与洞察

- **类时序建模的新视角**：将 MRI 切片间的空间关系重新定义为"类时序"数据，这个类比非常精准且有启发性。这一视角可以迁移到任何具有空间连续性的序列化数据处理上，如 CT 扫描、超声序列等
- **Mesh-Cast 的维度交换设计**：通过简单的维度转置操作，让同一个序列模块在时序和通道两个维度上交替工作，既优雅又高效。这种设计思路可以推广到其他需要多维度序列建模的场景
- **TPS 训练策略的去偏效果**：先用 Shuffle 数据训练强制模型学习通用模式，再用有序数据精调学习特定依赖，这种"先泛后专"的训练策略思路值得借鉴

## 局限与展望

- 目前仅在脑肿瘤 MRI 分割上验证，未扩展到其他医学影像模态（CT、超声等）
- Mesh-Cast 机制中序列模块需要在时序和通道两个维度各执行一次，当切片数 $T$ 和通道数 $C$ 都较大时计算量仍会增长
- 仅使用 BraTS 2019 和 2023 两个数据集，数据多样性有限
- 对于极端不规则或跨度很大的肿瘤，序列模型能否有效捕获长距离依赖有待验证

## 相关工作与启发

- **vs Swin UNETR**: Swin UNETR 使用 Transformer 架构处理 3D 体积数据，计算量大（137G）。M-Net 以 2D 切片方式处理但通过 Mesh-Cast 捕获 3D 信息，计算更高效且性能更优
- **vs Mamba UNet**: Mamba UNet 将 Mamba SSM 用于单帧内的序列处理。M-Net 将 Mamba SSM 提升到跨帧的时序维度，充分利用了切片间的空间连续性
- **vs nnUNet**: nnUNet 是自配置的强基线，推理时间极长（97 分钟）。M-Net 在所有指标上超越 nnUNet 且推理快 6 倍以上

## 评分

- 新颖性: ⭐⭐⭐⭐ Mesh-Cast 维度交换机制设计巧妙，类时序建模视角新颖
- 实验充分度: ⭐⭐⭐⭐ 两个数据集上与 12 种方法对比，消融覆盖序列模型选择、Mesh-Cast 组件和 TPS 策略
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整，图示详细
- 价值: ⭐⭐⭐⭐ 提供了一种通用的 MRI 序列分割框架，Mesh-Cast 可扩展到其他模态

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation](../../CVPR2026/medical_imaging/pgr-net_prior-guided_roi_reasoning_network_for_brain_tumor_mri_segmentation.md)
- [\[ICCV 2025\] Scaling Tumor Segmentation: Best Lessons from Real and Synthetic Data](scaling_tumor_segmentation_best_lessons_from_real_and_synthetic_data.md)
- [\[ICCV 2025\] MRGen: Segmentation Data Engine for Underrepresented MRI Modalities](mrgen_segmentation_data_engine_for_underrepresented_mri_modalities.md)
- [\[ICCV 2025\] UKBOB: One Billion MRI Labeled Masks for Generalizable 3D Medical Image Segmentation](ukbob_one_billion_mri_labeled_masks_for_generalizable_3d_medical_image_segmentat.md)
- [\[ICCV 2025\] RadGPT: Constructing 3D Image-Text Tumor Datasets](radgpt_constructing_3d_image-text_tumor_datasets.md)

<!-- RELATED:END -->
