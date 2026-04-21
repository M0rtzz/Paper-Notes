---
title: >-
  [论文解读] Divide, Conquer and Unite: Hierarchical Style-Recalibrated Prototype Alignment for Federated Medical Segmentation
description: >-
  [AAAI 2026][医学图像][联邦学习] 针对联邦医学图像分割中的"层间风格偏差累积"和"上下文表征不完整"两大挑战，提出FedBCS框架：通过频域自适应风格重校准（FSR）构建领域不变原型，并设计上下文感知的双层原型对齐（CDPA）融合编解码器多层级语义，在组织核分割和前列腺MRI分割任务上达到SOTA。
tags:
  - AAAI 2026
  - 医学图像
  - 联邦学习
  - 医学图像分割
  - 原型对齐
  - 频域风格校准
  - 特征异构
---

# Divide, Conquer and Unite: Hierarchical Style-Recalibrated Prototype Alignment for Federated Medical Segmentation

**会议**: AAAI 2026  
**arXiv**: [2511.10945](https://arxiv.org/abs/2511.10945)  
**代码**: https://github.com/zxy1234321/FedBCS (有)  
**领域**: Medical Imaging / 联邦学习  
**关键词**: 联邦学习, 医学图像分割, 原型对齐, 频域风格校准, 特征异构

## 一句话总结

针对联邦医学图像分割中的"层间风格偏差累积"和"上下文表征不完整"两大挑战，提出FedBCS框架：通过频域自适应风格重校准（FSR）构建领域不变原型，并设计上下文感知的双层原型对齐（CDPA）融合编解码器多层级语义，在组织核分割和前列腺MRI分割任务上达到SOTA。

## 研究背景与动机

联邦学习（FL）允许多个医疗机构在不共享数据的前提下协作训练全局模型，是医学影像分析的重要范式。然而，不同机构的扫描设备和成像协议差异导致严重的特征异构（feature heterogeneity），即相同标签分布下特征分布P(x|y)在客户端间差异显著。

现有基于原型（prototype）的联邦学习方法试图通过类均值特征向量进行跨客户端alignment，但存在两个关键缺陷：

**层间风格偏差累积**（Layerwise Style Bias Accumulation）：不同医疗协议引入的风格差异在网络中间层逐层累积。现有方法仅在输入层做风格归一化或对齐最终层特征，忽略了中间层的风格偏差，导致同一解剖结构在不同机构间的特征表示越来越不同

**上下文表征不完整**（Incomplete Contextual Representation Learning）：现有方法仅使用最后一层编码器特征构建原型，浅层编码的局部纹理/边缘细节和深层的抽象结构信息被忽略，导致语义理解不完整

核心idea：将内容-风格解耦引入原型构建过程（频域操作），并通过编解码器多层级原型融合实现更细粒度的跨客户端语义对齐。

## 方法详解

### 整体框架

FedBCS基于标准的联邦学习pipeline（FedAvg框架），客户端使用共享UNet架构，每轮本地训练后上传原型到服务器。核心创新在本地原型构建和对齐阶段：先通过FSR模块消除特征的领域风格偏差，再通过CDPA从编码器和解码器多层提取和融合原型。服务器端通过FINCH聚类聚合多客户端原型，再分发回指导本地训练。

### 关键设计

1. **频域风格重校准模块（FSR）**:

    - 功能：在原型构建之前，从特征中去除领域特定的风格变化，保留语义内容
    - 核心思路：对编码器特征做2D傅里叶变换，将其分解为编码风格信息的幅度谱和保留语义内容的相位谱。引入可学习参数自适应加权归一化幅度和原始幅度：$\hat{z}_{enc} = IFT(\lambda_s^{norm}\chi_{norm} + \lambda_s^{org}\chi, \gamma)$。权重通过全局平均池化 + 可学习线性映射 + sigmoid生成
    - 设计动机：风格信息自然编码在频率域的幅度谱中，而语义内容在相位谱中。直接对幅度做实例归一化虽然能去风格，但可能丢失有用信息。可学习参数让模型自适应决定保留多少原始风格

2. **上下文感知双层原型对齐（CDPA）**:

    - 功能：从编码器和解码器的多个层级提取领域不变原型，并进行层间融合和跨客户端对齐
    - 核心思路：对每个层级k，分别从编码器和解码器提取类别原型。将相邻层的原型拼接（浅层+深层），通过轻量级1×1卷积融合模块压缩，得到紧凑的多尺度原型。服务器端用FINCH聚类对来自不同客户端的同类原型进行分组，并计算均值原型
    - 设计动机：浅层编码局部组织纹理和边缘，深层编码全局解剖结构。知道器官整体形状（深层信息）有助于在模糊区域更好地确定组织边界（浅层信息）

3. **对比+一致性联合损失**:

    - 功能：使本地特征与全局原型对齐
    - 核心思路：对比损失使特征靠近同类原型、远离异类原型；一致性损失使编码器和解码器的特征分别靠近对应的全局均值原型
    - 设计动机：对比损失捕捉类间判别性，一致性损失确保跨机构的特征一致性

### 损失函数 / 训练策略

总损失 $L_{total} = L_{MP} + L_{dice}$，其中 $L_{MP} = L_{contra} + L_{consis}$。$L_{contra}$ 是InfoNCE风格的对比损失，$L_{consis}$ 是均方误差一致性损失。

训练400轮通信，每轮1个本地epoch。组织核分割用SGD（lr=0.01），前列腺MRI分割用Adam（lr=1e-4），weight decay=1e-4，batch size=6。

## 实验关键数据

### 主实验

| 方法 | 组织核分割AVG(Dice) | △ | 前列腺MRI AVG | △ |
|------|-------------------|-----|-------------|-----|
| FedAvg | 69.50 | - | 78.80 | - |
| FedProx | 69.00 | -0.50 | 78.10 | -0.70 |
| HarmoFL | 71.90 | +2.40 | 81.20 | +2.40 |
| FPL | 71.66 | +2.16 | 77.40 | -1.40 |
| FedPLVM | 71.00 | +1.50 | 77.50 | -1.30 |
| FedUV | 70.00 | +0.50 | 78.50 | -0.30 |
| **FedBCS** | **74.10** | **+4.60** | **82.60** | **+3.80** |

### 消融实验

| 配置 | 组织核 AVG | 前列腺 AVG | 说明 |
|------|-----------|-----------|------|
| Baseline (FedAvg) | 69.50 | 78.80 | 无FSR无CDPA |
| +CDPA only | 72.20 | 81.90 | 仅加多层对齐 |
| +FSR only | 72.30 | 80.10 | 仅加频域风格校准 |
| +CDPA+FSR (Full) | **74.10** | **82.60** | 两个模块互补 |

### 关键发现

- FSR和CDPA两个模块各自独立带来~2-3%提升，联合使用提升更大（4.6%/3.8%），说明二者解决的是互补的问题
- 通信效率高：FedBCS每个客户端每轮仅上传4个原型（2类×2层），而FedPLVM平均上传212-365个原型，但FedBCS性能大幅领先（+5.1%）
- FSR处理层间风格偏差优于仅在输入层做幅度归一化（图4b），验证了频域重校准在特征层面操作的必要性
- 温度参数τ在不同数据集上最优值不同（组织核0.005 vs 前列腺MRI 0.4），但在较宽范围内都超过第二名

## 亮点与洞察

- 频域风格解耦引入联邦原型学习是巧妙的组合：利用傅里叶变换的幅度-相位天然对应风格-内容的性质
- 提供了理论收敛保证（Theorem 1和2），在联邦学习论文中增加可信度
- 通信效率的优势值得关注：多层原型融合后仍只上传2个原型/类，比FedPLVM少50-90倍

## 局限与展望

- 仅在UNet上验证，对更复杂架构（ViT-based）的适用性未知
- 二分类分割任务（前景/背景），多器官多类别场景的扩展性待验证
- 可学习的风格参数可能在客户端数量很少时过拟合
- 理论分析假设梯度方差有界等条件较强，实际中可能不完全成立

## 相关工作与启发

- 与HarmoFL（输入级频域操作）相比，FSR在特征级操作更灵活
- CDPA的编解码器双路径思路可推广到其他需要多层级特征对齐的场景
- FINCH无参聚类避免了预设原型数量的超参数，是联邦原型学习中的实用选择

## 评分
- 新颖性: ⭐⭐⭐⭐ （频域风格校准+多层原型对齐的组合较新）
- 实验充分度: ⭐⭐⭐⭐⭐ （消融、通信效率、收敛分析、超参数分析全面）
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐ （对联邦医学分割中的特征异构问题有实质帮助）

<!-- RELATED:START -->

## 相关论文

- [Federated CLIP for Resource-Efficient Heterogeneous Medical Image Classification](federated_clip_for_resource-efficient_heterogeneous_medical_image_classification.md)
- [HiFusion: Hierarchical Intra-Spot Alignment and Regional Context Fusion for Spatial Gene Expression Prediction from Histopathology](hifusion_hierarchical_intra-spot_alignment_and_regional_context_fusion_for_spati.md)
- [Alleviating Textual Reliance in Medical Language-guided Segmentation via Prototype-driven Semantic Approximation](../../ICCV2025/medical_imaging/alleviating_textual_reliance_in_medical_language-guided_segmentation_via_prototy.md)
- [Bidirectional Channel-selective Semantic Interaction for Semi-Supervised Medical Segmentation](bidirectional_channel-selective_semantic_interaction_for_semi-supervised_medical.md)
- [FunKAN: Functional Kolmogorov-Arnold Network for Medical Image Enhancement and Segmentation](funkan_functional_kolmogorov-arnold_network_for_medical_image_enhancement_and_se.md)

<!-- RELATED:END -->
