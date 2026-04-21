---
title: >-
  [论文解读] An OpenMind for 3D Medical Vision Self-Supervised Learning
description: >-
  [ICCV 2025][医学图像][3D 医学图像] 发布了最大的公开 3D 医学影像预训练数据集 OpenMind（114k 脑部 MRI），并系统性地对比了多种 3D 自监督学习方法在 CNN（ResEnc-L）和 Transformer（Primus-M）上的表现，证明 MAE 预训练在分割任务上最优、对比学习在分类任务上最优，且首次展示预训练 Transformer 可在部分数据集上超越从头训练的 CNN。
tags:
  - ICCV 2025
  - 医学图像
  - 3D 医学图像
  - 自监督预训练
  - benchmark
  - 脑部 MRI
  - 基础模型
---

# An OpenMind for 3D Medical Vision Self-Supervised Learning

**会议**: ICCV 2025  
**arXiv**: [2412.17041](https://arxiv.org/abs/2412.17041)  
**代码**: [OpenMind](https://github.com/MIC-DKFZ/openmind)  
**领域**: 医学图像 / 自监督学习  
**关键词**: 3D 医学图像, 自监督预训练, benchmark, 脑部 MRI, 基础模型

## 一句话总结

发布了最大的公开 3D 医学影像预训练数据集 OpenMind（114k 脑部 MRI），并系统性地对比了多种 3D 自监督学习方法在 CNN（ResEnc-L）和 Transformer（Primus-M）上的表现，证明 MAE 预训练在分割任务上最优、对比学习在分类任务上最优，且首次展示预训练 Transformer 可在部分数据集上超越从头训练的 CNN。

## 研究背景与动机

3D 医学图像自监督学习（SSL）领域存在严重的碎片化问题，无法确定 SOTA 方法，原因有三：

**缺乏大规模开放数据集**：大型数据集（如 UK-BioBank >10 万、ABCD >4 万）需审批才能使用，且数据使用协议（DUA）限制苛刻（如需在论文标题中包含数据集名称、需内部审稿等）。已有的公开 SSL 方法要么基于受限的大数据集开发，要么仅在小规模公开数据上验证。

**缺乏可比性**：不同方法使用不同的预训练数据、不同的网络架构（CNN/ViT/Swin/混合）、且在不同的下游任务上评估，导致公平比较几乎不可能。

**微调策略的重要性被忽视**：不同的微调策略对预训练模型的最终表现影响巨大，但现有工作缺少系统性的微调策略对比。

**核心矛盾**：如何在统一框架下公平评估 3D 医学图像 SSL 方法的真实水平？

**切入角度**：从数据、基准、代码三个层面解决——提供最大开放数据集 + 统一架构/训练/评估框架 + 开源所有预训练权重。

## 方法详解

### 整体框架

本工作的主要贡献是系统工程而非新算法：

1. **OpenMind 数据集**：114k 3D 头颈部 MRI，23 种模态，CC-BY 许可
2. **OpenMind 基准**：在统一条件下对比 7+ 种 SSL 方法 × 2 种架构 × 15 个下游数据集
3. **开源生态**：预训练/微调代码 + 模型权重

### 关键设计

#### 1. OpenMind 数据集构建

- **数据来源**：OpenNeuro 平台 800 个独立研究
- **原始数据**：71k 3D MRI + 15k 4D 扩散加权成像
- **DWI 预处理**：将 4D DWI 转化为 3 类 3D 衍生图像（MD maps、FA maps、T2 加权），新增 43k 张
- **去面部化处理**：生成匿名化掩码和解剖掩码，避免重建类 SSL 方法被去面部区域"惩罚"
- **元数据标准化**：统一参与者人口统计信息、扫描参数等
- **图像质量评分（IQS）**：由两位评审者独立对 800 个数据集的每种模态打 1-5 分

#### 2. 基准设计

**统一设置**：
- 所有方法在 OpenMind 上预训练 1000 epoch × 250 steps/epoch
- 4×A100 DDP 训练
- 两种架构：ResEnc-L（CNN）和 Primus-M（Transformer）

**下游评估**：
- 4 个开发数据集（用于超参调优）+ 8 个测试分割数据集 + 3 个测试分类数据集
- 所有数据集 50/50 划分训练/测试
- 分割用 nnU-Net 框架微调；分类用独立框架

**SSL 方法**：VoCo、SimCLR、VolumeFusion、ModelsGenesis、MAE、S3D、SimMIM、SwinUNETR 预训练方案

#### 3. 微调策略对比

设计了 5 种微调策略以平衡预训练表征保持与下游适应：
- **Default**：多项式衰减 lr，初始 1e-2/1e-3
- **Frozen**：仅训练解码器
- **Warm-Up**：线性预热后接默认策略
- **Valley**：先训练解码器（线性下降 lr）→ 线性预热 → 默认
- **Sawtooth**：两阶段预热（解码器冻结编码器 → 全网络）→ 默认

### 损失函数 / 训练策略

- 分割下游：NFL 损失（nnU-Net 默认），学习率多项式衰减
- 分类下游：200 epoch 微调
- 预训练：各 SSL 方法保持原始损失设计

## 实验关键数据

### 主实验（分割，DSC%）

| 预训练方法 | 架构 | ID Mean | OOD Mean | 总均值 |
|-----------|------|---------|----------|--------|
| Scratch 1k | ResEnc-L | 64.15 | 89.43 | 70.47 |
| MAE | ResEnc-L | **65.11** | **88.30** | **70.91** |
| S3D | ResEnc-L | 64.46 | 88.06 | 70.36 |
| MG | ResEnc-L | 64.37 | 88.09 | 70.30 |
| Scratch 1k | Primus-M | 60.05 | 87.90 | 67.01 |
| MAE | Primus-M | **64.34** | **88.69** | **70.42** |
| VoCo | Primus-M | 52.00 | 74.43 | 57.61 |

MAE 预训练的 CNN 在仅 150 epoch 微调即超越 1000 epoch 从头训练基线；MAE 预训练的 Transformer 提升约 3.5% DSC，接近 CNN 从头训练水平。

### 消融实验（微调策略对比，CNN）

| 微调策略 | VoCo | SimCLR | VF | MG | MAE | S3D | 平均 |
|---------|------|--------|----|----|-----|-----|------|
| Default | 77.09 | 74.48 | 76.70 | 76.65 | 76.49 | 76.65 | 76.15 |
| Frozen | 58.28 | 61.73 | 34.73 | 59.48 | 57.18 | 59.40 | 55.25 |
| Warm-Up | 75.47 | 76.30 | 76.60 | 77.36 | 77.75 | 76.40 | 76.33 |
| Sawtooth | **75.96** | **76.23** | **77.68** | **76.66** | **77.50** | **76.87** | **76.60** |

Sawtooth 策略总体最优；Frozen 策略大幅下降说明当前 SSL 表征的泛化能力仍然不足。

### 关键发现

1. **重建类方法（MAE）在分割上最优**，对比类方法（VoCo、SimCLR）**在分类上最优**——没有方法同时在两个任务上表现最好
2. **Transformer 从预训练中获益更大**：MAE 预训练的 Primus-M 在 ATL、COS、ACD 数据集上已超越最强预训练 CNN
3. **微调策略至关重要**：不当策略可使预训练增益完全丧失
4. **数据质量过滤有效但有限**：移除最低质量数据有小幅正向效果，但过度过滤（保留 33%）反而降低性能
5. **考虑去面部化区域可提升 MAE 表征质量**：在重建损失中排除匿名化区域后，MAE 和 S3D 性能均有提升

## 亮点与洞察

1. **里程碑式的资源贡献**：114k 3D MRI + CC-BY 许可，对 3D 医学 SSL 社区意义重大
2. **首次证明预训练 Transformer 可接近/超越从头 CNN**：打破了 3D 医学图像中 Transformer 一直弱于 CNN 的定论
3. **全局 vs 局部表征的深刻洞察**：对比学习适合分类（全局），MAE 适合分割（局部），SwinUNETR 混合目标训练反而偏向分类
4. **完整的可复现体系**：预训练检查点 + nnU-Net 集成微调代码

## 局限与展望

- 分类结果可靠性较低，分类框架不如 nnU-Net 成熟
- 仅预训练 1000 epoch，更长训练可能改变方法排名
- 数据中心化研究的初步证据较弱（仅用简单过滤策略）
- 未探索参数高效微调（PEFT）方法
- 数据集仅包含头颈部 MRI，对腹部/胸部等区域属于域外

## 相关工作与启发

- nnU-Net 系列为分割评估提供了可靠框架，使得本工作的分割结论具有较高可信度
- DINOv2 和 MAE 在自然图像领域的成功启发了本工作对 3D 版本的系统评估
- 数据质量过滤的研究（DataComp 等）提示 OpenMind 数据集有潜力支持更复杂的数据中心化方法

## 评分

- **新颖性**: ⭐⭐⭐ — 主要贡献是资源和基准，而非新方法
- **技术深度**: ⭐⭐⭐⭐ — 系统性基准设计严谨，发现有深度
- **实用价值**: ⭐⭐⭐⭐⭐ — 对 3D 医学 SSL 社区极具推动力
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，实验全面透彻

<!-- RELATED:START -->

## 相关论文

- [Vector Contrastive Learning for Pixel-wise Pretraining in Medical Vision](vector_contrastive_learning_for_pixel-wise_pretraining_in_medical_vision.md)
- [Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training](boosting_vision_semantic_density_with_anatomy_normality_mode.md)
- [scSSL-Bench: Benchmarking Self-Supervised Learning for Single-Cell Data](../../ICML2025/medical_imaging/scssl-bench_benchmarking_self-supervised_learning_for_single-cell_data.md)
- [UKBOB: One Billion MRI Labeled Masks for Generalizable 3D Medical Image Segmentation](ukbob_one_billion_mri_labeled_masks_for_generalizable_3d_medical_image_segmentat.md)
- [SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images](seganypet_universal_promptable_segmentation_from_positron_emission_tomography_im.md)

<!-- RELATED:END -->
