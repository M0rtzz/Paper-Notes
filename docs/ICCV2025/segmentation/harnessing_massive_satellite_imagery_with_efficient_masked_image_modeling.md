---
title: >-
  [论文解读] Harnessing Massive Satellite Imagery with Efficient Masked Image Modeling
description: >-
  [ICCV 2025][图像分割][遥感基础模型] 提出一个遥感模型预训练流水线，包括 1300 万张光学遥感图像数据集 OpticalRS-13M 和基于语义丰富度选择性编码/重建的高效 MIM 方法 SelectiveMAE，仅用 40% 图像 patch 即可训练出与全量 patch 相当的模型，同时实现 2 倍以上加速。
tags:
  - ICCV 2025
  - 图像分割
  - 遥感基础模型
  - 掩码图像建模
  - 大规模数据集
  - 高效预训练
  - SelectiveMAE
---

# Harnessing Massive Satellite Imagery with Efficient Masked Image Modeling

**会议**: ICCV 2025  
**arXiv**: [2406.11933](https://arxiv.org/abs/2406.11933)  
**代码**: [SelectiveMAE](https://github.com/fengxiangwang/SelectiveMAE)  
**领域**: 遥感图像 / 自监督学习  
**关键词**: 遥感基础模型, 掩码图像建模, 大规模数据集, 高效预训练, SelectiveMAE

## 一句话总结

提出一个遥感模型预训练流水线，包括 1300 万张光学遥感图像数据集 OpticalRS-13M 和基于语义丰富度选择性编码/重建的高效 MIM 方法 SelectiveMAE，仅用 40% 图像 patch 即可训练出与全量 patch 相当的模型，同时实现 2 倍以上加速。

## 研究背景与动机

遥感基础模型（RSFM）的发展依赖于大规模自监督预训练，其中掩码图像建模（MIM）是核心方法。然而，现有遥感领域面临两大瓶颈：

**数据集规模与多样性不足**：现有遥感数据集（如 MillionAID ~100 万张）远小于自然图像数据集（ImageNet-21k ~1400 万），且多以场景级分类为主，缺少目标检测与像素分割的细粒度信息，限制了 MIM 学习泛化表征的能力。

**MIM 训练效率低下**：传统 MAE 需要重建所有被掩码的 patch（通常 75%），而遥感图像的一个显著特点是前景稀疏、背景冗余。对大量无语义信息的背景 patch 进行编码和重建造成了不必要的计算开销。以 ViT-B 在 100 万张遥感图像上预训练为例，需要 107 小时（8×A100），扩展到千万级数据集时代价更加高昂。

**核心矛盾**：如何在遥感场景下既扩大数据规模提升表征质量，又降低 MIM 的计算开销？

**切入角度**：作者从两个问题出发——(1) 是否需要重建所有冗余背景 patch？(2) 能否进一步压缩编码器的可见 patch 比例（如 ≤25% → ≤15%）？基于此提出了选择性编码与重建策略。

## 方法详解

### 整体框架

本文提出的流水线包含两个核心部分：
- **OpticalRS-13M 数据集构建**：收集、筛选、切片、去重形成 1300 万张光学遥感图像
- **SelectiveMAE 高效预训练**：通过 HOG 特征量化 patch 语义丰富度，选择性编码和部分重建

### 关键设计

#### 1. OpticalRS-13M 数据集

基于 DiRS 原则（多样性、丰富性、可扩展性）收集近十年公开遥感数据集，经过以下预处理：
- **排除**：仅保留可见光图像，排除多光谱和 SAR 数据
- **切片**：将高分辨率图像随机裁剪为 64×64 到 1024×1024 的子图
- **去重**：两阶段去重——感知哈希粗筛 + 人工精审

最终数据集包含 12 个主类别（含"事件"类如火灾、洪水等），相较之前数据集至少大 4 倍，且在 t-SNE 可视化中展现出更丰富的特征分布。

#### 2. 部分重建（Partial Reconstruction）

传统 MAE 掩码比为 75%，重建所有掩码 patch。SelectiveMAE 引入重建比 $r$（默认 25%），仅重建语义最丰富的 patch：

- 用 HOG 算法计算每个 patch 的方向梯度直方图特征值
- 按 HOG 值排序，选择 Top-$\lfloor r \times N \rfloor$ 个 patch 进行重建
- 解码器采用 CrossMAE 的轻量交叉注意力设计

**设计动机**：遥感图像中大量背景 patch 的重建对表征学习贡献有限，选择语义丰富的 patch 重建可以在不损失性能的前提下大幅提升吞吐量。

#### 3. 渐进式语义 Token 选择（PSTS）

直接将掩码比提高到 85%（仅 15% 可见 patch）会导致梯度爆炸和训练不稳定。受课程学习启发，PSTS 模块分阶段动态选择编码 patch：

- **初始化**：用 HOG 选择 $s = (1-m)/2$ 比例的高语义 patch 作为种子集 $S^I$
- **阶段 1（近邻）**：选择与种子集余弦距离最小的 patch → 语义相似，容易学
- **阶段 2（互补）**：选择与种子集距离最大的 patch → 语义互补，更具挑战
- **阶段 3（随机）**：随机选择 → 增强鲁棒性

这种"从易到难"的策略有效避免了高掩码比下的训练崩溃。

### 损失函数 / 训练策略

- 损失函数：MSE（与 MAE 一致），仅在选定的重建 patch 上计算
- 学习率按 $m/r$ 比例缩放以匹配 MAE 的损失方差
- 12 层解码器，85% 掩码比，25% 重建比
- 800 epoch 预训练时 warmup 60 epoch

## 实验关键数据

### 主实验

| 模型 | 骨干 | 吞吐量/min | AID 20%/50% | RESISC-45 10%/20% | DIOR mAP50 | LoveDA mIoU |
|------|------|-----------|-------------|-------------------|------------|-------------|
| MAE† | ViT-B | 264k | 96.58/98.02 | 92.44/94.43 | 75.40 | 52.80 |
| SelectiveMAE† | ViT-B | **556k** | 96.90/98.12 | 93.35/94.58 | 75.70 | 53.05 |
| SelectiveMAE | ViT-L | 533k | **97.49/98.52** | **94.73/96.36** | **78.70** | 53.92 |
| OREOLE | ViT-G(914M) | - | 96.71/- | -/- | 77.40 | 54.00 |

†: 在 400 万子集上预训练 800 epoch。SelectiveMAE 在所有主流遥感任务上达到 SOTA，且吞吐量是 MAE 的 2.1 倍。

### 消融实验

| 方法 | 吞吐量/min | AID 20%/50% | RESISC-45 10%/20% |
|------|-----------|-------------|-------------------|
| Adamae (2.36M) | 498k | 88.78/91.25 | 85.72/87.44 |
| Swin-B (88M) | 356k | 93.21/96.48 | 89.94/93.72 |
| HOG (无参数) | **556k** | 93.17/96.12 | 89.21/92.31 |

HOG 作为无参数方法，在速度上远超学习型方法（Swin-B），性能接近但快 56%。

### 关键发现

- **40% patch 足矣**：仅用 15% 编码 + 25% 重建即可训练出与 MAE 相当甚至更优的模型
- **数据集多样性 > 数量**：等量吞吐下，300 万张 × 267 epoch 优于 1300 万张 × 67 epoch，表明 OpticalRS-13M 数据多样性高，需更长训练才能充分利用
- **效率优势随模型增大更显著**：ViT-L 加速比（2.3×）> ViT-B（2.1×），GPU 显存节省 1.6-1.8×
- **渐进策略关键**：far-near-random 顺序会导致梯度爆炸，near-far-random 性能提升约 4%

## 亮点与洞察

1. **问题驱动的方法设计**：从遥感图像"前景稀疏、背景冗余"的固有特征出发，自然地导出了选择性编码和重建策略
2. **仅用 HOG 无参数方法**：避免了引入额外可训练模块的复杂性，简洁高效
3. **课程学习思想的巧妙应用**：PSTS 解决了高掩码比下的训练不稳定问题
4. **端到端流水线**：从数据创建到高效预训练的完整方案，具有良好的工程实用价值

## 局限与展望

- 目前仅支持可见光图像，未涵盖多光谱、SAR 等模态
- HOG 作为手工特征，可能无法捕获高层语义信息
- PSTS 的阶段划分（near → far → random）较为粗粒度，可考虑更平滑的过渡
- 未探讨 SelectiveMAE 在自然图像领域的迁移性

## 相关工作与启发

- CrossMAE 的部分重建思想是直接灵感来源，但作者发现随机选择重建 patch 在遥感中会降低性能，需要基于语义丰富度的选择
- 课程学习（Curriculum Learning）为解决高掩码比训练不稳定提供了方法论支撑
- 对遥感基础模型（如 RVSA、OREOLE）的全面对比展示了流水线的竞争力

## 评分

- **新颖性**: ⭐⭐⭐ — 选择性重建和渐进 token 选择有创新，但整体思路由 CrossMAE 延伸而来
- **技术深度**: ⭐⭐⭐⭐ — 分析清晰系统，实验全面
- **实用价值**: ⭐⭐⭐⭐⭐ — 2倍加速+大规模数据集，对遥感社区有重大实用意义
- **写作质量**: ⭐⭐⭐⭐ — 逻辑清楚，图表精美

<!-- RELATED:START -->

## 相关论文

- [SeiT++: Masked Token Modeling Improves Storage-Efficient Training](../../ECCV2024/segmentation/seit_masked_token_modeling_improves_storage-efficient_training.md)
- [Alberta Wells Dataset: Pinpointing Oil and Gas Wells from Satellite Imagery](../../ICML2025/segmentation/alberta_wells_dataset_pinpointing_oil_and_gas_wells_from_satellite_imagery.md)
- [Generalizable Slum Detection from Satellite Imagery with Mixture-of-Experts](../../AAAI2026/segmentation/generalizable_slum_detection_from_satellite_imagery_with_mixture-of-experts.md)
- [Using Multiple Input Modalities Can Improve Data-Efficiency and O.O.D. Generalization for ML with Satellite Imagery](../../ICML2025/segmentation/using_multiple_input_modalities_can_improve_data-efficiency_and_ood_generalizati.md)
- [Masked Representation Modeling for Domain-Adaptive Segmentation](../../CVPR2026/segmentation/mrm_masked_representation_modeling_domain_adaptive.md)

<!-- RELATED:END -->
