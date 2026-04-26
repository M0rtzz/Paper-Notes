---
title: >-
  [论文解读] WildSAT: Learning Satellite Image Representations from Wildlife Observations
description: >-
  [ICCV 2025][遥感][遥感表征学习] 提出 WildSAT，利用公民科学平台上的数百万地理标记野生动物观测数据，通过对比学习将卫星图像、物种位置和文本描述对齐，显著提升遥感图像表征质量，并支持零样本文本检索。
tags:
  - ICCV 2025
  - 遥感
  - 遥感表征学习
  - 对比学习
  - 野生动物观测
  - 跨模态
  - 卫星图像
---

# WildSAT: Learning Satellite Image Representations from Wildlife Observations

**会议**: ICCV 2025  
**arXiv**: [2412.14428](https://arxiv.org/abs/2412.14428)  
**代码**: https://github.com/cvl-umass/wildsat (有)  
**领域**: Remote Sensing / Representation Learning  
**关键词**: 遥感表征学习, 对比学习, 野生动物观测, 跨模态, 卫星图像

## 一句话总结

提出 WildSAT，利用公民科学平台上的数百万地理标记野生动物观测数据，通过对比学习将卫星图像、物种位置和文本描述对齐，显著提升遥感图像表征质量，并支持零样本文本检索。

## 研究背景与动机

遥感图像表征学习面临的核心问题是监督信号的获取。现有方案包括：
- **自监督学习**（SeCo、Prithvi）：利用时空不变性或掩码自编码器，但缺乏语义监督
- **有监督学习**（SatlasPretrain）：大规模多任务标签，但标注代价高昂
- **跨模态学习**（GRAFT、TaxaBind、RemoteCLIP）：对齐地面图像或文本，但主要关注人造特征（道路、建筑）

本文的关键洞察是：**物种分布编码了丰富的生态与环境信息**。例如山羊出现在崎岖山区，仙人掌鹪鹩栖息在沙漠仙人掌中——物种的栖息地偏好直接反映了当地自然环境特征。这种信息来自 iNaturalist 等平台的数亿观测数据，是免费且全球分布的。然而，利用野生动物观测来提升遥感表征的潜力此前几乎未被探索。

## 方法详解

### 整体框架

WildSAT 采用多模态对比学习框架，联合训练以下三种信号：
1. **卫星图像**：不同时间的同一地点 Sentinel-2 图像提供时序增强
2. **物种位置**：通过 SINR 模型将经纬度编码为位置向量，包含环境协变量（气候数据）
3. **文本描述**：物种对应 Wikipedia 页面的栖息地、习性等文本，通过 GritLM 编码

图像编码器 $f_\theta$ 可以是任意架构（ResNet50、ViT-B/16 等），输出三个线性投影头分别对应图像、文本和位置模态的嵌入。

### 关键设计

**三路对比学习**：
- $\mathcal{L}_{img}$：同一地点不同时间的卫星图像互为正样本（加几何增强）
- $\mathcal{L}_{txt}$：卫星图像嵌入与 Wikipedia 文本嵌入对齐
- $\mathcal{L}_{loc}$：卫星图像嵌入与 SINR 位置嵌入对齐

所有损失基于 InfoNCE，总目标为三者之和。

**参数高效微调策略**：
- 域外预训练模型（如 ImageNet）：ResNet50 用 Scale and Shift Fine-tuning（仅调 BatchNorm），ViT 用 DoRa（仅调 Attention）
- 随机初始化或同域预训练模型：全参数微调
- 这保证了不丢失原有领域知识

**数据构建**：
- iNaturalist 数据集提供 3550 万观测、47375 种物种
- 对应 Sentinel-2 卫星图像（10m/pixel，512×512）
- Wikipedia 文本提供 127484 个段落、37889 种物种
- 总计 980376 个训练样本

### 损失函数 / 训练策略

$$\min_\theta [\mathcal{L}_{img} + \mathcal{L}_{txt} + \mathcal{L}_{loc}]$$

每个对比损失均采用标准 InfoNCE：

$$\mathcal{L}_{con}(\mathbf{z}_i, \mathbf{e}_{1,...,n}) = -\log \frac{\exp(\mathbf{z}_i \cdot \mathbf{e}_i / \tau)}{\sum_j \exp(\mathbf{z}_i \cdot \mathbf{e}_j / \tau)}$$

训练时每个 image-location 对随机采样一个文本段落。

## 实验关键数据

### 主实验

在 7 个下游分类数据集 + 2 个分割数据集上评估 Linear Probing 性能（20 个基线模型）：

| 数据集 | Base 平均 | +WildSAT 平均 | 提升 |
|--------|----------|--------------|------|
| AID | 72.7 | 79.4 | +6.7 |
| EuroSAT | 88.9 | 94.3 | +5.4 |
| RESISC45 | 77.8 | 83.5 | +5.7 |
| So2Sat20k | 37.9 | 48.2 | +10.3 |
| UCM | 81.8 | 87.9 | +6.1 |
| BEN20k | 45.7 | 53.4 | +7.7 |

WildSAT 在 115 个设置中的 108 个取得提升，平均提升 4.3%-10.4%。

与 CLIP-based 方法对比（ViT-B/16）：

| 方法 | 平均分类性能 |
|------|------------|
| TaxaBind | 59.8% |
| GRAFT | 65.0% |
| RemoteCLIP | 71.0% |
| CLIP | 71.6% |
| **WildSAT** | **76.6%** |

### 消融实验

各模态贡献消融（Random ResNet50 → ImageNet ResNet50）：

| loc | env | text | img-a | Random R50 | ImageNet R50 | Random ViT | ImageNet ViT |
|:---:|:---:|:----:|:-----:|-----------|-------------|-----------|-------------|
| | | | | 24.3% | 93.2% | 25.2% | 84.4% |
| ✓ | | | | 44.2% | 95.0% | 41.6% | — |

- 仅位置信号即可为随机模型带来 +20% 的巨大提升
- 完整四模态组合效果最佳

分割任务结果：

| 模型 | Cashew1k IoU | SAcrop3k IoU |
|------|-------------|-------------|
| Random | 40.1% → 72.6% | 18.0% → 20.3% |
| SatlasNet | 55.2% → 71.0% | 19.4% → 20.5% |

### 关键发现

1. **卫星预训练模型受益最大**：SeCo、SatlasNet 等可提升高达 10%，因为 WildSAT 补充了栖息地相关信息
2. **ViT 比 CNN 收益更大**：Transformer 的灵活注意力机制更易适应多模态融合
3. **WildSAT 减少了栖息地相关类别的误报**：通过混淆矩阵分析，So2Sat20k 上所有类别的真正率提高，主要来自减少 "Scattered trees"、"Dense trees" 等栖息地类别的假阳性
4. 支持零样本文本检索（如输入 "desert" 或 "ibex" 可检索对应地貌卫星图）

## 亮点与洞察

1. **独特的监督信号**：野生动物观测数据是免费、全球分布、自然产生的生态标签，与人造特征形成互补
2. **通用框架**：WildSAT 可以作为 continual pre-training 增强现有模型（SatlasNet、SeCo、Prithvi 均受益）
3. **零样本能力**：通过文本对齐实现地理位置的语义检索，这是先前遥感表征方法无法做到的
4. 与人造特征（WikiSatNet）的互补性——自然环境信息和人造结构信息共同构建更完整的地表理解

## 局限与展望

1. 物种观测数据存在地理偏差（欧美地区密度高，非洲、亚洲稀疏），可能影响全球泛化
2. 仅使用 RGB 三通道，Sentinel-2 的多光谱优势未充分利用（附录有初步多光谱实验）
3. Wikipedia 文本质量参差不齐，部分物种描述可能不准确或缺失
4. 线性探测评估可能低估了表征的全部能力，完整微调结果未提供

## 相关工作与启发

- **SatlasPretrain**：大规模有监督遥感预训练，WildSAT 证明可进一步改进其表征
- **GRAFT**：地面图像-卫星图像对齐，但主要关注人造特征
- **TaxaBind**：首个使用物种位置+卫星图像的多模态方法，但聚焦生态任务而非遥感
- 启发：公民科学数据（eBird、iNaturalist）是被低估的监督信号来源，可推广到更多 Earth observation 任务

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4.5 |
| 技术深度 | 3.5 |
| 实验充分性 | 5 |
| 写作质量 | 4.5 |
| 实用价值 | 4.5 |
| 总评 | 4.5 |

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Learning Representations of Satellite Images From Metadata Supervision](../../ECCV2024/remote_sensing/learning_representations_of_satellite_images_from_metadata_supervision.md)
- [\[ICCV 2025\] AstroLoc: Robust Space to Ground Image Localizer](astroloc_robust_space_to_ground_image_localizer.md)
- [\[CVPR 2026\] Are Pretrained Image Matchers Good Enough for SAR-Optical Satellite Registration?](../../CVPR2026/remote_sensing/pretrained_image_matchers_for_sar_optical_satellite_registration.md)
- [\[ECCV 2024\] Weakly-Supervised Camera Localization by Ground-to-Satellite Image Registration](../../ECCV2024/remote_sensing/weakly-supervised_camera_localization_by_ground-to-satellite_image_registration.md)
- [\[ICCV 2025\] Pan-Crafter: Learning Modality-Consistent Alignment for Pan-Sharpening](pan-crafter_learning_modality-consistent_alignment_for_pan-sharpening.md)

<!-- RELATED:END -->
