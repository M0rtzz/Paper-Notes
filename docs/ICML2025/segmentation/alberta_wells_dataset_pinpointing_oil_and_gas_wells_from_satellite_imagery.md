---
title: >-
  [论文解读] Alberta Wells Dataset: Pinpointing Oil and Gas Wells from Satellite Imagery
description: >-
  [ICML 2025][图像分割][遥感] 提出首个大规模油气井检测基准数据集 Alberta Wells Dataset（213k+ 井位、188k+ 卫星图像 patch），将废弃/暂停/活跃油气井的定位问题建模为二值分割和目标检测任务，并评估了多种 CNN 和 Transformer 基线模型。
tags:
  - ICML 2025
  - 图像分割
  - 遥感
  - 油气井检测
  - 二值分割
  - 目标检测
  - 卫星图像
---

# Alberta Wells Dataset: Pinpointing Oil and Gas Wells from Satellite Imagery

**会议**: ICML 2025  
**arXiv**: [2410.09032](https://arxiv.org/abs/2410.09032)  
**代码**: [GitHub](https://github.com/RolnickLab/Alberta_Wells_Dataset)  
**领域**: 分割  
**关键词**: 遥感, 油气井检测, 二值分割, 目标检测, 卫星图像

## 一句话总结

提出首个大规模油气井检测基准数据集 Alberta Wells Dataset（213k+ 井位、188k+ 卫星图像 patch），将废弃/暂停/活跃油气井的定位问题建模为二值分割和目标检测任务，并评估了多种 CNN 和 Transformer 基线模型。

## 研究背景与动机

全球有数百万口废弃油气井正在向大气泄漏甲烷（强效温室气体）、向地下水渗漏有毒化合物。仅加拿大就有约 37 万口废弃井，年排放等效约 50 万吨 CO₂；美国约有 400 万口，年排放超 500 万吨 CO₂ 当量。这些井可以通过封堵来减缓危害，但大量废弃井位置未知——宾夕法尼亚州估计高达 90% 的废弃井未被记录。

遥感技术结合机器学习为大规模定位废弃井提供了可能，但已有数据集规模极小（500–12,000 井）、覆盖区域有限、且通常仅包含活跃井，无法用于检测废弃或暂停井。本文旨在填补这一空白，提供首个真正大规模、覆盖多种井状态的公开基准数据集。

## 方法详解

### 整体框架

本文的核心贡献是 **Alberta Wells Dataset（AWD）**——一个面向油气井识别的大规模基准数据集及其上的基线评估。整体流程包括三个部分：

1. **数据采集与质量控制**：从 Alberta Energy Regulator (AER) 获取井位元数据，经领域专家审核过滤、去重、分类
2. **卫星图像获取与标注生成**：使用 Planet Labs 4 波段（RGB+近红外）高分辨率卫星图像，为每个 patch 生成分割掩码和 COCO 格式检测标注
3. **基线模型评估**：在二值分割和目标检测两种任务设定下评估多种深度学习模型

### 关键设计

#### 数据采集与质量控制流程

原始 AER ST37 数据包含约 637k 条元数据记录和 512k 条地理坐标记录，但存在大量重复和状态不准确的问题。处理流程：

- **去重**：元数据按许可证号去重（保留最新更新）；shapefile 按许可日期去重
- **融合与过滤**：合并两个数据源，由领域专家制定规则将井分为三类：
    - **活跃（Active）**：107,139 口，状态为 Flowing/Pumping/Gas Lift
    - **暂停（Suspended）**：55,007 口，状态为 Suspension
    - **废弃（Abandoned）**：54,947 口，状态为 Abandoned/Junked and Abandoned
- **坐标去重**：对重复坐标保留最新钻井日期的记录
- **边界校验**：确保所有井位于 Alberta 省界内
- 最终筛选后约 217k 条有效记录

#### Patch 生成

将 Alberta 省划分为不重叠的正方形图像 patch，每个 patch 边长 1050m（面积约 1.1025 km²）。确保含井 patch 和无井 patch 数量大致相等。最终数据集包含 188,688 个 patch，其中 94,344 个含井，覆盖 213,447 口井。

#### 基于聚类的数据集划分算法

为保证训练/验证/测试集的地理多样性，提出两级 K-Means 聚类划分算法（Algorithm 1）：

- **Step 1**：对所有含井 patch 的质心坐标进行 K-Means 聚类（M=300 个簇），形成局部区域簇 $k_1$
- **Step 2**：对 $k_1$ 簇的质心再做 K-Means 聚类（N=30 个超级簇），形成代表城市/大地理区域的 $k_2$ 超级簇
- **Step 3**：在每个超级簇 $k_2$ 中，选择含最少井位的两个 $k_1$ 簇分别分配到验证集和测试集，其余分配到训练集
- **Step 4**：将无井 patch 按凸包半径分配到对应簇中
- **Step 5**：不平衡修正——若非井 patch 过多则按比例剔除，否则从未分配的非井 patch 中采样补充

这种方法确保每个数据划分都包含来自不同地理区域的样本，同时避免数据泄漏。

#### 卫星图像获取与标注

- **图像源**：PlanetScope 4 波段（RGB + 近红外），PSB.SD 仪器，分辨率约 3m/pixel
- **选择 Planet Labs 的理由**：日更新频率保证一致性；多光谱（NIR 有助于检测地面凹陷）；全球覆盖
- **分割标注**：以 90m 直径圆圈标注每口井位（实际 70–120m），生成二值分割掩码和多类分割掩码
- **检测标注**：同尺度定义 bounding box，采用 COCO 格式
- **数据增强**：随机 resize 到 256×256；水平/垂直翻转（各 p=0.25）；通道归一化

### 损失函数 / 训练策略

**分割模型**：
- CNN 基线（U-Net, PAN, DeepLabV3+）：BCELogits 损失，ResNet50 backbone，batch size 128，cosine annealing 学习率，AdamW 优化器，50 epochs
- Transformer 基线（Segformer, UperNet）：Dice 损失，polynomial 学习率衰减；Segformer 用 mit-b0-ade backbone (batch 128)；UperNet 用 ConvNeXt/Swin backbone (batch 64)

**检测模型**：
- RetinaNet/SSD Lite：batch size 512；Faster R-CNN/FCOS：batch size 256
- DETR：batch size 64
- 所有检测模型均用 ResNet50 backbone（SSD Lite 用 MobileNet），cosine annealing，AdamW，120 epochs

## 实验关键数据

### 主实验

**Table 4: 二值分割结果**

| 模型 | Backbone | 参数量 | IoU | F1 | Precision | Recall |
|------|----------|--------|-----|------|-----------|--------|
| U-Net | ResNet50 | 32.52M | 58.0±0.5 | 61.9±0.8 | **90.2±2.2** | 62.3±1.6 |
| U-Net | ResNeXt50 | 32M | 58.2±0.2 | 62.1±0.3 | 88.2±3.5 | 63.6±1.7 |
| U-Net | SE_ResNet50 | 35.06M | 58.9±0.7 | 62.9±0.7 | 88.8±1.6 | 64.4±1.4 |
| U-Net | EfficientNetB6 | 43.83M | **60.4±0.3** | **64.8±0.4** | 87.8±0.4 | 66.3±0.3 |
| PAN | ResNet50 | 24.26M | 57.8±0.8 | 61.5±0.9 | 89.3±1.2 | 61.5±0.9 |
| DeepLabV3+ | ResNet50 | 26.68M | 56.8±0.7 | 60.6±0.7 | 89.4±1.3 | 61.8±1.1 |
| Segformer | mit-b0-ade | 3.72M | 57.6±0.5 | 61.3±0.6 | 82.6±2.9 | 69.2±2.1 |
| UperNet | ConvNeXt-S | 128.29M | 59.4±0.1 | 63.5±0.1 | 81.5±0.5 | 71.5±0.4 |
| UperNet | ConvNeXt-B | 146.27M | 59.7±0.3 | 63.8±0.2 | 81.1±0.7 | 72.2±0.2 |
| UperNet | Swin-S | 81.15M | 59.9±0.7 | 64.2±0.7 | 80.6±0.5 | **73.1±0.1** |

**Table 5: 目标检测结果**

| 模型 | Backbone | 参数量 | IoU@0.1 | IoU@0.3 | IoU@0.5 | mAP@50 | mAP@50:95 |
|------|----------|--------|---------|---------|---------|--------|-----------|
| RetinaNet | ResNet50 | 18.87M | 24.58 | 43.07 | 59.79 | 0.18 | 0.63 |
| Faster R-CNN | ResNet50 | 41.09M | **36.79** | 46.95 | 61.29 | 5.20 | 19.12 |
| FCOS | ResNet50 | 31.85M | 34.79 | 48.51 | 62.66 | 9.67 | 30.46 |
| SSD Lite | MobileNet | 3.71M | 33.91 | **50.30** | **65.07** | 9.76 | 25.14 |
| DETR | ResNet50 | 41.47M | 41.78 | 51.15 | 63.17 | **15.22** | **38.45** |

### 消融实验

**近红外波段对分割的影响（U-Net + ResNet50）**

| 配置 | IoU | F1 | Precision | Recall |
|------|-----|------|-----------|--------|
| RGB+NIR | **58.0±0.5** | **61.9±0.8** | **90.2±2.2** | 62.3±1.6 |
| RGB only | 56.6±0.4 | 60.5±0.4 | 87.0±1.4 | **62.5±0.1** |

**近红外波段对检测的影响（FCOS + ResNet50）**

| 配置 | IoU@0.1 | IoU@0.3 | IoU@0.5 | mAP@50 | mAP@50:95 |
|------|---------|---------|---------|--------|-----------|
| RGB+NIR | **34.79** | **48.51** | **62.66** | **9.67** | **30.46** |
| RGB only | 32.39 | 46.80 | 61.23 | 5.70 | 20.00 |

**训练数据中井类型的影响（U-Net + ResNet50）**

| 指标 | 仅活跃井 | 全部类型 | 说明 |
|------|---------|---------|------|
| IoU | 0.502 | **0.576** | +14.7% |
| F1 | 0.503 | **0.614** | +22.1% |
| Precision | **0.998** | 0.913 | 仅训练活跃井精度极高但召回极低 |
| Recall | 0.502 | **0.614** | 召回提升显著 |

### 关键发现

1. **分割优于检测**：整体而言，分割任务的性能高于检测，说明分割可能是更适合真实场景井位定位的建模方式
2. **U-Net + EfficientNetB6 分割最优**：IoU 60.4%、F1 64.8%，得益于更大的感受野
3. **UperNet + Swin 召回最高**（73.1%）：对于需要最小化漏检的监测场景更合适
4. **DETR 在检测中表现最全面**：mAP@50 达 15.22，mAP@50:95 达 38.45，全局上下文建模能力突出
5. **NIR 波段显著提升性能**：在分割和检测中均带来一致改进，特别是 mAP@50:95 从 20.0 提升到 30.5
6. **多类型井联合训练至关重要**：仅用活跃井训练无法有效检测废弃/暂停井

## 亮点与洞察

- **规模空前的数据集**：213k 口井、94k 含井 patch，比之前最大数据集（12,490 口）大一个数量级以上
- **地理聚类划分算法**设计精巧：两级 K-Means 保证训练/测试集在地理上无泄漏且多样
- **实际应用导向**：废弃井是甲烷排放的重要且不确定性最大的来源，本数据集直接服务于气候变化缓解
- **多光谱的价值**：实验证明近红外波段对检测地面凹陷等井位特征至关重要
- **问题建模的洞察**：将井位识别同时建模为分割和检测，发现分割更适合此任务

## 局限与展望

1. **标注噪声**：依赖 AER 官方记录，可能有未记录的真实井位导致假阴性标注
2. **地理局限性**：仅覆盖 Alberta 省，迁移到其他地区的 zero/few-shot 能力未验证
3. **高密度区域退化**：大多数 patch 仅含 1–5 口井，罕见高密度区域检测性能下降
4. **废弃/暂停井的视觉信号弱**：植被遮挡和设施退化使这类井更难检测
5. **整体性能仍有大幅提升空间**：最佳分割 IoU 仅 60.4%，检测 mAP@50 仅 15.2%
6. **未利用多类分割标注**：数据集提供了活跃/暂停/废弃的多类标注但未做多类分割实验
7. **可尝试方向**：SAM 等通用分割大模型微调；多时相分析；自监督预训练；半监督学习利用未标注区域

## 相关工作与启发

- 遥感 + ML 用于气候变化行动已有广泛探索（土地利用分类、作物分类、森林监测等），本文将其扩展到油气井检测
- 油气基础设施检测的相关数据集（NEPU-OWOD、Well Pad Dataset 等）规模较小且仅含活跃井
- 本文的地理聚类划分方法可推广到其他遥感数据集的构建，值得借鉴
- 启发：对于小目标（井位在图中仅 ~30 像素）的遥感检测，多光谱信息和适当的任务建模（分割 vs 检测）选择至关重要

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 创新性 | ⭐⭐⭐⭐ | 首个大规模油气井检测基准，数据集构建方法和划分算法具有创新性 |
| 技术深度 | ⭐⭐⭐ | 方法以数据集构建为主，模型部分为标准基线评估 |
| 实验完整性 | ⭐⭐⭐⭐ | 多种分割/检测模型、NIR 消融、井类型消融，实验全面 |
| 实用价值 | ⭐⭐⭐⭐⭐ | 直接服务于气候变化缓解的实际需求，数据集公开且附代码 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，包含详尽的 Datasheet 和附录 |
| 综合评分 | ⭐⭐⭐⭐ | 优秀的 benchmark 数据集论文，社会影响力大 |

<!-- RELATED:START -->

## 相关论文

- [Harnessing Massive Satellite Imagery with Efficient Masked Image Modeling](../../ICCV2025/segmentation/harnessing_massive_satellite_imagery_with_efficient_masked_image_modeling.md)
- [Using Multiple Input Modalities Can Improve Data-Efficiency and O.O.D. Generalization for ML with Satellite Imagery](using_multiple_input_modalities_can_improve_data-efficiency_and_ood_generalizati.md)
- [Generalizable Slum Detection from Satellite Imagery with Mixture-of-Experts](../../AAAI2026/segmentation/generalizable_slum_detection_from_satellite_imagery_with_mixture-of-experts.md)
- [Self-supervised Synthetic Pretraining for Inference of Stellar Mass Embedded in Dense Gas](../../NeurIPS2025/segmentation/self-supervised_synthetic_pretraining_for_inference_of_stellar_mass_embedded_in_.md)
- [Plain-Det: A Plain Multi-Dataset Object Detector](../../ECCV2024/segmentation/plain-det_a_plain_multi-dataset_object_detector.md)

<!-- RELATED:END -->
