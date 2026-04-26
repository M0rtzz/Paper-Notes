---
title: >-
  [论文解读] OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images
description: >-
  [ICCV 2025][目标检测][开放提示检测] 提出OpenRSD通用遥感开放提示目标检测框架，支持文本和图像多模态提示，集成对齐头和融合头平衡速度与精度，配合三阶段训练流水线和47万张图像的ORSD+数据集，在7个公开数据集上取得最优平均性能，同时保持20.8 FPS实时推理。
tags:
  - ICCV 2025
  - 目标检测
  - 开放提示检测
  - 遥感目标检测
  - 旋转框检测
  - 多模态提示
  - 自训练
---

# OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images

**会议**: ICCV 2025  
**arXiv**: [2503.06146](https://arxiv.org/abs/2503.06146)  
**代码**: 无（论文称将发布）  
**领域**: 目标检测 / 遥感  
**关键词**: 开放提示检测, 遥感目标检测, 旋转框检测, 多模态提示, 自训练

## 一句话总结

提出OpenRSD通用遥感开放提示目标检测框架，支持文本和图像多模态提示，集成对齐头和融合头平衡速度与精度，配合三阶段训练流水线和47万张图像的ORSD+数据集，在7个公开数据集上取得最优平均性能，同时保持20.8 FPS实时推理。

## 研究背景与动机

遥感目标检测面临三大挑战：

**封闭集检测的局限**：大多数研究聚焦于固定类别集的闭集检测，模型对新类别和新场景泛化性差

**现有开放词汇方法不足**：已有遥感OVD方法（DescReg、CastDet、OVA-DETR）受限于小规模数据集，无法应对遥感特有的旋转目标检测、小目标检测等挑战

**速度与精度矛盾**：深度跨模态融合方法（如Grounding DINO）精度高但推理慢，对大规模遥感图像分析不实用

OpenRSD的目标是构建一个支持多模态提示、兼顾旋转/水平框检测、平衡精度与速度的通用遥感检测基础模型。

## 方法详解

### 整体框架

OpenRSD基于RTMDet改进，包含三个核心组件：
- **多尺度图像特征编码**：RTMDet-L骨干 + Neck提取多尺度特征
- **提示构建模块**：使用SkyCLIP（文本）和DINOv2（图像）提取提示嵌入
- **多任务检测头**：对齐头（alignment head）和融合头（fusion head）并行工作

### 关键设计

1. **多模态提示构建 (Prompt Construction)**

   支持文本和图像两种提示输入：
   - **文本提示**：使用SkyCLIP文本编码器提取嵌入，用GPT-4为每个类别生成10-15个多样化文本提示
   - **图像提示**：将GT框扩大1.25倍裁剪，用DINOv2提取视觉嵌入；通过DINOv2+MLP分类器筛选置信度最高的100张图作为提示集

   提示嵌入通过独立MLP映射到256维空间，训练时每次随机选择一种提示类型，每类随机采样3-7个嵌入。随机采样负类别以增强鲁棒性。

2. **对齐头 (Alignment Head)**

   适合大词汇量快速检测。计算检测特征与提示嵌入的相似度进行分类：

   $$s_i^c = \max_{j \in \{j|l_j=c\}} \left(\alpha \frac{z_i \cdot p_j}{\|p_j\|} + \beta\right)$$

   额外引入监督对比损失稳定特征对齐：

   $$\mathcal{L}_{\text{ct}} = \sum_i -\frac{1}{|P(i)|} \sum_{j \in P(i)} \log \frac{e^{(z_i^* \cdot p_j / \tau)}}{\sum_{k \in A(i)} e^{(z_i^* \cdot p_j / \tau)}}$$

   总损失 $\mathcal{L}_{\text{aln}} = \mathcal{L}_{\text{ct}} + \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{box}}$，同时支持水平框和旋转框回归。

3. **融合头 (Fusion Head)**

   引入轻量跨模态融合增强检测性能。使用三层交叉注意力实现提示嵌入与图像特征的双向交互：
   - 提示嵌入→图像特征的cross-attention
   - 图像特征→更新后提示嵌入的cross-attention

   引入可学习的类别嵌入（class embedding），为每个类别分配随机ID并映射为嵌入，防止不同粒度任务间的冲突。

   总检测损失 $\mathcal{L}_{\text{det}} = \mathcal{L}_{\text{fus}} + \mathcal{L}_{\text{aln}}$

### 损失函数 / 训练策略

**三阶段训练流水线**：

1. **预训练**：仅训练检测模块，使用7个标注数据集 + Million-AID（SAM生成区域提议+DINOv2聚类），896×896输入，360k迭代
2. **微调**：扩展到10个不同粒度数据集，经验采样率训练，832×832输入，288k迭代
3. **自训练**：用微调后检测器生成伪标签，覆盖200个预定义类别；经SkyCLIP过滤（相似度阈值0.24）、类别树硬负样本挖掘、类无关NMS去重；自标注数据以1/2采样率与原始数据混合继续训练

## 实验关键数据

### 主实验

7个公开遥感数据集上的OBB检测结果（AP50）：

| 方法 | DIOR-R | DOTA-v1.0 | DOTA-v2.0 | FAIR1M-2.0 | WHU-Mix | SpaceNet | HRSC2016 | 平均 | FPS |
|------|--------|-----------|-----------|------------|---------|----------|----------|------|-----|
| Oriented R-CNN (Swin-T) | 65.8 | 76.0 | 67.4 | 45.0 | 79.3 | 44.6 | 79.1 | 65.3 | 6.9 |
| RTMDet-R-L | 70.1 | 71.5 | 66.7 | 47.6 | 79.9 | 47.4 | 79.9 | 66.2 | 23.8 |
| CastDet* | 63.3 | 72.1 | 63.1 | 37.7 | 77.7 | 45.0 | 74.1 | 61.9 | 6.8 |
| **OpenRSD (Text)** | **73.7** | **76.9** | **70.1** | 46.1 | 79.7 | **50.2** | **88.1** | **69.3** | 20.8 |

HBB检测中，OpenRSD比YOLO-World-L平均高8.7%精度，比Grounding DINO-T快4倍（20.8 vs 5.4 FPS）且精度相当。

### 消融实验

**模块消融**：

| 对齐头 | 融合头 | 类别嵌入 | DIOR-R | DOTA-v2.0 | FAIR1M-2.0 | HRSC2016 | 平均 |
|--------|--------|----------|--------|-----------|------------|----------|------|
| ✓ | | | 71.5 | 68.8 | 45.4 | 84.3 | 63.3 |
| ✓ | ✓ | | 73.1 | 69.5 | 45.3 | 73.1 | 61.8 |
| ✓ | ✓ | ✓ | 72.9 | **70.4** | **45.9** | **84.6** | **64.3** |

**训练阶段消融**：预训练→微调→自训练逐步提高平均AP，从47.2→65.5→66.9。

### 关键发现

- **融合头精度更高**：推理时使用融合头比对齐头平均高1.2%，深度跨模态交互增强了多数据集联合训练的泛化性
- **文本提示更稳定**：1个提示即稳定工作；图像提示随数量增加才逐步改善
- **跨域泛化优秀**：在4个未见过数据集上，OpenRSD比RTMDet-L平均高2.2%，在小目标数据集SODA-A上提升5.2%
- **自训练有效**：全参数微调的自训练策略最优，HRSC2016上从84.6提升到87.8

## 亮点与洞察

1. **双头设计灵活实用**：对齐头速度快、词汇可扩展；融合头精度高——可根据应用场景选择
2. **三阶段训练策略**贯穿"适配→泛化→补全"的完整逻辑链
3. **图像提示**减少对专家知识依赖，遥感领域许多细粒度类别（如飞机型号）难以用文本准确描述
4. **ORSD+大规模数据集**：47万张图像、200个类别，为遥感开放检测提供训练基础

## 局限与展望

- 构建ORSD+依赖多个特定数据集的人工整合和伪标签生成，可扩展性受数据源限制
- 未探索语义分割等更密集任务的扩展
- 自训练中SkyCLIP过滤的阈值（0.24）为手动设定，自适应阈值选择可能更优

## 相关工作与启发

- 结合YOLO-World的轻量交互和Grounding DINO的深度融合，在遥感场景下取得了很好的平衡
- 三阶段训练可启发其他领域（如医学影像）构建通用开放检测器
- 图像提示+文本提示的混合输入模式值得推广

## 评分

- **新颖性**: ⭐⭐⭐⭐ 双头设计+三阶段训练的组合较新颖，但各组件并非首创
- **实验充分度**: ⭐⭐⭐⭐⭐ 7个评测+4个跨域+详细消融，非常充分
- **写作质量**: ⭐⭐⭐ 部分描述有语法错误和重复，可进一步打磨
- **实用价值**: ⭐⭐⭐⭐⭐ 20.8 FPS的实时遥感检测，实际部署价值高

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images](../../CVPR2025/object_detection/small_target_detection_based_on_mask-enhanced_attention_fusion_of_visible_and_in.md)
- [\[CVPR 2026\] Fourier Angle Alignment for Oriented Object Detection in Remote Sensing](../../CVPR2026/object_detection/fourier_angle_alignment_for_oriented_object_detection_in_remote_sensing.md)
- [\[AAAI 2026\] SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection](../../AAAI2026/object_detection/sm3det_a_unified_model_for_multi-modal_remote_sensing_object_detection.md)
- [\[ECCV 2024\] MutDet: Mutually Optimizing Pre-training for Remote Sensing Object Detection](../../ECCV2024/object_detection/mutdet_mutually_optimizing_pre-training_for_remote_sensing_object_detection.md)
- [\[ICCV 2025\] 3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection](3dmood_lifting_2d_to_3d_for_monocular_openset_object_detecti.md)

<!-- RELATED:END -->
