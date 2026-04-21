---
title: >-
  [论文解读] MEDISEG: A Dataset of Medication Images with Instance Segmentation Masks for Preventing Adverse Drug Events
description: >-
  [CVPR 2026][图像分割][药物识别] 提出MEDISEG数据集——32种药片类型共8262张真实多药丸场景图像（含dosette box中重叠/遮挡/不同光照），提供实例分割标注，YOLOv8/v9在3-Pills子集mAP@50达99.5%、32-Pills达80.1%，few-shot实验证明MEDISEG作为base训练集显著优于CURE数据集。
tags:
  - CVPR 2026
  - 图像分割
  - 药物识别
  - 实例分割
  - 药物安全
  - dosette box
  - few-shot检测
  - YOLOv8
  - YOLOv9
---

# MEDISEG: A Dataset of Medication Images with Instance Segmentation Masks for Preventing Adverse Drug Events

**会议**: CVPR 2026  
**arXiv**: [2603.10825](https://arxiv.org/abs/2603.10825)  
**代码**: 无（数据集公开，CC BY 4.0）  
**领域**: 实例分割 / 医疗影像 / 数据集  
**关键词**: 药物识别, 实例分割, 药物安全, dosette box, few-shot检测, YOLOv8, YOLOv9  

## 一句话总结
提出MEDISEG数据集——32种药片类型共8262张真实多药丸场景图像（含dosette box中重叠/遮挡/不同光照），提供实例分割标注，YOLOv8/v9在3-Pills子集mAP@50达99.5%、32-Pills达80.1%，few-shot实验证明MEDISEG作为base训练集显著优于CURE数据集。

## 背景与动机
药物错误（medication errors）和不良药物事件（Adverse Drug Events, ADEs）是全球患者安全的重大威胁。据统计，英国每年约有2.37亿次药物错误，美国ADEs每年导致约10万人死亡。其中用药错误（administration errors）占很大比例，尤其是多种药物混装在dosette box/pill organiser中时，形状/颜色相似的药片极易混淆。

计算机视觉辅助药物识别是有前景的解决方案，但现有数据集存在严重缺陷：
- **National Library of Medicine (NLM) Pill Image Dataset**: 受控环境下单个药丸拍摄，无重叠/遮挡
- **CURE Dataset**: 包含19000+张图像但仅有目标检测bbox标注，无实例分割
- **RxImage/C3PI**: 类似受控环境，不反映真实使用场景

缺乏在真实多药丸场景（多个药丸堆叠、遮挡、光照变化）下的实例级分割标注数据集。

## 核心问题
如何构建一个反映真实dosette box用药场景的药物图像数据集，提供实例分割标注，支持多药丸重叠/遮挡下的药物自动识别？

## 方法详解

### 数据采集
- **设备**: iPhone 12 Pro Max（主摄像头）
- **拍摄场景**: dosette box内多种药片混合放置，模拟真实用药场景
- **光照条件**: 自然光和人工照明下多种条件
- **药片类型**: 32种常见处方药和OTC药片，涵盖不同形状（圆形、椭圆形、胶囊型）、颜色、大小

### 数据集结构
两个子集分别面向不同难度的识别任务：

1. **3-Pills子集**: 
    - 3种外观高度相似的药片类型
    - 聚焦细粒度区分能力
    - 每张图像含少量药丸，测试模型对相似药物的辨别精度

2. **32-Pills子集**:
    - 全部32种药片类型
    - 每帧最多含13个药丸
    - 大量重叠、遮挡和部分可见情况
    - 总计8262张图像

### 标注流程
- **标注工具**: COCO Annotator
- **标注格式**: COCO实例分割格式（多边形掩码）
- **标注内容**: 每个药丸的精确轮廓边界，支持实例级分割
- **质量控制**: 多轮校验保证标注一致性

### Baseline模型验证

#### 目标检测/实例分割
- **YOLOv8**: 在3-Pills和32-Pills上分别训练评估
- **YOLOv9**: 同样双子集评估
- **训练策略**: 标准COCO预训练权重微调

#### Few-shot检测实验
- **框架**: FsDet（Few-shot Detection）
- **设计思路**: 用MEDISEG作为base class训练集，测试对unseen药片类型的识别能力
- **对比**: 用CURE数据集作为base训练 vs 用MEDISEG作为base训练
- **超参优化**: 遗传算法（GA）搜索，70次迭代优化

## 实验关键数据

### 目标检测性能（mAP@50）

| 模型 | 3-Pills | 32-Pills |
|------|---------|----------|
| YOLOv8 | **99.5%** | **80.1%** |
| YOLOv9 | ~99% | ~78% |

### Few-shot实验

| Base训练集 | Novel类mAP@50 |
|-----------|---------------|
| CURE | 较低（基线） |
| MEDISEG | **显著提升** |

MEDISEG作为base训练集在few-shot迁移到unseen药片类型时，性能大幅优于CURE，说明MEDISEG中多药丸真实场景的实例分割标注为模型提供了更丰富的空间、遮挡和上下文信息。

### 数据集统计

| 指标 | 3-Pills | 32-Pills |
|------|---------|----------|
| 药片类型数 | 3 | 32 |
| 最大药丸/帧 | ~3 | 13 |
| 标注类型 | 实例分割 | 实例分割 |

## 亮点
- **数据集开创性**: 首个面向真实多药丸dosette box场景的实例分割数据集，填补了受控环境 → 真实场景的数据空白
- **实用价值高**: 直接对准药物管理中的安全痛点（ADE预防），具有明确的临床应用前景
- **Few-shot迁移验证**: 不仅是一个数据集，还验证了MEDISEG作为少样本学习base class训练集的优越性
- **GA超参搜索**: 用遗传算法70次迭代优化few-shot检测器超参数，方法论严谨
- **CC BY 4.0许可**: 开放获取，有利于社区复现和扩展

## 局限与展望
- **药片种类受限**: 32种药片仅覆盖小部分常用药物，实际药房可能有数百种
- **单一采集设备**: 仅用iPhone 12 Pro Max拍摄，缺乏不同手机/摄像头的多样性
- **32-Pills mAP仅80.1%**: 在真正部署时，约20%的漏检/误识别在医疗场景中可能不可接受
- **缺乏跨数据集泛化评估**: 未测试在其他药物数据集上的迁移性能（除few-shot外）
- **无磨损/破损药片**: 实际使用中药片可能有磨损、碎裂、褪色等情况
- **标注者一致性**: 论文未详细报告inter-annotator agreement指标

## 与相关工作的对比
- **NLM Pill Image Dataset**: 单药丸/受控背景/仅分类标签 → MEDISEG多药丸/真实场景/实例分割
- **CURE Dataset**: 19000+图像但仅bbox → MEDISEG提供多边形mask，支持精确轮廓分割
- **RxImage/C3PI**: 药品参考图像，不适合训练检测模型 → MEDISEG专为检测/分割设计
- **EPill-Seg (MICCAI)**: 聚焦内窥镜下药丸分割，场景完全不同 → MEDISEG针对日常用药场景

## 启发与关联
- 数据集构建思路可推广到其他医疗安全场景（如手术器械清点、输液瓶识别）
- Few-shot实验证明真实场景数据的迁移价值——对"应该投资高质量标注还是大量弱标注"这个问题有启发
- 32类药片的细粒度区分与一般细粒度视觉识别（鸟类、汽车型号）共享挑战，可借鉴其方法
- 后续可结合SAM等通用分割模型做zero-shot/few-shot药物分割

## 评分
- 新颖性: ⭐⭐⭐ — 核心贡献是数据集而非方法创新，但填补了一个重要空白
- 实验充分度: ⭐⭐⭐⭐ — YOLOv8/v9 baseline + few-shot实验 + GA超参搜索，但缺少更多SOTA方法对比
- 写作质量: ⭐⭐⭐⭐ — 数据集论文结构清晰，动机论证充分
- 价值: ⭐⭐⭐⭐ — 对药物安全领域有直接实用价值，CC BY 4.0开放许可利于社区

<!-- RELATED:START -->

## 相关论文

- [MEDISEG: 药物图像实例分割数据集——预防不良药物事件](a_dataset_of_medication_images_with_instance_segme.md)
- [Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)
- [Dataset Enhancement with Instance-Level Augmentations](../../ECCV2024/segmentation/dataset_enhancement_with_instance-level_augmentations.md)
- [Phrase-Instance Alignment for Generalized Referring Segmentation](phrase-instance_alignment_for_generalized_referring_segmentation.md)
- [CA-LoRA: Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](ca-lora_concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)

<!-- RELATED:END -->
