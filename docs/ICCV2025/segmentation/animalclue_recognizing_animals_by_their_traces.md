---
title: >-
  [论文解读] AnimalClue: Recognizing Animals by their Traces
description: >-
  [ICCV 2025][图像分割][动物痕迹识别] 提出 AnimalClue，首个大规模动物痕迹识别数据集，包含 159,605 个边界框覆盖 968 个物种的五类间接线索（脚印、粪便、蛋、骨骼、羽毛），并建立了分类、检测、实例分割和特征预测四项基准。
tags:
  - ICCV 2025
  - 图像分割
  - 动物痕迹识别
  - 野生动物保护
  - 间接证据
  - 数据集
  - 实例分割
---

# AnimalClue: Recognizing Animals by their Traces

**会议**: ICCV 2025  
**arXiv**: [2507.20240](https://arxiv.org/abs/2507.20240)  
**代码**: https://dahlian00.github.io/AnimalCluePage/  
**领域**: 分割 / 目标检测 / 图像分类  
**关键词**: 动物痕迹识别, 野生动物保护, 间接证据, 数据集, 实例分割

## 一句话总结

提出 AnimalClue，首个大规模动物痕迹识别数据集，包含 159,605 个边界框覆盖 968 个物种的五类间接线索（脚印、粪便、蛋、骨骼、羽毛），并建立了分类、检测、实例分割和特征预测四项基准。

## 研究背景与动机

野生动物监测对生物多样性保护至关重要。计算机视觉已在动物直接识别（外观检测）方面取得显著进展，但通过间接证据（如脚印、粪便）识别物种的研究仍然匮乏。现有数据集存在严重不足：
- OpenAnimalTracks 仅含 18 种、3,579 个边界框
- FeathersV1 仅支持分类任务
- 现有数据集物种覆盖少、标注类型单一

生态调查中广泛依赖间接证据进行物种识别，但高度依赖人工，急需自动化的计算机视觉解决方案。AnimalClue 的目标是填补这一空白，提供一个涵盖多种痕迹类型、多种任务的大规模基准。

## 方法详解

### 整体框架

AnimalClue 是一个数据集+基准工作，核心贡献在于数据构建和实验评估：

### 关键设计

1. **数据来源与收集**

    - 从 iNaturalist 平台收集数据，选择研究级（research-grade）图片，确保标签由多位公民科学家验证
    - 仅选择 Creative Commons 许可的图片，移除模糊、不清晰和含人脸的图片
    - 涵盖五类动物痕迹：脚印（18,291 bbox）、粪便（18,932 bbox）、骨骼（16,553 bbox）、蛋（29,434 bbox）、羽毛（76,395 bbox）

2. **标注策略**

    - 脚印：仅标注边界框（因为脚印是痕迹而非物理实体，部分区域模糊）
    - 粪便、骨骼、蛋、羽毛：提供像素级分割 mask
    - 粪便和鸡蛋使用 SAM 辅助初始标注，作者人工校对
    - 同一 iNaturalist 条目的多张图片不跨训练/测试集划分，防止数据泄漏

3. **细粒度特征标注**

    - 共标注 22 种生态和行为属性，包括：
        - 分类学信息（目、科）
        - 饮食类型（草食、肉食、杂食）
        - 活动模式（昼行、夜行、晨昏活动）
        - 栖息地偏好（森林、草原、沙漠、湿地、山地、城市）
        - 气候分布（热带、亚热带、温带、北方、极地）
        - 社会行为（群居、迁徙、捕食者）

4. **频率分类**

    - 按训练集频率将物种分为 frequent（前20%）、intermediate（中60%）和 rare（后20%）三组
    - 分别针对五种痕迹类型独立划分

### 数据集统计

| 痕迹类型 | 边界框数 | 图片数 | 物种数 | 科数 | 目数 |
|----------|---------|--------|--------|------|------|
| 脚印 | 18,291 | 7,581 | 117 | 46 | 20 |
| 粪便 | 18,932 | 6,433 | 101 | 46 | 21 |
| 骨骼 | 16,553 | 12,908 | 269 | 112 | 45 |
| 蛋 | 29,434 | 9,394 | 283 | 67 | 20 |
| 羽毛 | 76,395 | 60,491 | 555 | 89 | 30 |

## 实验关键数据

### 主实验——分类

| 模型 | 脚印(Species) | 粪便(Species) | 蛋(Species) | 骨骼(Species) | 羽毛(Species) |
|------|-------------|-------------|-----------|-------------|-------------|
| VGG-16 | 28.8 | 29.6 | 45.2 | 14.7 | 56.7 |
| ResNet-50 | 23.7 | 29.4 | 41.1 | 18.3 | 59.7 |
| ViT-B | 29.2 | 32.2 | 46.7 | 15.0 | 55.9 |
| Swin-B | **32.3** | **38.6** | **49.4** | **20.5** | **65.3** |

### 消融/检测结果

| 检测模型 | 脚印(Species mAP) | 蛋(Species mAP) | 羽毛(Species mAP) |
|----------|-------------------|-----------------|-------------------|
| YOLOv8 | 0.10 | 0.13 | 0.25 |
| YOLOv11 | 0.10 | 0.14 | 0.25 |
| RT-DETR | 0.10 | 0.04 | 0.17 |
| DINO | 0.08 | 0.20 | 0.15 |

| 分割模型 | 粪便(Species) | 蛋(Species) | 骨骼(Species) | 羽毛(Species) |
|----------|-------------|-----------|-------------|-------------|
| YOLOv8 | 0.11 | 0.11 | 0.07 | 0.24 |
| MaskDINO | **0.13** | **0.25** | **0.07** | 0.18 |
| YOLOv11 | 0.11 | 0.12 | 0.06 | **0.24** |

### 关键发现

- Swin-B 在各分类任务中一致表现最佳，Transformer 架构更适合捕捉痕迹的细粒度特征
- 羽毛识别准确率最高（65.3%），尽管物种数最多（555种），因其颜色和纹理特征鲜明
- 骨骼识别最困难（20.5%），因外观随身体部位变化大
- 稀有物种识别极具挑战：Swin-B 在稀有脚印种级仅 14.2%，稀有羽毛仅 2.52%
- 检测和分割任务整体 mAP 偏低（最高 Order 检测 0.57），表明该任务远未解决
- 经过 AnimalClue 微调的 CLIP 在 t-SNE 可视化中展现最佳特征分离

## 亮点与洞察

- **独特的问题设置**：从间接证据识别动物物种，与传统直接外观识别互补，具有重要的生态学应用价值
- **规模与全面性**：968 种、5 类痕迹、4 项任务 + 22 种特征标注，远超已有数据集
- **揭示关键挑战**：稀有种泛化困难、物种级检测/分割 mAP 极低，表明该领域有大量研究空间

## 局限与展望

- 物种分布极不均衡，长尾问题严重
- 脚印仅有边界框标注，缺少分割 mask
- 目前仅评估了标准模型，未探索预训练或域适应等策略
- 跨痕迹类型的联合识别（如同时利用脚印和粪便识别同一物种）未被探索
- 数据主要来自 iNaturalist，可能存在地理和物种偏差

## 相关工作与启发

- 与传统的动物外观识别数据集（iNat、CUB-200）形成互补
- 22 种特征标注为多任务学习和零样本学习提供了丰富的辅助信号
- 可启发将此方法扩展到其他间接证据识别场景（如犯罪现场、考古学）

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个大规模间接动物痕迹数据集，问题设置新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 四项任务基准，多模型全面评估，频率分析完善
- 写作质量: ⭐⭐⭐⭐ 数据集构建描述清晰，统计完整
- 价值: ⭐⭐⭐⭐ 为野生动物监测的CV研究开辟新方向，数据集可持续产生影响

<!-- RELATED:START -->

## 相关论文

- [SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation](score_scene_context_matters_in_open-vocabulary_remote_sensing_instance_segmentat.md)
- [ROADWork: A Dataset and Benchmark for Learning to Recognize, Observe, Analyze and Drive Through Work Zones](roadwork_a_dataset_and_benchmark_for_learning_to_recognize_observe_analyze_and_d.md)
- [MOVE: Motion-Guided Few-Shot Video Object Segmentation](move_motion-guided_few-shot_video_object_segmentation.md)
- [Beyond Single Images: Retrieval Self-Augmented Unsupervised Camouflaged Object Detection](beyond_single_images_retrieval_self-augmented_unsupervised_camouflaged_object_de.md)
- [OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation](omnisam_omnidirectional_segment_anything_model_for_uda_in_panoramic_semantic_seg.md)

<!-- RELATED:END -->
