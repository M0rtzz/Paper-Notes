---
title: >-
  [论文解读] LoD-Loc v3: Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment
description: >-
  [CVPR 2026][图像分割][无人机定位] 本文提出LoD-Loc v3，通过构建10万图像的大规模合成实例分割数据集InsLoD-Loc和将定位范式从语义轮廓对齐升级为实例轮廓对齐，解决了基于LoD城市模型的无人机定位中跨场景泛化差和密集城市歧义两大痛点，在Tokyo-LoDv3密集场景上比SOTA的(2m,2°)精度提升2000%。
tags:
  - CVPR 2026
  - 图像分割
  - 无人机定位
  - LoD城市模型
  - 实例分割
  - 合成数据
  - 轮廓对齐
---

# LoD-Loc v3: Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment

**会议**: CVPR 2026  
**arXiv**: [2603.19609](https://arxiv.org/abs/2603.19609)  
**代码**: [项目主页](https://nudt-sawlab.github.io/LoD-Locv3/)  
**领域**: 分割 / 无人机定位  
**关键词**: 无人机定位, LoD城市模型, 实例分割, 合成数据, 轮廓对齐

## 一句话总结
本文提出LoD-Loc v3，通过构建10万图像的大规模合成实例分割数据集InsLoD-Loc和将定位范式从语义轮廓对齐升级为实例轮廓对齐，解决了基于LoD城市模型的无人机定位中跨场景泛化差和密集城市歧义两大痛点，在Tokyo-LoDv3密集场景上比SOTA的(2m,2°)精度提升2000%。

## 研究背景与动机
1. **领域现状**：UAV视觉定位的主流方法依赖高精度3D重建（SfM/摄影测量），虽然精度高但模型构建和维护成本大，数据量庞大，且存在隐私和安全问题。基于LoD（Level-of-Detail）城市模型的定位方法是更轻量的替代方案——LoD模型只保留建筑几何结构，遵循CityGML标准，已在美国、中国、日本、德国等国家大规模建设。
2. **现有痛点**：LoD-Loc v2通过将图像中的建筑语义分割轮廓与LoD模型渲染的轮廓对齐来定位，但存在两个关键问题：(1) **泛化差**——在一个城市训练的模型部署到另一个城市时性能严重下降；(2) **密集场景歧义**——在密集城市中，多栋建筑的语义轮廓合并为一大块，不同pose渲染出的语义mask高度相似，导致无法区分。
3. **核心矛盾**：语义分割只区分"建筑"和"背景"，在密集建筑区域所有建筑连成一片，丧失了区分性信息。而不同pose下建筑的实例级排列是唯一的。
4. **本文目标** (1) 通过大规模合成数据解决跨场景泛化问题；(2) 通过实例级对齐解决密集场景的pose歧义问题。
5. **切入角度**：城市中的定位本质上是一个实例对齐过程——需要将图像中每栋可见建筑与LoD模型中的对应建筑实例匹配。
6. **核心 idea**：用实例分割替代语义分割来提取建筑轮廓，并用Dice系数进行实例级一对一匹配来评估pose假设。

## 方法详解

### 整体框架
三阶段pipeline：(1) LoD模型实例化——为每栋建筑分配唯一ID/颜色；(2) 建筑实例分割——用SAM-based模型从查询图像中提取每栋建筑的独立mask；(3) 粗精定位——在4-DoF搜索空间中通过实例轮廓对齐评估pose假设。粗阶段均匀采样pose空间，精阶段用粒子滤波迭代优化。

### 关键设计

1. **InsLoD-Loc合成数据集（10万图像）**:

    - 功能：提供大规模、多样化的训练数据以实现跨场景零样本泛化
    - 核心思路：两阶段数据生成pipeline——(a) 使用UE5 + Cesium插件流式加载Google Earth Photorealistic 3D Tileset + AirSim插件渲染逼真的RGB图像；(b) 从公开LoD模型源获取对应的LoD模型，统一坐标系后用OSG渲染引擎生成实例mask。最终覆盖6个国家40个飞行区域，3种相机配置（不同FOV/分辨率/采样策略），高度200-500m
    - 设计动机：LoD-Loc v2仅在单个场景训练导致泛化差。InsLoD-Loc覆盖商业/工业/住宅/教育/医疗/郊区等多种用地类型，确保训练数据的多样性

2. **LoD模型实例化**:

    - 功能：为每栋建筑分配唯一标识符，支持实例级渲染
    - 核心思路：将无纹理的LoD模型解析为图$G=(V,E,F)$，每栋建筑$B_i$对应一个连通分量$G_i$，通过图分割将模型划分为$M$个不相交的建筑实例。每个实例分配唯一的24位RGB颜色ID，渲染时直接生成实例mask
    - 设计动机：语义分割只能得到"建筑 vs 非建筑"的二值mask，实例化后每栋建筑有独立mask，为后续的一对一匹配提供基础

3. **基于SAM的建筑实例分割**:

    - 功能：从查询图像中自动提取每栋建筑的独立mask
    - 核心思路：在SAM架构基础上增加可学习的Prompter Module——SAM编码器提取图像特征$F_{embed}$，Prompter Module从中预测prompt嵌入，SAM解码器生成实例mask集合$\mathcal{S}_q = \{M_q^j\}_{j=1}^N$。训练时冻结SAM编码器并用LoRA微调，只更新Prompter Module和SAM解码器
    - 设计动机：SAM具有强大的零样本分割能力，但不能自动生成实例分割结果。Prompter Module将SAM转化为自动的、任务特定的实例分割pipeline

### 实例轮廓对齐评估函数

对查询图像的每个预测实例$M_q^j$，在渲染实例集$\mathcal{S}_{hyp}$中找最佳匹配（最高Dice系数$d_j^*$）。最终cost为所有实例匹配分数的加权和。提供两种权重策略：

- **置信度加权**：$c_{ins}^{(conf)} = \sum_j \frac{s_j}{\sum_i s_i} d_j^*$
- **面积加权**：$c_{ins}^{(area)} = \sum_j \frac{A_j}{\sum_i A_i} d_j^*$

### 损失函数 / 训练策略
多任务训练损失 $L = L_{rpn} + L_{roi}$。RPN损失监督候选框生成，RoI损失监督最终分类/回归/mask预测。AdamW优化器，学习率$2 \times 10^{-4}$，权重衰减0.05，余弦退火，训练20个epoch。SAM编码器使用ViT-Huge预训练权重，LoRA微调。

## 实验关键数据

### 主实验（UAVD4L-LoDv2数据集，定位成功率%）

| 方法 | 训练数据 | in-Traj 2m-2° | out-Traj 2m-2° | in-Traj 5m-5° | out-Traj 5m-5° |
|------|---------|--------------|---------------|--------------|---------------|
| MC-Loc(DINOv2) | - | 1.20 | 2.40 | 17.40 | 26.10 |
| LoD-Loc | 分布内† | 49.56 | 54.20 | 89.09 | 89.51 |
| LoD-Loc v2 | 分布内† | 93.70 | 97.90 | 99.50 | 100.00 |
| **LoD-Loc v3** | InsLoD-Loc | **97.60** | **97.40** | **99.70** | **99.40** |

### Tokyo-LoDv3密集场景测试

| 方法 | Grid 2m-2° | Grid 5m-5° | Seq 2m-2° | Seq 5m-5° |
|------|-----------|-----------|----------|----------|
| LoD-Loc v2† | 22.70 | 74.70 | 35.60 | 92.00 |
| **LoD-Loc v3** | **39.30** | **89.90** | **50.30** | **97.30** |

### 消融实验（语义 vs 实例对齐，相同InsLoD-Loc数据训练）

| 对齐方式 | Grid 2m-2°/3m-3°/5m-5° | Seq 2m-2°/3m-3°/5m-5° |
|---------|----------------------|---------------------|
| LoD-Loc v2(语义) | 19.60/39.40/72.10 | 21.50/47.80/89.00 |
| **LoD-Loc v3(实例)** | **38.10/65.40/86.40** | **49.80/79.90/95.80** |

### 关键发现
- **跨场景零样本超越分布内训练**：LoD-Loc v3仅在合成数据InsLoD-Loc上训练，在UAVD4L-LoDv2上超越了分布内训练的LoD-Loc v2。这证明了足够多样化的合成数据可以替代分布内真实数据
- **实例对齐是关键而非数据量**：消融实验中，在相同InsLoD-Loc数据上训练的语义版(19.60%)远不如实例版(38.10%)，确认性能提升来自实例级范式而非数据规模
- **密集场景改善巨大**：Tokyo-LoDv3密集场景中，LoD-Loc v2几乎失败，而v3实现了显著提升，验证了实例对齐在消歧中的核心作用
- **面积加权和置信度加权性能相近**：两种策略各有优劣，面积加权在Swiss-EPFL上稍好
- **CAD-Loc等特征匹配方法完全失败**：所有基于SIFT/SuperPoint/LoFTR的方法在LoD模型上都是0%成功率，因为LoD模型没有纹理

## 亮点与洞察
- **从语义到实例的范式转换**思路简洁但效果巨大：同样的数据、同样的定位框架，只是将轮廓匹配从语义级改为实例级，密集场景性能翻倍。这揭示了在歧义场景中"精细粒度匹配"的重要性
- **UE5+Google Earth+OSG的数据生成pipeline**具有很强的工程价值：RGB渲染和实例mask渲染分别使用不同引擎但精确对齐，可扩展到任何有LoD模型的城市
- **LoD城市模型作为定位基准地图的潜力**：相比SfM点云，LoD模型极其轻量（只有几何外壳），已在全球多国大规模构建，有巨大的实际应用前景

## 局限与展望
- 实例分割在极端恶劣天气下可能失败
- LoD模型的精度本身有限，部分区域存在对齐误差
- 粗-精两阶段搜索在超大搜索空间下效率有限
- 仅在城市场景验证，无法处理非建筑区域（如森林、农田）
- 依赖4-DoF简化假设（重力方向已知），完整6-DoF情况未探索

## 相关工作与启发
- **vs LoD-Loc v2**: 直接前身，v3在v2框架上将语义升级为实例，并用大规模合成数据解决泛化问题
- **vs LoD-Loc**: 最早版本用线框对齐，需要高细节LoD2/3模型，v2/v3降到LoD1
- **vs CAD-Loc/MC-Loc**: 基于特征匹配/对齐的方法在无纹理LoD模型上完全失败
- **vs SAM**: 本文展示了SAM在领域特定任务上的适配方法——加Prompter Module + LoRA微调

## 评分
- 新颖性: ⭐⭐⭐⭐ 语义→实例的范式转换虽非技术突破但洞察深刻，合成数据pipeline设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、7+种baseline、多种消融、密集场景专项测试
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，技术方案叙述完整
- 价值: ⭐⭐⭐⭐ 对全球范围UAV导航有实际应用潜力，方法可立即部署

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Phrase-Instance Alignment for Generalized Referring Segmentation](phrase-instance_alignment_for_generalized_referring_segmentation.md)
- [\[CVPR 2026\] GeoSURGE: Geo-localization using Semantic Fusion with Hierarchy of Geographic Embeddings](geosurge_geo-localization_using_semantic_fusion_with_hierarchy_of_geographic_emb.md)
- [\[CVPR 2026\] SouPLe: Enhancing Audio-Visual Localization and Segmentation with Learnable Prompt Contexts](souple_enhancing_audio-visual_localization_and_segmentation_with_learnable_promp.md)
- [\[CVPR 2026\] MEDISEG: A Dataset of Medication Images with Instance Segmentation Masks for Preventing Adverse Drug Events](a_dataset_of_medication_images_with_instance_segmentation_masks_for_preventing_a.md)
- [\[CVPR 2026\] RecycleLoRA: Rank-Revealing QR-Based Dual-LoRA Subspace Adaptation for Domain Generalized Semantic Segmentation](recyclelora_rank-revealing_qr-based_dual-lora_subspace_adaptation_for_domain_gen.md)

<!-- RELATED:END -->
