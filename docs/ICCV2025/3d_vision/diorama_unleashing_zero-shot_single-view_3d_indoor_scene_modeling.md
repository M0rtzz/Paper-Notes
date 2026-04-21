---
title: >-
  [论文解读] Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling
description: >-
  [ICCV 2025][3D视觉][零样本场景建模] 提出Diorama，首个零样本开放世界系统，从单张RGB图像通过模块化管线（开放世界感知+基于CAD的场景建模）生成完整的3D室内场景，包含建筑结构和物体摆放，无需端到端训练或人工标注。
tags:
  - ICCV 2025
  - 3D视觉
  - 零样本场景建模
  - 单视图3D重建
  - CAD检索
  - 基础模型
  - 室内场景
---

# Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling

**会议**: ICCV 2025  
**arXiv**: [2411.19492](https://arxiv.org/abs/2411.19492)  
**代码**: [有](https://3dlg-hcvc.github.io/diorama/)  
**领域**: 3D视觉  
**关键词**: 零样本场景建模, 单视图3D重建, CAD检索, 基础模型, 室内场景

## 一句话总结

提出Diorama，首个零样本开放世界系统，从单张RGB图像通过模块化管线（开放世界感知+基于CAD的场景建模）生成完整的3D室内场景，包含建筑结构和物体摆放，无需端到端训练或人工标注。

## 研究背景与动机

从单张图像重建结构化3D场景是计算机视觉的基础任务。现有方法存在明显局限：

**重建方法的不足**：NeRF/3DGS等方法产生的是不完整的表面网格，与现代图形管线不兼容；缺乏组合性和可交互性

**CAD对齐方法的局限**：Mask2CAD、ROCA、DiffCAD等方法需要大量标注数据进行端到端训练，且不建模建筑结构（墙壁、地板等）

**生成式方法的weakness**：LLM驱动的场景生成方法缺乏精细的3D空间知识，无法准确定位物体

**基础模型的机遇**：近期研究表明基础模型具有3D感知能力，但尚未被充分用于整体场景建模

Diorama回答了这样一个问题："能否仅用基础模型从单张真实图像进行整体3D场景建模？"

## 方法详解

### 整体框架

系统包含两大组件：
1. **开放世界感知**：从输入图像理解场景（物体识别、深度/法线估计、建筑重建、场景图生成）
2. **基于CAD的场景建模**：组装紧凑的3D场景（CAD检索、9-DoF位姿估计、语义感知的布局优化）

### 关键设计

**1. 开放世界感知**

- **物体检测与分割**：结合OWLv2（开放词汇检测器）和SAM（分割一切）定位物体实例
- **深度/法线估计**：使用Metric3DV2估计度量深度和法线图，反投影获得点云
- **场景图生成**：利用GPT-4o的视觉理解能力，通过SoM方式生成支撑层级关系图$G = \langle V, E \rangle$

**2. PlainRecon建筑重建**

简洁而有效的建筑重建管线：
- 分割并移除物体 → 图像修复得到"空房间"
- 深度/法线估计提取建筑元素点云
- 基于法线的聚类拟合3D平面

**3. 多模态CAD检索**

使用DuoDuoCLIP统一编码文本/图像/3D形状，采用层级检索策略：先通过文本检索确保类别正确 → 再通过图像查询按外观重排

**4. 零样本9-DoF位姿估计**

- 利用DinoV2提取语义丰富的patch特征
- 计算查询图像与CAD多视图渲染之间的2D对应
- 通过深度提升到3D对应 → Umeyama算法+RANSAC求解刚体变换
- 借用GigaPose的scale预测网络增强鲁棒性

**5. 分阶段布局优化**

解决粗糙位姿导致的物理不合理性（穿模、悬浮等）：
- **阶段1-方向**：对齐接触面法线和支撑面法线（$e_{align}$），保持物体朝向（$e_{sem}$）
- **阶段2-放置**：确保物体放在支撑面上（$e_{place}$），维持相对距离（$e_{rel}$）
- **阶段3-空间**：处理物体包含关系（如书架内的书），惩罚突出（$e_{vol}$）
- **阶段4-细化**：再次运行放置阶段 + 碰撞检测惩罚（$e_{col}$，SAT算法）

### 损失函数 / 训练策略

无需训练。所有模块基于预训练基础模型的零样本推理：
- 场景理解：OWLv2 + SAM + Metric3DV2 + GPT-4o
- 形状检索：DuoDuoCLIP（预训练联合嵌入）
- 位姿估计：DinoV2特征匹配 + 几何求解
- 布局优化是可微的梯度下降过程

## 实验关键数据

### 主实验

**SSDB数据集系统级对比：**

| 方法 | rAcc↑ | tAcc↑ | sAcc↑ | Acc↑ | CD↓ | 用户偏好↑ | API费用($)↓ | 时间(min)↓ |
|------|-------|-------|-------|------|-----|---------|-----------|-----------|
| ACDC | 0.20 | 0.56 | 0.36 | 0.04 | 14.0 | 14.9% | 1.44 | 23.2 |
| **Diorama** | **0.23** | **0.68** | **0.49** | **0.08** | **9.5** | **85.1%** | **0.12** | **3.7** |

**建筑重建对比：**

| 方法 | 深度 | IoU↑ | PE↓ | EE↓ | CDb↓ | 时间(s)↓ |
|------|------|------|-----|-----|------|---------|
| RaC | DAv2 | 40.3 | 39.8 | 23.2 | 0.645 | 43 |
| ACDC | DAv2 | 46.8 | 17.0 | 31.0 | 0.563 | 32 |
| **PlainRecon** | M3Dv2 | **58.6** | **9.6** | **18.9** | **0.447** | 29 |

### 消融实验

**位姿估计方法对比（SSDB）：**

| 方法 | rAcc↑ | tAcc↑ | sAcc↑ | Acc↑ | 碰撞↓ | 关系正确率↑ |
|------|-------|-------|-------|------|-------|----------|
| BM基线 | 0.34 | 0.93 | 0.52 | 0.19 | 11.43 | 0.58 |
| ZSP | 0.36 | 0.92 | 0.59 | 0.25 | 8.76 | 0.60 |
| GigaPose | 0.36 | 0.95 | 0.71 | 0.27 | 7.91 | 0.61 |
| **Diorama** | **0.47** | **0.95** | **0.70** | **0.37** | **6.42** | **0.62** |

**3D形状检索（CD↓）：**

| 模型 | SS-Household | OOD-Household | SS-Furniture | OOD-Furniture |
|------|-------------|---------------|-------------|---------------|
| CLIP-H | 8.5 | 11.2 | 5.1 | 8.8 |
| DD-V | 6.4 | 12.0 | 8.9 | 12.1 |
| **DD-H (Ours)** | **5.5** | **9.9** | **3.2** | **7.6** |

### 关键发现

- Diorama获得85.1%的用户偏好，API费用仅$0.12/场景
- PlainRecon在建筑重建上全面超越基线，IoU提升12+点
- 层级文本+图像检索（DD-H）在所有类别和分布设定下都最优
- 分阶段优化显著降低碰撞，提升支撑关系正确率
- 系统可泛化到真实互联网图像和text-to-scene任务

## 亮点与洞察

1. **模块化设计的优势**：每个子任务使用最合适的基础模型，支持灵活替换；无需收集3D标注数据
2. **PlainRecon管线**：修复→深度→法线聚类的三步策略在简洁性和效果间取得了很好的平衡
3. **场景图驱动的优化**：支撑层级关系为布局优化提供了语义约束，使优化有明确的目标
4. **多假设输出**：通过不同的检索结果生成多个语义相似但外观不同的场景方案

## 局限与展望

1. 深度估计误差会级联传播到后续模块，特别是大场景中误差更明显
2. 检索的CAD与实际物体存在几何/外观不匹配，限制了场景的精确还原
3. 对严重遮挡的物体识别和位姿估计能力有限
4. 当前评估以室内场景为主，户外场景未验证
5. 场景图生成依赖GPT-4o的API，增加了系统复杂性和成本

## 相关工作与启发

- 与IM2CAD等经典方法对比，Diorama的开放世界能力是质的飞跃
- 与DiffCAD等训练式方法不同，零样本方式避免了real-synthetic domain gap
- 启发性地展示了基础模型组合在3D任务上的强大潜力

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （首个零样本开放世界单视图场景建模系统）
- 实验充分度: ⭐⭐⭐⭐ （模块级+系统级评估全面，但数据集规模有限）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，管线复杂但描述得当）
- 价值: ⭐⭐⭐⭐⭐ （开创性工作，展示了基础模型组合的巨大潜力）

<!-- RELATED:START -->

## 相关论文

- [Zero-Shot Inexact CAD Model Alignment from a Single Image](zero-shot_inexact_cad_model_alignment_from_a_single_image.md)
- [Boosting Multi-View Indoor 3D Object Detection via Adaptive 3D Volume Construction](boosting_multi-view_indoor_3d_object_detection_via_adaptive_3d_volume.md)
- [CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image](catsplat_contextaware_transformer_with_spatial_guidance_for.md)
- [MonoMobility: Zero-Shot 3D Mobility Analysis from Monocular Videos](monomobility_zero-shot_3d_mobility_analysis_from_monocular_videos.md)
- [Accelerate 3D Object Detection Models via Zero-Shot Attention Key Pruning](accelerate_3d_object_detection_models_via_zero-shot_attention_key_pruning.md)

<!-- RELATED:END -->
