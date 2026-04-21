---
title: >-
  [论文解读] Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces
description: >-
  [CVPR 2025][3D视觉][3D场景图] 提出功能性3D场景图新任务，利用VLM和LLM通过渐进式检测-描述-推理pipeline从RGB-D图像中构建包含物体、交互元素及其功能关系的3D场景图，并建立了FunGraph3D真实世界数据集。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D场景图
  - 功能关系
  - 交互元素
  - 基础模型
  - 开放词汇
---

# Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces

**会议**: CVPR 2025  
**arXiv**: [2503.19199](https://arxiv.org/abs/2503.19199)  
**代码**: https://openfungraph.github.io  
**领域**: 3D视觉  
**关键词**: 3D场景图, 功能关系, 交互元素, 基础模型, 开放词汇

## 一句话总结

提出功能性3D场景图新任务，利用VLM和LLM通过渐进式检测-描述-推理pipeline从RGB-D图像中构建包含物体、交互元素及其功能关系的3D场景图，并建立了FunGraph3D真实世界数据集。

## 研究背景与动机

**领域现状**：3D场景图将室内实体组织为图结构。现有方法如Open3DSG、ConceptGraph等已能推断场景图，但节点仅限于物体，边仅表示空间关系（如"电视在墙上"）。

**现有痛点**：缺少小型交互元素节点（开关、把手、按钮等）和功能性关系边（如"开关控制灯"）。空间关系已被物体位置隐含编码，价值有限。功能关系对机器人操作至关重要。

**核心矛盾**：构建功能性场景图需理解部件级交互元素和因果关系，但缺乏训练数据，且许多功能关系无法从静态视觉推断。

**本文目标**：(1) 形式化功能性3D场景图；(2) 无需训练数据，利用基础模型零样本构建；(3) 建立标注数据集。

**切入角度**：VLM和LLM中已编码丰富功能知识——VLM能识别物体/交互元素，LLM掌握常识功能关系。通过精心prompt和渐进推理可零样本提取功能知识。

**核心 idea**：将功能性场景图构建分解为三阶段——渐进检测（物体→交互元素）、多视角VLM+LLM描述、顺序功能关系推理（局部→远程）。

## 方法详解

### 整体框架

OpenFunGraph定义$\mathcal{G} = (\mathcal{O}, \mathcal{I}, \mathcal{R})$，$\mathcal{O}$为物体，$\mathcal{I}$为交互元素，$\mathcal{R}$为功能关系边。Pipeline：节点检测→节点描述→功能关系推理。

### 关键设计

1. **渐进式节点检测**:

    - 功能：检测物体和交互元素并3D融合
    - 核心思路：RAM++识别物体标签→GroundingDINO检测。交互元素：向GPT-4查询每个物体的可能交互元素列表，将物体标签+元素标签拼接（如"door. handle"）作为GroundingDINO提示，提高小目标检测。深度图反投影+多视角融合到3D
    - 设计动机："从物体到部件"的渐进策略加物体上下文提示大幅提高小目标检测率

2. **多视角协作描述**:

    - 功能：为每个节点生成自然语言描述
    - 核心思路：物体：选top-$N_v$视角用LLAVA描述，GPT-4综合。交互元素：多尺度放大+红色轮廓高亮引导VLM注意力，多视角综合描述
    - 设计动机：单视角易受遮挡影响，小交互元素需多尺度+高亮才能让VLM正确识别

3. **顺序功能关系推理**:

    - 功能：推断物体与交互元素的功能关系
    - 核心思路：先处理局部关系（柜门-把手）——3D空间重叠筛选+LLM常识判断。再处理远程关系（灯-开关）——LLM生成候选→VLM视觉验证→LLM分配置信度。Chain-of-Thought式分步推理
    - 设计动机：局部/远程关系推理逻辑完全不同，分步比一次性推断更可靠

### 损失函数 / 训练策略

完全基于基础模型零样本推理，无需训练。使用RAM++/GroundingDINO检测、LLAVA v1.6视觉理解、GPT-4常识推理。

## 实验关键数据

### 主实验

节点检测Recall@10（SceneFun3D/FunGraph3D）：

| 方法 | 物体 | 交互元素 | 整体 |
|------|------|---------|------|
| Open3DSG* | 70.7/58.1 | 61.8/33.9 | 64.7/43.6 |
| ConceptGraph*+IED | 77.1/66.3 | 59.5/33.4 | 66.0/45.0 |
| **OpenFunGraph** | **87.8/79.1** | **79.5/57.6** | **82.8/65.8** |

### 消融实验

| 配置 | 影响 |
|------|------|
| 无辅助物体标签提示 | 交互元素检测大幅下降 |
| 无多尺度描述 | 描述质量差 |
| 一步推理(无局部/远程分步) | 关系准确率下降 |

### 关键发现

- 交互元素检测是最大瓶颈（79.5% vs 87.8%物体），部件级检测仍很有挑战
- ConceptGraph几乎无法检测交互元素（原始仅8.6%），现有场景图方法从未考虑部件级理解
- FunGraph3D比SceneFun3D更具挑战性

## 亮点与洞察

- **功能性3D场景图**的问题定义是重要贡献——从空间关系到功能关系是质变
- **交互元素的辅助标签提示策略**简单有效，可迁移到任何部件检测场景
- **局部→远程的顺序推理**避免了LLM面对大量候选时的推理混乱

## 局限与展望

- 静态观察无法确定某些远程关系
- 依赖多个大模型串联，推理成本高
- 仅处理室内场景
- 评估指标基于嵌入余弦相似度可能有偏差

## 相关工作与启发

- **vs Open3DSG**: 基于CLIP+GNN推断空间关系，无法处理部件和功能关系
- **vs ConceptGraph**: 只关注物体和空间关系，交互元素检测能力极弱
- **vs SceneFun3D**: 提供交互元素标注但缺少物体标注和关系标注

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 功能性3D场景图是全新任务
- 实验充分度: ⭐⭐⭐⭐ 两个数据集评估充分
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰、pipeline图直观
- 价值: ⭐⭐⭐⭐⭐ 开创性工作，对机器人操作有巨大潜在影响

<!-- RELATED:START -->

## 相关论文

- [Masked Point-Entity Contrast for Open-Vocabulary 3D Scene Understanding](masked_point-entity_contrast_for_open-vocabulary_3d_scene_understanding.md)
- [Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)
- [Open-Vocabulary Octree-Graph for 3D Scene Understanding](../../ICCV2025/3d_vision/open-vocabulary_octree-graph_for_3d_scene_understanding.md)
- [SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding](seeground_see_and_ground_for_zero-shot_open-vocabulary_3d_visual_grounding.md)
- [Open-World Amodal Appearance Completion](open-world_amodal_appearance_completion.md)

<!-- RELATED:END -->
