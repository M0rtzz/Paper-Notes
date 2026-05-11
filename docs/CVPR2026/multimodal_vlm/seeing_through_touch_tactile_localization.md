---
title: >-
  [论文解读] Seeing Through Touch: Tactile-Driven Visual Localization of Material Regions
description: >-
  [CVPR 2026][多模态][触觉定位] 提出触觉定位任务——给定触觉输入识别图像中具有相同材质属性的区域，通过局部视觉-触觉对齐和材质多样性配对策略学习密集跨模态特征，构建两个新的触觉-材质分割数据集。
tags:
  - CVPR 2026
  - 多模态
  - 触觉定位
  - 多模态VLM
  - 材质分割
  - 跨模态学习
  - 数据集
---

# Seeing Through Touch: Tactile-Driven Visual Localization of Material Regions

**会议**: CVPR 2026  
**arXiv**: [2604.11579](https://arxiv.org/abs/2604.11579)  
**代码**: [https://mm.kaist.ac.kr/projects/SeeingThroughTouch/](https://mm.kaist.ac.kr/projects/SeeingThroughTouch/)  
**领域**: 多模态VLM  
**关键词**: 触觉定位, 视觉-触觉对齐, 材质分割, 跨模态学习, 数据集

## 一句话总结
提出触觉定位任务——给定触觉输入识别图像中具有相同材质属性的区域，通过局部视觉-触觉对齐和材质多样性配对策略学习密集跨模态特征，构建两个新的触觉-材质分割数据集。

## 研究背景与动机

**领域现状**：视觉-触觉学习主要聚焦于全局对齐（判断图像和触觉是否对应同一材质），但缺乏空间定位能力——无法在视觉场景中找到"摸起来一样"的区域。

**现有痛点**：(1) 全局对齐方法无法定位材质区域；(2) 现有数据集以近距离特写为主，视觉帧几乎无变化且单一材质填满画面，缺乏场景级多材质图像；(3) 缺少触觉-材质分割的评估基准。

**核心矛盾**：触觉定位需要细粒度的局部跨模态对应，但现有方法和数据都只提供粗粒度的全局对齐。

**核心 idea**：学习局部视觉-触觉对齐产生触觉显著性图，并通过材质多样性配对扩展有效训练对。

## 方法详解

### 整体框架
触觉编码器提取触觉特征（全局池化） + 视觉编码器提取空间特征图 → 计算密集相似度图 $M[h,w] = \bar{f}_t \cdot f_v[h,w]$ → 最大池化得到相似度分数用于对比学习 → 推理时直接用相似度图做触觉定位。

### 关键设计

1. **局部视觉-触觉对齐**:

    - 功能：学习空间分辨的跨模态特征
    - 核心思路：将触觉特征全局池化为 1D 向量，与视觉特征图的每个空间位置做点积得到相似度图，最大池化后用于对比学习。DINOv3 作为双编码器骨干，冻结视觉骨干仅训练对齐器
    - 设计动机：最大池化使模型关注图像中最匹配的区域，而非所有区域的平均响应，自然适合定位任务

2. **材质多样性配对策略**:

    - 功能：扩展有效训练对，增强跨实例泛化
    - 核心思路：域内配对——同一材质类别的不同触觉实例和不同视觉帧可以跨实例组合为正样本对；域外配对——收集网络场景图像并基于材质类别匹配触觉样本，利用"相似材质产生相似触觉"的假设
    - 设计动机：Touch-and-Go 中同一实例的视觉帧几乎相同导致有效训练对极少，跨实例和跨域配对大幅增加了多样性

3. **野外图像收集与过滤**:

    - 功能：补充场景级多材质图像
    - 核心思路：用 LLM 为每种材质类别生成多样搜索短语（如"brick chimney in a cozy living room"），从搜索引擎收集图像，用 CLIP 相似度过滤错分类样本，加上 MINC 材质数据集的图像
    - 设计动机：TG 数据集图像太近距离且单材质，无法训练场景级定位能力

### 损失函数 / 训练策略
对称对比学习损失（InfoNCE），冻结视觉骨干训练触觉编码器和两个对齐器模块。

## 实验关键数据

### 主实验

| 数据集 | 指标 | STT | 之前最佳 | 提升 |
|--------|------|-----|---------|------|
| TG-Seg (新) | mIoU | 显著优 | ImageBind | 大幅 |
| Web-Mat-Seg (新) | mIoU | 显著优 | UniTouch | 大幅 |
| OpenSurfaces | F1 | 优 | 基线 | 提升 |

### 消融实验

| 配置 | mIoU | 说明 |
|------|------|------|
| Full (域内+域外) | 最优 | 完整模型 |
| 仅域内配对 | 次优 | 缺少场景级泛化 |
| 仅标准配对 | 差 | 有效训练对太少 |
| 全局对齐替代 | 差 | 无空间定位能力 |

### 关键发现
- 局部对齐显著优于全局对齐，证明空间分辨的跨模态特征是定位的关键
- 材质多样性配对（尤其域外图像）是泛化到场景级图像的关键因素
- 弱触觉信号（如轻触或不确定材质）的处理能力因域外数据增加而显著提升

## 亮点与洞察
- **新任务定义**：触觉定位是一个自然但未被正式研究的问题，可启发更多感官交互研究
- **"相似材质相似触觉"的利用**：简单假设大幅扩展训练数据，是跨模态学习中解决数据稀缺的通用策略

## 局限与展望
- 触觉传感器类型固定（GelSight），跨传感器泛化未验证
- 材质类别粒度有限（18 类）
- 未来可扩展到更细粒度的材质属性（如粗糙度、硬度）

## 相关工作与启发
- **vs ImageBind/UniTouch**: 全局对齐方法，无法定位
- **vs TaRF**: 在 3D NeRF 中做触觉定位，限于重建场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 任务定义新颖，数据策略实用
- 实验充分度: ⭐⭐⭐⭐ 构建了两个新数据集进行评估
- 写作质量: ⭐⭐⭐⭐ 动机和方法描述清晰
- 价值: ⭐⭐⭐⭐ 开辟了触觉定位的新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] SToLa: Self-Adaptive Touch-Language Framework with Tactile Commonsense Reasoning in Open-Ended Scenarios](../../AAAI2026/multimodal_vlm/stola_self-adaptive_touch-language_framework_with_tactile_commonsense_reasoning_.md)
- [\[CVPR 2026\] PinPoint: Focus, Don't Prune — Identifying Instruction-Relevant Regions for Information-Rich Image Understanding](focus_dont_prune_identifying_instruction-relevant_regions_for_information-rich_i.md)
- [\[CVPR 2026\] VLM-Loc: Localization in Point Cloud Maps via Vision-Language Models](vlm-loc_localization_in_point_cloud_maps_via_vision-language_models.md)
- [\[CVPR 2026\] EmoVerse: A MLLMs-Driven Emotion Representation Dataset for Interpretable Visual Emotion Analysis](emoverse_a_mllms-driven_emotion_representation_dataset_for_interpretable_visual_.md)
- [\[CVPR 2026\] Generate, Analyze, and Refine: Training-Free Sound Source Localization via MLLM Meta-Reasoning](generate_analyze_and_refine_training-free_sound_source_localization_via_mllm_met.md)

</div>

<!-- RELATED:END -->
