---
title: >-
  [论文解读] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions
description: >-
  [CVPR 2025][3D视觉][人物交互重建] 提出首个开放词汇野外3D人物交互(HOI)数据集 Open3DHOI（2.5k+图像，133类物体，120类动作），并设计基于3D Gaussian Splatting的HOI优化器，通过Gaussian渲染实现人物空间交互重建和接触区域学习。
tags:
  - CVPR 2025
  - 3D视觉
  - 人物交互重建
  - 开放词汇
  - 3D高斯溅射
  - 接触区域估计
  - HOI数据集
---

# Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions

**会议**: CVPR 2025  
**arXiv**: [2503.15898](https://arxiv.org/abs/2503.15898)  
**代码**: [GitHub](https://wenboran2002.github.io/3dhoi/)  
**领域**: 3D视觉  
**关键词**: 人物交互重建, 开放词汇, 3D高斯溅射, 接触区域估计, HOI数据集

## 一句话总结

提出首个开放词汇野外3D人物交互(HOI)数据集 Open3DHOI（2.5k+图像，133类物体，120类动作），并设计基于3D Gaussian Splatting的HOI优化器，通过Gaussian渲染实现人物空间交互重建和接触区域学习。

## 研究背景与动机

- 从单张图像重建3D人物交互（HOI）是计算机视觉的基础问题，但受限于3D数据的匮乏
- 现有3D HOI数据集（如BEHAVE、InterCap）主要在室内固定环境录制，物体类别极少（8-40类），远不及2D HOI数据集的丰富度
- WildHOI和3DIR虽使用野外图像，但仅包含少量物体类别且使用不真实的CAD模型
- 2D HOI数据集（如HICO-DET、HAKE）提供了丰富的2D标注和多样化的物体类别
- 单图像3D重建技术（如InstantMesh）的成熟使得从2D HOI图像重建3D资产成为可能
- 现有训练free的3D HOI重建方法（如PHOSA）仅使用轮廓损失优化物体位姿，性能受限
- 基于训练的方法虽然对特定物体类别效果好，但难以泛化到开放世界环境
- 缺乏统一评估3D交互质量的度量标准

## 方法详解

### 整体框架

系统分为**数据标注流水线**和**3D HOI重建优化器**两部分。标注流水线从2D HOI数据集（HAKE + SWIG-HOI）中选取15k+图像，用InstantMesh重建物体、OSX重建人体，经粗重建（深度投影对齐）和精标注（Blender + 网页工具）后获得2.5k+高质量3D HOI标注。重建优化器Gaussian-HOI基于3D Gaussian Splatting，同时优化人体SMPL-X参数和物体6D位姿，并通过Gaussian属性学习接触区域。

### 关键设计

**设计一：粗到精的3D HOI标注流水线**
- **功能**：从单视角图像高效获取高质量3D HOI标注
- **核心思路**：先用遮挡补全+Stable Diffusion修复被遮挡的物体区域，再用单目深度估计+掩码提取生成人和物体的深度点云，通过点云匹配获得粗重建。然后经过Filtering Tool（评估SMPL-X和物体重建质量、标注接触区域）和3D Interaction Tool（Blender粗调+网页工具精调）完成精标注
- **设计动机**：直接从2D标注和现有重建工具出发，避免了多视角RGBD采集的高成本；粗到精策略大幅降低人工标注量

**设计二：HOI-Gaussian优化器**
- **功能**：从单张图像无训练地重建3D人物交互
- **核心思路**：用SMPL-X顶点初始化人体Gaussian $g_h$，物体网格顶点初始化物体Gaussian $g_o$，通过可学习参数 $W_{obj}$ 优化物体6D位姿。合成交互Gaussian $g_{hoi} = g_h \oplus g_o$，利用Gaussian渲染损失实现2D对齐，结合碰撞、深度和接触损失优化3D空间关系
- **设计动机**：相比PHOSA仅用轮廓损失，3D Gaussian利用颜色匹配和深度信息，减少轮廓相似但实际偏差大的情况

**设计三：基于Gaussian的接触区域学习**
- **功能**：自动识别人体与物体的潜在接触区域
- **核心思路**：利用Gaussian渲染中遮挡区域的不透明度 $\alpha$ 自然降低的特性。先通过法线方向设置背面点低不透明度，优化过程中被遮挡区域的 $\alpha$ 下降，结合Chamfer距离约束，计算接触分数 $c = w_\alpha \cdot \text{Norm}(\alpha^h) + w_d \cdot d_C(p^h, p^o)^h$
- **设计动机**：单目图像难以直接确定接触区域，但可通过Gaussian渲染的不透明度变化间接推断人体被物体遮挡的部分即为潜在接触区域

### 损失函数

总损失 $\mathcal{L} = w_r \cdot \mathcal{L}_r + w_{hoi} \cdot \mathcal{L}_{hoi}$：
- 渲染损失 $\mathcal{L}_r$：分别对 $g_{hoi}$、$g_h$、$g_o$ 计算L1 + L2 mask + SSIM + LPIPS
- HOI损失 $\mathcal{L}_{hoi} = \mathcal{L}_{cont} + \mathcal{L}_{colli} + \mathcal{L}_{depth}$：接触损失（人体接触区域与物体的Chamfer距离）+ 碰撞损失 + 序数深度损失

## 实验关键数据

### 主实验：物体位姿重建比较

| 方法 | Scale↓ | Translation(cm)↓ | Rotation↓ | Chamfer Dist.(cm) |
|------|--------|-------------------|-----------|-------------------|
| PHOSA | 0.39 | 77.79 | 0.95 | 49.1 |
| Ours w/o HOI Loss | 0.25 | 38.66 | 0.45 | 16.9 |
| **Ours** | **0.16** | **38.44** | **0.41** | 19.3 |

### 消融实验：碰撞-接触评估（$Co^2$ 指标）

| 方法 | $Co^2$↓ | Collision↓ | Contact↓ |
|------|---------|------------|----------|
| PHOSA | 0.431 | 0.105 | 0.326 |
| Coarse Recon | 0.248 | 0.083 | 0.165 |
| Gs only | 0.287 | 0.136 | 0.151 |
| Gs & depth & colli | 0.188 | 0.045 | 0.143 |
| **Gs & depth & colli & cont** | **0.181** | 0.053 | **0.128** |

### 关键发现
- Gaussian-HOI在所有物体位姿指标上大幅超越PHOSA（Translation误差降低50%+）
- 仅用Gaussian优化不足以改善3D交互质量，需要HOI损失配合
- 接触损失在降低Contact score的同时可能略微增加碰撞（物体被拉向接触区域），$Co^2$ 指标平衡了这一trade-off
- PointLLM对3D HOI理解能力有限，给定物体名称显著提升动作推理（Top-1从20%到47%）

## 亮点与洞察

1. **首个开放词汇野外3D HOI数据集**：133类物体 + 120类动作，远超BEHAVE（10类物体）和3DIR（21类）
2. **Gaussian渲染的创新应用**：不仅用于2D对齐优化，还巧妙利用不透明度属性推断接触区域
3. **设计了 $Co^2$ 评估指标**：统一评估碰撞和接触质量，填补了3D HOI重建评估的空白
4. **标注方法的可扩展性**：流水线可利用未来更强的3D-AIGC工具提升效率

## 局限与展望

- 数据集规模（2.5k+）对于训练大规模模型仍不足
- 物体重建质量依赖InstantMesh，精细交互区域（如手指抓持）仍难以准确重建
- 仅标注单人场景，多人交互尚未覆盖
- PointLLM等通用3D理解模型在HOI任务上表现差，需要更多细粒度数据驱动改进
- 未来可结合多视角输入或视频序列进一步提升重建质量

## 相关工作与启发

- 与PHOSA等基于轮廓的优化方法相比，3D Gaussian能利用更丰富的颜色和深度信息
- GauHuman证明了3D Gaussian可以调整人体参数，本文将其扩展到人物交互场景
- $Co^2$ 指标设计思路（平衡碰撞和接触）可推广到其他需要评估物理合理性的任务

## 评分

⭐⭐⭐⭐ — 数据集构建有重要价值，Gaussian-HOI优化器设计合理；但作为测试集规模有限，重建方法在复杂交互上仍有提升空间。

<!-- RELATED:START -->

## 相关论文

- [Reconstructing Animals and the Wild](reconstructing_animals_and_the_wild.md)
- [HOI3DGen: Generating High-Quality Human-Object-Interactions in 3D](hoi3dgen_generating_high-quality_human-object-interactions_in_3d.md)
- [SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding](seeground_see_and_ground_for_zero-shot_open-vocabulary_3d_visual_grounding.md)
- [Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces](open-vocabulary_functional_3d_scene_graphs_for_real-world_indoor_spaces.md)
- [GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance Grounding](great_geometry-intention_collaborative_inference_for_open-vocabulary_3d_object_a.md)

<!-- RELATED:END -->
