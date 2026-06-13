---
title: >-
  [论文解读] Open-World Amodal Appearance Completion
description: >-
  [CVPR 2025][3D视觉][开放世界物体补全] 提出一种免训练的开放世界 amodal 外观补全框架，接受灵活的自然语言查询（包括直接名称和抽象描述），通过统一分割、遮挡分析和迭代 inpainting 重建被遮挡物体的完整外观，输出 RGBA 格式支持 3D 重建和图像编辑等下游应用。 Amodal complet…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "开放世界物体补全"
  - "遮挡推理"
  - "免训练框架"
  - "语言引导"
  - "RGBA输出"
---

# Open-World Amodal Appearance Completion

**会议**: CVPR 2025  
**arXiv**: [2411.13019](https://arxiv.org/abs/2411.13019)  
**代码**: 无  
**领域**: 3D Vision / Amodal Completion  
**关键词**: 开放世界物体补全, 遮挡推理, 免训练框架, 语言引导, RGBA输出

## 一句话总结

提出一种免训练的开放世界 amodal 外观补全框架，接受灵活的自然语言查询（包括直接名称和抽象描述），通过统一分割、遮挡分析和迭代 inpainting 重建被遮挡物体的完整外观，输出 RGBA 格式支持 3D 重建和图像编辑等下游应用。

## 研究背景与动机

Amodal completion（非模态补全）旨在推断并重建物体被遮挡的部分，在 AR、3D 重建和内容创作中至关重要。然而现有方法存在严重局限：(1) **封闭类别**——PD-MC 等方法依赖预定义物体类别，遇到未见类别时失效；(2) **需要训练数据**——Pix2gestalt 依赖大量监督数据训练；(3) **不支持语言交互**——用户无法通过自然语言指定目标物体。

开放世界场景中，物体类别多样且不可预测，遮挡关系复杂（包括模糊背景遮挡）。需要一种**免训练、支持自然语言指定任意物体、能处理复杂遮挡**的通用框架。

本文引入"推理式 amodal completion"概念：系统根据图像和语言查询推断并重建被查询物体的完整外观，支持具体词汇（如"polar bear"）和抽象查询（如"What is the mammal in this image"）。

## 方法详解

### 整体框架

Pipeline 分四步：(1) **文本查询解析与分割**：用 VLM（LISA）根据文本查询生成可见区域 mask $M_{\text{visible}}$，同时用自动标注系统分割所有物体和背景；(2) **遮挡分析**：用 InstaOrderNet 判断哪些 segment 遮挡目标，生成遮挡 mask $M_{\text{occ}}$；(3) **提示词生成**：通过 CLIP 匹配选择最佳 inpainting 提示；(4) **迭代 inpainting**：使用预训练 inpainting 模型逐步重建遮挡区域，输出 RGBA。

### 关键设计

**1. 开放世界分割与背景处理**

- **功能**: 识别场景中所有可能的遮挡物，包括难以识别的背景区域
- **核心思路**: 先用开集标签模型 + 开集检测器 + SAM 分割所有可命名物体得到集合 $S$，再对未分割区域 $B = I - \bigcup S_i$ 通过形态学操作（腐蚀+膨胀）分割成独立的背景段 $\{B_1, \ldots, B_k\}$，其中 $B_j = \text{Morph}(I - \bigcup_{i=1}^{m} S_i)$
- **设计动机**: 传统分割忽略无法用类别标签描述的背景区域（灌木、地面等），但这些区域可能遮挡目标物体。背景分割确保所有潜在遮挡物都被考虑

**2. 遮挡分析与边界感知**

- **功能**: 确定哪些 segment 遮挡了目标物体，生成遮挡 mask 引导 inpainting
- **核心思路**: 使用 InstaOrderNet 对每个 segment（包括背景段）与目标物体做成对遮挡顺序判断，合并所有遮挡 segment 为 $M_{\text{occ}} = \bigcup_{occ_i=1} S_i \cup \bigcup_{occ_j=1} B_j$。对触及图像边界的情况，通过膨胀操作扩展遮挡 mask：$M_{\text{occ}} \leftarrow M_{\text{occ}} \cup (d(M_{\text{visible}}) \cap \bigcup_{e \in E} \text{edge}_e)$
- **设计动机**: 遮挡关系不能简单由空间位置判断，需要专门的遮挡顺序推理模型。边界感知处理目标物体超出图像范围的情况

**3. CLIP 引导提示选择与迭代 Inpainting**

- **功能**: 自动生成最佳 inpainting 提示词，迭代重建遮挡区域
- **核心思路**: 将目标可见区域与所有候选标签 $T \cup Q$ 通过 CLIP 匹配选最佳提示 $P = \arg\max_{t_i} \text{CLIP}(I_{\text{target}}, t_i)$。Inpainting 迭代执行：$I_{\text{inpaint}}^{(t+1)} = \phi(I_{\text{inpaint}}^{(t)}, M_{\text{occ}}^{(t)}, P)$，每步更新遮挡 mask 和 amodal mask，当 $\Delta M_{\text{occ}}^{(t)} < \epsilon$ 或达最大迭代时终止。最终通过 alpha blending 融合原始可见区域和重建区域
- **设计动机**: 用户查询可能是抽象描述，CLIP 匹配确保提示词与目标物体视觉属性对齐。迭代 inpainting 逐步扩展重建区域，处理复杂遮挡

### 损失函数 / 训练策略

- **免训练框架**：不需要额外训练，利用预训练模型（LISA、SAM、InstaOrderNet、CLIP、inpainting 模型）的组合
- 迭代终止条件：遮挡 mask 变化低于阈值 $\epsilon$ 或达最大迭代 $T$
- Alpha blending 确保原始可见区域像素不被修改

## 实验关键数据

### 主实验

在 2565 个实例、553 个类别的评估数据集上的对比：

| 方法 | CLIP↑ | LPIPS↓ | Feature Sim.↑ | SSIM↑ |
|------|-------|--------|---------------|-------|
| PD w/o MC | 24.553 | 0.614 | 0.404 | 0.395 |
| PD-MC | 27.984 | 0.628 | 0.364 | 0.413 |
| Pix2gestalt | 27.417 | 0.442 | 0.548 | 0.714 |
| **Ours** | **28.181** | **0.320** | **0.646** | **0.731** |

### 消融实验

提示词变体和背景分割的影响：

| 配置 | CLIP↑ | LPIPS↓ | Feature Sim.↑ | SSIM↑ |
|------|-------|--------|---------------|-------|
| Q only | 28.563 | 0.327 | 0.633 | 0.724 |
| T only | 28.043 | 0.324 | 0.636 | 0.725 |
| T∪Q w/o bg seg. | 28.071 | 0.333 | 0.620 | 0.713 |
| **T∪Q w/ bg seg.** | **28.181** | **0.320** | **0.646** | **0.731** |

### 关键发现

1. **背景分割对复杂遮挡至关重要**：添加背景段后 SSIM 从 0.713→0.731，LPIPS 从 0.333→0.320
2. **在开放世界中显著优于封闭类别方法**：PD-MC 遇到未见类别时完全失效，Pix2gestalt 有时仅对输入做最小改动
3. **组合提示 $T \cup Q$ 在实际场景中最有效**：虽然 Q-only 的 CLIP 分数最高（因直接匹配标签），但对抽象查询不适用
4. 人类偏好研究显示本方法在重建质量上获得最高偏好率

## 亮点与洞察

1. **"推理式 amodal completion"概念具有前瞻性**：允许抽象查询（如"图中的哺乳动物是什么"）进行物体补全，人机交互更自然
2. **免训练 pipeline 的模块化设计**优秀：各模块可独立升级（更好的分割/inpainting 模型），框架保持不变
3. **RGBA 输出格式**对下游应用友好：可直接用于图像编辑、3D 重建、AR 场景

## 局限与展望

1. 依赖预训练图像生成模型，可能引入伪影（如动物姿态不匹配）
2. 对于严重遮挡或模糊查询，分割准确性可能不足
3. 缺少 ground-truth amodal 数据的场景下评估指标有限
4. 可探索将遮挡推理与生成过程端到端联合优化

## 相关工作与启发

- **PD-MC**: 利用预训练模型但受限于预定义类别，本文去除类别约束
- **Pix2gestalt**: 监督学习方法，需大量训练数据且泛化有限
- **LISA**: 语言引导分割模型，本文利用其进行初始可见 mask 生成
- **InstaOrderNet**: 遮挡顺序推理模型，使本文能判断哪些区域遮挡目标

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将 amodal completion 扩展到开放世界+语言引导设置
- **实验充分度**: ⭐⭐⭐ — 有量化对比和人类评估，但评估数据集较小，缺少 ground truth
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，pipeline 描述详尽
- **价值**: ⭐⭐⭐⭐ — 对 AR/3D 重建有实用价值，模块化免训练设计易于部署

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos](depthcrafter_generating_consistent_long_depth_sequences_for_open-world_videos.md)
- [\[CVPR 2025\] Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces](open-vocabulary_functional_3d_scene_graphs_for_real-world_indoor_spaces.md)
- [\[NeurIPS 2025\] Towards 3D Objectness Learning in an Open World](../../NeurIPS2025/3d_vision/towards_3d_objectness_learning_in_an_open_world.md)
- [\[ICCV 2025\] Contact-Aware Amodal Completion for Human-Object Interaction via Multi-Regional Inpainting](../../ICCV2025/3d_vision/contact-aware_amodal_completion_for_human-object_interaction_via_multi-regional_.md)
- [\[ICCV 2025\] Amodal Depth Anything: Amodal Depth Estimation in the Wild](../../ICCV2025/3d_vision/amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)

</div>

<!-- RELATED:END -->
