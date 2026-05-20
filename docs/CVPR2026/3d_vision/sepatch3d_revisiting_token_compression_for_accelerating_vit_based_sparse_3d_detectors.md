---
title: >-
  [论文解读] SEPatch3D: Revisiting Token Compression for Accelerating ViT-based Sparse Multi-View 3D Object Detectors
description: >-
  [CVPR 2026][3D视觉][目标检测] 提出 SEPatch3D，通过时空感知的动态 patch 大小选择和基于熵的信息性 patch 筛选增强机制，在 ViT 基稀疏多视角 3D 检测中实现 57% 推理加速且保持可比检测精度。
tags:
  - "CVPR 2026"
  - "3D视觉"
  - "目标检测"
  - "token compression"
  - "patch size selection"
  - "multi-view detection"
  - "ViT acceleration"
---

# SEPatch3D: Revisiting Token Compression for Accelerating ViT-based Sparse Multi-View 3D Object Detectors

**会议**: CVPR 2026  
**arXiv**: [2604.14563](https://arxiv.org/abs/2604.14563)  
**代码**: [github.com/Mingqj/SEPatch3D](https://github.com/Mingqj/SEPatch3D)  
**领域**: 3D视觉  
**关键词**: 3D object detection, token compression, patch size selection, multi-view detection, ViT acceleration

## 一句话总结

提出 SEPatch3D，通过时空感知的动态 patch 大小选择和基于熵的信息性 patch 筛选增强机制，在 ViT 基稀疏多视角 3D 检测中实现 57% 推理加速且保持可比检测精度。

## 研究背景与动机

ViT 基稀疏查询式多视角 3D 检测器（如 StreamPETR）性能优异但推理延迟高。现有 token 压缩策略的局限：(1) token 剪枝可能丢弃对困难负样本学习至关重要的信息性背景区域；(2) token 合并的不规则聚合破坏上下文一致性；(3) 简单增大 patch 大小超过阈值（如>18）会因丢失细粒度语义线索而性能下降。核心观察：增大 patch 可降低计算但需同时保留语义重要区域的细粒度信息。

## 方法详解

### 整体框架

两阶段策略：(1) 动态双 patch 嵌入——SPSS 模块根据时空线索自适应选择 patch 大小；(2) 选择性跨粒度特征增强——IPS 模块筛选信息性 patch，CGFE 模块用细粒度 patch 增强粗粒度 patch。

### 关键设计

1. **时空感知 Patch 大小选择 (SPSS)**: 利用前一帧对象查询的平均深度 $\bar{D}^{T-1}$ 和深度趋势斜率变化 $\Delta S^{T-1}$ 动态决定当前帧 patch 大小。远距物体+趋势远离→大 patch 减少计算；近距物体+趋势靠近→小 patch 保留细节；否则保持前帧设置确保时间稳定性。

2. **基于熵的信息性 Patch 选择 (IPS)**: 通过运动对齐的历史查询进行交叉注意力增强 patch 特征后，计算 L2 归一化特征的信息熵。熵值超过场景均值的 patch 被选为信息性区域，采用自适应阈值而非固定 Top-K 以适应不同场景复杂度。

3. **跨粒度特征增强 (CGFE)**: 被选中的粗粒度 patch 作为 query，对应区域的原始细粒度 patch 作为 key/value，通过位置编码增强的交叉注意力注入细节信息，残差连接保留全局结构。

### 损失函数 / 训练策略

继承 StreamPETR 的检测损失。双 patch 嵌入中原始 16×16 小 patch 提供细粒度特征参考，灵活大 patch 用于高效推理。端到端训练。

## 实验关键数据

### 主实验

| 方法 | 骨干 | NDS(%) | mAP(%) | 推理时间 |
|------|------|--------|--------|---------|
| StreamPETR (patch=16) | ViT | 基线 | 基线 | 基线 |
| ToC3D-faster | ViT | 略低 | 略低 | 加速 |
| SEPatch3D-faster | ViT | **可比** | **可比** | **-57%** |

在 nuScenes 上推理加速 57%，性能下降不到 1 点；比 ToC3D-faster 额外快 20%。Argoverse 2 上同样验证有效。

### 消融实验

- SPSS 的深度-趋势联合决策优于仅用深度或仅用趋势
- 自适应熵阈值优于固定 Top-K 选择
- CGFE 的跨粒度增强对保持检测精度至关重要

### 关键发现

- patch 大小增大到 18 以上性能开始下降，但通过选择性增强可延续加速收益
- 信息性 patch 往往对应纹理丰富或边缘区域，正是粗 patch 损失最大的地方
- 时空感知选择有效避免了连续帧间 patch 大小的突变

## 亮点与洞察

- "增大 patch + 选择性增强"比"剪枝/合并"更适合 3D 检测的思路新颖
- 利用检测查询的时空信息指导 backbone 压缩策略的跨层交互设计巧妙
- 不同于 ToC3D 的前景导向剪枝，保留了对困难负样本学习有价值的背景信息

## 局限与展望

- 预定义的大小 patch 对集合（$P_s$, $P_l$）和深度阈值 $\theta$ 需要手动设定
- 细粒度 patch 始终需要计算（虽然不全程参与 ViT 块）
- 仅在 StreamPETR 基线上验证，对其他稀疏检测器的泛化性未测试

## 相关工作与启发

- 动态 patch 大小选择思路可推广到其他需要效率-精度平衡的 ViT 应用
- 时空查询指导 backbone 计算的范式突破了传统单向信息流
- 跨粒度特征增强可用于多尺度表示学习

## 评分

7/10 — 动机清晰、方法实用、加速效果显著，在自动驾驶场景有实际价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Fast SceneScript: Fast and Accurate Language-Based 3D Scene Understanding via Multi-Token Prediction](fast_scenescript_fast_and_accurate_language-based_3d_scene_understanding_via_mul.md)
- [\[CVPR 2026\] Learning Multi-View Spatial Reasoning from Cross-View Relations](learning_multi-view_spatial_reasoning_from_cross-view_relations.md)
- [\[ECCV 2024\] MVDiffusion++: A Dense High-Resolution Multi-View Diffusion Model for Single or Sparse-View 3D Object Reconstruction](../../ECCV2024/3d_vision/mvdiffusion_a_dense_high-resolution_multi-view_diffusion_model_for_single_or_spa.md)
- [\[CVPR 2026\] Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass](coherent_humanscene_reconstruction_from_multiperso.md)
- [\[CVPR 2026\] LTGS: Long-Term Gaussian Scene Chronology From Sparse View Updates](ltgs_long-term_gaussian_scene_chronology_from_sparse_view_updates.md)

</div>

<!-- RELATED:END -->
