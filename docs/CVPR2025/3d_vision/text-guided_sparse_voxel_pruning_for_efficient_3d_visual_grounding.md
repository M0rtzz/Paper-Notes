---
title: >-
  [论文解读] Text-Guided Sparse Voxel Pruning for Efficient 3D Visual Grounding
description: >-
  [CVPR 2025][3D视觉][3D视觉定位] 提出TSP3D，首个基于多层稀疏卷积架构的单阶段3D视觉定位框架，通过文本引导剪枝（TGP）和基于补全的添加（CBA）实现高效的3D-文本交互，在ScanRefer上以12.43 FPS的速度取得46.71% Acc@0.5的SOTA精度。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D视觉定位
  - 稀疏卷积
  - 文本引导剪枝
  - 多模态融合
  - 实时推理
---

# Text-Guided Sparse Voxel Pruning for Efficient 3D Visual Grounding

**会议**: CVPR 2025  
**arXiv**: [2502.10392](https://arxiv.org/abs/2502.10392)  
**代码**: [https://github.com/GWxuan/TSP3D](https://github.com/GWxuan/TSP3D)  
**领域**: 3D视觉  
**关键词**: 3D视觉定位, 稀疏卷积, 文本引导剪枝, 多模态融合, 实时推理

## 一句话总结

提出TSP3D，首个基于多层稀疏卷积架构的单阶段3D视觉定位框架，通过文本引导剪枝（TGP）和基于补全的添加（CBA）实现高效的3D-文本交互，在ScanRefer上以12.43 FPS的速度取得46.71% Acc@0.5的SOTA精度。

## 研究背景与动机

3D视觉定位（3DVG）任务要求根据自然语言描述在3D场景中定位目标物体，是机器人和AR/VR等领域的基础多模态感知任务。现有方法面临两大挑战：

1. **两阶段方法速度慢**：先检测再匹配的范式导致冗余特征计算，推理速度通常低于3 FPS，难以满足实时需求
2. **单阶段方法精度不足**：基于点云的架构（如PointNet++）依赖FPS和kNN等耗时操作，且需要激进的下采样，损失细粒度几何信息，速度仍低于6 FPS

作者发现多层稀疏卷积架构在3D目标检测中已取得领先的速度和精度，但直接迁移到3DVG面临核心难题：3DVG需要3D场景表示与文本特征的深度交互，而稀疏卷积架构中体素数量巨大（随上采样指数增长），使得跨模态注意力计算代价不可承受。

因此，如何在保持稀疏卷积高效性的同时实现深度多模态交互，是本文要解决的核心问题。

## 方法详解

### 整体框架

TSP3D基于三层稀疏卷积架构构建单阶段3DVG流程：输入点云经体素化后通过三个MinkResBlock提取多层体素特征 $V_l$（$l=1,2,3$），同时用RoBERTa编码文本特征 $T$。核心创新在于上采样阶段引入TGP和CBA模块，逐层融合3D与文本特征后通过卷积头预测目标边界框。

### 关键设计

**设计一：文本引导剪枝（Text-Guided Pruning, TGP）**

- **功能**：在保持高效推理的同时实现深度跨模态交互
- **核心思路**：在每层上采样前，利用文本信息指导体素剪枝，逐步聚焦目标区域。配置两级TGP：场景级TGP（level 3→2）区分前景/背景，目标级TGP（level 2→1）聚焦文本提及的目标和参考物体
- **设计动机**：稀疏卷积中体素数量巨大，直接交叉注意力计算不可行。通过先剪枝再交互，将体素数量减少至原始的约7%，使注意力计算变得可行

具体地，TGP对体素特征进行FPS采样后，与文本特征做交叉注意力并预测保留概率 $\hat{M}$，通过Heaviside阶跃函数二值化得到剪枝掩码：

$$U_l^P = U_l \odot \Theta(\mathcal{I}(\hat{M}, U_l) - \sigma)$$

简化版本去掉FPS，将特征增强和剪枝预测合并为一次交互，并移至剪枝之前执行，进一步降低计算开销。

**设计二：基于补全的添加（Completion-Based Addition, CBA）**

- **功能**：修复剪枝过程中误删的目标区域几何信息
- **核心思路**：利用文本特征查询backbone特征中可能属于目标的体素，与剪枝后的上采样特征互补。先通过交叉注意力增强backbone特征 $V_l'$，再预测目标区域掩码 $M_l^{tar}$，找出被误剪的位置 $M_l^{mis}$，通过插值补全
- **设计动机**：剪枝不可避免会误删小物体或窄物体的表示。完全添加计算量大且破坏聚焦效果，纯剪枝感知添加则过度依赖剪枝结果。CBA以极小的计算开销精准补全关键区域

$$M_l^{mis} = M_l^{tar} \wedge (\neg \mathcal{C}(U_l^G, V_l))$$

**设计三：多级监督策略**

- **功能**：为TGP和CBA提供精确的训练信号
- **核心思路**：场景级TGP的监督基于所有物体中心生成 $L \times L \times L$ 立方体掩码；目标级TGP基于目标和文本提及的参考物体生成掩码；CBA的补全监督与分类共享，以目标物体中心附近的体素为正样本
- **设计动机**：不同层级的剪枝目标不同，场景级需保留所有物体区域，目标级需聚焦语义相关区域

### 损失函数

总损失为四部分之和：

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{pruning} + \lambda_2 \mathcal{L}_{com} + \lambda_3 \mathcal{L}_{class} + \lambda_4 \mathcal{L}_{bbox}$$

其中剪枝损失、补全损失和分类损失使用Focal Loss处理类别不平衡，边界框回归使用DIoU损失。所有权重 $\lambda$ 均设为1。

## 实验关键数据

### 主实验：ScanRefer数据集

| 方法 | 类型 | Acc@0.25 | Acc@0.5 | FPS |
|------|------|----------|---------|-----|
| MCLN (ECCV'24) | Two-stage | 57.17 | 45.53 | 3.17 |
| G³-LQ (CVPR'24) | Two-stage | 56.90 | 45.58 | - |
| EDA (CVPR'23) | Single-stage | 53.83 | 41.70 | 5.98 |
| MCLN (ECCV'24) | Single-stage | 54.30 | 42.64 | 5.45 |
| **TSP3D (Ours)** | **Single-stage** | **56.45** | **46.71** | **12.43** |

### NR3D/SR3D数据集

| 方法 | 类型 | NR3D | SR3D |
|------|------|------|------|
| MCLN (ECCV'24) | Two-stage (det) | 46.1 | 53.9 |
| MCLN (ECCV'24) | Single-stage | 45.7 | 53.4 |
| **TSP3D (Ours)** | **Single-stage** | **48.7** | **57.1** |

### 消融实验

| 组件 | Acc@0.25 | Acc@0.5 | FPS |
|------|----------|---------|-----|
| TSP3D-B (基线) | 44.87 | 29.05 | 14.58 |
| + TGP | 54.63 | 44.91 | 12.84 |
| + TGP + CBA (完整TSP3D) | 56.45 | 46.71 | 12.43 |

### 关键发现

1. TSP3D是首个超过10 FPS的3DVG方法，比此前最快的单阶段方法提速100%以上
2. TGP将体素数量减少至原始的~7%，同时大幅提升精度（+15.86 Acc@0.5）
3. CBA以极小的速度代价（-0.41 FPS）带来+1.80 Acc@0.5的提升
4. 作为单阶段方法，首次在精度上超越所有两阶段方法

## 亮点与洞察

1. **稀疏卷积+剪枝的范式创新**：首次将多层稀疏卷积架构引入3DVG，通过"先剪枝再交互"的策略巧妙解决了体素数量过大导致注意力不可行的问题
2. **渐进式聚焦机制**：场景级→目标级的两级剪枝设计，模拟了人类"先看全景再看细节"的视觉搜索过程
3. **补全与剪枝的互补设计**：CBA不是简单地恢复所有被剪区域，而是有选择地补全可能属于目标的区域，兼顾了效率和准确性

## 局限与展望

1. 剪枝阈值 $\sigma$ 和补全阈值 $\tau$ 需要手动调节，不同场景可能需要不同设置
2. 文本编码器使用固定的RoBERTa，未探索更强的语言模型（如LLM）对性能的影响
3. 未在更大规模的3D场景（如室外场景）上验证泛化能力
4. 补全模块依赖backbone特征质量，当初始特征提取不佳时补全效果可能受限

## 相关工作与启发

- **FCAF3D/TR3D**：多层稀疏卷积在3D检测中的成功经验，直接启发了TSP3D的架构设计
- **DSPDet3D**：稀疏卷积处理小物体的经验，与CBA的补全思想相呼应
- **BUTD-DETR/EDA**：单阶段3DVG的先驱，但受限于点云架构的效率瓶颈
- 启发：在其他需要大规模特征交互的任务中，"先剪枝再交互"的策略可能同样有效

## 评分

⭐⭐⭐⭐ — 首次在3DVG中实现实时推理（12.43 FPS），且精度全面超越现有方法。"文本引导剪枝"的设计简洁高效，是将稀疏卷积扩展到多模态任务的优秀范例。不足在于实验仅限室内场景数据集。

<!-- RELATED:START -->

## 相关论文

- [ProxyTransformation: Preshaping Point Cloud Manifold with Proxy Attention for 3D Visual Grounding](proxytransformation_preshaping_point_cloud_manifold_with_proxy_attention_for_3d_.md)
- [SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding](seeground_see_and_ground_for_zero-shot_open-vocabulary_3d_visual_grounding.md)
- [ShapeShifter: 3D Variations Using Multiscale and Sparse Point-Voxel Diffusion](shapeshifter_3d_variations_using_multiscale_and_sparse_point-voxel_diffusion.md)
- [Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions](grounding_3d_object_affordance_with_language_instructions_visual_observations_an.md)
- [MoST: Efficient Monarch Sparse Tuning for 3D Representation Learning](most_efficient_monarch_sparse_tuning_for_3d_representation_learning.md)

<!-- RELATED:END -->
