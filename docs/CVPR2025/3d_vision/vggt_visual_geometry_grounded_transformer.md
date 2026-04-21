---
title: >-
  [论文解读] VGGT: Visual Geometry Grounded Transformer
description: >-
  [CVPR 2025][3D视觉][3D重建] VGGT是一个大型前馈Transformer，能在不到一秒内从一张到数百张图像中直接预测相机参数、深度图、点云图和3D点轨迹，无需后处理优化即可超越现有方法。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D重建
  - Transformer
  - 多任务学习
  - 深度估计
  - 点云跟踪
---

# VGGT: Visual Geometry Grounded Transformer

**会议**: CVPR 2025  
**arXiv**: [2503.11651](https://arxiv.org/abs/2503.11651)  
**代码**: [github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt)  
**领域**: 3D Vision  
**关键词**: 3D重建, 前馈式Transformer, 多任务学习, 深度估计, 点云跟踪

## 一句话总结

VGGT是一个大型前馈Transformer，能在不到一秒内从一张到数百张图像中直接预测相机参数、深度图、点云图和3D点轨迹，无需后处理优化即可超越现有方法。

## 研究背景与动机

传统3D重建依赖视觉几何方法和迭代优化（如Bundle Adjustment），计算代价高昂且流程复杂。虽然DUSt3R/MASt3R等近期方法取得了进展，但它们只能处理两张图像，处理多张图像时仍需后处理来融合成对重建结果。

本文提出的核心问题是：**能否用一个纯前馈神经网络直接完成3D重建的所有任务？** 作者认为，随着网络能力的增强和大规模3D标注数据的积累，无需几何后处理的端到端方法已经可行。

现有方法的主要局限：
- DUSt3R/MASt3R需要10+秒的全局对齐后处理
- 已有的多视图3D方法通常只专注于单一任务（如单目深度估计或新视角合成）
- 没有一个统一模型能同时高质量地预测所有关键3D属性

VGGT的目标是构建一个类似NLP中GPT、视觉中DINO的**通用3D基础模型**，能作为backbone增强下游任务。

## 方法详解

### 整体框架

VGGT采用标准大型Transformer架构（约12亿参数），以DINO作为图像编码器将输入图像转化为token，然后通过**交替注意力（Alternating-Attention）**机制处理，最后通过不同预测头输出相机参数 $\mathbf{g}_i$、深度图 $D_i$、点云图 $P_i$ 和跟踪特征 $T_i$。第一张图像被选作世界参考坐标系。

### 关键设计

**1. 交替注意力（Alternating-Attention, AA）机制**

- **功能**：在帧内自注意力和全局自注意力之间交替执行，平衡帧内特征归一化和跨帧信息融合
- **核心思路**：帧内注意力单独处理每帧的token $t_k^I$，全局注意力联合处理所有帧token $t^I$。使用 $L=24$ 层交替结构
- **设计动机**：纯全局自注意力由于缺乏帧内归一化会导致性能下降；交叉注意力虽然信息融合强但性能不如自注意力变体。消融实验显示AA在ETH3D上Overall指标从1.061/0.827降至0.709

**2. 冗余多任务预测（Over-complete Predictions）**

- **功能**：同时训练并预测相机参数、深度图、点云图和点轨迹，即使它们之间存在数学冗余关系
- **核心思路**：通过多任务损失 $\mathcal{L} = \mathcal{L}_{\text{camera}} + \mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{pmap}} + \lambda\mathcal{L}_{\text{track}}$ 联合训练。推理时，用深度+相机参数反投影得到的点云比直接用点云头更准确
- **设计动机**：尽管深度图+相机参数可以推导出点云图，但联合训练这些相关任务能相互增强精度。消融实验证实去掉任一任务都会降低点云精度

**3. 可学习的相机token和Register token**

- **功能**：让Transformer区分第一帧（参考坐标系）和其他帧，并提供额外的信息容量
- **核心思路**：为每帧添加1个相机token $t_i^{\mathbf{g}}$ 和4个register token $t_i^R$。第一帧使用与其他帧不同的可学习token初始化，输出相机token通过4层额外自注意力层预测相机参数
- **设计动机**：通过不同的初始化token让网络感知坐标系约定（第一帧为identity），帧内自注意力使camera token与对应帧的图像token匹配

### 损失函数

总损失为四项之和：$\mathcal{L} = \mathcal{L}_{\text{camera}} + \mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{pmap}} + 0.05 \cdot \mathcal{L}_{\text{track}}$

- **相机损失**：Huber损失 $\sum_{i=1}^{N} \|\hat{\mathbf{g}}_i - \mathbf{g}_i\|_\epsilon$
- **深度损失**：不确定性加权重建损失 + 梯度损失 + 不确定性正则项
- **点云图损失**：与深度损失结构相同但使用点云不确定性
- **跟踪损失**：L1对应点位置误差 + 可见性二元交叉熵

## 实验关键数据

### 主实验：相机位姿估计（RealEstate10K & CO3Dv2，10帧）

| 方法 | Re10K AUC@30↑ | CO3Dv2 AUC@30↑ | 时间 |
|------|-------------|---------------|------|
| DUSt3R | 67.7 | 76.7 | ~7s |
| MASt3R | 76.4 | 81.8 | ~9s |
| VGGSfM v2 | 78.9 | 83.4 | ~10s |
| Fast3R | 72.7 | 82.5 | ~0.2s |
| **VGGT (FF)** | **85.3** | **88.2** | **~0.2s** |
| VGGT + BA | 93.5 | 91.8 | ~1.8s |

### 消融实验：架构设计对点云估计的影响（ETH3D）

| 架构变体 | Acc.↓ | Comp.↓ | Overall↓ |
|---------|-------|--------|----------|
| Cross-Attention | 1.287 | 0.835 | 1.061 |
| Global Self-Attn Only | 1.032 | 0.621 | 0.827 |
| **Alternating-Attn** | **0.901** | **0.518** | **0.709** |

### 关键发现

- 在**无需后处理**的情况下，VGGT在相机位姿估计上超越所有需要优化的baseline 6+ AUC点
- 深度估计在DTU上Overall从DUSt3R的1.741大幅降至0.382，接近使用GT相机的方法
- 在ScanNet两视图匹配上超越专用方法Roma（AUC@5: 33.9 vs 31.8）
- 多任务联合训练对每个子任务都有增益，移除任何一个任务都会降低整体性能
- 作为特征backbone微调后在动态点追踪和新视角合成上都展现出强大能力

## 亮点与洞察

1. **统一基础模型范式**：将3D重建问题类比NLP的GPT、视觉的DINO，证明了通用3D基础模型的可行性。12亿参数的"大力出奇迹"策略在3D领域首次获得全面成功
2. **极简设计理念**：仅使用标准Transformer+交替注意力，无需任何3D归纳偏置（无交叉注意力、无几何约束模块），让数据驱动学习完成3D理解
3. **推理时分解优于直接预测**：尽管训练时联合监督点云图，推理时用深度+相机反投影比直接用点云头更准确——这是一个反直觉但实用的发现

## 局限与展望

- 模型需要大量3D标注数据和64张A100训练9天的算力
- 当前仅处理静态场景，缺乏对动态物体的显式建模（虽然通过微调可部分解决）
- 对于极端无重叠或严重遮挡场景的鲁棒性有待验证
- 未来方向：更高效的训练策略、与语言模型结合实现3D理解与交互、扩展到视频和动态场景

## 相关工作与启发

- **与DUSt3R/MASt3R的关系**：VGGT继承了点云图表示的思路，但通过多帧处理和交替注意力彻底消除了后处理需求
- **与VGGSfM的关系**：同一研究组（Oxford VGG）的工作，VGGSfM用端到端可微BA，VGGT则用纯前馈替代
- **对3D基础模型的启发**：证明了足够大的Transformer + 足够多的3D数据可以学会"隐式的多视图三角化"，不需要显式的几何推理模块

## 评分

⭐⭐⭐⭐⭐

Meta AI + Oxford VGG联合出品的重磅工作，首次实现了真正意义上的"一个前馈模型解决所有3D任务"。在多个benchmark上全面SOTA，代码开源，对后续3D基础模型研究有深远影响。唯一限制是训练资源门槛极高。

<!-- RELATED:START -->

## 相关论文

- [Quantized Visual Geometry Grounded Transformer](../../ICLR2026/3d_vision/quantized_visual_geometry_grounded_transformer.md)
- [SphereUFormer: A U-Shaped Transformer for Spherical 360 Perception](sphereuformer_a_u-shaped_transformer_for_spherical_360_perception.md)
- [GGPT: Geometry-Grounded Point Transformer](../../CVPR2026/3d_vision/ggpt_geometry_grounded_point_transformer.md)
- [End-to-End HOI Reconstruction Transformer with Graph-based Encoding](end-to-end_hoi_reconstruction_transformer_with_graph-based_encoding.md)
- [RDD: Robust Feature Detector and Descriptor Using Deformable Transformer](rdd_robust_feature_detector_and_descriptor_using_deformable_transformer.md)

<!-- RELATED:END -->
