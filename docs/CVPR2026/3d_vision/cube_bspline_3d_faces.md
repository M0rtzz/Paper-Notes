---
title: >-
  [论文解读] CUBE: Representing 3D Faces with Learnable B-Spline Volumes
description: >-
  [CVPR 2026][3D视觉][B样条体] 提出 CUBE（Control-based Unified B-spline Encoding），一种结合 B 样条体和可学习高维控制特征的混合几何表示，通过两阶段解码（B 样条基插值 + 轻量 MLP 残差）实现可编辑、高精度的 3D 人脸重建和扫描配准。
tags:
  - "CVPR 2026"
  - "3D视觉"
  - "B样条体"
  - "人脸表示"
  - "扫描配准"
  - "局部控制"
  - "几何编辑"
---

# CUBE: Representing 3D Faces with Learnable B-Spline Volumes

**会议**: CVPR 2026  
**arXiv**: [2604.12894](https://arxiv.org/abs/2604.12894)  
**代码**: 无  
**领域**: 3D视觉 / 人脸重建  
**关键词**: B样条体, 人脸表示, 扫描配准, 局部控制, 几何编辑

## 一句话总结

提出 CUBE（Control-based Unified B-spline Encoding），一种结合 B 样条体和可学习高维控制特征的混合几何表示，通过两阶段解码（B 样条基插值 + 轻量 MLP 残差）实现可编辑、高精度的 3D 人脸重建和扫描配准。

## 研究背景与动机

**领域现状**：3D 人脸表示主要有三种范式：3D 形变模型（3DMM）提供压缩、解缠的线性空间但细节有限；非线性神经模型提升灵活性但缺乏可解释性和局部控制；隐式表示提供高细节但缺乏语义对应且需要昂贵的等值面提取。

**现有痛点**：3DMM 受限于固定拓扑和低维参数空间，无法捕获个体化高频细节。神经模型缺乏局部编辑能力。隐式模型与标准图形管线不兼容。

**核心矛盾**：表示的局部可控性、几何表达力和计算效率三者难以兼顾。

**本文目标**：设计一种兼具 B 样条局部控制特性和神经网络表达力的混合人脸表示。

**切入角度**：将传统 B 样条体的 3D 控制点替换为高维可学习控制特征，用轻量 MLP 补充高频细节。

**核心 idea**：高维控制特征格（如 8×8×8）定义连续的参数域到欧式空间映射，B 样条基提供局部支持属性实现局部编辑。

## 方法详解

### 整体框架

CUBE 由高维控制特征格参数化。给定固定模板网格上的 3D 坐标，B 样条基局部混合控制特征产生高维向量，其前三维定义基础网格位置，完整特征送入轻量 MLP 预测残差位移。输出为密集语义对应的 3D 表面。

### 关键设计

1. **高维控制特征格**:

    - 功能：用紧凑的格点替代稠密网格参数化 3D 人脸形状
    - 核心思路：传统 B 样条体用 3D 控制点，CUBE 用高维（如 32 维）控制特征替代。B 样条基在查询点处局部混合邻域控制特征，产生高维特征向量。这保留了 B 样条的局部支持特性——修改单个控制特征只影响局部区域
    - 设计动机：标准 B 样条的 3D 控制点表达力不足以用少量格点表示复杂人脸形状

2. **两阶段解码**:

    - 功能：兼顾全局形状和局部细节
    - 核心思路：混合后的高维特征向量的前三维直接定义粗糙的基础网格（全局形状），完整特征向量再送入轻量 MLP 预测从基础形状的残差位移（高频细节）
    - 设计动机：B 样条基天然平滑，难以表示高频几何。MLP 补充了这一表达力，同时保持局部支持（因为 MLP 输入来自局部混合的特征）

3. **基于 Transformer 的编码器**:

    - 功能：从非结构化点云或单目图像预测 CUBE 控制特征
    - 核心思路：训练 Transformer 编码器将非结构化 3D 头部扫描（或单目图像）映射为 CUBE 控制特征格，实现前馈式扫描配准和图像重建
    - 设计动机：CUBE 的参数空间紧凑（如 8³×32 = 16K 参数），适合直接回归

### 损失函数 / 训练策略

顶点到顶点 L2 损失 + 法线一致性损失 + 拉普拉斯平滑正则化。编码器和 CUBE 解码器端到端训练。

## 实验关键数据

### 主实验

| 方法 | 类型 | 扫描配准误差↓ | 对应精度↑ |
|------|------|-----------|----------|
| BPS | 基点集 | 2.85 | 82.3% |
| Shape-my-face | PointNet | 2.42 | 85.1% |
| ImFace | 隐式 | 2.15 | 87.5% |
| **CUBE** | B样条 | **1.89** | **91.2%** |

### 消融实验

| 配置 | 扫描误差↓ | 说明 |
|------|----------|------|
| 完整 CUBE | 1.89 | 高维特征 + MLP 残差 |
| 无 MLP 残差 | 2.35 | 仅 B 样条基 |
| 3D 控制点 (传统) | 2.78 | 无高维特征 |
| 格点 16³ | 1.85 | 更多控制点 |
| 格点 4³ | 2.45 | 较少控制点 |

### 关键发现

- MLP 残差贡献显著（去掉后误差增加 24%），说明高频细节建模重要
- 高维控制特征 vs 3D 控制点：误差从 2.78 降到 2.35（降 15%），证明高维特征增强了表达力
- 8³ 格点已足够：增加到 16³ 仅微小提升

## 亮点与洞察

- 将 NURBS 这一 CAD 经典表示引入人脸建模并用可学习特征增强表达力，是一个优雅的混合设计
- 局部支持特性的保留使得交互式编辑成为可能：通过交换或修改单个控制特征实现局部人脸编辑
- 两阶段解码（B 样条粗糙 + MLP 精细）的思路可推广到其他几何表示

## 局限与展望

- 仅针对人脸，头发和配饰未建模
- 与隐式表示相比在极端表情下的细节可能不足
- 格点大小的选择需要权衡表达力和效率
- 可扩展到全身或手部等其他部位

## 相关工作与启发

- **vs 3DMM (FLAME)**: 3DMM 用线性 PCA 基，CUBE 用 B 样条体 + MLP，表达力更强且保持局部可控
- **vs ImFace**: ImFace 是隐式 SDF，需要 Marching Cubes 提取网格；CUBE 通过模板查询直接输出网格

## 评分

- 新颖性: ⭐⭐⭐⭐ B 样条体 + 高维特征的混合表示有创意
- 实验充分度: ⭐⭐⭐⭐ 扫描配准和图像重建两个应用的验证
- 写作质量: ⭐⭐⭐⭐ 表示设计描述清晰
- 价值: ⭐⭐⭐⭐ 对可编辑人脸建模有实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Representing 3D Shapes with 64 Latent Vectors for 3D Diffusion Models](../../ICCV2025/3d_vision/representing_3d_shapes_with_64_latent_vectors_for_3d_diffusion_models.md)
- [\[ECCV 2024\] NOVUM: Neural Object Volumes for Robust Object Classification](../../ECCV2024/3d_vision/novum_neural_object_volumes_for_robust_object_classification.md)
- [\[CVPR 2025\] Volumetric Surfaces: Representing Fuzzy Geometries with Layered Meshes](../../CVPR2025/3d_vision/volumetric_surfaces_representing_fuzzy_geometries_with_layered_meshes.md)
- [\[ICCV 2025\] SL2A-INR: Single-Layer Learnable Activation for Implicit Neural Representation](../../ICCV2025/3d_vision/sl2a-inr_single-layer_learnable_activation_for_implicit_neural_representation.md)
- [\[CVPR 2025\] Learnable Infinite Taylor Gaussian for Dynamic View Rendering](../../CVPR2025/3d_vision/learnable_infinite_taylor_gaussian_for_dynamic_view_rendering.md)

</div>

<!-- RELATED:END -->
