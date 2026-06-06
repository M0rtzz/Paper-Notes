---
title: >-
  [论文解读] Differentiable Convex Polyhedra Optimization from Multi-view Images
description: >-
  [ECCV 2024][3D视觉][可微渲染] 提出一种基于对偶变换和三平面交点求解的可微凸多面体构造方法，绕过隐式场监督，直接利用多视角图像损失进行梯度优化，实现高保真的凸多面体形状表示。
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "可微渲染"
  - "凸多面体"
  - "形状表示"
  - "对偶变换"
  - "多视角重建"
---

# Differentiable Convex Polyhedra Optimization from Multi-view Images

**会议**: ECCV 2024  
**arXiv**: [2407.15686](https://arxiv.org/abs/2407.15686)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 可微渲染, 凸多面体, 形状表示, 对偶变换, 多视角重建

## 一句话总结

提出一种基于对偶变换和三平面交点求解的可微凸多面体构造方法，绕过隐式场监督，直接利用多视角图像损失进行梯度优化，实现高保真的凸多面体形状表示。

## 研究背景与动机

**领域现状**：用简单几何基元（凸多面体）组合表示复杂形状是一种有前景的方法，在形状解析、物理模拟、碰撞检测和紧凑网格表示等下游任务中有广泛应用。CVXNet和BSPNet等方法将形状表示为多个凸多面体的并集。

**现有痛点**：现有方法（如CVXNet、BSPNet）严重依赖隐式场（SDF或占用场）作为监督信号，需要水密（watertight）网格作为训练数据。预处理数据集为离散采样点+GT隐式值的过程对计算资源和存储都造成巨大负担。此外，形状组合操作（sigmoid近似占用、softmin/softmax近似交集/并集）导致不准确的SDF或有偏梯度估计。

**核心矛盾**：凸多面体的构造（超平面交点计算）本身是不可微的，无法直接用于基于梯度的优化。现有方法用隐式场绕过这一问题，但引入了大量计算开销和精度损失。

**本文目标** 消除对隐式场的依赖，实现纯显式表面模型的可微凸多面体优化，仅用多视角图像作为监督。

**切入角度**：利用计算几何中的对偶变换（duality transform），将超平面交点计算分解为"不可微的拓扑发现"+"可微的顶点位置求解"两步。

**核心 idea**：通过对偶变换非可微地确定每个凸多面体顶点由哪三个平面相交构成，然后用可微线性方程求解器计算顶点位置，使梯度从图像损失传播到平面参数。

## 方法详解

### 整体框架

给定一组超平面参数 $\theta$，通过以下pipeline构建可微凸多面体：
1. 对偶变换将超平面映射为对偶空间中的点
2. 在对偶空间计算凸包，每个凸包面片对应原始空间中的一个顶点
3. 记录每个顶点由哪三个平面相交构成（平面ID）
4. 用可微线性方程求解器重新计算顶点位置
5. 构建三角面片，送入可微渲染器计算图像损失

### 关键设计

1. **可微凸多面体渲染（核心梯度链）**:

    - 功能：将图像损失的梯度传播到超平面参数
    - 核心思路：将梯度链分解为三部分：
    $\frac{\partial \mathcal{L}}{\partial \theta} = \underbrace{\frac{\partial \mathcal{L}}{\partial \mathcal{I}}}_{\text{图像损失}} \cdot \underbrace{\frac{\partial \mathcal{I}}{\partial \mathcal{V}}}_{\text{可微渲染}} \cdot \underbrace{\frac{\partial \mathcal{V}}{\partial \theta}}_{\text{可微顶点位置}}$
    - 设计动机：直接计算 $\partial \mathcal{I}/\partial \theta$ 不可行（无显式网格、自相交问题、体渲染显存爆炸），因此通过构建显式网格间接获取梯度。

2. **对偶变换求超平面交点**:

    - 功能：确定每个凸多面体顶点由哪三个超平面相交构成
    - 核心思路：给定超平面 $a_i^T x \leq b_i$，将其对偶映射为点 $(a_{i_x}/b_i, a_{i_y}/b_i, a_{i_z}/b_i)$。在对偶空间计算凸包，对偶凸包的每个面片对应原始空间中一个顶点，面片的三个顶点ID即为相交的三个平面ID。
    - 设计动机：半空间交点问题可优雅地转化为对偶空间的凸包问题，且用QHull库高效求解。这一步不需要可微。

3. **可微三平面交点求解**:

    - 功能：根据已知的平面ID，可微地计算顶点坐标
    - 核心思路：顶点位置通过求解线性方程组得到：$Ax = b$，其中 $A$ 的行是超平面法向量，$b$ 是偏移量。PyTorch的线性代数求解器天然支持反向传播。
    - 设计动机：将不可微的拓扑发现和可微的几何计算解耦，是本文最关键的洞察。

4. **优化策略（Optimization Strategies）**:

    - **Persistent Convex**：约束所有平面偏移量 $b \in \mathbb{R}^+$，确保原点始终在凸多面体内部，避免退化。为灵活性引入额外的平移参数。
    - **Convex Purging**：移除体积低于阈值的小凸多面体，提高渲染效率。
    - **Plane Purging**：移除未参与任何顶点计算的冗余超平面（对偶顶点落在对偶凸包内部的平面）。
    - **Convex Densification**：对凸多面体mesh做Loop细分，用细分后mesh的凸包方程替换原平面参数，增强高曲率区域表示能力。
    - **Convex Spawning**：随机重新初始化被清除的凸多面体，避免陷入局部最小值。

### 损失函数 / 训练策略

- 仅使用图像 $L_1$ 损失作为监督
- 学习率：凸多面体平移 $1 \times 10^{-2}$，超平面参数 $1 \times 10^{-3}$
- 总优化步数20000步，包含10次densification和spawning
- 使用NVDiffRast作为可微渲染器
- 每个形状独立优化（非学习型方法）

## 实验关键数据

### 主实验（ShapeNet形状重建）

| 方法 | L1 CD | L2 CD (×1000) | Normal Consistency |
|------|-------|---------------|-------------------|
| VP (32 cuboids) | 0.048 | 1.861 | 0.738 |
| CVXNet (64 convexes) | 0.024 | 0.526 | 0.891 |
| BSPNet (64 convexes) | 0.027 | 0.699 | 0.745 |
| **本文 (32 convexes)** | **0.022** | **0.592** | **0.925** |

### 多视角纹理重建（DTU + NeRF Synthetic）

| 数据集 | 方法 | L2 CD |
|--------|------|-------|
| DTU S40 | DBW | 1.16 |
| DTU S40 | **本文** | **1.01** |
| NeRF Lego | DBW | 0.91 |
| NeRF Lego | **本文** | **0.44** |
| NeRF Chair | DBW | 1.07 |
| NeRF Chair | **本文** | **0.35** |

### 消融实验

| 配置 | L2 CD | 说明 |
|------|-------|------|
| 16 Convex | 0.32 | 抽象度高，细节不足 |
| 32 Convex | 0.14 | 默认配置 |
| 64 Convex | 0.08 | 细节提升 |
| 128 Convex | 0.05 | 接近饱和 |
| 去掉Densification | 0.46 | 曲面表示能力大幅下降 |
| 去掉Spawning | 0.54 | 局部最小值，重建不完整 |

### 关键发现

- 仅用32个凸多面体即可超越使用64个凸多面体的CVXNet和BSPNet
- 在薄结构和精细细节（如Lamp类别）上优势特别明显
- Densification和Spawning对重建质量至关重要
- 对有大空腔的物体（如cabinet）效果较差，因为纯RGB监督存在遮挡问题

## 亮点与洞察

- **巧妙的可微/不可微解耦**：将拓扑发现（哪些平面相交）和几何计算（交点在哪）分开，前者不需要可微，后者天然可微，非常优雅
- **无需3D隐式场监督**：仅需多视角图像，大幅降低数据要求，训练数据规模可扩大数个量级
- **凸多面体的并集由渲染器自然处理**：z-buffer或光线-物体求交天然处理并集操作，无需显式的布尔运算
- **形状解析能力**：每个凸多面体天然对应物体的一个语义部分，可用于shape parsing

## 局限与展望

- 凸多面体表示的细节精度仍不如普通网格和隐式表示
- 目前限于单物体重建，未扩展到完整场景
- 非静态网格拓扑导致无法做UV展开，只能用体积纹理
- Densification和purging过程较启发式
- 依赖可微网格渲染的固有限制（如三角形自相交产生的隐式边缘无梯度）

## 相关工作与启发

- **vs CVXNet/BSPNet**: 两者都依赖隐式场监督，需要watertight mesh和大量存储；本文用图像监督，且每个顶点位置是精确的三平面交点而非近似
- **vs 3DGS**: 本文借鉴了3DGS的densification/purging思想，但表示基元从高斯球换成凸多面体，更适合形状解析
- **vs DifferentiableBlockWorlds**: DBW用长方体和超二次曲面，本文用凸多面体，在NeRF Synthetic数据集上大幅领先

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 对偶变换+可微求解的组合非常优雅，首次实现不依赖隐式场的可微凸多面体优化
- 实验充分度: ⭐⭐⭐⭐ 覆盖形状重建、纹理重建、形状解析，消融实验充分
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，方法描述完整
- 价值: ⭐⭐⭐⭐ 为凸多面体形状表示开辟了基于图像监督的新方向，有望影响物理仿真和碰撞检测等下游应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images](mvsplat_efficient_3d_gaussian_splatting_from_sparse_multi-view_images.md)
- [\[ECCV 2024\] LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation](lgm_large_multi-view_gaussian_model_for_high-resolution_3d_content_creation.md)
- [\[ECCV 2024\] MVDD: Multi-View Depth Diffusion Models](mvdd_multi-view_depth_diffusion_models.md)
- [\[ECCV 2024\] CrossScore: Towards Multi-View Image Evaluation and Scoring](crossscore_towards_multi-view_image_evaluation_and_scoring.md)
- [\[ECCV 2024\] SparseSSP: 3D Subcellular Structure Prediction from Sparse-View Transmitted Light Images](sparsessp_3d_subcellular_structure_prediction_from_sparse-view_transmitted_light.md)

</div>

<!-- RELATED:END -->
