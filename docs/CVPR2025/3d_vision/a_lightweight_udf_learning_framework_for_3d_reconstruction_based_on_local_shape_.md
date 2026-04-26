---
title: >-
  [论文解读] A Lightweight UDF Learning Framework for 3D Reconstruction Based on Local Shape Functions
description: >-
  [CVPR 2025][3D视觉][无符号距离场] 本文提出LoSF-UDF，一种基于局部形状函数学习无符号距离场（UDF）的轻量级框架，仅需在合成的局部点云patch上训练一次（653KB参数、0.5GB数据），即可泛化重建各种类型的3D表面，且对噪声和离群点具有鲁棒性。
tags:
  - CVPR 2025
  - 3D视觉
  - 无符号距离场
  - 点云重建
  - 局部形状函数
  - 轻量化网络
  - 表面重建
---

# A Lightweight UDF Learning Framework for 3D Reconstruction Based on Local Shape Functions

**会议**: CVPR 2025  
**arXiv**: [2407.01330](https://arxiv.org/abs/2407.01330)  
**代码**: [https://jbhu67.github.io/LoSF-UDF.github.io/](https://jbhu67.github.io/LoSF-UDF.github.io/)  
**领域**: 3D视觉  
**关键词**: 无符号距离场, 点云重建, 局部形状函数, 轻量化网络, 表面重建

## 一句话总结
本文提出LoSF-UDF，一种基于局部形状函数学习无符号距离场（UDF）的轻量级框架，仅需在合成的局部点云patch上训练一次（653KB参数、0.5GB数据），即可泛化重建各种类型的3D表面，且对噪声和离群点具有鲁棒性。

## 研究背景与动机
3D表面重建是计算机视觉和图形学的核心问题。符号距离场（SDF）和占有场（Occupancy Field）只能重建封闭（watertight）几何体，无法处理开放表面和非流形结构。无符号距离场（UDF）没有内外的概念，理论上可以表示任意拓扑的几何体，近年来受到越来越多关注。

然而现有UDF学习方法面临两大瓶颈：1）无监督方法（如CAP-UDF、LevelSetUDF）需要逐个点云迭代优化，训练耗时长达15分钟/个；2）有监督方法（如GeoUDF）依赖大规模3D形状数据集训练，泛化能力受限于训练数据分布，且训练数据准备成本高（如ShapeNet需要120GB）。

本文的核心洞察非常优雅：UDF是一个局部量，不依赖全局上下文。在局部区域内，3D形状呈现简单的数学模式——光滑区域可以用二次曲面近似，尖锐特征可以用简单函数构造。因此可以用数学函数生成的局部patch来训练网络，一次训练即可泛化到各种全局形状。

## 方法详解

### 整体框架
LoSF-UDF的pipeline分为两阶段：1）在合成的局部patch数据集上训练UDF预测网络；2）对任意输入点云，提取每个查询点周围的局部patch，用训练好的网络预测UDF值，最后用DCUDF提取网格。

### 关键设计

1. **局部形状函数（Local Shape Functions）**:

    - **光滑patch**：基于微分几何理论，正则曲面上任一点的局部几何可以用二次曲面近似：$z = \frac{1}{2}(\kappa_1 x^2 + \kappa_2 y^2)$，其中$\kappa_1, \kappa_2$为主曲率。通过调节主曲率，可以生成椭球型、双曲型、抛物型和平面型四种基本局部几何
    - **尖锐patch**：针对不可微点（折痕、尖端、角点、V鞍点），设计了统一的函数形式$z = 1 - h \cdot g(x,y)$，其中$h$控制尖锐程度，$g$根据特征类型选择不同函数
    - 在半径$r_0$的圆内采样$m$个点生成patch，沿原点法线方向采样查询点并计算其UDF真值
    - 整个训练集仅需0.5GB存储，远小于GeoUDF的120GB

2. **注意力UDF预测网络**:

    - **Point-Net**：提取局部patch $\mathcal{P}$ 的几何特征 $\mathbf{f}_p$，使用4个ResNet块
    - **Vectors-Net**：对查询点到patch各点的向量集合 $\mathcal{V} = \{\mathbf{p}_i - \mathbf{q}\}$ 提取距离相关特征 $\mathbf{f}_v$
    - **Cross-Attention模块**：以patch特征为Key-Value，向量特征为Query，通过交叉注意力融合出与UDF估计最相关的几何特征 $\mathbf{f}_G$
    - 拼接距离向量 $\mathbf{d}$ 和 $\mathbf{f}_G$，经全连接层输出UDF值，使用softplus激活保证非负性
    - 整个网络仅653KB可训练参数

3. **去噪模块（Denoising Module）**:

    - 针对噪声点云，直接计算的距离向量会包含误差
    - 设计了一个轻量模块预测距离修正量 $\Delta\mathbf{d}$，加到原始距离上以提高UDF估计精度
    - 使用L1稀疏性正则化约束修正量

4. **与无监督方法的集成**:

    - LoSF-UDF可以作为CAP-UDF、LevelSetUDF等无监督方法的初始化
    - 设计了时间衰减权重$\alpha_t$从1逐渐降至0的集成损失函数
    - 前期依靠LoSF提供合理初始化，后期由无监督损失细化

### 损失函数 / 训练策略
- UDF预测损失：$\mathcal{L}_u = |\mathcal{U}(\mathbf{q}) - \mathcal{U}_\Theta(\mathbf{q})|$（L1损失）
- 去噪正则化：$\mathcal{L}_r = |\Delta\mathbf{d}|$
- 总损失：$\mathcal{L} = \mathcal{L}_u + \lambda_d \mathcal{L}_r$，其中$\lambda_d = 0.01$
- 数据增强：随机旋转、20%的光滑patch随机截断模拟开放边界、30%数据添加高斯噪声$\mathcal{N}(0, 0.1)$
- 评估时将点云归一化到$[-0.5, 0.5]$边界框，提取半径$r=0.018$内的局部patch

## 实验关键数据

### 主实验

| 数据集 | 指标 | LoSF-UDF | CAP-UDF | LevelSetUDF | GeoUDF | DUDF |
|--------|------|----------|---------|-------------|--------|------|
| ShapeNetCars (Clean) | CD↓ | **0.611** | 1.324 | 1.223 | 0.832 | 0.773 |
| ShapeNetCars (Noise) | CD↓ | **0.614** | 1.208 | 1.176 | 1.869 | 1.939 |
| ShapeNetCars (Outliers) | CD↓ | **0.705** | 2.918 | 2.763 | 2.598 | 2.949 |
| DeepFashion3D (Clean) | CD↓ | 0.548 | 0.885 | 0.881 | 0.625 | **0.506** |
| DeepFashion3D (Outliers) | CD↓ | **0.559** | 5.150 | 5.083 | 3.883 | 2.558 |
| Real ScanData | CD↓ | **1.551** | 2.101 | 1.637 | 1.762 | 4.207 |

### 消融实验

| 配置 | CD | 说明 |
|------|-----|------|
| 无Points-Net和Cross-Attn | 较高 | 无法学习局部几何特征 |
| 无Cross-Attn | 中等 | 缺少几何-距离特征融合 |
| 完整模型 | 最低 | 交叉注意力有效融合特征 |
| 无去噪模块（噪声输入） | 高 | 表面粗糙 |
| 有去噪模块（噪声输入） | 低 | 表面光滑稳定 |

### 效率对比

| 方法 | SRB时间(min) | 参数量 | 存储(KB) |
|------|-------------|--------|----------|
| LoSF-UDF | 0.87 | 167K | **653** |
| GeoUDF | 0.08 | 253K | 990 |
| CAP-UDF | 15.87 | 463K | 1809 |
| LevelSetUDF | 15.08 | 463K | 1809 |
| LoSF+CAP-UDF | **6.84** | - | - |

### 关键发现
- 在噪声和离群点条件下，LoSF-UDF的优势最为突出——CD指标比次优方法低3-4倍
- 与无监督方法集成后，收敛速度提升2-3倍（如CAP-UDF从15.87分钟降至6.84分钟）
- 仅653KB参数就能达到甚至超越需要120GB训练数据的方法

## 亮点与洞察
- 核心思想极其简洁：UDF是局部量，可以用数学函数生成训练数据。这个洞察建立在微分几何的坚实理论基础上
- 「一次训练、处处适用」的范式打破了现有UDF方法对大规模3D数据集的依赖
- 轻量化做到了极致：653KB参数+0.5GB训练数据，但性能不输甚至超越百GB级别训练的方法
- 作为初始化模块的设计很有实用价值，可以即插即用地加速现有无监督方法
- 去噪模块的设计巧妙——不是去噪点云，而是修正距离向量，计算量极小

## 局限与展望
- 纯局部方法无法处理大面积缺失区域的补全（这是全局方法的优势）
- 推理速度（0.87分钟/模型）虽然比无监督方法快，但比GeoUDF（0.08分钟）慢一个量级
- 局部patch半径$r$的选择需要根据点云密度调整，虽然作者提供了bias分析方法，但仍需人工干预
- 点云密度要求至少30点/单位面积，对极低密度点云需要额外的上采样预处理
- 尖锐特征的函数设计仍是手工的四种类型，可能遗漏某些复杂的非光滑几何

## 相关工作与启发
- Ohtake等人2003年提出的多尺度Partition of Unity框架启发了局部曲面重建的思路
- GeoUDF首次从局部角度学习UDF，但依赖ShapeNet训练；本文进一步将训练数据简化为数学函数生成的patch
- CAP-UDF和LevelSetUDF是代表性的无监督UDF方法，本文作为初始化与之集成形成互补
- DCUDF作为从UDF提取网格的SOTA方法，被各方法统一采用以确保公平比较

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用数学函数合成局部patch训练UDF的思路新颖且有理论基础
- 实验充分度: ⭐⭐⭐⭐ 多数据集多指标对比+消融实验+集成实验，覆盖面广
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，数学推导严谨，但部分实验细节在补充材料中
- 价值: ⭐⭐⭐⭐⭐ 轻量化+鲁棒性的组合解决了UDF学习的实际痛点，可即插即用

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Stable-SCore: A Stable Registration-Based Framework for 3D Shape Correspondence](stable-score_a_stable_registration-based_framework_for_3d_shape_correspondence.md)
- [\[CVPR 2025\] Evolving High-Quality Rendering and Reconstruction in a Unified Framework with Contribution-Adaptive Regularization](evolving_high-quality_rendering_and_reconstruction_in_a_unified_framework_with_c.md)
- [\[CVPR 2025\] Touch2Shape: Touch-Conditioned 3D Diffusion for Shape Exploration and Reconstruction](touch2shape_touch-conditioned_3d_diffusion_for_shape_exploration_and_reconstruct.md)
- [\[CVPR 2025\] GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping](gausshdr_high_dynamic_range_gaussian_splatting_via_learning_unified_3d_and_2d_lo.md)
- [\[CVPR 2025\] DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction](dualpm_dual_point_maps_shape_pose.md)

<!-- RELATED:END -->
