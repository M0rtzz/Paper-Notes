---
title: "4Deform: Neural Surface Deformation for Robust Shape Interpolation"
description: "提出4Deform，利用神经隐式表示和连续速度场实现鲁棒的形状插值，通过修改水平集方程和物理正则化器处理拓扑变化"
tags:
  - CVPR2025
  - 3D Vision
  - Shape Interpolation
  - Neural Implicit Representation
  - Level Set
---

# 4Deform: Neural Surface Deformation for Robust Shape Interpolation

**会议**: CVPR 2025  
**机构**: TU Munich / University of Bonn  
**arXiv**: 2502.20208  
**主题**: 神经表面变形 / 形状插值  

## 研究背景与动机

形状插值是计算机图形学和3D视觉中的基础问题，目标是在两个给定的3D形状之间生成平滑、自然的中间过渡序列。传统方法主要包括基于网格的方法和基于变形场的方法，但它们在处理拓扑变化（如分裂、合并）时存在严重局限性。

**现有方法的核心困境：**

**基于网格的方法**：要求源形状和目标形状具有相同的拓扑结构（相同的顶点数和连接关系），当形状发生拓扑变化时完全失效。例如，一个球体变为一个环面的过程无法用网格变形来表示

**基于对应关系的方法**：需要预先建立精确的点对点对应关系，这在形状差异较大时非常困难且容易出错

**基于隐式场的简单插值**：直接在SDF空间中线性插值会产生不自然的中间形状，因为SDF值的线性组合不保证产生有意义的几何

**现有神经方法**：如OccFlow等虽然使用了神经网络，但通常在变形的物理合理性方面缺乏约束

本文的核心动机是：**能否利用神经隐式表示的灵活性，结合物理启发的正则化，实现既能处理拓扑变化又能保持变形物理合理性的形状插值方法？**

## 方法详解

### 整体框架

4Deform采用神经隐式表示（Neural Implicit Representation, NIR）结合连续速度场的方式进行形状插值。核心思想是将形状变形建模为欧氏空间中隐式表面在连续速度场驱动下的演化过程。

### 修改的水平集方程

传统水平集方程描述隐式surface在速度场下的运动：

$$\frac{\partial \varphi}{\partial t} + V^T \nabla \varphi = 0$$

本文引入了松弛项，提出修改后的水平集方程：

$$\frac{\partial \varphi}{\partial t} + V^T \nabla \varphi = -\lambda_l \varphi \cdot R(x, t)$$

其中 $R(x,t)$ 是一个可学习的松弛函数，$\lambda_l$ 控制松弛强度。这个松弛项的引入允许SDF值在演化过程中适当调整，使得拓扑变化成为可能。

### 物理正则化器

为了确保速度场的物理合理性，本文设计了两个关键的正则化器：

**空间平滑正则化（Spatial Smoothness）：**

$$L_s = \int \| (-\alpha \Delta + \gamma I) V \|^2 \, dx$$

其中 $\alpha$ 控制拉普拉斯平滑的强度，$\gamma$ 是恒等项系数。该正则化确保速度场在空间上是平滑的，避免不自然的局部扭曲。

**体积保持正则化（Volume Preservation）：**

$$L_v = \int | \nabla \cdot V |^2 \, dx$$

通过惩罚速度场的散度，鼓励变形过程中局部体积的保持。当 $\nabla \cdot V = 0$ 时，变形为不可压缩的。

### AutoDecoder架构

模型采用AutoDecoder架构，使用潜在编码 $z = z_0 \oplus z_1$ 来条件化速度场的生成。其中 $z_0$ 和 $z_1$ 分别对应源形状和目标形状的潜在表示。

| 组件 | 输入 | 输出 | 作用 |
|------|------|------|------|
| SDF网络 | 空间坐标 $x$, 时间 $t$, 潜在码 $z$ | SDF值 $\varphi(x,t)$ | 表示任意时刻的隐式表面 |
| 速度场网络 | 空间坐标 $x$, 时间 $t$, 潜在码 $z$ | 速度向量 $V(x,t)$ | 驱动表面演化 |
| 松弛网络 | 空间坐标 $x$, 时间 $t$ | 松弛值 $R(x,t)$ | 允许拓扑变化 |
| AutoDecoder | 形状对 | 潜在码 $z_0, z_1$ | 编码形状特征 |

### 对应关系损失

当已知部分点对应关系时，可以使用额外的对应关系损失：

$$L_m = \sum_{(p_i, q_i)} \| \Phi(p_i, t=1) - q_i \|^2$$

其中 $\Phi(p_i, t=1)$ 是源点 $p_i$ 通过速度场积分到 $t=1$ 时刻的位置。

### 总体损失函数

$$L = L_{\text{sdf}} + \lambda_s L_s + \lambda_v L_v + \lambda_m L_m + \lambda_r L_r$$

其中 $L_{\text{sdf}}$ 是SDF重建损失，$L_r$ 是水平集方程的残差损失。

## 实验结果

### 主要对比

在多个数据集上验证了方法的有效性：

- 在涉及拓扑变化的形状对上，4Deform显著优于传统网格变形方法
- 物理正则化器有效提升了中间形状的质量，避免了不自然的局部变形
- AutoDecoder架构使得模型能够泛化到训练分布内的新形状对

### 消融实验

- 移除空间平滑正则化 $L_s$ 导致中间形状出现明显的局部扭曲
- 移除体积保持正则化 $L_v$ 导致形状在插值过程中出现不自然的膨胀/收缩
- 松弛项 $R(x,t)$ 对处理拓扑变化至关重要

## 总结与展望

4Deform通过将神经隐式表示与物理启发的正则化相结合，为形状插值问题提供了一个统一且鲁棒的框架。修改后的水平集方程使方法能够自然地处理拓扑变化，而物理正则化器确保了变形的合理性。AutoDecoder架构提供了在形状空间中进行灵活插值的能力。未来工作可以探索将该方法扩展到动态场景重建和4D内容生成。

<!-- RELATED:START -->

## 相关论文

- [NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement](../../ICCV2025/3d_vision/neuraleaf_neural_parametric_leaf_models_with_shape_and_deformation_disentangleme.md)
- [Geometry in Style: 3D Stylization via Surface Normal Deformation](geometry_in_style_3d_stylization_via_surface_normal_deformation.md)
- [Thin-Shell-SfT: Fine-Grained Monocular Non-Rigid 3D Surface Tracking with Neural Deformation Fields](thin-shell-sft_fine-grained_monocular_non-rigid_3d_surface_tracking_with_neural_.md)
- [MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion](mp-sfm_monocular_surface_priors_for_robust_structure-from-motion.md)
- [Toward Robust Neural Reconstruction from Sparse Point Sets](toward_robust_neural_reconstruction_from_sparse_point_sets.md)

<!-- RELATED:END -->
