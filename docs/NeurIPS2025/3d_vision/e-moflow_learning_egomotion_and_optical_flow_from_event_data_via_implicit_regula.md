---
title: >-
  [论文解读] E-MoFlow: Learning Egomotion and Optical Flow from Event Data via Implicit Regularization
description: >-
  [NeurIPS 2025][3D视觉][事件相机] 提出 E-MoFlow，通过将光流建模为隐式神经表示、自运动建模为连续样条，并利用微分几何约束联合优化两者，在无监督范式下实现事件数据的 6-DoF 自运动和稠密光流联合估计。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 事件相机
  - 光流估计
  - 自运动估计
  - 隐式正则化
  - 无监督学习
  - Neural ODE
---

# E-MoFlow: Learning Egomotion and Optical Flow from Event Data via Implicit Regularization

**会议**: NeurIPS 2025  
**arXiv**: [2510.12753](https://arxiv.org/abs/2510.12753)  
**代码**: [项目页面](https://akawincent.github.io/EMoFlow/)  
**领域**: 3d_vision  
**关键词**: 事件相机, 光流估计, 自运动估计, 隐式正则化, 无监督学习, Neural ODE  

## 一句话总结

提出 E-MoFlow，通过将光流建模为隐式神经表示、自运动建模为连续样条，并利用微分几何约束联合优化两者，在无监督范式下实现事件数据的 6-DoF 自运动和稠密光流联合估计。

## 背景与动机

事件相机异步感知逐像素亮度变化，具有高动态范围和低延迟等优势，但其数据特性给运动估计带来独特挑战：

1. **光流的孔径问题**：事件数据的局部性导致估计的光流本质上是法向光流，无法恢复完整运动场
2. **6-DoF 自运动不可解**：无深度先验时从事件数据估计完整 6-DoF 运动理论上不可解
3. **缺乏可靠数据关联**：事件数据没有传统帧图像的长期特征匹配能力

现有解决方案的局限：
- **显式空间-时间正则化**（如附加平滑损失）：引入偏差且增加计算开销
- **显式结构-运动参数化**（联合估计光流+深度+运动）：增加自由度，更容易陷入局部最小值

核心创新：用**隐式正则化**替代显式正则化——通过表示本身的归纳偏置（神经网络平滑性、样条连续性）和微分几何约束（绕过显式深度估计）自然施加约束。

## 核心问题

如何在无监督框架下联合估计事件数据的光流和 6-DoF 自运动，避免显式正则化的偏差和显式深度估计的局部最小值？

## 方法详解

### 连续光流表示

将光流建模为隐式神经表示，给定时间 $t$ 和归一化像素坐标 $\mathbf{x}$，网络输出光流向量（速度场）：

$$\mathbf{u}_\theta(t, \mathbf{x}) = \text{NN}_\theta(t, \mathbf{x})$$

事件的 warp 轨迹建模为 Neural ODE：

$$\frac{d\mathbf{x}_k(t)}{dt} = \text{NN}_\theta(t, \mathbf{x}_k(t)), \quad \mathbf{x}_k(t_k) = \mathbf{x}_k$$

积分解：$\mathbf{x}_k(t) = \mathbf{x}_k + \int_{t_k}^t \text{NN}_\theta(s, \mathbf{x}_k(s)) ds$

反向梯度通过伴随 ODE 高效计算，避免直接反向传播的梯度爆炸和内存问题：

$$\frac{d\mathbf{a}_k(t)}{dt} = -\mathbf{a}_k(t)^\top \frac{\partial \text{NN}_\theta(t, \mathbf{x}_k(t))}{\partial \mathbf{x}_k(t)}$$

关键区别：直接建模速度场而非位移，适用于更激烈的运动场景。

### 连续自运动表示

用三次 B 样条表示相机的角速度和线速度：

$$[\boldsymbol{\omega}_\beta(t); \boldsymbol{\nu}_\beta(t)] = \sum_{i=0}^n \mathbf{B}_{i,3}(t) \beta_i$$

其中 $\beta_i \in \mathbb{R}^6$ 为控制点，$\mathbf{B}_{i,3}$ 为三次基函数。样条的连续性自然编码了时间平滑性先验。

### 微分光流损失

基于对比度最大化：将事件 warp 到参考时间 $t_{\text{ref}}$ 后累积为 IWE（Image of Warped Events），最大化其对比度：

$$L_{\text{flow}} = -\frac{1}{HW} \sum_{i,j} (I_{ij} - \mu_I)^2$$

### 微分几何损失

基于微分极线约束，在齐次坐标下联合约束光流和自运动，**无需显式深度估计**：

$$L_{\text{geometry}} = \left\| \hat{\mathbf{u}}_\theta(t, \mathbf{x})^\top [\boldsymbol{\nu}_\beta(t)]_\times \hat{\mathbf{x}} - \hat{\mathbf{x}}^\top \mathbf{s}_\beta(t) \hat{\mathbf{x}} \right\|_2^2$$

其中 $\mathbf{s}_\beta(t) = \frac{1}{2}([\boldsymbol{\nu}_\beta(t)]_\times [\boldsymbol{\omega}_\beta(t)]_\times + [\boldsymbol{\omega}_\beta(t)]_\times [\boldsymbol{\nu}_\beta(t)]_\times)$。

两个损失联合优化避免光流估计陷入局部最小值（如纯平移场景的退化解）。

### 训练流程

$$\min_{\theta, \beta} \mathbb{E}_{t_{\text{ref}}} [L_{\text{flow}}(\mathcal{E}_{\text{neigh}}(t_{\text{ref}}), \theta)] + \mathbb{E}_{\{\mathbf{x}, t\} \sim \mathcal{E}} [L_{\text{geometry}}(t, \mathbf{x}, \theta, \beta)]$$

对每个事件序列从头优化（test-time optimization），不需要预训练。

## 实验关键数据

### MVSEC 光流估计 (dt=1)

| 方法 | 类型 | 平均 EPE ↓ | 平均 %Out ↓ |
|------|------|-----------|------------|
| ADM-Flow | 有监督 | 0.533 | 0.340 |
| MultiCM-V2 | 模型无关 | 0.348 | 0.055 |
| MultiCM | 模型无关 | 0.455 | 0.270 |
| EV-MGRFlowNet | 无监督 | 0.495 | 0.958 |
| **E-MoFlow** | **无监督** | **0.450** | **0.328** |

无监督方法中最佳，与有监督方法 ADM-Flow 接近。

### MVSEC 光流估计 (dt=4, 大运动)

| 方法 | 类型 | 平均 EPE ↓ |
|------|------|-----------|
| MultiCM-V2 | 模型无关 | 1.108 |
| EV-MGRFlowNet | 无监督 | 1.763 |
| **E-MoFlow** | **无监督** | **1.773** |

大运动场景下与其他无监督方法可比，略逊于 MultiCM-V2（该方法也联合估计深度+运动）。

### DSEC 光流估计

在 DSEC 数据集上同样在无监督方法中取得最佳或次佳结果，验证了泛化性。

### 自运动估计

在 MVSEC 上实现了事件相机 6-DoF 自运动估计的 SOTA（无监督类），角速度和线速度误差均优于现有方法。

## 亮点

1. **隐式正则化设计精妙**：通过 Neural ODE 的网络平滑性和 B 样条的连续性施加空间-时间先验，无需显式损失项
2. **绕过深度估计**：微分几何约束直接关联光流和自运动，避免增加优化自由度导致的局部最小值
3. **完全无监督**：不需要任何标注、深度或灰度图像监督
4. 光流+自运动联合估计在事件相机领域具有重要意义

## 局限与展望

- 逐序列优化（test-time optimization），推理效率较低
- dt=4 大运动场景下性能不如 MultiCM-V2，大位移仍是挑战
- 仅在 MVSEC 和 DSEC 两个数据集上验证，缺少更多真实场景
- 未与最新的有监督大模型方法对比

## 与相关工作的对比

- **vs MultiCM/MultiCM-V2**：MultiCM 系列需要显式深度估计和运动场参数化，E-MoFlow 通过微分几何约束绕过深度
- **vs USL-EV-FlowNet**：USL-EV-FlowNet 用重投影损失联合预测光流和自运动，但需要离散化体积表示；E-MoFlow 用连续隐式表示
- **vs EvLinearSolver**：线性解法需要先验知识（如已知角速度），无法完整恢复 6-DoF；E-MoFlow 完全自主

## 启发与关联

隐式正则化的思想具有广泛适用性：通过选择具有合适归纳偏置的表示来替代显式正则项。Neural ODE 建模事件轨迹的方式可推广到其他异步数据的运动估计。微分几何约束替代运动场方程避免深度估计是一个值得推广的技巧。

## 评分

- ⭐ 新颖性: 9/10 — 隐式正则化+微分几何约束联合优化的框架设计非常优雅
- ⭐ 实验充分度: 7/10 — 两个数据集、与多种方法对比，但缺少更多真实场景和效率分析
- ⭐ 写作质量: 8/10 — 数学推导严谨，动机阐述清晰
- ⭐ 价值: 8/10 — 事件相机领域的重要工作，联合估计框架有实际意义

<!-- RELATED:START -->

## 相关论文

- [Linearly Constrained Diffusion Implicit Models](linearly_constrained_diffusion_implicit_models.md)
- [EventHub: Data Factory for Generalizable Event-Based Stereo Networks without Active Sensors](../../CVPR2026/3d_vision/eventhub_data_factory_for_generalizable_event-based_stereo_networks_without_acti.md)
- [EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)
- [Flow-NeRF: Joint Learning of Geometry, Poses, and Dense Flow within Unified Neural Representations](../../CVPR2025/3d_vision/flow-nerf_joint_learning_of_geometry_poses_and_dense_flow_within_unified_neural_.md)
- [EAG3R: Event-Augmented 3D Geometry Estimation for Dynamic and Extreme-Lighting Scenes](eag3r_event-augmented_3d_geometry_estimation_for_dynamic_and_extreme-lighting_sc.md)

<!-- RELATED:END -->
