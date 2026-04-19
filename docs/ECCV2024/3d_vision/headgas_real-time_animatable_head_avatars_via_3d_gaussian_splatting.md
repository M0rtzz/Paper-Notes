---
title: >-
  [论文解读] HeadGaS: Real-Time Animatable Head Avatars via 3D Gaussian Splatting
description: >-
  [ECCV 2024][3D视觉][3D Gaussian Splatting] 提出HeadGaS，通过为每个3D高斯基元配备可学习的潜在特征基底，利用表情参数线性混合特征并经MLP预测表情相关的颜色和不透明度，实现**实时（250+ fps）**且高质量的可动画头部重建，PSNR超越基线约2 dB。
tags:
  - ECCV 2024
  - 3D视觉
  - 3D Gaussian Splatting
  - 头部动画
  - 可驱动头部建模
  - 表情迁移
  - 实时渲染
---

# HeadGaS: Real-Time Animatable Head Avatars via 3D Gaussian Splatting

**会议**: ECCV 2024  
**arXiv**: [2312.02902](https://arxiv.org/abs/2312.02902)  
**代码**: 无公开代码  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, 头部动画, 可驱动头部建模, 表情迁移, 实时渲染

## 一句话总结

提出HeadGaS，通过为每个3D高斯基元配备可学习的潜在特征基底，利用表情参数线性混合特征并经MLP预测表情相关的颜色和不透明度，实现**实时（250+ fps）**且高质量的可动画头部重建，PSNR超越基线约2 dB。

## 研究背景与动机

**领域现状**: 可动画3D头部重建是数字人、AR/VR和远程会议的核心技术。基于NeRF的方法（如INSTA、NeRFBlendShape）在质量和速度之间存在trade-off，交互帧率仅10-15 fps。

**现有痛点**: NeRF方法因体渲染的密集采样限制了渲染速度；显式方法（网格、点云）虽有更强的几何约束但难以保持照片级真实感（如INSTA出现三角面片伪影）。

**核心矛盾**: 3DGS原始设计是静态场景表示，不支持表情驱动的动态外观变化。直觉上应移动高斯位置来建模动态，但这使优化更复杂。

**本文目标**: 如何将3DGS扩展为可动画的头部表示，同时保持实时渲染速度和高保真度。

**切入角度**: 受传统blendshape模型启发，不移动高斯基元，而是通过改变其**不透明度和颜色**来表示面部动态——即通过"过表示"（over-representation）实现运动。

**核心 idea**: 每个高斯携带可学习特征基底，表情参数作为混合权重在潜在空间进行线性组合，经轻量MLP输出表情相关的颜色和不透明度。

## 方法详解

### 整体框架

输入单目视频，预处理获取每帧的头部姿态和表情参数（兼容FLAME和FaceWarehouse）。模型核心是**特征增强的3D高斯集合**：

$$\mathcal{G}_a = (\boldsymbol{\Sigma}, \boldsymbol{\mu}, \mathbf{F})$$

其中 $\mathbf{F} \in \mathbb{R}^{B \times f_{dim}}$ 是可学习特征基底（$B$=表情维度，$f_{dim}$=特征维度）。渲染时，表情参数 $\mathbf{e}_i$ 与特征基底混合生成帧特定特征，经MLP预测颜色和不透明度，再通过标准的tile-based光栅化渲染图像。

### 关键设计

1. **特征基底混合 (Feature Blending)**: 每个高斯拥有特征基底 $\mathbf{F} \in \mathbb{R}^{B \times f_{dim}}$，给定第 $i$ 帧的表情权重 $\mathbf{e}_i \in \mathbb{R}^B$，通过线性混合得到帧特定特征：

$$\mathbf{f}_i = \mathbf{F}^T \mathbf{e}_i + \mathbf{f}_0$$

其中 $\mathbf{f}_0$ 是偏置项。这类似传统blendshape在潜在特征空间而非几何空间进行混合。$B=52$（FLAME前52个表情系数），$f_{dim}=32$。设计动机：让每个高斯独立学习动态特征，比让单一MLP学习所有高斯的全局动态更具表达力。

2. **轻量MLP预测颜色与不透明度**: 帧特定特征 $\mathbf{f}_i$ 与位置编码 $\psi(\boldsymbol{\mu})$ 输入两层MLP（隐层64通道）：

$$\mathbf{c}_i, \alpha_i = \phi(\mathbf{f}_i, \psi(\boldsymbol{\mu}))$$

输出颜色 $\mathbf{c}_i \in \mathbb{R}^{3(k+1)^2}$（SH系数，$k=3$）和不透明度 $\alpha_i \in [0,1]$（sigmoid约束）。MLP极小以不影响实时渲染。

3. **通过不透明度变化建模运动**: 核心观察（Fig. 2）——面部动态区域存在**重叠的高斯基元**，在不同表情下交替变为可见/不可见。例如闭嘴时唇部高斯不透明度高，张嘴时这些高斯变透明而下巴位置的另一组高斯浮现。这种"过表示"策略避免了显式3D位移带来的优化困难。

4. **其他不可行替代方案的验证**:

    - 直接混合显式参数（无MLP）：动态区域出现严重伪影
    - 预测位置/旋转偏移 $\Delta(\mu, R)$：在3DGS的启发式优化中加入3D运动维度导致几何不一致
    - 表情向量仅作MLP条件（无blending）：单一MLP难以学习所有高斯的全局动态

### 损失函数 / 训练策略

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_1(I_r, I_{gt}) + \lambda_s \mathcal{L}_{\text{SSIM}}(I_r, I_{gt}) + \lambda_p \mathcal{L}_p(I_r, I_{gt})$$

- $\lambda_1=0.8$, $\lambda_s=0.2$, $\lambda_p=0.1$
- 感知损失 $\mathcal{L}_p$ 基于VGG网络，10K迭代后激活，仅在头部bbox区域计算
- 初始化：2500个高斯中心点（来自3DMM网格顶点子集），特征基底 $\mathbf{F}$ 初始化为零
- 密集化在500-15K迭代间执行，总训练50K迭代，单V100约1小时

## 实验关键数据

### 主实验

| 方法 | 数据集 | PSNR↑ | SSIM↑ | LPIPS↓ | 渲染时间(s)↓ |
|------|--------|-------|-------|--------|-------------|
| NHA | INSTA | 26.99 | 0.942 | 0.043 | 0.63 |
| INSTA | INSTA | 28.61 | 0.944 | 0.047 | 0.05 |
| NeRFBlendShape | INSTA | 30.52 | 0.955 | 0.056 | 0.10 |
| PointAvatar | INSTA | 30.68 | 0.952 | 0.058 | 0.1-1.5 |
| **HeadGaS** | **INSTA** | **32.50** | **0.971** | **0.033** | **0.004** |
| NeRFBlendShape | NBS | 34.34 | 0.970 | 0.031 | 0.10 |
| **HeadGaS** | **NBS** | **36.66** | **0.976** | **0.026** | **0.004** |

HeadGaS在INSTA和NBS数据集上PSNR领先约2 dB，渲染速度提升10-25×。

### 消融实验

| 变体 | L2↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-----|-------|-------|--------|
| Ours w/o blending | 0.0012 | 30.28 | 0.955 | 0.041 |
| Ours w/ $\Delta(\mu,R)$ | 0.0014 | 29.83 | 0.953 | 0.045 |
| Ours change all | 0.0014 | 29.65 | 0.951 | 0.041 |
| Ours w/o MLP | 0.0009 | 32.08 | 0.968 | 0.033 |
| Ours w/o $\mathcal{L}_p$ | 0.0008 | 32.11 | 0.969 | 0.046 |
| **Ours (完整)** | **0.0008** | **32.50** | **0.971** | **0.033** |

### 关键发现

- **不透明度建模动态 > 位置变形**: 改变颜色/不透明度（PSNR 32.50）远优于移动高斯位置（29.83），因为3DGS的启发式优化机制下空间运动引入几何不一致
- **特征空间混合 > 显式参数混合**: 潜在特征混合+MLP解码避免了表情权重直接操控颜色/不透明度的表达力限制
- **感知损失的作用**: 移除 $\mathcal{L}_p$ 后LPIPS从0.033退化到0.046，质量损失主要在细节纹理

## 亮点与洞察

- **反直觉但有效的动态建模**: 不移动高斯而是让高斯"出现/消失"来表示运动，简洁而有效
- **与3DMM解耦的设计**: 不绑定特定参数化模型，已验证兼容FLAME和FaceWarehouse两种不同的3DMM
- **极致速度**: 512²分辨率250 fps，比交互式NeRF方法快25-250×
- **跨主体表情迁移**: 无需额外训练即可将一个人的表情参数驱动另一个人的模型

## 局限与展望

- **内存消耗**: 每个高斯存储 $B \times f_{dim}$ 浮点数的特征基底，在表情维度B较大时开销显著
- **依赖头部跟踪质量**: 跟踪器的姿态/表情估计误差会直接传导到渲染结果
- **训练数据覆盖限制**: 若某些表情仅在正面观察到，侧面渲染该表情会失败
- 未探索压缩特征基底的方案，可结合HAC类方法进一步减小模型体积
- 可扩展至全身/手部等更复杂的动态区域

## 相关工作与启发

- **NeRFBlendShape**: 同样使用表情参数混合，但混合的是多级哈希网格场而非单个高斯特征
- **PointAvatar**: 可变形点云表示，几何约束更强但真实感不足
- **INSTA**: 基于FLAME网格变形+InstantNGP，速度接近实时但有网格伪影
- **4D-GS / Dynamic 3DGS**: 通用动态场景方法，HeadGaS针对头部做了更精简的特化设计

## 评分

- **新颖性**: ⭐⭐⭐⭐ — "通过不透明度变化而非几何变形建模面部动态"的insight原创而反直觉
- **实验充分度**: ⭐⭐⭐⭐ — 3个数据集、8个基线对比、完整消融和跨主体迁移实验
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，不透明度变化可视化（Fig 2）非常直观
- **实用价值**: ⭐⭐⭐⭐⭐ — 实时驱动数字头像，AR/VR直接可用

<!-- RELATED:START -->

## 相关论文

- [A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis](a_compact_dynamic_3d_gaussian_representation_for_realtime_dy.md)
- [VersatileGaussian: Real-Time Neural Rendering for Versatile Tasks Using Gaussian Splatting](versatilegaussian_real-time_neural_rendering_for_versatile_tasks_using_gaussian_.md)
- [FastGHA: Generalized Few-Shot 3D Gaussian Head Avatars with Real-Time Animation](../../ICLR2026/3d_vision/fastgha_generalized_few-shot_3d_gaussian_head_avatars_with_real-time_animation.md)
- [TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting](talkinggaussian_structure-persistent_3d_talking_head_synthesis_via_gaussian_spla.md)
- [NGP-RT: Fusing Multi-Level Hash Features with Lightweight Attention for Real-Time Novel View Synthesis](ngp-rt_fusing_multi-level_hash_features_with_lightweight_attention_for_real-time.md)

<!-- RELATED:END -->
