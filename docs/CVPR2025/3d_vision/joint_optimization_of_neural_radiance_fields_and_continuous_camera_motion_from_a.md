---
title: >-
  [论文解读] Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video
description: >-
  [CVPR 2025][3D视觉][NeRF] 将相机运动建模为时间连续的角速度和线速度，通过速度积分避免直接优化大范围 camera-to-world 变换，结合时间依赖 NeRF 和 SDF flow 约束，无需深度先验即可从单目视频联合优化位姿和场景几何。
tags:
  - CVPR 2025
  - 3D视觉
  - NeRF
  - 位姿估计
  - continuous motion
  - SDF
  - velocity integration
  - prior-free
---

# Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video

**会议**: CVPR 2025  
**arXiv**: [2504.19819](https://arxiv.org/abs/2504.19819)  
**代码**: [HoangChuongNguyen/cope-nerf](https://github.com/HoangChuongNguyen/cope-nerf)  
**领域**: 3d_vision  
**关键词**: NeRF, camera pose estimation, continuous motion, SDF, velocity integration, prior-free

## 一句话总结

将相机运动建模为时间连续的角速度和线速度，通过速度积分避免直接优化大范围 camera-to-world 变换，结合时间依赖 NeRF 和 SDF flow 约束，无需深度先验即可从单目视频联合优化位姿和场景几何。

## 研究背景与动机

**领域现状**: NeRF 在新视角合成方面表现出色，但训练需要精确的预计算相机位姿（通常来自 COLMAP），这带来额外计算开销且缺乏可微性。

**现有痛点**:
- NeRFmm、BARF 等联合优化方法受限于前向场景或需要良好初始化
- NoPe-NeRF 依赖预训练深度网络作为先验
- 所有逐帧独立优化位姿的方法都在大范围相机运动场景下挣扎，因为 camera-to-world 变换可能非常大

**核心矛盾**: 逐帧独立优化位姿需要直接估计 camera-to-world 的大变换，这在优化初期极易陷入局部最优；而视频的时间连续性信息未被充分利用。

**本文目标**: 无需任何先验（无深度、无初始位姿），仅从单目视频的 RGB 图像和相机内参联合优化相机运动和 3D 场景。

**切入角度**: 将离散位姿优化转化为连续运动估计——用 MLP 预测每个时刻的角速度和线速度，通过积分得到任意两帧间的相对变换。

## 方法详解

### 整体框架

1. **运动网络** $\phi_v(t)$ 预测每个时刻的角速度 $\boldsymbol{\omega}(t)$ 和线速度 $\mathbf{v}(t)$
2. **时间依赖 NeRF** $(\phi_g, \phi_c)$ 表示每个时刻 $t$ 的局部场景几何（SDF）和外观
3. 通过速度积分（Euler 方法）计算任意两帧间的相对变换 $\mathbf{P}_{t_1 \to t_2}$
4. 多种一致性损失约束场景几何和相机运动的一致性
5. 后期固定位姿，在 world frame 训练完整场景 NeRF

### 关键设计

**1. 连续相机运动建模**
- **功能**: 用 MLP 将时间 $t$ 映射为 $(\boldsymbol{\omega}(t), \mathbf{v}(t)) \in \mathbb{R}^6$，通过 Euler 方法积分得到旋转矩阵和平移向量。
- **核心思路**: $\mathbf{R}_{t\to t+l} = \prod_{u=0}^{U-1} \psi(\boldsymbol{\omega}(t+u\Delta t)\Delta t)$，$\mathbf{t}_{t\to t+l} = \sum_{u=0}^{U-1} \mathbf{v}(t+u\Delta t)\Delta t$，步长 $\Delta t = l/U$，使用 10 个子区间。
- **设计动机**: 将大运动分解为无穷小增量的累积，极大简化了大范围运动的学习。不同于逐帧独立优化或 SLAM 式的相邻帧相对变换，连续建模天然支持跨帧约束和平滑性。

**2. 时间依赖 NeRF**
- **功能**: NeRF 以 $(\mathbf{x}, t)$ 为输入预测 SDF $s(\mathbf{x},t)$ 和颜色 $\mathbf{c}(\mathbf{x},t)$，在每个时刻 $t$ 的局部相机坐标系下定义场景。
- **核心思路**: 训练初期位姿噪声大时，每个时刻的 NeRF 只需解释邻近帧的局部场景——这比在全局坐标系下用噪声位姿训练更稳定。
- **设计动机**: 避免了早期噪声位姿导致 3D 点映射到全局坐标系后的累积误差。时间依赖设计还使得 SDF flow 约束成为可能。

**3. SDF Flow 与运动一致性约束**
- **功能**: 利用 SDF 时间导数与相机运动的线性关系约束场景和运动的一致性：$\mathcal{L}_{flow} = \left|\frac{\partial s}{\partial t} + (\boldsymbol{\omega} \times \mathbf{x} + \mathbf{v})^T \mathbf{n}\right|$。
- **核心思路**: 对于静态场景，表面点的 SDF 在相机运动下满足刚体约束，SDF 的时间变化率等于相机速度在表面法线上的投影。
- **设计动机**: 这个物理约束强制每个时刻的场景表面变化与相机运动一致，防止时间依赖 NeRF 在各时刻独立"漂移"。

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{rgb} + \lambda_1 \mathcal{L}_{eik} + \lambda_2 \mathcal{L}_{flow} + \lambda_3 \mathcal{L}_{photo} + \lambda_4 \mathcal{L}_{sdf}$$

- $\mathcal{L}_{rgb}$: 渲染颜色 L2 损失
- $\mathcal{L}_{eik}$: Eikonal 正则化（SDF 梯度模为 1）
- $\mathcal{L}_{flow}$: SDF 与运动一致性
- $\mathcal{L}_{photo}$: 光度一致性（投影到邻帧的颜色差异 L1）
- $\mathcal{L}_{sdf}$: 各时刻 SDF 与 world frame SDF 的一致性（前 200 epoch 权重为 0，渐增）

两阶段训练：早期联合优化运动+时间依赖 NeRF；后期固定位姿，仅在 world frame 训练全场景 NeRF 5000 epoch。

## 实验关键数据

### 主实验——深度评估

| 方法 | 深度先验 | Co3D AbRel↓ | Co3D δ₁↑ | ScanNet AbRel↓ | ScanNet δ₁↑ |
|---|---|---|---|---|---|
| NeRFmm | 无 | 0.293 | 0.464 | 0.319 | 0.486 |
| NoPe-NeRF | 有 | 0.176 | 0.721 | 0.141 | 0.828 |
| CF3DGS | 有 | 0.211 | 0.732 | 0.157 | 0.803 |
| **Ours** | **无** | **0.031** | **0.975** | **0.063** | **0.952** |

深度误差降低超过 82%（Co3D）和 78%（ScanNet）。

### 主实验——位姿评估

| 方法 | Co3D RPE_t↓ | Co3D RPE_r↓ | Co3D ATE↓ |
|---|---|---|---|
| NeRFmm | 0.500 | 2.785 | 0.054 |
| NoPe-NeRF | 0.281 | 1.449 | 0.050 |
| CF3DGS | 0.097 | 0.402 | 0.011 |
| **Ours** | **0.024** | **0.064** | **0.002** |

所有场景位姿误差均最低。

### 消融实验（Co3D 平均）

| 设置 | AbRel↓ | δ₁↑ | RPE_t↓ | PSNR↑ |
|---|---|---|---|---|
| Full | **0.031** | **0.975** | **0.023** | 27.49 |
| w/o $\mathcal{L}_{flow}$ | 0.084 | 0.877 | 0.114 | 26.31 |
| w/o $\mathcal{L}_{photo}$ | 0.399 | 0.386 | 0.216 | 21.79 |
| w/o 时间依赖 NeRF | 0.287 | 0.504 | 0.299 | 24.08 |
| w/o 运动网络（逐帧优化） | 0.080 | 0.912 | 0.028 | 26.23 |

### 关键发现

1. **光度一致性最关键**: 去掉 $\mathcal{L}_{photo}$ 后 AbRel 恶化 12.9×，PSNR 下降 5.7dB。
2. **时间依赖 NeRF 必不可少**: 直接在全局坐标系训练（去掉时间依赖）导致位姿误差暴增 13×。
3. **SDF flow 约束显著提升位姿精度**: 去掉后 RPE_t 从 0.023 升至 0.114（5×）。
4. **连续运动优于逐帧优化**: 运动网络相比逐帧位姿在深度和 PSNR 上均有提升。

## 亮点与洞察

- 连续运动建模从根本上解决了大范围运动下联合优化的难题——将"大跳跃"分解为"小积分"
- 时间依赖 NeRF + SDF flow 约束的配合非常巧妙：前者提供局部稳定性，后者保证跨时间一致性
- 完全 prior-free，却在位姿和深度上大幅超越依赖深度先验的方法
- 两阶段训练策略合理：先学准运动，再精修完整场景

## 局限与展望

- 仅适用于静态场景，动态物体会违反刚体假设
- 基于 NeRF 的方法训练速度较慢
- 相机内参假设已知（未处理变焦等情况）
- 未与更新的 3DGS+位姿方法（如 InstantSplat）对比
- 可探索将连续运动建模扩展到 3DGS 框架以加速

## 相关工作与启发

- **NoPe-NeRF**: 需要深度先验的联合优化 baseline，本文无需任何先验即大幅超越
- **CF3DGS**: 3DGS 联合优化方法，在几何精度上不如本文的 SDF-based 方案
- **BARF / NeRFmm**: 早期联合优化工作，受限于前向场景或初始化要求
- **CasualSAM (Li et al.)**: SDF flow 与 scene flow 线性关系的理论来源

## 评分

⭐⭐⭐⭐ — 连续运动建模+时间依赖 NeRF 的思路原创性强，实验结果全面碾压 prior-based 方法；但限于静态场景和 NeRF 框架，实用性有一定局限。

<!-- RELATED:START -->

## 相关论文

- [Exploiting Deblurring Networks for Radiance Fields](exploiting_deblurring_networks_for_radiance_fields.md)
- [RelationField: Relate Anything in Radiance Fields](relationfield_relate_anything_in_radiance_fields.md)
- [PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](pbr-nerf_inverse_rendering_with_physics-based_neural_fields.md)
- [4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)
- [Dynamic Neural Radiance Field from Defocused Monocular Video](../../ECCV2024/3d_vision/dynamic_neural_radiance_field_from_defocused_monocular_video.md)

<!-- RELATED:END -->
