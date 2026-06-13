---
title: >-
  [论文解读] Global Motion Corresponder for 3D Point-Based Scene Interpolation under Large Motion
description: >-
  [ICCV 2025][3D视觉][场景插值] 提出Global Motion Corresponder (GMC),通过学习将两个时刻的3D Gaussian映射到共享规范空间的一元势场,实现大运动条件下的鲁棒场景插值和外推。 动态场景插值从两组离散多视角帧重建连续运动,是计算机视觉的基础挑战。现有方法都依赖一个关键假设:…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "场景插值"
  - "大运动"
  - "3D Gaussian Splatting"
  - "SE(3)变换"
  - "语义对应"
---

# Global Motion Corresponder for 3D Point-Based Scene Interpolation under Large Motion

**会议**: ICCV 2025  
**arXiv**: [2508.20136](https://arxiv.org/abs/2508.20136)  
**代码**: [项目页面](https://junrul.github.io/gmc/)  
**领域**: 3D视觉  
**关键词**: 场景插值, 大运动, 3D Gaussian Splatting, SE(3)变换, 语义对应

## 一句话总结

提出Global Motion Corresponder (GMC),通过学习将两个时刻的3D Gaussian映射到共享规范空间的一元势场,实现大运动条件下的鲁棒场景插值和外推。

## 研究背景与动机

动态场景插值从两组离散多视角帧重建连续运动,是计算机视觉的基础挑战。现有方法都依赖一个关键假设:**相邻时间步之间运动足够小**,使得位移可以被局部线性模型近似。

当运动较大时,这一假设根本性地崩溃:

**局部邻域搜索失效** — 点的局部邻域变得不可靠,物体已远离原始位置

**全局匹配歧义** — 一个点可能有多个合理匹配,使点对点对应变得模糊

**交叉匹配** — 简单最近邻匹配导致criss-cross匹配,无法用于场景插值

现有方法分为变形场方法(4DGS, Deformable-3DGS)和迭代细化方法(Dynamic Gaussian, PAPR In Motion),两者在大运动下都会失败。

## 方法详解

### 核心思路

用**一元势场**(Unary Potential Fields)代替直接的点对点匹配,为每个时刻学习SE(3)变换,将两组Gaussian映射到共享规范空间:

$$\underset{\hat{\boldsymbol{\mu}}^{(0)}_i}{\underbrace{\boldsymbol{R}^{(0)}_i\boldsymbol{\mu}_i^{(0)}+\boldsymbol{t}^{(0)}_i}} = \underset{\hat{\boldsymbol{\mu}}^{(1)}_j}{\underbrace{\boldsymbol{R}^{(1)}_j\boldsymbol{\mu}_j^{(1)}+\boldsymbol{t}^{(1)}_j}}$$

### 一元势场设计

$\mathcal{F}(\tilde{\boldsymbol{f}}, \boldsymbol{\mu}) = (\boldsymbol{R}, \boldsymbol{t})$

输入PCA投影的DINO语义特征和3D坐标,输出SE(3)变换。利用MLP的归纳偏置保证输出平滑性:语义相似的点自然预测相似的变换。每个时刻需要独立的MLP($\mathcal{F}_0, \mathcal{F}_1$)。

### 能量损失

$$E_{i,j} = w_c\|\boldsymbol{c}_i - \boldsymbol{c}_j\|_2^2 + w_f\|\boldsymbol{f}_i - \boldsymbol{f}_j\|_2^2 + w_\mu\|\hat{\boldsymbol{\mu}}_i - \hat{\boldsymbol{\mu}}_j\|_2^2$$

关键区别:使用**规范空间中的**空间距离替代原始欧氏空间中的距离。

双向损失确保所有点都找到对应:
$$\mathcal{L}_E = \sum_{g_i \in \mathcal{G}_0} \min_{g_j \in \mathcal{G}_1} E_{i,j} + \sum_{g_j \in \mathcal{G}_1} \min_{g_i \in \mathcal{G}_0} E_{j,i}$$

### 局部等距损失

$$\mathcal{L}_{iso} = \frac{1}{kN}\sum_{g_i}\sum_{g_j \in NN_i}\left|\|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\|_2^2 - \|\hat{\boldsymbol{\mu}}_i - \hat{\boldsymbol{\mu}}_j\|_2^2\right|$$

保持局部几何关系,促进局部刚性变换。总损失: $\mathcal{L} = \mathcal{L}_E + \alpha \mathcal{L}_{iso}$

### 联合细化与防退化

训练后联合更新GMC和Gaussian集合,使用渲染损失进一步优化。对位置输入 $\boldsymbol{\mu}$ 施加Dropout防止所有点坍缩到原点的退化解。

## 实验

### 场景插值定量对比 (SI-FID↓)

| 方法 | Ball | Boat | Butterfly | Car | Dolphin | Knight | Microwave | Seagull |
|------|------|------|-----------|-----|---------|--------|-----------|---------|
| 4DGS | – | 328.8 | – | 460.0 | – | – | 258.2 | 294.0 |
| Deformable-3DGS | – | 811.1 | – | 800.1 | – | – | 709.6 | 633.6 |

(注："-"表示完全无法产生合理渲染)

### 外推能力

GMC支持运动外推,而所有baseline方法均无法实现。可以预测超出给定两个时刻范围的运动。

### 关键发现

1. 在大运动场景中,现有基于变形场或迭代细化的方法完全失败
2. 语义特征(DINO)对建立初始"软"对应至关重要
3. 双向损失是必要的,单向损失会导致部分点没有对应
4. 局部等距约束需要从零开始逐渐增加权重,先建立对应再强制刚性

## 亮点与洞察

1. **问题重新定义** — 将大运动下的场景插值转化为全局对应建立问题
2. **规范空间的巧妙利用** — 通过将两组Gaussian映射到共享空间,避免了直接的点对点匹配
3. **DINO语义的有效利用** — 语义特征帮助在大运动下建立有意义的初始匹配
4. **外推能力** — 作为副产品,自然支持运动外推

## 局限性

- 需要每个时刻的多视角图像重建3DGS作为输入
- 假设场景中的运动近似局部刚性
- 对于非常复杂的拓扑变化(如流体)可能不适用

## 相关工作

- **变形场方法**: 4DGS, Deformable-3DGS
- **迭代细化方法**: Dynamic Gaussian, PAPR In Motion
- **视觉特征对应**: DINO, Zero-Shot 3D Shape Correspondence
- **点云插值**: 传统方法在大运动下同样失败

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (规范空间+一元势场的设计非常巧妙)
- 技术深度: ⭐⭐⭐⭐ (能量函数和各损失设计合理)
- 实验充分度: ⭐⭐⭐⭐ (展示了插值和外推能力)
- 实用价值: ⭐⭐⭐⭐ (解决了大运动场景的实际痛点)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images](comogaussian_continuous_motionaware_gaussian_splatting_from.md)
- [\[ICCV 2025\] Estimating 2D Camera Motion with Hybrid Motion Basis](estimating_2d_camera_motion_with_hybrid_motion_basis.md)
- [\[ICCV 2025\] SceneMI: Motion In-betweening for Modeling Human-Scene Interactions](scenemi_motion_in-betweening_for_modeling_human-scene_interaction.md)
- [\[ICCV 2025\] Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image](image_as_an_imu_estimating_camera_motion_from_a_single_motion-blurred_image.md)
- [\[ECCV 2024\] TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos](../../ECCV2024/3d_vision/tram_global_trajectory_and_motion_of_3d_humans_from_in-the-wild_videos.md)

</div>

<!-- RELATED:END -->
