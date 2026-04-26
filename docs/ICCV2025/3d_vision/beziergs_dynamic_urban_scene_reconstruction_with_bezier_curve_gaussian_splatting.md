---
title: >-
  [论文解读] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting
description: >-
  [ICCV 2025][3D视觉][3D Gaussian Splatting] 提出用可学习的Bézier曲线建模动态物体运动轨迹的3D高斯溅射方法（BezierGS），摆脱对精确目标标注框的依赖，在Waymo和nuPlan数据集上的动态和静态场景重建均达到SOTA。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D Gaussian Splatting
  - 动态场景重建
  - Bézier曲线
  - 新视角合成
  - 自动驾驶
---

# BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting

**会议**: ICCV 2025  
**arXiv**: [2506.22099](https://arxiv.org/abs/2506.22099)  
**代码**: [github.com/fudan-zvg/BezierGS](https://github.com/fudan-zvg/BezierGS)  
**领域**: 自动驾驶 / 3D场景重建  
**关键词**: 3D Gaussian Splatting, 动态场景重建, Bézier曲线, 新视角合成, 自动驾驶

## 一句话总结

提出用可学习的Bézier曲线建模动态物体运动轨迹的3D高斯溅射方法（BezierGS），摆脱对精确目标标注框的依赖，在Waymo和nuPlan数据集上的动态和静态场景重建均达到SOTA。

## 研究背景与动机

动态3D街景建模是自动驾驶的基础需求，高质量的场景重建可以为闭环评估提供仿真环境。现有方法主要面临以下挑战：
- **基于标注框的方法**（Street Gaussians、OmniRe等）严重依赖人工标注的物体位姿（包括方向和位置），标注误差直接影响重建质量。nuPlan等数据集标注精度不佳时，这类方法性能大幅下降。
- **基于自监督的方法**（S³Gaussian通过时空分解网络隐式建模运动，PVG用周期振动拼接轨迹段）虽然免去标注依赖，但S³Gaussian的隐式建模难以优化，PVG的周期振动假设不符合真实运动规律，且分段轨迹难以利用同一物体的时间一致性。

BezierGS正是针对这些问题提出：用Bézier曲线显式且可学习地建模运动轨迹，兼具灵活性和精确性。

## 方法详解

### 整体框架

场景由静态背景高斯和动态前景高斯两部分组成。静态部分直接采用标准3DGS进行全局优化；动态部分的每个高斯原语通过可学习Bézier曲线建模其运动轨迹。给定时间戳 $\tau$，计算动态高斯位置后与静态高斯合并渲染，再与天空立方体贴图合成最终图像。

### 关键设计

1. **Bézier曲线轨迹建模**:

    - 每个动态物体拥有一条中心轨迹Bézier曲线 $\boldsymbol{\gamma}(t,g) = \sum_{i=0}^{n} b_{i,n}(t) \boldsymbol{p}_i^g$，由 $n+1$ 个可学习控制点定义
    - 每个高斯原语还拥有一条偏移轨迹曲线 $\boldsymbol{\delta}(t)$，表示相对物体中心的偏移
    - 最终位置为 $\boldsymbol{\mu}(\tau,g) = \boldsymbol{\delta}(t) + \boldsymbol{\gamma}(t,g)$，其中 $t = f(\tau,g)$ 是时间到Bézier参数的映射
    - 本文使用标准三次Bézier曲线（$n=3$），已被广泛验证在轨迹建模中的有效性
    - 核心优势：可学习控制点能自动修正标注位姿误差，显式曲线便于利用时间一致性

2. **时间-Bézier映射（Time-to-Bézier）**:

    - 物体沿Bézier曲线的运动是非匀速的，因此需要从时间戳 $\tau$ 到Bézier参数 $t$ 的映射
    - 不同物体的映射不同（速度各异），同样用Bézier曲线建模此映射 $t = f(\tau,g)$
    - 隐式地表达物体速度信息

3. **分组曲线间一致性约束（Inter-Curve Consistency Loss）**:

    - 动态高斯原语自由度过高，单个原语可能偏离所属动态物体，导致新视角下出现漂浮和伪影
    - 对于刚体（如车辆），各部分的运动轨迹偏差幅度应保持恒定
    - 通过约束偏移曲线 $\boldsymbol{\delta}(t)$ 的模长使其与首尾控制点模长均值一致来实现：
    $\mathcal{L}_{icc} = \left\| \|\boldsymbol{\delta}(t)\| - \frac{\|\boldsymbol{p}_0\| + \|\boldsymbol{p}_n\|}{2} \right\|_1$
    - 有效抑制局部几何变化过大的问题

### 损失函数 / 训练策略

总损失由多项组成：

$$\mathcal{L} = (1-\lambda_r)\mathcal{L}_1 + \lambda_r\mathcal{L}_{ssim} + \lambda_o^{sky}\mathcal{L}_o^{sky} + \lambda_{icc}\mathcal{L}_{icc} + \lambda_{dr}\mathcal{L}_{dr} + \lambda_v\mathcal{L}_v + \lambda_d\mathcal{L}_d$$

- **动态渲染损失 $\mathcal{L}_{dr}$**：通过Grounded-SAM获取动态物体mask，对动态高斯的单独渲染结果施加RGB+opacity监督，确保动静分离
- **速度损失 $\mathcal{L}_v$**：利用Bézier曲线可解析求导的优势，直接计算高斯速度并渲染速度图，确保动态运动被限制在动态区域内
- **深度损失 $\mathcal{L}_d$**：利用LiDAR投影的稀疏逆深度图增强几何感知
- **天空opacity损失 $\mathcal{L}_o^{sky}$**：在天空mask区域最小化高斯opacity

超参数设置：$\lambda_r=0.2$, $\lambda_{icc}=0.01$, $\lambda_{dr}=0.1$, $\lambda_v=1.0$, $\lambda_d=1.0$。单卡A6000训练30k迭代。

## 实验关键数据

### 主实验

| 数据集 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | Dyn-PSNR↑ |
|--------|------|-------|-------|--------|-----------|
| Waymo (NVS) | DeformableGS | 29.52 | 0.889 | 0.100 | 24.66 |
| Waymo (NVS) | Street Gaussians | 28.92 | 0.877 | 0.110 | 25.54 |
| Waymo (NVS) | OmniRe | 29.41 | 0.884 | 0.101 | 25.85 |
| Waymo (NVS) | PVG | 29.64 | 0.864 | 0.179 | 24.46 |
| Waymo (NVS) | **BezierGS** | **31.51** | **0.903** | **0.092** | **28.51** |
| nuPlan (NVS) | OmniRe | 26.01 | 0.819 | 0.173 | 23.90 |
| nuPlan (NVS) | PVG | 26.38 | 0.772 | 0.222 | 19.69 |
| nuPlan (NVS) | **BezierGS** | **29.42** | **0.860** | **0.133** | **25.12** |

Waymo上PSNR提升1.87dB，Dyn-PSNR提升2.66dB；nuPlan上PSNR提升3.04dB，LPIPS降低16.35%。

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | Dyn-PSNR↑ |
|------|-------|-------|--------|-----------|
| w/o $\mathcal{L}_{icc}$ | 30.83 | 0.900 | 0.096 | 26.15 |
| w/o $\mathcal{L}_{dr}$ | 30.99 | 0.891 | 0.099 | 28.07 |
| w/o $\mathcal{L}_v$ | 31.40 | 0.901 | 0.094 | 28.29 |
| w/o time-to-Bézier | 31.36 | 0.899 | 0.094 | 27.97 |
| w/ MLP轨迹(DeformableGS) | 29.58 | 0.898 | 0.087 | 24.78 |
| w/ 正弦轨迹(PVG) | 29.65 | 0.877 | 0.099 | 26.27 |
| **BezierGS (完整)** | **31.51** | **0.903** | **0.092** | **28.51** |

### 关键发现

- ICC损失贡献最大，去掉后Dyn-PSNR下降2.36dB，说明曲线间一致性对动态物体建模至关重要
- Bézier曲线相比MLP轨迹和正弦轨迹分别提升约2dB和1.9dB PSNR
- 在nuPlan标注质量差的场景下，BezierGS优势更加明显（能自动纠正位姿误差）

## 亮点与洞察

- Bézier曲线作为参数化曲线，比MLP更容易优化，比周期振动更符合真实运动
- 将传统标注框方法视为BezierGS的特例（偏移在物体坐标系中恒定、框位姿固定），展现了方法的统一性
- 速度可通过Bernstein基函数解析求导获得，使速度约束自然嵌入框架中

## 局限与展望

- 仍需初始化阶段的物体感知（分组依赖3D检测或标注）
- 三次Bézier曲线对极复杂轨迹（如U型转弯）可能需要分段表示
- 天空模型采用简单立方体贴图，对复杂天空场景可能不够精细

## 相关工作与启发

- 可将Bézier轨迹思想推广到机器人或移动物体的轨迹预测任务中
- 曲线间一致性约束的思路可用于其他需要维持物体几何一致性的任务

## 评分
- 新颖性: ⭐⭐⭐⭐ Bézier曲线建模运动轨迹是自然且巧妙的选择，统一了标注框和自监督方法
- 实验充分度: ⭐⭐⭐⭐⭐ 两个大规模benchmark、完整消融、与多种SOTA的定量定性对比
- 写作质量: ⭐⭐⭐⭐ 动机清晰、公式推导完整、图示直观
- 价值: ⭐⭐⭐⭐ 解决了实际中标注精度限制重建质量的痛点，对自动驾驶仿真有直接价值

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curv.md)
- [\[ICCV 2025\] Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction](curve-aware_gaussian_splatting_for_3d_parametric_curve_reconstruction.md)
- [\[ICCV 2025\] Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)
- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](../../NeurIPS2025/3d_vision/dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[ICCV 2025\] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)

<!-- RELATED:END -->
