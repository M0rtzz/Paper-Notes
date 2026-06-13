---
title: >-
  [论文解读] GaussianUpdate: Continual 3D Gaussian Splatting Update for Changing Environments
description: >-
  [ICCV 2025][3D视觉][持续学习] 提出GaussianUpdate，首次将3D高斯表示与持续学习结合，通过三阶段更新策略（外观更新→几何布局更新→联合精炼）和可见性感知生成式回放，实现时变场景的实时渲染和变化可视化。 真实场景随时间变化（物体增减、光照变化），但现有方法存在根本问题： 重新训练成本高：每次变化都…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "持续学习"
  - "3D高斯"
  - "场景更新"
  - "变化检测"
  - "生成式回放"
---

# GaussianUpdate: Continual 3D Gaussian Splatting Update for Changing Environments

**会议**: ICCV 2025  
**arXiv**: [2508.08867](https://arxiv.org/abs/2508.08867)  
**代码**: [项目页](https://zju3dv.github.io/GaussianUpdate)  
**领域**: 3D视觉  
**关键词**: 持续学习, 3D高斯, 场景更新, 变化检测, 生成式回放

## 一句话总结

提出GaussianUpdate，首次将3D高斯表示与持续学习结合，通过三阶段更新策略（外观更新→几何布局更新→联合精炼）和可见性感知生成式回放，实现时变场景的实时渲染和变化可视化。

## 研究背景与动机

真实场景随时间变化（物体增减、光照变化），但现有方法存在根本问题：

**重新训练成本高**：每次变化都重新训练模型，存储空间随时间增长

**灾难性遗忘**：直接用新数据更新会丢失历史场景信息

**动态建模不适用**：4D方法假设连续动态变化，无法处理物体突然出现/消失的离散变化

**NeRF-based方法低效**：CLNeRF等无法定位变化区域，渲染效率低

GaussianUpdate的优势：显式3D高斯表示天然支持精确定位和管理变化的高斯基元。

## 方法详解

### 全局外观模型

用4D哈希网格 $\mathbf{H}$ 和小型MLP $\mathbf{F}$ 建模光照变化：

$$\{\Delta s_i^t, \Delta Y_i^t\} = F(H\{\mu_i, t\})$$

为每个高斯推断时间依赖的增量缩放和球谐属性。

### 三阶段更新策略

**第一阶段：全局外观更新**

在布局不变区域学习光照变化：
1. 渲染上一时刻图像 $I_f^{t-1}$，与新图像 $I_f^t$ 对比
2. 用SAM获取实例分割mask，计算IoU确定布局不变区域 $M_f$
3. 仅在不变区域优化哈希编码：
$$\mathcal{L}_{st} = (1-\lambda)\mathcal{L}_1 \cdot \mathcal{M}_f + \lambda\mathcal{L}_{D-SSIM} \cdot \mathcal{M}_f$$

**第二阶段：几何布局更新**

固定外观模型，处理物体增减：
- **新物体**：从COLMAP生成新高斯基元 $G_{add}$
- **消失物体**：学习移除因子 $m$，通过陡峭sigmoid激活：
$$\psi(m) = \frac{1}{1 + e^{-1000m}}$$

- 正则化驱动 $\psi(m)$ 趋向0或1：
$$\mathcal{L}_{reg} = \lambda_2(1-\psi(m))\psi(m) + \lambda_3 BCE(\psi(m), 1)$$

- DBSCAN过滤稀疏离群点

**第三阶段：联合精炼**

剪枝后联合优化外观模型和新高斯参数。重要性剪枝减少内存：
$$v_i = \max_{n \in N_1}(\alpha_i^n T_i^n)$$

### 可见性感知持续学习

- **可见性池** $P_v$：存储所有高斯在整个时间段的状态（活跃/非活跃）
- **生成式回放**：利用高斯渲染的高效性，用历史相机位姿重新渲染过去场景作为训练数据
- 无需存储额外图像，仅需记录相机外参

## 实验

### 基准数据集对比

| 方法 | Breville PSNR↑ | Kitchen PSNR↑ | Living room PSNR↑ |
|------|---------------|---------------|-------------------|
| Baseline | 20.66 | 16.44 | 17.28 |
| CLNeRF | 28.02 | 28.40 | 24.58 |
| 4DGS | 28.92 | 27.03 | 24.41 |
| **Ours** | **30.11** | **28.02** | **26.10** |
| Upper Bound | 30.36 | 27.99 | 26.22 |

GaussianUpdate在所有场景中超越CLNeRF和4DGS，接近上界（每个时刻独立训练）。

| 方法 | Community PSNR↑ | Community SSIM↑ |
|------|-----------------|-----------------|
| CLNeRF | 22.88 | 0.629 |
| 4DGS | 22.99 | 0.711 |
| **Ours** | **23.88** | **0.764** |

### 关键优势

- **实时渲染**：基于3DGS的显式表示确保实时渲染
- **变化可视化**：可见性池支持在不同时刻渲染场景并可视化变化
- 接近Upper Bound说明信息保留充分

## 亮点与洞察

1. **三阶段解耦策略**：避免外观和几何变化的耦合歧义
2. **显式变化管理**：区分三种变化类型（光照、消失、新增）分别处理
3. **生成式回放免存储**：利用高斯渲染效率，仅存相机位姿即可回放历史
4. **可见性池设计**：显式表示的天然优势，支持基元级场景管理

## 局限性

- COLMAP-based添加策略依赖特征匹配质量
- SAM检测布局不变区域可能不够精确
- 仅支持离散时间步更新，不支持连续动态
- 哈希网格随时间步增多带来存储增长

## 相关工作

- CLNeRF: NeRF持续学习
- 4DGS: 动态3D高斯
- NeRF-w: 外观编码处理光照变化

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (3DGS+持续学习首创)
- 技术深度: ⭐⭐⭐⭐⭐ (三阶段策略+可见性池+生成回放)
- 实验充分度: ⭐⭐⭐⭐ (10个场景+多基线对比)
- 实用价值: ⭐⭐⭐⭐⭐ (实时渲染+变化可视化极具应用价值)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization](cl-splats_continual_learning_of_gaussian_splatting_with_local_optimization.md)
- [\[ICCV 2025\] TRAN-D: 2D Gaussian Splatting-based Sparse-view Transparent Object Depth Reconstruction via Physics Simulation for Scene Update](2d_gaussian_splattingbased_sparseview_transparent_object_dep.md)
- [\[CVPR 2025\] WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments](../../CVPR2025/3d_vision/wildgs-slam_monocular_gaussian_splatting_slam_in_dynamic_environments.md)
- [\[CVPR 2025\] ProtoDepth: Unsupervised Continual Depth Completion with Prototypes](../../CVPR2025/3d_vision/protodepth_unsupervised_continual_depth_completion_with_prototypes.md)
- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
