---
title: >-
  [论文解读] CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images
description: >-
  [CVPR 2025][3D视觉][3D高斯溅射] 提出CoCoGaussian，利用物理摄影散焦原理（弥散圆/Circle of Confusion）在3D高斯溅射框架中建模散焦模糊，仅使用散焦图像即可精确重建3D场景并渲染清晰的新视角图像。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D高斯溅射
  - 散焦模糊
  - 弥散圆
  - 新视角合成
  - 深度估计
---

# CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images

**会议**: CVPR 2025  
**arXiv**: [2412.16028](https://arxiv.org/abs/2412.16028)  
**代码**: [https://Jho-Yonsei.github.io/CoCoGaussian](https://Jho-Yonsei.github.io/CoCoGaussian)  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 散焦模糊, 弥散圆, 新视角合成, 深度估计

## 一句话总结

提出CoCoGaussian，利用物理摄影散焦原理（弥散圆/Circle of Confusion）在3D高斯溅射框架中建模散焦模糊，仅使用散焦图像即可精确重建3D场景并渲染清晰的新视角图像。

## 研究背景与动机

现实拍摄中，相机有限景深导致的散焦模糊不可避免——小光圈需要更长曝光时间（引发运动模糊），大光圈则导致浅景深散焦。现有3DGS和NeRF方法假设输入为清晰图像，这在实际场景中不现实。已有的处理模糊图像的方法（如Deblur-NeRF、BAGS等）大多依赖纯学习策略而忽略了摄影物理原理，而DoF-NeRF虽引入CoC概念但基于隐式表示，渲染速度慢。本文的核心动机是：**将物理的弥散圆原理与3DGS的显式表示结合**，在保持实时渲染速度的同时精确建模散焦效果。

## 方法详解

### 整体框架

CoCoGaussian基于3DGS框架，为每个基础高斯（Base Gaussian $\mathbf{G}_B$）生成$M$组CoC高斯（$\mathbf{G}_{CoC}$），共产生$(M+1)$组高斯。通过一个MLP $h_\theta$ 预测光圈参数$K$、方向向量$\mathbf{d}$、缩放因子$\beta$、以及尺度/旋转偏移$\delta\mathbf{s}_{CoC}$和$\delta\mathbf{q}_{CoC}$。最终将所有高斯组分别渲染的$(M+1)$张图像通过浅层CNN计算像素级权重进行加权求和，得到最终散焦图像。

### 关键设计

1. **基于3D高斯的CoC直径计算**:
    - 功能：从高斯深度和可学习光圈信息计算弥散圆直径
    - 核心思路：将CoC直径公式简化为 $\sigma(\mu_B) \approx K \times |1/d(\mu_B) - 1/d_F|$，其中$K = f \times D$为可学习标量（焦距×光圈直径），$d_F$为可学习焦平面距离，$d(\mu_B)$为相机到高斯均值的欧几里得距离
    - 设计动机：利用摄影物理先验约束CoC大小，避免纯学习方法的过拟合；$K$和$d_F$设为可学习使模型能自适应不同场景

2. **自适应CoC高斯生成**:
    - 功能：生成围绕基础高斯的CoC形状高斯，并处理深度不可靠情况
    - 核心思路：引入可学习缩放因子$\beta \in (0,1]$，修正偏移量为 $\Delta\mu_{CoC;m} = \frac{\sigma(\mu_B)}{2}\beta_m \mathbf{d}_m$。$\beta$确保CoC高斯位于CoC边界内部，同时当深度不准确导致CoC过大时自适应缩小
    - 设计动机：场景中折射/反射表面会导致高斯深度不可靠，单纯依赖深度计算的CoC直径可能有误；$\beta$降低了对深度的依赖性

3. **加权求和渲染与可定制景深**:
    - 功能：将多组渲染图像融合为最终散焦图像，并支持运行时调整景深和焦平面
    - 核心思路：浅层CNN $\mathcal{F}$ 对$(M+1)$张图像计算softmax归一化的像素级权重$\mathcal{W}$，加权求和得到最终图像。由于模型学习了$K$和$d_F$，可在渲染时调整这些参数实现不同景深效果
    - 设计动机：借鉴blind deblurring思路，将CoC高斯组视为模糊核；可定制景深对AR/VR应用有重要价值

### 损失函数 / 训练策略

损失函数为标准的L1损失与D-SSIM损失的加权组合：

$$\mathcal{L}_{rgb} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{D\text{-}SSIM}$$

其中$\lambda = 0.3$。训练时使用COLMAP获取相机位姿和初始点云，焦平面$d_F$初始化为相机到SfM点云的平均距离以确保稳定训练。

## 实验关键数据

### 主实验

| 数据集 | 指标 | CoCoGaussian | BAGS (之前SOTA) | 提升 |
|--------|------|-------------|-----------------|------|
| Deblur-NeRF 合成 | PSNR/SSIM/LPIPS | 30.84/0.9212/0.0478 | 30.65/0.9128/0.0631 | +0.19/+0.008/-0.015 |
| Deblur-NeRF 真实 | PSNR/SSIM/LPIPS | 23.70/0.7531/0.0825 | 23.48/0.7408/0.0962 | +0.22/+0.012/-0.014 |
| DoF-NeRF 真实 | PSNR/SSIM/LPIPS | 30.14/0.9127/0.0701 | 29.87/0.8816/0.1100 | +0.27/+0.031/-0.040 |

### 消融实验

| 配置 | PSNR | SSIM | LPIPS | 说明 |
|------|------|------|-------|------|
| Baseline (3DGS) | 25.72 | 0.8291 | 0.1817 | 无散焦处理 |
| w/o CoC | 28.29 | 0.8778 | 0.0927 | 无物理CoC约束，过拟合 |
| w/o CoC方向向量 | 28.91 | 0.8893 | 0.0896 | 固定方向受限 |
| w/o CoC缩放因子 | 29.46 | 0.9058 | 0.0793 | 过度依赖深度 |
| w/o 光圈参数K | 27.37 | 0.8510 | 0.1113 | 无法精确计算CoC大小 |
| **完整模型** | **30.14** | **0.9127** | **0.0701** | 所有组件协同 |

### 关键发现

- CoC物理先验是核心：去掉CoC后PSNR下降1.85，说明单靠参数学习会过拟合
- 光圈参数$K$贡献最大：去掉后PSNR下降2.77，是确定CoC大小的关键
- 在全对焦图像（NeRF-LLFF）上也优于基线3DGS（PSNR 27.76 vs 27.10），说明建模微小CoC有助于表示精度

## 亮点与洞察

- **物理先验+学习**的结合思路优雅：用摄影光学公式约束框架，再用学习补偿不确定性
- 场景的**景深和焦平面可在渲染时自由调整**，实现了传统3DGS不具备的创意控制能力
- 方法对全对焦图像也有效，泛化性好

## 局限与展望

- 自适应缩放因子$\beta$目前仅能处理CoC直径被高估的情况（$\beta \leq 1$），低估时无法补偿
- 模型生成$M \times N$个额外高斯，增加了训练和渲染的计算开销
- Deblur-NeRF真实数据集存在清晰/散焦图像间光照差异，影响评估可靠性

## 相关工作与启发

- 与Deblur-NeRF/BAGS等学习式去模糊方法不同，本文**把物理光学先验嵌入3DGS**，是物理+学习混合范式的好例子
- DoF-NeRF虽然也用CoC但基于隐式表示速度慢，CoCoGaussian在保持物理合理性的同时实现了实时渲染
- 可定制景深的能力对AR/VR内容创作有直接应用价值

## 评分

- 新颖性: ⭐⭐⭐⭐ 将物理CoC建模融入3DGS是有趣的思路，但核心是已知光学公式的应用
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、完整消融、全对焦泛化实验、可定制景深展示
- 写作质量: ⭐⭐⭐⭐ 物理原理讲解清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ 解决了实际场景中散焦图像的3D重建问题，且提供景深定制能力

<!-- RELATED:START -->

## 相关论文

- [Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](../../NeurIPS2025/3d_vision/dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [Blurry-Edges: Photon-Limited Depth Estimation from Defocused Boundaries](blurry-edges_photon-limited_depth_estimation_from_defocused_boundaries.md)
- [Improving Gaussian Splatting with Localized Points Management](improving_gaussian_splatting_with_localized_points_management.md)
- [SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)
- [HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting](hybridgs_decoupling_transients_and_statics_with_2d_and_3d_gaussian_splatting.md)

<!-- RELATED:END -->
