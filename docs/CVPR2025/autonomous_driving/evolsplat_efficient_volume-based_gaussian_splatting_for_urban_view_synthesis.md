---
title: >-
  [论文解读] EVolSplat: Efficient Volume-based Gaussian Splatting for Urban View Synthesis
description: >-
  [CVPR 2025][自动驾驶][3D高斯泼溅] 提出 EVolSplat，一个基于稀疏3D卷积的前馈城市场景3D高斯泼溅方法，通过全局统一体素预测高斯参数（而非像素对齐），结合遮挡感知的基于图像的渲染（IBR）着色，在 KITTI-360 上达到 23.26dB PSNR / 83.81 FPS。
tags:
  - CVPR 2025
  - 自动驾驶
  - 3D高斯泼溅
  - 城市场景
  - 前馈重建
  - 稀疏3D卷积
  - 实时渲染
---

# EVolSplat: Efficient Volume-based Gaussian Splatting for Urban View Synthesis

**会议**: CVPR 2025  
**arXiv**: [2503.20168](https://arxiv.org/abs/2503.20168)  
**代码**: https://xdimlab.github.io/EVolSplat/ (有)  
**领域**: 自动驾驶 / 新视角合成  
**关键词**: 3D高斯泼溅, 城市场景, 前馈重建, 稀疏3D卷积, 实时渲染

## 一句话总结

提出 EVolSplat，一个基于稀疏3D卷积的前馈城市场景3D高斯泼溅方法，通过全局统一体素预测高斯参数（而非像素对齐），结合遮挡感知的基于图像的渲染（IBR）着色，在 KITTI-360 上达到 23.26dB PSNR / 83.81 FPS。

## 研究背景与动机

**领域现状**：城市场景的新视角合成是自动驾驶仿真的核心需求。per-scene 优化方法（如 3DGS、Street Gaussians）需要每场景几十分钟训练；前馈方法（如 MVSplat）速度快但基于像素对齐的高斯预测在城市大场景中存在多视角不一致问题。

**现有痛点**：像素对齐方法将3D高斯关联到每个像素射线上，导致：（1）不同视角预测的高斯位置不一致，多视角融合时冲突；（2）远景和天空区域缺乏合理表示；（3）深度估计误差直接传递到高斯位置。

**核心矛盾**：前馈速度 vs 空间一致性——像素空间操作快但不一致，3D空间操作一致但密集体素计算成本高。

**切入角度**：用稀疏3D卷积在统一全局体素中预测高斯参数，只在有点云的位置分配计算资源。

**核心 idea**：稀疏3D-CNN 全局体素预测 + 遮挡感知IBR着色 + 半球背景高斯 = 一致且高效的城市场景新视角合成。

## 方法详解

### 整体框架

输入多视角图像+单目深度估计生成初始3D点云，构建稀疏体素网格。稀疏3D-CNN 提取几何特征，MLP 预测每个体素的高斯参数（位置偏移/缩放/旋转/不透明度）。颜色通过遮挡感知 IBR 从输入图像的2D纹理中查询获得。远景/天空用半球背景高斯建模。

### 关键设计

1. **稀疏3D-CNN 体素预测**:

    - 功能：在全局3D空间中一致地预测高斯参数
    - 核心思路：将单目深度估计得到的3D点云体素化，用 MinkowskiNet 稀疏卷积提取特征并递归精化位置。MLP 预测位置偏移 $\Delta p$、缩放、旋转和不透明度。位置偏移修正深度估计误差
    - 设计动机：与像素对齐方法不同，全局体素保证了多视角几何一致性，稀疏卷积只在有点云的位置计算，效率接近密集2D方法

2. **遮挡感知基于图像的渲染（IBR）着色**:

    - 功能：从输入图像直接获取高斯颜色，而非网络预测
    - 核心思路：将3D高斯中心投影回输入视角获取2D特征，用可见性图（渲染输入视角检查该高斯是否可见）过滤被遮挡的视角，用注意力机制融合多视角颜色
    - 设计动机：网络预测颜色在城市场景中细节不足，IBR 保留了输入图像的高频纹理。遮挡可见性检查解决了大基线下的颜色不一致问题

3. **半球背景高斯**:

    - 功能：建模远景和天空
    - 核心思路：在场景外创建半球面分布的高斯，用MLP从方向向量预测球谐系数
    - 设计动机：城市场景中天空和远处建筑占大量像素但缺乏深度信息，专门的背景建模避免了前景高斯被"浪费"在远景上

### 损失函数 / 训练策略

$\mathcal{L} = (1-0.2)\mathcal{L}_1 + 0.2\mathcal{L}_{SSIM} + 0.1\mathcal{L}_{entropy}$，熵正则化鼓励不透明度接近0或1（避免半透明伪影）。KITTI-360 上训练160个场景，每场景30对立体图像。

## 实验关键数据

### 主实验

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS |
|------|-------|-------|--------|-----|
| MVSplat | 21.22 | 0.695 | 0.246 | - |
| EDUS | 22.13 | 0.761 | 0.178 | - |
| **EVolSplat** | **23.26** | **0.797** | **0.179** | **83.81** |

Waymo 零样本泛化：PSNR 23.43, SSIM 0.786。

### 消融实验

| 配置 | PSNR | 说明 |
|------|------|------|
| 无 IBR | 21.06 | IBR 贡献 +2.2 dB |
| 无位置偏移 | 22.49 | 偏移修正深度误差 +0.77 dB |
| 无遮挡检查 | 22.97 | 可见性过滤 +0.29 dB |
| 完整模型 | **23.26** | — |

### 关键发现
- **IBR 贡献最大**：从输入图像直接查询颜色比网络预测高 2+ dB，城市场景的高频纹理必须保留
- **零样本泛化到 Waymo**：KITTI-360 训练的模型直接在 Waymo 上达到 23.43 dB，说明全局体素表示具有跨数据集泛化能力

## 亮点与洞察
- **稀疏3D卷积的效率突破**——只在有点云的体素位置计算，避免了密集3D卷积的内存爆炸
- **IBR vs 网络预测颜色**——城市场景纹理细节丰富，直接从图像查颜色远优于用网络"想象"颜色

## 局限与展望
- 不处理动态物体（移动车辆/行人）
- 依赖单目深度估计质量
- 固定体素大小可能不适应所有场景尺度

## 相关工作与启发
- **vs MVSplat**: 像素对齐导致多视角不一致，EVolSplat 的全局体素天然一致
- **vs per-scene 优化**: 前馈推理无需训练，速度快几个数量级但质量稍低

## 评分
- 新颖性: ⭐⭐⭐⭐ 稀疏3D卷积+IBR 的组合设计在城市场景中效果出色
- 实验充分度: ⭐⭐⭐⭐ KITTI-360 + Waymo 零样本，充分的消融
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图表说服力强
- 价值: ⭐⭐⭐⭐ 为城市场景前馈3DGS提供了强基线

<!-- RELATED:START -->

## 相关论文

- [Extrapolated Urban View Synthesis Benchmark](../../ICCV2025/autonomous_driving/extrapolated_urban_view_synthesis_benchmark.md)
- [LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)
- [GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction](gaussianformer-2_probabilistic_gaussian_superposition_for_efficient_3d_occupancy.md)
- [Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](spectral-geometric_neural_fields_for_pose-free_lidar_view_synthesis.md)
- [3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation](../../ICCV2025/autonomous_driving/3d_gaussian_splatting_driven_multi-view_robust_physical_adversarial_camouflage_g.md)

<!-- RELATED:END -->
