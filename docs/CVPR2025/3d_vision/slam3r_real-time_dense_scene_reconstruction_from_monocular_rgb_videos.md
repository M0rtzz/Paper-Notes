---
title: >-
  [论文解读] SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos
description: >-
  [CVPR 2025][3D视觉][实时稠密重建] SLAM3R提出了一个两层级的前馈神经网络系统，通过Image-to-Points(I2P)网络从视频片段直接回归局部3D点图，再通过Local-to-World(L2W)网络渐进式对齐到全局坐标系，全程无需显式求解相机参数，在20+ FPS下实现了SOTA的稠密重建精度和完整度。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "实时稠密重建"
  - "端到端3D重建"
  - "前馈神经网络"
  - "无参数SLAM"
  - "视频重建"
---

# SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos

**会议**: CVPR 2025  
**arXiv**: [2412.09401](https://arxiv.org/abs/2412.09401)  
**代码**: [GitHub](https://github.com/PKU-VCL-3DV/SLAM3R)  
**领域**: 3D视觉 / 稠密重建  
**关键词**: 实时稠密重建, 端到端3D重建, 前馈神经网络, 无参数SLAM, 视频重建

## 一句话总结

SLAM3R提出了一个两层级的前馈神经网络系统，通过Image-to-Points(I2P)网络从视频片段直接回归局部3D点图，再通过Local-to-World(L2W)网络渐进式对齐到全局坐标系，全程无需显式求解相机参数，在20+ FPS下实现了SOTA的稠密重建精度和完整度。

## 研究背景与动机

稠密3D重建长期面临精度、完整度和效率三者难以兼顾的困难。传统方法依赖SfM+MVS的多阶段流水线，虽重建质量高但需离线处理。现有单目稠密SLAM系统通过神经隐式或3DGS表示改善质量，但运行速度远低于实时（如NICER-SLAM不到1 FPS）。

DUSt3R开创了端到端的稠密重建方向，但多视图扩展需要穷举图像对配和全局优化，效率低下。Spann3R通过增量管线加速但产生明显的累积漂移。SLAM3R的核心思路是：在每个层级都使用多帧输入来最小化漂移，同时通过自包含的检索模块引用长时历史中的相似帧作为全局参考。

## 方法详解

### 整体框架

SLAM3R使用滑动窗口将输入视频转为重叠片段。I2P网络处理每个窗口内的帧，选择关键帧定义局部坐标系，回归所有帧的稠密3D点图。L2W网络将局部重建渐进式融合到全局坐标系。两个模块共享相似的ViT架构，整个过程不显式求解任何相机参数。

### 关键设计1: Image-to-Points多视图扩展

- **功能**: 将DUSt3R从双视图扩展到多视图，直接从视频窗口预测稠密3D点图
- **核心思路**: 使用共享编码器$E_{img}$独立编码每帧，关键帧解码器$D_{key}$引入**多视图交叉注意力**——将关键帧查询与每个支持帧的键值独立做交叉注意力，再通过max-pooling聚合多视图信息。支持帧解码器$D_{sup}$沿用DUSt3R架构仅与关键帧交互。默认选中间帧为关键帧（与其他帧重叠最大）
- **设计动机**: DUSt3R原始的双视图设计在多视图场景下需要穷举配对，效率低。多视图交叉注意力允许同时处理任意数量的支持帧，独立交叉注意力+max-pooling的设计简洁高效

### 关键设计2: Local-to-World渐进式全局配准

- **功能**: 将局部重建对齐到全局坐标系，消除累积漂移
- **核心思路**: 维护有限容量$B$的缓冲集存储已配准帧，使用reservoir采样策略。注册新帧时用检索模块（共享I2P前$r$个decoder block + 线性投影 + 平均池化）从缓冲集中选top-$K$最相关场景帧。将3D点图通过patch嵌入编码为几何token与视觉token相加，送入配准解码器$D_{reg}$和场景解码器$D_{sce}$
- **设计动机**: Spann3R的逐帧增量导致严重漂移。SLAM3R通过多帧参考和长时历史检索提供更全局的参考，有效减少漂移

### 关键设计3: 自包含的检索与场景初始化

- **功能**: 高效选择最佳参考帧，确保场景初始化的准确性
- **核心思路**: 检索模块在特征空间衡量视觉相似度和基线适宜度，选top-K场景帧。场景初始化时对第一个窗口执行$L$次I2P（遍历每帧作为关键帧），选择总置信度最高的结果
- **设计动机**: retrieval模块复用I2P的decoder block，零额外参数。初始化的准确性对全局重建至关重要

### 损失函数

I2P损失：$\mathcal{L}_{I2P} = \sum_{i=1}^{L} M_i \cdot (\hat{C}_i \cdot \text{L1}(\frac{1}{\hat{z}}\hat{X}_i, \frac{1}{z}X_i) - \alpha \log \hat{C}_i)$，使用置信度加权的L1距离和归一化尺度。L2W损失类似但不做归一化（输出需对齐场景帧尺度）。

## 实验关键数据

### 主实验: 7-Scenes数据集重建质量 (Acc./Comp. cm)

| 方法 | Avg Acc↓ | Avg Comp↓ | FPS |
|------|---------|----------|-----|
| DUSt3R | 2.19 | 3.24 | ≪1 |
| MASt3R | 3.04 | 3.90 | ≪1 |
| Spann3R | 3.42 | 2.41 | >50 |
| **SLAM3R** | **1.63** | **1.31** | **~25** |

### 消融: 置信度过滤的效果

| 配置 | Avg Acc↓ | Avg Comp↓ |
|------|---------|----------|
| SLAM3R-NoConf | 2.40 | 2.24 |
| **SLAM3R** | **1.63** | **1.31** |

### 关键发现

- 精度(Acc)和完整度(Comp)均大幅领先DUSt3R和Spann3R
- 25 FPS实现实时性能，比DUSt3R全局优化快几个数量级
- 相比Spann3R不到一半的漂移
- 置信度过滤去除不可靠点后Acc从2.40降至1.63

## 亮点与洞察

1. **无相机参数的稠密SLAM**：完全跳过相机参数估计，直接在统一坐标系中预测3D点图，概念上的简化带来效率和质量的双重提升
2. **两层级共享架构**：I2P和L2W使用相似的多视图交叉注意力架构，设计一致性强
3. **检索式长时记忆**：reservoir采样+特征检索实现了对任意长视频的可扩展处理

## 局限与展望

- 需要大规模数据集训练，模型泛化到分布外场景的能力有待验证
- 目前仅处理静态场景
- 滑动窗口策略限制了可处理的帧间运动幅度
- 未来可结合动态场景重建和更大规模的训练数据

## 相关工作与启发

- **DUSt3R**: 开创了端到端稠密3D重建，SLAM3R在其基础上扩展至多视图和增量式
- **Spann3R**: 并行工作以空间记忆扩展DUSt3R到视频，但累积漂移严重
- **DROID-SLAM**: 迭代更新位姿和深度，但重建质量不如SLAM3R
- 启发：前馈式3D预测+渐进式融合可能是密集重建的未来范式

## 评分

⭐⭐⭐⭐⭐ — 在稠密3D重建的三个核心指标（精度、完整度、效率）上同时达到最优，20+ FPS的实时性能具有重要实用价值。两层级框架设计清晰，解耦了局部重建和全局配准问题。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ODHSR: Online Dense 3D Reconstruction of Humans and Scenes from Monocular Videos](odhsr_online_dense_3d_reconstruction_of_humans_and_scenes_from_monocular_videos.md)
- [\[CVPR 2025\] MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors](mast3r-slam_real-time_dense_slam_with_3d_reconstruction_priors.md)
- [\[CVPR 2025\] SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video](splinegs_robust_motion-adaptive_spline_for_real-time_dynamic_3d_gaussians_from_m.md)
- [\[ICCV 2025\] FROSS: Faster-than-Real-Time Online 3D Semantic Scene Graph Generation from RGB-D Images](../../ICCV2025/3d_vision/fross_faster-than-real-time_online_3d_semantic_scene_graph_generation_from_rgb-d.md)
- [\[CVPR 2025\] Mobile-GS: Real-time Gaussian Splatting for Mobile Devices](mobile-gs_real-time_gaussian_splatting_for_mobile_devices.md)

</div>

<!-- RELATED:END -->
