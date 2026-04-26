---
title: >-
  [论文解读] Unleashing the Temporal Potential of Stereo Event Cameras for Continuous-Time 3D Perception
description: >-
  [ICCV 2025][自动驾驶][事件相机] 提出首个仅依赖立体事件相机的 3D 目标检测框架，通过语义-几何双重滤波模块和目标中心 ROI 对齐，在 blind time 期间实现连续时间 3D 检测，在动态大运动场景下显著优于依赖同步传感器的方法（Ev-3DOD），行人 AP3D 甚至超越使用 LiDAR+RGB+Event 的方法。
tags:
  - ICCV 2025
  - 自动驾驶
  - 事件相机
  - 立体视觉
  - 3D目标检测
  - 连续时间感知
  - 语义-几何双重滤波
---

# Unleashing the Temporal Potential of Stereo Event Cameras for Continuous-Time 3D Perception

**会议**: ICCV 2025  
**arXiv**: [2508.02288](https://arxiv.org/abs/2508.02288)  
**代码**: [GitHub](https://github.com/mickeykang16/Ev-Stereo3D)  
**领域**: 自动驾驶  
**关键词**: 事件相机, 立体视觉, 3D目标检测, 连续时间感知, 语义-几何双重滤波

## 一句话总结
提出首个仅依赖立体事件相机的 3D 目标检测框架，通过语义-几何双重滤波模块和目标中心 ROI 对齐，在 blind time 期间实现连续时间 3D 检测，在动态大运动场景下显著优于依赖同步传感器的方法（Ev-3DOD），行人 AP3D 甚至超越使用 LiDAR+RGB+Event 的方法。

## 研究背景与动机
3D 目标检测是自动驾驶的基础任务。LiDAR 和 RGB 相机虽然广泛使用，但受限于固定帧率（10-20 Hz），在高速场景中存在感知空白期（blind time），帧间无法采集数据，导致检测延迟。

事件相机以其异步特性和高时间分辨率提供了解决方案：

**Ev-3DOD 的局限**：最近的方法将事件相机与 LiDAR/RGB 集成实现连续时间检测，但在 blind time 期间高度依赖上一帧 LiDAR 的 3D 几何信息。当快速运动导致场景剧变时，过去的几何信息不再适用，误差随时间累积，导致频繁检测失败

**新出现物体**：依赖同步传感器的方法只能在 active time（有 LiDAR/RGB 的时刻）检测新物体，blind time 中新出现在视野中的物体无法被捕获

**事件数据的稀疏性**：事件数据缺乏语义和几何信息，直接用于 3D 检测面临准确分类和回归的困难

**核心 idea**：完全抛弃同步传感器，仅用立体事件相机计算 3D 几何信息 → 通过语义-几何双重滤波互补增强 → 目标中心 ROI 对齐提升回归精度 → 在 blind time 实现任意时间点的 3D 检测。

## 方法详解

### 整体框架
系统分四部分：(1) 几何平面扫描体（从立体事件数据构建 3D 几何体积）→ (2) 语义-几何双重滤波（互补增强语义和深度信息）→ (3) 全局 3D 检测器（BEV 上的 anchor-based 检测）→ (4) 目标中心 ROI 对齐（局部精细回归）。

输入为时间窗口 $\Delta\tau$ 内的左右事件流，转换为 voxel grid 表示 $E_L, E_R$。通过调整 $\Delta\tau$ 可在任意时间点检测。

### 关键设计
1. **几何平面扫描体**:

    - 功能：从立体事件数据构建 3D 几何体积
    - 核心思路：修改 PSMNet 设计特征提取器，增加额外特征头将语义特征 $F^{sem}$ 和几何特征 $F^{geo}$ 分离。几何特征用于构建平面扫描体：
    $\mathcal{V}_{geo}(u,v,w) = (F_L^{geo}(u,v) \| F_R^{geo}(u - \frac{fL}{d(w)}, v))$
      通过 3D 卷积 + softmax 生成深度概率体 $\mathcal{P}_{geo}$，再映射到统一 3D 体素空间 $\mathbf{V}_{geo}^{3D}$
    - 设计动机：分离语义和几何特征使网络能专注于各自任务——语义特征在 2D 平面上对物体激活好，几何特征在全场景寻找立体对应

2. **语义-几何双重滤波**:

    - 功能：利用语义信息精炼深度，同时用深度信息增强语义特征
    - 核心思路：
        - **语义引导深度精炼**：计算初始深度 $D_{init}$，基于语义特征匹配计算相似度 $S(u,v)$，结合深度置信度 $C(u,v)$（深度概率分布的方差），通过邻域概率加权生成精炼后的深度体 $\tilde{\mathcal{P}}_{geo}$：
       $$W_m(u,v) = S_m(u,v) \cdot \text{sigmoid}(C_m(u,v))$$
       $$\tilde{\mathcal{P}}_{geo}(u,v,w) = \sum_{m=1}^{M} \mathcal{P}_{geo}(u,v,w) \cdot \text{softmax}_m(W_m(u,v))$$
        - **几何滤波语义体**：用精炼深度 $\tilde{D}$ 将右相机语义特征 warp 到左相机，通过 Transformer 通道注意力机制处理遮挡和错位：
       $$\tilde{F}_L^{sem} = F_L^{sem} + \text{MLP}(\mathbb{A}) + \mathbb{A}$$
       语义体通过深度概率 mask 映射到 3D：$\mathbf{V}_{sem}^{3D}(x,y,z) = \tilde{F}_L^{sem}(u,v) \cdot \tilde{\mathcal{P}}_{geo}(u,v,d^{-1}(z))$
    - 设计动机：事件数据空间稀疏，单独的语义或几何特征都不充分；双重滤波让两者互补增强

3. **目标中心 ROI 对齐**:

    - 功能：对全局检测结果进行基于目标的局部精细调整
    - 核心思路：使用语义体 $\mathbf{V}_{sem}^{3D}$ 的 BEV 投影 $\mathbf{B}_{sem}^{2D}$（因语义特征聚焦于物体且事件数据提供丰富边缘信息）。将全局检测框划分为 $k \times k$ 网格，在语义 BEV 上进行 ROI pooling，得到目标中心局部特征 $\hat{\mathbf{B}}_{sem}^{2D}$。通过 MLP 预测局部偏移 $\Delta P_L$，与全局预测叠加得到最终结果：$\tilde{P}_G = g_d(P_G, \Delta P_L)$
    - 设计动机：单阶段全局回归在物体运动和 ego 运动导致的歧义下精度受限，需要目标级别的局部对齐

### 损失函数 / 训练策略
$$\mathcal{L} = \mathcal{L}_{depth}^{init} + \mathcal{L}_{depth}^{refine} + \mathcal{L}_{2D} + \mathcal{L}_{cls} + \mathcal{L}_{reg}^{global} + \mathcal{L}_{reg}^{local}$$
- $\mathcal{L}_{depth}^{refine}$：精炼深度的 smooth L1 损失
- $\mathcal{L}_{reg}^{global/local}$：全局/局部 3D 回归损失

使用 DSEC-3DOD 数据集的 100 FPS 标注训练和评估。事件流切片 $\Delta\tau = 10$ms。

## 实验关键数据

### 主实验（blind time 3D 检测）

| 模态 | 方法 | VEH AP3D Easy↑ | VEH APBEV Easy↑ | PED AP3D Easy↑ | PED APBEV Easy↑ |
|------|------|---------------|----------------|---------------|----------------|
| LiDAR | VoxelNeXt | 12.66 | 31.46 | 10.59 | 12.77 |
| LiDAR+RGB | LoGoNet | 17.65 | 32.55 | 11.66 | 15.09 |
| LiDAR+RGB+Event | Ev-3DOD | **29.53** | **49.31** | 18.42 | **29.06** |
| RGB-Stereo | LIGA | 14.26 | 27.25 | 6.02 | 8.73 |
| **Event-Stereo** | **Ours** | 23.47 | 40.13 | **19.86** | 22.91 |

无 LiDAR/RGB 输入，行人 AP3D 超越 Ev-3DOD（19.86 vs 18.42）。

### 消融实验

| 配置 | mAP Easy 3D↑ | mAP Easy BEV↑ | 说明 |
|------|-------------|--------------|------|
| 仅几何体 (G) | 12.92 | 20.78 | 基线 |
| G + 语义体 (S) | 14.60 | 24.85 | +1.68/+4.07 |
| G + SDR | 14.56 | 21.73 | 语义引导深度精炼 |
| G+S + SDR | 15.95 | 26.12 | 两者结合 |
| G+S + SDR (DSGF) | 15.85 | 27.78 | 完整双重滤波 |
| G+S + DSGF + OCRA | **21.66** | **31.58** | +ROI对齐，提升巨大 |

### 运动尺度实验（关键）

| Motion Scale | Time Slice | 方法 | VEH AP3D/BEV | PED AP3D/BEV |
|-------------|-----------|------|-------------|-------------|
| ×2 | ×20 | Ev-3DOD | 15.96/31.13 | 1.47/2.63 |
| ×2 | ×20 | **Ours** | **23.47/40.13** | **19.86/22.91** |
| ×4 | ×20 | Ev-3DOD | 6.36/11.70 | 0.39/0.43 |
| ×4 | ×20 | **Ours** | **22.72/39.81** | **19.04/22.71** |

随运动尺度增大，Ev-3DOD 性能断崖式下降（PED AP3D: 18.42 → 1.47 → 0.39），本方法保持稳定（19.86 → 19.86 → 19.04）。

### 关键发现
- 事件立体在 blind time 的 3D 检测中展现了巨大潜力：无需 LiDAR 几何信息即可持续计算 3D 结构
- 行人检测上超越 LiDAR+RGB+Event 的 Ev-3DOD，因行人检测需要细节信息，事件数据的高时间分辨率+语义特征互补方案更有效
- 运动尺度实验揭示了关键差异：依赖同步传感器的方法在大运动下严重退化，异步事件方法稳定
- ROI 对齐模块贡献最大（mAP 3D: 15.85 → 21.66），可能因事件数据的稀疏性使全局回归困难
- Ev-3DOD 离 active time 越远性能越差，本方法不受此限制
- 可检测 blind time 中新出现的物体（Ev-3DOD 做不到）

## 亮点与洞察
- **首个纯事件立体 3D 目标检测系统**：证明了事件相机独立完成 3D 检测的可行性
- **语义-几何双重滤波**的互补设计精巧：语义帮助深度去歧义，深度帮助语义增强空间感知
- **运动尺度实验设计**创新：通过时间轴缩放模拟大运动场景，克服了 DSEC-3DOD 数据集运动慢的限制
- 连续时间检测能力（$\Delta\tau$ 可变）使系统可适配不同速度场景
- 行人 AP3D 超越多模态融合方法是一个令人惊喜的结果

## 局限与展望
- 车辆 AP3D 仍不如 Ev-3DOD（23.47 vs 29.53），大物体的几何估计仍需 LiDAR 级精度
- 仅在 DSEC-3DOD 一个数据集验证，泛化性有待更多数据集确认
- 事件数据的语义信息本质上弱于 RGB，分类性能受限
- 计算量未详细报告，立体匹配+双重滤波的实时性是否满足自动驾驶需求存疑
- 运动尺度实验基于时间轴缩放近似，与真实大运动场景存在差异

## 相关工作与启发
- 验证了事件相机在 3D 感知中不仅是辅助传感器，更可作为主传感器
- 双重滤波的思路可推广到其他多模态/多任务融合场景
- ROI 对齐使用语义 BEV 特征（而非几何特征）的选择有启发性——稀疏数据下物体边缘信息更重要
- 为未来的纯事件自动驾驶系统奠定了重要基础

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个纯事件立体 3D 检测，问题定义和方法设计均具创新性
- 实验充分度: ⭐⭐⭐⭐ 消融详细，运动尺度实验设计巧妙，但仅单数据集
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法图直观，实验分析到位
- 价值: ⭐⭐⭐⭐⭐ 展示了事件相机在自动驾驶中的变革性潜力，开码促进后续研究

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] EV-3DOD: Pushing the Temporal Boundaries of 3D Object Detection with Event Cameras](../../CVPR2025/autonomous_driving/ev-3dod_pushing_the_temporal_boundaries_of_3d_object_detection_with_event_camera.md)
- [\[ECCV 2024\] LiDAR-Event Stereo Fusion with Hallucinations](../../ECCV2024/autonomous_driving/lidar-event_stereo_fusion_with_hallucinations.md)
- [\[ICCV 2025\] TrafficLoc: Localizing Traffic Surveillance Cameras in 3D Scenes](trafficloc_localizing_traffic_surveillance_cameras_in_3d_scenes.md)
- [\[ICCV 2025\] MAESTRO: Task-Relevant Optimization via Adaptive Feature Enhancement and Suppression for Multi-task 3D Perception](maestro_task-relevant_optimization_via_adaptive_feature_enhancement_and_suppress.md)
- [\[ICCV 2025\] Towards Open-World Generation of Stereo Images and Unsupervised Matching](towards_open-world_generation_of_stereo_images_and_unsupervised_matching.md)

<!-- RELATED:END -->
