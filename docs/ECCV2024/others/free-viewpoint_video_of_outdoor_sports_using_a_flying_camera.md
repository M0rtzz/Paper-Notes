---
title: >-
  [论文解读] Free-Viewpoint Video of Outdoor Sports Using a Flying Camera
description: >-
  [ECCV 2024][自由视点视频] 提出了一种基于无人机RGB相机的系统，能够重建户外运动场景中的4D动态人体和3D无界背景，实现任意时刻的自由视点视频渲染。 1. 领域现状： 自由视点视频（Free-Viewpoint Video, FVV）技术近年来在体育赛事直播、体育分析和虚拟现实等领域受到广泛关注…
tags:
  - "ECCV 2024"
  - "自由视点视频"
  - "无人机"
  - "户外运动"
  - "4D重建"
  - "神经辐射场"
---

# Free-Viewpoint Video of Outdoor Sports Using a Flying Camera

**会议**: ECCV 2024  
**arXiv**: 无  
**代码**: 无  
**领域**: 其他  
**关键词**: 自由视点视频, 无人机, 户外运动, 4D重建, 神经辐射场

## 一句话总结

提出了一种基于无人机RGB相机的系统，能够重建户外运动场景中的4D动态人体和3D无界背景，实现任意时刻的自由视点视频渲染。

## 研究背景与动机

1. **领域现状**: 自由视点视频（Free-Viewpoint Video, FVV）技术近年来在体育赛事直播、体育分析和虚拟现实等领域受到广泛关注。现有方法主要依赖于密集相机阵列（如多机位转播系统）或手持移动相机来获取多视角数据，以实现自由视点渲染。然而，户外运动场景具有大范围人体运动和大尺度场景结构的特点，使得任务极具挑战性。

2. **现有痛点**: 
    - **密集相机阵列方案**：需要大量相机资源，成本高昂且部署复杂，难以适用于开放的户外运动场景
    - **手持相机方案**：单个手持相机难以跟踪快速移动的运动员，同时覆盖足够的场景范围，在真实运动场景中表现不佳
    - **动态-静态联合重建**：户外运动同时涉及动态运动员和大尺度静态背景（360°环境），现有方法难以同时高质量重建两者

3. **核心矛盾**: 如何使用单一低成本RGB相机（而非昂贵的多相机设置）同时重建高质量的4D动态人体和3D大尺度户外背景场景，并实现时空自由的视点渲染。

4. **本文目标**: 设计一套完整的基于无人机的自由视点视频系统，使用单台无人机搭载的RGB相机，实现户外运动场景的4D人体重建和3D无界场景重建。

5. **切入角度**: 利用无人机（drone/flying camera）作为采集平台——无人机可以灵活地围绕运动员飞行，从多角度获取图像序列，从而弥补单相机视角有限的缺陷。同时提出系统级的标定和动作捕捉子模块，提升整体鲁棒性和效率。

6. **核心 idea**: 用一台无人机搭载RGB相机围绕运动员飞行拍摄，结合专门设计的标定、动作捕捉和神经渲染子模块，实现户外运动场景的低成本自由视点视频生成。

## 方法详解

### 整体框架

系统流程包含以下主要阶段：
1. **数据采集**：无人机围绕运动员飞行，拍摄RGB视频序列
2. **相机标定**：从无人机视频中精确估计每一帧的相机内外参数
3. **人体动作捕捉**：从单目视频中恢复运动员的3D人体姿态和运动
4. **4D动态人体重建**：基于人体运动序列重建可变形的4D人体模型
5. **3D背景重建**：利用无人机拍摄的多视角图像重建360°无界户外场景
6. **自由视点渲染**：将重建的动态人体和静态背景进行组合，渲染任意视点任意时刻的图像

### 关键设计

1. **无人机标定模块（Drone Calibration）**: 
    - 功能：从无人机飞行视频中估计精确的相机参数
    - 核心思路：结合无人机IMU数据和视觉SfM（Structure-from-Motion）技术，利用无人机的飞行轨迹约束和地面特征点实现精确标定
    - 设计动机：户外场景缺乏结构化标定物，无人机飞行过程中存在抖动和非均匀运动，需要鲁棒的标定方案

2. **单目人体动作捕捉模块（Human Motion Capture）**: 
    - 功能：从单目无人机视频中恢复运动员的3D骨骼姿态和体型参数
    - 核心思路：采用基于SMPL人体模型的单目动作捕捉方法，结合时序一致性约束和运动平滑先验，确保长序列运动的稳定性
    - 设计动机：户外运动涉及大幅度、快速的人体运动（如跑步、跳跃），传统单帧姿态估计容易产生抖动和不一致

3. **4D人体+3D场景联合重建（4D Human + 3D Scene Reconstruction）**: 
    - 功能：同时重建可变形的4D动态人体和360°无界户外背景
    - 核心思路：将人体和背景分离处理——人体部分基于参数化人体模型进行可变形重建，背景部分利用多视角图像进行无界NeRF重建，最后在渲染时进行组合
    - 设计动机：动态人体和静态背景具有不同的运动特性和尺度特征，分离处理可以针对各自特点进行优化

### 损失函数 / 训练策略

- **光度重建损失**：衡量渲染图像与真实输入图像之间的像素级差异
- **感知损失**：利用预训练网络提取特征，衡量渲染质量的感知差异
- **正则化损失**：对人体变形场和背景几何施加平滑性约束，防止过拟合
- **分阶段训练**：先训练背景NeRF，再训练动态人体模型，最后进行联合微调

## 实验关键数据

### 主实验

作者收集了一个名为AerialRecon的真实户外运动数据集，包含多个户外运动场景（如跑步、投掷等）。

| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|---------|------|
| AerialRecon | PSNR（人体） | 优于对比方法 | 现有SOTA系统 | 显著提升 |
| AerialRecon | SSIM（人体） | 优于对比方法 | 现有SOTA系统 | 显著提升 |
| AerialRecon | LPIPS（人体） | 优于对比方法 | 现有SOTA系统 | 显著提升 |
| AerialRecon | 背景渲染质量 | 优于对比方法 | 现有SOTA系统 | 显著提升 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无标定优化 | PSNR下降 | 精确标定对重建质量至关重要 |
| 无时序约束 | 运动平滑性下降 | 时序一致性约束提升动作捕捉稳定性 |
| 无人体-背景分离 | 整体质量下降 | 分离处理策略有效提升各部分重建质量 |

### 关键发现

- 无人机平台相比手持相机在户外运动场景中具有显著优势，能够覆盖更大的视角范围
- 系统级的协同设计（标定→动捕→重建→渲染）比单独优化各模块更为有效
- 在真实户外运动场景中，该系统展现出优于现有SOTA系统的性能和适用性
- AerialRecon数据集填补了户外运动自由视点视频研究的数据空白

## 亮点与洞察

- **实用性强**：使用单台消费级无人机即可实现自由视点视频，大幅降低了成本和部署难度
- **系统性设计**：不是简单堆叠现有方法，而是从数据采集、标定、动捕到渲染进行端到端的系统级设计，各模块之间相互配合
- **真实场景验证**：在真实户外运动场景中进行了验证，而非仅在受控实验室环境中测试
- **新数据集贡献**：提供了AerialRecon数据集，为后续研究提供了基准

## 局限与展望

- 单台无人机拍摄的视频帧率和覆盖角度有限，可能影响快速运动的重建质量
- 户外光照变化（如天气变化、阴影）可能影响渲染一致性
- 当前的人体重建假设场景中只有单个运动员，多人场景的扩展需要进一步研究
- 实时渲染能力有限，目前可能无法满足实时直播需求
- 无人机飞行受限于电池续航和飞行法规，限制了实际应用场景

## 相关工作与启发

- **Neural Body / HumanNeRF / NeuralMan**：基于NeRF的人体重建方法，本文在此基础上扩展到户外场景
- **Instant-NGP / Mip-NeRF 360**：无界场景NeRF重建方法，本文用于背景重建
- **SMPL / SMPL-X**：参数化人体模型，为人体动作捕捉提供先验
- 启发：利用无人机等低成本移动平台进行场景采集是一个有价值的研究方向

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将无人机平台用于户外运动自由视点视频生成，系统设计新颖
- 实验充分度: ⭐⭐⭐ 在真实场景验证但定量比较可能受限于数据集规模
- 写作质量: ⭐⭐⭐⭐ 系统描述清晰，模块设计合理
- 价值: ⭐⭐⭐⭐ 具有很强的实用价值，降低了自由视点视频的技术门槛

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] ET: The Exceptional Trajectories - Text-to-Camera-Trajectory Generation with Character Awareness](et_the_exceptional_trajectories_text-to-camera-trajectory_generation_with_charac.md)
- [\[ECCV 2024\] Learning Anomalies with Normality Prior for Unsupervised Video Anomaly Detection](learning_anomalies_with_normality_prior_for_unsupervised_video_anomaly_detection.md)
- [\[ECCV 2024\] STSP: Spatial-Temporal Subspace Projection for Video Class-Incremental Learning](stsp_spatial-temporal_subspace_projection_for_video_class-incremental_learning.md)
- [\[ECCV 2024\] Real-Data-Driven 2000 FPS Color Video from Mosaicked Chromatic Spikes](real-data-driven_2000_fps_color_video_from_mosaicked_chromatic_spikes.md)
- [\[ECCV 2024\] Interleaving One-Class and Weakly-Supervised Models with Adaptive Thresholding for Unsupervised Video Anomaly Detection](interleaving_one-class_and_weakly-supervised_models_with_adaptive_thresholding_f.md)

</div>

<!-- RELATED:END -->
