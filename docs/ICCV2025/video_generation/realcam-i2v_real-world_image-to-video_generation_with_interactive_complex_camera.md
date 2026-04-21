---
title: >-
  [论文解读] RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control
description: >-
  [ICCV 2025][Camera Control] 提出 RealCam-I2V，通过集成单目度量深度估计构建3D场景实现度量尺度对齐训练，并提供交互式3D场景轨迹绘制界面和场景约束噪声整形机制，解决了现有轨迹引导I2V方法的尺度不一致和真实世界可用性问题。
tags:
  - ICCV 2025
  - Camera Control
  - Metric Depth
  - 扩散模型
  - Noise Shaping
  - I2V
---

# RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control

**会议**: ICCV 2025  
**arXiv**: [2502.10059](https://arxiv.org/abs/2502.10059)  
**代码**: [https://zgctroy.github.io/RealCam-I2V](https://zgctroy.github.io/RealCam-I2V)  
**领域**: 3D视觉 / 可控视频生成 / 相机控制  
**关键词**: Camera Control, Metric Depth, Video Diffusion, Noise Shaping, I2V  

## 一句话总结
提出 RealCam-I2V，通过集成单目度量深度估计构建3D场景实现度量尺度对齐训练，并提供交互式3D场景轨迹绘制界面和场景约束噪声整形机制，解决了现有轨迹引导I2V方法的尺度不一致和真实世界可用性问题。

## 研究背景与动机

相机轨迹引导的I2V生成相比文本方法提供更精确的相机控制。但现有方法（MotionCtrl、CameraCtrl、CamI2V）存在两个**核心问题**：

**尺度不一致**：训练数据中的相机参数来自COLMAP的相对尺度重建，不同视频间的尺度各不相同。这意味着同样的平移参数在不同场景中对应完全不同的相机运动幅度，模型无法学到物理一致的相机运动规律。

**真实世界可用性差**：用户面对任意真实图像时不知道场景深度和尺度，无法提供精确的相机外参数。即使是摄影专家也难以凭空给出合理的6DoF轨迹参数。

**核心洞察**：通过引入度量深度估计作为预处理步骤，可以同时解决这两个问题——为训练提供统一的度量尺度，为推理提供直观的3D交互界面。

## 方法详解

### 整体框架
- **训练**：将相机参数从相对尺度对齐到度量尺度
- **推理**：用度量深度构建点云 → 用户在3D场景中交互式绘制轨迹 → 场景约束噪声整形增强控制

### 度量场景尺度对齐 (Metric Scene-scale Alignment)

对每个训练视频：
1. 用 Depth Anything V2 (metric版) 预测参考帧的度量深度图 $D(u,v) = f_{depth}(I)$
2. 将深度图反投影到3D空间构建度量点云
3. 将度量点云与COLMAP重建的相对尺度点云对齐，得到缩放因子 $\alpha$
4. 将相对平移转换为度量平移：

$$c_{\text{cam}}^{\text{metric}} = \begin{bmatrix} R & \alpha \cdot T \\ 0 & 1 \end{bmatrix}$$

这确保了跨视频的相机参数具有一致的物理意义——相同数值的平移在所有场景中对应相同的实际距离。

### 交互式3D场景推理

推理时，用户提供参考图像，系统自动：
1. 估计度量深度 → 反投影为3D点云
2. 用户在3D点云场景中拖拽绘制相机轨迹
3. 渲染轨迹预览视频（无需运行扩散模型，实时反馈）
4. 满意后触发视频生成

### 场景约束噪声整形 (Scene-constrained Noise Shaping)

在扩散去噪的**高噪声阶段** ($t > 0.9$) 利用预览视频引导生成：

$$z_t = m \cdot (\alpha_t z_{\text{preview}} + \sigma_t \epsilon) + (1-m) \cdot z_t$$

其中 $m$ 标识选中的参考像素。关键细节：
- 仅选择当前视角下可见的像素
- 排除邻域含不可见像素的边缘像素（避免深度预测误差）
- 每个时间步重新采样 $\epsilon$（避免固定噪声覆盖有效信息）
- 仅在高噪声阶段应用（$t > 0.9$），在低噪声阶段交给条件模型保持动态内容生成能力

### 训练细节
- 基于DynamiCrafter作为I2V基础模型
- 冻结基础模型和深度预测器参数，仅训练新增模块
- RealEstate10K数据集，58K训练 / 6K测试
- Adam优化器，学习率 $1\times10^{-4}$，混合精度fp16 + ZeRO-1

## 实验

### 主实验：与SOTA方法对比

| 方法 | RotErr ↓ | TransErr(相对) ↓ | TransErr(度量) ↓ | CamMC(相对) ↓ | FVD(VideoGPT) ↓ |
|------|---------|---------------|---------------|-------------|---------------|
| DynamiCrafter | 3.34 | 9.80 | 14.14 | 15.73 | 106.02 |
| MotionCtrl | 1.05 | 2.29 | 6.82 | 7.23 | 70.29 |
| CameraCtrl | 0.74 | 1.76 | 5.51 | 5.76 | 69.20 |
| CamI2V | 0.41 | 1.34 | 3.29 | 3.42 | 62.44 |
| **RealCam-I2V** | **0.39** | **1.29** | **2.23** | **2.36** | **53.72** |

在度量尺度上提升32%+，FVD提升14.8%。

### 消融实验：MSA和SNS的作用

| 方法 | MSA | SNS | TransErr(度量) ↓ | FVD(VideoGPT) ↓ |
|------|-----|-----|---------------|---------------|
| CamI2V基线 | - | - | 3.29 | 62.44 |
| +MSA | ✓ | - | 2.65 | 60.52 |
| **+MSA+SNS (RealCam-I2V)** | ✓ | ✓ | **2.23** | **53.72** |

**关键发现**：
- MSA单独就能将度量误差降低20%，验证了尺度对齐的重要性
- SNS进一步降低16%误差并大幅提升视觉质量（FVD降低11%）
- MSA对所有基线模型都有效（MotionCtrl、CameraCtrl均有提升）

## 亮点与洞察
1. **抓住了关键痛点**：尺度不一致是相机控制视频生成的根本性问题，解决方案简单而有效
2. **一轮生成 vs 多轮迭代**：通过3D预览解耦相机调整和视频生成，避免了昂贵的多轮扩散采样
3. **即插即用**：MSA和SNS作为插件可无缝集成到现有I2V基础模型
4. **扩展应用**：支持循环视频生成、生成式帧插值和平滑场景转换

## 局限性
- 度量深度估计可能在室外/大尺度场景不够准确
- 场景约束噪声整形在高度动态场景中可能限制动态内容生成
- 依赖COLMAP提供训练数据的相对位姿，COLMAP失败的视频被丢弃
- 3D场景仅基于单帧深度估计，遮挡区域缺乏几何信息

## 相关工作
- 轨迹控制：MotionCtrl、CameraCtrl、CamI2V使用相对尺度
- 度量深度：4DiM、AC3D也使用度量深度但方法不同
- 无训练方法：CamTrol渲染静态点云但无训练时对齐

## 评分
- **创新性**: ★★★★☆ — 度量尺度对齐+噪声整形的组合有效且新颖
- **实用性**: ★★★★★ — 交互式3D界面极大降低了用户门槛
- **实验**: ★★★★☆ — 消融充分，跨基线泛化验证了通用性
- **写作**: ★★★★☆ — 问题分析深入，方法动机清晰

<!-- RELATED:START -->

## 相关论文

- [TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation](tip-i2v_a_million-scale_real_text_and_image_prompt_dataset_for_image-to-video_ge.md)
- [Free-Form Motion Control: Controlling the 6D Poses of Camera and Objects in Video Generation](free-form_motion_control_controlling_the_6d_poses_of_camera_and_objects_in_video.md)
- [ReCamMaster: Camera-Controlled Generative Rendering from A Single Video](recammaster_camera-controlled_generative_rendering_from_a_single_video.md)
- [WorldScore: A Unified Evaluation Benchmark for World Generation](worldscore_a_unified_evaluation_benchmark_for_world_generation.md)
- [MotionStream: Real-Time Video Generation with Interactive Motion Controls](../../ICLR2026/video_generation/motionstream_real-time_video_generation_with_interactive_motion_controls.md)

<!-- RELATED:END -->
