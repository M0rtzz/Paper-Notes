---
title: >-
  [论文解读] TAPIP3D: Tracking Any Point in Persistent 3D Geometry
description: >-
  [NeurIPS 2025][3D视觉][3D点跟踪] 提出TAPIP3D，将视频表示为相机稳定化的时空3D特征点云，通过3D邻域到邻域（N2N）注意力机制在持久3D几何空间中迭代精化多帧点轨迹，显著超越现有3D点跟踪方法。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 3D点跟踪
  - 特征点云
  - 世界坐标系
  - 邻域注意力
  - 相机消除
---

# TAPIP3D: Tracking Any Point in Persistent 3D Geometry

**会议**: NeurIPS 2025  
**arXiv**: [2504.14717](https://arxiv.org/abs/2504.14717)  
**代码**: [Project Page](https://tapip3d.github.io/)  
**领域**: 3D视觉  
**关键词**: 3D点跟踪, 特征点云, 世界坐标系, 邻域注意力, 相机消除

## 一句话总结

提出TAPIP3D，将视频表示为相机稳定化的时空3D特征点云，通过3D邻域到邻域（N2N）注意力机制在持久3D几何空间中迭代精化多帧点轨迹，显著超越现有3D点跟踪方法。

## 研究背景与动机

视频中的长期点跟踪是机器人和动作识别的重要工具。粒子级运动估计为捕捉物体姿态变化、关节运动、可变形结构提供了统一框架。然而，现有方法存在关键局限：

**2D跟踪的根本问题**：大多数点跟踪器在像素空间（如CoTracker系列）或像素空间+深度信息（如SpatialTracker、DELTA）中操作。但视频中大部分表观运动来自相机移动而非物体运动，在2D空间跟踪会将两者混淆，增加跟踪难度。

**UVD坐标系的不足**：现有3D跟踪器（SpatialTracker、DELTA）使用UVD坐标（像素坐标+深度），这是一种2.5D表示，没有真正利用3D几何结构。在大相机运动下，UVD空间中的轨迹复杂且不规则。

**特征提取方式的局限**：SpatialTracker用三平面（Triplane）投影来特征化3D点云，虽然速度快但损失了几何信息。DELTA简单地将深度作为额外通道叠加到2D相关图上，本质上仍是2D方法的扩展。

本文的核心洞察是：如果用深度和相机位姿将视频提升到一个世界坐标系的3D特征点云中，相机运动就被"消除"了，剩下的仅是场景中物体的真实3D运动，轨迹变得更平滑、更容易跟踪。

## 方法详解

### 整体框架

TAPIP3D输入RGB-D视频（深度可来自传感器、估计器或GT），输出查询点的3D轨迹。主要流程：（1）将每帧视频特征提升为3D特征点云；（2）可选地用相机位姿转换到世界坐标系消除相机运动；（3）通过3D N2N注意力在特征点云中提取局部3D上下文；（4）通过Transformer迭代精化轨迹估计。

### 关键设计

1. **3D特征点云视频表示**：将传统2D特征图"增强"为带3D坐标的表示——每个2D单元既存储 $C$ 维特征向量，也存储对应的 $(X,Y,Z)$ 3D坐标（由深度反投影得到）。构建多尺度特征：
    $\mathcal{F}_l = \{\mathbf{F}^{l,t} \in \mathbb{R}^{\frac{H}{\ell 2^{l-1}} \times \frac{W}{\ell 2^{l-1}} \times (3+C)}\}_{t=1}^T$
   关键是坐标用最近邻插值、特征用平均池化进行下采样，确保3D几何不被模糊化。

2. **3D邻域到邻域（N2N）注意力**：这是核心创新。取代了传统2D方块相关窗口，使用3D k-NN建立真正基于3D距离的邻域关系。

    - **Support tokens**：对每个查询点，在其起始帧特征点云中找 $K$ 个3D最近邻作为"支撑"组，捕捉查询点的局部形状信息
    - **Context tokens**：在每个时间步，用当前轨迹估计坐标 $\tau_q^t$ 在该帧特征点云中找 $K$ 个3D最近邻作为"上下文"组
    - **双向交叉注意力**：support组和context组之间做双向交叉注意力，再通过注意力池化压缩为每个查询每个时间步的摘要向量
    - 在交叉注意力中融入3D相对偏移的位置编码，增强空间感知能力
    - 对所有尺度并行执行，得到多尺度邻域信息 $\mathcal{N}_q^t$

3. **3D轨迹更新Transformer**：将N2N注意力特征与轨迹信息组合成token：
    $G_q^t = [\mathcal{N}_q^t, \gamma(\tau_q^t - \tau_q^{t-1}), \gamma(\tau_q^{t+1} - \tau_q^t), \gamma(\pi_t(\tau_q^t)), o_q^t, \gamma(t)]$
   其中显式加入2D投影坐标 $\pi_t(\tau_q^t)$ 帮助模型识别超出图像边界的点。使用代理token的Transformer处理时空注意力，输出位置和可见性的增量更新。

4. **世界坐标系跟踪 vs 相机坐标系跟踪**：

    - TAPIP3D-camera：在相机坐标系中跟踪（不使用相机位姿）
    - TAPIP3D-world：使用相机位姿将所有帧的点云变换到第一帧相机坐标系（即"世界"坐标），消除相机运动
    - 两者共享同一训练权重，仅推理时坐标系不同

### 损失函数 / 训练策略

训练损失为深度自适应的位置损失+可见性交叉熵损失：
$$\mathcal{L} = \sum_{q=1}^Q \sum_{t=1}^T \frac{1}{d_q^t} \|\tau_q^t - \tilde{\tau}_q^t\|_2 + \alpha_{vis} \text{CE}(o_q^t, \tilde{o}_q^t)$$

$1/d_q^t$ 的深度缩放使远处点的损失降低，避免远处深度不精确导致梯度不稳定。训练4次迭代精化，每次输出都有监督（用折扣因子 $\gamma=0.8$ 递减权重）。训练数据为Kubric MOVi-F合成数据集，8×L40S GPU训练200K迭代约4.2天。

## 实验关键数据

### 主实验

**TAPVid-3D真实世界基准（使用MegaSaM估计深度和位姿）：**

| 方法 | ADT AJ↑ | ADT APD↑ | DriveTrack AJ↑ | PStudio AJ↑ | 平均AJ↑ |
|------|---------|---------|----------------|-------------|---------|
| CoTracker3 + M-SaM | 20.4 | 30.1 | 14.1 | 17.4 | 17.3 |
| SpatialTracker + M-SaM | 15.9 | 23.8 | 7.7 | 15.3 | 13.0 |
| DELTA + M-SaM | 21.0 | 29.3 | 14.6 | 17.7 | 17.8 |
| TAPIP3D-camera + M-SaM | 21.6 | 31.0 | 14.6 | 18.1 | 18.1 |
| **TAPIP3D-world + M-SaM** | **23.5** | **32.8** | **14.9** | **18.1** | **18.8** |

**LSFOdyssey合成基准（GT Depth对比）：**

| 方法 | AJ3D↑ | APD3D↑ | AJ2D↑ |
|------|-------|--------|-------|
| DELTA + GT | 37.7 | 50.1 | 72.4 |
| TAPIP3D-camera + GT | 68.3 | 83.2 | 76.0 |
| **TAPIP3D-world + GT** | **72.2** | **85.8** | **78.5** |

### 消融实验

| 配置 | LSFOdyssey AJ3D↑ | APD3D↑ | 说明 |
|------|------------------|--------|------|
| UV+D坐标系 | 63.4 | 77.0 | 现有方法常用 |
| UV+log(D)坐标系 | 62.9 | 77.9 | 对数深度无明显帮助 |
| XYZ(相机)坐标系 | 67.1 | 81.6 | 显著优于UVD |
| **XYZ(世界)坐标系** | **70.7** | **84.1** | 最佳 |
| 相机坐标，无N2N注意力 | 59.4 | 72.7 | 基线 |
| **相机坐标，有N2N注意力** | **67.1** | **81.6** | N2N提升巨大 |
| 世界坐标，无N2N注意力 | 62.1 | 75.1 | 世界坐标本身有帮助 |
| **世界坐标，有N2N注意力** | **70.7** | **84.1** | 两者叠加效果最好 |

### 关键发现

- 在有GT深度的合成数据上，TAPIP3D-world的AJ3D几乎是DELTA的两倍（72.2 vs 37.7），展示了3D特征点云表示的巨大优势
- 当深度质量提升（MegaSaM→GT）时，TAPIP3D的性能提升幅度远大于基线方法，说明模型能更有效利用高质量深度
- 3D k-NN比固定2D邻域在DexYCB-Pt上提升AJ3D从27.7到29.8
- 在ADT数据集（大相机运动）上，世界坐标系跟踪优势最为显著（23.5 vs 21.6）
- 推理速度10 FPS、仅需约2.6GB VRAM（跟踪1024查询点、32帧）

## 亮点与洞察

- **世界坐标系跟踪的先驱**：TAPIP3D是首个能在世界坐标系（相机运动被消除）中进行3D点跟踪的方法，利用了近期深度估计和相机位姿估计的进步
- **3D N2N注意力设计精巧**：用support组捕捉查询点局部形状、context组捕捉目标区域上下文，双向交叉注意力融合两者，有效解决匹配歧义
- **显存高效训练策略**：通过在每次迭代后detach梯度并立即反传，显存从48GB+降至约20GB，不牺牲性能
- **生态友好**：仅在合成数据上训练就能在真实世界基准上达到SOTA

## 局限与展望

- 性能受深度图质量影响显著，在极端深度变化或小型远处模糊物体的场景中可能出现深度闪烁
- 当缺乏高质量深度时，其2D指标可能弱于纯2D或UVD空间的跟踪器（因为2D轨迹由3D投影而来依赖几何一致性）
- 可通过深度补全和噪声过滤的预处理来缓解深度质量问题
- 随着3D视觉重建模型的进步，TAPIP3D的鲁棒性会自然提升

## 相关工作与启发

构建在CoTracker3的迭代精化框架之上，但用3D特征点云替代2D特征图、用N2N注意力替代2D相关窗口。与SpatialTracker的三平面表示相比，直接使用3D特征点云更显式地编码几何结构，尽管速度略慢但精度大幅提升。本文的N2N注意力将LocoTrack中2D CNN的区域到区域匹配思想推广到了3D特征点云的交叉注意力。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次提出世界坐标系3D点跟踪，N2N注意力设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖4个基准（真实/合成、不同深度源），消融详尽
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，图示直观地对比了UVD和XYZ轨迹差异
- 价值: ⭐⭐⭐⭐⭐ 利用深度估计进步实现质的飞跃，为3D点跟踪建立新范式

<!-- RELATED:START -->

## 相关论文

- [Online Segment Any 3D Thing as Instance Tracking](online_segment_any_3d_thing_as_instance_tracking.md)
- [Multi-View 3D Point Tracking](../../ICCV2025/3d_vision/multi-view_3d_point_tracking.md)
- [Continuous 3D Perception Model with Persistent State](../../CVPR2025/3d_vision/continuous_3d_perception_model_with_persistent_state.md)
- [GAP: Gaussianize Any Point Clouds with Text Guidance](../../ICCV2025/3d_vision/gap_gaussianize_any_point_clouds_with_text_guidance.md)
- [Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors](learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)

<!-- RELATED:END -->
