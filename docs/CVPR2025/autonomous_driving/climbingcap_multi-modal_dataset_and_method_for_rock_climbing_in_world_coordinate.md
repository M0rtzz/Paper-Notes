---
title: >-
  [论文解读] ClimbingCap: Multi-Modal Dataset and Method for Rock Climbing in World Coordinate
description: >-
  [CVPR 2025][自动驾驶][motion capture] 提出首个攀岩运动多模态数据集 AscendMotion（412K 帧 RGB+LiDAR+IMU，22 名专业攀岩者，12 面攀岩墙），以及 ClimbingCap 方法通过分离坐标解码、三重后处理优化和半监督训练实现世界坐标系下的 3D 攀岩动作恢复，MPJPE 达 75.45mm。
tags:
  - CVPR 2025
  - 自动驾驶
  - motion capture
  - climbing
  - 多模态
  - SMPL
  - LiDAR
---

# ClimbingCap: Multi-Modal Dataset and Method for Rock Climbing in World Coordinate

**会议**: CVPR 2025  
**arXiv**: [2503.21268](https://arxiv.org/abs/2503.21268)  
**代码**: http://www.lidarhumanmotion.net/climbingcap/  
**领域**: 人体理解  
**关键词**: motion capture, climbing, multi-modal dataset, SMPL, LiDAR

## 一句话总结
提出首个攀岩运动多模态数据集 AscendMotion（412K 帧 RGB+LiDAR+IMU，22 名专业攀岩者，12 面攀岩墙），以及 ClimbingCap 方法通过分离坐标解码、三重后处理优化和半监督训练实现世界坐标系下的 3D 攀岩动作恢复，MPJPE 达 75.45mm。

## 研究背景与动机

**领域现状**：人体运动捕捉在平地行走/跑步等场景已有大量数据集（AMASS、Human3.6M 等）和方法（WHAM、GVHMR 等），但这些方法都假设人体在水平面上运动。

**现有痛点**：(1) 攀岩涉及大幅度全身运动、严重自遮挡、以及垂直方向的大位移（数米），现有方法无法处理。(2) 缺乏攀岩场景的动作捕捉数据集。(3) 世界坐标系下的全局定位（米级精度）和关节级精细姿态（毫米级精度）需要同时解决。

**核心矛盾**：垂直攀岩运动的全局轨迹估计（哪个高度？）和局部姿态恢复（手脚位置？）是两个耦合但尺度不同的问题，单一坐标系下的解码器难以同时优化两者。

**本文目标** (1) 构建首个多模态攀岩数据集；(2) 设计能在世界坐标系下恢复垂直攀爬运动的方法。

**切入角度**：RGB 提供外观/姿态线索，LiDAR 提供精确深度和全局位置，分离相机坐标（姿态）和世界坐标（位置）的解码可以解耦两个问题。

**核心 idea**：多模态输入 + 分离坐标解码 + 物理约束后处理 + 半监督利用未标注数据，实现世界坐标系下的攀岩动作恢复。

## 方法详解

### 整体框架
三阶段流程：(1) Separate Coordinate Decoding (SCD)——分别预测相机坐标下的 SMPL 参数和世界坐标下的全局位移；(2) 后处理——三个物理约束损失优化结果；(3) 半监督 Teacher-Student 训练利用未标注数据。

### 关键设计

1. **Separate Coordinate Decoding (SCD)**

    - 功能：将姿态估计和全局定位解耦
    - RGB 特征提取：ViT backbone → 图像特征
    - 点云特征提取：PointNet++ → 几何特征
    - 相机坐标解码器：迭代优化 SMPL 参数 $(\theta, \beta, \Delta c)$，3 层 SIREN 估计器，隐藏维度 64
    - 全局位移解码器：$\Gamma^{trans}_{i+1} = \Psi \cdot t_{out} + \Gamma^{trans}_i$
    - 损失：$\mathcal{L} = \mathcal{L}_{kp3d} + \mathcal{L}_{kp2d} + \mathcal{L}_\theta^{smpl} + \mathcal{L}_\beta^{smpl} + \mathcal{L}_{traj}$

2. **三重后处理优化**

    - $\mathcal{L}_{LWD}$（LiDAR-World-Distance）：不同身体部位加权的点云-人体距离约束
    - $\mathcal{L}_{SDS}$（Scene-Direction Smoothing）：平滑速度方向变化，避免轨迹跳变
    - $\mathcal{L}_{VLR}$（Velocity-Limb-Relation）：利用点云优化肢体位置

3. **半监督训练**

    - Teacher 模型在标注数据训练后，在 441 分钟未标注数据上生成伪标签
    - Student 模型从 Teacher 克隆参数，在伪标签上训练

### 数据集 AscendMotion
- 22 名专业攀岩者（含国家队成员），12 面攀岩墙
- 标注：344 分钟（PTP 时间同步，多阶段全局优化 + 人工审核）
- 未标注：441 分钟
- 传感器：Ouster-OS1 LiDAR（128 线 20FPS）+ Hik 1080P RGB（20FPS）+ Xsens MVN IMU（17 个传感器 60FPS）

## 实验关键数据

### 主实验（AscendMotion 水平/垂直场景）

| 方法 | MPJPE↓ | WA-MPJPE↓ | W-MPJPE↓ | RTE↓ | PCK₀.₃ |
|------|--------|-----------|----------|------|--------|
| WHAM | 123.05 | 115.72 | — | — | — |
| GVHMR | 89.94 | 76.85 | — | — | — |
| LiDARCapV2 | — | — | — | — | — |
| **ClimbingCap** | **75.45** | **62.95** | **78.99** | **1.57** | **0.91** |

### 消融实验

| 配置 | MPJPE↓ | WA-MPJPE↓ |
|------|--------|-----------|
| RGB only | 105.67 | — |
| w/o $\mathcal{L}_{LWD}$ | 80.46 | — |
| w/o $\mathcal{L}_{SDS}$ | 99.13 | — |
| w/o $\mathcal{L}_{VLR}$ | 91.10 | — |
| w/o semi-supervised | 77.43 | — |
| **Full** | **75.45** | **62.95** |

### 关键发现
- $\mathcal{L}_{SDS}$ 贡献最大（去掉后 MPJPE +23.68mm），方向平滑对垂直攀岩至关重要
- LiDAR 提供约 30mm 的改善（105.67→75.45），证明深度信息在大位移场景中的价值
- 半监督训练提供约 2mm 的额外改善
- CIMI4D 跨数据集泛化：PCK₀.₃=0.86，证明方法的通用性

## 亮点与洞察
- **首个攀岩数据集**填补了极端运动捕捉的重要空白，22 名专业攀岩者的数据质量很高
- **分离坐标解码**设计巧妙——相机坐标处理局部姿态，世界坐标处理全局定位，解耦后各自优化更容易
- **物理约束后处理**中 $\mathcal{L}_{SDS}$ 的设计很有针对性——攀岩中人体运动方向变化必须平滑
- 半监督利用大量未标注攀岩数据的策略可扩展性好

## 局限与展望
- 仅限攀岩场景，其他垂直运动（攀爬、跳伞、空中杂技）未验证
- LiDAR 设备成本高，限制了实际部署
- 垂直场景的 MPJPE（88.92mm）高于水平场景（75.45mm），仍有改进空间
- 未探索端到端训练（当前是多阶段流水线）

## 相关工作与启发
- **vs WHAM**: 假设水平运动，攀岩场景完全失效
- **vs GVHMR**: 全局优化更好但仍假设准平面运动
- **vs ImmFusion/FusionPose**: 也使用多模态但没有世界坐标恢复能力

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个攀岩数据集 + 世界坐标恢复方法
- 实验充分度: ⭐⭐⭐⭐ 完整消融 + 跨数据集泛化
- 写作质量: ⭐⭐⭐⭐ 数据集构建流程非常详尽
- 价值: ⭐⭐⭐⭐ 为极端运动捕捉提供了重要基础设施

<!-- RELATED:START -->

## 相关论文

- [UAVScenes: A Multi-Modal Dataset for UAVs](../../ICCV2025/autonomous_driving/uavscenes_a_multi-modal_dataset_for_uavs.md)
- [V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](../../NeurIPS2025/autonomous_driving/v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)
- [Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation](helvipad_a_real-world_dataset_for_omnidirectional_stereo_depth_estimation.md)
- [Multi-modal Knowledge Distillation-based Human Trajectory Forecasting](multi-modal_knowledge_distillation-based_human_trajectory_forecasting.md)
- [Distilling Multi-modal Large Language Models for Autonomous Driving](distilling_multi-modal_large_language_models_for_autonomous_driving.md)

<!-- RELATED:END -->
