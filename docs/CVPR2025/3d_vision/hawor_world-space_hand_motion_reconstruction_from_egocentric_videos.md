---
title: >-
  [论文解读] HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos
description: >-
  [CVPR 2025][3D视觉][手部运动重建] HaWoR 首次实现了从第一人称视频重建世界坐标系下的3D手部运动，通过将任务解耦为相机空间手部重建 + 自适应SLAM相机轨迹估计，并引入运动补全网络处理手部出视野的情况，在 HOT3D 数据集上取得 SOTA 的全局轨迹精度（ATE 3.36mm）和手部重建质量（PA-MPJPE 4.79mm）。
tags:
  - CVPR 2025
  - 3D视觉
  - 手部运动重建
  - 世界坐标系
  - 第一人称视频
  - SLAM
  - 运动补全
---

# HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos

**会议**: CVPR 2025  
**arXiv**: [2501.02973](https://arxiv.org/abs/2501.02973)  
**代码**: https://github.com/ZJHTerry18/HaWoR (有)  
**领域**: 3D视觉  
**关键词**: 手部运动重建, 世界坐标系, 第一人称视频, SLAM, 运动补全

## 一句话总结
HaWoR 首次实现了从第一人称视频重建世界坐标系下的3D手部运动，通过将任务解耦为相机空间手部重建 + 自适应SLAM相机轨迹估计，并引入运动补全网络处理手部出视野的情况，在 HOT3D 数据集上取得 SOTA 的全局轨迹精度（ATE 3.36mm）和手部重建质量（PA-MPJPE 4.79mm）。

## 研究背景与动机

1. **领域现状**：3D手部姿态估计已取得显著进展，但现有方法几乎都在相机坐标系下工作，忽略了手在世界空间中的运动轨迹。

2. **现有痛点**：第一人称视频中手和相机同时运动，仅相机空间的重建无法反映真实运动。而且手频繁出视野、严重遮挡、快速运动，使得世界空间重建极具挑战。

3. **核心矛盾**：世界空间手部重建面临两个独特难点：一是手部轨迹尺度在第一人称视角下天然复杂（不同于第三人称全身），二是手频繁离开视野导致运动序列不完整。同时，人体运动可以用运动先验约束，但手部运动的先验难以构建。

4. **本文目标** 给定第一人称视频，如何准确重建手在世界坐标系中的完整3D运动轨迹？

5. **切入角度**：解耦问题为两个更简单的子任务——相机空间手部运动重建 + 世界空间相机轨迹估计，然后合成世界空间手部运动。

6. **核心 idea**：通过时序注意力增强的手部重建网络 + 自适应去手SLAM + transformer运动补全网络，实现首个端到端的世界空间手部运动重建系统。

## 方法详解

### 整体框架
输入为第一人称视频序列。流程分三步：(1) 手部运动估计网络 $\mathcal{M}$ 从视频帧中重建相机空间的MANO参数序列；(2) 自适应SLAM模块估计世界空间相机轨迹（去手掩膜 + Metric3D尺度校准）；(3) 运动补全网络 $\mathcal{F}$ 将不完整的相机空间手部运动先转到规范空间，补全缺失帧，再转回世界空间。

### 关键设计

1. **带时序注意力的手部运动估计网络**:

    - 功能：从视频帧序列重建高保真相机空间手部运动
    - 核心思路：基于 WiLoR 的预训练ViT backbone提取每帧特征。引入两级时序注意力模块：**IAM（图像注意力模块）** 在ViT特征层面跨帧融合，增强截断手部区域的特征鲁棒性；**PAM（姿态注意力模块）** 在MANO参数层面做时序自注意力，直接学习手部运动先验约束重建的时序一致性。每帧输出MANO姿态 $\tilde{\Theta}_t$、形状 $\tilde{\beta}_t$、全局朝向 $\tilde{\Phi}_t$ 和相机空间平移 $\tilde{\Gamma}_t$。
    - 设计动机：单帧方法缺乏时序一致性导致抖动，且对截断/遮挡手部鲁棒性差。两级注意力分别在特征和参数层面注入时序信息，互补解决这两个问题。

2. **自适应第一人称SLAM + 度量尺度估计**:

    - 功能：从第一人称视频估计世界空间相机轨迹
    - 核心思路：基于DROID-SLAM，但直接使用会因手部占大面积而受干扰。采用双重掩膜策略：将重建的手部投影到图像空间生成手部掩膜 $\mathbf{M}_t$，同时掩掉输入图像和SLAM的置信度图 $\hat{w}_t = (1-\mathbf{M}_t) \cdot w_t$，确保只有背景像素参与bundle adjustment。用 Metric3D 预测度量尺度深度 $\mathbf{D}_t$，并提出自适应采样模块（AdaSM）：排除手部区域和过远/过近点，仅在可靠中间距离范围内优化尺度因子 $\alpha$：$E(\alpha) = \sum_{p \in S_t} \mathcal{L}_{GM}(\mathbf{D}_t(p) - \alpha \cdot \mathbf{d}_t(p))$。
    - 设计动机：标准SLAM在第一人称手部视频中严重退化（手是最大的动态物体）。直接使用度量网络的深度也不准确（近距离和远距离都有偏差），动态采样策略大幅提高了尺度估计的鲁棒性。

3. **运动补全网络（Motion Infiller）**:

    - 功能：补全手部离开视野时的缺失运动帧
    - 核心思路：先将不完整的MANO序列从各帧相机空间转换到规范空间（以首帧手部位姿为原点），去除相机运动干扰。用transformer encoder架构处理带位置编码的序列，其中缺失帧用SLERP（球面线性插值）和线性插值初始化。transformer学习从上下文帧预测缺失帧的MANO参数。训练使用HOT3D数据集（提供第一人称和第三人称视角，便于标注哪些帧手部不可见），并通过随机掩膜进行数据增强。
    - 设计动机：第一人称视频中手部30-50%的时间不在视野内，如果不补全会导致轨迹断裂。规范空间转换标准化了输入，降低了补全难度。SLERP初始化显著减轻了网络负担。

### 损失函数 / 训练策略
手部重建损失 $\mathcal{L}_\mathcal{M}$：3D关节L1 + 2D关节L1 + MANO参数L2。运动补全损失 $\mathcal{L}_\mathcal{F}$：世界平移L1 + 全局旋转L1 + 手部姿态L1 + 形状L1。推理速度仅需40ms/帧，比优化方法HMP-SLAM（160ms/帧）快75%。

## 实验关键数据

### 主实验

| 数据集 | 指标 | HaWoR | 之前最好 | 提升 |
|--------|------|-------|---------|------|
| DexYCB | PA-MPJPE↓ | **4.76** | 5.01 (WiLoR) | -5.0% |
| DexYCB (75-100%遮挡) | PA-MPJPE↓ | **5.07** | 5.68 (WiLoR) | -10.7% |
| HOT3D | ATE↓ (相机) | **3.36** | 3.80 (DROID) | -11.6% |
| HOT3D | ATE-S↓ (含尺度) | **14.61** | 21.07 (DROID+M3D) | -30.7% |
| HOT3D | W-MPJPE↓ (世界) | **33.20** | 119.41 (HMP-SLAM) | -72.2% |
| HOT3D | PA-MPJPE↓ | **4.79** | 6.00 (WiLoR-SLAM) | -20.2% |

### 消融实验

| 配置 | PA-MPJPE | W-MPJPE | Accel | 说明 |
|------|---------|---------|-------|------|
| Full model | 4.79 | 33.20 | 5.41 | 完整HaWoR |
| w/o Pretrained ViT | 7.59 | 86.80 | 9.09 | 预训练至关重要 |
| w/o IAM & PAM | 5.07 | 44.60 | 8.42 | 缺少时序模块 |
| w/o PAM | 4.80 | 36.32 | 6.03 | PAM对时序一致性关键 |
| Infiller: Last Pose | - | 116.79 | - | 最简单baseline |
| Infiller: LERP | - | 75.01 | - | 插值baseline |
| Infiller: Proposed | - | **66.25** | - | 学习补全效果最好 |

### 关键发现
- 预训练ViT是最重要的单因素，去掉后PA-MPJPE从4.79飙升到7.59
- IAM+PAM双层时序注意力将W-MPJPE从44.60降到33.20，加速误差从8.42降到5.41，验证了时序信息在两个层面都很重要
- 自适应SLAM（去手掩膜）将ATE从3.80降到3.36mm，看似不大但在ATE-S（含尺度）上差距巨大（21.07→14.61）
- 运动补全网络相比简单插值（LERP）将W-MPJPE再降12%（75.01→66.25）
- HaWoR比优化方法HMP-SLAM快4倍（40ms vs 160ms/帧），且精度大幅领先

## 亮点与洞察
- **解耦策略的smart**：将困难的世界空间手部重建分解为两个有成熟方法支撑的子问题，降低了端到端学习的难度
- **自适应去手SLAM**：简单但有效——掩掉手部即可让SLAM在第一人称视频中正常工作，这个insight可推广到任何有大面积动态前景的SLAM场景
- **规范空间运动补全**：先去除相机运动再做补全，等价于将多变的坐标系归一化，大幅简化了学习问题

## 局限与展望
- 依赖off-the-shelf检测器和跟踪器，这些组件的失败会级联影响整个系统
- 运动补全网络在极长时间缺失（>几十帧）时精度可能下降
- 仅在HOT3D实验室数据集上验证世界空间性能，真实野外场景的泛化能力未知
- 没有建模双手交互关系，两只手独立重建

## 相关工作与启发
- **vs WHAM/TRAM**: 人体全局运动重建的方法，HaWoR将类似思路迁移到手部，但面临手部特有挑战（更频繁的遮挡、更小的尺度）
- **vs HaMeR/WiLoR**: 单帧手部重建SOTA，HaWoR在其上增加时序建模和全局轨迹能力
- **vs SLAHMR**: 优化式全局人体运动重建，HaWoR采用前向推理方式实现4×更快的速度

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个世界空间手部运动重建方法，问题定义有开创性
- 实验充分度: ⭐⭐⭐⭐⭐ 多维度消融（手部/SLAM/补全），与多种baseline比较
- 写作质量: ⭐⭐⭐⭐ 问题分解清晰，实验组织系统
- 价值: ⭐⭐⭐⭐⭐ 解锁第一人称手部全局运动理解，对AR/VR和行为分析极有价值

<!-- RELATED:START -->

## 相关论文

- [Layered Motion Fusion: Lifting Motion Segmentation to 3D in Egocentric Videos](layered_motion_fusion_lifting_motion_segmentation_to_3d_in_egocentric_videos.md)
- [Estimating Body and Hand Motion in an Ego-sensed World](estimating_body_and_hand_motion_in_an_ego-sensed_world.md)
- [HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos](hot3d_hand_and_object_tracking_in_3d_from_egocentric_multi-view_videos.md)
- [DuoMo: Dual Motion Diffusion for World-Space Human Reconstruction](../../CVPR2026/3d_vision/duomo_dual_motion_diffusion_for_world-space_human_reconstruction.md)
- [Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)

<!-- RELATED:END -->
