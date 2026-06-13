---
title: >-
  [论文解读] Dual Exposure Stereo for Extended Dynamic Range 3D Imaging
description: >-
  [CVPR 2025][3D视觉][双曝光立体] 提出双曝光立体方法（Dual-Exposure Stereo），通过自动控制立体相机的双曝光参数扩展有效动态范围，并设计运动感知的双曝光深度估计网络，在宽动态范围场景中实现鲁棒的 3D 成像。 领域现状：立体成像是流行的 3D 成像技术，但传统相机的动态范围有限…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "双曝光立体"
  - "高动态范围"
  - "深度估计"
  - "自动曝光控制"
  - "机器人视觉"
---

# Dual Exposure Stereo for Extended Dynamic Range 3D Imaging

**会议**: CVPR 2025  
**arXiv**: [2412.02351](https://arxiv.org/abs/2412.02351)  
**代码**: 无（将公开）  
**领域**: 3D视觉 / 立体视觉  
**关键词**: 双曝光立体, 高动态范围, 深度估计, 自动曝光控制, 机器人视觉

## 一句话总结

提出双曝光立体方法（Dual-Exposure Stereo），通过自动控制立体相机的双曝光参数扩展有效动态范围，并设计运动感知的双曝光深度估计网络，在宽动态范围场景中实现鲁棒的 3D 成像。

## 研究背景与动机

**领域现状**：立体成像是流行的 3D 成像技术，但传统相机的动态范围有限，在极端光照条件下（如隧道出口、夜间强光），过曝和欠曝区域会严重影响视差估计。

**现有痛点**：现有自动曝光控制（AEC）方法为每帧单独调整，无法扩展相机的原生动态范围；曝光包围技术使用预定义曝光，不能自适应场景变化。

**核心矛盾**：场景动态范围可能远超相机原生动态范围，单一曝光无法同时覆盖亮暗区域。

**本文目标**：设计一种双曝光策略，在交替帧中使用不同曝光，结合 AEC 和曝光包围的优点。

**切入角度**：在交替帧中用不同曝光拍摄立体图像，当场景 DR 超过相机 DR 时自动发散双曝光。

**核心 idea**：自动双曝光控制（ADEC）动态调整双曝光参数 + 运动感知特征融合网络利用双曝光图像进行深度估计。

## 方法详解

### 整体框架

系统分两部分：(1) ADEC 根据直方图统计自动控制双曝光参数；(2) 双曝光深度估计网络从四张图像（两帧×两视角）估计视差图，通过光流对齐和加权特征融合扩展有效 DR。

### 关键设计

1. **自动双曝光控制（ADEC）**:

    - 功能：自适应调整双曝光参数以覆盖更宽的场景动态范围
    - 核心思路：计算直方图偏度 $S_i$ 和极端像素比例 $L_i, H_i$。当 $L_i > \tau_h$ 且 $H_i > \tau_h$ 时判定场景 DR 超过相机 DR，发散双曝光；否则通过零化偏度使双曝光收敛
    - 设计动机：结合 AEC 的自适应性和曝光包围的 DR 扩展能力

2. **双曝光特征融合**:

    - 功能：融合不同曝光下的特征以获得覆盖亮暗区域的统一特征
    - 核心思路：用光流网络估计帧间运动并对齐第二帧特征到第一帧，通过梯形权重函数 $W_i^c$ 根据像素亮度加权融合：$\hat{F}^c = (W_1^c \cdot F_1^c + W_{2\to1}^c \cdot F_{2\to1}^c) / (W_1^c + W_{2\to1}^c + \epsilon)$
    - 设计动机：过曝/欠曝像素权重降低，良好曝光像素权重提高，确保融合特征质量

3. **运动感知视差估计**:

    - 功能：从融合特征构建代价体并估计视差图
    - 核心思路：用预训练特征提取器获取双曝光特征，光流对齐后加权融合，构建左右视角的相关体 $C(x,y,d)$，送入视差估计网络
    - 设计动机：双曝光特征融合编码了亮暗区域的信息，扩展了 3D 成像的有效 DR

### 损失函数 / 训练策略

使用 CARLA 模拟器生成包含 1000 个训练视频的合成数据集，包含日间、黄昏、夜间等多种光照条件。在合成数据上微调视差估计网络。

## 实验关键数据

### 主实验

在合成和真实世界数据集上都优于其他曝光控制方法，在宽 DR 场景中深度估计准确度显著提升。

### 消融实验

- 对比固定曝光 vs ADEC：ADEC 在宽 DR 场景中优势明显
- 双曝光 vs 单曝光深度估计：双曝光融合显著改善过曝/欠曝区域的深度精度

### 关键发现

- 双曝光方法可以应用于任意位深度的相机，不限于特定硬件
- 曝光差异过大会影响立体匹配，需要限制双曝光间隔

## 亮点与洞察

- 实际搭建了机器人视觉系统（轮式机器人 + 立体相机 + LiDAR）采集真实数据
- ADEC 的偏度-发散机制设计简洁高效
- 提供了合成和真实世界两套数据集

## 局限与展望

- 对快速运动场景的光流估计可能不准确
- 双曝光策略使有效帧率减半
- 真实世界数据集规模有限

## 相关工作与启发

- 与事件相机等替代传感器方法互补
- 可扩展到多曝光（>2）的方案
- 动态范围扩展思路可应用于其他基于视觉的 3D 任务

## 评分

- 新颖性：7/10 — 双曝光思路直观但有效
- 技术深度：7/10 — 系统设计完整
- 实验充分度：8/10 — 合成 + 真实世界验证
- 写作质量：7/10 — 清晰规范

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Pixel-Aligned RGB-NIR Stereo Imaging and Dataset for Robot Vision](pixel-aligned_rgb-nir_stereo_imaging_and_dataset_for_robot_vision.md)
- [\[CVPR 2025\] MVSAnywhere: Zero-Shot Multi-View Stereo](mvsanywhere_zero-shot_multi-view_stereo.md)
- [\[CVPR 2025\] MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos](megasam_accurate_fast_and_robust_structure_and_motion_from_casual_dynamic_videos.md)
- [\[CVPR 2025\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)
- [\[CVPR 2025\] IRIS: Inverse Rendering of Indoor Scenes from Low Dynamic Range Images](iris_inverse_rendering_of_indoor_scenes_from_low_dynamic_range_images.md)

</div>

<!-- RELATED:END -->
