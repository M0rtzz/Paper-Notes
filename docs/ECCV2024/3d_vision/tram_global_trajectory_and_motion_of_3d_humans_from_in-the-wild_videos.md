---
title: >-
  [论文解读] TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos
description: >-
  [ECCV 2024][3D视觉] 提出TRAM，一个两阶段方法，通过鲁棒化SLAM恢复度量尺度相机运动 + 视频Transformer（VIMO）回归相机坐标系下的人体运动，组合两者实现准确的世界坐标系3D人体全局轨迹与动作重建。
tags:
  - ECCV 2024
  - 3D视觉
---

# TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos

**会议**: ECCV 2024  
**arXiv**: [2403.17346](https://arxiv.org/abs/2403.17346)  
**代码**: [项目页](https://yufu-wang.github.io/tram4d/)  
**领域**: 3D视觉

## 一句话总结

提出TRAM，一个两阶段方法，通过鲁棒化SLAM恢复度量尺度相机运动 + 视频Transformer（VIMO）回归相机坐标系下的人体运动，组合两者实现准确的世界坐标系3D人体全局轨迹与动作重建。

## 研究背景与动机

- 从野外视频恢复人类在世界空间中的完整运动（全局轨迹+局部姿态）至关重要但极具挑战
- 传统SLAM假设静态环境，移动人体会降低估计精度；单目SLAM仅恢复尺度不确定的轨迹
- GLAMR/SLAHMR等方法依赖MoCap学习的运动prior推断轨迹尺度，遇到复杂场景（爬楼梯、跑酷）泛化差
- WHAM通过直接回归轨迹取得好效果，但依赖MoCap数据限制了对新颖轨迹的预测
- **核心洞察**：如果能精确定位相机轨迹（度量尺度）并在相机坐标系内估计人体运动，则世界坐标下的人体运动= 相机运动 ∘ 相对运动

## 方法详解

### 整体框架

1. **Masked DROID-SLAM**：双重掩码使SLAM对动态人体鲁棒
2. **度量尺度估计**：利用ZoeDepth度量深度预测对齐SLAM深度，恢复真实尺度
3. **VIMO**：基于HMR2.0的视频Transformer模型，回归相机坐标系下的SMPL姿态和位置

### 关键设计

**双重掩码DROID-SLAM**：
- 掩码输入图像：减小动态物体对全局特征的干扰
- 掩码DBA中的flow置信度：等价于从重投影误差计算中移除动态区域坐标
- 使用YOLOv7+SAM检测并分割动态物体

**度量尺度估计**：
- 对每帧独立求解 α*d_SLAM ≈ D_ZoeDepth，使用German-McClure鲁棒损失
- 取全序列中位数消除异常帧的影响
- 排除远处区域（天空等深度不准区域）

**VIMO视频Transformer**：
- 冻结HMR2.0的ViT-H骨干，新增两个时序Transformer
- 第一个时序Transformer：对ViT每个空间位置的patch token做跨时间注意力（因子化时空模型）
- 第二个时序Transformer：直接在SMPL姿态序列上做编-解码（在姿态空间学运动prior而非隐特征空间）
- 全"transformer化"设计，端到端从视频训练

### 损失函数

$$\mathcal{L} = \lambda_{2D}\mathcal{L}_{2D} + \lambda_{3D}\mathcal{L}_{3D} + \lambda_{SMPL}\mathcal{L}_{SMPL} + \lambda_{V}\mathcal{L}_{V}$$

分别约束2D关节重投影、3D关节、SMPL参数、顶点。

## 实验关键数据

### 主实验

EMDB 2数据集人体全局轨迹评估：

| 方法 | PA-MPJPE↓ | WA-MPJPE100↓ | W-MPJPE100↓ | RTE(%)↓ | ERVE↓ |
|------|-----------|--------------|-------------|---------|-------|
| GLAMR | 56.0 | 280.8 | 726.6 | 11.4 | 18.0 |
| SLAHMR | 61.5 | 326.9 | 776.1 | 10.2 | 19.7 |
| WHAM | 38.2 | 133.3 | 343.9 | 4.6 | 14.7 |
| **TRAM** | **38.1** | **76.4** | **222.4** | **1.4** | **10.3** |

### 消融实验

相机轨迹评估（EMDB 2, ATE in m）：

| 方法 | Short(5) | Medium(10) | Long(10) | Average |
|------|----------|------------|----------|---------|
| DROID-SLAM | 0.40 | 2.55 | 3.31 | 2.42 |
| DROID+Mask Image | 0.36 | 0.63 | 2.74 | 1.42 |
| DROID+Mask DBA | 0.45 | 0.42 | 1.63 | 0.91 |
| **Masked DROID** | **0.32** | **0.20** | **0.44** | **0.32** |

### 关键发现

- 全局轨迹RTE误差降低约70%（4.6%→1.4%），验证了场景中心点方法的优越性
- 双重掩码使DROID-SLAM的ATE从2.42m降至0.32m，尤其在长序列上效果显著
- WHAM在复杂轨迹（大曲线、上下楼梯）上失败，TRAM因不依赖MoCap运动prior而能泛化
- 度量尺度估计与GT尺度相比仅增加约30cm误差（0.32→0.66m）
- VIMO在3DPW上PA-MPJPE达34.1mm，优于HMR2.0和WHAM

## 亮点与洞察

- 两阶段解耦设计巧妙：利用场景背景（而非人体运动模型）估计度量尺度，避免了MoCap先验的泛化瓶颈
- 在SMPL姿态空间（而非隐特征空间）做时序建模更直接有效
- Masked DROID是一个实用的工程贡献，解决了视觉SLAM在有大面积动态物体时的鲁棒性问题
- 场景提供尺度信息的洞察极具启发性（"看到蜘蛛侠在摩天大楼间荡秋千就知道距离"）

## 局限性

- 相机完全静止时SLAM无法工作，不适用于三脚架固定相机场景
- ZoeDepth深度预测在某些场景不准，影响尺度估计
- VIMO训练数据有限（3DPW+H36M+BEDLAM），更多视频数据将进一步提升性能

## 评分

- 新颖性：⭐⭐⭐⭐
- 有效性：⭐⭐⭐⭐⭐ — 全局轨迹误差大幅降低
- 实用性：⭐⭐⭐⭐ — 适用于野外视频
- 推荐度：⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)
- [Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)
- [TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)

<!-- RELATED:END -->
