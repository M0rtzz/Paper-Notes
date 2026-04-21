---
title: >-
  [论文解读] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting
description: >-
  [NeurIPS 2025][3D视觉][事件相机] EF-3DGS 首次将事件相机引入自由轨迹场景重建，通过事件生成模型（EGM）重建帧间潜在图像做连续监督、对比度最大化（CMax）结合线性事件模型（LEGM）挖掘运动信息校准位姿，以及光度 BA + Fixed-GS 策略解决颜色不一致问题，在高速场景下 PSNR 提升 3dB、ATE 降低 40%。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 事件相机
  - 3D Gaussian Splatting
  - free-trajectory
  - 姿态估计
  - 新视角合成
---

# EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting

**会议**: NeurIPS 2025  
**arXiv**: [2410.15392](https://arxiv.org/abs/2410.15392)  
**代码**: 待确认  
**领域**: 3d_vision  
**关键词**: 事件相机, 3D Gaussian Splatting, free-trajectory, 姿态估计, 新视角合成

## 一句话总结

EF-3DGS 首次将事件相机引入自由轨迹场景重建，通过事件生成模型（EGM）重建帧间潜在图像做连续监督、对比度最大化（CMax）结合线性事件模型（LEGM）挖掘运动信息校准位姿，以及光度 BA + Fixed-GS 策略解决颜色不一致问题，在高速场景下 PSNR 提升 3dB、ATE 降低 40%。

## 背景与动机

- 3DGS 从带位姿的图像集合优化场景表示，在新视角合成取得巨大进展
- 自由轨迹视频重建面临两大挑战：(1) 位姿不准确，(2) 高速场景下帧间重叠不足导致欠约束
- 已有 pose-free 方法（LocalRF, CF-3DGS）在高速/低帧率场景性能严重退化
- 事件相机具备高时间分辨率和低延迟，能在帧间盲区提供丰富的亮度和运动信息
- 将事件数据无缝集成到 3DGS 存在技术难点：事件是差分信号 vs 3DGS 渲染绝对亮度

## 核心问题

如何利用事件相机的高时间分辨率特性，在高速自由轨迹场景中同时优化相机位姿和 3DGS 场景重建质量？

## 方法详解

### 1. EGM 驱动优化

将帧间时间区间均分为 N 个子区间，累积事件帧。利用最近帧和累积事件重建中间时刻的潜在亮度图：

$$I_{i,j} = I_{i,0} \cdot \exp\left(\sum_{n=0}^{j-1} E_{i,n} \cdot C\right)$$

以此为监督信号，将 3DGS 优化从离散帧拓展到连续事件流：

$$\mathcal{L}_{EGM} = (1-\lambda)\mathcal{L}_1(\hat{I}_t, I_t) + \lambda\mathcal{L}_{D\text{-}SSIM}(\hat{I}_t, I_t)$$

### 2. CMax + LEGM 联合优化

CMax 框架利用事件的时空关联估计运动场。将前 r 个子区间的事件帧通过光流 warp 回参考帧，光流由 3DGS 渲染深度和相对位姿推算。

对比度最大化损失：最大化 warped 事件图像（IPWE）的方差：

$$\mathcal{L}_{cm} = -\text{Var}(\text{IPWE}_{i,j})$$

LEGM 梯度损失基于线性事件模型建立 IPWE 与渲染图像亮度变化的联系：

$$\mathcal{L}_{grad} = \|C \cdot \text{IPWE}_{i,j} - (\hat{L}(\mathbf{u}) - \hat{L}(\mathbf{u} + F_{i,j \to j+1}))\|^2$$

$$\mathcal{L}_{LEGM} = \lambda_{cm}\mathcal{L}_{cm} + \lambda_{grad}\mathcal{L}_{grad}$$

### 3. 光度 BA + Fixed-GS 策略

PBA 对随机采样时刻建立光度重投影误差，将渲染像素投影到最近 RGB 帧计算一致性。

Fixed-GS 两阶段策略：
- 第一阶段：全参数优化（位置、透明度、旋转、缩放、SH），使用事件+帧损失
- 第二阶段：仅优化 SH 系数（颜色），其余参数固定，仅在 RGB 帧上训练

两阶段比例 4:1，有效解决事件流无色彩信息导致的颜色失真。

### 总损失

$$\mathcal{L}_{event} = \mathcal{L}_{EGM} + \mathcal{L}_{LEGM} + \lambda_{PBA}\mathcal{L}_{PBA}$$

## 实验关键数据

### Tanks and Temples 基准（不同帧率）

| 方法 | Pose-Free | 6FPS PSNR↑ | 2FPS PSNR↑ | 1FPS PSNR↑ |
|------|-----------|---------|---------|---------|
| CF-3DGS | Yes | 26.05 | 22.08 | 20.53 |
| Event-3DGS (E+F) | No | 26.32 | 23.44 | 22.41 |
| EvCF-3DGS | Yes | 26.07 | 22.81 | 21.73 |
| **EF-3DGS** | **Yes** | **26.66** | **24.43** | **23.96** |

在 1FPS 极端高速场景下 EF-3DGS 比 CF-3DGS 高 **3.43dB**。

### 位姿估计

EF-3DGS 在所有帧率下 ATE 均为最低，高速场景降幅约 40%。在新采集的 RealEv-DAVIS 真实事件数据集上同样显著领先。

## 亮点

- 首次将事件相机引入自由轨迹 3DGS 场景重建任务
- 从事件相机成像原理出发推导三个互补损失函数（EGM/CMax+LEGM/PBA），设计严谨
- Fixed-GS 两阶段训练策略巧妙分离结构和颜色优化，解决事件流无颜色信息的核心矛盾
- 高速场景（1FPS）PSNR 提升 3dB 以上，实用意义显著

## 局限与展望

- 需要事件+帧同步的硬件（DAVIS 相机），实际部署成本较高
- 事件噪声模型较简化（固定对比度阈值 C），真实场景噪声更复杂
- 渐进式场景扩展继承自 LocalRF/CF-3DGS，超长序列效率待验证
- 动态场景处理能力未明确讨论

## 与相关工作的对比

- vs **CF-3DGS**: 纯帧方法，高速场景严重退化；EF-3DGS 通过事件流补充帧间信息
- vs **Event-3DGS**: 需要已知位姿；EF-3DGS 是 pose-free 的
- vs **E-NeRF/EventNeRF**: 基于 NeRF 的事件方法，需已知位姿且限于小场景
- vs **EvCF-3DGS**: 简单将事件损失加入 CF-3DGS，缺少 CMax 运动约束和 Fixed-GS 策略

## 启发与关联

- CMax 框架在 3DGS 中的应用思路可拓展到其他需要亚帧级运动估计的任务
- Fixed-GS 两阶段训练策略可推广到其他多模态（如热红外+可见光）场景建模
- 事件相机在 VR/AR、FPV 无人机、自动驾驶等高速场景有天然优势

## 评分

- ⭐ 新颖性: 4.5/5 — 首创事件辅助自由轨迹 3DGS，三个损失函数设计精巧
- ⭐ 实验充分度: 4/5 — 公开基准+自建真实数据集，多帧率全面对比
- ⭐ 写作质量: 4/5 — 方法从事件成像原理推导，逻辑清晰
- ⭐ 价值: 4.5/5 — 高速场景重建痛点的有效解决方案

<!-- RELATED:START -->

## 相关论文

- [IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera](../../CVPR2025/3d_vision/inceventgs_pose-free_gaussian_splatting_from_a_single_event_camera.md)
- [ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS](zpressor_bottleneck-aware_compression_for_scalable_feed-forward_3dgs.md)
- [E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction](../../CVPR2026/3d_vision/e2egs_event-to-edge_gaussian_splatting_for_pose-free_3d_reconstruction.md)
- [Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting](../../CVPR2025/3d_vision/diet-gs_diffusion_prior_and_event_stream-assisted_motion_deblurring_3d_gaussian_.md)

<!-- RELATED:END -->
