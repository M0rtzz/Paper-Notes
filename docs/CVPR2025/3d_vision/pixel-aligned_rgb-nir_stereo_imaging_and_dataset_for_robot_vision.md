---
title: >-
  [论文解读] Pixel-Aligned RGB-NIR Stereo Imaging and Dataset for Robot Vision
description: >-
  [CVPR 2025][3D视觉][RGB-NIR融合] 本文开发了一套基于棱镜分光的像素对齐 RGB-NIR 立体相机系统，搭建在移动机器人上采集了大规模多光照条件数据集，并提出了图像融合和特征融合两种方法，使现有 RGB 预训练视觉模型无需/少量微调即可利用 NIR 信息，在深度估计、目标检测和 SfM 等任务上取得显著提升。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "RGB-NIR融合"
  - "像素对齐"
  - "立体深度估计"
  - "机器人视觉"
  - "多光谱数据集"
---

# Pixel-Aligned RGB-NIR Stereo Imaging and Dataset for Robot Vision

**会议**: CVPR 2025  
**arXiv**: [2411.18025](https://arxiv.org/abs/2411.18025)  
**代码**: 无  
**领域**: 自动驾驶 / 机器人视觉 / 多光谱成像  
**关键词**: RGB-NIR融合, 像素对齐, 立体深度估计, 机器人视觉, 多光谱数据集

## 一句话总结

本文开发了一套基于棱镜分光的像素对齐 RGB-NIR 立体相机系统，搭建在移动机器人上采集了大规模多光照条件数据集，并提出了图像融合和特征融合两种方法，使现有 RGB 预训练视觉模型无需/少量微调即可利用 NIR 信息，在深度估计、目标检测和 SfM 等任务上取得显著提升。

## 研究背景与动机

**领域现状**：RGB 相机是视觉计算的主要数据来源，但在低光照（夜间、暗室）条件下性能严重下降。近红外（NIR）成像借助主动照明可以在人眼不可见的波段（750-1000nm）提供有效信息，与 RGB 形成互补。RGB-NIR 多光谱融合在目标检测、3D 重建等任务中已有研究。

**现有痛点**：现有 RGB-NIR 系统（如 Kinect、Intel RealSense D415）使用分离的 RGB 和 NIR 相机，两者存在不同视角，导致像素未对齐。为对齐图像需要进行图像配准和位姿估计，这引入误差积累，而且产生了"先有鸡还是先有蛋"的问题——深度估计需要融合图像，而融合又需要深度来做配准。

**核心矛盾**：序贯采集方案只能拍静态场景；自定义 CFA 四通道传感器方案对 RGB 和 NIR 使用同一曝光，无法适应动态范围差异大的挑战性场景（如主动 NIR 照明下的暗室）。

**本文目标**：(1) 构建真正像素级对齐的 RGB-NIR 立体成像系统；(2) 采集涵盖多种光照条件的大规模数据集；(3) 设计融合方法让现有 RGB 模型能直接利用 NIR 信息。

**切入角度**：使用二色棱镜（dichroic prism）将 RGB 和 NIR 光在同一光轴上分离到两个独立 CMOS 传感器，从硬件层面实现像素级对齐，从根本上绕开配准问题。

**核心 idea**：通过棱镜分光实现像素对齐的 RGB-NIR 成像，消除了传统多光谱系统中配准和深度依赖的问题，并提出学习型空间自适应权重融合方法，让下游模型无需修改即可受益。

## 方法详解

### 整体框架

系统由搭载在移动机器人（AgileX Ranger Mini 2.0）上的一对像素对齐 RGB-NIR 双传感器立体相机、主动 NIR 照明和 LiDAR 组成。每个相机内部通过二色棱镜将入射光分为 RGB（穿透）和 NIR（反射）两路，分别由独立 CMOS 捕获。输出为四通道（R, G, B, NIR）像素对齐图像对 + LiDAR 点云。在此基础上，提出两种融合方法：(1) 图像级融合——将 RGBN 融合为三通道图像直接输入预训练模型；(2) 特征级融合——在特征空间融合 RGB 和 NIR 并微调下游网络。

### 关键设计

1. **棱镜分光像素对齐相机**:

    - 功能：在同一光轴上同时获取像素级对齐的 RGB 和 NIR 图像
    - 核心思路：使用 JAI FS-1600D-10GE 双传感器相机，内部二色棱镜将可见光和近红外光分离到两个独立 CMOS 上。RGB 和 NIR 传感器可以独立设置曝光时间和增益（$t_R = t_G = t_B \neq t_{NIR}$），适应不同光照条件下两个波段的动态范围差异
    - 设计动机：从硬件层面彻底解决了 RGB-NIR 像素不对齐的问题，无需任何图像配准步骤，消除了"深度-配准"的鸡蛋问题

2. **学习型空间自适应权重图像融合**:

    - 功能：将四通道 RGBN 融合为三通道图像，可直接输入 RGB 预训练模型
    - 核心思路：在 HSV 色彩空间中，保持色调 H 和饱和度 S 不变，对亮度通道 V 与 NIR 做加权融合 $I_{fusion} = M^{-1}[I_H, I_S, \alpha I_V + \beta I_{NIR}]$。与传统预定义固定权重不同，本文用基于注意力的 MLP 学习空间变化的 $\alpha, \beta$：先用 ResNet 编码器提取 RGB 和 NIR 各自的 256 通道特征图，通过注意力融合模块得到融合特征，再用解码器回归逐像素的 $\alpha, \beta$。最后用 NIR 图像做引导滤波进一步优化
    - 设计动机：不同场景区域中 RGB 和 NIR 的信息量不同（如近处 NIR 照明强、远处环境光主导），固定权重无法适应，学习自适应权重能根据场景动态选择最优融合比例

3. **交替相关体积特征融合立体深度估计**:

    - 功能：在特征层面融合 RGB-NIR 信息用于立体深度估计
    - 核心思路：基于 RAFT-Stereo 网络，对左右相机的 RGB 和 NIR 图像分别提取特征，通过注意力融合得到 $F_{fusion}$。关键创新是同时构建融合特征和 NIR 特征的两个相关体积（correlation volume），并在 GRU 迭代中交替使用两者进行视差估计。$F_{fusion}^{left}$ 作为上下文特征初始化 GRU 隐状态
    - 设计动机：交替使用融合和 NIR 相关体积能充分利用跨光谱信息。NIR 在环境光照变化下更稳定，而融合特征保留了 RGB 的色彩判别力，两者互补提升匹配精度

### 损失函数 / 训练策略

图像融合模型仅使用光度损失和立体深度重建损失训练，不需要训练下游视觉模型。特征融合的立体深度网络使用合成数据集的视差重建损失和真实数据集的 LiDAR 损失进行微调。

## 实验关键数据

### 主实验

**图像融合对比（预训练 RAFT-Stereo + YOLO）**:

| 方法 | 深度 RMSE (m) ↓ | 检测 mAP ↑ |
|------|----------------|-----------|
| RGB only | 8.943 | 0.756 |
| NIR only | 9.646 | 0.703 |
| HSV baseline | 7.692 | 0.744 |
| DarkVision | 8.313 | 0.762 |
| Adaptive | 7.830 | 0.773 |
| **Ours** | **7.567** | **0.809** |

**特征融合立体深度估计对比**:

| 方法 | 深度 RMSE (m) ↓ |
|------|----------------|
| RAFT-Stereo (RGB) | 8.943 |
| RAFT-Stereo (NIR) | 9.646 |
| CS-Stereo | 8.941 |
| DPSNet | 7.633 |
| Image fusion (ours) | 7.567 |
| **Feature fusion (ours)** | **6.747** |

### 消融实验

| 相关体积配置 | 深度误差 (m) ↓ | 说明 |
|------------|--------------|------|
| 仅融合体积 | 7.440 | 缺少 NIR 鲁棒匹配 |
| 交替 RGB-NIR 体积 | 8.571 | 跨光谱匹配困难 |
| 交替融合-RGB-NIR 体积 | 7.426 | 信息冗余 |
| **交替融合-NIR 体积** | **6.747** | 最优组合 |

### 关键发现

- 交替使用融合和 NIR 相关体积的策略最优，说明 NIR 的照明鲁棒性与融合特征的判别力互补效果最佳
- 图像级融合（无需重训练下游模型）已能带来显著提升，说明像素对齐本身就是巨大的优势
- 在低光照和夜间场景中，RGB-NIR 融合的优势尤为明显，NIR 主动照明弥补了 RGB 的不足

## 亮点与洞察

- **硬件-算法协同设计思路**：通过棱镜分光从硬件层面解决配准问题，避免了软件配准的误差积累，这种"在正确的层面解决问题"的思路值得借鉴
- **融合图像可直接用于预训练模型**：不需要重训练就能提升性能，这大大降低了部署成本，对实际机器人系统很有吸引力
- **独立曝光控制**：RGB 和 NIR 传感器独立设置曝光时间，能适应从强日光到全黑暗的各种场景，这是单传感器四通道方案做不到的

## 局限与展望

- 系统依赖特定的双传感器棱镜相机（JAI FS-1600D），成本高于消费级相机
- 主动 NIR 照明范围有限，在远距离场景中 NIR 信息可能不足
- 数据集规模（43 场景 80K 帧）相对于大规模自动驾驶数据集偏小
- 未探索将像素对齐 RGBN 用于生成模型的潜力，这是一个有趣的未来方向
- 可以考虑结合 LiDAR 时间飞行信息进一步提升 3D 成像精度

## 相关工作与启发

- **vs Kinect/RealSense**: 这些商用设备 RGB 和 NIR 相机位于不同位置，需要深度依赖的配准，本文从根本上避免了这个问题
- **vs 自定义 CFA 四通道传感器**: 单传感器全局曝光无法适应 RGB 和 NIR 动态范围差异大的场景，本文的双传感器独立曝光更灵活
- **vs RGB-Thermal 分光系统 [Guo et al.]**: 本文将分光思路从热红外迁移到近红外，并加入了立体视觉和 LiDAR 同步，完整性更高

## 评分

- 新颖性: ⭐⭐⭐⭐ 硬件系统设计优秀，但算法创新较为增量
- 实验充分度: ⭐⭐⭐⭐ 多任务（深度、检测、SfM）验证且有消融，但缺少与更多SOTA方法对比
- 写作质量: ⭐⭐⭐⭐ 系统描述清晰，硬件和算法组织合理
- 价值: ⭐⭐⭐⭐ 数据集和系统对机器人视觉社区有较大价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Dual Exposure Stereo for Extended Dynamic Range 3D Imaging](dual_exposure_stereo_extended_dr_3d.md)
- [\[CVPR 2025\] EgoPressure: A Dataset for Hand Pressure and Pose Estimation in Egocentric Vision](egopressure_a_dataset_for_hand_pressure_and_pose_estimation_in_egocentric_vision.md)
- [\[CVPR 2025\] Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation](helvipad_a_real-world_dataset_for_omnidirectional_stereo_depth_estimation.md)
- [\[ICLR 2026\] Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints](../../ICLR2026/3d_vision/generalizable_coarse-to-fine_robot_manipulation_via_language-aligned_3d_keypoint.md)
- [\[ICLR 2026\] NOVA3R: Non-pixel-aligned Visual Transformer for Amodal 3D Reconstruction](../../ICLR2026/3d_vision/nova3r_non-pixel-aligned_visual_transformer_for_amodal_3d_reconstruction.md)

</div>

<!-- RELATED:END -->
