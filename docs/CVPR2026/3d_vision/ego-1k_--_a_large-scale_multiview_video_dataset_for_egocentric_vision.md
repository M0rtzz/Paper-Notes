---
title: >-
  [论文解读] Ego-1K: A Large-Scale Multiview Video Dataset for Egocentric Vision
description: >-
  [CVPR 2026][3D视觉][第一人称视觉] 提出 Ego-1K，一个包含 956 段短视频的大规模时间同步第一人称多视角视频数据集（12+4 相机、60Hz），填补了第一人称动态 3D 重建领域的数据空白，并展示立体深度引导可大幅提升 4D 新视角合成质量。
tags:
  - CVPR 2026
  - 3D视觉
  - 第一人称视觉
  - 多视角数据集
  - 动态场景重建
  - 新视角合成
  - 手物交互
---

# Ego-1K: A Large-Scale Multiview Video Dataset for Egocentric Vision

**会议**: CVPR 2026  
**arXiv**: [2603.13741](https://arxiv.org/abs/2603.13741)  
**代码**: [数据集](https://huggingface.co/datasets/facebook/ego-1k)  
**领域**: 3D视觉  
**关键词**: 第一人称视觉, 多视角数据集, 动态场景重建, 新视角合成, 手物交互

## 一句话总结

提出 Ego-1K，一个包含 956 段短视频的大规模时间同步第一人称多视角视频数据集（12+4 相机、60Hz），填补了第一人称动态 3D 重建领域的数据空白，并展示立体深度引导可大幅提升 4D 新视角合成质量。

## 研究背景与动机

混合现实设备和第一人称世界建模需要从佩戴者视角进行逼真的 4D 重建。但现有数据集存在关键缺口：

- **NVS 数据集**（如 Neural 3D Video、DiVA360）：提供多视角但为外心视角（exocentric），缺乏第一人称视角
- **第一人称数据集**（如 Ego4D、EPIC-KITCHENS）：规模大但以单目/双目为主，关注活动识别而非 3D 重建
- **多视角第一人称数据集**（如 EgoExo4D、HOT3D）：仅 2-3 个第一人称相机，视角数量不足

核心需求：**同时满足大规模、高相机数、第一人称视角、精确同步**的动态场景数据集。该数据集特有的挑战包括近距离手部运动带来的大视差、快速图像运动和频繁遮挡。

## 方法详解

### 整体框架

Ego-1K 不是算法论文，而是数据集+基准论文。核心贡献包括：(1) 设计并构建多相机头戴采集系统；(2) 提出立体一致性评估协议；(3) 提出 4D NVS 评估协议；(4) 提出立体深度引导的 3DGS 基线。

### 关键设计

1. **多相机采集系统**：定制头戴设备集成 Quest 3 头显（4 个前向相机）+ 12 个外部鱼眼相机（8MP 全局快门、190° FOV、f2.8），所有 16 个相机通过无线同步器硬件同步至 60Hz。12 个外部相机通过 USB 3.1 连接到背包电脑（双 8 端口 USB 适配器），以 8-bit raw Bayer 格式流式传输。系统还包含 2 个 iToF 传感器（30Hz 交替采集）和 IMU（800Hz），总原始数据量约 15 GB/s。设计动机：现有头戴设备最多 2-3 个相机，不足以支撑稠密 3D 重建。

2. **标定系统（离线 + 在线）**：

    - **离线标定**：实验室环境使用 5 个大型 Calibu 标定板，求解所有相机内外参
    - **在线标定**：补偿镜头运动导致的 0.1-0.2° 旋转偏移和温度引起的焦距变化（对应 1-3 像素偏移），优化相机朝向和焦距，保持其他参数固定
    - 在线标定使中位 MAD 评分降低 35%

3. **研究版数据集**：将 12 个鱼眼相机去畸变为 6 对校正立体对（1280×1280, 130° 水平 FOV），便于处理。不含 Quest 3 RGB 相机（卷帘快门、分辨率和色彩配置与外部相机不同）。单段录像约 19 GB，完整数据集 17.5 TB。

4. **立体深度引导的 3DGS 基线**：核心发现是现有 NVS 方法在该数据集上严重不足，但立体基础模型能提供较好的深度估计。因此提出：

    - 用 Foundation Stereo 双向运行（L→R 和 R→L）获取深度图
    - TSDF 融合所有立体深度图得到水密表面
    - 从融合表面采样点（含法线和颜色）初始化 3D Gaussians
    - 用少量迭代微调，最小化光度损失 $\mathcal{L}=(1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{\text{D-SSIM}}$（$\lambda=0.1$）
    - 每帧独立优化，构成稠密 4D 重建

### 损失函数 / 训练策略

- 评估不涉及模型训练，仅微调 3DGS
- 训练/测试划分：10 个训练视角 + 2 个测试视角（目标立体对 3-4）
- 实验子集：10% 数据集（96 段录像）

## 实验关键数据

### 主实验

4D NVS 重建评估（目标对 3-4 为测试视角，其余 10 个视角训练）：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| 3DGS（逐帧） | 21.22 | 0.709 | 0.260 |
| K-Planes | 16.46 | 0.597 | 0.443 |
| Spacetime Gaussians | 24.76 | 0.780 | 0.270 |
| **3DGS + 立体引导** | **29.12** | **0.830** | **0.115** |

立体引导比原始 3DGS 提升 7.9 dB PSNR，比 Spacetime Gaussians 提升 4.4 dB。

### 消融实验

立体方法一致性评估（将 5 对视差图 warp 到目标对计算一致性）：

| 立体方法 | MAD ↓ (mm) | MAD<1mm ↑ | SD ↓ (mm) |
|----------|-----------|-----------|----------|
| **Foundation Stereo** | **1.6** | **74.0%** | 42.5 |
| Selective-Stereo | 8.0 | 0.0% | 46.2 |
| BiDAStereo | 2.2 | 3.1% | **8.3** |
| StereoAnywhere | 1.7 | 29.5% | 10.4 |

Foundation Stereo 整体一致性最佳（MAD 最低），BiDAStereo 极端离群值最少（SD 最低）。

### 关键发现

- 现有 NVS 方法（3DGS、K-Planes）在第一人称动态场景中严重不足，K-Planes 仅 16.46 dB
- 动态模型（K-Planes、Spacetime Gaussians）是为物体中心或固定位姿多视角视频设计的，无法有效处理自运动 + 近距离手部运动 + 大视差的组合
- 对于近距离动态物体（手），性能差距更大；对于远处物体（旁观者），差距较小
- 在线标定对立体估计精度至关重要，使 MAD 降低 35%

## 亮点与洞察

- 填补了一个明确的数据空白：领域内首个同时满足大规模 + 高相机数 + 第一人称 + 精确同步的动态场景数据集
- 提出的立体一致性评估协议（无需 GT 深度）非常实用，可迁移到其他多视角系统
- 核心发现有启发性：逐帧初始化比端到端动态模型更有效，关键瓶颈在于几何初始化而非时序建模
- 数据集设计细节值得学习：在线标定、全局快门选择、鱼眼去畸变参数选择等工程决策都有充分理由

## 局限与展望

- Quest 3 的 4 个相机未纳入研究版数据集（卷帘快门差异），利用率可提升
- iToF 数据因运动伪影和相位歧义未使用，未来可结合多模态融合
- 当前基线是逐帧 3DGS，缺乏时序一致性建模；可探索时空正则化或场景流先验
- 数据集聚焦手物交互，场景多样性可进一步扩展（如户外、多人协作）
- 原始数据集 88 TB，存储和带宽门槛较高
- 缺乏语义标注（手部关键点、物体类别等），限制了下游任务评测

## 相关工作与启发

- **Ego4D / EgoExo4D**：大规模第一人称数据集，但相机少、关注活动识别
- **Neural 3D Video / DiVA360**：多视角 NVS 数据集，但为外心视角
- **Foundation Stereo**：表现最佳的立体基础模型，作为几何先验效果显著
- **3DGS**：新视角合成骨干，+ 立体初始化大幅提升
- 启发：随着智能眼镜普及，第一人称多视角重建是重要方向；几何先验比纯学习方法更可靠

## 评分

- 新颖性: ⭐⭐⭐ 数据集贡献为主，方法侧立体引导思路较直觉但验证充分
- 实验充分度: ⭐⭐⭐⭐ 立体评估 + NVS 评估 + 多基线对比，评估协议设计严谨
- 写作质量: ⭐⭐⭐⭐ 数据集描述详尽，表格对比全面，工程细节充分
- 价值: ⭐⭐⭐⭐ 填补了明确数据空白，将推动第一人称 3D/4D 重建研究

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] SceneScribe-1M: A Large-Scale Video Dataset with Comprehensive Geometric and Semantic Annotations](scenescribe-1m_a_large-scale_video_dataset_with_comprehensive_geometric_and_sema.md)
- [\[CVPR 2026\] FaceCam: Portrait Video Camera Control via Scale-Aware Conditioning](facecam_portrait_video_camera_control_via_scale-aware_conditioning.md)
- [\[CVPR 2025\] HD-EPIC: A Highly-Detailed Egocentric Video Dataset](../../CVPR2025/3d_vision/hd-epic_a_highly-detailed_egocentric_video_dataset.md)
- [\[ICCV 2025\] HumanOLAT: A Large-Scale Dataset for Full-Body Human Relighting and Novel-View Synthesis](../../ICCV2025/3d_vision/humanolat_a_large-scale_dataset_for_full-body_human_relighting_and_novel-view_sy.md)
- [\[CVPR 2026\] Indoor Asset Detection in Large Scale 360° Drone-Captured Imagery via 3D Gaussian Splatting](indoor_asset_detection_in_large_scale_360_drone-captured_imagery_via_3d_gaussian.md)

<!-- RELATED:END -->
