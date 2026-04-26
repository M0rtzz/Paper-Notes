---
title: >-
  [论文解读] VerseCrafter: Dynamic Realistic Video World Model with 4D Geometric Control
description: >-
  [CVPR 2026][3D视觉][视频世界模型] 提出 VerseCrafter，一个基于4D几何控制表示（静态背景点云 + 逐物体3D高斯轨迹）的视频世界模型，通过轻量 GeoAdapter 将4D控制信号注入冻结的 Wan2.1-14B 视频扩散模型，实现了对相机和多物体运动的精确、解耦控制，同时构建了包含 35K 样本的真实世界数据集 VerseControl4D。
tags:
  - CVPR 2026
  - 3D视觉
  - 视频世界模型
  - 4D几何控制
  - 3D高斯轨迹
  - 相机与物体运动
  - 视频扩散模型
---

# VerseCrafter: Dynamic Realistic Video World Model with 4D Geometric Control

**会议**: CVPR 2026  
**arXiv**: [2601.05138](https://arxiv.org/abs/2601.05138)  
**代码**: https://sixiaozheng.github.io/VerseCrafter_page/ (有)  
**领域**: 3D视觉 / 视频生成  
**关键词**: 视频世界模型、4D几何控制、3D高斯轨迹、相机与物体运动、视频扩散模型

## 一句话总结
提出 VerseCrafter，一个基于4D几何控制表示（静态背景点云 + 逐物体3D高斯轨迹）的视频世界模型，通过轻量 GeoAdapter 将4D控制信号注入冻结的 Wan2.1-14B 视频扩散模型，实现了对相机和多物体运动的精确、解耦控制，同时构建了包含 35K 样本的真实世界数据集 VerseControl4D。

## 研究背景与动机

1. **领域现状**：视频世界模型旨在模拟动态真实世界环境，近期方法通过文本、动作或相机轨迹调节视频生成。相机控制方面已有 CameraCtrl、MotionCtrl 等工作通过 Plücker 编码或 3D 先验注入实现视点控制。物体运动控制主要依赖 2D 线索（轨迹点、光流、mask、2D bbox）。

2. **现有痛点**：(a) 2D 控制信号在大视角变化下不鲁棒，缺乏 3D 感知；(b) 更先进的 3D 信号如 3D bbox 过于刚性、SMPL-X 只适用于人体、稀疏3D轨迹常有噪声和不完整；(c) 现有方法的控制空间是碎片化的，相机和物体运动不在统一坐标系下，无法协调控制。

3. **核心矛盾**：理想的世界模型应模拟完整的 4D 时空空间，但视频只捕捉 2D 投影。需要一种紧凑、可编辑且类别无关的 4D 几何状态表示来统一相机和多物体运动控制。

4. **本文目标**：(a) 设计统一的 4D 几何控制表示；(b) 在共享世界坐标系下实现相机与多物体运动的解耦控制；(c) 构建大规模训练数据。

5. **切入角度**：用 3D 高斯分布描述物体的概率性 3D 占据——均值定义运动路径，协方差捕捉空间范围和朝向，天然支持软性、灵活、类别无关的物体建模。

6. **核心 idea**：用静态背景点云 + 逐物体 3D 高斯轨迹在共享世界坐标系下构成统一 4D 几何状态，渲染为多通道控制图后通过 GeoAdapter 驱动冻结视频扩散模型生成。

## 方法详解

### 整体框架
输入：一张参考图像 + 文本提示。首先估计深度和相机内参（MoGe-2），获取用户指定的物体 mask（Grounded SAM2），构建 4D 几何控制（背景点云 + 3D 高斯轨迹）。用户在共享世界坐标系中指定相机轨迹，将所有控制信号渲染为逐帧 4D 控制图（背景 RGB/深度 + 轨迹 RGB/深度 + 软融合 mask）。控制图经 Wan Encoder 编码后输入 GeoAdapter，通过残差调制注入冻结的 Wan2.1-14B DiT 骨干，结合 umT5 文本嵌入生成视频。

### 关键设计

1. **4D 几何控制表示**：
    - 功能：在共享世界坐标系下统一表示场景的静态结构和动态物体
    - 核心思路：背景点云 $P^{\text{bg}}$ 通过将输入图像的非物体区域像素反投影到 3D 获得。每个物体的 3D 高斯轨迹 $\{\mathcal{G}_o^t\}_{t=1}^T$ 由均值 $\boldsymbol{\mu}_o^t$（位置）和协方差 $\mathbf{\Sigma}_o^t$（形状/朝向）组成，初始化时对物体点云做全协方差高斯拟合。用户可在 Blender 中将高斯可视化为椭球体，通过拖拽和关键帧指定轨迹。
    - 设计动机：相比 3D bbox（过于刚性）、SMPL-X（仅限人体）、稀疏轨迹点（无形状信息），3D 高斯以概率方式软性描述物体占据，类别无关且低维可编辑。

2. **4D 控制图渲染**：
    - 功能：将 4D 几何状态转为模型可消费的控制信号
    - 核心思路：每帧渲染三类图：(i) 背景 RGB/深度——将 $P^{\text{bg}}$ 在目标相机视角下投影；(ii) 轨迹 RGB/深度——将逐物体高斯投影为软椭圆足迹；(iii) 软融合 mask——通过反转背景可见性并合并高斯足迹得到，标记扩散模型应合成/覆盖的区域。第一帧保留原始输入图像。背景和轨迹通过解耦通道渲染，将相机运动与物体运动分离。
    - 设计动机：解耦渲染确保控制图之间不会互相干扰——背景变化只源于相机运动，轨迹变化只源于物体运动，同时保持几何一致性。

3. **GeoAdapter 架构**：
    - 功能：将 4D 几何控制信号注入冻结的视频扩散模型
    - 核心思路：四组 RGB/深度控制图经冻结 Wan Encoder 编码，软融合 mask 重塑到潜在分辨率，所有几何潜在量按通道拼接形成时空几何张量。GeoAdapter 是一个轻量 DiT 分支，与 Wan-DiT 共享隐藏维度但层数少得多。每隔 $k=5$ 个 DiT block 配对一个 GeoAdapter block，其输出经零初始化线性投影作为残差加到对应 DiT block 上。GeoAdapter block 从配对的 DiT block 权重初始化以稳定训练。
    - 设计动机：adapter 式设计只引入少量额外参数，保持骨干权重冻结，继承 Wan2.1 强大的视频先验。零初始化确保训练初期不扰动骨干。

4. **VerseControl4D 数据集构建**：
    - 功能：提供大规模真实世界视频的 4D 几何控制标注
    - 核心思路：从 Sekai-Real-HQ 和 SpatialVID-HQ 提取 81 帧片段，经物体过滤（Grounded SAM2 检测 1-6 个可控物体）和质量过滤（美学/亮度评分），用 Qwen2.5-VL-72B 生成文本描述，用 MoGe-2 + UniDepth V2 + MegaSAM 估计深度和相机轨迹，重建 3D 点云并拟合高斯轨迹，渲染控制图。共 35K 训练 + 1K 验证样本。
    - 设计动机：解决 4D 控制数据稀缺的瓶颈，全自动化流水线使大规模训练成为可能。

### 损失函数 / 训练策略
- 使用 Adam 优化器，学习率 2e-5，恒定学习率 + 100 步预热
- 分阶段训练：先在 480P 上训练 2,500 步，再在 720P 上微调 2,500 步
- Classifier-free guidance（CFG）训练：以 0.1 概率随机丢弃文本条件
- 推理时使用 50 步去噪，CFG 尺度 5.0
- 16 张 96GB GPU，总训练时间约 380 小时

## 实验关键数据

### 主实验（联合相机+物体运动控制）

| 方法 | Overall Score↑ | Imaging Quality↑ | RotErr↓ | TransErr↓ | ObjMC↓ |
|------|---------------|-----------------|---------|-----------|--------|
| Perception-as-Control | 83.66 | 66.81 | 5.006 | 8.767 | 6.556 |
| Yume | 85.47 | 71.16 | 7.560 | 8.735 | 7.959 |
| Uni3C | 83.55 | 68.06 | 1.361 | 7.731 | 5.883 |
| **VerseCrafter** | **88.10** | **72.70** | **0.890** | **3.103** | **2.507** |

### 消融实验（3D 表示与控制设计）

| 配置 | Overall Score | RotErr | TransErr | ObjMC |
|------|-------------|--------|----------|-------|
| Full (3D Gaussian) | **88.10** | **0.890** | **3.103** | **2.507** |
| 3D Bounding Box | 85.45 | 1.350 | 3.805 | 4.520 |
| 3D Point Trajectory | 85.57 | 1.298 | 3.281 | 6.896 |
| w/o depth | 85.64 | 1.177 | 3.900 | 4.929 |
| BG & FG Merged | 85.72 | 1.080 | 3.803 | 3.726 |

### 关键发现
- **3D 高斯轨迹全面优于 3D bbox 和点轨迹**：ObjMC 分别从 4.520、6.896 降到 2.507，因为高斯提供了形状和朝向信息。点轨迹的 ObjMC 最差（6.896），因为无法编码物体大小。
- **深度信息至关重要**：去掉深度后前后景排序错误（灯柱被拉到建筑物前方），TransErr 从 3.103 升到 3.900。
- **解耦控制优于合并控制**：合并背景和前景控制后物体运动精度显著下降（ObjMC 从 2.507 升到 3.726），因为模型无法区分相机运动和物体运动。
- **静态场景相机控制**：VerseCrafter 的 RotErr（0.650）和 TransErr（2.587）远低于 FlashWorld（1.792/3.257）和 ViewCrafter（2.101/9.868），体现了 4D 几何控制的精确性。

## 亮点与洞察
- **3D 高斯轨迹作为通用物体运动表示**：用概率分布替代刚性几何体，兼顾形状编码和灵活性，可自然处理任意类别的物体。这种表示可迁移到自动驾驶、机器人等需要物体运动预测的场景。
- **解耦渲染思想精巧**：将相机运动（背景变化）和物体运动（前景变化）通过独立控制通道分离，使模型可以分别学习两种运动模式而不混淆。这一思路对任何多信号控制的生成模型都有参考价值。
- **零初始化 + 权重继承的 adapter 训练策略**：GeoAdapter 从配对 DiT block 权重初始化，输出经零初始化线性层注入，既确保训练稳定又让 adapter 快速适应新任务。

## 局限与展望
- **推理成本高**：生成一个 81 帧 720P 视频需要 8 张 96GB GPU 约 1152 秒，离实时应用还很远。
- **依赖单视图深度估计**：背景点云和初始高斯从单张图像重建，大基线视角变化时会出现遮挡区域的缺失。
- **数据集主要是户外/城市场景**：VerseControl4D 来源于 Sekai-Real-HQ 和 SpatialVID-HQ，室内复杂场景的覆盖可能不足。
- **物体交互未建模**：多物体间的碰撞、遮挡等物理交互没有显式约束，可能产生不真实的穿透效果。

## 相关工作与启发
- **vs Yume**: Yume 通过文本/动作 token 控制4D生成，但缺乏精确的相机和物体运动控制（RotErr=7.560 远高于本文的 0.890）。VerseCrafter 用显式几何状态替代隐式控制，精度大幅提升。
- **vs Uni3C**: Uni3C 使用 SMPL-X 控制物体运动，受限于人体类别，且只能控制单人。VerseCrafter 的 3D 高斯轨迹是类别无关的，支持多物体。
- **vs ControlNet**: GeoAdapter 的设计灵感来自 ControlNet 的 adapter 式注入，但扩展到 4D 时空控制，且使用解耦的多通道控制图。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 4D 几何控制表示和 3D 高斯轨迹作为运动控制信号是全新的设计
- 实验充分度: ⭐⭐⭐⭐⭐ 联合控制/相机控制/消融实验全面，定量定性比较充分
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，但公式密度较高
- 价值: ⭐⭐⭐⭐⭐ 为视频世界模型提供了统一的 4D 控制接口，数据集和方法都有很高的复用价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] FaceCam: Portrait Video Camera Control via Scale-Aware Conditioning](facecam_portrait_video_camera_control_via_scale-aware_conditioning.md)
- [\[ICCV 2025\] Geo4D: Leveraging Video Generators for Geometric 4D Scene Reconstruction](../../ICCV2025/3d_vision/geo4d_leveraging_video_generators_for_geometric_4d_scene_reconstruction.md)
- [\[CVPR 2026\] OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery](onlinehmr_video-based_online_world-grounded_human_mesh_recovery.md)
- [\[CVPR 2026\] SceneScribe-1M: A Large-Scale Video Dataset with Comprehensive Geometric and Semantic Annotations](scenescribe-1m_a_large-scale_video_dataset_with_comprehensive_geometric_and_sema.md)
- [\[CVPR 2026\] Iris: Bringing Real-World Priors into Diffusion Model for Monocular Depth Estimation](iris_bringing_realworld_priors_into_diffusion_model_for_monocular_depth_estimation.md)

<!-- RELATED:END -->
