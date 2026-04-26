---
title: >-
  [论文解读] ODHSR: Online Dense 3D Reconstruction of Humans and Scenes from Monocular Videos
description: >-
  [CVPR 2025][3D视觉][人体重建] ODHSR 提出首个统一框架，从单目 RGB 视频中以在线方式同时完成相机追踪、人体姿态估计和人-场景联合稠密重建，基于 3D Gaussian Splatting 实现了比离线方法快 75 倍的速度，且重建质量达到或超越 SOTA。
tags:
  - CVPR 2025
  - 3D视觉
  - 人体重建
  - 场景重建
  - 3D高斯溅射
  - SLAM
  - 单目视频
---

# ODHSR: Online Dense 3D Reconstruction of Humans and Scenes from Monocular Videos

**会议**: CVPR 2025  
**arXiv**: [2504.13167](https://arxiv.org/abs/2504.13167)  
**代码**: https://eth-ait.github.io/ODHSR  
**领域**: 3D视觉  
**关键词**: 人体重建, 场景重建, 3D高斯溅射, SLAM, 单目视频

## 一句话总结

ODHSR 提出首个统一框架，从单目 RGB 视频中以在线方式同时完成相机追踪、人体姿态估计和人-场景联合稠密重建，基于 3D Gaussian Splatting 实现了比离线方法快 75 倍的速度，且重建质量达到或超越 SOTA。

## 研究背景与动机

**领域现状**：从单目视频重建三维世界一直是计算机视觉的核心问题。现有方法要么只关注人体重建（如 3DGS-Avatar、GauHuman），要么只关注场景重建（如 MonoGS 等 SLAM 方法），少数同时处理人和场景的方法（如 HSR、HUGS）要么需要预标定的相机位姿和人体位姿，要么需要数天的训练时间。

**现有痛点**：HSR 基于隐式神经场（NeRF/SDF），训练极慢（数天级别）且渲染质量有限；HUGS 虽然用了 3D Gaussian Splatting，但其 triplane 方案在在线设定下特征收敛缓慢，不适合实时处理；Vid2Avatar 场景重建效果差且易产生大量伪影。更重要的是，这些方法都是离线的，无法满足机器人等实时应用需求。

**核心矛盾**：在线设定下，如何从仅有的单目 RGB 视频中同时解耦相机运动、人体运动并重建高保真的人-场景表示？视角稀少、人体遮挡、动态衣物、光照变化等因素使得这一问题极具挑战性。

**本文目标**：设计一个统一在线框架，同时输出（1）相机轨迹，（2）人体全局姿态，（3）人和场景的稠密光度学重建，且不依赖任何预标定信息。

**切入角度**：3D Gaussian Splatting 提供了直接的显式梯度流，允许联合优化高斯参数和位姿参数；同时利用单目深度先验和 SMPL 人体先验来建立人与场景之间的空间关联。

**核心 idea**：将 3DGS-based SLAM 扩展为人-场景联合框架，通过遮挡感知的人体轮廓渲染和单目几何先验，在一个在线 pipeline 中同时完成追踪、位姿估计和稠密重建。

## 方法详解

### 整体框架

输入为单目 RGB 视频序列。对每一帧，系统首先通过已有的高斯表示渲染当前帧的预测图像，计算残差来联合优化相机位姿和人体位姿（Tracking 线程）。通过关键帧筛选后，在 Mapping 线程中用局部关键帧窗口联合训练人体和场景的 3D 高斯表示。最后在所有关键帧上执行全局 Bundle Adjustment 微调整体表示。系统包含 Tracking 和 Mapping 两个并行线程以保证效率。

### 关键设计

1. **3D Avatar 表示（刚性+非刚性变形）**:

    - 功能：在规范空间中用 3D 高斯表示人体，并通过 SMPL 驱动的骨架变形将其变换到世界坐标
    - 核心思路：每个高斯具有位置、偏移、旋转、尺度、不透明度、颜色和可学习的 LBS 权重。变形分解为刚性（SMPL 关节驱动的 LBS）和非刚性（时间-姿态依赖的局部变形）两部分。非刚性变形通过多分辨率哈希编码网络 $F_\phi$ 建模，输入为高斯中心、时间步和姿态参数，输出局部位移 $\Delta\mu'_H$、旋转偏移 $\Delta R_H$ 和环境遮蔽因子 $\Delta c_H$。再通过 LBS 加权关节变换 $P = \sum_j W_{H,j} M_j$ 将规范空间高斯变换到世界坐标
    - 设计动机：纯骨架变形只能建模关节刚性运动，无法处理衣物褶皱等动态形变；非刚性模块用哈希编码网络而非大型 MLP，兼顾表达力和训练速度

2. **遮挡感知的人体轮廓渲染**:

    - 功能：在联合渲染中正确处理人与场景的遮挡关系，生成真实的人体轮廓
    - 核心思路：将人体和场景高斯合并为全局集合 $G = G_S + G_H$ 输入光栅化器。由于高斯按深度排序，自然处理遮挡。人体轮廓 $\hat{O}_H = \sum_j \alpha_j \prod_{k=1}^{N_j}(1-\alpha_k)$，其中 $N_j$ 包含所有深度小于第 $j$ 个人体高斯的高斯体（含场景高斯）。这样当场景物体在人前面时，人体轮廓会正确被遮挡
    - 设计动机：简单的人体 mask 渲染忽略了场景遮挡，导致人-场景解耦失败。遮挡感知渲染让轮廓监督信号更准确，有效引导人体与场景高斯的分离

3. **联合位姿优化（Tracking）**:

    - 功能：同时估计每帧的相机位姿 $T$ 和人体姿态 $\theta$
    - 核心思路：固定高斯表示，通过多个互补损失联合优化。包括 RGB 光度损失、光流损失（用于避免 RGB 损失的局部最优，仅在静态区域计算）、单目深度一致性损失（在逆深度空间对齐渲染深度和预测深度）、人体轮廓损失、2D 关键点损失。最终损失为所有项的加权和 $L_{pose} = \lambda_{rgb}L_{rgb} + \lambda_{flow}L_{flow} + \lambda_{disp}L_{disp} + \lambda_{sil}L_{sil} + \lambda_{kp}L_{kp}$
    - 设计动机：单一损失不足以约束如此高维的位姿空间。光流提供帧间运动约束，单目深度提供几何先验，轮廓和关键点提供人体特有的约束，多信号互补确保收敛到正确解

### 损失函数 / 训练策略

Mapping 阶段额外引入三个正则化：（1）LBS 权重正则化 $L_{LBS}$，用 SMPL 的蒙皮权重监督避免过拟合；（2）规范中心正则化 $L_{center}$，防止高斯偏移过大；（3）局部变形正则化 $L_{deform}$，惩罚变形幅度以稳定训练。Mapping 使用局部关键帧窗口加两个随机历史关键帧，兼顾新观测和全局记忆。

初始化策略：首帧通过 WHAM 获取初始人体姿态，用 ViTPose 的 2D 关键点 loss 精细化。利用单目深度估计器的深度与 SMPL 网格深度通过 RANSAC 对齐求解尺度和偏移，从而用有尺度的深度初始化场景高斯。

## 实验关键数据

### 主实验

| 数据集/指标 | ODHSR (Full) | HUGS | HSR | Vid2Avatar | 3DGS-Avatar |
|------------|-------------|------|-----|------------|-------------|
| EMDB 全图 PSNR↑ | **23.790** | 21.605 | 18.675 | 16.656 | - |
| EMDB 全图 SSIM↑ | **0.767** | 0.659 | 0.463 | 0.413 | - |
| EMDB 人体 PSNR↑ | **28.955** | 26.165 | 25.127 | 24.258 | 27.952 |
| NeuMan 全图 PSNR↑ | **26.470** | 26.667 | 21.676 | 15.640 | - |
| 训练 FPS↑ | **0.141** | 0.042 | 0.002 | <0.001 | 0.112 |
| 渲染 FPS↑ | **85** | 40 | 0.05 | 0.02 | 60 |

### 消融实验

| 配置 | PSNR↑ | ATE RMSE↓ | WA-MPJPE↓ |
|------|-------|-----------|-----------|
| Full model | **23.790** | **0.084** | **175.215** |
| w/o $L_{flow}$ | 22.593 | 0.214 | 301.621 |
| w/o $L_{keypoint}$ | 22.263 | 0.121 | 230.875 |
| w/o $L_{disp}$ | 22.769 | 0.165 | 252.547 |
| w/o $L_{sil}$ | 22.648 | 0.148 | 240.838 |

### 关键发现

- 去掉光流损失对相机追踪影响最大（ATE 从 0.084 增至 0.214），因为帧间运动约束是追踪的核心
- 去掉关键点损失对全局人体姿态估计影响最大（WA-MPJPE 从 175 增至 231），关键点提供了最直接的人体对齐信号
- 四个损失之间存在协同效应：关键点和轮廓加速人体位姿收敛，准确的位姿促进深度对齐，清晰的几何又反过来改善光流和深度一致性
- 在线方法的重建质量全面超越离线的 HSR（基于 NeRF/SDF）和 HUGS（基于 triplane），说明显式高斯表示+联合优化的策略优势明显
- 人体姿态估计达到 WA-MPJPE 175mm，远超初始化的 WHAM（636mm），证明重建优化后的位姿精化非常有效

## 亮点与洞察

- **在线统一框架的首创性**：将追踪、位姿估计、重建三个独立任务统一到一个 3DGS-SLAM 框架中，比 HSR 快 75 倍却质量更好。这种系统级的设计思路值得借鉴
- **人作为尺度参考**：利用 SMPL 人体的已知尺寸来校正单目深度的尺度，巧妙解决了单目系统的尺度模糊问题。这个 trick 可以迁移到任何有已知物体尺寸的单目重建场景
- **遮挡感知轮廓的简洁实现**：不需要额外网络，直接利用高斯溅射光栅化器的深度排序特性实现正确遮挡，既高效又自然

## 局限与展望

- 依赖 WHAM 和 ViTPose 等预训练模型提供初始化，在这些模型失效的场景（如严重遮挡、非常规姿态）下可能表现不佳
- 目前仅支持单人场景，多人场景需要额外的人体检测和分割
- 关键帧选择策略依赖启发式规则，可能遗漏信息量大的帧
- 非刚性变形的哈希编码网络对时间外推敏感，长序列可能出现漂移

## 相关工作与启发

- **vs HSR**: HSR 用 NeRF/SDF 隐式表示做人-场景联合建模，训练需数天；本文用显式高斯表示，训练和渲染速度快几个数量级，且重建质量更好
- **vs HUGS**: HUGS 也用 3DGS 但采用 triplane 表示人体，离线设定下需要预标定位姿；本文在线运行且用更简洁的规范空间高斯+哈希变形
- **vs 3DGS-Avatar**: 专注人体重建且需要已知相机位姿；本文同时解决场景重建和位姿估计，更通用

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个在线统一框架，但各组件（3DGS SLAM、SMPL变形、单目先验）单独来看并非全新
- 实验充分度: ⭐⭐⭐⭐ 在两个数据集上全面对比，消融实验清晰，但缺少多人场景和更大规模的评估
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，动机和方法描述流畅，图表信息量大
- 价值: ⭐⭐⭐⭐ 对人-场景联合重建的实时化有重要推动，系统设计有很好的工程参考价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos](slam3r_real-time_dense_scene_reconstruction_from_monocular_rgb_videos.md)
- [\[CVPR 2025\] SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes](spectromotion_dynamic_3d_reconstruction_of_specular_scenes.md)
- [\[CVPR 2025\] Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)
- [\[CVPR 2025\] A Unified Image-Dense Annotation Generation Model for Underwater Scenes](a_unified_image-dense_annotation_generation_model_for_underwater_scenes.md)
- [\[CVPR 2025\] HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos](hawor_world-space_hand_motion_reconstruction_from_egocentric_videos.md)

<!-- RELATED:END -->
