---
title: >-
  [论文解读] StreetCrafter: Street View Synthesis with Controllable Video Diffusion Models
description: >-
  [CVPR 2025][街景合成] 提出 StreetCrafter，利用 LiDAR 点云渲染作为像素级条件来控制视频扩散模型，实现精确相机控制的街景新视角合成，并可将生成先验蒸馏到动态 3DGS 表示中实现实时渲染。
tags:
  - CVPR 2025
  - 街景合成
  - 视频扩散模型
  - LiDAR条件
  - 新视角合成
  - 3DGS蒸馏
---

# StreetCrafter: Street View Synthesis with Controllable Video Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2412.13188](https://arxiv.org/abs/2412.13188)  
**代码**: [GitHub](https://zju3dv.github.io/street_crafter)  
**领域**: 自动驾驶  
**关键词**: 街景合成, 视频扩散模型, LiDAR条件, 新视角合成, 3DGS蒸馏

## 一句话总结

提出 StreetCrafter，利用 LiDAR 点云渲染作为像素级条件来控制视频扩散模型，实现精确相机控制的街景新视角合成，并可将生成先验蒸馏到动态 3DGS 表示中实现实时渲染。

## 研究背景与动机

自动驾驶模拟器需要高质量的视角合成能力。基于 3DGS 的方法在训练轨迹附近可生成高质量图像，但当视角偏离训练轨迹时产生严重伪影——这源于训练数据中对偏移区域的观测不足和重建方法外推能力有限。

视频扩散模型具有强大的生成先验，可从少量输入图像生成逼真视角。但现有方法依赖文本提示作为控制信号，是高层级指令，缺乏精细控制能力，不适用于自动驾驶场景。

核心观察是：LiDAR 点云渲染提供了精确的几何信息，尽管不完整且有噪声，但可作为精确的相机位姿表示。通过将点云渲染作为视频扩散模型的像素级条件，可以在保持精细相机控制的同时充分利用生成先验。实验发现，即使仅在单车道驾驶数据上训练，也能在测试时生成跨多车道的高质量视图。

## 方法详解

### 整体框架

StreetCrafter 基于 Vista（从 SVD 微调的驾驶世界模型）构建，包含三个核心组件：(1) LiDAR 条件构建——聚合相邻帧彩色 LiDAR 形成全局点云并渲染到目标视角；(2) 可控视频扩散模型——将 LiDAR 渲染图作为像素级条件注入 U-Net；(3) 动态 3DGS 蒸馏——利用扩散模型生成新视角图像作为额外监督。

### 关键设计一：LiDAR 像素级条件构建

**功能**：为视频扩散模型提供精确的像素级位姿控制信号

**核心思路**：将 LiDAR 点云投影到标定图像平面获取颜色，利用目标跟踪分离动态物体和静态背景。给定新相机位姿 $\mathbf{C}_i$，在时间窗口 $l$（$\pm 1$s）内聚合 LiDAR 点形成统一点云 $\mathbf{P}$，动态物体点云通过位姿 $\mathbf{T}_o^{t_i}$ 变换到世界坐标系。为每个点分配固定半径进行点光栅化，生成条件图像 $\mathbf{I}^c_i$。

**设计动机**：相比相机参数嵌入等抽象控制信号，LiDAR 渲染图以像素对齐方式建立新轨迹与输入图像的连接，网络只需从带噪条件恢复干净图像，无需学习从相机参数到视频帧的复杂映射过程。

### 关键设计二：零卷积条件注入与训练

**功能**：将 LiDAR 条件高效注入视频扩散模型

**核心思路**：输入图像和 LiDAR 条件通过预训练 VAE 编码器编码为潜空间表示 $\{\mathbf{z}_i\}$ 和 $\{\mathbf{z}^c_i\}$。通过可训练的零卷积层 $\Theta_z$ 处理 LiDAR 潜码，并与噪声潜码逐元素相加：

$$\hat{\mathbf{z}}_{i,t} = \mathbf{z}_{i,t} + \mathcal{Z}(\mathbf{z}^c_i; \Theta_z)$$

训练损失为标准扩散去噪目标：$\mathcal{L} = \mathbb{E}[\|\mathbf{z}_i - \mathcal{F}_\theta(\hat{\mathbf{z}}_{i,t}, t, \mathbf{c}_{\text{ref}}, \mathbf{c}_p)\|_2^2]$

**设计动机**：零初始化保证初始输出与原始扩散模型一致，训练时仅引入最小修改即可提供充分引导，不增加额外计算代价。训练中以 15% 概率独立丢弃参考图像和 LiDAR 条件以支持 classifier-free guidance。

### 关键设计三：渐进式 3DGS 蒸馏

**功能**：将扩散模型先验蒸馏到 3DGS 表示以实现实时渲染

**核心思路**：从 3DGS 渲染的有伪影图像开始，编码并加噪后通过扩散模型去噪得到高质量新视角图像作为额外监督。采用渐进优化策略——逐步降低噪声尺度 $s$：早期训练依赖扩散先验去除伪影，后期逐步细化细节。新视角使用 LPIPS 损失强调语义层面一致性：

$$\mathcal{L}_{\text{novel}} = \lambda_{\text{novel}} \mathcal{L}_{\text{lpips}}$$

**设计动机**：从带噪渲染结果出发（而非纯噪声）有助于保持整体场景结构并减少去噪步骤。结合 3D 表示的一致性和扩散模型的生成能力，实现最佳性能。

### 损失函数

输入视角：$\mathcal{L}_{\text{input}} = \lambda_1 \mathcal{L}_1 + \lambda_{\text{ssim}} \mathcal{L}_{\text{ssim}} + \lambda_{\text{lpips}} \mathcal{L}_{\text{lpips}} + \mathcal{L}_g$，其中 $\mathcal{L}_g$ 包含 LiDAR 深度损失、天空掩码损失和运动目标正则化。新视角：$\mathcal{L}_{\text{novel}} = 0.1 \cdot \mathcal{L}_{\text{lpips}}$。

## 实验关键数据

### 主实验：Waymo Open Dataset 新视角合成

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS |
|------|-------|-------|--------|-----|
| 3DGS + LiDAR | 26.87 | 0.851 | 0.182 | 实时 |
| Street Gaussians | 27.52 | 0.862 | 0.171 | 实时 |
| EmerNeRF | 26.21 | 0.843 | 0.195 | - |
| NeuRAD | 27.14 | 0.858 | 0.178 | - |
| **StreetCrafter-V** | **28.34** | **0.874** | **0.148** | 0.2fps |
| **StreetCrafter-G** | **28.91** | **0.883** | **0.139** | **实时** |

### 视角外推对比（lateral shift 3m）

| 方法 | PSNR↑ | LPIPS↓ |
|------|-------|--------|
| Street Gaussians | 22.71 | 0.312 |
| NeuRAD | 23.45 | 0.287 |
| **StreetCrafter-G** | **25.83** | **0.198** |

### 关键发现

- 在视角外推（偏移 3m）设置下，StreetCrafter-G 比 Street Gaussians 提升 **+3.1 PSNR**，显示扩散先验对分布外视角的巨大优势
- 仅在单车道数据训练即可泛化到多车道测试视角
- LiDAR 条件也可用于无需逐场景优化的场景编辑（物体移除、替换、平移）
- 蒸馏后的 3DGS 版本（Ours-G）在保持实时渲染的同时，质量超过纯扩散方法（Ours-V）

## 亮点与洞察

1. **LiDAR 作为像素级位姿条件**：相比 camera pose embedding，LiDAR 渲染图提供了更强的像素对齐引导信号，极大简化了网络的学习难度
2. **扩散-重建互补**：将扩散模型的泛化能力与 3DGS 的一致性/实时性结合，通过渐进蒸馏策略实现两者优势互补
3. **场景编辑零成本**：LiDAR 条件的可操控性使得场景编辑（物体移除/替换）不需要逐场景优化

## 局限与展望

- 扩散模型推理延迟高（576×1024 分辨率仅 0.2fps），实际应用必须依赖蒸馏
- 需要高质量 LiDAR 数据和准确的目标跟踪作为输入
- 长距离视角偏移或极端天气下的泛化能力未充分验证
- 未来可探索与更高效的扩散模型（如 consistency models）结合

## 相关工作与启发

- **Vista**：基础的驾驶世界模型，StreetCrafter 在此基础上添加 LiDAR 条件控制
- **Street Gaussians**：动态场景 3DGS 建模的基线方法，StreetCrafter 在蒸馏阶段基于此框架
- **ReconFusion / CAT3D**：利用扩散模型辅助 3D 重建的思路类似，但针对静态场景且缺乏精确相机控制

## 评分

⭐⭐⭐⭐ — 方法设计思路清晰，LiDAR 作为像素级条件的创意新颖且实用。在 Waymo 上的视角外推实验令人印象深刻。扩散-3DGS 蒸馏框架具有良好的工程价值。对自动驾驶仿真有实际推动意义。

<!-- RELATED:START -->

## 相关论文

- [InterDyn: Controllable Interactive Dynamics with Video Diffusion Models](interdyn_controllable_interactive_dynamics_with_video_diffusion_models.md)
- [SpatialDreamer: Self-supervised Stereo Video Synthesis from Monocular Input](spatialdreamer_self-supervised_stereo_video_synthesis_from_monocular_input.md)
- [MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling](mimo_controllable_character_video_synthesis_with_spatial_decomposed_modeling.md)
- [SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion](../../ECCV2024/video_generation/sv3d_novel_multi-view_synthesis_and_3d_generation_from_a_single_image_using_late.md)
- [LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis](levitor_3d_trajectory_oriented_image-to-video_synthesis.md)

<!-- RELATED:END -->
