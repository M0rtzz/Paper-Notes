---
title: >-
  [论文解读] RayNova: Scale-Temporal Autoregressive World Modeling in Ray Space
description: >-
  [CVPR 2026][3D视觉][世界模型] 提出 RayNova，一种基于双因果（尺度+时间）自回归的几何无关多视角世界模型，利用相对 Plücker 光线位置编码实现统一的 4D 时空推理，在 nuScenes 上取得 SOTA 多视角视频生成效果。
tags:
  - CVPR 2026
  - 3D视觉
  - 世界模型
  - 多视角视频生成
  - 自回归
  - Plücker光线
  - 自动驾驶
---

# RayNova: Scale-Temporal Autoregressive World Modeling in Ray Space

**会议**: CVPR 2026  
**arXiv**: [2602.20685](https://arxiv.org/abs/2602.20685)  
**作者**: Yichen Xie, Chensheng Peng, Mazen Abdelfattah, Yihan Hu 等 (Applied Intuition, UC Berkeley)  
**代码**: [项目页](https://raynova-ai.github.io/)  
**领域**: 3D视觉  
**关键词**: 世界模型, 多视角视频生成, 自回归, Plücker光线, 自动驾驶

## 一句话总结

提出 RayNova，一种基于双因果（尺度+时间）自回归的几何无关多视角世界模型，利用相对 Plücker 光线位置编码实现统一的 4D 时空推理，在 nuScenes 上取得 SOTA 多视角视频生成效果。

## 背景与动机

世界基础模型 (WFM) 旨在模拟真实世界的物理演化。现有方法存在根本限制：

1. **空间-时间解耦设计**：空间用多视角邻接关系，时间用视频生成技术，分别处理，限制了对新相机配置和快速运动的适应性
2. **强 3D 先验依赖**：依赖点云/BEV 等显式 3D 表示，限制了在开放世界的泛化能力
3. **固定相机配置绑定**：多数方法假设固定的传感器布局和相邻关系

## 核心问题

如何在保持物理合理性的同时，以最小归纳偏置构建可泛化到任意相机配置和运动的世界模型？

## 方法详解

### 3.1 Next-Scale Prediction（基础）

基于视觉自回归模型，将每张图像量化为 $K$ 个多尺度 token map $X_{1:K}$，从粗到细自回归生成：

$$p(X_1, \ldots, X_K) = \prod_{k=1}^K p(X_k | X_1, \ldots, X_{k-1})$$

### 3.2 双因果自回归

**尺度因果性**：同一帧的所有视图联合建模（因为它们描述同一 3D 空间），按尺度递进生成：

$$p(X_1^{1:V}, \ldots, X_K^{1:V}) = \prod_{k=1}^K p(X_k^{1:V} | X_1^{1:V}, \ldots, X_{k-1}^{1:V})$$

**时间因果性**：当前帧以所有历史帧的所有视图为条件，不假设同相机帧间的强依赖：

$$p(X_{1:K}^{1:V,1:T}) = \prod_{t=1}^T \prod_{k=1}^K p(X_k^{1:V,t} | X_{1:k-1}^{1:V,1:t})$$

### 3.3 各向同性时空表示

**核心创新**：基于相对 Plücker 光线的旋转位置编码 (RoPE)。

对每个 token，计算其 Plücker 光线 $\mathbf{p}_k^{v,t} = (\mathbf{m}, \mathbf{d}, t) \in \mathbb{R}^7$，其中 $\mathbf{m} = \mathbf{o}^{v,t} \times \mathbf{d}_k^{v,t}$。

将 RoPE 扩展到 7D 空间：

$$\mathbf{R} = \begin{bmatrix} \mathbf{R_m} & 0 & 0 \\ 0 & \mathbf{R_d} & 0 \\ 0 & 0 & \text{RoPE}_{d/4}(t) \end{bmatrix}$$

注意力分数基于 token 间的**相对**位置：

$$a_{i,j} = \mathbf{q}_i^T \mathbf{R}_\Delta^{i,j} \mathbf{k}_j, \quad \mathbf{R}_\Delta^{i,j} = \mathbf{R}_i^T \mathbf{R}_j$$

**关键优势**：
- 对所有尺度/视图/帧各向同性，无特定相机配置假设
- 相对编码天然支持外推到训练分布之外

### 3.4 Transformer 架构

每个 block 包含三层注意力：
1. **Image-wise self-attention**：每张图独立处理，配合 2D Axial RoPE，保证图像真实性
2. **Global self-attention**：跨视图跨帧的统一注意力 + Plücker 光线 RoPE，保证时空一致性
3. **Image-wise cross-attention**：融合文本/3D bbox/HD map 等条件

条件处理：bbox 投影 8 个角点到图像空间编码 + T5 文本嵌入；地图采样 3D 点后投影 + PointNet 编码。

### 3.5 长视频递归训练

为解决长视频生成中的分布漂移，提出递归训练策略：
- 逐帧前向/反向传播，梯度累积后统一更新
- 缓存 latent 特征（而非 KV）→ 节省 50% GPU 显存，保留 KV 投影层的梯度
- 在 visual token 输入中引入随机位翻转噪声模拟推理误差

## 实验关键数据

| 方法 | 分辨率 | FID ↓ | FVD ↓ | 吞吐量 ↑ (img/s) |
|------|--------|-------|-------|----------------|
| MagicDrive | 224×400 | 16.2 | - | 1.76 |
| DriveDreamer | 256×448 | 14.9 | 341 | 0.37 |
| Panacea | 256×512 | 17.0 | 139 | 0.67 |
| **RayNova** | **384×672** | **10.5** | **91** | **1.96** |

| 评估维度 | 方法 | 指标 (相对 Oracle) |
|---------|------|-------------------|
| 目标条件 (StreamPETR) | Panacea | 32.1 NDS (68%) |
| | **RayNova** | **41.9 NDS (89%)** |
| 目标条件 (SparseFusion) | X-Drive | 69.6 NDS (95%) |
| | **RayNova** | **72.0 NDS (99%)** |
| 新视角合成 FID (shift 4m) | StreetGaussian | 67.44 |
| | **RayNova** | **17.48** |

## 亮点

- **几何无关设计**：不依赖点云/BEV/深度等 3D 先验，仅通过相对光线位置编码实现几何感知
- **双因果自回归**：统一的尺度+时间因果框架，比解耦的空间-时间注意力更灵活
- **超强新视角泛化**：零样本适配未见相机配置，4m 位移下 FID 仅 17.48 vs StreetGaussian 67.44
- **高效生成**：1.96 img/s 吞吐量远超扩散模型 baseline（0.37-1.76）
- **异构数据兼容**：可混合使用不同传感器配置/分辨率/帧率的训练数据

## 局限与展望

- 使用基于图像的 VAE，可能影响 FID/FVD 指标
- 训练数据量（~60小时）相比一些私有数据方法仍有限
- 递归训练需要更长的训练时间
- 地图条件的 3D 点投影缺乏高度信息
- 实验仅在驾驶场景验证，未验证室内等其他场景

## 与相关工作的对比

- vs **Panacea**：Panacea 假设多帧同相机的强依赖关系，受限于特定相机配置；RayNova 完全解耦，FVD 91 vs 139
- vs **X-Drive**：X-Drive 用点云作为 3D 先验，RayNova 无需任何 3D 表示
- vs **StreetGaussian/OmniRe**：显式 3D 表示在大幅相机偏移下急剧退化（FID 67+），RayNova 保持稳健（17.48）
- vs **BEVWorld**：BEV 表示绑定于特定高度平面，RayNova 的光线空间更通用

## 启发与关联

- 相对 Plücker 光线编码的设计思路可推广到其他需要几何感知的生成任务
- 双因果自回归为多模态/多分辨率生成提供了统一框架
- 递归训练解决分布漂移的方案对其他长序列生成任务有借鉴意义
- 与 VAR (Visual Autoregressive Model) 的结合值得关注

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 双因果自回归 + 相对光线位置编码是全新的设计范式
- 实验充分度: ⭐⭐⭐⭐ — 多维度评估（质量/条件/新视角/运动），但仅限驾驶场景
- 写作质量: ⭐⭐⭐⭐⭐ — 数学推导严密，图示优秀
- 价值: ⭐⭐⭐⭐⭐ — 开创了几何无关世界模型的新方向

<!-- RELATED:START -->

## 相关论文

- [DuoMo: Dual Motion Diffusion for World-Space Human Reconstruction](duomo_dual_motion_diffusion_for_world-space_human_reconstruction.md)
- [OpenVO: Open-World Visual Odometry with Temporal Dynamics Awareness](openvo_open-world_visual_odometry_with_temporal_dynamics_awareness.md)
- [Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)
- [GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport](glint_modeling_scene-scale_transparency_via_gaussian_radiance_transport.md)
- [UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling](../../ICCV2025/3d_vision/ust-ssm_unified_spatio-temporal_state_space_models_for_point_cloud_video_modelin.md)

<!-- RELATED:END -->
