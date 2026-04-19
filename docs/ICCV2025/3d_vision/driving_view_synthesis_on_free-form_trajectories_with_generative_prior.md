---
title: >-
  [论文解读] DriveX: Driving View Synthesis on Free-form Trajectories with Generative Prior
description: >-
  [ICCV 2025][3D视觉][自动驾驶仿真] 提出驾驶视图合成框架DriveX，通过渐进式地将视频扩散模型的生成先验蒸馏到3DGS表示中——设计inpainting-based视频修复任务来生成新轨迹伪标注，迭代优化3D重建，实现自由轨迹上的高质量实时渲染。
tags:
  - ICCV 2025
  - 3D视觉
  - 自动驾驶仿真
  - 自由轨迹视图合成
  - 3D高斯
  - 视频扩散
  - 生成先验蒸馏
  - 图像修复
---

# DriveX: Driving View Synthesis on Free-form Trajectories with Generative Prior

**会议**: ICCV 2025  
**arXiv**: [2412.01717](https://arxiv.org/abs/2412.01717)  
**代码**: [https://fudan-zvg.github.io/DriveX](https://fudan-zvg.github.io/DriveX)  
**领域**: 3D视觉  
**关键词**: 自动驾驶仿真, 自由轨迹视图合成, 3D高斯, 视频扩散, 生成先验蒸馏, Inpainting修复

## 一句话总结
提出驾驶视图合成框架DriveX，通过渐进式地将视频扩散模型的生成先验蒸馏到3DGS表示中——设计inpainting-based视频修复任务来生成新轨迹伪标注，迭代优化3D重建，实现自由轨迹上的高质量实时渲染。

## 研究背景与动机

### 核心问题
构建可与驾驶策略交互的虚拟驾驶世界（driving simulation）是开发鲁棒自动驾驶系统的关键。需求包括：高效训练数据扩充、安全关键长尾场景合成、端到端驾驶系统闭环评估。**核心瓶颈**：现有重建方法在录制轨迹上效果好，但在自由新轨迹（如变道、平移）上会严重退化。

### 驾驶场景的独特挑战
相比通用新视图合成，驾驶场景面临：
- **极度稀疏视角**：前向行驶的单轨迹视频，帧间重叠极小
- **大面积无纹理区域**：道路、天空等
- **有限轨迹多样性**：数据集中几乎只有直行轨迹，缺乏偏离轨迹的视图

### 现有方法不足

**纯重建方法**（StreetGaussian, PVG等）：录制轨迹上表现好但外推能力差，偏离轨迹即产生伪影

**纯生成方法**（视频世界模型）：多样性好但缺少底层3D表示，无法保证多轨迹视频的几何/纹理一致性

**重建+生成方法**的困难：
   - SGD/VEGS：精确姿态控制在驾驶数据有限多样性下难以学习
   - DriveDreamer4D：仅用单帧参考生成整段新轨迹视频，约束不足导致幻觉和不一致
   - FreeSim/Difix3D+：需要精心策划训练数据来模拟偏离轨迹渲染的退化模式

### 核心洞察

**将问题转化为inpainting任务**：无需让生成模型识别退化区域，直接用不可靠性mask告诉模型哪些区域需要修复。这样：
1. 训练时无需模拟特定退化模式
2. 生成模型可以专注于其擅长的语义填充
3. 渲染结果和修复结果可以互相强化

## 方法详解

### 整体框架（重建-生成交织）
在3DGS优化过程中，交替执行：
1. 用录制轨迹真值监督3DGS优化（标准流程）
2. 用视频扩散模型修复新轨迹渲染→生成伪真值→作为额外监督优化3DGS
3. 每 $K$ 步更新伪真值，两者协同进化

### 关键设计

#### 1. 新轨迹伪真值生成（Video Restoration）
给定新轨迹 $\mathcal{T}'$，从当前训练中的3DGS $\mathcal{G}_t$ 渲染视频 $\mathcal{V}_t'$（含伪影），用视频扩散模型修复：
$$\mathcal{V}'_{t,refine} = \mathcal{D}(\mathcal{V}'_t, \mathcal{M})$$
修复后的视频作为伪真值，通过photometric loss优化3DGS：
$$\mathcal{L}' = \mathcal{L}_{img}(\mathcal{V}'_t, \mathcal{V}'_{t,refine})$$

#### 2. 不可靠性Mask构造（核心创新）
通过几何一致性检测渲染图像中的伪影区域：
- 将录制轨迹上最近的真实图像 $I_{rec}$ 通过渲染深度 $D_{ren}$ 做图像warping到新视角
- Warp公式：$(x,y,d) = \psi(p|P)$ 投影到 $I_{rec}$，采样颜色得到伪图像 $\hat{I}_{ren}$
- 用SSIM比较渲染图和warped图的结构相似度：
$$\mathcal{M} = \mathbb{1}(\text{SSIM}(I_{ren}, \hat{I}_{ren}) < \tau)$$
$\tau = 0.65$。因为warping同时涉及渲染深度和渲染图像，不一致即指示几何或外观不可靠区域。

#### 3. 渐进迭代精化
不是一次性生成伪真值，而是在优化过程中每 $K=3000$ 步更新一次：
- 先用标准重建训练 $T_0=50000$ 步
- 后续30000步引入生成先验
- 随3DGS质量提升，渲染给扩散模型的条件也更好→修复结果也更好→正反馈循环

#### 4. 新轨迹采样策略
采用横向平移相机轨迹（panning camera），从录制前视图 $P'_0$ 开始逐步横移：
$$P'_i = [R_0 | \frac{i}{F} s \mathbf{v} + T_0]$$
其中 $s$ 控制最大平移距离，$\mathbf{v}$ 为平移方向。关键设计：
- 初始帧即为录制视图→为生成模型提供精确参考
- $s$ 渐进增大→早期优化不佳时避免过远偏移
- 利用多相机（侧视图辅助前视图不可见区域的mask）

#### 5. Inpainting-based生成模型训练
训练策略巧妙避开了驾驶数据缺乏多样轨迹的问题：
- 用**边缘感知策略**在训练图像上生成mask：Sobel边缘检测→边缘概率抽样→3×3 patch遮挡
- 前视和侧视相机视频均可用于训练（侧视相机在前进轨迹上的视角类似于前视相机的横向偏移）
- 基于[39]初始化后在WOD上微调15000步

#### 6. LiDAR辅助
- 将相邻±2帧的LiDAR点投影到新视角作为彩色LiDAR投影条件
- 动态物体的LiDAR点按跟踪姿态对齐
- LiDAR深度作为新视角的稀疏深度监督

### 总损失
$$\mathcal{L}_{total} = \mathcal{L}_{img}(\mathcal{V}_t, \mathcal{V}_{gt}) + \mathcal{L}_{img}(\mathcal{V}'_t, \mathcal{V}'_{k,refine})$$

## 实验

### 主实验：Waymo数据集新轨迹合成

| 方法 | ±0m PSNR↑ | ±1m IoU↑ | ±2m IoU↑ | ±3m IoU↑ | ±3m FID↓ | FPS |
|------|------|------|------|------|------|------|
| EmerNeRF | 29.60 | 0.1147 | 0.0881 | 0.0735 | 122.83 | 0.12 |
| PVG | 29.98 | 0.1082 | 0.0291 | 0.0170 | 123.53 | 48 |
| StreetGaussian | 29.76 | 0.2671 | 0.2122 | 0.1720 | 93.04 | 34 |
| **DriveX** | 29.74 | **0.2880** | **0.2710** | **0.2699** | **79.78** | 34 |

关键发现：
- ±3m偏移下IoU提升56.9%（vs StreetGaussian），FID降低14.3%
- 推理时仅用3DGS渲染，速度与StreetGaussian一致（34 FPS）
- 录制轨迹质量基本持平，证明生成先验未干扰原始重建

### 与生成方法对比

| 方法 | Lane change NTA-IoU↑ | ±3m NTA-IoU↑ | ±3m NTL-IoU↑ |
|------|------|------|------|
| DriveDreamer4D | 0.495 | 0.340 | 51.32 |
| ReconDreamer | 0.554 | 0.539 | 54.58 |
| **DriveX** | **0.620** | **0.567** | **58.29** |

优于DriveDreamer4D的单帧参考生成和ReconDreamer。

### 消融实验

**不可靠性Mask的作用**：

| 配置 | IoU↑ | FID↓ |
|------|------|------|
| mask all ($\tau=1.0$) | 0.2153 | 102.69 |
| w/o mask ($\tau=-1.0$) | 0.2193 | 76.26 |
| **w/ mask ($\tau=0.65$)** | **0.2263** | **74.34** |

全部mask → 扩散模型从头生成→不一致；不用mask → 模型需自行识别退化→效果次优。

**迭代精化的更新频率**：

| Buffer间隔K | IoU↑ | FID↓ | 时间倍率 |
|------|------|------|------|
| 500 | 0.2271 | 75.33 | 4.8× |
| 3000 | 0.2263 | 74.34 | 1.6× |
| 6000 | 0.2228 | 75.98 | 1.3× |

K=3000在质量和效率间取得良好平衡。

## 亮点与洞察
1. **Inpainting公式化的精巧设计**：将"如何识别退化"和"如何修复"解耦——mask处理前者，扩散模型处理后者。避免了模拟退化模式的困境
2. **渐进协同进化**：3DGS变好→条件更好→生成更好→3DGS更好的正反馈设计
3. **训练-推理一致性**：推理时仅用3DGS渲染，无需运行扩散模型，保持实时速度
4. **几何驱动的mask比语义mask更通用**：通过depth warping检测不一致，不依赖特定场景语义

## 局限性
1. 训练时需要多次运行视频扩散模型，总时间约为基础方法的1.6×
2. 极端偏移（>6m）时可能超出扩散模型的泛化范围
3. 依赖LiDAR数据提供稀疏深度锚点，纯视觉场景不适用
4. 新轨迹限于横向平移，未验证更复杂的轨迹模式（如旋转、纵向偏移）

## 相关工作
- **重建驾驶仿真**: StreetGaussian/PVG（3DGS-based）→ 本文（+生成先验）
- **生成驾驶仿真**: GAIA-1/GenAD（视频世界模型）→ 无3D一致性
- **稀疏视图重建+扩散先验**: ZeroNVS/ReconFusion → 本文（驾驶场景定制化inpainting）

## 评分
- 新颖性：⭐⭐⭐⭐⭐ — Inpainting公式化+渐进蒸馏的框架设计原创性强
- 技术深度：⭐⭐⭐⭐ — 各组件设计有深度，但扩散模型本身是微调现有模型
- 实验充分度：⭐⭐⭐⭐ — Waymo上充分验证，缺少nuScenes等其他数据集
- 实用价值：⭐⭐⭐⭐⭐ — 直接提升驾驶仿真质量，推理不增加开销

<!-- RELATED:START -->

## 相关论文

- [GAS: Generative Avatar Synthesis from a Single Image](gas_generative_avatar_synthesis_from_a_single_image.md)
- [MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction](moga_3d_generative_avatar_prior_for_monocular_gaussian_avatar_reconstruction.md)
- [Decompositional Neural Scene Reconstruction with Generative Diffusion Prior](../../CVPR2025/3d_vision/decompositional_neural_scene_reconstruction_with_generative_diffusion_prior.md)
- [RayZer: A Self-supervised Large View Synthesis Model](rayzer_a_self-supervised_large_view_synthesis_model.md)
- [Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis](../../ECCV2024/3d_vision/generative_camera_dolly_extreme_monocular_dynamic_novel_view_synthesis.md)

<!-- RELATED:END -->
