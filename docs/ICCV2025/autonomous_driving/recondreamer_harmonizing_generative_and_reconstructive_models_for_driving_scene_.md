---
title: >-
  [论文解读] ReconDreamer++: Harmonizing Generative and Reconstructive Models for Driving Scene Representation
description: >-
  [ICCV 2025][自动驾驶][驾驶场景重建] ReconDreamer++ 在 ReconDreamer 基础上，通过引入新轨迹可变形网络（NTDNet）弥合生成数据与真实观测的域差距，并独立建模地面以保留几何先验，在 Waymo 上实现了原始轨迹性能持平 Street Gaussians、新轨迹 NTA-IoU 提升 6.1%、FID 改善 23.0% 的显著效果。
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "驾驶场景重建"
  - "3D高斯溅射"
  - "新视角合成"
  - "域差距弥合"
  - "地面建模"
---

# ReconDreamer++: Harmonizing Generative and Reconstructive Models for Driving Scene Representation

**会议**: ICCV 2025  
**arXiv**: [2503.18438](https://arxiv.org/abs/2503.18438)  
**代码**: [https://recondreamer-plus.github.io](https://recondreamer-plus.github.io) (项目页面)  
**领域**: 自动驾驶  
**关键词**: 驾驶场景重建, 3D高斯溅射, 新视角合成, 域差距弥合, 地面建模

## 一句话总结

ReconDreamer++ 在 ReconDreamer 基础上，通过引入新轨迹可变形网络（NTDNet）弥合生成数据与真实观测的域差距，并独立建模地面以保留几何先验，在 Waymo 上实现了原始轨迹性能持平 Street Gaussians、新轨迹 NTA-IoU 提升 6.1%、FID 改善 23.0% 的显著效果。

## 研究背景与动机

**领域现状**：结合重建模型（3DGS）与生成模型（视频扩散）已成为自动驾驶闭环仿真的主流范式。ReconDreamer 利用世界模型 DriveRestorer 在新轨迹上恢复视频质量并混合训练 4DGS 模型，在大幅横向变道（3-6米）场景取得了显著成功。

**现有痛点**：(1) **域差距被忽视**：现有方法将生成数据与真实传感器观测混合训练同一组高斯参数，忽略了两类数据的固有差异。这导致原始轨迹上性能明显下降（如 ReconDreamer 在原始轨迹的 PSNR/SSIM/LPIPS 均显著低于 Street Gaussians）；(2) **地面建模质量差**：自动驾驶中相机光轴通常与地面平行，从新视角观察时地面重建常出现条纹/模糊伪影。用缺乏3D一致性的渲染结果作为 DriveRestorer 的条件会放大这些误差。

**核心矛盾**：利用生成模型增强新轨迹性能的同时，引入的合成数据却损害了原始轨迹性能，这使得方法在实际仿真中难以同时满足两种场景的质量要求。地面是结构化元素中最敏感的区域，其质量直接影响车道线等关键信息的可靠性。

**本文目标** (1) 如何弥合生成数据与真实观测之间的域差距？(2) 如何提升地面等结构化元素的建模质量？(3) 如何在提升新轨迹性能的同时不损害原始轨迹质量？

**切入角度**：将生成数据视为与真实数据来自不同域的"噪声"输入，引入可学习空间变形来对齐两者；将地面从场景中分离并固定其几何位置，只优化外观参数。

**核心 idea**：通过 NTDNet 学习新旧轨迹间高斯参数的空间变形来弥合域差距，同时独立建模地面并保留 LiDAR 几何先验，在不损害原轨迹质量的前提下显著提升新轨迹渲染效果。

## 方法详解

### 整体框架

驾驶场景分解为三个组件：**地面**、**非地面静态背景**和**动态物体**。对原始轨迹相机位姿直接渲染（跳过 NTDNet）；对新轨迹相机位姿，高斯参数经 NTDNet 修正后再渲染。然后使用 ReconDreamer 的 DriveRestorer 对新轨迹渲染结果进行视频恢复，恢复结果与原始数据共同训练重建模型。

### 关键设计

1. **场景三分解建模**:

    - 功能：将场景独立建模为地面、非地面背景和动态物体三部分
    - 核心思路：
        - **地面模型 $\mathcal{G}^g$**：使用道路点云分割初始化高斯位置 $\boldsymbol{x}$。训练时固定位置 $\boldsymbol{x}$ 和高斯数量，只优化不透明度 $\boldsymbol{\gamma}$、协方差 $\boldsymbol{\Sigma}$（分解为旋转 $\boldsymbol{R}$ 和缩放 $\boldsymbol{S}$，$\boldsymbol{\Sigma} = \boldsymbol{RSS}^T\boldsymbol{R}^T$）和颜色 $\boldsymbol{c}$
        - **非地面背景 $\mathcal{G}^{bg}$**：非地面点云初始化 + 随机额外点，所有参数完全优化
        - **动态物体 $\{\mathcal{G}_i^o\}$**：在局部物体坐标系定义，渲染时通过旋转/平移变换到世界坐标系 $\boldsymbol{x}_w = R_t \boldsymbol{x}_o + T_t$
    - 设计动机：自动驾驶相机光轴近乎平行于地面，导致地面重建不稳定。固定地面几何位置减少了搜索空间，多帧融合点云提供了充足先验，从而增强泛化能力

2. **新轨迹可变形网络 (NTDNet)**:

    - 功能：学习从原始轨迹到新轨迹之间高斯参数的空间变形，系统性弥合域差距
    - 核心思路：包含位姿特征模块 $\mathcal{F}_\phi$ 和时间场模块 $\mathcal{F}_\theta$。首先计算归一化 delta pose：$\Delta p_t = \frac{p_t^{novel} - p_t^{ori}}{L}$。然后生成高斯参数偏移：$\Delta g = \mathcal{F}_{out}(\mathcal{F}_\phi(\Delta p_t) + \mathcal{F}_\theta(\text{PE}(g), \text{PE}(t)))$，最终 $g' = g + \Delta g$
    - 设计动机：直接用同一组高斯参数渲染新轨迹会导致域差距问题。NTDNet 根据轨迹偏移量和时间步自适应调整高斯参数，关键是**只在新轨迹时激活**，原始轨迹直接用原始参数，从而避免损害原轨迹质量。两个模块都是 8 层 MLP，隐藏维度 256，非常轻量

3. **新视角深度监督**:

    - 功能：为新轨迹提供额外的几何约束
    - 核心思路：融合所有帧的静态点云 $Pts_{fuse} = \bigcup_{t=1}^F \{p \in Pts_t | \text{IsStatic}(p)\}$，然后投影到新视角相机得到稀疏深度图，用动态物体 mask 过滤后作为静态深度监督：$D_{novel} = \text{Proj}(Pts_{fuse}, C_{novel}) \odot (1 - M_{novel})$
    - 设计动机：单帧点云重投影在域差距下可能导致性能下降，多帧融合的静态点云更稳定可靠

### 损失函数 / 训练策略

$\mathcal{L} = \lambda_1 \mathcal{L}_{RGB} + \lambda_2 \mathcal{L}_{SSIM} + \lambda_3 \mathcal{L}_{Depth}$

沿用 ReconDreamer 的两阶段训练：(1) 部分训练重建模型 → 渲染低质量视频 → 与真实视频配对训练 DriveRestorer；(2) DriveRestorer 修复新轨迹渲染结果 → 修复结果+原始数据混合训练最终模型。总训练 50,000 步。NTDNet 仅 8 层 MLP + 256 隐藏维度。

## 实验关键数据

### 主实验 - Waymo 数据集

| 方法 | 原始轨迹 PSNR↑/SSIM↑/LPIPS↓ | 3m变道 NTA-IoU↑ | 3m NTL-IoU↑ | 3m FID↓ |
|------|---------------------------|----------------|-------------|---------|
| Street Gaussians | 36.50/0.957/0.115 | 0.498 | 53.19 | 130.75 |
| DriveDreamer4D | 34.37/0.945/0.132 | 0.457 | 53.30 | 113.45 |
| ReconDreamer | 34.31/0.943/0.152 | 0.539 | 54.58 | 93.56 |
| **ReconDreamer++** | **36.29/0.957/0.108** | **0.572** | **57.06** | **72.02** |

原始轨迹：LPIPS 比 Street Gaussians 改善 6.4%；新轨迹(3m)：NTA-IoU +6.1%，NTL-IoU +4.5%，FID -23.0%。

### 消融实验 - Waymo

| Depth Loss | Ground Model | NTDNet | 原轨迹 PSNR/LPIPS | 3m NTA-IoU | 3m NTL-IoU | 3m FID |
|------------|-------------|--------|-------------------|------------|------------|--------|
| ✗ | ✗ | ✗ | 34.31/0.152 | 0.539 | 54.58 | 93.56 |
| ✓ | ✗ | ✗ | 35.13/0.121 | 0.561 | 55.91 | 77.51 |
| ✓ | ✓ | ✗ | 34.93/0.117 | 0.566 | 56.89 | 75.22 |
| ✓ | ✓ | ✓ | **36.29/0.108** | **0.572** | **57.06** | **72.02** |

三个组件逐步累加均有贡献，NTDNet 对原始轨迹提升最大（PSNR +1.36），Ground Model 对 NTL-IoU 提升最大（+1.79）。

### 跨数据集验证

| 数据集 | 方法 | 关键指标 | 改善 |
|--------|------|----------|------|
| nuScenes | ReconDreamer++ vs ReconDreamer | 3m NTA-IoU: 0.365 vs 0.325 | +12.3% |
| PandaSet | ReconDreamer++ vs ReconDreamer | 3m FID: 71.7 vs 74.9 | -4.3% |
| EUVS | ReconDreamer++ vs ReconDreamer | LPIPS: 0.306 vs 0.329 | -7.0% |

### 关键发现

- **NTDNet 是关键**：仅在新轨迹时激活的设计使其在改善新轨迹性能的同时大幅提升原始轨迹质量（PSNR 从 34.93→36.29），完美解决了生成模型引入的域差距问题
- **Ground Model 对车道线至关重要**：独立建模地面（固定位置、只优化外观）使 6m 大变道的 NTL-IoU 从 53.55 提升到 55.34（+3.3%）
- **深度监督是基础**：高质量的几何约束为后续组件奠定基础，单独引入即可将 FID 从 93.56 降到 77.51
- **四个数据集一致有效**：Waymo/nuScenes/PandaSet/EUVS 上均优于前作，证明框架的通用性

## 亮点与洞察

- **"新轨迹才变形"的设计**是最巧妙的工程决策：NTDNet 只在新轨迹激活，原始轨迹直接渲染。这避免了变形网络引入的额外噪声影响已有质量，同时让网络专注于学习轨迹差异带来的空间变形。这一思路可推广到任何domain gap问题——只在跨域输入上施加适配，源域保持不变
- **固定地面几何、只学外观**：利用多帧融合 LiDAR 点云提供的强几何先验，将地面的优化空间从位置+外观降低到仅外观。对于相机光轴近平行于地面的自动驾驶场景，这大大降低了优化难度
- **多帧融合静态点云提供深度监督**：比单帧点云更稳定、覆盖更完整，避免了动态物体的干扰

## 局限与展望

- NTDNet 使用简单的 MLP 建模变形，未考虑空间局部性——远处和近处的高斯应有不同的变形模式
- 仍依赖 DriveRestorer 的两阶段训练流程，端到端训练可能更高效
- Ground Model 依赖道路点云分割的准确性，分割错误会传播到几何先验
- 动态物体的建模沿用 Street Gaussians，未针对新轨迹做特殊处理
- 对更极端的变道场景（>6m）未验证，实际应用中可能需要更大范围的泛化

## 相关工作与启发

- **vs ReconDreamer**: 直接前作，通过 DriveRestorer 增强新轨迹但忽略域差距导致原轨迹性能下降。ReconDreamer++ 的 NTDNet 完美解决了这一trade-off
- **vs Street Gaussians**: 纯重建方法在原始轨迹表现最好但无法泛化到新轨迹。ReconDreamer++ 在原轨迹达到 Street Gaussians 水平的同时在新轨迹远超之
- **vs DriveDreamer4D**: 世界模型直接生成新轨迹视频混合训练 4DGS，但同样受域差距影响导致原轨迹退化（PSNR 34.37 vs 36.50）
- **vs FreeVS**: 依赖 LiDAR 点云的自由视角合成，上部无 LiDAR 覆盖区域无法处理

## 评分

- 新颖性: ⭐⭐⭐⭐ NTDNet 和地面独立建模的设计巧妙但改进幅度属增量
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集全面验证+详尽消融+定性定量结合
- 写作质量: ⭐⭐⭐⭐ 问题动机阐述清晰，消融逻辑清楚
- 价值: ⭐⭐⭐⭐ 有效解决了生成-重建混合训练的域差距问题，对闭环仿真有实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration](../../CVPR2025/autonomous_driving/recondreamer_crafting_world_models_for_driving_scene_reconstruction_via_online_r.md)
- [\[ICCV 2025\] DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation](dist-4d_disentangled_spatiotemporal_diffusion_with_metric_depth_for_4d_driving_s.md)
- [\[ICCV 2025\] Leveraging 2D Priors and SDF Guidance for Dynamic Urban Scene Rendering](leveraging_2d_priors_and_sdf_guidance_for_urban_scene_rendering.md)
- [\[ICCV 2025\] GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting](gs-livm_real-time_photo-realistic_lidar-inertial-visual_mapping_with_gaussian_sp.md)
- [\[ICCV 2025\] ETA: Efficiency through Thinking Ahead, A Dual Approach to Self-Driving with Large Models](eta_efficiency_through_thinking_ahead_a_dual_approach_to_self-driving_with_large.md)

</div>

<!-- RELATED:END -->
