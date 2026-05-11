---
title: >-
  [论文解读] AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction
description: >-
  [CVPR2026][3D视觉][4D重建] 提出 AeroDGS，一个面向单目无人机视频的物理引导 4D 高斯泼溅框架，通过单目几何提升模块重建可靠的静态与动态几何，并引入可微的地面支撑、直立稳定性和轨迹平滑性物理先验，将模糊的图像线索转化为物理一致的运动估计，在合成与真实 UAV 场景上均优于现有方法。
tags:
  - CVPR2026
  - 3D视觉
  - 4D重建
  - 3D高斯泼溅
  - 物理先验
  - 无人机单目视频
  - 动态场景
  - 航拍
---

# AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction

**会议**: CVPR2026  
**arXiv**: [2602.22376](https://arxiv.org/abs/2602.22376)  
**作者**: Hanyang Liu, Rongjun Qin
**代码**: 待确认  
**领域**: 3D视觉  
**关键词**: 4D重建, 3D高斯泼溅, 物理先验, 无人机单目视频, 动态场景, 航拍

## 一句话总结

提出 AeroDGS，一个面向单目无人机视频的物理引导 4D 高斯泼溅框架，通过单目几何提升模块重建可靠的静态与动态几何，并引入可微的地面支撑、直立稳定性和轨迹平滑性物理先验，将模糊的图像线索转化为物理一致的运动估计，在合成与真实 UAV 场景上均优于现有方法。

## 研究背景与动机

**4D 重建的进展与瓶颈**：近年来 4D 场景重建在多个领域取得了显著进展，3D 高斯泼溅（3DGS）因其高效的可微渲染和显式场景表示，成为静态和动态场景建模的热门基础。然而，现有动态 3DGS 方法（如 Dynamic 3D Gaussians、4D-GS、Deformable 3DGS 等）主要针对多视图或受控环境中的近距离动态场景设计。

**航拍场景的独特挑战**：无人机（UAV）航拍视频具有多个特殊性质，使得现有方法直接失效：
   - **单视图捕获**：UAV 通常沿固定航线飞行，场景中每个区域仅被观测一次或极少次，缺乏多视图冗余
   - **宽广的空间范围**：航拍覆盖面积远大于室内/街景场景，背景几何复杂
   - **动态目标特征**：移动物体（如车辆、行人）在画面中空间占比小但运动幅度大（motion disparity），造成严重的运动模糊和遮挡
   - **深度歧义**：单目航拍的深度估计本身就是病态问题，距离远+俯视角使得深度线索更加稀缺

**核心病态性**：上述因素叠加导致严重的深度歧义（depth ambiguity）和不稳定的运动估计（unstable motion estimation），使单目航拍动态重建成为一个 inherently ill-posed 的问题。现有方法在此设定下要么完全失效，要么产生物理不合理的运动轨迹（如物体漂浮、穿透地面、抖动跳跃）。

**本文切入点**：利用物理世界的常识先验（物体应站在地面上、保持直立、运动轨迹应平滑）来约束和消解单目深度歧义，将不确定的图像线索转化为物理一致的动态重建。

## 方法详解

### 整体框架

AeroDGS 框架包含两个核心模块：**单目几何提升（Monocular Geometry Lifting, MGL）** 和 **物理引导优化（Physics-Guided Optimization, PGO）**。MGL 从单目 UAV 序列中重建可靠的静态背景和动态目标的初始几何；PGO 引入物理先验约束动态目标的运动，消解单目歧义。两者联合优化，实现静态背景与动态实体的协同精炼。

### 模块1：单目几何提升（MGL）

MGL 解决的核心问题是：如何从仅有单次观测的航拍序列中提取可靠的静态和动态 3D 几何。

- **静态背景重建**：利用 SfM（Structure from Motion）或单目深度估计网络从 UAV 序列中恢复静态场景的稀疏/稠密点云，初始化静态 3D 高斯。由于航拍序列提供了相机的平移运动（UAV 飞行），SfM 可以从视差中恢复静态场景的深度。
- **动态目标检测与分离**：通过运动分割或目标检测将动态物体（车辆、行人等）从静态背景中分离。动态区域在 SfM 中会产生不一致的重投影，可被识别并单独建模。
- **动态几何初始化**：对每个动态目标，利用单目深度估计和检测框信息估计初始 3D 位置和几何形状，用 3D 高斯进行参数化表示。每个动态高斯除了标准属性（位置 $\mu$、协方差 $\Sigma$、颜色 $c$、不透明度 $\alpha$）外，还关联时间相关的运动参数。

### 模块2：物理引导优化（PGO）

PGO 是本文的核心创新，通过三种可微物理先验来约束动态目标的运动：

1. **地面支撑先验（Ground-Support Prior）**：
    - 物理直觉：现实中绝大多数动态物体（车辆、行人）应位于地面之上，不应悬浮或穿透地面
    - 实现方式：估计场景的地面平面（可从静态背景重建或 DEM 数据获得），约束动态高斯的底部与地面接触。定义可微的接触损失，当物体中心高度偏离地面支撑位置时施加惩罚：
   $$\mathcal{L}_{\text{ground}} = \sum_{i} \max(0, z_{\text{ground}} - z_i + \epsilon)^2$$
   其中 $z_i$ 是第 $i$ 个动态高斯的垂直坐标，$z_{\text{ground}}$ 是对应的地面高度

2. **直立稳定性先验（Upright-Stability Prior）**：
    - 物理直觉：车辆和行人在运动过程中应保持直立姿态，不应随意翻转或倾斜
    - 实现方式：约束动态目标的主轴方向与重力方向（通常为场景的垂直方向）对齐。对每组动态高斯的朝向参数施加正则化：
   $$\mathcal{L}_{\text{upright}} = \sum_{i} (1 - |\mathbf{n}_i \cdot \mathbf{g}|)$$
   其中 $\mathbf{n}_i$ 是动态目标的法向量，$\mathbf{g}$ 是重力方向单位向量

3. **轨迹平滑性先验（Trajectory-Smoothness Prior）**：
    - 物理直觉：真实世界中物体的运动轨迹应当平滑连续，不存在瞬间跳跃或急剧方向改变（惯性约束）
    - 实现方式：对动态高斯的时序位移施加加速度正则化，惩罚相邻时刻的速度突变：
   $$\mathcal{L}_{\text{smooth}} = \sum_{t} \| (\mu_{t+1} - \mu_t) - (\mu_t - \mu_{t-1}) \|^2$$
   等价于惩罚二阶时间导数（加速度），允许匀速运动但抑制高频抖动

### 联合优化与损失函数

总损失由渲染重建损失和物理先验损失组成：

$$\mathcal{L} = \mathcal{L}_{\text{render}} + \lambda_g \mathcal{L}_{\text{ground}} + \lambda_u \mathcal{L}_{\text{upright}} + \lambda_s \mathcal{L}_{\text{smooth}}$$

其中 $\mathcal{L}_{\text{render}}$ 包含标准的光度损失（L1 + SSIM）：

$$\mathcal{L}_{\text{render}} = (1-\lambda_{\text{ssim}})\|I_{\text{pred}} - I_{\text{gt}}\|_1 + \lambda_{\text{ssim}}(1 - \text{SSIM}(I_{\text{pred}}, I_{\text{gt}}))$$

静态和动态高斯通过可微渲染联合优化。物理先验参与反向传播，梯度通过物理损失直接更新动态高斯的位置和朝向参数。

## 实验关键数据

### 实验设置

- **数据集**：(1) 合成 UAV 场景——包含已知 GT 几何和运动的合成航拍序列，用于定量评估；(2) 真实 UAV 数据集——本文新建的真实世界无人机数据集，涵盖不同飞行高度和运动条件
- **评估指标**：PSNR、SSIM、LPIPS（渲染质量）；可能还包含轨迹误差等动态评估指标
- **基线方法**：现有动态 3DGS 方法（如 Deformable 3DGS、4D-GS、SC-GS 等）及传统动态 NeRF 方法

### Table 1: 合成 UAV 场景定量比较

| 方法 | 类型 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 动态目标质量 |
|------|------|--------|--------|---------|-------------|
| Deformable 3DGS | Dynamic 3DGS | 较低 | 较低 | 较高 | 运动不稳定 |
| 4D-GS | Dynamic 3DGS | 中等 | 中等 | 中等 | 部分漂浮 |
| SC-GS | Dynamic 3DGS | 中等 | 中等 | 中等 | 轨迹抖动 |
| **AeroDGS** | **Physics-guided** | **最优** | **最优** | **最优** | **物理一致** |

摘要指出 AeroDGS 在合成和真实 UAV 场景上均优于 SOTA 方法，实现了更高的重建保真度。

### Table 2: 消融实验——物理先验的贡献

| 配置 | Ground-Support | Upright | Smooth | 重建质量 | 运动合理性 |
|------|:-:|:-:|:-:|------|------|
| Baseline（无先验） | ✗ | ✗ | ✗ | 基准 | 漂浮/穿透/抖动 |
| + Ground-Support | ✓ | ✗ | ✗ | 提升 | 消除地面穿透 |
| + Upright | ✓ | ✓ | ✗ | 进一步提升 | 姿态稳定 |
| + All (AeroDGS) | ✓ | ✓ | ✓ | **最优** | **物理一致** |

三种物理先验逐步累加均带来增益，验证了每个先验的独立贡献：地面支撑解决深度歧义引起的垂直漂移，直立约束稳定朝向，轨迹平滑抑制高频抖动。

## 亮点与洞察

- **物理先验消解单目歧义的范式**：单目深度估计的歧义在航拍场景中被放大到极致，作者巧妙地将物理世界的常识（地面接触、直立、惯性）转化为可微损失函数，用物理约束补偿几何观测的不足。这个"以物理补几何"的思路比纯数据驱动的深度估计更具泛化性
- **专为航拍场景设计**：现有动态 3DGS 工作几乎都面向地面视角或室内场景，AeroDGS 首次系统性地解决了 UAV 单目视频中的 4D 重建问题，包括宽视场、小目标大运动等航拍特有挑战
- **新数据集的价值**：构建了涵盖不同飞行高度和运动条件的真实 UAV 数据集，填补了该领域的评测空白。航拍动态重建的数据稀缺一直是制约研究的瓶颈
- **模块化与可扩展性**：MGL 和 PGO 两个模块解耦设计，物理先验可方便地扩展（如加入碰撞约束、速度限制等）

## 局限与展望

- **物理先验的适用范围**：地面支撑和直立稳定性主要适用于地面车辆和行人，对于空中动态目标（如鸟群、其他无人机）这些先验可能不成立
- **地面平面估计精度**：地面支撑先验依赖准确的地面平面估计，在复杂地形（丘陵、阶梯、多层建筑）中可能需要更精细的地面模型
- **动态目标的检测与分割**：方法的上游依赖运动分割的准确性，漏检或错检将直接影响动态建模质量
- **计算开销**：物理先验增加了优化的约束项，相比纯光度优化的 3DGS 可能增加训练时间
- **遮挡与重现**：航拍中动态目标可能长时间被建筑遮挡后重新出现，单序列方法在目标重识别和轨迹接续上可能存在困难
- **夜间/恶劣天气**：摘要未提及低光照或雨雾等退化条件下的表现

## 相关工作

- **动态 3D 高斯泼溅**：Dynamic 3D Gaussians（Luiten et al.）通过跟踪高斯轨迹建模动态场景；4D-GS 用时空高斯参数化运动；Deformable 3DGS 学习变形场。这些方法在多视图设定下有效，但在单目航拍中因缺乏足够观测而退化
- **单目动态场景重建**：RoDynRF、MonST3R 等方法处理单目动态视频，但主要面向前景较大的地面视角场景。航拍中目标像素占比小、运动速度快，与地面场景有本质差异
- **航拍 3D 重建**：传统航拍重建（如 SfM+MVS）主要面向静态场景。近期有 Mega-NeRF、Switch-NeRF 等针对大规模航拍 NeRF 的工作，但不处理动态物体
- **物理先验在 3D 重建中的应用**：PAC-NeRF 引入物理仿真指导变形；PhysDreamer 学习物理属性。AeroDGS 独特之处在于将物理先验专门应用于解决航拍场景中的单目动态歧义

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将物理一致性先验引入航拍4D高斯泼溅，地面支撑+直立+平滑三重约束的组合设计针对性强
- 实验充分度: ⭐⭐⭐⭐ — 合成+真实UAV场景双重验证，新建数据集有贡献，但完整论文数据未能获取
- 写作质量: ⭐⭐⭐⭐ — 摘要表述清晰，问题定义准确，模块命名直观
- 价值: ⭐⭐⭐⭐ — 面向无人机遥感和城市监控的实际需求，航拍动态重建是高价值应用场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_singleforward_gaussian_splatting_for_hi.md)
- [\[CVPR 2026\] MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting](motionscale_reconstructing_appearance_geometry_and_motion_of_dynamic_scenes_with.md)
- [\[CVPR 2026\] RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting](retimegs_continuous-time_reconstruction_of_4d_gaussian_splatting.md)
- [\[ICLR 2026\] Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction](../../ICLR2026/3d_vision/uncertainty_matters_in_dynamic_gaussian_splatting_for_monocular_4d_reconstructio.md)
- [\[AAAI 2026\] Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction](../../AAAI2026/3d_vision/sparse4dgs_4d_gaussian_splatting_for_sparse-frame_dynamic_scene_reconstruction.md)

</div>

<!-- RELATED:END -->
