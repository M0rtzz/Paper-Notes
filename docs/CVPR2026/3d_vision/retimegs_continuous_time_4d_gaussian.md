---
title: >-
  [论文解读] RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting
description: >-
  [CVPR 2026][3D视觉][4D Gaussian Splatting] 提出 RetimeGS，通过正则化时间不透明度（双 Sigmoid 短尾分布）和 Catmull-Rom 样条轨迹建模高斯基元的连续运动，结合双向光流监督、三重渲染和动态拉伸策略，解决 4DGS 帧间插值时的时间混叠（ghosting），在 Stage-Capture 数据集上达到 30.08 dB PSNR（超越先前 SOTA 1.29 dB）。
tags:
  - CVPR 2026
  - 3D视觉
  - 4D Gaussian Splatting
  - continuous-time
  - 光流
  - spline trajectory
  - temporal aliasing
---

# RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting

**会议**: CVPR 2026  
**arXiv**: [2603.13783](https://arxiv.org/abs/2603.13783)  
**代码**: [Project Page](https://william-wang2.github.io/RetimeGS/)  
**领域**: 3D视觉 / 动态场景重建  
**关键词**: 4D Gaussian Splatting, continuous-time, optical flow, spline trajectory, temporal aliasing

## 一句话总结
提出 RetimeGS，通过正则化时间不透明度（双 Sigmoid 短尾分布）和 Catmull-Rom 样条轨迹建模高斯基元的连续运动，结合双向光流监督、三重渲染和动态拉伸策略，解决 4DGS 帧间插值时的时间混叠（ghosting），在 Stage-Capture 数据集上达到 30.08 dB PSNR（超越先前 SOTA 1.29 dB）。

## 研究背景与动机

**领域现状**：4D 高斯溅射（4DGS）方法将 3D 高斯基元扩展到时间维度，以实现动态场景的高保真重建。根据时间参数化方式，现有方法可分为两类：基于变形场的方法（在规范空间中通过变形场、控制点或物理约束建模动态）和基于 4D 基元的方法（通过时间不透明度控制基元的出现与消失）。

**现有痛点**：(a) 基于变形场的方法假设动态主要来源于几何运动，难以处理物体可见性或纹理外观随时间变化的场景，且在大运动或帧间重叠有限时对应关系估计不可靠；(b) 基于 4D 基元的方法的时间不透明度仅在整数时间戳上被监督且缺乏正则化，导致过拟合到离散帧（temporal aliasing），在插值中间帧时出现鬼影（ghosting）——半透明的重叠结构来自相邻输入帧；(c) 简单地对时间不透明度做低通滤波（拉宽基元的时间支撑）虽可解决混叠，但需要跨多帧的准确轨迹估计，失败时会产生另一种鬼影。

**核心矛盾**：4D 基元需要动态出现/消失以捕捉可见性变化，但同时必须跨越输入帧之间的完整时间区间；既要有准确的连续轨迹，又不能依赖跨多帧的对应关系估计。

**本文目标**：设计一种 4DGS 表示方法，使基元能在任意时间戳下产生无鬼影、时间连贯的渲染结果，尤其在低帧率、大运动场景下实现高质量连续时间插值。

**切入角度**：将时间混叠视为 4DGS 的根本问题（类比 3D Mip-Splatting 解决空间混叠），通过三个设计原则解决：(i) 基元能动态出现/消失；(ii) 正则化防止在稀疏时间采样下退化；(iii) 在基元持续期间保持准确一致的轨迹。

**核心 idea**：用短尾时间不透明度替代自由优化的时间分布，用 Catmull-Rom 样条替代线性运动假设，用双向光流提供显式轨迹监督。

## 方法详解

### 整体框架
RetimeGS 以多视角视频和对应的双向光流（由 WAFT 估计）为输入，重建 4D 场景。每个 4D 高斯基元的参数扩展为 $(\mu_\tau, \tau_l, \tau_r, \boldsymbol{\mu}, \boldsymbol{v}, \boldsymbol{s}, \boldsymbol{q}(t), \boldsymbol{h}, \sigma)$，其中新增参数控制时间不透明度和空间轨迹。整个 pipeline 包含四个互补的训练策略：双向光流轨迹监督、三重渲染、动态拉伸与周期性重定位、光流感知初始化。使用 VGGT 估计初始点云，MCMC 策略控制基元密度，所有场景训练 20,000 次迭代。

### 关键设计

1. **正则化时间不透明度（Regularized Temporal Opacity）**:

    - 功能：定义每个基元在时间轴上的可见性分布，控制其出现与消失
    - 核心思路：时间不透明度 $\sigma_\tau(t)$ 由两个 Sigmoid 函数的乘积构成，分别以左右时间边界 $\mu_\tau - \tau_l$ 和 $\mu_\tau + \tau_r$ 为中心。初始化时 $\mu_\tau = (t_i + t_{i+1})/2$，$\tau_l = \tau_r = \Delta t / 2$，非可优化参数，确保每组基元居中且覆盖两个相邻输入帧之间的完整区间。在视频边界处将 Sigmoid 替换为常数 1，避免可见性下降
    - 设计动机：短尾分布防止基元退化到单帧，同时相邻基元组在输入帧处混合进出，确保无缝过渡。与拉伸高斯分布不同，短尾分布不需要跨多帧的准确轨迹估计

2. **Catmull-Rom 样条轨迹（Spline-based Spatial Mean）**:

    - 功能：参数化每个基元在时间区间 $[t_i, t_{i+1}]$ 内的连续空间位置 $\boldsymbol{x}(t)$
    - 核心思路：用 4 个控制点定义 Catmull-Rom 样条。内部控制点 $\boldsymbol{p}_1, \boldsymbol{p}_2$ 对应帧 $t_i, t_{i+1}$ 的位置（由伪均值 $\boldsymbol{\mu}$ 和速度 $\boldsymbol{v}_2$ 推导），外部控制点 $\boldsymbol{p}_0, \boldsymbol{p}_3$ 由相邻时间区间的速度 $\boldsymbol{v}_1, \boldsymbol{v}_3$ 确定曲率。公式：$\boldsymbol{p}_{1} = \boldsymbol{\mu} - \frac{1}{2}\Delta t \cdot \boldsymbol{v}_2$，$\boldsymbol{p}_{2} = \boldsymbol{\mu} + \frac{1}{2}\Delta t \cdot \boldsymbol{v}_2$
    - 设计动机：线性速度假设在稀疏时间采样下产生分段线性运动伪影，样条可平滑插值运动。实验表明优化伪均值和速度比直接优化控制点更容易收敛

3. **双向光流轨迹监督（Bidirectional Flow Supervision）**:

    - 功能：利用光流建立帧间粗对应关系，为轨迹参数提供显式监督
    - 核心思路：对每个输入帧 $t_i$，将前后两组基元的 3D 控制点位移投影到 2D 图像平面，光栅化为前向/后向光流图，与真实光流 $\mathbf{F}^{\mathrm{fwd}}, \mathbf{F}^{\mathrm{bwd}}$ 进行像素级监督。光流学习率从 0.5 指数衰减至 $10^{-6}$，训练稳定后逐步切换到 RGB 监督
    - 设计动机：仅靠 RGB 监督在稀疏时间输入下无法学到可靠对应关系，光流提供了兼具几何约束和运动监督的信号

4. **三重渲染监督（Triple Rendering）**:

    - 功能：解决两个相邻基元组各自只重建部分区域、合并后才完整的不均匀覆盖问题
    - 核心思路：对每个内部帧 $t_i$，渲染三张图像：使用所有基元的完整渲染、前一组基元的单独渲染、后一组基元的单独渲染，三张图像均用真值 RGB 监督
    - 设计动机：如果只监督合并后的渲染结果，每组基元可能只学到部分区域的重建，导致中间帧出现欠重建（如一组遗漏左袖纹理，另一组遗漏右袖纹理）

### 损失函数 / 训练策略
- **RGB 损失**：标准重建损失，包含三重渲染的三张图像
- **光流损失**：前向/后向光流的像素级 L2 损失，权重随训练指数衰减
- **正则化**：不透明度正则化（权重 0.01）+ 尺度正则化（权重 0.1）
- **动态拉伸**：每 3,000 次迭代，检测颜色相似且速度接近零的相邻基元，将其时间边界拉伸合并，以减少静态区域的冗余表示。被拉伸基元以概率 $1 - 1/(k+1)$ 被剪枝
- **加权重定位**：MCMC 策略每 100 次迭代执行一次，重定位分数 $s = \sigma / (\tau_l + \tau_r)$ 鼓励将基元调配到动态区域
- **光流感知初始化**：用 VGGT 估计初始点云，通过多视角光流反投影估计 3D 速度初始值

## 实验关键数据

### 主实验
Stage-Capture 数据集（9 个场景，32 个同步 4K 相机，22 FPS → 训练用 11 FPS），前景区域指标：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| Deform-GS | 28.45 | 0.867 | 0.0272 |
| STGS | 25.34 | 0.825 | 0.0357 |
| GaussianFlow | 25.91 | 0.825 | 0.0339 |
| Ex4DGS | 25.95 | 0.811 | 0.0379 |
| 2D Lifting (FILM+STGS) | 28.79 | 0.886 | 0.0267 |
| **RetimeGS** | **30.08** | **0.904** | **0.0225** |

Neural3DV 数据集（Flame Steak + Flame Salmon，30→3 FPS）：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| Deform-GS | 31.79 | 0.952 | 0.081 |
| STGS | 32.52 | 0.959 | 0.079 |
| 2D Lifting | 33.17 | 0.960 | 0.080 |
| **RetimeGS** | **33.22** | 0.959 | **0.074** |

### 消融实验
Stage-Capture 数据集（前景区域）：

| 配置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| w/o 光流初始化 | 29.69 | 0.899 | 0.0227 |
| w/o 光流监督 | 27.24 | 0.861 | 0.0282 |
| w/o 三重渲染 | 27.16 | 0.849 | 0.0319 |
| w/o 动态拉伸 | 28.81 | 0.886 | 0.0247 |
| 线性轨迹替代样条 | 28.50 | 0.884 | 0.0243 |
| **完整 RetimeGS** | **30.08** | **0.904** | **0.0225** |

光流估计器消融（WAFT vs SEA-RAFT）：

| 光流方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|----------|--------|--------|---------|
| WAFT | 30.08 | 0.904 | 0.0225 |
| SEA-RAFT | 29.73 | 0.898 | 0.0253 |

### 关键发现
- 三重渲染（-2.92 dB）和光流监督（-2.84 dB）是最关键的组件，缺少任一个都会导致严重质量下降
- 动态拉伸在 1M 基元预算下将约 9% 的基元识别为跨多帧的静态基元，有效减少冗余，时间总和约 2.26M，等效减少 2.26× 基元数量
- 样条轨迹相比线性轨迹在圆形运动等非线性场景中优势明显（+1.58 dB）
- 训练时间约 3794 秒（STGS 为 1407 秒），峰值显存 3.14 GB（STGS 为 2.47 GB），三重渲染和光流监督增加了训练开销

## 亮点与洞察
- **时间混叠是 4DGS 的根本问题**：过拟合离散帧索引导致插值失败，这一问题的清晰定义和系统解决是本文最大贡献
- **短尾时间不透明度设计精巧**：既允许基元动态出现/消失，又强制覆盖帧间区间，避免了变形方法和 4D 基元方法的各自局限
- **光流提供自然的运动监督**：无需额外标注，光流既是轨迹的监督信号又是几何约束，且与样条参数化配合良好
- **三重渲染思路简洁有效**：一个简单的监督策略解决了基元组不均匀覆盖的问题，-2.92 dB 的消融结果令人印象深刻

## 局限与展望
- 依赖光流估计质量，当帧间运动超过约 50 像素（1K 分辨率）或 FPS 极低（<7.5）时光流不可靠，方法退化
- 训练开销约为 STGS 的 2.7×（3794s vs 1407s），主要来自三重渲染和光流监督
- 相邻基元组在输入帧处的不连续性可能导致轻微闪烁，用统一的 4D 表示解决此问题是未来方向
- 时间不透明度的 $\gamma=0.005$ 超参数较敏感，需要针对不同场景调整
- 当前仅在舞台捕捉（多视角同步相机）场景验证，泛化到野外单目视频的能力未知

## 相关工作与启发
- **vs STGS**：STGS 使用 1D 高斯建模时间不透明度但缺乏正则化，导致过拟合离散帧；RetimeGS 用短尾 Sigmoid 分布 + 非可优化初始化解决了这一问题（+4.74 dB）
- **vs GaussianFlow**：GaussianFlow 引入前向光流监督但缺乏时间不透明度正则化，优化器仍可通过缩短基元时间支撑来满足光流约束；RetimeGS 同时做双向光流监督和时间正则化（+4.17 dB）
- **vs Deform-GS**：基于变形的方法用单组基元表示所有帧，可捕获粗略全局轨迹但在快速运动/可见性变化区域无法建立精细对应（+1.63 dB）
- **vs SplineGS**：SplineGS 用样条驱动控制点，但针对单目 4D 重建，不适用于多视角帧插值任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 对时间混叠问题的清晰定义和系统解决方案设计精巧，短尾不透明度 + 样条轨迹 + 光流监督的组合有原创性
- 实验充分度: ⭐⭐⭐⭐⭐ 主实验 + 完整消融 + 逐场景分析 + 光流估计器消融 + 训练效率分析 + 失败案例讨论，实验设计非常全面
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述详细，动机与设计之间的逻辑链条完整
- 价值: ⭐⭐⭐⭐ 解决了 4DGS 帧插值中的关键问题，对动态场景重建领域有实质性推进

<!-- RELATED:START -->

## 相关论文

- [BulletGen: Improving 4D Reconstruction with Bullet-Time Generation](bulletgen_improving_4d_reconstruction_with_bullet-time_generation.md)
- [4C4D: 4 Camera 4D Gaussian Splatting](4c4d_4_camera_4d_gaussian_splatting.md)
- [Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos](learning_explicit_continuous_motion_representation_for_dynamic_gaussian_splattin.md)
- [EMGauss: Continuous Slice-to-3D Reconstruction via Dynamic Gaussian Modeling in Volume Electron Microscopy](emgauss_continuous_slice-to-3d_reconstruction_via_dynamic_gaussian_modeling_in_v.md)
- [GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes](gp-4dgs_probabilistic_4d_gaussian_splatting_from_monocular_video_via_variational.md)

<!-- RELATED:END -->
