---
title: >-
  [论文解读] IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera
description: >-
  [CVPR 2025][3D视觉][事件相机] 本文提出 IncEventGS，首个仅用单目事件相机、无需已知位姿即可增量重建 3D 高斯溅射场景的方法，采用跟踪-建图 SLAM 范式联合优化相机运动和场景表示，在新视角合成和位姿估计上均超越现有方法。 领域现状：3D 高斯溅射（3DGS）和 NeRF 在新视角合成方面取得巨…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "事件相机"
  - "3D高斯溅射"
  - "增量重建"
  - "视觉里程计"
  - "SLAM"
---

# IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera

**会议**: CVPR 2025  
**arXiv**: [2410.08107](https://arxiv.org/abs/2410.08107)  
**代码**: [https://github.com/WU-CVGL/IncEventGS](https://github.com/WU-CVGL/IncEventGS)  
**领域**: 3D视觉  
**关键词**: 事件相机, 3D高斯溅射, 增量重建, 视觉里程计, SLAM

## 一句话总结

本文提出 IncEventGS，首个仅用单目事件相机、无需已知位姿即可增量重建 3D 高斯溅射场景的方法，采用跟踪-建图 SLAM 范式联合优化相机运动和场景表示，在新视角合成和位姿估计上均超越现有方法。

## 研究背景与动机

**领域现状**：3D 高斯溅射（3DGS）和 NeRF 在新视角合成方面取得巨大进展，通常使用 RGB 或 RGB-D 相机+COLMAP 提供初始位姿和点云。事件相机（event camera）作为仿生传感器，具有高时间分辨率、高动态范围、低延迟和低功耗等优势，在快速运动和低光照条件下表现远优于帧相机。

**现有痛点**：（1）现有事件相机 NeRF 方法（E-NeRF、EventNeRF、Robust e-NeRF）均需要 ground-truth 位姿，严重限制了实际应用；位姿通常由配对的 RGB 图像通过 COLMAP 计算或由动作捕捉系统提供。（2）事件数据异步稀疏，无法直接用于传统 3D 重建算法。（3）现有事件 SLAM 方法通常只恢复稀疏深度图（2.5D）而非稠密 3D 场景。（4）二阶段方法（E2VID 转图像 → COLMAP 估位姿 → 3DGS）在事件数据转换和位姿估计中都会引入较大误差。

**核心矛盾**：事件相机数据的异步性与 3DGS 要求的同步亮度图和精确位姿之间存在鸿沟。同时，纯事件流缺少绝对亮度信息，使得初始化和位姿估计更加困难。

**本文目标** 如何仅从单目事件相机的事件流出发，无需任何先验位姿，增量式地同时恢复 3D 高斯场景表示和相机运动轨迹？

**切入角度**：借鉴传统 SLAM 的跟踪-建图范式，但将底层场景表示换成 3DGS。关键观察是事件流可以通过亮度变化模型与 3DGS 渲染的亮度图建立可微分联系，从而支持同时优化位姿和场景。针对缺乏初始位姿和点云的问题，设计了基于单目深度估计的引导初始化。

**核心 idea**：将事件相机 SLAM 的跟踪-建图范式与 3D 高斯溅射结合，通过事件亮度变化模型建立可微分损失进行联合 bundle adjustment。

## 方法详解

### 整体框架

IncEventGS 将事件流按固定时间窗口分成 chunk，每个 chunk 关联一个连续时间轨迹参数化（在 $\mathfrak{se}(3)$ 空间线性插值）。系统交替执行两阶段：（1）**跟踪**：固定 3DGS，仅优化最新 chunk 的相机轨迹参数；（2）**建图**：使用滑动窗口 bundle adjustment，联合优化最近 $n_w=20$ 个 chunk 的 3DGS 参数和轨迹参数。随着相机移动，在新探索区域增量添加高斯点。

### 关键设计

1. **事件-3DGS 可微分联系**:

    - 功能：建立事件流与 3DGS 渲染之间的可微分损失
    - 核心思路：在每个 chunk 内随机采样两个接近的时间戳 $t_k$ 和 $t_{k+\Delta t}$，从轨迹参数化插值得到对应位姿 $T_k$ 和 $T_{k+\Delta t}$，从 3DGS 渲染两张亮度图 $\hat{I}_k$ 和 $\hat{I}_{k+\Delta t}$。合成亮度变化为 $\hat{E}(\mathbf{x}) = \log(\hat{I}_{k+\Delta t}) - \log(\hat{I}_k)$，实测亮度变化 $E(\mathbf{x})$ 从对应时间段的事件流累积。最小化 $\|E - \hat{E}\|_2$ 作为核心损失。
    - 设计动机：事件相机记录的是亮度变化而非绝对亮度，对数差分模型天然对应事件触发机制。两个时间戳的亮度图渲染使 3DGS 参数和位姿同时可微，支持联合优化。

2. **增量跟踪与建图**:

    - 功能：实现无需先验位姿的增量式 3D 重建
    - 核心思路：**跟踪阶段**仅优化最新 chunk 的 $T_{start}$ 和 $T_{end}$，利用已重建的 3DGS 场景作为参考，最小化事件损失。**建图阶段**使用滑动窗口（$n_w=20$）进行 bundle adjustment，联合优化窗口内所有 chunk 的轨迹和 3DGS 参数。同时在新探索区域添加高斯点：用已有 3DGS 渲染深度图，反投影到 3D 空间作为新点的位置，用可见性掩码 $V < \lambda_V$ 控制仅在未覆盖区域添加。
    - 设计动机：tracking-mapping 范式久经考验（如 ORB-SLAM），但需要适配 3DGS 和事件数据。滑动窗口 BA 平衡了全局一致性和计算效率。增量添加点+透明高斯剪枝保证场景逐步扩展。

3. **深度引导初始化**:

    - 功能：解决纯事件流下 3DGS 初始化困难的问题
    - 核心思路：用前 $m=3$ 个 chunk 训练初始 3DGS 和位姿（位姿初始化为近单位矩阵，点云在包围盒内随机采样）。训练一定迭代后，用单目深度估计网络从渲染的亮度图预测稠密深度图，将像素深度反投影得到新的高斯中心位置重初始化，然后重新训练。
    - 设计动机：消融实验表明深度初始化使 ATE 从 1.534cm 骤降到 0.046cm——短基线事件数据的初始 3D 结构质量极差，深度估计网络提供了从 2D 到 3D 的关键先验。没有好的初始结构，后续增量扩展会偏离。

### 损失函数 / 训练策略

跟踪阶段仅用事件 L2 损失（Eq.10）。建图阶段用事件 L2 损失 + SSIM 结构相似性损失的组合（$\mathcal{L} = (1-\lambda)\mathcal{L}_{event} + \lambda\mathcal{L}_{ssim}$，$\lambda=0.05$）。每个 chunk 时间窗口 50ms。初始化 4500 步，跟踪 200 步，建图 1500 步。事件流按事件数等分为 $n_{seg}=100$ 段进行时间戳采样，采样窗口大小 $n_{low}=400k$, $n_{up}=500k$（合成数据集）。

## 实验关键数据

### 主实验

| 场景 (Replica) | 指标 | IncEventGS (无位姿) | Robust e-NeRF (GT位姿) | E2VID+COLMAP+3DGS |
|----------------|------|---------------------|------------------------|-------------------|
| room0 | PSNR↑ | **24.31** | 17.26 | 14.45 |
| room2 | PSNR↑ | **23.75** | 16.43 | 15.74 |
| office0 | PSNR↑ | **25.64** | 18.93 | 18.91 |
| office2 | PSNR↑ | **21.74** | 16.81 | 14.03 |
| office3 | PSNR↑ | **21.18** | 19.22 | 13.25 |

即使不使用 GT 位姿，IncEventGS 仍大幅超越使用 GT 位姿的 NeRF 方法（平均 +5-7 dB PSNR）。

| 数据集 | 方法 | ATE (cm)↓ |
|--------|------|-----------|
| Replica room0 | DEVO | 0.289 |
| Replica room0 | **IncEventGS** | **0.046** |
| TUM-VIE desk | DEVO | 0.732 |
| TUM-VIE desk | **IncEventGS** | **0.231** |
| TUM-VIE 6dof | DEVO | 2.93 |
| TUM-VIE 6dof | **IncEventGS** | **0.251** |

位姿估计精度全面超越 SOTA 事件视觉里程计 DEVO。

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | ATE (cm)↓ | 说明 |
|------|-------|-------|--------|-----------|------|
| Full model | 21.74 | 0.82 | 0.23 | 0.046 | 完整模型 |
| w/o 深度初始化 | 17.80 | 0.76 | 0.26 | 1.534 | ATE 暴增 33 倍 |

| 事件窗口 ($n_{low}$-$n_{up}$) | PSNR↑ | ATE↓ | 说明 |
|------|-------|------|------|
| 1k-10k | 16.07 | 0.167 | 窗口太小缺信息 |
| 400k-500k | **21.74** | **0.046** | 最优配置 |
| 600k-700k | 18.06 | 0.214 | 窗口太大内存增且运动过大 |

### 关键发现

- 深度初始化是系统成功的决定性因素——去掉后 PSNR 下降 4 dB、ATE 暴增 33 倍，说明短基线事件数据的 3D 重建极度依赖好的初始化
- 事件窗口大小需要精心选择（400k-500k 最优），太小缺乏信息、太大运动过大匹配困难
- 即使不用 GT 位姿，3DGS 表示+事件 BA 的组合仍远优于用 GT 位姿的 NeRF 方法，体现了 3DGS 表示的优越性
- 在 TUM-VIE 真实数据集上，IncEventGS 的位姿估计甚至优于立体事件相机方法 ESVO2，说明单目+3DGS BA 的强大

## 亮点与洞察

- **首个纯事件无位姿 3DGS 重建**：填补了重要的研究空白。之前的事件 NeRF 方法都依赖 GT 位姿，实际应用中获取位姿本身就需要帧相机
- **3DGS 表示的优势在事件重建中被放大**：对比 NeRF 方法提升 5-7 dB，说明 3DGS 的显式表示+可微分渲染在弱监督信号（事件流仅提供亮度变化）下更容易优化
- **深度估计网络作为初始化先验**的使用很巧妙：不需要深度网络持续参与，仅在 bootstrap 阶段用一次就能显著改善后续所有优化

## 局限与展望

- 合成数据上表现远好于真实数据，真实事件相机的噪声和对比度阈值估计误差影响较大
- 滑动窗口 BA 仅考虑局部一致性，长轨迹可能存在累积漂移（未报告回环检测）
- 事件流缺乏绝对亮度信息，NVS 指标使用了线性颜色校正，说明恢复的绝对亮度可能有偏差
- 初始化依赖单目深度估计网络的质量，在该网络泛化性差的场景中可能失效
- 未与同期并行工作 EvGGS、Event3DGS 在完全相同设置下对比（这些方法使用 GT 位姿）

## 相关工作与启发

- **vs E-NeRF / EventNeRF / Robust e-NeRF**: 这些方法都用 NeRF 表示且需要 GT 位姿，IncEventGS 使用 3DGS 且不需要位姿，NVS 指标全面碾压
- **vs DEVO**: DEVO 是 SOTA 事件视觉里程计但仅估计位姿不重建场景，IncEventGS 同时完成位姿估计+稠密重建，且位姿精度更高（room0: 0.046 vs 0.289 cm）
- **vs MonoGS / GS-SLAM**: 这些是帧相机 3DGS SLAM 方法，IncEventGS 将类似思路迁移到事件相机域，处理了事件数据特有的异步性和缺乏绝对亮度的挑战

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个无位姿事件相机 3DGS 重建，问题设定有重要实际意义
- 实验充分度: ⭐⭐⭐⭐ 合成+真实数据集评估、NVS+位姿双指标、消融研究充分，但缺少与同期工作的直接对比
- 写作质量: ⭐⭐⭐⭐ Pipeline 图清晰，方法描述系统完整
- 价值: ⭐⭐⭐⭐⭐ 对事件相机 3D 重建领域有重要推进，去除位姿依赖大幅提升实际应用可能性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction](../../CVPR2026/3d_vision/e2egs_event-to-edge_gaussian_splatting_for_pose-free_3d_reconstruction.md)
- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[CVPR 2025\] EventFly: Event Camera Perception from Ground to the Sky](eventfly_event_camera_perception_from_ground_to_the_sky.md)
- [\[CVPR 2025\] SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)
- [\[CVPR 2025\] PGC: Physics-Based Gaussian Cloth from a Single Pose](pgc_physics-based_gaussian_cloth_from_a_single_pose.md)

</div>

<!-- RELATED:END -->
