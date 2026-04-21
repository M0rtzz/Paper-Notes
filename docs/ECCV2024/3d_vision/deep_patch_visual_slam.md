---
title: >-
  [论文解读] Deep Patch Visual SLAM
description: >-
  [ECCV 2024][3D视觉][视觉SLAM] 基于 DPVO 视觉里程计系统，通过高效的邻近回环检测和经典回环检测机制，将其扩展为完整的 SLAM 系统 DPV-SLAM，在单 GPU 上实现实时、高精度、低显存的单目视觉 SLAM。
tags:
  - ECCV 2024
  - 3D视觉
  - 视觉SLAM
  - 回环检测
  - 稀疏光流
  - 深度网络
  - 单目视觉
---

# Deep Patch Visual SLAM

**会议**: ECCV 2024  
**arXiv**: [2408.01654](https://arxiv.org/abs/2408.01654)  
**代码**: https://github.com/princeton-vl/DPVO (有)  
**领域**: 3D视觉  
**关键词**: 视觉SLAM, 回环检测, 稀疏光流, 深度网络, 单目视觉

## 一句话总结

基于 DPVO 视觉里程计系统，通过高效的邻近回环检测和经典回环检测机制，将其扩展为完整的 SLAM 系统 DPV-SLAM，在单 GPU 上实现实时、高精度、低显存的单目视觉 SLAM。

## 研究背景与动机

**领域现状**：基于深度网络的视觉 SLAM（如 DROID-SLAM）在精度上已显著超越传统方法，被广泛用于单目深度估计、视图合成、3D 人体姿态等下游任务的子系统。

**现有痛点**：
   - **显存开销巨大**：DROID-SLAM 等方法需要 24G 显存，因为需要为所有帧存储密集特征图
   - **无法单 GPU 实时**：深度 SLAM 的前端和后端争抢 GPU 资源，CUDA 操作实际上是串行的，导致帧率从 30Hz 骤降至 <1Hz
   - **跨域泛化差**：传统方法在室内效果好但室外失败，深度方法反之

**核心矛盾**：深度 SLAM 系统要实现回环检测需要全局优化，但全局优化需要存储大量深度特征，导致显存线性增长且阻塞前端推理。

**本文目标** 构建一个单 GPU 上高效运行、跨域鲁棒的单目深度 SLAM 系统。

**切入角度**：DPVO 用稀疏光流替代密集对应，大幅降低了单帧成本。作者观察到 patch graph 中边的方向可以任意翻转而不影响优化，从而巧妙控制哪些帧需存储密集特征。

**核心 idea**：通过单向边 patch graph 设计最小化特征存储，将邻近回环和经典回环混入同一优化中，实现单 GPU 实时深度 SLAM。

## 方法详解

### 整体框架

DPV-SLAM 以 DPVO 视觉里程计为基础，引入两种回环检测机制：(1) 基于相机邻近性的回环检测——检测重访位置并通过全局 bundle adjustment 优化；(2) 经典回环检测——基于图像检索和位姿图优化。两者在单进程单 GPU 上运行，前者与里程计共享场景图，后者在 CPU 上并行执行。

### 关键设计

1. **Patch Graph 场景表示**:

    - **功能**：用稀疏的 $p \times p$ patch 替代密集深度图来表示场景。
    - **核心思路**：每个帧 $i$ 包含若干 patch $\mathbf{P}_{ik} = (\mathbf{x}, \mathbf{y}, \mathbf{1}, \mathbf{d})^T$，其中 $\mathbf{d}$ 为逆深度估计。patch 通过有向边连接到其他帧，重投影为 $\mathbf{P}'_{ikj} = \Pi[G_j^{-1} \cdot G_i \cdot \Pi^{-1}(\mathbf{P}_{ik})]$。优化目标为最小化重投影误差：
    $\arg\min_{G,\mathbf{d}} \sum_i \sum_k \sum_j \|\Pi[G_j^{-1} \cdot G_i \cdot \Pi^{-1}(\mathbf{P}_{ik})] - \mathcal{I}_{ikj}\|^2_{\Sigma_{ikj}}$
    - **设计动机**：稀疏 patch 相比密集光流大幅节省存储和计算。

2. **邻近回环检测 (Proximity Loop Closure)**:

    - **功能**：检测相机重访先前位置，插入长程边并进行全局 bundle adjustment。
    - **核心思路**：利用 patch graph 边方向可翻转的特性——每条边的相关性运算 $\mathbf{C}(u,v,\alpha,\beta) = \langle \mathbf{g}(u,v), \mathbf{f}(\mathbf{P}'(u,v) + \Delta_{\alpha\beta}) \rangle$ 只需存储目标帧的密集特征图。因此，创建从旧帧 patch 指向新帧的单向边，只需永久存储所有历史帧的 patch 特征（约 0.6G / 1K 帧），而无需密集特征图。
    - **高效全局优化**：贡献了 CUDA 加速的块稀疏 bundle adjustment 实现，将里程计因子和回环因子混合在同一优化中。回环检测每次仅需 0.1-0.18s，远快于 DROID-SLAM 的 0.5-5s。
    - **设计动机**：避免双 GPU 需求，利用边方向不影响优化结果的独特性质来最小化存储。

3. **经典回环检测 (Classical Loop Closure)**:

    - **功能**：检测和纠正尺度漂移，通过图像检索和 $Sim(3)$ 位姿图优化。
    - **核心思路**：
        - 使用 dBoW2 进行图像检索（ORB 特征），提取和检索在独立 CPU 进程中并行进行
        - 用现成的关键点检测器和匹配器估计 2D 对应，三角化深度后通过 RANSAC+Umeyama 进行 3D 点云对齐，估计 7-DOF 漂移 $\Delta S^{loop}_{jk} \in Sim(3)$
        - 优化位姿图目标：$\arg\min_{S_1,...S_N} \sum_i \|r_i\|^2 + \sum_{(j,k)} \|r_{jk}\|^2$，其中平滑项 $r_i = \log_{Sim(3)}(\Delta S_{(i,i+1)}^{-1} \cdot S_i^{-1} \cdot S_{i+1})$，回环项 $r_{jk} = \log_{Sim(3)}(\Delta S^{loop}_{jk} \cdot S_j^{-1} \cdot S_k)$
    - **设计动机**：邻近检测在存在尺度漂移时不足以检测回环（如户外长序列），需要基于外观的检测补充。

### 损失函数 / 训练策略

- 沿用 DPVO 的训练策略，仅在合成数据 TartanAir 上训练
- 使用可微分 bundle adjustment 层，通过监督相机位姿来端到端学习异常值剔除
- 关键点随机选取（无需检测器），效果出奇地好

## 实验关键数据

### 主实验（EuRoC-MAV 数据集 - ATE 指标）

| 方法 | 平均 ATE↓ | 帧率 (FPS) | 显存 (VRAM) |
|------|----------|-----------|-------------|
| DPV-SLAM | **0.024** | **50** | **5.0G** |
| DROID-SLAM | 0.022 | 20 | 20G |
| DPVO (仅里程计) | 0.105 | 60 | 4.0G |
| GO-SLAM | 0.035 | 6.4 | 7.2G |

### KITTI 数据集（ATE 指标，单位：米）

| 方法 | 平均 ATE↓ | FPS |
|------|----------|-----|
| DPV-SLAM++ | **25.76** | 39 |
| DPVO | 53.61 | 48 |
| DROID-SLAM | - (多个序列失败) | 17 |
| LDSO | 22.42 | 49 |

### TUM-RGBD 数据集

| 方法 | 平均 ATE↓ | 失败序列数 |
|------|----------|-----------|
| DPV-SLAM++ | 0.054 | 0 |
| DROID-SLAM | 0.038 | 0 |
| ORB-SLAM3 | - | 5/9 失败 |
| DeFlowSLAM | 0.114 | 0 |

### 关键发现

- DPV-SLAM 在 EuRoC 上达到与 DROID-SLAM 相当的精度（0.024 vs 0.022），速度快 2.5 倍，显存仅 1/4
- 相比基础 DPVO 系统，误差降低 4 倍（0.105→0.024），速度和显存开销仅略微增加
- **跨域鲁棒性突出**：0 次灾难性失败，室内外均表现良好，这是其他方法难以做到的
- DROID-SLAM 室内优秀但 KITTI 户外多次失败；传统方法反之；DPV-SLAM++ 两者兼顾
- 邻近回环每次仅需 0.1-0.18s，而 DROID-SLAM 后端需要 0.5-5s

## 亮点与洞察

- **边方向可翻转的洞察**：patch graph 中边的方向可以任意调整而基本不影响优化结果（因为每个因子同时约束源和目标相机位姿），这是 DPVO patch 表示的独特性质，巧妙地解决了显存问题
- **单进程单 GPU 设计哲学**：避免了多 GPU/多进程的复杂性，同时保持实时性能，是工程与算法的优雅结合
- **随机关键点选择有效**：DPV-SLAM 继承了 DPVO 随机选取关键点的策略，不使用特征检测器，最高置信度的点反而常在低纹理区域

## 局限与展望

- 仅支持单目视频，未利用立体 / 惯性信息
- 邻近回环在存在严重尺度漂移时效果有限，需要经典回环补充
- 合成数据训练的泛化性虽好，但在特定领域（如自动驾驶）可能不如专门调优的方法
- 定位为 systems paper，核心贡献偏工程优化，新颖性相对有限

## 相关工作与启发

- **vs DROID-SLAM**: 同源系统，DPV-SLAM 用稀疏 patch 替代密集光流，显存和速度改善显著，但在最佳条件下精度略逊
- **vs DPVO**: DPV-SLAM 是 DPVO 的回环扩展，误差降低 4 倍，证明了回环检测对里程计系统的关键价值
- **vs ORB-SLAM3**: 经典方法在室内常失败（快速相机运动），但户外 KITTI 表现好；DPV-SLAM++ 通过结合深度+经典方法实现全场景鲁棒
- **vs GO-SLAM / DeFlowSLAM**: 基于 DROID 的其他扩展，但显存更高（7-8G）且帧率更低

## 评分

- 新颖性: ⭐⭐⭐ 主要是工程优化和系统集成，边方向翻转的洞察有新意但核心 idea 源自 DPVO
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 4 个数据集（室内/室外）、多种速度/内存对比、与 10+ 方法比较
- 写作质量: ⭐⭐⭐⭐ 系统论文逻辑清晰，问题定义精确，开源完整
- 价值: ⭐⭐⭐⭐ 对需要实时相机位姿的下游应用（视图合成、3D重建）具有很高的实用价值

<!-- RELATED:START -->

## 相关论文

- [I²-SLAM: Inverting Imaging Process for Robust Photorealistic Dense SLAM](i2-slam_inverting_imaging_process_for_robust_photorealistic_dense_slam.md)
- [SGS-SLAM: Semantic Gaussian Splatting for Neural Dense SLAM](sgs-slam_semantic_gaussian_splatting_for_neural_dense_slam.md)
- [CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-Aware 3D Gaussian Field](cg-slam_efficient_dense_rgb-d_slam_in_a_consistent_uncertainty-aware_3d_gaussian.md)
- [VGGT-SLAM++: Visual SLAM with DEM-Based Covisibility and Local Bundle Adjustment](../../CVPR2026/3d_vision/vggt-slam.md)
- [Benchmarking Egocentric Visual-Inertial SLAM at City Scale](../../ICCV2025/3d_vision/benchmarking_egocentric_visualinertial_slam_at_city_scale.md)

<!-- RELATED:END -->
